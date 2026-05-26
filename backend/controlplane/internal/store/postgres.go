package store

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store/sqlc"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/jackc/pgx/v5/pgxpool"
)

type PostgresStore struct {
	pool    *pgxpool.Pool
	queries *sqlc.Queries
}

func NewPostgresStore(pool *pgxpool.Pool) *PostgresStore {
	return &PostgresStore{
		pool:    pool,
		queries: sqlc.New(pool),
	}
}

func (s *PostgresStore) CreateThread(ctx context.Context, input domain.CreateThreadInput) (domain.ThreadRecord, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.ThreadRecord{}, err
	}
	defer tx.Rollback(ctx)

	q := s.queries.WithTx(tx)
	now := domain.Now()
	row, err := q.CreateThread(ctx, sqlc.CreateThreadParams{
		ThreadID:     domain.NewID("thread"),
		UserID:       input.UserID,
		Title:        nullableText(input.Title),
		Status:       string(domain.ThreadStatusActive),
		CreatedAt:    timestamptz(now),
		UpdatedAt:    timestamptz(now),
		LatestRunID:  pgtype.Text{},
		CheckpointID: pgtype.Text{},
		Summary:      pgtype.Text{},
		Metadata:     jsonBytes(input.Metadata),
	})
	if err != nil {
		return domain.ThreadRecord{}, err
	}
	for _, msg := range input.InitialMessages {
		if _, err := q.InsertThreadMessage(ctx, sqlc.InsertThreadMessageParams{
			MessageID: domain.NewID("msg"),
			ThreadID:  row.ThreadID,
			Role:      msg.Role,
			Content:   msg.Content,
			CreatedAt: timestamptz(now),
			Metadata:  jsonBytes(msg.Metadata),
			RunID:     pgtype.Text{},
		}); err != nil {
			return domain.ThreadRecord{}, err
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.ThreadRecord{}, err
	}
	return threadFromRow(row), nil
}

func (s *PostgresStore) GetThread(ctx context.Context, threadID string) (domain.ThreadRecord, error) {
	row, err := s.queries.GetThread(ctx, threadID)
	if err != nil {
		return domain.ThreadRecord{}, mapPgError(err)
	}
	return threadFromRow(row), nil
}

func (s *PostgresStore) ListThreads(ctx context.Context, limit int) ([]domain.ThreadRecord, error) {
	rows, err := s.queries.ListThreads(ctx, limit32(limit, 100))
	if err != nil {
		return nil, err
	}
	threads := make([]domain.ThreadRecord, 0, len(rows))
	for _, row := range rows {
		threads = append(threads, threadFromRow(row))
	}
	return threads, nil
}

func (s *PostgresStore) ListThreadMessages(ctx context.Context, threadID string) ([]domain.ThreadMessage, error) {
	rows, err := s.queries.ListThreadMessages(ctx, threadID)
	if err != nil {
		return nil, err
	}
	messages := make([]domain.ThreadMessage, 0, len(rows))
	for _, row := range rows {
		messages = append(messages, threadMessageFromRow(row))
	}
	return messages, nil
}

func (s *PostgresStore) CreateRun(ctx context.Context, input domain.CreateRunInput) (domain.RunRecord, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.RunRecord{}, err
	}
	defer tx.Rollback(ctx)

	q := s.queries.WithTx(tx)
	now := domain.Now()
	row, err := q.CreateRun(ctx, sqlc.CreateRunParams{
		RunID:        domain.NewID("run"),
		ThreadID:     input.ThreadID,
		UserID:       input.UserID,
		Goal:         input.Goal,
		Status:       string(domain.RunStatusQueued),
		WorkflowKind: "deep_agents",
		Mode:         nullableText("durable"),
		CreatedAt:    timestamptz(now),
		UpdatedAt:    timestamptz(now),
		Metadata:     jsonBytes(input.Metadata),
	})
	if err != nil {
		return domain.RunRecord{}, mapPgError(err)
	}
	if err := q.SetThreadLatestRun(ctx, sqlc.SetThreadLatestRunParams{
		ThreadID:    input.ThreadID,
		LatestRunID: nullableText(row.RunID),
		UpdatedAt:   timestamptz(now),
	}); err != nil {
		return domain.RunRecord{}, mapPgError(err)
	}
	for _, msg := range input.Messages {
		if _, err := q.InsertThreadMessage(ctx, sqlc.InsertThreadMessageParams{
			MessageID: domain.NewID("msg"),
			ThreadID:  input.ThreadID,
			Role:      msg.Role,
			Content:   msg.Content,
			CreatedAt: timestamptz(now),
			Metadata:  jsonBytes(msg.Metadata),
			RunID:     nullableText(row.RunID),
		}); err != nil {
			return domain.RunRecord{}, err
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.RunRecord{}, err
	}
	return runFromRow(row), nil
}

func (s *PostgresStore) GetRun(ctx context.Context, runID string) (domain.RunRecord, error) {
	row, err := s.queries.GetRun(ctx, runID)
	if err != nil {
		return domain.RunRecord{}, mapPgError(err)
	}
	return runFromRow(row), nil
}

func (s *PostgresStore) UpdateRunStatus(ctx context.Context, runID string, status domain.RunStatus, responseText string, errorText string) (domain.RunRecord, error) {
	row, err := s.queries.UpdateRunStatus(ctx, sqlc.UpdateRunStatusParams{
		RunID:     runID,
		Status:    string(status),
		Column3:   responseText,
		Column4:   errorText,
		UpdatedAt: timestamptz(domain.Now()),
	})
	if err != nil {
		return domain.RunRecord{}, mapPgError(err)
	}
	return runFromRow(row), nil
}

func (s *PostgresStore) AppendRunEvent(ctx context.Context, input domain.AppendRunEventInput) (domain.RunEventRecord, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.RunEventRecord{}, err
	}
	defer tx.Rollback(ctx)

	if _, err := tx.Exec(ctx, "SELECT pg_advisory_xact_lock(hashtext($1)::bigint)", input.RunID); err != nil {
		return domain.RunEventRecord{}, err
	}
	q := s.queries.WithTx(tx)
	sequence, err := q.NextRunEventSequence(ctx, input.RunID)
	if err != nil {
		return domain.RunEventRecord{}, mapPgError(err)
	}
	row, err := q.AppendRunEvent(ctx, sqlc.AppendRunEventParams{
		EventID:        domain.NewID("event"),
		SequenceNumber: int64(sequence),
		RunID:          input.RunID,
		ThreadID:       nullableText(input.ThreadID),
		EventKind:      input.EventKind,
		Ts:             timestamptz(domain.Now()),
		Message:        nullableText(input.Message),
		Payload:        jsonBytes(input.Payload),
	})
	if err != nil {
		return domain.RunEventRecord{}, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.RunEventRecord{}, err
	}
	return runEventFromRow(row), nil
}

func (s *PostgresStore) ListRunEvents(ctx context.Context, runID string, limit int) ([]domain.RunEventRecord, error) {
	rows, err := s.queries.ListRunEvents(ctx, sqlc.ListRunEventsParams{
		RunID: runID,
		Limit: limit32(limit, 500),
	})
	if err != nil {
		return nil, err
	}
	events := make([]domain.RunEventRecord, 0, len(rows))
	for _, row := range rows {
		events = append(events, runEventFromRow(row))
	}
	return events, nil
}

func (s *PostgresStore) CreateArtifact(ctx context.Context, input domain.CreateArtifactInput) (domain.ArtifactRecord, error) {
	now := domain.Now()
	row, err := s.queries.CreateArtifact(ctx, sqlc.CreateArtifactParams{
		ArtifactID: domain.NewID("artifact"),
		RunID:      input.RunID,
		ThreadID:   nullableText(input.ThreadID),
		Kind:       input.Kind,
		Path:       nullableText(input.Path),
		Title:      nullableText(input.Title),
		MimeType:   nullableText(input.MimeType),
		SizeBytes:  nullableInt8(input.SizeBytes),
		Sha256:     nullableText(input.SHA256),
		StorageUri: nullableText(input.StorageURI),
		CreatedAt:  timestamptz(now),
		UpdatedAt:  timestamptz(now),
		Metadata:   jsonBytes(input.Metadata),
	})
	if err != nil {
		return domain.ArtifactRecord{}, mapPgError(err)
	}
	return artifactFromRow(row), nil
}

func (s *PostgresStore) ListRunArtifacts(ctx context.Context, runID string, limit int) ([]domain.ArtifactRecord, error) {
	rows, err := s.queries.ListRunArtifacts(ctx, sqlc.ListRunArtifactsParams{
		RunID: runID,
		Limit: limit32(limit, 500),
	})
	if err != nil {
		return nil, err
	}
	artifacts := make([]domain.ArtifactRecord, 0, len(rows))
	for _, row := range rows {
		artifacts = append(artifacts, artifactFromRow(row))
	}
	return artifacts, nil
}

func (s *PostgresStore) GetArtifact(ctx context.Context, artifactID string) (domain.ArtifactRecord, error) {
	row, err := s.queries.GetArtifact(ctx, artifactID)
	if err != nil {
		return domain.ArtifactRecord{}, mapPgError(err)
	}
	return artifactFromRow(row), nil
}

func threadFromRow(row sqlc.ControlThread) domain.ThreadRecord {
	return domain.ThreadRecord{
		ThreadID:     row.ThreadID,
		UserID:       row.UserID,
		Title:        textValue(row.Title),
		Status:       domain.ThreadStatus(row.Status),
		CreatedAt:    timeValue(row.CreatedAt),
		UpdatedAt:    timeValue(row.UpdatedAt),
		LatestRunID:  textValue(row.LatestRunID),
		CheckpointID: textValue(row.CheckpointID),
		Summary:      textValue(row.Summary),
		Metadata:     jsonMap(row.Metadata),
	}
}

func threadMessageFromRow(row sqlc.ControlThreadMessage) domain.ThreadMessage {
	return domain.ThreadMessage{
		MessageID: row.MessageID,
		ThreadID:  row.ThreadID,
		Role:      row.Role,
		Content:   row.Content,
		CreatedAt: timeValue(row.CreatedAt),
		Metadata:  jsonMap(row.Metadata),
		RunID:     textValue(row.RunID),
	}
}

func runFromRow(row sqlc.ControlRun) domain.RunRecord {
	return domain.RunRecord{
		RunID:           row.RunID,
		ThreadID:        row.ThreadID,
		UserID:          row.UserID,
		Goal:            row.Goal,
		Status:          domain.RunStatus(row.Status),
		WorkflowKind:    row.WorkflowKind,
		Mode:            textValue(row.Mode),
		CurrentNode:     textValue(row.CurrentNode),
		ParentRunID:     textValue(row.ParentRunID),
		PlannerVersion:  textValue(row.PlannerVersion),
		AgentRole:       textValue(row.AgentRole),
		TraceGroupID:    textValue(row.TraceGroupID),
		CheckpointID:    textValue(row.CheckpointID),
		CheckpointState: jsonMap(row.CheckpointState),
		BudgetState:     jsonMap(row.BudgetState),
		ResponseText:    textValue(row.ResponseText),
		Error:           textValue(row.Error),
		CreatedAt:       timeValue(row.CreatedAt),
		UpdatedAt:       timeValue(row.UpdatedAt),
		StartedAt:       timePtr(row.StartedAt),
		CompletedAt:     timePtr(row.CompletedAt),
		Metadata:        jsonMap(row.Metadata),
	}
}

func runEventFromRow(row sqlc.ControlRunEvent) domain.RunEventRecord {
	return domain.RunEventRecord{
		EventID:      row.EventID,
		Sequence:     row.SequenceNumber,
		RunID:        row.RunID,
		ThreadID:     textValue(row.ThreadID),
		EventKind:    row.EventKind,
		EventType:    textValue(row.EventType),
		NodeName:     textValue(row.NodeName),
		TaskID:       textValue(row.TaskID),
		CheckpointID: textValue(row.CheckpointID),
		ScopeID:      textValue(row.ScopeID),
		AgentRole:    textValue(row.AgentRole),
		Level:        textValue(row.Level),
		TS:           timeValue(row.Ts),
		Message:      textValue(row.Message),
		Payload:      jsonMap(row.Payload),
	}
}

func artifactFromRow(row sqlc.ControlArtifact) domain.ArtifactRecord {
	return domain.ArtifactRecord{
		ArtifactID:    row.ArtifactID,
		RunID:         row.RunID,
		ThreadID:      textValue(row.ThreadID),
		Kind:          row.Kind,
		Path:          textValue(row.Path),
		SourcePath:    textValue(row.SourcePath),
		PreviewPath:   textValue(row.PreviewPath),
		Title:         textValue(row.Title),
		ResultGroupID: textValue(row.ResultGroupID),
		MimeType:      textValue(row.MimeType),
		SizeBytes:     int8Value(row.SizeBytes),
		SHA256:        textValue(row.Sha256),
		StorageURI:    textValue(row.StorageUri),
		ToolName:      textValue(row.ToolName),
		Category:      textValue(row.Category),
		CreatedAt:     timeValue(row.CreatedAt),
		UpdatedAt:     timeValue(row.UpdatedAt),
		Metadata:      jsonMap(row.Metadata),
	}
}

func mapPgError(err error) error {
	if errors.Is(err, pgx.ErrNoRows) {
		return ErrNotFound
	}
	return err
}

func jsonBytes(value domain.JSONMap) []byte {
	if value == nil {
		value = domain.JSONMap{}
	}
	data, _ := json.Marshal(value)
	return data
}

func jsonMap(data []byte) domain.JSONMap {
	if len(data) == 0 {
		return domain.JSONMap{}
	}
	var value domain.JSONMap
	if err := json.Unmarshal(data, &value); err != nil {
		return domain.JSONMap{}
	}
	return value
}

func nullableText(value string) pgtype.Text {
	if value == "" {
		return pgtype.Text{}
	}
	return pgtype.Text{String: value, Valid: true}
}

func textValue(value pgtype.Text) string {
	if !value.Valid {
		return ""
	}
	return value.String
}

func nullableInt8(value int64) pgtype.Int8 {
	if value == 0 {
		return pgtype.Int8{}
	}
	return pgtype.Int8{Int64: value, Valid: true}
}

func int8Value(value pgtype.Int8) int64 {
	if !value.Valid {
		return 0
	}
	return value.Int64
}

func timestamptz(value time.Time) pgtype.Timestamptz {
	return pgtype.Timestamptz{Time: value.UTC(), Valid: true}
}

func timeValue(value pgtype.Timestamptz) time.Time {
	if !value.Valid {
		return time.Time{}
	}
	return value.Time.UTC()
}

func timePtr(value pgtype.Timestamptz) *time.Time {
	if !value.Valid {
		return nil
	}
	t := value.Time.UTC()
	return &t
}

func limit32(limit int, fallback int32) int32 {
	if limit <= 0 {
		return fallback
	}
	return int32(limit)
}
