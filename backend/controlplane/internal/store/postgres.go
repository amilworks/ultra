package store

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store/sqlc"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
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

func (s *PostgresStore) CreateUser(ctx context.Context, input domain.CreateUserInput) (domain.UserAccount, error) {
	now := domain.Now()
	userID := strings.TrimSpace(input.UserID)
	if userID == "" {
		userID = domain.NewID("user")
	}
	role := strings.TrimSpace(input.Role)
	if role == "" {
		role = "researcher"
	}
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	row := s.pool.QueryRow(ctx, `
INSERT INTO control_users (user_id, email, display_name, role, status, org_id, created_at, updated_at, metadata)
VALUES ($1, NULLIF($2, ''), NULLIF($3, ''), $4, $5, NULLIF($6, ''), $7, $8, $9)
RETURNING user_id, COALESCE(email, ''), COALESCE(display_name, ''), role, status, COALESCE(org_id, ''), created_at, updated_at, metadata`,
		userID,
		normalizeEmail(input.Email),
		strings.TrimSpace(input.DisplayName),
		role,
		status,
		strings.TrimSpace(input.OrgID),
		now,
		now,
		jsonBytes(input.Metadata),
	)
	return scanUserAccount(row)
}

func (s *PostgresStore) CreateOrganization(ctx context.Context, input domain.CreateOrganizationInput) (domain.Organization, error) {
	now := domain.Now()
	orgID := normalizeOrgID(input.OrgID)
	if orgID == "" {
		orgID = domain.NewID("org")
	}
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	name := strings.TrimSpace(input.Name)
	if name == "" {
		name = orgID
	}
	row := s.pool.QueryRow(ctx, `
INSERT INTO control_organizations (org_id, name, status, created_at, updated_at, metadata)
VALUES ($1, $2, $3, $4, $5, $6)
RETURNING org_id, name, status, created_at, updated_at, metadata`,
		orgID,
		name,
		status,
		now,
		now,
		jsonBytes(input.Metadata),
	)
	return scanOrganization(row)
}

func (s *PostgresStore) ListOrganizations(ctx context.Context, limit int, query string) ([]domain.Organization, error) {
	query = strings.TrimSpace(query)
	rows, err := s.pool.Query(ctx, `
SELECT org_id, name, status, created_at, updated_at, metadata
FROM control_organizations
WHERE $1 = ''
   OR org_id ILIKE '%' || $1 || '%'
   OR name ILIKE '%' || $1 || '%'
   OR status ILIKE '%' || $1 || '%'
ORDER BY created_at DESC
LIMIT $2`, query, limit32(limit, 250))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	orgs := []domain.Organization{}
	for rows.Next() {
		org, err := scanOrganization(rows)
		if err != nil {
			return nil, err
		}
		orgs = append(orgs, org)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return orgs, nil
}

func (s *PostgresStore) ListUsers(ctx context.Context, limit int, query string) ([]domain.UserAccount, error) {
	query = strings.TrimSpace(query)
	rows, err := s.pool.Query(ctx, `
SELECT user_id, COALESCE(email, ''), COALESCE(display_name, ''), role, status, COALESCE(org_id, ''), created_at, updated_at, metadata
FROM control_users
WHERE $1 = ''
   OR user_id ILIKE '%' || $1 || '%'
   OR COALESCE(email, '') ILIKE '%' || $1 || '%'
   OR COALESCE(display_name, '') ILIKE '%' || $1 || '%'
   OR role ILIKE '%' || $1 || '%'
   OR COALESCE(org_id, '') ILIKE '%' || $1 || '%'
ORDER BY created_at DESC
LIMIT $2`, query, limit32(limit, 250))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	users := []domain.UserAccount{}
	for rows.Next() {
		user, err := scanUserAccount(rows)
		if err != nil {
			return nil, err
		}
		users = append(users, user)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return users, nil
}

func (s *PostgresStore) GetUserByID(ctx context.Context, userID string) (domain.UserAccount, bool, error) {
	userID = strings.TrimSpace(userID)
	if userID == "" {
		return domain.UserAccount{}, false, nil
	}
	row := s.pool.QueryRow(ctx, `
SELECT user_id, COALESCE(email, ''), COALESCE(display_name, ''), role, status, COALESCE(org_id, ''), created_at, updated_at, metadata
FROM control_users
WHERE user_id = $1`, userID)
	user, err := scanUserAccount(row)
	if err != nil {
		if errors.Is(err, ErrNotFound) {
			return domain.UserAccount{}, false, nil
		}
		return domain.UserAccount{}, false, err
	}
	return user, true, nil
}

func (s *PostgresStore) GetUserByEmail(ctx context.Context, email string) (domain.UserAccount, bool, error) {
	email = normalizeEmail(email)
	if email == "" {
		return domain.UserAccount{}, false, nil
	}
	row := s.pool.QueryRow(ctx, `
SELECT user_id, COALESCE(email, ''), COALESCE(display_name, ''), role, status, COALESCE(org_id, ''), created_at, updated_at, metadata
FROM control_users
WHERE lower(email) = $1`, email)
	user, err := scanUserAccount(row)
	if err != nil {
		if errors.Is(err, ErrNotFound) {
			return domain.UserAccount{}, false, nil
		}
		return domain.UserAccount{}, false, err
	}
	return user, true, nil
}

func (s *PostgresStore) UpdateUserStatus(ctx context.Context, userID string, status string) (domain.UserAccount, error) {
	userID = strings.TrimSpace(userID)
	status = strings.TrimSpace(status)
	if status == "" {
		status = "disabled"
	}
	row := s.pool.QueryRow(ctx, `
UPDATE control_users
SET status = $2,
    updated_at = $3
WHERE user_id = $1
RETURNING user_id, COALESCE(email, ''), COALESCE(display_name, ''), role, status, COALESCE(org_id, ''), created_at, updated_at, metadata`,
		userID,
		status,
		domain.Now(),
	)
	return scanUserAccount(row)
}

func (s *PostgresStore) UpsertBisqueCredential(ctx context.Context, input domain.UpsertBisqueCredentialInput) (domain.BisqueCredentialRecord, error) {
	now := domain.Now()
	sessionID := strings.TrimSpace(input.SessionID)
	if sessionID == "" {
		sessionID = domain.NewID("bisque_session")
	}
	userID := strings.TrimSpace(input.UserID)
	if userID == "" {
		userID = "local-user"
	}
	orgID := strings.TrimSpace(input.OrgID)
	if orgID == "" {
		orgID = "local-org"
	}
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	lastVerifiedAt := pgtype.Timestamptz{}
	if !input.LastVerifiedAt.IsZero() {
		lastVerifiedAt = pgtype.Timestamptz{Time: input.LastVerifiedAt.UTC(), Valid: true}
	}
	row := s.pool.QueryRow(ctx, `
INSERT INTO control_bisque_credentials (
  session_id, user_id, org_id, root_url, username,
  password_ciphertext, password_nonce, password_key_id, password_algorithm,
  status, last_verified_at, created_at, updated_at, metadata
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
ON CONFLICT (user_id, org_id, root_url) DO UPDATE
SET username = EXCLUDED.username,
    password_ciphertext = EXCLUDED.password_ciphertext,
    password_nonce = EXCLUDED.password_nonce,
    password_key_id = EXCLUDED.password_key_id,
    password_algorithm = EXCLUDED.password_algorithm,
    status = EXCLUDED.status,
    last_verified_at = EXCLUDED.last_verified_at,
    updated_at = EXCLUDED.updated_at,
    metadata = EXCLUDED.metadata
RETURNING session_id, user_id, COALESCE(org_id, ''), root_url, username,
          password_ciphertext, password_nonce, password_key_id, password_algorithm,
          status, last_verified_at, created_at, updated_at, metadata`,
		sessionID,
		userID,
		orgID,
		strings.TrimRight(strings.TrimSpace(input.RootURL), "/"),
		strings.TrimSpace(input.Username),
		strings.TrimSpace(input.PasswordCiphertext),
		strings.TrimSpace(input.PasswordNonce),
		strings.TrimSpace(input.PasswordKeyID),
		strings.TrimSpace(input.PasswordAlgorithm),
		status,
		lastVerifiedAt,
		now,
		now,
		jsonBytes(input.Metadata),
	)
	return scanBisqueCredential(row)
}

func (s *PostgresStore) GetBisqueCredentialBySessionID(ctx context.Context, sessionID string) (domain.BisqueCredentialRecord, bool, error) {
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return domain.BisqueCredentialRecord{}, false, nil
	}
	row := s.pool.QueryRow(ctx, `
SELECT session_id, user_id, COALESCE(org_id, ''), root_url, username,
       password_ciphertext, password_nonce, password_key_id, password_algorithm,
       status, last_verified_at, created_at, updated_at, metadata
FROM control_bisque_credentials
WHERE session_id = $1 AND status <> 'deleted'`,
		sessionID,
	)
	record, err := scanBisqueCredential(row)
	if errors.Is(err, ErrNotFound) {
		return domain.BisqueCredentialRecord{}, false, nil
	}
	if err != nil {
		return domain.BisqueCredentialRecord{}, false, err
	}
	return record, true, nil
}

func (s *PostgresStore) DeleteBisqueCredentialBySessionID(ctx context.Context, sessionID string) error {
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return nil
	}
	_, err := s.pool.Exec(ctx, `
UPDATE control_bisque_credentials
SET status = 'deleted',
    updated_at = $2
WHERE session_id = $1`,
		sessionID,
		domain.Now(),
	)
	return err
}

func (s *PostgresStore) UpsertWorkerHeartbeat(ctx context.Context, input domain.UpsertWorkerHeartbeatInput) (domain.WorkerHeartbeatRecord, error) {
	now := domain.Now()
	workerID := strings.TrimSpace(input.WorkerID)
	if workerID == "" {
		return domain.WorkerHeartbeatRecord{}, ErrConflict
	}
	workerKind := strings.TrimSpace(input.WorkerKind)
	if workerKind == "" {
		workerKind = "worker"
	}
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "alive"
	}
	heartbeatAt := input.LastHeartbeatAt
	if heartbeatAt.IsZero() {
		heartbeatAt = now
	}
	heartbeatAt = heartbeatAt.UTC()
	startedAt := input.StartedAt
	if startedAt.IsZero() {
		startedAt = heartbeatAt
	}
	startedAt = startedAt.UTC()
	row := s.pool.QueryRow(ctx, `
INSERT INTO control_worker_heartbeats (
  worker_id, worker_kind, status, current_run_id, hostname, version,
  started_at, last_heartbeat_at, updated_at, metadata
)
VALUES ($1, $2, $3, NULLIF($4, ''), NULLIF($5, ''), NULLIF($6, ''), $7, $8, $9, $10)
ON CONFLICT (worker_id) DO UPDATE
SET worker_kind = EXCLUDED.worker_kind,
    status = EXCLUDED.status,
    current_run_id = EXCLUDED.current_run_id,
    hostname = EXCLUDED.hostname,
    version = EXCLUDED.version,
    started_at = control_worker_heartbeats.started_at,
    last_heartbeat_at = EXCLUDED.last_heartbeat_at,
    updated_at = EXCLUDED.updated_at,
    metadata = EXCLUDED.metadata
RETURNING worker_id, worker_kind, status, COALESCE(current_run_id, ''), COALESCE(hostname, ''), COALESCE(version, ''),
          started_at, last_heartbeat_at, updated_at, metadata`,
		workerID,
		workerKind,
		status,
		strings.TrimSpace(input.CurrentRunID),
		strings.TrimSpace(input.Hostname),
		strings.TrimSpace(input.Version),
		startedAt,
		heartbeatAt,
		now,
		jsonBytes(input.Metadata),
	)
	return scanWorkerHeartbeat(row)
}

func (s *PostgresStore) ListWorkerHeartbeats(ctx context.Context, limit int) ([]domain.WorkerHeartbeatRecord, error) {
	rows, err := s.pool.Query(ctx, `
SELECT worker_id, worker_kind, status, COALESCE(current_run_id, ''), COALESCE(hostname, ''), COALESCE(version, ''),
       started_at, last_heartbeat_at, updated_at, metadata
FROM control_worker_heartbeats
ORDER BY last_heartbeat_at DESC
LIMIT $1`, limit32(limit, 250))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	workers := []domain.WorkerHeartbeatRecord{}
	for rows.Next() {
		worker, err := scanWorkerHeartbeat(rows)
		if err != nil {
			return nil, err
		}
		workers = append(workers, worker)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return workers, nil
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

func (s *PostgresStore) GetThreadForUser(ctx context.Context, threadID string, userID string) (domain.ThreadRecord, error) {
	row, err := s.queries.GetThreadForUser(ctx, sqlc.GetThreadForUserParams{
		ThreadID: threadID,
		UserID:   userID,
	})
	if err != nil {
		return domain.ThreadRecord{}, mapPgError(err)
	}
	return threadFromRow(row), nil
}

func (s *PostgresStore) UpdateThreadForUser(ctx context.Context, input domain.UpdateThreadInput) (domain.ThreadRecord, error) {
	now := domain.Now()
	row, err := s.pool.Query(ctx, `
UPDATE control_threads
SET title = COALESCE(NULLIF($3, ''), title),
    metadata = COALESCE(metadata, '{}'::jsonb) || $4::jsonb,
    updated_at = $5
WHERE thread_id = $1 AND user_id = $2
RETURNING thread_id, user_id, title, status, created_at, updated_at, latest_run_id, checkpoint_id, summary, metadata`,
		input.ThreadID,
		input.UserID,
		normalizedThreadTitle(input.Title),
		jsonBytes(mapOrEmpty(input.Metadata)),
		now,
	)
	if err != nil {
		return domain.ThreadRecord{}, mapPgError(err)
	}
	defer row.Close()
	thread, err := pgx.CollectOneRow(row, pgx.RowToStructByName[sqlc.ControlThread])
	if err != nil {
		return domain.ThreadRecord{}, mapPgError(err)
	}
	return threadFromRow(thread), nil
}

func (s *PostgresStore) ApplyGeneratedThreadTitle(ctx context.Context, input domain.ApplyGeneratedThreadTitleInput) (domain.ThreadRecord, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.ThreadRecord{}, err
	}
	defer tx.Rollback(ctx)

	row, err := lockedControlThread(ctx, tx, input.ThreadID)
	if err != nil {
		return domain.ThreadRecord{}, mapPgError(err)
	}
	thread := threadFromRow(row)
	title := normalizedThreadTitle(input.Title)
	if title == "" || !generatedThreadTitleEligible(thread) {
		if err := tx.Commit(ctx); err != nil {
			return domain.ThreadRecord{}, err
		}
		return thread, nil
	}
	now := domain.Now()
	metadata := generatedThreadTitleMetadata(thread.Metadata, input, thread.Title, now)
	updated, err := tx.Query(ctx, `
UPDATE control_threads
SET title = $2,
    metadata = $3,
    updated_at = $4
WHERE thread_id = $1
RETURNING thread_id, user_id, title, status, created_at, updated_at, latest_run_id, checkpoint_id, summary, metadata`,
		input.ThreadID,
		title,
		jsonBytes(metadata),
		now,
	)
	if err != nil {
		return domain.ThreadRecord{}, mapPgError(err)
	}
	defer updated.Close()
	updatedThread, err := pgx.CollectOneRow(updated, pgx.RowToStructByName[sqlc.ControlThread])
	if err != nil {
		return domain.ThreadRecord{}, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.ThreadRecord{}, err
	}
	return threadFromRow(updatedThread), nil
}

func lockedControlThread(ctx context.Context, tx pgx.Tx, threadID string) (sqlc.ControlThread, error) {
	rows, err := tx.Query(ctx, `
SELECT thread_id, user_id, title, status, created_at, updated_at, latest_run_id, checkpoint_id, summary, metadata
FROM control_threads
WHERE thread_id = $1
FOR UPDATE`, threadID)
	if err != nil {
		return sqlc.ControlThread{}, err
	}
	defer rows.Close()
	return pgx.CollectOneRow(rows, pgx.RowToStructByName[sqlc.ControlThread])
}

func (s *PostgresStore) ListThreads(ctx context.Context, limit int, offset int, status string) (domain.ThreadListPage, error) {
	resolvedLimit := limit32(limit, 100)
	resolvedOffset := max(offset, 0)
	resolvedStatus := strings.TrimSpace(status)
	totalCount, err := s.queries.CountThreads(ctx, resolvedStatus)
	if err != nil {
		return domain.ThreadListPage{}, err
	}
	rows, err := s.queries.ListThreads(ctx, sqlc.ListThreadsParams{
		Column1: resolvedStatus,
		Limit:   resolvedLimit,
		Offset:  int32(resolvedOffset),
	})
	if err != nil {
		return domain.ThreadListPage{}, err
	}
	threads := make([]domain.ThreadRecord, 0, len(rows))
	for _, row := range rows {
		threads = append(threads, threadFromRow(row))
	}
	return domain.ThreadListPage{
		Threads:    threads,
		TotalCount: int(totalCount),
		Limit:      int(resolvedLimit),
		Offset:     resolvedOffset,
	}, nil
}

func (s *PostgresStore) ListThreadsForUser(ctx context.Context, userID string, limit int, offset int, status string) (domain.ThreadListPage, error) {
	resolvedLimit := limit32(limit, 100)
	resolvedOffset := max(offset, 0)
	resolvedStatus := strings.TrimSpace(status)
	totalCount, err := s.queries.CountThreadsForUser(ctx, sqlc.CountThreadsForUserParams{
		UserID:  userID,
		Column2: resolvedStatus,
	})
	if err != nil {
		return domain.ThreadListPage{}, err
	}
	rows, err := s.queries.ListThreadsForUser(ctx, sqlc.ListThreadsForUserParams{
		UserID:  userID,
		Column2: resolvedStatus,
		Limit:   resolvedLimit,
		Offset:  int32(resolvedOffset),
	})
	if err != nil {
		return domain.ThreadListPage{}, err
	}
	threads := make([]domain.ThreadRecord, 0, len(rows))
	for _, row := range rows {
		threads = append(threads, threadFromRow(row))
	}
	return domain.ThreadListPage{
		Threads:    threads,
		TotalCount: int(totalCount),
		Limit:      int(resolvedLimit),
		Offset:     resolvedOffset,
	}, nil
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

func (s *PostgresStore) ListThreadMessagesForUser(ctx context.Context, threadID string, userID string) ([]domain.ThreadMessage, error) {
	if _, err := s.GetThreadForUser(ctx, threadID, userID); err != nil {
		return nil, err
	}
	rows, err := s.queries.ListThreadMessagesForUser(ctx, sqlc.ListThreadMessagesForUserParams{
		ThreadID: threadID,
		UserID:   userID,
	})
	if err != nil {
		return nil, err
	}
	messages := make([]domain.ThreadMessage, 0, len(rows))
	for _, row := range rows {
		messages = append(messages, threadMessageFromRow(row))
	}
	return messages, nil
}

func (s *PostgresStore) AppendThreadMessage(ctx context.Context, message domain.ThreadMessage) (domain.ThreadMessage, error) {
	if message.MessageID == "" {
		message.MessageID = domain.NewID("msg")
	}
	if message.CreatedAt.IsZero() {
		message.CreatedAt = domain.Now()
	}
	row, err := s.queries.InsertThreadMessage(ctx, sqlc.InsertThreadMessageParams{
		MessageID: message.MessageID,
		ThreadID:  message.ThreadID,
		Role:      message.Role,
		Content:   message.Content,
		CreatedAt: timestamptz(message.CreatedAt),
		Metadata:  jsonBytes(message.Metadata),
		RunID:     nullableText(message.RunID),
	})
	if err != nil {
		return domain.ThreadMessage{}, mapPgError(err)
	}
	return threadMessageFromRow(row), nil
}

func (s *PostgresStore) CreateRun(ctx context.Context, input domain.CreateRunInput) (domain.RunRecord, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.RunRecord{}, err
	}
	defer tx.Rollback(ctx)

	q := s.queries.WithTx(tx)
	now := domain.Now()
	workflowKind := input.WorkflowKind
	if workflowKind == "" {
		workflowKind = "deep_agents"
	}
	mode := input.Mode
	if mode == "" {
		mode = "durable"
	}
	row, err := q.CreateRun(ctx, sqlc.CreateRunParams{
		RunID:        domain.NewID("run"),
		ThreadID:     input.ThreadID,
		UserID:       input.UserID,
		Goal:         input.Goal,
		Status:       string(domain.RunStatusQueued),
		WorkflowKind: workflowKind,
		Mode:         nullableText(mode),
		CreatedAt:    timestamptz(now),
		UpdatedAt:    timestamptz(now),
		Metadata:     jsonBytes(input.Metadata),
	})
	if err != nil {
		return domain.RunRecord{}, mapPgError(err)
	}
	if !input.Internal {
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
				RunID:     nullableText(threadMessageRunID(msg, row.RunID)),
			}); err != nil {
				return domain.RunRecord{}, err
			}
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.RunRecord{}, err
	}
	return runFromRow(row), nil
}

func (s *PostgresStore) FindRunByIdempotencyKey(ctx context.Context, threadID string, userID string, idempotencyKey string) (domain.RunRecord, bool, error) {
	row, err := s.pool.Query(ctx, `
SELECT run_id, thread_id, user_id, goal, status, workflow_kind, mode, current_node, parent_run_id,
       planner_version, agent_role, trace_group_id, checkpoint_id, checkpoint_state, budget_state,
       response_text, error, created_at, updated_at, started_at, completed_at, metadata
FROM control_runs
WHERE thread_id = $1 AND user_id = $2 AND metadata->>'idempotency_key' = $3
ORDER BY created_at ASC
LIMIT 1`, threadID, userID, idempotencyKey)
	if err != nil {
		return domain.RunRecord{}, false, err
	}
	defer row.Close()
	rows, err := pgx.CollectRows(row, pgx.RowToStructByName[sqlc.ControlRun])
	if err != nil {
		return domain.RunRecord{}, false, err
	}
	if len(rows) == 0 {
		return domain.RunRecord{}, false, nil
	}
	return runFromRow(rows[0]), true, nil
}

func (s *PostgresStore) MarkRunDispatched(ctx context.Context, runID string, dispatchedAt time.Time) (domain.RunRecord, error) {
	if dispatchedAt.IsZero() {
		dispatchedAt = domain.Now()
	}
	tag, err := s.pool.Exec(ctx, `
UPDATE control_runs
SET metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object('job_dispatched_at', $2::text),
    updated_at = $3
WHERE run_id = $1`,
		runID,
		dispatchedAt.UTC().Format(time.RFC3339Nano),
		domain.Now(),
	)
	if err != nil {
		return domain.RunRecord{}, mapPgError(err)
	}
	if tag.RowsAffected() == 0 {
		return domain.RunRecord{}, ErrNotFound
	}
	return s.GetRun(ctx, runID)
}

func (s *PostgresStore) GetRun(ctx context.Context, runID string) (domain.RunRecord, error) {
	row, err := s.queries.GetRun(ctx, runID)
	if err != nil {
		return domain.RunRecord{}, mapPgError(err)
	}
	return runFromRow(row), nil
}

func (s *PostgresStore) GetRunForUser(ctx context.Context, runID string, userID string) (domain.RunRecord, error) {
	row, err := s.queries.GetRunForUser(ctx, sqlc.GetRunForUserParams{
		RunID:  runID,
		UserID: userID,
	})
	if err != nil {
		return domain.RunRecord{}, mapPgError(err)
	}
	return runFromRow(row), nil
}

func (s *PostgresStore) ListRuns(ctx context.Context, threadID string, status string, limit int, offset int) ([]domain.RunRecord, error) {
	rows, err := s.queries.ListRuns(ctx, sqlc.ListRunsParams{
		Column1: threadID,
		Column2: status,
		Limit:   limit32(limit, 100),
		Offset:  int32(max(offset, 0)),
	})
	if err != nil {
		return nil, err
	}
	runs := make([]domain.RunRecord, 0, len(rows))
	for _, row := range rows {
		runs = append(runs, runFromRow(row))
	}
	return runs, nil
}

func (s *PostgresStore) ListRunsForUser(ctx context.Context, userID string, threadID string, status string, limit int, offset int) ([]domain.RunRecord, error) {
	if strings.TrimSpace(threadID) != "" {
		if _, err := s.GetThreadForUser(ctx, threadID, userID); err != nil {
			return nil, err
		}
	}
	rows, err := s.queries.ListRunsForUser(ctx, sqlc.ListRunsForUserParams{
		UserID:  userID,
		Column2: strings.TrimSpace(threadID),
		Column3: strings.TrimSpace(status),
		Limit:   limit32(limit, 100),
		Offset:  int32(max(offset, 0)),
	})
	if err != nil {
		return nil, err
	}
	runs := make([]domain.RunRecord, 0, len(rows))
	for _, row := range rows {
		runs = append(runs, runFromRow(row))
	}
	return runs, nil
}

func (s *PostgresStore) GetRunLease(ctx context.Context, runID string) (domain.RunLeaseRecord, bool, error) {
	lease, err := scanRunLease(s.pool.QueryRow(ctx, `
SELECT run_id, worker_id, lease_token, lease_expires_at, created_at, updated_at
FROM control_run_leases
WHERE run_id = $1`, runID))
	if err == nil {
		return lease, true, nil
	}
	if errors.Is(err, ErrNotFound) {
		var exists bool
		if existsErr := s.pool.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM control_runs WHERE run_id = $1)`, runID).Scan(&exists); existsErr != nil {
			return domain.RunLeaseRecord{}, false, existsErr
		}
		if !exists {
			return domain.RunLeaseRecord{}, false, ErrNotFound
		}
		return domain.RunLeaseRecord{}, false, nil
	}
	return domain.RunLeaseRecord{}, false, err
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
		if errors.Is(err, pgx.ErrNoRows) {
			existing, getErr := s.GetRun(ctx, runID)
			if getErr != nil {
				return domain.RunRecord{}, getErr
			}
			if isTerminalRunStatus(existing.Status) {
				return existing, nil
			}
		}
		return domain.RunRecord{}, mapPgError(err)
	}
	return runFromRow(row), nil
}

func (s *PostgresStore) CompleteRun(ctx context.Context, input domain.CompleteRunInput) (domain.RunRecord, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.RunRecord{}, err
	}
	defer tx.Rollback(ctx)

	row, err := lockedControlRun(ctx, tx, input.RunID)
	if err != nil {
		return domain.RunRecord{}, mapPgError(err)
	}
	run := runFromRow(row)
	responseText := strings.TrimSpace(input.ResponseText)
	if run.Status == domain.RunStatusSucceeded {
		if strings.TrimSpace(run.ResponseText) == "" && responseText != "" {
			updated, err := repairSucceededRunResponseTextTx(ctx, tx, run.RunID, responseText, domain.Now())
			if err != nil {
				return domain.RunRecord{}, mapPgError(err)
			}
			run = runFromRow(updated)
		}
		if err := appendCompletedAssistantMessageTx(ctx, tx, run, responseText); err != nil {
			return domain.RunRecord{}, err
		}
		if err := tx.Commit(ctx); err != nil {
			return domain.RunRecord{}, err
		}
		return run, nil
	}
	if isTerminalRunStatus(run.Status) {
		if err := tx.Commit(ctx); err != nil {
			return domain.RunRecord{}, err
		}
		return run, nil
	}
	if err := appendCompletedAssistantMessageTx(ctx, tx, run, responseText); err != nil {
		return domain.RunRecord{}, err
	}
	updated, err := completeControlRunTx(ctx, tx, run.RunID, responseText, domain.Now())
	if err != nil {
		return domain.RunRecord{}, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.RunRecord{}, err
	}
	return runFromRow(updated), nil
}

func lockedControlRun(ctx context.Context, tx pgx.Tx, runID string) (sqlc.ControlRun, error) {
	rows, err := tx.Query(ctx, `
SELECT run_id, thread_id, user_id, goal, status, workflow_kind, mode, current_node, parent_run_id,
       planner_version, agent_role, trace_group_id, checkpoint_id, checkpoint_state, budget_state,
       response_text, error, created_at, updated_at, started_at, completed_at, metadata
FROM control_runs
WHERE run_id = $1
FOR UPDATE`, runID)
	if err != nil {
		return sqlc.ControlRun{}, err
	}
	defer rows.Close()
	return pgx.CollectOneRow(rows, pgx.RowToStructByName[sqlc.ControlRun])
}

func repairSucceededRunResponseTextTx(ctx context.Context, tx pgx.Tx, runID string, responseText string, now time.Time) (sqlc.ControlRun, error) {
	rows, err := tx.Query(ctx, `
UPDATE control_runs
SET response_text = $2,
    updated_at = $3
WHERE run_id = $1
  AND status = 'succeeded'
  AND COALESCE(response_text, '') = ''
RETURNING run_id, thread_id, user_id, goal, status, workflow_kind, mode, current_node, parent_run_id,
          planner_version, agent_role, trace_group_id, checkpoint_id, checkpoint_state, budget_state,
          response_text, error, created_at, updated_at, started_at, completed_at, metadata`,
		runID,
		responseText,
		timestamptz(now),
	)
	if err != nil {
		return sqlc.ControlRun{}, err
	}
	defer rows.Close()
	return pgx.CollectOneRow(rows, pgx.RowToStructByName[sqlc.ControlRun])
}

func appendCompletedAssistantMessageTx(ctx context.Context, tx pgx.Tx, run domain.RunRecord, responseText string) error {
	if responseText == "" || isInternalRunMetadata(run.Metadata) {
		return nil
	}
	var exists bool
	if err := tx.QueryRow(ctx, `
SELECT EXISTS(
  SELECT 1
  FROM control_thread_messages
  WHERE thread_id = $1
    AND run_id = $2
    AND lower(btrim(role)) = 'assistant'
    AND content = $3
)`, run.ThreadID, run.RunID, responseText).Scan(&exists); err != nil {
		return err
	}
	if exists {
		return nil
	}
	now := domain.Now()
	_, err := tx.Exec(ctx, `
INSERT INTO control_thread_messages (message_id, thread_id, role, content, created_at, metadata, run_id)
VALUES ($1, $2, 'assistant', $3, $4, '{}'::jsonb, $5)`,
		domain.NewID("msg"),
		run.ThreadID,
		responseText,
		now,
		run.RunID,
	)
	return mapPgError(err)
}

func completeControlRunTx(ctx context.Context, tx pgx.Tx, runID string, responseText string, now time.Time) (sqlc.ControlRun, error) {
	rows, err := tx.Query(ctx, `
UPDATE control_runs
SET status = 'succeeded',
    response_text = NULLIF($2, ''),
    error = NULL,
    updated_at = $3,
    completed_at = $3
WHERE run_id = $1
  AND status NOT IN ('succeeded', 'failed', 'canceled')
RETURNING run_id, thread_id, user_id, goal, status, workflow_kind, mode, current_node, parent_run_id,
          planner_version, agent_role, trace_group_id, checkpoint_id, checkpoint_state, budget_state,
          response_text, error, created_at, updated_at, started_at, completed_at, metadata`,
		runID,
		responseText,
		timestamptz(now),
	)
	if err != nil {
		return sqlc.ControlRun{}, err
	}
	defer rows.Close()
	return pgx.CollectOneRow(rows, pgx.RowToStructByName[sqlc.ControlRun])
}

func (s *PostgresStore) AcquireRunLease(ctx context.Context, input domain.AcquireRunLeaseInput) (domain.RunLeaseRecord, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.RunLeaseRecord{}, err
	}
	defer tx.Rollback(ctx)

	var status string
	if err := tx.QueryRow(ctx, `SELECT status FROM control_runs WHERE run_id = $1 FOR UPDATE`, input.RunID).Scan(&status); err != nil {
		return domain.RunLeaseRecord{}, mapPgError(err)
	}
	if isTerminalRunStatus(domain.RunStatus(status)) {
		return domain.RunLeaseRecord{}, ErrConflict
	}
	now := leaseNow(input.Now)
	ttl := positiveLeaseTTL(input.TTL)
	existing, err := scanRunLease(tx.QueryRow(ctx, `
SELECT run_id, worker_id, lease_token, lease_expires_at, created_at, updated_at
FROM control_run_leases
WHERE run_id = $1
FOR UPDATE`, input.RunID))
	if err == nil && existing.LeaseExpiresAt.After(now) {
		return domain.RunLeaseRecord{}, ErrConflict
	}
	if err != nil && !errors.Is(err, ErrNotFound) {
		return domain.RunLeaseRecord{}, err
	}
	lease := domain.RunLeaseRecord{
		RunID:          input.RunID,
		WorkerID:       strings.TrimSpace(input.WorkerID),
		LeaseToken:     domain.NewID("lease"),
		LeaseExpiresAt: now.Add(ttl),
		CreatedAt:      now,
		UpdatedAt:      now,
	}
	row := tx.QueryRow(ctx, `
INSERT INTO control_run_leases (run_id, worker_id, lease_token, lease_expires_at, created_at, updated_at)
VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT (run_id) DO UPDATE
SET worker_id = EXCLUDED.worker_id,
    lease_token = EXCLUDED.lease_token,
    lease_expires_at = EXCLUDED.lease_expires_at,
    updated_at = EXCLUDED.updated_at
RETURNING run_id, worker_id, lease_token, lease_expires_at, created_at, updated_at`,
		lease.RunID,
		lease.WorkerID,
		lease.LeaseToken,
		lease.LeaseExpiresAt,
		lease.CreatedAt,
		lease.UpdatedAt,
	)
	lease, err = scanRunLease(row)
	if err != nil {
		return domain.RunLeaseRecord{}, err
	}
	if _, err := tx.Exec(ctx, `
UPDATE control_runs
SET status = 'running',
    updated_at = $2,
    started_at = COALESCE(started_at, $2)
WHERE run_id = $1
  AND status NOT IN ('succeeded', 'failed', 'canceled')`, input.RunID, now); err != nil {
		return domain.RunLeaseRecord{}, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.RunLeaseRecord{}, err
	}
	return lease, nil
}

func (s *PostgresStore) RenewRunLease(ctx context.Context, input domain.RenewRunLeaseInput) (domain.RunLeaseRecord, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.RunLeaseRecord{}, err
	}
	defer tx.Rollback(ctx)

	var status string
	if err := tx.QueryRow(ctx, `SELECT status FROM control_runs WHERE run_id = $1 FOR UPDATE`, input.RunID).Scan(&status); err != nil {
		return domain.RunLeaseRecord{}, mapPgError(err)
	}
	if isTerminalRunStatus(domain.RunStatus(status)) {
		return domain.RunLeaseRecord{}, ErrConflict
	}
	now := leaseNow(input.Now)
	existing, err := scanRunLease(tx.QueryRow(ctx, `
SELECT run_id, worker_id, lease_token, lease_expires_at, created_at, updated_at
FROM control_run_leases
WHERE run_id = $1
FOR UPDATE`, input.RunID))
	if err != nil {
		if errors.Is(err, ErrNotFound) {
			return domain.RunLeaseRecord{}, ErrConflict
		}
		return domain.RunLeaseRecord{}, err
	}
	if existing.LeaseToken != strings.TrimSpace(input.LeaseToken) || !existing.LeaseExpiresAt.After(now) {
		return domain.RunLeaseRecord{}, ErrConflict
	}
	lease, err := scanRunLease(tx.QueryRow(ctx, `
UPDATE control_run_leases
SET lease_expires_at = $3,
    updated_at = $4
WHERE run_id = $1 AND lease_token = $2
RETURNING run_id, worker_id, lease_token, lease_expires_at, created_at, updated_at`,
		input.RunID,
		strings.TrimSpace(input.LeaseToken),
		now.Add(positiveLeaseTTL(input.TTL)),
		now,
	))
	if err != nil {
		return domain.RunLeaseRecord{}, err
	}
	if _, err := tx.Exec(ctx, `UPDATE control_runs SET updated_at = $2 WHERE run_id = $1`, input.RunID, now); err != nil {
		return domain.RunLeaseRecord{}, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.RunLeaseRecord{}, err
	}
	return lease, nil
}

func (s *PostgresStore) ReleaseRunLease(ctx context.Context, input domain.ReleaseRunLeaseInput) error {
	tag, err := s.pool.Exec(ctx, `DELETE FROM control_run_leases WHERE run_id = $1 AND lease_token = $2`, input.RunID, strings.TrimSpace(input.LeaseToken))
	if err != nil {
		return mapPgError(err)
	}
	if tag.RowsAffected() > 0 {
		return nil
	}
	var exists bool
	if err := s.pool.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM control_runs WHERE run_id = $1)`, input.RunID).Scan(&exists); err != nil {
		return err
	}
	if !exists {
		return ErrNotFound
	}
	var activeToken string
	err = s.pool.QueryRow(ctx, `SELECT lease_token FROM control_run_leases WHERE run_id = $1`, input.RunID).Scan(&activeToken)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil
	}
	if err != nil {
		return mapPgError(err)
	}
	return ErrConflict
}

func (s *PostgresStore) ClearRunLease(ctx context.Context, runID string) (domain.RunLeaseRecord, bool, error) {
	lease, err := scanRunLease(s.pool.QueryRow(ctx, `
DELETE FROM control_run_leases
WHERE run_id = $1
RETURNING run_id, worker_id, lease_token, lease_expires_at, created_at, updated_at`, runID))
	if err == nil {
		return lease, true, nil
	}
	if !errors.Is(err, ErrNotFound) {
		return domain.RunLeaseRecord{}, false, err
	}
	var exists bool
	if err := s.pool.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM control_runs WHERE run_id = $1)`, runID).Scan(&exists); err != nil {
		return domain.RunLeaseRecord{}, false, mapPgError(err)
	}
	if !exists {
		return domain.RunLeaseRecord{}, false, ErrNotFound
	}
	return domain.RunLeaseRecord{}, false, nil
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
	eventID := input.EventID
	if eventID == "" {
		eventID = domain.NewID("event")
	}
	ts := input.TS
	if ts.IsZero() {
		ts = domain.Now()
	}
	row, err := q.AppendRunEvent(ctx, sqlc.AppendRunEventParams{
		EventID:        eventID,
		SequenceNumber: int64(sequence),
		RunID:          input.RunID,
		ThreadID:       nullableText(input.ThreadID),
		EventKind:      input.EventKind,
		EventType:      nullableText(input.EventType),
		NodeName:       nullableText(input.NodeName),
		TaskID:         nullableText(input.TaskID),
		CheckpointID:   nullableText(input.CheckpointID),
		ScopeID:        nullableText(input.ScopeID),
		AgentRole:      nullableText(input.AgentRole),
		Level:          nullableText(input.Level),
		Ts:             timestamptz(ts),
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

func (s *PostgresStore) GetRunEvent(ctx context.Context, eventID string) (domain.RunEventRecord, bool, error) {
	if eventID == "" {
		return domain.RunEventRecord{}, false, nil
	}
	row, err := s.queries.GetRunEvent(ctx, eventID)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return domain.RunEventRecord{}, false, nil
		}
		return domain.RunEventRecord{}, false, mapPgError(err)
	}
	return runEventFromRow(row), true, nil
}

func (s *PostgresStore) ListRunEvents(ctx context.Context, runID string, limit int) ([]domain.RunEventRecord, error) {
	rows, err := s.queries.ListRunEvents(ctx, sqlc.ListRunEventsParams{
		RunID: runID,
		Limit: limit32(limit, 500),
	})
	if err != nil {
		return nil, err
	}
	return runEventsFromRows(rows), nil
}

func (s *PostgresStore) ListRunEventsForUser(ctx context.Context, runID string, userID string, limit int) ([]domain.RunEventRecord, error) {
	rows, err := s.queries.ListRunEventsForUser(ctx, sqlc.ListRunEventsForUserParams{
		RunID:  runID,
		UserID: userID,
		Limit:  limit32(limit, 500),
	})
	if err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		if _, err := s.GetRunForUser(ctx, runID, userID); err != nil {
			return nil, err
		}
	}
	return runEventsFromRows(rows), nil
}

func (s *PostgresStore) ListRunEventsAfter(ctx context.Context, runID string, afterSequence int64, limit int) ([]domain.RunEventRecord, error) {
	rows, err := s.queries.ListRunEventsAfter(ctx, sqlc.ListRunEventsAfterParams{
		RunID:          runID,
		SequenceNumber: afterSequence,
		Limit:          limit32(limit, 500),
	})
	if err != nil {
		return nil, err
	}
	return runEventsFromRows(rows), nil
}

func (s *PostgresStore) ListRunEventsAfterForUser(ctx context.Context, runID string, userID string, afterSequence int64, limit int) ([]domain.RunEventRecord, error) {
	rows, err := s.queries.ListRunEventsAfterForUser(ctx, sqlc.ListRunEventsAfterForUserParams{
		RunID:          runID,
		UserID:         userID,
		SequenceNumber: afterSequence,
		Limit:          limit32(limit, 500),
	})
	if err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		if _, err := s.GetRunForUser(ctx, runID, userID); err != nil {
			return nil, err
		}
	}
	return runEventsFromRows(rows), nil
}

func runEventsFromRows(rows []sqlc.ControlRunEvent) []domain.RunEventRecord {
	events := make([]domain.RunEventRecord, 0, len(rows))
	for _, row := range rows {
		events = append(events, runEventFromRow(row))
	}
	return events
}

func (s *PostgresStore) CreateArtifact(ctx context.Context, input domain.CreateArtifactInput) (domain.ArtifactRecord, error) {
	artifactID := input.ArtifactID
	if artifactID == "" {
		artifactID = domain.NewID("artifact")
	}
	if input.ArtifactID != "" {
		existing, err := s.GetArtifact(ctx, input.ArtifactID)
		if err == nil {
			return existing, nil
		}
		if !errors.Is(err, ErrNotFound) {
			return domain.ArtifactRecord{}, err
		}
	}
	now := domain.Now()
	row, err := s.queries.CreateArtifact(ctx, sqlc.CreateArtifactParams{
		ArtifactID:    artifactID,
		RunID:         input.RunID,
		ThreadID:      nullableText(input.ThreadID),
		Kind:          input.Kind,
		Path:          nullableText(input.Path),
		SourcePath:    nullableText(input.SourcePath),
		PreviewPath:   nullableText(input.PreviewPath),
		Title:         nullableText(input.Title),
		ResultGroupID: nullableText(input.ResultGroupID),
		MimeType:      nullableText(input.MimeType),
		SizeBytes:     nullableInt8(input.SizeBytes),
		Sha256:        nullableText(input.SHA256),
		StorageUri:    nullableText(input.StorageURI),
		ToolName:      nullableText(input.ToolName),
		Category:      nullableText(input.Category),
		CreatedAt:     timestamptz(now),
		UpdatedAt:     timestamptz(now),
		Metadata:      jsonBytes(input.Metadata),
	})
	if err != nil {
		if input.ArtifactID != "" {
			if existing, getErr := s.GetArtifact(ctx, input.ArtifactID); getErr == nil {
				return existing, nil
			}
		}
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

func (s *PostgresStore) ListRunArtifactsForUser(ctx context.Context, runID string, userID string, limit int) ([]domain.ArtifactRecord, error) {
	if _, err := s.GetRunForUser(ctx, runID, userID); err != nil {
		return nil, err
	}
	rows, err := s.queries.ListRunArtifactsForUser(ctx, sqlc.ListRunArtifactsForUserParams{
		RunID:  runID,
		UserID: userID,
		Limit:  limit32(limit, 500),
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

func (s *PostgresStore) GetArtifactForUser(ctx context.Context, artifactID string, userID string) (domain.ArtifactRecord, error) {
	row, err := s.queries.GetArtifactForUser(ctx, sqlc.GetArtifactForUserParams{
		ArtifactID: artifactID,
		UserID:     userID,
	})
	if err != nil {
		return domain.ArtifactRecord{}, mapPgError(err)
	}
	return artifactFromRow(row), nil
}

func (s *PostgresStore) UpsertResource(ctx context.Context, input domain.UpsertResourceInput) (domain.ResourceRecord, error) {
	resourceID := strings.TrimSpace(input.ResourceID)
	if resourceID == "" {
		resourceID = domain.NewID("file")
	}
	ownerUserID := strings.TrimSpace(input.OwnerUserID)
	if ownerUserID == "" {
		ownerUserID = "local-user"
	}
	sourceType := strings.TrimSpace(input.SourceType)
	if sourceType == "" {
		sourceType = "upload"
	}
	resourceKind := strings.TrimSpace(input.ResourceKind)
	if resourceKind == "" {
		resourceKind = "file"
	}
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	now := domain.Now()
	createdAt := input.CreatedAt
	if createdAt.IsZero() {
		createdAt = now
	}
	updatedAt := input.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = now
	}
	row, err := s.queries.UpsertResource(ctx, sqlc.UpsertResourceParams{
		ResourceID:         resourceID,
		OwnerUserID:        ownerUserID,
		OwnerOrgID:         nullableText(input.OwnerOrgID),
		OwnerRole:          nullableText(input.OwnerRole),
		OriginalName:       strings.TrimSpace(input.OriginalName),
		ContentType:        nullableText(input.ContentType),
		SizeBytes:          input.SizeBytes,
		Sha256:             nullableText(input.SHA256),
		StorageUri:         nullableText(input.StorageURI),
		StoragePath:        nullableText(input.StoragePath),
		SourceType:         sourceType,
		ResourceKind:       resourceKind,
		SourceUri:          nullableText(input.SourceURI),
		ProjectID:          nullableText(input.ProjectID),
		Status:             status,
		CreatedAt:          timestamptz(createdAt),
		UpdatedAt:          timestamptz(updatedAt),
		DeletedAt:          nullableTimestamptz(input.DeletedAt),
		RetentionExpiresAt: nullableTimestamptz(input.RetentionExpiresAt),
		Metadata:           jsonBytes(input.Metadata),
	})
	if err != nil {
		return domain.ResourceRecord{}, mapPgError(err)
	}
	return resourceFromRow(row), nil
}

func (s *PostgresStore) GetResourceForUser(ctx context.Context, resourceID string, userID string, orgID string) (domain.ResourceRecord, error) {
	row, err := s.queries.GetResourceForUser(ctx, sqlc.GetResourceForUserParams{
		ResourceID:  strings.TrimSpace(resourceID),
		OwnerUserID: strings.TrimSpace(userID),
		OwnerOrgID:  nullableText(orgID),
	})
	if err != nil {
		return domain.ResourceRecord{}, mapPgError(err)
	}
	return resourceFromRow(row), nil
}

func (s *PostgresStore) ListResourcesForUser(ctx context.Context, input domain.ResourceListInput) (domain.ResourceListPage, error) {
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	params := sqlc.ListResourcesForUserParams{
		OwnerUserID: strings.TrimSpace(input.UserID),
		OwnerOrgID:  nullableText(input.OrgID),
		Status:      status,
		Column4:     strings.TrimSpace(input.Kind),
		Column5:     strings.TrimSpace(input.Source),
		Column6:     strings.TrimSpace(input.ProjectID),
		Column7:     strings.TrimSpace(input.Query),
		Limit:       limit32(input.Limit, 200),
		Offset:      offset32(input.Offset),
	}
	rows, err := s.queries.ListResourcesForUser(ctx, params)
	if err != nil {
		return domain.ResourceListPage{}, err
	}
	count, err := s.queries.CountResourcesForUser(ctx, sqlc.CountResourcesForUserParams{
		OwnerUserID: params.OwnerUserID,
		OwnerOrgID:  params.OwnerOrgID,
		Status:      params.Status,
		Column4:     params.Column4,
		Column5:     params.Column5,
		Column6:     params.Column6,
		Column7:     params.Column7,
	})
	if err != nil {
		return domain.ResourceListPage{}, err
	}
	resources := make([]domain.ResourceRecord, 0, len(rows))
	for _, row := range rows {
		resources = append(resources, resourceFromRow(row))
	}
	return domain.ResourceListPage{
		Resources:  resources,
		TotalCount: int(count),
		Limit:      int(params.Limit),
		Offset:     int(params.Offset),
	}, nil
}

func (s *PostgresStore) ListResources(ctx context.Context, limit int, offset int) ([]domain.ResourceRecord, error) {
	rows, err := s.queries.ListResources(ctx, sqlc.ListResourcesParams{
		Limit:  limit32(limit, 1000),
		Offset: offset32(offset),
	})
	if err != nil {
		return nil, err
	}
	resources := make([]domain.ResourceRecord, 0, len(rows))
	for _, row := range rows {
		resources = append(resources, resourceFromRow(row))
	}
	return resources, nil
}

func (s *PostgresStore) SoftDeleteResourceForUser(ctx context.Context, resourceID string, userID string, orgID string, deletedAt time.Time) (domain.ResourceRecord, error) {
	if deletedAt.IsZero() {
		deletedAt = domain.Now()
	}
	row, err := s.queries.SoftDeleteResourceForUser(ctx, sqlc.SoftDeleteResourceForUserParams{
		ResourceID:         strings.TrimSpace(resourceID),
		OwnerUserID:        strings.TrimSpace(userID),
		OwnerOrgID:         nullableText(orgID),
		DeletedAt:          timestamptz(deletedAt),
		RetentionExpiresAt: timestamptz(deletedAt.UTC().Add(defaultResourceRetention)),
	})
	if err != nil {
		return domain.ResourceRecord{}, mapPgError(err)
	}
	return resourceFromRow(row), nil
}

func (s *PostgresStore) RestoreResourceForUser(ctx context.Context, resourceID string, userID string, orgID string, restoredAt time.Time) (domain.ResourceRecord, error) {
	if restoredAt.IsZero() {
		restoredAt = domain.Now()
	}
	row, err := s.queries.RestoreResourceForUser(ctx, sqlc.RestoreResourceForUserParams{
		ResourceID:  strings.TrimSpace(resourceID),
		OwnerUserID: strings.TrimSpace(userID),
		OwnerOrgID:  nullableText(orgID),
		UpdatedAt:   timestamptz(restoredAt),
	})
	if err != nil {
		return domain.ResourceRecord{}, mapPgError(err)
	}
	return resourceFromRow(row), nil
}

func (s *PostgresStore) ResourceStorageStats(ctx context.Context) (domain.ResourceStorageStats, error) {
	row, err := s.queries.ResourceStorageStats(ctx)
	if err != nil {
		return domain.ResourceStorageStats{}, err
	}
	return domain.ResourceStorageStats{
		TotalResources: int(row.TotalResources),
		TotalBytes:     row.TotalBytes,
	}, nil
}

func (s *PostgresStore) CreateResourceEvent(ctx context.Context, input domain.AppendResourceEventInput) (domain.ResourceEventRecord, error) {
	eventID := strings.TrimSpace(input.EventID)
	if eventID == "" {
		eventID = domain.NewID("resource_event")
	}
	ts := input.TS
	if ts.IsZero() {
		ts = domain.Now()
	}
	row, err := s.queries.CreateResourceEvent(ctx, sqlc.CreateResourceEventParams{
		EventID:     eventID,
		ResourceID:  strings.TrimSpace(input.ResourceID),
		ActorUserID: nullableText(input.ActorUserID),
		ActorOrgID:  nullableText(input.ActorOrgID),
		EventType:   strings.TrimSpace(input.EventType),
		Ts:          timestamptz(ts),
		Metadata:    jsonBytes(input.Metadata),
	})
	if err != nil {
		return domain.ResourceEventRecord{}, mapPgError(err)
	}
	return resourceEventFromRow(row), nil
}

func (s *PostgresStore) ListResourceEvents(ctx context.Context, resourceID string, limit int) ([]domain.ResourceEventRecord, error) {
	rows, err := s.queries.ListResourceEvents(ctx, sqlc.ListResourceEventsParams{
		ResourceID: strings.TrimSpace(resourceID),
		Limit:      limit32(limit, 200),
	})
	if err != nil {
		return nil, mapPgError(err)
	}
	events := make([]domain.ResourceEventRecord, 0, len(rows))
	for _, row := range rows {
		events = append(events, resourceEventFromRow(row))
	}
	return events, nil
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

func resourceFromRow(row sqlc.ControlResource) domain.ResourceRecord {
	return domain.ResourceRecord{
		ResourceID:         row.ResourceID,
		OriginalName:       row.OriginalName,
		ContentType:        textValue(row.ContentType),
		SizeBytes:          row.SizeBytes,
		SHA256:             textValue(row.Sha256),
		StorageURI:         textValue(row.StorageUri),
		StoragePath:        textValue(row.StoragePath),
		SourceType:         row.SourceType,
		ResourceKind:       row.ResourceKind,
		SourceURI:          textValue(row.SourceUri),
		ProjectID:          textValue(row.ProjectID),
		OwnerUserID:        row.OwnerUserID,
		OwnerOrgID:         textValue(row.OwnerOrgID),
		OwnerRole:          textValue(row.OwnerRole),
		Status:             row.Status,
		CreatedAt:          timeValue(row.CreatedAt),
		UpdatedAt:          timeValue(row.UpdatedAt),
		DeletedAt:          timeValue(row.DeletedAt),
		RetentionExpiresAt: timeValue(row.RetentionExpiresAt),
		Metadata:           jsonMap(row.Metadata),
	}
}

func resourceEventFromRow(row sqlc.ControlResourceEvent) domain.ResourceEventRecord {
	return domain.ResourceEventRecord{
		EventID:     row.EventID,
		ResourceID:  row.ResourceID,
		ActorUserID: textValue(row.ActorUserID),
		ActorOrgID:  textValue(row.ActorOrgID),
		EventType:   row.EventType,
		TS:          timeValue(row.Ts),
		Metadata:    jsonMap(row.Metadata),
	}
}

type scanner interface {
	Scan(dest ...any) error
}

func scanOrganization(row scanner) (domain.Organization, error) {
	var org domain.Organization
	var metadata []byte
	if err := row.Scan(
		&org.OrgID,
		&org.Name,
		&org.Status,
		&org.CreatedAt,
		&org.UpdatedAt,
		&metadata,
	); err != nil {
		return domain.Organization{}, mapPgError(err)
	}
	org.CreatedAt = org.CreatedAt.UTC()
	org.UpdatedAt = org.UpdatedAt.UTC()
	org.Metadata = jsonMap(metadata)
	return org, nil
}

func scanUserAccount(row scanner) (domain.UserAccount, error) {
	var user domain.UserAccount
	var metadata []byte
	if err := row.Scan(
		&user.UserID,
		&user.Email,
		&user.DisplayName,
		&user.Role,
		&user.Status,
		&user.OrgID,
		&user.CreatedAt,
		&user.UpdatedAt,
		&metadata,
	); err != nil {
		return domain.UserAccount{}, mapPgError(err)
	}
	user.CreatedAt = user.CreatedAt.UTC()
	user.UpdatedAt = user.UpdatedAt.UTC()
	user.Metadata = jsonMap(metadata)
	return user, nil
}

func scanBisqueCredential(row scanner) (domain.BisqueCredentialRecord, error) {
	var record domain.BisqueCredentialRecord
	var lastVerifiedAt pgtype.Timestamptz
	var metadata []byte
	if err := row.Scan(
		&record.SessionID,
		&record.UserID,
		&record.OrgID,
		&record.RootURL,
		&record.Username,
		&record.PasswordCiphertext,
		&record.PasswordNonce,
		&record.PasswordKeyID,
		&record.PasswordAlgorithm,
		&record.Status,
		&lastVerifiedAt,
		&record.CreatedAt,
		&record.UpdatedAt,
		&metadata,
	); err != nil {
		return domain.BisqueCredentialRecord{}, mapPgError(err)
	}
	if lastVerifiedAt.Valid {
		record.LastVerifiedAt = lastVerifiedAt.Time.UTC()
	}
	record.CreatedAt = record.CreatedAt.UTC()
	record.UpdatedAt = record.UpdatedAt.UTC()
	record.Metadata = jsonMap(metadata)
	return record, nil
}

func scanRunLease(row scanner) (domain.RunLeaseRecord, error) {
	var lease domain.RunLeaseRecord
	if err := row.Scan(
		&lease.RunID,
		&lease.WorkerID,
		&lease.LeaseToken,
		&lease.LeaseExpiresAt,
		&lease.CreatedAt,
		&lease.UpdatedAt,
	); err != nil {
		return domain.RunLeaseRecord{}, mapPgError(err)
	}
	lease.LeaseExpiresAt = lease.LeaseExpiresAt.UTC()
	lease.CreatedAt = lease.CreatedAt.UTC()
	lease.UpdatedAt = lease.UpdatedAt.UTC()
	return lease, nil
}

func scanWorkerHeartbeat(row scanner) (domain.WorkerHeartbeatRecord, error) {
	var worker domain.WorkerHeartbeatRecord
	var metadata []byte
	if err := row.Scan(
		&worker.WorkerID,
		&worker.WorkerKind,
		&worker.Status,
		&worker.CurrentRunID,
		&worker.Hostname,
		&worker.Version,
		&worker.StartedAt,
		&worker.LastHeartbeatAt,
		&worker.UpdatedAt,
		&metadata,
	); err != nil {
		return domain.WorkerHeartbeatRecord{}, mapPgError(err)
	}
	worker.StartedAt = worker.StartedAt.UTC()
	worker.LastHeartbeatAt = worker.LastHeartbeatAt.UTC()
	worker.UpdatedAt = worker.UpdatedAt.UTC()
	worker.Metadata = jsonMap(metadata)
	return worker, nil
}

func mapPgError(err error) error {
	if errors.Is(err, pgx.ErrNoRows) {
		return ErrNotFound
	}
	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) && pgErr.Code == "23505" {
		return ErrConflict
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

func nullableTimestamptz(value time.Time) pgtype.Timestamptz {
	if value.IsZero() {
		return pgtype.Timestamptz{}
	}
	return timestamptz(value)
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

func offset32(offset int) int32 {
	if offset <= 0 {
		return 0
	}
	return int32(offset)
}
