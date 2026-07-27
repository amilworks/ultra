package store

import (
	"context"
	"errors"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

const runSteerMessageColumns = `steer_id, run_id, thread_id, user_id, message_id, content, status, created_at, applied_at, updated_at`

func scanRunSteerMessage(row pgx.Row) (domain.RunSteerMessageRecord, error) {
	var record domain.RunSteerMessageRecord
	var appliedAt *time.Time
	if err := row.Scan(
		&record.SteerID,
		&record.RunID,
		&record.ThreadID,
		&record.UserID,
		&record.MessageID,
		&record.Content,
		&record.Status,
		&record.CreatedAt,
		&appliedAt,
		&record.UpdatedAt,
	); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return domain.RunSteerMessageRecord{}, ErrNotFound
		}
		return domain.RunSteerMessageRecord{}, err
	}
	record.AppliedAt = appliedAt
	return record, nil
}

// CreateRunSteerMessage accepts a steer in one transaction: it locks the run
// row (serializing against CloseRunSteerBarrier so acceptance and barrier
// close cannot interleave into a lost steer), verifies the run is live and
// the barrier open, then inserts the steer row AND its transcript message
// together — recovery requeue reseeds jobs from the transcript, so the two
// must never be observable apart. Retrying an existing steer_id returns the
// stored row.
func (s *PostgresStore) CreateRunSteerMessage(
	ctx context.Context,
	input domain.CreateRunSteerMessageInput,
) (domain.RunSteerMessageRecord, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.RunSteerMessageRecord{}, err
	}
	defer tx.Rollback(ctx)

	var status string
	if err := tx.QueryRow(ctx,
		`SELECT status FROM control_runs WHERE run_id = $1 FOR UPDATE`,
		input.RunID,
	).Scan(&status); err != nil {
		return domain.RunSteerMessageRecord{}, mapPgError(err)
	}
	// Idempotency lookup is scoped to THIS run: steer_id is client-supplied,
	// and an unscoped match would return (and 200-acknowledge) another run's —
	// potentially another user's — steer record. A cross-run collision falls
	// through to the INSERT below, whose global primary key rejects it with a
	// conflict instead of a disclosure.
	existing, err := scanRunSteerMessage(tx.QueryRow(ctx,
		`SELECT `+runSteerMessageColumns+` FROM control_run_steer_messages WHERE steer_id = $1 AND run_id = $2`,
		input.SteerID,
		input.RunID,
	))
	if err == nil {
		return existing, tx.Commit(ctx)
	}
	if !errors.Is(err, ErrNotFound) {
		return domain.RunSteerMessageRecord{}, err
	}
	if isTerminalRunStatus(domain.RunStatus(status)) {
		return domain.RunSteerMessageRecord{}, ErrSteeringClosed
	}
	var barrierClosed bool
	if err := tx.QueryRow(ctx,
		`SELECT EXISTS (SELECT 1 FROM control_run_steer_barriers WHERE run_id = $1)`,
		input.RunID,
	).Scan(&barrierClosed); err != nil {
		return domain.RunSteerMessageRecord{}, mapPgError(err)
	}
	if barrierClosed {
		return domain.RunSteerMessageRecord{}, ErrSteeringClosed
	}
	var steerCount int
	if err := tx.QueryRow(ctx,
		`SELECT COUNT(*) FROM control_run_steer_messages WHERE run_id = $1`,
		input.RunID,
	).Scan(&steerCount); err != nil {
		return domain.RunSteerMessageRecord{}, mapPgError(err)
	}
	if steerCount >= maxSteerMessagesPerRun {
		return domain.RunSteerMessageRecord{}, ErrSteeringClosed
	}
	now := input.CreatedAt
	if now.IsZero() {
		now = domain.Now()
	}
	record, err := scanRunSteerMessage(tx.QueryRow(ctx, `
INSERT INTO control_run_steer_messages
  (steer_id, run_id, thread_id, user_id, message_id, content, status, created_at, updated_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $8)
RETURNING `+runSteerMessageColumns,
		input.SteerID,
		input.RunID,
		input.ThreadID,
		input.UserID,
		input.MessageID,
		input.Content,
		domain.RunSteerStatusPending,
		now,
	))
	if err != nil {
		return domain.RunSteerMessageRecord{}, mapPgError(err)
	}
	if _, err := tx.Exec(ctx, `
INSERT INTO control_thread_messages (message_id, thread_id, role, content, created_at, metadata, run_id)
VALUES ($1, $2, 'user', $3, $4, $5, $6)
ON CONFLICT (message_id) DO NOTHING`,
		input.MessageID,
		input.ThreadID,
		input.Content,
		now,
		jsonBytes(domain.JSONMap{"kind": "steering", "steer_id": input.SteerID}),
		input.RunID,
	); err != nil {
		return domain.RunSteerMessageRecord{}, mapPgError(err)
	}
	return record, tx.Commit(ctx)
}

func (s *PostgresStore) ListRunSteerMessages(
	ctx context.Context,
	runID string,
) ([]domain.RunSteerMessageRecord, error) {
	var exists bool
	if err := s.pool.QueryRow(ctx,
		`SELECT EXISTS (SELECT 1 FROM control_runs WHERE run_id = $1)`, runID,
	).Scan(&exists); err != nil {
		return nil, mapPgError(err)
	}
	if !exists {
		return nil, ErrNotFound
	}
	rows, err := s.pool.Query(ctx,
		`SELECT `+runSteerMessageColumns+`
FROM control_run_steer_messages WHERE run_id = $1 ORDER BY created_at, steer_id`,
		runID,
	)
	if err != nil {
		return nil, mapPgError(err)
	}
	defer rows.Close()
	var records []domain.RunSteerMessageRecord
	for rows.Next() {
		record, err := scanRunSteerMessage(rows)
		if err != nil {
			return nil, err
		}
		records = append(records, record)
	}
	return records, rows.Err()
}

func (s *PostgresStore) MarkRunSteerMessageApplied(
	ctx context.Context,
	runID string,
	steerID string,
	appliedAt time.Time,
) (domain.RunSteerMessageRecord, bool, error) {
	if appliedAt.IsZero() {
		appliedAt = domain.Now()
	}
	record, err := scanRunSteerMessage(s.pool.QueryRow(ctx, `
UPDATE control_run_steer_messages
SET status = $3, applied_at = $4, updated_at = $4
WHERE run_id = $1 AND steer_id = $2 AND status = $5
RETURNING `+runSteerMessageColumns,
		runID, steerID, domain.RunSteerStatusApplied, appliedAt, domain.RunSteerStatusPending,
	))
	if err == nil {
		return record, true, nil
	}
	if !errors.Is(err, ErrNotFound) {
		return domain.RunSteerMessageRecord{}, false, err
	}
	record, err = scanRunSteerMessage(s.pool.QueryRow(ctx,
		`SELECT `+runSteerMessageColumns+`
FROM control_run_steer_messages WHERE run_id = $1 AND steer_id = $2`,
		runID, steerID,
	))
	if err != nil {
		return domain.RunSteerMessageRecord{}, false, err
	}
	return record, false, nil
}

// CloseRunSteerBarrier locks the run row (the same lock CreateRunSteerMessage
// takes), closes the barrier idempotently, and returns still-pending steers.
// Any steer accepted before this call is in the returned list; any accept
// after it fails with ErrSteeringClosed — no steer can fall between.
func (s *PostgresStore) CloseRunSteerBarrier(
	ctx context.Context,
	runID string,
	closedAt time.Time,
) ([]domain.RunSteerMessageRecord, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback(ctx)

	var status string
	if err := tx.QueryRow(ctx,
		`SELECT status FROM control_runs WHERE run_id = $1 FOR UPDATE`, runID,
	).Scan(&status); err != nil {
		return nil, mapPgError(err)
	}
	if closedAt.IsZero() {
		closedAt = domain.Now()
	}
	if _, err := tx.Exec(ctx, `
INSERT INTO control_run_steer_barriers (run_id, closed_at)
VALUES ($1, $2)
ON CONFLICT (run_id) DO NOTHING`,
		runID, closedAt,
	); err != nil {
		return nil, mapPgError(err)
	}
	rows, err := tx.Query(ctx,
		`SELECT `+runSteerMessageColumns+`
FROM control_run_steer_messages
WHERE run_id = $1 AND status = $2
ORDER BY created_at, steer_id`,
		runID, domain.RunSteerStatusPending,
	)
	if err != nil {
		return nil, mapPgError(err)
	}
	defer rows.Close()
	var pending []domain.RunSteerMessageRecord
	for rows.Next() {
		record, err := scanRunSteerMessage(rows)
		if err != nil {
			return nil, err
		}
		pending = append(pending, record)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return pending, tx.Commit(ctx)
}

func (s *PostgresStore) OpenRunSteerBarrier(ctx context.Context, runID string) error {
	if _, err := s.pool.Exec(ctx,
		`DELETE FROM control_run_steer_barriers WHERE run_id = $1`, runID,
	); err != nil {
		return mapPgError(err)
	}
	return nil
}

func (s *PostgresStore) MarkPendingRunSteerMessagesMissed(
	ctx context.Context,
	runID string,
	at time.Time,
) ([]domain.RunSteerMessageRecord, error) {
	if at.IsZero() {
		at = domain.Now()
	}
	rows, err := s.pool.Query(ctx, `
UPDATE control_run_steer_messages
SET status = $3, updated_at = $4
WHERE run_id = $1 AND status = $2
RETURNING `+runSteerMessageColumns,
		runID, domain.RunSteerStatusPending, domain.RunSteerStatusMissed, at,
	)
	if err != nil {
		return nil, mapPgError(err)
	}
	defer rows.Close()
	var changed []domain.RunSteerMessageRecord
	for rows.Next() {
		record, err := scanRunSteerMessage(rows)
		if err != nil {
			return nil, err
		}
		changed = append(changed, record)
	}
	return changed, rows.Err()
}
