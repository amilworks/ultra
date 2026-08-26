package store

import (
	"context"
	"fmt"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

// TestApplyPostgresSchemaOnFreshDatabase applies the consolidated schema.sql
// to a brand-new, empty database. The routine store Postgres tests reuse an
// already-migrated database, which can hide ordering defects between dependent
// DDL statements. Only a truly fresh apply exercises the statement order end
// to end.
func TestApplyPostgresSchemaOnFreshDatabase(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if strings.TrimSpace(dsn) == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	admin, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New(admin): %v", err)
	}
	defer admin.Close()

	name := fmt.Sprintf("ultra_fresh_apply_%d", time.Now().UnixNano())
	if _, err := admin.Exec(ctx, "CREATE DATABASE "+name); err != nil {
		t.Fatalf("create fresh database: %v", err)
	}
	defer func() {
		dropCtx, dropCancel := context.WithTimeout(context.Background(), time.Minute)
		defer dropCancel()
		if _, err := admin.Exec(dropCtx, "DROP DATABASE IF EXISTS "+name+" WITH (FORCE)"); err != nil {
			t.Errorf("drop fresh database %s: %v", name, err)
		}
	}()

	parsed, err := url.Parse(dsn)
	if err != nil {
		t.Fatalf("parse test DSN: %v", err)
	}
	parsed.Path = "/" + name
	fresh, err := pgxpool.New(ctx, parsed.String())
	if err != nil {
		t.Fatalf("pgxpool.New(fresh): %v", err)
	}
	defer fresh.Close()

	if err := ApplyPostgresSchema(ctx, fresh); err != nil {
		t.Fatalf("ApplyPostgresSchema on a fresh database: %v", err)
	}
	// Applying again must stay idempotent: the reapply bridges in schema.sql
	// are exercised with the schema now fully present.
	if err := ApplyPostgresSchema(ctx, fresh); err != nil {
		t.Fatalf("ApplyPostgresSchema reapply: %v", err)
	}

	// Hold open the same RowExclusiveLock a worker checkpoint upsert takes. A
	// steady-state schema reapply must inspect the already-correct index and
	// return without executing CREATE INDEX, whose ShareLock would otherwise
	// wait for this transaction until the apply context expires.
	if _, err := fresh.Exec(ctx, `
INSERT INTO control_threads (
  thread_id, user_id, title, status, created_at, updated_at, metadata
) VALUES (
  'thread-live-checkpoint', 'user-live-checkpoint', 'Live checkpoint lock test',
  'active', now(), now(), '{}'::jsonb
);
INSERT INTO control_runs (
  run_id, thread_id, user_id, goal, status, workflow_kind,
  created_at, updated_at, metadata
) VALUES (
  'run-live-checkpoint', 'thread-live-checkpoint', 'user-live-checkpoint',
  'Exercise checkpoint schema locking', 'running', 'deep_agents',
  now(), now(), '{}'::jsonb
)`); err != nil {
		t.Fatalf("seed live checkpoint run: %v", err)
	}
	checkpointWriter, err := fresh.Begin(ctx)
	if err != nil {
		t.Fatalf("begin live checkpoint writer: %v", err)
	}
	defer func() { _ = checkpointWriter.Rollback(context.Background()) }()
	if _, err := checkpointWriter.Exec(ctx, `
INSERT INTO deepagents_checkpoint_threads (thread_id, state, updated_at)
VALUES ('run-live-checkpoint', '\x01'::bytea, now())
ON CONFLICT (thread_id) DO UPDATE SET
  state = EXCLUDED.state,
  updated_at = EXCLUDED.updated_at`); err != nil {
		t.Fatalf("hold live checkpoint upsert: %v", err)
	}
	steadyStateCtx, steadyStateCancel := context.WithTimeout(ctx, 10*time.Second)
	if err := ApplyPostgresSchema(steadyStateCtx, fresh); err != nil {
		steadyStateCancel()
		t.Fatalf("steady-state schema reapply blocked behind live checkpoint writer: %v", err)
	}
	steadyStateCancel()
	if err := checkpointWriter.Rollback(ctx); err != nil {
		t.Fatalf("rollback live checkpoint writer: %v", err)
	}

	// Recreate the exact legacy worker-owned shape: valid columns, no run FK,
	// a mismatched same-name index, and an orphan left by a deleted run. A
	// rolling upgrade must repair the index, erase that orphan, and install the
	// cascade before a worker can write another row.
	if _, err := fresh.Exec(ctx, `
DROP TABLE deepagents_checkpoint_threads;
CREATE TABLE deepagents_checkpoint_threads (
  thread_id text PRIMARY KEY,
  state bytea NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX deepagents_checkpoint_threads_updated_at_idx
  ON deepagents_checkpoint_threads(thread_id);
INSERT INTO deepagents_checkpoint_threads (thread_id, state)
VALUES ('deleted-run', '\x01'::bytea)`); err != nil {
		t.Fatalf("seed legacy checkpoint table: %v", err)
	}
	if err := ApplyPostgresSchema(ctx, fresh); err != nil {
		t.Fatalf("ApplyPostgresSchema legacy checkpoint upgrade: %v", err)
	}
	var orphanCount int
	if err := fresh.QueryRow(ctx, `
SELECT COUNT(*) FROM deepagents_checkpoint_threads WHERE thread_id = 'deleted-run'`).Scan(&orphanCount); err != nil {
		t.Fatalf("count upgraded checkpoint orphans: %v", err)
	}
	if orphanCount != 0 {
		t.Fatalf("legacy checkpoint upgrade retained %d orphan rows", orphanCount)
	}
	var cascadeFKs int
	if err := fresh.QueryRow(ctx, `
SELECT COUNT(*)
FROM pg_constraint constraint_row
WHERE constraint_row.conrelid = 'deepagents_checkpoint_threads'::regclass
  AND constraint_row.confrelid = 'control_runs'::regclass
  AND constraint_row.contype = 'f'
  AND constraint_row.confdeltype = 'c'`).Scan(&cascadeFKs); err != nil {
		t.Fatalf("inspect checkpoint run cascade: %v", err)
	}
	if cascadeFKs != 1 {
		t.Fatalf("checkpoint run cascade count = %d, want 1", cascadeFKs)
	}
	var checkpointIndexDefinition string
	if err := fresh.QueryRow(ctx, `
SELECT pg_get_indexdef('deepagents_checkpoint_threads_updated_at_idx'::regclass)`).Scan(&checkpointIndexDefinition); err != nil {
		t.Fatalf("inspect reconciled checkpoint index: %v", err)
	}
	if !strings.Contains(checkpointIndexDefinition, "USING btree (updated_at)") {
		t.Fatalf("checkpoint index was not reconciled: %s", checkpointIndexDefinition)
	}
}
