package store

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5/pgconn"
)

func TestApplyPostgresSchemaExecutesCurrentControlSchema(t *testing.T) {
	t.Parallel()

	execer := &fakeSchemaExecer{}
	if err := ApplyPostgresSchema(context.Background(), execer); err != nil {
		t.Fatalf("ApplyPostgresSchema() error = %v, want nil", err)
	}
	if execer.sql == "" {
		t.Fatalf("ApplyPostgresSchema() did not execute SQL")
	}
	for _, table := range requiredPostgresControlTables {
		if !strings.Contains(execer.sql, "CREATE TABLE IF NOT EXISTS "+table) {
			t.Fatalf("ApplyPostgresSchema() SQL missing table %s", table)
		}
	}
	if !strings.Contains(execer.sql, "control_runs_idempotency_unique_idx") {
		t.Fatalf("ApplyPostgresSchema() SQL missing control_runs idempotency unique index")
	}
}

func TestApplyPostgresSchemaDropsRedundantHotPathIndexes(t *testing.T) {
	t.Parallel()

	execer := &fakeSchemaExecer{}
	if err := ApplyPostgresSchema(context.Background(), execer); err != nil {
		t.Fatalf("ApplyPostgresSchema() error = %v, want nil", err)
	}
	for _, indexName := range []string{
		"control_run_events_run_sequence_idx",
		"control_run_events_run_event_idx",
		"control_data_agent_job_events_job_sequence_idx",
	} {
		if !strings.Contains(execer.sql, "DROP INDEX IF EXISTS "+indexName) {
			t.Fatalf("ApplyPostgresSchema() SQL missing redundant index drop for %s", indexName)
		}
		if strings.Contains(execer.sql, "CREATE INDEX IF NOT EXISTS "+indexName) {
			t.Fatalf("ApplyPostgresSchema() SQL recreates redundant index %s", indexName)
		}
	}
}

func TestApplyPostgresSchemaWrapsExecutionErrors(t *testing.T) {
	t.Parallel()

	execer := &fakeSchemaExecer{err: errors.New("permission denied")}
	err := ApplyPostgresSchema(context.Background(), execer)
	if err == nil {
		t.Fatalf("ApplyPostgresSchema() error = nil, want execution error")
	}
	if !strings.Contains(err.Error(), "apply postgres schema") {
		t.Fatalf("ApplyPostgresSchema() error = %q, want migration context", err)
	}
}

type fakeSchemaExecer struct {
	sql string
	err error
}

func (e *fakeSchemaExecer) Exec(ctx context.Context, sql string, args ...any) (pgconn.CommandTag, error) {
	_ = ctx
	_ = args
	e.sql = sql
	return pgconn.CommandTag{}, e.err
}
