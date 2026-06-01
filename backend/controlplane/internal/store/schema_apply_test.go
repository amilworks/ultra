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
