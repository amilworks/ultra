package store

import (
	"context"
	_ "embed"
	"fmt"

	"github.com/jackc/pgx/v5/pgconn"
)

//go:embed schema.sql
var postgresControlSchemaSQL string

type schemaExecer interface {
	Exec(ctx context.Context, sql string, arguments ...any) (pgconn.CommandTag, error)
}

func ApplyPostgresSchema(ctx context.Context, db schemaExecer) error {
	// The whole schema runs as one implicit transaction. Concurrent appliers
	// (rolling deploys running `migrate` from several replicas, or parallel
	// test packages) would otherwise interleave DDL lock acquisition and can
	// deadlock; the advisory lock serializes them.
	script := "SELECT pg_advisory_xact_lock(hashtext('ultra_control_schema_apply')::bigint);\n" + postgresControlSchemaSQL
	if _, err := db.Exec(ctx, script); err != nil {
		return fmt.Errorf("apply postgres schema: %w", err)
	}
	return nil
}
