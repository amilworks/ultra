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
	if _, err := db.Exec(ctx, postgresControlSchemaSQL); err != nil {
		return fmt.Errorf("apply postgres schema: %w", err)
	}
	return nil
}
