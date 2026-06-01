package store

import (
	"context"
	"fmt"
	"slices"
	"strings"

	"github.com/jackc/pgx/v5"
)

type schemaQuerier interface {
	QueryRow(ctx context.Context, sql string, args ...any) pgx.Row
}

var requiredPostgresControlTables = []string{
	"control_threads",
	"control_organizations",
	"control_users",
	"control_thread_messages",
	"control_runs",
	"control_run_events",
	"control_run_leases",
	"control_worker_heartbeats",
	"control_artifacts",
}

func VerifyPostgresSchema(ctx context.Context, db schemaQuerier) error {
	var presentTables []string
	err := db.QueryRow(ctx, `
SELECT COALESCE(array_agg(table_name::text ORDER BY table_name), ARRAY[]::text[])
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name = ANY($1::text[])
`, requiredPostgresControlTables).Scan(&presentTables)
	if err != nil {
		return fmt.Errorf("verify postgres schema: %w", err)
	}

	present := map[string]struct{}{}
	for _, table := range presentTables {
		present[table] = struct{}{}
	}
	missing := make([]string, 0)
	for _, table := range requiredPostgresControlTables {
		if _, ok := present[table]; !ok {
			missing = append(missing, table)
		}
	}
	if len(missing) > 0 {
		slices.Sort(missing)
		return fmt.Errorf("postgres control schema is not ready; apply migrations before starting: missing tables %s", strings.Join(missing, ", "))
	}
	return nil
}
