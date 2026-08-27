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
	"control_user_token_usage_daily",
	"control_user_token_usage_lifetime",
	"control_run_token_usage",
	"control_run_token_usage_finalized",
	"control_thread_messages",
	"control_runs",
	"deepagents_checkpoint_threads",
	"control_run_events",
	"control_run_event_sequences",
	"control_run_leases",
	"control_notes",
	"control_note_read_grants",
	"control_note_run_usage",
	"control_note_append_proposals",
	"control_note_append_operations",
	"control_run_steer_messages",
	"control_run_steer_barriers",
	"control_worker_heartbeats",
	"control_artifacts",
	"control_resources",
	"control_resource_purge_tombstones",
	"control_resource_search_documents",
	"control_resource_search_facts",
	"control_resource_events",
	"control_resource_share_grants",
	"control_resource_collections",
	"control_resource_collection_share_grants",
	"control_resource_collection_members",
	"control_dataset_snapshots",
	"control_dataset_snapshot_resources",
	"control_dataset_snapshot_share_grants",
	"control_dataset_snapshot_events",
	"control_data_agent_jobs",
	"control_data_agent_job_resources",
	"control_data_agent_job_events",
	"control_data_agent_job_leases",
	"control_upload_sessions",
	"control_upload_session_files",
	"control_upload_session_events",
	"control_upload_chunks",
	"control_bisque_credentials",
	"control_training_models",
	"control_training_domains",
	"control_training_lineages",
	"control_training_model_versions",
	"control_training_model_status",
	"control_training_gate_policies",
	"control_training_guardrail_clauses",
	"control_training_gate_config_events",
	"control_training_gold_sets",
	"control_training_gold_items",
	"control_training_replay_pool",
	"control_training_benchmark_runs",
	"control_training_canary_observations",
	"control_training_retrain_requests",
	"control_training_model_version_events",
	"control_training_jobs",
	"control_training_job_events",
	"control_training_job_leases",
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

	var checkpointRunCascade bool
	if err := db.QueryRow(ctx, `
SELECT EXISTS (
  SELECT 1
  FROM pg_constraint constraint_row
  WHERE constraint_row.conrelid = 'deepagents_checkpoint_threads'::regclass
    AND constraint_row.confrelid = 'control_runs'::regclass
    AND constraint_row.contype = 'f'
    AND constraint_row.confdeltype = 'c'
    AND constraint_row.conkey = ARRAY[
      (SELECT attnum FROM pg_attribute
       WHERE attrelid = 'deepagents_checkpoint_threads'::regclass
         AND attname = 'thread_id')
    ]::smallint[]
    AND constraint_row.confkey = ARRAY[
      (SELECT attnum FROM pg_attribute
       WHERE attrelid = 'control_runs'::regclass
         AND attname = 'run_id')
    ]::smallint[]
)`).Scan(&checkpointRunCascade); err != nil {
		return fmt.Errorf("verify postgres checkpoint ownership: %w", err)
	}
	if !checkpointRunCascade {
		return fmt.Errorf("postgres control schema is not ready; apply migrations before starting: deepagents_checkpoint_threads must reference control_runs(run_id) ON DELETE CASCADE")
	}
	return nil
}
