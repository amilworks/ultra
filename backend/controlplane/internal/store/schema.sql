CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

CREATE TABLE IF NOT EXISTS control_threads (
  thread_id text PRIMARY KEY,
  user_id text NOT NULL,
  title text,
  status text NOT NULL,
  created_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL,
  latest_run_id text,
  checkpoint_id text,
  summary text,
  metadata jsonb NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS control_organizations (
  org_id text PRIMARY KEY,
  name text NOT NULL,
  status text NOT NULL,
  created_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL,
  metadata jsonb NOT NULL DEFAULT '{}'
);

INSERT INTO control_organizations (org_id, name, status, created_at, updated_at, metadata)
VALUES ('local-org', 'Local Organization', 'active', now(), now(), '{"source":"dev_default"}'::jsonb)
ON CONFLICT (org_id) DO NOTHING;

CREATE TABLE IF NOT EXISTS control_users (
  user_id text PRIMARY KEY,
  email text UNIQUE,
  display_name text,
  role text NOT NULL,
  status text NOT NULL,
  org_id text,
  created_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL,
  metadata jsonb NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS control_user_token_usage_daily (
  user_id text NOT NULL,
  day date NOT NULL,
  input_tokens bigint NOT NULL DEFAULT 0,
  output_tokens bigint NOT NULL DEFAULT 0,
  total_tokens bigint NOT NULL DEFAULT 0,
  run_count bigint NOT NULL DEFAULT 0,
  updated_at timestamptz NOT NULL,
  PRIMARY KEY (user_id, day)
);

CREATE TABLE IF NOT EXISTS control_user_token_usage_lifetime (
  user_id text PRIMARY KEY,
  input_tokens bigint NOT NULL DEFAULT 0,
  output_tokens bigint NOT NULL DEFAULT 0,
  total_tokens bigint NOT NULL DEFAULT 0,
  peak_daily_total bigint NOT NULL DEFAULT 0,
  last_active_day date,
  updated_at timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS control_run_token_usage (
  run_id text NOT NULL,
  usage_event_id text NOT NULL,
  user_id text NOT NULL,
  model text NOT NULL DEFAULT '',
  day date NOT NULL,
  input_tokens bigint NOT NULL DEFAULT 0,
  output_tokens bigint NOT NULL DEFAULT 0,
  total_tokens bigint NOT NULL DEFAULT 0,
  occurred_at timestamptz NOT NULL,
  created_at timestamptz NOT NULL,
  PRIMARY KEY (run_id, usage_event_id)
);

CREATE TABLE IF NOT EXISTS control_run_token_usage_finalized (
  run_id text PRIMARY KEY,
  user_id text NOT NULL,
  day date NOT NULL,
  finalized_at timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS control_thread_messages (
  message_id text PRIMARY KEY,
  thread_id text NOT NULL REFERENCES control_threads(thread_id) ON DELETE CASCADE,
  role text NOT NULL,
  content text NOT NULL,
  created_at timestamptz NOT NULL,
  metadata jsonb NOT NULL DEFAULT '{}',
  run_id text
);

CREATE TABLE IF NOT EXISTS control_runs (
  run_id text PRIMARY KEY,
  thread_id text NOT NULL REFERENCES control_threads(thread_id) ON DELETE CASCADE,
  user_id text NOT NULL,
  goal text NOT NULL,
  status text NOT NULL,
  workflow_kind text NOT NULL,
  mode text,
  current_node text,
  parent_run_id text,
  planner_version text,
  agent_role text,
  trace_group_id text,
  checkpoint_id text,
  checkpoint_state jsonb,
  budget_state jsonb,
  response_text text,
  error text,
  created_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL,
  started_at timestamptz,
  completed_at timestamptz,
  metadata jsonb NOT NULL DEFAULT '{}'
);

-- Per-run event sequence allocator. Appends serialize on this row's lock and
-- read the next sequence from it in one statement, instead of an advisory
-- lock plus MAX() in a multi-statement transaction. last_sequence may run
-- ahead of the events table (a failed append after a counter bump leaves a
-- gap); consumers only rely on sequences being unique and increasing.
-- Declared before control_run_events on purpose: the append statement locks
-- the counter row first and the events table second, and keeping DDL in the
-- same order avoids opposite-order lock acquisition against live appenders.
CREATE TABLE IF NOT EXISTS control_run_event_sequences (
  run_id text PRIMARY KEY REFERENCES control_runs(run_id) ON DELETE CASCADE,
  last_sequence bigint NOT NULL
);

CREATE TABLE IF NOT EXISTS control_run_events (
  event_id text PRIMARY KEY,
  sequence_number bigint NOT NULL,
  source_sequence bigint,
  run_id text NOT NULL REFERENCES control_runs(run_id) ON DELETE CASCADE,
  thread_id text,
  event_kind text NOT NULL,
  event_type text,
  node_name text,
  task_id text,
  checkpoint_id text,
  scope_id text,
  agent_role text,
  level text,
  ts timestamptz NOT NULL,
  message text,
  payload jsonb NOT NULL DEFAULT '{}',
  UNIQUE(run_id, sequence_number)
);

-- Backfill source_sequence exactly once. schema.sql is re-applied on every
-- `migrate`, and an unconditional `UPDATE ... WHERE source_sequence IS NULL`
-- would full-table-scan control_run_events every time (no index can find NULLs
-- under the partial unique index). Gate the one-time column-add + backfill on
-- the column being absent: fresh databases already declare it in CREATE TABLE
-- above (so this is skipped), and databases predating the column pay the
-- rewrite once. After that the column exists and this block is a no-op.
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'control_run_events' AND column_name = 'source_sequence'
  ) THEN
    ALTER TABLE control_run_events ADD COLUMN source_sequence bigint;
    UPDATE control_run_events SET source_sequence = sequence_number;
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS control_run_leases (
  run_id text PRIMARY KEY REFERENCES control_runs(run_id) ON DELETE CASCADE,
  worker_id text NOT NULL,
  lease_token text NOT NULL UNIQUE,
  lease_expires_at timestamptz NOT NULL,
  created_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS control_worker_heartbeats (
  worker_id text PRIMARY KEY,
  worker_kind text NOT NULL,
  status text NOT NULL,
  current_run_id text,
  hostname text,
  version text,
  started_at timestamptz NOT NULL,
  last_heartbeat_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL,
  metadata jsonb NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS control_artifacts (
  artifact_id text PRIMARY KEY,
  run_id text NOT NULL REFERENCES control_runs(run_id) ON DELETE CASCADE,
  thread_id text,
  kind text NOT NULL,
  path text,
  source_path text,
  preview_path text,
  title text,
  result_group_id text,
  mime_type text,
  size_bytes bigint,
  sha256 text,
  storage_uri text,
  tool_name text,
  category text,
  created_at timestamptz NOT NULL,
  updated_at timestamptz,
  metadata jsonb NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS control_resources (
  resource_id text PRIMARY KEY,
  owner_user_id text NOT NULL,
  owner_org_id text,
  owner_role text,
  original_name text NOT NULL,
  content_type text,
  size_bytes bigint NOT NULL DEFAULT 0,
  sha256 text,
  storage_uri text,
  storage_path text,
  source_type text NOT NULL DEFAULT 'upload',
  resource_kind text NOT NULL DEFAULT 'file',
  source_uri text,
  project_id text,
  status text NOT NULL DEFAULT 'active',
  created_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL,
  deleted_at timestamptz,
  retention_expires_at timestamptz,
  metadata jsonb NOT NULL DEFAULT '{}'
);

ALTER TABLE control_resources ADD COLUMN IF NOT EXISTS project_id text;

CREATE TABLE IF NOT EXISTS control_resource_search_documents (
  resource_id text PRIMARY KEY REFERENCES control_resources(resource_id) ON DELETE CASCADE,
  owner_user_id text NOT NULL,
  owner_org_id text,
  project_id text,
  status text NOT NULL DEFAULT 'active',
  search_text text NOT NULL DEFAULT '',
  search_vector tsvector NOT NULL DEFAULT ''::tsvector,
  updated_at timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS control_resource_search_facts (
  resource_id text NOT NULL REFERENCES control_resources(resource_id) ON DELETE CASCADE,
  owner_user_id text NOT NULL DEFAULT '',
  owner_org_id text,
  project_id text,
  status text NOT NULL DEFAULT 'active',
  fact_key text NOT NULL,
  fact_text text NOT NULL DEFAULT '',
  fact_number double precision,
  fact_source text NOT NULL DEFAULT '',
  updated_at timestamptz NOT NULL
);

ALTER TABLE control_resource_search_facts ADD COLUMN IF NOT EXISTS owner_user_id text NOT NULL DEFAULT '';
ALTER TABLE control_resource_search_facts ADD COLUMN IF NOT EXISTS owner_org_id text;
ALTER TABLE control_resource_search_facts ADD COLUMN IF NOT EXISTS project_id text;
ALTER TABLE control_resource_search_facts ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'active';

UPDATE control_resource_search_facts sf
SET owner_user_id = r.owner_user_id,
    owner_org_id = r.owner_org_id,
    project_id = r.project_id,
    status = r.status
FROM control_resources r
WHERE sf.resource_id = r.resource_id
  AND sf.owner_user_id = '';

CREATE TABLE IF NOT EXISTS control_resource_events (
  event_id text PRIMARY KEY,
  resource_id text NOT NULL REFERENCES control_resources(resource_id) ON DELETE CASCADE,
  actor_user_id text,
  actor_org_id text,
  event_type text NOT NULL,
  ts timestamptz NOT NULL,
  metadata jsonb NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS control_resource_share_grants (
  grant_id text PRIMARY KEY,
  resource_id text NOT NULL REFERENCES control_resources(resource_id) ON DELETE CASCADE,
  owner_user_id text NOT NULL,
  owner_org_id text,
  owner_role text,
  grantee_user_id text,
  grantee_org_id text,
  role text NOT NULL DEFAULT 'read',
  status text NOT NULL DEFAULT 'active',
  created_by_user_id text,
  created_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL,
  revoked_at timestamptz,
  metadata jsonb NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS control_resource_collections (
  collection_id text PRIMARY KEY,
  owner_user_id text NOT NULL,
  owner_org_id text,
  owner_role text,
  project_id text,
  parent_collection_id text REFERENCES control_resource_collections(collection_id) ON DELETE SET NULL,
  name text NOT NULL,
  description text,
  collection_type text NOT NULL DEFAULT 'collection',
  status text NOT NULL DEFAULT 'active',
  created_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL,
  metadata jsonb NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS control_resource_collection_share_grants (
  grant_id text PRIMARY KEY,
  collection_id text NOT NULL REFERENCES control_resource_collections(collection_id) ON DELETE CASCADE,
  owner_user_id text NOT NULL,
  owner_org_id text,
  owner_role text,
  grantee_user_id text,
  grantee_org_id text,
  role text NOT NULL DEFAULT 'read',
  status text NOT NULL DEFAULT 'active',
  created_by_user_id text,
  created_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL,
  revoked_at timestamptz,
  metadata jsonb NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS control_resource_collection_members (
  collection_id text NOT NULL REFERENCES control_resource_collections(collection_id) ON DELETE CASCADE,
  resource_id text NOT NULL REFERENCES control_resources(resource_id) ON DELETE CASCADE,
  position bigint NOT NULL DEFAULT 0,
  added_by_user_id text,
  added_at timestamptz NOT NULL,
  metadata jsonb NOT NULL DEFAULT '{}',
  PRIMARY KEY (collection_id, resource_id)
);

CREATE TABLE IF NOT EXISTS control_dataset_snapshots (
  snapshot_id text PRIMARY KEY,
  owner_user_id text NOT NULL,
  owner_org_id text,
  owner_role text,
  project_id text,
  source_collection_id text REFERENCES control_resource_collections(collection_id) ON DELETE SET NULL,
  name text NOT NULL,
  description text,
  status text NOT NULL DEFAULT 'active',
  resource_count bigint NOT NULL DEFAULT 0,
  total_bytes bigint NOT NULL DEFAULT 0,
  created_by_user_id text,
  created_at timestamptz NOT NULL,
  metadata jsonb NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS control_dataset_snapshot_resources (
  snapshot_id text NOT NULL REFERENCES control_dataset_snapshots(snapshot_id) ON DELETE CASCADE,
  resource_id text NOT NULL,
  position bigint NOT NULL,
  original_name text NOT NULL,
  content_type text,
  size_bytes bigint NOT NULL DEFAULT 0,
  sha256 text,
  source_type text NOT NULL DEFAULT 'upload',
  resource_kind text NOT NULL DEFAULT 'file',
  storage_uri text,
  source_uri text,
  project_id text,
  resource_created_at timestamptz,
  metadata jsonb NOT NULL DEFAULT '{}',
  PRIMARY KEY (snapshot_id, resource_id)
);

CREATE TABLE IF NOT EXISTS control_dataset_snapshot_share_grants (
  grant_id text PRIMARY KEY,
  snapshot_id text NOT NULL REFERENCES control_dataset_snapshots(snapshot_id) ON DELETE CASCADE,
  owner_user_id text NOT NULL,
  owner_org_id text,
  owner_role text,
  grantee_user_id text,
  grantee_org_id text,
  role text NOT NULL DEFAULT 'read',
  status text NOT NULL DEFAULT 'active',
  created_by_user_id text,
  created_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL,
  revoked_at timestamptz,
  metadata jsonb NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS control_dataset_snapshot_events (
  event_id text PRIMARY KEY,
  snapshot_id text NOT NULL REFERENCES control_dataset_snapshots(snapshot_id) ON DELETE CASCADE,
  actor_user_id text,
  actor_org_id text,
  event_type text NOT NULL,
  ts timestamptz NOT NULL,
  metadata jsonb NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS control_data_agent_jobs (
  job_id text PRIMARY KEY,
  owner_user_id text NOT NULL,
  owner_org_id text,
  owner_role text,
  project_id text,
  job_type text NOT NULL,
  status text NOT NULL DEFAULT 'queued',
  resource_count bigint NOT NULL DEFAULT 0,
  progress_completed bigint NOT NULL DEFAULT 0,
  progress_total bigint NOT NULL DEFAULT 0,
  error text,
  created_by_user_id text,
  created_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL,
  started_at timestamptz,
  completed_at timestamptz,
  input_selector jsonb NOT NULL DEFAULT '{}',
  output_summary jsonb NOT NULL DEFAULT '{}',
  metadata jsonb NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS control_data_agent_job_resources (
  job_id text NOT NULL REFERENCES control_data_agent_jobs(job_id) ON DELETE CASCADE,
  resource_id text NOT NULL REFERENCES control_resources(resource_id) ON DELETE CASCADE,
  position bigint NOT NULL DEFAULT 0,
  metadata jsonb NOT NULL DEFAULT '{}',
  PRIMARY KEY (job_id, resource_id)
);

-- Batch-analysis jobs (analysis.megaseg / analysis.rarespot) register the resources
-- they PRODUCE (masks, bbox CSV, report) in this same junction with io_role='output',
-- distinct from the 'input' images. Lets a job report "what did you produce?".
ALTER TABLE control_data_agent_job_resources ADD COLUMN IF NOT EXISTS io_role text NOT NULL DEFAULT 'input';
CREATE INDEX IF NOT EXISTS idx_data_agent_job_resources_io_role
  ON control_data_agent_job_resources (job_id, io_role);

CREATE TABLE IF NOT EXISTS control_data_agent_job_events (
  event_id text PRIMARY KEY,
  job_id text NOT NULL REFERENCES control_data_agent_jobs(job_id) ON DELETE CASCADE,
  sequence bigint NOT NULL,
  event_type text NOT NULL,
  actor_user_id text,
  actor_org_id text,
  ts timestamptz NOT NULL,
  message text,
  metadata jsonb NOT NULL DEFAULT '{}',
  UNIQUE (job_id, sequence)
);

CREATE TABLE IF NOT EXISTS control_data_agent_job_leases (
  job_id text PRIMARY KEY REFERENCES control_data_agent_jobs(job_id) ON DELETE CASCADE,
  worker_id text NOT NULL,
  lease_token text NOT NULL UNIQUE,
  lease_expires_at timestamptz NOT NULL,
  created_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS control_upload_sessions (
  session_id text PRIMARY KEY,
  owner_user_id text NOT NULL,
  owner_org_id text,
  owner_role text,
  project_id text,
  source_type text NOT NULL DEFAULT 'upload',
  status text NOT NULL,
  total_bytes bigint NOT NULL DEFAULT 0,
  bytes_received bigint NOT NULL DEFAULT 0,
  bytes_verified bigint NOT NULL DEFAULT 0,
  bytes_committed bigint NOT NULL DEFAULT 0,
  idempotency_key text,
  browser_fingerprint text,
  error text,
  created_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL,
  completed_at timestamptz,
  metadata jsonb NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS control_upload_session_files (
  session_id text NOT NULL REFERENCES control_upload_sessions(session_id) ON DELETE CASCADE,
  file_token text NOT NULL,
  resource_id text,
  original_name text NOT NULL,
  relative_path text,
  content_type text,
  size_bytes bigint NOT NULL DEFAULT 0,
  declared_sha256 text,
  computed_sha256 text,
  status text NOT NULL,
  error text,
  created_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL,
  completed_at timestamptz,
  metadata jsonb NOT NULL DEFAULT '{}',
  PRIMARY KEY (session_id, file_token)
);

CREATE TABLE IF NOT EXISTS control_upload_session_events (
  event_id text PRIMARY KEY,
  session_id text NOT NULL REFERENCES control_upload_sessions(session_id) ON DELETE CASCADE,
  actor_user_id text,
  actor_org_id text,
  event_type text NOT NULL,
  ts timestamptz NOT NULL,
  metadata jsonb NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS control_upload_chunks (
  session_id text NOT NULL,
  file_token text NOT NULL,
  chunk_index integer NOT NULL,
  byte_offset bigint NOT NULL,
  size_bytes bigint NOT NULL DEFAULT 0,
  sha256 text NOT NULL,
  status text NOT NULL,
  storage_uri text,
  received_at timestamptz,
  verified_at timestamptz,
  error text,
  metadata jsonb NOT NULL DEFAULT '{}',
  PRIMARY KEY (session_id, file_token, chunk_index),
  FOREIGN KEY (session_id, file_token)
    REFERENCES control_upload_session_files(session_id, file_token)
    ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS control_bisque_credentials (
  session_id text PRIMARY KEY,
  user_id text NOT NULL,
  org_id text,
  root_url text NOT NULL,
  username text NOT NULL,
  password_ciphertext text NOT NULL,
  password_nonce text NOT NULL,
  password_key_id text NOT NULL,
  password_algorithm text NOT NULL,
  status text NOT NULL,
  last_verified_at timestamptz,
  created_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL,
  metadata jsonb NOT NULL DEFAULT '{}',
  UNIQUE(user_id, org_id, root_url)
);

DROP INDEX IF EXISTS control_run_events_run_sequence_idx;
DROP INDEX IF EXISTS control_run_events_run_event_idx;
DROP INDEX IF EXISTS control_data_agent_job_events_job_sequence_idx;

CREATE INDEX IF NOT EXISTS control_thread_messages_thread_created_idx ON control_thread_messages(thread_id, created_at);
CREATE INDEX IF NOT EXISTS control_runs_user_status_updated_idx ON control_runs(user_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS control_runs_thread_status_updated_idx ON control_runs(thread_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS control_threads_user_status_updated_idx ON control_threads(user_id, status, updated_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS control_runs_idempotency_unique_idx
  ON control_runs(thread_id, user_id, (metadata->>'idempotency_key'))
  WHERE COALESCE(metadata->>'idempotency_key', '') <> '';
-- Partial index for the admin activity aggregate: only tool-call/artifact
-- events pay the write cost, and the dashboard's grouped count query reads
-- just these rows instead of scanning the whole events table.
CREATE INDEX IF NOT EXISTS control_run_events_admin_activity_idx ON control_run_events(event_kind, ts)
  WHERE event_kind IN ('tool_call.started', 'artifact.created');
CREATE UNIQUE INDEX IF NOT EXISTS control_run_events_run_source_sequence_idx
  ON control_run_events(run_id, source_sequence)
  WHERE source_sequence IS NOT NULL;
CREATE INDEX IF NOT EXISTS control_run_leases_expires_idx ON control_run_leases(lease_expires_at);
CREATE INDEX IF NOT EXISTS control_worker_heartbeats_kind_status_idx ON control_worker_heartbeats(worker_kind, status, last_heartbeat_at DESC);
CREATE INDEX IF NOT EXISTS control_worker_heartbeats_last_seen_idx ON control_worker_heartbeats(last_heartbeat_at DESC);
CREATE INDEX IF NOT EXISTS control_artifacts_run_created_idx ON control_artifacts(run_id, created_at DESC);
CREATE INDEX IF NOT EXISTS control_artifacts_sha_idx ON control_artifacts(sha256);
CREATE INDEX IF NOT EXISTS control_resources_owner_status_created_idx ON control_resources(owner_user_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS control_resources_owner_org_status_idx ON control_resources(owner_user_id, owner_org_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS control_resources_project_status_idx ON control_resources(project_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS control_resources_sha_idx ON control_resources(sha256);
CREATE INDEX IF NOT EXISTS control_resources_source_uri_idx ON control_resources(source_uri);
CREATE INDEX IF NOT EXISTS control_resources_tag_keys_idx ON control_resources USING GIN ((metadata->'tag_keys'));
CREATE INDEX IF NOT EXISTS control_resource_search_documents_vector_idx ON control_resource_search_documents USING GIN (search_vector);
CREATE INDEX IF NOT EXISTS control_resource_search_documents_owner_status_idx ON control_resource_search_documents(owner_user_id, owner_org_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS control_resource_search_documents_project_status_idx ON control_resource_search_documents(project_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS control_resource_search_facts_resource_idx ON control_resource_search_facts(resource_id);
CREATE INDEX IF NOT EXISTS control_resource_search_facts_number_idx ON control_resource_search_facts(fact_key, fact_number, resource_id) WHERE fact_number IS NOT NULL;
CREATE INDEX IF NOT EXISTS control_resource_search_facts_text_idx ON control_resource_search_facts(fact_key, fact_text, resource_id) WHERE fact_text <> '';
CREATE INDEX IF NOT EXISTS control_resource_search_facts_owner_number_idx ON control_resource_search_facts(owner_user_id, owner_org_id, status, fact_key, fact_number, resource_id) WHERE fact_number IS NOT NULL;
CREATE INDEX IF NOT EXISTS control_resource_search_facts_owner_text_idx ON control_resource_search_facts(owner_user_id, owner_org_id, status, fact_key, fact_text, resource_id) WHERE fact_text <> '';
CREATE INDEX IF NOT EXISTS control_resource_events_resource_ts_idx ON control_resource_events(resource_id, ts DESC);
CREATE INDEX IF NOT EXISTS control_resource_events_ts_idx ON control_resource_events(ts DESC, event_id ASC);
CREATE INDEX IF NOT EXISTS control_resource_events_type_ts_idx ON control_resource_events(event_type, ts DESC);
CREATE INDEX IF NOT EXISTS control_resource_share_grants_resource_status_idx ON control_resource_share_grants(resource_id, status);
CREATE INDEX IF NOT EXISTS control_resource_share_grants_grantee_user_idx ON control_resource_share_grants(grantee_user_id, status);
CREATE INDEX IF NOT EXISTS control_resource_share_grants_grantee_org_idx ON control_resource_share_grants(grantee_org_id, status);
CREATE INDEX IF NOT EXISTS control_resource_collections_owner_type_idx ON control_resource_collections(owner_user_id, owner_org_id, collection_type, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS control_resource_collections_project_idx ON control_resource_collections(project_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS control_resource_collection_share_grants_collection_status_idx ON control_resource_collection_share_grants(collection_id, status);
CREATE INDEX IF NOT EXISTS control_resource_collection_share_grants_grantee_user_idx ON control_resource_collection_share_grants(grantee_user_id, status);
CREATE INDEX IF NOT EXISTS control_resource_collection_share_grants_grantee_org_idx ON control_resource_collection_share_grants(grantee_org_id, status);
CREATE INDEX IF NOT EXISTS control_resource_collection_members_resource_idx ON control_resource_collection_members(resource_id);
CREATE INDEX IF NOT EXISTS control_resource_collection_members_position_idx ON control_resource_collection_members(collection_id, position, added_at);
CREATE INDEX IF NOT EXISTS control_dataset_snapshots_owner_status_idx ON control_dataset_snapshots(owner_user_id, owner_org_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS control_dataset_snapshots_collection_idx ON control_dataset_snapshots(source_collection_id, created_at DESC);
CREATE INDEX IF NOT EXISTS control_dataset_snapshot_resources_position_idx ON control_dataset_snapshot_resources(snapshot_id, position);
CREATE INDEX IF NOT EXISTS control_dataset_snapshot_resources_resource_idx ON control_dataset_snapshot_resources(resource_id);
CREATE INDEX IF NOT EXISTS control_dataset_snapshot_share_grants_snapshot_status_idx ON control_dataset_snapshot_share_grants(snapshot_id, status);
CREATE INDEX IF NOT EXISTS control_dataset_snapshot_share_grants_grantee_user_idx ON control_dataset_snapshot_share_grants(grantee_user_id, status);
CREATE INDEX IF NOT EXISTS control_dataset_snapshot_share_grants_grantee_org_idx ON control_dataset_snapshot_share_grants(grantee_org_id, status);
CREATE INDEX IF NOT EXISTS control_dataset_snapshot_events_snapshot_ts_idx ON control_dataset_snapshot_events(snapshot_id, ts DESC);
CREATE INDEX IF NOT EXISTS control_dataset_snapshot_events_type_ts_idx ON control_dataset_snapshot_events(event_type, ts DESC);
CREATE INDEX IF NOT EXISTS control_data_agent_jobs_owner_status_idx ON control_data_agent_jobs(owner_user_id, owner_org_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS control_data_agent_jobs_type_idx ON control_data_agent_jobs(job_type, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS control_data_agent_jobs_project_idx ON control_data_agent_jobs(project_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS control_data_agent_job_resources_resource_idx ON control_data_agent_job_resources(resource_id);
CREATE INDEX IF NOT EXISTS control_data_agent_job_resources_position_idx ON control_data_agent_job_resources(job_id, position);
CREATE INDEX IF NOT EXISTS control_data_agent_job_leases_expires_idx ON control_data_agent_job_leases(lease_expires_at);
CREATE INDEX IF NOT EXISTS control_upload_sessions_owner_status_idx ON control_upload_sessions(owner_user_id, owner_org_id, status, updated_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS control_upload_sessions_idempotency_idx
  ON control_upload_sessions(owner_user_id, COALESCE(owner_org_id, ''), idempotency_key)
  WHERE COALESCE(idempotency_key, '') <> '';
CREATE INDEX IF NOT EXISTS control_upload_session_files_resource_idx ON control_upload_session_files(resource_id);
CREATE INDEX IF NOT EXISTS control_upload_session_events_session_ts_idx ON control_upload_session_events(session_id, ts DESC);
CREATE INDEX IF NOT EXISTS control_upload_chunks_status_idx ON control_upload_chunks(session_id, file_token, status);
CREATE INDEX IF NOT EXISTS control_organizations_status_updated_idx ON control_organizations(status, updated_at DESC);
CREATE INDEX IF NOT EXISTS control_users_org_status_idx ON control_users(org_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS control_users_email_idx ON control_users(lower(email));
CREATE INDEX IF NOT EXISTS control_user_token_usage_daily_user_day_idx ON control_user_token_usage_daily(user_id, day DESC);
CREATE INDEX IF NOT EXISTS control_run_token_usage_user_day_idx ON control_run_token_usage(user_id, day DESC);
CREATE INDEX IF NOT EXISTS control_run_token_usage_run_idx ON control_run_token_usage(run_id);
CREATE INDEX IF NOT EXISTS control_bisque_credentials_user_status_idx ON control_bisque_credentials(user_id, org_id, status, updated_at DESC);

-- ---------------------------------------------------------------------------
-- GoldGate training subsystem (M0): model-agnostic persistence for gold-gated
-- continual finetuning. The registry row + guardrail clause rows are DATA so a
-- new model onboards with zero DDL/enum/route changes (the M5 acceptance test).
-- Design: planning/2026-07-07-goldgate-continual-finetuning-plan.md
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS control_training_models (
  model_key text PRIMARY KEY,
  task_type text NOT NULL,
  display_name text NOT NULL,
  dataset_format text NOT NULL,
  metric_schema text NOT NULL,
  requires_phash boolean NOT NULL DEFAULT false,
  capabilities jsonb NOT NULL DEFAULT '[]',
  executor jsonb NOT NULL DEFAULT '{}',
  classes jsonb,
  leakage_defenses_extra jsonb NOT NULL DEFAULT '[]',
  metadata jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS control_training_domains (
  domain_id text PRIMARY KEY,
  name text NOT NULL,
  description text,
  metadata jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS control_training_lineages (
  lineage_id text PRIMARY KEY,
  domain_id text NOT NULL REFERENCES control_training_domains(domain_id),
  model_key text NOT NULL REFERENCES control_training_models(model_key),
  scope text NOT NULL DEFAULT 'shared',
  owner_user_id text,
  parent_lineage_id text,
  active_version_id text,
  metadata jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS control_training_model_versions (
  version_id text PRIMARY KEY,
  lineage_id text NOT NULL REFERENCES control_training_lineages(lineage_id),
  model_key text NOT NULL REFERENCES control_training_models(model_key),
  status text NOT NULL CHECK (status IN ('candidate','canary','active','retired','rejected')),
  is_frozen boolean NOT NULL DEFAULT false,
  weights_uri text,
  source_job_id text,
  artifact_run_id text,
  metrics jsonb NOT NULL DEFAULT '{}',
  metadata jsonb NOT NULL DEFAULT '{}',
  activated_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

-- Sync-job-owned CACHE row (single writer: the training.sync worker from M1 on).
-- active_model_version mirrors control_training_lineages.active_version_id and
-- retrain_gate_thresholds mirrors the control_training_gate_policies row; any
-- writer that updates the canonical tables MUST refresh this row in the same
-- transaction (the seed-parity tests pin the seed copies to one fixture).
CREATE TABLE IF NOT EXISTS control_training_model_status (
  model_key text PRIMARY KEY REFERENCES control_training_models(model_key),
  dataset_name text,
  dataset_id text,
  model_health text,
  reviewed_images bigint NOT NULL DEFAULT 0,
  unreviewed_images bigint NOT NULL DEFAULT 0,
  class_counts jsonb NOT NULL DEFAULT '{}',
  per_class_new_objects jsonb NOT NULL DEFAULT '{}',
  unsupported_class_counts jsonb NOT NULL DEFAULT '{}',
  last_sync_at timestamptz,
  last_retrain_at timestamptz,
  active_model_version text,
  retrain_gate boolean NOT NULL DEFAULT false,
  retrain_gate_reasons jsonb NOT NULL DEFAULT '[]',
  retrain_gate_counts jsonb NOT NULL DEFAULT '{}',
  retrain_gate_thresholds jsonb NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS control_training_gate_policies (
  model_key text PRIMARY KEY REFERENCES control_training_models(model_key),
  min_reviewed bigint NOT NULL,
  min_new_objects bigint NOT NULL,
  min_per_class_objects jsonb NOT NULL DEFAULT '{}',
  min_days bigint NOT NULL
);

CREATE TABLE IF NOT EXISTS control_training_guardrail_clauses (
  model_key text NOT NULL REFERENCES control_training_models(model_key),
  clause_key text NOT NULL,
  metric_path text NOT NULL,
  comparator text NOT NULL CHECK (comparator IN ('max_drop_vs_active','abs_floor','max_rise_vs_active','abs_ceiling')),
  value real NOT NULL,
  slice text,
  params jsonb NOT NULL DEFAULT '{}',
  enabled boolean NOT NULL DEFAULT true,
  required boolean NOT NULL DEFAULT false,
  PRIMARY KEY (model_key, clause_key)
);

CREATE TABLE IF NOT EXISTS control_training_gate_config_events (
  event_id text PRIMARY KEY,
  model_key text NOT NULL,
  table_name text NOT NULL,
  change jsonb NOT NULL DEFAULT '{}',
  actor_user_id text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS control_training_gold_sets (
  gold_set_id text PRIMARY KEY,
  model_key text NOT NULL REFERENCES control_training_models(model_key),
  version bigint NOT NULL,
  content_hash text UNIQUE,
  item_count bigint NOT NULL DEFAULT 0,
  label_stats jsonb NOT NULL DEFAULT '{}',
  strata_summary jsonb NOT NULL DEFAULT '{}',
  split_manifest_uri text,
  provenance jsonb NOT NULL DEFAULT '{}',
  status text NOT NULL CHECK (status IN ('draft','freezing','frozen','failed','retired')),
  created_at timestamptz NOT NULL DEFAULT now(),
  created_by_user_id text NOT NULL,
  frozen_at timestamptz,
  UNIQUE (model_key, version)
);

CREATE TABLE IF NOT EXISTS control_training_gold_items (
  gold_set_id text NOT NULL REFERENCES control_training_gold_sets(gold_set_id) ON DELETE CASCADE,
  item_id text NOT NULL,
  source_ref jsonb NOT NULL DEFAULT '{}',
  slice text NOT NULL CHECK (slice IN ('prior_train','held_out_test')),
  label_kind text NOT NULL CHECK (label_kind IN ('boxes','mask','class')),
  content_sha256 text NOT NULL,
  phash text,
  gt_label_sha256 text NOT NULL,
  gt_label_uri text NOT NULL,
  width bigint,
  height bigint,
  metadata jsonb NOT NULL DEFAULT '{}',
  footprint_geom jsonb,
  strata_tags jsonb NOT NULL DEFAULT '{}',
  PRIMARY KEY (gold_set_id, item_id)
);

CREATE TABLE IF NOT EXISTS control_training_replay_pool (
  model_key text NOT NULL REFERENCES control_training_models(model_key),
  content_sha256 text NOT NULL,
  source_ref jsonb,
  label_stats jsonb,
  site_id text,
  forgetting_risk real,
  last_used_epoch bigint,
  priority real,
  PRIMARY KEY (model_key, content_sha256)
);

CREATE TABLE IF NOT EXISTS control_training_benchmark_runs (
  run_id text PRIMARY KEY,
  model_version_id text NOT NULL,
  gold_set_id text NOT NULL,
  gold_set_content_hash text NOT NULL,
  metric_schema text NOT NULL,
  kernel_version text NOT NULL,
  metrics jsonb NOT NULL DEFAULT '{}',
  guardrails_passed boolean NOT NULL,
  guardrails_reasons jsonb NOT NULL DEFAULT '[]',
  report_uri text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS control_training_canary_observations (
  observation_id text PRIMARY KEY,
  model_key text,
  canary_version_id text NOT NULL,
  active_version_id text NOT NULL,
  run_id text NOT NULL,
  canary_metrics jsonb,
  active_metrics jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);
-- model_key scopes the UI drift-echo list route; nullable because rows written
-- before the column existed cannot be backfilled (new writes always set it).
ALTER TABLE control_training_canary_observations ADD COLUMN IF NOT EXISTS model_key text;

CREATE TABLE IF NOT EXISTS control_training_retrain_requests (
  request_id text PRIMARY KEY,
  model_key text NOT NULL REFERENCES control_training_models(model_key),
  training_job_id text,
  status text NOT NULL DEFAULT 'queued',
  note text,
  error text,
  model_version text,
  gating_summary jsonb NOT NULL DEFAULT '{}',
  benchmark_report_artifact_id text,
  requested_by_user_id text,
  created_at timestamptz NOT NULL DEFAULT now(),
  started_at timestamptz,
  finished_at timestamptz
);

CREATE TABLE IF NOT EXISTS control_training_model_version_events (
  event_id text PRIMARY KEY,
  version_id text NOT NULL,
  event_type text NOT NULL,
  actor_user_id text NOT NULL,
  from_status text,
  to_status text,
  benchmark_run_id text,
  gold_set_content_hash text,
  reason text,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS control_training_jobs (
  job_id text PRIMARY KEY,
  model_key text NOT NULL REFERENCES control_training_models(model_key),
  job_type text NOT NULL,
  status text NOT NULL DEFAULT 'queued',
  gpu_pool text,
  params jsonb NOT NULL DEFAULT '{}',
  progress_completed bigint NOT NULL DEFAULT 0,
  progress_total bigint NOT NULL DEFAULT 0,
  error text,
  owner_user_id text NOT NULL,
  owner_org_id text,
  created_by_user_id text,
  created_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL,
  started_at timestamptz,
  completed_at timestamptz,
  output_summary jsonb NOT NULL DEFAULT '{}',
  metadata jsonb NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS control_training_job_events (
  event_id text PRIMARY KEY,
  job_id text NOT NULL REFERENCES control_training_jobs(job_id) ON DELETE CASCADE,
  sequence bigint NOT NULL,
  event_type text NOT NULL,
  actor_user_id text,
  actor_org_id text,
  ts timestamptz NOT NULL,
  message text,
  metadata jsonb NOT NULL DEFAULT '{}',
  UNIQUE (job_id, sequence)
);

CREATE TABLE IF NOT EXISTS control_training_job_leases (
  job_id text PRIMARY KEY REFERENCES control_training_jobs(job_id) ON DELETE CASCADE,
  worker_id text NOT NULL,
  lease_token text NOT NULL UNIQUE,
  lease_expires_at timestamptz NOT NULL,
  created_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL
);

CREATE INDEX IF NOT EXISTS control_training_lineages_model_scope_idx ON control_training_lineages(model_key, scope, updated_at DESC);
CREATE INDEX IF NOT EXISTS control_training_lineages_domain_idx ON control_training_lineages(domain_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS control_training_model_versions_lineage_idx ON control_training_model_versions(lineage_id, created_at DESC);
CREATE INDEX IF NOT EXISTS control_training_model_versions_model_status_idx ON control_training_model_versions(model_key, status, created_at DESC);
CREATE INDEX IF NOT EXISTS control_training_gold_items_set_slice_idx ON control_training_gold_items(gold_set_id, slice);
CREATE INDEX IF NOT EXISTS control_training_gold_sets_model_status_idx ON control_training_gold_sets(model_key, status, created_at DESC);
CREATE INDEX IF NOT EXISTS control_training_benchmark_runs_version_idx ON control_training_benchmark_runs(model_version_id, created_at DESC);
CREATE INDEX IF NOT EXISTS control_training_benchmark_runs_gold_hash_idx ON control_training_benchmark_runs(gold_set_content_hash, created_at DESC);
CREATE INDEX IF NOT EXISTS control_training_replay_pool_priority_idx ON control_training_replay_pool(model_key, priority DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS control_training_retrain_requests_model_idx ON control_training_retrain_requests(model_key, created_at DESC);
CREATE INDEX IF NOT EXISTS control_training_model_version_events_version_idx ON control_training_model_version_events(version_id, created_at DESC);
CREATE INDEX IF NOT EXISTS control_training_canary_observations_canary_idx ON control_training_canary_observations(canary_version_id, created_at DESC);
CREATE INDEX IF NOT EXISTS control_training_canary_observations_model_idx ON control_training_canary_observations(model_key, created_at DESC);
CREATE INDEX IF NOT EXISTS control_training_jobs_model_status_idx ON control_training_jobs(model_key, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS control_training_jobs_type_idx ON control_training_jobs(job_type, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS control_training_job_events_job_ts_idx ON control_training_job_events(job_id, ts DESC);

-- GoldGate M0 seed: the documented direct-DB writes (registry + version 0 +
-- lineage + gate policy + guardrail clauses). model_key 'yolov5_rarespot' is
-- the ONE canonical spelling (FE, Go, DB, NATS, storage dirs). The baked
-- RareSpotWeights.pt becomes version 0: active, frozen, unbenchmarked — every
-- future weight must beat it on a frozen gold set to replace it.
INSERT INTO control_training_models (model_key, task_type, display_name, dataset_format, metric_schema, requires_phash, capabilities, executor, classes, leakage_defenses_extra, metadata)
VALUES (
  'yolov5_rarespot', 'detection', 'RareSpot prairie-dog/burrow', 'yolo_txt_tiles_512', 'detection.v1', true,
  '["SYNC","ASSEMBLE","FINETUNE","BENCHMARK"]'::jsonb,
  '{"gpu_pool":"titan","min_vram_gb":8,"shm_gb":8,"wall_clock_budget_s":21600,"cpu_only":false}'::jsonb,
  '{"0":"prairie_dog","1":"burrow"}'::jsonb,
  '["aerial_geospatial_overlap"]'::jsonb,
  '{"framework":"PyTorch/YOLOv5","description":"Prairie dog and burrow detection on aerial survey imagery (512px tiles). Gold-gated continual finetuning.","dimensions":["2d"],"workflow":"rarespot_ecology","gt_layer_priority":["gt2","New Ground Truth"]}'::jsonb
)
ON CONFLICT (model_key) DO NOTHING;

INSERT INTO control_training_domains (domain_id, name, description)
VALUES ('ecology', 'Ecology', 'Field-survey detection and segmentation models.')
ON CONFLICT (domain_id) DO NOTHING;

INSERT INTO control_training_lineages (lineage_id, domain_id, model_key, scope, active_version_id)
VALUES ('yolov5_rarespot-shared', 'ecology', 'yolov5_rarespot', 'shared', 'yolov5_rarespot-v0')
ON CONFLICT (lineage_id) DO NOTHING;

-- Guarded on "the model has NO versions at all", not just the v0 PK: schema.sql
-- re-runs on every deploy, and a plain ON CONFLICT would resurrect a deleted v0
-- as a SECOND status='active' row after later promotions (single-active invariant).
INSERT INTO control_training_model_versions (version_id, lineage_id, model_key, status, is_frozen, weights_uri, metrics, metadata, activated_at)
SELECT
  'yolov5_rarespot-v0', 'yolov5_rarespot-shared', 'yolov5_rarespot', 'active', true,
  'data/models/yolo/RareSpotWeights.pt',
  '{}'::jsonb,
  '{"is_baked":true,"provenance":"pre-GoldGate checkpoint; trained --noval on all_overfit.yaml (lineage focuswin_generalize4 -> allfullneg_calibrate2); no held-out validation existed at training time"}'::jsonb,
  now()
WHERE NOT EXISTS (SELECT 1 FROM control_training_model_versions WHERE model_key = 'yolov5_rarespot');

INSERT INTO control_training_model_status (model_key, dataset_name, model_health, active_model_version, retrain_gate, retrain_gate_reasons, retrain_gate_thresholds)
VALUES (
  'yolov5_rarespot', 'Prairie_Dog_Active_Learning', 'watch', 'yolov5_rarespot-v0', false,
  '["No reviewed training data has been synced yet - the sync path ships with M1.","Cannot check the gold-set precondition - no gold set has been frozen yet."]'::jsonb,
  '{"min_reviewed":50,"min_new_objects":200,"min_per_class_objects":{"prairie_dog":20,"burrow":20},"min_days":3}'::jsonb
)
ON CONFLICT (model_key) DO NOTHING;

INSERT INTO control_training_gate_policies (model_key, min_reviewed, min_new_objects, min_per_class_objects, min_days)
VALUES ('yolov5_rarespot', 50, 200, '{"prairie_dog":20,"burrow":20}'::jsonb, 3)
ON CONFLICT (model_key) DO NOTHING;

INSERT INTO control_training_guardrail_clauses (model_key, clause_key, metric_path, comparator, value, slice, params, enabled, required) VALUES
  ('yolov5_rarespot', 'agg_map50',           'aggregate.map50',                'max_drop_vs_active', 0.005, NULL,            '{}'::jsonb,                      true, true),
  ('yolov5_rarespot', 'agg_map50_95',        'aggregate.map50_95',             'max_drop_vs_active', 0.005, NULL,            '{}'::jsonb,                      true, true),
  ('yolov5_rarespot', 'class_recall_delta',  'per_class.*.recall_at_op',       'max_drop_vs_active', 0.02,  NULL,            '{}'::jsonb,                      true, true),
  ('yolov5_rarespot', 'class_recall_abs',    'per_class.*.recall_at_op',       'abs_floor',          0.50,  NULL,            '{}'::jsonb,                      true, true),
  ('yolov5_rarespot', 'slice_prior_map50',   'per_slice.prior_train.map50',    'max_drop_vs_active', 0.02,  'prior_train',   '{"min_label_count":10}'::jsonb,  true, true),
  ('yolov5_rarespot', 'slice_held_map50',    'per_slice.held_out_test.map50',  'max_drop_vs_active', 0.005, 'held_out_test', '{"min_label_count":10}'::jsonb,  true, true),
  ('yolov5_rarespot', 'class_ap50_collapse', 'per_class.*.ap50',               'max_drop_vs_active', 0.05,  NULL,            '{}'::jsonb,                      true, true),
  ('yolov5_rarespot', 'class_ap50_abs',      'per_class.*.ap50',               'abs_floor',          0.10,  NULL,            '{"strict":true}'::jsonb,         true, true),
  ('yolov5_rarespot', 'fp_empty_ceiling',    'aggregate.fp_per_empty_frame',   'max_rise_vs_active', 0.10,  NULL,            '{}'::jsonb,                      true, true),
  ('yolov5_rarespot', 'precision_delta',     'aggregate.precision_at_op',      'max_drop_vs_active', 0.03,  NULL,            '{}'::jsonb,                      true, true)
ON CONFLICT (model_key, clause_key) DO NOTHING;

-- Consolidated CALPHAD revision-ledger schema (migration 000008).

CREATE TABLE IF NOT EXISTS control_calphad_input_blobs (
  input_sha256 text PRIMARY KEY CHECK (input_sha256 ~ '^[0-9a-f]{64}$'),
  input_size_bytes bigint NOT NULL,
  encoding text NOT NULL,
  payload bytea NOT NULL,
  created_at timestamptz NOT NULL,
  CONSTRAINT control_calphad_input_blob_binding_unique
    UNIQUE (input_sha256, input_size_bytes),
  CONSTRAINT control_calphad_input_blob_size_check
    CHECK (input_size_bytes BETWEEN 1 AND 67108864),
  CONSTRAINT control_calphad_input_blob_encoding_check
    CHECK (encoding = 'raw'),
  CONSTRAINT control_calphad_input_blob_payload_sha_check
    CHECK (encode(sha256(payload), 'hex') = input_sha256),
  CONSTRAINT control_calphad_input_blob_payload_size_check
    CHECK (octet_length(payload) = input_size_bytes)
);

CREATE TABLE IF NOT EXISTS control_calphad_revisions (
  revision_id text PRIMARY KEY,
  resource_id text NOT NULL UNIQUE,
  owner_user_id text NOT NULL,
  owner_org_id text,
  sha256 text NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
  size_bytes bigint NOT NULL CHECK (size_bytes > 0),
  database_format text NOT NULL,
  assessment_pressure_min_pa double precision,
  assessment_pressure_max_pa double precision,
  parent_revision_id text REFERENCES control_calphad_revisions(revision_id),
  created_by_user_id text,
  created_at timestamptz NOT NULL,
  metadata jsonb NOT NULL DEFAULT '{}' CHECK (jsonb_typeof(metadata) = 'object'),
  CONSTRAINT control_calphad_revisions_binding_unique
    UNIQUE (revision_id, resource_id, sha256, size_bytes, database_format),
  CONSTRAINT control_calphad_revisions_database_format_check
    CHECK (database_format IN ('tdb', 'dat')),
  CONSTRAINT control_calphad_revisions_pressure_binding_unique
    UNIQUE (revision_id, assessment_pressure_min_pa, assessment_pressure_max_pa),
  CONSTRAINT control_calphad_revisions_pressure_limits_check
    CHECK (assessment_pressure_min_pa IS NOT NULL AND assessment_pressure_max_pa IS NOT NULL AND
           assessment_pressure_min_pa >= 1e-9 AND assessment_pressure_max_pa <= 1e12 AND
           assessment_pressure_min_pa <= assessment_pressure_max_pa),
  CONSTRAINT control_calphad_revisions_pressure_metadata_check
    CHECK (jsonb_typeof(metadata->'assessment_pressure_limits_Pa') = 'array' AND
           jsonb_array_length(metadata->'assessment_pressure_limits_Pa') = 2 AND
           jsonb_typeof(metadata->'assessment_pressure_limits_Pa'->0) = 'number' AND
           jsonb_typeof(metadata->'assessment_pressure_limits_Pa'->1) = 'number' AND
           (metadata->'assessment_pressure_limits_Pa'->>0)::double precision = assessment_pressure_min_pa AND
           (metadata->'assessment_pressure_limits_Pa'->>1)::double precision = assessment_pressure_max_pa),
  CONSTRAINT control_calphad_revisions_owner_declaration_check
    CHECK (
      metadata ? 'owner_declaration' AND
      jsonb_typeof(metadata->'owner_declaration') = 'object' AND
      metadata->'owner_declaration' = jsonb_build_object(
        'schema_version', metadata->'owner_declaration'->'schema_version',
        'authority', metadata->'owner_declaration'->'authority',
        'database_id', metadata->'owner_declaration'->'database_id',
        'source', metadata->'owner_declaration'->'source',
        'license_id', metadata->'owner_declaration'->'license_id',
        'assessment_scope', metadata->'owner_declaration'->'assessment_scope',
        'reference_state', metadata->'owner_declaration'->'reference_state',
        'assessment_temperature_limits_K', metadata->'owner_declaration'->'assessment_temperature_limits_K',
        'assessment_pressure_limits_Pa', metadata->'owner_declaration'->'assessment_pressure_limits_Pa',
        'database_format', metadata->'owner_declaration'->'database_format'
      ) AND
      jsonb_typeof(metadata->'owner_declaration'->'schema_version') = 'string' AND
      jsonb_typeof(metadata->'owner_declaration'->'authority') = 'string' AND
      jsonb_typeof(metadata->'owner_declaration'->'database_id') = 'string' AND
      jsonb_typeof(metadata->'owner_declaration'->'source') = 'string' AND
      jsonb_typeof(metadata->'owner_declaration'->'license_id') = 'string' AND
      jsonb_typeof(metadata->'owner_declaration'->'assessment_scope') = 'string' AND
      jsonb_typeof(metadata->'owner_declaration'->'reference_state') = 'string' AND
      jsonb_typeof(metadata->'owner_declaration'->'database_format') = 'string' AND
      metadata->'owner_declaration'->>'schema_version' = 'ultra.calphad.owner-declaration.v1' AND
      metadata->'owner_declaration'->>'authority' = 'resource_owner' AND
      metadata->'owner_declaration'->>'database_format' = database_format AND
      char_length(btrim(metadata->'owner_declaration'->>'database_id')) BETWEEN 1 AND 512 AND
      char_length(btrim(metadata->'owner_declaration'->>'source')) BETWEEN 1 AND 1024 AND
      char_length(btrim(metadata->'owner_declaration'->>'license_id')) BETWEEN 1 AND 128 AND
      char_length(btrim(metadata->'owner_declaration'->>'assessment_scope')) BETWEEN 1 AND 1024 AND
      char_length(btrim(metadata->'owner_declaration'->>'reference_state')) BETWEEN 1 AND 512 AND
      jsonb_typeof(metadata->'owner_declaration'->'assessment_temperature_limits_K') = 'array' AND
      jsonb_array_length(metadata->'owner_declaration'->'assessment_temperature_limits_K') = 2 AND
      jsonb_typeof(metadata->'owner_declaration'->'assessment_temperature_limits_K'->0) = 'number' AND
      jsonb_typeof(metadata->'owner_declaration'->'assessment_temperature_limits_K'->1) = 'number' AND
      (metadata->'owner_declaration'->'assessment_temperature_limits_K'->>0)::double precision >= 1 AND
      (metadata->'owner_declaration'->'assessment_temperature_limits_K'->>0)::double precision <
        (metadata->'owner_declaration'->'assessment_temperature_limits_K'->>1)::double precision AND
      (metadata->'owner_declaration'->'assessment_temperature_limits_K'->>1)::double precision <= 10000 AND
      metadata->'owner_declaration'->'assessment_pressure_limits_Pa' =
        jsonb_build_array(assessment_pressure_min_pa, assessment_pressure_max_pa)
    ),
  CONSTRAINT control_calphad_revisions_input_blob_fkey
    FOREIGN KEY (sha256, size_bytes)
    REFERENCES control_calphad_input_blobs(input_sha256, input_size_bytes),
  CHECK (parent_revision_id IS NULL OR parent_revision_id <> revision_id)
);

CREATE INDEX IF NOT EXISTS control_calphad_revisions_owner_created_idx
  ON control_calphad_revisions(owner_user_id, owner_org_id, created_at DESC);
CREATE INDEX IF NOT EXISTS control_calphad_revisions_parent_idx
  ON control_calphad_revisions(parent_revision_id) WHERE parent_revision_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS control_calphad_evidence_blobs (
  evidence_sha256 text PRIMARY KEY CHECK (evidence_sha256 ~ '^[0-9a-f]{64}$'),
  evidence_size_bytes bigint NOT NULL CHECK (evidence_size_bytes BETWEEN 1 AND 33554432),
  encoding text NOT NULL CHECK (encoding = 'raw'),
  payload bytea NOT NULL,
  created_at timestamptz NOT NULL,
  CONSTRAINT control_calphad_evidence_blob_binding_unique
    UNIQUE (evidence_sha256, evidence_size_bytes),
  CONSTRAINT control_calphad_evidence_blob_payload_sha_check
    CHECK (encode(sha256(payload), 'hex') = evidence_sha256),
  CHECK (octet_length(payload) = evidence_size_bytes)
);

CREATE TABLE IF NOT EXISTS control_calphad_validation_events (
  validation_id text PRIMARY KEY,
  revision_id text NOT NULL,
  resource_id text NOT NULL,
  database_sha256 text NOT NULL,
  database_size_bytes bigint NOT NULL,
  database_format text NOT NULL,
  assessment_pressure_min_pa double precision,
  assessment_pressure_max_pa double precision,
  database_inventory_sha256 text,
  request_sha256 text,
  status text NOT NULL,
  operation text NOT NULL,
  failure_domain text,
  failure_stage text,
  failure_code text,
  evidence_path text,
  evidence_sha256 text,
  evidence_size_bytes bigint,
  runtime_image_id text,
  pycalphad_version text,
  run_id text,
  inspection_evidence_sha256 text,
  evidence_contract_version text,
  created_by_authority text NOT NULL,
  created_at timestamptz NOT NULL,
  metadata jsonb NOT NULL DEFAULT '{}',
  CONSTRAINT control_calphad_validation_status_check
    CHECK (status IN ('pending', 'input_validated', 'equilibrium_completed', 'scheil_completed', 'failed', 'timeout', 'unsupported')),
  CONSTRAINT control_calphad_validation_operation_check
    CHECK (operation IN ('registration', 'inspect', 'equilibrium', 'scheil')),
  CONSTRAINT control_calphad_validation_evidence_sha_check
    CHECK (evidence_sha256 IS NULL OR evidence_sha256 ~ '^[0-9a-f]{64}$'),
  CONSTRAINT control_calphad_validation_evidence_size_check
    CHECK (evidence_size_bytes IS NULL OR evidence_size_bytes BETWEEN 1 AND 33554432),
  CONSTRAINT control_calphad_validation_runtime_image_check
    CHECK (runtime_image_id IS NULL OR runtime_image_id ~ '^sha256:[0-9a-f]{64}$'),
  CONSTRAINT control_calphad_validation_inspection_sha_check
    CHECK (inspection_evidence_sha256 IS NULL OR inspection_evidence_sha256 ~ '^[0-9a-f]{64}$'),
  CONSTRAINT control_calphad_validation_inventory_sha_check
    CHECK (database_inventory_sha256 IS NULL OR database_inventory_sha256 ~ '^[0-9a-f]{64}$'),
  CONSTRAINT control_calphad_validation_request_sha_check
    CHECK (request_sha256 IS NULL OR request_sha256 ~ '^[0-9a-f]{64}$'),
  CONSTRAINT control_calphad_validation_evidence_contract_check
    CHECK (evidence_contract_version IS NULL OR
           evidence_contract_version = 'ultra.calphad.retained-evidence.v2'),
  CONSTRAINT control_calphad_validation_authority_check
    CHECK (created_by_authority IN ('control_plane', 'trusted_worker')),
  CONSTRAINT control_calphad_validation_metadata_check
    CHECK (jsonb_typeof(metadata) = 'object'),
  CONSTRAINT control_calphad_validation_database_sha_check
    CHECK (database_sha256 ~ '^[0-9a-f]{64}$'),
  CONSTRAINT control_calphad_validation_database_size_check
    CHECK (database_size_bytes > 0),
  CONSTRAINT control_calphad_validation_database_format_check
    CHECK (database_format IN ('tdb', 'dat')),
  CONSTRAINT control_calphad_validation_pressure_limits_check
    CHECK (assessment_pressure_min_pa IS NOT NULL AND assessment_pressure_max_pa IS NOT NULL AND
           assessment_pressure_min_pa >= 1e-9 AND assessment_pressure_max_pa <= 1e12 AND
           assessment_pressure_min_pa <= assessment_pressure_max_pa),
  CONSTRAINT control_calphad_validation_pressure_metadata_check
    CHECK (jsonb_typeof(metadata->'assessment_pressure_limits_Pa') = 'array' AND
           jsonb_array_length(metadata->'assessment_pressure_limits_Pa') = 2 AND
           jsonb_typeof(metadata->'assessment_pressure_limits_Pa'->0) = 'number' AND
           jsonb_typeof(metadata->'assessment_pressure_limits_Pa'->1) = 'number' AND
           (metadata->'assessment_pressure_limits_Pa'->>0)::double precision = assessment_pressure_min_pa AND
           (metadata->'assessment_pressure_limits_Pa'->>1)::double precision = assessment_pressure_max_pa),
  CONSTRAINT control_calphad_validation_revision_binding_fkey
  FOREIGN KEY (revision_id, resource_id, database_sha256, database_size_bytes, database_format)
    REFERENCES control_calphad_revisions(revision_id, resource_id, sha256, size_bytes, database_format),
  CONSTRAINT control_calphad_validation_pressure_binding_fkey
  FOREIGN KEY (revision_id, assessment_pressure_min_pa, assessment_pressure_max_pa)
    REFERENCES control_calphad_revisions
      (revision_id, assessment_pressure_min_pa, assessment_pressure_max_pa),
  CONSTRAINT control_calphad_validation_evidence_blob_fkey
  FOREIGN KEY (evidence_sha256, evidence_size_bytes)
    REFERENCES control_calphad_evidence_blobs(evidence_sha256, evidence_size_bytes),
  CONSTRAINT control_calphad_validation_run_fkey
  FOREIGN KEY (run_id) REFERENCES control_runs(run_id),
  CONSTRAINT control_calphad_validation_inspection_blob_fkey
  FOREIGN KEY (inspection_evidence_sha256) REFERENCES control_calphad_evidence_blobs(evidence_sha256),
  CONSTRAINT control_calphad_validation_inspection_lineage_check
    CHECK ((operation IN ('equilibrium', 'scheil')) = (inspection_evidence_sha256 IS NOT NULL)),
  CONSTRAINT control_calphad_validation_pycalphad_version_check
    CHECK (operation = 'registration' OR pycalphad_version = '0.11.2'),
  CONSTRAINT control_calphad_validation_worker_identity_check
    CHECK ((operation = 'registration' AND database_inventory_sha256 IS NULL AND
            request_sha256 IS NULL AND evidence_contract_version IS NULL) OR
           (operation <> 'registration' AND request_sha256 IS NOT NULL AND
            (database_inventory_sha256 IS NOT NULL OR
             (operation = 'inspect' AND status IN ('failed', 'timeout', 'unsupported'))) AND
            evidence_contract_version = 'ultra.calphad.retained-evidence.v2')),
  CONSTRAINT control_calphad_validation_registration_status_check
    CHECK ((operation = 'registration') = (status = 'pending')),
  CONSTRAINT control_calphad_validation_registration_authority_check
    CHECK ((operation = 'registration') = (created_by_authority = 'control_plane')),
  CONSTRAINT control_calphad_validation_input_operation_check
    CHECK (status <> 'input_validated' OR operation = 'inspect'),
  CONSTRAINT control_calphad_validation_equilibrium_operation_check
    CHECK (status <> 'equilibrium_completed' OR operation = 'equilibrium'),
  CONSTRAINT control_calphad_validation_scheil_operation_check
    CHECK (status <> 'scheil_completed' OR operation = 'scheil'),
  CONSTRAINT control_calphad_validation_evidence_tuple_check
    CHECK ((evidence_path IS NULL AND evidence_sha256 IS NULL AND evidence_size_bytes IS NULL) OR
         (evidence_path IS NOT NULL AND evidence_sha256 IS NOT NULL AND evidence_size_bytes IS NOT NULL)),
  CONSTRAINT control_calphad_validation_retained_evidence_check
    CHECK (operation = 'registration' OR evidence_path IS NOT NULL),
  CONSTRAINT control_calphad_validation_failure_tuple_check
    CHECK (
      (status NOT IN ('failed', 'timeout', 'unsupported') AND
       failure_domain IS NULL AND failure_stage IS NULL AND failure_code IS NULL) OR
      (status IN ('failed', 'timeout', 'unsupported') AND
       failure_domain IS NOT NULL AND failure_stage IS NOT NULL AND failure_code IS NOT NULL AND
       failure_domain IN ('input', 'scientific', 'platform') AND
       failure_stage IN ('parse', 'solver', 'result_validation', 'sandbox_runtime') AND
       failure_code IN (
         'calphad_parse_failed', 'calphad_parse_timeout', 'calphad_parse_unsupported',
         'calphad_solver_failed', 'calphad_solver_timeout', 'calphad_solver_unsupported',
         'calphad_result_invalid', 'calphad_runtime_internal_failure',
         'calphad_sandbox_failed', 'calphad_sandbox_timeout'
       ) AND (
         (failure_code = 'calphad_parse_failed' AND status = 'failed' AND
          failure_domain IN ('input', 'scientific') AND failure_stage = 'parse' AND operation = 'inspect') OR
         (failure_code = 'calphad_parse_timeout' AND status = 'timeout' AND
          failure_domain = 'scientific' AND failure_stage = 'parse' AND operation = 'inspect') OR
         (failure_code = 'calphad_parse_unsupported' AND status = 'unsupported' AND
          failure_domain = 'input' AND failure_stage = 'parse' AND operation = 'inspect') OR
         (failure_code = 'calphad_solver_failed' AND status = 'failed' AND
          failure_domain IN ('input', 'scientific') AND failure_stage = 'solver' AND
          operation IN ('equilibrium', 'scheil')) OR
         (failure_code = 'calphad_solver_timeout' AND status = 'timeout' AND
          failure_domain = 'scientific' AND failure_stage = 'solver' AND
          operation IN ('equilibrium', 'scheil')) OR
         (failure_code = 'calphad_solver_unsupported' AND status = 'unsupported' AND
          failure_domain = 'scientific' AND failure_stage = 'solver' AND
          operation IN ('equilibrium', 'scheil')) OR
         (failure_code = 'calphad_result_invalid' AND status = 'failed' AND
          failure_domain = 'scientific' AND failure_stage = 'result_validation') OR
         (failure_code = 'calphad_runtime_internal_failure' AND status = 'failed' AND
          failure_domain = 'platform' AND
          ((operation = 'inspect' AND failure_stage = 'parse') OR
           (operation IN ('equilibrium', 'scheil') AND failure_stage = 'solver'))) OR
         (failure_code = 'calphad_sandbox_failed' AND status = 'failed' AND
          failure_domain = 'platform' AND failure_stage = 'sandbox_runtime') OR
         (failure_code = 'calphad_sandbox_timeout' AND status = 'timeout' AND
          failure_domain = 'platform' AND failure_stage = 'sandbox_runtime')
       ))
    ),
  CONSTRAINT control_calphad_validation_evidence_path_check
    CHECK (evidence_path IS NULL OR evidence_path = '/outputs/calphad/' ||
    CASE operation WHEN 'inspect' THEN 'inspection' ELSE operation END ||
    '/' || evidence_sha256 || '.json'),
  CONSTRAINT control_calphad_validation_runtime_binding_check
    CHECK ((operation = 'registration' AND evidence_path IS NULL AND evidence_sha256 IS NULL AND
          evidence_size_bytes IS NULL AND runtime_image_id IS NULL AND pycalphad_version IS NULL AND run_id IS NULL) OR
         (operation <> 'registration' AND runtime_image_id IS NOT NULL AND
          pycalphad_version IS NOT NULL AND run_id IS NOT NULL AND
          char_length(btrim(pycalphad_version)) BETWEEN 1 AND 128 AND
          char_length(btrim(run_id)) BETWEEN 1 AND 512))
);

-- Compatibility for a database on which an earlier IF-NOT-EXISTS draft of
-- this schema was applied. Backfill only immutable revision facts; evidence
-- bytes cannot be reconstructed. The input-blob FK remains NOT VALID for
-- historical revisions while still being enforced for every new insert;
-- ledger reads and promotion fail closed until exact input is retained.
DROP TRIGGER IF EXISTS control_calphad_validation_append_only ON control_calphad_validation_events;
ALTER TABLE control_calphad_revisions
  ADD COLUMN IF NOT EXISTS assessment_pressure_min_pa double precision,
  ADD COLUMN IF NOT EXISTS assessment_pressure_max_pa double precision,
  ADD COLUMN IF NOT EXISTS database_format text;
ALTER TABLE control_calphad_validation_events
  ADD COLUMN IF NOT EXISTS database_sha256 text,
  ADD COLUMN IF NOT EXISTS database_size_bytes bigint,
  ADD COLUMN IF NOT EXISTS inspection_evidence_sha256 text,
  ADD COLUMN IF NOT EXISTS database_inventory_sha256 text,
  ADD COLUMN IF NOT EXISTS request_sha256 text,
  ADD COLUMN IF NOT EXISTS evidence_contract_version text,
  ADD COLUMN IF NOT EXISTS failure_domain text,
  ADD COLUMN IF NOT EXISTS failure_stage text,
  ADD COLUMN IF NOT EXISTS failure_code text,
  ADD COLUMN IF NOT EXISTS assessment_pressure_min_pa double precision,
  ADD COLUMN IF NOT EXISTS assessment_pressure_max_pa double precision,
  ADD COLUMN IF NOT EXISTS database_format text;
UPDATE control_calphad_validation_events validation
SET database_sha256 = revision.sha256,
    database_size_bytes = revision.size_bytes
FROM control_calphad_revisions revision
WHERE validation.revision_id = revision.revision_id
  AND (validation.database_sha256 IS NULL OR validation.database_size_bytes IS NULL);
ALTER TABLE control_calphad_validation_events
  ALTER COLUMN database_sha256 SET NOT NULL,
  ALTER COLUMN database_size_bytes SET NOT NULL;

ALTER TABLE control_calphad_validation_events
  DROP CONSTRAINT IF EXISTS control_calphad_validation_revision_binding_fkey,
  DROP CONSTRAINT IF EXISTS control_calphad_validation_evidence_contract_check,
  DROP CONSTRAINT IF EXISTS control_calphad_validation_worker_identity_check,
  DROP CONSTRAINT IF EXISTS control_calphad_validation_status_check,
  DROP CONSTRAINT IF EXISTS control_calphad_validation_operation_check,
  DROP CONSTRAINT IF EXISTS control_calphad_validation_success_evidence_check,
  DROP CONSTRAINT IF EXISTS control_calphad_validation_inspection_lineage_check,
  DROP CONSTRAINT IF EXISTS control_calphad_validation_equilibrium_operation_check,
  DROP CONSTRAINT IF EXISTS control_calphad_validation_scheil_operation_check,
  DROP CONSTRAINT IF EXISTS control_calphad_validation_evidence_path_check,
  DROP CONSTRAINT IF EXISTS control_calphad_validation_retained_evidence_check,
  DROP CONSTRAINT IF EXISTS control_calphad_validation_failure_tuple_check;
ALTER TABLE control_calphad_revisions
  DROP CONSTRAINT IF EXISTS control_calphad_revisions_owner_declaration_check,
  DROP CONSTRAINT IF EXISTS control_calphad_revisions_binding_unique;

DO $$
DECLARE
  check_record record;
BEGIN
  FOR check_record IN
    SELECT * FROM (VALUES
      ('control_calphad_validation_status_check', $check$CHECK (status IN ('pending', 'input_validated', 'equilibrium_completed', 'scheil_completed', 'failed', 'timeout', 'unsupported'))$check$),
      ('control_calphad_validation_operation_check', $check$CHECK (operation IN ('registration', 'inspect', 'equilibrium', 'scheil'))$check$),
      ('control_calphad_validation_evidence_sha_check', $check$CHECK (evidence_sha256 IS NULL OR evidence_sha256 ~ '^[0-9a-f]{64}$')$check$),
      ('control_calphad_validation_evidence_size_check', $check$CHECK (evidence_size_bytes IS NULL OR evidence_size_bytes BETWEEN 1 AND 33554432)$check$),
      ('control_calphad_validation_runtime_image_check', $check$CHECK (runtime_image_id IS NULL OR runtime_image_id ~ '^sha256:[0-9a-f]{64}$')$check$),
      ('control_calphad_validation_inspection_sha_check', $check$CHECK (inspection_evidence_sha256 IS NULL OR inspection_evidence_sha256 ~ '^[0-9a-f]{64}$')$check$),
      ('control_calphad_validation_inventory_sha_check', $check$CHECK (database_inventory_sha256 IS NULL OR database_inventory_sha256 ~ '^[0-9a-f]{64}$')$check$),
      ('control_calphad_validation_request_sha_check', $check$CHECK (request_sha256 IS NULL OR request_sha256 ~ '^[0-9a-f]{64}$')$check$),
      ('control_calphad_validation_evidence_contract_check', $check$CHECK (evidence_contract_version IS NULL OR evidence_contract_version = 'ultra.calphad.retained-evidence.v2')$check$),
      ('control_calphad_validation_authority_check', $check$CHECK (created_by_authority IN ('control_plane', 'trusted_worker'))$check$),
      ('control_calphad_validation_metadata_check', $check$CHECK (jsonb_typeof(metadata) = 'object')$check$),
      ('control_calphad_validation_registration_status_check', $check$CHECK ((operation = 'registration') = (status = 'pending'))$check$),
      ('control_calphad_validation_registration_authority_check', $check$CHECK ((operation = 'registration') = (created_by_authority = 'control_plane'))$check$),
      ('control_calphad_validation_input_operation_check', $check$CHECK (status <> 'input_validated' OR operation = 'inspect')$check$),
      ('control_calphad_validation_equilibrium_operation_check', $check$CHECK (status <> 'equilibrium_completed' OR operation = 'equilibrium')$check$),
      ('control_calphad_validation_scheil_operation_check', $check$CHECK (status <> 'scheil_completed' OR operation = 'scheil')$check$),
      ('control_calphad_validation_evidence_tuple_check', $check$CHECK ((evidence_path IS NULL AND evidence_sha256 IS NULL AND evidence_size_bytes IS NULL) OR (evidence_path IS NOT NULL AND evidence_sha256 IS NOT NULL AND evidence_size_bytes IS NOT NULL))$check$),
      ('control_calphad_validation_retained_evidence_check', $check$CHECK (operation = 'registration' OR evidence_path IS NOT NULL)$check$),
      ('control_calphad_validation_evidence_path_check', $check$CHECK (evidence_path IS NULL OR evidence_path = '/outputs/calphad/' || CASE operation WHEN 'inspect' THEN 'inspection' ELSE operation END || '/' || evidence_sha256 || '.json')$check$),
      ('control_calphad_validation_runtime_binding_check', $check$CHECK ((operation = 'registration' AND evidence_path IS NULL AND evidence_sha256 IS NULL AND evidence_size_bytes IS NULL AND runtime_image_id IS NULL AND pycalphad_version IS NULL AND run_id IS NULL) OR (operation <> 'registration' AND runtime_image_id IS NOT NULL AND pycalphad_version IS NOT NULL AND run_id IS NOT NULL AND char_length(btrim(pycalphad_version)) BETWEEN 1 AND 128 AND char_length(btrim(run_id)) BETWEEN 1 AND 512))$check$),
      ('control_calphad_validation_worker_identity_check', $check$CHECK ((operation = 'registration' AND database_inventory_sha256 IS NULL AND request_sha256 IS NULL AND evidence_contract_version IS NULL) OR (operation <> 'registration' AND request_sha256 IS NOT NULL AND (database_inventory_sha256 IS NOT NULL OR (operation = 'inspect' AND status IN ('failed', 'timeout', 'unsupported'))) AND evidence_contract_version = 'ultra.calphad.retained-evidence.v2'))$check$),
      ('control_calphad_validation_failure_tuple_check', $check$CHECK ((status NOT IN ('failed', 'timeout', 'unsupported') AND failure_domain IS NULL AND failure_stage IS NULL AND failure_code IS NULL) OR (status IN ('failed', 'timeout', 'unsupported') AND failure_domain IS NOT NULL AND failure_stage IS NOT NULL AND failure_code IS NOT NULL AND failure_domain IN ('input', 'scientific', 'platform') AND failure_stage IN ('parse', 'solver', 'result_validation', 'sandbox_runtime') AND failure_code IN ('calphad_parse_failed', 'calphad_parse_timeout', 'calphad_parse_unsupported', 'calphad_solver_failed', 'calphad_solver_timeout', 'calphad_solver_unsupported', 'calphad_result_invalid', 'calphad_runtime_internal_failure', 'calphad_sandbox_failed', 'calphad_sandbox_timeout') AND ((failure_code = 'calphad_parse_failed' AND status = 'failed' AND failure_domain IN ('input', 'scientific') AND failure_stage = 'parse' AND operation = 'inspect') OR (failure_code = 'calphad_parse_timeout' AND status = 'timeout' AND failure_domain = 'scientific' AND failure_stage = 'parse' AND operation = 'inspect') OR (failure_code = 'calphad_parse_unsupported' AND status = 'unsupported' AND failure_domain = 'input' AND failure_stage = 'parse' AND operation = 'inspect') OR (failure_code = 'calphad_solver_failed' AND status = 'failed' AND failure_domain IN ('input', 'scientific') AND failure_stage = 'solver' AND operation IN ('equilibrium', 'scheil')) OR (failure_code = 'calphad_solver_timeout' AND status = 'timeout' AND failure_domain = 'scientific' AND failure_stage = 'solver' AND operation IN ('equilibrium', 'scheil')) OR (failure_code = 'calphad_solver_unsupported' AND status = 'unsupported' AND failure_domain = 'scientific' AND failure_stage = 'solver' AND operation IN ('equilibrium', 'scheil')) OR (failure_code = 'calphad_result_invalid' AND status = 'failed' AND failure_domain = 'scientific' AND failure_stage = 'result_validation') OR (failure_code = 'calphad_runtime_internal_failure' AND status = 'failed' AND failure_domain = 'platform' AND ((operation = 'inspect' AND failure_stage = 'parse') OR (operation IN ('equilibrium', 'scheil') AND failure_stage = 'solver'))) OR (failure_code = 'calphad_sandbox_failed' AND status = 'failed' AND failure_domain = 'platform' AND failure_stage = 'sandbox_runtime') OR (failure_code = 'calphad_sandbox_timeout' AND status = 'timeout' AND failure_domain = 'platform' AND failure_stage = 'sandbox_runtime'))))$check$)
      ,('control_calphad_validation_pressure_limits_check', $check$CHECK (assessment_pressure_min_pa IS NOT NULL AND assessment_pressure_max_pa IS NOT NULL AND assessment_pressure_min_pa >= 1e-9 AND assessment_pressure_max_pa <= 1e12 AND assessment_pressure_min_pa <= assessment_pressure_max_pa)$check$)
      ,('control_calphad_validation_pressure_metadata_check', $check$CHECK (jsonb_typeof(metadata->'assessment_pressure_limits_Pa') = 'array' AND jsonb_array_length(metadata->'assessment_pressure_limits_Pa') = 2 AND jsonb_typeof(metadata->'assessment_pressure_limits_Pa'->0) = 'number' AND jsonb_typeof(metadata->'assessment_pressure_limits_Pa'->1) = 'number' AND (metadata->'assessment_pressure_limits_Pa'->>0)::double precision = assessment_pressure_min_pa AND (metadata->'assessment_pressure_limits_Pa'->>1)::double precision = assessment_pressure_max_pa)$check$)
      ,('control_calphad_validation_database_format_check', $check$CHECK (database_format IS NOT NULL AND database_format IN ('tdb', 'dat'))$check$)
    ) AS checks(constraint_name, definition)
  LOOP
    IF NOT EXISTS (
      SELECT 1 FROM pg_constraint
      WHERE conrelid = 'control_calphad_validation_events'::regclass
        AND conname = check_record.constraint_name
    ) THEN
      EXECUTE format(
        'ALTER TABLE control_calphad_validation_events ADD CONSTRAINT %I %s NOT VALID',
        check_record.constraint_name,
        check_record.definition
      );
    END IF;
  END LOOP;
  IF NOT EXISTS (
	SELECT 1 FROM pg_constraint
	WHERE conrelid = 'control_calphad_revisions'::regclass
	  AND conname = 'control_calphad_revisions_pressure_binding_unique'
  ) THEN
	ALTER TABLE control_calphad_revisions
	  ADD CONSTRAINT control_calphad_revisions_pressure_binding_unique
	  UNIQUE (revision_id, assessment_pressure_min_pa, assessment_pressure_max_pa);
  END IF;
  IF NOT EXISTS (
	SELECT 1 FROM pg_constraint
	WHERE conrelid = 'control_calphad_revisions'::regclass
	  AND conname = 'control_calphad_revisions_pressure_limits_check'
  ) THEN
	ALTER TABLE control_calphad_revisions
	  ADD CONSTRAINT control_calphad_revisions_pressure_limits_check
	  CHECK (assessment_pressure_min_pa IS NOT NULL AND assessment_pressure_max_pa IS NOT NULL AND
	         assessment_pressure_min_pa >= 1e-9 AND assessment_pressure_max_pa <= 1e12 AND
	         assessment_pressure_min_pa <= assessment_pressure_max_pa) NOT VALID;
  END IF;
  IF NOT EXISTS (
	SELECT 1 FROM pg_constraint
	WHERE conrelid = 'control_calphad_revisions'::regclass
	  AND conname = 'control_calphad_revisions_pressure_metadata_check'
  ) THEN
	ALTER TABLE control_calphad_revisions
	  ADD CONSTRAINT control_calphad_revisions_pressure_metadata_check
	  CHECK (jsonb_typeof(metadata->'assessment_pressure_limits_Pa') = 'array' AND
	         jsonb_array_length(metadata->'assessment_pressure_limits_Pa') = 2 AND
	         jsonb_typeof(metadata->'assessment_pressure_limits_Pa'->0) = 'number' AND
	         jsonb_typeof(metadata->'assessment_pressure_limits_Pa'->1) = 'number' AND
	         (metadata->'assessment_pressure_limits_Pa'->>0)::double precision = assessment_pressure_min_pa AND
	         (metadata->'assessment_pressure_limits_Pa'->>1)::double precision = assessment_pressure_max_pa) NOT VALID;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'control_calphad_revisions'::regclass
      AND conname = 'control_calphad_revisions_database_format_check'
  ) THEN
    ALTER TABLE control_calphad_revisions
      ADD CONSTRAINT control_calphad_revisions_database_format_check
      CHECK (database_format IS NOT NULL AND database_format IN ('tdb', 'dat')) NOT VALID;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'control_calphad_revisions'::regclass
      AND conname = 'control_calphad_revisions_owner_declaration_check'
  ) THEN
    ALTER TABLE control_calphad_revisions
      ADD CONSTRAINT control_calphad_revisions_owner_declaration_check
      CHECK (
        metadata ? 'owner_declaration' AND
        jsonb_typeof(metadata->'owner_declaration') = 'object' AND
        metadata->'owner_declaration' = jsonb_build_object(
          'schema_version', metadata->'owner_declaration'->'schema_version',
          'authority', metadata->'owner_declaration'->'authority',
          'database_id', metadata->'owner_declaration'->'database_id',
          'source', metadata->'owner_declaration'->'source',
          'license_id', metadata->'owner_declaration'->'license_id',
          'assessment_scope', metadata->'owner_declaration'->'assessment_scope',
          'reference_state', metadata->'owner_declaration'->'reference_state',
          'assessment_temperature_limits_K', metadata->'owner_declaration'->'assessment_temperature_limits_K',
          'assessment_pressure_limits_Pa', metadata->'owner_declaration'->'assessment_pressure_limits_Pa',
          'database_format', metadata->'owner_declaration'->'database_format'
        ) AND
        jsonb_typeof(metadata->'owner_declaration'->'schema_version') = 'string' AND
        jsonb_typeof(metadata->'owner_declaration'->'authority') = 'string' AND
        jsonb_typeof(metadata->'owner_declaration'->'database_id') = 'string' AND
        jsonb_typeof(metadata->'owner_declaration'->'source') = 'string' AND
        jsonb_typeof(metadata->'owner_declaration'->'license_id') = 'string' AND
        jsonb_typeof(metadata->'owner_declaration'->'assessment_scope') = 'string' AND
        jsonb_typeof(metadata->'owner_declaration'->'reference_state') = 'string' AND
        jsonb_typeof(metadata->'owner_declaration'->'database_format') = 'string' AND
        metadata->'owner_declaration'->>'schema_version' = 'ultra.calphad.owner-declaration.v1' AND
        metadata->'owner_declaration'->>'authority' = 'resource_owner' AND
        metadata->'owner_declaration'->>'database_format' = database_format AND
        char_length(btrim(metadata->'owner_declaration'->>'database_id')) BETWEEN 1 AND 512 AND
        char_length(btrim(metadata->'owner_declaration'->>'source')) BETWEEN 1 AND 1024 AND
        char_length(btrim(metadata->'owner_declaration'->>'license_id')) BETWEEN 1 AND 128 AND
        char_length(btrim(metadata->'owner_declaration'->>'assessment_scope')) BETWEEN 1 AND 1024 AND
        char_length(btrim(metadata->'owner_declaration'->>'reference_state')) BETWEEN 1 AND 512 AND
        jsonb_typeof(metadata->'owner_declaration'->'assessment_temperature_limits_K') = 'array' AND
        jsonb_array_length(metadata->'owner_declaration'->'assessment_temperature_limits_K') = 2 AND
        jsonb_typeof(metadata->'owner_declaration'->'assessment_temperature_limits_K'->0) = 'number' AND
        jsonb_typeof(metadata->'owner_declaration'->'assessment_temperature_limits_K'->1) = 'number' AND
        (metadata->'owner_declaration'->'assessment_temperature_limits_K'->>0)::double precision >= 1 AND
        (metadata->'owner_declaration'->'assessment_temperature_limits_K'->>0)::double precision <
          (metadata->'owner_declaration'->'assessment_temperature_limits_K'->>1)::double precision AND
        (metadata->'owner_declaration'->'assessment_temperature_limits_K'->>1)::double precision <= 10000 AND
        metadata->'owner_declaration'->'assessment_pressure_limits_Pa' =
          jsonb_build_array(assessment_pressure_min_pa, assessment_pressure_max_pa)
      ) NOT VALID;
  END IF;
  IF NOT EXISTS (
	SELECT 1 FROM pg_constraint
	WHERE conrelid = 'control_calphad_input_blobs'::regclass
	  AND conname = 'control_calphad_input_blob_binding_unique'
  ) THEN
	ALTER TABLE control_calphad_input_blobs
	  ADD CONSTRAINT control_calphad_input_blob_binding_unique
	  UNIQUE (input_sha256, input_size_bytes);
  END IF;
  IF NOT EXISTS (
	SELECT 1 FROM pg_constraint
	WHERE conrelid = 'control_calphad_input_blobs'::regclass
	  AND conname = 'control_calphad_input_blob_size_check'
  ) THEN
	ALTER TABLE control_calphad_input_blobs
	  ADD CONSTRAINT control_calphad_input_blob_size_check
	  CHECK (input_size_bytes BETWEEN 1 AND 67108864) NOT VALID;
  END IF;
  IF NOT EXISTS (
	SELECT 1 FROM pg_constraint
	WHERE conrelid = 'control_calphad_input_blobs'::regclass
	  AND conname = 'control_calphad_input_blob_encoding_check'
  ) THEN
	ALTER TABLE control_calphad_input_blobs
	  ADD CONSTRAINT control_calphad_input_blob_encoding_check
	  CHECK (encoding = 'raw') NOT VALID;
  END IF;
  IF NOT EXISTS (
	SELECT 1 FROM pg_constraint
	WHERE conrelid = 'control_calphad_input_blobs'::regclass
	  AND conname = 'control_calphad_input_blob_payload_sha_check'
  ) THEN
	ALTER TABLE control_calphad_input_blobs
	  ADD CONSTRAINT control_calphad_input_blob_payload_sha_check
	  CHECK (encode(sha256(payload), 'hex') = input_sha256) NOT VALID;
  END IF;
  IF NOT EXISTS (
	SELECT 1 FROM pg_constraint
	WHERE conrelid = 'control_calphad_input_blobs'::regclass
	  AND conname = 'control_calphad_input_blob_payload_size_check'
  ) THEN
	ALTER TABLE control_calphad_input_blobs
	  ADD CONSTRAINT control_calphad_input_blob_payload_size_check
	  CHECK (octet_length(payload) = input_size_bytes) NOT VALID;
  END IF;
  IF NOT EXISTS (
	SELECT 1 FROM pg_constraint
	WHERE conrelid = 'control_calphad_validation_events'::regclass
	  AND conname = 'control_calphad_validation_pressure_binding_fkey'
  ) THEN
	ALTER TABLE control_calphad_validation_events
	  ADD CONSTRAINT control_calphad_validation_pressure_binding_fkey
	  FOREIGN KEY (revision_id, assessment_pressure_min_pa, assessment_pressure_max_pa)
	  REFERENCES control_calphad_revisions
	    (revision_id, assessment_pressure_min_pa, assessment_pressure_max_pa)
	  NOT VALID;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'control_calphad_evidence_blobs'::regclass
      AND conname = 'control_calphad_evidence_blob_payload_sha_check'
  ) THEN
    ALTER TABLE control_calphad_evidence_blobs
      ADD CONSTRAINT control_calphad_evidence_blob_payload_sha_check
      CHECK (encode(sha256(payload), 'hex') = evidence_sha256) NOT VALID;
  END IF;
  IF NOT EXISTS (
	SELECT 1 FROM pg_constraint
	WHERE conrelid = 'control_calphad_revisions'::regclass
	  AND conname = 'control_calphad_revisions_input_blob_fkey'
  ) THEN
	ALTER TABLE control_calphad_revisions
	  ADD CONSTRAINT control_calphad_revisions_input_blob_fkey
	  FOREIGN KEY (sha256, size_bytes)
	  REFERENCES control_calphad_input_blobs(input_sha256, input_size_bytes)
	  NOT VALID;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'control_calphad_revisions'::regclass
      AND conname = 'control_calphad_revisions_binding_unique'
  ) THEN
    ALTER TABLE control_calphad_revisions
      ADD CONSTRAINT control_calphad_revisions_binding_unique
      UNIQUE (revision_id, resource_id, sha256, size_bytes, database_format);
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'control_calphad_validation_events'::regclass
      AND conname = 'control_calphad_validation_database_sha_check'
  ) THEN
    ALTER TABLE control_calphad_validation_events
      ADD CONSTRAINT control_calphad_validation_database_sha_check
      CHECK (database_sha256 ~ '^[0-9a-f]{64}$') NOT VALID;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'control_calphad_validation_events'::regclass
      AND conname = 'control_calphad_validation_database_size_check'
  ) THEN
    ALTER TABLE control_calphad_validation_events
      ADD CONSTRAINT control_calphad_validation_database_size_check
      CHECK (database_size_bytes > 0) NOT VALID;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'control_calphad_validation_events'::regclass
      AND conname = 'control_calphad_validation_revision_binding_fkey'
  ) THEN
    ALTER TABLE control_calphad_validation_events
      ADD CONSTRAINT control_calphad_validation_revision_binding_fkey
      FOREIGN KEY (revision_id, resource_id, database_sha256, database_size_bytes, database_format)
      REFERENCES control_calphad_revisions(revision_id, resource_id, sha256, size_bytes, database_format)
      NOT VALID;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'control_calphad_validation_events'::regclass
      AND conname = 'control_calphad_validation_evidence_blob_fkey'
  ) THEN
    ALTER TABLE control_calphad_validation_events
      ADD CONSTRAINT control_calphad_validation_evidence_blob_fkey
      FOREIGN KEY (evidence_sha256, evidence_size_bytes)
      REFERENCES control_calphad_evidence_blobs(evidence_sha256, evidence_size_bytes)
      NOT VALID;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'control_calphad_validation_events'::regclass
      AND conname = 'control_calphad_validation_run_fkey'
  ) THEN
    ALTER TABLE control_calphad_validation_events
      ADD CONSTRAINT control_calphad_validation_run_fkey
      FOREIGN KEY (run_id) REFERENCES control_runs(run_id) NOT VALID;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'control_calphad_validation_events'::regclass
      AND conname = 'control_calphad_validation_inspection_blob_fkey'
  ) THEN
    ALTER TABLE control_calphad_validation_events
      ADD CONSTRAINT control_calphad_validation_inspection_blob_fkey
      FOREIGN KEY (inspection_evidence_sha256)
      REFERENCES control_calphad_evidence_blobs(evidence_sha256) NOT VALID;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'control_calphad_validation_events'::regclass
      AND conname = 'control_calphad_validation_inspection_lineage_check'
  ) THEN
    ALTER TABLE control_calphad_validation_events
      ADD CONSTRAINT control_calphad_validation_inspection_lineage_check
      CHECK ((operation IN ('equilibrium', 'scheil')) =
             (inspection_evidence_sha256 IS NOT NULL)) NOT VALID;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'control_calphad_validation_events'::regclass
      AND conname = 'control_calphad_validation_pycalphad_version_check'
  ) THEN
    ALTER TABLE control_calphad_validation_events
      ADD CONSTRAINT control_calphad_validation_pycalphad_version_check
      CHECK (operation = 'registration' OR pycalphad_version = '0.11.2') NOT VALID;
  END IF;
END;
$$;
ALTER TABLE control_calphad_revisions
  VALIDATE CONSTRAINT control_calphad_revisions_owner_declaration_check;
ALTER TABLE control_calphad_validation_events
  VALIDATE CONSTRAINT control_calphad_validation_status_check,
  VALIDATE CONSTRAINT control_calphad_validation_operation_check,
  VALIDATE CONSTRAINT control_calphad_validation_evidence_sha_check,
  VALIDATE CONSTRAINT control_calphad_validation_evidence_size_check,
  VALIDATE CONSTRAINT control_calphad_validation_runtime_image_check,
  VALIDATE CONSTRAINT control_calphad_validation_inspection_sha_check,
  VALIDATE CONSTRAINT control_calphad_validation_inventory_sha_check,
  VALIDATE CONSTRAINT control_calphad_validation_request_sha_check,
  VALIDATE CONSTRAINT control_calphad_validation_authority_check,
  VALIDATE CONSTRAINT control_calphad_validation_metadata_check,
  VALIDATE CONSTRAINT control_calphad_validation_registration_status_check,
  VALIDATE CONSTRAINT control_calphad_validation_registration_authority_check,
  VALIDATE CONSTRAINT control_calphad_validation_input_operation_check,
  VALIDATE CONSTRAINT control_calphad_validation_equilibrium_operation_check,
  VALIDATE CONSTRAINT control_calphad_validation_scheil_operation_check,
  VALIDATE CONSTRAINT control_calphad_validation_evidence_tuple_check,
  VALIDATE CONSTRAINT control_calphad_validation_retained_evidence_check,
  VALIDATE CONSTRAINT control_calphad_validation_failure_tuple_check,
  VALIDATE CONSTRAINT control_calphad_validation_worker_identity_check,
  VALIDATE CONSTRAINT control_calphad_validation_evidence_contract_check,
  VALIDATE CONSTRAINT control_calphad_validation_evidence_path_check,
  VALIDATE CONSTRAINT control_calphad_validation_runtime_binding_check,
  VALIDATE CONSTRAINT control_calphad_validation_database_sha_check,
  VALIDATE CONSTRAINT control_calphad_validation_database_size_check,
  VALIDATE CONSTRAINT control_calphad_validation_revision_binding_fkey,
  VALIDATE CONSTRAINT control_calphad_validation_inspection_blob_fkey,
  VALIDATE CONSTRAINT control_calphad_validation_inspection_lineage_check,
  VALIDATE CONSTRAINT control_calphad_validation_pycalphad_version_check;
ALTER TABLE control_calphad_evidence_blobs
  VALIDATE CONSTRAINT control_calphad_evidence_blob_payload_sha_check;
ALTER TABLE control_calphad_input_blobs
  VALIDATE CONSTRAINT control_calphad_input_blob_size_check,
  VALIDATE CONSTRAINT control_calphad_input_blob_encoding_check,
  VALIDATE CONSTRAINT control_calphad_input_blob_payload_sha_check,
  VALIDATE CONSTRAINT control_calphad_input_blob_payload_size_check;

CREATE INDEX IF NOT EXISTS control_calphad_validation_revision_created_idx
  ON control_calphad_validation_events(revision_id, created_at DESC, validation_id DESC);
CREATE INDEX IF NOT EXISTS control_calphad_validation_run_idx
  ON control_calphad_validation_events(run_id) WHERE run_id IS NOT NULL;
DROP INDEX IF EXISTS control_calphad_validation_run_operation_uidx;
DROP INDEX IF EXISTS control_calphad_validation_request_uidx;
CREATE UNIQUE INDEX IF NOT EXISTS control_calphad_validation_evidence_uidx
  ON control_calphad_validation_events(revision_id, run_id, operation, evidence_sha256)
  WHERE run_id IS NOT NULL AND evidence_sha256 IS NOT NULL;
CREATE INDEX IF NOT EXISTS control_calphad_validation_request_idx
  ON control_calphad_validation_events
    (revision_id, run_id, operation, request_sha256, created_at DESC)
  WHERE request_sha256 IS NOT NULL;
DROP INDEX IF EXISTS control_calphad_validation_inspection_lineage_idx;
CREATE INDEX control_calphad_validation_inspection_lineage_idx
  ON control_calphad_validation_events
    (revision_id, run_id, database_format, runtime_image_id, database_inventory_sha256, evidence_sha256)
  WHERE operation = 'inspect' AND status = 'input_validated';

CREATE OR REPLACE FUNCTION public.ultra_validate_calphad_revision_parent()
RETURNS trigger LANGUAGE plpgsql SET search_path = pg_catalog AS $$
BEGIN
  IF NEW.parent_revision_id IS NOT NULL AND NOT EXISTS (
	SELECT 1 FROM public.control_calphad_revisions parent
    WHERE parent.revision_id = NEW.parent_revision_id
      AND parent.resource_id <> NEW.resource_id
      AND parent.owner_user_id = NEW.owner_user_id
      AND COALESCE(parent.owner_org_id, '') = COALESCE(NEW.owner_org_id, '')
  ) THEN
    RAISE EXCEPTION 'CALPHAD parent revision must be a different resource in the same owner tenant'
      USING ERRCODE = '23503';
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS control_calphad_revisions_parent_guard ON control_calphad_revisions;
CREATE TRIGGER control_calphad_revisions_parent_guard
BEFORE INSERT ON control_calphad_revisions
FOR EACH ROW EXECUTE FUNCTION public.ultra_validate_calphad_revision_parent();

CREATE OR REPLACE FUNCTION public.ultra_validate_calphad_validation_run_authority()
RETURNS trigger LANGUAGE plpgsql SET search_path = pg_catalog AS $$
BEGIN
  IF NEW.created_by_authority = 'trusted_worker' AND NOT EXISTS (
    SELECT 1
	FROM public.control_runs run_record
	JOIN public.control_run_leases lease ON lease.run_id = run_record.run_id
	JOIN public.control_calphad_revisions revision ON revision.revision_id = NEW.revision_id
    WHERE run_record.run_id = NEW.run_id
      AND run_record.status = 'running'
      AND lease.lease_expires_at > clock_timestamp()
      AND run_record.user_id = revision.owner_user_id
      AND (COALESCE(revision.owner_org_id, '') = '' OR
           COALESCE(run_record.metadata->>'org_id', '') = revision.owner_org_id)
  ) THEN
    RAISE EXCEPTION 'CALPHAD_RUN_LEASE_INVALID: trusted CALPHAD validation requires the owner run and its active unexpired lease'
      USING ERRCODE = '28000';
  END IF;
  IF NEW.created_by_authority = 'trusted_worker' AND NOT EXISTS (
    SELECT 1
	FROM public.control_runs run_record
    WHERE run_record.run_id = NEW.run_id
      AND jsonb_typeof(run_record.metadata->'calphad_runtime_policy') = 'object'
      AND run_record.metadata->'calphad_runtime_policy' = jsonb_build_object(
		'schema_version', 'ultra.calphad.runtime-policy.v2',
        'authority', 'control_plane',
        'runtime_image_id', NEW.runtime_image_id,
		'pycalphad_version', '0.11.2',
		'network', 'none',
		'no_new_privileges', true,
		'read_only_root_filesystem', true,
		'cap_drop_all', true,
		'cpus_at_most', 8,
		'memory_bytes_at_most', 34359738368,
		'pids_at_most', 4096
      )
      AND NEW.pycalphad_version = '0.11.2'
  ) THEN
    RAISE EXCEPTION 'CALPHAD_RUNTIME_POLICY_INVALID: validation runtime is not authorized by server-stamped run metadata'
      USING ERRCODE = '28000';
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS control_calphad_validation_run_authority ON control_calphad_validation_events;
CREATE TRIGGER control_calphad_validation_run_authority
BEFORE INSERT ON control_calphad_validation_events
FOR EACH ROW EXECUTE FUNCTION public.ultra_validate_calphad_validation_run_authority();

CREATE OR REPLACE FUNCTION public.ultra_validate_calphad_pressure_binding()
RETURNS trigger LANGUAGE plpgsql SET search_path = pg_catalog AS $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM public.control_calphad_revisions revision
    WHERE revision.revision_id = NEW.revision_id
      AND revision.resource_id = NEW.resource_id
      AND revision.sha256 = NEW.database_sha256
      AND revision.size_bytes = NEW.database_size_bytes
      AND revision.database_format = NEW.database_format
      AND revision.assessment_pressure_min_pa = NEW.assessment_pressure_min_pa
      AND revision.assessment_pressure_max_pa = NEW.assessment_pressure_max_pa
      AND revision.metadata->'assessment_pressure_limits_Pa' = jsonb_build_array(
        revision.assessment_pressure_min_pa, revision.assessment_pressure_max_pa
      )
      AND revision.metadata->'owner_declaration'->>'schema_version' = 'ultra.calphad.owner-declaration.v1'
      AND revision.metadata->'owner_declaration'->>'authority' = 'resource_owner'
      AND revision.metadata->'owner_declaration'->>'database_format' = revision.database_format
      AND revision.metadata->'owner_declaration'->'assessment_pressure_limits_Pa' = jsonb_build_array(
        revision.assessment_pressure_min_pa, revision.assessment_pressure_max_pa
      )
      AND NEW.metadata->'assessment_pressure_limits_Pa' = jsonb_build_array(
        NEW.assessment_pressure_min_pa, NEW.assessment_pressure_max_pa
      )
  ) THEN
    RAISE EXCEPTION 'CALPHAD_PRESSURE_BINDING_INVALID: validation pressure limits must match the immutable owner declaration'
      USING ERRCODE = '23514';
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS control_calphad_validation_pressure_binding ON control_calphad_validation_events;
DROP TRIGGER IF EXISTS control_calphad_validation_z_pressure_binding ON control_calphad_validation_events;
CREATE TRIGGER control_calphad_validation_z_pressure_binding
BEFORE INSERT ON control_calphad_validation_events
FOR EACH ROW EXECUTE FUNCTION public.ultra_validate_calphad_pressure_binding();

CREATE OR REPLACE FUNCTION public.ultra_validate_calphad_input_retention()
RETURNS trigger LANGUAGE plpgsql SET search_path = pg_catalog AS $$
BEGIN
  IF NOT EXISTS (
	SELECT 1
	FROM public.control_calphad_revisions revision
	JOIN public.control_calphad_input_blobs blob
	  ON blob.input_sha256 = revision.sha256
	 AND blob.input_size_bytes = revision.size_bytes
	 AND octet_length(blob.payload) = revision.size_bytes
	 AND encode(sha256(blob.payload), 'hex') = revision.sha256
	WHERE revision.revision_id = NEW.revision_id
	  AND revision.resource_id = NEW.resource_id
	  AND revision.sha256 = NEW.database_sha256
	  AND revision.size_bytes = NEW.database_size_bytes
	  AND revision.database_format = NEW.database_format
  ) THEN
	RAISE EXCEPTION 'CALPHAD_INPUT_RETENTION_REQUIRED: validation requires exact retained CALPHAD database bytes'
	  USING ERRCODE = '23514';
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS control_calphad_validation_input_retention ON control_calphad_validation_events;
CREATE TRIGGER control_calphad_validation_input_retention
BEFORE INSERT ON control_calphad_validation_events
FOR EACH ROW EXECUTE FUNCTION public.ultra_validate_calphad_input_retention();

CREATE OR REPLACE FUNCTION public.ultra_validate_calphad_equilibrium_inspection_lineage()
RETURNS trigger LANGUAGE plpgsql SET search_path = pg_catalog AS $$
BEGIN
  IF NEW.operation IN ('equilibrium', 'scheil') AND NOT EXISTS (
    SELECT 1
	FROM public.control_calphad_validation_events inspection
	JOIN public.control_calphad_evidence_blobs blob
      ON blob.evidence_sha256 = inspection.evidence_sha256
     AND blob.evidence_size_bytes = inspection.evidence_size_bytes
     AND octet_length(blob.payload) = inspection.evidence_size_bytes
     AND encode(sha256(blob.payload), 'hex') = inspection.evidence_sha256
    WHERE inspection.revision_id = NEW.revision_id
      AND inspection.run_id = NEW.run_id
      AND inspection.operation = 'inspect'
      AND inspection.status = 'input_validated'
      AND inspection.database_format = NEW.database_format
      AND inspection.runtime_image_id = NEW.runtime_image_id
      AND inspection.database_inventory_sha256 = NEW.database_inventory_sha256
      AND inspection.evidence_contract_version = 'ultra.calphad.retained-evidence.v2'
      AND inspection.evidence_sha256 = NEW.inspection_evidence_sha256
  ) THEN
    RAISE EXCEPTION 'CALPHAD_INSPECTION_REQUIRED: solver operation requires exact retained inspection evidence and database inventory for the same revision, run, and runtime image'
      USING ERRCODE = '23514';
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS control_calphad_equilibrium_inspection_lineage ON control_calphad_validation_events;
CREATE TRIGGER control_calphad_equilibrium_inspection_lineage
BEFORE INSERT ON control_calphad_validation_events
FOR EACH ROW EXECUTE FUNCTION public.ultra_validate_calphad_equilibrium_inspection_lineage();

CREATE OR REPLACE FUNCTION public.ultra_reject_calphad_ledger_mutation()
RETURNS trigger LANGUAGE plpgsql SET search_path = pg_catalog AS $$
BEGIN
  RAISE EXCEPTION 'CALPHAD governance ledger is append-only';
END;
$$;

DROP TRIGGER IF EXISTS control_calphad_revisions_append_only ON control_calphad_revisions;
CREATE TRIGGER control_calphad_revisions_append_only
BEFORE UPDATE OR DELETE ON control_calphad_revisions
FOR EACH ROW EXECUTE FUNCTION public.ultra_reject_calphad_ledger_mutation();

DROP TRIGGER IF EXISTS control_calphad_validation_append_only ON control_calphad_validation_events;
CREATE TRIGGER control_calphad_validation_append_only
BEFORE UPDATE OR DELETE ON control_calphad_validation_events
FOR EACH ROW EXECUTE FUNCTION public.ultra_reject_calphad_ledger_mutation();

DROP TRIGGER IF EXISTS control_calphad_revisions_no_truncate ON control_calphad_revisions;
CREATE TRIGGER control_calphad_revisions_no_truncate
BEFORE TRUNCATE ON control_calphad_revisions
FOR EACH STATEMENT EXECUTE FUNCTION public.ultra_reject_calphad_ledger_mutation();

DROP TRIGGER IF EXISTS control_calphad_validation_no_truncate ON control_calphad_validation_events;
CREATE TRIGGER control_calphad_validation_no_truncate
BEFORE TRUNCATE ON control_calphad_validation_events
FOR EACH STATEMENT EXECUTE FUNCTION public.ultra_reject_calphad_ledger_mutation();

DROP TRIGGER IF EXISTS control_calphad_evidence_blobs_append_only ON control_calphad_evidence_blobs;
CREATE TRIGGER control_calphad_evidence_blobs_append_only
BEFORE UPDATE OR DELETE ON control_calphad_evidence_blobs
FOR EACH ROW EXECUTE FUNCTION public.ultra_reject_calphad_ledger_mutation();

DROP TRIGGER IF EXISTS control_calphad_evidence_blobs_no_truncate ON control_calphad_evidence_blobs;
CREATE TRIGGER control_calphad_evidence_blobs_no_truncate
BEFORE TRUNCATE ON control_calphad_evidence_blobs
FOR EACH STATEMENT EXECUTE FUNCTION public.ultra_reject_calphad_ledger_mutation();

DROP TRIGGER IF EXISTS control_calphad_input_blobs_append_only ON control_calphad_input_blobs;
CREATE TRIGGER control_calphad_input_blobs_append_only
BEFORE UPDATE OR DELETE ON control_calphad_input_blobs
FOR EACH ROW EXECUTE FUNCTION public.ultra_reject_calphad_ledger_mutation();

DROP TRIGGER IF EXISTS control_calphad_input_blobs_no_truncate ON control_calphad_input_blobs;
CREATE TRIGGER control_calphad_input_blobs_no_truncate
BEFORE TRUNCATE ON control_calphad_input_blobs
FOR EACH STATEMENT EXECUTE FUNCTION public.ultra_reject_calphad_ledger_mutation();

-- Execute-only CALPHAD writer and tenant-capacity schema (migration 000009).

-- CALPHAD writes are admitted only through two fixed migration-owner
-- SECURITY DEFINER functions. The serving role receives their exact EXECUTE
-- signatures from GrantPostgresServingPrivileges and no raw table INSERT.

CREATE TABLE IF NOT EXISTS control_calphad_tenant_capacity (
  owner_user_id text NOT NULL,
  owner_org_id text NOT NULL DEFAULT '',
  max_retained_bytes bigint NOT NULL DEFAULT 4294967296,
  max_validation_events bigint NOT NULL DEFAULT 100000,
  retained_input_bytes bigint NOT NULL DEFAULT 0,
  retained_evidence_bytes bigint NOT NULL DEFAULT 0,
  validation_events bigint NOT NULL DEFAULT 0,
  updated_at timestamptz NOT NULL,
  PRIMARY KEY (owner_user_id, owner_org_id),
  CONSTRAINT control_calphad_tenant_capacity_limits_check
    CHECK (max_retained_bytes > 0 AND max_validation_events > 0),
  CONSTRAINT control_calphad_tenant_capacity_counters_check
    CHECK (retained_input_bytes >= 0 AND retained_evidence_bytes >= 0 AND
           validation_events >= 0 AND
           retained_input_bytes::numeric + retained_evidence_bytes::numeric <= max_retained_bytes::numeric AND
           validation_events <= max_validation_events)
);

DO $$
DECLARE
  inconsistent_count bigint;
BEGIN
  SELECT count(*) INTO inconsistent_count
  FROM public.control_calphad_revisions revision
  LEFT JOIN public.control_calphad_input_blobs blob
    ON blob.input_sha256 = revision.sha256
   AND blob.input_size_bytes = revision.size_bytes
   AND octet_length(blob.payload) = revision.size_bytes
   AND encode(sha256(blob.payload), 'hex') = revision.sha256
  WHERE blob.input_sha256 IS NULL;
  IF inconsistent_count <> 0 THEN
    RAISE EXCEPTION 'CALPHAD_INPUT_RETENTION_REQUIRED: capacity backfill found % inconsistent revisions', inconsistent_count
      USING ERRCODE = '23514';
  END IF;

  SELECT count(*) INTO inconsistent_count
  FROM public.control_calphad_validation_events validation
  LEFT JOIN public.control_calphad_evidence_blobs blob
    ON blob.evidence_sha256 = validation.evidence_sha256
   AND blob.evidence_size_bytes = validation.evidence_size_bytes
   AND octet_length(blob.payload) = validation.evidence_size_bytes
   AND encode(sha256(blob.payload), 'hex') = validation.evidence_sha256
  WHERE validation.operation <> 'registration' AND blob.evidence_sha256 IS NULL;
  IF inconsistent_count <> 0 THEN
    RAISE EXCEPTION 'CALPHAD_EVIDENCE_RETENTION_REQUIRED: capacity backfill found % inconsistent events', inconsistent_count
      USING ERRCODE = '23514';
  END IF;
END;
$$;

CREATE OR REPLACE FUNCTION public.ultra_append_calphad_validation_v1(
  p_resource_id text,
  p_owner_user_id text,
  p_owner_org_id text,
  p_database_sha256 text,
  p_database_size_bytes bigint,
  p_database_format text,
  p_owner_declaration jsonb,
  p_assessment_pressure_min_pa double precision,
  p_assessment_pressure_max_pa double precision,
  p_database_inventory_sha256 text,
  p_request_sha256 text,
  p_status text,
  p_operation text,
  p_failure_domain text,
  p_failure_stage text,
  p_failure_code text,
  p_evidence_path text,
  p_evidence_sha256 text,
  p_evidence_size_bytes bigint,
  p_evidence_payload bytea,
  p_runtime_image_id text,
  p_pycalphad_version text,
  p_run_id text,
  p_inspection_evidence_sha256 text,
  p_lease_worker_id text,
  p_lease_token text,
  p_metadata jsonb
)
RETURNS TABLE (
  validation_id text,
  revision_id text,
  resource_id text,
  database_sha256 text,
  database_size_bytes bigint,
  database_format text,
  assessment_pressure_min_pa double precision,
  assessment_pressure_max_pa double precision,
  database_inventory_sha256 text,
  request_sha256 text,
  status text,
  operation text,
  failure_domain text,
  failure_stage text,
  failure_code text,
  evidence_path text,
  evidence_sha256 text,
  evidence_size_bytes bigint,
  runtime_image_id text,
  pycalphad_version text,
  run_id text,
  inspection_evidence_sha256 text,
  evidence_contract_version text,
  created_by_authority text,
  created_at timestamptz,
  metadata jsonb,
  evidence_blob_retained boolean
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog
AS $$
DECLARE
  run_record public.control_runs%ROWTYPE;
  lease_record public.control_run_leases%ROWTYPE;
  resource_record public.control_resources%ROWTYPE;
  revision_record public.control_calphad_revisions%ROWTYPE;
  existing_validation public.control_calphad_validation_events%ROWTYPE;
  inserted_validation public.control_calphad_validation_events%ROWTYPE;
  capacity_record public.control_calphad_tenant_capacity%ROWTYPE;
  normalized_org text;
  derived_format text;
  calphad_metadata jsonb;
  declared_temperature jsonb;
  expected_declaration jsonb;
  evidence_json jsonb;
  validation_metadata jsonb;
  stored_size bigint;
  stored_payload bytea;
  selected_file_count bigint;
  descriptor_candidate_count bigint;
  descriptor_exact_count bigint;
  capacity_updated bigint;
  new_validation_id text;
  created_timestamp timestamptz;
BEGIN
  p_resource_id := btrim(p_resource_id);
  p_owner_user_id := btrim(p_owner_user_id);
  normalized_org := COALESCE(NULLIF(btrim(p_owner_org_id), ''), '');
  p_database_sha256 := lower(btrim(p_database_sha256));
  p_database_format := btrim(p_database_format);
  p_database_inventory_sha256 := NULLIF(lower(btrim(p_database_inventory_sha256)), '');
  p_request_sha256 := lower(btrim(p_request_sha256));
  p_status := btrim(p_status);
  p_operation := btrim(p_operation);
  p_failure_domain := NULLIF(btrim(p_failure_domain), '');
  p_failure_stage := NULLIF(btrim(p_failure_stage), '');
  p_failure_code := NULLIF(btrim(p_failure_code), '');
  p_evidence_path := btrim(p_evidence_path);
  p_evidence_sha256 := lower(btrim(p_evidence_sha256));
  p_runtime_image_id := lower(btrim(p_runtime_image_id));
  p_pycalphad_version := btrim(p_pycalphad_version);
  p_run_id := btrim(p_run_id);
  p_inspection_evidence_sha256 := NULLIF(lower(btrim(p_inspection_evidence_sha256)), '');
  p_lease_worker_id := btrim(p_lease_worker_id);

  IF p_resource_id = '' OR p_owner_user_id = '' OR p_run_id = '' OR
     p_lease_worker_id = '' OR p_lease_token IS NULL OR btrim(p_lease_token) = '' OR
     p_database_sha256 !~ '^[0-9a-f]{64}$' OR p_database_size_bytes <= 0 OR
     p_database_format NOT IN ('tdb', 'dat') OR
     p_assessment_pressure_min_pa < 1e-9 OR
     p_assessment_pressure_max_pa > 1e12 OR
     p_assessment_pressure_min_pa > p_assessment_pressure_max_pa OR
     p_request_sha256 !~ '^[0-9a-f]{64}$' OR
     p_operation NOT IN ('inspect', 'equilibrium', 'scheil') OR
     p_status NOT IN ('input_validated', 'equilibrium_completed', 'scheil_completed', 'failed', 'timeout', 'unsupported') OR
     p_evidence_sha256 !~ '^[0-9a-f]{64}$' OR
     p_evidence_size_bytes NOT BETWEEN 1 AND 33554432 OR
     p_evidence_payload IS NULL OR octet_length(p_evidence_payload) <> p_evidence_size_bytes OR
     encode(sha256(p_evidence_payload), 'hex') <> p_evidence_sha256 OR
     p_evidence_path <> ('/outputs/calphad/' ||
       CASE p_operation WHEN 'inspect' THEN 'inspection' ELSE p_operation END ||
       '/' || p_evidence_sha256 || '.json') OR
     p_runtime_image_id !~ '^sha256:[0-9a-f]{64}$' OR
     p_pycalphad_version <> '0.11.2' OR
     jsonb_typeof(p_owner_declaration) <> 'object' OR
     jsonb_typeof(p_metadata) <> 'object' OR
     octet_length(convert_to(p_metadata::text, 'UTF8')) > 65536 THEN
    RAISE EXCEPTION 'CALPHAD_VALIDATION_BINDING_INVALID: validation request is not canonical or content-bound'
      USING ERRCODE = '23514';
  END IF;

  BEGIN
    evidence_json := convert_from(p_evidence_payload, 'UTF8')::jsonb;
  EXCEPTION WHEN others THEN
    RAISE EXCEPTION 'CALPHAD_EVIDENCE_SCHEMA_INVALID: retained evidence is not UTF-8 JSON'
      USING ERRCODE = '23514';
  END;
  IF jsonb_typeof(evidence_json) IS DISTINCT FROM 'object' OR
     (SELECT count(*) FROM jsonb_object_keys(evidence_json)) <> 7 OR
     jsonb_typeof(evidence_json->'schema_version') IS DISTINCT FROM 'string' OR
     jsonb_typeof(evidence_json->'operation') IS DISTINCT FROM 'string' OR
     evidence_json->>'operation' IS DISTINCT FROM p_operation OR
     jsonb_typeof(evidence_json->'database_binding') IS DISTINCT FROM 'object' OR
     jsonb_typeof(evidence_json->'request') IS DISTINCT FROM 'object' OR
     jsonb_typeof(evidence_json->'execution_contract') IS DISTINCT FROM 'object' OR
     jsonb_typeof(evidence_json->'validation_persistence') IS DISTINCT FROM 'object' OR
     NOT (
       (evidence_json->>'schema_version' IS NOT DISTINCT FROM 'ultra.calphad.tool-evidence.v3' AND
        ((p_operation = 'inspect' AND p_status = 'input_validated') OR
         (p_operation = 'equilibrium' AND p_status = 'equilibrium_completed') OR
         (p_operation = 'scheil' AND p_status = 'scheil_completed')) AND
        evidence_json ?& ARRAY[
          'schema_version', 'operation', 'database_binding', 'request', 'result',
          'execution_contract', 'validation_persistence'
        ] AND jsonb_typeof(evidence_json->'result') IS NOT DISTINCT FROM 'object') OR
       (evidence_json->>'schema_version' IS NOT DISTINCT FROM 'ultra.calphad.failure-evidence.v1' AND
        p_status IN ('failed', 'timeout', 'unsupported') AND
        evidence_json ?& ARRAY[
          'schema_version', 'operation', 'database_binding', 'request', 'outcome',
          'execution_contract', 'validation_persistence'
        ] AND jsonb_typeof(evidence_json->'outcome') IS NOT DISTINCT FROM 'object')
     ) OR
     (SELECT count(*) FROM jsonb_object_keys(evidence_json->'database_binding')) <> 15 OR
     NOT (evidence_json->'database_binding' ?& ARRAY[
       'kind', 'database_id', 'resource_id', 'sha256', 'size_bytes', 'database_format',
       'source', 'license_id', 'assessment_scope', 'reference_state',
       'temperature_limits_K', 'assessment_pressure_limits_Pa', 'binding_schema',
       'binding_authority', 'declaration_authority'
     ]) OR
     evidence_json#>>'{database_binding,kind}' IS DISTINCT FROM 'resource' OR
     evidence_json#>>'{database_binding,resource_id}' IS DISTINCT FROM p_resource_id OR
     lower(btrim(evidence_json#>>'{database_binding,sha256}')) IS DISTINCT FROM p_database_sha256 OR
     evidence_json#>'{database_binding,size_bytes}' IS DISTINCT FROM to_jsonb(p_database_size_bytes) OR
     evidence_json#>>'{database_binding,database_format}' IS DISTINCT FROM p_database_format OR
     evidence_json#>>'{database_binding,binding_schema}' IS DISTINCT FROM 'ultra.selected_resource.v1' OR
     evidence_json#>>'{database_binding,binding_authority}' IS DISTINCT FROM 'control_resource_catalog' OR
     evidence_json#>>'{database_binding,declaration_authority}' IS DISTINCT FROM 'resource_owner' OR
     evidence_json#>>'{database_binding,database_id}' IS DISTINCT FROM p_owner_declaration->>'database_id' OR
     evidence_json#>>'{database_binding,source}' IS DISTINCT FROM p_owner_declaration->>'source' OR
     evidence_json#>>'{database_binding,license_id}' IS DISTINCT FROM p_owner_declaration->>'license_id' OR
     evidence_json#>>'{database_binding,assessment_scope}' IS DISTINCT FROM
       p_owner_declaration->>'assessment_scope' OR
     evidence_json#>>'{database_binding,reference_state}' IS DISTINCT FROM
       p_owner_declaration->>'reference_state' OR
     evidence_json#>'{database_binding,temperature_limits_K}' IS DISTINCT FROM
       p_owner_declaration->'assessment_temperature_limits_K' OR
     evidence_json#>'{database_binding,assessment_pressure_limits_Pa}' IS DISTINCT FROM
       jsonb_build_array(p_assessment_pressure_min_pa, p_assessment_pressure_max_pa) OR
     evidence_json#>>'{request,runtime_image_id}' IS DISTINCT FROM p_runtime_image_id OR
     evidence_json->'execution_contract' IS DISTINCT FROM jsonb_build_object(
       'interface', 'fixed ultra_deepagents.materials.calphad public surface',
       'caller_code_accepted', false,
       'caller_models_or_solver_options_accepted', false,
       'network', 'none',
       'no_new_privileges', true,
       'read_only_root_filesystem', true,
       'cap_drop_all', true,
       'cpus_at_most', 8,
       'memory_bytes_at_most', 34359738368,
       'pids_at_most', 4096,
       'runtime_image_id', p_runtime_image_id,
       'max_components', 32,
       'max_phases', 128,
       'max_axis_values', 64,
       'max_grid_points', 256,
       'wall_time_seconds', 30,
       'max_result_bytes', 16777216
     ) OR
     (SELECT count(*) FROM jsonb_object_keys(evidence_json->'validation_persistence')) <> 4 OR
     NOT (evidence_json->'validation_persistence' ?& ARRAY[
       'catalog_status', 'catalog_metadata_updated', 'mode', 'note'
     ]) OR
     evidence_json#>>'{validation_persistence,catalog_status}' IS DISTINCT FROM 'pending' OR
     evidence_json#>'{validation_persistence,catalog_metadata_updated}' IS DISTINCT FROM 'false'::jsonb OR
     evidence_json#>>'{validation_persistence,mode}' IS DISTINCT FROM 'immutable_per_run_evidence' OR
     jsonb_typeof(evidence_json#>'{validation_persistence,note}') IS DISTINCT FROM 'string' OR
     char_length(btrim(evidence_json#>>'{validation_persistence,note}')) NOT BETWEEN 1 AND 1024 THEN
    RAISE EXCEPTION 'CALPHAD_EVIDENCE_SCHEMA_INVALID: retained evidence root or database binding is inconsistent'
      USING ERRCODE = '23514';
  END IF;

  SELECT run.* INTO run_record
  FROM public.control_runs run
  WHERE run.run_id = p_run_id
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'CALPHAD_RUN_LEASE_INVALID: active run is missing'
      USING ERRCODE = '28000';
  END IF;
  SELECT lease.* INTO lease_record
  FROM public.control_run_leases lease
  WHERE lease.run_id = p_run_id
  FOR UPDATE;
  IF NOT FOUND OR run_record.status <> 'running' OR
     lease_record.lease_expires_at <= clock_timestamp() OR
     lease_record.worker_id <> p_lease_worker_id OR
     sha256(convert_to(lease_record.lease_token, 'UTF8')) <>
       sha256(convert_to(p_lease_token, 'UTF8')) THEN
    RAISE EXCEPTION 'CALPHAD_RUN_LEASE_INVALID: worker identity or active lease does not match'
      USING ERRCODE = '28000';
  END IF;
  IF run_record.metadata->'calphad_runtime_policy' <> jsonb_build_object(
       'schema_version', 'ultra.calphad.runtime-policy.v2',
       'authority', 'control_plane',
       'runtime_image_id', p_runtime_image_id,
       'pycalphad_version', '0.11.2',
       'network', 'none',
       'no_new_privileges', true,
       'read_only_root_filesystem', true,
       'cap_drop_all', true,
       'cpus_at_most', 8,
       'memory_bytes_at_most', 34359738368,
       'pids_at_most', 4096
     ) THEN
    RAISE EXCEPTION 'CALPHAD_RUNTIME_POLICY_INVALID: run policy does not authorize this runtime'
      USING ERRCODE = '28000';
  END IF;

  SELECT resource.* INTO resource_record
  FROM public.control_resources resource
  WHERE resource.resource_id = p_resource_id
    AND resource.owner_user_id = p_owner_user_id
    AND (COALESCE(NULLIF(btrim(resource.owner_org_id), ''), '') = '' OR
         COALESCE(NULLIF(btrim(resource.owner_org_id), ''), '') = normalized_org)
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'CALPHAD_RESOURCE_NOT_FOUND' USING ERRCODE = 'P0002';
  END IF;
  derived_format := CASE
    WHEN lower(btrim(resource_record.original_name)) ~ '\.tdb$' THEN 'tdb'
    WHEN lower(btrim(resource_record.original_name)) ~ '\.dat$' THEN 'dat'
    ELSE ''
  END;
  IF resource_record.status <> 'active' OR derived_format <> p_database_format OR
     lower(btrim(resource_record.sha256)) <> p_database_sha256 OR
     resource_record.size_bytes <> p_database_size_bytes THEN
    RAISE EXCEPTION 'CALPHAD_RESOURCE_BINDING_INVALID: live catalog binding changed'
      USING ERRCODE = '23514';
  END IF;

  SELECT revision.* INTO revision_record
  FROM public.control_calphad_revisions revision
  WHERE revision.resource_id = p_resource_id;
  IF NOT FOUND OR revision_record.owner_user_id <> run_record.user_id OR
     (COALESCE(NULLIF(btrim(revision_record.owner_org_id), ''), '') <> '' AND
      COALESCE(NULLIF(btrim(run_record.metadata->>'org_id'), ''), '') <>
        COALESCE(NULLIF(btrim(revision_record.owner_org_id), ''), '')) OR
     revision_record.sha256 <> p_database_sha256 OR
     revision_record.size_bytes <> p_database_size_bytes OR
     revision_record.database_format <> p_database_format OR
     revision_record.assessment_pressure_min_pa <> p_assessment_pressure_min_pa OR
     revision_record.assessment_pressure_max_pa <> p_assessment_pressure_max_pa OR
     revision_record.metadata->'owner_declaration' <> p_owner_declaration THEN
    RAISE EXCEPTION 'CALPHAD_REVISION_BINDING_INVALID: immutable revision does not match callback authority'
      USING ERRCODE = '23514';
  END IF;

  calphad_metadata := resource_record.metadata->'calphad';
  declared_temperature := COALESCE(
    calphad_metadata->'assessment_temperature_limits_K',
    calphad_metadata->'tdb_temperature_limits_K'
  );
  expected_declaration := jsonb_build_object(
    'schema_version', 'ultra.calphad.owner-declaration.v1',
    'authority', 'resource_owner',
    'database_id', COALESCE(NULLIF(btrim(calphad_metadata->>'database_id'), ''), p_resource_id),
    'source', btrim(calphad_metadata->>'source'),
    'license_id', COALESCE(NULLIF(btrim(calphad_metadata->>'license_id'), ''),
                           btrim(calphad_metadata->>'license_identifier')),
    'assessment_scope', btrim(calphad_metadata->>'assessment_scope'),
    'reference_state', btrim(calphad_metadata->>'reference_state'),
    'assessment_temperature_limits_K', declared_temperature,
    'assessment_pressure_limits_Pa', calphad_metadata->'assessment_pressure_limits_Pa',
    'database_format', derived_format
  );
  IF p_owner_declaration <> expected_declaration OR
     (calphad_metadata ? 'assessment_temperature_limits_K' AND
      calphad_metadata ? 'tdb_temperature_limits_K' AND
      calphad_metadata->'assessment_temperature_limits_K' <>
        calphad_metadata->'tdb_temperature_limits_K') THEN
    RAISE EXCEPTION 'CALPHAD_OWNER_DECLARATION_INVALID: live and immutable provenance differ'
      USING ERRCODE = '23514';
  END IF;

  IF jsonb_typeof(run_record.metadata->'file_ids') <> 'array' OR
     jsonb_typeof(run_record.metadata->'resource_descriptors') <> 'array' THEN
    RAISE EXCEPTION 'CALPHAD_SELECTED_RESOURCE_INVALID: run lacks server-selected resource authority'
      USING ERRCODE = '23514';
  END IF;
  SELECT count(*) INTO selected_file_count
  FROM jsonb_array_elements_text(run_record.metadata->'file_ids') selected(value)
  WHERE selected.value = p_resource_id;
  SELECT count(*) INTO descriptor_candidate_count
  FROM jsonb_array_elements(run_record.metadata->'resource_descriptors') descriptor(value)
  WHERE descriptor.value->>'resource_id' = p_resource_id OR
        descriptor.value->>'file_id' = p_resource_id;
  SELECT count(*) INTO descriptor_exact_count
  FROM jsonb_array_elements(run_record.metadata->'resource_descriptors') descriptor(value)
  WHERE descriptor.value->>'type' = 'selected_resource'
    AND descriptor.value->>'binding_schema' = 'ultra.selected_resource.v1'
    AND descriptor.value->>'authority' = 'control_resource_catalog'
    AND descriptor.value->>'resource_id' = p_resource_id
    AND descriptor.value->>'file_id' = p_resource_id
    AND lower(btrim(descriptor.value->>'sha256')) = p_database_sha256
    AND descriptor.value->'size_bytes' = to_jsonb(p_database_size_bytes)
    AND descriptor.value->>'database_format' = p_database_format
    AND CASE
          WHEN lower(btrim(descriptor.value->>'original_name')) ~ '\.tdb$' THEN 'tdb'
          WHEN lower(btrim(descriptor.value->>'original_name')) ~ '\.dat$' THEN 'dat'
          ELSE ''
        END = p_database_format
    AND descriptor.value->>'calphad_governance_scope' = 'owner_validation'
    AND jsonb_build_object(
      'schema_version', 'ultra.calphad.owner-declaration.v1',
      'authority', 'resource_owner',
      'database_id', COALESCE(NULLIF(btrim(descriptor.value#>>'{metadata,calphad,database_id}'), ''), p_resource_id),
      'source', btrim(descriptor.value#>>'{metadata,calphad,source}'),
      'license_id', COALESCE(NULLIF(btrim(descriptor.value#>>'{metadata,calphad,license_id}'), ''),
                             btrim(descriptor.value#>>'{metadata,calphad,license_identifier}')),
      'assessment_scope', btrim(descriptor.value#>>'{metadata,calphad,assessment_scope}'),
      'reference_state', btrim(descriptor.value#>>'{metadata,calphad,reference_state}'),
      'assessment_temperature_limits_K', COALESCE(
        descriptor.value#>'{metadata,calphad,assessment_temperature_limits_K}',
        descriptor.value#>'{metadata,calphad,tdb_temperature_limits_K}'
      ),
      'assessment_pressure_limits_Pa',
        descriptor.value#>'{metadata,calphad,assessment_pressure_limits_Pa}',
      'database_format', descriptor.value->>'database_format'
    ) = p_owner_declaration;
  IF selected_file_count <> 1 OR descriptor_candidate_count <> 1 OR descriptor_exact_count <> 1 THEN
    RAISE EXCEPTION 'CALPHAD_SELECTED_RESOURCE_INVALID: descriptor is missing, ambiguous, or content-mismatched'
      USING ERRCODE = '23514';
  END IF;

  SELECT blob.input_size_bytes, blob.payload INTO stored_size, stored_payload
  FROM public.control_calphad_input_blobs blob
  WHERE blob.input_sha256 = revision_record.sha256;
  IF NOT FOUND OR stored_size <> revision_record.size_bytes OR
     octet_length(stored_payload) <> revision_record.size_bytes OR
     encode(sha256(stored_payload), 'hex') <> revision_record.sha256 THEN
    RAISE EXCEPTION 'CALPHAD_INPUT_RETENTION_REQUIRED: exact revision bytes are missing'
      USING ERRCODE = '23514';
  END IF;

  IF p_operation IN ('equilibrium', 'scheil') AND NOT EXISTS (
    SELECT 1
    FROM public.control_calphad_validation_events inspection
    JOIN public.control_calphad_evidence_blobs blob
      ON blob.evidence_sha256 = inspection.evidence_sha256
     AND blob.evidence_size_bytes = inspection.evidence_size_bytes
     AND octet_length(blob.payload) = inspection.evidence_size_bytes
     AND encode(sha256(blob.payload), 'hex') = inspection.evidence_sha256
    WHERE inspection.revision_id = revision_record.revision_id
      AND inspection.run_id = p_run_id
      AND inspection.operation = 'inspect'
      AND inspection.status = 'input_validated'
      AND inspection.runtime_image_id = p_runtime_image_id
      AND inspection.database_format = p_database_format
      AND inspection.database_inventory_sha256 = p_database_inventory_sha256
      AND inspection.assessment_pressure_min_pa = p_assessment_pressure_min_pa
      AND inspection.assessment_pressure_max_pa = p_assessment_pressure_max_pa
      AND inspection.evidence_contract_version = 'ultra.calphad.retained-evidence.v2'
      AND inspection.evidence_sha256 = p_inspection_evidence_sha256
  ) THEN
    RAISE EXCEPTION 'CALPHAD_INSPECTION_REQUIRED: exact retained inspection lineage is missing'
      USING ERRCODE = '23514';
  END IF;

  SELECT validation.* INTO existing_validation
  FROM public.control_calphad_validation_events validation
  WHERE validation.revision_id = revision_record.revision_id
    AND validation.run_id = p_run_id
    AND validation.operation = p_operation
    AND validation.evidence_sha256 = p_evidence_sha256;
  IF FOUND THEN
    IF existing_validation.resource_id <> p_resource_id OR
       existing_validation.database_sha256 <> p_database_sha256 OR
       existing_validation.database_size_bytes <> p_database_size_bytes OR
       existing_validation.database_format <> p_database_format OR
       existing_validation.assessment_pressure_min_pa <> p_assessment_pressure_min_pa OR
       existing_validation.assessment_pressure_max_pa <> p_assessment_pressure_max_pa OR
       existing_validation.database_inventory_sha256 IS DISTINCT FROM p_database_inventory_sha256 OR
       existing_validation.request_sha256 <> p_request_sha256 OR
       existing_validation.status <> p_status OR
       existing_validation.failure_domain IS DISTINCT FROM p_failure_domain OR
       existing_validation.failure_stage IS DISTINCT FROM p_failure_stage OR
       existing_validation.failure_code IS DISTINCT FROM p_failure_code OR
       existing_validation.evidence_path <> p_evidence_path OR
       existing_validation.evidence_size_bytes <> p_evidence_size_bytes OR
       existing_validation.runtime_image_id <> p_runtime_image_id OR
       existing_validation.pycalphad_version <> p_pycalphad_version OR
       existing_validation.inspection_evidence_sha256 IS DISTINCT FROM p_inspection_evidence_sha256 OR
       existing_validation.evidence_contract_version <> 'ultra.calphad.retained-evidence.v2' OR
       existing_validation.created_by_authority <> 'trusted_worker' THEN
      RAISE EXCEPTION 'CALPHAD_VALIDATION_CONFLICT: evidence identity was reused with different authority fields'
        USING ERRCODE = '23505';
    END IF;
    SELECT blob.evidence_size_bytes, blob.payload INTO stored_size, stored_payload
    FROM public.control_calphad_evidence_blobs blob
    WHERE blob.evidence_sha256 = p_evidence_sha256;
    IF NOT FOUND OR stored_size <> p_evidence_size_bytes OR stored_payload <> p_evidence_payload THEN
      RAISE EXCEPTION 'CALPHAD_EVIDENCE_RETENTION_REQUIRED: replay evidence bytes are missing or inconsistent'
        USING ERRCODE = '23514';
    END IF;
    IF lease_record.lease_expires_at <= clock_timestamp() THEN
      RAISE EXCEPTION 'CALPHAD_RUN_LEASE_INVALID: lease expired before replay completion'
        USING ERRCODE = '28000';
    END IF;
    RETURN QUERY SELECT existing_validation.validation_id, existing_validation.revision_id,
      existing_validation.resource_id, existing_validation.database_sha256,
      existing_validation.database_size_bytes, existing_validation.database_format,
      existing_validation.assessment_pressure_min_pa, existing_validation.assessment_pressure_max_pa,
      existing_validation.database_inventory_sha256, existing_validation.request_sha256,
      existing_validation.status, existing_validation.operation, existing_validation.failure_domain,
      existing_validation.failure_stage, existing_validation.failure_code, existing_validation.evidence_path,
      existing_validation.evidence_sha256, existing_validation.evidence_size_bytes,
      existing_validation.runtime_image_id, existing_validation.pycalphad_version,
      existing_validation.run_id, existing_validation.inspection_evidence_sha256,
      existing_validation.evidence_contract_version, existing_validation.created_by_authority,
      existing_validation.created_at, existing_validation.metadata, true;
    RETURN;
  END IF;

  INSERT INTO public.control_calphad_evidence_blobs
    (evidence_sha256, evidence_size_bytes, encoding, payload, created_at)
  VALUES (p_evidence_sha256, p_evidence_size_bytes, 'raw', p_evidence_payload, clock_timestamp())
  ON CONFLICT ON CONSTRAINT control_calphad_evidence_blobs_pkey DO NOTHING;
  SELECT blob.evidence_size_bytes, blob.payload INTO stored_size, stored_payload
  FROM public.control_calphad_evidence_blobs blob
  WHERE blob.evidence_sha256 = p_evidence_sha256;
  IF stored_size <> p_evidence_size_bytes OR stored_payload <> p_evidence_payload THEN
    RAISE EXCEPTION 'CALPHAD_EVIDENCE_RETENTION_REQUIRED: retained evidence conflicts with callback bytes'
      USING ERRCODE = '23514';
  END IF;

  SELECT capacity.* INTO capacity_record
  FROM public.control_calphad_tenant_capacity capacity
  WHERE capacity.owner_user_id = btrim(revision_record.owner_user_id)
    AND capacity.owner_org_id = COALESCE(NULLIF(btrim(revision_record.owner_org_id), ''), '')
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'CALPHAD_CAPACITY_STATE_INVALID: tenant capacity row is missing'
      USING ERRCODE = '23514';
  END IF;
  UPDATE public.control_calphad_tenant_capacity capacity
  SET retained_evidence_bytes = capacity.retained_evidence_bytes + p_evidence_size_bytes,
      validation_events = capacity.validation_events + 1,
      updated_at = clock_timestamp()
  WHERE capacity.owner_user_id = capacity_record.owner_user_id
    AND capacity.owner_org_id = capacity_record.owner_org_id
    AND capacity.retained_input_bytes::numeric + capacity.retained_evidence_bytes::numeric +
        p_evidence_size_bytes::numeric <= capacity.max_retained_bytes::numeric
    AND capacity.validation_events < capacity.max_validation_events
  RETURNING 1 INTO capacity_updated;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'CALPHAD_TENANT_CAPACITY_EXCEEDED: retained-byte or validation-event capacity exhausted'
      USING ERRCODE = '23514';
  END IF;

  IF run_record.status <> 'running' OR lease_record.lease_expires_at <= clock_timestamp() OR
     lease_record.worker_id <> p_lease_worker_id OR
     sha256(convert_to(lease_record.lease_token, 'UTF8')) <>
       sha256(convert_to(p_lease_token, 'UTF8')) THEN
    RAISE EXCEPTION 'CALPHAD_RUN_LEASE_INVALID: lease expired or changed before event insert'
      USING ERRCODE = '28000';
  END IF;

  new_validation_id := 'calphad_validation_' || replace(gen_random_uuid()::text, '-', '');
  created_timestamp := clock_timestamp();
  validation_metadata := jsonb_build_object(
      'server_managed', true,
      'revision_id', revision_record.revision_id,
      'assessment_pressure_limits_Pa',
        jsonb_build_array(p_assessment_pressure_min_pa, p_assessment_pressure_max_pa)
    );
  INSERT INTO public.control_calphad_validation_events
    (validation_id, revision_id, resource_id, database_sha256, database_size_bytes,
     database_format, assessment_pressure_min_pa, assessment_pressure_max_pa,
     database_inventory_sha256, request_sha256, status, operation,
     failure_domain, failure_stage, failure_code, evidence_path,
     evidence_sha256, evidence_size_bytes, runtime_image_id, pycalphad_version, run_id,
     inspection_evidence_sha256, evidence_contract_version, created_by_authority,
     created_at, metadata)
  VALUES (new_validation_id, revision_record.revision_id, p_resource_id,
          p_database_sha256, p_database_size_bytes, p_database_format,
          p_assessment_pressure_min_pa, p_assessment_pressure_max_pa,
          p_database_inventory_sha256, p_request_sha256, p_status, p_operation,
          p_failure_domain, p_failure_stage, p_failure_code, p_evidence_path,
          p_evidence_sha256, p_evidence_size_bytes, p_runtime_image_id,
          p_pycalphad_version, p_run_id, p_inspection_evidence_sha256,
          'ultra.calphad.retained-evidence.v2', 'trusted_worker',
          created_timestamp, validation_metadata)
  ON CONFLICT DO NOTHING
  RETURNING * INTO inserted_validation;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'CALPHAD_VALIDATION_CONFLICT: concurrent evidence identity conflict'
      USING ERRCODE = '23505';
  END IF;

  RETURN QUERY SELECT inserted_validation.validation_id, inserted_validation.revision_id,
    inserted_validation.resource_id, inserted_validation.database_sha256,
    inserted_validation.database_size_bytes, inserted_validation.database_format,
    inserted_validation.assessment_pressure_min_pa, inserted_validation.assessment_pressure_max_pa,
    inserted_validation.database_inventory_sha256, inserted_validation.request_sha256,
    inserted_validation.status, inserted_validation.operation, inserted_validation.failure_domain,
    inserted_validation.failure_stage, inserted_validation.failure_code, inserted_validation.evidence_path,
    inserted_validation.evidence_sha256, inserted_validation.evidence_size_bytes,
    inserted_validation.runtime_image_id, inserted_validation.pycalphad_version,
    inserted_validation.run_id, inserted_validation.inspection_evidence_sha256,
    inserted_validation.evidence_contract_version, inserted_validation.created_by_authority,
    inserted_validation.created_at, inserted_validation.metadata, true;
END;
$$;

WITH revision_usage AS (
  SELECT btrim(owner_user_id) AS owner_user_id,
         COALESCE(NULLIF(btrim(owner_org_id), ''), '') AS owner_org_id,
         sum(size_bytes)::bigint AS retained_input_bytes
  FROM public.control_calphad_revisions
  GROUP BY btrim(owner_user_id), COALESCE(NULLIF(btrim(owner_org_id), ''), '')
), event_usage AS (
  SELECT btrim(revision.owner_user_id) AS owner_user_id,
         COALESCE(NULLIF(btrim(revision.owner_org_id), ''), '') AS owner_org_id,
         COALESCE(sum(validation.evidence_size_bytes), 0)::bigint AS retained_evidence_bytes,
         count(*)::bigint AS validation_events
  FROM public.control_calphad_validation_events validation
  JOIN public.control_calphad_revisions revision
    ON revision.revision_id = validation.revision_id
  GROUP BY btrim(revision.owner_user_id), COALESCE(NULLIF(btrim(revision.owner_org_id), ''), '')
), usage AS (
  SELECT COALESCE(revision_usage.owner_user_id, event_usage.owner_user_id) AS owner_user_id,
         COALESCE(revision_usage.owner_org_id, event_usage.owner_org_id) AS owner_org_id,
         COALESCE(revision_usage.retained_input_bytes, 0) AS retained_input_bytes,
         COALESCE(event_usage.retained_evidence_bytes, 0) AS retained_evidence_bytes,
         COALESCE(event_usage.validation_events, 0) AS validation_events
  FROM revision_usage
  FULL OUTER JOIN event_usage USING (owner_user_id, owner_org_id)
)
INSERT INTO public.control_calphad_tenant_capacity
 (owner_user_id, owner_org_id, max_retained_bytes, max_validation_events,
  retained_input_bytes, retained_evidence_bytes, validation_events, updated_at)
SELECT owner_user_id, owner_org_id,
       GREATEST(4294967296::bigint, retained_input_bytes + retained_evidence_bytes),
       GREATEST(100000::bigint, validation_events),
       retained_input_bytes, retained_evidence_bytes, validation_events, clock_timestamp()
FROM usage
ON CONFLICT (owner_user_id, owner_org_id) DO UPDATE
SET retained_input_bytes = EXCLUDED.retained_input_bytes,
    retained_evidence_bytes = EXCLUDED.retained_evidence_bytes,
    validation_events = EXCLUDED.validation_events,
    max_retained_bytes = GREATEST(
      public.control_calphad_tenant_capacity.max_retained_bytes,
      EXCLUDED.retained_input_bytes + EXCLUDED.retained_evidence_bytes
    ),
    max_validation_events = GREATEST(
      public.control_calphad_tenant_capacity.max_validation_events,
      EXCLUDED.validation_events
    ),
    updated_at = EXCLUDED.updated_at;

CREATE OR REPLACE FUNCTION public.ultra_create_calphad_revision_v1(
  p_resource_id text,
  p_owner_user_id text,
  p_owner_org_id text,
  p_parent_revision_id text,
  p_expected_sha256 text,
  p_expected_size_bytes bigint,
  p_database_format text,
  p_assessment_pressure_min_pa double precision,
  p_assessment_pressure_max_pa double precision,
  p_input_payload bytea,
  p_metadata jsonb
)
RETURNS TABLE (
  revision_id text,
  resource_id text,
  owner_user_id text,
  owner_org_id text,
  sha256 text,
  size_bytes bigint,
  database_format text,
  assessment_pressure_min_pa double precision,
  assessment_pressure_max_pa double precision,
  parent_revision_id text,
  created_by_user_id text,
  created_at timestamptz,
  metadata jsonb
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog
AS $$
DECLARE
  resource_record public.control_resources%ROWTYPE;
  existing_revision public.control_calphad_revisions%ROWTYPE;
  parent_revision public.control_calphad_revisions%ROWTYPE;
  inserted_revision public.control_calphad_revisions%ROWTYPE;
  capacity_updated bigint;
  stored_size bigint;
  stored_payload bytea;
  normalized_org text;
  normalized_parent text;
  derived_format text;
  calphad_metadata jsonb;
  declared_temperature jsonb;
  expected_declaration jsonb;
  supplied_declaration jsonb;
  revision_metadata jsonb;
  created_timestamp timestamptz;
  new_revision_id text;
  new_validation_id text;
BEGIN
  p_resource_id := btrim(p_resource_id);
  p_owner_user_id := btrim(p_owner_user_id);
  normalized_org := COALESCE(NULLIF(btrim(p_owner_org_id), ''), '');
  normalized_parent := NULLIF(btrim(p_parent_revision_id), '');
  p_expected_sha256 := lower(btrim(p_expected_sha256));
  p_database_format := btrim(p_database_format);

  IF p_resource_id = '' OR p_owner_user_id = '' OR
     p_expected_sha256 !~ '^[0-9a-f]{64}$' OR
     p_expected_size_bytes NOT BETWEEN 1 AND 67108864 OR
     p_database_format NOT IN ('tdb', 'dat') OR
     p_assessment_pressure_min_pa < 1e-9 OR
     p_assessment_pressure_max_pa > 1e12 OR
     p_assessment_pressure_min_pa > p_assessment_pressure_max_pa OR
     p_input_payload IS NULL OR octet_length(p_input_payload) <> p_expected_size_bytes OR
     encode(sha256(p_input_payload), 'hex') <> p_expected_sha256 OR
     jsonb_typeof(p_metadata) <> 'object' OR
     octet_length(convert_to(p_metadata::text, 'UTF8')) > 65536 THEN
    RAISE EXCEPTION 'CALPHAD_REVISION_BINDING_INVALID: revision request is not canonical or content-bound'
      USING ERRCODE = '23514';
  END IF;

  SELECT resource.* INTO resource_record
  FROM public.control_resources resource
  WHERE resource.resource_id = p_resource_id
    AND resource.owner_user_id = p_owner_user_id
    AND (COALESCE(NULLIF(btrim(resource.owner_org_id), ''), '') = '' OR
         COALESCE(NULLIF(btrim(resource.owner_org_id), ''), '') = normalized_org)
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'CALPHAD_RESOURCE_NOT_FOUND' USING ERRCODE = 'P0002';
  END IF;

  derived_format := CASE
    WHEN lower(btrim(resource_record.original_name)) ~ '\.tdb$' THEN 'tdb'
    WHEN lower(btrim(resource_record.original_name)) ~ '\.dat$' THEN 'dat'
    ELSE ''
  END;
  IF btrim(resource_record.status) <> 'active' OR
     derived_format = '' OR derived_format <> p_database_format OR
     lower(btrim(COALESCE(resource_record.content_type, ''))) NOT IN
       ('', 'application/octet-stream', 'text/plain', 'application/x-thermocalc-tdb') OR
     lower(btrim(resource_record.sha256)) <> p_expected_sha256 OR
     resource_record.size_bytes <> p_expected_size_bytes THEN
    RAISE EXCEPTION 'CALPHAD_RESOURCE_BINDING_INVALID: live catalog binding changed or format is unsupported'
      USING ERRCODE = '23514';
  END IF;

  calphad_metadata := resource_record.metadata->'calphad';
  declared_temperature := COALESCE(
    calphad_metadata->'assessment_temperature_limits_K',
    calphad_metadata->'tdb_temperature_limits_K'
  );
  IF jsonb_typeof(calphad_metadata) <> 'object' OR
     (calphad_metadata ? 'assessment_temperature_limits_K' AND
      calphad_metadata ? 'tdb_temperature_limits_K' AND
      calphad_metadata->'assessment_temperature_limits_K' <>
        calphad_metadata->'tdb_temperature_limits_K') THEN
    RAISE EXCEPTION 'CALPHAD_OWNER_DECLARATION_INVALID: owner declaration is missing or contradictory'
      USING ERRCODE = '23514';
  END IF;
  expected_declaration := jsonb_build_object(
    'schema_version', 'ultra.calphad.owner-declaration.v1',
    'authority', 'resource_owner',
    'database_id', COALESCE(NULLIF(btrim(calphad_metadata->>'database_id'), ''), p_resource_id),
    'source', btrim(calphad_metadata->>'source'),
    'license_id', COALESCE(NULLIF(btrim(calphad_metadata->>'license_id'), ''),
                           btrim(calphad_metadata->>'license_identifier')),
    'assessment_scope', btrim(calphad_metadata->>'assessment_scope'),
    'reference_state', btrim(calphad_metadata->>'reference_state'),
    'assessment_temperature_limits_K', declared_temperature,
    'assessment_pressure_limits_Pa', calphad_metadata->'assessment_pressure_limits_Pa',
    'database_format', derived_format
  );
  supplied_declaration := p_metadata->'owner_declaration';
  IF supplied_declaration IS NULL OR supplied_declaration <> expected_declaration OR
     supplied_declaration->'assessment_pressure_limits_Pa' <>
       jsonb_build_array(p_assessment_pressure_min_pa, p_assessment_pressure_max_pa) THEN
    RAISE EXCEPTION 'CALPHAD_OWNER_DECLARATION_INVALID: immutable owner declaration does not match the live resource'
      USING ERRCODE = '23514';
  END IF;
  revision_metadata := jsonb_build_object(
      'server_managed', true,
      'owner_declaration', expected_declaration,
      'assessment_pressure_limits_Pa',
        jsonb_build_array(p_assessment_pressure_min_pa, p_assessment_pressure_max_pa)
    );

  SELECT revision.* INTO existing_revision
  FROM public.control_calphad_revisions revision
  WHERE revision.resource_id = p_resource_id;
  IF FOUND THEN
    IF existing_revision.owner_user_id <> resource_record.owner_user_id OR
       COALESCE(NULLIF(btrim(existing_revision.owner_org_id), ''), '') <>
         COALESCE(NULLIF(btrim(resource_record.owner_org_id), ''), '') OR
       existing_revision.sha256 <> p_expected_sha256 OR
       existing_revision.size_bytes <> p_expected_size_bytes OR
       existing_revision.database_format <> p_database_format OR
       existing_revision.assessment_pressure_min_pa <> p_assessment_pressure_min_pa OR
       existing_revision.assessment_pressure_max_pa <> p_assessment_pressure_max_pa OR
       existing_revision.metadata->'owner_declaration' <> expected_declaration OR
       existing_revision.parent_revision_id IS DISTINCT FROM normalized_parent THEN
      RAISE EXCEPTION 'CALPHAD_REVISION_CONFLICT: existing revision differs from the immutable request'
        USING ERRCODE = '23505';
    END IF;
    INSERT INTO public.control_calphad_input_blobs
      (input_sha256, input_size_bytes, encoding, payload, created_at)
    VALUES (p_expected_sha256, p_expected_size_bytes, 'raw', p_input_payload, clock_timestamp())
    ON CONFLICT (input_sha256) DO NOTHING;
    SELECT blob.input_size_bytes, blob.payload INTO stored_size, stored_payload
    FROM public.control_calphad_input_blobs blob
    WHERE blob.input_sha256 = p_expected_sha256;
    IF stored_size <> p_expected_size_bytes OR stored_payload <> p_input_payload THEN
      RAISE EXCEPTION 'CALPHAD_INPUT_RETENTION_REQUIRED: retained input conflicts with the request'
        USING ERRCODE = '23514';
    END IF;
    RETURN QUERY SELECT existing_revision.revision_id, existing_revision.resource_id,
      existing_revision.owner_user_id, existing_revision.owner_org_id,
      existing_revision.sha256, existing_revision.size_bytes, existing_revision.database_format,
      existing_revision.assessment_pressure_min_pa, existing_revision.assessment_pressure_max_pa,
      existing_revision.parent_revision_id, existing_revision.created_by_user_id,
      existing_revision.created_at, existing_revision.metadata;
    RETURN;
  END IF;

  IF normalized_parent IS NOT NULL THEN
    SELECT revision.* INTO parent_revision
    FROM public.control_calphad_revisions revision
    WHERE revision.revision_id = normalized_parent;
    IF NOT FOUND OR parent_revision.resource_id = p_resource_id OR
       parent_revision.owner_user_id <> resource_record.owner_user_id OR
       COALESCE(NULLIF(btrim(parent_revision.owner_org_id), ''), '') <>
         COALESCE(NULLIF(btrim(resource_record.owner_org_id), ''), '') THEN
      RAISE EXCEPTION 'CALPHAD_PARENT_NOT_FOUND' USING ERRCODE = 'P0002';
    END IF;
  END IF;

  INSERT INTO public.control_calphad_input_blobs
    (input_sha256, input_size_bytes, encoding, payload, created_at)
  VALUES (p_expected_sha256, p_expected_size_bytes, 'raw', p_input_payload, clock_timestamp())
  ON CONFLICT (input_sha256) DO NOTHING;
  SELECT blob.input_size_bytes, blob.payload INTO stored_size, stored_payload
  FROM public.control_calphad_input_blobs blob
  WHERE blob.input_sha256 = p_expected_sha256;
  IF stored_size <> p_expected_size_bytes OR stored_payload <> p_input_payload THEN
    RAISE EXCEPTION 'CALPHAD_INPUT_RETENTION_REQUIRED: retained input conflicts with the request'
      USING ERRCODE = '23514';
  END IF;

  INSERT INTO public.control_calphad_tenant_capacity
    (owner_user_id, owner_org_id, updated_at)
  VALUES (btrim(resource_record.owner_user_id),
          COALESCE(NULLIF(btrim(resource_record.owner_org_id), ''), ''), clock_timestamp())
  ON CONFLICT ON CONSTRAINT control_calphad_tenant_capacity_pkey DO NOTHING;
  UPDATE public.control_calphad_tenant_capacity capacity
  SET retained_input_bytes = capacity.retained_input_bytes + p_expected_size_bytes,
      validation_events = capacity.validation_events + 1,
      updated_at = clock_timestamp()
  WHERE capacity.owner_user_id = btrim(resource_record.owner_user_id)
    AND capacity.owner_org_id = COALESCE(NULLIF(btrim(resource_record.owner_org_id), ''), '')
    AND capacity.retained_input_bytes::numeric + capacity.retained_evidence_bytes::numeric +
        p_expected_size_bytes::numeric <= capacity.max_retained_bytes::numeric
    AND capacity.validation_events < capacity.max_validation_events
  RETURNING 1 INTO capacity_updated;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'CALPHAD_TENANT_CAPACITY_EXCEEDED: retained-byte or validation-event capacity exhausted'
      USING ERRCODE = '23514';
  END IF;

  created_timestamp := clock_timestamp();
  new_revision_id := 'calphad_revision_' || replace(gen_random_uuid()::text, '-', '');
  new_validation_id := 'calphad_validation_' || replace(gen_random_uuid()::text, '-', '');
  INSERT INTO public.control_calphad_revisions
    (revision_id, resource_id, owner_user_id, owner_org_id, sha256, size_bytes,
     database_format, assessment_pressure_min_pa, assessment_pressure_max_pa,
     parent_revision_id, created_by_user_id, created_at, metadata)
  VALUES (new_revision_id, p_resource_id, resource_record.owner_user_id,
          NULLIF(COALESCE(NULLIF(btrim(resource_record.owner_org_id), ''), ''), ''),
          p_expected_sha256, p_expected_size_bytes, p_database_format,
          p_assessment_pressure_min_pa, p_assessment_pressure_max_pa,
          normalized_parent, p_owner_user_id, created_timestamp, revision_metadata)
  RETURNING * INTO inserted_revision;

  INSERT INTO public.control_calphad_validation_events
    (validation_id, revision_id, resource_id, database_sha256, database_size_bytes,
     database_format, assessment_pressure_min_pa, assessment_pressure_max_pa,
     status, operation, created_by_authority, created_at, metadata)
  VALUES (new_validation_id, new_revision_id, p_resource_id, p_expected_sha256,
          p_expected_size_bytes, p_database_format, p_assessment_pressure_min_pa,
          p_assessment_pressure_max_pa, 'pending', 'registration', 'control_plane',
          created_timestamp, jsonb_build_object(
            'server_managed', true,
            'assessment_pressure_limits_Pa',
              jsonb_build_array(p_assessment_pressure_min_pa, p_assessment_pressure_max_pa)
          ));

  RETURN QUERY SELECT inserted_revision.revision_id, inserted_revision.resource_id,
    inserted_revision.owner_user_id, inserted_revision.owner_org_id,
    inserted_revision.sha256, inserted_revision.size_bytes, inserted_revision.database_format,
    inserted_revision.assessment_pressure_min_pa, inserted_revision.assessment_pressure_max_pa,
    inserted_revision.parent_revision_id, inserted_revision.created_by_user_id,
    inserted_revision.created_at, inserted_revision.metadata;
END;
$$;

REVOKE ALL ON FUNCTION public.ultra_create_calphad_revision_v1(
  text, text, text, text, text, bigint, text, double precision,
  double precision, bytea, jsonb
) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.ultra_append_calphad_validation_v1(
  text, text, text, text, bigint, text, jsonb, double precision,
  double precision, text, text, text, text, text, text, text, text,
  text, bigint, bytea, text, text, text, text, text, text, jsonb
) FROM PUBLIC;

REVOKE EXECUTE ON FUNCTION public.ultra_validate_calphad_revision_parent() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION public.ultra_validate_calphad_validation_run_authority() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION public.ultra_validate_calphad_pressure_binding() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION public.ultra_validate_calphad_input_retention() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION public.ultra_validate_calphad_equilibrium_inspection_lineage() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION public.ultra_reject_calphad_ledger_mutation() FROM PUBLIC;
