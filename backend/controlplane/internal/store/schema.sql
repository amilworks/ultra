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

-- Mid-run steering (Phase 1). One row per accepted steering message. The
-- message_id references the steer's control_thread_messages row AND doubles as
-- the LangGraph message id, so every copy of the steer — middleware-injected,
-- requeue-seeded, continuation-appended — collapses to a single graph message
-- via the add_messages id-upsert.
CREATE TABLE IF NOT EXISTS control_run_steer_messages (
  steer_id text PRIMARY KEY,
  run_id text NOT NULL REFERENCES control_runs(run_id) ON DELETE CASCADE,
  thread_id text NOT NULL,
  user_id text NOT NULL,
  message_id text NOT NULL,
  content text NOT NULL,
  file_ids jsonb NOT NULL DEFAULT '[]'::jsonb,
  status text NOT NULL DEFAULT 'pending',
  created_at timestamptz NOT NULL,
  applied_at timestamptz,
  updated_at timestamptz NOT NULL
);

ALTER TABLE control_run_steer_messages
  ADD COLUMN IF NOT EXISTS file_ids jsonb NOT NULL DEFAULT '[]'::jsonb;

CREATE INDEX IF NOT EXISTS idx_control_run_steer_messages_run
  ON control_run_steer_messages(run_id);

-- A row here means the run is finalizing: the worker closed the steer barrier
-- and no further steers are accepted (clients fall back to Phase 0 queueing).
-- Recovery requeue deletes the row so a fresh attempt accepts steers again.
CREATE TABLE IF NOT EXISTS control_run_steer_barriers (
  run_id text PRIMARY KEY REFERENCES control_runs(run_id) ON DELETE CASCADE,
  closed_at timestamptz NOT NULL
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
CREATE INDEX IF NOT EXISTS control_resource_collections_parent_idx ON control_resource_collections(parent_collection_id, status);
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
