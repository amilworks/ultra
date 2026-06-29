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
