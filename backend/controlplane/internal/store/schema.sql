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

CREATE INDEX IF NOT EXISTS control_runs_user_status_updated_idx ON control_runs(user_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS control_runs_thread_status_updated_idx ON control_runs(thread_id, status, updated_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS control_runs_idempotency_unique_idx
  ON control_runs(thread_id, user_id, (metadata->>'idempotency_key'))
  WHERE COALESCE(metadata->>'idempotency_key', '') <> '';
CREATE INDEX IF NOT EXISTS control_run_events_run_sequence_idx ON control_run_events(run_id, sequence_number);
CREATE INDEX IF NOT EXISTS control_run_events_run_event_idx ON control_run_events(run_id, event_id);
CREATE INDEX IF NOT EXISTS control_run_leases_expires_idx ON control_run_leases(lease_expires_at);
CREATE INDEX IF NOT EXISTS control_worker_heartbeats_kind_status_idx ON control_worker_heartbeats(worker_kind, status, last_heartbeat_at DESC);
CREATE INDEX IF NOT EXISTS control_worker_heartbeats_last_seen_idx ON control_worker_heartbeats(last_heartbeat_at DESC);
CREATE INDEX IF NOT EXISTS control_artifacts_run_created_idx ON control_artifacts(run_id, created_at DESC);
CREATE INDEX IF NOT EXISTS control_artifacts_sha_idx ON control_artifacts(sha256);
CREATE INDEX IF NOT EXISTS control_organizations_status_updated_idx ON control_organizations(status, updated_at DESC);
CREATE INDEX IF NOT EXISTS control_users_org_status_idx ON control_users(org_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS control_users_email_idx ON control_users(lower(email));
CREATE INDEX IF NOT EXISTS control_bisque_credentials_user_status_idx ON control_bisque_credentials(user_id, org_id, status, updated_at DESC);
