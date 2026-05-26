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

CREATE INDEX IF NOT EXISTS control_runs_user_status_updated_idx ON control_runs(user_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS control_runs_thread_status_updated_idx ON control_runs(thread_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS control_run_events_run_sequence_idx ON control_run_events(run_id, sequence_number);
CREATE INDEX IF NOT EXISTS control_run_events_run_event_idx ON control_run_events(run_id, event_id);
CREATE INDEX IF NOT EXISTS control_artifacts_run_created_idx ON control_artifacts(run_id, created_at DESC);
CREATE INDEX IF NOT EXISTS control_artifacts_sha_idx ON control_artifacts(sha256);
