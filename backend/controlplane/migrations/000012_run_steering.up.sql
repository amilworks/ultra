-- Mid-run steering (Phase 1): durable steering messages + the finalization
-- barrier. Deploy record — the canonical schema is store/schema.sql, applied
-- idempotently by `ultra-control migrate`.
CREATE TABLE IF NOT EXISTS control_run_steer_messages (
  steer_id text PRIMARY KEY,
  run_id text NOT NULL REFERENCES control_runs(run_id) ON DELETE CASCADE,
  thread_id text NOT NULL,
  user_id text NOT NULL,
  message_id text NOT NULL,
  content text NOT NULL,
  status text NOT NULL DEFAULT 'pending',
  created_at timestamptz NOT NULL,
  applied_at timestamptz,
  updated_at timestamptz NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_control_run_steer_messages_run
  ON control_run_steer_messages(run_id);

CREATE TABLE IF NOT EXISTS control_run_steer_barriers (
  run_id text PRIMARY KEY REFERENCES control_runs(run_id) ON DELETE CASCADE,
  closed_at timestamptz NOT NULL
);
