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

CREATE INDEX IF NOT EXISTS control_user_token_usage_daily_user_day_idx ON control_user_token_usage_daily(user_id, day DESC);
CREATE INDEX IF NOT EXISTS control_run_token_usage_user_day_idx ON control_run_token_usage(user_id, day DESC);
CREATE INDEX IF NOT EXISTS control_run_token_usage_run_idx ON control_run_token_usage(run_id);
