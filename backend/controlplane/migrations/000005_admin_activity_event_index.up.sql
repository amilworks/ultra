-- Partial index backing the admin dashboard's tool-call/artifact activity
-- aggregate; only events of these kinds pay the write cost.
CREATE INDEX IF NOT EXISTS control_run_events_admin_activity_idx ON control_run_events(event_kind, ts)
  WHERE event_kind IN ('tool_call.started', 'artifact.created');
