DROP INDEX IF EXISTS control_run_events_run_source_sequence_idx;

ALTER TABLE control_run_events
  DROP COLUMN IF EXISTS source_sequence;
