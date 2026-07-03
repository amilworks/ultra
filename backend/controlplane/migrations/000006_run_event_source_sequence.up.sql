ALTER TABLE control_run_events
  ADD COLUMN IF NOT EXISTS source_sequence bigint;

UPDATE control_run_events
SET source_sequence = sequence_number
WHERE source_sequence IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS control_run_events_run_source_sequence_idx
  ON control_run_events(run_id, source_sequence)
  WHERE source_sequence IS NOT NULL;
