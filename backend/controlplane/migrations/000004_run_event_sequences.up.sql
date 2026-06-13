-- Per-run event sequence allocator: appends serialize on this row's lock and
-- allocate the next sequence in a single statement, replacing the advisory
-- lock + MAX() multi-statement transaction (5 round trips -> 1).
CREATE TABLE IF NOT EXISTS control_run_event_sequences (
  run_id text PRIMARY KEY REFERENCES control_runs(run_id) ON DELETE CASCADE,
  last_sequence bigint NOT NULL
);

-- Seed counters for runs that already have events. The allocator also
-- self-heals (it never issues a sequence at or below the events table's
-- current MAX), so this backfill is an optimization, not a correctness
-- requirement.
INSERT INTO control_run_event_sequences (run_id, last_sequence)
SELECT run_id, MAX(sequence_number)
FROM control_run_events
GROUP BY run_id
ON CONFLICT (run_id) DO NOTHING;
