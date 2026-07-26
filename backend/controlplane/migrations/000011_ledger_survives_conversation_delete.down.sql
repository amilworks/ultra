-- Restore the NO ACTION foreign keys.
--
-- This will FAIL if any conversation has been erased since the up migration
-- ran: ledger rows whose run_id was set to NULL are fine (the FK permits NULL),
-- but control_run_specs rows now pointing at a deleted run cannot satisfy a
-- re-added foreign key. That is not a flaw in this migration — it is the
-- unavoidable consequence of having erased data, and it is exactly why the
-- forward direction exists.
--
-- If this must be run after erasures have happened, delete the dangling
-- control_run_specs rows first (they name runs that no longer exist), accepting
-- that doing so requires disabling control_run_specs_append_only, which is a
-- deliberate act someone should have to think about.

ALTER TABLE control_calphad_validation_events
  DROP CONSTRAINT IF EXISTS control_calphad_validation_run_fkey;

ALTER TABLE control_calphad_validation_events
  ADD CONSTRAINT control_calphad_validation_run_fkey
  FOREIGN KEY (run_id) REFERENCES control_runs(run_id);

ALTER TABLE control_run_specs
  ADD CONSTRAINT control_run_specs_run_id_fkey
  FOREIGN KEY (run_id) REFERENCES control_runs(run_id);
