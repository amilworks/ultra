-- Let a conversation be erased without destroying the append-only audit ledgers.
--
-- Conversation delete is true erasure: it deletes the control_threads row and
-- lets the schema cascades take messages, runs, events and artifacts with it.
-- A family of ledger tables made that impossible for ~10% of conversations
-- (19 of 189 on the development database), in a way no amount of reading
-- schema.sql would reveal — none of these tables is declared there.
--
-- Each one is caught in the same pincer:
--
--   * it references control_runs(run_id) or control_threads(thread_id) with
--     NO ACTION, so the parent cannot be deleted while its row exists...
--   * ...and an *_append_only trigger raises on any DELETE, so the row cannot
--     be removed first.
--
-- Net effect: DELETE /v2/threads/{id} aborted with a foreign-key violation and
-- the user's delete failed. Deleting a thread whose runs carried a run spec was
-- simply impossible.
--
-- Both guarantees are deliberate and both are kept. These ledgers record SYSTEM
-- behaviour — admission decisions, capacity and quota reservations, authority
-- snapshots, knowledge queries, validation events — not user content, so the
-- rows survive. What goes is their referential grip on the conversation.
--
-- Every column below is NOT NULL (most are primary keys), so ON DELETE SET NULL
-- is not available and the foreign key is dropped outright. The column keeps its
-- id as plain text: still forensically useful, since it names a run or thread
-- that once existed, but no longer able to block that row from being erased.
-- control_calphad_validation_events.run_id is the one nullable case, so it keeps
-- a real foreign key and degrades to NULL — the honest representation, because
-- the run really is gone.
--
-- Deliberately NOT touched: the append-only triggers themselves. Weakening an
-- immutability guarantee to make a delete convenient is the wrong trade, and
-- these rows are not what the user asked to erase.
--
-- To find any that appear later:
--
--   SELECT src.relname, a.attname FROM pg_constraint c
--   JOIN pg_class src ON src.oid = c.conrelid
--   JOIN pg_class tgt ON tgt.oid = c.confrelid
--   JOIN unnest(c.conkey) AS k(attnum) ON true
--   JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.attnum
--   WHERE c.contype='f' AND c.confdeltype IN ('a','r')
--     AND tgt.relname IN ('control_runs','control_threads');
--
-- TestHardDeleteSweepsEveryNonCascadingReferencer runs exactly that query.

-- Nullable: keep the foreign key, degrade the link.
ALTER TABLE control_calphad_validation_events
  DROP CONSTRAINT IF EXISTS control_calphad_validation_run_fkey;
ALTER TABLE control_calphad_validation_events
  ADD CONSTRAINT control_calphad_validation_run_fkey
  FOREIGN KEY (run_id) REFERENCES control_runs(run_id) ON DELETE SET NULL;

-- NOT NULL: drop the foreign key, keep the value.
ALTER TABLE control_run_specs
  DROP CONSTRAINT IF EXISTS control_run_specs_run_id_fkey;

ALTER TABLE control_knowledge_query_events
  DROP CONSTRAINT IF EXISTS control_knowledge_queries_run_fkey;

ALTER TABLE control_ultra_admission_capacity_reservations
  DROP CONSTRAINT IF EXISTS control_ultra_admission_capacity_reservations_run_id_fkey;
ALTER TABLE control_ultra_admission_capacity_reservations
  DROP CONSTRAINT IF EXISTS control_ultra_admission_capacity_reservations_thread_id_fkey;

ALTER TABLE control_ultra_admission_quota_reservations
  DROP CONSTRAINT IF EXISTS control_ultra_admission_quota_reservations_run_id_fkey;
ALTER TABLE control_ultra_admission_quota_reservations
  DROP CONSTRAINT IF EXISTS control_ultra_admission_quota_reservations_thread_id_fkey;

ALTER TABLE control_ultra_admission_quotes
  DROP CONSTRAINT IF EXISTS control_ultra_admission_quotes_thread_id_fkey;

ALTER TABLE control_ultra_admission_reservation_releases
  DROP CONSTRAINT IF EXISTS control_ultra_admission_reservation_releases_run_id_fkey;

ALTER TABLE control_ultra_live_authority_snapshots
  DROP CONSTRAINT IF EXISTS control_ultra_live_authority_snapshots_thread_id_fkey;

ALTER TABLE control_ultra_roots
  DROP CONSTRAINT IF EXISTS control_ultra_roots_thread_id_fkey;
