DROP TABLE IF EXISTS control_note_append_operations;
DROP TABLE IF EXISTS control_note_append_proposals;
DROP TABLE IF EXISTS control_note_read_grants;
DROP TABLE IF EXISTS control_note_run_usage;
ALTER TABLE control_notes DROP COLUMN IF EXISTS content_digest;
ALTER TABLE control_notes DROP COLUMN IF EXISTS revision;

-- Preserve deepagents_checkpoint_threads and its run cascade on rollback. It is
-- runtime durability and conversation-erasure infrastructure, not Note content,
-- and dropping either the rows or their ownership boundary would be unsafe.
