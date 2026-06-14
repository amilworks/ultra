-- ListThreadMessages / ListThreadMessagesForUser filter on thread_id and sort
-- by created_at; without this index every call seq-scans the whole table.
-- The table is small enough today that a plain (locking) CREATE INDEX is fine;
-- on a large live deployment run CREATE INDEX CONCURRENTLY manually instead.
CREATE INDEX IF NOT EXISTS control_thread_messages_thread_created_idx
  ON control_thread_messages(thread_id, created_at);
