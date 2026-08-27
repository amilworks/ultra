DROP TABLE IF EXISTS control_note_direct_append_operations;
DROP TRIGGER IF EXISTS control_notes_create_receipt_tombstone ON control_notes;
DROP FUNCTION IF EXISTS ultra_control_note_create_receipt_tombstone();
DROP TABLE IF EXISTS control_note_create_receipts;
DROP TRIGGER IF EXISTS control_notes_content_recency ON control_notes;
DROP FUNCTION IF EXISTS ultra_control_note_content_recency();
ALTER TABLE control_notes DROP COLUMN IF EXISTS content_updated_at;
