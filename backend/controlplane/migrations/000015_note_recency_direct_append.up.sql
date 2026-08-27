DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'control_notes'
      AND column_name = 'content_updated_at'
  ) THEN
    ALTER TABLE control_notes ADD COLUMN content_updated_at timestamptz;
    ALTER TABLE control_notes ALTER COLUMN content_updated_at SET DEFAULT now();
  ELSIF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'control_notes'
      AND column_name = 'content_updated_at'
      AND column_default IS NOT NULL
  ) THEN
    ALTER TABLE control_notes ALTER COLUMN content_updated_at SET DEFAULT now();
  END IF;
END $$;

DO $$
BEGIN
  IF to_regprocedure('public.ultra_control_note_content_recency()') IS NULL THEN
    EXECUTE $function$
      CREATE FUNCTION ultra_control_note_content_recency()
      RETURNS trigger
      LANGUAGE plpgsql
      AS $body$
      BEGIN
        IF NEW.title IS DISTINCT FROM OLD.title
           OR NEW.body_markdown IS DISTINCT FROM OLD.body_markdown THEN
          NEW.content_updated_at := now();
        ELSE
          NEW.content_updated_at := COALESCE(OLD.content_updated_at, OLD.updated_at);
        END IF;
        RETURN NEW;
      END;
      $body$
    $function$;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_trigger
    WHERE tgrelid = 'control_notes'::regclass
      AND tgname = 'control_notes_content_recency'
      AND NOT tgisinternal
  ) THEN
    EXECUTE 'CREATE TRIGGER control_notes_content_recency
      BEFORE UPDATE ON control_notes
      FOR EACH ROW EXECUTE FUNCTION ultra_control_note_content_recency()';
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS control_note_create_receipts (
  user_id text NOT NULL,
  idempotency_key text NOT NULL,
  request_digest text,
  note_id text REFERENCES control_notes(note_id) ON DELETE SET NULL,
  created_at timestamptz NOT NULL,
  PRIMARY KEY (user_id, idempotency_key)
);

DO $$
BEGIN
  IF to_regclass('public.idx_control_note_create_receipts_note') IS NULL THEN
    EXECUTE 'CREATE INDEX idx_control_note_create_receipts_note
      ON control_note_create_receipts(note_id) WHERE note_id IS NOT NULL';
  END IF;
END $$;

DO $$
BEGIN
  IF to_regprocedure('public.ultra_control_note_create_receipt_tombstone()') IS NULL THEN
    EXECUTE $function$
      CREATE FUNCTION ultra_control_note_create_receipt_tombstone()
      RETURNS trigger
      LANGUAGE plpgsql
      AS $body$
      BEGIN
        UPDATE control_note_create_receipts
        SET note_id = NULL, request_digest = NULL
        WHERE note_id = OLD.note_id;
        RETURN OLD;
      END;
      $body$
    $function$;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_trigger
    WHERE tgrelid = 'control_notes'::regclass
      AND tgname = 'control_notes_create_receipt_tombstone'
      AND NOT tgisinternal
  ) THEN
    EXECUTE 'CREATE TRIGGER control_notes_create_receipt_tombstone
      BEFORE DELETE ON control_notes
      FOR EACH ROW EXECUTE FUNCTION ultra_control_note_create_receipt_tombstone()';
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS control_note_direct_append_operations (
  operation_id text PRIMARY KEY,
  user_id text NOT NULL,
  note_id text NOT NULL REFERENCES control_notes(note_id) ON DELETE CASCADE,
  idempotency_key text NOT NULL,
  request_digest text NOT NULL,
  before_revision bigint NOT NULL,
  after_revision bigint NOT NULL,
  undo_revision bigint,
  append_start_byte integer NOT NULL,
  appended_bytes integer NOT NULL,
  append_sha256 text NOT NULL,
  before_content_digest text NOT NULL,
  after_content_digest text NOT NULL,
  created_at timestamptz NOT NULL,
  undone_at timestamptz
);

DO $$
BEGIN
  IF to_regclass('public.idx_control_note_direct_append_owner_key') IS NULL THEN
    EXECUTE 'CREATE UNIQUE INDEX idx_control_note_direct_append_owner_key
      ON control_note_direct_append_operations(user_id, idempotency_key)';
  END IF;
  IF to_regclass('public.idx_control_note_direct_append_owner_created') IS NULL THEN
    EXECUTE 'CREATE INDEX idx_control_note_direct_append_owner_created
      ON control_note_direct_append_operations(user_id, created_at DESC)';
  END IF;
END $$;
