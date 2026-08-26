ALTER TABLE control_notes ADD COLUMN IF NOT EXISTS revision bigint NOT NULL DEFAULT 1;
ALTER TABLE control_notes ADD COLUMN IF NOT EXISTS content_digest text NOT NULL DEFAULT '';

-- The worker historically created this checkpoint table itself. Take it into
-- the control schema, remove legacy orphans, and bind each temporary checkpoint
-- to its run so hard run/conversation deletion is complete and cannot be
-- reversed by a late checkpoint upsert.
CREATE TABLE IF NOT EXISTS deepagents_checkpoint_threads (
  thread_id text PRIMARY KEY,
  state bytea NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS deepagents_checkpoint_threads_updated_at_idx
  ON deepagents_checkpoint_threads(updated_at);

LOCK TABLE deepagents_checkpoint_threads IN SHARE ROW EXCLUSIVE MODE;

DELETE FROM deepagents_checkpoint_threads checkpoint
WHERE NOT EXISTS (
  SELECT 1 FROM control_runs run WHERE run.run_id = checkpoint.thread_id
);

DO $$
DECLARE
  constraint_name text;
BEGIN
  FOR constraint_name IN
    SELECT constraint_row.conname
    FROM pg_constraint constraint_row
    WHERE constraint_row.conrelid = 'deepagents_checkpoint_threads'::regclass
      AND constraint_row.contype = 'f'
      AND constraint_row.conkey = ARRAY[
        (SELECT attnum FROM pg_attribute
         WHERE attrelid = 'deepagents_checkpoint_threads'::regclass
           AND attname = 'thread_id')
      ]::smallint[]
      AND NOT (
        constraint_row.confrelid = 'control_runs'::regclass
        AND constraint_row.confdeltype = 'c'
        AND constraint_row.confkey = ARRAY[
          (SELECT attnum FROM pg_attribute
           WHERE attrelid = 'control_runs'::regclass
             AND attname = 'run_id')
        ]::smallint[]
      )
  LOOP
    EXECUTE format(
      'ALTER TABLE deepagents_checkpoint_threads DROP CONSTRAINT %I',
      constraint_name
    );
  END LOOP;

  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint constraint_row
    WHERE constraint_row.conrelid = 'deepagents_checkpoint_threads'::regclass
      AND constraint_row.contype = 'f'
      AND constraint_row.confrelid = 'control_runs'::regclass
      AND constraint_row.confdeltype = 'c'
      AND constraint_row.conkey = ARRAY[
        (SELECT attnum FROM pg_attribute
         WHERE attrelid = 'deepagents_checkpoint_threads'::regclass
           AND attname = 'thread_id')
      ]::smallint[]
      AND constraint_row.confkey = ARRAY[
        (SELECT attnum FROM pg_attribute
         WHERE attrelid = 'control_runs'::regclass
           AND attname = 'run_id')
      ]::smallint[]
  ) THEN
    ALTER TABLE deepagents_checkpoint_threads
      ADD CONSTRAINT deepagents_checkpoint_threads_run_fk
      FOREIGN KEY (thread_id) REFERENCES control_runs(run_id) ON DELETE CASCADE;
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS control_note_read_grants (
  token_hash text PRIMARY KEY,
  run_id text NOT NULL REFERENCES control_runs(run_id) ON DELETE CASCADE,
  user_id text NOT NULL,
  note_id text NOT NULL REFERENCES control_notes(note_id) ON DELETE CASCADE,
  revision bigint NOT NULL,
  expires_at timestamptz NOT NULL,
  created_at timestamptz NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_control_note_read_grants_user_expiry
  ON control_note_read_grants(user_id, expires_at);
CREATE INDEX IF NOT EXISTS idx_control_note_read_grants_expiry
  ON control_note_read_grants(expires_at);

CREATE TABLE IF NOT EXISTS control_note_run_usage (
  run_id text PRIMARY KEY REFERENCES control_runs(run_id) ON DELETE CASCADE,
  user_id text NOT NULL,
  search_calls integer NOT NULL DEFAULT 0,
  read_calls integer NOT NULL DEFAULT 0,
  read_bytes bigint NOT NULL DEFAULT 0,
  updated_at timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS control_note_append_proposals (
  proposal_id text PRIMARY KEY,
  run_id text NOT NULL REFERENCES control_runs(run_id) ON DELETE CASCADE,
  user_id text NOT NULL,
  note_id text NOT NULL REFERENCES control_notes(note_id) ON DELETE CASCADE,
  base_revision bigint NOT NULL,
  body_markdown text NOT NULL,
  body_sha256 text NOT NULL,
  idempotency_key text NOT NULL,
  request_digest text NOT NULL,
  committed_body_sha256 text NOT NULL DEFAULT '',
  status text NOT NULL DEFAULT 'pending',
  operation_id text,
  expires_at timestamptz NOT NULL,
  created_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL,
  CHECK (status IN ('pending', 'committed', 'expired'))
);
CREATE INDEX IF NOT EXISTS idx_control_note_append_proposals_user_expiry
  ON control_note_append_proposals(user_id, status, expires_at);
CREATE INDEX IF NOT EXISTS idx_control_note_append_proposals_pending_expiry
  ON control_note_append_proposals(expires_at)
  WHERE status = 'pending';
CREATE UNIQUE INDEX IF NOT EXISTS idx_control_note_append_proposals_run_idempotency
  ON control_note_append_proposals(run_id, idempotency_key);
CREATE UNIQUE INDEX IF NOT EXISTS idx_control_note_append_proposals_run_content
  ON control_note_append_proposals(run_id, note_id, base_revision, body_sha256)
  WHERE status IN ('pending', 'committed');

CREATE TABLE IF NOT EXISTS control_note_append_operations (
  operation_id text PRIMARY KEY,
  -- A committed proposal is part of the operation's idempotency receipt and
  -- cannot be removed independently. Note/run hard deletion still cascades to
  -- both rows through their own foreign keys in the same statement.
  proposal_id text NOT NULL UNIQUE REFERENCES control_note_append_proposals(proposal_id) ON DELETE NO ACTION,
  run_id text NOT NULL REFERENCES control_runs(run_id) ON DELETE CASCADE,
  user_id text NOT NULL,
  note_id text NOT NULL REFERENCES control_notes(note_id) ON DELETE CASCADE,
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
CREATE INDEX IF NOT EXISTS idx_control_note_append_operations_user_created
  ON control_note_append_operations(user_id, created_at DESC);
