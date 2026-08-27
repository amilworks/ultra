package store

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/jackc/pgx/v5"
)

const (
	noteColumns             = `note_id, user_id, org_id, title, body_markdown, pinned, editor_mode, revision, content_digest, created_at, updated_at`
	notesSnippetSourceLimit = "500"
	maxStoredNoteBodyBytes  = 2 << 20
	maxNoteAppendBodyBytes  = 32 << 10
	maxNoteSearchCalls      = 32
	maxNoteReadCalls        = 64
	maxNoteReadBytes        = 512 << 10
	maxNoteReadCallBytes    = 16 * 4 * 1024
	noteSearchQueryTimeout  = 2 * time.Second
)

func scanNote(row interface{ Scan(...any) error }) (domain.NoteRecord, error) {
	var record domain.NoteRecord
	var orgID *string
	if err := row.Scan(
		&record.NoteID, &record.UserID, &orgID, &record.Title,
		&record.BodyMarkdown, &record.Pinned, &record.EditorMode,
		&record.Revision, &record.ContentDigest, &record.CreatedAt,
		&record.UpdatedAt,
	); err != nil {
		return domain.NoteRecord{}, mapPgError(err)
	}
	if orgID != nil {
		record.OrgID = *orgID
	}
	ensureNoteIdentity(&record)
	return record, nil
}

func ensureNoteIdentity(record *domain.NoteRecord) {
	if record.Revision <= 0 {
		record.Revision = 1
	}
	if record.ContentDigest == "" {
		record.ContentDigest = domain.ComputeNoteContentDigest(record.Title, record.BodyMarkdown)
	}
}

func escapeNoteLike(query string) string {
	replacer := strings.NewReplacer(`\`, `\\`, `%`, `\%`, `_`, `\_`)
	return replacer.Replace(query)
}

func (s *PostgresStore) CreateNote(ctx context.Context, record domain.NoteRecord) (domain.NoteRecord, error) {
	ensureNoteIdentity(&record)
	return scanNote(s.pool.QueryRow(ctx, `
INSERT INTO control_notes (
  note_id, user_id, org_id, title, body_markdown, pinned, editor_mode,
  revision, content_digest, created_at, updated_at
)
VALUES ($1, $2, NULLIF($3, ''), $4, $5, $6, $7, $8, $9, $10, $10)
RETURNING `+noteColumns,
		record.NoteID, record.UserID, record.OrgID, record.Title,
		record.BodyMarkdown, record.Pinned, record.EditorMode,
		record.Revision, record.ContentDigest, record.CreatedAt,
	))
}

func (s *PostgresStore) GetNoteForUser(ctx context.Context, noteID string, userID string) (domain.NoteRecord, error) {
	return scanNote(s.pool.QueryRow(ctx,
		`SELECT `+noteColumns+` FROM control_notes WHERE note_id = $1 AND user_id = $2`,
		noteID, userID,
	))
}

func (s *PostgresStore) UpdateNoteForUser(ctx context.Context, noteID string, userID string, input domain.NoteUpdateInput) (domain.NoteRecord, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.NoteRecord{}, mapPgError(err)
	}
	defer tx.Rollback(ctx) //nolint:errcheck
	record, err := scanNote(tx.QueryRow(ctx,
		`SELECT `+noteColumns+` FROM control_notes WHERE note_id = $1 AND user_id = $2 FOR UPDATE`,
		noteID, userID,
	))
	if err != nil {
		return domain.NoteRecord{}, err
	}
	if input.ExpectedRevision <= 0 || record.Revision != input.ExpectedRevision {
		return domain.NoteRecord{}, ErrNoteRevisionConflict
	}
	applyNoteUpdate(&record, input)
	record.ContentDigest = domain.ComputeNoteContentDigest(record.Title, record.BodyMarkdown)
	record, err = scanNote(tx.QueryRow(ctx, `
UPDATE control_notes SET
  title = $3, body_markdown = $4, pinned = $5, editor_mode = $6,
  revision = revision + 1, content_digest = $7, updated_at = now()
WHERE note_id = $1 AND user_id = $2 AND revision = $8
RETURNING `+noteColumns,
		noteID, userID, record.Title, record.BodyMarkdown, record.Pinned,
		record.EditorMode, record.ContentDigest, input.ExpectedRevision,
	))
	if err != nil {
		if errors.Is(err, ErrNotFound) {
			return domain.NoteRecord{}, ErrNoteRevisionConflict
		}
		return domain.NoteRecord{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.NoteRecord{}, mapPgError(err)
	}
	return record, nil
}

func applyNoteUpdate(record *domain.NoteRecord, input domain.NoteUpdateInput) {
	if input.Title != nil {
		record.Title = *input.Title
	}
	if input.BodyMarkdown != nil {
		record.BodyMarkdown = *input.BodyMarkdown
	}
	if input.Pinned != nil {
		record.Pinned = *input.Pinned
	}
	if input.EditorMode != nil {
		record.EditorMode = *input.EditorMode
	}
}

func (s *PostgresStore) DeleteNoteForUser(ctx context.Context, noteID string, userID string) error {
	tag, err := s.pool.Exec(ctx, `DELETE FROM control_notes WHERE note_id = $1 AND user_id = $2`, noteID, userID)
	if err != nil {
		return mapPgError(err)
	}
	if tag.RowsAffected() == 0 {
		return ErrNotFound
	}
	return nil
}

func (s *PostgresStore) ListNotesForUser(ctx context.Context, input domain.NoteListInput) (domain.NoteListPage, error) {
	limit := input.Limit
	if limit <= 0 || limit > 200 {
		limit = 100
	}
	offset := input.Offset
	if offset < 0 {
		offset = 0
	}
	page := domain.NoteListPage{Notes: []domain.NoteListItem{}}
	query := strings.TrimSpace(input.Query)
	if query == "" {
		if err := s.pool.QueryRow(ctx, `SELECT COUNT(*) FROM control_notes WHERE user_id = $1`, input.UserID).Scan(&page.TotalCount); err != nil {
			return domain.NoteListPage{}, mapPgError(err)
		}
		rows, err := s.pool.Query(ctx, `
SELECT note_id, title, left(body_markdown, `+notesSnippetSourceLimit+`), pinned, revision, updated_at
FROM control_notes
WHERE user_id = $1
ORDER BY pinned DESC, updated_at DESC
LIMIT $2 OFFSET $3`, input.UserID, limit, offset)
		if err != nil {
			return domain.NoteListPage{}, mapPgError(err)
		}
		defer rows.Close()
		return collectNoteItems(rows, page)
	}
	pattern := "%" + escapeNoteLike(query) + "%"
	if err := s.pool.QueryRow(ctx, `
SELECT COUNT(*) FROM control_notes
WHERE user_id = $1 AND (title ILIKE $2 ESCAPE '\' OR body_markdown ILIKE $2 ESCAPE '\')`, input.UserID, pattern).Scan(&page.TotalCount); err != nil {
		return domain.NoteListPage{}, mapPgError(err)
	}
	rows, err := s.pool.Query(ctx, `
SELECT note_id, title,
       substring(body_markdown from GREATEST(strpos(lower(body_markdown), lower($2)) - 120, 1) for `+notesSnippetSourceLimit+`),
       pinned, revision, updated_at
FROM control_notes
WHERE user_id = $1 AND (title ILIKE $3 ESCAPE '\' OR body_markdown ILIKE $3 ESCAPE '\')
ORDER BY CASE WHEN lower(title) = lower($2) THEN 0 WHEN title ILIKE $3 ESCAPE '\' THEN 1 ELSE 2 END,
         pinned DESC, updated_at DESC
LIMIT $4 OFFSET $5`, input.UserID, query, pattern, limit, offset)
	if err != nil {
		return domain.NoteListPage{}, mapPgError(err)
	}
	defer rows.Close()
	return collectNoteItems(rows, page)
}

func collectNoteItems(rows interface {
	Next() bool
	Scan(...any) error
	Err() error
}, page domain.NoteListPage) (domain.NoteListPage, error) {
	for rows.Next() {
		var item domain.NoteListItem
		if err := rows.Scan(&item.NoteID, &item.Title, &item.Snippet, &item.Pinned, &item.Revision, &item.UpdatedAt); err != nil {
			return domain.NoteListPage{}, mapPgError(err)
		}
		page.Notes = append(page.Notes, item)
	}
	if err := rows.Err(); err != nil {
		return domain.NoteListPage{}, mapPgError(err)
	}
	return page, nil
}

func (s *PostgresStore) SearchNotesForUser(ctx context.Context, input domain.NoteSearchInput) ([]domain.NoteSearchHit, error) {
	queryContext, cancel := context.WithTimeout(ctx, noteSearchQueryTimeout)
	defer cancel()
	limit := input.Limit
	if limit <= 0 || limit > 21 {
		limit = 10
	}
	query := strings.TrimSpace(input.Query)
	pattern := "%" + escapeNoteLike(query) + "%"
	rows, err := s.pool.Query(queryContext, `
SELECT note_id, title,
       substring(body_markdown from GREATEST(strpos(lower(body_markdown), lower($2)) - 120, 1) for 500),
       pinned, revision, updated_at
FROM control_notes
WHERE user_id = $1 AND (title ILIKE $3 ESCAPE '\' OR body_markdown ILIKE $3 ESCAPE '\')
ORDER BY CASE WHEN lower(title) = lower($2) THEN 0 WHEN title ILIKE $3 ESCAPE '\' THEN 1 ELSE 2 END,
		pinned DESC, updated_at DESC
LIMIT $4`, input.UserID, query, pattern, limit)
	if err != nil {
		return nil, mapNoteSearchError(err)
	}
	defer rows.Close()
	hits := make([]domain.NoteSearchHit, 0, limit)
	for rows.Next() {
		var hit domain.NoteSearchHit
		if err := rows.Scan(&hit.NoteID, &hit.Title, &hit.Snippet, &hit.Pinned, &hit.Revision, &hit.UpdatedAt); err != nil {
			return nil, mapNoteSearchError(err)
		}
		hits = append(hits, hit)
	}
	if err := rows.Err(); err != nil {
		return nil, mapNoteSearchError(err)
	}
	return hits, nil
}

func mapNoteSearchError(err error) error {
	if errors.Is(err, context.DeadlineExceeded) {
		return ErrNoteSearchTimeout
	}
	return mapPgError(err)
}

func (s *PostgresStore) ConsumeNoteSearchBudget(ctx context.Context, runID string, userID string) error {
	tag, err := s.pool.Exec(ctx, `
INSERT INTO control_note_run_usage (run_id, user_id, search_calls, read_calls, read_bytes, updated_at)
VALUES ($1, $2, 1, 0, 0, now())
ON CONFLICT (run_id) DO UPDATE
SET search_calls = control_note_run_usage.search_calls + 1, updated_at = now()
WHERE control_note_run_usage.user_id = EXCLUDED.user_id
  AND control_note_run_usage.search_calls < $3`, runID, userID, maxNoteSearchCalls)
	if err != nil {
		return mapPgError(err)
	}
	if tag.RowsAffected() == 0 {
		return ErrNoteRetrievalBudget
	}
	return nil
}

func (s *PostgresStore) ConsumeNoteReadBudget(ctx context.Context, runID string, userID string, returnedBytes int) error {
	if returnedBytes < 0 || returnedBytes > maxNoteReadCallBytes {
		return ErrNoteRetrievalBudget
	}
	tag, err := s.pool.Exec(ctx, `
INSERT INTO control_note_run_usage (run_id, user_id, search_calls, read_calls, read_bytes, updated_at)
VALUES ($1, $2, 0, 1, $3, now())
ON CONFLICT (run_id) DO UPDATE
SET read_calls = control_note_run_usage.read_calls + 1,
    read_bytes = control_note_run_usage.read_bytes + EXCLUDED.read_bytes,
    updated_at = now()
WHERE control_note_run_usage.user_id = EXCLUDED.user_id
  AND control_note_run_usage.read_calls < $4
  AND control_note_run_usage.read_bytes + EXCLUDED.read_bytes <= $5`,
		runID, userID, returnedBytes, maxNoteReadCalls, maxNoteReadBytes)
	if err != nil {
		return mapPgError(err)
	}
	if tag.RowsAffected() == 0 {
		return ErrNoteRetrievalBudget
	}
	return nil
}

func (s *PostgresStore) CreateNoteReadGrant(ctx context.Context, grant domain.NoteReadGrantRecord) error {
	if _, err := s.pool.Exec(ctx, `DELETE FROM control_note_read_grants WHERE user_id = $1 AND expires_at <= now()`, grant.UserID); err != nil {
		return mapPgError(err)
	}
	tag, err := s.pool.Exec(ctx, `
INSERT INTO control_note_read_grants (token_hash, run_id, user_id, note_id, revision, expires_at, created_at)
SELECT $1, $2, $3, n.note_id, $5, $6, now()
FROM control_notes n
WHERE n.note_id = $4 AND n.user_id = $3 AND n.revision = $5
ON CONFLICT (token_hash) DO NOTHING`, grant.TokenHash, grant.RunID, grant.UserID,
		grant.NoteID, grant.Revision, grant.ExpiresAt)
	if err != nil {
		return mapPgError(err)
	}
	if tag.RowsAffected() == 0 {
		return ErrNoteRevisionConflict
	}
	return nil
}

func (s *PostgresStore) ExpireNoteReadGrants(ctx context.Context, now time.Time, limit int) (int, error) {
	_ = now // PostgreSQL time is authoritative for expiry decisions.
	if limit <= 0 || limit > 1000 {
		limit = 200
	}
	tag, err := s.pool.Exec(ctx, `
WITH expired AS (
  SELECT token_hash
  FROM control_note_read_grants
  WHERE expires_at <= now()
  ORDER BY expires_at
  LIMIT $1
  FOR UPDATE SKIP LOCKED
)
DELETE FROM control_note_read_grants g
USING expired
WHERE g.token_hash = expired.token_hash`, limit)
	if err != nil {
		return 0, mapPgError(err)
	}
	return int(tag.RowsAffected()), nil
}

func (s *PostgresStore) CreateNoteAppendProposal(ctx context.Context, input domain.CreateNoteAppendProposalInput) (domain.NoteAppendProposalRecord, error) {
	if strings.TrimSpace(input.IdempotencyKey) == "" || input.RequestDigest == "" ||
		strings.TrimSpace(input.BodyMarkdown) == "" || len(input.BodyMarkdown) > maxNoteAppendBodyBytes ||
		input.ExpectedRevision <= 0 || input.ReadTokenHash == "" {
		return domain.NoteAppendProposalRecord{}, ErrConflict
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.NoteAppendProposalRecord{}, mapPgError(err)
	}
	defer tx.Rollback(ctx) //nolint:errcheck

	if existing, found, err := findExistingNoteProposal(ctx, tx, input); err != nil {
		return domain.NoteAppendProposalRecord{}, err
	} else if found {
		if existing.RequestDigest != input.RequestDigest {
			return domain.NoteAppendProposalRecord{}, ErrConflict
		}
		return existing, nil
	}

	var noteTitle string
	var currentRevision int64
	err = tx.QueryRow(ctx, `
SELECT n.title, n.revision
FROM control_note_read_grants g
JOIN control_notes n ON n.note_id = g.note_id AND n.user_id = g.user_id
WHERE g.token_hash = $1 AND g.run_id = $2 AND g.user_id = $3
  AND g.note_id = $4 AND g.revision = $5 AND g.expires_at > now()
FOR UPDATE OF n`, input.ReadTokenHash, input.RunID, input.UserID,
		input.NoteID, input.ExpectedRevision).Scan(&noteTitle, &currentRevision)
	if err != nil {
		if errors.Is(mapPgError(err), ErrNotFound) {
			return domain.NoteAppendProposalRecord{}, ErrNoteReadTokenInvalid
		}
		return domain.NoteAppendProposalRecord{}, mapPgError(err)
	}
	if currentRevision != input.ExpectedRevision {
		return domain.NoteAppendProposalRecord{}, ErrNoteRevisionConflict
	}

	proposal, err := scanNoteAppendProposal(tx.QueryRow(ctx, `
INSERT INTO control_note_append_proposals (
  proposal_id, run_id, user_id, note_id, base_revision, body_markdown,
  body_sha256, idempotency_key, request_digest, status, expires_at,
  created_at, updated_at
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'pending', $10, now(), now())
ON CONFLICT DO NOTHING
RETURNING proposal_id, run_id, note_id, $11, user_id, base_revision,
          body_markdown, body_sha256, committed_body_sha256,
          idempotency_key, request_digest, status, COALESCE(operation_id, ''),
          expires_at, created_at, updated_at`,
		input.ProposalID, input.RunID, input.UserID, input.NoteID,
		input.ExpectedRevision, input.BodyMarkdown, domain.NoteBodySHA256(input.BodyMarkdown),
		input.IdempotencyKey, input.RequestDigest, input.ExpiresAt, noteTitle))
	if err != nil {
		if !errors.Is(err, ErrNotFound) {
			return domain.NoteAppendProposalRecord{}, err
		}
		existing, found, findErr := findExistingNoteProposal(ctx, tx, input)
		if findErr != nil {
			return domain.NoteAppendProposalRecord{}, findErr
		}
		if !found || existing.RequestDigest != input.RequestDigest {
			return domain.NoteAppendProposalRecord{}, ErrConflict
		}
		proposal = existing
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.NoteAppendProposalRecord{}, mapPgError(err)
	}
	return proposal, nil
}

func findExistingNoteProposal(ctx context.Context, tx pgx.Tx, input domain.CreateNoteAppendProposalInput) (domain.NoteAppendProposalRecord, bool, error) {
	existing, err := scanNoteAppendProposal(tx.QueryRow(ctx, noteAppendProposalSelect+`
WHERE p.run_id = $1 AND p.user_id = $2 AND p.idempotency_key = $3`,
		input.RunID, input.UserID, input.IdempotencyKey))
	if err == nil {
		return existing, true, nil
	}
	if !errors.Is(err, ErrNotFound) {
		return domain.NoteAppendProposalRecord{}, false, err
	}
	existing, err = scanNoteAppendProposal(tx.QueryRow(ctx, noteAppendProposalSelect+`
WHERE p.run_id = $1 AND p.user_id = $2 AND p.note_id = $3
  AND p.base_revision = $4 AND p.body_sha256 = $5
  AND p.status IN ('pending', 'committed')`, input.RunID,
		input.UserID, input.NoteID, input.ExpectedRevision,
		domain.NoteBodySHA256(input.BodyMarkdown)))
	if err == nil {
		return existing, true, nil
	}
	if errors.Is(err, ErrNotFound) {
		return domain.NoteAppendProposalRecord{}, false, nil
	}
	return domain.NoteAppendProposalRecord{}, false, err
}

func (s *PostgresStore) GetNoteAppendProposalForUser(ctx context.Context, proposalID string, userID string) (domain.NoteAppendProposalRecord, error) {
	if _, err := s.pool.Exec(ctx, `
UPDATE control_note_append_proposals SET status = 'expired', body_markdown = '', updated_at = now()
WHERE proposal_id = $1 AND user_id = $2 AND status = 'pending' AND expires_at <= now()`, proposalID, userID); err != nil {
		return domain.NoteAppendProposalRecord{}, mapPgError(err)
	}
	return scanNoteAppendProposal(s.pool.QueryRow(ctx, noteAppendProposalSelect+`
WHERE p.proposal_id = $1 AND p.user_id = $2`, proposalID, userID))
}

func (s *PostgresStore) ExpireNoteAppendProposals(ctx context.Context, now time.Time, limit int) (int, error) {
	_ = now // PostgreSQL time is authoritative for expiry decisions.
	if limit <= 0 || limit > 1000 {
		limit = 200
	}
	tag, err := s.pool.Exec(ctx, `
WITH expired AS (
  SELECT proposal_id
  FROM control_note_append_proposals
  WHERE status = 'pending' AND expires_at <= now()
  ORDER BY expires_at
  LIMIT $1
  FOR UPDATE SKIP LOCKED
)
UPDATE control_note_append_proposals p
SET status = 'expired', body_markdown = '', updated_at = now()
FROM expired
WHERE p.proposal_id = expired.proposal_id`, limit)
	if err != nil {
		return 0, mapPgError(err)
	}
	return int(tag.RowsAffected()), nil
}

const noteAppendProposalSelect = `
SELECT p.proposal_id, p.run_id, p.note_id, n.title, p.user_id, p.base_revision,
       p.body_markdown, p.body_sha256, p.committed_body_sha256,
       p.idempotency_key, p.request_digest, p.status, COALESCE(p.operation_id, ''),
       p.expires_at, p.created_at, p.updated_at
FROM control_note_append_proposals p
JOIN control_notes n ON n.note_id = p.note_id `

func scanNoteAppendProposal(row interface{ Scan(...any) error }) (domain.NoteAppendProposalRecord, error) {
	var record domain.NoteAppendProposalRecord
	if err := row.Scan(
		&record.ProposalID, &record.RunID, &record.NoteID, &record.NoteTitle,
		&record.UserID, &record.BaseRevision, &record.BodyMarkdown,
		&record.BodySHA256, &record.CommittedBodySHA256, &record.IdempotencyKey,
		&record.RequestDigest, &record.Status, &record.OperationID,
		&record.ExpiresAt, &record.CreatedAt, &record.UpdatedAt,
	); err != nil {
		return domain.NoteAppendProposalRecord{}, mapPgError(err)
	}
	return record, nil
}

func (s *PostgresStore) CommitNoteAppendProposalForUser(ctx context.Context, input domain.CommitNoteAppendProposalInput) (domain.NoteAppendOperationRecord, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.NoteAppendOperationRecord{}, mapPgError(err)
	}
	defer tx.Rollback(ctx) //nolint:errcheck
	proposal, active, err := scanNoteAppendProposalActive(tx.QueryRow(ctx, `
SELECT p.proposal_id, p.run_id, p.note_id, n.title, p.user_id, p.base_revision,
       p.body_markdown, p.body_sha256, p.committed_body_sha256,
       p.idempotency_key, p.request_digest, p.status, COALESCE(p.operation_id, ''),
       p.expires_at, p.created_at, p.updated_at, p.expires_at > now()
FROM control_note_append_proposals p JOIN control_notes n ON n.note_id = p.note_id
WHERE p.proposal_id = $1 AND p.user_id = $2 FOR UPDATE OF p`, input.ProposalID, input.UserID))
	if err != nil {
		return domain.NoteAppendOperationRecord{}, err
	}
	if proposal.Status == domain.NoteAppendProposalStatusCommitted {
		if input.BodyMarkdown != nil && domain.NoteBodySHA256(*input.BodyMarkdown) != proposal.CommittedBodySHA256 {
			return domain.NoteAppendOperationRecord{}, ErrConflict
		}
		operation, err := scanNoteAppendOperation(tx.QueryRow(ctx, noteAppendOperationSelect+`
WHERE o.operation_id = $1 AND o.user_id = $2`, proposal.OperationID, input.UserID))
		return operation.public(), err
	}
	if proposal.Status != domain.NoteAppendProposalStatusPending || !active {
		if proposal.Status == domain.NoteAppendProposalStatusPending {
			if _, err := tx.Exec(ctx, `UPDATE control_note_append_proposals SET status = 'expired', body_markdown = '', updated_at = now() WHERE proposal_id = $1`, proposal.ProposalID); err != nil {
				return domain.NoteAppendOperationRecord{}, mapPgError(err)
			}
			if err := tx.Commit(ctx); err != nil {
				return domain.NoteAppendOperationRecord{}, mapPgError(err)
			}
		}
		return domain.NoteAppendOperationRecord{}, ErrNoteProposalExpired
	}
	body := proposal.BodyMarkdown
	if input.BodyMarkdown != nil {
		body = *input.BodyMarkdown
	}
	if strings.TrimSpace(body) == "" || len(body) > maxNoteAppendBodyBytes {
		return domain.NoteAppendOperationRecord{}, ErrConflict
	}
	note, err := scanNote(tx.QueryRow(ctx,
		`SELECT `+noteColumns+` FROM control_notes WHERE note_id = $1 AND user_id = $2 FOR UPDATE`,
		proposal.NoteID, input.UserID))
	if err != nil {
		return domain.NoteAppendOperationRecord{}, err
	}
	if note.Revision != proposal.BaseRevision {
		return domain.NoteAppendOperationRecord{}, ErrNoteRevisionConflict
	}
	suffix := noteAppendSuffix(note.BodyMarkdown, body)
	if len(note.BodyMarkdown)+len(suffix) > maxStoredNoteBodyBytes {
		return domain.NoteAppendOperationRecord{}, ErrConflict
	}
	startByte := len(note.BodyMarkdown)
	beforeDigest := note.ContentDigest
	note.BodyMarkdown += suffix
	note.ContentDigest = domain.ComputeNoteContentDigest(note.Title, note.BodyMarkdown)
	note, err = scanNote(tx.QueryRow(ctx, `
UPDATE control_notes SET body_markdown = $3, revision = revision + 1,
  content_digest = $4, updated_at = now()
WHERE note_id = $1 AND user_id = $2 AND revision = $5
RETURNING `+noteColumns, note.NoteID, input.UserID, note.BodyMarkdown,
		note.ContentDigest, proposal.BaseRevision))
	if err != nil {
		if errors.Is(err, ErrNotFound) {
			return domain.NoteAppendOperationRecord{}, ErrNoteRevisionConflict
		}
		return domain.NoteAppendOperationRecord{}, err
	}
	operation, err := scanNoteAppendOperation(tx.QueryRow(ctx, `
INSERT INTO control_note_append_operations (
  operation_id, proposal_id, run_id, user_id, note_id, before_revision,
  after_revision, append_start_byte, appended_bytes, append_sha256,
  before_content_digest, after_content_digest, created_at
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, now())
RETURNING operation_id, proposal_id, run_id, note_id, $13, user_id,
  before_revision, after_revision, COALESCE(undo_revision, 0), appended_bytes,
  before_content_digest, after_content_digest, append_start_byte, append_sha256,
  created_at, undone_at`, input.OperationID, proposal.ProposalID, proposal.RunID,
		input.UserID, proposal.NoteID, proposal.BaseRevision, note.Revision,
		startByte, len(suffix), domain.NoteBodySHA256(suffix), beforeDigest,
		note.ContentDigest, note.Title))
	if err != nil {
		return domain.NoteAppendOperationRecord{}, err
	}
	if _, err := tx.Exec(ctx, `
UPDATE control_note_append_proposals
SET status = 'committed', body_markdown = '', committed_body_sha256 = $2,
    operation_id = $3, updated_at = now()
WHERE proposal_id = $1`, proposal.ProposalID, domain.NoteBodySHA256(body), operation.OperationID); err != nil {
		return domain.NoteAppendOperationRecord{}, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.NoteAppendOperationRecord{}, mapPgError(err)
	}
	return operation.public(), nil
}

func scanNoteAppendProposalActive(row interface{ Scan(...any) error }) (domain.NoteAppendProposalRecord, bool, error) {
	var record domain.NoteAppendProposalRecord
	var active bool
	if err := row.Scan(
		&record.ProposalID, &record.RunID, &record.NoteID, &record.NoteTitle,
		&record.UserID, &record.BaseRevision, &record.BodyMarkdown,
		&record.BodySHA256, &record.CommittedBodySHA256, &record.IdempotencyKey,
		&record.RequestDigest, &record.Status, &record.OperationID,
		&record.ExpiresAt, &record.CreatedAt, &record.UpdatedAt, &active,
	); err != nil {
		return domain.NoteAppendProposalRecord{}, false, mapPgError(err)
	}
	return record, active, nil
}

type storedNoteAppendOperation struct {
	domain.NoteAppendOperationRecord
	AppendStartByte int
	AppendSHA256    string
}

const noteAppendOperationSelect = `
SELECT o.operation_id, o.proposal_id, o.run_id, o.note_id, n.title, o.user_id,
       o.before_revision, o.after_revision, COALESCE(o.undo_revision, 0),
       o.appended_bytes, o.before_content_digest, o.after_content_digest,
       o.append_start_byte, o.append_sha256, o.created_at, o.undone_at
FROM control_note_append_operations o
JOIN control_notes n ON n.note_id = o.note_id `

func scanNoteAppendOperation(row interface{ Scan(...any) error }) (storedNoteAppendOperation, error) {
	var record storedNoteAppendOperation
	if err := row.Scan(
		&record.OperationID, &record.ProposalID, &record.RunID, &record.NoteID,
		&record.NoteTitle, &record.UserID, &record.BeforeRevision,
		&record.AfterRevision, &record.UndoRevision, &record.AppendedBytes,
		&record.BeforeContentDigest, &record.AfterContentDigest,
		&record.AppendStartByte, &record.AppendSHA256, &record.CreatedAt,
		&record.UndoneAt,
	); err != nil {
		return storedNoteAppendOperation{}, mapPgError(err)
	}
	return record, nil
}

func (record storedNoteAppendOperation) public() domain.NoteAppendOperationRecord {
	return record.NoteAppendOperationRecord
}

func (s *PostgresStore) GetNoteAppendOperationForUser(ctx context.Context, operationID string, userID string) (domain.NoteAppendOperationRecord, error) {
	record, err := scanNoteAppendOperation(s.pool.QueryRow(ctx, noteAppendOperationSelect+`
WHERE o.operation_id = $1 AND o.user_id = $2`, operationID, userID))
	return record.public(), err
}

func (s *PostgresStore) UndoNoteAppendOperationForUser(ctx context.Context, input domain.UndoNoteAppendOperationInput) (domain.NoteAppendOperationRecord, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.NoteAppendOperationRecord{}, mapPgError(err)
	}
	defer tx.Rollback(ctx) //nolint:errcheck
	operation, err := scanNoteAppendOperation(tx.QueryRow(ctx, noteAppendOperationSelect+`
WHERE o.operation_id = $1 AND o.user_id = $2 FOR UPDATE OF o`, input.OperationID, input.UserID))
	if err != nil {
		return domain.NoteAppendOperationRecord{}, err
	}
	if operation.UndoneAt != nil {
		return operation.public(), nil
	}
	note, err := scanNote(tx.QueryRow(ctx,
		`SELECT `+noteColumns+` FROM control_notes WHERE note_id = $1 AND user_id = $2 FOR UPDATE`,
		operation.NoteID, input.UserID))
	if err != nil {
		return domain.NoteAppendOperationRecord{}, err
	}
	endByte := operation.AppendStartByte + operation.AppendedBytes
	if note.Revision != operation.AfterRevision || operation.AppendStartByte < 0 ||
		operation.AppendedBytes <= 0 || endByte != len(note.BodyMarkdown) ||
		domain.NoteBodySHA256(note.BodyMarkdown[operation.AppendStartByte:endByte]) != operation.AppendSHA256 {
		return domain.NoteAppendOperationRecord{}, ErrNoteUndoConflict
	}
	note.BodyMarkdown = note.BodyMarkdown[:operation.AppendStartByte]
	note.ContentDigest = domain.ComputeNoteContentDigest(note.Title, note.BodyMarkdown)
	note, err = scanNote(tx.QueryRow(ctx, `
UPDATE control_notes SET body_markdown = $3, revision = revision + 1,
  content_digest = $4, updated_at = now()
WHERE note_id = $1 AND user_id = $2 AND revision = $5
RETURNING `+noteColumns, note.NoteID, input.UserID, note.BodyMarkdown,
		note.ContentDigest, operation.AfterRevision))
	if err != nil {
		if errors.Is(err, ErrNotFound) {
			return domain.NoteAppendOperationRecord{}, ErrNoteUndoConflict
		}
		return domain.NoteAppendOperationRecord{}, err
	}
	operation, err = scanNoteAppendOperation(tx.QueryRow(ctx, `
UPDATE control_note_append_operations SET undone_at = now(), undo_revision = $3
WHERE operation_id = $1 AND user_id = $2
RETURNING operation_id, proposal_id, run_id, note_id,
  (SELECT title FROM control_notes WHERE note_id = control_note_append_operations.note_id),
  user_id, before_revision, after_revision, COALESCE(undo_revision, 0),
  appended_bytes, before_content_digest, after_content_digest,
  append_start_byte, append_sha256, created_at, undone_at`,
		input.OperationID, input.UserID, note.Revision))
	if err != nil {
		return domain.NoteAppendOperationRecord{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.NoteAppendOperationRecord{}, mapPgError(err)
	}
	return operation.public(), nil
}

func noteAppendSuffix(existing string, addition string) string {
	if existing == "" || strings.HasSuffix(existing, "\n\n") || strings.HasPrefix(addition, "\n") {
		return addition
	}
	if strings.HasSuffix(existing, "\n") {
		return "\n" + addition
	}
	return "\n\n" + addition
}
