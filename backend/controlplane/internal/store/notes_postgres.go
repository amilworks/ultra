package store

import (
	"context"
	"strings"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

const noteColumns = `note_id, user_id, org_id, title, body_markdown, pinned, editor_mode, created_at, updated_at`

// notesSnippetSourceLimit bounds how much body the LIST query ships per row —
// the snippet is a row preview, never the document.
const notesSnippetSourceLimit = "300"

func scanNote(row interface{ Scan(...any) error }) (domain.NoteRecord, error) {
	var record domain.NoteRecord
	var orgID *string
	if err := row.Scan(
		&record.NoteID,
		&record.UserID,
		&orgID,
		&record.Title,
		&record.BodyMarkdown,
		&record.Pinned,
		&record.EditorMode,
		&record.CreatedAt,
		&record.UpdatedAt,
	); err != nil {
		return domain.NoteRecord{}, mapPgError(err)
	}
	if orgID != nil {
		record.OrgID = *orgID
	}
	return record, nil
}

// escapeNoteLike neutralizes LIKE metacharacters in a user query so a search
// for "100%" matches the literal text instead of everything.
func escapeNoteLike(query string) string {
	replacer := strings.NewReplacer(`\`, `\\`, `%`, `\%`, `_`, `\_`)
	return replacer.Replace(query)
}

func (s *PostgresStore) CreateNote(ctx context.Context, record domain.NoteRecord) (domain.NoteRecord, error) {
	return scanNote(s.pool.QueryRow(ctx, `
INSERT INTO control_notes (note_id, user_id, org_id, title, body_markdown, pinned, editor_mode, created_at, updated_at)
VALUES ($1, $2, NULLIF($3, ''), $4, $5, $6, $7, $8, $8)
RETURNING `+noteColumns,
		record.NoteID,
		record.UserID,
		record.OrgID,
		record.Title,
		record.BodyMarkdown,
		record.Pinned,
		record.EditorMode,
		record.CreatedAt,
	))
}

// GetNoteForUser is owner-scoped by construction: the WHERE clause carries the
// user id, so another user's note id behaves exactly like a missing note.
func (s *PostgresStore) GetNoteForUser(ctx context.Context, noteID string, userID string) (domain.NoteRecord, error) {
	return scanNote(s.pool.QueryRow(ctx,
		`SELECT `+noteColumns+` FROM control_notes WHERE note_id = $1 AND user_id = $2`,
		noteID, userID,
	))
}

func (s *PostgresStore) UpdateNoteForUser(ctx context.Context, noteID string, userID string, input domain.NoteUpdateInput) (domain.NoteRecord, error) {
	return scanNote(s.pool.QueryRow(ctx, `
UPDATE control_notes SET
  title = COALESCE($3, title),
  body_markdown = COALESCE($4, body_markdown),
  pinned = COALESCE($5, pinned),
  editor_mode = COALESCE($6, editor_mode),
  updated_at = now()
WHERE note_id = $1 AND user_id = $2
RETURNING `+noteColumns,
		noteID, userID, input.Title, input.BodyMarkdown, input.Pinned, input.EditorMode,
	))
}

func (s *PostgresStore) DeleteNoteForUser(ctx context.Context, noteID string, userID string) error {
	tag, err := s.pool.Exec(ctx,
		`DELETE FROM control_notes WHERE note_id = $1 AND user_id = $2`,
		noteID, userID,
	)
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
	listSelect := `
SELECT note_id, title, left(body_markdown, ` + notesSnippetSourceLimit + `), pinned, updated_at
FROM control_notes`

	if query == "" {
		if err := s.pool.QueryRow(ctx,
			`SELECT COUNT(*) FROM control_notes WHERE user_id = $1`, input.UserID,
		).Scan(&page.TotalCount); err != nil {
			return domain.NoteListPage{}, mapPgError(err)
		}
		rows, err := s.pool.Query(ctx, listSelect+`
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
	if err := s.pool.QueryRow(ctx,
		`SELECT COUNT(*) FROM control_notes WHERE user_id = $1 AND (title ILIKE $2 OR body_markdown ILIKE $2)`,
		input.UserID, pattern,
	).Scan(&page.TotalCount); err != nil {
		return domain.NoteListPage{}, mapPgError(err)
	}
	rows, err := s.pool.Query(ctx, listSelect+`
WHERE user_id = $1 AND (title ILIKE $2 OR body_markdown ILIKE $2)
ORDER BY pinned DESC, updated_at DESC
LIMIT $3 OFFSET $4`, input.UserID, pattern, limit, offset)
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
		if err := rows.Scan(&item.NoteID, &item.Title, &item.Snippet, &item.Pinned, &item.UpdatedAt); err != nil {
			return domain.NoteListPage{}, mapPgError(err)
		}
		page.Notes = append(page.Notes, item)
	}
	if err := rows.Err(); err != nil {
		return domain.NoteListPage{}, mapPgError(err)
	}
	return page, nil
}
