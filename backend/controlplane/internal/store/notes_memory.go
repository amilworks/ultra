package store

import (
	"context"
	"sort"
	"strings"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

// The memory twin mirrors notes_postgres.go semantics exactly — owner-scoped
// reads, COALESCE-style partial updates, pinned-then-recency ordering, hard
// deletes — so handler tests exercise the same contract production runs.

func (s *MemoryStore) CreateNote(ctx context.Context, record domain.NoteRecord) (domain.NoteRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.notes == nil {
		s.notes = map[string]domain.NoteRecord{}
	}
	if _, exists := s.notes[record.NoteID]; exists {
		return domain.NoteRecord{}, ErrConflict
	}
	record.UpdatedAt = record.CreatedAt
	s.notes[record.NoteID] = record
	return record, nil
}

func (s *MemoryStore) GetNoteForUser(ctx context.Context, noteID string, userID string) (domain.NoteRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	record, ok := s.notes[noteID]
	if !ok || record.UserID != userID {
		return domain.NoteRecord{}, ErrNotFound
	}
	return record, nil
}

func (s *MemoryStore) UpdateNoteForUser(ctx context.Context, noteID string, userID string, input domain.NoteUpdateInput) (domain.NoteRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	record, ok := s.notes[noteID]
	if !ok || record.UserID != userID {
		return domain.NoteRecord{}, ErrNotFound
	}
	if input.Title != nil {
		record.Title = *input.Title
	}
	if input.BodyMarkdown != nil {
		record.BodyMarkdown = *input.BodyMarkdown
	}
	if input.Pinned != nil {
		record.Pinned = *input.Pinned
	}
	record.UpdatedAt = domain.Now()
	s.notes[noteID] = record
	return record, nil
}

func (s *MemoryStore) DeleteNoteForUser(ctx context.Context, noteID string, userID string) error {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	record, ok := s.notes[noteID]
	if !ok || record.UserID != userID {
		return ErrNotFound
	}
	delete(s.notes, record.NoteID)
	return nil
}

func (s *MemoryStore) ListNotesForUser(ctx context.Context, input domain.NoteListInput) (domain.NoteListPage, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	limit := input.Limit
	if limit <= 0 || limit > 200 {
		limit = 100
	}
	offset := input.Offset
	if offset < 0 {
		offset = 0
	}
	query := strings.ToLower(strings.TrimSpace(input.Query))

	matched := []domain.NoteRecord{}
	for _, record := range s.notes {
		if record.UserID != input.UserID {
			continue
		}
		if query != "" &&
			!strings.Contains(strings.ToLower(record.Title), query) &&
			!strings.Contains(strings.ToLower(record.BodyMarkdown), query) {
			continue
		}
		matched = append(matched, record)
	}
	sort.Slice(matched, func(i, j int) bool {
		if matched[i].Pinned != matched[j].Pinned {
			return matched[i].Pinned
		}
		return matched[i].UpdatedAt.After(matched[j].UpdatedAt)
	})

	page := domain.NoteListPage{Notes: []domain.NoteListItem{}, TotalCount: len(matched)}
	for index := offset; index < len(matched) && len(page.Notes) < limit; index++ {
		record := matched[index]
		snippet := record.BodyMarkdown
		if len(snippet) > 300 {
			snippet = snippet[:300]
		}
		page.Notes = append(page.Notes, domain.NoteListItem{
			NoteID:    record.NoteID,
			Title:     record.Title,
			Snippet:   snippet,
			Pinned:    record.Pinned,
			UpdatedAt: record.UpdatedAt,
		})
	}
	return page, nil
}
