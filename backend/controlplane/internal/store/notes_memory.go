package store

import (
	"context"
	"sort"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

type memoryNoteRunUsage struct {
	UserID      string
	SearchCalls int
	ReadCalls   int
	ReadBytes   int
}

type memoryNoteAppendOperation struct {
	domain.NoteAppendOperationRecord
	AppendStartByte int
	AppendSHA256    string
}

func (s *MemoryStore) CreateNote(ctx context.Context, record domain.NoteRecord) (domain.NoteRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, exists := s.notes[record.NoteID]; exists {
		return domain.NoteRecord{}, ErrConflict
	}
	if record.EditorMode == "" {
		record.EditorMode = domain.NoteEditorModeMarkdown
	}
	ensureNoteIdentity(&record)
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
	ensureNoteIdentity(&record)
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
	ensureNoteIdentity(&record)
	if input.ExpectedRevision <= 0 || input.ExpectedRevision != record.Revision {
		return domain.NoteRecord{}, ErrNoteRevisionConflict
	}
	applyNoteUpdate(&record, input)
	record.Revision++
	record.ContentDigest = domain.ComputeNoteContentDigest(record.Title, record.BodyMarkdown)
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
	delete(s.notes, noteID)
	for token, grant := range s.noteReadGrants {
		if grant.NoteID == noteID {
			delete(s.noteReadGrants, token)
		}
	}
	for proposalID, proposal := range s.noteAppendProposals {
		if proposal.NoteID == noteID {
			delete(s.noteAppendProposals, proposalID)
		}
	}
	for operationID, operation := range s.noteAppendOperations {
		if operation.NoteID == noteID {
			delete(s.noteAppendOperations, operationID)
		}
	}
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
	matched := make([]domain.NoteRecord, 0)
	for _, record := range s.notes {
		if record.UserID != input.UserID {
			continue
		}
		if query != "" && !strings.Contains(strings.ToLower(record.Title), query) &&
			!strings.Contains(strings.ToLower(record.BodyMarkdown), query) {
			continue
		}
		matched = append(matched, record)
	}
	sortNotes(matched, query)
	page := domain.NoteListPage{Notes: []domain.NoteListItem{}, TotalCount: len(matched)}
	for index := offset; index < len(matched) && len(page.Notes) < limit; index++ {
		record := matched[index]
		page.Notes = append(page.Notes, domain.NoteListItem{
			NoteID: record.NoteID, Title: record.Title,
			Snippet: noteMatchSnippet(record.BodyMarkdown, query, 500),
			Pinned:  record.Pinned, Revision: maxInt64(record.Revision, 1),
			UpdatedAt: record.UpdatedAt,
		})
	}
	return page, nil
}

func (s *MemoryStore) SearchNotesForUser(ctx context.Context, input domain.NoteSearchInput) ([]domain.NoteSearchHit, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	limit := input.Limit
	if limit <= 0 || limit > 21 {
		limit = 10
	}
	query := strings.ToLower(strings.TrimSpace(input.Query))
	matched := make([]domain.NoteRecord, 0)
	for _, record := range s.notes {
		if record.UserID == input.UserID &&
			(strings.Contains(strings.ToLower(record.Title), query) || strings.Contains(strings.ToLower(record.BodyMarkdown), query)) {
			matched = append(matched, record)
		}
	}
	sortNotes(matched, query)
	if len(matched) > limit {
		matched = matched[:limit]
	}
	hits := make([]domain.NoteSearchHit, 0, len(matched))
	for _, record := range matched {
		hits = append(hits, domain.NoteSearchHit{
			NoteID: record.NoteID, Title: record.Title,
			Snippet: noteMatchSnippet(record.BodyMarkdown, query, 500),
			Pinned:  record.Pinned, Revision: maxInt64(record.Revision, 1),
			UpdatedAt: record.UpdatedAt,
		})
	}
	return hits, nil
}

func sortNotes(records []domain.NoteRecord, query string) {
	sort.Slice(records, func(i, j int) bool {
		if query != "" {
			iTitle, jTitle := strings.ToLower(records[i].Title), strings.ToLower(records[j].Title)
			iRank, jRank := 2, 2
			if iTitle == query {
				iRank = 0
			} else if strings.Contains(iTitle, query) {
				iRank = 1
			}
			if jTitle == query {
				jRank = 0
			} else if strings.Contains(jTitle, query) {
				jRank = 1
			}
			if iRank != jRank {
				return iRank < jRank
			}
		}
		if records[i].Pinned != records[j].Pinned {
			return records[i].Pinned
		}
		if !records[i].UpdatedAt.Equal(records[j].UpdatedAt) {
			return records[i].UpdatedAt.After(records[j].UpdatedAt)
		}
		return records[i].NoteID < records[j].NoteID
	})
}

func noteMatchSnippet(body string, lowerQuery string, maxRunes int) string {
	if body == "" || maxRunes <= 0 {
		return ""
	}
	startRune := 0
	if lowerQuery != "" {
		bodyRunes := []rune(body)
		queryRunes := []rune(lowerQuery)
		for index := 0; index+len(queryRunes) <= len(bodyRunes); index++ {
			matches := true
			for queryIndex, queryRune := range queryRunes {
				if unicode.ToLower(bodyRunes[index+queryIndex]) != unicode.ToLower(queryRune) {
					matches = false
					break
				}
			}
			if matches {
				startRune = index - 120
				if startRune < 0 {
					startRune = 0
				}
				endRune := startRune + maxRunes
				if endRune > len(bodyRunes) {
					endRune = len(bodyRunes)
				}
				return string(bodyRunes[startRune:endRune])
			}
		}
	}
	start := 0
	for count := 0; count < startRune && start < len(body); count++ {
		_, size := utf8.DecodeRuneInString(body[start:])
		start += size
	}
	end := start
	for count := 0; count < maxRunes && end < len(body); count++ {
		_, size := utf8.DecodeRuneInString(body[end:])
		end += size
	}
	return body[start:end]
}

func maxInt64(value int64, minimum int64) int64 {
	if value < minimum {
		return minimum
	}
	return value
}

func (s *MemoryStore) ConsumeNoteSearchBudget(ctx context.Context, runID string, userID string) error {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	usage := s.noteRunUsage[runID]
	if usage.UserID != "" && usage.UserID != userID {
		return ErrNoteRetrievalBudget
	}
	if usage.SearchCalls >= maxNoteSearchCalls {
		return ErrNoteRetrievalBudget
	}
	usage.UserID = userID
	usage.SearchCalls++
	s.noteRunUsage[runID] = usage
	return nil
}

func (s *MemoryStore) ConsumeNoteReadBudget(ctx context.Context, runID string, userID string, returnedBytes int) error {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	usage := s.noteRunUsage[runID]
	if returnedBytes < 0 || returnedBytes > maxNoteReadCallBytes || (usage.UserID != "" && usage.UserID != userID) ||
		usage.ReadCalls >= maxNoteReadCalls || usage.ReadBytes+returnedBytes > maxNoteReadBytes {
		return ErrNoteRetrievalBudget
	}
	usage.UserID = userID
	usage.ReadCalls++
	usage.ReadBytes += returnedBytes
	s.noteRunUsage[runID] = usage
	return nil
}

func (s *MemoryStore) CreateNoteReadGrant(ctx context.Context, grant domain.NoteReadGrantRecord) error {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	for token, existing := range s.noteReadGrants {
		if existing.UserID == grant.UserID && !existing.ExpiresAt.After(grant.CreatedAt) {
			delete(s.noteReadGrants, token)
		}
	}
	note, ok := s.notes[grant.NoteID]
	if !ok || note.UserID != grant.UserID {
		return ErrNotFound
	}
	ensureNoteIdentity(&note)
	if note.Revision != grant.Revision {
		return ErrNoteRevisionConflict
	}
	if _, exists := s.noteReadGrants[grant.TokenHash]; exists {
		return ErrConflict
	}
	s.noteReadGrants[grant.TokenHash] = grant
	return nil
}

func (s *MemoryStore) ExpireNoteReadGrants(ctx context.Context, now time.Time, limit int) (int, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	if limit <= 0 || limit > 1000 {
		limit = 200
	}
	expired := 0
	for tokenHash, grant := range s.noteReadGrants {
		if expired >= limit {
			break
		}
		if grant.ExpiresAt.After(now) {
			continue
		}
		delete(s.noteReadGrants, tokenHash)
		expired++
	}
	return expired, nil
}

func (s *MemoryStore) CreateNoteAppendProposal(ctx context.Context, input domain.CreateNoteAppendProposalInput) (domain.NoteAppendProposalRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	if strings.TrimSpace(input.IdempotencyKey) == "" || input.RequestDigest == "" ||
		strings.TrimSpace(input.BodyMarkdown) == "" || len(input.BodyMarkdown) > maxNoteAppendBodyBytes ||
		input.ExpectedRevision <= 0 || input.ReadTokenHash == "" {
		return domain.NoteAppendProposalRecord{}, ErrConflict
	}
	bodySHA := domain.NoteBodySHA256(input.BodyMarkdown)
	for _, existing := range s.noteAppendProposals {
		if existing.RunID == input.RunID && existing.UserID == input.UserID && existing.IdempotencyKey == input.IdempotencyKey {
			if existing.RequestDigest != input.RequestDigest {
				return domain.NoteAppendProposalRecord{}, ErrConflict
			}
			return s.publicMemoryNoteProposal(existing), nil
		}
	}
	for _, existing := range s.noteAppendProposals {
		if existing.RunID == input.RunID && existing.UserID == input.UserID && existing.NoteID == input.NoteID &&
			existing.BaseRevision == input.ExpectedRevision && existing.BodySHA256 == bodySHA &&
			existing.Status != domain.NoteAppendProposalStatusExpired {
			if existing.RequestDigest != input.RequestDigest {
				return domain.NoteAppendProposalRecord{}, ErrConflict
			}
			return s.publicMemoryNoteProposal(existing), nil
		}
	}
	grant, ok := s.noteReadGrants[input.ReadTokenHash]
	if !ok || grant.RunID != input.RunID || grant.UserID != input.UserID || grant.NoteID != input.NoteID ||
		grant.Revision != input.ExpectedRevision || !grant.ExpiresAt.After(input.Now) {
		return domain.NoteAppendProposalRecord{}, ErrNoteReadTokenInvalid
	}
	note, ok := s.notes[input.NoteID]
	if !ok || note.UserID != input.UserID {
		return domain.NoteAppendProposalRecord{}, ErrNotFound
	}
	ensureNoteIdentity(&note)
	if note.Revision != input.ExpectedRevision {
		return domain.NoteAppendProposalRecord{}, ErrNoteRevisionConflict
	}
	if _, exists := s.noteAppendProposals[input.ProposalID]; exists {
		return domain.NoteAppendProposalRecord{}, ErrConflict
	}
	proposal := domain.NoteAppendProposalRecord{
		ProposalID: input.ProposalID, RunID: input.RunID, UserID: input.UserID,
		NoteID: input.NoteID, NoteTitle: note.Title, BaseRevision: input.ExpectedRevision,
		BodyMarkdown: input.BodyMarkdown, BodySHA256: bodySHA,
		IdempotencyKey: input.IdempotencyKey, RequestDigest: input.RequestDigest,
		Status: domain.NoteAppendProposalStatusPending, ExpiresAt: input.ExpiresAt,
		CreatedAt: input.Now, UpdatedAt: input.Now,
	}
	storedProposal := proposal
	storedProposal.NoteTitle = ""
	s.noteAppendProposals[proposal.ProposalID] = storedProposal
	return proposal, nil
}

// Proposal titles are projections of the live Note, not retained proposal
// metadata. This keeps the memory twin aligned with the Postgres join and
// prevents long-lived expired/committed metadata from preserving a title.
func (s *MemoryStore) publicMemoryNoteProposal(proposal domain.NoteAppendProposalRecord) domain.NoteAppendProposalRecord {
	proposal.NoteTitle = ""
	if note, ok := s.notes[proposal.NoteID]; ok && note.UserID == proposal.UserID {
		proposal.NoteTitle = note.Title
	}
	return proposal
}

func (s *MemoryStore) GetNoteAppendProposalForUser(ctx context.Context, proposalID string, userID string) (domain.NoteAppendProposalRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	proposal, ok := s.noteAppendProposals[proposalID]
	if !ok || proposal.UserID != userID {
		return domain.NoteAppendProposalRecord{}, ErrNotFound
	}
	if proposal.Status == domain.NoteAppendProposalStatusPending && !proposal.ExpiresAt.After(domain.Now()) {
		proposal.Status = domain.NoteAppendProposalStatusExpired
		proposal.BodyMarkdown = ""
		proposal.UpdatedAt = domain.Now()
		s.noteAppendProposals[proposalID] = proposal
	}
	return s.publicMemoryNoteProposal(proposal), nil
}

func (s *MemoryStore) ExpireNoteAppendProposals(ctx context.Context, now time.Time, limit int) (int, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	if limit <= 0 || limit > 1000 {
		limit = 200
	}
	expired := 0
	for proposalID, proposal := range s.noteAppendProposals {
		if expired >= limit {
			break
		}
		if proposal.Status != domain.NoteAppendProposalStatusPending || proposal.ExpiresAt.After(now) {
			continue
		}
		proposal.Status = domain.NoteAppendProposalStatusExpired
		proposal.BodyMarkdown = ""
		proposal.UpdatedAt = now
		s.noteAppendProposals[proposalID] = proposal
		expired++
	}
	return expired, nil
}

func (s *MemoryStore) CommitNoteAppendProposalForUser(ctx context.Context, input domain.CommitNoteAppendProposalInput) (domain.NoteAppendOperationRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	proposal, ok := s.noteAppendProposals[input.ProposalID]
	if !ok || proposal.UserID != input.UserID {
		return domain.NoteAppendOperationRecord{}, ErrNotFound
	}
	if proposal.Status == domain.NoteAppendProposalStatusCommitted {
		if input.BodyMarkdown != nil && domain.NoteBodySHA256(*input.BodyMarkdown) != proposal.CommittedBodySHA256 {
			return domain.NoteAppendOperationRecord{}, ErrConflict
		}
		operation, ok := s.noteAppendOperations[proposal.OperationID]
		if !ok {
			return domain.NoteAppendOperationRecord{}, ErrNotFound
		}
		return s.publicMemoryNoteOperation(operation), nil
	}
	if proposal.Status != domain.NoteAppendProposalStatusPending || !proposal.ExpiresAt.After(input.Now) {
		proposal.Status = domain.NoteAppendProposalStatusExpired
		proposal.BodyMarkdown = ""
		proposal.UpdatedAt = input.Now
		s.noteAppendProposals[proposal.ProposalID] = proposal
		return domain.NoteAppendOperationRecord{}, ErrNoteProposalExpired
	}
	body := proposal.BodyMarkdown
	if input.BodyMarkdown != nil {
		body = *input.BodyMarkdown
	}
	if strings.TrimSpace(body) == "" || len(body) > maxNoteAppendBodyBytes {
		return domain.NoteAppendOperationRecord{}, ErrConflict
	}
	note, ok := s.notes[proposal.NoteID]
	if !ok || note.UserID != input.UserID {
		return domain.NoteAppendOperationRecord{}, ErrNotFound
	}
	ensureNoteIdentity(&note)
	if note.Revision != proposal.BaseRevision {
		return domain.NoteAppendOperationRecord{}, ErrNoteRevisionConflict
	}
	suffix := noteAppendSuffix(note.BodyMarkdown, body)
	if len(note.BodyMarkdown)+len(suffix) > maxStoredNoteBodyBytes {
		return domain.NoteAppendOperationRecord{}, ErrConflict
	}
	if _, exists := s.noteAppendOperations[input.OperationID]; exists {
		return domain.NoteAppendOperationRecord{}, ErrConflict
	}
	startByte := len(note.BodyMarkdown)
	beforeDigest := note.ContentDigest
	note.BodyMarkdown += suffix
	note.Revision++
	note.ContentDigest = domain.ComputeNoteContentDigest(note.Title, note.BodyMarkdown)
	note.UpdatedAt = input.Now
	operation := memoryNoteAppendOperation{
		NoteAppendOperationRecord: domain.NoteAppendOperationRecord{
			OperationID: input.OperationID, ProposalID: proposal.ProposalID, RunID: proposal.RunID,
			NoteID: note.NoteID, NoteTitle: note.Title, UserID: input.UserID,
			BeforeRevision: proposal.BaseRevision, AfterRevision: note.Revision,
			AppendedBytes: len(suffix), BeforeContentDigest: beforeDigest,
			AfterContentDigest: note.ContentDigest, CreatedAt: input.Now,
		},
		AppendStartByte: startByte, AppendSHA256: domain.NoteBodySHA256(suffix),
	}
	s.notes[note.NoteID] = note
	storedOperation := operation
	storedOperation.NoteTitle = ""
	s.noteAppendOperations[operation.OperationID] = storedOperation
	proposal.Status = domain.NoteAppendProposalStatusCommitted
	proposal.BodyMarkdown = ""
	proposal.CommittedBodySHA256 = domain.NoteBodySHA256(body)
	proposal.OperationID = operation.OperationID
	proposal.UpdatedAt = input.Now
	s.noteAppendProposals[proposal.ProposalID] = proposal
	return operation.NoteAppendOperationRecord, nil
}

// Operation receipts retain only content-free metadata. The current Note title
// is joined into browser responses transiently, matching the Postgres store.
func (s *MemoryStore) publicMemoryNoteOperation(operation memoryNoteAppendOperation) domain.NoteAppendOperationRecord {
	public := operation.NoteAppendOperationRecord
	public.NoteTitle = ""
	if note, ok := s.notes[operation.NoteID]; ok && note.UserID == operation.UserID {
		public.NoteTitle = note.Title
	}
	return public
}

func (s *MemoryStore) GetNoteAppendOperationForUser(ctx context.Context, operationID string, userID string) (domain.NoteAppendOperationRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	operation, ok := s.noteAppendOperations[operationID]
	if !ok || operation.UserID != userID {
		return domain.NoteAppendOperationRecord{}, ErrNotFound
	}
	return s.publicMemoryNoteOperation(operation), nil
}

func (s *MemoryStore) UndoNoteAppendOperationForUser(ctx context.Context, input domain.UndoNoteAppendOperationInput) (domain.NoteAppendOperationRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	operation, ok := s.noteAppendOperations[input.OperationID]
	if !ok || operation.UserID != input.UserID {
		return domain.NoteAppendOperationRecord{}, ErrNotFound
	}
	if operation.UndoneAt != nil {
		return s.publicMemoryNoteOperation(operation), nil
	}
	note, ok := s.notes[operation.NoteID]
	if !ok || note.UserID != input.UserID {
		return domain.NoteAppendOperationRecord{}, ErrNotFound
	}
	ensureNoteIdentity(&note)
	endByte := operation.AppendStartByte + operation.AppendedBytes
	if note.Revision != operation.AfterRevision || operation.AppendStartByte < 0 || operation.AppendedBytes <= 0 ||
		endByte != len(note.BodyMarkdown) || domain.NoteBodySHA256(note.BodyMarkdown[operation.AppendStartByte:endByte]) != operation.AppendSHA256 {
		return domain.NoteAppendOperationRecord{}, ErrNoteUndoConflict
	}
	note.BodyMarkdown = note.BodyMarkdown[:operation.AppendStartByte]
	note.Revision++
	note.ContentDigest = domain.ComputeNoteContentDigest(note.Title, note.BodyMarkdown)
	note.UpdatedAt = input.Now
	undoneAt := input.Now
	operation.UndoneAt = &undoneAt
	operation.UndoRevision = note.Revision
	operation.NoteTitle = ""
	s.notes[note.NoteID] = note
	s.noteAppendOperations[input.OperationID] = operation
	return s.publicMemoryNoteOperation(operation), nil
}
