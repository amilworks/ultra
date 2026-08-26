package httpapi

import (
	"context"
	"errors"
	"net/http"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/go-chi/chi/v5"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

// noteCapabilityStore is the optional store capability behind the Notes
// surface — the same pattern as the other capability interfaces: implemented
// by both the Postgres store and the memory twin, type-asserted off
// deps.Store so deployments without it degrade to 501 instead of panicking.
type noteCapabilityStore interface {
	CreateNote(ctx context.Context, record domain.NoteRecord) (domain.NoteRecord, error)
	GetNoteForUser(ctx context.Context, noteID string, userID string) (domain.NoteRecord, error)
	UpdateNoteForUser(ctx context.Context, noteID string, userID string, input domain.NoteUpdateInput) (domain.NoteRecord, error)
	DeleteNoteForUser(ctx context.Context, noteID string, userID string) error
	ListNotesForUser(ctx context.Context, input domain.NoteListInput) (domain.NoteListPage, error)
	SearchNotesForUser(ctx context.Context, input domain.NoteSearchInput) ([]domain.NoteSearchHit, error)
	ConsumeNoteSearchBudget(ctx context.Context, runID string, userID string) error
	ConsumeNoteReadBudget(ctx context.Context, runID string, userID string, returnedBytes int) error
	CreateNoteReadGrant(ctx context.Context, grant domain.NoteReadGrantRecord) error
	CreateNoteAppendProposal(ctx context.Context, input domain.CreateNoteAppendProposalInput) (domain.NoteAppendProposalRecord, error)
	GetNoteAppendProposalForUser(ctx context.Context, proposalID string, userID string) (domain.NoteAppendProposalRecord, error)
	GetNoteAppendOperationForUser(ctx context.Context, operationID string, userID string) (domain.NoteAppendOperationRecord, error)
	CommitNoteAppendProposalForUser(ctx context.Context, input domain.CommitNoteAppendProposalInput) (domain.NoteAppendOperationRecord, error)
	UndoNoteAppendOperationForUser(ctx context.Context, input domain.UndoNoteAppendOperationInput) (domain.NoteAppendOperationRecord, error)
}

const (
	maxNoteTitleLength = 512
	// maxNoteBodyBytes bounds a single note body (2 MB of markdown is a
	// document, not a note; anything bigger belongs in Resources).
	maxNoteBodyBytes = 2 << 20
)

type noteWriteRequest struct {
	Title            *string `json:"title"`
	BodyMarkdown     *string `json:"body_markdown"`
	Pinned           *bool   `json:"pinned"`
	EditorMode       *string `json:"editor_mode"`
	ExpectedRevision *int64  `json:"expected_revision"`
}

func (deps ServerDeps) notesCapability(w http.ResponseWriter) (noteCapabilityStore, bool) {
	capability, ok := deps.Store.(noteCapabilityStore)
	if !ok {
		writeError(w, http.StatusNotImplemented, errors.New("notes are not available on this deployment"))
		return nil, false
	}
	return capability, true
}

func validateNoteWrite(req noteWriteRequest) error {
	if req.Title != nil && utf8.RuneCountInString(*req.Title) > maxNoteTitleLength {
		return errors.New("title is too long")
	}
	if req.BodyMarkdown != nil && len(*req.BodyMarkdown) > maxNoteBodyBytes {
		return errors.New("note body exceeds the 2 MB limit — larger documents belong in Resources")
	}
	if req.EditorMode != nil &&
		*req.EditorMode != domain.NoteEditorModeMarkdown &&
		*req.EditorMode != domain.NoteEditorModePlaintext {
		return errors.New(`editor_mode must be "markdown" or "plaintext"`)
	}
	return nil
}

func (deps ServerDeps) handleListNotes(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	capability, ok := deps.notesCapability(w)
	if !ok {
		return
	}
	principal := deps.principalFromRequest(r, "")
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	offset, _ := strconv.Atoi(r.URL.Query().Get("offset"))
	page, err := capability.ListNotesForUser(r.Context(), domain.NoteListInput{
		UserID: principal.UserID,
		Query:  r.URL.Query().Get("query"),
		Limit:  limit,
		Offset: offset,
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"notes":       page.Notes,
		"total_count": page.TotalCount,
	})
}

func (deps ServerDeps) handleCreateNote(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	capability, ok := deps.notesCapability(w)
	if !ok {
		return
	}
	var req noteWriteRequest
	if r.Body != nil && r.ContentLength != 0 {
		if !decodeJSON(w, r, &req) {
			return
		}
	}
	if err := validateNoteWrite(req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	principal := deps.principalFromRequest(r, "")
	record := domain.NoteRecord{
		NoteID:     domain.NewID("note"),
		UserID:     principal.UserID,
		OrgID:      principal.OrgID,
		EditorMode: domain.NoteEditorModeMarkdown,
		CreatedAt:  domain.Now(),
	}
	if req.Title != nil {
		record.Title = strings.TrimSpace(*req.Title)
	}
	if req.BodyMarkdown != nil {
		record.BodyMarkdown = *req.BodyMarkdown
	}
	if req.Pinned != nil {
		record.Pinned = *req.Pinned
	}
	if req.EditorMode != nil {
		record.EditorMode = *req.EditorMode
	}
	created, err := capability.CreateNote(r.Context(), record)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, created)
}

func (deps ServerDeps) handleGetNote(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	capability, ok := deps.notesCapability(w)
	if !ok {
		return
	}
	principal := deps.principalFromRequest(r, "")
	record, err := capability.GetNoteForUser(r.Context(), chi.URLParam(r, "note_id"), principal.UserID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, record)
}

func (deps ServerDeps) handleUpdateNote(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	capability, ok := deps.notesCapability(w)
	if !ok {
		return
	}
	var req noteWriteRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	if err := validateNoteWrite(req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if req.ExpectedRevision != nil && *req.ExpectedRevision <= 0 {
		writeError(w, http.StatusBadRequest, errors.New("expected_revision must be positive"))
		return
	}
	principal := deps.principalFromRequest(r, "")
	expectedRevision := int64(0)
	if req.ExpectedRevision != nil {
		expectedRevision = *req.ExpectedRevision
	} else {
		if deps.noteModelFeatures.requireExpectedRevision {
			writeError(w, http.StatusBadRequest, errors.New("expected_revision is required"))
			return
		}
		// Transitional rollout compatibility for already-open browser bundles.
		// The owner lookup avoids inventing authority and the actual write still
		// uses the exact same atomic CAS path. This mode is temporary last-write-
		// wins behavior and must be disabled before model append proposals.
		current, err := capability.GetNoteForUser(r.Context(), chi.URLParam(r, "note_id"), principal.UserID)
		if err != nil {
			writeStoreError(w, err)
			return
		}
		expectedRevision = current.Revision
	}
	var title *string
	if req.Title != nil {
		trimmed := strings.TrimSpace(*req.Title)
		title = &trimmed
	}
	record, err := capability.UpdateNoteForUser(r.Context(), chi.URLParam(r, "note_id"), principal.UserID, domain.NoteUpdateInput{
		ExpectedRevision: expectedRevision,
		Title:            title,
		BodyMarkdown:     req.BodyMarkdown,
		Pinned:           req.Pinned,
		EditorMode:       req.EditorMode,
	})
	if err != nil {
		writeNoteStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, record)
}

func (deps ServerDeps) handleDeleteNote(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	capability, ok := deps.notesCapability(w)
	if !ok {
		return
	}
	principal := deps.principalFromRequest(r, "")
	if err := capability.DeleteNoteForUser(r.Context(), chi.URLParam(r, "note_id"), principal.UserID); err != nil {
		writeStoreError(w, err)
		return
	}
	// Hard deletion: the note is erased, not concealed.
	writeJSON(w, http.StatusOK, map[string]string{"status": "deleted"})
}
