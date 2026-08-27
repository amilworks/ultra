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
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

// noteCapabilityStore is the optional store capability behind the Notes
// surface — the same pattern as the other capability interfaces: implemented
// by both the Postgres store and the memory twin, type-asserted off
// deps.Store so deployments without it degrade to 501 instead of panicking.
type noteCapabilityStore interface {
	CreateNote(ctx context.Context, record domain.NoteRecord) (domain.NoteRecord, error)
	CreateNoteForUserIdempotent(ctx context.Context, input domain.CreateNoteIdempotentInput) (domain.NoteRecord, bool, error)
	FindNoteCreateReplayForUser(ctx context.Context, userID string, idempotencyKey string, requestDigest string) (domain.NoteRecord, bool, error)
	GetNoteForUser(ctx context.Context, noteID string, userID string) (domain.NoteRecord, error)
	UpdateNoteForUser(ctx context.Context, noteID string, userID string, input domain.NoteUpdateInput) (domain.NoteRecord, error)
	DeleteNoteForUser(ctx context.Context, noteID string, userID string) error
	ListNotesForUser(ctx context.Context, input domain.NoteListInput) (domain.NoteListPage, error)
	SearchNotesForUser(ctx context.Context, input domain.NoteSearchInput) (domain.NoteSearchPage, error)
	ConsumeNoteSearchBudget(ctx context.Context, runID string, userID string) error
	ConsumeNoteReadBudget(ctx context.Context, runID string, userID string, returnedBytes int) error
	CreateNoteReadGrant(ctx context.Context, grant domain.NoteReadGrantRecord) error
	CreateNoteAppendProposal(ctx context.Context, input domain.CreateNoteAppendProposalInput) (domain.NoteAppendProposalRecord, error)
	GetNoteAppendProposalForUser(ctx context.Context, proposalID string, userID string) (domain.NoteAppendProposalRecord, error)
	GetNoteAppendOperationForUser(ctx context.Context, operationID string, userID string) (domain.NoteAppendOperationRecord, error)
	CommitNoteAppendProposalForUser(ctx context.Context, input domain.CommitNoteAppendProposalInput) (domain.NoteAppendOperationRecord, error)
	UndoNoteAppendOperationForUser(ctx context.Context, input domain.UndoNoteAppendOperationInput) (domain.NoteAppendOperationRecord, error)
	DirectAppendNoteForUser(ctx context.Context, input domain.DirectNoteAppendInput) (domain.NoteDirectAppendOperationRecord, bool, error)
	FindNoteDirectAppendReplayForUser(ctx context.Context, userID string, idempotencyKey string, requestDigest string) (domain.NoteDirectAppendOperationRecord, bool, error)
	UndoDirectNoteAppendForUser(ctx context.Context, input domain.UndoDirectNoteAppendInput) (domain.NoteDirectAppendOperationRecord, error)
}

const (
	maxNoteTitleLength = 512
	// maxNoteBodyBytes bounds a single note body (2 MB of markdown is a
	// document, not a note; anything bigger belongs in Resources).
	maxNoteBodyBytes                       = 2 << 20
	maxNoteCreateIdempotencyKeyBytes       = 256
	maxDirectNoteAppendBodyBytes           = 32 << 10
	maxDirectNoteAppendIdempotencyKeyBytes = 256
	noteCreateNotCommittedCode             = "note_create_not_committed"
	noteAppendNotCommittedCode             = "note_append_not_committed"
	noteCreateReplayDeletedCode            = "note_create_replay_deleted"
	noteAppendTargetUnavailableCode        = "note_append_target_unavailable"
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

func writeNoteNotCommitted(w http.ResponseWriter, code string, err error) {
	writeJSON(w, http.StatusBadRequest, map[string]any{
		"error": err.Error(),
		"code":  code,
	})
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
	sortMode := strings.TrimSpace(r.URL.Query().Get("sort"))
	if sortMode == "" {
		sortMode = string(domain.NoteListSortBrowse)
	}
	if sortMode != string(domain.NoteListSortBrowse) && sortMode != string(domain.NoteListSortRecent) {
		writeError(w, http.StatusBadRequest, errors.New(`sort must be "browse" or "recent"`))
		return
	}
	page, err := capability.ListNotesForUser(r.Context(), domain.NoteListInput{
		UserID: principal.UserID,
		Query:  r.URL.Query().Get("query"),
		Sort:   domain.NoteListSort(sortMode),
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
	status := http.StatusCreated
	var created domain.NoteRecord
	var err error
	idempotencyHeaders := r.Header.Values("Idempotency-Key")
	if len(idempotencyHeaders) > 0 {
		idempotencyKey := ""
		if len(idempotencyHeaders) == 1 {
			idempotencyKey = strings.TrimSpace(idempotencyHeaders[0])
		}
		if idempotencyKey == "" || len(idempotencyKey) > maxNoteCreateIdempotencyKeyBytes {
			writeError(w, http.StatusBadRequest, errors.New("Idempotency-Key header must contain between 1 and 256 bytes"))
			return
		}
		requestDigest := domain.ComputeNoteCreateRequestDigest(
			record.UserID, record.OrgID, record.Title, record.BodyMarkdown, record.Pinned, record.EditorMode,
		)
		replay, found, lookupErr := capability.FindNoteCreateReplayForUser(
			r.Context(), record.UserID, idempotencyKey, requestDigest,
		)
		if lookupErr != nil {
			writeNoteStoreError(w, lookupErr)
			return
		}
		if found {
			writeJSON(w, http.StatusOK, replay)
			return
		}
		if validationErr := validateNoteWrite(req); validationErr != nil {
			writeNoteNotCommitted(w, noteCreateNotCommittedCode, validationErr)
			return
		}
		var first bool
		created, first, err = capability.CreateNoteForUserIdempotent(r.Context(), domain.CreateNoteIdempotentInput{
			Record: record, IdempotencyKey: idempotencyKey, RequestDigest: requestDigest,
		})
		if !first {
			status = http.StatusOK
		}
	} else {
		if validationErr := validateNoteWrite(req); validationErr != nil {
			writeError(w, http.StatusBadRequest, validationErr)
			return
		}
		created, err = capability.CreateNote(r.Context(), record)
	}
	if err != nil {
		writeNoteStoreError(w, err)
		return
	}
	writeJSON(w, status, created)
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

type directNoteAppendRequest struct {
	BodyMarkdown     string `json:"body_markdown"`
	ExpectedRevision int64  `json:"expected_revision"`
}

func (deps ServerDeps) handleDirectNoteAppend(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	if deps.workerRequestAuth(r) != workerAuthAbsent {
		writeError(w, http.StatusUnauthorized, errors.New("browser authentication required"))
		return
	}
	var req directNoteAppendRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	idempotencyHeaders := r.Header.Values("Idempotency-Key")
	idempotencyKey := ""
	if len(idempotencyHeaders) == 1 {
		idempotencyKey = strings.TrimSpace(idempotencyHeaders[0])
	}
	if idempotencyKey == "" || len(idempotencyKey) > maxDirectNoteAppendIdempotencyKeyBytes {
		writeError(w, http.StatusBadRequest, errors.New("Idempotency-Key header must contain between 1 and 256 bytes"))
		return
	}
	capability, ok := deps.notesCapability(w)
	if !ok {
		return
	}
	principal := deps.principalFromRequest(r, "")
	noteID := strings.TrimSpace(chi.URLParam(r, "note_id"))
	requestDigest := domain.ComputeNoteDirectAppendRequestDigest(
		principal.UserID, noteID, req.ExpectedRevision, req.BodyMarkdown,
	)
	replay, found, lookupErr := capability.FindNoteDirectAppendReplayForUser(
		r.Context(), principal.UserID, idempotencyKey, requestDigest,
	)
	if lookupErr != nil {
		writeNoteStoreError(w, lookupErr)
		return
	}
	if found {
		writeJSON(w, http.StatusOK, replay)
		return
	}
	if req.ExpectedRevision <= 0 {
		writeNoteNotCommitted(w, noteAppendNotCommittedCode, errors.New("expected_revision must be positive"))
		return
	}
	if strings.TrimSpace(req.BodyMarkdown) == "" || len(req.BodyMarkdown) > maxDirectNoteAppendBodyBytes {
		writeNoteNotCommitted(w, noteAppendNotCommittedCode, errors.New("body_markdown must be non-empty and no larger than 32 KiB"))
		return
	}
	receipt, created, err := capability.DirectAppendNoteForUser(r.Context(), domain.DirectNoteAppendInput{
		OperationID: domain.NewID("ndop"), UserID: principal.UserID, NoteID: noteID,
		ExpectedRevision: req.ExpectedRevision, BodyMarkdown: req.BodyMarkdown,
		IdempotencyKey: idempotencyKey, RequestDigest: requestDigest, Now: domain.Now(),
	})
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			writeJSON(w, http.StatusNotFound, map[string]any{
				"error": "note append target is unavailable", "code": noteAppendTargetUnavailableCode,
			})
			return
		}
		writeNoteStoreError(w, err)
		return
	}
	status := http.StatusOK
	if created {
		status = http.StatusCreated
	}
	writeJSON(w, status, receipt)
}

func (deps ServerDeps) handleUndoDirectNoteAppend(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	if deps.workerRequestAuth(r) != workerAuthAbsent {
		writeError(w, http.StatusUnauthorized, errors.New("browser authentication required"))
		return
	}
	capability, ok := deps.notesCapability(w)
	if !ok {
		return
	}
	principal := deps.principalFromRequest(r, "")
	receipt, err := capability.UndoDirectNoteAppendForUser(r.Context(), domain.UndoDirectNoteAppendInput{
		OperationID: chi.URLParam(r, "operation_id"), UserID: principal.UserID, Now: domain.Now(),
	})
	if err != nil {
		writeNoteStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, receipt)
}
