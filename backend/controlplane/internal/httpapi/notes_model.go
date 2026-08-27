package httpapi

import (
	"bytes"
	"context"
	"crypto/subtle"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/go-chi/chi/v5"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

const (
	maxModelNoteQueryRunes      = 512
	defaultModelNoteSearchLimit = 8
	maxModelNoteSearchLimit     = 20
	defaultModelNoteReadChars   = 8000
	maxModelNoteReadChars       = 16000
	maxModelNoteAppendBodyBytes = 32 << 10
	maxNoteProposalIdempotency  = 256
	noteReadGrantLifetime       = 10 * time.Minute
	noteAppendProposalLifetime  = 15 * time.Minute
)

type noteModelFeatureConfig struct {
	initialized             bool
	readEnabled             bool
	proposalEnabled         bool
	requireExpectedRevision bool
}

func (config noteModelFeatureConfig) proposalsAvailable() bool {
	return config.proposalEnabled && config.requireExpectedRevision
}

func newNoteModelFeatureConfigFromEnv() noteModelFeatureConfig {
	return noteModelFeatureConfig{
		initialized:             true,
		readEnabled:             envFeatureEnabled("ULTRA_CONTROL_MODEL_NOTES_READ_ENABLED"),
		proposalEnabled:         envFeatureEnabled("ULTRA_CONTROL_MODEL_NOTES_PROPOSALS_ENABLED"),
		requireExpectedRevision: envFeatureEnabled("ULTRA_CONTROL_NOTES_REQUIRE_EXPECTED_REVISION"),
	}
}

func envFeatureEnabled(name string) bool {
	raw, exists := os.LookupEnv(name)
	return featureSettingEnabled(raw, exists)
}

func featureSettingEnabled(raw string, exists bool) bool {
	if !exists {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "1", "true", "on", "yes", "enabled":
		return true
	default:
		return false
	}
}

type noteWorkerAuthority struct {
	Run   domain.RunRecord
	Scope domain.NoteAccessScope
}

func (deps ServerDeps) authorizeNoteWorkerRequest(w http.ResponseWriter, r *http.Request) (noteWorkerAuthority, bool) {
	if deps.workerRequestAuth(r) != workerAuthValid {
		writeError(w, http.StatusUnauthorized, errors.New("valid worker token required"))
		return noteWorkerAuthority{}, false
	}
	if !isWorkerScopedEndpoint(r) {
		writeError(w, http.StatusUnauthorized, errors.New("worker credential is not authorized for this endpoint"))
		return noteWorkerAuthority{}, false
	}
	pathRunID := strings.TrimSpace(chi.URLParam(r, "run_id"))
	headerRunID := strings.TrimSpace(r.Header.Get("X-Ultra-Run-Id"))
	if pathRunID == "" || headerRunID == "" || pathRunID != headerRunID || deps.Store == nil {
		writeError(w, http.StatusUnauthorized, errors.New("active run lease required"))
		return noteWorkerAuthority{}, false
	}
	run, err := deps.Store.GetRun(r.Context(), pathRunID)
	if err != nil || run.Status != domain.RunStatusRunning || strings.TrimSpace(run.UserID) == "" {
		writeError(w, http.StatusUnauthorized, errors.New("active run lease required"))
		return noteWorkerAuthority{}, false
	}
	if rawProfile, exists := run.Metadata[domain.EvaluationProfileMetadataKey]; exists {
		profile, ok := rawProfile.(string)
		if !ok || strings.TrimSpace(profile) != "" {
			writeError(w, http.StatusForbidden, errors.New("protected runs cannot access Notes"))
			return noteWorkerAuthority{}, false
		}
	}
	principalMetadata, _ := jsonMapValue(run.Metadata["principal"])
	role, _ := safeMetadataString(principalMetadata["role"], 128)
	if trustedRunOrgID(run) == "" || role == "" {
		writeError(w, http.StatusUnauthorized, errors.New("trusted run principal required"))
		return noteWorkerAuthority{}, false
	}
	lease, found, err := deps.Store.GetRunLease(r.Context(), run.RunID)
	if err != nil || !found || !lease.LeaseExpiresAt.After(domain.Now()) {
		writeError(w, http.StatusUnauthorized, errors.New("active run lease required"))
		return noteWorkerAuthority{}, false
	}
	workerID := strings.TrimSpace(r.Header.Get("X-Ultra-Worker-Id"))
	leaseToken := strings.TrimSpace(r.Header.Get("X-Ultra-Run-Lease-Token"))
	if workerID == "" || workerID != lease.WorkerID || leaseToken == "" ||
		subtle.ConstantTimeCompare([]byte(leaseToken), []byte(lease.LeaseToken)) != 1 {
		writeError(w, http.StatusUnauthorized, errors.New("active run lease required"))
		return noteWorkerAuthority{}, false
	}
	scope, ok := domain.NoteAccessScopeFromRun(run)
	if !ok {
		writeError(w, http.StatusForbidden, errors.New("run is not authorized to access Notes"))
		return noteWorkerAuthority{}, false
	}
	return noteWorkerAuthority{Run: run, Scope: scope}, true
}

type modelNoteSearchRequest struct {
	Query  string `json:"query"`
	Sort   string `json:"sort"`
	Limit  int    `json:"limit"`
	Cursor string `json:"cursor"`
}

type modelNoteSearchCursor struct {
	Version          int       `json:"v"`
	RunDigest        string    `json:"r"`
	QueryDigest      string    `json:"q"`
	Sort             string    `json:"s"`
	SnapshotAt       time.Time `json:"a"`
	Rank             int       `json:"k"`
	ContentUpdatedAt time.Time `json:"t"`
	NoteID           string    `json:"n"`
}

func (deps ServerDeps) handleModelNoteSearch(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	if !deps.noteModelFeatures.readEnabled {
		writeError(w, http.StatusServiceUnavailable, errors.New("model Notes access is disabled"))
		return
	}
	authority, ok := deps.authorizeNoteWorkerRequest(w, r)
	if !ok {
		return
	}
	if authority.Scope.Mode != domain.NoteAccessModeSearch {
		writeError(w, http.StatusForbidden, errors.New("run is not authorized to search Notes"))
		return
	}
	var req modelNoteSearchRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	req.Query = strings.TrimSpace(req.Query)
	req.Sort = strings.TrimSpace(req.Sort)
	if req.Sort == "" {
		req.Sort = string(domain.NoteSearchSortRelevance)
	}
	if req.Sort != string(domain.NoteSearchSortRelevance) && req.Sort != string(domain.NoteSearchSortRecent) {
		writeError(w, http.StatusBadRequest, errors.New(`sort must be "relevance" or "recent"`))
		return
	}
	if utf8.RuneCountInString(req.Query) > maxModelNoteQueryRunes ||
		(req.Query == "" && req.Sort != string(domain.NoteSearchSortRecent)) {
		writeError(w, http.StatusBadRequest, errors.New("query must contain between 1 and 512 characters unless sort is recent"))
		return
	}
	limit := req.Limit
	if limit <= 0 {
		limit = defaultModelNoteSearchLimit
	}
	if limit > maxModelNoteSearchLimit {
		limit = maxModelNoteSearchLimit
	}
	var snapshotAt time.Time
	var after *domain.NoteSearchPageAnchor
	if strings.TrimSpace(req.Cursor) != "" {
		cursor, err := decodeModelNoteSearchCursor(req.Cursor)
		if err != nil || cursor.Sort != req.Sort ||
			subtle.ConstantTimeCompare([]byte(cursor.RunDigest), []byte(domain.NoteBodySHA256(authority.Run.RunID))) != 1 ||
			subtle.ConstantTimeCompare([]byte(cursor.QueryDigest), []byte(domain.ComputeNoteContentDigest(req.Sort, req.Query))) != 1 {
			writeError(w, http.StatusBadRequest, errors.New("invalid or mismatched note search cursor"))
			return
		}
		snapshotAt = cursor.SnapshotAt
		after = &domain.NoteSearchPageAnchor{
			Rank: cursor.Rank, ContentUpdatedAt: cursor.ContentUpdatedAt, NoteID: cursor.NoteID,
		}
	}
	capability, ok := deps.notesCapability(w)
	if !ok {
		return
	}
	if err := capability.ConsumeNoteSearchBudget(r.Context(), authority.Run.RunID, authority.Run.UserID); err != nil {
		writeNoteStoreError(w, err)
		return
	}
	page, err := capability.SearchNotesForUser(r.Context(), domain.NoteSearchInput{
		UserID: authority.Run.UserID, Query: req.Query,
		Sort: domain.NoteSearchSort(req.Sort), Limit: limit + 1,
		SnapshotAt: snapshotAt, After: after,
	})
	if err != nil {
		writeNoteStoreError(w, err)
		return
	}
	hits := page.Notes
	hasMore := len(hits) > limit
	if hasMore {
		hits = hits[:limit]
	}
	nextCursor := ""
	if hasMore {
		nextCursor = encodeModelNoteSearchCursor(
			authority.Run.RunID, req.Query, req.Sort, page.SnapshotAt, hits[len(hits)-1],
		)
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"notes": hits, "has_more": hasMore, "next_cursor": nextCursor,
	})
}

func encodeModelNoteSearchCursor(runID string, query string, sortMode string, snapshotAt time.Time, last domain.NoteSearchHit) string {
	payload, _ := json.Marshal(modelNoteSearchCursor{
		Version: 2, RunDigest: domain.NoteBodySHA256(runID),
		QueryDigest: domain.ComputeNoteContentDigest(sortMode, query),
		Sort:        sortMode, SnapshotAt: snapshotAt, Rank: last.SortRank,
		ContentUpdatedAt: last.ContentUpdatedAt, NoteID: last.NoteID,
	})
	return base64.RawURLEncoding.EncodeToString(payload)
}

func decodeModelNoteSearchCursor(value string) (modelNoteSearchCursor, error) {
	if len(value) > 2048 {
		return modelNoteSearchCursor{}, errors.New("cursor is too long")
	}
	payload, err := base64.RawURLEncoding.DecodeString(value)
	if err != nil {
		return modelNoteSearchCursor{}, err
	}
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	var cursor modelNoteSearchCursor
	if err := decoder.Decode(&cursor); err != nil {
		return modelNoteSearchCursor{}, err
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return modelNoteSearchCursor{}, errors.New("cursor has trailing data")
	}
	if cursor.Version != 2 || len(cursor.RunDigest) != 64 || len(cursor.QueryDigest) != 64 ||
		(cursor.Sort != string(domain.NoteSearchSortRelevance) && cursor.Sort != string(domain.NoteSearchSortRecent)) ||
		cursor.SnapshotAt.IsZero() || cursor.ContentUpdatedAt.IsZero() || cursor.ContentUpdatedAt.After(cursor.SnapshotAt) ||
		cursor.NoteID == "" || len(cursor.NoteID) > 256 ||
		cursor.Rank < 0 || cursor.Rank > 2 ||
		(cursor.Sort == string(domain.NoteSearchSortRecent) && cursor.Rank != 0) {
		return modelNoteSearchCursor{}, errors.New("invalid cursor fields")
	}
	return cursor, nil
}

type modelNoteReadRequest struct {
	NoteID   string `json:"note_id"`
	Cursor   string `json:"cursor"`
	MaxChars int    `json:"max_chars"`
}

type noteReadCursor struct {
	NoteID   string `json:"n"`
	Revision int64  `json:"r"`
	Offset   int    `json:"o"`
}

func (deps ServerDeps) handleModelNoteRead(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	if !deps.noteModelFeatures.readEnabled {
		writeError(w, http.StatusServiceUnavailable, errors.New("model Notes access is disabled"))
		return
	}
	authority, ok := deps.authorizeNoteWorkerRequest(w, r)
	if !ok {
		return
	}
	var req modelNoteReadRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	req.NoteID = strings.TrimSpace(req.NoteID)
	if req.NoteID == "" || len(req.NoteID) > 512 {
		writeError(w, http.StatusBadRequest, errors.New("note_id is required"))
		return
	}
	if authority.Scope.Mode == domain.NoteAccessModeSelected && !authority.Scope.Contains(req.NoteID) {
		writeError(w, http.StatusNotFound, store.ErrNotFound)
		return
	}
	maxChars := req.MaxChars
	if maxChars <= 0 {
		maxChars = defaultModelNoteReadChars
	}
	if maxChars > maxModelNoteReadChars {
		maxChars = maxModelNoteReadChars
	}
	capability, ok := deps.notesCapability(w)
	if !ok {
		return
	}
	note, err := capability.GetNoteForUser(r.Context(), req.NoteID, authority.Run.UserID)
	if err != nil {
		writeNoteStoreError(w, err)
		return
	}
	if reference, selected := authority.Scope.Reference(note.NoteID); selected && reference.Revision != note.Revision {
		writeNoteStoreError(w, store.ErrNoteRevisionConflict)
		return
	}
	start := 0
	if req.Cursor != "" {
		cursor, err := decodeNoteReadCursor(req.Cursor)
		if err != nil || cursor.NoteID != note.NoteID || cursor.Revision != note.Revision || cursor.Offset < 0 || cursor.Offset > len(note.BodyMarkdown) ||
			(cursor.Offset < len(note.BodyMarkdown) && !utf8.RuneStart(note.BodyMarkdown[cursor.Offset])) {
			writeError(w, http.StatusBadRequest, errors.New("invalid or stale note cursor"))
			return
		}
		start = cursor.Offset
	}
	end := noteChunkEnd(note.BodyMarkdown, start, maxChars)
	body := note.BodyMarkdown[start:end]
	if err := capability.ConsumeNoteReadBudget(r.Context(), authority.Run.RunID, authority.Run.UserID, len(body)); err != nil {
		writeNoteStoreError(w, err)
		return
	}
	readToken := domain.NewID("nread")
	now := domain.Now()
	if err := capability.CreateNoteReadGrant(r.Context(), domain.NoteReadGrantRecord{
		TokenHash: domain.NoteBodySHA256(readToken), RunID: authority.Run.RunID,
		UserID: authority.Run.UserID, NoteID: note.NoteID, Revision: note.Revision,
		ExpiresAt: now.Add(noteReadGrantLifetime), CreatedAt: now,
	}); err != nil {
		writeNoteStoreError(w, err)
		return
	}
	nextCursor := ""
	if end < len(note.BodyMarkdown) {
		nextCursor = encodeNoteReadCursor(note.NoteID, note.Revision, end)
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"note_id": note.NoteID, "title": note.Title, "revision": note.Revision,
		"content_digest": note.ContentDigest, "body_markdown": body,
		"start_byte": start, "end_byte": end, "next_cursor": nextCursor,
		"has_more": end < len(note.BodyMarkdown), "read_token": readToken,
	})
}

func noteChunkEnd(body string, start int, maxChars int) int {
	end := start
	for count := 0; count < maxChars && end < len(body); count++ {
		_, size := utf8.DecodeRuneInString(body[end:])
		end += size
	}
	return end
}

func encodeNoteReadCursor(noteID string, revision int64, offset int) string {
	payload, _ := json.Marshal(noteReadCursor{NoteID: noteID, Revision: revision, Offset: offset})
	return base64.RawURLEncoding.EncodeToString(payload)
}

func decodeNoteReadCursor(value string) (noteReadCursor, error) {
	// note_id itself is bounded at 512 bytes, so leave enough room for the
	// encoded identity, revision, and byte offset while still rejecting abuse.
	if len(value) > 1024 {
		return noteReadCursor{}, errors.New("cursor is too long")
	}
	payload, err := base64.RawURLEncoding.DecodeString(value)
	if err != nil {
		return noteReadCursor{}, err
	}
	var cursor noteReadCursor
	if err := json.Unmarshal(payload, &cursor); err != nil {
		return noteReadCursor{}, err
	}
	return cursor, nil
}

type modelNoteProposalRequest struct {
	NoteID           string `json:"note_id"`
	ExpectedRevision int64  `json:"expected_revision"`
	BodyMarkdown     string `json:"body_markdown"`
	ReadToken        string `json:"read_token"`
	IdempotencyKey   string `json:"idempotency_key"`
}

func (deps ServerDeps) handleModelNoteAppendProposal(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	if !deps.noteModelFeatures.proposalEnabled {
		writeError(w, http.StatusServiceUnavailable, errors.New("model Note proposals are disabled"))
		return
	}
	if !deps.noteModelFeatures.requireExpectedRevision {
		writeError(w, http.StatusServiceUnavailable, errors.New("model Note proposals require strict Note revision enforcement"))
		return
	}
	authority, ok := deps.authorizeNoteWorkerRequest(w, r)
	if !ok {
		return
	}
	if !authority.Scope.AllowAppendProposal {
		writeError(w, http.StatusForbidden, errors.New("run is not authorized to propose Note appends"))
		return
	}
	var req modelNoteProposalRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	req.NoteID = strings.TrimSpace(req.NoteID)
	req.ReadToken = strings.TrimSpace(req.ReadToken)
	req.IdempotencyKey = strings.TrimSpace(req.IdempotencyKey)
	if authority.Scope.Mode == domain.NoteAccessModeSelected && !authority.Scope.Contains(req.NoteID) {
		writeError(w, http.StatusNotFound, store.ErrNotFound)
		return
	}
	if req.NoteID == "" || req.ExpectedRevision <= 0 || req.ReadToken == "" || len(req.ReadToken) > 256 ||
		req.IdempotencyKey == "" || len(req.IdempotencyKey) > maxNoteProposalIdempotency ||
		strings.TrimSpace(req.BodyMarkdown) == "" || len(req.BodyMarkdown) > maxModelNoteAppendBodyBytes {
		writeError(w, http.StatusBadRequest, errors.New("valid note_id, expected_revision, body_markdown, read_token, and idempotency_key are required"))
		return
	}
	capability, ok := deps.notesCapability(w)
	if !ok {
		return
	}
	now := domain.Now()
	requestDigest := domain.ComputeNoteContentDigest(req.NoteID+":"+strconv.FormatInt(req.ExpectedRevision, 10), req.BodyMarkdown)
	proposal, err := capability.CreateNoteAppendProposal(r.Context(), domain.CreateNoteAppendProposalInput{
		ProposalID: domain.NewID("nprop"), RunID: authority.Run.RunID,
		UserID: authority.Run.UserID, NoteID: req.NoteID,
		ExpectedRevision: req.ExpectedRevision, BodyMarkdown: req.BodyMarkdown,
		ReadTokenHash:  domain.NoteBodySHA256(req.ReadToken),
		IdempotencyKey: req.IdempotencyKey, RequestDigest: requestDigest,
		Now: now, ExpiresAt: now.Add(noteAppendProposalLifetime),
	})
	if err != nil {
		writeNoteStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, noteProposalMetadata(proposal, false))
}

func (deps ServerDeps) handleGetNoteAppendProposal(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	if deps.workerRequestAuth(r) != workerAuthAbsent {
		writeError(w, http.StatusUnauthorized, errors.New("browser authentication required"))
		return
	}
	if !deps.noteModelFeatures.proposalEnabled {
		writeError(w, http.StatusServiceUnavailable, errors.New("model Note proposals are disabled"))
		return
	}
	capability, ok := deps.notesCapability(w)
	if !ok {
		return
	}
	principal := deps.principalFromRequest(r, "")
	proposal, err := capability.GetNoteAppendProposalForUser(r.Context(), chi.URLParam(r, "proposal_id"), principal.UserID)
	if err != nil {
		writeNoteStoreError(w, err)
		return
	}
	response := noteProposalMetadata(proposal, true)
	if proposal.Status == domain.NoteAppendProposalStatusCommitted && proposal.OperationID != "" {
		operation, err := capability.GetNoteAppendOperationForUser(r.Context(), proposal.OperationID, principal.UserID)
		if err != nil {
			writeNoteStoreError(w, err)
			return
		}
		response["operation"] = operation
	}
	writeJSON(w, http.StatusOK, response)
}

type commitNoteProposalRequest struct {
	BodyMarkdown *string `json:"body_markdown"`
}

func (deps ServerDeps) handleCommitNoteAppendProposal(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	if deps.workerRequestAuth(r) != workerAuthAbsent {
		writeError(w, http.StatusUnauthorized, errors.New("browser authentication required"))
		return
	}
	if !deps.noteModelFeatures.proposalEnabled {
		writeError(w, http.StatusServiceUnavailable, errors.New("model Note proposals are disabled"))
		return
	}
	if !deps.noteModelFeatures.requireExpectedRevision {
		writeError(w, http.StatusServiceUnavailable, errors.New("Note proposal commits require strict Note revision enforcement"))
		return
	}
	var req commitNoteProposalRequest
	if r.Body != nil && r.ContentLength != 0 && !decodeJSON(w, r, &req) {
		return
	}
	if req.BodyMarkdown != nil && (strings.TrimSpace(*req.BodyMarkdown) == "" || len(*req.BodyMarkdown) > maxModelNoteAppendBodyBytes) {
		writeError(w, http.StatusBadRequest, errors.New("body_markdown must be non-empty and no larger than 32 KiB"))
		return
	}
	capability, ok := deps.notesCapability(w)
	if !ok {
		return
	}
	principal := deps.principalFromRequest(r, "")
	receipt, err := capability.CommitNoteAppendProposalForUser(r.Context(), domain.CommitNoteAppendProposalInput{
		ProposalID: chi.URLParam(r, "proposal_id"), OperationID: domain.NewID("nop"),
		UserID: principal.UserID, BodyMarkdown: req.BodyMarkdown, Now: domain.Now(),
	})
	if err != nil {
		writeNoteStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, receipt)
}

func (deps ServerDeps) handleUndoNoteAppendOperation(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	if deps.workerRequestAuth(r) != workerAuthAbsent {
		writeError(w, http.StatusUnauthorized, errors.New("browser authentication required"))
		return
	}
	if !deps.noteModelFeatures.proposalEnabled {
		writeError(w, http.StatusServiceUnavailable, errors.New("model Note proposals are disabled"))
		return
	}
	capability, ok := deps.notesCapability(w)
	if !ok {
		return
	}
	principal := deps.principalFromRequest(r, "")
	receipt, err := capability.UndoNoteAppendOperationForUser(r.Context(), domain.UndoNoteAppendOperationInput{
		OperationID: chi.URLParam(r, "operation_id"), UserID: principal.UserID, Now: domain.Now(),
	})
	if err != nil {
		writeNoteStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, receipt)
}

func noteProposalMetadata(proposal domain.NoteAppendProposalRecord, includeBrowserState bool) map[string]any {
	response := map[string]any{
		"proposal_id": proposal.ProposalID, "note_id": proposal.NoteID,
		"note_title": proposal.NoteTitle, "expected_revision": proposal.BaseRevision,
		"status": proposal.Status, "expires_at": proposal.ExpiresAt,
		"created_at": proposal.CreatedAt,
	}
	if includeBrowserState && proposal.OperationID != "" {
		response["operation_id"] = proposal.OperationID
	}
	if includeBrowserState && proposal.Status == domain.NoteAppendProposalStatusPending {
		response["body_markdown"] = proposal.BodyMarkdown
	}
	return response
}

func writeNoteStoreError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, store.ErrNoteRevisionConflict):
		writeJSON(w, http.StatusConflict, map[string]any{"error": "note changed since it was read", "code": "note_revision_conflict"})
	case errors.Is(err, store.ErrNoteReadTokenInvalid):
		writeJSON(w, http.StatusConflict, map[string]any{"error": "read the current note before proposing a change", "code": "note_read_required"})
	case errors.Is(err, store.ErrNoteProposalExpired):
		writeJSON(w, http.StatusConflict, map[string]any{"error": "note append proposal expired", "code": "note_proposal_expired"})
	case errors.Is(err, store.ErrNoteUndoConflict):
		writeJSON(w, http.StatusConflict, map[string]any{"error": "note changed after the append and can no longer be undone safely", "code": "note_undo_conflict"})
	case errors.Is(err, store.ErrNoteRetrievalBudget):
		writeJSON(w, http.StatusTooManyRequests, map[string]any{"error": "run Notes retrieval budget exhausted", "code": "note_retrieval_budget_exhausted"})
	case errors.Is(err, store.ErrNoteSearchTimeout):
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"error": "note search timed out", "code": "note_search_timeout"})
	case errors.Is(err, store.ErrNoteAppendIdempotencyConflict):
		writeJSON(w, http.StatusConflict, map[string]any{"error": "Idempotency-Key is already bound to a different Note append request", "code": "note_append_idempotency_conflict"})
	case errors.Is(err, store.ErrNoteAppendNotCommitted):
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error(), "code": noteAppendNotCommittedCode})
	case errors.Is(err, store.ErrNoteCreateIdempotencyConflict):
		writeJSON(w, http.StatusConflict, map[string]any{"error": "Idempotency-Key is already bound to a different Note create request", "code": "note_create_idempotency_conflict"})
	case errors.Is(err, store.ErrNoteCreateReplayDeleted):
		writeJSON(w, http.StatusGone, map[string]any{"error": "the idempotent Note create no longer has a live result", "code": noteCreateReplayDeletedCode})
	default:
		writeStoreError(w, err)
	}
}

func (deps ServerDeps) authorizeRunNoteSelection(ctx context.Context, userID string, selection domain.JSONMap) (domain.JSONMap, bool, error) {
	scope, present, valid := domain.ParseNoteAccessScope(selection)
	if !valid {
		return nil, true, errors.New("selection_context.note_access is invalid")
	}
	if !present {
		return selection, false, nil
	}
	capability, ok := deps.Store.(noteCapabilityStore)
	if !ok {
		return nil, true, errors.New("notes are not available on this deployment")
	}
	for index, reference := range scope.Notes {
		note, err := capability.GetNoteForUser(ctx, reference.NoteID, userID)
		if err != nil {
			return nil, true, err
		}
		if reference.Revision > 0 && reference.Revision != note.Revision {
			return nil, true, store.ErrNoteRevisionConflict
		}
		scope.Notes[index].Revision = note.Revision
	}
	return domain.CanonicalNoteAccessSelection(selection, scope), true, nil
}
