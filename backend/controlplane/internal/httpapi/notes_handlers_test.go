package httpapi

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

func newNotesTestRouter(t *testing.T) http.Handler {
	return newNotesTestRouterWithRevisionPolicy(t, true)
}

func newNotesTestRouterWithRevisionPolicy(t *testing.T, requireExpectedRevision bool) http.Handler {
	t.Helper()
	mem := store.NewMemoryStore()
	return newNotesTestRouterWithStore(t, mem, requireExpectedRevision)
}

func newNotesTestRouterWithStore(t *testing.T, backing runcontrol.Store, requireExpectedRevision bool) http.Handler {
	t.Helper()
	return NewRouter(ServerDeps{
		Version: "test-version",
		Runs:    runcontrol.NewService(backing, eventbus.NewMemoryBus()),
		Store:   backing,
		noteModelFeatures: noteModelFeatureConfig{
			initialized: true, requireExpectedRevision: requireExpectedRevision,
		},
	})
}

type historicalDirectReplayStore struct {
	*store.MemoryStore
	userID         string
	idempotencyKey string
	requestDigest  string
	receipt        domain.NoteDirectAppendOperationRecord
}

func (s *historicalDirectReplayStore) FindNoteDirectAppendReplayForUser(ctx context.Context, userID string, idempotencyKey string, requestDigest string) (domain.NoteDirectAppendOperationRecord, bool, error) {
	if userID != s.userID || idempotencyKey != s.idempotencyKey {
		return s.MemoryStore.FindNoteDirectAppendReplayForUser(ctx, userID, idempotencyKey, requestDigest)
	}
	if requestDigest != s.requestDigest {
		return domain.NoteDirectAppendOperationRecord{}, true, store.ErrNoteAppendIdempotencyConflict
	}
	return s.receipt, true, nil
}

func TestNotesRevisionRolloutSupportsLegacyThenStrictCAS(t *testing.T) {
	t.Parallel()
	compat := newNotesTestRouterWithRevisionPolicy(t, false)
	created := notesRequest(compat, http.MethodPost, "/v2/notes", "ada", `{"title":"compat","body_markdown":"draft"}`)
	var legacyNote domain.NoteRecord
	if err := json.Unmarshal(created.Body.Bytes(), &legacyNote); err != nil || created.Code != http.StatusCreated {
		t.Fatalf("compat create = %d note=%+v err=%v", created.Code, legacyNote, err)
	}
	legacyPatch := notesRequest(compat, http.MethodPatch, "/v2/notes/"+legacyNote.NoteID, "ada", `{"body_markdown":"legacy autosave"}`)
	var legacyUpdated domain.NoteRecord
	if err := json.Unmarshal(legacyPatch.Body.Bytes(), &legacyUpdated); err != nil || legacyPatch.Code != http.StatusOK {
		t.Fatalf("compat patch = %d note=%+v err=%v", legacyPatch.Code, legacyUpdated, err)
	}
	if legacyUpdated.BodyMarkdown != "legacy autosave" || legacyUpdated.Revision != legacyNote.Revision+1 {
		t.Fatalf("compat patch did not use revisioned store path: %+v", legacyUpdated)
	}

	strict := newNotesTestRouterWithRevisionPolicy(t, true)
	created = notesRequest(strict, http.MethodPost, "/v2/notes", "ada", `{"title":"strict","body_markdown":"draft"}`)
	var strictNote domain.NoteRecord
	_ = json.Unmarshal(created.Body.Bytes(), &strictNote)
	rejected := notesRequest(strict, http.MethodPatch, "/v2/notes/"+strictNote.NoteID, "ada", `{"body_markdown":"missing CAS"}`)
	if rejected.Code != http.StatusBadRequest || !strings.Contains(rejected.Body.String(), "expected_revision") {
		t.Fatalf("strict missing-revision patch = %d body=%s, want 400", rejected.Code, rejected.Body.String())
	}
	unchanged := notesRequest(strict, http.MethodGet, "/v2/notes/"+strictNote.NoteID, "ada", "")
	var current domain.NoteRecord
	_ = json.Unmarshal(unchanged.Body.Bytes(), &current)
	if current.BodyMarkdown != "draft" || current.Revision != strictNote.Revision {
		t.Fatalf("strict rejected patch changed Note: %+v", current)
	}
}

func notesRequest(router http.Handler, method, path, user, body string) *httptest.ResponseRecorder {
	var reader *strings.Reader
	if body == "" {
		reader = strings.NewReader("")
	} else {
		reader = strings.NewReader(body)
	}
	req := httptest.NewRequest(method, path, reader)
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	if user != "" {
		req.Header.Set("X-Ultra-User-Id", user)
	}
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	return rec
}

func performDirectNoteAppendRequest(router http.Handler, noteID, user, idempotencyKey, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, "/v2/notes/"+noteID+"/append", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Ultra-User-Id", user)
	if idempotencyKey != "" {
		req.Header.Set("Idempotency-Key", idempotencyKey)
	}
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	return rec
}

func performCreateNoteRequest(router http.Handler, user, idempotencyKey, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, "/v2/notes", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Ultra-User-Id", user)
	req.Header.Set("Idempotency-Key", idempotencyKey)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	return rec
}

func TestCreateNoteIsOptInIdempotentAcrossLostResponseAndHardDelete(t *testing.T) {
	t.Parallel()
	router := newNotesTestRouter(t)
	payload := `{"title":"Captured draft","body_markdown":"exact private selection","pinned":true,"editor_mode":"plaintext"}`
	first := performCreateNoteRequest(router, "ada", "stable-create-key", payload)
	var created domain.NoteRecord
	if err := json.Unmarshal(first.Body.Bytes(), &created); err != nil || first.Code != http.StatusCreated {
		t.Fatalf("first create = %d note=%+v body=%s err=%v", first.Code, created, first.Body.String(), err)
	}
	// Simulate a committed request whose response was lost: the same local
	// draft/key must replay the original Note rather than create a duplicate.
	replay := performCreateNoteRequest(router, "ada", "stable-create-key", payload)
	var replayed domain.NoteRecord
	_ = json.Unmarshal(replay.Body.Bytes(), &replayed)
	if replay.Code != http.StatusOK || replayed.NoteID != created.NoteID {
		t.Fatalf("create replay = %d note=%+v body=%s", replay.Code, replayed, replay.Body.String())
	}
	listed := notesRequest(router, http.MethodGet, "/v2/notes", "ada", "")
	var page struct {
		Notes      []domain.NoteListItem `json:"notes"`
		TotalCount int                   `json:"total_count"`
	}
	_ = json.Unmarshal(listed.Body.Bytes(), &page)
	if listed.Code != http.StatusOK || page.TotalCount != 1 || len(page.Notes) != 1 {
		t.Fatalf("list after replay = %d page=%+v", listed.Code, page)
	}
	conflict := performCreateNoteRequest(router, "ada", "stable-create-key", `{"title":"Different request"}`)
	if conflict.Code != http.StatusConflict || !strings.Contains(conflict.Body.String(), "note_create_idempotency_conflict") {
		t.Fatalf("create idempotency conflict = %d body=%s", conflict.Code, conflict.Body.String())
	}
	if blank := performCreateNoteRequest(router, "ada", "", payload); blank.Code != http.StatusBadRequest {
		t.Fatalf("blank create key = %d body=%s", blank.Code, blank.Body.String())
	}
	if deleted := notesRequest(router, http.MethodDelete, "/v2/notes/"+created.NoteID, "ada", ""); deleted.Code != http.StatusOK {
		t.Fatalf("delete Note = %d body=%s", deleted.Code, deleted.Body.String())
	}
	for name, deletedReplay := range map[string]*httptest.ResponseRecorder{
		"same request":      performCreateNoteRequest(router, "ada", "stable-create-key", payload),
		"different request": performCreateNoteRequest(router, "ada", "stable-create-key", `{"title":"Different request"}`),
	} {
		if deletedReplay.Code != http.StatusGone || !strings.Contains(deletedReplay.Body.String(), noteCreateReplayDeletedCode) {
			t.Fatalf("%s after deletion = %d body=%s, want terminal replay-deleted response", name, deletedReplay.Code, deletedReplay.Body.String())
		}
	}
}

func TestCreateReplayPrecedesTightenedValidationAndMissIsExplicitlyDefinitive(t *testing.T) {
	t.Parallel()
	mem := store.NewMemoryStore()
	router := newNotesTestRouterWithStore(t, mem, true)
	title := strings.Repeat("x", maxNoteTitleLength+1)
	record := domain.NoteRecord{
		NoteID: "note_historical_create", UserID: "ada", OrgID: "local-org", Title: title,
		EditorMode: domain.NoteEditorModeMarkdown, CreatedAt: domain.Now(),
	}
	digest := domain.ComputeNoteCreateRequestDigest(
		record.UserID, record.OrgID, record.Title, record.BodyMarkdown, record.Pinned, record.EditorMode,
	)
	if _, _, err := mem.CreateNoteForUserIdempotent(context.Background(), domain.CreateNoteIdempotentInput{
		Record: record, IdempotencyKey: "historical-create-key", RequestDigest: digest,
	}); err != nil {
		t.Fatalf("seed historical create receipt: %v", err)
	}
	payloadBytes, _ := json.Marshal(map[string]any{"title": title})
	payload := string(payloadBytes)

	replay := performCreateNoteRequest(router, "ada", "historical-create-key", payload)
	var replayed domain.NoteRecord
	_ = json.Unmarshal(replay.Body.Bytes(), &replayed)
	if replay.Code != http.StatusOK || replayed.NoteID != record.NoteID {
		t.Fatalf("historical exact replay = %d note=%+v body=%s", replay.Code, replayed, replay.Body.String())
	}
	conflictingPayload, _ := json.Marshal(map[string]any{"title": strings.Repeat("y", maxNoteTitleLength+1)})
	conflict := performCreateNoteRequest(router, "ada", "historical-create-key", string(conflictingPayload))
	if conflict.Code != http.StatusConflict || !strings.Contains(conflict.Body.String(), "note_create_idempotency_conflict") {
		t.Fatalf("historical conflicting replay = %d body=%s", conflict.Code, conflict.Body.String())
	}
	miss := performCreateNoteRequest(router, "ada", "new-invalid-create-key", payload)
	if miss.Code != http.StatusBadRequest || !strings.Contains(miss.Body.String(), noteCreateNotCommittedCode) {
		t.Fatalf("invalid create receipt miss = %d body=%s", miss.Code, miss.Body.String())
	}
	malformed := performCreateNoteRequest(router, "ada", "malformed-create-key", `{"title":`)
	if malformed.Code != http.StatusBadRequest || strings.Contains(malformed.Body.String(), noteCreateNotCommittedCode) {
		t.Fatalf("pre-lookup create decode failure = %d body=%s, want ambiguous generic 400", malformed.Code, malformed.Body.String())
	}
}

func TestDirectNoteAppendIsOwnerScopedIdempotentAndConditionallyUndoable(t *testing.T) {
	t.Parallel()
	router := newNotesTestRouter(t)
	created := notesRequest(router, http.MethodPost, "/v2/notes", "ada", `{"title":"Capture target","body_markdown":"baseline"}`)
	var note domain.NoteRecord
	if err := json.Unmarshal(created.Body.Bytes(), &note); err != nil || created.Code != http.StatusCreated {
		t.Fatalf("create = %d note=%+v err=%v", created.Code, note, err)
	}
	body := fmt.Sprintf(`{"body_markdown":"exact capture","expected_revision":%d}`, note.Revision)
	appended := performDirectNoteAppendRequest(router, note.NoteID, "ada", "capture-key", body)
	var receipt domain.NoteDirectAppendOperationRecord
	if err := json.Unmarshal(appended.Body.Bytes(), &receipt); err != nil || appended.Code != http.StatusCreated {
		t.Fatalf("append = %d receipt=%+v body=%s err=%v", appended.Code, receipt, appended.Body.String(), err)
	}
	if receipt.OperationID == "" || receipt.BeforeRevision != note.Revision || receipt.AfterRevision != note.Revision+1 {
		t.Fatalf("append receipt = %+v", receipt)
	}
	for _, secret := range []string{"exact capture", "capture-key", "request_digest", "append_sha256"} {
		if strings.Contains(appended.Body.String(), secret) {
			t.Fatalf("content-free receipt leaked %q: %s", secret, appended.Body.String())
		}
	}
	replay := performDirectNoteAppendRequest(router, note.NoteID, "ada", "capture-key", body)
	var replayReceipt domain.NoteDirectAppendOperationRecord
	_ = json.Unmarshal(replay.Body.Bytes(), &replayReceipt)
	if replay.Code != http.StatusOK || replayReceipt.OperationID != receipt.OperationID {
		t.Fatalf("append replay = %d receipt=%+v", replay.Code, replayReceipt)
	}
	conflict := performDirectNoteAppendRequest(router, note.NoteID, "ada", "capture-key",
		fmt.Sprintf(`{"body_markdown":"different capture","expected_revision":%d}`, note.Revision))
	if conflict.Code != http.StatusConflict || !strings.Contains(conflict.Body.String(), "note_append_idempotency_conflict") {
		t.Fatalf("idempotency conflict = %d body=%s", conflict.Code, conflict.Body.String())
	}
	stale := performDirectNoteAppendRequest(router, note.NoteID, "ada", "capture-key-stale", body)
	if stale.Code != http.StatusConflict || !strings.Contains(stale.Body.String(), "note_revision_conflict") {
		t.Fatalf("stale append = %d body=%s", stale.Code, stale.Body.String())
	}
	foreign := performDirectNoteAppendRequest(router, note.NoteID, "mallory", "foreign-key", body)
	if foreign.Code != http.StatusNotFound {
		t.Fatalf("foreign append = %d body=%s, want 404", foreign.Code, foreign.Body.String())
	}
	currentRec := notesRequest(router, http.MethodGet, "/v2/notes/"+note.NoteID, "ada", "")
	var current domain.NoteRecord
	_ = json.Unmarshal(currentRec.Body.Bytes(), &current)
	if current.BodyMarkdown != "baseline\n\nexact capture" || current.ContentUpdatedAt.Before(note.ContentUpdatedAt) {
		t.Fatalf("appended Note = %+v", current)
	}
	undoPath := "/v2/note-direct-append-operations/" + receipt.OperationID + "/undo"
	if foreignUndo := notesRequest(router, http.MethodPost, undoPath, "mallory", ""); foreignUndo.Code != http.StatusNotFound {
		t.Fatalf("foreign undo = %d, want 404", foreignUndo.Code)
	}
	undo := notesRequest(router, http.MethodPost, undoPath, "ada", "")
	var undone domain.NoteDirectAppendOperationRecord
	_ = json.Unmarshal(undo.Body.Bytes(), &undone)
	if undo.Code != http.StatusOK || undone.UndoRevision != receipt.AfterRevision+1 {
		t.Fatalf("undo = %d receipt=%+v body=%s", undo.Code, undone, undo.Body.String())
	}
	if undoReplay := notesRequest(router, http.MethodPost, undoPath, "ada", ""); undoReplay.Code != http.StatusOK {
		t.Fatalf("undo replay = %d body=%s", undoReplay.Code, undoReplay.Body.String())
	}
	currentRec = notesRequest(router, http.MethodGet, "/v2/notes/"+note.NoteID, "ada", "")
	_ = json.Unmarshal(currentRec.Body.Bytes(), &current)
	if current.BodyMarkdown != "baseline" {
		t.Fatalf("body after undo = %q", current.BodyMarkdown)
	}
	if deleted := notesRequest(router, http.MethodDelete, "/v2/notes/"+note.NoteID, "ada", ""); deleted.Code != http.StatusOK {
		t.Fatalf("delete appended Note = %d body=%s", deleted.Code, deleted.Body.String())
	}
	deletedReplay := performDirectNoteAppendRequest(router, note.NoteID, "ada", "capture-key", body)
	if deletedReplay.Code != http.StatusNotFound || !strings.Contains(deletedReplay.Body.String(), noteAppendTargetUnavailableCode) {
		t.Fatalf("append replay after deletion = %d body=%s", deletedReplay.Code, deletedReplay.Body.String())
	}
}

func TestDirectAppendReplayPrecedesTightenedValidationAndMissIsExplicitlyDefinitive(t *testing.T) {
	t.Parallel()
	const (
		noteID = "note_historical_append"
		userID = "ada"
		key    = "historical-append-key"
		body   = "   "
	)
	requestDigest := domain.ComputeNoteDirectAppendRequestDigest(userID, noteID, 1, body)
	backing := &historicalDirectReplayStore{
		MemoryStore:    store.NewMemoryStore(),
		userID:         userID,
		idempotencyKey: key,
		requestDigest:  requestDigest,
		receipt: domain.NoteDirectAppendOperationRecord{
			OperationID: "ndop_historical", NoteID: noteID, NoteTitle: "Historical",
			BeforeRevision: 1, AfterRevision: 2, AppendedBytes: 3,
			BeforeContentDigest: "before", AfterContentDigest: "after", CreatedAt: domain.Now(),
		},
	}
	router := newNotesTestRouterWithStore(t, backing, true)
	payload := `{"body_markdown":"   ","expected_revision":1}`

	replay := performDirectNoteAppendRequest(router, noteID, userID, key, payload)
	if replay.Code != http.StatusOK || !strings.Contains(replay.Body.String(), "ndop_historical") {
		t.Fatalf("historical direct replay = %d body=%s", replay.Code, replay.Body.String())
	}
	conflict := performDirectNoteAppendRequest(router, noteID, userID, key, `{"body_markdown":"  ","expected_revision":1}`)
	if conflict.Code != http.StatusConflict || !strings.Contains(conflict.Body.String(), "note_append_idempotency_conflict") {
		t.Fatalf("historical direct conflict = %d body=%s", conflict.Code, conflict.Body.String())
	}
	miss := performDirectNoteAppendRequest(router, noteID, userID, "new-invalid-append-key", payload)
	if miss.Code != http.StatusBadRequest || !strings.Contains(miss.Body.String(), noteAppendNotCommittedCode) {
		t.Fatalf("invalid direct receipt miss = %d body=%s", miss.Code, miss.Body.String())
	}
}

func TestDirectAppendCombinedSizeFailureIsExplicitlyDefinitive(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	backing := store.NewMemoryStore()
	note, err := backing.CreateNote(ctx, domain.NoteRecord{
		NoteID: "note_http_full", UserID: "ada", Title: "Full",
		BodyMarkdown: strings.Repeat("x", maxNoteBodyBytes),
		EditorMode:   domain.NoteEditorModeMarkdown, CreatedAt: domain.Now(),
	})
	if err != nil {
		t.Fatalf("seed full Note: %v", err)
	}
	router := newNotesTestRouterWithStore(t, backing, true)
	payload := fmt.Sprintf(`{"body_markdown":"capture","expected_revision":%d}`, note.Revision)
	rejected := performDirectNoteAppendRequest(router, note.NoteID, note.UserID, "full-note-capture", payload)
	if rejected.Code != http.StatusBadRequest || !strings.Contains(rejected.Body.String(), noteAppendNotCommittedCode) {
		t.Fatalf("combined-size append = %d body=%s, want definitive uncommitted code", rejected.Code, rejected.Body.String())
	}
	current, err := backing.GetNoteForUser(ctx, note.NoteID, note.UserID)
	if err != nil || current.Revision != note.Revision || current.BodyMarkdown != note.BodyMarkdown {
		t.Fatalf("combined-size rejection mutated Note: revision=%d body=%d err=%v", current.Revision, len(current.BodyMarkdown), err)
	}
	requestDigest := domain.ComputeNoteDirectAppendRequestDigest(note.UserID, note.NoteID, note.Revision, "capture")
	if _, found, err := backing.FindNoteDirectAppendReplayForUser(ctx, note.UserID, "full-note-capture", requestDigest); err != nil || found {
		t.Fatalf("combined-size rejection stored receipt found=%t err=%v", found, err)
	}
}

func TestDirectNoteAppendPreservesExactCaptureOnBlankNoteAndValidatesKey(t *testing.T) {
	t.Parallel()
	router := newNotesTestRouter(t)
	created := notesRequest(router, http.MethodPost, "/v2/notes", "ada", `{"title":"Blank"}`)
	var note domain.NoteRecord
	_ = json.Unmarshal(created.Body.Bytes(), &note)
	payload := fmt.Sprintf(`{"body_markdown":"  exact line\nsecond line  ","expected_revision":%d}`, note.Revision)
	if missing := performDirectNoteAppendRequest(router, note.NoteID, "ada", "", payload); missing.Code != http.StatusBadRequest {
		t.Fatalf("missing key = %d body=%s", missing.Code, missing.Body.String())
	}
	duplicateRequest := httptest.NewRequest(http.MethodPost, "/v2/notes/"+note.NoteID+"/append", strings.NewReader(payload))
	duplicateRequest.Header.Set("Content-Type", "application/json")
	duplicateRequest.Header.Set("X-Ultra-User-Id", "ada")
	duplicateRequest.Header.Add("Idempotency-Key", "blank-capture")
	duplicateRequest.Header.Add("Idempotency-Key", "ambiguous-retry-key")
	duplicate := httptest.NewRecorder()
	router.ServeHTTP(duplicate, duplicateRequest)
	if duplicate.Code != http.StatusBadRequest || strings.Contains(duplicate.Body.String(), noteAppendNotCommittedCode) {
		t.Fatalf("duplicate key headers = %d body=%s", duplicate.Code, duplicate.Body.String())
	}
	if appended := performDirectNoteAppendRequest(router, note.NoteID, "ada", "blank-capture", payload); appended.Code != http.StatusCreated {
		t.Fatalf("blank append = %d body=%s", appended.Code, appended.Body.String())
	}
	got := notesRequest(router, http.MethodGet, "/v2/notes/"+note.NoteID, "ada", "")
	var current domain.NoteRecord
	_ = json.Unmarshal(got.Body.Bytes(), &current)
	if current.BodyMarkdown != "  exact line\nsecond line  " {
		t.Fatalf("blank Note capture = %q, want exact bytes", current.BodyMarkdown)
	}
}

// The whole surface is owner-scoped: CRUD round-trips for the owner, and a
// stranger sees 404 for every operation — a foreign note id must behave
// exactly like a missing one (no existence oracle).
func TestNotesCrudIsOwnerScoped(t *testing.T) {
	t.Parallel()
	router := newNotesTestRouter(t)

	created := notesRequest(router, http.MethodPost, "/v2/notes", "ada",
		`{"title":"Field protocol","body_markdown":"Transect spacing 40 m."}`)
	if created.Code != http.StatusCreated {
		t.Fatalf("create = %d body=%s", created.Code, created.Body.String())
	}
	var note domain.NoteRecord
	if err := json.Unmarshal(created.Body.Bytes(), &note); err != nil {
		t.Fatalf("decode create: %v", err)
	}
	if note.NoteID == "" || note.Title != "Field protocol" || note.Pinned {
		t.Fatalf("created note = %+v", note)
	}
	if note.EditorMode != domain.NoteEditorModeMarkdown {
		t.Fatalf("new notes default to the markdown surface, got %q", note.EditorMode)
	}

	if rec := notesRequest(router, http.MethodGet, "/v2/notes/"+note.NoteID, "ada", ""); rec.Code != http.StatusOK {
		t.Fatalf("owner get = %d", rec.Code)
	}
	for _, method := range []string{http.MethodGet, http.MethodDelete} {
		if rec := notesRequest(router, method, "/v2/notes/"+note.NoteID, "mallory", ""); rec.Code != http.StatusNotFound {
			t.Fatalf("stranger %s = %d, want 404", method, rec.Code)
		}
	}
	if rec := notesRequest(router, http.MethodPatch, "/v2/notes/"+note.NoteID, "mallory", fmt.Sprintf(`{"title":"mine now","expected_revision":%d}`, note.Revision)); rec.Code != http.StatusNotFound {
		t.Fatalf("stranger patch = %d, want 404", rec.Code)
	}

	updated := notesRequest(router, http.MethodPatch, "/v2/notes/"+note.NoteID, "ada",
		fmt.Sprintf(`{"body_markdown":"Transect spacing 40 m. Flag GSD < 1.2 cm.","pinned":true,"editor_mode":"plaintext","expected_revision":%d}`, note.Revision))
	if updated.Code != http.StatusOK {
		t.Fatalf("owner patch = %d body=%s", updated.Code, updated.Body.String())
	}
	var patched domain.NoteRecord
	if err := json.Unmarshal(updated.Body.Bytes(), &patched); err != nil {
		t.Fatalf("decode patch: %v", err)
	}
	if !patched.Pinned || patched.Title != "Field protocol" || !strings.Contains(patched.BodyMarkdown, "GSD") {
		t.Fatalf("patched note = %+v (partial update must not clobber untouched fields)", patched)
	}
	if patched.EditorMode != domain.NoteEditorModePlaintext {
		t.Fatalf("editor_mode = %q, want the flip to plaintext persisted", patched.EditorMode)
	}
	stale := notesRequest(router, http.MethodPatch, "/v2/notes/"+note.NoteID, "ada", fmt.Sprintf(`{"title":"stale","expected_revision":%d}`, note.Revision))
	if stale.Code != http.StatusConflict || !strings.Contains(stale.Body.String(), "note_revision_conflict") {
		t.Fatalf("stale patch = %d body=%s", stale.Code, stale.Body.String())
	}

	// The mode is sticky: a later partial update without editor_mode keeps it.
	retitled := notesRequest(router, http.MethodPatch, "/v2/notes/"+note.NoteID, "ada", fmt.Sprintf(`{"title":"Field protocol v2","expected_revision":%d}`, patched.Revision))
	var afterRetitle domain.NoteRecord
	if err := json.Unmarshal(retitled.Body.Bytes(), &afterRetitle); err != nil {
		t.Fatalf("decode retitle: %v", err)
	}
	if afterRetitle.EditorMode != domain.NoteEditorModePlaintext {
		t.Fatalf("editor_mode after unrelated patch = %q, want sticky plaintext", afterRetitle.EditorMode)
	}

	if rec := notesRequest(router, http.MethodDelete, "/v2/notes/"+note.NoteID, "ada", ""); rec.Code != http.StatusOK {
		t.Fatalf("owner delete = %d", rec.Code)
	}
	// Hard delete: gone means gone.
	if rec := notesRequest(router, http.MethodGet, "/v2/notes/"+note.NoteID, "ada", ""); rec.Code != http.StatusNotFound {
		t.Fatalf("get after delete = %d, want 404", rec.Code)
	}
}

func TestNotesListPinsFirstSearchesAndScopes(t *testing.T) {
	t.Parallel()
	router := newNotesTestRouter(t)

	seed := []struct {
		user, title, body string
		pinned            bool
	}{
		{"ada", "Reading list", "spatial transcriptomics papers", false},
		{"ada", "Field protocol", "prairie dog transects", true},
		{"ada", "Lab meeting", "decisions about NGFF", false},
		{"grace", "Grace's note", "not ada's business", true},
	}
	for _, item := range seed {
		payload, _ := json.Marshal(map[string]any{
			"title": item.title, "body_markdown": item.body, "pinned": item.pinned,
		})
		if rec := notesRequest(router, http.MethodPost, "/v2/notes", item.user, string(payload)); rec.Code != http.StatusCreated {
			t.Fatalf("seed %q = %d", item.title, rec.Code)
		}
	}

	list := notesRequest(router, http.MethodGet, "/v2/notes", "ada", "")
	if list.Code != http.StatusOK {
		t.Fatalf("list = %d", list.Code)
	}
	var page struct {
		Notes      []domain.NoteListItem `json:"notes"`
		TotalCount int                   `json:"total_count"`
	}
	if err := json.Unmarshal(list.Body.Bytes(), &page); err != nil {
		t.Fatalf("decode list: %v", err)
	}
	if page.TotalCount != 3 || len(page.Notes) != 3 {
		t.Fatalf("ada sees %d/%d notes, want exactly her 3", len(page.Notes), page.TotalCount)
	}
	if !page.Notes[0].Pinned || page.Notes[0].Title != "Field protocol" {
		t.Fatalf("first row = %+v, want the pinned note first", page.Notes[0])
	}
	for _, item := range page.Notes {
		if item.Title == "Grace's note" {
			t.Fatalf("cross-user leak: %+v", item)
		}
	}
	recent := notesRequest(router, http.MethodGet, "/v2/notes?sort=recent", "ada", "")
	var recentPage struct {
		Notes []domain.NoteListItem `json:"notes"`
	}
	if err := json.Unmarshal(recent.Body.Bytes(), &recentPage); err != nil || recent.Code != http.StatusOK {
		t.Fatalf("decode recent list: status=%d err=%v", recent.Code, err)
	}
	if len(recentPage.Notes) != 3 || recentPage.Notes[0].Title != "Lab meeting" {
		t.Fatalf("recent list = %+v, want newest content regardless of pin", recentPage.Notes)
	}
	if invalid := notesRequest(router, http.MethodGet, "/v2/notes?sort=popular", "ada", ""); invalid.Code != http.StatusBadRequest {
		t.Fatalf("invalid list sort = %d body=%s", invalid.Code, invalid.Body.String())
	}

	search := notesRequest(router, http.MethodGet, "/v2/notes?query=prairie", "ada", "")
	var searched struct {
		Notes []domain.NoteListItem `json:"notes"`
	}
	if err := json.Unmarshal(search.Body.Bytes(), &searched); err != nil {
		t.Fatalf("decode search: %v", err)
	}
	if len(searched.Notes) != 1 || searched.Notes[0].Title != "Field protocol" {
		t.Fatalf("search hit = %+v, want the body match", searched.Notes)
	}
}

func TestNotesRejectsOversizedWrites(t *testing.T) {
	t.Parallel()
	router := newNotesTestRouter(t)
	huge := strings.Repeat("x", maxNoteBodyBytes+1)
	rec := notesRequest(router, http.MethodPost, "/v2/notes", "ada", `{"body_markdown":"`+huge+`"}`)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("oversized create = %d, want 400", rec.Code)
	}
}

func TestNotesRejectsUnknownEditorMode(t *testing.T) {
	t.Parallel()
	router := newNotesTestRouter(t)
	if rec := notesRequest(router, http.MethodPost, "/v2/notes", "ada", `{"editor_mode":"wysiwyg"}`); rec.Code != http.StatusBadRequest {
		t.Fatalf("unknown editor_mode create = %d, want 400", rec.Code)
	}
	created := notesRequest(router, http.MethodPost, "/v2/notes", "ada", `{"title":"n"}`)
	var note domain.NoteRecord
	if err := json.Unmarshal(created.Body.Bytes(), &note); err != nil {
		t.Fatalf("decode create: %v", err)
	}
	if rec := notesRequest(router, http.MethodPatch, "/v2/notes/"+note.NoteID, "ada", `{"editor_mode":"rich"}`); rec.Code != http.StatusBadRequest {
		t.Fatalf("unknown editor_mode patch = %d, want 400", rec.Code)
	}
}
