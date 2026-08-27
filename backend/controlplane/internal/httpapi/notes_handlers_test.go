package httpapi

import (
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
	return NewRouter(ServerDeps{
		Version: "test-version",
		Runs:    runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:   mem,
		noteModelFeatures: noteModelFeatureConfig{
			initialized: true, requireExpectedRevision: requireExpectedRevision,
		},
	})
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
