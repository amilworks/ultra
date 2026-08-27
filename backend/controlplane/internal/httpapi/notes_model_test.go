package httpapi

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

type modelNotesHarness struct {
	store  *store.MemoryStore
	runs   *runcontrol.Service
	router http.Handler
	run    domain.RunRecord
	lease  domain.RunLeaseRecord
	note   domain.NoteRecord
}

func newModelNotesHarness(t *testing.T, mode domain.NoteAccessMode) modelNotesHarness {
	return newModelNotesHarnessWithProposal(t, mode, true)
}

func newModelNotesHarnessWithProposal(t *testing.T, mode domain.NoteAccessMode, allowAppendProposal bool) modelNotesHarness {
	t.Helper()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	runs := runcontrol.NewService(mem, eventbus.NewMemoryBus())
	thread, err := runs.CreateThread(ctx, runcontrol.CreateThreadRequest{UserID: "ada", Title: "Notes test"})
	if err != nil {
		t.Fatalf("create thread: %v", err)
	}
	now := domain.Now()
	note, err := mem.CreateNote(ctx, domain.NoteRecord{
		NoteID: "note_primary", UserID: "ada", OrgID: "local-org",
		Title: "Calibration log", BodyMarkdown: "baseline", EditorMode: domain.NoteEditorModeMarkdown,
		CreatedAt: now,
	})
	if err != nil {
		t.Fatalf("create note: %v", err)
	}
	scope := domain.NoteAccessScope{
		Mode: mode, Notes: []domain.NoteReference{{NoteID: note.NoteID, Revision: note.Revision}},
		AllowAppendProposal: allowAppendProposal,
	}
	run, err := runs.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID, UserID: "ada", Goal: "use my notes",
		SelectionContext: domain.CanonicalNoteAccessSelection(nil, scope),
		Metadata:         metadataWithPrincipal(nil, requestPrincipal{UserID: "ada", OrgID: "local-org", Role: "researcher"}),
	})
	if err != nil {
		t.Fatalf("create run: %v", err)
	}
	lease, err := mem.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID: run.RunID, WorkerID: "worker-notes", TTL: time.Minute, Now: now,
	})
	if err != nil {
		t.Fatalf("acquire lease: %v", err)
	}
	router := NewRouter(ServerDeps{
		Version: "test", Runs: runs, Store: mem, WorkerToken: "worker-secret",
		noteModelFeatures: noteModelFeatureConfig{
			initialized: true, readEnabled: true, proposalEnabled: true, requireExpectedRevision: true,
		},
	})
	return modelNotesHarness{store: mem, runs: runs, router: router, run: run, lease: lease, note: note}
}

func (h modelNotesHarness) workerRequest(method string, path string, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Ultra-Worker-Token", "worker-secret")
	req.Header.Set("X-Ultra-Run-Id", h.run.RunID)
	req.Header.Set("X-Ultra-Worker-Id", h.lease.WorkerID)
	req.Header.Set("X-Ultra-Run-Lease-Token", h.lease.LeaseToken)
	rec := httptest.NewRecorder()
	h.router.ServeHTTP(rec, req)
	return rec
}

func TestModelNotesReadProposalCommitAndConditionalUndo(t *testing.T) {
	t.Parallel()
	h := newModelNotesHarness(t, domain.NoteAccessModeSearch)
	base := "/v2/runs/" + h.run.RunID

	search := h.workerRequest(http.MethodPost, base+"/note-search", `{"query":"calibration","limit":10}`)
	if search.Code != http.StatusOK {
		t.Fatalf("search = %d body=%s", search.Code, search.Body.String())
	}
	var searchPage struct {
		Notes   []domain.NoteSearchHit `json:"notes"`
		HasMore bool                   `json:"has_more"`
	}
	if err := json.Unmarshal(search.Body.Bytes(), &searchPage); err != nil {
		t.Fatalf("decode search: %v", err)
	}
	if len(searchPage.Notes) != 1 || searchPage.Notes[0].NoteID != h.note.NoteID || searchPage.HasMore {
		t.Fatalf("search page = %+v", searchPage)
	}

	read := h.workerRequest(http.MethodPost, base+"/note-read", `{"note_id":"note_primary","max_chars":16000}`)
	if read.Code != http.StatusOK {
		t.Fatalf("read = %d body=%s", read.Code, read.Body.String())
	}
	var readResult struct {
		Revision  int64  `json:"revision"`
		Body      string `json:"body_markdown"`
		ReadToken string `json:"read_token"`
		HasMore   bool   `json:"has_more"`
	}
	if err := json.Unmarshal(read.Body.Bytes(), &readResult); err != nil {
		t.Fatalf("decode read: %v", err)
	}
	if readResult.Body != "baseline" || readResult.ReadToken == "" || readResult.HasMore {
		t.Fatalf("read result = %+v", readResult)
	}

	proposalBody := fmt.Sprintf(`{"note_id":"note_primary","expected_revision":%d,"body_markdown":"new fact","read_token":%q,"idempotency_key":"tool-call-1"}`, readResult.Revision, readResult.ReadToken)
	created := h.workerRequest(http.MethodPost, base+"/note-append-proposals", proposalBody)
	if created.Code != http.StatusCreated {
		t.Fatalf("proposal = %d body=%s", created.Code, created.Body.String())
	}
	if strings.Contains(created.Body.String(), "new fact") {
		t.Fatal("worker proposal response leaked exact proposed text")
	}
	var proposal struct {
		ProposalID string `json:"proposal_id"`
	}
	if err := json.Unmarshal(created.Body.Bytes(), &proposal); err != nil || proposal.ProposalID == "" {
		t.Fatalf("decode proposal: %v %#v", err, proposal)
	}

	replay := h.workerRequest(http.MethodPost, base+"/note-append-proposals", proposalBody)
	var replayProposal struct {
		ProposalID string `json:"proposal_id"`
	}
	_ = json.Unmarshal(replay.Body.Bytes(), &replayProposal)
	if replay.Code != http.StatusCreated || replayProposal.ProposalID != proposal.ProposalID {
		t.Fatalf("idempotent replay = %d %+v", replay.Code, replayProposal)
	}
	conflictBody := strings.Replace(proposalBody, "new fact", "different fact", 1)
	if rec := h.workerRequest(http.MethodPost, base+"/note-append-proposals", conflictBody); rec.Code != http.StatusConflict ||
		!strings.Contains(rec.Body.String(), "note_append_idempotency_conflict") {
		t.Fatalf("same idempotency key with different body = %d body=%s, want typed 409", rec.Code, rec.Body.String())
	}

	get := notesRequest(h.router, http.MethodGet, "/v2/note-append-proposals/"+proposal.ProposalID, "ada", "")
	if get.Code != http.StatusOK || !strings.Contains(get.Body.String(), "new fact") {
		t.Fatalf("browser proposal = %d body=%s", get.Code, get.Body.String())
	}
	if rec := notesRequest(h.router, http.MethodGet, "/v2/note-append-proposals/"+proposal.ProposalID, "mallory", ""); rec.Code != http.StatusNotFound {
		t.Fatalf("foreign proposal = %d, want 404", rec.Code)
	}

	commit := notesRequest(h.router, http.MethodPost, "/v2/note-append-proposals/"+proposal.ProposalID+"/commit", "ada", `{}`)
	if commit.Code != http.StatusOK {
		t.Fatalf("commit = %d body=%s", commit.Code, commit.Body.String())
	}
	var receipt domain.NoteAppendOperationRecord
	if err := json.Unmarshal(commit.Body.Bytes(), &receipt); err != nil {
		t.Fatalf("decode receipt: %v", err)
	}
	if receipt.OperationID == "" || receipt.BeforeRevision != 1 || receipt.AfterRevision != 2 {
		t.Fatalf("receipt = %+v", receipt)
	}
	note, err := h.store.GetNoteForUser(context.Background(), h.note.NoteID, "ada")
	if err != nil || note.BodyMarkdown != "baseline\n\nnew fact" || note.Revision != 2 {
		t.Fatalf("committed note = %+v err=%v", note, err)
	}
	committedWorkerReplay := h.workerRequest(http.MethodPost, base+"/note-append-proposals", proposalBody)
	if committedWorkerReplay.Code != http.StatusCreated ||
		strings.Contains(committedWorkerReplay.Body.String(), "new fact") ||
		strings.Contains(committedWorkerReplay.Body.String(), "operation_id") {
		t.Fatalf("committed worker replay was not metadata-redacted: %d body=%s", committedWorkerReplay.Code, committedWorkerReplay.Body.String())
	}

	committedGet := notesRequest(h.router, http.MethodGet, "/v2/note-append-proposals/"+proposal.ProposalID, "ada", "")
	if committedGet.Code != http.StatusOK || strings.Contains(committedGet.Body.String(), "new fact") || !strings.Contains(committedGet.Body.String(), `"operation"`) {
		t.Fatalf("committed proposal reload = %d body=%s", committedGet.Code, committedGet.Body.String())
	}
	replayedCommit := notesRequest(h.router, http.MethodPost, "/v2/note-append-proposals/"+proposal.ProposalID+"/commit", "ada", `{}`)
	var replayReceipt domain.NoteAppendOperationRecord
	_ = json.Unmarshal(replayedCommit.Body.Bytes(), &replayReceipt)
	if replayedCommit.Code != http.StatusOK || replayReceipt.OperationID != receipt.OperationID {
		t.Fatalf("commit replay = %d %+v", replayedCommit.Code, replayReceipt)
	}

	undo := notesRequest(h.router, http.MethodPost, "/v2/note-append-operations/"+receipt.OperationID+"/undo", "ada", "")
	if undo.Code != http.StatusOK {
		t.Fatalf("undo = %d body=%s", undo.Code, undo.Body.String())
	}
	note, _ = h.store.GetNoteForUser(context.Background(), h.note.NoteID, "ada")
	if note.BodyMarkdown != "baseline" || note.Revision != 3 {
		t.Fatalf("undone note = %+v", note)
	}
	undoReplay := notesRequest(h.router, http.MethodPost, "/v2/note-append-operations/"+receipt.OperationID+"/undo", "ada", "")
	if undoReplay.Code != http.StatusOK {
		t.Fatalf("undo replay = %d", undoReplay.Code)
	}
}

func TestModelNotesSearchLimitTwentyUsesLookahead(t *testing.T) {
	t.Parallel()
	h := newModelNotesHarness(t, domain.NoteAccessModeSearch)
	baseTime := domain.Now().Add(-time.Hour)
	for index := 0; index < 21; index++ {
		_, err := h.store.CreateNote(context.Background(), domain.NoteRecord{
			NoteID: fmt.Sprintf("note_match_%02d", index), UserID: "ada", Title: fmt.Sprintf("match %02d", index),
			EditorMode: domain.NoteEditorModeMarkdown, CreatedAt: baseTime.Add(time.Duration(index) * time.Millisecond),
		})
		if err != nil {
			t.Fatalf("seed note: %v", err)
		}
	}
	rec := h.workerRequest(http.MethodPost, "/v2/runs/"+h.run.RunID+"/note-search", `{"query":"match","limit":20}`)
	var page struct {
		Notes   []domain.NoteSearchHit `json:"notes"`
		HasMore bool                   `json:"has_more"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &page); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if rec.Code != http.StatusOK || len(page.Notes) != 20 || !page.HasMore {
		t.Fatalf("search = %d count=%d has_more=%t", rec.Code, len(page.Notes), page.HasMore)
	}
}

func TestModelNotesRecentSearchIsCursorPagedAndBoundToRequest(t *testing.T) {
	t.Parallel()
	h := newModelNotesHarness(t, domain.NoteAccessModeSearch)
	tieTime := domain.Now().Add(-2 * time.Hour)
	for _, noteID := range []string{"note_recent_b", "note_recent_a"} {
		if _, err := h.store.CreateNote(context.Background(), domain.NoteRecord{
			NoteID: noteID, UserID: "ada", Title: "Recent " + noteID,
			BodyMarkdown: "shared recent body", Pinned: noteID == "note_recent_b",
			EditorMode: domain.NoteEditorModeMarkdown, CreatedAt: tieTime,
		}); err != nil {
			t.Fatalf("seed %s: %v", noteID, err)
		}
	}
	base := "/v2/runs/" + h.run.RunID + "/note-search"
	first := h.workerRequest(http.MethodPost, base, `{"query":"","sort":"recent","limit":1}`)
	var firstPage struct {
		Notes      []domain.NoteSearchHit `json:"notes"`
		HasMore    bool                   `json:"has_more"`
		NextCursor string                 `json:"next_cursor"`
	}
	if err := json.Unmarshal(first.Body.Bytes(), &firstPage); err != nil || first.Code != http.StatusOK {
		t.Fatalf("first recent page = %d page=%+v body=%s err=%v", first.Code, firstPage, first.Body.String(), err)
	}
	if len(firstPage.Notes) != 1 || firstPage.Notes[0].NoteID != h.note.NoteID || !firstPage.HasMore || firstPage.NextCursor == "" {
		t.Fatalf("first recent page = %+v", firstPage)
	}
	// A content mutation of an already-returned Note and a new Note both sort
	// ahead of the old boundary. Offset pagination would now duplicate or skip;
	// the snapshot-bound keyset excludes both from the in-flight search.
	returned, err := h.store.GetNoteForUser(context.Background(), firstPage.Notes[0].NoteID, "ada")
	if err != nil {
		t.Fatalf("get returned Note: %v", err)
	}
	changed := "mutated after the first page"
	if _, err := h.store.UpdateNoteForUser(context.Background(), returned.NoteID, "ada", domain.NoteUpdateInput{
		ExpectedRevision: returned.Revision, BodyMarkdown: &changed,
	}); err != nil {
		t.Fatalf("mutate returned Note: %v", err)
	}
	if _, err := h.store.CreateNote(context.Background(), domain.NoteRecord{
		NoteID: "note_recent_new", UserID: "ada", Title: "Created after page one",
		BodyMarkdown: "newer than the search snapshot", EditorMode: domain.NoteEditorModeMarkdown,
		CreatedAt: domain.Now(),
	}); err != nil {
		t.Fatalf("create concurrent Note: %v", err)
	}
	secondBody := fmt.Sprintf(`{"query":"","sort":"recent","limit":1,"cursor":%q}`, firstPage.NextCursor)
	second := h.workerRequest(http.MethodPost, base, secondBody)
	var secondPage struct {
		Notes      []domain.NoteSearchHit `json:"notes"`
		HasMore    bool                   `json:"has_more"`
		NextCursor string                 `json:"next_cursor"`
	}
	_ = json.Unmarshal(second.Body.Bytes(), &secondPage)
	if second.Code != http.StatusOK || len(secondPage.Notes) != 1 || secondPage.Notes[0].NoteID != "note_recent_a" ||
		!secondPage.HasMore || secondPage.NextCursor == "" {
		t.Fatalf("second recent page = %d %+v body=%s", second.Code, secondPage, second.Body.String())
	}
	third := h.workerRequest(http.MethodPost, base,
		fmt.Sprintf(`{"query":"","sort":"recent","limit":1,"cursor":%q}`, secondPage.NextCursor))
	var thirdPage struct {
		Notes   []domain.NoteSearchHit `json:"notes"`
		HasMore bool                   `json:"has_more"`
	}
	_ = json.Unmarshal(third.Body.Bytes(), &thirdPage)
	if third.Code != http.StatusOK || len(thirdPage.Notes) != 1 || thirdPage.Notes[0].NoteID != "note_recent_b" || thirdPage.HasMore {
		t.Fatalf("third recent page = %d %+v body=%s", third.Code, thirdPage, third.Body.String())
	}
	for name, body := range map[string]string{
		"malformed":  `{"query":"","sort":"recent","cursor":"not-base64"}`,
		"query swap": fmt.Sprintf(`{"query":"shared","sort":"recent","cursor":%q}`, firstPage.NextCursor),
		"sort swap":  fmt.Sprintf(`{"query":"shared","sort":"relevance","cursor":%q}`, firstPage.NextCursor),
	} {
		if rec := h.workerRequest(http.MethodPost, base, body); rec.Code != http.StatusBadRequest {
			t.Fatalf("%s cursor = %d body=%s, want 400", name, rec.Code, rec.Body.String())
		}
	}
	if rec := h.workerRequest(http.MethodPost, base, `{"query":"","sort":"relevance"}`); rec.Code != http.StatusBadRequest {
		t.Fatalf("blank relevance query = %d body=%s, want 400", rec.Code, rec.Body.String())
	}
	if rec := h.workerRequest(http.MethodPost, base, `{"query":"","sort":"popular"}`); rec.Code != http.StatusBadRequest {
		t.Fatalf("invalid sort = %d body=%s, want 400", rec.Code, rec.Body.String())
	}
}

func TestModelNotesRelevanceCursorCarriesRankAndRejectsSnapshotDrift(t *testing.T) {
	t.Parallel()
	h := newModelNotesHarness(t, domain.NoteAccessModeSearch)
	baseTime := domain.Now().Add(-3 * time.Hour)
	for _, seed := range []domain.NoteRecord{
		{NoteID: "note_rank_exact", UserID: "ada", Title: "calibration", BodyMarkdown: "exact", EditorMode: domain.NoteEditorModeMarkdown, CreatedAt: baseTime},
		{NoteID: "note_rank_body", UserID: "ada", Title: "Other", BodyMarkdown: "calibration in body", EditorMode: domain.NoteEditorModeMarkdown, CreatedAt: baseTime.Add(time.Hour)},
	} {
		if _, err := h.store.CreateNote(context.Background(), seed); err != nil {
			t.Fatalf("seed %s: %v", seed.NoteID, err)
		}
	}
	path := "/v2/runs/" + h.run.RunID + "/note-search"
	requestPage := func(cursor string) (int, struct {
		Notes      []domain.NoteSearchHit `json:"notes"`
		HasMore    bool                   `json:"has_more"`
		NextCursor string                 `json:"next_cursor"`
	}) {
		t.Helper()
		body := `{"query":"calibration","sort":"relevance","limit":1}`
		if cursor != "" {
			body = fmt.Sprintf(`{"query":"calibration","sort":"relevance","limit":1,"cursor":%q}`, cursor)
		}
		rec := h.workerRequest(http.MethodPost, path, body)
		var page struct {
			Notes      []domain.NoteSearchHit `json:"notes"`
			HasMore    bool                   `json:"has_more"`
			NextCursor string                 `json:"next_cursor"`
		}
		_ = json.Unmarshal(rec.Body.Bytes(), &page)
		return rec.Code, page
	}
	code, first := requestPage("")
	if code != http.StatusOK || len(first.Notes) != 1 || first.Notes[0].NoteID != "note_rank_exact" || !first.HasMore {
		t.Fatalf("first relevance page = %d %+v", code, first)
	}
	exact, _ := h.store.GetNoteForUser(context.Background(), "note_rank_exact", "ada")
	mutated := "changed after search start"
	if _, err := h.store.UpdateNoteForUser(context.Background(), exact.NoteID, "ada", domain.NoteUpdateInput{
		ExpectedRevision: exact.Revision, BodyMarkdown: &mutated,
	}); err != nil {
		t.Fatalf("mutate returned exact-title Note: %v", err)
	}
	if _, err := h.store.CreateNote(context.Background(), domain.NoteRecord{
		NoteID: "note_rank_new_exact", UserID: "ada", Title: "calibration",
		BodyMarkdown: "created after snapshot", EditorMode: domain.NoteEditorModeMarkdown, CreatedAt: domain.Now(),
	}); err != nil {
		t.Fatalf("create concurrent exact-title Note: %v", err)
	}
	code, second := requestPage(first.NextCursor)
	if code != http.StatusOK || len(second.Notes) != 1 || second.Notes[0].NoteID != h.note.NoteID || !second.HasMore {
		t.Fatalf("second relevance page = %d %+v", code, second)
	}
	code, third := requestPage(second.NextCursor)
	if code != http.StatusOK || len(third.Notes) != 1 || third.Notes[0].NoteID != "note_rank_body" || third.HasMore {
		t.Fatalf("third relevance page = %d %+v", code, third)
	}
}

func TestModelNoteSearchRelevanceAndRecentUseDifferentStableOrders(t *testing.T) {
	t.Parallel()
	h := newModelNotesHarness(t, domain.NoteAccessModeSearch)
	baseTime := domain.Now().Add(-4 * time.Hour)
	seeds := []domain.NoteRecord{
		{NoteID: "note_order_exact", UserID: "ada", Title: "needle", BodyMarkdown: "old", EditorMode: domain.NoteEditorModeMarkdown, CreatedAt: baseTime},
		{NoteID: "note_order_title", UserID: "ada", Title: "needle details", BodyMarkdown: "middle", EditorMode: domain.NoteEditorModeMarkdown, CreatedAt: baseTime.Add(time.Hour)},
		{NoteID: "note_order_body", UserID: "ada", Title: "Pinned old label", BodyMarkdown: "new needle body", Pinned: true, EditorMode: domain.NoteEditorModeMarkdown, CreatedAt: baseTime.Add(2 * time.Hour)},
	}
	for _, seed := range seeds {
		if _, err := h.store.CreateNote(context.Background(), seed); err != nil {
			t.Fatalf("seed %s: %v", seed.NoteID, err)
		}
	}
	path := "/v2/runs/" + h.run.RunID + "/note-search"
	for _, test := range []struct {
		body string
		want string
	}{
		{body: `{"query":"needle","sort":"relevance","limit":3}`, want: "note_order_exact"},
		{body: `{"query":"needle","sort":"recent","limit":3}`, want: "note_order_body"},
	} {
		rec := h.workerRequest(http.MethodPost, path, test.body)
		var page struct {
			Notes []domain.NoteSearchHit `json:"notes"`
		}
		_ = json.Unmarshal(rec.Body.Bytes(), &page)
		if rec.Code != http.StatusOK || len(page.Notes) != 3 || page.Notes[0].NoteID != test.want {
			t.Fatalf("search %s = %d %+v", test.body, rec.Code, page.Notes)
		}
	}
}

func TestModelNotesSelectedScopeAndCursorAreFailClosed(t *testing.T) {
	t.Parallel()
	h := newModelNotesHarness(t, domain.NoteAccessModeSelected)
	base := "/v2/runs/" + h.run.RunID
	if rec := h.workerRequest(http.MethodPost, base+"/note-search", `{"query":"baseline"}`); rec.Code != http.StatusForbidden {
		t.Fatalf("selected-only search = %d, want 403", rec.Code)
	}
	other, _ := h.store.CreateNote(context.Background(), domain.NoteRecord{NoteID: "note_other", UserID: "ada", Title: "Other", BodyMarkdown: "🙂x", EditorMode: domain.NoteEditorModeMarkdown, CreatedAt: domain.Now()})
	if rec := h.workerRequest(http.MethodPost, base+"/note-read", fmt.Sprintf(`{"note_id":%q}`, other.NoteID)); rec.Code != http.StatusNotFound {
		t.Fatalf("unselected read = %d, want 404", rec.Code)
	}
	first := h.workerRequest(http.MethodPost, base+"/note-read", `{"note_id":"note_primary","max_chars":1}`)
	var chunk struct {
		NextCursor string `json:"next_cursor"`
	}
	_ = json.Unmarshal(first.Body.Bytes(), &chunk)
	if first.Code != http.StatusOK || chunk.NextCursor == "" {
		t.Fatalf("first chunk = %d body=%s", first.Code, first.Body.String())
	}
	if rec := h.workerRequest(http.MethodPost, base+"/note-read", fmt.Sprintf(`{"note_id":%q,"cursor":%q}`, other.NoteID, chunk.NextCursor)); rec.Code != http.StatusNotFound {
		t.Fatalf("cross-note cursor = %d, want scope 404", rec.Code)
	}
}

func TestModelNoteReadCursorIsBoundToNoteIdentity(t *testing.T) {
	t.Parallel()
	h := newModelNotesHarness(t, domain.NoteAccessModeSearch)
	other, _ := h.store.CreateNote(context.Background(), domain.NoteRecord{
		NoteID: "note_cursor_other", UserID: "ada", Title: "Other",
		BodyMarkdown: "other body", EditorMode: domain.NoteEditorModeMarkdown, CreatedAt: domain.Now(),
	})
	base := "/v2/runs/" + h.run.RunID
	first := h.workerRequest(http.MethodPost, base+"/note-read", `{"note_id":"note_primary","max_chars":1}`)
	var chunk struct {
		NextCursor string `json:"next_cursor"`
	}
	_ = json.Unmarshal(first.Body.Bytes(), &chunk)
	if first.Code != http.StatusOK || chunk.NextCursor == "" {
		t.Fatalf("first chunk = %d body=%s", first.Code, first.Body.String())
	}
	rec := h.workerRequest(http.MethodPost, base+"/note-read", fmt.Sprintf(`{"note_id":%q,"cursor":%q}`, other.NoteID, chunk.NextCursor))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("cross-note cursor = %d body=%s, want 400", rec.Code, rec.Body.String())
	}
}

func TestModelNotesRejectUnknownProtectedProfileAndMissingLease(t *testing.T) {
	t.Parallel()
	h := newModelNotesHarness(t, domain.NoteAccessModeSearch)
	if _, _, err := h.store.ClearRunLease(context.Background(), h.run.RunID); err != nil {
		t.Fatalf("clear lease: %v", err)
	}
	if rec := h.workerRequest(http.MethodPost, "/v2/runs/"+h.run.RunID+"/note-read", `{"note_id":"note_primary"}`); rec.Code != http.StatusUnauthorized {
		t.Fatalf("read without active lease = %d, want 401", rec.Code)
	}

	ctx := context.Background()
	mem := store.NewMemoryStore()
	runs := runcontrol.NewService(mem, eventbus.NewMemoryBus())
	thread, _ := runs.CreateThread(ctx, runcontrol.CreateThreadRequest{UserID: "ada", Title: "protected"})
	note, _ := mem.CreateNote(ctx, domain.NoteRecord{NoteID: "note_protected", UserID: "ada", Title: "private", BodyMarkdown: "body", EditorMode: domain.NoteEditorModeMarkdown, CreatedAt: domain.Now()})
	metadata := metadataWithPrincipal(nil, requestPrincipal{UserID: "ada", OrgID: "local-org", Role: "researcher"})
	metadata[domain.EvaluationProfileMetadataKey] = "unknown_profile"
	metadata["selection_context"] = domain.CanonicalNoteAccessSelection(nil, domain.NoteAccessScope{Mode: domain.NoteAccessModeSearch, Notes: []domain.NoteReference{{NoteID: note.NoteID, Revision: note.Revision}}})
	run, err := mem.CreateRun(ctx, domain.CreateRunInput{ThreadID: thread.ThreadID, UserID: "ada", Goal: "protected", Metadata: metadata})
	if err != nil {
		t.Fatalf("create malformed stored run: %v", err)
	}
	lease, err := mem.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{RunID: run.RunID, WorkerID: "worker-notes", TTL: time.Minute, Now: domain.Now()})
	if err != nil {
		t.Fatalf("lease malformed run: %v", err)
	}
	protected := modelNotesHarness{store: mem, run: run, lease: lease, router: NewRouter(ServerDeps{
		Runs: runs, Store: mem, WorkerToken: "worker-secret",
		noteModelFeatures: noteModelFeatureConfig{
			initialized: true, readEnabled: true, proposalEnabled: true, requireExpectedRevision: true,
		},
	})}
	if rec := protected.workerRequest(http.MethodPost, "/v2/runs/"+run.RunID+"/note-read", fmt.Sprintf(`{"note_id":%q}`, note.NoteID)); rec.Code != http.StatusForbidden {
		t.Fatalf("unknown protected profile read = %d body=%s, want 403", rec.Code, rec.Body.String())
	}
}

func TestModelNotesWorkerAuthorityRejectsForgedLeaseIdentity(t *testing.T) {
	t.Parallel()
	h := newModelNotesHarness(t, domain.NoteAccessModeSearch)
	path := "/v2/runs/" + h.run.RunID + "/note-read"
	cases := []struct {
		name, token, runID, workerID, leaseToken string
	}{
		{"invalid token", "wrong", h.run.RunID, h.lease.WorkerID, h.lease.LeaseToken},
		{"path header mismatch", "worker-secret", "run_other", h.lease.WorkerID, h.lease.LeaseToken},
		{"wrong worker", "worker-secret", h.run.RunID, "worker-other", h.lease.LeaseToken},
		{"wrong lease token", "worker-secret", h.run.RunID, h.lease.WorkerID, "lease-other"},
	}
	for _, test := range cases {
		test := test
		t.Run(test.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{"note_id":"note_primary"}`))
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("X-Ultra-Worker-Token", test.token)
			req.Header.Set("X-Ultra-Run-Id", test.runID)
			req.Header.Set("X-Ultra-Worker-Id", test.workerID)
			req.Header.Set("X-Ultra-Run-Lease-Token", test.leaseToken)
			rec := httptest.NewRecorder()
			h.router.ServeHTTP(rec, req)
			if rec.Code != http.StatusUnauthorized {
				t.Fatalf("status = %d body=%s, want 401", rec.Code, rec.Body.String())
			}
		})
	}
}

func TestModelNotesNeverRevealForeignOwnerRows(t *testing.T) {
	t.Parallel()
	h := newModelNotesHarness(t, domain.NoteAccessModeSearch)
	foreign, _ := h.store.CreateNote(context.Background(), domain.NoteRecord{
		NoteID: "note_mallory_secret", UserID: "mallory", Title: "xenosecret",
		BodyMarkdown: "xenosecret body", EditorMode: domain.NoteEditorModeMarkdown, CreatedAt: domain.Now(),
	})
	base := "/v2/runs/" + h.run.RunID
	if rec := h.workerRequest(http.MethodPost, base+"/note-read", fmt.Sprintf(`{"note_id":%q}`, foreign.NoteID)); rec.Code != http.StatusNotFound {
		t.Fatalf("foreign read = %d body=%s, want 404", rec.Code, rec.Body.String())
	}
	search := h.workerRequest(http.MethodPost, base+"/note-search", `{"query":"xenosecret"}`)
	var page struct {
		Notes []domain.NoteSearchHit `json:"notes"`
	}
	_ = json.Unmarshal(search.Body.Bytes(), &page)
	if search.Code != http.StatusOK || len(page.Notes) != 0 {
		t.Fatalf("foreign search = %d %+v", search.Code, page.Notes)
	}
}

func TestSelectedScopeRejectsProposalDespiteForgedReadGrant(t *testing.T) {
	t.Parallel()
	h := newModelNotesHarness(t, domain.NoteAccessModeSelected)
	now := domain.Now()
	other, _ := h.store.CreateNote(context.Background(), domain.NoteRecord{
		NoteID: "note_unselected_proposal", UserID: "ada", Title: "Other",
		BodyMarkdown: "other", EditorMode: domain.NoteEditorModeMarkdown, CreatedAt: now,
	})
	readToken := "forged-direct-grant"
	if err := h.store.CreateNoteReadGrant(context.Background(), domain.NoteReadGrantRecord{
		TokenHash: domain.NoteBodySHA256(readToken), RunID: h.run.RunID, UserID: "ada",
		NoteID: other.NoteID, Revision: other.Revision, CreatedAt: now, ExpiresAt: now.Add(time.Minute),
	}); err != nil {
		t.Fatalf("seed forged grant: %v", err)
	}
	body := fmt.Sprintf(`{"note_id":%q,"expected_revision":%d,"body_markdown":"append","read_token":%q,"idempotency_key":"forged-scope"}`, other.NoteID, other.Revision, readToken)
	rec := h.workerRequest(http.MethodPost, "/v2/runs/"+h.run.RunID+"/note-append-proposals", body)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("unselected proposal = %d body=%s, want 404", rec.Code, rec.Body.String())
	}
}

func TestProposalFailsAfterConcurrentBrowserEdit(t *testing.T) {
	t.Parallel()
	h := newModelNotesHarness(t, domain.NoteAccessModeSearch)
	base := "/v2/runs/" + h.run.RunID
	read := h.workerRequest(http.MethodPost, base+"/note-read", `{"note_id":"note_primary"}`)
	var result struct {
		Revision  int64  `json:"revision"`
		ReadToken string `json:"read_token"`
	}
	_ = json.Unmarshal(read.Body.Bytes(), &result)
	patch := notesRequest(h.router, http.MethodPatch, "/v2/notes/"+h.note.NoteID, "ada", fmt.Sprintf(`{"body_markdown":"browser edit","expected_revision":%d}`, result.Revision))
	if patch.Code != http.StatusOK {
		t.Fatalf("browser patch = %d body=%s", patch.Code, patch.Body.String())
	}
	proposal := h.workerRequest(http.MethodPost, base+"/note-append-proposals", fmt.Sprintf(`{"note_id":"note_primary","expected_revision":%d,"body_markdown":"stale append","read_token":%q,"idempotency_key":"stale-proposal"}`, result.Revision, result.ReadToken))
	if proposal.Code != http.StatusConflict || !strings.Contains(proposal.Body.String(), "note_revision_conflict") {
		t.Fatalf("stale proposal = %d body=%s", proposal.Code, proposal.Body.String())
	}
}

func TestReviewedAppendIsIdempotentAndUndoRefusesLaterEdits(t *testing.T) {
	t.Parallel()
	h := newModelNotesHarness(t, domain.NoteAccessModeSearch)
	base := "/v2/runs/" + h.run.RunID
	read := h.workerRequest(http.MethodPost, base+"/note-read", `{"note_id":"note_primary"}`)
	var result struct {
		Revision  int64  `json:"revision"`
		ReadToken string `json:"read_token"`
	}
	if err := json.Unmarshal(read.Body.Bytes(), &result); err != nil || read.Code != http.StatusOK {
		t.Fatalf("read = %d result=%+v err=%v", read.Code, result, err)
	}
	created := h.workerRequest(http.MethodPost, base+"/note-append-proposals", fmt.Sprintf(
		`{"note_id":"note_primary","expected_revision":%d,"body_markdown":"draft fact","read_token":%q,"idempotency_key":"reviewed-proposal"}`,
		result.Revision, result.ReadToken,
	))
	var proposal struct {
		ProposalID string `json:"proposal_id"`
	}
	if err := json.Unmarshal(created.Body.Bytes(), &proposal); err != nil || created.Code != http.StatusCreated {
		t.Fatalf("proposal = %d result=%+v err=%v", created.Code, proposal, err)
	}
	commitPath := "/v2/note-append-proposals/" + proposal.ProposalID + "/commit"
	commit := notesRequest(h.router, http.MethodPost, commitPath, "ada", `{"body_markdown":"reviewed fact"}`)
	var receipt domain.NoteAppendOperationRecord
	if err := json.Unmarshal(commit.Body.Bytes(), &receipt); err != nil || commit.Code != http.StatusOK {
		t.Fatalf("reviewed commit = %d receipt=%+v err=%v", commit.Code, receipt, err)
	}
	note, err := h.store.GetNoteForUser(context.Background(), h.note.NoteID, "ada")
	if err != nil || note.BodyMarkdown != "baseline\n\nreviewed fact" {
		t.Fatalf("reviewed Note = %+v err=%v", note, err)
	}
	replay := notesRequest(h.router, http.MethodPost, commitPath, "ada", `{"body_markdown":"reviewed fact"}`)
	var replayReceipt domain.NoteAppendOperationRecord
	_ = json.Unmarshal(replay.Body.Bytes(), &replayReceipt)
	if replay.Code != http.StatusOK || replayReceipt.OperationID != receipt.OperationID {
		t.Fatalf("reviewed commit replay = %d receipt=%+v", replay.Code, replayReceipt)
	}
	if conflict := notesRequest(h.router, http.MethodPost, commitPath, "ada", `{"body_markdown":"different review"}`); conflict.Code != http.StatusConflict {
		t.Fatalf("different reviewed replay = %d body=%s, want 409", conflict.Code, conflict.Body.String())
	}

	patch := notesRequest(h.router, http.MethodPatch, "/v2/notes/"+h.note.NoteID, "ada", fmt.Sprintf(
		`{"body_markdown":"later browser edit","expected_revision":%d}`, receipt.AfterRevision,
	))
	if patch.Code != http.StatusOK {
		t.Fatalf("later browser edit = %d body=%s", patch.Code, patch.Body.String())
	}
	undo := notesRequest(h.router, http.MethodPost, "/v2/note-append-operations/"+receipt.OperationID+"/undo", "ada", "")
	if undo.Code != http.StatusConflict || !strings.Contains(undo.Body.String(), "note_undo_conflict") {
		t.Fatalf("unsafe undo = %d body=%s, want conditional 409", undo.Code, undo.Body.String())
	}
}

func TestModelNotesRetrievalBudgetReturnsTypedRateLimit(t *testing.T) {
	t.Parallel()
	h := newModelNotesHarness(t, domain.NoteAccessModeSearch)
	for index := 0; index < 32; index++ {
		if err := h.store.ConsumeNoteSearchBudget(context.Background(), h.run.RunID, "ada"); err != nil {
			t.Fatalf("seed search budget %d: %v", index, err)
		}
	}
	rec := h.workerRequest(http.MethodPost, "/v2/runs/"+h.run.RunID+"/note-search", `{"query":"baseline"}`)
	if rec.Code != http.StatusTooManyRequests || !strings.Contains(rec.Body.String(), "note_retrieval_budget_exhausted") {
		t.Fatalf("exhausted search budget = %d body=%s, want typed 429", rec.Code, rec.Body.String())
	}
}

func TestModelNotesFeatureSwitchesAreAuthoritative(t *testing.T) {
	t.Parallel()
	h := newModelNotesHarness(t, domain.NoteAccessModeSearch)
	h.router = NewRouter(ServerDeps{
		Runs: h.runs, Store: h.store, WorkerToken: "worker-secret",
		noteModelFeatures: noteModelFeatureConfig{initialized: true, readEnabled: false, proposalEnabled: false, requireExpectedRevision: true},
	})
	rec := h.workerRequest(http.MethodPost, "/v2/runs/"+h.run.RunID+"/note-search", `{"query":"baseline"}`)
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("disabled read = %d, want 503", rec.Code)
	}
	h.router = NewRouter(ServerDeps{
		Runs: h.runs, Store: h.store, WorkerToken: "worker-secret",
		noteModelFeatures: noteModelFeatureConfig{initialized: true, readEnabled: true, proposalEnabled: false, requireExpectedRevision: true},
	})
	rec = h.workerRequest(http.MethodPost, "/v2/runs/"+h.run.RunID+"/note-append-proposals", `{}`)
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("disabled proposal = %d, want 503", rec.Code)
	}
	h.router = NewRouter(ServerDeps{
		Runs: h.runs, Store: h.store, WorkerToken: "worker-secret",
		noteModelFeatures: noteModelFeatureConfig{initialized: true, readEnabled: true, proposalEnabled: true, requireExpectedRevision: false},
	})
	if rec := h.workerRequest(http.MethodPost, "/v2/runs/"+h.run.RunID+"/note-search", `{"query":"baseline"}`); rec.Code != http.StatusOK {
		t.Fatalf("read during CAS compatibility stage = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	if rec := h.workerRequest(http.MethodPost, "/v2/runs/"+h.run.RunID+"/note-append-proposals", `{}`); rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("proposal without strict CAS = %d body=%s, want 503", rec.Code, rec.Body.String())
	}
	if rec := notesRequest(h.router, http.MethodPost, "/v2/note-append-proposals/missing/commit", "ada", `{}`); rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("browser commit without strict CAS = %d body=%s, want 503", rec.Code, rec.Body.String())
	}
}

func TestModelNoteProposalRequiresPerRunProposalAuthority(t *testing.T) {
	t.Parallel()
	h := newModelNotesHarnessWithProposal(t, domain.NoteAccessModeSearch, false)

	rec := h.workerRequest(http.MethodPost, "/v2/runs/"+h.run.RunID+"/note-append-proposals", `{}`)
	if rec.Code != http.StatusForbidden {
		t.Fatalf("proposal without per-run authority = %d body=%s, want 403", rec.Code, rec.Body.String())
	}
}

func TestNotesFeatureFlagsDefaultOffAndRejectTypos(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name, raw    string
		exists, want bool
	}{
		{name: "absent", exists: false, want: false},
		{name: "empty", raw: "", exists: true, want: false},
		{name: "true", raw: "true", exists: true, want: true},
		{name: "one", raw: "1", exists: true, want: true},
		{name: "enabled", raw: "enabled", exists: true, want: true},
		{name: "false", raw: "false", exists: true, want: false},
		{name: "typo", raw: "treu", exists: true, want: false},
	} {
		test := test
		t.Run(test.name, func(t *testing.T) {
			if got := featureSettingEnabled(test.raw, test.exists); got != test.want {
				t.Fatalf("featureSettingEnabled(%q, %t) = %t, want %t", test.raw, test.exists, got, test.want)
			}
		})
	}
}

func TestCreateRunAuthorizesAndCanonicalizesNoteScope(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	runs := runcontrol.NewService(mem, eventbus.NewMemoryBus())
	thread, _ := runs.CreateThread(ctx, runcontrol.CreateThreadRequest{UserID: "ada", Title: "scope"})
	note, _ := mem.CreateNote(ctx, domain.NoteRecord{NoteID: "note_scope", UserID: "ada", Title: "scope", EditorMode: domain.NoteEditorModeMarkdown, CreatedAt: domain.Now()})
	foreign, _ := mem.CreateNote(ctx, domain.NoteRecord{NoteID: "note_foreign", UserID: "mallory", Title: "foreign", EditorMode: domain.NoteEditorModeMarkdown, CreatedAt: domain.Now()})
	router := NewRouter(ServerDeps{
		Runs: runs, Store: mem,
		noteModelFeatures: noteModelFeatureConfig{
			initialized: true, readEnabled: true, proposalEnabled: true, requireExpectedRevision: true,
		},
	})
	body := fmt.Sprintf(`{"goal":"use this note","selection_context":{"note_access":{"mode":"selected","notes":[{"note_id":%q}],"allow_append_proposal":true}}}`, note.NoteID)
	rec := notesRequest(router, http.MethodPost, "/v2/threads/"+thread.ThreadID+"/runs", "ada", body)
	if rec.Code != http.StatusOK {
		t.Fatalf("create scoped run = %d body=%s", rec.Code, rec.Body.String())
	}
	var run domain.RunRecord
	if err := json.Unmarshal(rec.Body.Bytes(), &run); err != nil {
		t.Fatalf("decode run: %v", err)
	}
	scope, ok := domain.NoteAccessScopeFromRun(run)
	if !ok || len(scope.Notes) != 1 || scope.Notes[0].NoteID != note.NoteID || scope.Notes[0].Revision != note.Revision || !scope.AllowAppendProposal {
		t.Fatalf("stored scope = %+v ok=%t", scope, ok)
	}
	if enabled, ok := run.Metadata[domain.ModelNotesProposalsEnabledMetadataKey].(bool); !ok || !enabled {
		t.Fatalf("proposal availability metadata = %#v, want true bool", run.Metadata[domain.ModelNotesProposalsEnabledMetadataKey])
	}
	foreignBody := fmt.Sprintf(`{"goal":"steal","selection_context":{"note_access":{"mode":"selected","notes":[{"note_id":%q}]}}}`, foreign.NoteID)
	if rec := notesRequest(router, http.MethodPost, "/v2/threads/"+thread.ThreadID+"/runs", "ada", foreignBody); rec.Code != http.StatusNotFound {
		t.Fatalf("foreign selected Note = %d, want 404", rec.Code)
	}
	staleBody := fmt.Sprintf(`{"goal":"stale","selection_context":{"note_access":{"mode":"selected","notes":[{"note_id":%q,"revision":%d}]}}}`, note.NoteID, note.Revision+1)
	if rec := notesRequest(router, http.MethodPost, "/v2/threads/"+thread.ThreadID+"/runs", "ada", staleBody); rec.Code != http.StatusConflict {
		t.Fatalf("stale selected Note = %d body=%s, want 409", rec.Code, rec.Body.String())
	}
	spoof := `{"goal":"spoof","metadata":{"selection_context":{"note_access":{"mode":"search"}},"model_notes_proposals_enabled":true}}`
	spoofRec := notesRequest(router, http.MethodPost, "/v2/threads/"+thread.ThreadID+"/runs", "ada", spoof)
	if spoofRec.Code != http.StatusOK {
		t.Fatalf("run with ignored metadata spoof = %d body=%s", spoofRec.Code, spoofRec.Body.String())
	}
	var spoofRun domain.RunRecord
	_ = json.Unmarshal(spoofRec.Body.Bytes(), &spoofRun)
	if _, ok := domain.NoteAccessScopeFromRun(spoofRun); ok {
		t.Fatalf("free-form metadata minted Notes scope: %+v", spoofRun.Metadata)
	}
	if _, exists := spoofRun.Metadata[domain.ModelNotesProposalsEnabledMetadataKey]; exists {
		t.Fatalf("free-form metadata minted proposal availability: %+v", spoofRun.Metadata)
	}
}

func TestCreateRunNoteScopeReplaySurvivesFeatureDisableEditAndDelete(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	runs := runcontrol.NewService(mem, eventbus.NewMemoryBus())
	thread, err := runs.CreateThread(ctx, runcontrol.CreateThreadRequest{UserID: "ada", Title: "scope replay"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	note, err := mem.CreateNote(ctx, domain.NoteRecord{
		NoteID: "note_replay", UserID: "ada", Title: "Replay", BodyMarkdown: "one",
		EditorMode: domain.NoteEditorModeMarkdown, CreatedAt: domain.Now(),
	})
	if err != nil {
		t.Fatalf("CreateNote: %v", err)
	}
	enabled := NewRouter(ServerDeps{
		Runs: runs, Store: mem,
		noteModelFeatures: noteModelFeatureConfig{
			initialized: true, readEnabled: true, proposalEnabled: true, requireExpectedRevision: true,
		},
	})
	body := fmt.Sprintf(`{"goal":"use my note","messages":[{"role":"user","content":"Please use this note."}],"idempotency_key":"note-replay","selection_context":{"note_access":{"mode":"selected","notes":[{"note_id":%q}],"allow_append_proposal":true}}}`, note.NoteID)
	firstRec := notesRequest(enabled, http.MethodPost, "/v2/threads/"+thread.ThreadID+"/runs", "ada", body)
	if firstRec.Code != http.StatusOK {
		t.Fatalf("first create = %d body=%s", firstRec.Code, firstRec.Body.String())
	}
	var first domain.RunRecord
	if err := json.Unmarshal(firstRec.Body.Bytes(), &first); err != nil {
		t.Fatalf("decode first run: %v", err)
	}
	if first.Metadata[domain.ModelNotesProposalsEnabledMetadataKey] != true {
		t.Fatalf("stored proposal flag = %#v, want true", first.Metadata[domain.ModelNotesProposalsEnabledMetadataKey])
	}

	updatedTitle := "Replay changed"
	updatedBody := "two"
	if _, err := mem.UpdateNoteForUser(ctx, note.NoteID, "ada", domain.NoteUpdateInput{
		Title: &updatedTitle, BodyMarkdown: &updatedBody, ExpectedRevision: note.Revision,
	}); err != nil {
		t.Fatalf("UpdateNoteForUser: %v", err)
	}
	disabled := NewRouter(ServerDeps{
		Runs: runs, Store: mem,
		noteModelFeatures: noteModelFeatureConfig{initialized: true, requireExpectedRevision: true},
	})
	assertReplay := func(stage string) {
		t.Helper()
		rec := notesRequest(disabled, http.MethodPost, "/v2/threads/"+thread.ThreadID+"/runs", "ada", body)
		if rec.Code != http.StatusOK {
			t.Fatalf("%s replay = %d body=%s", stage, rec.Code, rec.Body.String())
		}
		var replay domain.RunRecord
		if err := json.Unmarshal(rec.Body.Bytes(), &replay); err != nil {
			t.Fatalf("decode %s replay: %v", stage, err)
		}
		if replay.RunID != first.RunID {
			t.Fatalf("%s replay run = %s, want %s", stage, replay.RunID, first.RunID)
		}
	}
	assertReplay("edited Note")
	if err := mem.DeleteNoteForUser(ctx, note.NoteID, "ada"); err != nil {
		t.Fatalf("DeleteNoteForUser: %v", err)
	}
	assertReplay("deleted Note")

	conflicts := map[string]string{
		"different id":         fmt.Sprintf(`{"goal":"use my note","idempotency_key":"note-replay","selection_context":{"note_access":{"mode":"selected","notes":[{"note_id":"note_other"}],"allow_append_proposal":true}}}`),
		"different mode":       `{"goal":"use my note","idempotency_key":"note-replay","selection_context":{"note_access":{"mode":"search","notes":[],"allow_append_proposal":true}}}`,
		"explicit revision":    fmt.Sprintf(`{"goal":"use my note","idempotency_key":"note-replay","selection_context":{"note_access":{"mode":"selected","notes":[{"note_id":%q,"revision":%d}],"allow_append_proposal":true}}}`, note.NoteID, note.Revision+1),
		"different allow flag": fmt.Sprintf(`{"goal":"use my note","idempotency_key":"note-replay","selection_context":{"note_access":{"mode":"selected","notes":[{"note_id":%q}],"allow_append_proposal":false}}}`, note.NoteID),
	}
	for name, conflictBody := range conflicts {
		rec := notesRequest(disabled, http.MethodPost, "/v2/threads/"+thread.ThreadID+"/runs", "ada", conflictBody)
		if rec.Code != http.StatusConflict {
			t.Errorf("%s = %d body=%s, want 409", name, rec.Code, rec.Body.String())
		}
	}
	newKeyBody := strings.Replace(body, `"note-replay"`, `"note-new-key"`, 1)
	if rec := notesRequest(disabled, http.MethodPost, "/v2/threads/"+thread.ThreadID+"/runs", "ada", newKeyBody); rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("new Notes run while disabled = %d body=%s, want 503", rec.Code, rec.Body.String())
	}
}

func TestCreateRunRejectsNotesCombinedWithUnsupportedSelections(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	runs := runcontrol.NewService(mem, eventbus.NewMemoryBus())
	thread, err := runs.CreateThread(ctx, runcontrol.CreateThreadRequest{UserID: "ada", Title: "exclusive Notes context"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	router := NewRouter(ServerDeps{
		Runs: runs, Store: mem,
		noteModelFeatures: noteModelFeatureConfig{initialized: true, readEnabled: true},
	})
	base := `"goal":"search my notes","selection_context":{"note_access":{"mode":"search"}}`
	for name, field := range map[string]string{
		"file_ids":             `"file_ids":["file_one"]`,
		"resource selections":  `"resource_uris":["resource://one"]`,
		"resource descriptors": `"resource_descriptors":[{"uri":"resource://one"}]`,
		"dataset_uris":         `"dataset_uris":["dataset://one"]`,
		"knowledge_context":    `"knowledge_context":{"topic":"one"}`,
		"workflow_hint":        `"workflow_hint":{"kind":"one"}`,
		"selected_tool_names":  `"selected_tool_names":["tool_one"]`,
	} {
		body := "{" + base + "," + field + "}"
		rec := notesRequest(router, http.MethodPost, "/v2/threads/"+thread.ThreadID+"/runs", "ada", body)
		if rec.Code != http.StatusBadRequest {
			t.Errorf("%s = %d body=%s, want 400", name, rec.Code, rec.Body.String())
		}
		if !strings.Contains(rec.Body.String(), name) && name != "resource descriptors" {
			t.Errorf("%s error is not actionable: %s", name, rec.Body.String())
		}
	}
	ordinary := "{" + base + `,"messages":[{"role":"user","content":"Use my notes to answer this question."}]}`
	if rec := notesRequest(router, http.MethodPost, "/v2/threads/"+thread.ThreadID+"/runs", "ada", ordinary); rec.Code != http.StatusOK {
		t.Fatalf("ordinary message text with Notes = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
}

func TestExpireNoteAppendProposalsErasesExactText(t *testing.T) {
	t.Parallel()
	h := newModelNotesHarness(t, domain.NoteAccessModeSearch)
	read := h.workerRequest(http.MethodPost, "/v2/runs/"+h.run.RunID+"/note-read", `{"note_id":"note_primary"}`)
	var readResult struct {
		Revision  int64  `json:"revision"`
		ReadToken string `json:"read_token"`
	}
	_ = json.Unmarshal(read.Body.Bytes(), &readResult)
	now := domain.Now()
	proposal, err := h.store.CreateNoteAppendProposal(context.Background(), domain.CreateNoteAppendProposalInput{
		ProposalID: "nprop_expiry", RunID: h.run.RunID, UserID: "ada", NoteID: h.note.NoteID,
		ExpectedRevision: readResult.Revision, BodyMarkdown: "private exact proposal",
		ReadTokenHash: domain.NoteBodySHA256(readResult.ReadToken), IdempotencyKey: "expiry-key",
		RequestDigest: domain.ComputeNoteContentDigest(h.note.NoteID+":1", "private exact proposal"),
		Now:           now, ExpiresAt: now.Add(time.Second),
	})
	if err != nil {
		t.Fatalf("create proposal: %v", err)
	}
	if expired, err := h.store.ExpireNoteAppendProposals(context.Background(), now.Add(2*time.Second), 100); err != nil || expired != 1 {
		t.Fatalf("expire = %d err=%v", expired, err)
	}
	proposal, err = h.store.GetNoteAppendProposalForUser(context.Background(), proposal.ProposalID, "ada")
	if err != nil || proposal.Status != domain.NoteAppendProposalStatusExpired || proposal.BodyMarkdown != "" {
		t.Fatalf("expired proposal = %+v err=%v", proposal, err)
	}
	replacement, err := h.store.CreateNoteAppendProposal(context.Background(), domain.CreateNoteAppendProposalInput{
		ProposalID: "nprop_replacement", RunID: h.run.RunID, UserID: "ada", NoteID: h.note.NoteID,
		ExpectedRevision: readResult.Revision, BodyMarkdown: "private exact proposal",
		ReadTokenHash: domain.NoteBodySHA256(readResult.ReadToken), IdempotencyKey: "replacement-key",
		RequestDigest: domain.ComputeNoteContentDigest(h.note.NoteID+":1", "private exact proposal"),
		Now:           now.Add(2 * time.Second), ExpiresAt: now.Add(time.Minute),
	})
	if err != nil || replacement.ProposalID != "nprop_replacement" {
		t.Fatalf("replacement after expiry = %+v err=%v", replacement, err)
	}
}
