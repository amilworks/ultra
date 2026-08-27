package httpapi

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

func newSteeringTestRouter(t *testing.T) (http.Handler, *runcontrol.Service, domain.RunRecord) {
	t.Helper()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{
		Version:     "test-version",
		Runs:        service,
		Store:       mem,
		WorkerToken: "steer-worker-secret",
	})
	thread, err := service.CreateThread(context.Background(), runcontrol.CreateThreadRequest{
		UserID: "ada", Title: "steering",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(context.Background(), runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "ada",
		Goal:     "long analysis",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "long analysis"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	return router, service, run
}

func TestSteerEndpointOwnershipAndWorkerGates(t *testing.T) {
	t.Parallel()
	router, _, run := newSteeringTestRouter(t)

	post := func(path, body, userID, workerToken string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
		if userID != "" {
			req.Header.Set("X-Ultra-User-Id", userID)
		}
		if workerToken != "" {
			req.Header.Set("X-Ultra-Worker-Token", workerToken)
		}
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		return rec
	}

	steerPath := "/v2/runs/" + run.RunID + "/steer"

	// A stranger must not learn the run exists, let alone steer it.
	if rec := post(steerPath, `{"text":"hijack"}`, "mallory", ""); rec.Code != http.StatusNotFound {
		t.Fatalf("stranger steer = %d, want 404", rec.Code)
	}

	// The owner steers.
	rec := post(steerPath, `{"steer_id":"steer_a","text":"Also check the baseline."}`, "ada", "")
	if rec.Code != http.StatusOK {
		t.Fatalf("owner steer = %d body=%s", rec.Code, rec.Body.String())
	}
	var record domain.RunSteerMessageRecord
	if err := json.Unmarshal(rec.Body.Bytes(), &record); err != nil {
		t.Fatalf("decode steer: %v", err)
	}
	if record.Status != domain.RunSteerStatusPending || record.MessageID == "" {
		t.Fatalf("steer record = %+v", record)
	}

	// Barrier and ack are worker-only — user identity is not enough.
	if rec := post(steerPath+"/barrier", `{}`, "ada", ""); rec.Code != http.StatusForbidden {
		t.Fatalf("user barrier = %d, want 403", rec.Code)
	}
	if rec := post(steerPath+"/steer_a/ack", `{}`, "ada", ""); rec.Code != http.StatusForbidden {
		t.Fatalf("user ack = %d, want 403", rec.Code)
	}

	// The worker acks, then closes the barrier.
	if rec := post(steerPath+"/steer_a/ack", `{"worker_id":"w1"}`, "", "steer-worker-secret"); rec.Code != http.StatusOK {
		t.Fatalf("worker ack = %d body=%s", rec.Code, rec.Body.String())
	}
	barrierRec := post(steerPath+"/barrier", `{}`, "", "steer-worker-secret")
	if barrierRec.Code != http.StatusOK {
		t.Fatalf("worker barrier = %d body=%s", barrierRec.Code, barrierRec.Body.String())
	}
	var barrier struct {
		Pending []domain.RunSteerMessageRecord `json:"pending"`
	}
	if err := json.Unmarshal(barrierRec.Body.Bytes(), &barrier); err != nil {
		t.Fatalf("decode barrier: %v", err)
	}
	if len(barrier.Pending) != 0 {
		t.Fatalf("barrier pending = %+v, want none (already applied)", barrier.Pending)
	}

	// Post-barrier steers 409 with the Phase 0 fallback code.
	late := post(steerPath, `{"text":"too late"}`, "ada", "")
	if late.Code != http.StatusConflict {
		t.Fatalf("post-barrier steer = %d, want 409", late.Code)
	}
	var conflict map[string]string
	if err := json.Unmarshal(late.Body.Bytes(), &conflict); err != nil {
		t.Fatalf("decode conflict: %v", err)
	}
	if conflict["code"] != "steering_closed" {
		t.Fatalf("conflict code = %q, want steering_closed", conflict["code"])
	}
}

func TestSteerEndpointQueuesNoteScopedRunsWithTypedConflict(t *testing.T) {
	t.Parallel()
	router, service, ordinaryRun := newSteeringTestRouter(t)
	noteRun, err := service.CreateRun(context.Background(), runcontrol.CreateRunRequest{
		ThreadID: ordinaryRun.ThreadID,
		UserID:   "ada",
		Goal:     "Use my Notes.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Use my Notes."}},
		SelectionContext: domain.CanonicalNoteAccessSelection(nil, domain.NoteAccessScope{
			Mode: domain.NoteAccessModeSearch,
		}),
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	req := httptest.NewRequest(
		http.MethodPost,
		"/v2/runs/"+noteRun.RunID+"/steer",
		strings.NewReader(`{"steer_id":"steer_notes","text":"Do not append anything."}`),
	)
	req.Header.Set("X-Ultra-User-Id", "ada")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusConflict {
		t.Fatalf("Notes steer = %d body=%s, want 409", rec.Code, rec.Body.String())
	}
	var conflict map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &conflict); err != nil {
		t.Fatalf("decode conflict: %v", err)
	}
	if conflict["code"] != "steering_closed" {
		t.Fatalf("conflict code = %q, want steering_closed", conflict["code"])
	}
	steers, err := service.ListRunSteerMessages(context.Background(), noteRun.RunID)
	if err != nil {
		t.Fatalf("ListRunSteerMessages: %v", err)
	}
	if len(steers) != 0 {
		t.Fatalf("typed conflict still persisted Notes steer: %+v", steers)
	}
}

func TestSteerListVisibleToWorkerAndOwnerOnly(t *testing.T) {
	t.Parallel()
	router, service, run := newSteeringTestRouter(t)
	if _, err := service.SteerRun(context.Background(), runcontrol.SteerRunRequest{
		RunID: run.RunID, UserID: "ada", SteerID: "steer_a", Text: "One more.",
	}); err != nil {
		t.Fatalf("SteerRun: %v", err)
	}

	get := func(userID, workerToken string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodGet, "/v2/runs/"+run.RunID+"/steer", nil)
		if userID != "" {
			req.Header.Set("X-Ultra-User-Id", userID)
		}
		if workerToken != "" {
			req.Header.Set("X-Ultra-Worker-Token", workerToken)
		}
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		return rec
	}

	if rec := get("mallory", ""); rec.Code != http.StatusNotFound {
		t.Fatalf("stranger list = %d, want 404", rec.Code)
	}
	for _, tc := range []struct{ user, token string }{{"ada", ""}, {"", "steer-worker-secret"}} {
		rec := get(tc.user, tc.token)
		if rec.Code != http.StatusOK {
			t.Fatalf("list (%q,%q) = %d body=%s", tc.user, tc.token, rec.Code, rec.Body.String())
		}
		var body struct {
			SteerMessages []domain.RunSteerMessageRecord `json:"steer_messages"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
			t.Fatalf("decode list: %v", err)
		}
		if len(body.SteerMessages) != 1 || body.SteerMessages[0].SteerID != "steer_a" {
			t.Fatalf("list = %+v", body.SteerMessages)
		}
	}
}

// Attachments ride the steer through the trusted channel: the record echoes
// the sanitized ids, the worker-visible list carries them, and the transcript
// row's metadata preserves them for requeue reseeding. Malformed and oversized
// id lists are rejected before anything persists.
func TestSteerAttachmentsFlowThroughRecordAndTranscript(t *testing.T) {
	t.Parallel()
	mem := store.NewMemoryStore()
	service := runcontrol.NewService(mem, eventbus.NewMemoryBus())
	router := NewRouter(ServerDeps{
		Version:     "test-version",
		Runs:        service,
		Store:       mem,
		WorkerToken: "steer-worker-secret",
	})
	thread, err := service.CreateThread(context.Background(), runcontrol.CreateThreadRequest{
		UserID: "ada", Title: "steering attachments",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(context.Background(), runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "ada",
		Goal:     "long analysis",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "long analysis"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	post := func(body string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPost, "/v2/runs/"+run.RunID+"/steer", strings.NewReader(body))
		req.Header.Set("X-Ultra-User-Id", "ada")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		return rec
	}

	// Duplicates collapse; blanks drop; order is preserved.
	rec := post(`{"steer_id":"steer_files","text":"Use this image instead.","file_ids":["file_abc123"," ","file_def456","file_abc123"]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("steer with files = %d body=%s", rec.Code, rec.Body.String())
	}
	var record domain.RunSteerMessageRecord
	if err := json.Unmarshal(rec.Body.Bytes(), &record); err != nil {
		t.Fatalf("decode steer: %v", err)
	}
	if len(record.FileIDs) != 2 || record.FileIDs[0] != "file_abc123" || record.FileIDs[1] != "file_def456" {
		t.Fatalf("record.FileIDs = %v, want [file_abc123 file_def456]", record.FileIDs)
	}

	// The worker's list view (the injection source) sees the same ids.
	listed, err := service.ListRunSteerMessages(context.Background(), run.RunID)
	if err != nil {
		t.Fatalf("ListRunSteerMessages: %v", err)
	}
	if len(listed) != 1 || len(listed[0].FileIDs) != 2 {
		t.Fatalf("listed steers = %+v, want one with two file ids", listed)
	}

	// The transcript message metadata carries the ids for requeue reseeding.
	messages, err := mem.ListThreadMessages(context.Background(), run.ThreadID)
	if err != nil {
		t.Fatalf("ListThreadMessages: %v", err)
	}
	var steerMessage *domain.ThreadMessage
	for i := range messages {
		if messages[i].MessageID == record.MessageID {
			steerMessage = &messages[i]
			break
		}
	}
	if steerMessage == nil {
		t.Fatalf("steer transcript message %q not found", record.MessageID)
	}
	metadataFileIDs, _ := steerMessage.Metadata["file_ids"].([]string)
	if len(metadataFileIDs) != 2 {
		t.Fatalf("transcript metadata file_ids = %#v, want two entries", steerMessage.Metadata["file_ids"])
	}

	// Path-hostile ids are rejected up front.
	if rec := post(`{"text":"bad","file_ids":["../../etc/passwd"]}`); rec.Code != http.StatusBadRequest {
		t.Fatalf("hostile file id = %d, want 400", rec.Code)
	}
	// So are oversized attachment lists.
	oversized := `{"text":"too many","file_ids":[`
	for i := 0; i < 17; i++ {
		if i > 0 {
			oversized += ","
		}
		oversized += `"file_` + strings.Repeat("a", 3) + string(rune('a'+i)) + `"`
	}
	oversized += `]}`
	if rec := post(oversized); rec.Code != http.StatusBadRequest {
		t.Fatalf("oversized file list = %d, want 400", rec.Code)
	}
}
