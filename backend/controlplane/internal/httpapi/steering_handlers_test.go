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
