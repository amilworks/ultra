package httpapi

import (
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

func TestHealthAndPublicConfig(t *testing.T) {
	t.Parallel()

	router := NewRouter(ServerDeps{
		Version: "test-version",
	})

	healthReq := httptest.NewRequest(http.MethodGet, "/v1/health", nil)
	healthRec := httptest.NewRecorder()
	router.ServeHTTP(healthRec, healthReq)

	if healthRec.Code != http.StatusOK {
		t.Fatalf("health status = %d, want 200", healthRec.Code)
	}
	var health map[string]string
	if err := json.Unmarshal(healthRec.Body.Bytes(), &health); err != nil {
		t.Fatalf("decode health: %v", err)
	}
	if health["status"] != "ok" {
		t.Fatalf("health status body = %q, want ok", health["status"])
	}
	if health["ts"] == "" {
		t.Fatalf("health response must include ts")
	}

	configReq := httptest.NewRequest(http.MethodGet, "/v1/config/public", nil)
	configRec := httptest.NewRecorder()
	router.ServeHTTP(configRec, configReq)

	if configRec.Code != http.StatusOK {
		t.Fatalf("config status = %d, want 200", configRec.Code)
	}
	var config map[string]any
	if err := json.Unmarshal(configRec.Body.Bytes(), &config); err != nil {
		t.Fatalf("decode config: %v", err)
	}
	if config["app_version"] != "test-version" {
		t.Fatalf("app_version = %v, want test-version", config["app_version"])
	}
}

func TestV2ThreadRunArtifactHandlers(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	createThreadBody := strings.NewReader(`{"title":"Research","initial_messages":[{"role":"user","content":"hello"}]}`)
	createThreadReq := httptest.NewRequest(http.MethodPost, "/v2/threads", createThreadBody)
	createThreadReq.Header.Set("Content-Type", "application/json")
	createThreadRec := httptest.NewRecorder()
	router.ServeHTTP(createThreadRec, createThreadReq)
	if createThreadRec.Code != http.StatusOK {
		t.Fatalf("create thread status = %d body=%s", createThreadRec.Code, createThreadRec.Body.String())
	}
	var thread map[string]any
	if err := json.Unmarshal(createThreadRec.Body.Bytes(), &thread); err != nil {
		t.Fatalf("decode thread: %v", err)
	}
	threadID, ok := thread["thread_id"].(string)
	if !ok || threadID == "" {
		t.Fatalf("thread response missing thread_id: %+v", thread)
	}

	createRunBody := strings.NewReader(`{"goal":"hello","messages":[{"role":"user","content":"hello"}],"file_ids":["file-1"],"reasoning_mode":"deep"}`)
	createRunReq := httptest.NewRequest(http.MethodPost, "/v2/threads/"+threadID+"/runs", createRunBody)
	createRunReq.Header.Set("Content-Type", "application/json")
	createRunRec := httptest.NewRecorder()
	router.ServeHTTP(createRunRec, createRunReq)
	if createRunRec.Code != http.StatusOK {
		t.Fatalf("create run status = %d body=%s", createRunRec.Code, createRunRec.Body.String())
	}
	var run map[string]any
	if err := json.Unmarshal(createRunRec.Body.Bytes(), &run); err != nil {
		t.Fatalf("decode run: %v", err)
	}
	runID, ok := run["run_id"].(string)
	if !ok || runID == "" {
		t.Fatalf("run response missing run_id: %+v", run)
	}
	if run["thread_id"] != threadID {
		t.Fatalf("run thread = %v, want %s", run["thread_id"], threadID)
	}

	eventsReq := httptest.NewRequest(http.MethodGet, "/v2/runs/"+runID+"/events?limit=10", nil)
	eventsRec := httptest.NewRecorder()
	router.ServeHTTP(eventsRec, eventsReq)
	if eventsRec.Code != http.StatusOK {
		t.Fatalf("events status = %d body=%s", eventsRec.Code, eventsRec.Body.String())
	}
	var events struct {
		RunID  string                  `json:"run_id"`
		Count  int                     `json:"count"`
		Events []domain.RunEventRecord `json:"events"`
	}
	if err := json.Unmarshal(eventsRec.Body.Bytes(), &events); err != nil {
		t.Fatalf("decode events: %v", err)
	}
	if events.Count != 1 || events.Events[0].EventKind != "run.accepted" {
		t.Fatalf("events = %+v, want run.accepted", events)
	}
}
