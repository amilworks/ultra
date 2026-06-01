package httpapi

import (
	"bytes"
	"context"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

type fakeQueueDiagnosticsProvider struct {
	diagnostics eventbus.QueueDiagnostics
	err         error
}

func (p fakeQueueDiagnosticsProvider) QueueDiagnostics(context.Context) (eventbus.QueueDiagnostics, error) {
	return p.diagnostics, p.err
}

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
	if config["admin_enabled"] != false {
		t.Fatalf("admin_enabled = %v, want false without explicit local admin deps", config["admin_enabled"])
	}
}

func TestDevAuthGuestSessionLifecycle(t *testing.T) {
	t.Parallel()

	router := NewRouter(ServerDeps{Version: "test-version", DevAdminEnabled: true})

	sessionReq := httptest.NewRequest(http.MethodGet, "/v1/auth/session", nil)
	sessionRec := httptest.NewRecorder()
	router.ServeHTTP(sessionRec, sessionReq)
	if sessionRec.Code != http.StatusOK {
		t.Fatalf("default session status = %d body=%s", sessionRec.Code, sessionRec.Body.String())
	}
	var defaultSession map[string]any
	if err := json.Unmarshal(sessionRec.Body.Bytes(), &defaultSession); err != nil {
		t.Fatalf("decode default session: %v", err)
	}
	if defaultSession["authenticated"] != true || defaultSession["mode"] != "guest" {
		t.Fatalf("default session = %#v, want local guest session", defaultSession)
	}
	if defaultSession["is_admin"] != true {
		t.Fatalf("default session is_admin = %#v, want local dev admin access", defaultSession["is_admin"])
	}
	user, ok := defaultSession["user"].(map[string]any)
	if !ok || user["role"] != "admin" {
		t.Fatalf("default session user = %#v, want admin role", defaultSession["user"])
	}

	guestBody := strings.NewReader(`{"name":"Ada Lovelace","email":"ada@example.org","affiliation":"Analytical Engine Lab"}`)
	guestReq := httptest.NewRequest(http.MethodPost, "/v1/auth/guest", guestBody)
	guestReq.Header.Set("Content-Type", "application/json")
	guestRec := httptest.NewRecorder()
	router.ServeHTTP(guestRec, guestReq)
	if guestRec.Code != http.StatusOK {
		t.Fatalf("guest auth status = %d body=%s", guestRec.Code, guestRec.Body.String())
	}
	var guestSession map[string]any
	if err := json.Unmarshal(guestRec.Body.Bytes(), &guestSession); err != nil {
		t.Fatalf("decode guest session: %v", err)
	}
	if guestSession["authenticated"] != true || guestSession["username"] != "Ada Lovelace" {
		t.Fatalf("guest session = %#v, want authenticated Ada", guestSession)
	}
	if guestSession["is_admin"] != true {
		t.Fatalf("guest session is_admin = %#v, want local dev admin access", guestSession["is_admin"])
	}
	if len(guestRec.Result().Cookies()) == 0 {
		t.Fatalf("guest auth should set a dev session cookie")
	}
	cookieSessionReq := httptest.NewRequest(http.MethodGet, "/v1/auth/session", nil)
	cookieSessionReq.AddCookie(guestRec.Result().Cookies()[0])
	cookieSessionRec := httptest.NewRecorder()
	router.ServeHTTP(cookieSessionRec, cookieSessionReq)
	var cookieSession map[string]any
	if err := json.Unmarshal(cookieSessionRec.Body.Bytes(), &cookieSession); err != nil {
		t.Fatalf("decode cookie session: %v", err)
	}
	if cookieSession["username"] != "Ada Lovelace" || cookieSession["mode"] != "guest" {
		t.Fatalf("cookie session = %#v, want persisted guest", cookieSession)
	}
	if cookieSession["is_admin"] != true {
		t.Fatalf("cookie session is_admin = %#v, want local dev admin access", cookieSession["is_admin"])
	}

	logoutReq := httptest.NewRequest(http.MethodPost, "/v1/auth/logout", nil)
	logoutRec := httptest.NewRecorder()
	router.ServeHTTP(logoutRec, logoutReq)
	if logoutRec.Code != http.StatusOK {
		t.Fatalf("logout status = %d body=%s", logoutRec.Code, logoutRec.Body.String())
	}
	var logoutSession map[string]any
	if err := json.Unmarshal(logoutRec.Body.Bytes(), &logoutSession); err != nil {
		t.Fatalf("decode logout session: %v", err)
	}
	if logoutSession["authenticated"] != false {
		t.Fatalf("logout session = %#v, want unauthenticated response", logoutSession)
	}
}

func TestDevAuthSessionCanDisableLocalAdmin(t *testing.T) {
	t.Parallel()

	router := NewRouter(ServerDeps{Version: "test-version", DevAdminEnabled: false})

	sessionReq := httptest.NewRequest(http.MethodGet, "/v2/auth/session", nil)
	sessionRec := httptest.NewRecorder()
	router.ServeHTTP(sessionRec, sessionReq)
	if sessionRec.Code != http.StatusOK {
		t.Fatalf("session status = %d body=%s", sessionRec.Code, sessionRec.Body.String())
	}
	var session map[string]any
	if err := json.Unmarshal(sessionRec.Body.Bytes(), &session); err != nil {
		t.Fatalf("decode session: %v", err)
	}
	if session["is_admin"] != false {
		t.Fatalf("session is_admin = %#v, want disabled local admin", session["is_admin"])
	}
	user, ok := session["user"].(map[string]any)
	if !ok || user["role"] != "researcher" {
		t.Fatalf("session user = %#v, want researcher role", session["user"])
	}
}

func TestV2HealthConfigAndAuthAliases(t *testing.T) {
	t.Parallel()

	router := NewRouter(ServerDeps{Version: "test-version", DevAdminEnabled: true})

	for _, path := range []string{"/v2/health", "/v2/config/public", "/v2/auth/session"} {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("%s status = %d body=%s", path, rec.Code, rec.Body.String())
		}
	}

	guestReq := httptest.NewRequest(http.MethodPost, "/v2/auth/guest", strings.NewReader(`{"name":"Grace Hopper"}`))
	guestReq.Header.Set("Content-Type", "application/json")
	guestRec := httptest.NewRecorder()
	router.ServeHTTP(guestRec, guestReq)
	if guestRec.Code != http.StatusOK {
		t.Fatalf("guest alias status = %d body=%s", guestRec.Code, guestRec.Body.String())
	}

	loginReq := httptest.NewRequest(http.MethodPost, "/v2/auth/login", strings.NewReader(`{"username":"local-user"}`))
	loginReq.Header.Set("Content-Type", "application/json")
	loginRec := httptest.NewRecorder()
	router.ServeHTTP(loginRec, loginReq)
	if loginRec.Code != http.StatusOK {
		t.Fatalf("login alias status = %d body=%s", loginRec.Code, loginRec.Body.String())
	}

	logoutReq := httptest.NewRequest(http.MethodPost, "/v2/auth/logout", nil)
	logoutRec := httptest.NewRecorder()
	router.ServeHTTP(logoutRec, logoutReq)
	if logoutRec.Code != http.StatusOK {
		t.Fatalf("logout alias status = %d body=%s", logoutRec.Code, logoutRec.Body.String())
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

	createRunBody := strings.NewReader(`{"goal":"hello","messages":[{"role":"user","content":"hello"}],"file_ids":["file-1"],"resource_uris":["bisque://resource/1"],"dataset_uris":["bisque://dataset/2"],"selected_tool_names":["rarespot_ecology_inference"],"knowledge_context":{"active_paper":"arxiv:2509.26626"},"workflow_hint":{"id":"rarespot_ecology"},"selection_context":{"source":"sidebar"},"budgets":{"max_runtime_seconds":1800},"reasoning_mode":"deep","benchmark":{"suite":"http-context"}}`)
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
	if run["workflow_kind"] != "rarespot_ecology" {
		t.Fatalf("run workflow_kind = %v, want rarespot_ecology", run["workflow_kind"])
	}
	metadata, ok := run["metadata"].(map[string]any)
	if !ok {
		t.Fatalf("run metadata missing: %+v", run)
	}
	if got := metadata["file_ids"]; !jsonArrayEquals(got, []string{"file-1"}) {
		t.Fatalf("metadata file_ids = %#v, want file-1", got)
	}
	if got := metadata["resource_uris"]; !jsonArrayEquals(got, []string{"bisque://resource/1"}) {
		t.Fatalf("metadata resource_uris = %#v, want resource URI", got)
	}
	if got := metadata["dataset_uris"]; !jsonArrayEquals(got, []string{"bisque://dataset/2"}) {
		t.Fatalf("metadata dataset_uris = %#v, want dataset URI", got)
	}
	knowledge, ok := metadata["knowledge_context"].(map[string]any)
	if !ok || knowledge["active_paper"] != "arxiv:2509.26626" {
		t.Fatalf("metadata knowledge_context = %#v, want active paper", metadata["knowledge_context"])
	}
	if metadata["reasoning_mode"] != "deep" {
		t.Fatalf("metadata reasoning_mode = %#v, want deep", metadata["reasoning_mode"])
	}
	benchmark, ok := metadata["benchmark"].(map[string]any)
	if !ok || benchmark["suite"] != "http-context" {
		t.Fatalf("metadata benchmark = %#v, want http context", metadata["benchmark"])
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

func TestV2ThreadAndRunCreationUsesDevPrincipalHeaders(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem, Bus: bus})

	createThreadReq := httptest.NewRequest(http.MethodPost, "/v2/threads", strings.NewReader(`{"user_id":"body-user","title":"Principal thread"}`))
	createThreadReq.Header.Set("Content-Type", "application/json")
	createThreadReq.Header.Set("X-Ultra-User-Id", "ada")
	createThreadReq.Header.Set("X-Ultra-Org-Id", "allen-institute")
	createThreadReq.Header.Set("X-Ultra-Role", "admin")
	createThreadRec := httptest.NewRecorder()
	router.ServeHTTP(createThreadRec, createThreadReq)
	if createThreadRec.Code != http.StatusOK {
		t.Fatalf("create thread status = %d body=%s", createThreadRec.Code, createThreadRec.Body.String())
	}
	var thread domain.ThreadRecord
	if err := json.Unmarshal(createThreadRec.Body.Bytes(), &thread); err != nil {
		t.Fatalf("decode thread: %v", err)
	}
	if thread.UserID != "ada" {
		t.Fatalf("thread user_id = %q, want principal header user", thread.UserID)
	}
	threadPrincipal, ok := thread.Metadata["principal"].(map[string]any)
	if !ok {
		t.Fatalf("thread metadata = %+v, want principal metadata", thread.Metadata)
	}
	if threadPrincipal["user_id"] != "ada" || threadPrincipal["org_id"] != "allen-institute" || threadPrincipal["role"] != "admin" {
		t.Fatalf("thread principal = %+v, want header principal", threadPrincipal)
	}

	createRunReq := httptest.NewRequest(http.MethodPost, "/v2/threads/"+thread.ThreadID+"/runs", strings.NewReader(`{"user_id":"body-user","goal":"Run attributed work","messages":[{"role":"user","content":"Run attributed work"}],"metadata":{"existing":"kept"}}`))
	createRunReq.Header.Set("Content-Type", "application/json")
	createRunReq.Header.Set("X-Ultra-User-Id", "ada")
	createRunReq.Header.Set("X-Ultra-Org-Id", "allen-institute")
	createRunReq.Header.Set("X-Ultra-Role", "admin")
	createRunRec := httptest.NewRecorder()
	router.ServeHTTP(createRunRec, createRunReq)
	if createRunRec.Code != http.StatusOK {
		t.Fatalf("create run status = %d body=%s", createRunRec.Code, createRunRec.Body.String())
	}
	var run domain.RunRecord
	if err := json.Unmarshal(createRunRec.Body.Bytes(), &run); err != nil {
		t.Fatalf("decode run: %v", err)
	}
	if run.UserID != "ada" {
		t.Fatalf("run user_id = %q, want principal header user", run.UserID)
	}
	if run.Metadata["existing"] != "kept" {
		t.Fatalf("run metadata existing = %+v, want caller metadata preserved", run.Metadata)
	}
	runPrincipal, ok := run.Metadata["principal"].(map[string]any)
	if !ok {
		t.Fatalf("run metadata = %+v, want principal metadata", run.Metadata)
	}
	if runPrincipal["user_id"] != "ada" || runPrincipal["org_id"] != "allen-institute" || runPrincipal["role"] != "admin" {
		t.Fatalf("run principal = %+v, want header principal", runPrincipal)
	}
}

func TestV2ThreadAndRunCreationDefaultsLocalDevPrincipal(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem, Bus: bus})

	createThreadReq := httptest.NewRequest(http.MethodPost, "/v2/threads", strings.NewReader(`{"title":"Default principal thread"}`))
	createThreadReq.Header.Set("Content-Type", "application/json")
	createThreadRec := httptest.NewRecorder()
	router.ServeHTTP(createThreadRec, createThreadReq)
	if createThreadRec.Code != http.StatusOK {
		t.Fatalf("create thread status = %d body=%s", createThreadRec.Code, createThreadRec.Body.String())
	}
	var thread domain.ThreadRecord
	if err := json.Unmarshal(createThreadRec.Body.Bytes(), &thread); err != nil {
		t.Fatalf("decode thread: %v", err)
	}
	if thread.UserID != "local-user" {
		t.Fatalf("thread user_id = %q, want default local-user", thread.UserID)
	}
	principal, ok := thread.Metadata["principal"].(map[string]any)
	if !ok || principal["org_id"] != "local-org" || principal["role"] != "researcher" {
		t.Fatalf("thread principal = %+v, want default local principal", thread.Metadata["principal"])
	}

	createRunReq := httptest.NewRequest(http.MethodPost, "/v2/threads/"+thread.ThreadID+"/runs", strings.NewReader(`{"goal":"Run default principal work"}`))
	createRunReq.Header.Set("Content-Type", "application/json")
	createRunRec := httptest.NewRecorder()
	router.ServeHTTP(createRunRec, createRunReq)
	if createRunRec.Code != http.StatusOK {
		t.Fatalf("create run status = %d body=%s", createRunRec.Code, createRunRec.Body.String())
	}
	var run domain.RunRecord
	if err := json.Unmarshal(createRunRec.Body.Bytes(), &run); err != nil {
		t.Fatalf("decode run: %v", err)
	}
	if run.UserID != "local-user" {
		t.Fatalf("run user_id = %q, want default local-user", run.UserID)
	}
	runPrincipal, ok := run.Metadata["principal"].(map[string]any)
	if !ok || runPrincipal["user_id"] != "local-user" || runPrincipal["org_id"] != "local-org" || runPrincipal["role"] != "researcher" {
		t.Fatalf("run principal = %+v, want default local principal", run.Metadata["principal"])
	}
}

func TestV2ListRunsHandler(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "user-1",
		Title:  "runs",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	first, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "queued run",
	})
	if err != nil {
		t.Fatalf("CreateRun first: %v", err)
	}
	second, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "running run",
	})
	if err != nil {
		t.Fatalf("CreateRun second: %v", err)
	}
	if _, err := mem.UpdateRunStatus(ctx, second.RunID, domain.RunStatusRunning, "", ""); err != nil {
		t.Fatalf("UpdateRunStatus: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/runs?limit=20&status=queued", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("list runs status = %d body=%s", rec.Code, rec.Body.String())
	}

	var response struct {
		Count int                `json:"count"`
		Runs  []domain.RunRecord `json:"runs"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode list runs: %v", err)
	}
	if response.Count != 1 || len(response.Runs) != 1 || response.Runs[0].RunID != first.RunID {
		t.Fatalf("runs = %+v, want queued first run only", response)
	}
}

func TestV2AdminAndTrainingReadEndpointsAreOwnedByGo(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "user-1",
		Title:  "admin",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "inspect me",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	if _, err := mem.UpdateRunStatus(ctx, run.RunID, domain.RunStatusFailed, "", "synthetic failure"); err != nil {
		t.Fatalf("UpdateRunStatus: %v", err)
	}

	cases := []struct {
		path       string
		wantKeys   []string
		statusCode int
	}{
		{path: "/v2/admin/overview", wantKeys: []string{"generated_at", "runtime", "kpis", "recent_issues"}, statusCode: http.StatusOK},
		{path: "/v2/admin/orgs", wantKeys: []string{"count", "organizations"}, statusCode: http.StatusOK},
		{path: "/v2/admin/users", wantKeys: []string{"count", "users"}, statusCode: http.StatusOK},
		{path: "/v2/admin/runs?status=failed", wantKeys: []string{"count", "runs"}, statusCode: http.StatusOK},
		{path: "/v2/admin/issues", wantKeys: []string{"count", "issues"}, statusCode: http.StatusOK},
		{path: "/v2/training/models", wantKeys: []string{"count", "models"}, statusCode: http.StatusOK},
		{path: "/v2/training/prairie/status", wantKeys: []string{"dataset_name", "model_health", "retrain_gate_reasons"}, statusCode: http.StatusOK},
		{path: "/v2/training/prairie/retrain-requests", wantKeys: []string{"count", "requests"}, statusCode: http.StatusOK},
		{path: "/v2/training/domains", wantKeys: []string{"count", "domains"}, statusCode: http.StatusOK},
		{path: "/v2/training/domains/prairie/lineages", wantKeys: []string{"count", "lineages"}, statusCode: http.StatusOK},
		{path: "/v2/training/lineages/prairie-default/versions", wantKeys: []string{"count", "versions"}, statusCode: http.StatusOK},
	}
	for _, tc := range cases {
		req := httptest.NewRequest(http.MethodGet, tc.path, nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != tc.statusCode {
			t.Fatalf("%s status = %d body=%s", tc.path, rec.Code, rec.Body.String())
		}
		var payload map[string]any
		if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
			t.Fatalf("%s decode: %v body=%s", tc.path, err, rec.Body.String())
		}
		for _, key := range tc.wantKeys {
			if _, ok := payload[key]; !ok {
				t.Fatalf("%s missing key %q in %#v", tc.path, key, payload)
			}
		}
	}
}

func TestV2AdminCreateUserPersistsFirstClassAccount(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	body := `{"email":"grace@example.org","display_name":"Grace Hopper","role":"admin","org_id":"local-org"}`
	req := httptest.NewRequest(http.MethodPost, "/v2/admin/users", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusCreated {
		t.Fatalf("create user status = %d body=%s", rec.Code, rec.Body.String())
	}
	var created domain.UserAccount
	if err := json.Unmarshal(rec.Body.Bytes(), &created); err != nil {
		t.Fatalf("decode created user: %v body=%s", err, rec.Body.String())
	}
	if created.UserID == "" || created.Email != "grace@example.org" || created.DisplayName != "Grace Hopper" {
		t.Fatalf("created user = %+v, want persisted account fields", created)
	}

	listReq := httptest.NewRequest(http.MethodGet, "/v2/admin/users?q=grace", nil)
	listRec := httptest.NewRecorder()
	router.ServeHTTP(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("list users status = %d body=%s", listRec.Code, listRec.Body.String())
	}
	var payload adminUserListResponse
	if err := json.Unmarshal(listRec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode list users: %v body=%s", err, listRec.Body.String())
	}
	if payload.Count != 1 {
		t.Fatalf("user count = %d, want 1 payload=%+v", payload.Count, payload)
	}
	got := payload.Users[0]
	if got.UserID != created.UserID || got.Email != "grace@example.org" || got.DisplayName != "Grace Hopper" || got.Role != "admin" || got.Status != "active" {
		t.Fatalf("listed user = %+v, want created account plus telemetry", got)
	}
}

func TestV2AdminCreateUserDuplicateEmailReturnsConflict(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	first := httptest.NewRequest(http.MethodPost, "/v2/admin/users", strings.NewReader(`{"email":"ada@example.org"}`))
	first.Header.Set("Content-Type", "application/json")
	firstRec := httptest.NewRecorder()
	router.ServeHTTP(firstRec, first)
	if firstRec.Code != http.StatusCreated {
		t.Fatalf("first create status = %d body=%s", firstRec.Code, firstRec.Body.String())
	}

	duplicate := httptest.NewRequest(http.MethodPost, "/v2/admin/users", strings.NewReader(`{"email":"ADA@example.org"}`))
	duplicate.Header.Set("Content-Type", "application/json")
	duplicateRec := httptest.NewRecorder()
	router.ServeHTTP(duplicateRec, duplicate)
	if duplicateRec.Code != http.StatusConflict {
		t.Fatalf("duplicate create status = %d body=%s", duplicateRec.Code, duplicateRec.Body.String())
	}
	if !strings.Contains(strings.ToLower(duplicateRec.Body.String()), "already exists") {
		t.Fatalf("duplicate response should explain conflict: %s", duplicateRec.Body.String())
	}
}

func TestV2AdminDeleteUserSoftDisablesAccount(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	create := httptest.NewRequest(http.MethodPost, "/v2/admin/users", strings.NewReader(`{"email":"remove-me@example.org","display_name":"Remove Me"}`))
	create.Header.Set("Content-Type", "application/json")
	createRec := httptest.NewRecorder()
	router.ServeHTTP(createRec, create)
	if createRec.Code != http.StatusCreated {
		t.Fatalf("create user status = %d body=%s", createRec.Code, createRec.Body.String())
	}
	var created domain.UserAccount
	if err := json.Unmarshal(createRec.Body.Bytes(), &created); err != nil {
		t.Fatalf("decode created user: %v", err)
	}

	req := httptest.NewRequest(http.MethodDelete, "/v2/admin/users/"+created.UserID, nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("delete user status = %d body=%s", rec.Code, rec.Body.String())
	}
	var disabled domain.UserAccount
	if err := json.Unmarshal(rec.Body.Bytes(), &disabled); err != nil {
		t.Fatalf("decode disabled user: %v body=%s", err, rec.Body.String())
	}
	if disabled.UserID != created.UserID || disabled.Status != "disabled" {
		t.Fatalf("disabled user = %+v, want same user with disabled status", disabled)
	}

	listReq := httptest.NewRequest(http.MethodGet, "/v2/admin/users?q=remove-me", nil)
	listRec := httptest.NewRecorder()
	router.ServeHTTP(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("list users status = %d body=%s", listRec.Code, listRec.Body.String())
	}
	var payload adminUserListResponse
	if err := json.Unmarshal(listRec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode list users: %v body=%s", err, listRec.Body.String())
	}
	if payload.Count != 1 || payload.Users[0].Status != "disabled" {
		t.Fatalf("listed users = %+v, want disabled account retained", payload.Users)
	}
}

func TestV2AdminCreateOrganizationPersistsFirstClassOrg(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	body := `{"org_id":"allen-institute","name":"Allen Institute","status":"active","metadata":{"source":"admin_console"}}`
	req := httptest.NewRequest(http.MethodPost, "/v2/admin/orgs", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusCreated {
		t.Fatalf("create org status = %d body=%s", rec.Code, rec.Body.String())
	}
	var created domain.Organization
	if err := json.Unmarshal(rec.Body.Bytes(), &created); err != nil {
		t.Fatalf("decode created org: %v body=%s", err, rec.Body.String())
	}
	if created.OrgID != "allen-institute" || created.Name != "Allen Institute" || created.Status != "active" {
		t.Fatalf("created org = %+v, want persisted organization fields", created)
	}

	listReq := httptest.NewRequest(http.MethodGet, "/v2/admin/orgs?q=allen", nil)
	listRec := httptest.NewRecorder()
	router.ServeHTTP(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("list orgs status = %d body=%s", listRec.Code, listRec.Body.String())
	}
	var payload adminOrganizationListResponse
	if err := json.Unmarshal(listRec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode list orgs: %v body=%s", err, listRec.Body.String())
	}
	if payload.Count != 1 || payload.Organizations[0].OrgID != created.OrgID {
		t.Fatalf("org list = %+v, want created organization", payload)
	}
}

func TestV2AdminCreateOrganizationDuplicateIDReturnsConflict(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	first := httptest.NewRequest(http.MethodPost, "/v2/admin/orgs", strings.NewReader(`{"org_id":"smithsonian","name":"Smithsonian"}`))
	first.Header.Set("Content-Type", "application/json")
	firstRec := httptest.NewRecorder()
	router.ServeHTTP(firstRec, first)
	if firstRec.Code != http.StatusCreated {
		t.Fatalf("first create status = %d body=%s", firstRec.Code, firstRec.Body.String())
	}

	duplicate := httptest.NewRequest(http.MethodPost, "/v2/admin/orgs", strings.NewReader(`{"org_id":"smithsonian","name":"Smithsonian duplicate"}`))
	duplicate.Header.Set("Content-Type", "application/json")
	duplicateRec := httptest.NewRecorder()
	router.ServeHTTP(duplicateRec, duplicate)
	if duplicateRec.Code != http.StatusConflict {
		t.Fatalf("duplicate create status = %d body=%s", duplicateRec.Code, duplicateRec.Body.String())
	}
}

func TestV2RunLeaseClaimRenewAndRelease(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	thread, err := service.CreateThread(context.Background(), runcontrol.CreateThreadRequest{
		UserID: "user-1",
		Title:  "lease",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(context.Background(), runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "long worker run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "long worker run"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	claim := httptest.NewRequest(http.MethodPost, "/v2/runs/"+run.RunID+"/lease", strings.NewReader(`{"worker_id":"worker-a","ttl_seconds":60}`))
	claim.Header.Set("Content-Type", "application/json")
	claimRec := httptest.NewRecorder()
	router.ServeHTTP(claimRec, claim)
	if claimRec.Code != http.StatusOK {
		t.Fatalf("claim status = %d body=%s", claimRec.Code, claimRec.Body.String())
	}
	var lease domain.RunLeaseRecord
	if err := json.Unmarshal(claimRec.Body.Bytes(), &lease); err != nil {
		t.Fatalf("decode lease: %v body=%s", err, claimRec.Body.String())
	}
	if lease.RunID != run.RunID || lease.WorkerID != "worker-a" || lease.LeaseToken == "" {
		t.Fatalf("lease = %+v, want worker-a token", lease)
	}

	competing := httptest.NewRequest(http.MethodPost, "/v2/runs/"+run.RunID+"/lease", strings.NewReader(`{"worker_id":"worker-b","ttl_seconds":60}`))
	competing.Header.Set("Content-Type", "application/json")
	competingRec := httptest.NewRecorder()
	router.ServeHTTP(competingRec, competing)
	if competingRec.Code != http.StatusConflict {
		t.Fatalf("competing claim status = %d body=%s", competingRec.Code, competingRec.Body.String())
	}

	renewBody := `{"lease_token":"` + lease.LeaseToken + `","ttl_seconds":120}`
	renew := httptest.NewRequest(http.MethodPatch, "/v2/runs/"+run.RunID+"/lease", strings.NewReader(renewBody))
	renew.Header.Set("Content-Type", "application/json")
	renewRec := httptest.NewRecorder()
	router.ServeHTTP(renewRec, renew)
	if renewRec.Code != http.StatusOK {
		t.Fatalf("renew status = %d body=%s", renewRec.Code, renewRec.Body.String())
	}

	releaseBody := `{"lease_token":"` + lease.LeaseToken + `"}`
	release := httptest.NewRequest(http.MethodDelete, "/v2/runs/"+run.RunID+"/lease", strings.NewReader(releaseBody))
	release.Header.Set("Content-Type", "application/json")
	releaseRec := httptest.NewRecorder()
	router.ServeHTTP(releaseRec, release)
	if releaseRec.Code != http.StatusOK {
		t.Fatalf("release status = %d body=%s", releaseRec.Code, releaseRec.Body.String())
	}

	reclaim := httptest.NewRequest(http.MethodPost, "/v2/runs/"+run.RunID+"/lease", strings.NewReader(`{"worker_id":"worker-b","ttl_seconds":60}`))
	reclaim.Header.Set("Content-Type", "application/json")
	reclaimRec := httptest.NewRecorder()
	router.ServeHTTP(reclaimRec, reclaim)
	if reclaimRec.Code != http.StatusOK {
		t.Fatalf("reclaim status = %d body=%s", reclaimRec.Code, reclaimRec.Body.String())
	}
}

func TestV2AdminOverviewIncludesRuntimeTransportSummary(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{
		Version:      "test-version",
		Runs:         service,
		Store:        mem,
		ArtifactRoot: "/tmp/ultra-artifacts",
		UploadRoot:   "/tmp/ultra-uploads",
		Runtime: RuntimeSummary{
			AppVersion:              "test-version",
			StoreBackend:            "memory",
			DispatchMode:            "nats_jetstream",
			JobTransport:            "nats_jetstream",
			EventTransport:          "nats_jetstream_to_local_fanout",
			StubWorkerEnabled:       false,
			NATSConfigured:          true,
			NATSStream:              "ULTRA_RUNS",
			NATSJobsSubject:         "ultra.runs.jobs",
			NATSRareSpotJobsSubject: "ultra.runs.rarespot.jobs",
			NATSEventsSubject:       "ultra.runs.events",
			NATSCancelSubject:       "ultra.runs.cancel",
			NATSEventConsumer:       "ultra-control-event-ingest",
			ArtifactRoot:            "/tmp/ultra-artifacts",
			UploadRoot:              "/tmp/ultra-uploads",
		},
		QueueDiagnostics: fakeQueueDiagnosticsProvider{
			diagnostics: eventbus.QueueDiagnostics{
				Available:      true,
				Mode:           "nats_jetstream",
				Stream:         "ULTRA_RUNS",
				StreamSubjects: []string{"ultra.runs.jobs", "ultra.runs.events", "ultra.runs.cancel"},
				StreamMessages: 42,
				StreamBytes:    4096,
				FirstSequence:  10,
				LastSequence:   52,
				ConsumerCount:  2,
				Consumers: []eventbus.QueueConsumerDiagnostics{{
					Name:                "ultra-deepagents-worker",
					Role:                "deepagents",
					Subject:             "ultra.runs.jobs",
					Active:              true,
					AckWaitSeconds:      600,
					MaxDeliver:          5,
					PendingMessages:     3,
					InFlightMessages:    1,
					RedeliveredMessages: 2,
					WaitingPullRequests: 1,
				}},
			},
		},
	})

	req := httptest.NewRequest(http.MethodGet, "/v2/admin/overview", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("admin overview status = %d body=%s", rec.Code, rec.Body.String())
	}
	var payload struct {
		Runtime RuntimeSummary        `json:"runtime"`
		Queue   adminQueueDiagnostics `json:"queue"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode admin overview: %v", err)
	}
	runtime := payload.Runtime
	if runtime.DispatchMode != "nats_jetstream" || runtime.JobTransport != "nats_jetstream" {
		t.Fatalf("runtime transport = %+v, want nats dispatch/job transport", runtime)
	}
	if !runtime.NATSConfigured || runtime.NATSStream != "ULTRA_RUNS" || runtime.NATSJobsSubject != "ultra.runs.jobs" {
		t.Fatalf("nats runtime fields = %+v, want configured subjects", runtime)
	}
	if runtime.StubWorkerEnabled {
		t.Fatalf("stub worker enabled = true, want false for NATS runtime: %+v", runtime)
	}
	if runtime.ArtifactRoot == "" || runtime.UploadRoot == "" {
		t.Fatalf("runtime roots = %+v, want artifact/upload roots for operator diagnostics", runtime)
	}
	if !payload.Queue.Available || payload.Queue.Stream != "ULTRA_RUNS" || payload.Queue.StreamMessages != 42 {
		t.Fatalf("queue diagnostics = %+v, want stream health", payload.Queue)
	}
	if len(payload.Queue.Consumers) != 1 {
		t.Fatalf("queue consumers = %+v, want one worker consumer", payload.Queue.Consumers)
	}
	consumer := payload.Queue.Consumers[0]
	if consumer.Name != "ultra-deepagents-worker" || consumer.PendingMessages != 3 || consumer.InFlightMessages != 1 || consumer.RedeliveredMessages != 2 {
		t.Fatalf("consumer diagnostics = %+v, want pending/in-flight/redelivery counts", consumer)
	}
}

func TestV2WorkerHeartbeatFeedsAdminOverview(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{
		Version: "test-version",
		Runs:    service,
		Store:   mem,
		Bus:     bus,
	})

	body := `{
		"worker_id":"deepagents-worker-a",
		"worker_kind":"deepagents",
		"status":"busy",
		"current_run_id":"run_123",
		"hostname":"host-a",
		"version":"worker-test-version",
		"metadata":{"active_tasks":1}
	}`
	req := httptest.NewRequest(http.MethodPost, "/v2/workers/heartbeat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("worker heartbeat status = %d body=%s", rec.Code, rec.Body.String())
	}
	var heartbeat domain.WorkerHeartbeatRecord
	if err := json.Unmarshal(rec.Body.Bytes(), &heartbeat); err != nil {
		t.Fatalf("decode heartbeat: %v", err)
	}
	if heartbeat.WorkerID != "deepagents-worker-a" || heartbeat.Status != "busy" || heartbeat.CurrentRunID != "run_123" {
		t.Fatalf("heartbeat response = %+v, want busy worker", heartbeat)
	}

	overviewReq := httptest.NewRequest(http.MethodGet, "/v2/admin/overview", nil)
	overviewRec := httptest.NewRecorder()
	router.ServeHTTP(overviewRec, overviewReq)
	if overviewRec.Code != http.StatusOK {
		t.Fatalf("admin overview status = %d body=%s", overviewRec.Code, overviewRec.Body.String())
	}
	var overview struct {
		Workers []adminWorkerRecord `json:"workers"`
	}
	if err := json.Unmarshal(overviewRec.Body.Bytes(), &overview); err != nil {
		t.Fatalf("decode admin overview: %v", err)
	}
	if len(overview.Workers) != 1 {
		t.Fatalf("workers = %+v, want one worker", overview.Workers)
	}
	worker := overview.Workers[0]
	if worker.WorkerID != "deepagents-worker-a" || worker.WorkerKind != "deepagents" || !worker.Active || worker.Stale {
		t.Fatalf("admin worker = %+v, want active deepagents worker", worker)
	}
	if worker.CurrentRunID == nil || *worker.CurrentRunID != "run_123" {
		t.Fatalf("admin worker current_run_id = %v, want run_123", worker.CurrentRunID)
	}
	if worker.HeartbeatAgeSeconds == nil || *worker.HeartbeatAgeSeconds > 5 {
		t.Fatalf("heartbeat age = %v, want fresh worker heartbeat", worker.HeartbeatAgeSeconds)
	}
}

func TestV2AdminSurfacesStaleRunningRunSignals(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "user-1",
		Title:  "stale admin",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "long autonomous run",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	if _, err := mem.UpdateRunStatus(ctx, run.RunID, domain.RunStatusRunning, "", ""); err != nil {
		t.Fatalf("UpdateRunStatus: %v", err)
	}
	if _, err := mem.AppendRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt_stale_tool",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "tool_call.started",
		TS:        domain.Now().Add(-24 * time.Minute),
		Payload:   domain.JSONMap{"tool_name": "execute"},
	}); err != nil {
		t.Fatalf("AppendRunEvent tool: %v", err)
	}
	if _, err := mem.AppendRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt_stale_delta",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "message.delta",
		TS:        domain.Now().Add(-23 * time.Minute),
		Payload:   domain.JSONMap{"delta": "working"},
	}); err != nil {
		t.Fatalf("AppendRunEvent delta: %v", err)
	}
	if _, err := mem.AppendRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt_stale_artifact",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "artifact.created",
		TS:        domain.Now().Add(-22 * time.Minute),
		Payload:   domain.JSONMap{"artifact_id": "artifact-1"},
	}); err != nil {
		t.Fatalf("AppendRunEvent artifact: %v", err)
	}
	if _, err := mem.AppendRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt_stale_heartbeat",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.heartbeat",
		TS:        domain.Now().Add(-20 * time.Minute),
		Payload:   domain.JSONMap{"stage": "silent_compute"},
	}); err != nil {
		t.Fatalf("AppendRunEvent: %v", err)
	}

	runsReq := httptest.NewRequest(http.MethodGet, "/v2/admin/runs?status=running", nil)
	runsRec := httptest.NewRecorder()
	router.ServeHTTP(runsRec, runsReq)
	if runsRec.Code != http.StatusOK {
		t.Fatalf("admin runs status = %d body=%s", runsRec.Code, runsRec.Body.String())
	}
	var runsPayload struct {
		Runs []struct {
			RunID                string   `json:"run_id"`
			Status               string   `json:"status"`
			Stale                bool     `json:"stale"`
			StaleReason          *string  `json:"stale_reason"`
			LastEventKind        *string  `json:"last_event_kind"`
			LastEventAt          *string  `json:"last_event_at"`
			LastEventSequence    *int64   `json:"last_event_sequence"`
			LastActivitySeconds  *float64 `json:"last_activity_age_seconds"`
			EventCount           int      `json:"event_count"`
			MessageDeltaCount    int      `json:"message_delta_count"`
			ToolCallCount        int      `json:"tool_call_count"`
			ArtifactCount        int      `json:"artifact_count"`
			HeartbeatCount       int      `json:"heartbeat_count"`
			LastToolName         *string  `json:"last_tool_name"`
			LastToolAt           *string  `json:"last_tool_at"`
			FirstDeltaSeconds    *float64 `json:"first_delta_latency_seconds"`
			FirstToolSeconds     *float64 `json:"first_tool_latency_seconds"`
			FirstArtifactSeconds *float64 `json:"first_artifact_latency_seconds"`
		} `json:"runs"`
	}
	if err := json.Unmarshal(runsRec.Body.Bytes(), &runsPayload); err != nil {
		t.Fatalf("decode admin runs: %v", err)
	}
	if len(runsPayload.Runs) != 1 {
		t.Fatalf("runs = %+v, want one stale running run", runsPayload.Runs)
	}
	record := runsPayload.Runs[0]
	if record.RunID != run.RunID || record.Status != "running" {
		t.Fatalf("run record = %+v, want running %s", record, run.RunID)
	}
	if !record.Stale || record.StaleReason == nil || !strings.Contains(*record.StaleReason, "No worker event") {
		t.Fatalf("stale fields = %+v, want stale worker-event reason", record)
	}
	if record.LastEventKind == nil || *record.LastEventKind != "run.heartbeat" {
		t.Fatalf("last_event_kind = %v, want run.heartbeat", record.LastEventKind)
	}
	if record.LastEventAt == nil || record.LastEventSequence == nil || *record.LastEventSequence < 1 || record.EventCount < 1 {
		t.Fatalf("event metadata = %+v, want latest event details", record)
	}
	if record.LastActivitySeconds == nil || *record.LastActivitySeconds < 600 {
		t.Fatalf("last_activity_age_seconds = %v, want stale age", record.LastActivitySeconds)
	}
	if record.MessageDeltaCount != 1 || record.ToolCallCount != 1 || record.ArtifactCount != 1 || record.HeartbeatCount != 1 {
		t.Fatalf("event counts = %+v, want one delta/tool/artifact/heartbeat", record)
	}
	if record.LastToolName == nil || *record.LastToolName != "execute" || record.LastToolAt == nil {
		t.Fatalf("last tool metadata = %+v, want execute", record)
	}
	if record.FirstDeltaSeconds == nil || record.FirstToolSeconds == nil || record.FirstArtifactSeconds == nil {
		t.Fatalf("first event latencies = %+v, want latency diagnostics", record)
	}

	issuesReq := httptest.NewRequest(http.MethodGet, "/v2/admin/issues", nil)
	issuesRec := httptest.NewRecorder()
	router.ServeHTTP(issuesRec, issuesReq)
	if issuesRec.Code != http.StatusOK {
		t.Fatalf("admin issues status = %d body=%s", issuesRec.Code, issuesRec.Body.String())
	}
	var issuesPayload struct {
		Issues []adminIssueRecord `json:"issues"`
	}
	if err := json.Unmarshal(issuesRec.Body.Bytes(), &issuesPayload); err != nil {
		t.Fatalf("decode admin issues: %v", err)
	}
	if len(issuesPayload.Issues) != 1 {
		t.Fatalf("issues = %+v, want one stalled_run issue", issuesPayload.Issues)
	}
	issue := issuesPayload.Issues[0]
	if issue.IssueType != "stalled_run" || issue.RunID != run.RunID || issue.Severity != "high" {
		t.Fatalf("issue = %+v, want high stalled_run for %s", issue, run.RunID)
	}
	if issue.Metadata["last_event_kind"] != "run.heartbeat" {
		t.Fatalf("issue metadata = %+v, want last_event_kind", issue.Metadata)
	}
}

func TestV2AdminRunsIncludeActiveRunLeaseOwnership(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "user-lease",
		Title:  "lease admin",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-lease",
		Goal:     "long autonomous run with a lease",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	lease, err := mem.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    run.RunID,
		WorkerID: "deepagents-worker-a",
		TTL:      10 * time.Minute,
		Now:      domain.Now(),
	})
	if err != nil {
		t.Fatalf("AcquireRunLease: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/admin/runs?status=running", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("admin runs status = %d body=%s", rec.Code, rec.Body.String())
	}
	var payload struct {
		Runs []struct {
			RunID                      string   `json:"run_id"`
			LeaseWorkerID              *string  `json:"lease_worker_id"`
			LeaseExpiresAt             *string  `json:"lease_expires_at"`
			LeaseActive                bool     `json:"lease_active"`
			LeaseExpired               bool     `json:"lease_expired"`
			LeaseSecondsRemaining      *float64 `json:"lease_seconds_remaining"`
			LeaseLastRenewedAt         *string  `json:"lease_last_renewed_at"`
			LeaseLastRenewedAgeSeconds *float64 `json:"lease_last_renewed_age_seconds"`
		} `json:"runs"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode admin runs: %v", err)
	}
	if len(payload.Runs) != 1 {
		t.Fatalf("runs = %+v, want one running run", payload.Runs)
	}
	record := payload.Runs[0]
	if record.RunID != run.RunID {
		t.Fatalf("run_id = %q, want %q", record.RunID, run.RunID)
	}
	if record.LeaseWorkerID == nil || *record.LeaseWorkerID != "deepagents-worker-a" {
		t.Fatalf("lease_worker_id = %v, want worker owner", record.LeaseWorkerID)
	}
	if record.LeaseExpiresAt == nil || *record.LeaseExpiresAt != lease.LeaseExpiresAt.UTC().Format(time.RFC3339Nano) {
		t.Fatalf("lease_expires_at = %v, want %s", record.LeaseExpiresAt, lease.LeaseExpiresAt.UTC().Format(time.RFC3339Nano))
	}
	if !record.LeaseActive || record.LeaseExpired {
		t.Fatalf("lease active/expired = %t/%t, want active non-expired", record.LeaseActive, record.LeaseExpired)
	}
	if record.LeaseSecondsRemaining == nil || *record.LeaseSecondsRemaining <= 0 {
		t.Fatalf("lease_seconds_remaining = %v, want positive", record.LeaseSecondsRemaining)
	}
	if record.LeaseLastRenewedAt == nil || record.LeaseLastRenewedAgeSeconds == nil {
		t.Fatalf("lease renewal fields missing: %+v", record)
	}
}

func TestV2AdminRunsFlagExpiredRunLeaseAsStale(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "user-lease",
		Title:  "expired lease admin",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-lease",
		Goal:     "long autonomous run with an expired lease",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	_, err = mem.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    run.RunID,
		WorkerID: "deepagents-worker-expired",
		TTL:      time.Minute,
		Now:      domain.Now().Add(-10 * time.Minute),
	})
	if err != nil {
		t.Fatalf("AcquireRunLease: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/admin/runs?status=running", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("admin runs status = %d body=%s", rec.Code, rec.Body.String())
	}
	var payload struct {
		Runs []struct {
			RunID                 string   `json:"run_id"`
			Stale                 bool     `json:"stale"`
			StaleReason           *string  `json:"stale_reason"`
			LeaseWorkerID         *string  `json:"lease_worker_id"`
			LeaseActive           bool     `json:"lease_active"`
			LeaseExpired          bool     `json:"lease_expired"`
			LeaseSecondsRemaining *float64 `json:"lease_seconds_remaining"`
		} `json:"runs"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode admin runs: %v", err)
	}
	if len(payload.Runs) != 1 {
		t.Fatalf("runs = %+v, want one running run", payload.Runs)
	}
	record := payload.Runs[0]
	if record.RunID != run.RunID {
		t.Fatalf("run_id = %q, want %q", record.RunID, run.RunID)
	}
	if record.LeaseWorkerID == nil || *record.LeaseWorkerID != "deepagents-worker-expired" {
		t.Fatalf("lease_worker_id = %v, want expired worker owner", record.LeaseWorkerID)
	}
	if record.LeaseActive || !record.LeaseExpired {
		t.Fatalf("lease active/expired = %t/%t, want expired inactive", record.LeaseActive, record.LeaseExpired)
	}
	if record.LeaseSecondsRemaining == nil || *record.LeaseSecondsRemaining != 0 {
		t.Fatalf("lease_seconds_remaining = %v, want zero for expired lease", record.LeaseSecondsRemaining)
	}
	if !record.Stale || record.StaleReason == nil || !strings.Contains(*record.StaleReason, "lease expired") {
		t.Fatalf("stale fields = %+v, want expired lease reason", record)
	}
}

func TestV2UploadAndResourceHandlers(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       service,
		Store:      mem,
		UploadRoot: uploadRoot,
	})

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("files", "prairie.jpg")
	if err != nil {
		t.Fatalf("CreateFormFile: %v", err)
	}
	pngBytes := testPNGBytes(t, 3, 2)
	if _, err := part.Write(pngBytes); err != nil {
		t.Fatalf("write multipart file: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("close multipart writer: %v", err)
	}

	uploadReq := httptest.NewRequest(http.MethodPost, "/v2/uploads", &body)
	uploadReq.Header.Set("Content-Type", writer.FormDataContentType())
	uploadReq.Header.Set("X-Ultra-User-Id", "field-researcher")
	uploadReq.Header.Set("X-Ultra-Org-Id", "smithsonian")
	uploadReq.Header.Set("X-Ultra-Role", "admin")
	uploadRec := httptest.NewRecorder()
	router.ServeHTTP(uploadRec, uploadReq)
	if uploadRec.Code != http.StatusOK {
		t.Fatalf("upload status = %d body=%s", uploadRec.Code, uploadRec.Body.String())
	}
	var uploadResponse struct {
		FileCount int `json:"file_count"`
		Uploaded  []struct {
			FileID       string `json:"file_id"`
			OriginalName string `json:"original_name"`
			ContentType  string `json:"content_type"`
			SizeBytes    int64  `json:"size_bytes"`
			SHA256       string `json:"sha256"`
			PreviewURL   string `json:"preview_url"`
			Principal    struct {
				UserID string `json:"user_id"`
				OrgID  string `json:"org_id"`
				Role   string `json:"role"`
			} `json:"principal"`
		} `json:"uploaded"`
	}
	if err := json.Unmarshal(uploadRec.Body.Bytes(), &uploadResponse); err != nil {
		t.Fatalf("decode upload response: %v", err)
	}
	if uploadResponse.FileCount != 1 || len(uploadResponse.Uploaded) != 1 {
		t.Fatalf("upload response = %+v, want one uploaded file", uploadResponse)
	}
	uploaded := uploadResponse.Uploaded[0]
	if uploaded.FileID == "" || uploaded.OriginalName != "prairie.jpg" || uploaded.SHA256 == "" {
		t.Fatalf("uploaded metadata = %+v, want id/name/checksum", uploaded)
	}
	if uploaded.Principal.UserID != "field-researcher" || uploaded.Principal.OrgID != "smithsonian" || uploaded.Principal.Role != "admin" {
		t.Fatalf("uploaded principal = %+v, want request principal", uploaded.Principal)
	}

	matches, err := filepath.Glob(filepath.Join(uploadRoot, uploaded.FileID+"__*"))
	if err != nil {
		t.Fatalf("glob uploaded file: %v", err)
	}
	if len(matches) != 1 {
		t.Fatalf("uploaded files under root = %v, want one match for file id", matches)
	}

	listReq := httptest.NewRequest(http.MethodGet, "/v2/resources?limit=20&kind=image", nil)
	listRec := httptest.NewRecorder()
	router.ServeHTTP(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("list resources status = %d body=%s", listRec.Code, listRec.Body.String())
	}
	var listResponse struct {
		Count     int `json:"count"`
		Resources []struct {
			FileID       string `json:"file_id"`
			OriginalName string `json:"original_name"`
			SourceType   string `json:"source_type"`
			ResourceKind string `json:"resource_kind"`
			PreviewURL   string `json:"preview_url"`
			Principal    struct {
				UserID string `json:"user_id"`
				OrgID  string `json:"org_id"`
				Role   string `json:"role"`
			} `json:"principal"`
		} `json:"resources"`
	}
	if err := json.Unmarshal(listRec.Body.Bytes(), &listResponse); err != nil {
		t.Fatalf("decode resources response: %v", err)
	}
	if listResponse.Count != 1 || listResponse.Resources[0].FileID != uploaded.FileID {
		t.Fatalf("resources = %+v, want uploaded resource", listResponse)
	}
	if listResponse.Resources[0].SourceType != "upload" || listResponse.Resources[0].ResourceKind != "image" {
		t.Fatalf("resource classification = %+v, want uploaded image", listResponse.Resources[0])
	}
	if listResponse.Resources[0].Principal.UserID != "field-researcher" || listResponse.Resources[0].Principal.OrgID != "smithsonian" || listResponse.Resources[0].Principal.Role != "admin" {
		t.Fatalf("resource principal = %+v, want upload principal", listResponse.Resources[0].Principal)
	}

	displayReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+uploaded.FileID+"/display", nil)
	displayRec := httptest.NewRecorder()
	router.ServeHTTP(displayRec, displayReq)
	if displayRec.Code != http.StatusOK {
		t.Fatalf("display status = %d body=%s", displayRec.Code, displayRec.Body.String())
	}
	if !bytes.Equal(displayRec.Body.Bytes(), pngBytes) {
		t.Fatalf("display body = %q, want uploaded PNG bytes", displayRec.Body.String())
	}

	sliceReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+uploaded.FileID+"/slice?axis=z&z=0", nil)
	sliceRec := httptest.NewRecorder()
	router.ServeHTTP(sliceRec, sliceReq)
	if sliceRec.Code != http.StatusOK {
		t.Fatalf("slice status = %d body=%s", sliceRec.Code, sliceRec.Body.String())
	}
	if !bytes.Equal(sliceRec.Body.Bytes(), pngBytes) {
		t.Fatalf("slice body = %q, want uploaded PNG bytes", sliceRec.Body.String())
	}

	viewerReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+uploaded.FileID+"/viewer", nil)
	viewerRec := httptest.NewRecorder()
	router.ServeHTTP(viewerRec, viewerReq)
	if viewerRec.Code != http.StatusOK {
		t.Fatalf("viewer status = %d body=%s", viewerRec.Code, viewerRec.Body.String())
	}
	var viewerResponse struct {
		Kind         string `json:"kind"`
		FileID       string `json:"file_id"`
		OriginalName string `json:"original_name"`
		AxisSizes    struct {
			X int `json:"X"`
			Y int `json:"Y"`
			Z int `json:"Z"`
			C int `json:"C"`
			T int `json:"T"`
		} `json:"axis_sizes"`
		ServiceURLs struct {
			Display string `json:"display"`
			Preview string `json:"preview"`
		} `json:"service_urls"`
	}
	if err := json.Unmarshal(viewerRec.Body.Bytes(), &viewerResponse); err != nil {
		t.Fatalf("decode viewer response: %v", err)
	}
	if viewerResponse.Kind != "image" || viewerResponse.FileID != uploaded.FileID || viewerResponse.OriginalName != "prairie.jpg" {
		t.Fatalf("viewer identity = %+v, want uploaded image metadata", viewerResponse)
	}
	if viewerResponse.AxisSizes.X != 3 || viewerResponse.AxisSizes.Y != 2 || viewerResponse.AxisSizes.Z != 1 || viewerResponse.AxisSizes.T != 1 {
		t.Fatalf("viewer axis sizes = %+v, want 3x2 image", viewerResponse.AxisSizes)
	}
	if viewerResponse.ServiceURLs.Display != "/v2/uploads/"+uploaded.FileID+"/display" {
		t.Fatalf("viewer display URL = %q, want V2 display URL", viewerResponse.ServiceURLs.Display)
	}

	captionReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+uploaded.FileID+"/caption", nil)
	captionRec := httptest.NewRecorder()
	router.ServeHTTP(captionRec, captionReq)
	if captionRec.Code != http.StatusOK {
		t.Fatalf("caption status = %d body=%s", captionRec.Code, captionRec.Body.String())
	}
	var captionResponse struct {
		FileID  string `json:"file_id"`
		Caption string `json:"caption"`
		Source  string `json:"source"`
	}
	if err := json.Unmarshal(captionRec.Body.Bytes(), &captionResponse); err != nil {
		t.Fatalf("decode caption response: %v", err)
	}
	if captionResponse.FileID != uploaded.FileID || !strings.Contains(captionResponse.Caption, "prairie.jpg") || captionResponse.Source != "fallback" {
		t.Fatalf("caption response = %+v, want fallback caption for uploaded image", captionResponse)
	}
}

func TestV2Sam3InteractiveSegmentationIsExplicitlyNotConfigured(t *testing.T) {
	t.Parallel()

	router := NewRouter(ServerDeps{
		Version: "test-version",
		Runs:    runcontrol.NewService(store.NewMemoryStore(), eventbus.NewMemoryBus()),
		Store:   store.NewMemoryStore(),
	})

	req := httptest.NewRequest(http.MethodPost, "/v2/segment/sam3/interactive", strings.NewReader(`{"file_ids":["file_1"],"annotations":[]}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("SAM3 status = %d body=%s, want 501 not configured", rec.Code, rec.Body.String())
	}
	var response map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode SAM3 response: %v", err)
	}
	if response["status"] != "not_configured" || response["service"] != "ultra-control-v2" {
		t.Fatalf("SAM3 response = %#v, want explicit V2 not-configured payload", response)
	}
}

func TestV2BisqueImportIsExplicitlyNotConfigured(t *testing.T) {
	t.Parallel()

	router := NewRouter(ServerDeps{
		Version: "test-version",
		Runs:    runcontrol.NewService(store.NewMemoryStore(), eventbus.NewMemoryBus()),
		Store:   store.NewMemoryStore(),
	})

	req := httptest.NewRequest(http.MethodPost, "/v2/uploads/from-bisque", strings.NewReader(`{"resources":["https://bisque.example.org/data_service/image/1"]}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("BisQue import status = %d body=%s, want 501 not configured", rec.Code, rec.Body.String())
	}
	var response map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode BisQue import response: %v", err)
	}
	if response["status"] != "not_configured" || response["service"] != "ultra-control-v2" {
		t.Fatalf("BisQue import response = %#v, want explicit V2 not-configured payload", response)
	}
}

func testPNGBytes(t *testing.T, width int, height int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.Set(x, y, color.RGBA{R: uint8(40 * x), G: uint8(60 * y), B: 120, A: 255})
		}
	}
	var buffer bytes.Buffer
	if err := png.Encode(&buffer, img); err != nil {
		t.Fatalf("encode test PNG: %v", err)
	}
	return buffer.Bytes()
}

func TestCreateRunReusesIdempotencyKeyFromHeader(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	thread, err := service.CreateThread(context.Background(), runcontrol.CreateThreadRequest{
		UserID: "user-1",
		Title:  "idempotency",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}

	body := `{"user_id":"user-1","goal":"hello","messages":[{"role":"user","content":"hello"}]}`
	firstReq := httptest.NewRequest(http.MethodPost, "/v2/threads/"+thread.ThreadID+"/runs", strings.NewReader(body))
	firstReq.Header.Set("Content-Type", "application/json")
	firstReq.Header.Set("Idempotency-Key", "prompt-key-http")
	firstRec := httptest.NewRecorder()
	router.ServeHTTP(firstRec, firstReq)
	if firstRec.Code != http.StatusOK {
		t.Fatalf("first create run status = %d body=%s", firstRec.Code, firstRec.Body.String())
	}

	secondReq := httptest.NewRequest(http.MethodPost, "/v2/threads/"+thread.ThreadID+"/runs", strings.NewReader(body))
	secondReq.Header.Set("Content-Type", "application/json")
	secondReq.Header.Set("Idempotency-Key", "prompt-key-http")
	secondRec := httptest.NewRecorder()
	router.ServeHTTP(secondRec, secondReq)
	if secondRec.Code != http.StatusOK {
		t.Fatalf("second create run status = %d body=%s", secondRec.Code, secondRec.Body.String())
	}

	var firstRun domain.RunRecord
	if err := json.Unmarshal(firstRec.Body.Bytes(), &firstRun); err != nil {
		t.Fatalf("decode first run: %v", err)
	}
	var secondRun domain.RunRecord
	if err := json.Unmarshal(secondRec.Body.Bytes(), &secondRun); err != nil {
		t.Fatalf("decode second run: %v", err)
	}
	if secondRun.RunID != firstRun.RunID {
		t.Fatalf("second run id = %q, want original %q", secondRun.RunID, firstRun.RunID)
	}

	events, err := mem.ListRunEvents(context.Background(), firstRun.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("events = %d, want exactly one accepted event", len(events))
	}
	select {
	case <-bus.Jobs():
	case <-time.After(time.Second):
		t.Fatalf("expected first job")
	}
	select {
	case job := <-bus.Jobs():
		t.Fatalf("unexpected duplicate job: %+v", job)
	default:
	}
}

func TestListRunEventsSupportsAfterSequenceCursor(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "user-1",
		Title:  "long trace",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "run a long trace",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "run a long trace"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	for idx := 0; idx < 5; idx++ {
		if _, err := mem.AppendRunEvent(ctx, domain.AppendRunEventInput{
			RunID:     run.RunID,
			ThreadID:  thread.ThreadID,
			EventKind: "message.delta",
			Message:   "chunk",
			Payload:   domain.JSONMap{"idx": idx},
		}); err != nil {
			t.Fatalf("AppendRunEvent %d: %v", idx, err)
		}
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/runs/"+run.RunID+"/events?limit=2&after_sequence=3", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("events status = %d body=%s", rec.Code, rec.Body.String())
	}

	var response runEventsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode events: %v", err)
	}
	if response.Count != 2 {
		t.Fatalf("count = %d, want 2", response.Count)
	}
	got := []int64{response.Events[0].Sequence, response.Events[1].Sequence}
	want := []int64{4, 5}
	if got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("sequences = %v, want %v", got, want)
	}
}

func TestCancelRunPublishesCanceledEventAndCancelSignal(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem, Bus: bus})

	thread, err := service.CreateThread(context.Background(), runcontrol.CreateThreadRequest{
		UserID: "user-1",
		Title:  "cancel",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(context.Background(), runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "long run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "long run"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	drainRunEvents(bus)

	req := httptest.NewRequest(http.MethodPost, "/v2/runs/"+run.RunID+"/cancel", strings.NewReader(`{"reason":"user requested"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("cancel status = %d body=%s", rec.Code, rec.Body.String())
	}

	select {
	case event := <-bus.Events():
		if event.EventKind != "run.canceled" || event.RunID != run.RunID {
			t.Fatalf("event = %+v, want run.canceled for run", event)
		}
	case <-time.After(time.Second):
		t.Fatalf("expected canceled event fanout")
	}
	select {
	case cancel := <-bus.Cancellations():
		if cancel.RunID != run.RunID || cancel.Reason != "user requested" {
			t.Fatalf("cancel signal = %+v, want run/reason", cancel)
		}
	case <-time.After(time.Second):
		t.Fatalf("expected cancel signal")
	}
}

func TestAdminRequeueRunPublishesRecoveryJob(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem, Bus: bus})

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "user-1",
		Title:  "requeue",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "long run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "long run"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	drainJobs(bus)
	drainRunEvents(bus)
	if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt_" + run.RunID + "_started",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.started",
	}); err != nil {
		t.Fatalf("IngestRunEvent started: %v", err)
	}
	drainRunEvents(bus)

	req := httptest.NewRequest(http.MethodPost, "/v2/admin/runs/"+run.RunID+"/requeue", strings.NewReader(`{"reason":"expired lease"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("admin requeue status = %d body=%s", rec.Code, rec.Body.String())
	}
	var response adminRunActionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode requeue response: %v", err)
	}
	if response.RunID != run.RunID || response.Status != string(domain.RunStatusRunning) || !response.Updated {
		t.Fatalf("requeue response = %+v, want running updated response", response)
	}
	select {
	case job := <-bus.Jobs():
		if job.RunID != run.RunID || job.DispatchID == "" {
			t.Fatalf("job = %+v, want original run with fresh dispatch id", job)
		}
		if got := job.Metadata["requeue_reason"]; got != "expired lease" {
			t.Fatalf("job requeue reason = %#v, want expired lease", got)
		}
	case <-time.After(time.Second):
		t.Fatalf("expected requeued job")
	}
	select {
	case event := <-bus.Events():
		if event.EventKind != "run.requeued" || event.RunID != run.RunID {
			t.Fatalf("event = %+v, want run.requeued", event)
		}
	case <-time.After(time.Second):
		t.Fatalf("expected run.requeued fanout")
	}
}

func TestArtifactDownloadServesFilesUnderArtifactRootAndRejectsTraversal(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	artifactRoot := t.TempDir()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{
		Version:      "test-version",
		Runs:         service,
		Store:        mem,
		ArtifactRoot: artifactRoot,
	})

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{UserID: "user-1", Title: "artifacts"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "artifact",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	reportPath := filepath.Join(artifactRoot, run.RunID, "report.md")
	if err := os.MkdirAll(filepath.Dir(reportPath), 0o755); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	if err := os.WriteFile(reportPath, []byte("# RareSpot report\n"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	artifact, err := mem.CreateArtifact(ctx, domain.CreateArtifactInput{
		RunID:    run.RunID,
		ThreadID: thread.ThreadID,
		Kind:     "report",
		Path:     "report.md",
		MimeType: "text/markdown",
		Title:    "RareSpot report",
	})
	if err != nil {
		t.Fatalf("CreateArtifact report: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/artifacts/"+artifact.ArtifactID+"/download", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("download status = %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "RareSpot report") {
		t.Fatalf("download body = %q, want report content", rec.Body.String())
	}
	if got := rec.Header().Get("Content-Type"); !strings.Contains(got, "text/markdown") {
		t.Fatalf("content type = %q, want markdown", got)
	}

	pathReq := httptest.NewRequest(http.MethodGet, "/v2/runs/"+run.RunID+"/artifacts/download?path=report.md", nil)
	pathRec := httptest.NewRecorder()
	router.ServeHTTP(pathRec, pathReq)
	if pathRec.Code != http.StatusOK {
		t.Fatalf("path download status = %d body=%s", pathRec.Code, pathRec.Body.String())
	}
	if !strings.Contains(pathRec.Body.String(), "RareSpot report") {
		t.Fatalf("path download body = %q, want report content", pathRec.Body.String())
	}

	traversal, err := mem.CreateArtifact(ctx, domain.CreateArtifactInput{
		RunID:    run.RunID,
		ThreadID: thread.ThreadID,
		Kind:     "report",
		Path:     "../secret.md",
		MimeType: "text/markdown",
	})
	if err != nil {
		t.Fatalf("CreateArtifact traversal: %v", err)
	}
	traversalReq := httptest.NewRequest(http.MethodGet, "/v2/artifacts/"+traversal.ArtifactID+"/download", nil)
	traversalRec := httptest.NewRecorder()
	router.ServeHTTP(traversalRec, traversalReq)
	if traversalRec.Code != http.StatusBadRequest {
		t.Fatalf("traversal status = %d body=%s, want 400", traversalRec.Code, traversalRec.Body.String())
	}
}

func jsonArrayEquals(value any, want []string) bool {
	values, ok := value.([]any)
	if !ok || len(values) != len(want) {
		return false
	}
	for index, item := range values {
		if item != want[index] {
			return false
		}
	}
	return true
}

func drainRunEvents(bus *eventbus.MemoryBus) {
	for {
		select {
		case <-bus.Events():
		default:
			return
		}
	}
}

func drainJobs(bus *eventbus.MemoryBus) {
	for {
		select {
		case <-bus.Jobs():
		default:
			return
		}
	}
}
