package httpapi

import (
	"context"
	"encoding/json"
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

func TestCreateRunProtectsMaterialsCleanroomEvaluationProfile(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{UserID: "evaluator", Title: "profile auth"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem, Bus: bus})

	invalid := createEvaluationProfileRun(t, router, thread.ThreadID, "admin", map[string]any{
		"evaluation_profile": "unknown_profile",
	})
	if invalid.Code != http.StatusBadRequest {
		t.Fatalf("unknown profile status = %d body=%s", invalid.Code, invalid.Body.String())
	}
	forbidden := createEvaluationProfileRun(t, router, thread.ThreadID, "researcher", map[string]any{
		"evaluation_profile": string(domain.EvaluationProfileMaterialsCleanroomV1),
	})
	if forbidden.Code != http.StatusForbidden {
		t.Fatalf("researcher profile status = %d body=%s", forbidden.Code, forbidden.Body.String())
	}
	metadataForgery := createEvaluationProfileRun(t, router, thread.ThreadID, "researcher", map[string]any{
		"metadata": map[string]any{
			domain.EvaluationProfileMetadataKey: string(domain.EvaluationProfileMaterialsCleanroomV1),
		},
	})
	if metadataForgery.Code != http.StatusOK {
		t.Fatalf("metadata-only run status = %d body=%s", metadataForgery.Code, metadataForgery.Body.String())
	}
	var ordinaryRun domain.RunRecord
	if err := json.Unmarshal(metadataForgery.Body.Bytes(), &ordinaryRun); err != nil {
		t.Fatalf("decode metadata-only run: %v", err)
	}
	if _, found := ordinaryRun.Metadata[domain.EvaluationProfileMetadataKey]; found {
		t.Fatalf("metadata-only request granted profile: %#v", ordinaryRun.Metadata)
	}
	if job := receiveHTTPProfileJob(t, bus); job.EvaluationProfile != "" {
		t.Fatalf("metadata-only job profile = %q", job.EvaluationProfile)
	}

	authorized := createEvaluationProfileRun(t, router, thread.ThreadID, "ADMIN", map[string]any{
		"evaluation_profile": string(domain.EvaluationProfileMaterialsCleanroomV1),
		"metadata": map[string]any{
			domain.EvaluationProfileMetadataKey: "caller_forgery",
		},
	})
	if authorized.Code != http.StatusOK {
		t.Fatalf("admin profile status = %d body=%s", authorized.Code, authorized.Body.String())
	}
	var protectedRun domain.RunRecord
	if err := json.Unmarshal(authorized.Body.Bytes(), &protectedRun); err != nil {
		t.Fatalf("decode protected run: %v", err)
	}
	if got := protectedRun.Metadata[domain.EvaluationProfileMetadataKey]; got != string(domain.EvaluationProfileMaterialsCleanroomV1) {
		t.Fatalf("protected run profile = %#v", got)
	}
	if job := receiveHTTPProfileJob(t, bus); job.EvaluationProfile != domain.EvaluationProfileMaterialsCleanroomV1 {
		t.Fatalf("protected job profile = %q", job.EvaluationProfile)
	}
}

func createEvaluationProfileRun(t *testing.T, router http.Handler, threadID string, role string, overrides map[string]any) *httptest.ResponseRecorder {
	t.Helper()
	body := map[string]any{
		"goal":     "evaluate materials analysis",
		"messages": []map[string]string{{"role": "user", "content": "evaluate materials analysis"}},
	}
	for key, value := range overrides {
		body[key] = value
	}
	payload, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v2/threads/"+threadID+"/runs", strings.NewReader(string(payload)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Ultra-User-Id", "evaluator")
	req.Header.Set("X-Ultra-Role", role)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	return rec
}

func receiveHTTPProfileJob(t *testing.T, bus *eventbus.MemoryBus) eventbus.Job {
	t.Helper()
	select {
	case job := <-bus.Jobs():
		return job
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for run job")
		return eventbus.Job{}
	}
}
