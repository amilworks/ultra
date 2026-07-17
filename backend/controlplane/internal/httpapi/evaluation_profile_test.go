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

// No evaluation profile is currently supported, so every non-empty
// evaluation_profile is rejected at the wire boundary, and free-form metadata
// can never grant one. The admin gate and profile propagation in the handler
// stay as the guards that re-arm if a profile is reintroduced.
func TestCreateRunRejectsEveryEvaluationProfile(t *testing.T) {
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

	// "materials_cleanroom_v1" was the only profile before the materials platform
	// was removed; an admin principal must not be able to resurrect it.
	for _, profile := range []string{"unknown_profile", "materials_cleanroom_v1"} {
		for _, role := range []string{"researcher", "ADMIN"} {
			rec := createEvaluationProfileRun(t, router, thread.ThreadID, role, map[string]any{
				"evaluation_profile": profile,
			})
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("profile %q as %q status = %d body=%s, want 400",
					profile, role, rec.Code, rec.Body.String())
			}
		}
	}

	metadataForgery := createEvaluationProfileRun(t, router, thread.ThreadID, "researcher", map[string]any{
		"metadata": map[string]any{
			domain.EvaluationProfileMetadataKey: "materials_cleanroom_v1",
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
}

func createEvaluationProfileRun(t *testing.T, router http.Handler, threadID string, role string, overrides map[string]any) *httptest.ResponseRecorder {
	t.Helper()
	body := map[string]any{
		"goal":     "evaluate scientific analysis",
		"messages": []map[string]string{{"role": "user", "content": "evaluate scientific analysis"}},
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
