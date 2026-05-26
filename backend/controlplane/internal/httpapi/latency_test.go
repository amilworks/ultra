package httpapi

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/latency"
)

func TestHealthLatencyBudgetInMemory(t *testing.T) {
	t.Parallel()
	router := NewRouter(ServerDeps{Version: "test-version"})
	req := httptest.NewRequest(http.MethodGet, "/v1/health", nil)
	start := time.Now()
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	elapsed := time.Since(start)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	if elapsed > latency.HealthConfigP95 {
		t.Fatalf("health elapsed = %s, budget = %s", elapsed, latency.HealthConfigP95)
	}
}
