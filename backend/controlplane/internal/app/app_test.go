package app

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/config"
)

func TestNewAppServesHealth(t *testing.T) {
	t.Parallel()
	application, err := New(config.Config{AppVersion: "test-version"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	req := httptest.NewRequest(http.MethodGet, "/v1/health", nil)
	rec := httptest.NewRecorder()
	application.Handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
}
