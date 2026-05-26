package httpapi

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
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
