package httpapi

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
)

type ServerDeps struct {
	Version string
}

func NewRouter(deps ServerDeps) http.Handler {
	r := chi.NewRouter()
	r.Get("/v1/health", handleHealth)
	r.Get("/v1/config/public", handlePublicConfig(deps))
	return r
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{
		"status": "ok",
		"ts":     time.Now().UTC().Format(time.RFC3339Nano),
	})
}

func handlePublicConfig(deps ServerDeps) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{
			"app_name":    "BisQue Ultra",
			"app_version": deps.Version,
			"features": map[string]bool{
				"v2_runs": true,
			},
		})
	}
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
