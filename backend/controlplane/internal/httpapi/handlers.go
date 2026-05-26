package httpapi

import (
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
	"github.com/go-chi/chi/v5"
)

type ServerDeps struct {
	Version string
	Runs    *runcontrol.Service
	Store   runcontrol.Store
	Bus     runEventSource
}

type runEventSource interface {
	Events() <-chan domain.RunEventRecord
}

func NewRouter(deps ServerDeps) http.Handler {
	r := chi.NewRouter()
	r.Get("/v1/health", handleHealth)
	r.Get("/v1/config/public", handlePublicConfig(deps))
	r.Get("/v1/auth/session", handleAuthSession)
	r.Route("/v2", func(r chi.Router) {
		r.Get("/threads", deps.handleListThreads)
		r.Post("/threads", deps.handleCreateThread)
		r.Get("/threads/{thread_id}", deps.handleGetThread)
		r.Get("/threads/{thread_id}/messages", deps.handleListThreadMessages)
		r.Post("/threads/{thread_id}/runs", deps.handleCreateRun)
		r.Get("/runs/{run_id}", deps.handleGetRun)
		r.Post("/runs/{run_id}/cancel", deps.handleCancelRun)
		r.Get("/runs/{run_id}/events", deps.handleListRunEvents)
		r.Get("/runs/{run_id}/artifacts", deps.handleListRunArtifacts)
		r.Get("/artifacts/{artifact_id}", deps.handleGetArtifact)
	})
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

func handleAuthSession(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"authenticated": false,
		"user":          nil,
	})
}

type createThreadRequest struct {
	UserID          string                 `json:"user_id"`
	Title           string                 `json:"title"`
	Metadata        map[string]any         `json:"metadata"`
	InitialMessages []domain.ThreadMessage `json:"initial_messages"`
}

type createRunRequest struct {
	UserID   string                 `json:"user_id"`
	Goal     string                 `json:"goal"`
	Messages []domain.ThreadMessage `json:"messages"`
	Metadata map[string]any         `json:"metadata"`
}

type cancelRunRequest struct {
	Reason   string         `json:"reason"`
	Metadata map[string]any `json:"metadata"`
}

type listThreadsResponse struct {
	Count   int                   `json:"count"`
	Threads []domain.ThreadRecord `json:"threads"`
}

type threadMessagesResponse struct {
	ThreadID string                 `json:"thread_id"`
	Count    int                    `json:"count"`
	Messages []domain.ThreadMessage `json:"messages"`
}

type runEventsResponse struct {
	RunID  string                  `json:"run_id"`
	Count  int                     `json:"count"`
	Events []domain.RunEventRecord `json:"events"`
}

type runArtifactsResponse struct {
	RunID     string                  `json:"run_id"`
	Count     int                     `json:"count"`
	Artifacts []domain.ArtifactRecord `json:"artifacts"`
}

type artifactResponse struct {
	Artifact domain.ArtifactRecord `json:"artifact"`
}

func (deps ServerDeps) handleListThreads(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	threads, err := deps.Store.ListThreads(r.Context(), parseLimit(r, 100))
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, listThreadsResponse{Count: len(threads), Threads: threads})
}

func (deps ServerDeps) handleCreateThread(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	var req createThreadRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	thread, err := deps.Runs.CreateThread(r.Context(), runcontrol.CreateThreadRequest{
		UserID:          req.UserID,
		Title:           req.Title,
		Metadata:        domain.JSONMap(req.Metadata),
		InitialMessages: req.InitialMessages,
	})
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, thread)
}

func (deps ServerDeps) handleGetThread(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	thread, err := deps.Store.GetThread(r.Context(), chi.URLParam(r, "thread_id"))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, thread)
}

func (deps ServerDeps) handleListThreadMessages(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	threadID := chi.URLParam(r, "thread_id")
	messages, err := deps.Store.ListThreadMessages(r.Context(), threadID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, threadMessagesResponse{ThreadID: threadID, Count: len(messages), Messages: messages})
}

func (deps ServerDeps) handleCreateRun(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	var req createRunRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	run, err := deps.Runs.CreateRun(r.Context(), runcontrol.CreateRunRequest{
		ThreadID: chi.URLParam(r, "thread_id"),
		UserID:   req.UserID,
		Goal:     req.Goal,
		Messages: req.Messages,
		Metadata: domain.JSONMap(req.Metadata),
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, run)
}

func (deps ServerDeps) handleGetRun(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	run, err := deps.Store.GetRun(r.Context(), chi.URLParam(r, "run_id"))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, run)
}

func (deps ServerDeps) handleCancelRun(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	var req cancelRunRequest
	if r.Body != nil && r.ContentLength != 0 {
		if !decodeJSON(w, r, &req) {
			return
		}
	}
	runID := chi.URLParam(r, "run_id")
	run, err := deps.Store.UpdateRunStatus(r.Context(), runID, domain.RunStatusCanceled, "", req.Reason)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	_, err = deps.Store.AppendRunEvent(r.Context(), domain.AppendRunEventInput{
		RunID:     run.RunID,
		ThreadID:  run.ThreadID,
		EventKind: "run.canceled",
		Message:   "Run canceled.",
		Payload:   domain.JSONMap{"reason": req.Reason},
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, run)
}

func (deps ServerDeps) handleListRunEvents(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	runID := chi.URLParam(r, "run_id")
	events, err := deps.Store.ListRunEvents(r.Context(), runID, parseLimit(r, 500))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if r.URL.Query().Get("stream") == "true" {
		deps.streamRunEvents(w, r, runID, events)
		return
	}
	writeJSON(w, http.StatusOK, runEventsResponse{RunID: runID, Count: len(events), Events: events})
}

func (deps ServerDeps) streamRunEvents(w http.ResponseWriter, r *http.Request, runID string, replay []domain.RunEventRecord) {
	if deps.Bus == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "run event stream is not configured"})
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	for _, event := range replay {
		if err := writeSSE(w, "run_event", event); err != nil {
			return
		}
	}

	heartbeat := time.NewTicker(15 * time.Second)
	defer heartbeat.Stop()
	for {
		select {
		case <-r.Context().Done():
			return
		case event := <-deps.Bus.Events():
			if event.RunID != runID {
				continue
			}
			if err := writeSSE(w, "run_event", event); err != nil {
				return
			}
		case <-heartbeat.C:
			if err := writeSSE(w, "heartbeat", map[string]string{"status": "ok"}); err != nil {
				return
			}
		}
	}
}

func (deps ServerDeps) handleListRunArtifacts(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	runID := chi.URLParam(r, "run_id")
	artifacts, err := deps.Store.ListRunArtifacts(r.Context(), runID, parseLimit(r, 500))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, runArtifactsResponse{RunID: runID, Count: len(artifacts), Artifacts: artifacts})
}

func (deps ServerDeps) handleGetArtifact(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	artifact, err := deps.Store.GetArtifact(r.Context(), chi.URLParam(r, "artifact_id"))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, artifactResponse{Artifact: artifact})
}

func (deps ServerDeps) ready(w http.ResponseWriter) bool {
	if deps.Runs == nil || deps.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "run control service is not configured"})
		return false
	}
	return true
}

func decodeJSON(w http.ResponseWriter, r *http.Request, target any) bool {
	defer r.Body.Close()
	decoder := json.NewDecoder(r.Body)
	if err := decoder.Decode(target); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return false
	}
	return true
}

func parseLimit(r *http.Request, fallback int) int {
	raw := r.URL.Query().Get("limit")
	if raw == "" {
		return fallback
	}
	limit, err := strconv.Atoi(raw)
	if err != nil || limit < 1 {
		return fallback
	}
	return limit
}

func writeStoreError(w http.ResponseWriter, err error) {
	if errors.Is(err, store.ErrNotFound) {
		writeError(w, http.StatusNotFound, err)
		return
	}
	writeError(w, http.StatusInternalServerError, err)
}

func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, map[string]string{"error": err.Error()})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
