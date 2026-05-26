# Go Run Control Spine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first working Go control-plane backend that exposes the frontend-compatible v1/v2 run APIs, persists run state, streams run events, dispatches deterministic worker jobs, and measures the latency targets from the design spec.

**Architecture:** This plan implements Milestone 1 from `docs/superpowers/specs/2026-05-26-deep-agents-go-backend-design.md`. Go owns HTTP routing, run/thread/artifact state, server-sent events, NATS JetStream dispatch, and Postgres persistence; Python Deep Agents and the warm sandbox are represented by a deterministic worker stub so the control plane can be tested end to end before the real runtime is attached.

**Tech Stack:** Go 1.23+, `net/http`, `chi/v5`, OpenAPI, `oapi-codegen`, `pgx/v5`, `sqlc`, Postgres, NATS JetStream, `log/slog`, Prometheus, OpenTelemetry-ready middleware, Vitest frontend contract checks.

---

## Scope Split

The approved design covers multiple independent subsystems. This plan covers only the **Go Run Control Spine** because it produces a working, testable backend slice on its own.

Separate follow-up plans should cover:

- Python Deep Agents runtime construction with official Deep Agents backends, memory, context, and async subagents.
- Warm sandbox pool that upgrades the previous Docker sandbox from per-command `docker run --rm` to per-run warm containers.
- In-house model services and GPU scheduling.
- Deep Agents memory, skills, policies, and background consolidation.

This plan still creates the interfaces those follow-up plans plug into.

## File Structure

Create a new Go service under `backend/controlplane/` so it does not collide with the deleted legacy Python backend.

```text
backend/controlplane/
  go.mod
  go.sum
  Makefile
  sqlc.yaml
  cmd/ultra-control/main.go
  api/openapi.yaml
  api/oapi-codegen.yaml
  internal/app/app.go
  internal/config/config.go
  internal/domain/models.go
  internal/httpapi/handlers.go
  internal/httpapi/handlers_test.go
  internal/httpapi/middleware.go
  internal/httpapi/sse.go
  internal/httpapi/sse_test.go
  internal/openapi/generated.gen.go
  internal/runcontrol/service.go
  internal/runcontrol/service_test.go
  internal/store/memory.go
  internal/store/memory_test.go
  internal/store/postgres.go
  internal/store/queries.sql
  internal/store/schema.sql
  internal/store/sqlc/
  internal/eventbus/bus.go
  internal/eventbus/memory.go
  internal/eventbus/nats.go
  internal/eventbus/memory_test.go
  internal/worker/stub.go
  internal/worker/stub_test.go
  internal/latency/targets.go
  internal/latency/targets_test.go
  migrations/000001_run_control.up.sql
  migrations/000001_run_control.down.sql
```

Modify these existing repo files:

```text
Makefile
.env.example
```

The root `Makefile` should gain wrapper targets that delegate into `backend/controlplane/Makefile`. The existing frontend should not be modified in this plan unless a contract test reveals a genuine mismatch.

## Data and API Contracts

Use these canonical IDs and statuses in Go:

```go
type ThreadStatus string
const (
    ThreadStatusActive   ThreadStatus = "active"
    ThreadStatusArchived ThreadStatus = "archived"
    ThreadStatusDeleted  ThreadStatus = "deleted"
)

type RunStatus string
const (
    RunStatusQueued          RunStatus = "queued"
    RunStatusRunning         RunStatus = "running"
    RunStatusWaitingForInput RunStatus = "waiting_for_input"
    RunStatusWaitingForTask  RunStatus = "waiting_for_task"
    RunStatusSucceeded       RunStatus = "succeeded"
    RunStatusFailed          RunStatus = "failed"
    RunStatusCanceled        RunStatus = "canceled"
)
```

Use UUIDv7-compatible string IDs if available through the selected library. If a UUIDv7 dependency is not chosen in Task 1, use `github.com/google/uuid` v4 strings and keep ID creation behind `internal/domain.NewID(prefix string)` so it can be swapped without touching handlers.

SSE event names must match the existing frontend parser:

```text
token
run_event
done
error
heartbeat
```

The OpenAPI schemas must match `frontend/src/types-v2.ts` for these response types:

```text
V2ThreadRecord
V2ThreadListResponse
V2ThreadMessageListResponse
V2RunRecord
V2RunListResponse
V2GraphEventRecord
V2RunEventsResponse
V2ArtifactRecord
V2ArtifactListResponse
V2ArtifactResponse
```

## Task 1: Scaffold Go Control Plane Module

**Files:**
- Create: `backend/controlplane/go.mod`
- Create: `backend/controlplane/Makefile`
- Create: `backend/controlplane/cmd/ultra-control/main.go`
- Create: `backend/controlplane/internal/config/config.go`
- Create: `backend/controlplane/internal/httpapi/handlers.go`
- Create: `backend/controlplane/internal/httpapi/handlers_test.go`
- Create: `backend/controlplane/internal/app/app.go`
- Modify: `Makefile`

- [ ] **Step 1: Write the failing health/config test**

Create `backend/controlplane/internal/httpapi/handlers_test.go`:

```go
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
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
cd backend/controlplane && go test ./internal/httpapi -run TestHealthAndPublicConfig -count=1
```

Expected: FAIL because `go.mod`, package `httpapi`, and `NewRouter` do not exist.

- [ ] **Step 3: Add the minimal Go module and HTTP router**

Create `backend/controlplane/go.mod`:

```go
module github.com/amilworks/bisque-ultra/backend/controlplane

go 1.23

require github.com/go-chi/chi/v5 v5.2.3
```

Create `backend/controlplane/internal/config/config.go`:

```go
package config

import (
	"os"
	"strconv"
	"time"
)

type Config struct {
	AppName      string
	AppVersion   string
	HTTPAddr     string
	ReadTimeout  time.Duration
	WriteTimeout time.Duration
	IdleTimeout  time.Duration
}

func Load() Config {
	return Config{
		AppName:      envString("ULTRA_CONTROL_APP_NAME", "BisQue Ultra Control Plane"),
		AppVersion:   envString("ULTRA_CONTROL_APP_VERSION", "dev"),
		HTTPAddr:     envString("ULTRA_CONTROL_HTTP_ADDR", "127.0.0.1:8088"),
		ReadTimeout:  envDurationSeconds("ULTRA_CONTROL_READ_TIMEOUT_SECONDS", 10),
		WriteTimeout: envDurationSeconds("ULTRA_CONTROL_WRITE_TIMEOUT_SECONDS", 0),
		IdleTimeout:  envDurationSeconds("ULTRA_CONTROL_IDLE_TIMEOUT_SECONDS", 120),
	}
}

func envString(key string, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func envDurationSeconds(key string, fallback int) time.Duration {
	raw := os.Getenv(key)
	if raw == "" {
		return time.Duration(fallback) * time.Second
	}
	value, err := strconv.Atoi(raw)
	if err != nil || value < 0 {
		return time.Duration(fallback) * time.Second
	}
	return time.Duration(value) * time.Second
}
```

Create `backend/controlplane/internal/httpapi/handlers.go`:

```go
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
```

Create `backend/controlplane/internal/app/app.go`:

```go
package app

import (
	"net/http"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/config"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/httpapi"
)

func NewHTTPHandler(cfg config.Config) http.Handler {
	return httpapi.NewRouter(httpapi.ServerDeps{
		Version: cfg.AppVersion,
	})
}
```

Create `backend/controlplane/cmd/ultra-control/main.go`:

```go
package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/app"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/config"
)

func main() {
	cfg := config.Load()
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))

	server := &http.Server{
		Addr:         cfg.HTTPAddr,
		Handler:      app.NewHTTPHandler(cfg),
		ReadTimeout:  cfg.ReadTimeout,
		WriteTimeout: cfg.WriteTimeout,
		IdleTimeout:  cfg.IdleTimeout,
	}

	errs := make(chan error, 1)
	go func() {
		logger.Info("starting control plane", "addr", cfg.HTTPAddr)
		errs <- server.ListenAndServe()
	}()

	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)

	select {
	case sig := <-signals:
		logger.Info("shutting down", "signal", sig.String())
	case err := <-errs:
		if !errors.Is(err, http.ErrServerClosed) {
			logger.Error("server failed", "error", err)
			os.Exit(1)
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := server.Shutdown(ctx); err != nil {
		logger.Error("shutdown failed", "error", err)
		os.Exit(1)
	}
}
```

Create `backend/controlplane/Makefile`:

```makefile
.PHONY: test run tidy

test:
	go test ./...

run:
	go run ./cmd/ultra-control

tidy:
	go mod tidy
```

Modify the root `Makefile` `.PHONY` line to add `control-test control-run control-tidy`, then add these targets:

```makefile
control-test: ## Run Go control plane tests
	$(MAKE) -C backend/controlplane test

control-run: ## Run Go control plane API
	$(MAKE) -C backend/controlplane run

control-tidy: ## Tidy Go control plane module
	$(MAKE) -C backend/controlplane tidy
```

- [ ] **Step 4: Run the test and tidy**

Run:

```bash
cd backend/controlplane && go mod tidy && go test ./...
```

Expected: PASS for `TestHealthAndPublicConfig`.

- [ ] **Step 5: Commit**

```bash
git add Makefile backend/controlplane
git commit -m "feat: scaffold go control plane"
```

## Task 2: Define OpenAPI Contract and Generated Types

**Files:**
- Create: `backend/controlplane/api/openapi.yaml`
- Create: `backend/controlplane/api/oapi-codegen.yaml`
- Create: `backend/controlplane/internal/openapi/openapi_test.go`
- Generate: `backend/controlplane/internal/openapi/generated.gen.go`
- Modify: `backend/controlplane/go.mod`
- Modify: `backend/controlplane/Makefile`

- [ ] **Step 1: Write the failing OpenAPI coverage test**

Create `backend/controlplane/internal/openapi/openapi_test.go`:

```go
package openapi_test

import (
	"os"
	"strings"
	"testing"
)

func TestOpenAPIIncludesFrontendV2Routes(t *testing.T) {
	t.Parallel()

	data, err := os.ReadFile("../../api/openapi.yaml")
	if err != nil {
		t.Fatalf("read openapi.yaml: %v", err)
	}
	doc := string(data)
	required := []string{
		"/v1/health:",
		"/v1/config/public:",
		"/v1/auth/session:",
		"/v2/threads:",
		"/v2/threads/{thread_id}:",
		"/v2/threads/{thread_id}/messages:",
		"/v2/threads/{thread_id}/runs:",
		"/v2/runs:",
		"/v2/runs/{run_id}:",
		"/v2/runs/{run_id}/cancel:",
		"/v2/runs/{run_id}/events:",
		"/v2/runs/{run_id}/artifacts:",
		"/v2/artifacts/{artifact_id}:",
		"V2ThreadRecord:",
		"V2RunRecord:",
		"V2GraphEventRecord:",
		"V2ArtifactRecord:",
	}
	for _, needle := range required {
		if !strings.Contains(doc, needle) {
			t.Fatalf("openapi.yaml missing %s", needle)
		}
	}
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
cd backend/controlplane && go test ./internal/openapi -run TestOpenAPIIncludesFrontendV2Routes -count=1
```

Expected: FAIL because `api/openapi.yaml` does not exist.

- [ ] **Step 3: Add OpenAPI and generation config**

Create `backend/controlplane/api/oapi-codegen.yaml`:

```yaml
package: openapi
output: internal/openapi/generated.gen.go
generate:
  chi-server: true
  strict-server: true
  models: true
  embedded-spec: true
output-options:
  skip-prune: true
```

Create `backend/controlplane/api/openapi.yaml` with the v1/v2 routes listed in the spec. Include these schema names exactly:

```yaml
openapi: 3.0.3
info:
  title: BisQue Ultra Control Plane
  version: 0.1.0
paths:
  /v1/health:
    get:
      operationId: getHealth
      responses:
        "200":
          description: Health response
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/HealthResponse"
  /v1/config/public:
    get:
      operationId: getPublicConfig
      responses:
        "200":
          description: Public config
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/PublicConfigResponse"
  /v1/auth/session:
    get:
      operationId: getAuthSession
      responses:
        "200":
          description: Auth session
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/AuthSessionResponse"
  /v2/threads:
    get:
      operationId: listThreads
      parameters:
        - name: limit
          in: query
          schema: { type: integer, minimum: 1, maximum: 500, default: 100 }
        - name: offset
          in: query
          schema: { type: integer, minimum: 0, default: 0 }
        - name: status
          in: query
          schema: { type: string }
      responses:
        "200":
          description: Thread list
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/V2ThreadListResponse"
    post:
      operationId: createThread
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/V2ThreadCreateRequest"
      responses:
        "200":
          description: Created thread
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/V2ThreadRecord"
  /v2/threads/{thread_id}:
    get:
      operationId: getThread
      parameters:
        - $ref: "#/components/parameters/ThreadID"
      responses:
        "200":
          description: Thread
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/V2ThreadRecord"
    put:
      operationId: upsertThread
      parameters:
        - $ref: "#/components/parameters/ThreadID"
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/V2ThreadUpsertRequest"
      responses:
        "200":
          description: Upserted thread
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/V2ThreadRecord"
  /v2/threads/{thread_id}/messages:
    get:
      operationId: listThreadMessages
      parameters:
        - $ref: "#/components/parameters/ThreadID"
      responses:
        "200":
          description: Thread messages
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/V2ThreadMessageListResponse"
  /v2/threads/{thread_id}/runs:
    post:
      operationId: createRun
      parameters:
        - $ref: "#/components/parameters/ThreadID"
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/V2RunCreateRequest"
      responses:
        "200":
          description: Created run or stream
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/V2RunRecord"
            text/event-stream:
              schema:
                type: string
  /v2/runs:
    get:
      operationId: listRuns
      parameters:
        - name: thread_id
          in: query
          schema: { type: string }
        - name: status
          in: query
          schema: { type: string }
        - name: limit
          in: query
          schema: { type: integer, minimum: 1, maximum: 500, default: 100 }
        - name: offset
          in: query
          schema: { type: integer, minimum: 0, default: 0 }
      responses:
        "200":
          description: Run list
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/V2RunListResponse"
  /v2/runs/{run_id}:
    get:
      operationId: getRun
      parameters:
        - $ref: "#/components/parameters/RunID"
      responses:
        "200":
          description: Run
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/V2RunRecord"
  /v2/runs/{run_id}/resume:
    post:
      operationId: resumeRun
      parameters:
        - $ref: "#/components/parameters/RunID"
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/V2RunResumeRequest"
      responses:
        "200":
          description: Resumed run
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/V2RunRecord"
  /v2/runs/{run_id}/cancel:
    post:
      operationId: cancelRun
      parameters:
        - $ref: "#/components/parameters/RunID"
      requestBody:
        required: false
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/V2RunCancelRequest"
      responses:
        "200":
          description: Canceled run
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/V2RunRecord"
  /v2/runs/{run_id}/events:
    get:
      operationId: listRunEvents
      parameters:
        - $ref: "#/components/parameters/RunID"
        - name: limit
          in: query
          schema: { type: integer, minimum: 1, maximum: 1000, default: 500 }
        - name: stream
          in: query
          schema: { type: boolean, default: false }
      responses:
        "200":
          description: Run events or event stream
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/V2RunEventsResponse"
            text/event-stream:
              schema:
                type: string
  /v2/runs/{run_id}/artifacts:
    get:
      operationId: listRunArtifacts
      parameters:
        - $ref: "#/components/parameters/RunID"
        - name: limit
          in: query
          schema: { type: integer, minimum: 1, maximum: 1000, default: 500 }
      responses:
        "200":
          description: Run artifacts
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/V2ArtifactListResponse"
  /v2/artifacts/{artifact_id}:
    get:
      operationId: getArtifact
      parameters:
        - name: artifact_id
          in: path
          required: true
          schema: { type: string }
      responses:
        "200":
          description: Artifact
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/V2ArtifactResponse"
components:
  parameters:
    ThreadID:
      name: thread_id
      in: path
      required: true
      schema: { type: string }
    RunID:
      name: run_id
      in: path
      required: true
      schema: { type: string }
  schemas:
    JsonObject:
      type: object
      additionalProperties: true
    HealthResponse:
      type: object
      required: [status, ts]
      properties:
        status: { type: string }
        ts: { type: string }
    PublicConfigResponse:
      type: object
      additionalProperties: true
    AuthSessionResponse:
      type: object
      required: [authenticated, user]
      properties:
        authenticated: { type: boolean }
        user:
          nullable: true
          $ref: "#/components/schemas/JsonObject"
    V2ThreadMessage:
      type: object
      required: [role, content]
      properties:
        message_id: { type: string, nullable: true }
        thread_id: { type: string, nullable: true }
        role: { type: string }
        content: { type: string }
        created_at: { type: string, nullable: true }
        metadata:
          nullable: true
          $ref: "#/components/schemas/JsonObject"
        run_id: { type: string, nullable: true }
    V2ThreadRecord:
      type: object
      required: [thread_id, status, created_at, updated_at, metadata]
      properties:
        thread_id: { type: string }
        user_id: { type: string, nullable: true }
        title: { type: string, nullable: true }
        status: { type: string }
        created_at: { type: string }
        updated_at: { type: string }
        latest_run_id: { type: string, nullable: true }
        checkpoint_id: { type: string, nullable: true }
        summary: { type: string, nullable: true }
        metadata:
          $ref: "#/components/schemas/JsonObject"
    V2ThreadListResponse:
      type: object
      required: [count, threads]
      properties:
        count: { type: integer }
        threads:
          type: array
          items:
            $ref: "#/components/schemas/V2ThreadRecord"
    V2ThreadMessageListResponse:
      type: object
      required: [thread_id, count, messages]
      properties:
        thread_id: { type: string }
        count: { type: integer }
        messages:
          type: array
          items:
            $ref: "#/components/schemas/V2ThreadMessage"
    V2ThreadCreateRequest:
      type: object
      properties:
        title: { type: string, nullable: true }
        metadata:
          nullable: true
          $ref: "#/components/schemas/JsonObject"
        initial_messages:
          type: array
          items:
            $ref: "#/components/schemas/V2ThreadMessage"
        conversation_id: { type: string, nullable: true }
    V2ThreadUpsertRequest:
      type: object
      properties:
        title: { type: string, nullable: true }
        metadata:
          nullable: true
          $ref: "#/components/schemas/JsonObject"
        messages:
          type: array
          items:
            $ref: "#/components/schemas/V2ThreadMessage"
    V2RunBudget:
      type: object
      additionalProperties: true
      properties:
        max_tool_calls: { type: integer, nullable: true }
        max_runtime_seconds: { type: integer, nullable: true }
    V2RunCreateRequest:
      type: object
      required: [messages]
      properties:
        goal: { type: string, nullable: true }
        messages:
          type: array
          items:
            $ref: "#/components/schemas/V2ThreadMessage"
        file_ids:
          type: array
          items: { type: string }
        resource_uris:
          type: array
          items: { type: string }
        dataset_uris:
          type: array
          items: { type: string }
        selected_tool_names:
          type: array
          items: { type: string }
        knowledge_context:
          nullable: true
          $ref: "#/components/schemas/JsonObject"
        selection_context:
          nullable: true
          $ref: "#/components/schemas/JsonObject"
        workflow_hint:
          nullable: true
          $ref: "#/components/schemas/JsonObject"
        reasoning_mode: { type: string, nullable: true }
        budgets:
          nullable: true
          $ref: "#/components/schemas/V2RunBudget"
        benchmark:
          nullable: true
          $ref: "#/components/schemas/JsonObject"
        metadata:
          nullable: true
          $ref: "#/components/schemas/JsonObject"
    V2RunResumeRequest:
      type: object
      properties:
        decision: { type: string, nullable: true }
        note: { type: string, nullable: true }
        metadata:
          nullable: true
          $ref: "#/components/schemas/JsonObject"
    V2RunCancelRequest:
      type: object
      properties:
        reason: { type: string, nullable: true }
        metadata:
          nullable: true
          $ref: "#/components/schemas/JsonObject"
    V2RunRecord:
      type: object
      required: [run_id, goal, status, workflow_kind, created_at, updated_at, metadata]
      properties:
        run_id: { type: string }
        thread_id: { type: string, nullable: true }
        user_id: { type: string, nullable: true }
        goal: { type: string }
        status: { type: string }
        workflow_kind: { type: string }
        mode: { type: string, nullable: true }
        current_node: { type: string, nullable: true }
        parent_run_id: { type: string, nullable: true }
        planner_version: { type: string, nullable: true }
        agent_role: { type: string, nullable: true }
        trace_group_id: { type: string, nullable: true }
        checkpoint_id: { type: string, nullable: true }
        checkpoint_state:
          nullable: true
          $ref: "#/components/schemas/JsonObject"
        budget_state:
          nullable: true
          $ref: "#/components/schemas/JsonObject"
        response_text: { type: string, nullable: true }
        error: { type: string, nullable: true }
        created_at: { type: string }
        updated_at: { type: string }
        started_at: { type: string, nullable: true }
        completed_at: { type: string, nullable: true }
        metadata:
          $ref: "#/components/schemas/JsonObject"
    V2RunListResponse:
      type: object
      required: [count, runs]
      properties:
        count: { type: integer }
        runs:
          type: array
          items:
            $ref: "#/components/schemas/V2RunRecord"
    V2GraphEventRecord:
      type: object
      required: [run_id, event_kind, payload]
      properties:
        event_id: { nullable: true }
        run_id: { type: string }
        thread_id: { type: string, nullable: true }
        event_kind: { type: string }
        event_type: { type: string, nullable: true }
        node_name: { type: string, nullable: true }
        task_id: { type: string, nullable: true }
        checkpoint_id: { type: string, nullable: true }
        scope_id: { type: string, nullable: true }
        agent_role: { type: string, nullable: true }
        level: { type: string, nullable: true }
        ts: { type: string, nullable: true }
        message: { type: string, nullable: true }
        payload:
          $ref: "#/components/schemas/JsonObject"
    V2RunEventsResponse:
      type: object
      required: [run_id, count, events]
      properties:
        run_id: { type: string }
        count: { type: integer }
        events:
          type: array
          items:
            $ref: "#/components/schemas/V2GraphEventRecord"
    V2ArtifactRecord:
      type: object
      required: [artifact_id, run_id, kind, created_at, metadata]
      properties:
        artifact_id: { type: string }
        run_id: { type: string }
        thread_id: { type: string, nullable: true }
        kind: { type: string }
        path: { type: string, nullable: true }
        source_path: { type: string, nullable: true }
        preview_path: { type: string, nullable: true }
        title: { type: string, nullable: true }
        result_group_id: { type: string, nullable: true }
        mime_type: { type: string, nullable: true }
        size_bytes: { type: integer, format: int64, nullable: true }
        sha256: { type: string, nullable: true }
        storage_uri: { type: string, nullable: true }
        tool_name: { type: string, nullable: true }
        category: { type: string, nullable: true }
        created_at: { type: string }
        updated_at: { type: string, nullable: true }
        metadata:
          $ref: "#/components/schemas/JsonObject"
    V2ArtifactListResponse:
      type: object
      required: [run_id, count, artifacts]
      properties:
        run_id: { type: string }
        count: { type: integer }
        artifacts:
          type: array
          items:
            $ref: "#/components/schemas/V2ArtifactRecord"
    V2ArtifactResponse:
      type: object
      required: [artifact]
      properties:
        artifact:
          $ref: "#/components/schemas/V2ArtifactRecord"
```

Add generation dependencies:

```bash
cd backend/controlplane
go get github.com/oapi-codegen/oapi-codegen/v2/cmd/oapi-codegen@latest
go get github.com/oapi-codegen/runtime@latest
```

Modify `backend/controlplane/Makefile`:

```makefile
.PHONY: test run tidy generate

generate:
	go run github.com/oapi-codegen/oapi-codegen/v2/cmd/oapi-codegen --config api/oapi-codegen.yaml api/openapi.yaml

test:
	go test ./...

run:
	go run ./cmd/ultra-control

tidy:
	go mod tidy
```

- [ ] **Step 4: Generate code and run tests**

Run:

```bash
cd backend/controlplane && make generate && go mod tidy && go test ./...
```

Expected: PASS and `internal/openapi/generated.gen.go` exists.

- [ ] **Step 5: Commit**

```bash
git add backend/controlplane
git commit -m "feat: add control plane openapi contract"
```

## Task 3: Add Domain Models and In-Memory Store

**Files:**
- Create: `backend/controlplane/internal/domain/models.go`
- Create: `backend/controlplane/internal/store/memory.go`
- Create: `backend/controlplane/internal/store/memory_test.go`

- [ ] **Step 1: Write failing store tests**

Create `backend/controlplane/internal/store/memory_test.go`:

```go
package store

import (
	"context"
	"testing"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

func TestMemoryStoreThreadRunEventArtifactFlow(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	thread, err := store.CreateThread(ctx, domain.CreateThreadInput{
		UserID: "user-1",
		Title: "Microscopy analysis",
		InitialMessages: []domain.ThreadMessage{{
			Role:    "user",
			Content: "Segment these images.",
		}},
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	if thread.ThreadID == "" || thread.Status != domain.ThreadStatusActive {
		t.Fatalf("unexpected thread: %+v", thread)
	}

	run, err := store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Segment these images.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Segment these images."}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	if run.Status != domain.RunStatusQueued {
		t.Fatalf("run status = %s, want queued", run.Status)
	}

	event, err := store.AppendRunEvent(ctx, domain.AppendRunEventInput{
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.started",
		Message:   "Run started.",
		Payload:   domain.JSONMap{"phase": "planning"},
	})
	if err != nil {
		t.Fatalf("AppendRunEvent: %v", err)
	}
	if event.EventID == "" {
		t.Fatalf("event id must be set")
	}

	artifact, err := store.CreateArtifact(ctx, domain.CreateArtifactInput{
		RunID:      run.RunID,
		ThreadID:   thread.ThreadID,
		Kind:       "report",
		Path:       "outputs/report.md",
		Title:      "Report",
		MimeType:   "text/markdown",
		SizeBytes:  42,
		SHA256:     "abc123",
		StorageURI: "file://outputs/report.md",
		Metadata:   domain.JSONMap{"source": "stub"},
	})
	if err != nil {
		t.Fatalf("CreateArtifact: %v", err)
	}
	if artifact.ArtifactID == "" {
		t.Fatalf("artifact id must be set")
	}

	events, err := store.ListRunEvents(ctx, run.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	if len(events) != 1 || events[0].EventKind != "run.started" {
		t.Fatalf("events = %+v, want one run.started", events)
	}

	artifacts, err := store.ListRunArtifacts(ctx, run.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunArtifacts: %v", err)
	}
	if len(artifacts) != 1 || artifacts[0].Path != "outputs/report.md" {
		t.Fatalf("artifacts = %+v, want report", artifacts)
	}
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
cd backend/controlplane && go test ./internal/store -run TestMemoryStoreThreadRunEventArtifactFlow -count=1
```

Expected: FAIL because domain and store packages do not exist.

- [ ] **Step 3: Add domain models and memory store**

Create `backend/controlplane/internal/domain/models.go`:

```go
package domain

import (
	"crypto/rand"
	"encoding/hex"
	"strings"
	"time"
)

type JSONMap map[string]any

type ThreadStatus string
type RunStatus string

const (
	ThreadStatusActive   ThreadStatus = "active"
	ThreadStatusArchived ThreadStatus = "archived"
	ThreadStatusDeleted  ThreadStatus = "deleted"

	RunStatusQueued          RunStatus = "queued"
	RunStatusRunning         RunStatus = "running"
	RunStatusWaitingForInput RunStatus = "waiting_for_input"
	RunStatusWaitingForTask  RunStatus = "waiting_for_task"
	RunStatusSucceeded       RunStatus = "succeeded"
	RunStatusFailed          RunStatus = "failed"
	RunStatusCanceled        RunStatus = "canceled"
)

type ThreadMessage struct {
	MessageID string    `json:"message_id,omitempty"`
	ThreadID  string    `json:"thread_id,omitempty"`
	Role      string    `json:"role"`
	Content   string    `json:"content"`
	CreatedAt time.Time `json:"created_at,omitempty"`
	Metadata  JSONMap   `json:"metadata,omitempty"`
	RunID     string    `json:"run_id,omitempty"`
}

type ThreadRecord struct {
	ThreadID     string       `json:"thread_id"`
	UserID       string       `json:"user_id,omitempty"`
	Title        string       `json:"title,omitempty"`
	Status       ThreadStatus `json:"status"`
	CreatedAt    time.Time    `json:"created_at"`
	UpdatedAt    time.Time    `json:"updated_at"`
	LatestRunID  string       `json:"latest_run_id,omitempty"`
	CheckpointID string       `json:"checkpoint_id,omitempty"`
	Summary      string       `json:"summary,omitempty"`
	Metadata     JSONMap      `json:"metadata"`
}

type RunRecord struct {
	RunID           string    `json:"run_id"`
	ThreadID        string    `json:"thread_id,omitempty"`
	UserID          string    `json:"user_id,omitempty"`
	Goal            string    `json:"goal"`
	Status          RunStatus `json:"status"`
	WorkflowKind    string    `json:"workflow_kind"`
	Mode            string    `json:"mode,omitempty"`
	CurrentNode     string    `json:"current_node,omitempty"`
	ParentRunID     string    `json:"parent_run_id,omitempty"`
	PlannerVersion  string    `json:"planner_version,omitempty"`
	AgentRole       string    `json:"agent_role,omitempty"`
	TraceGroupID    string    `json:"trace_group_id,omitempty"`
	CheckpointID    string    `json:"checkpoint_id,omitempty"`
	CheckpointState JSONMap   `json:"checkpoint_state,omitempty"`
	BudgetState      JSONMap   `json:"budget_state,omitempty"`
	ResponseText     string    `json:"response_text,omitempty"`
	Error            string    `json:"error,omitempty"`
	CreatedAt        time.Time `json:"created_at"`
	UpdatedAt        time.Time `json:"updated_at"`
	StartedAt        *time.Time `json:"started_at,omitempty"`
	CompletedAt      *time.Time `json:"completed_at,omitempty"`
	Metadata         JSONMap   `json:"metadata"`
}

type RunEventRecord struct {
	EventID      string    `json:"event_id,omitempty"`
	Sequence     int64     `json:"sequence,omitempty"`
	RunID        string    `json:"run_id"`
	ThreadID     string    `json:"thread_id,omitempty"`
	EventKind    string    `json:"event_kind"`
	EventType    string    `json:"event_type,omitempty"`
	NodeName     string    `json:"node_name,omitempty"`
	TaskID       string    `json:"task_id,omitempty"`
	CheckpointID string   `json:"checkpoint_id,omitempty"`
	ScopeID      string   `json:"scope_id,omitempty"`
	AgentRole    string   `json:"agent_role,omitempty"`
	Level        string   `json:"level,omitempty"`
	TS           time.Time `json:"ts,omitempty"`
	Message      string   `json:"message,omitempty"`
	Payload      JSONMap  `json:"payload"`
}

type ArtifactRecord struct {
	ArtifactID    string    `json:"artifact_id"`
	RunID         string    `json:"run_id"`
	ThreadID      string    `json:"thread_id,omitempty"`
	Kind          string    `json:"kind"`
	Path          string    `json:"path,omitempty"`
	SourcePath    string    `json:"source_path,omitempty"`
	PreviewPath   string    `json:"preview_path,omitempty"`
	Title         string    `json:"title,omitempty"`
	ResultGroupID string    `json:"result_group_id,omitempty"`
	MimeType      string    `json:"mime_type,omitempty"`
	SizeBytes     int64     `json:"size_bytes,omitempty"`
	SHA256        string    `json:"sha256,omitempty"`
	StorageURI    string    `json:"storage_uri,omitempty"`
	ToolName      string    `json:"tool_name,omitempty"`
	Category      string    `json:"category,omitempty"`
	CreatedAt     time.Time `json:"created_at"`
	UpdatedAt     time.Time `json:"updated_at,omitempty"`
	Metadata      JSONMap   `json:"metadata"`
}

type CreateThreadInput struct {
	UserID          string
	Title           string
	Metadata        JSONMap
	InitialMessages []ThreadMessage
}

type CreateRunInput struct {
	ThreadID string
	UserID   string
	Goal     string
	Messages []ThreadMessage
	Metadata JSONMap
}

type AppendRunEventInput struct {
	RunID     string
	ThreadID  string
	EventKind string
	Message   string
	Payload   JSONMap
}

type CreateArtifactInput struct {
	RunID      string
	ThreadID   string
	Kind       string
	Path       string
	Title      string
	MimeType   string
	SizeBytes  int64
	SHA256     string
	StorageURI string
	Metadata   JSONMap
}

func NewID(prefix string) string {
	var bytes [16]byte
	if _, err := rand.Read(bytes[:]); err != nil {
		return strings.TrimSuffix(prefix, "_") + "_" + time.Now().UTC().Format("20060102150405.000000000")
	}
	return strings.TrimSuffix(prefix, "_") + "_" + hex.EncodeToString(bytes[:])
}

func Now() time.Time {
	return time.Now().UTC()
}
```

Create `backend/controlplane/internal/store/memory.go` with a mutex-protected store implementing the methods used by the test:

```go
package store

import (
	"context"
	"errors"
	"sort"
	"sync"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

var ErrNotFound = errors.New("not found")

type MemoryStore struct {
	mu        sync.RWMutex
	threads   map[string]domain.ThreadRecord
	messages  map[string][]domain.ThreadMessage
	runs      map[string]domain.RunRecord
	events    map[string][]domain.RunEventRecord
	artifacts map[string]domain.ArtifactRecord
}

func NewMemoryStore() *MemoryStore {
	return &MemoryStore{
		threads:   map[string]domain.ThreadRecord{},
		messages:  map[string][]domain.ThreadMessage{},
		runs:      map[string]domain.RunRecord{},
		events:    map[string][]domain.RunEventRecord{},
		artifacts: map[string]domain.ArtifactRecord{},
	}
}

func (s *MemoryStore) CreateThread(ctx context.Context, input domain.CreateThreadInput) (domain.ThreadRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()

	now := domain.Now()
	thread := domain.ThreadRecord{
		ThreadID:  domain.NewID("thread"),
		UserID:    input.UserID,
		Title:     input.Title,
		Status:    domain.ThreadStatusActive,
		CreatedAt: now,
		UpdatedAt: now,
		Metadata:  mapOrEmpty(input.Metadata),
	}
	s.threads[thread.ThreadID] = thread
	for _, msg := range input.InitialMessages {
		msg.MessageID = domain.NewID("msg")
		msg.ThreadID = thread.ThreadID
		msg.CreatedAt = now
		msg.Metadata = mapOrEmpty(msg.Metadata)
		s.messages[thread.ThreadID] = append(s.messages[thread.ThreadID], msg)
	}
	return thread, nil
}

func (s *MemoryStore) GetThread(ctx context.Context, threadID string) (domain.ThreadRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	thread, ok := s.threads[threadID]
	if !ok {
		return domain.ThreadRecord{}, ErrNotFound
	}
	return thread, nil
}

func (s *MemoryStore) ListThreads(ctx context.Context, limit int) ([]domain.ThreadRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	threads := make([]domain.ThreadRecord, 0, len(s.threads))
	for _, thread := range s.threads {
		threads = append(threads, thread)
	}
	sort.Slice(threads, func(i, j int) bool {
		return threads[i].UpdatedAt.After(threads[j].UpdatedAt)
	})
	return take(threads, limit), nil
}

func (s *MemoryStore) ListThreadMessages(ctx context.Context, threadID string) ([]domain.ThreadMessage, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	messages := append([]domain.ThreadMessage(nil), s.messages[threadID]...)
	return messages, nil
}

func (s *MemoryStore) CreateRun(ctx context.Context, input domain.CreateRunInput) (domain.RunRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.threads[input.ThreadID]; !ok {
		return domain.RunRecord{}, ErrNotFound
	}
	now := domain.Now()
	run := domain.RunRecord{
		RunID:        domain.NewID("run"),
		ThreadID:     input.ThreadID,
		UserID:       input.UserID,
		Goal:         input.Goal,
		Status:       domain.RunStatusQueued,
		WorkflowKind: "deep_agents",
		Mode:         "durable",
		CreatedAt:    now,
		UpdatedAt:    now,
		Metadata:     mapOrEmpty(input.Metadata),
	}
	s.runs[run.RunID] = run
	thread := s.threads[input.ThreadID]
	thread.LatestRunID = run.RunID
	thread.UpdatedAt = now
	s.threads[input.ThreadID] = thread
	for _, msg := range input.Messages {
		msg.MessageID = domain.NewID("msg")
		msg.ThreadID = input.ThreadID
		msg.RunID = run.RunID
		msg.CreatedAt = now
		msg.Metadata = mapOrEmpty(msg.Metadata)
		s.messages[input.ThreadID] = append(s.messages[input.ThreadID], msg)
	}
	return run, nil
}

func (s *MemoryStore) GetRun(ctx context.Context, runID string) (domain.RunRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	run, ok := s.runs[runID]
	if !ok {
		return domain.RunRecord{}, ErrNotFound
	}
	return run, nil
}

func (s *MemoryStore) UpdateRunStatus(ctx context.Context, runID string, status domain.RunStatus, responseText string, errorText string) (domain.RunRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	run, ok := s.runs[runID]
	if !ok {
		return domain.RunRecord{}, ErrNotFound
	}
	now := domain.Now()
	run.Status = status
	run.ResponseText = responseText
	run.Error = errorText
	run.UpdatedAt = now
	if status == domain.RunStatusRunning && run.StartedAt == nil {
		run.StartedAt = &now
	}
	if status == domain.RunStatusSucceeded || status == domain.RunStatusFailed || status == domain.RunStatusCanceled {
		run.CompletedAt = &now
	}
	s.runs[runID] = run
	return run, nil
}

func (s *MemoryStore) AppendRunEvent(ctx context.Context, input domain.AppendRunEventInput) (domain.RunEventRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.runs[input.RunID]; !ok {
		return domain.RunEventRecord{}, ErrNotFound
	}
	seq := int64(len(s.events[input.RunID]) + 1)
	event := domain.RunEventRecord{
		EventID:   domain.NewID("event"),
		Sequence:  seq,
		RunID:     input.RunID,
		ThreadID:  input.ThreadID,
		EventKind: input.EventKind,
		TS:        domain.Now(),
		Message:   input.Message,
		Payload:   mapOrEmpty(input.Payload),
	}
	s.events[input.RunID] = append(s.events[input.RunID], event)
	return event, nil
}

func (s *MemoryStore) ListRunEvents(ctx context.Context, runID string, limit int) ([]domain.RunEventRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	events := append([]domain.RunEventRecord(nil), s.events[runID]...)
	if limit > 0 && len(events) > limit {
		events = events[len(events)-limit:]
	}
	return events, nil
}

func (s *MemoryStore) CreateArtifact(ctx context.Context, input domain.CreateArtifactInput) (domain.ArtifactRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.runs[input.RunID]; !ok {
		return domain.ArtifactRecord{}, ErrNotFound
	}
	now := domain.Now()
	artifact := domain.ArtifactRecord{
		ArtifactID: domain.NewID("artifact"),
		RunID:      input.RunID,
		ThreadID:   input.ThreadID,
		Kind:       input.Kind,
		Path:       input.Path,
		Title:      input.Title,
		MimeType:   input.MimeType,
		SizeBytes:  input.SizeBytes,
		SHA256:     input.SHA256,
		StorageURI: input.StorageURI,
		CreatedAt:  now,
		UpdatedAt:  now,
		Metadata:   mapOrEmpty(input.Metadata),
	}
	s.artifacts[artifact.ArtifactID] = artifact
	return artifact, nil
}

func (s *MemoryStore) ListRunArtifacts(ctx context.Context, runID string, limit int) ([]domain.ArtifactRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	artifacts := []domain.ArtifactRecord{}
	for _, artifact := range s.artifacts {
		if artifact.RunID == runID {
			artifacts = append(artifacts, artifact)
		}
	}
	sort.Slice(artifacts, func(i, j int) bool {
		return artifacts[i].CreatedAt.After(artifacts[j].CreatedAt)
	})
	return take(artifacts, limit), nil
}

func (s *MemoryStore) GetArtifact(ctx context.Context, artifactID string) (domain.ArtifactRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	artifact, ok := s.artifacts[artifactID]
	if !ok {
		return domain.ArtifactRecord{}, ErrNotFound
	}
	return artifact, nil
}

func mapOrEmpty(value domain.JSONMap) domain.JSONMap {
	if value == nil {
		return domain.JSONMap{}
	}
	return value
}

func take[T any](values []T, limit int) []T {
	if limit <= 0 || len(values) <= limit {
		return values
	}
	return values[:limit]
}
```

- [ ] **Step 4: Run store tests**

Run:

```bash
cd backend/controlplane && go test ./internal/store -run TestMemoryStoreThreadRunEventArtifactFlow -count=1
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/controlplane/internal/domain backend/controlplane/internal/store
git commit -m "feat: add run control domain store"
```

## Task 4: Add Run Control Service and Deterministic Worker Stub

**Files:**
- Create: `backend/controlplane/internal/runcontrol/service.go`
- Create: `backend/controlplane/internal/runcontrol/service_test.go`
- Create: `backend/controlplane/internal/eventbus/bus.go`
- Create: `backend/controlplane/internal/eventbus/memory.go`
- Create: `backend/controlplane/internal/worker/stub.go`
- Create: `backend/controlplane/internal/worker/stub_test.go`

- [ ] **Step 1: Write failing run control test**

Create `backend/controlplane/internal/runcontrol/service_test.go`:

```go
package runcontrol

import (
	"context"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

func TestServiceCreateRunEmitsAcceptedAndDispatches(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Test thread",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}

	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Run deterministic worker.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Run deterministic worker."}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	if run.Status != domain.RunStatusQueued {
		t.Fatalf("run status = %s, want queued", run.Status)
	}

	events, err := mem.ListRunEvents(ctx, run.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	if len(events) != 1 || events[0].EventKind != "run.accepted" {
		t.Fatalf("events = %+v, want run.accepted", events)
	}

	select {
	case job := <-bus.Jobs():
		if job.RunID != run.RunID || job.ThreadID != thread.ThreadID {
			t.Fatalf("job = %+v, want run/thread ids", job)
		}
	case <-time.After(time.Second):
		t.Fatalf("expected dispatched job")
	}
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
cd backend/controlplane && go test ./internal/runcontrol -run TestServiceCreateRunEmitsAcceptedAndDispatches -count=1
```

Expected: FAIL because `runcontrol` and `eventbus` do not exist.

- [ ] **Step 3: Implement memory event bus and run control service**

Create `backend/controlplane/internal/eventbus/bus.go`:

```go
package eventbus

import (
	"context"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

type Job struct {
	RunID    string
	ThreadID string
	UserID   string
	Goal     string
}

type Bus interface {
	PublishJob(ctx context.Context, job Job) error
	PublishRunEvent(ctx context.Context, event domain.RunEventRecord) error
}
```

Create `backend/controlplane/internal/eventbus/memory.go`:

```go
package eventbus

import (
	"context"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

type MemoryBus struct {
	jobs   chan Job
	events chan domain.RunEventRecord
}

func NewMemoryBus() *MemoryBus {
	return &MemoryBus{
		jobs:   make(chan Job, 64),
		events: make(chan domain.RunEventRecord, 1024),
	}
}

func (b *MemoryBus) PublishJob(ctx context.Context, job Job) error {
	select {
	case b.jobs <- job:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (b *MemoryBus) PublishRunEvent(ctx context.Context, event domain.RunEventRecord) error {
	select {
	case b.events <- event:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (b *MemoryBus) Jobs() <-chan Job {
	return b.jobs
}

func (b *MemoryBus) Events() <-chan domain.RunEventRecord {
	return b.events
}
```

Create `backend/controlplane/internal/runcontrol/service.go`:

```go
package runcontrol

import (
	"context"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
)

type Store interface {
	CreateThread(context.Context, domain.CreateThreadInput) (domain.ThreadRecord, error)
	GetThread(context.Context, string) (domain.ThreadRecord, error)
	ListThreads(context.Context, int) ([]domain.ThreadRecord, error)
	ListThreadMessages(context.Context, string) ([]domain.ThreadMessage, error)
	CreateRun(context.Context, domain.CreateRunInput) (domain.RunRecord, error)
	GetRun(context.Context, string) (domain.RunRecord, error)
	UpdateRunStatus(context.Context, string, domain.RunStatus, string, string) (domain.RunRecord, error)
	AppendRunEvent(context.Context, domain.AppendRunEventInput) (domain.RunEventRecord, error)
	ListRunEvents(context.Context, string, int) ([]domain.RunEventRecord, error)
	CreateArtifact(context.Context, domain.CreateArtifactInput) (domain.ArtifactRecord, error)
	ListRunArtifacts(context.Context, string, int) ([]domain.ArtifactRecord, error)
	GetArtifact(context.Context, string) (domain.ArtifactRecord, error)
}

type Service struct {
	store Store
	bus   eventbus.Bus
}

type CreateThreadRequest struct {
	UserID          string
	Title           string
	Metadata        domain.JSONMap
	InitialMessages []domain.ThreadMessage
}

type CreateRunRequest struct {
	ThreadID string
	UserID   string
	Goal     string
	Messages []domain.ThreadMessage
	Metadata domain.JSONMap
}

func NewService(store Store, bus eventbus.Bus) *Service {
	return &Service{store: store, bus: bus}
}

func (s *Service) CreateThread(ctx context.Context, req CreateThreadRequest) (domain.ThreadRecord, error) {
	return s.store.CreateThread(ctx, domain.CreateThreadInput{
		UserID:          req.UserID,
		Title:           req.Title,
		Metadata:        req.Metadata,
		InitialMessages: req.InitialMessages,
	})
}

func (s *Service) CreateRun(ctx context.Context, req CreateRunRequest) (domain.RunRecord, error) {
	run, err := s.store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: req.ThreadID,
		UserID:   req.UserID,
		Goal:     req.Goal,
		Messages: req.Messages,
		Metadata: req.Metadata,
	})
	if err != nil {
		return domain.RunRecord{}, err
	}
	event, err := s.store.AppendRunEvent(ctx, domain.AppendRunEventInput{
		RunID:     run.RunID,
		ThreadID:  run.ThreadID,
		EventKind: "run.accepted",
		Message:   "Run accepted.",
		Payload:   domain.JSONMap{"status": string(run.Status)},
	})
	if err != nil {
		return domain.RunRecord{}, err
	}
	if err := s.bus.PublishRunEvent(ctx, event); err != nil {
		return domain.RunRecord{}, err
	}
	if err := s.bus.PublishJob(ctx, eventbus.Job{
		RunID:    run.RunID,
		ThreadID: run.ThreadID,
		UserID:   run.UserID,
		Goal:     run.Goal,
	}); err != nil {
		return domain.RunRecord{}, err
	}
	return run, nil
}
```

- [ ] **Step 4: Add deterministic worker stub**

Create `backend/controlplane/internal/worker/stub.go`:

```go
package worker

import (
	"context"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
)

type StubWorker struct {
	store runcontrol.Store
	bus   eventbus.Bus
}

func NewStubWorker(store runcontrol.Store, bus eventbus.Bus) *StubWorker {
	return &StubWorker{store: store, bus: bus}
}

func (w *StubWorker) RunJob(ctx context.Context, job eventbus.Job) error {
	if _, err := w.store.UpdateRunStatus(ctx, job.RunID, domain.RunStatusRunning, "", ""); err != nil {
		return err
	}
	events := []domain.AppendRunEventInput{
		{RunID: job.RunID, ThreadID: job.ThreadID, EventKind: "run.started", Message: "Planning started.", Payload: domain.JSONMap{"phase": "planning"}},
		{RunID: job.RunID, ThreadID: job.ThreadID, EventKind: "message.delta", Message: "Analyzing request.", Payload: domain.JSONMap{"delta": "Analyzing request."}},
		{RunID: job.RunID, ThreadID: job.ThreadID, EventKind: "artifact.created", Message: "Created deterministic report.", Payload: domain.JSONMap{"path": "outputs/stub-report.md"}},
	}
	for _, input := range events {
		select {
		case <-ctx.Done():
			_, _ = w.store.UpdateRunStatus(context.Background(), job.RunID, domain.RunStatusCanceled, "", "canceled")
			return ctx.Err()
		default:
		}
		event, err := w.store.AppendRunEvent(ctx, input)
		if err != nil {
			return err
		}
		if err := w.bus.PublishRunEvent(ctx, event); err != nil {
			return err
		}
		time.Sleep(5 * time.Millisecond)
	}
	if _, err := w.store.CreateArtifact(ctx, domain.CreateArtifactInput{
		RunID:      job.RunID,
		ThreadID:   job.ThreadID,
		Kind:       "report",
		Path:       "outputs/stub-report.md",
		Title:      "Deterministic Stub Report",
		MimeType:   "text/markdown",
		SizeBytes:  128,
		SHA256:     "stub-sha256",
		StorageURI: "memory://outputs/stub-report.md",
		Metadata:   domain.JSONMap{"worker": "stub"},
	}); err != nil {
		return err
	}
	if _, err := w.store.UpdateRunStatus(ctx, job.RunID, domain.RunStatusSucceeded, "Deterministic worker completed.", ""); err != nil {
		return err
	}
	event, err := w.store.AppendRunEvent(ctx, domain.AppendRunEventInput{
		RunID:     job.RunID,
		ThreadID:  job.ThreadID,
		EventKind: "run.completed",
		Message:   "Run completed.",
		Payload:   domain.JSONMap{"status": "succeeded"},
	})
	if err != nil {
		return err
	}
	return w.bus.PublishRunEvent(ctx, event)
}
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
cd backend/controlplane && go test ./internal/runcontrol ./internal/worker -count=1
```

Expected: PASS.

Commit:

```bash
git add backend/controlplane/internal/runcontrol backend/controlplane/internal/eventbus backend/controlplane/internal/worker
git commit -m "feat: add run dispatch service"
```

## Task 5: Implement HTTP v2 JSON Handlers

**Files:**
- Modify: `backend/controlplane/internal/httpapi/handlers.go`
- Modify: `backend/controlplane/internal/httpapi/handlers_test.go`
- Modify: `backend/controlplane/internal/app/app.go`

- [ ] **Step 1: Write failing HTTP v2 contract test**

Append to `backend/controlplane/internal/httpapi/handlers_test.go`:

```go
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

	createRunBody := strings.NewReader(`{"goal":"hello","messages":[{"role":"user","content":"hello"}]}`)
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
		RunID  string                    `json:"run_id"`
		Count  int                       `json:"count"`
		Events []domain.RunEventRecord   `json:"events"`
	}
	if err := json.Unmarshal(eventsRec.Body.Bytes(), &events); err != nil {
		t.Fatalf("decode events: %v", err)
	}
	if events.Count != 1 || events.Events[0].EventKind != "run.accepted" {
		t.Fatalf("events = %+v, want run.accepted", events)
	}
}
```

Add imports for `strings`, `eventbus`, `runcontrol`, `store`, and `domain`.

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
cd backend/controlplane && go test ./internal/httpapi -run TestV2ThreadRunArtifactHandlers -count=1
```

Expected: FAIL because `ServerDeps` lacks `Runs` and `Store`, and v2 routes do not exist.

- [ ] **Step 3: Implement JSON handlers**

Do not encode camelCase Go structs directly unless the structs have explicit
snake_case JSON tags. The frontend contract expects `thread_id`, `run_id`,
`event_kind`, `created_at`, and other snake_case fields.

Extend `ServerDeps`:

```go
type ServerDeps struct {
	Version string
	Runs    *runcontrol.Service
	Store   runcontrol.Store
}
```

Register routes:

```go
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
```

Implement each handler using the service/store methods and `writeJSON`. Use this response shape for events:

```go
type runEventsResponse struct {
	RunID  string                  `json:"run_id"`
	Count  int                     `json:"count"`
	Events []domain.RunEventRecord `json:"events"`
}
```

For cancellation, call `Store.UpdateRunStatus(ctx, runID, domain.RunStatusCanceled, "", reason)` and append a `run.canceled` event.

- [ ] **Step 4: Run handler tests**

Run:

```bash
cd backend/controlplane && go test ./internal/httpapi -count=1
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/controlplane/internal/httpapi backend/controlplane/internal/app
git commit -m "feat: add v2 control plane handlers"
```

## Task 6: Implement SSE Event Streaming

**Files:**
- Create: `backend/controlplane/internal/httpapi/sse.go`
- Create: `backend/controlplane/internal/httpapi/sse_test.go`
- Modify: `backend/controlplane/internal/httpapi/handlers.go`
- Modify: `backend/controlplane/internal/eventbus/memory.go`

- [ ] **Step 1: Write failing SSE test**

Create `backend/controlplane/internal/httpapi/sse_test.go`:

```go
package httpapi

import (
	"bufio"
	"context"
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

func TestRunEventsStreamEmitsRunEventEnvelope(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem, Bus: bus})

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{UserID: "user-1", Title: "stream"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "stream",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "stream"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/runs/"+run.RunID+"/events?stream=true", nil).WithContext(ctx)
	rec := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		router.ServeHTTP(rec, req)
		close(done)
	}()

	time.Sleep(20 * time.Millisecond)
	cancel()
	<-done

	body := rec.Body.String()
	if !strings.Contains(body, "event: run_event") {
		t.Fatalf("stream body missing run_event: %s", body)
	}
	if !strings.Contains(body, "run.accepted") {
		t.Fatalf("stream body missing run.accepted: %s", body)
	}

	scanner := bufio.NewScanner(strings.NewReader(body))
	foundData := false
	for scanner.Scan() {
		if strings.HasPrefix(scanner.Text(), "data:") {
			foundData = true
		}
	}
	if !foundData {
		t.Fatalf("stream body missing data lines: %s", body)
	}
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
cd backend/controlplane && go test ./internal/httpapi -run TestRunEventsStreamEmitsRunEventEnvelope -count=1
```

Expected: FAIL because streaming is not implemented.

- [ ] **Step 3: Implement SSE helpers**

Create `backend/controlplane/internal/httpapi/sse.go`:

```go
package httpapi

import (
	"encoding/json"
	"fmt"
	"net/http"
)

func writeSSE(w http.ResponseWriter, eventName string, payload any) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "event: %s\n", eventName); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
		return err
	}
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}
	return nil
}
```

Extend `ServerDeps` with:

```go
Bus interface {
	Events() <-chan domain.RunEventRecord
}
```

In `handleListRunEvents`, when `stream=true`, set:

```go
w.Header().Set("Content-Type", "text/event-stream")
w.Header().Set("Cache-Control", "no-cache")
w.Header().Set("Connection", "keep-alive")
```

Replay existing events for the run, then listen on the bus event channel until request context cancellation. Write matching events as:

```go
_ = writeSSE(w, "run_event", event)
```

Write heartbeat events on a 15-second ticker:

```go
_ = writeSSE(w, "heartbeat", map[string]string{"status": "ok"})
```

- [ ] **Step 4: Run SSE tests**

Run:

```bash
cd backend/controlplane && go test ./internal/httpapi -run TestRunEventsStreamEmitsRunEventEnvelope -count=1
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/controlplane/internal/httpapi backend/controlplane/internal/eventbus
git commit -m "feat: stream run events over sse"
```

## Task 7: Add Postgres Schema, sqlc Queries, and Repository

**Files:**
- Create: `backend/controlplane/sqlc.yaml`
- Create: `backend/controlplane/migrations/000001_run_control.up.sql`
- Create: `backend/controlplane/migrations/000001_run_control.down.sql`
- Create: `backend/controlplane/internal/store/schema.sql`
- Create: `backend/controlplane/internal/store/queries.sql`
- Generate: `backend/controlplane/internal/store/sqlc/`
- Create: `backend/controlplane/internal/store/postgres.go`
- Create: `backend/controlplane/internal/store/postgres_test.go`
- Modify: `backend/controlplane/go.mod`
- Modify: `backend/controlplane/Makefile`

- [ ] **Step 1: Write failing Postgres integration test**

Create `backend/controlplane/internal/store/postgres_test.go`:

```go
package store

import (
	"context"
	"os"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

func TestPostgresStoreThreadRunEventArtifactFlow(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	store := NewPostgresStore(pool)
	thread, err := store.CreateThread(ctx, domain.CreateThreadInput{
		UserID: "pg-user",
		Title:  "Postgres flow",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   "pg-user",
		Goal:     "persist run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "persist run"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	event, err := store.AppendRunEvent(ctx, domain.AppendRunEventInput{
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.accepted",
		Message:   "accepted",
		Payload:   domain.JSONMap{"ok": true},
	})
	if err != nil {
		t.Fatalf("AppendRunEvent: %v", err)
	}
	if event.Sequence < 1 {
		t.Fatalf("event sequence = %d, want >= 1", event.Sequence)
	}
}
```

- [ ] **Step 2: Run the test and verify it skips or fails correctly**

Run without a DSN:

```bash
cd backend/controlplane && go test ./internal/store -run TestPostgresStoreThreadRunEventArtifactFlow -count=1
```

Expected: SKIP with `ULTRA_CONTROL_TEST_DATABASE_URL is not set`.

Run with a DSN after `make postgres-init`:

```bash
cd backend/controlplane && ULTRA_CONTROL_TEST_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:55432/bisque_ultra_test go test ./internal/store -run TestPostgresStoreThreadRunEventArtifactFlow -count=1
```

Expected: FAIL because schema and `NewPostgresStore` are not implemented.

- [ ] **Step 3: Add schema and indexes**

Create `backend/controlplane/migrations/000001_run_control.up.sql` and copy it to `backend/controlplane/internal/store/schema.sql`:

```sql
CREATE TABLE IF NOT EXISTS control_threads (
  thread_id text PRIMARY KEY,
  user_id text NOT NULL,
  title text,
  status text NOT NULL,
  created_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL,
  latest_run_id text,
  checkpoint_id text,
  summary text,
  metadata jsonb NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS control_thread_messages (
  message_id text PRIMARY KEY,
  thread_id text NOT NULL REFERENCES control_threads(thread_id) ON DELETE CASCADE,
  role text NOT NULL,
  content text NOT NULL,
  created_at timestamptz NOT NULL,
  metadata jsonb NOT NULL DEFAULT '{}',
  run_id text
);

CREATE TABLE IF NOT EXISTS control_runs (
  run_id text PRIMARY KEY,
  thread_id text NOT NULL REFERENCES control_threads(thread_id) ON DELETE CASCADE,
  user_id text NOT NULL,
  goal text NOT NULL,
  status text NOT NULL,
  workflow_kind text NOT NULL,
  mode text,
  current_node text,
  parent_run_id text,
  planner_version text,
  agent_role text,
  trace_group_id text,
  checkpoint_id text,
  checkpoint_state jsonb,
  budget_state jsonb,
  response_text text,
  error text,
  created_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL,
  started_at timestamptz,
  completed_at timestamptz,
  metadata jsonb NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS control_run_events (
  event_id text PRIMARY KEY,
  sequence_number bigint NOT NULL,
  run_id text NOT NULL REFERENCES control_runs(run_id) ON DELETE CASCADE,
  thread_id text,
  event_kind text NOT NULL,
  event_type text,
  node_name text,
  task_id text,
  checkpoint_id text,
  scope_id text,
  agent_role text,
  level text,
  ts timestamptz NOT NULL,
  message text,
  payload jsonb NOT NULL DEFAULT '{}',
  UNIQUE(run_id, sequence_number)
);

CREATE TABLE IF NOT EXISTS control_artifacts (
  artifact_id text PRIMARY KEY,
  run_id text NOT NULL REFERENCES control_runs(run_id) ON DELETE CASCADE,
  thread_id text,
  kind text NOT NULL,
  path text,
  source_path text,
  preview_path text,
  title text,
  result_group_id text,
  mime_type text,
  size_bytes bigint,
  sha256 text,
  storage_uri text,
  tool_name text,
  category text,
  created_at timestamptz NOT NULL,
  updated_at timestamptz,
  metadata jsonb NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS control_runs_user_status_updated_idx ON control_runs(user_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS control_runs_thread_status_updated_idx ON control_runs(thread_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS control_run_events_run_sequence_idx ON control_run_events(run_id, sequence_number);
CREATE INDEX IF NOT EXISTS control_run_events_run_event_idx ON control_run_events(run_id, event_id);
CREATE INDEX IF NOT EXISTS control_artifacts_run_created_idx ON control_artifacts(run_id, created_at DESC);
CREATE INDEX IF NOT EXISTS control_artifacts_sha_idx ON control_artifacts(sha256);
```

Create `backend/controlplane/migrations/000001_run_control.down.sql`:

```sql
DROP TABLE IF EXISTS control_artifacts;
DROP TABLE IF EXISTS control_run_events;
DROP TABLE IF EXISTS control_runs;
DROP TABLE IF EXISTS control_thread_messages;
DROP TABLE IF EXISTS control_threads;
```

- [ ] **Step 4: Add sqlc config and queries**

Create `backend/controlplane/sqlc.yaml`:

```yaml
version: "2"
sql:
  - engine: "postgresql"
    schema: "internal/store/schema.sql"
    queries: "internal/store/queries.sql"
    gen:
      go:
        package: "sqlc"
        out: "internal/store/sqlc"
        sql_package: "pgx/v5"
        emit_json_tags: true
        emit_prepared_queries: true
```

Create `backend/controlplane/internal/store/queries.sql` with named queries for all store methods:

```sql
-- name: CreateThread :one
INSERT INTO control_threads (thread_id, user_id, title, status, created_at, updated_at, latest_run_id, checkpoint_id, summary, metadata)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
RETURNING *;

-- name: GetThread :one
SELECT * FROM control_threads WHERE thread_id = $1;

-- name: ListThreads :many
SELECT * FROM control_threads ORDER BY updated_at DESC LIMIT $1;

-- name: InsertThreadMessage :one
INSERT INTO control_thread_messages (message_id, thread_id, role, content, created_at, metadata, run_id)
VALUES ($1, $2, $3, $4, $5, $6, $7)
RETURNING *;

-- name: ListThreadMessages :many
SELECT * FROM control_thread_messages WHERE thread_id = $1 ORDER BY created_at ASC;

-- name: CreateRun :one
INSERT INTO control_runs (run_id, thread_id, user_id, goal, status, workflow_kind, mode, created_at, updated_at, metadata)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
RETURNING *;

-- name: SetThreadLatestRun :exec
UPDATE control_threads SET latest_run_id = $2, updated_at = $3 WHERE thread_id = $1;

-- name: GetRun :one
SELECT * FROM control_runs WHERE run_id = $1;

-- name: ListRuns :many
SELECT * FROM control_runs
WHERE ($1::text = '' OR thread_id = $1)
  AND ($2::text = '' OR status = $2)
ORDER BY updated_at DESC
LIMIT $3 OFFSET $4;

-- name: UpdateRunStatus :one
UPDATE control_runs
SET status = $2,
    response_text = NULLIF($3, ''),
    error = NULLIF($4, ''),
    updated_at = $5,
    started_at = CASE WHEN $2 = 'running' AND started_at IS NULL THEN $5 ELSE started_at END,
    completed_at = CASE WHEN $2 IN ('succeeded', 'failed', 'canceled') THEN $5 ELSE completed_at END
WHERE run_id = $1
RETURNING *;

-- name: NextRunEventSequence :one
SELECT COALESCE(MAX(sequence_number), 0) + 1 AS next_sequence FROM control_run_events WHERE run_id = $1;

-- name: AppendRunEvent :one
INSERT INTO control_run_events (event_id, sequence_number, run_id, thread_id, event_kind, ts, message, payload)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
RETURNING *;

-- name: ListRunEvents :many
SELECT * FROM control_run_events WHERE run_id = $1 ORDER BY sequence_number DESC LIMIT $2;

-- name: CreateArtifact :one
INSERT INTO control_artifacts (artifact_id, run_id, thread_id, kind, path, title, mime_type, size_bytes, sha256, storage_uri, created_at, updated_at, metadata)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
RETURNING *;

-- name: ListRunArtifacts :many
SELECT * FROM control_artifacts WHERE run_id = $1 ORDER BY created_at DESC LIMIT $2;

-- name: GetArtifact :one
SELECT * FROM control_artifacts WHERE artifact_id = $1;
```

- [ ] **Step 5: Generate sqlc and implement Postgres adapter**

Run:

```bash
cd backend/controlplane
go get github.com/jackc/pgx/v5@latest
go get github.com/jackc/pgx/v5/pgxpool@latest
go install github.com/sqlc-dev/sqlc/cmd/sqlc@latest
sqlc generate
```

Create `backend/controlplane/internal/store/postgres.go` mapping `sqlc` rows to `domain` records. Keep JSON conversion in small helpers:

```go
func jsonBytes(value domain.JSONMap) []byte {
	if value == nil {
		value = domain.JSONMap{}
	}
	data, _ := json.Marshal(value)
	return data
}

func jsonMap(data []byte) domain.JSONMap {
	if len(data) == 0 {
		return domain.JSONMap{}
	}
	var value domain.JSONMap
	if err := json.Unmarshal(data, &value); err != nil {
		return domain.JSONMap{}
	}
	return value
}
```

Use transactions for `CreateThread`, `CreateRun`, and `AppendRunEvent` so messages, latest-run pointers, and event sequence numbers are consistent.

- [ ] **Step 6: Run Postgres integration test**

Run:

```bash
make postgres-init
docker compose -f docker-compose.postgres.yml exec -T postgres psql -U postgres -d bisque_ultra_test < backend/controlplane/migrations/000001_run_control.up.sql
cd backend/controlplane
ULTRA_CONTROL_TEST_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:55432/bisque_ultra_test go test ./internal/store -run TestPostgresStoreThreadRunEventArtifactFlow -count=1
```

Expected: PASS after applying `migrations/000001_run_control.up.sql` to `bisque_ultra_test`.

- [ ] **Step 7: Commit**

```bash
git add backend/controlplane
git commit -m "feat: persist run control state in postgres"
```

## Task 8: Add NATS JetStream Bus

**Files:**
- Create: `backend/controlplane/internal/eventbus/nats.go`
- Create: `backend/controlplane/internal/eventbus/nats_test.go`
- Modify: `backend/controlplane/go.mod`
- Modify: `backend/controlplane/Makefile`
- Modify: `.env.example`

- [ ] **Step 1: Write NATS bus integration test**

Create `backend/controlplane/internal/eventbus/nats_test.go`:

```go
package eventbus

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

func TestNATSBusPublishesJobAndRunEvent(t *testing.T) {
	url := os.Getenv("ULTRA_CONTROL_TEST_NATS_URL")
	if url == "" {
		t.Skip("ULTRA_CONTROL_TEST_NATS_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	bus, err := NewNATSBus(ctx, NATSConfig{
		URL:       url,
		Stream:    "ULTRA_TEST",
		JobsSubject: "ultra.test.jobs",
		EventsSubject: "ultra.test.events",
	})
	if err != nil {
		t.Fatalf("NewNATSBus: %v", err)
	}
	defer bus.Close()

	if err := bus.PublishJob(ctx, Job{RunID: "run-1", ThreadID: "thread-1", UserID: "user-1", Goal: "test"}); err != nil {
		t.Fatalf("PublishJob: %v", err)
	}
	if err := bus.PublishRunEvent(ctx, domain.RunEventRecord{RunID: "run-1", EventKind: "run.accepted", Payload: domain.JSONMap{"ok": true}}); err != nil {
		t.Fatalf("PublishRunEvent: %v", err)
	}
}
```

- [ ] **Step 2: Run and verify skip or fail**

Run:

```bash
cd backend/controlplane && go test ./internal/eventbus -run TestNATSBusPublishesJobAndRunEvent -count=1
```

Expected without NATS URL: SKIP. With NATS URL: FAIL because `NewNATSBus` does not exist.

- [ ] **Step 3: Implement NATS JetStream bus**

Run:

```bash
cd backend/controlplane && go get github.com/nats-io/nats.go@latest
```

Create `backend/controlplane/internal/eventbus/nats.go`:

```go
package eventbus

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/nats-io/nats.go"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

type NATSConfig struct {
	URL           string
	Stream        string
	JobsSubject   string
	EventsSubject string
}

type NATSBus struct {
	conn *nats.Conn
	js   nats.JetStreamContext
	cfg  NATSConfig
}

func NewNATSBus(ctx context.Context, cfg NATSConfig) (*NATSBus, error) {
	_ = ctx
	conn, err := nats.Connect(cfg.URL)
	if err != nil {
		return nil, err
	}
	js, err := conn.JetStream()
	if err != nil {
		conn.Close()
		return nil, err
	}
	_, err = js.AddStream(&nats.StreamConfig{
		Name:     cfg.Stream,
		Subjects: []string{cfg.JobsSubject, cfg.EventsSubject},
		Storage:  nats.FileStorage,
	})
	if err != nil && !errors.Is(err, nats.ErrStreamNameAlreadyInUse) {
		conn.Close()
		return nil, err
	}
	return &NATSBus{conn: conn, js: js, cfg: cfg}, nil
}

func (b *NATSBus) PublishJob(ctx context.Context, job Job) error {
	return b.publish(ctx, b.cfg.JobsSubject, job)
}

func (b *NATSBus) PublishRunEvent(ctx context.Context, event domain.RunEventRecord) error {
	return b.publish(ctx, b.cfg.EventsSubject, event)
}

func (b *NATSBus) publish(ctx context.Context, subject string, value any) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	_, err = b.js.Publish(subject, data, nats.Context(ctx))
	return err
}

func (b *NATSBus) Close() {
	b.conn.Drain()
	b.conn.Close()
}
```

Add `.env.example` entries:

```bash
ULTRA_CONTROL_NATS_URL=nats://127.0.0.1:4222
ULTRA_CONTROL_NATS_STREAM=ULTRA_RUNS
ULTRA_CONTROL_NATS_JOBS_SUBJECT=ultra.runs.jobs
ULTRA_CONTROL_NATS_EVENTS_SUBJECT=ultra.runs.events
```

- [ ] **Step 4: Run eventbus tests**

Run:

```bash
cd backend/controlplane && go test ./internal/eventbus -count=1
```

Expected: PASS for memory bus tests and SKIP for NATS integration when no NATS URL is configured.

- [ ] **Step 5: Commit**

```bash
git add .env.example backend/controlplane
git commit -m "feat: add jetstream event bus"
```

## Task 9: Wire App Dependencies and Worker Loop

**Files:**
- Modify: `backend/controlplane/internal/app/app.go`
- Modify: `backend/controlplane/cmd/ultra-control/main.go`
- Create: `backend/controlplane/internal/app/app_test.go`

- [ ] **Step 1: Write failing app wiring test**

Create `backend/controlplane/internal/app/app_test.go`:

```go
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
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
cd backend/controlplane && go test ./internal/app -run TestNewAppServesHealth -count=1
```

Expected: FAIL because `New` and `App` do not exist.

- [ ] **Step 3: Implement app wiring**

Change `internal/app/app.go` to:

```go
package app

import (
	"net/http"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/config"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/httpapi"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/worker"
)

type App struct {
	Handler http.Handler
	Store   *store.MemoryStore
	Bus     *eventbus.MemoryBus
	Worker  *worker.StubWorker
}

func New(cfg config.Config) (*App, error) {
	memStore := store.NewMemoryStore()
	memBus := eventbus.NewMemoryBus()
	runService := runcontrol.NewService(memStore, memBus)
	stubWorker := worker.NewStubWorker(memStore, memBus)
	handler := httpapi.NewRouter(httpapi.ServerDeps{
		Version: cfg.AppVersion,
		Runs:    runService,
		Store:   memStore,
		Bus:     memBus,
	})
	return &App{
		Handler: handler,
		Store:   memStore,
		Bus:     memBus,
		Worker:  stubWorker,
	}, nil
}

func NewHTTPHandler(cfg config.Config) http.Handler {
	application, err := New(cfg)
	if err != nil {
		panic(err)
	}
	return application.Handler
}
```

Update `cmd/ultra-control/main.go` to call `app.New(cfg)`, use `application.Handler`, and start a goroutine that reads `application.Bus.Jobs()` and calls `application.Worker.RunJob(ctx, job)`.

- [ ] **Step 4: Run app tests**

Run:

```bash
cd backend/controlplane && go test ./internal/app ./cmd/ultra-control -count=1
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/controlplane/internal/app backend/controlplane/cmd/ultra-control
git commit -m "feat: wire control plane application"
```

## Task 10: Add Latency Target Tests and Benchmarks

**Files:**
- Create: `backend/controlplane/internal/latency/targets.go`
- Create: `backend/controlplane/internal/latency/targets_test.go`
- Create: `backend/controlplane/internal/httpapi/latency_test.go`

- [ ] **Step 1: Write failing latency target test**

Create `backend/controlplane/internal/latency/targets_test.go`:

```go
package latency

import (
	"testing"
	"time"
)

func TestTargetsMatchDesignSpec(t *testing.T) {
	t.Parallel()
	if HealthConfigP95 != 50*time.Millisecond {
		t.Fatalf("HealthConfigP95 = %s", HealthConfigP95)
	}
	if CreateRunAcceptedP95 != 200*time.Millisecond {
		t.Fatalf("CreateRunAcceptedP95 = %s", CreateRunAcceptedP95)
	}
	if FirstVisibleRunEventP95 != 300*time.Millisecond {
		t.Fatalf("FirstVisibleRunEventP95 = %s", FirstVisibleRunEventP95)
	}
	if EventFanoutAfterIngestP95 != 100*time.Millisecond {
		t.Fatalf("EventFanoutAfterIngestP95 = %s", EventFanoutAfterIngestP95)
	}
}
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
cd backend/controlplane && go test ./internal/latency -run TestTargetsMatchDesignSpec -count=1
```

Expected: FAIL because package `latency` does not exist.

- [ ] **Step 3: Add latency targets**

Create `backend/controlplane/internal/latency/targets.go`:

```go
package latency

import "time"

const (
	HealthConfigP95            = 50 * time.Millisecond
	AuthenticatedLightGETP95   = 100 * time.Millisecond
	CreateThreadP95            = 150 * time.Millisecond
	CreateRunAcceptedP95       = 200 * time.Millisecond
	SSEStreamOpenP95           = 250 * time.Millisecond
	FirstVisibleRunEventP95    = 300 * time.Millisecond
	PythonSupervisorDispatchP95 = 500 * time.Millisecond
	WarmSandboxCommandStartP95 = 250 * time.Millisecond
	WarmSandboxCommandOverheadP95 = 100 * time.Millisecond
	ColdSandboxLeaseP95        = 3 * time.Second
	ArtifactMetadataWriteP95   = 150 * time.Millisecond
	SmallArtifactSignedURLP95  = 100 * time.Millisecond
	CancelSignalAcceptedP95    = 200 * time.Millisecond
	CancelPropagatedP95        = 1 * time.Second
	EventFanoutAfterIngestP95   = 100 * time.Millisecond
	RunListSearchP95           = 200 * time.Millisecond
)
```

- [ ] **Step 4: Add lightweight handler latency tests**

Create `backend/controlplane/internal/httpapi/latency_test.go`:

```go
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
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
cd backend/controlplane && go test ./internal/latency ./internal/httpapi -count=1
```

Expected: PASS.

Commit:

```bash
git add backend/controlplane/internal/latency backend/controlplane/internal/httpapi/latency_test.go
git commit -m "test: add control plane latency targets"
```

## Task 11: Add Root Documentation and Environment Wiring

**Files:**
- Modify: `.env.example`
- Modify: `Makefile`
- Create: `backend/controlplane/README.md`

- [ ] **Step 1: Write failing documentation guard test**

Create `backend/controlplane/internal/config/config_test.go`:

```go
package config

import "testing"

func TestLoadDefaults(t *testing.T) {
	t.Setenv("ULTRA_CONTROL_APP_VERSION", "test-version")
	cfg := Load()
	if cfg.AppVersion != "test-version" {
		t.Fatalf("AppVersion = %q", cfg.AppVersion)
	}
	if cfg.HTTPAddr == "" {
		t.Fatalf("HTTPAddr must have default")
	}
}
```

- [ ] **Step 2: Run config test**

Run:

```bash
cd backend/controlplane && go test ./internal/config -run TestLoadDefaults -count=1
```

Expected: PASS if Task 1 config is correct.

- [ ] **Step 3: Add env examples**

Append to `.env.example`:

```bash
# Go control plane
ULTRA_CONTROL_APP_NAME=BisQue Ultra Control Plane
ULTRA_CONTROL_APP_VERSION=0.1.0
ULTRA_CONTROL_HTTP_ADDR=127.0.0.1:8088
ULTRA_CONTROL_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:55432/bisque_ultra
ULTRA_CONTROL_TEST_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:55432/bisque_ultra_test
ULTRA_CONTROL_NATS_URL=nats://127.0.0.1:4222
ULTRA_CONTROL_NATS_STREAM=ULTRA_RUNS
ULTRA_CONTROL_NATS_JOBS_SUBJECT=ultra.runs.jobs
ULTRA_CONTROL_NATS_EVENTS_SUBJECT=ultra.runs.events
```

Create `backend/controlplane/README.md`:

```markdown
# BisQue Ultra Go Control Plane

This service is the new Go run-control spine for BisQue Ultra.

## Run

```bash
make control-run
```

## Test

```bash
make control-test
```

## Postgres Integration

```bash
make postgres-init
docker compose -f docker-compose.postgres.yml exec -T postgres psql -U postgres -d bisque_ultra_test < backend/controlplane/migrations/000001_run_control.up.sql
cd backend/controlplane
ULTRA_CONTROL_TEST_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:55432/bisque_ultra_test go test ./internal/store -run TestPostgresStoreThreadRunEventArtifactFlow -count=1
```

## Contract

The public contract lives in `api/openapi.yaml`. Regenerate Go types with:

```bash
cd backend/controlplane
make generate
```
```

- [ ] **Step 4: Run documentation-adjacent checks**

Run:

```bash
make control-test
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add .env.example Makefile backend/controlplane/README.md backend/controlplane/internal/config/config_test.go
git commit -m "docs: document go control plane runtime"
```

## Task 12: Final Verification Gate

**Files:**
- No new files.

- [ ] **Step 1: Run Go formatting**

Run:

```bash
cd backend/controlplane && gofmt -w $(find . -name '*.go' -not -path './internal/openapi/generated.gen.go')
```

Expected: command exits 0.

- [ ] **Step 2: Regenerate code**

Run:

```bash
cd backend/controlplane && make generate
```

Expected: command exits 0 and generated file is up to date.

- [ ] **Step 3: Run all Go tests**

Run:

```bash
make control-test
```

Expected: PASS.

- [ ] **Step 4: Run frontend API contract test subset**

Run:

```bash
pnpm --dir frontend test:unit -- src/lib/api.test.ts
```

Expected: PASS. The Go OpenAPI contract test is the primary v2 route coverage in this plan; the frontend unit test confirms the existing frontend test harness still passes.

- [ ] **Step 5: Run doc/spec scans**

Run:

```bash
rg -n "TB""D|TO""DO|FIX""ME|place""holder" docs/superpowers/plans/2026-05-26-go-run-control-spine.md docs/superpowers/specs/2026-05-26-deep-agents-go-backend-design.md
git diff --check
```

Expected: no matches from `rg`; `git diff --check` exits 0.

- [ ] **Step 6: Commit final verification fixes**

If formatting or generated files changed:

```bash
git add backend/controlplane frontend/src/lib/api-v2.test.ts
git commit -m "chore: verify go control plane spine"
```

If no files changed, do not create an empty commit.

## Implementation Notes

- Keep handlers thin. Domain behavior belongs in `internal/runcontrol`.
- Keep the store interface small. Do not expose SQLC-generated types outside `internal/store`.
- Keep the in-memory store even after Postgres works. It gives fast unit tests and deterministic SSE tests.
- Use NATS JetStream for durable dispatch, but keep the memory bus for unit tests.
- Real Python Deep Agents runtime is intentionally outside this first plan. The deterministic worker proves the Go control plane can accept a run, dispatch work, stream events, persist artifacts, and finish a run.
- Warm sandbox implementation is intentionally outside this first plan. The sandbox plan should reuse the event and run lifecycle built here.

## Spec Coverage Review

Covered by this plan:

- Go API with v1 health/config/session and v2 threads/runs/events/artifacts.
- `net/http` + `chi` shell.
- OpenAPI and `oapi-codegen`.
- `pgxpool`, `sqlc`, Postgres migrations, and critical indexes.
- NATS JetStream subjects for dispatch and run events.
- Deterministic worker stub standing in for Python Deep Agents.
- Start run, stream run events, cancel run, list artifacts.
- Latency target constants and smoke latency tests.

Not covered by this plan because they are separate subsystems:

- Official Deep Agents runtime construction.
- Deep Agents memory/skills/policies.
- Async scientific subagents.
- Warm Docker sandbox pool.
- In-house GPU model services.

Those items should be planned immediately after this spine is implemented and verified.
