# Deep Agents Go Backend Design

Date: 2026-05-26
Status: Draft for user review

## Summary

BisQue Ultra should use a high-performance Go control plane around a Deep
Agents-native Python runtime. Go owns reliability, concurrency, scheduling,
events, artifacts, auth, quotas, and operational state. Python owns the
researcher-facing agent harness, context engineering, memory, MCP tools,
scientific tools, and sandbox-backed code execution.

The system should feel like ChatGPT or Codex for researchers, but with access
to in-house segmentation, detection, reconstruction, analysis, and reporting
tools. It must support long-running autonomous work, hundreds of tool calls,
multiple collaborating agents, fast code execution, resumable state, and strong
scientific provenance.

## Deep Agents Sources

This design is grounded in the current Deep Agents documentation:

- Context engineering: https://docs.langchain.com/oss/python/deepagents/context-engineering
- Backends: https://docs.langchain.com/oss/python/deepagents/backends
- Memory: https://docs.langchain.com/oss/python/deepagents/memory
- Async subagents: https://docs.langchain.com/oss/python/deepagents/async-subagents
- Sandboxes: https://docs.langchain.com/oss/python/deepagents/sandboxes
- Event streaming: https://docs.langchain.com/oss/python/deepagents/event-streaming
- Going to production: https://docs.langchain.com/oss/python/deepagents/going-to-production

The implementation should keep checking these docs before changing Deep Agents
integration details.

## Go Implementation Sources

The Go control plane should be grounded in boring, well-supported libraries:

- Go `net/http` routing enhancements: https://go.dev/blog/routing-enhancements
- chi router: https://github.com/go-chi/chi
- Gin router reference: https://github.com/gin-gonic/gin
- pgx PostgreSQL driver: https://github.com/jackc/pgx
- sqlc type-safe SQL generation: https://sqlc.dev/
- NATS JetStream: https://docs.nats.io/nats-concepts/jetstream
- Connect RPC: https://connectrpc.com/
- oapi-codegen: https://github.com/oapi-codegen/oapi-codegen
- OpenTelemetry Go: https://opentelemetry.io/docs/languages/go/
- Prometheus Go client: https://prometheus.io/docs/guides/go-application/

## Goals

- Provide a fast, reliable backend for researcher-heavy usage.
- Support long-running autonomous runs with hundreds of tool calls.
- Let multiple agents collaborate on complex scientific problems.
- Preserve a ChatGPT/Codex-like user experience.
- Run code quickly in isolated sandboxes.
- Integrate in-house scientific models as first-class tools.
- Maintain durable run state, events, outputs, approvals, and audit trails.
- Use Deep Agents features directly for context engineering, memory, backends,
  async subagents, sandbox execution, and event streaming.
- Keep the Python agent layer productive while Go handles production-grade
  reliability.

## Non-Goals

- Do not reimplement an agent framework in Go.
- Do not make LangGraph graph authoring the primary development interface for
  researchers or app developers.
- Do not let Deep Agents be the only source of truth for platform state.
- Do not run heavy GPU segmentation, detection, or reconstruction inside the
  general code sandbox.
- Do not expose a giant uncurated tool list to the model.

## Core Architecture

```text
React frontend
  -> Go control plane
      -> Postgres metadata store
      -> artifact/object store
      -> event stream fanout
      -> run queue and scheduler
      -> sandbox/container pool manager
      -> GPU/model job scheduler
      -> auth, RBAC, quotas, audit logs

      -> Python Deep Agents runtime
          -> supervisor agent
          -> async scientific subagents
          -> MCP tools
          -> in-house model tools
          -> memory, skills, policies
          -> warm sandbox backend

          -> model services
              -> segmentation
              -> detection
              -> reconstruction
              -> embeddings
              -> batch inference
```

Go and Python communicate through a typed internal protocol. The first
implementation can use HTTP plus server-sent events. If throughput or tail
latency demands it later, the internal transport can move to gRPC without
changing frontend contracts.

## Go Implementation Stack

Use a standard-library-first Go stack with small, focused dependencies.

### Public HTTP API

Use `net/http` with `chi/v5` for the public API. Go 1.22 added method-aware
routes and wildcards to `net/http`, so the standard library is now a stronger
baseline. `chi` remains the recommended router because it keeps the standard
`http.Handler` model, adds route groups and middleware ergonomics, and avoids a
framework-specific request context.

Do not use Gin as the default control-plane framework. Gin is fast and viable,
but router microbenchmarks are not the limiting factor for this product. The
control plane needs long-lived streams, standard middleware, generated
contracts, observability, auth, and clean modular handlers. Keeping handlers as
plain `net/http` makes the system easier to test, instrument, and evolve.

### API Contracts

Use `oapi-codegen` from OpenAPI specs for public route models and handlers.
Start with strict server generation where practical so handlers receive typed
request objects and return typed response objects. This should cover the
frontend-facing API and any stable internal HTTP contracts.

### Database

Use `pgx/v5` with `pgxpool` for PostgreSQL access. Prefer direct `pgx` over
`database/sql` because the system targets Postgres and benefits from
Postgres-specific features such as efficient batch queries, `COPY`, `LISTEN` /
`NOTIFY`, prepared statement caching, and lower overhead.

Use `sqlc` to generate type-safe query methods from hand-written SQL. This
keeps the hot path explicit and fast without introducing an ORM.

Use SQL migrations through `golang-migrate` or `goose`; choose one before
implementation and keep all schema changes versioned in the Go service.

### Queue and Event Bus

Use Postgres as source of truth and NATS JetStream as the high-throughput
durable event/work bus. JetStream should carry run events, worker dispatch,
subagent task state, and replayable streams. Go should persist critical state
before publishing or acknowledge only after state is recoverable.

Use plain Go channels only for in-process fanout. Do not use in-memory queues
for durable run scheduling.

### Internal RPC

Start internal Go-to-Python calls with typed HTTP plus server-sent events for
simplicity. Add Connect RPC when the Python bridge, model services, or sandbox
manager need a stronger typed RPC surface or bidirectional streaming. Connect
fits the `net/http` stack and can coexist with the public API.

### Observability

Use `log/slog` for structured logs, OpenTelemetry for traces and metrics, and
Prometheus for scrapeable service metrics. Every run, model call, tool call,
queue transition, database query class, sandbox command, and artifact write
should have trace/span metadata that includes `run_id`, `project_id`, and
`org_id` where safe.

### Configuration

Use typed config loaded once at startup and passed through constructors. Avoid
global mutable config in handlers. Keep per-run settings in the run record and
runtime context so retries and replays can use the same configuration.

## Component Responsibilities

### Go Control Plane

Go owns the platform truth:

- User, organization, project, session, thread, and run metadata.
- Run lifecycle: queued, running, waiting, blocked, succeeded, failed,
  canceled, retrying.
- Admission control, queueing, priorities, and concurrency limits.
- Per-run budgets for model calls, tool calls, wall time, CPU, memory, GPU
  jobs, and artifact storage.
- Event ingestion from Python and event streaming to the frontend.
- Artifact indexing, content hashing, signed downloads, and retention policy.
- Cancellation propagation to Python workers, sandboxes, and model services.
- Audit records for tool calls, model calls, approvals, artifacts, and memory
  writes.
- Health checks, metrics, logs, and operator controls.

Go should be boring, fast, and strict. It should not reason. It should make
every autonomous action observable and recoverable.

### Python Deep Agents Runtime

Python owns the agent experience:

- Build the Deep Agents supervisor agent.
- Configure Deep Agents backends, memory, skills, policies, permissions, and
  middleware.
- Register async scientific subagents.
- Provide curated MCP and internal scientific tools.
- Execute lightweight analysis and verification code in a sandbox.
- Call dedicated model services for heavy scientific inference.
- Stream Deep Agents events to Go.
- Produce structured final outputs and durable scientific reports.

Python can decide how to solve a research task. Go decides whether that run is
allowed, budgeted, visible, cancelable, and persisted.

### Scientific Model Services

In-house segmentation, detection, reconstruction, and embedding systems should
run as dedicated services or jobs, not inside the generic code sandbox.

Reasons:

- GPU scheduling and memory limits differ from code execution.
- Model versions and checkpoint hashes must be tracked explicitly.
- Heavy inference needs batching, queueing, and warm model workers.
- Results need scientific provenance independent of agent text.

The Deep Agents runtime sees these services as typed tools. The platform sees
them as auditable jobs.

## Deep Agents Integration

### Agent Construction

The main Python runtime should expose a supervisor Deep Agent with:

- A static system prompt for the research assistant role.
- A typed runtime context schema containing user, organization, project, thread,
  run, permissions, feature flags, and selected data handles.
- A curated tool list.
- A `CompositeBackend` for sandbox, memory, skills, policies, and outputs.
- Async subagents for parallel research and specialist work.
- Middleware for call limits, retries, event capture, and optional
  summarization.

Current Deep Agents backend docs say new code should pass backend instances
directly instead of deprecated backend factories. We should avoid copying the
old project's backend-factory pattern directly. Two acceptable designs are:

1. Construct a per-run Deep Agent instance with a concrete backend bound to the
   Go run context.
2. Implement a runtime-aware backend instance that resolves the active run from
   Deep Agents runtime/config helpers and leases the correct sandbox.

The first milestone should use option 1 because it is explicit, easy to test,
and avoids deprecated wiring. We can optimize construction later if profiling
shows graph construction overhead matters.

### Runtime Context

Deep Agents docs distinguish `thread_id` from runtime `context`. The design
should follow that exactly:

- `thread_id` scopes conversation history and checkpoints.
- `context` carries per-run data for tools and middleware.

Runtime context should include:

```text
assistant_id
org_id
user_id
project_id
thread_id
run_id
model_profile
selected_file_ids
selected_resource_uris
selected_dataset_uris
allowed_tool_packs
budget
auth_claims
artifact_root
sandbox_policy
```

Runtime context should propagate to subagents. Tools should read runtime
context rather than asking the model to pass hidden identifiers.

## Context Engineering

The system should use Deep Agents context features directly.

### Input Context

Always-loaded memory should stay small:

- User preferences.
- Project conventions.
- Critical lab policies.
- High-level active research context.

Detailed procedures should live in skills, not always-loaded memory. Skills
provide progressive disclosure and should be organized as focused workflows:

- Microscopy quantification.
- Segmentation QC.
- Detection benchmark review.
- Reconstruction workflow.
- Statistical analysis.
- Reproducible report writing.
- Literature synthesis.

### Runtime Context

Runtime context should carry operational facts that tools need but the model
does not need to see by default. Examples include user IDs, project IDs, feature
flags, tool permissions, object storage handles, and model service credentials.

If the model must see a derived instruction, add it through a controlled prompt
or middleware. Do not dump raw runtime context into the prompt.

### Offloading and Summarization

Long tool outputs should be written to files and referenced by path. The main
agent should retrieve details with `read_file`, `grep`, or dedicated output
tools when needed.

Deep Agents already offloads large tool call inputs/results and summarizes
history near context limits. We should keep those mechanisms enabled and add
the optional summarization tool middleware once the first long-run flows are
stable. The supervisor prompt should encourage summarization between phases:

- Planning complete.
- Data staged.
- Model inference complete.
- QC complete.
- Final report drafting begins.

### Context Isolation With Subagents

Use subagents for noisy, multi-step work:

- Literature reviewer.
- Methods critic.
- Imaging analyst.
- Segmentation QC analyst.
- Statistical analyst.
- Code/debugging analyst.
- Report synthesizer.

Each subagent must return concise synthesized findings to the supervisor and
write large intermediate artifacts to files. The supervisor should not inherit
raw search results, long logs, or full data dumps.

## Memory Design

Use Deep Agents memory, but scope it carefully.

### Memory Paths

```text
/memories/preferences.md
/memories/research_context.md
/memories/projects/<project_id>/notebook.md
/skills/<skill_name>/SKILL.md
/policies/org.md
/policies/tool_safety.md
```

### Backend Routing

Use `CompositeBackend`:

```text
default         -> warm sandbox backend for run/thread files and execution scratch
/memories/      -> StoreBackend, scoped by assistant_id/user_id/project_id
/skills/        -> StoreBackend, scoped by assistant_id/org_id or user_id
/policies/      -> StoreBackend, scoped by org_id, read-only to agents
/outputs/       -> platform artifact backend
```

The Deep Agents backend docs warn that multi-user deployments need explicit
namespace factories. Namespace keys must include user and organization scope
where appropriate. Do not rely on legacy assistant-wide defaults.

Deep Agents docs also note that internal offloaded tool results and conversation
history are written to the default backend. Because this design needs the
`execute` tool, the default backend is the warm sandbox backend. User-facing
outputs must still be promoted through `/outputs/` so Go can hash, index, and
retain them outside the sandbox.

### Write Policy

- User memory is writable by the user's agent.
- Project memory is writable only within that project.
- Organization policies are read-only to agents and populated by application
  code.
- Shared memory writes require human approval or an application-side policy.
- Memory writes are audited as tool calls.

### Background Consolidation

For performance, most memory consolidation should happen outside the hot path.
A background Deep Agent should periodically review recent completed runs and
merge durable lessons into memory. This follows the Deep Agents memory docs'
background consolidation pattern and avoids slowing active researchers.

## Async Subagents

Use Deep Agents async subagents for collaborative work.

### Starting Topology

Start with a single Python deployment for the supervisor and common subagents.
The async-subagents docs recommend this as the starting point because it avoids
network latency between agents.

### Split Topology

Move subagents to separate deployments when they need different scaling or
compute profiles:

- Literature subagent can scale on CPU.
- Code execution subagent needs sandbox capacity.
- Imaging subagent may need high-memory CPU workers.
- Reconstruction subagent may need GPU scheduling.

### Worker Pool Sizing

Each active supervisor and subagent consumes worker capacity. A run with one
supervisor and three concurrent subagents needs at least four worker slots.
The Go scheduler should account for this before admitting a run.

Go should track:

```text
run_id
supervisor_task_id
subagent_task_ids
subagent_name
status
started_at
finished_at
budget_consumed
last_event_id
```

### Async Task Behavior

The supervisor prompt must reinforce Deep Agents guidance:

- Launch async subagents for work that can proceed independently.
- Do not immediately poll in a loop after launching.
- Use full task IDs when checking or canceling.
- Always refresh task status before reporting it to the user.

Go should also enforce polling and concurrency limits externally.

## Sandbox Design

Reuse and upgrade the previous sandbox implementation:

```text
ultra_agent/src/ultra_agent/code_execution
ultra_agent/docker/sandbox.Dockerfile
ultra_agent/tests/test_code_execution.py
```

The existing implementation already provides:

- Deep Agents `BaseSandbox` integration.
- Docker isolation.
- `/workspace` working directory.
- No network by default.
- CPU, memory, pid, timeout, and output limits.
- Path escape prevention.
- Upload and download helpers.
- Workspace cleanup.

### Required Upgrade: Warm Sandbox Backend

The old implementation uses `docker run --rm` for each command. That is safe
but too slow for hundreds of tool calls. The new backend should keep a warm
container per run/thread/subagent and use `docker exec` for commands.

Lifecycle:

```text
Go admits run
  -> sandbox pool leases or creates container
  -> Python Deep Agent receives concrete sandbox backend
  -> execute tool uses docker exec inside warm container
  -> outputs are saved through platform tools
  -> sandbox expires on TTL or explicit cleanup
```

Container policy:

- One warm sandbox per active run by default.
- Optional separate sandbox per heavy async subagent.
- Mounted run workspace.
- Read-only root filesystem where practical.
- Writable `/workspace` and `/tmp`.
- Network disabled by default.
- No host secrets mounted.
- CPU, memory, pid, process, disk, and wall-time limits.
- Image pinned by digest in production.

### Sandbox as Tool

Follow the Deep Agents sandbox docs' "sandbox as tool" pattern. The agent runs
outside the sandbox and uses sandbox tools for execution. This keeps API keys
outside the sandbox, preserves agent state when sandbox execution fails, and
allows multiple sandboxes in parallel.

### Artifact Flow

The sandbox is not canonical project storage. It is a temporary execution
environment.

Files flow as:

```text
project artifact store
  -> stage selected files into /workspace
  -> execute analysis
  -> save figures/tables/datasets/reports via output tools
  -> Go indexes artifacts with hashes and provenance
```

The agent must save user-facing files as outputs before claiming they are
available.

## Tool Architecture

Tools should be curated into packs.

### Tool Pack Types

- Core workspace tools: stage files, save outputs, list artifacts.
- Code execution tools: Deep Agents filesystem and `execute`.
- Scientific model tools: segmentation, detection, reconstruction.
- Viewer tools: image metadata, slices, thumbnails, overlays.
- Literature tools: search, fetch, summarize, cite.
- Statistics tools: tests, power analysis, uncertainty reporting.
- Admin/debug tools: restricted to trusted operators.

### MCP Usage

MCP should be used for pluggable external or internal capabilities, but not as
the only boundary for high-value scientific operations. Important in-house
models should have typed platform contracts and then be exposed to Deep Agents
as tools.

### Tool Descriptions

Deep Agents context docs emphasize tool descriptions as prompt context. Every
tool must include:

- When to use it.
- When not to use it.
- Required inputs.
- Output shape.
- Cost or latency expectations.
- Whether it is deterministic.
- Whether it can create, mutate, or delete data.

## Event Streaming

Python should use Deep Agents event streaming and forward normalized events to
Go. Go then streams to the frontend through the existing run-event shape.

The frontend-facing event stream is part of the perceived-performance contract.
The backend should acknowledge a run quickly, then continuously show useful
movement even when the underlying research task takes minutes or hours.

Events should include:

```text
run.started
run.status_changed
message.delta
model.call.started
model.call.completed
tool.call.started
tool.call.delta
tool.call.completed
tool.call.failed
subagent.started
subagent.status_changed
subagent.completed
artifact.created
approval.requested
approval.resolved
memory.write.requested
memory.write.completed
run.completed
run.failed
run.canceled
```

Deep Agents event streams expose top-level messages, subagent streams, nested
subagents, and tool calls. The Python bridge should preserve that hierarchy in
event metadata.

### Event Pipeline

Use this event path:

```text
Python Deep Agents stream
  -> Python event normalizer
  -> Go event ingest endpoint
  -> Postgres run_events append
  -> NATS JetStream publish
  -> in-memory per-run SSE fanout
  -> frontend
```

Go should assign or validate monotonic per-producer sequence numbers. The SSE
handler should support reconnect with `Last-Event-ID` and replay from
Postgres or JetStream. UI-only token deltas may be coalesced for efficiency,
but lifecycle events, tool starts/completions, artifact creation, approvals,
and errors must not be dropped.

## Performance Design

### Critical Path

Fast path for a researcher prompt:

1. Frontend sends message to Go.
2. Go creates run record and admits it if budgets allow.
3. Go emits `run.accepted` and opens or reuses the SSE stream.
4. Go dispatches the Python supervisor run through the worker queue.
5. Python uses warm Deep Agents runtime and warm sandbox lease.
6. Python streams tokens and tool events to Go.
7. Go fans events to frontend and persists them asynchronously where safe.

### Latency Targets

Targets are local-lab or same-region deployment targets. Cloud and remote model
latency may raise model-specific timings, but the Go control plane should still
hit its own targets.

```text
health/config p95                         < 50 ms
authenticated lightweight GET p95          < 100 ms
create thread p95                          < 150 ms
create run record + run.accepted p95       < 200 ms
SSE stream open p95                        < 250 ms
first visible run event p95                < 300 ms
Python supervisor dispatch p95             < 500 ms
warm sandbox command start p95             < 250 ms
warm sandbox command overhead p95          < 100 ms beyond command runtime
cold sandbox lease p95                     < 3 s
artifact metadata write p95                < 150 ms
small artifact signed URL p95              < 100 ms
cancel signal accepted p95                 < 200 ms
cancel propagated to Python/sandbox p95    < 1 s
event fanout after ingest p95              < 100 ms
run list/search p95                        < 200 ms
```

Performance tests should track p50, p95, and p99. A run can take a long time;
the interface should not feel idle.

### Snappy UX Contract

The backend should make every user action feel acknowledged and alive:

- Return or stream `run.accepted` before starting expensive work.
- Show the current phase: planning, staging data, launching subagents, running
  code, calling a model service, generating outputs, writing final report.
- Stream subagent lifecycle events as soon as they launch and complete.
- Stream tool-call starts before tool execution finishes.
- Coalesce token deltas into small timed batches if needed, but never delay
  lifecycle events behind token traffic.
- Save large outputs as artifacts and stream lightweight references.
- Keep the frontend connected during long tasks with heartbeat events.
- Make cancellation immediate from the user's perspective, even if cleanup
  continues in the background.

### Performance Tactics

- Preload and cache Deep Agents app configuration.
- Preconstruct or memoize reusable Python agent components where Deep Agents
  allows it without violating per-run backend/context isolation.
- Prewarm sandbox containers from a pinned image.
- Use `docker exec` instead of container-per-command execution.
- Keep model services warm and separate from code sandboxes.
- Use async subagents for independent work.
- Use Go admission control to prevent worker exhaustion.
- Use bounded queues per org/project/user.
- Use artifact references instead of large message payloads.
- Use background memory consolidation to reduce hot-path latency.
- Use split deployments only when a subagent needs independent scaling.
- Use prepared SQL through `pgx` and generated query methods through `sqlc`.
- Keep event ingestion append-only and avoid cross-table transactions on token
  deltas.
- Batch or sample verbose low-value telemetry, but never sample audit-critical
  events.
- Use indexes specifically for active runs, run events by sequence, thread
  message history, artifact lookup, and run search.
- Load test with realistic long-running event volume, not only REST request
  bursts.

### Concurrency Controls

Go should enforce:

```text
max_active_runs_per_user
max_active_runs_per_org
max_subagents_per_run
max_sandboxes_per_run
max_model_calls_per_run
max_tool_calls_per_run
max_gpu_jobs_per_org
max_wall_time_per_run
max_artifact_bytes_per_run
```

Deep Agents middleware should also enforce model/tool call limits inside the
agent loop. Go limits protect the platform. Deep Agents limits protect each run
from runaway behavior.

## Reliability Design

### Run State

Go persists run state independently from Python. A Python crash must not erase
knowledge of:

- What the run was asked to do.
- What events already happened.
- Which artifacts exist.
- Which subagents were launched.
- Which budgets were consumed.
- Which approval was pending.
- Whether the run can retry, resume, or must fail.

### Hot-Path Persistence

Persist state according to event importance:

- Synchronous before acknowledgement: run creation, cancellation requests,
  approvals, artifact records, tool-call completion records, model-call
  completion records, final run status, and all error events.
- Buffered with short flush intervals: token deltas, progress messages, verbose
  stdout/stderr chunks, and heartbeat events.
- Stored as artifacts instead of event payloads: large logs, tables, images,
  notebooks, model outputs, and long tool results.

The event stream should remain responsive under high volume. If Postgres is
temporarily slow, Go should apply backpressure to Python workers and preserve
critical events rather than letting unbounded memory grow.

### Idempotency

All Python-to-Go writes should include:

```text
run_id
event_id
sequence_number
idempotency_key
producer_id
timestamp
```

Go should deduplicate repeated events.

### Queue Semantics

Use NATS JetStream for worker dispatch and event replay, with at-least-once
delivery and idempotent consumers. Every worker action that mutates platform
state must be safe to receive twice. Postgres remains the final authority for
run status, artifact records, budgets, approvals, and audit history.

### Cancellation

Cancellation starts in Go and propagates to:

- Python supervisor run.
- Active async subagents.
- Active sandbox commands.
- Active model-service jobs.
- Pending queue entries.

Sandbox command cancellation should send a process signal first, then kill the
container if the process does not exit within the grace window.

### Retry Policy

Retry only safe operations automatically:

- transient model provider failures
- network timeouts
- model service unavailable responses
- sandbox container startup failures before user code runs

Do not automatically retry destructive tools, memory writes, approvals, or
non-idempotent external side effects.

## Security

- Authenticate users in Go.
- Authorize every run, tool pack, project, and artifact in Go.
- Keep API keys and service secrets outside the sandbox.
- Disable sandbox network by default.
- Use an explicit auth proxy only if sandbox network access becomes necessary.
- Enforce read-only organization memory.
- Require approval for shared memory writes and sensitive actions.
- Block path traversal and host filesystem access.
- Pin sandbox images by digest in production.
- Log tool calls and memory writes for audit.

## Data Model

Minimum Go tables:

```text
users
organizations
projects
threads
messages
runs
run_events
subagent_tasks
tool_calls
model_calls
artifacts
artifact_versions
approvals
memory_audit_events
sandbox_leases
gpu_jobs
tool_registry
tool_pack_permissions
```

Critical indexes:

```text
runs(org_id, project_id, status, updated_at desc)
runs(user_id, status, updated_at desc)
run_events(run_id, sequence_number)
run_events(run_id, event_id)
subagent_tasks(run_id, status)
tool_calls(run_id, started_at)
model_calls(run_id, started_at)
artifacts(run_id, created_at desc)
artifacts(project_id, sha256)
sandbox_leases(run_id, status)
gpu_jobs(org_id, status, priority, created_at)
```

Artifacts should include:

```text
artifact_id
run_id
project_id
path
mime_type
size_bytes
sha256
storage_uri
created_by_tool
model_version
input_artifact_ids
metadata_json
created_at
```

## API Surface

The frontend already expects v1/v2/v3-style run, chat, upload, artifact, and
session APIs. The first backend should prioritize:

```text
GET  /v1/health
GET  /v1/config/public
GET  /v1/auth/session
POST /v2/threads
GET  /v2/threads
GET  /v2/threads/{thread_id}
GET  /v2/threads/{thread_id}/messages
POST /v2/threads/{thread_id}/runs
GET  /v2/runs/{run_id}
POST /v2/runs/{run_id}/cancel
GET  /v2/runs/{run_id}/events
GET  /v2/runs/{run_id}/artifacts
GET  /v2/artifacts/{artifact_id}
```

The Go backend can provide compatibility adapters for existing v1/v3 frontend
routes while the new v2 surface becomes the clean control-plane API.

## Milestones

### Milestone 1: Run Control Spine

- Go API with users/projects/threads/runs/events/artifacts.
- `net/http` + `chi` route shell.
- OpenAPI contract with `oapi-codegen` generated server types.
- `pgxpool` database connection and `sqlc` generated query package.
- NATS JetStream subjects for run dispatch and run events.
- Python Deep Agents supervisor worker.
- Start run, stream tokens/events, cancel run.
- One curated model/tool call.
- Basic artifact save and download.
- Latency test for run creation, first event, SSE fanout, and cancellation.

### Milestone 2: Warm Sandbox

- Port old Docker sandbox.
- Replace per-command `docker run --rm` behavior with warm container leases.
- Stage selected files into `/workspace`.
- Save reports, figures, tables, datasets, notebooks, and scripts as outputs.
- Add sandbox cleanup, TTL, metrics, and tests.
- Benchmark cold lease, warm command overhead, concurrent sandbox commands, and
  cancellation latency.

### Milestone 3: Async Scientific Subagents

- Add literature, imaging, stats, methods critic, and report subagents.
- Track subagent task IDs in Go.
- Stream subagent lifecycle and tool calls.
- Enforce per-run subagent limits.

### Milestone 4: Memory, Skills, Policies

- Configure `CompositeBackend`.
- Add user/project memory.
- Add read-only organization policy memory.
- Add focused scientific skills.
- Add background memory consolidation.

### Milestone 5: In-House Model Services

- Add typed tools for segmentation, detection, and reconstruction.
- Add GPU job scheduling and model-version provenance.
- Add QC/reporting workflows around model outputs.

### Milestone 6: Production Hardening

- Quotas and operator controls.
- Idempotent event ingestion.
- Retries and cancellation hardening.
- Security review for sandbox and memory writes.
- Load tests for concurrent agents and hundreds of tool calls.
- Evaluation suite for long-running scientific workflows.

## Testing Strategy

### Go

- Unit tests for schedulers, quotas, state transitions, and idempotency.
- API contract tests against frontend clients.
- Event-stream ordering and reconnect tests.
- Integration tests with Postgres and artifact storage.
- Load tests for many concurrent runs and event fanout.
- Latency budget tests for lightweight API routes, run creation, SSE startup,
  event fanout, cancellation, and artifact metadata writes.
- Queue redelivery tests proving NATS JetStream consumers are idempotent.
- Database query plan checks for active-run, event replay, artifact, and search
  queries.

### Python

- Deep Agents construction tests.
- Tool registry and permission tests.
- Sandbox path safety tests ported from the old project.
- Warm sandbox lifecycle tests.
- Event bridge tests using captured Deep Agents event streams.
- Memory namespace and read-only policy tests.
- Async subagent launch/check/cancel tests.

### End-to-End

- Long run with 200+ tool calls.
- Multi-subagent run with concurrent literature, code, and imaging tasks.
- Cancel run while sandbox command is active.
- Retry transient model service failure.
- Generate report with artifact provenance.
- Resume conversation with prior memory and project context.
- Measure first visible event, warm sandbox overhead, event throughput, and
  frontend reconnect/replay behavior during a long run.

## Open Design Assumptions

- The first deployment target is self-hosted or lab-hosted, with optional cloud
  deployment later.
- Postgres is the durable metadata store.
- NATS JetStream is the durable queue and replayable event bus.
- Artifact storage can start local and move to S3-compatible object storage.
- Sandbox network access is disabled by default.
- Heavy scientific inference runs outside the generic code sandbox.
- Public Go APIs use `net/http`, `chi`, OpenAPI, `oapi-codegen`, `pgx`, and
  `sqlc` unless implementation testing exposes a concrete blocker.
- Deep Agents remains the primary Python agent harness unless official docs or
  production testing reveal a blocking issue.

These are assumptions for the draft, not unresolved gaps.

## Review Checklist

- Deep Agents primitives are used directly rather than reimplemented.
- Go owns production truth and reliability.
- Python owns researcher-facing agent behavior.
- Sandboxes are fast enough for hundreds of tool calls.
- Latency targets are explicit and testable.
- The Go stack is standard-library-first and avoids framework lock-in.
- Memory is scoped and safe for multi-user research environments.
- Async subagents are part of the design, not an afterthought.
- Artifacts and provenance are first-class.
- The first milestone is small enough to implement and verify.
