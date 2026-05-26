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

Reuse and upgrade the previous sandbox from:

```text
/Users/macbook/Documents/ultra_agent/src/ultra_agent/code_execution
/Users/macbook/Documents/ultra_agent/docker/sandbox.Dockerfile
/Users/macbook/Documents/ultra_agent/tests/test_code_execution.py
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

## Performance Design

### Critical Path

Fast path for a researcher prompt:

1. Frontend sends message to Go.
2. Go creates run record and admits it if budgets allow.
3. Go starts Python supervisor run.
4. Python uses warm Deep Agents runtime and warm sandbox lease.
5. Python streams tokens and tool events to Go.
6. Go fans events to frontend and persists them asynchronously.

### Performance Tactics

- Preload and cache Deep Agents app configuration.
- Prewarm sandbox containers from a pinned image.
- Use `docker exec` instead of container-per-command execution.
- Keep model services warm and separate from code sandboxes.
- Use async subagents for independent work.
- Use Go admission control to prevent worker exhaustion.
- Use bounded queues per org/project/user.
- Use artifact references instead of large message payloads.
- Use background memory consolidation to reduce hot-path latency.
- Use split deployments only when a subagent needs independent scaling.

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
- Python Deep Agents supervisor worker.
- Start run, stream tokens/events, cancel run.
- One curated model/tool call.
- Basic artifact save and download.

### Milestone 2: Warm Sandbox

- Port old Docker sandbox.
- Replace per-command `docker run --rm` behavior with warm container leases.
- Stage selected files into `/workspace`.
- Save reports, figures, tables, datasets, notebooks, and scripts as outputs.
- Add sandbox cleanup, TTL, metrics, and tests.

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

## Open Design Assumptions

- The first deployment target is self-hosted or lab-hosted, with optional cloud
  deployment later.
- Postgres is the durable metadata store.
- Artifact storage can start local and move to S3-compatible object storage.
- Sandbox network access is disabled by default.
- Heavy scientific inference runs outside the generic code sandbox.
- Deep Agents remains the primary Python agent harness unless official docs or
  production testing reveal a blocking issue.

These are assumptions for the draft, not unresolved placeholders.

## Review Checklist

- Deep Agents primitives are used directly rather than reimplemented.
- Go owns production truth and reliability.
- Python owns researcher-facing agent behavior.
- Sandboxes are fast enough for hundreds of tool calls.
- Memory is scoped and safe for multi-user research environments.
- Async subagents are part of the design, not an afterthought.
- Artifacts and provenance are first-class.
- The first milestone is small enough to implement and verify.
