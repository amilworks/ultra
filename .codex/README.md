# Ultra Codex Agent Workflow

Ultra uses project-scoped Codex agents for development work only. These agents
help build and review Ultra; they do not change the in-product Deep Agents
runtime by themselves.

The default workflow is:

1. Spawn a read-only recon swarm for the relevant slices.
2. Synthesize the findings in the parent thread.
3. Let exactly one single writer make code changes.
4. Spawn read-only reviewers against the diff.
5. Run at most two repair loops.
6. Verify with the narrowest relevant Ultra commands and report residual risk.

All custom agents inherit the parent model. The project config sets
`model_reasoning_effort = "xhigh"`, allows up to 8 bounded threads, gives slow
specialists up to 7200 seconds, and keeps `max_depth = 1` so child agents do not
recursively spawn more agents. Raise depth only after an explicit design change
explains the safety, cost, checkpointing, and reviewer stop controls.

The `natsDocs` MCP server points at `https://docs.nats.io/~gitbook/mcp` so NATS
specialists can consult current official NATS documentation. This is a docs MCP
endpoint, not a live broker connection.

## Recommended Prompt

Use the Ultra bounded swarm workflow for this task.

First phase: no edits. Spawn read-only agents for the relevant independent
slices:

- `ultra_explorer` for broad map and conventions
- `ultra_architect` for boundaries, invariants, and plan shape
- `ultra_control_plane` for Go run/control-plane paths
- `ultra_deepagents_runtime` for Python Deep Agents runtime paths
- `ultra_frontend_trace` for chat trace, artifact, and UX paths
- `ultra_security_data` for auth, uploads, storage, deletion, and provenance
- `ultra_test_verifier` for tests, CI, and acceptance evidence
- `ultra_trace_debugger` for exact run forensics, timelines, blast-radius maps,
  Postgres run-event traces, and evidence-led debugging
- `ultra_nats_expert` for NATS/JetStream subjects, streams, consumers,
  redelivery, queue diagnostics, and meaningful NATS benchmarks

Wait for the read-only agents, then synthesize one plan. Only
`ultra_implementer` may edit files. After implementation, spawn
`ultra_reviewer`, `ultra_security_data`, and any relevant domain specialist in
read-only mode. If they find blocking issues, run one focused repair pass with
`ultra_implementer`; stop after two repair loops and report remaining risks
honestly.

## Development Rules

- Keep parallel work read-heavy by default.
- Do not allow multiple agents to edit the same code at the same time.
- Treat auth, worker tokens, host paths, uploads, retention, BisQue secrets,
  data provenance, run leases, NATS subjects/streams/consumers, and artifacts
  as high-risk surfaces.
- Preserve Ultra's existing trace and frontend performance invariants: structural
  events and artifacts are the trust surface; raw token chatter is not.
- Prefer existing Make targets from `AGENTS.md` and `README.md`; when changing
  frontend V2 API calls, also check backend OpenAPI drift.

## Specialist Memory

Read-only specialists that debug durable systems keep explicit memory in
`.codex/memory`:

- `.codex/memory/ultra-trace-debugger.md`
- `.codex/memory/ultra-nats-expert.md`

The agents must read their memory before each relevant task. They should report
durable lessons as "Memory updates" rather than editing files themselves; the
parent or `ultra_implementer` applies accepted updates.

## When To Use The New Specialists

Use `ultra_trace_debugger` when the problem is unclear, high-risk, or tied to a
specific run. Give it the `run_id` when possible. It should reconstruct the
timeline from Postgres rows, event sequences, artifacts, token usage, leases,
heartbeats, and thread messages before proposing a fix.

Use `ultra_nats_expert` whenever work touches `backend/controlplane/internal/eventbus`,
`nats_worker.py`, worker dispatch, JetStream durability, replay, cancellation,
queue diagnostics, or local-stack NATS operations. Ask it for benchmark designs
that measure product behavior: recovery, redelivery, duplicate collapse, event
ingest latency, and user-visible trace health.
