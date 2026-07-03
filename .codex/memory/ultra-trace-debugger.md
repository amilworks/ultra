# Ultra Trace Debugger Memory

Purpose: persistent operating memory for `ultra_trace_debugger`. Read this
before every trace/debug task. Add durable lessons only when they would change a
future investigation.

## Core Rules

- Stay read-only by default: SELECT-only SQL, no NATS publishes, no file edits,
  no git writes, no live-service mutation.
- Exact run forensics begin with the `run_id`; if it is missing, first trace
  from `thread_id`, visible event IDs, artifact IDs, frontend URL state, or
  worker logs until the run is identified.
- Build a durable timeline from Postgres and JetStream before relying on logs or
  transcript prose.
- Preserve event ordering, sequence gaps, duplicate event IDs, worker IDs,
  lease tokens, task IDs, scope IDs, node names, and payload fields.
- Debug one falsifiable hypothesis at a time. A confident story without a
  falsification check is not enough for Ultra.
- Reports must separate facts, inferences, hypotheses, unknowns, and suggested
  repairs.

## Run Forensics Query Pack

Use `psql` with `-v run_id='...' -P pager=off`. Redact DSNs and secrets. For
production or staging DSNs, require explicit parent approval before connecting.

```sql
SELECT
  run_id, thread_id, user_id, goal, status, workflow_kind, mode, current_node,
  parent_run_id, planner_version, agent_role, trace_group_id, checkpoint_id,
  created_at, updated_at, started_at, completed_at, error, metadata
FROM control_runs
WHERE run_id = :'run_id';

SELECT
  run_id, worker_id, lease_token, lease_expires_at, created_at, updated_at
FROM control_run_leases
WHERE run_id = :'run_id';

SELECT
  worker_id, worker_kind, status, current_run_id, hostname, version,
  started_at, last_heartbeat_at, updated_at, metadata
FROM control_worker_heartbeats
WHERE current_run_id = :'run_id'
   OR metadata::text ILIKE '%' || :'run_id' || '%'
ORDER BY updated_at;

SELECT
  sequence_number, event_id, event_kind, event_type, agent_role, task_id,
  scope_id, node_name, level, ts, message, payload
FROM control_run_events
WHERE run_id = :'run_id'
ORDER BY sequence_number;

SELECT
  artifact_id, kind, title, path, source_path, preview_path, storage_uri,
  tool_name, category, created_at, updated_at, metadata
FROM control_artifacts
WHERE run_id = :'run_id'
ORDER BY created_at, artifact_id;

SELECT
  usage_event_id, model, day, input_tokens, output_tokens, total_tokens,
  occurred_at, created_at
FROM control_run_token_usage
WHERE run_id = :'run_id'
ORDER BY occurred_at, usage_event_id;

SELECT
  message_id, role, left(content, 1000) AS content_prefix, created_at,
  metadata, run_id
FROM control_thread_messages
WHERE thread_id = (
  SELECT thread_id FROM control_runs WHERE run_id = :'run_id'
)
ORDER BY created_at, message_id;
```

## Boundary Checklist

- Frontend: request payload, SSE/stream handling, trace hydration, artifact
  rendering, token usage display, cancellation UX.
- Go HTTP/API: route, auth principal, request validation, OpenAPI drift,
  audit/provenance event.
- Run control: idempotency key, run status transition, dispatch marker,
  cancellation, terminal-state guard.
- Store/sqlc: transaction boundaries, event sequence allocator, row locks,
  tenant filters, retry behavior.
- NATS/eventbus: subject, stream, message ID, durable consumer, ack/redelivery,
  queue diagnostics.
- Python worker: durable pull consumer, active run lock, lease renewals,
  heartbeat, tool execution, terminal status check, ack/nak timing.
- Sandbox/artifacts: output path, artifact promotion, storage URI/path,
  preview generation, deletion/retention interactions.

## Report Contract

Use this shape unless the parent asks for something narrower:

1. Symptom.
2. Evidence gathered.
3. Timeline.
4. Boundary trace.
5. Blast-radius map.
6. Root-cause hypothesis.
7. Falsification checks.
8. Recommended fix and tests.
9. Residual risks.
10. Memory updates for this file.

## Durable Lessons

- 2026-07-02: For Deep Agents live-run triage, separate backend stalls from
  compute-bound opaque `execute` calls. The healthy signature is contiguous
  `control_run_events.source_sequence`, fresh `control_run_leases` and
  `control_worker_heartbeats`, an active `ultra.sandbox.run=<run_id>` Docker
  container burning CPU, and empty `/outputs`/`control_artifacts` because the
  script only writes results at the end. In that case the UX problem is long-tool
  progress opacity, not NATS/control-plane delivery.
- 2026-07-01: When debugging missing NATS-backed run events, query both
  `sequence_number` and `source_sequence`. `sequence_number` is Go/Postgres
  ingest order; `source_sequence` is the Python worker producer order from the
  top-level NATS JSON `sequence`. The dangerous signature is a later terminal
  event arriving with a predecessor gap: fixed code returns
  `ErrRunEventPredecessorPending`, NAKs the message, and retries after the
  missing predecessor is stored instead of ACK-dropping earlier deltas/artifacts
  behind terminal state.
- 2026-07-01: For NATS-backed run-event gaps, distinguish local partition
  ordering from horizontal queue-group ordering. A single Go process now uses a
  nonblocking per-partition FIFO so a hot run cannot stop cold-run ingest, but
  multiple control-plane replicas in one JetStream deliver group can still
  process same-run events out of producer order. Trace symptoms include
  terminal `run.completed` before earlier deltas/artifacts, missing events after
  terminal status, and DB sequence numbers that reflect ingest order rather than
  Python worker source sequence.
- 2026-07-01: Run-event sequence gaps can be legitimate because
  `control_run_event_sequences` may advance before a failed append. Treat gaps
  as evidence to inspect, not automatic corruption.
- 2026-07-01: Retention/debug traces must inspect both `storage_path` and
  `storage_uri`; catalog rows can point at absolute `file://` sources while the
  path column stores only a basename.
- 2026-07-01: A Postgres `deadlock detected (SQLSTATE 40P01)` during
  `make control-integration` can come from concurrent package test binaries
  sharing `bisque_ultra_test`: app tests call `MigratePostgres` while store/app
  tests perform normal DML. Trace schema advisory locks as migration-only
  protection, then check Go package parallelism (`go test -p`) and shared DB
  names before blaming NATS.
- 2026-07-01: When tracing stuck Data Agent jobs, do not stop at status
  `queued`. Inspect `control_data_agent_job_leases` and the latest
  `control_data_agent_job_events` row. The dangerous signature is queued with
  no lease after an expired-lease recovery plus latest
  `data_agent.job.dispatch_failed`: that means NATS publish failed after the
  lease was cleared, and recovery must retry dispatch until a later
  `data_agent.job.dispatched` event appears.
- 2026-07-01: For Data Agent NATS worker failures in WorkOS mode, test the
  callback route through the real local app surface (`localhost:5174`) with the
  configured worker token. Owner headers alone can pass dev/unit tests but still
  hit the WorkOS browser-session gate; a healthy worker-token callback to a
  missing job should return tenant-scoped `404`, not `401 authentication
  required`.
- 2026-07-01: Run cancel traces should check ordering: durable
  `run.canceled` append/apply first, then run-event fanout, then NATS cancel
  signal. If NATS cancel publish fails, the run should still be terminal and the
  worker status monitor can observe the control plane's `canceled` status.
- 2026-07-01: Follow-up crash-window signature for Deep Agents runs: active
  `control_run_leases` row, run status `running`, and no
  `control_worker_heartbeats` row for the lease owner. Current recovery may wait
  for the full lease TTL rather than treating the missing heartbeat as stale;
  test `RecoverExpiredRunLeases` with a lease owner that never posted a first
  heartbeat.
- 2026-07-01: Missing first heartbeat for a run-lease owner is now a recovery
  signal after the stale lease-owner window. The red test shape is: create run,
  drain initial NATS job, acquire a long TTL run lease with `UpdatedAt` older
  than `WorkerHeartbeatStaleAfter`, do not insert a worker heartbeat row, then
  call `RecoverExpiredRunLeases`. Expected behavior is one `run.requeued`, old
  lease token rejected, one replacement NATS job, and metadata containing
  `worker_heartbeat_missing: true` with no fake zero
  `worker_last_heartbeat_at`.
