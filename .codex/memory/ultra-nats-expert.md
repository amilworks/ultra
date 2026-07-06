# Ultra NATS Expert Memory

Purpose: persistent operating memory for `ultra_nats_expert`. Read this before
every NATS, JetStream, worker-dispatch, redelivery, or queue-diagnostics task.

## Official Reference

- Use the project `natsDocs` MCP server first for current NATS documentation.
- The configured endpoint is `https://docs.nats.io/~gitbook/mcp`; it is a
  GitBook documentation MCP endpoint, not an operational NATS server.
- Prefer official NATS and JetStream docs over memory for protocol semantics,
  consumer options, clustering, security, monitoring, and benchmark guidance.

## Ultra Surface Map

- Go eventbus: `backend/controlplane/internal/eventbus/nats.go`.
- App wiring and integration gates:
  `backend/controlplane/internal/app/app.go` and
  `backend/controlplane/internal/app/app_integration_test.go`.
- Python worker:
  `backend/deepagents_runtime/src/ultra_deepagents/nats_worker.py`.
- Local stack: `docker-compose.yml`, `Makefile`, and scripts under `scripts/`.
- Current Docker image: `nats:2.10-alpine` with JetStream data persisted in the
  `ultra-nats` volume.

## Current Implementation Notes

- Ultra uses JetStream for durable job/event transport and Postgres for
  authoritative run state.
- The Go bus sets a 24 hour duplicate window, infinite reconnects, 2 second
  reconnect wait, and 5 second drain timeout.
- The stream includes jobs, data-agent jobs, events, and cancellation subjects.
- Job, data-agent job, run-event, and cancel publishes use NATS message IDs for
  deduplication.
- Run-event ingest uses a durable consumer, explicit acks, deliver-all replay,
  instant replay, `AckWait` around 60 seconds, `MaxDeliver` around 20,
  partitioned workers, and delayed negative acks.
- Queue diagnostics inspect stream and consumer state; use them before guessing
  about lag or stalled workers.
- The Python worker uses a durable pull consumer, control-plane leases,
  heartbeat updates, active-run locking, ack extension, terminal-state checks,
  and reconnect supervision.
- Correctness is at-least-once: idempotency keys, durable event IDs, lease
  tokens, terminal-state guards, and replay-safe handlers are the real safety
  boundary.

## Read-Only Diagnostics

Prefer existing project commands:

```sh
make status-control-stack
make control-integration
make deepagents-worker-test
```

When the NATS CLI is available and the target is a local/test server, read-only
inspection may include:

```sh
nats --server "$ULTRA_CONTROL_NATS_URL" stream info "$ULTRA_CONTROL_NATS_STREAM"
nats --server "$ULTRA_CONTROL_NATS_URL" consumer info "$ULTRA_CONTROL_NATS_STREAM" "$DURABLE"
nats --server "$ULTRA_CONTROL_NATS_URL" server report jetstream
```

Never purge streams, delete consumers, publish probe messages, alter server
config, or operate on production/staging NATS without explicit parent approval.

## Benchmark Discipline

Benchmark Ultra behavior, not abstract broker throughput:

- Concurrent run submissions with repeated idempotency keys.
- Worker outage and restart while jobs are pending.
- Control-plane outage and replay while worker events are published.
- 1k and 10k+ run-event bursts, including token deltas and tool events.
- Multiple workers in the queue group with lease contention.
- Data-agent job fanout and cancellation races.
- High artifact/event fanout followed by frontend hydration.

Track throughput, p50/p95/p99 publish latency, ingest latency, pending messages,
in-flight messages, redeliveries, ack-floor movement, stream bytes, CPU/memory,
Postgres append time, duplicate-event collapse, and user-visible recovery time.

## Review Traps

- Do not assume ordering across subjects, workers, or queue subscribers.
- Check subject overlap when stream reconciliation changes.
- Check duplicate-window mismatches when retry/idempotency behavior changes.
- Ack-on-error is usually a data-loss bug unless the handler proves durable
  idempotent persistence already happened.
- Unbounded redelivery can hide poison messages and make recovery noisy.
- Consumer recreation can reset delivery semantics; explain the migration path.
- `MaxAckPending`, pull batch size, worker concurrency, and Postgres ingest
  throughput must be reasoned about together.

## Durable Lessons

- 2026-07-02: Local run `run_29d8d9694720dd7cc78e2722963086ed` validated the
  partitioned run-event/source-sequence fix under a real Builder workload:
  partition `36`, subject `ultra.runs.events.p.36`, partition durable
  `ultra-control-event-ingest-p-36` caught up with
  `pending=0/ack_pending=0/redelivered=0`, and Postgres stored contiguous
  `source_sequence=1..1151` with no gaps or duplicates. Separate signal:
  active Deep Agents job had `ack_pending=1` and historical `redelivered=3`;
  active-run duplicate handling protected execution, but ack-progress/redelivery
  observability remains worth follow-up.
- 2026-07-01: Go now preserves the Python worker's top-level producer
  `sequence` as `AppendRunEventInput.SourceSequence` and persists it to
  `control_run_events.source_sequence`. `runcontrol.Service.IngestRunEvent`
  treats a source sequence gap as `ErrRunEventPredecessorPending`, causing the
  NATS event handler to NAK/redeliver instead of ACK-dropping source-earlier
  deltas after a terminal event. This is a safety rail, not the final
  horizontal ordering architecture: queue-group replicas can still receive the
  same run's messages on different processes, so the frontier follow-up remains
  broker-visible deterministic run partitions with one strict in-flight
  consumer per partition.
- 2026-07-01: One slow/hot run must not block NATS event ingest for unrelated
  runs. The old `SubscribeAllRunEvents` callback sent into a bounded partition
  channel; when a hot partition filled, the single NATS callback stopped before
  dispatching cold-run events to idle partitions. Red proof:
  `TestNATSRunEventConsumerDoesNotLetFullHotPartitionBlockColdRun` with a live
  JetStream consumer. Fix: per-partition FIFO buffers preserve in-process
  per-run order while making callback handoff nonblocking; JetStream
  `MaxAckPending` remains the outer delivered-message bound.
- 2026-07-01: Open architecture follow-up: per-run ordering is still only local
  to one Go process. Multiple control-plane replicas sharing the same JetStream
  push queue group can process later events for a run while another replica is
  blocked on an earlier event, and Go currently does not persist the Python
  source `sequence` field from worker events. Do not claim horizontal event
  ingest is frontier-level until this is tested and solved with key ownership,
  source-sequence buffering, or another explicit ordering contract.
- 2026-07-01: Benchmark follow-up: add a live NATS -> Go event ingest ->
  Postgres backlog-drain performance regression. Existing Postgres and
  runcontrol benchmarks bypass JetStream; existing app integration proves
  correctness but not a drain-time budget after control-plane outage.
- 2026-07-01: Treat the NATS docs MCP as the first external reference for this
  agent, but keep operational diagnostics against Ultra's local/test NATS
  broker and project queue diagnostics.
- 2026-07-01: Ultra's NATS correctness claim should always be phrased as
  at-least-once plus idempotent Postgres/control-plane handling, not exactly-once
  messaging.
- 2026-07-01: Queue diagnostics must distinguish durable consumer existence
  from active processing. For push consumers use JetStream `PushBound`; for pull
  consumers use waiting pull requests or in-flight ack-pending messages. Pending
  messages with no waiting/in-flight work should show a missing worker, not a
  healthy consumer.
- 2026-07-01: Deep Agents worker consumer reconciliation must compare
  `AckPolicy` explicitly. Accepting an existing non-explicit-ack durable
  bypasses ack/nak, ack extension, lease-conflict redelivery, and publish-failure
  redelivery paths.
- 2026-07-01: When the Python worker receives and acks a job because the
  control plane reports `not_found` or terminal status, publish deterministic
  `run.worker_skipped` before acking. If the diagnostic publish fails before a
  run lock is acquired, NAK the job instead of acking so JetStream can redeliver
  rather than silently losing forensic evidence.
- 2026-07-01: `make control-integration` must serialize the selected Go package
  test binaries with `go test -p=1` while they share one live
  `bisque_ultra_test` database. The schema advisory lock protects
  migration-vs-migration only; it does not protect migration DDL from normal
  test DML in another package.
- 2026-07-01: Data Agent worker consumer reconciliation must compare
  `AckPolicy` explicitly too. A legacy/manual `AckNone` durable can otherwise
  look config-compatible while bypassing explicit ack/nak, lease-conflict
  redelivery, and publish-failure retry assumptions.
- 2026-07-01: Expired Data Agent job lease recovery must not treat
  `data_agent.job.requeued` as proof that the NATS dispatch happened. If
  publishing the recovered job fails after the lease is deleted, append
  `data_agent.job.dispatch_failed` and keep queued/no-lease jobs whose latest
  event is `dispatch_failed` recoverable until a later
  `data_agent.job.dispatched` event closes the retry cursor.
- 2026-07-01: Data Agent skipped-delivery handling now mirrors the Deep Agents
  skip-audit rule. Before ACKing a terminal delivery, the Python Data Agent NATS
  worker appends deterministic `data_agent.job.skipped` through
  `POST /v2/data-agent/jobs/{job_id}/events`; append failure NAKs for
  redelivery. True `not_found` jobs cannot persist job-timeline events because
  of the job FK, so the safe behavior remains NAK/retry until an orphan/dead
  letter audit surface exists.
- 2026-07-01: In WorkOS mode, Data Agent worker callbacks must use the shared
  worker token plus owner headers. The control plane treats GET job, lease,
  status, events, and outputs callbacks as worker-scoped; the request principal
  for those callbacks comes from `X-Ultra-User-Id`/`X-Ultra-Org-Id`, while the
  token bypasses the WorkOS browser-session gate.
- 2026-07-01: Run cancellation must be durable-state-first. `run.canceled` in
  Postgres/control run events is the source of truth; NATS cancel publish and
  run-event fanout are post-durable best-effort latency paths. A cancel-subject
  publish failure must not keep a run non-terminal, and a canceled event append
  failure must not publish a cancel signal.
- 2026-07-01: Follow-up candidate: live NATS stream reconciliation should be
  monotonic over subjects. Starting a secondary bus that omits
  `DataAgentJobsSubject` must not remove the existing Data Agent jobs subject
  from the shared stream; add a live regression that checks the union of stream
  subjects and proves `PublishDataAgentJob` still succeeds after secondary bus
  startup.
- 2026-07-01: Stream reconciliation is now monotonic over subjects. The red
  live NATS regression was a full bus creating jobs/events/cancel/data-agent
  subjects, followed by a secondary worker-style bus without
  `DataAgentJobsSubject`; before the fix, the stream subjects lost
  `*.data_agent.jobs`. `ensureNATSStream` now bases updates on `StreamInfo.Config`,
  merges existing+requested subjects, preserves unrelated stream settings such
  as retention/storage/description, and raises the duplicate window to Ultra's
  required 24h when an older stream is shorter. Verification:
  `ULTRA_CONTROL_TEST_NATS_URL=nats://127.0.0.1:4222 go test ./internal/eventbus -count=1`,
  `go test ./... -count=1`, and `make control-integration`.
- 2026-07-05: Data Agent NATS payloads are not a browser-user path, but current
  local/deploy evidence shows shared or unauthenticated NATS access rather than
  subject-isolated publisher/subscriber credentials. Treat
  `DataAgentJobEnvelope` fields as untrusted in Python workers: validate
  `job_id` and resource-derived path segments before any filesystem staging.
