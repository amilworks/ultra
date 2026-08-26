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

For production-shaped durability checks, run the deterministic soak gate and the live Postgres + NATS gate:

```bash
make control-soak
make control-integration
```

`control-soak` is fast and in-process; it stress-tests concurrent idempotent run creation, thousands of replayable worker events, terminal lifecycle updates, and durable thread messages. `control-integration` expects a local Postgres test database and NATS JetStream endpoint, then verifies worker events published through NATS are ingested into Postgres and replay correctly after the app is restarted, including a 1,200-delta autonomous-run event stream paged back from durable storage, worker terminal events published while the Go ingest process is offline, concurrent idempotent retries producing one durable run plus one JetStream job, completed assistant answers hydrating from durable thread messages after restart, durable worker heartbeat rows for fleet liveness, and run leases blocking duplicate workers across restart while still allowing recovery after lease expiry.

The control plane also runs an enabled-by-default expired-lease recovery loop. Configure it with `ULTRA_CONTROL_RUN_RECOVERY_ENABLED`, `ULTRA_CONTROL_RUN_RECOVERY_INTERVAL_SECONDS`, and `ULTRA_CONTROL_RUN_RECOVERY_BATCH_LIMIT`. The loop only requeues non-terminal runs whose Go-owned worker lease has expired, so a browser refresh or worker crash does not strand an autonomous chat while active workers keep their lease.

Notes rollout is deliberately fail-closed. All three gates default to `false` when absent: `ULTRA_CONTROL_NOTES_REQUIRE_EXPECTED_REVISION` makes browser PATCH require the revision it read, `ULTRA_CONTROL_MODEL_NOTES_READ_ENABLED` enables bounded run-scoped search/read, and `ULTRA_CONTROL_MODEL_NOTES_PROPOSALS_ENABLED` enables proposal review surfaces. Proposal creation and commit additionally require strict revision enforcement even if the proposal flag is accidentally enabled early. Values are read at control-plane startup and only explicit `true`, `1`, `on`, `yes`, or `enabled` turns a gate on.

Both model gates are exposed read-only as `features.model_notes_read` and `features.model_notes_proposals` from `/v1/config/public` and `/v2/config/public`; proposal availability is reported only when strict revision enforcement is active. Creating a new run with `selection_context.note_access` returns `503` while model reads are disabled. A matching idempotent replay of an already-created Notes run remains available during rollback or a kill-switch event and is reconciled against its stored canonical Note revisions instead of rereading mutable or deleted Notes. Notes access is a dedicated private-context run mode and cannot currently be combined with selected files, resources, datasets, knowledge, workflows, or tools. `allow_append_proposal` only lets the model create a browser-review proposal; it is never write authority by itself.

Use this production order: deploy the backend with all gates false (legacy browser PATCH is owner-scoped and still travels through atomic CAS), deploy the revision-aware frontend and refresh or age out old browser bundles, set `ULTRA_CONTROL_NOTES_REQUIRE_EXPECTED_REVISION=true`, load-test owner-scoped search against the production Notes corpus before enabling model reads, and finally enable model proposals. Docker Compose falls back to all three gates being true when those variables are unset for end-to-end local development; values in a local `.env` still override those fallbacks.

## Postgres Integration

```bash
make postgres-init
ULTRA_CONTROL_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:55432/bisque_ultra_test make control-migrate
cd backend/controlplane
ULTRA_CONTROL_TEST_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:55432/bisque_ultra_test go test ./internal/store -run TestPostgresStoreThreadRunEventArtifactFlow -count=1
```

Production startup verifies that the Postgres control-plane schema is present before serving. Run `make control-migrate` during deploys before starting `ultra-control`. The schema owns first-class admin users and organizations (`control_users`, `control_organizations`) and worker fleet heartbeats (`control_worker_heartbeats`) so local admin account management and autonomous-worker operations stay aligned with a future WorkOS-backed identity boundary.

## NATS Integration

```bash
docker run --rm -p 4223:4222 nats:2-alpine -js
cd backend/controlplane
ULTRA_CONTROL_TEST_NATS_URL=nats://127.0.0.1:4223 go test ./internal/eventbus -run TestNATSBusPublishesJobAndRunEvent -count=1
```

The NATS integration gate also verifies deterministic `Nats-Msg-Id` retry deduplication for jobs, worker events, and cancellation signals so uncertain publish retries do not enqueue duplicate autonomous work.

## Contract

The public contract lives in `api/openapi.yaml`. Regenerate Go types and sqlc queries with:

```bash
make control-generate
```
