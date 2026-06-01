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
