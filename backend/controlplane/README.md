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

## NATS Integration

```bash
docker run --rm -p 4223:4222 nats:2-alpine -js
cd backend/controlplane
ULTRA_CONTROL_TEST_NATS_URL=nats://127.0.0.1:4223 go test ./internal/eventbus -run TestNATSBusPublishesJobAndRunEvent -count=1
```

## Contract

The public contract lives in `api/openapi.yaml`. Regenerate Go types and sqlc queries with:

```bash
make control-generate
```
