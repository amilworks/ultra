# BisQue Docker Compose Setup

This Docker Compose setup allows you to easily run BisQue with persistent data storage and intelligent build caching.

## Quick Start

```bash
# Copy environment template
cp .env.example .env

# Build and start (uses Docker cache intelligently)
docker compose up --build -d
```

## Zero-Copy Legacy Store

If your data already lives on storage visible to the BisQue container, you can register it without uploading bytes. Mount the fixture or NFS export into the container, then point BisQue at that mounted path with `file:///...`.

```yaml
services:
  bisque:
    volumes:
      - ./fixtures/bisque_legacy_data:/mnt/bisque_legacy_data:ro
    environment:
      - BISQUE_EXTRA_FILE_STORES=legacy
      - BISQUE_STORE_LEGACY_MOUNTURL=file:///mnt/bisque_legacy_data
      - BISQUE_STORE_LEGACY_TOP=/mnt/bisque_legacy_data
      - BISQUE_STORE_LEGACY_READONLY=true
```

Use a comma-separated `BISQUE_EXTRA_FILE_STORES` list for additional stores, and define a matching `BISQUE_STORE_<NAME>_MOUNTURL` for each one. The name is uppercased in the env var prefix, so `legacy` maps to `BISQUE_STORE_LEGACY_*`.

## Daily Usage

```bash
# Normal startup
docker compose up -d

# Stop BisQue
docker compose down

# View logs
docker compose logs -f

# Restart without rebuilding
docker compose restart
```

## Local NFS Zero-Copy Smoke

Use the local override when you want to test zero-copy registration against a
read-only mounted tree inside the container.

```bash
docker compose -f docker-compose.yml -f docker-compose.nfs-local.yml up --build -d

# Inspect the generated store config inside the running container.
docker compose exec bisque grep -n "legacy_nfs" /source/config/site.cfg

# Dry-run the crawl first, then perform the registration in place.
docker compose exec bisque bq-path sync /mnt/bisque_legacy_data --dry-run
docker compose exec bisque bq-path sync /mnt/bisque_legacy_data --report /tmp/bisque_legacy_sync.jsonl
```

The override mounts `./test_fixtures/bisque_legacy_data` at
`/mnt/bisque_legacy_data:ro` and configures a read-only `legacy_nfs` file
store. Registered resources continue to point at the mounted `file://` paths
instead of copying bytes into `/source/data/uploads`.

## Complete cleanup and build
```bash
# stop container with cleaning persistent data
docker compose down -v
docker compose up --build -d
# or without using cache
docker compose build --no-cache && docker compose up -d
# to check the logs
docker compose logs -f
