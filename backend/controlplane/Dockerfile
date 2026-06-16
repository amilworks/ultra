# syntax=docker/dockerfile:1
#
# Go control plane (BisQue Ultra API). Build context = backend/controlplane.
#   docker build -t ultra-control:local backend/controlplane
# The Postgres schema is embedded in the binary (internal/store/schema.sql via
# go:embed), so the runtime image needs only the binary — no migrations dir.

# ---------- build ----------
FROM golang:1.25-bookworm AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
# Pure-Go build (pgx, NATS, chi — no cgo) so the runtime can be a slim image.
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" \
      -o /out/ultra-control ./cmd/ultra-control

# ---------- runtime ----------
FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
  && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/ultra-control /usr/local/bin/ultra-control

# Apply the embedded schema (idempotent; retries until Postgres accepts), then serve.
COPY <<'EOF' /usr/local/bin/entrypoint.sh
#!/bin/sh
set -e
echo "[control] applying schema (idempotent)..."
until ultra-control migrate; do
  echo "[control] database not ready, retrying in 2s..."
  sleep 2
done
echo "[control] starting API on ${ULTRA_CONTROL_HTTP_ADDR:-0.0.0.0:8000}"
exec ultra-control serve
EOF
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
