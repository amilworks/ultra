#!/usr/bin/env bash
set -euo pipefail

RELEASE_SHA="${1:-}"
if [ -z "$RELEASE_SHA" ]; then
  echo "Usage: $0 <git-sha>" >&2
  exit 1
fi

ULTRA_RELEASE_ROOT="${ULTRA_RELEASE_ROOT:-/srv/ultra}"
RELEASE_DIR="$ULTRA_RELEASE_ROOT/releases/$RELEASE_SHA/backend"
CURRENT_LINK="$ULTRA_RELEASE_ROOT/current"

if [ ! -d "$RELEASE_DIR" ]; then
  echo "Backend release directory not found: $RELEASE_DIR" >&2
  exit 1
fi

wait_for_health() {
  local url="$1"
  local label="$2"
  local attempt
  for attempt in $(seq 1 90); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "$label healthy: $url"
      return 0
    fi
    sleep 2
  done
  echo "$label failed health check: $url" >&2
  return 1
}

echo "Preparing backend release: $RELEASE_DIR"
cd "$RELEASE_DIR"
uv sync --frozen

ln -sfn "$RELEASE_DIR" "$CURRENT_LINK"
systemctl daemon-reload

systemctl restart ultra-backend@1
wait_for_health "http://127.0.0.1:8001/v1/health" "ultra-backend@1"

systemctl restart ultra-backend@2
wait_for_health "http://127.0.0.1:8002/v1/health" "ultra-backend@2"

echo "Backend deploy complete for $RELEASE_SHA"
