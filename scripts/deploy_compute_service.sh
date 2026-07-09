#!/usr/bin/env bash
# Deploy the Ultra model-agnostic GPU compute service on a GPU node.
# Mirrors deploy_megaseg_service.sh: symlink release -> build image -> install
# unit -> restart -> health-gate. Idempotent; safe to run back to back.
set -euo pipefail

RELEASE_ID="${1:-}"
if [ -z "$RELEASE_ID" ]; then
  echo "Usage: $0 <release-id>" >&2
  exit 1
fi

COMPUTE_ROOT="${ULTRA_COMPUTE_ROOT:-/srv/ultra/compute-service}"
ENV_FILE="${ULTRA_COMPUTE_ENV_FILE:-/etc/ultra/compute-service.env}"
RELEASE_DIR="$COMPUTE_ROOT/releases/$RELEASE_ID"
CURRENT_LINK="$COMPUTE_ROOT/current"
UNIT_SOURCE="$RELEASE_DIR/deploy/systemd/ultra-compute.service"
UNIT_TARGET="/etc/systemd/system/ultra-compute.service"

if [ ! -d "$RELEASE_DIR" ]; then
  echo "Compute release directory not found: $RELEASE_DIR" >&2
  exit 1
fi
if [ ! -f "$ENV_FILE" ]; then
  echo "Compute env file not found: $ENV_FILE" >&2
  exit 1
fi
if [ ! -f "$UNIT_SOURCE" ]; then
  echo "Compute systemd unit not found in release: $UNIT_SOURCE" >&2
  exit 1
fi

read_env_value() {
  local key="$1"
  awk -F= -v key="$key" '$1==key {print substr($0, index($0, "=") + 1); exit}' "$ENV_FILE"
}

wait_for_health() {
  local token="$1" port="$2" attempt
  for attempt in $(seq 1 90); do
    if curl -fsS -H "Authorization: Bearer ${token}" "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "Compute service healthy on port ${port}"
      return 0
    fi
    sleep 2
  done
  echo "Compute service failed health check on port ${port}" >&2
  # Surface the last health body + recent logs to aid diagnosis.
  curl -sS -H "Authorization: Bearer ${token}" "http://127.0.0.1:${port}/health" || true
  journalctl -u ultra-compute.service -n 40 --no-pager || true
  return 1
}

API_KEY="$(read_env_value ULTRA_COMPUTE_API_KEY)"
PORT="$(read_env_value ULTRA_COMPUTE_SERVICE_PORT)"
PORT="${PORT:-8030}"
if [ -z "$API_KEY" ]; then
  echo "ULTRA_COMPUTE_API_KEY is missing in $ENV_FILE" >&2
  exit 1
fi

echo "Preparing compute release: $RELEASE_DIR"
mkdir -p "$COMPUTE_ROOT/releases" "$COMPUTE_ROOT/jobs" "$COMPUTE_ROOT/work"
ln -sfn "$RELEASE_DIR" "$CURRENT_LINK"

install -m 0644 "$UNIT_SOURCE" "$UNIT_TARGET"

echo "Building ultra-compute:current image (this bakes torch + yolov5 + deps)"
docker build \
  -t ultra-compute:current \
  -f "$CURRENT_LINK/services/compute_service/Dockerfile" \
  "$CURRENT_LINK"

systemctl daemon-reload
systemctl enable ultra-compute.service >/dev/null 2>&1 || true
systemctl restart ultra-compute.service

wait_for_health "$API_KEY" "$PORT"
echo "Compute deploy complete for $RELEASE_ID"
