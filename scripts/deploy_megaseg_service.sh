#!/usr/bin/env bash
set -euo pipefail

RELEASE_ID="${1:-}"
if [ -z "$RELEASE_ID" ]; then
  echo "Usage: $0 <release-id>" >&2
  exit 1
fi

MEGASEG_ROOT="${ULTRA_MEGASEG_ROOT:-/srv/ultra/megaseg-service}"
ENV_FILE="${MEGASEG_ENV_FILE:-/etc/ultra/megaseg-service.env}"
RELEASE_DIR="$MEGASEG_ROOT/releases/$RELEASE_ID"
CURRENT_LINK="$MEGASEG_ROOT/current"
UNIT_SOURCE="$RELEASE_DIR/deploy/systemd/ultra-megaseg.service"
UNIT_TARGET="/etc/systemd/system/ultra-megaseg.service"

if [ ! -d "$RELEASE_DIR" ]; then
  echo "Megaseg release directory not found: $RELEASE_DIR" >&2
  exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
  echo "Megaseg env file not found: $ENV_FILE" >&2
  exit 1
fi

if [ ! -f "$UNIT_SOURCE" ]; then
  echo "Megaseg systemd unit not found in release: $UNIT_SOURCE" >&2
  exit 1
fi

read_env_value() {
  local key="$1"
  local value
  value="$(awk -F= -v key="$key" '$1==key {print substr($0, index($0, "=") + 1); exit}' "$ENV_FILE")"
  printf '%s\n' "${value:-}"
}

wait_for_health() {
  local token="$1"
  local port="$2"
  local attempt
  for attempt in $(seq 1 90); do
    if curl -fsS -H "Authorization: Bearer ${token}" "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "Megaseg service healthy on port ${port}"
      return 0
    fi
    sleep 2
  done
  echo "Megaseg service failed health check on port ${port}" >&2
  return 1
}

MEGASEG_API_KEY="$(read_env_value MEGASEG_API_KEY)"
MEGASEG_SERVICE_PORT="$(read_env_value MEGASEG_SERVICE_PORT)"
MEGASEG_SERVICE_PORT="${MEGASEG_SERVICE_PORT:-8010}"

if [ -z "$MEGASEG_API_KEY" ]; then
  echo "MEGASEG_API_KEY is missing in $ENV_FILE" >&2
  exit 1
fi

echo "Preparing Megaseg release: $RELEASE_DIR"
mkdir -p "$MEGASEG_ROOT/releases" "$MEGASEG_ROOT/jobs" "$MEGASEG_ROOT/artifacts" "$MEGASEG_ROOT/models"
ln -sfn "$RELEASE_DIR" "$CURRENT_LINK"

install -m 0644 "$UNIT_SOURCE" "$UNIT_TARGET"

echo "Building ultra-megaseg:current image"
docker build \
  -t ultra-megaseg:current \
  -f "$CURRENT_LINK/services/megaseg_service/Dockerfile" \
  "$CURRENT_LINK"

systemctl daemon-reload
systemctl enable ultra-megaseg.service >/dev/null 2>&1 || true
systemctl restart ultra-megaseg.service

wait_for_health "$MEGASEG_API_KEY" "$MEGASEG_SERVICE_PORT"
echo "Megaseg deploy complete for $RELEASE_ID"
