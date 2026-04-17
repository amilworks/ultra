#!/usr/bin/env bash
set -euo pipefail

RELEASE_ID="${1:-}"
if [ -z "$RELEASE_ID" ]; then
  echo "Usage: $0 <release-id>" >&2
  exit 1
fi

CODEEXEC_ROOT="${ULTRA_CODEEXEC_ROOT:-/srv/ultra/codeexec-service}"
ENV_FILE="${CODEEXEC_ENV_FILE:-/etc/ultra/codeexec-service.env}"
RELEASE_DIR="$CODEEXEC_ROOT/releases/$RELEASE_ID"
CURRENT_LINK="$CODEEXEC_ROOT/current"
UNIT_SOURCE="$RELEASE_DIR/deploy/systemd/ultra-codeexec.service"
UNIT_TARGET="/etc/systemd/system/ultra-codeexec.service"

if [ ! -d "$RELEASE_DIR" ]; then
  echo "Code execution release directory not found: $RELEASE_DIR" >&2
  exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
  echo "Code execution env file not found: $ENV_FILE" >&2
  exit 1
fi

if [ ! -f "$UNIT_SOURCE" ]; then
  echo "Code execution systemd unit not found in release: $UNIT_SOURCE" >&2
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
      echo "Code execution service healthy on port ${port}"
      return 0
    fi
    sleep 2
  done
  echo "Code execution service failed health check on port ${port}" >&2
  return 1
}

CODEEXEC_API_KEY="$(read_env_value CODEEXEC_API_KEY)"
CODEEXEC_SERVICE_PORT="$(read_env_value CODEEXEC_SERVICE_PORT)"
CODEEXEC_SERVICE_PORT="${CODEEXEC_SERVICE_PORT:-8020}"

if [ -z "$CODEEXEC_API_KEY" ]; then
  echo "CODEEXEC_API_KEY is missing in $ENV_FILE" >&2
  exit 1
fi

echo "Preparing code execution release: $RELEASE_DIR"
mkdir -p "$CODEEXEC_ROOT/releases" "$CODEEXEC_ROOT/jobs" "$CODEEXEC_ROOT/artifacts"
ln -sfn "$RELEASE_DIR" "$CURRENT_LINK"

install -m 0644 "$UNIT_SOURCE" "$UNIT_TARGET"

echo "Building ultra-codeexec-job:current"
docker build \
  -t ultra-codeexec-job:current \
  -f "$CURRENT_LINK/services/codeexec_service/worker.Dockerfile" \
  "$CURRENT_LINK"

echo "Building ultra-codeexec-service:current"
docker build \
  -t ultra-codeexec-service:current \
  -f "$CURRENT_LINK/services/codeexec_service/Dockerfile" \
  "$CURRENT_LINK"

systemctl daemon-reload
systemctl enable ultra-codeexec.service >/dev/null 2>&1 || true
systemctl restart ultra-codeexec.service

wait_for_health "$CODEEXEC_API_KEY" "$CODEEXEC_SERVICE_PORT"
echo "Code execution deploy complete for $RELEASE_ID"
