#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/monorepo_env.sh
source "$SCRIPT_DIR/monorepo_env.sh"

ROOT="$(monorepo_root)"
MONOREPO_ENV_FILE="$(resolve_monorepo_env_file)"
BISQUE_ROOT_URL="$(read_monorepo_env BISQUE_ROOT http://localhost:8080 "$MONOREPO_ENV_FILE")"
API_URL="$(read_monorepo_env ORCHESTRATOR_API_URL http://localhost:8000 "$MONOREPO_ENV_FILE")"
API_HOST="$(url_host "$API_URL")"
API_PORT="$(url_port "$API_URL")"

cleanup() {
  local exit_code
  exit_code=$?
  if [ -n "${frontend_pid:-}" ] && kill -0 "$frontend_pid" >/dev/null 2>&1; then
    kill "$frontend_pid" >/dev/null 2>&1 || true
  fi
  if [ -n "${api_pid:-}" ] && kill -0 "$api_pid" >/dev/null 2>&1; then
    kill "$api_pid" >/dev/null 2>&1 || true
  fi
  wait "${frontend_pid:-}" >/dev/null 2>&1 || true
  wait "${api_pid:-}" >/dev/null 2>&1 || true
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

echo "Using env file: $MONOREPO_ENV_FILE"
ensure_platform_running "$MONOREPO_ENV_FILE"
wait_for_http_ok "$BISQUE_ROOT_URL/image_service/formats" "BisQue platform" 90 2

(
  cd "$ROOT"
  uv run uvicorn src.api.main:app --reload --host "$API_HOST" --port "$API_PORT"
) &
api_pid=$!

(
  cd "$ROOT"
  pnpm --dir frontend dev
) &
frontend_pid=$!

echo "BisQue platform: $BISQUE_ROOT_URL"
echo "Bisque Ultra API: $API_URL"
echo "Bisque Ultra frontend: http://localhost:5173"

while kill -0 "$api_pid" >/dev/null 2>&1 && kill -0 "$frontend_pid" >/dev/null 2>&1; do
  sleep 1
done

wait "$api_pid"
wait "$frontend_pid"
