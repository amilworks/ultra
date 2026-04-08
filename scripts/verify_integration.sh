#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/monorepo_env.sh
source "$SCRIPT_DIR/monorepo_env.sh"

ROOT="$(monorepo_root)"
MONOREPO_ENV_FILE="$(resolve_monorepo_env_file)"
BISQUE_ROOT_URL="$(read_monorepo_env BISQUE_ROOT http://localhost:8080 "$MONOREPO_ENV_FILE")"
API_URL="$(read_monorepo_env ORCHESTRATOR_API_URL http://localhost:8000 "$MONOREPO_ENV_FILE")"
AUTH_MODE="$(read_monorepo_env BISQUE_AUTH_MODE dual "$MONOREPO_ENV_FILE")"
API_HOST="$(url_host "$API_URL")"
API_PORT="$(url_port "$API_URL")"
TMP_DIR="$(mktemp -d)"
API_LOG="$TMP_DIR/bisque-ultra-api.log"

cleanup() {
  local exit_code
  exit_code=$?
  if [ -n "${api_pid:-}" ] && kill -0 "$api_pid" >/dev/null 2>&1; then
    kill "$api_pid" >/dev/null 2>&1 || true
  fi
  wait "${api_pid:-}" >/dev/null 2>&1 || true
  rm -rf "$TMP_DIR"
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

echo "Using env file: $MONOREPO_ENV_FILE"
ensure_platform_running "$MONOREPO_ENV_FILE"
wait_for_http_ok "$BISQUE_ROOT_URL/image_service/formats" "BisQue platform" 90 2

(
  cd "$ROOT"
  uv run uvicorn src.api.main:app --host "$API_HOST" --port "$API_PORT" >"$API_LOG" 2>&1
) &
api_pid=$!
wait_for_http_ok "$API_URL/v1/health" "Bisque Ultra API" 60 2

config_payload="$(curl -fsS "$API_URL/v1/config/public")"
CONFIG_PAYLOAD="$config_payload" python3 - "$BISQUE_ROOT_URL" <<'PY'
import json
import os
import sys

expected_root = sys.argv[1].rstrip("/")
payload = json.loads(os.environ["CONFIG_PAYLOAD"])
actual_root = str(payload.get("bisque_root") or "").rstrip("/")
if actual_root != expected_root:
    raise SystemExit(f"Expected bisque_root={expected_root}, got {actual_root}")
if payload.get("bisque_auth_enabled") is not True:
    raise SystemExit("Public config did not report BisQue auth as enabled.")
PY

if [ "$AUTH_MODE" = "oidc" ]; then
  session_payload="$(curl -fsS "$API_URL/v1/auth/session")"
  SESSION_PAYLOAD="$session_payload" python3 - <<'PY'
import json
import os

payload = json.loads(os.environ["SESSION_PAYLOAD"])
if payload.get("authenticated") not in (False, None):
    raise SystemExit("Expected unauthenticated session response in OIDC mode smoke check.")
PY
else
  guest_payload="$(curl -fsS -X POST "$API_URL/v1/auth/guest" -H "Content-Type: application/json" -d '{"name":"Integration Smoke","email":"smoke@example.com","affiliation":"bisque-ultra"}')"
  GUEST_PAYLOAD="$guest_payload" python3 - <<'PY'
import json
import os

payload = json.loads(os.environ["GUEST_PAYLOAD"])
if payload.get("authenticated") is not True:
    raise SystemExit("Guest auth smoke check did not authenticate.")
if str(payload.get("mode") or "").strip().lower() != "guest":
    raise SystemExit(f"Expected guest mode, got {payload.get('mode')!r}")
PY
fi

echo "Integration smoke check passed for $API_URL -> $BISQUE_ROOT_URL"
