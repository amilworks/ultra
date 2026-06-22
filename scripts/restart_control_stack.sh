#!/usr/bin/env bash
set -euo pipefail
trap '' HUP

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STATE_DIR="$ROOT/.tmp/control-stack"
LOG_DIR="${ULTRA_STACK_LOG_DIR:-$STATE_DIR/logs}"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"

CONTROL_PID_FILE="$STATE_DIR/control.pid"
WORKER_PID_FILE="$STATE_DIR/deepagents-worker.pid"
FRONTEND_PID_FILE="$STATE_DIR/frontend.pid"

mkdir -p "$STATE_DIR" "$LOG_DIR"

log() {
  printf '[control-stack] %s\n' "$1"
}

load_env() {
  if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
  fi
}

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
    exit 1
  fi
}

load_env

API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-localhost}"
FRONTEND_PORT="${FRONTEND_PORT:-5174}"
ULTRA_STACK_USE_POSTGRES="${ULTRA_STACK_USE_POSTGRES:-1}"
ULTRA_STACK_START_NATS="${ULTRA_STACK_START_NATS:-1}"
ULTRA_STACK_START_WORKER="${ULTRA_STACK_START_WORKER:-1}"
ULTRA_STACK_START_FRONTEND="${ULTRA_STACK_START_FRONTEND:-1}"
ULTRA_STACK_STOP_SCREEN_SESSIONS="${ULTRA_STACK_STOP_SCREEN_SESSIONS:-1}"

ULTRA_CONTROL_DATABASE_URL="${ULTRA_CONTROL_DATABASE_URL:-postgresql://postgres:postgres@127.0.0.1:55432/bisque_ultra}"
ULTRA_CONTROL_NATS_URL="${ULTRA_CONTROL_NATS_URL:-nats://127.0.0.1:4222}"
ULTRA_CONTROL_HTTP_ADDR="${ULTRA_CONTROL_HTTP_ADDR:-$API_HOST:$API_PORT}"
ULTRA_CONTROL_ARTIFACT_ROOT="${ULTRA_CONTROL_ARTIFACT_ROOT:-$ROOT/data/artifacts}"
ULTRA_CONTROL_UPLOAD_ROOT="${ULTRA_CONTROL_UPLOAD_ROOT:-${ULTRA_RESOURCE_ROOT:-$ROOT/data/uploads}}"
ULTRA_CONTROL_IMAGE_SERVICE_URL="${ULTRA_CONTROL_IMAGE_SERVICE_URL:-}"
ULTRA_CONTROL_DEV_ADMIN_ENABLED="${ULTRA_CONTROL_DEV_ADMIN_ENABLED:-true}"
ULTRA_CONTROL_BISQUE_ROOT_URL="${ULTRA_CONTROL_BISQUE_ROOT_URL:-${BISQUE_ROOT:-${BISQUE_SERVER:-}}}"
ULTRA_CONTROL_BISQUE_USERNAME="${ULTRA_CONTROL_BISQUE_USERNAME:-${BISQUE_USERNAME:-${BISQUE_USER:-}}}"
ULTRA_CONTROL_BISQUE_PASSWORD="${ULTRA_CONTROL_BISQUE_PASSWORD:-${BISQUE_PASSWORD:-}}"
ULTRA_CONTROL_BISQUE_MAX_IMPORT_BYTES="${ULTRA_CONTROL_BISQUE_MAX_IMPORT_BYTES:-536870912}"
ULTRA_CONTROL_SECRET_ENCRYPTION_KEY="${ULTRA_CONTROL_SECRET_ENCRYPTION_KEY:-${ULTRA_SECRET_ENCRYPTION_KEY:-}}"
ULTRA_CONTROL_SECRET_ENCRYPTION_KEY_ID="${ULTRA_CONTROL_SECRET_ENCRYPTION_KEY_ID:-local-dev-key}"
ULTRA_CONTROL_AUTH_PROVIDER="${ULTRA_CONTROL_AUTH_PROVIDER:-dev}"
ULTRA_CONTROL_WORKOS_CLIENT_ID="${ULTRA_CONTROL_WORKOS_CLIENT_ID:-}"
ULTRA_CONTROL_WORKOS_API_KEY="${ULTRA_CONTROL_WORKOS_API_KEY:-}"
ULTRA_CONTROL_WORKOS_REDIRECT_URI="${ULTRA_CONTROL_WORKOS_REDIRECT_URI:-http://$FRONTEND_HOST:$FRONTEND_PORT/v2/auth/workos/callback}"
ULTRA_CONTROL_WORKOS_POST_LOGIN_REDIRECT_URI="${ULTRA_CONTROL_WORKOS_POST_LOGIN_REDIRECT_URI:-http://$FRONTEND_HOST:$FRONTEND_PORT/}"
ULTRA_CONTROL_WORKOS_LOGOUT_REDIRECT_URI="${ULTRA_CONTROL_WORKOS_LOGOUT_REDIRECT_URI:-http://$FRONTEND_HOST:$FRONTEND_PORT/}"
ULTRA_CONTROL_WORKOS_COOKIE_PASSWORD="${ULTRA_CONTROL_WORKOS_COOKIE_PASSWORD:-}"
ULTRA_CONTROL_WORKOS_BASE_URL="${ULTRA_CONTROL_WORKOS_BASE_URL:-}"
ULTRA_CONTROL_WORKOS_COOKIE_SECURE="${ULTRA_CONTROL_WORKOS_COOKIE_SECURE:-false}"

if [ -z "$ULTRA_CONTROL_SECRET_ENCRYPTION_KEY" ] && [ -n "$ULTRA_CONTROL_BISQUE_ROOT_URL" ]; then
  local_secret_key_file="$STATE_DIR/control-secret-encryption-key"
  if [ ! -s "$local_secret_key_file" ]; then
    python3 - <<'PY' >"$local_secret_key_file"
import base64
import secrets

print(base64.b64encode(secrets.token_bytes(32)).decode("ascii"))
PY
    chmod 600 "$local_secret_key_file"
  fi
  ULTRA_CONTROL_SECRET_ENCRYPTION_KEY="$(tr -d '[:space:]' <"$local_secret_key_file")"
  log "Using generated local control secret encryption key at $local_secret_key_file"
fi

# Workers authenticate run-status/lease/heartbeat calls with a shared token so
# durable run leases work even when the control plane is WorkOS-gated.
ULTRA_CONTROL_WORKER_TOKEN="${ULTRA_CONTROL_WORKER_TOKEN:-}"
if [ -z "$ULTRA_CONTROL_WORKER_TOKEN" ]; then
  local_worker_token_file="$STATE_DIR/control-worker-token"
  if [ ! -s "$local_worker_token_file" ]; then
    python3 - <<'PY' >"$local_worker_token_file"
import secrets

print(secrets.token_hex(32))
PY
    chmod 600 "$local_worker_token_file"
  fi
  ULTRA_CONTROL_WORKER_TOKEN="$(tr -d '[:space:]' <"$local_worker_token_file")"
  log "Using generated local control worker token at $local_worker_token_file"
fi

OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8001/v1}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-oss-120b}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# Qwen3.6-27B vision-reasoner (on-prem vLLM on tesla). Enabled for the local stack so
# the coordinator can delegate vision tasks to the vision-reasoner subagent. Key is
# read from the gitignored repo file by default.
QWEN_VLM_ENABLED="${QWEN_VLM_ENABLED:-true}"
QWEN_VLM_BASE_URL="${QWEN_VLM_BASE_URL:-http://tesla.ece.ucsb.edu:8000/v1}"
QWEN_VLM_MODEL="${QWEN_VLM_MODEL:-Qwen3.6-27B}"
QWEN_VLM_API_KEY_FILE="${QWEN_VLM_API_KEY_FILE:-$ROOT/qwen36-vllm.api-key}"

ULTRA_DEEPAGENTS_WORKSPACE_ROOT="${ULTRA_DEEPAGENTS_WORKSPACE_ROOT:-$ROOT/data/deepagents/workspaces}"
ULTRA_DEEPAGENTS_MEMORY_ROOT="${ULTRA_DEEPAGENTS_MEMORY_ROOT:-$ROOT/data/deepagents/memory}"
ULTRA_ARTIFACT_ROOT="${ULTRA_ARTIFACT_ROOT:-$ULTRA_CONTROL_ARTIFACT_ROOT}"
ULTRA_UPLOAD_ROOTS="${ULTRA_UPLOAD_ROOTS:-$ULTRA_CONTROL_UPLOAD_ROOT}"
ULTRA_RARESPOT_UPLOAD_ROOTS="${ULTRA_RARESPOT_UPLOAD_ROOTS:-$ULTRA_CONTROL_UPLOAD_ROOT}"
ULTRA_RARESPOT_ALLOWED_INPUT_ROOTS="${ULTRA_RARESPOT_ALLOWED_INPUT_ROOTS:-$ULTRA_CONTROL_UPLOAD_ROOT:$ULTRA_CONTROL_ARTIFACT_ROOT}"

if [ "$ULTRA_STACK_USE_POSTGRES" != "1" ]; then
  ULTRA_CONTROL_DATABASE_URL=""
fi

wait_for_http() {
  local url="$1"
  local label="$2"
  local attempts="${3:-90}"
  local delay_seconds="${4:-1}"
  local attempt
  for attempt in $(seq 1 "$attempts"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$delay_seconds"
  done
  log "$label did not become ready at $url"
  return 1
}

wait_for_tcp() {
  local host="$1"
  local port="$2"
  local label="$3"
  local attempts="${4:-40}"
  local delay_seconds="${5:-1}"
  local attempt
  for attempt in $(seq 1 "$attempts"); do
    if python3 - "$host" "$port" >/dev/null 2>&1 <<'PY'; then
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
with socket.create_connection((host, port), timeout=1):
    pass
PY
      return 0
    fi
    sleep "$delay_seconds"
  done
  log "$label did not become ready on $host:$port"
  return 1
}

check_model_endpoint() {
  local models_url
  models_url="${OPENAI_BASE_URL%/}/models"
  if curl -fsS "$models_url" >/dev/null 2>&1; then
    log "Model endpoint responding at $models_url"
  else
    log "WARNING: model endpoint not responding at $models_url"
  fi
}

read_pid_file() {
  local pid_file="$1"
  if [ -f "$pid_file" ]; then
    tr -d '[:space:]' <"$pid_file"
  fi
}

kill_pid_file() {
  local pid_file="$1"
  local label="$2"
  local pid
  pid="$(read_pid_file "$pid_file")"
  if [ -n "${pid:-}" ] && kill -0 "$pid" >/dev/null 2>&1; then
    log "Stopping $label pid $pid"
    kill "$pid" >/dev/null 2>&1 || true
    sleep 1
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
  fi
  rm -f "$pid_file"
}

kill_repo_python_module() {
  local module="$1"
  local label="$2"
  if ! command -v lsof >/dev/null 2>&1; then
    return 0
  fi
  local pids
  pids="$(ps -axo pid=,command= | awk -v module="$module" 'index($0, "-m " module) > 0 { print $1 }' || true)"
  if [ -z "$pids" ]; then
    return 0
  fi
  local pid cwd matched_pids
  matched_pids=""
  for pid in $pids; do
    cwd="$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -n 1)"
    case "$cwd" in
      "$ROOT"|"$ROOT"/*)
        matched_pids="${matched_pids:+$matched_pids }$pid"
        ;;
    esac
  done
  if [ -z "$matched_pids" ]; then
    return 0
  fi
  log "Stopping stale $label process(es): $matched_pids"
  kill $matched_pids >/dev/null 2>&1 || true
  sleep 1
  for pid in $matched_pids; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
  done
}

kill_port() {
  local port="$1"
  local label="$2"
  if ! command -v lsof >/dev/null 2>&1; then
    return 0
  fi
  local pids
  pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
  if [ -z "$pids" ]; then
    return 0
  fi
  log "Clearing $label port $port"
  kill $pids >/dev/null 2>&1 || true
  sleep 1
  pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    kill -9 $pids >/dev/null 2>&1 || true
  fi
}

stop_screen_sessions() {
  if [ "$ULTRA_STACK_STOP_SCREEN_SESSIONS" != "1" ] || ! command -v screen >/dev/null 2>&1; then
    return 0
  fi
  for session in ultra-main-control ultra-main-worker ultra-main-frontend; do
    while read -r screen_session; do
      case "$screen_session" in
        *."$session")
          log "Stopping legacy screen session $screen_session"
          screen -S "$screen_session" -X quit >/dev/null 2>&1 || true
          ;;
      esac
    done < <(screen -ls | awk '{print $1}') || true
  done
}

ensure_nats() {
  if [ "$ULTRA_STACK_START_NATS" != "1" ]; then
    return 0
  fi
  require_command docker
  local nats_port
  nats_port="${ULTRA_STACK_NATS_PORT:-4222}"
  if docker inspect ultra-nats-dev >/dev/null 2>&1; then
    if [ "$(docker inspect -f '{{.State.Running}}' ultra-nats-dev 2>/dev/null || echo false)" != "true" ]; then
      log "Starting existing NATS container ultra-nats-dev"
      docker start ultra-nats-dev >/dev/null
    fi
  else
    log "Creating NATS JetStream container ultra-nats-dev"
    docker run -d --name ultra-nats-dev -p "$nats_port:4222" nats:2-alpine -js >/dev/null
  fi
  wait_for_tcp 127.0.0.1 "$nats_port" "NATS"
}

ensure_postgres() {
  if [ "$ULTRA_STACK_USE_POSTGRES" != "1" ]; then
    log "Postgres disabled; control plane will use memory store."
    return 0
  fi
  require_command docker
  local postgres_port postgres_db postgres_test_db postgres_user postgres_password
  postgres_port="${POSTGRES_PORT:-55432}"
  postgres_db="${POSTGRES_DB:-bisque_ultra}"
  postgres_test_db="${ULTRA_CONTROL_TEST_DATABASE_NAME:-${postgres_db}_test}"
  postgres_user="${POSTGRES_USER:-postgres}"
  postgres_password="${POSTGRES_PASSWORD:-postgres}"
  if docker inspect bisque-ultra-postgres >/dev/null 2>&1; then
    if [ "$(docker inspect -f '{{.State.Running}}' bisque-ultra-postgres 2>/dev/null || echo false)" != "true" ]; then
      log "Starting existing Postgres container bisque-ultra-postgres"
      docker start bisque-ultra-postgres >/dev/null
    else
      log "Using existing Postgres container bisque-ultra-postgres"
    fi
    wait_for_tcp 127.0.0.1 "$postgres_port" "Postgres"
  else
    log "Creating local Postgres container"
    (cd "$ROOT" && make postgres-init)
  fi
  log "Ensuring local Postgres databases exist"
  docker exec \
    -e PGUSER="$postgres_user" \
    -e PGPASSWORD="$postgres_password" \
    -e DB="$postgres_db" \
    -e TEST_DB="$postgres_test_db" \
    bisque-ultra-postgres sh -lc '
      psql -U "$PGUSER" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='\''$DB'\''" | grep -q 1 ||
        psql -U "$PGUSER" -d postgres -c "CREATE DATABASE \"$DB\"";
      psql -U "$PGUSER" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='\''$TEST_DB'\''" | grep -q 1 ||
        psql -U "$PGUSER" -d postgres -c "CREATE DATABASE \"$TEST_DB\"";
    '
  log "Applying Go control-plane Postgres schema"
  (
    cd "$ROOT/backend/controlplane"
    ULTRA_CONTROL_DATABASE_URL="$ULTRA_CONTROL_DATABASE_URL" go run ./cmd/ultra-control migrate
  )
}

start_detached() {
  local pid_file="$1"
  local log_file="$2"
  local workdir="$3"
  local session_name="$4"
  shift 4
  local runner
  runner="$STATE_DIR/run-$session_name.sh"
  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd %q\n' "$workdir"
    printf 'echo "$$" > %q\n' "$pid_file"
    printf 'exec'
    local arg
    for arg in "$@"; do
      printf ' %q' "$arg"
    done
    printf ' > %q 2>&1 < /dev/null\n' "$log_file"
  } >"$runner"
  chmod +x "$runner"
  rm -f "$pid_file"
  if command -v screen >/dev/null 2>&1; then
    while read -r screen_session; do
      case "$screen_session" in
        *."$session_name")
          screen -S "$screen_session" -X quit >/dev/null 2>&1 || true
          ;;
      esac
    done < <(screen -ls | awk '{print $1}') || true
    screen -dmS "$session_name" bash "$runner"
  else
    (
      nohup bash "$runner" >/dev/null 2>&1 < /dev/null &
      echo "$!" >"$pid_file"
    )
  fi
}

python_entrypoint() {
  if [ -x "$ROOT/backend/deepagents_runtime/.venv/bin/python" ]; then
    printf '%s\n' "$ROOT/backend/deepagents_runtime/.venv/bin/python"
    return 0
  fi
  printf '%s\n' uv
}

start_control() {
  log "Starting Go control plane on http://$ULTRA_CONTROL_HTTP_ADDR"
  start_detached "$CONTROL_PID_FILE" "$LOG_DIR/control.log" "$ROOT/backend/controlplane" "ultra-main-control" \
    env \
    ULTRA_CONTROL_HTTP_ADDR="$ULTRA_CONTROL_HTTP_ADDR" \
    ULTRA_CONTROL_DATABASE_URL="$ULTRA_CONTROL_DATABASE_URL" \
    ULTRA_CONTROL_NATS_URL="$ULTRA_CONTROL_NATS_URL" \
    ULTRA_CONTROL_ARTIFACT_ROOT="$ULTRA_CONTROL_ARTIFACT_ROOT" \
    ULTRA_CONTROL_UPLOAD_ROOT="$ULTRA_CONTROL_UPLOAD_ROOT" \
    ULTRA_CONTROL_IMAGE_SERVICE_URL="$ULTRA_CONTROL_IMAGE_SERVICE_URL" \
    ULTRA_CONTROL_DEV_ADMIN_ENABLED="$ULTRA_CONTROL_DEV_ADMIN_ENABLED" \
    ULTRA_CONTROL_BISQUE_ROOT_URL="$ULTRA_CONTROL_BISQUE_ROOT_URL" \
    ULTRA_CONTROL_BISQUE_USERNAME="$ULTRA_CONTROL_BISQUE_USERNAME" \
    ULTRA_CONTROL_BISQUE_PASSWORD="$ULTRA_CONTROL_BISQUE_PASSWORD" \
    ULTRA_CONTROL_BISQUE_MAX_IMPORT_BYTES="$ULTRA_CONTROL_BISQUE_MAX_IMPORT_BYTES" \
    ULTRA_CONTROL_SECRET_ENCRYPTION_KEY="$ULTRA_CONTROL_SECRET_ENCRYPTION_KEY" \
    ULTRA_CONTROL_SECRET_ENCRYPTION_KEY_ID="$ULTRA_CONTROL_SECRET_ENCRYPTION_KEY_ID" \
    ULTRA_CONTROL_WORKER_TOKEN="$ULTRA_CONTROL_WORKER_TOKEN" \
    ULTRA_CONTROL_AUTH_PROVIDER="$ULTRA_CONTROL_AUTH_PROVIDER" \
    ULTRA_CONTROL_WORKOS_CLIENT_ID="$ULTRA_CONTROL_WORKOS_CLIENT_ID" \
    ULTRA_CONTROL_WORKOS_API_KEY="$ULTRA_CONTROL_WORKOS_API_KEY" \
    ULTRA_CONTROL_WORKOS_REDIRECT_URI="$ULTRA_CONTROL_WORKOS_REDIRECT_URI" \
    ULTRA_CONTROL_WORKOS_POST_LOGIN_REDIRECT_URI="$ULTRA_CONTROL_WORKOS_POST_LOGIN_REDIRECT_URI" \
    ULTRA_CONTROL_WORKOS_LOGOUT_REDIRECT_URI="$ULTRA_CONTROL_WORKOS_LOGOUT_REDIRECT_URI" \
    ULTRA_CONTROL_WORKOS_COOKIE_PASSWORD="$ULTRA_CONTROL_WORKOS_COOKIE_PASSWORD" \
    ULTRA_CONTROL_WORKOS_BASE_URL="$ULTRA_CONTROL_WORKOS_BASE_URL" \
    ULTRA_CONTROL_WORKOS_COOKIE_SECURE="$ULTRA_CONTROL_WORKOS_COOKIE_SECURE" \
    go run ./cmd/ultra-control
}

start_deepagents_worker() {
  if [ "$ULTRA_STACK_START_WORKER" != "1" ]; then
    return 0
  fi
  # Durable worker entrypoint: python -m ultra_deepagents.nats_worker
  log "Starting Deep Agents NATS worker"
  local python_bin
  python_bin="$(python_entrypoint)"
  if [ "$python_bin" = "uv" ]; then
    start_detached "$WORKER_PID_FILE" "$LOG_DIR/deepagents-worker.log" "$ROOT/backend/deepagents_runtime" "ultra-main-worker" \
      env \
      OPENAI_BASE_URL="$OPENAI_BASE_URL" \
      OPENAI_MODEL="$OPENAI_MODEL" \
      OPENAI_API_KEY="$OPENAI_API_KEY" \
      QWEN_VLM_ENABLED="$QWEN_VLM_ENABLED" \
      QWEN_VLM_BASE_URL="$QWEN_VLM_BASE_URL" \
      QWEN_VLM_MODEL="$QWEN_VLM_MODEL" \
      QWEN_VLM_API_KEY_FILE="$QWEN_VLM_API_KEY_FILE" \
      ULTRA_CONTROL_BASE_URL="http://$ULTRA_CONTROL_HTTP_ADDR" \
      ULTRA_CONTROL_WORKER_TOKEN="$ULTRA_CONTROL_WORKER_TOKEN" \
      ULTRA_CONTROL_NATS_URL="$ULTRA_CONTROL_NATS_URL" \
      ULTRA_CONTROL_DATABASE_URL="$ULTRA_CONTROL_DATABASE_URL" \
      ULTRA_NATS_URL="$ULTRA_CONTROL_NATS_URL" \
      ULTRA_CONTROL_ARTIFACT_ROOT="$ULTRA_CONTROL_ARTIFACT_ROOT" \
      ULTRA_ARTIFACT_ROOT="$ULTRA_ARTIFACT_ROOT" \
      ULTRA_UPLOAD_ROOTS="$ULTRA_UPLOAD_ROOTS" \
      ULTRA_RARESPOT_UPLOAD_ROOTS="$ULTRA_RARESPOT_UPLOAD_ROOTS" \
      ULTRA_DEEPAGENTS_WORKSPACE_ROOT="$ULTRA_DEEPAGENTS_WORKSPACE_ROOT" \
      ULTRA_DEEPAGENTS_MEMORY_ROOT="$ULTRA_DEEPAGENTS_MEMORY_ROOT" \
      ULTRA_DEEPAGENTS_TIMEOUT_SECONDS="${ULTRA_DEEPAGENTS_TIMEOUT_SECONDS:-0}" \
      ULTRA_DEEPAGENTS_MODEL_STREAM_IDLE_TIMEOUT_SECONDS="${ULTRA_DEEPAGENTS_MODEL_STREAM_IDLE_TIMEOUT_SECONDS:-0}" \
      ULTRA_DEEPAGENTS_MODEL_STREAM_IDLE_MAX_RECOVERIES="${ULTRA_DEEPAGENTS_MODEL_STREAM_IDLE_MAX_RECOVERIES:-2}" \
      ULTRA_DEEPAGENTS_RECURSION_LIMIT="${ULTRA_DEEPAGENTS_RECURSION_LIMIT:-1000}" \
      ULTRA_DEEPAGENTS_COMPLETION_MAX_CONTINUATIONS="${ULTRA_DEEPAGENTS_COMPLETION_MAX_CONTINUATIONS:-8}" \
      ULTRA_DEEPAGENTS_SANDBOX_CPUS="${ULTRA_DEEPAGENTS_SANDBOX_CPUS:-0}" \
      ULTRA_DEEPAGENTS_SANDBOX_MEMORY="${ULTRA_DEEPAGENTS_SANDBOX_MEMORY:-}" \
      ULTRA_DEEPAGENTS_SANDBOX_PIDS_LIMIT="${ULTRA_DEEPAGENTS_SANDBOX_PIDS_LIMIT:-0}" \
      ULTRA_DEEPAGENTS_SANDBOX_TIMEOUT_SECONDS="${ULTRA_DEEPAGENTS_SANDBOX_TIMEOUT_SECONDS:-21600}" \
      ULTRA_DEEPAGENTS_SANDBOX_OUTPUT_LIMIT_BYTES="${ULTRA_DEEPAGENTS_SANDBOX_OUTPUT_LIMIT_BYTES:-0}" \
      ULTRA_DEEPAGENTS_WORKER_MAX_CONCURRENCY="${ULTRA_DEEPAGENTS_WORKER_MAX_CONCURRENCY:-64}" \
      ULTRA_DEEPAGENTS_WORKER_ACK_WAIT_SECONDS="${ULTRA_DEEPAGENTS_WORKER_ACK_WAIT_SECONDS:-600}" \
      ULTRA_DEEPAGENTS_WORKER_ACK_PROGRESS_INTERVAL_SECONDS="${ULTRA_DEEPAGENTS_WORKER_ACK_PROGRESS_INTERVAL_SECONDS:-60}" \
      ULTRA_DEEPAGENTS_WORKER_MAX_DELIVER="${ULTRA_DEEPAGENTS_WORKER_MAX_DELIVER:-5}" \
      ULTRA_DEEPAGENTS_BUILDER_ENABLED="${ULTRA_DEEPAGENTS_BUILDER_ENABLED:-false}" \
      ULTRA_DEEPAGENTS_BUILDER_BASE_URL="${ULTRA_DEEPAGENTS_BUILDER_BASE_URL:-}" \
      ULTRA_DEEPAGENTS_BUILDER_MODEL="${ULTRA_DEEPAGENTS_BUILDER_MODEL:-}" \
      ULTRA_DEEPAGENTS_BUILDER_API_KEY_FILE="${ULTRA_DEEPAGENTS_BUILDER_API_KEY_FILE:-}" \
      ULTRA_DEEPAGENTS_BUILDER_MAX_INPUT_TOKENS="${ULTRA_DEEPAGENTS_BUILDER_MAX_INPUT_TOKENS:-0}" \
      ULTRA_DEEPAGENTS_BUILDER_MULTIMODAL="${ULTRA_DEEPAGENTS_BUILDER_MULTIMODAL:-false}" \
      ULTRA_DEEPAGENTS_BUILDER_GOAL_MAX_ITERATIONS="${ULTRA_DEEPAGENTS_BUILDER_GOAL_MAX_ITERATIONS:-6}" \
      ULTRA_DEEPAGENTS_BUILDER_RECURSION_LIMIT="${ULTRA_DEEPAGENTS_BUILDER_RECURSION_LIMIT:-400}" \
      uv run --python 3.11 python -m ultra_deepagents.nats_worker
  else
    start_detached "$WORKER_PID_FILE" "$LOG_DIR/deepagents-worker.log" "$ROOT/backend/deepagents_runtime" "ultra-main-worker" \
      env \
      OPENAI_BASE_URL="$OPENAI_BASE_URL" \
      OPENAI_MODEL="$OPENAI_MODEL" \
      OPENAI_API_KEY="$OPENAI_API_KEY" \
      QWEN_VLM_ENABLED="$QWEN_VLM_ENABLED" \
      QWEN_VLM_BASE_URL="$QWEN_VLM_BASE_URL" \
      QWEN_VLM_MODEL="$QWEN_VLM_MODEL" \
      QWEN_VLM_API_KEY_FILE="$QWEN_VLM_API_KEY_FILE" \
      ULTRA_CONTROL_BASE_URL="http://$ULTRA_CONTROL_HTTP_ADDR" \
      ULTRA_CONTROL_WORKER_TOKEN="$ULTRA_CONTROL_WORKER_TOKEN" \
      ULTRA_CONTROL_NATS_URL="$ULTRA_CONTROL_NATS_URL" \
      ULTRA_CONTROL_DATABASE_URL="$ULTRA_CONTROL_DATABASE_URL" \
      ULTRA_NATS_URL="$ULTRA_CONTROL_NATS_URL" \
      ULTRA_CONTROL_ARTIFACT_ROOT="$ULTRA_CONTROL_ARTIFACT_ROOT" \
      ULTRA_ARTIFACT_ROOT="$ULTRA_ARTIFACT_ROOT" \
      ULTRA_UPLOAD_ROOTS="$ULTRA_UPLOAD_ROOTS" \
      ULTRA_RARESPOT_UPLOAD_ROOTS="$ULTRA_RARESPOT_UPLOAD_ROOTS" \
      ULTRA_DEEPAGENTS_WORKSPACE_ROOT="$ULTRA_DEEPAGENTS_WORKSPACE_ROOT" \
      ULTRA_DEEPAGENTS_MEMORY_ROOT="$ULTRA_DEEPAGENTS_MEMORY_ROOT" \
      ULTRA_DEEPAGENTS_TIMEOUT_SECONDS="${ULTRA_DEEPAGENTS_TIMEOUT_SECONDS:-0}" \
      ULTRA_DEEPAGENTS_MODEL_STREAM_IDLE_TIMEOUT_SECONDS="${ULTRA_DEEPAGENTS_MODEL_STREAM_IDLE_TIMEOUT_SECONDS:-0}" \
      ULTRA_DEEPAGENTS_MODEL_STREAM_IDLE_MAX_RECOVERIES="${ULTRA_DEEPAGENTS_MODEL_STREAM_IDLE_MAX_RECOVERIES:-2}" \
      ULTRA_DEEPAGENTS_RECURSION_LIMIT="${ULTRA_DEEPAGENTS_RECURSION_LIMIT:-1000}" \
      ULTRA_DEEPAGENTS_COMPLETION_MAX_CONTINUATIONS="${ULTRA_DEEPAGENTS_COMPLETION_MAX_CONTINUATIONS:-8}" \
      ULTRA_DEEPAGENTS_SANDBOX_CPUS="${ULTRA_DEEPAGENTS_SANDBOX_CPUS:-0}" \
      ULTRA_DEEPAGENTS_SANDBOX_MEMORY="${ULTRA_DEEPAGENTS_SANDBOX_MEMORY:-}" \
      ULTRA_DEEPAGENTS_SANDBOX_PIDS_LIMIT="${ULTRA_DEEPAGENTS_SANDBOX_PIDS_LIMIT:-0}" \
      ULTRA_DEEPAGENTS_SANDBOX_TIMEOUT_SECONDS="${ULTRA_DEEPAGENTS_SANDBOX_TIMEOUT_SECONDS:-21600}" \
      ULTRA_DEEPAGENTS_SANDBOX_OUTPUT_LIMIT_BYTES="${ULTRA_DEEPAGENTS_SANDBOX_OUTPUT_LIMIT_BYTES:-0}" \
      ULTRA_DEEPAGENTS_WORKER_MAX_CONCURRENCY="${ULTRA_DEEPAGENTS_WORKER_MAX_CONCURRENCY:-64}" \
      ULTRA_DEEPAGENTS_WORKER_ACK_WAIT_SECONDS="${ULTRA_DEEPAGENTS_WORKER_ACK_WAIT_SECONDS:-600}" \
      ULTRA_DEEPAGENTS_WORKER_ACK_PROGRESS_INTERVAL_SECONDS="${ULTRA_DEEPAGENTS_WORKER_ACK_PROGRESS_INTERVAL_SECONDS:-60}" \
      ULTRA_DEEPAGENTS_WORKER_MAX_DELIVER="${ULTRA_DEEPAGENTS_WORKER_MAX_DELIVER:-5}" \
      ULTRA_DEEPAGENTS_BUILDER_ENABLED="${ULTRA_DEEPAGENTS_BUILDER_ENABLED:-false}" \
      ULTRA_DEEPAGENTS_BUILDER_BASE_URL="${ULTRA_DEEPAGENTS_BUILDER_BASE_URL:-}" \
      ULTRA_DEEPAGENTS_BUILDER_MODEL="${ULTRA_DEEPAGENTS_BUILDER_MODEL:-}" \
      ULTRA_DEEPAGENTS_BUILDER_API_KEY_FILE="${ULTRA_DEEPAGENTS_BUILDER_API_KEY_FILE:-}" \
      ULTRA_DEEPAGENTS_BUILDER_MAX_INPUT_TOKENS="${ULTRA_DEEPAGENTS_BUILDER_MAX_INPUT_TOKENS:-0}" \
      ULTRA_DEEPAGENTS_BUILDER_MULTIMODAL="${ULTRA_DEEPAGENTS_BUILDER_MULTIMODAL:-false}" \
      ULTRA_DEEPAGENTS_BUILDER_GOAL_MAX_ITERATIONS="${ULTRA_DEEPAGENTS_BUILDER_GOAL_MAX_ITERATIONS:-6}" \
      ULTRA_DEEPAGENTS_BUILDER_RECURSION_LIMIT="${ULTRA_DEEPAGENTS_BUILDER_RECURSION_LIMIT:-400}" \
      "$python_bin" -m ultra_deepagents.nats_worker
  fi
}

start_frontend() {
  if [ "$ULTRA_STACK_START_FRONTEND" != "1" ]; then
    return 0
  fi
  log "Starting frontend on http://$FRONTEND_HOST:$FRONTEND_PORT"
  start_detached "$FRONTEND_PID_FILE" "$LOG_DIR/frontend.log" "$ROOT/frontend" "ultra-main-frontend" \
    env FORCE_COLOR=1 pnpm exec vite --host "$FRONTEND_HOST" --port "$FRONTEND_PORT" --strictPort
}

stop_services() {
  kill_pid_file "$FRONTEND_PID_FILE" "frontend"
  kill_pid_file "$WORKER_PID_FILE" "Deep Agents worker"
  kill_pid_file "$CONTROL_PID_FILE" "Go control plane"
  stop_screen_sessions
  kill_repo_python_module "ultra_deepagents.nats_worker" "Deep Agents worker"
  kill_port "$API_PORT" "Go control plane"
  if [ "$ULTRA_STACK_START_FRONTEND" = "1" ]; then
    kill_port "$FRONTEND_PORT" "frontend"
  fi
}

start_services() {
  require_command curl
  require_command go
  require_command pnpm
  require_command python3
  ensure_nats
  ensure_postgres
  start_control
  wait_for_http "http://$API_HOST:$API_PORT/v1/health" "Go control plane"
  start_deepagents_worker
  start_frontend
  if [ "$ULTRA_STACK_START_FRONTEND" = "1" ]; then
    wait_for_http "http://$FRONTEND_HOST:$FRONTEND_PORT" "Frontend"
  fi
  status_services
}

status_services() {
  log "Status"
  check_model_endpoint
  if curl -fsS "http://$API_HOST:$API_PORT/v1/health" >/dev/null 2>&1; then
    log "Go control plane responding on http://$API_HOST:$API_PORT"
  else
    log "Go control plane not responding on http://$API_HOST:$API_PORT"
  fi
  if curl -fsS "http://$API_HOST:$API_PORT/v2/admin/overview" >/dev/null 2>&1; then
    curl -fsS "http://$API_HOST:$API_PORT/v2/admin/overview" | python3 -c '
import json
import sys

payload = json.load(sys.stdin)
runtime = payload.get("runtime", {})
queue = payload.get("queue", {})
workers = payload.get("workers", [])
print("[control-stack] store_backend=" + str(runtime.get("store_backend", "unknown")))
print("[control-stack] dispatch_mode=" + str(runtime.get("dispatch_mode", "unknown")))
print("[control-stack] nats_consumers=" + str(queue.get("consumer_count", 0)))
print("[control-stack] worker_heartbeats=" + str(len(workers)))
if runtime.get("store_backend") != "postgres":
    print("[control-stack] WARNING: control plane is not using Postgres durability")
'
  fi
  if [ "$ULTRA_STACK_START_FRONTEND" = "1" ]; then
    if curl -fsS "http://$FRONTEND_HOST:$FRONTEND_PORT" >/dev/null 2>&1; then
      log "Frontend responding on http://$FRONTEND_HOST:$FRONTEND_PORT"
    else
      log "Frontend not responding on http://$FRONTEND_HOST:$FRONTEND_PORT"
    fi
  fi
  log "Logs: $LOG_DIR"
}

command="${1:-restart}"
case "$command" in
  start)
    start_services
    ;;
  restart)
    stop_services
    start_services
    ;;
  stop)
    stop_services
    ;;
  status)
    status_services
    ;;
  migrate)
    ensure_postgres
    ;;
  *)
    echo "Usage: $0 [start|restart|stop|status|migrate]" >&2
    exit 1
    ;;
esac
