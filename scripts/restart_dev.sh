#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STATE_DIR="$ROOT/.tmp/dev"

API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-localhost}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

BACKEND_PID_FILE="$STATE_DIR/backend.pid"
FRONTEND_PID_FILE="$STATE_DIR/frontend.pid"

mkdir -p "$STATE_DIR"

api_pid=""
frontend_pid=""

log() {
  printf '[restart-dev] %s\n' "$1"
}

read_pid_file() {
  local pid_file="$1"
  if [ -f "$pid_file" ]; then
    tr -d '[:space:]' <"$pid_file"
  fi
}

remove_pid_file() {
  local pid_file="$1"
  rm -f "$pid_file"
}

kill_pid_file() {
  local pid_file="$1"
  local name="$2"
  local pid
  pid="$(read_pid_file "$pid_file")"
  if [ -n "${pid:-}" ] && kill -0 "$pid" >/dev/null 2>&1; then
    log "Stopping $name pid $pid"
    kill "$pid" >/dev/null 2>&1 || true
    sleep 1
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
  fi
  remove_pid_file "$pid_file"
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

wait_for_http() {
  local url="$1"
  local label="$2"
  local attempts="${3:-60}"
  local delay_seconds="${4:-1}"

  local attempt
  for attempt in $(seq 1 "$attempts"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$delay_seconds"
  done

  log "$label did not become ready. Inspect logs:"
  return 1
}

cleanup_started_services() {
  local exit_code="$?"
  if [ -n "$frontend_pid" ] && kill -0 "$frontend_pid" >/dev/null 2>&1; then
    kill "$frontend_pid" >/dev/null 2>&1 || true
  fi
  if [ -n "$api_pid" ] && kill -0 "$api_pid" >/dev/null 2>&1; then
    kill "$api_pid" >/dev/null 2>&1 || true
  fi
  wait "$frontend_pid" >/dev/null 2>&1 || true
  wait "$api_pid" >/dev/null 2>&1 || true
  remove_pid_file "$BACKEND_PID_FILE"
  remove_pid_file "$FRONTEND_PID_FILE"
  exit "$exit_code"
}

start_backend() {
  log "Starting backend with reload on http://$API_HOST:$API_PORT"
  (
    cd "$ROOT"
    local reload_args=(
      --reload
      --reload-dir src
    )
    if [ -d "$ROOT/tests" ]; then
      reload_args+=(--reload-dir tests)
    fi
    env PYTHONUNBUFFERED=1 uv run uvicorn src.api.main:app \
      "${reload_args[@]}" \
      --host "$API_HOST" \
      --port "$API_PORT"
  ) &
  api_pid="$!"
  echo "$api_pid" >"$BACKEND_PID_FILE"
}

start_frontend() {
  log "Starting frontend on http://$FRONTEND_HOST:$FRONTEND_PORT"
  (
    cd "$ROOT"
    env FORCE_COLOR=1 pnpm --dir frontend dev \
      --host "$FRONTEND_HOST" \
      --port "$FRONTEND_PORT"
  ) &
  frontend_pid="$!"
  echo "$frontend_pid" >"$FRONTEND_PID_FILE"
}

stop_services() {
  kill_pid_file "$BACKEND_PID_FILE" "backend"
  kill_pid_file "$FRONTEND_PID_FILE" "frontend"
  kill_port "$API_PORT" "backend"
  kill_port "$FRONTEND_PORT" "frontend"
}

start_services() {
  start_backend
  start_frontend
  trap cleanup_started_services EXIT INT TERM
  wait_for_http "http://$API_HOST:$API_PORT/v1/health" "Backend" 90 1
  wait_for_http "http://$FRONTEND_HOST:$FRONTEND_PORT" "Frontend" 90 1
  log "Backend ready: http://$API_HOST:$API_PORT"
  log "Frontend ready: http://$FRONTEND_HOST:$FRONTEND_PORT"
  log "Press Ctrl+C in this terminal to stop both services."
  while kill -0 "$api_pid" >/dev/null 2>&1 && kill -0 "$frontend_pid" >/dev/null 2>&1; do
    sleep 1
  done
  wait "$api_pid"
  wait "$frontend_pid"
}

show_status() {
  if curl -fsS "http://$API_HOST:$API_PORT/v1/health" >/dev/null 2>&1; then
    log "Backend responding on http://$API_HOST:$API_PORT"
  else
    log "Backend not responding on http://$API_HOST:$API_PORT"
  fi

  if curl -fsS "http://$FRONTEND_HOST:$FRONTEND_PORT" >/dev/null 2>&1; then
    log "Frontend responding on http://$FRONTEND_HOST:$FRONTEND_PORT"
  else
    log "Frontend not responding on http://$FRONTEND_HOST:$FRONTEND_PORT"
  fi
}

command="${1:-restart}"

case "$command" in
  restart)
    stop_services
    start_services
    ;;
  start)
    start_services
    ;;
  stop)
    stop_services
    ;;
  status)
    show_status
    ;;
  *)
    echo "Usage: $0 [restart|start|stop|status]" >&2
    exit 1
    ;;
esac
