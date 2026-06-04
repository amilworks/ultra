#!/usr/bin/env bash
set -euo pipefail

RELEASE_SHA="${1:-}"
if [ -z "$RELEASE_SHA" ]; then
  echo "Usage: $0 <git-sha>" >&2
  exit 1
fi

ULTRA_RELEASE_ROOT="${ULTRA_RELEASE_ROOT:-/srv/ultra}"
RELEASE_DIR="$ULTRA_RELEASE_ROOT/releases/$RELEASE_SHA"
CURRENT_LINK="$ULTRA_RELEASE_ROOT/current"
ULTRA_BACKEND_ENV_FILE="${ULTRA_BACKEND_ENV_FILE:-/etc/ultra/ultra-backend.env}"
ULTRA_PYTHON_ROOT="${ULTRA_PYTHON_ROOT:-$ULTRA_RELEASE_ROOT/python}"
ULTRA_DEEPAGENTS_VENV_ROOT="${ULTRA_DEEPAGENTS_VENV_ROOT:-$ULTRA_RELEASE_ROOT/deepagents-venvs}"
UV_PYTHON_VERSION="${UV_PYTHON_VERSION:-3.11}"
SYSTEMD_UNIT_DIR="${SYSTEMD_UNIT_DIR:-/etc/systemd/system}"

CONTROL_DIR="$RELEASE_DIR/backend/controlplane"
DEEPAGENTS_DIR="$RELEASE_DIR/backend/deepagents_runtime"
BIN_DIR="$RELEASE_DIR/bin"
DEEPAGENTS_VENV_DIR="$ULTRA_DEEPAGENTS_VENV_ROOT/$RELEASE_SHA"
CONTROL_HEALTH_URL="${ULTRA_CONTROL_HEALTH_URL:-http://127.0.0.1:8000/v1/health}"
CONTROL_ADMIN_URL="${ULTRA_CONTROL_ADMIN_URL:-http://127.0.0.1:8000/v2/admin/overview}"

resolve_uv_bin() {
  local candidate

  if [ -n "${UV_BIN:-}" ] && [ -x "${UV_BIN:-}" ]; then
    printf '%s\n' "$UV_BIN"
    return 0
  fi

  if command -v uv >/dev/null 2>&1; then
    command -v uv
    return 0
  fi

  for candidate in \
    "/home/${SUDO_USER:-}/.local/bin/uv" \
    "/root/.local/bin/uv" \
    "/usr/local/bin/uv" \
    "/usr/bin/uv"
  do
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  echo "Unable to locate uv; set UV_BIN to the absolute path of the uv executable." >&2
  exit 1
}

load_backend_env() {
  if [ ! -f "$ULTRA_BACKEND_ENV_FILE" ]; then
    echo "Backend env file not found: $ULTRA_BACKEND_ENV_FILE" >&2
    exit 1
  fi

  set -a
  # shellcheck disable=SC1090
  . "$ULTRA_BACKEND_ENV_FILE"
  set +a
}

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "Missing required environment value in $ULTRA_BACKEND_ENV_FILE: $name" >&2
    exit 1
  fi
}

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

check_systemctl() {
  if ! command -v systemctl >/dev/null 2>&1; then
    echo "systemctl is required for production control-stack deploys." >&2
    exit 1
  fi
}

install_systemd_units() {
  local unit
  for unit in \
    ultra-control.service \
    ultra-deepagents-worker.service \
    ultra-rarespot-worker.service \
    ultra-control-stack.target
  do
    install -m 0644 "$RELEASE_DIR/deploy/systemd/$unit" "$SYSTEMD_UNIT_DIR/$unit"
  done
}

if [ ! -d "$RELEASE_DIR" ]; then
  echo "Release root not found: $RELEASE_DIR" >&2
  echo "The Go control stack expects a full repo release at /srv/ultra/releases/<git-sha>." >&2
  exit 1
fi

if [ ! -d "$CONTROL_DIR" ]; then
  echo "Go control-plane directory not found: $CONTROL_DIR" >&2
  exit 1
fi

if [ ! -d "$DEEPAGENTS_DIR" ]; then
  echo "Deep Agents runtime directory not found: $DEEPAGENTS_DIR" >&2
  exit 1
fi

UV_BIN="$(resolve_uv_bin)"
load_backend_env
require_env ULTRA_CONTROL_DATABASE_URL
require_env ULTRA_CONTROL_NATS_URL
require_env ULTRA_CONTROL_ARTIFACT_ROOT

if [ "${ULTRA_CONTROL_AUTH_PROVIDER:-}" = "workos" ]; then
  require_env ULTRA_CONTROL_WORKOS_CLIENT_ID
  require_env ULTRA_CONTROL_WORKOS_API_KEY
  require_env ULTRA_CONTROL_WORKOS_REDIRECT_URI
  require_env ULTRA_CONTROL_WORKOS_COOKIE_PASSWORD
fi

echo "Preparing Go control-stack release: $RELEASE_DIR"
mkdir -p "$BIN_DIR" "$ULTRA_PYTHON_ROOT" "$ULTRA_DEEPAGENTS_VENV_ROOT"
mkdir -p "$ULTRA_CONTROL_ARTIFACT_ROOT"
if [ -n "${ULTRA_CONTROL_UPLOAD_ROOT:-}" ]; then
  mkdir -p "$ULTRA_CONTROL_UPLOAD_ROOT"
fi

echo "Building ultra-control"
if [ "${ULTRA_CONTROL_SKIP_BUILD:-0}" = "1" ]; then
  if [ ! -x "$BIN_DIR/ultra-control" ]; then
    echo "ULTRA_CONTROL_SKIP_BUILD=1 but $BIN_DIR/ultra-control is missing or not executable." >&2
    exit 1
  fi
  echo "Using prebuilt ultra-control binary at $BIN_DIR/ultra-control"
else
  (
    cd "$CONTROL_DIR"
    go build -trimpath -o "$BIN_DIR/ultra-control" ./cmd/ultra-control
  )
fi

echo "Preparing Deep Agents worker environment"
rm -rf "$DEEPAGENTS_VENV_DIR" "$DEEPAGENTS_DIR/.venv"
env UV_PYTHON_INSTALL_DIR="$ULTRA_PYTHON_ROOT" \
  "$UV_BIN" python install "$UV_PYTHON_VERSION"
env UV_PYTHON="$UV_PYTHON_VERSION" \
  UV_PYTHON_INSTALL_DIR="$ULTRA_PYTHON_ROOT" \
  UV_PROJECT_ENVIRONMENT="$DEEPAGENTS_VENV_DIR" \
  UV_LINK_MODE=copy \
  "$UV_BIN" sync --frozen --extra rarespot --project "$DEEPAGENTS_DIR"
ln -sfn "$DEEPAGENTS_VENV_DIR" "$DEEPAGENTS_DIR/.venv"

echo "Applying Go control-plane migrations with ultra-control migrate"
(
  cd "$CONTROL_DIR"
  "$BIN_DIR/ultra-control" migrate
)

ln -sfn "$RELEASE_DIR" "$CURRENT_LINK"
check_systemctl
install_systemd_units
systemctl daemon-reload

systemctl restart ultra-control
wait_for_health "$CONTROL_HEALTH_URL" "ultra-control"

systemctl restart ultra-deepagents-worker
systemctl restart ultra-rarespot-worker
sleep 2

systemctl is-active --quiet ultra-deepagents-worker
echo "ultra-deepagents-worker active"
systemctl is-active --quiet ultra-rarespot-worker
echo "ultra-rarespot-worker active"

if curl -fsS "$CONTROL_ADMIN_URL" >/dev/null 2>&1; then
  echo "Control admin overview reachable: $CONTROL_ADMIN_URL"
else
  echo "WARNING: Control admin overview was not reachable without an admin session: $CONTROL_ADMIN_URL" >&2
fi

echo "Go control-stack deploy complete for $RELEASE_SHA"
