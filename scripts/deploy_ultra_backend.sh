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
ULTRA_VENV_ROOT="${ULTRA_VENV_ROOT:-$ULTRA_RELEASE_ROOT/venvs}"
ULTRA_PYTHON_ROOT="${ULTRA_PYTHON_ROOT:-$ULTRA_RELEASE_ROOT/python}"
ULTRA_BACKEND_ENV_FILE="${ULTRA_BACKEND_ENV_FILE:-/etc/ultra/ultra-backend.env}"
UV_PYTHON_VERSION="${UV_PYTHON_VERSION:-3.10}"
VENV_DIR="$ULTRA_VENV_ROOT/$RELEASE_SHA"

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

UV_BIN="$(resolve_uv_bin)"

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

validate_codeexec_service_env() {
  if [ ! -f "$ULTRA_BACKEND_ENV_FILE" ]; then
    echo "Skipping code execution env validation because $ULTRA_BACKEND_ENV_FILE does not exist yet."
    return 0
  fi
  python3 - "$ULTRA_BACKEND_ENV_FILE" <<'PY'
from pathlib import Path
import sys

env_path = Path(sys.argv[1])
values = {}
for raw_line in env_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    values[key.strip()] = value.strip().strip('"').strip("'")

backend = values.get("CODE_EXECUTION_DEFAULT_BACKEND", "").strip().lower()
service_url = values.get("CODE_EXECUTION_SERVICE_URL", "").strip()
service_key = values.get("CODE_EXECUTION_SERVICE_API_KEY", "").strip()
enabled = values.get("CODE_EXECUTION_ENABLED", "true").strip().lower()

if backend == "service":
    missing = []
    if enabled == "false":
        missing.append("CODE_EXECUTION_ENABLED=true")
    if not service_url:
        missing.append("CODE_EXECUTION_SERVICE_URL")
    if not service_key:
        missing.append("CODE_EXECUTION_SERVICE_API_KEY")
    if missing:
        raise SystemExit(
            "Service-backed code execution is enabled, but the backend env is missing: "
            + ", ".join(missing)
        )
    print("Validated service-backed code execution env.")
PY
}

echo "Preparing backend release: $RELEASE_DIR"
cd "$RELEASE_DIR"
validate_codeexec_service_env

# Keep managed Python and virtualenvs on local disk so systemd can execute them
# reliably even when releases live on a shared mount.
mkdir -p "$ULTRA_VENV_ROOT" "$ULTRA_PYTHON_ROOT"
rm -rf "$VENV_DIR" "$RELEASE_DIR/.venv"

env UV_PYTHON_INSTALL_DIR="$ULTRA_PYTHON_ROOT" \
  "$UV_BIN" python install "$UV_PYTHON_VERSION"

env UV_PYTHON="$UV_PYTHON_VERSION" \
  UV_PYTHON_INSTALL_DIR="$ULTRA_PYTHON_ROOT" \
  UV_PROJECT_ENVIRONMENT="$VENV_DIR" \
  UV_LINK_MODE=copy \
  "$UV_BIN" sync --frozen

ln -sfn "$VENV_DIR" "$RELEASE_DIR/.venv"

ln -sfn "$RELEASE_DIR" "$CURRENT_LINK"
systemctl daemon-reload

systemctl restart ultra-backend@1
wait_for_health "http://127.0.0.1:8001/v1/health" "ultra-backend@1"

systemctl restart ultra-backend@2
wait_for_health "http://127.0.0.1:8002/v1/health" "ultra-backend@2"

echo "Backend deploy complete for $RELEASE_SHA"
