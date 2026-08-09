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
ULTRA_SANDBOX_IDENTITY_ENV_FILE="${ULTRA_SANDBOX_IDENTITY_ENV_FILE:-/etc/ultra/ultra-sandbox-image.env}"
ULTRA_PYTHON_ROOT="${ULTRA_PYTHON_ROOT:-$ULTRA_RELEASE_ROOT/python}"
ULTRA_DEEPAGENTS_VENV_ROOT="${ULTRA_DEEPAGENTS_VENV_ROOT:-$ULTRA_RELEASE_ROOT/deepagents-venvs}"
UV_PYTHON_VERSION="${UV_PYTHON_VERSION:-3.11}"
SYSTEMD_UNIT_DIR="${SYSTEMD_UNIT_DIR:-/etc/systemd/system}"

# Node role selects which units + build steps run. Default "all" preserves the
# legacy single-node behavior (control + workers together). "edge" = control +
# Postgres + NATS; "compute" = image-service + convert + agent workers
# (no control, no DB migration, no Go build).
DEPLOY_ROLE="${DEPLOY_ROLE:-all}"
case "$DEPLOY_ROLE" in
  all)     ROLE_CONTROL=1; ROLE_WORKERS=1 ;;
  edge)    ROLE_CONTROL=1; ROLE_WORKERS=0 ;;
  compute) ROLE_CONTROL=0; ROLE_WORKERS=1 ;;
  *) echo "Unknown DEPLOY_ROLE=$DEPLOY_ROLE (expected all|edge|compute)" >&2; exit 1 ;;
esac

CONTROL_DIR="$RELEASE_DIR/backend/controlplane"
DEEPAGENTS_DIR="$RELEASE_DIR/backend/deepagents_runtime"
DEEPAGENTS_WORKER_LOCK="$DEEPAGENTS_DIR/requirements.worker.lock"
SANDBOX_DOCKERFILE="$RELEASE_DIR/deploy/docker/deepagents-sandbox.Dockerfile"
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

verify_sandbox_runtime_image_binding() {
  local actual image

  image="${ULTRA_DEEPAGENTS_SANDBOX_IMAGE:-bisque-ultra-codeexec:py311}"
  if ! actual="$(docker image inspect --format '{{.Id}}' "$image" 2>/dev/null)"; then
    actual=""
  fi
  if [[ ! "$actual" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    echo "Sandbox image $image does not resolve to an immutable sha256:<64hex> image ID (got ${actual:-missing})." >&2
    exit 1
  fi
  RESOLVED_SANDBOX_IMAGE_ID="$actual"
  echo "Resolved sandbox runtime image: $image -> $actual"
}

install_sandbox_identity_env() {
  local identity_dir temporary

  if [[ ! "${RESOLVED_SANDBOX_IMAGE_ID:-}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    echo "Refusing to install a non-immutable sandbox identity override." >&2
    exit 1
  fi
  identity_dir="$(dirname "$ULTRA_SANDBOX_IDENTITY_ENV_FILE")"
  temporary="$ULTRA_SANDBOX_IDENTITY_ENV_FILE.tmp.$$"
  mkdir -p "$identity_dir"
  (
    umask 077
    printf 'ULTRA_DEEPAGENTS_SANDBOX_IMAGE=%s\n' "$RESOLVED_SANDBOX_IMAGE_ID" >"$temporary"
  )
  chmod 0600 "$temporary"
  mv -f "$temporary" "$ULTRA_SANDBOX_IDENTITY_ENV_FILE"
  echo "Installed immutable worker sandbox identity: $ULTRA_SANDBOX_IDENTITY_ENV_FILE"
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

build_sandbox_image() {
  local current_revision image

  if [ "${ULTRA_DEEPAGENTS_SKIP_SANDBOX_IMAGE_BUILD:-0}" = "1" ]; then
    echo "Skipping Deep Agents sandbox image build by request."
    return 0
  fi

  if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required to build the Deep Agents sandbox image." >&2
    exit 1
  fi

  if [ ! -f "$SANDBOX_DOCKERFILE" ]; then
    echo "Deep Agents sandbox Dockerfile not found: $SANDBOX_DOCKERFILE" >&2
    exit 1
  fi

  image="${ULTRA_DEEPAGENTS_SANDBOX_IMAGE:-bisque-ultra-codeexec:py311}"
  if [ "${ULTRA_DEEPAGENTS_FORCE_SANDBOX_IMAGE_BUILD:-0}" != "1" ] \
    && docker image inspect "$image" >/dev/null 2>&1; then
    current_revision="$(docker image inspect --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' "$image" 2>/dev/null || true)"
    if [ "$current_revision" = "$RELEASE_SHA" ]; then
      echo "Deep Agents sandbox image already matches release $RELEASE_SHA: $image"
      return 0
    fi
    echo "Deep Agents sandbox image revision ${current_revision:-unlabeled} does not match $RELEASE_SHA; rebuilding."
  fi

  echo "Building Deep Agents sandbox image: $image"
  docker build --build-arg "VCS_REF=$RELEASE_SHA" -f "$SANDBOX_DOCKERFILE" -t "$image" "$RELEASE_DIR"
}

install_systemd_units() {
  local unit units
  case "$DEPLOY_ROLE" in
    all)
      units="ultra-control.service ultra-deepagents-worker.service ultra-control-stack.target"
      ;;
    edge)
      units="ultra-control.service ultra-postgres.service ultra-nats.service"
      ;;
    compute)
      units="ultra-imgsvc.service ultra-image-convert-worker.service ultra-deepagents-worker-node.service ultra-analysis-worker.service"
      ;;
  esac
  for unit in $units; do
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

if [ "$ROLE_WORKERS" = 1 ]; then
  UV_BIN="$(resolve_uv_bin)"
fi
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

if [ "$ROLE_CONTROL" = 1 ]; then
if [ -f "$RELEASE_DIR/release-manifest.json" ]; then
  if [ ! -x "$BIN_DIR/ultra-control" ]; then
    echo "Immutable release manifest exists but its prebuilt control binary is unavailable." >&2
    exit 1
  fi
  echo "Using manifest-bound immutable control binary at $BIN_DIR/ultra-control"
elif [ "${ULTRA_CONTROL_SKIP_BUILD:-0}" = "1" ]; then
  if [ ! -x "$BIN_DIR/ultra-control" ]; then
    echo "ULTRA_CONTROL_SKIP_BUILD=1 but $BIN_DIR/ultra-control is missing or not executable." >&2
    exit 1
  fi
  echo "Using prebuilt legacy control binary at $BIN_DIR/ultra-control"
else
  echo "Building ultra-control for a legacy source-only release"
  (
    cd "$CONTROL_DIR"
    go build -trimpath -o "$BIN_DIR/ultra-control" ./cmd/ultra-control
  )
fi
fi

if [ "$ROLE_WORKERS" = 1 ]; then
echo "Preparing Deep Agents worker environment"
if [ ! -f "$DEEPAGENTS_WORKER_LOCK" ]; then
  echo "Deep Agents worker lock not found: $DEEPAGENTS_WORKER_LOCK" >&2
  exit 1
fi
rm -rf "$DEEPAGENTS_VENV_DIR" "$DEEPAGENTS_DIR/.venv"
env UV_PYTHON_INSTALL_DIR="$ULTRA_PYTHON_ROOT" \
  "$UV_BIN" python install "$UV_PYTHON_VERSION"
env UV_PYTHON_INSTALL_DIR="$ULTRA_PYTHON_ROOT" \
  "$UV_BIN" venv --python "$UV_PYTHON_VERSION" "$DEEPAGENTS_VENV_DIR"
"$UV_BIN" pip sync \
  --python "$DEEPAGENTS_VENV_DIR/bin/python" \
  --require-hashes \
  --only-binary=:all: \
  "$DEEPAGENTS_WORKER_LOCK"
"$UV_BIN" pip install \
  --python "$DEEPAGENTS_VENV_DIR/bin/python" \
  --no-build-isolation \
  --no-deps \
  "$DEEPAGENTS_DIR"
"$UV_BIN" pip check --python "$DEEPAGENTS_VENV_DIR/bin/python"
"$DEEPAGENTS_VENV_DIR/bin/python" -c \
  "import numpy; assert numpy.__version__ == '1.26.4'; import ultra_deepagents.agent; import ultra_deepagents.nats_worker"
ln -sfn "$DEEPAGENTS_VENV_DIR" "$DEEPAGENTS_DIR/.venv"

build_sandbox_image
verify_sandbox_runtime_image_binding
install_sandbox_identity_env
fi

if [ "$ROLE_CONTROL" = 1 ]; then
  echo "Applying Go control-plane migrations with ultra-control migrate"
  (
    cd "$CONTROL_DIR"
    "$BIN_DIR/ultra-control" migrate
  )
fi

ln -sfn "$RELEASE_DIR" "$CURRENT_LINK"
check_systemctl
install_systemd_units
systemctl daemon-reload

if [ "$DEPLOY_ROLE" = "edge" ]; then
  systemctl enable ultra-postgres ultra-nats >/dev/null 2>&1 || true
fi

if [ "$ROLE_CONTROL" = 1 ]; then
  systemctl enable ultra-control >/dev/null 2>&1 || true
  systemctl restart ultra-control
  wait_for_health "$CONTROL_HEALTH_URL" "ultra-control"
fi

if [ "$ROLE_WORKERS" = 1 ]; then
  if [ "$DEPLOY_ROLE" = "compute" ]; then
    for unit in ultra-imgsvc ultra-image-convert-worker ultra-deepagents-worker-node ultra-analysis-worker; do
      systemctl enable "$unit" >/dev/null 2>&1 || true
      systemctl restart "$unit"
    done
    wait_for_health "${ULTRA_IMGSVC_HEALTH_URL:-http://127.0.0.1:8099/healthz}" "ultra-imgsvc"
    sleep 2
    for unit in ultra-image-convert-worker ultra-deepagents-worker-node ultra-analysis-worker; do
      systemctl is-active --quiet "$unit" && echo "$unit active" || echo "WARNING: $unit not active" >&2
    done
  else
    systemctl enable ultra-deepagents-worker >/dev/null 2>&1 || true
    systemctl restart ultra-deepagents-worker
    sleep 2
    systemctl is-active --quiet ultra-deepagents-worker && echo "ultra-deepagents-worker active"
  fi
fi

if [ "$ROLE_CONTROL" = 1 ]; then
  if curl -fsS "$CONTROL_ADMIN_URL" >/dev/null 2>&1; then
    echo "Control admin overview reachable: $CONTROL_ADMIN_URL"
  else
    echo "WARNING: Control admin overview was not reachable without an admin session: $CONTROL_ADMIN_URL" >&2
  fi
fi

echo "Go control-stack deploy complete for $RELEASE_SHA"
