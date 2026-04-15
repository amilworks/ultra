#!/usr/bin/env bash
set -euo pipefail

ULTRA_ENV="${ULTRA_ENV:-/etc/ultra/ultra-backend.env}"
PLATFORM_ENV="${PLATFORM_ENV:-/etc/ultra/platform.env}"
ULTRA_RELEASE_ROOT="${ULTRA_RELEASE_ROOT:-/srv/ultra}"

ensure_system_packages() {
  if ! command -v apt-get >/dev/null 2>&1 || ! command -v dpkg >/dev/null 2>&1; then
    return 0
  fi

  local missing=()
  if ! dpkg -s libgl1 >/dev/null 2>&1; then
    missing+=(libgl1)
  fi
  if ! dpkg -s libglib2.0-0 >/dev/null 2>&1; then
    missing+=(libglib2.0-0)
  fi

  if [ "${#missing[@]}" -eq 0 ]; then
    return 0
  fi

  if [ "$(id -u)" -ne 0 ]; then
    echo "Warning: missing runtime packages for OpenCV/Ultralytics: ${missing[*]}" >&2
    echo "Re-run this script with sudo or install them manually before enabling YOLO/SAM tools." >&2
    return 0
  fi

  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y "${missing[@]}"
}

load_env_file() {
  local env_path="$1"
  local raw_line line key value

  [ -f "$env_path" ] || return 0

  while IFS= read -r raw_line || [ -n "$raw_line" ]; do
    line="${raw_line#"${raw_line%%[![:space:]]*}"}"
    if [ -z "$line" ] || [ "${line#\#}" != "$line" ] || [[ "$line" != *=* ]]; then
      continue
    fi
    key="${line%%=*}"
    value="${line#*=}"
    key="${key%"${key##*[![:space:]]}"}"
    value="${value#"${value%%[![:space:]]*}"}"
    export "$key=$value"
  done < "$env_path"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --ultra-env)
      ULTRA_ENV="$2"
      shift 2
      ;;
    --platform-env)
      PLATFORM_ENV="$2"
      shift 2
      ;;
    --root)
      ULTRA_RELEASE_ROOT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

ensure_system_packages

mkdir -p \
  "$ULTRA_RELEASE_ROOT/releases" \
  "$ULTRA_RELEASE_ROOT/platform-releases" \
  "$ULTRA_RELEASE_ROOT/ops" \
  "$ULTRA_RELEASE_ROOT/models/yolo" \
  "$ULTRA_RELEASE_ROOT/models/medsam2/checkpoints" \
  "$ULTRA_RELEASE_ROOT/models/sam3" \
  "$ULTRA_RELEASE_ROOT/runtime" \
  "$ULTRA_RELEASE_ROOT/shared/artifacts" \
  "$ULTRA_RELEASE_ROOT/shared/uploads" \
  "$ULTRA_RELEASE_ROOT/shared/sessions" \
  "$ULTRA_RELEASE_ROOT/shared/science" \
  /var/log/ultra \
  "$(dirname "$ULTRA_ENV")" \
  "$(dirname "$PLATFORM_ENV")"

load_env_file "$PLATFORM_ENV"

PLATFORM_DATA_ROOT="${PLATFORM_DATA_ROOT:-}"
if [ -n "$PLATFORM_DATA_ROOT" ]; then
  mkdir -p \
    "$PLATFORM_DATA_ROOT/bisque-config" \
    "$PLATFORM_DATA_ROOT/bisque-data" \
    "$PLATFORM_DATA_ROOT/bisque-public" \
    "$PLATFORM_DATA_ROOT/bisque-reports" \
    "$PLATFORM_DATA_ROOT/bisque-staging"
fi

echo "Created production layout under $ULTRA_RELEASE_ROOT"
echo "Place backend env at: $ULTRA_ENV"
echo "Place platform env at: $PLATFORM_ENV"
echo "Next steps:"
echo "  1. Fill both env files from deploy/env/*.example"
echo "  2. Render nginx configs with scripts/render_production_templates.py"
echo "  3. Install deploy/systemd/*.service and deploy/systemd/*.target"
