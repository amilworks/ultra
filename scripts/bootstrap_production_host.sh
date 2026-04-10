#!/usr/bin/env bash
set -euo pipefail

ULTRA_ENV="${ULTRA_ENV:-/etc/ultra/ultra-backend.env}"
PLATFORM_ENV="${PLATFORM_ENV:-/etc/ultra/platform.env}"
ULTRA_RELEASE_ROOT="${ULTRA_RELEASE_ROOT:-/srv/ultra}"

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

mkdir -p \
  "$ULTRA_RELEASE_ROOT/releases" \
  "$ULTRA_RELEASE_ROOT/platform-releases" \
  "$ULTRA_RELEASE_ROOT/ops" \
  "$ULTRA_RELEASE_ROOT/shared/artifacts" \
  "$ULTRA_RELEASE_ROOT/shared/uploads" \
  "$ULTRA_RELEASE_ROOT/shared/sessions" \
  "$ULTRA_RELEASE_ROOT/shared/science" \
  /var/log/ultra \
  "$(dirname "$ULTRA_ENV")" \
  "$(dirname "$PLATFORM_ENV")"

if [ -f "$PLATFORM_ENV" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$PLATFORM_ENV"
  set +a
fi

PLATFORM_DATA_ROOT="${PLATFORM_DATA_ROOT:-}"
if [ -n "$PLATFORM_DATA_ROOT" ]; then
  mkdir -p \
    "$PLATFORM_DATA_ROOT/postgres" \
    "$PLATFORM_DATA_ROOT/keycloak" \
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
