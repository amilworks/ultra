#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/monorepo_env.sh
source "$SCRIPT_DIR/monorepo_env.sh"

ROOT="$(monorepo_root)"
MONOREPO_ENV_FILE="$(resolve_monorepo_env_file)"
BISQUE_ROOT_URL="$(read_monorepo_env BISQUE_ROOT http://localhost:8080 "$MONOREPO_ENV_FILE")"
CONTAINER_NAME="$(read_monorepo_env CONTAINER_NAME bisque-server "$MONOREPO_ENV_FILE")"
EXPECTED_AUTH_MODE="$(platform_expected_auth_mode "$MONOREPO_ENV_FILE")"

echo "Using env file: $MONOREPO_ENV_FILE"
ensure_platform_running "$MONOREPO_ENV_FILE"
assert_platform_provenance "$CONTAINER_NAME"
wait_for_http_ok "$BISQUE_ROOT_URL/image_service/formats" "BisQue platform" 90 2
curl -fsS "$BISQUE_ROOT_URL/image_service/formats" >/dev/null

(
  cd "$ROOT/platform/bisque"
  BISQUE_URL="$BISQUE_ROOT_URL" ./scripts/dev/test_auth_baseline.sh "--expect-${EXPECTED_AUTH_MODE}"
)

echo "Platform smoke check passed for $BISQUE_ROOT_URL (mode=${EXPECTED_AUTH_MODE})"
