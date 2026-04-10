#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-up}"
ENV_FILE="${ENV_FILE:-/etc/ultra/platform.env}"
ULTRA_ENV="${ULTRA_ENV:-/etc/ultra/ultra-backend.env}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PLATFORM_DEPLOY_MODE="${PLATFORM_DEPLOY_MODE:-}"
BUILD_PLATFORM_IMAGES="${BUILD_PLATFORM_IMAGES:-0}"

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
    if [[ "$value" == \"*\" && "$value" == *\" ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
      value="${value:1:${#value}-2}"
    fi
    export "$key=$value"
  done < "$env_path"
}

load_env_file "$ENV_FILE"
[ -f "$ULTRA_ENV" ] && load_env_file "$ULTRA_ENV"

export BISQUE_AUTH_MODE="${BISQUE_AUTH_MODE_OVERRIDE:-oidc}"
export BISQUE_AUTH_LOCAL_TOKEN_ENABLED="${BISQUE_AUTH_LOCAL_TOKEN_ENABLED_OVERRIDE:-false}"
export BISQUE_AUTH_COOKIE_SECURE="${BISQUE_AUTH_COOKIE_SECURE:-true}"
export BISQUE_BEAKER_SESSION_SECURE="${BISQUE_BEAKER_SESSION_SECURE:-true}"
export BISQUE_BEAKER_SESSION_HTTPONLY="${BISQUE_BEAKER_SESSION_HTTPONLY:-true}"
export BISQUE_BEAKER_SESSION_SAMESITE="${BISQUE_BEAKER_SESSION_SAMESITE:-Lax}"
export KEYCLOAK_CLIENT_DIRECT_ACCESS_GRANTS="${KEYCLOAK_CLIENT_DIRECT_ACCESS_GRANTS:-false}"
export KEYCLOAK_CLIENT_STANDARD_FLOW="${KEYCLOAK_CLIENT_STANDARD_FLOW:-true}"
export KEYCLOAK_REALM_IMPORT_FILE="${KEYCLOAK_REALM_IMPORT_FILE:-/etc/ultra/keycloak-realm-bisque.json}"
export KEYCLOAK_REALM_NAME="${KEYCLOAK_REALM_NAME:-bisque}"

if [ -z "${KEYCLOAK_ADMIN_SERVER_URL:-}" ]; then
  KEYCLOAK_HTTP_RELATIVE_PATH="${KEYCLOAK_HTTP_RELATIVE_PATH:-/auth}"
  if [[ "$KEYCLOAK_HTTP_RELATIVE_PATH" != /* ]]; then
    KEYCLOAK_HTTP_RELATIVE_PATH="/$KEYCLOAK_HTTP_RELATIVE_PATH"
  fi
  KEYCLOAK_HTTP_RELATIVE_PATH="${KEYCLOAK_HTTP_RELATIVE_PATH%/}"
  [ -n "$KEYCLOAK_HTTP_RELATIVE_PATH" ] || KEYCLOAK_HTTP_RELATIVE_PATH="/auth"
  export KEYCLOAK_ADMIN_SERVER_URL="http://127.0.0.1:8080${KEYCLOAK_HTTP_RELATIVE_PATH}"
fi

if [ -z "$PLATFORM_DEPLOY_MODE" ]; then
  PLATFORM_DEPLOY_MODE="${PLATFORM_DEPLOY_MODE:-single-node}"
fi

COMPOSE_ARGS=(
  --env-file "$ENV_FILE"
  -f "$ROOT/platform/bisque/docker-compose.with-engine.yml"
  -f "$ROOT/platform/bisque/docker-compose.production.yml"
)
SERVICES=(bisque postgres keycloak)

if [ "$PLATFORM_DEPLOY_MODE" = "platform-node" ]; then
  COMPOSE_ARGS+=(-f "$ROOT/platform/bisque/docker-compose.platform-node.yml")
  SERVICES+=(platform-caddy)
fi

render_platform_proxy() {
  local render_dir

  [ "$PLATFORM_DEPLOY_MODE" = "platform-node" ] || return 0

  if [ -z "${PLATFORM_CADDYFILE:-}" ]; then
    echo "PLATFORM_CADDYFILE must be set when PLATFORM_DEPLOY_MODE=platform-node" >&2
    exit 1
  fi

  render_dir="$(mktemp -d)"
  python3 "$ROOT/scripts/render_production_templates.py" \
    --ultra-env "$ENV_FILE" \
    --platform-env "$ENV_FILE" \
    --template Caddyfile.platform-node.template \
    --output-dir "$render_dir"
  mkdir -p "$(dirname "$PLATFORM_CADDYFILE")"
  install -m 0644 "$render_dir/Caddyfile.platform-node" "$PLATFORM_CADDYFILE"
  rm -rf "$render_dir"
}

render_keycloak_realm() {
  python3 "$ROOT/deploy/keycloak/sync_keycloak_realm.py" render \
    --output "${KEYCLOAK_REALM_IMPORT_FILE}"
}

reconcile_keycloak_client() {
  python3 "$ROOT/deploy/keycloak/sync_keycloak_realm.py" reconcile \
    --server "$KEYCLOAK_ADMIN_SERVER_URL" \
    --realm "$KEYCLOAK_REALM_NAME"
}

case "$ACTION" in
  up)
    render_platform_proxy
    render_keycloak_realm
    if [ "$BUILD_PLATFORM_IMAGES" = "1" ]; then
      docker compose "${COMPOSE_ARGS[@]}" up -d --build "${SERVICES[@]}"
    else
      docker compose "${COMPOSE_ARGS[@]}" up -d "${SERVICES[@]}"
    fi
    reconcile_keycloak_client
    ;;
  down)
    docker compose "${COMPOSE_ARGS[@]}" down --remove-orphans
    ;;
  logs)
    docker compose "${COMPOSE_ARGS[@]}" logs -f "${SERVICES[@]}"
    ;;
  config)
    render_platform_proxy
    render_keycloak_realm
    docker compose "${COMPOSE_ARGS[@]}" config
    ;;
  *)
    echo "Usage: $0 [up|down|logs|config]" >&2
    exit 1
    ;;
esac
