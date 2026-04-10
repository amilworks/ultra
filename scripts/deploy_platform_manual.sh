#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-up}"
ENV_FILE="${ENV_FILE:-/etc/ultra/platform.env}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PLATFORM_DEPLOY_MODE="${PLATFORM_DEPLOY_MODE:-}"

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

load_env_file "$ENV_FILE"

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

case "$ACTION" in
  up)
    render_platform_proxy
    docker compose "${COMPOSE_ARGS[@]}" up -d --build "${SERVICES[@]}"
    ;;
  down)
    docker compose "${COMPOSE_ARGS[@]}" down --remove-orphans
    ;;
  logs)
    docker compose "${COMPOSE_ARGS[@]}" logs -f "${SERVICES[@]}"
    ;;
  config)
    render_platform_proxy
    docker compose "${COMPOSE_ARGS[@]}" config
    ;;
  *)
    echo "Usage: $0 [up|down|logs|config]" >&2
    exit 1
    ;;
esac
