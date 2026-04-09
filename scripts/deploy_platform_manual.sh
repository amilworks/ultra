#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-up}"
ENV_FILE="${ENV_FILE:-/etc/ultra/platform.env}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_ARGS=(
  --env-file "$ENV_FILE"
  -f "$ROOT/platform/bisque/docker-compose.with-engine.yml"
  -f "$ROOT/platform/bisque/docker-compose.production.yml"
)
SERVICES=(bisque postgres keycloak)

case "$ACTION" in
  up)
    docker compose "${COMPOSE_ARGS[@]}" up -d --build "${SERVICES[@]}"
    ;;
  down)
    docker compose "${COMPOSE_ARGS[@]}" down --remove-orphans
    ;;
  logs)
    docker compose "${COMPOSE_ARGS[@]}" logs -f "${SERVICES[@]}"
    ;;
  config)
    docker compose "${COMPOSE_ARGS[@]}" config
    ;;
  *)
    echo "Usage: $0 [up|down|logs|config]" >&2
    exit 1
    ;;
esac
