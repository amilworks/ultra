#!/usr/bin/env bash

monorepo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

resolve_monorepo_env_file() {
  local root
  root="$(monorepo_root)"
  if [ -n "${ENV_FILE:-}" ]; then
    printf '%s\n' "$ENV_FILE"
    return
  fi
  if [ -f "$root/.env" ]; then
    printf '%s\n' "$root/.env"
    return
  fi
  printf '%s\n' "$root/.env.example"
}

read_monorepo_env() {
  local key default env_file
  key="$1"
  default="${2:-}"
  env_file="${3:-$(resolve_monorepo_env_file)}"
  python3 - "$env_file" "$key" "$default" <<'PY'
import os
import pathlib
import sys

env_file = pathlib.Path(sys.argv[1])
key = sys.argv[2]
default = sys.argv[3]

existing = os.environ.get(key)
if existing not in (None, ""):
    print(existing)
    raise SystemExit(0)

value = None
if env_file.exists():
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        current_key, current_value = line.split("=", 1)
        if current_key.strip() != key:
            continue
        current_value = current_value.strip()
        if len(current_value) >= 2 and current_value[0] == current_value[-1] and current_value[0] in {"'", '"'}:
            value = current_value[1:-1]
        else:
            value = current_value.split(" #", 1)[0].rstrip()
        break

print(value if value is not None else default)
PY
}

platform_compose() {
  local env_file root
  env_file="${1:-$(resolve_monorepo_env_file)}"
  shift || true
  root="$(monorepo_root)"
  docker compose \
    --env-file "$env_file" \
    -f "$root/platform/bisque/docker-compose.with-engine.yml" \
    -f "$root/platform/bisque/docker-compose.oidc.yml" \
    "$@"
}

expected_platform_working_dir() {
  local root
  root="$(monorepo_root)"
  printf '%s\n' "$root/platform/bisque"
}

expected_platform_config_files() {
  local root
  root="$(monorepo_root)"
  printf '%s,%s\n' \
    "$root/platform/bisque/docker-compose.with-engine.yml" \
    "$root/platform/bisque/docker-compose.oidc.yml"
}

platform_container_provenance() {
  local container_name
  container_name="$1"
  docker inspect \
    --format 'project={{index .Config.Labels "com.docker.compose.project"}} wd={{index .Config.Labels "com.docker.compose.project.working_dir"}} files={{index .Config.Labels "com.docker.compose.project.config_files"}} image={{.Config.Image}}' \
    "$container_name" 2>/dev/null || true
}

platform_container_matches_expected() {
  local container_name expected_workdir expected_files actual_workdir actual_files
  container_name="$1"
  expected_workdir="$(expected_platform_working_dir)"
  expected_files="$(expected_platform_config_files)"
  actual_workdir="$(docker inspect --format '{{index .Config.Labels "com.docker.compose.project.working_dir"}}' "$container_name" 2>/dev/null || true)"
  actual_files="$(docker inspect --format '{{index .Config.Labels "com.docker.compose.project.config_files"}}' "$container_name" 2>/dev/null || true)"
  [ "$actual_workdir" = "$expected_workdir" ] && [ "$actual_files" = "$expected_files" ]
}

assert_platform_provenance() {
  local container_name
  container_name="$1"
  if ! docker container inspect "$container_name" >/dev/null 2>&1; then
    echo "Platform container not found: $container_name" >&2
    return 1
  fi
  if platform_container_matches_expected "$container_name"; then
    return 0
  fi
  echo "Existing $container_name does not match the canonical platform compose source." >&2
  echo "Actual: $(platform_container_provenance "$container_name")" >&2
  echo "Expected wd=$(expected_platform_working_dir) files=$(expected_platform_config_files)" >&2
  return 1
}

platform_expected_auth_mode() {
  local env_file mode
  env_file="${1:-$(resolve_monorepo_env_file)}"
  mode="$(read_monorepo_env BISQUE_AUTH_MODE dual "$env_file")"
  case "${mode}" in
    local) printf 'legacy\n' ;;
    legacy|dual|oidc) printf '%s\n' "$mode" ;;
    *) printf 'dual\n' ;;
  esac
}

ensure_platform_running() {
  local env_file container_name
  env_file="${1:-$(resolve_monorepo_env_file)}"
  container_name="$(read_monorepo_env CONTAINER_NAME bisque-server "$env_file")"

  platform_compose "$env_file" config >/dev/null
  if docker container inspect "$container_name" >/dev/null 2>&1; then
    assert_platform_provenance "$container_name"
    platform_compose "$env_file" up -d bisque postgres keycloak >/dev/null
    return 0
  fi

  platform_compose "$env_file" up -d bisque postgres keycloak >/dev/null
}

wait_for_http_ok() {
  local url label attempts sleep_seconds i
  url="$1"
  label="$2"
  attempts="${3:-60}"
  sleep_seconds="${4:-2}"
  i=1
  while [ "$i" -le "$attempts" ]; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$sleep_seconds"
    i=$((i + 1))
  done
  echo "$label did not become ready at $url" >&2
  return 1
}

url_host() {
  python3 - "$1" <<'PY'
from urllib.parse import urlparse
import sys

parsed = urlparse(sys.argv[1])
print(parsed.hostname or "127.0.0.1")
PY
}

url_port() {
  python3 - "$1" <<'PY'
from urllib.parse import urlparse
import sys

parsed = urlparse(sys.argv[1])
if parsed.port:
    print(parsed.port)
elif parsed.scheme == "https":
    print(443)
else:
    print(80)
PY
}
