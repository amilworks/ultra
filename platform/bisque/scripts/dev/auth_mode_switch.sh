#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "Usage: $0 <legacy|dual|oidc> [site-cfg-path]"
  exit 1
fi

MODE="$1"
case "${MODE}" in
  legacy|dual|oidc) ;;
  *)
    echo "Invalid mode: ${MODE}. Expected one of: legacy, dual, oidc"
    exit 1
    ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SITE_CFG="${2:-${BISQUE_SITE_CFG:-${ROOT_DIR}/source/config/site.cfg}}"

if [ ! -f "${SITE_CFG}" ]; then
  echo "site.cfg not found: ${SITE_CFG}"
  exit 1
fi

LOCAL_TOKEN_ENABLED="${BISQUE_AUTH_LOCAL_TOKEN_ENABLED:-}"
if [ -z "${LOCAL_TOKEN_ENABLED}" ]; then
  if [ "${MODE}" = "oidc" ]; then
    LOCAL_TOKEN_ENABLED="false"
  else
    LOCAL_TOKEN_ENABLED="true"
  fi
fi

python3 - "${SITE_CFG}" "${MODE}" "${LOCAL_TOKEN_ENABLED}" <<'PY'
import configparser
import sys
from pathlib import Path

cfg_path = Path(sys.argv[1])
mode = sys.argv[2].strip().lower()
local_token_enabled = sys.argv[3].strip().lower()

cfg = configparser.RawConfigParser()
cfg.optionxform = str
cfg.read(cfg_path)

if not cfg.has_section("main"):
    cfg.add_section("main")
if not cfg.has_section("app:main"):
    cfg.add_section("app:main")

def setopt(name: str, value: str) -> None:
    cfg.set("main", name, value)
    cfg.set("app:main", name, value)

setopt("bisque.auth.mode", mode)
setopt("bisque.auth.local_token.enabled", local_token_enabled)

with cfg_path.open("w", encoding="utf-8") as fh:
    cfg.write(fh)

print(f"updated {cfg_path} mode={mode} local_token_enabled={local_token_enabled}")
PY
