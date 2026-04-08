#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_DIR="${ROOT_DIR}/source"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"

if [ ! -d "${VENV_DIR}" ]; then
  echo "Missing virtualenv at ${VENV_DIR}"
  echo "Run: ./scripts/dev/bootstrap_uv_backend.sh"
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

detect_host_ip() {
  local ip=""
  if command -v ipconfig >/dev/null 2>&1; then
    ip="$(ipconfig getifaddr en0 2>/dev/null || true)"
    if [ -z "${ip}" ]; then
      ip="$(ipconfig getifaddr en1 2>/dev/null || true)"
    fi
  fi
  if [ -z "${ip}" ] && command -v hostname >/dev/null 2>&1; then
    ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  fi
  if [ -z "${ip}" ]; then
    ip="127.0.0.1"
  fi
  echo "${ip}"
}

HOST_ACCESS_IP="${HOST_ACCESS_IP:-$(detect_host_ip)}"
BISQUE_ROOT="${BISQUE_ROOT:-http://${HOST_ACCESS_IP}:8080}"
ENGINE_ROOT="${ENGINE_ROOT:-http://${HOST_ACCESS_IP}:27000/engine_service/}"
SITE_CFG="${SOURCE_DIR}/config/site.cfg"
SITE_CFG_BACKUP="${SOURCE_DIR}/config/site.cfg.m2.bak"

if [ ! -f "${SITE_CFG_BACKUP}" ]; then
  cp "${SITE_CFG}" "${SITE_CFG_BACKUP}"
fi

SITE_CFG="${SITE_CFG}" HOST_ACCESS_IP="${HOST_ACCESS_IP}" python - <<'PY'
from pathlib import Path
import os
import re

cfg_path = Path(os.environ["SITE_CFG"])
host_ip = os.environ["HOST_ACCESS_IP"]
text = cfg_path.read_text()
text = re.sub(r"^servers\s*=.*$", "servers = h1,e1", text, flags=re.MULTILINE)
text = re.sub(r"^bisque\.server\s*=.*$", f"bisque.server = http://{host_ip}:8080", text, flags=re.MULTILINE)
text = re.sub(r"^bisque\.engine\s*=.*$", f"bisque.engine = http://{host_ip}:27000", text, flags=re.MULTILINE)
text = re.sub(r"^h1\.services_disabled\s*=.*$", "h1.services_disabled = ", text, flags=re.MULTILINE)
cfg_path.write_text(text)
PY

cd "${SOURCE_DIR}"
bq-admin server stop >/dev/null 2>&1 || true
bq-admin server start

for i in $(seq 1 90); do
  if curl -sS -f "${BISQUE_ROOT}/services" >/dev/null 2>&1 && \
     curl -sS -f "${ENGINE_ROOT%/}/_services" >/dev/null 2>&1; then
    break
  fi
  sleep 1
  if [ "${i}" -eq 90 ]; then
    echo "BisQue host/engine readiness timeout"
    exit 1
  fi
done

if [ "${SKIP_REGISTER:-0}" != "1" ]; then
  bq-admin module register -a -p -u admin:admin -r "${BISQUE_ROOT}" "${ENGINE_ROOT}"
fi

echo "bisque_root=${BISQUE_ROOT}"
echo "engine_root=${ENGINE_ROOT}"
echo "site_cfg_backup=${SITE_CFG_BACKUP}"
