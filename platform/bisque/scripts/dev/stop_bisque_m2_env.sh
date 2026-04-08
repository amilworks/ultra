#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_DIR="${ROOT_DIR}/source"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
SITE_CFG="${SOURCE_DIR}/config/site.cfg"
SITE_CFG_BACKUP="${SOURCE_DIR}/config/site.cfg.m2.bak"

if [ -d "${VENV_DIR}" ]; then
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
fi

cd "${SOURCE_DIR}"
bq-admin server stop >/dev/null 2>&1 || true

if [ -f "${SITE_CFG_BACKUP}" ]; then
  cp "${SITE_CFG_BACKUP}" "${SITE_CFG}"
  rm -f "${SITE_CFG_BACKUP}"
fi

echo "bisque_stopped=1"
