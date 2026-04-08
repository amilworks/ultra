#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_DIR="${ROOT_DIR}/source"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"

if [ ! -d "${VENV_DIR}" ]; then
  echo "Virtual environment not found at ${VENV_DIR}"
  echo "Run: ./scripts/dev/bootstrap_uv_backend.sh"
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

cd "${SOURCE_DIR}"

PYTHONPATH="${SOURCE_DIR}/bqengine:${SOURCE_DIR}/bqserver:${SOURCE_DIR}/bqcore:${SOURCE_DIR}/bqapi" \
pytest -q "${SOURCE_DIR}/bqengine/bq/engine/tests/test_engine_service_hardening.py"
