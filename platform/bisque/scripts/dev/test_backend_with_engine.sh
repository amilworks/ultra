#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_DIR="${ROOT_DIR}/source"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
BISQUE_ROOT="${BISQUE_ROOT:-http://127.0.0.1:8080}"
ENGINE_ROOT="${ENGINE_ROOT:-http://127.0.0.1:27000/engine_service/}"
ENGINE_MODULE="${ENGINE_MODULE:-EdgeDetection}"
BISQUE_TEST_IMAGES_DIR="${BISQUE_TEST_IMAGES_DIR:-${SOURCE_DIR}/tests/test_images}"

if [ ! -d "${VENV_DIR}" ]; then
  echo "Virtual environment not found at ${VENV_DIR}"
  echo "Run: ./scripts/dev/bootstrap_uv_backend.sh"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required for engine-backed module workflows"
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "docker daemon is not reachable"
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

"${ROOT_DIR}/scripts/dev/init_test_config.sh"

cd "${SOURCE_DIR}"

tmp_site_cfg="$(mktemp)"
cp "config/site.cfg" "${tmp_site_cfg}"

cleanup() {
  bq-admin server stop >/dev/null 2>&1 || true
  cp "${tmp_site_cfg}" "config/site.cfg" >/dev/null 2>&1 || true
  rm -f "${tmp_site_cfg}"
}
trap cleanup EXIT

python - <<'PY'
from pathlib import Path
import re

cfg_path = Path("config/site.cfg")
text = cfg_path.read_text()

text = re.sub(r"^servers\s*=.*$", "servers = h1,e1", text, flags=re.MULTILINE)
text = re.sub(r"^bisque\.server\s*=.*$", "bisque.server = http://127.0.0.1:8080", text, flags=re.MULTILINE)
text = re.sub(r"^bisque\.engine\s*=.*$", "bisque.engine = http://127.0.0.1:27000", text, flags=re.MULTILINE)
text = re.sub(r"^h1\.services_disabled\s*=.*$", "h1.services_disabled = ", text, flags=re.MULTILINE)

cfg_path.write_text(text)
PY

bq-admin server stop >/dev/null 2>&1 || true

echo "Starting BisQue server with host + engine services..."
bq-admin server start

echo "Waiting for BisQue host + engine readiness..."
for i in $(seq 1 45); do
  if curl -sS -f "${BISQUE_ROOT}/services" >/dev/null 2>&1 && \
     curl -sS -f "${ENGINE_ROOT%/}/_services" >/dev/null 2>&1; then
    break
  fi
  sleep 1
  if [ "$i" -eq 45 ]; then
    echo "BisQue host/engine failed to become ready"
    exit 1
  fi
done

engine_services_xml="$(curl -fsS "${ENGINE_ROOT%/}/_services")"
if ! printf '%s' "${engine_services_xml}" | grep -q "name=\"${ENGINE_MODULE}\""; then
  echo "Expected module not advertised by engine service: ${ENGINE_MODULE}"
  exit 1
fi
echo "Engine service advertises ${ENGINE_MODULE}"

refresh_status="$(curl -sS -o /tmp/engine_refresh.smoke.out -w "%{http_code}" "${ENGINE_ROOT%/}/?refresh=true")"
if [ "${refresh_status}" -ne 200 ]; then
  echo "Engine refresh check failed for /engine_service/?refresh=true (HTTP ${refresh_status})"
  exit 1
fi

engine_asset_status="$(curl -sS -o "/tmp/${ENGINE_MODULE}.engine.asset.out" -w "%{http_code}" "${ENGINE_ROOT%/}/${ENGINE_MODULE}/public/thumbnail.jpg")"
if [ "${engine_asset_status}" -ne 200 ]; then
  echo "Engine module static asset check failed for /engine_service/${ENGINE_MODULE}/public/thumbnail.jpg (HTTP ${engine_asset_status})"
  exit 1
fi
echo "Engine refresh and static asset checks passed"

bad_execute_status="$(curl -sS -o "/tmp/${ENGINE_MODULE}.engine.bad_execute.out" -w "%{http_code}" -H "Content-Type: text/xml" --data-binary '<broken>' "${ENGINE_ROOT%/}/${ENGINE_MODULE}/execute")"
if [ "${bad_execute_status}" -ne 400 ]; then
  echo "Malformed engine execute request expected HTTP 400 but got ${bad_execute_status}"
  exit 1
fi
echo "Engine execute rejects malformed XML with HTTP 400"

echo "Registering engine modules into module_service..."
register_log="/tmp/module_register_with_engine.log"
if ! bq-admin module register -a -p -u admin:admin -r "${BISQUE_ROOT}" "${ENGINE_ROOT}" >"${register_log}" 2>&1; then
  cat "${register_log}"
  echo "Module registration failed"
  exit 1
fi

module_xml="$(curl -fsS "${BISQUE_ROOT}/module_service/")"
if ! printf '%s' "${module_xml}" | grep -q "name=\"${ENGINE_MODULE}\""; then
  echo "module_service missing registered module: ${ENGINE_MODULE}"
  exit 1
fi
echo "module_service contains ${ENGINE_MODULE}"

module_status="$(curl -sS -o "/tmp/${ENGINE_MODULE}.module.smoke.out" -w "%{http_code}" "${BISQUE_ROOT}/module_service/${ENGINE_MODULE}")"
if [ "${module_status}" -eq 404 ] || [ "${module_status}" -ge 500 ]; then
  echo "Registered module endpoint check failed for /module_service/${ENGINE_MODULE} (HTTP ${module_status})"
  exit 1
fi
echo "Module endpoint /module_service/${ENGINE_MODULE} reachable (HTTP ${module_status})"

browser_status="$(curl -sS -o /tmp/client_browser_route.smoke.out -w "%{http_code}" "${BISQUE_ROOT}/client_service/browser?resource=/data_service/image")"
if [ "${browser_status}" -ne 200 ]; then
  echo "Browser route check failed for /client_service/browser?resource=/data_service/image (HTTP ${browser_status})"
  exit 1
fi

for required_asset in "/core/extjs/ext-all.js" "/core/jquery/jquery.min.js"; do
  if ! grep -q "${required_asset}" /tmp/client_browser_route.smoke.out; then
    echo "Browser route HTML missing expected asset reference: ${required_asset}"
    exit 1
  fi
  asset_code="$(curl -sS -o "/tmp/$(basename "${required_asset}").asset.out" -w "%{http_code}" "${BISQUE_ROOT}${required_asset}")"
  if [ "${asset_code}" -ne 200 ]; then
    echo "Asset fetch failed: ${required_asset} (HTTP ${asset_code})"
    exit 1
  fi
done
echo "Browser route and critical UI assets are reachable"

echo "Running backend smoke tests (query + test_images)..."
PYTHONPATH="${SOURCE_DIR}/bqserver:${SOURCE_DIR}/bqcore" \
pytest -q "${SOURCE_DIR}/bqserver/bq/table/tests/test_runquery.py"

PYTHONPATH="${SOURCE_DIR}/bqserver:${SOURCE_DIR}/bqcore" \
BISQUE_TEST_IMAGES_DIR="${BISQUE_TEST_IMAGES_DIR}" \
pytest -q "${SOURCE_DIR}/bqserver/bq/image_service/tests/test_local_test_images_smoke.py"

echo "Engine-backed backend smoke completed."
