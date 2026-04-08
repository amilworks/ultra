#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_DIR="${ROOT_DIR}/source"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
BISQUE_TEST_IMAGES_DIR="${BISQUE_TEST_IMAGES_DIR:-${SOURCE_DIR}/tests/test_images}"

if [ ! -d "${VENV_DIR}" ]; then
  echo "Virtual environment not found at ${VENV_DIR}"
  echo "Run: ./scripts/dev/bootstrap_uv_backend.sh"
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

"${ROOT_DIR}/scripts/dev/init_test_config.sh"

cd "${SOURCE_DIR}"

cleanup() {
  bq-admin server stop >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Starting BisQue server..."
bq-admin server start

echo "Waiting for http://localhost:8080/services ..."
for i in $(seq 1 30); do
  if curl -sS -f http://localhost:8080/services >/dev/null 2>&1; then
    break
  fi
  sleep 1
  if [ "$i" -eq 30 ]; then
    echo "BisQue failed to become ready on localhost:8080"
    exit 1
  fi
done

services_xml="$(curl -fsS http://localhost:8080/services)"
for required_service in client_service module_service image_service; do
  if ! printf '%s' "${services_xml}" | grep -q "type=\"${required_service}\""; then
    echo "Required service missing from /services: ${required_service}"
    exit 1
  fi
done

for service_path in client_service module_service image_service; do
  status_code="$(curl -sS -o "/tmp/${service_path}.smoke.out" -w "%{http_code}" "http://localhost:8080/${service_path}/")"
  if [ "${status_code}" -eq 404 ] || [ "${status_code}" -ge 500 ]; then
    echo "Service endpoint check failed for /${service_path}/ (HTTP ${status_code})"
    exit 1
  fi
  echo "Service endpoint /${service_path}/ reachable (HTTP ${status_code})"
done

root_status="$(curl -sS -o /tmp/root_ui.smoke.out -w "%{http_code}" "http://localhost:8080/")"
case "${root_status}" in
  200|301|302|303|307|308) ;;
  *)
    echo "UI root endpoint check failed for / (HTTP ${root_status})"
    exit 1
    ;;
esac
echo "UI root endpoint / reachable (HTTP ${root_status})"

client_ui_status="$(curl -sS -o /tmp/client_ui.smoke.out -w "%{http_code}" "http://localhost:8080/client_service/")"
if [ "${client_ui_status}" -ne 200 ]; then
  echo "UI endpoint check failed for /client_service/ (HTTP ${client_ui_status})"
  exit 1
fi
if ! grep -qi "<title>" /tmp/client_ui.smoke.out; then
  echo "UI endpoint /client_service/ did not return expected HTML title"
  exit 1
fi
echo "UI endpoint /client_service/ reachable (HTTP ${client_ui_status})"

browser_ui_status="$(curl -sS -o /tmp/client_browser_ui.smoke.out -w "%{http_code}" "http://localhost:8080/client_service/browser?resource=/data_service/image")"
if [ "${browser_ui_status}" -ne 200 ]; then
  echo "UI browser route check failed for /client_service/browser?resource=/data_service/image (HTTP ${browser_ui_status})"
  exit 1
fi

for required_asset in "/core/extjs/ext-all.js" "/core/jquery/jquery.min.js"; do
  if ! grep -q "${required_asset}" /tmp/client_browser_ui.smoke.out; then
    echo "UI browser route missing expected asset reference: ${required_asset}"
    exit 1
  fi
  asset_code="$(curl -sS -o "/tmp/$(basename "${required_asset}").smoke.asset.out" -w "%{http_code}" "http://localhost:8080${required_asset}")"
  if [ "${asset_code}" -ne 200 ]; then
    echo "Asset fetch failed for ${required_asset} (HTTP ${asset_code})"
    exit 1
  fi
done
echo "UI browser route /client_service/browser?resource=/data_service/image reachable (HTTP ${browser_ui_status})"

echo "Server is up. Running backend smoke tests..."
PYTHONPATH="${SOURCE_DIR}/bqserver:${SOURCE_DIR}/bqcore" \
pytest -q "${SOURCE_DIR}/bqserver/bq/table/tests/test_runquery.py"

PYTHONPATH="${SOURCE_DIR}/bqserver:${SOURCE_DIR}/bqcore" \
BISQUE_TEST_IMAGES_DIR="${BISQUE_TEST_IMAGES_DIR}" \
pytest -q "${SOURCE_DIR}/bqserver/bq/image_service/tests/test_local_test_images_smoke.py"

echo "Backend smoke tests completed."
