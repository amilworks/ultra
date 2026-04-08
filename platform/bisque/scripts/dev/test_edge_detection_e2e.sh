#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_DIR="${ROOT_DIR}/source"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"

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
  echo "${ip}"
}

HOST_ACCESS_IP="${HOST_ACCESS_IP:-$(detect_host_ip)}"
if [ -z "${HOST_ACCESS_IP}" ]; then
  HOST_ACCESS_IP="127.0.0.1"
fi

BISQUE_ROOT="${BISQUE_ROOT:-http://${HOST_ACCESS_IP}:8080}"
ENGINE_ROOT="${ENGINE_ROOT:-http://${HOST_ACCESS_IP}:27000/engine_service/}"
ENGINE_MODULE="${ENGINE_MODULE:-EdgeDetection}"
EDGE_IMAGE_TAG="${EDGE_IMAGE_TAG:-edgedetection:v1.0.0}"
E2E_IMAGE_PATH="${E2E_IMAGE_PATH:-${ROOT_DIR}/test_images/00-db8XD5ccsYdeysUv2Jr9fc.ome-tiff.ome.tif}"
E2E_TIMEOUT_SECONDS="${E2E_TIMEOUT_SECONDS:-300}"
DOCKER_BUILD_PLATFORM="${DOCKER_BUILD_PLATFORM:-linux/amd64}"

if [ ! -d "${VENV_DIR}" ]; then
  echo "Virtual environment not found at ${VENV_DIR}"
  echo "Run: ./scripts/dev/bootstrap_uv_backend.sh"
  exit 1
fi

if [ ! -f "${E2E_IMAGE_PATH}" ]; then
  echo "E2E image not found: ${E2E_IMAGE_PATH}"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required for EdgeDetection E2E"
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

HOST_ACCESS_IP="${HOST_ACCESS_IP}" python - <<'PY'
from pathlib import Path
import re
import os

cfg_path = Path("config/site.cfg")
text = cfg_path.read_text()
host_ip = os.environ["HOST_ACCESS_IP"]
text = re.sub(r"^servers\s*=.*$", "servers = h1,e1", text, flags=re.MULTILINE)
text = re.sub(r"^bisque\.server\s*=.*$", f"bisque.server = http://{host_ip}:8080", text, flags=re.MULTILINE)
text = re.sub(r"^bisque\.engine\s*=.*$", f"bisque.engine = http://{host_ip}:27000", text, flags=re.MULTILINE)
text = re.sub(r"^h1\.services_disabled\s*=.*$", "h1.services_disabled = ", text, flags=re.MULTILINE)
cfg_path.write_text(text)
PY

if ! docker image inspect "${EDGE_IMAGE_TAG}" >/dev/null 2>&1; then
  echo "Building ${EDGE_IMAGE_TAG} from source/modules/EdgeDetection ..."
  docker build --platform "${DOCKER_BUILD_PLATFORM}" -t "${EDGE_IMAGE_TAG}" "${SOURCE_DIR}/modules/EdgeDetection"
else
  echo "Using existing Docker image ${EDGE_IMAGE_TAG}"
fi

bq-admin server stop >/dev/null 2>&1 || true

echo "Starting BisQue host + engine services..."
bq-admin server start

echo "Waiting for host + engine readiness..."
for i in $(seq 1 60); do
  if curl -sS -f "${BISQUE_ROOT}/services" >/dev/null 2>&1 && \
     curl -sS -f "${ENGINE_ROOT%/}/_services" >/dev/null 2>&1; then
    break
  fi
  sleep 1
  if [ "$i" -eq 60 ]; then
    echo "BisQue host/engine failed to become ready"
    exit 1
  fi
done

if ! curl -fsS "${ENGINE_ROOT%/}/_services" | grep -q "name=\"${ENGINE_MODULE}\""; then
  echo "Engine did not advertise module ${ENGINE_MODULE}"
  exit 1
fi

echo "Registering ${ENGINE_MODULE} in module_service..."
bq-admin module register -a -p -u admin:admin -r "${BISQUE_ROOT}" "${ENGINE_ROOT}" >/tmp/edge_register_e2e.log 2>&1 || {
  cat /tmp/edge_register_e2e.log
  echo "Module registration failed"
  exit 1
}

if ! curl -fsS "${BISQUE_ROOT}/module_service/" | grep -q "name=\"${ENGINE_MODULE}\""; then
  echo "module_service missing registered module ${ENGINE_MODULE}"
  exit 1
fi

echo "Running EdgeDetection E2E execution..."
PYTHONPATH="${SOURCE_DIR}/bqapi:${SOURCE_DIR}/bqcore:${SOURCE_DIR}/bqserver" \
BISQUE_ROOT="${BISQUE_ROOT}" \
ENGINE_MODULE="${ENGINE_MODULE}" \
E2E_IMAGE_PATH="${E2E_IMAGE_PATH}" \
E2E_TIMEOUT_SECONDS="${E2E_TIMEOUT_SECONDS}" \
python -u - <<'PY'
import os
import time
from lxml import etree
from bqapi import BQSession

bisque_root = os.environ["BISQUE_ROOT"]
module_name = os.environ["ENGINE_MODULE"]
image_path = os.environ["E2E_IMAGE_PATH"]
timeout_seconds = int(os.environ["E2E_TIMEOUT_SECONDS"])

session = BQSession().init_local(
    "admin",
    "admin",
    moduleuri=f"/module_service/{module_name}",
    bisque_root=bisque_root,
    create_mex=True,
)

resource_xml = etree.Element("image", name=os.path.basename(image_path))
upload_response = session.postblob(image_path, xml=resource_xml)
upload_tree = etree.XML(upload_response)
uploaded = upload_tree.find("./*")
if uploaded is None or uploaded.get("uri") is None:
    raise RuntimeError("Upload did not return a resource URI")
input_image_uri = uploaded.get("uri")
print(f"uploaded_input_uri={input_image_uri}")

mex = etree.Element("mex")
inputs = etree.SubElement(mex, "tag", name="inputs")
etree.SubElement(inputs, "tag", name="Input Image", type="resource", value=input_image_uri)

execute_url = f"{bisque_root}/module_service/{module_name}/execute"
execute_response = session.postxml(execute_url, mex)
if execute_response is None or execute_response.get("uri") is None:
    raise RuntimeError("Module execute did not return a MEX URI")

mex_uri = execute_response.get("uri")
print(f"mex_uri={mex_uri}")

deadline = time.time() + timeout_seconds
last_status = None
final_mex = None
while time.time() < deadline:
    final_mex = session.fetchxml(mex_uri, view="deep")
    status = final_mex.get("value", "")
    if status != last_status:
        print(f"mex_status={status}")
        last_status = status
    if status in {"FINISHED", "FAILED"}:
        break
    time.sleep(2)
else:
    raise RuntimeError(f"MEX timeout after {timeout_seconds}s (last_status={last_status})")

status = final_mex.get("value", "")
if status != "FINISHED":
    error_tags = final_mex.xpath('.//tag[@name="error_message"]')
    error_values = [t.get("value", "") for t in error_tags if t is not None]
    raise RuntimeError(f"MEX finished with status={status}; errors={error_values}")

output_uri = None
for node in final_mex.xpath('./tag[@name="outputs"]//*[@uri]'):
    ntype = (node.get("type") or "").lower()
    if node.tag == "image":
        output_uri = node.get("uri")
        break
    if ntype == "image" and node.tag != "tag":
        output_uri = node.get("uri")
        break

if output_uri is None:
    for node in final_mex.xpath('./tag[@name="outputs"]//*[@value]'):
        ntype = (node.get("type") or "").lower()
        val = node.get("value", "")
        if (node.tag == "image" or ntype == "image" or node.get("name") == "Output Image") and val:
            output_uri = val
            break

if output_uri is None:
    raise RuntimeError("No output image URI found in MEX outputs")

output_resource = session.fetchxml(output_uri, view="short")
print(f"output_image_uri={output_uri}")
print(f"output_image_name={output_resource.get('name', '')}")
print("e2e_result=PASS")
PY

echo "EdgeDetection E2E completed successfully."
