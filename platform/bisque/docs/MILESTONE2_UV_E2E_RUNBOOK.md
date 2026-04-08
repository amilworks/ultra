# Milestone 2 Runbook: uv Local BisQue + bqapi Module E2E

Date: 2026-02-17
Repository: `/Users/macbook/Documents/phd/bisque_ultra/bisqueUCSB3`

## Goal
Run BisQue locally with `uv`, verify core services/UI, execute modules end-to-end through `bqapi`, and validate output image retrieval for both built-in and custom modules.

This runbook is written so a human or LLM can reproduce Milestone 2 without hidden steps.

## What this validates
- Host service + engine service startup under `uv`.
- `client_service`, `module_service`, and `image_service` availability.
- Browser/UI route viability (`/client_service/browser?resource=/data_service/image`).
- `EdgeDetection` execution with output image download via `bqapi`.
- Custom module (`SimpleUniversalProcess`) registration + execution on both test inputs.
- Repeat-run stress checks and artifact capture.

## Prerequisites
From repo root:

```bash
cd /Users/macbook/Documents/phd/bisque_ultra/bisqueUCSB3
```

Required:
- `uv`
- Docker daemon running
- Built virtualenv at `.venv` (use Milestone 1 bootstrap)
- Test input files in `test_images/`

Optional but recommended sanity checks:

```bash
./scripts/dev/check_system_deps.sh
```

## 1. Bootstrap uv environment (if needed)

```bash
./scripts/dev/bootstrap_uv_backend.sh
```

Quick import sanity:

```bash
source .venv/bin/activate
python -c "import pkg_resources, tg, bqapi; print('import-ok')"
bq-admin --help >/dev/null && echo bq_admin_ok
```

## 2. Build custom module image and start Milestone 2 environment

Build + tag custom module image:

```bash
./scripts/dev/build_simpleuniversal_module.sh
```

Start host+engine with temporary Milestone 2 config patching and module registration:

```bash
./scripts/dev/start_bisque_m2_env.sh
```

Expected output includes:
- `bisque_root=http://<host-ip>:8080`
- `engine_root=http://<host-ip>:27000/engine_service/`
- `Registered` for `EdgeDetection` and `SimpleUniversalProcess`

## 3. Verify services, UI, and module visibility
Use your host IP reported by start script. Example below uses `192.168.1.70`.

```bash
ROOT=http://192.168.1.70:8080
ENGINE=http://192.168.1.70:27000/engine_service

curl -fsS "$ROOT/services" | rg 'type="client_service"|type="module_service"|type="image_service"'
curl -sS -o /tmp/ui_root.html -w "%{http_code}\n" "$ROOT/"
curl -sS -o /tmp/ui_client.html -w "%{http_code}\n" "$ROOT/client_service/"
curl -sS -o /tmp/ui_browser.html -w "%{http_code}\n" "$ROOT/client_service/browser?resource=/data_service/image"
curl -sS -o /tmp/ext-all.js -w "%{http_code}\n" "$ROOT/core/extjs/ext-all.js"
curl -sS -o /tmp/jquery.min.js -w "%{http_code}\n" "$ROOT/core/jquery/jquery.min.js"

curl -fsS "$ROOT/module_service/" | rg 'name="EdgeDetection"|name="SimpleUniversalProcess"'
curl -fsS "$ENGINE/_services" | rg 'name="EdgeDetection"|name="SimpleUniversalProcess"'
```

Expected:
- `/` => `302`
- `/client_service/` => `200`
- `/client_service/browser?...` => `200`
- extjs/jquery assets => `200`

Reference artifact from validation run:
- `artifacts/m2_e2e/service_ui_validation_20260217.json`

## 4. Built-in EdgeDetection module E2E (single-run harness)

```bash
./scripts/dev/test_edge_detection_e2e.sh
```

Expected in output:
- `mex_status=... -> FINISHED`
- `output_image_uri=http://<host-ip>:8080/data_service/...`
- `e2e_result=PASS`

## 5. bqapi module runner: EdgeDetection (supported input path)
`EdgeDetection` is validated on TIFF image input.

```bash
source .venv/bin/activate
python scripts/dev/bqapi_module_runner.py \
  --bisque-root http://192.168.1.70:8080 \
  --user admin --password admin \
  --module EdgeDetection \
  --images test_images/00-db8XD5ccsYdeysUv2Jr9fc.ome-tiff.ome.tif \
  --repeat 5 \
  --concurrency 1 \
  --summary-json artifacts/m2_e2e/edge_stress_summary_seq.json
```

Expected summary:
- `ok: true`
- `tasks_success: 5`
- output images saved under `artifacts/bqapi_outputs/EdgeDetection_<timestamp>/`

## 6. bqapi module runner: custom module on both test images
Custom module `SimpleUniversalProcess` is used to validate two-input robustness.

```bash
source .venv/bin/activate
python scripts/dev/bqapi_module_runner.py \
  --bisque-root http://192.168.1.70:8080 \
  --user admin --password admin \
  --module SimpleUniversalProcess \
  --images test_images/00-db8XD5ccsYdeysUv2Jr9fc.ome-tiff.ome.tif test_images/NPH_shunt_005_85yo.nii.gz \
  --repeat 2 \
  --concurrency 1 \
  --summary-json artifacts/m2_e2e/custom_bqapi_summary.json
```

Expected summary:
- `ok: true`
- `tasks_success: 4`
- output images saved under `artifacts/bqapi_outputs/SimpleUniversalProcess_<timestamp>/`

Recorded successful artifact:
- `artifacts/m2_e2e/custom_bqapi_summary.json`

## 7. Extended stress evidence
Recorded stress artifacts:
- `artifacts/m2_e2e/edge_stress_summary_seq.json` (`EdgeDetection`, 5/5)
- `artifacts/m2_e2e/custom_stress_summary.json` (`SimpleUniversalProcess`, 10/10)

Inspect summaries:

```bash
jq '{ok,module,tasks_total,tasks_success,tasks_failed,avg_seconds,run_dir}' artifacts/m2_e2e/edge_stress_summary_seq.json
jq '{ok,module,tasks_total,tasks_success,tasks_failed,avg_seconds,run_dir}' artifacts/m2_e2e/custom_bqapi_summary.json
jq '{ok,module,tasks_total,tasks_success,tasks_failed,avg_seconds}' artifacts/m2_e2e/custom_stress_summary.json
```

## 8. Jupyter/local-script style bqapi example
Use this in a notebook cell or Python script:

```python
from lxml import etree
from bqapi import BQSession
from bqapi.util import fetch_blob

ROOT = "http://192.168.1.70:8080"
USER = "admin"
PASSWORD = "admin"
MODULE = "EdgeDetection"
IMAGE_PATH = "test_images/00-db8XD5ccsYdeysUv2Jr9fc.ome-tiff.ome.tif"

session = BQSession().init_local(USER, PASSWORD, bisque_root=ROOT, create_mex=False)

# Upload
resource_xml = etree.Element("image", name="input.tif")
upload_response = session.postblob(IMAGE_PATH, xml=resource_xml)
upload_tree = etree.XML(upload_response)
input_uri = upload_tree.find("./*").get("uri")

# Execute
mex = etree.Element("mex")
inputs = etree.SubElement(mex, "tag", name="inputs")
etree.SubElement(inputs, "tag", name="Input Image", type="resource", value=input_uri)
execute_response = session.postxml(f"/module_service/{MODULE}/execute", mex)
mex_uri = execute_response.get("uri")

# Poll
while True:
    mex_state = session.fetchxml(mex_uri, view="deep")
    status = mex_state.get("value", "")
    if status in {"FINISHED", "FAILED"}:
        break

if status != "FINISHED":
    raise RuntimeError(f"Module failed: {status}")

# Resolve output URI
output_uri = None
for node in mex_state.xpath('./tag[@name="outputs"]//tag[@type="image"]'):
    output_uri = node.get("value") or node.get("uri")
    if output_uri:
        break

if not output_uri:
    raise RuntimeError("No output image URI")

# Download output
downloaded = fetch_blob(session, output_uri, dest="artifacts/bqapi_outputs")
print("output_uri=", output_uri)
print("output_local=", downloaded.get(output_uri))

session.close()
```

## 9. Custom module internals and registration notes
Custom module location:
- `source/modules/SimpleUniversalProcess/`

Key files:
- `source/modules/SimpleUniversalProcess/SimpleUniversalProcess.xml`
- `source/modules/SimpleUniversalProcess/runtime-module.cfg`
- `source/modules/SimpleUniversalProcess/Dockerfile`
- `source/modules/SimpleUniversalProcess/src/BQ_run_module.py`

Behavior:
- Decodable image inputs: grayscale + blur + Canny.
- Non-decodable/binary-like inputs: deterministic byte-to-image fallback, then processing.

Engine registration path:
- `start_bisque_m2_env.sh` runs:
  - host+engine startup
  - `bq-admin module register -a -p -u admin:admin -r <root> <engine_root>`

## 10. Known compatibility findings
1. EdgeDetection input compatibility:
- TIFF input works reliably.
- Mixed run including `NPH_shunt_005_85yo.nii.gz` produced no image output for that item in `EdgeDetection` (`artifacts/m2_e2e/edge_bqapi_summary.json`).
- Use custom module for two-input robustness checks.

2. Apple Silicon + amd64 image emulation:
- Under higher concurrency, occasional long/stalled turnaround can occur.
- Deterministic local validation recommendation: `--concurrency 1`.

3. White page in `client_service`:
- Usually missing static assets deployment.
- Remediation:

```bash
./scripts/dev/init_test_config.sh
```

Then re-check:

```bash
curl -sS -o /tmp/ext-all.js -w "%{http_code}\n" http://192.168.1.70:8080/core/extjs/ext-all.js
curl -sS -o /tmp/jquery.min.js -w "%{http_code}\n" http://192.168.1.70:8080/core/jquery/jquery.min.js
```

## 11. Cleanup / shutdown

```bash
./scripts/dev/stop_bisque_m2_env.sh
```

This stops the local BisQue services and restores `source/config/site.cfg` from backup.

## 12. Repro checklist (copy/paste order)

```bash
cd /Users/macbook/Documents/phd/bisque_ultra/bisqueUCSB3
./scripts/dev/bootstrap_uv_backend.sh
./scripts/dev/build_simpleuniversal_module.sh
./scripts/dev/start_bisque_m2_env.sh
./scripts/dev/test_edge_detection_e2e.sh
source .venv/bin/activate
python scripts/dev/bqapi_module_runner.py --bisque-root http://192.168.1.70:8080 --user admin --password admin --module EdgeDetection --images test_images/00-db8XD5ccsYdeysUv2Jr9fc.ome-tiff.ome.tif --repeat 5 --concurrency 1 --summary-json artifacts/m2_e2e/edge_stress_summary_seq.json
python scripts/dev/bqapi_module_runner.py --bisque-root http://192.168.1.70:8080 --user admin --password admin --module SimpleUniversalProcess --images test_images/00-db8XD5ccsYdeysUv2Jr9fc.ome-tiff.ome.tif test_images/NPH_shunt_005_85yo.nii.gz --repeat 2 --concurrency 1 --summary-json artifacts/m2_e2e/custom_bqapi_summary.json
./scripts/dev/stop_bisque_m2_env.sh
```

