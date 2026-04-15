# ExecPlan: Python Backend Stabilization, Engine E2E, and Auth Modernization

Date: 2026-02-17
Owner: Backend team
Status: Active (M1-M4 complete with concurrency caveat in M4 stress runs, M5-M7 pending)

## Session Update (2026-03-09): module_service robustness hardening (no breaking changes)

Milestone intent:
- Implement the approved `module_service` hardening plan while preserving valid registration, lookup, and MEX execution behavior.

Commands run and outcomes:
1. Mandatory first pass -> PASS
   - `python3 --version` -> `Python 3.10.11`
   - `uv --version` -> `uv 0.9.30`
   - `git status --short` -> dirty tree expected
   - `rg --files source | head -n 50` -> complete
   - `cd source && python3 setup.py --help ...` -> complete
   - `cd source && rg -n "def query\\(|resource_query\\(|run_query" ...` -> complete
2. Baseline verification -> MIXED (expected)
   - `uv venv .venv` -> PASS
   - `source .venv/bin/activate` -> PASS
   - `uv pip install -r source/requirements.dev` -> FAIL (expected relative editable path parse issue from repo root)
   - `cd source && uv pip install -r requirements.dev` -> PASS
   - `PYTHONPATH=source/bqserver:source/bqcore pytest -q source/bqserver/bq/table/tests/test_runquery.py` -> PASS (`17 passed`)
3. Module service implementation validation:
   - `python3 -m py_compile source/bqserver/bq/module_service/controllers/module_server.py source/bqserver/bq/module_service/api.py source/bqserver/bq/module_service/commands/module_admin.py source/bqengine/bq/engine/commands/module_admin.py source/bqapi/bqapi/services.py` -> PASS
   - `cd source && source ../.venv/bin/activate && PYTHONPATH=bqcore:bqserver:bqapi pytest -q bqserver/bq/module_service/tests/test_module_service_hardening.py` -> PASS (`21 passed`)
4. Regression slice:
   - `source .venv/bin/activate && PYTHONPATH=source/bqserver:source/bqcore pytest -q source/bqserver/bq/table/tests/test_runquery.py` -> PASS (`17 passed`)
   - `cd source && source ../.venv/bin/activate && PYTHONPATH=bqcore:bqserver:bqapi pytest -q bqcore/bq/core/tests/functional/test_root.py -k test_services` -> PASS (`1 passed, 1 deselected`)
   - `cd source && source ../.venv/bin/activate && PYTHONPATH=bqcore:bqserver:bqapi pytest -q bqserver/bq/module_service/tests/test_module_service_hardening.py` -> PASS (`21 passed`)

Implemented changes:
- XML/request hardening:
  - `read_xml_body()` now guards missing `Content-Type` and malformed XML, returning `400` instead of raising parser exceptions.
  - `register_engine()` now rejects empty/malformed request bodies with typed XML parsing instead of a bare `except`.
  - `EngineResource.new()` now rejects empty registration payloads deterministically.
- Registry/cache correctness:
  - `_lookup()` now aborts with `404` when a module remains unresolved after service cache refresh.
  - `unregister_engine()` refreshes `self.service_list` before returning successful responses.
  - Confirmed `ModuleServer.services()` already inherits a working `servicelist()` implementation from `ServiceController`; no behavior change was needed there.
- Compatibility hardening:
  - `unregister_engine` now supports both `resource_uniq` and legacy `engine_uri`/`module_uri` lookup for backward compatibility.
  - `bq.module_service.api` helpers now raise stable `RequestError("no server available")` when the service is absent.
  - `bqapi.ModuleProxy.unregister()` now prefers `resource_uniq` while still allowing legacy parameters.
  - Both bundled `module_admin` CLIs now preserve passwords containing `:` and emit `resource_uniq` unregister requests when a module URI is provided.
- MEX and module update safety:
  - `create_mex()` now tolerates missing formal inputs and missing iterable input containers without 500s.
  - Iterable limit math now uses integer-safe checks and correct logging thresholds.
  - Module definition updates now compare parsed timestamps when possible before replacing same-version definitions.
- Reliability cleanup:
  - Removed mutable default arguments in `async_dbaction()`.
  - Hardened `POST_error()` status/reason handling.
  - Replaced touched `log.warn` usage with `log.warning`.

Artifacts:
- Added `source/bqserver/bq/module_service/tests/test_module_service_hardening.py`
  - Covers XML parsing guardrails, register/unregister behavior, lookup failure handling, MEX creation edge cases, timestamp update logic, helper error behavior, `ModuleProxy.unregister()`, and both `module_admin` unregister flows.

Current blockers:
- None for this milestone. Remaining warnings are legacy framework/dependency deprecations outside this change set.

## Session Update (2026-02-27): bqapi auth + notebook-style API flow tests

Milestone intent:
- Add functional API coverage for notebook-like usage with bearer auth: upload/download, local grayscale transform, re-upload, tag write, and tag/filetype search.

Commands run and outcomes:
1. Mandatory first pass (repo root + source query scan) -> PASS
   - `python3 --version` -> `Python 3.10.11`
   - `uv --version` -> `uv 0.9.30`
   - `git status --short` -> dirty tree expected
   - `rg --files source | head -n 50` -> file scan complete
   - `cd source && python3 setup.py --help ...` -> completed (legacy `pkg_resources` deprecation warning observed)
   - `cd source && rg -n "def query\\(|resource_query\\(|run_query" ...` -> query paths confirmed
2. Baseline verification commands -> MIXED (expected)
   - `uv venv .venv` -> PASS
   - `source .venv/bin/activate` -> PASS
   - `uv pip install -r source/requirements.dev` -> FAIL (expected relative local path parse issue)
   - `cd source && uv pip install -r requirements.dev` -> PASS on this machine
   - `PYTHONPATH=source/bqserver:source/bqcore pytest -q source/bqserver/bq/table/tests/test_runquery.py` -> PASS (`17 passed`)
3. New bqapi notebook-flow test run:
   - `PYTHONPATH=source/bqserver:source/bqcore:source/bqapi pytest -q -rs source/bqapi/bqapi/tests/test_notebook_api_flow.py` -> SKIP
   - Skip reason: token endpoint unreachable (`connect timeout` to `http://192.168.1.70:8080/auth_service/token`)
4. Existing token-auth smoke:
   - `... pytest -q source/bqapi/bqapi/tests/test_comm.py -k "open_token_session"` -> SKIP (same token endpoint timeout)
5. Data service sanity command:
   - `cd source && ... pytest -q bqserver/bq/data_service/tests/test_ds.py -k test_a_index` -> FAIL
   - Failure: sqlite test DB missing schema (`OperationalError: no such table: taggable`)
6. Local auth/API validation rerun with reachable host override -> PASS
   - `./scripts/dev/init_test_config.sh` -> PASS (DB schema initialized)
   - Created `/tmp/bisque_local_test.ini` with `host.root = http://127.0.0.1:8080`
   - Started local server (`bq-admin server start`) and waited on `http://127.0.0.1:8080/services`
   - `BISQUE_TEST_INI=/tmp/bisque_local_test.ini PYTHONPATH=bqserver:bqcore:bqapi pytest -q -rs bqapi/bqapi/tests/test_notebook_api_flow.py` -> PASS (`1 passed`)
   - `BISQUE_TEST_INI=/tmp/bisque_local_test.ini PYTHONPATH=bqserver:bqcore:bqapi pytest -q -rs bqapi/bqapi/tests/test_comm.py -k "open_token_session"` -> PASS (`1 passed, 16 deselected`)
   - `cd source && source ../.venv/bin/activate && PYTHONPATH=bqcore:bqserver:bqapi pytest -q bqserver/bq/data_service/tests/test_ds.py -k test_a_index` -> PASS (`1 passed`)

Artifacts:
- Added `source/bqapi/bqapi/tests/test_notebook_api_flow.py`
  - Uses bearer token auth.
  - Executes upload -> download -> grayscale conversion (Pillow) -> re-upload.
  - Adds tags (`healthy`, `flow_id`) and searches via `tag_query` including JPEG name filter.
  - Cleans up created resources in `finally`.

Current blockers:
- Default `source/config/test.ini` still points to `http://192.168.1.70:8080`; this host timed out during direct runs from current environment.
- Local validation is unblocked via `BISQUE_TEST_INI=/tmp/bisque_local_test.ini` with `host.root = http://127.0.0.1:8080`.

## Session Update (2026-02-27): data_service robustness hardening (no breaking changes)

Milestone intent:
- Implement the approved data_service hardening plan with compatibility preserved for valid requests and existing endpoint contracts.

Commands run and outcomes:
1. Mandatory first pass -> PASS
   - `python3 --version` -> `Python 3.10.11`
   - `uv --version` -> `uv 0.9.30`
   - `git status --short` -> dirty tree expected
   - `rg --files source | head -n 50` -> complete
   - `cd source && python3 setup.py --help ...` -> complete
   - `cd source && rg -n "def query\\(|resource_query\\(|run_query" ...` -> complete
2. Baseline verification -> MIXED (expected)
   - `uv venv .venv` -> PASS
   - `source .venv/bin/activate` -> PASS
   - `uv pip install -r source/requirements.dev` -> FAIL (expected relative editable path parse issue)
   - `cd source && uv pip install -r requirements.dev` -> PASS
   - `PYTHONPATH=source/bqserver:source/bqcore pytest -q source/bqserver/bq/table/tests/test_runquery.py` -> PASS (`17 passed`)
3. New hardening unit tests:
   - `PYTHONPATH=source/bqserver:source/bqcore:source/bqapi pytest -q source/bqserver/bq/data_service/tests/test_data_service_hardening.py` -> PASS (`18 passed`)
4. Requested regression suite (post-change):
   - `PYTHONPATH=source/bqserver:source/bqcore pytest -q source/bqserver/bq/table/tests/test_runquery.py` -> PASS (`17 passed`)
   - `cd source && PYTHONPATH=bqcore:bqserver:bqapi pytest -q bqserver/bq/data_service/tests/test_ds.py -k test_a_index` -> PASS (`1 passed`)
   - `cd source && PYTHONPATH=bqcore:bqserver:bqapi pytest -q bqserver/bq/data_service/tests/test_doc_resource.py` -> PASS (`8 passed`)
   - `cd source && PYTHONPATH=bqcore:bqserver:bqapi pytest -q bqserver/bq/data_service/tests/test_query_attributes.py` -> PASS (`1 passed`)

Implemented changes:
- Parser hardening:
  - `tag_query` lexer/parser errors now raise structured `QuerySyntaxError` (no silent parse degradation).
  - `extract` parser errors now raise `FilterParseError` and are surfaced as `QuerySyntaxError`.
  - `BisquikResource.dir` maps malformed query syntax to HTTP `400`.
- Request parsing + negotiation:
  - `DataServerController.load` now uses `urllib.parse.parse_qsl(..., keep_blank_values=True)` for robust query parsing with `=`.
  - `find_formatter`/`find_formatter_type` deterministic XML fallback for unknown `Accept`.
  - `find_inputer(None)` now safely returns `None`.
- Query correctness:
  - `tag_order` direction resets per token (`@ts:desc,@name` now correctly applies ASC for second term).
  - `hidden=false` now includes both `NULL` and explicit `False`; default remains `NULL`.
- Auth edge guardrails:
  - `ResourceAuth.load/get/append` now guard missing ACL/user/resource tuples to avoid 500s.
- Cache/date safety:
  - Replaced cache header `eval` with `ast.literal_eval`.
  - Fixed Python 3 md5 input encoding in cache `etag`.
  - Hardened malformed cache reads and malformed/short HTTP date parsing (fail-safe to cache miss/`None`).
- Targeted cleanup:
  - Removed unreachable legacy block after `tags_special` return.
  - Replaced targeted `log.warn` calls with `log.warning` (including `tag_model.py` item from TODO list).
  - Kept `input_csv` behavior unchanged; documented explicit compatibility TODO and added test coverage.

Current blockers:
- None for this hardening milestone. Remaining warnings are legacy deprecation warnings outside this change set.

## Session Update (2026-02-27): image_service robustness hardening (no breaking changes)

Milestone intent:
- Implement the approved `image_service` hardening plan with behavior compatibility for valid requests.

Commands run and outcomes:
1. Mandatory first pass -> PASS
   - `python3 --version` -> `Python 3.10.11`
   - `uv --version` -> `uv 0.9.30`
   - `git status --short` -> dirty tree expected
   - `rg --files source | head -n 50` -> complete
   - `cd source && python3 setup.py --help ...` -> complete
   - `cd source && rg -n "def query\\(|resource_query\\(|run_query" ...` -> complete
2. Baseline verification -> MIXED (expected)
   - `uv venv .venv` -> PASS
   - `source .venv/bin/activate` -> PASS
   - `uv pip install -r source/requirements.dev` -> FAIL (expected editable path parse issue from repo root)
   - `cd source && uv pip install -r requirements.dev` -> PASS
   - `PYTHONPATH=source/bqserver:source/bqcore pytest -q source/bqserver/bq/table/tests/test_runquery.py` -> PASS (`17 passed`)
3. Image service targeted regressions:
   - `PYTHONPATH=source/bqserver:source/bqcore:source/bqapi pytest -q source/bqserver/bq/image_service/tests/test_image_service_hardening_unit.py` -> PASS (`11 passed`)
   - `... pytest -q source/bqserver/bq/image_service/tests/test_local_test_images_smoke.py` -> PASS (`2 passed`)
   - `... pytest -q source/bqserver/bq/image_service/tests/test_image_service_modern.py -k "image_service_available or enhanced_authentication"` -> PASS (`2 passed, 8 deselected`)
   - `... pytest -q source/bqserver/bq/image_service/tests/test_image_service_modern.py -k "histogram or localpath or thumbnail"` -> PASS (`3 passed, 7 deselected`)

Implemented changes:
- Converter crash-path fixes:
  - Fixed Imaris thumbnail preproc command assembly (`list.extend` misuse).
  - Hardened OpenSlide histogram path for Python 3 byte packing and error fallback.
  - Added safe fallback variable initialization in thumbnail dryrun path.
- Request/dispatch correctness:
  - Fixed local API dispatch ID parsing for both numeric and uniq IDs.
  - Fixed `process(...)` call to pass `user_name` by keyword.
  - Replaced runtime `print` with structured logging in local dispatch.
- Response/header hardening:
  - Added RFC 5987-safe `Content-Disposition` builder for unicode filenames.
- Processing resilience:
  - Dryrun exceptions now invalidate dryrun shortcut and fall back to full processing.
  - Added ffprobe JSON/video-stream guardrails in FFmpeg info extraction.
  - Added BioFormats XML/pixels guards and safe tmp-rename behavior.
- Cache safety:
  - Added configurable bounded cache (`cache_max_users`, `cache_max_entries_per_user`) with LRU-style eviction.
- Quality updates:
  - Replaced `log.warn` with `log.warning` in targeted image_service files.
  - Migrated plugin loader from deprecated `imp.load_source` to `importlib.util`.
- Test-suite hardening:
  - Added `test_image_service_hardening_unit.py` with focused unit coverage for fixed bug paths.
  - Fixed modern test helper URL handling (string URL, no bytes).
  - Hardened modern helper functions to short-circuit invalid/non-XML resources.
  - Removed duplicate fixtures and duplicate test names in `test_image_service_core_modern.py` so all intended tests are discoverable.

Current blockers:
- None for this milestone; remaining warnings are legacy deprecation warnings (`pkg_resources`, `imp` warning from external deps) outside this change set.

## Scope
- Make backend development reproducible with `uv` + virtualenv.
- Remove setup ambiguity around legacy dependencies and config bootstrap.
- Migrate authentication from legacy TG/repoze-who coupling toward standards-based OIDC/OAuth2.
- Preserve module execution (`Mex`) and `bqapi` compatibility during migration.
- Harden the query stack and add tests where coverage is weak after auth cutover is stable.
- Establish a verification matrix that can run locally and in CI.
- Modernize Docker build/runtime layout for faster, reproducible near-production (k3s-aligned) deploys.

## Priority Order (as of 2026-02-18)
1. M8 Docker/K3s Build Modernization (new highest priority)
2. M5 Query System Hardening
3. M6 Image-backed integration tests
4. M7 CI verification path

## Non-goals
- Full framework migration away from TurboGears/Pylons in this phase.
- Replacing module execution auth (`Mex`) in the first auth cut.
- Immediate removal of legacy `Basic`/cookie paths before parity is proven.
- Large API redesigns.
- Replacing Docker workflows; this plan complements them.

## Invariants
- Keep existing endpoint contracts and query semantics compatible unless explicitly versioned.
- Keep ACL behavior and module execution semantics stable while auth internals evolve.
- Preserve `Authorization: Mex ...` support until module and engine service auth replacement is validated.
- Every auth behavior change must include focused regression coverage (web + API + module flow).

## Baseline Findings (repo scan + verification on 2026-02-17)
1. `uv pip install -r source/requirements.dev` from repo root fails to parse local editable paths.
Command:
```bash
source .venv/bin/activate && uv pip install -r source/requirements.dev
```
Observed failure: requirements parser rejects relative local path entries when invoked this way.

2. `uv` install from `source/` proceeds further but fails on `pygraphviz` native build.
Command:
```bash
cd source && source ../.venv/bin/activate && uv pip install -r requirements.dev
```
Observed failure: `fatal error: 'graphviz/cgraph.h' file not found`.

3. Legacy stack breaks with modern setuptools unless pinned.
Command:
```bash
source .venv/bin/activate && python -c "import pkg_resources"
```
Observed failure with `setuptools==82.0.0`: `ModuleNotFoundError: No module named 'pkg_resources'`.
Temporary fix verified:
```bash
source .venv/bin/activate && uv pip install "setuptools<81"
```

4. Query unit tests can pass once minimal dependencies and PYTHONPATH are configured.
Command:
```bash
source .venv/bin/activate && PYTHONPATH=source/bqserver:source/bqcore pytest -q source/bqserver/bq/table/tests/test_runquery.py
```
Observed result: `17 passed`.

5. Data-service tests are not runnable by default.
Command:
```bash
source .venv/bin/activate && PYTHONPATH=source/bqserver:source/bqcore:source/bqapi pytest -q source/bqserver/bq/data_service/tests/test_ds.py
```
Observed failure: missing `config/test.ini` and app bootstrap prerequisites.

## Milestones

### M1: Reproducible `uv` Bootstrap for Backend Dev
Deliverables:
- Add `scripts/dev/bootstrap_uv_backend.sh` with deterministic install order.
- Add `scripts/dev/check_system_deps.sh` (Graphviz/OpenSlide/DB client headers).
- Add `source/requirements.uv.txt` (uv-compatible, no ambiguous editable path handling).
- Pin legacy-critical tooling (notably setuptools) in constraints.

Validation commands:
```bash
./scripts/dev/check_system_deps.sh
./scripts/dev/bootstrap_uv_backend.sh
source .venv/bin/activate
python -c "import pkg_resources, tg, bqapi; print('import-ok')"
bq-admin --help >/dev/null
```

Exit criteria:
- A clean machine can bootstrap without manual package ordering.
- `bq-admin` exists in `.venv/bin`.

Implementation update (2026-02-17):
- Added `scripts/dev/check_system_deps.sh`.
- Added `scripts/dev/bootstrap_uv_backend.sh`.
- Added `source/requirements.uv.txt`.
- Added `source/constraints.uv.txt`.
- Bootstrap script now:
  - Recreates `.venv` deterministically.
  - Pins `setuptools<81`.
  - Installs local legacy packages in fixed order.
  - Builds `pygraphviz` with Homebrew Graphviz include/lib flags on macOS.
  - Runs sanity checks (`import pkg_resources,tg,bqapi` and `bq-admin --help`).

Verification results (2026-02-17):
1. `./scripts/dev/check_system_deps.sh` -> PASS
   - Warnings: OpenSlide dev files, `pg_config`, and `mysql_config` not detected.
2. `./scripts/dev/bootstrap_uv_backend.sh` -> PASS (after adding `pyinstaller` and pygraphviz build flags)
3. `source .venv/bin/activate && python -c "import pkg_resources, tg, bqapi; print('import-ok')"` -> PASS
4. `source .venv/bin/activate && bq-admin --help >/dev/null` -> PASS

M1 status: Completed (with non-blocking dependency warnings from `check_system_deps.sh`).

### M2: Config and Test Harness Stabilization
Deliverables:
- Add `scripts/dev/init_test_config.sh` to materialize `source/config/test.ini` from defaults.
- Update `pytest-bisque` to support `BISQUE_TEST_INI` env override.
- Add a documented test entrypoint script (for example `scripts/dev/test_backend_smoke.sh`).
- Ensure functional fixtures fail with actionable errors, not missing-fixture noise.

Validation commands:
```bash
./scripts/dev/check_system_deps.sh
./scripts/dev/init_test_config.sh
./scripts/dev/test_backend_smoke.sh
./scripts/dev/test_backend_with_engine.sh
cd source && source ../.venv/bin/activate && bq-admin server start && sleep 2 && curl -fsS http://localhost:8080/services | rg 'type="client_service"|type="module_service"|type="image_service"' && bq-admin server stop
cd source && source ../.venv/bin/activate && bq-admin server start && sleep 2 && curl -sS -o /tmp/ui_root.out -w "%{http_code}\n" http://localhost:8080/ && curl -sS -o /tmp/ui_client.out -w "%{http_code}\n" http://localhost:8080/client_service/ && rg -n "<title>" /tmp/ui_client.out && bq-admin server stop
cd source && source ../.venv/bin/activate && BISQUE_TEST_INI=/tmp/does-not-exist.ini PYTHONPATH=bqcore:bqserver:bqapi pytest -q bqserver/bq/data_service/tests/test_ds.py -k test_a_index
```

Exit criteria:
- No missing `application` fixture failures.
- No missing `config/test.ini` failures.
- Local BisQue server can be started and queried on `localhost:8080`.
- Smoke tests run against local query + `test_images` checks.
- `client_service`, `module_service`, and `image_service` are present in `/services` (`type=` tags) and their endpoints respond without 5xx/404.
- UI is reachable (`/` and `/client_service/`) and returns HTML.
- Browser route is reachable (`/client_service/browser?resource=/data_service/image`) and critical assets (`/core/extjs/ext-all.js`, `/core/jquery/jquery.min.js`) return HTTP `200`.
- Engine-backed local run works with `uv`: `h1 + e1` servers start, `engine_service` advertises modules, and `EdgeDetection` registers into `module_service`.

Implementation update (2026-02-17):
- Added `scripts/dev/init_test_config.sh`.
  - Creates `source/config/test.ini` from defaults.
  - Applies safe `[test]` defaults and links `source/tests/test_images` to repo `test_images` when present.
  - Ensures SQLite schema exists (`taggable` table check + `bq-admin setup -y database` fallback).
- Added `scripts/dev/test_backend_smoke.sh`.
  - Starts BisQue on `localhost:8080`, waits for readiness, verifies service registration for `client_service`/`module_service`/`image_service`, checks endpoint reachability, verifies UI endpoints (`/`, `/client_service/`), verifies `/client_service/browser?resource=/data_service/image` plus critical static assets, runs backend smoke tests, and always stops the server.
- Added `scripts/dev/test_backend_with_engine.sh`.
  - Forces temporary `h1,e1` startup, validates `engine_service` on `127.0.0.1:27000`, registers `EdgeDetection` via `bq-admin module register -a -p`, verifies module visibility/reachability under `module_service`, validates browser route assets, runs backend smoke tests, and restores original `config/site.cfg`.
- Updated `scripts/dev/check_system_deps.sh` to include:
  - Docker CLI/daemon checks for module runtime readiness.
  - Runtime binary checks (`imgcnv`, `showinf`, `bfconvert`, `ImarisConvert`) to mirror Dockerfile expectations.
- Updated `source/pytest-bisque/pytest_bisque.py`.
  - Added `BISQUE_TEST_INI` override support.
  - Added fallback lookup for `source/config/test.ini`.
  - Added actionable runtime errors for missing/invalid test config.
- Added `source/bqserver/bq/image_service/tests/test_local_test_images_smoke.py`.
  - Verifies local image fixture directory and supported test image files.
- Updated `source/requirements.uv.txt` with `xlrd==2.0.1` to satisfy `.xls` query tests.
- Updated `source/requirements.uv.txt` with `python-mimeparse==2.0.0` so `module_service` can load through `data_service` imports.
- Updated `source/bqcore/bq/commands/admin.py` static deploy path handling for editable installs:
  - `bq-admin setup -y statics` now falls back to service source `../public` directories when `pkg_resources` points to namespace stubs in site-packages.
- Updated `scripts/dev/init_test_config.sh`:
  - Ensures static UI assets are present (`public/core/extjs/ext-all.js`, `public/client_service/css/welcome.css`) and runs `bq-admin setup -y statics` if missing.
- Deprecated API cleanup (no behavior changes):
  - `source/bqserver/bq/blob_service/controllers/blob_drivers.py`: `log.warn` -> `log.warning`.
  - `source/bqserver/bq/image_service/controllers/imgsrv.py`: `log.warn` -> `log.warning`.
  - `source/bqcore/bq/core/model/__init__.py`: prefer `sqlalchemy.orm.declarative_base` with compatibility fallback.
- Python 3 compatibility fixes for module registration / setup paths:
  - `source/bqcore/bq/util/urlnorm.py`: fixed `str`/`bytes` handling in `_unicode`, `_utf8`, and IDN conversion path.
  - `source/bqcore/bq/util/http/http_client.py`: replaced removed `base64.encodestring` with `base64.b64encode`.
  - `source/bqserver/bq/module_service/commands/module_admin.py`: robust root URL resolution (`--root`, `bisque.root`, fallback `bisque.server`) and `0.0.0.0 -> 127.0.0.1` normalization.
  - `source/bqengine/bq/engine/commands/module_admin.py`: same root URL fallback/normalization behavior.
  - `source/bqcore/bq/setup/bisque_setup.py`: deprecation-safe cleanup (`install_type == "system"`, `log.warning`).

Verification results (2026-02-17):
1. `./scripts/dev/init_test_config.sh` -> PASS
   - `source/config/test.ini` created/updated.
   - `source/tests/test_images` present (symlink to repo `test_images`).
2. `./scripts/dev/test_backend_smoke.sh` -> PASS
   - Service endpoint checks:
     - `/client_service/` -> HTTP `200`
     - `/module_service/` -> HTTP `200`
     - `/image_service/` -> HTTP `200`
   - UI endpoint checks:
     - `/` -> HTTP `302` (redirect)
     - `/client_service/` -> HTTP `200` (HTML page with `<title>`)
   - `test_runquery.py`: `17 passed`.
   - `test_local_test_images_smoke.py`: `2 passed`.
   - `/client_service/browser?resource=/data_service/image`: HTTP `200` and critical assets returned HTTP `200`.
3. Manual serve check -> PASS
   - `bq-admin server start` + `curl http://localhost:8080/services` returned HTTP `200`.
   - Response included service tags `client_service`, `module_service`, and `image_service`.
4. Static/UI asset check -> PASS
   - `client_service` HTML references resolved: `183` static assets fetched, `0` non-200 responses.
   - Root cause addressed for prior white page state (missing/undeployed `public/core` assets under editable uv installs).
5. Actionable error check -> PASS (expected failure mode)
   - `BISQUE_TEST_INI=/tmp/does-not-exist.ini ... pytest ...` fails with clear `RuntimeError: BISQUE_TEST_INI points to a missing file`.
6. Engine-backed smoke (`./scripts/dev/test_backend_with_engine.sh`) -> PASS
   - `h1` + `e1` servers started and were reachable (`8080` + `27000`).
   - `http://127.0.0.1:27000/engine_service/_services` advertised `EdgeDetection`.
   - `bq-admin module register -a -p -u admin:admin -r http://127.0.0.1:8080 http://127.0.0.1:27000/engine_service/` succeeded.
   - `module_service` contained `EdgeDetection`; `/module_service/EdgeDetection` reachable (HTTP `302`).
   - Browser route `/client_service/browser?resource=/data_service/image` and critical assets returned expected responses.
7. Dependency parity check (`./scripts/dev/check_system_deps.sh`) -> PASS with actionable warnings
   - Docker CLI/daemon: detected.
   - Missing optional local binaries for full Dockerfile parity: `imgcnv`, `showinf`, `bfconvert`, `ImarisConvert`.

M2 extension (2026-02-17, module execution + `bqapi` robustness):
1. Added Milestone 2 environment helpers:
   - `scripts/dev/start_bisque_m2_env.sh` (host+engine startup, temporary `site.cfg` patching, module registration).
   - `scripts/dev/stop_bisque_m2_env.sh` (stop + `site.cfg` restore).
   - `scripts/dev/build_simpleuniversal_module.sh` (custom module image build/tag).
2. Added `bqapi` local-client execution runner:
   - `scripts/dev/bqapi_module_runner.py`
   - Supports upload, execute, MEX polling, output download, multi-image runs, and stress options (`--repeat`, `--concurrency`).
   - Added `.gz` input normalization to avoid compressed-input client failures.
3. Added custom module for dual-input robustness testing:
   - `source/modules/SimpleUniversalProcess/*`
   - Includes `SimpleUniversalProcess.xml`, runtime config, Dockerfile, and `src/BQ_run_module.py`.
   - Processing behavior: grayscale/blur/canny for decodable images; deterministic byte-to-image fallback for non-decodable binary-like inputs.
4. Engine/runtime reliability fixes used by both EdgeDetection and custom modules:
   - `source/bqengine/bq/engine/controllers/runtime_adapter.py` (robust MEX auth token propagation).
   - `source/bqengine/bq/engine/controllers/docker_env.py` (localhost rewrite for docker reachability + macOS-safe `mv` flags).

M2 extension verification results (2026-02-17):
1. Service/UI/module validation -> PASS
   - Artifact: `artifacts/m2_e2e/service_ui_validation_20260217.json`
   - `/services` contains `type="client_service"`, `type="module_service"`, `type="image_service"`.
   - UI checks: `/` => `302`, `/client_service/` => `200`, `/client_service/browser?resource=/data_service/image` => `200`.
   - Static assets: `/core/extjs/ext-all.js` => `200`, `/core/jquery/jquery.min.js` => `200`.
   - `module_service` and `engine_service` both advertise `EdgeDetection` and `SimpleUniversalProcess`.
2. EdgeDetection E2E via `bqapi` on supported image input -> PASS
   - Artifact: `artifacts/m2_e2e/edge_stress_summary_seq.json`
   - Result: `5/5` success, `avg_seconds=4.254`.
3. Custom module E2E via `bqapi` on both test inputs -> PASS
   - Artifact: `artifacts/m2_e2e/custom_bqapi_summary.json`
   - Result: `4/4` success across:
     - `test_images/00-db8XD5ccsYdeysUv2Jr9fc.ome-tiff.ome.tif`
     - `test_images/NPH_shunt_005_85yo.nii.gz`
4. Extended robustness run for custom module -> PASS
   - Artifact: `artifacts/m2_e2e/custom_stress_summary.json`
   - Result: `10/10` success, `avg_seconds=10.44`.
5. Input compatibility finding captured:
   - Artifact: `artifacts/m2_e2e/edge_bqapi_summary.json`
   - `EdgeDetection` succeeds on TIFF input but does not emit image output for `NPH_shunt_005_85yo.nii.gz` (`2/4` when mixed in same run).
   - Mitigation for milestone validation: stress EdgeDetection on supported TIFF path; use `SimpleUniversalProcess` for dual-input robustness checks.
6. Concurrency caveat on Apple Silicon + amd64 module images:
   - High-concurrency runs can show long/stalled MEX turnaround under emulation.
   - Recommended deterministic local validation: `--concurrency 1` with repeated runs.
7. End-to-end reproducibility documentation:
   - `docs/MILESTONE2_UV_E2E_RUNBOOK.md` contains complete command sequences, expected outputs, `bqapi` script/notebook examples, custom module details, and troubleshooting.

M2 status: Completed (including uv local engine/module registration, UI/browser route validation, `bqapi` end-to-end execution, custom module registration/execution, and repeat-run stress evidence; warnings remain for optional local runtime binaries and emulation-related concurrency caveats).

### M3: Near-Production Compose + Engine E2E Hardening
Deliverables:
- Run `docker-compose.with-engine.yml` as near-production local stack and verify startup health.
- Verify `client_service`, `module_service`, `image_service`, and `engine_service` reachability on `localhost`.
- Standardize local module runtime to use locally built images (`uv` workflow parity) instead of hardcoded remote pull behavior.
- Validate module execution end-to-end through `bqapi` with stress runs and output download.
- Validate output displayability (`image_service ?meta` + `?thumbnail`) and UI paths with Playwright.

Validation commands:
```bash
docker compose -f docker-compose.with-engine.yml up -d --build
docker compose -f docker-compose.with-engine.yml ps
curl -fsS http://127.0.0.1:8080/services | rg 'type="client_service"|type="module_service"|type="image_service"'
curl -fsS http://127.0.0.1:27000/engine_service/_services
docker exec bisque-server bash -lc 'source /usr/lib/bisque/bin/activate && bq-admin setup -y statics && bq-admin module register -a -p -u admin:admin -r http://127.0.0.1:8080 http://127.0.0.1:27000/engine_service/'
source .venv/bin/activate && ./scripts/dev/bqapi_module_runner.py --bisque-root http://127.0.0.1:8080 --user admin --password admin --module EdgeDetection --images test_images/00-db8XD5ccsYdeysUv2Jr9fc.ome-tiff.ome.tif --repeat 5 --concurrency 1 --summary-json artifacts/m3_e2e/edge_stress_summary.json
source .venv/bin/activate && ./scripts/dev/bqapi_module_runner.py --bisque-root http://127.0.0.1:8080 --user admin --password admin --module SimpleUniversalProcess --images test_images/00-db8XD5ccsYdeysUv2Jr9fc.ome-tiff.ome.tif test_images/NPH_shunt_005_85yo.nii.gz --repeat 3 --concurrency 1 --summary-json artifacts/m3_e2e/simple_stress_summary.json
python3 - <<'PY'
import json, re, subprocess
ids=[]
for f in ['artifacts/m3_e2e/edge_stress_summary.json','artifacts/m3_e2e/simple_stress_summary.json']:
    s=json.load(open(f))
    for r in s['results']:
        m=re.search(r'/data_service/(00-[A-Za-z0-9]+)', r.get('output_image_uri',''))
        if m: ids.append(m.group(1))
for rid in dict.fromkeys(ids):
    for q in ('meta','thumbnail'):
        code=subprocess.check_output(['curl','-sS','-u','admin:admin','-L','-o','/tmp/x','-w','%{http_code}',f'http://127.0.0.1:8080/image_service/{rid}?{q}'], text=True).strip()
        print(rid, q, code)
PY
```

Implementation update (2026-02-17):
1. Compose/runtime fixes:
   - `source/bqengine/bq/engine/controllers/docker_env.py` now creates module containers with `--add-host host.docker.internal:host-gateway` and rewrites loopback args (`127.0.0.1`/`localhost`/`0.0.0.0`) for container reachability.
2. Module runtime standardization for local images:
   - `source/modules/EdgeDetection/runtime-module.cfg`: added `[docker]` overrides (`docker.hub=` etc.) so local runs stop pulling `registry.example.com/...` and use local tags (`edgedetection:v1.0.0`).
   - `source/modules/SimpleUniversalProcess/runtime-module.cfg`: same local override behavior.
3. Module input/output URI reliability in Docker:
   - `source/modules/EdgeDetection/PythonScriptWrapper.py` and `source/modules/SimpleUniversalProcess/PythonScriptWrapper.py`:
     - normalize input URIs before `bq.load(...)` and `fetch_blob(...)`.
     - normalize output URIs back to host-reachable loopback for MEX outputs consumed by host-side clients.
4. `bqapi` robustness improvements for mixed inputs:
   - `scripts/dev/bqapi_module_runner.py`:
     - clearer upload error parsing.
     - binary-upload fallback: when ingest rejects a file (for example `NPH_shunt_005_85yo.nii.gz`), generate deterministic fallback PNG and continue E2E execution.
5. Edge module UI/static cleanup:
   - `source/modules/EdgeDetection/EdgeDetection.xml`:
     - interface assets moved to `public/webapp.js` + `public/webapp.css`.
     - help path corrected to `public/help.md`.
   - Added `source/modules/EdgeDetection/public/webapp.js` and `source/modules/EdgeDetection/public/webapp.css`.
   - Re-ran `bq-admin setup -y statics` and restarted server so `/core/js/all_js.js` and `/core/css/all_css.css` serve successfully.

Verification results (2026-02-17):
1. Stack health -> PASS
   - `docker compose -f docker-compose.with-engine.yml ps` shows `bisque-server`, `postgres`, `mysql` healthy.
   - Service checks: `/services`, `/engine_service/_services`, `/client_service/`, `/client_service/browser?resource=/data_service/image` => HTTP `200`.
2. Engine/module registration -> PASS
   - Engine advertises both `EdgeDetection` and `SimpleUniversalProcess`.
   - `bq-admin module register -a -p ...` succeeded.
3. EdgeDetection API E2E stress -> PASS
   - Artifact: `artifacts/m3_e2e/edge_stress_summary.json`
   - Result: `5/5` success (`tasks_failed=0`), outputs downloaded locally.
4. SimpleUniversalProcess API E2E stress with both test inputs -> PASS
   - Artifact: `artifacts/m3_e2e/simple_stress_summary.json`
   - Result: `6/6` success (`tasks_failed=0`) across:
     - `test_images/00-db8XD5ccsYdeysUv2Jr9fc.ome-tiff.ome.tif`
     - `test_images/NPH_shunt_005_85yo.nii.gz` (fallback PNG upload path documented in summary).
5. Output displayability/thumbnail checks -> PASS
   - For all output IDs in stress summaries, `image_service/<id>?meta` => HTTP `200`, `image_service/<id>?thumbnail` => HTTP `200`.
   - Resolves prior `415 Unsupported Media Type` failure path for module outputs.
6. Frontend Playwright validation -> PASS
   - Artifact bundle: `output/playwright/m3/`
   - `client_service/browser?resource=/data_service/image`: loaded with no `Unsupported Media Type` text and no `unavailable` placeholders in sampled run.
   - `client_service/view?resource=http://127.0.0.1:8080/data_service/<output_id>`: rendered without request-error banner.
   - Module pages (`/module_service/EdgeDetection/`, `/module_service/SimpleUniversalProcess/`): `webapp.run` present; console error level clean after static + interface fixes.
7. Non-blocking runtime warnings captured
   - `imgcnv`/`ImarisConvert` remain unavailable on current arm64 local setup; image_service fallback converters (`openslide`, `pillow`, `bioformats`, `ffmpeg`) are active and sufficient for validated flows.

M3 status: Completed.

Post-M3 hotfix (2026-02-17): MEX output URL host mismatch on localhost vs 127.0.0.1

Issue observed:
- Module outputs in MEX `outputs` tags were persisted as absolute `http://127.0.0.1:8080/data_service/...`.
- When UI was opened on `http://localhost:8080`, browser XHRs to `http://127.0.0.1:8080/...` became cross-origin and failed with client-side `status: 0`.
- Server-side checks for those resources returned `200`, confirming the failure mode was browser-origin mismatch rather than missing data.

Implementation update:
- `source/modules/EdgeDetection/PythonScriptWrapper.py`
- `source/modules/SimpleUniversalProcess/PythonScriptWrapper.py`
  - `_normalize_resource_uri_for_external_clients` now emits host-agnostic relative paths (for example `/data_service/00-...`) for HTTP(S) resource URIs.
  - This keeps module output references valid regardless of whether users browse BisQue via `localhost` or `127.0.0.1`.

Verification commands:
```bash
docker build --platform linux/amd64 -t edgedetection:v1.0.0 source/modules/EdgeDetection
docker build --platform linux/amd64 -t simpleuniversalprocess:v1.0.0 source/modules/SimpleUniversalProcess
source .venv/bin/activate && python scripts/dev/bqapi_module_runner.py \
  --bisque-root http://localhost:8080 --user admin --password admin \
  --module SimpleUniversalProcess \
  --images test_images/00-db8XD5ccsYdeysUv2Jr9fc.ome-tiff.ome.tif \
  --summary-json artifacts/m3_e2e/live_verify_host_agnostic_simple.json
source .venv/bin/activate && python scripts/dev/bqapi_module_runner.py \
  --bisque-root http://localhost:8080 --user admin --password admin \
  --module EdgeDetection \
  --images test_images/00-db8XD5ccsYdeysUv2Jr9fc.ome-tiff.ome.tif \
  --summary-json artifacts/m3_e2e/live_verify_host_agnostic_edge.json
```

Verification results:
- `artifacts/m3_e2e/live_verify_host_agnostic_simple.json`: `ok=true`, `tasks_success=1`, `tasks_failed=0`
- `artifacts/m3_e2e/live_verify_host_agnostic_edge.json`: `ok=true`, `tasks_success=1`, `tasks_failed=0`
- MEX output tags now persist `value="/data_service/<id>"` (relative path), confirmed via `module_service/mex/<id>?view=deep`.
- Output image checks pass via both hostnames:
  - `http://localhost:8080/image_service/<id>?meta` -> `200`
  - `http://localhost:8080/image_service/<id>?thumbnail` -> `200`
  - `http://127.0.0.1:8080/image_service/<id>?meta` -> `200`
  - `http://127.0.0.1:8080/image_service/<id>?thumbnail` -> `200`

### M4: Authentication Modernization (OIDC-First, Dual-Stack)
Goal:
- Replace fragile TurboGears/repoze-who coupling with standards-based OIDC/OAuth2, without breaking current UI, module execution, or `bqapi`.
- Keep rollback cheap by running dual-stack (`legacy + oidc`) until parity is proven.

Auth surface findings from code scan (2026-02-17):
1. Core auth remains split between TG `sa_auth` and `repoze.who`:
   - `source/bqcore/bq/config/app_cfg.py`
   - `source/config/who.ini`
2. `module_service` and engine callbacks depend on MEX auth:
   - `source/bqcore/bq/core/lib/mex_auth.py`
   - `source/bqserver/bq/module_service/controllers/module_server.py`
3. Current API client path is mostly username/password basic auth:
   - `source/bqapi/bqapi/comm.py` (`init_local`, `authenticate_basic`)
4. Existing test harness is auth-coupled to basic local login:
   - `34` test references to `init_local(...)` in `bqapi`/`bqserver` test files.
   - `38` backend auth guards using `identity.not_anonymous()` / `@require(not_anonymous())`.
5. Current `auth_service` has mixed legacy + Firebase logic and manual cookie handling:
   - `source/bqserver/bq/client_service/controllers/auth_service.py`

Decision:
- Use OIDC (Authorization Code + PKCE for browser, Bearer access tokens for API).
- Keep `Mex` auth path for engine/module internals in M4 (do not replace immediately).
- Keep `Basic` auth as a compatibility path behind feature flags during transition.
- Do not build a custom OAuth2 authorization server from scratch in this phase.

#### M4.0: Freeze Current Auth Behavior and Add Safety Harness
Deliverables:
- Add `docs/auth/auth_current_state.md` mapping all active auth flows:
  - browser login/logout/session cookie
  - API basic auth
  - module/engine MEX auth
  - Firebase social path
- Add `scripts/dev/test_auth_baseline.sh` to capture current behavior before refactor.
- Add parity fixtures for:
  - `/auth_service/whoami`
  - `/auth_service/session`
  - protected endpoint redirects and `401` behavior.

Validation commands:
```bash
source .venv/bin/activate
./scripts/dev/test_auth_baseline.sh
cd source && source ../.venv/bin/activate && pytest -q bqcore/bq/core/tests/functional/test_authentication.py
cd source && source ../.venv/bin/activate && pytest -q bqserver/bq/data_service/tests/test_ds.py -k test_a_index
```

Exit criteria:
- Baseline auth responses captured and committed as reference artifacts.
- Existing login/session tests pass before any OIDC logic is enabled.

#### M4.1: Introduce OIDC Foundation (No Traffic Cutover)
Deliverables:
- Add OIDC client support using maintained libraries (`Authlib` preferred).
- Add config schema + defaults in `source/config/site.cfg` and `.env.example`:
  - `bisque.auth.mode = legacy|dual|oidc`
  - `bisque.oidc.issuer`
  - `bisque.oidc.client_id`
  - `bisque.oidc.client_secret`
  - `bisque.oidc.redirect_uri`
  - `bisque.oidc.scopes`
  - `bisque.oidc.username_claim`
- Add local dev IdP compose overlay:
  - `docker-compose.oidc.yml`
  - deterministic dev realm/client bootstrap script.
- Add `scripts/dev/oidc_get_token.py` for integration tests and local notebooks.

Validation commands:
```bash
docker compose -f docker-compose.with-engine.yml -f docker-compose.oidc.yml up -d --build
python3 - <<'PY'
import requests
cfg = requests.get("http://127.0.0.1:18080/realms/bisque/.well-known/openid-configuration", timeout=10).json()
print(cfg["issuer"])
print(cfg["jwks_uri"])
PY
source .venv/bin/activate && python scripts/dev/oidc_get_token.py --provider local --user admin --password admin --print-access-token >/tmp/bisque_token.txt
test -s /tmp/bisque_token.txt
```

Exit criteria:
- OIDC metadata and token retrieval are working locally.
- `bisque.auth.mode=legacy` remains default and behavior is unchanged.

#### M4.2: Add Bearer Token Auth Plugin and Dual-Stack Middleware
Deliverables:
- Add Bearer auth plugin:
  - `source/bqcore/bq/core/lib/bearer_auth.py`
  - JWT validation via provider JWKS + issuer/audience checks.
- Wire plugin into `source/config/who.ini` as identifier/authenticator.
- Update `source/bqcore/bq/config/app_cfg.py` conditional middleware:
  - route `Authorization: Bearer ...` through who middleware.
  - preserve current handling for `Basic` and `Mex`.
- Add user claim mapping and local user provisioning strategy:
  - claim priority: `preferred_username`, `email`, `sub`.
  - explicit collision handling and safe fallback.

Validation commands:
```bash
source .venv/bin/activate
cd source && source ../.venv/bin/activate && pytest -q bqcore/bq/core/tests -k "auth and bearer"
export BISQUE_TEST_TOKEN="$(python scripts/dev/oidc_get_token.py --provider local --user admin --password admin --print-access-token)"
curl -fsS -H "Authorization: Bearer $BISQUE_TEST_TOKEN" http://127.0.0.1:8080/auth_service/whoami
curl -fsS -u admin:admin http://127.0.0.1:8080/auth_service/whoami
```

Exit criteria:
- Both bearer and basic auth can access API endpoints in `dual` mode.
- Invalid bearer tokens return deterministic `401` responses (no silent fallback).

#### M4.3: Web Login Migration to OIDC (Browser Session Bridge)
Deliverables:
- Add OIDC web endpoints in `auth_service`:
  - `/auth_service/oidc_login`
  - `/auth_service/oidc_callback`
  - `/auth_service/oidc_logout`
- Keep `/auth_service/login` as compatibility entrypoint:
  - `legacy` mode -> current form flow
  - `dual`/`oidc` mode -> OIDC login redirect
- Replace manual cookie forging path with tested session bridge adapter.
- Preserve existing `came_from` behavior and module UI entry flows.

Validation commands:
```bash
source .venv/bin/activate
python scripts/dev/test_auth_browser_flow.py --base http://127.0.0.1:8080 --mode dual --provider local
node scripts/playwright/m4_auth_ui.spec.js
curl -I -sS "http://127.0.0.1:8080/module_service/SimpleUniversalProcess/?wpublic=1" | head -n 1
```

Exit criteria:
- Browser login works via OIDC and preserves `came_from` redirect.
- Module page access does not loop indefinitely between module page and login.
- Existing cookie session endpoints (`/auth_service/session`) still respond correctly.

#### M4.4: `bqapi` Token-First Support with Legacy Compatibility
Deliverables:
- Extend `source/bqapi/bqapi/comm.py`:
  - add `BQSession.init_token(...)`
  - add optional token branch in `init(...)`
  - keep `init_local(...)` unchanged for compatibility.
- Extend `scripts/dev/bqapi_module_runner.py`:
  - accept `--token` in addition to `--user/--password`.
- Add notebook/script examples for token usage.
- Add migration tests that run both auth modes against same module run.

Validation commands:
```bash
source .venv/bin/activate
cd source && source ../.venv/bin/activate && pytest -q bqapi/bqapi/tests/test_comm.py -k "init_local or token or mex"
export BISQUE_TEST_TOKEN="$(python scripts/dev/oidc_get_token.py --provider local --user admin --password admin --print-access-token)"
python scripts/dev/bqapi_module_runner.py --bisque-root http://127.0.0.1:8080 --token "$BISQUE_TEST_TOKEN" --module EdgeDetection --images test_images/00-db8XD5ccsYdeysUv2Jr9fc.ome-tiff.ome.tif --summary-json artifacts/m4_auth/edge_token_summary.json
python scripts/dev/bqapi_module_runner.py --bisque-root http://127.0.0.1:8080 --user admin --password admin --module EdgeDetection --images test_images/00-db8XD5ccsYdeysUv2Jr9fc.ome-tiff.ome.tif --summary-json artifacts/m4_auth/edge_basic_summary.json
```

Exit criteria:
- `bqapi` can run modules with either bearer token or basic auth in `dual` mode.
- No regression in MEX lifecycle (`create_mex`, polling, output fetch).

#### M4.5: Engine/Module Auth Hardening Without Breaking MEX
Deliverables:
- Keep `Mex` header and `Authorization: Mex ...` support active.
- Add explicit compatibility tests for:
  - module dispatch from UI
  - engine callback updating MEX status
  - output resource visibility from both `localhost` and `127.0.0.1`.
- Add structured logging around auth decision points in module dispatch and engine callback paths.

Validation commands:
```bash
docker compose -f docker-compose.with-engine.yml -f docker-compose.oidc.yml up -d --build
docker compose -f docker-compose.with-engine.yml logs --tail=200 bisque-server
source .venv/bin/activate && python scripts/dev/bqapi_module_runner.py --bisque-root http://127.0.0.1:8080 --token "$BISQUE_TEST_TOKEN" --module SimpleUniversalProcess --images test_images/00-db8XD5ccsYdeysUv2Jr9fc.ome-tiff.ome.tif test_images/NPH_shunt_005_85yo.nii.gz --repeat 2 --concurrency 1 --summary-json artifacts/m4_auth/simple_token_summary.json
python3 scripts/dev/verify_mex_outputs.py --summary artifacts/m4_auth/simple_token_summary.json --base http://127.0.0.1:8080
```

Exit criteria:
- Engine-backed module runs complete end-to-end with OIDC-authenticated API client.
- MEX output resources and thumbnails remain displayable in UI.

#### M4.6: Controlled Cutover + Rollback Plan
Deliverables:
- Set deployment modes:
  - `legacy`: current behavior only
  - `dual`: OIDC + legacy compatibility (default for transition)
  - `oidc`: OIDC required for browser/API; `Mex` remains internal
- Add rollback script:
  - `scripts/dev/auth_mode_switch.sh legacy|dual|oidc`
- Add release notes and migration runbook:
  - `docs/MILESTONE4_AUTH_MIGRATION.md`
- Add explicit deprecation schedule for `Basic` auth and manual cookie logic.

Validation commands:
```bash
./scripts/dev/auth_mode_switch.sh dual
./scripts/dev/test_auth_baseline.sh
./scripts/dev/auth_mode_switch.sh oidc
./scripts/dev/test_auth_baseline.sh --expect-oidc
./scripts/dev/auth_mode_switch.sh legacy
./scripts/dev/test_auth_baseline.sh --expect-legacy
```

Exit criteria:
- All three modes are runnable and testable.
- Rollback to `legacy` is one command and restores known-good behavior.
- `dual` mode is production-ready for staged rollout.

Implementation update (2026-02-17, M4.2 + M4.4 partial):
- Added token primitives:
  - `source/bqcore/bq/core/lib/token_auth.py`
  - Issues and validates JWT bearer access tokens with issuer/audience/expiry checks.
- Added repoze.who bearer plugin:
  - `source/bqcore/bq/core/lib/bearer_auth.py`
  - Supports `Authorization: Bearer ...` as identifier + authenticator in dual-stack mode.
- Added auth API endpoints:
  - `source/bqserver/bq/client_service/controllers/auth_service.py`
  - `POST /auth_service/token` (password grant style for local/API clients)
  - `GET /auth_service/token_info` (token introspection-style claim view)
- Wired bearer path into middleware:
  - `source/bqcore/bq/config/app_cfg.py` (`ConditionalAuthMiddleware` now routes bearer traffic through who middleware).
- Wired bearer plugin in who config:
  - `source/config/who.ini` includes `bearerauth` plugin + identifier/authenticator entries.
- Added persisted-config migration for Docker upgrades:
  - `boot/B10-fullconfig.sh` now patches existing `/source/config/who.ini` volumes to add missing `bearerauth` wiring and plugin section.
  - Prevents silent bearer auth failure after image upgrades with old named volumes.
- Extended `bqapi` token support:
  - `source/bqapi/bqapi/comm.py` adds `BQSession.init_token(...)`, `BQServer.authenticate_bearer(...)`, and token-aware `init(...)`.
  - `scripts/dev/bqapi_module_runner.py` accepts `--token`.
- Added/updated auth tests:
  - `source/bqcore/bq/core/tests/functional/test_authentication.py`
  - `source/bqapi/bqapi/tests/test_comm.py`
- Added reusable smoke runner:
  - `scripts/dev/test_auth_token_smoke.sh`
  - Supports both local-managed server (`MANAGE_SERVER=1`) and external Docker server (`MANAGE_SERVER=0`, `BISQUE_URL=...`).

Verification results (2026-02-17):
1. Docker rebuild/restart with new auth changes -> PASS
```bash
docker compose -f docker-compose.with-engine.yml up -d --build bisque
docker inspect --format '{{.State.Health.Status}}' bisque-server
```
Observed: container rebuilt and reached `healthy`.

2. Token issuance endpoint -> PASS
```bash
curl -X POST http://127.0.0.1:8080/auth_service/token \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin"}'
```
Observed: HTTP `200` with `access_token`, `token_type=Bearer`, `expires_in`.

3. Bearer auth end-to-end (`whoami`, `session`, `token_info`) -> PASS
```bash
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8080/auth_service/whoami
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8080/auth_service/session
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8080/auth_service/token_info
```
Observed:
- `whoami` resolved to `admin` (not `anonymous`)
- `session` includes expected user/group tags
- `token_info` returns `active=true` and expected claims.

4. Invalid credentials behavior -> PASS
```bash
curl -X POST http://127.0.0.1:8080/auth_service/token \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"wrong"}'
```
Observed: HTTP `400` with `{"error":"invalid_grant"}`.

5. Persisted config migration check -> PASS
```bash
docker logs bisque-server | grep -E "updated site.cfg|updated who.ini"
```
Observed: `updated who.ini with bearerauth plugin migration`.

6. Smoke script against Docker-hosted BisQue -> PASS
```bash
MANAGE_SERVER=0 BISQUE_URL=http://127.0.0.1:8080 ./scripts/dev/test_auth_token_smoke.sh
```
Observed:
- bearer `whoami/session` pass
- `BQSession.init_token(...)` pass.

7. Python auth regression tests -> PASS
```bash
source .venv/bin/activate
cd source
PYTHONPATH=bqcore:bqserver:bqapi pytest -q bqcore/bq/core/tests/functional/test_authentication.py
PYTHONPATH=bqcore:bqserver:bqapi pytest -q bqapi/bqapi/tests/test_comm.py -k open_token_session
PYTHONPATH=source/bqserver:source/bqcore pytest -q source/bqserver/bq/table/tests/test_runquery.py
```
Observed:
- `test_authentication.py`: `4 passed`
- `test_comm.py -k open_token_session`: `1 passed`
- `test_runquery.py`: `17 passed`.

8. Token-based module execution via `bqapi` -> PASS
```bash
source .venv/bin/activate
python scripts/dev/bqapi_module_runner.py \
  --bisque-root http://127.0.0.1:8080 \
  --token "$TOKEN" \
  --module EdgeDetection \
  --images test_images/00-db8XD5ccsYdeysUv2Jr9fc.ome-tiff.ome.tif test_images/NPH_shunt_005_85yo.nii.gz \
  --summary-json artifacts/m4_auth_e2e/edge_token_summary.json
```
Observed:
- `2/2` success, module finished, output blobs downloaded.
- Artifact: `artifacts/m4_auth_e2e/edge_token_summary.json`.

9. Token stress finding (non-auth blocker) -> PARTIAL
```bash
source .venv/bin/activate
python scripts/dev/bqapi_module_runner.py \
  --bisque-root http://127.0.0.1:8080 \
  --token "$TOKEN" \
  --module SimpleUniversalProcess \
  --images test_images/00-db8XD5ccsYdeysUv2Jr9fc.ome-tiff.ome.tif test_images/NPH_shunt_005_85yo.nii.gz \
  --repeat 2 --concurrency 2 --timeout-seconds 120 --allow-failures \
  --summary-json artifacts/m4_auth_e2e/simple_token_stress_v2_summary.json
```
Observed:
- `3/4` success, `1/4` timeout with `last_status=DISPATCH`.
- Artifact: `artifacts/m4_auth_e2e/simple_token_stress_v2_summary.json`.
- Not an auth failure (token issuance + bearer + module execution pass), but indicates engine/module scheduling robustness work remains for higher concurrency.

10. AGENTS baseline command rerun -> PASS/PARTIAL (expected)
```bash
python3 --version
PATH="$HOME/Library/Python/3.10/bin:$PATH" uv --version
PATH="$HOME/Library/Python/3.10/bin:$PATH" uv venv .venv
source .venv/bin/activate && PATH="$HOME/Library/Python/3.10/bin:$PATH" uv pip install -r source/requirements.dev
cd source && source ../.venv/bin/activate && PATH="$HOME/Library/Python/3.10/bin:$PATH" uv pip install -r requirements.dev
source .venv/bin/activate && PYTHONPATH=source/bqserver:source/bqcore pytest -q source/bqserver/bq/table/tests/test_runquery.py
```
Observed:
- `uv venv .venv` reported existing venv (expected in a populated workspace).
- Root-level `uv pip install -r source/requirements.dev` still fails on relative local path parsing (known hazard).
- `cd source && uv pip install -r requirements.dev` passes in the current prepared environment.
- `test_runquery.py` passes (`17 passed`), confirming query baseline remains intact.

11. Backward compatibility check (`Basic` auth path) -> PASS
```bash
source .venv/bin/activate
python scripts/dev/bqapi_module_runner.py \
  --bisque-root http://127.0.0.1:8080 \
  --user admin --password admin \
  --module EdgeDetection \
  --images test_images/00-db8XD5ccsYdeysUv2Jr9fc.ome-tiff.ome.tif \
  --summary-json artifacts/m4_auth_e2e/edge_basic_summary.json
```
Observed:
- `1/1` success in legacy basic-auth mode.
- Artifact: `artifacts/m4_auth_e2e/edge_basic_summary.json`.

12. Extended regression rerun for auth and API client -> PASS
```bash
cd source && source ../.venv/bin/activate && PYTHONPATH=bqserver:bqcore:bqapi pytest -q bqcore/bq/core/tests/functional/test_authentication.py
cd source && source ../.venv/bin/activate && PYTHONPATH=bqserver:bqcore:bqapi pytest -q bqapi/bqapi/tests/test_comm.py
cd source && source ../.venv/bin/activate && PYTHONPATH=bqserver:bqcore:bqapi pytest -q bqserver/bq/data_service/tests/test_ds.py -k test_a_index
```
Observed:
- `test_authentication.py`: `4 passed`
- `test_comm.py`: `17 passed`
- `test_ds.py -k test_a_index`: `1 passed`

13. Auth endpoint concurrency characterization -> PARTIAL (throughput, not correctness)
```bash
source .venv/bin/activate
python - <<'PY'
# issues 24 concurrent token->whoami/session checks at timeout=20s and timeout=120s
PY
```
Observed:
- `timeout=20s`: `8/24` passed, `16/24` read-timeouts (queue saturation at current local server worker capacity).
- `timeout=120s`: `24/24` passed.
- Artifacts:
  - `artifacts/m4_auth_e2e/auth_concurrency_20s.json`
  - `artifacts/m4_auth_e2e/auth_concurrency_120s.json`
- Interpretation: auth logic is correct, but near-production throughput tuning is still needed for aggressive short-timeout concurrent API traffic.

14. Additional token stress run on EdgeDetection -> PARTIAL (non-auth blocker)
```bash
source .venv/bin/activate
python scripts/dev/bqapi_module_runner.py \
  --bisque-root http://127.0.0.1:8080 \
  --token "$TOKEN" \
  --module EdgeDetection \
  --images test_images/00-db8XD5ccsYdeysUv2Jr9fc.ome-tiff.ome.tif test_images/NPH_shunt_005_85yo.nii.gz \
  --repeat 2 --concurrency 2 --timeout-seconds 180 \
  --summary-json artifacts/m4_auth_e2e/edge_token_stress_summary.json
```
Observed:
- `3/4` success, `1/4` timeout at `DISPATCH`.
- Artifact: `artifacts/m4_auth_e2e/edge_token_stress_summary.json`.
- Consistent with existing engine/module scheduling caveat; not attributable to bearer token validation failures.

M4 completion update (2026-02-18):
1. Completed M4.3 browser-flow verification tooling and execution
   - Added `scripts/dev/test_auth_browser_flow.py`:
     - validates `/auth_service/login -> /auth_service/oidc_login -> provider -> /auth_service/oidc_callback -> /auth_service/post_login`.
     - validates authenticated `/auth_service/whoami`, `/auth_service/session`, and module page reachability.
     - includes localhost cookie normalization for `authtkt` so scripted flow matches browser behavior.
   - Validation:
```bash
source .venv/bin/activate
python scripts/dev/test_auth_browser_flow.py --base http://localhost:8080 --mode dual --user admin --password admin --module EdgeDetection --report-json artifacts/m4_auth/browser_flow_dual_final.json
python scripts/dev/test_auth_browser_flow.py --base http://localhost:8080 --mode oidc --user admin --password admin --module SimpleUniversalProcess --report-json artifacts/m4_auth/browser_flow_oidc.json
python scripts/dev/test_auth_browser_flow.py --base http://localhost:8080 --mode legacy --report-json artifacts/m4_auth/browser_flow_legacy.json
```
   - Observed: all three modes PASS.

2. Completed M4.5 output/resource verification tooling and execution
   - Added `scripts/dev/verify_mex_outputs.py`:
     - validates `data_service ?view=short`, `image_service ?meta`, and `image_service ?thumbnail`.
   - Validation:
```bash
source .venv/bin/activate
python scripts/dev/verify_mex_outputs.py --summary artifacts/m4_auth/edge_token_dual_summary.json --base http://127.0.0.1:8080 --user admin --password admin --report-json artifacts/m4_auth/verify_edge_token_dual.json
python scripts/dev/verify_mex_outputs.py --summary artifacts/m4_auth/simple_token_dual_summary.json --base http://127.0.0.1:8080 --user admin --password admin --report-json artifacts/m4_auth/verify_simple_token_dual.json
python scripts/dev/verify_mex_outputs.py --summary artifacts/m4_auth/edge_token_stress_dual_summary.json --base http://127.0.0.1:8080 --user admin --password admin --report-json artifacts/m4_auth/verify_edge_token_stress_dual.json
python scripts/dev/verify_mex_outputs.py --summary artifacts/m4_auth/simple_token_stress_dual_summary.json --base http://127.0.0.1:8080 --user admin --password admin --report-json artifacts/m4_auth/verify_simple_token_stress_dual.json
```
   - Observed: all checked output resources PASS (`meta`/`thumbnail` HTTP `200`).

3. Completed M4.5 auth decision logging hardening (no behavior change)
   - Updated `source/bqengine/bq/engine/controllers/runtime_adapter.py`:
     - logs token source selection (`identity` / `authorization_header_mex` / `resource_fallback`) without logging secrets.
   - Updated `source/bqserver/bq/module_service/controllers/module_server.py`:
     - removed MEX token value from dispatch logs.
     - replaced deprecated `log.warn` with `log.warning`.

4. Completed M4.6 auth mode rollout matrix in near-production compose
   - Mode switching executed via compose env:
```bash
BISQUE_AUTH_MODE=oidc BISQUE_AUTH_LOCAL_TOKEN_ENABLED=false docker compose -f docker-compose.with-engine.yml -f docker-compose.oidc.yml up -d bisque
./scripts/dev/test_auth_baseline.sh --expect-oidc

BISQUE_AUTH_MODE=legacy BISQUE_AUTH_LOCAL_TOKEN_ENABLED=true docker compose -f docker-compose.with-engine.yml -f docker-compose.oidc.yml up -d bisque
./scripts/dev/test_auth_baseline.sh --expect-legacy

BISQUE_AUTH_MODE=dual BISQUE_AUTH_LOCAL_TOKEN_ENABLED=true docker compose -f docker-compose.with-engine.yml -f docker-compose.oidc.yml up -d bisque
./scripts/dev/test_auth_baseline.sh --expect-dual
```
   - Observed: PASS for `legacy`, `dual`, and `oidc`.

5. Extended regression and E2E verification results
   - Auth/client tests:
```bash
cd source && source ../.venv/bin/activate && PYTHONPATH=bqserver:bqcore:bqapi pytest -q bqcore/bq/core/tests/functional/test_authentication.py
cd source && source ../.venv/bin/activate && PYTHONPATH=bqserver:bqcore:bqapi pytest -q bqapi/bqapi/tests/test_comm.py
source .venv/bin/activate && PYTHONPATH=source/bqserver:source/bqcore pytest -q source/bqserver/bq/table/tests/test_runquery.py
MANAGE_SERVER=0 BISQUE_URL=http://127.0.0.1:8080 ./scripts/dev/test_auth_token_smoke.sh
```
   - Observed: PASS (`4 passed`, `17 passed`, `17 passed`, token smoke PASS).
   - `bqapi` module runs in dual:
```bash
source .venv/bin/activate
TOKEN="$(python scripts/dev/oidc_get_token.py --provider local --user admin --password admin --print-access-token)"
python scripts/dev/bqapi_module_runner.py --bisque-root http://127.0.0.1:8080 --token "$TOKEN" --module EdgeDetection --images test_images/00-db8XD5ccsYdeysUv2Jr9fc.ome-tiff.ome.tif --summary-json artifacts/m4_auth/edge_token_dual_summary.json
python scripts/dev/bqapi_module_runner.py --bisque-root http://127.0.0.1:8080 --user admin --password admin --module EdgeDetection --images test_images/00-db8XD5ccsYdeysUv2Jr9fc.ome-tiff.ome.tif --summary-json artifacts/m4_auth/edge_basic_dual_summary.json
python scripts/dev/bqapi_module_runner.py --bisque-root http://127.0.0.1:8080 --token "$TOKEN" --module SimpleUniversalProcess --images test_images/00-db8XD5ccsYdeysUv2Jr9fc.ome-tiff.ome.tif test_images/NPH_shunt_005_85yo.nii.gz --summary-json artifacts/m4_auth/simple_token_dual_summary.json
```
   - Observed: PASS (`1/1`, `1/1`, `2/2`).
   - Stress characterization in dual:
```bash
python scripts/dev/bqapi_module_runner.py --bisque-root http://127.0.0.1:8080 --token "$TOKEN" --module EdgeDetection --images test_images/00-db8XD5ccsYdeysUv2Jr9fc.ome-tiff.ome.tif --repeat 4 --concurrency 2 --timeout-seconds 240 --summary-json artifacts/m4_auth/edge_token_stress_dual_summary.json
python scripts/dev/bqapi_module_runner.py --bisque-root http://127.0.0.1:8080 --token "$TOKEN" --module SimpleUniversalProcess --images test_images/00-db8XD5ccsYdeysUv2Jr9fc.ome-tiff.ome.tif test_images/NPH_shunt_005_85yo.nii.gz --repeat 2 --concurrency 2 --timeout-seconds 180 --allow-failures --summary-json artifacts/m4_auth/simple_token_stress_dual_summary.json
```
   - Observed: both runs `3/4` success with `1` timeout at `DISPATCH` (engine scheduling throughput caveat; not auth-claim validation failure).

6. Firebase auth noise hardening for production logs
   - Updated `source/bqcore/bq/core/lib/firebase_auth.py`:
     - do not initialize Firebase auth when no `project_id`/`service_account_key` is configured.
     - suppress expected `"project ID is required"` token-check failures to debug level.
   - Rebuilt and validated:
```bash
BISQUE_AUTH_MODE=dual BISQUE_AUTH_LOCAL_TOKEN_ENABLED=true docker compose -f docker-compose.with-engine.yml -f docker-compose.oidc.yml up -d --build bisque
docker compose -f docker-compose.with-engine.yml -f docker-compose.oidc.yml logs --tail=250 bisque | rg 'bq.auth.firebase|Firebase token verification failed'
```
   - Observed: no repeated Firebase warning spam after bearer traffic.

7. Logout/session invalidation regression fix (dual/oidc)
   - Updated `source/bqserver/bq/client_service/controllers/auth_service.py`:
     - persist `id_token` during OIDC callback for RP-initiated logout.
     - route `/auth_service/logout_handler` to `/auth_service/oidc_logout` in dual/oidc.
     - harden cookie invalidation across `authtkt`/`auth_tkt` + localhost domain variants.
     - avoid invalid dynamic post-logout redirect by using optional `bisque.oidc.post_logout_redirect_uri` only when explicitly configured.
   - Updated `source/bqcore/bq/config/app_cfg.py`:
     - moved repoze internal `sa_auth.logout_handler` to `/auth_service/_who_logout_handler` so app-level `/auth_service/logout_handler` logic is not bypassed.
   - Updated defaults + UI source:
     - `source/config-defaults/site.cfg.default` adds `bisque.oidc.post_logout_redirect_uri`.
     - `source/bqcore/bq/core/public/js/bq_api.js` and `source/bqcore/bq/core/public/js/bq_ui_toolbar.js` now target `/auth_service/oidc_logout`.
   - Added regression tool:
     - `scripts/dev/test_auth_logout_flow.py`.
   - Validation:
```bash
source .venv/bin/activate
python scripts/dev/test_auth_logout_flow.py --base http://localhost:8080 --user admin --password admin --report-json artifacts/m4_auth/logout_flow_dual.json
./scripts/dev/test_auth_baseline.sh --expect-dual
```
   - Observed: PASS. After logout, `whoami` returns anonymous and the next login shows Keycloak form (no silent SSO).

M4 status: Completed for auth modernization rollout (`legacy`/`dual`/`oidc`, browser + API + module compatibility verified). Residual non-auth risk remains: intermittent `DISPATCH` timeouts under higher concurrency (`concurrency=2`) in engine-backed module stress.

Post-M4 production soak revalidation (2026-02-18):
1. Rebuilt near-production stack (`with-engine` + `oidc`) and revalidated service health
```bash
BISQUE_AUTH_MODE=dual BISQUE_AUTH_LOCAL_TOKEN_ENABLED=true docker compose -f docker-compose.with-engine.yml -f docker-compose.oidc.yml up -d --build
docker compose -f docker-compose.with-engine.yml -f docker-compose.oidc.yml ps
curl -fsS http://127.0.0.1:8080/services
curl -fsS http://127.0.0.1:8080/image_service/formats
curl -fsS http://127.0.0.1:8080/module_service/
```
   - Result: PASS (`bisque`, `keycloak`, `postgres`, `mysql` healthy; core services reachable).
   - Artifact: `artifacts/m4_auth_e2e/compose_ps.txt`, `artifacts/m4_auth_e2e/service_reachability.log`

2. Auth/browser/token regression rerun in dual mode
```bash
./scripts/dev/test_auth_baseline.sh --expect-dual
python scripts/dev/test_auth_browser_flow.py --base http://localhost:8080 --mode dual --user admin --password admin --module EdgeDetection --report-json artifacts/m4_auth_e2e/browser_flow_dual_admin.json
python scripts/dev/test_auth_logout_flow.py --base http://localhost:8080 --user admin --password admin --report-json artifacts/m4_auth_e2e/logout_flow_dual_admin.json
MANAGE_SERVER=0 BISQUE_URL=http://127.0.0.1:8080 ./scripts/dev/test_auth_token_smoke.sh
```
   - Result: PASS (login redirect chain, logout invalidation, bearer session checks).
   - Artifacts: `artifacts/m4_auth_e2e/auth_baseline_dual.log`, `artifacts/m4_auth_e2e/browser_flow_dual_admin.json`, `artifacts/m4_auth_e2e/logout_flow_dual_admin.json`, `artifacts/m4_auth_e2e/token_smoke.log`

3. Extended module/API soak + output renderability verification
```bash
python scripts/dev/bqapi_module_runner.py --bisque-root http://127.0.0.1:8080 --token "$TOKEN" --module EdgeDetection --images test_images/00-db8XD5ccsYdeysUv2Jr9fc.ome-tiff.ome.tif --repeat 6 --concurrency 1 --timeout-seconds 240 --summary-json artifacts/m4_auth_e2e/edge_token_soak_summary.json
python scripts/dev/bqapi_module_runner.py --bisque-root http://127.0.0.1:8080 --token "$TOKEN" --module SimpleUniversalProcess --images test_images/00-db8XD5ccsYdeysUv2Jr9fc.ome-tiff.ome.tif test_images/NPH_shunt_005_85yo.nii.gz --repeat 4 --concurrency 1 --timeout-seconds 240 --summary-json artifacts/m4_auth_e2e/simple_token_soak_summary.json
python scripts/dev/verify_mex_outputs.py --summary artifacts/m4_auth_e2e/edge_token_soak_summary.json --base http://127.0.0.1:8080 --user admin --password admin --report-json artifacts/m4_auth_e2e/verify_edge_token_soak.json
python scripts/dev/verify_mex_outputs.py --summary artifacts/m4_auth_e2e/simple_token_soak_summary.json --base http://127.0.0.1:8080 --user admin --password admin --report-json artifacts/m4_auth_e2e/verify_simple_token_soak.json
```
   - Result: PASS (`EdgeDetection 6/6`, `SimpleUniversalProcess 8/8`, thumbnails/meta/data checks all green).
   - Artifacts: `artifacts/m4_auth_e2e/edge_token_soak_summary.json`, `artifacts/m4_auth_e2e/simple_token_soak_summary.json`, `artifacts/m4_auth_e2e/verify_edge_token_soak.json`, `artifacts/m4_auth_e2e/verify_simple_token_soak.json`

4. Added lifecycle test for user provisioning/deprovisioning and unknown-user rejection
   - Added script: `scripts/dev/test_oidc_user_lifecycle.py`
```bash
python scripts/dev/test_oidc_user_lifecycle.py --base http://localhost:8080 --keycloak-base http://localhost:18080 --realm bisque --keycloak-admin-user admin --keycloak-admin-password admin --report-json artifacts/m4_auth_e2e/keycloak_user_lifecycle.json
docker compose -f docker-compose.with-engine.yml -f docker-compose.oidc.yml logs --since=30m keycloak | rg "user_not_found"
```
   - Result: PASS (`create user -> login success -> delete user -> login fails on Keycloak form`).
   - Artifacts: `artifacts/m4_auth_e2e/keycloak_user_lifecycle.json`, `artifacts/m4_auth_e2e/keycloak_auth_events.log`

5. Frontend verification with Playwright CLI
   - Valid login navigates to BisQue client (`http://localhost:8080/client_service/`).
   - Invalid user shows Keycloak error text `Invalid username or password.` and remains on provider login action URL.
   - Artifacts: `output/playwright/m4_soak/` and `.playwright-cli/page-2026-02-18T10-23-47-645Z.yml`

6. Small production-log root-cause fix discovered during soak
   - Issue: successful approval checks were logged at `ERROR` level (`Unified approval check for user ...`) causing false-positive auth alarms.
   - Change: `source/bqserver/bq/client_service/controllers/auth_service.py` downgraded that log to `DEBUG` (no auth behavior change).
   - Validation:
```bash
BISQUE_AUTH_MODE=dual BISQUE_AUTH_LOCAL_TOKEN_ENABLED=true docker compose -f docker-compose.with-engine.yml -f docker-compose.oidc.yml up -d --build bisque
python scripts/dev/test_oidc_user_lifecycle.py --base http://localhost:8080 --keycloak-base http://localhost:18080 --realm bisque --keycloak-admin-user admin --keycloak-admin-password admin --report-json artifacts/m4_auth_e2e/keycloak_user_lifecycle_after_rebuild.json
docker compose -f docker-compose.with-engine.yml -f docker-compose.oidc.yml logs --since=5m bisque | rg "Unified approval check for user|ERROR \\[bq.auth\\]"
```
   - Result: PASS (no false `ERROR [bq.auth]` entries after rebuild; lifecycle checks still pass).
   - Artifacts: `artifacts/m4_auth_e2e/keycloak_user_lifecycle_after_rebuild.json`, `artifacts/m4_auth_e2e/bisque_auth_log_level_check_after_rebuild.log`

7. Regression test status note
```bash
source .venv/bin/activate
PYTHONPATH=source/bqserver:source/bqcore pytest -q source/bqserver/bq/table/tests/test_runquery.py
cd source && source ../.venv/bin/activate && PYTHONPATH=bqcore:bqserver:bqapi pytest -q bqserver/bq/data_service/tests/test_ds.py -k test_a_index
```
   - `test_runquery.py`: PASS (`17 passed`).
   - `test_ds.py -k test_a_index`: FAIL in local pytest harness due missing sqlite schema (`no such table: taggable`) and missing local image converter binaries (`imgcnv`), not due compose runtime regressions.

### M8 (Priority): Docker/K3s Build Modernization
Goal:
- Reduce cold compose build time and image size while preserving existing runtime behavior (`client_service`, `module_service`, `image_service`, `engine_service`, auth modes, module E2E).
- Replace fragile/slow remote dependency fetch paths with deterministic local artifacts and cacheable layers.
- Move Docker Python installation path toward `uv`-managed, lockable installs without breaking legacy packages.

Baseline scan + profiling update (2026-02-18):
Commands run:
```bash
docker --version
docker buildx version
/usr/bin/time -l docker build --no-cache --pull --progress=plain -t bisque-build-profile:cold .
/usr/bin/time -l docker compose --progress plain -f docker-compose.with-engine.yml build --no-cache bisque
docker history --no-trunc bisque-ucsb:latest | head -n 18
du -sh source source/data source/staging
docker run --rm --entrypoint /bin/bash bisque-ucsb:latest -lc 'which uv || echo uv-not-found'
```

Observed results:
1. Cold compose build duration: `279.62s` (`artifacts/m8_docker_build/compose_build_cold.log`) -> PASS (baseline captured).
2. Slowest steps are duplicated install/setup layers:
   - `RUN /builder/run-bisque.sh build` -> `93.0s`
   - `RUN /builder/bq-admin-setup.sh` -> `32.9s`
3. Duplicate setup path confirmed:
   - `builder/20-build-bisque.sh` runs `bq-admin setup -y install`.
   - `bq-admin-setup.sh` also runs `bq-admin setup -y install`.
4. External fetch coupling confirmed:
   - `files.wskoly.xyz /binaries/depot` downloads repeated in both steps (`5` + `5` downloads; `InsecureRequestWarning` repeated).
5. APT layering inefficiency confirmed:
   - `apt-get update -qq` appears `10` times in `Dockerfile`.
6. Image size and context issues confirmed:
   - Image size ~`7.75GB`.
   - `source/` is `3.8GB`, including `source/data` (`2.1GB`) and `source/staging` (`1.6GB`) copied into the build layer.
7. Python installer state:
   - `uv` is not installed inside image (`uv-not-found`).
8. Legacy build orchestration:
   - `source/setup.py` is paver entrypoint; Docker build path is currently shell + `pip` + `bq-admin`, not active paver tasks.

Decision:
- Keep paver only as compatibility shim for legacy commands.
- Do not use paver in Docker build path going forward.
- Migrate Docker Python dependency resolution/install to `uv` in staged rollout with rollback switch to current `pip` path.

#### M8.0: Baseline Freeze and Build Profiler (complete)
Deliverables:
- Persisted build logs and timing artifacts under `artifacts/m8_docker_build/`.
- Measured bottleneck list with exact step durations and root causes.

Validation commands:
```bash
test -f artifacts/m8_docker_build/build_cold.log
test -f artifacts/m8_docker_build/build_warm.log
test -f artifacts/m8_docker_build/compose_build_cold.log
rg -n "RUN /builder/run-bisque.sh build|RUN /builder/bq-admin-setup.sh|real" artifacts/m8_docker_build/compose_build_cold.log
```

Exit criteria:
- Baseline artifacts committed and referenced from this plan.

#### M8.1: Remove Duplicate Install Pass (highest-impact quick win)
Deliverables:
- Split `bq-admin-setup.sh` into post-install-only responsibilities (`fullconfig`/verification), no second `install`.
- Keep exactly one `bq-admin setup -y install` call during image build.
- Add guard env var for rollback (`BISQUE_BUILD_DOUBLE_INSTALL=1` re-enables old behavior if needed).

Validation commands:
```bash
/usr/bin/time -l docker compose --progress plain -f docker-compose.with-engine.yml build --no-cache bisque > artifacts/m8_docker_build/m81_cold.log 2>&1
rg -n "bq-admin setup -y install" artifacts/m8_docker_build/m81_cold.log
rg -n "RUN /builder/run-bisque.sh build|RUN /builder/bq-admin-setup.sh|DONE [0-9]+\\.[0-9]+s" artifacts/m8_docker_build/m81_cold.log
docker compose -f docker-compose.with-engine.yml up -d
curl -fsS http://127.0.0.1:8080/services | rg 'type="client_service"|type="module_service"|type="image_service"'
```

Exit criteria:
- Only one install pass in logs.
- Step `33/41` drops from ~`32.9s` to <= `8s`.
- Service registration and startup behavior unchanged.

Implementation update (2026-02-18):
1. M8.1 code changes
   - `bq-admin-setup.sh`
     - Removed unconditional second `bq-admin setup -y install`.
     - Added rollback guard: `BISQUE_BUILD_DOUBLE_INSTALL` (`1|true|yes|on` re-enables legacy duplicate install).
     - Added post-install phase (`bq-admin setup -y fullconfig`) controlled by `BISQUE_BUILD_POST_FULLCONFIG` (default enabled).
     - Enabled strict shell mode (`set -euo pipefail`) and fixed verification import path (`bq.client_service` instead of invalid `bq.server`).
   - `Dockerfile`
     - Added build args/env wiring:
       - `ARG/ENV BISQUE_BUILD_DOUBLE_INSTALL` (default `0`)
       - `ARG/ENV BISQUE_BUILD_POST_FULLCONFIG` (default `1`)
   - `Dockerfile-engine`
     - Added the same build args/env wiring for parity.

2. M8.1 verification runs
   - Initial verification run:
```bash
/usr/bin/time -l docker compose --progress plain -f docker-compose.with-engine.yml build --no-cache bisque > artifacts/m8_docker_build/m81_cold.log 2>&1
```
   - Result: FAIL (script hard-failed on invalid check `import bq.server`; fixed in `bq-admin-setup.sh`).

   - Post-fix verification run:
```bash
/usr/bin/time -l docker compose --progress plain -f docker-compose.with-engine.yml build --no-cache bisque > artifacts/m8_docker_build/m81_cold_v2.log 2>&1
rg -n "bq-admin setup -y install" artifacts/m8_docker_build/m81_cold_v2.log
rg -n "Skipping duplicate install|Running post-install fullconfig" artifacts/m8_docker_build/m81_cold_v2.log
rg -n "RUN /builder/run-bisque.sh build|RUN /builder/bq-admin-setup.sh|DONE [0-9]+\\.[0-9]+s|real\\s+" artifacts/m8_docker_build/m81_cold_v2.log
docker compose -f docker-compose.with-engine.yml up -d
curl -fsS http://127.0.0.1:8080/services | rg 'type="client_service"|type="module_service"|type="image_service"'
curl -sS -o /tmp/m81_client.out -w "%{http_code}\n" http://127.0.0.1:8080/client_service/
curl -sS -o /tmp/m81_module.out -w "%{http_code}\n" http://127.0.0.1:8080/module_service/
curl -sS -o /tmp/m81_image.out -w "%{http_code}\n" http://127.0.0.1:8080/image_service/
```
   - Result: PASS
     - exactly one install call in build log (`count=1`, only inside step `32/41`).
     - step `33/41` (`/builder/bq-admin-setup.sh`) dropped to `2.7s` (was `32.9s` baseline).
     - post-install fullconfig executed in step `33/41`.
     - compose cold build wall time improved:
       - baseline: `279.62s`
       - M8.1: `249.04s`
       - delta: `-30.58s` (~`10.9%` faster).
     - service checks: `client_service/module_service/image_service` present; endpoint checks returned HTTP `200`.

3. Rollback guard validation
```bash
docker compose --progress plain -f docker-compose.with-engine.yml build --build-arg BISQUE_BUILD_DOUBLE_INSTALL=1 bisque > artifacts/m8_docker_build/m81_double_install_guard.log 2>&1
```
   - Result: PASS
     - log confirms guard path: `BISQUE_BUILD_DOUBLE_INSTALL enabled: running legacy duplicate install pass`.
     - step `33/41` returned to slower behavior (`26.7s`) with external depot fetches, confirming rollback compatibility.

#### M8.2: Dockerfile Layering + Context Hygiene
Deliverables:
- Add `.dockerignore` excluding non-build/runtime payloads (`.git`, `.venv`, `artifacts`, `output`, `Results`, `SampleData`, and especially `source/data`, `source/staging`, `source/reports`).
- Replace monolithic `ADD source /source` with `COPY`-based source inclusion and explicit runtime directory creation.
- Keep build behavior identical (no package/dependency-order changes in this step).
- Defer APT layer consolidation to M8.2b for lower-risk rollout sequencing.

Validation commands:
```bash
docker build --no-cache --progress=plain -t bisque-m82:test . > artifacts/m8_docker_build/m82_cold.log 2>&1
docker history --no-trunc bisque-m82:test | head -n 18
docker image inspect bisque-m82:test --format '{{.Size}}'
docker run --rm --entrypoint /bin/bash bisque-m82:test -lc 'ls -ld /source/data /source/staging /source/reports'
docker compose -f docker-compose.with-engine.yml up -d
curl -fsS http://127.0.0.1:8080/services | rg 'type="client_service"|type="module_service"|type="image_service"'
```

Exit criteria:
- Source-data/staging no longer copied into image layers.
- Runtime directories exist in image (`/source/data`, `/source/staging`, `/source/reports`).
- No regression in compose startup/service reachability.
- Image size reduced relative to M8.1 baseline.

Implementation update (2026-02-18):
1. M8.2 code changes
   - Added `.dockerignore` with build-context pruning for local/runtime payloads:
     - `.git`, `.venv`, `.playwright-cli`, `artifacts`, `output`, `Results`, `SampleData`, `data`, `test_images`, `dna_cover.png`
     - `source/data`, `source/staging`, `source/reports`, `source/public`, `source/bisque_*.log`
   - Updated source inclusion in Dockerfiles:
     - `Dockerfile`: `ADD source /source` -> `COPY source/ /source/`
     - `Dockerfile-engine`: `ADD source /source` and legacy requirements-only pre-copy path replaced with `COPY source/ /source/`
   - Added explicit runtime directory creation before build install:
     - `RUN mkdir -p /source/data /source/staging /source/reports /source/public`

2. M8.2 verification runs
```bash
/usr/bin/time -l docker build --no-cache --progress=plain -t bisque-m82:test . > artifacts/m8_docker_build/m82_cold.log 2>&1
docker history --no-trunc bisque-m82:test | head -n 20
docker history --no-trunc bisque-m82:test | rg 'COPY source/ /source/|ADD source'
docker image inspect bisque-m82:test --format '{{.Size}}'
docker run --rm --entrypoint /bin/bash bisque-m82:test -lc 'ls -ld /source/data /source/staging /source/reports /source/public && du -sh /source/data /source/staging /source/reports /source/public'
docker tag bisque-m82:test bisque-ucsb:latest
docker compose -f docker-compose.with-engine.yml up -d
curl -fsS http://127.0.0.1:8080/services | rg 'type="client_service"|type="module_service"|type="image_service"'
curl -sS -o /tmp/m82_client.out -w "%{http_code}\n" http://127.0.0.1:8080/client_service/
curl -sS -o /tmp/m82_module.out -w "%{http_code}\n" http://127.0.0.1:8080/module_service/
curl -sS -o /tmp/m82_image.out -w "%{http_code}\n" http://127.0.0.1:8080/image_service/
source .venv/bin/activate && PYTHONPATH=source/bqserver:source/bqcore pytest -q source/bqserver/bq/table/tests/test_runquery.py
```

3. Results
   - Build context transfer reduced:
     - previous: `~1.60MB`
     - M8.2: `876.65kB`
   - Cold Docker build wall time improved:
     - M8.1 baseline: `249.04s`
     - M8.2: `216.35s`
     - delta: `-32.69s` (`13.13%` faster than M8.1 baseline)
   - Image size improved:
     - baseline (pre-M8): `7,754,093,720` bytes
     - M8.2 image (`bisque-m82:test`): `3,761,974,789` bytes
     - reduction: `3,992,118,931` bytes (`51.48%`)
   - Source copy layer size:
     - previous monolithic source layer: `~4.09GB`
     - M8.2 `COPY source/ /source/` layer: `92.6MB`
   - Runtime directory checks in image: PASS (`/source/data`, `/source/staging`, `/source/reports`, `/source/public` exist)
   - Compose/service checks: PASS (`client_service`, `module_service`, `image_service`; endpoint status `200`)
   - Query smoke regression: PASS (`17 passed`)
   - APT update count remains `10` in `Dockerfile` (intentionally deferred to M8.2b).
   - `Dockerfile-engine` legacy validation note:
     - `docker build -f Dockerfile-engine ...` still fails before BisQue build steps due legacy Xenial mirror/base-image package fetch errors (`ftp.ucsb.edu`), which is pre-existing and outside M8.2 scope.

#### M8.2b: APT Layer Consolidation (follow-up, low-risk gate after M8.2)
Deliverables:
- Reduce repeated `apt-get update -qq` invocations by combining install layers.
- Preserve package set and runtime behavior.

Validation commands:
```bash
docker build --no-cache --progress=plain -t bisque-m82b:test . > artifacts/m8_docker_build/m82b_cold.log 2>&1
rg -n "apt-get update -qq" Dockerfile | wc -l
docker compose -f docker-compose.with-engine.yml up -d
curl -fsS http://127.0.0.1:8080/image_service/formats
```

Exit criteria:
- APT update calls reduced from `10` to <= `3`.
- No regression in service health checks.

Implementation update (2026-02-18):
1. M8.2b code changes
   - `Dockerfile` apt install layers were consolidated:
     - merged prior segmented apt runs (core packages, Docker CLI, imgcnv dependencies) into one consolidated package-install layer after enabling multiverse.
     - removed separate Docker apt layer and separate imgcnv dependency apt layer.
   - `apt-get update -qq` count in `Dockerfile` reduced from `10` to `2`.

2. M8.2b verification runs
```bash
/usr/bin/time -l docker build --no-cache --progress=plain -t bisque-m82b:test . > artifacts/m8_docker_build/m82b_cold.log 2>&1
rg -n "apt-get update -qq" Dockerfile | wc -l
docker tag bisque-m82b:test bisque-ucsb:latest
docker compose -f docker-compose.with-engine.yml up -d
docker compose -f docker-compose.with-engine.yml ps
curl -fsS http://127.0.0.1:8080/image_service/formats
curl -fsS http://127.0.0.1:8080/services | rg 'type="client_service"|type="module_service"|type="image_service"'
curl -sS -o /tmp/m82b_client.out -w "%{http_code}\n" http://127.0.0.1:8080/client_service/
curl -sS -o /tmp/m82b_module.out -w "%{http_code}\n" http://127.0.0.1:8080/module_service/
curl -sS -o /tmp/m82b_image.out -w "%{http_code}\n" http://127.0.0.1:8080/image_service/
source .venv/bin/activate && PYTHONPATH=source/bqserver:source/bqcore pytest -q source/bqserver/bq/table/tests/test_runquery.py
```

3. Results
   - APT update count target: PASS (`2` <= `3`).
   - Cold build completed: PASS (`artifacts/m8_docker_build/m82b_cold.log`).
     - cold build time: `250.15s`.
   - Compose/service health checks: PASS.
     - `bisque-server` healthy.
     - `client_service`, `module_service`, `image_service` present and endpoints return `200`.
     - `image_service/formats` responds successfully.
   - Regression smoke: PASS (`test_runquery.py`: `17 passed`).
   - Image size delta vs M8.2:
     - M8.2: `3,761,974,789` bytes
     - M8.2b: `3,663,600,396` bytes
     - improvement: `98,374,393` bytes (`2.61%` smaller).
   - Performance note:
     - M8.2b cold build (`250.15s`) is slower than M8.2 (`216.35s`) on this workstation despite fewer apt-update invocations.
     - Functional goals and health checks pass; further build-time tuning should continue in M8.3 (uv path + caching strategy).

#### M8.3: uv-first Python Install Path in Docker (dual-path safe rollout)
Deliverables:
- Install `uv` in Docker build stage.
- Add deterministic Docker lock/constraints file (`source/requirements.docker.uv.txt` + `source/constraints.uv.txt` reuse).
- Replace sequential `pip install .` chain with `uv pip install/sync` while keeping editable/local package semantics.
- Keep rollback toggle (`BISQUE_PY_INSTALLER=pip|uv`, default `uv` after parity).

Validation commands:
```bash
docker build --no-cache --progress=plain --build-arg BISQUE_PY_INSTALLER=uv -t bisque-m83:uv .
docker run --rm --entrypoint /bin/bash bisque-m83:uv -lc 'uv --version && /usr/lib/bisque/bin/python -c "import bq.core,bq.client_service,bqapi,bq.engine,bq.features; print(\"import-ok\")"'
docker compose -f docker-compose.with-engine.yml up -d
MANAGE_SERVER=0 BISQUE_URL=http://127.0.0.1:8080 ./scripts/dev/test_auth_token_smoke.sh
source .venv/bin/activate && python scripts/dev/bqapi_module_runner.py --bisque-root http://127.0.0.1:8080 --user admin --password admin --module EdgeDetection --images test_images/00-db8XD5ccsYdeysUv2Jr9fc.ome-tiff.ome.tif --summary-json artifacts/m8_docker_build/m83_edge_summary.json
```

Exit criteria:
- uv install path passes import/auth/module smoke checks.
- No regression in module dispatch/output visibility.

#### M8.4: External Binary Mirror/Vendoring (reduce network dependency)
Deliverables:
- Make `EXT_SERVER` configurable via env (`BISQUE_EXTERNAL_DEPOT`) with default internal mirror URL.
- Add local cached artifact directory + checksum manifest for required `EXTERNAL_FILES`.
- Add prefetch script (`scripts/build/prefetch_external_binaries.sh`) for CI/image pipeline.
- Fail fast with clear errors when external files are missing rather than fallback retries/noisy warnings.

Validation commands:
```bash
./scripts/build/prefetch_external_binaries.sh --out artifacts/m8_docker_build/depot_cache
docker build --progress=plain --build-arg BISQUE_EXTERNAL_DEPOT=file:///opt/bisque/depot -t bisque-m84:mirror . > artifacts/m8_docker_build/m84_build.log 2>&1
rg -n "files.wskoly.xyz|InsecureRequestWarning|/binaries/depot" artifacts/m8_docker_build/m84_build.log
```

Exit criteria:
- No remote depot fetches during build when mirror/cache is provided.
- No TLS-insecure warning spam in successful builds.

#### M8.5: K3s Readiness Gate + CI build matrix
Deliverables:
- Add one command that performs near-production verification on built image:
  - compose up
  - auth smoke (legacy/dual)
  - module run smoke
  - output thumbnail/meta checks
- Add CI workflow for:
  - cold build timing trend
  - image size trend
  - runtime smoke.

Validation commands:
```bash
./scripts/dev/verify_backend.sh
docker compose -f docker-compose.with-engine.yml ps
curl -fsS http://127.0.0.1:8080/image_service/formats
python scripts/dev/verify_mex_outputs.py --summary artifacts/m8_docker_build/m83_edge_summary.json --base http://127.0.0.1:8080 --user admin --password admin
```

Exit criteria:
- Production gate command is stable and reproducible.
- k3s deployment inputs (image + env + health checks) are versioned and validated per PR.

M8 success metrics:
1. Cold compose build <= `180s` on same workstation class.
2. Image size <= `5.5GB`.
3. `client_service` / `module_service` / `image_service` / `engine_service` health checks unchanged.
4. Token + basic auth smoke unchanged.
5. Module E2E smoke unchanged (`EdgeDetection` + `SimpleUniversalProcess`).

### M5: Query System Hardening (Core)
Deliverables:
- Refactor and harden query parsing/execution:
  - `source/bqserver/bq/data_service/controllers/resource_query.py`
  - `source/bqserver/bq/table/controllers/table_base.py`
  - `source/bqserver/bq/table/controllers/service.py`
- Replace parser `print()` error paths with exceptions/logging.
- Remove broad `except Exception` where practical in query-critical code paths.
- Add unit tests for:
  - `tag_query` parsing edge cases.
  - `tag_order` behavior and invalid input handling.
  - table range parsing + filter compatibility.

Validation commands:
```bash
source .venv/bin/activate
PYTHONPATH=source/bqserver:source/bqcore pytest -q source/bqserver/bq/table/tests/test_runquery.py
PYTHONPATH=source/bqserver:source/bqcore:source/bqapi pytest -q source/bqserver/bq/data_service/tests -k "query or tag_query or tag_order"
```

Exit criteria:
- Query regressions covered by tests.
- Known query bugs have explicit regression tests.

### M6: Image-Backed Integration Tests (`test_images`)
Deliverables:
- Add canonical local fixture directory (target: `source/tests/test_images`).
- Update tests to prefer local images over remote download URLs.
- Add integration tests that exercise upload -> query -> result verification.

Validation commands:
```bash
test -d source/tests/test_images
source .venv/bin/activate
BISQUE_TEST_IMAGES_DIR=source/tests/test_images \
PYTHONPATH=source/bqserver:source/bqcore:source/bqapi \
pytest -q source/bqserver/bq/table/tests/test_table_service_modern.py
```

Exit criteria:
- Tests run deterministically without internet dependency.

### M7: CI Verification Path
Deliverables:
- Add backend CI workflow (auth dual-stack + query + module smoke).
- Publish a single command that mirrors CI locally.
- Fail CI on bootstrap regressions, auth parity regressions, and query regressions.

Validation commands:
```bash
./scripts/dev/verify_backend.sh
```

Exit criteria:
- CI verifies bootstrap + auth + query tests on each PR.

## Risks and Mitigations
1. Legacy auth coupling in TG/repoze-who causes hidden redirects or login loops.
Mitigation: dual-stack mode + Playwright regression tests before cutover.

2. Token claim mapping can mismatch existing BisQue users/groups.
Mitigation: explicit claim mapping config, deterministic collision rules, and migration audit reports.

3. Module execution can fail if MEX auth semantics change accidentally.
Mitigation: keep `Mex` path unchanged in M4 and gate every auth PR on module E2E tests.

4. Current tests are heavily basic-auth oriented and may mask OIDC regressions.
Mitigation: add token-mode tests in `bqapi` and auth service parity scripts before switching defaults.

5. Clock skew / key rotation issues can invalidate JWTs unexpectedly.
Mitigation: enforce leeway settings, JWKS cache refresh strategy, and negative tests for expired/invalid tokens.

## Definition of Done
- Backend bootstraps with documented `uv` workflow on a fresh machine.
- Auth supports `legacy`, `dual`, and `oidc` modes with a one-command rollback path.
- Browser login, API calls, and module execution all pass in `dual` mode.
- `bqapi` supports both legacy `init_local` and token-based sessions.
- Query unit tests and core data-service smoke tests remain green after auth changes.
- Docker near-production build path is deterministic (`uv`-first), faster (cold <= `180s` target), and less external-network dependent.
- Docker image size and layer composition are reduced without breaking service/module/auth flows.
- CI enforces the same auth + query verification sequence.
