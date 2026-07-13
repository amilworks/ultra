# MatTools promotion gate

`scripts/mattools_promotion_gate.py` is the reproducible MatTools lane for the
materials-readiness contract. It does not contain or download the benchmark.
An operator must supply an explicit checkout of the official repository.

## Locked upstream evidence

- Repository: <https://github.com/Grenzlinie/MatTools>
- Revision: `1803a6abfe23a9da56c894076c59117873b758ff`
- Protected clean 3,193-file Git-tree manifest SHA-256:
  `c70c9c5b1d085643372728e4017c28282e190cd452afa2f5e7fd3366e1a9528e`
- Dataset card and DOI:
  <https://huggingface.co/datasets/SiyuLiu/MatTools>, `10.57967/hf/5486`
- Repository license: Apache-2.0
- Dataset-card license: CC-BY-NC-4.0
- Official scientific pins: pymatgen `2024.8.9` and
  pymatgen-analysis-defects `2024.7.19`

The upstream paper and repository define 49 parent questions and 138
scientific properties. The runner reports a parent as runnable when generated
code returns a parseable, non-empty dictionary. Its verifier scripts then
classify each property. The promotion protocol repeats the complete corpus
three times, so the fixed aggregate denominators are 147 parents and 414
properties. Passing requires at least 118 runnable parents and 249 accepted
properties.

## Evaluator-image decision

The repository's build inputs are internally inconsistent. Its Dockerfile uses
Python 3.11.8, while `pyproject.toml` requires Python `>=3.13,<4.0` and every
entry in the exported `requirements.txt` is guarded by that same lower bound.
Empirical, non-task probes established both failure modes:

- Building the unmodified upstream Dockerfile on Python 3.11.8 succeeds but
  ignores all 309 exported requirement entries. Neither pymatgen nor
  pymatgen-analysis-defects is installed.
- Python 3.13 activates the export, but no NumPy 1.26.4 binary exists for that
  interpreter. The locked environment therefore cannot be installed unchanged.

Ultra consequently uses a clearly labeled **reviewed reconstruction variant**,
not an upstream-published or “official” image artifact. It retains the upstream
Python 3.11.8 interpreter, changes only the export's global `>=3.13`/`==3.13`
markers to 3.11, and adds hash-pinned `ruamel.yaml.clib==0.2.12`, which was the
latest release available at the upstream commit date and is required by
ruamel.yaml on CPython below 3.13. The image is linux/arm64-specific.

The tracked build and lock inputs are:

- `deploy/docker/mattools-evaluator.Dockerfile`
- `deploy/docker/mattools-evaluator-supplemental-requirements.txt`
- `deploy/docker/mattools-evaluator-linux-arm64-lock.json`
- `scripts/build_mattools_evaluator.py`

The lock records all 290 installed distributions, Python 3.11.8, the
digest-pinned base image, the exact 2,756-file `tool_source_code` manifest,
requirements transformations, strict-shadow hash, and platform. Its current
file SHA-256 is
`5e9e9432267584e1e902e434f031f1006180ca5b4b0888c5e3747592b86324b1`.
The locally verified image ID is
`sha256:3c6318dfcb3a070123cf5779368bceb2b3ddd4cac4bd2cf5c77180abe1ad7b27`;
operators must still pass the actually inspected immutable ID explicitly.

The upstream README also names `grenzlinie/mat-tools:latest`. A metadata-only
audit resolved that tag to registry manifest
`sha256:f17faff921a093d7ea2bba508a907b348a19035f64b6087d7b62658eac813556`
and config image ID
`sha256:507832acf342902e89e3a8faa130515fdefec0defa0960badfef40a3d9c74d2e`.
It is linux/arm64, Python 3.11.8, has 297 distributions and the two exact
scientific pins. It has no labels or embedded public Git revision, predates the
public repository history, and differs from the pinned checkout in its
requirements plus one notebook and two `.DS_Store` files. It is therefore
pin-able and usable for investigation, but is not automatically approved as a
comparable evaluator. Its complete package map and provenance are retained in
`deploy/docker/mattools-upstream-published-linux-arm64-audit.json` (SHA-256
`8eb3645e63ecc382cb8191e3ece45f180c921821c264484c6801d7a2a41e6d27`).

The reviewed evaluator lock must enumerate every installed distribution, not
only the two scientific pins. It also records that this is a variant and binds
the Git-tracked build inputs:

The lock's `build.builder_path` and `build.builder_sha256` fields bind the
evaluator builder itself in addition to the Dockerfile, supplemental
requirements, and strict shadow. A builder change therefore requires an
explicitly reviewed lock refresh.

```json
{
  "schema_version": "1",
  "environment_kind": "reviewed-reconstruction-variant",
  "official_artifact": false,
  "python_version": "3.11.8",
  "platform": {"docker": "linux/arm64", "machine": "aarch64"},
  "build": {"tool_source_manifest_sha256": "..."},
  "packages": {"distribution-name": "exact-version"}
}
```

## Safety and comparability rules

- Ultra receives `question.txt` plus a generic code-output contract. The task
  directory name, `properties.json`, and `new_unit_test.py` are not sent.
- Every request goes through `/v2/threads` and
  `/v2/threads/{thread_id}/runs`; there is no model-endpoint client.
- The request includes only generic top-level
  `selection_context: {suggested_domain: materials}` routing. Benchmark name,
  revision, task ID/order, trial, filenames, function names, workflow hints,
  and thread-title identifiers do not cross the runtime/model boundary.
- `--model-id` and `--provider-id` are operator declarations, not API model
  selectors. A comparable run requires matching model and provider IDs in
  observable runtime events. Missing or mismatched runtime provenance blocks
  promotion; do not override that result when an older runtime emits a model
  but omits its provider.
- Candidate code is captured but never executed by the harness or host.
- Scoring invokes the unmodified upstream `src/result_analysis.py`. Generated
  code and verification run in the reviewed `mat-tool-ben` reconstruction
  variant. Reports preserve that classification and never call it an official
  image artifact.
- Snapshot verification requires the exact clean Git revision and hashes all
  3,193 tracked paths, including every file copied by the Dockerfile. Dirty or
  archive-only snapshots are rejected.
- The evaluator image ID must differ from Ultra's production runtime image
  digest. Production versions are provenance only and never substitute for the
  pinned evaluator stack.
- Two immediate evaluator replays must agree on runnable/scientific
  classifications.
- Every completed replay is sealed before it is checkpointed. The write-once
  `terminal-replays/<sha256>.json` record binds the official input/log,
  strict-shadow JSON and stdout/stderr, evaluator image before/after, optional
  workbook, and all parsed classifications. Resume and report regeneration
  reject a replay whose record, order, hash, or campaign-relative seal path has
  changed. The audit also exact-compares the referenced seal set with the
  direct, content-addressed files in `terminal-attempts/` and
  `terminal-replays/`; an orphan/replaced seal blocks promotion. Failed replay
  artifacts are sealed and retained, and any failed-replay history makes that
  campaign non-comparable instead of allowing score cherry-picking.
- The published score preserves upstream's loose substring behavior. A
  separate hashed shadow evaluator captures raw verifier output before
  `run_test()` normalizes any string containing `ok`; the strict scientific
  gate accepts only a raw JSON string exactly equal to `ok` or validated partial
  counters. Both published and strict scores are reported.
- The upstream summary also misclassifies some verifier failures as function
  errors. Promotion FRR therefore comes from the strict shadow's independent
  parseable-nonempty-dictionary observation. The upstream runner's historical
  runnable count remains separately labeled and never substitutes for semantic
  FRR.
- Upstream's host-side Docker orchestration runs in an isolated `uv`
  Python 3.11.9 environment locked with hashes in
  `scripts/mattools-validator-requirements.lock.txt`; the smaller
  `scripts/mattools-validator-requirements.txt` is its reviewed input. A real
  no-task import/package-map preflight runs before any Ultra submission. Those
  host packages parse evaluator output but never substitute for the scientific
  packages in the evaluator image, and no arbitrary validator-Python override
  is supported.
- Exact evaluator status additionally requires a reviewed, committed JSON lock
  containing the evaluator Python version, platform, complete resolved package
  map, build labels, requirements hashes, exact tool-source manifest, and
  strict-shadow hash. Matching only pymatgen and pymatgen-analysis-defects is
  insufficient. Until the new lock and every referenced build input are
  committed and unchanged from Git HEAD, the harness deliberately rejects it.
- A subset is diagnostic-only. The harness will not calculate a comparable
  aggregate rate for fewer than three complete 49-question trials.
- Secrets are accepted only through named environment variables. Their values
  are not stored. Event payloads are represented by hashes in the checkpoint.
- Terminal Ultra failures are retained as failing submissions. Resume polls the
  same run/idempotency key; it cannot replace a terminal attempt.
- Comparable attempts require completed production `execute` tool-call evidence
  and an execute-event image digest matching the declared immutable production
  runtime. A code artifact alone is insufficient.
- `report` rehashes prompt/code/response/trace/artifact files, reconstructs
  official JSONL, reparses upstream logs, and reparses raw strict-shadow output.
  Editable checkpoint score/provenance booleans are never authoritative.
- The report exposes independently summable evidence on each
  `trials[].attempts[]` record. `scoring_evidence.replays[]` keeps the historical
  upstream classification under `published_upstream` and the promotion facts
  under `strict_shadow`; these namespaces are deliberately separate. A primary
  replay, exact replay count, and replay-consistency bit are derived from the
  same sealed records rather than from aggregate checkpoint counters.
- Report timestamps derive from the immutable checkpoint update time. Given the
  same checkpoint and evidence, `results.json`, `results.md`, and manifest v2
  are byte-for-byte deterministic. `verify-report` is a read-only verifier: it
  rehashes the three manifest records, revalidates terminal attempt/replay
  seals, regenerates JSON and Markdown, reconstructs the manifest, and exact-
  compares all bytes. It never submits or executes a benchmark task.

The per-attempt scoring contract is:

```json
{
  "scoring_evidence": {
    "schema_version": "1",
    "task_id": "...",
    "ordinal": 1,
    "subtask_count": 3,
    "expected_replay_count": 2,
    "replay_count": 2,
    "complete": true,
    "replay_consistent": true,
    "primary": {
      "replay": 1,
      "replay_terminal_record_sha256": "...",
      "published_upstream": {
        "classification": "success|partial|function_error",
        "runnable": true,
        "scientific_pass": 3,
        "scientific_fail": 0
      },
      "strict_shadow": {
        "semantic_runnable": true,
        "strict_scientific_classification": "strict_success|strict_partial|strict_failure|strict_function_error|strict_unverifiable_truncated|strict_invalid_counters",
        "strict_scientific_pass": 3,
        "strict_scientific_fail": 0,
        "strict_exact_ok": true,
        "raw_verifier_output_sha256": "..."
      }
    },
    "replays": ["same shape as primary for every configured replay"]
  }
}
```

Final aggregation must sum `primary.strict_shadow.semantic_runnable` for FRR
and `primary.strict_shadow.strict_scientific_pass` for strict TSR only after
checking all 147 attempt records, their subtask denominators, exact replay
counts, `complete`, `replay_consistent`, and terminal-record hashes. Published
upstream fields are retained for benchmark-history comparison and must not
substitute for those promotion measures.

The unmodified upstream Docker client creates default-network containers and
sets no resource limits. The harness does not repair or override that behavior,
so it does **not** enforce or independently prove sandbox isolation. A passing
protected lane requires external host/daemon policy evidence, a JSON
attestation bound to that evidence and the exact image ID, plus a detached
operator signature:

```json
{
  "attestation_kind": "external_sandbox_isolation",
  "evaluator_image_id": "sha256:...",
  "network_egress_denied": true,
  "host_access_denied": true,
  "resource_limits_enforced": true,
  "external_enforcement": true,
  "enforcement_mechanism": "reviewed host or Docker-daemon policy identifier",
  "isolation_evidence_path": "mattools-isolation-evidence.json",
  "isolation_evidence_sha256": "sha256:...",
  "signed_by": "release operator identity",
  "signed_at": "2026-07-09T00:00:00Z"
}
```

The referenced isolation-evidence JSON is also machine-checked. It must bind to
the same image and include an observed container ID/time, a blocked network
egress probe, zero host mounts/no Docker socket, and positive memory, PID, and
CPU limits. A free-form assertion file does not pass.

Sign the exact attestation bytes and retain the public key separately, for
example with `openssl dgst -sha256 -sign operator-private.pem -out
mattools-sandbox.sig mattools-sandbox.json`. The harness verifies the evidence
hash and detached signature with `openssl`; the public key must itself be a
reviewed, unchanged file tracked by the current Ultra Git commit. That integrity check is not a
substitute for an independent review of whether the external policy was
actually enforced. Without all three files, the harness refuses to execute any
candidate code; it does not merely downgrade the report afterward.

## Usage

Verify a supplied snapshot without submitting any run:

```bash
MATTOOLS_BENCHMARK_ROOT=/path/to/MatTools make mattools-promotion-inspect
```

Run lean unit tests:

```bash
make mattools-promotion-test
```

Read-only verification of an existing report bundle (no task submission or
candidate execution):

```bash
uv run --python 3.11 python scripts/mattools_promotion_gate.py verify-report \
  --benchmark-root /path/to/MatTools \
  --report-manifest /secure/results/mattools-release-candidate/report_manifest.json
```

The verifier validates an `ultra.mattools.report_bundle.v2` manifest and emits
`ultra.mattools.report_revalidation.v1` evidence with
`task_execution_performed: false`. Final readiness must require `valid`,
`bundle_exact`, `manifest_integrity_valid`, `checkpoint_evidence_valid`,
`checkpoint_exact`, `results_json_exact`, `results_markdown_exact`, and
`manifest_exact` all to be true; it must still independently require the
promotion thresholds and every hard gate from the regenerated report.

Build or verify the evaluator reconstruction without submitting or executing
any benchmark task:

```bash
MATTOOLS_BENCHMARK_ROOT=/path/to/MatTools make mattools-evaluator-build
MATTOOLS_BENCHMARK_ROOT=/path/to/MatTools make mattools-evaluator-verify
```

The verification command hashes the full embedded tool source and installed
distribution map under `--network none --read-only`, validates the scientific
pins and strict-shadow capture source, and reports
`task_execution_performed: false`.

Run one complete 49-question diagnostic trial through the live control plane
and exact evaluator. It reports trial-local FRR/TSR but is structurally
non-promotable; the later three trials must be clean, independent promotion
trials:

```bash
export ULTRA_LIVE_TRACE_COOKIE='ultra_workos_session=...'
MATTOOLS_BENCHMARK_ROOT=/path/to/MatTools \
MATTOOLS_OUTPUT_DIR=/secure/results/mattools-diagnostic \
MATTOOLS_SANDBOX_ATTESTATION=/secure/attestations/mattools-sandbox.json \
MATTOOLS_SANDBOX_SIGNATURE=/secure/attestations/mattools-sandbox.sig \
MATTOOLS_SANDBOX_PUBLIC_KEY=security/release-operator-public.pem \
MATTOOLS_EVALUATOR_IMAGE_ID=sha256:... \
MATTOOLS_EVALUATOR_ENV_LOCK=deploy/docker/mattools-evaluator-linux-arm64-lock.json \
ULTRA_RUNTIME_IMAGE_DIGEST=sha256:... \
ULTRA_RUNTIME_PYMATGEN_VERSION=2026.5.4 \
ULTRA_RUNTIME_DEFECTS_VERSION=2025.1.18 \
ULTRA_MODEL_ID=... \
ULTRA_PROVIDER_ID=... \
MATTOOLS_LICENSE_BASIS=noncommercial \
MATTOOLS_USE_PURPOSE='noncommercial internal qualification' \
make mattools-promotion-diagnostic
```

Run the full protected lane only after the diagnostic is accepted:

```bash
MATTOOLS_BENCHMARK_ROOT=/path/to/MatTools \
MATTOOLS_OUTPUT_DIR=/secure/results/mattools-release-candidate \
MATTOOLS_SANDBOX_ATTESTATION=/secure/attestations/mattools-sandbox.json \
MATTOOLS_SANDBOX_SIGNATURE=/secure/attestations/mattools-sandbox.sig \
MATTOOLS_SANDBOX_PUBLIC_KEY=security/release-operator-public.pem \
MATTOOLS_EVALUATOR_IMAGE_ID=sha256:... \
MATTOOLS_EVALUATOR_ENV_LOCK=deploy/docker/mattools-evaluator-linux-arm64-lock.json \
ULTRA_RUNTIME_IMAGE_DIGEST=sha256:... \
ULTRA_RUNTIME_PYMATGEN_VERSION=2026.5.4 \
ULTRA_RUNTIME_DEFECTS_VERSION=2025.1.18 \
ULTRA_MODEL_ID=... \
ULTRA_PROVIDER_ID=... \
MATTOOLS_LICENSE_BASIS=noncommercial \
MATTOOLS_USE_PURPOSE='noncommercial internal qualification' \
make mattools-promotion-gate
```

The output directory contains atomic `state.json` checkpoints, write-once
terminal attempt/replay records, redacted trace records, exact captured code,
upstream-compatible per-trial JSONL, raw upstream logs/workbooks,
`results.json`, `results.md`, and deterministic manifest v2. Manifest v2 binds
the exact checkpoint plus regenerated JSON and Markdown hashes and declares the
read-only byte-exact regeneration contract.
Incomplete or non-comparable campaigns use `null` aggregate rates rather than
fabricating a zero score.

`MATTOOLS_LICENSE_BASIS=noncommercial` is a structured operator declaration,
not an automatic legal conclusion. If use is separately licensed, set
`MATTOOLS_LICENSE_BASIS=separately_licensed` and bind the reviewed external
license evidence with `MATTOOLS_LICENSE_EVIDENCE_SHA256=<64 lowercase hex>`
(the canonical plain digest emitted by `sha256sum`, without a `sha256:`
prefix); the license document itself is not copied into the report.
