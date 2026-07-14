from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "materials_readiness_gate.py"
SPEC = importlib.util.spec_from_file_location("materials_readiness_gate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)
REAL_HTTP_REVALIDATION = gate._run_calphad_real_http_revalidation


GIT_SHA = "a" * 40
DOMAIN_IMAGE = "sha256:" + "1" * 64
RUNTIME_IMAGE = "sha256:" + "2" * 64
EVALUATOR_IMAGE = "sha256:" + "3" * 64
MODEL_ID = "deepseek_v4"
PROVIDER_ID = "openai"


def test_cross_language_source_contract_matches_producer() -> None:
    module_path = ROOT / "scripts/calphad_cross_language_gate.py"
    spec = importlib.util.spec_from_file_location("readiness_cross_language_contract", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        assert tuple(path.as_posix() for path in module.SOURCE_PATHS) == (
            gate.REQUIRED_CALPHAD_CROSS_LANGUAGE_SOURCE_FILES
        )
    finally:
        sys.modules.pop(spec.name, None)


def test_real_http_revalidation_requires_test_and_package_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package = gate.CALPHAD_REAL_HTTP_REVALIDATION_PACKAGE
    test_name = gate.CALPHAD_REAL_HTTP_REVALIDATION_TEST

    def completed(events: list[dict[str, Any]], returncode: int) -> subprocess.CompletedProcess:
        output = "".join(json.dumps(event) + "\n" for event in events).encode()
        return subprocess.CompletedProcess(
            gate.CALPHAD_REAL_HTTP_REVALIDATION_COMMAND,
            returncode,
            stdout=output,
            stderr=b"",
        )

    database_input = tmp_path / "database.tdb"
    inspection = tmp_path / "inspect.json"
    equilibrium = tmp_path / "equilibrium.json"
    database_input.write_text("$ readiness revalidation fixture\n")
    inspection.write_text("{}")
    equilibrium.write_text("{}")
    (tmp_path / "backend/controlplane").mkdir(parents=True)
    passing = [
        {"Action": "pass", "Test": test_name, "Package": package},
        {"Action": "pass", "Package": package},
    ]

    def passing_run(*_args: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        environment = kwargs["env"]
        assert environment["ULTRA_CALPHAD_DATABASE_INPUT_ARTIFACT"] == str(database_input.resolve())
        assert environment["ULTRA_CALPHAD_INSPECTION_ARTIFACT"] == str(inspection.resolve())
        assert environment["ULTRA_CALPHAD_EQUILIBRIUM_ARTIFACT"] == str(equilibrium.resolve())
        return completed(passing, 0)

    monkeypatch.setattr(subprocess, "run", passing_run)
    result = REAL_HTTP_REVALIDATION(
        tmp_path, database_input, inspection, equilibrium, RUNTIME_IMAGE
    )
    assert result["valid"] is True

    package_failure = [
        {"Action": "pass", "Test": test_name, "Package": package},
        {"Action": "fail", "Package": package},
    ]
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: completed(package_failure, 1),
    )
    result = REAL_HTTP_REVALIDATION(
        tmp_path, database_input, inspection, equilibrium, RUNTIME_IMAGE
    )
    assert result["valid"] is False


def test_real_http_revalidation_rejects_empty_scientific_artifacts(tmp_path: Path) -> None:
    database_input = tmp_path / "database.tdb"
    inspection = tmp_path / "inspect.json"
    equilibrium = tmp_path / "equilibrium.json"
    database_input.write_text("$ readiness revalidation fixture\n")
    inspection.write_text("{}")
    equilibrium.write_text("{}")
    result = REAL_HTTP_REVALIDATION(
        ROOT,
        database_input,
        inspection,
        equilibrium,
        RUNTIME_IMAGE,
    )
    assert result["valid"] is False


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, content: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return _sha(path)


def _manifest_hash(file_hashes: dict[str, str]) -> str:
    text = "".join(f"{file_hashes[path]}  {path}\n" for path in sorted(file_hashes))
    return hashlib.sha256(text.encode()).hexdigest()


def _skills_hash(repository_root: Path) -> str:
    skills = repository_root / "backend" / "deepagents_runtime" / "skills"
    hashes = {
        str(path.relative_to(repository_root)): _sha(path)
        for path in sorted(skills.rglob("*"))
        if path.is_file()
    }
    return _manifest_hash(hashes)


def _worker_cleanroom_records(
    run_id: str,
    thread_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_sha = hashlib.sha256(run_id.encode()).hexdigest()
    thread_sha = hashlib.sha256(thread_id.encode()).hexdigest()
    goal_sha = hashlib.sha256(f"goal:{run_id}".encode()).hexdigest()
    user_sha = hashlib.sha256(b"researcher-1").hexdigest()
    payload = {
        "schema_version": "1",
        "attestation_kind": "worker_evaluation_profile",
        "worker_owned": True,
        "evaluation_profile": gate.MATERIALS_CLEANROOM_PROFILE,
        "profile_source": "typed_job_envelope",
        "trusted_envelope_field": "evaluation_profile",
        "namespace_id": f"{gate.MATERIALS_CLEANROOM_PROFILE}-{run_sha}",
        "run_id_sha256": run_sha,
        "thread_id_sha256": thread_sha,
        "user_id_sha256": user_sha,
        "goal_sha256": goal_sha,
        "input_policy": "goal_only",
        "provided_message_count": 1,
        "effective_message_count": 1,
        "prior_thread_context_discarded": True,
        "same_run_retry_state_allowed": True,
        "run_scoped_workspace": True,
        "run_scoped_memory": True,
        "disabled_capabilities": list(gate.WORKER_CLEANROOM_DISABLED_CAPABILITIES),
    }
    payload["attestation_sha256"] = gate.canonical_json_sha256(payload)
    attestation = {
        "valid": True,
        "payload": payload,
        "source_payload_keys": sorted(gate.WORKER_EVALUATION_ATTESTATION_FIELDS),
    }
    binding = {
        "evaluation_profile": gate.MATERIALS_CLEANROOM_PROFILE,
        "worker_event_count": 1,
        "worker_attestation_valid": True,
        "server_attestation_valid": True,
        "identity_hash_checks": {
            "run_id_sha256": True,
            "thread_id_sha256": True,
            "goal_sha256": True,
            "user_id_sha256": True,
        },
        "user_identity_independently_bound": True,
        "valid": True,
    }
    return [attestation], binding


def _make_repository(root: Path) -> dict[str, str]:
    values = {
        "requirements": _write(
            root / "deploy/docker/materials-requirements.txt", "pymatgen==2026.5.4\n"
        ),
        "dockerfile": _write(
            root / "deploy/docker/materials-domain-gate.Dockerfile", "FROM scratch\n"
        ),
        "domain_test": _write(
            root
            / "backend/deepagents_runtime/tests/domain_correctness/test_materials_invariants.py",
            "def test_materials(): assert True\n",
        ),
        "harness": _write(root / "scripts/mattools_promotion_gate.py", "# harness\n"),
        "shadow": _write(root / "scripts/mattools_strict_shadow.py", "# shadow\n"),
        "safe_parser": _write(root / "scripts/mattools_safe_parser.py", "# safe parser\n"),
        "runner_wrapper": _write(root / "scripts/mattools_runner_wrapper.py", "# runner wrapper\n"),
        "semantic_repairs": _write(
            root / "scripts/mattools_semantic_repairs.py", "# semantic repairs\n"
        ),
        "validator_input": _write(
            root / "scripts/mattools-validator-requirements.txt",
            (ROOT / "scripts/mattools-validator-requirements.txt").read_text(),
        ),
        "validator_lock": _write(
            root / "scripts/mattools-validator-requirements.lock.txt",
            (ROOT / "scripts/mattools-validator-requirements.lock.txt").read_text(),
        ),
        "evaluator_builder": _write(
            root / "scripts/build_mattools_evaluator.py", "# evaluator builder\n"
        ),
        "evaluator_dockerfile": _write(
            root / "deploy/docker/mattools-evaluator.Dockerfile", "FROM scratch\n"
        ),
        "evaluator_supplemental": _write(
            root / "deploy/docker/mattools-evaluator-supplemental-requirements.txt",
            "fixture==1\n",
        ),
        "materials_validation": _write(
            root / "backend/deepagents_runtime/src/ultra_deepagents/materials/validation.py",
            (
                ROOT / "backend/deepagents_runtime/src/ultra_deepagents/materials/validation.py"
            ).read_text(),
        ),
    }
    for relative in gate.REQUIRED_CALPHAD_LEDGER_SOURCE_FILES:
        path = root / relative
        if not path.exists():
            _write(path, f"fixture for {relative}\n")
    for relative in gate.REQUIRED_CALPHAD_CROSS_LANGUAGE_SOURCE_FILES:
        path = root / relative
        if not path.exists():
            _write(path, f"cross-language fixture for {relative}\n")
    for relative in gate.REQUIRED_CALPHAD_RELEASE_INPUT_FILES:
        path = root / relative
        if not path.exists():
            _write(path, f"release fixture for {relative}\n")
    database_path = (
        root / "backend/deepagents_runtime/materials_data/calphad/alcow_CALPHAD-2017-Wang.tdb"
    )
    database_sha = _sha(database_path)
    _write(
        root / "backend/deepagents_runtime/materials_data/calphad/manifest.json",
        json.dumps(
            {
                "schema_version": "1",
                "databases": [
                    {
                        "database_id": "nist-al-co-w",
                        "filename": database_path.name,
                        "sha256": database_sha,
                        "size_bytes": database_path.stat().st_size,
                        "format": "tdb",
                        "assessment_pressure_limits_Pa": [1e-9, 1e12],
                        "elements": ["AL", "CO", "W"],
                        "phases": ["BCC_A2", "FCC_A1"],
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    _write(
        root / "backend/deepagents_runtime/skills/materials-structure-thermo/SKILL.md",
        "# Materials structure and thermodynamics\n",
    )
    values["skills"] = _skills_hash(root)
    return values


def _make_benchmark(root: Path) -> tuple[dict[str, Any], gate.ReadinessPolicy]:
    file_hashes: dict[str, str] = {}
    control_paths = ("LICENSE", "src/result_analysis.py", "src/docker_sandbox.py")
    for relative in control_paths:
        file_hashes[relative] = _write(root / relative, f"fixture for {relative}\n")

    tasks: list[dict[str, Any]] = []
    remaining_subtasks = 138
    for ordinal in range(1, 50):
        task_id = f"task_{ordinal:02d}"
        subtask_count = 3 if ordinal <= 40 else 2
        remaining_subtasks -= subtask_count
        base = f"src/question_segments/pymatgen_analysis_defects/{task_id}"
        question = f"{base}/question.txt"
        expected = f"{base}/properties.json"
        verifier = f"{base}/new_unit_test.py"
        file_hashes[question] = _write(root / question, f"Question {ordinal}\n")
        file_hashes[expected] = _write(root / expected, '{"properties": {}}\n')
        file_hashes[verifier] = _write(root / verifier, "def run_test(): return 'ok'\n")
        tasks.append(
            {
                "task_id": task_id,
                "ordinal": ordinal,
                "subtask_count": subtask_count,
                "question": {"path": "question.txt", "sha256": file_hashes[question]},
                "expected_values": {
                    "path": "properties.json",
                    "sha256": file_hashes[expected],
                    "isolated_from_ultra": True,
                },
                "verifier": {
                    "path": "new_unit_test.py",
                    "sha256": file_hashes[verifier],
                    "isolated_from_ultra": True,
                },
            }
        )
    assert remaining_subtasks == 0
    manifest = _manifest_hash(file_hashes)
    revision = "b" * 40
    policy = gate.ReadinessPolicy(
        official_revision=revision,
        official_manifest_sha256=manifest,
    )
    benchmark = {
        "name": "MatTools-real-world",
        "revision": revision,
        "revision_source": "git",
        "sha256": manifest,
        "official_manifest_sha256": manifest,
        "strict_official": True,
        "tracked_file_count": len(file_hashes),
        "full_git_tree_hashed": True,
        "git_checkout_clean": True,
        "parent_count": 49,
        "scientific_subtask_count": 138,
        "tracked_file_hashes": file_hashes,
        "control_file_hashes": {path: file_hashes[path] for path in control_paths},
        "tasks": tasks,
    }
    return benchmark, policy


def _make_isolation_attestation(root: Path) -> dict[str, Any]:
    evidence = root / "isolation-evidence.json"
    attestation = root / "sandbox-attestation.json"
    signature = root / "sandbox-attestation.sig"
    public_key = root / "operator-public-key.pem"
    evidence_payload = {
        "schema_version": "1",
        "evaluator_image_id": EVALUATOR_IMAGE,
        "observed_at": "2026-07-09T00:00:00Z",
        "observed_container_id": "fixture-container",
        "network_egress_probe": {"attempted": True, "result": "blocked"},
        "host_access_probe": {"host_mount_count": 0, "docker_socket_mounted": False},
        "resource_limits": {
            "memory_bytes": 8 * 1024**3,
            "pids_limit": 1024,
            "nano_cpus": 2_000_000_000,
        },
    }
    _write(evidence, json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n")
    evidence_sha = _sha(evidence)
    signed_payload = {
        "attestation_kind": "external_sandbox_isolation",
        "evaluator_image_id": EVALUATOR_IMAGE,
        "network_egress_denied": True,
        "host_access_denied": True,
        "resource_limits_enforced": True,
        "external_enforcement": True,
        "enforcement_mechanism": "dedicated offline worker policy",
        "signed_by": "release-operator",
        "signed_at": "2026-07-09T00:00:00Z",
        "isolation_evidence_path": evidence.name,
        "isolation_evidence_sha256": "sha256:" + evidence_sha,
    }
    _write(attestation, json.dumps(signed_payload, indent=2, sort_keys=True) + "\n")
    _write(signature, "detached-signature-fixture\n")
    _write(public_key, "public-key-fixture\n")
    return {
        "valid": True,
        "issues": [],
        "harness_enforces_isolation": False,
        "upstream_runner_declares_network_isolation": False,
        "upstream_runner_declares_resource_limits": False,
        "path": str(attestation),
        "sha256": _sha(attestation),
        "detached_signature_path": str(signature),
        "detached_signature_sha256": _sha(signature),
        "operator_public_key_path": str(public_key),
        "operator_public_key_sha256": _sha(public_key),
        "operator_public_key_trusted_from_git_head": True,
        "public_key_trust_anchor": "current Ultra Git HEAD",
        "operator_signature_verified": True,
        "signature_error": None,
        "signed_by": "release-operator",
        "signed_at": "2026-07-09T00:00:00Z",
        "attestation_kind": "external_sandbox_isolation",
        "evaluator_image_id": EVALUATOR_IMAGE,
        "network_egress_denied": True,
        "host_access_denied": True,
        "resource_limits_enforced": True,
        "external_enforcement": True,
        "enforcement_mechanism": "dedicated offline worker policy",
        "isolation_evidence_path": str(evidence),
        "isolation_evidence_sha256": "sha256:" + evidence_sha,
        "declared_isolation_evidence_sha256": "sha256:" + evidence_sha,
        "external_isolation_evidence_semantics_valid": True,
        "external_isolation_evidence_summary": {
            "schema_version": "1",
            "evaluator_image_id": EVALUATOR_IMAGE,
            "observed_at": "2026-07-09T00:00:00Z",
            "observed_container_id": "fixture-container",
            "network_egress_blocked": True,
            "host_access_blocked": True,
            "resource_limits_present": True,
        },
    }


def _calphad_experimental_benchmark_wrapper() -> dict[str, Any]:
    report = {
        "schema_version": "ultra.calphad.experimental_benchmark.v1",
        "benchmark_id": "materials.calphad.al_co_w_experimental_two_lane.v1",
        "status": "passed",
        "required_independent_invariant": True,
        "production_promotion_blocked": False,
        "blocking_reasons": [],
        "source_manifest": {
            "relative_path": (
                "backend/deepagents_runtime/materials_data/calphad/"
                "experimental_benchmark_manifest.json"
            ),
            "sha256": "e" * 64,
        },
        "database_binding": {
            "database_id": "nist-al-co-w-wang-2017",
            "sha256": "d" * 64,
        },
        "lanes": {
            "calibration": {
                "classification": "calibration",
                "independent_validation": False,
                "required": True,
                "status": "passed",
                "observation_count": 6,
                "metrics": {
                    "weighted_rms_z": 0.49,
                    "weighted_rms_z_max": 1.0,
                    "max_abs_z": 0.79,
                    "max_abs_z_max": 2.0,
                },
            },
            "held_out": {
                "classification": "held_out",
                "independent_validation": True,
                "required": True,
                "status": "passed",
                "observation_count": 4,
                "metrics": {
                    "mae_K": 12.34,
                    "mae_K_max": 20.0,
                    "max_abs_error_K": 20.42,
                    "max_abs_error_K_max": 30.0,
                },
                "observations": [
                    {
                        "reported_uncertainty_K": None,
                        "uncertainty_status": "not_reported_numerically",
                    }
                    for _ in range(4)
                ],
            },
        },
    }
    return {
        "relative_path": "calphad-experimental-benchmark.json",
        "sha256": "f" * 64,
        "size_bytes": len(json.dumps(report)),
        "report": report,
    }


def _make_domain_report(repository_root: Path, hashes: dict[str, str]) -> dict[str, Any]:
    invariants = [
        {
            "schema_version": "1",
            "validator_id": validator_id,
            "test_id": f"test_materials_invariant_{index:02d}",
            "required": True,
            "outcome": "pass",
            "observed": {"value": index},
            "expected": {"value": index},
            "tolerance_rationale": "exact controlled invariant",
            "units": "dimensionless",
            "convention": "declared fixture convention",
            "library_versions": {"fixture-library": "1.0"},
        }
        for index, validator_id in enumerate(gate.REQUIRED_DOMAIN_VALIDATORS, start=1)
    ]
    return {
        "schema_version": 1,
        "gate": "materials-domain-gate",
        "scope": "deterministic-domain-invariants",
        "status": "passed",
        "failures": [],
        "junit": {
            "tests": gate.REQUIRED_DOMAIN_INVARIANT_COUNT,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "time_seconds": 1.0,
        },
        "invariants": invariants,
        "invariant_evidence": {
            "schema_version": "1",
            "junit_property": "materials_invariant_evidence",
            "record_count": gate.REQUIRED_DOMAIN_INVARIANT_COUNT,
            "passed": gate.REQUIRED_DOMAIN_INVARIANT_COUNT,
            "failed": 0,
            "errors": [],
            "complete": True,
        },
        "pytest": {"exit_code": 0, "command": ["pytest"]},
        "version_drift": [],
        "calphad_experimental_benchmark": _calphad_experimental_benchmark_wrapper(),
        "runtime": {
            "calphad_runtime_preflight": {
                "path": "/outputs/calphad-runtime-junit.xml",
                "required": True,
                "validated": True,
                "junit": {
                    "tests": gate.CALPHAD_RUNTIME_TEST_COUNT,
                    "failures": 0,
                    "errors": 0,
                    "skipped": 0,
                    "time_seconds": 1.0,
                },
                "core_tests": gate.CALPHAD_RUNTIME_CORE_TEST_COUNT,
                "typed_cli_tests": gate.CALPHAD_RUNTIME_CLI_TEST_COUNT,
                "required_adversarial_test_names": sorted(
                    gate.REQUIRED_CALPHAD_ADVERSARIAL_TEST_NAMES
                ),
            }
        },
        "requirements": {
            "path": "/opt/ultra/materials-requirements.txt",
            "sha256": hashes["requirements"],
            "source_sha256": hashes["requirements"],
        },
        "test_source": {
            "path": (
                "/workspace/backend/deepagents_runtime/tests/domain_correctness/"
                "test_materials_invariants.py"
            ),
            "sha256": hashes["domain_test"],
        },
        "git": {"sha": GIT_SHA, "ref": "release", "dirty": False},
        "image": {
            "ref": "materials-gate:release",
            "id": DOMAIN_IMAGE,
            "digest": DOMAIN_IMAGE,
            "dockerfile_sha256": hashes["dockerfile"],
        },
        "provenance_policy": {
            "required": True,
            "status": "enforced",
            "issues": [],
            "would_pass_if_enforced": True,
            "promotion_provenance_enforced": True,
            "immutable_image_identifiers": [
                {"source": "image.id", "value": DOMAIN_IMAGE, "digest": DOMAIN_IMAGE}
            ],
        },
    }


def _make_production_parity_report(
    root: Path, deterministic: dict[str, Any], repository_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence_root = root / "production-parity"
    domain_path = evidence_root / "domain/materials-domain-gate.json"
    calphad_path = evidence_root / "calphad-embedded-probe.json"
    runtime_path = evidence_root / "calphad-runtime-junit.xml"
    tools_path = evidence_root / "calphad-tools-junit.xml"

    embedded_domain = copy.deepcopy(deterministic)
    embedded_domain["image"].update(id=RUNTIME_IMAGE, digest=RUNTIME_IMAGE)
    embedded_domain["provenance_policy"]["immutable_image_identifiers"] = [
        {"source": "image.id", "value": RUNTIME_IMAGE, "digest": RUNTIME_IMAGE}
    ]
    domain_sha = _write(domain_path, json.dumps(embedded_domain, indent=2, sort_keys=True) + "\n")

    manifest_path = (
        repository_root / "backend/deepagents_runtime/materials_data/calphad/manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = _sha(manifest_path)
    material_paths = sorted(
        (repository_root / "backend/deepagents_runtime/src/ultra_deepagents/materials").glob("*.py")
    )
    material_hashes = {path.name: _sha(path) for path in material_paths}
    manifest_database = manifest["databases"][0]
    calphad = {
        "schema_version": 1,
        "status": "passed",
        "equilibrium_schema_version": "ultra.calphad.equilibrium.v2",
        "baked_materials_path": "/opt/ultra-runtime/ultra_deepagents/materials",
        "materials_source_hashes": material_hashes,
        "materials_baked_hashes": material_hashes,
        "source_manifest_sha256": manifest_sha,
        "embedded_manifest_sha256": manifest_sha,
        "database_count": 1,
        "databases": [
            {
                "database_id": manifest_database["database_id"],
                "filename": manifest_database["filename"],
                "sha256": manifest_database["sha256"],
                "size_bytes": manifest_database["size_bytes"],
                "format": manifest_database["format"],
                "assessment_pressure_limits_Pa": manifest_database["assessment_pressure_limits_Pa"],
                "elements": manifest_database["elements"],
                "phases": manifest_database["phases"],
                "pycalphad_parse_supported": True,
                "ultra_inspection_supported": True,
            }
        ],
    }
    calphad_sha = _write(calphad_path, json.dumps(calphad, indent=2, sort_keys=True) + "\n")
    runtime_cases = "".join(
        f'<testcase classname="tests.test_calphad_runtime" name="{name}" />'
        for name in gate.REQUIRED_CALPHAD_RUNTIME_CORE_TEST_NAMES
    )
    runtime_cases += "".join(
        f'<testcase classname="tests.test_calphad_cli" name="{name}" />'
        for name in gate.REQUIRED_CALPHAD_RUNTIME_CLI_TEST_NAMES
    )
    runtime_sha = _write(
        runtime_path,
        f'<testsuites><testsuite tests="{gate.CALPHAD_RUNTIME_TEST_COUNT}" '
        'failures="0" errors="0" skipped="0">'
        f"{runtime_cases}</testsuite></testsuites>\n",
    )
    tools_cases = "".join(
        f'<testcase classname="tests.test_calphad_tools" name="{name}" />'
        for name in gate.REQUIRED_CALPHAD_TOOL_TEST_NAMES
    )
    tools_sha = _write(
        tools_path,
        f'<testsuites><testsuite tests="{gate.CALPHAD_TOOLS_TEST_COUNT}" '
        'failures="0" errors="0" skipped="0">'
        f"{tools_cases}</testsuite></testsuites>\n",
    )
    required_files = [
        {
            "path": "backend/controlplane/internal/domain/calphad.go",
            "sha256": "a" * 64,
            "size_bytes": 100,
        }
    ]
    required_materials = {
        "aggregate_sha256": gate.canonical_json_sha256(required_files),
        "file_count": len(required_files),
        "files": required_files,
    }
    frontend_files = [
        {
            "path": "frontend/dist/assets/app.js",
            "sha256": "b" * 64,
            "size_bytes": 200,
        },
        {
            "path": "frontend/dist/index.html",
            "sha256": "c" * 64,
            "size_bytes": 300,
        },
    ]
    release_artifacts = {
        "control_binary": {
            "path": "bin/ultra-control",
            "sha256": "d" * 64,
            "size_bytes": 4096,
        },
        "frontend_dist": {
            "path": "frontend/dist",
            "aggregate_sha256": gate.canonical_json_sha256(frontend_files),
            "file_count": len(frontend_files),
            "files": frontend_files,
        },
    }
    output = b"production parity passed\n"
    staged_sha = "6" * 64
    release_input_hashes = {
        relative: _sha(repository_root / relative)
        for relative in gate.REQUIRED_CALPHAD_RELEASE_INPUT_FILES
    }
    parity = {
        "schema_version": 1,
        "gate": "production-materials-sandbox-parity",
        "scope": "production-full",
        "claim": "full production DockerSandboxBackend image parity",
        "generated_at_utc": "2026-07-09T00:00:00Z",
        "expected_git_sha": GIT_SHA,
        "calphad_release_contract": {
            "manifest_sha256": manifest_sha,
            "release_input_sha256s": dict(sorted(release_input_hashes.items())),
            "runtime_test_count": gate.CALPHAD_RUNTIME_TEST_COUNT,
            "core_runtime_test_count": gate.CALPHAD_RUNTIME_CORE_TEST_COUNT,
            "typed_cli_test_count": gate.CALPHAD_RUNTIME_CLI_TEST_COUNT,
            "calphad_tools_test_count": gate.CALPHAD_TOOLS_TEST_COUNT,
            "required_adversarial_test_names": sorted(gate.REQUIRED_CALPHAD_ADVERSARIAL_TEST_NAMES),
        },
        "failures": [],
        "status": "passed",
        "full_production_image_parity": True,
        "source": {
            "kind": "git_archive_release_manifest",
            "expected_git_sha": GIT_SHA,
            "observed_git_sha": GIT_SHA,
            "manifest_path": "release-manifest.json",
            "manifest_sha256": "e" * 64,
            "required_materials": required_materials,
            "release_artifacts": release_artifacts,
            "tracked_worktree_clean": True,
            "staged_index_clean": True,
            "untracked_files_clean": True,
        },
        "verified_release_artifacts": release_artifacts,
        "base_image": {
            "ref": "ultra-production:release",
            "image_id": RUNTIME_IMAGE,
            "revision": GIT_SHA,
            "title": "Ultra Deep Agents scientific sandbox",
            "entrypoint": [],
        },
        "executed_image": {
            "ref": "ultra-production:release",
            "image_id": RUNTIME_IMAGE,
            "revision": GIT_SHA,
            "title": "Ultra Deep Agents scientific sandbox",
            "entrypoint": [],
            "entrypoint_adapter": False,
            "base_image_id": RUNTIME_IMAGE,
        },
        "sandbox": {
            "network": "none",
            "cpus": 8.0,
            "memory": "32g",
            "pids_limit": 4096,
            "shm_size": "8g",
            "timeout_seconds": 21600,
            "output_limit_bytes": 52_428_800,
            "gpus": "",
            "max_concurrency": 8,
            "no_new_privileges": True,
            "source": "exported_worker_environment",
            "backend": "DockerSandboxBackend",
            "network_none": True,
            "rootfs_read_only": True,
            "capabilities_dropped": True,
            "immutable_image_id": RUNTIME_IMAGE,
            "policy_source": "exported_worker_environment",
        },
        "execution": {
            "exit_code": 0,
            "truncated": False,
            "output_size_bytes": len(output),
            "output_sha256": hashlib.sha256(output).hexdigest(),
        },
        "staged_source": {
            "file_count": 1,
            "aggregate_sha256": staged_sha,
            "files": [
                {
                    "path": "scripts/materials_readiness_gate.py",
                    "sha256": staged_sha,
                    "size_bytes": 1,
                }
            ],
        },
        "required_domain_validators": list(gate.REQUIRED_DOMAIN_VALIDATORS),
        "domain_gate": {
            "relative_path": "domain/materials-domain-gate.json",
            "sha256": domain_sha,
            "report": embedded_domain,
        },
        "calphad_runtime": {
            "relative_path": "calphad-runtime-junit.xml",
            "sha256": runtime_sha,
            "junit": {
                "tests": gate.CALPHAD_RUNTIME_TEST_COUNT,
                "failures": 0,
                "errors": 0,
                "skipped": 0,
            },
            "required_core_test_names": list(gate.REQUIRED_CALPHAD_RUNTIME_CORE_TEST_NAMES),
            "required_typed_cli_test_names": list(gate.REQUIRED_CALPHAD_RUNTIME_CLI_TEST_NAMES),
        },
        "calphad_tool_orchestration": {
            "scope": "host-worker-runtime-orchestration-contract",
            "relative_path": "calphad-tools-junit.xml",
            "sha256": tools_sha,
            "junit": {
                "tests": gate.CALPHAD_TOOLS_TEST_COUNT,
                "failures": 0,
                "errors": 0,
                "skipped": 0,
            },
            "required_test_names": list(gate.REQUIRED_CALPHAD_TOOL_TEST_NAMES),
            "execution": {
                "runner": "uv-frozen-project-with-pytest-8.4.2",
                "exit_code": 0,
                "stdout_size_bytes": 0,
                "stdout_sha256": hashlib.sha256(b"").hexdigest(),
                "stderr_size_bytes": 0,
                "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            },
            "binding": {
                "git_sha": GIT_SHA,
                "runtime_image_id": RUNTIME_IMAGE,
                "source_kind": "git_archive_release_manifest",
                "release_artifacts": release_artifacts,
            },
        },
        "calphad": {
            "relative_path": "calphad-embedded-probe.json",
            "sha256": calphad_sha,
            "report": calphad,
        },
    }
    report_bytes = (
        json.dumps(parity, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n"
    ).encode()
    report_sha = hashlib.sha256(report_bytes).hexdigest()
    report_path = evidence_root / f"production-materials-sandbox-parity-{report_sha}.json"
    report_path.write_bytes(report_bytes)
    return parity, {
        "path": str(report_path.resolve()),
        "size_bytes": len(report_bytes),
        "sha256": report_sha,
    }


def _make_mattools_report(
    repository_root: Path,
    hashes: dict[str, str],
    benchmark: dict[str, Any],
    attestation: dict[str, Any],
) -> dict[str, Any]:
    host_lock_path = repository_root / "scripts/mattools-validator-requirements.lock.txt"
    host_input_path = repository_root / "scripts/mattools-validator-requirements.txt"
    host_resolved, host_lock_issues = gate._parse_hashed_requirements_lock(host_lock_path)
    assert host_lock_issues == []
    uv = shutil.which("uv")
    assert uv is not None
    uv_path = Path(uv).resolve()
    host_environment = {
        "schema_version": "1",
        "python_version": gate.EXPECTED_HOST_VALIDATOR_PYTHON,
        "python_implementation": gate.EXPECTED_HOST_VALIDATOR_IMPLEMENTATION,
        "python_executable_sha256": "7" * 64,
        "platform": "fixture-platform",
        "task_execution_performed": False,
        "required_packages": dict(gate.EXPECTED_HOST_VALIDATOR_PACKAGES),
        "resolved_packages": host_resolved,
        "resolved_packages_sha256": gate.canonical_json_sha256(host_resolved),
        "requirements_input_path": str(host_input_path),
        "requirements_input_sha256": hashes["validator_input"],
        "requirements_lock_path": str(host_lock_path),
        "requirements_lock_sha256": hashes["validator_lock"],
        "validator_command": [
            str(uv_path),
            "run",
            "--isolated",
            "--no-project",
            "--python",
            gate.EXPECTED_HOST_VALIDATOR_PYTHON,
            "--with-requirements",
            str(host_lock_path),
            "python",
        ],
    }
    evaluator_packages = {
        **gate.EXPECTED_EVALUATOR_PACKAGES,
        "fixture-transitive": "1.0",
    }
    evaluator_packages = dict(sorted(evaluator_packages.items()))
    evaluator_platform = {
        "docker": "linux/arm64",
        "machine": "aarch64",
        "python_implementation": "CPython",
        "system": "Linux",
    }
    evaluator_build = {
        "base_image": "python@sha256:" + "8" * 64,
        "dockerfile_path": "deploy/docker/mattools-evaluator.Dockerfile",
        "dockerfile_sha256": hashes["evaluator_dockerfile"],
        "adapted_requirements_sha256": "9" * 64,
        "supplemental_requirements_path": (
            "deploy/docker/mattools-evaluator-supplemental-requirements.txt"
        ),
        "supplemental_requirements_sha256": hashes["evaluator_supplemental"],
        "tool_source_file_count": 2756,
        "tool_source_manifest_sha256": "a" * 64,
        "candidate_fixture_file_count": gate.EXPECTED_CANDIDATE_FIXTURE_FILE_COUNT,
        "candidate_fixture_manifest_sha256": (gate.EXPECTED_CANDIDATE_FIXTURE_MANIFEST_SHA256),
        "candidate_visible_source_policy": gate.EXPECTED_CANDIDATE_VISIBLE_SOURCE_POLICY,
        "strict_shadow_path": "scripts/mattools_strict_shadow.py",
        "strict_shadow_sha256": hashes["shadow"],
        "safe_parser_path": "scripts/mattools_safe_parser.py",
        "safe_parser_sha256": hashes["safe_parser"],
        "runner_wrapper_path": "scripts/mattools_runner_wrapper.py",
        "runner_wrapper_sha256": hashes["runner_wrapper"],
        "semantic_repairs_path": "scripts/mattools_semantic_repairs.py",
        "semantic_repairs_sha256": hashes["semantic_repairs"],
        "builder_path": "scripts/build_mattools_evaluator.py",
        "builder_sha256": hashes["evaluator_builder"],
    }
    evaluator_upstream = {
        "revision": benchmark["revision"],
        "manifest_sha256": benchmark["sha256"],
        "dockerfile_python": "3.11.8",
        "project_python": ">=3.13,<4.0",
        "requirements_sha256": "b" * 64,
    }
    evaluator_lock_payload = {
        "schema_version": "1",
        "environment_kind": "reviewed-reconstruction-variant",
        "official_artifact": False,
        "variant_reason": "fixture reviewed reconstruction",
        "python_version": "3.11.8",
        "platform": evaluator_platform,
        "upstream": evaluator_upstream,
        "build": evaluator_build,
        "package_map_sha256": gate.canonical_json_sha256(evaluator_packages),
        "packages": evaluator_packages,
    }
    evaluator_lock_path = repository_root / gate.EXPECTED_EVALUATOR_LOCK_PATH
    evaluator_lock_sha = _write(
        evaluator_lock_path,
        json.dumps(evaluator_lock_payload, indent=2, sort_keys=True) + "\n",
    )
    approved_lock = {
        **evaluator_lock_payload,
        "path": gate.EXPECTED_EVALUATOR_LOCK_PATH,
        "sha256": evaluator_lock_sha,
        "approved_from_git_head": True,
    }
    evaluator_labels = {
        "io.ultra.mattools.adapted-requirements-sha256": evaluator_build[
            "adapted_requirements_sha256"
        ],
        "io.ultra.mattools.base-image": evaluator_build["base_image"],
        "io.ultra.mattools.environment-kind": evaluator_lock_payload["environment_kind"],
        "io.ultra.mattools.official-artifact": "false",
        "io.ultra.mattools.snapshot-manifest-sha256": evaluator_upstream["manifest_sha256"],
        "io.ultra.mattools.safe-parser-sha256": evaluator_build["safe_parser_sha256"],
        "io.ultra.mattools.runner-wrapper-sha256": evaluator_build["runner_wrapper_sha256"],
        "io.ultra.mattools.semantic-repairs-sha256": evaluator_build["semantic_repairs_sha256"],
        "io.ultra.mattools.strict-shadow-sha256": evaluator_build["strict_shadow_sha256"],
        "io.ultra.mattools.supplemental-requirements-sha256": evaluator_build[
            "supplemental_requirements_sha256"
        ],
        "io.ultra.mattools.target-platform": evaluator_platform["docker"],
        "io.ultra.mattools.tool-source-manifest-sha256": evaluator_build[
            "tool_source_manifest_sha256"
        ],
        "io.ultra.mattools.candidate-fixture-file-count": str(
            evaluator_build["candidate_fixture_file_count"]
        ),
        "io.ultra.mattools.candidate-fixture-manifest-sha256": evaluator_build[
            "candidate_fixture_manifest_sha256"
        ],
        "io.ultra.mattools.candidate-visible-source-policy": evaluator_build[
            "candidate_visible_source_policy"
        ],
        "io.ultra.mattools.upstream-requirements-sha256": evaluator_upstream["requirements_sha256"],
        "org.opencontainers.image.revision": evaluator_upstream["revision"],
    }
    trials: list[dict[str, Any]] = []
    for trial_number in range(1, 4):
        attempts = []
        for task in benchmark["tasks"]:
            run_id = f"run_{trial_number}_{task['ordinal']}"
            thread_id = f"thread_{trial_number}_{task['ordinal']}"
            worker_attestations, cleanroom_binding = _worker_cleanroom_records(
                run_id,
                thread_id,
            )
            published = {
                "classification": "strict_pass",
                "runnable": True,
                "scientific_pass": task["subtask_count"],
                "scientific_fail": 0,
            }
            strict = {
                "semantic_runnable": True,
                "strict_scientific_classification": "strict_pass",
                "strict_scientific_pass": task["subtask_count"],
                "strict_scientific_fail": 0,
                "strict_exact_ok": True,
                "raw_verifier_output_sha256": "d" * 64,
            }
            scoring_replays = [
                {
                    "replay": replay,
                    "replay_terminal_record_sha256": hashlib.sha256(
                        f"{trial_number}:{task['ordinal']}:{replay}".encode()
                    ).hexdigest(),
                    "published_upstream": published,
                    "strict_shadow": strict,
                }
                for replay in (1, 2)
            ]
            attempts.append(
                {
                    "task_id": task["task_id"],
                    "ordinal": task["ordinal"],
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "run_status": "succeeded",
                    "submission_status": "captured",
                    "artifact_ids": [f"artifact_{trial_number}_{task['ordinal']}"],
                    "solution_artifact_id": f"artifact_{trial_number}_{task['ordinal']}",
                    "code_sha256": "c" * 64,
                    "actual_runtime_provenance": {
                        "validated": True,
                        "model_observable": True,
                        "provider_observable": True,
                        "model_matches_declaration": True,
                        "provider_matches_declaration": True,
                        "observed_model_ids": [MODEL_ID],
                        "observed_provider_ids": [PROVIDER_ID],
                    },
                    "trace_summary": {
                        "production_execute_started_count": 1,
                        "production_execute_terminal_count": 1,
                        "production_execute_completed_count": 1,
                        "production_execute_tool_evidence": True,
                        "observed_execute_image_digests": [RUNTIME_IMAGE],
                        "server_cleanroom_profile_attested": True,
                        "server_evaluation_profiles": [gate.MATERIALS_CLEANROOM_PROFILE],
                        "worker_cleanroom_attestations": worker_attestations,
                    },
                    "cleanroom_binding": cleanroom_binding,
                    "evaluation": {
                        "task_id": task["task_id"],
                        "ordinal": task["ordinal"],
                        "classification": "strict_pass",
                        "runnable": True,
                        "scientific_pass": task["subtask_count"],
                        "scientific_fail": 0,
                    },
                    "scoring_evidence": {
                        "schema_version": "1",
                        "task_id": task["task_id"],
                        "ordinal": task["ordinal"],
                        "subtask_count": task["subtask_count"],
                        "expected_replay_count": 2,
                        "replay_count": 2,
                        "complete": True,
                        "replay_consistent": True,
                        "primary": scoring_replays[0],
                        "replays": scoring_replays,
                    },
                }
            )
        trials.append(
            {
                "trial": trial_number,
                "status": "complete",
                "runnable": 49,
                "published_runner_runnable": 49,
                "runnable_denominator": 49,
                "function_runnable_rate": 1.0,
                "scientific_pass": 138,
                "strict_scientific_pass": 138,
                "scientific_denominator": 138,
                "task_success_rate": 1.0,
                "strict_task_success_rate": 1.0,
                "replay_count": 2,
                "reproducible": True,
                "evaluator_environment": {
                    "image_id": EVALUATOR_IMAGE,
                    "production_runtime_image_digest": RUNTIME_IMAGE,
                    "independent_from_production_runtime": True,
                    "comparable": True,
                    "full_environment_lock_matches": True,
                    "resolved_environment_sha256": gate.canonical_json_sha256(evaluator_packages),
                    "resolved_packages": evaluator_packages,
                    "packages": dict(gate.EXPECTED_EVALUATOR_PACKAGES),
                    "required_packages": dict(gate.EXPECTED_EVALUATOR_PACKAGES),
                    "approved_environment_lock": approved_lock,
                    "environment_kind": "reviewed-reconstruction-variant",
                    "official_artifact": False,
                    "python_version": "3.11.8",
                    "platform": evaluator_platform,
                    "task_execution_performed": False,
                    "labels_match_approved_lock": True,
                    "embedded_inputs_match_approved_lock": True,
                    "platform_matches_approved_lock": True,
                    "image_labels": dict(sorted(evaluator_labels.items())),
                    "embedded_inputs": {
                        "candidate_fixture_file_count": evaluator_build[
                            "candidate_fixture_file_count"
                        ],
                        "candidate_fixture_manifest_sha256": evaluator_build[
                            "candidate_fixture_manifest_sha256"
                        ],
                        "candidate_visible_non_fixture_paths": [],
                        "candidate_visible_executable_source_paths": [],
                        "candidate_visible_dependency_test_paths": {
                            "pymatgen": [],
                            "pymatgen-analysis-defects": [],
                        },
                        "upstream_requirements_sha256": evaluator_upstream["requirements_sha256"],
                        "adapted_requirements_sha256": evaluator_build[
                            "adapted_requirements_sha256"
                        ],
                        "supplemental_requirements_sha256": evaluator_build[
                            "supplemental_requirements_sha256"
                        ],
                    },
                },
                "sandbox_policy_attestation": copy.deepcopy(attestation),
                "runner": {
                    "host_validator_command": host_environment["validator_command"],
                    "host_validator_executable_sha256": _sha(uv_path),
                    "host_requirements_sha256": hashes["validator_lock"],
                    "host_input_requirements_sha256": hashes["validator_input"],
                    "host_validator_environment": host_environment,
                },
                "attempts": attempts,
            }
        )

    required_gates = {
        name: True
        for name in (
            "official_snapshot",
            "license_attested",
            "checkpoint_evidence_integrity",
            "checkpoint_non_erasure_integrity",
            "replay_terminal_evidence_integrity",
            "three_trial_completeness",
            "actual_ultra_control_plane_path",
            "actual_model_provider_provenance",
            "production_execute_tool_evidence",
            "server_authorized_cleanroom_profile",
            "worker_enforced_cleanroom_profile",
            "production_execute_runtime_image_attestation",
            "required_solution_artifacts",
            "expected_values_and_verifiers_isolated",
            "official_evaluator_environment_exact",
            "evaluator_independent_from_production",
            "immediate_replay_reproducible",
            "external_sandbox_isolation_evidence",
            "provenance_complete",
            "ultra_worktree_clean",
            "per_trial_mattools_function_runnable",
            "per_trial_strict_scientific_task_success",
            "mattools_function_runnable",
            "mattools_task_success",
            "strict_scientific_task_success",
        )
    }
    return {
        "schema_version": "1",
        "generated_at": "2026-07-09T00:00:00Z",
        "campaign_id": "fixture-campaign",
        "benchmark": benchmark,
        "ultra": {
            "commit": GIT_SHA,
            "dirty": False,
            "skills_sha256": hashes["skills"],
            "skills_file_count": 1,
        },
        "harness": {
            "path": "scripts/mattools_promotion_gate.py",
            "sha256": hashes["harness"],
            "host_validator_requirements_path": (
                "scripts/mattools-validator-requirements.lock.txt"
            ),
            "host_validator_requirements_sha256": hashes["validator_lock"],
            "host_validator_input_requirements_path": (
                "scripts/mattools-validator-requirements.txt"
            ),
            "host_validator_input_requirements_sha256": hashes["validator_input"],
            "host_validator_environment": host_environment,
            "strict_shadow_path": "scripts/mattools_strict_shadow.py",
            "strict_shadow_sha256": hashes["shadow"],
            "semantic_repairs_path": "scripts/mattools_semantic_repairs.py",
            "semantic_repairs_sha256": hashes["semantic_repairs"],
        },
        "runtime_environment": {
            "image_digest": RUNTIME_IMAGE,
            "operator_declared_model_id": MODEL_ID,
            "operator_declared_provider_id": PROVIDER_ID,
            "observed_model_ids": [MODEL_ID],
            "observed_provider_ids": [PROVIDER_ID],
            "actual_model_provider_provenance_validated": True,
            "evaluation_profile": gate.MATERIALS_CLEANROOM_PROFILE,
        },
        "official_evaluator_environment": {
            "required_packages": dict(gate.EXPECTED_EVALUATOR_PACKAGES),
            "source_revision": benchmark["revision"],
            "approved_lock": approved_lock,
            "observed_trials": [trial["evaluator_environment"] for trial in trials],
        },
        "license_attestation": {
            "accepted": True,
            "use_basis": "noncommercial",
            "use_purpose": "authorized non-commercial internal platform evaluation",
            "repository_license": "Apache-2.0",
            "dataset_card_license": "CC-BY-NC-4.0",
            "separate_license_evidence_sha256": None,
            "attested_at": "2026-07-09T00:00:00Z",
        },
        "checkpoint_evidence_audit": {
            "valid": True,
            "issues": [],
            "verified_attempt_count": 147,
            "recomputed_replay_count": 6,
            "audited_at": "2026-07-09T00:00:00Z",
            "trusted_state_booleans": False,
            "terminal_attempt_directory_exact": True,
            "terminal_replay_directory_exact": True,
            "expected_attempt_count": 147,
            "actual_attempt_count": 147,
            "attempt_key_set_exact": True,
            "terminal_attempts_non_replaced": True,
            "verified_replay_terminal_record_count": 6,
            "expected_replay_terminal_record_count": 6,
            "terminal_replays_non_replaced": True,
            "failed_replay_terminal_record_count": 0,
        },
        "trials": trials,
        "counts": {
            "runnable": 147,
            "runnable_denominator": 147,
            "runnable_minimum": 118,
            "per_trial_runnable_minimum": 40,
            "scientific_pass": 414,
            "scientific_denominator": 414,
            "scientific_minimum": 249,
            "per_trial_scientific_minimum": 83,
            "strict_scientific_pass": 414,
            "terminal_attempts": 147,
            "expected_attempts_for_configured_run": 147,
        },
        "rates": {
            "function_runnable": 1.0,
            "task_success": 1.0,
            "strict_task_success": 1.0,
        },
        "hard_gates": required_gates,
        "promotion": {
            "scope": "MatTools benchmark lane only",
            "passed": True,
            "full_materials_production_ready": False,
            "reasons": [],
        },
    }


def _make_live_trace(tmp_path: Path) -> dict[str, Any]:
    module_name = "_materials_validation_fixture"
    module_path = ROOT / "backend/deepagents_runtime/src/ultra_deepagents/materials/validation.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    validation_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = validation_module
    try:
        spec.loader.exec_module(validation_module)
        check = validation_module.ValidationCheck(
            validator_id="materials.trace.retained.v1",
            outcome=validation_module.ValidationOutcome.PASS,
            observed={"value": 1.0},
            expected={"value": 1.0, "absolute_tolerance": 0.0},
            units="dimensionless",
            tolerance_rationale="exact retained-artifact fixture",
            required=True,
            critical=True,
            library_versions={"fixture": "1"},
            evidence=(
                validation_module.EvidenceArtifact(
                    name="result.json",
                    sha256="d" * 64,
                    artifact_id="artifact_result",
                    path="/outputs/result.json",
                    size_bytes=123,
                ),
            ),
        )
        assessment = validation_module.assess_scientific_status(
            run_status="succeeded",
            checks=(check,),
            required_validator_ids=("materials.trace.retained.v1",),
        )
        validation_bytes = validation_module.canonical_record_json(assessment).encode("utf-8")
    finally:
        sys.modules.pop(module_name, None)
    validation_sha = hashlib.sha256(validation_bytes).hexdigest()
    tmp_path.mkdir(parents=True, exist_ok=True)
    retained_path = tmp_path / f"materials-validation-{validation_sha}.json"
    retained_path.write_bytes(validation_bytes)
    run_id = "run_trace_1"
    return {
        "thread_id": "thread_trace_1",
        "prompt": {
            "run_id": run_id,
            "thread_id": "thread_trace_1",
            "status": "succeeded",
            "tool_names": ["read_file", "execute", "write_file"],
            "remote_mutation_intents": [],
            "remote_mutation_scope_valid": True,
            "materials_validation": {
                "artifact_id": "artifact_validation",
                "canonical_sha256": validation_sha,
                "record_sha256": validation_sha,
                "size_bytes": len(validation_bytes),
                "retained_path": str(retained_path),
                "retained_sha256": validation_sha,
                "retained_size_bytes": len(validation_bytes),
                "durable_path": "outputs/materials_validation.json",
                "path": "materials_validation.json",
                "valid": True,
                "verified": True,
                "evidence_verified": True,
                "run_status": "succeeded",
                "scientific_status": "verified",
                "silent_success": False,
            },
            "artifacts": [
                {
                    "artifact_id": "artifact_validation",
                    "run_id": run_id,
                    "path": "materials_validation.json",
                    "tool_name": "outputs_collector",
                    "download_ok": True,
                    "sha256": validation_sha,
                    "size_bytes": len(validation_bytes),
                },
                {
                    "artifact_id": "artifact_result",
                    "run_id": run_id,
                    "path": "result.json",
                    "tool_name": "execute",
                    "download_ok": True,
                    "sha256": "d" * 64,
                    "size_bytes": 123,
                },
            ],
        },
        "materials_quality": {
            "passed": True,
            "issues": [],
            "score": 10.0,
            "quality_scope": "trace_and_first_party_validation_record",
            "independent_scientific_verification": False,
            "scientific_conclusion_verified": False,
            "signals": {
                "code_execution": True,
                "durable_validation_artifact": True,
                "hashed_evidence": True,
                "materials_skill": True,
                "no_silent_success": True,
                "remote_mutation_aligned": True,
                "remote_mutation_scope_valid": True,
                "first_party_scientific_record_valid": True,
                "terminal_ok": True,
                "validation_present": True,
                "validation_valid": True,
            },
        },
    }


def _make_calphad_ledger_report(repository_root: Path, evidence_root: Path) -> dict[str, Any]:
    files = []
    hashes: dict[str, str] = {}
    for relative in gate.REQUIRED_CALPHAD_LEDGER_SOURCE_FILES:
        path = repository_root / relative
        digest = _sha(path)
        hashes[relative] = digest
        files.append(
            {
                "path": relative,
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
    events = [
        {
            "Action": "pass",
            "Test": name,
            "Package": gate.CALPHAD_LEDGER_TEST_PACKAGES[name],
        }
        for name in gate.REQUIRED_CALPHAD_LEDGER_TESTS
    ]
    events.extend(
        {
            "Action": "pass",
            "Test": test_name,
            "Package": gate.CALPHAD_LEDGER_TEST_PACKAGES[gate.CALPHAD_POSTGRES_TEST],
        }
        for tests in gate.CALPHAD_POSTGRES_INVARIANT_TEST_EVIDENCE.values()
        for test_name in tests
        if "/" in test_name
    )
    observed_database = {
        "database": "ultra_qualification",
        "server_address": "127.0.0.1",
        "server_port": 5432,
        "role": "ultra_qualification_role",
        "transaction_read_only": "off",
        "role_superuser": False,
        "role_create_role": False,
        "role_create_database": False,
        "role_replication": False,
        "role_bypass_rls": False,
        "calphad_owned_tables": [],
        "calphad_owned_functions": [],
        "calphad_owner_roles": ["ultra_qualification_migration"],
        "calphad_reachable_roles": [],
        "calphad_owner_role_reachable": False,
        "public_schema_owner": "pg_database_owner",
        "public_owner_role_reachable": False,
        "can_create_public_schema": False,
        "calphad_select_all": True,
        "calphad_insert_all": False,
        "calphad_insert_any": False,
        "calphad_execute_create_revision": True,
        "calphad_execute_append_validation": True,
        "calphad_writer_functions_exact": True,
        "calphad_execute_unexpected_writer": False,
        "calphad_execute_internal": False,
        "calphad_public_execute": False,
        "calphad_unexpected_table_acl_grantees": [],
        "calphad_unexpected_function_acl_grantees": [],
        "calphad_mutation_privilege": False,
        "connection_target_host": "postgres",
        "connection_target_port": 5432,
    }
    events.append(
        {
            "Action": "output",
            "Test": gate.CALPHAD_POSTGRES_TEST,
            "Package": gate.CALPHAD_LEDGER_TEST_PACKAGES[gate.CALPHAD_POSTGRES_TEST],
            "Output": gate.CALPHAD_POSTGRES_IDENTITY_MARKER
            + json.dumps(observed_database, sort_keys=True),
        }
    )
    log_payload = "".join(json.dumps(event) + "\n" for event in events)
    log_digest = hashlib.sha256(log_payload.encode()).hexdigest()
    log_path = evidence_root / f"calphad-ledger-go-test-{log_digest}.jsonl"
    _write(log_path, log_payload)
    recomputed = gate._calphad_go_test_log_evidence(log_path)
    assert recomputed["valid"] is True, recomputed["issues"]
    return {
        "schema_version": "1",
        "gate": "calphad-ledger-postgres-qualification",
        "status": "passed",
        "qualification_database": True,
        "production_database_used": False,
        "database": {
            "scheme": "postgresql",
            "host": "postgres",
            "port": 5432,
            "database": "ultra_qualification",
            "serving_role": "ultra_qualification_role",
            "migration_role": "ultra_qualification_migration",
            "credentials_recorded": False,
        },
        "observed_database": observed_database,
        "git_sha": GIT_SHA,
        "repository_clean": True,
        "failures": [],
        "source_manifest": {
            "file_count": len(files),
            "aggregate_sha256": _manifest_hash(hashes),
            "files": files,
        },
        "tests": recomputed["records"],
        "summary": {
            "passed": len(gate.REQUIRED_CALPHAD_LEDGER_TESTS),
            "failed": 0,
            "skipped": 0,
        },
        "postgres_invariants": recomputed["invariant_outcomes"],
        "postgres_invariant_evidence": recomputed["invariant_records"],
        "runner": {
            "command": list(gate.CALPHAD_LEDGER_GO_COMMAND),
            "database_credentials_recorded": False,
            "go_test_log": {
                "path": log_path.name,
                "sha256": log_digest,
                "size_bytes": len(log_payload.encode()),
            },
        },
    }


def _make_calphad_cross_language_bundle(
    repository_root: Path,
    evidence_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    resource_id = "calphad-cross-language-" + GIT_SHA[:20]
    database_format = "tdb"
    database_payload = b"$ CALPHAD readiness retained-input fixture\n"
    database_sha = hashlib.sha256(database_payload).hexdigest()
    database_size = len(database_payload)
    database_path = evidence_root / "artifacts" / "database" / f"{database_sha}.{database_format}"
    database_path.parent.mkdir(parents=True, exist_ok=True)
    database_path.write_bytes(database_payload)
    database_input = {
        "path": database_path.relative_to(evidence_root).as_posix(),
        "sha256": database_sha,
        "size_bytes": database_size,
        "format": database_format,
    }
    database_manifest_sha = "b" * 64
    binding = {
        "kind": "resource",
        "database_id": "nist-al-co-w-wang-2017",
        "resource_id": resource_id,
        "database_format": database_format,
        "sha256": database_sha,
        "size_bytes": database_size,
        "source": "https://materialsdata.nist.gov/handle/11256/948",
        "license_id": "CC0-1.0",
        "assessment_scope": "Published Al-Co-W assessment",
        "reference_state": "SER",
        "temperature_limits_K": [300.0, 2000.0],
        "assessment_pressure_limits_Pa": [101325.0, 101325.0],
        "binding_schema": "ultra.selected_resource.v1",
        "binding_authority": "control_resource_catalog",
        "declaration_authority": "resource_owner",
    }
    database_result = {
        "schema_version": "1",
        "path": f"/workspace/.ultra/calphad/staged/{database_sha}.{database_format}",
        "name": f"{database_sha}.{database_format}",
        "format": database_format,
        "sha256": database_sha,
        "size_bytes": database_size,
        "pycalphad_version": "0.11.2",
        "manifest_sha256": database_manifest_sha,
    }
    execution_contract = {
        "interface": "fixed ultra_deepagents.materials.calphad public surface",
        "caller_code_accepted": False,
        "caller_models_or_solver_options_accepted": False,
        "network": "none",
        "no_new_privileges": True,
        "read_only_root_filesystem": True,
        "cap_drop_all": True,
        "cpus_at_most": 8.0,
        "memory_bytes_at_most": 32 * 1024**3,
        "pids_at_most": 4096,
        "runtime_image_id": RUNTIME_IMAGE,
        "max_components": 32,
        "max_phases": 128,
        "max_axis_values": 64,
        "max_grid_points": 256,
        "wall_time_seconds": 30.0,
        "max_result_bytes": 16 * 1024 * 1024,
    }
    persistence_contract = {
        "catalog_status": "pending",
        "catalog_metadata_updated": False,
        "mode": "immutable_per_run_evidence",
        "note": "server callback pending",
    }

    def artifact(operation: str, *, inspection_sha: str = "") -> dict[str, Any]:
        request: dict[str, Any] = {
            "operation": operation,
            "runtime_image_id": RUNTIME_IMAGE,
            "selection": {"components": None, "phases": None},
        }
        if operation == "equilibrium":
            request["inspection_artifact_sha256"] = inspection_sha
            request["selection"] = {
                "components": ["AL", "CO", "W", "VA"],
                "phases": ["BCC_B2", "FCC_A1", "LIQUID"],
            }
            request["conditions"] = {
                "temperatures_K": [1173.0],
                "pressures_Pa": [101325.0],
                "independent_compositions": {"AL": [0.675], "CO": [0.26]},
            }
        result: dict[str, Any] = dict(database_result)
        if operation == "equilibrium":
            result = {
                "schema_version": "ultra.calphad.equilibrium.v2",
                "database": dict(database_result),
                "request": {},
                "result": {},
                "warnings": [],
                "evidence": {},
            }
        payload = {
            "schema_version": "ultra.calphad.tool-evidence.v3",
            "operation": operation,
            "database_binding": binding,
            "request": request,
            "result": result,
            "execution_contract": execution_contract,
            "validation_persistence": persistence_contract,
        }
        encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        digest = hashlib.sha256(encoded).hexdigest()
        directory = "inspection" if operation == "inspect" else "equilibrium"
        path = evidence_root / "artifacts" / directory / f"{digest}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
        return {
            "path": path.relative_to(evidence_root).as_posix(),
            "sha256": digest,
            "size_bytes": len(encoded),
        }

    inspection = artifact("inspect")
    equilibrium = artifact("equilibrium", inspection_sha=inspection["sha256"])
    equilibrium["inspection_artifact_sha256"] = inspection["sha256"]
    source_manifest = []
    for relative in gate.REQUIRED_CALPHAD_CROSS_LANGUAGE_SOURCE_FILES:
        path = repository_root / relative
        source_manifest.append(
            {"path": relative, "sha256": _sha(path), "size_bytes": path.stat().st_size}
        )
    inspect_request_sha = "e" * 64
    equilibrium_request_sha = "f" * 64
    inventory_sha = "c" * 64
    backend_marker = {
        "schema_version": "ultra.calphad.cross-language-qualification.v1",
        "live_http_callback": True,
        "live_postgres": True,
        "database": {
            "name": "ultra_qualification",
            "server_address": "127.0.0.1",
            "server_port": 5432,
            "connection_target_host": "postgres",
            "connection_target_port": 5432,
            "transaction_read_only": "off",
            "serving_role": "ultra_qualification_role",
            "migration_role": "ultra_qualification_migration",
            "serving_role_superuser": False,
            "serving_role_create_role": False,
            "serving_role_create_database": False,
            "serving_role_replication": False,
            "serving_role_bypass_rls": False,
            "serving_role_owned_tables": [],
            "serving_role_owned_functions": [],
            "calphad_owner_roles": ["ultra_qualification_migration"],
            "calphad_reachable_roles": [],
            "calphad_owner_role_reachable": False,
            "public_schema_owner": "pg_database_owner",
            "public_owner_role_reachable": False,
            "can_create_public_schema": False,
            "serving_role_select_all": True,
            "serving_role_insert_all": False,
            "serving_role_insert_any": False,
            "serving_role_execute_create_revision": True,
            "serving_role_execute_append_validation": True,
            "serving_writer_functions_exact": True,
            "serving_execute_unexpected_writer": False,
            "serving_role_execute_internal": False,
            "serving_role_public_execute": False,
            "serving_unexpected_table_acl_grantees": [],
            "serving_unexpected_function_acl_grantees": [],
            "serving_role_mutation_privilege": False,
        },
        "resource_id": resource_id,
        "revision_id": "calphad-revision-fixture",
        "run_id": "calphad-run-fixture",
        "runtime_image_id": RUNTIME_IMAGE,
        "pycalphad_version": "0.11.2",
        "database_sha256": database_sha,
        "database_size_bytes": database_size,
        "database_format": database_format,
        "assessment_pressure_limits_Pa": [101325.0, 101325.0],
        "database_inventory_sha256": inventory_sha,
        "inspect": {
            "evidence_sha256": inspection["sha256"],
            "evidence_size_bytes": inspection["size_bytes"],
            "request_sha256": inspect_request_sha,
            "evidence_retention": "retained",
            "promotable": True,
            "postgres_bytes_exact": True,
        },
        "equilibrium": {
            "evidence_sha256": equilibrium["sha256"],
            "evidence_size_bytes": equilibrium["size_bytes"],
            "request_sha256": equilibrium_request_sha,
            "inspection_evidence_sha256": inspection["sha256"],
            "evidence_retention": "retained",
            "promotable": True,
            "postgres_bytes_exact": True,
        },
    }
    go_package = "github.com/amilworks/bisque-ultra/backend/controlplane/integration"
    go_test = "TestCalphadTypedCLIHTTPPostgresQualification"
    go_events = [
        {
            "Action": "output",
            "Test": go_test,
            "Package": go_package,
            "Output": "CALPHAD_CROSS_LANGUAGE_EVIDENCE "
            + json.dumps(backend_marker, sort_keys=True, separators=(",", ":"))
            + "\n",
        },
        {"Action": "pass", "Test": go_test, "Package": go_package},
        {"Action": "pass", "Package": go_package},
    ]
    go_log_payload = "".join(json.dumps(event) + "\n" for event in go_events).encode()
    go_log_digest = hashlib.sha256(go_log_payload).hexdigest()
    go_log_path = evidence_root / f"go-test-{go_log_digest}.jsonl"
    go_log_path.parent.mkdir(parents=True, exist_ok=True)
    go_log_path.write_bytes(go_log_payload)
    image_inspect_payload = json.dumps(
        [
            {
                "Id": RUNTIME_IMAGE,
                "Config": {
                    "Labels": {
                        "org.opencontainers.image.title": "Ultra Deep Agents scientific sandbox",
                        "org.opencontainers.image.revision": GIT_SHA,
                    },
                    "Env": ["PYTHONPATH=/opt/ultra-runtime"],
                },
            }
        ],
        sort_keys=True,
    ).encode()
    image_inspect_digest = hashlib.sha256(image_inspect_payload).hexdigest()
    image_inspect_path = evidence_root / f"docker-image-inspect-{image_inspect_digest}.json"
    image_inspect_path.write_bytes(image_inspect_payload)
    image_inspect_record = {
        "path": image_inspect_path.name,
        "sha256": image_inspect_digest,
        "size_bytes": len(image_inspect_payload),
    }
    report = {
        "schema_version": "ultra.calphad.cross-language-gate.v1",
        "gate": "calphad-typed-cli-http-postgres-cross-language",
        "generated_at_utc": "2026-07-10T00:00:00Z",
        "expected_git_sha": GIT_SHA,
        "repository": {"head_sha": GIT_SHA, "clean": True},
        "source_manifest": source_manifest,
        "generation": {
            "mode": "pinned_image",
            "runtime_identity_kind": "immutable_oci_image",
            "image_ref": "ultra-runtime:fixture",
            "runtime_image_id": RUNTIME_IMAGE,
            "image_title": "Ultra Deep Agents scientific sandbox",
            "image_revision": GIT_SHA,
            "pythonpath": "/opt/ultra-runtime",
            "image_inspected": True,
            "docker_image_inspect": image_inspect_record,
            "pycalphad_version": "0.11.2",
            "sandbox_policy": {
                "enforced_by_gate": True,
                "network": "none",
                "read_only_root_filesystem": True,
                "no_new_privileges": True,
                "cap_drop_all": True,
                "cpus_at_most": 8,
                "memory_bytes_at_most": 32 * 1024**3,
                "pids_at_most": 4096,
            },
        },
        "resource": {
            "resource_id": resource_id,
            "database_id": "nist-al-co-w-wang-2017",
            "database_sha256": database_sha,
            "database_size_bytes": database_size,
            "database_format": database_format,
            "assessment_pressure_limits_Pa": [101325.0, 101325.0],
            "license_id": "CC0-1.0",
            "source": "https://materialsdata.nist.gov/handle/11256/948",
        },
        "typed_cli_artifacts": {
            "database_input": database_input,
            "inspect": inspection,
            "equilibrium": equilibrium,
        },
        "backend": {
            "command": [
                "go",
                "test",
                "-json",
                "-count=1",
                "./integration",
                "-run",
                "^TestCalphadTypedCLIHTTPPostgresQualification$",
            ],
            "test": {
                "name": go_test,
                "package": go_package,
                "action": "pass",
            },
            "go_test_log": {
                "path": go_log_path.name,
                "sha256": go_log_digest,
                "size_bytes": len(go_log_payload),
            },
            **backend_marker,
        },
        "checks": {name: True for name in gate.REQUIRED_CALPHAD_CROSS_LANGUAGE_CHECKS},
        "production_live_qualified": True,
        "promotable": True,
        "status": "qualified",
    }
    report_payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    report_digest = hashlib.sha256(report_payload).hexdigest()
    report_path = evidence_root / f"calphad-cross-language-qualification-{report_digest}.json"
    report_path.write_bytes(report_payload)
    report_meta = {
        "path": str(report_path.resolve()),
        "sha256": report_digest,
        "size_bytes": len(report_payload),
    }
    manifest = {
        "schema_version": "ultra.calphad.cross-language-report-manifest.v1",
        "report": {**report_meta, "path": report_path.name},
        "production_live_qualified": True,
        "runtime_image_id": RUNTIME_IMAGE,
        "expected_git_sha": GIT_SHA,
    }
    manifest_path = evidence_root / "report_manifest.json"
    manifest_meta = _write_json_report(manifest_path, manifest)
    return report, report_meta, manifest, manifest_meta


def _write_json_report(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    digest = _write(path, content)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": digest,
    }


def _rewrite_calphad_ledger_report(valid_inputs: dict[str, Any]) -> None:
    report = valid_inputs["calphad_ledger"]
    old_meta = valid_inputs["input_metadata"]["calphad_ledger_report"]
    root = Path(old_meta["path"]).parent
    content = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(content).hexdigest()
    valid_inputs["input_metadata"]["calphad_ledger_report"] = _write_json_report(
        root / f"calphad-ledger-postgres-qualification-{digest}.json",
        report,
    )


@pytest.fixture(autouse=True)
def _verified_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "_verify_detached_signature", lambda _record: (True, None))
    monkeypatch.setattr(gate, "_public_key_git_anchored", lambda _path, _root: True)
    monkeypatch.setattr(
        gate,
        "_run_calphad_real_http_revalidation",
        lambda *_args, **_kwargs: {
            "valid": True,
            "command": list(gate.CALPHAD_REAL_HTTP_REVALIDATION_COMMAND),
            "test": gate.CALPHAD_REAL_HTTP_REVALIDATION_TEST,
            "package": gate.CALPHAD_REAL_HTTP_REVALIDATION_PACKAGE,
            "exit_code": 0,
            "stdout_sha256": "a" * 64,
            "stderr_sha256": "b" * 64,
            "issues": [],
        },
    )

    def no_task_probe(command: list[str], _snapshot: Path) -> dict[str, Any]:
        resolved, issues = gate._parse_hashed_requirements_lock(Path(command[7]))
        assert issues == []
        return {
            "schema_version": "1",
            "python_version": gate.EXPECTED_HOST_VALIDATOR_PYTHON,
            "python_implementation": gate.EXPECTED_HOST_VALIDATOR_IMPLEMENTATION,
            "python_executable_sha256": "7" * 64,
            "platform": "fixture-platform",
            "task_execution_performed": False,
            "required_packages": dict(gate.EXPECTED_HOST_VALIDATOR_PACKAGES),
            "resolved_packages": dict(sorted(resolved.items())),
        }

    monkeypatch.setattr(gate, "_run_host_validator_no_task_probe", no_task_probe)

    def report_bundle(
        mattools: dict[str, Any],
        _repository_root: Path,
        _benchmark_root: Path,
        _manifest_path: Path,
    ) -> dict[str, Any]:
        return {
            "schema_version": "1",
            "revalidation_kind": "ultra.mattools.report_revalidation.v1",
            "valid": True,
            "bundle_exact": True,
            "manifest_integrity_valid": True,
            "checkpoint_evidence_valid": True,
            "checkpoint_exact": True,
            "results_json_exact": True,
            "results_markdown_exact": True,
            "manifest_exact": True,
            "task_execution_performed": False,
            "promotion_passed": mattools["promotion"]["passed"],
            "checkpoint_evidence_audit": mattools["checkpoint_evidence_audit"],
            "process_exit_code": 0,
            "issues": [],
        }

    monkeypatch.setattr(gate, "_revalidate_mattools_report_bundle", report_bundle)
    monkeypatch.setattr(
        gate,
        "_production_bundle_evidence",
        lambda _report, _path, _root: {
            "valid": True,
            "source_bound": True,
            "validator_failures": [],
            "issues": [],
        },
    )


@pytest.fixture
def valid_inputs(tmp_path: Path) -> dict[str, Any]:
    repository_root = tmp_path / "ultra"
    benchmark_root = tmp_path / "MatTools"
    hashes = _make_repository(repository_root)
    benchmark, policy = _make_benchmark(benchmark_root)
    attestation = _make_isolation_attestation(tmp_path / "attestation")
    domain = _make_domain_report(repository_root, hashes)
    production_parity, production_parity_meta = _make_production_parity_report(
        tmp_path, domain, repository_root
    )
    mattools = _make_mattools_report(repository_root, hashes, benchmark, attestation)
    calphad_ledger_root = tmp_path / "calphad-ledger"
    calphad_ledger = _make_calphad_ledger_report(repository_root, calphad_ledger_root)
    calphad_ledger_payload = (json.dumps(calphad_ledger, indent=2, sort_keys=True) + "\n").encode()
    calphad_ledger_digest = hashlib.sha256(calphad_ledger_payload).hexdigest()
    calphad_ledger_meta = _write_json_report(
        calphad_ledger_root / f"calphad-ledger-postgres-qualification-{calphad_ledger_digest}.json",
        calphad_ledger,
    )
    cross_language_root = tmp_path / "calphad-cross-language"
    (
        calphad_cross_language,
        calphad_cross_language_meta,
        calphad_cross_language_manifest,
        calphad_cross_language_manifest_meta,
    ) = _make_calphad_cross_language_bundle(repository_root, cross_language_root)
    campaign_root = tmp_path / "mattools-campaign"
    mattools_meta = _write_json_report(campaign_root / "results.json", mattools)
    markdown_path = campaign_root / "results.md"
    checkpoint_path = campaign_root / "state.json"
    _write(markdown_path, "# MatTools fixture\n")
    _write(checkpoint_path, '{"schema_version":"1"}\n')
    mattools_manifest = {
        "schema_version": "2",
        "manifest_kind": "ultra.mattools.report_bundle.v2",
        "generated_at": "2026-07-09T00:00:00Z",
        "campaign_id": mattools["campaign_id"],
        "benchmark_sha256": mattools["benchmark"]["sha256"],
        "checkpoint_evidence_audit_sha256": gate.canonical_json_sha256(
            mattools["checkpoint_evidence_audit"]
        ),
        "regeneration": {
            "helper": "revalidate_report_bundle",
            "cli_subcommand": "verify-report",
            "comparison": "byte_exact",
            "task_execution_performed": False,
        },
        "results_json": {
            "path": str((campaign_root / "results.json").resolve()),
            "sha256": mattools_meta["sha256"],
        },
        "results_markdown": {
            "path": str(markdown_path.resolve()),
            "sha256": _sha(markdown_path),
        },
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": _sha(checkpoint_path),
        },
    }
    mattools_manifest_meta = _write_json_report(
        campaign_root / "report_manifest.json", mattools_manifest
    )
    return {
        "domain": domain,
        "production_parity": production_parity,
        "calphad_ledger": calphad_ledger,
        "calphad_cross_language": calphad_cross_language,
        "calphad_cross_language_manifest": calphad_cross_language_manifest,
        "mattools": mattools,
        "mattools_manifest": mattools_manifest,
        "traces": [_make_live_trace(tmp_path / "live-trace-evidence")],
        "repository_root": repository_root,
        "benchmark_root": benchmark_root,
        "policy": policy,
        "expected": gate.ExpectedProvenance(
            git_sha=GIT_SHA,
            domain_image=DOMAIN_IMAGE,
            runtime_image=RUNTIME_IMAGE,
            evaluator_image=EVALUATOR_IMAGE,
        ),
        "repository_state": {"commit": GIT_SHA, "dirty": False},
        "benchmark_state": {
            "revision": policy.official_revision,
            "dirty": False,
            "inspection_ok": True,
            "tracked_files": sorted(benchmark["tracked_file_hashes"]),
        },
        "input_metadata": {
            "production_parity_report": production_parity_meta,
            "calphad_ledger_report": calphad_ledger_meta,
            "calphad_cross_language_report": calphad_cross_language_meta,
            "calphad_cross_language_report_manifest": calphad_cross_language_manifest_meta,
            "mattools_report": mattools_meta,
            "mattools_report_manifest": mattools_manifest_meta,
        },
        "mattools_results_path": campaign_root / "results.json",
        "mattools_manifest_path": campaign_root / "report_manifest.json",
    }


def _evaluate(inputs: dict[str, Any]) -> dict[str, Any]:
    inputs["input_metadata"]["mattools_report"] = _write_json_report(
        inputs["mattools_results_path"], inputs["mattools"]
    )
    inputs["mattools_manifest"]["results_json"]["sha256"] = inputs["input_metadata"][
        "mattools_report"
    ]["sha256"]
    inputs["input_metadata"]["mattools_report_manifest"] = _write_json_report(
        inputs["mattools_manifest_path"], inputs["mattools_manifest"]
    )
    return gate.evaluate_readiness(
        deterministic_report=inputs["domain"],
        production_parity_report=inputs["production_parity"],
        calphad_ledger_report=inputs["calphad_ledger"],
        calphad_cross_language_report=inputs["calphad_cross_language"],
        calphad_cross_language_report_manifest=inputs["calphad_cross_language_manifest"],
        mattools_report=inputs["mattools"],
        mattools_report_manifest=inputs["mattools_manifest"],
        live_trace_reports=inputs["traces"],
        repository_root=inputs["repository_root"],
        benchmark_root=inputs["benchmark_root"],
        expected=inputs["expected"],
        policy=inputs["policy"],
        repository_state=inputs["repository_state"],
        benchmark_state=inputs["benchmark_state"],
        input_metadata=inputs["input_metadata"],
    )


def test_complete_clean_campaign_promotes(valid_inputs: dict[str, Any]) -> None:
    report = _evaluate(valid_inputs)

    assert report["status"] == "candidate_for_attestation", report["evidence_revalidation"][
        "calphad_cross_language"
    ]["issues"]
    assert report["promotion"]["passed"] is True
    assert report["promotion"]["evidence_passed"] is True
    assert report["promotion"]["attestation_required"] is True
    assert report["promotion"]["distribution_ready"] is False
    assert report["promotion"]["full_materials_production_ready"] is False
    assert all(report["hard_gates"].values())
    assert report["counts"]["mattools"]["runnable"] == 147
    assert report["counts"]["mattools"]["strict_scientific_pass"] == 414
    assert report["counts"]["calphad_cross_language"]["passed"] is True
    assert report["evidence_revalidation"]["calphad_cross_language"]["valid"] is True
    assert report["counts"]["designated_live_traces"] == 1


@pytest.mark.parametrize("artifact_name", ["database_input", "inspect"])
def test_cross_language_retained_artifact_tamper_blocks_promotion(
    valid_inputs: dict[str, Any],
    artifact_name: str,
) -> None:
    report_root = Path(
        valid_inputs["input_metadata"]["calphad_cross_language_report"]["path"]
    ).parent
    artifact_path = (
        report_root
        / valid_inputs["calphad_cross_language"]["typed_cli_artifacts"][artifact_name]["path"]
    )
    artifact_path.write_bytes(artifact_path.read_bytes() + b"tampered")
    report = _evaluate(valid_inputs)
    assert report["promotion"]["passed"] is False
    assert report["hard_gates"]["calphad_typed_cli_http_postgres_cross_language_qualified"] is False


def test_exact_published_thresholds_promote(valid_inputs: dict[str, Any]) -> None:
    target_runnable = (40, 40, 40)
    target_scientific = (83, 83, 83)
    for trial, runnable_target, scientific_target in zip(
        valid_inputs["mattools"]["trials"],
        target_runnable,
        target_scientific,
        strict=True,
    ):
        remaining = scientific_target
        for index, attempt in enumerate(trial["attempts"]):
            task = valid_inputs["mattools"]["benchmark"]["tasks"][index]
            scientific_pass = min(task["subtask_count"], remaining)
            remaining -= scientific_pass
            runnable = index < runnable_target
            attempt["evaluation"]["runnable"] = runnable
            attempt["evaluation"]["scientific_pass"] = scientific_pass
            attempt["evaluation"]["scientific_fail"] = task["subtask_count"] - scientific_pass
            for replay in attempt["scoring_evidence"]["replays"]:
                replay["published_upstream"]["runnable"] = runnable
                replay["published_upstream"]["scientific_pass"] = scientific_pass
                replay["published_upstream"]["scientific_fail"] = (
                    task["subtask_count"] - scientific_pass
                )
                replay["strict_shadow"]["semantic_runnable"] = runnable
                replay["strict_shadow"]["strict_scientific_pass"] = scientific_pass
                replay["strict_shadow"]["strict_scientific_fail"] = (
                    task["subtask_count"] - scientific_pass
                )
        assert remaining == 0
        trial["runnable"] = runnable_target
        trial["published_runner_runnable"] = runnable_target
        trial["function_runnable_rate"] = runnable_target / 49
        trial["scientific_pass"] = scientific_target
        trial["strict_scientific_pass"] = scientific_target
        trial["task_success_rate"] = scientific_target / 138
        trial["strict_task_success_rate"] = scientific_target / 138
    counts = valid_inputs["mattools"]["counts"]
    counts["runnable"] = 120
    counts["scientific_pass"] = 249
    counts["strict_scientific_pass"] = 249
    rates = valid_inputs["mattools"]["rates"]
    rates["function_runnable"] = 120 / 147
    rates["task_success"] = 249 / 414
    rates["strict_task_success"] = 249 / 414

    report = _evaluate(valid_inputs)

    assert report["promotion"]["passed"] is True
    assert report["rates"]["mattools_function_runnable"] >= 0.80
    assert report["rates"]["mattools_strict_task_success"] >= 0.60


def test_recomputed_per_trial_runnable_floor_blocks_aggregate_spoof(
    valid_inputs: dict[str, Any],
) -> None:
    trial = valid_inputs["mattools"]["trials"][0]
    for attempt in trial["attempts"][:10]:
        for replay in attempt["scoring_evidence"]["replays"]:
            replay["strict_shadow"]["semantic_runnable"] = False
    trial["runnable"] = 39
    counts = valid_inputs["mattools"]["counts"]
    counts["runnable"] = 137
    valid_inputs["mattools"]["rates"]["function_runnable"] = 137 / 147

    report = _evaluate(valid_inputs)

    assert report["hard_gates"]["mattools_function_runnable_rate"] is True
    assert report["hard_gates"]["mattools_per_trial_function_runnable"] is False
    assert report["promotion"]["passed"] is False


def test_recomputed_per_trial_strict_floor_blocks_aggregate_spoof(
    valid_inputs: dict[str, Any],
) -> None:
    trial = valid_inputs["mattools"]["trials"][0]
    remaining = 82
    for index, attempt in enumerate(trial["attempts"]):
        task = valid_inputs["mattools"]["benchmark"]["tasks"][index]
        scientific_pass = min(task["subtask_count"], remaining)
        remaining -= scientific_pass
        for replay in attempt["scoring_evidence"]["replays"]:
            replay["strict_shadow"]["strict_scientific_pass"] = scientific_pass
            replay["strict_shadow"]["strict_scientific_fail"] = (
                task["subtask_count"] - scientific_pass
            )
    assert remaining == 0
    trial["strict_scientific_pass"] = 82
    counts = valid_inputs["mattools"]["counts"]
    counts["strict_scientific_pass"] = 358
    valid_inputs["mattools"]["rates"]["strict_task_success"] = 358 / 414

    report = _evaluate(valid_inputs)

    assert report["hard_gates"]["mattools_strict_scientific_correctness"] is True
    assert report["hard_gates"]["mattools_per_trial_strict_scientific_correctness"] is False
    assert report["promotion"]["passed"] is False


def test_non_runnable_zero_strict_score_may_omit_raw_verifier_digest(
    valid_inputs: dict[str, Any],
) -> None:
    trial = valid_inputs["mattools"]["trials"][0]
    attempt = trial["attempts"][0]
    task_total = attempt["scoring_evidence"]["subtask_count"]
    for replay in attempt["scoring_evidence"]["replays"]:
        strict = replay["strict_shadow"]
        strict["semantic_runnable"] = False
        strict["strict_scientific_pass"] = 0
        strict["strict_scientific_fail"] = task_total
        strict["raw_verifier_output_sha256"] = None
    trial["runnable"] = 48
    trial["strict_scientific_pass"] -= task_total
    counts = valid_inputs["mattools"]["counts"]
    counts["runnable"] = 146
    counts["strict_scientific_pass"] -= task_total
    valid_inputs["mattools"]["rates"]["function_runnable"] = 146 / 147
    valid_inputs["mattools"]["rates"]["strict_task_success"] = (
        counts["strict_scientific_pass"] / 414
    )

    report = _evaluate(valid_inputs)

    assert report["promotion"]["passed"] is True
    assert report["hard_gates"]["mattools_three_trial_coverage"] is True


@pytest.mark.parametrize(
    "make_invalid",
    [
        lambda strict, _task_total: strict.update(raw_verifier_output_sha256=None),
        lambda strict, _task_total: strict.update(
            semantic_runnable=False,
            raw_verifier_output_sha256=None,
        ),
    ],
)
def test_raw_verifier_digest_null_fails_closed_outside_non_runnable_zero_score(
    valid_inputs: dict[str, Any],
    make_invalid: Any,
) -> None:
    attempt = valid_inputs["mattools"]["trials"][0]["attempts"][0]
    task_total = attempt["scoring_evidence"]["subtask_count"]
    for replay in attempt["scoring_evidence"]["replays"]:
        make_invalid(replay["strict_shadow"], task_total)

    report = _evaluate(valid_inputs)

    assert report["hard_gates"]["mattools_three_trial_coverage"] is False
    assert report["promotion"]["passed"] is False


@pytest.mark.parametrize(
    ("mutation", "expected_gate"),
    [
        (lambda data: data["repository_state"].update(dirty=True), "aggregator_checkout_clean"),
        (lambda data: data["domain"]["git"].update(dirty=True), "same_clean_git_sha"),
        (
            lambda data: data["production_parity"].update(scope="ci-pinned-materials"),
            "production_full_image_parity",
        ),
        (
            lambda data: data["production_parity"]["sandbox"].update(
                source="ci_fixed_limits", policy_source="ci_fixed_limits"
            ),
            "production_full_image_parity",
        ),
        (
            lambda data: data["production_parity"]["calphad_runtime"]["junit"].update(skipped=1),
            "production_full_image_parity",
        ),
        (
            lambda data: data["production_parity"]["source"].update(kind="clean_git_checkout"),
            "production_full_image_parity",
        ),
        (
            lambda data: data["production_parity"].update(verified_release_artifacts={}),
            "production_full_image_parity",
        ),
        (
            lambda data: data["production_parity"]["calphad_tool_orchestration"]["junit"].update(
                skipped=1
            ),
            "production_full_image_parity",
        ),
        (
            lambda data: data["production_parity"]["calphad_tool_orchestration"]["binding"].update(
                runtime_image_id=DOMAIN_IMAGE
            ),
            "production_full_image_parity",
        ),
        (
            lambda data: data["calphad_ledger"]["postgres_invariants"].update(
                evidence_bytes_server_verified=False
            ),
            "calphad_postgres_ledger_qualified",
        ),
        (
            lambda data: data["calphad_cross_language"].update(
                production_live_qualified=False, promotable=False
            ),
            "calphad_typed_cli_http_postgres_cross_language_qualified",
        ),
        (
            lambda data: data["calphad_cross_language"]["generation"].update(
                runtime_image_id=DOMAIN_IMAGE
            ),
            "calphad_typed_cli_http_postgres_cross_language_qualified",
        ),
        (
            lambda data: data["domain"]["provenance_policy"].update(
                status="not_enforced", promotion_provenance_enforced=False
            ),
            "deterministic_clean_provenance_enforced",
        ),
        (
            lambda data: data["benchmark_state"].update(dirty=True),
            "benchmark_evidence_rehashed",
        ),
        (
            lambda data: data["domain"]["junit"].update(
                tests=gate.REQUIRED_DOMAIN_INVARIANT_COUNT - 1
            ),
            "deterministic_required_invariants",
        ),
        (lambda data: data["domain"]["junit"].update(skipped=1), "deterministic_zero_skip"),
        (
            lambda data: data["domain"]["invariant_evidence"].update(complete=False),
            "deterministic_invariant_evidence_complete",
        ),
        (
            lambda data: data["mattools"]["counts"].update(runnable=117),
            "mattools_function_runnable_rate",
        ),
        (
            lambda data: data["mattools"]["counts"].update(strict_scientific_pass=248),
            "mattools_strict_scientific_correctness",
        ),
        (
            lambda data: data["mattools"]["hard_gates"].update(
                expected_values_and_verifiers_isolated=False
            ),
            "mattools_answer_isolation",
        ),
        (
            lambda data: data["mattools"]["hard_gates"].update(
                checkpoint_non_erasure_integrity=False
            ),
            "mattools_checkpoint_non_erasure",
        ),
        (
            lambda data: data["mattools"]["hard_gates"].update(
                official_evaluator_environment_exact=False
            ),
            "mattools_evaluator_environment_exact",
        ),
        (
            lambda data: data["mattools"]["official_evaluator_environment"].update(
                required_packages={"pymatgen": "latest"}
            ),
            "mattools_report_schema",
        ),
        (
            lambda data: data["mattools"]["harness"]["host_validator_environment"].update(
                python_executable_sha256="invalid"
            ),
            "mattools_host_validator_environment_exact",
        ),
        (
            lambda data: data["mattools"]["trials"][0]["evaluator_environment"][
                "resolved_packages"
            ].update({"fixture-transitive": "changed"}),
            "mattools_evaluator_environment_exact",
        ),
        (
            lambda data: data["mattools"]["trials"][0]["attempts"][0]["trace_summary"].update(
                production_execute_tool_evidence=False
            ),
            "mattools_production_execute_evidence",
        ),
        (
            lambda data: data["mattools"]["trials"][0]["attempts"][0]["trace_summary"].update(
                server_cleanroom_profile_attested=False
            ),
            "mattools_server_authorized_cleanroom_profile",
        ),
        (
            lambda data: data["mattools"]["trials"][0]["attempts"][0]["cleanroom_binding"].update(
                valid=False
            ),
            "mattools_worker_enforced_cleanroom_profile",
        ),
        (
            lambda data: data["mattools"]["trials"][0]["attempts"][0]["trace_summary"][
                "worker_cleanroom_attestations"
            ][0]["payload"].update(goal_sha256="0" * 64),
            "mattools_worker_enforced_cleanroom_profile",
        ),
        (
            lambda data: data["mattools"]["trials"][0]["evaluator_environment"][
                "image_labels"
            ].update({"io.ultra.mattools.candidate-visible-source-policy": "full-source"}),
            "mattools_evaluator_environment_exact",
        ),
        (
            lambda data: data["mattools"]["trials"][0]["evaluator_environment"]["embedded_inputs"][
                "candidate_visible_dependency_test_paths"
            ]["pymatgen"].append("pymatgen/tests/test_hidden.py"),
            "mattools_evaluator_environment_exact",
        ),
        (
            lambda data: data["mattools"]["trials"][0]["attempts"][0]["trace_summary"].update(
                production_execute_completed_count=0
            ),
            "mattools_production_execute_evidence",
        ),
        (
            lambda data: data["mattools"]["trials"][0]["attempts"][0]["trace_summary"].update(
                tool_names=["execute", "bisque_create_dataset"]
            ),
            "mattools_no_unauthorized_remote_mutation",
        ),
        (
            lambda data: data["mattools"]["trials"][0]["attempts"][0][
                "actual_runtime_provenance"
            ].update(validated=False),
            "mattools_observable_model_provider",
        ),
        (
            lambda data: data["traces"][0]["prompt"]["materials_validation"].update(
                silent_success=True
            ),
            "live_traces_no_silent_success",
        ),
        (
            lambda data: data["traces"][0]["materials_quality"]["signals"].update(
                remote_mutation_aligned=False
            ),
            "live_traces_remote_mutation_authorized",
        ),
        (
            lambda data: data["traces"][0]["materials_quality"]["signals"].pop(
                "remote_mutation_scope_valid"
            ),
            "live_traces_remote_mutation_authorized",
        ),
        (
            lambda data: data["traces"][0]["materials_quality"]["signals"].update(
                first_party_scientific_record_valid=False
            ),
            "live_traces_first_party_records_valid",
        ),
        (
            lambda data: data["traces"][0]["prompt"]["materials_validation"].pop("retained_path"),
            "live_traces_retained_validation_artifacts",
        ),
        (
            lambda data: data["mattools"]["license_attestation"].update(accepted=False),
            "external_license_attestation",
        ),
        (lambda data: data["traces"].clear(), "live_traces_designated"),
    ],
)
def test_fail_closed_on_decisive_gate_regression(
    valid_inputs: dict[str, Any], mutation: Any, expected_gate: str
) -> None:
    mutation(valid_inputs)

    report = _evaluate(valid_inputs)

    assert report["status"] == "blocked"
    assert report["promotion"]["passed"] is False
    assert report["hard_gates"][expected_gate] is False
    assert expected_gate in report["promotion"]["reasons"]


def test_recomputes_counts_instead_of_trusting_published_booleans(
    valid_inputs: dict[str, Any],
) -> None:
    valid_inputs["mattools"]["trials"][0]["attempts"].pop()

    report = _evaluate(valid_inputs)

    assert report["hard_gates"]["mattools_three_trial_coverage"] is False
    assert report["promotion"]["passed"] is False


def test_legacy_mattools_report_without_new_contracts_fails_closed(
    valid_inputs: dict[str, Any],
) -> None:
    counts = valid_inputs["mattools"]["counts"]
    counts.pop("per_trial_runnable_minimum")
    counts.pop("per_trial_scientific_minimum")
    hard = valid_inputs["mattools"]["hard_gates"]
    hard.pop("server_authorized_cleanroom_profile")
    hard.pop("worker_enforced_cleanroom_profile")

    report = _evaluate(valid_inputs)

    assert report["hard_gates"]["mattools_published_counts_consistent"] is False
    assert report["hard_gates"]["mattools_lane_hard_gates"] is False
    assert report["promotion"]["passed"] is False


def test_semantic_repair_build_input_tamper_blocks_evaluator_and_repository_evidence(
    valid_inputs: dict[str, Any],
) -> None:
    repair_path = valid_inputs["repository_root"] / "scripts/mattools_semantic_repairs.py"
    repair_path.write_text("# tampered semantic repairs\n", encoding="utf-8")

    report = _evaluate(valid_inputs)

    assert report["hard_gates"]["repository_evidence_rehashed"] is False
    assert report["hard_gates"]["mattools_evaluator_environment_exact"] is False
    assert report["promotion"]["passed"] is False


def test_tampered_retained_live_validation_bytes_block_promotion(
    valid_inputs: dict[str, Any],
) -> None:
    retained = Path(valid_inputs["traces"][0]["prompt"]["materials_validation"]["retained_path"])
    retained.write_bytes(retained.read_bytes() + b"\n")

    report = _evaluate(valid_inputs)

    assert report["status"] == "blocked"
    assert report["hard_gates"]["live_traces_retained_validation_artifacts"] is False
    assert report["hard_gates"]["live_traces_evidence_integrity"] is False


def test_semantic_runnable_is_independent_of_published_upstream_bug(
    valid_inputs: dict[str, Any],
) -> None:
    attempt = valid_inputs["mattools"]["trials"][0]["attempts"][0]
    attempt["evaluation"]["runnable"] = False
    for replay in attempt["scoring_evidence"]["replays"]:
        replay["published_upstream"]["runnable"] = False
    valid_inputs["mattools"]["trials"][0]["published_runner_runnable"] = 48

    report = _evaluate(valid_inputs)

    assert report["promotion"]["passed"] is True
    assert report["rates"]["mattools_function_runnable"] == 1.0


def test_forged_strict_aggregate_without_per_attempt_strict_evidence_blocks(
    valid_inputs: dict[str, Any],
) -> None:
    for trial in valid_inputs["mattools"]["trials"]:
        for attempt in trial["attempts"]:
            total = attempt["scoring_evidence"]["subtask_count"]
            for replay in attempt["scoring_evidence"]["replays"]:
                replay["strict_shadow"]["strict_scientific_pass"] = 0
                replay["strict_shadow"]["strict_scientific_fail"] = total
        trial["strict_scientific_pass"] = 0
        trial["strict_task_success_rate"] = 0.0
    valid_inputs["mattools"]["counts"]["strict_scientific_pass"] = 249
    valid_inputs["mattools"]["rates"]["strict_task_success"] = 249 / 414

    report = _evaluate(valid_inputs)

    assert report["hard_gates"]["mattools_published_counts_consistent"] is False
    assert report["promotion"]["passed"] is False


def test_rehashes_referenced_repository_and_attestation_evidence(
    valid_inputs: dict[str, Any],
) -> None:
    harness = valid_inputs["repository_root"] / "scripts/mattools_promotion_gate.py"
    harness.write_text("# modified after report\n", encoding="utf-8")
    isolation = Path(
        valid_inputs["mattools"]["trials"][0]["sandbox_policy_attestation"][
            "isolation_evidence_path"
        ]
    )
    isolation.write_text("tampered\n", encoding="utf-8")

    report = _evaluate(valid_inputs)

    assert report["hard_gates"]["repository_evidence_rehashed"] is False
    assert report["hard_gates"]["external_isolation_attestations"] is False
    assert report["promotion"]["passed"] is False


def test_signed_isolation_json_cannot_disagree_with_unsigned_report_fields(
    valid_inputs: dict[str, Any],
) -> None:
    record = valid_inputs["mattools"]["trials"][0]["sandbox_policy_attestation"]
    signed_path = Path(record["path"])
    signed = json.loads(signed_path.read_text(encoding="utf-8"))
    signed["network_egress_denied"] = False
    signed_path.write_text(json.dumps(signed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record["sha256"] = _sha(signed_path)

    report = _evaluate(valid_inputs)

    assert report["hard_gates"]["external_isolation_attestations"] is False
    assert report["promotion"]["passed"] is False


def test_host_validator_rejects_incomplete_self_consistent_lock(
    valid_inputs: dict[str, Any],
) -> None:
    lock = valid_inputs["repository_root"] / "scripts/mattools-validator-requirements.lock.txt"
    incomplete = "docker==7.1.0 \\\n    --hash=sha256:" + "1" * 64 + "\n"
    lock.write_text(incomplete, encoding="utf-8")
    digest = _sha(lock)
    harness = valid_inputs["mattools"]["harness"]
    harness["host_validator_requirements_sha256"] = digest
    environment = harness["host_validator_environment"]
    environment["requirements_lock_sha256"] = digest
    for trial in valid_inputs["mattools"]["trials"]:
        trial["runner"]["host_requirements_sha256"] = digest

    report = _evaluate(valid_inputs)

    assert report["hard_gates"]["mattools_host_validator_environment_exact"] is False
    assert report["promotion"]["passed"] is False


def test_host_validator_rejects_fake_or_overridden_uv_command(
    valid_inputs: dict[str, Any],
) -> None:
    environment = valid_inputs["mattools"]["harness"]["host_validator_environment"]
    command = list(environment["validator_command"])
    command[0] = "/fixture/fake-uv"
    command.extend(["--python", "3.13"])
    environment["validator_command"] = command
    for trial in valid_inputs["mattools"]["trials"]:
        trial["runner"]["host_validator_command"] = command
        trial["runner"]["host_validator_executable_sha256"] = "f" * 64

    report = _evaluate(valid_inputs)

    assert report["hard_gates"]["mattools_host_validator_environment_exact"] is False
    assert report["promotion"]["passed"] is False


def test_calphad_ledger_claims_cannot_override_failed_retained_go_event(
    valid_inputs: dict[str, Any],
) -> None:
    ledger = valid_inputs["calphad_ledger"]
    old_log = (
        Path(valid_inputs["input_metadata"]["calphad_ledger_report"]["path"]).parent
        / ledger["runner"]["go_test_log"]["path"]
    )
    events = [json.loads(line) for line in old_log.read_text(encoding="utf-8").splitlines()]
    target = gate.REQUIRED_CALPHAD_LEDGER_TESTS[0]
    next(event for event in events if event.get("Test") == target)["Action"] = "fail"
    payload = "".join(json.dumps(event) + "\n" for event in events).encode()
    digest = hashlib.sha256(payload).hexdigest()
    new_log = old_log.parent / f"calphad-ledger-go-test-{digest}.jsonl"
    new_log.write_bytes(payload)
    ledger["runner"]["go_test_log"] = {
        "path": new_log.name,
        "sha256": digest,
        "size_bytes": len(payload),
    }
    _rewrite_calphad_ledger_report(valid_inputs)

    report = _evaluate(valid_inputs)

    assert report["hard_gates"]["calphad_postgres_ledger_qualified"] is False
    assert report["promotion"]["passed"] is False


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("role_create_database", True),
        ("role_replication", True),
        ("calphad_reachable_roles", ["pg_monitor"]),
        ("calphad_insert_all", True),
        ("calphad_insert_any", True),
        ("calphad_execute_create_revision", False),
        ("calphad_execute_append_validation", False),
        ("calphad_writer_functions_exact", False),
        ("calphad_execute_unexpected_writer", True),
        ("calphad_execute_internal", True),
        ("calphad_public_execute", True),
        ("calphad_unexpected_table_acl_grantees", ["PUBLIC"]),
        ("calphad_unexpected_function_acl_grantees", ["pg_monitor"]),
    ],
)
def test_calphad_ledger_execute_only_role_evidence_fails_closed(
    valid_inputs: dict[str, Any], field: str, unsafe_value: object
) -> None:
    ledger = valid_inputs["calphad_ledger"]
    old_log = (
        Path(valid_inputs["input_metadata"]["calphad_ledger_report"]["path"]).parent
        / ledger["runner"]["go_test_log"]["path"]
    )
    events = [json.loads(line) for line in old_log.read_text(encoding="utf-8").splitlines()]
    identity_event = next(
        event
        for event in events
        if str(event.get("Output") or "").startswith(gate.CALPHAD_POSTGRES_IDENTITY_MARKER)
    )
    identity = json.loads(
        str(identity_event["Output"])[len(gate.CALPHAD_POSTGRES_IDENTITY_MARKER) :]
    )
    identity[field] = unsafe_value
    identity_event["Output"] = gate.CALPHAD_POSTGRES_IDENTITY_MARKER + json.dumps(
        identity, sort_keys=True
    )
    payload = "".join(json.dumps(event) + "\n" for event in events).encode()
    digest = hashlib.sha256(payload).hexdigest()
    new_log = old_log.parent / f"calphad-ledger-go-test-{digest}.jsonl"
    new_log.write_bytes(payload)
    ledger["observed_database"] = identity
    ledger["runner"]["go_test_log"] = {
        "path": new_log.name,
        "sha256": digest,
        "size_bytes": len(payload),
    }
    _rewrite_calphad_ledger_report(valid_inputs)

    report = _evaluate(valid_inputs)

    assert report["hard_gates"]["calphad_postgres_ledger_qualified"] is False
    assert report["promotion"]["passed"] is False


def test_calphad_ledger_rejects_production_like_database_even_when_report_is_rehashed(
    valid_inputs: dict[str, Any],
) -> None:
    valid_inputs["calphad_ledger"]["database"]["database"] = "production_ci"
    _rewrite_calphad_ledger_report(valid_inputs)

    report = _evaluate(valid_inputs)

    assert report["hard_gates"]["calphad_postgres_ledger_qualified"] is False
    assert report["promotion"]["passed"] is False


def test_mattools_campaign_manifest_rehashes_checkpoint(valid_inputs: dict[str, Any]) -> None:
    checkpoint = Path(valid_inputs["mattools_manifest"]["checkpoint"]["path"])
    checkpoint.write_text('{"tampered":true}\n', encoding="utf-8")

    report = _evaluate(valid_inputs)

    assert report["hard_gates"]["mattools_report_manifest_integrity"] is False
    assert report["promotion"]["passed"] is False


def test_expected_immutable_image_mismatch_blocks(valid_inputs: dict[str, Any]) -> None:
    valid_inputs["expected"] = gate.ExpectedProvenance(
        git_sha=GIT_SHA,
        domain_image="sha256:" + "9" * 64,
        runtime_image=RUNTIME_IMAGE,
        evaluator_image=EVALUATOR_IMAGE,
    )

    report = _evaluate(valid_inputs)

    assert report["hard_gates"]["expected_immutable_images"] is False


def test_missing_full_image_parity_attestation_blocks(valid_inputs: dict[str, Any]) -> None:
    valid_inputs["production_parity"] = {}
    valid_inputs["input_metadata"]["production_parity_report"] = {}

    report = _evaluate(valid_inputs)

    assert report["hard_gates"]["production_full_image_parity"] is False
    assert report["promotion"]["full_materials_production_ready"] is False


def test_tampered_full_image_companion_evidence_blocks(valid_inputs: dict[str, Any]) -> None:
    report_path = Path(valid_inputs["input_metadata"]["production_parity_report"]["path"])
    companion = report_path.parent / "calphad-embedded-probe.json"
    companion.write_text('{"status":"passed"}\n', encoding="utf-8")

    report = _evaluate(valid_inputs)

    parity = report["evidence_revalidation"]["production_parity"]
    assert parity["checks"]["retained_companion_evidence"] is False
    assert report["hard_gates"]["production_full_image_parity"] is False


def test_readiness_reparses_runtime_junit_and_rejects_legacy_29_test_suite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calphad-runtime-junit.xml"
    cases = "".join(
        f'<testcase classname="tests.test_calphad_runtime" name="{name}" />'
        for name in gate.REQUIRED_CALPHAD_RUNTIME_CORE_TEST_NAMES[:28]
    )
    cases += (
        '<testcase classname="tests.test_calphad_cli" '
        f'name="{gate.REQUIRED_CALPHAD_RUNTIME_CLI_TEST_NAMES[0]}" />'
    )
    path.write_text(
        '<testsuites><testsuite tests="29" failures="0" errors="0" skipped="0">'
        f"{cases}</testsuite></testsuites>\n",
        encoding="utf-8",
    )

    evidence = gate._calphad_runtime_junit_evidence(path)

    assert evidence["summary"]["tests"] == 29
    assert evidence["valid"] is False
    assert any("exactly 39 tests" in issue for issue in evidence["issues"])


def test_readiness_reparses_tools_junit_and_rejects_legacy_30_test_suite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calphad-tools-junit.xml"
    cases = "".join(
        f'<testcase classname="tests.test_calphad_tools" name="{name}" />'
        for name in gate.REQUIRED_CALPHAD_TOOL_TEST_NAMES[:30]
    )
    path.write_text(
        '<testsuites><testsuite tests="30" failures="0" errors="0" skipped="0">'
        f"{cases}</testsuite></testsuites>\n",
        encoding="utf-8",
    )

    evidence = gate._calphad_tools_junit_evidence(path)

    assert evidence["summary"]["tests"] == 30
    assert evidence["valid"] is False
    assert any("exactly 56 tests" in issue for issue in evidence["issues"])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report["databases"][0].update(format="dat"),
        lambda report: report["databases"][0].update(assessment_pressure_limits_Pa=[1e12, 1e-9]),
        lambda report: report["materials_source_hashes"].update({"calphad.py": "f" * 64}),
        lambda report: report.update(source_manifest_sha256="f" * 64),
    ],
)
def test_calphad_probe_fields_are_rebound_to_repository_bytes(
    valid_inputs: dict[str, Any], mutation: Any
) -> None:
    calphad = copy.deepcopy(valid_inputs["production_parity"]["calphad"]["report"])
    mutation(calphad)

    evidence = gate._calphad_probe_source_evidence(calphad, valid_inputs["repository_root"])

    assert evidence["valid"] is False


def test_stale_calphad_release_counts_and_domain_preflight_fail_closed(
    valid_inputs: dict[str, Any],
) -> None:
    parity = copy.deepcopy(valid_inputs["production_parity"])
    parity["calphad_release_contract"]["runtime_test_count"] = 29
    parity["calphad_release_contract"]["calphad_tools_test_count"] = 30
    contract = gate._calphad_release_contract_evidence(parity, valid_inputs["repository_root"])
    domain = copy.deepcopy(parity["domain_gate"]["report"])
    domain["runtime"]["calphad_runtime_preflight"]["junit"]["tests"] = 29

    assert contract["valid"] is False
    assert gate._domain_calphad_preflight_valid(domain) is False

    assert gate._domain_calphad_experimental_benchmark_valid(domain) is True
    domain["calphad_experimental_benchmark"]["report"]["lanes"]["held_out"]["metrics"][
        "mae_K_max"
    ] = 21.0
    assert gate._domain_calphad_experimental_benchmark_valid(domain) is False


def test_json_and_markdown_outputs_are_auditable(
    valid_inputs: dict[str, Any], tmp_path: Path
) -> None:
    report = _evaluate(valid_inputs)
    json_path = tmp_path / "materials-readiness.json"
    markdown_path = tmp_path / "materials-readiness.md"
    manifest_path = tmp_path / "materials-readiness-manifest.json"

    manifest = gate.write_outputs(
        report,
        json_path=json_path,
        markdown_path=markdown_path,
        manifest_path=manifest_path,
    )

    stored = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert stored["promotion"]["passed"] is True
    assert stored["promotion"]["full_materials_production_ready"] is False
    assert manifest["attestation_required"] is True
    assert manifest["full_materials_production_ready"] is False
    assert "# Materials Production-Readiness Gate" in markdown
    assert "CANDIDATE — ATTESTATION REQUIRED" in markdown
    assert "147/147" in markdown
    assert "414/414" in markdown
    immutable_report = tmp_path / manifest["report"]["path"]
    assert immutable_report.read_bytes() == json_path.read_bytes()
    assert _sha(immutable_report) == manifest["report"]["sha256"]
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest


def test_safe_evidence_path_preserves_final_symlink_manifest_semantics(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("target bytes\n", encoding="utf-8")
    link = tmp_path / "tracked-link"
    link.symlink_to(target.name)

    resolved = gate._safe_relative_path(tmp_path, link.name)

    assert resolved is not None and resolved.is_symlink()
    assert gate.sha256_tracked_path(resolved) == hashlib.sha256(target.name.encode()).hexdigest()


def test_required_invariant_set_tracks_the_release_domain_suite() -> None:
    source = (
        ROOT / "backend/deepagents_runtime/tests/domain_correctness/test_materials_invariants.py"
    ).read_text(encoding="utf-8")
    observed = set(re.findall(r'validator_id="([a-z0-9_.:-]+)"', source))

    assert observed == set(gate.REQUIRED_DOMAIN_VALIDATORS)
    assert len(observed) == gate.REQUIRED_DOMAIN_INVARIANT_COUNT


def test_missing_nist_calphad_invariant_blocks_promotion(
    valid_inputs: dict[str, Any],
) -> None:
    required_id = "materials.calphad.nist_al_co_w_phase_field_checkpoints.v2"
    valid_inputs["domain"]["invariants"] = [
        record
        for record in valid_inputs["domain"]["invariants"]
        if record["validator_id"] != required_id
    ]
    valid_inputs["domain"]["invariant_evidence"].update(
        record_count=gate.REQUIRED_DOMAIN_INVARIANT_COUNT - 1,
        passed=gate.REQUIRED_DOMAIN_INVARIANT_COUNT - 1,
    )

    report = _evaluate(valid_inputs)

    assert report["hard_gates"]["deterministic_invariant_evidence_complete"] is False
    assert report["promotion"]["passed"] is False
