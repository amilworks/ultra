from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/materials_promotion_envelope.py"
MODULE_NAME = "_ultra_materials_promotion_envelope_test"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = gate
SPEC.loader.exec_module(gate)

GIT_SHA = "a" * 40
DOMAIN_IMAGE = "sha256:" + "b" * 64
RUNTIME_IMAGE = "sha256:" + "c" * 64
RUNTIME_OCI = "sha256:" + "d" * 64
EVALUATOR_IMAGE = "sha256:" + "e" * 64
REPOSITORY = "amilworks/ultra"
REPOSITORY_ID = "1204778765"
OWNER_ID = "22850980"
RUN_ID = "987654321"
RUN_ATTEMPT = 2
MODEL_ID = "materials-model-v1"
PROVIDER_ID = "internal-openai-compatible-provider"
LICENSE_PURPOSE = "Noncommercial internal materials quality qualification"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_json(path: Path, value: Any) -> dict[str, Any]:
    data = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "path": str(path.absolute()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _cleanroom_attempt(run_id: str, thread_id: str) -> dict[str, Any]:
    run_sha = hashlib.sha256(run_id.encode()).hexdigest()
    thread_sha = hashlib.sha256(thread_id.encode()).hexdigest()
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
        "user_id_sha256": hashlib.sha256(b"researcher-1").hexdigest(),
        "goal_sha256": hashlib.sha256(f"goal:{run_id}".encode()).hexdigest(),
        "input_policy": "goal_only",
        "provided_message_count": 1,
        "effective_message_count": 1,
        "prior_thread_context_discarded": True,
        "same_run_retry_state_allowed": True,
        "run_scoped_workspace": True,
        "run_scoped_memory": True,
        "disabled_capabilities": list(gate.WORKER_CLEANROOM_DISABLED_CAPABILITIES),
    }
    payload["attestation_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return {
        "run_id": run_id,
        "thread_id": thread_id,
        "trace_summary": {
            "server_cleanroom_profile_attested": True,
            "server_evaluation_profiles": [gate.MATERIALS_CLEANROOM_PROFILE],
            "worker_cleanroom_attestations": [
                {
                    "valid": True,
                    "payload": payload,
                    "source_payload_keys": sorted(gate.WORKER_EVALUATION_ATTESTATION_FIELDS),
                }
            ],
        },
        "cleanroom_binding": {
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
        },
    }


def _release_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "release_sha": GIT_SHA,
        "release_name": GIT_SHA,
        "source": {
            "repository": REPOSITORY,
            "ref": "main",
            "github_run_id": RUN_ID,
            "required_materials": {},
        },
        "materials": {
            "full_materials_production_ready": False,
            "required_post_image_gate": "materials-production-readiness",
            "required_evidence": {
                "production_parity_scope": "production-full",
                "calphad_cross_language_requires_production_runtime_image": True,
                "mattools_runnable_minimum": 118,
                "mattools_scientific_minimum": 249,
            },
        },
    }


def _write_release_tarball(path: Path, manifest_bytes: bytes) -> None:
    with tarfile.open(path, mode="w:gz") as archive:
        directory = tarfile.TarInfo(GIT_SHA)
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o755
        archive.addfile(directory)
        info = tarfile.TarInfo(f"{GIT_SHA}/release-manifest.json")
        info.size = len(manifest_bytes)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(manifest_bytes))
        payload = b"release payload\n"
        info = tarfile.TarInfo(f"{GIT_SHA}/README.txt")
        info.size = len(payload)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(payload))


def _role_value(name: str, path: Path, root: Path) -> str:
    return f"{name}={path.relative_to(root).as_posix()}"


def _build_fixture(tmp_path: Path, *, separately_licensed: bool = False) -> dict[str, Any]:
    evidence_root = tmp_path / "restricted-evidence"
    evidence_root.mkdir()
    reports = evidence_root / "reports"
    campaign = evidence_root / "mattools-campaign"
    license_root = evidence_root / "license"
    release = evidence_root / "release"
    reports.mkdir()
    campaign.mkdir()
    license_root.mkdir()
    release.mkdir()

    license_path: Path | None = None
    license_sha: str | None = None
    if separately_licensed:
        license_path = license_root / "license-evidence.bin"
        license_path.write_bytes(b"reviewed separate MatTools license evidence\n")
        license_sha = hashlib.sha256(license_path.read_bytes()).hexdigest()

    deterministic_path = reports / "materials-domain-gate.json"
    deterministic_meta = _write_json(
        deterministic_path,
        {
            "schema_version": 1,
            "gate": "materials-domain-gate",
            "scope": "deterministic-domain-invariants",
            "status": "passed",
            "failures": [],
        },
    )
    parity_path = reports / "production-parity.json"
    parity_meta = _write_json(
        parity_path,
        {
            "schema_version": 1,
            "gate": "production-materials-sandbox-parity",
            "scope": "production-full",
            "status": "passed",
            "failures": [],
            "expected_git_sha": GIT_SHA,
            "full_production_image_parity": True,
            "evidence_bundle": {"schema_version": 1, "promotable": True},
            "executed_image": {"image_id": RUNTIME_IMAGE, "revision": GIT_SHA},
        },
    )
    ledger_path = reports / "calphad-ledger.json"
    ledger_meta = _write_json(
        ledger_path,
        {
            "schema_version": "1",
            "gate": "calphad-ledger-postgres-qualification",
            "status": "passed",
            "qualification_database": True,
            "production_database_used": False,
            "git_sha": GIT_SHA,
            "repository_clean": True,
            "failures": [],
        },
    )
    cross_path = reports / "calphad-cross-language.json"
    cross_meta = _write_json(
        cross_path,
        {
            "schema_version": "ultra.calphad.cross-language-gate.v1",
            "gate": "calphad-typed-cli-http-postgres-cross-language",
            "status": "qualified",
            "expected_git_sha": GIT_SHA,
            "production_live_qualified": True,
            "promotable": True,
            "generation": {
                "mode": "pinned-image",
                "image_inspected": True,
                "runtime_image_id": RUNTIME_IMAGE,
            },
            "backend": {
                "live_http_callback": True,
                "live_postgres": True,
                "runtime_image_id": RUNTIME_IMAGE,
            },
        },
    )
    cross_manifest_path = reports / "calphad-cross-language-manifest.json"
    cross_manifest_meta = _write_json(
        cross_manifest_path,
        {
            "schema_version": "ultra.calphad.cross-language-report-manifest.v1",
            "report": {
                "path": cross_path.name,
                "sha256": cross_meta["sha256"],
                "size_bytes": cross_meta["size_bytes"],
            },
            "production_live_qualified": True,
            "runtime_image_id": RUNTIME_IMAGE,
            "expected_git_sha": GIT_SHA,
        },
    )
    live_path = reports / "materials-live-trace.json"
    live_meta = _write_json(live_path, {"schema_version": "1", "status": "passed"})

    checkpoint_path = campaign / "state.json"
    checkpoint_meta = _write_json(checkpoint_path, {"campaign_id": "campaign-1"})
    markdown_path = campaign / "results.md"
    markdown_path.write_text("# MatTools results\n", encoding="utf-8")
    markdown_sha = hashlib.sha256(markdown_path.read_bytes()).hexdigest()
    semantic_repairs_sha256 = "1" * 64
    evaluator_build = {
        "adapted_requirements_sha256": "2" * 64,
        "base_image": "python:3.11.8@sha256:" + "3" * 64,
        "candidate_fixture_file_count": gate.EXPECTED_CANDIDATE_FIXTURE_FILE_COUNT,
        "candidate_fixture_manifest_sha256": (gate.EXPECTED_CANDIDATE_FIXTURE_MANIFEST_SHA256),
        "candidate_visible_source_policy": gate.EXPECTED_CANDIDATE_VISIBLE_SOURCE_POLICY,
        "runner_wrapper_sha256": "4" * 64,
        "safe_parser_sha256": "5" * 64,
        "semantic_repairs_path": "scripts/mattools_semantic_repairs.py",
        "semantic_repairs_sha256": semantic_repairs_sha256,
        "strict_shadow_sha256": "6" * 64,
        "supplemental_requirements_sha256": "7" * 64,
        "tool_source_manifest_sha256": "8" * 64,
    }
    evaluator_upstream = {
        "manifest_sha256": "9" * 64,
        "requirements_sha256": "a" * 64,
        "revision": "b" * 40,
    }
    evaluator_platform = {"docker": "linux/arm64"}
    approved_lock = {
        "environment_kind": "reviewed-reconstruction-variant",
        "official_artifact": False,
        "build": evaluator_build,
        "upstream": evaluator_upstream,
        "platform": evaluator_platform,
    }
    evaluator_labels = {
        "io.ultra.mattools.adapted-requirements-sha256": evaluator_build[
            "adapted_requirements_sha256"
        ],
        "io.ultra.mattools.base-image": evaluator_build["base_image"],
        "io.ultra.mattools.environment-kind": approved_lock["environment_kind"],
        "io.ultra.mattools.official-artifact": "false",
        "io.ultra.mattools.snapshot-manifest-sha256": evaluator_upstream["manifest_sha256"],
        "io.ultra.mattools.safe-parser-sha256": evaluator_build["safe_parser_sha256"],
        "io.ultra.mattools.runner-wrapper-sha256": evaluator_build["runner_wrapper_sha256"],
        "io.ultra.mattools.semantic-repairs-sha256": semantic_repairs_sha256,
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
    evaluator_embedded_inputs = {
        "candidate_fixture_file_count": evaluator_build["candidate_fixture_file_count"],
        "candidate_fixture_manifest_sha256": evaluator_build["candidate_fixture_manifest_sha256"],
        "candidate_visible_non_fixture_paths": [],
        "candidate_visible_executable_source_paths": [],
        "candidate_visible_dependency_test_paths": {
            "pymatgen": [],
            "pymatgen-analysis-defects": [],
        },
        "upstream_requirements_sha256": evaluator_upstream["requirements_sha256"],
        "adapted_requirements_sha256": evaluator_build["adapted_requirements_sha256"],
        "supplemental_requirements_sha256": evaluator_build["supplemental_requirements_sha256"],
    }
    mattools_trials = []
    for trial in range(1, 4):
        attempts = [
            _cleanroom_attempt(
                f"run-{trial}-{ordinal}",
                f"thread-{trial}-{ordinal}",
            )
            for ordinal in range(1, gate.PARENTS_PER_TRIAL + 1)
        ]
        mattools_trials.append(
            {
                "trial": trial,
                "status": "complete",
                "runnable": 40,
                "published_runner_runnable": 40,
                "runnable_denominator": gate.PARENTS_PER_TRIAL,
                "scientific_pass": 83,
                "strict_scientific_pass": 83,
                "scientific_denominator": gate.SCIENTIFIC_SUBTASKS_PER_TRIAL,
                "attempts": attempts,
                "evaluator_environment": {
                    "approved_environment_lock": approved_lock,
                    "labels_match_approved_lock": True,
                    "embedded_inputs_match_approved_lock": True,
                    "full_environment_lock_matches": True,
                    "comparable": True,
                    "image_labels": dict(sorted(evaluator_labels.items())),
                    "embedded_inputs": evaluator_embedded_inputs,
                },
            }
        )
    per_trial_counts = [
        {
            "trial": trial["trial"],
            "attempts": len(trial["attempts"]),
            "runnable": trial["runnable"],
            "published_runner_runnable": trial["published_runner_runnable"],
            "scientific_pass": trial["scientific_pass"],
            "strict_scientific_pass": trial["strict_scientific_pass"],
        }
        for trial in mattools_trials
    ]
    mattools_path = campaign / "results.json"
    mattools_meta = _write_json(
        mattools_path,
        {
            "schema_version": "1",
            "campaign_id": "campaign-1",
            "ultra": {"commit": GIT_SHA, "dirty": False},
            "runtime_environment": {
                "image_digest": RUNTIME_IMAGE,
                "operator_declared_model_id": MODEL_ID,
                "operator_declared_provider_id": PROVIDER_ID,
                "observed_model_ids": [MODEL_ID],
                "observed_provider_ids": [PROVIDER_ID],
                "actual_model_provider_provenance_validated": True,
                "evaluation_profile": gate.MATERIALS_CLEANROOM_PROFILE,
            },
            "harness": {
                "semantic_repairs_path": "scripts/mattools_semantic_repairs.py",
                "semantic_repairs_sha256": semantic_repairs_sha256,
            },
            "official_evaluator_environment": {"approved_lock": approved_lock},
            "license_attestation": {
                "accepted": True,
                "use_basis": ("separately_licensed" if separately_licensed else "noncommercial"),
                "use_purpose": LICENSE_PURPOSE,
                "separate_license_evidence_sha256": license_sha,
                "repository_license": "Apache-2.0",
                "dataset_card_license": "CC-BY-NC-4.0",
                "attested_at": "2026-07-11T00:00:00Z",
            },
            "trials": mattools_trials,
            "counts": {
                "runnable": 120,
                "runnable_denominator": 147,
                "runnable_minimum": 118,
                "per_trial_runnable_minimum": 40,
                "scientific_pass": 249,
                "strict_scientific_pass": 249,
                "scientific_denominator": 414,
                "scientific_minimum": 249,
                "per_trial_scientific_minimum": 83,
                "terminal_attempts": 147,
                "expected_attempts_for_configured_run": 147,
            },
            "rates": {
                "function_runnable": 120 / 147,
                "task_success": 249 / 414,
                "strict_task_success": 249 / 414,
            },
            "hard_gates": {
                "three_trial_completeness": True,
                "server_authorized_cleanroom_profile": True,
                "worker_enforced_cleanroom_profile": True,
                "per_trial_mattools_function_runnable": True,
                "per_trial_strict_scientific_task_success": True,
            },
            "promotion": {
                "scope": "MatTools benchmark lane only",
                "passed": True,
                "full_materials_production_ready": False,
                "reasons": [],
            },
        },
    )
    mattools_manifest_path = campaign / "report_manifest.json"
    mattools_manifest_meta = _write_json(
        mattools_manifest_path,
        {
            "schema_version": "2",
            "manifest_kind": "ultra.mattools.report_bundle.v2",
            "campaign_id": "campaign-1",
            "regeneration": {
                "helper": "revalidate_report_bundle",
                "cli_subcommand": "verify-report",
                "comparison": "byte_exact",
                "task_execution_performed": False,
            },
            "results_json": {
                "path": str(mattools_path.absolute()),
                "sha256": mattools_meta["sha256"],
            },
            "results_markdown": {
                "path": str(markdown_path.absolute()),
                "sha256": markdown_sha,
            },
            "checkpoint": {
                "path": str(checkpoint_path.absolute()),
                "sha256": checkpoint_meta["sha256"],
            },
        },
    )

    release_manifest_path = release / "release-manifest.json"
    _write_json(release_manifest_path, _release_manifest())
    release_tarball_path = release / f"ultra-release-{GIT_SHA}.tar.gz"
    _write_release_tarball(release_tarball_path, release_manifest_path.read_bytes())

    inputs = {
        "deterministic_report": deterministic_meta,
        "production_parity_report": parity_meta,
        "calphad_ledger_report": ledger_meta,
        "calphad_cross_language_report": cross_meta,
        "calphad_cross_language_report_manifest": cross_manifest_meta,
        "mattools_report": mattools_meta,
        "mattools_report_manifest": mattools_manifest_meta,
        "live_trace_reports": [live_meta],
    }
    readiness = {
        "schema_version": "1",
        "gate": "materials-production-readiness",
        "scope": "full-materials-production-readiness",
        "status": "candidate_for_attestation",
        "inputs": inputs,
        "expected_provenance": {
            "git_sha": GIT_SHA,
            "domain_image": DOMAIN_IMAGE,
            "runtime_image": RUNTIME_IMAGE,
            "evaluator_image": EVALUATOR_IMAGE,
        },
        "counts": {
            "production_parity": {"scope": "production-full", "passed": True},
            "calphad_ledger": {"passed": True, "tests": 5, "source_files": 4},
            "calphad_cross_language": {
                "passed": True,
                "live_http_callback": True,
                "live_postgres": True,
                "source_files": 5,
            },
            "deterministic": {"passed": True, "total": 13, "skipped": 0},
            "mattools": {
                "runnable": 120,
                "runnable_denominator": 147,
                "runnable_minimum": 118,
                "per_trial_runnable_minimum": 40,
                "scientific_pass": 249,
                "strict_scientific_pass": 249,
                "scientific_denominator": 414,
                "scientific_minimum": 249,
                "per_trial_scientific_minimum": 83,
                "per_trial": per_trial_counts,
                "recomputed_attempts": 147,
            },
            "designated_live_traces": 1,
        },
        "rates": {
            "mattools_function_runnable": 120 / 147,
            "mattools_official_task_success": 249 / 414,
            "mattools_strict_task_success": 249 / 414,
        },
        "hard_gates": {
            "all_decisive_evidence": True,
            "mattools_server_authorized_cleanroom_profile": True,
            "mattools_worker_enforced_cleanroom_profile": True,
            "mattools_per_trial_function_runnable": True,
            "mattools_per_trial_strict_scientific_correctness": True,
        },
        "promotion": {
            "passed": True,
            "evidence_passed": True,
            "attestation_required": True,
            "distribution_ready": False,
            "full_materials_production_ready": False,
            "product_label": "materials science promotion candidate",
            "reasons": [],
        },
    }
    readiness_bytes = _json_bytes(readiness)
    readiness_sha = hashlib.sha256(readiness_bytes).hexdigest()
    readiness_path = reports / f"materials-production-readiness-{readiness_sha}.json"
    readiness_path.write_bytes(readiness_bytes)
    readiness_manifest_path = reports / "materials-production-readiness-manifest.json"
    _write_json(
        readiness_manifest_path,
        {
            "schema_version": "1",
            "report": {
                "path": readiness_path.name,
                "sha256": readiness_sha,
                "size_bytes": len(readiness_bytes),
            },
            "promotion_passed": True,
            "evidence_passed": True,
            "attestation_required": True,
            "full_materials_production_ready": False,
        },
    )

    workflow_file = tmp_path / "materials-production-qualification.yml"
    workflow_file.write_text("name: Materials Production Qualification\n", encoding="utf-8")
    root_manifest = tmp_path / "restricted" / "materials-evidence-root-v1.json"
    envelope = tmp_path / "public" / "materials-release-envelope-v1.json"
    roles = [
        _role_value("readiness_report", readiness_path, evidence_root),
        _role_value("readiness_manifest", readiness_manifest_path, evidence_root),
        _role_value("deterministic_report", deterministic_path, evidence_root),
        _role_value("production_parity_report", parity_path, evidence_root),
        _role_value("calphad_ledger_report", ledger_path, evidence_root),
        _role_value("calphad_cross_language_report", cross_path, evidence_root),
        _role_value("calphad_cross_language_manifest", cross_manifest_path, evidence_root),
        _role_value("mattools_report", mattools_path, evidence_root),
        _role_value("mattools_manifest", mattools_manifest_path, evidence_root),
        _role_value("live_trace:1", live_path, evidence_root),
        _role_value("release_tarball", release_tarball_path, evidence_root),
        _role_value("release_manifest", release_manifest_path, evidence_root),
    ]
    if license_path is not None:
        roles.append(_role_value("license_evidence", license_path, evidence_root))
    argv = [
        "create",
        "--evidence-root",
        str(evidence_root),
        "--evidence-root-manifest",
        str(root_manifest),
        "--envelope",
        str(envelope),
        "--repository",
        REPOSITORY,
        "--repository-id",
        REPOSITORY_ID,
        "--owner-id",
        OWNER_ID,
        "--source-git-sha",
        GIT_SHA,
        "--source-ref",
        "refs/heads/main",
        "--workflow-path",
        ".github/workflows/materials-production-qualification.yml",
        "--workflow-file",
        str(workflow_file),
        "--workflow-signer-digest",
        GIT_SHA,
        "--run-id",
        RUN_ID,
        "--run-attempt",
        str(RUN_ATTEMPT),
        "--environment",
        "materials-production-qualification",
        "--event-name",
        "workflow_dispatch",
        "--runtime-oci-digest",
        RUNTIME_OCI,
        "--runtime-config-id",
        RUNTIME_IMAGE,
        "--domain-image-id",
        DOMAIN_IMAGE,
        "--evaluator-image-id",
        EVALUATOR_IMAGE,
        "--license-basis",
        "separately_licensed" if separately_licensed else "noncommercial",
        "--license-purpose",
        LICENSE_PURPOSE,
        "--model-identity",
        MODEL_ID,
        "--provider-identity",
        PROVIDER_ID,
        "--restricted-store-locator-sha256",
        "f" * 64,
    ]
    if license_sha is not None:
        argv.extend(("--license-evidence-sha256", license_sha))
    for role in roles:
        argv.extend(("--role", role))
    create_args = gate.build_parser().parse_args(argv)
    return {
        "evidence_root": evidence_root,
        "root_manifest": root_manifest,
        "envelope": envelope,
        "workflow_file": workflow_file,
        "create_args": create_args,
        "roles": roles,
        "readiness_path": readiness_path,
        "readiness_manifest_path": readiness_manifest_path,
        "release_tarball_path": release_tarball_path,
        "mattools_path": mattools_path,
        "parity_path": parity_path,
        "ledger_path": ledger_path,
        "cross_path": cross_path,
        "mattools_manifest_path": mattools_manifest_path,
    }


@pytest.fixture
def valid_fixture(tmp_path: Path) -> dict[str, Any]:
    fixture = _build_fixture(tmp_path)
    gate.create_release_envelope(fixture["create_args"])
    return fixture


def _verification_args(fixture: dict[str, Any], bundle: Path, output: Path) -> argparse.Namespace:
    workflow_sha = hashlib.sha256(fixture["workflow_file"].read_bytes()).hexdigest()
    return gate.build_parser().parse_args(
        [
            "verify-attestation",
            "--evidence-root",
            str(fixture["evidence_root"]),
            "--evidence-root-manifest",
            str(fixture["root_manifest"]),
            "--envelope",
            str(fixture["envelope"]),
            "--bundle",
            str(bundle),
            "--output",
            str(output),
            "--repository",
            REPOSITORY,
            "--repository-id",
            REPOSITORY_ID,
            "--owner-id",
            OWNER_ID,
            "--signer-repo",
            REPOSITORY,
            "--signer-workflow",
            f"{REPOSITORY}/.github/workflows/materials-production-qualification.yml",
            "--signer-digest",
            GIT_SHA,
            "--source-digest",
            GIT_SHA,
            "--source-ref",
            "refs/heads/main",
            "--expected-run-id",
            RUN_ID,
            "--expected-run-attempt",
            str(RUN_ATTEMPT),
            "--expected-environment",
            "materials-production-qualification",
            "--expected-event-name",
            "workflow_dispatch",
            "--expected-workflow-sha256",
            workflow_sha,
        ]
    )


def _gh_result(envelope: Path) -> list[dict[str, Any]]:
    workflow_identity = (
        "https://github.com/amilworks/ultra/.github/workflows/"
        "materials-production-qualification.yml@refs/heads/main"
    )
    envelope_sha = hashlib.sha256(envelope.read_bytes()).hexdigest()
    certificate = {
        "subjectAlternativeName": workflow_identity,
        "issuer": "https://token.actions.githubusercontent.com",
        "runnerEnvironment": "github-hosted",
        "sourceRepositoryURI": "https://github.com/amilworks/ultra",
        "sourceRepositoryDigest": GIT_SHA,
        "sourceRepositoryRef": "refs/heads/main",
        "sourceRepositoryIdentifier": REPOSITORY_ID,
        "sourceRepositoryOwnerURI": "https://github.com/amilworks",
        "sourceRepositoryOwnerIdentifier": OWNER_ID,
        "buildSignerURI": workflow_identity,
        "buildSignerDigest": GIT_SHA,
        "buildConfigURI": workflow_identity,
        "buildConfigDigest": GIT_SHA,
        "buildTrigger": "workflow_dispatch",
        "runInvocationURI": (
            f"https://github.com/amilworks/ultra/actions/runs/{RUN_ID}/attempts/{RUN_ATTEMPT}"
        ),
        "sourceRepositoryVisibilityAtSigning": "public",
        "githubWorkflowRepository": REPOSITORY,
        "githubWorkflowSHA": GIT_SHA,
        "githubWorkflowRef": "refs/heads/main",
        "githubWorkflowTrigger": "workflow_dispatch",
    }
    return [
        {
            "attestation": {},
            "verificationResult": {
                "signature": {"certificate": certificate},
                "verifiedTimestamps": [{"type": "Tlog", "timestamp": "2026-07-11T00:00:00Z"}],
                "statement": {
                    "_type": "https://in-toto.io/Statement/v1",
                    "predicateType": "https://slsa.dev/provenance/v1",
                    "subject": [
                        {
                            "name": envelope.name,
                            "digest": {"sha256": envelope_sha},
                        }
                    ],
                    "predicate": {},
                },
            },
        }
    ]


def test_create_is_deterministic_and_candidate_only(valid_fixture: dict[str, Any]) -> None:
    root_before = valid_fixture["root_manifest"].read_bytes()
    envelope_before = valid_fixture["envelope"].read_bytes()

    gate.create_release_envelope(valid_fixture["create_args"])

    assert valid_fixture["root_manifest"].read_bytes() == root_before
    assert valid_fixture["envelope"].read_bytes() == envelope_before
    envelope = json.loads(envelope_before)
    assert envelope["claim"] == {
        "status": "candidate_for_attestation",
        "evidence_passed": True,
        "attestation_required": True,
        "distribution_ready": False,
        "full_materials_production_ready": False,
    }
    serialized = envelope_before.decode()
    assert MODEL_ID not in serialized
    assert PROVIDER_ID not in serialized
    assert LICENSE_PURPOSE not in serialized


@pytest.mark.parametrize("mutation", ["tamper", "extra", "missing"])
def test_root_verification_rejects_tree_mutation(
    valid_fixture: dict[str, Any], mutation: str
) -> None:
    if mutation == "tamper":
        valid_fixture["mattools_path"].write_bytes(
            valid_fixture["mattools_path"].read_bytes() + b"tampered"
        )
    elif mutation == "extra":
        (valid_fixture["evidence_root"] / "unexpected.bin").write_bytes(b"extra")
    else:
        valid_fixture["mattools_path"].unlink()

    with pytest.raises(gate.PromotionEnvelopeError):
        gate.verify_evidence_root(valid_fixture["evidence_root"], valid_fixture["root_manifest"])


def test_root_verification_rejects_symlink(valid_fixture: dict[str, Any]) -> None:
    (valid_fixture["evidence_root"] / "link").symlink_to(valid_fixture["mattools_path"])
    with pytest.raises(gate.PromotionEnvelopeError, match="symlink"):
        gate.verify_evidence_root(valid_fixture["evidence_root"], valid_fixture["root_manifest"])


def test_root_verification_rejects_hardlinks(valid_fixture: dict[str, Any]) -> None:
    hardlink = valid_fixture["evidence_root"] / "hardlink"
    hardlink.hardlink_to(valid_fixture["mattools_path"])
    with pytest.raises(gate.PromotionEnvelopeError, match="hard-linked"):
        gate.verify_evidence_root(valid_fixture["evidence_root"], valid_fixture["root_manifest"])


@pytest.mark.parametrize(
    "roles",
    [
        ["readiness_report=../escape.json"],
        ["readiness_report=/absolute.json"],
        ["readiness_report=a.json", "readiness_report=b.json"],
        ["live_trace:2=trace.json"],
    ],
)
def test_role_parser_rejects_unsafe_or_incomplete_roles(roles: list[str]) -> None:
    with pytest.raises(gate.PromotionEnvelopeError):
        gate._parse_roles(roles)


def test_blocked_readiness_cannot_create_envelope(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    report = json.loads(fixture["readiness_path"].read_text())
    report["status"] = "blocked"
    fixture["readiness_path"].write_bytes(_json_bytes(report))
    with pytest.raises(gate.PromotionEnvelopeError):
        gate.create_release_envelope(fixture["create_args"])
    assert not fixture["envelope"].exists()


def _validate_fixture_mattools_identity(fixture: dict[str, Any]) -> dict[str, Any]:
    report = json.loads(fixture["mattools_path"].read_text(encoding="utf-8"))
    return gate._validate_mattools_identity(
        report,
        source_git_sha=GIT_SHA,
        expected_runtime_image=RUNTIME_IMAGE,
        license_basis="noncommercial",
        license_purpose=LICENSE_PURPOSE,
        license_evidence_sha256=None,
        model_identity=MODEL_ID,
        provider_identity=PROVIDER_ID,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report["trials"][0]["attempts"][0]["trace_summary"].update(
            server_cleanroom_profile_attested=False
        ),
        lambda report: report["trials"][0]["attempts"][0]["cleanroom_binding"].update(valid=False),
        lambda report: report["trials"][0]["attempts"][0]["trace_summary"][
            "worker_cleanroom_attestations"
        ][0]["payload"].update(goal_sha256="0" * 64),
    ],
)
def test_mattools_identity_recomputes_server_and_worker_cleanroom(
    tmp_path: Path,
    mutation: Any,
) -> None:
    fixture = _build_fixture(tmp_path)
    report = json.loads(fixture["mattools_path"].read_text(encoding="utf-8"))
    mutation(report)
    fixture["mattools_path"].write_bytes(_json_bytes(report))

    with pytest.raises(gate.PromotionEnvelopeError, match="clean-room proof"):
        _validate_fixture_mattools_identity(fixture)


def test_mattools_identity_recomputes_per_trial_floor_despite_aggregate_pass(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    report = json.loads(fixture["mattools_path"].read_text(encoding="utf-8"))
    report["trials"][0]["runnable"] = 39
    report["counts"]["runnable"] = 119
    report["rates"]["function_runnable"] = 119 / 147
    fixture["mattools_path"].write_bytes(_json_bytes(report))

    with pytest.raises(gate.PromotionEnvelopeError, match="score floor"):
        _validate_fixture_mattools_identity(fixture)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report["trials"][0]["evaluator_environment"]["image_labels"].update(
            {"io.ultra.mattools.candidate-visible-source-policy": "full-source"}
        ),
        lambda report: report["harness"].update(semantic_repairs_sha256="0" * 64),
        lambda report: report["official_evaluator_environment"]["approved_lock"]["build"].update(
            candidate_fixture_file_count=140
        ),
        lambda report: report["trials"][0]["evaluator_environment"]["embedded_inputs"][
            "candidate_visible_dependency_test_paths"
        ]["pymatgen"].append("pymatgen/tests/test_hidden.py"),
    ],
)
def test_mattools_identity_binds_fixture_only_labels_and_semantic_repairs(
    tmp_path: Path,
    mutation: Any,
) -> None:
    fixture = _build_fixture(tmp_path)
    report = json.loads(fixture["mattools_path"].read_text(encoding="utf-8"))
    mutation(report)
    fixture["mattools_path"].write_bytes(_json_bytes(report))

    with pytest.raises(gate.PromotionEnvelopeError):
        _validate_fixture_mattools_identity(fixture)


def test_readiness_candidate_requires_named_cleanroom_and_per_trial_gates(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    report = json.loads(fixture["readiness_path"].read_text(encoding="utf-8"))
    report["hard_gates"].pop("mattools_worker_enforced_cleanroom_profile")

    with pytest.raises(gate.PromotionEnvelopeError, match="clean-room or per-trial"):
        gate._validate_readiness_candidate(report)


def test_readiness_candidate_rejects_legacy_counts_without_per_trial_contract(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    report = json.loads(fixture["readiness_path"].read_text(encoding="utf-8"))
    report["counts"]["mattools"].pop("per_trial")
    report["counts"]["mattools"].pop("per_trial_runnable_minimum")

    with pytest.raises(gate.PromotionEnvelopeError):
        gate._validate_readiness_candidate(report)


def test_private_identity_values_are_derived_from_retained_mattools_report(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    args = fixture["create_args"]
    args.license_purpose = None
    args.model_identity = None
    args.provider_identity = None

    _, envelope = gate.create_release_envelope(args)

    assert (
        envelope["license"]["use_purpose_sha256"]
        == hashlib.sha256(LICENSE_PURPOSE.encode()).hexdigest()
    )
    assert (
        envelope["runtime_identity"]["model_identity_sha256"]
        == hashlib.sha256(MODEL_ID.encode()).hexdigest()
    )
    assert (
        envelope["runtime_identity"]["provider_identity_sha256"]
        == hashlib.sha256(PROVIDER_ID.encode()).hexdigest()
    )


def test_separate_license_evidence_is_closed_and_publicly_hashed(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path, separately_licensed=True)

    manifest, envelope = gate.create_release_envelope(fixture["create_args"])

    roles = {role["name"]: role for role in manifest["roles"]}
    assert "license_evidence" in roles
    assert envelope["license"]["use_basis"] == "separately_licensed"
    assert (
        envelope["license"]["separate_license_evidence_sha256"]
        == roles["license_evidence"]["sha256"]
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_identity", "different-model", "model identity"),
        ("provider_identity", "different-provider", "provider identity"),
        ("license_basis", "separately_licensed", "license basis"),
    ],
)
def test_private_identity_or_license_mismatch_blocks_candidate(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    fixture = _build_fixture(tmp_path)
    setattr(fixture["create_args"], field, value)

    with pytest.raises(gate.PromotionEnvelopeError, match=message):
        gate.create_release_envelope(fixture["create_args"])


def test_separate_license_digest_mismatch_blocks_candidate(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path, separately_licensed=True)
    fixture["create_args"].license_evidence_sha256 = "0" * 64

    with pytest.raises(gate.PromotionEnvelopeError, match="license evidence"):
        gate.create_release_envelope(fixture["create_args"])


def test_readiness_report_must_keep_content_addressed_filename(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    old_path = fixture["readiness_path"]
    wrong_path = old_path.with_name("materials-production-readiness.json")
    old_path.rename(wrong_path)
    manifest = json.loads(fixture["readiness_manifest_path"].read_text())
    manifest["report"]["path"] = wrong_path.name
    fixture["readiness_manifest_path"].write_bytes(_json_bytes(manifest))
    old_role = _role_value("readiness_report", old_path, fixture["evidence_root"])
    new_role = _role_value("readiness_report", wrong_path, fixture["evidence_root"])
    fixture["create_args"].role = [
        new_role if role == old_role else role for role in fixture["create_args"].role
    ]

    with pytest.raises(gate.PromotionEnvelopeError, match="content-addressed"):
        gate.create_release_envelope(fixture["create_args"])


def test_deterministic_total_below_fixed_minimum_blocks_candidate(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    old_path = fixture["readiness_path"]
    report = json.loads(old_path.read_text())
    report["counts"]["deterministic"]["total"] = 12
    payload = _json_bytes(report)
    digest = hashlib.sha256(payload).hexdigest()
    new_path = old_path.with_name(f"materials-production-readiness-{digest}.json")
    new_path.write_bytes(payload)
    old_path.unlink()
    manifest = json.loads(fixture["readiness_manifest_path"].read_text())
    manifest["report"] = {
        "path": new_path.name,
        "sha256": digest,
        "size_bytes": len(payload),
    }
    fixture["readiness_manifest_path"].write_bytes(_json_bytes(manifest))
    old_role = _role_value("readiness_report", old_path, fixture["evidence_root"])
    new_role = _role_value("readiness_report", new_path, fixture["evidence_root"])
    fixture["create_args"].role = [
        new_role if role == old_role else role for role in fixture["create_args"].role
    ]

    with pytest.raises(gate.PromotionEnvelopeError, match="deterministic total count"):
        gate.create_release_envelope(fixture["create_args"])


def test_release_tarball_link_member_blocks_candidate(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    tarball = fixture["release_tarball_path"]
    with tarfile.open(tarball, mode="w:gz") as archive:
        link = tarfile.TarInfo(f"{GIT_SHA}/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "release-manifest.json"
        archive.addfile(link)

    with pytest.raises(gate.PromotionEnvelopeError, match="link, device"):
        gate.create_release_envelope(fixture["create_args"])


@pytest.mark.parametrize(
    ("path_key", "mutation"),
    [
        ("parity_path", ("evidence_bundle", "promotable", False)),
        ("ledger_path", (None, "production_database_used", True)),
        ("cross_path", ("backend", "live_postgres", False)),
        ("mattools_manifest_path", ("results_json", "sha256", "0" * 64)),
    ],
)
def test_decisive_lane_forgery_blocks_candidate(
    tmp_path: Path, path_key: str, mutation: tuple[str | None, str, Any]
) -> None:
    fixture = _build_fixture(tmp_path)
    path = fixture[path_key]
    payload = json.loads(path.read_text())
    parent, key, value = mutation
    target = payload if parent is None else payload[parent]
    target[key] = value
    path.write_bytes(_json_bytes(payload))

    with pytest.raises(gate.PromotionEnvelopeError):
        gate.create_release_envelope(fixture["create_args"])


@pytest.mark.parametrize(
    "payload",
    [
        {"password": "hidden"},
        {"nested": {"dsn": "postgresql://user:pass@db.example/test"}},
        {"value": "Bearer abc.def.ghi"},
        {"value": "-----BEGIN PRIVATE KEY-----"},
        {"value": "sk-proj-abcdefghijklmnopqrstuvwxyz"},
        {"value": "https://example.test/file?token=not-public"},
        {"value": "AKIAABCDEFGHIJKLMNOP"},
    ],
)
def test_secret_scanner_rejects_sensitive_public_fields(payload: dict[str, Any]) -> None:
    with pytest.raises(gate.PromotionEnvelopeError):
        gate._assert_secret_free(payload)


def test_successful_exact_attestation_is_only_full_ready_path(
    valid_fixture: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "final-verification.json"
    args = _verification_args(valid_fixture, bundle, output)
    observed: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(_gh_result(valid_fixture["envelope"])).encode(),
            stderr=b"",
        )

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    result = gate.verify_attestation(args)

    assert result["decision"] == {
        "distribution_ready": True,
        "full_materials_production_ready": True,
        "reasons": [],
    }
    assert result["release"]["release_sha"] == GIT_SHA
    assert result["images"]["runtime_oci_manifest_digest"] == RUNTIME_OCI
    assert result["qualification_metrics"]["counts"]["mattools"]["runnable"] == 120
    assert result["qualification_metrics"]["counts"]["mattools"]["strict_scientific_pass"] == 249
    assert json.loads(output.read_text()) == result
    command = observed["command"]
    assert command[:3] == ["gh", "attestation", "verify"]
    assert "--cert-identity" in command
    assert "--signer-workflow" not in command
    assert "--deny-self-hosted-runners" in command
    assert command[command.index("--predicate-type") + 1] == "https://slsa.dev/provenance/v1"
    assert observed["kwargs"]["check"] is False
    assert "shell" not in observed["kwargs"]


def test_failed_attestation_emits_no_final_ready_report(
    valid_fixture: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "final-verification.json"
    args = _verification_args(valid_fixture, bundle, output)

    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 1, stdout=b"", stderr=b"verification failed"
        ),
    )
    with pytest.raises(gate.PromotionEnvelopeError, match="exit status 1"):
        gate.verify_attestation(args)
    assert not output.exists()


@pytest.mark.parametrize(
    ("certificate_field", "wrong_value"),
    [
        ("runnerEnvironment", "self-hosted"),
        ("sourceRepositoryIdentifier", "1"),
        ("sourceRepositoryOwnerIdentifier", "2"),
        ("sourceRepositoryRef", "refs/pull/1/merge"),
        ("buildTrigger", "pull_request"),
        (
            "runInvocationURI",
            "https://github.com/amilworks/ultra/actions/runs/1/attempts/1",
        ),
    ],
)
def test_certificate_policy_rejects_identity_mix_and_match(
    valid_fixture: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    certificate_field: str,
    wrong_value: str,
) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "final-verification.json"
    args = _verification_args(valid_fixture, bundle, output)
    gh_result = _gh_result(valid_fixture["envelope"])
    gh_result[0]["verificationResult"]["signature"]["certificate"][certificate_field] = wrong_value
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(gh_result).encode(), stderr=b""
        ),
    )

    with pytest.raises(gate.PromotionEnvelopeError):
        gate.verify_attestation(args)
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["wrong_subject", "no_timestamp", "multiple"])
def test_attestation_result_shape_fails_closed(
    valid_fixture: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "final-verification.json"
    args = _verification_args(valid_fixture, bundle, output)
    gh_result = _gh_result(valid_fixture["envelope"])
    if mutation == "wrong_subject":
        gh_result[0]["verificationResult"]["statement"]["subject"][0]["digest"]["sha256"] = "0" * 64
    elif mutation == "no_timestamp":
        gh_result[0]["verificationResult"]["verifiedTimestamps"] = []
    else:
        gh_result.append(json.loads(json.dumps(gh_result[0])))
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(gh_result).encode(), stderr=b""
        ),
    )

    with pytest.raises(gate.PromotionEnvelopeError):
        gate.verify_attestation(args)
    assert not output.exists()


def test_custom_verifier_executable_is_rejected_before_execution(
    valid_fixture: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "final-verification.json"
    args = _verification_args(valid_fixture, bundle, output)
    args.gh_command = "attacker-gh"

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("an unreviewed verifier must not execute")

    monkeypatch.setattr(gate.subprocess, "run", unexpected)
    with pytest.raises(gate.PromotionEnvelopeError, match="reviewed gh"):
        gate.verify_attestation(args)
    assert not output.exists()


def test_wrong_run_attempt_fails_before_gh(
    valid_fixture: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "final-verification.json"
    args = _verification_args(valid_fixture, bundle, output)
    args.expected_run_attempt += 1

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("gh must not run for an envelope policy mismatch")

    monkeypatch.setattr(gate.subprocess, "run", unexpected)
    with pytest.raises(gate.PromotionEnvelopeError, match="run_attempt"):
        gate.verify_attestation(args)
    assert not output.exists()


def test_duplicate_json_key_in_manifest_is_rejected(valid_fixture: dict[str, Any]) -> None:
    original = valid_fixture["root_manifest"].read_text()
    valid_fixture["root_manifest"].write_text(
        original.replace("{", '{"schema_version":"shadow",', 1), encoding="utf-8"
    )
    with pytest.raises(gate.PromotionEnvelopeError, match="duplicate JSON key"):
        gate.verify_evidence_root(valid_fixture["evidence_root"], valid_fixture["root_manifest"])


def test_manifest_inside_evidence_root_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    manifest = root / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(gate.PromotionEnvelopeError, match="outside"):
        gate.verify_evidence_root(root, manifest)


def test_main_never_prints_full_ready_for_create(
    valid_fixture: dict[str, Any], capsys: Any
) -> None:
    args = valid_fixture["create_args"]
    argv = []
    for key, value in vars(args).items():
        if key == "command":
            continue
        if key == "role":
            for role in value:
                argv.extend(("--role", role))
            continue
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        argv.extend((flag, str(value)))
    assert gate.main(["create", *argv]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["full_materials_production_ready"] is False
