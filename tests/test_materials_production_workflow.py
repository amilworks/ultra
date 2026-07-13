from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT / ".github/workflows/materials-production-qualification.yml"
PINNED_ACTION = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")


def _workflow() -> dict[str, Any]:
    # BaseLoader keeps GitHub's `on` key and every expression/value as strings.
    source = WORKFLOW_PATH.read_text(encoding="utf-8")
    _assert_unique_yaml_mapping_keys(source)
    return yaml.load(source, Loader=yaml.BaseLoader)


def _assert_unique_yaml_mapping_keys(source: str) -> None:
    root = yaml.compose(source, Loader=yaml.BaseLoader)
    assert root is not None

    def visit(node: yaml.Node) -> None:
        if isinstance(node, yaml.MappingNode):
            seen: dict[str, int] = {}
            for key, value in node.value:
                rendered_key = str(getattr(key, "value", ""))
                line = key.start_mark.line + 1
                assert rendered_key not in seen, (
                    f"duplicate YAML mapping key {rendered_key!r} at lines "
                    f"{seen.get(rendered_key)} and {line}"
                )
                seen[rendered_key] = line
                visit(key)
                visit(value)
        elif isinstance(node, yaml.SequenceNode):
            for value in node.value:
                visit(value)

    visit(root)


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    steps = job.get("steps")
    assert isinstance(steps, list)
    return steps


def _run_text(job: dict[str, Any]) -> str:
    return "\n".join(str(step.get("run", "")) for step in _steps(job))


def _step_named(job: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [step for step in _steps(job) if step.get("name") == name]
    assert len(matches) == 1
    return matches[0]


def test_workflow_has_unique_yaml_keys_and_single_evaluator_retag() -> None:
    source = WORKFLOW_PATH.read_text(encoding="utf-8")
    _assert_unique_yaml_mapping_keys(source)

    retag = 'docker image tag "${EVALUATOR_IMAGE_REF}" mat-tool-ben'
    assert source.count(retag) == 1


def test_workflow_is_manual_main_only_and_globally_singleton() -> None:
    workflow = _workflow()

    assert set(workflow["on"]) == {"workflow_dispatch"}
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert inputs["source_ref"]["options"] == ["refs/heads/main"]
    assert inputs["source_ref"]["default"] == "refs/heads/main"
    assert set(inputs) == {
        "source_sha",
        "source_ref",
        "domain_image_ref",
        "domain_image_id",
        "runtime_image_ref",
        "runtime_image_id",
        "evaluator_image_ref",
        "evaluator_image_id",
        "operator_public_key_sha256",
        "mattools_license_basis",
    }
    assert len(inputs) == 10
    assert workflow["concurrency"] == {
        "group": "materials-production-qualification",
        "cancel-in-progress": "false",
    }
    assert workflow["permissions"] == {}


def test_qualifier_is_protected_ephemeral_arm64_and_has_no_oidc() -> None:
    qualify = _workflow()["jobs"]["qualify"]

    assert qualify["environment"] == {"name": "materials-production-qualification"}
    assert qualify["runs-on"]["group"] == "materials-production-qualification-ephemeral"
    assert qualify["runs-on"]["labels"] == [
        "self-hosted",
        "Linux",
        "ARM64",
    ]
    assert qualify["permissions"] == {"contents": "read"}
    assert int(qualify["timeout-minutes"]) == 1380
    assert int(qualify["timeout-minutes"]) < 24 * 60
    campaign = _step_named(
        qualify, "Verify evaluator and run exactly three complete MatTools trials"
    )
    assert campaign["env"]["MATTOOLS_CONCURRENCY"] == "8"
    worst_case_seconds = ((147 + 8 - 1) // 8) * 1800 + 2 * 7200
    assert worst_case_seconds < int(qualify["timeout-minutes"]) * 60
    assert "id-token" not in qualify["permissions"]
    assert not any("actions/attest@" in str(step.get("uses", "")) for step in _steps(qualify))

    preflight = _step_named(qualify, "Fail closed on invocation, trust anchor, and runner policy")[
        "run"
    ]
    assert '[[ "${GITHUB_EVENT_NAME}" == "workflow_dispatch" ]]' in preflight
    assert '[[ "${GITHUB_REF}" == "refs/heads/main" ]]' in preflight
    assert '[[ "${GITHUB_SHA}" == "${SOURCE_SHA}" ]]' in preflight
    assert '[[ "${GITHUB_WORKFLOW_SHA}" == "${SOURCE_SHA}" ]]' in preflight
    assert '[[ "${RUNNER_ARCH}" == "ARM64" ]]' in preflight
    assert '[[ "${RUNNER_ENVIRONMENT}" == "self-hosted" ]]' in preflight


def test_operator_key_and_immutable_images_fail_closed_before_execution() -> None:
    qualify = _workflow()["jobs"]["qualify"]
    preflight = _step_named(qualify, "Fail closed on invocation, trust anchor, and runner policy")[
        "run"
    ]
    image_check = _step_named(qualify, "Verify exact preloaded image identities")["run"]

    assert 'key_path="security/release-operator-public.pem"' in preflight
    assert '[[ -f "${key_path}" ]]' in preflight
    assert '[[ ! -L "${key_path}" ]]' in preflight
    assert "git ls-files --error-unmatch" in preflight
    assert '[[ "${observed_key_sha256}" == "${OPERATOR_PUBLIC_KEY_SHA256}" ]]' in preflight
    assert preflight.count("digest_ref_re") >= 5
    assert 'case "${MATTOOLS_LICENSE_BASIS}" in' in preflight
    assert "noncommercial)" in preflight
    assert "separately_licensed)" in preflight
    assert '[[ -z "${MATTOOLS_LICENSE_EVIDENCE_SHA256}" ]]' in preflight
    assert '[[ -z "${MATTOOLS_LICENSE_EVIDENCE_PATH}" ]]' in preflight
    assert '[[ -f "${MATTOOLS_LICENSE_EVIDENCE_PATH}" ]]' in preflight
    assert '[[ "${observed_license_sha256}" == "${MATTOOLS_LICENSE_EVIDENCE_SHA256}" ]]' in (
        preflight
    )
    documentation = (
        ROOT / "backend/deepagents_runtime/MATTOOLS_PROMOTION_GATE.md"
    ).read_text(encoding="utf-8")
    assert "MATTOOLS_LICENSE_EVIDENCE_SHA256=<64 lowercase hex>" in documentation
    assert "without a `sha256:`" in documentation

    for image, image_id in (
        ("DOMAIN_IMAGE_REF", "DOMAIN_IMAGE_ID"),
        ("RUNTIME_IMAGE_REF", "RUNTIME_IMAGE_ID"),
        ("EVALUATOR_IMAGE_REF", "EVALUATOR_IMAGE_ID"),
    ):
        assert f'verify_image "${{{image}}}" "${{{image_id}}}"' in image_check
    assert '[[ "${domain_revision}" == "${SOURCE_SHA}" ]]' in image_check
    assert '[[ "${runtime_revision}" == "${SOURCE_SHA}" ]]' in image_check


def test_runtime_credentials_are_step_scoped_not_job_wide() -> None:
    qualify = _workflow()["jobs"]["qualify"]
    secret_names = {
        "ULTRA_LIVE_TRACE_COOKIE",
        "ULTRA_LIVE_TRACE_AUTHORIZATION",
        "ULTRA_MODEL_ID",
        "ULTRA_PROVIDER_ID",
    }
    assert secret_names.isdisjoint(qualify["env"])

    preflight = _step_named(qualify, "Fail closed on invocation, trust anchor, and runner policy")[
        "env"
    ]
    live_trace = _step_named(
        qualify, "Run designated live CALPHAD trace with retained validation bytes"
    )["env"]
    mattools = _step_named(
        qualify, "Verify evaluator and run exactly three complete MatTools trials"
    )["env"]
    envelope = _step_named(
        qualify, "Create and verify the restricted closure and sanitized envelope"
    )

    assert set(preflight) == {
        "ULTRA_LIVE_TRACE_COOKIE",
        "ULTRA_LIVE_TRACE_AUTHORIZATION",
    }
    assert set(live_trace) == {
        "ULTRA_LIVE_TRACE_COOKIE",
        "ULTRA_LIVE_TRACE_AUTHORIZATION",
    }
    assert secret_names.issubset(mattools)
    assert "env" not in envelope

    model_identity_steps = [
        step["name"]
        for step in _steps(qualify)
        if {"ULTRA_MODEL_ID", "ULTRA_PROVIDER_ID"}.intersection(step.get("env", {}))
    ]
    assert model_identity_steps == [
        "Verify evaluator and run exactly three complete MatTools trials"
    ]


def test_all_required_production_lanes_and_candidate_boundary_are_explicit() -> None:
    qualify = _workflow()["jobs"]["qualify"]
    commands = _run_text(qualify)

    for command in (
        "scripts/build_ultra_release_artifact.sh",
        "make materials-domain-gate",
        "make materials-production-parity",
        "make calphad-ledger-qualification",
        "make calphad-cross-language-qualification",
        "make mattools-evaluator-verify",
        "make mattools-promotion-gate",
        "make materials-production-readiness",
    ):
        assert command in commands

    assert "LEDGER_PG_CONTAINER" in commands
    assert "CROSS_LANGUAGE_PG_CONTAINER" in commands
    assert "ultra_calphad_ledger_qualification" in commands
    assert "ultra_calphad_cross_language_qualification" in commands
    assert commands.count("docker run --detach --pull always") == 2
    assert commands.count("--tmpfs /var/lib/postgresql:rw,nosuid,nodev,size=4g") == 2
    assert "--tmpfs /var/lib/postgresql/data:" not in commands
    assert "docker container rm --force --volumes" in commands

    assert "--suggested-domain materials" in commands
    assert "--require-materials-quality" in commands
    assert "--materials-performance" in commands
    assert "--require-materials-performance" in commands
    assert "--materials-max-model-calls 8" in commands
    assert "--materials-max-input-tokens 250000" in commands
    assert "--materials-max-tool-calls 12" in commands
    assert "--materials-max-input-amplification 6.0" in commands
    assert "--verify-downloads" in commands
    assert "--require-downloads" in commands
    assert "--materials-evidence-dir" in commands
    assert "--upload-calphad-manifest" in commands
    assert "T=1173 K" in commands
    assert "X(CO)=0.26" in commands
    assert "X(W)=0.065" in commands
    assert "AL4W, AL5CO2, and BCC_B2" in commands
    assert "all 18 declared phases" in commands

    mattools_step = _step_named(
        qualify, "Verify evaluator and run exactly three complete MatTools trials"
    )["run"]
    assert 'mattools_qualification_log="${EVIDENCE_ROOT}/operator-logs/' in (mattools_step)
    assert '} > "${mattools_qualification_log}" 2>&1' in mattools_step
    assert "tee " not in mattools_step
    assert 'len(report["trials"]) == 3' in mattools_step
    assert 'report["hard_gates"]["three_trial_completeness"] is True' in mattools_step

    assert 'report["status"] == "candidate_for_attestation"' in commands
    assert 'report["promotion"]["evidence_passed"] is True' in commands
    assert 'report["promotion"]["attestation_required"] is True' in commands
    assert 'report["promotion"]["distribution_ready"] is False' in commands
    assert 'report["promotion"]["full_materials_production_ready"] is False' in commands


def test_closure_is_created_at_final_store_path_and_never_relocated() -> None:
    qualify = _workflow()["jobs"]["qualify"]
    job_env = qualify["env"]
    commands = _run_text(qualify)

    assert job_env["EVIDENCE_ROOT"].startswith(
        "${{ vars.MATERIALS_RESTRICTED_EVIDENCE_STORE_ROOT }}/"
    )
    assert job_env["EVIDENCE_ROOT"].endswith("/evidence")
    assert job_env["EVIDENCE_ROOT_MANIFEST"].endswith("/materials-evidence-root-v1.json")
    assert 'cp -a "${EVIDENCE_ROOT}"' not in commands
    assert "--restricted-store-locator-sha256" in commands
    assert '--restricted-store-locator "' not in commands

    for expected in (
        "scripts/materials_promotion_envelope.py create",
        "scripts/materials_promotion_envelope.py verify-root",
        '--role "readiness_report=',
        '--role "deterministic_report=',
        '--role "production_parity_report=',
        '--role "calphad_ledger_report=',
        '--role "calphad_cross_language_report=',
        '--role "mattools_report=',
        '--role "live_trace:1=',
        '--role "release_tarball=',
        'license_role_args+=(--role "license_evidence=',
        "license_evidence_args+=(",
        "--license-evidence-sha256",
    ):
        assert expected in commands

    create_step = _step_named(
        qualify, "Create and verify the restricted closure and sanitized envelope"
    )["run"]
    assert "--license-purpose" not in create_step
    assert "--model-identity" not in create_step
    assert "--provider-identity" not in create_step
    assert '"${license_evidence_args[@]}"' in create_step
    assert 'chmod -R a-w "${ARCHIVE_ROOT}"' in commands


def test_only_sanitized_envelope_bundle_and_verdict_cross_job_boundaries() -> None:
    jobs = _workflow()["jobs"]
    uploads = {
        step["name"]: step
        for job in jobs.values()
        for step in _steps(job)
        if "actions/upload-artifact@" in step.get("uses", "")
    }
    assert set(uploads) == {
        "Transfer only the sanitized release envelope",
        "Transfer only the safe envelope attestation bundle",
        "Transfer only the sanitized final verification report",
    }
    assert uploads["Transfer only the sanitized release envelope"]["with"]["path"].endswith(
        "/public/materials-release-envelope-v1.json"
    )
    assert uploads["Transfer only the safe envelope attestation bundle"]["with"]["path"] == (
        "${{ steps.attest-envelope.outputs.bundle-path }}"
    )
    assert uploads["Transfer only the sanitized final verification report"]["with"][
        "path"
    ].endswith("/materials-production-verdict.json")

    for upload in uploads.values():
        assert upload["with"]["if-no-files-found"] == "error"
        assert "if" not in upload
        path = upload["with"]["path"]
        assert "EVIDENCE_ROOT" not in path
        assert "MATTOOLS" not in path
        assert "evidence-root" not in path
        assert "report_manifest" not in path

    downloads = [
        step["with"]["name"]
        for job in jobs.values()
        for step in _steps(job)
        if "actions/download-artifact@" in step.get("uses", "")
    ]
    assert (
        downloads.count("materials-release-envelope-${{ github.run_id }}-${{ github.run_attempt }}")
        == 2
    )
    assert (
        "materials-envelope-attestation-bundle-${{ github.run_id }}-${{ github.run_attempt }}"
        in downloads
    )
    assert (
        "materials-production-verdict-${{ github.run_id }}-${{ github.run_attempt }}" in downloads
    )

    transfer_check = _step_named(jobs["attest"], "Refuse any unexpected transferred file")["run"]
    assert '[[ "${#transferred[@]}" -eq 1 ]]' in transfer_check
    assert '[[ "${transferred[0]}" == "${PUBLIC_ENVELOPE}" ]]' in transfer_check


def test_job_topology_has_independent_protected_verifier() -> None:
    jobs = _workflow()["jobs"]
    assert set(jobs) == {"qualify", "attest", "verify", "attest-verdict"}
    assert jobs["attest"]["needs"] == "qualify"
    assert jobs["verify"]["needs"] == "attest"
    assert jobs["attest-verdict"]["needs"] == "verify"

    verify = jobs["verify"]
    assert verify["environment"] == {"name": "materials-production-verification"}
    assert verify["runs-on"] == {
        "group": "materials-production-verification",
        "labels": ["self-hosted", "Linux", "ARM64"],
    }
    assert verify["permissions"] == {"contents": "read"}
    assert "id-token" not in verify["permissions"]
    assert not any("actions/attest@" in step.get("uses", "") for step in _steps(verify))

    command = _step_named(verify, "Revalidate the sealed closure and exact GitHub attestation")
    assert command["env"] == {"GH_TOKEN": "${{ github.token }}"}
    command = command["run"]
    assert '[[ ! -w "${EVIDENCE_ROOT}" ]]' in command
    assert '[[ ! -w "${EVIDENCE_ROOT_MANIFEST}" ]]' in command
    assert 'findmnt --noheadings --output OPTIONS --target "${EVIDENCE_ROOT}"' in command
    assert "*,ro,*)" in command
    assert "scripts/mattools_promotion_gate.py verify-report" in command
    assert '--report-manifest "${EVIDENCE_ROOT}/mattools/report_manifest.json"' in command
    assert "scripts/materials_readiness_gate.py" in command
    assert '--deterministic-report "${EVIDENCE_ROOT}/deterministic/materials-domain-gate.json"' in (
        command
    )
    assert '--mattools-report "${EVIDENCE_ROOT}/mattools/results.json"' in command
    assert '--live-trace "${EVIDENCE_ROOT}/live-trace/materials-live-trace.json"' in command
    assert 'report["status"] == "candidate_for_attestation"' in command
    assert "all(value is True for value in hard_gates.values())" in command
    assert 'report["promotion"]["full_materials_production_ready"] is False' in command
    for executable in (
        "git",
        "go",
        "gh",
        "findmnt",
        "openssl",
        "python3",
        "sha256sum",
        "uv",
    ):
        assert f"command -v {executable} >/dev/null" in command
    assert command.index("mattools_promotion_gate.py verify-report") < command.index(
        "materials_promotion_envelope.py verify-attestation"
    )
    assert command.index("materials_readiness_gate.py") < command.index(
        "materials_promotion_envelope.py verify-attestation"
    )
    assert "scripts/materials_promotion_envelope.py verify-attestation" in command
    for argument in (
        '--repository "${EXPECTED_REPOSITORY}"',
        '--repository-id "${EXPECTED_REPOSITORY_ID}"',
        '--owner-id "${EXPECTED_OWNER_ID}"',
        '--signer-repo "${EXPECTED_REPOSITORY}"',
        '--signer-workflow "${EXPECTED_REPOSITORY}/${WORKFLOW_PATH}"',
        '--signer-digest "${SOURCE_SHA}"',
        '--source-digest "${SOURCE_SHA}"',
        '--source-ref "${SOURCE_REF}"',
        '--expected-run-id "${GITHUB_RUN_ID}"',
        '--expected-run-attempt "${GITHUB_RUN_ATTEMPT}"',
        "--expected-environment materials-production-qualification",
        "--expected-event-name workflow_dispatch",
        '--expected-workflow-sha256 "${workflow_sha256}"',
        "--cert-oidc-issuer https://token.actions.githubusercontent.com",
    ):
        assert argument in command
    assert '--output "${verdict}"' in command
    assert "${EVIDENCE_ROOT}/materials-production-verdict" not in command


def test_fresh_hosted_attestation_jobs_have_exact_permissions_and_pin() -> None:
    jobs = _workflow()["jobs"]
    expected_permissions = {
        "contents": "read",
        "id-token": "write",
        "attestations": "write",
        "artifact-metadata": "write",
    }
    for name in ("attest", "attest-verdict"):
        job = jobs[name]
        assert job["runs-on"] == "ubuntu-latest"
        assert job["permissions"] == expected_permissions

    envelope_attestation = _step_named(
        jobs["attest"], "Create GitHub Sigstore provenance for the sanitized envelope"
    )
    verdict_attestation = _step_named(
        jobs["attest-verdict"], "Attest the sanitized final materials verdict"
    )
    for attestation_step in (envelope_attestation, verdict_attestation):
        assert attestation_step["uses"] == "actions/attest@a1948c3f048ba23858d222213b7c278aabede763"
    assert envelope_attestation["with"]["subject-path"].endswith(
        "/materials-release-envelope/materials-release-envelope-v1.json"
    )
    assert verdict_attestation["with"]["subject-path"].endswith(
        "/materials-production-verdict/materials-production-verdict.json"
    )

    for job in jobs.values():
        for step in _steps(job):
            action = step.get("uses")
            if action:
                assert PINNED_ACTION.fullmatch(action), action


def test_workflow_cannot_deploy_or_inject_dispatch_inputs_into_shell() -> None:
    workflow = _workflow()
    source = WORKFLOW_PATH.read_text(encoding="utf-8")
    commands = "\n".join(_run_text(job) for job in workflow["jobs"].values())

    assert "make deploy" not in commands
    assert "scripts/deploy_" not in commands
    assert "kubectl apply" not in commands
    assert "docker push" not in commands
    assert "full_materials_production_ready=true" not in commands
    assert '"full_materials_production_ready": true' not in commands
    assert "full_materials_production_ready = True" not in commands

    # Dispatch strings are first placed in YAML env fields, then consumed as
    # quoted shell variables. Direct expression substitution in `run` blocks is
    # a command-injection boundary and is forbidden.
    assert "${{ inputs." not in commands
    assert source.count("actions/upload-artifact@") == 3
