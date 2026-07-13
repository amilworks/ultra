from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest
from ultra_deepagents.live_trace import (
    MATERIALS_SKILL_NAMES,
    _apply_upload_calphad_manifests,
    _calphad_upload_metadata_from_manifest,
    _inspect_materials_validation_artifact,
    evaluate_materials_trace_performance,
    evaluate_materials_trace_quality,
    summarize_run_trace,
)
from ultra_deepagents.materials.trace_binding import typed_materials_result_binding
from ultra_deepagents.materials.validation import (
    EvidenceArtifact,
    ValidationCheck,
    ValidationOutcome,
    assess_scientific_status,
)


def _record_payload() -> dict:
    assessment = assess_scientific_status(
        run_status="succeeded",
        checks=[
            ValidationCheck(
                validator_id="xrd.fcc_first_peak",
                outcome=ValidationOutcome.PASS,
                observed={"two_theta_deg": 44.5855, "hkl": [1, 1, 1]},
                expected={"two_theta_deg": 44.59, "absolute_tolerance_deg": 0.35},
                units="degree",
                tolerance_rationale="CuKa doublet/library rounding allowance",
                required=True,
                critical=True,
                library_versions={"pymatgen": "2026.5.4"},
                evidence=(
                    EvidenceArtifact(
                        name="xrd_pattern.json",
                        sha256="b" * 64,
                        path="/outputs/xrd_pattern.json",
                        size_bytes=400,
                    ),
                ),
            )
        ],
        required_validator_ids=["xrd.fcc_first_peak"],
    )
    return assessment.to_dict()


class _ArtifactClient:
    def __init__(self, payload: dict, *, exact_bytes: bytes | None = None) -> None:
        self.raw = exact_bytes or json.dumps(payload, sort_keys=True).encode("utf-8")

    def download_artifact(self, artifact_id: str) -> bytes:
        assert artifact_id == "artifact-validation"
        return self.raw


def _artifact(size_bytes: int = 1000) -> dict:
    return {
        "artifact_id": "artifact-validation",
        "kind": "json",
        "path": "outputs/materials_validation.json",
        "mime_type": "application/json",
        "size_bytes": size_bytes,
    }


def _evidence_artifact(*, sha256: str = "b" * 64) -> dict:
    return {
        "artifact_id": "artifact-xrd",
        "kind": "json",
        "path": "outputs/xrd_pattern.json",
        "mime_type": "application/json",
        "size_bytes": 400,
        "sha256": sha256,
    }


def _typed_evidence_artifact(*, operation: str, result_sha256: str) -> dict:
    return {
        "artifact_id": f"typed-{operation}:{result_sha256}",
        "kind": "json",
        "path": f"outputs/typed/{operation}/{result_sha256}.json",
        "mime_type": "application/json",
        "size_bytes": 400,
        "sha256": result_sha256,
    }


def _typed_record_payload(*, operation: str, result_sha256: str) -> dict:
    evidence = EvidenceArtifact(
        name=f"typed {operation} analysis record",
        sha256=result_sha256,
        artifact_id=f"typed-{operation}:{result_sha256}",
        size_bytes=400,
    )
    validator_id = f"materials.bounded_tool.{operation}"
    assessment = assess_scientific_status(
        run_status="succeeded",
        checks=[
            ValidationCheck(
                validator_id=validator_id,
                outcome=ValidationOutcome.PASS,
                observed={"bounded_result": True},
                expected={"bounded_result": True},
                units="1",
                tolerance_rationale="Exact closed-schema typed-result identity check.",
                required=True,
                critical=True,
                library_versions={"ultra-materials-kernel": "1"},
                evidence=(evidence,),
            )
        ],
        required_validator_ids=[validator_id],
    )
    return assessment.to_dict()


def _path_bound_typed_record_payload(
    *,
    operation: str,
    result_sha256: str,
    evidence_path: str,
) -> dict:
    evidence = EvidenceArtifact(
        name=f"typed {operation} content-addressed result",
        sha256=result_sha256,
        path=evidence_path,
        size_bytes=400,
    )
    validator_id = f"materials.bounded_tool.{operation}"
    assessment = assess_scientific_status(
        run_status="succeeded",
        checks=[
            ValidationCheck(
                validator_id=validator_id,
                outcome=ValidationOutcome.PASS,
                observed={"bounded_result": True},
                expected={"bounded_result": True},
                units="1",
                tolerance_rationale="Exact closed content-addressed result identity check.",
                required=True,
                critical=True,
                library_versions={"ultra-materials-kernel": "1"},
                evidence=(evidence,),
            )
        ],
        required_validator_ids=[validator_id],
    )
    return assessment.to_dict()


def _unverified_cp_record_payload(*, result_sha256: str) -> dict:
    operation = "analytical_slip_geometry"
    evidence = EvidenceArtifact(
        name="typed analytical slip geometry record",
        sha256=result_sha256,
        artifact_id=f"typed-{operation}:{result_sha256}",
        size_bytes=400,
    )
    geometry_id = "crystal_plasticity.geometry_invariants"
    phase_binding_id = "crystal_plasticity.phase_structure_assignment_bound"
    assessment = assess_scientific_status(
        run_status="succeeded",
        checks=[
            ValidationCheck(
                validator_id=geometry_id,
                outcome=ValidationOutcome.PASS,
                observed={"canonical_geometry_checks_passed": True},
                expected={"canonical_geometry_checks_passed": True},
                units="1",
                tolerance_rationale="Exact bounded geometry-invariant check.",
                required=True,
                critical=True,
                library_versions={"slip_geometry_reference": "DAMASK-3.1.0-transcription"},
                evidence=(evidence,),
            ),
            ValidationCheck(
                validator_id=phase_binding_id,
                outcome=ValidationOutcome.SKIP,
                observed={"phase_id": "gamma", "caller_declared_structure": "fcc"},
                expected={"independent_phase_structure_binding": True},
                units="1",
                tolerance_rationale=(
                    "A caller declaration cannot independently establish phase identity."
                ),
                required=True,
                critical=True,
                library_versions={"slip_geometry_reference": "DAMASK-3.1.0-transcription"},
                evidence=(evidence,),
            ),
        ],
        required_validator_ids=[geometry_id, phase_binding_id],
    )
    return assessment.to_dict()


_RUNTIME_IMAGE_DIGEST = "sha256:" + "d" * 64


def _write_calphad_upload_fixture(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    upload = tmp_path / "reviewed.tdb"
    upload.write_bytes(b"ELEMENT /- ELECTRON_GAS 0 0 0 !\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "databases": [
                    {
                        "database_id": "reviewed-db",
                        "filename": upload.name,
                        "format": "tdb",
                        "sha256": sha256(upload.read_bytes()).hexdigest(),
                        "size_bytes": upload.stat().st_size,
                        "source_uri": "https://materials.example/reviewed-db",
                        "license_id": "CC0-1.0",
                        "assessment_scope": "Reviewed test assessment.",
                        "reference_state": "SER",
                        "tdb_temperature_limits_K": [298.15, 3000.0],
                        "assessment_pressure_limits_Pa": [101325.0, 101325.0],
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return upload, manifest


def test_calphad_upload_manifest_binds_exact_bytes_and_owner_declarations(
    tmp_path: Path,
) -> None:
    upload, manifest = _write_calphad_upload_fixture(tmp_path)

    metadata, evidence = _calphad_upload_metadata_from_manifest(upload, manifest)

    assert metadata == {
        "calphad": {
            "database_id": "reviewed-db",
            "source": "https://materials.example/reviewed-db",
            "license_id": "CC0-1.0",
            "assessment_scope": "Reviewed test assessment.",
            "reference_state": "SER",
            "tdb_temperature_limits_K": [298.15, 3000.0],
            "assessment_pressure_limits_Pa": [101325.0, 101325.0],
            "declaration_authority": "resource_owner",
        }
    }
    assert evidence["file_sha256"] == sha256(upload.read_bytes()).hexdigest()
    assert evidence["file_size_bytes"] == upload.stat().st_size
    assert evidence["database_format"] == "tdb"
    assert evidence["assessment_pressure_limits_Pa"] == [101325.0, 101325.0]

    upload.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="do not match"):
        _calphad_upload_metadata_from_manifest(upload, manifest)

    upload, manifest = _write_calphad_upload_fixture(tmp_path / "symlink-case")
    linked_upload = tmp_path / "linked.tdb"
    linked_upload.symlink_to(upload)
    with pytest.raises(ValueError, match="non-symlink"):
        _calphad_upload_metadata_from_manifest(linked_upload, manifest)


@pytest.mark.parametrize(
    "pressure_limits",
    [None, [0.0, 101325.0], [101326.0, 101325.0], [101325.0, 1e12 + 1.0]],
)
def test_calphad_upload_manifest_requires_bounded_pressure_scope(
    tmp_path: Path,
    pressure_limits: list[float] | None,
) -> None:
    upload, manifest = _write_calphad_upload_fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    record = payload["databases"][0]
    if pressure_limits is None:
        record.pop("assessment_pressure_limits_Pa")
    else:
        record["assessment_pressure_limits_Pa"] = pressure_limits
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="pressure limits"):
        _calphad_upload_metadata_from_manifest(upload, manifest)


def test_calphad_upload_manifest_format_must_match_readable_suffix(tmp_path: Path) -> None:
    upload, manifest = _write_calphad_upload_fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["databases"][0]["format"] = "dat"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="format does not match"):
        _calphad_upload_metadata_from_manifest(upload, manifest)


def test_apply_upload_calphad_manifest_patches_before_run_selection(tmp_path: Path) -> None:
    upload, manifest = _write_calphad_upload_fixture(tmp_path)

    class Client:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        def patch_resource_metadata(self, file_id: str, metadata: dict) -> dict:
            self.calls.append((file_id, metadata))
            return {"resource_id": file_id, "metadata": metadata}

    client = Client()
    applied = _apply_upload_calphad_manifests(
        client,  # type: ignore[arg-type]
        upload_paths=[upload],
        file_ids=["file-reviewed"],
        manifest_paths=[manifest],
    )

    assert [call[0] for call in client.calls] == ["file-reviewed"]
    assert applied[0]["file_id"] == "file-reviewed"
    assert applied[0]["database_id"] == "reviewed-db"
    with pytest.raises(ValueError, match="correspond"):
        _apply_upload_calphad_manifests(
            client,  # type: ignore[arg-type]
            upload_paths=[upload],
            file_ids=["file-reviewed"],
            manifest_paths=[manifest, manifest],
        )


def _successful_materials_tool_events(
    *,
    execute_exit_code: int | None = 0,
    skill_name: str = "materials-characterization",
) -> list[dict]:
    completed_execute_payload = {
        "tool_name": "execute",
        "tool_call_id": "execute-1",
        "runtime_image_digest": _RUNTIME_IMAGE_DIGEST,
    }
    if execute_exit_code is not None:
        completed_execute_payload["exit_code"] = execute_exit_code
    return [
        {
            "event_kind": "tool_call.started",
            "payload": {
                "tool_name": "read_file",
                "tool_call_id": "read-skill-1",
                "file_path": f"/skills/{skill_name}/SKILL.md",
            },
        },
        {
            "event_kind": "tool_call.completed",
            "payload": {"tool_name": "read_file", "tool_call_id": "read-skill-1"},
        },
        {
            "event_kind": "tool_call.started",
            "payload": {
                "tool_name": "execute",
                "tool_call_id": "execute-1",
                "command": "python /workspace/xrd.py",
                "runtime_image_digest": _RUNTIME_IMAGE_DIGEST,
            },
        },
        {
            "event_kind": "tool_call.completed",
            "payload": completed_execute_payload,
        },
    ]


def _successful_typed_materials_tool_events(
    *,
    tool_name: str = "materials_analyze_crystal_slip",
    operation: str = "analytical_slip_geometry",
    result_sha256: str = "a" * 64,
    validation_sha256: str | None = None,
    skill_name: str = "materials-crystal-plasticity",
) -> list[dict]:
    completed_payload = {
        "tool_name": tool_name,
        "tool_call_id": "cp-typed-1",
        "scientific_operation": operation,
        "result_artifact_sha256": result_sha256,
        "scientific_result_ok": True,
    }
    if validation_sha256 is not None:
        completed_payload["materials_validation_artifact_sha256"] = validation_sha256
    return [
        {
            "event_kind": "tool_call.started",
            "payload": {
                "tool_name": "read_file",
                "tool_call_id": "read-skill-typed",
                "file_path": f"/skills/{skill_name}/SKILL.md",
            },
        },
        {
            "event_kind": "tool_call.completed",
            "payload": {
                "tool_name": "read_file",
                "tool_call_id": "read-skill-typed",
            },
        },
        {
            "event_kind": "tool_call.started",
            "payload": {
                "tool_name": tool_name,
                "tool_call_id": "cp-typed-1",
            },
        },
        {
            "event_kind": "tool_call.completed",
            "payload": completed_payload,
        },
    ]


def test_typed_materials_binding_uses_exact_full_structured_output() -> None:
    result_sha256 = "a" * 64
    validation_sha256 = "c" * 64
    output = json.dumps(
        {
            "ok": True,
            "operation": "analytical_slip_geometry",
            "analysis_artifact": {"sha256": result_sha256},
            "materials_validation_artifact": {"sha256": validation_sha256},
        }
    )

    payload = typed_materials_result_binding(
        "materials_analyze_crystal_slip",
        output,
    )

    assert payload["scientific_operation"] == "analytical_slip_geometry"
    assert payload["result_artifact_sha256"] == result_sha256
    assert payload["materials_validation_artifact_sha256"] == validation_sha256
    assert payload["scientific_result_ok"] is True


def _usage_event(
    usage_event_id: str,
    *,
    input_tokens: int,
    output_tokens: int,
) -> dict:
    return {
        "event_kind": "run.token_usage",
        "payload": {
            "usage_event_id": usage_event_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def _remote_mutation_tool_events(tool_name: str) -> list[dict]:
    return [
        {
            "event_kind": "tool_call.started",
            "payload": {
                "tool_name": tool_name,
                "tool_call_id": f"{tool_name}-1",
            },
        },
        {
            "event_kind": "tool_call.completed",
            "payload": {
                "tool_name": tool_name,
                "tool_call_id": f"{tool_name}-1",
            },
        },
    ]


def test_trace_inspector_recomputes_and_hashes_materials_validation_record():
    summary = _inspect_materials_validation_artifact(
        _ArtifactClient(_record_payload()), [_artifact(), _evidence_artifact()]
    )

    assert summary is not None
    assert summary["valid"] is True
    assert summary["scientific_status"] == "verified"
    assert summary["silent_success"] is False
    assert summary["outcomes"] == {"pass": 1}
    assert summary["evidence_count"] == 1
    assert summary["evidence_verified"] is True
    assert summary["required_validator_ids"] == ["xrd.fcc_first_peak"]
    assert summary["capability_supported"] is True
    assert summary["contradiction_failures"] == []
    assert len(summary["record_sha256"]) == 64
    assert summary["independent"] is False


def test_trace_inspector_retains_exact_content_addressed_validation_bytes(tmp_path):
    client = _ArtifactClient(_record_payload())

    summary = _inspect_materials_validation_artifact(
        client,
        [_artifact(), _evidence_artifact()],
        evidence_dir=tmp_path,
    )

    assert summary is not None
    assert summary["valid"] is True
    retained = tmp_path / f"materials-validation-{summary['record_sha256']}.json"
    assert retained.read_bytes() == client.raw
    assert summary["retained_path"] == str(retained.resolve())
    assert summary["retained_sha256"] == summary["record_sha256"]
    assert summary["retained_size_bytes"] == len(client.raw)


def test_trace_inspector_rejects_self_awarded_verified_status():
    payload = _record_payload()
    payload["checks"][0]["outcome"] = "fail"
    summary = _inspect_materials_validation_artifact(
        _ArtifactClient(payload), [_artifact(), _evidence_artifact()]
    )

    assert summary is not None
    assert summary["valid"] is False
    assert "self-inconsistent" in summary["error"]


def test_trace_inspector_fails_closed_when_canonicalization_raises(monkeypatch):
    def fail_canonicalization(_assessment):
        raise ValueError("non-finite canonical value")

    monkeypatch.setattr(
        "ultra_deepagents.live_trace.canonical_record_json",
        fail_canonicalization,
    )

    summary = _inspect_materials_validation_artifact(
        _ArtifactClient(_record_payload()), [_artifact(), _evidence_artifact()]
    )

    assert summary is not None
    assert summary["valid"] is False
    assert "canonical" in summary["error"].lower()


def test_materials_trace_quality_requires_skill_execution_and_valid_record():
    validation = _inspect_materials_validation_artifact(
        _ArtifactClient(_record_payload()), [_artifact(), _evidence_artifact()]
    )
    turn = summarize_run_trace(
        run={
            "run_id": "run-materials",
            "goal": "Calculate the CuKa XRD pattern for FCC Ni.",
            "status": "succeeded",
            "response_text": "Validated FCC Ni XRD result.",
        },
        events=[*_successful_materials_tool_events(), {"event_kind": "run.completed"}],
        artifacts=[_artifact(), _evidence_artifact()],
        materials_validation=validation,
    )

    result = evaluate_materials_trace_quality({"prompt": turn})
    assert result["passed"] is True
    assert result["score"] == 10.0
    assert result["independent_scientific_verification"] is False
    assert result["signals"]["matched_skill_read_completion"] is True
    assert result["signals"]["matched_execute_completion"] is True
    assert result["signals"]["execute_runtime_image_attested"] is True
    assert result["signals"]["no_failed_tool_terminal"] is True


def test_materials_trace_quality_rejects_unrelated_validation_for_typed_operation():
    validation = _inspect_materials_validation_artifact(
        _ArtifactClient(_record_payload()), [_artifact(), _evidence_artifact()]
    )
    turn = summarize_run_trace(
        run={
            "run_id": "run-cp-typed",
            "goal": "Calculate FCC resolved shear stress with the typed tool.",
            "status": "succeeded",
            "response_text": "Validated typed crystal-plasticity geometry.",
        },
        events=[*_successful_typed_materials_tool_events(), {"event_kind": "run.completed"}],
        artifacts=[_artifact(), _evidence_artifact()],
        materials_validation=validation,
    )

    result = evaluate_materials_trace_quality({"prompt": turn})

    assert result["passed"] is False
    assert result["signals"]["matched_execute_completion"] is False
    assert result["signals"]["matched_typed_scientific_completion"] is True
    assert result["signals"]["typed_validation_binding_ok"] is False
    assert result["signals"]["first_party_scientific_execution"] is False
    assert any("not bound to compatible validation evidence" in issue for issue in result["issues"])


def test_materials_trace_quality_accepts_identity_bound_verified_typed_operation():
    operation = "diffraction_profile_metrics"
    result_sha256 = "e" * 64
    payload = _typed_record_payload(operation=operation, result_sha256=result_sha256)
    client = _ArtifactClient(payload)
    evidence_artifact = _typed_evidence_artifact(
        operation=operation,
        result_sha256=result_sha256,
    )
    validation_artifact = _artifact(size_bytes=len(client.raw))
    validation = _inspect_materials_validation_artifact(
        client,
        [validation_artifact, evidence_artifact],
    )
    assert validation is not None
    turn = summarize_run_trace(
        run={
            "run_id": "run-diffraction-typed",
            "goal": "Calculate bounded diffraction profile metrics.",
            "status": "succeeded",
            "response_text": "Validated typed diffraction metrics.",
        },
        events=[
            *_successful_typed_materials_tool_events(
                tool_name="materials_calculate_diffraction_profile_metrics",
                operation=operation,
                result_sha256=result_sha256,
                validation_sha256=validation["record_sha256"],
                skill_name="materials-characterization-advanced",
            ),
            {"event_kind": "run.completed"},
        ],
        artifacts=[validation_artifact, evidence_artifact],
        materials_validation=validation,
    )

    result = evaluate_materials_trace_quality({"prompt": turn})

    assert result["passed"] is True
    assert result["score"] == 10.0
    assert result["signals"]["typed_validation_binding_ok"] is True
    assert result["signals"]["first_party_scientific_execution"] is True


@pytest.mark.parametrize(
    ("tool_name", "operation", "artifact_key", "evidence_directory", "skill_name"),
    [
        (
            "calphad_run_equilibrium",
            "equilibrium",
            "equilibrium_artifact",
            "calphad/equilibrium",
            "materials-structure-thermo",
        ),
        (
            "calphad_run_scheil",
            "scheil",
            "scheil_artifact",
            "calphad/scheil",
            "materials-processing-kinetics",
        ),
        (
            "materials_transport_coefficients",
            "transport_coefficients",
            "artifact",
            "kinetics/transport_coefficients",
            "materials-processing-kinetics",
        ),
        (
            "materials_run_diffusion_1d",
            "single_phase_diffusion_1d",
            "artifact",
            "kinetics/single_phase_diffusion_1d",
            "materials-processing-kinetics",
        ),
        (
            "materials_run_binary_precipitation_kwn",
            "binary_precipitation_kwn",
            "artifact",
            "kinetics/binary_precipitation_kwn",
            "materials-processing-kinetics",
        ),
    ],
)
def test_calphad_and_kinetics_typed_paths_bind_exact_content_addressed_results(
    tool_name: str,
    operation: str,
    artifact_key: str,
    evidence_directory: str,
    skill_name: str,
) -> None:
    result_sha256 = sha256(operation.encode("utf-8")).hexdigest()
    evidence_path = f"/outputs/{evidence_directory}/{result_sha256}.json"
    emitted_binding = typed_materials_result_binding(
        tool_name,
        json.dumps(
            {
                "ok": True,
                "operation": operation,
                artifact_key: {"sha256": result_sha256},
            }
        ),
    )
    assert emitted_binding == {
        "scientific_operation": operation,
        "result_artifact_sha256": result_sha256,
        "scientific_result_ok": True,
    }

    payload = _path_bound_typed_record_payload(
        operation=operation,
        result_sha256=result_sha256,
        evidence_path=evidence_path,
    )
    client = _ArtifactClient(payload)
    validation_artifact = _artifact(size_bytes=len(client.raw))
    evidence_artifact = {
        "artifact_id": f"artifact-{operation}",
        "kind": "json",
        "path": evidence_path,
        "mime_type": "application/json",
        "size_bytes": 400,
        "sha256": result_sha256,
    }
    validation = _inspect_materials_validation_artifact(
        client,
        [validation_artifact, evidence_artifact],
    )
    assert validation is not None
    turn = summarize_run_trace(
        run={
            "run_id": f"run-{operation}",
            "goal": f"Run bounded {operation}.",
            "status": "succeeded",
            "response_text": "Returned a separately validated bounded result.",
        },
        events=[
            *_successful_typed_materials_tool_events(
                tool_name=tool_name,
                operation=operation,
                result_sha256=result_sha256,
                validation_sha256=validation["record_sha256"],
                skill_name=skill_name,
            ),
            {"event_kind": "run.completed"},
        ],
        artifacts=[validation_artifact, evidence_artifact],
        materials_validation=validation,
    )

    result = evaluate_materials_trace_quality({"prompt": turn})

    assert validation["typed_result_bindings"] == [
        {
            "operation": operation,
            "result_artifact_sha256": result_sha256,
            "validator_ids": [f"materials.bounded_tool.{operation}"],
        }
    ]
    assert result["signals"]["typed_validation_binding_ok"] is True
    assert result["passed"] is True


def test_typed_path_binding_rejects_filename_and_evidence_digest_mismatch() -> None:
    filename_sha256 = "a" * 64
    evidence_sha256 = "b" * 64
    evidence_path = f"/outputs/calphad/equilibrium/{filename_sha256}.json"
    payload = _path_bound_typed_record_payload(
        operation="equilibrium",
        result_sha256=evidence_sha256,
        evidence_path=evidence_path,
    )
    client = _ArtifactClient(payload)
    validation = _inspect_materials_validation_artifact(
        client,
        [
            _artifact(size_bytes=len(client.raw)),
            {
                "artifact_id": "artifact-equilibrium",
                "kind": "json",
                "path": evidence_path,
                "mime_type": "application/json",
                "size_bytes": 400,
                "sha256": evidence_sha256,
            },
        ],
    )

    assert validation is not None
    assert validation["evidence_verified"] is True
    assert validation["typed_result_bindings"] == []


def test_materials_trace_quality_preserves_honest_unverified_cp_status():
    operation = "analytical_slip_geometry"
    result_sha256 = "f" * 64
    validation_payload = _unverified_cp_record_payload(result_sha256=result_sha256)
    client = _ArtifactClient(validation_payload)
    evidence_artifact = _typed_evidence_artifact(
        operation=operation,
        result_sha256=result_sha256,
    )
    validation_artifact = _artifact(size_bytes=len(client.raw))
    validation = _inspect_materials_validation_artifact(
        client,
        [validation_artifact, evidence_artifact],
    )
    assert validation is not None
    turn = summarize_run_trace(
        run={
            "run_id": "run-cp-unverified",
            "goal": "Calculate FCC slip geometry with an opaque phase declaration.",
            "status": "succeeded",
            "response_text": "Geometry completed; phase assignment remains unverified.",
        },
        events=[
            *_successful_typed_materials_tool_events(
                operation=operation,
                result_sha256=result_sha256,
                validation_sha256=validation["record_sha256"],
            ),
            {"event_kind": "run.completed"},
        ],
        artifacts=[validation_artifact, evidence_artifact],
        materials_validation=validation,
    )

    result = evaluate_materials_trace_quality({"prompt": turn})

    assert validation["scientific_status"] == "unverified"
    assert validation["evidence_verified"] is True
    assert result["signals"]["typed_validation_binding_ok"] is True
    assert result["signals"]["first_party_scientific_execution"] is True
    assert result["signals"]["first_party_scientific_record_valid"] is False
    assert result["passed"] is False


def test_materials_trace_quality_does_not_let_typed_success_mask_bad_execute():
    validation = _inspect_materials_validation_artifact(
        _ArtifactClient(_record_payload()), [_artifact(), _evidence_artifact()]
    )
    events = [
        *_successful_typed_materials_tool_events(),
        {
            "event_kind": "tool_call.started",
            "payload": {
                "tool_name": "execute",
                "tool_call_id": "bad-side-path",
                "runtime_image_digest": _RUNTIME_IMAGE_DIGEST,
            },
        },
        {
            "event_kind": "tool_call.completed",
            "payload": {
                "tool_name": "execute",
                "tool_call_id": "bad-side-path",
                "runtime_image_digest": _RUNTIME_IMAGE_DIGEST,
                "exit_code": 2,
            },
        },
    ]
    turn = summarize_run_trace(
        run={
            "run_id": "run-cp-bad-side-path",
            "goal": "Calculate FCC resolved shear stress.",
            "status": "succeeded",
            "response_text": "Claimed typed result despite failed side path.",
        },
        events=events,
        artifacts=[_artifact(), _evidence_artifact()],
        materials_validation=validation,
    )

    result = evaluate_materials_trace_quality({"prompt": turn})

    assert result["passed"] is False
    assert result["signals"]["matched_typed_scientific_completion"] is True
    assert result["signals"]["execute_exit_code_ok"] is False
    assert any("exit code 0" in issue for issue in result["issues"])


def test_materials_trace_performance_dedupes_usage_and_applies_tripwires():
    events = [
        *_successful_typed_materials_tool_events(),
        _usage_event("run-cp:model:1", input_tokens=24_000, output_tokens=800),
        _usage_event("run-cp:model:2", input_tokens=27_000, output_tokens=1_200),
        # Transport redelivery of the first model call must not inflate the totals.
        _usage_event("run-cp:model:1", input_tokens=24_000, output_tokens=800),
    ]
    turn = summarize_run_trace(
        run={
            "run_id": "run-cp-performance",
            "goal": "Calculate FCC resolved shear stress.",
            "status": "succeeded",
            "response_text": "Done.",
        },
        events=events,
        artifacts=[],
        terminal_ms=1_500.0,
    )

    assert turn["token_usage"] == {
        "model_call_count": 2,
        "input_tokens": 51_000,
        "output_tokens": 2_000,
        "total_tokens": 53_000,
        "first_input_tokens": 24_000,
        "last_input_tokens": 27_000,
        "peak_input_tokens": 27_000,
        "input_amplification_vs_peak": 1.888889,
        "duplicate_event_count": 1,
        "conflicting_duplicate_count": 0,
        "invalid_event_count": 0,
    }
    result = evaluate_materials_trace_performance(
        {"prompt": turn},
        max_model_calls_per_turn=3,
        max_input_tokens_per_turn=60_000,
        max_tool_calls_per_turn=3,
        max_input_amplification_vs_peak=2.0,
        max_terminal_ms_per_turn=2_000.0,
    )

    assert result["passed"] is True
    assert result["issues"] == []

    regressed = evaluate_materials_trace_performance(
        {"prompt": turn},
        max_model_calls_per_turn=1,
        max_input_tokens_per_turn=50_000,
        max_tool_calls_per_turn=1,
        max_input_amplification_vs_peak=1.5,
        max_terminal_ms_per_turn=1_000.0,
    )
    assert regressed["passed"] is False
    assert len(regressed["issues"]) == 5


def test_materials_trace_performance_rejects_conflicting_or_malformed_usage():
    malformed = {
        "event_kind": "run.token_usage",
        "payload": {
            "usage_event_id": "bad-total",
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 99,
        },
    }
    events = [
        _usage_event("same-id", input_tokens=100, output_tokens=10),
        _usage_event("same-id", input_tokens=101, output_tokens=10),
        malformed,
    ]
    turn = summarize_run_trace(
        run={
            "run_id": "run-bad-usage",
            "status": "succeeded",
            "response_text": "Done.",
        },
        events=events,
        artifacts=[],
    )

    result = evaluate_materials_trace_performance({"prompt": turn})

    assert result["passed"] is False
    assert turn["token_usage"]["model_call_count"] == 1
    assert turn["token_usage"]["conflicting_duplicate_count"] == 1
    assert turn["token_usage"]["invalid_event_count"] == 1
    assert any("malformed token-usage" in issue for issue in result["issues"])
    assert any("conflicting duplicate" in issue for issue in result["issues"])


@pytest.mark.parametrize("skill_name", sorted(MATERIALS_SKILL_NAMES))
def test_materials_trace_quality_recognizes_every_materials_skill(skill_name: str):
    validation = _inspect_materials_validation_artifact(
        _ArtifactClient(_record_payload()), [_artifact(), _evidence_artifact()]
    )
    turn = summarize_run_trace(
        run={
            "run_id": f"run-{skill_name}",
            "goal": "Run a materials analysis with auditable validation.",
            "status": "succeeded",
            "response_text": "Validated materials result.",
        },
        events=[
            *_successful_materials_tool_events(skill_name=skill_name),
            {"event_kind": "run.completed"},
        ],
        artifacts=[_artifact(), _evidence_artifact()],
        materials_validation=validation,
    )

    result = evaluate_materials_trace_quality({"prompt": turn})

    assert result["passed"] is True
    assert result["signals"]["matched_skill_read_completion"] is True


def test_materials_trace_rejects_started_only_required_tools():
    validation = _inspect_materials_validation_artifact(
        _ArtifactClient(_record_payload()), [_artifact(), _evidence_artifact()]
    )
    turn = summarize_run_trace(
        run={
            "run_id": "run-materials",
            "goal": "Calculate the CuKa XRD pattern for FCC Ni.",
            "status": "succeeded",
            "response_text": "Claimed result.",
        },
        events=[
            event
            for event in _successful_materials_tool_events()
            if event["event_kind"] == "tool_call.started"
        ],
        artifacts=[_artifact(), _evidence_artifact()],
        materials_validation=validation,
    )

    result = evaluate_materials_trace_quality({"prompt": turn})

    assert result["passed"] is False
    assert result["score"] < 10.0
    assert result["signals"]["matched_skill_read_completion"] is False
    assert result["signals"]["matched_execute_completion"] is False


@pytest.mark.parametrize(
    ("events", "expected_issue"),
    [
        (
            [
                *_successful_materials_tool_events()[:3],
                {
                    "event_kind": "tool_call.completed",
                    "payload": {
                        "tool_name": "execute",
                        "tool_call_id": "different-execute-call",
                        "runtime_image_digest": _RUNTIME_IMAGE_DIGEST,
                        "exit_code": 0,
                    },
                },
            ],
            "matched successful execute completion",
        ),
        (
            [
                *_successful_materials_tool_events()[:3],
                {
                    "event_kind": "tool_call.completed",
                    "payload": {
                        "tool_name": "execute",
                        "tool_call_id": "execute-1",
                        "exit_code": 0,
                    },
                },
            ],
            "immutable runtime image",
        ),
        (_successful_materials_tool_events(execute_exit_code=2), "exit code 0"),
        (
            [
                *_successful_materials_tool_events(),
                {
                    "event_kind": "tool_call.failed",
                    "payload": {
                        "tool_name": "read_file",
                        "tool_call_id": "failed-read-2",
                    },
                },
            ],
            "failed tool terminal",
        ),
    ],
)
def test_materials_trace_rejects_adversarial_tool_lifecycles(events, expected_issue):
    validation = _inspect_materials_validation_artifact(
        _ArtifactClient(_record_payload()), [_artifact(), _evidence_artifact()]
    )
    turn = summarize_run_trace(
        run={
            "run_id": "run-materials",
            "goal": "Calculate the CuKa XRD pattern for FCC Ni.",
            "status": "succeeded",
            "response_text": "Claimed result.",
        },
        events=events,
        artifacts=[_artifact(), _evidence_artifact()],
        materials_validation=validation,
    )

    result = evaluate_materials_trace_quality({"prompt": turn})

    assert result["passed"] is False
    assert any(expected_issue in issue for issue in result["issues"])


def test_materials_trace_is_not_scientifically_verified_from_run_success_alone():
    turn = summarize_run_trace(
        run={
            "run_id": "run-materials",
            "goal": "Make an IPF key.",
            "status": "succeeded",
            "response_text": "Done.",
        },
        events=_successful_materials_tool_events(),
        artifacts=[],
    )

    result = evaluate_materials_trace_quality({"prompt": turn})
    assert result["passed"] is False
    assert result["signals"]["terminal_ok"] is True
    assert result["signals"]["first_party_scientific_record_valid"] is False
    assert result["independent_scientific_verification"] is False
    assert result["scientific_conclusion_verified"] is False
    assert "materials_validation.json was not inspected" in result["issues"]


def test_trace_inspector_rejects_fabricated_evidence_digest():
    summary = _inspect_materials_validation_artifact(
        _ArtifactClient(_record_payload()),
        [_artifact(), _evidence_artifact(sha256="c" * 64)],
    )

    assert summary is not None
    assert summary["valid"] is True
    assert summary["evidence_verified"] is False
    assert "SHA-256" in summary["evidence_errors"][0]


def test_trace_inspector_only_accepts_validation_record_from_durable_outputs():
    misplaced = {**_artifact(), "path": "workspace/materials_validation.json"}
    assert (
        _inspect_materials_validation_artifact(
            _ArtifactClient(_record_payload()), [misplaced, _evidence_artifact()]
        )
        is None
    )


def test_trace_inspector_accepts_control_plane_collector_basename_form():
    collected_validation = {
        **_artifact(),
        "path": "materials_validation.json",
        "kind": "artifact",
        "run_id": "run-materials",
        "tool_name": "outputs_collector",
    }
    collected_evidence = {
        **_evidence_artifact(),
        "path": "xrd_pattern.json",
        "kind": "artifact",
        "run_id": "run-materials",
        "tool_name": "outputs_collector",
    }

    summary = _inspect_materials_validation_artifact(
        _ArtifactClient(_record_payload()), [collected_validation, collected_evidence]
    )

    assert summary is not None
    assert summary["valid"] is True
    assert summary["durable_path"] == "outputs/materials_validation.json"
    assert summary["evidence_verified"] is True


def test_trace_inspector_rejects_untrusted_basename_without_collector_provenance():
    basename = {**_artifact(), "path": "materials_validation.json"}

    assert (
        _inspect_materials_validation_artifact(
            _ArtifactClient(_record_payload()), [basename, _evidence_artifact()]
        )
        is None
    )


def test_materials_trace_rejects_remote_mutation_without_run_scope():
    validation = _inspect_materials_validation_artifact(
        _ArtifactClient(_record_payload()), [_artifact(), _evidence_artifact()]
    )
    turn = summarize_run_trace(
        run={
            "run_id": "run-materials",
            "goal": "Compute XRD and save durable outputs.",
            "status": "succeeded",
            "response_text": "Done.",
        },
        events=[
            *_successful_materials_tool_events(),
            *_remote_mutation_tool_events("bisque_upload_workspace_files"),
        ],
        artifacts=[_artifact(), _evidence_artifact()],
        materials_validation=validation,
    )

    result = evaluate_materials_trace_quality({"prompt": turn})

    assert turn["remote_mutation_intents"] == []
    assert turn["remote_mutation_scope_valid"] is True
    assert result["passed"] is False
    assert result["signals"]["remote_mutation_scope_valid"] is True
    assert result["signals"]["remote_mutation_aligned"] is False
    assert any("exceeded the immutable run capability" in issue for issue in result["issues"])


def test_materials_trace_rejects_remote_mutation_outside_valid_run_scope():
    validation = _inspect_materials_validation_artifact(
        _ArtifactClient(_record_payload()), [_artifact(), _evidence_artifact()]
    )
    turn = summarize_run_trace(
        run={
            "run_id": "run-materials",
            "goal": "Compute XRD, then create a BisQue dataset.",
            "status": "succeeded",
            "response_text": "Done.",
            "metadata": {"remote_mutation_intents": ["bisque.upload"]},
        },
        events=[
            *_successful_materials_tool_events(),
            *_remote_mutation_tool_events("bisque_create_dataset"),
        ],
        artifacts=[_artifact(), _evidence_artifact()],
        materials_validation=validation,
    )

    result = evaluate_materials_trace_quality({"prompt": turn})

    assert turn["remote_mutation_intents"] == ["bisque.upload"]
    assert turn["remote_mutation_scope_valid"] is True
    assert result["passed"] is False
    assert result["signals"]["remote_mutation_scope_valid"] is True
    assert result["signals"]["remote_mutation_aligned"] is False


@pytest.mark.parametrize(
    ("raw_scope", "expected_intents"),
    [
        ("bisque.upload", []),
        (["bisque.delete"], []),
        (["bisque.upload", "bisque.upload"], ["bisque.upload"]),
        (["bisque.upload", 1], []),
    ],
)
def test_materials_trace_rejects_malformed_remote_mutation_scope(
    raw_scope: object,
    expected_intents: list[str],
):
    validation = _inspect_materials_validation_artifact(
        _ArtifactClient(_record_payload()), [_artifact(), _evidence_artifact()]
    )
    turn = summarize_run_trace(
        run={
            "run_id": "run-materials",
            "goal": "Compute XRD and save durable outputs.",
            "status": "succeeded",
            "response_text": "Done.",
            "metadata": {"remote_mutation_intents": raw_scope},
        },
        events=_successful_materials_tool_events(),
        artifacts=[_artifact(), _evidence_artifact()],
        materials_validation=validation,
    )

    result = evaluate_materials_trace_quality({"prompt": turn})

    assert turn["remote_mutation_intents"] == expected_intents
    assert turn["remote_mutation_scope_valid"] is False
    assert result["passed"] is False
    assert result["signals"]["remote_mutation_scope_valid"] is False
    assert result["signals"]["remote_mutation_aligned"] is False


def test_materials_trace_accepts_called_subset_of_valid_remote_mutation_scope():
    validation = _inspect_materials_validation_artifact(
        _ArtifactClient(_record_payload()), [_artifact(), _evidence_artifact()]
    )
    turn = summarize_run_trace(
        run={
            "run_id": "run-materials",
            "goal": "Compute XRD, upload the output, and prepare a BisQue dataset if useful.",
            "status": "succeeded",
            "response_text": "Uploaded the validated output.",
            "metadata": {
                "remote_mutation_intents": [
                    "bisque.create_dataset",
                    "bisque.upload",
                ]
            },
        },
        events=[
            *_successful_materials_tool_events(),
            *_remote_mutation_tool_events("bisque_upload_workspace_files"),
        ],
        artifacts=[_artifact(), _evidence_artifact()],
        materials_validation=validation,
    )

    result = evaluate_materials_trace_quality({"prompt": turn})

    assert turn["remote_mutation_intents"] == [
        "bisque.upload",
        "bisque.create_dataset",
    ]
    assert turn["remote_mutation_scope_valid"] is True
    assert result["passed"] is True
    assert result["signals"]["remote_mutation_scope_valid"] is True
    assert result["signals"]["remote_mutation_aligned"] is True
