"""Closed, model-nonextensible crystal-plasticity tools.

The tools in this module expose the already-qualified analytical kernels without
requiring an agent to discover Python APIs through the general code runner.  They
deliberately stop at crystallographic geometry, resolved shear, and structural
CPFE-input validation.  No constitutive integration or FE/spectral solve is
available through this surface.  CPFE source digests are caller declarations:
their syntax is validated here, but their referenced bytes are not resolved or
re-hashed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from langchain.tools import tool

from ultra_deepagents.materials.crystal_plasticity import (
    DAMASK_CRYSTAL_SOURCE_URL,
    DAMASK_REFERENCE_VERSION,
    CPFEInputContract,
    CrystalPlasticityInputError,
    CrystalPlasticityUnsupportedError,
    SourceProvenance,
    canonical_slip_systems,
    execute_cpfe,
    resolved_shear_stresses,
    uniaxial_schmid_factors,
    validate_cpfe_input_contract,
    validate_crystal_to_sample_rotation,
)
from ultra_deepagents.materials.validation import (
    EvidenceArtifact,
    ValidationCheck,
    ValidationOutcome,
    assess_scientific_status,
    canonical_record_json,
    parse_assessment_record,
)

TOOL_RESULT_SCHEMA_VERSION = "ultra.materials.crystal-plasticity-tool-result.v1"
MAX_TYPED_INPUT_BYTES = 1024 * 1024
MAX_TYPED_OUTPUT_BYTES = 512 * 1024
MAX_INLINE_ORIENTATION_BYTES = 128 * 1024
HYDROSTATIC_RELATIVE_TOLERANCE = 1.0e-12


class CrystalPlasticityToolError(RuntimeError):
    """A typed request failed before producing a supported result."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code
        self.public_message = " ".join((message or code).split())[:1000]


def _canonical_json(value: Any, *, label: str) -> bytes:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise CrystalPlasticityToolError(
            "invalid_typed_input", f"{label} must be finite JSON"
        ) from exc
    if not payload or len(payload) > MAX_TYPED_INPUT_BYTES:
        raise CrystalPlasticityToolError(
            "typed_input_too_large",
            f"{label} exceeds the {MAX_TYPED_INPUT_BYTES}-byte typed-input limit",
        )
    return payload


def _input_evidence(payload: bytes) -> dict[str, Any]:
    return {
        "algorithm": "sha256",
        "canonicalization": "UTF-8 JSON, sorted keys, compact separators, finite numbers",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _json_artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "media_type": "application/json",
        "canonical_json": payload.decode("utf-8"),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _validation_artifact(
    *,
    operation: str,
    analysis_artifact: Mapping[str, Any],
    check_specs: Sequence[Mapping[str, Any]],
    capability_supported: bool,
) -> dict[str, Any]:
    evidence = EvidenceArtifact(
        name=f"typed {operation} analysis record",
        sha256=str(analysis_artifact["sha256"]),
        artifact_id=f"typed-{operation}:{analysis_artifact['sha256']}",
        size_bytes=int(analysis_artifact["size_bytes"]),
    )
    checks = tuple(
        ValidationCheck(
            validator_id=str(spec["validator_id"]),
            outcome=ValidationOutcome(str(spec["outcome"])),
            observed=spec["observed"],
            expected=spec["expected"],
            units=str(spec["units"]),
            tolerance_rationale=str(spec["tolerance_rationale"]),
            required=True,
            critical=bool(spec.get("critical", False)),
            library_versions={
                str(name): str(version)
                for name, version in spec.get(
                    "library_versions",
                    {
                        "numpy": np.__version__,
                        "slip_geometry_reference": (
                            f"DAMASK-{DAMASK_REFERENCE_VERSION}-transcription"
                        ),
                    },
                ).items()
            },
            evidence=(evidence,),
            message=str(spec.get("message", "")),
        )
        for spec in check_specs
    )
    required_ids = tuple(check.validator_id for check in checks)
    assessment = assess_scientific_status(
        run_status="succeeded",
        checks=checks,
        required_validator_ids=required_ids,
        capability_supported=capability_supported,
    )
    canonical = canonical_record_json(assessment)
    # Recompute the decision from the exact bytes before returning them to the model.
    parse_assessment_record(json.loads(canonical))
    payload = canonical.encode("utf-8")
    validator_ids = {check.validator_id for check in checks}
    return {
        "media_type": "application/json",
        "canonical_json": canonical,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "validation_scope": (
            "bounded_typed_geometry_with_caller_declared_phase_structure"
            if "crystal_plasticity.phase_structure_assignment_bound" in validator_ids
            else (
                "bounded_typed_tool_operation_with_declaration_only_provenance"
                if "crystal_plasticity.source_provenance_bytes_bound" in validator_ids
                else "bounded_typed_tool_operation"
            )
        ),
        "scientific_status": assessment.scientific_status.value,
        "verified": assessment.verified,
    }


def _bounded_response(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(payload) > MAX_TYPED_OUTPUT_BYTES:
        raise CrystalPlasticityToolError(
            "typed_output_too_large",
            f"result exceeds the {MAX_TYPED_OUTPUT_BYTES}-byte typed-output limit",
        )
    return dict(value)


def _public_error(
    exc: CrystalPlasticityToolError | CrystalPlasticityInputError,
    *,
    operation: str,
    contract_valid: bool | None = None,
    attempt_execution_requested: bool | None = None,
) -> dict[str, Any]:
    if isinstance(exc, CrystalPlasticityToolError):
        code = exc.code
        message = exc.public_message
    else:
        code = "invalid_crystal_plasticity_input"
        message = " ".join(str(exc).split())[:1000]
    result: dict[str, Any] = {
        "schema_version": TOOL_RESULT_SCHEMA_VERSION,
        "ok": False,
        "operation": operation,
        "error": code,
        "message": message,
        "partial_results_returned": False,
        "cpfe_solved": False,
    }
    if contract_valid is not None:
        result["contract_valid"] = contract_valid
    if attempt_execution_requested is not None:
        result["execution"] = {
            "attempt_requested": attempt_execution_requested,
            "attempted": False,
            "supported": False,
            "status": "not_attempted_invalid_contract",
        }
    return _bounded_response(result)


def _clean_phase_id(value: Any) -> str:
    if not isinstance(value, str):
        raise CrystalPlasticityToolError(
            "invalid_typed_input", "phase_id must be a nonblank string"
        )
    phase_id = value.strip()
    if not phase_id or len(phase_id) > 256 or any(ord(character) < 32 for character in phase_id):
        raise CrystalPlasticityToolError(
            "invalid_typed_input", "phase_id must be a nonblank bounded string"
        )
    return phase_id


def _finite_nonzero(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise CrystalPlasticityToolError(
            "invalid_typed_input", f"{field_name} must be a finite nonzero number"
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CrystalPlasticityToolError(
            "invalid_typed_input", f"{field_name} must be a finite nonzero number"
        ) from exc
    if not math.isfinite(number) or number == 0.0:
        raise CrystalPlasticityToolError(
            "invalid_typed_input", f"{field_name} must be a finite nonzero number"
        )
    return number


def analyze_crystal_slip_typed(
    *,
    phase_id: str,
    crystal_structure: str,
    slip_families: Sequence[str],
    c_over_a: float | None = None,
    rotation_crystal_to_sample: Sequence[Sequence[float]] | None = None,
    stress_sample: Sequence[Sequence[float]] | None = None,
    stress_unit: str | None = None,
    load_axis_sample: Sequence[float] | None = None,
    hydrostatic_control_stress: float | None = None,
) -> dict[str, Any]:
    """Enumerate one phase's slip geometry and optionally analyze one orientation.

    The request accepts no code, paths, constitutive parameters, or solver options.
    Supplying a stress requires one active crystal-to-sample rotation and an explicit
    stress unit.  A load axis requests classical uniaxial Schmid factors; it does not
    assert that the supplied stress state is uniaxial.
    """

    operation = "analytical_slip_geometry"
    raw_request = {
        "phase_id": phase_id,
        "crystal_structure": crystal_structure,
        "slip_families": list(slip_families),
        "c_over_a": c_over_a,
        "rotation_crystal_to_sample": rotation_crystal_to_sample,
        "stress_sample": stress_sample,
        "stress_unit": stress_unit,
        "load_axis_sample": load_axis_sample,
        "hydrostatic_control_stress": hydrostatic_control_stress,
    }
    try:
        request_payload = _canonical_json(raw_request, label="crystal-plasticity request")
        normalized_phase_id = _clean_phase_id(phase_id)
        systems = canonical_slip_systems(
            crystal_structure,
            families=slip_families,
            c_over_a=c_over_a,
        )

        has_rotation = rotation_crystal_to_sample is not None
        has_stress = stress_sample is not None
        has_axis = load_axis_sample is not None
        has_hydrostatic = hydrostatic_control_stress is not None
        if (has_stress or has_axis or has_hydrostatic) and not has_rotation:
            raise CrystalPlasticityToolError(
                "invalid_typed_input",
                "rotation_crystal_to_sample is required for stress, load-axis, or hydrostatic analysis",
            )
        if has_stress and not stress_unit:
            raise CrystalPlasticityToolError(
                "invalid_typed_input", "stress_unit is required when stress_sample is supplied"
            )
        if has_hydrostatic and not stress_unit:
            raise CrystalPlasticityToolError(
                "invalid_typed_input",
                "stress_unit is required when hydrostatic_control_stress is supplied",
            )
        if stress_unit and not (has_stress or has_hydrostatic):
            raise CrystalPlasticityToolError(
                "invalid_typed_input",
                "stress_unit is only valid with stress_sample or hydrostatic_control_stress",
            )

        rotation: np.ndarray | None = None
        if rotation_crystal_to_sample is not None:
            rotation = validate_crystal_to_sample_rotation(rotation_crystal_to_sample)

        resolved: np.ndarray | None = None
        if stress_sample is not None:
            assert rotation is not None and stress_unit is not None
            resolved = resolved_shear_stresses(
                stress_sample=stress_sample,
                rotation_crystal_to_sample=rotation,
                slip_systems=systems,
                stress_unit=stress_unit,
            ).resolved_shear_stress

        schmid: np.ndarray | None = None
        if load_axis_sample is not None:
            assert rotation is not None
            schmid = uniaxial_schmid_factors(
                load_axis_sample=load_axis_sample,
                rotation_crystal_to_sample=rotation,
                slip_systems=systems,
            )

        hydrostatic_control: dict[str, Any] = {
            "applied": False,
            "passed": None,
            "maximum_absolute_resolved_shear": None,
            "tolerance": None,
            "stress_unit": stress_unit,
        }
        if hydrostatic_control_stress is not None:
            assert rotation is not None and stress_unit is not None
            hydrostatic_value = _finite_nonzero(
                hydrostatic_control_stress,
                field_name="hydrostatic_control_stress",
            )
            hydrostatic = resolved_shear_stresses(
                stress_sample=np.eye(3, dtype=float) * hydrostatic_value,
                rotation_crystal_to_sample=rotation,
                slip_systems=systems,
                stress_unit=stress_unit,
            ).resolved_shear_stress
            maximum_hydrostatic_shear = float(np.max(np.abs(hydrostatic)))
            tolerance = max(
                1.0e-12,
                abs(hydrostatic_value) * HYDROSTATIC_RELATIVE_TOLERANCE,
            )
            hydrostatic_control = {
                "applied": True,
                "hydrostatic_stress": hydrostatic_value,
                "stress_unit": stress_unit,
                "maximum_absolute_resolved_shear": maximum_hydrostatic_shear,
                "tolerance": tolerance,
                "passed": maximum_hydrostatic_shear <= tolerance,
            }

        system_records: list[dict[str, Any]] = []
        for index, system in enumerate(systems):
            system_records.append(
                {
                    "system_id": system.system_id,
                    "family": system.family,
                    "direction_indices": list(system.direction_indices),
                    "plane_indices": list(system.plane_indices),
                    "slip_direction_crystal": list(system.slip_direction_crystal),
                    "plane_normal_crystal": list(system.plane_normal_crystal),
                    "resolved_shear_stress": (None if resolved is None else float(resolved[index])),
                    "schmid_factor": None if schmid is None else float(schmid[index]),
                }
            )

        direction_norm_error = max(
            abs(float(np.linalg.norm(system.slip_direction_crystal)) - 1.0) for system in systems
        )
        normal_norm_error = max(
            abs(float(np.linalg.norm(system.plane_normal_crystal)) - 1.0) for system in systems
        )
        direction_plane_dot = max(
            abs(float(np.dot(system.slip_direction_crystal, system.plane_normal_crystal)))
            for system in systems
        )
        response: dict[str, Any] = {
            "schema_version": TOOL_RESULT_SCHEMA_VERSION,
            "ok": True,
            "operation": operation,
            "input_evidence": _input_evidence(request_payload),
            "phase_id": normalized_phase_id,
            "crystal_structure": systems[0].crystal_structure,
            "c_over_a": c_over_a,
            "frames": {
                "orientation": "crystal_to_sample",
                "stress": "sample",
            },
            "stress_unit": stress_unit,
            "system_count": len(systems),
            "geometry_reference": {
                "kind": "deterministic_transcription",
                "backend": systems[0].reference_backend,
                "damask_version": DAMASK_REFERENCE_VERSION,
                "source_url": DAMASK_CRYSTAL_SOURCE_URL,
                "live_reference_cross_check_performed": False,
            },
            "systems": system_records,
            "summary": {
                "maximum_absolute_resolved_shear_stress": (
                    None if resolved is None else float(np.max(np.abs(resolved)))
                ),
                "maximum_schmid_factor": (None if schmid is None else float(np.max(schmid))),
            },
            "controls": {
                "maximum_slip_direction_norm_error": direction_norm_error,
                "maximum_plane_normal_norm_error": normal_norm_error,
                "maximum_direction_plane_absolute_dot": direction_plane_dot,
                "hydrostatic_zero_shear": hydrostatic_control,
            },
            "phase_structure_assignment": {
                "status": "unverified",
                "mode": "caller_declared_without_independent_source",
                "phase_id": normalized_phase_id,
                "crystal_structure": systems[0].crystal_structure,
                "independent_source_supplied": False,
                "assignment_independently_verified": False,
                "phase_name_semantics_interpreted": False,
            },
            "capability_boundary": {
                "geometry_calculated": True,
                "resolved_shear_calculated": resolved is not None,
                "classical_uniaxial_schmid_factors_calculated": schmid is not None,
                "phase_structure_assignment_declared": True,
                "phase_structure_assignment_independently_verified": False,
                "phase_identity_inferred_from_phase_id": False,
                "slip_activity_inferred": False,
                "crss_applied": False,
                "constitutive_response_calculated": False,
                "cpfe_solved": False,
            },
            "partial_results_returned": False,
            "cpfe_solved": False,
        }
        geometry_maximum_error = max(
            direction_norm_error,
            normal_norm_error,
            direction_plane_dot,
        )
        check_specs: list[dict[str, Any]] = [
            {
                "validator_id": "crystal_plasticity.geometry_unit_orthogonality",
                "outcome": "pass" if geometry_maximum_error <= 1.0e-12 else "fail",
                "observed": {
                    "maximum_slip_direction_norm_error": direction_norm_error,
                    "maximum_plane_normal_norm_error": normal_norm_error,
                    "maximum_direction_plane_absolute_dot": direction_plane_dot,
                },
                "expected": {"maximum_absolute_error": 1.0e-12},
                "units": "1",
                "tolerance_rationale": (
                    "The canonical slip directions/normals must have unit norm and be "
                    "orthogonal within the kernel's 1e-12 absolute geometry tolerance."
                ),
                "critical": True,
            },
            {
                "validator_id": "crystal_plasticity.phase_structure_assignment_bound",
                "outcome": "skip",
                "observed": {
                    "phase_id": normalized_phase_id,
                    "crystal_structure": systems[0].crystal_structure,
                    "caller_declaration_present": True,
                    "independent_source_supplied": False,
                    "assignment_independently_verified": False,
                    "phase_name_semantics_interpreted": False,
                },
                "expected": {
                    "independent_source_supplied": True,
                    "assignment_independently_verified": True,
                },
                "units": "1",
                "tolerance_rationale": (
                    "A phase label has no intrinsic crystallographic semantics. Geometry may "
                    "be evaluated for the caller-selected structure, but the phase-to-structure "
                    "association requires independent source binding with zero tolerance for "
                    "semantic inference from the phase name."
                ),
                "critical": True,
                "message": (
                    "phase_id and crystal_structure are caller-declared; this tool neither "
                    "interprets the phase name nor independently verifies their association."
                ),
            },
        ]
        if resolved is not None:
            finite_count = int(np.count_nonzero(np.isfinite(resolved)))
            check_specs.append(
                {
                    "validator_id": "crystal_plasticity.resolved_shear_finite",
                    "outcome": "pass" if finite_count == len(systems) else "fail",
                    "observed": {
                        "finite_system_count": finite_count,
                        "system_count": len(systems),
                    },
                    "expected": {"finite_system_count": len(systems)},
                    "units": str(stress_unit),
                    "tolerance_rationale": (
                        "Every signed resolved-shear value must be finite; no missing or "
                        "non-finite system value is admissible."
                    ),
                    "critical": True,
                }
            )
        if schmid is not None:
            minimum_schmid = float(np.min(schmid))
            maximum_schmid = float(np.max(schmid))
            check_specs.append(
                {
                    "validator_id": "crystal_plasticity.uniaxial_schmid_bounds",
                    "outcome": (
                        "pass"
                        if minimum_schmid >= -1.0e-12 and maximum_schmid <= 0.5 + 1.0e-12
                        else "fail"
                    ),
                    "observed": {"minimum": minimum_schmid, "maximum": maximum_schmid},
                    "expected": {"minimum": 0.0, "maximum": 0.5},
                    "units": "1",
                    "tolerance_rationale": (
                        "Classical absolute uniaxial Schmid factors are bounded by 0.5; "
                        "1e-12 covers floating-point roundoff only."
                    ),
                    "critical": True,
                }
            )
        if hydrostatic_control["applied"]:
            check_specs.append(
                {
                    "validator_id": "crystal_plasticity.hydrostatic_zero_shear",
                    "outcome": "pass" if hydrostatic_control["passed"] else "fail",
                    "observed": hydrostatic_control["maximum_absolute_resolved_shear"],
                    "expected": {"maximum": hydrostatic_control["tolerance"]},
                    "units": str(stress_unit),
                    "tolerance_rationale": (
                        "Hydrostatic Cauchy stress must resolve to zero shear; the tolerance "
                        "is max(1e-12 stress units, 1e-12 times the control magnitude)."
                    ),
                    "critical": True,
                }
            )
        analysis_artifact = _json_artifact(response)
        response["analysis_artifact"] = analysis_artifact
        validation_artifact = _validation_artifact(
            operation=operation,
            analysis_artifact=analysis_artifact,
            check_specs=check_specs,
            capability_supported=True,
        )
        response["materials_validation_artifact"] = validation_artifact
        response["scientific_status"] = validation_artifact["scientific_status"]
        response["verified"] = validation_artifact["verified"]
        return _bounded_response(response)
    except (CrystalPlasticityToolError, CrystalPlasticityInputError) as exc:
        return _public_error(exc, operation=operation)


def _provenance_record(value: SourceProvenance) -> dict[str, str]:
    return {
        "source_id": value.source_id,
        "source_type": value.source_type,
        "citation": value.citation,
        "sha256": value.sha256,
    }


def _validated_contract_summary(contract: CPFEInputContract) -> dict[str, Any]:
    orientations = contract.orientations_crystal_to_sample.tolist()
    orientation_payload = json.dumps(
        orientations,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    orientations_returned_inline = len(orientation_payload) <= MAX_INLINE_ORIENTATION_BYTES
    return {
        "schema_version": contract.schema_version,
        "phase": {
            "phase_id": contract.phase_id,
            "crystal_structure": contract.crystal_structure,
            "symmetry": contract.symmetry,
            "c_over_a": contract.c_over_a,
            "provenance": _provenance_record(contract.phase_provenance),
        },
        "frames": {"orientation": "crystal_to_sample", "stress": "sample"},
        "units": {"stress": "Pa", "length": "m", "time": "s"},
        "orientation_count": int(contract.orientations_crystal_to_sample.shape[0]),
        "orientations": orientations if orientations_returned_inline else None,
        "orientation_evidence": {
            "canonicalization": "UTF-8 compact finite JSON array in validated input order",
            "sha256": hashlib.sha256(orientation_payload).hexdigest(),
            "size_bytes": len(orientation_payload),
            "returned_inline": orientations_returned_inline,
        },
        "slip_families": list(contract.slip_families),
        "crss": {
            "unit": "Pa",
            "values": dict(contract.crss_pa),
            "provenance": _provenance_record(contract.crss_provenance),
        },
        "hardening": {
            "model_id": contract.hardening_model_id,
            "parameters": dict(contract.hardening_parameters),
            "parameter_units": dict(contract.hardening_parameter_units),
            "provenance": _provenance_record(contract.hardening_provenance),
            "model_specific_semantics_validated": False,
        },
    }


def validate_cpfe_contract_typed(
    *,
    contract: Mapping[str, Any],
    attempt_execution: bool = False,
) -> dict[str, Any]:
    """Validate the closed CPFE contract and optionally exercise its refusal boundary."""

    operation = "cpfe_contract_validation"
    try:
        request_payload = _canonical_json(
            {"contract": contract, "attempt_execution": attempt_execution},
            label="CPFE contract request",
        )
        if not isinstance(attempt_execution, bool):
            raise CrystalPlasticityToolError(
                "invalid_typed_input", "attempt_execution must be a boolean"
            )
        validated = validate_cpfe_input_contract(contract)
    except (CrystalPlasticityToolError, CrystalPlasticityInputError) as exc:
        return _public_error(
            exc,
            operation=operation,
            contract_valid=False,
            attempt_execution_requested=(
                attempt_execution if isinstance(attempt_execution, bool) else False
            ),
        )

    execution: dict[str, Any]
    execution_boundary_failed = False
    if attempt_execution:
        try:
            execute_cpfe(validated)
        except CrystalPlasticityUnsupportedError as exc:
            execution = {
                "attempt_requested": True,
                "attempted": True,
                "supported": False,
                "status": "unsupported",
                "error_type": type(exc).__name__,
                "message": " ".join(str(exc).split())[:1000],
            }
        else:
            execution_boundary_failed = True
            execution = {
                "attempt_requested": True,
                "attempted": True,
                "supported": False,
                "status": "unexpected_return_rejected",
                "error_type": "CrystalPlasticityToolError",
                "message": "the unqualified CPFE boundary returned without refusing execution",
            }
    else:
        execution = {
            "attempt_requested": False,
            "attempted": False,
            "supported": False,
            "status": "not_requested",
            "message": validated.unsupported_reason,
        }

    response = {
        "schema_version": TOOL_RESULT_SCHEMA_VERSION,
        "ok": not execution_boundary_failed,
        "operation": operation,
        "input_evidence": _input_evidence(request_payload),
        "contract_valid": True,
        "validated_contract": _validated_contract_summary(validated),
        "provenance_binding": {
            "status": "unverified",
            "mode": "caller_declared_digest_only",
            "digest_syntax_validated": True,
            "source_bytes_resolved": False,
            "source_bytes_rehashed": False,
            "digest_match_independently_verified": False,
        },
        "execution": execution,
        "capability_boundary": {
            "closed_contract_schema_validated": True,
            "proper_rotations_validated": True,
            "phase_symmetry_consistency_validated": True,
            "si_units_validated": True,
            "crss_coverage_and_provenance_declaration_validated": True,
            "hardening_structure_and_provenance_declaration_validated": True,
            "source_provenance_bytes_bound": False,
            "hardening_model_specific_semantics_validated": False,
            "constitutive_integrator_bound": False,
            "finite_element_or_spectral_solver_bound": False,
            "cpfe_solved": False,
        },
        "partial_results_returned": False,
        "cpfe_solved": False,
    }
    if execution_boundary_failed:
        response["error"] = "unqualified_cpfe_execution_returned"
        response["message"] = execution["message"]
    else:
        analysis_artifact = _json_artifact(response)
        response["analysis_artifact"] = analysis_artifact
        response["materials_validation_artifact"] = _validation_artifact(
            operation=operation,
            analysis_artifact=analysis_artifact,
            check_specs=(
                {
                    "validator_id": "crystal_plasticity.cpfe_contract_structure",
                    "outcome": "pass",
                    "observed": {
                        "contract_valid": True,
                        "phase_id": validated.phase_id,
                        "orientation_count": int(validated.orientations_crystal_to_sample.shape[0]),
                        "slip_family_count": len(validated.slip_families),
                        "execution_status": execution["status"],
                    },
                    "expected": {
                        "closed_schema_version": validated.schema_version,
                        "one_phase": True,
                        "proper_crystal_to_sample_orientations": True,
                        "si_units": True,
                        "complete_crss_and_hardening_provenance_declarations": True,
                    },
                    "units": "1",
                    "tolerance_rationale": (
                        "The CPFE input contract is closed and categorical: every required "
                        "phase/frame/unit/CRSS/hardening/provenance-declaration field must "
                        "validate exactly."
                    ),
                    "critical": True,
                },
                {
                    "validator_id": "crystal_plasticity.source_provenance_bytes_bound",
                    "outcome": "skip",
                    "observed": {
                        "caller_declarations_present": True,
                        "digest_syntax_validated": True,
                        "source_bytes_resolved": False,
                        "source_bytes_rehashed": False,
                        "digest_match_independently_verified": False,
                    },
                    "expected": {
                        "source_bytes_resolved": True,
                        "source_bytes_rehashed": True,
                        "digest_match_independently_verified": True,
                    },
                    "units": "1",
                    "tolerance_rationale": (
                        "Provenance binding requires exact source-byte retrieval and SHA-256 "
                        "replay; digest syntax alone has zero evidentiary tolerance."
                    ),
                    "critical": True,
                    "library_versions": {"hash_algorithm": "sha256"},
                    "message": (
                        "Phase, CRSS, and hardening digest declarations are structurally valid, "
                        "but their source bytes were not resolved or re-hashed."
                    ),
                },
            ),
            # Structural contract validation is supported. A requested CPFE execution is not.
            capability_supported=not attempt_execution,
        )
    return _bounded_response(response)


def _json_response(value: Mapping[str, Any]) -> str:
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True)


def build_crystal_plasticity_tools() -> list[Any]:
    """Build the bounded analytical and CPFE-contract tool surface."""

    @tool
    def materials_analyze_crystal_slip(
        phase_id: str,
        crystal_structure: str,
        slip_families: list[str],
        c_over_a: float | None = None,
        rotation_crystal_to_sample: list[list[float]] | None = None,
        stress_sample: list[list[float]] | None = None,
        stress_unit: str | None = None,
        load_axis_sample: list[float] | None = None,
        hydrostatic_control_stress: float | None = None,
    ) -> str:
        """Enumerate canonical DAMASK-3.1.0-transcribed FCC/BCC/HCP slip geometry and optionally calculate one orientation's resolved shear and classical uniaxial Schmid factors. phase_id is mandatory but opaque: the tool does not infer crystallographic semantics from its name or independently verify the caller-declared phase_id/crystal_structure association, so geometry checks may pass while overall scientific_status remains unverified. HCP requires the measured/supplied c_over_a; never assume an ideal ratio. Rotation is the active crystal-to-sample matrix, stress is a symmetric sample-frame Cauchy tensor, and stress_unit must be Pa/kPa/MPa/GPa. Set hydrostatic_control_stress to run the zero-shear control in the same unit. Returns exact content-addressed analysis_artifact.canonical_json and recomputed materials_validation_artifact.canonical_json strings for direct output writes; do not reconstruct them. This geometry-only tool never infers active slip, applies CRSS, or solves CPFE. It accepts no code, paths, or solver options."""

        return _json_response(
            analyze_crystal_slip_typed(
                phase_id=phase_id,
                crystal_structure=crystal_structure,
                slip_families=slip_families,
                c_over_a=c_over_a,
                rotation_crystal_to_sample=rotation_crystal_to_sample,
                stress_sample=stress_sample,
                stress_unit=stress_unit,
                load_axis_sample=load_axis_sample,
                hydrostatic_control_stress=hydrostatic_control_stress,
            )
        )

    @tool
    def materials_validate_cpfe_contract(
        contract: dict[str, Any],
        attempt_execution: bool = False,
    ) -> str:
        """Validate a closed schema-v1 CPFE input contract for one phase. The contract must explicitly declare active crystal-to-sample frames, sample stress, SI units, proper orientations, a canonical FCC/BCC/HCP slip-family set, one positive Pa-valued CRSS per family, a structurally complete hardening block, and phase/CRSS/hardening source digests. Digest syntax is checked, but source bytes are not resolved or re-hashed, so provenance binding remains unverified. Set attempt_execution=true to exercise the fail-closed boundary: contract validity can pass, but execution and its canonical materials-validation status report unsupported because no constitutive integrator or FE/spectral solver is bound. Returns exact content-addressed analysis and materials-validation canonical JSON strings for direct output writes. No stress-strain curve, convergence result, or CPFE solve is fabricated."""

        return _json_response(
            validate_cpfe_contract_typed(
                contract=contract,
                attempt_execution=attempt_execution,
            )
        )

    return [materials_analyze_crystal_slip, materials_validate_cpfe_contract]


__all__ = [
    "CrystalPlasticityToolError",
    "TOOL_RESULT_SCHEMA_VERSION",
    "analyze_crystal_slip_typed",
    "build_crystal_plasticity_tools",
    "validate_cpfe_contract_typed",
]
