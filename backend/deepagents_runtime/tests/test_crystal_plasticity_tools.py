"""First-class bounded crystal-plasticity tool qualification."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import pytest
import ultra_deepagents.crystal_plasticity_tools as crystal_plasticity_tools
from ultra_deepagents.crystal_plasticity_tools import (
    TOOL_RESULT_SCHEMA_VERSION,
    analyze_crystal_slip_typed,
    build_crystal_plasticity_tools,
    validate_cpfe_contract_typed,
)
from ultra_deepagents.materials.crystal_plasticity import (
    FCC_111_110,
    HCP_PYRAMIDAL_CA,
)
from ultra_deepagents.materials.validation import parse_assessment_record


def _provenance(character: str) -> dict[str, str]:
    return {
        "source_id": f"synthetic-{character}",
        "source_type": "user_declared",
        "citation": "Synthetic typed-tool acceptance control; not materials evidence",
        "sha256": character * 64,
    }


def _cpfe_contract() -> dict[str, Any]:
    return {
        "schema_version": "1",
        "phase": {
            "phase_id": "gamma",
            "crystal_structure": "fcc",
            "symmetry": "m-3m",
            "provenance": _provenance("a"),
        },
        "frames": {"orientation": "crystal_to_sample", "stress": "sample"},
        "units": {"stress": "Pa", "length": "m", "time": "s"},
        "orientations": [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        "slip_families": [FCC_111_110],
        "crss": {
            "unit": "Pa",
            "values": {FCC_111_110: 45.0e6},
            "provenance": _provenance("b"),
        },
        "hardening": {
            "model_id": "voce-isotropic-v1",
            "parameters": {
                "saturation_crss": 150.0e6,
                "initial_hardening_modulus": 800.0e6,
            },
            "parameter_units": {
                "saturation_crss": "Pa",
                "initial_hardening_modulus": "Pa",
            },
            "provenance": _provenance("c"),
        },
    }


def test_cp01_natural_prompt_control_runs_without_code_discovery() -> None:
    result = analyze_crystal_slip_typed(
        phase_id="gamma",
        crystal_structure="fcc",
        slip_families=[FCC_111_110],
        rotation_crystal_to_sample=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        stress_sample=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 100.0]],
        stress_unit="MPa",
        load_axis_sample=[0.0, 0.0, 1.0],
        hydrostatic_control_stress=123.0,
    )

    assert result["ok"] is True
    assert result["schema_version"] == TOOL_RESULT_SCHEMA_VERSION
    assert result["phase_id"] == "gamma"
    assert result["system_count"] == 12
    assert result["geometry_reference"]["damask_version"] == "3.1.0"
    assert result["geometry_reference"]["kind"] == "deterministic_transcription"
    assert result["geometry_reference"]["live_reference_cross_check_performed"] is False
    assert len({system["system_id"] for system in result["systems"]}) == 12
    assert result["summary"]["maximum_schmid_factor"] == pytest.approx(1 / math.sqrt(6))
    assert result["summary"]["maximum_absolute_resolved_shear_stress"] == pytest.approx(
        100 / math.sqrt(6)
    )
    assert all(0.0 <= system["schmid_factor"] <= 0.5 for system in result["systems"])
    hydrostatic = result["controls"]["hydrostatic_zero_shear"]
    assert hydrostatic["applied"] is True
    assert hydrostatic["passed"] is True
    assert hydrostatic["maximum_absolute_resolved_shear"] <= 1.0e-12
    assert result["capability_boundary"] == {
        "geometry_calculated": True,
        "resolved_shear_calculated": True,
        "classical_uniaxial_schmid_factors_calculated": True,
        "phase_structure_assignment_declared": True,
        "phase_structure_assignment_independently_verified": False,
        "phase_identity_inferred_from_phase_id": False,
        "slip_activity_inferred": False,
        "crss_applied": False,
        "constitutive_response_calculated": False,
        "cpfe_solved": False,
    }
    assert len(result["input_evidence"]["sha256"]) == 64
    assert result["phase_structure_assignment"] == {
        "status": "unverified",
        "mode": "caller_declared_without_independent_source",
        "phase_id": "gamma",
        "crystal_structure": "fcc",
        "independent_source_supplied": False,
        "assignment_independently_verified": False,
        "phase_name_semantics_interpreted": False,
    }
    analysis_artifact = result["analysis_artifact"]
    analysis_bytes = analysis_artifact["canonical_json"].encode("utf-8")
    assert hashlib.sha256(analysis_bytes).hexdigest() == analysis_artifact["sha256"]
    assert len(analysis_bytes) == analysis_artifact["size_bytes"]
    retained_analysis = json.loads(analysis_artifact["canonical_json"])
    assert retained_analysis["system_count"] == 12
    assert "analysis_artifact" not in retained_analysis
    validation_artifact = result["materials_validation_artifact"]
    validation_bytes = validation_artifact["canonical_json"].encode("utf-8")
    assert hashlib.sha256(validation_bytes).hexdigest() == validation_artifact["sha256"]
    assessment = parse_assessment_record(json.loads(validation_artifact["canonical_json"]))
    assert assessment.scientific_status.value == "unverified"
    assert assessment.verified is False
    assert result["scientific_status"] == "unverified"
    assert result["verified"] is False
    assert validation_artifact["validation_scope"] == (
        "bounded_typed_geometry_with_caller_declared_phase_structure"
    )
    assert len(assessment.checks) == 5
    checks = {check.validator_id: check for check in assessment.checks}
    phase_binding = checks["crystal_plasticity.phase_structure_assignment_bound"]
    assert phase_binding.outcome.value == "skip"
    assert phase_binding.required is True
    assert phase_binding.critical is True
    assert phase_binding.observed["assignment_independently_verified"] is False
    assert all(
        check.evidence[0].sha256 == analysis_artifact["sha256"] for check in assessment.checks
    )


def test_analytical_phase_label_has_no_inferred_structure_semantics() -> None:
    result = analyze_crystal_slip_typed(
        phase_id="alpha-Ti-hcp",
        crystal_structure="fcc",
        slip_families=[FCC_111_110],
    )

    assert result["ok"] is True
    assert result["system_count"] == 12
    assert result["phase_id"] == "alpha-Ti-hcp"
    assert result["crystal_structure"] == "fcc"
    assert result["phase_structure_assignment"]["status"] == "unverified"
    assert result["phase_structure_assignment"]["phase_name_semantics_interpreted"] is False
    assert result["capability_boundary"]["phase_identity_inferred_from_phase_id"] is False
    assert result["scientific_status"] == "unverified"
    assert result["verified"] is False
    assessment = parse_assessment_record(
        json.loads(result["materials_validation_artifact"]["canonical_json"])
    )
    checks = {check.validator_id: check for check in assessment.checks}
    assert checks["crystal_plasticity.geometry_unit_orthogonality"].outcome.value == "pass"
    assert checks["crystal_plasticity.phase_structure_assignment_bound"].outcome.value == "skip"
    assert assessment.scientific_status.value == "unverified"


def test_cp02_missing_hcp_lattice_ratio_returns_typed_failure_without_partial_geometry() -> None:
    result = analyze_crystal_slip_typed(
        phase_id="alpha",
        crystal_structure="hcp",
        slip_families=[HCP_PYRAMIDAL_CA],
    )

    assert result["ok"] is False
    assert result["error"] == "invalid_crystal_plasticity_input"
    assert "c_over_a" in result["message"]
    assert result["partial_results_returned"] is False
    assert result["cpfe_solved"] is False
    assert "systems" not in result


def test_cp03_valid_contract_and_execution_refusal_are_reported_separately() -> None:
    result = validate_cpfe_contract_typed(
        contract=_cpfe_contract(),
        attempt_execution=True,
    )

    assert result["ok"] is True
    assert result["contract_valid"] is True
    assert result["validated_contract"]["phase"]["phase_id"] == "gamma"
    assert result["validated_contract"]["orientation_count"] == 1
    assert result["validated_contract"]["orientations"] == _cpfe_contract()["orientations"]
    assert result["validated_contract"]["orientation_evidence"]["returned_inline"] is True
    assert result["validated_contract"]["crss"]["values"][FCC_111_110] == 45.0e6
    assert result["validated_contract"]["hardening"]["model_specific_semantics_validated"] is False
    assert result["execution"]["attempted"] is True
    assert result["execution"]["supported"] is False
    assert result["execution"]["status"] == "unsupported"
    assert result["execution"]["error_type"] == "CrystalPlasticityUnsupportedError"
    assert result["cpfe_solved"] is False
    assert result["capability_boundary"]["constitutive_integrator_bound"] is False
    assert result["capability_boundary"]["finite_element_or_spectral_solver_bound"] is False
    assert json.loads(result["analysis_artifact"]["canonical_json"])["contract_valid"] is True
    assessment = parse_assessment_record(
        json.loads(result["materials_validation_artifact"]["canonical_json"])
    )
    assert assessment.scientific_status.value == "unsupported"
    checks = {check.validator_id: check for check in assessment.checks}
    assert checks["crystal_plasticity.cpfe_contract_structure"].outcome.value == "pass"
    assert checks["crystal_plasticity.source_provenance_bytes_bound"].outcome.value == "skip"
    assert result["provenance_binding"] == {
        "status": "unverified",
        "mode": "caller_declared_digest_only",
        "digest_syntax_validated": True,
        "source_bytes_resolved": False,
        "source_bytes_rehashed": False,
        "digest_match_independently_verified": False,
    }


def test_cpfe_orientation_output_is_hash_bound_when_inline_budget_is_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(crystal_plasticity_tools, "MAX_INLINE_ORIENTATION_BYTES", 1)

    result = validate_cpfe_contract_typed(contract=_cpfe_contract())

    orientation_evidence = result["validated_contract"]["orientation_evidence"]
    assert result["ok"] is True
    assert result["validated_contract"]["orientations"] is None
    assert orientation_evidence["returned_inline"] is False
    assert len(orientation_evidence["sha256"]) == 64
    assessment = parse_assessment_record(
        json.loads(result["materials_validation_artifact"]["canonical_json"])
    )
    checks = {check.validator_id: check for check in assessment.checks}
    assert checks["crystal_plasticity.cpfe_contract_structure"].outcome.value == "pass"
    assert checks["crystal_plasticity.source_provenance_bytes_bound"].outcome.value == "skip"
    assert assessment.scientific_status.value == "unverified"
    assert assessment.verified is False
    assert result["capability_boundary"]["source_provenance_bytes_bound"] is False


def test_invalid_cpfe_contract_never_attempts_execution_or_returns_partial_results() -> None:
    contract = _cpfe_contract()
    del contract["crss"]["provenance"]

    result = validate_cpfe_contract_typed(contract=contract, attempt_execution=True)

    assert result["ok"] is False
    assert result["contract_valid"] is False
    assert result["execution"] == {
        "attempt_requested": True,
        "attempted": False,
        "supported": False,
        "status": "not_attempted_invalid_contract",
    }
    assert result["partial_results_returned"] is False
    assert result["cpfe_solved"] is False


def test_cpfe_tool_rejects_an_unexpected_return_from_the_unqualified_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(crystal_plasticity_tools, "execute_cpfe", lambda _contract: None)

    result = validate_cpfe_contract_typed(contract=_cpfe_contract(), attempt_execution=True)

    assert result["ok"] is False
    assert result["contract_valid"] is True
    assert result["error"] == "unqualified_cpfe_execution_returned"
    assert result["execution"]["status"] == "unexpected_return_rejected"
    assert result["execution"]["supported"] is False
    assert result["cpfe_solved"] is False
    assert "analysis_artifact" not in result
    assert "materials_validation_artifact" not in result


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "phase_id": "gamma",
            "crystal_structure": "fcc",
            "slip_families": [FCC_111_110],
            "stress_sample": [[0.0, 0.0, 0.0]] * 3,
            "stress_unit": "MPa",
        },
        {
            "phase_id": "gamma",
            "crystal_structure": "fcc",
            "slip_families": [FCC_111_110],
            "rotation_crystal_to_sample": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "stress_sample": [[0.0, 0.0, 0.0]] * 3,
        },
        {
            "phase_id": "gamma",
            "crystal_structure": "fcc",
            "slip_families": [FCC_111_110],
            "rotation_crystal_to_sample": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "hydrostatic_control_stress": math.nan,
            "stress_unit": "MPa",
        },
    ],
)
def test_analytical_tool_fails_closed_on_incomplete_or_nonfinite_analysis_inputs(
    kwargs: dict[str, Any],
) -> None:
    result = analyze_crystal_slip_typed(**kwargs)

    assert result["ok"] is False
    assert result["partial_results_returned"] is False
    assert result["cpfe_solved"] is False


def test_public_tool_schema_is_closed_to_scientific_values_not_code_or_paths() -> None:
    tools = {item.name: item for item in build_crystal_plasticity_tools()}

    assert set(tools) == {
        "materials_analyze_crystal_slip",
        "materials_validate_cpfe_contract",
    }
    analytical_args = tools["materials_analyze_crystal_slip"].args
    assert set(analytical_args) == {
        "phase_id",
        "crystal_structure",
        "slip_families",
        "c_over_a",
        "rotation_crystal_to_sample",
        "stress_sample",
        "stress_unit",
        "load_axis_sample",
        "hydrostatic_control_stress",
    }
    assert all(
        "default" not in analytical_args[name]
        for name in ("phase_id", "crystal_structure", "slip_families")
    )
    contract_args = tools["materials_validate_cpfe_contract"].args
    assert set(contract_args) == {"contract", "attempt_execution"}
    assert "default" not in contract_args["contract"]
    assert contract_args["attempt_execution"]["default"] is False
    encoded_schema = json.dumps({name: item.args for name, item in tools.items()}).lower()
    assert "path" not in encoded_schema
    assert "code" not in encoded_schema
    assert "command" not in encoded_schema
    assert "solver_option" not in encoded_schema


def test_langchain_wrappers_return_the_same_canonical_bounded_records() -> None:
    tools = {item.name: item for item in build_crystal_plasticity_tools()}

    analytical = json.loads(
        tools["materials_analyze_crystal_slip"].func(
            phase_id="gamma",
            crystal_structure="fcc",
            slip_families=[FCC_111_110],
            rotation_crystal_to_sample=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            stress_sample=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 100.0]],
            stress_unit="MPa",
            load_axis_sample=[0.0, 0.0, 1.0],
            hydrostatic_control_stress=123.0,
        )
    )
    cpfe = json.loads(
        tools["materials_validate_cpfe_contract"].func(
            contract=_cpfe_contract(),
            attempt_execution=True,
        )
    )

    assert analytical["summary"]["maximum_schmid_factor"] == pytest.approx(1 / math.sqrt(6))
    assert analytical["materials_validation_artifact"]["scientific_status"] == "unverified"
    assert analytical["phase_structure_assignment"]["status"] == "unverified"
    assert cpfe["contract_valid"] is True
    assert cpfe["execution"]["status"] == "unsupported"
    assert cpfe["materials_validation_artifact"]["scientific_status"] == "unsupported"
