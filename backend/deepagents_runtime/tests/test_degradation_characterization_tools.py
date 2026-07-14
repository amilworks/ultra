"""Qualification for bounded degradation and characterization tools."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from ultra_deepagents.degradation_characterization_tools import (
    TOOL_RESULT_SCHEMA_VERSION,
    build_degradation_characterization_tools,
    calculate_diffraction_profile_metrics_typed,
    convert_uniform_corrosion_typed,
    evaluate_mode_i_lefm_typed,
    evaluate_norton_arrhenius_creep_typed,
    evaluate_oxidation_mass_gain_typed,
    fit_paris_law_typed,
    fit_rigid_registration_typed,
    processing_method_support_typed,
)
from ultra_deepagents.materials.validation import parse_assessment_record


def _degradation_provenance(character: str = "a") -> dict[str, str]:
    return {
        "artifact_id": f"synthetic-{character}",
        "sha256": character * 64,
        "locator": f"fixture://bounded-tool/{character}",
        "citation": "Synthetic typed-tool control; not materials evidence",
    }


def _characterization_provenance(character: str = "b") -> dict[str, str]:
    return {
        "artifact_id": f"synthetic-{character}",
        "sha256": character * 64,
        "locator": f"fixture://bounded-tool/{character}",
        "processing_history_id": "synthetic-none",
    }


def _interval(quantity: str, unit: str, lower: float, upper: float) -> dict[str, Any]:
    return {"quantity": quantity, "unit": unit, "lower": lower, "upper": upper}


def _assert_sealed_success(
    result: dict[str, Any],
    operation: str,
    *,
    provenance_declarations_present: bool = True,
) -> None:
    assert result["schema_version"] == TOOL_RESULT_SCHEMA_VERSION
    assert result["ok"] is True
    assert result["operation"] == operation
    assert result["partial_results_returned"] is False
    encoded = json.dumps(result, allow_nan=False, sort_keys=True)
    assert "NaN" not in encoded and "Infinity" not in encoded

    analysis = result["analysis_artifact"]
    analysis_bytes = analysis["canonical_json"].encode("utf-8")
    assert hashlib.sha256(analysis_bytes).hexdigest() == analysis["sha256"]
    assert len(analysis_bytes) == analysis["size_bytes"]
    retained = json.loads(analysis["canonical_json"])
    assert retained["operation"] == operation
    assert "analysis_artifact" not in retained
    assert retained["input_evidence"] == result["input_evidence"]

    validation = result["materials_validation_artifact"]
    validation_bytes = validation["canonical_json"].encode("utf-8")
    assert hashlib.sha256(validation_bytes).hexdigest() == validation["sha256"]
    assessment = parse_assessment_record(json.loads(validation["canonical_json"]))
    checks = {check.validator_id: check for check in assessment.checks}
    calculation_id = f"materials.bounded_tool.{operation}"
    assert checks[calculation_id].outcome.value == "pass"
    assert checks[calculation_id].evidence[0].sha256 == analysis["sha256"]
    if provenance_declarations_present:
        binding_id = "materials.bounded_tool.provenance_bytes_bound"
        assert result["provenance_binding"] == {
            "status": "unverified",
            "mode": "caller_declared_digest_only",
            "digest_syntax_validated": True,
            "source_bytes_resolved": False,
            "source_bytes_rehashed": False,
            "digest_match_independently_verified": False,
        }
        assert retained["provenance_binding"] == result["provenance_binding"]
        assert checks[binding_id].outcome.value == "skip"
        assert binding_id in assessment.required_validator_ids
        assert assessment.verified is False
        assert assessment.scientific_status.value == "unverified"
        assert validation["validation_scope"].endswith("declaration_only_provenance")
    else:
        assert "provenance_binding" not in result
        assert assessment.verified is True
        assert assessment.scientific_status.value == "verified"


def _lefm_geometry(*, evaluated_parameter: float = 0.1) -> dict[str, Any]:
    return {
        "geometry_id": "synthetic-centered-crack",
        "crack_length_definition": "half crack length from centerline to one tip",
        "nominal_stress_definition": "remote gross-section tensile stress",
        "geometry_factor": 1.12,
        "domain": _interval("crack_length_over_crack_plus_remaining_ligament", "1", 0.01, 0.6),
        "evaluated_parameter": evaluated_parameter,
        "provenance": _degradation_provenance("a"),
    }


def _paris_conditions() -> dict[str, Any]:
    return {
        "material_state_id": "synthetic-state",
        "environment_id": "synthetic-dry-air",
        "load_ratio": 0.1,
        "temperature_k": 298.15,
        "cycle_frequency_hz": 10.0,
        "waveform_id": "constant-amplitude-sine",
        "specimen_thickness_m": 0.012,
        "specimen_geometry_id": "synthetic-compact-tension",
        "delta_k_definition_id": "applied-linear-elastic-Kmax-minus-Kmin",
        "crack_growth_rate_method_id": "incremental-polynomial-reduction-v1",
    }


def test_mode_i_tool_returns_finite_applicability_screen_not_failure_prediction() -> None:
    result = evaluate_mode_i_lefm_typed(
        nominal_tensile_stress_pa=100.0e6,
        crack_length_m=0.01,
        remaining_ligament_m=0.09,
        thickness_m=0.02,
        yield_strength_pa=500.0e6,
        constraint_state="plane_strain",
        minimum_dimension_to_plastic_zone_ratio=20.0,
        geometry=_lefm_geometry(),
        criterion_provenance=_degradation_provenance("c"),
    )

    _assert_sealed_success(result, "mode_i_lefm_screen")
    assert result["result"]["stress_intensity_mpa_sqrt_m"] == pytest.approx(
        1.12 * 100.0 * math.sqrt(math.pi * 0.01)
    )
    assert result["result"]["applicability_passed"] is True
    assert result["capability_boundary"]["failure_or_life_predicted"] is False
    assert result["result"]["standard_compliance_claimed"] is False


def test_mode_i_geometry_coordinate_mismatch_fails_without_partial_result() -> None:
    result = evaluate_mode_i_lefm_typed(
        nominal_tensile_stress_pa=100.0e6,
        crack_length_m=0.01,
        remaining_ligament_m=0.09,
        thickness_m=0.02,
        yield_strength_pa=500.0e6,
        constraint_state="plane_strain",
        minimum_dimension_to_plastic_zone_ratio=20.0,
        geometry=_lefm_geometry(evaluated_parameter=0.2),
        criterion_provenance=_degradation_provenance("c"),
    )

    assert result["ok"] is False
    assert result["error"] == "invalid_degradation_input"
    assert result["partial_results_returned"] is False
    assert "result" not in result
    assert "analysis_artifact" not in result


def test_provenance_cannot_smuggle_or_echo_a_filesystem_path() -> None:
    provenance = _degradation_provenance("c")
    provenance["locator"] = "/private/tmp/untrusted-parameter.json"
    result = evaluate_mode_i_lefm_typed(
        nominal_tensile_stress_pa=100.0e6,
        crack_length_m=0.01,
        remaining_ligament_m=0.09,
        thickness_m=0.02,
        yield_strength_pa=500.0e6,
        constraint_state="plane_strain",
        minimum_dimension_to_plastic_zone_ratio=20.0,
        geometry=_lefm_geometry(),
        criterion_provenance=provenance,
    )

    assert result["ok"] is False
    assert result["error"] == "invalid_typed_input"
    assert result["scientific_result_returned"] is False
    assert "/private/tmp" not in result["message"]


def test_mode_i_failed_applicability_is_a_result_but_never_a_verified_lefm_claim() -> None:
    result = evaluate_mode_i_lefm_typed(
        nominal_tensile_stress_pa=600.0e6,
        crack_length_m=0.01,
        remaining_ligament_m=0.09,
        thickness_m=0.02,
        yield_strength_pa=500.0e6,
        constraint_state="plane_strain",
        minimum_dimension_to_plastic_zone_ratio=20.0,
        geometry=_lefm_geometry(),
        criterion_provenance=_degradation_provenance("c"),
    )

    assert result["ok"] is True
    assert result["result"]["applicability_passed"] is False
    validation = result["materials_validation_artifact"]
    assessment = parse_assessment_record(json.loads(validation["canonical_json"]))
    assert assessment.verified is False
    assert assessment.scientific_status.value == "failed"
    checks = {check.validator_id: check for check in assessment.checks}
    assert checks["materials.bounded_tool.mode_i_lefm_screen"].outcome.value == "fail"
    assert checks["materials.bounded_tool.provenance_bytes_bound"].outcome.value == "skip"


def test_paris_tool_scores_holdout_and_only_predicts_inside_calibration_domain() -> None:
    delta_k = [5.0, 7.0, 10.0, 12.0, 15.0, 18.0, 20.0]
    rate = [2.0e-12 * value**3.1 for value in delta_k]
    result = fit_paris_law_typed(
        delta_k_mpa_sqrt_m=delta_k,
        crack_growth_rate_m_per_cycle=rate,
        calibration_indices=[0, 2, 3, 5, 6],
        held_out_indices=[1, 4],
        conditions=_paris_conditions(),
        observations_provenance=_degradation_provenance("d"),
        prediction_delta_k_mpa_sqrt_m=[8.0],
    )

    _assert_sealed_success(result, "paris_law_fit")
    assert result["result"]["coefficient_c"] == pytest.approx(2.0e-12, rel=1e-12)
    assert result["result"]["exponent_m"] == pytest.approx(3.1, rel=1e-12)
    assert result["result"]["calibration_partition"]["count"] == 5
    assert result["result"]["held_out_partition"]["count"] == 2
    assert result["result"]["predicted_growth_rate_m_per_cycle"] == pytest.approx(
        [1.2606918792651946e-09]
    )
    assert result["capability_boundary"]["component_failure_predicted"] is False


@pytest.mark.parametrize(
    ("held_out_indices", "prediction", "error"),
    [
        ([4, 5], None, "outside_calibration_domain"),
        ([1, 3], [30.0], "outside_calibration_domain"),
    ],
)
def test_paris_tool_refuses_holdout_or_prediction_extrapolation(
    held_out_indices: list[int],
    prediction: list[float] | None,
    error: str,
) -> None:
    delta_k = [10.0, 12.0, 15.0, 18.0, 20.0, 25.0]
    rate = [2.0e-12 * value**3 for value in delta_k]
    calibration = [0, 1, 2, 3] if held_out_indices == [4, 5] else [0, 2, 4, 5]
    result = fit_paris_law_typed(
        delta_k_mpa_sqrt_m=delta_k,
        crack_growth_rate_m_per_cycle=rate,
        calibration_indices=calibration,
        held_out_indices=held_out_indices,
        conditions=_paris_conditions(),
        observations_provenance=_degradation_provenance("d"),
        prediction_delta_k_mpa_sqrt_m=prediction,
    )

    assert result["ok"] is False
    assert result["error"] == error
    assert result["partial_results_returned"] is False


def test_creep_oxidation_and_corrosion_tools_return_only_bounded_conversions() -> None:
    creep = evaluate_norton_arrhenius_creep_typed(
        pre_exponential_per_s=1.0e-4,
        reference_stress_pa=100.0e6,
        stress_exponent=4.0,
        activation_energy_j_per_mol=200_000.0,
        stress_domain_pa=_interval("stress", "Pa", 50.0e6, 300.0e6),
        temperature_domain_k=_interval("temperature", "K", 900.0, 1200.0),
        material_state_id="synthetic-state",
        environment_id="synthetic-argon",
        stress_measure_id="von-Mises-effective-stress",
        model_provenance=_degradation_provenance("e"),
        stress_pa=200.0e6,
        temperature_k=1000.0,
    )
    _assert_sealed_success(creep, "norton_arrhenius_secondary_creep")
    expected_creep = 1.0e-4 * 2.0**4 * math.exp(-200_000.0 / (8.31446261815324 * 1000.0))
    assert creep["result"]["effective_secondary_creep_rate_per_s"] == pytest.approx(expected_creep)
    assert creep["capability_boundary"]["rupture_or_damage_calculated"] is False

    linear = evaluate_oxidation_mass_gain_typed(
        law="linear",
        rate_constant=1.0e-3,
        rate_constant_unit="kg*m^-2*s^-1",
        initial_areal_mass_gain_kg_per_m2=1.0e-2,
        time_domain_s=_interval("time", "s", 0.0, 100.0),
        temperature_domain_k=_interval("temperature", "K", 1073.0, 1073.0),
        material_state_id="synthetic-state",
        environment_id="synthetic-dry-air",
        area_basis_id="initial-total-geometric-exposed-area",
        model_provenance=_degradation_provenance("f"),
        exposure_time_s=10.0,
        temperature_k=1073.0,
    )
    _assert_sealed_success(linear, "oxidation_areal_mass_gain")
    assert linear["result"]["areal_mass_gain_kg_per_m2"] == pytest.approx(0.02)
    assert linear["result"]["model"]["rate_constant_unit"] == "kg*m^-2*s^-1"
    assert linear["capability_boundary"]["oxide_thickness_calculated"] is False

    parabolic = evaluate_oxidation_mass_gain_typed(
        law="parabolic",
        rate_constant=4.0e-4,
        rate_constant_unit="kg^2*m^-4*s^-1",
        initial_areal_mass_gain_kg_per_m2=3.0e-2,
        time_domain_s=_interval("time", "s", 0.0, 100.0),
        temperature_domain_k=_interval("temperature", "K", 1073.0, 1073.0),
        material_state_id="synthetic-state",
        environment_id="synthetic-dry-air",
        area_basis_id="initial-total-geometric-exposed-area",
        model_provenance=_degradation_provenance("f"),
        exposure_time_s=4.0,
        temperature_k=1073.0,
    )
    _assert_sealed_success(parabolic, "oxidation_areal_mass_gain")
    assert parabolic["result"]["areal_mass_gain_kg_per_m2"] == pytest.approx(0.05)
    assert parabolic["result"]["model"]["rate_constant_unit"] == "kg^2*m^-4*s^-1"

    corrosion = convert_uniform_corrosion_typed(
        corrosion_current_density_a_per_m2=1.0,
        equivalent_mass_kg_per_mol_electron=0.055845 / 2.0,
        density_kg_per_m3=7874.0,
        current_efficiency=0.8,
        duration_s=365.25 * 24.0 * 3600.0,
        material_state_id="synthetic-state",
        environment_id="synthetic-cell",
        current_density_area_basis_id="geometric-electrode-area-before-exposure",
        current_density_provenance=_degradation_provenance("1"),
        equivalent_mass_provenance=_degradation_provenance("2"),
        density_provenance=_degradation_provenance("3"),
        efficiency_provenance=_degradation_provenance("4"),
    )
    _assert_sealed_success(corrosion, "faraday_uniform_corrosion_conversion")
    assert corrosion["result"]["average_uniform_penetration_m"] > 0
    assert corrosion["capability_boundary"]["pitting_or_crevice_depth_calculated"] is False


def test_constant_law_oxidation_rejects_multi_temperature_calibration() -> None:
    result = evaluate_oxidation_mass_gain_typed(
        law="linear",
        rate_constant=1.0e-3,
        rate_constant_unit="kg*m^-2*s^-1",
        initial_areal_mass_gain_kg_per_m2=1.0e-2,
        time_domain_s=_interval("time", "s", 0.0, 100.0),
        temperature_domain_k=_interval("temperature", "K", 1073.0, 1173.0),
        material_state_id="synthetic-state",
        environment_id="synthetic-dry-air",
        area_basis_id="initial-total-geometric-exposed-area",
        model_provenance=_degradation_provenance("f"),
        exposure_time_s=10.0,
        temperature_k=1073.0,
    )

    assert result["ok"] is False
    assert result["error"] == "invalid_degradation_input"
    assert "singleton isothermal temperature" in result["message"]
    assert result["scientific_result_returned"] is False


def test_constant_law_oxidation_rejects_second_temperature_with_same_constant() -> None:
    result = evaluate_oxidation_mass_gain_typed(
        law="linear",
        rate_constant=1.0e-3,
        rate_constant_unit="kg*m^-2*s^-1",
        initial_areal_mass_gain_kg_per_m2=1.0e-2,
        time_domain_s=_interval("time", "s", 0.0, 100.0),
        temperature_domain_k=_interval("temperature", "K", 1073.0, 1073.0),
        material_state_id="synthetic-state",
        environment_id="synthetic-dry-air",
        area_basis_id="initial-total-geometric-exposed-area",
        model_provenance=_degradation_provenance("f"),
        exposure_time_s=10.0,
        temperature_k=1074.0,
    )

    assert result["ok"] is False
    assert result["error"] == "outside_calibration_domain"
    assert result["scientific_result_returned"] is False


@pytest.mark.parametrize(
    ("law", "rate_constant_unit"),
    [
        ("linear", "kg^2*m^-4*s^-1"),
        ("parabolic", "kg*m^-2*s^-1"),
        ("linear", "kg/m^2/s"),
    ],
)
def test_constant_law_oxidation_rejects_wrong_or_alias_rate_constant_units(
    law: str,
    rate_constant_unit: str,
) -> None:
    result = evaluate_oxidation_mass_gain_typed(
        law=law,
        rate_constant=1.0e-3,
        rate_constant_unit=rate_constant_unit,
        initial_areal_mass_gain_kg_per_m2=1.0e-2,
        time_domain_s=_interval("time", "s", 0.0, 100.0),
        temperature_domain_k=_interval("temperature", "K", 1073.0, 1073.0),
        material_state_id="synthetic-state",
        environment_id="synthetic-dry-air",
        area_basis_id="initial-total-geometric-exposed-area",
        model_provenance=_degradation_provenance("f"),
        exposure_time_s=10.0,
        temperature_k=1073.0,
    )

    assert result["ok"] is False
    assert result["error"] == "invalid_degradation_input"
    assert "rate_constant_unit must be exactly" in result["message"]
    assert result["scientific_result_returned"] is False


@pytest.mark.parametrize(
    ("runner", "expected_error"),
    [
        (
            lambda: evaluate_norton_arrhenius_creep_typed(
                pre_exponential_per_s=1.0e-4,
                reference_stress_pa=100.0e6,
                stress_exponent=4.0,
                activation_energy_j_per_mol=200_000.0,
                stress_domain_pa=_interval("stress", "Pa", 50.0e6, 300.0e6),
                temperature_domain_k=_interval("temperature", "K", 900.0, 1200.0),
                material_state_id="synthetic-state",
                environment_id="synthetic-argon",
                stress_measure_id="von-Mises-effective-stress",
                model_provenance=_degradation_provenance("e"),
                stress_pa=400.0e6,
                temperature_k=1000.0,
            ),
            "outside_calibration_domain",
        ),
        (
            lambda: evaluate_oxidation_mass_gain_typed(
                law="cubic",
                rate_constant=1.0e-3,
                rate_constant_unit="kg*m^-2*s^-1",
                initial_areal_mass_gain_kg_per_m2=0.0,
                time_domain_s=_interval("time", "s", 0.0, 100.0),
                temperature_domain_k=_interval("temperature", "K", 1000.0, 1100.0),
                material_state_id="synthetic-state",
                environment_id="air",
                area_basis_id="initial-area",
                model_provenance=_degradation_provenance("f"),
                exposure_time_s=10.0,
                temperature_k=1073.0,
            ),
            "invalid_degradation_input",
        ),
        (
            lambda: convert_uniform_corrosion_typed(
                corrosion_current_density_a_per_m2=1.0,
                equivalent_mass_kg_per_mol_electron=0.055845 / 2.0,
                density_kg_per_m3=7874.0,
                current_efficiency=1.1,
                duration_s=1.0,
                material_state_id="state",
                environment_id="cell",
                current_density_area_basis_id="area",
                current_density_provenance=_degradation_provenance("1"),
                equivalent_mass_provenance=_degradation_provenance("2"),
                density_provenance=_degradation_provenance("3"),
                efficiency_provenance=_degradation_provenance("4"),
            ),
            "invalid_degradation_input",
        ),
    ],
)
def test_scalar_degradation_tools_fail_closed_on_invalid_scope(
    runner: Any,
    expected_error: str,
) -> None:
    result = runner()
    assert result["ok"] is False
    assert result["error"] == expected_error
    assert result["scientific_result_returned"] is False


def test_diffraction_metrics_tool_reports_statistical_convention_without_refinement() -> None:
    result = calculate_diffraction_profile_metrics_typed(
        coordinate=[20.0, 20.1, 20.2, 20.3],
        observed_intensity=[10.0, 20.0, 30.0, 40.0],
        calculated_intensity=[9.0, 22.0, 999.0, 39.0],
        coordinate_unit="degree_2theta",
        observed_intensity_unit="count",
        calculated_intensity_unit="count",
        observed_provenance=_characterization_provenance("a"),
        calculated_provenance=_characterization_provenance("b"),
        included_mask=[True, True, False, True],
        uncertainties=[1.0, 2.0, 0.0, 2.0],
        uncertainty_semantics="independent_absolute_1sigma",
        refined_parameter_count=1,
        independent_constraint_count=0,
    )

    _assert_sealed_success(result, "diffraction_profile_metrics")
    metrics = result["result"]
    assert metrics["rp"] == pytest.approx(0.05714285714285714)
    assert metrics["rwp"] == pytest.approx(0.06123724356957945)
    assert metrics["rexp"] == pytest.approx(0.05773502691896258)
    assert metrics["chi_square"] == pytest.approx(2.25)
    assert metrics["reduced_chi_square"] == pytest.approx(1.125)
    assert metrics["goodness_of_fit"] == pytest.approx(1.0606601717798212)
    assert metrics["degrees_of_freedom"] == 2
    assert result["capability_boundary"]["rietveld_refinement_performed"] is False
    assert metrics["validation_only"] is True


@pytest.mark.parametrize(
    "override",
    [
        {"calculated_intensity_unit": "arbitrary_unit"},
        {"uncertainty_semantics": "standard_error"},
        {"coordinate": [20.0, 19.9, 20.2, 20.3]},
    ],
)
def test_diffraction_metrics_tool_rejects_unit_semantic_or_axis_ambiguity(
    override: dict[str, Any],
) -> None:
    request: dict[str, Any] = {
        "coordinate": [20.0, 20.1, 20.2, 20.3],
        "observed_intensity": [10.0, 12.0, 8.0, 6.0],
        "calculated_intensity": [9.0, 13.0, 8.0, 7.0],
        "coordinate_unit": "degree_2theta",
        "observed_intensity_unit": "count",
        "calculated_intensity_unit": "count",
        "observed_provenance": _characterization_provenance("a"),
        "calculated_provenance": _characterization_provenance("b"),
        "uncertainties": [1.0, 1.0, 1.0, 1.0],
        "uncertainty_semantics": "independent_absolute_1sigma",
        "refined_parameter_count": 1,
    }
    request.update(override)
    result = calculate_diffraction_profile_metrics_typed(**request)

    assert result["ok"] is False
    assert result["error"] == "invalid_characterization_input"
    assert result["partial_results_returned"] is False


def test_large_inline_profile_fails_at_typed_input_cap_before_numerical_work() -> None:
    oversized = list(range(200_000))
    result = calculate_diffraction_profile_metrics_typed(
        coordinate=oversized,
        observed_intensity=oversized,
        calculated_intensity=oversized,
        coordinate_unit="degree_2theta",
        observed_intensity_unit="count",
        calculated_intensity_unit="count",
        observed_provenance=_characterization_provenance("a"),
        calculated_provenance=_characterization_provenance("b"),
    )

    assert result["ok"] is False
    assert result["error"] == "typed_input_too_large"
    assert result["scientific_result_returned"] is False
    assert "result" not in result


def _registration_points() -> tuple[list[list[float]], list[list[float]]]:
    source = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 0.5, 2.0],
        ]
    )
    angle = math.radians(30.0)
    rotation = np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    target = source @ rotation.T + np.asarray([2.0, -1.0, 0.5])
    return source.tolist(), target.tolist()


def test_registration_tool_scores_and_retains_every_fixed_partition_residual() -> None:
    source, target = _registration_points()
    result = fit_rigid_registration_typed(
        source_points=source,
        target_points=target,
        source_frame_id="ebsd-stage",
        target_frame_id="apt-reconstruction",
        source_coordinate_unit="um",
        target_coordinate_unit="um",
        source_provenance=_characterization_provenance("c"),
        target_provenance=_characterization_provenance("d"),
        calibration_indices=[0, 1, 2, 3],
        held_out_indices=[4, 5],
    )

    _assert_sealed_success(result, "held_out_rigid_registration")
    registration = result["result"]
    assert registration["rotation_determinant"] == pytest.approx(1.0)
    assert registration["held_out_statistics"]["maximum"] < 1e-12
    assert registration["calibration_partition"]["count"] == 4
    assert registration["held_out_partition"]["count"] == 2
    assert registration["calibration_indices"] == [0, 1, 2, 3]
    assert registration["held_out_indices"] == [4, 5]
    assert max(registration["calibration_residual_norms"]) < 1e-12
    assert max(registration["held_out_residual_norms"]) < 1e-12
    assert result["capability_boundary"]["feature_identity_established"] is False


def test_ac02_natural_prompt_registration_matches_noisy_heldout_oracle() -> None:
    result = fit_rigid_registration_typed(
        source_points=[
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [3.0, -1.0],
            [-2.0, 0.5],
        ],
        target_points=[
            [2.5, -1.25],
            [4.097271020094586, -0.046369953695903],
            [1.898184976847952, -0.451364489952707],
            [2.796820486895245, -0.049549466800659],
            [5.447721553293927, -0.243190440591148],
            [0.60182146832939, -1.75431229128045],
        ],
        source_frame_id="ebsd-detector-frame",
        target_frame_id="apt-reconstruction-frame",
        source_coordinate_unit="um",
        target_coordinate_unit="um",
        source_provenance=_characterization_provenance("c"),
        target_provenance=_characterization_provenance("d"),
        calibration_indices=[0, 1, 2],
        held_out_indices=[3, 4, 5],
    )

    _assert_sealed_success(result, "held_out_rigid_registration")
    registration = result["result"]
    np.testing.assert_allclose(
        registration["rotation_source_to_target"],
        [
            [0.7986355100472928, -0.6018150231520483],
            [0.6018150231520483, 0.7986355100472928],
        ],
        rtol=0.0,
        atol=1e-14,
    )
    assert registration["translation_source_to_target"] == pytest.approx([2.5, -1.25])
    assert registration["rotation_determinant"] == pytest.approx(1.0)
    assert registration["held_out_residual_norms"] == pytest.approx([0.223606797749979, 0.05, 0.3])
    assert registration["held_out_statistics"]["rmse"] == pytest.approx(0.21794494717703367)


def test_registration_rejects_incomplete_partition_and_reflection() -> None:
    source, target = _registration_points()
    incomplete = fit_rigid_registration_typed(
        source_points=source,
        target_points=target,
        source_frame_id="source",
        target_frame_id="target",
        source_coordinate_unit="um",
        target_coordinate_unit="um",
        source_provenance=_characterization_provenance("c"),
        target_provenance=_characterization_provenance("d"),
        calibration_indices=[0, 1, 2, 3],
        held_out_indices=[4],
    )
    assert incomplete["ok"] is False
    assert incomplete["error"] == "invalid_characterization_input"

    reflected = np.asarray(source)
    reflected[:, 0] *= -1.0
    reflection = fit_rigid_registration_typed(
        source_points=source,
        target_points=reflected.tolist(),
        source_frame_id="source",
        target_frame_id="target",
        source_coordinate_unit="um",
        target_coordinate_unit="um",
        source_provenance=_characterization_provenance("c"),
        target_provenance=_characterization_provenance("d"),
        calibration_indices=[0, 1, 2, 3],
        held_out_indices=[4, 5],
    )
    assert reflection["ok"] is False
    assert reflection["error"] == "improper_rotation_required"
    assert reflection["scientific_result_returned"] is False


def test_registration_refuses_correspondence_count_that_could_overrun_typed_output() -> None:
    points = [[0.0, 0.0]] * 20_001
    result = fit_rigid_registration_typed(
        source_points=points,
        target_points=points,
        source_frame_id="source",
        target_frame_id="target",
        source_coordinate_unit="um",
        target_coordinate_unit="um",
        source_provenance=_characterization_provenance("c"),
        target_provenance=_characterization_provenance("d"),
        calibration_indices=[0, 1, 2],
        held_out_indices=list(range(3, 20_001)),
    )

    assert result["ok"] is False
    assert result["error"] == "typed_input_too_large"
    assert result["scientific_result_returned"] is False


def test_processing_support_is_discovery_only_and_phase_field_remains_external() -> None:
    result = processing_method_support_typed()

    _assert_sealed_success(
        result,
        "processing_method_support",
        provenance_declarations_present=False,
    )
    methods = result["result"]["methods"]
    assert methods["scheil_gulliver"]["status"] == "qualified_runtime"
    assert methods["back_diffusion"]["tool"] == "materials_run_diffusion_1d"
    assert methods["precipitation"]["tool"] == "materials_run_binary_precipitation_kwn"
    assert methods["phase_field"]["status"] == "requires_external_hpc_solver"
    assert result["capability_boundary"]["scheil_execution_performed"] is False
    assert result["capability_boundary"]["phase_field_execution_performed"] is False


def test_public_tool_schemas_are_scientific_only_and_closed_nested_records() -> None:
    tools = {item.name: item for item in build_degradation_characterization_tools()}

    assert set(tools) == {
        "materials_evaluate_mode_i_lefm",
        "materials_fit_paris_law",
        "materials_evaluate_norton_arrhenius_creep",
        "materials_evaluate_oxidation_mass_gain",
        "materials_convert_uniform_corrosion",
        "materials_calculate_diffraction_profile_metrics",
        "materials_fit_held_out_rigid_registration",
        "materials_processing_method_support",
    }
    assert tools["materials_processing_method_support"].args == {}
    oxidation_args = tools["materials_evaluate_oxidation_mass_gain"].args
    assert "rate_constant_unit" in oxidation_args
    assert "default" not in oxidation_args["rate_constant_unit"]
    encoded_schema = json.dumps({name: item.args for name, item in tools.items()}).lower()
    assert '"code"' not in encoded_schema
    assert '"command"' not in encoded_schema
    assert '"path"' not in encoded_schema
    assert '"solver_options"' not in encoded_schema
    full_schemas = json.dumps(
        {name: item.tool_call_schema.model_json_schema() for name, item in tools.items()}
    ).lower()
    assert full_schemas.count('"additionalproperties": false') >= 5


def test_langchain_tool_invocation_returns_typed_json_not_python_repr() -> None:
    tools = {item.name: item for item in build_degradation_characterization_tools()}
    rendered = tools["materials_processing_method_support"].invoke({})
    decoded = json.loads(rendered)

    assert decoded["ok"] is True
    assert decoded["operation"] == "processing_method_support"
    assert decoded["result"]["methods"]["phase_field"]["status"] == ("requires_external_hpc_solver")

    lefm_rendered = tools["materials_evaluate_mode_i_lefm"].invoke(
        {
            "nominal_tensile_stress_pa": 100.0e6,
            "crack_length_m": 0.01,
            "remaining_ligament_m": 0.09,
            "thickness_m": 0.02,
            "yield_strength_pa": 500.0e6,
            "constraint_state": "plane_strain",
            "minimum_dimension_to_plastic_zone_ratio": 20.0,
            "geometry": _lefm_geometry(),
            "criterion_provenance": _degradation_provenance("c"),
        }
    )
    lefm_decoded = json.loads(lefm_rendered)
    assert lefm_decoded["ok"] is True
    assert lefm_decoded["result"]["applicability_passed"] is True

    with pytest.raises(ValidationError, match="nominal_tensile_stress_pa"):
        tools["materials_evaluate_mode_i_lefm"].invoke(
            {
                "nominal_tensile_stress_pa": True,
                "crack_length_m": 0.01,
                "remaining_ligament_m": 0.09,
                "thickness_m": 0.02,
                "yield_strength_pa": 500.0e6,
                "constraint_state": "plane_strain",
                "minimum_dimension_to_plastic_zone_ratio": 20.0,
                "geometry": _lefm_geometry(),
                "criterion_provenance": _degradation_provenance("c"),
            }
        )

    with pytest.raises(ValidationError, match="rate_constant_unit"):
        tools["materials_evaluate_oxidation_mass_gain"].invoke(
            {
                "law": "linear",
                "rate_constant": 1.0e-3,
                "initial_areal_mass_gain_kg_per_m2": 1.0e-2,
                "time_domain_s": _interval("time", "s", 0.0, 100.0),
                "temperature_domain_k": _interval("temperature", "K", 1073.0, 1073.0),
                "material_state_id": "synthetic-state",
                "environment_id": "synthetic-dry-air",
                "area_basis_id": "initial-total-geometric-exposed-area",
                "model_provenance": _degradation_provenance("f"),
                "exposure_time_s": 10.0,
                "temperature_k": 1073.0,
            }
        )


def test_builder_can_register_each_narrow_group_without_unrelated_schema_cost() -> None:
    degradation = {
        item.name
        for item in build_degradation_characterization_tools(
            include_characterization=False,
            include_processing_support=False,
        )
    }
    characterization = {
        item.name
        for item in build_degradation_characterization_tools(
            include_degradation=False,
            include_processing_support=False,
        )
    }
    processing = {
        item.name
        for item in build_degradation_characterization_tools(
            include_degradation=False,
            include_characterization=False,
        )
    }

    assert degradation == {
        "materials_evaluate_mode_i_lefm",
        "materials_fit_paris_law",
        "materials_evaluate_norton_arrhenius_creep",
        "materials_evaluate_oxidation_mass_gain",
        "materials_convert_uniform_corrosion",
    }
    assert characterization == {
        "materials_calculate_diffraction_profile_metrics",
        "materials_fit_held_out_rigid_registration",
    }
    assert processing == {"materials_processing_method_support"}


def test_natural_stress_prompts_name_first_class_tools_not_python_discovery() -> None:
    prompt_catalog = (
        Path(__file__).parent / "fixtures" / "materials_natural_prompts" / "README.md"
    ).read_text(encoding="utf-8")

    for name in (
        "materials_evaluate_mode_i_lefm",
        "materials_fit_paris_law",
        "materials_evaluate_norton_arrhenius_creep",
        "materials_evaluate_oxidation_mass_gain",
        "materials_convert_uniform_corrosion",
        "materials_calculate_diffraction_profile_metrics",
        "materials_fit_held_out_rigid_registration",
        "materials_processing_method_support",
    ):
        assert f"`{name}`" in prompt_catalog
    assert "call `processing_method_support()` in the sandbox" not in prompt_catalog
    assert "Use `calculate_diffraction_profile_metrics`" not in prompt_catalog
