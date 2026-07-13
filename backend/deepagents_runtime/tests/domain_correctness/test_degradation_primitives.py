from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from ultra_deepagents.materials.degradation import (
    FARADAY_CONSTANT_C_PER_MOL,
    CalibrationDomainError,
    ClosedInterval,
    CorrosionPenetrationInputs,
    DegradationInputError,
    EvidenceProvenance,
    GeometryFactorCalibration,
    NortonArrheniusCreepModel,
    OxidationKineticsModel,
    ParisTestConditions,
    convert_corrosion_current_to_uniform_penetration,
    evaluate_mode_i_lefm,
    evaluate_norton_arrhenius_creep_rate,
    evaluate_oxidation_mass_gain,
    fit_paris_law,
)


def _provenance(name: str = "source") -> EvidenceProvenance:
    return EvidenceProvenance(
        artifact_id=name,
        sha256=(name[0].lower() if name[0].lower() in "abcdef" else "a") * 64,
        locator=f"resource://{name}",
        citation=f"Qualified source for {name}",
    )


def _geometry() -> GeometryFactorCalibration:
    return GeometryFactorCalibration(
        geometry_id="finite-width-centered-crack",
        crack_length_definition="half crack length from centerline to one crack tip",
        nominal_stress_definition="remote gross-section tensile stress",
        geometry_factor=1.12,
        domain=ClosedInterval(
            quantity="crack_length_over_crack_plus_remaining_ligament",
            unit="1",
            lower=0.01,
            upper=0.6,
        ),
        evaluated_parameter=0.1,
        provenance=_provenance("geometry"),
    )


def _lefm_kwargs() -> dict[str, object]:
    return {
        "nominal_tensile_stress_pa": 100.0e6,
        "crack_length_m": 0.01,
        "remaining_ligament_m": 0.09,
        "thickness_m": 0.02,
        "yield_strength_pa": 500.0e6,
        "constraint_state": "plane_strain",
        "minimum_dimension_to_plastic_zone_ratio": 20.0,
        "geometry": _geometry(),
        "criterion_provenance": _provenance("criterion"),
    }


def _paris_conditions() -> ParisTestConditions:
    return ParisTestConditions(
        material_state_id="heat-treatment-A-grain-state-7",
        environment_id="dry-laboratory-air-batch-2",
        load_ratio=0.1,
        temperature_k=298.15,
        cycle_frequency_hz=10.0,
        waveform_id="constant-amplitude-sine",
        specimen_thickness_m=0.012,
        specimen_geometry_id="compact-tension-lot-2",
        delta_k_definition_id="applied-linear-elastic-Kmax-minus-Kmin",
        crack_growth_rate_method_id="incremental-polynomial-reduction-v1",
    )


def _paris_data() -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    delta_k = np.array([5.0, 7.0, 10.0, 12.0, 15.0, 18.0, 20.0])
    growth = 2.0e-12 * delta_k**3.1
    return delta_k, growth, [0, 2, 3, 5, 6], [1, 4]


def _creep_model() -> NortonArrheniusCreepModel:
    return NortonArrheniusCreepModel(
        pre_exponential_per_s=1.0e-4,
        reference_stress_pa=100.0e6,
        stress_exponent=4.0,
        activation_energy_j_per_mol=200_000.0,
        stress_domain_pa=ClosedInterval("stress", "Pa", 50.0e6, 300.0e6),
        temperature_domain_k=ClosedInterval("temperature", "K", 900.0, 1200.0),
        material_state_id="state-A",
        environment_id="argon-1bar",
        stress_measure_id="von-Mises-effective-stress",
        provenance=_provenance("creep"),
    )


def _oxidation_model(law: str = "linear") -> OxidationKineticsModel:
    rate = 1.0e-3 if law == "linear" else 4.0e-4
    initial = 1.0e-2 if law == "linear" else 3.0e-2
    return OxidationKineticsModel(
        law=law,
        rate_constant=rate,
        rate_constant_unit=("kg*m^-2*s^-1" if law == "linear" else "kg^2*m^-4*s^-1"),
        initial_areal_mass_gain_kg_per_m2=initial,
        time_domain_s=ClosedInterval("time", "s", 0.0, 100.0),
        temperature_domain_k=ClosedInterval("temperature", "K", 1073.0, 1073.0),
        material_state_id="coupon-state-A",
        environment_id="dry-air-flow-cell-1",
        area_basis_id="initial-total-geometric-exposed-area",
        provenance=_provenance("oxidation"),
    )


def _corrosion_inputs() -> CorrosionPenetrationInputs:
    return CorrosionPenetrationInputs(
        corrosion_current_density_a_per_m2=1.0,
        equivalent_mass_kg_per_mol_electron=0.055845 / 2.0,
        density_kg_per_m3=7874.0,
        current_efficiency=0.8,
        duration_s=365.25 * 24.0 * 3600.0,
        material_state_id="iron-reference-state",
        environment_id="polarization-cell-1",
        current_density_area_basis_id="geometric-electrode-area-before-exposure",
        current_density_provenance=_provenance("current"),
        equivalent_mass_provenance=_provenance("equivalent"),
        density_provenance=_provenance("density"),
        efficiency_provenance=_provenance("efficiency"),
    )


def test_mode_i_lefm_matches_analytical_k_and_plastic_zone() -> None:
    result = evaluate_mode_i_lefm(**_lefm_kwargs())
    expected_k = 1.12 * 100.0e6 * math.sqrt(math.pi * 0.01)
    expected_zone = (expected_k / 500.0e6) ** 2 / (6.0 * math.pi)

    assert result.stress_intensity_pa_sqrt_m == pytest.approx(expected_k)
    assert result.stress_intensity_mpa_sqrt_m == pytest.approx(expected_k / 1.0e6)
    assert result.plastic_zone_radius_m == pytest.approx(expected_zone)
    assert result.minimum_dimension_to_plastic_zone_ratio == pytest.approx(0.01 / expected_zone)
    assert result.derived_geometry_parameter == pytest.approx(0.1)
    assert result.applicability_passed is True
    assert all(check.passed for check in result.applicability_checks)
    assert result.standard_compliance_claimed is False
    assert "not an ASTM E399 test" in result.limitation


def test_mode_i_lefm_reports_failed_applicability_without_calling_it_valid() -> None:
    result = evaluate_mode_i_lefm(
        **{
            **_lefm_kwargs(),
            "nominal_tensile_stress_pa": 600.0e6,
            "minimum_dimension_to_plastic_zone_ratio": 1.0e6,
        }
    )

    assert result.applicability_passed is False
    checks = {check.check_id: check.passed for check in result.applicability_checks}
    assert checks["nominal-stress-below-yield-strength"] is False
    assert checks["small-scale-yielding-dimension-separation"] is False


def test_mode_i_plane_stress_plastic_zone_is_three_times_plane_strain() -> None:
    plane_strain = evaluate_mode_i_lefm(**_lefm_kwargs())
    plane_stress = evaluate_mode_i_lefm(
        **{
            **_lefm_kwargs(),
            "constraint_state": "plane_stress",
        }
    )

    assert plane_stress.plastic_zone_radius_m == pytest.approx(
        3.0 * plane_strain.plastic_zone_radius_m
    )


def test_mode_i_refuses_zero_load_instead_of_emitting_nonfinite_audit_values() -> None:
    with pytest.raises(DegradationInputError, match="strictly positive"):
        evaluate_mode_i_lefm(
            **{
                **_lefm_kwargs(),
                "nominal_tensile_stress_pa": 0.0,
            }
        )


def test_mode_i_rejects_a_geometry_coordinate_inconsistent_with_specimen_dimensions() -> None:
    with pytest.raises(DegradationInputError, match="evaluated_parameter does not match"):
        evaluate_mode_i_lefm(
            **{
                **_lefm_kwargs(),
                "remaining_ligament_m": 0.04,
            }
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("crack_length_m", 0.0, "strictly positive"),
        ("remaining_ligament_m", -1.0, "strictly positive"),
        ("thickness_m", math.inf, "finite scalar"),
        ("yield_strength_pa", True, "finite scalar"),
        ("constraint_state", "axisymmetric", "plane_stress"),
        ("minimum_dimension_to_plastic_zone_ratio", 0.0, "strictly positive"),
    ],
)
def test_mode_i_rejects_ambiguous_or_nonphysical_inputs(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(DegradationInputError, match=message):
        evaluate_mode_i_lefm(**{**_lefm_kwargs(), field: value})


def test_geometry_factor_refuses_extrapolation_and_non_dimensionless_domain() -> None:
    with pytest.raises(CalibrationDomainError, match="outside calibrated"):
        GeometryFactorCalibration(
            geometry_id="test",
            crack_length_definition="half length",
            nominal_stress_definition="remote tensile stress",
            geometry_factor=1.0,
            domain=ClosedInterval("crack_length_over_crack_plus_remaining_ligament", "1", 0.1, 0.5),
            evaluated_parameter=0.7,
            provenance=_provenance(),
        )
    with pytest.raises(DegradationInputError, match="unit must be '1'"):
        GeometryFactorCalibration(
            geometry_id="test",
            crack_length_definition="half length",
            nominal_stress_definition="remote tensile stress",
            geometry_factor=1.0,
            domain=ClosedInterval("crack_length_over_crack_plus_remaining_ligament", "m", 0.1, 0.5),
            evaluated_parameter=0.2,
            provenance=_provenance(),
        )


def test_paris_fit_recovers_exact_power_law_and_scores_holdout() -> None:
    delta_k, growth, calibration, held_out = _paris_data()
    fit = fit_paris_law(
        delta_k,
        growth,
        calibration_indices=calibration,
        held_out_indices=held_out,
        conditions=_paris_conditions(),
        observations_provenance=_provenance("fatigue"),
    )

    assert fit.coefficient_c == pytest.approx(2.0e-12, rel=1.0e-12)
    assert fit.exponent_m == pytest.approx(3.1, rel=1.0e-12)
    assert fit.calibration_residuals.root_mean_square_log_error < 1.0e-12
    assert fit.held_out_residuals.root_mean_square_log_error < 1.0e-12
    assert fit.delta_k_domain_mpa_sqrt_m.lower == 5.0
    assert fit.delta_k_domain_mpa_sqrt_m.upper == 20.0
    assert fit.standard_compliance_claimed is False
    assert fit.regression_space == "natural_log_da_dN_vs_natural_log_delta_K"
    assert fit.weighting_scheme == "unweighted_ordinary_least_squares"
    assert fit.validation_only is True
    predicted = fit.predict_growth_rate_m_per_cycle(
        [6.0, 19.0],
        conditions=_paris_conditions(),
    )
    np.testing.assert_allclose(predicted, 2.0e-12 * np.array([6.0, 19.0]) ** 3.1)
    assert predicted.flags.writeable is False


def test_paris_heldout_values_cannot_leak_into_parameter_fit() -> None:
    delta_k, growth, calibration, held_out = _paris_data()
    clean = fit_paris_law(
        delta_k,
        growth,
        calibration_indices=calibration,
        held_out_indices=held_out,
        conditions=_paris_conditions(),
        observations_provenance=_provenance("fatigue"),
    )
    corrupted_growth = growth.copy()
    corrupted_growth[held_out] *= np.array([10.0, 0.1])
    corrupted = fit_paris_law(
        delta_k,
        corrupted_growth,
        calibration_indices=calibration,
        held_out_indices=held_out,
        conditions=_paris_conditions(),
        observations_provenance=_provenance("fatigue"),
    )

    assert corrupted.coefficient_c == clean.coefficient_c
    assert corrupted.exponent_m == clean.exponent_m
    assert corrupted.calibration_residuals == clean.calibration_residuals
    assert corrupted.held_out_residuals.root_mean_square_log_error > 2.0


def test_paris_prediction_refuses_delta_k_extrapolation_and_condition_transfer() -> None:
    delta_k, growth, calibration, held_out = _paris_data()
    fit = fit_paris_law(
        delta_k,
        growth,
        calibration_indices=calibration,
        held_out_indices=held_out,
        conditions=_paris_conditions(),
        observations_provenance=_provenance("fatigue"),
    )

    with pytest.raises(CalibrationDomainError, match="outside"):
        fit.predict_growth_rate_m_per_cycle([4.99], conditions=_paris_conditions())
    with pytest.raises(CalibrationDomainError, match="conditions differ"):
        fit.predict_growth_rate_m_per_cycle(
            [10.0],
            conditions=replace(_paris_conditions(), environment_id="salt-fog"),
        )
    with pytest.raises(CalibrationDomainError, match="conditions differ"):
        fit.predict_growth_rate_m_per_cycle(
            [10.0],
            conditions=replace(_paris_conditions(), load_ratio=0.11),
        )
    with pytest.raises(CalibrationDomainError, match="conditions differ"):
        fit.predict_growth_rate_m_per_cycle(
            [10.0],
            conditions=replace(
                _paris_conditions(),
                delta_k_definition_id="closure-corrected-effective-delta-K",
            ),
        )


def test_paris_holdout_must_be_interpolation_evidence_not_extrapolation() -> None:
    delta_k, growth, _calibration, _held_out = _paris_data()
    with pytest.raises(CalibrationDomainError, match="interpolation rather than extrapolation"):
        fit_paris_law(
            delta_k,
            growth,
            calibration_indices=[1, 2, 3, 4, 5],
            held_out_indices=[0, 6],
            conditions=_paris_conditions(),
            observations_provenance=_provenance("fatigue"),
        )


@pytest.mark.parametrize(
    ("calibration", "held_out", "message"),
    [
        ([0, 1], [2, 3, 4, 5, 6], "at least 3"),
        ([0, 1, 2, 3, 4], [5], "at least 2"),
        ([0, 1, 2, 3, 4], [4, 5], "disjoint"),
        ([0, 1, 2], [3, 4], "complete observation partition"),
        ([0, 1, 1, 2, 3], [4, 5], "duplicate"),
        ([0, 1, 2, 3, 4], [5, 99], "out-of-range"),
        ([0, 1, 2, 3, True], [5, 6], "integers"),
    ],
)
def test_paris_fit_rejects_partition_leakage_and_malformed_indices(
    calibration: list[int],
    held_out: list[int],
    message: str,
) -> None:
    delta_k, growth, _calibration, _held_out = _paris_data()
    with pytest.raises(DegradationInputError, match=message):
        fit_paris_law(
            delta_k,
            growth,
            calibration_indices=calibration,
            held_out_indices=held_out,
            conditions=_paris_conditions(),
            observations_provenance=_provenance("fatigue"),
        )


def test_paris_fit_rejects_nonpositive_nonfinite_and_non_paris_trend() -> None:
    delta_k, growth, calibration, held_out = _paris_data()
    for bad_delta_k in (
        np.array([5.0, 7.0, 10.0, 12.0, 15.0, 18.0, 0.0]),
        np.array([5.0, 7.0, 10.0, 12.0, 15.0, 18.0, np.nan]),
    ):
        with pytest.raises(DegradationInputError):
            fit_paris_law(
                bad_delta_k,
                growth,
                calibration_indices=calibration,
                held_out_indices=held_out,
                conditions=_paris_conditions(),
                observations_provenance=_provenance("fatigue"),
            )
    with pytest.raises(DegradationInputError, match="strictly positive"):
        fit_paris_law(
            delta_k,
            np.where(growth > 0.0, 0.0, growth),
            calibration_indices=calibration,
            held_out_indices=held_out,
            conditions=_paris_conditions(),
            observations_provenance=_provenance("fatigue"),
        )

    with pytest.raises(DegradationInputError, match="not booleans"):
        fit_paris_law(
            np.array([True, True, True, True, True, True, True]),
            growth,
            calibration_indices=calibration,
            held_out_indices=held_out,
            conditions=_paris_conditions(),
            observations_provenance=_provenance("fatigue"),
        )
    with pytest.raises(DegradationInputError, match="not strings"):
        fit_paris_law(
            ["5", "7", "10", "12", "15", "18", "20"],
            growth,
            calibration_indices=calibration,
            held_out_indices=held_out,
            conditions=_paris_conditions(),
            observations_provenance=_provenance("fatigue"),
        )
    with pytest.raises(DegradationInputError, match="exponent"):
        fit_paris_law(
            delta_k,
            1.0e-8 / delta_k,
            calibration_indices=calibration,
            held_out_indices=held_out,
            conditions=_paris_conditions(),
            observations_provenance=_provenance("fatigue"),
        )


def test_paris_vectorized_prediction_handles_large_bounded_batch() -> None:
    delta_k, growth, calibration, held_out = _paris_data()
    fit = fit_paris_law(
        delta_k,
        growth,
        calibration_indices=calibration,
        held_out_indices=held_out,
        conditions=_paris_conditions(),
        observations_provenance=_provenance("fatigue"),
    )
    query = np.linspace(5.0, 20.0, 100_000)
    predicted = fit.predict_growth_rate_m_per_cycle(query, conditions=_paris_conditions())

    assert predicted.shape == query.shape
    assert np.all(np.diff(predicted) > 0.0)


def test_norton_arrhenius_rate_matches_analytical_equation() -> None:
    model = _creep_model()
    result = evaluate_norton_arrhenius_creep_rate(
        model,
        stress_pa=200.0e6,
        temperature_k=1000.0,
        material_state_id="state-A",
        environment_id="argon-1bar",
    )
    expected = 1.0e-4 * 2.0**4 * math.exp(-200_000.0 / (8.31446261815324 * 1000.0))

    assert result.effective_secondary_creep_rate_per_s == pytest.approx(expected)
    assert result.secondary_steady_state_only is True
    assert "does not predict primary/tertiary creep" in result.limitation


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"stress_pa": 49.0e6}, "outside calibrated"),
        ({"temperature_k": 1200.1}, "outside calibrated"),
        ({"material_state_id": "state-B"}, "material_state_id differs"),
        ({"environment_id": "air"}, "environment_id differs"),
    ],
)
def test_norton_arrhenius_refuses_domain_transfer(
    kwargs: dict[str, object],
    message: str,
) -> None:
    inputs: dict[str, object] = {
        "stress_pa": 200.0e6,
        "temperature_k": 1000.0,
        "material_state_id": "state-A",
        "environment_id": "argon-1bar",
    }
    inputs.update(kwargs)
    with pytest.raises(CalibrationDomainError, match=message):
        evaluate_norton_arrhenius_creep_rate(_creep_model(), **inputs)


def test_norton_model_rejects_wrong_units_and_underflow() -> None:
    with pytest.raises(DegradationInputError, match="stress.*Pa"):
        replace(_creep_model(), stress_domain_pa=ClosedInterval("stress", "MPa", 50.0, 300.0))
    underflow_model = replace(_creep_model(), activation_energy_j_per_mol=1.0e308)
    with pytest.raises(DegradationInputError, match="underflowed"):
        evaluate_norton_arrhenius_creep_rate(
            underflow_model,
            stress_pa=200.0e6,
            temperature_k=1000.0,
            material_state_id="state-A",
            environment_id="argon-1bar",
        )


def test_linear_and_parabolic_oxidation_laws_match_analytical_values() -> None:
    linear = evaluate_oxidation_mass_gain(
        _oxidation_model("linear"),
        exposure_time_s=10.0,
        temperature_k=1073.0,
        material_state_id="coupon-state-A",
        environment_id="dry-air-flow-cell-1",
    )
    parabolic = evaluate_oxidation_mass_gain(
        _oxidation_model("parabolic"),
        exposure_time_s=4.0,
        temperature_k=1073.0,
        material_state_id="coupon-state-A",
        environment_id="dry-air-flow-cell-1",
    )

    assert linear.areal_mass_gain_kg_per_m2 == pytest.approx(0.02)
    assert linear.model.rate_constant_unit == "kg*m^-2*s^-1"
    assert parabolic.areal_mass_gain_kg_per_m2 == pytest.approx(0.05)
    assert parabolic.model.rate_constant_unit == "kg^2*m^-4*s^-1"
    assert "not oxide thickness" in parabolic.limitation


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"exposure_time_s": 101.0}, "outside calibrated"),
        ({"temperature_k": 1074.0}, "outside calibrated"),
        ({"material_state_id": "coupon-state-B"}, "material_state_id differs"),
        ({"environment_id": "wet-air"}, "environment_id differs"),
    ],
)
def test_oxidation_refuses_domain_transfer(kwargs: dict[str, object], message: str) -> None:
    inputs: dict[str, object] = {
        "exposure_time_s": 10.0,
        "temperature_k": 1073.0,
        "material_state_id": "coupon-state-A",
        "environment_id": "dry-air-flow-cell-1",
    }
    inputs.update(kwargs)
    with pytest.raises(CalibrationDomainError, match=message):
        evaluate_oxidation_mass_gain(_oxidation_model(), **inputs)


def test_oxidation_model_rejects_unknown_law_wrong_domain_units_and_negative_rate() -> None:
    with pytest.raises(DegradationInputError, match="linear.*parabolic"):
        replace(_oxidation_model(), law="logarithmic")
    with pytest.raises(DegradationInputError, match="time.*'s'"):
        replace(_oxidation_model(), time_domain_s=ClosedInterval("time", "h", 0.0, 1.0))
    with pytest.raises(DegradationInputError, match="cannot be negative"):
        replace(_oxidation_model(), rate_constant=-1.0)
    with pytest.raises(DegradationInputError, match="rate_constant_unit must be exactly"):
        replace(_oxidation_model("linear"), rate_constant_unit="kg^2*m^-4*s^-1")
    with pytest.raises(DegradationInputError, match="rate_constant_unit must be exactly"):
        replace(_oxidation_model("parabolic"), rate_constant_unit="kg*m^-2*s^-1")
    with pytest.raises(DegradationInputError, match="rate_constant_unit must be exactly"):
        replace(_oxidation_model("linear"), rate_constant_unit="kg/m^2/s")
    with pytest.raises(DegradationInputError, match="singleton isothermal temperature"):
        replace(
            _oxidation_model(),
            temperature_domain_k=ClosedInterval("temperature", "K", 1073.0, 1173.0),
        )


def test_corrosion_conversion_matches_faraday_law_with_explicit_efficiency() -> None:
    inputs = _corrosion_inputs()
    result = convert_corrosion_current_to_uniform_penetration(inputs)
    expected_flux = 1.0 * (0.055845 / 2.0) * 0.8 / FARADAY_CONSTANT_C_PER_MOL
    expected_rate = expected_flux / 7874.0

    assert result.uniform_mass_loss_flux_kg_per_m2_s == pytest.approx(expected_flux)
    assert result.average_uniform_penetration_rate_m_per_s == pytest.approx(expected_rate)
    assert result.average_uniform_penetration_m == pytest.approx(expected_rate * inputs.duration_s)
    assert result.standard_compliance_claimed is False
    assert "spatially uniform dissolution" in result.limitation


def test_corrosion_zero_current_or_duration_produces_zero_not_nan() -> None:
    zero_current = convert_corrosion_current_to_uniform_penetration(
        replace(_corrosion_inputs(), corrosion_current_density_a_per_m2=0.0)
    )
    zero_duration = convert_corrosion_current_to_uniform_penetration(
        replace(_corrosion_inputs(), duration_s=0.0)
    )

    assert zero_current.average_uniform_penetration_rate_m_per_s == 0.0
    assert zero_current.average_uniform_penetration_m == 0.0
    assert zero_duration.average_uniform_penetration_rate_m_per_s > 0.0
    assert zero_duration.average_uniform_penetration_m == 0.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("corrosion_current_density_a_per_m2", -1.0, "cannot be negative"),
        ("equivalent_mass_kg_per_mol_electron", 0.0, "strictly positive"),
        ("density_kg_per_m3", math.nan, "finite scalar"),
        ("current_efficiency", 0.0, "strictly positive"),
        ("current_efficiency", 1.01, "cannot exceed 1"),
        ("duration_s", -1.0, "cannot be negative"),
    ],
)
def test_corrosion_rejects_ambiguous_or_nonphysical_values(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(DegradationInputError, match=message):
        replace(_corrosion_inputs(), **{field: value})


def test_content_provenance_and_closed_intervals_fail_closed() -> None:
    with pytest.raises(DegradationInputError, match="sha256"):
        EvidenceProvenance(
            artifact_id="source",
            sha256="abc",
            locator="resource://source",
            citation="citation",
        )
    with pytest.raises(DegradationInputError, match="lower bound exceeds"):
        ClosedInterval("stress", "Pa", 2.0, 1.0)
    interval = ClosedInterval("temperature", "K", 900.0, 1000.0)
    assert interval.require_contains(900.0, field_name="temperature") == 900.0
    assert interval.require_contains(1000.0, field_name="temperature") == 1000.0
    with pytest.raises(CalibrationDomainError, match="outside calibrated"):
        interval.require_contains(899.9, field_name="temperature")


def test_scientific_quantity_definitions_and_area_bases_cannot_be_blank() -> None:
    with pytest.raises(DegradationInputError, match="crack_length_definition is required"):
        replace(_geometry(), crack_length_definition="")
    with pytest.raises(DegradationInputError, match="delta_k_definition_id is required"):
        replace(_paris_conditions(), delta_k_definition_id="")
    with pytest.raises(DegradationInputError, match="stress_measure_id is required"):
        replace(_creep_model(), stress_measure_id="")
    with pytest.raises(DegradationInputError, match="area_basis_id is required"):
        replace(_oxidation_model(), area_basis_id="")
    with pytest.raises(DegradationInputError, match="current_density_area_basis_id is required"):
        replace(_corrosion_inputs(), current_density_area_basis_id="")


def test_successful_result_scalars_are_finite_and_canonical_json_safe() -> None:
    lefm = evaluate_mode_i_lefm(**_lefm_kwargs())
    creep = evaluate_norton_arrhenius_creep_rate(
        _creep_model(),
        stress_pa=200.0e6,
        temperature_k=1000.0,
        material_state_id="state-A",
        environment_id="argon-1bar",
    )
    oxidation = evaluate_oxidation_mass_gain(
        _oxidation_model(),
        exposure_time_s=10.0,
        temperature_k=1073.0,
        material_state_id="coupon-state-A",
        environment_id="dry-air-flow-cell-1",
    )
    corrosion = convert_corrosion_current_to_uniform_penetration(_corrosion_inputs())

    assert all(
        math.isfinite(value)
        for value in (
            lefm.stress_intensity_pa_sqrt_m,
            lefm.plastic_zone_radius_m,
            lefm.minimum_dimension_to_plastic_zone_ratio,
            creep.effective_secondary_creep_rate_per_s,
            oxidation.areal_mass_gain_kg_per_m2,
            corrosion.uniform_mass_loss_flux_kg_per_m2_s,
            corrosion.average_uniform_penetration_rate_m_per_s,
            corrosion.average_uniform_penetration_m,
        )
    )
