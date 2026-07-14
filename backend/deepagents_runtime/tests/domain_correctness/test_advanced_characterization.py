from __future__ import annotations

import math

import numpy as np
import pytest
from ultra_deepagents.materials.advanced_characterization import (
    INDEPENDENT_1SIGMA,
    CharacterizationInputError,
    DataProvenance,
    ReflectionRegistrationError,
    calculate_diffraction_profile_metrics,
    fit_rigid_registration,
)

_OBSERVED_PROVENANCE = DataProvenance(
    artifact_id="measured-pattern-001",
    sha256="a" * 64,
    locator="resource://measured-pattern-001#/entry/data",
    processing_history_id="raw-detector-correction-v2",
)
_CALCULATED_PROVENANCE = DataProvenance(
    artifact_id="calculated-pattern-001",
    sha256="b" * 64,
    locator="artifact://calculated-pattern-001/profile.csv",
    processing_history_id="external-refinement-run-17",
)
_SOURCE_PROVENANCE = DataProvenance(
    artifact_id="ebsd-points-001",
    sha256="c" * 64,
    locator="resource://ebsd-points-001#/coordinates",
)
_TARGET_PROVENANCE = DataProvenance(
    artifact_id="apt-points-001",
    sha256="d" * 64,
    locator="resource://apt-points-001#/landmarks",
)


def _profile_kwargs() -> dict[str, object]:
    return {
        "coordinate_unit": "degree_2theta",
        "observed_intensity_unit": "counts",
        "calculated_intensity_unit": "counts",
        "observed_provenance": _OBSERVED_PROVENANCE,
        "calculated_provenance": _CALCULATED_PROVENANCE,
    }


def _registration_kwargs() -> dict[str, object]:
    return {
        "source_frame_id": "ebsd-detector-frame",
        "target_frame_id": "apt-reconstruction-frame",
        "source_coordinate_unit": "um",
        "target_coordinate_unit": "um",
        "source_provenance": _SOURCE_PROVENANCE,
        "target_provenance": _TARGET_PROVENANCE,
        "calibration_indices": [0, 1, 2],
        "held_out_indices": [3, 4, 5],
    }


def test_diffraction_metrics_match_analytical_r_factors_and_chi_square() -> None:
    result = calculate_diffraction_profile_metrics(
        [20.0, 20.1, 20.2, 20.3],
        [10.0, 20.0, 30.0, 40.0],
        [9.0, 22.0, 999.0, 39.0],
        included_mask=[True, True, False, True],
        uncertainties=[1.0, 2.0, 0.0, 2.0],
        uncertainty_semantics=INDEPENDENT_1SIGMA,
        refined_parameter_count=1,
        independent_constraint_count=0,
        **_profile_kwargs(),
    )

    # Included residuals are [1, -2, 1], weights are [1, 1/4, 1/4].
    assert result.rp == pytest.approx(4.0 / 70.0)
    assert result.rwp == pytest.approx(math.sqrt(2.25 / 600.0))
    assert result.degrees_of_freedom == 2
    assert result.rexp == pytest.approx(math.sqrt(2.0 / 600.0))
    assert result.chi_square == pytest.approx(2.25)
    assert result.reduced_chi_square == pytest.approx(1.125)
    assert result.goodness_of_fit == pytest.approx(math.sqrt(1.125))
    assert result.included_point_count == 3
    assert result.total_point_count == 4
    assert result.weighting_scheme == "inverse_variance_from_independent_absolute_1sigma"
    assert result.observed_provenance.sha256 == "a" * 64
    assert result.method_reference_url.startswith("https://journals.iucr.org/")
    assert result.validation_only is True
    assert "do not perform or validate Rietveld refinement" in result.limitation


def test_diffraction_unit_weighted_rwp_does_not_claim_statistical_metrics() -> None:
    result = calculate_diffraction_profile_metrics(
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 4.0],
        [1.0, 1.0, 5.0],
        refined_parameter_count=1,
        **_profile_kwargs(),
    )

    assert result.rp == pytest.approx(2.0 / 7.0)
    assert result.rwp == pytest.approx(math.sqrt(2.0 / 21.0))
    assert result.degrees_of_freedom == 2
    assert result.rexp is None
    assert result.chi_square is None
    assert result.reduced_chi_square is None
    assert result.goodness_of_fit is None
    assert result.weighting_scheme == "unit_weights_non_statistical"


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("coordinate", [1.0, np.nan, 3.0]),
        ("observed_intensity", [1.0, np.inf, 3.0]),
        ("calculated_intensity", [1.0, 2.0, -np.inf]),
        ("uncertainties", [1.0, np.nan, 1.0]),
    ],
)
def test_diffraction_rejects_nonfinite_values_even_when_a_point_is_masked(
    field: str,
    values: list[float],
) -> None:
    arguments: dict[str, object] = {
        "coordinate": [1.0, 2.0, 3.0],
        "observed_intensity": [1.0, 2.0, 3.0],
        "calculated_intensity": [1.0, 2.0, 3.0],
        "included_mask": [True, False, True],
        "uncertainties": [1.0, 1.0, 1.0],
        "uncertainty_semantics": INDEPENDENT_1SIGMA,
        **_profile_kwargs(),
    }
    arguments[field] = values

    with pytest.raises(CharacterizationInputError, match="non-finite"):
        calculate_diffraction_profile_metrics(**arguments)


def test_diffraction_rejects_zero_denominators_and_selected_zero_uncertainty() -> None:
    with pytest.raises(CharacterizationInputError, match="Rp is undefined"):
        calculate_diffraction_profile_metrics(
            [1.0, 2.0],
            [0.0, 0.0],
            [0.0, 1.0],
            **_profile_kwargs(),
        )

    with pytest.raises(CharacterizationInputError, match="strictly positive"):
        calculate_diffraction_profile_metrics(
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            uncertainties=[1.0, 0.0],
            uncertainty_semantics=INDEPENDENT_1SIGMA,
            **_profile_kwargs(),
        )


def test_diffraction_rejects_ambiguous_units_masks_uncertainties_and_dof() -> None:
    with pytest.raises(CharacterizationInputError, match="units must match exactly"):
        calculate_diffraction_profile_metrics(
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            **{
                **_profile_kwargs(),
                "calculated_intensity_unit": "arbitrary_unit",
            },
        )

    with pytest.raises(CharacterizationInputError, match="boolean array"):
        calculate_diffraction_profile_metrics(
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            included_mask=[1, 0],
            **_profile_kwargs(),
        )

    with pytest.raises(CharacterizationInputError, match="uncertainty_semantics"):
        calculate_diffraction_profile_metrics(
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            uncertainties=[1.0, 1.0],
            uncertainty_semantics="relative_weights",
            **_profile_kwargs(),
        )

    with pytest.raises(CharacterizationInputError, match=r"N - P \+ C"):
        calculate_diffraction_profile_metrics(
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            uncertainties=[1.0, 1.0],
            uncertainty_semantics=INDEPENDENT_1SIGMA,
            refined_parameter_count=2,
            **_profile_kwargs(),
        )


def test_provenance_rejects_missing_or_non_content_addressed_sources() -> None:
    with pytest.raises(CharacterizationInputError, match="sha256"):
        DataProvenance(
            artifact_id="not-content-addressed",
            sha256="abc",
            locator="resource://not-content-addressed",
        )
    with pytest.raises(CharacterizationInputError, match="locator is required"):
        DataProvenance(artifact_id="missing-locator", sha256="e" * 64, locator="")


def test_2d_registration_recovers_known_transform_and_scores_held_out_noise() -> None:
    source = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [3.0, -1.0],
            [-2.0, 0.5],
        ]
    )
    angle = math.radians(37.0)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    translation = np.array([2.5, -1.25])
    clean_target = source @ rotation.T + translation
    held_out_noise = np.array([[0.1, -0.2], [-0.05, 0.0], [0.0, 0.3]])
    target = clean_target.copy()
    target[3:] += held_out_noise

    result = fit_rigid_registration(source, target, **_registration_kwargs())

    np.testing.assert_allclose(result.rotation_source_to_target, rotation, atol=1.0e-12)
    np.testing.assert_allclose(result.translation_source_to_target, translation, atol=1.0e-12)
    np.testing.assert_allclose(result.calibration_residual_norms, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        result.held_out_residual_norms,
        np.linalg.norm(held_out_noise, axis=1),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        result.transform(
            source,
            source_frame_id="ebsd-detector-frame",
            coordinate_unit="um",
        ),
        clean_target,
        atol=1.0e-12,
    )
    assert result.held_out_statistics.rmse == pytest.approx(
        math.sqrt(float(np.mean(np.sum(np.square(held_out_noise), axis=1))))
    )
    assert result.rotation_determinant == pytest.approx(1.0)
    assert result.method_reference_doi.endswith("S0567739476001873")
    assert result.validation_only is True


def test_held_out_points_cannot_leak_into_registration_fit() -> None:
    source = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [3.0, -1.0],
            [-2.0, 0.5],
        ]
    )
    angle = math.radians(-23.0)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    translation = np.array([-4.0, 7.0])
    target = source @ rotation.T + translation
    corrupted_held_out_target = target.copy()
    corrupted_held_out_target[3:] += np.array([[1000.0, -400.0], [-800.0, 600.0], [99.0, 51.0]])

    clean = fit_rigid_registration(source, target, **_registration_kwargs())
    corrupted = fit_rigid_registration(
        source,
        corrupted_held_out_target,
        **_registration_kwargs(),
    )

    np.testing.assert_array_equal(
        corrupted.rotation_source_to_target,
        clean.rotation_source_to_target,
    )
    np.testing.assert_array_equal(
        corrupted.translation_source_to_target,
        clean.translation_source_to_target,
    )
    assert corrupted.held_out_statistics.maximum > 100.0


def test_registration_rejects_an_unreported_correspondence_outside_both_partitions() -> None:
    source = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [3.0, -1.0], [-2.0, 0.5]])
    target = source + np.asarray([2.5, -1.25])
    source = np.vstack([source, [20.0, 20.0]])
    target = np.vstack([target, [-999.0, 400.0]])

    with pytest.raises(CharacterizationInputError, match="cover every correspondence"):
        fit_rigid_registration(source, target, **_registration_kwargs())


def test_3d_registration_recovers_an_analytical_proper_rotation() -> None:
    source = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 1.0],
            [-1.0, 2.0, 0.5],
        ]
    )
    angle = math.radians(48.0)
    rotation = np.array(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ]
    )
    translation = np.array([0.25, -3.0, 9.0])
    target = source @ rotation.T + translation
    kwargs = {
        **_registration_kwargs(),
        "calibration_indices": [0, 1, 2, 3],
        "held_out_indices": [4, 5],
    }

    result = fit_rigid_registration(source, target, **kwargs)

    np.testing.assert_allclose(result.rotation_source_to_target, rotation, atol=1.0e-12)
    np.testing.assert_allclose(result.translation_source_to_target, translation, atol=1.0e-12)
    assert result.calibration_statistics.maximum < 1.0e-12
    assert result.held_out_statistics.maximum < 1.0e-12


def test_registration_rejects_reflections_and_rank_deficient_calibration() -> None:
    source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]])
    reflected_target = source @ np.diag([-1.0, 1.0]).T + np.array([2.0, 3.0])
    kwargs = {
        **_registration_kwargs(),
        "calibration_indices": [0, 1, 2, 3],
        "held_out_indices": [4],
    }
    with pytest.raises(ReflectionRegistrationError, match="require a reflection"):
        fit_rigid_registration(source, reflected_target, **kwargs)

    collinear = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    angle = math.radians(10.0)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    target = collinear @ rotation.T
    with pytest.raises(CharacterizationInputError, match="rank deficient or nearly collinear"):
        fit_rigid_registration(collinear, target, **kwargs)


def test_registration_rejects_unit_frame_split_and_index_ambiguity() -> None:
    source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0], [-1.0, 1.0]])
    target = source + np.array([3.0, -2.0])

    with pytest.raises(CharacterizationInputError, match="units must match exactly"):
        fit_rigid_registration(
            source,
            target,
            **{**_registration_kwargs(), "target_coordinate_unit": "nm"},
        )
    with pytest.raises(CharacterizationInputError, match="distinct coordinate frames"):
        fit_rigid_registration(
            source,
            target,
            **{**_registration_kwargs(), "target_frame_id": "ebsd-detector-frame"},
        )
    with pytest.raises(CharacterizationInputError, match="disjoint"):
        fit_rigid_registration(
            source,
            target,
            **{**_registration_kwargs(), "held_out_indices": [2, 3, 4]},
        )
    with pytest.raises(CharacterizationInputError, match="integer array"):
        fit_rigid_registration(
            source,
            target,
            **{**_registration_kwargs(), "calibration_indices": [0.0, 1.0, 2.0]},
        )
    with pytest.raises(CharacterizationInputError, match="duplicate"):
        fit_rigid_registration(
            source,
            target,
            **{**_registration_kwargs(), "held_out_indices": [3, 3, 4]},
        )


def test_registration_rejects_nonfinite_data_and_transform_frame_mismatch() -> None:
    source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0], [-1.0, 1.0]])
    target = source + np.array([3.0, -2.0])
    nonfinite = source.copy()
    nonfinite[5, 0] = np.nan
    with pytest.raises(CharacterizationInputError, match="non-finite"):
        fit_rigid_registration(nonfinite, target, **_registration_kwargs())

    result = fit_rigid_registration(source, target, **_registration_kwargs())
    with pytest.raises(CharacterizationInputError, match="source frame mismatch"):
        result.transform(
            source,
            source_frame_id="tem-stage-frame",
            coordinate_unit="um",
        )
    with pytest.raises(CharacterizationInputError, match="coordinate unit mismatch"):
        result.transform(
            source,
            source_frame_id="ebsd-detector-frame",
            coordinate_unit="nm",
        )
    with pytest.raises(ValueError, match="read-only"):
        result.rotation_source_to_target[0, 0] = 0.0
