"""Deterministic validation primitives for measured materials characterization.

This module intentionally implements *validation*, not diffraction refinement,
indexing, segmentation, reconstruction, or chemical quantification.  It supplies
two pieces that are shared by measured XRD/Rietveld, EBSD, 4D-STEM, TEM, and APT
workflows:

* convention-explicit powder-profile residual metrics; and
* leakage-resistant rigid registration with an independently held-out point set.

The implementations are NumPy-only so their numerical behavior can be tested in
the lightweight runtime.  A successful result says that two arrays agree under
the stated metric or rigid-transform model.  It does not establish phase identity,
instrument calibration, refinement uniqueness, or physical model validity.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np

DIFFRACTION_METRICS_SCHEMA_VERSION = "ultra.materials.diffraction-profile-validation.v1"
RIGID_REGISTRATION_SCHEMA_VERSION = "ultra.materials.rigid-registration-validation.v1"
INDEPENDENT_1SIGMA = "independent_absolute_1sigma"
MAX_PROFILE_POINTS = 20_000_000
MAX_REGISTRATION_POINTS = 10_000_000
MIN_CALIBRATION_SINGULAR_VALUE_RATIO = 1.0e-12
PROFILE_METRICS_REFERENCE_URL = "https://journals.iucr.org/j/issues/1999/01/00/gl0561/"
KABSCH_REFERENCE_DOI = "https://doi.org/10.1107/S0567739476001873"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CharacterizationInputError(ValueError):
    """Input data or metadata is malformed, ambiguous, or mathematically invalid."""


class ReflectionRegistrationError(CharacterizationInputError):
    """The least-squares correspondence requires an improper rotation."""


def _clean_text(value: str, *, field_name: str, max_length: int = 256) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise CharacterizationInputError(f"{field_name} is required")
    if len(cleaned) > max_length:
        raise CharacterizationInputError(f"{field_name} exceeds {max_length} characters")
    if any(ord(character) < 32 for character in cleaned):
        raise CharacterizationInputError(f"{field_name} contains control characters")
    return cleaned


@dataclass(frozen=True)
class DataProvenance:
    """Content-addressed origin of a measured or calculated array."""

    artifact_id: str
    sha256: str
    locator: str
    processing_history_id: str | None = None

    def __post_init__(self) -> None:
        artifact_id = _clean_text(self.artifact_id, field_name="artifact_id")
        digest = str(self.sha256 or "").strip().lower()
        locator = _clean_text(self.locator, field_name="locator", max_length=2048)
        history = str(self.processing_history_id or "").strip() or None
        if not _SHA256.fullmatch(digest):
            raise CharacterizationInputError("sha256 must be exactly 64 hexadecimal digits")
        if history is not None:
            history = _clean_text(
                history,
                field_name="processing_history_id",
                max_length=256,
            )
        object.__setattr__(self, "artifact_id", artifact_id)
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "locator", locator)
        object.__setattr__(self, "processing_history_id", history)


@dataclass(frozen=True)
class DiffractionProfileMetrics:
    """Measured-versus-calculated profile statistics under explicit conventions.

    ``rp`` uses the conventional ``sum(abs(y_obs-y_calc)) / sum(y_obs)``.  ``rwp`` uses
    inverse-variance weights when uncertainties are supplied and unit weights
    otherwise.  ``rexp`` and the chi-square statistics are emitted only for
    independent absolute one-sigma uncertainties and a positive, explicit
    degrees of freedom ``N - P + C``.
    """

    schema_version: str
    coordinate_unit: str
    intensity_unit: str
    total_point_count: int
    included_point_count: int
    weighting_scheme: str
    rp: float
    rwp: float
    rexp: float | None
    goodness_of_fit: float | None
    reduced_chi_square: float | None
    chi_square: float | None
    degrees_of_freedom: int | None
    refined_parameter_count: int | None
    independent_constraint_count: int
    observed_provenance: DataProvenance
    calculated_provenance: DataProvenance
    method_reference_url: str = PROFILE_METRICS_REFERENCE_URL
    validation_only: bool = True
    limitation: str = (
        "Profile residuals validate numerical agreement only; they do not perform or "
        "validate Rietveld refinement, phase identification, calibration, uncertainty "
        "independence, or model uniqueness."
    )


@dataclass(frozen=True)
class ResidualStatistics:
    """Euclidean point-residual summary in the registration coordinate unit."""

    count: int
    rmse: float
    mean: float
    median: float
    maximum: float


@dataclass(frozen=True)
class RigidRegistrationResult:
    """Proper rigid transform from a source frame into a distinct target frame."""

    schema_version: str
    source_frame_id: str
    target_frame_id: str
    coordinate_unit: str
    source_provenance: DataProvenance
    target_provenance: DataProvenance
    calibration_indices: tuple[int, ...]
    held_out_indices: tuple[int, ...]
    rotation_source_to_target: np.ndarray
    translation_source_to_target: np.ndarray
    calibration_residual_norms: np.ndarray
    held_out_residual_norms: np.ndarray
    calibration_statistics: ResidualStatistics
    held_out_statistics: ResidualStatistics
    calibration_cross_covariance_singular_values: np.ndarray
    rotation_determinant: float
    method_reference_doi: str = KABSCH_REFERENCE_DOI
    proper_rotation_enforced: bool = True
    validation_only: bool = True
    limitation: str = (
        "The transform validates only the supplied point correspondences. It does not "
        "establish feature identity, non-rigid registration quality, segmentation "
        "accuracy, crystallographic indexing, or chemical reconstruction validity."
    )

    def __post_init__(self) -> None:
        for value in (
            self.rotation_source_to_target,
            self.translation_source_to_target,
            self.calibration_residual_norms,
            self.held_out_residual_norms,
            self.calibration_cross_covariance_singular_values,
        ):
            value.setflags(write=False)

    def transform(
        self,
        points: Sequence[Sequence[float]] | np.ndarray,
        *,
        source_frame_id: str,
        coordinate_unit: str,
    ) -> np.ndarray:
        """Transform points, rejecting an input bound to a different frame or unit."""

        frame = _clean_text(source_frame_id, field_name="source_frame_id")
        unit = _clean_text(coordinate_unit, field_name="coordinate_unit")
        if frame != self.source_frame_id:
            raise CharacterizationInputError(
                f"source frame mismatch: expected {self.source_frame_id!r}, got {frame!r}"
            )
        if unit != self.coordinate_unit:
            raise CharacterizationInputError(
                f"coordinate unit mismatch: expected {self.coordinate_unit!r}, got {unit!r}"
            )
        try:
            array = np.asarray(points, dtype=float)
        except (TypeError, ValueError) as exc:
            raise CharacterizationInputError("points must contain numeric values") from exc
        dimension = self.rotation_source_to_target.shape[0]
        single_point = array.ndim == 1
        if single_point:
            if array.shape != (dimension,):
                raise CharacterizationInputError(
                    f"points must have shape ({dimension},) or (N, {dimension})"
                )
            array = array[np.newaxis, :]
        elif array.ndim != 2 or array.shape[1] != dimension:
            raise CharacterizationInputError(
                f"points must have shape ({dimension},) or (N, {dimension})"
            )
        if not np.all(np.isfinite(array)):
            raise CharacterizationInputError("points contain non-finite coordinates")
        with np.errstate(over="ignore", invalid="ignore"):
            transformed = (
                array @ self.rotation_source_to_target.T + self.translation_source_to_target
            )
        if not np.all(np.isfinite(transformed)):
            raise CharacterizationInputError("point transformation produced non-finite values")
        output = np.array(transformed[0] if single_point else transformed, copy=True)
        output.setflags(write=False)
        return cast(np.ndarray, output)


def _as_finite_profile_array(
    value: Sequence[float] | np.ndarray,
    *,
    field_name: str,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CharacterizationInputError(f"{field_name} must contain numeric values") from exc
    if array.ndim != 1:
        raise CharacterizationInputError(f"{field_name} must be a one-dimensional array")
    if array.size == 0:
        raise CharacterizationInputError(f"{field_name} cannot be empty")
    if array.size > MAX_PROFILE_POINTS:
        raise CharacterizationInputError(
            f"{field_name} exceeds the {MAX_PROFILE_POINTS} point safety cap"
        )
    if not np.all(np.isfinite(array)):
        raise CharacterizationInputError(f"{field_name} contains non-finite values")
    return cast(np.ndarray, array)


def _validated_nonnegative_integer(
    value: int | None,
    *,
    field_name: str,
    allow_none: bool,
) -> int | None:
    if value is None:
        if allow_none:
            return None
        raise CharacterizationInputError(f"{field_name} is required")
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise CharacterizationInputError(f"{field_name} must be an integer")
    integer = int(value)
    if integer < 0:
        raise CharacterizationInputError(f"{field_name} cannot be negative")
    return integer


def calculate_diffraction_profile_metrics(
    coordinate: Sequence[float] | np.ndarray,
    observed_intensity: Sequence[float] | np.ndarray,
    calculated_intensity: Sequence[float] | np.ndarray,
    *,
    coordinate_unit: str,
    observed_intensity_unit: str,
    calculated_intensity_unit: str,
    observed_provenance: DataProvenance,
    calculated_provenance: DataProvenance,
    included_mask: Sequence[bool] | np.ndarray | None = None,
    uncertainties: Sequence[float] | np.ndarray | None = None,
    uncertainty_semantics: str | None = None,
    refined_parameter_count: int | None = None,
    independent_constraint_count: int = 0,
) -> DiffractionProfileMetrics:
    """Calculate profile residual metrics without fitting or changing the model.

    ``included_mask`` is inclusion-valued: ``True`` means the point contributes.
    When ``uncertainties`` are supplied they must be absolute one-standard-
    deviation values in ``observed_intensity_unit`` and statistically independent;
    this is the only uncertainty convention for which Rexp and reduced chi-square
    are reported.
    """

    coordinate_values = _as_finite_profile_array(coordinate, field_name="coordinate")
    observed = _as_finite_profile_array(observed_intensity, field_name="observed_intensity")
    calculated = _as_finite_profile_array(
        calculated_intensity,
        field_name="calculated_intensity",
    )
    if coordinate_values.size != observed.size or observed.size != calculated.size:
        raise CharacterizationInputError(
            "coordinate, observed_intensity, and calculated_intensity lengths must match"
        )
    if coordinate_values.size > 1 and not np.all(coordinate_values[1:] > coordinate_values[:-1]):
        raise CharacterizationInputError("coordinate values must be strictly increasing")

    coordinate_unit = _clean_text(coordinate_unit, field_name="coordinate_unit")
    observed_unit = _clean_text(
        observed_intensity_unit,
        field_name="observed_intensity_unit",
    )
    calculated_unit = _clean_text(
        calculated_intensity_unit,
        field_name="calculated_intensity_unit",
    )
    if observed_unit != calculated_unit:
        raise CharacterizationInputError(
            "observed and calculated intensity units must match exactly before comparison"
        )
    if not isinstance(observed_provenance, DataProvenance):
        raise CharacterizationInputError("observed_provenance must be DataProvenance")
    if not isinstance(calculated_provenance, DataProvenance):
        raise CharacterizationInputError("calculated_provenance must be DataProvenance")

    if included_mask is None:
        mask = np.ones(observed.shape, dtype=bool)
    else:
        try:
            mask = np.asarray(included_mask)
        except (TypeError, ValueError) as exc:
            raise CharacterizationInputError("included_mask must contain boolean values") from exc
        if mask.dtype.kind != "b" or mask.ndim != 1 or mask.shape != observed.shape:
            raise CharacterizationInputError(
                "included_mask must be a one-dimensional boolean array matching the profile"
            )
    included_count = int(np.count_nonzero(mask))
    if included_count == 0:
        raise CharacterizationInputError("included_mask must select at least one point")

    parameter_count = _validated_nonnegative_integer(
        refined_parameter_count,
        field_name="refined_parameter_count",
        allow_none=True,
    )
    constraint_count_value = _validated_nonnegative_integer(
        independent_constraint_count,
        field_name="independent_constraint_count",
        allow_none=False,
    )
    assert constraint_count_value is not None
    if parameter_count is None and constraint_count_value:
        raise CharacterizationInputError(
            "independent_constraint_count requires refined_parameter_count"
        )
    if parameter_count is not None and constraint_count_value > parameter_count:
        raise CharacterizationInputError(
            "independent_constraint_count cannot exceed refined_parameter_count"
        )

    observed_selected = observed[mask]
    calculated_selected = calculated[mask]
    with np.errstate(over="ignore", invalid="ignore"):
        residual = observed_selected - calculated_selected
        rp_denominator = float(np.sum(observed_selected, dtype=np.float64))
    if not np.all(np.isfinite(residual)):
        raise CharacterizationInputError("profile subtraction produced non-finite residuals")
    if not math.isfinite(rp_denominator) or rp_denominator <= 0.0:
        raise CharacterizationInputError(
            "Rp is undefined because the included observed-intensity sum is not positive"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        rp = float(np.sum(np.abs(residual), dtype=np.float64) / rp_denominator)

    statistical_weighting = uncertainties is not None
    if uncertainties is not None:
        uncertainty_values = _as_finite_profile_array(
            uncertainties,
            field_name="uncertainties",
        )
        if uncertainty_values.shape != observed.shape:
            raise CharacterizationInputError("uncertainties must match the profile length")
        semantics = str(uncertainty_semantics or "").strip()
        if semantics != INDEPENDENT_1SIGMA:
            raise CharacterizationInputError(
                f"uncertainty_semantics must be {INDEPENDENT_1SIGMA!r}"
            )
        uncertainty_selected = uncertainty_values[mask]
        if np.any(uncertainty_selected <= 0.0):
            raise CharacterizationInputError(
                "included uncertainties must be strictly positive absolute one-sigma values"
            )
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            weights = np.reciprocal(np.square(uncertainty_selected))
        if not np.all(np.isfinite(weights)):
            raise CharacterizationInputError("uncertainties produce non-finite weights")
        weighting_scheme = "inverse_variance_from_independent_absolute_1sigma"
    else:
        if uncertainty_semantics is not None:
            raise CharacterizationInputError(
                "uncertainty_semantics cannot be supplied without uncertainties"
            )
        weights = np.ones(observed_selected.shape, dtype=float)
        weighting_scheme = "unit_weights_non_statistical"

    with np.errstate(over="ignore", invalid="ignore"):
        weighted_residual_sum = float(np.sum(weights * np.square(residual), dtype=np.float64))
        rwp_denominator = float(np.sum(weights * np.square(observed_selected), dtype=np.float64))
    if (
        not math.isfinite(weighted_residual_sum)
        or not math.isfinite(rwp_denominator)
        or rwp_denominator <= 0.0
    ):
        raise CharacterizationInputError(
            "Rwp is undefined because weighted sums are non-finite or its denominator is zero"
        )
    rwp = math.sqrt(weighted_residual_sum / rwp_denominator)

    degrees_of_freedom: int | None = None
    rexp: float | None = None
    reduced_chi_square: float | None = None
    goodness_of_fit: float | None = None
    chi_square: float | None = None
    if parameter_count is not None:
        degrees_of_freedom = included_count - parameter_count + constraint_count_value
        if degrees_of_freedom <= 0:
            raise CharacterizationInputError(
                "degrees of freedom N - P + C must be strictly positive"
            )
    if statistical_weighting and degrees_of_freedom is not None:
        chi_square = weighted_residual_sum
        rexp = math.sqrt(degrees_of_freedom / rwp_denominator)
        reduced_chi_square = chi_square / degrees_of_freedom
        goodness_of_fit = rwp / rexp

    values_to_check = (rp, rwp, rexp, reduced_chi_square, goodness_of_fit, chi_square)
    if any(value is not None and not math.isfinite(value) for value in values_to_check):
        raise CharacterizationInputError("profile metrics contain a non-finite result")

    return DiffractionProfileMetrics(
        schema_version=DIFFRACTION_METRICS_SCHEMA_VERSION,
        coordinate_unit=coordinate_unit,
        intensity_unit=observed_unit,
        total_point_count=int(observed.size),
        included_point_count=included_count,
        weighting_scheme=weighting_scheme,
        rp=rp,
        rwp=rwp,
        rexp=rexp,
        goodness_of_fit=goodness_of_fit,
        reduced_chi_square=reduced_chi_square,
        chi_square=chi_square,
        degrees_of_freedom=degrees_of_freedom,
        refined_parameter_count=parameter_count,
        independent_constraint_count=constraint_count_value,
        observed_provenance=observed_provenance,
        calculated_provenance=calculated_provenance,
    )


def _as_finite_points(
    value: Sequence[Sequence[float]] | np.ndarray,
    *,
    field_name: str,
) -> np.ndarray:
    try:
        points = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CharacterizationInputError(f"{field_name} must contain numeric values") from exc
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise CharacterizationInputError(f"{field_name} must have shape (N, 2) or (N, 3)")
    if points.shape[0] == 0:
        raise CharacterizationInputError(f"{field_name} cannot be empty")
    if points.shape[0] > MAX_REGISTRATION_POINTS:
        raise CharacterizationInputError(
            f"{field_name} exceeds the {MAX_REGISTRATION_POINTS} point safety cap"
        )
    if not np.all(np.isfinite(points)):
        raise CharacterizationInputError(f"{field_name} contains non-finite coordinates")
    return cast(np.ndarray, points)


def _validated_indices(
    values: Sequence[int] | np.ndarray,
    *,
    field_name: str,
    point_count: int,
) -> tuple[int, ...]:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise CharacterizationInputError(f"{field_name} must contain integer values") from exc
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise CharacterizationInputError(f"{field_name} must be a one-dimensional integer array")
    if array.size == 0:
        raise CharacterizationInputError(f"{field_name} cannot be empty")
    indices = tuple(int(value) for value in array.tolist())
    if len(set(indices)) != len(indices):
        raise CharacterizationInputError(f"{field_name} contains duplicate indices")
    if any(index < 0 or index >= point_count for index in indices):
        raise CharacterizationInputError(f"{field_name} contains an out-of-range index")
    return indices


def _validate_full_dimensional_calibration(
    centered: np.ndarray,
    *,
    field_name: str,
) -> None:
    try:
        singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise CharacterizationInputError(
            f"{field_name} calibration singular-value decomposition did not converge"
        ) from exc
    dimension = centered.shape[1]
    if singular_values.size < dimension or singular_values[0] <= 0.0:
        raise CharacterizationInputError(
            f"{field_name} calibration points do not span {dimension} dimensions"
        )
    ratio = float(singular_values[dimension - 1] / singular_values[0])
    if not math.isfinite(ratio) or ratio <= MIN_CALIBRATION_SINGULAR_VALUE_RATIO:
        raise CharacterizationInputError(
            f"{field_name} calibration points are rank deficient or nearly collinear"
        )


def _residual_statistics(residual_norms: np.ndarray) -> ResidualStatistics:
    maximum = float(np.max(residual_norms))
    rmse = (
        0.0
        if maximum == 0.0
        else maximum * math.sqrt(float(np.mean(np.square(residual_norms / maximum))))
    )
    return ResidualStatistics(
        count=int(residual_norms.size),
        rmse=rmse,
        mean=float(np.mean(residual_norms)),
        median=float(np.median(residual_norms)),
        maximum=maximum,
    )


def fit_rigid_registration(
    source_points: Sequence[Sequence[float]] | np.ndarray,
    target_points: Sequence[Sequence[float]] | np.ndarray,
    *,
    source_frame_id: str,
    target_frame_id: str,
    source_coordinate_unit: str,
    target_coordinate_unit: str,
    source_provenance: DataProvenance,
    target_provenance: DataProvenance,
    calibration_indices: Sequence[int] | np.ndarray,
    held_out_indices: Sequence[int] | np.ndarray,
) -> RigidRegistrationResult:
    """Fit a proper source-to-target rigid transform using calibration points only.

    The Kabsch/SVD objective is ``target ~= R @ source + t`` for column vectors.
    Calibration and held-out indices must be unique, disjoint, and together cover
    every supplied correspondence; omitted difficult points would invalidate a
    held-out claim. Both calibration point sets must span the complete 2D or 3D
    coordinate space; this deliberately rejects ill-conditioned
    collinear/coplanar calibrations. If the unconstrained least-squares optimum is
    a reflection (``det(R) < 0``), the function fails instead of silently
    converting it into a different proper rotation.
    """

    source = _as_finite_points(source_points, field_name="source_points")
    target = _as_finite_points(target_points, field_name="target_points")
    if source.shape != target.shape:
        raise CharacterizationInputError("source_points and target_points shapes must match")

    source_frame = _clean_text(source_frame_id, field_name="source_frame_id")
    target_frame = _clean_text(target_frame_id, field_name="target_frame_id")
    if source_frame == target_frame:
        raise CharacterizationInputError(
            "source_frame_id and target_frame_id must identify distinct coordinate frames"
        )
    source_unit = _clean_text(source_coordinate_unit, field_name="source_coordinate_unit")
    target_unit = _clean_text(target_coordinate_unit, field_name="target_coordinate_unit")
    if source_unit != target_unit:
        raise CharacterizationInputError(
            "source and target coordinate units must match exactly before registration"
        )
    if not isinstance(source_provenance, DataProvenance):
        raise CharacterizationInputError("source_provenance must be DataProvenance")
    if not isinstance(target_provenance, DataProvenance):
        raise CharacterizationInputError("target_provenance must be DataProvenance")

    calibration = _validated_indices(
        calibration_indices,
        field_name="calibration_indices",
        point_count=source.shape[0],
    )
    held_out = _validated_indices(
        held_out_indices,
        field_name="held_out_indices",
        point_count=source.shape[0],
    )
    overlap = set(calibration).intersection(held_out)
    if overlap:
        raise CharacterizationInputError(
            "calibration_indices and held_out_indices must be disjoint to prevent leakage"
        )
    covered = set(calibration).union(held_out)
    if covered != set(range(source.shape[0])):
        raise CharacterizationInputError(
            "calibration_indices and held_out_indices must cover every correspondence"
        )
    dimension = source.shape[1]
    if len(calibration) < dimension + 1:
        raise CharacterizationInputError(
            f"calibration_indices require at least {dimension + 1} points in {dimension}D"
        )

    source_calibration = source[np.asarray(calibration)]
    target_calibration = target[np.asarray(calibration)]
    with np.errstate(over="ignore", invalid="ignore"):
        source_centroid = np.mean(source_calibration, axis=0)
        target_centroid = np.mean(target_calibration, axis=0)
        source_centered = source_calibration - source_centroid
        target_centered = target_calibration - target_centroid
    if not np.all(np.isfinite(source_centered)) or not np.all(np.isfinite(target_centered)):
        raise CharacterizationInputError(
            "calibration centering produced non-finite values; rescale the coordinates"
        )
    _validate_full_dimensional_calibration(source_centered, field_name="source")
    _validate_full_dimensional_calibration(target_centered, field_name="target")

    with np.errstate(over="ignore", invalid="ignore"):
        cross_covariance = source_centered.T @ target_centered
    if not np.all(np.isfinite(cross_covariance)):
        raise CharacterizationInputError(
            "registration cross-covariance is non-finite; rescale the coordinates"
        )
    try:
        left, singular_values, right_transpose = np.linalg.svd(cross_covariance)
    except np.linalg.LinAlgError as exc:
        raise CharacterizationInputError(
            "registration cross-covariance singular-value decomposition did not converge"
        ) from exc
    rotation = right_transpose.T @ left.T
    determinant = float(np.linalg.det(rotation))
    if determinant < 0.0:
        raise ReflectionRegistrationError(
            "point correspondences require a reflection; only proper rigid rotations are allowed"
        )
    if not math.isclose(determinant, 1.0, rel_tol=1.0e-10, abs_tol=1.0e-10):
        raise CharacterizationInputError("SVD did not produce a numerically proper rotation")
    if not np.allclose(rotation.T @ rotation, np.eye(dimension), rtol=1.0e-10, atol=1.0e-10):
        raise CharacterizationInputError("SVD rotation is not numerically orthogonal")

    with np.errstate(over="ignore", invalid="ignore"):
        translation = target_centroid - rotation @ source_centroid
        predicted = source @ rotation.T + translation
        residual_norms = np.linalg.norm(target - predicted, axis=1)
    calibration_residuals = np.array(residual_norms[np.asarray(calibration)], copy=True)
    held_out_residuals = np.array(residual_norms[np.asarray(held_out)], copy=True)
    if not np.all(np.isfinite(residual_norms)):
        raise CharacterizationInputError("registration produced non-finite residuals")

    return RigidRegistrationResult(
        schema_version=RIGID_REGISTRATION_SCHEMA_VERSION,
        source_frame_id=source_frame,
        target_frame_id=target_frame,
        coordinate_unit=source_unit,
        source_provenance=source_provenance,
        target_provenance=target_provenance,
        calibration_indices=calibration,
        held_out_indices=held_out,
        rotation_source_to_target=np.array(rotation, copy=True),
        translation_source_to_target=np.array(translation, copy=True),
        calibration_residual_norms=calibration_residuals,
        held_out_residual_norms=held_out_residuals,
        calibration_statistics=_residual_statistics(calibration_residuals),
        held_out_statistics=_residual_statistics(held_out_residuals),
        calibration_cross_covariance_singular_values=np.array(singular_values, copy=True),
        rotation_determinant=determinant,
    )
