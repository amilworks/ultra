"""Conservative mechanics/degradation data-reduction primitives.

The functions in this module evaluate small, convention-explicit models whose
applicability can be audited deterministically.  They do **not** perform a
fracture, fatigue-life, creep-damage, oxidation/diffusion, or corrosion solve.

All numerical interfaces use the SI quantities named in their fields, except
the Paris-law independent variable, which deliberately uses ``MPa*sqrt(m)`` so
the fitted coefficient retains the units commonly reported with laboratory
fatigue-crack-growth data.  Parameters and observations carry caller-declared
source identifiers and digests; byte retrieval and digest replay are outside
this numerical kernel.  Every calibrated model refuses extrapolation outside
its recorded domain.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

DEGRADATION_SCHEMA_VERSION = "ultra.materials.degradation-primitives.v1"
GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324
# N_A and e are exact in the revised SI, so their product is exact as represented here.
FARADAY_CONSTANT_C_PER_MOL = 96485.33212331001
MAX_OBSERVATIONS = 2_000_000

NASA_LEFM_REFERENCE_URL = (
    "https://ntrs.nasa.gov/api/citations/19970013996/downloads/19970013996.pdf"
)
NASA_CREEP_REFERENCE_URL = (
    "https://ntrs.nasa.gov/api/citations/20210015451/downloads/NASA-TM-20210015451.pdf"
)
ASTM_E399_SCOPE_URL = "https://store.astm.org/standards/e399"
ASTM_E647_SCOPE_URL = "https://store.astm.org/e0647-24.html"
ASTM_G102_SCOPE_URL = "https://store.astm.org/standards/g102"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONSTRAINT_STATES = frozenset({"plane_stress", "plane_strain"})
_OXIDATION_LAWS = frozenset({"linear", "parabolic"})
_OXIDATION_RATE_CONSTANT_UNITS = {
    "linear": "kg*m^-2*s^-1",
    "parabolic": "kg^2*m^-4*s^-1",
}
_GEOMETRY_PARAMETER = "crack_length_over_crack_plus_remaining_ligament"


class DegradationInputError(ValueError):
    """An input is malformed, ambiguous, non-finite, or dimensionally incomplete."""


class CalibrationDomainError(DegradationInputError):
    """A requested evaluation would extrapolate beyond its calibrated domain."""


def _clean_text(value: object, *, field_name: str, max_length: int = 1024) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise DegradationInputError(f"{field_name} is required")
    if len(cleaned) > max_length:
        raise DegradationInputError(f"{field_name} exceeds {max_length} characters")
    if any(ord(character) < 32 for character in cleaned):
        raise DegradationInputError(f"{field_name} contains control characters")
    return cleaned


def _finite_scalar(value: Any, *, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise DegradationInputError(f"{field_name} must be a finite scalar")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DegradationInputError(f"{field_name} must be a finite scalar") from exc
    if not math.isfinite(number):
        raise DegradationInputError(f"{field_name} must be a finite scalar")
    return number


def _positive(value: object, *, field_name: str) -> float:
    number = _finite_scalar(value, field_name=field_name)
    if number <= 0.0:
        raise DegradationInputError(f"{field_name} must be strictly positive")
    return number


def _nonnegative(value: object, *, field_name: str) -> float:
    number = _finite_scalar(value, field_name=field_name)
    if number < 0.0:
        raise DegradationInputError(f"{field_name} cannot be negative")
    return number


@dataclass(frozen=True)
class EvidenceProvenance:
    """Caller-declared origin and digest for observations, parameters, or criteria.

    This value object validates the declaration's structure.  It does not resolve
    the referenced bytes or prove that ``sha256`` matches them.
    """

    artifact_id: str
    sha256: str
    locator: str
    citation: str

    def __post_init__(self) -> None:
        artifact_id = _clean_text(self.artifact_id, field_name="artifact_id", max_length=256)
        digest = str(self.sha256 or "").strip().lower()
        locator = _clean_text(self.locator, field_name="locator", max_length=2048)
        citation = _clean_text(self.citation, field_name="citation", max_length=2048)
        if not _SHA256.fullmatch(digest):
            raise DegradationInputError("sha256 must be exactly 64 hexadecimal digits")
        object.__setattr__(self, "artifact_id", artifact_id)
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "locator", locator)
        object.__setattr__(self, "citation", citation)


@dataclass(frozen=True)
class ClosedInterval:
    """Inclusive, finite calibration interval for one named quantity and unit."""

    quantity: str
    unit: str
    lower: float
    upper: float

    def __post_init__(self) -> None:
        quantity = _clean_text(self.quantity, field_name="quantity", max_length=128)
        unit = _clean_text(self.unit, field_name="unit", max_length=64)
        lower = _finite_scalar(self.lower, field_name=f"{quantity}.lower")
        upper = _finite_scalar(self.upper, field_name=f"{quantity}.upper")
        if lower > upper:
            raise DegradationInputError(f"{quantity} interval lower bound exceeds upper bound")
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def require_contains(self, value: object, *, field_name: str) -> float:
        number = _finite_scalar(value, field_name=field_name)
        if number < self.lower or number > self.upper:
            raise CalibrationDomainError(
                f"{field_name}={number!r} is outside calibrated {self.quantity} "
                f"domain [{self.lower}, {self.upper}] {self.unit}"
            )
        return number


@dataclass(frozen=True)
class GeometryFactorCalibration:
    """One evaluated geometry factor bound to its calibration coordinate."""

    geometry_id: str
    crack_length_definition: str
    nominal_stress_definition: str
    geometry_factor: float
    domain: ClosedInterval
    evaluated_parameter: float
    provenance: EvidenceProvenance

    def __post_init__(self) -> None:
        geometry_id = _clean_text(self.geometry_id, field_name="geometry_id", max_length=256)
        crack_definition = _clean_text(
            self.crack_length_definition,
            field_name="crack_length_definition",
            max_length=512,
        )
        stress_definition = _clean_text(
            self.nominal_stress_definition,
            field_name="nominal_stress_definition",
            max_length=512,
        )
        factor = _positive(self.geometry_factor, field_name="geometry_factor")
        if self.domain.unit != "1":
            raise DegradationInputError("geometry-factor calibration domain unit must be '1'")
        if self.domain.quantity != _GEOMETRY_PARAMETER:
            raise DegradationInputError(
                "geometry-factor domain quantity must be "
                f"{_GEOMETRY_PARAMETER!r} so it can be derived from specimen dimensions"
            )
        evaluated = self.domain.require_contains(
            self.evaluated_parameter,
            field_name=f"geometry.{self.domain.quantity}",
        )
        if not isinstance(self.provenance, EvidenceProvenance):
            raise DegradationInputError("geometry provenance must be EvidenceProvenance")
        object.__setattr__(self, "geometry_id", geometry_id)
        object.__setattr__(self, "crack_length_definition", crack_definition)
        object.__setattr__(self, "nominal_stress_definition", stress_definition)
        object.__setattr__(self, "geometry_factor", factor)
        object.__setattr__(self, "evaluated_parameter", evaluated)


@dataclass(frozen=True)
class ApplicabilityCheck:
    """One numerical or categorical LEFM applicability check."""

    check_id: str
    passed: bool
    observed: str
    criterion: str


@dataclass(frozen=True)
class LEFMModeIResult:
    """Mode-I stress intensity and a declared small-scale-yielding audit."""

    schema_version: str
    stress_intensity_pa_sqrt_m: float
    stress_intensity_mpa_sqrt_m: float
    plastic_zone_radius_m: float
    constraint_state: str
    minimum_dimension_to_plastic_zone_ratio: float
    required_minimum_ratio: float
    derived_geometry_parameter: float
    applicability_checks: tuple[ApplicabilityCheck, ...]
    applicability_passed: bool
    geometry: GeometryFactorCalibration
    criterion_provenance: EvidenceProvenance
    method_reference_url: str = NASA_LEFM_REFERENCE_URL
    standard_scope_reference_url: str = ASTM_E399_SCOPE_URL
    standard_compliance_claimed: bool = False
    limitation: str = (
        "This is an LEFM algebra and small-scale-yielding screen, not an ASTM E399 test, "
        "fracture-toughness measurement, residual-stress analysis, or failure prediction."
    )


def evaluate_mode_i_lefm(
    *,
    nominal_tensile_stress_pa: float,
    crack_length_m: float,
    remaining_ligament_m: float,
    thickness_m: float,
    yield_strength_pa: float,
    constraint_state: str,
    minimum_dimension_to_plastic_zone_ratio: float,
    geometry: GeometryFactorCalibration,
    criterion_provenance: EvidenceProvenance,
) -> LEFMModeIResult:
    """Evaluate ``K_I = Y sigma sqrt(pi a)`` and declared LEFM controls.

    The plastic-zone convention is ``r_p=(K/sigma_y)^2/(alpha*pi)``, with
    ``alpha=2`` for plane stress and ``alpha=6`` for plane strain.  The caller
    supplies and cites the required separation ratio; this function does not
    silently invent a universal validity threshold.
    """

    stress = _positive(nominal_tensile_stress_pa, field_name="nominal_tensile_stress_pa")
    crack = _positive(crack_length_m, field_name="crack_length_m")
    ligament = _positive(remaining_ligament_m, field_name="remaining_ligament_m")
    thickness = _positive(thickness_m, field_name="thickness_m")
    yield_strength = _positive(yield_strength_pa, field_name="yield_strength_pa")
    required_ratio = _positive(
        minimum_dimension_to_plastic_zone_ratio,
        field_name="minimum_dimension_to_plastic_zone_ratio",
    )
    state = _clean_text(constraint_state, field_name="constraint_state")
    if state not in _CONSTRAINT_STATES:
        raise DegradationInputError(
            "constraint_state must be either 'plane_stress' or 'plane_strain'"
        )
    if not isinstance(geometry, GeometryFactorCalibration):
        raise DegradationInputError("geometry must be GeometryFactorCalibration")
    if not isinstance(criterion_provenance, EvidenceProvenance):
        raise DegradationInputError("criterion_provenance must be EvidenceProvenance")

    derived_geometry_parameter = crack / (crack + ligament)
    geometry.domain.require_contains(
        derived_geometry_parameter,
        field_name=f"derived_geometry.{geometry.domain.quantity}",
    )
    if not math.isclose(
        derived_geometry_parameter,
        geometry.evaluated_parameter,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise DegradationInputError(
            "geometry evaluated_parameter does not match crack_length_m / "
            "(crack_length_m + remaining_ligament_m)"
        )

    stress_intensity = geometry.geometry_factor * stress * math.sqrt(math.pi * crack)
    if not math.isfinite(stress_intensity):
        raise DegradationInputError("stress-intensity calculation overflowed")
    alpha = 2.0 if state == "plane_stress" else 6.0
    plastic_zone = (stress_intensity / yield_strength) ** 2 / (alpha * math.pi)
    if not math.isfinite(plastic_zone):
        raise DegradationInputError("plastic-zone calculation overflowed")
    dimension_ratio = min(crack, ligament, thickness) / plastic_zone
    if not math.isfinite(dimension_ratio):
        raise DegradationInputError("small-scale-yielding ratio is non-finite")

    checks = (
        ApplicabilityCheck(
            check_id="geometry-factor-domain",
            passed=True,
            observed=(
                f"derived {geometry.domain.quantity}={derived_geometry_parameter} "
                f"{geometry.domain.unit}"
            ),
            criterion=(
                f"within [{geometry.domain.lower}, {geometry.domain.upper}] {geometry.domain.unit}"
            ),
        ),
        ApplicabilityCheck(
            check_id="nominal-stress-below-yield-strength",
            passed=stress < yield_strength,
            observed=f"sigma/sigma_y={stress / yield_strength:.17g}",
            criterion="sigma/sigma_y < 1",
        ),
        ApplicabilityCheck(
            check_id="small-scale-yielding-dimension-separation",
            passed=dimension_ratio >= required_ratio,
            observed=f"min(a, ligament, thickness)/r_p={dimension_ratio:.17g}",
            criterion=f">= {required_ratio:.17g} (caller-supplied, cited)",
        ),
    )
    return LEFMModeIResult(
        schema_version=DEGRADATION_SCHEMA_VERSION,
        stress_intensity_pa_sqrt_m=stress_intensity,
        stress_intensity_mpa_sqrt_m=stress_intensity / 1.0e6,
        plastic_zone_radius_m=plastic_zone,
        constraint_state=state,
        minimum_dimension_to_plastic_zone_ratio=dimension_ratio,
        required_minimum_ratio=required_ratio,
        derived_geometry_parameter=derived_geometry_parameter,
        applicability_checks=checks,
        applicability_passed=all(check.passed for check in checks),
        geometry=geometry,
        criterion_provenance=criterion_provenance,
    )


@dataclass(frozen=True)
class ParisTestConditions:
    """Conditions held fixed for one classical Paris-regime calibration."""

    material_state_id: str
    environment_id: str
    load_ratio: float
    temperature_k: float
    cycle_frequency_hz: float
    waveform_id: str
    specimen_thickness_m: float
    specimen_geometry_id: str
    delta_k_definition_id: str
    crack_growth_rate_method_id: str

    def __post_init__(self) -> None:
        material = _clean_text(self.material_state_id, field_name="material_state_id")
        environment = _clean_text(self.environment_id, field_name="environment_id")
        ratio = _finite_scalar(self.load_ratio, field_name="load_ratio")
        if ratio >= 1.0:
            raise DegradationInputError("load_ratio must be less than 1")
        temperature = _positive(self.temperature_k, field_name="temperature_k")
        frequency = _positive(self.cycle_frequency_hz, field_name="cycle_frequency_hz")
        waveform = _clean_text(self.waveform_id, field_name="waveform_id")
        thickness = _positive(self.specimen_thickness_m, field_name="specimen_thickness_m")
        geometry = _clean_text(self.specimen_geometry_id, field_name="specimen_geometry_id")
        delta_k_definition = _clean_text(
            self.delta_k_definition_id,
            field_name="delta_k_definition_id",
        )
        growth_method = _clean_text(
            self.crack_growth_rate_method_id,
            field_name="crack_growth_rate_method_id",
        )
        object.__setattr__(self, "material_state_id", material)
        object.__setattr__(self, "environment_id", environment)
        object.__setattr__(self, "load_ratio", ratio)
        object.__setattr__(self, "temperature_k", temperature)
        object.__setattr__(self, "cycle_frequency_hz", frequency)
        object.__setattr__(self, "waveform_id", waveform)
        object.__setattr__(self, "specimen_thickness_m", thickness)
        object.__setattr__(self, "specimen_geometry_id", geometry)
        object.__setattr__(self, "delta_k_definition_id", delta_k_definition)
        object.__setattr__(self, "crack_growth_rate_method_id", growth_method)


@dataclass(frozen=True)
class ResidualSummary:
    count: int
    root_mean_square_log_error: float
    maximum_absolute_log_error: float


@dataclass(frozen=True)
class ParisLawFit:
    """Log-linear Paris fit with an immutable calibration/holdout partition."""

    schema_version: str
    coefficient_c: float
    exponent_m: float
    coefficient_unit: str
    delta_k_domain_mpa_sqrt_m: ClosedInterval
    conditions: ParisTestConditions
    calibration_indices: tuple[int, ...]
    held_out_indices: tuple[int, ...]
    calibration_residuals: ResidualSummary
    held_out_residuals: ResidualSummary
    observations_provenance: EvidenceProvenance
    regression_space: str
    weighting_scheme: str
    method_reference_url: str = ASTM_E647_SCOPE_URL
    standard_compliance_claimed: bool = False
    validation_only: bool = True
    limitation: str = (
        "This unweighted log-space fit describes only positive Paris-regime crack-growth "
        "observations inside the recorded domain. It does not propagate measurement uncertainty "
        "or model initiation, threshold, overload/sequence effects, short cracks, closure, "
        "instability, variable-amplitude life, or component failure."
    )

    def predict_growth_rate_m_per_cycle(
        self,
        delta_k_mpa_sqrt_m: Sequence[float] | np.ndarray | float,
        *,
        conditions: ParisTestConditions,
    ) -> np.ndarray:
        """Predict inside the calibration domain under exactly matching conditions."""

        if conditions != self.conditions:
            raise CalibrationDomainError(
                "Paris-law conditions differ from calibration; load ratio, environment, "
                "temperature, frequency, waveform, material state, and thickness must match"
            )
        values = _positive_array(
            delta_k_mpa_sqrt_m,
            field_name="delta_k_mpa_sqrt_m",
            allow_scalar=True,
        )
        if np.any(values < self.delta_k_domain_mpa_sqrt_m.lower) or np.any(
            values > self.delta_k_domain_mpa_sqrt_m.upper
        ):
            raise CalibrationDomainError(
                "delta_k_mpa_sqrt_m contains values outside the calibrated Paris domain"
            )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            logarithm = math.log(self.coefficient_c) + self.exponent_m * np.log(values)
            predicted = np.exp(logarithm)
        if not np.all(np.isfinite(predicted)) or np.any(predicted <= 0.0):
            raise DegradationInputError("Paris-law prediction overflowed or underflowed")
        output = np.array(predicted, dtype=float, copy=True)
        output.setflags(write=False)
        return cast(np.ndarray, output)


def _positive_array(
    value: Sequence[float] | np.ndarray | float,
    *,
    field_name: str,
    allow_scalar: bool = False,
) -> np.ndarray:
    if isinstance(value, (bool, np.bool_)):
        raise DegradationInputError(f"{field_name} must contain numeric values")
    try:
        raw_array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise DegradationInputError(f"{field_name} must contain numeric values") from exc
    if raw_array.dtype.kind in {"U", "S"}:
        raise DegradationInputError(f"{field_name} must contain numeric values, not strings")
    if np.issubdtype(raw_array.dtype, np.bool_) or (
        raw_array.dtype == object
        and any(isinstance(item, (bool, np.bool_)) for item in raw_array.flat)
    ):
        raise DegradationInputError(f"{field_name} must contain numeric values, not booleans")
    try:
        array = np.asarray(raw_array, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DegradationInputError(f"{field_name} must contain numeric values") from exc
    if allow_scalar and array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1 or array.size == 0:
        raise DegradationInputError(f"{field_name} must be a nonempty one-dimensional array")
    if array.size > MAX_OBSERVATIONS:
        raise DegradationInputError(
            f"{field_name} exceeds the {MAX_OBSERVATIONS} observation safety cap"
        )
    if not np.all(np.isfinite(array)):
        raise DegradationInputError(f"{field_name} contains non-finite values")
    if np.any(array <= 0.0):
        raise DegradationInputError(f"{field_name} must contain strictly positive values")
    return cast(np.ndarray, array)


def _partition_indices(
    count: int,
    calibration_indices: Sequence[int],
    held_out_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    normalized: list[np.ndarray] = []
    for field_name, values, minimum in (
        ("calibration_indices", calibration_indices, 3),
        ("held_out_indices", held_out_indices, 2),
    ):
        if isinstance(values, (str, bytes, bytearray)):
            raise DegradationInputError(f"{field_name} must be a sequence of integers")
        items = list(values)
        if len(items) < minimum:
            raise DegradationInputError(f"{field_name} must contain at least {minimum} rows")
        if any(
            isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer))
            for item in items
        ):
            raise DegradationInputError(f"{field_name} must contain integers")
        array = np.asarray(items, dtype=np.int64)
        if np.any(array < 0) or np.any(array >= count):
            raise DegradationInputError(f"{field_name} contains an out-of-range row")
        if np.unique(array).size != array.size:
            raise DegradationInputError(f"{field_name} contains duplicate rows")
        normalized.append(array)
    calibration, held_out = normalized
    if np.intersect1d(calibration, held_out).size:
        raise DegradationInputError("calibration and held-out rows must be disjoint")
    if np.union1d(calibration, held_out).size != count:
        raise DegradationInputError(
            "calibration and held-out rows must form a complete observation partition"
        )
    return calibration, held_out


def _residual_summary(residual: np.ndarray) -> ResidualSummary:
    if not np.all(np.isfinite(residual)):
        raise DegradationInputError("Paris-law residuals are non-finite")
    with np.errstate(over="ignore", invalid="ignore"):
        root_mean_square = float(np.sqrt(np.mean(np.square(residual))))
    maximum = float(np.max(np.abs(residual)))
    if not math.isfinite(root_mean_square) or not math.isfinite(maximum):
        raise DegradationInputError("Paris-law residual summary overflowed")
    return ResidualSummary(
        count=int(residual.size),
        root_mean_square_log_error=root_mean_square,
        maximum_absolute_log_error=maximum,
    )


def fit_paris_law(
    delta_k_mpa_sqrt_m: Sequence[float] | np.ndarray,
    crack_growth_rate_m_per_cycle: Sequence[float] | np.ndarray,
    *,
    calibration_indices: Sequence[int],
    held_out_indices: Sequence[int],
    conditions: ParisTestConditions,
    observations_provenance: EvidenceProvenance,
) -> ParisLawFit:
    """Fit ``da/dN = C (Delta K)^m`` using calibration rows only."""

    delta_k = _positive_array(delta_k_mpa_sqrt_m, field_name="delta_k_mpa_sqrt_m")
    growth_rate = _positive_array(
        crack_growth_rate_m_per_cycle,
        field_name="crack_growth_rate_m_per_cycle",
    )
    if delta_k.size != growth_rate.size:
        raise DegradationInputError("delta K and crack-growth-rate lengths must match")
    if not isinstance(conditions, ParisTestConditions):
        raise DegradationInputError("conditions must be ParisTestConditions")
    if not isinstance(observations_provenance, EvidenceProvenance):
        raise DegradationInputError("observations_provenance must be EvidenceProvenance")
    calibration, held_out = _partition_indices(
        int(delta_k.size),
        calibration_indices,
        held_out_indices,
    )

    x = np.log(delta_k)
    y = np.log(growth_rate)
    x_calibration = x[calibration]
    if float(np.ptp(x_calibration)) <= 1.0e-12:
        raise DegradationInputError("calibration delta K values do not span a fit domain")
    design = np.column_stack((np.ones(calibration.size), x_calibration))
    coefficients, _residual, rank, _singular = np.linalg.lstsq(
        design,
        y[calibration],
        rcond=None,
    )
    if rank != 2:
        raise DegradationInputError("Paris-law calibration design is rank deficient")
    log_c, exponent = (float(coefficients[0]), float(coefficients[1]))
    if not math.isfinite(log_c) or not math.isfinite(exponent) or exponent <= 0.0:
        raise DegradationInputError("fitted Paris exponent must be finite and strictly positive")
    try:
        coefficient = math.exp(log_c)
    except OverflowError as exc:
        raise DegradationInputError("fitted Paris coefficient overflowed or underflowed") from exc
    if not math.isfinite(coefficient) or coefficient <= 0.0:
        raise DegradationInputError("fitted Paris coefficient overflowed or underflowed")
    predicted_log = log_c + exponent * x
    residuals = y - predicted_log
    domain = ClosedInterval(
        quantity="delta_k",
        unit="MPa*sqrt(m)",
        lower=float(np.min(delta_k[calibration])),
        upper=float(np.max(delta_k[calibration])),
    )
    if np.any(delta_k[held_out] < domain.lower) or np.any(delta_k[held_out] > domain.upper):
        raise CalibrationDomainError(
            "held-out delta K rows must lie inside the calibration interval so holdout "
            "residuals measure interpolation rather than extrapolation"
        )
    return ParisLawFit(
        schema_version=DEGRADATION_SCHEMA_VERSION,
        coefficient_c=coefficient,
        exponent_m=exponent,
        coefficient_unit="m/cycle/(MPa*sqrt(m))^m",
        delta_k_domain_mpa_sqrt_m=domain,
        conditions=conditions,
        calibration_indices=tuple(int(value) for value in calibration),
        held_out_indices=tuple(int(value) for value in held_out),
        calibration_residuals=_residual_summary(residuals[calibration]),
        held_out_residuals=_residual_summary(residuals[held_out]),
        observations_provenance=observations_provenance,
        regression_space="natural_log_da_dN_vs_natural_log_delta_K",
        weighting_scheme="unweighted_ordinary_least_squares",
    )


@dataclass(frozen=True)
class NortonArrheniusCreepModel:
    """Calibrated secondary-creep rate model in a closed material/environment domain."""

    pre_exponential_per_s: float
    reference_stress_pa: float
    stress_exponent: float
    activation_energy_j_per_mol: float
    stress_domain_pa: ClosedInterval
    temperature_domain_k: ClosedInterval
    material_state_id: str
    environment_id: str
    stress_measure_id: str
    provenance: EvidenceProvenance

    def __post_init__(self) -> None:
        pre_exponential = _positive(
            self.pre_exponential_per_s,
            field_name="pre_exponential_per_s",
        )
        reference_stress = _positive(self.reference_stress_pa, field_name="reference_stress_pa")
        exponent = _positive(self.stress_exponent, field_name="stress_exponent")
        activation = _nonnegative(
            self.activation_energy_j_per_mol,
            field_name="activation_energy_j_per_mol",
        )
        if self.stress_domain_pa.quantity != "stress" or self.stress_domain_pa.unit != "Pa":
            raise DegradationInputError("stress_domain_pa must describe 'stress' in 'Pa'")
        if (
            self.temperature_domain_k.quantity != "temperature"
            or self.temperature_domain_k.unit != "K"
        ):
            raise DegradationInputError("temperature_domain_k must describe 'temperature' in 'K'")
        if self.stress_domain_pa.lower <= 0.0 or self.temperature_domain_k.lower <= 0.0:
            raise DegradationInputError("creep stress and temperature domains must be positive")
        material = _clean_text(self.material_state_id, field_name="material_state_id")
        environment = _clean_text(self.environment_id, field_name="environment_id")
        stress_measure = _clean_text(self.stress_measure_id, field_name="stress_measure_id")
        if not isinstance(self.provenance, EvidenceProvenance):
            raise DegradationInputError("provenance must be EvidenceProvenance")
        object.__setattr__(self, "pre_exponential_per_s", pre_exponential)
        object.__setattr__(self, "reference_stress_pa", reference_stress)
        object.__setattr__(self, "stress_exponent", exponent)
        object.__setattr__(self, "activation_energy_j_per_mol", activation)
        object.__setattr__(self, "material_state_id", material)
        object.__setattr__(self, "environment_id", environment)
        object.__setattr__(self, "stress_measure_id", stress_measure)


@dataclass(frozen=True)
class CreepRateResult:
    schema_version: str
    effective_secondary_creep_rate_per_s: float
    stress_pa: float
    temperature_k: float
    model: NortonArrheniusCreepModel
    method_reference_url: str = NASA_CREEP_REFERENCE_URL
    secondary_steady_state_only: bool = True
    limitation: str = (
        "The scalar model does not predict primary/tertiary creep, rupture, damage, multiaxial "
        "flow direction, microstructural evolution, oxidation coupling, or remaining life."
    )


def evaluate_norton_arrhenius_creep_rate(
    model: NortonArrheniusCreepModel,
    *,
    stress_pa: float,
    temperature_k: float,
    material_state_id: str,
    environment_id: str,
) -> CreepRateResult:
    """Evaluate ``A (sigma/sigma_ref)^n exp(-Q/(R T))`` inside its domain."""

    if not isinstance(model, NortonArrheniusCreepModel):
        raise DegradationInputError("model must be NortonArrheniusCreepModel")
    if _clean_text(material_state_id, field_name="material_state_id") != model.material_state_id:
        raise CalibrationDomainError("material_state_id differs from creep calibration")
    if _clean_text(environment_id, field_name="environment_id") != model.environment_id:
        raise CalibrationDomainError("environment_id differs from creep calibration")
    stress = model.stress_domain_pa.require_contains(stress_pa, field_name="stress_pa")
    temperature = model.temperature_domain_k.require_contains(
        temperature_k,
        field_name="temperature_k",
    )
    logarithm = (
        math.log(model.pre_exponential_per_s)
        + model.stress_exponent * math.log(stress / model.reference_stress_pa)
        - model.activation_energy_j_per_mol / (GAS_CONSTANT_J_PER_MOL_K * temperature)
    )
    try:
        rate = math.exp(logarithm)
    except OverflowError as exc:
        raise DegradationInputError("creep-rate evaluation overflowed or underflowed") from exc
    if not math.isfinite(rate) or rate <= 0.0:
        raise DegradationInputError("creep-rate evaluation overflowed or underflowed")
    return CreepRateResult(
        schema_version=DEGRADATION_SCHEMA_VERSION,
        effective_secondary_creep_rate_per_s=rate,
        stress_pa=stress,
        temperature_k=temperature,
        model=model,
    )


@dataclass(frozen=True)
class OxidationKineticsModel:
    """Linear or parabolic areal mass-gain model at one calibrated temperature.

    The constant has no temperature-dependence term, so a multi-temperature
    calibration interval would falsely imply transferability.  The temperature
    domain is therefore required to be a singleton isothermal condition.
    """

    law: str
    rate_constant: float
    rate_constant_unit: str
    initial_areal_mass_gain_kg_per_m2: float
    time_domain_s: ClosedInterval
    temperature_domain_k: ClosedInterval
    material_state_id: str
    environment_id: str
    area_basis_id: str
    provenance: EvidenceProvenance

    def __post_init__(self) -> None:
        law = _clean_text(self.law, field_name="law").lower()
        if law not in _OXIDATION_LAWS:
            raise DegradationInputError("law must be either 'linear' or 'parabolic'")
        constant = _nonnegative(self.rate_constant, field_name="rate_constant")
        expected_rate_constant_unit = _OXIDATION_RATE_CONSTANT_UNITS[law]
        if (
            not isinstance(self.rate_constant_unit, str)
            or self.rate_constant_unit != expected_rate_constant_unit
        ):
            raise DegradationInputError(
                f"rate_constant_unit must be exactly '{expected_rate_constant_unit}' "
                f"for the {law} oxidation law"
            )
        initial = _nonnegative(
            self.initial_areal_mass_gain_kg_per_m2,
            field_name="initial_areal_mass_gain_kg_per_m2",
        )
        if self.time_domain_s.quantity != "time" or self.time_domain_s.unit != "s":
            raise DegradationInputError("time_domain_s must describe 'time' in 's'")
        if self.time_domain_s.lower < 0.0:
            raise DegradationInputError("oxidation time domain cannot be negative")
        if (
            self.temperature_domain_k.quantity != "temperature"
            or self.temperature_domain_k.unit != "K"
        ):
            raise DegradationInputError("temperature_domain_k must describe 'temperature' in 'K'")
        if self.temperature_domain_k.lower <= 0.0:
            raise DegradationInputError("oxidation temperature domain must be positive")
        if self.temperature_domain_k.lower != self.temperature_domain_k.upper:
            raise DegradationInputError(
                "constant-law oxidation requires a singleton isothermal temperature domain"
            )
        material = _clean_text(self.material_state_id, field_name="material_state_id")
        environment = _clean_text(self.environment_id, field_name="environment_id")
        area_basis = _clean_text(self.area_basis_id, field_name="area_basis_id")
        if not isinstance(self.provenance, EvidenceProvenance):
            raise DegradationInputError("provenance must be EvidenceProvenance")
        object.__setattr__(self, "law", law)
        object.__setattr__(self, "rate_constant", constant)
        object.__setattr__(self, "initial_areal_mass_gain_kg_per_m2", initial)
        object.__setattr__(self, "material_state_id", material)
        object.__setattr__(self, "environment_id", environment)
        object.__setattr__(self, "area_basis_id", area_basis)


@dataclass(frozen=True)
class OxidationMassGainResult:
    schema_version: str
    areal_mass_gain_kg_per_m2: float
    exposure_time_s: float
    temperature_k: float
    model: OxidationKineticsModel
    validation_only: bool = True
    limitation: str = (
        "Areal mass gain is not oxide thickness or metal loss. The constant is valid only at the "
        "single calibrated isothermal temperature and has no Arrhenius temperature dependence. "
        "This model excludes transient, breakaway, spallation, volatilization, cyclic, "
        "transport-limited, and multiphase effects."
    )


def evaluate_oxidation_mass_gain(
    model: OxidationKineticsModel,
    *,
    exposure_time_s: float,
    temperature_k: float,
    material_state_id: str,
    environment_id: str,
) -> OxidationMassGainResult:
    """Evaluate a linear or parabolic mass-gain law at its exact isothermal temperature."""

    if not isinstance(model, OxidationKineticsModel):
        raise DegradationInputError("model must be OxidationKineticsModel")
    if _clean_text(material_state_id, field_name="material_state_id") != model.material_state_id:
        raise CalibrationDomainError("material_state_id differs from oxidation calibration")
    if _clean_text(environment_id, field_name="environment_id") != model.environment_id:
        raise CalibrationDomainError("environment_id differs from oxidation calibration")
    time = model.time_domain_s.require_contains(exposure_time_s, field_name="exposure_time_s")
    temperature = model.temperature_domain_k.require_contains(
        temperature_k,
        field_name="temperature_k",
    )
    initial = model.initial_areal_mass_gain_kg_per_m2
    try:
        if model.law == "linear":
            mass_gain = initial + model.rate_constant * time
        else:
            mass_gain = math.sqrt(initial**2 + model.rate_constant * time)
    except OverflowError as exc:
        raise DegradationInputError("oxidation mass-gain evaluation overflowed") from exc
    if not math.isfinite(mass_gain):
        raise DegradationInputError("oxidation mass-gain evaluation overflowed")
    return OxidationMassGainResult(
        schema_version=DEGRADATION_SCHEMA_VERSION,
        areal_mass_gain_kg_per_m2=mass_gain,
        exposure_time_s=time,
        temperature_k=temperature,
        model=model,
    )


@dataclass(frozen=True)
class CorrosionPenetrationInputs:
    """Inputs for Faraday-law conversion to average uniform penetration."""

    corrosion_current_density_a_per_m2: float
    equivalent_mass_kg_per_mol_electron: float
    density_kg_per_m3: float
    current_efficiency: float
    duration_s: float
    material_state_id: str
    environment_id: str
    current_density_area_basis_id: str
    current_density_provenance: EvidenceProvenance
    equivalent_mass_provenance: EvidenceProvenance
    density_provenance: EvidenceProvenance
    efficiency_provenance: EvidenceProvenance

    def __post_init__(self) -> None:
        current = _nonnegative(
            self.corrosion_current_density_a_per_m2,
            field_name="corrosion_current_density_a_per_m2",
        )
        equivalent = _positive(
            self.equivalent_mass_kg_per_mol_electron,
            field_name="equivalent_mass_kg_per_mol_electron",
        )
        density = _positive(self.density_kg_per_m3, field_name="density_kg_per_m3")
        efficiency = _positive(self.current_efficiency, field_name="current_efficiency")
        if efficiency > 1.0:
            raise DegradationInputError("current_efficiency cannot exceed 1")
        duration = _nonnegative(self.duration_s, field_name="duration_s")
        material = _clean_text(self.material_state_id, field_name="material_state_id")
        environment = _clean_text(self.environment_id, field_name="environment_id")
        area_basis = _clean_text(
            self.current_density_area_basis_id,
            field_name="current_density_area_basis_id",
        )
        for field_name in (
            "current_density_provenance",
            "equivalent_mass_provenance",
            "density_provenance",
            "efficiency_provenance",
        ):
            if not isinstance(getattr(self, field_name), EvidenceProvenance):
                raise DegradationInputError(f"{field_name} must be EvidenceProvenance")
        object.__setattr__(self, "corrosion_current_density_a_per_m2", current)
        object.__setattr__(self, "equivalent_mass_kg_per_mol_electron", equivalent)
        object.__setattr__(self, "density_kg_per_m3", density)
        object.__setattr__(self, "current_efficiency", efficiency)
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "material_state_id", material)
        object.__setattr__(self, "environment_id", environment)
        object.__setattr__(self, "current_density_area_basis_id", area_basis)


@dataclass(frozen=True)
class CorrosionPenetrationResult:
    schema_version: str
    uniform_mass_loss_flux_kg_per_m2_s: float
    average_uniform_penetration_rate_m_per_s: float
    average_uniform_penetration_m: float
    inputs: CorrosionPenetrationInputs
    faraday_constant_c_per_mol: float = FARADAY_CONSTANT_C_PER_MOL
    method_reference_url: str = ASTM_G102_SCOPE_URL
    standard_compliance_claimed: bool = False
    limitation: str = (
        "This Faraday-law conversion assumes the stated constant current efficiency and spatially "
        "uniform dissolution. It does not predict pitting, crevice/galvanic corrosion, passivation, "
        "transport limitation, time-varying current, localized depth, or component life."
    )


def convert_corrosion_current_to_uniform_penetration(
    inputs: CorrosionPenetrationInputs,
) -> CorrosionPenetrationResult:
    """Convert corrosion current to average uniform loss using Faraday's law."""

    if not isinstance(inputs, CorrosionPenetrationInputs):
        raise DegradationInputError("inputs must be CorrosionPenetrationInputs")
    mass_flux = (
        inputs.corrosion_current_density_a_per_m2
        * inputs.equivalent_mass_kg_per_mol_electron
        * inputs.current_efficiency
        / FARADAY_CONSTANT_C_PER_MOL
    )
    rate = mass_flux / inputs.density_kg_per_m3
    penetration = rate * inputs.duration_s
    if not all(math.isfinite(value) for value in (mass_flux, rate, penetration)):
        raise DegradationInputError("corrosion penetration conversion overflowed")
    return CorrosionPenetrationResult(
        schema_version=DEGRADATION_SCHEMA_VERSION,
        uniform_mass_loss_flux_kg_per_m2_s=mass_flux,
        average_uniform_penetration_rate_m_per_s=rate,
        average_uniform_penetration_m=penetration,
        inputs=inputs,
    )
