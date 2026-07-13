#!/usr/bin/env python3
"""Run Ultra's provenance-bound two-lane Al-Co-W CALPHAD benchmark.

The calibration lane reproduces uncertainty-bearing phase-composition data that
contributed to the bound thermodynamic assessment.  The held-out lane compares
solidus/liquidus predictions with two post-assessment primary DTA studies.  The
two lanes are deliberately never collapsed into a single claim of independent
validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


BENCHMARK_SCHEMA_VERSION = "1"
BENCHMARK_ID = "materials.calphad.al_co_w_experimental_two_lane.v1"
BENCHMARK_REPORT_SCHEMA_VERSION = "ultra.calphad.experimental_benchmark.v1"
DEFAULT_MANIFEST_RELATIVE_PATH = (
    "backend/deepagents_runtime/materials_data/calphad/"
    "experimental_benchmark_manifest.json"
)
DATABASE_ID = "nist-al-co-w-wang-2017"
CALIBRATION_LANE_ID = "nist_2014_phase_vertex_calibration"
HELD_OUT_LANE_ID = "post_assessment_dta_holdout"
CALIBRATION_OBSERVATION_COUNT = 6
HELD_OUT_OBSERVATION_COUNT = 4
CALIBRATION_WEIGHTED_RMS_Z_MAX = 1.0
CALIBRATION_MAX_ABS_Z_MAX = 2.0
HELD_OUT_MAE_K_MAX = 20.0
HELD_OUT_MAX_ABS_ERROR_K_MAX = 30.0
PHASE_FRACTION_EPSILON = 1e-8
BISECTION_ITERATIONS = 16
MAXIMUM_BOUNDARY_RESOLUTION_K = 0.002
_FLOAT_TOLERANCE = 1e-12


class BenchmarkConfigurationError(ValueError):
    """The source manifest or solver output cannot support the benchmark claim."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise BenchmarkConfigurationError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BenchmarkConfigurationError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise BenchmarkConfigurationError(f"cannot stat {label} {path}: {exc}") from exc
    if not stat.S_ISREG(mode):
        raise BenchmarkConfigurationError(f"{label} must be a regular non-symlink file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkConfigurationError(f"cannot parse {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkConfigurationError(f"{label} must be a JSON object")
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkConfigurationError(f"{label} must be an object")
    return value


def _sequence(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise BenchmarkConfigurationError(f"{label} must be an array")
    return value


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkConfigurationError(f"{label} must be a non-empty string")
    return value.strip()


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise BenchmarkConfigurationError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkConfigurationError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise BenchmarkConfigurationError(f"{label} must be a finite number")
    return number


def _strict_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkConfigurationError(f"{label} must be an integer")
    return value


def _require_exact_number(value: Any, expected: float, *, label: str) -> float:
    number = _finite(value, label=label)
    if not math.isclose(number, expected, rel_tol=0.0, abs_tol=_FLOAT_TOLERANCE):
        raise BenchmarkConfigurationError(f"{label} must remain fixed at {expected}")
    return number


def _contained_regular_file(root: Path, relative: str, *, label: str) -> Path:
    candidate_relative = Path(relative)
    if (
        candidate_relative.is_absolute()
        or not candidate_relative.parts
        or ".." in candidate_relative.parts
    ):
        raise BenchmarkConfigurationError(f"{label} must be a safe repository-relative path")
    candidate = root.joinpath(candidate_relative)
    try:
        mode = candidate.lstat().st_mode
    except OSError as exc:
        raise BenchmarkConfigurationError(f"cannot stat {label} {candidate}: {exc}") from exc
    if not stat.S_ISREG(mode):
        raise BenchmarkConfigurationError(f"{label} must be a regular non-symlink file")
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise BenchmarkConfigurationError(f"{label} escapes the repository root") from exc
    return resolved_candidate


def _composition(
    value: Any,
    *,
    label: str,
    expected_components: Sequence[str] = ("AL", "CO", "W"),
) -> dict[str, float]:
    raw = _mapping(value, label=label)
    if set(raw) != set(expected_components):
        raise BenchmarkConfigurationError(
            f"{label} must contain exactly {sorted(expected_components)}"
        )
    result = {
        component: _finite(raw[component], label=f"{label}.{component}")
        for component in expected_components
    }
    if any(number < 0.0 or number > 1.0 for number in result.values()):
        raise BenchmarkConfigurationError(f"{label} fractions must lie in [0, 1]")
    if not math.isclose(sum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise BenchmarkConfigurationError(f"{label} fractions must sum to one")
    return result


def _indexed_objects(
    values: Any,
    *,
    key: str,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(_sequence(values, label=label)):
        record = _mapping(value, label=f"{label}[{index}]")
        identity = _text(record.get(key), label=f"{label}[{index}].{key}")
        if identity in result:
            raise BenchmarkConfigurationError(f"duplicate {label} identity: {identity}")
        result[identity] = record
    return result


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    """Validate provenance, lane separation, units, and locked score policy."""

    if manifest.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise BenchmarkConfigurationError("unsupported benchmark manifest schema")
    if manifest.get("benchmark_id") != BENCHMARK_ID:
        raise BenchmarkConfigurationError("unexpected benchmark identity")

    binding = _mapping(manifest.get("database_binding"), label="database_binding")
    if binding.get("database_id") != DATABASE_ID or binding.get("format") != "tdb":
        raise BenchmarkConfigurationError("benchmark database identity/format is not fixed")
    if binding.get("pycalphad_version") != "0.11.2":
        raise BenchmarkConfigurationError("benchmark requires pycalphad 0.11.2")
    _require_exact_number(binding.get("pressure_Pa"), 101325.0, label="database pressure")
    components = _sequence(binding.get("components"), label="database components")
    if components != ["AL", "CO", "W", "VA"]:
        raise BenchmarkConfigurationError("database components must remain AL, CO, W, VA")

    database_manifest_path = _contained_regular_file(
        repository_root,
        _text(
            binding.get("database_manifest_relative_path"),
            label="database manifest path",
        ),
        label="database manifest",
    )
    database_path = _contained_regular_file(
        repository_root,
        _text(binding.get("database_relative_path"), label="database path"),
        label="database",
    )
    database_sha256 = sha256_file(database_path)
    database_size = database_path.stat().st_size
    if binding.get("sha256") != database_sha256:
        raise BenchmarkConfigurationError("benchmark database SHA-256 differs from bound bytes")
    if _strict_int(binding.get("size_bytes"), label="database size") != database_size:
        raise BenchmarkConfigurationError("benchmark database size differs from bound bytes")

    database_manifest = _load_json_object(database_manifest_path, label="database manifest")
    database_entries = _indexed_objects(
        database_manifest.get("databases"), key="database_id", label="database manifest entries"
    )
    database_entry = database_entries.get(DATABASE_ID)
    if database_entry is None:
        raise BenchmarkConfigurationError("bound database is absent from the database manifest")
    if (
        database_entry.get("sha256") != database_sha256
        or database_entry.get("size_bytes") != database_size
        or database_entry.get("publication_doi") != binding.get("assessment_doi")
    ):
        raise BenchmarkConfigurationError(
            "experimental benchmark and assessed-database manifest bindings disagree"
        )

    sources = _indexed_objects(manifest.get("sources"), key="source_id", label="sources")
    expected_sources = {
        "nist_lass_2014_900c": ("calibration_source", "CC0-1.0", False),
        "tomaszewska_2018_dta": ("held_out_source", "CC-BY-4.0", True),
        "migas_2020_crystallization": ("held_out_source", "CC-BY-4.0", True),
    }
    if set(sources) != set(expected_sources):
        raise BenchmarkConfigurationError("benchmark must contain exactly three reviewed sources")
    for source_id, (role, license_id, independent) in expected_sources.items():
        source = sources[source_id]
        if (
            source.get("source_role") != role
            or source.get("license_id") != license_id
            or source.get("independent_of_bound_assessment") is not independent
        ):
            raise BenchmarkConfigurationError(f"source provenance drift: {source_id}")
        _text(source.get("license_uri"), label=f"{source_id} license URI")
        _text(source.get("article_doi"), label=f"{source_id} DOI")
        if independent and source.get("measurement_uncertainty_status") != (
            "not_reported_numerically"
        ):
            raise BenchmarkConfigurationError(
                f"held-out source must explicitly disclose absent numerical uncertainty: {source_id}"
            )

    lanes = _indexed_objects(manifest.get("lanes"), key="lane_id", label="lanes")
    if set(lanes) != {CALIBRATION_LANE_ID, HELD_OUT_LANE_ID}:
        raise BenchmarkConfigurationError("benchmark must contain exactly calibration and held-out lanes")
    calibration = lanes[CALIBRATION_LANE_ID]
    held_out = lanes[HELD_OUT_LANE_ID]
    if calibration.get("classification") != "calibration" or calibration.get("required") is not True:
        raise BenchmarkConfigurationError("calibration lane classification/requirement drifted")
    if (
        held_out.get("classification") != "held_out"
        or held_out.get("required") is not True
        or held_out.get("independent_of_bound_assessment") is not True
    ):
        raise BenchmarkConfigurationError("held-out lane independence/requirement drifted")

    calibration_calculation = _mapping(
        calibration.get("calculation"), label="calibration calculation"
    )
    temperature_c = _finite(
        calibration_calculation.get("temperature_degC"),
        label="calibration temperature degC",
    )
    temperature_k = _finite(
        calibration_calculation.get("temperature_K"), label="calibration temperature K"
    )
    if not math.isclose(temperature_k, temperature_c + 273.15, abs_tol=1e-12, rel_tol=0.0):
        raise BenchmarkConfigurationError("calibration Celsius-to-kelvin conversion is inconsistent")
    _require_exact_number(
        calibration_calculation.get("pressure_Pa"),
        101325.0,
        label="calibration pressure",
    )
    _composition(
        calibration_calculation.get("bulk_composition_atomic_fraction"),
        label="calibration composition",
    )
    expected_phases = _sequence(
        calibration_calculation.get("expected_stable_phase_models"),
        label="calibration expected phases",
    )
    if expected_phases != ["BCC_B2", "CO3W", "L12_FCC"]:
        raise BenchmarkConfigurationError("calibration phase-model contract drifted")

    calibration_observations = _indexed_objects(
        calibration.get("observations"),
        key="observation_id",
        label="calibration observations",
    )
    if len(calibration_observations) != CALIBRATION_OBSERVATION_COUNT:
        raise BenchmarkConfigurationError("calibration lane must contain six scalar observations")
    for observation_id, observation in calibration_observations.items():
        if observation.get("phase_model") not in expected_phases:
            raise BenchmarkConfigurationError(f"unknown calibration phase: {observation_id}")
        if observation.get("component") not in {"AL", "W"}:
            raise BenchmarkConfigurationError(f"unknown calibration component: {observation_id}")
        mean = _finite(
            observation.get("mean_atomic_fraction"), label=f"{observation_id} mean"
        )
        ci95 = _finite(
            observation.get("ci95_half_width_atomic_fraction"),
            label=f"{observation_id} confidence interval",
        )
        if not 0.0 <= mean <= 1.0 or not 0.0 < ci95 <= 1.0:
            raise BenchmarkConfigurationError(
                f"calibration composition/uncertainty is out of range: {observation_id}"
            )
    calibration_metrics = _mapping(calibration.get("metrics"), label="calibration metrics")
    _require_exact_number(
        calibration_metrics.get("weighted_rms_z_max"),
        CALIBRATION_WEIGHTED_RMS_Z_MAX,
        label="calibration weighted RMS threshold",
    )
    _require_exact_number(
        calibration_metrics.get("max_abs_z_max"),
        CALIBRATION_MAX_ABS_Z_MAX,
        label="calibration max z threshold",
    )

    calculations = _indexed_objects(
        held_out.get("calculations"), key="condition_id", label="held-out calculations"
    )
    if set(calculations) != {
        "tomaszewska_2018_measured_composition",
        "migas_2020_nominal_composition",
    }:
        raise BenchmarkConfigurationError("held-out calculation identities drifted")
    for condition_id, calculation in calculations.items():
        source_id = _text(
            calculation.get("source_id"), label=f"{condition_id} source identity"
        )
        if source_id not in sources or sources[source_id].get("source_role") != "held_out_source":
            raise BenchmarkConfigurationError(f"invalid held-out source binding: {condition_id}")
        _composition(
            calculation.get("bulk_composition_atomic_fraction"),
            label=f"{condition_id} composition",
        )
        for transition in ("solidus", "liquidus"):
            bracket = _sequence(
                calculation.get(f"{transition}_bracket_K"),
                label=f"{condition_id} {transition} bracket",
            )
            if len(bracket) != 2:
                raise BenchmarkConfigurationError(f"{condition_id} {transition} bracket is invalid")
            lower = _finite(bracket[0], label=f"{condition_id} {transition} lower")
            upper = _finite(bracket[1], label=f"{condition_id} {transition} upper")
            if lower >= upper:
                raise BenchmarkConfigurationError(f"{condition_id} {transition} bracket is empty")

    held_out_observations = _indexed_objects(
        held_out.get("observations"), key="observation_id", label="held-out observations"
    )
    if len(held_out_observations) != HELD_OUT_OBSERVATION_COUNT:
        raise BenchmarkConfigurationError("held-out lane must contain four observations")
    transition_pairs: set[tuple[str, str]] = set()
    for observation_id, observation in held_out_observations.items():
        condition_id = _text(
            observation.get("condition_id"), label=f"{observation_id} condition"
        )
        source_id = _text(observation.get("source_id"), label=f"{observation_id} source")
        transition = _text(
            observation.get("transition"), label=f"{observation_id} transition"
        )
        if condition_id not in calculations or calculations[condition_id].get("source_id") != source_id:
            raise BenchmarkConfigurationError(f"held-out observation binding drift: {observation_id}")
        if transition not in {"solidus", "liquidus"}:
            raise BenchmarkConfigurationError(f"invalid held-out transition: {observation_id}")
        pair = (condition_id, transition)
        if pair in transition_pairs:
            raise BenchmarkConfigurationError(f"duplicate condition/transition observation: {pair}")
        transition_pairs.add(pair)
        temperature_c = _finite(
            observation.get("temperature_degC"), label=f"{observation_id} degC"
        )
        temperature_k = _finite(
            observation.get("temperature_K"), label=f"{observation_id} K"
        )
        if not math.isclose(temperature_k, temperature_c + 273.15, abs_tol=1e-12, rel_tol=0.0):
            raise BenchmarkConfigurationError(
                f"held-out Celsius-to-kelvin conversion is inconsistent: {observation_id}"
            )
        if observation.get("uncertainty_K") is not None or observation.get(
            "uncertainty_status"
        ) != "not_reported_numerically":
            raise BenchmarkConfigurationError(
                f"held-out numerical uncertainty must remain explicitly unreported: {observation_id}"
            )
    if len(transition_pairs) != HELD_OUT_OBSERVATION_COUNT:
        raise BenchmarkConfigurationError("held-out condition/transition coverage is incomplete")

    boundary_policy = _mapping(
        held_out.get("solver_boundary_policy"), label="held-out boundary policy"
    )
    if boundary_policy.get("liquid_phase_model") != "LIQUID":
        raise BenchmarkConfigurationError("held-out liquid phase model drifted")
    _require_exact_number(
        boundary_policy.get("phase_fraction_epsilon"),
        PHASE_FRACTION_EPSILON,
        label="phase fraction epsilon",
    )
    if _strict_int(
        boundary_policy.get("bisection_iterations"), label="bisection iterations"
    ) != BISECTION_ITERATIONS:
        raise BenchmarkConfigurationError("bisection iteration count must remain fixed")
    _require_exact_number(
        boundary_policy.get("maximum_reported_boundary_resolution_K"),
        MAXIMUM_BOUNDARY_RESOLUTION_K,
        label="maximum boundary resolution",
    )
    held_out_metrics = _mapping(held_out.get("metrics"), label="held-out metrics")
    if _strict_int(
        held_out_metrics.get("observation_count"), label="held-out observation count"
    ) != HELD_OUT_OBSERVATION_COUNT:
        raise BenchmarkConfigurationError("held-out metric count drifted")
    _require_exact_number(
        held_out_metrics.get("mae_K_max"),
        HELD_OUT_MAE_K_MAX,
        label="held-out MAE threshold",
    )
    _require_exact_number(
        held_out_metrics.get("max_abs_error_K_max"),
        HELD_OUT_MAX_ABS_ERROR_K_MAX,
        label="held-out maximum-error threshold",
    )

    return {
        "database_path": database_path,
        "database_manifest_path": database_manifest_path,
        "database_sha256": database_sha256,
        "database_size_bytes": database_size,
        "sources": sources,
        "lanes": lanes,
    }


def score_predictions(
    manifest: Mapping[str, Any],
    *,
    calibration_predictions: Mapping[str, float],
    held_out_predictions: Mapping[str, Mapping[str, float]],
    solver_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score already-computed predictions without importing a solver."""

    lanes = _indexed_objects(manifest.get("lanes"), key="lane_id", label="lanes")
    calibration = lanes[CALIBRATION_LANE_ID]
    held_out = lanes[HELD_OUT_LANE_ID]
    calibration_observations = _indexed_objects(
        calibration.get("observations"),
        key="observation_id",
        label="calibration observations",
    )
    if set(calibration_predictions) != set(calibration_observations):
        raise BenchmarkConfigurationError("calibration prediction coverage is not exact")
    calibration_records: list[dict[str, Any]] = []
    z_values: list[float] = []
    for observation_id, observation in calibration_observations.items():
        predicted = _finite(
            calibration_predictions[observation_id],
            label=f"{observation_id} prediction",
        )
        observed = _finite(
            observation.get("mean_atomic_fraction"), label=f"{observation_id} observed"
        )
        ci95 = _finite(
            observation.get("ci95_half_width_atomic_fraction"),
            label=f"{observation_id} confidence interval",
        )
        residual = predicted - observed
        z_value = abs(residual) / ci95
        z_values.append(z_value)
        calibration_records.append(
            {
                "observation_id": observation_id,
                "phase_source_label": observation.get("phase_source_label"),
                "phase_model": observation.get("phase_model"),
                "component": observation.get("component"),
                "observed_mean_atomic_fraction": observed,
                "observed_ci95_half_width_atomic_fraction": ci95,
                "predicted_atomic_fraction": predicted,
                "residual_atomic_fraction": residual,
                "abs_normalized_residual_z": z_value,
            }
        )
    weighted_rms_z = math.sqrt(sum(value * value for value in z_values) / len(z_values))
    maximum_z = max(z_values)
    calibration_passed = (
        weighted_rms_z <= CALIBRATION_WEIGHTED_RMS_Z_MAX + _FLOAT_TOLERANCE
        and maximum_z <= CALIBRATION_MAX_ABS_Z_MAX + _FLOAT_TOLERANCE
    )

    held_out_observations = _indexed_objects(
        held_out.get("observations"), key="observation_id", label="held-out observations"
    )
    expected_conditions = {
        str(observation["condition_id"]) for observation in held_out_observations.values()
    }
    if set(held_out_predictions) != expected_conditions:
        raise BenchmarkConfigurationError("held-out prediction condition coverage is not exact")
    held_out_records: list[dict[str, Any]] = []
    absolute_errors: list[float] = []
    for observation_id, observation in held_out_observations.items():
        condition_id = str(observation["condition_id"])
        transition = str(observation["transition"])
        condition_predictions = _mapping(
            held_out_predictions[condition_id], label=f"{condition_id} predictions"
        )
        prediction_key = f"{transition}_K"
        predicted_k = _finite(
            condition_predictions.get(prediction_key), label=f"{condition_id} {prediction_key}"
        )
        observed_k = _finite(
            observation.get("temperature_K"), label=f"{observation_id} observed K"
        )
        residual_k = predicted_k - observed_k
        absolute_error_k = abs(residual_k)
        absolute_errors.append(absolute_error_k)
        held_out_records.append(
            {
                "observation_id": observation_id,
                "source_id": observation.get("source_id"),
                "condition_id": condition_id,
                "transition": transition,
                "measurement_definition": observation.get("measurement_definition"),
                "observed_temperature_degC": observation.get("temperature_degC"),
                "observed_temperature_K": observed_k,
                "reported_uncertainty_K": observation.get("uncertainty_K"),
                "uncertainty_status": observation.get("uncertainty_status"),
                "predicted_temperature_degC": predicted_k - 273.15,
                "predicted_temperature_K": predicted_k,
                "residual_K_predicted_minus_observed": residual_k,
                "absolute_error_K": absolute_error_k,
            }
        )
    mae_k = sum(absolute_errors) / len(absolute_errors)
    maximum_error_k = max(absolute_errors)
    held_out_passed = (
        mae_k <= HELD_OUT_MAE_K_MAX + _FLOAT_TOLERANCE
        and maximum_error_k <= HELD_OUT_MAX_ABS_ERROR_K_MAX + _FLOAT_TOLERANCE
    )
    status = "passed" if calibration_passed and held_out_passed else "failed"
    blocking_reasons: list[str] = []
    if not calibration_passed:
        blocking_reasons.append("assessment-basis calibration lane exceeded its locked z limits")
    if not held_out_passed:
        blocking_reasons.append(
            "independent thermometric holdout exceeded the locked 20 K MAE or 30 K max limit"
        )
    return {
        "status": status,
        "production_promotion_blocked": not (calibration_passed and held_out_passed),
        "blocking_reasons": blocking_reasons,
        "lanes": {
            "calibration": {
                "lane_id": CALIBRATION_LANE_ID,
                "classification": "calibration",
                "independent_validation": False,
                "required": True,
                "status": "passed" if calibration_passed else "failed",
                "observation_count": len(calibration_records),
                "metrics": {
                    "weighted_rms_z": weighted_rms_z,
                    "weighted_rms_z_max": CALIBRATION_WEIGHTED_RMS_Z_MAX,
                    "max_abs_z": maximum_z,
                    "max_abs_z_max": CALIBRATION_MAX_ABS_Z_MAX,
                },
                "observations": calibration_records,
            },
            "held_out": {
                "lane_id": HELD_OUT_LANE_ID,
                "classification": "held_out",
                "independent_validation": True,
                "required": True,
                "uncertainty_qualification": (
                    "source articles do not report numerical measurement uncertainty; "
                    "fixed engineering tolerances are used"
                ),
                "status": "passed" if held_out_passed else "failed",
                "observation_count": len(held_out_records),
                "metrics": {
                    "mae_K": mae_k,
                    "mae_K_max": HELD_OUT_MAE_K_MAX,
                    "max_abs_error_K": maximum_error_k,
                    "max_abs_error_K_max": HELD_OUT_MAX_ABS_ERROR_K_MAX,
                },
                "observations": held_out_records,
            },
        },
        "solver_evidence": dict(solver_evidence or {}),
    }


class PyCalphadPredictor:
    """Small deterministic adapter around the pinned pycalphad equilibrium API."""

    def __init__(self, database_path: Path, binding: Mapping[str, Any]) -> None:
        try:
            import numpy as np
            import pycalphad
            from pycalphad import Database, equilibrium
            from pycalphad import variables as v
        except ImportError as exc:
            raise BenchmarkConfigurationError(f"pinned pycalphad runtime is unavailable: {exc}") from exc
        if pycalphad.__version__ != binding.get("pycalphad_version"):
            raise BenchmarkConfigurationError(
                "installed pycalphad version differs from the benchmark binding"
            )
        self._np = np
        self._pycalphad = pycalphad
        self._equilibrium = equilibrium
        self._v = v
        self._database = Database(str(database_path))
        self._components = [str(value) for value in binding["components"]]
        self._phases = list(self._database.phases)
        self._pressure_pa = float(binding["pressure_Pa"])
        self._liquid_fraction_cache: dict[tuple[float, float, float], float] = {}

    @property
    def versions(self) -> dict[str, str]:
        return {
            "pycalphad": str(self._pycalphad.__version__),
            "numpy": str(self._np.__version__),
        }

    def _solve(self, temperature_k: float, composition: Mapping[str, float]) -> Any:
        return self._equilibrium(
            self._database,
            self._components,
            self._phases,
            {
                self._v.P: self._pressure_pa,
                self._v.T: [float(temperature_k)],
                self._v.X("AL"): [float(composition["AL"])],
                self._v.X("W"): [float(composition["W"])],
            },
        )

    def calibration_predictions(
        self,
        calculation: Mapping[str, Any],
        observations: Mapping[str, Mapping[str, Any]],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        composition = _composition(
            calculation.get("bulk_composition_atomic_fraction"),
            label="calibration composition",
        )
        result = self._solve(float(calculation["temperature_K"]), composition)
        phase_values = self._np.asarray(result.Phase.values).reshape(-1)
        amount_values = self._np.asarray(result.NP.values, dtype=float).reshape(-1)
        stable: dict[str, float] = {}
        for phase in sorted({str(value) for value in phase_values if str(value)}):
            amount = float(self._np.nansum(amount_values[phase_values == phase]))
            if amount > PHASE_FRACTION_EPSILON:
                stable[phase] = amount
        expected = set(str(value) for value in calculation["expected_stable_phase_models"])
        if set(stable) != expected:
            raise BenchmarkConfigurationError(
                f"calibration stable phases differ from the reviewed contract: {stable}"
            )
        amount_sum = sum(stable.values())
        if not math.isclose(amount_sum, 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise BenchmarkConfigurationError("calibration phase amounts do not close to one")

        component_values: dict[str, Any] = {}
        for component in {str(value["component"]) for value in observations.values()}:
            component_values[component] = self._np.asarray(
                result.X.sel(component=component).values,
                dtype=float,
            ).reshape(-1)
        predictions: dict[str, float] = {}
        for observation_id, observation in observations.items():
            phase = str(observation["phase_model"])
            indices = self._np.flatnonzero(phase_values == phase)
            if len(indices) != 1:
                raise BenchmarkConfigurationError(
                    f"calibration phase {phase} did not produce exactly one stable vertex"
                )
            value = float(component_values[str(observation["component"])][int(indices[0])])
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise BenchmarkConfigurationError(
                    f"calibration phase composition is invalid: {observation_id}"
                )
            predictions[observation_id] = value
        return predictions, {
            "temperature_K": float(calculation["temperature_K"]),
            "pressure_Pa": self._pressure_pa,
            "bulk_composition_atomic_fraction": composition,
            "stable_phase_amounts": stable,
            "phase_amount_closure_error": abs(amount_sum - 1.0),
        }

    def _liquid_fraction(self, temperature_k: float, composition: Mapping[str, float]) -> float:
        key = (
            round(float(temperature_k), 12),
            float(composition["AL"]),
            float(composition["W"]),
        )
        cached = self._liquid_fraction_cache.get(key)
        if cached is not None:
            return cached
        result = self._solve(float(temperature_k), composition)
        phase_values = self._np.asarray(result.Phase.values).reshape(-1)
        amount_values = self._np.asarray(result.NP.values, dtype=float).reshape(-1)
        liquid_fraction = float(
            self._np.nansum(amount_values[phase_values == "LIQUID"])
        )
        if (
            not math.isfinite(liquid_fraction)
            or liquid_fraction < -1e-8
            or liquid_fraction > 1.0 + 1e-8
        ):
            raise BenchmarkConfigurationError("solver returned an invalid liquid phase fraction")
        liquid_fraction = min(1.0, max(0.0, liquid_fraction))
        self._liquid_fraction_cache[key] = liquid_fraction
        return liquid_fraction

    def _boundary(
        self,
        bracket: Sequence[Any],
        *,
        composition: Mapping[str, float],
        transition: str,
    ) -> dict[str, float]:
        lower = _finite(bracket[0], label=f"{transition} lower bracket")
        upper = _finite(bracket[1], label=f"{transition} upper bracket")

        def crossed(fraction: float) -> bool:
            if transition == "solidus":
                return fraction > PHASE_FRACTION_EPSILON
            if transition == "liquidus":
                return fraction >= 1.0 - PHASE_FRACTION_EPSILON
            raise BenchmarkConfigurationError(f"unsupported transition: {transition}")

        lower_fraction = self._liquid_fraction(lower, composition)
        upper_fraction = self._liquid_fraction(upper, composition)
        if crossed(lower_fraction) or not crossed(upper_fraction):
            raise BenchmarkConfigurationError(
                f"{transition} solver bracket does not straddle its boundary"
            )
        for _ in range(BISECTION_ITERATIONS):
            midpoint = (lower + upper) / 2.0
            midpoint_fraction = self._liquid_fraction(midpoint, composition)
            if crossed(midpoint_fraction):
                upper = midpoint
                upper_fraction = midpoint_fraction
            else:
                lower = midpoint
                lower_fraction = midpoint_fraction
        width = upper - lower
        if width > MAXIMUM_BOUNDARY_RESOLUTION_K + _FLOAT_TOLERANCE:
            raise BenchmarkConfigurationError(
                f"{transition} boundary resolution {width} K exceeds the locked limit"
            )
        return {
            "temperature_K": (lower + upper) / 2.0,
            "lower_K": lower,
            "upper_K": upper,
            "resolution_K": width,
            "lower_liquid_fraction": lower_fraction,
            "upper_liquid_fraction": upper_fraction,
        }

    def held_out_predictions(
        self,
        calculations: Mapping[str, Mapping[str, Any]],
    ) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
        predictions: dict[str, dict[str, float]] = {}
        evidence: dict[str, Any] = {}
        for condition_id, calculation in calculations.items():
            composition = _composition(
                calculation.get("bulk_composition_atomic_fraction"),
                label=f"{condition_id} composition",
            )
            solidus = self._boundary(
                calculation["solidus_bracket_K"],
                composition=composition,
                transition="solidus",
            )
            liquidus = self._boundary(
                calculation["liquidus_bracket_K"],
                composition=composition,
                transition="liquidus",
            )
            if solidus["temperature_K"] >= liquidus["temperature_K"]:
                raise BenchmarkConfigurationError(
                    f"predicted solidus is not below liquidus: {condition_id}"
                )
            predictions[condition_id] = {
                "solidus_K": solidus["temperature_K"],
                "liquidus_K": liquidus["temperature_K"],
            }
            evidence[condition_id] = {
                "composition_atomic_fraction": composition,
                "solidus": solidus,
                "liquidus": liquidus,
            }
        return predictions, evidence


def load_validated_manifest(
    repository_root: Path,
    manifest_path: Path | None = None,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    root = repository_root.expanduser().resolve()
    selected_manifest = manifest_path or root / DEFAULT_MANIFEST_RELATIVE_PATH
    if not selected_manifest.is_absolute():
        selected_manifest = root / selected_manifest
    try:
        selected_manifest.resolve().relative_to(root)
    except ValueError as exc:
        raise BenchmarkConfigurationError("benchmark manifest escapes the repository root") from exc
    manifest = _load_json_object(selected_manifest, label="experimental benchmark manifest")
    validated = validate_manifest(manifest, repository_root=root)
    return manifest, selected_manifest.resolve(), validated


def run_benchmark(
    *,
    repository_root: Path,
    manifest_path: Path | None = None,
    predictor: Any | None = None,
) -> dict[str, Any]:
    manifest, selected_manifest, validated = load_validated_manifest(
        repository_root, manifest_path
    )
    lanes = validated["lanes"]
    calibration = lanes[CALIBRATION_LANE_ID]
    held_out = lanes[HELD_OUT_LANE_ID]
    calibration_observations = _indexed_objects(
        calibration.get("observations"),
        key="observation_id",
        label="calibration observations",
    )
    calculations = _indexed_objects(
        held_out.get("calculations"), key="condition_id", label="held-out calculations"
    )
    active_predictor = predictor or PyCalphadPredictor(
        validated["database_path"],
        _mapping(manifest.get("database_binding"), label="database binding"),
    )
    calibration_predictions, calibration_evidence = active_predictor.calibration_predictions(
        _mapping(calibration.get("calculation"), label="calibration calculation"),
        calibration_observations,
    )
    held_out_predictions, held_out_evidence = active_predictor.held_out_predictions(calculations)
    scored = score_predictions(
        manifest,
        calibration_predictions=calibration_predictions,
        held_out_predictions=held_out_predictions,
        solver_evidence={
            "calibration": calibration_evidence,
            "held_out": held_out_evidence,
        },
    )
    return {
        "schema_version": BENCHMARK_REPORT_SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "status": scored["status"],
        "required_independent_invariant": True,
        "production_promotion_blocked": scored["production_promotion_blocked"],
        "blocking_reasons": scored["blocking_reasons"],
        "source_manifest": {
            "relative_path": str(selected_manifest.relative_to(repository_root.resolve())),
            "sha256": sha256_file(selected_manifest),
            "size_bytes": selected_manifest.stat().st_size,
        },
        "database_binding": {
            "database_id": DATABASE_ID,
            "sha256": validated["database_sha256"],
            "size_bytes": validated["database_size_bytes"],
            "format": "tdb",
            "assessment_doi": manifest["database_binding"]["assessment_doi"],
        },
        "library_versions": dict(active_predictor.versions),
        "conventions": dict(_mapping(manifest.get("conventions"), label="conventions")),
        "lanes": scored["lanes"],
        "solver_evidence": scored["solver_evidence"],
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("/workspace"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_benchmark(
            repository_root=args.repo_root,
            manifest_path=args.manifest,
        )
    except BenchmarkConfigurationError as exc:
        print(f"CALPHAD experimental benchmark configuration error: {exc}")
        return 2
    if args.output is not None:
        _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
