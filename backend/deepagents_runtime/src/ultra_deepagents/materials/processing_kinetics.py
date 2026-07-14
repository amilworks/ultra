"""Bounded processing/kinetics calculations backed by qualified solvers.

This shared NumPy-1 module executes the classic Scheil--Gulliver path.  Three
additional selected-resource tools execute in a separately pinned Kawin 0.5 /
NumPy-2 image: transport coefficients, single-phase 1-D diffusion (including
post-solidification-only back diffusion), and bounded binary KWN
precipitation.  Phase field still requires an external qualified solver.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .calphad import (
    COMPOSITION_TOLERANCE,
    DEFAULT_RESULT_BYTES,
    DEFAULT_WALL_TIME_SECONDS,
    MAX_RESULT_BYTES,
    PHASE_FRACTION_TOLERANCE,
    CalphadExecutionError,
    CalphadInputError,
    CalphadTimeoutError,
    CalphadUnsupportedError,
    _bounded_positive_int,
    _bounded_wall_seconds,
    _calculate_equilibrium,
    _canonical_json_bytes,
    _finite_axis,
    _load_inspected_database,
    _result_points,
    _sha256_bytes,
    _wall_time_limit,
    _WallTimeExceededError,
    _warning_messages,
    canonicalize_equilibrium_compositions,
)

SCHEIL_SCHEMA_VERSION = "ultra.materials.scheil-gulliver.v1"
QUALIFIED_SCHEIL_VERSION = "0.3.0"
MAX_SCHEIL_STEPS = 4096
DEFAULT_SCHEIL_STEPS = 2048
MIN_STEP_TEMPERATURE_K = 0.01
MAX_STEP_TEMPERATURE_K = 500.0
MIN_STOP_LIQUID_FRACTION = 1e-8
MAX_STOP_LIQUID_FRACTION = 0.1
SCHEIL_MASS_BALANCE_TOLERANCE = 1e-6
SCHEIL_STOP_CRITERION_ABSOLUTE_TOLERANCE = 1e-12
SCHEIL_CANONICAL_SCALAR_UPPER_BOUND_BYTES = 32
SCHEIL_RESULT_FIXED_UPPER_BOUND_BYTES = 256 * 1024

SCHEIL_ASSUMPTIONS = (
    "Perfect mixing (infinite diffusion) in the liquid.",
    "Local equilibrium at the solid/liquid interface.",
    "No diffusion in solid phases after they form.",
    "Constant pressure of 101325 Pa and a one-mole calculation basis.",
)


def _finite_scalar(
    value: Any,
    *,
    field_name: str,
    minimum: float,
    maximum: float,
) -> float:
    values = _finite_axis(value, field_name=field_name, minimum=minimum, maximum=maximum)
    if len(values) != 1:
        raise CalphadInputError(f"{field_name} must be one finite scalar")
    return float(values[0])


def _scheil_result_upper_bound_bytes(
    *,
    max_steps: int,
    physical_components: Sequence[str],
    phases: Sequence[str],
    database_manifest: Mapping[str, Any],
) -> int:
    """Conservative serialized-result bound used before solver allocation.

    Every retained point can contribute temperature/solid/liquid values, two
    amount series for every solid phase, and one composition value for every
    physical-component/phase pair.  CPython's finite binary64 JSON spelling is
    shorter than 32 bytes including a separator; the larger constant keeps this
    calculation independent of the particular values returned by the solver.
    """

    phase_count = len(phases)
    component_count = len(physical_components)
    solid_phase_count = max(0, phase_count - 1)
    scalar_series = 3 + (2 * solid_phase_count) + (phase_count * component_count)
    scalar_bytes = max_steps * scalar_series * SCHEIL_CANONICAL_SCALAR_UPPER_BOUND_BYTES
    # Phase/component names occur as request values and nested object keys. Use
    # their actual UTF-8 sizes with a generous multiplier for JSON quoting and
    # repeated amount/composition maps.
    label_bytes = 8 * (
        sum(len(name.encode("utf-8")) + 8 for name in phases)
        + phase_count * sum(len(name.encode("utf-8")) + 8 for name in physical_components)
    )
    manifest_bytes = len(_canonical_json_bytes(database_manifest))
    return SCHEIL_RESULT_FIXED_UPPER_BOUND_BYTES + (2 * manifest_bytes) + label_bytes + scalar_bytes


def _qualified_scheil_solver() -> Any:
    try:
        import scheil
        from scheil import simulate_scheil_solidification
    except ImportError as exc:
        raise CalphadUnsupportedError(
            f"Scheil execution requires the pinned scheil=={QUALIFIED_SCHEIL_VERSION} runtime"
        ) from exc
    actual_version = str(getattr(scheil, "__version__", ""))
    if actual_version != QUALIFIED_SCHEIL_VERSION:
        raise CalphadUnsupportedError(
            "Scheil runtime version is outside the qualified envelope: "
            f"expected {QUALIFIED_SCHEIL_VERSION}, got {actual_version or 'unknown'}"
        )
    return simulate_scheil_solidification


def _simulate_scheil(
    database: Any,
    components: Sequence[str],
    phases: Sequence[str],
    composition: Mapping[Any, float],
    *,
    start_temperature_K: float,  # noqa: N803
    step_temperature_K: float,  # noqa: N803
    liquid_phase_name: str,
    stop_liquid_fraction: float,
) -> Any:
    solver = _qualified_scheil_solver()
    return solver(
        database,
        list(components),
        list(phases),
        dict(composition),
        start_temperature_K,
        step_temperature=step_temperature_K,
        liquid_phase_name=liquid_phase_name,
        stop=stop_liquid_fraction,
        adaptive=True,
        verbose=False,
    )


def _validated_series(
    values: Any,
    *,
    name: str,
    count: int | None = None,
    maximum_count: int | None = None,
) -> list[float]:
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise CalphadExecutionError(f"Scheil result {name} is not a series") from exc
    result: list[float] = []
    hard_limit = count if count is not None else maximum_count
    for index, raw in enumerate(iterator):
        if hard_limit is not None and index >= hard_limit:
            expected = f"expected {count}" if count is not None else f"limit is {maximum_count}"
            raise CalphadExecutionError(f"Scheil result {name} exceeds its point bound; {expected}")
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CalphadExecutionError(f"Scheil result {name} is non-numeric") from exc
        if not math.isfinite(value):
            raise CalphadExecutionError(f"Scheil result {name} contains a non-finite value")
        result.append(0.0 if value == 0 else value)
    if count is not None and len(result) != count:
        raise CalphadExecutionError(
            f"Scheil result {name} has {len(result)} points; expected {count}"
        )
    return result


def _validated_phase_compositions(
    raw: Any,
    *,
    count: int,
    phases: set[str],
    components: set[str],
) -> dict[str, dict[str, list[float | None]]]:
    if not isinstance(raw, Mapping):
        raise CalphadExecutionError("Scheil phase_compositions is not a mapping")
    result: dict[str, dict[str, list[float | None]]] = {}
    for raw_phase, raw_components in sorted(raw.items(), key=lambda item: str(item[0])):
        phase = str(raw_phase).strip().upper()
        if phase not in phases:
            raise CalphadExecutionError(f"Scheil returned unrequested phase {phase!r}")
        if not isinstance(raw_components, Mapping):
            raise CalphadExecutionError(f"Scheil compositions for {phase} are not a mapping")
        phase_result: dict[str, list[float | None]] = {}
        for raw_component, raw_values in sorted(
            raw_components.items(), key=lambda item: str(item[0])
        ):
            component = str(raw_component).strip().upper()
            if component in {"VA", "/-"} or component not in components:
                raise CalphadExecutionError(
                    f"Scheil returned an unexpected composition component {component!r}"
                )
            try:
                iterator = iter(raw_values)
            except TypeError as exc:
                raise CalphadExecutionError(
                    f"Scheil composition {phase}/{component} is not a series"
                ) from exc
            normalized: list[float | None] = []
            for point_index, raw_value in enumerate(iterator):
                if point_index >= count:
                    raise CalphadExecutionError(
                        f"Scheil composition {phase}/{component} exceeds {count} points"
                    )
                try:
                    value = float(raw_value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise CalphadExecutionError(
                        f"Scheil composition {phase}/{component} is non-numeric"
                    ) from exc
                if math.isnan(value):
                    normalized.append(None)
                    continue
                if not math.isfinite(value):
                    raise CalphadExecutionError(
                        f"Scheil composition {phase}/{component} is infinite"
                    )
                if value < -COMPOSITION_TOLERANCE or value > 1 + COMPOSITION_TOLERANCE:
                    raise CalphadExecutionError(
                        f"Scheil composition {phase}/{component} is outside [0, 1]"
                    )
                normalized.append(min(1.0, max(0.0, value)))
            if len(normalized) != count:
                raise CalphadExecutionError(
                    f"Scheil composition {phase}/{component} has the wrong point count"
                )
            phase_result[component] = normalized
        if set(phase_result) != components:
            missing = sorted(components - set(phase_result))
            raise CalphadExecutionError(
                f"Scheil composition {phase} is missing physical components {missing!r}"
            )
        for point_index in range(count):
            point_values = [values[point_index] for values in phase_result.values()]
            finite_values = [value for value in point_values if value is not None]
            if finite_values and len(finite_values) != len(point_values):
                raise CalphadExecutionError(
                    f"Scheil composition {phase} is partially missing at point {point_index}"
                )
            if finite_values and not math.isclose(
                math.fsum(finite_values),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise CalphadExecutionError(
                    f"Scheil composition {phase} does not close at point {point_index}"
                )
        result[phase] = phase_result
    return result


def _validated_scheil_mass_balance(
    *,
    bulk_composition: Mapping[str, float],
    fraction_liquid: Sequence[float],
    phase_amounts: Mapping[str, Sequence[float]],
    cumulative_phase_amounts: Mapping[str, Sequence[float]],
    phase_compositions: Mapping[str, Mapping[str, Sequence[float | None]]],
    liquid_phase_name: str,
) -> dict[str, Any]:
    """Reconstruct every elemental inventory from retained Scheil increments.

    At retained point ``i``, the one-mole-basis inventory is
    ``f_L[i] * x_L[i] + sum_p sum_{j<=i}(delta_f_p[j] * x_p[j])``.
    The upstream result intentionally stores no initial liquid composition at
    point zero, so that declared single-liquid initial condition is closed
    directly against the governed bulk composition.
    """

    components = tuple(sorted(bulk_composition))
    if not components or liquid_phase_name not in phase_compositions:
        raise CalphadExecutionError("Scheil elemental mass-balance inputs are incomplete")
    count = len(fraction_liquid)
    if count < 2:
        raise CalphadExecutionError("Scheil elemental mass balance needs at least two points")
    if not math.isclose(
        float(fraction_liquid[0]),
        1.0,
        rel_tol=0.0,
        abs_tol=PHASE_FRACTION_TOLERANCE,
    ):
        raise CalphadExecutionError("Scheil initial mass-balance point must be all liquid")

    for phase, increments in phase_amounts.items():
        if phase not in cumulative_phase_amounts or phase not in phase_compositions:
            raise CalphadExecutionError(
                f"Scheil phase {phase!r} lacks cumulative amounts or compositions"
            )
        running = 0.0
        for point_index, increment in enumerate(increments):
            if point_index == 0 and not math.isclose(
                float(increment),
                0.0,
                rel_tol=0.0,
                abs_tol=PHASE_FRACTION_TOLERANCE,
            ):
                raise CalphadExecutionError(
                    f"Scheil initial solid increment for {phase} must be zero"
                )
            running = math.fsum((running, float(increment)))
            if not math.isclose(
                running,
                float(cumulative_phase_amounts[phase][point_index]),
                rel_tol=0.0,
                abs_tol=SCHEIL_MASS_BALANCE_TOLERANCE,
            ):
                raise CalphadExecutionError(
                    "Scheil cumulative phase amount disagrees with retained increments "
                    f"for {phase} at point {point_index}"
                )

    maximum_errors = {component: 0.0 for component in components}
    running_solid_inventory = {component: 0.0 for component in components}
    final_reconstructed: dict[str, float] = {}
    for point_index in range(count):
        if point_index > 0:
            for phase, increments in phase_amounts.items():
                increment = float(increments[point_index])
                for component in components:
                    composition = phase_compositions[phase][component][point_index]
                    if composition is None:
                        if increment > PHASE_FRACTION_TOLERANCE:
                            raise CalphadExecutionError(
                                f"Scheil solid composition {phase}/{component} is missing "
                                f"for a positive increment at point {point_index}"
                            )
                        continue
                    running_solid_inventory[component] = math.fsum(
                        (
                            running_solid_inventory[component],
                            increment * float(composition),
                        )
                    )
        reconstructed: dict[str, float] = {}
        for component in components:
            target = float(bulk_composition[component])
            if point_index == 0:
                reconstructed_value = target
            else:
                liquid_composition = phase_compositions[liquid_phase_name][component][point_index]
                if liquid_composition is None:
                    raise CalphadExecutionError(
                        "Scheil liquid composition is missing while liquid remains at "
                        f"point {point_index}"
                    )
                reconstructed_value = math.fsum(
                    (
                        float(fraction_liquid[point_index]) * float(liquid_composition),
                        running_solid_inventory[component],
                    )
                )
            error = abs(reconstructed_value - target)
            if error > SCHEIL_MASS_BALANCE_TOLERANCE:
                raise CalphadExecutionError(
                    "Scheil elemental mass balance does not close for "
                    f"{component} at point {point_index}: absolute_error={error:.12g}"
                )
            maximum_errors[component] = max(maximum_errors[component], error)
            reconstructed[component] = reconstructed_value
        if not math.isclose(
            math.fsum(reconstructed.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=SCHEIL_MASS_BALANCE_TOLERANCE,
        ):
            raise CalphadExecutionError(
                f"Scheil reconstructed component inventory does not sum to one at {point_index}"
            )
        final_reconstructed = reconstructed

    return {
        "basis": "one_mole_initial_bulk",
        "formula": (
            "bulk_x[c] = fraction_liquid[i] * liquid_x[c,i] + "
            "sum_phase,sum_step<=i(solid_increment[phase,step] * solid_x[phase,c,step])"
        ),
        "absolute_tolerance": SCHEIL_MASS_BALANCE_TOLERANCE,
        "maximum_absolute_component_error": max(maximum_errors.values()),
        "maximum_absolute_error_by_component": maximum_errors,
        "final_reconstructed_bulk_composition_mole_fraction": final_reconstructed,
        "all_retained_points_closed": True,
    }


def _validated_scheil_result(
    raw: Any,
    *,
    requested_phases: Sequence[str],
    physical_components: Sequence[str],
    bulk_composition: Mapping[str, float],
    liquid_phase_name: str,
    assessment_temperature_limits_K: Sequence[float] | None,  # noqa: N803
    stop_liquid_fraction: float,
    max_steps: int,
) -> dict[str, Any]:
    if str(getattr(raw, "method", "")).casefold() != "scheil":
        raise CalphadExecutionError("solver did not identify its result as a Scheil calculation")
    if not bool(getattr(raw, "converged", False)):
        raise CalphadExecutionError(
            "Scheil solver did not reach the requested residual-liquid criterion; "
            "the partial/final-fill path is not accepted as scientific evidence"
        )
    temperatures = _validated_series(
        getattr(raw, "temperatures", None),
        name="temperatures",
        maximum_count=max_steps,
    )
    raw_count = len(temperatures)
    if raw_count < 2 or raw_count > max_steps:
        raise CalphadExecutionError(
            f"Scheil result point count {raw_count} is outside [2, {max_steps}]"
        )
    if any(right > left + 1e-10 for left, right in zip(temperatures, temperatures[1:])):
        raise CalphadExecutionError("Scheil temperatures are not monotonically non-increasing")
    if assessment_temperature_limits_K is not None:
        lower, upper = (float(value) for value in assessment_temperature_limits_K)
        if min(temperatures) < lower or max(temperatures) > upper:
            raise CalphadExecutionError(
                "Scheil solver left the declared assessment/TDB temperature limits"
            )

    fraction_solid = _validated_series(
        getattr(raw, "fraction_solid", None), name="fraction_solid", count=raw_count
    )
    fraction_liquid = _validated_series(
        getattr(raw, "fraction_liquid", None), name="fraction_liquid", count=raw_count
    )
    # scheil 0.3.0 appends a same-temperature, nominally fully-solid point even
    # after the requested liquid criterion was reached.  Its phase increments
    # do not necessarily close at that synthetic fill point.  Retain the last
    # physically converged step and discard only this exact, detectable shape.
    terminal_fill_discarded = (
        raw_count >= 3
        and math.isclose(temperatures[-1], temperatures[-2], rel_tol=0.0, abs_tol=1e-10)
        and fraction_liquid[-2] < stop_liquid_fraction
        and math.isclose(fraction_solid[-1], 1.0, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(fraction_liquid[-1], 0.0, rel_tol=0.0, abs_tol=1e-12)
    )
    count = raw_count - 1 if terminal_fill_discarded else raw_count
    temperatures = temperatures[:count]
    fraction_solid = fraction_solid[:count]
    fraction_liquid = fraction_liquid[:count]
    for index, (solid, liquid) in enumerate(zip(fraction_solid, fraction_liquid)):
        if (
            solid < -PHASE_FRACTION_TOLERANCE
            or solid > 1 + PHASE_FRACTION_TOLERANCE
            or liquid < -PHASE_FRACTION_TOLERANCE
            or liquid > 1 + PHASE_FRACTION_TOLERANCE
        ):
            raise CalphadExecutionError(f"Scheil fraction is outside [0, 1] at point {index}")
        if not math.isclose(solid + liquid, 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise CalphadExecutionError(f"Scheil solid/liquid fractions do not close at {index}")
    if any(
        right + PHASE_FRACTION_TOLERANCE < left
        for left, right in zip(fraction_solid, fraction_solid[1:])
    ):
        raise CalphadExecutionError("Scheil fraction_solid is not monotonically non-decreasing")
    if not math.isclose(
        fraction_solid[0], 0.0, rel_tol=0.0, abs_tol=PHASE_FRACTION_TOLERANCE
    ) or not math.isclose(fraction_liquid[0], 1.0, rel_tol=0.0, abs_tol=PHASE_FRACTION_TOLERANCE):
        raise CalphadExecutionError("Scheil initial point must be zero solid and all liquid")
    if fraction_liquid[-1] >= stop_liquid_fraction + SCHEIL_STOP_CRITERION_ABSOLUTE_TOLERANCE:
        raise CalphadExecutionError(
            "Scheil result claims convergence above the requested residual-liquid criterion"
        )

    phase_set = set(requested_phases)
    phase_amounts_raw = getattr(raw, "phase_amounts", None)
    cumulative_raw = getattr(raw, "cum_phase_amounts", None)
    if not isinstance(phase_amounts_raw, Mapping) or not isinstance(cumulative_raw, Mapping):
        raise CalphadExecutionError("Scheil phase amounts are not mappings")
    phase_amounts: dict[str, list[float]] = {}
    cumulative: dict[str, list[float]] = {}
    for raw_phase, raw_values in sorted(phase_amounts_raw.items(), key=lambda item: str(item[0])):
        phase = str(raw_phase).strip().upper()
        if phase == liquid_phase_name or phase not in phase_set:
            raise CalphadExecutionError(f"Scheil returned invalid solid phase {phase!r}")
        values = _validated_series(raw_values, name=f"phase_amounts/{phase}", count=raw_count)[
            :count
        ]
        if any(value < -PHASE_FRACTION_TOLERANCE for value in values):
            raise CalphadExecutionError(f"Scheil returned negative increments for {phase}")
        phase_amounts[phase] = [max(0.0, value) for value in values]
    for raw_phase, raw_values in sorted(cumulative_raw.items(), key=lambda item: str(item[0])):
        phase = str(raw_phase).strip().upper()
        if phase not in phase_amounts:
            raise CalphadExecutionError(f"Scheil cumulative phases disagree for {phase!r}")
        values = _validated_series(raw_values, name=f"cum_phase_amounts/{phase}", count=raw_count)[
            :count
        ]
        if any(value < -PHASE_FRACTION_TOLERANCE for value in values):
            raise CalphadExecutionError(f"Scheil returned negative cumulative amount for {phase}")
        if any(right + PHASE_FRACTION_TOLERANCE < left for left, right in zip(values, values[1:])):
            raise CalphadExecutionError(f"Scheil cumulative amount decreases for {phase}")
        cumulative[phase] = [max(0.0, value) for value in values]
    if set(cumulative) != set(phase_amounts):
        raise CalphadExecutionError("Scheil instantaneous and cumulative phase sets disagree")
    for point_index, solid in enumerate(fraction_solid):
        cumulative_sum = math.fsum(values[point_index] for values in cumulative.values())
        if not math.isclose(cumulative_sum, solid, rel_tol=0.0, abs_tol=1e-6):
            raise CalphadExecutionError(
                f"Scheil cumulative solid phases do not close at point {point_index}"
            )

    phase_compositions = _validated_phase_compositions(
        getattr(raw, "phase_compositions", None),
        count=raw_count,
        phases=phase_set,
        components=set(physical_components),
    )
    if terminal_fill_discarded:
        phase_compositions = {
            phase: {component: values[:count] for component, values in component_values.items()}
            for phase, component_values in phase_compositions.items()
        }
    if liquid_phase_name not in phase_compositions:
        raise CalphadExecutionError("Scheil result lacks liquid phase compositions")
    if set(bulk_composition) != set(physical_components):
        raise CalphadExecutionError("Scheil governed bulk composition is incomplete")
    mass_balance = _validated_scheil_mass_balance(
        bulk_composition=bulk_composition,
        fraction_liquid=fraction_liquid,
        phase_amounts=phase_amounts,
        cumulative_phase_amounts=cumulative,
        phase_compositions=phase_compositions,
        liquid_phase_name=liquid_phase_name,
    )
    return {
        "point_count": count,
        "temperatures_K": temperatures,
        "fraction_solid": fraction_solid,
        "fraction_liquid": fraction_liquid,
        "solid_phase_increment_fraction": phase_amounts,
        "solid_phase_cumulative_fraction": cumulative,
        "phase_composition_mole_fraction": phase_compositions,
        "elemental_mass_balance": mass_balance,
        "converged": True,
        "qualified_terminal_point": "last_residual_liquid_point",
        "discarded_upstream_terminal_fill_point": terminal_fill_discarded,
        "closure_tolerances": {
            "phase_fraction_absolute": 1e-6,
            "composition_absolute": 1e-6,
            "elemental_mass_balance_absolute": SCHEIL_MASS_BALANCE_TOLERANCE,
            "residual_liquid_criterion_absolute": (SCHEIL_STOP_CRITERION_ABSOLUTE_TOLERANCE),
        },
    }


def run_scheil_solidification(  # noqa: N803 - scientific unit suffix is part of the API
    path: str | Path,
    *,
    components: Iterable[str],
    phases: Iterable[str],
    independent_composition: Mapping[str, float],
    start_temperature_K: float,  # noqa: N803
    step_temperature_K: float = 1.0,  # noqa: N803
    pressure_Pa: float = 101325.0,  # noqa: N803
    liquid_phase_name: str = "LIQUID",
    stop_liquid_fraction: float = 1e-4,
    source: str = "",
    license_id: str = "",
    artifact_id: str = "",
    assessment_scope: str = "",
    reference_state: str = "",
    database_id: str = "",
    expected_sha256: str = "",
    expected_size_bytes: int | None = None,
    assessment_temperature_limits_K: Iterable[float] | None = None,  # noqa: N803
    assessment_pressure_limits_Pa: Iterable[float] | None = None,  # noqa: N803
    max_steps: int = DEFAULT_SCHEIL_STEPS,
    wall_time_seconds: float = DEFAULT_WALL_TIME_SECONDS,
    max_result_bytes: int = DEFAULT_RESULT_BYTES,
) -> dict[str, Any]:
    """Run a provenance-bound, fail-closed classic Scheil--Gulliver path.

    This is not a back-diffusion, finite-rate diffusion, precipitation, or
    phase-field solver.  Its only kinetic idealization is the classic
    Scheil--Gulliver assumption set recorded in every response.
    """

    database, manifest = _load_inspected_database(
        path,
        components=components,
        phases=phases,
        source=source,
        license_id=license_id,
        artifact_id=artifact_id,
        assessment_scope=assessment_scope,
        reference_state=reference_state,
        database_id=database_id,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
        assessment_temperature_limits_K=assessment_temperature_limits_K,
        assessment_pressure_limits_Pa=assessment_pressure_limits_Pa,
    )
    normalized_components = tuple(manifest["requested_components"])
    normalized_phases = tuple(manifest["requested_phases"])
    physical_components = tuple(
        component for component in normalized_components if component not in {"VA", "/-"}
    )
    normalized_liquid = str(liquid_phase_name or "").strip().upper()
    if normalized_liquid not in normalized_phases:
        raise CalphadInputError("liquid_phase_name must be one of the requested phases")
    start_temperature = _finite_scalar(
        start_temperature_K,
        field_name="start_temperature_K",
        minimum=1.0,
        maximum=10_000.0,
    )
    step_temperature = _finite_scalar(
        step_temperature_K,
        field_name="step_temperature_K",
        minimum=MIN_STEP_TEMPERATURE_K,
        maximum=MAX_STEP_TEMPERATURE_K,
    )
    pressure = _finite_scalar(
        pressure_Pa,
        field_name="pressure_Pa",
        minimum=1e-9,
        maximum=1e12,
    )
    if pressure != 101325.0:
        raise CalphadInputError("qualified Scheil execution requires pressure_Pa=101325 exactly")
    stop_fraction = _finite_scalar(
        stop_liquid_fraction,
        field_name="stop_liquid_fraction",
        minimum=MIN_STOP_LIQUID_FRACTION,
        maximum=MAX_STOP_LIQUID_FRACTION,
    )
    step_limit = _bounded_positive_int(max_steps, field_name="max_steps", maximum=MAX_SCHEIL_STEPS)
    wall_limit = _bounded_wall_seconds(wall_time_seconds)
    result_limit = _bounded_positive_int(
        max_result_bytes, field_name="max_result_bytes", maximum=MAX_RESULT_BYTES
    )
    assessment_limits = manifest["assessment_temperature_limits_K"]
    if assessment_limits is None:
        raise CalphadInputError(
            "qualified Scheil execution requires declared assessment_temperature_limits_K "
            "to preflight the solver step bound"
        )
    if not (assessment_limits[0] <= start_temperature <= assessment_limits[1]):
        raise CalphadInputError("start_temperature_K is outside the declared assessment/TDB limits")
    pressure_limits = manifest["assessment_pressure_limits_Pa"]
    if pressure_limits is not None and not (pressure_limits[0] <= pressure <= pressure_limits[1]):
        raise CalphadInputError("pressure_Pa is outside the declared assessment limits")
    minimum_effective_step = step_temperature / (1.2**6)
    worst_case_points = 2 + math.ceil(
        max(0.0, start_temperature - assessment_limits[0]) / minimum_effective_step
    )
    if worst_case_points > step_limit:
        raise CalphadInputError(
            "temperature range/step can exceed max_steps after solver step refinement"
        )
    estimated_result_upper_bound = _scheil_result_upper_bound_bytes(
        max_steps=min(step_limit, worst_case_points),
        physical_components=physical_components,
        phases=normalized_phases,
        database_manifest=manifest,
    )
    if estimated_result_upper_bound > result_limit:
        raise CalphadInputError(
            "requested Scheil phase/component/step cardinality cannot fit the governed "
            "max_result_bytes bound: conservative_upper_bound="
            f"{estimated_result_upper_bound} > {result_limit}"
        )

    composition_axes, dependent_component, closure_records = canonicalize_equilibrium_compositions(
        independent_composition,
        components=normalized_components,
    )
    if any(len(values) != 1 for values in composition_axes.values()) or len(closure_records) != 1:
        raise CalphadInputError("Scheil execution requires one composition point")
    from pycalphad import variables as v

    pycalphad_composition = {
        v.X(component): values[0] for component, values in composition_axes.items()
    }
    preflight_conditions = {
        v.T: [start_temperature],
        v.P: [pressure],
        v.N: [1.0],
        **{v.X(component): list(values) for component, values in composition_axes.items()},
    }
    timeout_stage = "liquid preflight"
    try:
        # One POSIX timer covers the liquid-state preflight, result validation, and
        # Scheil solve.  Restarting the same wall limit for each stage would make the
        # advertised limit look end-to-end while allowing nearly twice that duration.
        with _wall_time_limit(wall_limit):
            preflight_dataset = _calculate_equilibrium(
                database,
                normalized_components,
                normalized_phases,
                preflight_conditions,
            )
            preflight_points = _result_points(
                preflight_dataset,
                condition_axes={"T": (start_temperature,), "P": (pressure,), "N": (1.0,)},
                compositions=composition_axes,
                dependent_component=dependent_component,
                requested_components=normalized_components,
                requested_phases=normalized_phases,
            )
            if len(preflight_points) != 1:
                raise CalphadExecutionError(
                    "Scheil liquid preflight did not return exactly one point"
                )
            preflight_phases = preflight_points[0]["stable_phases"]
            liquid_amount = math.fsum(
                float(phase["NP_phase_fraction"])
                for phase in preflight_phases
                if phase["name"] == normalized_liquid
            )
            if not math.isclose(
                liquid_amount,
                1.0,
                rel_tol=0.0,
                abs_tol=PHASE_FRACTION_TOLERANCE,
            ):
                raise CalphadInputError(
                    "start_temperature_K must be a single-phase liquid equilibrium state"
                )

            timeout_stage = "solidification solver"
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                try:
                    raw_result = _simulate_scheil(
                        database,
                        normalized_components,
                        normalized_phases,
                        pycalphad_composition,
                        start_temperature_K=start_temperature,
                        step_temperature_K=step_temperature,
                        liquid_phase_name=normalized_liquid,
                        stop_liquid_fraction=stop_fraction,
                    )
                except (CalphadInputError, CalphadExecutionError):
                    raise
                except _WallTimeExceededError:
                    raise
                except AssertionError as exc:
                    raise CalphadExecutionError(
                        "Scheil solver returned internally inconsistent result lengths"
                    ) from exc
                except Exception as exc:
                    raise CalphadExecutionError("Scheil solver failed") from exc

            timeout_stage = "scientific result validation"
            result = _validated_scheil_result(
                raw_result,
                requested_phases=normalized_phases,
                physical_components=physical_components,
                bulk_composition=closure_records[0],
                liquid_phase_name=normalized_liquid,
                assessment_temperature_limits_K=assessment_limits,
                stop_liquid_fraction=stop_fraction,
                max_steps=step_limit,
            )
    except _WallTimeExceededError as exc:
        raise CalphadTimeoutError(
            f"Scheil {timeout_stage} exceeded the shared {wall_limit}-second "
            "end-to-end scientific wall-time limit"
        ) from exc
    runtime_warnings = {
        "This path is not a back-diffusion, finite-rate diffusion, precipitation, or phase-field calculation.",
        "A converged numerical path does not validate the thermodynamic assessment or extrapolation domain.",
    }
    if result["discarded_upstream_terminal_fill_point"]:
        runtime_warnings.add(
            "The scheil 0.3.0 same-temperature terminal fill point was discarded because its phase increments did not represent the already-satisfied residual-liquid criterion."
        )
    request_record = {
        "components": list(normalized_components),
        "phases": list(normalized_phases),
        "independent_composition_mole_fraction": {
            component: values[0] for component, values in composition_axes.items()
        },
        "bulk_composition_mole_fraction": closure_records[0],
        "dependent_component": dependent_component,
        "start_temperature_K": start_temperature,
        "step_temperature_K": step_temperature,
        "pressure_Pa": pressure,
        "total_amount_mol": 1.0,
        "liquid_phase_name": normalized_liquid,
        "stop_liquid_fraction": stop_fraction,
    }
    response: dict[str, Any] = {
        "schema_version": SCHEIL_SCHEMA_VERSION,
        "method": "Scheil-Gulliver",
        "database": manifest,
        "request": request_record,
        "result": result,
        "assumptions": list(SCHEIL_ASSUMPTIONS),
        "warnings": sorted(
            set(manifest["warnings"])
            | set(_warning_messages(captured))
            | runtime_warnings
            | set((manifest.get("registry_manifest") or {}).get("caveats") or [])
        ),
        "solver": {
            "name": "scheil",
            "version": QUALIFIED_SCHEIL_VERSION,
            "pycalphad_version": manifest["pycalphad_version"],
            "adaptive_constitution_sampling": True,
            "replay_determinism_claimed": False,
        },
        "units": {
            "temperature": "K",
            "pressure": "Pa",
            "amount": "mol",
            "composition": "mole_fraction",
            "phase_fraction": "fraction_of_one_mole_basis",
        },
        "limits": {
            "max_steps": step_limit,
            "wall_time_seconds": wall_limit,
            "wall_time_scope": "shared_liquid_preflight_validation_and_solidification_solve",
            "max_result_bytes": result_limit,
            "conservative_result_upper_bound_bytes": estimated_result_upper_bound,
        },
    }
    evidence_payload = dict(response)
    response["evidence"] = {
        "sha256": _sha256_bytes(_canonical_json_bytes(evidence_payload)),
        "algorithm": "sha256",
        "canonicalization": "UTF-8 JSON, sorted keys, compact separators, finite numbers",
    }
    encoded = _canonical_json_bytes(response)
    if len(encoded) > result_limit:
        raise CalphadExecutionError(
            f"serialized Scheil evidence exceeds max_result_bytes: {len(encoded)} > {result_limit}"
        )
    return response


def processing_method_support() -> dict[str, dict[str, Any]]:
    """Return the honest execution boundary for processing/kinetics methods."""

    return {
        "scheil_gulliver": {
            "status": "qualified_runtime",
            "solver": f"scheil=={QUALIFIED_SCHEIL_VERSION}",
            "assumptions": list(SCHEIL_ASSUMPTIONS),
        },
        "back_diffusion": {
            "status": "qualified_isolated_runtime",
            "solver": "kawin==0.5.0",
            "tool": "materials_run_diffusion_1d",
            "scope": "post_solidification_single_phase_1d_only",
            "required_evidence": [
                "diffusion/mobility data with units and provenance",
                "length scale or dendrite-arm-spacing model",
                "isothermal duration and zero-flux boundary applicability",
            ],
        },
        "mobility_diffusion": {
            "status": "qualified_isolated_runtime",
            "solver": "kawin==0.5.0",
            "tools": ["materials_transport_coefficients", "materials_run_diffusion_1d"],
            "scope": (
                "one selected phase at fixed 101325 Pa; MF/MQ multicomponent or binary "
                "DF/DQ transport; isothermal Cartesian 1-D zero-flux diffusion"
            ),
            "required_evidence": ["mobility or diffusivity database", "phase/frame definition"],
        },
        "precipitation": {
            "status": "qualified_isolated_runtime",
            "solver": "kawin==0.5.0",
            "tool": "materials_run_binary_precipitation_kwn",
            "scope": (
                "binary isothermal spherical KWN; one matrix and one precipitate; "
                "homogeneous bulk nucleation; fixed nonadaptive bins; infinite "
                "precipitate diffusion"
            ),
            "required_evidence": [
                "selected kinetic TDB",
                "sourced matrix and precipitate molar volumes",
                "sourced interfacial and elastic strain energies",
                "sourced homogeneous bulk nucleation-site density",
                "isothermal temperature and duration",
            ],
        },
        "phase_field": {
            "status": "requires_external_hpc_solver",
            "required_evidence": [
                "governing free-energy functional",
                "kinetic coefficients",
                "boundary/initial conditions",
                "mesh convergence and benchmark definition",
            ],
        },
    }
