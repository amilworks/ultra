"""Bounded Kawin 0.5 operations for the isolated NumPy-2 image."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import signal
import warnings
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .contract import (
    QUALIFIED_PRESSURE_PA,
    REQUEST_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    inspect_database,
    normalize_components,
    normalize_composition,
    normalize_limits,
    require_bool,
    require_float,
    require_int,
    require_keys,
    require_mapping,
    require_nonempty_string,
    require_numeric_sequence,
    require_temperature_in_assessment,
    require_token,
    select_transport_family,
)
from .errors import (
    KineticsExecutionError,
    KineticsInputError,
    KineticsTimeoutError,
    KineticsUnsupportedError,
)

QUALIFIED_VERSIONS = {
    "kawin": "0.5.0",
    "numpy": "2.4.6",
    "pycalphad": "0.11.2",
    "scipy": "1.17.1",
}
MAX_SOLVER_STEPS = 1_000_000


class _WallTimeExceededError(Exception):
    pass


class _StepLimitExceededError(Exception):
    pass


def _alarm_handler(_signum: int, _frame: Any) -> None:
    raise _WallTimeExceededError


@contextmanager
def _wall_time_limit(seconds: float):
    if not hasattr(signal, "setitimer"):
        raise KineticsUnsupportedError("the isolated kinetics runtime requires POSIX wall timers")
    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _alarm_handler)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise KineticsExecutionError(
            "result cannot be represented as finite canonical JSON"
        ) from exc
    return text.encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _runtime_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in QUALIFIED_VERSIONS}


def _require_runtime() -> dict[str, str]:
    actual = _runtime_versions()
    mismatches = [
        f"{name}=={actual[name]} (expected {expected})"
        for name, expected in QUALIFIED_VERSIONS.items()
        if actual[name] != expected
    ]
    if mismatches:
        raise KineticsUnsupportedError(
            "kinetics dependency versions are outside the qualified envelope: "
            + "; ".join(mismatches)
        )
    if int(actual["numpy"].split(".", maxsplit=1)[0]) != 2:
        raise KineticsUnsupportedError("Kawin execution must run in the isolated NumPy-2 image")
    return actual


def runtime_support() -> dict[str, Any]:
    """Return the exact execution boundary without relabeling unsupported models."""

    versions = _require_runtime()
    return {
        "schema_version": "ultra.materials.kinetics-runtime-support.v1",
        "runtime": {
            "name": "ultra-isolated-kawin",
            "versions": versions,
            "shared_numpy_1_26_sandbox_modified": False,
            "pressure_behavior": "Kawin 0.5 thermodynamic calls use a fixed 101325 Pa",
        },
        "operations": {
            "transport_coefficients": {
                "status": "executable",
                "scope": "single selected phase; MF/MQ multicomponent or binary DF/DQ",
            },
            "single_phase_diffusion_1d": {
                "status": "executable",
                "scope": "isothermal Cartesian finite-volume diffusion with zero-flux boundaries",
            },
            "post_solidification_back_diffusion_1d": {
                "status": "executable_limited",
                "scope": "same single-phase diffusion solver after solidification; no moving interface",
            },
            "binary_precipitation_kwn": {
                "status": "executable_limited",
                "scope": "isothermal spherical bulk-nucleated binary KWN with explicit parameters",
            },
            "coupled_solidification_back_diffusion": {
                "status": "unsupported",
                "reason": "no qualified moving-interface or Brody-Flemings/Clyne-Kurz solver",
            },
            "phase_field": {
                "status": "external_hpc_required",
                "reason": "requires a governed PDE/free-energy model, mesh and convergence study",
            },
            "phase_field_readiness": {
                "status": "contract_validation_only",
                "scope": "validate a provenance-bound external-solver submission contract; no PDE execution",
            },
        },
    }


def _warnings(captured: Sequence[warnings.WarningMessage]) -> list[str]:
    return sorted({f"{item.category.__name__}: {item.message}" for item in captured})


def _as_finite_array(value: Any, *, field: str, ndim: int | None = None):
    import numpy as np

    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise KineticsExecutionError(f"solver result {field} is not numeric") from exc
    if ndim is not None and array.ndim != ndim:
        raise KineticsExecutionError(f"solver result {field} has the wrong rank")
    if not np.all(np.isfinite(array)):
        raise KineticsExecutionError(f"solver result {field} contains non-finite values")
    return array


def _pressure(value: Any) -> float:
    pressure = require_float(
        value,
        field="pressure_Pa",
        minimum=1e-9,
        maximum=1e12,
    )
    if pressure != QUALIFIED_PRESSURE_PA:
        raise KineticsInputError(
            f"qualified Kawin execution requires pressure_Pa={QUALIFIED_PRESSURE_PA} exactly"
        )
    return pressure


def _finish_response(response: dict[str, Any], *, max_result_bytes: int) -> dict[str, Any]:
    evidence_payload = dict(response)
    response["evidence"] = {
        "algorithm": "sha256",
        "canonicalization": "UTF-8 JSON, sorted keys, compact separators, finite numbers",
        "sha256": _sha256(_canonical_json_bytes(evidence_payload)),
    }
    encoded = _canonical_json_bytes(response)
    if len(encoded) > max_result_bytes:
        raise KineticsExecutionError(
            f"serialized evidence exceeds max_result_bytes: {len(encoded)} > {max_result_bytes}"
        )
    return response


def _base_request(value: Any, *, operation: str) -> Mapping[str, Any]:
    request = require_mapping(value, field="request")
    schema = require_nonempty_string(
        request.get("schema_version"), field="schema_version", maximum=128
    )
    if schema != REQUEST_SCHEMA_VERSION:
        raise KineticsInputError(f"schema_version must be {REQUEST_SCHEMA_VERSION!r}")
    actual_operation = require_nonempty_string(
        request.get("operation"), field="operation", maximum=128
    )
    if actual_operation != operation:
        raise KineticsInputError(f"operation must be {operation!r}")
    return request


def _transport_coefficients(
    value: Any,
    *,
    workspace_root: Path,
    versions: Mapping[str, str],
) -> dict[str, Any]:
    operation = "transport_coefficients"
    request = _base_request(value, operation=operation)
    require_keys(
        request,
        field="request",
        required={
            "schema_version",
            "operation",
            "database",
            "components",
            "phase",
            "independent_composition_mole_fraction",
            "temperature_K",
            "pressure_Pa",
            "limits",
        },
    )
    components = normalize_components(request["components"])
    phase = require_token(request["phase"], field="phase")
    independent, full = normalize_composition(
        request["independent_composition_mole_fraction"], components=components
    )
    temperature = require_float(
        request["temperature_K"], field="temperature_K", minimum=1.0, maximum=10_000.0
    )
    pressure = _pressure(request["pressure_Pa"])
    limits = normalize_limits(request["limits"])
    inspected = inspect_database(
        request["database"],
        workspace_root=workspace_root,
        components=components,
        phases=[phase],
    )
    require_temperature_in_assessment(temperature, inspected.manifest)
    family = select_transport_family(
        inspected.kinetic_inventory[phase],
        components=components,
        binary_solute_only_ok=True,
    )

    from kawin.thermo import GeneralThermodynamics

    composition_arg: float | list[float]
    if len(components) == 2:
        composition_arg = independent[components[1]]
    else:
        composition_arg = [independent[component] for component in components[1:]]
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            with _wall_time_limit(limits["wall_time_seconds"]):
                thermodynamics = GeneralThermodynamics(
                    inspected.database,
                    list(components),
                    [phase],
                )
                tracer_raw = thermodynamics.getTracerDiffusivity(
                    composition_arg,
                    temperature,
                    phase=phase,
                )
                interdiff_raw = thermodynamics.getInterdiffusivity(
                    composition_arg,
                    temperature,
                    phase=phase,
                )
    except _WallTimeExceededError as exc:
        raise KineticsTimeoutError("transport coefficient calculation exceeded wall time") from exc
    except (KineticsInputError, KineticsUnsupportedError, KineticsExecutionError):
        raise
    except Exception as exc:
        raise KineticsExecutionError("Kawin transport coefficient calculation failed") from exc

    tracer = _as_finite_array(tracer_raw, field="tracer_diffusivity").reshape(-1)
    if tracer.size != len(components):
        raise KineticsExecutionError("Kawin returned the wrong tracer-diffusivity component count")
    reported_components: tuple[str, ...]
    if family == "DF/DQ direct diffusivity":
        reported_components = (components[1],)
    else:
        reported_components = components
    tracer_result = {
        component: float(tracer[components.index(component)]) for component in reported_components
    }
    if any(value <= 0 for value in tracer_result.values()):
        raise KineticsExecutionError("Kawin returned a non-positive assessed tracer diffusivity")

    interdiff = _as_finite_array(interdiff_raw, field="interdiffusivity")
    expected = len(components) - 1
    if expected == 1:
        interdiff = interdiff.reshape(1, 1)
    if interdiff.shape != (expected, expected):
        raise KineticsExecutionError("Kawin returned the wrong interdiffusivity matrix shape")

    runtime_warnings = _warnings(captured)
    runtime_warnings.append(
        "The selected phase is constrained for coefficient evaluation; global phase stability was not assessed."
    )
    if family == "DF/DQ direct diffusivity":
        runtime_warnings.append(
            "Direct DF/DQ supports this binary solute coefficient only; no multicomponent cross-diffusion claim is made."
        )
    response = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "operation": operation,
        "method": "Kawin CALPHAD transport coefficients",
        "database": inspected.manifest,
        "kinetic_parameter_inventory": inspected.kinetic_inventory,
        "request": {
            "components": list(components),
            "reference_component": components[0],
            "phase": phase,
            "bulk_composition_mole_fraction": full,
            "temperature_K": temperature,
            "pressure_Pa": pressure,
        },
        "result": {
            "tracer_diffusivity_m2_per_s": tracer_result,
            "interdiffusivity_m2_per_s": interdiff.tolist(),
            "interdiffusivity_rows": list(components[1:]),
            "interdiffusivity_columns": list(components[1:]),
            "reference_component": components[0],
            "transport_parameter_family_used": family,
            "cross_diffusion_supported": family == "MF/MQ mobility",
            "flux_frame": "volume-fixed",
        },
        "solver": {
            "name": "kawin",
            "versions": dict(versions),
            "phase_constraint": "single selected phase; metastable evaluation allowed",
            "global_phase_stability_assessed": False,
            "pressure_behavior": "Kawin 0.5 thermodynamic calls use a fixed 101325 Pa; pressure was not swept or exposed as a backend solver variable.",
        },
        "scientific_status": "numerically_executed_not_experimentally_validated",
        "warnings": sorted(set(runtime_warnings)),
        "limits": limits,
    }
    return _finish_response(response, max_result_bytes=limits["max_result_bytes"])


def _diffusion_profile(value: Any, *, components: Sequence[str]) -> dict[str, Any]:
    profile = require_mapping(value, field="initial_profile")
    require_keys(
        profile,
        field="initial_profile",
        required={
            "coordinates_m",
            "independent_composition_mole_fraction",
            "interpolation",
            "source",
        },
    )
    coordinates = require_numeric_sequence(
        profile["coordinates_m"],
        field="initial_profile.coordinates_m",
        minimum_length=2,
        maximum_length=1024,
        minimum=-1e3,
        maximum=1e3,
    )
    if any(right <= left for left, right in zip(coordinates, coordinates[1:])):
        raise KineticsInputError("initial_profile.coordinates_m must be strictly increasing")
    raw_compositions = require_mapping(
        profile["independent_composition_mole_fraction"],
        field="initial_profile.independent_composition_mole_fraction",
    )
    independent_components = tuple(components[1:])
    if set(raw_compositions) != set(independent_components):
        raise KineticsInputError(
            "initial_profile compositions must contain exactly every independent component"
        )
    compositions: dict[str, list[float]] = {}
    for component in independent_components:
        values = require_numeric_sequence(
            raw_compositions[component],
            field=f"initial_profile.independent_composition_mole_fraction.{component}",
            minimum_length=len(coordinates),
            maximum_length=len(coordinates),
            minimum=1e-10,
            maximum=1.0 - 1e-10,
        )
        compositions[component] = values
    for index in range(len(coordinates)):
        independent_sum = math.fsum(compositions[item][index] for item in independent_components)
        if independent_sum > 1.0 - 1e-10:
            raise KineticsInputError(
                f"initial_profile leaves too little reference component at index {index}"
            )
    interpolation = require_nonempty_string(
        profile["interpolation"], field="initial_profile.interpolation", maximum=64
    )
    if interpolation != "linear":
        raise KineticsUnsupportedError("only explicit linear profile interpolation is qualified")
    return {
        "coordinates_m": coordinates,
        "independent_composition_mole_fraction": compositions,
        "interpolation": interpolation,
        "source": require_nonempty_string(profile["source"], field="initial_profile.source"),
    }


def _diffusion_application(value: Any) -> dict[str, str]:
    application = require_mapping(value, field="application")
    kind = require_nonempty_string(application.get("kind"), field="application.kind", maximum=128)
    if kind == "generic_single_phase_diffusion":
        require_keys(application, field="application", required={"kind"})
        return {"kind": kind}
    if kind == "post_solidification_back_diffusion":
        require_keys(
            application,
            field="application",
            required={"kind", "length_scale_source", "solidification_coupling"},
        )
        coupling = require_nonempty_string(
            application["solidification_coupling"],
            field="application.solidification_coupling",
            maximum=128,
        )
        if coupling != "post_solidification_only":
            raise KineticsUnsupportedError(
                "this runtime supports back diffusion only after solidification, without a moving interface"
            )
        return {
            "kind": kind,
            "length_scale_source": require_nonempty_string(
                application["length_scale_source"], field="application.length_scale_source"
            ),
            "solidification_coupling": coupling,
        }
    raise KineticsUnsupportedError(
        "application.kind must be generic_single_phase_diffusion or "
        "post_solidification_back_diffusion"
    )


def _validate_composition_array(array: Any, *, components: Sequence[str], field: str):
    import numpy as np

    result = _as_finite_array(array, field=field, ndim=2)
    if result.shape[1] != len(components):
        raise KineticsExecutionError(f"solver result {field} has the wrong component count")
    if np.any(result < -1e-12) or np.any(result > 1.0 + 1e-12):
        raise KineticsExecutionError(f"solver result {field} leaves [0, 1]")
    closure = np.sum(result, axis=1)
    if not np.allclose(closure, 1.0, rtol=0.0, atol=1e-10):
        raise KineticsExecutionError(f"solver result {field} compositions do not close")
    return np.clip(result, 0.0, 1.0)


def _single_phase_diffusion_1d(
    value: Any,
    *,
    workspace_root: Path,
    versions: Mapping[str, str],
) -> dict[str, Any]:
    operation = "single_phase_diffusion_1d"
    request = _base_request(value, operation=operation)
    require_keys(
        request,
        field="request",
        required={
            "schema_version",
            "operation",
            "database",
            "components",
            "phase",
            "temperature_K",
            "pressure_Pa",
            "duration_s",
            "domain_m",
            "mesh_cells",
            "max_solver_steps",
            "boundary_condition",
            "initial_profile",
            "application",
            "limits",
        },
    )
    components = normalize_components(request["components"])
    phase = require_token(request["phase"], field="phase")
    temperature = require_float(
        request["temperature_K"], field="temperature_K", minimum=1.0, maximum=10_000.0
    )
    pressure = _pressure(request["pressure_Pa"])
    duration = require_float(
        request["duration_s"],
        field="duration_s",
        minimum=1e-12,
        maximum=1e15,
        include_minimum=False,
    )
    domain = require_numeric_sequence(
        request["domain_m"],
        field="domain_m",
        minimum_length=2,
        maximum_length=2,
        minimum=-1e3,
        maximum=1e3,
    )
    if domain[0] >= domain[1]:
        raise KineticsInputError("domain_m must be strictly increasing")
    mesh_cells = require_int(request["mesh_cells"], field="mesh_cells", minimum=8, maximum=512)
    max_steps = require_int(
        request["max_solver_steps"],
        field="max_solver_steps",
        minimum=1,
        maximum=MAX_SOLVER_STEPS,
    )
    boundary = require_mapping(request["boundary_condition"], field="boundary_condition")
    require_keys(boundary, field="boundary_condition", required={"kind"})
    boundary_kind = require_nonempty_string(
        boundary["kind"], field="boundary_condition.kind", maximum=64
    )
    if boundary_kind != "zero_flux":
        raise KineticsUnsupportedError("only zero_flux boundaries are qualified")
    profile = _diffusion_profile(request["initial_profile"], components=components)
    if profile["coordinates_m"][0] != domain[0] or profile["coordinates_m"][-1] != domain[1]:
        raise KineticsInputError("initial_profile coordinates must exactly span domain_m")
    application = _diffusion_application(request["application"])
    limits = normalize_limits(request["limits"])
    inspected = inspect_database(
        request["database"],
        workspace_root=workspace_root,
        components=components,
        phases=[phase],
    )
    require_temperature_in_assessment(temperature, inspected.manifest)
    family = select_transport_family(
        inspected.kinetic_inventory[phase],
        components=components,
        binary_solute_only_ok=True,
    )

    import numpy as np
    from kawin.diffusion import SinglePhaseModel, TemperatureParameters
    from kawin.diffusion.DiffusionParameters import DiffusionConstraints
    from kawin.diffusion.mesh import Cartesian1D, ExperimentalProfile1D, ProfileBuilder
    from kawin.thermo import GeneralThermodynamics

    class BoundedSinglePhaseModel(SinglePhaseModel):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.ultra_step_count = 0

        def postProcess(self, time: float, x: Any):  # noqa: N802 - upstream API
            self.ultra_step_count += 1
            if self.ultra_step_count > max_steps:
                raise _StepLimitExceededError
            return super().postProcess(time, x)

    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            with _wall_time_limit(limits["wall_time_seconds"]):
                thermodynamics = GeneralThermodynamics(
                    inspected.database,
                    list(components),
                    [phase],
                )
                profile_builder = ProfileBuilder()
                values = np.column_stack(
                    [
                        profile["independent_composition_mole_fraction"][component]
                        for component in components[1:]
                    ]
                )
                profile_builder.addBuildStep(
                    ExperimentalProfile1D(profile["coordinates_m"], values),
                    list(components[1:]),
                )
                mesh = Cartesian1D(list(components[1:]), domain, mesh_cells)
                mesh.setResponseProfile(profile_builder)
                constraints = DiffusionConstraints()
                constraints.minComposition = 0.0
                model = BoundedSinglePhaseModel(
                    mesh,
                    list(components),
                    [phase],
                    thermodynamics,
                    TemperatureParameters(temperature),
                    constraints=constraints,
                    record=False,
                )
                initial = np.array(model.getCompositions(), dtype=np.float64, copy=True)
                model.solve(duration, verbose=False)
                final = np.array(model.getCompositions(), dtype=np.float64, copy=True)
    except _WallTimeExceededError as exc:
        raise KineticsTimeoutError("single-phase diffusion exceeded wall time") from exc
    except _StepLimitExceededError as exc:
        raise KineticsExecutionError("single-phase diffusion exceeded max_solver_steps") from exc
    except (KineticsInputError, KineticsUnsupportedError, KineticsExecutionError):
        raise
    except Exception as exc:
        raise KineticsExecutionError("Kawin single-phase diffusion failed") from exc

    initial = _validate_composition_array(initial, components=components, field="initial_profile")
    final = _validate_composition_array(final, components=components, field="final_profile")
    if initial.shape != (mesh_cells, len(components)) or final.shape != initial.shape:
        raise KineticsExecutionError("Kawin returned the wrong diffusion profile shape")
    if not math.isclose(float(model.currentTime), duration, rel_tol=1e-12, abs_tol=1e-12):
        raise KineticsExecutionError("Kawin diffusion did not reach the requested duration")
    initial_mean = np.mean(initial, axis=0)
    final_mean = np.mean(final, axis=0)
    mass_error = np.abs(final_mean - initial_mean)
    if np.any(mass_error > 1e-8):
        raise KineticsExecutionError("zero-flux diffusion failed component mass closure")
    coordinates = _as_finite_array(mesh.z, field="mesh_coordinates", ndim=2).reshape(-1)
    if len(coordinates) != mesh_cells or np.any(np.diff(coordinates) <= 0):
        raise KineticsExecutionError("Kawin returned invalid mesh coordinates")

    runtime_warnings = _warnings(captured)
    runtime_warnings.extend(
        [
            "The selected phase is constrained throughout; global phase stability and transformations were not assessed.",
            "A single mesh is numerical execution evidence, not a mesh-convergence study.",
        ]
    )
    if application["kind"] == "post_solidification_back_diffusion":
        runtime_warnings.append(
            "Back diffusion here means post-solidification single-phase diffusion only; no moving solid/liquid interface or coupled solidification model was solved."
        )
    response = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "operation": operation,
        "method": "Kawin single-phase 1D finite-volume diffusion",
        "application": application,
        "database": inspected.manifest,
        "kinetic_parameter_inventory": inspected.kinetic_inventory,
        "request": {
            "components": list(components),
            "reference_component": components[0],
            "phase": phase,
            "temperature_K": temperature,
            "pressure_Pa": pressure,
            "duration_s": duration,
            "domain_m": domain,
            "mesh_cells": mesh_cells,
            "boundary_condition": "zero_flux",
            "initial_profile": profile,
        },
        "result": {
            "coordinates_m": coordinates.tolist(),
            "initial_composition_mole_fraction": {
                component: initial[:, index].tolist() for index, component in enumerate(components)
            },
            "final_composition_mole_fraction": {
                component: final[:, index].tolist() for index, component in enumerate(components)
            },
            "time_s": float(model.currentTime),
            "solver_steps": int(model.ultra_step_count),
            "transport_parameter_family_used": family,
            "flux_frame": "volume-fixed",
            "numerical_verification": {
                "initial_spatial_mean_mole_fraction": {
                    component: float(initial_mean[index])
                    for index, component in enumerate(components)
                },
                "final_spatial_mean_mole_fraction": {
                    component: float(final_mean[index])
                    for index, component in enumerate(components)
                },
                "absolute_mass_closure_error": {
                    component: float(mass_error[index])
                    for index, component in enumerate(components)
                },
                "mass_closure_tolerance": 1e-8,
                "grid_convergence_assessed": False,
            },
        },
        "solver": {
            "name": "kawin",
            "versions": dict(versions),
            "geometry": "Cartesian 1D",
            "spatial_discretization": "finite volume",
            "phase_constraint": "single selected phase; metastable evaluation allowed",
            "global_phase_stability_assessed": False,
            "pressure_behavior": "Kawin 0.5 thermodynamic calls use a fixed 101325 Pa; pressure was not swept or exposed as a backend solver variable.",
        },
        "scientific_status": "numerically_executed_requires_grid_and_experimental_validation",
        "warnings": sorted(set(runtime_warnings)),
        "limits": {**limits, "max_solver_steps": max_steps},
    }
    return _finish_response(response, max_result_bytes=limits["max_result_bytes"])


def _precipitation_matrix(value: Any) -> dict[str, Any]:
    matrix = require_mapping(value, field="matrix")
    require_keys(
        matrix,
        field="matrix",
        required={
            "molar_volume_m3_per_mol",
            "atoms_per_unit_cell",
            "bulk_nucleation_site_density_per_m3",
            "grain_boundary_energy_J_per_m2",
            "source",
        },
    )
    return {
        "molar_volume_m3_per_mol": require_float(
            matrix["molar_volume_m3_per_mol"],
            field="matrix.molar_volume_m3_per_mol",
            minimum=1e-8,
            maximum=1e-2,
            include_minimum=False,
        ),
        "atoms_per_unit_cell": require_int(
            matrix["atoms_per_unit_cell"],
            field="matrix.atoms_per_unit_cell",
            minimum=1,
            maximum=10_000,
        ),
        "bulk_nucleation_site_density_per_m3": require_float(
            matrix["bulk_nucleation_site_density_per_m3"],
            field="matrix.bulk_nucleation_site_density_per_m3",
            minimum=1.0,
            maximum=1e40,
            include_minimum=False,
        ),
        "grain_boundary_energy_J_per_m2": require_float(
            matrix["grain_boundary_energy_J_per_m2"],
            field="matrix.grain_boundary_energy_J_per_m2",
            minimum=0.0,
            maximum=100.0,
        ),
        "source": require_nonempty_string(matrix["source"], field="matrix.source"),
    }


def _precipitation_phase(value: Any) -> dict[str, Any]:
    precipitate = require_mapping(value, field="precipitate")
    require_keys(
        precipitate,
        field="precipitate",
        required={
            "molar_volume_m3_per_mol",
            "atoms_per_unit_cell",
            "interfacial_energy_J_per_m2",
            "constant_elastic_strain_energy_J_per_m3",
            "infinite_precipitate_diffusion",
            "source",
        },
    )
    infinite_diffusion = require_bool(
        precipitate["infinite_precipitate_diffusion"],
        field="precipitate.infinite_precipitate_diffusion",
    )
    if not infinite_diffusion:
        raise KineticsUnsupportedError(
            "the qualified binary KWN contract currently requires infinite precipitate diffusion"
        )
    return {
        "molar_volume_m3_per_mol": require_float(
            precipitate["molar_volume_m3_per_mol"],
            field="precipitate.molar_volume_m3_per_mol",
            minimum=1e-8,
            maximum=1e-2,
            include_minimum=False,
        ),
        "atoms_per_unit_cell": require_int(
            precipitate["atoms_per_unit_cell"],
            field="precipitate.atoms_per_unit_cell",
            minimum=1,
            maximum=10_000,
        ),
        "interfacial_energy_J_per_m2": require_float(
            precipitate["interfacial_energy_J_per_m2"],
            field="precipitate.interfacial_energy_J_per_m2",
            minimum=1e-12,
            maximum=100.0,
            include_minimum=False,
        ),
        "constant_elastic_strain_energy_J_per_m3": require_float(
            precipitate["constant_elastic_strain_energy_J_per_m3"],
            field="precipitate.constant_elastic_strain_energy_J_per_m3",
            minimum=0.0,
            maximum=1e13,
        ),
        "infinite_precipitate_diffusion": True,
        "source": require_nonempty_string(precipitate["source"], field="precipitate.source"),
    }


def _population_balance(value: Any) -> dict[str, Any]:
    population = require_mapping(value, field="population_balance")
    require_keys(
        population,
        field="population_balance",
        required={"min_radius_m", "max_radius_m", "bins", "adaptive", "max_history_points"},
    )
    minimum = require_float(
        population["min_radius_m"],
        field="population_balance.min_radius_m",
        minimum=1e-12,
        maximum=1e-3,
        include_minimum=False,
    )
    maximum = require_float(
        population["max_radius_m"],
        field="population_balance.max_radius_m",
        minimum=minimum,
        maximum=1e-2,
        include_minimum=False,
    )
    adaptive = require_bool(population["adaptive"], field="population_balance.adaptive")
    if adaptive:
        raise KineticsUnsupportedError(
            "adaptive population-balance bin resizing is outside the qualified deterministic contract"
        )
    return {
        "min_radius_m": minimum,
        "max_radius_m": maximum,
        "bins": require_int(
            population["bins"], field="population_balance.bins", minimum=25, maximum=400
        ),
        "adaptive": False,
        "max_history_points": require_int(
            population["max_history_points"],
            field="population_balance.max_history_points",
            minimum=8,
            maximum=2048,
        ),
    }


def _representative_indices(length: int, maximum: int, arrays: Sequence[Any]) -> list[int]:
    import numpy as np

    if length <= maximum:
        return list(range(length))
    selected = {0, length - 1}
    for array in arrays:
        selected.add(int(np.argmin(array)))
        selected.add(int(np.argmax(array)))
    if len(selected) > maximum:
        return sorted(selected)[: maximum - 1] + [length - 1]
    for index in np.linspace(0, length - 1, maximum, dtype=np.int64):
        selected.add(int(index))
        if len(selected) >= maximum:
            break
    if len(selected) < maximum:
        for index in range(length):
            selected.add(index)
            if len(selected) >= maximum:
                break
    return sorted(selected)


def _reconstruct_binary_bulk_solute(
    matrix_composition: Any,
    precipitate_volume_fraction: Any,
    fraction_weighted_precipitate_solute: Any,
):
    """Apply Kawin KWNEuler's binary solute balance exactly.

    Kawin defines ``fconc`` as the already fraction-weighted precipitate solute
    contribution, so it must not be multiplied by volume fraction a second time:
    ``x0 = (1 - f_v) * x_matrix + fconc``.
    """

    return (
        1.0 - precipitate_volume_fraction
    ) * matrix_composition + fraction_weighted_precipitate_solute


def _binary_precipitation_kwn(
    value: Any,
    *,
    workspace_root: Path,
    versions: Mapping[str, str],
) -> dict[str, Any]:
    operation = "binary_precipitation_kwn"
    request = _base_request(value, operation=operation)
    require_keys(
        request,
        field="request",
        required={
            "schema_version",
            "operation",
            "database",
            "components",
            "matrix_phase",
            "precipitate_phase",
            "initial_solute_mole_fraction",
            "temperature_K",
            "temperature_source",
            "pressure_Pa",
            "duration_s",
            "driving_force_method",
            "matrix",
            "precipitate",
            "nucleation",
            "population_balance",
            "max_solver_steps",
            "limits",
        },
    )
    components = normalize_components(request["components"])
    if len(components) != 2:
        raise KineticsUnsupportedError("binary_precipitation_kwn requires exactly two components")
    matrix_phase = require_token(request["matrix_phase"], field="matrix_phase")
    precipitate_phase = require_token(request["precipitate_phase"], field="precipitate_phase")
    if matrix_phase == precipitate_phase:
        raise KineticsInputError("matrix_phase and precipitate_phase must differ")
    initial_solute = require_float(
        request["initial_solute_mole_fraction"],
        field="initial_solute_mole_fraction",
        minimum=1e-10,
        maximum=1.0 - 1e-10,
    )
    temperature = require_float(
        request["temperature_K"], field="temperature_K", minimum=1.0, maximum=10_000.0
    )
    temperature_source = require_nonempty_string(
        request["temperature_source"], field="temperature_source"
    )
    pressure = _pressure(request["pressure_Pa"])
    duration = require_float(
        request["duration_s"],
        field="duration_s",
        minimum=1e-12,
        maximum=1e15,
        include_minimum=False,
    )
    driving_force_method = require_nonempty_string(
        request["driving_force_method"], field="driving_force_method", maximum=64
    )
    if driving_force_method != "tangent":
        raise KineticsUnsupportedError("only Kawin's tangent driving-force method is qualified")
    matrix = _precipitation_matrix(request["matrix"])
    precipitate = _precipitation_phase(request["precipitate"])
    nucleation = require_mapping(request["nucleation"], field="nucleation")
    require_keys(nucleation, field="nucleation", required={"site", "source"})
    nucleation_site = require_nonempty_string(
        nucleation["site"], field="nucleation.site", maximum=64
    )
    if nucleation_site != "bulk":
        raise KineticsUnsupportedError("only explicit homogeneous bulk nucleation is qualified")
    nucleation_source = require_nonempty_string(nucleation["source"], field="nucleation.source")
    population = _population_balance(request["population_balance"])
    max_steps = require_int(
        request["max_solver_steps"],
        field="max_solver_steps",
        minimum=1,
        maximum=MAX_SOLVER_STEPS,
    )
    limits = normalize_limits(request["limits"])
    inspected = inspect_database(
        request["database"],
        workspace_root=workspace_root,
        components=components,
        phases=[matrix_phase, precipitate_phase],
    )
    require_temperature_in_assessment(temperature, inspected.manifest)
    family = select_transport_family(
        inspected.kinetic_inventory[matrix_phase],
        components=components,
        binary_solute_only_ok=True,
    )

    import numpy as np
    from kawin.precipitation import (
        MatrixParameters,
        PrecipitateModel,
        PrecipitateParameters,
        VolumeParameter,
    )
    from kawin.thermo import BinaryThermodynamics

    class BoundedPrecipitateModel(PrecipitateModel):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.ultra_step_count = 0

        def postProcess(self, time: float, x: Any):  # noqa: N802 - upstream API
            self.ultra_step_count += 1
            if self.ultra_step_count > max_steps:
                raise _StepLimitExceededError
            return super().postProcess(time, x)

    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            with _wall_time_limit(limits["wall_time_seconds"]):
                thermodynamics = BinaryThermodynamics(
                    inspected.database,
                    list(components),
                    [matrix_phase, precipitate_phase],
                    drivingForceMethod=driving_force_method,
                )
                matrix_parameters = MatrixParameters([components[1]])
                matrix_parameters.initComposition = initial_solute
                matrix_parameters.volume.setVolume(
                    matrix["molar_volume_m3_per_mol"],
                    VolumeParameter.MOLAR_VOLUME,
                    matrix["atoms_per_unit_cell"],
                )
                matrix_parameters.nucleationSites.setBulkDensity(
                    matrix["bulk_nucleation_site_density_per_m3"]
                )
                matrix_parameters.GBenergy = matrix["grain_boundary_energy_J_per_m2"]

                precipitate_parameters = PrecipitateParameters(precipitate_phase)
                precipitate_parameters.gamma = precipitate["interfacial_energy_J_per_m2"]
                precipitate_parameters.volume.setVolume(
                    precipitate["molar_volume_m3_per_mol"],
                    VolumeParameter.MOLAR_VOLUME,
                    precipitate["atoms_per_unit_cell"],
                )
                precipitate_parameters.nucleation.setNucleationType("bulk")
                precipitate_parameters.shapeFactor.setSpherical(ar=1)
                precipitate_parameters.strainEnergy.setConstantElasticEnergy(
                    precipitate["constant_elastic_strain_energy_J_per_m3"]
                )
                precipitate_parameters.infinitePrecipitateDiffusion = True

                model = BoundedPrecipitateModel(
                    matrix_parameters,
                    precipitate_parameters,
                    thermodynamics,
                    temperature,
                )
                model.setPBMParameters(
                    cMin=population["min_radius_m"],
                    cMax=population["max_radius_m"],
                    bins=population["bins"],
                    minBins=population["bins"],
                    maxBins=population["bins"],
                    adaptive=False,
                    phase=precipitate_phase,
                )
                model.solve(duration, verbose=False)
    except _WallTimeExceededError as exc:
        raise KineticsTimeoutError("binary KWN precipitation exceeded wall time") from exc
    except _StepLimitExceededError as exc:
        raise KineticsExecutionError("binary KWN precipitation exceeded max_solver_steps") from exc
    except (KineticsInputError, KineticsUnsupportedError, KineticsExecutionError):
        raise
    except Exception as exc:
        raise KineticsExecutionError("Kawin binary KWN precipitation failed") from exc

    data = model.data
    time = _as_finite_array(data.time, field="time", ndim=1)
    matrix_composition = _as_finite_array(
        data.composition[:, 0], field="matrix_composition", ndim=1
    )
    volume_fraction = _as_finite_array(data.volFrac[:, 0], field="volume_fraction", ndim=1)
    average_radius = _as_finite_array(data.Ravg[:, 0], field="average_radius", ndim=1)
    precipitate_density = _as_finite_array(
        data.precipitateDensity[:, 0], field="precipitate_density", ndim=1
    )
    nucleation_rate = _as_finite_array(data.nucRate[:, 0], field="nucleation_rate", ndim=1)
    driving_force = _as_finite_array(data.drivingForce[:, 0], field="driving_force", ndim=1)
    temperatures = _as_finite_array(data.temperature, field="temperature", ndim=1)
    history_length = len(time)
    if not all(
        len(item) == history_length
        for item in (
            matrix_composition,
            volume_fraction,
            average_radius,
            precipitate_density,
            nucleation_rate,
            driving_force,
            temperatures,
        )
    ):
        raise KineticsExecutionError("Kawin precipitation history lengths disagree")
    if history_length != model.ultra_step_count + 1 or history_length > max_steps + 1:
        raise KineticsExecutionError("Kawin precipitation step accounting is inconsistent")
    if history_length < 2 or np.any(np.diff(time) <= 0):
        raise KineticsExecutionError("Kawin precipitation time is not strictly increasing")
    if not math.isclose(float(time[-1]), duration, rel_tol=1e-12, abs_tol=1e-12):
        raise KineticsExecutionError("Kawin precipitation did not reach the requested duration")
    if not np.allclose(temperatures, temperature, rtol=0.0, atol=1e-10):
        raise KineticsExecutionError("Kawin precipitation left the isothermal contract")
    if np.any(matrix_composition < -1e-12) or np.any(matrix_composition > 1.0 + 1e-12):
        raise KineticsExecutionError("Kawin returned invalid matrix composition")
    if np.any(volume_fraction < -1e-12) or np.any(volume_fraction >= 1.0 - 1e-12):
        raise KineticsExecutionError("Kawin returned invalid or fully transformed volume fraction")
    for array, field in (
        (average_radius, "average_radius"),
        (precipitate_density, "precipitate_density"),
        (nucleation_rate, "nucleation_rate"),
    ):
        if np.any(array < -1e-20):
            raise KineticsExecutionError(f"Kawin returned negative {field}")
    fconc = _as_finite_array(data.fconc[:, :, 0], field="precipitate_solute_content", ndim=2)
    if fconc.shape != (history_length, 1) or np.any(fconc < -1e-12):
        raise KineticsExecutionError("Kawin returned invalid precipitate solute content")
    reconstructed = _reconstruct_binary_bulk_solute(
        matrix_composition,
        volume_fraction,
        fconc[:, 0],
    )
    mass_error = np.abs(reconstructed - initial_solute)
    if np.any(mass_error > 1e-8):
        raise KineticsExecutionError("binary KWN precipitation failed solute mass closure")

    pbm = model.getPBM(precipitate_phase)
    particle_radii = _as_finite_array(pbm.PSDsize, field="particle_radius", ndim=1)
    particle_density_per_bin = _as_finite_array(
        pbm.PSD, field="particle_number_density_per_bin", ndim=1
    )
    if len(particle_radii) != len(particle_density_per_bin):
        raise KineticsExecutionError("Kawin returned inconsistent final particle-size bins")
    if np.any(np.diff(particle_radii) <= 0) or np.any(particle_density_per_bin < -1e-20):
        raise KineticsExecutionError("Kawin returned an invalid final particle-size distribution")

    indices = _representative_indices(
        history_length,
        population["max_history_points"],
        [
            matrix_composition,
            volume_fraction,
            average_radius,
            precipitate_density,
            nucleation_rate,
            driving_force,
        ],
    )
    runtime_warnings = _warnings(captured)
    runtime_warnings.extend(
        [
            "The KWN result assumes spherical precipitates, homogeneous bulk nucleation, constant elastic strain-energy density, and infinite diffusion within precipitates.",
            "Only the declared matrix/precipitate pair is modeled; competing phases and global phase stability were not assessed.",
            "A single particle-size grid is numerical execution evidence, not a bin-convergence study.",
            "A numerically closed KWN result does not validate interfacial energy, nucleation-site density, mobility assessment, or extrapolation against experiment.",
        ]
    )
    response = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "operation": operation,
        "method": "Kawin binary Kampmann-Wagner Numerical precipitation",
        "database": inspected.manifest,
        "kinetic_parameter_inventory": inspected.kinetic_inventory,
        "request": {
            "components": list(components),
            "reference_component": components[0],
            "solute_component": components[1],
            "matrix_phase": matrix_phase,
            "precipitate_phase": precipitate_phase,
            "initial_bulk_composition_mole_fraction": {
                components[0]: 1.0 - initial_solute,
                components[1]: initial_solute,
            },
            "temperature_K": temperature,
            "temperature_source": temperature_source,
            "pressure_Pa": pressure,
            "duration_s": duration,
            "driving_force_method": driving_force_method,
            "matrix": matrix,
            "precipitate": precipitate,
            "nucleation": {"site": nucleation_site, "source": nucleation_source},
            "population_balance": population,
        },
        "result": {
            "final": {
                "time_s": float(time[-1]),
                "matrix_solute_mole_fraction": float(matrix_composition[-1]),
                "precipitate_volume_fraction": float(volume_fraction[-1]),
                "average_equivalent_spherical_radius_m": float(average_radius[-1]),
                "precipitate_number_density_per_m3": float(precipitate_density[-1]),
                "nucleation_rate_per_m3_s": float(nucleation_rate[-1]),
                "driving_force_J_per_m3": float(driving_force[-1]),
                "reconstructed_bulk_solute_mole_fraction": float(reconstructed[-1]),
            },
            "history": {
                "source_point_count": history_length,
                "retained_point_count": len(indices),
                "retention": "endpoints, extrema of reported series, then uniform index fill",
                "time_s": time[indices].tolist(),
                "temperature_K": temperatures[indices].tolist(),
                "matrix_solute_mole_fraction": matrix_composition[indices].tolist(),
                "precipitate_volume_fraction": volume_fraction[indices].tolist(),
                "average_equivalent_spherical_radius_m": average_radius[indices].tolist(),
                "precipitate_number_density_per_m3": precipitate_density[indices].tolist(),
                "nucleation_rate_per_m3_s": nucleation_rate[indices].tolist(),
                "driving_force_J_per_m3": driving_force[indices].tolist(),
            },
            "final_particle_size_distribution": {
                "equivalent_spherical_radius_m": particle_radii.tolist(),
                "particle_number_density_per_bin_per_m3": particle_density_per_bin.tolist(),
                "stored_quantity_note": "Kawin stores number density in each discrete radius bin, not density per metre of radius.",
            },
            "upstream_quantity_contract": {
                "volFrac": "dimensionless precipitate volume fraction",
                "fconc": "fraction-weighted precipitate solute contribution on the mole-fraction balance basis",
                "precipitateDensity": "number of precipitates per m3",
                "nucRate": "nucleation events per m3 per s after multiplication by available site density",
                "drivingForce": "volumetric driving force in J per m3",
                "PBM.PSD": "number density in each discrete radius bin per m3",
                "PBM.PSDsize": "mean radius of each discrete bin in m",
            },
            "solver_steps": int(model.ultra_step_count),
            "transport_parameter_family_used": family,
            "numerical_verification": {
                "maximum_absolute_solute_mass_closure_error": float(np.max(mass_error)),
                "solute_mass_closure_tolerance": 1e-8,
                "population_balance_grid_convergence_assessed": False,
            },
        },
        "assumptions": [
            "Binary matrix and one precipitate phase.",
            "Isothermal thermal history.",
            "Homogeneous bulk nucleation with an explicit site density.",
            "Spherical precipitates with one explicit interfacial energy.",
            "Constant elastic strain-energy density.",
            "Infinite diffusion within precipitates.",
        ],
        "solver": {
            "name": "kawin",
            "versions": dict(versions),
            "model": "binary KWN Eulerian population balance",
            "global_phase_stability_assessed": False,
            "pressure_behavior": "Kawin 0.5 thermodynamic calls use a fixed 101325 Pa; pressure was not swept or exposed as a backend solver variable.",
        },
        "scientific_status": "numerically_executed_requires_bin_and_experimental_validation",
        "warnings": sorted(set(runtime_warnings)),
        "limits": {**limits, "max_solver_steps": max_steps},
    }
    return _finish_response(response, max_result_bytes=limits["max_result_bytes"])


def execute_request(value: Any, *, workspace_root: str | Path) -> dict[str, Any]:
    """Execute one closed request in the qualified isolated runtime."""

    versions = _require_runtime()
    root = Path(workspace_root)
    try:
        root = root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise KineticsInputError("workspace_root does not exist") from exc
    request = require_mapping(value, field="request")
    operation = require_nonempty_string(request.get("operation"), field="operation", maximum=128)
    if operation == "transport_coefficients":
        result = _transport_coefficients(value, workspace_root=root, versions=versions)
    elif operation == "single_phase_diffusion_1d":
        result = _single_phase_diffusion_1d(value, workspace_root=root, versions=versions)
    elif operation == "binary_precipitation_kwn":
        result = _binary_precipitation_kwn(value, workspace_root=root, versions=versions)
    elif operation == "phase_field_readiness":
        from .phase_field import validate_phase_field_readiness

        result = validate_phase_field_readiness(
            value,
            workspace_root=root,
            versions=versions,
            finish_response=_finish_response,
        )
    elif operation in {"phase_field", "coupled_solidification_back_diffusion"}:
        raise KineticsUnsupportedError(
            f"{operation} requires a separately qualified external solver and cannot run in Kawin"
        )
    else:
        raise KineticsUnsupportedError(f"unknown kinetics operation {operation!r}")

    # Bind the returned evidence to the exact canonical request bytes received
    # at this boundary.  Operation-specific normalized request echoes remain
    # useful scientific metadata, while this digest prevents a runtime result
    # for one request from being accepted as the result for another.
    request_bytes = _canonical_json_bytes(value)
    unsigned = dict(result)
    unsigned.pop("evidence", None)
    unsigned["input_request_evidence"] = {
        "algorithm": "sha256",
        "canonicalization": "UTF-8 JSON, sorted keys, compact separators, finite numbers",
        "sha256": _sha256(request_bytes),
        "size_bytes": len(request_bytes),
    }
    limits = require_mapping(unsigned.get("limits"), field="result.limits")
    max_result_bytes = require_int(
        limits.get("max_result_bytes"),
        field="result.limits.max_result_bytes",
        minimum=1024,
        maximum=64 * 1024 * 1024,
    )
    return _finish_response(unsigned, max_result_bytes=max_result_bytes)
