"""Agent-facing, selected-resource tools for the isolated Kawin runtime.

The public tool schemas expose physical model inputs, never filesystem paths,
container commands, dependency choices, resource caps, or arbitrary Kawin
options. The host stages only a server-selected governed TDB and verifies the
content-addressed runtime result before returning it to the model.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain.tools import ToolRuntime, tool

from ultra_deepagents.code_execution.docker import DockerSandboxBackend
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.materials.calphad_tools import (
    CalphadToolError,
    _read_source,
    _resource_binding,
    _safe_directory,
    _secure_write_new,
    _selected_resource_governance_scope,
    _stage_resource_database,
)

REQUEST_SCHEMA_VERSION = "ultra.materials.kinetics-request.v1"
RESULT_SCHEMA_VERSION = "ultra.materials.kinetics-result.v1"
TOOL_EVIDENCE_SCHEMA_VERSION = "ultra.materials.kinetics-tool-evidence.v1"
QUALIFIED_PRESSURE_PA = 101325.0
FIXED_WALL_TIME_SECONDS = 20.0
FIXED_MAX_RESULT_BYTES = 8 * 1024 * 1024
FIXED_MAX_SOLVER_STEPS = 200_000
MAX_REQUEST_BYTES = 1024 * 1024
MAX_BACKEND_OUTPUT_BYTES = FIXED_MAX_RESULT_BYTES + 128 * 1024
QUALIFIED_VERSIONS = {
    "kawin": "0.5.0",
    "numpy": "2.4.6",
    "pycalphad": "0.11.2",
    "scipy": "1.17.1",
}

_IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_REQUEST_PATH_RE = re.compile(r"/workspace/\.ultra/kinetics/requests/[0-9a-f]{64}\.json")


class KineticsToolError(RuntimeError):
    """A typed kinetics request failed before producing trusted evidence."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code
        self.public_message = " ".join((message or code).split())[:1000]


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise KineticsToolError("invalid_typed_input", "input must be finite JSON") from exc


def _safe_json(payload: bytes) -> Any:
    def reject_constant(value: str) -> None:
        raise KineticsToolError("invalid_runtime_output", f"runtime returned {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise KineticsToolError("invalid_runtime_output", "runtime returned duplicate keys")
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise KineticsToolError("invalid_runtime_output", "runtime output is not JSON") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _database_request(
    settings: Any,
    context: AgentRunContext,
    backend: DockerSandboxBackend,
    *,
    upload_roots: tuple[str | Path, ...],
    resource_id: str,
) -> dict[str, Any]:
    try:
        scope = _selected_resource_governance_scope(context, resource_id=resource_id)
        binding = _resource_binding(settings, context, resource_id=resource_id)
    except CalphadToolError as exc:
        message = exc.public_message
        if exc.code == "selected_resource_required":
            message = "Kinetics resource_id must be explicitly selected when the run is created"
        raise KineticsToolError(exc.code, message) from exc
    if scope not in {"owner_validation", "read_only_usage"}:
        raise KineticsToolError("selected_governed_resource_required")
    if binding.get("database_format") != "tdb":
        raise KineticsToolError(
            "kinetics_tdb_required",
            "Kawin execution requires a selected, governed .tdb resource",
        )
    try:
        staged = _stage_resource_database(
            backend,
            upload_roots=tuple(Path(root) for root in upload_roots),
            binding=binding,
        )
    except CalphadToolError as exc:
        raise KineticsToolError(exc.code, exc.public_message) from exc
    return {
        "path": staged["path"],
        "sha256": staged["sha256"],
        "size_bytes": staged["size_bytes"],
        "artifact_id": staged["resource_id"],
        "source": staged["source"],
        "license_id": staged["license_id"],
        "assessment_scope": staged["assessment_scope"],
        "reference_state": staged["reference_state"],
        "assessment_temperature_limits_K": staged["temperature_limits_K"],
        "assessment_pressure_limits_Pa": staged["assessment_pressure_limits_Pa"],
    }


def _request_path(backend: DockerSandboxBackend, request: Mapping[str, Any]) -> str:
    payload = _canonical_json(request)
    if not payload or len(payload) > MAX_REQUEST_BYTES:
        raise KineticsToolError("typed_request_too_large")
    digest = _sha256(payload)
    directory = _safe_directory(backend.workspace_dir, ".ultra", "kinetics", "requests")
    target = directory / f"{digest}.json"
    try:
        _secure_write_new(target, payload, mode=0o444)
    except Exception as exc:
        try:
            existing = _read_source(target)
        except Exception:
            raise KineticsToolError("kinetics_request_staging_failed") from exc
        if existing != payload:
            raise KineticsToolError("kinetics_request_staging_failed") from exc
    return f"/workspace/.ultra/kinetics/requests/{digest}.json"


def _cli_command(request_path: str) -> str:
    if _REQUEST_PATH_RE.fullmatch(request_path) is None:
        raise KineticsToolError("unsafe_typed_request_path")
    arguments = [
        "python3",
        "-I",
        "-m",
        "ultra_deepagents.kinetics_runtime.cli",
        "--request",
        request_path,
        "--workspace-root",
        "/workspace",
    ]
    return " ".join(shlex.quote(argument) for argument in arguments)


def _execution_contract(backend: DockerSandboxBackend) -> dict[str, Any]:
    config = backend.config
    image = str(config.image or "").strip().lower()
    if _IMAGE_ID_RE.fullmatch(image) is None:
        raise KineticsToolError("immutable_kinetics_image_required")
    if (
        config.network != "none"
        or config.no_new_privileges is not True
        or not 0 < float(config.cpus) <= 2.0
        or str(config.memory).lower() != "8g"
        or not 0 < int(config.pids_limit) <= 256
        or not FIXED_WALL_TIME_SECONDS < int(config.timeout_seconds) <= 30
        or not FIXED_MAX_RESULT_BYTES < int(config.output_limit_bytes) <= MAX_BACKEND_OUTPUT_BYTES
        or str(config.gpus or "").strip()
    ):
        raise KineticsToolError("unbounded_kinetics_backend")
    return {
        "runtime_image_id": image,
        "network": "none",
        "read_only_root_filesystem": True,
        "cap_drop_all": True,
        "no_new_privileges": True,
        "caller_code_or_paths_accepted": False,
        "selected_governed_tdb_required": True,
        "cpus": float(config.cpus),
        "memory": "8g",
        "pids_limit": int(config.pids_limit),
        "outer_wall_time_seconds": int(config.timeout_seconds),
        "inner_wall_time_seconds": FIXED_WALL_TIME_SECONDS,
        "maximum_result_bytes": FIXED_MAX_RESULT_BYTES,
        "maximum_solver_steps": FIXED_MAX_SOLVER_STEPS,
    }


def _finite_number(value: Any, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise KineticsToolError("invalid_runtime_output", f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise KineticsToolError("invalid_runtime_output", f"{label} is outside bounds")
    return result


def _finite_sequence(value: Any, *, label: str, length: int) -> list[float]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise KineticsToolError("invalid_runtime_output", f"{label} is not an array")
    if len(value) != length:
        raise KineticsToolError("invalid_runtime_output", f"{label} has the wrong length")
    return [_finite_number(item, label=f"{label}[]") for item in value]


def _verify_runtime_result(
    value: Any,
    *,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise KineticsToolError("invalid_runtime_output")
    operation = request["operation"]
    if value.get("schema_version") != RESULT_SCHEMA_VERSION or value.get("operation") != operation:
        raise KineticsToolError("invalid_runtime_output", "runtime schema or operation mismatch")
    request_payload = _canonical_json(request)
    if value.get("input_request_evidence") != {
        "algorithm": "sha256",
        "canonicalization": "UTF-8 JSON, sorted keys, compact separators, finite numbers",
        "sha256": _sha256(request_payload),
        "size_bytes": len(request_payload),
    }:
        raise KineticsToolError("invalid_runtime_output", "runtime request identity mismatch")
    database = value.get("database")
    expected_database = request["database"]
    if not isinstance(database, dict) or any(
        database.get(key) != expected_database[key]
        for key in ("artifact_id", "sha256", "size_bytes")
    ):
        raise KineticsToolError("invalid_runtime_output", "database provenance mismatch")
    solver = value.get("solver")
    if not isinstance(solver, dict) or solver.get("versions") != QUALIFIED_VERSIONS:
        raise KineticsToolError("invalid_runtime_output", "runtime versions mismatch")
    expected_limits: dict[str, Any] = {
        "wall_time_seconds": FIXED_WALL_TIME_SECONDS,
        "max_result_bytes": FIXED_MAX_RESULT_BYTES,
    }
    if operation != "transport_coefficients":
        expected_limits["max_solver_steps"] = FIXED_MAX_SOLVER_STEPS
    if value.get("limits") != expected_limits:
        raise KineticsToolError("invalid_runtime_output", "runtime limit evidence mismatch")
    evidence = value.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != {
        "algorithm",
        "canonicalization",
        "sha256",
    }:
        raise KineticsToolError("invalid_runtime_output", "runtime evidence is malformed")
    unsigned = dict(value)
    unsigned.pop("evidence")
    if evidence.get("algorithm") != "sha256" or evidence.get("sha256") != _sha256(
        _canonical_json(unsigned)
    ):
        raise KineticsToolError("invalid_runtime_output", "runtime evidence digest mismatch")

    result = value.get("result")
    if not isinstance(result, dict):
        raise KineticsToolError("invalid_runtime_output", "runtime result is missing")
    if operation == "transport_coefficients":
        tracer = result.get("tracer_diffusivity_m2_per_s")
        matrix = result.get("interdiffusivity_m2_per_s")
        components = request["components"]
        dimension = len(components) - 1
        if not isinstance(tracer, dict) or not tracer:
            raise KineticsToolError("invalid_runtime_output", "tracer diffusivity is missing")
        for name, coefficient in tracer.items():
            if name not in components:
                raise KineticsToolError("invalid_runtime_output", "unexpected tracer component")
            _finite_number(coefficient, label="tracer diffusivity", minimum=0.0)
        family = result.get("transport_parameter_family_used")
        expected_tracer_components = (
            set(components[1:]) if family == "DF/DQ direct diffusivity" else set(components)
        )
        if family not in {"DF/DQ direct diffusivity", "MF/MQ mobility"} or set(tracer) != (
            expected_tracer_components
        ):
            raise KineticsToolError("invalid_runtime_output", "tracer component coverage mismatch")
        if not isinstance(matrix, list) or len(matrix) != dimension:
            raise KineticsToolError("invalid_runtime_output", "interdiffusivity shape mismatch")
        for row in matrix:
            _finite_sequence(row, label="interdiffusivity", length=dimension)
        if (
            result.get("interdiffusivity_rows") != components[1:]
            or result.get("interdiffusivity_columns") != components[1:]
        ):
            raise KineticsToolError("invalid_runtime_output", "interdiffusivity labels mismatch")
        if result.get("reference_component") != components[0]:
            raise KineticsToolError("invalid_runtime_output", "reference component mismatch")
    elif operation == "single_phase_diffusion_1d":
        cells = request["mesh_cells"]
        coordinates = _finite_sequence(
            result.get("coordinates_m"), label="coordinates", length=cells
        )
        if any(right <= left for left, right in zip(coordinates, coordinates[1:], strict=False)):
            raise KineticsToolError("invalid_runtime_output", "diffusion coordinates are unordered")
        expected_spacing = (request["domain_m"][1] - request["domain_m"][0]) / cells
        expected_coordinates = [
            request["domain_m"][0] + (index + 0.5) * expected_spacing for index in range(cells)
        ]
        if any(
            not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-15)
            for observed, expected in zip(coordinates, expected_coordinates, strict=True)
        ):
            raise KineticsToolError("invalid_runtime_output", "diffusion mesh mismatch")
        composition_series: dict[str, dict[str, list[float]]] = {}
        for key in ("initial_composition_mole_fraction", "final_composition_mole_fraction"):
            compositions = result.get(key)
            if not isinstance(compositions, dict) or set(compositions) != set(
                request["components"]
            ):
                raise KineticsToolError("invalid_runtime_output", f"{key} components mismatch")
            composition_series[key] = {}
            for component, series in compositions.items():
                values = _finite_sequence(series, label=key, length=cells)
                if any(item < -1e-12 or item > 1.0 + 1e-12 for item in values):
                    raise KineticsToolError("invalid_runtime_output", f"{key} leaves [0,1]")
                composition_series[key][component] = values
            for index in range(cells):
                closure = math.fsum(
                    composition_series[key][name][index] for name in request["components"]
                )
                if not math.isclose(closure, 1.0, rel_tol=0.0, abs_tol=2e-10):
                    raise KineticsToolError(
                        "invalid_runtime_output", f"{key} does not close to one"
                    )
        if result.get("time_s") != request["duration_s"]:
            raise KineticsToolError("invalid_runtime_output", "diffusion duration mismatch")
        steps = _finite_number(result.get("solver_steps"), label="solver steps", minimum=1.0)
        if not steps.is_integer() or steps > FIXED_MAX_SOLVER_STEPS:
            raise KineticsToolError("invalid_runtime_output", "diffusion step count mismatch")
        verification = result.get("numerical_verification")
        if not isinstance(verification, dict):
            raise KineticsToolError("invalid_runtime_output", "diffusion verification missing")
        tolerance = _finite_number(
            verification.get("mass_closure_tolerance"),
            label="diffusion mass closure tolerance",
            minimum=0.0,
        )
        errors = verification.get("absolute_mass_closure_error")
        if (
            tolerance > 1e-8
            or not isinstance(errors, dict)
            or set(errors) != set(request["components"])
        ):
            raise KineticsToolError("invalid_runtime_output", "diffusion mass closure failed")
        for component in request["components"]:
            initial_mean = (
                math.fsum(composition_series["initial_composition_mole_fraction"][component])
                / cells
            )
            final_mean = (
                math.fsum(composition_series["final_composition_mole_fraction"][component]) / cells
            )
            independently_recomputed_error = abs(final_mean - initial_mean)
            reported_error = _finite_number(
                errors[component], label="diffusion mass closure", minimum=0.0
            )
            if independently_recomputed_error > tolerance or not math.isclose(
                reported_error,
                independently_recomputed_error,
                rel_tol=1e-6,
                abs_tol=1e-14,
            ):
                raise KineticsToolError("invalid_runtime_output", "diffusion mass closure failed")
    elif operation == "binary_precipitation_kwn":
        final = result.get("final")
        if not isinstance(final, dict) or final.get("time_s") != request["duration_s"]:
            raise KineticsToolError("invalid_runtime_output", "precipitation duration mismatch")
        for key in (
            "matrix_solute_mole_fraction",
            "precipitate_volume_fraction",
            "average_equivalent_spherical_radius_m",
            "precipitate_number_density_per_m3",
            "nucleation_rate_per_m3_s",
            "reconstructed_bulk_solute_mole_fraction",
        ):
            _finite_number(final.get(key), label=f"precipitation {key}", minimum=0.0)
        if any(
            float(final[key]) > 1.0 + 1e-12
            for key in (
                "matrix_solute_mole_fraction",
                "precipitate_volume_fraction",
                "reconstructed_bulk_solute_mole_fraction",
            )
        ):
            raise KineticsToolError("invalid_runtime_output", "precipitation fraction exceeds one")
        if not math.isclose(
            float(final["reconstructed_bulk_solute_mole_fraction"]),
            float(request["initial_solute_mole_fraction"]),
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise KineticsToolError("invalid_runtime_output", "precipitation mass closure failed")
        _finite_number(final.get("driving_force_J_per_m3"), label="driving force")
        distribution = result.get("final_particle_size_distribution")
        bins = request["population_balance"]["bins"]
        if not isinstance(distribution, dict):
            raise KineticsToolError("invalid_runtime_output", "particle distribution missing")
        radii = _finite_sequence(
            distribution.get("equivalent_spherical_radius_m"),
            label="particle radii",
            length=bins,
        )
        _finite_sequence(
            distribution.get("particle_number_density_per_bin_per_m3"),
            label="particle density",
            length=bins,
        )
        if any(value < 0.0 for value in distribution["particle_number_density_per_bin_per_m3"]):
            raise KineticsToolError("invalid_runtime_output", "particle density is negative")
        if any(right <= left for left, right in zip(radii, radii[1:], strict=False)):
            raise KineticsToolError("invalid_runtime_output", "particle radii are unordered")
        steps = _finite_number(result.get("solver_steps"), label="solver steps", minimum=1.0)
        if not steps.is_integer() or steps > FIXED_MAX_SOLVER_STEPS:
            raise KineticsToolError("invalid_runtime_output", "precipitation step count mismatch")
        verification = result.get("numerical_verification")
        if not isinstance(verification, dict):
            raise KineticsToolError("invalid_runtime_output", "precipitation verification missing")
        tolerance = _finite_number(
            verification.get("solute_mass_closure_tolerance"),
            label="precipitation mass closure tolerance",
            minimum=0.0,
        )
        error = _finite_number(
            verification.get("maximum_absolute_solute_mass_closure_error"),
            label="precipitation mass closure",
            minimum=0.0,
        )
        if tolerance > 1e-8 or error > tolerance:
            raise KineticsToolError("invalid_runtime_output", "precipitation mass closure failed")
    else:
        raise KineticsToolError("unsupported_kinetics_operation")
    return value


def _persist_evidence(
    backend: DockerSandboxBackend,
    *,
    request: Mapping[str, Any],
    result: Mapping[str, Any],
    execution_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if backend.outputs_dir is None:
        raise KineticsToolError("outputs_mount_required")
    envelope = {
        "schema_version": TOOL_EVIDENCE_SCHEMA_VERSION,
        "operation": request["operation"],
        "selected_resource": {
            "resource_id": request["database"]["artifact_id"],
            "sha256": request["database"]["sha256"],
            "size_bytes": request["database"]["size_bytes"],
        },
        "request_sha256": _sha256(_canonical_json(request)),
        "runtime_result": result,
        "execution_contract": execution_contract,
    }
    payload = _canonical_json(envelope)
    if len(payload) > FIXED_MAX_RESULT_BYTES + 128 * 1024:
        raise KineticsToolError("typed_evidence_too_large")
    digest = _sha256(payload)
    operation = str(request["operation"])
    directory = _safe_directory(backend.outputs_dir, "kinetics", operation)
    target = directory / f"{digest}.json"
    try:
        _secure_write_new(target, payload, mode=0o444)
    except Exception as exc:
        try:
            existing = _read_source(target)
        except Exception:
            raise KineticsToolError("kinetics_evidence_persistence_failed") from exc
        if existing != payload:
            raise KineticsToolError("kinetics_evidence_persistence_failed") from exc
    return {
        "path": f"/outputs/kinetics/{operation}/{digest}.json",
        "sha256": digest,
        "size_bytes": len(payload),
        "content_addressed": True,
    }


def execute_kinetics_request_typed(
    backend: DockerSandboxBackend,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute one already-constructed request through the fixed image boundary."""

    execution_contract = _execution_contract(backend)
    request_path = _request_path(backend, request)
    response = backend.execute(_cli_command(request_path))
    output = str(getattr(response, "output", "") or "").encode("utf-8")
    if len(output) > MAX_BACKEND_OUTPUT_BYTES or getattr(response, "truncated", False):
        raise KineticsToolError("kinetics_runtime_output_too_large")
    parsed = _safe_json(output.strip())
    exit_code = getattr(response, "exit_code", None)
    if exit_code != 0:
        error = parsed.get("error") if isinstance(parsed, dict) else None
        code = error.get("code") if isinstance(error, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        raise KineticsToolError(
            str(code or "kinetics_runtime_failed"),
            str(message or "the bounded kinetics runtime rejected the request"),
        )
    result = _verify_runtime_result(parsed, request=request)
    artifact = _persist_evidence(
        backend,
        request=request,
        result=result,
        execution_contract=execution_contract,
    )
    return {"ok": True, "operation": request["operation"], "result": result, "artifact": artifact}


def _base_request(operation: str, database: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "operation": operation,
        "database": dict(database),
        "pressure_Pa": QUALIFIED_PRESSURE_PA,
        "limits": {
            "wall_time_seconds": FIXED_WALL_TIME_SECONDS,
            "max_result_bytes": FIXED_MAX_RESULT_BYTES,
        },
    }


def _public_failure(exc: KineticsToolError) -> str:
    return json.dumps(
        {"ok": False, "error": exc.code, "message": exc.public_message},
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )


def build_kinetics_tools(
    settings: Any,
    *,
    backend: DockerSandboxBackend,
    upload_roots: Iterable[str | Path] = (),
) -> list[Any]:
    """Build the three model-nonextensible kinetics tools for one agent run."""

    resolved_upload_roots = tuple(upload_roots)

    @tool
    def materials_transport_coefficients(
        runtime: ToolRuntime[AgentRunContext],
        resource_id: str,
        components: list[str],
        phase: str,
        independent_composition_mole_fraction: dict[str, float],
        temperature_K: float,  # noqa: N803 - scientific unit suffix is the schema
    ) -> str:
        """Calculate tracer and volume-fixed interdiffusion coefficients with Kawin 0.5 from one explicitly selected governed TDB. Components are ordered with the reference component first; provide every remaining independent mole fraction, one phase, and temperature in K. MF/MQ supports multicomponent cross diffusion; direct DF/DQ is accepted only for a binary and returns only the assessed solute. Pressure is fixed at 101325 Pa. No paths, code, solver options, or resource limits are accepted."""

        try:
            database = _database_request(
                settings,
                runtime.context,
                backend,
                upload_roots=resolved_upload_roots,
                resource_id=resource_id,
            )
            request = _base_request("transport_coefficients", database)
            request.update(
                {
                    "components": components,
                    "phase": phase,
                    "independent_composition_mole_fraction": independent_composition_mole_fraction,
                    "temperature_K": temperature_K,
                }
            )
            result = execute_kinetics_request_typed(backend, request)
        except KineticsToolError as exc:
            return _public_failure(exc)
        return json.dumps(result, allow_nan=False, indent=2, sort_keys=True)

    @tool
    def materials_run_diffusion_1d(
        runtime: ToolRuntime[AgentRunContext],
        resource_id: str,
        components: list[str],
        phase: str,
        temperature_K: float,  # noqa: N803 - scientific unit suffix is the schema
        duration_s: float,
        domain_m: list[float],
        mesh_cells: int,
        initial_profile_coordinates_m: list[float],
        initial_independent_composition_mole_fraction: dict[str, list[float]],
        initial_profile_source: str,
        application_kind: str = "generic_single_phase_diffusion",
        length_scale_source: str = "not applicable to generic diffusion",
    ) -> str:
        """Run isothermal single-phase Cartesian 1-D finite-volume diffusion with Kawin from one selected governed TDB. Uses linear interpolation and zero-flux boundaries. For post-solidification back diffusion set application_kind='post_solidification_back_diffusion' and give the physical length-scale source; this is explicitly post-solidification only and never claims a moving interface. Pressure, solver-step cap, wall cap, and output cap are fixed by the platform. No paths, code, or arbitrary solver options are accepted."""

        try:
            database = _database_request(
                settings,
                runtime.context,
                backend,
                upload_roots=resolved_upload_roots,
                resource_id=resource_id,
            )
            application = {"kind": application_kind}
            if application_kind == "post_solidification_back_diffusion":
                application.update(
                    {
                        "length_scale_source": length_scale_source,
                        "solidification_coupling": "post_solidification_only",
                    }
                )
            request = _base_request("single_phase_diffusion_1d", database)
            request.update(
                {
                    "components": components,
                    "phase": phase,
                    "temperature_K": temperature_K,
                    "duration_s": duration_s,
                    "domain_m": domain_m,
                    "mesh_cells": mesh_cells,
                    "max_solver_steps": FIXED_MAX_SOLVER_STEPS,
                    "boundary_condition": {"kind": "zero_flux"},
                    "initial_profile": {
                        "coordinates_m": initial_profile_coordinates_m,
                        "independent_composition_mole_fraction": (
                            initial_independent_composition_mole_fraction
                        ),
                        "interpolation": "linear",
                        "source": initial_profile_source,
                    },
                    "application": application,
                }
            )
            result = execute_kinetics_request_typed(backend, request)
        except KineticsToolError as exc:
            return _public_failure(exc)
        return json.dumps(result, allow_nan=False, indent=2, sort_keys=True)

    @tool
    def materials_run_binary_precipitation_kwn(
        runtime: ToolRuntime[AgentRunContext],
        resource_id: str,
        components: list[str],
        matrix_phase: str,
        precipitate_phase: str,
        initial_solute_mole_fraction: float,
        temperature_K: float,  # noqa: N803 - scientific unit suffix is the schema
        temperature_source: str,
        duration_s: float,
        matrix_molar_volume_m3_per_mol: float,
        matrix_atoms_per_unit_cell: int,
        bulk_nucleation_site_density_per_m3: float,
        grain_boundary_energy_J_per_m2: float,  # noqa: N803
        matrix_parameter_source: str,
        precipitate_molar_volume_m3_per_mol: float,
        precipitate_atoms_per_unit_cell: int,
        interfacial_energy_J_per_m2: float,  # noqa: N803
        constant_elastic_strain_energy_J_per_m3: float,  # noqa: N803
        precipitate_parameter_source: str,
        nucleation_source: str,
        minimum_radius_m: float,
        maximum_radius_m: float,
        population_balance_bins: int,
    ) -> str:
        """Run bounded isothermal binary Kampmann-Wagner Numerical precipitation with Kawin from one selected governed TDB. The qualified scope is one matrix and one spherical precipitate, tangent driving force, homogeneous bulk nucleation, constant elastic strain-energy density, infinite precipitate diffusion, and a fixed nonadaptive radius grid. All physical parameters require explicit sources. Pressure is fixed at 101325 Pa; wall/result/step/history caps are platform-fixed. No paths, code, heterogeneous nucleation, adaptive bins, or arbitrary solver options are accepted."""

        try:
            database = _database_request(
                settings,
                runtime.context,
                backend,
                upload_roots=resolved_upload_roots,
                resource_id=resource_id,
            )
            request = _base_request("binary_precipitation_kwn", database)
            request.update(
                {
                    "components": components,
                    "matrix_phase": matrix_phase,
                    "precipitate_phase": precipitate_phase,
                    "initial_solute_mole_fraction": initial_solute_mole_fraction,
                    "temperature_K": temperature_K,
                    "temperature_source": temperature_source,
                    "duration_s": duration_s,
                    "driving_force_method": "tangent",
                    "matrix": {
                        "molar_volume_m3_per_mol": matrix_molar_volume_m3_per_mol,
                        "atoms_per_unit_cell": matrix_atoms_per_unit_cell,
                        "bulk_nucleation_site_density_per_m3": (
                            bulk_nucleation_site_density_per_m3
                        ),
                        "grain_boundary_energy_J_per_m2": grain_boundary_energy_J_per_m2,
                        "source": matrix_parameter_source,
                    },
                    "precipitate": {
                        "molar_volume_m3_per_mol": precipitate_molar_volume_m3_per_mol,
                        "atoms_per_unit_cell": precipitate_atoms_per_unit_cell,
                        "interfacial_energy_J_per_m2": interfacial_energy_J_per_m2,
                        "constant_elastic_strain_energy_J_per_m3": (
                            constant_elastic_strain_energy_J_per_m3
                        ),
                        "infinite_precipitate_diffusion": True,
                        "source": precipitate_parameter_source,
                    },
                    "nucleation": {"site": "bulk", "source": nucleation_source},
                    "population_balance": {
                        "min_radius_m": minimum_radius_m,
                        "max_radius_m": maximum_radius_m,
                        "bins": population_balance_bins,
                        "adaptive": False,
                        "max_history_points": 128,
                    },
                    "max_solver_steps": FIXED_MAX_SOLVER_STEPS,
                }
            )
            result = execute_kinetics_request_typed(backend, request)
        except KineticsToolError as exc:
            return _public_failure(exc)
        return json.dumps(result, allow_nan=False, indent=2, sort_keys=True)

    return [
        materials_transport_coefficients,
        materials_run_diffusion_1d,
        materials_run_binary_precipitation_kwn,
    ]


__all__ = [
    "KineticsToolError",
    "build_kinetics_tools",
    "execute_kinetics_request_typed",
]
