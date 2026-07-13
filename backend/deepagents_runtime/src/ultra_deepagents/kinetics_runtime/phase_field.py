"""Strict phase-field submission validation without pretending to solve a PDE."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .contract import (
    REQUEST_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    normalize_limits,
    require_bool,
    require_float,
    require_int,
    require_keys,
    require_mapping,
    require_nonempty_string,
    safe_existing_file,
)
from .errors import KineticsInputError, KineticsUnsupportedError

_MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
_FIELD_KINDS = {
    "conserved": ("Cahn-Hilliard", "mobility"),
    "nonconserved": ("Allen-Cahn", "relaxation"),
}


def _string_sequence(
    value: Any,
    *,
    field: str,
    minimum: int = 1,
    maximum: int = 64,
) -> list[str]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise KineticsInputError(f"{field} must be an array")
    if len(value) < minimum or len(value) > maximum:
        raise KineticsInputError(f"{field} length must be in [{minimum}, {maximum}]")
    result = [
        require_nonempty_string(item, field=f"{field}[{index}]", maximum=256)
        for index, item in enumerate(value)
    ]
    if len(set(result)) != len(result):
        raise KineticsInputError(f"{field} must not contain duplicates")
    return result


def _artifact(value: Any, *, workspace_root: Path, field: str) -> dict[str, Any]:
    artifact = require_mapping(value, field=field)
    require_keys(
        artifact,
        field=field,
        required={
            "path",
            "sha256",
            "size_bytes",
            "artifact_id",
            "source",
            "license_id",
            "assessment_scope",
        },
    )
    path = safe_existing_file(
        artifact["path"], workspace_root=workspace_root, field=f"{field}.path"
    )
    size = require_int(
        artifact["size_bytes"],
        field=f"{field}.size_bytes",
        minimum=1,
        maximum=_MAX_ARTIFACT_BYTES,
    )
    payload = path.read_bytes()
    if len(payload) != size:
        raise KineticsInputError(f"{field}.size_bytes does not match the staged artifact")
    expected = require_nonempty_string(
        artifact["sha256"], field=f"{field}.sha256", maximum=64
    ).lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise KineticsInputError(f"{field}.sha256 must be a lowercase SHA-256 digest")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise KineticsInputError(f"{field}.sha256 does not match the staged artifact")
    return {
        "artifact_id": require_nonempty_string(
            artifact["artifact_id"], field=f"{field}.artifact_id", maximum=256
        ),
        "sha256": actual,
        "size_bytes": size,
        "source": require_nonempty_string(
            artifact["source"], field=f"{field}.source", maximum=4096
        ),
        "license_id": require_nonempty_string(
            artifact["license_id"], field=f"{field}.license_id", maximum=512
        ),
        "assessment_scope": require_nonempty_string(
            artifact["assessment_scope"],
            field=f"{field}.assessment_scope",
            maximum=4096,
        ),
    }


def _strictly_decreasing_positive(value: Any, *, field: str) -> list[float]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise KineticsInputError(f"{field} must be an array")
    if len(value) < 3 or len(value) > 16:
        raise KineticsInputError(f"{field} must contain 3 to 16 refinement levels")
    result = [
        require_float(
            item,
            field=f"{field}[{index}]",
            minimum=1e-30,
            maximum=1e30,
            include_minimum=False,
        )
        for index, item in enumerate(value)
    ]
    if any(right >= left for left, right in zip(result, result[1:], strict=False)):
        raise KineticsInputError(f"{field} must be strictly decreasing")
    return result


def _fields(value: Any, *, workspace_root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise KineticsInputError("model.fields must be an array")
    if not 1 <= len(value) <= 32:
        raise KineticsInputError("model.fields must contain 1 to 32 fields")
    result: list[dict[str, Any]] = []
    field_kinds: dict[str, str] = {}
    for index, raw in enumerate(value):
        field = require_mapping(raw, field=f"model.fields[{index}]")
        require_keys(
            field,
            field=f"model.fields[{index}]",
            required={
                "name",
                "kind",
                "equation",
                "physical_quantity",
                "unit",
                "initial_condition_artifact",
            },
        )
        name = require_nonempty_string(
            field["name"], field=f"model.fields[{index}].name", maximum=128
        )
        if name in field_kinds:
            raise KineticsInputError("model.fields names must be unique")
        kind = require_nonempty_string(
            field["kind"], field=f"model.fields[{index}].kind", maximum=32
        ).lower()
        if kind not in _FIELD_KINDS:
            raise KineticsInputError(
                f"model.fields[{index}].kind must be conserved or nonconserved"
            )
        equation = require_nonempty_string(
            field["equation"], field=f"model.fields[{index}].equation", maximum=64
        )
        expected_equation = _FIELD_KINDS[kind][0]
        if equation != expected_equation:
            raise KineticsInputError(
                f"model.fields[{index}].equation must be {expected_equation!r} for {kind} fields"
            )
        field_kinds[name] = kind
        result.append(
            {
                "name": name,
                "kind": kind,
                "equation": equation,
                "physical_quantity": require_nonempty_string(
                    field["physical_quantity"],
                    field=f"model.fields[{index}].physical_quantity",
                    maximum=256,
                ),
                "unit": require_nonempty_string(
                    field["unit"], field=f"model.fields[{index}].unit", maximum=128
                ),
                "initial_condition_artifact": _artifact(
                    field["initial_condition_artifact"],
                    workspace_root=workspace_root,
                    field=f"model.fields[{index}].initial_condition_artifact",
                ),
            }
        )
    return result, field_kinds


def _coefficient(
    value: Any,
    *,
    field: str,
    workspace_root: Path,
    allowed_kinds: set[str] | None = None,
) -> dict[str, Any]:
    item = require_mapping(value, field=field)
    required = {"value", "unit", "source_artifact", "temperature_limits_K"}
    if allowed_kinds is not None:
        required |= {"field", "kind"}
    else:
        required |= {"field_i", "field_j"}
    require_keys(item, field=field, required=required)
    result: dict[str, Any] = {
        "value": require_float(item["value"], field=f"{field}.value", minimum=0.0, maximum=1e30),
        "unit": require_nonempty_string(item["unit"], field=f"{field}.unit", maximum=128),
        "source_artifact": _artifact(
            item["source_artifact"], workspace_root=workspace_root, field=f"{field}.source_artifact"
        ),
    }
    temperature_limits = item["temperature_limits_K"]
    if (
        isinstance(temperature_limits, (str, bytes, bytearray))
        or not isinstance(temperature_limits, Sequence)
        or len(temperature_limits) != 2
    ):
        raise KineticsInputError(f"{field}.temperature_limits_K must contain two values")
    result["temperature_limits_K"] = [
        require_float(
            temperature_limits[index],
            field=f"{field}.temperature_limits_K[{index}]",
            minimum=1.0,
            maximum=10000.0,
        )
        for index in range(2)
    ]
    if result["temperature_limits_K"][0] >= result["temperature_limits_K"][1]:
        raise KineticsInputError(f"{field}.temperature_limits_K must be strictly increasing")
    if allowed_kinds is not None:
        result["field"] = require_nonempty_string(
            item["field"], field=f"{field}.field", maximum=128
        )
        result["kind"] = require_nonempty_string(
            item["kind"], field=f"{field}.kind", maximum=32
        ).lower()
        if result["kind"] not in allowed_kinds:
            raise KineticsInputError(f"{field}.kind is unsupported")
    else:
        result["field_i"] = require_nonempty_string(
            item["field_i"], field=f"{field}.field_i", maximum=128
        )
        result["field_j"] = require_nonempty_string(
            item["field_j"], field=f"{field}.field_j", maximum=128
        )
    return result


def _sequence(value: Any, *, field: str, minimum: int, maximum: int) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise KineticsInputError(f"{field} must be an array")
    if not minimum <= len(value) <= maximum:
        raise KineticsInputError(f"{field} length must be in [{minimum}, {maximum}]")
    return list(value)


def validate_phase_field_readiness(
    value: Any,
    *,
    workspace_root: Path,
    versions: Mapping[str, str],
    finish_response: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Validate a bounded external phase-field submission; never execute it."""

    request = require_mapping(value, field="request")
    require_keys(
        request,
        field="request",
        required={
            "schema_version",
            "operation",
            "solver_target",
            "model",
            "domain_mesh",
            "boundary_conditions",
            "integration",
            "convergence_plan",
            "validation_plan",
            "limits",
        },
    )
    if request["schema_version"] != REQUEST_SCHEMA_VERSION:
        raise KineticsInputError(f"schema_version must be {REQUEST_SCHEMA_VERSION!r}")
    if request["operation"] != "phase_field_readiness":
        raise KineticsInputError("operation must be 'phase_field_readiness'")
    limits = normalize_limits(request["limits"])

    target = require_mapping(request["solver_target"], field="solver_target")
    require_keys(
        target,
        field="solver_target",
        required={"name", "version", "execution_environment", "image_digest"},
    )
    image_digest = require_nonempty_string(
        target["image_digest"], field="solver_target.image_digest", maximum=71
    ).lower()
    if (
        not image_digest.startswith("sha256:")
        or len(image_digest) != 71
        or any(character not in "0123456789abcdef" for character in image_digest[7:])
    ):
        raise KineticsInputError("solver_target.image_digest must be an immutable SHA-256 digest")
    normalized_target = {
        "name": require_nonempty_string(target["name"], field="solver_target.name", maximum=128),
        "version": require_nonempty_string(
            target["version"], field="solver_target.version", maximum=128
        ),
        "execution_environment": require_nonempty_string(
            target["execution_environment"],
            field="solver_target.execution_environment",
            maximum=512,
        ),
        "image_digest": image_digest,
    }

    model = require_mapping(request["model"], field="model")
    require_keys(
        model,
        field="model",
        required={
            "temperature_K",
            "fields",
            "free_energy",
            "kinetic_coefficients",
            "gradient_energy_coefficients",
        },
    )
    temperature = require_float(
        model["temperature_K"], field="model.temperature_K", minimum=1.0, maximum=10000.0
    )
    fields, field_kinds = _fields(model["fields"], workspace_root=workspace_root)

    free_energy = require_mapping(model["free_energy"], field="model.free_energy")
    require_keys(
        free_energy,
        field="model.free_energy",
        required={"model", "energy_density_unit", "phases", "artifact", "temperature_limits_K"},
    )
    if free_energy["energy_density_unit"] != "J_per_m3":
        raise KineticsUnsupportedError(
            "phase-field readiness currently requires free energy normalized to J_per_m3"
        )
    free_limits_raw = free_energy["temperature_limits_K"]
    if (
        isinstance(free_limits_raw, (str, bytes, bytearray))
        or not isinstance(free_limits_raw, Sequence)
        or len(free_limits_raw) != 2
    ):
        raise KineticsInputError("model.free_energy.temperature_limits_K must contain two values")
    free_limits = [
        require_float(
            free_limits_raw[index],
            field=f"model.free_energy.temperature_limits_K[{index}]",
            minimum=1.0,
            maximum=10000.0,
        )
        for index in range(2)
    ]
    if free_limits[0] >= free_limits[1] or not free_limits[0] <= temperature <= free_limits[1]:
        raise KineticsInputError(
            "model.temperature_K must lie inside the free-energy assessment limits"
        )
    normalized_free_energy = {
        "model": require_nonempty_string(
            free_energy["model"], field="model.free_energy.model", maximum=256
        ),
        "energy_density_unit": "J_per_m3",
        "phases": _string_sequence(free_energy["phases"], field="model.free_energy.phases"),
        "temperature_limits_K": free_limits,
        "artifact": _artifact(
            free_energy["artifact"],
            workspace_root=workspace_root,
            field="model.free_energy.artifact",
        ),
    }

    kinetic_raw = _sequence(
        model["kinetic_coefficients"],
        field="model.kinetic_coefficients",
        minimum=len(fields),
        maximum=64,
    )
    kinetic = [
        _coefficient(
            raw,
            field=f"model.kinetic_coefficients[{index}]",
            workspace_root=workspace_root,
            allowed_kinds={"mobility", "relaxation"},
        )
        for index, raw in enumerate(kinetic_raw)
    ]
    covered_fields: set[str] = set()
    for index, coefficient in enumerate(kinetic):
        name = coefficient["field"]
        if name not in field_kinds:
            raise KineticsInputError(
                f"model.kinetic_coefficients[{index}].field does not name a model field"
            )
        expected_kind = _FIELD_KINDS[field_kinds[name]][1]
        if coefficient["kind"] != expected_kind:
            raise KineticsInputError(
                f"model.kinetic_coefficients[{index}].kind must be {expected_kind!r} for field {name!r}"
            )
        if name in covered_fields:
            raise KineticsInputError("each model field must have exactly one kinetic coefficient")
        if (
            not coefficient["temperature_limits_K"][0]
            <= temperature
            <= coefficient["temperature_limits_K"][1]
        ):
            raise KineticsInputError(
                f"model.temperature_K is outside the kinetic assessment for field {name!r}"
            )
        covered_fields.add(name)
    if covered_fields != set(field_kinds):
        raise KineticsInputError("each model field must have exactly one kinetic coefficient")

    gradient_raw = _sequence(
        model["gradient_energy_coefficients"],
        field="model.gradient_energy_coefficients",
        minimum=1,
        maximum=256,
    )
    gradient = [
        _coefficient(
            raw,
            field=f"model.gradient_energy_coefficients[{index}]",
            workspace_root=workspace_root,
        )
        for index, raw in enumerate(gradient_raw)
    ]
    gradient_pairs: set[tuple[str, str]] = set()
    for index, coefficient in enumerate(gradient):
        pair = tuple(sorted((coefficient["field_i"], coefficient["field_j"])))
        if any(name not in field_kinds for name in pair):
            raise KineticsInputError(
                f"model.gradient_energy_coefficients[{index}] names an unknown field"
            )
        if pair in gradient_pairs:
            raise KineticsInputError("gradient-energy field pairs must be unique")
        if (
            not coefficient["temperature_limits_K"][0]
            <= temperature
            <= coefficient["temperature_limits_K"][1]
        ):
            raise KineticsInputError(
                f"model.temperature_K is outside gradient-energy assessment {index}"
            )
        gradient_pairs.add(pair)

    mesh = require_mapping(request["domain_mesh"], field="domain_mesh")
    require_keys(
        mesh,
        field="domain_mesh",
        required={"dimensions", "extent_m", "cells", "spatial_discretization", "mesh_source"},
    )
    dimensions = require_int(
        mesh["dimensions"], field="domain_mesh.dimensions", minimum=1, maximum=3
    )
    extent = _sequence(
        mesh["extent_m"], field="domain_mesh.extent_m", minimum=dimensions, maximum=dimensions
    )
    cells = _sequence(
        mesh["cells"], field="domain_mesh.cells", minimum=dimensions, maximum=dimensions
    )
    normalized_cells = [
        require_int(item, field=f"domain_mesh.cells[{index}]", minimum=2, maximum=1_000_000)
        for index, item in enumerate(cells)
    ]
    normalized_mesh = {
        "dimensions": dimensions,
        "extent_m": [
            require_float(
                item,
                field=f"domain_mesh.extent_m[{index}]",
                minimum=1e-30,
                maximum=1e6,
                include_minimum=False,
            )
            for index, item in enumerate(extent)
        ],
        "cells": normalized_cells,
        "spatial_discretization": require_nonempty_string(
            mesh["spatial_discretization"],
            field="domain_mesh.spatial_discretization",
            maximum=256,
        ),
        "mesh_source": require_nonempty_string(
            mesh["mesh_source"], field="domain_mesh.mesh_source", maximum=4096
        ),
    }
    if math.prod(normalized_cells) > 100_000_000:
        raise KineticsInputError("domain_mesh contains more than 100 million cells")

    boundary_raw = _sequence(
        request["boundary_conditions"],
        field="boundary_conditions",
        minimum=len(fields),
        maximum=256,
    )
    boundaries: list[dict[str, Any]] = []
    boundary_fields: set[str] = set()
    for index, raw in enumerate(boundary_raw):
        boundary = require_mapping(raw, field=f"boundary_conditions[{index}]")
        require_keys(
            boundary,
            field=f"boundary_conditions[{index}]",
            required={"field", "boundary", "kind", "unit", "source"},
            optional={"value"},
        )
        name = require_nonempty_string(
            boundary["field"], field=f"boundary_conditions[{index}].field", maximum=128
        )
        if name not in field_kinds:
            raise KineticsInputError(f"boundary_conditions[{index}].field is unknown")
        kind = require_nonempty_string(
            boundary["kind"], field=f"boundary_conditions[{index}].kind", maximum=32
        ).lower()
        if kind not in {"periodic", "zero_flux", "dirichlet", "neumann"}:
            raise KineticsInputError(f"boundary_conditions[{index}].kind is unsupported")
        has_value = "value" in boundary
        if (kind in {"dirichlet", "neumann"}) != has_value:
            raise KineticsInputError(
                f"boundary_conditions[{index}] must provide value only for dirichlet or neumann"
            )
        normalized_boundary: dict[str, Any] = {
            "field": name,
            "boundary": require_nonempty_string(
                boundary["boundary"],
                field=f"boundary_conditions[{index}].boundary",
                maximum=128,
            ),
            "kind": kind,
            "unit": require_nonempty_string(
                boundary["unit"], field=f"boundary_conditions[{index}].unit", maximum=128
            ),
            "source": require_nonempty_string(
                boundary["source"],
                field=f"boundary_conditions[{index}].source",
                maximum=4096,
            ),
        }
        if has_value:
            normalized_boundary["value"] = require_float(
                boundary["value"],
                field=f"boundary_conditions[{index}].value",
                minimum=-1e30,
                maximum=1e30,
            )
        boundaries.append(normalized_boundary)
        boundary_fields.add(name)
    if boundary_fields != set(field_kinds):
        raise KineticsInputError("every model field must have at least one boundary condition")

    integration = require_mapping(request["integration"], field="integration")
    require_keys(
        integration,
        field="integration",
        required={
            "duration_s",
            "initial_time_step_s",
            "maximum_time_step_s",
            "time_integrator",
            "nonlinear_relative_tolerance",
            "linear_relative_tolerance",
            "maximum_nonlinear_iterations",
        },
    )
    duration = require_float(
        integration["duration_s"], field="integration.duration_s", minimum=1e-30, maximum=1e30
    )
    initial_step = require_float(
        integration["initial_time_step_s"],
        field="integration.initial_time_step_s",
        minimum=1e-30,
        maximum=duration,
    )
    maximum_step = require_float(
        integration["maximum_time_step_s"],
        field="integration.maximum_time_step_s",
        minimum=initial_step,
        maximum=duration,
    )
    normalized_integration = {
        "duration_s": duration,
        "initial_time_step_s": initial_step,
        "maximum_time_step_s": maximum_step,
        "time_integrator": require_nonempty_string(
            integration["time_integrator"], field="integration.time_integrator", maximum=128
        ),
        "nonlinear_relative_tolerance": require_float(
            integration["nonlinear_relative_tolerance"],
            field="integration.nonlinear_relative_tolerance",
            minimum=1e-15,
            maximum=1e-2,
        ),
        "linear_relative_tolerance": require_float(
            integration["linear_relative_tolerance"],
            field="integration.linear_relative_tolerance",
            minimum=1e-15,
            maximum=1e-2,
        ),
        "maximum_nonlinear_iterations": require_int(
            integration["maximum_nonlinear_iterations"],
            field="integration.maximum_nonlinear_iterations",
            minimum=1,
            maximum=1_000_000,
        ),
    }

    convergence = require_mapping(request["convergence_plan"], field="convergence_plan")
    require_keys(
        convergence,
        field="convergence_plan",
        required={
            "mesh_characteristic_lengths_m",
            "maximum_time_steps_s",
            "observables",
            "maximum_relative_change",
        },
    )
    normalized_convergence = {
        "mesh_characteristic_lengths_m": _strictly_decreasing_positive(
            convergence["mesh_characteristic_lengths_m"],
            field="convergence_plan.mesh_characteristic_lengths_m",
        ),
        "maximum_time_steps_s": _strictly_decreasing_positive(
            convergence["maximum_time_steps_s"],
            field="convergence_plan.maximum_time_steps_s",
        ),
        "observables": _string_sequence(
            convergence["observables"], field="convergence_plan.observables"
        ),
        "maximum_relative_change": require_float(
            convergence["maximum_relative_change"],
            field="convergence_plan.maximum_relative_change",
            minimum=1e-12,
            maximum=0.5,
        ),
    }

    validation = require_mapping(request["validation_plan"], field="validation_plan")
    require_keys(
        validation,
        field="validation_plan",
        required={
            "held_out_dataset",
            "calibration_artifact_ids",
            "calibration_and_validation_disjoint",
            "metrics",
        },
    )
    held_out = _artifact(
        validation["held_out_dataset"],
        workspace_root=workspace_root,
        field="validation_plan.held_out_dataset",
    )
    calibration_ids = _string_sequence(
        validation["calibration_artifact_ids"],
        field="validation_plan.calibration_artifact_ids",
    )
    disjoint = require_bool(
        validation["calibration_and_validation_disjoint"],
        field="validation_plan.calibration_and_validation_disjoint",
    )
    if not disjoint or held_out["artifact_id"] in calibration_ids:
        raise KineticsInputError(
            "phase-field validation must be explicitly held out from calibration"
        )
    metrics_raw = _sequence(
        validation["metrics"], field="validation_plan.metrics", minimum=1, maximum=64
    )
    metrics: list[dict[str, Any]] = []
    for index, raw in enumerate(metrics_raw):
        metric = require_mapping(raw, field=f"validation_plan.metrics[{index}]")
        require_keys(
            metric,
            field=f"validation_plan.metrics[{index}]",
            required={"name", "unit", "acceptance_operator", "acceptance_value"},
        )
        operator = require_nonempty_string(
            metric["acceptance_operator"],
            field=f"validation_plan.metrics[{index}].acceptance_operator",
            maximum=8,
        )
        if operator not in {"<", "<=", ">", ">="}:
            raise KineticsInputError(
                f"validation_plan.metrics[{index}].acceptance_operator is unsupported"
            )
        metrics.append(
            {
                "name": require_nonempty_string(
                    metric["name"], field=f"validation_plan.metrics[{index}].name", maximum=256
                ),
                "unit": require_nonempty_string(
                    metric["unit"], field=f"validation_plan.metrics[{index}].unit", maximum=128
                ),
                "acceptance_operator": operator,
                "acceptance_value": require_float(
                    metric["acceptance_value"],
                    field=f"validation_plan.metrics[{index}].acceptance_value",
                    minimum=-1e30,
                    maximum=1e30,
                ),
            }
        )

    response = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "operation": "phase_field_readiness",
        "result": {
            "status": "submission_contract_complete_not_executed",
            "execution_performed": False,
            "pde_solution_available": False,
            "convergence_assessed": False,
            "held_out_validation_performed": False,
            "external_solver_adapter_qualification_required": True,
        },
        "solver_target": normalized_target,
        "model": {
            "temperature_K": temperature,
            "fields": fields,
            "free_energy": normalized_free_energy,
            "kinetic_coefficients": kinetic,
            "gradient_energy_coefficients": gradient,
        },
        "domain_mesh": normalized_mesh,
        "boundary_conditions": boundaries,
        "integration": normalized_integration,
        "convergence_plan": normalized_convergence,
        "validation_plan": {
            "held_out_dataset": held_out,
            "calibration_artifact_ids": calibration_ids,
            "calibration_and_validation_disjoint": True,
            "metrics": metrics,
        },
        "runtime": {
            "validator": "ultra-isolated-kawin",
            "versions": dict(versions),
            "network_access_used": False,
        },
        "scientific_status": "input_contract_validated_external_execution_not_qualified",
        "warnings": [
            "No phase-field PDE was executed.",
            "The external adapter must verify coefficient dimensions against its weak form.",
            "Mesh/time-step convergence and held-out validation are required plans, not completed evidence.",
        ],
        "limits": limits,
    }
    return finish_response(response, max_result_bytes=limits["max_result_bytes"])
