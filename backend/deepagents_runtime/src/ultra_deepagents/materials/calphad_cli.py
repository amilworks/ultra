"""Fixed, networkless CLI for typed CALPHAD inspection/equilibrium/Scheil tools.

The agent-facing tool writes a content-addressed request under ``/workspace``
and invokes this module with no caller-authored command or Python.  This module
revalidates the request, confines database paths, calls only the bounded public
CALPHAD runtime, and writes canonical content-addressed evidence to
``/outputs/calphad``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .calphad import (
    CalphadExecutionError,
    CalphadInputError,
    CalphadTimeoutError,
    CalphadUnsupportedError,
    canonicalize_equilibrium_compositions,
    inspect_calphad_input,
    run_calphad_equilibrium,
)
from .processing_kinetics import (
    DEFAULT_SCHEIL_STEPS,
    MAX_STEP_TEMPERATURE_K,
    MAX_STOP_LIQUID_FRACTION,
    MIN_STEP_TEMPERATURE_K,
    MIN_STOP_LIQUID_FRACTION,
    run_scheil_solidification,
)

REQUEST_SCHEMA_VERSION = "ultra.calphad.typed-request.v2"
TOOL_EVIDENCE_SCHEMA_VERSION = "ultra.calphad.tool-evidence.v3"
FAILURE_EVIDENCE_SCHEMA_VERSION = "ultra.calphad.failure-evidence.v1"
RESULT_MARKER = "ULTRA_CALPHAD_TOOL_RESULT="

WORKSPACE_ROOT = Path("/workspace")
REQUEST_ROOT = WORKSPACE_ROOT / ".ultra/calphad/requests"
STAGED_DATABASE_ROOTS = (
    WORKSPACE_ROOT / ".ultra/calphad/staged",
    WORKSPACE_ROOT / "staged_uploads",
    WORKSPACE_ROOT / "staged_resources",
)
OUTPUT_ROOT = Path("/outputs/calphad")
EMBEDDED_REGISTRY_ROOT = Path("/opt/ultra-calphad")

MAX_REQUEST_BYTES = 1024 * 1024
MAX_TYPED_COMPONENTS = 32
MAX_TYPED_PHASES = 128
MAX_TYPED_AXIS_VALUES = 64
MAX_TYPED_GRID_POINTS = 256
FIXED_WALL_TIME_SECONDS = 30.0
FIXED_MAX_RESULT_BYTES = 16 * 1024 * 1024
FIXED_SCHEIL_MAX_STEPS = DEFAULT_SCHEIL_STEPS
FIXED_SCHEIL_PRESSURE_PA = 101325.0
FIXED_SCHEIL_LIQUID_PHASE = "LIQUID"
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
TYPED_SANDBOX_MAX_CPUS = 8.0
TYPED_SANDBOX_MAX_MEMORY_BYTES = 32 * 1024**3
TYPED_SANDBOX_MAX_PIDS = 4096

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NAME_RE = re.compile(r"^[A-Za-z0-9_+./:-]{1,128}$")
_FAILURE_CODE_RE = re.compile(r"^[a-z0-9_]{1,64}$")

FAILURE_STATUSES = frozenset({"failed", "timeout", "unsupported"})
FAILURE_DOMAINS = frozenset({"input", "scientific", "platform"})
FAILURE_STAGES = frozenset({"parse", "solver", "result_validation", "sandbox_runtime"})
FAILURE_CODES = frozenset(
    {
        "calphad_parse_failed",
        "calphad_parse_timeout",
        "calphad_parse_unsupported",
        "calphad_solver_failed",
        "calphad_solver_timeout",
        "calphad_solver_unsupported",
        "calphad_result_invalid",
        "calphad_runtime_internal_failure",
        "calphad_sandbox_failed",
        "calphad_sandbox_timeout",
    }
)


def valid_failure_tuple(
    *,
    status: str,
    failure_domain: str,
    failure_stage: str,
    failure_code: str,
    solver_started: bool,
    operation: str | None = None,
) -> bool:
    if (
        status not in FAILURE_STATUSES
        or failure_domain not in FAILURE_DOMAINS
        or failure_stage not in FAILURE_STAGES
        or failure_code not in FAILURE_CODES
        or not isinstance(solver_started, bool)
    ):
        return False
    expected: dict[str, tuple[str, set[str], str, set[bool]]] = {
        "calphad_parse_failed": ("failed", {"input", "scientific"}, "parse", {False}),
        "calphad_parse_timeout": ("timeout", {"scientific"}, "parse", {False}),
        "calphad_parse_unsupported": ("unsupported", {"input"}, "parse", {False}),
        "calphad_solver_failed": ("failed", {"input", "scientific"}, "solver", {False, True}),
        "calphad_solver_timeout": ("timeout", {"scientific"}, "solver", {True}),
        "calphad_solver_unsupported": (
            "unsupported",
            {"scientific"},
            "solver",
            {False},
        ),
        "calphad_result_invalid": (
            "failed",
            {"scientific"},
            "result_validation",
            {False, True},
        ),
        "calphad_runtime_internal_failure": (
            "failed",
            {"platform"},
            failure_stage,
            {False},
        ),
        "calphad_sandbox_failed": (
            "failed",
            {"platform"},
            "sandbox_runtime",
            {False},
        ),
        "calphad_sandbox_timeout": (
            "timeout",
            {"platform"},
            "sandbox_runtime",
            {False},
        ),
    }
    expected_status, domains, stage, solver_values = expected[failure_code]
    if (
        status != expected_status
        or failure_domain not in domains
        or failure_stage != stage
        or solver_started not in solver_values
        or (
            failure_code == "calphad_runtime_internal_failure"
            and failure_stage not in {"parse", "solver"}
        )
        or (
            failure_code == "calphad_solver_failed"
            and solver_started != (failure_domain == "scientific")
        )
    ):
        return False
    if operation is None:
        return True
    if operation == "inspect":
        if failure_stage == "solver" or solver_started:
            return False
    elif operation in {"equilibrium", "scheil"}:
        if failure_stage == "parse":
            return False
    else:
        return False
    if failure_code == "calphad_result_invalid":
        return solver_started == (operation in {"equilibrium", "scheil"})
    return True


class TypedCalphadError(RuntimeError):
    """A fail-closed typed-CALPHAD request or evidence error."""


@dataclass
class CalphadTerminalFailureError(RuntimeError):
    """A bounded terminal outcome eligible for exact retained failure evidence."""

    status: str
    failure_domain: str
    failure_stage: str
    failure_code: str
    exit_code: int | None
    solver_started: bool

    def __post_init__(self) -> None:
        if (
            not valid_failure_tuple(
                status=self.status,
                failure_domain=self.failure_domain,
                failure_stage=self.failure_stage,
                failure_code=self.failure_code,
                solver_started=self.solver_started,
            )
            or _FAILURE_CODE_RE.fullmatch(self.failure_code) is None
            or isinstance(self.exit_code, bool)
            or (self.exit_code is not None and not isinstance(self.exit_code, int))
            or not isinstance(self.solver_started, bool)
        ):
            raise ValueError("invalid typed CALPHAD terminal outcome")
        RuntimeError.__init__(self, self.failure_code)


def _reject_constant(value: str) -> None:
    raise TypedCalphadError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TypedCalphadError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TypedCalphadError("typed CALPHAD evidence must be finite JSON") from exc


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_regular_file(path: Path, *, maximum: int, label: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise TypedCalphadError(f"{label} is missing or unreadable") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise TypedCalphadError(f"{label} must be a regular non-symlink file")
    if before.st_size <= 0 or before.st_size > maximum:
        raise TypedCalphadError(f"{label} size is outside its fixed bound")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TypedCalphadError(f"could not securely open {label}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise TypedCalphadError(f"{label} must be a regular file")
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise TypedCalphadError(f"{label} changed while opening")
        remaining = opened.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise TypedCalphadError(f"{label} was truncated while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise TypedCalphadError(f"{label} grew while reading")
        after = os.fstat(descriptor)
        if (after.st_size, after.st_mtime_ns) != (opened.st_size, opened.st_mtime_ns):
            raise TypedCalphadError(f"{label} changed while reading")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _reject_symlink_components(path: Path, *, root: Path, label: str) -> None:
    """Reject symlinks in every lexical component from ``root`` through ``path``."""

    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise TypedCalphadError(f"{label} is outside its fixed root") from exc
    current = root
    for part in relative.parts:
        if part in {"", ".", ".."}:
            raise TypedCalphadError(f"{label} has an unsafe path component")
        current = current / part
        try:
            info = current.lstat()
        except OSError as exc:
            raise TypedCalphadError(f"{label} is missing") from exc
        if stat.S_ISLNK(info.st_mode):
            raise TypedCalphadError(f"{label} must not contain symbolic links")


def _confined_database_path(raw_path: Any) -> Path:
    if not isinstance(raw_path, str):
        raise TypedCalphadError("resource database path must be a staged workspace path")
    candidate = Path(raw_path)
    try:
        workspace_root = WORKSPACE_ROOT.resolve(strict=True)
    except OSError as exc:
        raise TypedCalphadError("workspace root is missing") from exc
    if not candidate.is_absolute() or not _is_under(candidate, workspace_root):
        raise TypedCalphadError("resource database path must be a staged workspace path")
    try:
        _reject_symlink_components(
            candidate,
            root=workspace_root,
            label="staged CALPHAD database",
        )
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise TypedCalphadError("staged CALPHAD database is missing") from exc
    allowed = tuple(root.resolve(strict=True) for root in STAGED_DATABASE_ROOTS if root.exists())
    if not allowed or not any(_is_under(resolved, root) for root in allowed):
        raise TypedCalphadError("CALPHAD database path is outside the staged resource roots")
    if resolved.suffix.casefold() not in {".tdb", ".dat"}:
        raise TypedCalphadError("staged CALPHAD database has an unsupported extension")
    return resolved


def _require_exact_keys(
    value: Any,
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypedCalphadError(f"{label} must be an object")
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing or unknown:
        raise TypedCalphadError(
            f"{label} schema mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return value


def _required_text(value: Any, *, label: str, maximum: int = 4096) -> str:
    if not isinstance(value, str):
        raise TypedCalphadError(f"{label} must be a string")
    text = value.strip()
    if not text or len(text) > maximum:
        raise TypedCalphadError(f"{label} is missing or exceeds its fixed bound")
    if any(ord(character) < 32 or ord(character) == 127 for character in text):
        raise TypedCalphadError(f"{label} contains control characters")
    return text


def _required_sha(value: Any, *, label: str) -> str:
    text = _required_text(value, label=label, maximum=64).lower()
    if not _SHA256_RE.fullmatch(text):
        raise TypedCalphadError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _required_positive_int(value: Any, *, label: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > maximum:
        raise TypedCalphadError(f"{label} must be an integer in [1, {maximum}]")
    return value


def _finite_number(value: Any, *, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypedCalphadError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < minimum or number > maximum:
        raise TypedCalphadError(f"{label} is outside [{minimum}, {maximum}]")
    return 0.0 if number == 0 else number


def _name_list(value: Any, *, label: str, maximum: int, allow_none: bool) -> list[str] | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, list) or not value or len(value) > maximum:
        raise TypedCalphadError(f"{label} must contain 1..{maximum} names")
    names: list[str] = []
    for raw in value:
        if not isinstance(raw, str):
            raise TypedCalphadError(f"{label} names must be strings")
        name = raw.strip().upper()
        if not _NAME_RE.fullmatch(name):
            raise TypedCalphadError(f"{label} contains an invalid name")
        names.append(name)
    if len(set(names)) != len(names):
        raise TypedCalphadError(f"{label} contains duplicate names")
    return sorted(names)


def _axis(
    value: Any,
    *,
    label: str,
    minimum: float,
    maximum: float,
) -> list[float]:
    if not isinstance(value, list) or not value or len(value) > MAX_TYPED_AXIS_VALUES:
        raise TypedCalphadError(f"{label} must contain 1..{MAX_TYPED_AXIS_VALUES} finite values")
    values = [_finite_number(item, label=label, minimum=minimum, maximum=maximum) for item in value]
    if len(set(values)) != len(values):
        raise TypedCalphadError(f"{label} contains duplicate values")
    return sorted(values)


def _temperature_limits(value: Any) -> list[float]:
    limits = _axis(value, label="temperature_limits_K", minimum=1.0, maximum=10_000.0)
    if len(limits) != 2 or limits[0] >= limits[1]:
        raise TypedCalphadError("temperature_limits_K must be [minimum, maximum]")
    return limits


def _pressure_limits(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise TypedCalphadError("assessment_pressure_limits_Pa must be [minimum, maximum]")
    limits = [
        _finite_number(
            item,
            label="assessment_pressure_limits_Pa",
            minimum=1e-9,
            maximum=1e12,
        )
        for item in value
    ]
    if limits[0] > limits[1]:
        raise TypedCalphadError("assessment_pressure_limits_Pa must be nondecreasing")
    return limits


def _database_format(value: Any, *, label: str = "database.database_format") -> str:
    database_format = _required_text(value, label=label, maximum=3)
    if database_format not in {"tdb", "dat"}:
        raise TypedCalphadError(f"{label} must be tdb or dat")
    return database_format


def _validated_database(database: Any) -> dict[str, Any]:
    value = _require_exact_keys(
        database,
        required={"kind", "database_id"},
        optional={
            "path",
            "resource_id",
            "database_format",
            "sha256",
            "size_bytes",
            "source",
            "license_id",
            "assessment_scope",
            "reference_state",
            "temperature_limits_K",
            "assessment_pressure_limits_Pa",
            "binding_schema",
            "binding_authority",
            "declaration_authority",
        },
        label="database",
    )
    kind = _required_text(value["kind"], label="database.kind", maximum=32)
    database_id = _required_text(value["database_id"], label="database.database_id", maximum=512)
    if kind == "embedded":
        if set(value) != {"kind", "database_id"}:
            raise TypedCalphadError("embedded database accepts only its reviewed database_id")
        return {"kind": kind, "database_id": database_id}
    if kind != "resource":
        raise TypedCalphadError("database.kind must be resource or embedded")
    required_resource = {
        "kind",
        "database_id",
        "path",
        "resource_id",
        "database_format",
        "sha256",
        "size_bytes",
        "source",
        "license_id",
        "assessment_scope",
        "reference_state",
        "temperature_limits_K",
        "assessment_pressure_limits_Pa",
        "binding_schema",
        "binding_authority",
        "declaration_authority",
    }
    if set(value) != required_resource:
        raise TypedCalphadError("resource database requires complete catalog binding/provenance")
    resource_id = _required_text(value["resource_id"], label="database.resource_id", maximum=512)
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", resource_id):
        raise TypedCalphadError("database.resource_id is invalid")
    database_format = _database_format(value["database_format"])
    database_path = _confined_database_path(value["path"])
    if database_path.suffix.casefold() != f".{database_format}":
        raise TypedCalphadError(
            "staged CALPHAD database suffix does not match database.database_format"
        )
    normalized = {
        "kind": kind,
        "database_id": database_id,
        "path": str(database_path),
        "resource_id": resource_id,
        "database_format": database_format,
        "sha256": _required_sha(value["sha256"], label="database.sha256"),
        "size_bytes": _required_positive_int(
            value["size_bytes"], label="database.size_bytes", maximum=64 * 1024 * 1024
        ),
        "source": _required_text(value["source"], label="database.source"),
        "license_id": _required_text(value["license_id"], label="database.license_id", maximum=256),
        "assessment_scope": _required_text(
            value["assessment_scope"], label="database.assessment_scope"
        ),
        "reference_state": _required_text(
            value["reference_state"], label="database.reference_state", maximum=1024
        ),
        "temperature_limits_K": _temperature_limits(value["temperature_limits_K"]),
        "assessment_pressure_limits_Pa": _pressure_limits(value["assessment_pressure_limits_Pa"]),
        "binding_schema": _required_text(
            value["binding_schema"], label="database.binding_schema", maximum=128
        ),
        "binding_authority": _required_text(
            value["binding_authority"], label="database.binding_authority", maximum=128
        ),
        "declaration_authority": _required_text(
            value["declaration_authority"],
            label="database.declaration_authority",
            maximum=128,
        ),
    }
    if normalized["binding_authority"] != "control_resource_catalog":
        raise TypedCalphadError("resource binding is not server-authored")
    if normalized["binding_schema"] not in {
        "ultra.selected_resource.v1",
        "ultra.catalog_resource.v1",
    }:
        raise TypedCalphadError("resource binding schema is unsupported")
    if normalized["declaration_authority"] != "resource_owner":
        raise TypedCalphadError("resource provenance is not owner-declared")
    payload = _read_regular_file(
        Path(normalized["path"]), maximum=64 * 1024 * 1024, label="staged CALPHAD database"
    )
    if len(payload) != normalized["size_bytes"] or _sha256(payload) != normalized["sha256"]:
        raise TypedCalphadError("staged CALPHAD database does not match catalog SHA-256/size")
    return normalized


def _runtime_kwargs(database: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    if database["kind"] == "embedded":
        return EMBEDDED_REGISTRY_ROOT, {"database_id": database["database_id"]}
    return Path(str(database["path"])), {
        "source": database["source"],
        "license_id": database["license_id"],
        "artifact_id": database["resource_id"],
        "assessment_scope": database["assessment_scope"],
        "reference_state": database["reference_state"],
        "expected_sha256": database["sha256"],
        "expected_size_bytes": database["size_bytes"],
        "assessment_temperature_limits_K": database["temperature_limits_K"],
        "assessment_pressure_limits_Pa": database["assessment_pressure_limits_Pa"],
    }


def _validated_request(request: Any) -> dict[str, Any]:
    root = _require_exact_keys(
        request,
        required={
            "schema_version",
            "operation",
            "runtime_image_id",
            "database",
            "selection",
        },
        optional={"inspection_artifact_sha256", "conditions"},
        label="request",
    )
    if root["schema_version"] != REQUEST_SCHEMA_VERSION:
        raise TypedCalphadError("unsupported typed CALPHAD request schema")
    operation = _required_text(root["operation"], label="operation", maximum=32)
    if operation not in {"inspect", "equilibrium", "scheil"}:
        raise TypedCalphadError("operation must be inspect, equilibrium, or scheil")
    runtime_image_id = _required_text(
        root["runtime_image_id"], label="runtime_image_id", maximum=71
    ).lower()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", runtime_image_id):
        raise TypedCalphadError("runtime_image_id must be an immutable Docker image ID")
    database = _validated_database(root["database"])
    selection = _require_exact_keys(
        root["selection"], required={"components", "phases"}, label="selection"
    )
    components = _name_list(
        selection["components"],
        label="components",
        maximum=MAX_TYPED_COMPONENTS,
        allow_none=operation == "inspect",
    )
    phases = _name_list(
        selection["phases"],
        label="phases",
        maximum=MAX_TYPED_PHASES,
        allow_none=operation == "inspect",
    )
    normalized: dict[str, Any] = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "operation": operation,
        "runtime_image_id": runtime_image_id,
        "database": database,
        "selection": {"components": components, "phases": phases},
    }
    if operation == "inspect":
        if "inspection_artifact_sha256" in root or "conditions" in root:
            raise TypedCalphadError("inspection request contains solver-only fields")
        return normalized

    if "inspection_artifact_sha256" not in root or "conditions" not in root:
        raise TypedCalphadError(f"{operation} requires prior inspection evidence and conditions")
    normalized["inspection_artifact_sha256"] = _required_sha(
        root["inspection_artifact_sha256"], label="inspection_artifact_sha256"
    )
    if operation == "scheil":
        conditions = _require_exact_keys(
            root["conditions"],
            required={
                "independent_composition_mole_fraction",
                "start_temperature_K",
                "step_temperature_K",
                "pressure_Pa",
                "stop_liquid_fraction",
            },
            label="conditions",
        )
        start_temperature = _finite_number(
            conditions["start_temperature_K"],
            label="start_temperature_K",
            minimum=1.0,
            maximum=10_000.0,
        )
        step_temperature = _finite_number(
            conditions["step_temperature_K"],
            label="step_temperature_K",
            minimum=MIN_STEP_TEMPERATURE_K,
            maximum=MAX_STEP_TEMPERATURE_K,
        )
        pressure = _finite_number(
            conditions["pressure_Pa"],
            label="pressure_Pa",
            minimum=1e-9,
            maximum=1e12,
        )
        if pressure != FIXED_SCHEIL_PRESSURE_PA:
            raise TypedCalphadError(
                f"qualified Scheil execution requires pressure_Pa={FIXED_SCHEIL_PRESSURE_PA:g} exactly"
            )
        stop_liquid_fraction = _finite_number(
            conditions["stop_liquid_fraction"],
            label="stop_liquid_fraction",
            minimum=MIN_STOP_LIQUID_FRACTION,
            maximum=MAX_STOP_LIQUID_FRACTION,
        )
        assert components is not None and phases is not None
        if FIXED_SCHEIL_LIQUID_PHASE not in phases:
            raise TypedCalphadError(f"Scheil phases must include {FIXED_SCHEIL_LIQUID_PHASE}")
        if database["kind"] == "resource":
            temperature_limits = database["temperature_limits_K"]
            pressure_limits = database["assessment_pressure_limits_Pa"]
            if not temperature_limits[0] <= start_temperature <= temperature_limits[1]:
                raise TypedCalphadError(
                    "start_temperature_K is outside the resource assessment temperature limits"
                )
            if not pressure_limits[0] <= pressure <= pressure_limits[1]:
                raise TypedCalphadError(
                    "pressure_Pa is outside the resource assessment pressure limits"
                )
            minimum_effective_step = step_temperature / (1.2**6)
            worst_case_points = 2 + math.ceil(
                max(0.0, start_temperature - temperature_limits[0]) / minimum_effective_step
            )
            if worst_case_points > FIXED_SCHEIL_MAX_STEPS:
                raise TypedCalphadError(
                    "temperature range/step can exceed the fixed Scheil step limit"
                )
        raw_composition = conditions["independent_composition_mole_fraction"]
        if not isinstance(raw_composition, dict) or len(raw_composition) > MAX_TYPED_COMPONENTS:
            raise TypedCalphadError(
                "independent_composition_mole_fraction must be a bounded object"
            )
        composition: dict[str, float] = {}
        for raw_name, raw_value in raw_composition.items():
            names = _name_list(
                [raw_name], label="composition component", maximum=1, allow_none=False
            )
            assert names is not None
            composition[names[0]] = _finite_number(
                raw_value,
                label=f"X({names[0]})",
                minimum=0.0,
                maximum=1.0,
            )
        try:
            canonical_compositions, _, closure_records = canonicalize_equilibrium_compositions(
                composition,
                components=components,
            )
        except CalphadInputError as exc:
            raise TypedCalphadError(str(exc)) from exc
        if len(closure_records) != 1 or any(
            len(values) != 1 for values in canonical_compositions.values()
        ):
            raise TypedCalphadError("Scheil execution requires one composition point")
        normalized["conditions"] = {
            "independent_composition_mole_fraction": {
                component: values[0] for component, values in sorted(canonical_compositions.items())
            },
            "start_temperature_K": start_temperature,
            "step_temperature_K": step_temperature,
            "pressure_Pa": pressure,
            "stop_liquid_fraction": stop_liquid_fraction,
        }
        return normalized

    conditions = _require_exact_keys(
        root["conditions"],
        required={"temperatures_K", "pressures_Pa", "independent_compositions"},
        label="conditions",
    )
    temperatures = _axis(
        conditions["temperatures_K"], label="temperatures_K", minimum=1.0, maximum=10_000.0
    )
    pressures = _axis(conditions["pressures_Pa"], label="pressures_Pa", minimum=1e-9, maximum=1e12)
    if database["kind"] == "resource":
        pressure_limits = database["assessment_pressure_limits_Pa"]
        if pressures[0] < pressure_limits[0] or pressures[-1] > pressure_limits[1]:
            raise TypedCalphadError(
                "pressures_Pa is outside the resource assessment pressure limits"
            )
    raw_compositions = conditions["independent_compositions"]
    if not isinstance(raw_compositions, dict) or len(raw_compositions) > MAX_TYPED_COMPONENTS:
        raise TypedCalphadError("independent_compositions must be a bounded object")
    compositions: dict[str, list[float]] = {}
    for raw_name, raw_axis in raw_compositions.items():
        names = _name_list([raw_name], label="composition component", maximum=1, allow_none=False)
        assert names is not None
        compositions[names[0]] = _axis(raw_axis, label=f"X({names[0]})", minimum=0.0, maximum=1.0)
    assert components is not None
    try:
        canonical_compositions, _, _ = canonicalize_equilibrium_compositions(
            compositions,
            components=components,
        )
    except CalphadInputError as exc:
        raise TypedCalphadError(str(exc)) from exc
    compositions = {component: list(axis) for component, axis in canonical_compositions.items()}
    grid_points = len(temperatures) * len(pressures)
    for axis in compositions.values():
        grid_points *= len(axis)
    if grid_points > MAX_TYPED_GRID_POINTS:
        raise TypedCalphadError(
            f"typed CALPHAD grid exceeds the {MAX_TYPED_GRID_POINTS}-point limit"
        )
    normalized["conditions"] = {
        "temperatures_K": temperatures,
        "pressures_Pa": pressures,
        "independent_compositions": dict(sorted(compositions.items())),
    }
    return normalized


def _request_from_path(path: Path) -> dict[str, Any]:
    try:
        resolved_root = REQUEST_ROOT.resolve(strict=True)
        _reject_symlink_components(path, root=resolved_root, label="typed CALPHAD request")
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise TypedCalphadError("typed CALPHAD request is missing") from exc
    if not _is_under(resolved, resolved_root) or resolved.parent != resolved_root:
        raise TypedCalphadError("typed CALPHAD request is outside the fixed request root")
    if not _SHA256_RE.fullmatch(resolved.stem) or resolved.suffix != ".json":
        raise TypedCalphadError("typed CALPHAD request filename is invalid")
    payload = _read_regular_file(resolved, maximum=MAX_REQUEST_BYTES, label="CALPHAD request")
    if _sha256(payload) != resolved.stem:
        raise TypedCalphadError("typed CALPHAD request bytes do not match its filename")
    try:
        value = json.loads(
            payload,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TypedCalphadError("typed CALPHAD request is not valid finite UTF-8 JSON") from exc
    return _validated_request(value)


def _binding_identity(database: Mapping[str, Any]) -> dict[str, Any]:
    if database["kind"] == "embedded":
        return {"kind": "embedded", "database_id": database["database_id"]}
    return {
        key: database[key]
        for key in (
            "kind",
            "database_id",
            "resource_id",
            "database_format",
            "sha256",
            "size_bytes",
            "source",
            "license_id",
            "assessment_scope",
            "reference_state",
            "temperature_limits_K",
            "assessment_pressure_limits_Pa",
            "binding_schema",
            "binding_authority",
            "declaration_authority",
        )
    }


def _inspection_evidence(sha256: str) -> dict[str, Any]:
    path = OUTPUT_ROOT / "inspection" / f"{sha256}.json"
    payload = _read_regular_file(path, maximum=MAX_ARTIFACT_BYTES, label="inspection evidence")
    if _sha256(payload) != sha256:
        raise TypedCalphadError("inspection evidence hash mismatch")
    try:
        value = json.loads(
            payload,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TypedCalphadError("inspection evidence is not valid finite JSON") from exc
    evidence = _require_exact_keys(
        value,
        required={
            "schema_version",
            "operation",
            "database_binding",
            "request",
            "result",
            "execution_contract",
            "validation_persistence",
        },
        label="inspection evidence",
    )
    if (
        evidence["schema_version"] != TOOL_EVIDENCE_SCHEMA_VERSION
        or evidence["operation"] != "inspect"
    ):
        raise TypedCalphadError("artifact is not typed CALPHAD inspection evidence")
    return evidence


def _ensure_inventory_chain(
    inspection: Mapping[str, Any],
    *,
    database: Mapping[str, Any],
    components: list[str],
    phases: list[str],
) -> None:
    if inspection.get("database_binding") != _binding_identity(database):
        raise TypedCalphadError("inspection evidence belongs to a different database revision")
    result = inspection.get("result")
    if not isinstance(result, dict):
        raise TypedCalphadError("inspection evidence has no database inventory")
    available_components = set(result.get("available_components") or result.get("components") or [])
    available_phases = set(result.get("available_phases") or result.get("phases") or [])
    if "VA" in available_components and "VA" not in components:
        raise TypedCalphadError(
            "solver components must include VA when the retained inspection inventory declares it"
        )
    if not set(components).issubset(available_components):
        raise TypedCalphadError("requested components are absent from inspection inventory")
    if not set(phases).issubset(available_phases):
        raise TypedCalphadError("requested phases are absent from inspection inventory")
    inspected_sha = str(result.get("sha256") or "")
    if database["kind"] == "resource" and inspected_sha != database["sha256"]:
        raise TypedCalphadError("inspection database SHA-256 disagrees with catalog binding")


def _ensure_result_database_format(
    result: Any,
    *,
    database: Mapping[str, Any],
    operation: str,
) -> None:
    if not isinstance(result, dict):
        raise TypedCalphadError("CALPHAD result does not contain a database manifest")
    manifest: Any = result if operation == "inspect" else result.get("database")
    if not isinstance(manifest, dict):
        raise TypedCalphadError("CALPHAD result does not contain a database manifest")
    result_format = _database_format(manifest.get("format"), label="CALPHAD result database format")
    manifest_path = _required_text(
        manifest.get("path"), label="CALPHAD result database path", maximum=4096
    )
    if Path(manifest_path).suffix.casefold() != f".{result_format}":
        raise TypedCalphadError(
            "CALPHAD result database path suffix does not match its manifest format"
        )
    if database["kind"] == "resource" and result_format != database["database_format"]:
        raise TypedCalphadError(
            "CALPHAD result database format does not match the resource binding"
        )


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    current = path
    while True:
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise TypedCalphadError("CALPHAD output directory is unsafe")
        if current == OUTPUT_ROOT or current.parent == current:
            break
        current = current.parent


def _operation_directory(operation: str) -> str:
    directories = {
        "inspect": "inspection",
        "equilibrium": "equilibrium",
        "scheil": "scheil",
    }
    try:
        return directories[operation]
    except KeyError as exc:
        raise TypedCalphadError("unsupported typed CALPHAD operation") from exc


def _write_content_addressed(operation: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    payload = _canonical_json(evidence)
    if len(payload) <= 0 or len(payload) > MAX_ARTIFACT_BYTES:
        raise TypedCalphadError("typed CALPHAD evidence exceeds its fixed artifact bound")
    digest = _sha256(payload)
    directory = OUTPUT_ROOT / _operation_directory(operation)
    _ensure_directory(directory)
    target = directory / f"{digest}.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags, 0o444)
    except FileExistsError:
        existing = _read_regular_file(
            target, maximum=MAX_ARTIFACT_BYTES, label="existing CALPHAD evidence"
        )
        if existing != payload:
            raise TypedCalphadError("content-addressed CALPHAD evidence collision")
    except OSError as exc:
        raise TypedCalphadError("could not create CALPHAD evidence artifact") from exc
    else:
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise TypedCalphadError("could not finish CALPHAD evidence artifact")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return {
        "path": f"/outputs/calphad/{directory.name}/{digest}.json",
        "sha256": digest,
        "size_bytes": len(payload),
    }


def _evidence_request(normalized: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in normalized.items() if key not in {"database", "schema_version"}
    }


def _execution_contract(normalized: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "interface": "fixed ultra_deepagents.materials.calphad public surface",
        "caller_code_accepted": False,
        "caller_models_or_solver_options_accepted": False,
        "network": "none",
        "no_new_privileges": True,
        "read_only_root_filesystem": True,
        "cap_drop_all": True,
        "cpus_at_most": TYPED_SANDBOX_MAX_CPUS,
        "memory_bytes_at_most": TYPED_SANDBOX_MAX_MEMORY_BYTES,
        "pids_at_most": TYPED_SANDBOX_MAX_PIDS,
        "runtime_image_id": normalized["runtime_image_id"],
        "max_components": MAX_TYPED_COMPONENTS,
        "max_phases": MAX_TYPED_PHASES,
        "max_axis_values": MAX_TYPED_AXIS_VALUES,
        "max_grid_points": MAX_TYPED_GRID_POINTS,
        "wall_time_seconds": FIXED_WALL_TIME_SECONDS,
        "max_result_bytes": FIXED_MAX_RESULT_BYTES,
    }


def _validation_persistence_contract() -> dict[str, Any]:
    return {
        "catalog_status": "pending",
        "catalog_metadata_updated": False,
        "mode": "immutable_per_run_evidence",
        "note": (
            "This evidence records one bounded runtime outcome for this run; it does not "
            "promote mutable owner metadata to server-verified catalog status."
        ),
    }


def _terminal_failure_result(
    normalized: Mapping[str, Any], failure: CalphadTerminalFailureError
) -> dict[str, Any]:
    if not valid_failure_tuple(
        status=failure.status,
        failure_domain=failure.failure_domain,
        failure_stage=failure.failure_stage,
        failure_code=failure.failure_code,
        solver_started=failure.solver_started,
        operation=str(normalized["operation"]),
    ):
        raise TypedCalphadError("inconsistent typed CALPHAD terminal outcome")
    outcome = {
        "status": failure.status,
        "failure_domain": failure.failure_domain,
        "failure_stage": failure.failure_stage,
        "failure_code": failure.failure_code,
        "exit_code": failure.exit_code,
        "solver_started": failure.solver_started,
    }
    evidence = {
        "schema_version": FAILURE_EVIDENCE_SCHEMA_VERSION,
        "operation": normalized["operation"],
        "database_binding": _binding_identity(normalized["database"]),
        "request": _evidence_request(normalized),
        "outcome": outcome,
        "execution_contract": _execution_contract(normalized),
        "validation_persistence": _validation_persistence_contract(),
    }
    artifact = _write_content_addressed(str(normalized["operation"]), evidence)
    return {
        "ok": False,
        "operation": normalized["operation"],
        "error": failure.failure_code,
        "status": failure.status,
        "artifact": artifact,
        "outcome": outcome,
        "inspection_artifact_sha256": normalized.get("inspection_artifact_sha256"),
    }


def _execute_normalized_request(normalized: dict[str, Any]) -> dict[str, Any]:
    database = normalized["database"]
    path, runtime_kwargs = _runtime_kwargs(database)
    selection = normalized["selection"]
    operation = normalized["operation"]
    if operation == "inspect":
        try:
            result = inspect_calphad_input(
                path,
                components=selection["components"],
                phases=selection["phases"],
                **runtime_kwargs,
            )
        except CalphadUnsupportedError as exc:
            raise CalphadTerminalFailureError(
                "unsupported", "input", "parse", "calphad_parse_unsupported", 2, False
            ) from exc
        except CalphadTimeoutError as exc:
            raise CalphadTerminalFailureError(
                "timeout", "scientific", "parse", "calphad_parse_timeout", 124, False
            ) from exc
        except CalphadInputError as exc:
            raise CalphadTerminalFailureError(
                "failed", "input", "parse", "calphad_parse_failed", 2, False
            ) from exc
        except CalphadExecutionError as exc:
            raise CalphadTerminalFailureError(
                "failed", "scientific", "parse", "calphad_parse_failed", 1, False
            ) from exc
        except Exception as exc:
            raise CalphadTerminalFailureError(
                "failed",
                "platform",
                "parse",
                "calphad_runtime_internal_failure",
                1,
                False,
            ) from exc
        inspection_sha256 = None
    else:
        inspection_sha256 = normalized["inspection_artifact_sha256"]
        inspection = _inspection_evidence(inspection_sha256)
        components = selection["components"]
        phases = selection["phases"]
        assert isinstance(components, list) and isinstance(phases, list)
        _ensure_inventory_chain(
            inspection,
            database=database,
            components=components,
            phases=phases,
        )
        inspection_request = inspection.get("request")
        if (
            not isinstance(inspection_request, dict)
            or inspection_request.get("runtime_image_id") != normalized["runtime_image_id"]
        ):
            raise TypedCalphadError(
                "inspection evidence was produced by a different immutable runtime image"
            )
        conditions = normalized["conditions"]
        try:
            if operation == "equilibrium":
                result = run_calphad_equilibrium(
                    path,
                    components=components,
                    phases=phases,
                    temperatures_K=conditions["temperatures_K"],
                    pressures_Pa=conditions["pressures_Pa"],
                    total_amount_mol=1.0,
                    independent_compositions=conditions["independent_compositions"],
                    max_grid_points=MAX_TYPED_GRID_POINTS,
                    wall_time_seconds=FIXED_WALL_TIME_SECONDS,
                    max_result_bytes=FIXED_MAX_RESULT_BYTES,
                    **runtime_kwargs,
                )
            else:
                result = run_scheil_solidification(
                    path,
                    components=components,
                    phases=phases,
                    independent_composition=conditions["independent_composition_mole_fraction"],
                    start_temperature_K=conditions["start_temperature_K"],
                    step_temperature_K=conditions["step_temperature_K"],
                    pressure_Pa=conditions["pressure_Pa"],
                    liquid_phase_name=FIXED_SCHEIL_LIQUID_PHASE,
                    stop_liquid_fraction=conditions["stop_liquid_fraction"],
                    max_steps=FIXED_SCHEIL_MAX_STEPS,
                    wall_time_seconds=FIXED_WALL_TIME_SECONDS,
                    max_result_bytes=FIXED_MAX_RESULT_BYTES,
                    **runtime_kwargs,
                )
        except CalphadUnsupportedError as exc:
            # The bounded runtime raises this type only while loading the
            # database/runtime, before its numerical solver entry point. Keep
            # this invariant paired with solver_started=false.
            raise CalphadTerminalFailureError(
                "unsupported",
                "scientific",
                "solver",
                "calphad_solver_unsupported",
                2,
                False,
            ) from exc
        except CalphadTimeoutError as exc:
            raise CalphadTerminalFailureError(
                "timeout", "scientific", "solver", "calphad_solver_timeout", 124, True
            ) from exc
        except CalphadInputError as exc:
            raise CalphadTerminalFailureError(
                "failed", "input", "solver", "calphad_solver_failed", 2, False
            ) from exc
        except CalphadExecutionError as exc:
            raise CalphadTerminalFailureError(
                "failed", "scientific", "solver", "calphad_solver_failed", 1, True
            ) from exc
        except Exception as exc:
            raise CalphadTerminalFailureError(
                "failed",
                "platform",
                "solver",
                "calphad_runtime_internal_failure",
                1,
                False,
            ) from exc

    try:
        _ensure_result_database_format(result, database=database, operation=operation)
    except Exception as exc:
        raise CalphadTerminalFailureError(
            "failed",
            "scientific",
            "result_validation",
            "calphad_result_invalid",
            1,
            operation in {"equilibrium", "scheil"},
        ) from exc

    evidence = {
        "schema_version": TOOL_EVIDENCE_SCHEMA_VERSION,
        "operation": operation,
        "database_binding": _binding_identity(database),
        "request": _evidence_request(normalized),
        "result": result,
        "execution_contract": _execution_contract(normalized),
        "validation_persistence": _validation_persistence_contract(),
    }
    artifact = _write_content_addressed(normalized["operation"], evidence)
    return {
        "ok": True,
        "operation": normalized["operation"],
        "artifact": artifact,
        "inspection_artifact_sha256": inspection_sha256,
    }


def execute_request(request: Any) -> dict[str, Any]:
    """Validate and execute one already-decoded typed request.

    This function is kept separate from the CLI so pinned-pycalphad tests can
    exercise the exact scientific path without shelling out.
    """

    normalized = _validated_request(request)
    try:
        return _execute_normalized_request(normalized)
    except CalphadTerminalFailureError as exc:
        return _terminal_failure_result(normalized, exc)


def _emit(payload: Mapping[str, Any]) -> None:
    encoded = _canonical_json(dict(payload)).decode("utf-8")
    print(f"{RESULT_MARKER}{encoded}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ultra-calphad-typed")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        _emit(
            {
                "ok": True,
                "schema_version": REQUEST_SCHEMA_VERSION,
                "operations": ["inspect", "equilibrium", "scheil"],
                "caller_code_accepted": False,
            }
        )
        return 0
    if args.request is None:
        parser.error("--request is required")
    try:
        normalized = _request_from_path(args.request)
    except (TypedCalphadError, CalphadInputError):
        # No artifact is authored until a complete canonical typed request and
        # authorized database binding have both been established.
        _emit({"ok": False, "error": "invalid_calphad_request"})
        return 2
    try:
        result = _execute_normalized_request(normalized)
    except CalphadTerminalFailureError as exc:
        result = _terminal_failure_result(normalized, exc)
        _emit(result)
        return exc.exit_code if exc.exit_code is not None else 1
    except TypedCalphadError:
        # Inspection-lineage and other trusted pre-execution failures are request
        # rejection, not scientific terminal evidence.
        _emit({"ok": False, "error": "invalid_calphad_request"})
        return 2
    except Exception:
        _emit({"ok": False, "error": "calphad_internal_failure"})
        return 1
    _emit(result)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by sandbox integration
    sys.exit(main())
