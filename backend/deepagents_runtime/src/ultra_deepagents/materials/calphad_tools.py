"""Agent-facing typed CALPHAD tools backed by the immutable Docker sandbox.

These tools deliberately do not expose a generic command, Python, model,
parameter, or filesystem-path input.  They resolve a server-authorized resource
or the release-reviewed embedded registry, securely stage immutable bytes, and
invoke :mod:`ultra_deepagents.materials.calphad_cli` with a content-addressed
request.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import math
import os
import re
import shlex
import stat
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote

from langchain.tools import ToolRuntime, tool

from ultra_deepagents.code_execution.docker import DockerSandboxBackend
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.context_tools import (
    _find_uploaded_file,
    _resolved_upload_roots,
    _safe_upload_file_id,
    _selected_resource_binding,
)
from ultra_deepagents.materials.calphad import (
    CalphadInputError,
    canonicalize_equilibrium_compositions,
)
from ultra_deepagents.materials.calphad_cli import (
    FAILURE_CODES,
    FAILURE_DOMAINS,
    FAILURE_EVIDENCE_SCHEMA_VERSION,
    FAILURE_STAGES,
    FAILURE_STATUSES,
    FIXED_MAX_RESULT_BYTES,
    FIXED_SCHEIL_LIQUID_PHASE,
    FIXED_SCHEIL_MAX_STEPS,
    FIXED_SCHEIL_PRESSURE_PA,
    FIXED_WALL_TIME_SECONDS,
    MAX_ARTIFACT_BYTES,
    MAX_REQUEST_BYTES,
    MAX_TYPED_AXIS_VALUES,
    MAX_TYPED_COMPONENTS,
    MAX_TYPED_GRID_POINTS,
    MAX_TYPED_PHASES,
    REQUEST_SCHEMA_VERSION,
    RESULT_MARKER,
    TOOL_EVIDENCE_SCHEMA_VERSION,
    TYPED_SANDBOX_MAX_CPUS,
    TYPED_SANDBOX_MAX_MEMORY_BYTES,
    TYPED_SANDBOX_MAX_PIDS,
    valid_failure_tuple,
)
from ultra_deepagents.materials.processing_kinetics import (
    MAX_STEP_TEMPERATURE_K,
    MAX_STOP_LIQUID_FRACTION,
    MIN_STEP_TEMPERATURE_K,
    MIN_STOP_LIQUID_FRACTION,
    QUALIFIED_SCHEIL_VERSION,
    SCHEIL_ASSUMPTIONS,
    SCHEIL_MASS_BALANCE_TOLERANCE,
    SCHEIL_SCHEMA_VERSION,
)

_MAX_DATABASE_BYTES = 64 * 1024 * 1024
_MAX_MODEL_SUMMARY_BYTES = 64 * 1024
_MAX_SUMMARY_INVENTORY_NAMES = 256
_MAX_SUMMARY_POINTS = 16
_MAX_SUMMARY_PHASES_PER_POINT = 16
_MAX_RETAINED_INSPECTION_COMPONENTS = 256
_MAX_RETAINED_INSPECTION_PHASES = 2048
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NAME_RE = re.compile(r"^[A-Za-z0-9_+./:-]{1,128}$")
_TRUSTED_RUNTIME_ROOT = Path("/opt/ultra-runtime")
_CALPHAD_LEDGER_TIMEOUT_SECONDS = 30.0
_MAX_CALPHAD_EVIDENCE_GZIP_BYTES = 11 * 1024 * 1024
_PINNED_PYCALPHAD_VERSION = "0.11.2"


class CalphadToolError(RuntimeError):
    """A typed CALPHAD tool failed before producing trusted evidence."""

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
    except (TypeError, ValueError) as exc:
        raise CalphadToolError("nonfinite_or_nonjson_input") from exc


def _safe_json_loads(payload: bytes, *, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise CalphadToolError("nonfinite_output", f"{label} contains {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CalphadToolError("invalid_backend_output", f"{label} has duplicate keys")
            result[key] = value
        return result

    try:
        return json.loads(
            payload,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalphadToolError("invalid_backend_output", f"{label} is not valid JSON") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _bounded_name_list(
    value: list[str] | None,
    *,
    label: str,
    maximum: int,
    allow_none: bool,
) -> list[str] | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, list) or not value or len(value) > maximum:
        raise CalphadToolError("invalid_typed_input", f"{label} must contain 1..{maximum} names")
    normalized: list[str] = []
    for raw in value:
        if not isinstance(raw, str):
            raise CalphadToolError("invalid_typed_input", f"{label} names must be strings")
        name = raw.strip().upper()
        if not _NAME_RE.fullmatch(name):
            raise CalphadToolError("invalid_typed_input", f"{label} contains an invalid name")
        normalized.append(name)
    if len(set(normalized)) != len(normalized):
        raise CalphadToolError("invalid_typed_input", f"{label} contains duplicates")
    return sorted(normalized)


def _bounded_axis(
    value: list[float],
    *,
    label: str,
    minimum: float,
    maximum: float,
) -> list[float]:
    if not isinstance(value, list) or not value or len(value) > MAX_TYPED_AXIS_VALUES:
        raise CalphadToolError(
            "invalid_typed_input",
            f"{label} must contain 1..{MAX_TYPED_AXIS_VALUES} values",
        )
    normalized: list[float] = []
    for raw in value:
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            raise CalphadToolError("invalid_typed_input", f"{label} must be numeric")
        number = float(raw)
        if not math.isfinite(number) or number < minimum or number > maximum:
            raise CalphadToolError("invalid_typed_input", f"{label} is outside its bound")
        normalized.append(0.0 if number == 0 else number)
    if len(set(normalized)) != len(normalized):
        raise CalphadToolError("invalid_typed_input", f"{label} contains duplicates")
    return sorted(normalized)


def _bounded_limits(
    value: Any,
    *,
    label: str,
    minimum: float,
    maximum: float,
    allow_fixed: bool,
) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise CalphadToolError("invalid_typed_input", f"{label} must contain two values")
    normalized: list[float] = []
    for raw in value:
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            raise CalphadToolError("invalid_typed_input", f"{label} must be numeric")
        number = float(raw)
        if not math.isfinite(number) or number < minimum or number > maximum:
            raise CalphadToolError("invalid_typed_input", f"{label} is outside its bound")
        normalized.append(number)
    if normalized[0] > normalized[1] or (not allow_fixed and normalized[0] == normalized[1]):
        qualifier = "nondecreasing" if allow_fixed else "strictly increasing"
        raise CalphadToolError("invalid_typed_input", f"{label} must be {qualifier}")
    return normalized


def _bounded_compositions(value: dict[str, list[float]]) -> dict[str, list[float]]:
    if not isinstance(value, dict) or len(value) > MAX_TYPED_COMPONENTS:
        raise CalphadToolError(
            "invalid_typed_input", "independent_compositions must be a bounded object"
        )
    result: dict[str, list[float]] = {}
    for raw_name, axis in value.items():
        name_list = _bounded_name_list(
            [raw_name], label="composition component", maximum=1, allow_none=False
        )
        assert name_list is not None
        name = name_list[0]
        if name in result:
            raise CalphadToolError(
                "invalid_typed_input", "independent_compositions contains duplicates"
            )
        result[name] = _bounded_axis(axis, label=f"X({name})", minimum=0.0, maximum=1.0)
    return dict(sorted(result.items()))


def _bounded_scalar(
    value: Any,
    *,
    label: str,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CalphadToolError("invalid_typed_input", f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < minimum or number > maximum:
        raise CalphadToolError("invalid_typed_input", f"{label} is outside its bound")
    return 0.0 if number == 0 else number


def _bounded_scalar_composition(value: dict[str, float]) -> dict[str, float]:
    if not isinstance(value, dict) or len(value) > MAX_TYPED_COMPONENTS:
        raise CalphadToolError(
            "invalid_typed_input",
            "independent_composition_mole_fraction must be a bounded object",
        )
    result: dict[str, float] = {}
    for raw_name, raw_value in value.items():
        names = _bounded_name_list(
            [raw_name], label="composition component", maximum=1, allow_none=False
        )
        assert names is not None
        name = names[0]
        if name in {"VA", "/-"}:
            raise CalphadToolError(
                "invalid_typed_input",
                "vacancy and pseudo-components cannot have bulk composition axes",
            )
        if name in result:
            raise CalphadToolError(
                "invalid_typed_input",
                "independent_composition_mole_fraction contains duplicates",
            )
        result[name] = _bounded_scalar(
            raw_value,
            label=f"X({name})",
            minimum=0.0,
            maximum=1.0,
        )
    return dict(sorted(result.items()))


def _safe_directory(root: Path, *parts: str) -> Path:
    current = root
    try:
        root_info = current.lstat()
    except FileNotFoundError:
        current.mkdir(parents=True, mode=0o755)
        root_info = current.lstat()
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise CalphadToolError("unsafe_staging_root")
    for part in parts:
        if not part or part in {".", ".."} or "/" in part or "\\" in part:
            raise CalphadToolError("unsafe_staging_path")
        candidate = current / part
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            candidate.mkdir(mode=0o755)
            info = candidate.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise CalphadToolError("unsafe_staging_path")
        current = candidate
    return current


def _read_source(path: Path) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise CalphadToolError("resource_file_unavailable") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise CalphadToolError("resource_not_regular_file")
    if before.st_size <= 0 or before.st_size > _MAX_DATABASE_BYTES:
        raise CalphadToolError("resource_size_out_of_bounds")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CalphadToolError("resource_secure_open_failed") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev,
            opened.st_ino,
        ) != (before.st_dev, before.st_ino):
            raise CalphadToolError("resource_changed_during_stage")
        remaining = opened.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise CalphadToolError("resource_changed_during_stage")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise CalphadToolError("resource_changed_during_stage")
        after = os.fstat(descriptor)
        if (after.st_size, after.st_mtime_ns) != (opened.st_size, opened.st_mtime_ns):
            raise CalphadToolError("resource_changed_during_stage")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _secure_write_new(path: Path, payload: bytes, *, mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, mode)
    except OSError as exc:
        raise CalphadToolError("secure_stage_write_failed") from exc
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise CalphadToolError("secure_stage_write_failed")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _owner_provenance(binding: Mapping[str, Any]) -> dict[str, Any]:
    metadata = binding.get("metadata")
    calphad = metadata.get("calphad") if isinstance(metadata, dict) else None
    if not isinstance(calphad, dict):
        raise CalphadToolError(
            "complete_resource_provenance_required",
            "CALPHAD owner declarations are missing",
        )
    source = str(calphad.get("source") or "").strip()
    license_id = str(calphad.get("license_id") or calphad.get("license_identifier") or "").strip()
    assessment_scope = str(calphad.get("assessment_scope") or "").strip()
    reference_state = str(calphad.get("reference_state") or "").strip()
    database_id = str(calphad.get("database_id") or binding.get("resource_id") or "").strip()
    temperature_limit_declaration = calphad.get("tdb_temperature_limits_K")
    pressure_limit_declaration = calphad.get("assessment_pressure_limits_Pa")
    if not all((source, license_id, assessment_scope, reference_state, database_id)):
        raise CalphadToolError(
            "complete_resource_provenance_required",
            "source, license_id, assessment_scope, reference_state, and database identity are required",
        )
    try:
        temperature_limits = _bounded_limits(
            temperature_limit_declaration,
            label="owner tdb_temperature_limits_K",
            minimum=1.0,
            maximum=10_000.0,
            allow_fixed=False,
        )
        pressure_limits = _bounded_limits(
            pressure_limit_declaration,
            label="owner assessment_pressure_limits_Pa",
            minimum=1e-9,
            maximum=1e12,
            allow_fixed=True,
        )
    except CalphadToolError as exc:
        raise CalphadToolError(
            "complete_resource_provenance_required",
            "tdb_temperature_limits_K and assessment_pressure_limits_Pa must be finite "
            "[minimum, maximum] ranges",
        ) from exc
    if calphad.get("declaration_authority") != "resource_owner":
        raise CalphadToolError(
            "complete_resource_provenance_required",
            "CALPHAD declarations are not labeled resource_owner",
        )
    return {
        "database_id": database_id,
        "source": source,
        "license_id": license_id,
        "assessment_scope": assessment_scope,
        "reference_state": reference_state,
        "temperature_limits_K": temperature_limits,
        "assessment_pressure_limits_Pa": pressure_limits,
        "declaration_authority": "resource_owner",
    }


def _descriptor_database_format(binding: Mapping[str, Any]) -> str:
    declared_format = binding.get("database_format")
    if declared_format not in {"tdb", "dat"}:
        raise CalphadToolError(
            "server_catalog_binding_required",
            "the server-selected resource descriptor requires database_format=tdb or dat",
        )
    original_name = binding.get("original_name")
    if not isinstance(original_name, str) or not original_name.strip():
        raise CalphadToolError(
            "unsupported_calphad_resource_format",
            "the server-selected resource descriptor requires an original .tdb or .dat name",
        )
    suffix = Path(original_name.strip()).suffix.casefold()
    if suffix not in {".tdb", ".dat"}:
        raise CalphadToolError(
            "unsupported_calphad_resource_format",
            "the server-selected resource descriptor must end in .tdb or .dat",
        )
    filename_format = suffix.removeprefix(".")
    if filename_format != declared_format:
        raise CalphadToolError(
            "calphad_descriptor_format_mismatch",
            "database_format does not match the server-selected original_name suffix",
        )
    return filename_format


def _complete_binding(binding: Mapping[str, Any], *, resource_id: str) -> dict[str, Any]:
    sha256 = str(binding.get("sha256") or "").strip().lower()
    size_bytes = binding.get("size_bytes")
    if not _SHA256_RE.fullmatch(sha256):
        raise CalphadToolError("server_catalog_binding_required", "catalog SHA-256 is missing")
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes <= 0
        or size_bytes > _MAX_DATABASE_BYTES
    ):
        raise CalphadToolError("server_catalog_binding_required", "catalog size is missing")
    if binding.get("binding_authority") != "control_resource_catalog":
        raise CalphadToolError("server_catalog_binding_required")
    binding_schema = str(binding.get("binding_schema") or "")
    if binding_schema not in {"ultra.selected_resource.v1", "ultra.catalog_resource.v1"}:
        raise CalphadToolError("server_catalog_binding_required")
    provenance = _owner_provenance(binding)
    database_format = _descriptor_database_format(binding)
    return {
        "kind": "resource",
        "database_id": provenance["database_id"],
        "resource_id": resource_id,
        "database_format": database_format,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "source": provenance["source"],
        "license_id": provenance["license_id"],
        "assessment_scope": provenance["assessment_scope"],
        "reference_state": provenance["reference_state"],
        "temperature_limits_K": provenance["temperature_limits_K"],
        "assessment_pressure_limits_Pa": provenance["assessment_pressure_limits_Pa"],
        "binding_schema": binding_schema,
        "binding_authority": "control_resource_catalog",
        "declaration_authority": "resource_owner",
    }


def _resource_binding(
    settings: Any,
    context: AgentRunContext,
    *,
    resource_id: str,
) -> dict[str, Any]:
    del settings
    if not resource_id or not _safe_upload_file_id(resource_id):
        raise CalphadToolError("invalid_resource_id")
    if resource_id not in set(context.selected_file_ids):
        raise CalphadToolError(
            "selected_resource_required",
            "CALPHAD resource_id must be explicitly selected when the run is created",
        )
    selected = _selected_resource_binding(context, file_id=resource_id)
    if not selected:
        raise CalphadToolError("server_catalog_binding_required")
    return _complete_binding(selected, resource_id=resource_id)


def _selected_resource_governance_scope(
    context: AgentRunContext,
    *,
    resource_id: str,
) -> str:
    resource_id = str(resource_id or "").strip()
    if resource_id not in set(context.selected_file_ids):
        raise CalphadToolError(
            "selected_resource_required",
            "CALPHAD resource_id must be explicitly selected when the run is created",
        )
    selected = _selected_resource_binding(context, file_id=resource_id)
    scope = selected.get("calphad_governance_scope") if selected else None
    if scope not in {"owner_validation", "read_only_usage"}:
        raise CalphadToolError(
            "server_catalog_binding_required",
            "CALPHAD governance scope is missing from the selected resource capability",
        )
    return str(scope)


def _stage_resource_database(
    backend: DockerSandboxBackend,
    *,
    upload_roots: tuple[Path, ...],
    binding: dict[str, Any],
) -> dict[str, Any]:
    source = _find_uploaded_file(binding["resource_id"], upload_roots)
    if source is None:
        raise CalphadToolError("resource_file_unavailable")
    expected_suffix = f".{binding['database_format']}"
    if source.suffix.casefold() != expected_suffix:
        raise CalphadToolError(
            "calphad_resource_format_mismatch",
            "the staged source suffix does not match the server-selected database format",
        )
    payload = _read_source(source)
    if len(payload) != binding["size_bytes"] or _sha256(payload) != binding["sha256"]:
        raise CalphadToolError(
            "resource_catalog_hash_mismatch",
            "resource bytes do not match the server-authored SHA-256/size",
        )
    stage_root = _safe_directory(backend.workspace_dir, ".ultra", "calphad", "staged")
    target_name = f"{binding['sha256']}{expected_suffix}"
    target = stage_root / target_name
    try:
        _secure_write_new(target, payload, mode=0o444)
    except CalphadToolError:
        existing = _read_source(target)
        if existing != payload:
            raise
    staged = dict(binding)
    staged["path"] = f"/workspace/.ultra/calphad/staged/{target_name}"
    return staged


def _database_request(
    settings: Any,
    context: AgentRunContext,
    backend: DockerSandboxBackend,
    *,
    upload_roots: tuple[Path, ...],
    resource_id: str,
    embedded_database_id: str,
) -> dict[str, Any]:
    resource_id = str(resource_id or "").strip()
    embedded_database_id = str(embedded_database_id or "").strip()
    if bool(resource_id) == bool(embedded_database_id):
        raise CalphadToolError(
            "exactly_one_database_source_required",
            "set exactly one of resource_id or embedded_database_id",
        )
    if embedded_database_id:
        if len(embedded_database_id) > 512 or any(
            ord(character) < 32 or ord(character) == 127 for character in embedded_database_id
        ):
            raise CalphadToolError("invalid_embedded_database_id")
        return {"kind": "embedded", "database_id": embedded_database_id}
    binding = _resource_binding(settings, context, resource_id=resource_id)
    return _stage_resource_database(backend, upload_roots=upload_roots, binding=binding)


def _write_request(backend: DockerSandboxBackend, request: Mapping[str, Any]) -> str:
    payload = _canonical_json(request)
    if len(payload) <= 0 or len(payload) > MAX_REQUEST_BYTES:
        raise CalphadToolError("typed_request_too_large")
    digest = _sha256(payload)
    request_root = _safe_directory(backend.workspace_dir, ".ultra", "calphad", "requests")
    target = request_root / f"{digest}.json"
    try:
        _secure_write_new(target, payload, mode=0o444)
    except CalphadToolError:
        # Identical content-addressed requests can be repeated. Existing bytes
        # are accepted only after a no-follow re-read and exact comparison.
        existing = _read_source(target)
        if len(existing) > MAX_REQUEST_BYTES or existing != payload:
            raise
    return f"/workspace/.ultra/calphad/requests/{digest}.json"


def _expected_execution_contract(request: Mapping[str, Any]) -> dict[str, Any]:
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
        "runtime_image_id": request["runtime_image_id"],
        "max_components": MAX_TYPED_COMPONENTS,
        "max_phases": MAX_TYPED_PHASES,
        "max_axis_values": MAX_TYPED_AXIS_VALUES,
        "max_grid_points": MAX_TYPED_GRID_POINTS,
        "wall_time_seconds": FIXED_WALL_TIME_SECONDS,
        "max_result_bytes": FIXED_MAX_RESULT_BYTES,
    }


def _expected_validation_persistence() -> dict[str, Any]:
    return {
        "catalog_status": "pending",
        "catalog_metadata_updated": False,
        "mode": "immutable_per_run_evidence",
        "note": (
            "This evidence records one bounded runtime outcome for this run; it does not "
            "promote mutable owner metadata to server-verified catalog status."
        ),
    }


def _operation_directory_name(operation: str) -> str:
    directories = {
        "inspect": "inspection",
        "equilibrium": "equilibrium",
        "scheil": "scheil",
    }
    try:
        return directories[operation]
    except KeyError as exc:
        raise CalphadToolError("invalid_backend_output") from exc


def _write_host_failure_artifact(
    backend: DockerSandboxBackend,
    request: Mapping[str, Any],
    *,
    status: str,
    failure_code: str,
    exit_code: int | None,
) -> dict[str, Any]:
    """Author bounded host evidence when the sandbox cannot emit a trusted marker."""

    if backend.outputs_dir is None:
        raise CalphadToolError("outputs_mount_required")
    if (
        status not in {"failed", "timeout"}
        or failure_code not in {"calphad_sandbox_failed", "calphad_sandbox_timeout"}
        or isinstance(exit_code, bool)
        or (exit_code is not None and not isinstance(exit_code, int))
    ):
        raise CalphadToolError("invalid_backend_output")
    operation = str(request.get("operation") or "")
    if operation not in {"inspect", "equilibrium", "scheil"}:
        raise CalphadToolError("invalid_backend_output")
    expected_request = {
        key: value for key, value in request.items() if key not in {"database", "schema_version"}
    }
    outcome = {
        "status": status,
        "failure_domain": "platform",
        "failure_stage": "sandbox_runtime",
        "failure_code": failure_code,
        "exit_code": exit_code,
        "solver_started": False,
    }
    evidence = {
        "schema_version": FAILURE_EVIDENCE_SCHEMA_VERSION,
        "operation": operation,
        "database_binding": _expected_database_binding(request.get("database", {})),
        "request": expected_request,
        "outcome": outcome,
        "execution_contract": _expected_execution_contract(request),
        "validation_persistence": _expected_validation_persistence(),
    }
    payload = _canonical_json(evidence)
    if not payload or len(payload) > MAX_ARTIFACT_BYTES:
        raise CalphadToolError("typed_failure_evidence_too_large")
    digest = _sha256(payload)
    directory_name = _operation_directory_name(operation)
    directory = _safe_directory(backend.outputs_dir, "calphad", directory_name)
    target = directory / f"{digest}.json"
    try:
        _secure_write_new(target, payload, mode=0o444)
    except CalphadToolError:
        existing = _read_source(target)
        if existing != payload:
            raise
    artifact = {
        "path": f"/outputs/calphad/{directory_name}/{digest}.json",
        "sha256": digest,
        "size_bytes": len(payload),
    }
    return {
        "ok": False,
        "operation": operation,
        "error": failure_code,
        "status": status,
        "artifact": artifact,
        "outcome": outcome,
        "inspection_artifact_sha256": request.get("inspection_artifact_sha256"),
    }


def _parse_backend_result(
    response: Any,
    *,
    backend: DockerSandboxBackend,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    exit_code = getattr(response, "exit_code", None)
    if isinstance(exit_code, bool) or (exit_code is not None and not isinstance(exit_code, int)):
        exit_code = None
    output = str(getattr(response, "output", "") or "")
    marker_lines = [line for line in output.splitlines() if line.startswith(RESULT_MARKER)]

    # A complete typed request and authorized binding already exist at this
    # point. If the outer sandbox kills the process before a unique trusted
    # marker, the host is the authority for the canonical platform timeout.
    if getattr(response, "truncated", False) or len(marker_lines) != 1:
        is_timeout = exit_code == 124
        return _write_host_failure_artifact(
            backend,
            request,
            status="timeout" if is_timeout else "failed",
            failure_code=("calphad_sandbox_timeout" if is_timeout else "calphad_sandbox_failed"),
            exit_code=exit_code,
        )

    raw = marker_lines[0][len(RESULT_MARKER) :].encode("utf-8")
    try:
        result = _safe_json_loads(raw, label="sandbox result")
    except CalphadToolError:
        is_timeout = exit_code == 124
        return _write_host_failure_artifact(
            backend,
            request,
            status="timeout" if is_timeout else "failed",
            failure_code=("calphad_sandbox_timeout" if is_timeout else "calphad_sandbox_failed"),
            exit_code=exit_code,
        )
    if not isinstance(result, dict):
        return _write_host_failure_artifact(
            backend,
            request,
            status="timeout" if exit_code == 124 else "failed",
            failure_code=(
                "calphad_sandbox_timeout" if exit_code == 124 else "calphad_sandbox_failed"
            ),
            exit_code=exit_code,
        )
    if result.get("ok") is True:
        if exit_code != 0:
            return _write_host_failure_artifact(
                backend,
                request,
                status="timeout" if exit_code == 124 else "failed",
                failure_code=(
                    "calphad_sandbox_timeout" if exit_code == 124 else "calphad_sandbox_failed"
                ),
                exit_code=exit_code,
            )
        return result
    outcome = result.get("outcome")
    if (
        isinstance(result.get("artifact"), dict)
        and isinstance(outcome, dict)
        and result.get("status") in FAILURE_STATUSES
        and outcome.get("exit_code") == exit_code
    ):
        return result
    return _write_host_failure_artifact(
        backend,
        request,
        status="timeout" if exit_code == 124 else "failed",
        failure_code=("calphad_sandbox_timeout" if exit_code == 124 else "calphad_sandbox_failed"),
        exit_code=exit_code,
    )


def _artifact_host_path(backend: DockerSandboxBackend, artifact: Mapping[str, Any]) -> Path:
    if backend.outputs_dir is None:
        raise CalphadToolError("outputs_mount_required")
    path = str(artifact.get("path") or "")
    sha256 = str(artifact.get("sha256") or "").lower()
    size_bytes = artifact.get("size_bytes")
    if not _SHA256_RE.fullmatch(sha256):
        raise CalphadToolError("invalid_artifact_binding")
    expected_path = re.fullmatch(
        rf"/outputs/calphad/(inspection|equilibrium|scheil)/({sha256})\.json", path
    )
    if expected_path is None:
        raise CalphadToolError("invalid_artifact_binding")
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
        raise CalphadToolError("invalid_artifact_binding")
    current = backend.outputs_dir
    try:
        root_info = current.lstat()
    except OSError as exc:
        raise CalphadToolError("invalid_artifact_binding") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise CalphadToolError("invalid_artifact_binding")
    for part in ("calphad", expected_path.group(1)):
        current = current / part
        try:
            info = current.lstat()
        except OSError as exc:
            raise CalphadToolError("invalid_artifact_binding") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise CalphadToolError("invalid_artifact_binding")
    return current / f"{sha256}.json"


def _expected_database_binding(database: Mapping[str, Any]) -> dict[str, Any]:
    if database.get("kind") == "embedded":
        return {"kind": "embedded", "database_id": database.get("database_id")}
    return {
        key: database.get(key)
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


def _retained_inventory_names(value: Any, *, label: str, maximum: int) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > maximum:
        raise CalphadToolError("invalid_inspection_artifact", f"{label} is not bounded")
    names: list[str] = []
    for raw_name in value:
        if not isinstance(raw_name, str):
            raise CalphadToolError("invalid_inspection_artifact", f"{label} is not textual")
        name = raw_name.strip().upper()
        if name != raw_name or not _NAME_RE.fullmatch(name):
            raise CalphadToolError("invalid_inspection_artifact", f"{label} is not canonical")
        names.append(name)
    if names != sorted(names) or len(set(names)) != len(names):
        raise CalphadToolError("invalid_inspection_artifact", f"{label} is not canonical")
    return names


def _retained_inspection_inventory(
    backend: DockerSandboxBackend,
    *,
    inspection_sha256: str,
    database: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    """Authenticate one retained inspection artifact before equilibrium hashing."""

    if backend.outputs_dir is None:
        raise CalphadToolError("outputs_mount_required")
    candidate = backend.outputs_dir / "calphad" / "inspection" / f"{inspection_sha256}.json"
    try:
        candidate_info = candidate.lstat()
    except OSError as exc:
        raise CalphadToolError("inspection_artifact_unavailable") from exc
    artifact = {
        "path": f"/outputs/calphad/inspection/{inspection_sha256}.json",
        "sha256": inspection_sha256,
        "size_bytes": candidate_info.st_size,
    }
    try:
        host_path = _artifact_host_path(backend, artifact)
        payload = _read_source(host_path)
    except CalphadToolError as exc:
        raise CalphadToolError("invalid_inspection_artifact") from exc
    if (
        len(payload) != candidate_info.st_size
        or len(payload) > MAX_ARTIFACT_BYTES
        or _sha256(payload) != inspection_sha256
    ):
        raise CalphadToolError("inspection_artifact_hash_mismatch")
    try:
        evidence = _safe_json_loads(payload, label="retained CALPHAD inspection evidence")
    except CalphadToolError as exc:
        raise CalphadToolError("invalid_inspection_artifact") from exc
    if not isinstance(evidence, dict) or set(evidence) != {
        "schema_version",
        "operation",
        "database_binding",
        "request",
        "result",
        "execution_contract",
        "validation_persistence",
    }:
        raise CalphadToolError("invalid_inspection_artifact")
    if (
        evidence.get("schema_version") != TOOL_EVIDENCE_SCHEMA_VERSION
        or evidence.get("operation") != "inspect"
    ):
        raise CalphadToolError("invalid_inspection_artifact")
    if evidence.get("database_binding") != _expected_database_binding(database):
        raise CalphadToolError("inspection_artifact_binding_mismatch")
    request = evidence.get("request")
    if not isinstance(request, dict) or set(request) != {
        "operation",
        "runtime_image_id",
        "selection",
    }:
        raise CalphadToolError("invalid_inspection_artifact")
    if request.get("operation") != "inspect":
        raise CalphadToolError("invalid_inspection_artifact")
    if request.get("runtime_image_id") != backend.config.image:
        raise CalphadToolError("inspection_artifact_runtime_mismatch")
    selection = request.get("selection")
    if not isinstance(selection, dict) or set(selection) != {"components", "phases"}:
        raise CalphadToolError("invalid_inspection_artifact")
    normalized_selection: dict[str, list[str] | None] = {}
    for label, maximum in (
        ("components", MAX_TYPED_COMPONENTS),
        ("phases", MAX_TYPED_PHASES),
    ):
        raw_names = selection[label]
        try:
            normalized_names = _bounded_name_list(
                raw_names,
                label=f"inspection {label}",
                maximum=maximum,
                allow_none=True,
            )
        except CalphadToolError as exc:
            raise CalphadToolError("invalid_inspection_artifact") from exc
        if normalized_names != raw_names:
            raise CalphadToolError("invalid_inspection_artifact")
        normalized_selection[label] = normalized_names
    inspection_request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "operation": "inspect",
        "runtime_image_id": backend.config.image,
        "database": dict(database),
        "selection": normalized_selection,
    }
    if (
        evidence.get("execution_contract") != _expected_execution_contract(inspection_request)
        or evidence.get("validation_persistence") != _expected_validation_persistence()
    ):
        raise CalphadToolError("invalid_inspection_artifact")
    try:
        _verify_result_database_format(evidence, operation="inspect", database=database)
    except CalphadToolError as exc:
        raise CalphadToolError("invalid_inspection_artifact") from exc
    result = evidence.get("result")
    if (
        not isinstance(result, dict)
        or result.get("schema_version") != "1"
        or result.get("pycalphad_version") != _PINNED_PYCALPHAD_VERSION
    ):
        raise CalphadToolError("invalid_inspection_artifact")
    available_components = _retained_inventory_names(
        result.get("available_components"),
        label="available_components",
        maximum=_MAX_RETAINED_INSPECTION_COMPONENTS,
    )
    available_phases = _retained_inventory_names(
        result.get("available_phases"),
        label="available_phases",
        maximum=_MAX_RETAINED_INSPECTION_PHASES,
    )
    if result.get("components") != available_components or result.get("phases") != available_phases:
        raise CalphadToolError("invalid_inspection_artifact")
    if database.get("kind") == "resource":
        if (
            result.get("sha256") != database.get("sha256")
            or result.get("size_bytes") != database.get("size_bytes")
            or result.get("artifact_id") != database.get("resource_id")
        ):
            raise CalphadToolError("inspection_artifact_binding_mismatch")
    else:
        registry = result.get("registry_manifest")
        if not isinstance(registry, dict) or registry.get("database_id") != database.get(
            "database_id"
        ):
            raise CalphadToolError("inspection_artifact_binding_mismatch")
    return available_components, available_phases


def _verify_result_database_format(
    evidence: Mapping[str, Any],
    *,
    operation: str,
    database: Mapping[str, Any],
) -> None:
    result = evidence.get("result")
    if not isinstance(result, dict):
        raise CalphadToolError("invalid_artifact_evidence")
    manifest: Any = result if operation == "inspect" else result.get("database")
    if not isinstance(manifest, dict):
        raise CalphadToolError("invalid_artifact_evidence")
    result_format = manifest.get("format")
    manifest_path = manifest.get("path")
    if (
        result_format not in {"tdb", "dat"}
        or not isinstance(manifest_path, str)
        or Path(manifest_path).suffix.casefold() != f".{result_format}"
    ):
        raise CalphadToolError("invalid_artifact_evidence")
    if database.get("kind") == "resource" and result_format != database.get("database_format"):
        raise CalphadToolError("artifact_request_binding_mismatch")


def _scheil_series(
    value: Any,
    *,
    field: str,
    count: int,
    allow_none: bool = False,
) -> list[float | None]:
    if not isinstance(value, list) or len(value) != count:
        raise CalphadToolError("invalid_artifact_evidence", f"{field} has the wrong length")
    result: list[float | None] = []
    for raw in value:
        if raw is None and allow_none:
            result.append(None)
            continue
        result.append(_finite_evidence_number(raw, field=field))
    return result


def _verify_scheil_runtime_result(
    runtime_result: Any,
    *,
    request: Mapping[str, Any],
) -> None:
    """Independently recheck bounded Scheil evidence at the trusted host boundary."""

    if not isinstance(runtime_result, dict) or set(runtime_result) != {
        "schema_version",
        "method",
        "database",
        "request",
        "result",
        "assumptions",
        "warnings",
        "solver",
        "units",
        "limits",
        "evidence",
    }:
        raise CalphadToolError("invalid_artifact_evidence")
    if (
        runtime_result.get("schema_version") != SCHEIL_SCHEMA_VERSION
        or runtime_result.get("method") != "Scheil-Gulliver"
        or runtime_result.get("assumptions") != list(SCHEIL_ASSUMPTIONS)
    ):
        raise CalphadToolError("invalid_artifact_evidence")
    warnings = runtime_result.get("warnings")
    if (
        not isinstance(warnings, list)
        or any(not isinstance(item, str) or not item for item in warnings)
        or not any("not a back-diffusion" in item for item in warnings)
        or not any("phase-field" in item for item in warnings)
    ):
        raise CalphadToolError("invalid_artifact_evidence")
    solver = runtime_result.get("solver")
    if not isinstance(solver, dict) or solver != {
        "name": "scheil",
        "version": QUALIFIED_SCHEIL_VERSION,
        "pycalphad_version": _PINNED_PYCALPHAD_VERSION,
        "adaptive_constitution_sampling": True,
        "replay_determinism_claimed": False,
    }:
        raise CalphadToolError("invalid_artifact_evidence")
    if runtime_result.get("units") != {
        "temperature": "K",
        "pressure": "Pa",
        "amount": "mol",
        "composition": "mole_fraction",
        "phase_fraction": "fraction_of_one_mole_basis",
    } or runtime_result.get("limits") != {
        "max_steps": FIXED_SCHEIL_MAX_STEPS,
        "wall_time_seconds": FIXED_WALL_TIME_SECONDS,
        "max_result_bytes": FIXED_MAX_RESULT_BYTES,
    }:
        raise CalphadToolError("invalid_artifact_evidence")
    inner_evidence = runtime_result.get("evidence")
    evidence_payload = dict(runtime_result)
    evidence_payload.pop("evidence", None)
    if not isinstance(inner_evidence, dict) or inner_evidence != {
        "sha256": _sha256(_canonical_json(evidence_payload)),
        "algorithm": "sha256",
        "canonicalization": "UTF-8 JSON, sorted keys, compact separators, finite numbers",
    }:
        raise CalphadToolError("invalid_artifact_evidence")

    selection = request.get("selection")
    typed_conditions = request.get("conditions")
    request_record = runtime_result.get("request")
    if (
        not isinstance(selection, dict)
        or not isinstance(typed_conditions, dict)
        or not isinstance(request_record, dict)
        or set(request_record)
        != {
            "components",
            "phases",
            "independent_composition_mole_fraction",
            "bulk_composition_mole_fraction",
            "dependent_component",
            "start_temperature_K",
            "step_temperature_K",
            "pressure_Pa",
            "total_amount_mol",
            "liquid_phase_name",
            "stop_liquid_fraction",
        }
    ):
        raise CalphadToolError("invalid_artifact_evidence")
    components = selection.get("components")
    phases = selection.get("phases")
    typed_composition = typed_conditions.get("independent_composition_mole_fraction")
    if (
        not isinstance(components, list)
        or not isinstance(phases, list)
        or not isinstance(typed_composition, dict)
    ):
        raise CalphadToolError("invalid_artifact_evidence")
    try:
        canonical_compositions, dependent_component, closure_records = (
            canonicalize_equilibrium_compositions(typed_composition, components=components)
        )
    except CalphadInputError as exc:
        raise CalphadToolError("invalid_artifact_evidence") from exc
    canonical_composition = {
        component: values[0] for component, values in canonical_compositions.items()
    }
    if (
        len(closure_records) != 1
        or request_record.get("components") != components
        or request_record.get("phases") != phases
        or request_record.get("independent_composition_mole_fraction") != canonical_composition
        or request_record.get("bulk_composition_mole_fraction") != closure_records[0]
        or request_record.get("dependent_component") != dependent_component
        or request_record.get("start_temperature_K") != typed_conditions.get("start_temperature_K")
        or request_record.get("step_temperature_K") != typed_conditions.get("step_temperature_K")
        or request_record.get("pressure_Pa") != FIXED_SCHEIL_PRESSURE_PA
        or typed_conditions.get("pressure_Pa") != FIXED_SCHEIL_PRESSURE_PA
        or request_record.get("total_amount_mol") != 1.0
        or request_record.get("liquid_phase_name") != FIXED_SCHEIL_LIQUID_PHASE
        or request_record.get("stop_liquid_fraction")
        != typed_conditions.get("stop_liquid_fraction")
    ):
        raise CalphadToolError("invalid_artifact_evidence")

    database_manifest = runtime_result.get("database")
    database_binding = request.get("database")
    if not isinstance(database_manifest, dict) or not isinstance(database_binding, dict):
        raise CalphadToolError("invalid_artifact_evidence")
    if database_manifest.get("pycalphad_version") != _PINNED_PYCALPHAD_VERSION:
        raise CalphadToolError("invalid_artifact_evidence")
    if database_binding.get("kind") == "resource":
        expected_database_fields = {
            "sha256": database_binding.get("sha256"),
            "size_bytes": database_binding.get("size_bytes"),
            "artifact_id": database_binding.get("resource_id"),
            "source": database_binding.get("source"),
            "license_id": database_binding.get("license_id"),
            "assessment_scope": database_binding.get("assessment_scope"),
            "reference_state": database_binding.get("reference_state"),
            "assessment_temperature_limits_K": database_binding.get("temperature_limits_K"),
            "assessment_pressure_limits_Pa": database_binding.get("assessment_pressure_limits_Pa"),
        }
        if any(
            database_manifest.get(field) != expected
            for field, expected in expected_database_fields.items()
        ):
            raise CalphadToolError("artifact_request_binding_mismatch")
    else:
        registry = database_manifest.get("registry_manifest")
        if not isinstance(registry, dict) or registry.get("database_id") != database_binding.get(
            "database_id"
        ):
            raise CalphadToolError("artifact_request_binding_mismatch")

    result_record = runtime_result.get("result")
    if not isinstance(result_record, dict) or set(result_record) != {
        "point_count",
        "temperatures_K",
        "fraction_solid",
        "fraction_liquid",
        "solid_phase_increment_fraction",
        "solid_phase_cumulative_fraction",
        "phase_composition_mole_fraction",
        "elemental_mass_balance",
        "converged",
        "qualified_terminal_point",
        "discarded_upstream_terminal_fill_point",
        "closure_tolerances",
    }:
        raise CalphadToolError("invalid_artifact_evidence")
    count = result_record.get("point_count")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not 2 <= count <= FIXED_SCHEIL_MAX_STEPS
    ):
        raise CalphadToolError("invalid_artifact_evidence")
    temperatures = _scheil_series(
        result_record.get("temperatures_K"), field="Scheil temperature", count=count
    )
    fraction_solid = _scheil_series(
        result_record.get("fraction_solid"), field="Scheil solid fraction", count=count
    )
    fraction_liquid = _scheil_series(
        result_record.get("fraction_liquid"), field="Scheil liquid fraction", count=count
    )
    assert all(value is not None for value in temperatures)
    assert all(value is not None for value in fraction_solid)
    assert all(value is not None for value in fraction_liquid)
    temperature_values = [float(value) for value in temperatures]
    solid_values = [float(value) for value in fraction_solid]
    liquid_values = [float(value) for value in fraction_liquid]
    assessment_limits = database_manifest.get("assessment_temperature_limits_K")
    if (
        any(right > left + 1e-10 for left, right in zip(temperature_values, temperature_values[1:]))
        or not isinstance(assessment_limits, list)
        or len(assessment_limits) != 2
        or min(temperature_values) < float(assessment_limits[0])
        or max(temperature_values) > float(assessment_limits[1])
        or not math.isclose(solid_values[0], 0.0, rel_tol=0.0, abs_tol=1e-8)
        or not math.isclose(liquid_values[0], 1.0, rel_tol=0.0, abs_tol=1e-8)
        or any(
            solid < -1e-8
            or solid > 1 + 1e-8
            or liquid < -1e-8
            or liquid > 1 + 1e-8
            or not math.isclose(solid + liquid, 1.0, rel_tol=0.0, abs_tol=1e-8)
            for solid, liquid in zip(solid_values, liquid_values)
        )
        or any(right + 1e-8 < left for left, right in zip(solid_values, solid_values[1:]))
        or liquid_values[-1] >= float(typed_conditions["stop_liquid_fraction"]) + 1e-8
        or result_record.get("converged") is not True
        or result_record.get("qualified_terminal_point") != "last_residual_liquid_point"
        or not isinstance(result_record.get("discarded_upstream_terminal_fill_point"), bool)
        or result_record.get("closure_tolerances")
        != {
            "phase_fraction_absolute": 1e-6,
            "composition_absolute": 1e-6,
            "elemental_mass_balance_absolute": SCHEIL_MASS_BALANCE_TOLERANCE,
        }
    ):
        raise CalphadToolError("invalid_artifact_evidence")

    increments_raw = result_record.get("solid_phase_increment_fraction")
    cumulative_raw = result_record.get("solid_phase_cumulative_fraction")
    compositions_raw = result_record.get("phase_composition_mole_fraction")
    if (
        not isinstance(increments_raw, dict)
        or not increments_raw
        or not isinstance(cumulative_raw, dict)
        or set(cumulative_raw) != set(increments_raw)
        or not isinstance(compositions_raw, dict)
        or FIXED_SCHEIL_LIQUID_PHASE not in compositions_raw
        or not set(increments_raw).issubset(compositions_raw)
        or not set(compositions_raw).issubset(phases)
    ):
        raise CalphadToolError("invalid_artifact_evidence")
    increment_series: dict[str, list[float]] = {}
    cumulative_series: dict[str, list[float]] = {}
    for phase in sorted(increments_raw):
        if phase == FIXED_SCHEIL_LIQUID_PHASE or phase not in phases:
            raise CalphadToolError("invalid_artifact_evidence")
        increments = _scheil_series(
            increments_raw[phase], field=f"Scheil {phase} increment", count=count
        )
        cumulative = _scheil_series(
            cumulative_raw[phase], field=f"Scheil {phase} cumulative amount", count=count
        )
        increment_values = [float(value) for value in increments if value is not None]
        cumulative_values = [float(value) for value in cumulative if value is not None]
        if (
            len(increment_values) != count
            or len(cumulative_values) != count
            or any(value < -1e-8 for value in [*increment_values, *cumulative_values])
            or not math.isclose(increment_values[0], 0.0, rel_tol=0.0, abs_tol=1e-8)
        ):
            raise CalphadToolError("invalid_artifact_evidence")
        running = 0.0
        for index, increment in enumerate(increment_values):
            running = math.fsum((running, increment))
            if not math.isclose(running, cumulative_values[index], rel_tol=0.0, abs_tol=1e-6):
                raise CalphadToolError("invalid_artifact_evidence")
        increment_series[phase] = increment_values
        cumulative_series[phase] = cumulative_values
    for index, solid in enumerate(solid_values):
        if not math.isclose(
            math.fsum(values[index] for values in cumulative_series.values()),
            solid,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise CalphadToolError("invalid_artifact_evidence")

    physical_components = sorted(
        component for component in components if component not in {"VA", "/-"}
    )
    phase_compositions: dict[str, dict[str, list[float | None]]] = {}
    for phase, raw_components in compositions_raw.items():
        if not isinstance(raw_components, dict) or set(raw_components) != set(physical_components):
            raise CalphadToolError("invalid_artifact_evidence")
        component_series = {
            component: _scheil_series(
                raw_components[component],
                field=f"Scheil {phase}/{component} composition",
                count=count,
                allow_none=True,
            )
            for component in physical_components
        }
        for index in range(count):
            values = [component_series[component][index] for component in physical_components]
            finite_values = [float(value) for value in values if value is not None]
            if finite_values and len(finite_values) != len(values):
                raise CalphadToolError("invalid_artifact_evidence")
            if finite_values and (
                any(value < -1e-8 or value > 1 + 1e-8 for value in finite_values)
                or not math.isclose(math.fsum(finite_values), 1.0, rel_tol=0.0, abs_tol=1e-6)
            ):
                raise CalphadToolError("invalid_artifact_evidence")
        phase_compositions[phase] = component_series

    bulk_composition = closure_records[0]
    running_solid_inventory = {component: 0.0 for component in physical_components}
    maximum_errors = {component: 0.0 for component in physical_components}
    final_reconstructed: dict[str, float] = {}
    for index in range(count):
        if index > 0:
            for phase, increments in increment_series.items():
                increment = increments[index]
                for component in physical_components:
                    value = phase_compositions[phase][component][index]
                    if value is None:
                        if increment > 1e-8:
                            raise CalphadToolError("invalid_artifact_evidence")
                        continue
                    running_solid_inventory[component] = math.fsum(
                        (running_solid_inventory[component], increment * value)
                    )
        reconstructed: dict[str, float] = {}
        for component in physical_components:
            if index == 0:
                reconstructed_value = bulk_composition[component]
            else:
                liquid_composition = phase_compositions[FIXED_SCHEIL_LIQUID_PHASE][component][index]
                if liquid_composition is None:
                    raise CalphadToolError("invalid_artifact_evidence")
                reconstructed_value = math.fsum(
                    (
                        liquid_values[index] * liquid_composition,
                        running_solid_inventory[component],
                    )
                )
            error = abs(reconstructed_value - bulk_composition[component])
            if error > SCHEIL_MASS_BALANCE_TOLERANCE:
                raise CalphadToolError("invalid_artifact_evidence")
            maximum_errors[component] = max(maximum_errors[component], error)
            reconstructed[component] = reconstructed_value
        if not math.isclose(math.fsum(reconstructed.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise CalphadToolError("invalid_artifact_evidence")
        final_reconstructed = reconstructed

    mass_balance = result_record.get("elemental_mass_balance")
    if not isinstance(mass_balance, dict) or set(mass_balance) != {
        "basis",
        "formula",
        "absolute_tolerance",
        "maximum_absolute_component_error",
        "maximum_absolute_error_by_component",
        "final_reconstructed_bulk_composition_mole_fraction",
        "all_retained_points_closed",
    }:
        raise CalphadToolError("invalid_artifact_evidence")
    reported_errors = mass_balance.get("maximum_absolute_error_by_component")
    reported_final = mass_balance.get("final_reconstructed_bulk_composition_mole_fraction")
    reported_maximum = _finite_evidence_number(
        mass_balance.get("maximum_absolute_component_error"),
        field="Scheil maximum mass-balance error",
    )
    if (
        mass_balance.get("basis") != "one_mole_initial_bulk"
        or mass_balance.get("absolute_tolerance") != SCHEIL_MASS_BALANCE_TOLERANCE
        or mass_balance.get("all_retained_points_closed") is not True
        or not isinstance(reported_errors, dict)
        or set(reported_errors) != set(physical_components)
        or not isinstance(reported_final, dict)
        or set(reported_final) != set(physical_components)
        or not math.isclose(
            reported_maximum,
            max(maximum_errors.values()),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or any(
            not math.isclose(
                _finite_evidence_number(reported_errors[component], field="Scheil mass error"),
                maximum_errors[component],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                _finite_evidence_number(
                    reported_final[component], field="Scheil reconstructed bulk composition"
                ),
                final_reconstructed[component],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for component in physical_components
        )
    ):
        raise CalphadToolError("invalid_artifact_evidence")


def _verified_artifact(
    backend: DockerSandboxBackend,
    result: Mapping[str, Any],
    *,
    operation: str,
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = result.get("artifact")
    if not isinstance(artifact, dict):
        raise CalphadToolError("invalid_artifact_binding")
    host_path = _artifact_host_path(backend, artifact)
    payload = _read_source(host_path)
    if (
        len(payload) != artifact["size_bytes"]
        or len(payload) > MAX_ARTIFACT_BYTES
        or _sha256(payload) != artifact["sha256"]
    ):
        raise CalphadToolError("artifact_hash_mismatch")
    evidence = _safe_json_loads(payload, label="CALPHAD evidence")
    if not isinstance(evidence, dict) or evidence.get("operation") != operation:
        raise CalphadToolError("invalid_artifact_evidence")
    expected_request = {
        key: value for key, value in request.items() if key not in {"database", "schema_version"}
    }
    if (
        evidence.get("database_binding") != _expected_database_binding(request.get("database", {}))
        or evidence.get("request") != expected_request
    ):
        raise CalphadToolError("artifact_request_binding_mismatch")
    schema_version = evidence.get("schema_version")
    if schema_version == FAILURE_EVIDENCE_SCHEMA_VERSION:
        if set(evidence) != {
            "schema_version",
            "operation",
            "database_binding",
            "request",
            "outcome",
            "execution_contract",
            "validation_persistence",
        }:
            raise CalphadToolError("invalid_artifact_evidence")
        outcome = evidence.get("outcome")
        if not isinstance(outcome, dict) or set(outcome) != {
            "status",
            "failure_domain",
            "failure_stage",
            "failure_code",
            "exit_code",
            "solver_started",
        }:
            raise CalphadToolError("invalid_artifact_evidence")
        exit_code = outcome.get("exit_code")
        if (
            outcome.get("status") not in FAILURE_STATUSES
            or outcome.get("failure_domain") not in FAILURE_DOMAINS
            or outcome.get("failure_stage") not in FAILURE_STAGES
            or outcome.get("failure_code") not in FAILURE_CODES
            or isinstance(exit_code, bool)
            or (exit_code is not None and not isinstance(exit_code, int))
            or not isinstance(outcome.get("solver_started"), bool)
            or result.get("ok") is not False
            or result.get("operation") != operation
            or result.get("status") != outcome.get("status")
            or result.get("error") != outcome.get("failure_code")
            or result.get("outcome") != outcome
            or not valid_failure_tuple(
                status=str(outcome.get("status") or ""),
                failure_domain=str(outcome.get("failure_domain") or ""),
                failure_stage=str(outcome.get("failure_stage") or ""),
                failure_code=str(outcome.get("failure_code") or ""),
                solver_started=outcome.get("solver_started"),
                operation=operation,
            )
            or evidence.get("execution_contract") != _expected_execution_contract(request)
            or evidence.get("validation_persistence") != _expected_validation_persistence()
        ):
            raise CalphadToolError("invalid_artifact_evidence")
        return dict(artifact), evidence
    if schema_version != TOOL_EVIDENCE_SCHEMA_VERSION or set(evidence) != {
        "schema_version",
        "operation",
        "database_binding",
        "request",
        "result",
        "execution_contract",
        "validation_persistence",
    }:
        raise CalphadToolError("invalid_artifact_evidence")
    _verify_result_database_format(
        evidence,
        operation=operation,
        database=request.get("database", {}),
    )
    if operation == "equilibrium":
        runtime_result = evidence.get("result")
        if (
            not isinstance(runtime_result, dict)
            or runtime_result.get("schema_version") != "ultra.calphad.equilibrium.v2"
        ):
            raise CalphadToolError("invalid_artifact_evidence")
    elif operation == "scheil":
        _verify_scheil_runtime_result(evidence.get("result"), request=request)
    persistence = evidence.get("validation_persistence")
    if not isinstance(persistence, dict) or persistence.get("catalog_status") != "pending":
        raise CalphadToolError("invalid_artifact_evidence")
    return dict(artifact), evidence


def _typed_cli_command(request_path: str) -> str:
    if not re.fullmatch(r"/workspace/\.ultra/calphad/requests/[0-9a-f]{64}\.json", request_path):
        raise CalphadToolError("unsafe_typed_request_path")
    bootstrap = (
        "import sys;"
        f"sys.path.insert(0,{str(_TRUSTED_RUNTIME_ROOT)!r});"
        "from ultra_deepagents.materials.calphad_cli import main;"
        "raise SystemExit(main(['--request',sys.argv[1]]))"
    )
    return f"python3 -I -c {shlex.quote(bootstrap)} {request_path}"


def _execute_typed_request(
    backend: DockerSandboxBackend,
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    request_path = _write_request(backend, request)
    # Isolated mode removes /workspace and PYTHONPATH from import resolution;
    # only the release-baked /opt runtime is inserted before importing the CLI.
    # request_path is SHA-derived and validated by _typed_cli_command.
    response = backend.execute(_typed_cli_command(request_path))
    result = _parse_backend_result(response, backend=backend, request=request)
    return _verified_artifact(
        backend,
        result,
        operation=str(request["operation"]),
        request=request,
    )


def _artifact_summary(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": artifact["path"],
        "sha256": artifact["sha256"],
        "size_bytes": artifact["size_bytes"],
        "content_addressed": True,
    }


def _failure_summary(artifact: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    outcome = evidence.get("outcome")
    if not isinstance(outcome, dict):
        raise CalphadToolError("invalid_artifact_evidence")
    operation = evidence.get("operation")
    if operation not in {"inspect", "equilibrium", "scheil"}:
        raise CalphadToolError("invalid_artifact_evidence")
    artifact_key = {
        "inspect": "inspection_artifact",
        "equilibrium": "equilibrium_artifact",
        "scheil": "scheil_artifact",
    }[operation]
    summary = {
        "ok": False,
        "operation": operation,
        "error": outcome.get("failure_code"),
        "status": outcome.get("status"),
        "failure_domain": outcome.get("failure_domain"),
        "failure_stage": outcome.get("failure_stage"),
        "failure_code": outcome.get("failure_code"),
        "exit_code": outcome.get("exit_code"),
        "solver_started": outcome.get("solver_started"),
        artifact_key: _artifact_summary(artifact),
        "artifact_created": True,
        "catalog_validation_status": "pending",
        "scientific_status": "unverified",
    }
    request = evidence.get("request")
    if operation in {"equilibrium", "scheil"} and isinstance(request, dict):
        summary["inspection_artifact_sha256"] = request.get("inspection_artifact_sha256")
    return _bounded_summary(summary)


def _bounded_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    encoded = _canonical_json(value)
    if len(encoded) > _MAX_MODEL_SUMMARY_BYTES:
        raise CalphadToolError("model_summary_bound_exceeded")
    return dict(value)


def _inspect_summary(artifact: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    result = evidence.get("result")
    if not isinstance(result, dict):
        raise CalphadToolError("invalid_artifact_evidence")
    components = list(result.get("available_components") or result.get("components") or [])
    phases = list(result.get("available_phases") or result.get("phases") or [])
    if not all(isinstance(name, str) for name in [*components, *phases]):
        raise CalphadToolError("invalid_artifact_evidence")
    binding = evidence.get("database_binding")
    summary = {
        "ok": True,
        "operation": "inspect",
        "database": {
            "kind": binding.get("kind") if isinstance(binding, dict) else None,
            "database_id": binding.get("database_id") if isinstance(binding, dict) else None,
            "resource_id": binding.get("resource_id") if isinstance(binding, dict) else None,
            "database_format": binding.get("database_format")
            if isinstance(binding, dict)
            else None,
            "sha256": result.get("sha256"),
            "size_bytes": result.get("size_bytes"),
            "pycalphad_version": result.get("pycalphad_version"),
            "assessment_temperature_limits_K": result.get("assessment_temperature_limits_K"),
            "assessment_pressure_limits_Pa": result.get("assessment_pressure_limits_Pa"),
        },
        "inventory": {
            "component_count": len(components),
            "components": components[:_MAX_SUMMARY_INVENTORY_NAMES],
            "components_truncated": len(components) > _MAX_SUMMARY_INVENTORY_NAMES,
            "phase_count": len(phases),
            "phases": phases[:_MAX_SUMMARY_INVENTORY_NAMES],
            "phases_truncated": len(phases) > _MAX_SUMMARY_INVENTORY_NAMES,
            "parameter_count": result.get("parameter_count"),
        },
        "inspection_artifact": _artifact_summary(artifact),
        "catalog_validation_status": "pending",
        "scientific_status": "unverified",
        "next_step": (
            "Pass inspection_artifact.sha256, a typed component subset, a typed phase subset, "
            "and bounded conditions to calphad_run_equilibrium or calphad_run_scheil."
        ),
    }
    return _bounded_summary(summary)


def _stable_phase_summary(phase: Mapping[str, Any]) -> dict[str, Any]:
    summary = {"name": phase.get("name")}
    if "NP_phase_fraction" in phase:
        summary["NP_phase_fraction"] = phase["NP_phase_fraction"]
    elif "NP_mole_fraction" in phase:  # compatibility with pre-v2 evidence
        summary["NP_phase_fraction"] = phase["NP_mole_fraction"]
    return summary


def _finite_evidence_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise CalphadToolError("invalid_artifact_evidence", f"{field} must be finite")
    return float(value)


def _equilibrium_summary(
    artifact: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    result = evidence.get("result")
    if not isinstance(result, dict):
        raise CalphadToolError("invalid_artifact_evidence")
    result_record = result.get("result")
    request_record = result.get("request")
    if not isinstance(result_record, dict) or not isinstance(request_record, dict):
        raise CalphadToolError("invalid_artifact_evidence")
    requested_components = request_record.get("components")
    dependent_component = request_record.get("dependent_component")
    request_conditions = request_record.get("conditions")
    if (
        not isinstance(requested_components, list)
        or not requested_components
        or any(not isinstance(component, str) for component in requested_components)
        or not isinstance(dependent_component, str)
        or not isinstance(request_conditions, dict)
    ):
        raise CalphadToolError("invalid_artifact_evidence")
    physical_components = sorted(
        component for component in requested_components if component not in {"VA", "/-"}
    )
    raw_independent_compositions = request_conditions.get("independent_compositions")
    if (
        not physical_components
        or dependent_component != physical_components[0]
        or not isinstance(raw_independent_compositions, dict)
        or set(raw_independent_compositions) != set(physical_components[1:])
    ):
        raise CalphadToolError("invalid_artifact_evidence")
    independent_composition_axes: dict[str, list[float]] = {}
    for component, record in raw_independent_compositions.items():
        if (
            not isinstance(record, dict)
            or record.get("units") != "mole_fraction"
            or not isinstance(record.get("values"), list)
            or not record["values"]
        ):
            raise CalphadToolError("invalid_artifact_evidence")
        independent_composition_axes[component] = [
            _finite_evidence_number(value, field="independent composition")
            for value in record["values"]
        ]
    points = result_record.get("points")
    if not isinstance(points, list):
        raise CalphadToolError("invalid_artifact_evidence")
    point_summaries: list[dict[str, Any]] = []
    stable_names: set[str] = set()
    for point_index, point in enumerate(points):
        if not isinstance(point, dict):
            raise CalphadToolError("invalid_artifact_evidence")
        required_point_fields = {
            "conditions",
            "stable_phases",
            "stable_phase_vertices",
            "phase_fraction_sum",
            "bulk_composition_residual_by_component",
            "maximum_bulk_composition_residual",
            "GM_J_per_mol",
            "chemical_potentials_J_per_mol",
            "gibbs_from_chemical_potentials_J_per_mol",
            "gibbs_euler_residual_J_per_mol",
        }
        if not required_point_fields.issubset(point):
            raise CalphadToolError("invalid_artifact_evidence")
        raw_phases = point.get("stable_phases")
        raw_vertices = point.get("stable_phase_vertices")
        if (
            not isinstance(raw_phases, list)
            or not raw_phases
            or not isinstance(raw_vertices, list)
            or not raw_vertices
        ):
            raise CalphadToolError("invalid_artifact_evidence")
        residuals = point.get("bulk_composition_residual_by_component")
        chemical_potentials = point.get("chemical_potentials_J_per_mol")
        if not isinstance(residuals, dict) or not isinstance(chemical_potentials, dict):
            raise CalphadToolError("invalid_artifact_evidence")
        normalized_residuals = {
            str(component): _finite_evidence_number(value, field="bulk composition residual")
            for component, value in residuals.items()
        }
        normalized_chemical_potentials = {
            str(component): _finite_evidence_number(value, field="chemical potential")
            for component, value in chemical_potentials.items()
        }
        phase_fraction_sum = _finite_evidence_number(
            point.get("phase_fraction_sum"), field="phase fraction sum"
        )
        gm_value = _finite_evidence_number(point.get("GM_J_per_mol"), field="molar Gibbs energy")
        maximum_residual = _finite_evidence_number(
            point.get("maximum_bulk_composition_residual"),
            field="maximum bulk composition residual",
        )
        gibbs_from_mu = _finite_evidence_number(
            point.get("gibbs_from_chemical_potentials_J_per_mol"),
            field="Gibbs energy reconstructed from chemical potentials",
        )
        gibbs_euler_residual = _finite_evidence_number(
            point.get("gibbs_euler_residual_J_per_mol"),
            field="Gibbs-Euler residual",
        )
        phase_fraction_values: list[float] = []
        for phase in raw_phases:
            if not isinstance(phase, dict) or not isinstance(phase.get("name"), str):
                raise CalphadToolError("invalid_artifact_evidence")
            phase_fraction_values.append(
                _finite_evidence_number(
                    phase.get("NP_phase_fraction"), field="stable phase fraction"
                )
            )
        for vertex in raw_vertices:
            if not isinstance(vertex, dict) or not isinstance(vertex.get("phase"), str):
                raise CalphadToolError("invalid_artifact_evidence")
            _finite_evidence_number(
                vertex.get("NP_phase_fraction"), field="stable vertex phase fraction"
            )
            composition = vertex.get("composition_mole_fraction")
            if not isinstance(composition, dict) or not composition:
                raise CalphadToolError("invalid_artifact_evidence")
            for value in composition.values():
                _finite_evidence_number(value, field="stable vertex composition")
            composition_sum = _finite_evidence_number(
                vertex.get("composition_sum"), field="stable vertex composition sum"
            )
            if not math.isclose(composition_sum, 1.0, rel_tol=0.0, abs_tol=1e-8):
                raise CalphadToolError("invalid_artifact_evidence")
        if (
            not math.isclose(phase_fraction_sum, 1.0, rel_tol=0.0, abs_tol=1e-6)
            or not math.isclose(
                math.fsum(phase_fraction_values),
                phase_fraction_sum,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            or maximum_residual < 0
            or maximum_residual > 1e-8
            or any(value < 0 or value > 1e-8 for value in normalized_residuals.values())
            or gibbs_euler_residual < 0
            or gibbs_euler_residual > 1e-3
        ):
            raise CalphadToolError("invalid_artifact_evidence")
        stable_names.update(phase["name"] for phase in raw_phases)
        if point_index >= _MAX_SUMMARY_POINTS:
            continue
        phase_summaries = [
            _stable_phase_summary(phase) for phase in raw_phases[:_MAX_SUMMARY_PHASES_PER_POINT]
        ]
        point_summaries.append(
            {
                "conditions": point.get("conditions"),
                "stable_phases": phase_summaries,
                "stable_phases_truncated": len(raw_phases) > len(phase_summaries),
                "stable_phase_vertices": [
                    {
                        "vertex_index": vertex.get("vertex_index"),
                        "phase": vertex.get("phase"),
                        "NP_phase_fraction": vertex.get("NP_phase_fraction"),
                        "composition_mole_fraction": vertex.get("composition_mole_fraction"),
                        "composition_sum": vertex.get("composition_sum"),
                    }
                    for vertex in raw_vertices[:_MAX_SUMMARY_PHASES_PER_POINT]
                ],
                "stable_phase_vertices_truncated": (
                    len(raw_vertices) > _MAX_SUMMARY_PHASES_PER_POINT
                ),
                "phase_fraction_sum": phase_fraction_sum,
                "bulk_composition_residual_by_component": normalized_residuals,
                "maximum_bulk_composition_residual": maximum_residual,
                "GM_J_per_mol": gm_value,
                "chemical_potentials_J_per_mol": normalized_chemical_potentials,
                "gibbs_from_chemical_potentials_J_per_mol": gibbs_from_mu,
                "gibbs_euler_residual_J_per_mol": gibbs_euler_residual,
            }
        )
    database_record = result.get("database")
    summary = {
        "ok": True,
        "operation": "equilibrium",
        "database_sha256": (result.get("database") or {}).get("sha256")
        if isinstance(result.get("database"), dict)
        else None,
        "database_format": (result.get("database") or {}).get("format")
        if isinstance(result.get("database"), dict)
        else None,
        "inspection_artifact_sha256": evidence.get("request", {}).get("inspection_artifact_sha256")
        if isinstance(evidence.get("request"), dict)
        else None,
        "point_count": result_record.get("point_count", len(points)),
        "points": point_summaries,
        "points_truncated": len(points) > len(point_summaries),
        "stable_phase_names": sorted(stable_names),
        "phase_selection": request_record.get("phase_selection"),
        "dependent_component": dependent_component,
        "independent_composition_axes": dict(sorted(independent_composition_axes.items())),
        "pycalphad_version": database_record.get("pycalphad_version")
        if isinstance(database_record, dict)
        else None,
        "assessment_temperature_limits_K": database_record.get("assessment_temperature_limits_K")
        if isinstance(database_record, dict)
        else None,
        "assessment_pressure_limits_Pa": database_record.get("assessment_pressure_limits_Pa")
        if isinstance(database_record, dict)
        else None,
        "units": result_record.get("units"),
        "equilibrium_artifact": _artifact_summary(artifact),
        "catalog_validation_status": "pending",
        "scientific_status": "unverified",
        "evidence_note": (
            "Full chemical potentials, phase compositions, all grid points, warnings, and "
            "provenance remain in the content-addressed artifact."
        ),
    }
    return _bounded_summary(summary)


def _scheil_summary(artifact: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    runtime_result = evidence.get("result")
    if not isinstance(runtime_result, dict):
        raise CalphadToolError("invalid_artifact_evidence")
    result = runtime_result.get("result")
    request_record = runtime_result.get("request")
    database = runtime_result.get("database")
    if (
        not isinstance(result, dict)
        or not isinstance(request_record, dict)
        or not isinstance(database, dict)
    ):
        raise CalphadToolError("invalid_artifact_evidence")
    count = int(result["point_count"])
    temperatures = result["temperatures_K"]
    fraction_solid = result["fraction_solid"]
    fraction_liquid = result["fraction_liquid"]
    cumulative = result["solid_phase_cumulative_fraction"]
    compositions = result["phase_composition_mole_fraction"]
    if count <= _MAX_SUMMARY_POINTS:
        sampled_indices = list(range(count))
    else:
        sampled_indices = sorted(
            {
                round(index * (count - 1) / (_MAX_SUMMARY_POINTS - 1))
                for index in range(_MAX_SUMMARY_POINTS)
            }
        )
    ranked_phases = sorted(
        cumulative,
        key=lambda phase: (-float(cumulative[phase][-1]), phase),
    )
    summary_phases = ranked_phases[:_MAX_SUMMARY_PHASES_PER_POINT]
    trace = [
        {
            "point_index": index,
            "temperature_K": temperatures[index],
            "fraction_solid": fraction_solid[index],
            "fraction_liquid": fraction_liquid[index],
            "solid_phase_cumulative_fraction": {
                phase: cumulative[phase][index] for phase in summary_phases
            },
        }
        for index in sampled_indices
    ]
    last_available_compositions: dict[str, Any] = {}
    for phase in [FIXED_SCHEIL_LIQUID_PHASE, *summary_phases]:
        if phase not in compositions:
            continue
        component_series = compositions[phase]
        for index in range(count - 1, -1, -1):
            values = {
                component: component_values[index]
                for component, component_values in component_series.items()
            }
            if all(value is not None for value in values.values()):
                last_available_compositions[phase] = {
                    "point_index": index,
                    "temperature_K": temperatures[index],
                    "composition_mole_fraction": values,
                }
                break
    request = evidence.get("request")
    warnings = runtime_result.get("warnings")
    assert isinstance(warnings, list)
    mass_balance = result["elemental_mass_balance"]
    summary = {
        "ok": True,
        "operation": "scheil",
        "method": "Scheil-Gulliver",
        "model_scope": {
            "qualified": "classic_scheil_gulliver",
            "not_claimed": [
                "back_diffusion",
                "finite_rate_solid_diffusion",
                "precipitation",
                "phase_field",
            ],
        },
        "database_sha256": database.get("sha256"),
        "database_format": database.get("format"),
        "inspection_artifact_sha256": request.get("inspection_artifact_sha256")
        if isinstance(request, dict)
        else None,
        "components": request_record.get("components"),
        "phases": request_record.get("phases"),
        "bulk_composition_mole_fraction": request_record.get("bulk_composition_mole_fraction"),
        "dependent_component": request_record.get("dependent_component"),
        "conditions": {
            key: request_record.get(key)
            for key in (
                "start_temperature_K",
                "step_temperature_K",
                "pressure_Pa",
                "stop_liquid_fraction",
            )
        },
        "point_count": count,
        "temperature_range_K": [temperatures[-1], temperatures[0]],
        "final_fraction_solid": fraction_solid[-1],
        "final_fraction_liquid": fraction_liquid[-1],
        "solid_phase_names": ranked_phases,
        "final_solid_phase_cumulative_fraction": {
            phase: cumulative[phase][-1] for phase in summary_phases
        },
        "solid_phases_truncated": len(ranked_phases) > len(summary_phases),
        "sampled_path": trace,
        "path_truncated": len(trace) < count,
        "last_available_phase_composition_mole_fraction": last_available_compositions,
        "elemental_mass_balance": {
            "all_retained_points_closed": mass_balance.get("all_retained_points_closed"),
            "absolute_tolerance": mass_balance.get("absolute_tolerance"),
            "maximum_absolute_component_error": mass_balance.get(
                "maximum_absolute_component_error"
            ),
            "maximum_absolute_error_by_component": mass_balance.get(
                "maximum_absolute_error_by_component"
            ),
        },
        "assumptions": list(SCHEIL_ASSUMPTIONS),
        "warnings": warnings[:_MAX_SUMMARY_POINTS],
        "warnings_truncated": len(warnings) > _MAX_SUMMARY_POINTS,
        "solver": runtime_result.get("solver"),
        "limits": runtime_result.get("limits"),
        "scheil_artifact": _artifact_summary(artifact),
        "catalog_validation_status": "pending",
        "scientific_status": "unverified",
        "evidence_note": (
            "The complete temperature path, phase increments/compositions, pointwise elemental "
            "closure, warnings, and provenance remain in the content-addressed artifact. "
            "Numerical convergence does not validate the thermodynamic assessment."
        ),
    }
    return _bounded_summary(summary)


def _persist_calphad_catalog_validation(
    settings: Any,
    context: AgentRunContext,
    *,
    resource_id: str,
    operation: str,
    artifact: Mapping[str, Any],
    evidence_bytes: bytes,
    runtime_image_id: str,
    pycalphad_version: str,
) -> dict[str, Any]:
    """Append trusted runtime evidence to the server-managed CALPHAD ledger."""

    import httpx

    run_id = str(context.run_id or "").strip()
    worker_token = str(getattr(settings, "control_worker_token", "") or "").strip()
    base_url = str(getattr(settings, "control_base_url", "") or "").rstrip("/")
    lease_worker_id = str(context.run_lease_worker_id or "").strip()
    lease_token = str(context.run_lease_token or "").strip()
    if not run_id or not worker_token or not base_url or not lease_worker_id or not lease_token:
        raise CalphadToolError("calphad_governance_persistence_failed")
    if operation not in {"inspect", "equilibrium", "scheil"}:
        raise CalphadToolError("calphad_governance_persistence_failed")
    if not evidence_bytes or len(evidence_bytes) > MAX_ARTIFACT_BYTES:
        raise CalphadToolError("calphad_governance_persistence_failed")
    artifact_sha256 = str(artifact.get("sha256") or "").strip().lower()
    artifact_size_bytes = artifact.get("size_bytes")
    artifact_directory = _operation_directory_name(operation)
    expected_artifact_path = f"/outputs/calphad/{artifact_directory}/{artifact_sha256}.json"
    version_identity = str(pycalphad_version or "")
    if (
        not _SHA256_RE.fullmatch(artifact_sha256)
        or isinstance(artifact_size_bytes, bool)
        or not isinstance(artifact_size_bytes, int)
        or artifact_size_bytes != len(evidence_bytes)
        or _sha256(evidence_bytes) != artifact_sha256
        or artifact.get("path") != expected_artifact_path
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(runtime_image_id or "")) is None
        or not version_identity
        or version_identity != version_identity.strip()
        or len(version_identity) > 128
    ):
        raise CalphadToolError("calphad_governance_persistence_failed")
    evidence_record = _safe_json_loads(evidence_bytes, label="CALPHAD ledger evidence")
    evidence_binding = (
        evidence_record.get("database_binding") if isinstance(evidence_record, dict) else None
    )
    if not isinstance(evidence_binding, dict):
        raise CalphadToolError("calphad_governance_persistence_failed")
    evidence_schema = evidence_record.get("schema_version")
    failure_domain = ""
    failure_stage = ""
    failure_code = ""
    if evidence_schema == FAILURE_EVIDENCE_SCHEMA_VERSION:
        outcome = evidence_record.get("outcome")
        if not isinstance(outcome, dict):
            raise CalphadToolError("calphad_governance_persistence_failed")
        status = str(outcome.get("status") or "")
        failure_domain = str(outcome.get("failure_domain") or "")
        failure_stage = str(outcome.get("failure_stage") or "")
        failure_code = str(outcome.get("failure_code") or "")
        if (
            set(outcome)
            != {
                "status",
                "failure_domain",
                "failure_stage",
                "failure_code",
                "exit_code",
                "solver_started",
            }
            or status not in FAILURE_STATUSES
            or failure_domain not in FAILURE_DOMAINS
            or failure_stage not in FAILURE_STAGES
            or failure_code not in FAILURE_CODES
            or not valid_failure_tuple(
                status=status,
                failure_domain=failure_domain,
                failure_stage=failure_stage,
                failure_code=failure_code,
                solver_started=outcome.get("solver_started"),
                operation=operation,
            )
        ):
            raise CalphadToolError("calphad_governance_persistence_failed")
    else:
        status = {
            "inspect": "input_validated",
            "equilibrium": "equilibrium_completed",
            "scheil": "scheil_completed",
        }[operation]
    database_sha256 = str(evidence_binding.get("sha256") or "").strip().lower()
    database_size_bytes = evidence_binding.get("size_bytes")
    database_format = evidence_binding.get("database_format")
    request_record = evidence_record.get("request") if isinstance(evidence_record, dict) else None
    expected_inspection_sha256 = ""
    if operation in {"equilibrium", "scheil"}:
        if not isinstance(request_record, dict):
            raise CalphadToolError("calphad_governance_persistence_failed")
        expected_inspection_sha256 = (
            str(request_record.get("inspection_artifact_sha256") or "").strip().lower()
        )
    if (
        evidence_binding.get("resource_id") != resource_id
        or not _SHA256_RE.fullmatch(database_sha256)
        or database_format not in {"tdb", "dat"}
        or isinstance(database_size_bytes, bool)
        or not isinstance(database_size_bytes, int)
        or database_size_bytes <= 0
        or (
            operation in {"equilibrium", "scheil"}
            and not _SHA256_RE.fullmatch(expected_inspection_sha256)
        )
    ):
        raise CalphadToolError("calphad_governance_persistence_failed")
    compressed_evidence = gzip.compress(evidence_bytes, compresslevel=9, mtime=0)
    if not compressed_evidence or len(compressed_evidence) > _MAX_CALPHAD_EVIDENCE_GZIP_BYTES:
        raise CalphadToolError("calphad_governance_persistence_failed")
    payload = {
        "status": status,
        "operation": operation,
        "evidence_path": str(artifact.get("path") or ""),
        "evidence_sha256": str(artifact.get("sha256") or ""),
        "evidence_size_bytes": artifact.get("size_bytes"),
        "runtime_image_id": str(runtime_image_id or ""),
        "pycalphad_version": str(pycalphad_version or ""),
        "evidence_gzip_base64": base64.b64encode(compressed_evidence).decode("ascii"),
    }
    if evidence_schema == FAILURE_EVIDENCE_SCHEMA_VERSION:
        payload.update(
            {
                "failure_domain": failure_domain,
                "failure_stage": failure_stage,
                "failure_code": failure_code,
            }
        )
    url = (
        f"{base_url}/v2/runs/{quote(run_id, safe='')}/resources/"
        f"{quote(resource_id, safe='')}/calphad/validations"
    )
    try:
        with httpx.Client(timeout=_CALPHAD_LEDGER_TIMEOUT_SECONDS) as client:
            response = client.post(
                url,
                json=payload,
                headers={
                    "X-Ultra-Worker-Token": worker_token,
                    "X-Ultra-Run-Id": run_id,
                    "X-Ultra-Worker-Id": lease_worker_id,
                    "X-Ultra-Run-Lease-Token": lease_token,
                },
            )
            response.raise_for_status()
            response_body = response.json()
    except Exception as exc:  # noqa: BLE001 - ledger persistence is deliberately fail closed
        raise CalphadToolError("calphad_governance_persistence_failed") from exc
    if not isinstance(response_body, dict):
        raise CalphadToolError("calphad_governance_persistence_failed")
    revision = response_body.get("revision")
    validation = response_body.get("validation")
    if (
        not isinstance(revision, dict)
        or not isinstance(validation, dict)
        or not str(revision.get("revision_id") or "").strip()
        or not str(validation.get("validation_id") or "").strip()
        or revision.get("resource_id") != resource_id
        or revision.get("sha256") != database_sha256
        or revision.get("size_bytes") != database_size_bytes
        or revision.get("database_format") != database_format
        or validation.get("resource_id") != resource_id
        or validation.get("database_sha256") != revision.get("sha256")
        or validation.get("database_size_bytes") != revision.get("size_bytes")
        or validation.get("database_format") != revision.get("database_format")
        or validation.get("status") != status
        or validation.get("operation") != operation
        or validation.get("evidence_path") != payload["evidence_path"]
        or validation.get("evidence_sha256") != payload["evidence_sha256"]
        or validation.get("evidence_size_bytes") != payload["evidence_size_bytes"]
        or validation.get("runtime_image_id") != payload["runtime_image_id"]
        or validation.get("pycalphad_version") != payload["pycalphad_version"]
        or validation.get("run_id") != run_id
        or str(validation.get("inspection_evidence_sha256") or "").strip().lower()
        != expected_inspection_sha256
        or validation.get("evidence_retention") != "retained"
        or validation.get("promotable") is not (evidence_schema != FAILURE_EVIDENCE_SCHEMA_VERSION)
        or str(validation.get("failure_domain") or "") != failure_domain
        or str(validation.get("failure_stage") or "") != failure_stage
        or str(validation.get("failure_code") or "") != failure_code
        or validation.get("created_by_authority") != "trusted_worker"
    ):
        raise CalphadToolError("calphad_governance_persistence_failed")
    return {
        "mode": "server_managed_append_only_ledger",
        "persisted": True,
        "revision_id": revision.get("revision_id"),
        "validation_id": validation.get("validation_id"),
        "validation_status": status,
        "created_by_authority": "trusted_worker",
    }


def _governed_calphad_result(
    settings: Any,
    context: AgentRunContext,
    backend: DockerSandboxBackend,
    result: dict[str, Any],
    *,
    operation: str,
    resource_id: str,
    embedded_database_id: str,
) -> dict[str, Any]:
    """Attach catalog governance without letting the model author ledger claims."""

    artifact_created = result.get("artifact_created") is True
    if result.get("ok") is not True and not artifact_created:
        return result
    if embedded_database_id:
        governed = dict(result)
        governed["catalog_ledger"] = {
            "mode": "embedded_release_registry",
            "persisted": False,
        }
        return governed
    if not resource_id:
        return result
    try:
        governance_scope = _selected_resource_governance_scope(
            context, resource_id=str(resource_id).strip()
        )
    except CalphadToolError as exc:
        failed = dict(result)
        failed.update(
            {
                "ok": False,
                "error": exc.code,
                "message": exc.public_message,
                "artifact_created": True,
                "catalog_validation_status": "persistence_failed",
            }
        )
        return failed
    if governance_scope == "read_only_usage":
        governed = dict(result)
        governed["catalog_validation_status"] = "read_only_unpromoted"
        governed["catalog_ledger"] = {
            "mode": "shared_read_only_artifact",
            "persisted": False,
            "promotable": False,
        }
        return governed
    artifact_key = {
        "inspect": "inspection_artifact",
        "equilibrium": "equilibrium_artifact",
        "scheil": "scheil_artifact",
    }[operation]
    artifact = result.get(artifact_key)
    if operation == "inspect" and isinstance(result.get("database"), dict):
        pycalphad_version = (result.get("database") or {}).get("pycalphad_version")
    elif operation == "scheil" and isinstance(result.get("solver"), dict):
        pycalphad_version = (result.get("solver") or {}).get("pycalphad_version")
    else:
        pycalphad_version = result.get("pycalphad_version")
    if artifact_created:
        pycalphad_version = _PINNED_PYCALPHAD_VERSION
    try:
        if not isinstance(artifact, dict):
            raise CalphadToolError("calphad_governance_persistence_failed")
        evidence_path = _artifact_host_path(backend, artifact)
        evidence_bytes = _read_source(evidence_path)
        if (
            len(evidence_bytes) != artifact.get("size_bytes")
            or len(evidence_bytes) > MAX_ARTIFACT_BYTES
            or _sha256(evidence_bytes) != artifact.get("sha256")
        ):
            raise CalphadToolError("calphad_governance_persistence_failed")
        ledger = _persist_calphad_catalog_validation(
            settings,
            context,
            resource_id=str(resource_id).strip(),
            operation=operation,
            artifact=artifact,
            evidence_bytes=evidence_bytes,
            runtime_image_id=str(backend.config.image or ""),
            pycalphad_version=str(pycalphad_version or ""),
        )
    except CalphadToolError as exc:
        failed = dict(result)
        failed.update(
            {
                "ok": False,
                "error": exc.code,
                "message": (
                    "CALPHAD evidence was created but its server-managed validation record "
                    "could not be persisted"
                ),
                "artifact_created": True,
                "catalog_validation_status": "persistence_failed",
                "catalog_ledger": {
                    "mode": "server_managed_append_only_ledger",
                    "persisted": False,
                },
            }
        )
        return failed
    governed = dict(result)
    governed["catalog_validation_status"] = ledger["validation_status"]
    governed["catalog_ledger"] = ledger
    return governed


def inspect_calphad_database_typed(
    settings: Any,
    context: AgentRunContext,
    backend: DockerSandboxBackend,
    *,
    upload_roots: Iterable[str | Path] = (),
    resource_id: str = "",
    embedded_database_id: str = "",
    components: list[str] | None = None,
    phases: list[str] | None = None,
) -> dict[str, Any]:
    """Run the fixed typed inspection primitive and return a bounded summary."""

    try:
        if resource_id:
            _selected_resource_governance_scope(context, resource_id=resource_id)
        database = _database_request(
            settings,
            context,
            backend,
            upload_roots=_resolved_upload_roots(upload_roots),
            resource_id=resource_id,
            embedded_database_id=embedded_database_id,
        )
        request = {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "operation": "inspect",
            "runtime_image_id": backend.config.image,
            "database": database,
            "selection": {
                "components": _bounded_name_list(
                    components,
                    label="components",
                    maximum=MAX_TYPED_COMPONENTS,
                    allow_none=True,
                ),
                "phases": _bounded_name_list(
                    phases,
                    label="phases",
                    maximum=MAX_TYPED_PHASES,
                    allow_none=True,
                ),
            },
        }
        artifact, evidence = _execute_typed_request(backend, request)
        if evidence.get("schema_version") == FAILURE_EVIDENCE_SCHEMA_VERSION:
            return _failure_summary(artifact, evidence)
        return _inspect_summary(artifact, evidence)
    except CalphadToolError as exc:
        return {
            "ok": False,
            "error": exc.code,
            "message": exc.public_message,
            "catalog_validation_status": "pending",
            "artifact_created": False,
        }


def run_calphad_equilibrium_typed(  # noqa: N803 - scientific unit suffix is public API
    settings: Any,
    context: AgentRunContext,
    backend: DockerSandboxBackend,
    *,
    inspection_artifact_sha256: str,
    components: list[str],
    phases: list[str],
    temperatures_K: list[float],  # noqa: N803 - unit suffix is the public schema
    pressures_Pa: list[float],  # noqa: N803 - unit suffix is the public schema
    independent_compositions: dict[str, list[float]],
    upload_roots: Iterable[str | Path] = (),
    resource_id: str = "",
    embedded_database_id: str = "",
) -> dict[str, Any]:
    """Run the fixed typed equilibrium primitive and return a bounded summary."""

    try:
        inspection_sha = str(inspection_artifact_sha256 or "").strip().lower()
        if not _SHA256_RE.fullmatch(inspection_sha):
            raise CalphadToolError("valid_inspection_artifact_sha256_required")
        normalized_components = _bounded_name_list(
            components,
            label="components",
            maximum=MAX_TYPED_COMPONENTS,
            allow_none=False,
        )
        normalized_phases = _bounded_name_list(
            phases,
            label="phases",
            maximum=MAX_TYPED_PHASES,
            allow_none=False,
        )
        assert normalized_components is not None and normalized_phases is not None
        normalized_temperatures = _bounded_axis(
            temperatures_K, label="temperatures_K", minimum=1.0, maximum=10_000.0
        )
        normalized_pressures = _bounded_axis(
            pressures_Pa, label="pressures_Pa", minimum=1e-9, maximum=1e12
        )
        normalized_compositions = _bounded_compositions(independent_compositions)
        if resource_id:
            _selected_resource_governance_scope(context, resource_id=resource_id)
        database = _database_request(
            settings,
            context,
            backend,
            upload_roots=_resolved_upload_roots(upload_roots),
            resource_id=resource_id,
            embedded_database_id=embedded_database_id,
        )
        if database["kind"] == "resource":
            pressure_limits = database["assessment_pressure_limits_Pa"]
            if (
                normalized_pressures[0] < pressure_limits[0]
                or normalized_pressures[-1] > pressure_limits[1]
            ):
                raise CalphadToolError(
                    "invalid_typed_input",
                    "pressures_Pa is outside the owner-declared assessment pressure limits",
                )
        available_components, available_phases = _retained_inspection_inventory(
            backend,
            inspection_sha256=inspection_sha,
            database=database,
        )
        if not set(normalized_components).issubset(available_components) or not set(
            normalized_phases
        ).issubset(available_phases):
            raise CalphadToolError(
                "inspection_inventory_mismatch",
                "the requested selection is absent from the retained inspection inventory",
            )
        if "VA" in available_components and "VA" not in normalized_components:
            if len(normalized_components) >= MAX_TYPED_COMPONENTS:
                raise CalphadToolError(
                    "invalid_typed_input",
                    "components cannot include the inspected VA within the typed component limit",
                )
            normalized_components = sorted([*normalized_components, "VA"])
        try:
            canonical_compositions, _, _ = canonicalize_equilibrium_compositions(
                normalized_compositions,
                components=normalized_components,
            )
        except CalphadInputError as exc:
            raise CalphadToolError("invalid_typed_input", str(exc)) from exc
        normalized_compositions = {
            component: list(axis) for component, axis in canonical_compositions.items()
        }
        grid_points = len(normalized_temperatures) * len(normalized_pressures)
        for axis in normalized_compositions.values():
            grid_points *= len(axis)
        if grid_points > MAX_TYPED_GRID_POINTS:
            raise CalphadToolError(
                "typed_grid_limit_exceeded",
                f"grid exceeds {MAX_TYPED_GRID_POINTS} points",
            )
        request = {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "operation": "equilibrium",
            "runtime_image_id": backend.config.image,
            "database": database,
            "selection": {
                "components": normalized_components,
                "phases": normalized_phases,
            },
            "inspection_artifact_sha256": inspection_sha,
            "conditions": {
                "temperatures_K": normalized_temperatures,
                "pressures_Pa": normalized_pressures,
                "independent_compositions": normalized_compositions,
            },
        }
        artifact, evidence = _execute_typed_request(backend, request)
        if evidence.get("schema_version") == FAILURE_EVIDENCE_SCHEMA_VERSION:
            return _failure_summary(artifact, evidence)
        return _equilibrium_summary(artifact, evidence)
    except CalphadToolError as exc:
        return {
            "ok": False,
            "error": exc.code,
            "message": exc.public_message,
            "catalog_validation_status": "pending",
            "artifact_created": False,
        }


def run_calphad_scheil_typed(  # noqa: N803 - scientific unit suffix is public API
    settings: Any,
    context: AgentRunContext,
    backend: DockerSandboxBackend,
    *,
    inspection_artifact_sha256: str,
    components: list[str],
    phases: list[str],
    independent_composition_mole_fraction: dict[str, float],
    start_temperature_K: float,  # noqa: N803 - unit suffix is the public schema
    step_temperature_K: float,  # noqa: N803 - unit suffix is the public schema
    pressure_Pa: float = FIXED_SCHEIL_PRESSURE_PA,  # noqa: N803
    stop_liquid_fraction: float = 1e-4,
    upload_roots: Iterable[str | Path] = (),
    resource_id: str = "",
    embedded_database_id: str = "",
) -> dict[str, Any]:
    """Run one governed, bounded classic Scheil--Gulliver solidification path."""

    try:
        inspection_sha = str(inspection_artifact_sha256 or "").strip().lower()
        if not _SHA256_RE.fullmatch(inspection_sha):
            raise CalphadToolError("valid_inspection_artifact_sha256_required")
        normalized_components = _bounded_name_list(
            components,
            label="components",
            maximum=MAX_TYPED_COMPONENTS,
            allow_none=False,
        )
        normalized_phases = _bounded_name_list(
            phases,
            label="phases",
            maximum=MAX_TYPED_PHASES,
            allow_none=False,
        )
        assert normalized_components is not None and normalized_phases is not None
        if FIXED_SCHEIL_LIQUID_PHASE not in normalized_phases:
            raise CalphadToolError(
                "invalid_typed_input",
                f"phases must include {FIXED_SCHEIL_LIQUID_PHASE}",
            )
        normalized_composition = _bounded_scalar_composition(independent_composition_mole_fraction)
        normalized_start_temperature = _bounded_scalar(
            start_temperature_K,
            label="start_temperature_K",
            minimum=1.0,
            maximum=10_000.0,
        )
        normalized_step_temperature = _bounded_scalar(
            step_temperature_K,
            label="step_temperature_K",
            minimum=MIN_STEP_TEMPERATURE_K,
            maximum=MAX_STEP_TEMPERATURE_K,
        )
        normalized_pressure = _bounded_scalar(
            pressure_Pa,
            label="pressure_Pa",
            minimum=1e-9,
            maximum=1e12,
        )
        if normalized_pressure != FIXED_SCHEIL_PRESSURE_PA:
            raise CalphadToolError(
                "invalid_typed_input",
                f"qualified Scheil execution requires pressure_Pa={FIXED_SCHEIL_PRESSURE_PA:g}",
            )
        normalized_stop_fraction = _bounded_scalar(
            stop_liquid_fraction,
            label="stop_liquid_fraction",
            minimum=MIN_STOP_LIQUID_FRACTION,
            maximum=MAX_STOP_LIQUID_FRACTION,
        )
        if resource_id:
            _selected_resource_governance_scope(context, resource_id=resource_id)
        database = _database_request(
            settings,
            context,
            backend,
            upload_roots=_resolved_upload_roots(upload_roots),
            resource_id=resource_id,
            embedded_database_id=embedded_database_id,
        )
        if database["kind"] == "resource":
            temperature_limits = database["temperature_limits_K"]
            pressure_limits = database["assessment_pressure_limits_Pa"]
            if not temperature_limits[0] <= normalized_start_temperature <= temperature_limits[1]:
                raise CalphadToolError(
                    "invalid_typed_input",
                    "start_temperature_K is outside the owner-declared assessment limits",
                )
            if not pressure_limits[0] <= normalized_pressure <= pressure_limits[1]:
                raise CalphadToolError(
                    "invalid_typed_input",
                    "pressure_Pa is outside the owner-declared assessment limits",
                )
            minimum_effective_step = normalized_step_temperature / (1.2**6)
            worst_case_points = 2 + math.ceil(
                max(0.0, normalized_start_temperature - temperature_limits[0])
                / minimum_effective_step
            )
            if worst_case_points > FIXED_SCHEIL_MAX_STEPS:
                raise CalphadToolError(
                    "scheil_step_limit_exceeded",
                    "temperature range/step can exceed the fixed Scheil step limit",
                )
        available_components, available_phases = _retained_inspection_inventory(
            backend,
            inspection_sha256=inspection_sha,
            database=database,
        )
        if not set(normalized_components).issubset(available_components) or not set(
            normalized_phases
        ).issubset(available_phases):
            raise CalphadToolError(
                "inspection_inventory_mismatch",
                "the requested selection is absent from the retained inspection inventory",
            )
        if "VA" in available_components and "VA" not in normalized_components:
            if len(normalized_components) >= MAX_TYPED_COMPONENTS:
                raise CalphadToolError(
                    "invalid_typed_input",
                    "components cannot include the inspected VA within the typed component limit",
                )
            normalized_components = sorted([*normalized_components, "VA"])
        try:
            canonical_compositions, _, closure_records = canonicalize_equilibrium_compositions(
                normalized_composition,
                components=normalized_components,
            )
        except CalphadInputError as exc:
            raise CalphadToolError("invalid_typed_input", str(exc)) from exc
        if len(closure_records) != 1 or any(
            len(values) != 1 for values in canonical_compositions.values()
        ):
            raise CalphadToolError(
                "invalid_typed_input", "Scheil requires exactly one bulk composition"
            )
        normalized_composition = {
            component: values[0] for component, values in sorted(canonical_compositions.items())
        }
        request = {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "operation": "scheil",
            "runtime_image_id": backend.config.image,
            "database": database,
            "selection": {
                "components": normalized_components,
                "phases": normalized_phases,
            },
            "inspection_artifact_sha256": inspection_sha,
            "conditions": {
                "independent_composition_mole_fraction": normalized_composition,
                "start_temperature_K": normalized_start_temperature,
                "step_temperature_K": normalized_step_temperature,
                "pressure_Pa": normalized_pressure,
                "stop_liquid_fraction": normalized_stop_fraction,
            },
        }
        artifact, evidence = _execute_typed_request(backend, request)
        if evidence.get("schema_version") == FAILURE_EVIDENCE_SCHEMA_VERSION:
            return _failure_summary(artifact, evidence)
        return _scheil_summary(artifact, evidence)
    except CalphadToolError as exc:
        return {
            "ok": False,
            "error": exc.code,
            "message": exc.public_message,
            "catalog_validation_status": "pending",
            "artifact_created": False,
        }


def build_calphad_tools(
    settings: Any,
    *,
    backend: DockerSandboxBackend,
    upload_roots: Iterable[str | Path] = (),
) -> list[Any]:
    """Build the three typed CALPHAD tools for one run's immutable sandbox."""

    resolved_upload_roots = tuple(upload_roots)

    @tool
    def calphad_inspect_database(
        runtime: ToolRuntime[AgentRunContext],
        resource_id: str = "",
        embedded_database_id: str = "",
        components: list[str] | None = None,
        phases: list[str] | None = None,
    ) -> str:
        """Inspect one authorized CALPHAD database with the fixed bounded pycalphad primitive. Set exactly one source: resource_id for a resource explicitly selected when this run was created, or embedded_database_id (currently nist-al-co-w-wang-2017) for Ultra's reviewed registry. No paths or code are accepted. User resources require server SHA/size plus owner-declared source, license, assessment scope, reference state, and temperature/pressure bounds. Returns bounded inventory plus a content-addressed inspection artifact SHA required by calphad_run_equilibrium or calphad_run_scheil."""

        result = inspect_calphad_database_typed(
            settings,
            runtime.context,
            backend,
            upload_roots=resolved_upload_roots,
            resource_id=resource_id,
            embedded_database_id=embedded_database_id,
            components=components,
            phases=phases,
        )
        result = _governed_calphad_result(
            settings,
            runtime.context,
            backend,
            result,
            operation="inspect",
            resource_id=resource_id,
            embedded_database_id=embedded_database_id,
        )
        return json.dumps(result, allow_nan=False, indent=2, sort_keys=True)

    @tool
    def calphad_run_equilibrium(  # noqa: N803 - scientific unit suffix is tool schema
        runtime: ToolRuntime[AgentRunContext],
        inspection_artifact_sha256: str,
        components: list[str],
        phases: list[str],
        temperatures_K: list[float],  # noqa: N803 - unit suffix is the tool schema
        pressures_Pa: list[float],  # noqa: N803 - unit suffix is the tool schema
        independent_compositions: dict[str, list[float]],
        resource_id: str = "",
        embedded_database_id: str = "",
    ) -> str:
        """Run a finite bounded CALPHAD equilibrium grid through the fixed pycalphad primitive. First call calphad_inspect_database, then pass its artifact SHA and choose physical components/phases from that inventory; Ultra authenticates that artifact and automatically retains VA when its exact inventory declares VA. Set the same resource_id or embedded_database_id. T is K, P is Pa, compositions are independent mole-fraction axes; total amount is fixed at 1 mol. No code, paths, solver models, parameters, or options are accepted. Full evidence is content-addressed under /outputs."""

        result = run_calphad_equilibrium_typed(
            settings,
            runtime.context,
            backend,
            inspection_artifact_sha256=inspection_artifact_sha256,
            components=components,
            phases=phases,
            temperatures_K=temperatures_K,
            pressures_Pa=pressures_Pa,
            independent_compositions=independent_compositions,
            upload_roots=resolved_upload_roots,
            resource_id=resource_id,
            embedded_database_id=embedded_database_id,
        )
        result = _governed_calphad_result(
            settings,
            runtime.context,
            backend,
            result,
            operation="equilibrium",
            resource_id=resource_id,
            embedded_database_id=embedded_database_id,
        )
        return json.dumps(result, allow_nan=False, indent=2, sort_keys=True)

    @tool
    def calphad_run_scheil(  # noqa: N803 - scientific unit suffix is tool schema
        runtime: ToolRuntime[AgentRunContext],
        inspection_artifact_sha256: str,
        components: list[str],
        phases: list[str],
        independent_composition_mole_fraction: dict[str, float],
        start_temperature_K: float,  # noqa: N803 - unit suffix is the tool schema
        step_temperature_K: float,  # noqa: N803 - unit suffix is the tool schema
        pressure_Pa: float = FIXED_SCHEIL_PRESSURE_PA,  # noqa: N803
        stop_liquid_fraction: float = 1e-4,
        resource_id: str = "",
        embedded_database_id: str = "",
    ) -> str:
        """Run one governed classic Scheil--Gulliver solidification path. First call calphad_inspect_database, pass its artifact SHA, select a defensible complete phase set including LIQUID, and set exactly one scalar bulk composition using one fewer physical component than the alloy system; Ultra canonicalizes the dependent component and retains VA automatically without X(VA). Set the same selected resource_id or reviewed embedded_database_id. The starting state must be single-phase liquid. Temperature is K, pressure is fixed to 101325 Pa, composition is mole fraction, and the residual-liquid criterion is bounded. No paths, code, solver models, arbitrary options, back-diffusion, or phase-field claims are accepted. Execution uses immutable pinned scheil/pycalphad with fixed time/result/step caps and independently rechecks every retained elemental inventory."""

        result = run_calphad_scheil_typed(
            settings,
            runtime.context,
            backend,
            inspection_artifact_sha256=inspection_artifact_sha256,
            components=components,
            phases=phases,
            independent_composition_mole_fraction=independent_composition_mole_fraction,
            start_temperature_K=start_temperature_K,
            step_temperature_K=step_temperature_K,
            pressure_Pa=pressure_Pa,
            stop_liquid_fraction=stop_liquid_fraction,
            upload_roots=resolved_upload_roots,
            resource_id=resource_id,
            embedded_database_id=embedded_database_id,
        )
        result = _governed_calphad_result(
            settings,
            runtime.context,
            backend,
            result,
            operation="scheil",
            resource_id=resource_id,
            embedded_database_id=embedded_database_id,
        )
        return json.dumps(result, allow_nan=False, indent=2, sort_keys=True)

    return [calphad_inspect_database, calphad_run_equilibrium, calphad_run_scheil]


__all__ = [
    "CalphadToolError",
    "build_calphad_tools",
    "inspect_calphad_database_typed",
    "run_calphad_equilibrium_typed",
    "run_calphad_scheil_typed",
]
