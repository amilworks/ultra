"""Run-scoped agent tools for validated materials sensor series.

The tool surface deliberately accepts a selected resource id, never a filesystem path.  The
control plane has already authorized ``AgentRunContext.selected_file_ids``; this module reuses
the common upload staging path and then opens only the staged Zarr root.  Results are bounded
metadata/envelope summaries, not raw signals and not claims of object-store or viewer support.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from langchain.tools import ToolRuntime, tool

from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.context_tools import (
    _find_uploaded_file,
    _resolved_upload_roots,
    stage_uploaded_files,
)
from ultra_deepagents.sensors.reader import (
    DEFAULT_VALIDATION_MAX_DECODED_BYTES,
    DEFAULT_VALIDATION_MAX_READS,
    DEFAULT_VALIDATION_MAX_VALUES,
    DEFAULT_VALIDATION_MAX_WALL_SECONDS,
    SENSOR_SCHEMA,
    SensorValidationBudget,
    SensorValidationError,
    build_min_max_envelope,
    open_sensor_series,
)

SENSOR_INSPECTION_SCHEMA = "ultra.sensor-inspection.v1"

# Host-side tool budgets are intentionally smaller than the format validator's per-read cap.
# Larger series remain available through explicit metadata preflight and stored multiscales;
# the coordinator must not launch an unbounded value scan from a natural-language request.
MAX_VALUE_VALIDATION_SAMPLES = 5_000_000
MAX_GENERATED_ENVELOPE_SAMPLES = 250_000
MAX_ENVELOPE_BUCKETS = 2_048
MAX_SOURCE_TREE_ENTRIES = 200_000
MAX_SOURCE_TREE_BYTES = 512 * 1024 * 1024
MAX_SOURCE_PREFLIGHT_SECONDS = 5.0

SENSOR_FORMAT_BINDING_SCHEMA = "ultra.sensor-format-binding.v1"

_FILE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LINK_DECLARED_UNVERIFIED = "declared_unverified"
_LINK_CATALOG_IDENTITY_VERIFIED = "catalog_identity_verified"
_LINK_WARNING_PREFIX = "linked_resource_declared_unverified:"
_SENSOR_FORMAT_FIELDS = frozenset(
    {
        "schema",
        "authority",
        "container",
        "sensor_schema",
        "resource_sha256",
        "detection",
    }
)
_SENSOR_GOAL_TOKENS = (
    "acoustic emission",
    "data acquisition",
    "daq waveform",
    "extensometer",
    "load cell",
    "sensor data",
    "sensor series",
    "sensor telemetry",
    "stress strain",
    "stress-strain",
    "synchronized waveform",
    "thermal telemetry",
    "thermocouple",
    "waveform",
)


def _trusted_sensor_format(descriptor: dict[str, Any], selected: set[str]) -> bool:
    """Accept only the closed, catalog-authored marker bound to this resource digest."""

    if (
        str(descriptor.get("type") or "").strip() != "selected_resource"
        or str(descriptor.get("binding_schema") or "").strip() != "ultra.selected_resource.v1"
        or str(descriptor.get("authority") or "").strip() != "control_resource_catalog"
    ):
        return False
    resource_id = str(descriptor.get("resource_id") or "").strip()
    file_id = str(descriptor.get("file_id") or "").strip()
    digest = str(descriptor.get("sha256") or "").strip().lower()
    marker = descriptor.get("sensor_format")
    if (
        resource_id not in selected
        or file_id != resource_id
        or not _SHA256_RE.fullmatch(digest)
        or not isinstance(marker, dict)
        or set(marker) != _SENSOR_FORMAT_FIELDS
    ):
        return False
    return marker == {
        "schema": SENSOR_FORMAT_BINDING_SCHEMA,
        "authority": "control_resource_catalog",
        "container": "zarr",
        "sensor_schema": SENSOR_SCHEMA,
        "resource_sha256": digest,
        "detection": "bounded_root_attributes",
    }


def should_register_sensor_tools(context: AgentRunContext | None) -> bool:
    """Keep the typed surface scoped to selected, sensor-shaped runs."""

    if context is None or not context.selected_file_ids:
        return False
    goal = " ".join(str(context.goal or "").lower().split())
    if any(token in goal for token in _SENSOR_GOAL_TOKENS):
        return True
    selected = set(context.selected_file_ids)
    return any(
        _trusted_sensor_format(descriptor, selected)
        for descriptor in context.resource_descriptors
        if isinstance(descriptor, dict)
    )


def _error(
    code: str,
    message: str,
    *,
    resource: dict[str, Any] | None = None,
    path: str | None = None,
    preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": False,
        "schema": SENSOR_INSPECTION_SCHEMA,
        "error": {"code": code, "message": message},
    }
    if path:
        result["error"]["path"] = path
    if resource:
        result["resource"] = resource
    if preflight:
        result["preflight"] = preflight
    return result


def _selected_file_id(context: AgentRunContext, requested: str) -> tuple[str | None, str | None]:
    selected = tuple(dict.fromkeys(str(value or "").strip() for value in context.selected_file_ids))
    selected = tuple(value for value in selected if value and _FILE_ID_RE.fullmatch(value))
    candidate = str(requested or "").strip()
    if not candidate:
        if len(selected) == 1:
            return selected[0], None
        if not selected:
            return None, "no selected uploaded resource is available to this run"
        return None, "file_id is required when more than one uploaded resource is selected"
    if not _FILE_ID_RE.fullmatch(candidate):
        return None, "file_id is not a valid selected-resource identifier"
    if candidate not in set(selected):
        return None, "file_id is not authorized by this run's selected resources"
    return candidate, None


def _safe_stage_token(file_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", file_id).strip("._")


def _check_existing_stage_path(context: AgentRunContext, file_id: str) -> str | None:
    """Reject model-created symlinks before common staging writes into the workspace."""

    workspace = Path(context.workspace_root).expanduser().absolute()
    stage_root = workspace / "staged_uploads"
    token_root = stage_root / _safe_stage_token(file_id)
    for candidate in (workspace, stage_root, token_root):
        if candidate.is_symlink():
            return "staging path contains a symbolic link"
    if not token_root.exists():
        return None
    try:
        for directory, dirnames, filenames in os.walk(token_root, followlinks=False):
            base = Path(directory)
            for name in (*dirnames, *filenames):
                if (base / name).is_symlink():
                    return "staging path contains a symbolic link"
    except OSError:
        return "staging path could not be safely inspected"
    return None


@dataclass(frozen=True)
class _SourceTreePreflight:
    kind: str
    entry_count: int
    size_bytes: int
    elapsed_seconds: float

    def public(self, *, catalog_size_bytes: int | None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": self.kind,
            "entry_count": self.entry_count,
            "size_bytes": self.size_bytes,
            "elapsed_seconds": self.elapsed_seconds,
            "limits": {
                "max_entries": MAX_SOURCE_TREE_ENTRIES,
                "max_bytes": MAX_SOURCE_TREE_BYTES,
                "max_seconds": MAX_SOURCE_PREFLIGHT_SECONDS,
            },
        }
        if catalog_size_bytes is not None:
            result["catalog_size_bytes"] = catalog_size_bytes
            result["catalog_size_matches_observed"] = catalog_size_bytes == self.size_bytes
        return result


def _selected_catalog_size_bytes(context: AgentRunContext, file_id: str) -> int | None:
    for descriptor in context.resource_descriptors:
        if not isinstance(descriptor, dict):
            continue
        if (
            str(descriptor.get("type") or "").strip() != "selected_resource"
            or str(descriptor.get("binding_schema") or "").strip() != "ultra.selected_resource.v1"
            or str(descriptor.get("authority") or "").strip() != "control_resource_catalog"
            or str(descriptor.get("resource_id") or "").strip() != file_id
            or str(descriptor.get("file_id") or "").strip() != file_id
        ):
            continue
        value = descriptor.get("size_bytes")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value
    return None


def _check_upload_source(
    source: Path,
) -> tuple[_SourceTreePreflight | None, tuple[str, str] | None]:
    """Bound bytes, entries, and wall time before ``copytree`` can follow the tree."""

    started_at = time.monotonic()

    def elapsed() -> float:
        return max(0.0, time.monotonic() - started_at)

    def failure(code: str, message: str) -> tuple[None, tuple[str, str]]:
        return None, (code, message)

    try:
        root_stat = source.lstat()
    except OSError:
        return failure(
            "unsafe_sensor_source",
            "selected upload source could not be safely inspected",
        )
    root_mode = root_stat.st_mode
    if stat.S_ISLNK(root_mode):
        return failure("unsafe_sensor_source", "selected upload source contains a symbolic link")
    if stat.S_ISREG(root_mode):
        if root_stat.st_size > MAX_SOURCE_TREE_BYTES:
            return failure(
                "sensor_source_budget_exceeded",
                f"selected upload exceeds the {MAX_SOURCE_TREE_BYTES} byte staging budget",
            )
        return (
            _SourceTreePreflight(
                kind="file",
                entry_count=1,
                size_bytes=root_stat.st_size,
                elapsed_seconds=elapsed(),
            ),
            None,
        )
    if not stat.S_ISDIR(root_mode):
        return failure(
            "unsafe_sensor_source",
            "selected upload source is not a regular file or directory",
        )

    pending = [source]
    entry_count = 0
    total_bytes = 0
    try:
        while pending:
            if elapsed() > MAX_SOURCE_PREFLIGHT_SECONDS:
                return failure(
                    "sensor_source_budget_exceeded",
                    "selected upload tree exceeds the staging preflight wall-time budget",
                )
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    if elapsed() > MAX_SOURCE_PREFLIGHT_SECONDS:
                        return failure(
                            "sensor_source_budget_exceeded",
                            "selected upload tree exceeds the staging preflight wall-time budget",
                        )
                    entry_count += 1
                    if entry_count > MAX_SOURCE_TREE_ENTRIES:
                        return failure(
                            "sensor_source_budget_exceeded",
                            "selected upload tree exceeds the staging entry budget",
                        )
                    if entry.is_symlink():
                        return failure(
                            "unsafe_sensor_source",
                            "selected upload tree contains a symbolic link",
                        )
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        size = entry.stat(follow_symlinks=False).st_size
                        total_bytes += size
                        if total_bytes > MAX_SOURCE_TREE_BYTES:
                            return failure(
                                "sensor_source_budget_exceeded",
                                f"selected upload tree exceeds the {MAX_SOURCE_TREE_BYTES} byte staging budget",
                            )
                    else:
                        return failure(
                            "unsafe_sensor_source",
                            "selected upload tree contains a non-regular entry",
                        )
    except OSError:
        return failure(
            "unsafe_sensor_source",
            "selected upload tree could not be safely inspected",
        )
    return (
        _SourceTreePreflight(
            kind="directory",
            entry_count=entry_count,
            size_bytes=total_bytes,
            elapsed_seconds=elapsed(),
        ),
        None,
    )


def _public_resource(staged: dict[str, Any], file_id: str) -> dict[str, Any]:
    public: dict[str, Any] = {
        "file_id": file_id,
        "kind": str(staged.get("kind") or ""),
        "sandbox_path": str(staged.get("sandbox_path") or ""),
    }
    for key in (
        "resource_id",
        "binding_schema",
        "binding_authority",
        "original_name",
        "content_type",
        "resource_kind",
        "source_type",
        "sha256",
        "size_bytes",
        "sensor_format",
    ):
        if key in staged:
            public[key] = staged[key]
    return public


def _trusted_tree_identity(
    staged: dict[str, Any],
) -> tuple[dict[str, str] | None, str | None]:
    raw = staged.get("tree_identity")
    if raw is None:
        return None, None
    if not isinstance(raw, dict):
        return None, "catalog tree identity is not a bounded object"
    top_sha256 = str(staged.get("sha256") or "").strip().lower()
    manifest_sha256 = str(raw.get("tree_manifest_sha256") or "").strip().lower()
    expected = {
        "schema": "ultra.resource-tree-identity.v1",
        "authority": "control_resource_catalog",
        "canonical_json_schema": "ultra.canonical-json.v1",
        "tree_manifest_schema": "ultra.tree-manifest.v1",
        "tree_manifest_path": ".ultra/tree-manifest.json",
        "tree_manifest_sha256": manifest_sha256,
        "scope": "all_regular_files_except_tree_manifest",
    }
    if (
        not _SHA256_RE.fullmatch(top_sha256)
        or manifest_sha256 != top_sha256
        or any(str(raw.get(key) or "").strip() != value for key, value in expected.items())
    ):
        return None, "catalog tree identity is not bound to the selected resource digest"
    return expected, None


def _selected_catalog_digest(
    context: AgentRunContext,
    resource_id: str,
) -> tuple[str | None, str | None]:
    """Resolve one linked identity only from a control-plane selected descriptor.

    Absence from the selected set is not an execution error: the format-level
    declaration remains useful, but explicitly unverified. Once the linked ID is
    selected, a missing or malformed descriptor fails closed because the run has
    claimed catalog authority that cannot be reconstructed safely.
    """

    selected = {str(value or "").strip() for value in context.selected_file_ids}
    if resource_id not in selected:
        return None, None
    candidates = [
        descriptor
        for descriptor in context.resource_descriptors
        if isinstance(descriptor, dict)
        and (
            str(descriptor.get("resource_id") or "").strip() == resource_id
            or str(descriptor.get("file_id") or "").strip() == resource_id
        )
    ]
    if len(candidates) != 1:
        return None, "selected linked resource does not have exactly one catalog descriptor"
    descriptor = candidates[0]
    if (
        str(descriptor.get("type") or "").strip() != "selected_resource"
        or str(descriptor.get("binding_schema") or "").strip() != "ultra.selected_resource.v1"
        or str(descriptor.get("authority") or "").strip() != "control_resource_catalog"
        or str(descriptor.get("resource_id") or "").strip() != resource_id
        or str(descriptor.get("file_id") or "").strip() != resource_id
    ):
        return None, "selected linked resource descriptor is not catalog-authoritative"
    digest = str(descriptor.get("sha256") or "").strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        return None, "selected linked resource descriptor has no valid catalog digest"
    return digest, None


def _bind_linked_resource_identities(
    series: Any,
    context: AgentRunContext,
    *,
    sensor_resource_id: str,
) -> tuple[Any | None, dict[str, Any] | None, tuple[str, str] | None]:
    """Bind sensor-declared links to independently authorized catalog records."""

    bound = []
    verified_count = 0
    for linked in series.linked_resources:
        if linked.resource_id == sensor_resource_id:
            return (
                None,
                None,
                (
                    "linked_resource_self_reference",
                    "sensor metadata cannot claim the selected sensor resource itself as an "
                    "independently linked resource",
                ),
            )
        catalog_digest, descriptor_error = _selected_catalog_digest(
            context,
            linked.resource_id,
        )
        if descriptor_error is not None:
            return (
                None,
                None,
                ("invalid_selected_linked_resource", descriptor_error),
            )
        if catalog_digest is None:
            bound.append(linked)
            continue
        if catalog_digest != linked.sha256:
            return (
                None,
                None,
                (
                    "linked_resource_catalog_digest_mismatch",
                    "sensor-declared linked-resource digest does not match the "
                    "tenant-authorized selected catalog resource",
                ),
            )
        bound.append(
            replace(
                linked,
                verification_status=_LINK_CATALOG_IDENTITY_VERIFIED,
                verification_authority="control_resource_catalog",
            )
        )
        verified_count += 1

    warnings = [
        warning for warning in series.warnings if not warning.startswith(_LINK_WARNING_PREFIX)
    ]
    warnings.extend(
        f"{_LINK_WARNING_PREFIX}{linked.role}:{linked.resource_id}"
        for linked in bound
        if linked.verification_status == _LINK_DECLARED_UNVERIFIED
    )
    series = replace(series, linked_resources=tuple(bound), warnings=tuple(warnings))
    declared_count = len(bound)
    sensor_tree_verified = series.lineage.status == "tree_verified"
    if declared_count == 0:
        status = "not_applicable"
    elif verified_count == declared_count and sensor_tree_verified:
        status = "verified_cross_resource_identity"
    elif verified_count == declared_count:
        status = "linked_catalog_identity_verified_sensor_tree_unverified"
    elif verified_count > 0:
        status = "partially_catalog_identity_verified"
    else:
        status = "declared_unverified"
    summary = {
        "status": status,
        "declared_count": declared_count,
        "catalog_identity_verified_count": verified_count,
        "all_linked_resources_run_selected": (
            declared_count > 0 and verified_count == declared_count
        ),
        "sensor_tree_verified": sensor_tree_verified,
        "cross_resource_identity_verified": (
            declared_count > 0 and verified_count == declared_count and sensor_tree_verified
        ),
        "authority": "control_resource_catalog" if verified_count else None,
        "scope": "resource identity only; linked content semantics are not validated",
    }
    return series, summary, None


def _series_sample_budget(series: Any) -> int:
    clocks = {clock.clock_id: clock.sample_count for clock in series.clocks}
    return sum(clocks[channel.clock_id] for channel in series.channels)


def _validation_work_estimate(series: Any) -> dict[str, int]:
    """Count the exact value/read plan implemented by the strict validator.

    This is deliberately based on parsed metadata, including unused explicit clocks and each
    repeated source/quality scan required to prove a stored multiscale level.  Admission can
    therefore fail before the second, value-reading open rather than after many bounded reads.
    Decoded bytes remain dtype-dependent and are enforced by ``SensorValidationBudget`` at the
    array boundary.
    """

    clocks = {clock.clock_id: clock for clock in series.clocks}
    channels = {channel.channel_id: channel for channel in series.channels}
    values = 0
    reads = 0

    def add_scan(count: int) -> None:
        nonlocal values, reads
        values += count
        reads += math.ceil(count / 65_536) if count else 0

    for clock in series.clocks:
        if clock.kind == "explicit":
            add_scan(clock.sample_count)
    for channel in series.channels:
        count = clocks[channel.clock_id].sample_count
        add_scan(count)
        if channel.validity_array is not None:
            add_scan(count)
        if channel.saturation_array is not None:
            add_scan(count)
        if channel.uncertainty_array is not None:
            add_scan(count)
    for level in series.multiscales:
        for envelope_channel in level.channels:
            channel = channels[envelope_channel.channel_id]
            count = clocks[channel.clock_id].sample_count
            add_scan(count)
            if channel.validity_array is not None:
                add_scan(count)
            envelope_count = math.ceil(count / level.factor)
            add_scan(envelope_count)
            add_scan(envelope_count)
    return {"decoded_values": values, "read_operations": reads}


def _bounded_series_shape(series: Any) -> str | None:
    limits = (
        ("clocks", len(series.clocks), 128),
        ("channels", len(series.channels), 256),
        ("coordinate_frames", len(series.coordinate_frames), 64),
        ("coordinate_transforms", len(series.coordinate_transforms), 128),
        ("multiscales", len(series.multiscales), 32),
        ("linked_resources", len(series.linked_resources), 128),
    )
    for label, count, limit in limits:
        if count > limit:
            return f"sensor metadata has {count} {label}; tool response limit is {limit}"
    return None


def _sanitize_validation_error(exc: SensorValidationError, root: Path) -> tuple[str, str]:
    path = exc.path
    root_text = str(root)
    if path == root_text or path.startswith(root_text + os.sep):
        path = "sensor_root"
    message = exc.message.replace(root_text, "sensor_root")
    return path, message


def _channel_envelope(
    root: Path,
    series: Any,
    *,
    channel_id: str,
    max_buckets: int,
) -> dict[str, Any]:
    if isinstance(max_buckets, bool) or not isinstance(max_buckets, int):
        return _error("invalid_envelope_budget", "max_buckets must be an integer")
    if max_buckets <= 0 or max_buckets > MAX_ENVELOPE_BUCKETS:
        return _error(
            "invalid_envelope_budget",
            f"max_buckets must be between 1 and {MAX_ENVELOPE_BUCKETS}",
        )
    channel = next((item for item in series.channels if item.channel_id == channel_id), None)
    if channel is None:
        return _error("unknown_sensor_channel", "channel_id is not declared by the sensor series")
    clock = next(item for item in series.clocks if item.clock_id == channel.clock_id)
    if clock.sample_count > MAX_GENERATED_ENVELOPE_SAMPLES:
        return _error(
            "generated_envelope_budget_exceeded",
            "channel is too large for an on-demand envelope; store a validated min/max "
            f"multiscale or use at most {MAX_GENERATED_ENVELOPE_SAMPLES} samples",
        )
    try:
        import zarr  # type: ignore[import-not-found]

        group = zarr.open_group(str(root), mode="r")
        values = group[channel.array_path][:].tolist()
        validity = group[channel.validity_array][:].tolist() if channel.validity_array else None
        saturation = (
            group[channel.saturation_array][:].tolist() if channel.saturation_array else None
        )
        envelope = build_min_max_envelope(
            values,
            max_buckets=max_buckets,
            validity=validity,
            saturation=saturation,
        )
    except SensorValidationError as exc:
        path, message = _sanitize_validation_error(exc, root)
        return _error(exc.code, message, path=path)
    except Exception:  # noqa: BLE001 - tool errors must not expose host paths/library internals
        return _error("envelope_read_failed", "the validated channel could not be read safely")
    return {
        "ok": True,
        "channel_id": channel.channel_id,
        "quantity_kind_uri": channel.quantity_kind_uri,
        "unit": asdict(channel.unit),
        "calibration_id": channel.calibration.calibration_id,
        "envelope": asdict(envelope),
    }


def inspect_selected_sensor_series_resource(
    context: AgentRunContext,
    *,
    upload_roots: tuple[str | Path, ...] = (),
    file_id: str = "",
    validate_values: bool = True,
    envelope_channel_id: str = "",
    max_buckets: int = 512,
) -> dict[str, Any]:
    """Stage and inspect one run-selected ``ultra.sensor-series.v1`` Zarr resource."""

    selected_file_id, selection_error = _selected_file_id(context, file_id)
    if selected_file_id is None:
        return _error("selected_sensor_resource_required", str(selection_error or ""))

    stage_error = _check_existing_stage_path(context, selected_file_id)
    if stage_error:
        return _error("unsafe_sensor_staging_path", stage_error)

    catalog_size_bytes = _selected_catalog_size_bytes(context, selected_file_id)
    if catalog_size_bytes is not None and catalog_size_bytes > MAX_SOURCE_TREE_BYTES:
        return _error(
            "sensor_source_budget_exceeded",
            f"cataloged resource exceeds the {MAX_SOURCE_TREE_BYTES} byte staging budget",
            preflight={
                "source_staging": {
                    "catalog_size_bytes": catalog_size_bytes,
                    "limit_bytes": MAX_SOURCE_TREE_BYTES,
                    "filesystem_scanned": False,
                    "copied": False,
                }
            },
        )

    # Locate only after the id has been proven to belong to this run.  The private locator is
    # intentionally reused so this tool cannot grow a second, subtly different upload lookup.
    roots = _resolved_upload_roots(upload_roots)
    source = _find_uploaded_file(selected_file_id, roots)
    if source is None:
        return _error("selected_sensor_resource_missing", "selected resource is unavailable")
    source_preflight, source_failure = _check_upload_source(source)
    if source_failure is not None:
        code, message = source_failure
        return _error(
            code,
            message,
            preflight={
                "source_staging": {
                    "catalog_size_bytes": catalog_size_bytes,
                    "filesystem_scanned": True,
                    "copied": False,
                    "limits": {
                        "max_entries": MAX_SOURCE_TREE_ENTRIES,
                        "max_bytes": MAX_SOURCE_TREE_BYTES,
                        "max_seconds": MAX_SOURCE_PREFLIGHT_SECONDS,
                    },
                }
            },
        )
    assert source_preflight is not None
    source_preflight_public = source_preflight.public(catalog_size_bytes=catalog_size_bytes)
    source_preflight_public["filesystem_scanned"] = True
    source_preflight_public["copied"] = False

    try:
        staged_result = stage_uploaded_files(
            context,
            upload_roots=upload_roots,
            file_ids=(selected_file_id,),
        )
    except Exception:  # noqa: BLE001 - do not echo host paths from copy failures
        return _error(
            "sensor_staging_failed",
            "selected sensor resource could not be staged",
            preflight={"source_staging": source_preflight_public},
        )
    candidates = [
        item
        for item in staged_result.get("staged_files", [])
        if isinstance(item, dict) and item.get("file_id") == selected_file_id
    ]
    if not staged_result.get("ok") or len(candidates) != 1:
        return _error(
            "sensor_staging_failed",
            "selected sensor resource could not be staged",
            preflight={"source_staging": source_preflight_public},
        )
    source_preflight_public["copied"] = True
    staged = candidates[0]
    resource = _public_resource(staged, selected_file_id)
    # Context staging reconstructs this object from a control-plane descriptor;
    # validate it again at the scientific reader boundary before granting lineage
    # authority or returning it to the model.
    tree_identity, tree_identity_error = _trusted_tree_identity(staged)
    if tree_identity_error:
        return _error(
            "invalid_catalog_tree_identity",
            tree_identity_error,
            resource=resource,
        )
    if tree_identity is not None:
        resource["tree_identity"] = tree_identity
    if staged.get("kind") != "directory":
        return _error(
            "sensor_zarr_directory_required",
            f"{SENSOR_SCHEMA} must be uploaded as a directory-format Zarr resource",
            resource=resource,
        )

    staged_path = Path(str(staged.get("staged_path") or ""))
    workspace = Path(context.workspace_root).expanduser().resolve()
    try:
        resolved_path = staged_path.resolve(strict=True)
        resolved_path.relative_to(workspace)
    except (OSError, ValueError):
        return _error(
            "unsafe_sensor_staging_path",
            "staged sensor root is outside the run workspace",
            resource=resource,
        )
    stage_error = _check_existing_stage_path(context, selected_file_id)
    if stage_error:
        return _error("unsafe_sensor_staging_path", stage_error, resource=resource)

    try:
        preflight_series = open_sensor_series(resolved_path, validate_values=False)
    except SensorValidationError as exc:
        path, message = _sanitize_validation_error(exc, resolved_path)
        return _error(exc.code, message, resource=resource, path=path)
    except Exception:  # noqa: BLE001 - do not expose host paths/library internals
        return _error(
            "sensor_open_failed",
            "selected directory could not be opened as a sensor-series Zarr resource",
            resource=resource,
        )
    if tree_identity is not None and (
        preflight_series.lineage.tree_manifest_path != tree_identity["tree_manifest_path"]
    ):
        return _error(
            "tree_identity_manifest_path_mismatch",
            "sensor lineage manifest path does not match the catalog tree identity",
            resource=resource,
        )

    shape_error = _bounded_series_shape(preflight_series)
    if shape_error:
        return _error("sensor_metadata_budget_exceeded", shape_error, resource=resource)
    bound_preflight_series, preflight_linked_lineage, linked_error = (
        _bind_linked_resource_identities(
            preflight_series,
            context,
            sensor_resource_id=selected_file_id,
        )
    )
    if linked_error is not None:
        code, message = linked_error
        return _error(code, message, resource=resource)
    if bound_preflight_series is None or preflight_linked_lineage is None:
        return _error(
            "linked_resource_binding_failed",
            "linked-resource identity binding did not produce a validation result",
            resource=resource,
        )
    preflight_series = bound_preflight_series
    sample_budget = _series_sample_budget(preflight_series)
    validation_work = _validation_work_estimate(preflight_series)
    preflight = {
        "source_staging": source_preflight_public,
        "schema": preflight_series.schema,
        "series_id": preflight_series.series_id,
        "modality": preflight_series.modality,
        "clock_count": len(preflight_series.clocks),
        "channel_count": len(preflight_series.channels),
        "aggregate_channel_samples": sample_budget,
        "validation_work_estimate": validation_work,
        "values_validated": False,
        "lineage_status": preflight_series.lineage.status,
        "linked_resource_lineage_status": preflight_linked_lineage["status"],
    }
    if not isinstance(validate_values, bool):
        return _error(
            "invalid_validation_mode",
            "validate_values must be true or false",
            resource=resource,
            preflight=preflight,
        )
    if validate_values and sample_budget > MAX_VALUE_VALIDATION_SAMPLES:
        return _error(
            "value_validation_budget_exceeded",
            "full value validation was not run; retry with validate_values=false for an "
            f"explicit metadata-only preflight or reduce the aggregate channel sample count "
            f"to {MAX_VALUE_VALIDATION_SAMPLES}",
            resource=resource,
            preflight=preflight,
        )

    series = preflight_series
    validation_budget: SensorValidationBudget | None = None
    if validate_values:
        validation_budget = SensorValidationBudget(
            max_values=DEFAULT_VALIDATION_MAX_VALUES,
            max_decoded_bytes=DEFAULT_VALIDATION_MAX_DECODED_BYTES,
            max_reads=DEFAULT_VALIDATION_MAX_READS,
            max_wall_seconds=DEFAULT_VALIDATION_MAX_WALL_SECONDS,
        )
        try:
            validation_budget.ensure_plan_fits(
                values=validation_work["decoded_values"],
                reads=validation_work["read_operations"],
            )
        except SensorValidationError as exc:
            return _error(
                exc.code,
                exc.message,
                resource=resource,
                path=exc.path,
                preflight=preflight,
            )
        try:
            series = open_sensor_series(
                resolved_path,
                validate_values=True,
                expected_tree_manifest_sha256=(
                    tree_identity["tree_manifest_sha256"] if tree_identity is not None else None
                ),
                validation_budget=validation_budget,
            )
        except SensorValidationError as exc:
            path, message = _sanitize_validation_error(exc, resolved_path)
            return _error(exc.code, message, resource=resource, path=path, preflight=preflight)
        except Exception:  # noqa: BLE001 - do not expose host paths/library internals
            return _error(
                "sensor_value_validation_failed",
                "sensor values could not be validated safely",
                resource=resource,
                preflight=preflight,
            )
    elif tree_identity is not None:
        try:
            series = open_sensor_series(
                resolved_path,
                validate_values=False,
                expected_tree_manifest_sha256=tree_identity["tree_manifest_sha256"],
            )
        except SensorValidationError as exc:
            path, message = _sanitize_validation_error(exc, resolved_path)
            return _error(exc.code, message, resource=resource, path=path, preflight=preflight)
        except Exception:  # noqa: BLE001 - do not expose host paths/library internals
            return _error(
                "sensor_lineage_validation_failed",
                "sensor tree lineage could not be validated safely",
                resource=resource,
                preflight=preflight,
            )

    bound_series, linked_resource_lineage, linked_error = _bind_linked_resource_identities(
        series,
        context,
        sensor_resource_id=selected_file_id,
    )
    if linked_error is not None:
        code, message = linked_error
        return _error(code, message, resource=resource, preflight=preflight)
    if bound_series is None or linked_resource_lineage is None:
        return _error(
            "linked_resource_binding_failed",
            "linked-resource identity binding did not produce a validation result",
            resource=resource,
            preflight=preflight,
        )
    series = bound_series

    envelope: dict[str, Any] | None = None
    requested_channel = str(envelope_channel_id or "").strip()
    if requested_channel:
        if not series.values_validated:
            return _error(
                "envelope_requires_value_validation",
                "an on-demand envelope requires validate_values=true",
                resource=resource,
                preflight=preflight,
            )
        envelope = _channel_envelope(
            resolved_path,
            series,
            channel_id=requested_channel,
            max_buckets=max_buckets,
        )
        if not envelope.get("ok"):
            envelope["resource"] = resource
            envelope["preflight"] = preflight
            return envelope

    validation_budget_summary: dict[str, Any]
    if validation_budget is not None:
        validation_budget_summary = validation_budget.snapshot()
    else:
        validation_budget_summary = {
            "status": "not_consumed_metadata_only",
            "max_values": DEFAULT_VALIDATION_MAX_VALUES,
            "max_decoded_bytes": DEFAULT_VALIDATION_MAX_DECODED_BYTES,
            "max_reads": DEFAULT_VALIDATION_MAX_READS,
            "max_wall_seconds": DEFAULT_VALIDATION_MAX_WALL_SECONDS,
        }

    result: dict[str, Any] = {
        "ok": True,
        "schema": SENSOR_INSPECTION_SCHEMA,
        "resource": resource,
        "series": asdict(series),
        "validation": {
            "source_staging": source_preflight_public,
            "metadata_validated": True,
            "values_validated": series.values_validated,
            "aggregate_channel_samples": sample_budget,
            "value_validation_sample_limit": MAX_VALUE_VALIDATION_SAMPLES,
            "validation_budget": validation_budget_summary,
            "lineage_status": series.lineage.status,
            "lineage_authority": (
                "out_of_band_tree_digest_verified"
                if series.lineage.status == "tree_verified"
                else "no_out_of_band_tree_digest"
            ),
            "tree_lineage": {
                "status": series.lineage.status,
                "authority": (
                    "out_of_band_tree_digest_verified"
                    if series.lineage.status == "tree_verified"
                    else "no_out_of_band_tree_digest"
                ),
                "scope": "selected sensor resource tree only",
            },
            "linked_resource_lineage": linked_resource_lineage,
        },
    }
    if envelope is not None:
        result["generated_envelope"] = envelope
    return result


def inspect_selected_sensor_series_text(
    context: AgentRunContext,
    *,
    upload_roots: tuple[str | Path, ...] = (),
    file_id: str = "",
    validate_values: bool = True,
    envelope_channel_id: str = "",
    max_buckets: int = 512,
) -> str:
    return json.dumps(
        inspect_selected_sensor_series_resource(
            context,
            upload_roots=upload_roots,
            file_id=file_id,
            validate_values=validate_values,
            envelope_channel_id=envelope_channel_id,
            max_buckets=max_buckets,
        ),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )


def build_sensor_tools(upload_roots: tuple[str | Path, ...] = ()) -> list[Any]:
    """Build the selected-resource sensor inspection tool."""

    resolved_upload_roots = tuple(upload_roots)

    @tool
    def inspect_selected_sensor_series(
        runtime: ToolRuntime[AgentRunContext],
        file_id: str = "",
        validate_values: bool = True,
        envelope_channel_id: str = "",
        max_buckets: int = 512,
    ) -> str:
        """Inspect a selected ultra.sensor-series.v1 Zarr resource. The tool accepts only a run-selected file ID (never a path), validates clocks/channels/units/calibration/uncertainty/quality, reports sensor-tree and linked-resource identity separately, and can return one bounded spike-preserving min/max envelope. Set validate_values=false only for an explicitly metadata-only preflight."""

        return inspect_selected_sensor_series_text(
            runtime.context,
            upload_roots=resolved_upload_roots,
            file_id=file_id,
            validate_values=validate_values,
            envelope_channel_id=envelope_channel_id,
            max_buckets=max_buckets,
        )

    return [inspect_selected_sensor_series]


__all__ = [
    "MAX_ENVELOPE_BUCKETS",
    "MAX_GENERATED_ENVELOPE_SAMPLES",
    "MAX_SOURCE_PREFLIGHT_SECONDS",
    "MAX_SOURCE_TREE_BYTES",
    "MAX_SOURCE_TREE_ENTRIES",
    "MAX_VALUE_VALIDATION_SAMPLES",
    "SENSOR_INSPECTION_SCHEMA",
    "SENSOR_FORMAT_BINDING_SCHEMA",
    "build_sensor_tools",
    "inspect_selected_sensor_series_resource",
    "inspect_selected_sensor_series_text",
    "should_register_sensor_tools",
]
