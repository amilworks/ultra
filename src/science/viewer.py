from __future__ import annotations

import colorsys
import hashlib
import json
import math
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from src.science.derivatives import file_derivative_key, get_cached_file_derivative
from src.science.imaging import _render_preview_image, load_scientific_image

ViewAxis = Literal["z", "y", "x"]

VIEWER_TILE_SIZE = 256
DEFAULT_HISTOGRAM_BINS = 64
_DIRECT_2D_LARGE_DIMENSION = 8192
_DIRECT_2D_LARGE_PIXELS = 36_000_000
_DEFAULT_CHANNEL_COLOR_HEX = (
    "#ffffff",
    "#ff0000",
    "#00ff00",
    "#0000ff",
    "#00ffff",
    "#ff00ff",
    "#ffff00",
)
_HDF5_SUFFIXES = (".h5", ".hdf5", ".he5", ".h5ebsd", ".dream3d")
_HDF5_TREE_MAX_NODES = 512
_HDF5_TREE_MAX_CHILDREN = 64
_HDF5_ROOT_ATTR_LIMIT = 24
_HDF5_DATASET_ATTR_LIMIT = 32
_HDF5_SAMPLE_TARGET = 4096
_HDF5_SAMPLE_VALUE_LIMIT = 16
_HDF5_TABLE_FIELD_LIMIT = 24
_HDF5_PREVIEW_SLICE_TARGET = 8192
_HDF5_TABLE_PREVIEW_LIMIT = 32
_HDF5_TABLE_CHART_TARGET = 256
_HDF5_MATERIALS_STRUCTURE_GROUPS = frozenset(
    {"CellData", "Grain Data", "CellFeatureData", "CellEnsembleData"}
)
_HDF5_MATERIALS_CAPABILITY_ORDER = (
    "maps",
    "orientation",
    "ipf",
    "grain_metrics",
    "topology",
    "synthetic_stats",
)
_HDF5_MATERIALS_ROLE_GROUPS = {
    "grain_id_map": "maps",
    "phase_id_map": "maps",
    "ipf_map": "maps",
    "orientation_euler": "orientation",
    "orientation_quaternion": "orientation",
    "grain_volume": "grains",
    "grain_neighbors": "grains",
    "surface_flag": "grains",
    "grain_topology": "grains",
    "ensemble_metadata": "metadata",
    "target_phase_fraction": "synthetic",
    "feature_size_distribution": "synthetic",
    "feature_size_vs_neighbors": "synthetic",
    "misorientation_bins": "synthetic",
    "odf": "synthetic",
}

_ORIENTATION_OPPOSITES = {
    "R": "L",
    "L": "R",
    "A": "P",
    "P": "A",
    "S": "I",
    "I": "S",
    "H": "F",
    "F": "H",
}

_ORDINARY_DISPLAY_IMAGE_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
)
_SCALAR_VOLUME_CACHE_VERSION = 1


def _infer_modality(*, payload: dict[str, Any], original_name: str) -> str:
    lower_name = str(original_name or "").lower()
    reader = str(payload.get("reader") or "").lower()
    metadata_payload = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    microscopy_metadata = (
        metadata_payload.get("microscopy")
        if isinstance(metadata_payload.get("microscopy"), dict)
        else {}
    )
    if lower_name.endswith(".nii") or lower_name.endswith(".nii.gz") or "nibabel" in reader:
        return "medical"
    if lower_name.endswith(_ORDINARY_DISPLAY_IMAGE_SUFFIXES):
        return "image"
    if "bioio" in reader:
        if microscopy_metadata:
            return "microscopy"
        if lower_name.endswith(
            (".ome.tif", ".ome.tiff", ".ome.zarr", ".czi", ".nd2", ".lif", ".dv", ".tif", ".tiff")
        ):
            return "microscopy"
        return "image"
    if (
        lower_name.endswith(".tif")
        or lower_name.endswith(".tiff")
        or lower_name.endswith(".ome.zarr")
    ):
        return "microscopy"
    return "image"


def is_ordinary_display_image_path(path: str | Path | None) -> bool:
    lower_name = str(path or "").strip().lower()
    return bool(lower_name and lower_name.endswith(_ORDINARY_DISPLAY_IMAGE_SUFFIXES))


def _negative_orientation_label(label: str | None, fallback: str) -> str:
    safe = str(label or "").strip().upper()
    if safe in _ORIENTATION_OPPOSITES:
        return _ORIENTATION_OPPOSITES[safe]
    if safe:
        return f"-{safe}"
    return str(fallback or "").strip() or "X"


def _positive_orientation_label(label: str | None, fallback: str) -> str:
    safe = str(label or "").strip().upper()
    if safe in _ORIENTATION_OPPOSITES:
        return safe
    if safe:
        return f"+{safe}"
    return str(fallback or "").strip() or "X"


def _orientation_axis_entry(positive_label: str | None, fallback_axis: str) -> dict[str, str]:
    positive = str(positive_label or fallback_axis).strip().upper()
    if not positive:
        positive = str(fallback_axis).strip().upper() or "X"
    return {
        "positive": positive,
        "negative": _negative_orientation_label(positive, f"-{positive}"),
    }


def _default_orientation_axis_labels() -> dict[str, dict[str, str]]:
    return {
        "x": _orientation_axis_entry("X", "X"),
        "y": _orientation_axis_entry("Y", "Y"),
        "z": _orientation_axis_entry("Z", "Z"),
    }


def _normalize_orientation_axis_labels(value: Any) -> dict[str, dict[str, str]]:
    source = value if isinstance(value, dict) else {}
    normalized = _default_orientation_axis_labels()
    for axis in ("x", "y", "z"):
        entry = source.get(axis) if isinstance(source, dict) else None
        if not isinstance(entry, dict):
            continue
        normalized[axis] = {
            "positive": _orientation_axis_entry(entry.get("positive"), axis)["positive"],
            "negative": _orientation_axis_entry(entry.get("positive"), axis)["negative"]
            if entry.get("negative") in (None, "")
            else str(entry.get("negative")).strip().upper(),
        }
    return normalized


def _axis_positive_label(
    axis_labels: dict[str, dict[str, str]], axis_name: str, fallback: str
) -> str:
    key = str(axis_name or "").strip().lower()
    entry = axis_labels.get(key)
    if isinstance(entry, dict) and str(entry.get("positive") or "").strip():
        return str(entry["positive"]).strip().upper()
    return str(fallback or axis_name or "X").strip().upper()


def _build_orientation_labels(
    *,
    row_axis: str,
    col_axis: str,
    slice_axis: str | None,
) -> dict[str, str | None]:
    row = str(row_axis or "Y").upper()
    col = str(col_axis or "X").upper()
    depth = str(slice_axis or "").upper() or None
    return {
        "top": _negative_orientation_label(row, f"-{row}"),
        "bottom": _positive_orientation_label(row, f"+{row}"),
        "left": _negative_orientation_label(col, f"-{col}"),
        "right": _positive_orientation_label(col, f"+{col}"),
        "front": _negative_orientation_label(depth, f"-{depth}") if depth else None,
        "back": _positive_orientation_label(depth, f"+{depth}") if depth else None,
    }


def _has_positive_spacing(value: dict[str, Any] | None) -> bool:
    if not isinstance(value, dict):
        return False
    for axis in ("z", "y", "x"):
        try:
            numeric = float(value.get(axis))
        except Exception:
            numeric = float("nan")
        if math.isfinite(numeric) and numeric > 0:
            return True
    return False


def _measurement_policy(*, orientation_frame: str, physical_spacing: dict[str, Any] | None) -> str:
    has_spacing = _has_positive_spacing(physical_spacing)
    safe_frame = str(orientation_frame or "").strip().lower()
    if has_spacing and safe_frame in {"patient", "geospatial"}:
        return "orientation-aware"
    if has_spacing:
        return "spacing-aware"
    return "pixel-only"


def _image_render_policy(
    *,
    payload: dict[str, Any],
    modality: str,
    axis_sizes: dict[str, Any],
) -> str:
    semantic_kind = str(payload.get("semantic_kind") or "").strip().lower()
    if semantic_kind == "label":
        return "categorical"
    if semantic_kind in {"display", "rgb"}:
        return "display"
    if semantic_kind in {"vector", "analysis"}:
        return "analysis"

    metadata_payload = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    header_payload = (
        metadata_payload.get("header") if isinstance(metadata_payload.get("header"), dict) else {}
    )
    channel_count = max(1, int(axis_sizes.get("C") or 1))
    is_volume = bool(payload.get("is_volume")) or max(1, int(axis_sizes.get("Z") or 1)) > 1
    dtype_name = str(payload.get("array_dtype") or "uint8")
    color_mode = (
        str(header_payload.get("Color mode") or header_payload.get("Mode") or "").strip().upper()
    )
    if (
        channel_count in {3, 4}
        and color_mode in {"RGB", "RGBA"}
        and _dtype_format(dtype_name) == "u"
        and _dtype_bits(dtype_name) <= 16
    ):
        return "display"
    if (
        not is_volume
        and channel_count in {3, 4}
        and _dtype_format(dtype_name) == "u"
        and _dtype_bits(dtype_name) <= 16
    ):
        return "display"
    if modality == "medical":
        return "scalar"
    if modality == "microscopy":
        return "scalar"
    if channel_count == 1:
        return "scalar"
    if channel_count in {3, 4}:
        return "display"
    return "display"


def _hdf_render_policy(preview_kind: str | None) -> str:
    safe_kind = str(preview_kind or "").strip().lower()
    if safe_kind == "scalar_volume":
        return "scalar"
    if safe_kind == "label_volume":
        return "categorical"
    if safe_kind == "rgb_volume":
        return "display"
    return "analysis"


def _diagnostic_surface(
    *,
    render_policy: str,
    modality: str | None,
    is_volume: bool,
) -> str:
    if (
        bool(is_volume)
        and str(render_policy) == "scalar"
        and str(modality or "").strip().lower() == "medical"
    ):
        return "mpr"
    return "none"


def _image_display_capabilities(
    *,
    render_policy: str,
    modality: str,
    is_volume: bool,
    native_volume_supported: bool,
    channel_count: int,
    measurement_policy: str,
) -> list[str]:
    capabilities: list[str] = []
    safe_policy = str(render_policy or "").strip().lower()
    safe_modality = str(modality or "").strip().lower()

    if safe_policy == "scalar":
        capabilities.extend(["slice_navigation", "histogram"])
        if native_volume_supported:
            capabilities.append("volume_context")
        if measurement_policy != "pixel-only":
            capabilities.append("physical_scale")
        if safe_modality == "medical":
            capabilities.extend(["window_level", "scalar_probe"])
            if is_volume:
                capabilities.append("diagnostic_mpr")
        elif safe_modality == "microscopy":
            if channel_count > 1:
                capabilities.extend(["channel_mix", "channel_visibility", "channel_color"])
                if is_volume:
                    capabilities.append("volume_channel_selection")
            else:
                capabilities.append("intensity_window")
        else:
            capabilities.append("intensity_window")
    elif safe_policy == "categorical":
        capabilities.extend(["slice_navigation", "palette"])
        if native_volume_supported:
            capabilities.append("volume_context")
    elif safe_policy == "display":
        capabilities.extend(["slice_navigation", "display_composite"])
    else:
        capabilities.extend(["slice_navigation", "analysis"])

    deduped: list[str] = []
    for capability in capabilities:
        token = str(capability).strip()
        if token and token not in deduped:
            deduped.append(token)
    return deduped


def _hdf_display_capabilities(
    *,
    preview_kind: str | None,
    render_policy: str,
    volume_eligible: bool,
) -> list[str]:
    safe_kind = str(preview_kind or "").strip().lower()
    safe_policy = str(render_policy or "").strip().lower()
    if safe_kind == "scalar":
        return ["value"]
    if safe_kind in {"series", "table"}:
        return ["plot", "tabular"]
    if safe_policy == "scalar":
        output = ["slice_navigation", "histogram", "sample_values"]
        if volume_eligible:
            output.append("volume_context")
        return output
    if safe_policy == "categorical":
        output = ["slice_navigation", "histogram", "palette"]
        if volume_eligible:
            output.append("volume_context")
        return output
    if safe_policy == "display":
        return ["slice_navigation", "display_composite"]
    return ["slice_navigation", "component_analysis", "histogram"]


def _texture_policy(render_policy: str) -> str:
    safe_policy = str(render_policy or "").strip().lower()
    if safe_policy in {"categorical", "analysis"}:
        return "nearest"
    return "linear"


def _image_delivery_mode(*, backend_mode: str, tile_pyramid: str | None, is_volume: bool) -> str:
    safe_backend = str(backend_mode or "").strip().lower()
    safe_tile_pyramid = str(tile_pyramid or "").strip().lower()
    if safe_backend == "scalar":
        return "scalar"
    if safe_backend == "atlas":
        return "atlas"
    if not is_volume and safe_tile_pyramid == "deferred":
        return "deferred_multiscale"
    return "direct"


def _hdf_delivery_mode(*, render_policy: str, volume_eligible: bool) -> str:
    if not bool(volume_eligible):
        return "direct"
    safe_policy = str(render_policy or "").strip().lower()
    if safe_policy == "scalar":
        return "scalar"
    if safe_policy in {"categorical", "display"}:
        return "atlas"
    return "direct"


def _first_paint_mode(*, default_surface: str, delivery_mode: str) -> str:
    safe_surface = str(default_surface or "").strip().lower()
    safe_delivery = str(delivery_mode or "").strip().lower()
    if safe_surface == "volume" or (
        safe_delivery in {"scalar", "atlas"} and safe_surface not in {"2d", "mpr", "metadata"}
    ):
        return "webgl"
    return "image"


def _viewer_capabilities(
    *,
    delivery_mode: str,
    first_paint_mode: str,
    diagnostic_surface: str,
    texture_policy: str,
    display_capabilities: list[str] | tuple[str, ...],
) -> list[str]:
    capabilities: list[str] = []
    if str(first_paint_mode or "").strip().lower() == "webgl":
        capabilities.append("webgl_first_paint")
    else:
        capabilities.append("image_first_paint")

    safe_delivery = str(delivery_mode or "").strip().lower()
    if safe_delivery == "scalar":
        capabilities.append("scalar_volume_delivery")
    elif safe_delivery == "atlas":
        capabilities.append("atlas_volume_delivery")
    elif safe_delivery == "deferred_multiscale":
        capabilities.append("deferred_multiscale")
    else:
        capabilities.append("direct_delivery")

    if str(texture_policy or "").strip().lower() == "nearest":
        capabilities.append("nearest_sampling")
    else:
        capabilities.append("linear_sampling")

    if str(diagnostic_surface or "").strip().lower() == "mpr":
        capabilities.append("mpr_truth_surface")

    capabilities.extend(str(item).strip() for item in display_capabilities if str(item).strip())

    deduped: list[str] = []
    for capability in capabilities:
        if capability and capability not in deduped:
            deduped.append(capability)
    return deduped


def normalize_view_axis(value: str | None) -> ViewAxis:
    axis = str(value or "z").strip().lower()
    if axis not in {"z", "y", "x"}:
        return "z"
    return axis  # type: ignore[return-value]


def _finite_spacing(value: Any, default: float = 1.0) -> float:
    try:
        numeric = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(numeric) or numeric <= 0:
        return float(default)
    return float(numeric)


def _normalize_physical_spacing(value: Any) -> dict[str, float | None] | None:
    if not isinstance(value, dict):
        return None
    normalized = {
        "z": None,
        "y": None,
        "x": None,
    }
    has_positive_value = False
    for axis in ("z", "y", "x"):
        try:
            numeric = float(value.get(axis))
        except Exception:
            numeric = float("nan")
        if math.isfinite(numeric) and numeric > 0:
            normalized[axis] = float(numeric)
            has_positive_value = True
    return normalized if has_positive_value else None


def _dtype_bits(dtype_name: str) -> int:
    try:
        return int(np.dtype(dtype_name).itemsize * 8)
    except Exception:
        return 8


def _dtype_format(dtype_name: str) -> str:
    try:
        dtype = np.dtype(dtype_name)
    except Exception:
        return "u"
    if np.issubdtype(dtype, np.floating):
        return "f"
    if np.issubdtype(dtype, np.signedinteger):
        return "s"
    return "u"


def _channel_palette(channel_count: int) -> list[dict[str, Any]]:
    count = max(1, int(channel_count or 1))
    if count == 1:
        colors = ["#ffffff"]
    elif count == 2:
        colors = ["#ffffff", "#ff0000"]
    else:
        colors = list(_DEFAULT_CHANNEL_COLOR_HEX[: min(count, len(_DEFAULT_CHANNEL_COLOR_HEX))])
    while len(colors) < count:
        colors.append("#ffffff")
    output: list[dict[str, Any]] = []
    for index, color in enumerate(colors[:count]):
        output.append({"index": index, "hex": color, "rgb": _hex_to_rgb(color)})
    return output


def _hex_to_rgb(color: str) -> list[int]:
    value = str(color or "").strip().lstrip("#")
    if len(value) != 6:
        return [255, 255, 255]
    try:
        return [int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)]
    except Exception:
        return [255, 255, 255]


def _channel_names(channel_count: int) -> list[str]:
    count = max(1, int(channel_count or 1))
    if count == 1:
        return ["Intensity"]
    if count == 2:
        return ["Channel 1", "Channel 2"]
    if count == 3:
        return ["Red", "Green", "Blue"]
    return [f"Ch{i + 1}" for i in range(count)]


def _default_display_channels(channel_count: int) -> list[int]:
    count = max(1, int(channel_count or 1))
    if count == 1:
        return [0, 0, 0]
    if count == 2:
        return [0, 1, -1]
    return [0, 1, 2]


def _build_plane_descriptor(
    *,
    axis: ViewAxis,
    axis_sizes: dict[str, Any],
    physical_spacing: dict[str, Any] | None,
) -> dict[str, Any]:
    sizes = {key: max(1, int(axis_sizes.get(key) or 1)) for key in ("T", "C", "Z", "Y", "X")}
    spacing = physical_spacing if isinstance(physical_spacing, dict) else {}
    x_spacing = _finite_spacing(spacing.get("x"), 1.0)
    y_spacing = _finite_spacing(spacing.get("y"), 1.0)
    z_spacing = _finite_spacing(spacing.get("z"), 1.0)

    if axis == "x":
        pixel_width = int(sizes["Y"])
        pixel_height = int(sizes["Z"])
        plane_axes = ["Z", "Y"]
        row_spacing = z_spacing
        col_spacing = y_spacing
        label = "YZ plane"
    elif axis == "y":
        pixel_width = int(sizes["X"])
        pixel_height = int(sizes["Z"])
        plane_axes = ["Z", "X"]
        row_spacing = z_spacing
        col_spacing = x_spacing
        label = "XZ plane"
    else:
        pixel_width = int(sizes["X"])
        pixel_height = int(sizes["Y"])
        plane_axes = ["Y", "X"]
        row_spacing = y_spacing
        col_spacing = x_spacing
        label = "XY plane"

    world_width = max(1.0, float(pixel_width) * col_spacing)
    world_height = max(1.0, float(pixel_height) * row_spacing)
    return {
        "axis": axis,
        "label": label,
        "axes": plane_axes,
        "pixel_size": {"width": pixel_width, "height": pixel_height},
        "spacing": {"row": row_spacing, "col": col_spacing},
        "world_size": {"width": world_width, "height": world_height},
        "aspect_ratio": float(world_width / max(1e-9, world_height)),
    }


def build_tile_levels(
    width: int, height: int, tile_size: int = VIEWER_TILE_SIZE
) -> list[dict[str, Any]]:
    full_width = max(1, int(width))
    full_height = max(1, int(height))
    tile = max(64, int(tile_size))
    levels: list[tuple[int, int]] = [(full_width, full_height)]
    while max(levels[-1]) > tile:
        prev_width, prev_height = levels[-1]
        next_width = max(1, int(math.ceil(prev_width / 2.0)))
        next_height = max(1, int(math.ceil(prev_height / 2.0)))
        if next_width == prev_width and next_height == prev_height:
            break
        levels.append((next_width, next_height))
    levels.reverse()

    output: list[dict[str, Any]] = []
    for level_id, (level_width, level_height) in enumerate(levels):
        downsample = max(
            float(full_width) / float(level_width), float(full_height) / float(level_height)
        )
        output.append(
            {
                "level": int(level_id),
                "width": int(level_width),
                "height": int(level_height),
                "columns": int(math.ceil(level_width / tile)),
                "rows": int(math.ceil(level_height / tile)),
                "downsample": float(round(downsample, 4)),
            }
        )
    return output


def _build_atlas_scheme(
    *,
    slice_width: int,
    slice_height: int,
    slice_count: int,
    max_dimension: int,
) -> dict[str, Any]:
    count = max(1, int(slice_count or 1))
    width = max(1, int(slice_width or 1))
    height = max(1, int(slice_height or 1))
    limit = max(256, int(max_dimension or 2048))

    best: dict[str, Any] | None = None
    best_score: tuple[float, float, float] | None = None
    for columns in range(1, count + 1):
        rows = int(math.ceil(count / columns))
        scale = min(limit / float(columns * width), limit / float(rows * height), 1.0)
        scaled_width = max(1, int(math.floor(width * scale)))
        scaled_height = max(1, int(math.floor(height * scale)))
        atlas_width = max(1, scaled_width * columns)
        atlas_height = max(1, scaled_height * rows)
        balance_penalty = abs(columns - rows)
        area_penalty = float(atlas_width * atlas_height)
        score = (scale, -balance_penalty, -area_penalty)
        if best_score is None or score > best_score:
            best_score = score
            best = {
                "slice_count": count,
                "columns": columns,
                "rows": rows,
                "slice_width": scaled_width,
                "slice_height": scaled_height,
                "atlas_width": atlas_width,
                "atlas_height": atlas_height,
                "downsample": float(
                    round(max(width / float(scaled_width), height / float(scaled_height)), 4)
                ),
                "format": "png",
            }
    return best or {
        "slice_count": count,
        "columns": 1,
        "rows": count,
        "slice_width": width,
        "slice_height": height,
        "atlas_width": width,
        "atlas_height": height * count,
        "downsample": 1.0,
        "format": "png",
    }


def _extract_metadata_payload(payload: dict[str, Any]) -> dict[str, Any]:
    metadata_payload = payload.get("metadata")
    return dict(metadata_payload) if isinstance(metadata_payload, dict) else {}


def _build_phys(*, payload: dict[str, Any], file_id: str, original_name: str) -> dict[str, Any]:
    axis_sizes = payload.get("axis_sizes") or {"T": 1, "C": 1, "Z": 1, "Y": 1, "X": 1}
    spacing = _normalize_physical_spacing(payload.get("physical_spacing")) or {}
    metadata_payload = _extract_metadata_payload(payload)
    modality = _infer_modality(payload=payload, original_name=original_name)
    channel_count = max(1, int(axis_sizes.get("C") or 1))
    channel_palette = _channel_palette(channel_count)
    microscopy_metadata = (
        metadata_payload.get("microscopy")
        if isinstance(metadata_payload.get("microscopy"), dict)
        else {}
    )
    metadata_channel_names = (
        microscopy_metadata.get("channel_names") if isinstance(microscopy_metadata, dict) else None
    )
    if isinstance(metadata_channel_names, list):
        channel_names = [str(name) for name in metadata_channel_names[:channel_count]]
    else:
        channel_names = _channel_names(channel_count)
    while len(channel_names) < channel_count:
        channel_names.append(f"Ch{len(channel_names) + 1}")
    dtype_name = str(payload.get("array_dtype") or "uint8")
    pixel_units = ["px", "px", "px", "frame"]
    if spacing:
        pixel_units = ["x-unit", "y-unit", "z-unit", "frame"]
    dicom = (
        metadata_payload.get("dicom") if isinstance(metadata_payload.get("dicom"), dict) else None
    )
    window_center = None
    window_width = None
    if dicom:
        try:
            window_center = (
                float(dicom.get("wnd_center")) if dicom.get("wnd_center") is not None else None
            )
        except Exception:
            window_center = None
        try:
            window_width = (
                float(dicom.get("wnd_width")) if dicom.get("wnd_width") is not None else None
            )
        except Exception:
            window_width = None
    if modality == "medical" and window_center is None and window_width is None:
        array_min = float(payload.get("array_min") or 0.0)
        array_max = float(payload.get("array_max") or 0.0)
        if array_max > array_min:
            window_center = (array_min + array_max) / 2.0
            window_width = array_max - array_min

    return {
        "resource_uniq": file_id,
        "name": original_name,
        "x": int(axis_sizes.get("X") or 1),
        "y": int(axis_sizes.get("Y") or 1),
        "z": int(axis_sizes.get("Z") or 1),
        "t": int(axis_sizes.get("T") or 1),
        "ch": channel_count,
        "pixel_depth": _dtype_bits(dtype_name),
        "pixel_format": _dtype_format(dtype_name),
        "pixel_size": [
            _finite_spacing(spacing.get("x"), 1.0),
            _finite_spacing(spacing.get("y"), 1.0),
            _finite_spacing(spacing.get("z"), 1.0),
            1.0,
        ],
        "pixel_units": pixel_units,
        "channel_names": channel_names,
        "display_channels": _default_display_channels(channel_count),
        "channel_colors": channel_palette,
        "units": "physical" if spacing else "pixel",
        "dicom": {
            "modality": str(dicom.get("modality"))
            if dicom and dicom.get("modality") is not None
            else None,
            "wnd_center": window_center,
            "wnd_width": window_width,
        },
        "geo": metadata_payload.get("geo")
        if isinstance(metadata_payload.get("geo"), dict)
        else None,
        "coordinates": metadata_payload.get("coordinates")
        if isinstance(metadata_payload.get("coordinates"), dict)
        else None,
    }


def _build_display_defaults(
    *, payload: dict[str, Any], phys: dict[str, Any], modality: str
) -> dict[str, Any]:
    dicom = phys.get("dicom") if isinstance(phys.get("dicom"), dict) else {}
    if (
        modality == "medical"
        and dicom.get("wnd_center") is not None
        and dicom.get("wnd_width") is not None
    ):
        enhancement = f"hounsfield:{float(dicom['wnd_center']):.3f}:{float(dicom['wnd_width']):.3f}"
    else:
        enhancement = "d"
    channel_count = max(1, int(phys.get("ch") or len(phys.get("channel_names") or []) or 1))
    selected_indices = payload.get("selected_indices") or {}
    default_volume_channel = max(
        0,
        min(
            int(selected_indices.get("C") or 0),
            channel_count - 1,
        ),
    )
    return {
        "enhancement": enhancement,
        "negative": False,
        "rotate": 0,
        "fusion_method": "m",
        "channel_mode": "composite",
        "channels": [index for index in phys.get("display_channels", [0, 1, 2]) if int(index) >= 0],
        "channel_colors": [entry.get("hex") for entry in phys.get("channel_colors", [])],
        "time_index": int((payload.get("selected_indices") or {}).get("T") or 0),
        "z_index": int((payload.get("selected_indices") or {}).get("Z") or 0),
        "volume_channel": default_volume_channel,
        "volume_clip_min": {"x": 0.0, "y": 0.0, "z": 0.0},
        "volume_clip_max": {"x": 1.0, "y": 1.0, "z": 1.0},
    }


def _select_2d_delivery_policy(*, axis_sizes: dict[str, Any], is_volume: bool) -> dict[str, Any]:
    if is_volume:
        return {
            "backend_mode": "atlas",
            "tile_pyramid": "lazy",
            "warning": None,
        }
    width = max(1, int(axis_sizes.get("X") or 1))
    height = max(1, int(axis_sizes.get("Y") or 1))
    is_large_2d = (
        max(width, height) >= _DIRECT_2D_LARGE_DIMENSION
        or (width * height) >= _DIRECT_2D_LARGE_PIXELS
    )
    return {
        "backend_mode": "direct",
        "tile_pyramid": "deferred" if is_large_2d else "none",
        "warning": (
            "Large 2D image delivery is using the direct full-resolution path until a prepared pyramid is available."
            if is_large_2d
            else None
        ),
    }


def is_hdf5_viewer_path(path: str | Path | None) -> bool:
    name = str(path or "").strip().lower()
    return bool(name and any(name.endswith(suffix) for suffix in _HDF5_SUFFIXES))


def _load_h5py_module():
    try:
        import h5py  # type: ignore

        return h5py, None
    except Exception as exc:  # pragma: no cover - exercised indirectly when dependency is absent
        return None, exc


def _decode_hdf_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.decode("latin-1", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_safe_hdf_value(value: Any, *, max_items: int = 16) -> Any:
    value = _decode_hdf_scalar(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        if int(value.size) > max_items:
            return f"array(shape={list(value.shape)}, dtype={value.dtype})"
        return [_json_safe_hdf_value(item, max_items=max_items) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        if len(value) > max_items:
            return f"sequence(len={len(value)})"
        return [_json_safe_hdf_value(item, max_items=max_items) for item in value]
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                output["..."] = f"{len(value) - max_items} more entries"
                break
            output[str(key)] = _json_safe_hdf_value(item, max_items=max_items)
        return output
    return str(value)


def _hdf_shape_list(shape: Any) -> list[int]:
    try:
        return [int(value) for value in tuple(shape)]
    except Exception:
        return []


def _safe_hdf_group_keys(group: Any) -> list[str]:
    try:
        return [str(key) for key in list(group.keys())]
    except Exception:
        return []


def _safe_hdf_dataset(handle: Any, path: str | None) -> Any | None:
    if not path:
        return None
    try:
        node = handle.get(path)
    except Exception:
        return None
    if node is None or not hasattr(node, "shape"):
        return None
    return node


def _iter_hdf_data_container_paths(handle: Any) -> list[str]:
    root = handle.get("/DataContainers")
    if root is None:
        return []
    output: list[str] = []
    for key in _safe_hdf_group_keys(root):
        output.append(f"/DataContainers/{key}")
    return output


def _find_first_hdf_dataset(
    handle: Any, container_paths: list[str], relatives: list[str]
) -> str | None:
    for container_path in container_paths:
        for relative in relatives:
            candidate = f"{container_path}/{relative}"
            if _safe_hdf_dataset(handle, candidate) is not None:
                return candidate
    return None


def _read_hdf_string_dataset(handle: Any, paths: list[str]) -> list[str]:
    for path in paths:
        dataset = _safe_hdf_dataset(handle, path)
        if dataset is None:
            continue
        try:
            values = np.asarray(dataset[()]).reshape(-1)
        except Exception:
            continue
        output: list[str] = []
        for value in values:
            decoded = _decode_hdf_scalar(value)
            text = str(decoded).strip()
            if text:
                output.append(text)
        if output:
            return output
    return []


def _read_hdf_feature_count(handle: Any, paths: list[str]) -> int | None:
    for path in paths:
        dataset = _safe_hdf_dataset(handle, path)
        if dataset is None:
            continue
        try:
            values = np.asarray(dataset[()]).reshape(-1)
        except Exception:
            continue
        finite = values[np.isfinite(values)] if np.issubdtype(values.dtype, np.number) else values
        if getattr(finite, "size", 0) == 0:
            continue
        try:
            positive = np.asarray(finite, dtype=np.float64)
            positive = positive[positive > 0]
            if positive.size == 0:
                continue
            return int(round(float(np.sum(positive))))
        except Exception:
            continue
    return None


def _has_hdf_zero_sentinel(dataset: Any) -> bool:
    shape = _hdf_shape_list(getattr(dataset, "shape", ()))
    if not shape or int(shape[0]) <= 0:
        return False
    try:
        first = np.asarray(dataset[(0,) + (slice(None),) * max(0, dataset.ndim - 1)])
    except Exception:
        return False
    try:
        return bool(np.allclose(first.astype(np.float64), 0.0))
    except Exception:
        return False


def _read_hdf_row_count(handle: Any, paths: list[str]) -> int | None:
    for path in paths:
        dataset = _safe_hdf_dataset(handle, path)
        if dataset is None:
            continue
        shape = _hdf_shape_list(getattr(dataset, "shape", ()))
        if not shape:
            continue
        count = int(shape[0])
        if count > 1 and _has_hdf_zero_sentinel(dataset):
            count -= 1
        if count > 0:
            return count
    return None


def _sample_hdf_row_indices(
    row_count: int, *, target_rows: int = 512, drop_first: bool = False
) -> np.ndarray:
    total = max(0, int(row_count))
    if total <= 0:
        return np.asarray([], dtype=np.int64)
    start = 1 if drop_first and total > 1 else 0
    if total - start <= target_rows:
        return np.arange(start, total, dtype=np.int64)
    step = max(1, int(math.ceil((total - start) / float(target_rows))))
    return np.arange(start, total, step, dtype=np.int64)[:target_rows]


def _sample_hdf_rows(
    dataset: Any, *, target_rows: int = 512, drop_first: bool = False
) -> np.ndarray:
    shape = _hdf_shape_list(getattr(dataset, "shape", ()))
    if not shape:
        return np.asarray(dataset[()])
    indices = _sample_hdf_row_indices(shape[0], target_rows=target_rows, drop_first=drop_first)
    if indices.size == 0:
        return np.asarray([])
    return np.asarray(dataset[indices])


def _sample_hdf_numeric_column(
    dataset: Any,
    *,
    component: int = 0,
    target_rows: int = 4096,
    drop_first: bool = False,
) -> np.ndarray:
    sampled = _sample_hdf_rows(dataset, target_rows=target_rows, drop_first=drop_first)
    array = np.asarray(sampled)
    if array.ndim == 0:
        values = array.reshape(1)
    elif array.ndim == 1:
        values = array
    else:
        component_index = max(0, min(int(component), max(0, int(array.shape[-1]) - 1)))
        values = array[..., component_index].reshape(-1)
    try:
        numeric = np.asarray(values, dtype=np.float64)
    except Exception:
        return np.asarray([], dtype=np.float64)
    return numeric[np.isfinite(numeric)]


def _materials_semantic_role(dataset_path: str | None) -> str | None:
    lower_path = str(dataset_path or "").lower()
    if lower_path.endswith("/featureids"):
        return "grain_id_map"
    if lower_path.endswith("/phases"):
        if "/celldata/" in lower_path or "/cellfeaturedata/" in lower_path:
            return "phase_id_map"
        return "ensemble_metadata"
    if lower_path.endswith("/ipfcolor"):
        return "ipf_map"
    if "eulerangles" in lower_path:
        return "orientation_euler"
    if "quat" in lower_path:
        return "orientation_quaternion"
    if lower_path.endswith("/volumes"):
        return "grain_volume"
    if lower_path.endswith("/numneighbors"):
        return "grain_neighbors"
    if lower_path.endswith("/surfacefeatures"):
        return "surface_flag"
    if lower_path.endswith("/neighborlist") or lower_path.endswith("/sharedsurfacearealist"):
        return "grain_topology"
    if (
        lower_path.endswith("/phasename")
        or lower_path.endswith("/crystalstructures")
        or lower_path.endswith("/numfeatures")
    ):
        return "ensemble_metadata"
    if lower_path.endswith("/phasefraction"):
        return "target_phase_fraction"
    if "featuresize distribution" in lower_path:
        return "feature_size_distribution"
    if "featuresize vs neighbors distributions" in lower_path:
        return "feature_size_vs_neighbors"
    if lower_path.endswith("/misorientationbins"):
        return "misorientation_bins"
    if lower_path.endswith("/odf"):
        return "odf"
    return None


def _materials_units_hint(semantic_role: str | None) -> str | None:
    if semantic_role == "orientation_euler":
        return "radians"
    if semantic_role == "orientation_quaternion":
        return "unit quaternion"
    return None


def _materials_domain_tags(dataset_path: str | None, semantic_role: str | None) -> list[str]:
    tags = ["materials"]
    if semantic_role:
        tags.append(str(semantic_role))
    group = _HDF5_MATERIALS_ROLE_GROUPS.get(str(semantic_role or ""))
    if group:
        tags.append(group)
    lower_path = str(dataset_path or "").lower()
    if "statsgeneratordatacontainer" in lower_path:
        tags.extend(["dream3d", "synthetic"])
    elif "datacontainers" in lower_path and "_simpl_geometry" not in lower_path:
        tags.extend(["dream3d", "measured"])
    deduped: list[str] = []
    for tag in tags:
        safe = str(tag).strip()
        if safe and safe not in deduped:
            deduped.append(safe)
    return deduped


def _build_materials_histogram_chart(
    *,
    title: str,
    values: np.ndarray,
    source_paths: list[str],
    description: str,
    units_hint: str | None = None,
    bins: int = 24,
    discrete: bool = False,
    x_key: str = "label",
    y_key: str = "count",
) -> dict[str, Any] | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    data: list[dict[str, Any]] = []
    if discrete:
        unique, counts = np.unique(finite.astype(np.int64), return_counts=True)
        for value, count in zip(unique.tolist()[:32], counts.tolist()[:32]):
            data.append({x_key: str(value), y_key: int(count)})
    else:
        hist, edges = np.histogram(finite, bins=max(8, int(bins)))
        for index, count in enumerate(hist.tolist()):
            start = float(edges[index])
            end = float(edges[index + 1])
            data.append(
                {x_key: f"{start:.2f}-{end:.2f}", y_key: int(count), "start": start, "end": end}
            )
    return {
        "kind": "histogram",
        "title": title,
        "description": description,
        "x_key": x_key,
        "y_key": y_key,
        "data": data,
        "source_paths": source_paths,
        "units_hint": units_hint,
        "provenance": "Bounded sampled preview from the HDF5 dataset.",
    }


def _build_materials_bar_chart(
    *,
    title: str,
    data: list[dict[str, Any]],
    source_paths: list[str],
    description: str,
    x_key: str,
    y_key: str,
    units_hint: str | None = None,
    provenance: str | None = None,
) -> dict[str, Any] | None:
    if not data:
        return None
    return {
        "kind": "bar",
        "title": title,
        "description": description,
        "x_key": x_key,
        "y_key": y_key,
        "data": data,
        "source_paths": source_paths,
        "units_hint": units_hint,
        "provenance": provenance or "Bounded preview derived from the HDF5 dataset.",
    }


def _build_materials_scatter_chart(
    *,
    title: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
    source_paths: list[str],
    description: str,
    x_key: str,
    y_key: str,
    units_hint: str | None = None,
) -> dict[str, Any] | None:
    if x_values.size == 0 or y_values.size == 0:
        return None
    sample_count = min(int(x_values.size), int(y_values.size), 256)
    if sample_count <= 0:
        return None
    data = [
        {x_key: float(x_values[index]), y_key: float(y_values[index])}
        for index in range(sample_count)
        if math.isfinite(float(x_values[index])) and math.isfinite(float(y_values[index]))
    ]
    if not data:
        return None
    return {
        "kind": "scatter",
        "title": title,
        "description": description,
        "x_key": x_key,
        "y_key": y_key,
        "data": data,
        "source_paths": source_paths,
        "units_hint": units_hint,
        "provenance": "Bounded sampled rows from aligned HDF5 datasets.",
    }


def _detect_hdf5_materials(handle: Any) -> dict[str, Any]:
    container_paths = _iter_hdf_data_container_paths(handle)
    primary_containers: list[str] = []
    for container_path in container_paths:
        group = handle.get(container_path)
        child_names = set(_safe_hdf_group_keys(group))
        if "_SIMPL_GEOMETRY" in child_names and bool(
            child_names & _HDF5_MATERIALS_STRUCTURE_GROUPS
        ):
            primary_containers.append(container_path)
    if not primary_containers:
        return {
            "detected": False,
            "schema": None,
            "capabilities": [],
            "roles": {},
            "phase_names": [],
            "feature_count": None,
            "grain_count": None,
            "recommended_view": "explorer",
        }

    ordered_containers = primary_containers + [
        path for path in container_paths if path not in primary_containers
    ]
    roles: dict[str, str] = {}
    roles["grain_id_map"] = (
        _find_first_hdf_dataset(
            handle, ordered_containers, ["CellData/FeatureIds", "CellFeatureData/FeatureIds"]
        )
        or ""
    )
    roles["phase_id_map"] = (
        _find_first_hdf_dataset(
            handle, ordered_containers, ["CellData/Phases", "CellFeatureData/Phases"]
        )
        or ""
    )
    roles["ipf_map"] = (
        _find_first_hdf_dataset(handle, ordered_containers, ["CellData/IPFColor"]) or ""
    )
    roles["orientation_euler"] = (
        _find_first_hdf_dataset(
            handle,
            ordered_containers,
            ["CellData/EulerAngles", "Grain Data/EulerAngles", "Grain Data/FZAvgEuler"],
        )
        or ""
    )
    roles["orientation_quaternion"] = (
        _find_first_hdf_dataset(
            handle,
            ordered_containers,
            ["CellData/Quats", "CellData/FZQuats", "Grain Data/AvgQuats", "Grain Data/FZAvgQuats"],
        )
        or ""
    )
    roles["grain_volume"] = (
        _find_first_hdf_dataset(
            handle, ordered_containers, ["Grain Data/Volumes", "CellFeatureData/Volumes"]
        )
        or ""
    )
    roles["grain_neighbors"] = (
        _find_first_hdf_dataset(
            handle, ordered_containers, ["Grain Data/NumNeighbors", "CellFeatureData/NumNeighbors"]
        )
        or ""
    )
    roles["surface_flag"] = (
        _find_first_hdf_dataset(
            handle,
            ordered_containers,
            ["Grain Data/SurfaceFeatures", "CellFeatureData/SurfaceFeatures"],
        )
        or ""
    )
    neighbor_list_path = _find_first_hdf_dataset(
        handle, ordered_containers, ["Grain Data/NeighborList", "CellFeatureData/NeighborList"]
    )
    shared_surface_path = _find_first_hdf_dataset(
        handle,
        ordered_containers,
        ["Grain Data/SharedSurfaceAreaList", "CellFeatureData/SharedSurfaceAreaList"],
    )
    if neighbor_list_path and shared_surface_path:
        roles["grain_topology"] = neighbor_list_path
    ensemble_path = _find_first_hdf_dataset(
        handle,
        ordered_containers,
        [
            "CellEnsembleData/PhaseName",
            "CellEnsembleData/CrystalStructures",
            "CellEnsembleData/NumFeatures",
        ],
    )
    if ensemble_path:
        roles["ensemble_metadata"] = ensemble_path
    stats_root_path = "/DataContainers/StatsGeneratorDataContainer/CellEnsembleData/Statistics"
    stats_root = handle.get(stats_root_path)
    if stats_root is not None:
        phase_keys = _safe_hdf_group_keys(stats_root)
        for phase_key in phase_keys:
            base = f"{stats_root_path}/{phase_key}"
            if "target_phase_fraction" not in roles:
                path = f"{base}/PhaseFraction"
                if _safe_hdf_dataset(handle, path) is not None:
                    roles["target_phase_fraction"] = path
            if "feature_size_distribution" not in roles:
                for path in (
                    f"{base}/FeatureSize Distribution/Average",
                    f"{base}/FeatureSize Distribution",
                ):
                    if _safe_hdf_dataset(handle, path) is not None:
                        roles["feature_size_distribution"] = path
                        break
            if "feature_size_vs_neighbors" not in roles:
                path = f"{base}/FeatureSize Vs Neighbors Distributions/Average"
                if _safe_hdf_dataset(handle, path) is not None:
                    roles["feature_size_vs_neighbors"] = path
            if "misorientation_bins" not in roles:
                path = f"{base}/MisorientationBins"
                if _safe_hdf_dataset(handle, path) is not None:
                    roles["misorientation_bins"] = path
            if "odf" not in roles:
                path = f"{base}/ODF"
                if _safe_hdf_dataset(handle, path) is not None:
                    roles["odf"] = path

    roles = {key: value for key, value in roles.items() if value}
    capabilities: list[str] = []
    if roles.get("grain_id_map") or roles.get("phase_id_map"):
        capabilities.append("maps")
    if roles.get("orientation_euler") or roles.get("orientation_quaternion"):
        capabilities.append("orientation")
    if roles.get("ipf_map"):
        capabilities.append("ipf")
    if roles.get("grain_volume") or roles.get("grain_neighbors") or roles.get("surface_flag"):
        capabilities.append("grain_metrics")
    if roles.get("grain_topology") and shared_surface_path:
        capabilities.append("topology")
    if any(
        roles.get(key)
        for key in (
            "target_phase_fraction",
            "feature_size_distribution",
            "feature_size_vs_neighbors",
            "misorientation_bins",
            "odf",
        )
    ):
        capabilities.append("synthetic_stats")
    capabilities = [
        capability for capability in _HDF5_MATERIALS_CAPABILITY_ORDER if capability in capabilities
    ]

    phase_name_paths = [
        f"{container}/CellEnsembleData/PhaseName" for container in ordered_containers
    ]
    phase_name_paths.append(
        "/DataContainers/StatsGeneratorDataContainer/CellEnsembleData/PhaseName"
    )
    phase_names = _read_hdf_string_dataset(handle, phase_name_paths)

    num_feature_paths = [
        f"{container}/CellEnsembleData/NumFeatures" for container in ordered_containers
    ]
    feature_count = _read_hdf_feature_count(handle, num_feature_paths)
    grain_count = _read_hdf_row_count(
        handle,
        [
            path
            for key, path in roles.items()
            if key in {"grain_volume", "grain_neighbors", "surface_flag"}
        ],
    )
    recommended_map_dataset_path = (
        roles.get("ipf_map") or roles.get("grain_id_map") or roles.get("phase_id_map")
    )

    return {
        "detected": True,
        "schema": "dream3d",
        "capabilities": capabilities,
        "roles": roles,
        "phase_names": phase_names,
        "feature_count": feature_count,
        "grain_count": grain_count or feature_count,
        "recommended_view": "materials",
        "recommended_map_dataset_path": recommended_map_dataset_path,
    }


def _classify_hdf_preview_kind(dataset_path: str, shape: list[int], dtype_name: str) -> str | None:
    lower_path = str(dataset_path or "").lower()
    try:
        dtype = np.dtype(dtype_name)
    except Exception:
        dtype = np.dtype("float32")
    rank = len(shape)
    trailing = int(shape[-1]) if shape else 0
    integer_like = np.issubdtype(dtype, np.integer)
    uint8_like = dtype == np.dtype("uint8")

    if rank == 0:
        return "scalar"
    if rank == 1:
        return "series"
    if rank == 2:
        return "table"
    if rank == 3:
        if integer_like and any(
            token in lower_path for token in ("featureid", "feature_ids", "phase")
        ):
            return "label_volume"
        return "scalar_volume"
    if rank == 4:
        if trailing == 1:
            if integer_like and any(
                token in lower_path for token in ("featureid", "feature_ids", "phase")
            ):
                return "label_volume"
            return "scalar_volume"
        if trailing == 3 and (uint8_like or "ipfcolor" in lower_path or "color" in lower_path):
            return "rgb_volume"
        if trailing in {3, 4}:
            return "vector_volume"
        return "array"
    return "array"


def _extract_hdf_geometry(handle: Any) -> dict[str, Any] | None:
    geometry: dict[str, Any] | None = None

    def _visitor(name: str, obj: Any) -> Any:
        nonlocal geometry
        if geometry is not None:
            return True
        if not name.endswith("_SIMPL_GEOMETRY"):
            return None
        try:
            dimensions = obj.get("DIMENSIONS")
            spacing = obj.get("SPACING")
            origin = obj.get("ORIGIN")
        except Exception:
            return None
        if dimensions is None and spacing is None and origin is None:
            return None
        geometry = {
            "path": f"/{name}",
            "dimensions": _json_safe_hdf_value(dimensions[()]) if dimensions is not None else None,
            "spacing": _json_safe_hdf_value(spacing[()]) if spacing is not None else None,
            "origin": _json_safe_hdf_value(origin[()]) if origin is not None else None,
        }
        return True

    try:
        handle.visititems(_visitor)
    except Exception:
        return geometry
    return geometry


def _read_hdf_geometry_group(group: Any) -> dict[str, Any] | None:
    try:
        dimensions = group.get("DIMENSIONS")
        spacing = group.get("SPACING")
        origin = group.get("ORIGIN")
    except Exception:
        return None
    if dimensions is None and spacing is None and origin is None:
        return None
    return {
        "path": str(getattr(group, "name", "") or ""),
        "dimensions": _json_safe_hdf_value(dimensions[()]) if dimensions is not None else None,
        "spacing": _json_safe_hdf_value(spacing[()]) if spacing is not None else None,
        "origin": _json_safe_hdf_value(origin[()]) if origin is not None else None,
    }


def _normalize_hdf_dataset_path(dataset_path: str | None) -> str:
    parts = [part for part in str(dataset_path or "").strip().split("/") if part]
    if not parts:
        raise ValueError("dataset_path is required.")
    return "/" + "/".join(parts)


def _estimate_hdf_dataset_nbytes(shape: list[int], dtype_name: str) -> int | None:
    if not shape:
        try:
            return int(np.dtype(dtype_name).itemsize)
        except Exception:
            return None
    try:
        return int(math.prod(shape) * np.dtype(dtype_name).itemsize)
    except Exception:
        return None


def _sample_hdf_array(dataset: Any, *, target: int = _HDF5_SAMPLE_TARGET) -> np.ndarray:
    shape = _hdf_shape_list(getattr(dataset, "shape", ()))
    if not shape:
        return np.asarray(dataset[()])
    if int(math.prod(shape)) <= int(target):
        return np.asarray(dataset[()])
    rank = max(1, len(shape))
    target_side = max(1, int(round(float(target) ** (1.0 / float(rank)))))
    selection = tuple(
        slice(0, int(dim), max(1, math.ceil(int(dim) / target_side))) for dim in shape
    )
    sampled = np.asarray(dataset[selection])
    if int(sampled.size) <= int(target):
        return sampled
    return sampled.reshape(-1)[: int(target)]


def _summarize_hdf_attributes(
    attrs: Any, *, limit: int = _HDF5_DATASET_ATTR_LIMIT
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    try:
        items = list(attrs.items())
    except Exception:
        return output
    for index, (key, value) in enumerate(items):
        if index >= limit:
            output["..."] = f"{len(items) - limit} more attributes"
            break
        output[str(key)] = _json_safe_hdf_value(value)
    return output


def _hdf_preview_capabilities(preview_kind: str | None) -> list[str]:
    capability_map = {
        "scalar": ["value"],
        "series": ["plot"],
        "table": ["table", "plot"],
        "scalar_volume": ["slice", "histogram"],
        "label_volume": ["slice", "histogram"],
        "rgb_volume": ["slice"],
        "vector_volume": ["slice", "histogram"],
    }
    return list(capability_map.get(str(preview_kind or ""), []))


def _hdf_volume_eligibility(preview_kind: str | None) -> tuple[bool, str | None]:
    safe_kind = str(preview_kind or "").strip().lower()
    if safe_kind in {"scalar_volume", "label_volume"}:
        return True, None
    if safe_kind == "rgb_volume":
        return (
            False,
            "Display-oriented RGB datasets stay slice-only until a dedicated HDF5 volume display policy is added.",
        )
    if safe_kind == "vector_volume":
        return (
            False,
            "Component-aware vector and quaternion datasets stay slice-only in native HDF5 3D.",
        )
    if safe_kind:
        return False, "Only scalar or categorical volume datasets are eligible for native HDF5 3D."
    return False, "Dataset preview kind could not be classified for native HDF5 3D."


def _extract_hdf_dataset_geometry(handle: Any, dataset_path: str) -> dict[str, Any] | None:
    parts = [part for part in str(dataset_path or "").strip("/").split("/") if part]
    for index in range(len(parts) - 1, 0, -1):
        candidate = "/" + "/".join(parts[:index] + ["_SIMPL_GEOMETRY"])
        try:
            group = handle.get(candidate)
        except Exception:
            group = None
        if group is None:
            continue
        geometry = _read_hdf_geometry_group(group)
        if geometry is not None:
            return geometry
    return _extract_hdf_geometry(handle)


def _hdf_component_labels(
    dataset_path: str,
    count: int,
    preview_kind: str | None,
    *,
    dataset_name: str | None = None,
    field_names: list[str] | None = None,
) -> list[str]:
    safe_count = max(1, int(count or 1))
    if field_names:
        labels = [str(name) for name in field_names[:safe_count]]
        if len(labels) == safe_count:
            return labels
    lower_path = str(dataset_path or "").lower()
    lower_name = str(dataset_name or "").lower()
    tokens = f"{lower_path} {lower_name}"
    if safe_count == 3 and "euler" in tokens:
        return ["phi1", "Phi", "phi2"]
    if safe_count == 4 and "quat" in tokens:
        return ["q0", "q1", "q2", "q3"]
    if safe_count >= 3 and ("ipfcolor" in tokens or "color" in tokens):
        labels = ["R", "G", "B"]
        while len(labels) < safe_count:
            labels.append(f"channel_{len(labels) + 1}")
        return labels[:safe_count]
    if preview_kind == "table" and safe_count == 1:
        return [str(dataset_name or "value")]
    if preview_kind == "series":
        return [str(dataset_name or "value")]
    return [f"component_{index + 1}" for index in range(safe_count)]


def _hdf_component_count(
    shape: list[int],
    preview_kind: str | None,
    *,
    field_names: list[str] | None = None,
) -> int:
    if field_names:
        return max(1, len(field_names))
    if preview_kind in {"vector_volume", "rgb_volume"} and shape:
        return max(1, int(shape[-1]))
    if preview_kind == "table" and len(shape) >= 2:
        return max(1, int(shape[1]))
    return 1


def _hdf_axis_sizes_for_preview(
    shape: list[int], preview_kind: str | None
) -> dict[str, int] | None:
    if preview_kind not in {"scalar_volume", "label_volume", "vector_volume", "rgb_volume"}:
        return None
    if len(shape) >= 4:
        volume_shape = shape[:-1]
    else:
        volume_shape = shape
    if len(volume_shape) < 3:
        return None
    return {
        "T": 1,
        "C": 1,
        "Z": max(1, int(volume_shape[0])),
        "Y": max(1, int(volume_shape[1])),
        "X": max(1, int(volume_shape[2])),
    }


def _hdf_spacing_from_geometry(geometry: dict[str, Any] | None) -> dict[str, float] | None:
    spacing = geometry.get("spacing") if isinstance(geometry, dict) else None
    if not isinstance(spacing, list) or len(spacing) < 3:
        return None
    try:
        return {
            "x": _finite_spacing(spacing[0], 1.0),
            "y": _finite_spacing(spacing[1], 1.0),
            "z": _finite_spacing(spacing[2], 1.0),
        }
    except Exception:
        return None


def _build_hdf_preview_planes(
    shape: list[int],
    preview_kind: str | None,
    geometry: dict[str, Any] | None,
) -> tuple[list[str], dict[str, Any]]:
    axis_sizes = _hdf_axis_sizes_for_preview(shape, preview_kind)
    if axis_sizes is None:
        return [], {}
    spacing = _hdf_spacing_from_geometry(geometry)
    axes = ["z", "y", "x"]
    return axes, {
        axis: _build_plane_descriptor(
            axis=axis,  # type: ignore[arg-type]
            axis_sizes=axis_sizes,
            physical_spacing=spacing,
        )
        for axis in axes
    }


def _clamp_hdf_index(index: int | None, size: int) -> int:
    if size <= 1:
        return 0
    if index is None:
        return max(0, size // 2)
    return max(0, min(int(index), size - 1))


def _render_hdf_label_plane(plane: np.ndarray) -> np.ndarray:
    array = np.asarray(plane)
    if array.ndim != 2:
        array = np.squeeze(array)
    if array.ndim != 2:
        return _render_preview_image(array)
    normalized = np.nan_to_num(array, nan=0.0).astype(np.int64, copy=False)
    unique_values = np.unique(normalized)
    palette = np.zeros((int(unique_values.size), 3), dtype=np.uint8)
    for index, value in enumerate(unique_values.tolist()):
        if int(value) == 0:
            palette[index] = np.asarray([26, 32, 44], dtype=np.uint8)
            continue
        hue = ((int(value) * 2654435761) % 360) / 360.0
        red, green, blue = colorsys.hls_to_rgb(hue, 0.58, 0.48)
        palette[index] = np.asarray(
            [int(round(red * 255)), int(round(green * 255)), int(round(blue * 255))],
            dtype=np.uint8,
        )
    lookup = {int(value): palette[index] for index, value in enumerate(unique_values.tolist())}
    output = np.zeros((*normalized.shape, 3), dtype=np.uint8)
    for value, color in lookup.items():
        output[normalized == value] = color
    return output


def _render_hdf_preview_plane(plane: np.ndarray, preview_kind: str | None) -> np.ndarray:
    if preview_kind == "label_volume":
        return _render_hdf_label_plane(plane)
    return _render_preview_image(np.asarray(plane))


def _sample_hdf_numeric_values(
    dataset: Any,
    *,
    preview_kind: str | None,
    component: int | None = None,
    target: int = _HDF5_SAMPLE_TARGET,
) -> np.ndarray:
    sampled = _sample_hdf_array(dataset, target=target)
    array = np.asarray(sampled)
    if preview_kind in {"vector_volume", "rgb_volume"} and array.ndim >= 1:
        component_index = max(0, min(int(component or 0), max(0, int(array.shape[-1]) - 1)))
        array = array[..., component_index]
    elif (
        preview_kind in {"scalar_volume", "label_volume"}
        and array.ndim >= 4
        and int(array.shape[-1]) == 1
    ):
        array = array[..., 0]
    return np.asarray(array).reshape(-1)


def _build_histogram_records(
    values: np.ndarray,
    *,
    bins: int = DEFAULT_HISTOGRAM_BINS,
) -> tuple[list[dict[str, Any]], bool, float | None, float | None, int]:
    numeric = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return [], False, None, None, 0
    if np.allclose(finite, np.round(finite), equal_nan=False):
        unique, counts = np.unique(finite.astype(np.int64), return_counts=True)
        if unique.size <= min(64, max(8, int(bins or DEFAULT_HISTOGRAM_BINS))):
            return (
                [
                    {
                        "label": str(int(value)),
                        "start": float(value),
                        "end": float(value),
                        "count": int(count),
                    }
                    for value, count in zip(unique.tolist(), counts.tolist(), strict=False)
                ],
                True,
                float(np.min(finite)),
                float(np.max(finite)),
                int(finite.size),
            )
    hist, edges = np.histogram(finite, bins=max(8, int(bins or DEFAULT_HISTOGRAM_BINS)))
    records: list[dict[str, Any]] = []
    for start, end, count in zip(edges[:-1], edges[1:], hist.tolist(), strict=False):
        records.append(
            {
                "label": f"{float(start):.3g} to {float(end):.3g}",
                "start": float(start),
                "end": float(end),
                "count": int(count),
            }
        )
    return records, False, float(np.min(finite)), float(np.max(finite)), int(finite.size)


def render_hdf5_dataset_slice(
    *,
    file_path: str | Path,
    dataset_path: str,
    axis: str = "z",
    index: int | None = None,
    component: int | None = None,
) -> np.ndarray:
    normalized_path = _normalize_hdf_dataset_path(dataset_path)
    h5py, h5py_error = _load_h5py_module()
    if h5py is None:
        raise ValueError(f"h5py is unavailable: {h5py_error}")

    source = Path(str(file_path)).expanduser()
    with h5py.File(source, "r") as handle:
        node = handle.get(normalized_path)
        if node is None:
            raise FileNotFoundError(f"Dataset not found: {normalized_path}")
        if not isinstance(node, h5py.Dataset):
            raise ValueError(f"Path does not point to a dataset: {normalized_path}")

        shape = _hdf_shape_list(node.shape)
        preview_kind = _classify_hdf_preview_kind(normalized_path, shape, str(node.dtype))
        if preview_kind not in {"scalar_volume", "label_volume", "vector_volume", "rgb_volume"}:
            raise ValueError("Slice preview is only available for volume-like datasets.")

        axis_sizes = _hdf_axis_sizes_for_preview(shape, preview_kind)
        if axis_sizes is None:
            raise ValueError("Slice preview is unavailable for this dataset shape.")

        normalized_axis = normalize_view_axis(axis)
        selection: list[Any] = [slice(None)] * len(shape)
        axis_index = {"z": 0, "y": 1, "x": 2}[normalized_axis]
        size = int([axis_sizes["Z"], axis_sizes["Y"], axis_sizes["X"]][axis_index])
        selection[axis_index] = _clamp_hdf_index(index, size)

        component_count = _hdf_component_count(shape, preview_kind)
        component_index = max(0, min(int(component or 0), component_count - 1))
        if preview_kind == "rgb_volume":
            selection[-1] = slice(0, min(3, component_count))
        elif preview_kind == "vector_volume":
            selection[-1] = component_index
        elif len(shape) >= 4 and int(shape[-1]) == 1:
            selection[-1] = 0

        plane = np.asarray(node[tuple(selection)])
        return _render_hdf_preview_plane(plane, preview_kind)


def build_hdf5_dataset_histogram(
    *,
    file_id: str,
    file_path: str | Path,
    dataset_path: str,
    component: int | None = None,
    bins: int = DEFAULT_HISTOGRAM_BINS,
) -> dict[str, Any]:
    normalized_path = _normalize_hdf_dataset_path(dataset_path)
    h5py, h5py_error = _load_h5py_module()
    if h5py is None:
        raise ValueError(f"h5py is unavailable: {h5py_error}")

    source = Path(str(file_path)).expanduser()
    with h5py.File(source, "r") as handle:
        node = handle.get(normalized_path)
        if node is None:
            raise FileNotFoundError(f"Dataset not found: {normalized_path}")
        if not isinstance(node, h5py.Dataset):
            raise ValueError(f"Path does not point to a dataset: {normalized_path}")

        shape = _hdf_shape_list(node.shape)
        preview_kind = _classify_hdf_preview_kind(normalized_path, shape, str(node.dtype))
        if preview_kind not in {
            "scalar_volume",
            "label_volume",
            "vector_volume",
            "rgb_volume",
            "table",
            "series",
        }:
            raise ValueError("Histogram preview is unavailable for this dataset.")
        field_names = list(node.dtype.names or [])
        component_count = _hdf_component_count(shape, preview_kind, field_names=field_names)
        component_index = max(0, min(int(component or 0), component_count - 1))
        component_labels = _hdf_component_labels(
            normalized_path,
            component_count,
            preview_kind,
            dataset_name=normalized_path.rsplit("/", 1)[-1],
            field_names=field_names,
        )
        sampled = _sample_hdf_numeric_values(
            node,
            preview_kind=preview_kind,
            component=component_index,
            target=max(_HDF5_SAMPLE_TARGET, 8192),
        )
        records, discrete, minimum, maximum, sample_count = _build_histogram_records(
            sampled, bins=bins
        )
        return {
            "file_id": file_id,
            "dataset_path": normalized_path,
            "preview_kind": preview_kind,
            "component_index": component_index if component_count > 1 else None,
            "component_label": component_labels[component_index] if component_count > 1 else None,
            "sample_count": sample_count,
            "discrete": discrete,
            "min": minimum,
            "max": maximum,
            "bins": records,
        }


def build_hdf5_dataset_table_preview(
    *,
    file_id: str,
    file_path: str | Path,
    dataset_path: str,
    offset: int = 0,
    limit: int = _HDF5_TABLE_PREVIEW_LIMIT,
) -> dict[str, Any]:
    normalized_path = _normalize_hdf_dataset_path(dataset_path)
    h5py, h5py_error = _load_h5py_module()
    if h5py is None:
        raise ValueError(f"h5py is unavailable: {h5py_error}")

    safe_offset = max(0, int(offset or 0))
    safe_limit = max(1, min(int(limit or _HDF5_TABLE_PREVIEW_LIMIT), 128))
    source = Path(str(file_path)).expanduser()
    with h5py.File(source, "r") as handle:
        node = handle.get(normalized_path)
        if node is None:
            raise FileNotFoundError(f"Dataset not found: {normalized_path}")
        if not isinstance(node, h5py.Dataset):
            raise ValueError(f"Path does not point to a dataset: {normalized_path}")

        shape = _hdf_shape_list(node.shape)
        preview_kind = _classify_hdf_preview_kind(normalized_path, shape, str(node.dtype))
        if preview_kind not in {"table", "series"}:
            raise ValueError("Table preview is only available for table-like datasets.")

        total_rows = int(shape[0]) if shape else 1
        total_columns = int(shape[1]) if len(shape) >= 2 else 1
        dataset_name = normalized_path.rsplit("/", 1)[-1]
        field_names = list(node.dtype.names or [])
        column_labels = _hdf_component_labels(
            normalized_path,
            total_columns,
            preview_kind,
            dataset_name=dataset_name,
            field_names=field_names,
        )
        columns: list[dict[str, Any]] = []

        rows: list[dict[str, Any]] = []
        numeric_column_keys: list[str] = []
        start = min(safe_offset, max(0, total_rows - 1)) if total_rows > 0 else 0
        stop = min(total_rows, start + safe_limit)

        if field_names:
            preview_rows = np.asarray(node[start:stop]).reshape(-1)
            for field_name in field_names:
                field_dtype = node.dtype.fields[field_name][0] if node.dtype.fields else None
                numeric = bool(field_dtype is not None and np.issubdtype(field_dtype, np.number))
                key = str(field_name)
                columns.append(
                    {
                        "key": key,
                        "label": key,
                        "dtype": str(field_dtype or "unknown"),
                        "numeric": numeric,
                    }
                )
                if numeric:
                    numeric_column_keys.append(key)
            for row_index, row in enumerate(preview_rows, start=start):
                rows.append(
                    {
                        "row_index": int(row_index),
                        **{
                            str(field_name): _json_safe_hdf_value(row[field_name])
                            for field_name in field_names[:_HDF5_TABLE_FIELD_LIMIT]
                        },
                    }
                )
        else:
            if preview_kind == "series":
                preview_rows = np.asarray(node[start:stop]).reshape(-1, 1)
            else:
                preview_rows = np.asarray(node[start:stop])
                if preview_rows.ndim == 1:
                    preview_rows = preview_rows.reshape(-1, 1)
            for index, label in enumerate(
                column_labels[: preview_rows.shape[1] if preview_rows.ndim > 1 else 1]
            ):
                numeric = bool(np.issubdtype(node.dtype, np.number))
                key = str(label)
                columns.append(
                    {"key": key, "label": key, "dtype": str(node.dtype), "numeric": numeric}
                )
                if numeric:
                    numeric_column_keys.append(key)
            for row_index, row in enumerate(np.asarray(preview_rows), start=start):
                row_array = np.asarray(row).reshape(-1)
                rows.append(
                    {
                        "row_index": int(row_index),
                        **{
                            columns[column_index]["key"]: _json_safe_hdf_value(value)
                            for column_index, value in enumerate(row_array[: len(columns)])
                        },
                    }
                )

        charts: list[dict[str, Any]] = []
        if numeric_column_keys:
            chart_step = max(1, math.ceil(max(1, total_rows) / float(_HDF5_TABLE_CHART_TARGET)))
            sampled = np.asarray(node[::chart_step])
            if preview_kind == "series":
                sampled = np.asarray(sampled).reshape(-1, 1)
            elif sampled.ndim == 1:
                sampled = sampled.reshape(-1, 1)
            if sampled.size > 0:
                sampled_rows: list[dict[str, Any]] = []
                for sample_index, row in enumerate(sampled[:_HDF5_TABLE_CHART_TARGET]):
                    row_array = np.asarray(row).reshape(-1)
                    sampled_rows.append(
                        {
                            "row_index": int(sample_index * chart_step),
                            **{
                                columns[column_index]["key"]: float(row_array[column_index])
                                for column_index in range(min(len(columns), row_array.size))
                                if columns[column_index]["numeric"]
                            },
                        }
                    )
                primary_key = numeric_column_keys[0]
                if len(numeric_column_keys) >= 2:
                    charts.append(
                        {
                            "kind": "scatter",
                            "title": f"{numeric_column_keys[0]} vs {numeric_column_keys[1]}",
                            "description": "Sampled across the dataset to preserve overview structure.",
                            "x_key": numeric_column_keys[0],
                            "y_key": numeric_column_keys[1],
                            "data": sampled_rows,
                        }
                    )
                else:
                    charts.append(
                        {
                            "kind": "scatter",
                            "title": f"{primary_key} by row",
                            "description": "Row index versus value for a sampled subset of the table.",
                            "x_key": "row_index",
                            "y_key": primary_key,
                            "data": sampled_rows,
                        }
                    )
                primary_values = np.asarray(
                    [row[primary_key] for row in sampled_rows if primary_key in row],
                    dtype=np.float64,
                )
                hist_records, _discrete, _min, _max, _sample_count = _build_histogram_records(
                    primary_values, bins=32
                )
                if hist_records:
                    charts.append(
                        {
                            "kind": "histogram",
                            "title": f"{primary_key} distribution",
                            "description": "Distribution derived from the sampled numeric values.",
                            "x_key": "label",
                            "y_key": "count",
                            "data": hist_records,
                        }
                    )

        return {
            "file_id": file_id,
            "dataset_path": normalized_path,
            "preview_kind": preview_kind,
            "offset": safe_offset,
            "limit": safe_limit,
            "total_rows": total_rows,
            "total_columns": total_columns,
            "columns": columns,
            "rows": rows,
            "charts": charts,
        }


def build_hdf5_dataset_summary(
    *,
    file_id: str,
    file_path: str | Path,
    dataset_path: str,
) -> dict[str, Any]:
    normalized_path = _normalize_hdf_dataset_path(dataset_path)
    h5py, h5py_error = _load_h5py_module()
    if h5py is None:
        raise ValueError(f"h5py is unavailable: {h5py_error}")

    source = Path(str(file_path)).expanduser()
    with h5py.File(source, "r") as handle:
        node = handle.get(normalized_path)
        if node is None:
            raise FileNotFoundError(f"Dataset not found: {normalized_path}")
        if not isinstance(node, h5py.Dataset):
            raise ValueError(f"Path does not point to a dataset: {normalized_path}")

        shape = _hdf_shape_list(node.shape)
        dtype_name = str(node.dtype)
        preview_kind = _classify_hdf_preview_kind(normalized_path, shape, dtype_name)
        element_count = int(math.prod(shape)) if shape else 1
        estimated_bytes = _estimate_hdf_dataset_nbytes(shape, dtype_name)
        attributes = _summarize_hdf_attributes(getattr(node, "attrs", {}))
        capabilities = _hdf_preview_capabilities(preview_kind)
        geometry = _extract_hdf_dataset_geometry(handle, normalized_path)
        axis_sizes = _hdf_axis_sizes_for_preview(shape, preview_kind)
        physical_spacing = _hdf_spacing_from_geometry(geometry)
        volume_eligible, volume_reason = _hdf_volume_eligibility(preview_kind)
        if volume_eligible and "volume" not in capabilities:
            capabilities.append("volume")
        render_policy = _hdf_render_policy(preview_kind)
        measurement_policy = _measurement_policy(
            orientation_frame="voxel", physical_spacing=physical_spacing
        )
        diagnostic_surface = _diagnostic_surface(
            render_policy=render_policy,
            modality="materials" if _materials_semantic_role(normalized_path) else "unknown",
            is_volume=False,
        )
        display_capabilities = _hdf_display_capabilities(
            preview_kind=preview_kind,
            render_policy=render_policy,
            volume_eligible=bool(volume_eligible),
        )
        delivery_mode = _hdf_delivery_mode(
            render_policy=render_policy,
            volume_eligible=bool(volume_eligible),
        )
        first_paint_mode = _first_paint_mode(
            default_surface="2d",
            delivery_mode=delivery_mode,
        )
        texture_policy = _texture_policy(render_policy)
        viewer_capabilities = _viewer_capabilities(
            delivery_mode=delivery_mode,
            first_paint_mode=first_paint_mode,
            diagnostic_surface=diagnostic_surface,
            texture_policy=texture_policy,
            display_capabilities=display_capabilities,
        )
        field_names = list(node.dtype.names or [])
        component_count = _hdf_component_count(shape, preview_kind, field_names=field_names)
        component_labels = _hdf_component_labels(
            normalized_path,
            component_count,
            preview_kind,
            dataset_name=normalized_path.rsplit("/", 1)[-1],
            field_names=field_names,
        )
        slice_axes, preview_planes = _build_hdf_preview_planes(shape, preview_kind, geometry)
        semantic_role = _materials_semantic_role(normalized_path)
        units_hint = _materials_units_hint(semantic_role)
        materials_domain_tags = _materials_domain_tags(normalized_path, semantic_role)

        structured_fields: list[dict[str, Any]] = []
        for index, field_name in enumerate(field_names):
            if index >= _HDF5_TABLE_FIELD_LIMIT:
                structured_fields.append(
                    {
                        "name": "...",
                        "dtype": f"{len(field_names) - _HDF5_TABLE_FIELD_LIMIT} more fields",
                    }
                )
                break
            field_dtype = (
                node.dtype.fields[field_name][0]
                if node.dtype.fields and field_name in node.dtype.fields
                else None
            )
            structured_fields.append(
                {"name": str(field_name), "dtype": str(field_dtype or "unknown")}
            )

        sample_statistics: dict[str, Any] | None = None
        sample_values: Any = None
        sample_shape: list[int] = []
        try:
            sampled = _sample_hdf_array(node)
            sample_shape = _hdf_shape_list(getattr(sampled, "shape", ()))
            if node.dtype.names:
                preview_rows = np.asarray(sampled).reshape(-1)[:_HDF5_SAMPLE_VALUE_LIMIT]
                sample_values = [
                    {
                        str(field_name): _json_safe_hdf_value(row[field_name])
                        for field_name in field_names[:_HDF5_TABLE_FIELD_LIMIT]
                    }
                    for row in preview_rows
                ]
            elif sampled.ndim == 0:
                sample_values = _json_safe_hdf_value(sampled.item())
            else:
                preview_sample = sampled.reshape(-1)[:_HDF5_SAMPLE_VALUE_LIMIT]
                sample_values = _json_safe_hdf_value(preview_sample)

            if np.issubdtype(node.dtype, np.number) and not node.dtype.names:
                numeric = np.asarray(sampled, dtype=np.float64).reshape(-1)
                finite = numeric[np.isfinite(numeric)]
                if finite.size > 0:
                    sample_statistics = {
                        "sample_count": int(finite.size),
                        "min": float(np.min(finite)),
                        "max": float(np.max(finite)),
                        "mean": float(np.mean(finite)),
                    }
                    if np.issubdtype(node.dtype, np.integer):
                        sample_statistics["unique_values"] = int(np.unique(finite).size)
        except Exception:
            sample_values = None
            sample_statistics = None
            sample_shape = []

        dimension_summary: dict[str, int] | None = None
        if preview_kind == "table" and len(shape) >= 2:
            dimension_summary = {"rows": int(shape[0]), "columns": int(shape[1])}
        elif preview_kind in {"vector_volume", "rgb_volume"} and shape:
            dimension_summary = {"components": int(shape[-1])}
        elif preview_kind in {"scalar_volume", "label_volume"} and shape:
            volume_shape = shape[:-1] if len(shape) >= 4 and int(shape[-1]) == 1 else shape
            if len(volume_shape) >= 3:
                dimension_summary = {
                    "z": int(volume_shape[0]),
                    "y": int(volume_shape[1]),
                    "x": int(volume_shape[2]),
                }

        return {
            "file_id": file_id,
            "dataset_path": normalized_path,
            "dataset_name": normalized_path.rsplit("/", 1)[-1],
            "preview_kind": preview_kind,
            "semantic_role": semantic_role,
            "units_hint": units_hint,
            "materials_domain_tags": materials_domain_tags,
            "dtype": dtype_name,
            "shape": shape,
            "rank": len(shape),
            "element_count": element_count,
            "estimated_bytes": estimated_bytes,
            "dimension_summary": dimension_summary,
            "capabilities": capabilities,
            "render_policy": render_policy,
            "delivery_mode": delivery_mode,
            "diagnostic_surface": diagnostic_surface,
            "first_paint_mode": first_paint_mode,
            "measurement_policy": measurement_policy,
            "texture_policy": texture_policy,
            "display_capabilities": display_capabilities,
            "viewer_capabilities": viewer_capabilities,
            "volume_eligible": bool(volume_eligible),
            "volume_reason": volume_reason,
            "axis_sizes": axis_sizes,
            "physical_spacing": physical_spacing,
            "attributes": attributes,
            "geometry": geometry,
            "structured_fields": structured_fields,
            "component_count": component_count,
            "component_labels": component_labels,
            "slice_axes": slice_axes,
            "preview_planes": preview_planes,
            "sample_shape": sample_shape,
            "sample_values": sample_values,
            "sample_statistics": sample_statistics,
        }


def _build_materials_dataset_links(materials_payload: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    roles = materials_payload.get("roles") if isinstance(materials_payload, dict) else {}
    if not isinstance(roles, dict):
        return output
    for semantic_role, dataset_path in roles.items():
        path = str(dataset_path or "").strip()
        if not path:
            continue
        label = str(semantic_role).replace("_", " ").title()
        output.append(
            {
                "label": label,
                "dataset_path": path,
                "semantic_role": str(semantic_role),
                "group": _HDF5_MATERIALS_ROLE_GROUPS.get(str(semantic_role), "materials"),
            }
        )
    return output


def _build_materials_map_cards(
    handle: Any, materials_payload: dict[str, Any]
) -> list[dict[str, Any]]:
    role_order = [
        ("ipf_map", "IPF map", "Inverse-pole-figure color map from the materials dataset."),
        ("grain_id_map", "Feature IDs", "Categorical grain identifier map."),
        ("phase_id_map", "Phase map", "Categorical phase-id map."),
    ]
    roles = materials_payload.get("roles") if isinstance(materials_payload, dict) else {}
    maps: list[dict[str, Any]] = []
    if not isinstance(roles, dict):
        return maps
    for semantic_role, title, description in role_order:
        dataset_path = roles.get(semantic_role)
        dataset = _safe_hdf_dataset(handle, dataset_path)
        if dataset is None:
            continue
        preview_kind = _classify_hdf_preview_kind(
            str(dataset_path), _hdf_shape_list(dataset.shape), str(dataset.dtype)
        )
        maps.append(
            {
                "title": title,
                "description": description,
                "dataset_path": str(dataset_path),
                "semantic_role": semantic_role,
                "preview_kind": preview_kind,
            }
        )
    return maps


def _materials_phase_label(phase_id: int, phase_names: list[str]) -> str:
    if 0 <= int(phase_id) < len(phase_names):
        return str(phase_names[int(phase_id)])
    return f"Phase {int(phase_id)}"


def _build_materials_phase_fraction_chart(
    handle: Any, dataset_path: str, phase_names: list[str]
) -> dict[str, Any] | None:
    dataset = _safe_hdf_dataset(handle, dataset_path)
    if dataset is None:
        return None
    values = _sample_hdf_numeric_values(
        dataset, preview_kind="label_volume", target=max(_HDF5_SAMPLE_TARGET, 8192)
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    labels, counts = np.unique(values.astype(np.int64), return_counts=True)
    total = max(1, int(np.sum(counts)))
    data = [
        {
            "phase": _materials_phase_label(int(label), phase_names),
            "percent": float((count / total) * 100.0),
            "phase_id": int(label),
        }
        for label, count in zip(labels.tolist(), counts.tolist(), strict=False)
    ]
    return _build_materials_bar_chart(
        title="Phase fraction",
        data=data,
        source_paths=[dataset_path],
        description="Sampled phase-id distribution from the voxel map.",
        x_key="phase",
        y_key="percent",
        units_hint="percent",
        provenance="Bounded voxel sample from the phase map.",
    )


def _build_materials_target_phase_fraction_chart(
    handle: Any, phase_names: list[str]
) -> dict[str, Any] | None:
    stats_root = handle.get(
        "/DataContainers/StatsGeneratorDataContainer/CellEnsembleData/Statistics"
    )
    if stats_root is None:
        return None
    data: list[dict[str, Any]] = []
    source_paths: list[str] = []
    for phase_key in _safe_hdf_group_keys(stats_root):
        dataset_path = f"/DataContainers/StatsGeneratorDataContainer/CellEnsembleData/Statistics/{phase_key}/PhaseFraction"
        dataset = _safe_hdf_dataset(handle, dataset_path)
        if dataset is None:
            continue
        try:
            value = float(np.asarray(dataset[()]).reshape(-1)[0])
        except Exception:
            continue
        source_paths.append(dataset_path)
        data.append(
            {
                "phase": _materials_phase_label(int(phase_key), phase_names),
                "percent": float(value * 100.0),
                "phase_id": int(phase_key),
            }
        )
    if not data:
        return None
    return _build_materials_bar_chart(
        title="Target phase fraction",
        data=data,
        source_paths=source_paths,
        description="Per-phase target fraction declared in the DREAM3D statistics container.",
        x_key="phase",
        y_key="percent",
        units_hint="percent",
        provenance="Directly read from the DREAM3D statistics container.",
    )


def _build_materials_grain_charts(
    handle: Any, materials_payload: dict[str, Any]
) -> list[dict[str, Any]]:
    roles = materials_payload.get("roles") if isinstance(materials_payload, dict) else {}
    if not isinstance(roles, dict):
        return []
    charts: list[dict[str, Any]] = []
    phase_names = (
        materials_payload.get("phase_names") if isinstance(materials_payload, dict) else []
    )
    phase_names = [str(item) for item in phase_names] if isinstance(phase_names, list) else []
    phase_map_path = roles.get("phase_id_map")
    volume_path = roles.get("grain_volume")
    neighbors_path = roles.get("grain_neighbors")
    surface_path = roles.get("surface_flag")
    volume_ds = _safe_hdf_dataset(handle, volume_path)
    neighbors_ds = _safe_hdf_dataset(handle, neighbors_path)
    surface_ds = _safe_hdf_dataset(handle, surface_path)

    if phase_map_path:
        phase_chart = _build_materials_phase_fraction_chart(
            handle, str(phase_map_path), phase_names
        )
        if phase_chart:
            charts.append(phase_chart)

    if volume_ds is not None:
        volume_values = _sample_hdf_numeric_column(volume_ds, target_rows=4096, drop_first=True)
        chart = _build_materials_histogram_chart(
            title="Grain volume distribution",
            values=volume_values,
            source_paths=[str(volume_path)],
            description="Sampled grain-level volume values.",
            bins=24,
        )
        if chart:
            charts.append(chart)

    if neighbors_ds is not None:
        neighbor_values = _sample_hdf_numeric_column(
            neighbors_ds, target_rows=4096, drop_first=True
        )
        chart = _build_materials_histogram_chart(
            title="Neighbor count distribution",
            values=neighbor_values,
            source_paths=[str(neighbors_path)],
            description="Sampled grain-level neighbor counts.",
            discrete=True,
            bins=24,
        )
        if chart:
            charts.append(chart)

    if volume_ds is not None and neighbors_ds is not None:
        row_count = min(_hdf_shape_list(volume_ds.shape)[0], _hdf_shape_list(neighbors_ds.shape)[0])
        indices = _sample_hdf_row_indices(row_count, target_rows=256, drop_first=True)
        if indices.size > 0:
            volumes = np.asarray(volume_ds[indices]).reshape(-1)
            neighbors = np.asarray(neighbors_ds[indices]).reshape(-1)
            chart = _build_materials_scatter_chart(
                title="Volume vs neighbors",
                x_values=np.asarray(volumes, dtype=np.float64),
                y_values=np.asarray(neighbors, dtype=np.float64),
                source_paths=[str(volume_path), str(neighbors_path)],
                description="Aligned sampled grain rows.",
                x_key="volume",
                y_key="neighbors",
            )
            if chart:
                charts.append(chart)

    if surface_ds is not None:
        values = _sample_hdf_numeric_column(surface_ds, target_rows=4096, drop_first=True)
        labels, counts = np.unique(values.astype(np.int64), return_counts=True)
        data = [
            {"class": "Surface" if int(label) > 0 else "Interior", "count": int(count)}
            for label, count in zip(labels.tolist(), counts.tolist(), strict=False)
        ]
        chart = _build_materials_bar_chart(
            title="Surface vs interior grains",
            data=data,
            source_paths=[str(surface_path)],
            description="Sampled grain classification from the surface feature flag.",
            x_key="class",
            y_key="count",
        )
        if chart:
            charts.append(chart)

    return charts


def _build_materials_orientation_charts(
    handle: Any, materials_payload: dict[str, Any]
) -> list[dict[str, Any]]:
    roles = materials_payload.get("roles") if isinstance(materials_payload, dict) else {}
    if not isinstance(roles, dict):
        return []
    charts: list[dict[str, Any]] = []
    euler_path = None
    quat_path = None
    for candidate in (
        "/DataContainers/SyntheticVolumeDataContainer/Grain Data/EulerAngles",
        roles.get("orientation_euler"),
    ):
        if _safe_hdf_dataset(handle, candidate) is not None:
            euler_path = str(candidate)
            break
    for candidate in (
        "/DataContainers/SyntheticVolumeDataContainer/Grain Data/AvgQuats",
        "/DataContainers/SyntheticVolumeDataContainer/Grain Data/FZAvgQuats",
        roles.get("orientation_quaternion"),
    ):
        if _safe_hdf_dataset(handle, candidate) is not None:
            quat_path = str(candidate)
            break

    if euler_path:
        euler_ds = _safe_hdf_dataset(handle, euler_path)
        if euler_ds is not None:
            labels = ["phi1", "Phi", "phi2"]
            for component, label in enumerate(labels):
                values = _sample_hdf_numeric_column(
                    euler_ds, component=component, target_rows=4096, drop_first=True
                )
                chart = _build_materials_histogram_chart(
                    title=f"{label} distribution",
                    values=values,
                    source_paths=[euler_path],
                    description="Sampled Euler-angle distribution from the materials dataset.",
                    units_hint="radians",
                    bins=24,
                )
                if chart:
                    charts.append(chart)

    if quat_path:
        quat_ds = _safe_hdf_dataset(handle, quat_path)
        if quat_ds is not None:
            sampled = _sample_hdf_rows(quat_ds, target_rows=4096, drop_first=True)
            array = np.asarray(sampled)
            if array.ndim >= 2 and array.shape[-1] >= 4:
                norms = np.linalg.norm(array[..., :4].reshape(-1, 4), axis=1)
                chart = _build_materials_histogram_chart(
                    title="Quaternion norm QC",
                    values=np.asarray(norms, dtype=np.float64),
                    source_paths=[quat_path],
                    description="Norm check for sampled quaternion rows.",
                    units_hint="unit quaternion",
                    bins=24,
                )
                if chart:
                    charts.append(chart)

    return charts


def _build_materials_synthetic_stats(
    handle: Any, materials_payload: dict[str, Any]
) -> list[dict[str, Any]]:
    phase_names = (
        materials_payload.get("phase_names") if isinstance(materials_payload, dict) else []
    )
    phase_names = phase_names if isinstance(phase_names, list) else []
    charts: list[dict[str, Any]] = []
    roles = materials_payload.get("roles") if isinstance(materials_payload, dict) else {}
    if not isinstance(roles, dict):
        return charts

    phase_map_path = roles.get("phase_id_map")
    if phase_map_path:
        realized = _build_materials_phase_fraction_chart(
            handle, str(phase_map_path), [str(x) for x in phase_names]
        )
        if realized:
            realized["title"] = "Realized phase fraction"
            charts.append(realized)

    target_phase_chart = _build_materials_target_phase_fraction_chart(
        handle, [str(x) for x in phase_names]
    )
    if target_phase_chart:
        charts.append(target_phase_chart)

    stats_root = handle.get(
        "/DataContainers/StatsGeneratorDataContainer/CellEnsembleData/Statistics"
    )
    if stats_root is None:
        return charts

    for phase_key in _safe_hdf_group_keys(stats_root)[:4]:
        phase_label = _materials_phase_label(int(phase_key), [str(x) for x in phase_names])
        bin_path = f"/DataContainers/StatsGeneratorDataContainer/CellEnsembleData/Statistics/{phase_key}/BinNumber"
        average_path = f"/DataContainers/StatsGeneratorDataContainer/CellEnsembleData/Statistics/{phase_key}/FeatureSize Vs Neighbors Distributions/Average"
        misorientation_path = f"/DataContainers/StatsGeneratorDataContainer/CellEnsembleData/Statistics/{phase_key}/MisorientationBins"
        odf_path = f"/DataContainers/StatsGeneratorDataContainer/CellEnsembleData/Statistics/{phase_key}/ODF"
        feature_mean_path = f"/DataContainers/StatsGeneratorDataContainer/CellEnsembleData/Statistics/{phase_key}/FeatureSize Distribution/Average"
        feature_std_path = f"/DataContainers/StatsGeneratorDataContainer/CellEnsembleData/Statistics/{phase_key}/FeatureSize Distribution/Standard Deviation"

        feature_mean_ds = _safe_hdf_dataset(handle, feature_mean_path)
        feature_std_ds = _safe_hdf_dataset(handle, feature_std_path)
        if feature_mean_ds is not None:
            try:
                feature_mean = float(np.asarray(feature_mean_ds[()]).reshape(-1)[0])
                data = [{"metric": "Average", "value": feature_mean}]
                if feature_std_ds is not None:
                    feature_std = float(np.asarray(feature_std_ds[()]).reshape(-1)[0])
                    data.append({"metric": "Std. dev.", "value": feature_std})
                chart = _build_materials_bar_chart(
                    title=f"Feature size summary ({phase_label})",
                    data=data,
                    source_paths=[
                        path
                        for path in (feature_mean_path, feature_std_path)
                        if _safe_hdf_dataset(handle, path) is not None
                    ],
                    description="The DREAM3D stats container exposes summary moments for feature size in this phase.",
                    x_key="metric",
                    y_key="value",
                )
                if chart:
                    charts.append(chart)
            except Exception:
                pass

        bin_ds = _safe_hdf_dataset(handle, bin_path)
        avg_ds = _safe_hdf_dataset(handle, average_path)
        if bin_ds is not None and avg_ds is not None:
            try:
                bins = np.asarray(bin_ds[()]).reshape(-1)
                averages = np.asarray(avg_ds[()]).reshape(-1)
                data = [
                    {"bin": float(bins[index]), "average": float(averages[index])}
                    for index in range(min(len(bins), len(averages), 64))
                    if math.isfinite(float(bins[index])) and math.isfinite(float(averages[index]))
                ]
                chart = _build_materials_bar_chart(
                    title=f"Feature size vs neighbors ({phase_label})",
                    data=data,
                    source_paths=[bin_path, average_path],
                    description="Average feature-size trend by neighbor-count bin from the synthetic statistics container.",
                    x_key="bin",
                    y_key="average",
                    provenance="Directly read from the DREAM3D statistics container.",
                )
                if chart:
                    charts.append(chart)
            except Exception:
                pass

        misorientation_ds = _safe_hdf_dataset(handle, misorientation_path)
        if misorientation_ds is not None:
            chart = _build_materials_histogram_chart(
                title=f"Misorientation bins ({phase_label})",
                values=np.asarray(misorientation_ds[()]).reshape(-1),
                source_paths=[misorientation_path],
                description="Stored misorientation values from the synthetic statistics container.",
                bins=24,
            )
            if chart:
                charts.append(chart)

        odf_ds = _safe_hdf_dataset(handle, odf_path)
        if odf_ds is not None:
            chart = _build_materials_histogram_chart(
                title=f"ODF values ({phase_label})",
                values=np.asarray(odf_ds[()]).reshape(-1),
                source_paths=[odf_path],
                description="Stored ODF values from the synthetic statistics container. No reconstruction is performed in the viewer.",
                bins=24,
            )
            if chart:
                charts.append(chart)

    return charts


def build_hdf5_materials_dashboard(
    *,
    file_id: str,
    file_path: str | Path,
) -> dict[str, Any]:
    h5py, h5py_error = _load_h5py_module()
    if h5py is None:
        raise ValueError(f"h5py is unavailable: {h5py_error}")

    source = Path(str(file_path)).expanduser()
    with h5py.File(source, "r") as handle:
        materials_payload = _detect_hdf5_materials(handle)
        if not bool(materials_payload.get("detected")):
            raise ValueError(
                "Materials dashboard is only available for detected DREAM3D-style HDF5 resources."
            )
        geometry = _extract_hdf_geometry(handle)
        maps = _build_materials_map_cards(handle, materials_payload)
        return {
            "file_id": file_id,
            "schema": "dream3d",
            "overview": {
                "geometry": geometry,
                "spacing_note": "Spacing and origin come from linked geometry metadata. Verify calibration before quantitative interpretation.",
                "phase_names": [
                    str(name) for name in list(materials_payload.get("phase_names") or [])
                ],
                "feature_count": materials_payload.get("feature_count"),
                "grain_count": materials_payload.get("grain_count"),
                "capabilities": [
                    str(item) for item in list(materials_payload.get("capabilities") or [])
                ],
                "recommended_map_dataset_path": materials_payload.get(
                    "recommended_map_dataset_path"
                )
                or (maps[0]["dataset_path"] if maps else None),
            },
            "maps": maps,
            "grain_charts": _build_materials_grain_charts(handle, materials_payload),
            "orientation_charts": _build_materials_orientation_charts(handle, materials_payload),
            "synthetic_stats": _build_materials_synthetic_stats(handle, materials_payload),
            "dataset_links": _build_materials_dataset_links(materials_payload),
        }


def _build_hdf_tree_node(
    *,
    node: Any,
    path: str,
    depth: int,
    state: dict[str, Any],
) -> dict[str, Any] | None:
    h5py, _ = _load_h5py_module()
    if h5py is None:
        return None
    if int(state["node_count"]) >= int(state["max_nodes"]):
        state["truncated"] = True
        return None
    state["node_count"] += 1
    name = path.rsplit("/", 1)[-1] if "/" in path else path
    attributes_count = len(getattr(node, "attrs", {}))

    if isinstance(node, h5py.Dataset):
        state["dataset_count"] += 1
        shape = _hdf_shape_list(node.shape)
        dtype_name = str(node.dtype)
        preview_kind = _classify_hdf_preview_kind(path, shape, dtype_name)
        if preview_kind:
            dataset_kinds = state.setdefault("dataset_kinds", {})
            dataset_kinds[preview_kind] = int(dataset_kinds.get(preview_kind, 0)) + 1
            if not state.get("default_dataset_path") and preview_kind not in {
                "array",
                "series",
                "scalar",
            }:
                state["default_dataset_path"] = path
        return {
            "path": path,
            "name": name,
            "node_type": "dataset",
            "child_count": 0,
            "attributes_count": attributes_count,
            "shape": shape,
            "dtype": dtype_name,
            "preview_kind": preview_kind,
            "children": [],
        }

    state["group_count"] += 1
    children: list[dict[str, Any]] = []
    try:
        child_names = list(node.keys())
    except Exception:
        child_names = []
    child_count = len(child_names)
    if child_count > int(state["max_children"]):
        state["truncated"] = True
    if depth < int(state["max_depth"]):
        for child_name in child_names[: int(state["max_children"])]:
            child = node.get(child_name)
            if child is None:
                continue
            child_path = f"{path}/{child_name}" if path != "/" else f"/{child_name}"
            child_node = _build_hdf_tree_node(
                node=child,
                path=child_path,
                depth=depth + 1,
                state=state,
            )
            if child_node is not None:
                children.append(child_node)
    return {
        "path": path,
        "name": name or "/",
        "node_type": "group",
        "child_count": child_count,
        "attributes_count": attributes_count,
        "shape": None,
        "dtype": None,
        "preview_kind": None,
        "children": children,
    }


def build_hdf5_viewer_manifest(
    *,
    file_id: str,
    file_path: str | Path,
    original_name: str,
    enabled: bool = True,
) -> dict[str, Any]:
    default_plane = _build_plane_descriptor(
        axis="z",
        axis_sizes={"T": 1, "C": 1, "Z": 1, "Y": 1, "X": 1},
        physical_spacing=None,
    )
    default_axis_labels = _default_orientation_axis_labels()
    default_orientation_labels = _build_orientation_labels(
        row_axis="Y", col_axis="X", slice_axis=None
    )
    base_manifest: dict[str, Any] = {
        "kind": "hdf5",
        "file_id": file_id,
        "original_name": original_name,
        "modality": "unknown",
        "dims_order": "",
        "backend_mode": "hdf5",
        "axis_sizes": {"T": 1, "C": 1, "Z": 1, "Y": 1, "X": 1},
        "selected_indices": {"T": 0, "C": 0, "Z": 0},
        "is_volume": False,
        "is_timeseries": False,
        "is_multichannel": False,
        "phys": None,
        "display_defaults": None,
        "service_urls": {
            "dataset": f"/v1/uploads/{file_id}/hdf5/dataset",
            "slice": f"/v1/uploads/{file_id}/hdf5/preview/slice",
            "atlas": f"/v1/uploads/{file_id}/hdf5/preview/atlas",
            "histogram": f"/v1/uploads/{file_id}/hdf5/preview/histogram",
            "table": f"/v1/uploads/{file_id}/hdf5/preview/table",
        },
        "metadata": {
            "reader": "h5py",
            "dims_order": "",
            "array_shape": [],
            "array_dtype": "hdf5",
            "scene": None,
            "scene_count": 1,
            "header": {"Format": "HDF5"},
            "filename_hints": {},
            "exif": {},
            "geo": None,
            "dicom": None,
            "warnings": [],
        },
        "viewer": {
            "status": "ready",
            "warmup_mode": "lazy",
            "backend_mode": "hdf5",
            "default_surface": "metadata",
            "available_surfaces": ["metadata"],
            "default_axis": "z",
            "slice_axes": ["z"],
            "channel_mode": "single",
            "tile_scheme": {
                "tile_size": int(VIEWER_TILE_SIZE),
                "format": "png",
                "levels": build_tile_levels(1, 1, tile_size=VIEWER_TILE_SIZE),
            },
            "default_plane": default_plane,
            "planes": {"z": default_plane},
            "volume_mode": "none",
            "render_policy": "analysis",
            "delivery_mode": "direct",
            "diagnostic_surface": "none",
            "first_paint_mode": "image",
            "measurement_policy": "pixel-only",
            "texture_policy": "nearest",
            "display_capabilities": ["dataset_explorer"],
            "viewer_capabilities": [
                "image_first_paint",
                "direct_delivery",
                "nearest_sampling",
                "dataset_explorer",
            ],
            "orientation": {
                "frame": "voxel",
                "row_axis": "Y",
                "col_axis": "X",
                "slice_axis": None,
                "axis_labels": default_axis_labels,
                "labels": default_orientation_labels,
            },
            "asset_preparation": {
                "status": "ready",
                "native_supported": True,
                "tile_pyramid": "none",
                "volume_representation": "none",
            },
            "chunk_scheme": {
                "mode": "none",
                "axis": "z",
                "sample_count": 1,
            },
            "service_urls": {},
            "fallback_urls": {},
        },
        "hdf5": {
            "enabled": bool(enabled),
            "supported": True,
            "status": "ready",
            "error": None,
            "root_keys": [],
            "root_attributes": {},
            "summary": {
                "group_count": 0,
                "dataset_count": 0,
                "dataset_kinds": {},
                "truncated": False,
                "geometry": None,
            },
            "tree": [],
            "limitations": [
                "HDF5 previews are type-gated. Native 3D is limited to eligible scalar or categorical volumes within the atlas budget, and vector or quaternion datasets stay component-aware and slice-only.",
            ],
            "selected_dataset_path": None,
            "default_dataset_path": None,
            "materials": {
                "detected": False,
                "schema": None,
                "capabilities": [],
                "roles": {},
                "phase_names": [],
                "feature_count": None,
                "grain_count": None,
                "recommended_view": "explorer",
            },
        },
    }

    if not enabled:
        base_manifest["viewer"]["status"] = "degraded-fallback"
        base_manifest["viewer"]["asset_preparation"]["status"] = "degraded-fallback"
        base_manifest["hdf5"].update(
            {
                "supported": False,
                "status": "disabled",
                "error": "HDF5 viewer support is disabled by configuration.",
            }
        )
        base_manifest["metadata"]["warnings"] = [
            "HDF5 viewer support is disabled by configuration.",
        ]
        return base_manifest

    h5py, h5py_error = _load_h5py_module()
    if h5py is None:
        base_manifest["viewer"]["status"] = "degraded-fallback"
        base_manifest["viewer"]["asset_preparation"]["status"] = "degraded-fallback"
        base_manifest["hdf5"].update(
            {
                "supported": False,
                "status": "unsupported",
                "error": f"h5py is unavailable: {h5py_error}",
            }
        )
        base_manifest["metadata"]["warnings"] = [
            "HDF5 viewer dependency missing. Install h5py to inspect this file.",
        ]
        return base_manifest

    source = Path(str(file_path)).expanduser()
    try:
        with h5py.File(source, "r") as handle:
            root_keys = [str(key) for key in list(handle.keys())]
            root_attrs: dict[str, Any] = {}
            for index, (key, value) in enumerate(handle.attrs.items()):
                if index >= _HDF5_ROOT_ATTR_LIMIT:
                    root_attrs["..."] = (
                        f"{len(handle.attrs) - _HDF5_ROOT_ATTR_LIMIT} more attributes"
                    )
                    break
                root_attrs[str(key)] = _json_safe_hdf_value(value)
            state: dict[str, Any] = {
                "node_count": 0,
                "max_nodes": _HDF5_TREE_MAX_NODES,
                "max_children": _HDF5_TREE_MAX_CHILDREN,
                "max_depth": 4,
                "group_count": 0,
                "dataset_count": 0,
                "dataset_kinds": {},
                "truncated": False,
                "default_dataset_path": None,
            }
            tree: list[dict[str, Any]] = []
            for key in root_keys[:_HDF5_TREE_MAX_CHILDREN]:
                obj = handle.get(key)
                if obj is None:
                    continue
                node = _build_hdf_tree_node(node=obj, path=f"/{key}", depth=0, state=state)
                if node is not None:
                    tree.append(node)
            if len(root_keys) > _HDF5_TREE_MAX_CHILDREN:
                state["truncated"] = True
            geometry = _extract_hdf_geometry(handle)

            base_manifest["metadata"]["header"] = {
                "Format": "HDF5",
                "Root groups": str(len(root_keys)),
            }
            if geometry and isinstance(geometry.get("dimensions"), list):
                base_manifest["metadata"]["header"]["Geometry dimensions"] = " × ".join(
                    str(value) for value in geometry["dimensions"]
                )
            base_manifest["metadata"]["warnings"] = (
                ["Tree traversal truncated for performance."] if state["truncated"] else []
            )
            materials_payload = _detect_hdf5_materials(handle)
            default_dataset_path = state.get("default_dataset_path")
            if materials_payload.get("detected") and materials_payload.get(
                "recommended_map_dataset_path"
            ):
                default_dataset_path = str(materials_payload.get("recommended_map_dataset_path"))
            public_materials_payload = {
                key: value
                for key, value in materials_payload.items()
                if key != "recommended_map_dataset_path"
            }
            measurement_policy = _measurement_policy(
                orientation_frame="voxel",
                physical_spacing=_hdf_spacing_from_geometry(geometry),
            )
            display_capabilities = ["dataset_explorer"]
            if materials_payload.get("detected"):
                display_capabilities.append("materials_dashboard")
            viewer_capabilities = _viewer_capabilities(
                delivery_mode="direct",
                first_paint_mode="image",
                diagnostic_surface="none",
                texture_policy="nearest",
                display_capabilities=display_capabilities,
            )
            base_manifest["modality"] = (
                "materials" if materials_payload.get("detected") else "unknown"
            )
            base_manifest["viewer"].update(
                {
                    "measurement_policy": measurement_policy,
                    "display_capabilities": display_capabilities,
                    "viewer_capabilities": viewer_capabilities,
                }
            )
            base_manifest["hdf5"].update(
                {
                    "root_keys": root_keys,
                    "root_attributes": root_attrs,
                    "summary": {
                        "group_count": int(state["group_count"]),
                        "dataset_count": int(state["dataset_count"]),
                        "dataset_kinds": dict(state["dataset_kinds"]),
                        "truncated": bool(state["truncated"]),
                        "geometry": geometry,
                    },
                    "tree": tree,
                    "default_dataset_path": default_dataset_path,
                    "materials": public_materials_payload,
                }
            )
            return base_manifest
    except Exception as exc:
        base_manifest["viewer"]["status"] = "degraded-fallback"
        base_manifest["viewer"]["asset_preparation"]["status"] = "degraded-fallback"
        base_manifest["hdf5"].update(
            {
                "supported": False,
                "status": "unsupported",
                "error": str(exc),
            }
        )
        base_manifest["metadata"]["warnings"] = [f"HDF5 file could not be inspected: {exc}"]
        return base_manifest


def build_viewer_manifest(
    *,
    payload: dict[str, Any],
    file_id: str,
    original_name: str,
    tile_size: int = VIEWER_TILE_SIZE,
    atlas_max_dimension: int = 2048,
) -> dict[str, Any]:
    axis_sizes = payload.get("axis_sizes") or {"T": 1, "C": 1, "Z": 1, "Y": 1, "X": 1}
    physical_spacing = _normalize_physical_spacing(payload.get("physical_spacing"))
    is_volume = bool(payload.get("is_volume"))
    slice_axes = ["z", "y", "x"] if is_volume else ["z"]
    plane_descriptors = {
        axis: _build_plane_descriptor(
            axis=axis, axis_sizes=axis_sizes, physical_spacing=physical_spacing
        )
        for axis in slice_axes
    }
    default_plane = plane_descriptors["z"]
    tile_levels = build_tile_levels(
        width=int(default_plane["pixel_size"]["width"]),
        height=int(default_plane["pixel_size"]["height"]),
        tile_size=tile_size,
    )
    metadata_payload = _extract_metadata_payload(payload)
    modality = _infer_modality(payload=payload, original_name=original_name)
    orientation_payload = (
        metadata_payload.get("orientation")
        if isinstance(metadata_payload.get("orientation"), dict)
        else {}
    )
    orientation_frame = str(
        orientation_payload.get("frame") or ("voxel" if is_volume else "pixel")
    ).strip().lower() or ("voxel" if is_volume else "pixel")
    axis_labels = _normalize_orientation_axis_labels(orientation_payload.get("axis_labels"))
    row_axis = _axis_positive_label(
        axis_labels, default_plane["axes"][0] if default_plane["axes"] else "Y", "Y"
    )
    col_axis = _axis_positive_label(
        axis_labels, default_plane["axes"][1] if len(default_plane["axes"]) > 1 else "X", "X"
    )
    slice_axis = _axis_positive_label(axis_labels, "Z", "Z") if is_volume else None
    render_policy = _image_render_policy(payload=payload, modality=modality, axis_sizes=axis_sizes)
    measurement_policy = _measurement_policy(
        orientation_frame=orientation_frame,
        physical_spacing=physical_spacing,
    )
    diagnostic_surface = _diagnostic_surface(
        render_policy=render_policy,
        modality=modality,
        is_volume=is_volume,
    )
    channel_count = max(1, int(axis_sizes.get("C") or 1))
    scalar_volume_supported = bool(
        is_volume and render_policy == "scalar" and (channel_count == 1 or modality == "microscopy")
    )
    atlas_volume_supported = bool(is_volume and render_policy == "categorical")
    native_volume_supported = bool(scalar_volume_supported or atlas_volume_supported)
    surfaces = ["2d", "metadata"]
    if is_volume:
        surfaces.insert(1, "mpr")
        if native_volume_supported:
            surfaces.insert(2, "volume")
    display_capabilities = _image_display_capabilities(
        render_policy=render_policy,
        modality=modality,
        is_volume=is_volume,
        native_volume_supported=native_volume_supported,
        channel_count=channel_count,
        measurement_policy=measurement_policy,
    )
    default_surface = (
        str(diagnostic_surface)
        if str(diagnostic_surface) in surfaces
        else ("volume" if native_volume_supported else "2d")
    )
    array_min = float(payload.get("array_min") or 0.0)
    array_max = float(payload.get("array_max") or 0.0)
    phys = _build_phys(payload=payload, file_id=file_id, original_name=original_name)
    display_defaults = _build_display_defaults(payload=payload, phys=phys, modality=modality)
    delivery_policy = _select_2d_delivery_policy(axis_sizes=axis_sizes, is_volume=is_volume)
    atlas_scheme = _build_atlas_scheme(
        slice_width=int(default_plane["pixel_size"]["width"]),
        slice_height=int(default_plane["pixel_size"]["height"]),
        slice_count=int(axis_sizes.get("Z") or 1),
        max_dimension=atlas_max_dimension,
    )
    backend_mode = (
        "scalar"
        if scalar_volume_supported
        else (
            "atlas"
            if atlas_volume_supported
            else ("direct" if is_volume else str(delivery_policy["backend_mode"]))
        )
    )
    volume_mode = (
        "scalar" if scalar_volume_supported else ("atlas" if atlas_volume_supported else "none")
    )
    delivery_mode = _image_delivery_mode(
        backend_mode=backend_mode,
        tile_pyramid=str(delivery_policy.get("tile_pyramid") or ""),
        is_volume=is_volume,
    )
    texture_policy = _texture_policy(render_policy)
    first_paint_mode = _first_paint_mode(
        default_surface=default_surface,
        delivery_mode=delivery_mode,
    )
    viewer_capabilities = _viewer_capabilities(
        delivery_mode=delivery_mode,
        first_paint_mode=first_paint_mode,
        diagnostic_surface=diagnostic_surface,
        texture_policy=texture_policy,
        display_capabilities=display_capabilities,
    )
    service_urls = {
        "preview": f"/v1/uploads/{file_id}/preview",
        "display": (
            f"/v1/uploads/{file_id}/display"
            if render_policy == "display"
            and not is_volume
            and is_ordinary_display_image_path(original_name)
            else None
        ),
        "slice": f"/v1/uploads/{file_id}/slice",
        "tile": f"/v1/uploads/{file_id}/tiles",
        "atlas": f"/v1/uploads/{file_id}/atlas" if atlas_volume_supported else None,
        "scalar_volume": f"/v1/uploads/{file_id}/scalar-volume"
        if scalar_volume_supported
        else None,
        "histogram": f"/v1/uploads/{file_id}/histogram",
    }

    metadata_warnings = list(payload.get("warnings") or [])
    delivery_warning = delivery_policy.get("warning")
    if delivery_warning and delivery_warning not in metadata_warnings:
        metadata_warnings.append(str(delivery_warning))

    return {
        "kind": "image",
        "file_id": file_id,
        "original_name": original_name,
        "modality": modality,
        "dims_order": payload.get("dims_order") or "TCZYX",
        "axis_sizes": axis_sizes,
        "selected_indices": payload.get("selected_indices") or {"T": 0, "C": 0, "Z": 0},
        "is_volume": is_volume,
        "is_timeseries": bool(payload.get("is_timeseries")),
        "is_multichannel": bool(payload.get("is_multichannel")),
        "backend_mode": backend_mode,
        "phys": phys,
        "display_defaults": display_defaults,
        "service_urls": service_urls,
        "metadata": {
            "reader": payload.get("reader") or "unknown",
            "dims_order": payload.get("dims_order") or "TCZYX",
            "array_shape": payload.get("array_shape") or [],
            "array_dtype": payload.get("array_dtype") or "unknown",
            "array_min": array_min,
            "array_max": array_max,
            "intensity_stats": {"min": array_min, "max": array_max},
            "physical_spacing": physical_spacing,
            "scene": payload.get("scene"),
            "scene_count": int(payload.get("scene_count") or len(payload.get("scenes") or []) or 1),
            "header": metadata_payload.get("header") or {},
            "filename_hints": metadata_payload.get("filename_hints") or {},
            "exif": metadata_payload.get("exif") or {},
            "warnings": metadata_warnings,
            "geo": metadata_payload.get("geo")
            if isinstance(metadata_payload.get("geo"), dict)
            else None,
            "dicom": phys.get("dicom"),
            "microscopy": metadata_payload.get("microscopy")
            if isinstance(metadata_payload.get("microscopy"), dict)
            else None,
        },
        "viewer": {
            "status": "ready",
            "warmup_mode": "lazy",
            "backend_mode": backend_mode,
            "default_surface": default_surface,
            "available_surfaces": surfaces,
            "default_axis": "z",
            "slice_axes": list(plane_descriptors.keys()),
            "channel_mode": "composite",
            "tile_scheme": {
                "tile_size": int(tile_size),
                "format": "png",
                "levels": tile_levels,
            },
            "atlas_scheme": atlas_scheme,
            "default_plane": default_plane,
            "planes": plane_descriptors,
            "volume_mode": volume_mode,
            "render_policy": render_policy,
            "delivery_mode": delivery_mode,
            "diagnostic_surface": diagnostic_surface,
            "first_paint_mode": first_paint_mode,
            "measurement_policy": measurement_policy,
            "texture_policy": texture_policy,
            "display_capabilities": display_capabilities,
            "viewer_capabilities": viewer_capabilities,
            "orientation": {
                "frame": orientation_frame,
                "row_axis": row_axis,
                "col_axis": col_axis,
                "slice_axis": slice_axis,
                "axis_labels": axis_labels,
                "labels": _build_orientation_labels(
                    row_axis=row_axis,
                    col_axis=col_axis,
                    slice_axis=slice_axis,
                ),
            },
            "asset_preparation": {
                "status": "ready",
                "native_supported": True,
                "tile_pyramid": (
                    str(delivery_policy["tile_pyramid"])
                    if not is_volume
                    else ("lazy" if native_volume_supported else "none")
                ),
                "volume_representation": volume_mode,
            },
            "chunk_scheme": {
                "mode": volume_mode,
                "axis": "z",
                "sample_count": int(axis_sizes.get("Z") or 1) if is_volume else 1,
            },
            "display_defaults": display_defaults,
            "service_urls": service_urls,
            "fallback_urls": {
                "preview": service_urls["preview"],
                "slice": service_urls["slice"],
            },
        },
        "hdf5": None,
    }


def _clamp_plane_index(index: int | None, size: int) -> int:
    if size <= 1:
        return 0
    if index is None:
        return max(0, size // 2)
    return max(0, min(int(index), size - 1))


def _extract_axis_from_volume(
    volume: np.ndarray,
    *,
    array_order: str,
    axis: ViewAxis,
    x_index: int | None = None,
    y_index: int | None = None,
) -> np.ndarray:
    arr = np.asarray(volume)
    order = str(array_order or "").upper()
    if axis == "z":
        return arr
    if order == "CZYX":
        _c_size, z_size, y_size, x_size = map(int, arr.shape)
        if axis == "y":
            return arr[:, :, _clamp_plane_index(y_index, y_size), :]
        return arr[:, :, :, _clamp_plane_index(x_index, x_size)]
    if order == "ZYX":
        _z_size, y_size, x_size = map(int, arr.shape)
        if axis == "y":
            return arr[:, _clamp_plane_index(y_index, y_size), :]
        return arr[:, :, _clamp_plane_index(x_index, x_size)]
    raise ValueError(f"Unsupported volume array order for axis extraction: {order or 'unknown'}")


def _coerce_volume_to_czyx(volume: np.ndarray, *, array_order: str) -> np.ndarray:
    arr = np.asarray(volume)
    order = str(array_order or "").upper()
    if order == "CZYX":
        return arr
    if order == "ZYX":
        return arr[None, ...]
    raise ValueError(f"Unsupported volume array order for atlas rendering: {order or 'unknown'}")


def _normalize_hdf_volume_array(
    array: np.ndarray, *, preview_kind: str | None
) -> tuple[np.ndarray, str, dict[str, int], str]:
    arr = np.asarray(array)
    safe_kind = str(preview_kind or "").strip().lower()
    if safe_kind in {"scalar_volume", "label_volume"}:
        if arr.ndim >= 4 and int(arr.shape[-1]) == 1:
            arr = arr[..., 0]
        if arr.ndim != 3:
            raise ValueError("HDF5 scalar or label volume must resolve to a 3D array.")
        semantic_kind = "label" if safe_kind == "label_volume" else "scalar"
        axis_sizes = {
            "T": 1,
            "C": 1,
            "Z": int(arr.shape[0]),
            "Y": int(arr.shape[1]),
            "X": int(arr.shape[2]),
        }
        return arr, "ZYX", axis_sizes, semantic_kind
    if safe_kind == "rgb_volume":
        if arr.ndim != 4 or int(arr.shape[-1]) < 3:
            raise ValueError("HDF5 RGB volume must resolve to a Z/Y/X/3 array.")
        rgb = np.moveaxis(arr[..., :3], -1, 0)
        axis_sizes = {
            "T": 1,
            "C": int(rgb.shape[0]),
            "Z": int(rgb.shape[1]),
            "Y": int(rgb.shape[2]),
            "X": int(rgb.shape[3]),
        }
        return rgb, "CZYX", axis_sizes, "rgb"
    raise ValueError("Selected HDF5 dataset is not eligible for shared volume loading.")


def _load_view_volume_uncached(
    *,
    file_path: str,
    t_index: int | None = None,
    max_inline_elements: int = 1024,
) -> tuple[dict[str, Any], np.ndarray]:
    volume_payload = load_scientific_image(
        file_path=str(file_path),
        array_mode="volume",
        t_index=t_index,
        generate_preview=False,
        save_array=False,
        include_array=False,
        max_inline_elements=max_inline_elements,
        return_array=True,
    )
    if not volume_payload.get("success"):
        raise ValueError(str(volume_payload.get("error") or "Failed to load volume image."))
    volume = volume_payload.pop("_array", None)
    if volume is None:
        raise FileNotFoundError("Volume payload unavailable for atlas rendering.")
    return volume_payload, np.asarray(volume)


def _load_view_volume_source_uncached(
    *,
    file_path: str,
    dataset_path: str | None = None,
    t_index: int | None = None,
    max_inline_elements: int = 1024,
) -> tuple[dict[str, Any], np.ndarray]:
    if dataset_path:
        normalized_path = _normalize_hdf_dataset_path(dataset_path)
        h5py, h5py_error = _load_h5py_module()
        if h5py is None:
            raise ValueError(f"h5py is unavailable: {h5py_error}")

        source = Path(str(file_path)).expanduser()
        with h5py.File(source, "r") as handle:
            node = handle.get(normalized_path)
            if node is None:
                raise FileNotFoundError(f"Dataset not found: {normalized_path}")
            if not isinstance(node, h5py.Dataset):
                raise ValueError(f"Path does not point to a dataset: {normalized_path}")

            shape = _hdf_shape_list(node.shape)
            preview_kind = _classify_hdf_preview_kind(normalized_path, shape, str(node.dtype))
            volume_eligible, volume_reason = _hdf_volume_eligibility(preview_kind)
            if not volume_eligible:
                raise ValueError(
                    volume_reason
                    or "Selected HDF5 dataset is not eligible for shared volume loading."
                )

            geometry = _extract_hdf_dataset_geometry(handle, normalized_path)
            physical_spacing = _hdf_spacing_from_geometry(geometry)
            array = np.asarray(node[()])
            volume, array_order, axis_sizes, semantic_kind = _normalize_hdf_volume_array(
                array,
                preview_kind=preview_kind,
            )
            payload = {
                "success": True,
                "reader": "h5py",
                "file_path": str(source.resolve()),
                "dataset_path": normalized_path,
                "dims_order": array_order,
                "native_dims_order": array_order,
                "axis_sizes": axis_sizes,
                "selected_indices": {"T": 0, "C": 0, "Z": 0},
                "array_mode": "volume",
                "array_order": array_order,
                "native_array_order": array_order,
                "array_shape": list(volume.shape),
                "array_dtype": str(node.dtype),
                "array_min": float(np.min(array)) if int(np.size(array)) else 0.0,
                "array_max": float(np.max(array)) if int(np.size(array)) else 0.0,
                "physical_spacing": physical_spacing,
                "is_volume": True,
                "is_timeseries": False,
                "is_multichannel": bool(axis_sizes.get("C", 1) > 1),
                "warnings": [],
                "source_kind": "hdf5",
                "semantic_kind": semantic_kind,
                "preview_kind": preview_kind,
                "metadata": {
                    "header": {"Format": "HDF5"},
                    "hdf5": {"dataset_path": normalized_path},
                    "geometry": geometry,
                },
            }
            return payload, np.asarray(volume)

    payload, volume = _load_view_volume_uncached(
        file_path=file_path,
        t_index=t_index,
        max_inline_elements=max_inline_elements,
    )
    source_payload = dict(payload)
    source_payload.setdefault("source_kind", "image")
    source_payload.setdefault("semantic_kind", "scalar")
    return source_payload, np.asarray(volume)


def load_view_volume(
    *,
    file_path: str,
    t_index: int | None = None,
    max_inline_elements: int = 1024,
) -> tuple[dict[str, Any], np.ndarray]:
    return get_cached_file_derivative(
        derivative_kind="volume_source",
        file_path=file_path,
        factory=lambda: _load_view_volume_uncached(
            file_path=file_path,
            t_index=t_index,
            max_inline_elements=max_inline_elements,
        ),
        t_index=t_index,
        max_inline_elements=max_inline_elements,
    )


def load_view_volume_source(
    *,
    file_path: str,
    dataset_path: str | None = None,
    t_index: int | None = None,
    max_inline_elements: int = 1024,
) -> tuple[dict[str, Any], np.ndarray]:
    return get_cached_file_derivative(
        derivative_kind="volume_source",
        file_path=file_path,
        factory=lambda: _load_view_volume_source_uncached(
            file_path=file_path,
            dataset_path=dataset_path,
            t_index=t_index,
            max_inline_elements=max_inline_elements,
        ),
        dataset_path=str(dataset_path or ""),
        t_index=t_index,
        max_inline_elements=max_inline_elements,
    )


def _load_view_plane_uncached(
    *,
    file_path: str,
    axis: ViewAxis = "z",
    x_index: int | None = None,
    y_index: int | None = None,
    z_index: int | None = None,
    t_index: int | None = None,
    max_inline_elements: int = 1024,
) -> tuple[dict[str, Any], np.ndarray]:
    axis = normalize_view_axis(axis)
    if axis == "z":
        payload = load_scientific_image(
            file_path=str(file_path),
            array_mode="plane",
            t_index=t_index,
            z_index=z_index,
            generate_preview=False,
            save_array=False,
            include_array=False,
            max_inline_elements=max_inline_elements,
            return_array=True,
        )
        if not payload.get("success"):
            raise ValueError(str(payload.get("error") or "Failed to load plane image."))
        plane = payload.pop("_array", None)
        if plane is None:
            preview_path = Path(str(payload.get("preview_path") or "")).expanduser()
            if not preview_path.exists() or not preview_path.is_file():
                raise FileNotFoundError("Plane preview unavailable.")
            plane = np.asarray(Image.open(preview_path))
        return payload, np.asarray(plane)

    volume_payload, volume = load_view_volume(
        file_path=file_path,
        t_index=t_index,
        max_inline_elements=max_inline_elements,
    )
    plane = _extract_axis_from_volume(
        np.asarray(volume),
        array_order=str(volume_payload.get("array_order") or ""),
        axis=axis,
        x_index=x_index,
        y_index=y_index,
    )
    return volume_payload, plane


def load_view_plane(
    *,
    file_path: str,
    axis: ViewAxis = "z",
    x_index: int | None = None,
    y_index: int | None = None,
    z_index: int | None = None,
    t_index: int | None = None,
    max_inline_elements: int = 1024,
) -> tuple[dict[str, Any], np.ndarray]:
    normalized_axis = normalize_view_axis(axis)
    return get_cached_file_derivative(
        derivative_kind="plane_source",
        file_path=file_path,
        factory=lambda: _load_view_plane_uncached(
            file_path=file_path,
            axis=normalized_axis,
            x_index=x_index,
            y_index=y_index,
            z_index=z_index,
            t_index=t_index,
            max_inline_elements=max_inline_elements,
        ),
        axis=normalized_axis,
        x_index=x_index,
        y_index=y_index,
        z_index=z_index,
        t_index=t_index,
        max_inline_elements=max_inline_elements,
    )


def _parse_channel_color_hex(value: str | None) -> list[int] | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.startswith("#"):
        text = text[1:]
    if len(text) != 6:
        return None
    try:
        return [int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)]
    except Exception:
        return None


def _resolve_channel_colors(
    *,
    payload: dict[str, Any],
    override_colors: list[str] | None = None,
) -> list[list[int]]:
    phys = _build_phys(payload=payload, file_id="internal", original_name="internal")
    colors = [entry.get("rgb") or [255, 255, 255] for entry in phys.get("channel_colors", [])]
    if not override_colors:
        return colors
    output: list[list[int]] = []
    max_count = max(len(colors), len(override_colors))
    for index in range(max_count):
        parsed = (
            _parse_channel_color_hex(override_colors[index])
            if index < len(override_colors)
            else None
        )
        if parsed is not None:
            output.append(parsed)
        elif index < len(colors):
            output.append([int(component) for component in colors[index][:3]])
        else:
            output.append([255, 255, 255])
    return output


def _render_scalar_plane_rgb(
    plane: np.ndarray,
    *,
    enhancement: str,
    window_center: float | None,
    window_width: float | None,
    negative: bool,
) -> np.ndarray:
    normalized = _normalize_to_u8(
        np.asarray(plane),
        enhancement=enhancement,
        window_center=window_center,
        window_width=window_width,
    )
    output = np.repeat(normalized[..., None], 3, axis=-1)
    if negative:
        output = 255 - output
    return output


def _fuse_plane_to_rgb(
    plane: np.ndarray,
    *,
    channel_colors: list[list[int]],
    channel_indices: list[int],
    fusion_method: str,
    enhancement: str,
    window_center: float | None,
    window_width: float | None,
    negative: bool,
) -> np.ndarray:
    array = np.asarray(plane)
    if array.ndim == 2:
        return _render_scalar_plane_rgb(
            array,
            enhancement=enhancement,
            window_center=window_center,
            window_width=window_width,
            negative=negative,
        )
    if array.ndim == 3 and array.shape[0] <= 4 and array.shape[-1] > 4:
        volume = array[:, None, :, :]
        return _fuse_volume_to_rgb(
            volume,
            channel_colors=channel_colors,
            channel_indices=channel_indices,
            fusion_method=fusion_method,
            enhancement=enhancement,
            window_center=window_center,
            window_width=window_width,
            negative=negative,
        )[0]
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        output = _render_preview_image(array)
        if negative:
            output = 255 - output
        return output
    return _render_scalar_plane_rgb(
        np.squeeze(array),
        enhancement=enhancement,
        window_center=window_center,
        window_width=window_width,
        negative=negative,
    )


def _render_view_plane_image_uncached(
    *,
    file_path: str,
    axis: ViewAxis = "z",
    x_index: int | None = None,
    y_index: int | None = None,
    z_index: int | None = None,
    t_index: int | None = None,
    enhancement: str = "d",
    fusion_method: str = "m",
    negative: bool = False,
    channel_indices: list[int] | None = None,
    channel_colors: list[str] | None = None,
    max_inline_elements: int = 1024,
) -> np.ndarray:
    payload, plane = load_view_plane(
        file_path=file_path,
        axis=axis,
        x_index=x_index,
        y_index=y_index,
        z_index=z_index,
        t_index=t_index,
        max_inline_elements=max_inline_elements,
    )
    modality = _infer_modality(payload=payload, original_name=str(Path(file_path).name))
    render_policy = _image_render_policy(
        payload=payload,
        modality=modality,
        axis_sizes=payload.get("axis_sizes") or {},
    )
    if render_policy == "categorical":
        output = _render_hdf_label_plane(np.asarray(plane))
        if negative:
            output = 255 - output
        return output
    if render_policy == "display":
        output = _render_preview_image(np.asarray(plane))
        if negative:
            output = 255 - output
        return output
    phys = _build_phys(payload=payload, file_id="internal", original_name="internal")
    dicom = phys.get("dicom") if isinstance(phys.get("dicom"), dict) else {}
    window_center = dicom.get("wnd_center") if dicom else None
    window_width = dicom.get("wnd_width") if dicom else None
    colors = _resolve_channel_colors(payload=payload, override_colors=channel_colors)
    return _fuse_plane_to_rgb(
        np.asarray(plane),
        channel_colors=colors,
        channel_indices=channel_indices or [],
        fusion_method=fusion_method,
        enhancement=enhancement,
        window_center=float(window_center) if window_center is not None else None,
        window_width=float(window_width) if window_width is not None else None,
        negative=negative,
    )


def render_view_plane_image(
    *,
    file_path: str,
    axis: ViewAxis = "z",
    x_index: int | None = None,
    y_index: int | None = None,
    z_index: int | None = None,
    t_index: int | None = None,
    enhancement: str = "d",
    fusion_method: str = "m",
    negative: bool = False,
    channel_indices: list[int] | None = None,
    channel_colors: list[str] | None = None,
    max_inline_elements: int = 1024,
) -> np.ndarray:
    normalized_axis = normalize_view_axis(axis)
    normalized_channels = tuple(int(value) for value in (channel_indices or []))
    normalized_colors = tuple(str(value) for value in (channel_colors or []))
    return get_cached_file_derivative(
        derivative_kind="plane_rgb",
        file_path=file_path,
        factory=lambda: _render_view_plane_image_uncached(
            file_path=file_path,
            axis=normalized_axis,
            x_index=x_index,
            y_index=y_index,
            z_index=z_index,
            t_index=t_index,
            enhancement=enhancement,
            fusion_method=fusion_method,
            negative=negative,
            channel_indices=list(normalized_channels),
            channel_colors=list(normalized_colors),
            max_inline_elements=max_inline_elements,
        ),
        axis=normalized_axis,
        x_index=x_index,
        y_index=y_index,
        z_index=z_index,
        t_index=t_index,
        enhancement=enhancement,
        fusion_method=fusion_method,
        negative=bool(negative),
        channel_indices=normalized_channels,
        channel_colors=normalized_colors,
        max_inline_elements=max_inline_elements,
    )


def _normalize_window(
    array: np.ndarray, *, enhancement: str, window_center: float | None, window_width: float | None
) -> tuple[float, float]:
    finite = np.asarray(array, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    if enhancement.startswith("hounsfield"):
        parts = enhancement.split(":")
        if len(parts) >= 3:
            try:
                window_center = float(parts[1])
                window_width = float(parts[2])
            except Exception:
                pass
        if window_center is None:
            window_center = float(np.median(finite))
        if window_width is None or not math.isfinite(window_width) or window_width <= 0:
            window_width = float(np.max(finite) - np.min(finite)) or 1.0
        low = float(window_center - window_width / 2.0)
        high = float(window_center + window_width / 2.0)
        if high <= low:
            high = low + 1.0
        return low, high
    if enhancement == "f":
        low = float(np.min(finite))
        high = float(np.max(finite))
    else:
        low = float(np.percentile(finite, 1.0))
        high = float(np.percentile(finite, 99.0))
    if high <= low:
        high = low + 1.0
    return low, high


def _normalize_to_u8(
    array: np.ndarray,
    *,
    enhancement: str,
    window_center: float | None = None,
    window_width: float | None = None,
) -> np.ndarray:
    low, high = _normalize_window(
        array, enhancement=enhancement, window_center=window_center, window_width=window_width
    )
    scaled = (np.asarray(array, dtype=np.float32) - low) / max(1e-9, high - low)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _normalize_to_u16(
    array: np.ndarray,
    *,
    low: float,
    high: float,
) -> np.ndarray:
    safe_array = np.asarray(array, dtype=np.float32)
    scaled = (safe_array - float(low)) / max(1e-9, float(high) - float(low))
    scaled = np.clip(scaled, 0.0, 1.0)
    return np.round(scaled * 65535.0).astype(np.uint16)


def _select_channels(requested_channels: list[int] | None, channel_count: int) -> list[int]:
    if not requested_channels:
        return list(range(min(3, max(1, channel_count))))
    output: list[int] = []
    for value in requested_channels:
        try:
            channel = int(value)
        except Exception:
            continue
        if 0 <= channel < channel_count and channel not in output:
            output.append(channel)
    return output or [0]


def _fuse_volume_to_rgb(
    volume: np.ndarray,
    *,
    channel_colors: list[list[int]],
    channel_indices: list[int],
    fusion_method: str,
    enhancement: str,
    window_center: float | None,
    window_width: float | None,
    negative: bool,
) -> np.ndarray:
    czyx = np.asarray(volume)
    channel_count = int(czyx.shape[0])
    selected = _select_channels(channel_indices, channel_count)
    fused = np.zeros(
        (int(czyx.shape[1]), int(czyx.shape[2]), int(czyx.shape[3]), 3), dtype=np.float32
    )
    blend_count = 0
    for channel in selected:
        channel_volume = (
            _normalize_to_u8(
                czyx[channel],
                enhancement=enhancement,
                window_center=window_center,
                window_width=window_width,
            ).astype(np.float32)
            / 255.0
        )
        color = (
            np.asarray(
                channel_colors[channel] if channel < len(channel_colors) else [255, 255, 255],
                dtype=np.float32,
            )
            / 255.0
        )
        tinted = channel_volume[..., None] * color[None, None, None, :]
        if fusion_method == "a":
            fused += tinted
        else:
            fused = np.maximum(fused, tinted)
        blend_count += 1
    if fusion_method == "a" and blend_count > 0:
        fused /= float(blend_count)
    fused = np.clip(fused, 0.0, 1.0)
    output = (fused * 255.0).astype(np.uint8)
    if negative:
        output = 255 - output
    return output


def _volume_rgb(
    *,
    payload: dict[str, Any],
    volume: np.ndarray,
    enhancement: str,
    fusion_method: str,
    negative: bool,
    channel_indices: list[int] | None,
    channel_colors: list[str] | None = None,
) -> np.ndarray:
    semantic_kind = str(payload.get("semantic_kind") or "").strip().lower()
    if semantic_kind == "label":
        order = str(payload.get("array_order") or "").upper()
        arr = np.asarray(volume)
        if order == "CZYX":
            if int(arr.shape[0]) != 1:
                raise ValueError(
                    "Label volumes must not expose more than one channel in atlas rendering."
                )
            zyx = arr[0]
        elif order == "ZYX":
            zyx = arr
        else:
            raise ValueError(
                f"Unsupported label volume array order for atlas rendering: {order or 'unknown'}"
            )
        output = np.stack([_render_hdf_label_plane(plane) for plane in np.asarray(zyx)], axis=0)
        if negative:
            output = 255 - output
        return output
    czyx = _coerce_volume_to_czyx(volume, array_order=str(payload.get("array_order") or ""))
    phys = _build_phys(payload=payload, file_id="internal", original_name="internal")
    resolved_channel_colors = _resolve_channel_colors(
        payload=payload, override_colors=channel_colors
    )
    dicom = phys.get("dicom") if isinstance(phys.get("dicom"), dict) else {}
    window_center = dicom.get("wnd_center") if dicom else None
    window_width = dicom.get("wnd_width") if dicom else None
    return _fuse_volume_to_rgb(
        czyx,
        channel_colors=resolved_channel_colors,
        channel_indices=channel_indices or [],
        fusion_method=fusion_method,
        enhancement=enhancement,
        window_center=float(window_center) if window_center is not None else None,
        window_width=float(window_width) if window_width is not None else None,
        negative=negative,
    )


def render_volume_source_atlas_png(
    *,
    payload: dict[str, Any],
    volume: np.ndarray,
    enhancement: str = "d",
    fusion_method: str = "m",
    negative: bool = False,
    channel_indices: list[int] | None = None,
    channel_colors: list[str] | None = None,
    atlas_max_dimension: int = 2048,
) -> tuple[dict[str, Any], bytes]:
    volume_rgb = _volume_rgb(
        payload=payload,
        volume=volume,
        enhancement=enhancement,
        fusion_method=fusion_method,
        negative=negative,
        channel_indices=channel_indices,
        channel_colors=channel_colors,
    )
    slice_count = int(volume_rgb.shape[0])
    slice_height = int(volume_rgb.shape[1])
    slice_width = int(volume_rgb.shape[2])
    atlas_scheme = _build_atlas_scheme(
        slice_width=slice_width,
        slice_height=slice_height,
        slice_count=slice_count,
        max_dimension=atlas_max_dimension,
    )

    atlas_image = Image.new(
        "RGBA",
        (int(atlas_scheme["atlas_width"]), int(atlas_scheme["atlas_height"])),
        (0, 0, 0, 0),
    )
    resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    for slice_index in range(slice_count):
        slice_image = Image.fromarray(volume_rgb[slice_index], mode="RGB")
        if slice_image.width != int(atlas_scheme["slice_width"]) or slice_image.height != int(
            atlas_scheme["slice_height"]
        ):
            slice_image = slice_image.resize(
                (int(atlas_scheme["slice_width"]), int(atlas_scheme["slice_height"])),
                resample,
            )
        column = slice_index % int(atlas_scheme["columns"])
        row = slice_index // int(atlas_scheme["columns"])
        atlas_image.alpha_composite(
            slice_image.convert("RGBA"),
            dest=(
                column * int(atlas_scheme["slice_width"]),
                row * int(atlas_scheme["slice_height"]),
            ),
        )

    buffer = BytesIO()
    atlas_image.save(buffer, format="PNG")
    return atlas_scheme, buffer.getvalue()


def plan_volume_source_atlas(
    *,
    axis_sizes: dict[str, Any] | None,
    atlas_max_dimension: int = 2048,
) -> dict[str, Any]:
    normalized = axis_sizes or {}
    channel_count = max(1, int(normalized.get("C", 1) or 1))
    slice_count = max(1, int(normalized.get("Z", 1) or 1))
    slice_height = max(1, int(normalized.get("Y", 1) or 1))
    slice_width = max(1, int(normalized.get("X", 1) or 1))
    atlas_scheme = _build_atlas_scheme(
        slice_width=slice_width,
        slice_height=slice_height,
        slice_count=slice_count,
        max_dimension=atlas_max_dimension,
    )
    decoded_texture_bytes = int(atlas_scheme["atlas_width"]) * int(atlas_scheme["atlas_height"]) * 4
    voxel_count = channel_count * slice_count * slice_height * slice_width
    return {
        "atlas_scheme": atlas_scheme,
        "decoded_texture_bytes": int(decoded_texture_bytes),
        "voxel_count": int(voxel_count),
        "slice_count": int(slice_count),
        "slice_height": int(slice_height),
        "slice_width": int(slice_width),
        "channel_count": int(channel_count),
    }


def _render_view_tile_png_uncached(
    *,
    file_path: str,
    axis: ViewAxis,
    level: int,
    tile_x: int,
    tile_y: int,
    z_index: int | None = None,
    t_index: int | None = None,
    tile_size: int = VIEWER_TILE_SIZE,
) -> bytes:
    _payload, plane = load_view_plane(
        file_path=file_path,
        axis=axis,
        z_index=z_index,
        t_index=t_index,
        max_inline_elements=1024,
    )
    plane_u8 = _render_preview_image(np.asarray(plane))
    pil_image = Image.fromarray(plane_u8)
    levels = build_tile_levels(int(pil_image.width), int(pil_image.height), tile_size=tile_size)
    if level < 0 or level >= len(levels):
        raise ValueError(f"Tile level out of range: {level}")

    level_info = levels[level]
    target_width = int(level_info["width"])
    target_height = int(level_info["height"])
    if pil_image.width != target_width or pil_image.height != target_height:
        resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        pil_image = pil_image.resize((target_width, target_height), resample)

    tile = max(64, int(tile_size))
    left = int(tile_x) * tile
    top = int(tile_y) * tile
    if left < 0 or top < 0 or left >= pil_image.width or top >= pil_image.height:
        raise ValueError(f"Tile coordinate out of range: ({tile_x}, {tile_y})")
    right = min(pil_image.width, left + tile)
    bottom = min(pil_image.height, top + tile)
    tile_image = pil_image.crop((left, top, right, bottom))
    buffer = BytesIO()
    tile_image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_view_tile_png(
    *,
    file_path: str,
    axis: ViewAxis,
    level: int,
    tile_x: int,
    tile_y: int,
    z_index: int | None = None,
    t_index: int | None = None,
    tile_size: int = VIEWER_TILE_SIZE,
) -> bytes:
    normalized_axis = normalize_view_axis(axis)
    return get_cached_file_derivative(
        derivative_kind="tile_png",
        file_path=file_path,
        factory=lambda: _render_view_tile_png_uncached(
            file_path=file_path,
            axis=normalized_axis,
            level=level,
            tile_x=tile_x,
            tile_y=tile_y,
            z_index=z_index,
            t_index=t_index,
            tile_size=tile_size,
        ),
        axis=normalized_axis,
        level=level,
        tile_x=tile_x,
        tile_y=tile_y,
        z_index=z_index,
        t_index=t_index,
        tile_size=tile_size,
    )


def _render_view_atlas_png_uncached(
    *,
    file_path: str,
    enhancement: str = "d",
    fusion_method: str = "m",
    negative: bool = False,
    t_index: int | None = None,
    channel_indices: list[int] | None = None,
    channel_colors: list[str] | None = None,
    atlas_max_dimension: int = 2048,
) -> tuple[dict[str, Any], bytes]:
    payload, volume = load_view_volume_source(
        file_path=file_path,
        t_index=t_index,
        max_inline_elements=1024,
    )
    return render_volume_source_atlas_png(
        payload=payload,
        volume=volume,
        enhancement=enhancement,
        fusion_method=fusion_method,
        negative=negative,
        channel_indices=channel_indices,
        channel_colors=channel_colors,
        atlas_max_dimension=atlas_max_dimension,
    )


def render_view_atlas_png(
    *,
    file_path: str,
    enhancement: str = "d",
    fusion_method: str = "m",
    negative: bool = False,
    t_index: int | None = None,
    channel_indices: list[int] | None = None,
    channel_colors: list[str] | None = None,
    atlas_max_dimension: int = 2048,
) -> tuple[dict[str, Any], bytes]:
    normalized_channels = tuple(int(value) for value in (channel_indices or []))
    normalized_colors = tuple(str(value) for value in (channel_colors or []))
    return get_cached_file_derivative(
        derivative_kind="atlas_png",
        file_path=file_path,
        factory=lambda: _render_view_atlas_png_uncached(
            file_path=file_path,
            enhancement=enhancement,
            fusion_method=fusion_method,
            negative=negative,
            t_index=t_index,
            channel_indices=list(normalized_channels),
            channel_colors=list(normalized_colors),
            atlas_max_dimension=atlas_max_dimension,
        ),
        enhancement=enhancement,
        fusion_method=fusion_method,
        negative=bool(negative),
        t_index=t_index,
        channel_indices=normalized_channels,
        channel_colors=normalized_colors,
        atlas_max_dimension=atlas_max_dimension,
    )


def _render_view_histogram_uncached(
    *,
    file_path: str,
    t_index: int | None = None,
    channel_indices: list[int] | None = None,
    bins: int = DEFAULT_HISTOGRAM_BINS,
) -> dict[str, Any]:
    payload, volume = load_view_volume(
        file_path=file_path,
        t_index=t_index,
        max_inline_elements=1024,
    )
    czyx = _coerce_volume_to_czyx(volume, array_order=str(payload.get("array_order") or ""))
    selected = _select_channels(channel_indices, int(czyx.shape[0]))
    sample = czyx[selected].astype(np.float32).reshape(-1)
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        finite = np.asarray([0.0, 1.0], dtype=np.float32)
    hist, edges = np.histogram(finite, bins=max(8, int(bins or DEFAULT_HISTOGRAM_BINS)))
    return {
        "bins": hist.astype(int).tolist(),
        "edges": edges.astype(float).tolist(),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "channel_indices": selected,
        "time_index": int(t_index or 0),
    }


def render_view_histogram(
    *,
    file_path: str,
    t_index: int | None = None,
    channel_indices: list[int] | None = None,
    bins: int = DEFAULT_HISTOGRAM_BINS,
) -> dict[str, Any]:
    normalized_channels = tuple(int(value) for value in (channel_indices or []))
    return get_cached_file_derivative(
        derivative_kind="histogram",
        file_path=file_path,
        factory=lambda: _render_view_histogram_uncached(
            file_path=file_path,
            t_index=t_index,
            channel_indices=list(normalized_channels),
            bins=bins,
        ),
        clone_result=True,
        t_index=t_index,
        channel_indices=normalized_channels,
        bins=bins,
    )


def _load_scalar_volume_texture_uncached(
    *,
    file_path: str,
    dataset_path: str | None = None,
    t_index: int | None = None,
    channel_index: int | None = None,
    max_inline_elements: int = 1024,
) -> tuple[dict[str, Any], bytes]:
    payload, volume = load_view_volume_source(
        file_path=file_path,
        dataset_path=dataset_path,
        t_index=t_index,
        max_inline_elements=max_inline_elements,
    )
    modality = _infer_modality(payload=payload, original_name=str(Path(file_path).name))
    render_policy = _image_render_policy(
        payload=payload,
        modality=modality,
        axis_sizes=payload.get("axis_sizes") or {},
    )
    if render_policy != "scalar":
        raise ValueError("Selected dataset is not eligible for scalar 3D rendering.")

    array_order = str(payload.get("array_order") or "").upper()
    scalar_volume = np.asarray(volume)
    selected_channel = None
    if array_order == "CZYX":
        channel_count = max(1, int(scalar_volume.shape[0]))
        if channel_count > 1:
            if channel_index is None:
                raise ValueError(
                    "Native scalar 3D currently supports one scalar channel at a time. "
                    "Use Slice Views or choose a single channel."
                )
            safe_channel = max(0, min(int(channel_index), channel_count - 1))
            scalar_volume = scalar_volume[safe_channel]
            selected_channel = safe_channel
        else:
            scalar_volume = scalar_volume[0]
            selected_channel = 0
    elif array_order != "ZYX":
        raise ValueError(f"Unsupported scalar volume array order: {array_order or 'unknown'}")

    if scalar_volume.ndim != 3:
        raise ValueError("Scalar 3D rendering requires a single Z/Y/X volume.")

    finite = np.asarray(scalar_volume, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raw_min = 0.0
        raw_max = 1.0
    else:
        raw_min = float(np.min(finite))
        raw_max = float(np.max(finite))
        if not math.isfinite(raw_min) or not math.isfinite(raw_max) or raw_max <= raw_min:
            raw_min = float(raw_min) if math.isfinite(raw_min) else 0.0
            raw_max = raw_min + 1.0

    normalized = _normalize_to_u16(
        np.asarray(scalar_volume, dtype=np.float32),
        low=raw_min,
        high=raw_max,
    )
    axis_sizes = {
        "Z": int(normalized.shape[0]),
        "Y": int(normalized.shape[1]),
        "X": int(normalized.shape[2]),
    }
    return (
        {
            "axis_sizes": axis_sizes,
            "raw_min": raw_min,
            "raw_max": raw_max,
            "dtype": "uint16",
            "bytes_per_voxel": 2,
            "selected_channel": selected_channel,
        },
        normalized.tobytes(order="C"),
    )


def _scalar_volume_cache_dir(file_path: str | Path) -> Path:
    source = Path(file_path).expanduser().resolve()
    return source.parent / ".viewer-cache" / "scalar-volume"


def _scalar_volume_cache_prefix(file_path: str | Path) -> str:
    source = str(Path(file_path).expanduser().resolve())
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:24]


def _scalar_volume_cache_paths(
    *,
    file_path: str | Path,
    dataset_path: str | None,
    t_index: int | None,
    channel_index: int | None,
    max_inline_elements: int,
) -> tuple[Path, Path]:
    key = file_derivative_key(
        derivative_kind="scalar_volume_persistent",
        file_path=file_path,
        dataset_path=str(dataset_path or ""),
        t_index=t_index,
        channel_index=channel_index,
        max_inline_elements=max_inline_elements,
        cache_version=_SCALAR_VOLUME_CACHE_VERSION,
    )
    digest = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()
    stem = f"{_scalar_volume_cache_prefix(file_path)}-{digest}"
    cache_dir = _scalar_volume_cache_dir(file_path)
    return cache_dir / f"{stem}.json", cache_dir / f"{stem}.bin"


def _expected_scalar_volume_byte_length(metadata: dict[str, Any]) -> int:
    axis_sizes = metadata.get("axis_sizes") if isinstance(metadata.get("axis_sizes"), dict) else {}
    depth = max(0, int(axis_sizes.get("Z") or 0))
    height = max(0, int(axis_sizes.get("Y") or 0))
    width = max(0, int(axis_sizes.get("X") or 0))
    bytes_per_voxel = max(0, int(metadata.get("bytes_per_voxel") or 0))
    return depth * height * width * bytes_per_voxel


def _purge_scalar_volume_cache_paths(metadata_path: Path, data_path: Path) -> None:
    metadata_path.unlink(missing_ok=True)
    data_path.unlink(missing_ok=True)


def _read_scalar_volume_texture_from_disk(
    *,
    file_path: str,
    dataset_path: str | None,
    t_index: int | None,
    channel_index: int | None,
    max_inline_elements: int,
) -> tuple[dict[str, Any], bytes] | None:
    metadata_path, data_path = _scalar_volume_cache_paths(
        file_path=file_path,
        dataset_path=dataset_path,
        t_index=t_index,
        channel_index=channel_index,
        max_inline_elements=max_inline_elements,
    )
    if not metadata_path.exists() or not data_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("invalid scalar-volume metadata")
        volume_bytes = data_path.read_bytes()
    except Exception:
        _purge_scalar_volume_cache_paths(metadata_path, data_path)
        return None
    expected_length = _expected_scalar_volume_byte_length(metadata)
    if expected_length <= 0 or len(volume_bytes) != expected_length:
        _purge_scalar_volume_cache_paths(metadata_path, data_path)
        return None
    return metadata, volume_bytes


def _write_scalar_volume_texture_to_disk(
    *,
    file_path: str,
    dataset_path: str | None,
    t_index: int | None,
    channel_index: int | None,
    max_inline_elements: int,
    metadata: dict[str, Any],
    volume_bytes: bytes,
) -> None:
    metadata_path, data_path = _scalar_volume_cache_paths(
        file_path=file_path,
        dataset_path=dataset_path,
        t_index=t_index,
        channel_index=channel_index,
        max_inline_elements=max_inline_elements,
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_payload = dict(metadata)
    metadata_payload["byte_length"] = len(volume_bytes)
    temp_suffix = f".{os.getpid()}.tmp"
    metadata_temp = Path(f"{metadata_path}{temp_suffix}")
    data_temp = Path(f"{data_path}{temp_suffix}")
    data_temp.write_bytes(volume_bytes)
    metadata_temp.write_text(
        json.dumps(metadata_payload, ensure_ascii=True, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(str(data_temp), str(data_path))
    os.replace(str(metadata_temp), str(metadata_path))


def purge_scalar_volume_persistent_cache(file_path: str | Path) -> None:
    cache_dir = _scalar_volume_cache_dir(file_path)
    if not cache_dir.exists():
        return
    prefix = _scalar_volume_cache_prefix(file_path)
    for target in cache_dir.glob(f"{prefix}-*"):
        if target.is_file():
            target.unlink(missing_ok=True)


def _load_scalar_volume_texture_cached_or_persisted(
    *,
    file_path: str,
    dataset_path: str | None,
    t_index: int | None,
    channel_index: int | None,
    max_inline_elements: int,
) -> tuple[dict[str, Any], bytes]:
    cached = _read_scalar_volume_texture_from_disk(
        file_path=file_path,
        dataset_path=dataset_path,
        t_index=t_index,
        channel_index=channel_index,
        max_inline_elements=max_inline_elements,
    )
    if cached is not None:
        return cached
    created = _load_scalar_volume_texture_uncached(
        file_path=file_path,
        dataset_path=dataset_path,
        t_index=t_index,
        channel_index=channel_index,
        max_inline_elements=max_inline_elements,
    )
    _write_scalar_volume_texture_to_disk(
        file_path=file_path,
        dataset_path=dataset_path,
        t_index=t_index,
        channel_index=channel_index,
        max_inline_elements=max_inline_elements,
        metadata=created[0],
        volume_bytes=created[1],
    )
    return created


def load_scalar_volume_texture(
    *,
    file_path: str,
    dataset_path: str | None = None,
    t_index: int | None = None,
    channel_index: int | None = None,
    max_inline_elements: int = 1024,
) -> tuple[dict[str, Any], bytes]:
    return get_cached_file_derivative(
        derivative_kind="scalar_volume",
        file_path=file_path,
        factory=lambda: _load_scalar_volume_texture_cached_or_persisted(
            file_path=file_path,
            dataset_path=dataset_path,
            t_index=t_index,
            channel_index=channel_index,
            max_inline_elements=max_inline_elements,
        ),
        dataset_path=str(dataset_path or ""),
        t_index=t_index,
        channel_index=channel_index,
        max_inline_elements=max_inline_elements,
    )


__all__ = [
    "DEFAULT_HISTOGRAM_BINS",
    "VIEWER_TILE_SIZE",
    "ViewAxis",
    "build_hdf5_materials_dashboard",
    "build_hdf5_dataset_histogram",
    "build_hdf5_dataset_summary",
    "build_hdf5_dataset_table_preview",
    "build_hdf5_viewer_manifest",
    "build_tile_levels",
    "build_viewer_manifest",
    "is_hdf5_viewer_path",
    "load_view_plane",
    "load_view_volume",
    "load_view_volume_source",
    "normalize_view_axis",
    "plan_volume_source_atlas",
    "purge_scalar_volume_persistent_cache",
    "render_hdf5_dataset_slice",
    "render_volume_source_atlas_png",
    "render_view_atlas_png",
    "render_view_histogram",
    "render_view_plane_image",
    "render_view_tile_png",
    "load_scalar_volume_texture",
]
