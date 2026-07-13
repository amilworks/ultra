"""Build the Ultra UploadViewerInfo structure from an OME-Zarr image.

Mirrors ``imaging.viewerinfo.build_viewer_info`` so the Lens viewer + the control-plane
proxy treat a natively-served OME-Zarr identically to a libbioimage image. Time/channel
axes light up the existing time-scrubber + multichannel controls; ``backend_mode:"direct"``
routes pixel reads through ``/slice`` (per t/z), served natively by the ngff-service.
"""

from __future__ import annotations

import os
from typing import Any

from ultra_deepagents.ngff.reader import NgffImage

__all__ = ["build_ngff_viewer_info", "build_ngff_tile_scheme"]

# Above this long-edge (level 0), advertise a multiscale tile_scheme so Lens uses the bounded
# DeepZoom /tile path instead of reading a whole (gigapixel) plane via /slice. Smaller images
# (e.g. a 1848 px time-lapse) stay on the simpler, proven direct/slice path.
_TILE_THRESHOLD = int(os.environ.get("ULTRA_NGFF_TILE_THRESHOLD", "2048"))
_TILE_SIZE = int(os.environ.get("ULTRA_NGFF_TILE_SIZE", "256"))


def _format_quantity(value: float, unit: str) -> str:
    """Human label for a physical quantity, e.g. (1, "hour") -> "1 hour", (60, "hour") ->
    "60 hours". Pluralizes the unit naively (good enough for hour/second/minute/micrometer)."""
    num = f"{value:g}"
    if not unit:
        return num
    label = unit if value == 1 else (unit if unit.endswith("s") else f"{unit}s")
    return f"{num} {label}"


def _visible_channels(img: NgffImage) -> list[int]:
    active = [i for i, ch in enumerate(img.channels[: img.num_c]) if ch.active]
    return active or list(range(max(1, img.num_c)))


def build_ngff_tile_scheme(img: NgffImage) -> dict[str, Any] | None:
    """Build the viewer tile_scheme from the OME-Zarr multiscale levels — one entry per
    level (level index maps 1:1 to the /tile ``level`` param). Large single-level images
    still use bounded tiles; only small images use the direct full-slice path."""
    base_h, base_w = img.level_yx(0)
    if max(base_h, base_w) <= _TILE_THRESHOLD:
        return None
    levels: list[dict[str, Any]] = []
    for i in range(len(img.levels)):
        h, w = img.level_yx(i)
        if w <= 0 or h <= 0:
            continue
        levels.append(
            {
                "level": i,
                "width": w,
                "height": h,
                "columns": (w + _TILE_SIZE - 1) // _TILE_SIZE,
                "rows": (h + _TILE_SIZE - 1) // _TILE_SIZE,
                "downsample": max(1, round(base_w / w)),
            }
        )
    if not levels:
        return None
    return {"tile_size": _TILE_SIZE, "format": "png", "levels": levels}


def build_ngff_viewer_info(img: NgffImage) -> dict[str, Any]:
    t, c, z, y, x = img.num_t, img.num_c, img.num_z, img.num_y, img.num_x
    if x <= 0 or y <= 0:
        raise ValueError(f"OME-Zarr reports no pixel geometry ({x}x{y}); cannot decode")

    names = [ch.label for ch in img.channels[:c]] or ["Channel"]
    colors = ["#" + ch.color for ch in img.channels[:c]] or ["#FFFFFF"]
    visible = _visible_channels(img)
    tile_scheme = build_ngff_tile_scheme(img)
    has_tiles = tile_scheme is not None
    is_volume = z > 1
    is_multichannel = c > 1
    channel_mode = "composite" if is_multichannel else "single"
    modality = "microscopy"
    dims_order = "".join(ax.upper() for ax in img.axes) or "TCYX"
    dtype = str(img.dtype)
    spacing = {
        "x": img.physical.get("x") or 1.0,
        "y": img.physical.get("y") or 1.0,
        "z": img.physical.get("z") or 1.0,
    }
    # Microscopy z-stacks scrub plane-by-plane (slice_stack); a 2D/time image has no volume.
    volume_mode = "slice_stack" if is_volume else "none"

    phys = {
        "x": x,
        "y": y,
        "z": z,
        "t": t,
        "ch": c,
        "pixel_depth": img.dtype.itemsize * 8,
        "pixel_format": "f" if img.dtype.kind == "f" else ("s" if img.dtype.kind == "i" else "u"),
        "pixel_size": [spacing["x"], spacing["y"], spacing["z"], 1.0],
        "pixel_units": [
            img.units.get("x") or "pixel",
            img.units.get("y") or "pixel",
            img.units.get("z") or "pixel",
            img.units.get("t") or "frame",
        ],
        "channel_names": names,
        "display_channels": visible,
        "channel_colors": colors,
        "units": "physical",
    }
    display_defaults = {
        "enhancement": "d",
        "negative": False,
        "rotate": 0,
        "fusion_method": "m",
        "channel_mode": channel_mode,
        "channels": visible,
        "time_index": 0,
        "z_index": z // 2,
        "volume_channel": visible[0],
    }
    backend_mode = "pyramid" if has_tiles else "direct"
    delivery_mode = "deferred_multiscale" if has_tiles else "direct"
    fmt = f"OME-Zarr (NGFF {img.version})" if img.version else "OME-Zarr"
    metadata: dict[str, Any] = {
        "reader": "ngff",
        "dims_order": dims_order,
        # The real stored level-0 shape (matches dims_order), so the Metadata tab shows the
        # true array shape instead of a derived TCZYX guess.
        "array_shape": [int(s) for s in img.levels[0].shape],
        "array_dtype": dtype,
        "format": fmt,
        "physical_spacing": spacing,
        "scene_count": 1,
        "multiscale_levels": len(img.levels),
        # Effective diagonal array-to-physical transforms. Dataset transforms are
        # composed first, followed by multiscale-level transforms, exactly as NGFF
        # v0.4/v0.5 define. Units align positionally with every scale/translation.
        "ngff_coordinate_transforms": {
            "semantics": "effective-array-to-physical",
            "selected_multiscale_index": img.multiscale_index,
            "selected_multiscale_name": img.multiscale_name,
            "axes": [
                {"name": axis, "unit": unit}
                for axis, unit in zip(img.axes, img.levels[0].axis_units, strict=True)
            ],
            "levels": [
                {
                    "path": level.path,
                    "scale": list(level.scale),
                    "translation": list(level.translation),
                }
                for level in img.levels
            ],
        },
        "warnings": [],
    }
    # Exact over the complete smallest level or explicitly unknown.  Never label a sampled
    # plane as the store's true range; array_min/array_max are omitted when the scan budget
    # prevents an exact result.
    intensity = img.intensity_range()
    intensity_scope = (
        "complete_array"
        if len(img.levels) == 1
        else "complete_smallest_multiscale_level"
    )
    intensity_record: dict[str, Any] = {
        "status": img.intensity_range_status,
        "scope": intensity_scope,
        "level": len(img.levels) - 1,
    }
    metadata["intensity_range"] = intensity_record
    if intensity is not None:
        intensity_record["minimum"], intensity_record["maximum"] = intensity
        if len(img.levels) == 1:
            # Only the single-level case is also an exact range over the complete image
            # array, so the generic viewer's unqualified "Value range" label is truthful.
            metadata["array_min"], metadata["array_max"] = intensity
        else:
            metadata["warnings"].append(
                "unqualified value range omitted: exact statistics cover only the "
                "complete smallest multiscale level"
            )
    else:
        metadata["warnings"].append(
            f"intensity range omitted: {img.intensity_range_status}"
        )
    # Acquisition / provenance the NGFF store carries: the multiscale pyramid depth and the
    # source image name (multiscales.name / omero.name, e.g. a well/sample id).
    acquisition: dict[str, Any] = {"pyramid_levels": len(img.levels)}
    if img.name:
        acquisition["source_name"] = img.name
    metadata["acquisition"] = acquisition
    if modality == "microscopy" and names:
        microscopy: dict[str, Any] = {"channel_names": names, "dimensions_present": dims_order}
        # Time-lapse cadence from the NGFF time axis (scale + unit), e.g. "1 hour" every frame
        # over "60 hours". Lets the Dimensions group show the temporal sampling.
        increment = float(img.physical.get("t") or 0.0)
        time_unit = img.units.get("t") or ""
        if t > 1 and increment > 0:
            microscopy["timelapse_interval"] = _format_quantity(increment, time_unit)
            microscopy["total_time_duration"] = _format_quantity(increment * (t - 1), time_unit)
        metadata["microscopy"] = microscopy

    viewer: dict[str, Any] = {
        "status": "ready",
        "warmup_mode": "lazy",
        "backend_mode": backend_mode,
        "default_surface": "2d",
        "available_surfaces": ["2d", "metadata"],
        "default_axis": "z",
        "slice_axes": ["z"],
        "channel_mode": channel_mode,
        "volume_mode": volume_mode,
        "render_policy": "scalar",
        "delivery_mode": delivery_mode,
        "first_paint_mode": "webgl",
        "texture_policy": "linear",
        "asset_preparation": {
            "status": "ready",
            "native_supported": True,
            "tile_pyramid": "ready" if has_tiles else "none",
            "volume_representation": volume_mode,
        },
    }
    if has_tiles:
        viewer["tile_scheme"] = tile_scheme

    return {
        "kind": "image",
        "modality": modality,
        "backend_mode": backend_mode,
        "dims_order": dims_order,
        "axis_sizes": {"T": t, "C": c, "Z": z, "Y": y, "X": x},
        "selected_indices": {"T": 0, "C": 0, "Z": z // 2},
        "is_volume": is_volume,
        "is_timeseries": t > 1,
        "is_multichannel": is_multichannel,
        "phys": phys,
        "display_defaults": display_defaults,
        "metadata": metadata,
        "viewer": viewer,
        # top-level conveniences (mirror libbioimage build_viewer_info)
        "dtype": dtype,
        "pixel_depth": phys["pixel_depth"],
        "pixel_format": phys["pixel_format"],
        "physical_spacing": spacing,
        "channel_names": names,
        "channel_colors": colors,
        "format": "OME-Zarr",
        "reader": "ngff",
        "tile_scheme": tile_scheme,
    }
