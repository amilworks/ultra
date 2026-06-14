"""Map a libbioimage metadata dict into the viewer-info our frontend consumes.

`bim.meta()` returns a flat dict (see the engine-understanding doc). This module
turns it into the structured fields the Scientific Viewer needs to choose a
render mode: ``axis_sizes``, channels, physical spacing, dtype, and — critically
— a ``tile_scheme`` (built from the pyramid's resolution levels) which is what
makes the viewer use the bounded DeepZoom tile path. Pure functions, unit-tested.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = [
    "build_viewer_info",
    "build_tile_scheme",
    "build_channels",
    "paged_depth",
    "atlas_layout",
    "build_atlas_scheme",
    "ATLAS_CELL_CAP",
]

# Cap on a single atlas cell's largest dimension. The slice_stack 3D volume is
# assembled from a texture atlas (one cell per z-plane); capping the cell bounds
# the atlas image and the resulting Data3DTexture (cell_w*cols x cell_h*rows x z).
# 256 keeps a 100-plane volume well under GPU/canvas limits while staying crisp.
ATLAS_CELL_CAP = 256


def _int(meta: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(meta.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _float(meta: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(meta.get(key, default))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


def _dtype_name(pixel_format: str, depth: int) -> str:
    fmt = (pixel_format or "").lower()
    bits = depth if depth in (8, 16, 32, 64) else 16
    if "float" in fmt:
        return "float64" if bits >= 64 else "float32"
    signed = "signed" in fmt and "unsigned" not in fmt
    return f"{'int' if signed else 'uint'}{bits}"


def _parse_scales(raw: str) -> list[float]:
    out: list[float] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            continue
    return out


def _first_int(raw: Any, default: int) -> int:
    for part in str(raw or "").split(","):
        part = part.strip()
        if part:
            try:
                return int(float(part))
            except ValueError:
                continue
    return default


def _parse_color(raw: Any) -> dict[str, Any] | None:
    parts = [p.strip() for p in str(raw or "").split(",") if p.strip()]
    try:
        vals = [float(p) for p in parts[:3]]
    except ValueError:
        return None
    if not vals:
        return None
    scale = 255.0 if max(vals) <= 1.0 else 1.0
    rgb = [max(0, min(255, int(round(v * scale)))) for v in vals]
    rgb = (rgb + [0, 0, 0])[:3]
    return {"hex": "#%02x%02x%02x" % tuple(rgb), "rgb": rgb}


def paged_depth(meta: dict[str, Any]) -> int:
    """Number of scrubbable planes stored as document *pages*, or 0 if not paged.

    Plain multi-page documents (e.g. a z-stack saved as a paged TIFF, or a PIL/
    ImageJ multipage write) report their planes as ``image_num_p`` while leaving
    ``image_num_z == 1``. We treat those pages as the z depth — but only when no
    other dimension is multi-valued, so a paged file that actually interleaves
    channels/time (pages = z*c, etc.) is left to the engine's N-D ``-slice``.
    """
    z = _int(meta, "image_num_z", 1)
    t = _int(meta, "image_num_t", 1)
    c = _int(meta, "image_num_c", 1)
    pages = _int(meta, "image_num_p", 1)
    if pages > 1 and z <= 1 and t <= 1 and c <= 1:
        return pages
    return 0


def build_tile_scheme(meta: dict[str, Any], tile_size_default: int = 256) -> dict[str, Any] | None:
    """Build the viewer ``tile_scheme`` from a pyramid's resolution levels.

    Returns None for non-pyramidal sources (single level), so the viewer falls
    back to the direct/slice path.
    """
    x = _int(meta, "image_num_x")
    y = _int(meta, "image_num_y")
    levels_n = _int(meta, "image_num_resolution_levels")
    # The real engine reports `image_resolution_level_scales`; older fixtures used
    # the short alias `image_res_l_scales`. Accept either.
    scales = _parse_scales(meta.get("image_resolution_level_scales") or meta.get("image_res_l_scales", ""))
    if x <= 0 or y <= 0 or levels_n <= 1 or len(scales) <= 1:
        return None
    tile_size = _first_int(meta.get("tile_size_x"), tile_size_default)
    levels: list[dict[str, Any]] = []
    for i, scale in enumerate(scales):
        if scale <= 0:
            continue
        w = max(1, round(x * scale))
        h = max(1, round(y * scale))
        levels.append({
            "level": i,
            "width": w,
            "height": h,
            "columns": (w + tile_size - 1) // tile_size,
            "rows": (h + tile_size - 1) // tile_size,
            "downsample": max(1, round(1.0 / scale)),
        })
    if len(levels) <= 1:
        return None
    return {"tile_size": tile_size, "format": "png", "levels": levels}


def atlas_layout(width: int, height: int, depth: int, *, cell_cap: int = ATLAS_CELL_CAP) -> dict[str, int]:
    """Compute the texture-atlas grid layout for a z-stack volume.

    The atlas packs ``depth`` z-planes into a ``columns x rows`` grid of
    ``cell_w x cell_h`` cells. ``columns = ceil(sqrt(depth))`` (a near-square grid)
    and the cell is downsampled so its largest side is <= ``cell_cap`` (bounding
    the atlas image and the GPU volume texture). The engine and the viewer-info
    ``atlas_scheme`` MUST agree on this layout, so both call this one function —
    the frontend decodes the atlas using exactly these cell/grid dimensions.
    """
    w = max(1, int(width))
    h = max(1, int(height))
    d = max(1, int(depth))
    downsample = max(1, math.ceil(max(w, h) / max(1, cell_cap)))
    cell_w = max(1, round(w / downsample))
    cell_h = max(1, round(h / downsample))
    columns = max(1, math.ceil(math.sqrt(d)))
    rows = max(1, math.ceil(d / columns))
    return {
        "downsample": downsample,
        "cell_w": cell_w,
        "cell_h": cell_h,
        "columns": columns,
        "rows": rows,
        "slice_count": d,
    }


def build_atlas_scheme(
    meta: dict[str, Any], *, depth: int | None = None, cell_cap: int = ATLAS_CELL_CAP
) -> dict[str, Any] | None:
    """Build the viewer ``atlas_scheme`` for a slice-stack (z-stack) volume.

    Returns ``None`` for non-volumes (z<=1). ``depth`` overrides the z count
    (callers pass the already-resolved depth, which folds paged z-stacks). The
    returned dict is the contract the frontend's ``atlasToVolumeTexture`` reads:
    cell size, grid columns/rows, and the full atlas image dimensions.
    """
    x = _int(meta, "image_num_x")
    y = _int(meta, "image_num_y")
    z = int(depth) if depth is not None else (paged_depth(meta) or _int(meta, "image_num_z", 1))
    if x <= 0 or y <= 0 or z <= 1:
        return None
    lay = atlas_layout(x, y, z, cell_cap=cell_cap)
    return {
        "slice_count": lay["slice_count"],
        "columns": lay["columns"],
        "rows": lay["rows"],
        "slice_width": lay["cell_w"],
        "slice_height": lay["cell_h"],
        "atlas_width": lay["cell_w"] * lay["columns"],
        "atlas_height": lay["cell_h"] * lay["rows"],
        "downsample": lay["downsample"],
        "format": "png",
    }


def _convention_color_dict(index: int) -> dict[str, Any]:
    from ultra_deepagents.imaging import fusion

    rgb01 = fusion.convention_channel_color(index)
    rgb = [max(0, min(255, int(round(component * 255)))) for component in rgb01]
    return {"hex": "#%02x%02x%02x" % tuple(rgb), "rgb": rgb}


def build_channels(meta: dict[str, Any], channel_count: int) -> tuple[list[str], list[dict[str, Any]]]:
    names: list[str] = []
    colors: list[dict[str, Any]] = []
    for i in range(max(channel_count, 0)):
        name = meta.get(f"channels/channel:{i}/name")
        names.append(str(name) if name else f"Channel {i}")
        parsed = _parse_color(meta.get(f"channels/channel:{i}/color"))
        if parsed is None:
            # A multi-channel source with no LUT color is fluorescence: assign a
            # convention color (DAPI->blue, then green/red/magenta/...) so the
            # composite isn't a stack of identical white channels. Single-channel
            # stays white (grayscale).
            parsed = _convention_color_dict(i) if channel_count > 1 else {"hex": "#ffffff", "rgb": [255, 255, 255]}
        colors.append({"index": i, **parsed})
    return names, colors


def build_viewer_info(meta: dict[str, Any]) -> dict[str, Any]:
    """Map a libbioimage ``meta`` dict to the viewer-info structure."""
    x = _int(meta, "image_num_x")
    y = _int(meta, "image_num_y")
    # Fail closed on degenerate geometry. libbioimage will sometimes OPEN a malformed
    # or unsupported-but-recognized container yet report a 0-sized canvas. Without this
    # guard build_viewer_info would emit a 0x0 viewer that the frontend renders as a
    # blank canvas with no error (a C2 "blank canvas" failure). Raise instead so the
    # service maps it to a clean 422 "preview unavailable" — service.py keys the 422 on
    # the "cannot decode" marker, and monitoring won't count an undecodable file as a 500.
    if x <= 0 or y <= 0:
        raise ValueError(f"image reports no pixel geometry ({x}x{y}); cannot decode")
    z = _int(meta, "image_num_z", 1)
    c = _int(meta, "image_num_c", 1)
    t = _int(meta, "image_num_t", 1)
    # A paged z-stack (plain multi-page TIFF, etc.) reports planes as pages with
    # image_num_z=1 — surface them as the scrubbable z depth.
    pages_as_depth = paged_depth(meta)
    if pages_as_depth:
        z = pages_as_depth
    depth = _int(meta, "image_pixel_depth", 8)
    pixel_format_raw = str(meta.get("image_pixel_format", "unsigned integer"))
    pf_lower = pixel_format_raw.lower()
    if "float" in pf_lower:
        pixel_format = "f"
    elif "unsigned" in pf_lower:
        pixel_format = "u"
    elif "signed" in pf_lower:
        pixel_format = "s"
    else:
        pixel_format = "u"
    fmt = str(meta.get("format", ""))
    names, colors = build_channels(meta, c)
    spacing = {
        "x": _float(meta, "pixel_resolution_x", 1.0),
        "y": _float(meta, "pixel_resolution_y", 1.0),
        "z": _float(meta, "pixel_resolution_z", 1.0),
    }
    objective = meta.get("objectives/objective:0/name") or meta.get("objectives/objective:0/magnification")
    fmt_lower = fmt.lower()
    # An RGB(A) photo (incl. geospatial orthomosaics) is color data, NOT fluorescence
    # microscopy: classify by colorspace/channel-color like BisQue, not by a hardcoded
    # 3-channel assumption. Recognize an 8-bit unsigned image whose colorspace is RGB/
    # RGBA OR whose first three channels are named Red/Green/Blue (a 4th Alpha band is
    # transparency, not a science channel). Real fluorescence (>8-bit, >4 channels, or
    # non-RGB names like DAPI/FITC) is excluded and stays microscopy/composite.
    color_mode = str(meta.get("image_mode") or meta.get("ColorProfile/color_space") or "").strip().lower()
    rgb_named = [n.strip().lower() for n in names[:3]] == ["red", "green", "blue"]
    photo_like = (
        pixel_format == "u" and depth <= 8 and c in (3, 4)
        and (color_mode.startswith("rgb") or rgb_named)
    )
    microscopy_format = any(
        k in fmt_lower for k in ("czi", "fluoview", "lsm", "nd2", "ome", "lif", "oib", "scn", "svs", "ndpi", "vsi")
    )
    # A 2D RGB(A) photo. A z>1 RGB stack is left to the microscopy/volume path so a
    # genuine stack is never flattened onto the single-channel display surface — keep
    # modality/channel_mode/render_policy consistent by gating all three on is_photo.
    is_photo = photo_like and z <= 1
    if "dicom" in fmt_lower or "nifti" in fmt_lower:
        modality = "medical"
    elif is_photo:
        modality = "image"
    elif z > 1 or microscopy_format or (c > 1 and not is_photo):
        modality = "microscopy"
    else:
        modality = "image"
    dims = str(meta.get("image_dimensions", "XYCZT"))
    dtype = _dtype_name(pixel_format_raw, depth)
    tile_scheme = build_tile_scheme(meta)
    is_volume = z > 1
    has_tiles = tile_scheme is not None
    backend_mode = "pyramid" if has_tiles else "direct"
    # Medical volumes (DICOM/NIfTI) render as 3D scalar fields; microscopy
    # z-stacks scrub plane-by-plane (bandwidth-friendly) rather than downloading
    # the whole volume up front.
    if not is_volume:
        volume_mode = "none"
    elif modality == "medical":
        volume_mode = "scalar"
    else:
        volume_mode = "slice_stack"
    delivery_mode = "deferred_multiscale" if has_tiles else "direct"
    # An RGB(A) photo renders its native colors directly (no per-channel LUT fuse),
    # so it is a single display surface, not a composite of science channels.
    channel_mode = "composite" if (c > 1 and not is_photo) else "single"
    # Photos use the display (full-colour) render path; scalar science data uses the
    # window/level intensity path. This drives the viewer to a plain zoomable image
    # instead of the composite channel-pills + window/level controls.
    render_policy = "display" if is_photo else "scalar"
    available_surfaces = ["2d", "metadata"] + (["mpr"] if is_volume else [])

    phys = {
        "x": x, "y": y, "z": z, "t": t, "ch": c,
        "pixel_depth": depth, "pixel_format": pixel_format,
        "pixel_size": [spacing["x"], spacing["y"], spacing["z"], 1.0],
        "pixel_units": ["um", "um", "um", "frame"],
        "channel_names": names, "display_channels": [0],
        "channel_colors": colors, "units": "physical",
    }
    display_defaults = {
        "enhancement": "d", "negative": False, "rotate": 0, "fusion_method": "m",
        "channel_mode": channel_mode, "channels": [0],
        "time_index": 0, "z_index": z // 2, "volume_channel": 0,
    }
    metadata = {
        "reader": "libbioimage", "dims_order": dims, "array_dtype": dtype,
        "physical_spacing": spacing, "scene_count": 1, "warnings": [],
    }
    if names and modality == "microscopy":
        metadata["microscopy"] = {
            "channel_names": names, "objective": str(objective) if objective is not None else None,
            "dimensions_present": dims,
        }
    viewer = {
        "status": "ready", "warmup_mode": "lazy", "backend_mode": backend_mode,
        "default_surface": "2d", "available_surfaces": available_surfaces,
        "default_axis": "z", "slice_axes": ["z"], "channel_mode": channel_mode,
        "volume_mode": volume_mode, "render_policy": render_policy, "delivery_mode": delivery_mode,
        "first_paint_mode": "webgl", "texture_policy": "linear",
        "asset_preparation": {
            "status": "ready", "native_supported": True,
            "tile_pyramid": "ready" if has_tiles else "none",
            "volume_representation": volume_mode,
        },
    }
    if has_tiles:
        viewer["tile_scheme"] = tile_scheme
    # A slice_stack volume (microscopy z-stack) renders in 3D from a texture
    # atlas. Emit the authoritative grid/cell layout the engine assembles to, so
    # the frontend decodes the atlas with the exact same cell size and columns
    # (its native-size fallback would mis-tile a downsampled atlas).
    if volume_mode == "slice_stack":
        atlas_scheme = build_atlas_scheme(meta, depth=z)
        if atlas_scheme is not None:
            viewer["atlas_scheme"] = atlas_scheme

    return {
        "kind": "image",
        "modality": modality,
        "backend_mode": backend_mode,
        "dims_order": dims,
        "axis_sizes": {"T": t, "C": c, "Z": z, "Y": y, "X": x},
        "selected_indices": {"T": 0, "C": 0, "Z": z // 2},
        "is_volume": is_volume,
        "is_timeseries": t > 1,
        "is_multichannel": c > 1,
        "phys": phys,
        "display_defaults": display_defaults,
        "metadata": metadata,
        "viewer": viewer,
        # top-level conveniences for tests + the Go proxy
        "dtype": dtype,
        "pixel_depth": depth,
        "pixel_format": pixel_format,
        "physical_spacing": spacing,
        "channel_names": names,
        "channel_colors": colors,
        "objective": str(objective) if objective is not None else None,
        "format": fmt,
        "reader": "libbioimage",
        "tile_scheme": tile_scheme,
    }
