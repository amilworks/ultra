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


def _resolution_level_count(meta: dict[str, Any]) -> int:
    """Return the number of real pyramid levels, preferring format-reported actuals.

    CZI/libbioimage can report a synthetic power-of-two level count alongside the
    levels that are truly present in the file. Viewer tiles must follow the actual
    file levels, matching Fiji/Bio-Formats, so we prefer the ``*_actual`` fields.
    """
    return _int(meta, "image_num_resolution_levels_actual") or _int(meta, "image_num_resolution_levels")


def _resolution_scales(meta: dict[str, Any], levels_n: int) -> list[float]:
    raw = (
        meta.get("image_resolution_level_scales_actual")
        or meta.get("image_res_l_scales_actual")
        or meta.get("image_resolution_level_scales")
        or meta.get("image_res_l_scales", "")
    )
    scales = _parse_scales(raw)
    if levels_n > 0 and len(scales) > levels_n:
        return scales[:levels_n]
    return scales


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
    levels_n = _resolution_level_count(meta)
    scales = _resolution_scales(meta, levels_n)
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


def build_acquisition(meta: dict[str, Any]) -> dict[str, Any]:
    """Curated provenance + instrument facts pulled from the raw libbioimage meta.

    libbioimage already parses the container's embedded metadata — OME-XML for
    OME-TIFF (acquisition mode, objective, detector, experimenter), and TIFF/XMP
    tags for everything else (software, capture date) — but ``build_viewer_info``
    only forwarded a thin slice. This surfaces the scientifically meaningful,
    single-valued fields (per-channel detail like names/colors comes from ``phys``;
    the flat meta dict collapses per-channel/per-plane keys). Only present fields are
    emitted, so a sparse file yields a short dict (the UI hides empty groups).
    """
    def first(*keys: str) -> str | None:
        for k in keys:
            value = meta.get(k)
            if value is None:
                continue
            text = str(value).strip()
            if text and text.lower() not in ("-1", "unknown", "none", "nan"):
                return text
        return None

    acq: dict[str, Any] = {}
    software = first("TIFF/Software", "document/application", "OME/Creator")
    if software:
        acq["software"] = software
    acquired = first("OME/Image/AcquisitionDate", "Xmp/pix4d/AcquisitionDateTimeUTC", "TIFF/DateTime")
    if acquired:
        acq["acquired"] = acquired
    color_space = first("ColorProfile/color_space", "image_mode")
    if color_space:
        acq["color_space"] = color_space
    objective = first("objectives/objective:0/name", "objectives/objective:0/magnification")
    if objective:
        acq["objective"] = objective
    levels = _resolution_level_count(meta)
    if levels > 1:
        acq["pyramid_levels"] = levels
    # OME-XML instrument context (microscopy). Single-valued in the flat dict.
    for key, src in (
        ("acquisition_mode", "OME/Image/Pixels/Channel/AcquisitionMode"),
        ("objective_medium", "OME/Image/ObjectiveSettings/Medium"),
        ("refractive_index", "OME/Image/ObjectiveSettings/RefractiveIndex"),
        ("detector_binning", "OME/Image/Pixels/Channel/DetectorSettings/Binning"),
        ("experimenter", "OME/Experimenter/UserName"),
        ("source_name", "OME/Image/Name"),
    ):
        value = first(src)
        if value:
            acq[key] = value
    return acq


def build_mosaic(meta: dict[str, Any]) -> dict[str, Any] | None:
    """Detect a tiled-mosaic acquisition (a stage scan of overlapping fields-of-view).

    When such a scan is saved WITHOUT stitching, the reader assembles the fields by
    stage position with no blending or flat-field correction, so the displayed image
    shows per-field illumination seams (a checkerboard) — which looks like a rendering
    bug but is the raw data. We surface it (tile count + stitch/overlap context) so the
    viewer can label it instead of silently showing the artifact. Returns ``None`` for a
    normal single-field image. Format-general: matches the metadata-key SUFFIX (the full
    key path is container-specific) — e.g. CZI's ``.../Information/Image/SizeM``,
    ``.../SampleHolder/IsOnlineStitchingEnabled``, ``.../SampleHolder/Overlap``.
    """
    def find_suffix(*suffixes: str) -> Any:
        for key, value in meta.items():
            ks = str(key)
            for suf in suffixes:
                if ks == suf or ks.endswith("/" + suf):
                    return value
        return None

    tiles_raw = find_suffix("SizeM")  # CZI mosaic ("M") dimension = number of fields
    try:
        tiles = int(float(tiles_raw)) if tiles_raw is not None else 0
    except (TypeError, ValueError):
        tiles = 0
    if tiles <= 1:
        return None  # single field, not a mosaic

    mosaic: dict[str, Any] = {"tiles": tiles}
    stitch_raw = find_suffix("IsOnlineStitchingEnabled")
    if stitch_raw is not None:
        mosaic["stitched"] = str(stitch_raw).strip().lower() in ("true", "1")
    overlap_raw = find_suffix("SampleHolder/Overlap")
    try:
        if overlap_raw is not None:
            mosaic["overlap"] = round(float(overlap_raw), 4)
    except (TypeError, ValueError):
        pass
    return mosaic


def build_raw_header(meta: dict[str, Any]) -> dict[str, str]:
    """A bounded, de-noised view of the container's raw tags (TIFF/XMP/OME/...) for
    the collapsible 'Technical details' drawer — so nothing parsed is silently lost,
    while the calm curated groups still lead. Excludes the pixel-grid/tile/per-channel
    keys already shown elsewhere and the bulky per-plane OME entries, trims long
    values, and caps the count so the drawer stays scannable.
    """
    skip_prefix = ("image_num", "image_pixel", "image_res", "tile_", "channels/")
    skip_exact = {"image_dimensions", "image_mode", "format", "raw_endian", "metadata_version"}
    header: dict[str, str] = {}
    for key in sorted(meta.keys()):
        if key in skip_exact or any(key.startswith(p) for p in skip_prefix):
            continue
        if "/Plane" in key or "/TiffData" in key or "/BinData" in key:
            continue  # per-plane OME entries explode the dict
        text = str(meta[key]).strip()
        if not text or len(text) > 160:
            continue
        header[key] = text
        if len(header) >= 80:
            break
    return header


# Spatial-structure score (lag-1 autocorrelation, see engine._channel_signal_scores)
# below which a channel is treated as noise / dead / segmentation-mask and excluded
# from the default. Real imagery scores ~0.9+; pure noise ~0. The gap is enormous
# (measured 0.01-0.03 noise vs 0.93-0.98 signal on real AICS data) so the threshold
# is not delicate.
SIGNAL_NOISE_THRESHOLD = 0.30


def default_visible_channels(
    names: list[str],
    colors: list[dict[str, Any]],
    channel_count: int,
    channel_mode: str,
    signal_scores: list[float] | None = None,
) -> list[int]:
    """Default channels shown for a composite multichannel view.

    A multichannel fluorescence stack should open as a multichannel composite (so
    cells/nuclei are visible), not a single channel that renders as one diffuse
    blob in 3D. Pick ONE channel per distinct LUT color (so duplicate-color
    channels like CMDRP/CMDRP_1 don't pile on), skip brightfield/transmitted-light
    channels (near-white, or named bright/trans/bf/dic/phase), and cap the count to
    keep the render + GPU bounded. Single-channel / non-composite stays [0].

    ``signal_scores`` (per-channel spatial-structure, 0=noise..1=real imagery) lets
    us pick the REAL channel within a color group and drop dead/noise channels:
    files like the AICS OME-TIFFs carry paired channels (CMDRP/CMDRP_1, EGFP/EGFP_1)
    where the ``_1`` variant is pure noise — defaulting to it renders a noise blob in
    both 2D and 3D. With scores we keep the channel that actually contains structure
    instead of just the first index. Without scores, fall back to first-per-color.
    """
    if channel_count <= 1 or channel_mode != "composite":
        return [0]
    max_default = 4
    brightfield_markers = ("bright", "trans", "bf", "dic", "phase", "white")

    def score(i: int) -> float:
        if signal_scores is not None and i < len(signal_scores):
            return float(signal_scores[i])
        return 1.0  # no score available -> assume real

    def is_noise(i: int) -> bool:
        return signal_scores is not None and score(i) < SIGNAL_NOISE_THRESHOLD

    def dominant_color_key(rgb: Any) -> str | None:
        # Group channels by their DOMINANT primary (R/G/B) so near-duplicate LUT
        # colors (e.g. #ff0000 vs #ff0014, both red) collapse to one, while distinct
        # fluorophore colors stay separate. Achromatic (white/gray, hi-lo small) ->
        # None (treated as brightfield-like and skipped).
        if not isinstance(rgb, (list, tuple)) or len(rgb) < 3:
            return None
        r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
        if max(r, g, b) - min(r, g, b) < 38:  # ~15% of 255 -> achromatic
            return None
        if r >= g and r >= b:
            return "r"
        return "g" if g >= b else "b"

    def select(skip_noise: bool) -> list[int]:
        # Best channel per color: highest structure score, then lowest index. When
        # skip_noise, noise channels are excluded outright (so a color with only
        # noise channels is dropped rather than rendered as a noise blob).
        best_by_key: dict[str, int] = {}
        for i in range(channel_count):
            name = str(names[i]).lower() if i < len(names) else ""
            key = dominant_color_key(colors[i].get("rgb") if i < len(colors) else None)
            if key is None or any(k in name for k in brightfield_markers):
                continue
            if skip_noise and is_noise(i):
                continue
            cur = best_by_key.get(key)
            if cur is None or (score(i), -i) > (score(cur), -cur):
                best_by_key[key] = i
        chosen = sorted(best_by_key.values())
        return chosen[:max_default]

    chosen = select(skip_noise=True)
    if not chosen:  # every color was noise-only -> ignore scores rather than show nothing
        chosen = select(skip_noise=False)
    return chosen if chosen else [0]


def build_viewer_info(
    meta: dict[str, Any], signal_scores: list[float] | None = None
) -> dict[str, Any]:
    """Map a libbioimage ``meta`` dict to the viewer-info structure.

    ``signal_scores`` (optional, per-channel spatial-structure 0..1) is supplied by
    the engine for multichannel files so the default channel selection prefers real
    imagery over noise/segmentation channels (see :func:`default_visible_channels`).
    """
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
    visible_channels = default_visible_channels(names, colors, c, channel_mode, signal_scores)
    # Photos use the display (full-colour) render path; scalar science data uses the
    # window/level intensity path. This drives the viewer to a plain zoomable image
    # instead of the composite channel-pills + window/level controls.
    render_policy = "display" if is_photo else "scalar"
    # A z>1 image is a 3D volume, so it earns BOTH 3D surfaces: the orthogonal
    # The 3D surfaces (reslice + volume) are offered ONLY for medical (clinical)
    # volumes, which have the mature scalar MPR/volume path. Non-medical z-stacks
    # (microscopy) are intentionally 2D-only for now: the multichannel 3D
    # render is not yet reliable enough to ship, and a fast 2D view with first-class
    # Z/T scrubbing is the better experience. volume_mode stays "slice_stack" so the
    # 2D Z scrub still knows it is a stack — only the 3D SURFACES are withheld. (To
    # re-enable microscopy 3D, drop the modality guard.)
    volume_surfaces = ["mpr", "volume"] if (is_volume and modality == "medical") else []
    available_surfaces = ["2d", "metadata"] + volume_surfaces

    phys = {
        "x": x, "y": y, "z": z, "t": t, "ch": c,
        "pixel_depth": depth, "pixel_format": pixel_format,
        "pixel_size": [spacing["x"], spacing["y"], spacing["z"], 1.0],
        "pixel_units": ["um", "um", "um", "frame"],
        "channel_names": names, "display_channels": visible_channels,
        "channel_colors": colors, "units": "physical",
    }
    display_defaults = {
        "enhancement": "d", "negative": False, "rotate": 0, "fusion_method": "m",
        "channel_mode": channel_mode, "channels": visible_channels,
        "time_index": 0, "z_index": z // 2, "volume_channel": visible_channels[0],
    }
    metadata = {
        "reader": "libbioimage", "dims_order": dims, "array_dtype": dtype,
        # The REAL container format (e.g. "OME-TIFF"/"BigTIFF"), distinct from the
        # reader. The viewer's Format row should show this, not "libbioimage".
        "format": fmt,
        "physical_spacing": spacing, "scene_count": 1, "warnings": [],
    }
    acquisition = build_acquisition(meta)
    if acquisition:
        metadata["acquisition"] = acquisition
    raw_header = build_raw_header(meta)
    if raw_header:
        metadata["header"] = raw_header
    mosaic = build_mosaic(meta)
    if mosaic:
        metadata["mosaic"] = mosaic
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
