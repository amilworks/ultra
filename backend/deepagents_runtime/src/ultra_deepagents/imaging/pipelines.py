"""Builders for libbioimage read-pipeline strings.

A *pipeline* is the space-separated CLI-token string passed to
``libbioimage.read(filename, pipeline=...)`` (and to the ``imgcnv`` CLI). These
helpers encode the subset of the grammar we use to serve the V2 image endpoints,
so the grammar lives in one tested place instead of being string-built ad hoc at
call sites.

Grammar reference (verified against the engine source, v3.23):

- ``-tile SZ,XID,YID,L``          embedded tile: size, x-index, y-index, level
- ``-tile-roi X1,Y1,X2,Y2,SCALE`` arbitrary region at a scale
- ``-res-level L``                power-of-two pyramid level (0=native, 1=1/2, ...)
- ``-scale S``                    nearest level to scale S (overrides -res-level)
- ``-slice z:N,t:M``              N-D plane selection (z,t,serie,fov,...)
- ``-depth BITS,MODE,TYPE``       output depth, e.g. ``8,D,U`` display / ``32,D,F`` float
- ``-resize W,H,METHOD[,MX]``     resize; ``MX`` = bounding box, keep AR, no upsample
- ``-remap C1,C2,...``            channel selection/reorder (1-based; 0=empty)
- ``-textureatlas`` / ``-texturegrid R,C``   z-stack -> 2D grid

Channel *selection* is done with ``-remap`` (1-based), not ``-slice c:``.

All builders are pure functions returning strings; they validate arguments and
raise :class:`ValueError` on bad input. Compose with :func:`build`.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "DEPTH_DISPLAY_8U",
    "DEPTH_DISPLAY_8U_FULLRANGE",
    "DEPTH_SCALAR_F32",
    "build",
    "depth",
    "resize",
    "remap",
    "res_level",
    "scale",
    "nd_slice",
    "tile",
    "region",
    "slice_plane",
    "page_plane",
    "thumbnail",
    "atlas",
    "scalar_volume_plane",
]

# Common -depth presets.
#: 8-bit unsigned, linear data-range mapping — display tiles for SCALAR/scientific
#: data (>8-bit, float): maps the actual value range into the 8-bit display.
DEPTH_DISPLAY_8U = "8,D,U"
#: 8-bit unsigned, FULL type-range (identity for 8-bit) — display tiles for an 8-bit
#: RGB(A) PHOTO (orthomosaic, slide). Two reasons over data-range ("D"): (1) data-range
#: normalization collapses a CONSTANT channel to 0, so the alpha of a fully-opaque
#: native tile becomes 0 and the tile renders fully transparent (blank at max zoom);
#: (2) an 8-bit photo is already display-ready, so a per-tile contrast stretch is both
#: unwanted (alters true colors) and inconsistent across tiles.
DEPTH_DISPLAY_8U_FULLRANGE = "8,F,U"
#: 32-bit float, data-range — raw scalar values for client-side windowing/MPR.
DEPTH_SCALAR_F32 = "32,D,F"

_DEPTH_MODES = frozenset({"D", "F", "T", "E", "C", "N", "G", "L", "f1", "f2", "f3", "f4"})
_DEPTH_TYPES = frozenset({"U", "S", "F"})
_RESIZE_METHODS = frozenset(
    {
        "NN", "BL", "BC", "lanczos", "bessel", "point", "box", "hermite",
        "hanning", "hamming", "blackman", "gaussian", "quadratic", "catrom",
        "mitchell", "sinc", "blackmanbessel", "blackmansinc",
    }
)
_RESIZE_CONSTRAINTS = frozenset({"AR", "MX", "NOUP"})
# N-D dimensions accepted by ``-slice`` (channel selection uses ``-remap``).
_SLICE_DIMS = (
    "z", "t", "serie", "fov", "rotation", "scene",
    "illumination", "phase", "view", "item", "spectrum", "measure",
)


def build(*parts: str | None) -> str:
    """Join pipeline fragments into one string, dropping empties."""
    return " ".join(p for p in parts if p)


def _check_nonneg_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an int, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _check_positive_int(value: int, name: str) -> int:
    _check_nonneg_int(value, name)
    if value == 0:
        raise ValueError(f"{name} must be > 0")
    return value


def depth(bits: int = 8, mode: str = "D", type_: str = "U") -> str:
    """Build a ``-depth`` token. Defaults to 8-bit unsigned data-range display."""
    if bits not in (8, 16, 32, 64):
        raise ValueError(f"depth bits must be one of 8/16/32/64, got {bits}")
    if mode not in _DEPTH_MODES:
        raise ValueError(f"unknown depth mode {mode!r}")
    if type_ not in _DEPTH_TYPES:
        raise ValueError(f"unknown depth type {type_!r}")
    return f"-depth {bits},{mode},{type_}"


def remap(channels: Sequence[int]) -> str:
    """Build a ``-remap`` token (1-based channel indices; 0 means an empty channel)."""
    chans = list(channels)
    if not chans:
        raise ValueError("remap requires at least one channel")
    for c in chans:
        _check_nonneg_int(c, "channel")
    return "-remap " + ",".join(str(c) for c in chans)


def res_level(level: int) -> str:
    """Build a ``-res-level`` token (0=native, 1=1/2, 2=1/4, ...)."""
    _check_nonneg_int(level, "level")
    return f"-res-level {level}"


def scale(factor: float) -> str:
    """Build a ``-scale`` token; ``factor`` in (0, 1]."""
    if not (0 < factor <= 1):
        raise ValueError(f"scale must be in (0, 1], got {factor}")
    return f"-scale {factor:g}"


def resize(width: int, height: int = 0, method: str = "BC", constraint: str | None = None) -> str:
    """Build a ``-resize`` token. ``height=0`` preserves aspect ratio.

    ``constraint`` may be ``MX``/``NOUP`` (bounding box, keep AR, no upsample) or
    ``AR`` (keep AR). Use ``MX`` for thumbnails so small images are not upsampled.
    """
    _check_nonneg_int(width, "width")
    _check_nonneg_int(height, "height")
    if width == 0 and height == 0:
        raise ValueError("resize requires width or height")
    if method not in _RESIZE_METHODS:
        raise ValueError(f"unknown resize method {method!r}")
    token = f"-resize {width},{height},{method}"
    if constraint is not None:
        if constraint not in _RESIZE_CONSTRAINTS:
            raise ValueError(f"unknown resize constraint {constraint!r}")
        token += f",{constraint}"
    return token


def nd_slice(**dims: int) -> str:
    """Build a ``-slice`` token from N-D dimension indices, e.g. ``nd_slice(z=5, t=0)``.

    Only spatial/temporal dims are accepted; channel selection uses :func:`remap`.
    """
    if not dims:
        raise ValueError("nd_slice requires at least one dimension")
    parts: list[str] = []
    for name in _SLICE_DIMS:  # canonical order, deterministic output
        if name in dims and dims[name] is not None:
            _check_nonneg_int(dims[name], name)
            parts.append(f"{name}:{dims[name]}")
    unknown = set(dims) - set(_SLICE_DIMS)
    if unknown:
        raise ValueError(f"unknown slice dim(s): {sorted(unknown)} (use remap for channels)")
    if not parts:
        raise ValueError("nd_slice requires at least one non-None dimension")
    return "-slice " + ",".join(parts)


# --- High-level per-endpoint builders -------------------------------------------------

def tile(
    size: int,
    col: int,
    row: int,
    level: int = 0,
    *,
    out_depth: str = DEPTH_DISPLAY_8U,
    channels: Sequence[int] | None = None,
) -> str:
    """Pipeline for a single embedded tile (``/tiles`` endpoint).

    ``size`` px square tile at grid (``col`` = XID, ``row`` = YID) and pyramid
    ``level`` (0=native). Bounded read — never decodes the whole image.
    """
    _check_positive_int(size, "size")
    _check_nonneg_int(col, "col")
    _check_nonneg_int(row, "row")
    _check_nonneg_int(level, "level")
    return build(
        f"-tile {size},{col},{row},{level}",
        remap(channels) if channels else None,
        _depth_token(out_depth),
    )


def region(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    region_scale: float | None = None,
    out_depth: str = DEPTH_DISPLAY_8U,
    channels: Sequence[int] | None = None,
) -> str:
    """Pipeline for an arbitrary region of interest (``-tile-roi``).

    Reads only the sub-region (compositing overlapping source tiles), so it is a
    bounded read for tiled/pyramidal sources.
    """
    for v, n in ((x1, "x1"), (y1, "y1"), (x2, "x2"), (y2, "y2")):
        _check_nonneg_int(v, n)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("region requires x2>x1 and y2>y1")
    roi = f"-tile-roi {x1},{y1},{x2},{y2}"
    if region_scale is not None:
        if not (0 < region_scale <= 1):
            raise ValueError(f"region_scale must be in (0, 1], got {region_scale}")
        roi += f",{region_scale:g}"
    return build(
        roi,
        remap(channels) if channels else None,
        _depth_token(out_depth),
    )


def slice_plane(
    *,
    z: int | None = None,
    t: int | None = None,
    level: int | None = None,
    plane_scale: float | None = None,
    out_depth: str = DEPTH_DISPLAY_8U,
    channels: Sequence[int] | None = None,
    max_dim: int | None = None,
) -> str:
    """Pipeline for a 2D plane (``/slice`` endpoint): pick z/t, optional downscale.

    ``max_dim`` bounds the long edge (keep AR, no upsample) — used for transient
    z-scrub frames so a huge plane without a pyramid still returns a small, fast
    image (``-res-level`` alone is a no-op on a non-pyramidal source).
    """
    dims: dict[str, int] = {}
    if z is not None:
        dims["z"] = z
    if t is not None:
        dims["t"] = t
    return build(
        nd_slice(**dims) if dims else None,
        res_level(level) if level is not None else None,
        scale(plane_scale) if plane_scale is not None else None,
        remap(channels) if channels else None,
        resize(max_dim, max_dim, "BC", "MX") if max_dim else None,
        _depth_token(out_depth),
    )


def page_plane(
    page: int,
    *,
    level: int | None = None,
    plane_scale: float | None = None,
    out_depth: str = DEPTH_DISPLAY_8U,
    channels: Sequence[int] | None = None,
    max_dim: int | None = None,
) -> str:
    """Pipeline for one *page* of a multi-page document (``-page N``).

    Plain multi-page TIFFs (and similar paged formats) store their planes as
    document pages — reported as ``image_num_p`` with ``image_num_z == 1`` — which
    ``-slice z:N`` cannot address (it returns the first plane every time). This
    reads page ``N`` directly so a paged z-stack scrubs. Mirrors
    :func:`slice_plane`'s level/scale/depth/``max_dim`` handling.
    """
    _check_nonneg_int(page, "page")
    return build(
        f"-page {page}",
        res_level(level) if level is not None else None,
        scale(plane_scale) if plane_scale is not None else None,
        remap(channels) if channels else None,
        resize(max_dim, max_dim, "BC", "MX") if max_dim else None,
        _depth_token(out_depth),
    )


def thumbnail(
    max_size: int,
    *,
    z: int | None = None,
    level: int | None = None,
    out_depth: str = DEPTH_DISPLAY_8U,
    channels: Sequence[int] | None = None,
) -> str:
    """Pipeline for a thumbnail (or one z-scrub frame when ``z`` is given).

    Reads a downsampled level then bounds it to ``max_size`` keeping aspect ratio
    and never upsampling — the live z-scrub frame for ``GET .../slice?...&z=N``.
    """
    _check_positive_int(max_size, "max_size")
    dims = {"z": z} if z is not None else {}
    return build(
        nd_slice(**dims) if dims else None,
        res_level(level) if level is not None else None,
        remap(channels) if channels else None,
        resize(max_size, max_size, "BC", "MX"),
        _depth_token(out_depth),
    )


def atlas(
    *,
    grid: tuple[int, int] | None = None,
    level: int | None = None,
    atlas_scale: float | None = None,
    out_depth: str = DEPTH_DISPLAY_8U,
    channels: Sequence[int] | None = None,
) -> str:
    """Pipeline for a texture atlas (``/atlas`` endpoint): z-stack -> 2D grid.

    Apply ``level``/``atlas_scale`` so only downsampled planes are composed
    (bounded memory). ``grid`` forces rows,cols; otherwise the engine auto-fits.
    ``channels`` (with a float ``out_depth``) selects the channels to read for
    multi-channel fluorescence fusion.
    """
    if grid is not None:
        rows, cols = grid
        _check_positive_int(rows, "rows")
        _check_positive_int(cols, "cols")
        atlas_token = f"-texturegrid {rows},{cols}"
    else:
        atlas_token = "-textureatlas"
    return build(
        res_level(level) if level is not None else None,
        scale(atlas_scale) if atlas_scale is not None else None,
        atlas_token,
        remap(channels) if channels else None,
        _depth_token(out_depth),
    )


def scalar_volume_plane(z: int, *, t: int | None = None, channel: int | None = None) -> str:
    """Pipeline for one float plane of a scalar volume (``/scalar-volume`` assembly).

    Returns raw float values (``-depth 32,D,F``) for client-side windowing/MPR.
    Channel is selected via ``-remap`` (1-based).
    """
    dims: dict[str, int] = {"z": z}
    if t is not None:
        dims["t"] = t
    return build(
        nd_slice(**dims),
        remap([channel + 1]) if channel is not None else None,
        _depth_token(DEPTH_SCALAR_F32),
    )


def _depth_token(out_depth: str) -> str:
    """Accept either a preset string like ``"8,D,U"`` or a full ``-depth`` token."""
    if not out_depth:
        raise ValueError("out_depth must be non-empty")
    return out_depth if out_depth.startswith("-depth") else f"-depth {out_depth}"
