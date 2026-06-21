"""OME-NGFF rendering: window + composite channels into a display PNG.

Honors the OME `omero` block (per-channel color + window) when present; falls back to
robust auto-contrast (1st–99th percentile) when absent. A single white/grayscale channel
renders as 8-bit grayscale; multiple (or colored) channels composite additively, the way
napari/vizarr render OME-Zarr.
"""

from __future__ import annotations

import io

import numpy as np

from ultra_deepagents.ngff.reader import NgffChannel, NgffImage

__all__ = ["render_slice_png", "render_thumbnail_png", "render_tile_png"]


def _hex_to_rgb(color: str) -> tuple[float, float, float]:
    try:
        r = int(color[0:2], 16) / 255.0
        g = int(color[2:4], 16) / 255.0
        b = int(color[4:6], 16) / 255.0
        return (r, g, b)
    except (ValueError, IndexError):
        return (1.0, 1.0, 1.0)


def _auto_window(plane: np.ndarray) -> tuple[float, float]:
    """Robust 1st–99th percentile window. Subsamples large planes so auto-contrast
    stays cheap on big tiles."""
    flat = plane.reshape(-1)
    if flat.size > 500_000:
        step = flat.size // 500_000 + 1
        flat = flat[::step]
    finite = flat[np.isfinite(flat)] if np.issubdtype(flat.dtype, np.floating) else flat
    if finite.size == 0:
        return (0.0, 1.0)
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    return (lo, hi)


def _global_window(img: NgffImage, ci: int) -> tuple[float, float]:
    """Resolve the display window for a channel ONCE and cache it, so every slice/tile
    maps identically (no per-tile contrast checkerboard). Honors the OME `omero` window
    only when it actually overlaps the data — a stale/default window (e.g. 0–255 on
    uint16 data, the common case) would saturate/black the image, so we fall back to
    robust auto-contrast computed on the SMALLEST multiscale level (representative + cheap)."""
    cached = img.window_cache.get(ci)
    if cached is not None:
        return cached

    # Auto-contrast reference from the smallest level (fast; whole-image-representative).
    small_level = len(img.levels) - 1
    ref = img.read_plane(t=0, c=ci, z=0, level=small_level).astype(np.float32, copy=False)
    auto_lo, auto_hi = _auto_window(ref)

    window = (auto_lo, auto_hi)
    ch = img.channels[ci] if ci < len(img.channels) else None
    if ch is not None and ch.window_start is not None and ch.window_end is not None:
        start, end = float(ch.window_start), float(ch.window_end)
        # Use omero only if the window is non-degenerate AND overlaps the actual data.
        if end > start and not (end <= auto_lo or start >= auto_hi):
            window = (start, end)
    img.window_cache[ci] = window
    return window


def _window_to_uint8(plane: np.ndarray, start: float, end: float) -> np.ndarray:
    """Map a 2D plane to uint8 using a resolved [start, end] window."""
    p = plane.astype(np.float32, copy=False)
    if end <= start:
        end = start + 1.0
    scaled = (p - float(start)) / (float(end) - float(start))
    np.clip(scaled, 0.0, 1.0, out=scaled)
    return (scaled * 255.0 + 0.5).astype(np.uint8)


def _selected_channels(img: NgffImage, channels: list[int] | None) -> list[int]:
    if channels:
        return [c for c in channels if 0 <= c < img.num_c]
    active = [i for i, ch in enumerate(img.channels[: img.num_c]) if ch.active]
    return active or list(range(img.num_c))


def _is_grayscale(ch: NgffChannel) -> bool:
    return ch.color.upper() in ("FFFFFF", "")


def _compose_channels(img: NgffImage, planes: dict[int, np.ndarray], selected: list[int]) -> "Image.Image":
    """Composite already-read per-channel 2D arrays into a display image, using the global
    (cached) per-channel window so every plane/tile maps identically. Shared by the full-plane
    (slice/thumbnail) and bounded-region (tile) paths."""
    from PIL import Image

    # Single grayscale/white channel → 8-bit grayscale (matches brightfield/typical 1ch).
    if len(selected) == 1 and _is_grayscale(img.channels[selected[0]] if img.channels else NgffChannel("", "FFFFFF", None, None, True)):
        ci = selected[0]
        start, end = _global_window(img, ci)
        gray = _window_to_uint8(planes[ci], start, end)
        return Image.fromarray(gray, mode="L")

    # Otherwise composite the selected channels additively in RGB.
    rgb: np.ndarray | None = None
    for ci in selected:
        ch = img.channels[ci] if ci < len(img.channels) else NgffChannel("", "FFFFFF", None, None, True)
        start, end = _global_window(img, ci)
        gray = _window_to_uint8(planes[ci], start, end).astype(np.float32)
        if rgb is None:
            rgb = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.float32)
        rgb_w = _hex_to_rgb(ch.color)
        for k in range(3):
            if rgb_w[k]:
                rgb[:, :, k] += gray * rgb_w[k]
    if rgb is None:  # no channels at all
        return Image.new("L", (1, 1))
    np.clip(rgb, 0.0, 255.0, out=rgb)
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def _render(img: NgffImage, *, t: int, z: int, level: int, channels: list[int] | None) -> "Image.Image":
    selected = _selected_channels(img, channels)
    planes = {ci: img.read_plane(t=t, c=ci, z=z, level=level) for ci in selected}
    return _compose_channels(img, planes, selected)


def _to_png(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=False, compress_level=1)
    return buf.getvalue()


def render_slice_png(
    img: NgffImage,
    *,
    t: int = 0,
    z: int = 0,
    level: int = 0,
    channels: list[int] | None = None,
    max_dim: int | None = None,
) -> bytes:
    """Render one composited 2D plane (one t/z, selected channels) to PNG. ``max_dim``
    downscales the result (bounded scrub frame) while still only reading the level."""
    from PIL import Image

    image = _render(img, t=t, z=z, level=level, channels=channels)
    if max_dim and max(image.size) > max_dim:
        image.thumbnail((max_dim, max_dim), Image.BILINEAR)
    return _to_png(image)


def render_tile_png(
    img: NgffImage,
    *,
    level: int,
    col: int,
    row: int,
    tile_size: int = 256,
    t: int = 0,
    z: int = 0,
    channels: list[int] | None = None,
) -> bytes:
    """Render ONE DeepZoom tile (col,row) at a multiscale level — reading ONLY the chunks
    covering the tile's bounded region, so a gigapixel (1 TB) level-0 plane is never
    materialized whole. ``level`` maps 1:1 to the OME-Zarr multiscale dataset index
    (the tile_scheme advertises the same level list). Edge tiles return the cropped region."""
    from PIL import Image

    level = max(0, min(int(level), len(img.levels) - 1))
    tile_size = max(1, int(tile_size))
    h, w = img.level_yx(level)
    y0, x0 = int(row) * tile_size, int(col) * tile_size
    if y0 >= h or x0 >= w or y0 < 0 or x0 < 0:
        return _to_png(Image.new("L", (1, 1)))  # out-of-range: viewer ignores
    y1, x1 = min(y0 + tile_size, h), min(x0 + tile_size, w)
    selected = _selected_channels(img, channels)
    planes = {
        ci: img.read_region(level=level, y0=y0, y1=y1, x0=x0, x1=x1, t=t, c=ci, z=z)
        for ci in selected
    }
    return _to_png(_compose_channels(img, planes, selected))


def render_thumbnail_png(img: NgffImage, *, max_size: int = 256, t: int = 0, z: int = 0) -> bytes:
    """Render a bounded thumbnail from the smallest adequate multiscale level."""
    from PIL import Image

    level = img.thumbnail_level(max_size)
    image = _render(img, t=t, z=z, level=level, channels=None)
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.BILINEAR)
    return _to_png(image)
