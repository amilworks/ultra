"""A deterministic CPU poster for the Resources grid — not the WebGL render.

The grid needs a thumbnail before anything is downloaded, and the derive worker has no
GPU. So this is an orthographic z-buffer: project along the world axis with the smallest
extent (which shows the two largest extents, the informative face of an aerial or
corridor scan), stamp each element as its projected footprint, keep the nearest.

It is an approximation and the manifest says so. Specifically: one layer deep, no
alpha accumulation between overlapping splats, and a footprint circle standing in for
the projected ellipse. It exists to tell one scene apart from another in a grid, not to
be believed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

__all__ = [
    "DEFAULT_POSTER_SAMPLE",
    "POSTER_MAX_PIXELS",
    "PosterResult",
    "poster_stride",
    "render_poster",
]

POSTER_MAX_PIXELS = 512
# Elements stamped into the poster. Beyond this the thumbnail stops changing but the
# stamping cost keeps growing; the sample is a fixed stride over source order (never
# random, never spatially biased) and the rendered/total counts are reported.
DEFAULT_POSTER_SAMPLE = 400_000
# Footprint cap in pixels. A single near splat with a huge scale would otherwise paint
# over the whole poster.
_MAX_FOOTPRINT_PX = 4
# Percentile trimmed off each end when framing. Not a filter — every element is still
# drawn, clamped to the border; this only decides what the 512 px cover.
_POSTER_PERCENTILE = 1.0


@dataclass(frozen=True)
class PosterResult:
    """What was actually drawn, so the manifest can state it rather than imply it."""

    path: str
    width: int
    height: int
    rendered: int
    total: int
    axis: int  # world axis projected along (0=x, 1=y, 2=z)


def poster_stride(total: int, sample: int = DEFAULT_POSTER_SAMPLE) -> int:
    """Take every ``stride``-th element in source order; 1 when the scene is small."""
    if total <= sample or sample < 1:
        return 1
    return int(-(-total // sample))  # ceil, so the sample never exceeds `sample`


def _disc_offsets(radius: int) -> np.ndarray:
    """(k, 2) integer (du, dv) offsets covering a filled disc of ``radius`` pixels."""
    span = np.arange(-radius, radius + 1, dtype=np.int64)
    du, dv = np.meshgrid(span, span, indexing="ij")
    inside = (du * du + dv * dv) <= radius * radius
    return np.stack([du[inside], dv[inside]], axis=1)


def render_poster(
    path: str | os.PathLike[str],
    *,
    positions: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray | None = None,
    radii: np.ndarray | None = None,
    total: int | None = None,
    max_pixels: int = POSTER_MAX_PIXELS,
) -> PosterResult:
    """Render and write the poster PNG.

    ``colors`` are display sRGB in [0,1] (the clamped ``0.5 + C0*f_dc`` base for splats,
    the source bytes for points) — the PNG is an sRGB container, so the linearised wire
    colour would come out visibly dark. ``radii`` are world-unit footprint radii, absent
    for points.
    """
    xyz = np.ascontiguousarray(positions, dtype=np.float64)
    count = int(xyz.shape[0])
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("positions must be (n, 3)")
    if count < 1:
        raise ValueError("cannot render a poster for an empty scene")
    rgb = np.clip(np.asarray(colors, dtype=np.float32), 0.0, 1.0)
    if rgb.shape != (count, 3):
        raise ValueError("colors must be (n, 3)")
    alpha = (
        np.ones(count, dtype=np.float32)
        if opacities is None
        else np.clip(np.asarray(opacities, dtype=np.float32), 0.0, 1.0)
    )
    footprint = (
        np.zeros(count, dtype=np.float64) if radii is None else np.asarray(radii, np.float64)
    )

    # Frame on the robust extent, not the full bounding box. The measured point cloud has
    # outliers that stretch its box to 1255 x 354 x 3198 while the body of the scan
    # occupies roughly a third of that; framed on the full box the whole scene collapses
    # into a corner of the thumbnail. Outliers are still drawn — clamped to the border —
    # so nothing is dropped, only pushed to the edge.
    low, high = np.percentile(xyz, [_POSTER_PERCENTILE, 100.0 - _POSTER_PERCENTILE], axis=0)
    degenerate = high <= low
    low = np.where(degenerate, xyz.min(axis=0), low)
    high = np.where(degenerate, xyz.max(axis=0), high)
    extent = high - low
    axis = int(np.argmin(extent))
    horizontal, vertical = (a for a in range(3) if a != axis)
    span = max(float(extent[horizontal]), float(extent[vertical]))
    scale = (max_pixels - 1) / span if span > 0.0 else 1.0
    width = int(min(max_pixels, max(1, round(float(extent[horizontal]) * scale) + 1)))
    height = int(min(max_pixels, max(1, round(float(extent[vertical]) * scale) + 1)))

    px = np.clip(((xyz[:, horizontal] - low[horizontal]) * scale).astype(np.int64), 0, width - 1)
    # Flip vertically: image rows grow downward, world +v grows upward.
    py = (height - 1) - np.clip(
        ((xyz[:, vertical] - low[vertical]) * scale).astype(np.int64), 0, height - 1
    )
    radius_px = np.clip(np.rint(footprint * scale), 0, _MAX_FOOTPRINT_PX).astype(np.int64)

    # Expand each element into its footprint pixels, grouped by radius so the offsets
    # are computed once per radius instead of once per element.
    pixels: list[np.ndarray] = []
    source: list[np.ndarray] = []
    for radius in range(_MAX_FOOTPRINT_PX + 1):
        member = np.flatnonzero(radius_px == radius)
        if member.size == 0:
            continue
        offsets = _disc_offsets(radius)
        stamp_u = np.clip(px[member][:, None] + offsets[None, :, 0], 0, width - 1)
        stamp_v = np.clip(py[member][:, None] + offsets[None, :, 1], 0, height - 1)
        pixels.append((stamp_v * width + stamp_u).reshape(-1))
        source.append(np.repeat(member, offsets.shape[0]))
    flat_pixel = np.concatenate(pixels)
    flat_source = np.concatenate(source)

    # Painter's algorithm collapsed to a z-buffer: sort near-first along the projection
    # axis (the viewer sits on its positive side), then the first hit on each pixel wins.
    near_first = np.argsort(-xyz[flat_source, axis], kind="stable")
    flat_pixel = flat_pixel[near_first]
    flat_source = flat_source[near_first]
    unique_pixel, first_hit = np.unique(flat_pixel, return_index=True)
    winner = flat_source[first_hit]

    canvas = np.zeros((height * width, 4), dtype=np.float32)
    canvas[unique_pixel, 0:3] = rgb[winner]
    canvas[unique_pixel, 3] = alpha[winner]
    image = np.rint(canvas.reshape(height, width, 4) * 255.0).astype(np.uint8)
    Image.fromarray(image, mode="RGBA").save(os.fspath(path), format="PNG", optimize=True)
    return PosterResult(
        path=os.fspath(path),
        width=width,
        height=height,
        rendered=count,
        total=count if total is None else int(total),
        axis=axis,
    )
