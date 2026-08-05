"""ngff-service FastAPI app — native OME-Zarr serving.

Stateless: reads zarr chunks from a path (resolved + authorized by the control plane,
never the client) and renders viewer-info / slices / thumbnails. Blocking zarr reads run
in a threadpool (zarr/blosc release the GIL) so the async event loop stays responsive and
the process scales with threads (unlike the libbioimage process pool). Opened images are
cached per path+mtime so the parsed NGFF metadata + the resolved display windows are reused
across the many tile/slice requests of one viewing session.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Any

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from ultra_deepagents.imaging.constants import MAX_COMPOSITE_CHANNELS, MAX_TILE_EDGE
from ultra_deepagents.ngff.reader import (
    NgffError,
    NgffImage,
    open_ngff,
    process_plane_cache_info,
)
from ultra_deepagents.ngff.render import render_slice_png, render_thumbnail_png, render_tile_png
from ultra_deepagents.ngff.viewerinfo import build_ngff_viewer_info

# Transient z-scrub / pan frame long-edge cap (matches the libbioimage scrub bound).
SCRUB_MAX_DIMENSION = 1024
# How many opened images (parsed metadata + window cache + zarr handles) to keep warm.
_OPEN_CACHE_MAX = int(os.environ.get("ULTRA_NGFF_OPEN_CACHE", "64"))

_open_cache: OrderedDict[tuple[str, float], NgffImage] = OrderedDict()
_open_lock = threading.Lock()


def _stat_stamp(path: str) -> float:
    """mtime of the group metadata — invalidates the cache if the store is rewritten."""
    for marker in (".zattrs", "zarr.json"):
        p = os.path.join(path, marker)
        try:
            return os.path.getmtime(p)
        except OSError:
            continue
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _get_image(path: str) -> NgffImage:
    key = (path, _stat_stamp(path))
    with _open_lock:
        img = _open_cache.get(key)
        if img is not None:
            _open_cache.move_to_end(key)
            return img
    img = open_ngff(path)  # parse metadata (no pixel read)
    with _open_lock:
        _open_cache[key] = img
        _open_cache.move_to_end(key)
        while len(_open_cache) > _OPEN_CACHE_MAX:
            _open_cache.popitem(last=False)
    return img


def _parse_channels(channels: str | None) -> list[int] | None:
    if channels is None:
        return None
    out: list[int] = []
    for tok in channels.split(","):
        tok = tok.strip()
        if not tok or not tok.isdigit():
            raise ValueError("channels must be a comma-separated list of non-negative integers")
        out.append(int(tok))
    if len(set(out)) != len(out):
        raise ValueError("channels must not contain duplicates")
    if len(out) > MAX_COMPOSITE_CHANNELS:
        raise ValueError(f"channels supports at most {MAX_COMPOSITE_CHANNELS} selections")
    return out


def _parse_channel_colors(
    channel_colors: str | None, channels: list[int] | None
) -> list[str] | None:
    if channel_colors is None:
        return None
    colors = [part.strip().removeprefix("#").upper() for part in channel_colors.split(",")]
    if any(
        len(color) != 6 or any(char not in "0123456789ABCDEF" for char in color) for color in colors
    ):
        raise ValueError("channel colors must be comma-separated six-digit hex values")
    if channels is None or len(colors) != len(channels):
        raise ValueError("channel colors must match the selected channel count")
    return colors


def _validate_tile_size(size: int) -> int:
    if isinstance(size, bool) or size <= 0 or size > MAX_TILE_EDGE:
        raise ValueError(f"tile size must be between 1 and {MAX_TILE_EDGE} pixels")
    return size


def create_app() -> FastAPI:
    app = FastAPI(title="ultra-ngff-service")

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:  # noqa: D401
        return {
            "status": "ok",
            "service": "ngff",
            "open_cached": len(_open_cache),
            "decoded_plane_cache": process_plane_cache_info(),
        }

    @app.get("/viewerinfo")
    async def viewerinfo(path: str) -> Any:
        try:
            img = await run_in_threadpool(_get_image, path)
            return await run_in_threadpool(build_ngff_viewer_info, img)
        except (NgffError, ValueError) as exc:
            return JSONResponse(
                status_code=422, content={"error": "not a readable OME-Zarr", "detail": str(exc)}
            )

    @app.get("/slice")
    async def slice_plane(
        path: str,
        t: int = 0,
        z: int = 0,
        level: int | None = None,
        channels: str | None = None,
        channel_colors: str | None = None,
        full_resolution: bool = True,
    ) -> Response:
        try:
            selected = _parse_channels(channels)
            colors = _parse_channel_colors(channel_colors, selected)
            img = await run_in_threadpool(_get_image, path)
            lvl = 0 if level is None else int(level)
            max_dim = None if full_resolution else SCRUB_MAX_DIMENSION
            png = await run_in_threadpool(
                render_slice_png,
                img,
                t=t,
                z=z,
                level=lvl,
                channels=selected,
                channel_colors=colors,
                max_dim=max_dim,
            )
            return Response(content=png, media_type="image/png")
        except (NgffError, ValueError) as exc:
            return JSONResponse(
                status_code=422,
                content={"error": "cannot render OME-Zarr slice", "detail": str(exc)},
            )

    @app.get("/tile")
    async def tile(
        path: str,
        level: int = 0,
        col: int = 0,
        row: int = 0,
        size: int = 256,
        t: int = 0,
        z: int = 0,
        channels: str | None = None,
        channel_colors: str | None = None,
    ) -> Response:
        # One DeepZoom tile, reading ONLY the chunks covering the tile region (gigapixel-safe).
        try:
            size = _validate_tile_size(size)
            selected = _parse_channels(channels)
            colors = _parse_channel_colors(channel_colors, selected)
            img = await run_in_threadpool(_get_image, path)
            png = await run_in_threadpool(
                render_tile_png,
                img,
                level=int(level),
                col=int(col),
                row=int(row),
                tile_size=int(size),
                t=int(t),
                z=int(z),
                channels=selected,
                channel_colors=colors,
            )
            return Response(content=png, media_type="image/png")
        except (NgffError, ValueError) as exc:
            return JSONResponse(
                status_code=422,
                content={"error": "cannot render OME-Zarr tile", "detail": str(exc)},
            )

    @app.get("/thumbnail")
    async def thumbnail(path: str, max_size: int = 256, t: int = 0, z: int = 0) -> Response:
        try:
            img = await run_in_threadpool(_get_image, path)
            png = await run_in_threadpool(render_thumbnail_png, img, max_size=max_size, t=t, z=z)
            return Response(content=png, media_type="image/png")
        except (NgffError, ValueError) as exc:
            return JSONResponse(
                status_code=422,
                content={"error": "cannot render OME-Zarr thumbnail", "detail": str(exc)},
            )

    return app
