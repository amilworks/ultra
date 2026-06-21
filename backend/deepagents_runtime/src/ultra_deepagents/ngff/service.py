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
import time
from collections import OrderedDict
from typing import Any

from fastapi import FastAPI, Query, Response
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from ultra_deepagents.ngff.reader import NgffError, NgffImage, open_ngff
from ultra_deepagents.ngff.render import render_slice_png, render_thumbnail_png, render_tile_png
from ultra_deepagents.ngff.viewerinfo import build_ngff_viewer_info

# Transient z-scrub / pan frame long-edge cap (matches the libbioimage scrub bound).
SCRUB_MAX_DIMENSION = 1024
# How many opened images (parsed metadata + window cache + zarr handles) to keep warm.
_OPEN_CACHE_MAX = int(os.environ.get("ULTRA_NGFF_OPEN_CACHE", "64"))

_open_cache: "OrderedDict[tuple[str, float], NgffImage]" = OrderedDict()
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
    if not channels:
        return None
    out: list[int] = []
    for tok in channels.split(","):
        tok = tok.strip()
        if tok.isdigit():
            out.append(int(tok))
    return out or None


def create_app() -> FastAPI:
    app = FastAPI(title="ultra-ngff-service")

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:  # noqa: D401
        return {"status": "ok", "service": "ngff", "open_cached": len(_open_cache)}

    @app.get("/viewerinfo")
    async def viewerinfo(path: str) -> Any:
        try:
            img = await run_in_threadpool(_get_image, path)
            return await run_in_threadpool(build_ngff_viewer_info, img)
        except (NgffError, ValueError) as exc:
            return JSONResponse(status_code=422, content={"error": "not a readable OME-Zarr", "detail": str(exc)})

    @app.get("/slice")
    async def slice_plane(
        path: str,
        t: int = 0,
        z: int = 0,
        level: int | None = None,
        channels: str | None = None,
        full_resolution: bool = True,
    ) -> Response:
        try:
            img = await run_in_threadpool(_get_image, path)
            lvl = 0 if level is None else int(level)
            max_dim = None if full_resolution else SCRUB_MAX_DIMENSION
            png = await run_in_threadpool(
                render_slice_png, img, t=t, z=z, level=lvl,
                channels=_parse_channels(channels), max_dim=max_dim,
            )
            return Response(content=png, media_type="image/png")
        except (NgffError, ValueError) as exc:
            return JSONResponse(status_code=422, content={"error": "cannot render OME-Zarr slice", "detail": str(exc)})

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
    ) -> Response:
        # One DeepZoom tile, reading ONLY the chunks covering the tile region (gigapixel-safe).
        try:
            img = await run_in_threadpool(_get_image, path)
            png = await run_in_threadpool(
                render_tile_png, img, level=int(level), col=int(col), row=int(row),
                tile_size=int(size), t=int(t), z=int(z), channels=_parse_channels(channels),
            )
            return Response(content=png, media_type="image/png")
        except (NgffError, ValueError) as exc:
            return JSONResponse(status_code=422, content={"error": "cannot render OME-Zarr tile", "detail": str(exc)})

    @app.get("/thumbnail")
    async def thumbnail(path: str, max_size: int = 256, t: int = 0, z: int = 0) -> Response:
        try:
            img = await run_in_threadpool(_get_image, path)
            png = await run_in_threadpool(render_thumbnail_png, img, max_size=max_size, t=t, z=z)
            return Response(content=png, media_type="image/png")
        except (NgffError, ValueError) as exc:
            return JSONResponse(status_code=422, content={"error": "cannot render OME-Zarr thumbnail", "detail": str(exc)})

    return app
