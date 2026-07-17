"""FastAPI image service backed by libbioimage.

Internal service: the Go control plane proxies its V2 image endpoints here and
owns auth, quota, and catalog. This service owns decode/convert only. It returns
PNG bytes for image operations and JSON for metadata/histogram/formats.

``create_app`` takes a runner (:class:`~ultra_deepagents.imaging.pool.ImagePool`
in production, :class:`~ultra_deepagents.imaging.pool.InlineRunner` in tests).
FastAPI/uvicorn are only required to run the service, so they are imported lazily
and live in the ``imaging`` optional-dependency extra.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
from typing import Any

__all__ = ["create_app"]

_PNG = "image/png"

# --- Local pyramid cache ----------------------------------------------------
# A derived OME-BigTIFF pyramid has many scattered IFDs (z x channel x level);
# libbioimage scans them with thousands of tiny random reads. That is instant on
# local disk but HANGS over the cross-building barrel NFS (a single z-slice read
# measured >180s and tripped the op timeout, surfacing to users as "Failed to
# load image"). A pyramid is a regenerable cache, not durable data, so serve it
# from local disk: copy-on-first-use (one sequential NFS read, fast) into a
# size-bounded LRU, after which every tile/slice/atlas read is local. Best-effort:
# any error degrades to the original (NFS) path so serving never hard-fails.
_PYRAMID_CACHE_ENABLED = os.environ.get("ULTRA_IMGSVC_LOCAL_PYRAMID_CACHE", "1").strip().lower() not in ("0", "false", "no")
_PYRAMID_CACHE_DIR = (
    os.environ.get("ULTRA_IMGSVC_LOCAL_PYRAMID_CACHE_DIR", "").strip()
    or os.path.join(tempfile.gettempdir(), "ultra-pyramid-cache")
)


def _pyramid_cache_budget_bytes() -> int:
    raw = os.environ.get("ULTRA_IMGSVC_LOCAL_PYRAMID_CACHE_BYTES", "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return 64 * 1024 * 1024 * 1024  # 64 GiB


def _is_derived_pyramid(path: str) -> bool:
    return isinstance(path, str) and "/derived/" in path and path.endswith("__pyramid.tif")


def _evict_pyramid_cache(incoming: int) -> None:
    """LRU-evict cached pyramids by access time to keep the cache under budget."""
    try:
        budget = _pyramid_cache_budget_bytes()
        entries: list[tuple[float, int, str]] = []
        total = 0
        for name in os.listdir(_PYRAMID_CACHE_DIR):
            if not name.endswith(".tif"):
                continue
            fp = os.path.join(_PYRAMID_CACHE_DIR, name)
            try:
                s = os.stat(fp)
            except OSError:
                continue
            entries.append((s.st_atime, s.st_size, fp))
            total += s.st_size
        entries.sort()  # least-recently-accessed first
        while total + incoming > budget and entries:
            _, sz, fp = entries.pop(0)
            try:
                os.remove(fp)
                total -= sz
            except OSError:
                pass
    except OSError:
        pass


def localize_pyramid(path: str) -> str:
    """Return a local-disk copy of a derived pyramid (see module note); the original
    path for anything else or on any error (so reads never hard-fail)."""
    if not _PYRAMID_CACHE_ENABLED or not _is_derived_pyramid(path):
        return path
    try:
        st = os.stat(path)
    except OSError:
        return path
    key = hashlib.sha256(f"{path}|{st.st_size}|{int(st.st_mtime_ns)}".encode()).hexdigest()
    local = os.path.join(_PYRAMID_CACHE_DIR, key + ".tif")
    try:
        if os.path.exists(local):
            try:
                # LRU touch: bump atime for recency but PRESERVE mtime. The viewer-info
                # sidecar cache keys on (path, size, mtime_ns); bumping mtime here (the
                # old ``os.utime(local, None)`` set BOTH to now) changed that key on every
                # request, so a locally-cached pyramid's /viewerinfo missed its sidecar
                # every time and re-ran the ~300ms multichannel signal-score decode (and
                # accumulated an orphaned sidecar per hit). Keeping mtime stable lets the
                # sidecar hit after the first compute.
                st_local = os.stat(local)
                os.utime(local, (time.time(), st_local.st_mtime))  # (atime=now, mtime=unchanged)
            except OSError:
                pass
            return local
        os.makedirs(_PYRAMID_CACHE_DIR, exist_ok=True)
        _evict_pyramid_cache(st.st_size)
        tmp = f"{local}.tmp.{os.getpid()}"
        shutil.copyfile(path, tmp)  # whole-file sequential read: fast even over NFS
        os.replace(tmp, local)  # atomic publish
        return local
    except OSError:
        return path  # degrade to the NFS path; never hard-fail


async def _localize_pyramid_async(path: str) -> str:
    """``localize_pyramid`` off the event loop. A cold populate copies a multi-GB
    pyramid over NFS (~60s at the measured ~21MB/s barrel read rate); done inline
    in an ``async def`` handler that blocks this worker's entire event loop, so
    every other request routed to the process freezes for the duration. The warm
    path is a couple of stats — the threadpool hop costs microseconds."""
    from starlette.concurrency import run_in_threadpool  # lazy: service-only dep

    return await run_in_threadpool(localize_pyramid, path)


def _parse_fusion_request(channels: str | None, channel_colors: str | None):
    """Parse the multi-channel fusion query params shared by the image endpoints.

    ``channels`` is a comma list of 0-based channel indices (the engine's -remap
    is 1-based, so they are shifted). ``channel_colors`` is a comma list of hex LUT
    colors indexed by channel; they are realigned to the selected channels.

    Additive fusion is enabled ONLY for genuine multi-channel composites (2+
    selected channels with at least one color): single-channel and grayscale
    views keep the fast native display path. Returns
    (remap_channels_1based | None, aligned_colors | None).
    """
    from ultra_deepagents.imaging import fusion

    requested: list[int] | None = None
    if channels:
        requested = [int(part) for part in channels.split(",") if part.strip() != ""]
    by_channel: list | None = None
    if channel_colors:
        by_channel = [fusion.parse_hex_color(part) for part in channel_colors.split(",")]

    remap = [c + 1 for c in requested] if requested else None
    fuse = (
        requested is not None
        and len(requested) >= 2
        and by_channel is not None
        and any(c is not None for c in by_channel)
    )
    if not fuse:
        return remap, None
    # Realign colors to the selected channel order (channel_colors is indexed by
    # absolute channel; the remapped read returns the selected channels in order).
    aligned = [by_channel[ch] if 0 <= ch < len(by_channel) else None for ch in requested]
    if not any(c is not None for c in aligned):
        aligned = [fusion.convention_channel_color(i) for i in range(len(requested))]
    return remap, aligned


def create_app(runner: Any = None, *, prefer_real: bool = True):
    try:
        from fastapi import FastAPI, Response
    except Exception as exc:  # pragma: no cover - exercised only without fastapi
        raise RuntimeError(
            "image service requires fastapi/uvicorn (install the 'imaging' extra)"
        ) from exc

    if runner is None:
        from ultra_deepagents.imaging.pool import ImagePool

        runner = ImagePool(prefer_real=prefer_real)

    app = FastAPI(title="Ultra Image Service", version="0.1.0")
    app.state.runner = runner

    from fastapi.responses import JSONResponse

    # The engine raises ValueError for inputs it cannot decode or render — most often a
    # malformed/unsupported file that libbioimage reads as a 0-sized region (after the
    # engine's own transient-retry). That is a property of the FILE, not a server fault,
    # so map it to 422 instead of a 500: the client shows a clean "preview unavailable"
    # and monitoring doesn't count an undecodable upload as a server error. Genuine bugs
    # (any other exception) keep the default 500.
    _DECODE_ERROR_MARKERS = ("empty region", "cannot encode", "cannot decode", "unsupported")

    @app.exception_handler(ValueError)
    async def _engine_value_error(_request, exc: ValueError):  # noqa: ANN202
        message = str(exc)
        if any(marker in message for marker in _DECODE_ERROR_MARKERS):
            return JSONResponse(status_code=422, content={"error": "image could not be decoded or rendered", "detail": message})
        return JSONResponse(status_code=500, content={"error": "internal error", "detail": message})

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"status": "ok", "workers": runner.workers}

    @app.get("/formats")
    async def formats() -> dict[str, Any]:
        return {"formats": await runner.call("formats")}

    @app.get("/meta")
    async def meta(path: str) -> dict[str, Any]:
        return await runner.call("meta", await _localize_pyramid_async(path))

    @app.get("/tile")
    async def tile(
        path: str, level: int = 0, col: int = 0, row: int = 0, size: int = 512,
        channels: str | None = None, channel_colors: str | None = None,
    ):
        path = await _localize_pyramid_async(path)
        remap, fuse_colors = _parse_fusion_request(channels, channel_colors)
        png = await runner.call(
            "tile", path, level=level, col=col, row=row, tile_size=size,
            channels=remap, colors=fuse_colors,
        )
        return Response(content=png, media_type=_PNG)

    @app.get("/slice")
    async def slice_plane(
        path: str, z: int | None = None, t: int | None = None, level: int | None = None,
        channels: str | None = None, channel_colors: str | None = None,
        full_resolution: bool = True,
    ):
        path = await _localize_pyramid_async(path)
        remap, fuse_colors = _parse_fusion_request(channels, channel_colors)
        # full_resolution=False (transient z-scrub frame) lets the engine pick a
        # bounded pyramid level; the settled view (True) reads the native plane so
        # pixel measurements stay exact. An explicit level always wins.
        png = await runner.call(
            "slice_plane", path, z=z, t=t, level=level, channels=remap, colors=fuse_colors,
            full_resolution=full_resolution,
        )
        return Response(content=png, media_type=_PNG)

    @app.get("/thumbnail")
    async def thumbnail(
        path: str, max_size: int = 256, z: int | None = None, level: int | None = None,
        channels: str | None = None, channel_colors: str | None = None,
    ):
        path = await _localize_pyramid_async(path)
        remap, fuse_colors = _parse_fusion_request(channels, channel_colors)
        png = await runner.call(
            "thumbnail", path, max_size=max_size, z=z, level=level,
            channels=remap, colors=fuse_colors,
        )
        return Response(content=png, media_type=_PNG)

    @app.get("/atlas")
    async def atlas(
        path: str, grid_rows: int | None = None, grid_cols: int | None = None, level: int | None = None,
        scale: float | None = None, channels: str | None = None, channel_colors: str | None = None,
    ):
        path = await _localize_pyramid_async(path)
        remap, fuse_colors = _parse_fusion_request(channels, channel_colors)
        if getattr(runner, "workers", 1) > 1:
            # Multiple worker processes: fan the per-plane reads out across the pool
            # (the binding's global lock serializes them within one worker, so this is
            # the only way to parallelize a deep stack). Byte-identical to engine.atlas().
            from ultra_deepagents.imaging.atlas import assemble_atlas

            png = await assemble_atlas(runner, path, channels=remap, colors=fuse_colors, level=level)
        else:
            # Single process (InlineRunner / externally-managed): the sequential path.
            grid = (grid_rows, grid_cols) if grid_rows and grid_cols else None
            png = await runner.call(
                "atlas", path, grid=grid, level=level, atlas_scale=scale,
                channels=remap, colors=fuse_colors,
            )
        return Response(content=png, media_type=_PNG)

    @app.get("/histogram")
    async def histogram(path: str, bins: int = 256) -> dict[str, Any]:
        return await runner.call("histogram", await _localize_pyramid_async(path), bins=bins)

    @app.get("/viewerinfo")
    async def viewerinfo(path: str, name: str | None = None) -> dict[str, Any]:
        # ``name`` (the upload's original filename) lets HDF5-data files be detected
        # when the on-disk blob path has lost its extension; harmless for images.
        return await runner.call("viewer_info", await _localize_pyramid_async(path), name=name)

    @app.get("/video-poster")
    async def video_poster(path: str, t: float = 1.0, max_size: int = 512):
        # ffmpeg runs as a bounded subprocess, off the libbioimage decode pool.
        from ultra_deepagents.imaging import video

        try:
            png = await video.extract_poster(path, time_seconds=t, max_size=max_size)
        except video.VideoError as exc:
            return Response(content=str(exc), media_type="text/plain", status_code=415)
        return Response(content=png, media_type=_PNG, headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/scalar-volume")
    async def scalar_volume(path: str, channel: int = 0, t: int = 0):
        path = await _localize_pyramid_async(path)
        if getattr(runner, "workers", 1) > 1:
            from ultra_deepagents.imaging.atlas import assemble_scalar_volume

            vol = await assemble_scalar_volume(runner, path, channel=channel, t=t)
        else:
            vol = await runner.call("scalar_volume", path, channel=channel, t=t)
        headers = {
            "x-volume-width": str(vol["width"]),
            "x-volume-height": str(vol["height"]),
            "x-volume-depth": str(vol["depth"]),
            "x-volume-dtype": str(vol["dtype"]),
            "x-volume-bytes-per-voxel": str(vol["bytes_per_voxel"]),
            "x-volume-raw-min": str(float(vol["raw_min"])),
            "x-volume-raw-max": str(float(vol["raw_max"])),
            "x-volume-channel": str(vol["channel"]),
        }
        return Response(content=vol["data"], media_type="application/octet-stream", headers=headers)

    # --- HDF5 data viewer -------------------------------------------------------
    # These serve the frontend HDF5 explorer (frontend/src/components/viewer/hdf5).
    # The Go control plane resolves file_id -> path + forwards dataset_path (+ the
    # per-endpoint params). Response shapes are the frozen FE wire contract
    # (frontend/src/types.ts Hdf5* types). An unknown dataset -> 404; a non-numeric/
    # non-tabular/unsupported dataset -> 422 (via the ValueError handler above; the
    # reader raises messages carrying an "unsupported" marker). Heavy reads run in the
    # pool (runner.call), bounded + downsampled inside imaging/hdf5.py.
    from ultra_deepagents.imaging.hdf5 import Hdf5DatasetNotFound

    def _not_found(exc: Exception):
        return JSONResponse(status_code=404, content={"error": "dataset not found", "detail": str(exc)})

    @app.get("/hdf5/dataset")
    async def hdf5_dataset(path: str, dataset_path: str, file_id: str = ""):
        try:
            return await runner.call("hdf5_dataset_summary", path, dataset_path, file_id=file_id)
        except Hdf5DatasetNotFound as exc:
            return _not_found(exc)

    @app.get("/hdf5/preview/slice")
    async def hdf5_slice(
        path: str, dataset_path: str, axis: str = "z", index: int | None = None, component: int = 0,
    ):
        try:
            png = await runner.call(
                "hdf5_slice_png", path, dataset_path, axis=axis, index=index, component=component,
            )
        except Hdf5DatasetNotFound as exc:
            return _not_found(exc)
        return Response(content=png, media_type=_PNG, headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/hdf5/preview/atlas")
    async def hdf5_atlas(
        path: str, dataset_path: str, enhancement: str | None = None, fusion_method: str | None = None,
        negative: str | None = None, channels: str | None = None,
    ):
        try:
            png = await runner.call("hdf5_atlas_png", path, dataset_path)
        except Hdf5DatasetNotFound as exc:
            return _not_found(exc)
        return Response(content=png, media_type=_PNG, headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/hdf5/preview/scalar-volume")
    async def hdf5_scalar_volume(path: str, dataset_path: str, channel: int = 0):
        try:
            vol = await runner.call("hdf5_scalar_volume", path, dataset_path, channel=channel)
        except Hdf5DatasetNotFound as exc:
            return _not_found(exc)
        headers = {
            "x-volume-width": str(vol["width"]),
            "x-volume-height": str(vol["height"]),
            "x-volume-depth": str(vol["depth"]),
            "x-volume-dtype": str(vol["dtype"]),
            "x-volume-bytes-per-voxel": str(vol["bytes_per_voxel"]),
            "x-volume-raw-min": str(float(vol["raw_min"])),
            "x-volume-raw-max": str(float(vol["raw_max"])),
            "x-volume-channel": str(vol["channel"]),
        }
        return Response(content=vol["data"], media_type="application/octet-stream", headers=headers)

    @app.get("/hdf5/preview/histogram")
    async def hdf5_histogram(path: str, dataset_path: str, component: int = 0, bins: int = 24, file_id: str = ""):
        try:
            return await runner.call(
                "hdf5_histogram", path, dataset_path, component=component, bins=bins, file_id=file_id,
            )
        except Hdf5DatasetNotFound as exc:
            return _not_found(exc)

    @app.get("/hdf5/preview/table")
    async def hdf5_table(path: str, dataset_path: str, offset: int = 0, limit: int = 12, file_id: str = ""):
        try:
            return await runner.call(
                "hdf5_table", path, dataset_path, offset=offset, limit=limit, file_id=file_id,
            )
        except Hdf5DatasetNotFound as exc:
            return _not_found(exc)

    return app
