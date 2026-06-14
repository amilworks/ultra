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

from typing import Any

__all__ = ["create_app"]

_PNG = "image/png"


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
        return await runner.call("meta", path)

    @app.get("/tile")
    async def tile(
        path: str, level: int = 0, col: int = 0, row: int = 0, size: int = 512,
        channels: str | None = None, channel_colors: str | None = None,
    ):
        remap, fuse_colors = _parse_fusion_request(channels, channel_colors)
        png = await runner.call(
            "tile", path, level=level, col=col, row=row, tile_size=size,
            channels=remap, colors=fuse_colors,
        )
        return Response(content=png, media_type=_PNG)

    @app.get("/region")
    async def region(
        path: str, x1: int, y1: int, x2: int, y2: int, scale: float | None = None,
        channels: str | None = None, channel_colors: str | None = None,
    ):
        remap, fuse_colors = _parse_fusion_request(channels, channel_colors)
        png = await runner.call(
            "region", path, x1=x1, y1=y1, x2=x2, y2=y2, region_scale=scale,
            channels=remap, colors=fuse_colors,
        )
        return Response(content=png, media_type=_PNG)

    @app.get("/slice")
    async def slice_plane(
        path: str, z: int | None = None, t: int | None = None, level: int | None = None,
        channels: str | None = None, channel_colors: str | None = None,
        full_resolution: bool = True,
    ):
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
        return await runner.call("histogram", path, bins=bins)

    @app.get("/viewerinfo")
    async def viewerinfo(path: str) -> dict[str, Any]:
        return await runner.call("viewer_info", path)

    @app.get("/video-poster")
    async def video_poster(path: str, t: float = 1.0, max_size: int = 512):
        # ffmpeg runs as a bounded subprocess, off the libbioimage decode pool.
        from ultra_deepagents.imaging import video

        try:
            png = await video.extract_poster(path, time_seconds=t, max_size=max_size)
        except video.VideoError as exc:
            return Response(content=str(exc), media_type="text/plain", status_code=415)
        return Response(content=png, media_type=_PNG, headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/video-info")
    async def video_info(path: str) -> dict[str, Any]:
        from ultra_deepagents.imaging import video

        try:
            return await video.probe_info(path)
        except video.VideoError as exc:
            return Response(content=str(exc), media_type="text/plain", status_code=415)

    @app.get("/scalar-volume")
    async def scalar_volume(path: str, channel: int = 0, t: int = 0):
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

    return app
