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

import asyncio
import hashlib
import math
import operator
import os
import shutil
import stat
import tempfile
import threading
from contextlib import contextmanager, suppress
from typing import Any, cast

from ultra_deepagents.imaging.constants import (
    MAX_ATLAS_CELLS,
    MAX_ATLAS_GRID_EDGE,
    MAX_COMPOSITE_CHANNELS,
    MAX_TILE_EDGE,
    is_ultra_owned_pyramid,
)

__all__ = ["MAX_COMPOSITE_CHANNELS", "create_app"]

_PNG = "image/png"
_SCALAR_VOLUME_MAX_BYTES = 256 * 1024 * 1024
_SCALAR_DTYPE_BYTES = {"uint8": 1, "uint16": 2, "int16": 2, "float32": 4}
_SCALAR_VOLUME_RESIDENT_MULTIPLIER = 3
_DEFAULT_SCALAR_VOLUME_INFLIGHT_BYTES = 1024 * 1024 * 1024


def _scalar_volume_inflight_budget_bytes() -> int:
    raw = os.environ.get("ULTRA_IMGSVC_SCALAR_VOLUME_INFLIGHT_BYTES", "").strip()
    if not raw:
        return _DEFAULT_SCALAR_VOLUME_INFLIGHT_BYTES
    try:
        parsed = int(raw)
    except ValueError:
        return _DEFAULT_SCALAR_VOLUME_INFLIGHT_BYTES
    return parsed if parsed > 0 else _DEFAULT_SCALAR_VOLUME_INFLIGHT_BYTES


class _WeightedByteBudget:
    """Fail-fast process-local admission for large scalar response residency."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._used = 0
        self._lock = asyncio.Lock()

    async def try_acquire(self, weight: int) -> bool:
        if weight <= 0 or weight > self._capacity:
            return False
        async with self._lock:
            if self._used + weight > self._capacity:
                return False
            self._used += weight
            return True

    async def release(self, weight: int) -> None:
        async with self._lock:
            self._used = max(0, self._used - weight)


def _exact_integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"scalar-volume {field} must be an integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"scalar-volume {field} must be an integer") from exc


def _scalar_volume_envelope(vol: dict[str, Any]) -> tuple[Any, dict[str, str]]:
    width = _exact_integer(vol["width"], "width")
    height = _exact_integer(vol["height"], "height")
    depth = _exact_integer(vol["depth"], "depth")
    dtype = str(vol["dtype"]).strip().lower()
    bytes_per_voxel = _exact_integer(vol["bytes_per_voxel"], "bytes_per_voxel")
    if width <= 0 or height <= 0 or depth <= 0:
        raise ValueError("invalid scalar-volume geometry")
    if _SCALAR_DTYPE_BYTES.get(dtype) != bytes_per_voxel:
        raise ValueError("unsupported scalar-volume dtype/byte width")
    expected_length = width * height * depth * bytes_per_voxel
    if expected_length <= 0 or expected_length > _SCALAR_VOLUME_MAX_BYTES:
        raise ValueError("scalar-volume preview exceeds the bounded response policy")
    data = vol["data"]
    if len(data) != expected_length:
        raise ValueError("scalar-volume body length does not match geometry")

    raw_min = float(vol["raw_min"])
    raw_max = float(vol["raw_max"])
    slope = float(vol["scl_slope"])
    intercept = float(vol["scl_inter"])
    values = (
        raw_min,
        raw_max,
        slope,
        intercept,
        raw_min * slope + intercept,
        raw_max * slope + intercept,
    )
    if not all(math.isfinite(value) for value in values) or slope == 0 or raw_max < raw_min:
        raise ValueError("scalar-volume intensity metadata is invalid")

    channel = _exact_integer(vol["channel"], "channel")
    time_index = _exact_integer(vol["t"], "time")
    source_width = _exact_integer(vol["source_width"], "source_width")
    source_height = _exact_integer(vol["source_height"], "source_height")
    source_depth = _exact_integer(vol["source_depth"], "source_depth")
    downsample_x = _exact_integer(vol["downsample_x"], "downsample_x")
    downsample_y = _exact_integer(vol["downsample_y"], "downsample_y")
    downsample_z = _exact_integer(vol["downsample_z"], "downsample_z")
    preview_policy = str(vol["preview_policy"]).strip()
    sampling = str(vol.get("sampling", "box")).strip().lower()
    if channel < 0 or time_index < 0:
        raise ValueError("scalar-volume channel/time identity must be nonnegative")
    if (
        min(source_width, source_height, source_depth, downsample_x, downsample_y, downsample_z)
        <= 0
    ):
        raise ValueError("scalar-volume source geometry/provenance must be positive")
    delivered = (
        (source_width + downsample_x - 1) // downsample_x,
        (source_height + downsample_y - 1) // downsample_y,
        (source_depth + downsample_z - 1) // downsample_z,
    )
    if (
        delivered != (width, height, depth)
        or not preview_policy
        or sampling not in {"box", "nearest"}
    ):
        raise ValueError("scalar-volume provenance does not match the delivery grid")

    headers = {
        "x-volume-width": str(width),
        "x-volume-height": str(height),
        "x-volume-depth": str(depth),
        "x-volume-dtype": dtype,
        "x-volume-bytes-per-voxel": str(bytes_per_voxel),
        "x-volume-raw-min": str(raw_min),
        "x-volume-raw-max": str(raw_max),
        "x-volume-scl-slope": str(slope),
        "x-volume-scl-inter": str(intercept),
        "x-volume-channel": str(channel),
        "x-volume-time": str(time_index),
        "x-volume-source-width": str(source_width),
        "x-volume-source-height": str(source_height),
        "x-volume-source-depth": str(source_depth),
        "x-volume-downsample-x": str(downsample_x),
        "x-volume-downsample-y": str(downsample_y),
        "x-volume-downsample-z": str(downsample_z),
        "x-volume-preview-policy": preview_policy,
        "x-volume-sampling": sampling,
    }
    return data, headers


# --- Local pyramid cache ----------------------------------------------------
# A derived OME-BigTIFF pyramid has many scattered IFDs (z x channel x level);
# libbioimage scans them with thousands of tiny random reads. That is instant on
# local disk but HANGS over the cross-building barrel NFS (a single z-slice read
# measured >180s and tripped the op timeout, surfacing to users as "Failed to
# load image"). A pyramid is a regenerable cache, not durable data, so serve it
# from local disk: copy-on-first-use (one sequential NFS read, fast) into a
# size-bounded LRU, after which every tile/slice/atlas read is local. Best-effort:
# any error degrades to the original (NFS) path so serving never hard-fails.
_PYRAMID_CACHE_ENABLED = os.environ.get(
    "ULTRA_IMGSVC_LOCAL_PYRAMID_CACHE", "1"
).strip().lower() not in ("0", "false", "no")
_PYRAMID_CACHE_DIR = os.environ.get(
    "ULTRA_IMGSVC_LOCAL_PYRAMID_CACHE_DIR", ""
).strip() or os.path.join(tempfile.gettempdir(), "ultra-pyramid-cache")
_PYRAMID_CACHE_LOCKS_GUARD = threading.Lock()
_PYRAMID_CACHE_MUTATION_LOCK = threading.Lock()
_PYRAMID_CACHE_LOCKS: dict[str, tuple[threading.Lock, int]] = {}


def _pyramid_cache_budget_bytes() -> int:
    raw = os.environ.get("ULTRA_IMGSVC_LOCAL_PYRAMID_CACHE_BYTES", "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return 64 * 1024 * 1024 * 1024  # 64 GiB


def _is_derived_pyramid(path: str) -> bool:
    """Compatibility alias for the shared owned-pyramid classifier."""

    return bool(is_ultra_owned_pyramid(path))


def _pyramid_access_marker(path: str) -> str:
    return f"{path}.access"


def _regular_file_identity(path: str) -> tuple[int, int, int, int, int]:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise OSError("pyramid cache source must be a regular file")
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


@contextmanager
def _pyramid_cache_flight(local_path: str):
    with _PYRAMID_CACHE_LOCKS_GUARD:
        lock, references = _PYRAMID_CACHE_LOCKS.get(local_path, (threading.Lock(), 0))
        _PYRAMID_CACHE_LOCKS[local_path] = (lock, references + 1)
    lock.acquire()
    try:
        yield
    finally:
        lock.release()
        with _PYRAMID_CACHE_LOCKS_GUARD:
            current_lock, current_references = _PYRAMID_CACHE_LOCKS[local_path]
            if current_references == 1:
                del _PYRAMID_CACHE_LOCKS[local_path]
            else:
                _PYRAMID_CACHE_LOCKS[local_path] = (
                    current_lock,
                    current_references - 1,
                )


def _cached_pyramid_ready(path: str, expected_size: int) -> bool:
    try:
        identity = _regular_file_identity(path)
    except OSError:
        return False
    return identity[2] == expected_size


def _fsync_cache_directory(path: str) -> None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_stable_pyramid(
    source_path: str,
    local_path: str,
    expected_identity: tuple[int, int, int, int, int],
) -> bool:
    source_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    source_descriptor = os.open(source_path, source_flags)
    temp_path: str | None = None
    try:
        opened = os.fstat(source_descriptor)
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if (
            opened_identity != expected_identity
            or _regular_file_identity(source_path) != opened_identity
        ):
            return False
        temp_descriptor, temp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(local_path)}.tmp-",
            dir=os.path.dirname(local_path),
        )
        with (
            os.fdopen(source_descriptor, "rb", closefd=False) as source_stream,
            os.fdopen(temp_descriptor, "wb") as destination_stream,
        ):
            shutil.copyfileobj(source_stream, destination_stream, length=1024 * 1024)
            destination_stream.flush()
            os.fchmod(destination_stream.fileno(), 0o600)
            os.fsync(destination_stream.fileno())
        source_after = os.fstat(source_descriptor)
        source_after_identity = (
            source_after.st_dev,
            source_after.st_ino,
            source_after.st_size,
            source_after.st_mtime_ns,
            source_after.st_ctime_ns,
        )
        if (
            source_after_identity != expected_identity
            or _regular_file_identity(source_path) != expected_identity
            or os.stat(temp_path).st_size != expected_identity[2]
        ):
            return False
        os.replace(temp_path, local_path)
        temp_path = None
        _fsync_cache_directory(os.path.dirname(local_path))
        return True
    finally:
        os.close(source_descriptor)
        if temp_path is not None:
            with suppress(OSError):
                os.remove(temp_path)


def _touch_pyramid_access_marker(path: str) -> None:
    """Record cache recency without mutating the scientific source file's stat identity."""
    marker = _pyramid_access_marker(path)
    descriptor = os.open(marker, os.O_CREAT | os.O_WRONLY, 0o600)
    os.close(descriptor)
    os.utime(marker, None)


def _evict_pyramid_cache(incoming: int) -> None:
    """LRU-evict cached pyramids by access time to keep the cache under budget."""
    try:
        budget = _pyramid_cache_budget_bytes()
        entries: list[tuple[float, int, str]] = []
        total = 0
        cache_directories = (
            _PYRAMID_CACHE_DIR,
            os.path.join(_PYRAMID_CACHE_DIR, "derived"),
        )
        for directory in cache_directories:
            try:
                names = os.listdir(directory)
            except OSError:
                continue
            for name in names:
                if not name.endswith(".tif"):
                    continue
                fp = os.path.join(directory, name)
                try:
                    s = os.stat(fp)
                except OSError:
                    continue
                try:
                    recency = os.stat(_pyramid_access_marker(fp)).st_mtime
                except OSError:
                    recency = s.st_atime
                entries.append((recency, s.st_size, fp))
                total += s.st_size
        entries.sort()  # least-recently-accessed first
        while total + incoming > budget and entries:
            _, sz, fp = entries.pop(0)
            try:
                os.remove(fp)
                total -= sz
            except OSError:
                pass
            with suppress(OSError):
                os.remove(_pyramid_access_marker(fp))
    except OSError:
        pass


def localize_pyramid(path: str) -> str:
    """Return a local-disk copy of a derived pyramid (see module note); the original
    path for anything else or on any error (so reads never hard-fail)."""
    if not _PYRAMID_CACHE_ENABLED or not _is_derived_pyramid(path):
        return path
    try:
        source_identity = _regular_file_identity(path)
    except OSError:
        return path
    source_size = source_identity[2]
    budget = _pyramid_cache_budget_bytes()
    if source_size <= 0 or source_size > budget:
        return path
    key = hashlib.sha256(f"{path}|{source_identity}".encode()).hexdigest()
    local = os.path.join(
        _PYRAMID_CACHE_DIR,
        "derived",
        f"{key}__pyramid.tif",
    )
    try:
        if _cached_pyramid_ready(local, source_size):
            with suppress(OSError):
                # Keep LRU recency in a companion marker. Even an atime-only utime changes
                # ctime on the TIFF itself, which invalidates exact Mask plans while
                # another request is decoding the same warm localized source.
                _touch_pyramid_access_marker(local)
            return local
        with _pyramid_cache_flight(local):
            if _cached_pyramid_ready(local, source_size):
                with suppress(OSError):
                    _touch_pyramid_access_marker(local)
                return local
            with _PYRAMID_CACHE_MUTATION_LOCK:
                if _cached_pyramid_ready(local, source_size):
                    with suppress(OSError):
                        _touch_pyramid_access_marker(local)
                    return local
                if _regular_file_identity(path) != source_identity:
                    return path
                os.makedirs(os.path.dirname(local), exist_ok=True)
                _evict_pyramid_cache(source_size)
                if not _copy_stable_pyramid(path, local, source_identity):
                    return path
                with suppress(OSError):
                    _touch_pyramid_access_marker(local)
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

    return str(await run_in_threadpool(localize_pyramid, path))


def _parse_fusion_request(channels: str | None, channel_colors: str | None):
    """Parse the multi-channel fusion query params shared by the image endpoints.

    ``channels`` is a comma list of 0-based channel indices (the engine's -remap
    is 1-based, so they are shifted). ``channel_colors`` is a comma list of hex LUT
    colors already projected into the same selected-channel order.

    Additive fusion is enabled ONLY for genuine multi-channel composites (2+
    selected channels with at least one color): single-channel and grayscale
    views keep the fast native display path. Composite channel/color cardinality
    must match exactly. Returns (remap_channels_1based | None, selected_colors | None).
    """
    from ultra_deepagents.imaging import fusion

    requested: list[int] | None = None
    if channels is not None:
        parts = channels.split(",")
        if not parts or any(not part.strip() for part in parts):
            raise ValueError("channel selection must be a comma-separated integer list")
        try:
            requested = [int(part.strip()) for part in parts]
        except ValueError as exc:
            raise ValueError("channel selection must contain only integers") from exc
        if any(channel < 0 for channel in requested):
            raise ValueError("channel selection indices must be nonnegative")
        if len(set(requested)) != len(requested):
            raise ValueError("channel selection must not contain duplicates")
        if len(requested) > MAX_COMPOSITE_CHANNELS:
            raise ValueError(
                f"channel selection supports at most {MAX_COMPOSITE_CHANNELS} channels"
            )
    selected_colors: list | None = None
    if channel_colors is not None:
        color_parts = channel_colors.split(",")
        if requested is None or any(not part.strip() for part in color_parts):
            raise ValueError("channel colors require an explicit channel selection")
        normalized_colors = [part.strip().removeprefix("#") for part in color_parts]
        if any(
            len(color) != 6 or any(character not in "0123456789abcdefABCDEF" for character in color)
            for color in normalized_colors
        ):
            raise ValueError("channel colors must be six-digit hexadecimal colors")
        selected_colors = [fusion.parse_hex_color(color) for color in normalized_colors]
        if any(color is None for color in selected_colors):
            raise ValueError("channel colors must be valid hexadecimal colors")

    remap = [c + 1 for c in requested] if requested else None
    if (
        requested is not None
        and selected_colors is not None
        and len(selected_colors) != len(requested)
    ):
        raise ValueError("channel colors must match the selected channel count")
    fuse = (
        requested is not None
        and len(requested) >= 1
        and selected_colors is not None
        and any(color is not None for color in selected_colors)
    )
    if not fuse:
        return remap, None
    return remap, selected_colors


def _reject_repeated_selectors(request: Any, selectors: tuple[str, ...]) -> None:
    """Reject repeated scientific selectors before localization or decoder work.

    FastAPI otherwise keeps one value for scalar query parameters, which would make
    ``t=0&t=1`` or two channel lists silently ambiguous.
    """
    query = request.query_params
    for selector in selectors:
        if len(query.getlist(selector)) > 1:
            raise ValueError(f"repeated {selector} selector is not allowed")


def _validate_tile_size(size: int) -> int:
    if isinstance(size, bool) or size <= 0 or size > MAX_TILE_EDGE:
        raise ValueError(f"tile size must be between 1 and {MAX_TILE_EDGE} pixels")
    return size


async def _validate_channel_range(runner: Any, path: str, remap: list[int] | None) -> None:
    if remap is None:
        return
    meta = await runner.call("meta", path)
    channel_count = int(meta.get("image_num_c", 1) or 1)
    if any(channel < 1 or channel > channel_count for channel in remap):
        raise ValueError(f"channel selection is out of range for source C={channel_count}")


def _parse_histogram_channels(
    *,
    channel: int | None,
    channels: str | None,
    scope: str,
) -> list[int]:
    if channel is not None and channels is not None:
        raise ValueError("histogram channel selectors are ambiguous")
    if channels is not None:
        parts = channels.split(",")
        if not parts or any(part.strip() == "" for part in parts):
            raise ValueError("histogram channels must be a comma-separated integer list")
        try:
            selected = [int(part) for part in parts]
        except ValueError as exc:
            raise ValueError("histogram channels must be integers") from exc
    else:
        selected = [0 if channel is None else int(channel)]
    if any(value < 0 for value in selected):
        raise ValueError("histogram channel indices must be nonnegative")
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate histogram channel indices are not allowed")
    if scope == "volume" and len(selected) != 1:
        raise ValueError("volume histogram requires exactly one channel")
    return selected


def create_app(runner: Any = None, *, prefer_real: bool = True):
    try:
        from fastapi import FastAPI, HTTPException, Request, Response
    except Exception as exc:  # pragma: no cover - exercised only without fastapi
        raise RuntimeError(
            "image service requires fastapi/uvicorn (install the 'imaging' extra)"
        ) from exc
    # Endpoint annotations are postponed by ``from __future__ import annotations``;
    # expose the lazily imported request type so FastAPI can resolve it without making
    # FastAPI/Starlette an import-time dependency of this module.
    globals()["Request"] = Request

    if runner is None:
        from ultra_deepagents.imaging.pool import ImagePool

        runner = ImagePool(prefer_real=prefer_real)

    app = FastAPI(title="Ultra Image Service", version="0.1.0")
    app.state.runner = runner
    scalar_volume_budget = _WeightedByteBudget(_scalar_volume_inflight_budget_bytes())
    app.state.scalar_volume_budget = scalar_volume_budget

    from fastapi.responses import JSONResponse

    class _ScalarBudgetResponse(Response):
        def __init__(self, *, budget: _WeightedByteBudget, weight: int, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._budget = budget
            self._weight = weight

        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            try:
                await super().__call__(scope, receive, send)
            finally:
                await self._budget.release(self._weight)

    # The engine raises ValueError for inputs it cannot decode or render — most often a
    # malformed/unsupported file that libbioimage reads as a 0-sized region (after the
    # engine's own transient-retry). That is a property of the FILE, not a server fault,
    # so map it to 422 instead of a 500: the client shows a clean "preview unavailable"
    # and monitoring doesn't count an undecodable upload as a server error. Genuine bugs
    # (any other exception) keep the default 500.
    decode_error_markers = (
        "empty region",
        "cannot encode",
        "cannot decode",
        "unsupported",
        "out of range",
        "channel selection",
        "channel colors",
        "tile size",
        "duplicate channel",
        "repeated",
        "multiple scenes",
        "source plane input",
        "source generation",
        "decode work does not match",
        "read count does not match",
    )

    @app.exception_handler(ValueError)
    async def _engine_value_error(_request, exc: ValueError) -> Any:
        message = str(exc)
        if any(marker in message.lower() for marker in decode_error_markers):
            return JSONResponse(
                status_code=422,
                content={"error": "image could not be decoded or rendered", "detail": message},
            )
        return JSONResponse(status_code=500, content={"error": "internal error", "detail": message})

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"status": "ok", "workers": runner.workers}

    @app.get("/formats")
    async def formats() -> dict[str, Any]:
        return {"formats": await runner.call("formats")}

    @app.get("/meta")
    async def meta(path: str) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            await runner.call("meta", await _localize_pyramid_async(path)),
        )

    @app.get("/tile")
    async def tile(
        request: Request,
        path: str,
        level: int = 0,
        col: int = 0,
        row: int = 0,
        size: int = 512,
        t: int = 0,
        z: int = 0,
        channels: str | None = None,
        channel_colors: str | None = None,
    ):
        _reject_repeated_selectors(
            request,
            ("level", "col", "row", "size", "t", "z", "channels", "channel_colors"),
        )
        if min(level, col, row, t, z) < 0:
            raise HTTPException(
                status_code=422,
                detail="tile level, coordinates, time, and z must be nonnegative",
            )
        size = _validate_tile_size(size)
        remap, fuse_colors = _parse_fusion_request(channels, channel_colors)
        path = await _localize_pyramid_async(path)
        await _validate_channel_range(runner, path, remap)
        png = await runner.call(
            "tile",
            path,
            level=level,
            col=col,
            row=row,
            tile_size=size,
            t=t,
            z=z,
            channels=remap,
            colors=fuse_colors,
        )
        return Response(content=png, media_type=_PNG)

    @app.get("/slice")
    async def slice_plane(
        request: Request,
        path: str,
        z: int | None = None,
        t: int | None = None,
        level: int | None = None,
        channels: str | None = None,
        channel_colors: str | None = None,
        full_resolution: bool = True,
        scalar_render_mode: str = "intensity",
        scalar_threshold_value: float | None = None,
        scalar_threshold_foreground: str = "above",
    ):
        _reject_repeated_selectors(
            request,
            (
                "z",
                "t",
                "level",
                "channels",
                "channel_colors",
                "full_resolution",
                "scalar_render_mode",
                "scalar_threshold_value",
                "scalar_threshold_foreground",
            ),
        )
        if any(value is not None and value < 0 for value in (z, t, level)):
            raise HTTPException(
                status_code=422,
                detail="slice level, time, and z must be nonnegative",
            )
        remap, fuse_colors = _parse_fusion_request(channels, channel_colors)
        path = await _localize_pyramid_async(path)
        await _validate_channel_range(runner, path, remap)
        # full_resolution=False (transient z-scrub frame) lets the engine pick a
        # bounded pyramid level; the settled view (True) reads the native plane so
        # pixel measurements stay exact. An explicit level always wins.
        png = await runner.call(
            "slice_plane",
            path,
            z=z,
            t=t,
            level=level,
            channels=remap,
            colors=fuse_colors,
            full_resolution=full_resolution,
            scalar_render_mode=scalar_render_mode,
            scalar_threshold_value=scalar_threshold_value,
            scalar_threshold_foreground=scalar_threshold_foreground,
        )
        return Response(content=png, media_type=_PNG)

    @app.get("/thumbnail")
    async def thumbnail(
        path: str,
        max_size: int = 256,
        z: int | None = None,
        level: int | None = None,
        channels: str | None = None,
        channel_colors: str | None = None,
    ):
        remap, fuse_colors = _parse_fusion_request(channels, channel_colors)
        path = await _localize_pyramid_async(path)
        await _validate_channel_range(runner, path, remap)
        png = await runner.call(
            "thumbnail",
            path,
            max_size=max_size,
            z=z,
            level=level,
            channels=remap,
            colors=fuse_colors,
        )
        return Response(content=png, media_type=_PNG)

    @app.get("/atlas")
    async def atlas(
        request: Request,
        path: str,
        grid_rows: int | None = None,
        grid_cols: int | None = None,
        level: int | None = None,
        scale: float | None = None,
        t: int = 0,
        channels: str | None = None,
        channel_colors: str | None = None,
    ):
        _reject_repeated_selectors(
            request,
            ("grid_rows", "grid_cols", "level", "scale", "t", "channels", "channel_colors"),
        )
        if (level is not None and level < 0) or t < 0:
            raise HTTPException(
                status_code=422,
                detail="atlas level and time must be nonnegative",
            )
        if (grid_rows is None) != (grid_cols is None):
            raise HTTPException(
                status_code=422,
                detail="atlas grid rows and columns must be supplied together",
            )
        if (
            grid_rows is not None
            and grid_cols is not None
            and (
                grid_rows < 1
                or grid_cols < 1
                or grid_rows > MAX_ATLAS_GRID_EDGE
                or grid_cols > MAX_ATLAS_GRID_EDGE
                or grid_rows * grid_cols > MAX_ATLAS_CELLS
            )
        ):
            raise HTTPException(
                status_code=422,
                detail=f"atlas grid must contain 1 to {MAX_ATLAS_CELLS} bounded cells",
            )
        if scale is not None and (not math.isfinite(scale) or scale <= 0 or scale > 1):
            raise HTTPException(
                status_code=422,
                detail="atlas scale must be greater than 0 and at most 1",
            )
        remap, fuse_colors = _parse_fusion_request(channels, channel_colors)
        path = await _localize_pyramid_async(path)
        await _validate_channel_range(runner, path, remap)
        if getattr(runner, "workers", 1) > 1:
            # Multiple worker processes: fan the per-plane reads out across the pool
            # (the binding's global lock serializes them within one worker, so this is
            # the only way to parallelize a deep stack). Byte-identical to engine.atlas().
            from ultra_deepagents.imaging.atlas import assemble_atlas

            png = await assemble_atlas(
                runner, path, channels=remap, colors=fuse_colors, level=level, t=t
            )
        else:
            # Single process (InlineRunner / externally-managed): the sequential path.
            grid = (grid_rows, grid_cols) if grid_rows and grid_cols else None
            png = await runner.call(
                "atlas",
                path,
                grid=grid,
                level=level,
                atlas_scale=scale,
                channels=remap,
                colors=fuse_colors,
                t=t,
            )
        return Response(content=png, media_type=_PNG)

    @app.get("/histogram")
    async def histogram(
        path: str,
        bins: int = 256,
        channel: int | None = None,
        channels: str | None = None,
        t: int = 0,
        scope: str = "display",
    ) -> dict[str, Any]:
        normalized_scope = str(scope).strip().lower()
        if normalized_scope not in {"display", "volume"}:
            raise ValueError("unsupported histogram scope")
        selected = _parse_histogram_channels(
            channel=channel,
            channels=channels,
            scope=normalized_scope,
        )
        return cast(
            dict[str, Any],
            await runner.call(
                "histogram",
                await _localize_pyramid_async(path),
                bins=bins,
                channels=[value + 1 for value in selected],
                t=t,
                scope=normalized_scope,
            ),
        )

    @app.get("/viewerinfo")
    async def viewerinfo(path: str, name: str | None = None) -> dict[str, Any]:
        # ``name`` (the upload's original filename) lets HDF5-data files be detected
        # when the on-disk blob path has lost its extension; harmless for images.
        return cast(
            dict[str, Any],
            await runner.call(
                "viewer_info",
                await _localize_pyramid_async(path),
                name=name,
            ),
        )

    @app.get("/video-poster")
    async def video_poster(path: str, t: float = 1.0, max_size: int = 512):
        # ffmpeg runs as a bounded subprocess, off the libbioimage decode pool.
        from ultra_deepagents.imaging import video

        try:
            png = await video.extract_poster(path, time_seconds=t, max_size=max_size)
        except video.VideoError as exc:
            return Response(content=str(exc), media_type="text/plain", status_code=415)
        return Response(
            content=png, media_type=_PNG, headers={"Cache-Control": "private, max-age=3600"}
        )

    @app.get("/scalar-volume")
    async def scalar_volume(path: str, channel: int = 0, t: int = 0, sampling: str = "box"):
        path = await _localize_pyramid_async(path)
        from ultra_deepagents.imaging.atlas import (
            SCALAR_MASK_NATIVE_POLICY,
            validate_scalar_plan,
        )

        plan = await runner.call("scalar_plan", path, channel=channel, t=t, sampling=sampling)
        output_bytes = validate_scalar_plan(plan)
        resident_bytes = output_bytes * _SCALAR_VOLUME_RESIDENT_MULTIPLIER
        if not await scalar_volume_budget.try_acquire(resident_bytes):
            raise HTTPException(
                status_code=503,
                detail="scalar-volume process residency budget is currently exhausted",
                headers={"Retry-After": "1"},
            )
        try:
            if (
                getattr(runner, "workers", 1) > 1
                or plan.get("preview_policy") == SCALAR_MASK_NATIVE_POLICY
            ):
                from ultra_deepagents.imaging.atlas import assemble_scalar_volume

                vol = await assemble_scalar_volume(
                    runner,
                    path,
                    channel=channel,
                    t=t,
                    sampling=sampling,
                    plan=plan,
                )
            else:
                vol = await runner.call(
                    "scalar_volume", path, channel=channel, t=t, sampling=sampling
                )
            data, headers = _scalar_volume_envelope(vol)
            return _ScalarBudgetResponse(
                budget=scalar_volume_budget,
                weight=resident_bytes,
                content=data,
                media_type="application/octet-stream",
                headers=headers,
            )
        except BaseException:
            await scalar_volume_budget.release(resident_bytes)
            raise

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
        return JSONResponse(
            status_code=404, content={"error": "dataset not found", "detail": str(exc)}
        )

    @app.get("/hdf5/dataset")
    async def hdf5_dataset(path: str, dataset_path: str, file_id: str = ""):
        try:
            return await runner.call("hdf5_dataset_summary", path, dataset_path, file_id=file_id)
        except Hdf5DatasetNotFound as exc:
            return _not_found(exc)

    @app.get("/hdf5/materials/dashboard")
    async def hdf5_materials_dashboard(path: str, file_id: str = ""):
        try:
            return await runner.call("hdf5_materials_dashboard", path, file_id=file_id)
        except Hdf5DatasetNotFound as exc:
            return _not_found(exc)

    @app.get("/hdf5/preview/slice")
    async def hdf5_slice(
        path: str,
        dataset_path: str,
        axis: str = "z",
        index: int | None = None,
        component: int = 0,
        feature_ids: str | None = None,
    ):
        try:
            png = await runner.call(
                "hdf5_slice_png",
                path,
                dataset_path,
                axis=axis,
                index=index,
                component=component,
                feature_ids=feature_ids,
            )
        except Hdf5DatasetNotFound as exc:
            return _not_found(exc)
        return Response(
            content=png, media_type=_PNG, headers={"Cache-Control": "private, max-age=3600"}
        )

    @app.get("/hdf5/preview/atlas")
    async def hdf5_atlas(
        path: str,
        dataset_path: str,
        enhancement: str | None = None,
        fusion_method: str | None = None,
        negative: str | None = None,
        channels: str | None = None,
        component: int = 0,
        feature_ids: str | None = None,
    ):
        try:
            png = await runner.call(
                "hdf5_atlas_png",
                path,
                dataset_path,
                component=component,
                feature_ids=feature_ids,
            )
        except Hdf5DatasetNotFound as exc:
            return _not_found(exc)
        return Response(
            content=png, media_type=_PNG, headers={"Cache-Control": "private, max-age=3600"}
        )

    @app.get("/hdf5/preview/scalar-volume")
    async def hdf5_scalar_volume(path: str, dataset_path: str, channel: int = 0):
        try:
            vol = await runner.call("hdf5_scalar_volume", path, dataset_path, channel=channel)
        except Hdf5DatasetNotFound as exc:
            return _not_found(exc)
        data, headers = _scalar_volume_envelope(vol)
        return Response(content=data, media_type="application/octet-stream", headers=headers)

    @app.get("/hdf5/preview/histogram")
    async def hdf5_histogram(
        path: str, dataset_path: str, component: int = 0, bins: int = 24, file_id: str = ""
    ):
        try:
            return await runner.call(
                "hdf5_histogram",
                path,
                dataset_path,
                component=component,
                bins=bins,
                file_id=file_id,
            )
        except Hdf5DatasetNotFound as exc:
            return _not_found(exc)

    @app.get("/hdf5/preview/table")
    async def hdf5_table(
        path: str, dataset_path: str, offset: int = 0, limit: int = 12, file_id: str = ""
    ):
        try:
            return await runner.call(
                "hdf5_table",
                path,
                dataset_path,
                offset=offset,
                limit=limit,
                file_id=file_id,
            )
        except Hdf5DatasetNotFound as exc:
            return _not_found(exc)

    return app
