"""Image engine backends.

:class:`ImageEngine` is the interface the image service depends on. Two backends
implement it:

- :class:`LibBioImageEngine` — the real engine over ``libbioimage`` (requires
  ``libimgcnv.so`` + the ``libbioimage`` wheel + numpy + Pillow). It builds read
  pipelines with :mod:`~ultra_deepagents.imaging.pipelines` and encodes results
  to PNG. The binding serializes on a global lock, so one engine instance is used
  per worker process (see the process pool).
- :class:`StubEngine` — a deterministic, dependency-light backend (Pillow only)
  used by tests, CI, and local dev when the native lib is unavailable. It returns
  valid PNGs whose content is a pure function of the request, so the service and
  the Go proxy can be exercised end-to-end without the ``.so``.

Image-returning methods return encoded PNG ``bytes``; ``meta``/``histogram``
return plain dicts; ``formats`` returns the engine's supported format names.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from ultra_deepagents.imaging import fusion, pipelines, viewerinfo

__all__ = ["ImageEngine", "LibBioImageEngine", "StubEngine", "EngineUnavailable", "build_engine"]


def _engine_cache_entries() -> int:
    """Max entries for the per-worker metadata caches (env-overridable)."""
    try:
        value = int(os.environ.get("ULTRA_IMAGE_ENGINE_CACHE_ENTRIES", "256") or "256")
    except ValueError:
        return 256
    return value if value > 0 else 256


class _BoundedDict(dict):
    """A dict that evicts its oldest entry once it exceeds ``max_size`` (insertion
    order). Keeps a worker's metadata caches from growing without bound across many
    distinct files; the per-file working set is tiny, so eviction never thrashes."""

    def __init__(self, max_size: int) -> None:
        super().__init__()
        self._max = max(1, max_size)

    def __setitem__(self, key: Any, value: Any) -> None:
        if key in self:
            super().__delitem__(key)
        super().__setitem__(key, value)
        while len(self) > self._max:
            super().__delitem__(next(iter(self)))

# Target long-edge (px) for a non-full-resolution z-scrub frame: the engine reads
# the smallest pyramid level whose long edge is still >= this, so a transient scrub
# frame stays crisp at typical viewports while decoding/transferring far less than
# a native plane. Sub-target planes stay native (no downscale).
SCRUB_MAX_DIMENSION = 1024


# --- Persistent viewer-info sidecar cache -----------------------------------
# build_viewer_info on a large source is a multi-second cold decode over (often
# remote) NFS, and the engine's per-process caches don't share across the image
# service's worker processes — so a repeat open lands warm only by luck. Persist
# each result to a small JSON sidecar keyed on the source path + stat-stamp, so
# every worker (and a process/container restart, and the convert worker that
# pre-warms it at derivation) reads it instead of re-decoding. Location:
# ULTRA_IMGSVC_VIEWERINFO_CACHE_DIR — point it at a dir both the image service and
# convert worker mount (the convert scratch); set it to a barrel path to also
# survive a node reboot. Best-effort: any error falls through to a recompute.
def _viewerinfo_cache_dir() -> str:
    d = os.environ.get("ULTRA_IMGSVC_VIEWERINFO_CACHE_DIR", "").strip()
    return d or os.path.join(tempfile.gettempdir(), "ultra-viewerinfo-cache")


def _viewerinfo_cache_path(path: str) -> str | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    stamp = f"{st.st_size}:{int(st.st_mtime_ns)}"
    key = hashlib.sha256((str(path) + "|" + stamp).encode("utf-8")).hexdigest()
    return os.path.join(_viewerinfo_cache_dir(), key + ".json")


def read_viewerinfo_sidecar(path: str) -> dict[str, Any] | None:
    cp = _viewerinfo_cache_path(path)
    if not cp:
        return None
    try:
        with open(cp, "rb") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def write_viewerinfo_sidecar(path: str, info: dict[str, Any]) -> None:
    cp = _viewerinfo_cache_path(path)
    if not cp:
        return
    try:
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        tmp = f"{cp}.tmp.{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(info, fh)
        os.replace(tmp, cp)  # atomic publish
    except OSError:
        pass  # best-effort cache; never block serving


class EngineUnavailable(RuntimeError):
    """Raised when a backend's native dependencies are not installed."""


@runtime_checkable
class ImageEngine(Protocol):
    """Decode/convert operations the image service needs from the engine."""

    def formats(self) -> list[str]:
        """Supported format names (as the engine reports them)."""
        ...

    def meta(self, path: str) -> dict[str, Any]:
        """Image metadata dict (dims, channels, resolution, ...)."""
        ...

    def tile(
        self, path: str, *, level: int, col: int, row: int, tile_size: int = 512,
        channels: Sequence[int] | None = None, colors: Any = None, windows: Any = None,
    ) -> bytes:
        """One embedded tile as PNG (bounded read). ``colors`` enables additive
        multi-channel LUT compositing."""
        ...

    def region(
        self, path: str, *, x1: int, y1: int, x2: int, y2: int,
        region_scale: float | None = None, channels: Sequence[int] | None = None,
        colors: Any = None, windows: Any = None,
    ) -> bytes:
        """An arbitrary region of interest as PNG (bounded read)."""
        ...

    def slice_plane(
        self, path: str, *, z: int | None = None, t: int | None = None,
        level: int | None = None, plane_scale: float | None = None,
        channels: Sequence[int] | None = None, colors: Any = None, windows: Any = None,
        full_resolution: bool = True,
    ) -> bytes:
        """A 2D plane (z/t) as PNG. ``full_resolution=False`` may read a bounded level."""
        ...

    def thumbnail(
        self, path: str, *, max_size: int = 256, z: int | None = None,
        level: int | None = None, channels: Sequence[int] | None = None,
        colors: Any = None, windows: Any = None,
    ) -> bytes:
        """A thumbnail (or one z-scrub frame when ``z`` is set) as PNG."""
        ...

    def atlas(
        self, path: str, *, grid: tuple[int, int] | None = None,
        level: int | None = None, atlas_scale: float | None = None,
        channels: Sequence[int] | None = None, colors: Any = None, windows: Any = None,
    ) -> bytes:
        """A texture atlas (z-stack -> 2D grid) as PNG. ``colors`` enables
        additive multi-channel LUT compositing of the stack."""
        ...

    def histogram(
        self, path: str, *, bins: int = 256, channels: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        """Per-channel intensity histogram."""
        ...

    def viewer_info(self, path: str) -> dict[str, Any]:
        """Structured viewer metadata (axis_sizes, channels, spacing, tile_scheme)."""
        ...

    def scalar_volume(self, path: str, *, channel: int = 0, t: int = 0) -> dict[str, Any]:
        """Raw float volume bytes + dims/dtype/min/max for client-side rendering."""
        ...


class LibBioImageEngine:
    """Real engine over the ``libbioimage`` binding.

    One instance per worker process. Construction fails with
    :class:`EngineUnavailable` if the native library or numpy/Pillow are missing,
    so callers can fall back to :class:`StubEngine` cleanly.
    """

    def __init__(self, *, cache_size: int = 4) -> None:
        try:  # lazy, so importing this module never requires the native stack
            import libbioimage.libbioimage as bim
            import numpy  # noqa: F401
            from PIL import Image  # noqa: F401
        except Exception as exc:  # pragma: no cover - exercised only without the lib
            raise EngineUnavailable(
                "libbioimage engine unavailable: install libimgcnv.so + the "
                "libbioimage wheel (and numpy, Pillow). See "
                "planning/2026-06-11-libbioimage-engine-understanding.md."
            ) from exc
        self._np = numpy
        self._Image = Image
        self._bim = bim
        self._cache = bim.Cache(size=cache_size)
        # Per-worker metadata caches keyed by (path, mtime, channels). Bounded so a
        # long-lived worker that has served many DISTINCT files cannot grow its RSS
        # without limit (insertion-order eviction; the working set is tiny per file).
        self._signal_score_cache = _BoundedDict(_engine_cache_entries())
        # One robust global display window per (path, mtime, channels) so tiled scalar
        # reads share a single mapping instead of auto-scaling per tile (checkerboard).
        self._display_window_cache = _BoundedDict(_engine_cache_entries())

    # -- public API -----------------------------------------------------------------
    def formats(self) -> list[str]:
        fmts = getattr(self._bim, "formats", {})
        try:
            return sorted(fmts.keys()) if hasattr(fmts, "keys") else list(fmts)
        except Exception:  # pragma: no cover
            return []

    def meta(self, path: str) -> dict[str, Any]:
        return dict(self._bim.meta(path, self._cache))

    def _is_display_photo(self, path: str) -> bool:
        """True for an 8-bit RGB(A) PHOTO (orthomosaic, slide): already display-ready,
        so it uses FULL type-range (data-range would collapse a constant fully-opaque
        alpha channel to 0 and render a native tile blank). Everything else is
        scientific scalar data (>8-bit / float / single-channel) that needs a consistent
        GLOBAL window — letting libbioimage data-range each TILE independently is what
        checkerboards a tiled scalar image. Best-effort: a meta failure is treated as
        scalar (the safe, windowed path)."""
        try:
            meta = self._bim.meta(path, self._cache)
        except Exception:  # noqa: BLE001
            return False
        pf = str(meta.get("image_pixel_format", "")).lower()
        depth = int(meta.get("image_pixel_depth", 8) or 8)
        c = int(meta.get("image_num_c", 1) or 1)
        mode = str(meta.get("image_mode") or meta.get("ColorProfile/color_space") or "").strip().lower()
        names = [str(meta.get(f"channels/channel:{i}/name", "")).strip().lower() for i in range(3)]
        rgb_named = names == ["red", "green", "blue"]
        return ("unsigned" in pf or pf == "") and depth <= 8 and c in (3, 4) and (mode.startswith("rgb") or rgb_named)

    def _display_out_depth(self, path: str) -> str:
        """The display ``-depth`` for a non-fused read: FULL type-range for an 8-bit RGB
        photo, else data-range. (Tiled scalar reads no longer use this — they go through
        the global-window path; this remains for the full-extent slice/thumbnail reads,
        where a single read already data-ranges over the whole image.)"""
        return pipelines.DEPTH_DISPLAY_8U_FULLRANGE if self._is_display_photo(path) else pipelines.DEPTH_DISPLAY_8U

    def _display_global_windows(self, path: str, channels=None) -> list[tuple[float, float]]:
        """One robust [p1,p99] window per (read) channel, sampled over a downsampled
        level of the WHOLE image, cached per (path, mtime, channels). Applied to every
        tile/region so a tiled scalar image shares a single mapping instead of
        auto-scaling per tile."""
        np = self._np
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        key = (path, mtime, tuple(channels) if channels else None)
        cached = self._display_window_cache.get(key)
        if cached is not None:
            return cached
        try:
            level = _histogram_level_for_meta(dict(self._bim.meta(path, self._cache)))
        except Exception:  # noqa: BLE001 - bounded fallback, mirrors histogram()
            level = 2
        pipeline = pipelines.build(
            pipelines.res_level(level) if level is not None else None,
            pipelines.remap(channels) if channels else None,
            pipelines.depth(32, "D", "F"),
        )
        arr = self._bim.read(path, pipeline, self._cache)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        windows = [_robust_window(arr[c].ravel(), np) for c in range(arr.shape[0])]
        self._display_window_cache[key] = windows
        return windows

    def tile(self, path, *, level, col, row, tile_size=512, channels=None, colors=None, windows=None) -> bytes:
        if colors:
            return self._render_fused(
                path,
                pipelines.tile(tile_size, col, row, level, channels=channels, out_depth=pipelines.DEPTH_SCALAR_F32),
                colors,
                windows,
            )
        if not self._is_display_photo(path):
            # Scalar/scientific: apply ONE global window to every tile (per-tile
            # data-range would checkerboard the image).
            return self._render_windowed(
                path,
                pipelines.tile(tile_size, col, row, level, channels=channels, out_depth=pipelines.DEPTH_SCALAR_F32),
                self._display_global_windows(path, channels),
            )
        return self._render(
            path,
            pipelines.tile(tile_size, col, row, level, channels=channels, out_depth=self._display_out_depth(path)),
        )

    def region(self, path, *, x1, y1, x2, y2, region_scale=None, channels=None, colors=None, windows=None) -> bytes:
        if colors:
            return self._render_fused(
                path,
                pipelines.region(
                    x1, y1, x2, y2, region_scale=region_scale, channels=channels,
                    out_depth=pipelines.DEPTH_SCALAR_F32,
                ),
                colors,
                windows,
            )
        if not self._is_display_photo(path):
            return self._render_windowed(
                path,
                pipelines.region(
                    x1, y1, x2, y2, region_scale=region_scale, channels=channels,
                    out_depth=pipelines.DEPTH_SCALAR_F32,
                ),
                self._display_global_windows(path, channels),
            )
        return self._render(
            path,
            pipelines.region(
                x1, y1, x2, y2, region_scale=region_scale, channels=channels,
                out_depth=self._display_out_depth(path),
            ),
        )

    def slice_plane(self, path, *, z=None, t=None, level=None, plane_scale=None, channels=None, colors=None, windows=None, full_resolution=True) -> bytes:
        meta = self._bim.meta(path, self._cache)
        paged = z is not None and t is None and viewerinfo.paged_depth(meta)
        # A transient z-scrub frame (full_resolution=False) only needs to fill the
        # viewport. Two complementary bounds keep it fast: read a bounded pyramid
        # level when one exists, AND cap the long edge. The level alone is a no-op on
        # a non-pyramidal source (a flat z-stack stays at level 0, so the full plane
        # is decoded/encoded/transferred — measured no faster than full_resolution);
        # the long-edge cap makes scrub fast for *any* source (a 2048² plane drops
        # from ~4MB/0.47s to ~0.8MB/0.13s). The settled view (full_resolution=True)
        # reads native, so measurements/pixel readouts stay on the true plane grid.
        scrub_max_dim = None
        if not full_resolution:
            if level is None:
                try:
                    level = _thumbnail_level_for_meta(dict(meta), SCRUB_MAX_DIMENSION)
                except Exception:  # noqa: BLE001 - level selection is best-effort; native is the safe fallback
                    level = None
            scrub_max_dim = SCRUB_MAX_DIMENSION
        out_depth = pipelines.DEPTH_SCALAR_F32 if colors else self._display_out_depth(path)
        # A paged z-stack (plain multi-page TIFF, etc.) addresses planes by page,
        # not by -slice z:N (which would return the first plane every time). When a
        # bare z is requested and the file is page-based, read that page directly.
        if paged:
            pipeline = pipelines.page_plane(z, level=level, plane_scale=plane_scale, channels=channels, out_depth=out_depth, max_dim=scrub_max_dim)
        else:
            pipeline = pipelines.slice_plane(z=z, t=t, level=level, plane_scale=plane_scale, channels=channels, out_depth=out_depth, max_dim=scrub_max_dim)
        if colors:
            return self._render_fused(path, pipeline, colors, windows)
        return self._render(path, pipeline)

    def thumbnail(self, path, *, max_size=256, z=None, level=None, channels=None, colors=None, windows=None) -> bytes:
        if level is None:
            try:
                level = _thumbnail_level_for_meta(dict(self._bim.meta(path, self._cache)), max_size)
            except Exception:  # noqa: BLE001 - metadata failure falls back to direct thumbnail behavior
                level = None
        # A thumbnail of a valid image always has content, so an empty (0,0,0) region is
        # a transient engine hiccup (observed on large pyramids under concurrent mixed
        # load), never a real out-of-range edge tile. Retry the read a bounded number of
        # times before surfacing the error, and on the last attempt drop the computed
        # level so a bad level selection also self-heals — otherwise one decode glitch
        # leaves a broken thumbnail in the Resources grid.
        attempts = [level, level, None] if level is not None else [None, None]
        last_exc: ValueError | None = None
        for lv in attempts:
            try:
                if colors:
                    return self._render_fused(
                        path,
                        pipelines.thumbnail(max_size, z=z, level=lv, channels=channels, out_depth=pipelines.DEPTH_SCALAR_F32),
                        colors,
                        windows,
                    )
                return self._render(
                    path,
                    pipelines.thumbnail(max_size, z=z, level=lv, channels=channels, out_depth=self._display_out_depth(path)),
                )
            except ValueError as exc:
                if "empty region" not in str(exc):
                    raise
                last_exc = exc
        raise last_exc  # all attempts came back empty

    def atlas(self, path, *, grid=None, level=None, atlas_scale=None, channels=None, colors=None, windows=None) -> bytes:
        """Assemble a z-stack texture atlas (the 3D slice_stack volume source).

        The libbioimage ``read()`` binding ignores the write-time ``-textureatlas``
        directive and returns a single decoded plane, so the grid is assembled
        here from per-plane reads. Layout matches :func:`viewerinfo.atlas_layout`
        (and the ``atlas_scheme`` the frontend decodes): a ``columns x rows`` grid
        of ``cell_w x cell_h`` cells, plane ``z`` at ``(col=z%cols, row=z//cols)``,
        row-major. Planes are read at a bounded pyramid level for speed and a
        single global per-channel window is applied to every plane so brightness
        is consistent through depth (no per-slice flicker).
        """
        # Sequential reference: build the plan + cells with the SAME helpers the parallel
        # orchestrator (imaging/atlas.py) uses, so both produce byte-identical output.
        from ultra_deepagents.imaging.atlas import compose_atlas_png

        plan = self.atlas_plan(path, channels=channels, colors=colors, level=level)
        if windows is None:
            windows = self.atlas_windows(
                path, depth=plan["depth"], level=plan["read_level"],
                channels=plan["read_channels"], paged=plan["paged"],
            )
        cells = [
            self.atlas_cell(
                path, z=z, level=plan["read_level"], channels=plan["read_channels"],
                colors=plan["cell_colors"], windows=windows,
                cell_w=plan["cell_w"], cell_h=plan["cell_h"], paged=plan["paged"],
            )
            for z in range(plan["depth"])
        ]
        return compose_atlas_png(cells, plan)

    def atlas_plan(self, path, *, channels=None, colors=None, level=None) -> dict:
        """Layout + channel/level decisions for an atlas — the serializable plan the
        parallel orchestrator fans out from (one engine.meta read; no decode)."""
        meta = dict(self._bim.meta(path, self._cache))
        paged = bool(viewerinfo.paged_depth(meta))
        depth = viewerinfo.paged_depth(meta) or int(meta.get("image_num_z", 1) or 1)
        width = int(meta.get("image_num_x", 0) or 0)
        height = int(meta.get("image_num_y", 0) or 0)
        lay = viewerinfo.atlas_layout(width, height, depth)
        # Channels are 1-based (the service shifts the 0-based request via -remap).
        # Fusion needs >=1 selected channel + a color; otherwise render channel 0 as
        # grayscale (white LUT) so a single-channel z-stack still volumes.
        if colors and channels:
            read_channels = list(channels)
            # Preserve None entries (a selected channel with no assigned LUT color
            # contributes nothing in composite_channels) — do NOT tuple(None).
            cell_colors = [tuple(c) if c is not None else None for c in colors]
        else:
            read_channels = [channels[0]] if channels else [1]
            cell_colors = [(1.0, 1.0, 1.0)]
        read_level = level if level is not None else self._atlas_read_level(meta, lay["downsample"])
        return {
            "depth": depth,
            "columns": lay["columns"],
            "rows": lay["rows"],
            "cell_w": lay["cell_w"],
            "cell_h": lay["cell_h"],
            "read_level": read_level,
            "read_channels": read_channels,
            "cell_colors": cell_colors,
            "paged": paged,
        }

    def atlas_windows(self, path, *, depth, level, channels, paged):
        """The one global per-channel window for the whole stack (see _atlas_global_windows)."""
        return self._atlas_global_windows(path, depth, level, channels, paged)

    def atlas_cell(self, path, *, z, level, channels, colors, windows, cell_w, cell_h, paged):
        """Render one atlas cell: read plane z's channels (float), composite with the global
        windows, resize to the cell size. Returns a (cell_h, cell_w, 3) uint8 array."""
        plane = self._atlas_plane_float(path, z, level, channels, paged)
        cell = fusion.composite_channels(plane, colors, np=self._np, windows=windows)
        return self._resize_cell(cell, cell_w, cell_h)

    def atlas_cells(self, path, *, zs, level, channels, colors, windows, cell_w, cell_h, paged):
        """Render a contiguous range of atlas cells in one task — the unit of parallelism
        (one worker reads its chunk of planes sequentially; chunks run across workers)."""
        return [
            self.atlas_cell(
                path, z=z, level=level, channels=channels, colors=colors,
                windows=windows, cell_w=cell_w, cell_h=cell_h, paged=paged,
            )
            for z in zs
        ]

    def _atlas_read_level(self, meta, downsample) -> int:
        """Largest pyramid level whose ~2^L downsample does not exceed the atlas
        cell downsample, so per-plane reads are cheap yet still >= the cell size."""
        levels = int(meta.get("image_num_resolution_levels", 1) or 1)
        if levels <= 1 or downsample <= 1:
            return 0
        return max(0, min(levels - 1, int(math.floor(math.log2(downsample)))))

    def _atlas_plane_float(self, path, z, level, channels, paged):
        """Read one z-plane's selected channels as float (C, H, W)."""
        np = self._np
        if paged:
            pipeline = pipelines.page_plane(
                z, level=level, channels=channels, out_depth=pipelines.DEPTH_SCALAR_F32
            )
        else:
            pipeline = pipelines.slice_plane(
                z=z, level=level, channels=channels, out_depth=pipelines.DEPTH_SCALAR_F32
            )
        arr = self._bim.read(path, pipeline, self._cache)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        return np.ascontiguousarray(arr, dtype="float32")

    def _atlas_global_windows(self, path, depth, level, channels, paged):
        """One robust [p1, p99] window per channel, sampled across depth, so the
        whole volume shares a window (per-plane auto-scaling would flicker in z)."""
        np = self._np
        fractions = (0.2, 0.5, 0.8) if depth >= 3 else (0.0,)
        sample_zs = sorted({max(0, min(depth - 1, int(round(depth * f)))) for f in fractions})
        per_channel: list[list] = []
        for z in sample_zs:
            plane = self._atlas_plane_float(path, z, level, channels, paged)
            if not per_channel:
                per_channel = [[] for _ in range(plane.shape[0])]
            for ci in range(plane.shape[0]):
                per_channel[ci].append(plane[ci].ravel())
        windows = []
        for chunks in per_channel:
            values = np.concatenate(chunks) if chunks else np.zeros(1, dtype="float32")
            lo, hi = (float(v) for v in np.percentile(values, (1.0, 99.0)))
            if not hi > lo:
                hi = lo + 1.0
            windows.append((lo, hi))
        return windows

    def _resize_cell(self, cell, cell_w, cell_h):
        """Force a composited cell to the exact atlas cell size (so cells tile
        seamlessly and match the atlas_scheme the frontend decodes with)."""
        if cell.shape[0] == cell_h and cell.shape[1] == cell_w:
            return cell
        resized = self._Image.fromarray(cell).resize((cell_w, cell_h), self._Image.BILINEAR)
        return self._np.asarray(resized, dtype="uint8")

    def histogram(self, path, *, bins=256, channels=None) -> dict[str, Any]:
        np = self._np
        # Read a downsampled copy to bound memory; histogram of the raw values.
        try:
            level = _histogram_level_for_meta(dict(self._bim.meta(path, self._cache)))
        except Exception:  # noqa: BLE001 - preserve the historic bounded fallback on metadata failures
            level = 2
        arr = self._bim.read(
            path,
            pipelines.build(pipelines.res_level(level) if level is not None else None),
            self._cache,
        )
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        out: dict[str, Any] = {"bins": bins, "channels": []}
        for c in range(arr.shape[0]):
            if channels is not None and c not in channels:
                continue
            counts, edges = np.histogram(arr[c].ravel(), bins=bins)
            out["channels"].append(
                {"index": c, "counts": counts.tolist(), "min": float(edges[0]), "max": float(edges[-1])}
            )
        return out

    def viewer_info(self, path) -> dict[str, Any]:
        cached = read_viewerinfo_sidecar(path)
        if cached is not None:
            return cached
        meta = dict(self._bim.meta(path, self._cache))
        info = viewerinfo.build_viewer_info(meta, signal_scores=self._channel_signal_scores(path, meta))
        write_viewerinfo_sidecar(path, info)
        return info

    def _channel_signal_scores(self, path, meta) -> list[float] | None:
        """Per-channel spatial-structure score (lag-1 horizontal autocorrelation of
        one mid-z plane). Real imagery scores ~0.9+ (neighbouring pixels correlate);
        pure noise scores ~0. Lets the viewer default to channels that actually
        contain structure instead of dead/noise/segmentation channels (e.g. the
        ``_1`` noise variants in AICS OME-TIFFs). Multichannel only; returns ``None``
        on any failure so the caller falls back to the metadata-only default.

        Reads ALL channels in ONE multi-channel decode (the decode dominates cost, so
        one read of N channels ~= one channel; reading them separately was N× slower
        and timed out ``/viewerinfo`` on a non-pyramidal source). Result is cached per
        (path, mtime) so repeated opens are instant.
        """
        try:
            c = int(meta.get("image_num_c", 1) or 1)
            if c <= 1:
                return None
            try:
                cache_key = (path, os.path.getmtime(path), c)
            except OSError:
                cache_key = (path, None, c)
            if cache_key in self._signal_score_cache:
                return self._signal_score_cache[cache_key]
            # Guard: scoring decodes one full mid-z plane of all channels. On a
            # non-pyramidal source that is a full decode, so cap the work to keep
            # /viewerinfo well under its client timeout — for very large multichannel
            # planes, skip scoring (caller falls back to the metadata-only default)
            # rather than risk a slow open.
            x = int(meta.get("image_num_x", 0) or 0)
            y = int(meta.get("image_num_y", 0) or 0)
            if x * y * c > 12_000_000:
                self._signal_score_cache[cache_key] = None
                return None
            np = self._np
            pages = viewerinfo.paged_depth(meta)
            depth = pages or int(meta.get("image_num_z", 1) or 1)
            mid = max(0, depth // 2)
            chans = list(range(1, c + 1))  # -remap is 1-based
            if pages:
                pipeline = pipelines.page_plane(mid, channels=chans, out_depth=pipelines.DEPTH_SCALAR_F32)
            else:
                pipeline = pipelines.slice_plane(z=mid, channels=chans, out_depth=pipelines.DEPTH_SCALAR_F32)
            arr = np.asarray(self._bim.read(path, pipeline, self._cache), dtype="float32")
            if arr.ndim == 2:  # single plane came back without a channel axis
                arr = arr[None]
            scores: list[float] = []
            for ci in range(min(c, arr.shape[0])):
                s = arr[ci]
                if s.ndim != 2 or min(s.shape) < 4:
                    scores.append(1.0)  # too small to judge -> assume real
                    continue
                m = float(s.mean())
                var = float(((s - m) ** 2).mean())
                if var <= 1e-9:
                    scores.append(0.0)  # flat -> no structure
                    continue
                cov = float(((s[:, :-1] - m) * (s[:, 1:] - m)).mean())
                scores.append(max(0.0, min(1.0, cov / var)))
            while len(scores) < c:  # engine returned fewer channels than expected
                scores.append(1.0)
            self._signal_score_cache[cache_key] = scores
            return scores
        except Exception:
            return None

    def scalar_volume(self, path, *, channel=0, t=0) -> dict[str, Any]:
        # Sequential reference, sharing scalar_planes + build_scalar_volume_dict with the
        # parallel orchestrator (imaging/atlas.py) so both produce identical bytes.
        from ultra_deepagents.imaging.atlas import build_scalar_volume_dict

        plan = self.scalar_plan(path, channel=channel, t=t)
        planes = self.scalar_planes(
            path, zs=list(range(plan["depth"])), channel=plan["channel"], t=plan["t"], pages=plan["pages"]
        )
        return build_scalar_volume_dict(planes, plan["channel"])

    def scalar_plan(self, path, *, channel=0, t=0) -> dict[str, Any]:
        """Depth + addressing for a scalar volume (one engine.meta read; no decode)."""
        meta = self._bim.meta(path, self._cache)
        pages = viewerinfo.paged_depth(meta)  # paged z-stack -> planes are pages
        depth = pages or int(meta.get("image_num_z", 1) or 1)
        return {"depth": depth, "pages": int(pages), "channel": 0 if channel is None else int(channel), "t": int(t or 0)}

    def scalar_planes(self, path, *, zs, channel, t, pages):
        """Read a contiguous range of single-channel float planes (the unit of parallelism)."""
        np = self._np
        out = []
        for z in zs:
            if pages:
                pipeline = pipelines.page_plane(z, out_depth=pipelines.DEPTH_SCALAR_F32, channels=[channel + 1])
            else:
                pipeline = pipelines.scalar_volume_plane(z, t=(t or None), channel=channel)
            arr = self._bim.read(path, pipeline, self._cache)
            if arr.ndim == 3:  # (C,H,W) -> single remapped channel
                arr = arr[0]
            out.append(np.ascontiguousarray(arr, dtype="float32"))
        return out

    # -- internals ------------------------------------------------------------------
    def _render(self, path: str, pipeline: str) -> bytes:
        arr = self._bim.read(path, pipeline, self._cache)
        return self._encode_png(arr)

    def _render_windowed(self, path: str, pipeline: str, windows) -> bytes:
        # Read raw float (the pipeline carries -depth 32,D,F) and apply ONE global
        # per-channel window -> 8-bit, so every tile of a scalar image maps identically.
        # This replaces imgcnv's per-tile -depth 8,D,U, which auto-scales each tile to
        # its own range and checkerboards the image.
        arr = self._bim.read(path, pipeline, self._cache)
        return self._encode_png(_window_to_uint8(arr, windows, self._np))

    def _render_fused(self, path: str, pipeline: str, colors, windows=None) -> bytes:
        # Read the selected channels as raw float (the pipeline carries
        # -depth 32,D,F) and additively composite them with their LUT colors,
        # each channel windowed on its own robust percentile range.
        arr = self._bim.read(path, pipeline, self._cache)
        rgb = fusion.composite_channels(arr, colors, np=self._np, windows=windows)
        buf = io.BytesIO()
        self._Image.fromarray(rgb).save(buf, format="PNG")
        return buf.getvalue()

    def _encode_png(self, arr) -> bytes:
        np = self._np
        a = arr
        # An out-of-range/edge tile or empty region comes back with a zero-size
        # axis; surface a clear error instead of a cryptic Pillow type error.
        # (Found via real-engine validation: a level-0 tile past the grid edge.)
        if 0 in getattr(a, "shape", ()):
            raise ValueError(f"engine returned an empty region (shape {getattr(a, 'shape', None)})")
        if a.ndim == 3:  # (C, H, W) -> (H, W, C)
            a = np.transpose(a, (1, 2, 0))
            if a.shape[2] == 1:
                a = a[:, :, 0]
            elif a.shape[2] > 4:  # keep first three channels for display
                a = a[:, :, :3]
        elif a.ndim != 2:
            raise ValueError(f"cannot encode array with ndim={a.ndim}")
        if a.dtype != np.uint8:  # display pipelines emit 8-bit; guard anyway
            mx = float(a.max()) or 1.0
            a = (a.astype("float32") / mx * 255.0).clip(0, 255).astype("uint8")
        buf = io.BytesIO()
        self._Image.fromarray(a).save(buf, format="PNG")
        return buf.getvalue()


class StubEngine:
    """Deterministic Pillow-only backend for tests/CI without the native lib.

    Output is a pure function of the request, so callers can assert determinism
    and content-type and the Go proxy can be tested end-to-end.
    """

    _FORMATS = ["ome-tiff", "tiff", "czi", "nd2", "dicom", "nifti", "svs", "ndpi", "png", "jpeg"]

    def __init__(self) -> None:
        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover
            raise EngineUnavailable("StubEngine requires Pillow") from exc
        self._Image = Image

    def formats(self) -> list[str]:
        return list(self._FORMATS)

    def meta(self, path: str) -> dict[str, Any]:
        seed = _seed(path)
        return {
            "format": "STUB",
            "image_num_x": 1024,
            "image_num_y": 1024,
            "image_num_z": 1 + seed % 64,
            "image_num_c": 1 + seed % 3,
            "image_num_t": 1,
            "image_pixel_depth": 16,
            "image_pixel_format": "unsigned integer",
        }

    def tile(self, path, *, level, col, row, tile_size=512, channels=None, colors=None, windows=None) -> bytes:
        return self._png(tile_size, tile_size, _seed(path, "tile", level, col, row, channels, colors))

    def region(self, path, *, x1, y1, x2, y2, region_scale=None, channels=None, colors=None, windows=None) -> bytes:
        return self._png(min(x2 - x1, 1024), min(y2 - y1, 1024), _seed(path, "region", x1, y1, x2, y2, colors))

    def slice_plane(self, path, *, z=None, t=None, level=None, plane_scale=None, channels=None, colors=None, windows=None, full_resolution=True) -> bytes:
        return self._png(512, 512, _seed(path, "slice", z, t, level, colors, full_resolution))

    def thumbnail(self, path, *, max_size=256, z=None, level=None, channels=None, colors=None, windows=None) -> bytes:
        return self._png(max_size, max_size, _seed(path, "thumb", z, level, colors))

    def atlas(self, path, *, grid=None, level=None, atlas_scale=None, channels=None, colors=None, windows=None) -> bytes:
        return self._png(512, 512, _seed(path, "atlas", grid, level, channels, colors))

    def histogram(self, path, *, bins=256, channels=None) -> dict[str, Any]:
        seed = _seed(path, "hist", bins)
        counts = [((seed + i * 2654435761) % 997) for i in range(bins)]
        return {"bins": bins, "channels": [{"index": 0, "counts": counts, "min": 0.0, "max": 65535.0}]}

    def viewer_info(self, path) -> dict[str, Any]:
        seed = _seed(path)
        # Deterministic meta WITH a pyramid so the tile path is exercisable in tests.
        meta = {
            "format": "STUB", "image_num_x": 2048, "image_num_y": 2048,
            "image_num_z": 1 + seed % 16, "image_num_c": 1 + seed % 3, "image_num_t": 1,
            "image_pixel_depth": 16, "image_pixel_format": "unsigned integer",
            "image_num_resolution_levels": 4, "image_res_l_scales": "1.0,0.5,0.25,0.125",
            "tile_size_x": "256",
            "pixel_resolution_x": 0.5, "pixel_resolution_y": 0.5, "pixel_resolution_z": 1.0,
        }
        return viewerinfo.build_viewer_info(meta)

    def scalar_volume(self, path, *, channel=0, t=0) -> dict[str, Any]:
        seed = _seed(path, "vol", channel, t)
        w = h = 16
        d = 1 + seed % 8
        data = bytes(bytearray(((seed + i) & 0xFF) for i in range(w * h * d * 4)))
        return {
            "data": data, "width": w, "height": h, "depth": d, "dtype": "float32",
            "bytes_per_voxel": 4, "raw_min": 0.0, "raw_max": 255.0, "channel": channel or 0,
        }

    def _png(self, w: int, h: int, seed: int) -> bytes:
        w = max(1, int(w))
        h = max(1, int(h))
        # Deterministic gradient tinted by the seed; cheap and reproducible.
        r = seed & 0xFF
        g = (seed >> 8) & 0xFF
        b = (seed >> 16) & 0xFF
        data = bytearray(w * h * 3)
        for y in range(h):
            row = (y * 255) // h
            for x in range(w):
                i = (y * w + x) * 3
                data[i] = (r + ((x * 255) // w)) & 0xFF
                data[i + 1] = (g + row) & 0xFF
                data[i + 2] = b
        img = self._Image.frombytes("RGB", (w, h), bytes(data))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def _robust_window(values, np) -> tuple[float, float]:
    """A robust [p1, p99] intensity window for a 1-D value array — the same
    convention the atlas/fusion paths use. Guarantees hi > lo so the mapping never
    divides by zero on a constant region."""
    if getattr(values, "size", 0) == 0:
        return (0.0, 1.0)
    lo, hi = (float(v) for v in np.percentile(values, (1.0, 99.0)))
    if not hi > lo:
        hi = lo + 1.0
    return (lo, hi)


def _window_to_uint8(arr, windows, np):
    """Map a float array — (H,W) or (C,H,W) — to uint8 by applying a per-channel
    [lo,hi] window: ``out = clip((v - lo) / (hi - lo), 0, 1) * 255``.

    The window is supplied by the caller (one GLOBAL window per image), so EVERY tile
    of an image maps identically — this is what kills the per-tile contrast stretch
    (the checkerboard) that libbioimage's ``-depth 8,D,U`` produces on tiled reads.
    Pure + unit-tested. Preserves the input rank (2-D stays 2-D)."""
    single = arr.ndim == 2
    a = arr[np.newaxis, ...] if single else arr
    out = np.empty(a.shape, dtype="uint8")
    for c in range(a.shape[0]):
        lo, hi = windows[c] if c < len(windows) else (windows[-1] if windows else (0.0, 1.0))
        scale = 255.0 / max(float(hi) - float(lo), 1e-6)
        out[c] = np.clip((a[c].astype("float32") - float(lo)) * scale, 0.0, 255.0).astype("uint8")
    return out[0] if single else out


def _seed(*parts: Any) -> int:
    """Stable non-cryptographic seed from request parts (deterministic across runs)."""
    h = 1469598103934665603  # FNV-1a 64-bit
    for token in (str(p) for p in parts):
        for ch in token.encode("utf-8"):
            h ^= ch
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def _meta_int(meta: dict[str, Any], key: str) -> int:
    try:
        return int(float(meta.get(key) or 0))
    except (TypeError, ValueError):
        return 0


def _resolution_level_count(meta: dict[str, Any]) -> int:
    return _meta_int(meta, "image_num_resolution_levels_actual") or _meta_int(meta, "image_num_resolution_levels")


def _resolution_scales(meta: dict[str, Any]) -> list[float]:
    raw = (
        meta.get("image_resolution_level_scales_actual")
        or meta.get("image_res_l_scales_actual")
        or meta.get("image_resolution_level_scales")
        or meta.get("image_res_l_scales")
    )
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        values = []
    scales: list[float] = []
    for value in values:
        try:
            scale = float(value)
        except (TypeError, ValueError):
            continue
        if scale > 0:
            scales.append(scale)
    levels = _resolution_level_count(meta)
    if levels > 0 and len(scales) > levels:
        return scales[:levels]
    return scales


def _thumbnail_level_for_meta(meta: dict[str, Any], max_size: int) -> int | None:
    max_dim = max(_meta_int(meta, "image_num_x"), _meta_int(meta, "image_num_y"))
    if max_dim <= max(1, int(max_size)):
        return None

    scales = _resolution_scales(meta)
    if scales:
        best = 0
        for index, scale in enumerate(scales):
            if max_dim * scale >= max_size:
                best = index
            else:
                break
        return best if best > 0 else None

    levels = _resolution_level_count(meta)
    if levels <= 1:
        return None
    level = 0
    scaled_dim = float(max_dim)
    while level + 1 < levels and scaled_dim / 2 >= max_size:
        scaled_dim /= 2
        level += 1
    return level if level > 0 else None


def _histogram_level_for_meta(meta: dict[str, Any]) -> int | None:
    return _thumbnail_level_for_meta(meta, 1024)


def build_engine(*, prefer_real: bool = True, cache_size: int = 4) -> ImageEngine:
    """Construct the best available engine: the real one if possible, else the stub."""
    import logging

    log = logging.getLogger(__name__)
    if prefer_real:
        try:
            engine = LibBioImageEngine(cache_size=cache_size)
            log.info("imaging: using LibBioImageEngine")
            return engine
        except EngineUnavailable as exc:
            log.warning("imaging: libbioimage unavailable (%s); falling back to StubEngine", exc)
    return StubEngine()
