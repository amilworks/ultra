"""Pure-Python raster engine: tifffile (TIFF family) + bioio (CZI/ND2/LIF/DV).

This is the MIDDLE tier of the engine ladder built by
:func:`ultra_deepagents.imaging.engine.build_engine`:

    LibBioImageEngine  (native .so — fastest, full tiled-pyramid reads)
        -> BioioEngine (this module — real decode, pure Python, no native wheel)
            -> Hdf5OnlyEngine (HDF5 real; rasters refused as 422)

It exists so an environment WITHOUT the ``libbioimage`` wheel still decodes
real pixels instead of failing every raster. It is deliberately NOT a
placeholder generator — every pixel it returns is read from the file (contrast
``StubEngine``, which is tests-only and fabricates plausible images).

Conventions that MUST match the native engine (the service is shared)
---------------------------------------------------------------------
* **Channel indices on this interface are 1-BASED.** ``service._parse_fusion_request``
  shifts the 0-based wire request into libbioimage's ``-remap`` space before
  calling the engine, so every public method here converts 1-based -> 0-based at
  its boundary (:func:`_zero_based`). Getting this wrong renders a neighbouring
  channel — silently, with a 200.
* ``meta()`` emits a **libbioimage-shaped** dict so
  :func:`~ultra_deepagents.imaging.viewerinfo.build_viewer_info` — and therefore
  the whole viewer contract — is shared rather than reimplemented.
* The atlas/scalar-volume **plan/cell** methods mirror the native engine's, so the
  parallel pool orchestrator (:mod:`ultra_deepagents.imaging.atlas`) works here
  too and the sequential path produces identical output.

Design notes
------------
* Reads are BOUNDED. TIFF goes through ``series.aszarr()`` so a tile/region
  decodes only the chunks it overlaps; bioio goes through its dask array.
* Geometry is normalized to ``(T, C, Z, Y, X)``; an interleaved sample axis is
  resolved during the read so a plane is ALWAYS 2-D (no transposed garbage from
  planar/alpha layouts).
* One GLOBAL per-channel window per (path, mtime, level, channel), sampled from a
  bounded centre crop, so every tile of an image maps identically without
  decoding a whole level per tile.
"""

from __future__ import annotations

import bisect
import io
import math
import operator
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn

from ultra_deepagents.imaging import atlas as atlas_mod
from ultra_deepagents.imaging import fusion, scalar_semantics, viewerinfo
from ultra_deepagents.imaging.constants import (
    MAX_COMPOSITE_CHANNELS,
    MAX_VIEWERINFO_SIGNAL_CHANNELS,
)
from ultra_deepagents.imaging.engine import (
    SCRUB_MAX_DIMENSION,
    EngineUnavailable,
    _BoundedDict,
    _engine_cache_entries,
    _Hdf5EngineMixin,
    _robust_window,
    _window_to_uint8,
)

__all__ = ["BioioEngine", "TIFF_EXTENSIONS", "BIOIO_EXTENSIONS", "can_read"]

# TIFF family read directly with tifffile: it handles OME-TIFF, BigTIFF,
# multi-resolution (SubIFD/pyramid) series and the whole-slide TIFF variants,
# and is already a first-class dependency. The bioio tiff plugins are
# deliberately NOT installed (see pyproject: "pure redundancy").
TIFF_EXTENSIONS = (
    ".tif",
    ".tiff",
    ".btf",
    ".ome.tif",
    ".ome.tiff",
    ".svs",
    ".ndpi",
    ".qptiff",
    ".scn",
)

# Formats routed to bioio's plugin readers (installed per pyproject [imaging]).
BIOIO_EXTENSIONS = (".czi", ".nd2", ".lif", ".dv", ".r3d")

# Long edge of the centre crop sampled when deriving a global display window.
# Bounded so a window costs one small read, never a full-level decode.
_WINDOW_SAMPLE_EDGE = 1024
_SOURCE_REGION_MAX_BYTES = 4 * 1024 * 1024
_ATLAS_SOURCE_PLANE_MAX_BYTES = 64 * 1024 * 1024
_VOLUME_DECODED_CHUNK_MAX_BYTES = 64 * 1024 * 1024
_SCALAR_SOURCE_WORK_MAX_BYTES = atlas_mod.SCALAR_DECODE_WORK_MAX_BYTES
_MAX_DECODE_ADMISSION_READS = 4096

# Mirrors service.py's _DECODE_ERROR_MARKERS: a ValueError whose message carries
# one of these maps to 422 "preview unavailable" instead of a 500 server fault.
_DECODE_MARKERS = ("empty region", "cannot encode", "cannot decode", "unsupported")


def _has_decode_marker(message: str) -> bool:
    lowered = str(message).lower()
    return any(marker in lowered for marker in _DECODE_MARKERS)


def _as_decode_error(path: str, exc: BaseException) -> ValueError:
    """Wrap any reader failure so it carries a 422 marker.

    Without this a raw tifffile/bioio error ("not a TIFF file") surfaces as a 500
    server fault, which the control plane never caches — so a single bad file is
    re-decoded on every grid render.
    """
    if isinstance(exc, ValueError) and _has_decode_marker(str(exc)):
        return exc
    return ValueError(f"cannot decode {os.path.basename(path)} (unsupported or corrupt): {exc}")


def _lower_ext(path: str) -> str:
    name = os.path.basename(str(path)).strip().lower()
    # Longest match first so ".ome.tiff" wins over ".tiff".
    for ext in sorted((*TIFF_EXTENSIONS, *BIOIO_EXTENSIONS), key=len, reverse=True):
        if name.endswith(ext):
            return ext
        # Tolerate a numeric series/page suffix ("scan.czi_3"), as the convert lane does.
        marker = ext + "_"
        index = name.rfind(marker)
        if index >= 0 and name[index + len(marker) :].isdigit():
            return ext
    _, dot_ext = os.path.splitext(name)
    return dot_ext


def can_read(path: str) -> bool:
    """True when this engine has a reader for ``path``'s extension."""
    return _lower_ext(path) in (*TIFF_EXTENSIONS, *BIOIO_EXTENSIONS)


def _exact_index(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"scalar volume {field} index is out of range")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"scalar volume {field} index is out of range") from exc


def _scalar_dtype(dtype: Any) -> tuple[str, int, Any]:
    import numpy as np

    source = np.dtype(dtype)
    supported = {
        ("u", 1): ("uint8", np.dtype("u1")),
        ("u", 2): ("uint16", np.dtype("<u2")),
        ("i", 2): ("int16", np.dtype("<i2")),
        ("f", 4): ("float32", np.dtype("<f4")),
    }
    wire = supported.get((source.kind, source.itemsize))
    if wire is None:
        return "float32", 4, np.dtype("<f4")
    return wire[0], wire[1].itemsize, wire[1]


def _max_chunk_bytes(chunks: Any, dtype: Any) -> int:
    """Return the largest decoded chunk implied by Zarr or Dask chunk metadata."""
    import numpy as np

    if not isinstance(chunks, (tuple, list)) or not chunks:
        raise ValueError("decoded chunk geometry is unavailable")
    shape: list[int] = []
    for axis in chunks:
        values = axis if isinstance(axis, (tuple, list)) else (axis,)
        sizes: list[int] = []
        for raw in values:
            if isinstance(raw, bool):
                raise ValueError("decoded chunk geometry is invalid")
            try:
                size = operator.index(raw)
            except TypeError as exc:
                raise ValueError("decoded chunk geometry is invalid") from exc
            if size <= 0:
                raise ValueError("decoded chunk geometry is invalid")
            sizes.append(size)
        shape.append(max(sizes))
    try:
        itemsize = operator.index(np.dtype(dtype).itemsize)
    except (TypeError, ValueError) as exc:
        raise ValueError("decoded chunk geometry is invalid") from exc
    if isinstance(itemsize, bool) or itemsize <= 0:
        raise ValueError("decoded chunk geometry is invalid")
    return int(math.prod(shape)) * itemsize


@dataclass(frozen=True)
class _PreparedChunkAxis:
    size: int
    uniform_chunk_size: int | None
    boundaries: tuple[int, ...] | None
    max_chunk_size: int

    def touched_size(self, start: int, stop: int) -> int:
        if self.uniform_chunk_size is not None:
            chunk_size = self.uniform_chunk_size
            first_chunk = start // chunk_size
            last_chunk = (stop - 1) // chunk_size
            return (last_chunk - first_chunk + 1) * chunk_size

        boundaries = self.boundaries
        if boundaries is None:
            raise ValueError("decoded chunk geometry is invalid")
        first_chunk = bisect.bisect_right(boundaries, start)
        last_chunk = bisect.bisect_left(boundaries, stop)
        first_chunk_start = boundaries[first_chunk - 1] if first_chunk else 0
        return boundaries[last_chunk] - first_chunk_start


@dataclass(frozen=True)
class _PreparedDecodedChunkGeometry:
    axes: tuple[_PreparedChunkAxis, ...]
    itemsize: int
    max_chunk_bytes: int

    def estimate(self, selection: Sequence[Any]) -> int:
        try:
            selection_rank = len(selection)
        except TypeError as exc:
            raise ValueError("decoded selection is invalid") from exc
        if selection_rank != len(self.axes):
            raise ValueError("decoded selection is invalid")

        touched_axis_sizes: list[int] = []
        for axis, selector in zip(self.axes, selection, strict=True):
            if isinstance(selector, bool):
                raise ValueError("decoded selection is invalid")
            if isinstance(selector, slice):
                if any(
                    isinstance(value, bool)
                    for value in (selector.start, selector.stop, selector.step)
                ):
                    raise ValueError("decoded selection is invalid")
                try:
                    start, stop, step = selector.indices(axis.size)
                except (TypeError, ValueError) as exc:
                    raise ValueError("decoded selection is invalid") from exc
                if step != 1 or start >= stop:
                    raise ValueError("decoded selection is invalid")
            else:
                try:
                    index = operator.index(selector)
                except TypeError as exc:
                    raise ValueError("decoded selection is invalid") from exc
                if index < 0 or index >= axis.size:
                    raise ValueError("decoded selection is invalid")
                start, stop = index, index + 1
            touched = axis.touched_size(start, stop)
            if touched <= 0:
                raise ValueError("decoded selection is invalid")
            touched_axis_sizes.append(touched)
        return int(math.prod(touched_axis_sizes)) * self.itemsize


@dataclass(frozen=True)
class _PreparedGeometryFailure:
    message: str

    def raise_error(self) -> NoReturn:
        raise ValueError(self.message)


def _positive_geometry_integer(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("decoded chunk geometry is invalid")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError("decoded chunk geometry is invalid") from exc
    if normalized <= 0:
        raise ValueError("decoded chunk geometry is invalid")
    return normalized


def _prepare_decoded_chunk_geometry(
    shape: Any,
    chunks: Any,
    dtype: Any,
) -> _PreparedDecodedChunkGeometry:
    """Snapshot and validate chunk geometry once for repeated admission estimates."""
    import numpy as np

    if (
        not isinstance(shape, (tuple, list))
        or not shape
        or not isinstance(chunks, (tuple, list))
        or len(chunks) != len(shape)
    ):
        raise ValueError("decoded chunk geometry is invalid")
    axis_sizes = tuple(_positive_geometry_integer(value) for value in shape)
    try:
        itemsize = operator.index(np.dtype(dtype).itemsize)
    except (TypeError, ValueError) as exc:
        raise ValueError("decoded chunk geometry is invalid") from exc
    if isinstance(itemsize, bool) or itemsize <= 0:
        raise ValueError("decoded chunk geometry is invalid")

    axes: list[_PreparedChunkAxis] = []
    for size, raw_chunks in zip(axis_sizes, chunks, strict=True):
        if isinstance(raw_chunks, (tuple, list)):
            if not raw_chunks:
                raise ValueError("decoded chunk geometry is invalid")
            boundaries: list[int] = []
            total = 0
            maximum = 0
            for raw_chunk_size in raw_chunks:
                chunk_size = _positive_geometry_integer(raw_chunk_size)
                total += chunk_size
                maximum = max(maximum, chunk_size)
                boundaries.append(total)
            if total != size:
                raise ValueError("decoded chunk geometry is invalid")
            axes.append(
                _PreparedChunkAxis(
                    size=size,
                    uniform_chunk_size=None,
                    boundaries=tuple(boundaries),
                    max_chunk_size=maximum,
                )
            )
            continue

        chunk_size = _positive_geometry_integer(raw_chunks)
        axes.append(
            _PreparedChunkAxis(
                size=size,
                uniform_chunk_size=chunk_size,
                boundaries=None,
                max_chunk_size=chunk_size,
            )
        )

    frozen_axes = tuple(axes)
    max_chunk_bytes = int(math.prod(axis.max_chunk_size for axis in frozen_axes)) * itemsize
    return _PreparedDecodedChunkGeometry(
        axes=frozen_axes,
        itemsize=itemsize,
        max_chunk_bytes=max_chunk_bytes,
    )


def _decoded_selection_work_bytes(
    shape: Any,
    chunks: Any,
    dtype: Any,
    selection: Sequence[Any],
) -> int:
    """Sum every decoded chunk intersected by an N-D selection.

    Chunk bytes, including smaller edge chunks, are charged in full because
    TIFF/Zarr and Dask decompress the complete intersected chunk even for a
    one-pixel crop.
    """
    return _prepare_decoded_chunk_geometry(shape, chunks, dtype).estimate(selection)


def _zero_based(channels: Sequence[int] | None, count: int) -> list[int]:
    """Convert the interface's 1-BASED channel indices to 0-based array indices.

    ``None`` means every source channel for shared callers such as histograms.
    Pixel-render entry points choose their own bounded defaults before reading.
    Empty, duplicate, oversized, or out-of-range selections are rejected before
    any pixel read.
    """
    if channels is None:
        return list(range(max(count, 1)))
    values = list(channels)
    if not values:
        raise ValueError("channel selection must not be empty")
    if len(values) > MAX_COMPOSITE_CHANNELS:
        raise ValueError(f"channel selection supports at most {MAX_COMPOSITE_CHANNELS} channels")
    out: list[int] = []
    seen: set[int] = set()
    for raw in values:
        channel = _exact_index(raw, "channel")
        if channel < 1 or channel > count:
            raise ValueError(f"channel index {channel} is out of range for C={count}")
        if channel in seen:
            raise ValueError(f"duplicate channel index {channel}")
        seen.add(channel)
        out.append(channel - 1)
    return out


def _axis_index(value: Any, field: str, count: int) -> int:
    index = _exact_index(value, field)
    if index < 0 or index >= count:
        raise ValueError(f"{field} index {index} is out of range for {field.upper()}={count}")
    return index


def _require_single_scene(source: Any) -> None:
    if int(getattr(source, "scene_count", 1) or 1) != 1:
        raise ValueError("multiple scenes require an explicit scene identity for pixel preview")


def _require_bounded_volume_source(source: Any) -> None:
    _require_single_scene(source)
    chunk_bytes = getattr(source, "max_decoded_chunk_bytes", None)
    if chunk_bytes is not None and int(chunk_bytes) > _VOLUME_DECODED_CHUNK_MAX_BYTES:
        raise ValueError("decoded chunk exceeds the bounded volume limit; preview is unsupported")
    source_work = (
        int(source.x)
        * int(source.y)
        * int(source.z)
        * int(getattr(source.dtype, "itemsize", 0) or 0)
    )
    if source_work <= 0 or source_work > _SCALAR_SOURCE_WORK_MAX_BYTES:
        raise ValueError("scalar volume source work exceeds the bounded per-channel limit")


def _exact_plane_regions(source: Any) -> list[tuple[int, int, int, int]]:
    itemsize = max(1, int(getattr(source.dtype, "itemsize", 0) or 0))
    max_voxels = max(1, _SOURCE_REGION_MAX_BYTES // itemsize)
    block_width = min(int(source.x), max_voxels)
    block_height = max(1, min(int(source.y), max_voxels // block_width))
    return [
        (y0, min(int(source.y), y0 + block_height), x0, min(int(source.x), x0 + block_width))
        for y0 in range(0, int(source.y), block_height)
        for x0 in range(0, int(source.x), block_width)
    ]


def _nearest_plane_regions(source: Any, plan: dict[str, Any]) -> list[tuple[int, int, int, int]]:
    import numpy as np

    factor_x = int(plan["downsample_x"])
    factor_y = int(plan["downsample_y"])
    width = int(plan["width"])
    height = int(plan["height"])
    x_indices = np.minimum(
        int(source.x) - 1,
        np.arange(width, dtype="int64") * factor_x + factor_x // 2,
    )
    y_indices = np.minimum(
        int(source.y) - 1,
        np.arange(height, dtype="int64") * factor_y + factor_y // 2,
    )
    itemsize = max(1, int(np.dtype(source.dtype).itemsize))
    max_voxels = max(1, _SOURCE_REGION_MAX_BYTES // itemsize)
    regions: list[tuple[int, int, int, int]] = []
    output_x0 = 0
    while output_x0 < width:
        output_x1 = output_x0 + 1
        while (
            output_x1 < width
            and int(x_indices[output_x1]) - int(x_indices[output_x0]) + 1 <= max_voxels
        ):
            output_x1 += 1
        selected_x = x_indices[output_x0:output_x1]
        x0 = int(selected_x[0])
        x1 = int(selected_x[-1]) + 1
        max_source_height = max(1, max_voxels // (x1 - x0))
        output_y0 = 0
        while output_y0 < height:
            output_y1 = output_y0 + 1
            while (
                output_y1 < height
                and int(y_indices[output_y1]) - int(y_indices[output_y0]) + 1 <= max_source_height
            ):
                output_y1 += 1
            selected_y = y_indices[output_y0:output_y1]
            regions.append((int(selected_y[0]), int(selected_y[-1]) + 1, x0, x1))
            output_y0 = output_y1
        output_x0 = output_x1
    return regions


def _admit_decode_reads(
    source: Any,
    reads: Iterable[tuple[int, int, int, tuple[int, int, int, int] | None]],
    *,
    expected_read_count: int,
    label: str,
    max_work_bytes: int = _SCALAR_SOURCE_WORK_MAX_BYTES,
) -> int:
    if (
        isinstance(expected_read_count, bool)
        or not isinstance(expected_read_count, int)
        or expected_read_count <= 0
        or expected_read_count > _MAX_DECODE_ADMISSION_READS
    ):
        raise ValueError(f"{label} read count exceeds its bounded envelope")
    estimate_read_work = getattr(source, "estimate_read_work", None)
    if not callable(estimate_read_work):
        raise ValueError(f"{label} decoded chunk geometry is unavailable")
    total = 0
    actual_read_count = 0
    for t, channel, z, region in reads:
        actual_read_count += 1
        if actual_read_count > expected_read_count:
            raise ValueError(f"{label} read plan does not match its bounded envelope")
        work = estimate_read_work(t=t, c=channel, z=z, level=0, box=region)
        if isinstance(work, bool) or not isinstance(work, int) or work <= 0:
            raise ValueError(f"{label} decoded chunk geometry is invalid")
        total += work
        if total > max_work_bytes:
            raise ValueError(f"{label} decode work exceeds its bounded envelope")
    if actual_read_count != expected_read_count:
        raise ValueError(f"{label} read plan does not match its bounded envelope")
    return total


def _source_generation(path: str) -> tuple[int, int, int, int, int, int]:
    try:
        stat = os.stat(path)
    except OSError as exc:
        raise ValueError(f"cannot decode {os.path.basename(path)}: {exc}") from exc
    return (
        atlas_mod.SCALAR_SOURCE_GENERATION_VERSION,
        operator.index(stat.st_dev),
        operator.index(stat.st_ino),
        operator.index(stat.st_size),
        operator.index(stat.st_mtime_ns),
        operator.index(stat.st_ctime_ns),
    )


def _require_source_generation(
    path: str,
    expected: Any,
) -> tuple[int, int, int, int, int, int]:
    generation = atlas_mod.validate_scalar_source_generation(expected)
    if _source_generation(path) != generation:
        raise ValueError("exact Mask source generation changed")
    return generation


def _complete_exact_mask_decode_admission(
    source: Any,
    plan: dict[str, Any],
) -> tuple[int, int]:
    """Recompute all reads for the selected C/T across the complete native Z grid."""
    atlas_mod.validate_scalar_plan(plan)
    if str(plan.get("preview_policy", "")).strip().lower() != atlas_mod.SCALAR_MASK_NATIVE_POLICY:
        raise ValueError("complete decoder admission requires an exact Mask plan")
    regions = _nearest_plane_regions(source, plan)
    depth = int(plan["depth"])
    read_count = depth * len(regions)
    work = _admit_decode_reads(
        source,
        (
            (
                int(plan["t"]),
                int(plan["channel"]),
                min(
                    int(source.z) - 1,
                    output_z * int(plan["downsample_z"]) + int(plan["downsample_z"]) // 2,
                ),
                region,
            )
            for output_z in range(depth)
            for region in regions
        ),
        expected_read_count=read_count,
        label="nearest scalar volume",
    )
    return work, read_count


def _bounded_output_indices(zs: Iterable[Any], depth: int) -> list[int]:
    output_indices: list[int] = []
    seen: set[int] = set()
    for position, output_z in enumerate(zs):
        if position >= depth:
            raise ValueError("scalar volume output selection exceeds its bounded envelope")
        index = _exact_index(output_z, "z")
        if index < 0 or index >= depth:
            raise ValueError(f"scalar volume z index {index} is out of range")
        if index in seen:
            raise ValueError(f"duplicate scalar volume z index {index}")
        seen.add(index)
        output_indices.append(index)
    if not output_indices:
        raise ValueError("scalar volume output selection is empty")
    return output_indices


class _Plane:
    """A normalized view of one source: 5-D geometry + bounded plane reads."""

    def __init__(
        self,
        *,
        shape5: tuple[int, int, int, int, int],
        dtype: Any,
        level_shapes: list[tuple[int, int]],
        channel_names: list[str],
        spacing_zyx: tuple[float, float, float],
        fmt: str,
        tile_size: int | None,
        is_photo: bool,
        scene_count: int = 1,
        source_order: str = "",
        spacing_units_zyx: tuple[str, str, str] = ("um", "um", "um"),
        max_decoded_chunk_bytes: int | None = None,
    ) -> None:
        self.t, self.c, self.z, self.y, self.x = shape5
        self.dtype = dtype
        self.level_shapes = level_shapes or [(self.y, self.x)]
        self.channel_names = channel_names
        self.spacing_zyx = spacing_zyx
        self.spacing_units_zyx = spacing_units_zyx
        self.format = fmt
        self.tile_size = tile_size
        self.scene_count = max(1, int(scene_count))
        self.source_order = str(source_order or "")
        self.max_decoded_chunk_bytes = max_decoded_chunk_bytes
        self.source_generation: tuple[int, int, int, int, int, int] | None = None
        # An interleaved RGB(A) photo: rendered with a full-range map, never a
        # percentile stretch (that would destroy colour fidelity).
        self.is_photo = is_photo

    @property
    def level_count(self) -> int:
        return len(self.level_shapes)

    def level_scales(self) -> list[float]:
        base = float(max(self.level_shapes[0][1], 1))
        return [round(float(w) / base, 6) for (_h, w) in self.level_shapes]

    def read(
        self, *, t: int, c: int, z: int, level: int, box: tuple[int, int, int, int] | None = None
    ):  # pragma: no cover - overridden
        raise NotImplementedError

    def estimate_read_work(
        self,
        *,
        t: int,
        c: int,
        z: int,
        level: int,
        box: tuple[int, int, int, int] | None = None,
    ) -> int:  # pragma: no cover - overridden
        raise ValueError("decoded chunk geometry is unavailable")

    def close(self) -> None:  # pragma: no cover - overridden
        pass


class _TiffPlane(_Plane):
    """tifffile-backed source. Bounded reads via the level's zarr store."""

    # Axes tifffile uses for a plain page/sequence stack (no explicit Z).
    _PAGE_AXES = ("I", "Q")

    def __init__(self, path: str) -> None:
        import tifffile

        self._path = path
        self._tf = tifffile.TiffFile(path)
        self._scene_count = max(1, len(self._tf.series))
        # Scene selection is semantic. Until the public API carries an explicit
        # scene identity, scene 0 is metadata-only authority and no pixel surface
        # may silently choose a different (for example, largest) series.
        series = self._tf.series[0]
        self._series = series
        axes = str(series.axes or "")
        shape = tuple(int(n) for n in series.shape)
        dims = {ax: shape[i] for i, ax in enumerate(axes) if i < len(shape)}
        self._axes = axes

        samples = int(dims.get("S", 0) or 0)
        channel_count = int(dims.get("C", 1) or 1)
        # Interleaved/planar RGB(A) with no separate C axis: expose the samples as
        # channels so the viewer's photo path engages.
        self._samples_as_channels = samples >= 3 and "C" not in dims
        if self._samples_as_channels:
            channel_count = min(samples, 3)  # drop alpha for display
        is_photo = self._samples_as_channels

        # Depth: an explicit Z, else a page/sequence axis (plain multi-page stack).
        depth = int(dims.get("Z", 1) or 1)
        if depth <= 1:
            for page_axis in self._PAGE_AXES:
                pages = int(dims.get(page_axis, 1) or 1)
                if pages > 1:
                    depth = pages
                    break

        level_shapes: list[tuple[int, int]] = []
        try:
            for lvl in series.levels:
                lax = str(lvl.axes or "")
                lshape = tuple(int(n) for n in lvl.shape)
                ldims = {ax: lshape[i] for i, ax in enumerate(lax) if i < len(lshape)}
                level_shapes.append((int(ldims.get("Y", 0)), int(ldims.get("X", 0))))
        except Exception:  # noqa: BLE001 - non-pyramidal series
            level_shapes = []
        if not level_shapes:
            level_shapes = [(int(dims.get("Y", 0)), int(dims.get("X", 0)))]

        page = series.pages[0] if len(series.pages) else None
        spacing, spacing_units = (
            _tiff_spacing_with_units(self._tf, series)
            if self._scene_count == 1
            else ((1.0, 1.0, 1.0), ("voxel", "voxel", "voxel"))
        )
        max_decoded_chunk_bytes = 0
        for level in getattr(series, "levels", (series,)):
            for level_page in getattr(level, "pages", ()):
                chunks = getattr(level_page, "chunks", None)
                if chunks is None:
                    chunks = getattr(level_page, "shape", None)
                max_decoded_chunk_bytes = max(
                    max_decoded_chunk_bytes,
                    _max_chunk_bytes(chunks, getattr(level_page, "dtype", series.dtype)),
                )
        super().__init__(
            shape5=(
                int(dims.get("T", 1) or 1),
                channel_count,
                depth,
                int(dims.get("Y", 0) or 0),
                int(dims.get("X", 0) or 0),
            ),
            dtype=series.dtype,
            level_shapes=level_shapes,
            channel_names=(
                _tiff_channel_names(self._tf, channel_count, self._samples_as_channels)
                if self._scene_count == 1
                else []
            ),
            spacing_zyx=spacing,
            fmt="OME-TIFF" if getattr(self._tf, "is_ome", False) else "TIFF",
            tile_size=int(getattr(page, "tilewidth", 0) or 0) or None,
            is_photo=is_photo,
            scene_count=self._scene_count,
            source_order=(
                _tiff_source_dimension_order(self._tf, axes) if self._scene_count == 1 else axes
            ),
            spacing_units_zyx=spacing_units,
            max_decoded_chunk_bytes=max_decoded_chunk_bytes,
        )
        self._zarr_cache: dict[int, Any] = {}
        self._decoded_chunk_geometry_cache: dict[
            int, _PreparedDecodedChunkGeometry | _PreparedGeometryFailure
        ] = {}

    def _level_array(self, level: int):
        cached = self._zarr_cache.get(level)
        if cached is not None:
            return cached
        import zarr

        level = max(0, min(level, self.level_count - 1))
        arr = zarr.open(self._series.aszarr(level=level), mode="r")
        self._zarr_cache[level] = arr
        return arr

    def read(
        self, *, t: int, c: int, z: int, level: int, box: tuple[int, int, int, int] | None = None
    ):
        import numpy as np

        time_index = _axis_index(t, "time", self.t)
        channel_index = _axis_index(c, "channel", self.c)
        depth_index = _axis_index(z, "z", self.z)
        level_index = _axis_index(level, "level", self.level_count)
        if box is not None:
            y0, y1, x0, x1 = box
            if not (0 <= y0 < y1 <= self.level_shapes[level_index][0]):
                raise ValueError("source region y bounds are out of range")
            if not (0 <= x0 < x1 <= self.level_shapes[level_index][1]):
                raise ValueError("source region x bounds are out of range")
        arr = self._level_array(level_index)
        rank = len(getattr(arr, "shape", ()))
        index: list[Any] = []
        for position, ax in enumerate(self._axes):
            if position >= rank:
                break
            if ax == "T":
                index.append(time_index)
            elif ax == "C":
                index.append(channel_index)
            elif ax == "Z" or ax in self._PAGE_AXES:
                index.append(depth_index)
            elif ax == "Y":
                index.append(slice(box[0], box[1]) if box else slice(None))
            elif ax == "X":
                index.append(slice(box[2], box[3]) if box else slice(None))
            elif ax == "S":
                # Resolve the sample axis HERE, by position, so the result is
                # always 2-D. Indexing it after the fact assumed an interleaved
                # (H,W,S) layout and silently transposed planar (S,H,W) data.
                index.append(channel_index if self._samples_as_channels else 0)
            else:
                index.append(0)
        while len(index) < rank:
            index.append(slice(None))
        plane = np.asarray(arr[tuple(index)])
        while plane.ndim > 2:  # defensive: an unexpected trailing axis
            plane = plane[..., 0] if plane.shape[-1] <= 4 else plane[0]
        return plane

    def estimate_read_work(
        self,
        *,
        t: int,
        c: int,
        z: int,
        level: int,
        box: tuple[int, int, int, int] | None = None,
    ) -> int:
        time_index = _axis_index(t, "time", self.t)
        channel_index = _axis_index(c, "channel", self.c)
        depth_index = _axis_index(z, "z", self.z)
        level_index = _axis_index(level, "level", self.level_count)
        arr = self._level_array(level_index)
        geometry = self._level_decoded_chunk_geometry(level_index, arr)
        selection: list[Any] = []
        for position, axis in enumerate(self._axes):
            if position >= len(geometry.axes):
                break
            if axis == "T":
                selection.append(time_index)
            elif axis == "C":
                selection.append(channel_index)
            elif axis == "Z" or axis in self._PAGE_AXES:
                selection.append(depth_index)
            elif axis == "Y":
                selection.append(slice(box[0], box[1]) if box else slice(None))
            elif axis == "X":
                selection.append(slice(box[2], box[3]) if box else slice(None))
            elif axis == "S":
                selection.append(channel_index if self._samples_as_channels else 0)
            else:
                selection.append(0)
        while len(selection) < len(geometry.axes):
            selection.append(slice(None))
        return geometry.estimate(selection)

    def _level_decoded_chunk_geometry(
        self,
        level: int,
        array: Any,
    ) -> _PreparedDecodedChunkGeometry:
        cache = self._decoded_chunk_geometry_cache
        cached = cache.get(level)
        if isinstance(cached, _PreparedGeometryFailure):
            cached.raise_error()
        if cached is not None:
            return cached
        chunks = getattr(array, "chunks", None)
        if chunks is None:
            failure = _PreparedGeometryFailure("decoded chunk geometry is unavailable")
            cache[level] = failure
            failure.raise_error()
        try:
            prepared = _prepare_decoded_chunk_geometry(
                getattr(array, "shape", ()),
                chunks,
                getattr(array, "dtype", self.dtype),
            )
        except ValueError as exc:
            failure = _PreparedGeometryFailure(str(exc))
            cache[level] = failure
            failure.raise_error()
        cache[level] = prepared
        return prepared

    def close(self) -> None:
        self._zarr_cache.clear()
        self._decoded_chunk_geometry_cache.clear()
        try:
            self._tf.close()
        except Exception:  # noqa: BLE001
            pass


class _BioioPlane(_Plane):
    """bioio-backed source (CZI/ND2/LIF/DV). Planes materialize via dask."""

    def __init__(self, path: str) -> None:
        from bioio import BioImage

        self._img = BioImage(path)
        scene_count = 1
        scenes: list[Any] = []
        source_order = ""
        try:
            scenes = list(self._img.scenes)
            scene_count = max(1, len(scenes))
        except Exception:  # noqa: BLE001 - single-scene sources
            pass
        # BioIO readers may restore a plugin-specific current scene rather than
        # starting at zero. Derivation semantics explicitly bind the first scene,
        # so select it before reading dims, pixels, names, or physical spacing.
        # Passing the index is supported consistently across the installed CZI,
        # ND2, LIF, DV, and R3D plugins.
        if scenes:
            self._img.set_scene(0)
        try:
            source_order = str(self._img.dims.order or "")
        except Exception:  # noqa: BLE001
            source_order = ""
        self._dask = self._img.get_image_dask_data("TCZYX")
        # BioImage starts on scene 0 and the derivation lane explicitly converts
        # the first series. Preserve that exact producer selection even when the
        # container has multiple scenes; omitting it lets a replay silently bind
        # the committed artifact to a different source series.
        self.selected_scene_index = 0
        try:
            self.selected_scene_id = str(self._img.current_scene)
        except Exception:  # noqa: BLE001 - some plugins expose only ``scenes``
            self.selected_scene_id = str(scenes[0]) if scenes else ""
        shape = tuple(int(n) for n in self._dask.shape)
        try:
            names = [str(n) for n in (self._img.channel_names or [])]
        except Exception:  # noqa: BLE001
            names = []
        spacing = (1.0, 1.0, 1.0)
        spacing_units = ("voxel", "voxel", "voxel")
        try:
            px = self._img.physical_pixel_sizes
            values = (getattr(px, "Z", None), getattr(px, "Y", None), getattr(px, "X", None))
            spacing = tuple(float(value) if value is not None else 1.0 for value in values)
            spacing_units = tuple("um" if value is not None else "voxel" for value in values)
        except Exception:  # noqa: BLE001
            pass
        prepared_chunk_geometry = _prepare_decoded_chunk_geometry(
            self._dask.shape,
            self._dask.chunks,
            self._dask.dtype,
        )
        super().__init__(
            shape5=(shape[0], shape[1], shape[2], shape[3], shape[4]),
            dtype=self._dask.dtype,
            level_shapes=[(shape[3], shape[4])],  # bioio exposes no pyramid
            channel_names=names,
            spacing_zyx=spacing,
            fmt=_lower_ext(path).lstrip(".").upper() or "BIOIO",
            tile_size=None,
            is_photo=False,
            scene_count=scene_count,
            source_order=source_order,
            spacing_units_zyx=spacing_units,
            max_decoded_chunk_bytes=prepared_chunk_geometry.max_chunk_bytes,
        )
        self._decoded_chunk_geometry_cache: (
            _PreparedDecodedChunkGeometry | _PreparedGeometryFailure | None
        ) = prepared_chunk_geometry

    def read(
        self, *, t: int, c: int, z: int, level: int, box: tuple[int, int, int, int] | None = None
    ):
        import numpy as np

        time_index = _axis_index(t, "time", self.t)
        channel_index = _axis_index(c, "channel", self.c)
        depth_index = _axis_index(z, "z", self.z)
        _axis_index(level, "level", self.level_count)
        if box is not None:
            y0, y1, x0, x1 = box
            if not (0 <= y0 < y1 <= self.y and 0 <= x0 < x1 <= self.x):
                raise ValueError("source region bounds are out of range")
        sub = self._dask[time_index, channel_index, depth_index]
        if box:
            sub = sub[box[0] : box[1], box[2] : box[3]]
        return np.asarray(sub.compute())

    def estimate_read_work(
        self,
        *,
        t: int,
        c: int,
        z: int,
        level: int,
        box: tuple[int, int, int, int] | None = None,
    ) -> int:
        time_index = _axis_index(t, "time", self.t)
        channel_index = _axis_index(c, "channel", self.c)
        depth_index = _axis_index(z, "z", self.z)
        _axis_index(level, "level", self.level_count)
        y_selection = slice(box[0], box[1]) if box else slice(None)
        x_selection = slice(box[2], box[3]) if box else slice(None)
        return self._decoded_chunk_geometry().estimate(
            (time_index, channel_index, depth_index, y_selection, x_selection)
        )

    def _decoded_chunk_geometry(self) -> _PreparedDecodedChunkGeometry:
        cached = self._decoded_chunk_geometry_cache
        if isinstance(cached, _PreparedGeometryFailure):
            cached.raise_error()
        if cached is not None:
            return cached
        try:
            prepared = _prepare_decoded_chunk_geometry(
                self._dask.shape,
                self._dask.chunks,
                self._dask.dtype,
            )
        except ValueError as exc:
            failure = _PreparedGeometryFailure(str(exc))
            self._decoded_chunk_geometry_cache = failure
            failure.raise_error()
        self._decoded_chunk_geometry_cache = prepared
        return prepared

    def close(self) -> None:
        self._img = None
        self._dask = None
        self._decoded_chunk_geometry_cache = None


def _tiff_channel_names(tf: Any, channel_count: int, samples_as_channels: bool) -> list[str]:
    if samples_as_channels:
        # Naming them red/green/blue is load-bearing: viewerinfo keys its photo
        # (full-colour) render path on exactly these names.
        return ["red", "green", "blue"][:channel_count]
    try:
        meta = getattr(tf, "ome_metadata", None)
        if meta:
            import re

            return re.findall(r'<Channel[^>]*\bName="([^"]+)"', str(meta))[:channel_count]
    except Exception:  # noqa: BLE001 - channel names are best-effort
        pass
    return []


def _tiff_source_dimension_order(tf: Any, fallback: str) -> str:
    try:
        meta = getattr(tf, "ome_metadata", None)
        if meta:
            import re

            match = re.search(r'\bDimensionOrder="([A-Z]+)"', str(meta))
            if match:
                return match.group(1)
    except Exception:  # noqa: BLE001 - provenance is best effort
        pass
    return str(fallback or "")


# OME UnitsLength physical values -> microns. Pixel and reference-frame values
# are deliberately absent: they are valid metadata units, but not physical length.
_UNIT_TO_MICRON = {
    "Ym": 1e30,
    "Zm": 1e27,
    "Em": 1e24,
    "Pm": 1e21,
    "Tm": 1e18,
    "Gm": 1e15,
    "Mm": 1e12,
    "km": 1e9,
    "hm": 1e8,
    "dam": 1e7,
    "m": 1e6,
    "dm": 1e5,
    "cm": 1e4,
    "mm": 1e3,
    "µm": 1.0,
    "um": 1.0,
    "nm": 1e-3,
    "pm": 1e-6,
    "fm": 1e-9,
    "am": 1e-12,
    "zm": 1e-15,
    "ym": 1e-18,
    "Å": 1e-4,
    "thou": 25.4,
    "li": 25_400.0 / 12.0,
    "in": 25_400.0,
    "ft": 304_800.0,
    "yd": 914_400.0,
    "mi": 1_609_344_000.0,
    "ua": 1.495978707e17,
    "ly": 9.4607304725808e21,
    "pc": 3.085677581491367e22,
    "pt": 25_400.0 / 72.0,
    "micron": 1.0,
    "microns": 1.0,
    "angstrom": 1e-4,
}

# TIFF ResolutionUnit tag values (2 = inch, 3 = centimetre).
_RESOLUTION_UNIT_TO_MICRON = {2: 25_400.0, 3: 10_000.0}


def _tiff_spacing_with_units(
    tf: Any, series: Any
) -> tuple[tuple[float, float, float], tuple[str, str, str]]:
    """Return (z, y, x) spacing and honest per-axis units."""
    try:
        meta = getattr(tf, "ome_metadata", None)
        if meta:
            import re

            text = str(meta)

            def _axis(key: str) -> tuple[float, str, bool]:
                match = re.search(rf'\bPhysicalSize{key}="([0-9.eE+-]+)"', text)
                if not match:
                    return 1.0, "voxel", False
                value = float(match.group(1))
                unit_match = re.search(rf'\bPhysicalSize{key}Unit="([^"]+)"', text)
                unit = unit_match.group(1).strip() if unit_match else "µm"
                factor = _UNIT_TO_MICRON.get(unit)
                if factor is None:
                    factor = _UNIT_TO_MICRON.get(unit.lower())
                scaled = value * factor if factor is not None else value
                if not math.isfinite(scaled) or scaled <= 0:
                    return 1.0, "voxel", False
                if factor is not None:
                    return scaled, "um", True
                preserved = unit or "unknown"
                return scaled, preserved, True

            parsed = (_axis("Z"), _axis("Y"), _axis("X"))
            if any(present for _value, _unit, present in parsed):
                return (
                    tuple(value for value, _unit, _present in parsed),
                    tuple(unit for _value, unit, _present in parsed),
                )
    except Exception:  # noqa: BLE001
        pass
    try:
        page = series.pages[0]
        tags = getattr(page, "tags", {})

        unit_tag = tags.get("ResolutionUnit")
        unit_value = int(getattr(unit_tag, "value", 0) or 0)
        per_unit = _RESOLUTION_UNIT_TO_MICRON.get(unit_value)
        if per_unit is None:
            return (1.0, 1.0, 1.0), ("voxel", "voxel", "voxel")

        def _resolution(tag_name: str) -> float:
            tag = tags.get(tag_name)
            value = getattr(tag, "value", None)
            if not value or len(value) != 2:
                return 1.0
            numerator, denominator = float(value[0]), float(value[1])
            if (
                not math.isfinite(numerator)
                or not math.isfinite(denominator)
                or numerator <= 0
                or denominator <= 0
            ):
                return 1.0
            spacing = (denominator / numerator) * per_unit
            return spacing if math.isfinite(spacing) and spacing > 0 else 1.0

        return (
            (1.0, _resolution("YResolution"), _resolution("XResolution")),
            ("voxel", "um", "um"),
        )
    except Exception:  # noqa: BLE001
        pass
    return (1.0, 1.0, 1.0), ("voxel", "voxel", "voxel")


def _tiff_spacing(tf: Any, series: Any) -> tuple[float, float, float]:
    """Compatibility wrapper returning only the numeric (z, y, x) spacing."""
    return _tiff_spacing_with_units(tf, series)[0]


class BioioEngine(_Hdf5EngineMixin):
    """Real-decode engine without the native wheel. See module docstring."""

    def __init__(self, cache_size: int = 4) -> None:
        try:
            import numpy
            import tifffile  # noqa: F401
            from PIL import Image
        except Exception as exc:  # noqa: BLE001
            raise EngineUnavailable(
                f"bioio engine needs numpy + tifffile + Pillow: {exc!r}"
            ) from exc
        self._np = numpy
        self._Image = Image
        self._max_sources = max(1, min(cache_size, 8))
        self._sources: dict[Any, _Plane] = {}
        self._windows = _BoundedDict(_engine_cache_entries())
        self._signal_scores = _BoundedDict(_engine_cache_entries())
        self._scalar_profiles = _BoundedDict(_engine_cache_entries())

    # -- source handling ------------------------------------------------------
    def _source(self, path: str) -> _Plane:
        generation_before = _source_generation(path)
        key = (str(path), generation_before)
        cached = self._sources.get(key)
        if cached is not None:
            return cached
        ext = _lower_ext(path)
        try:
            if ext in TIFF_EXTENSIONS:
                source: _Plane = _TiffPlane(path)
            elif ext in BIOIO_EXTENSIONS:
                source = _BioioPlane(path)
            else:
                raise ValueError(f"unsupported format for the bioio engine: {ext or 'unknown'}")
        except Exception as exc:  # noqa: BLE001
            raise _as_decode_error(path, exc) from exc
        generation_after = _source_generation(path)
        if generation_after != generation_before:
            source.close()
            raise ValueError("cannot decode source because its generation changed while opening")
        source.source_generation = generation_after
        if source.x <= 0 or source.y <= 0:
            source.close()
            raise ValueError(
                f"image reports no pixel geometry ({source.x}x{source.y}); cannot decode"
            )
        # Bounded, and CLOSE what we evict — an open TiffFile pins parsed page
        # tables (hundreds of MB on a big pyramid) plus a file descriptor.
        while len(self._sources) >= self._max_sources:
            _evicted_key, evicted = next(iter(self._sources.items()))
            self._sources.pop(_evicted_key, None)
            try:
                evicted.close()
            except Exception:  # noqa: BLE001
                pass
        self._sources[key] = source
        return source

    def _level_for_size(self, source: _Plane, target_long_edge: int) -> int:
        best = 0
        for i, (h, w) in enumerate(source.level_shapes):
            if max(h, w) >= target_long_edge:
                best = i
        return best

    def _window_for(
        self, source: _Plane, path: str, level: int, channel: int, *, t: int = 0
    ) -> tuple[float, float]:
        """One global [lo,hi] per (file, level, channel), from a BOUNDED centre crop.

        Bounded so a single tile request never decodes a whole level; deterministic
        so every tile/slice of the image shares the mapping (no checkerboard).
        """
        np = self._np
        try:
            stamp = os.stat(path).st_mtime_ns
        except OSError:
            stamp = 0
        key = (str(path), stamp, level, channel, t)
        cached = self._windows.get(key)
        if cached is not None:
            return cached
        height, width = source.level_shapes[max(0, min(level, source.level_count - 1))]
        crop_h = min(height, _WINDOW_SAMPLE_EDGE)
        crop_w = min(width, _WINDOW_SAMPLE_EDGE)
        y0 = max(0, (height - crop_h) // 2)
        x0 = max(0, (width - crop_w) // 2)
        plane = source.read(
            t=t,
            c=channel,
            z=max(source.z // 2, 0),
            level=level,
            box=(y0, y0 + crop_h, x0, x0 + crop_w),
        )
        window = _robust_window(np.asarray(plane).reshape(-1).astype("float32"), np)
        self._windows[key] = window
        return window

    def _full_range_window(self, source: _Plane) -> tuple[float, float]:
        """The dtype's display range — what a photo must use so its colours are
        reproduced faithfully instead of per-channel autocontrast-stretched."""
        dtype = self._np.dtype(source.dtype)
        if dtype.kind == "u":
            return (0.0, float(2 ** (dtype.itemsize * 8) - 1))
        if dtype.kind == "f":
            return (0.0, 1.0)
        return (float(-(2 ** (dtype.itemsize * 8 - 1))), float(2 ** (dtype.itemsize * 8 - 1) - 1))

    def _planes_uint8(
        self,
        source: _Plane,
        path: str,
        *,
        t: int,
        z: int,
        level: int,
        zero_based: Sequence[int],
        colors: Any,
        box: tuple[int, int, int, int] | None = None,
    ):
        """Read the selected channels and map them to display pixels."""
        np = self._np
        planes = [source.read(t=t, c=c, z=z, level=level, box=box) for c in zero_based]
        if not planes or any(getattr(p, "size", 0) == 0 for p in planes):
            raise ValueError("engine returned an empty region")
        stack = np.stack([np.asarray(p) for p in planes], axis=0).astype("float32")
        if colors:
            windows = [self._window_for(source, path, level, c, t=t) for c in zero_based]
            return fusion.composite_channels(
                stack, _parse_colors(colors, len(zero_based)), np=np, windows=windows
            )
        if source.is_photo and len(zero_based) >= 3:
            full = self._full_range_window(source)
            return np.transpose(_window_to_uint8(stack[:3], [full] * 3, np), (1, 2, 0))
        return _window_to_uint8(
            stack[0], [self._window_for(source, path, level, zero_based[0], t=t)], np
        )

    def _render(
        self,
        source: _Plane,
        path: str,
        *,
        t: int,
        z: int,
        level: int,
        channels: Sequence[int] | None,
        colors: Any,
        box: tuple[int, int, int, int] | None = None,
        max_dim: int | None = None,
    ) -> bytes:
        if channels is None:
            zero_based = list(range(min(source.c, 3))) if source.is_photo else [0]
        else:
            zero_based = _zero_based(channels, source.c)
        out = self._planes_uint8(
            source, path, t=t, z=z, level=level, zero_based=zero_based, colors=colors, box=box
        )
        if max_dim:
            out = self._downscale(out, max_dim)
        return self._encode_png(out)

    def _downscale(self, arr, max_dim: int, *, sampling: str = "linear"):
        np = self._np
        h, w = int(arr.shape[0]), int(arr.shape[1])
        long_edge = max(h, w)
        if long_edge <= max_dim or long_edge == 0:
            return arr
        scale = max_dim / float(long_edge)
        size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        img = self._Image.fromarray(np.asarray(arr, dtype="uint8"))
        resample = (
            self._Image.Resampling.NEAREST
            if sampling == "nearest"
            else self._Image.Resampling.BILINEAR
        )
        return np.asarray(img.resize(size, resample))

    def _encode_png(self, arr) -> bytes:
        np = self._np
        a = np.asarray(arr)
        if 0 in getattr(a, "shape", ()):
            raise ValueError(f"engine returned an empty region (shape {a.shape})")
        if a.dtype != np.uint8:
            mx = float(a.max()) or 1.0
            a = (a.astype("float32") / mx * 255.0).clip(0, 255).astype("uint8")
        buf = io.BytesIO()
        self._Image.fromarray(a).save(buf, format="PNG")
        return buf.getvalue()

    # -- ImageEngine API ------------------------------------------------------
    def formats(self) -> list[str]:
        return sorted({e.lstrip(".") for e in (*TIFF_EXTENSIONS, *BIOIO_EXTENSIONS)})

    def meta(self, path: str) -> dict[str, Any]:
        """A libbioimage-SHAPED metadata dict, so viewerinfo stays shared."""
        source = self._source(path)
        dtype = self._np.dtype(source.dtype)
        pixel_format = {"f": "floating point", "i": "signed integer"}.get(
            dtype.kind, "unsigned integer"
        )
        meta: dict[str, Any] = {
            "format": source.format,
            # The engine normalizes every decoder to TCZYX. Keep the container's
            # original axis declaration separately so provenance is not lost.
            "image_dimensions": "TCZYX",
            "source_dimension_order": source.source_order,
            "image_num_scenes": source.scene_count,
            "volume_preview_supported": source.scene_count == 1,
            "image_num_x": source.x,
            "image_num_y": source.y,
            "image_num_z": source.z,
            "image_num_c": source.c,
            "image_num_t": source.t,
            "image_pixel_depth": int(dtype.itemsize * 8),
            "image_pixel_format": pixel_format,
            "image_num_resolution_levels": source.level_count,
            "image_res_l_scales": ",".join(str(s) for s in source.level_scales()),
            "pixel_resolution_z": source.spacing_zyx[0],
            "pixel_resolution_y": source.spacing_zyx[1],
            "pixel_resolution_x": source.spacing_zyx[2],
            "pixel_resolution_unit_z": source.spacing_units_zyx[0],
            "pixel_resolution_unit_y": source.spacing_units_zyx[1],
            "pixel_resolution_unit_x": source.spacing_units_zyx[2],
        }
        selected_scene_index = getattr(source, "selected_scene_index", None)
        selected_scene_id = getattr(source, "selected_scene_id", None)
        if selected_scene_index is not None:
            meta["selected_scene_index"] = selected_scene_index
        elif source.scene_count == 1:
            meta["selected_scene_index"] = 0
        if selected_scene_id is not None:
            meta["selected_scene_id"] = selected_scene_id
        if source.tile_size:
            meta["tile_size_x"] = str(source.tile_size)
        for i, name in enumerate(source.channel_names[: source.c]):
            meta[f"channels/channel:{i}/name"] = name
        return meta

    def viewer_info(self, path: str, name: str | None = None) -> dict[str, Any]:
        hdf5_info = self._maybe_hdf5_viewer_info(path, name)
        if hdf5_info is not None:
            return hdf5_info
        # Report the decoder that actually read the file — the metadata panel
        # must not credit libbioimage for a tifffile/bioio read.
        reader = "tifffile" if _lower_ext(path) in TIFF_EXTENSIONS else "bioio"
        meta = self.meta(path)
        source = self._source(path)
        membership_dtype = scalar_semantics.mask_membership_dtype(source.dtype)
        scalar_mask_capability = None
        if isinstance(source, _TiffPlane) and membership_dtype is not None:
            try:
                self._admit_scalar_mask_surfaces(path, source)
            except (TypeError, ValueError):
                pass
            else:
                scalar_mask_capability = {
                    "version": 1,
                    "source_authority": "original",
                    "source_format": (
                        "ome-tiff" if str(source.format).strip().lower() == "ome-tiff" else "tiff"
                    ),
                    "dtype": membership_dtype,
                    "threshold_domain": "raw",
                    "threshold_foreground": "above",
                    "slice_delivery": "thresholded_png",
                    "volume_delivery": "raw_scalar",
                    "volume_sampling": "nearest",
                    "channel_selection": "single",
                    "time_selection": "single",
                }
        data_semantics = None
        if (
            int(meta.get("image_num_scenes", 1) or 1) == 1
            and int(meta.get("image_num_c", 1) or 1) == 1
            and int(meta.get("image_num_z", 1) or 1) > 1
        ):
            try:
                data_semantics = self._scalar_profile(path, channel=0, t=0)["data_semantics"]
            except Exception:  # noqa: BLE001 - semantic profiling is best effort
                data_semantics = None
        return viewerinfo.build_viewer_info(
            meta,
            signal_scores=(
                self._channel_signal_scores(path)
                if int(meta.get("image_num_scenes", 1) or 1) == 1
                else None
            ),
            reader=reader,
            data_semantics=data_semantics,
            scalar_mask_capability=scalar_mask_capability,
        )

    def strict_publication_viewer_info(
        self, path: str, name: str | None = None
    ) -> dict[str, Any]:
        """Describe the explicitly selected scene 0 for strict derivative publication.

        The public descriptor remains metadata-only for a multi-scene container. The
        publication worker is different: ``_BioioPlane`` has already bound scene 0,
        and needs a pixel-bearing semantic fingerprint for exactly that selection.
        Container scene provenance is restored after deriving the scene-0 surface.
        """
        hdf5_info = self._maybe_hdf5_viewer_info(path, name)
        if hdf5_info is not None:
            return hdf5_info
        meta = self.meta(path)
        container_scene_count = int(meta.get("image_num_scenes", 1) or 1)
        selected_scene_index = int(meta.get("selected_scene_index", 0) or 0)
        selected_scene_id = meta.get("selected_scene_id")
        publication_meta = dict(meta)
        publication_meta["image_num_scenes"] = 1
        publication_meta["volume_preview_supported"] = True
        reader = "tifffile" if _lower_ext(path) in TIFF_EXTENSIONS else "bioio"
        descriptor = viewerinfo.build_viewer_info(
            publication_meta,
            signal_scores=None,
            reader=reader,
        )
        descriptor["scene_count"] = container_scene_count
        descriptor["selected_scene_index"] = selected_scene_index
        metadata = descriptor.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            descriptor["metadata"] = metadata
        metadata["scene_count"] = container_scene_count
        metadata["selected_scene_index"] = selected_scene_index
        if selected_scene_id is not None:
            scene_id = str(selected_scene_id)
            descriptor["selected_scene_id"] = scene_id
            metadata["selected_scene_id"] = scene_id
        return descriptor

    def _admit_scalar_mask_surfaces(self, path: str, source: _Plane) -> dict[str, Any]:
        if (
            source.scene_count != 1
            or source.is_photo
            or source.z <= 1
            or scalar_semantics.mask_membership_dtype(source.dtype) is None
        ):
            raise ValueError("source cannot preserve exact mask membership")
        plan = self.scalar_plan(path, channel=0, t=0, sampling="nearest")
        atlas_mod.validate_scalar_plan(plan)
        exact_regions = _exact_plane_regions(source)
        profile_zs, profile_regions = scalar_semantics.profile_decode_plan(source)
        exact_work = _admit_decode_reads(
            source,
            ((0, 0, 0, region) for region in exact_regions),
            expected_read_count=len(exact_regions),
            label="exact mask plane",
        )
        profile_read_count = len(profile_zs) * len(profile_regions)
        profile_work = _admit_decode_reads(
            source,
            ((0, 0, source_z, region) for source_z in profile_zs for region in profile_regions),
            expected_read_count=profile_read_count,
            label="scalar histogram",
            max_work_bytes=scalar_semantics.MAX_PROFILE_DECODE_WORK_BYTES,
        )
        nearest_work = atlas_mod.validate_scalar_decoder_admission(plan)
        nearest_read_count = int(plan["admitted_decode_read_count"])
        return {
            "plan": plan,
            "surfaces": {
                "exact_plane": {
                    "admitted_decode_work_bytes": exact_work,
                    "read_count": len(exact_regions),
                },
                "histogram": {
                    "admitted_decode_work_bytes": profile_work,
                    "read_count": profile_read_count,
                },
                "nearest_volume": {
                    "admitted_decode_work_bytes": nearest_work,
                    "read_count": nearest_read_count,
                },
            },
        }

    def _scalar_profile(
        self, path: str, *, channel: int, t: int, bins: int = 256
    ) -> dict[str, Any]:
        source = self._source(path)
        _require_single_scene(source)
        channel_index = _axis_index(channel, "channel", source.c)
        time_index = _axis_index(t, "time", source.t)
        key = scalar_semantics.profile_cache_key(path, channel_index, time_index, int(bins))
        cached = self._scalar_profiles.get(key)
        if cached is not None:
            return cached
        profile = scalar_semantics.profile_scalar_volume(
            source,
            path,
            channel=channel_index,
            t=time_index,
            bins=int(bins),
        )
        self._scalar_profiles[key] = profile
        return profile

    def _channel_signal_scores(self, path: str) -> list[float] | None:
        """Bounded per-channel spatial correlation from one centered mid-Z crop."""
        source = self._source(path)
        if source.c <= 1:
            return None
        if source.c > MAX_VIEWERINFO_SIGNAL_CHANNELS:
            return None
        try:
            stamp = os.stat(path).st_mtime_ns
        except OSError:
            stamp = 0
        key = (str(path), stamp, source.c)
        cached = self._signal_scores.get(key)
        if cached is not None:
            return cached
        np = self._np
        level = self._level_for_size(source, 256)
        height, width = source.level_shapes[level]
        crop_h, crop_w = min(height, 256), min(width, 256)
        y0, x0 = max(0, (height - crop_h) // 2), max(0, (width - crop_w) // 2)
        scores: list[float] = []
        try:
            for channel in range(source.c):
                plane = np.asarray(
                    source.read(
                        t=0,
                        c=channel,
                        z=max(source.z // 2, 0),
                        level=level,
                        box=(y0, y0 + crop_h, x0, x0 + crop_w),
                    ),
                    dtype="float32",
                )
                if plane.ndim != 2 or min(plane.shape) < 4:
                    scores.append(1.0)
                    continue
                centered = plane - float(plane.mean())
                variance = float((centered * centered).mean())
                if variance <= 1e-9:
                    scores.append(0.0)
                    continue
                horizontal = float((centered[:, :-1] * centered[:, 1:]).mean()) / variance
                vertical = float((centered[:-1, :] * centered[1:, :]).mean()) / variance
                scores.append(max(0.0, min(1.0, (horizontal + vertical) * 0.5)))
        except Exception:  # noqa: BLE001 - metadata scoring is best effort
            return None
        self._signal_scores[key] = scores
        return scores

    def tile(
        self,
        path,
        *,
        level=0,
        col=0,
        row=0,
        tile_size=512,
        t=0,
        z=0,
        channels=None,
        colors=None,
        windows=None,
    ) -> bytes:
        source = self._source(path)
        _require_single_scene(source)
        time_index = _axis_index(t, "time", source.t)
        depth_index = _axis_index(z, "z", source.z)
        lvl = max(0, min(int(level), source.level_count - 1))
        height, width = source.level_shapes[lvl]
        y0, x0 = int(row) * int(tile_size), int(col) * int(tile_size)
        if y0 >= height or x0 >= width:
            raise ValueError(f"engine returned an empty region (tile {col},{row} past level {lvl})")
        box = (y0, min(y0 + int(tile_size), height), x0, min(x0 + int(tile_size), width))
        try:
            return self._render(
                source,
                path,
                t=time_index,
                z=depth_index,
                level=lvl,
                channels=channels,
                colors=colors,
                box=box,
            )
        except Exception as exc:  # noqa: BLE001
            raise _as_decode_error(path, exc) from exc

    def region(
        self, path, *, x1, y1, x2, y2, region_scale=None, channels=None, colors=None, windows=None
    ) -> bytes:
        source = self._source(path)
        _require_single_scene(source)
        height, width = source.level_shapes[0]
        box = (max(0, int(y1)), min(int(y2), height), max(0, int(x1)), min(int(x2), width))
        if box[1] <= box[0] or box[3] <= box[2]:
            raise ValueError("engine returned an empty region (degenerate ROI)")
        max_dim = None
        if region_scale and 0 < float(region_scale) < 1:
            # Scale the LONG edge — deriving it from the width alone shrank tall
            # ROIs by their aspect ratio.
            long_edge = max(box[1] - box[0], box[3] - box[2])
            max_dim = max(1, int(round(long_edge * float(region_scale))))
        try:
            return self._render(
                source,
                path,
                t=0,
                z=0,
                level=0,
                channels=channels,
                colors=colors,
                box=box,
                max_dim=max_dim,
            )
        except Exception as exc:  # noqa: BLE001
            raise _as_decode_error(path, exc) from exc

    def slice_plane(
        self,
        path,
        *,
        z=None,
        t=None,
        level=None,
        plane_scale=None,
        channels=None,
        colors=None,
        windows=None,
        full_resolution=True,
        scalar_render_mode="intensity",
        scalar_threshold_value=None,
        scalar_threshold_foreground="above",
    ) -> bytes:
        hdf5_png = self._maybe_hdf5_slice(path, z)
        if hdf5_png is not None:
            return hdf5_png
        source = self._source(path)
        _require_single_scene(source)
        time_index = _axis_index(0 if t is None else t, "time", source.t)
        depth_index = _axis_index(0 if z is None else z, "z", source.z)
        selected_channels = _zero_based(channels, source.c)
        mask_mode = str(scalar_render_mode or "intensity").strip().lower() == "mask"
        if mask_mode:
            if not isinstance(source, _TiffPlane):
                raise ValueError("unsupported mask slice for this source decoder")
            if len(selected_channels) != 1:
                raise ValueError("mask slice requires exactly one selected channel")
            if scalar_threshold_foreground != "above":
                raise ValueError("unsupported mask threshold foreground")
            try:
                threshold = float(scalar_threshold_value)
            except (TypeError, ValueError) as exc:
                raise ValueError("mask slice threshold must be finite") from exc
            if not math.isfinite(threshold):
                raise ValueError("mask slice threshold must be finite")
            try:
                source_itemsize = max(1, self._np.dtype(source.dtype).itemsize)
                work_bytes = (
                    int(source.x)
                    * int(source.y)
                    * (source_itemsize + self._np.dtype("uint8").itemsize)
                )
                if work_bytes > _SCALAR_SOURCE_WORK_MAX_BYTES:
                    raise ValueError("mask slice exceeds the bounded exact source-read envelope")
                rendered = self._bounded_mask_plane(
                    source,
                    t=time_index,
                    channel=selected_channels[0],
                    z=depth_index,
                    threshold=threshold,
                )
                if not full_resolution:
                    rendered = self._downscale(rendered, SCRUB_MAX_DIMENSION, sampling="nearest")
                return self._encode_png(rendered)
            except Exception as exc:  # noqa: BLE001
                raise _as_decode_error(path, exc) from exc
        lvl = (
            int(level)
            if level is not None
            else (0 if full_resolution else self._level_for_size(source, SCRUB_MAX_DIMENSION))
        )
        lvl = max(0, min(lvl, source.level_count - 1))
        try:
            return self._render(
                source,
                path,
                t=time_index,
                z=depth_index,
                level=lvl,
                channels=channels,
                colors=colors,
                max_dim=None if full_resolution else SCRUB_MAX_DIMENSION,
            )
        except Exception as exc:  # noqa: BLE001
            raise _as_decode_error(path, exc) from exc

    def thumbnail(
        self,
        path,
        *,
        max_size=256,
        z=None,
        t=None,
        level=None,
        channels=None,
        colors=None,
        windows=None,
    ) -> bytes:
        hdf5_png = self._maybe_hdf5_thumbnail(path, max_size)
        if hdf5_png is not None:
            return hdf5_png
        source = self._source(path)
        _require_single_scene(source)
        time_index = _axis_index(0 if t is None else t, "time", source.t)
        depth_index = _axis_index(max(source.z // 2, 0) if z is None else z, "z", source.z)
        _zero_based(channels, source.c)
        lvl = int(level) if level is not None else self._level_for_size(source, int(max_size))
        lvl = max(0, min(lvl, source.level_count - 1))
        try:
            return self._render(
                source,
                path,
                t=time_index,
                z=depth_index,
                level=lvl,
                channels=channels,
                colors=colors,
                max_dim=int(max_size),
            )
        except Exception as exc:  # noqa: BLE001
            raise _as_decode_error(path, exc) from exc

    # -- atlas / scalar-volume: plan + cell methods -----------------------------
    # These mirror the native engine so the POOL orchestrator
    # (imaging/atlas.py assemble_atlas / assemble_scalar_volume) works here too.
    # Without them /atlas and /scalar-volume raise AttributeError -> 500 on every
    # multi-process image service.

    def atlas_plan(self, path, *, channels=None, colors=None, level=None, t=0) -> dict[str, Any]:
        source = self._source(path)
        _require_bounded_volume_source(source)
        time_index = _axis_index(t, "time", source.t)
        _zero_based(channels, source.c)
        depth = max(1, source.z)
        layout = viewerinfo.atlas_layout(source.x, source.y, depth)
        if colors and channels:
            read_channels = list(channels)
            cell_colors = [tuple(c) if c is not None else None for c in colors]
        else:
            read_channels = [channels[0]] if channels else [1]  # 1-based, like the native engine
            cell_colors = [(1.0, 1.0, 1.0)]
        read_level = (
            level
            if level is not None
            else self._level_for_size(source, max(layout["cell_w"], layout["cell_h"]))
        )
        read_level = max(0, min(int(read_level), source.level_count - 1))
        level_height, level_width = source.level_shapes[read_level]
        input_bytes = (
            level_height * level_width * self._np.dtype(source.dtype).itemsize * len(read_channels)
        )
        if input_bytes > _ATLAS_SOURCE_PLANE_MAX_BYTES:
            raise ValueError(
                "source plane input exceeds the bounded atlas limit; preview is unsupported"
            )
        return {
            "depth": depth,
            "columns": layout["columns"],
            "rows": layout["rows"],
            "cell_w": layout["cell_w"],
            "cell_h": layout["cell_h"],
            "read_level": read_level,
            "read_channels": read_channels,
            "cell_colors": cell_colors,
            "paged": False,
            "t": time_index,
        }

    def atlas_windows(self, path, *, depth, level, channels, paged, t=0):
        source = self._source(path)
        _require_bounded_volume_source(source)
        time_index = _axis_index(t, "time", source.t)
        return [
            self._window_for(source, path, level, c, t=time_index)
            for c in _zero_based(channels, source.c)
        ]

    def atlas_cell(self, path, *, z, level, channels, colors, windows, cell_w, cell_h, paged, t=0):
        np = self._np
        source = self._source(path)
        _require_bounded_volume_source(source)
        time_index = _axis_index(t, "time", source.t)
        depth_index = _axis_index(z, "z", source.z)
        level_index = _axis_index(level, "level", source.level_count)
        zero_based = _zero_based(channels, source.c)
        planes = [
            source.read(t=time_index, c=c, z=depth_index, level=level_index) for c in zero_based
        ]
        stack = np.stack([np.asarray(p) for p in planes], axis=0).astype("float32")
        cell = fusion.composite_channels(
            stack, _parse_colors(colors, len(zero_based)), np=np, windows=windows
        )
        # Resize to the ADVERTISED cell geometry: the frontend decodes the atlas
        # with the atlas_scheme from viewer-info, so a native-size cell would
        # mis-tile the volume (and blow the PNG up to tens of megapixels).
        if cell.shape[0] != cell_h or cell.shape[1] != cell_w:
            cell = np.asarray(
                self._Image.fromarray(np.asarray(cell, dtype="uint8")).resize(
                    (cell_w, cell_h), self._Image.BILINEAR
                )
            )
        return cell

    def atlas_cells(
        self, path, *, zs, level, channels, colors, windows, cell_w, cell_h, paged, t=0
    ):
        return [
            self.atlas_cell(
                path,
                z=z,
                level=level,
                channels=channels,
                colors=colors,
                windows=windows,
                cell_w=cell_w,
                cell_h=cell_h,
                paged=paged,
                t=t,
            )
            for z in zs
        ]

    def atlas(
        self,
        path,
        *,
        grid=None,
        level=None,
        atlas_scale=None,
        channels=None,
        colors=None,
        windows=None,
        t=0,
    ) -> bytes:
        """Sequential atlas — same plan/cells as the parallel path, so both agree."""
        try:
            plan = self.atlas_plan(path, channels=channels, colors=colors, level=level, t=t)
            if grid:
                plan = {**plan, "rows": int(grid[0]), "columns": int(grid[1])}
            cell_windows = self.atlas_windows(
                path,
                depth=plan["depth"],
                level=plan["read_level"],
                channels=plan["read_channels"],
                paged=plan["paged"],
                t=plan["t"],
            )
            cells = self.atlas_cells(
                path,
                zs=range(plan["depth"]),
                level=plan["read_level"],
                channels=plan["read_channels"],
                colors=plan["cell_colors"],
                windows=cell_windows,
                cell_w=plan["cell_w"],
                cell_h=plan["cell_h"],
                paged=plan["paged"],
                t=plan["t"],
            )
            return atlas_mod.compose_atlas_png(cells, plan)
        except Exception as exc:  # noqa: BLE001
            raise _as_decode_error(path, exc) from exc

    def scalar_plan(self, path, *, channel=0, t=0, sampling="box") -> dict[str, Any]:
        source = self._source(path)
        _require_bounded_volume_source(source)
        channel_index = _exact_index(0 if channel is None else channel, "channel")
        time_index = _exact_index(0 if t is None else t, "time")
        if channel_index < 0 or channel_index >= source.c:
            raise ValueError(
                f"scalar volume channel index {channel_index} is out of range for C={source.c}"
            )
        if time_index < 0 or time_index >= source.t:
            raise ValueError(
                f"scalar volume time index {time_index} is out of range for T={source.t}"
            )
        sampling_mode = str(sampling or "box").strip().lower()
        if sampling_mode not in {"box", "nearest"}:
            raise ValueError("unsupported scalar volume sampling")
        dtype, bytes_per_voxel, _canonical_dtype = _scalar_dtype(source.dtype)
        if (
            sampling_mode == "nearest"
            and scalar_semantics.mask_membership_dtype(source.dtype) is None
        ):
            raise ValueError(
                "unsupported nearest scalar sampling: exact Mask integer source required"
            )
        preview = (
            atlas_mod.plan_scalar_mask_native(
                source.x,
                source.y,
                max(1, source.z),
                dtype=dtype,
                bytes_per_voxel=bytes_per_voxel,
            )
            if sampling_mode == "nearest"
            else atlas_mod.plan_scalar_preview(
                source.x,
                source.y,
                max(1, source.z),
                spacing=(
                    source.spacing_zyx[2],
                    source.spacing_zyx[1],
                    source.spacing_zyx[0],
                ),
            )
        )
        plan = {
            **preview,
            "dtype": dtype,
            "bytes_per_voxel": bytes_per_voxel,
            "pages": 0,
            "channel": channel_index,
            "t": time_index,
            "sampling": sampling_mode,
            "preview_policy": preview["preview_policy"],
        }
        if plan["preview_policy"] == atlas_mod.SCALAR_MASK_NATIVE_POLICY:
            generation = getattr(source, "source_generation", None)
            if generation is None:
                generation = _source_generation(path)
            generation = atlas_mod.validate_scalar_source_generation(generation)
            admitted_work, read_count = _complete_exact_mask_decode_admission(source, plan)
            plan.update(
                {
                    "decode_admission": atlas_mod.SCALAR_DECODE_ADMISSION,
                    "admitted_decode_work_bytes": admitted_work,
                    "admitted_decode_read_count": read_count,
                    "admitted_source_dtype": dtype,
                    "admitted_source_bytes_per_voxel": bytes_per_voxel,
                    "source_generation": generation,
                }
            )
            _require_source_generation(path, generation)
            atlas_mod.validate_scalar_decoder_admission(plan)
        return plan

    def _bounded_scalar_plane(self, source: _Plane, plan: dict[str, Any], output_z: int):
        """BOX-average one delivered plane without materializing a source plane.

        Every decoder read has an explicit region no larger than
        ``_SOURCE_REGION_MAX_BYTES``. The accumulator is delivery-sized, so both
        I/O and resident memory stay bounded by the preview contract rather than
        native X/Y geometry.
        """
        np = self._np
        factor_x = int(plan["downsample_x"])
        factor_y = int(plan["downsample_y"])
        factor_z = int(plan["downsample_z"])
        height = int(plan["height"])
        width = int(plan["width"])
        source_start = output_z * factor_z
        source_end = min(source.z, source_start + factor_z)
        if plan.get("sampling") == "nearest":
            source_z = min(
                source.z - 1,
                source_start + factor_z // 2,
            )
            x_indices = np.minimum(
                source.x - 1,
                np.arange(width, dtype="int64") * factor_x + factor_x // 2,
            )
            y_indices = np.minimum(
                source.y - 1,
                np.arange(height, dtype="int64") * factor_y + factor_y // 2,
            )
            output = np.empty((height, width), dtype=source.dtype)
            source_itemsize = max(1, self._np.dtype(source.dtype).itemsize)
            max_voxels = max(1, _SOURCE_REGION_MAX_BYTES // source_itemsize)
            output_x0 = 0
            while output_x0 < width:
                output_x1 = output_x0 + 1
                while (
                    output_x1 < width
                    and int(x_indices[output_x1]) - int(x_indices[output_x0]) + 1 <= max_voxels
                ):
                    output_x1 += 1
                selected_x = x_indices[output_x0:output_x1]
                x0 = int(selected_x[0])
                x1 = int(selected_x[-1]) + 1
                source_width = x1 - x0
                max_source_height = max(1, max_voxels // source_width)
                output_y0 = 0
                while output_y0 < height:
                    output_y1 = output_y0 + 1
                    while (
                        output_y1 < height
                        and int(y_indices[output_y1]) - int(y_indices[output_y0]) + 1
                        <= max_source_height
                    ):
                        output_y1 += 1
                    selected_y = y_indices[output_y0:output_y1]
                    y0 = int(selected_y[0])
                    y1 = int(selected_y[-1]) + 1
                    raw = np.asarray(
                        source.read(
                            t=plan["t"],
                            c=plan["channel"],
                            z=source_z,
                            level=0,
                            box=(y0, y1, x0, x1),
                        )
                    )
                    if raw.shape != (y1 - y0, x1 - x0) or raw.nbytes > _SOURCE_REGION_MAX_BYTES:
                        raise ValueError("cannot decode a bounded nearest source region")
                    output[output_y0:output_y1, output_x0:output_x1] = raw[selected_y - y0][
                        :, selected_x - x0
                    ]
                    output_y0 = output_y1
                output_x0 = output_x1
            return output

        accumulator = np.zeros((height, width), dtype="float64")
        counts = np.zeros((height, width), dtype="uint32")

        source_itemsize = max(1, self._np.dtype(source.dtype).itemsize)
        max_voxels = max(1, _SOURCE_REGION_MAX_BYTES // source_itemsize)
        block_width = min(source.x, max_voxels)
        block_height = max(1, min(source.y, max_voxels // block_width))

        for source_z in range(source_start, source_end):
            for y0 in range(0, source.y, block_height):
                y1 = min(source.y, y0 + block_height)
                y_bins = np.arange(y0, y1, dtype="int64") // factor_y
                for x0 in range(0, source.x, block_width):
                    x1 = min(source.x, x0 + block_width)
                    raw = np.asarray(
                        source.read(
                            t=plan["t"],
                            c=plan["channel"],
                            z=source_z,
                            level=0,
                            box=(y0, y1, x0, x1),
                        ),
                        dtype="float64",
                    )
                    if raw.shape != (y1 - y0, x1 - x0):
                        raise ValueError(
                            "cannot decode bounded source region with the advertised geometry"
                        )
                    x_bins = np.arange(x0, x1, dtype="int64") // factor_x
                    for delivered_y in np.unique(y_bins):
                        rows = raw[y_bins == delivered_y]
                        column_sums = rows.sum(axis=0, dtype="float64")
                        np.add.at(accumulator[delivered_y], x_bins, column_sums)
                        np.add.at(counts[delivered_y], x_bins, rows.shape[0])

        if np.any(counts == 0):
            raise ValueError("cannot decode scalar preview with empty delivery cells")
        return accumulator / counts

    def _bounded_mask_plane(
        self,
        source: _Plane,
        *,
        t: int,
        channel: int,
        z: int,
        threshold: float,
    ):
        """Threshold one exact source plane while bounding every decoder region."""
        np = self._np
        if scalar_semantics.mask_membership_dtype(source.dtype) is None:
            raise ValueError("source dtype cannot preserve exact mask membership")
        canonical_threshold = scalar_semantics.canonical_mask_threshold(threshold, source.dtype)
        regions = _exact_plane_regions(source)
        _admit_decode_reads(
            source,
            ((t, channel, z, region) for region in regions),
            expected_read_count=len(regions),
            label="mask plane",
        )
        output = np.empty((int(source.y), int(source.x)), dtype="uint8")
        for y0, y1, x0, x1 in regions:
            raw = np.asarray(
                source.read(
                    t=t,
                    c=channel,
                    z=z,
                    level=0,
                    box=(y0, y1, x0, x1),
                )
            )
            if raw.shape != (y1 - y0, x1 - x0) or raw.nbytes > _SOURCE_REGION_MAX_BYTES:
                raise ValueError("cannot decode a bounded exact mask region")
            output[y0:y1, x0:x1] = np.where(raw > canonical_threshold, 255, 0)
        return output

    def scalar_planes(
        self,
        path,
        *,
        zs,
        channel,
        t,
        pages,
        sampling="box",
        plan: dict[str, Any] | None = None,
    ):
        np = self._np
        if plan is None:
            plan = self.scalar_plan(path, channel=channel, t=t, sampling=sampling)
        source = self._source(path)
        if str(plan.get("sampling", "box")).strip().lower() == "nearest":
            atlas_mod.validate_scalar_decoder_admission(plan)
            generation = _require_source_generation(path, plan.get("source_generation"))
            source_generation = atlas_mod.validate_scalar_source_generation(
                getattr(source, "source_generation", None)
            )
            if source_generation != generation:
                raise ValueError("exact Mask selected source generation does not match its plan")
            source_dtype, source_bytes_per_voxel, _canonical_dtype = _scalar_dtype(source.dtype)
            plan_channel = _exact_index(plan.get("channel"), "channel")
            plan_time = _exact_index(plan.get("t"), "time")
            plan_pages = _exact_index(plan.get("pages"), "pages")
            worker_channel = _exact_index(channel, "channel")
            worker_time = _exact_index(t, "time")
            worker_pages = _exact_index(pages, "pages")
            if (
                int(plan["source_width"]) != int(source.x)
                or int(plan["source_height"]) != int(source.y)
                or int(plan["source_depth"]) != int(source.z)
                or plan_channel < 0
                or plan_channel >= int(source.c)
                or plan_time < 0
                or plan_time >= int(source.t)
                or plan_pages != 0
                or plan_channel != worker_channel
                or plan_time != worker_time
                or plan_pages != worker_pages
                or str(plan["sampling"]).strip().lower() != str(sampling).strip().lower()
                or str(plan["dtype"]).strip().lower() != source_dtype
                or int(plan["bytes_per_voxel"]) != source_bytes_per_voxel
                or str(plan["admitted_source_dtype"]).strip().lower() != source_dtype
                or int(plan["admitted_source_bytes_per_voxel"]) != source_bytes_per_voxel
            ):
                raise ValueError("exact Mask worker plan does not match the selected source")
            complete_work, complete_read_count = _complete_exact_mask_decode_admission(source, plan)
            if complete_work != int(plan["admitted_decode_work_bytes"]):
                raise ValueError("exact Mask decode work does not match its admission plan")
            if complete_read_count != int(plan["admitted_decode_read_count"]):
                raise ValueError("exact Mask read count does not match its admission plan")
        _dtype_name, _bytes_per_voxel, canonical_dtype = _scalar_dtype(source.dtype)
        output_indices = _bounded_output_indices(zs, int(plan["depth"]))
        if plan["sampling"] == "nearest":
            regions = _nearest_plane_regions(source, plan)
            _admit_decode_reads(
                source,
                (
                    (
                        int(plan["t"]),
                        int(plan["channel"]),
                        min(
                            int(source.z) - 1,
                            output_z * int(plan["downsample_z"]) + int(plan["downsample_z"]) // 2,
                        ),
                        region,
                    )
                    for output_z in output_indices
                    for region in regions
                ),
                expected_read_count=len(output_indices) * len(regions),
                label="nearest scalar volume",
            )
        else:
            regions = _exact_plane_regions(source)
            factor_z = int(plan["downsample_z"])
            read_count = sum(
                min(int(source.z), (output_z + 1) * factor_z) - output_z * factor_z
                for output_z in output_indices
            ) * len(regions)
            _admit_decode_reads(
                source,
                (
                    (
                        int(plan["t"]),
                        int(plan["channel"]),
                        source_z,
                        region,
                    )
                    for output_z in output_indices
                    for source_z in range(
                        output_z * factor_z,
                        min(int(source.z), (output_z + 1) * factor_z),
                    )
                    for region in regions
                ),
                expected_read_count=read_count,
                label="box scalar volume",
            )
        out: list[Any] = []
        for output_z in output_indices:
            plane = self._bounded_scalar_plane(source, plan, output_z)
            if canonical_dtype.kind in ("u", "i"):
                limits = np.iinfo(canonical_dtype)
                plane = np.clip(np.rint(plane), limits.min, limits.max)
            out.append(np.ascontiguousarray(plane, dtype=canonical_dtype))
        if plan["sampling"] == "nearest":
            _require_source_generation(path, generation)
        return out

    def scalar_volume(self, path, *, channel=0, t=0, sampling="box") -> dict[str, Any]:
        try:
            plan = self.scalar_plan(path, channel=channel, t=t, sampling=sampling)
            # Validate BEFORE materializing: an over-budget volume must be refused
            # up front, not after allocating hundreds of megabytes.
            atlas_mod.validate_scalar_plan(plan)
            planes = self.scalar_planes(
                path,
                zs=range(plan["depth"]),
                channel=plan["channel"],
                t=plan["t"],
                pages=plan["pages"],
                sampling=plan["sampling"],
                plan=(
                    plan if plan["preview_policy"] == atlas_mod.SCALAR_MASK_NATIVE_POLICY else None
                ),
            )
            return atlas_mod.build_scalar_volume_dict(planes, plan["channel"], plan)
        except Exception as exc:  # noqa: BLE001
            raise _as_decode_error(path, exc) from exc

    def histogram(self, path, *, bins=256, channels=None, t=0, scope="volume") -> dict[str, Any]:
        source = self._source(path)
        _require_single_scene(source)
        normalized_scope = str(scope).strip().lower()
        if normalized_scope not in {"display", "volume"}:
            raise ValueError("unsupported histogram scope")
        time_index = _axis_index(t, "time", source.t)
        zero_based = _zero_based(channels, source.c)
        if normalized_scope == "display":
            level = self._level_for_size(source, _WINDOW_SAMPLE_EDGE)
            height, width = source.level_shapes[level]
            crop_h = min(height, _WINDOW_SAMPLE_EDGE)
            crop_w = min(width, _WINDOW_SAMPLE_EDGE)
            box = (
                max(0, (height - crop_h) // 2),
                max(0, (height - crop_h) // 2) + crop_h,
                max(0, (width - crop_w) // 2),
                max(0, (width - crop_w) // 2) + crop_w,
            )
            z_index = max(0, source.z // 2)
            planned_work = [
                source.estimate_read_work(
                    t=time_index,
                    c=channel,
                    z=z_index,
                    level=level,
                    box=box,
                )
                for channel in zero_based
            ]
            if (
                any(
                    isinstance(work, bool) or not isinstance(work, int) or work <= 0
                    for work in planned_work
                )
                or sum(planned_work) > scalar_semantics.MAX_PROFILE_DECODE_WORK_BYTES
            ):
                raise ValueError("display histogram decode work exceeds its bounded envelope")
            samples = [
                (
                    channel,
                    source.read(
                        t=time_index,
                        c=channel,
                        z=z_index,
                        level=level,
                        box=box,
                    ),
                )
                for channel in zero_based
            ]
            return scalar_semantics.display_histogram(
                samples,
                dtype=source.dtype,
                bins=int(bins),
                t=time_index,
                algorithm="bounded-center-plane-v1",
            )
        try:
            profiles = [
                self._scalar_profile(path, channel=channel, t=time_index, bins=int(bins))
                for channel in zero_based
            ]
            first = profiles[0]
            return {
                **first,
                "channels": [profile["channels"][0] for profile in profiles],
            }
        except Exception as exc:  # noqa: BLE001
            raise _as_decode_error(path, exc) from exc


def _parse_colors(colors: Any, count: int) -> list[tuple[float, float, float] | None]:
    """Normalize the service's colour spec into (r,g,b) floats per channel.

    ``None`` entries are PRESERVED: composite_channels treats a colourless
    channel as contributing nothing, which is how the native engine behaves.
    """
    out: list[tuple[float, float, float] | None] = []
    values = list(colors) if isinstance(colors, (list, tuple)) else []
    for i in range(count):
        parsed: tuple[float, float, float] | None = None
        if i < len(values):
            value = values[i]
            if isinstance(value, str):
                parsed = fusion.parse_hex_color(value)
            elif isinstance(value, (list, tuple)) and len(value) >= 3:
                parsed = (float(value[0]), float(value[1]), float(value[2]))
            elif value is None:
                out.append(None)
                continue
        out.append(parsed or fusion.convention_channel_color(i))
    return out
