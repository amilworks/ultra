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

import io
import math
import operator
import os
from collections.abc import Sequence
from typing import Any

from ultra_deepagents.imaging import atlas as atlas_mod
from ultra_deepagents.imaging import fusion, viewerinfo
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
_SCALAR_SOURCE_WORK_MAX_BYTES = 512 * 1024 * 1024 + _SOURCE_REGION_MAX_BYTES

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
            try:
                size = operator.index(raw)
            except TypeError as exc:
                raise ValueError("decoded chunk geometry is invalid") from exc
            if size <= 0:
                raise ValueError("decoded chunk geometry is invalid")
            sizes.append(size)
        shape.append(max(sizes))
    return int(math.prod(shape)) * int(np.dtype(dtype).itemsize)


def _zero_based(channels: Sequence[int] | None, count: int) -> list[int]:
    """Convert the interface's 1-BASED channel indices to 0-based array indices.

    ``None`` means "every channel". Empty, duplicate, or out-of-range selections
    are rejected before any pixel read so a malformed request cannot silently
    render a neighboring channel.
    """
    if channels is None:
        return list(range(max(count, 1)))
    values = list(channels)
    if not values:
        raise ValueError("channel selection must not be empty")
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

    def close(self) -> None:
        self._zarr_cache.clear()
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
        source_order = ""
        try:
            scenes = list(self._img.scenes)
            scene_count = max(1, len(scenes))
        except Exception:  # noqa: BLE001 - single-scene sources
            pass
        try:
            source_order = str(self._img.dims.order or "")
        except Exception:  # noqa: BLE001
            source_order = ""
        self._dask = self._img.get_image_dask_data("TCZYX")
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
        if scene_count > 1:
            # BioIO channel names and physical sizes are scene-local only after a
            # scene has been selected. Do not attach scene-0 values to the whole file.
            names = []
            spacing = (1.0, 1.0, 1.0)
            spacing_units = ("voxel", "voxel", "voxel")
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
            max_decoded_chunk_bytes=_max_chunk_bytes(self._dask.chunks, self._dask.dtype),
        )

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

    def close(self) -> None:
        self._img = None
        self._dask = None


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

    # -- source handling ------------------------------------------------------
    def _source(self, path: str) -> _Plane:
        try:
            stamp = os.stat(path).st_mtime_ns
        except OSError as exc:
            raise ValueError(f"cannot decode {os.path.basename(path)}: {exc}") from exc
        key = (str(path), stamp)
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
        zero_based = _zero_based(channels, source.c)
        out = self._planes_uint8(
            source, path, t=t, z=z, level=level, zero_based=zero_based, colors=colors, box=box
        )
        if max_dim:
            out = self._downscale(out, max_dim)
        return self._encode_png(out)

    def _downscale(self, arr, max_dim: int):
        np = self._np
        h, w = int(arr.shape[0]), int(arr.shape[1])
        long_edge = max(h, w)
        if long_edge <= max_dim or long_edge == 0:
            return arr
        scale = max_dim / float(long_edge)
        size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        img = self._Image.fromarray(np.asarray(arr, dtype="uint8"))
        return np.asarray(img.resize(size, self._Image.BILINEAR))

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
        if source.scene_count == 1:
            meta["selected_scene_index"] = 0
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
        return viewerinfo.build_viewer_info(
            meta,
            signal_scores=(
                self._channel_signal_scores(path)
                if int(meta.get("image_num_scenes", 1) or 1) == 1
                else None
            ),
            reader=reader,
        )

    def _channel_signal_scores(self, path: str) -> list[float] | None:
        """Bounded per-channel spatial correlation from one centered mid-Z crop."""
        source = self._source(path)
        if source.c <= 1:
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
        channels=None,
        colors=None,
        windows=None,
    ) -> bytes:
        source = self._source(path)
        _require_single_scene(source)
        lvl = max(0, min(int(level), source.level_count - 1))
        height, width = source.level_shapes[lvl]
        y0, x0 = int(row) * int(tile_size), int(col) * int(tile_size)
        if y0 >= height or x0 >= width:
            raise ValueError(f"engine returned an empty region (tile {col},{row} past level {lvl})")
        box = (y0, min(y0 + int(tile_size), height), x0, min(x0 + int(tile_size), width))
        try:
            return self._render(
                source, path, t=0, z=0, level=lvl, channels=channels, colors=colors, box=box
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
    ) -> bytes:
        hdf5_png = self._maybe_hdf5_slice(path, z)
        if hdf5_png is not None:
            return hdf5_png
        source = self._source(path)
        _require_single_scene(source)
        time_index = _axis_index(0 if t is None else t, "time", source.t)
        depth_index = _axis_index(0 if z is None else z, "z", source.z)
        _zero_based(channels, source.c)
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

    def scalar_plan(self, path, *, channel=0, t=0) -> dict[str, Any]:
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
        dtype, bytes_per_voxel, _canonical_dtype = _scalar_dtype(source.dtype)
        preview = atlas_mod.plan_scalar_preview(
            source.x,
            source.y,
            max(1, source.z),
            spacing=(source.spacing_zyx[2], source.spacing_zyx[1], source.spacing_zyx[0]),
        )
        return {
            **preview,
            "dtype": dtype,
            "bytes_per_voxel": bytes_per_voxel,
            "pages": 0,
            "channel": channel_index,
            "t": time_index,
        }

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

    def scalar_planes(self, path, *, zs, channel, t, pages):
        np = self._np
        source = self._source(path)
        plan = self.scalar_plan(path, channel=channel, t=t)
        _dtype_name, _bytes_per_voxel, canonical_dtype = _scalar_dtype(source.dtype)
        out: list[Any] = []
        for output_z in zs:
            output_z = _exact_index(output_z, "z")
            if output_z < 0 or output_z >= int(plan["depth"]):
                raise ValueError(f"scalar volume z index {output_z} is out of range")
            plane = self._bounded_scalar_plane(source, plan, output_z)
            if canonical_dtype.kind in ("u", "i"):
                limits = np.iinfo(canonical_dtype)
                plane = np.clip(np.rint(plane), limits.min, limits.max)
            out.append(np.ascontiguousarray(plane, dtype=canonical_dtype))
        return out

    def scalar_volume(self, path, *, channel=0, t=0) -> dict[str, Any]:
        try:
            plan = self.scalar_plan(path, channel=channel, t=t)
            # Validate BEFORE materializing: an over-budget volume must be refused
            # up front, not after allocating hundreds of megabytes.
            atlas_mod.validate_scalar_plan(plan)
            planes = self.scalar_planes(
                path,
                zs=range(plan["depth"]),
                channel=plan["channel"],
                t=plan["t"],
                pages=plan["pages"],
            )
            return atlas_mod.build_scalar_volume_dict(planes, plan["channel"], plan)
        except Exception as exc:  # noqa: BLE001
            raise _as_decode_error(path, exc) from exc

    def histogram(self, path, *, bins=256, channels=None, t=0) -> dict[str, Any]:
        np = self._np
        source = self._source(path)
        _require_single_scene(source)
        time_index = _axis_index(t, "time", source.t)
        zero_based = _zero_based(channels, source.c)
        lvl = self._level_for_size(source, 1024)
        height, width = source.level_shapes[lvl]
        crop_h, crop_w = min(height, 1024), min(width, 1024)
        y0, x0 = max(0, (height - crop_h) // 2), max(0, (width - crop_w) // 2)
        try:
            out = []
            for c in zero_based:
                plane = np.asarray(
                    source.read(
                        t=time_index,
                        c=c,
                        z=max(source.z // 2, 0),
                        level=lvl,
                        box=(y0, y0 + crop_h, x0, x0 + crop_w),
                    )
                ).astype("float32")
                counts, edges = np.histogram(plane.reshape(-1), bins=int(bins))
                out.append(
                    {
                        "index": c,
                        "counts": [int(v) for v in counts],
                        "min": float(edges[0]),
                        "max": float(edges[-1]),
                    }
                )
            return {"bins": int(bins), "channels": out}
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
