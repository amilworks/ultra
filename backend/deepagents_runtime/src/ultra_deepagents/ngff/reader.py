"""OME-NGFF (OME-Zarr) reader — lazy, chunk-level, spec-correct.

Parses the OME-NGFF metadata (`multiscales`, `axes`, `omero`) from the store's
attributes and exposes lazy access to each multiscale level via zarr-python. Reads
are always bounded to the requested plane/region, so the whole (possibly 1 TB) array
is never materialized.

Supports Zarr v2 and v3 stores (zarr-python ≥3 reads both). Axis handling is driven
by the NGFF `axes[].name` (the spec's canonical t/c/z/y/x), tolerating arbitrary axis
order and optional canonical axes (a 2D YX store, a TYX time-lapse, or a
multichannel ZYX volume). Generic Zarr arrays without declared NGFF axes are
rejected rather than assigned guessed trailing dimensions.
"""

from __future__ import annotations

import json
import math
import os
import stat
import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import product
from numbers import Real
from typing import Any, cast

import numpy as np

__all__ = ["NgffError", "NgffLevel", "NgffChannel", "NgffImage", "open_ngff", "is_ome_zarr"]

# Canonical dimension order Ultra's viewer-info uses.
_CANONICAL = ("t", "c", "z", "y", "x")
_SUPPORTED_NGFF_VERSIONS = frozenset({"0.4", "0.5"})
_MAX_MULTISCALES = 256
_MAX_LEVELS = 256
_MAX_MULTISCALE_NAME_LENGTH = 256
_MAX_ZARR_PATH_LENGTH = 1024

# Decoded-plane cache (the application-level chunk cache, layered above the OS page cache
# and below the Go edge tile LRU). The byte budget is process-wide, not per image: otherwise
# the open-image LRU could multiply a 256 MiB setting by 64 images (and again by every
# worker). A full 2D plane is cached per (image,level,t,c,z) only when it is below a byte
# threshold, while gigapixel level-0 planes are never materialized whole.
_PLANE_CACHE_MAX_BYTES = int(
    os.environ.get("ULTRA_NGFF_DECODE_CACHE_BYTES", str(256 * 1024 * 1024))
)
_PLANE_CACHE_MAX_PLANE_BYTES = int(
    os.environ.get("ULTRA_NGFF_DECODE_CACHE_PLANE_BYTES", str(48 * 1024 * 1024))
)
# Freshness checks are proportional to the chunks that back a cacheable plane, never to
# the size of the complete store. Pathological tiny-chunk planes skip the decoded cache.
_PLANE_CACHE_MAX_DEPENDENCY_FILES = int(
    os.environ.get("ULTRA_NGFF_DECODE_CACHE_DEPENDENCY_FILES", "4096")
)
_PLANE_CACHE_MAX_ENTRIES = int(os.environ.get("ULTRA_NGFF_DECODE_CACHE_ENTRIES", "4096"))
_PROCESS_PLANE_CACHE_LOCK = threading.RLock()
_PROCESS_PLANE_CACHE_BYTES = 0
_PROCESS_PLANE_CACHE_LRU: OrderedDict[
    tuple[int, tuple[int, int, int, int]],
    tuple[weakref.ReferenceType[NgffImage], int],
] = OrderedDict()
# A display window is estimated from nine spatially distributed patches. Each patch is
# at most this edge length and at most one quarter of the corresponding image dimension,
# so a tile request cannot accidentally decode a complete thumbnail level for contrast.
_DISPLAY_SAMPLE_PATCH_EDGE = int(os.environ.get("ULTRA_NGFF_DISPLAY_SAMPLE_PATCH_EDGE", "128"))
# Metadata must never present a sampled plane as the image's true range.  The exact
# min/max scan is therefore allowed only when the *entire* smallest multiscale array
# fits both caps.  Larger arrays report the range as unknown instead of sampling.
_INTENSITY_RANGE_MAX_ELEMENTS = int(
    os.environ.get("ULTRA_NGFF_INTENSITY_RANGE_MAX_ELEMENTS", "5000000")
)
_INTENSITY_RANGE_MAX_BYTES = int(
    os.environ.get("ULTRA_NGFF_INTENSITY_RANGE_MAX_BYTES", str(64 * 1024 * 1024))
)
# Full-plane reads back the direct 2D/scrub endpoints. Large images are advertised through
# the bounded tile path, so a direct read must never materialize an unbounded level-0 plane.
# Elements and source bytes are capped independently because dtype width varies.
_MAX_PLANE_READ_ELEMENTS = int(
    os.environ.get("ULTRA_NGFF_MAX_PLANE_READ_ELEMENTS", str(64 * 1024 * 1024))
)
_MAX_PLANE_READ_BYTES = int(
    os.environ.get("ULTRA_NGFF_MAX_PLANE_READ_BYTES", str(256 * 1024 * 1024))
)
# Zarr decodes a complete logical chunk before applying an array slice. This cap prevents a
# hostile or accidental chunk declaration from turning one bounded tile into a multi-GB decode.
_MAX_CHUNK_DECODED_BYTES = int(
    os.environ.get("ULTRA_NGFF_MAX_CHUNK_DECODED_BYTES", str(256 * 1024 * 1024))
)
# Chunk-file validation is proportional to the exact read. Tiny-chunk stores that exceed this
# budget are rejected for that read instead of causing an unbounded filesystem walk.
_MAX_CHUNK_FILES_PER_READ = int(os.environ.get("ULTRA_NGFF_MAX_CHUNK_FILES_PER_READ", str(1 << 16)))
# Root metadata is upload-controlled. Bound bytes and nesting before json.loads so malformed
# attributes fail as a domain error instead of exhausting memory or Python recursion.
_MAX_METADATA_JSON_BYTES = int(
    os.environ.get("ULTRA_NGFF_MAX_METADATA_BYTES", str(8 * 1024 * 1024))
)
_MAX_METADATA_JSON_DEPTH = int(os.environ.get("ULTRA_NGFF_MAX_METADATA_DEPTH", "200"))
# The current display renderer is defined only for real scalar samples. Rejecting other kinds
# avoids silently dropping the imaginary component or misinterpreting structured values.
_RENDERABLE_DTYPE_KINDS = frozenset("biuf")


class NgffError(RuntimeError):
    """The store is not a readable OME-Zarr (missing/invalid NGFF metadata)."""


_ChunkStamp = tuple[str, int, int, int, int, int, int]
_PlaneCacheEntry = tuple[np.ndarray, tuple[_ChunkStamp, ...]]


def _drop_process_plane_cache_owner(owner_id: int) -> None:
    """Forget accounting for a collected image without retaining that image globally."""

    global _PROCESS_PLANE_CACHE_BYTES
    with _PROCESS_PLANE_CACHE_LOCK:
        owned = [token for token in _PROCESS_PLANE_CACHE_LRU if token[0] == owner_id]
        for token in owned:
            _owner_ref, nbytes = _PROCESS_PLANE_CACHE_LRU.pop(token)
            _PROCESS_PLANE_CACHE_BYTES -= nbytes


def process_plane_cache_info() -> dict[str, int]:
    """Return process-wide decoded-plane cache accounting for health/qualification."""

    with _PROCESS_PLANE_CACHE_LOCK:
        return {
            "bytes": _PROCESS_PLANE_CACHE_BYTES,
            "max_bytes": _PLANE_CACHE_MAX_BYTES,
            "entries": len(_PROCESS_PLANE_CACHE_LRU),
            "max_entries": _PLANE_CACHE_MAX_ENTRIES,
        }


def clear_process_plane_cache() -> None:
    """Evict every decoded plane while leaving opened Zarr metadata reusable."""

    global _PROCESS_PLANE_CACHE_BYTES
    with _PROCESS_PLANE_CACHE_LOCK:
        for token, (owner_ref, _nbytes) in tuple(_PROCESS_PLANE_CACHE_LRU.items()):
            owner = owner_ref()
            if owner is not None:
                entry = owner._plane_cache.pop(token[1], None)
                if entry is not None:
                    owner._plane_cache_bytes -= int(entry[0].nbytes)
        _PROCESS_PLANE_CACHE_LRU.clear()
        _PROCESS_PLANE_CACHE_BYTES = 0


@dataclass
class NgffLevel:
    """One multiscale level: its dataset path, the lazy zarr array, and its per-axis
    effective array-to-physical scale and translation.

    ``scale`` and ``translation`` are the result of applying the dataset
    ``coordinateTransformations`` in order and then the selected multiscale's
    transformations in order. ``axis_units`` is aligned with both vectors and
    with :attr:`NgffImage.axes`.
    """

    path: str
    array: Any  # zarr.Array (lazy)
    shape: tuple[int, ...]
    scale: tuple[float, ...]
    translation: tuple[float, ...]
    axis_units: tuple[str, ...]


@dataclass
class NgffChannel:
    """Rendering hints for one channel, from the OME `omero` block (when present)."""

    label: str
    color: str  # "RRGGBB" hex (no '#')
    window_start: float | None
    window_end: float | None
    active: bool


@dataclass
class NgffImage:
    """An opened OME-Zarr image: axis layout, multiscale levels, and channel render hints."""

    path: str
    axes: list[str]  # lowercase names in stored order, e.g. ["t", "c", "y", "x"]
    levels: list[NgffLevel]
    dtype: np.dtype
    channels: list[NgffChannel]
    sizes: dict[str, int]  # full-res size per canonical axis; 1 when the axis is absent
    physical: dict[str, float]  # effective units/pixel per canonical axis
    translation: dict[str, float]  # effective physical translation per canonical axis
    units: dict[str, str]  # NGFF axis unit per canonical axis (e.g. x/y "micrometer", t "hour")
    omero: dict[str, Any] | None
    multiscale_name: str | None
    multiscale_index: int | None = None
    name: str | None = None  # human image name (multiscales.name or omero.name)
    version: str | None = None  # NGFF version (e.g. "0.4" / "0.5")
    # Cached exact intensity range over the complete smallest multiscale level.  The
    # associated status makes a budget-limited unknown distinguishable from a true value.
    _intensity_range: tuple[float, float] | None = None
    _intensity_range_evaluated: bool = False
    _intensity_range_status: str = "not_evaluated"
    # Cache of the resolved global display window per channel index — computed once
    # (omero-if-valid, else auto-contrast on the smallest level) so every slice/tile
    # renders with the SAME mapping (no per-tile checkerboard). Populated by the renderer.
    window_cache: dict[int, tuple[float, float]] = field(default_factory=dict)
    # Size-bounded LRU of decoded full planes (see module docstring). Kept on the image so
    # it lives exactly as long as the per-(path,mtime) open-image cache entry.
    _plane_cache: OrderedDict[tuple[int, int, int, int], _PlaneCacheEntry] = field(
        default_factory=OrderedDict
    )
    _plane_cache_bytes: int = 0
    _plane_cache_lock: threading.RLock = field(
        default_factory=lambda: _PROCESS_PLANE_CACHE_LOCK, repr=False, compare=False
    )
    _plane_cache_owner_ref: weakref.ReferenceType[NgffImage] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        owner_id = id(self)
        self._plane_cache_owner_ref = weakref.ref(
            self,
            lambda _reference, owner_id=owner_id: _drop_process_plane_cache_owner(owner_id),
        )

    # --- canonical-size convenience ---
    @property
    def num_t(self) -> int:
        return self.sizes["t"]

    @property
    def num_c(self) -> int:
        return self.sizes["c"]

    @property
    def num_z(self) -> int:
        return self.sizes["z"]

    @property
    def num_y(self) -> int:
        return self.sizes["y"]

    @property
    def num_x(self) -> int:
        return self.sizes["x"]

    def _axis_pos(self, axis: str) -> int | None:
        try:
            return self.axes.index(axis)
        except ValueError:
            return None

    def _validated_level(self, level: int) -> int:
        if isinstance(level, (bool, np.bool_)) or not isinstance(level, (int, np.integer)):
            raise NgffError("multiscale level must be an integer")
        value = int(level)
        if value < 0 or value >= len(self.levels):
            raise NgffError(f"multiscale level {value} is outside [0, {len(self.levels) - 1}]")
        return value

    @staticmethod
    def _validated_axis_index(axis: str, value: int, size: int) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise NgffError(f"{axis} index must be an integer")
        index = int(value)
        if index < 0 or index >= size:
            raise NgffError(f"{axis} index {index} is outside [0, {size - 1}]")
        return index

    def _spatial_axis_order(self) -> tuple[str, str]:
        order = tuple(axis for axis in self.axes if axis in ("y", "x"))
        if order not in (("y", "x"), ("x", "y")):
            raise NgffError("OME-NGFF viewer requires exactly one explicit 'y' and 'x' axis")
        return cast(tuple[str, str], order)

    def _index_tuple(
        self,
        level: int,
        *,
        t: int,
        c: int,
        z: int,
        y: slice = slice(None),
        x: slice = slice(None),
    ) -> tuple[Any, ...]:
        """Build the zarr index tuple in stored axis order without substituting indices."""
        sel: dict[str, Any] = {}
        lvl_shape = self.levels[level].shape
        for canonical, value in (("t", t), ("c", c), ("z", z)):
            pos = self._axis_pos(canonical)
            size = int(lvl_shape[pos]) if pos is not None else 1
            sel[canonical] = self._validated_axis_index(canonical, value, size)
        sel["y"] = y
        sel["x"] = x
        index: list[Any] = []
        for axis in self.axes:
            if axis in ("y", "x"):
                index.append(sel[axis])
            elif axis in sel:
                index.append(sel[axis])
            else:
                index.append(0)  # an unexpected extra axis: take its first element
        return tuple(index)

    def _validate_full_plane_read(self, level: int) -> None:
        height, width = self.level_yx(level)
        elements = height * width
        if elements > _MAX_PLANE_READ_ELEMENTS:
            raise NgffError(
                f"plane {width}x{height} at level {level} exceeds the "
                f"{_MAX_PLANE_READ_ELEMENTS}-element full-plane read budget; request a "
                "coarser multiscale level or use the tile path"
            )
        source_bytes = elements * int(self.dtype.itemsize)
        if source_bytes > _MAX_PLANE_READ_BYTES:
            raise NgffError(
                f"plane {width}x{height} at level {level} requires {source_bytes} source "
                f"bytes, exceeding the {_MAX_PLANE_READ_BYTES}-byte source-byte budget; "
                "request a coarser multiscale level or use the tile path"
            )

    def _local_chunk_paths(
        self,
        *,
        level: int,
        t: int,
        c: int,
        z: int,
        y_range: tuple[int, int] | None = None,
        x_range: tuple[int, int] | None = None,
    ) -> tuple[str, ...] | None:
        """Resolve local chunk files that cover one exact read, when cheaply enumerable.

        Sharded or non-local stores do not expose a one-logical-chunk/one-file mapping and
        return ``None``. Managed bundle ingestion independently rejects symlinks for those
        stores. Ordinary local v2/v3 chunk layouts are screened here as defense in depth.
        """

        array = self.levels[level].array
        chunks = getattr(array, "chunks", None)
        metadata = getattr(array, "metadata", None)
        encode_chunk_key = getattr(metadata, "encode_chunk_key", None)
        store_root = getattr(getattr(array, "store", None), "root", None)
        if (
            not isinstance(chunks, tuple)
            or len(chunks) != len(self.axes)
            or not callable(encode_chunk_key)
            or not isinstance(store_root, (str, os.PathLike))
        ):
            return None
        codecs = getattr(metadata, "codecs", ())
        if any("sharding" in type(codec).__name__.lower() for codec in codecs or ()):
            return None

        fixed = {"t": t, "c": c, "z": z}
        coordinate_ranges: list[range] = []
        chunk_count = 1
        for axis, size, chunk_size in zip(self.axes, self.levels[level].shape, chunks, strict=True):
            if not isinstance(chunk_size, int) or chunk_size <= 0:
                return None
            if axis == "y" and y_range is not None:
                axis_range = range(
                    y_range[0] // chunk_size,
                    (y_range[1] - 1) // chunk_size + 1,
                )
            elif axis == "x" and x_range is not None:
                axis_range = range(
                    x_range[0] // chunk_size,
                    (x_range[1] - 1) // chunk_size + 1,
                )
            elif axis in ("y", "x"):
                axis_range = range((int(size) + chunk_size - 1) // chunk_size)
            else:
                index = fixed.get(axis, 0)
                axis_range = range(index // chunk_size, index // chunk_size + 1)
            chunk_count *= len(axis_range)
            if chunk_count > _MAX_CHUNK_FILES_PER_READ:
                raise NgffError(
                    f"pixel read spans {chunk_count} chunk files, exceeding the "
                    f"{_MAX_CHUNK_FILES_PER_READ}-file chunk-file budget; use a coarser "
                    "multiscale level or a larger source chunk layout"
                )
            coordinate_ranges.append(axis_range)

        root = os.fspath(store_root)
        array_path = str(getattr(array, "path", ""))
        base = os.path.join(root, *array_path.split("/")) if array_path else root
        paths: list[str] = []
        for coordinates in product(*coordinate_ranges):
            try:
                encoded = encode_chunk_key(coordinates)
            except Exception:  # Unsupported encodings use managed-ingestion guarantees.
                return None
            if not isinstance(encoded, str) or not encoded or os.path.isabs(encoded):
                return None
            parts = encoded.split("/")
            if any(part in ("", ".", "..") for part in parts):
                return None
            paths.append(os.path.join(base, *parts))
        return tuple(paths)

    def _validate_local_chunk_files(
        self,
        *,
        level: int,
        t: int,
        c: int,
        z: int,
        y_range: tuple[int, int] | None = None,
        x_range: tuple[int, int] | None = None,
    ) -> None:
        paths = self._local_chunk_paths(
            level=level,
            t=t,
            c=c,
            z=z,
            y_range=y_range,
            x_range=x_range,
        )
        if paths is None:
            return
        for chunk_path in paths:
            try:
                info = os.lstat(chunk_path)
            except FileNotFoundError:
                continue  # unwritten chunks legitimately resolve to the array fill value
            except OSError as exc:
                raise NgffError(f"cannot inspect a chunk file before pixel decode: {exc}") from exc
            if stat.S_ISLNK(info.st_mode):
                raise NgffError("refusing to decode a chunk file that is a symbolic link")
            if not stat.S_ISREG(info.st_mode):
                raise NgffError("refusing to decode a chunk path that is not a regular file")

    def read_plane(self, *, t: int = 0, c: int = 0, z: int = 0, level: int = 0) -> np.ndarray:
        """Read one full 2D (Y, X) plane for the given t/c/z at a multiscale level.
        Only the chunks covering that plane are fetched. Sub-threshold planes are cached
        (size-aware LRU) so scrub/channel re-renders skip NFS + decompression."""
        level = self._validated_level(level)
        # Validate before cache lookup too: absent axes have size one and only index zero
        # is meaningful; a bad request must never alias a cached valid plane.
        for axis, value in (("t", t), ("c", c), ("z", z)):
            pos = self._axis_pos(axis)
            size = int(self.levels[level].shape[pos]) if pos is not None else 1
            self._validated_axis_index(axis, value, size)
        self._validate_full_plane_read(level)
        key = (level, int(t), int(c), int(z))
        with self._plane_cache_lock:
            cached = self._plane_cache.get(key)
        if cached is not None:
            cached_plane, cached_signature = cached
            current_signature = self._plane_dependency_signature(
                level=level, t=int(t), c=int(c), z=int(z)
            )
            with self._plane_cache_lock:
                # Another request may have refreshed the entry while filesystem stats ran.
                current_entry = self._plane_cache.get(key)
                if current_entry is cached and current_signature == cached_signature:
                    self._plane_cache.move_to_end(key)
                    process_key = (id(self), key)
                    if process_key in _PROCESS_PLANE_CACHE_LRU:
                        _PROCESS_PLANE_CACHE_LRU.move_to_end(process_key)
                    return cached_plane
                if current_entry is cached:
                    self._evict_cached_plane(key)
                    # A changed chunk may also invalidate a previously sampled display window.
                    self.window_cache.clear()

        before_signature = self._plane_dependency_signature(
            level=level, t=int(t), c=int(c), z=int(z)
        )
        idx = self._index_tuple(level, t=t, c=c, z=z)
        plane = _as_2d_yx(
            np.asarray(self.levels[level].array[idx]),
            stored_spatial_order=self._spatial_axis_order(),
        )
        after_signature = self._plane_dependency_signature(
            level=level, t=int(t), c=int(c), z=int(z)
        )
        # Never cache a read that raced a writer or whose exact dependency set is too
        # large/unsupported to validate cheaply on subsequent requests.
        if before_signature is not None and before_signature == after_signature:
            self._maybe_cache_plane(key, plane, before_signature)
        return plane

    def _evict_cached_plane(self, key: tuple[int, int, int, int]) -> None:
        global _PROCESS_PLANE_CACHE_BYTES
        with _PROCESS_PLANE_CACHE_LOCK:
            entry = self._plane_cache.pop(key, None)
            if entry is not None:
                self._plane_cache_bytes -= int(entry[0].nbytes)
            process_entry = _PROCESS_PLANE_CACHE_LRU.pop((id(self), key), None)
            if process_entry is not None:
                _PROCESS_PLANE_CACHE_BYTES -= process_entry[1]

    def _plane_dependency_signature(
        self, *, level: int, t: int, c: int, z: int
    ) -> tuple[_ChunkStamp, ...] | None:
        """Return cheap identity stamps for every local chunk backing one plane.

        The signature deliberately uses only the selected plane's encoded chunk files.
        It therefore catches ordinary in-place and atomic chunk rewrites without hashing
        a potentially terabyte-scale store. Unsupported stores, sharded codecs, and planes
        with pathological chunk counts simply opt out of the decoded-plane cache.
        """

        array = self.levels[level].array
        chunks = getattr(array, "chunks", None)
        metadata = getattr(array, "metadata", None)
        encode_chunk_key = getattr(metadata, "encode_chunk_key", None)
        store_root = getattr(getattr(array, "store", None), "root", None)
        if (
            not isinstance(chunks, tuple)
            or len(chunks) != len(self.axes)
            or not callable(encode_chunk_key)
            or not isinstance(store_root, (str, os.PathLike))
        ):
            return None

        # A logical chunk can be packed into a shared shard; encoded logical keys then do
        # not identify the dependency file. Disable this optimization rather than guess.
        codecs = getattr(metadata, "codecs", ())
        if any("sharding" in type(codec).__name__.lower() for codec in codecs or ()):
            return None

        fixed = {"t": t, "c": c, "z": z}
        coordinate_ranges: list[range] = []
        dependency_count = 1
        for axis, size, chunk_size in zip(self.axes, self.levels[level].shape, chunks, strict=True):
            if not isinstance(chunk_size, int) or chunk_size <= 0:
                return None
            if axis in ("y", "x"):
                count = (int(size) + chunk_size - 1) // chunk_size
                axis_range = range(count)
            else:
                index = fixed.get(axis, 0)
                axis_range = range(index // chunk_size, index // chunk_size + 1)
            dependency_count *= len(axis_range)
            if dependency_count > _MAX_CHUNK_FILES_PER_READ:
                raise NgffError(
                    f"pixel read spans {dependency_count} chunk files, exceeding the "
                    f"{_MAX_CHUNK_FILES_PER_READ}-file chunk-file budget; use a coarser "
                    "multiscale level or a larger source chunk layout"
                )
            if dependency_count > _PLANE_CACHE_MAX_DEPENDENCY_FILES:
                return None
            coordinate_ranges.append(axis_range)

        root = os.fspath(store_root)
        array_path = str(getattr(array, "path", ""))
        base = os.path.join(root, *array_path.split("/")) if array_path else root
        stamps: list[_ChunkStamp] = []
        for coordinates in product(*coordinate_ranges):
            try:
                encoded = encode_chunk_key(coordinates)
            except Exception:  # noqa: BLE001 - unsupported key encodings skip the cache
                return None
            if not isinstance(encoded, str) or not encoded or os.path.isabs(encoded):
                return None
            parts = encoded.split("/")
            if any(part in ("", ".", "..") for part in parts):
                return None
            relative = "/".join((*array_path.split("/"), *parts)) if array_path else encoded
            chunk_path = os.path.join(base, *parts)
            try:
                info = os.stat(chunk_path, follow_symlinks=False)
            except FileNotFoundError:
                # Missing chunks are valid Zarr fill-value dependencies. If a writer later
                # creates one, the sentinel changes and invalidates the decoded plane.
                stamps.append((relative, -1, -1, -1, -1, -1, -1))
                continue
            except OSError:
                return None
            if stat.S_ISLNK(info.st_mode):
                raise NgffError("refusing to decode a chunk file that is a symbolic link")
            if not stat.S_ISREG(info.st_mode):
                return None
            stamps.append(
                (
                    relative,
                    int(info.st_mode),
                    int(info.st_size),
                    int(info.st_mtime_ns),
                    int(info.st_ctime_ns),
                    int(info.st_ino),
                    int(info.st_dev),
                )
            )
        return tuple(stamps)

    def _maybe_cache_plane(
        self,
        key: tuple[int, int, int, int],
        plane: np.ndarray,
        signature: tuple[_ChunkStamp, ...],
    ) -> None:
        global _PROCESS_PLANE_CACHE_BYTES
        nbytes = int(plane.nbytes)
        # Never materialize a gigapixel plane into the cache — tiles read bounded regions.
        if nbytes > _PLANE_CACHE_MAX_PLANE_BYTES or nbytes > _PLANE_CACHE_MAX_BYTES:
            return
        with _PROCESS_PLANE_CACHE_LOCK:
            self._evict_cached_plane(key)
            self._plane_cache[key] = (plane, signature)
            self._plane_cache.move_to_end(key)
            self._plane_cache_bytes += nbytes
            process_key = (id(self), key)
            _PROCESS_PLANE_CACHE_LRU[process_key] = (self._plane_cache_owner_ref, nbytes)
            _PROCESS_PLANE_CACHE_LRU.move_to_end(process_key)
            _PROCESS_PLANE_CACHE_BYTES += nbytes
            while (
                _PROCESS_PLANE_CACHE_BYTES > _PLANE_CACHE_MAX_BYTES
                or len(_PROCESS_PLANE_CACHE_LRU) > _PLANE_CACHE_MAX_ENTRIES
            ):
                evicted_key, (owner_ref, evicted_nbytes) = _PROCESS_PLANE_CACHE_LRU.popitem(
                    last=False
                )
                _PROCESS_PLANE_CACHE_BYTES -= evicted_nbytes
                owner = owner_ref()
                if owner is None:
                    continue
                owner_entry = owner._plane_cache.pop(evicted_key[1], None)
                if owner_entry is not None:
                    owner._plane_cache_bytes -= int(owner_entry[0].nbytes)

    def read_region(
        self,
        *,
        level: int,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        t: int = 0,
        c: int = 0,
        z: int = 0,
    ) -> np.ndarray:
        """Read a bounded 2D (Y, X) region (a tile) — only the covering chunks."""
        level = self._validated_level(level)
        h, w = self.level_yx(level)
        bounds = (("y0", y0), ("y1", y1), ("x0", x0), ("x1", x1))
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            for _, value in bounds
        ):
            raise NgffError("region bounds must be integers")
        y0, y1, x0, x1 = int(y0), int(y1), int(x0), int(x1)
        if not (0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w):
            raise NgffError(f"region [{y0}:{y1}, {x0}:{x1}] is outside level geometry ({h}, {w})")
        idx = self._index_tuple(level, t=t, c=c, z=z, y=slice(y0, y1), x=slice(x0, x1))
        self._validate_local_chunk_files(
            level=level,
            t=int(t),
            c=int(c),
            z=int(z),
            y_range=(y0, y1),
            x_range=(x0, x1),
        )
        return _as_2d_yx(
            np.asarray(self.levels[level].array[idx]),
            stored_spatial_order=self._spatial_axis_order(),
        )

    def level_yx(self, level: int) -> tuple[int, int]:
        """(height, width) of a multiscale level."""
        level = self._validated_level(level)
        shp = self.levels[level].shape
        yp, xp = self._axis_pos("y"), self._axis_pos("x")
        if yp is None or xp is None:
            raise NgffError("OME-NGFF viewer requires explicit 'y' and 'x' axes")
        return int(shp[yp]), int(shp[xp])

    def read_display_sample(self, *, c: int = 0) -> np.ndarray:
        """Read a bounded, deterministic sample for global display-window estimation.

        Nine distributed patches come from the smallest multiscale level. A patch is at
        most 128x128 (configurable) and no more than one quarter of each non-degenerate
        spatial dimension, so first-tile rendering never performs a whole-plane read just
        to choose contrast. The returned vector is a rendering heuristic, never reported
        as an exact scientific intensity range.
        """

        level = len(self.levels) - 1
        h, w = self.level_yx(level)
        patch_edge = max(1, _DISPLAY_SAMPLE_PATCH_EDGE)
        patch_h = min(patch_edge, max(1, (h + 3) // 4))
        patch_w = min(patch_edge, max(1, (w + 3) // 4))
        y_starts = tuple(dict.fromkeys((0, (h - patch_h) // 2, h - patch_h)))
        x_starts = tuple(dict.fromkeys((0, (w - patch_w) // 2, w - patch_w)))
        patches = [
            self.read_region(
                level=level,
                y0=y0,
                y1=y0 + patch_h,
                x0=x0,
                x1=x0 + patch_w,
                c=c,
            ).reshape(-1)
            for y0 in y_starts
            for x0 in x_starts
        ]
        return np.concatenate(patches) if len(patches) > 1 else patches[0]

    def thumbnail_level(self, max_size: int) -> int:
        """Smallest level whose long edge is still >= max_size (bounded, fast)."""
        chosen = len(self.levels) - 1
        for i in range(len(self.levels)):
            h, w = self.level_yx(i)
            if max(h, w) <= max_size:
                return i
            chosen = i
        return chosen

    @property
    def intensity_range_status(self) -> str:
        """Why :meth:`intensity_range` is exact or unavailable."""

        return self._intensity_range_status

    def intensity_range(self) -> tuple[float, float] | None:
        """Exact finite min/max over the complete smallest multiscale array.

        The method never samples.  If the full smallest level exceeds either configured
        scan cap, or cannot be read, it returns ``None`` and records an explicit status.
        This range describes the stored smallest level, not unscanned full-resolution data.
        """
        if self._intensity_range_evaluated:
            return self._intensity_range
        self._intensity_range_evaluated = True
        try:
            small = len(self.levels) - 1
            level = self.levels[small]
            elements = math.prod(level.shape)
            decoded_bytes = elements * int(self.dtype.itemsize)
            if (
                elements > _INTENSITY_RANGE_MAX_ELEMENTS
                or decoded_bytes > _INTENSITY_RANGE_MAX_BYTES
            ):
                self._intensity_range_status = "unknown_budget_exceeded"
                return None
            flat = np.asarray(level.array[:]).reshape(-1)
            if np.issubdtype(flat.dtype, np.floating):
                flat = flat[np.isfinite(flat)]
            if flat.size == 0:
                self._intensity_range_status = "unknown_no_finite_values"
                return None
            lo, hi = float(flat.min()), float(flat.max())
        except Exception:  # noqa: BLE001 - metadata is best-effort, never fatal
            self._intensity_range_status = "unknown_read_error"
            return None
        self._intensity_range = (lo, hi)
        self._intensity_range_status = "exact_smallest_level"
        return self._intensity_range


def _as_2d_yx(plane: np.ndarray, *, stored_spatial_order: tuple[str, str]) -> np.ndarray:
    """Return a spatial read in canonical ``(Y, X)`` order.

    Scalar indexing drops every non-spatial dimension, while slices retain both spatial
    dimensions even when either has length one.  Rejecting any other rank avoids silently
    reshaping a malformed or unexpectedly indexed scientific array.
    """

    arr = np.asarray(plane)
    if arr.ndim != 2:
        raise NgffError(f"spatial read returned rank {arr.ndim}; expected exactly two dimensions")
    if stored_spatial_order == ("y", "x"):
        return cast(np.ndarray, arr)
    if stored_spatial_order == ("x", "y"):
        return cast(np.ndarray, arr.T)
    raise NgffError(f"unsupported stored spatial-axis order {stored_spatial_order!r}")


# --------------------------------------------------------------------------- #
# Opening + metadata parsing
# --------------------------------------------------------------------------- #


def _json_depth_within_limit(text: str) -> bool:
    """Check JSON bracket depth without recursing and without counting quoted brackets."""

    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > _MAX_METADATA_JSON_DEPTH:
                return False
        elif character in "]}":
            depth -= 1
    return True


def _load_bounded_json(path: str) -> Any:
    """Load one local metadata document through byte, nesting, and no-follow guards."""

    try:
        before = os.lstat(path)
    except OSError as exc:
        raise NgffError(
            f"cannot inspect OME-NGFF metadata {os.path.basename(path)!r}: {exc}"
        ) from exc
    if stat.S_ISLNK(before.st_mode):
        raise NgffError(
            f"refusing OME-NGFF metadata {os.path.basename(path)!r} that is a symbolic link"
        )
    if not stat.S_ISREG(before.st_mode):
        raise NgffError(f"OME-NGFF metadata {os.path.basename(path)!r} is not a regular file")
    if before.st_size > _MAX_METADATA_JSON_BYTES:
        raise NgffError(
            f"OME-NGFF metadata {os.path.basename(path)!r} exceeds the "
            f"{_MAX_METADATA_JSON_BYTES}-byte metadata byte budget"
        )

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise NgffError(f"cannot open OME-NGFF metadata {os.path.basename(path)!r}: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
        ):
            raise NgffError(
                f"OME-NGFF metadata {os.path.basename(path)!r} changed during validation"
            )
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read(_MAX_METADATA_JSON_BYTES + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(payload) > _MAX_METADATA_JSON_BYTES:
        raise NgffError(
            f"OME-NGFF metadata {os.path.basename(path)!r} exceeds the "
            f"{_MAX_METADATA_JSON_BYTES}-byte metadata byte budget"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NgffError(f"OME-NGFF metadata {os.path.basename(path)!r} is not valid UTF-8") from exc
    if not _json_depth_within_limit(text):
        raise NgffError(
            f"OME-NGFF metadata nesting exceeds the depth limit of {_MAX_METADATA_JSON_DEPTH}"
        )
    try:
        return json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise NgffError(
            f"invalid OME-NGFF metadata JSON in {os.path.basename(path)!r}: {exc}"
        ) from exc


def _read_attrs(path: str) -> dict[str, Any]:
    """Read the group attributes (.zattrs for v2, attributes in zarr.json for v3)
    directly from disk — robust to zarr version quirks."""
    zgroup = os.path.join(path, ".zgroup")
    if os.path.lexists(zgroup):
        marker = _load_bounded_json(zgroup)
        if not isinstance(marker, dict):
            raise NgffError("Zarr v2 group metadata must be a JSON object")
    zattrs = os.path.join(path, ".zattrs")
    if os.path.lexists(zattrs):
        loaded = _load_bounded_json(zattrs)
        if not isinstance(loaded, dict):
            raise NgffError("OME-NGFF group attributes must be a JSON object")
        return loaded
    zjson = os.path.join(path, "zarr.json")  # zarr v3 group metadata
    if os.path.lexists(zjson):
        meta = _load_bounded_json(zjson)
        if not isinstance(meta, dict):
            raise NgffError("Zarr v3 group metadata must be a JSON object")
        raw_attrs = meta.get("attributes", {})
        if not isinstance(raw_attrs, dict):
            raise NgffError("Zarr v3 group attributes must be a JSON object")
        return dict(raw_attrs)
    return {}


def is_ome_zarr(path: str) -> bool:
    """A directory is an OME-Zarr group if it carries group metadata AND OME `multiscales`."""
    if not os.path.isdir(path):
        return False
    has_group = os.path.isfile(os.path.join(path, ".zgroup")) or os.path.isfile(
        os.path.join(path, "zarr.json")
    )
    if not has_group:
        return False
    attrs = _read_attrs(path)
    return bool(_multiscales(attrs))


def _multiscales(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    ms = attrs.get("multiscales")
    if isinstance(ms, list) and ms:
        return list(ms)
    # Some writers nest under "ome" (NGFF 0.5).
    ome = attrs.get("ome")
    if isinstance(ome, dict) and isinstance(ome.get("multiscales"), list):
        return list(ome["multiscales"])
    return []


def _ngff_metadata(attrs: dict[str, Any]) -> tuple[list[Any], str | None, bool]:
    """Return ``(multiscales, container_version, nested_ome)``.

    Version 0.4 stores ``multiscales`` directly in group attributes and puts the
    version on each multiscale entry (where it is only a SHOULD). Version 0.5
    stores both the mandatory version and ``multiscales`` below ``attributes.ome``.
    A document containing both locations is ambiguous and is rejected by the
    scientific reader instead of assigning precedence silently.
    """

    root = attrs.get("multiscales")
    ome = attrs.get("ome")
    nested = ome.get("multiscales") if isinstance(ome, dict) else None
    has_root = isinstance(root, list) and bool(root)
    has_nested = isinstance(nested, list) and bool(nested)
    if has_root and has_nested:
        raise NgffError(
            "ambiguous OME-NGFF metadata: multiscales is present both at the group "
            "root and under 'ome'"
        )
    if has_nested and isinstance(ome, dict) and isinstance(nested, list):
        raw_version = ome.get("version")
        if not isinstance(raw_version, str) or raw_version.strip() != "0.5":
            raise NgffError("OME-NGFF metadata under 'ome' requires version '0.5'")
        return list(nested), "0.5", True
    if has_root and isinstance(root, list):
        return list(root), None, False
    return [], None, False


def _select_multiscale(
    multiscales: list[Any], selector: int | str | None
) -> tuple[dict[str, Any], int]:
    """Select one image without ever treating list position zero as implicit policy."""

    if len(multiscales) > _MAX_MULTISCALES:
        raise NgffError(
            f"OME-NGFF declares {len(multiscales)} multiscales; limit is {_MAX_MULTISCALES}"
        )
    if not multiscales:
        raise NgffError("OME-NGFF multiscales is empty")
    if any(not isinstance(entry, dict) for entry in multiscales):
        raise NgffError("every OME-NGFF multiscales entry must be an object")
    for index, entry in enumerate(multiscales):
        name = entry.get("name")
        if name is not None and (
            not isinstance(name, str) or not name or len(name) > _MAX_MULTISCALE_NAME_LENGTH
        ):
            raise NgffError(
                f"multiscales[{index}].name must contain 1..{_MAX_MULTISCALE_NAME_LENGTH} characters"
            )

    if selector is None:
        if len(multiscales) != 1:
            names = [
                str(entry.get("name")) if entry.get("name") is not None else "<unnamed>"
                for entry in multiscales
            ]
            raise NgffError(
                "OME-NGFF contains multiple multiscale images; choose one explicitly "
                f"by zero-based index or exact name (available: {names!r})"
            )
        return multiscales[0], 0

    if isinstance(selector, bool):
        raise NgffError("multiscale selector must be a zero-based integer index or exact name")
    if isinstance(selector, int):
        if selector < 0 or selector >= len(multiscales):
            raise NgffError(f"multiscale index {selector} is outside [0, {len(multiscales) - 1}]")
        return multiscales[selector], selector
    if not isinstance(selector, str):
        raise NgffError("multiscale selector must be a zero-based integer index or exact name")
    if not selector or len(selector) > _MAX_MULTISCALE_NAME_LENGTH:
        raise NgffError(f"multiscale name must contain 1..{_MAX_MULTISCALE_NAME_LENGTH} characters")
    matches = [index for index, entry in enumerate(multiscales) if entry.get("name") == selector]
    if not matches:
        raise NgffError(f"no OME-NGFF multiscale image is named {selector!r}")
    if len(matches) != 1:
        raise NgffError(f"OME-NGFF multiscale name {selector!r} is not unique")
    index = matches[0]
    return multiscales[index], index


def _selected_version(ms: dict[str, Any], container_version: str | None) -> str:
    """Resolve the selected image's supported metadata version."""

    entry_version = ms.get("version")
    if entry_version is not None and not isinstance(entry_version, str):
        raise NgffError("selected multiscale version must be a string")
    if container_version is not None:
        if entry_version is not None and entry_version.strip() != container_version:
            raise NgffError(
                "selected multiscale version conflicts with the OME-NGFF container version"
            )
        version = container_version
    else:
        # v0.4 says the per-entry version SHOULD be present, not MUST. Root-level
        # metadata with the complete v0.4 axes/transform contract is therefore
        # accepted as 0.4 when this optional field is absent.
        version = "0.4" if entry_version is None else entry_version.strip()
        if version == "0.5":
            raise NgffError("OME-NGFF 0.5 metadata must be nested under 'ome'")
    if version not in _SUPPORTED_NGFF_VERSIONS:
        raise NgffError(
            f"unsupported OME-NGFF multiscales version {version!r}; supported versions are 0.4 and 0.5"
        )
    return version


def _validated_dataset_path(value: Any, *, context: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not value and not allow_empty):
        raise NgffError(f"{context}.path must be a non-empty relative Zarr path")
    if len(value) > _MAX_ZARR_PATH_LENGTH:
        raise NgffError(f"{context}.path exceeds the {_MAX_ZARR_PATH_LENGTH}-character limit")
    if value.startswith(("/", "\\")) or "\\" in value:
        raise NgffError(f"{context}.path must use a relative forward-slash Zarr path")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts) and not (allow_empty and value == ""):
        raise NgffError(f"{context}.path contains an unsafe or empty path component")
    return value


def _validated_axes(
    ms: dict[str, Any], ndim: int, *, version: str
) -> tuple[list[str], tuple[str, ...], tuple[str | None, ...]]:
    raw_axes = ms.get("axes")
    if not isinstance(raw_axes, list) or len(raw_axes) != ndim:
        raise NgffError(f"OME-NGFF axes length must equal array dimensionality {ndim}")
    if ndim < 2 or ndim > 5:
        raise NgffError("OME-NGFF image arrays must have between 2 and 5 dimensions")

    names: list[str] = []
    types: list[str | None] = []
    units: list[str] = []
    canonical_types = {"t": "time", "c": "channel", "z": "space", "y": "space", "x": "space"}
    for index, raw_axis in enumerate(raw_axes):
        if not isinstance(raw_axis, dict):
            raise NgffError(f"OME-NGFF axes[{index}] must be an object")
        raw_name = raw_axis.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise NgffError(f"OME-NGFF axes[{index}].name must be a non-empty string")
        name = raw_name.strip().lower()
        if name in names:
            raise NgffError(f"OME-NGFF axis name {name!r} is duplicated")
        raw_type = raw_axis.get("type")
        if raw_type is not None and (not isinstance(raw_type, str) or not raw_type.strip()):
            raise NgffError(f"OME-NGFF axes[{index}].type must be null or a non-empty string")
        axis_type = raw_type.strip() if isinstance(raw_type, str) else None
        expected_type = canonical_types.get(name)
        if expected_type is not None and (
            (version == "0.5" and axis_type is None)
            or (axis_type is not None and axis_type != expected_type)
        ):
            raise NgffError(
                f"OME-NGFF canonical axis {name!r} requires type {expected_type!r}, "
                f"not {axis_type!r}"
            )
        raw_unit = raw_axis.get("unit")
        if raw_unit is not None and (not isinstance(raw_unit, str) or not raw_unit.strip()):
            raise NgffError(f"OME-NGFF axes[{index}].unit must be a non-empty string")
        names.append(name)
        # Canonical axis names carry an unambiguous effective type even when the
        # older v0.4 metadata omitted the recommended type field.
        types.append(axis_type if axis_type is not None else expected_type)
        units.append(raw_unit.strip() if isinstance(raw_unit, str) else "")

    if version == "0.5":
        space_count = types.count("space")
        time_count = types.count("time")
        middle_count = len(types) - space_count - time_count
        if space_count not in (2, 3):
            raise NgffError("OME-NGFF 0.5 axes must contain exactly two or three space axes")
        if time_count > 1:
            raise NgffError("OME-NGFF 0.5 axes may contain at most one time axis")
        if middle_count > 1:
            raise NgffError(
                "OME-NGFF 0.5 axes may contain at most one channel, null, or custom axis"
            )
        order = [
            0 if axis_type == "time" else 2 if axis_type == "space" else 1 for axis_type in types
        ]
        if order != sorted(order):
            raise NgffError(
                "OME-NGFF 0.5 axes must order time first, then channel/custom, then space"
            )
    return names, tuple(units), tuple(types)


def _validate_v05_dimension_names(
    ms: dict[str, Any], opened: list[tuple[str, Any, tuple[int, ...], dict[str, Any]]]
) -> None:
    """Enforce the OME-NGFF 0.5 array-dimension binding.

    Version 0.5 requires every Zarr v3 image array's ``dimension_names`` to equal,
    position for position, the selected multiscale's ``axes[].name`` values.  Without
    that binding, interpreting dimensions from group metadata alone can transpose or
    otherwise mislabel scientific coordinates.
    """

    raw_axes = cast(list[dict[str, Any]], ms["axes"])
    expected = tuple(cast(str, axis["name"]) for axis in raw_axes)
    for dataset_index, (path, array, _, _) in enumerate(opened):
        metadata = getattr(array, "metadata", None)
        dimension_names = getattr(metadata, "dimension_names", None)
        if dimension_names is None:
            # Retain compatibility with a future zarr-python public property while
            # still failing closed for v2 arrays or v3 arrays that omit the field.
            dimension_names = getattr(array, "dimension_names", None)
        if dimension_names is None:
            raise NgffError(
                f"datasets[{dataset_index}] array {path!r} is OME-NGFF 0.5 but has no "
                "Zarr dimension_names"
            )
        actual = tuple(dimension_names)
        if actual != expected:
            raise NgffError(
                f"datasets[{dataset_index}] array {path!r} dimension_names {actual!r} "
                f"do not match axes names {expected!r}"
            )


def _validate_viewer_axis_geometry(
    axes: list[str], opened: list[tuple[str, Any, tuple[int, ...], dict[str, Any]]]
) -> None:
    """Reject dimensions that the 2-D viewer cannot select without data loss.

    Canonical ``t/c/z`` axes have explicit request indices and ``y/x`` are returned. A
    custom axis is safe to remove only when it is singleton at every level. Selecting index
    zero of a non-singleton custom axis would silently substitute one scientific condition
    for an unrepresented selector, so such stores fail closed.
    """

    if axes.count("y") != 1 or axes.count("x") != 1:
        raise NgffError("OME-NGFF viewer requires exactly one explicit 'y' and 'x' axis")
    for axis_position, axis in enumerate(axes):
        if axis in _CANONICAL:
            continue
        non_singleton = [
            (dataset_index, path, shape[axis_position])
            for dataset_index, (path, _, shape, _) in enumerate(opened)
            if shape[axis_position] != 1
        ]
        if non_singleton:
            raise NgffError(
                f"unsupported non-singleton axis {axis!r}; no request selector exists "
                f"for level dimensions {non_singleton!r}"
            )


def _validate_resolution_shape_order(
    axes: list[str],
    axis_types: tuple[str | None, ...],
    opened: list[tuple[str, Any, tuple[int, ...], dict[str, Any]]],
) -> None:
    """Require spatial nonincrease and exact non-spatial pyramid invariants."""

    for dataset_index in range(1, len(opened)):
        previous_path, _, previous_shape, _ = opened[dataset_index - 1]
        current_path, _, current_shape, _ = opened[dataset_index]
        nonspatial_changes = [
            f"{axis}:{previous_shape[position]}->{current_shape[position]}"
            for position, (axis, axis_type) in enumerate(zip(axes, axis_types, strict=True))
            if axis_type != "space" and current_shape[position] != previous_shape[position]
        ]
        if nonspatial_changes:
            raise NgffError(
                f"datasets[{dataset_index}] {current_path!r} changes non-spatial "
                f"dimensions relative to {previous_path!r}: {nonspatial_changes!r}"
            )
        increases = [
            f"{axis}:{previous_shape[position]}->{current_shape[position]}"
            for position, (axis, axis_type) in enumerate(zip(axes, axis_types, strict=True))
            if axis_type == "space" and current_shape[position] > previous_shape[position]
        ]
        if increases:
            raise NgffError(
                f"datasets[{dataset_index}] {current_path!r} increases resolution dimensions "
                f"relative to {previous_path!r}: {increases!r}"
            )


def _validate_spatial_scale_order(axes: list[str], levels: list[NgffLevel]) -> None:
    """Require later pyramid levels not to claim finer spatial sampling."""

    spatial_positions = [index for index, axis in enumerate(axes) if axis in ("z", "y", "x")]
    for level_index in range(1, len(levels)):
        previous = levels[level_index - 1]
        current = levels[level_index]
        decreases = [
            f"{axes[position]}:{previous.scale[position]:g}->{current.scale[position]:g}"
            for position in spatial_positions
            if current.scale[position] < previous.scale[position]
        ]
        if decreases:
            raise NgffError(
                f"datasets[{level_index}] spatial scale decreases relative to the prior "
                f"resolution level: {decreases!r}"
            )


def _transform_vector(
    transform: dict[str, Any],
    *,
    kind: str,
    ndim: int,
    group: Any,
    context: str,
) -> tuple[float, ...]:
    """Read one inline or path-backed scale/translation vector, bounded to ``ndim``."""

    has_inline = kind in transform
    has_path = "path" in transform
    other_kind = "translation" if kind == "scale" else "scale"
    if other_kind in transform:
        raise NgffError(f"{context} contains a conflicting {other_kind!r} parameter")
    if has_inline == has_path:
        raise NgffError(f"{context} must contain exactly one of {kind!r} or 'path'")
    if has_inline:
        raw_vector = transform[kind]
        if not isinstance(raw_vector, list):
            raise NgffError(f"{context}.{kind} must be a JSON array")
    else:
        parameter_path = _validated_dataset_path(transform.get("path"), context=context)
        try:
            parameter_array = group[parameter_path]
        except Exception as exc:  # noqa: BLE001
            raise NgffError(
                f"{context}.path does not resolve to a transformation parameter array"
            ) from exc
        shape = tuple(int(size) for size in getattr(parameter_array, "shape", ()))
        if shape != (ndim,):
            raise NgffError(f"{context}.path array shape {shape!r} must be ({ndim},)")
        try:
            raw_vector = np.asarray(parameter_array[:]).tolist()
        except Exception as exc:  # noqa: BLE001
            raise NgffError(f"could not read {context}.path parameter array") from exc

    if len(raw_vector) != ndim:
        raise NgffError(
            f"{context} vector length {len(raw_vector)} does not match dimensionality {ndim}"
        )
    values: list[float] = []
    for index, raw_value in enumerate(raw_vector):
        if isinstance(raw_value, (bool, np.bool_)) or not isinstance(raw_value, Real):
            raise NgffError(f"{context}[{index}] must be a finite JSON number")
        value = float(raw_value)
        if not math.isfinite(value):
            raise NgffError(f"{context}[{index}] must be finite")
        if kind == "scale" and value <= 0.0:
            raise NgffError(f"{context}[{index}] scale must be positive")
        values.append(value)
    return tuple(values)


def _validated_transform_sequence(
    raw: Any, *, ndim: int, group: Any, context: str
) -> list[tuple[str, tuple[float, ...]]]:
    """Validate the v0.4/v0.5 multiscale transform subset.

    Both specification versions require exactly one scale and permit one
    translation after it in dataset and multiscale transformation lists.
    """

    if not isinstance(raw, list):
        raise NgffError(f"{context} must be a list")
    if len(raw) not in (1, 2):
        raise NgffError(
            f"{context} must contain exactly one scale and optionally one following translation"
        )
    kinds: list[str] = []
    for index, transform in enumerate(raw):
        if not isinstance(transform, dict):
            raise NgffError(f"{context}[{index}] must be an object")
        kind = transform.get("type")
        if kind not in ("scale", "translation"):
            raise NgffError(
                f"unsupported {context}[{index}] transformation type {kind!r}; "
                "only scale and translation are supported"
            )
        kinds.append(kind)
    if kinds.count("scale") != 1:
        raise NgffError(f"{context} must contain exactly one scale transformation")
    if kinds.count("translation") > 1:
        raise NgffError(f"{context} may contain at most one translation transformation")
    if kinds not in (["scale"], ["scale", "translation"]):
        raise NgffError(f"{context} must apply scale first and optional translation second")
    return [
        (
            kind,
            _transform_vector(
                raw[index],
                kind=kind,
                ndim=ndim,
                group=group,
                context=f"{context}[{index}]",
            ),
        )
        for index, kind in enumerate(kinds)
    ]


def _compose_transform_sequences(
    ndim: int, *sequences: list[tuple[str, tuple[float, ...]]]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Compose diagonal affine transforms in their declared application order."""

    scale = np.ones(ndim, dtype=np.float64)
    translation = np.zeros(ndim, dtype=np.float64)
    for sequence in sequences:
        for kind, vector in sequence:
            values = np.asarray(vector, dtype=np.float64)
            with np.errstate(over="ignore", invalid="ignore"):
                if kind == "scale":
                    scale *= values
                    translation *= values
                else:
                    translation += values
            if not np.all(np.isfinite(scale)) or not np.all(np.isfinite(translation)):
                raise NgffError("composed OME-NGFF coordinate transformation is non-finite")
    return tuple(float(value) for value in scale), tuple(float(value) for value in translation)


def open_ngff(path: str, *, multiscale: int | str | None = None) -> NgffImage:
    """Open one OME-Zarr image and parse its NGFF metadata.

    ``multiscale`` is either a bounded zero-based list index or an exact
    ``multiscales[].name``. It is optional only when exactly one multiscale image
    exists; ambiguous stores fail closed. Pixel arrays remain lazy. At most five
    values are read eagerly only when a valid transformation uses a path-backed
    scale or translation vector.
    """
    try:
        import zarr
    except Exception as exc:  # noqa: BLE001
        raise NgffError(f"zarr is not installed: {exc!r}") from exc

    if not os.path.isdir(path):
        raise NgffError(f"not a directory: {path!r}")
    attrs = _read_attrs(path)
    multiscales, container_version, nested_ome = _ngff_metadata(attrs)
    if not multiscales:
        raise NgffError(
            f"no OME-NGFF 'multiscales' metadata in {path!r}; generic Zarr axis "
            "semantics are not guessed by the OME-NGFF viewer"
        )

    try:
        group = zarr.open_group(path, mode="r")
    except Exception:  # noqa: BLE001 - fall back to generic open
        try:
            group = zarr.open(path, mode="r")
        except Exception as exc:  # noqa: BLE001
            raise NgffError(f"zarr could not open {path!r}: {exc!r}") from exc

    ms, selected_index = _select_multiscale(multiscales, multiscale)
    version = _selected_version(ms, container_version)
    datasets = ms.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise NgffError(f"OME-Zarr multiscale has no datasets in {path!r}")
    if len(datasets) > _MAX_LEVELS:
        raise NgffError(
            f"OME-NGFF declares {len(datasets)} resolution levels; limit is {_MAX_LEVELS}"
        )

    opened: list[tuple[str, Any, tuple[int, ...], dict[str, Any]]] = []
    ndim: int | None = None
    dtype: np.dtype | None = None
    seen_paths: set[str] = set()
    for dataset_index, ds in enumerate(datasets):
        if not isinstance(ds, dict):
            raise NgffError(f"OME-NGFF datasets[{dataset_index}] must be an object")
        dpath = _validated_dataset_path(
            ds.get("path"),
            context=f"datasets[{dataset_index}]",
        )
        if dpath in seen_paths:
            raise NgffError(f"OME-NGFF dataset path {dpath!r} is duplicated")
        seen_paths.add(dpath)
        try:
            arr = group if dpath == "" and hasattr(group, "shape") else group[dpath]
        except Exception as exc:  # noqa: BLE001
            raise NgffError(f"missing multiscale level {dpath!r} in {path!r}: {exc!r}") from exc
        try:
            shape = tuple(int(s) for s in arr.shape)
        except Exception as exc:  # noqa: BLE001
            raise NgffError(f"multiscale level {dpath!r} is not a readable Zarr array") from exc
        if any(size <= 0 for size in shape):
            raise NgffError(f"multiscale level {dpath!r} has a non-positive dimension")
        if ndim is None:
            ndim = len(shape)
        elif len(shape) != ndim:
            raise NgffError(
                f"multiscale level {dpath!r} dimensionality {len(shape)} does not match {ndim}"
            )
        current_dtype = np.dtype(arr.dtype)
        if current_dtype.kind not in _RENDERABLE_DTYPE_KINDS:
            raise NgffError(
                f"unsupported pixel dtype {current_dtype!r}: the OME-Zarr viewer renders "
                "only real scalar bool, integer, and floating-point arrays"
            )
        chunk_shape = getattr(arr, "chunks", None)
        if (
            isinstance(chunk_shape, tuple)
            and len(chunk_shape) == len(shape)
            and all(isinstance(chunk_size, int) and chunk_size > 0 for chunk_size in chunk_shape)
        ):
            decoded_chunk_bytes = math.prod(chunk_shape) * int(current_dtype.itemsize)
            if decoded_chunk_bytes > _MAX_CHUNK_DECODED_BYTES:
                raise NgffError(
                    f"multiscale level {dpath!r} declares {decoded_chunk_bytes} decoded bytes "
                    f"per chunk (chunks={chunk_shape}, dtype={current_dtype}), exceeding the "
                    f"{_MAX_CHUNK_DECODED_BYTES}-byte decoded chunk budget"
                )
        if dtype is None:
            dtype = current_dtype
        elif current_dtype != dtype:
            raise NgffError(
                f"multiscale level {dpath!r} dtype {current_dtype} does not match "
                f"base dtype {dtype}"
            )
        opened.append((dpath, arr, shape, ds))

    if ndim is None:
        raise NgffError("OME-NGFF multiscale contains no readable arrays")

    axes, axis_units, axis_types = _validated_axes(ms, ndim, version=version)
    if version == "0.5":
        _validate_v05_dimension_names(ms, opened)
    if "coordinateTransformations" in ms:
        global_sequence = _validated_transform_sequence(
            ms["coordinateTransformations"],
            ndim=ndim,
            group=group,
            context="multiscales.coordinateTransformations",
        )
    else:
        global_sequence = []

    _validate_resolution_shape_order(axes, axis_types, opened)
    _validate_viewer_axis_geometry(axes, opened)

    levels: list[NgffLevel] = []
    for dataset_index, (dpath, arr, shape, ds) in enumerate(opened):
        if "coordinateTransformations" not in ds:
            raise NgffError(f"datasets[{dataset_index}].coordinateTransformations is required")
        dataset_sequence = _validated_transform_sequence(
            ds["coordinateTransformations"],
            ndim=ndim,
            group=group,
            context=f"datasets[{dataset_index}].coordinateTransformations",
        )
        scale, level_translation = _compose_transform_sequences(
            ndim, dataset_sequence, global_sequence
        )
        levels.append(
            NgffLevel(
                path=dpath,
                array=arr,
                shape=shape,
                scale=scale,
                translation=level_translation,
                axis_units=axis_units,
            )
        )

    _validate_spatial_scale_order(axes, levels)

    # Canonical sizes from level 0 (full res). physical = scale per axis (t = time
    # increment); units = the NGFF axis unit (x/y "micrometer", t "hour", …).
    base = levels[0]
    sizes = {ax: 1 for ax in _CANONICAL}
    physical = {ax: 0.0 for ax in _CANONICAL}
    translation = {ax: 0.0 for ax in _CANONICAL}
    units = {ax: "" for ax in _CANONICAL}
    for canonical in _CANONICAL:
        if canonical in axes:
            pos = axes.index(canonical)
            sizes[canonical] = int(base.shape[pos])
            physical[canonical] = float(base.scale[pos])
            translation[canonical] = float(base.translation[pos])
            units[canonical] = axis_units[pos]

    ome = attrs.get("ome")
    if nested_ome:
        omero = ome.get("omero") if isinstance(ome, dict) else None
    else:
        omero = attrs.get("omero")
    channels = _parse_channels(omero, sizes["c"])

    name = ms.get("name")
    if not name and isinstance(omero, dict):
        name = omero.get("name")
    return NgffImage(
        path=path,
        axes=axes,
        levels=levels,
        dtype=dtype or np.dtype("uint16"),
        channels=channels,
        sizes=sizes,
        physical=physical,
        translation=translation,
        units=units,
        omero=omero if isinstance(omero, dict) else None,
        multiscale_name=str(ms.get("name")) if ms.get("name") else None,
        multiscale_index=selected_index,
        name=str(name) if name else None,
        version=version,
    )


def _parse_channels(omero: Any, num_c: int) -> list[NgffChannel]:
    """Build per-channel render hints from the OME `omero.channels` block; synthesize
    sensible defaults when absent so the renderer always has something to work with."""
    out: list[NgffChannel] = []
    if omero is not None and not isinstance(omero, dict):
        raise NgffError("OME-NGFF omero metadata must be an object")
    omero_channels = omero.get("channels") if isinstance(omero, dict) else None
    if isinstance(omero, dict):
        if not isinstance(omero_channels, list):
            raise NgffError("OME-NGFF omero.channels must be an array when omero is present")
        if len(omero_channels) != num_c:
            raise NgffError(
                f"OME-NGFF omero.channels has {len(omero_channels)} entries but the "
                f"channel dimension has size {num_c}"
            )
    if isinstance(omero_channels, list):
        for i, ch in enumerate(omero_channels):
            if not isinstance(ch, dict):
                raise NgffError(f"OME-NGFF omero.channels[{i}] must be an object")
            raw_window = ch.get("window")
            window = cast(dict[str, Any], raw_window) if isinstance(raw_window, dict) else {}
            out.append(
                NgffChannel(
                    label=str(ch.get("label") or f"Channel {i}"),
                    color=_normalize_hex(ch.get("color"), i),
                    window_start=_maybe_float(window.get("start")),
                    window_end=_maybe_float(window.get("end")),
                    active=bool(ch.get("active", True)),
                )
            )
        return out
    # No omero metadata: default channels (grayscale for single, RGB-ish for few).
    defaults = (
        ["FFFFFF"] if num_c <= 1 else ["FF0000", "00FF00", "0000FF", "FFFF00", "FF00FF", "00FFFF"]
    )
    for i in range(max(1, num_c)):
        out.append(
            NgffChannel(
                label=f"Channel {i}" if num_c > 1 else "Channel",
                color=defaults[i % len(defaults)],
                window_start=None,
                window_end=None,
                active=True,
            )
        )
    return out


def _normalize_hex(value: Any, index: int) -> str:
    fallbacks = ["FFFFFF", "FF0000", "00FF00", "0000FF", "FFFF00", "FF00FF", "00FFFF"]
    if isinstance(value, str):
        v = value.strip().lstrip("#")
        if len(v) in (6, 8):
            return v[:6].upper()
    return fallbacks[index % len(fallbacks)]


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
