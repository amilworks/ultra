"""OME-NGFF (OME-Zarr) reader — lazy, chunk-level, spec-correct.

Parses the OME-NGFF metadata (`multiscales`, `axes`, `omero`) from the store's
attributes and exposes lazy access to each multiscale level via zarr-python. Reads
are always bounded to the requested plane/region, so the whole (possibly 1 TB) array
is never materialized.

Supports Zarr v2 and v3 stores (zarr-python ≥3 reads both). Axis handling is driven
by the NGFF `axes[].name` (the spec's canonical t/c/z/y/x), tolerating arbitrary axis
order and missing axes (a 2D YX store, a TYX time-lapse, a multichannel ZYX volume).
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["NgffError", "NgffLevel", "NgffChannel", "NgffImage", "open_ngff", "is_ome_zarr"]

# Canonical dimension order Ultra's viewer-info uses.
_CANONICAL = ("t", "c", "z", "y", "x")

# Decoded-plane cache (the application-level chunk cache, layered above the OS page cache
# and below the Go edge tile LRU). A full 2D plane is cached per (level,t,c,z) ONLY when it
# is below a byte threshold, so scrubbing time/z and re-rendering channels reuse the decode
# (skip NFS + blosc) while a gigapixel level-0 plane is NEVER materialized whole (tiles read
# bounded regions instead). zarr v3 has no LRUStoreCache, so this is version-independent.
_PLANE_CACHE_MAX_BYTES = int(os.environ.get("ULTRA_NGFF_DECODE_CACHE_BYTES", str(256 * 1024 * 1024)))
_PLANE_CACHE_MAX_PLANE_BYTES = int(os.environ.get("ULTRA_NGFF_DECODE_CACHE_PLANE_BYTES", str(48 * 1024 * 1024)))


class NgffError(RuntimeError):
    """The store is not a readable OME-Zarr (missing/invalid NGFF metadata)."""


@dataclass
class NgffLevel:
    """One multiscale level: its dataset path, the lazy zarr array, and its per-axis
    scale (physical units per pixel, from the level's coordinateTransformations)."""

    path: str
    array: Any  # zarr.Array (lazy)
    shape: tuple[int, ...]
    scale: tuple[float, ...]


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
    physical: dict[str, float]  # units/pixel per canonical axis (best-effort; t = time increment)
    units: dict[str, str]  # NGFF axis unit per canonical axis (e.g. x/y "micrometer", t "hour")
    omero: dict[str, Any] | None
    multiscale_name: str | None
    name: str | None = None  # human image name (multiscales.name or omero.name)
    version: str | None = None  # NGFF version (e.g. "0.4" / "0.5")
    # Cached true data intensity range (min, max), computed lazily on the smallest level.
    _intensity_range: tuple[float, float] | None = None
    # Cache of the resolved global display window per channel index — computed once
    # (omero-if-valid, else auto-contrast on the smallest level) so every slice/tile
    # renders with the SAME mapping (no per-tile checkerboard). Populated by the renderer.
    window_cache: dict[int, tuple[float, float]] = field(default_factory=dict)
    # Size-bounded LRU of decoded full planes (see module docstring). Kept on the image so
    # it lives exactly as long as the per-(path,mtime) open-image cache entry.
    _plane_cache: "OrderedDict[tuple[int, int, int, int], np.ndarray]" = field(default_factory=OrderedDict)
    _plane_cache_bytes: int = 0

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
        """Build the zarr index tuple in the store's axis order. Scalar for t/c/z (a
        single plane), slices for y/x. Clamps t/c/z into range."""
        sel: dict[str, Any] = {}
        lvl_shape = self.levels[level].shape
        for canonical, value in (("t", t), ("c", c), ("z", z)):
            pos = self._axis_pos(canonical)
            if pos is not None:
                sel[canonical] = max(0, min(int(value), lvl_shape[pos] - 1))
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

    def read_plane(
        self, *, t: int = 0, c: int = 0, z: int = 0, level: int = 0
    ) -> np.ndarray:
        """Read one full 2D (Y, X) plane for the given t/c/z at a multiscale level.
        Only the chunks covering that plane are fetched. Sub-threshold planes are cached
        (size-aware LRU) so scrub/channel re-renders skip NFS + decompression."""
        level = max(0, min(level, len(self.levels) - 1))
        key = (level, int(t), int(c), int(z))
        cached = self._plane_cache.get(key)
        if cached is not None:
            self._plane_cache.move_to_end(key)
            return cached
        idx = self._index_tuple(level, t=t, c=c, z=z)
        plane = _as_2d_yx(np.asarray(self.levels[level].array[idx]))
        self._maybe_cache_plane(key, plane)
        return plane

    def _maybe_cache_plane(self, key: tuple[int, int, int, int], plane: np.ndarray) -> None:
        nbytes = int(plane.nbytes)
        # Never materialize a gigapixel plane into the cache — tiles read bounded regions.
        if nbytes > _PLANE_CACHE_MAX_PLANE_BYTES or nbytes > _PLANE_CACHE_MAX_BYTES:
            return
        self._plane_cache[key] = plane
        self._plane_cache.move_to_end(key)
        self._plane_cache_bytes += nbytes
        while self._plane_cache_bytes > _PLANE_CACHE_MAX_BYTES and self._plane_cache:
            _, evicted = self._plane_cache.popitem(last=False)
            self._plane_cache_bytes -= int(evicted.nbytes)

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
        level = max(0, min(level, len(self.levels) - 1))
        idx = self._index_tuple(level, t=t, c=c, z=z, y=slice(y0, y1), x=slice(x0, x1))
        return _as_2d_yx(np.asarray(self.levels[level].array[idx]))

    def level_yx(self, level: int) -> tuple[int, int]:
        """(height, width) of a multiscale level."""
        shp = self.levels[level].shape
        yp, xp = self._axis_pos("y"), self._axis_pos("x")
        return int(shp[yp]), int(shp[xp])

    def thumbnail_level(self, max_size: int) -> int:
        """Smallest level whose long edge is still >= max_size (bounded, fast)."""
        chosen = len(self.levels) - 1
        for i in range(len(self.levels)):
            h, w = self.level_yx(i)
            if max(h, w) <= max_size:
                return i
            chosen = i
        return chosen

    def intensity_range(self) -> tuple[float, float] | None:
        """True data (min, max) for the Metadata 'Value range', computed once on the
        SMALLEST multiscale level (cheap + representative; never reads full resolution).
        Returns None if it can't be computed (degenerate/empty)."""
        if self._intensity_range is not None:
            return self._intensity_range
        try:
            small = len(self.levels) - 1
            plane = self.read_plane(t=0, c=0, z=0, level=small)
            flat = plane.reshape(-1)
            if np.issubdtype(flat.dtype, np.floating):
                flat = flat[np.isfinite(flat)]
            if flat.size == 0:
                return None
            lo, hi = float(flat.min()), float(flat.max())
        except Exception:  # noqa: BLE001 - metadata is best-effort, never fatal
            return None
        if not (hi > lo):
            return None
        self._intensity_range = (lo, hi)
        return self._intensity_range


def _as_2d_yx(plane: np.ndarray) -> np.ndarray:
    """Squeeze a read result to 2D (Y, X). Indexing with scalars already drops t/c/z;
    this defends against a stray singleton axis."""
    arr = np.squeeze(plane)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    # More than 2 dims left (unexpected axis): collapse leading axes to the last 2D.
    return arr.reshape(arr.shape[-2], arr.shape[-1])


# --------------------------------------------------------------------------- #
# Opening + metadata parsing
# --------------------------------------------------------------------------- #

def _read_attrs(path: str) -> dict[str, Any]:
    """Read the group attributes (.zattrs for v2, attributes in zarr.json for v3)
    directly from disk — robust to zarr version quirks."""
    zattrs = os.path.join(path, ".zattrs")
    if os.path.isfile(zattrs):
        with open(zattrs, encoding="utf-8") as fh:
            return json.load(fh)
    zjson = os.path.join(path, "zarr.json")  # zarr v3 group metadata
    if os.path.isfile(zjson):
        with open(zjson, encoding="utf-8") as fh:
            meta = json.load(fh)
        return dict(meta.get("attributes", {}) or {})
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
        return ms
    # Some writers nest under "ome" (NGFF 0.5).
    ome = attrs.get("ome")
    if isinstance(ome, dict) and isinstance(ome.get("multiscales"), list):
        return ome["multiscales"]
    return []


def _fallback_single_scale_datasets(group: Any) -> list[dict[str, Any]]:
    """Best-effort single-level dataset list for a store missing `multiscales`.
    Prefers a child array named "0" (the OME base level), else a lone array child,
    else the root itself when it is a bare array. Returns [] when no single array
    can be unambiguously identified (e.g. a multi-array group) so the caller still
    raises rather than guessing."""
    def _is_array(obj: Any) -> bool:
        return hasattr(obj, "shape") and hasattr(obj, "dtype") and not hasattr(obj, "keys")

    if _is_array(group):
        return [{"path": ""}]
    try:
        keys = list(group.keys())
    except Exception:  # noqa: BLE001
        return []
    array_keys: list[str] = []
    for key in keys:
        try:
            child = group[key]
        except Exception:  # noqa: BLE001
            continue
        if _is_array(child):
            array_keys.append(str(key))
    if "0" in array_keys:
        return [{"path": "0"}]
    if len(array_keys) == 1:
        return [{"path": array_keys[0]}]
    return []


def _axis_names(ms: dict[str, Any], ndim: int) -> list[str]:
    """Lowercase axis names in stored order. Falls back to canonical trailing names
    (…t,c,z,y,x) when `axes` is absent (pre-0.3 NGFF)."""
    axes = ms.get("axes")
    if isinstance(axes, list) and axes:
        names: list[str] = []
        for ax in axes:
            if isinstance(ax, dict):
                names.append(str(ax.get("name", "")).lower())
            else:
                names.append(str(ax).lower())
        if all(names):
            return names
    fallback = list(_CANONICAL)[-ndim:]
    return fallback


def _axis_units(ms: dict[str, Any]) -> dict[str, str]:
    """Map canonical axis name -> NGFF unit string (e.g. {"x": "micrometer", "t": "hour"}),
    for the axes that declare one. Empty when `axes` is absent (pre-0.3 NGFF)."""
    out: dict[str, str] = {}
    axes = ms.get("axes")
    if isinstance(axes, list):
        for ax in axes:
            if isinstance(ax, dict):
                name = str(ax.get("name", "")).lower()
                unit = ax.get("unit")
                if name and isinstance(unit, str) and unit.strip():
                    out[name] = unit.strip()
    return out


def open_ngff(path: str) -> NgffImage:
    """Open an OME-Zarr store and parse its NGFF metadata. Raises NgffError if it isn't
    a readable OME-Zarr. Arrays are opened lazily (no pixel data read here)."""
    try:
        import zarr
    except Exception as exc:  # noqa: BLE001
        raise NgffError(f"zarr is not installed: {exc!r}") from exc

    if not os.path.isdir(path):
        raise NgffError(f"not a directory: {path!r}")
    attrs = _read_attrs(path)
    multiscales = _multiscales(attrs)

    try:
        group = zarr.open_group(path, mode="r")
    except Exception:  # noqa: BLE001 - fall back to generic open
        try:
            group = zarr.open(path, mode="r")
        except Exception as exc:  # noqa: BLE001
            raise NgffError(f"zarr could not open {path!r}: {exc!r}") from exc

    if multiscales:
        ms = multiscales[0]
        datasets = ms.get("datasets") or []
        if not datasets:
            raise NgffError(f"OME-Zarr multiscale has no datasets in {path!r}")
    else:
        # Fallback for a store that carries pixel arrays but no `multiscales`
        # metadata (e.g. a raw/single-scale conversion that never wrote the OME
        # attrs). This branch is only reached when the store would otherwise
        # 422, so it can never regress a spec-compliant store. Treat the
        # base-resolution array as a single level; axes fall back to the
        # canonical trailing names by ndim (see _axis_names).
        ms = {}
        datasets = _fallback_single_scale_datasets(group)
        if not datasets:
            raise NgffError(f"no OME-Zarr 'multiscales' metadata in {path!r}")

    levels: list[NgffLevel] = []
    ndim = 0
    dtype: np.dtype | None = None
    for ds in datasets:
        dpath = str(ds.get("path"))
        try:
            arr = group[dpath]
        except Exception as exc:  # noqa: BLE001
            raise NgffError(f"missing multiscale level {dpath!r} in {path!r}: {exc!r}") from exc
        shape = tuple(int(s) for s in arr.shape)
        ndim = len(shape)
        if dtype is None:
            dtype = np.dtype(arr.dtype)
        scale: tuple[float, ...] = tuple(1.0 for _ in shape)
        for ct in ds.get("coordinateTransformations") or []:
            if isinstance(ct, dict) and ct.get("type") == "scale" and isinstance(ct.get("scale"), list):
                vals = ct["scale"]
                if len(vals) == ndim:
                    scale = tuple(float(v) for v in vals)
        levels.append(NgffLevel(path=dpath, array=arr, shape=shape, scale=scale))

    axes = _axis_names(ms, ndim)
    if len(axes) != ndim:
        axes = list(_CANONICAL)[-ndim:]

    # Canonical sizes from level 0 (full res). physical = scale per axis (t = time
    # increment); units = the NGFF axis unit (x/y "micrometer", t "hour", …).
    base = levels[0]
    sizes = {ax: 1 for ax in _CANONICAL}
    physical = {ax: 0.0 for ax in _CANONICAL}
    raw_units = _axis_units(ms)
    units = {ax: "" for ax in _CANONICAL}
    for canonical in _CANONICAL:
        if canonical in axes:
            pos = axes.index(canonical)
            sizes[canonical] = int(base.shape[pos])
            physical[canonical] = float(base.scale[pos])
            units[canonical] = raw_units.get(canonical, "")

    omero = attrs.get("omero")
    if not isinstance(omero, dict):
        ome = attrs.get("ome")
        omero = ome.get("omero") if isinstance(ome, dict) else None
    channels = _parse_channels(omero, sizes["c"])

    name = ms.get("name")
    if not name and isinstance(omero, dict):
        name = omero.get("name")
    version = ms.get("version")
    if not version and isinstance(omero, dict):
        version = omero.get("version")

    return NgffImage(
        path=path,
        axes=axes,
        levels=levels,
        dtype=dtype or np.dtype("uint16"),
        channels=channels,
        sizes=sizes,
        physical=physical,
        units=units,
        omero=omero if isinstance(omero, dict) else None,
        multiscale_name=str(ms.get("name")) if ms.get("name") else None,
        name=str(name) if name else None,
        version=str(version) if version else None,
    )


def _parse_channels(omero: Any, num_c: int) -> list[NgffChannel]:
    """Build per-channel render hints from the OME `omero.channels` block; synthesize
    sensible defaults when absent so the renderer always has something to work with."""
    out: list[NgffChannel] = []
    omero_channels = omero.get("channels") if isinstance(omero, dict) else None
    if isinstance(omero_channels, list) and omero_channels:
        for i, ch in enumerate(omero_channels):
            ch = ch if isinstance(ch, dict) else {}
            window = ch.get("window") if isinstance(ch.get("window"), dict) else {}
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
    defaults = ["FFFFFF"] if num_c <= 1 else ["FF0000", "00FF00", "0000FF", "FFFF00", "FF00FF", "00FFFF"]
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
