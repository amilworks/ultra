"""HDF5 data-viewer reader (h5py-backed).

The frontend ships a complete HDF5 viewer (``frontend/src/components/viewer/hdf5``)
whose backend never existed. This module is that backend: it opens ``.h5``/``.hdf5``/
``.hdf`` files with :mod:`h5py` and produces exactly the JSON/PNG/binary
shapes the frontend's frozen wire contract (``frontend/src/types.ts`` ``Hdf5*`` types)
expects. It is a pure reader — no FastAPI, no libbioimage — so it is unit-testable
against synthetic files and is dispatched off the event loop through the engine pool
(see :mod:`ultra_deepagents.imaging.engine` / :mod:`ultra_deepagents.imaging.pool`).

Everything is **bounded**: the group/dataset tree walk is node-capped, every array read
is downsampled/windowed/strided to a preview budget, statistics and histograms are
sampled, and table previews are offset/limit windowed. Caps are module constants below
and are surfaced to the UI (``summary.truncated`` + ``limitations``) so a terabyte file
opens as a smaller, honest preview instead of loading a full array.

Entry points used by the engine:

- :func:`is_hdf5_data_file` — extension gate (``.ims`` Imaris deliberately excluded).
- :func:`build_hdf5_viewer_info` — the ``UploadViewerInfo.hdf5`` payload + top-level fields.
- :func:`dataset_summary` — ``Hdf5DatasetSummary``.
- :func:`slice_png` / :func:`atlas_png` — ``image/png`` bytes.
- :func:`scalar_volume` — raw voxel dict (``x-volume-*`` header contract).
- :func:`dataset_histogram` — ``Hdf5DatasetHistogramResponse``.
- :func:`table_preview` — ``Hdf5DatasetTablePreviewResponse``.
"""

from __future__ import annotations

import io
import itertools
import math
import os
from typing import Any

__all__ = [
    "HDF5_DATA_EXTS",
    "is_hdf5_data_file",
    "Hdf5Error",
    "Hdf5DatasetNotFound",
    "build_hdf5_viewer_info",
    "dataset_summary",
    "slice_png",
    "atlas_png",
    "scalar_volume",
    "dataset_histogram",
    "table_preview",
]

# --- Extension gate ---------------------------------------------------------
# ONLY these extensions are treated as HDF5-*data* (the explorer). ``.ims`` (Imaris),
# ``.mat`` v7.3, NetCDF4, and other HDF5-based containers are deliberately excluded:
# they are either images (libbioimage's job) or not images at all. Detection is by
# extension allowlist, NOT by HDF5 magic-byte sniffing, precisely so those files keep
# their existing routing.
HDF5_DATA_EXTS = (".h5", ".hdf5", ".hdf", ".dream3d")

# --- Bounding caps (documented; surfaced to the UI where relevant) ----------
MAX_TREE_NODES = 4000          # total nodes visited during the viewer tree walk
MAX_TREE_DEPTH = 64            # guards against pathological deep nesting / cycles
MAX_GROUP_CHILDREN = 512       # children materialized per group (rest → truncated)
PREVIEW_MAX_PLANE = 2048       # long-edge cap (px) for a preview slice plane
PREVIEW_MAX_VOXELS = 32_000_000  # preview grid voxel cap (~128MB float32); bounds volume+atlas
SAMPLE_MAX_VALUES = 2_000_000  # values sampled for statistics + histogram
WINDOW_SAMPLE_VALUES = 400_000  # values sampled for the robust display window
TABLE_CHART_MAX_ROWS = 4000    # rows sampled for table preview charts
HISTOGRAM_MAX_BINS = 256
DISCRETE_MAX_LABELS = 64       # >this many distinct integers → treat as continuous
LABEL_MAX_UNIQUE = 256         # unnamed integer volume with ≤this many distinct values → label_volume
ATLAS_CELL_CAP = 256           # matches viewerinfo.ATLAS_CELL_CAP (kept local to avoid import cost)
SAMPLE_CORNER = 4              # per-axis size of the sample_values corner block
MAX_ATTRS = 48                 # attributes materialized per node
MAX_ATTR_ARRAY = 32            # elements kept from an array-valued attribute
MAX_METADATA_ITEM_BYTES = 4096
MAX_METADATA_TOTAL_BYTES = 64 * 1024


# Volume preview kinds the frontend renders through the slice/atlas/volume surface.
_VOLUME_KINDS = frozenset({"scalar_volume", "label_volume", "rgb_volume", "vector_volume"})
# Dataset/group name hints that a categorical (label) integer volume.
_LABEL_NAME_HINTS = (
    "featureids", "grainids", "cellids", "phases", "phaseid", "labels", "mask",
    "segmentation", "ids", "regionids", "parentids",
)


class Hdf5Error(ValueError):
    """A recoverable HDF5-reader error. Subclasses ValueError so the service's
    existing ``@app.exception_handler(ValueError)`` can map decode-class messages to
    422 (the message carries a ``_DECODE_ERROR_MARKERS`` token like ``unsupported``)."""


class Hdf5DatasetNotFound(Hdf5Error):
    """Requested ``dataset_path`` is absent — mapped to 404."""


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def is_hdf5_data_file(name: str) -> bool:
    """True when ``name`` has an HDF5-data extension (tolerating a series/page suffix
    like ``vol.h5_2``). ``.ims`` and other HDF5-based image formats are excluded."""
    lower = (name or "").strip().lower()
    for ext in HDF5_DATA_EXTS:
        if lower.endswith(ext) or (ext + "_") in lower:
            return True
    return False


# ---------------------------------------------------------------------------
# JSON-safety + h5py helpers
# ---------------------------------------------------------------------------
def _to_jsonable(value: Any, *, _depth: int = 0) -> Any:
    """Coerce an h5py/numpy value into something ``json.dumps`` accepts.

    Handles numpy scalars/arrays, bytes (utf-8, replacement on error), non-finite
    floats (→ None, since JSON has no NaN/Inf), and nested lists/tuples. Bounded in
    depth so a pathological object attribute can't recurse forever.
    """
    import numpy as np

    if _depth > 6:
        return str(value)
    if value is None:
        return None
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.generic):
        item = value.item()
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, bytes):
            return item.decode("utf-8", "replace")
        return item
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _to_jsonable(value.item(), _depth=_depth + 1)
        return [_to_jsonable(v, _depth=_depth + 1) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v, _depth=_depth + 1) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v, _depth=_depth + 1) for k, v in value.items()}
    return str(value)


def _metadata_payload_bytes(value: Any, *, _depth: int = 0) -> int:
    """Conservative decoded-size estimate for already-bounded metadata values."""
    if _depth > 8:
        return MAX_METADATA_TOTAL_BYTES + 1
    if value is None or isinstance(value, (bool, int, float)):
        return 8
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8", "replace"))
    if isinstance(value, (list, tuple)):
        return sum(_metadata_payload_bytes(item, _depth=_depth + 1) for item in value)
    if isinstance(value, dict):
        return sum(
            len(str(key).encode("utf-8", "replace"))
            + _metadata_payload_bytes(item, _depth=_depth + 1)
            for key, item in value.items()
        )
    return len(str(value).encode("utf-8", "replace"))


def _iter_bounded_attrs(obj: Any):
    """Yield at most :data:`MAX_ATTRS` small attributes without eager key/value lists.

    Attribute array shape and fixed-width storage are checked through the attribute
    ID before reading. Variable-length attributes are accepted only as scalars and
    are rejected from the response if their decoded payload exceeds the per-item or
    cumulative byte budget.
    """
    import numpy as np

    total_bytes = 0
    try:
        keys = itertools.islice(obj.attrs.keys(), MAX_ATTRS)
        for key in keys:
            try:
                attr_id = obj.attrs.get_id(key)
                shape = tuple(int(value) for value in attr_id.shape)
                element_count = int(np.prod(shape)) if shape else 1
                dtype = attr_id.dtype
                if element_count > MAX_ATTR_ARRAY:
                    continue
                itemsize = int(getattr(dtype, "itemsize", 0) or 0)
                kind = getattr(dtype, "kind", "")
                if kind == "O" and element_count != 1:
                    continue
                if kind != "O" and (
                    itemsize <= 0
                    or itemsize > MAX_METADATA_ITEM_BYTES
                    or element_count * itemsize > MAX_METADATA_TOTAL_BYTES
                ):
                    continue
                raw = obj.attrs[key]
                value = _to_jsonable(raw)
                value_bytes = _metadata_payload_bytes(value)
                if (
                    value_bytes > MAX_METADATA_ITEM_BYTES
                    or total_bytes + value_bytes > MAX_METADATA_TOTAL_BYTES
                ):
                    continue
                total_bytes += value_bytes
                yield str(key), raw
            except Exception:  # noqa: BLE001 - one malformed attribute is isolated
                continue
    except Exception:  # noqa: BLE001 - a broken attribute block must not fail the walk
        return


def _read_attrs(obj: Any) -> dict[str, Any]:
    """A bounded, JSON-safe view of an h5py object's attributes."""
    out: dict[str, Any] = {}
    for key, raw in _iter_bounded_attrs(obj):
        try:
            out[key] = _to_jsonable(raw)
        except Exception:  # noqa: BLE001
            out[key] = None
    return out


def _dtype_str(dt: Any) -> str:
    """A readable dtype string for a dataset (compound → ``compound[a,b,...]``)."""
    try:
        if dt.names:
            return "compound[" + ",".join(dt.names) + "]"
    except Exception:  # noqa: BLE001
        pass
    if getattr(dt, "kind", "") in ("S", "U"):
        return "string"
    if getattr(dt, "kind", "") == "O":
        return "string"
    return str(dt)


def _is_compound(dt: Any) -> bool:
    return bool(getattr(dt, "names", None))


def _is_string_dtype(dt: Any) -> bool:
    return getattr(dt, "kind", "") in ("S", "U", "O")


def _is_numeric_dtype(dt: Any) -> bool:
    return getattr(dt, "kind", "") in ("i", "u", "f", "b")


def _is_integer_dtype(dt: Any) -> bool:
    return getattr(dt, "kind", "") in ("i", "u", "b")


_MAX_LINK_RESOLUTION_DEPTH = 16


def _hard_child(group: Any, key: str, *, _depth: int = 0) -> Any | None:
    """Resolve one direct child, following only links that stay INSIDE this file.

    Hard links resolve directly. Internal soft links are followed — they are used
    pervasively by legitimate scientific HDF5 (e.g. NeXus ``/entry/data`` -> detector
    data) — but the soft link's target is re-resolved through this SAME gate, so a
    soft link that chains to an external link is rejected at the external hop and the
    referenced outside file is never opened. External links (which reference OTHER
    files by path — a traversal risk on untrusted uploads), soft-link chains deeper
    than a bounded limit, and unknown link types are always rejected.
    """
    import h5py

    if not _is_group(group) or _depth > _MAX_LINK_RESOLUTION_DEPTH:
        return None
    try:
        link = group.get(str(key), getlink=True)
        if isinstance(link, h5py.HardLink):
            return group.get(str(key))
        if isinstance(link, h5py.SoftLink):
            target = str(getattr(link, "path", "") or "")
            if not target:
                return None
            root = group.file if target.startswith("/") else group
            return _hard_object_at_path(root, target, _depth=_depth + 1)
        # ExternalLink or an unknown link type: never dereferenced.
        return None
    except Exception:  # noqa: BLE001
        return None


def _hard_object_at_path(h5: Any, path: str, *, _depth: int = 0) -> Any | None:
    """Resolve ``path`` component-by-component, following only in-file links.

    ``_depth`` tracks soft-link resolution hops (not path length) so a soft-link
    chain cannot recurse without bound.
    """
    text = str(path or "")
    if "\x00" in text:
        return None
    parts = text.split("/")
    if text.startswith("/"):
        parts = parts[1:]
    if parts and parts[-1] == "":
        parts = parts[:-1]
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return h5 if text in {"/", ""} else None
    current = h5
    for index, part in enumerate(parts):
        current = _hard_child(current, part, _depth=_depth)
        if current is None:
            return None
        if index < len(parts) - 1 and not _is_group(current):
            return None
    return current


def _bounded_child_names(group: Any, limit: int = MAX_GROUP_CHILDREN):
    """Iterate child names lazily; obtaining a name does not dereference its link."""
    try:
        yield from itertools.islice(group.keys(), max(0, int(limit)))
    except Exception:  # noqa: BLE001
        return


def _object_address(obj: Any) -> int | None:
    try:
        import h5py

        return int(h5py.h5o.get_info(obj.id).addr)
    except Exception:  # noqa: BLE001
        return None


def _visit_hard_objects(h5: Any, callback: Any) -> None:
    """Bounded depth-first visitor over hard-linked groups/datasets only."""
    seen: set[int] = set()
    root_address = _object_address(h5)
    if root_address is not None:
        seen.add(root_address)
    state = {"nodes": 0, "stop": False}

    def _walk(group: Any, prefix: str, depth: int) -> None:
        if state["stop"] or depth > MAX_TREE_DEPTH:
            return
        for key in _bounded_child_names(group):
            if state["nodes"] >= MAX_TREE_NODES:
                state["stop"] = True
                return
            child = _hard_child(group, key)
            if child is None:
                continue
            address = _object_address(child)
            if address is not None and address in seen:
                continue
            if address is not None:
                seen.add(address)
            state["nodes"] += 1
            name = f"{prefix}/{key}" if prefix else str(key)
            if callback(name, child) is True:
                state["stop"] = True
                return
            if _is_group(child):
                _walk(child, name, depth + 1)
                if state["stop"]:
                    return

    _walk(h5, "", 1)


# ---------------------------------------------------------------------------
# Preview-kind classification + volume interpretation
# ---------------------------------------------------------------------------
def _classify_shallow(name: str, dt: Any, shape: tuple[int, ...]) -> str | None:
    """Preview-kind guess WITHOUT reading data (used for tree nodes + as the
    starting point for :func:`dataset_summary`). Label-vs-scalar and rgb-vs-vector
    are refined by sampling in the summary; here we use dtype/shape/name only."""
    if _is_compound(dt):
        return "table"
    rank = len(shape)
    if _is_string_dtype(dt):
        return "table" if rank == 1 else None
    if not _is_numeric_dtype(dt):
        return None
    if rank == 0:
        return None
    if rank == 1:
        return "series"

    def _scalar_or_label() -> str:
        # Name-hinted categorical volumes (FeatureIds/GrainIds/Phases/mask/...) render as
        # a label palette. dataset_summary additionally promotes an *unnamed* integer
        # volume with few distinct values via sampling (LABEL_MAX_UNIQUE).
        if _is_integer_dtype(dt) and any(h in name.strip().lower() for h in _LABEL_NAME_HINTS):
            return "label_volume"
        return "scalar_volume"

    if rank == 2:
        return _scalar_or_label()
    if rank == 3:
        last = shape[-1]
        if last in (3, 4) and shape[0] > last and shape[1] > last:
            # A 2D multi-component image (H, W, C).
            if _is_integer_dtype(dt):
                return "rgb_volume"
            return "vector_volume"
        return _scalar_or_label()
    if rank == 4:
        c = shape[-1]
        if c == 1:
            return _scalar_or_label()
        if c in (3, 4) and _is_integer_dtype(dt):
            return "rgb_volume"
        return "vector_volume"
    return None


def _interpret_volume(shape: tuple[int, ...], dt: Any) -> dict[str, Any] | None:
    """Map a numeric dataset shape onto a (Z, Y, X, C) preview volume + strides.

    Returns ``None`` for shapes that are not a renderable volume (rank 0/1, rank≥5
    without a component interpretation). ``layout`` records how to index the native
    dataset; ``s?`` are per-axis strides that fold the native grid into a preview
    grid (``p?``) bounded by :data:`PREVIEW_MAX_PLANE` / :data:`PREVIEW_MAX_VOXELS`.
    """
    rank = len(shape)
    if rank == 2:
        z, y, x, comp, layout = 1, int(shape[0]), int(shape[1]), 1, "hw"
    elif rank == 3:
        last = shape[-1]
        if last in (3, 4) and shape[0] > last and shape[1] > last:
            z, y, x, comp, layout = 1, int(shape[0]), int(shape[1]), int(last), "hwc"
        else:
            z, y, x, comp, layout = int(shape[0]), int(shape[1]), int(shape[2]), 1, "zyx"
    elif rank == 4:
        z, y, x, comp, layout = int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3]), "zyxc"
    else:
        return None
    if min(z, y, x) <= 0:
        return None
    sy = max(1, math.ceil(max(y, x) / PREVIEW_MAX_PLANE)) if max(y, x) > PREVIEW_MAX_PLANE else 1
    sx = sy
    sz = 1
    # Bound total preview voxels; grow strides uniformly-ish until under budget.
    def _pdim(n: int, s: int) -> int:
        return (n + s - 1) // s
    while _pdim(z, sz) * _pdim(y, sy) * _pdim(x, sx) > PREVIEW_MAX_VOXELS:
        if _pdim(z, sz) >= _pdim(y, sy) and _pdim(z, sz) >= _pdim(x, sx) and z > 1:
            sz += 1
        else:
            sy += 1
            sx = sy
    return {
        "layout": layout,
        "z": z, "y": y, "x": x, "comp": comp,
        "sz": sz, "sy": sy, "sx": sx,
        "pz": _pdim(z, sz), "py": _pdim(y, sy), "px": _pdim(x, sx),
    }


def _refine_preview_kind(shallow: str | None, dt: Any, vol: dict[str, Any] | None,
                         name: str, integer_unique: int | None) -> str:
    """Finalize the volume preview kind using the sampled unique count (for
    label detection) and the shallow guess. Non-volume kinds pass through."""
    if shallow not in _VOLUME_KINDS or vol is None:
        return shallow or ""
    comp = vol["comp"]
    layout = vol["layout"]
    if layout in ("hwc", "zyxc") and comp >= 2:
        if comp in (3, 4) and _is_integer_dtype(dt):
            return "rgb_volume"
        return "vector_volume"
    # Single-component numeric volume: label vs scalar.
    if _is_integer_dtype(dt):
        low = name.strip().lower()
        name_is_label = any(h in low for h in _LABEL_NAME_HINTS)
        few_unique = integer_unique is not None and integer_unique <= LABEL_MAX_UNIQUE
        if name_is_label or few_unique:
            return "label_volume"
    return "scalar_volume"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def _strided_sample(dset: Any, vol: dict[str, Any] | None, comp: int, max_values: int):
    """A bounded 1-D float32 sample of a numeric dataset's values (component-selected
    for vector layouts). Reads a strided hyperslab so even a huge dataset costs a
    bounded number of element reads."""
    import numpy as np

    shape = dset.shape
    if vol is None:
        # Fallback: rank 1 series or unusual numeric shape.
        n = int(np.prod(shape)) if shape else 1
        step = max(1, math.ceil(n / max_values)) if n > max_values else 1
        try:
            flat = np.asarray(dset[()]).ravel()[::step]
        except Exception:  # noqa: BLE001
            return np.zeros(0, dtype="float32")
        return flat.astype("float32", copy=False)
    z, y, x = vol["z"], vol["y"], vol["x"]
    total = z * y * x
    factor = max(1, math.ceil((total / max(1, max_values)) ** (1.0 / 3.0))) if total > max_values else 1
    sz2, sy2, sx2 = factor, factor, factor
    layout = vol["layout"]
    try:
        if layout == "zyx":
            arr = dset[::sz2, ::sy2, ::sx2]
        elif layout == "zyxc":
            arr = dset[::sz2, ::sy2, ::sx2, comp]
        elif layout == "hw":
            arr = dset[::sy2, ::sx2]
        elif layout == "hwc":
            arr = dset[::sy2, ::sx2, comp]
        else:  # pragma: no cover - defensive
            arr = np.asarray(dset[()])
    except Exception:  # noqa: BLE001
        return np.zeros(0, dtype="float32")
    return np.asarray(arr).ravel().astype("float32", copy=False)


def _sample_statistics(values, dt: Any) -> dict[str, Any]:
    """min/max/mean + (for integers) unique count over a bounded value sample."""
    import numpy as np

    values = values[np.isfinite(values)] if values.size else values
    n = int(values.size)
    stats: dict[str, Any] = {"sample_count": n, "min": None, "max": None, "mean": None, "unique_values": None}
    if n == 0:
        return stats
    stats["min"] = float(values.min())
    stats["max"] = float(values.max())
    stats["mean"] = float(values.mean())
    if _is_integer_dtype(dt):
        uniq = np.unique(values)
        stats["unique_values"] = int(uniq.size)
    return stats


def _robust_window(values) -> tuple[float, float]:
    """A [p1, p99] display window; guarantees hi > lo so mapping never divides by 0."""
    import numpy as np

    values = values[np.isfinite(values)] if getattr(values, "size", 0) else values
    if getattr(values, "size", 0) == 0:
        return (0.0, 1.0)
    lo, hi = (float(v) for v in np.percentile(values, (1.0, 99.0)))
    if not hi > lo:
        hi = lo + 1.0
    return (lo, hi)


# Per-worker cache of the global display window, keyed on (path, mtime, dataset, comp),
# so a z-scrub of one dataset shares a single mapping without re-sampling every slice.
_WINDOW_CACHE: dict[tuple, tuple[float, float]] = {}


def _global_window(path: str, dset: Any, dataset_path: str, vol: dict[str, Any] | None, comp: int) -> tuple[float, float]:
    try:
        mtime = os.stat(path).st_mtime_ns
    except OSError:
        mtime = 0
    key = (path, mtime, dataset_path, comp)
    cached = _WINDOW_CACHE.get(key)
    if cached is not None:
        return cached
    win = _robust_window(_strided_sample(dset, vol, comp, WINDOW_SAMPLE_VALUES))
    if len(_WINDOW_CACHE) > 512:
        _WINDOW_CACHE.clear()
    _WINDOW_CACHE[key] = win
    return win


# ---------------------------------------------------------------------------
# Normalization + PNG encoding
# ---------------------------------------------------------------------------
def _window_to_uint8(plane, lo: float, hi: float):
    import numpy as np

    scale = 255.0 / max(float(hi) - float(lo), 1e-6)
    a = np.asarray(plane, dtype="float32")
    return np.clip((a - float(lo)) * scale, 0.0, 255.0).astype("uint8")


def _label_to_rgb(plane):
    """Deterministic categorical palette: label→distinct color, background(0)→black."""
    import numpy as np

    p = np.asarray(plane).astype("int64")
    r = ((p * 131 + 53) % 256).astype("uint8")
    g = ((p * 197 + 97) % 256).astype("uint8")
    b = ((p * 251 + 149) % 256).astype("uint8")
    out = np.stack([r, g, b], axis=-1)
    out[p == 0] = 0
    return out


def _rgb_to_uint8(plane):
    """A (H, W, C) plane → (H, W, 3) uint8. uint8 data passes through; other dtypes
    are per-channel min/max scaled."""
    import numpy as np

    a = np.asarray(plane)
    if a.ndim == 2:
        a = a[..., np.newaxis]
    a = a[..., :3]
    if a.shape[-1] < 3:  # pad a 1/2-channel to 3 by repetition
        a = np.repeat(a[..., :1], 3, axis=-1)
    if a.dtype == np.uint8:
        return np.ascontiguousarray(a)
    out = np.empty(a.shape, dtype="uint8")
    for c in range(a.shape[-1]):
        ch = a[..., c].astype("float32")
        lo, hi = float(ch.min()), float(ch.max())
        scale = 255.0 / max(hi - lo, 1e-6)
        out[..., c] = np.clip((ch - lo) * scale, 0.0, 255.0).astype("uint8")
    return out


def _encode_png(arr) -> bytes:
    """(H, W) uint8 grayscale or (H, W, 3) uint8 RGB → PNG bytes."""
    import numpy as np
    from PIL import Image

    a = np.asarray(arr)
    if 0 in a.shape:
        raise Hdf5Error("cannot encode an empty region")
    if a.dtype != np.uint8:
        a = a.astype("uint8")
    mode = "L" if a.ndim == 2 else "RGB"
    buf = io.BytesIO()
    Image.fromarray(a, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Plane extraction (bounded, strided)
# ---------------------------------------------------------------------------
def _read_preview_plane(dset: Any, vol: dict[str, Any], axis: str, index: int, comp: int, *, rgb: bool):
    """Extract one preview plane (matching the summary's ``preview_planes`` dims).

    ``index`` is a *preview-grid* index; it maps to the native index ``index*stride``.
    Reads a single native plane/slab, strided in the in-plane axes, so cost is bounded
    to one plane regardless of dataset depth.
    """
    import numpy as np

    layout = vol["layout"]
    sz, sy, sx = vol["sz"], vol["sy"], vol["sx"]
    pz, py, px = vol["pz"], vol["py"], vol["px"]
    comp = max(0, min(comp, vol["comp"] - 1))

    def _clamp(i: int, n: int) -> int:
        return max(0, min(int(i), n - 1))

    if layout == "hw":
        arr = dset[::sy, ::sx]
    elif layout == "hwc":
        if rgb:
            arr = dset[::sy, ::sx, :3]
        else:
            arr = dset[::sy, ::sx, comp]
    elif layout == "zyx":
        if axis == "y":
            iy = _clamp(index, py) * sy
            arr = dset[:, iy, :][::sz, ::sx]
        elif axis == "x":
            ix = _clamp(index, px) * sx
            arr = dset[:, :, ix][::sz, ::sy]
        else:  # z
            iz = _clamp(index, pz) * sz
            arr = dset[iz][::sy, ::sx]
    elif layout == "zyxc":
        if axis == "y":
            iy = _clamp(index, py) * sy
            slab = dset[:, iy, :, :3] if rgb else dset[:, iy, :, comp]
            arr = slab[::sz, ::sx, :] if rgb else slab[::sz, ::sx]
        elif axis == "x":
            ix = _clamp(index, px) * sx
            slab = dset[:, :, ix, :3] if rgb else dset[:, :, ix, comp]
            arr = slab[::sz, ::sy, :] if rgb else slab[::sz, ::sy]
        else:  # z
            iz = _clamp(index, pz) * sz
            slab = dset[iz, :, :, :3] if rgb else dset[iz, :, :, comp]
            arr = slab[::sy, ::sx, :] if rgb else slab[::sy, ::sx]
    else:  # pragma: no cover - defensive
        raise Hdf5Error("unsupported dataset layout for slicing")
    return np.ascontiguousarray(arr)


def _read_preview_volume(dset: Any, vol: dict[str, Any], comp: int):
    """The full strided preview volume as float32 (Z', Y', X'). Bounded by
    :data:`PREVIEW_MAX_VOXELS`. Used by the scalar-volume endpoint."""
    import numpy as np

    layout = vol["layout"]
    sz, sy, sx = vol["sz"], vol["sy"], vol["sx"]
    comp = max(0, min(comp, vol["comp"] - 1))
    if layout == "zyx":
        arr = dset[::sz, ::sy, ::sx]
    elif layout == "zyxc":
        arr = dset[::sz, ::sy, ::sx, comp]
    elif layout == "hw":
        arr = dset[::sy, ::sx][np.newaxis, ...]
    elif layout == "hwc":
        arr = dset[::sy, ::sx, comp][np.newaxis, ...]
    else:  # pragma: no cover
        raise Hdf5Error("unsupported dataset layout for scalar volume")
    return np.ascontiguousarray(arr, dtype="float32")


# ---------------------------------------------------------------------------
# Geometry (_SIMPL_GEOMETRY)
# ---------------------------------------------------------------------------
def _bounded_geometry_vector(dset: Any) -> list[Any] | None:
    """Read one tiny geometry vector without trusting its dataset name alone.

    DREAM.3D geometry vectors normally contain three values.  A malformed file can
    attach that name to an arbitrarily large dataset, so reject rather than eagerly
    materialize anything beyond the existing attribute-array bound.
    """
    import numpy as np

    if dset is None or not _is_dataset(dset):
        return None
    try:
        dtype = dset.dtype
        itemsize = int(getattr(dtype, "itemsize", 0) or 0)
        if getattr(dtype, "kind", "") not in {"i", "u", "f"} or not (0 < itemsize <= 8):
            return None
        shape = tuple(int(value) for value in dset.shape)
        size = int(np.prod(shape)) if shape else 1
        if size < 3 or size > MAX_ATTR_ARRAY:
            return None
        values = np.asarray(dset[...]).ravel()
        return [_to_jsonable(value) for value in values[:3]]
    except Exception:  # noqa: BLE001
        return None


def _valid_geometry_vector(values: Any, *, positive: bool) -> bool:
    if not isinstance(values, list) or len(values) < 3:
        return False
    try:
        numbers = [float(value) for value in values[:3]]
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(value) and (not positive or value > 0) for value in numbers)


















def _cell_dataset_matches_geometry(shape: tuple[int, ...], dimensions: list[Any]) -> bool:
    """Return whether a CellData tuple shape is consistent with image dimensions.

    DREAM.3D stores ``DIMENSIONS`` as X/Y/Z while its HDF5 image arrays are
    normally Z/Y/X/(components). Some writers flatten cell tuples instead. Both
    representations are accepted; a merely present ``CellData`` group is not.
    """

    try:
        xyz = tuple(int(value) for value in dimensions[:3])
    except (TypeError, ValueError):
        return False
    if len(xyz) != 3 or any(value <= 0 for value in xyz):
        return False
    if len(shape) >= 3 and tuple(shape[:3]) in {xyz, tuple(reversed(xyz))}:
        return True
    voxel_count = math.prod(xyz)
    return bool(shape) and int(shape[0]) == voxel_count and len(shape) <= 2


def _geometry_cell_data_evidence(parent: Any, dimensions: list[Any] | None) -> tuple[bool, bool, str | None]:
    """Inspect bounded CellData metadata for geometry/tuple-shape consistency."""

    if parent is None or not isinstance(dimensions, list):
        return False, False, None
    has_cell_data = False
    first_cell_data_path: str | None = None
    for key in ("CellData", "Cell Data", "Cell_Data"):
        group = _hard_child(parent, key)
        if not _is_group(group):
            continue
        has_cell_data = True
        group_path = str(getattr(group, "name", "") or "") or None
        if first_cell_data_path is None:
            first_cell_data_path = group_path
        for child_key in _bounded_child_names(group):
            try:
                child = _hard_child(group, child_key)
                if child is None:
                    continue
                shape = tuple(int(value) for value in child.shape)
            except Exception:  # noqa: BLE001
                continue
            itemsize = int(getattr(getattr(child, "dtype", None), "itemsize", 0) or 0)
            if (
                _is_dataset(child)
                and getattr(child.dtype, "kind", "") in {"i", "u", "f", "b"}
                and 0 < itemsize <= 16
                and _cell_dataset_matches_geometry(shape, dimensions)
            ):
                return has_cell_data, True, group_path
    return has_cell_data, False, first_cell_data_path


def _find_geometry(h5: Any) -> dict[str, Any] | None:
    """Find the best DREAM.3D image geometry, not merely the first marker.

    StatsGenerator data containers commonly carry an empty ``_SIMPL_GEOMETRY``
    before the populated image container.  Candidates are ranked by valid geometry
    fields and then by an associated CellData group.  Only the three small geometry
    vectors are read, and metadata traversal stops at :data:`MAX_TREE_NODES`.
    """
    found: dict[str, Any] | None = None
    found_score: tuple[int, int, int, int, float] = (-1, -1, -1, -1, -1.0)
    visited = 0

    def _visit(name: str, obj: Any) -> bool | None:
        nonlocal found, found_score, visited
        visited += 1
        if visited > MAX_TREE_NODES:
            return True
        base = name.rsplit("/", 1)[-1]
        if base != "_SIMPL_GEOMETRY":
            return None
        try:
            dims = _hard_child(obj, "DIMENSIONS")
            spacing = _hard_child(obj, "SPACING")
            origin = _hard_child(obj, "ORIGIN")
            dimensions = _bounded_geometry_vector(dims)
            spacing_values = _bounded_geometry_vector(spacing)
            origin_values = _bounded_geometry_vector(origin)
            valid_dimensions = _valid_geometry_vector(dimensions, positive=True)
            valid_spacing = _valid_geometry_vector(spacing_values, positive=True)
            valid_origin = _valid_geometry_vector(origin_values, positive=False)
            valid_fields = sum((valid_dimensions, valid_spacing, valid_origin))
            parent = getattr(obj, "parent", None)
            has_cell_data, cell_data_consistent, cell_data_path = _geometry_cell_data_evidence(
                parent, dimensions
            )
            complete = bool(
                valid_dimensions
                and valid_spacing
                and valid_origin
                and cell_data_consistent
            )
            voxel_count = (
                math.prod(float(value) for value in dimensions[:3])
                if valid_dimensions and dimensions is not None
                else 0.0
            )
            score = (
                int(complete),
                valid_fields,
                int(cell_data_consistent),
                int(has_cell_data),
                voxel_count,
            )
            candidate = {
                "path": "/" + name,
                "dimensions": dimensions,
                "spacing": spacing_values,
                "origin": origin_values,
                "cell_data_path": cell_data_path,
                "cell_data_consistent": cell_data_consistent,
                "complete": complete,
            }
            if score > found_score:
                found = candidate
                found_score = score
        except Exception:  # noqa: BLE001
            return None
        return None

    try:
        _visit_hard_objects(h5, _visit)
    except Exception:  # noqa: BLE001
        return None
    return found


def _semantic_role(name: str, kind: str) -> str:
    low = name.strip().lower()
    if "ipf" in low or ("colors" in low and "ipf" in low):
        return "ipf_colors"
    if "ipfcolor" in low:
        return "ipf_colors"
    if "featureids" in low or "grainids" in low:
        return "feature_ids"
    if "confidence" in low:
        return "confidence_index"
    if "imagequality" in low or "image_quality" in low:
        return "image_quality"
    if "euler" in low:
        return "euler_angles"
    if "phase" in low:
        return "phases"
    if "quat" in low:
        return "quaternions"
    if kind == "label_volume":
        return "labels"
    if kind == "rgb_volume":
        return "color_map"
    if kind == "vector_volume":
        return "vector_field"
    return "scalar_field"


def _is_dataset(obj: Any) -> bool:
    import h5py

    return isinstance(obj, h5py.Dataset)


def _is_group(obj: Any) -> bool:
    import h5py

    return isinstance(obj, h5py.Group)


# ---------------------------------------------------------------------------
# Tree walk (bounded)
# ---------------------------------------------------------------------------
def _walk_tree(h5: Any) -> tuple[list[dict[str, Any]], dict[str, int], bool, int, int, str | None]:
    """Walk groups/datasets into ``Hdf5ViewerTreeNode``s (bounded by node/depth caps).

    Returns ``(tree, dataset_kinds, truncated, group_count, dataset_count, default_path)``.
    ``default_path`` is the largest renderable dataset (by element count), which the
    frontend uses as the initial selection.
    """
    import numpy as np

    state = {"nodes": 0, "truncated": False, "groups": 0, "datasets": 0}
    dataset_kinds: dict[str, int] = {}
    best_default = {"path": None, "score": -1.0}
    seen: set[int] = set()
    root_address = _object_address(h5)
    if root_address is not None:
        seen.add(root_address)

    def _node(name: str, obj: Any, path: str, depth: int) -> dict[str, Any] | None:
        if state["nodes"] >= MAX_TREE_NODES or depth > MAX_TREE_DEPTH:
            state["truncated"] = True
            return None
        state["nodes"] += 1
        attrs_count = 0
        try:
            attrs_count = len(obj.attrs)
        except Exception:  # noqa: BLE001
            attrs_count = 0
        if _is_group(obj):
            state["groups"] += 1
            try:
                child_count = len(obj)
            except Exception:  # noqa: BLE001
                child_count = 0
            children: list[dict[str, Any]] = []
            for key in _bounded_child_names(obj):
                if state["nodes"] >= MAX_TREE_NODES:
                    state["truncated"] = True
                    break
                child = _hard_child(obj, key)
                if child is None:
                    continue
                address = _object_address(child)
                if address is not None and address in seen:
                    continue
                if address is not None:
                    seen.add(address)
                child_path = (path.rstrip("/") + "/" + key) if path != "/" else "/" + key
                built = _node(key, child, child_path, depth + 1)
                if built is not None:
                    children.append(built)
            if child_count > MAX_GROUP_CHILDREN:
                state["truncated"] = True
            return {
                "path": path,
                "name": name,
                "node_type": "group",
                "child_count": int(child_count),
                "attributes_count": int(attrs_count),
                "shape": None,
                "dtype": None,
                "preview_kind": None,
                "children": children,
            }
        if _is_dataset(obj):
            state["datasets"] += 1
            try:
                dt = obj.dtype
                shape = tuple(int(s) for s in obj.shape)
            except Exception:  # noqa: BLE001
                dt, shape = None, ()
            kind = _classify_shallow(name, dt, shape) if dt is not None else None
            token = kind or "metadata"
            dataset_kinds[token] = dataset_kinds.get(token, 0) + 1
            if kind in _VOLUME_KINDS and shape:
                score = float(np.prod(shape))
                if score > best_default["score"]:
                    best_default["score"] = score
                    best_default["path"] = path
            return {
                "path": path,
                "name": name,
                "node_type": "dataset",
                "child_count": 0,
                "attributes_count": int(attrs_count),
                "shape": list(shape),
                "dtype": _dtype_str(dt) if dt is not None else None,
                "preview_kind": kind,
                "children": [],
            }
        return None

    tree: list[dict[str, Any]] = []
    try:
        root_child_count = len(h5)
    except Exception:  # noqa: BLE001
        root_child_count = 0
    for key in _bounded_child_names(h5):
        child = _hard_child(h5, key)
        if child is None:
            continue
        address = _object_address(child)
        if address is not None and address in seen:
            continue
        if address is not None:
            seen.add(address)
        built = _node(key, child, "/" + key, 1)
        if built is not None:
            tree.append(built)
    if root_child_count > MAX_GROUP_CHILDREN:
        state["truncated"] = True

    return (
        tree,
        dataset_kinds,
        state["truncated"],
        state["groups"],
        state["datasets"],
        best_default["path"],
    )


# ---------------------------------------------------------------------------
# viewerinfo payload
# ---------------------------------------------------------------------------
def build_hdf5_viewer_info(path: str, *, file_id: str = "", original_name: str = "") -> dict[str, Any]:
    """Build the ``kind:"hdf5"`` viewer-info payload for ``path``.

    Never raises on an undecodable/corrupt HDF5 file: returns ``supported:false`` +
    ``status:"unsupported"`` + a human ``error`` + an empty tree so the frontend
    renders "Metadata only" instead of crashing into the image pipeline.
    """
    name = original_name or os.path.basename(path)
    top: dict[str, Any] = {
        "kind": "hdf5",
        "file_id": file_id,
        "original_name": name,
        "modality": "image",
        "reader": "h5py",
    }
    try:
        import h5py
    except Exception:  # pragma: no cover - h5py is a hard dep of the imaging extra
        top["modality"] = "image"
        top["hdf5"] = _unsupported_hdf5("HDF5 reader (h5py) is not available")
        return top

    try:
        h5 = h5py.File(path, "r")
    except Exception as exc:  # noqa: BLE001
        top["hdf5"] = _unsupported_hdf5(f"could not open as HDF5: {exc}")
        return top

    try:
        root_keys = [str(key) for key in _bounded_child_names(h5)]
        root_attributes = _read_attrs(h5)
        tree, dataset_kinds, truncated, group_count, dataset_count, default_path = _walk_tree(h5)

        # Default selection: the largest renderable dataset from the tree walk.
        default_dataset_path = default_path

        geometry = _find_geometry(h5)

        limitations: list[str] = []
        if truncated:
            limitations.append(
                f"This file is large; the group tree was truncated to {MAX_TREE_NODES} nodes "
                "for a fast open. Datasets beyond the shown tree are not listed."
            )
        limitations.append("Previews are sampled/downsampled for interactivity; exact values are in the source file.")

        hdf5_block = {
            "enabled": True,
            "supported": True,
            "status": "ready",
            "error": None,
            "root_keys": root_keys,
            "root_attributes": root_attributes,
            "summary": {
                "group_count": int(group_count),
                "dataset_count": int(dataset_count),
                "dataset_kinds": dataset_kinds,
                "truncated": bool(truncated),
                "geometry": geometry,
            },
            "tree": tree,
            "limitations": limitations,
            "selected_dataset_path": default_dataset_path,
            "default_dataset_path": default_dataset_path,
        }
        top["hdf5"] = hdf5_block
        return top
    except Exception as exc:  # noqa: BLE001 - any walk failure degrades to metadata-only
        top["hdf5"] = _unsupported_hdf5(f"could not read HDF5 structure: {exc}")
        return top
    finally:
        try:
            h5.close()
        except Exception:  # noqa: BLE001
            pass


def _unsupported_hdf5(reason: str) -> dict[str, Any]:
    return {
        "enabled": True,
        "supported": False,
        "status": "unsupported",
        "error": reason,
        "root_keys": [],
        "root_attributes": {},
        "summary": {
            "group_count": 0,
            "dataset_count": 0,
            "dataset_kinds": {},
            "truncated": False,
            "geometry": None,
        },
        "tree": [],
        "limitations": [reason],
        "selected_dataset_path": None,
        "default_dataset_path": None,
    }


# ---------------------------------------------------------------------------
# Dataset resolution
# ---------------------------------------------------------------------------
def _open(path: str):
    import h5py

    try:
        return h5py.File(path, "r")
    except Exception as exc:  # noqa: BLE001
        raise Hdf5Error(f"could not open HDF5 file (unsupported or corrupt): {exc}") from exc


def _resolve_dataset(h5: Any, dataset_path: str):
    if not dataset_path or not str(dataset_path).strip():
        raise Hdf5DatasetNotFound("dataset not found: empty dataset_path")
    obj = _hard_object_at_path(h5, str(dataset_path).strip())
    if obj is None:
        raise Hdf5DatasetNotFound(f"dataset not found: {dataset_path}")
    if not _is_dataset(obj):
        raise Hdf5DatasetNotFound(f"path is a group, not a dataset: {dataset_path}")
    return obj


# ---------------------------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------------------------
def dataset_summary(path: str, dataset_path: str, *, file_id: str = "") -> dict[str, Any]:
    import numpy as np

    h5 = _open(path)
    try:
        dset = _resolve_dataset(h5, dataset_path)
        dt = dset.dtype
        shape = tuple(int(s) for s in dset.shape)
        rank = len(shape)
        name = dataset_path.rsplit("/", 1)[-1] or dataset_path
        element_count = int(np.prod(shape)) if shape else 1
        try:
            itemsize = int(dt.itemsize)
        except Exception:  # noqa: BLE001
            itemsize = 0
        estimated_bytes = itemsize * element_count if itemsize else None

        shallow = _classify_shallow(name, dt, shape)
        vol = _interpret_volume(shape, dt) if shallow in _VOLUME_KINDS else None

        attributes = _read_attrs(dset)

        # Geometry (DREAM.3D) attached if the dataset lives under an image geometry.
        geometry = _find_geometry(h5)

        # Structured fields (compound) → columns.
        structured_fields: list[dict[str, str]] = []
        if _is_compound(dt):
            for fname in dt.names:
                structured_fields.append({"name": str(fname), "dtype": _dtype_str(dt.fields[fname][0])})

        # Sampling (numeric volumes/series only).
        sample_stats = None
        integer_unique = None
        if _is_numeric_dtype(dt) and rank >= 1:
            sample = _strided_sample(dset, vol, 0, SAMPLE_MAX_VALUES)
            sample_stats = _sample_statistics(sample, dt)
            integer_unique = sample_stats.get("unique_values")

        preview_kind = _refine_preview_kind(shallow, dt, vol, name, integer_unique) or None

        # Component handling (vector/rgb volumes).
        component_count = int(vol["comp"]) if vol is not None else 1
        component_labels: list[str] = []
        if preview_kind == "vector_volume":
            component_labels = _component_labels(name, component_count)

        # Volume eligibility + policies.
        is_volume_kind = preview_kind in _VOLUME_KINDS
        volume_eligible = bool(is_volume_kind and vol is not None and (vol["pz"] > 1))
        volume_reason = None
        if is_volume_kind and not volume_eligible:
            if vol is not None and vol["pz"] <= 1:
                volume_reason = "2-D dataset — no volume axis"
            else:
                volume_reason = "Dataset has no renderable volume axis"
        elif not is_volume_kind:
            volume_reason = None

        render_policy = _render_policy(preview_kind)
        texture_policy = "nearest" if preview_kind == "label_volume" else "linear"
        delivery_mode = _delivery_mode(preview_kind, render_policy)

        # Axis sizes / dimension summary (PREVIEW grid — what the slice/volume endpoints serve).
        axis_sizes = None
        dimension_summary = None
        slice_axes: list[str] = []
        preview_planes: dict[str, Any] = {}
        atlas_scheme = None
        physical_spacing = _spacing_from_geometry(geometry)

        if vol is not None:
            pz, py, px = vol["pz"], vol["py"], vol["px"]
            axis_sizes = {"T": 1, "C": component_count, "Z": int(pz), "Y": int(py), "X": int(px)}
            dimension_summary = {"z": int(pz), "y": int(py), "x": int(px)}
            slice_axes = ["z", "y", "x"] if pz > 1 else ["z"]
            preview_planes = _build_preview_planes(vol, slice_axes, physical_spacing)
            if volume_eligible and render_policy != "scalar":
                atlas_scheme = _atlas_scheme(px, py, pz)

        # Capabilities (load-bearing: "volume" gates the Volume tab; "histogram" gates
        # the Distribution tab). Only numeric renderable data gets histogram.
        capabilities: list[str] = []
        if is_volume_kind:
            capabilities.append("slice")
        if volume_eligible:
            capabilities.append("volume")
        if _is_numeric_dtype(dt) and preview_kind in ("scalar_volume", "label_volume", "vector_volume"):
            capabilities.append("histogram")
        if sample_stats is not None:
            capabilities.append("statistics")

        # sample_values: a tiny corner excerpt (JSON-safe).
        sample_values = _sample_values_excerpt(dset, dt, shape)
        sample_shape = list(np.asarray(sample_values).shape) if sample_values is not None else []

        summary: dict[str, Any] = {
            "file_id": file_id,
            "dataset_path": dataset_path,
            "dataset_name": name,
            "preview_kind": preview_kind,
            "semantic_role": _semantic_role(name, preview_kind or "") if is_volume_kind else None,
            "units_hint": _units_hint(name, dset),
            "dtype": _dtype_str(dt),
            "shape": list(shape),
            "rank": int(rank),
            "element_count": element_count,
            "estimated_bytes": estimated_bytes,
            "dimension_summary": dimension_summary,
            "capabilities": capabilities,
            "render_policy": render_policy,
            "delivery_mode": delivery_mode,
            "diagnostic_surface": "none",
            "first_paint_mode": "image",
            "measurement_policy": "spacing-aware" if (physical_spacing and any(physical_spacing.values())) else "pixel-only",
            "texture_policy": texture_policy,
            "display_capabilities": [],
            "viewer_capabilities": [],
            "volume_eligible": volume_eligible,
            "volume_reason": volume_reason,
            "axis_sizes": axis_sizes,
            "physical_spacing": physical_spacing,
            "atlas_scheme": atlas_scheme,
            "attributes": attributes,
            "geometry": geometry,
            "structured_fields": structured_fields,
            "component_count": component_count,
            "component_labels": component_labels,
            "slice_axes": slice_axes,
            "preview_planes": preview_planes,
            "sample_shape": sample_shape,
            "sample_values": sample_values,
            "sample_statistics": sample_stats,
        }
        return summary
    finally:
        _safe_close(h5)


def _component_labels(name: str, count: int) -> list[str]:
    low = name.strip().lower()
    if "euler" in low and count == 3:
        return ["phi1", "Phi", "phi2"]
    if "quat" in low and count == 4:
        return ["q0", "q1", "q2", "q3"]
    return [f"component_{i}" for i in range(count)]


def _render_policy(preview_kind: str | None) -> str:
    if preview_kind == "scalar_volume":
        return "scalar"
    if preview_kind == "label_volume":
        return "categorical"
    if preview_kind == "rgb_volume":
        return "display"
    if preview_kind == "vector_volume":
        return "analysis"
    return "scalar"


def _delivery_mode(preview_kind: str | None, render_policy: str) -> str:
    if preview_kind in ("label_volume", "rgb_volume", "vector_volume"):
        return "atlas"
    if render_policy == "scalar" and preview_kind == "scalar_volume":
        return "scalar"
    return "direct"


def _units_hint(name: str, dataset: Any | None = None) -> str | None:
    """Return only bounded units declared in dataset metadata.

    Dataset names such as ``EulerAngles`` and ``EquivalentDiameters`` do not
    establish radians or micrometres; DREAM.3D pipelines can store either with
    other conventions. Unknown units stay unknown instead of becoming a
    scientifically false display label.
    """

    _ = name
    if dataset is None:
        return None
    try:
        for key, value in _iter_bounded_attrs(dataset):
            normalized = "".join(char for char in str(key).casefold() if char.isalnum())
            if normalized not in {"unit", "units", "unitname", "unitsname"}:
                continue
            value = _to_jsonable(value)
            while isinstance(value, list) and len(value) == 1:
                value = value[0]
            if not isinstance(value, str):
                continue
            text = value.strip()
            if (
                text
                and len(text) <= 128
                and all(ord(char) >= 32 and char != "\x7f" for char in text)
            ):
                return text
    except Exception:  # noqa: BLE001
        return None
    return None


def _spacing_from_geometry(geometry: dict[str, Any] | None) -> dict[str, Any]:
    if geometry and isinstance(geometry.get("spacing"), (list, tuple)) and len(geometry["spacing"]) >= 3:
        sp = geometry["spacing"]
        return {"x": _num(sp[0]), "y": _num(sp[1]), "z": _num(sp[2])}
    return {"x": None, "y": None, "z": None}


def _num(v: Any) -> float | None:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _build_preview_planes(vol: dict[str, Any], slice_axes: list[str], spacing: dict[str, Any]) -> dict[str, Any]:
    pz, py, px = vol["pz"], vol["py"], vol["px"]
    sx = spacing.get("x") or 1.0
    sy = spacing.get("y") or 1.0
    sz = spacing.get("z") or 1.0
    planes: dict[str, Any] = {}
    # z-plane: rows=Y, cols=X
    specs = {
        "z": {"label": "XY (axial)", "axes": ["y", "x"], "w": px, "h": py, "col": sx, "row": sy},
        "y": {"label": "XZ (coronal)", "axes": ["z", "x"], "w": px, "h": pz, "col": sx, "row": sz},
        "x": {"label": "YZ (sagittal)", "axes": ["z", "y"], "w": py, "h": pz, "col": sy, "row": sz},
    }
    for axis in slice_axes:
        s = specs[axis]
        w = int(s["w"]); h = int(s["h"])
        world_w = w * float(s["col"]); world_h = h * float(s["row"])
        planes[axis] = {
            "axis": axis,
            "label": s["label"],
            "axes": s["axes"],
            "pixel_size": {"width": w, "height": h},
            "spacing": {"row": float(s["row"]), "col": float(s["col"])},
            "world_size": {"width": world_w, "height": world_h},
            "aspect_ratio": (world_w / world_h) if world_h else 1.0,
        }
    return planes


def _atlas_scheme(px: int, py: int, pz: int, *, cell_cap: int = ATLAS_CELL_CAP) -> dict[str, Any]:
    w, h, d = max(1, int(px)), max(1, int(py)), max(1, int(pz))
    downsample = max(1, math.ceil(max(w, h) / max(1, cell_cap)))
    cell_w = max(1, round(w / downsample))
    cell_h = max(1, round(h / downsample))
    columns = max(1, math.ceil(math.sqrt(d)))
    rows = max(1, math.ceil(d / columns))
    return {
        "slice_count": d,
        "columns": columns,
        "rows": rows,
        "slice_width": cell_w,
        "slice_height": cell_h,
        "atlas_width": cell_w * columns,
        "atlas_height": cell_h * rows,
        "downsample": downsample,
        "format": "png",
    }


def _sample_values_excerpt(dset: Any, dt: Any, shape: tuple[int, ...]) -> Any:
    """A tiny JSON-safe corner block for the Inspector's ``<pre>`` excerpt."""
    n = SAMPLE_CORNER
    try:
        if _is_compound(dt):
            rows = dset[: min(n, shape[0] if shape else 0)]
            return [_compound_row_to_dict(dt, r) for r in rows]
        if _is_string_dtype(dt):
            if len(shape) == 0:
                return _to_jsonable(dset[()])
            flat = dset[tuple(slice(0, min(n, s)) for s in shape)]
            return _to_jsonable(flat)
        if len(shape) == 0:
            return _to_jsonable(dset[()])
        sl = tuple(slice(0, min(n, s)) for s in shape)
        return _to_jsonable(dset[sl])
    except Exception:  # noqa: BLE001
        return None


def _compound_row_to_dict(dt: Any, row: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for i, fname in enumerate(dt.names):
        try:
            out[str(fname)] = _to_jsonable(row[i])
        except Exception:  # noqa: BLE001
            out[str(fname)] = None
    return out


# ---------------------------------------------------------------------------
# Slice / atlas / scalar-volume (image + binary)
# ---------------------------------------------------------------------------
def _volume_context(path: str, dataset_path: str):
    """Open + resolve + interpret a dataset for a preview render. Raises 422-class
    ``Hdf5Error`` if the dataset is not a renderable volume."""
    h5 = _open(path)
    try:
        dset = _resolve_dataset(h5, dataset_path)
        dt = dset.dtype
        shape = tuple(int(s) for s in dset.shape)
        name = dataset_path.rsplit("/", 1)[-1] or dataset_path
        if not _is_numeric_dtype(dt):
            raise Hdf5Error("unsupported: dataset is not numeric and cannot be rendered as an image")
        vol = _interpret_volume(shape, dt)
        if vol is None:
            raise Hdf5Error("unsupported: dataset shape cannot be rendered as a preview volume")
        shallow = _classify_shallow(name, dt, shape)
        # Refine kind for label detection (small integer sample).
        integer_unique = None
        if _is_integer_dtype(dt):
            s = _strided_sample(dset, vol, 0, min(SAMPLE_MAX_VALUES, 200_000))
            integer_unique = _sample_statistics(s, dt).get("unique_values")
        kind = _refine_preview_kind(shallow, dt, vol, name, integer_unique)
        return h5, dset, dt, vol, kind
    except Exception:
        _safe_close(h5)
        raise


def slice_png(path: str, dataset_path: str, *, axis: str = "z", index: int | None = None,
              component: int = 0) -> bytes:
    """One preview slice as PNG. ``index`` is a preview-grid index (clamped, never 500)."""
    h5, dset, dt, vol, kind = _volume_context(path, dataset_path)
    try:
        axis = axis if axis in ("z", "y", "x") else "z"
        pdim = {"z": vol["pz"], "y": vol["py"], "x": vol["px"]}[axis]
        if index is None:
            index = pdim // 2
        rgb = kind == "rgb_volume"
        plane = _read_preview_plane(dset, vol, axis, int(index), int(component), rgb=rgb)
        if kind == "label_volume":
            img = _label_to_rgb(plane)
        elif kind == "rgb_volume":
            img = _rgb_to_uint8(plane)
        else:  # scalar_volume / vector_volume(component)
            lo, hi = _global_window(path, dset, dataset_path, vol, int(component))
            img = _window_to_uint8(plane, lo, hi)
        return _encode_png(img)
    finally:
        _safe_close(h5)


def atlas_png(path: str, dataset_path: str, **_ignored: Any) -> bytes:
    """The Z-atlas mosaic PNG matching the summary's ``atlas_scheme``. Extra params
    (enhancement/fusion_method/negative/channels) are accepted no-op for contract
    compatibility; the sole call site sends only ``dataset_path``."""
    import numpy as np
    from PIL import Image

    h5, dset, dt, vol, kind = _volume_context(path, dataset_path)
    try:
        pz, py, px = vol["pz"], vol["py"], vol["px"]
        scheme = _atlas_scheme(px, py, pz)
        cols, rows = scheme["columns"], scheme["rows"]
        cell_w, cell_h = scheme["slice_width"], scheme["slice_height"]
        atlas = np.zeros((cell_h * rows, cell_w * cols, 3), dtype="uint8")
        lo, hi = _global_window(path, dset, dataset_path, vol, 0) if kind not in ("label_volume", "rgb_volume") else (0.0, 1.0)
        for z in range(pz):
            plane = _read_preview_plane(dset, vol, "z", z, 0, rgb=(kind == "rgb_volume"))
            if kind == "label_volume":
                cell = _label_to_rgb(plane)
            elif kind == "rgb_volume":
                cell = _rgb_to_uint8(plane)
            else:
                gray = _window_to_uint8(plane, lo, hi)
                cell = np.stack([gray, gray, gray], axis=-1)
            if cell.shape[0] != cell_h or cell.shape[1] != cell_w:
                cell = np.asarray(Image.fromarray(cell).resize((cell_w, cell_h), Image.BILINEAR), dtype="uint8")
            r, c = divmod(z, cols)
            atlas[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w, :] = cell
        buf = io.BytesIO()
        Image.fromarray(atlas).save(buf, format="PNG")
        return buf.getvalue()
    finally:
        _safe_close(h5)


def scalar_volume(path: str, dataset_path: str, *, channel: int = 0) -> dict[str, Any]:
    """Raw voxel buffer + ``x-volume-*`` metadata for the GPU volume renderer. The
    preview grid is downsampled to :data:`PREVIEW_MAX_VOXELS`; ``raw_min``/``raw_max``
    are the sampled data range so windowing works."""
    import numpy as np

    h5, dset, dt, vol, kind = _volume_context(path, dataset_path)
    try:
        arr = _read_preview_volume(dset, vol, int(channel))
        if arr.ndim != 3:
            arr = np.atleast_3d(arr)
        raw_min = float(arr.min()) if arr.size else 0.0
        raw_max = float(arr.max()) if arr.size else 1.0
        if not raw_max > raw_min:
            raw_max = raw_min + 1.0
        return {
            "data": np.ascontiguousarray(arr, dtype="float32").tobytes(),
            "width": int(arr.shape[2]),
            "height": int(arr.shape[1]),
            "depth": int(arr.shape[0]),
            "dtype": "float32",
            "bytes_per_voxel": 4,
            "raw_min": raw_min,
            "raw_max": raw_max,
            "channel": int(channel),
        }
    finally:
        _safe_close(h5)


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------
def dataset_histogram(path: str, dataset_path: str, *, component: int = 0, bins: int = 24,
                      file_id: str = "") -> dict[str, Any]:
    import numpy as np

    bins = max(8, min(int(bins or 24), HISTOGRAM_MAX_BINS))
    h5 = _open(path)
    try:
        dset = _resolve_dataset(h5, dataset_path)
        dt = dset.dtype
        shape = tuple(int(s) for s in dset.shape)
        name = dataset_path.rsplit("/", 1)[-1] or dataset_path
        if not _is_numeric_dtype(dt):
            raise Hdf5Error("unsupported: cannot histogram a non-numeric dataset")
        vol = _interpret_volume(shape, dt) if len(shape) >= 2 else None
        sample = _strided_sample(dset, vol, int(component), SAMPLE_MAX_VALUES)
        sample = sample[np.isfinite(sample)] if sample.size else sample
        sample_count = int(sample.size)
        shallow = _classify_shallow(name, dt, shape)
        component_labels = _component_labels(name, vol["comp"]) if (vol and shallow == "vector_volume") else []
        component_label = component_labels[int(component)] if 0 <= int(component) < len(component_labels) else None

        out_bins: list[dict[str, Any]] = []
        discrete = False
        vmin = None
        vmax = None
        if sample_count == 0:
            pass
        elif _is_integer_dtype(dt):
            uniq, counts = np.unique(sample.astype("int64"), return_counts=True)
            if uniq.size <= max(bins, DISCRETE_MAX_LABELS):
                discrete = True
                vmin = float(uniq.min())
                vmax = float(uniq.max())
                for value, count in zip(uniq.tolist(), counts.tolist()):
                    out_bins.append({"label": str(value), "start": float(value), "end": float(value), "count": int(count)})
            else:
                out_bins, vmin, vmax = _continuous_bins(sample, bins, np)
        else:
            out_bins, vmin, vmax = _continuous_bins(sample, bins, np)

        return {
            "file_id": file_id,
            "dataset_path": dataset_path,
            "preview_kind": shallow,
            "component_index": int(component) if component_labels else None,
            "component_label": component_label,
            "sample_count": sample_count,
            "discrete": bool(discrete),
            "min": vmin,
            "max": vmax,
            "bins": out_bins,
        }
    finally:
        _safe_close(h5)


def _continuous_bins(sample, bins: int, np):
    lo = float(sample.min())
    hi = float(sample.max())
    if not hi > lo:
        hi = lo + 1.0
    counts, edges = np.histogram(sample, bins=bins, range=(lo, hi))
    out: list[dict[str, Any]] = []
    for i in range(len(counts)):
        start = float(edges[i])
        end = float(edges[i + 1])
        mid = (start + end) / 2.0
        out.append({"label": _fmt_num(mid), "start": start, "end": end, "count": int(counts[i])})
    return out, lo, hi


def _fmt_num(v: float) -> str:
    if abs(v) >= 1000 or (v != 0 and abs(v) < 0.01):
        return f"{v:.2e}"
    return f"{v:.3g}"


# ---------------------------------------------------------------------------
# Table preview
# ---------------------------------------------------------------------------
def table_preview(path: str, dataset_path: str, *, offset: int = 0, limit: int = 12,
                  file_id: str = "") -> dict[str, Any]:
    import numpy as np

    offset = max(0, int(offset))
    limit = max(1, int(limit))
    h5 = _open(path)
    try:
        dset = _resolve_dataset(h5, dataset_path)
        dt = dset.dtype
        shape = tuple(int(s) for s in dset.shape)
        name = dataset_path.rsplit("/", 1)[-1] or dataset_path

        if _is_compound(dt):
            columns, rows, total_rows, total_columns, charts = _compound_table(dset, dt, offset, limit, np)
            preview_kind = "table"
        elif _is_string_dtype(dt):
            columns, rows, total_rows, total_columns, charts = _string_table(dset, shape, offset, limit)
            preview_kind = "table"
        elif _is_numeric_dtype(dt) and len(shape) == 1:
            columns, rows, total_rows, total_columns, charts = _series_table(dset, name, offset, limit, np)
            preview_kind = "series"
        elif _is_numeric_dtype(dt) and len(shape) == 2:
            columns, rows, total_rows, total_columns, charts = _matrix_table(dset, shape, offset, limit, np)
            preview_kind = "table"
        else:
            raise Hdf5Error("unsupported: dataset is not tabular")

        return {
            "file_id": file_id,
            "dataset_path": dataset_path,
            "preview_kind": preview_kind,
            # Echo the offset the window actually started at (the FE's pagination
            # math — next offset, "Rows a-b of N", Next-enabled — trusts these
            # echoed values, so a beyond-end request must clamp, not echo raw).
            "offset": min(offset, int(total_rows)),
            "limit": limit,
            "total_rows": int(total_rows),
            "total_columns": int(total_columns),
            "columns": columns,
            "rows": rows,
            "charts": charts,
        }
    finally:
        _safe_close(h5)


def _compound_table(dset, dt, offset, limit, np):
    names = list(dt.names)
    total_rows = int(dset.shape[0]) if dset.shape else 0
    lo = min(offset, total_rows)
    hi = min(lo + limit, total_rows)
    block = dset[lo:hi] if total_rows else np.zeros(0, dtype=dt)
    columns = []
    numeric_fields: list[str] = []
    for fname in names:
        fdt = dt.fields[fname][0]
        numeric = _is_numeric_dtype(fdt)
        if numeric:
            numeric_fields.append(fname)
        columns.append({"key": str(fname), "label": str(fname), "dtype": _dtype_str(fdt), "numeric": bool(numeric)})
    rows = []
    for i, r in enumerate(block):
        row = _compound_row_to_dict(dt, r)
        row["row_index"] = lo + i
        rows.append(row)
    charts = _compound_charts(dset, dt, numeric_fields, total_rows, np)
    return columns, rows, total_rows, len(names), charts


def _compound_charts(dset, dt, numeric_fields, total_rows, np):
    charts: list[dict[str, Any]] = []
    if not numeric_fields or total_rows == 0:
        return charts
    step = max(1, math.ceil(total_rows / TABLE_CHART_MAX_ROWS))
    try:
        block = dset[::step]
    except Exception:  # noqa: BLE001
        return charts
    # Histogram of the first numeric field.
    f0 = numeric_fields[0]
    try:
        vals = np.asarray(block[f0], dtype="float64").ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size:
            counts, edges = np.histogram(vals, bins=min(24, max(8, int(math.sqrt(vals.size)))))
            data = [{"label": _fmt_num((edges[i] + edges[i + 1]) / 2), "count": int(counts[i])} for i in range(len(counts))]
            charts.append({
                "kind": "histogram", "title": f"{f0} distribution", "description": None,
                "x_key": "label", "y_key": "count", "data": data,
            })
    except Exception:  # noqa: BLE001
        pass
    # Scatter of the first two numeric fields.
    if len(numeric_fields) >= 2:
        f1 = numeric_fields[1]
        try:
            xs = np.asarray(block[f0], dtype="float64").ravel()
            ys = np.asarray(block[f1], dtype="float64").ravel()
            n = min(xs.size, ys.size, 1000)
            data = [{f0: _num(xs[i]), "value": _num(ys[i])} for i in range(n)
                    if math.isfinite(xs[i]) and math.isfinite(ys[i])]
            if data:
                charts.append({
                    "kind": "scatter", "title": f"{f0} vs {f1}", "description": None,
                    "x_key": f0, "y_key": "value", "data": data,
                })
        except Exception:  # noqa: BLE001
            pass
    return charts


def _string_table(dset, shape, offset, limit):
    total_rows = int(shape[0]) if shape else 0
    lo = min(offset, total_rows)
    hi = min(lo + limit, total_rows)
    rows = []
    if total_rows:
        block = dset[lo:hi]
        for i, v in enumerate(block):
            text = v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)
            rows.append({"value": text, "row_index": lo + i})
    columns = [{"key": "value", "label": "Value", "dtype": "string", "numeric": False}]
    return columns, rows, total_rows, 1, []


def _series_table(dset, name, offset, limit, np):
    total_rows = int(dset.shape[0])
    lo = min(offset, total_rows)
    hi = min(lo + limit, total_rows)
    block = np.asarray(dset[lo:hi]) if total_rows else np.zeros(0)
    key = "value"
    rows = [{key: _to_jsonable(block[i]), "row_index": lo + i} for i in range(block.shape[0])]
    columns = [{"key": key, "label": name or "Value", "dtype": _dtype_str(dset.dtype), "numeric": True}]
    # Histogram chart over a bounded sample of the whole series.
    charts: list[dict[str, Any]] = []
    step = max(1, math.ceil(total_rows / TABLE_CHART_MAX_ROWS)) if total_rows else 1
    try:
        sample = np.asarray(dset[::step], dtype="float64").ravel()
        sample = sample[np.isfinite(sample)]
        if sample.size:
            counts, edges = np.histogram(sample, bins=min(24, max(8, int(math.sqrt(sample.size)))))
            data = [{"label": _fmt_num((edges[i] + edges[i + 1]) / 2), "count": int(counts[i])} for i in range(len(counts))]
            charts.append({
                "kind": "histogram", "title": f"{name or 'Series'} distribution", "description": None,
                "x_key": "label", "y_key": "count", "data": data,
            })
    except Exception:  # noqa: BLE001
        pass
    return columns, rows, total_rows, 1, charts


def _matrix_table(dset, shape, offset, limit, np):
    total_rows = int(shape[0])
    total_columns = int(shape[1])
    ncols = min(total_columns, 32)  # cap displayed columns
    lo = min(offset, total_rows)
    hi = min(lo + limit, total_rows)
    block = np.asarray(dset[lo:hi, :ncols]) if total_rows else np.zeros((0, ncols))
    columns = [{"key": f"c{j}", "label": f"[{j}]", "dtype": _dtype_str(dset.dtype), "numeric": True} for j in range(ncols)]
    rows = []
    for i in range(block.shape[0]):
        row = {f"c{j}": _to_jsonable(block[i, j]) for j in range(ncols)}
        row["row_index"] = lo + i
        rows.append(row)
    return columns, rows, total_rows, total_columns, []








# ---------------------------------------------------------------------------
# util
# ---------------------------------------------------------------------------
def _safe_close(h5: Any) -> None:
    try:
        h5.close()
    except Exception:  # noqa: BLE001
        pass
