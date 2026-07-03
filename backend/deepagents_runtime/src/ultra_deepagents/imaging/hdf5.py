"""HDF5 data-viewer reader (h5py-backed).

The frontend ships a complete HDF5 viewer (``frontend/src/components/viewer/hdf5``)
whose backend never existed. This module is that backend: it opens ``.h5``/``.hdf5``/
``.hdf``/``.dream3d`` files with :mod:`h5py` and produces exactly the JSON/PNG/binary
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
- :func:`materials_dashboard` — ``Hdf5MaterialsDashboardResponse`` (DREAM.3D).
- :func:`slice_png` / :func:`atlas_png` — ``image/png`` bytes.
- :func:`scalar_volume` — raw voxel dict (``x-volume-*`` header contract).
- :func:`dataset_histogram` — ``Hdf5DatasetHistogramResponse``.
- :func:`table_preview` — ``Hdf5DatasetTablePreviewResponse``.
"""

from __future__ import annotations

import io
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
    "materials_dashboard",
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
    """Requested ``dataset_path`` (or materials schema) is absent — mapped to 404."""


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


def _read_attrs(obj: Any) -> dict[str, Any]:
    """A bounded, JSON-safe view of an h5py object's attributes."""
    out: dict[str, Any] = {}
    try:
        items = list(obj.attrs.items())
    except Exception:  # noqa: BLE001 - a broken attribute block must not fail the walk
        return out
    for key, raw in items[:MAX_ATTRS]:
        try:
            import numpy as np

            if isinstance(raw, np.ndarray) and raw.size > MAX_ATTR_ARRAY:
                raw = raw[:MAX_ATTR_ARRAY]
            out[str(key)] = _to_jsonable(raw)
        except Exception:  # noqa: BLE001
            out[str(key)] = None
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
# Geometry (DREAM.3D _SIMPL_GEOMETRY) + materials probe
# ---------------------------------------------------------------------------
def _find_geometry(h5: Any) -> dict[str, Any] | None:
    """Find a DREAM.3D image geometry (dimensions/spacing/origin) anywhere in the file."""
    found: dict[str, Any] | None = None

    def _visit(name: str, obj: Any) -> None:
        nonlocal found
        if found is not None:
            return
        base = name.rsplit("/", 1)[-1]
        if base != "_SIMPL_GEOMETRY":
            return
        try:
            dims = obj.get("DIMENSIONS")
            spacing = obj.get("SPACING")
            origin = obj.get("ORIGIN")
            found = {
                "path": "/" + name,
                "dimensions": _to_jsonable(dims[()]) if dims is not None else None,
                "spacing": _to_jsonable(spacing[()]) if spacing is not None else None,
                "origin": _to_jsonable(origin[()]) if origin is not None else None,
            }
        except Exception:  # noqa: BLE001
            found = None

    try:
        h5.visititems(_visit)
    except Exception:  # noqa: BLE001
        return None
    return found


def _looks_like_dream3d(h5: Any) -> bool:
    """DREAM.3D heuristic: a ``/DataContainers`` group (or ``/DataStructure`` in
    DREAM3D-NX), plus a version root attribute or an image geometry marker."""
    try:
        keys = set(h5.keys())
    except Exception:  # noqa: BLE001
        return False
    has_container = "DataContainers" in keys or "DataStructure" in keys
    if not has_container:
        return False
    try:
        attrs = {str(k).lower() for k in h5.attrs.keys()}
    except Exception:  # noqa: BLE001
        attrs = set()
    version_attr = any("dream" in a or "fileversion" in a or a == "version" for a in attrs)
    return version_attr or _find_geometry(h5) is not None


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


def _materials_probe(h5: Any) -> dict[str, Any]:
    """Detect DREAM.3D and collect the materials capabilities/roles/phase names +
    renderable cell-data maps and feature/grain tables. Shared by the viewerinfo
    payload and the dashboard. Bounded: walks group structure, reads only small
    string/ensemble arrays eagerly."""
    import numpy as np

    detected = _looks_like_dream3d(h5)
    result: dict[str, Any] = {
        "detected": detected,
        "geometry": None,
        "phase_names": [],
        "roles": {},
        "capabilities": [],
        "feature_count": None,
        "grain_count": None,
        "maps": [],          # list of (path, name, kind, role)
        "feature_groups": [],  # list of group paths holding per-feature 1-D arrays
        "recommended": None,
    }
    if not detected:
        return result
    result["geometry"] = _find_geometry(h5)

    maps: list[tuple[str, str, str, str]] = []
    feature_groups: list[str] = []
    phase_names: list[str] = []

    def _visit(name: str, obj: Any) -> None:
        base = name.rsplit("/", 1)[-1]
        # Phase / material names live in a small 1-D string ensemble array.
        if base in ("PhaseName", "MaterialName", "PhaseNames") and _is_dataset(obj):
            try:
                for v in np.asarray(obj[()]).ravel().tolist():
                    text = v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)
                    text = text.strip()
                    if text and text.lower() not in ("invalid phase", "unknown phase"):
                        phase_names.append(text)
            except Exception:  # noqa: BLE001
                pass
            return
        if not _is_dataset(obj):
            return
        try:
            dt = obj.dtype
            shape = tuple(int(s) for s in obj.shape)
        except Exception:  # noqa: BLE001
            return
        if _is_compound(dt) or _is_string_dtype(dt):
            return
        kind = _classify_shallow(base, dt, shape)
        parent = "/" + name.rsplit("/", 1)[0] if "/" in name else "/"
        low_parent = parent.lower()
        # Renderable spatial maps: volumes under a *CellData* group (the per-voxel
        # spatial arrays). "celldata" is a substring test that deliberately EXCLUDES
        # CellFeatureData/CellEnsembleData (per-feature/per-phase tables), so a
        # (num_features, 3) orientation array never leaks in as a spatial map.
        if kind in _VOLUME_KINDS and ("celldata" in low_parent or "cell_data" in low_parent):
            vol = _interpret_volume(shape, dt)
            if vol is not None and (vol["z"] > 1 or (vol["y"] > 1 and vol["x"] > 1)):
                maps.append(("/" + name, base, kind, _semantic_role(base, kind)))
        # Per-feature 1-D arrays (grain metrics) live under a *FeatureData*-like group.
        if ("featuredata" in low_parent or "feature_data" in low_parent) and len(shape) in (1, 2):
            grp = parent
            if grp not in feature_groups:
                feature_groups.append(grp)

    try:
        h5.visititems(_visit)
    except Exception:  # noqa: BLE001
        pass

    # Feature / grain count = first-dim of a feature-data group (minus the phantom
    # feature 0 that DREAM.3D reserves). Best-effort.
    feature_count = None
    for grp_path in feature_groups:
        try:
            grp = h5[grp_path]
            for k in grp.keys():
                d = grp[k]
                if _is_dataset(d) and len(d.shape) >= 1 and d.shape[0] > 1:
                    feature_count = int(d.shape[0])
                    break
        except Exception:  # noqa: BLE001
            continue
        if feature_count is not None:
            break

    caps: list[str] = ["maps"] if maps else []
    if feature_groups:
        caps.append("grain_metrics")
    if any(role == "euler_angles" or role == "quaternions" for (_p, _n, _k, role) in maps) or feature_groups:
        caps.append("orientation")

    roles = {role: path for (path, _n, _k, role) in maps}

    # Recommended map: IPF colors > feature ids > confidence/quality > first.
    priority = ["ipf_colors", "feature_ids", "confidence_index", "image_quality"]
    recommended = None
    for want in priority:
        for (path, _n, _k, role) in maps:
            if role == want:
                recommended = path
                break
        if recommended is not None:
            break
    if recommended is None and maps:
        recommended = maps[0][0]

    result.update(
        {
            "phase_names": phase_names,
            "roles": roles,
            "capabilities": caps,
            "feature_count": feature_count,
            "grain_count": feature_count,
            "maps": maps,
            "feature_groups": feature_groups,
            "recommended": recommended,
        }
    )
    return result


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
                child_keys = list(obj.keys())
            except Exception:  # noqa: BLE001
                child_keys = []
            children: list[dict[str, Any]] = []
            for key in child_keys[:MAX_GROUP_CHILDREN]:
                if state["nodes"] >= MAX_TREE_NODES:
                    state["truncated"] = True
                    break
                try:
                    child = obj[key]
                except Exception:  # noqa: BLE001
                    continue
                child_path = (path.rstrip("/") + "/" + key) if path != "/" else "/" + key
                built = _node(key, child, child_path, depth + 1)
                if built is not None:
                    children.append(built)
            if len(child_keys) > MAX_GROUP_CHILDREN:
                state["truncated"] = True
            return {
                "path": path,
                "name": name,
                "node_type": "group",
                "child_count": len(child_keys),
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
        root_keys = list(h5.keys())
    except Exception:  # noqa: BLE001
        root_keys = []
    for key in root_keys[:MAX_GROUP_CHILDREN]:
        try:
            child = h5[key]
        except Exception:  # noqa: BLE001
            continue
        built = _node(key, child, "/" + key, 1)
        if built is not None:
            tree.append(built)
    if len(root_keys) > MAX_GROUP_CHILDREN:
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
        root_keys = list(h5.keys())
        root_attributes = _read_attrs(h5)
        tree, dataset_kinds, truncated, group_count, dataset_count, default_path = _walk_tree(h5)
        materials_probe = _materials_probe(h5)

        materials_payload = None
        if materials_probe["detected"]:
            recommended_view = "materials" if materials_probe["maps"] else "explorer"
            materials_payload = {
                "detected": True,
                "schema": "dream3d",
                "capabilities": materials_probe["capabilities"],
                "roles": materials_probe["roles"],
                "phase_names": materials_probe["phase_names"],
                "feature_count": materials_probe["feature_count"],
                "grain_count": materials_probe["grain_count"],
                "recommended_view": recommended_view,
            }
            top["modality"] = "materials"

        # Default selection: for DREAM.3D prefer the recommended map; else the largest
        # renderable dataset from the tree walk.
        default_dataset_path = None
        if materials_payload is not None and materials_probe["recommended"]:
            default_dataset_path = materials_probe["recommended"]
        elif default_path is not None:
            default_dataset_path = default_path

        geometry = materials_probe["geometry"]

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
            "root_keys": [str(k) for k in root_keys],
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
            "materials": materials_payload,
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
        "materials": None,
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
    try:
        obj = h5.get(dataset_path)
    except Exception:  # noqa: BLE001
        obj = None
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
        has_z = bool(vol and vol["pz"] > 1)
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
            "units_hint": _units_hint(name),
            "materials_domain_tags": [],
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


def _units_hint(name: str) -> str | None:
    low = name.strip().lower()
    if "euler" in low:
        return "radians"
    if "diameter" in low or "size" in low:
        return "µm"
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
# Materials dashboard
# ---------------------------------------------------------------------------
def materials_dashboard(path: str, *, file_id: str = "") -> dict[str, Any]:
    import numpy as np

    h5 = _open(path)
    try:
        probe = _materials_probe(h5)
        if not probe["detected"]:
            raise Hdf5DatasetNotFound("materials dashboard is only available for DREAM.3D files")

        geometry = probe["geometry"]
        spacing_note = None
        if geometry and geometry.get("spacing"):
            spacing_note = f"Voxel spacing {geometry['spacing']} (from _SIMPL_GEOMETRY)."

        maps = []
        for (mpath, mname, kind, role) in probe["maps"]:
            maps.append({
                "title": _titleize(mname),
                "description": None,
                "dataset_path": mpath,
                "semantic_role": role,
                "preview_kind": kind,
            })

        dataset_links = []
        for (mpath, mname, kind, role) in probe["maps"]:
            dataset_links.append({
                "label": _titleize(mname),
                "dataset_path": mpath,
                "semantic_role": role,
                "group": mpath.rsplit("/", 1)[0] if "/" in mpath else "/",
            })

        grain_charts = _grain_charts(h5, probe, np)
        orientation_charts = _orientation_charts(h5, probe, np)
        synthetic_stats: list[dict[str, Any]] = []

        overview = {
            "geometry": geometry,
            "spacing_note": spacing_note,
            "phase_names": probe["phase_names"],
            "feature_count": probe["feature_count"],
            "grain_count": probe["grain_count"],
            "capabilities": probe["capabilities"],
            "recommended_map_dataset_path": probe["recommended"],
        }
        return {
            "file_id": file_id,
            "schema": "dream3d",
            "overview": overview,
            "maps": maps,
            "grain_charts": grain_charts,
            "orientation_charts": orientation_charts,
            "synthetic_stats": synthetic_stats,
            "dataset_links": dataset_links,
        }
    finally:
        _safe_close(h5)


def _grain_charts(h5, probe, np) -> list[dict[str, Any]]:
    """Grain-scale distributions from per-feature 1-D arrays (EquivalentDiameters,
    NumNeighbors, ...). Bounded sampling; empty list if nothing suitable."""
    charts: list[dict[str, Any]] = []
    wanted = ("equivalentdiameters", "numneighbors", "numelements", "volumes", "size")
    for grp_path in probe["feature_groups"]:
        try:
            grp = h5[grp_path]
        except Exception:  # noqa: BLE001
            continue
        for key in list(grp.keys()):
            low = key.lower()
            if not any(w in low for w in wanted):
                continue
            try:
                d = grp[key]
                if not _is_dataset(d) or not _is_numeric_dtype(d.dtype):
                    continue
                vals = np.asarray(d[()], dtype="float64").ravel()
                vals = vals[np.isfinite(vals)]
                # Drop the reserved feature-0 sentinel (0-value) that DREAM.3D writes.
                vals = vals[vals != 0] if vals.size > 1 else vals
                if vals.size < 2:
                    continue
                counts, edges = np.histogram(vals, bins=min(24, max(8, int(math.sqrt(vals.size)))))
                data = [{"label": _fmt_num((edges[i] + edges[i + 1]) / 2), "count": int(counts[i])} for i in range(len(counts))]
                charts.append({
                    "kind": "bar",
                    "title": f"{_titleize(key)} distribution",
                    "description": f"{vals.size} features sampled",
                    "x_key": "label",
                    "y_key": "count",
                    "data": data,
                    "source_paths": [grp_path.rstrip("/") + "/" + key],
                    "units_hint": _units_hint(key),
                    "provenance": "Per-feature values, feature 0 (reserved) excluded.",
                })
            except Exception:  # noqa: BLE001
                continue
            if len(charts) >= 4:
                return charts
    return charts


def _orientation_charts(h5, probe, np) -> list[dict[str, Any]]:
    """An orientation scatter (phi1 vs Phi) from a per-feature AvgEulerAngles array,
    if present. Bounded to 2000 points."""
    charts: list[dict[str, Any]] = []
    for grp_path in probe["feature_groups"]:
        try:
            grp = h5[grp_path]
        except Exception:  # noqa: BLE001
            continue
        for key in list(grp.keys()):
            if "euler" not in key.lower():
                continue
            try:
                d = grp[key]
                if not _is_dataset(d) or not _is_numeric_dtype(d.dtype):
                    continue
                arr = np.asarray(d[()], dtype="float64")
                if arr.ndim != 2 or arr.shape[1] < 2:
                    continue
                n = min(arr.shape[0], 2000)
                sel = arr[:n]
                data = [{"phi1": _num(sel[i, 0]), "value": _num(sel[i, 1])} for i in range(n)
                        if math.isfinite(sel[i, 0]) and math.isfinite(sel[i, 1])]
                if data:
                    charts.append({
                        "kind": "scatter",
                        "title": f"{_titleize(key)}: φ1 vs Φ",
                        "description": f"{len(data)} features",
                        "x_key": "phi1",
                        "y_key": "value",
                        "data": data,
                        "source_paths": [grp_path.rstrip("/") + "/" + key],
                        "units_hint": "radians",
                        "provenance": "Per-feature average orientation angles.",
                    })
            except Exception:  # noqa: BLE001
                continue
            if len(charts) >= 2:
                return charts
    return charts


def _titleize(name: str) -> str:
    text = str(name).replace("_", " ").strip()
    return text if text else name


# ---------------------------------------------------------------------------
# util
# ---------------------------------------------------------------------------
def _safe_close(h5: Any) -> None:
    try:
        h5.close()
    except Exception:  # noqa: BLE001
        pass
