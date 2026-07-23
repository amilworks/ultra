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
import itertools
import math
import operator
import os
import re
from typing import Any

__all__ = [
    "HDF5_DATA_EXTS",
    "is_hdf5_data_file",
    "Hdf5Error",
    "Hdf5DatasetNotFound",
    "build_hdf5_viewer_info",
    "dataset_summary",
    "default_dataset_path",
    "materials_dashboard",
    "thumbnail_png",
    "slice_png",
    "atlas_png",
    "scalar_volume",
    "dataset_histogram",
    "table_preview",
]

# --- Extension gate ---------------------------------------------------------
# ONLY these extensions are treated as HDF5-*data* (the tree/dataset explorer, and
# the DREAM.3D materials dashboard when that schema is detected). Every entry is a
# container that is ALWAYS HDF5 and is NEVER a single renderable image, so the
# explorer is the right home for it.
#
# Detection is an extension ALLOWLIST, never HDF5 magic-byte sniffing — precisely so
# HDF5-*based* formats that belong elsewhere keep their routing:
#   * ``.ims`` (Imaris)  -> an IMAGE; libbioimage/bioio decode it.
#   * ``.mat``           -> only v7.3 is HDF5 (older ones are not); not a tree target.
#   * ``.nc`` / ``.nc4`` -> NetCDF: v3 is not HDF5, and NetCDF4 has its own dimensional
#                           conventions a generic tree walk would misrepresent.
# Keep this list in sync with Go ``isHdf5Upload`` (handlers.go) and the frontend
# ``isHdf5ResourceName`` (ResourceBrowser.tsx) — a mismatch means the grid requests a
# thumbnail the sidecar won't render, or skips one it would.
HDF5_DATA_EXTS = (
    # Generic HDF5 containers.
    ".h5", ".hdf5", ".hdf", ".he5",
    # Materials: DREAM.3D reconstruction pipelines + EBSD orientation scans.
    ".dream3d", ".h5ebsd",
    # Single-cell bio: AnnData / Loom matrices.
    ".h5ad", ".loom",
    # NeXus: neutron / X-ray / synchrotron beamline data.
    ".nxs",
)

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
MATERIAL_PHASE_MAX_NAMES = 256  # stored ensemble labels read by the materials probe
MATERIAL_PHASE_MAX_ITEM_BYTES = 1024
MATERIAL_PHASE_MAX_TOTAL_BYTES = 16 * 1024
MATERIAL_ORIENTATION_MAX_ROWS = 2000
FEATURE_ID_SCAN_CHUNK_VALUES = 1_000_000
FEATURE_ID_MAX_TRACKED_IDENTITIES = 1_000_000
MAX_METADATA_ITEM_BYTES = 4096
MAX_METADATA_TOTAL_BYTES = 64 * 1024
FEATURE_FILTER_MAX_IDS = 64
FEATURE_FILTER_MAX_QUERY_BYTES = 1024
FEATURE_ID_UINT32_MAX = (1 << 32) - 1

_PHASE_NAMES_PROVENANCE = (
    "Read from stored DREAM.3D PhaseName/MaterialName metadata; "
    "no phase-identification algorithm was run."
)
_FEATURE_GROUP_NAMES = frozenset(
    {
        "cellfeaturedata",
        "featuredata",
        "grainfeaturedata",
        "graindata",
        "cellgraindata",
    }
)
_ENSEMBLE_GROUP_NAMES = frozenset(
    {
        "cellensembledata",
        "ensembledata",
        "phaseensembledata",
    }
)

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


class Hdf5DatasetNotFound(Hdf5Error):  # noqa: N818 - stable public exception name
    """Requested ``dataset_path`` (or materials schema) is absent — mapped to 404."""


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def is_hdf5_data_file(name: str) -> bool:
    """True when ``name`` has an HDF5-data extension (tolerating a numeric
    series/page suffix like ``vol.h5_2``). ``.ims`` and other HDF5-based image
    formats are excluded. The suffix tolerance is anchored to the END of the
    name — a bare substring match would hijack rasters like ``scan.h5_v2.tif``
    into the h5py path and break their previews."""
    lower = (name or "").strip().lower()
    for ext in HDF5_DATA_EXTS:
        if lower.endswith(ext):
            return True
        if re.search(re.escape(ext) + r"_\d+$", lower):
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


def _direct_hard_child(group: Any, key: str) -> Any | None:
    """Return a direct hard-linked child without following soft/external links."""
    import h5py

    if not _is_group(group):
        return None
    try:
        link = group.get(str(key), getlink=True)
        return group.get(str(key)) if isinstance(link, h5py.HardLink) else None
    except Exception:  # noqa: BLE001
        return None


def _direct_hard_object_at_path(h5: Any, path: str) -> Any | None:
    """Resolve every component of an absolute path through direct hard links only."""
    text = str(path or "")
    if not text.startswith("/") or "\x00" in text:
        return None
    current = h5
    for part in (segment for segment in text.split("/") if segment):
        current = _direct_hard_child(current, part)
        if current is None:
            return None
    return current


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


def _hard_links_scoped_to_parent(obj: Any, parent: Any) -> bool:
    """Fail closed when a hard-linked object has owners outside this parent."""
    try:
        import h5py

        address = _object_address(obj)
        reference_count = int(h5py.h5o.get_info(obj.id).rc)
        if address is None or reference_count <= 0:
            return False
        local_count = sum(
            1
            for key in _bounded_child_names(parent)
            if _object_address(_direct_hard_child(parent, key)) == address
        )
        return local_count == reference_count
    except Exception:  # noqa: BLE001
        return False


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


def _interpret_volume(
    shape: tuple[int, ...],
    dt: Any,
    *,
    exact_zyx: tuple[int, int, int] | None = None,
) -> dict[str, Any] | None:
    """Map a numeric dataset shape onto a (Z, Y, X, C) preview volume + strides.

    Returns ``None`` for shapes that are not a renderable volume (rank 0/1, rank≥5
    without a component interpretation). ``layout`` records how to index the native
    dataset; ``s?`` are per-axis strides that fold the native grid into a preview
    grid (``p?``) bounded by :data:`PREVIEW_MAX_PLANE` / :data:`PREVIEW_MAX_VOXELS`.
    """
    rank = len(shape)
    if rank == 3 and exact_zyx is not None and shape == exact_zyx:
        z, y, x, comp, layout = int(shape[0]), int(shape[1]), int(shape[2]), 1, "zyx"
    elif rank == 4 and exact_zyx is not None and shape[:3] == exact_zyx:
        z, y, x, comp, layout = int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3]), "zyxc"
    elif rank == 2:
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
    base_z_stride = max(1, math.ceil(z / ATLAS_CELL_CAP))
    base_plane_stride = max(1, math.ceil(max(y, x) / PREVIEW_MAX_PLANE))

    def _pdim(n: int, s: int) -> int:
        return (n + s - 1) // s

    def _preview_count(factor: int) -> int:
        return (
            _pdim(z, base_z_stride * factor)
            * _pdim(y, base_plane_stride * factor)
            * _pdim(x, base_plane_stride * factor)
        )

    # Algebraic/binary stride selection is O(log(shape)), including adversarial
    # sparse shapes whose dimensions are far larger than realistic allocations.
    factor = 1
    if _preview_count(factor) > PREVIEW_MAX_VOXELS:
        estimate = max(2, math.ceil((_preview_count(1) / PREVIEW_MAX_VOXELS) ** (1 / 3)))
        upper = estimate
        while _preview_count(upper) > PREVIEW_MAX_VOXELS:
            upper *= 2
        lower = 1
        while lower + 1 < upper:
            middle = (lower + upper) // 2
            if _preview_count(middle) <= PREVIEW_MAX_VOXELS:
                upper = middle
            else:
                lower = middle
        factor = upper
    sz = base_z_stride * factor
    sy = base_plane_stride * factor
    sx = base_plane_stride * factor
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
    """(H, W) grayscale, (H, W, 3) RGB, or (H, W, 4) RGBA → PNG bytes."""
    import numpy as np
    from PIL import Image

    a = np.asarray(arr)
    if 0 in a.shape:
        raise Hdf5Error("cannot encode an empty region")
    if a.dtype != np.uint8:
        a = a.astype("uint8")
    mode = "L" if a.ndim == 2 else ("RGBA" if a.shape[-1] == 4 else "RGB")
    buf = io.BytesIO()
    Image.fromarray(a, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


def _resize_nearest(array: Any, height: int, width: int):
    """Deterministic nearest-neighbor resize shared by filtered color and IDs."""
    import numpy as np

    source = np.asarray(array)
    if source.ndim < 2 or source.shape[0] <= 0 or source.shape[1] <= 0:
        raise Hdf5Error("cannot resize an empty feature preview")
    height, width = max(1, int(height)), max(1, int(width))
    if source.shape[:2] == (height, width):
        return np.ascontiguousarray(source)
    y_index = np.minimum(
        source.shape[0] - 1,
        np.floor((np.arange(height, dtype="float64") + 0.5) * source.shape[0] / height),
    ).astype("int64")
    x_index = np.minimum(
        source.shape[1] - 1,
        np.floor((np.arange(width, dtype="float64") + 0.5) * source.shape[1] / width),
    ).astype("int64")
    return np.ascontiguousarray(source[y_index[:, None], x_index[None, :]])


def _feature_mask_rgba(image: Any, identities: Any, selected: tuple[int, ...]):
    """Apply global raw-ID equality after target normalization."""
    import numpy as np

    rgb = np.asarray(image, dtype="uint8")
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    rgb = rgb[..., :3]
    ids = np.asarray(identities)
    if ids.shape != rgb.shape[:2]:
        raise Hdf5Error("unsupported: FeatureIds and target delivery grids do not match")
    mask = np.isin(ids, np.asarray(selected, dtype="uint64"))
    rgba = np.zeros((*rgb.shape[:2], 4), dtype="uint8")
    rgba[mask, :3] = rgb[mask]
    rgba[mask, 3] = 255
    return rgba


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
    if isinstance(comp, bool):
        raise Hdf5Error("scalar volume channel must be an integer")
    try:
        comp = operator.index(comp)
    except TypeError as exc:
        raise Hdf5Error("scalar volume channel must be an integer") from exc
    if comp < 0 or comp >= int(vol["comp"]):
        raise Hdf5Error(
            f"scalar volume channel {comp} is out of range for C={int(vol['comp'])}"
        )
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


def _atlas_component_index(value: Any, vol: dict[str, Any], kind: str) -> int:
    """Validate the scalar component represented by an atlas.

    Vector atlases expose one selected component. RGB/IPF atlases are already a
    complete color value and every other atlas kind is scalar, so a non-zero
    component for those kinds would claim to change data while doing nothing.
    """
    if isinstance(value, bool):
        raise Hdf5Error("unsupported: atlas component must be an integer")
    try:
        component = operator.index(value)
    except TypeError as exc:
        raise Hdf5Error("unsupported: atlas component must be an integer") from exc
    component_count = int(vol["comp"])
    if component < 0 or component >= component_count:
        raise Hdf5Error(
            f"unsupported: atlas component {component} is out of range for C={component_count}"
        )
    if kind != "vector_volume" and component != 0:
        raise Hdf5Error(f"unsupported: atlas component selection does not apply to {kind}")
    return int(component)


# ---------------------------------------------------------------------------
# Geometry (DREAM.3D _SIMPL_GEOMETRY) + materials probe
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


def _normalized_group_name(path: str) -> str:
    """Normalize only the final group segment for DREAM.3D naming variants."""
    segment = str(path).rstrip("/").rsplit("/", 1)[-1]
    return "".join(char for char in segment.casefold() if char.isalnum())


def _is_feature_group_path(path: str) -> bool:
    return _normalized_group_name(path) in _FEATURE_GROUP_NAMES


def _is_cell_data_group_path(path: str) -> bool:
    return _normalized_group_name(path) == "celldata"


def _is_ensemble_group_path(path: str) -> bool:
    return _normalized_group_name(path) in _ENSEMBLE_GROUP_NAMES


def _is_orientation_array(name: str, dset: Any, shape: tuple[int, ...]) -> bool:
    """Require a typed multi-component Euler/quaternion array, not a group hint."""
    if not _is_dataset(dset) or len(shape) < 2:
        return False
    itemsize = int(getattr(dset.dtype, "itemsize", 0) or 0)
    if getattr(dset.dtype, "kind", "") not in {"i", "u", "f"} or not (
        0 < itemsize <= 8
    ):
        return False
    normalized = "".join(char for char in str(name).casefold() if char.isalnum())
    components = int(shape[-1]) if shape else 0
    return ("euler" in normalized and components >= 3) or (
        "quat" in normalized and components >= 4
    )


def _read_bounded_phase_names(
    dset: Any,
    *,
    max_names: int,
    max_bytes: int,
) -> tuple[list[str], int, int]:
    """Read stored string labels one scalar at a time within count/byte caps."""
    import h5py
    import numpy as np

    if not _is_dataset(dset) or max_names <= 0 or max_bytes <= 0:
        return [], 0, 0
    try:
        dtype = dset.dtype
        string_info = h5py.check_string_dtype(dtype)
        if string_info is None and getattr(dtype, "kind", "") not in {"S", "U"}:
            return [], 0, 0
        itemsize = int(getattr(dtype, "itemsize", 0) or 0)
        # Fixed-width strings have a trustworthy pre-read bound. Vlen strings have
        # an object descriptor and are read one scalar at a time, then byte-checked.
        if getattr(dtype, "kind", "") != "O" and not (
            0 < itemsize <= MATERIAL_PHASE_MAX_ITEM_BYTES
        ):
            return [], 0, 0
        shape = tuple(int(value) for value in dset.shape)
        total_items = int(np.prod(shape)) if shape else 1
        limit = min(total_items, max_names, MATERIAL_PHASE_MAX_NAMES)
        indices = (iter([()]) if not shape else itertools.islice(np.ndindex(shape), limit))
        names: list[str] = []
        decoded_bytes = 0
        items_read = 0
        for index in indices:
            raw = dset[index]
            items_read += 1
            if isinstance(raw, np.ndarray):
                if raw.size != 1:
                    continue
                raw = raw.reshape(-1)[0]
            if isinstance(raw, np.generic):
                raw = raw.item()
            if isinstance(raw, bytes):
                raw_bytes = raw.rstrip(b"\x00")
                if len(raw_bytes) > MATERIAL_PHASE_MAX_ITEM_BYTES:
                    continue
                text = raw_bytes.decode("utf-8", "replace")
            elif isinstance(raw, str):
                raw_bytes = raw.encode("utf-8", "replace")
                if len(raw_bytes) > MATERIAL_PHASE_MAX_ITEM_BYTES:
                    continue
                text = raw
            else:
                continue
            text = text.strip()
            item_bytes = len(text.encode("utf-8", "replace"))
            if item_bytes > MATERIAL_PHASE_MAX_ITEM_BYTES:
                continue
            if decoded_bytes + item_bytes > max_bytes:
                break
            decoded_bytes += item_bytes
            names.append(text)
        return names, decoded_bytes, items_read
    except Exception:  # noqa: BLE001
        return [], 0, 0


def _feature_id_dataset(h5: Any, geometry: dict[str, Any] | None) -> Any | None:
    if not isinstance(geometry, dict):
        return None
    cell_data_path = geometry.get("cell_data_path")
    dimensions = geometry.get("dimensions")
    if not isinstance(cell_data_path, str) or not isinstance(dimensions, list):
        return None
    cell_group = _hard_object_at_path(h5, cell_data_path)
    if not _is_group(cell_group):
        return None
    candidates: list[tuple[int, Any]] = []
    for key in _bounded_child_names(cell_group):
        normalized = "".join(char for char in str(key).casefold() if char.isalnum())
        if normalized not in {"featureids", "grainids"}:
            continue
        dset = _hard_child(cell_group, key)
        if not _is_dataset(dset):
            continue
        try:
            shape = tuple(int(value) for value in dset.shape)
            itemsize = int(getattr(dset.dtype, "itemsize", 0) or 0)
            if (
                getattr(dset.dtype, "kind", "") in {"i", "u"}
                and 0 < itemsize <= 8
                and _cell_dataset_matches_geometry(shape, dimensions)
            ):
                candidates.append((0 if normalized == "featureids" else 1, dset))
        except Exception:  # noqa: BLE001
            continue
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def _bounded_hyperslabs(shape: tuple[int, ...], max_values: int):
    """Yield selections whose element count never exceeds ``max_values``."""
    if not shape or any(dimension <= 0 for dimension in shape):
        return
    remaining = max(1, int(max_values))
    block_shape = [1] * len(shape)
    for index in range(len(shape) - 1, -1, -1):
        take = min(int(shape[index]), remaining)
        block_shape[index] = max(1, take)
        remaining = max(1, remaining // block_shape[index])
    starts = [range(0, shape[index], block_shape[index]) for index in range(len(shape))]
    for origin in itertools.product(*starts):
        yield tuple(
            slice(start, min(start + block_shape[index], shape[index]))
            for index, start in enumerate(origin)
        )


def _scan_feature_ids(dset: Any) -> dict[str, Any]:
    """Completely scan a valid cell FeatureIds array in bounded hyperslabs.

    The set of distinct identities is itself capped. Crossing that cap makes the
    relationship unverified (``complete=False``) rather than silently sampling.
    """
    import numpy as np

    empty = {
        "complete": False,
        "positive_count": None,
        "positive_min": None,
        "positive_max": None,
        "has_negative": None,
    }
    if not _is_dataset(dset):
        return empty
    try:
        shape = tuple(int(value) for value in dset.shape)
        itemsize = int(getattr(dset.dtype, "itemsize", 0) or 0)
        if (
            not shape
            or getattr(dset.dtype, "kind", "") not in {"i", "u"}
            or not (0 < itemsize <= 8)
        ):
            return empty
        positive_ids: set[int] = set()
        has_negative = False
        for selection in _bounded_hyperslabs(shape, FEATURE_ID_SCAN_CHUNK_VALUES):
            block = np.asarray(dset[selection])
            if block.size > FEATURE_ID_SCAN_CHUNK_VALUES:
                return empty
            unique = np.unique(block)
            if getattr(dset.dtype, "kind", "") == "i" and unique.size:
                has_negative = has_negative or bool(unique[0] < 0)
            for value in unique:
                identity = int(value)
                if identity > 0:
                    positive_ids.add(identity)
                    if len(positive_ids) > FEATURE_ID_MAX_TRACKED_IDENTITIES:
                        return empty
        return {
            "complete": True,
            "positive_count": len(positive_ids),
            "positive_min": min(positive_ids) if positive_ids else None,
            "positive_max": max(positive_ids) if positive_ids else None,
            "has_negative": has_negative,
        }
    except Exception:  # noqa: BLE001
        return empty


def _feature_group_declaration(
    h5: Any,
    *,
    group_path: str,
    geometry_container: str | None,
    geometry_dimensions: list[Any] | None,
) -> dict[str, Any] | None:
    group = _hard_object_at_path(h5, group_path)
    if not _is_group(group):
        return None
    row_length_frequency: dict[int, int] = {}
    orientation = False
    for key in _bounded_child_names(group):
        dset = _hard_child(group, key)
        if not _is_dataset(dset):
            continue
        try:
            shape = tuple(int(value) for value in dset.shape)
            if len(shape) >= 1 and int(shape[0]) >= 1:
                row_count = int(shape[0])
                row_length_frequency[row_count] = row_length_frequency.get(row_count, 0) + 1
            orientation = orientation or _is_orientation_array(str(key), dset, shape)
        except Exception:  # noqa: BLE001
            continue
    if not row_length_frequency:
        return None
    stored_rows = max(
        row_length_frequency,
        key=lambda rows: (row_length_frequency[rows], -rows),
    )
    reserved_zero = _feature_group_has_reserved_zero(
        h5,
        group=group,
        geometry_container=geometry_container,
        geometry_dimensions=geometry_dimensions,
        stored_rows=stored_rows,
    )
    return {
        "path": group_path,
        "stored_rows": stored_rows,
        "declared_count": max(0, stored_rows - int(reserved_zero)),
        "modal_support": row_length_frequency[stored_rows],
        "reserved_zero": reserved_zero,
        "orientation": orientation,
    }


_RESERVED_ZERO_ATTRIBUTE_NAMES = {
    "haszerotuple",
    "reservedfeaturezero",
    "reservedtuplezero",
    "tuple0isreserved",
    "tuplezeroisreserved",
    "zerotupleisreserved",
}


def _attribute_bool(value: Any) -> bool | None:
    try:
        if hasattr(value, "tolist"):
            value = value.tolist()
        while isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        if isinstance(value, bytes):
            value = value.decode("utf-8", "replace")
        if isinstance(value, str):
            normalized = value.strip().casefold()
            if normalized in {"1", "true", "yes", "reserved"}:
                return True
            if normalized in {"0", "false", "no", "not reserved"}:
                return False
            return None
        if isinstance(value, (bool, int)):
            return bool(value)
    except Exception:  # noqa: BLE001
        return None
    return None


def _feature_group_has_reserved_zero(
    h5: Any,
    *,
    group: Any,
    geometry_container: str | None,
    geometry_dimensions: list[Any] | None,
    stored_rows: int,
) -> bool:
    """Require file/schema evidence before dropping per-feature tuple zero.

    An explicit group attribute is authoritative. Otherwise, a typed
    ``FeatureIds``/``GrainIds`` cell array in the selected DREAM.3D container
    establishes the schema relationship in which cell label zero refers to the
    reserved feature tuple. A group name alone is not sufficient evidence.
    """

    if stored_rows <= 0:
        return False
    try:
        for key, value in _iter_bounded_attrs(group):
            normalized = "".join(char for char in str(key).casefold() if char.isalnum())
            if normalized in _RESERVED_ZERO_ATTRIBUTE_NAMES:
                declared = _attribute_bool(value)
                if declared is not None:
                    return declared
    except Exception:  # noqa: BLE001
        pass
    if not geometry_container:
        return False
    for cell_name in ("CellData", "Cell Data", "Cell_Data"):
        cell_group = _hard_object_at_path(
            h5, geometry_container.rstrip("/") + "/" + cell_name
        )
        if not _is_group(cell_group):
            continue
        for child_key in _bounded_child_names(cell_group):
            normalized = "".join(
                char for char in str(child_key).casefold() if char.isalnum()
            )
            if normalized not in {"featureids", "grainids"}:
                continue
            try:
                dataset = _hard_child(cell_group, child_key)
                if dataset is None:
                    continue
                shape = tuple(int(value) for value in dataset.shape)
                if (
                    _is_dataset(dataset)
                    and getattr(dataset.dtype, "kind", "") in {"i", "u"}
                    and 0 < int(getattr(dataset.dtype, "itemsize", 0) or 0) <= 8
                    and isinstance(geometry_dimensions, list)
                    and _cell_dataset_matches_geometry(shape, geometry_dimensions)
                ):
                    return True
            except Exception:  # noqa: BLE001
                continue
    return False


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


def _looks_like_dream3d(h5: Any) -> bool:
    """DREAM.3D heuristic: a ``/DataContainers`` group (or ``/DataStructure`` in
    DREAM3D-NX), plus a version root attribute or an image geometry marker."""
    has_container = any(
        _is_group(_hard_child(h5, key)) for key in ("DataContainers", "DataStructure")
    )
    if not has_container:
        return False
    try:
        attrs = {str(k).lower() for k in itertools.islice(h5.attrs.keys(), MAX_ATTRS)}
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


def _feature_filter_ids(raw: str | None) -> tuple[int, ...] | None:
    """Parse the public comma-list contract into sorted, unique positive uint32 IDs."""
    if raw is None:
        return None
    text = str(raw)
    if (
        not text
        or len(text.encode("utf-8")) > FEATURE_FILTER_MAX_QUERY_BYTES
        or re.fullmatch(r"[0-9]+(?:,[0-9]+)*", text) is None
    ):
        raise Hdf5Error(
            "unsupported: feature_ids must be a comma-separated list of positive uint32 values"
        )
    values: set[int] = set()
    for token in text.split(","):
        value = int(token, 10)
        if value <= 0 or value > FEATURE_ID_UINT32_MAX:
            raise Hdf5Error("unsupported: feature_ids must contain only positive uint32 values")
        values.add(value)
    if len(values) > FEATURE_FILTER_MAX_IDS:
        raise Hdf5Error(
            f"unsupported: feature_ids supports at most {FEATURE_FILTER_MAX_IDS} unique values"
        )
    return tuple(sorted(values))


def _native_zyx_shape(dset: Any) -> tuple[int, int, int] | None:
    try:
        shape = tuple(int(value) for value in dset.shape)
    except Exception:  # noqa: BLE001
        return None
    if len(shape) == 3:
        return shape
    if len(shape) == 4 and shape[-1] >= 1:
        return shape[:3]
    return None


def _dataset_local_geometry(
    h5: Any, dataset_path: str, *, target: Any | None = None
) -> dict[str, Any] | None:
    """Resolve geometry owned by the selected direct-hard-linked CellData dataset."""
    text = str(dataset_path or "").strip()
    if not text.startswith("/") or "\x00" in text:
        return None
    parts = text.split("/")[1:]
    if len(parts) < 2 or any(part in {"", ".", ".."} for part in parts):
        return None
    target_name = parts[-1]
    cell_data_name = parts[-2]
    if _normalized_group_name(cell_data_name) != "celldata":
        return None
    container_path = "/" + "/".join(parts[:-2])
    container = _direct_hard_object_at_path(h5, container_path)
    cell_group = _direct_hard_child(container, cell_data_name)
    geometry_group = _direct_hard_child(container, "_SIMPL_GEOMETRY")
    direct_target = _direct_hard_child(cell_group, target_name)
    if not (
        _is_group(container)
        and _is_group(cell_group)
        and _is_group(geometry_group)
        and _is_dataset(direct_target)
        and _hard_links_scoped_to_parent(cell_group, container)
        and _hard_links_scoped_to_parent(geometry_group, container)
        and _hard_links_scoped_to_parent(direct_target, cell_group)
    ):
        return None
    direct_address = _object_address(direct_target)
    target_address = _object_address(target) if target is not None else direct_address
    if direct_address is None or target_address is None or direct_address != target_address:
        return None

    dimensions_dataset = _direct_hard_child(geometry_group, "DIMENSIONS")
    spacing_dataset = _direct_hard_child(geometry_group, "SPACING")
    origin_dataset = _direct_hard_child(geometry_group, "ORIGIN")
    if not all(
        _is_dataset(dataset)
        and _hard_links_scoped_to_parent(dataset, geometry_group)
        for dataset in (dimensions_dataset, spacing_dataset, origin_dataset)
    ):
        return None
    dimensions = _bounded_geometry_vector(dimensions_dataset)
    spacing = _bounded_geometry_vector(spacing_dataset)
    origin = _bounded_geometry_vector(origin_dataset)
    if not (
        _valid_geometry_vector(dimensions, positive=True)
        and _valid_geometry_vector(spacing, positive=True)
        and _valid_geometry_vector(origin, positive=False)
    ):
        return None
    try:
        xyz = tuple(int(value) for value in dimensions[:3])
        if any(float(dimensions[index]) != xyz[index] for index in range(3)):
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    if _native_zyx_shape(direct_target) != tuple(reversed(xyz)):
        return None

    return {
        "path": str(getattr(geometry_group, "name", "") or ""),
        "dimensions": dimensions,
        "spacing": spacing,
        "origin": origin,
        "cell_data_path": str(getattr(cell_group, "name", "") or ""),
        "cell_data_consistent": True,
        "complete": True,
    }


def _resolve_feature_registration(
    h5: Any,
    dataset_path: str,
    *,
    target: Any | None = None,
    target_kind: str | None = None,
) -> dict[str, Any]:
    """Resolve an exact DREAM.3D CellData target↔FeatureIds registration.

    This deliberately accepts only direct hard-linked siblings under the selected,
    complete image geometry.  Names or matching dimensions alone are insufficient:
    filtering the wrong array would look convincing while corrupting the science.
    """
    normalized_path = "/" + str(dataset_path or "").strip().strip("/")
    cell_data_path, _, target_name = normalized_path.rpartition("/")
    cell_data_path = cell_data_path or "/"
    if not target_name or not _is_cell_data_group_path(cell_data_path):
        raise Hdf5Error("unsupported: feature filtering requires a selected DREAM.3D CellData map")
    geometry = _dataset_local_geometry(h5, normalized_path, target=target)
    if geometry is None:
        raise Hdf5Error(
            "unsupported: feature filtering requires exact direct hard-linked ownership"
        )
    cell_group = _direct_hard_object_at_path(h5, cell_data_path)
    if not _is_group(cell_group):
        raise Hdf5Error("unsupported: DREAM.3D CellData must be direct hard linked")
    dimensions = geometry["dimensions"]
    try:
        xyz = tuple(int(value) for value in dimensions[:3])
        if any(float(dimensions[index]) != xyz[index] or xyz[index] <= 0 for index in range(3)):
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise Hdf5Error("unsupported: invalid DREAM.3D geometry dimensions") from exc
    expected_zyx = (xyz[2], xyz[1], xyz[0])

    direct_target = _direct_hard_child(cell_group, target_name)
    if not _is_dataset(direct_target):
        raise Hdf5Error("unsupported: selected CellData map must be direct hard linked")
    if target is not None and _object_address(direct_target) != _object_address(target):
        raise Hdf5Error("unsupported: selected CellData identity is ambiguous")
    target = direct_target

    target_shape = _native_zyx_shape(target)
    if target_shape != expected_zyx:
        raise Hdf5Error("unsupported: selected CellData map does not match native geometry ZYX")
    target_volume = _interpret_volume(
        tuple(int(v) for v in target.shape),
        target.dtype,
        exact_zyx=expected_zyx,
    )
    if target_volume is None or target_volume.get("layout") not in {"zyx", "zyxc"}:
        raise Hdf5Error("unsupported: selected CellData map has no exact ZYX volume layout")
    target_kind = target_kind or _classify_shallow(target_name, target.dtype, tuple(target.shape)) or ""
    target_role = _semantic_role(target_name, target_kind)
    dtype_kind = getattr(target.dtype, "kind", "")
    components = int(target_volume.get("comp", 0))
    role_is_exact = (
        (target_role == "feature_ids" and dtype_kind in {"i", "u"} and components == 1)
        or (target_role == "euler_angles" and dtype_kind == "f" and components == 3)
        or (target_role == "ipf_colors" and dtype_kind in {"i", "u"} and components in {3, 4})
    )
    if not role_is_exact:
        raise Hdf5Error("unsupported: selected CellData map is not FeatureIds, Euler angles, or IPF colors")

    feature_ids_present = "FeatureIds" in cell_group
    grain_ids_present = "GrainIds" in cell_group
    feature_ids = _direct_hard_child(cell_group, "FeatureIds") if feature_ids_present else None
    grain_ids = _direct_hard_child(cell_group, "GrainIds") if grain_ids_present else None
    if feature_ids_present and not _is_dataset(feature_ids):
        raise Hdf5Error("unsupported: FeatureIds must be a direct hard-linked dataset")
    if feature_ids_present and grain_ids_present:
        if not _is_dataset(grain_ids) or _object_address(feature_ids) != _object_address(grain_ids):
            raise Hdf5Error("unsupported: FeatureIds and GrainIds identities are ambiguous")
    identity_name = "FeatureIds" if feature_ids_present else "GrainIds"
    identity = feature_ids if feature_ids_present else grain_ids
    if not _is_dataset(identity):
        raise Hdf5Error("unsupported: no direct hard-linked FeatureIds identity volume")
    if not _hard_links_scoped_to_parent(identity, cell_group):
        raise Hdf5Error("unsupported: FeatureIds hard-linked ownership is ambiguous")
    identity_shape = tuple(int(value) for value in identity.shape)
    identity_zyx = _native_zyx_shape(identity)
    identity_volume = _interpret_volume(identity_shape, identity.dtype, exact_zyx=expected_zyx)
    if (
        getattr(identity.dtype, "kind", "") not in {"i", "u"}
        or identity_zyx != expected_zyx
        or identity_volume is None
        or identity_volume.get("layout") not in {"zyx", "zyxc"}
        or int(identity_volume.get("comp", 0)) != 1
    ):
        raise Hdf5Error("unsupported: FeatureIds must be a co-registered integer ZYX volume")
    stride = (target_volume["sz"], target_volume["sy"], target_volume["sx"])
    if stride != (identity_volume["sz"], identity_volume["sy"], identity_volume["sx"]):
        raise Hdf5Error("unsupported: target and FeatureIds preview grids do not match")

    source_path = f"{cell_data_path.rstrip('/')}/{identity_name}"
    return {
        "target": target,
        "target_volume": target_volume,
        "target_role": target_role,
        "identity": identity,
        "identity_volume": identity_volume,
        "source_dataset_path": source_path,
        "native_shape": expected_zyx,
        "preview_shape": (
            int(target_volume["pz"]),
            int(target_volume["py"]),
            int(target_volume["px"]),
        ),
        "preview_stride": stride,
    }


def _feature_filter_metadata(registration: dict[str, Any]) -> dict[str, Any]:
    native = registration["native_shape"]
    preview = registration["preview_shape"]
    stride = registration["preview_stride"]
    source = registration["source_dataset_path"]
    return {
        "supported": True,
        "source_dataset_path": source,
        "max_ids": FEATURE_FILTER_MAX_IDS,
        "background_id": 0,
        "provenance": "co_registered_raw_integer_feature_ids",
        "registration_key": (
            f"{source}|{native[0]}x{native[1]}x{native[2]}|"
            f"{stride[0]}x{stride[1]}x{stride[2]}"
        ),
        "target_role": registration["target_role"],
        "native_shape": list(native),
        "preview_shape": list(preview),
        "preview_stride": {"z": stride[0], "y": stride[1], "x": stride[2]},
    }


def _materials_probe(h5: Any) -> dict[str, Any]:
    """Detect DREAM.3D and collect the materials capabilities/roles/phase names +
    renderable cell-data maps and relationally validated feature/grain tables.

    Traversal is hard-link-only. Phase labels must be typed string datasets in an
    ensemble group belonging to the selected geometry container. Grain count is not
    inferred from table rows: it is published only after a complete cell FeatureIds
    scan proves that declared tuples and referenced positive identities agree.
    """

    detected = _looks_like_dream3d(h5)
    result: dict[str, Any] = {
        "detected": detected,
        "geometry": None,
        "phase_names": [],
        "phase_names_source": None,
        "phase_names_provenance": None,
        "roles": {},
        "capabilities": [],
        "feature_count": None,
        "grain_count": None,
        "declared_feature_tuple_count": None,
        "referenced_positive_feature_count": None,
        "feature_id_scan_complete": False,
        "feature_id_consistency": None,
        "maps": [],          # list of (path, name, kind, role)
        "feature_groups": [],  # selected relationship-backed feature group
        "feature_group_reserved_zero": {},
        "feature_zero_reserved": None,
        "recommended": None,
    }
    if not detected:
        return result
    result["geometry"] = _find_geometry(h5)
    geometry = result["geometry"]
    geometry_container = None
    if isinstance(geometry, dict) and isinstance(geometry.get("path"), str):
        geometry_container = geometry["path"].rsplit("/", 1)[0]

    maps: list[tuple[str, str, str, str]] = []
    feature_group_paths: list[str] = []
    phase_names: list[str] = []
    phase_names_seen: set[str] = set()
    phase_bytes = 0
    phase_items_read = 0
    map_orientation = False

    def _visit(name: str, obj: Any) -> bool | None:
        nonlocal phase_bytes, phase_items_read, map_orientation
        base = name.rsplit("/", 1)[-1]
        dataset_path = "/" + name
        parent = "/" + name.rsplit("/", 1)[0] if "/" in name else "/"
        in_selected_container = bool(
            geometry_container
            and dataset_path.startswith(geometry_container.rstrip("/") + "/")
        )

        # Phase/material labels are accepted only from a direct, typed ensemble
        # group inside the geometry container selected by _find_geometry.
        if base in ("PhaseName", "MaterialName", "PhaseNames") and _is_dataset(obj):
            ensemble_parent = parent.rsplit("/", 1)[0] if "/" in parent.rstrip("/") else "/"
            if not (
                in_selected_container
                and _is_ensemble_group_path(parent)
                and ensemble_parent == geometry_container
            ):
                return None
            remaining_names = MATERIAL_PHASE_MAX_NAMES - phase_items_read
            remaining_bytes = MATERIAL_PHASE_MAX_TOTAL_BYTES - phase_bytes
            stored_names, used_bytes, items_read = _read_bounded_phase_names(
                obj, max_names=remaining_names, max_bytes=remaining_bytes
            )
            phase_bytes += used_bytes
            phase_items_read += items_read
            for text in stored_names:
                normalized = text.casefold()
                if (
                    text
                    and normalized not in ("invalid phase", "unknown phase")
                    and normalized not in phase_names_seen
                ):
                    phase_names_seen.add(normalized)
                    phase_names.append(text)
            return None
        if not _is_dataset(obj):
            return None
        try:
            dt = obj.dtype
            shape = tuple(int(s) for s in obj.shape)
        except Exception:  # noqa: BLE001
            return None
        if _is_compound(dt) or _is_string_dtype(dt):
            return None
        kind = _classify_shallow(base, dt, shape)
        # Renderable spatial maps: volumes under a *CellData* group (the per-voxel
        # spatial arrays). Match the final normalized segment so similarly named
        # per-feature/per-phase groups never leak into the map list.
        if in_selected_container and kind in _VOLUME_KINDS and _is_cell_data_group_path(parent):
            vol = _interpret_volume(shape, dt)
            if vol is not None and (vol["z"] > 1 or (vol["y"] > 1 and vol["x"] > 1)):
                role = _semantic_role(base, kind)
                maps.append((dataset_path, base, kind, role))
                map_orientation = map_orientation or _is_orientation_array(base, obj, shape)
        # Per-feature arrays may use classic CellFeatureData or the real pipeline's
        # ``Grain Data`` spelling. Keep this to known final group names.
        if in_selected_container and _is_feature_group_path(parent) and len(shape) in (1, 2):
            grp = parent
            if grp not in feature_group_paths:
                feature_group_paths.append(grp)
        return None

    try:
        _visit_hard_objects(h5, _visit)
    except Exception:  # noqa: BLE001
        pass

    geometry_dimensions = geometry.get("dimensions") if isinstance(geometry, dict) else None
    declarations = [
        declaration
        for group_path in feature_group_paths
        if (
            declaration := _feature_group_declaration(
                h5,
                group_path=group_path,
                geometry_container=geometry_container,
                geometry_dimensions=geometry_dimensions,
            )
        )
        is not None
    ]
    feature_id_dset = _feature_id_dataset(h5, geometry)
    feature_id_scan = _scan_feature_ids(feature_id_dset)
    referenced_count = feature_id_scan["positive_count"]

    for declaration in declarations:
        consistency: bool | None = None
        declared_count = declaration["declared_count"]
        if feature_id_scan["complete"] and referenced_count is not None:
            consistency = bool(
                not feature_id_scan["has_negative"]
                and referenced_count == declared_count
                and (
                    (declared_count == 0 and feature_id_scan["positive_min"] is None)
                    or (
                        declared_count > 0
                        and feature_id_scan["positive_min"] == 1
                        and feature_id_scan["positive_max"] == declared_count
                    )
                )
            )
        declaration["consistency"] = consistency

    selected_declaration = None
    if declarations:
        def _relationship_score(item: tuple[int, dict[str, Any]]) -> tuple[int, int, int, int]:
            index, declaration = item
            consistency = declaration["consistency"]
            relationship_rank = 2 if consistency is True else (1 if consistency is False else 0)
            difference = (
                abs(declaration["declared_count"] - referenced_count)
                if referenced_count is not None
                else 0
            )
            # Relationship dominates; modal support resolves unverified ties. The
            # final tie preserves bounded traversal order rather than filename sort.
            return (
                relationship_rank,
                -difference,
                declaration["modal_support"],
                -index,
            )

        selected_declaration = max(enumerate(declarations), key=_relationship_score)[1]

    feature_groups: list[str] = []
    feature_group_reserved_zero: dict[str, bool] = {}
    feature_zero_reserved: bool | None = None
    declared_feature_tuple_count = None
    feature_id_consistency = None
    feature_count = None
    grain_count = None
    selected_orientation = False
    if selected_declaration is not None:
        selected_path = selected_declaration["path"]
        feature_groups = [selected_path]
        feature_zero_reserved = selected_declaration["reserved_zero"]
        feature_group_reserved_zero[selected_path] = feature_zero_reserved
        declared_feature_tuple_count = selected_declaration["declared_count"]
        feature_count = declared_feature_tuple_count
        feature_id_consistency = selected_declaration["consistency"]
        selected_orientation = selected_declaration["orientation"]
        if feature_id_consistency is True:
            grain_count = referenced_count

    caps: list[str] = ["maps"] if maps else []
    if feature_groups:
        caps.append("grain_metrics")
    if map_orientation or selected_orientation:
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
            "phase_names_source": "stored_metadata" if phase_names else None,
            "phase_names_provenance": _PHASE_NAMES_PROVENANCE if phase_names else None,
            "roles": roles,
            "capabilities": caps,
            "feature_count": feature_count,
            "grain_count": grain_count,
            "declared_feature_tuple_count": declared_feature_tuple_count,
            "referenced_positive_feature_count": referenced_count,
            "feature_id_scan_complete": bool(feature_id_scan["complete"]),
            "feature_id_consistency": feature_id_consistency,
            "maps": maps,
            "feature_groups": feature_groups,
            "feature_group_reserved_zero": feature_group_reserved_zero,
            "feature_zero_reserved": feature_zero_reserved,
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
        # locking=False: uploads live on an NFS mount where HDF5's default
        # POSIX file lock can wedge the open in an uninterruptible syscall.
        # We only ever open read-only, so the writer-coordination lock is moot.
        h5 = h5py.File(path, "r", locking=False)
    except Exception as exc:  # noqa: BLE001
        top["hdf5"] = _unsupported_hdf5(f"could not open as HDF5: {exc}")
        return top

    try:
        root_keys = [str(key) for key in _bounded_child_names(h5)]
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
                "phase_names_source": materials_probe["phase_names_source"],
                "phase_names_provenance": materials_probe["phase_names_provenance"],
                "feature_count": materials_probe["feature_count"],
                "grain_count": materials_probe["grain_count"],
                "declared_feature_tuple_count": materials_probe[
                    "declared_feature_tuple_count"
                ],
                "referenced_positive_feature_count": materials_probe[
                    "referenced_positive_feature_count"
                ],
                "feature_id_scan_complete": materials_probe["feature_id_scan_complete"],
                "feature_id_consistency": materials_probe["feature_id_consistency"],
                "feature_zero_reserved": materials_probe["feature_zero_reserved"],
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

        # Top-level axis_sizes for the DEFAULT dataset (preview grid — the same
        # index range the generic /slice endpoint serves), so the Resources-grid
        # hover z-scrub engages for HDF5 cards. Advisory: any failure just means
        # no scrub.
        if default_dataset_path:
            try:
                dset = _hard_object_at_path(h5, str(default_dataset_path))
                if dset is not None and _is_dataset(dset):
                    vol = _interpret_volume(tuple(dset.shape), dset.dtype)
                    if vol is not None:
                        top["axis_sizes"] = {
                            "T": 1,
                            "C": int(vol["comp"]),
                            "Z": int(vol["pz"]),
                            "Y": int(vol["py"]),
                            "X": int(vol["px"]),
                        }
            except Exception:  # noqa: BLE001 - axis sizes are advisory
                pass

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
    # The h5py import lives inside the try so an environment without h5py
    # degrades to the same 422 "unsupported" class as a corrupt file, instead
    # of a 500 with a traceback.
    try:
        import h5py

        # locking=False: files sit on NFS, where HDF5's default file lock can
        # hang the open indefinitely (uninterruptible D-state). Opens are always
        # read-only here, so the lock (which coordinates writers) is unnecessary.
        return h5py.File(path, "r", locking=False)
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
        registration = None
        role_hint = _semantic_role(name, shallow or "")
        if role_hint in {"feature_ids", "euler_angles", "ipf_colors"}:
            try:
                registration = _resolve_feature_registration(
                    h5,
                    dataset_path,
                    target=dset,
                    target_kind=shallow,
                )
            except Hdf5Error:
                registration = None
        if registration is not None:
            vol = registration["target_volume"]
            shallow = {
                "feature_ids": "label_volume",
                "euler_angles": "vector_volume",
                "ipf_colors": "rgb_volume",
            }[registration["target_role"]]
        else:
            vol = _interpret_volume(shape, dt) if shallow in _VOLUME_KINDS else None

        attributes = _read_attrs(dset)

        # Geometry is dataset-local: only a direct CellData sibling may calibrate it.
        geometry = _dataset_local_geometry(h5, dataset_path, target=dset)

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

        semantic_role = _semantic_role(name, preview_kind or "") if is_volume_kind else None
        render_policy = _render_policy(preview_kind)
        texture_policy = (
            "nearest"
            if preview_kind == "label_volume" or semantic_role == "ipf_colors"
            else "linear"
        )
        delivery_mode = _delivery_mode(preview_kind, render_policy)

        # Axis sizes / dimension summary (PREVIEW grid — what the slice/volume endpoints serve).
        axis_sizes = None
        dimension_summary = None
        slice_axes: list[str] = []
        preview_planes: dict[str, Any] = {}
        atlas_scheme = None
        physical_spacing = _preview_spacing_from_geometry(geometry, vol)

        if vol is not None:
            pz, py, px = vol["pz"], vol["py"], vol["px"]
            axis_sizes = {"T": 1, "C": component_count, "Z": int(pz), "Y": int(py), "X": int(px)}
            dimension_summary = {"z": int(pz), "y": int(py), "x": int(px)}
            slice_axes = ["z", "y", "x"] if pz > 1 else ["z"]
            preview_planes = _build_preview_planes(vol, slice_axes, physical_spacing)
            if volume_eligible and render_policy != "scalar":
                atlas_scheme = _atlas_scheme(px, py, pz)

        feature_filter = (
            _feature_filter_metadata(registration) if registration is not None else None
        )

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
        if feature_filter is not None:
            capabilities.append("feature_filter")

        # sample_values: a tiny corner excerpt (JSON-safe).
        sample_values = _sample_values_excerpt(dset, dt, shape)
        sample_shape = list(np.asarray(sample_values).shape) if sample_values is not None else []

        summary: dict[str, Any] = {
            "file_id": file_id,
            "dataset_path": dataset_path,
            "dataset_name": name,
            "preview_kind": preview_kind,
            "semantic_role": semantic_role,
            "feature_filter": feature_filter,
            "units_hint": _units_hint(name, dset),
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


def _preview_spacing_from_geometry(
    geometry: dict[str, Any] | None, vol: dict[str, Any] | None
) -> dict[str, Any]:
    """Calibrate native geometry spacing to the bounded preview grid.

    ``geometry`` remains raw XYZ source metadata; ``physical_spacing`` describes
    preview voxels.  Scaling by native/preview counts preserves the exact native
    physical extent even when ceil-based sampling produces a partial final stride.
    """
    missing = {"x": None, "y": None, "z": None}
    if vol is None or not geometry:
        return missing
    dimensions = geometry.get("dimensions")
    spacing = geometry.get("spacing")
    if not (
        _valid_geometry_vector(dimensions, positive=True)
        and _valid_geometry_vector(spacing, positive=True)
    ):
        return missing
    try:
        native_xyz = (int(vol["x"]), int(vol["y"]), int(vol["z"]))
        preview_xyz = (int(vol["px"]), int(vol["py"]), int(vol["pz"]))
        geometry_xyz = tuple(float(value) for value in dimensions[:3])
    except (KeyError, TypeError, ValueError, OverflowError):
        return missing
    if geometry_xyz != native_xyz or any(value <= 0 for value in preview_xyz):
        return missing

    raw = _spacing_from_geometry(geometry)
    try:
        source_extents = {
            axis: float(raw[axis]) * native_count
            for axis, native_count in zip(("x", "y", "z"), native_xyz, strict=True)
        }
    except (TypeError, ValueError, OverflowError):
        return missing
    if not all(math.isfinite(value) and value > 0 for value in source_extents.values()):
        return missing
    try:
        effective = {
            axis: source_extents[axis] / preview_count
            for axis, preview_count in zip(("x", "y", "z"), preview_xyz, strict=True)
        }
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return missing
    if not all(math.isfinite(value) and value > 0 for value in effective.values()):
        return missing
    return effective


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
        w = int(s["w"])
        h = int(s["h"])
        world_w = w * float(s["col"])
        world_h = h * float(s["row"])
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
        shallow = _classify_shallow(name, dt, shape)
        if _semantic_role(name, shallow or "") in {"feature_ids", "euler_angles", "ipf_colors"}:
            try:
                registration = _resolve_feature_registration(
                    h5,
                    dataset_path,
                    target=dset,
                    target_kind=shallow,
                )
                exact_kind = {
                    "feature_ids": "label_volume",
                    "euler_angles": "vector_volume",
                    "ipf_colors": "rgb_volume",
                }[registration["target_role"]]
                return h5, dset, dt, registration["target_volume"], exact_kind
            except Hdf5Error:
                pass
        vol = _interpret_volume(shape, dt)
        if vol is None:
            raise Hdf5Error("unsupported: dataset shape cannot be rendered as a preview volume")
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


# Per-worker cache of the default-dataset choice, keyed on (path, mtime): the
# tree walk is bounded but the DREAM.3D materials probe scans the full
# FeatureIds array, and the grid z-scrub calls this once per FRAME — uncached,
# one hover over a production-sized .dream3d card re-reads gigabytes per frame.
_DEFAULT_DATASET_CACHE: dict[tuple, str | None] = {}


def default_dataset_path(path: str) -> str | None:
    """The dataset the viewer would select by default: the recommended DREAM.3D
    map when the materials schema is detected, else the largest renderable
    dataset from the tree walk (same policy as :func:`build_hdf5_viewer_info`)."""
    try:
        mtime = os.stat(path).st_mtime_ns
    except OSError:
        mtime = 0
    key = (path, mtime)
    if key in _DEFAULT_DATASET_CACHE:
        return _DEFAULT_DATASET_CACHE[key]
    h5 = _open(path)
    try:
        _tree, _kinds, _truncated, _groups, _datasets, walk_default = _walk_tree(h5)
        probe = _materials_probe(h5)
    finally:
        _safe_close(h5)
    result: str | None = walk_default
    if probe["detected"] and probe["recommended"]:
        result = str(probe["recommended"])
    if len(_DEFAULT_DATASET_CACHE) > 512:
        _DEFAULT_DATASET_CACHE.clear()
    _DEFAULT_DATASET_CACHE[key] = result
    return result


def thumbnail_png(path: str, *, max_size: int = 512) -> bytes:
    """Grid thumbnail for an HDF5-data file: the default dataset's mid-Z preview
    slice, bounded to ``max_size`` on the long edge. Raises :class:`Hdf5Error`
    (a 422 through the service's decode-marker mapping) when the file holds no
    renderable dataset (tables/scalars only), so the grid falls back to its
    icon without counting a server fault."""
    from PIL import Image

    dataset_path = default_dataset_path(path)
    if not dataset_path:
        raise Hdf5Error("no renderable dataset in HDF5 file (unsupported for preview)")
    png = slice_png(path, dataset_path)
    img = Image.open(io.BytesIO(png))
    img.load()
    width, height = img.size
    long_edge = max(width, height)
    if long_edge <= max_size:
        return png
    scale = max_size / long_edge
    resized = img.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))), Image.BILINEAR
    )
    buf = io.BytesIO()
    resized.save(buf, format="PNG")
    return buf.getvalue()


def slice_png(path: str, dataset_path: str, *, axis: str = "z", index: int | None = None,
              component: int = 0, feature_ids: str | None = None) -> bytes:
    """One preview slice as PNG. ``index`` is a preview-grid index (clamped, never 500)."""
    selected = _feature_filter_ids(feature_ids)
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
        if selected is not None:
            registration = _resolve_feature_registration(
                h5,
                dataset_path,
                target=dset,
                target_kind=kind,
            )
            identity_plane = (
                plane
                if _object_address(registration["identity"]) == _object_address(dset)
                else _read_preview_plane(
                    registration["identity"],
                    registration["identity_volume"],
                    axis,
                    int(index),
                    0,
                    rgb=False,
                )
            )
            _validate_feature_identity_values(identity_plane)
            img = _feature_mask_rgba(img, identity_plane, selected)
        return _encode_png(img)
    finally:
        _safe_close(h5)


def atlas_png(
    path: str,
    dataset_path: str,
    *,
    component: int = 0,
    feature_ids: str | None = None,
    **_ignored: Any,
) -> bytes:
    """The Z-atlas mosaic PNG matching the summary's ``atlas_scheme``.

    ``component`` selects the scalar component for vector volumes. Display-only
    extras (enhancement/fusion_method/negative/channels) remain accepted no-ops for
    compatibility with the shared atlas URL surface.
    """
    import numpy as np
    from PIL import Image

    selected = _feature_filter_ids(feature_ids)
    h5, dset, dt, vol, kind = _volume_context(path, dataset_path)
    try:
        component = _atlas_component_index(component, vol, kind)
        pz, py, px = vol["pz"], vol["py"], vol["px"]
        scheme = _atlas_scheme(px, py, pz)
        cols, rows = scheme["columns"], scheme["rows"]
        cell_w, cell_h = scheme["slice_width"], scheme["slice_height"]
        channels = 4 if selected is not None else 3
        atlas = np.zeros((cell_h * rows, cell_w * cols, channels), dtype="uint8")
        lo, hi = _global_window(path, dset, dataset_path, vol, component) if kind not in ("label_volume", "rgb_volume") else (0.0, 1.0)
        dataset_name = dataset_path.rsplit("/", 1)[-1] or dataset_path
        preserve_voxel_colors = (
            kind == "label_volume" or _semantic_role(dataset_name, kind) == "ipf_colors"
        )
        registration = None
        if selected is not None:
            registration = _resolve_feature_registration(
                h5,
                dataset_path,
                target=dset,
                target_kind=kind,
            )
        for z in range(pz):
            plane = _read_preview_plane(
                dset, vol, "z", z, component, rgb=(kind == "rgb_volume")
            )
            if kind == "label_volume":
                cell = _label_to_rgb(plane)
            elif kind == "rgb_volume":
                cell = _rgb_to_uint8(plane)
            else:
                gray = _window_to_uint8(plane, lo, hi)
                cell = np.stack([gray, gray, gray], axis=-1)
            if selected is not None and registration is not None:
                # Normalize the target first, then map both target and identities
                # through one deterministic nearest-neighbor delivery grid.
                cell = _resize_nearest(cell, cell_h, cell_w)
                identity = (
                    plane
                    if _object_address(registration["identity"]) == _object_address(dset)
                    else _read_preview_plane(
                        registration["identity"],
                        registration["identity_volume"],
                        "z",
                        z,
                        0,
                        rgb=False,
                    )
                )
                _validate_feature_identity_values(identity)
                identity = _resize_nearest(identity, cell_h, cell_w)
                cell = _feature_mask_rgba(cell, identity, selected)
            elif cell.shape[0] != cell_h or cell.shape[1] != cell_w:
                # Labels and IPF colors encode exact per-voxel identities/orientations.
                # Bilinear resizing would invent categorical labels or orientation
                # colors that do not occur in the source field.
                if preserve_voxel_colors:
                    cell = _resize_nearest(cell, cell_h, cell_w)
                else:
                    cell = np.asarray(
                        Image.fromarray(cell).resize(
                            (cell_w, cell_h), Image.Resampling.BILINEAR
                        ),
                        dtype="uint8",
                    )
            r, c = divmod(z, cols)
            atlas[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w, :] = cell
        buf = io.BytesIO()
        Image.fromarray(atlas, mode="RGBA" if selected is not None else "RGB").save(buf, format="PNG")
        return buf.getvalue()
    finally:
        _safe_close(h5)


def _validate_feature_identity_values(values: Any) -> tuple[int, int]:
    import numpy as np

    array = np.asarray(values)
    if array.size == 0:
        return (0, 0)
    minimum = int(array.min())
    maximum = int(array.max())
    if minimum < 0 or maximum > FEATURE_ID_UINT32_MAX:
        raise Hdf5Error("unsupported: FeatureIds contains values outside uint32")
    return minimum, maximum


def scalar_volume(path: str, dataset_path: str, *, channel: int = 0) -> dict[str, Any]:
    """Raw voxel buffer + ``x-volume-*`` metadata for the GPU volume renderer. The
    preview grid is downsampled to :data:`PREVIEW_MAX_VOXELS`; ``raw_min``/``raw_max``
    are the sampled data range so windowing works."""
    import numpy as np

    h5, dset, dt, vol, kind = _volume_context(path, dataset_path)
    try:
        arr = _read_preview_volume(dset, vol, channel)
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
            # Identity rescale: HDF5 scalar arrays are used as raw values (no
            # NIfTI-style scl_slope/scl_inter header). The scalar-volume envelope
            # in imaging/service.py requires both keys, so emit the identity pair.
            "scl_slope": 1.0,
            "scl_inter": 0.0,
            "channel": operator.index(channel),
            "t": 0,
            "source_width": int(vol["x"]),
            "source_height": int(vol["y"]),
            "source_depth": int(vol["z"]),
            "downsample_x": int(vol["sx"]),
            "downsample_y": int(vol["sy"]),
            "downsample_z": int(vol["sz"]),
            "preview_policy": "stride-v1",
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
            "phase_names_source": probe["phase_names_source"],
            "phase_names_provenance": probe["phase_names_provenance"],
            "feature_count": probe["feature_count"],
            "grain_count": probe["grain_count"],
            "declared_feature_tuple_count": probe["declared_feature_tuple_count"],
            "referenced_positive_feature_count": probe[
                "referenced_positive_feature_count"
            ],
            "feature_id_scan_complete": probe["feature_id_scan_complete"],
            "feature_id_consistency": probe["feature_id_consistency"],
            "feature_zero_reserved": probe["feature_zero_reserved"],
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


def _bounded_feature_row_sample(
    dset,
    np,
    *,
    max_rows: int,
    max_columns: int | None = None,
    reserved_zero: bool,
):
    """Sample feature rows, excluding tuple zero only with schema evidence.

    The slice stride is computed before reading, so a multi-gigabyte feature table
    never enters memory in full. When tuple zero is established as reserved,
    exclusion is positional: zero-valued measurements in rows 1..N remain valid.
    """
    try:
        shape = tuple(int(value) for value in dset.shape)
        start_row = 1 if reserved_zero else 0
        if len(shape) not in (1, 2) or not shape or shape[0] <= start_row:
            return np.asarray([], dtype="float64")
        available_rows = shape[0] - start_row
        step = max(1, math.ceil(available_rows / max(1, max_rows)))
        row_slice = slice(start_row, shape[0], step)
        if len(shape) == 1:
            block = dset[row_slice]
        else:
            column_count = shape[1]
            if max_columns is not None:
                column_count = min(column_count, max_columns)
            if column_count <= 0:
                return np.asarray([], dtype="float64")
            block = dset[row_slice, :column_count]
        return np.asarray(block, dtype="float64")
    except Exception:  # noqa: BLE001
        return np.asarray([], dtype="float64")


def _grain_charts(h5, probe, np) -> list[dict[str, Any]]:
    """Grain-scale distributions from per-feature 1-D arrays (EquivalentDiameters,
    NumNeighbors, ...). Bounded sampling; empty list if nothing suitable."""
    charts: list[dict[str, Any]] = []
    wanted = ("equivalentdiameters", "numneighbors", "numelements", "volumes", "size")
    for grp_path in probe["feature_groups"]:
        reserved_zero = bool(probe["feature_group_reserved_zero"].get(grp_path, False))
        grp = _hard_object_at_path(h5, grp_path)
        if not _is_group(grp):
            continue
        for key in _bounded_child_names(grp):
            low = key.lower()
            if not any(w in low for w in wanted):
                continue
            try:
                d = _hard_child(grp, key)
                if not _is_dataset(d) or not _is_numeric_dtype(d.dtype):
                    continue
                vals = _bounded_feature_row_sample(
                    d,
                    np,
                    max_rows=TABLE_CHART_MAX_ROWS,
                    max_columns=1,
                    reserved_zero=reserved_zero,
                ).ravel()
                vals = vals[np.isfinite(vals)]
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
                    "units_hint": _units_hint(key, d),
                    "provenance": (
                        "Bounded sample of per-feature values; reserved feature row 0 "
                        "excluded by index based on selected-container FeatureIds/GrainIds "
                        "schema evidence. Zero-valued measurements in real rows are retained."
                        if reserved_zero
                        else "Bounded sample of per-feature values; no reserved feature row "
                        "was established by file/schema evidence, so row 0 is retained."
                    ),
                })
            except Exception:  # noqa: BLE001
                continue
            if len(charts) >= 4:
                return charts
    return charts


def _orientation_charts(h5, probe, np) -> list[dict[str, Any]]:
    """An orientation scatter (phi1 vs Phi) from a per-feature AvgEulerAngles array,
    if present. Bounded to :data:`MATERIAL_ORIENTATION_MAX_ROWS` points."""
    charts: list[dict[str, Any]] = []
    for grp_path in probe["feature_groups"]:
        reserved_zero = bool(probe["feature_group_reserved_zero"].get(grp_path, False))
        grp = _hard_object_at_path(h5, grp_path)
        if not _is_group(grp):
            continue
        for key in _bounded_child_names(grp):
            if "euler" not in key.lower():
                continue
            try:
                d = _hard_child(grp, key)
                if not _is_dataset(d) or not _is_numeric_dtype(d.dtype):
                    continue
                arr = _bounded_feature_row_sample(
                    d,
                    np,
                    max_rows=MATERIAL_ORIENTATION_MAX_ROWS,
                    max_columns=3,
                    reserved_zero=reserved_zero,
                )
                if arr.ndim != 2 or arr.shape[1] < 2:
                    continue
                data = [
                    {"phi1": _num(arr[i, 0]), "value": _num(arr[i, 1])}
                    for i in range(arr.shape[0])
                    if math.isfinite(arr[i, 0]) and math.isfinite(arr[i, 1])
                ]
                if data:
                    charts.append({
                        "kind": "scatter",
                        "title": f"{_titleize(key)}: φ1 vs Φ",
                        "description": f"{len(data)} features sampled",
                        "x_key": "phi1",
                        "y_key": "value",
                        "data": data,
                        "source_paths": [grp_path.rstrip("/") + "/" + key],
                        "units_hint": _units_hint(key, d),
                        "provenance": (
                            "Bounded sample of stored per-feature orientation angles; reserved "
                            "feature row 0 excluded by index based on selected-container "
                            "FeatureIds/GrainIds schema evidence."
                            if reserved_zero
                            else "Bounded sample of stored per-feature orientation angles; no "
                            "reserved feature row was established, so row 0 is retained."
                        ),
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
