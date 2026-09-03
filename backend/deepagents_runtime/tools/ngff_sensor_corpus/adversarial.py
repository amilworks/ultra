"""Malformed / adversarial OME-Zarr stores.

The control plane trusts the ``?path=`` it hands the ngff-service, but the *contents* of a
store are attacker-influenced: a user can upload a crafted OME-Zarr. Every case here is a
store that a hostile or broken writer could produce. Two classes:

  * ``reject`` — the reader MUST fail closed with a clear ``NgffError`` (surfaced as HTTP
    422), never crash the worker, hang, read out of bounds, or silently misinterpret.
  * ``probe`` — behaviour is not obviously specified; the harness records what actually
    happens (used for the security-relevant dtype and symlink cases).

Each builder returns an :class:`AdversarialCase`. Builders that cannot be constructed on a
given zarr version degrade to ``None`` and are skipped rather than aborting the corpus.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import zarr

__all__ = ["AdversarialCase", "build_adversarial"]


@dataclass
class AdversarialCase:
    name: str
    path: str
    classification: str  # "reject" | "probe"
    category: str
    expect_error_substr: str | None
    note: str


def _group(path: str, zarr_format: int = 2):
    return zarr.open_group(path, mode="w", zarr_format=zarr_format)


def _valid_axes_yx() -> list[dict[str, Any]]:
    return [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]


def _scale_tf(vec: list[float]) -> list[dict[str, Any]]:
    return [{"type": "scale", "scale": vec}]


# --------------------------------------------------------------------------- builders
def _no_multiscales(path: str) -> AdversarialCase:
    g = _group(path)
    g.create_array("0", shape=(8, 8), chunks=(8, 8), dtype="uint16")
    g.attrs["note"] = "a plain zarr group with no OME metadata"
    return AdversarialCase(
        "generic_zarr_no_multiscales",
        path,
        "reject",
        "not-ome-zarr",
        "no OME-NGFF 'multiscales'",
        "Generic Zarr must not be guessed as OME-Zarr.",
    )


def _axis_name_type_mismatch(path: str) -> AdversarialCase:
    g = _group(path)
    g.create_array("0", shape=(3, 8, 8), chunks=(1, 8, 8), dtype="uint16")
    g.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "bad",
            "axes": [
                {"name": "t", "type": "channel", "unit": "hour"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [{"path": "0", "coordinateTransformations": _scale_tf([1.0, 1.0, 1.0])}],
        }
    ]
    return AdversarialCase(
        "axis_name_type_mismatch",
        path,
        "reject",
        "spec-conformance",
        "requires type",
        "A channel axis mislabelled 't' (a real converter artifact).",
    )


def _ambiguous_multiscales(path: str) -> AdversarialCase:
    g = _group(path, zarr_format=3)
    g.create_array("0", shape=(8, 8), chunks=(8, 8), dtype="uint16", dimension_names=("y", "x"))
    ms = {
        "name": "m",
        "axes": _valid_axes_yx(),
        "datasets": [{"path": "0", "coordinateTransformations": _scale_tf([1.0, 1.0])}],
    }
    g.attrs["multiscales"] = [dict(ms, version="0.4")]
    g.attrs["ome"] = {"version": "0.5", "multiscales": [ms]}
    return AdversarialCase(
        "ambiguous_multiscales",
        path,
        "reject",
        "spec-conformance",
        "ambiguous OME-NGFF metadata",
        "multiscales present at both root and under 'ome'.",
    )


def _missing_dataset_transforms(path: str) -> AdversarialCase:
    g = _group(path)
    g.create_array("0", shape=(8, 8), chunks=(8, 8), dtype="uint16")
    g.attrs["multiscales"] = [
        {"version": "0.4", "name": "m", "axes": _valid_axes_yx(), "datasets": [{"path": "0"}]}
    ]
    return AdversarialCase(
        "missing_dataset_transforms",
        path,
        "reject",
        "spec-conformance",
        "coordinateTransformations is required",
        "Dataset lacks the required transform list.",
    )


def _nonfinite_scale(path: str) -> AdversarialCase:
    g = _group(path)
    g.create_array("0", shape=(8, 8), chunks=(8, 8), dtype="uint16")
    g.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "m",
            "axes": _valid_axes_yx(),
            "datasets": [
                {"path": "0", "coordinateTransformations": _scale_tf([float("nan"), 1.0])}
            ],
        }
    ]
    return AdversarialCase(
        "nonfinite_scale",
        path,
        "reject",
        "malformed-metadata",
        "must be finite",
        "A NaN scale factor must be rejected, not composed.",
    )


def _duplicate_axis(path: str) -> AdversarialCase:
    g = _group(path)
    g.create_array("0", shape=(8, 8), chunks=(8, 8), dtype="uint16")
    g.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "m",
            "axes": [{"name": "x", "type": "space"}, {"name": "x", "type": "space"}],
            "datasets": [{"path": "0", "coordinateTransformations": _scale_tf([1.0, 1.0])}],
        }
    ]
    return AdversarialCase(
        "duplicate_axis_name",
        path,
        "reject",
        "spec-conformance",
        "duplicated",
        "Two axes named 'x' (also no 'y').",
    )


def _wrong_omero_count(path: str) -> AdversarialCase:
    g = _group(path)
    g.create_array("0", shape=(3, 8, 8), chunks=(1, 8, 8), dtype="uint16")
    g.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "m",
            "axes": [
                {"name": "c", "type": "channel"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [{"path": "0", "coordinateTransformations": _scale_tf([1.0, 1.0, 1.0])}],
        }
    ]
    g.attrs["omero"] = {"channels": [{"label": "only-one", "color": "FFFFFF"}]}  # c=3 but 1 channel
    return AdversarialCase(
        "wrong_omero_channel_count",
        path,
        "reject",
        "spec-conformance",
        "omero.channels has",
        "omero declares 1 channel for a 3-channel array.",
    )


def _path_traversal(path: str) -> AdversarialCase:
    g = _group(path)
    g.create_array("0", shape=(8, 8), chunks=(8, 8), dtype="uint16")
    g.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "m",
            "axes": _valid_axes_yx(),
            "datasets": [
                {
                    "path": "../../../../etc/passwd",
                    "coordinateTransformations": _scale_tf([1.0, 1.0]),
                }
            ],
        }
    ]
    return AdversarialCase(
        "path_traversal_dataset",
        path,
        "reject",
        "security",
        "unsafe or empty path component",
        "Dataset path tries to escape the store root.",
    )


def _rank6(path: str) -> AdversarialCase:
    g = _group(path)
    g.create_array("0", shape=(1, 1, 1, 2, 3, 4), chunks=(1, 1, 1, 2, 3, 4), dtype="uint16")
    axes = [
        {"name": n, "type": ("time" if n == "t" else "channel" if n == "c" else "space")}
        for n in ("t", "c", "z", "z", "y", "x")
    ]
    # duplicate 'z' name would trip the duplicate check first, so use distinct: t,c,z,y,x is 5.
    axes = [
        {"name": "t", "type": "time"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space"},
        {"name": "b", "type": None},
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"},
    ]
    g.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "m",
            "axes": axes,
            "datasets": [{"path": "0", "coordinateTransformations": _scale_tf([1.0] * 6)}],
        }
    ]
    return AdversarialCase(
        "rank6_array",
        path,
        "reject",
        "spec-conformance",
        "between 2 and 5 dimensions",
        "6-D array exceeds the OME-NGFF rank bound.",
    )


def _custom_nonsingleton(path: str) -> AdversarialCase:
    g = _group(path)
    g.create_array("0", shape=(3, 8, 8), chunks=(1, 8, 8), dtype="uint16")
    g.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "m",
            "axes": [
                {"name": "b", "type": None},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [{"path": "0", "coordinateTransformations": _scale_tf([1.0, 1.0, 1.0])}],
        }
    ]
    return AdversarialCase(
        "custom_nonsingleton_axis",
        path,
        "reject",
        "spec-conformance",
        "non-singleton axis",
        "A custom axis 'b' of size 3 has no viewer selector.",
    )


def _v05_dimnames_mismatch(path: str) -> AdversarialCase:
    g = _group(path, zarr_format=3)
    g.create_array("0", shape=(8, 8), chunks=(8, 8), dtype="uint16", dimension_names=("x", "y"))
    g.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "name": "m",
                "axes": _valid_axes_yx(),
                "datasets": [{"path": "0", "coordinateTransformations": _scale_tf([1.0, 1.0])}],
            }
        ],
    }
    return AdversarialCase(
        "v05_dimension_names_mismatch",
        path,
        "reject",
        "spec-conformance",
        "do not match axes names",
        "Zarr dimension_names (x,y) contradict axes (y,x).",
    )


def _v05_at_root(path: str) -> AdversarialCase:
    g = _group(path, zarr_format=3)
    g.create_array("0", shape=(8, 8), chunks=(8, 8), dtype="uint16", dimension_names=("y", "x"))
    g.attrs["multiscales"] = [
        {
            "version": "0.5",
            "name": "m",
            "axes": _valid_axes_yx(),
            "datasets": [{"path": "0", "coordinateTransformations": _scale_tf([1.0, 1.0])}],
        }
    ]
    return AdversarialCase(
        "v05_declared_at_root",
        path,
        "reject",
        "spec-conformance",
        "must be nested under 'ome'",
        "version 0.5 at the root instead of under 'ome'.",
    )


def _scale_decreasing(path: str) -> AdversarialCase:
    g = _group(path)
    g.create_array("0", shape=(16, 16), chunks=(16, 16), dtype="uint16")
    g.create_array("1", shape=(8, 8), chunks=(8, 8), dtype="uint16")
    g.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "m",
            "axes": _valid_axes_yx(),
            "datasets": [
                {"path": "0", "coordinateTransformations": _scale_tf([2.0, 2.0])},
                {"path": "1", "coordinateTransformations": _scale_tf([1.0, 1.0])},
            ],
        }
    ]
    return AdversarialCase(
        "scale_decreasing_pyramid",
        path,
        "reject",
        "spec-conformance",
        "spatial scale decreases",
        "A coarser level claims finer sampling.",
    )


def _resolution_increasing(path: str) -> AdversarialCase:
    g = _group(path)
    g.create_array("0", shape=(8, 8), chunks=(8, 8), dtype="uint16")
    g.create_array("1", shape=(16, 16), chunks=(16, 16), dtype="uint16")
    g.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "m",
            "axes": _valid_axes_yx(),
            "datasets": [
                {"path": "0", "coordinateTransformations": _scale_tf([1.0, 1.0])},
                {"path": "1", "coordinateTransformations": _scale_tf([2.0, 2.0])},
            ],
        }
    ]
    return AdversarialCase(
        "resolution_increasing_pyramid",
        path,
        "reject",
        "spec-conformance",
        "increases resolution dimensions",
        "Level 1 is larger than level 0.",
    )


def _too_many_levels(path: str) -> AdversarialCase:
    g = _group(path)
    g.create_array("0", shape=(8, 8), chunks=(8, 8), dtype="uint16")
    datasets = [
        {"path": "0", "coordinateTransformations": _scale_tf([1.0, 1.0])} for _ in range(300)
    ]
    g.attrs["multiscales"] = [
        {"version": "0.4", "name": "m", "axes": _valid_axes_yx(), "datasets": datasets}
    ]
    return AdversarialCase(
        "too_many_levels",
        path,
        "reject",
        "resource-bound",
        "resolution levels; limit is",
        "300 declared levels exceed the 256 cap.",
    )


def _no_y_axis(path: str) -> AdversarialCase:
    g = _group(path)
    g.create_array("0", shape=(3, 8), chunks=(3, 8), dtype="uint16")
    g.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "m",
            "axes": [{"name": "c", "type": "channel"}, {"name": "x", "type": "space"}],
            "datasets": [{"path": "0", "coordinateTransformations": _scale_tf([1.0, 1.0])}],
        }
    ]
    return AdversarialCase(
        "no_y_axis",
        path,
        "reject",
        "spec-conformance",
        "exactly one explicit 'y' and 'x'",
        "Array has c,x but no y axis.",
    )


def _two_scales(path: str) -> AdversarialCase:
    g = _group(path)
    g.create_array("0", shape=(8, 8), chunks=(8, 8), dtype="uint16")
    g.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "m",
            "axes": _valid_axes_yx(),
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0]},
                        {"type": "scale", "scale": [2.0, 2.0]},
                    ],
                }
            ],
        }
    ]
    return AdversarialCase(
        "two_scale_transforms",
        path,
        "reject",
        "spec-conformance",
        "exactly one scale",
        "Two scale transforms in one dataset.",
    )


def _complex_dtype(path: str) -> AdversarialCase:
    g = _group(path)
    a = g.create_array("0", shape=(8, 8), chunks=(8, 8), dtype="complex64")
    a[:] = (
        np.random.default_rng(0).standard_normal((8, 8))
        + 1j * np.random.default_rng(1).standard_normal((8, 8))
    ).astype("complex64")
    g.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "m",
            "axes": _valid_axes_yx(),
            "datasets": [{"path": "0", "coordinateTransformations": _scale_tf([1.0, 1.0])}],
        }
    ]
    return AdversarialCase(
        "complex64_dtype",
        path,
        "reject",
        "dtype-surprise",
        "unsupported pixel dtype",
        "Complex (interferometry/radar phase) refused, not silently rendered real-part-only.",
    )


def _structured_dtype(path: str) -> AdversarialCase:
    g = _group(path)
    g.create_array("0", shape=(8, 8), chunks=(8, 8), dtype=np.dtype([("a", "u2"), ("b", "u2")]))
    g.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "m",
            "axes": _valid_axes_yx(),
            "datasets": [{"path": "0", "coordinateTransformations": _scale_tf([1.0, 1.0])}],
        }
    ]
    return AdversarialCase(
        "structured_void_dtype",
        path,
        "reject",
        "dtype-surprise",
        "unsupported pixel dtype",
        "Structured/void dtype refused at open, not a render-time TypeError (HTTP 500).",
    )


def _chunk_decompression_bomb(path: str) -> AdversarialCase:
    g = _group(path)
    # A modest array but a huge declared chunk: one tile would decode the whole chunk.
    g.create_array("0", shape=(4096, 4096), chunks=(20000, 20000), dtype="uint16", fill_value=0)
    g.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "m",
            "axes": _valid_axes_yx(),
            "datasets": [{"path": "0", "coordinateTransformations": _scale_tf([1.0, 1.0])}],
        }
    ]
    return AdversarialCase(
        "chunk_decompression_bomb",
        path,
        "reject",
        "resource-bound",
        "decoded chunk budget",
        "20000x20000 uint16 chunk (763 MiB decode) on a 4096 array.",
    )


def _deep_json_nesting(path: str) -> AdversarialCase:
    g = _group(path)
    g.create_array("0", shape=(8, 8), chunks=(8, 8), dtype="uint16")
    ms = (
        '{"multiscales":[{"version":"0.4","name":"m","axes":[{"name":"y","type":"space"},'
        '{"name":"x","type":"space"}],"datasets":[{"path":"0","coordinateTransformations":'
        '[{"type":"scale","scale":[1.0,1.0]}]}]}],'
    )
    n = 6000
    deep = '"junk":' + '{"n":' * n + "0" + "}" * n + "}"
    with open(os.path.join(path, ".zattrs"), "w") as fh:
        fh.write(ms + deep)  # bypasses json.dumps' own recursion
    return AdversarialCase(
        "deep_json_nesting",
        path,
        "reject",
        "malformed-metadata",
        "nesting exceeds the depth",
        "6000-deep .zattrs must fail closed, not RecursionError (500).",
    )


def _symlink_chunk_escape(path: str) -> AdversarialCase | None:
    """Replace one uncompressed chunk with a symlink to a host file of identical size.

    If the reader follows the symlink and returns its bytes, a crafted store could
    exfiltrate host files. We point at a benign sentinel file (0xAB bytes) and the harness
    checks whether those bytes surface in the decoded plane.
    """
    try:
        g = _group(path, zarr_format=2)
        # Uncompressed so a raw byte substitution is directly interpretable.
        a = g.create_array("0", shape=(8, 8), chunks=(8, 8), dtype="uint8", compressors=None)
        a[:] = np.arange(64, dtype="uint8").reshape(8, 8)
        g.attrs["multiscales"] = [
            {
                "version": "0.4",
                "name": "m",
                "axes": _valid_axes_yx(),
                "datasets": [{"path": "0", "coordinateTransformations": _scale_tf([1.0, 1.0])}],
            }
        ]
        # Locate the single chunk file written for array "0".
        array_dir = os.path.join(path, "0")
        chunk = None
        for root, _dirs, files in os.walk(array_dir):
            for f in files:
                if f in (".zarray", ".zattrs", "zarr.json"):
                    continue
                chunk = os.path.join(root, f)
        if chunk is None:
            return None
        sentinel = os.path.join(path, "_outside_sentinel.bin")
        with open(sentinel, "wb") as fh:
            fh.write(b"\xab" * 64)  # exactly one uncompressed uint8 8x8 chunk
        os.remove(chunk)
        os.symlink(sentinel, chunk)
        return AdversarialCase(
            "symlink_chunk_escape",
            path,
            "probe",
            "security",
            None,
            "A chunk symlinked to a host file; do the sentinel bytes (0xAB) surface?",
        )
    except Exception:
        return None


def _nonpositive_dim(path: str) -> AdversarialCase | None:
    """Declare a zero-length dimension by editing the array metadata after creation."""
    try:
        g = _group(path, zarr_format=2)
        g.create_array("0", shape=(4, 4), chunks=(4, 4), dtype="uint16")
        g.attrs["multiscales"] = [
            {
                "version": "0.4",
                "name": "m",
                "axes": _valid_axes_yx(),
                "datasets": [{"path": "0", "coordinateTransformations": _scale_tf([1.0, 1.0])}],
            }
        ]
        zarray = os.path.join(path, "0", ".zarray")
        with open(zarray) as fh:
            meta = json.load(fh)
        meta["shape"] = [4, 0]
        with open(zarray, "w") as fh:
            json.dump(meta, fh)
        return AdversarialCase(
            "nonpositive_dimension",
            path,
            "reject",
            "malformed-metadata",
            "non-positive dimension",
            "A level array declares a zero-length dimension.",
        )
    except Exception:
        return None


def build_adversarial(out_dir: str) -> list[AdversarialCase]:
    os.makedirs(out_dir, exist_ok=True)
    builders: list[Callable[[str], AdversarialCase | None]] = [
        _no_multiscales,
        _axis_name_type_mismatch,
        _ambiguous_multiscales,
        _missing_dataset_transforms,
        _nonfinite_scale,
        _duplicate_axis,
        _wrong_omero_count,
        _path_traversal,
        _rank6,
        _custom_nonsingleton,
        _v05_dimnames_mismatch,
        _v05_at_root,
        _scale_decreasing,
        _resolution_increasing,
        _too_many_levels,
        _no_y_axis,
        _two_scales,
        _complex_dtype,
        _structured_dtype,
        _chunk_decompression_bomb,
        _deep_json_nesting,
        _symlink_chunk_escape,
        _nonpositive_dim,
    ]
    cases: list[AdversarialCase] = []
    for b in builders:
        name = b.__name__.lstrip("_")
        p = os.path.join(out_dir, f"adversarial__{name}.zarr")
        case = b(p)
        if case is not None:
            cases.append(case)
    return cases
