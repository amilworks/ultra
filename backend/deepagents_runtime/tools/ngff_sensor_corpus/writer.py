"""Write a :class:`StoreSpec` to a spec-correct OME-Zarr store on disk.

The metadata this emits is deliberately built to satisfy the application reader's
validators (``ultra_deepagents.ngff.reader``): canonical axis types, one scale (plus
optional translation) per dataset, spatial-only downsampling across pyramid levels,
Zarr-v3 ``dimension_names`` bound to axis names for NGFF 0.5, and omero channel counts
that match the channel dimension. Pixels are generated one 2-D plane at a time so peak
memory is a single plane regardless of store size.
"""

from __future__ import annotations

import os
from itertools import product
from typing import Any

import numpy as np
import zarr

from . import signals
from .specs import StoreSpec

__all__ = ["build_corpus", "write_store"]

_CANONICAL_TYPE = {"t": "time", "c": "channel", "z": "space", "y": "space", "x": "space"}
_DEFAULT_SPATIAL_CHUNK = 256


def _level_shape(spec: StoreSpec, names: tuple[str, ...], level: int) -> tuple[int, ...]:
    out = []
    for name in names:
        size = spec.base[name]
        if name in ("y", "x"):
            size = max(1, size >> level)
        out.append(int(size))
    return tuple(out)


def _chunks(spec: StoreSpec, names: tuple[str, ...], shape: tuple[int, ...]) -> tuple[int, ...]:
    out = []
    for name, size in zip(names, shape, strict=True):
        if name in spec.chunks:
            out.append(min(spec.chunks[name], size))
        elif name in ("y", "x"):
            out.append(min(_DEFAULT_SPATIAL_CHUNK, size))
        else:
            out.append(1)  # one t/c/z index per chunk -> a plane touches minimal files
    return tuple(max(1, c) for c in out)


def _scale_vector(spec: StoreSpec, names: tuple[str, ...], level: int) -> list[float]:
    vec = []
    for name in names:
        base = float(spec.scale.get(name, 1.0))
        if name in ("y", "x"):
            base *= float(1 << level)  # coarser levels sample a larger physical step
        vec.append(base)
    return vec


def _cast_plane(plane: np.ndarray, spec: StoreSpec) -> np.ndarray:
    dt = np.dtype(spec.dtype)
    lo, hi = spec.value_range
    if lo == 0 and hi == 0:  # pass-through (label images)
        return np.rint(plane).astype(dt)
    scaled = lo + plane.astype(np.float64) * (hi - lo)
    if dt.kind in ("u", "i"):
        info = np.iinfo(dt)
        scaled = np.clip(np.rint(scaled), info.min, info.max)
    return scaled.astype(dt)


def _write_pixels(
    spec: StoreSpec, arr: Any, names: tuple[str, ...], shape: tuple[int, ...]
) -> None:
    """Fill one level's array plane-by-plane over its non-spatial coordinates."""
    pos = {name: i for i, name in enumerate(names)}
    yh = shape[pos["y"]]
    xw = shape[pos["x"]]
    spatial_order = tuple(n for n in names if n in ("y", "x"))

    def axis_range(name: str) -> range:
        return range(shape[pos[name]]) if name in pos else range(1)

    for t, c, z in product(axis_range("t"), axis_range("c"), axis_range("z")):
        plane = signals.plane(
            spec.signal,
            yh,
            xw,
            seed=abs(hash(spec.modality)) % 100_000,
            t=t,
            c=c,
            z=z,
            num_t=spec.base.get("t", 1),
            num_c=spec.base.get("c", 1),
            num_z=spec.base.get("z", 1),
        )
        if spatial_order == ("x", "y"):
            plane = plane.T
        block = _cast_plane(plane, spec)
        index: list[Any] = []
        for name in names:
            if name == "y" or name == "x":
                index.append(slice(None))
            elif name == "t":
                index.append(t)
            elif name == "c":
                index.append(c)
            elif name == "z":
                index.append(z)
        arr[tuple(index)] = block


def _omero_block(spec: StoreSpec) -> dict[str, Any] | None:
    if not spec.channels:
        return None
    c_size = spec.base.get("c", 1)
    if len(spec.channels) != c_size:
        raise ValueError(f"{spec.modality}: omero has {len(spec.channels)} channels but c={c_size}")
    lo, hi = spec.value_range
    if lo == 0 and hi == 0:
        lo, hi = (
            0.0,
            float(np.iinfo(np.dtype(spec.dtype)).max) if np.dtype(spec.dtype).kind in "ui" else 1.0,
        )
    channels = [
        {
            "label": ch.label,
            "color": ch.color,
            "active": True,
            "window": {"start": float(lo), "end": float(hi), "min": float(lo), "max": float(hi)},
        }
        for ch in spec.channels
    ]
    return {"name": spec.title, "channels": channels, "rdefs": {"model": "color"}}


def _multiscale(spec: StoreSpec, names: tuple[str, ...], n_levels: int) -> dict[str, Any]:
    axes = []
    for name, unit in spec.axes:
        axis: dict[str, Any] = {"name": name, "type": _CANONICAL_TYPE[name]}
        if unit:
            axis["unit"] = unit
        axes.append(axis)
    datasets = []
    for level in range(n_levels):
        transforms: list[dict[str, Any]] = [
            {"type": "scale", "scale": _scale_vector(spec, names, level)}
        ]
        if spec.translation:
            transforms.append(
                {
                    "type": "translation",
                    "translation": [float(spec.translation.get(n, 0.0)) for n in names],
                }
            )
        datasets.append({"path": str(level), "coordinateTransformations": transforms})
    ms: dict[str, Any] = {"name": spec.title[:256], "axes": axes, "datasets": datasets}
    if spec.ngff_version == "0.4":
        ms["version"] = "0.4"
    return ms


def write_store(spec: StoreSpec, out_dir: str) -> str:
    """Write ``spec`` under ``out_dir`` and return the store path."""
    names = spec.axis_names
    if names.count("y") != 1 or names.count("x") != 1:
        raise ValueError(f"{spec.modality}: needs exactly one y and one x axis")
    path = os.path.join(out_dir, spec.store_name)
    zarr_format = 3 if spec.zarr_version == 3 else 2
    group = zarr.open_group(path, mode="w", zarr_format=zarr_format)

    n_levels = max(1, spec.levels)
    for level in range(n_levels):
        shape = _level_shape(spec, names, level)
        chunks = _chunks(spec, names, shape)
        kwargs: dict[str, Any] = {
            "shape": shape,
            "chunks": chunks,
            "dtype": spec.dtype,
            "fill_value": 0,
        }
        if zarr_format == 3 and spec.ngff_version == "0.5":
            kwargs["dimension_names"] = names
        arr = group.create_array(str(level), **kwargs)
        if not spec.lazy_fill:
            _write_pixels(spec, arr, names, shape)

    ms = _multiscale(spec, names, n_levels)
    omero = _omero_block(spec)
    if spec.ngff_version == "0.5":
        ome: dict[str, Any] = {"version": "0.5", "multiscales": [ms]}
        if omero is not None:
            ome["omero"] = omero
        group.attrs["ome"] = ome
    else:
        group.attrs["multiscales"] = [ms]
        if omero is not None:
            group.attrs["omero"] = omero
    return path


def build_corpus(specs: list[StoreSpec], out_dir: str) -> list[tuple[StoreSpec, str]]:
    os.makedirs(out_dir, exist_ok=True)
    built = []
    for spec in specs:
        built.append((spec, write_store(spec, out_dir)))
    return built
