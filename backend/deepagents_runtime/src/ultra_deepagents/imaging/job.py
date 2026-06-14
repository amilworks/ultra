"""The ``image.derive_pyramid`` batch job: convert a source into a tiled pyramid.

This is the batch-processing half of the engine work. A job names a source image
and a destination; the runner invokes the tested convert primitive
(:func:`~ultra_deepagents.imaging.convert.derive_pyramid`) and reports the derived
artifact plus its pyramid metadata. The runner is pure and injectable so it is
unit-tested without the native engine; the NATS worker (`worker.py`) wraps it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ultra_deepagents.imaging.convert import ConvertResult, PyramidSpec, derive_pyramid

__all__ = ["DerivePyramidJob", "run_derive_pyramid_job", "ConvertFn", "MetaFn"]

ConvertFn = Callable[..., ConvertResult]
MetaFn = Callable[[str], dict[str, Any]]


@dataclass
class DerivePyramidJob:
    resource_id: str
    src_path: str
    dst_path: str
    tile_size: int = 512
    compression: str = "lzw"
    layout: str = "topdirs"
    fmt: str = "bigtiff"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DerivePyramidJob":
        return cls(
            resource_id=str(payload["resource_id"]),
            src_path=str(payload["src_path"]),
            dst_path=str(payload["dst_path"]),
            tile_size=int(payload.get("tile_size", 512)),
            compression=str(payload.get("compression", "lzw")),
            layout=str(payload.get("layout", "topdirs")),
            fmt=str(payload.get("fmt", "bigtiff")),
        )

    def spec(self) -> PyramidSpec:
        return PyramidSpec(
            tile_size=self.tile_size, compression=self.compression, layout=self.layout, fmt=self.fmt
        )


def _meta_int(meta: dict[str, Any], key: str, default: int = 1) -> int:
    try:
        return int(meta.get(key, default))
    except (TypeError, ValueError):
        return default


def _resolve_auto_fmt(meta: dict[str, Any] | None) -> str:
    """Choose the derived container from the source's dimensionality.

    Returns ``"ome-bigtiff"`` for a z-stack/volume (so the Z dimension and per-plane
    pyramid survive), else ``"bigtiff"`` (tile-addressable for flat 2D slides).
    Best-effort: missing meta falls back to ``"bigtiff"``.
    """
    if not meta:
        return "bigtiff"
    z = _meta_int(meta, "image_num_z", 1)
    t = _meta_int(meta, "image_num_t", 1)
    c = _meta_int(meta, "image_num_c", 1)
    pages = _meta_int(meta, "image_num_p", 1)
    is_volume = z > 1 or (pages > 1 and c <= 1 and t <= 1)
    return "ome-bigtiff" if is_volume else "bigtiff"


def _source_is_native_tiled_pyramid(meta: dict[str, Any] | None) -> bool:
    """Whether the source ALREADY exposes a tiled multi-resolution pyramid the image
    service can serve tiles from directly — making a derived pyramid redundant (e.g. a
    4.7GB orthomosaic with 8 native levels + 256px tiles otherwise spawns a 21GB copy).
    Borrowed from BisQue's native-pyramid fast path."""
    if not meta:
        return False
    return _meta_int(meta, "image_num_resolution_levels", 1) > 1 and _meta_int(meta, "tile_num_x", 0) > 0


def run_derive_pyramid_job(
    job: dict[str, Any] | DerivePyramidJob,
    *,
    convert_fn: ConvertFn = derive_pyramid,
    meta_fn: MetaFn | None = None,
) -> dict[str, Any]:
    """Run a derive-pyramid job and return a completion-event payload.

    ``convert_fn`` and ``meta_fn`` are injectable so this is testable without the
    engine. Raises whatever ``convert_fn`` raises (the worker maps that to a
    failed job); metadata extraction is best-effort.
    """
    spec_job = job if isinstance(job, DerivePyramidJob) else DerivePyramidJob.from_dict(job)
    # Read the source metadata once (best-effort) — it drives both the native-pyramid
    # skip and the fmt="auto" container choice.
    src_meta: dict[str, Any] | None = None
    if meta_fn is not None:
        try:
            src_meta = dict(meta_fn(spec_job.src_path))
        except Exception:  # noqa: BLE001 - metadata is best-effort; fall through and convert
            src_meta = None
    # Native-pyramid fast path: a source that is already a tiled multi-resolution
    # pyramid is served tile-by-tile directly, so skip the (potentially huge) convert.
    if _source_is_native_tiled_pyramid(src_meta):
        return {
            "resource_id": spec_job.resource_id,
            "derived_path": None,
            "status": "skipped_native_pyramid",
        }
    # fmt="auto": pick the output container from the SOURCE dimensionality. A
    # z-stack/volume must keep its Z planes — plain BigTIFF flattens a multi-channel
    # OME hyperstack to one plane, so volumes derive to OME-BigTIFF (Z preserved,
    # pyramidal => fast bounded /slice + /atlas reads). Flat 2D slides stay BigTIFF,
    # because the OME wrapper breaks this engine's embedded -tile reader (and 2D
    # serving is tile-based). Without a real engine (no meta), fall back to BigTIFF.
    if spec_job.fmt == "auto":
        spec_job.fmt = _resolve_auto_fmt(src_meta)
    convert_fn(spec_job.src_path, spec_job.dst_path, spec=spec_job.spec())
    result: dict[str, Any] = {
        "resource_id": spec_job.resource_id,
        "derived_path": spec_job.dst_path,
        "status": "succeeded",
        "tile_size": spec_job.tile_size,
        "compression": spec_job.compression,
        "layout": spec_job.layout,
        "fmt": spec_job.fmt,
    }
    if meta_fn is not None:
        try:
            meta = meta_fn(spec_job.dst_path)
            result["levels"] = meta.get("image_num_resolution_levels")
            result["scales"] = meta.get("image_res_l_scales")
            result["num_x"] = meta.get("image_num_x")
            result["num_y"] = meta.get("image_num_y")
        except Exception as exc:  # noqa: BLE001 - metadata is best-effort
            result["meta_warning"] = repr(exc)
    return result
