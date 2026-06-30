"""The ``image.derive_pyramid`` batch job: convert a source into a tiled pyramid.

This is the batch-processing half of the engine work. A job names a source image
and a destination; the runner invokes the tested convert primitive
(:func:`~ultra_deepagents.imaging.convert.derive_pyramid`) and reports the derived
artifact plus its pyramid metadata. The runner is pure and injectable so it is
unit-tested without the native engine; the NATS worker (`worker.py`) wraps it.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable

from ultra_deepagents.imaging.convert import ConvertResult, PyramidSpec, derive_pyramid
from ultra_deepagents.imaging.transcode import TranscodeResult, transcode_to_ome_tiff

__all__ = ["DerivePyramidJob", "run_derive_pyramid_job", "ConvertFn", "MetaFn", "TranscodeFn"]

ConvertFn = Callable[..., ConvertResult]
MetaFn = Callable[[str], dict[str, Any]]
TranscodeFn = Callable[..., TranscodeResult]


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
    # A z-stack, a TIME-LAPSE (t>1), or a paged stack all need OME-BigTIFF so the extra
    # dimension survives — plain BigTIFF flattens it to one plane (a time-lapse would
    # otherwise collapse to a single frame).
    is_volume = z > 1 or t > 1 or (pages > 1 and c <= 1 and t <= 1)
    return "ome-bigtiff" if is_volume else "bigtiff"


def _has_pixel_geometry(meta: dict[str, Any] | None) -> bool:
    """Whether the engine actually decoded the source (reported a real canvas). Empty
    meta / a 0-wide canvas means libbioimage recognized the container but cannot decode
    it (a Leica .lif, etc.) -> the bioio transcode fallback handles it."""
    return bool(meta) and _meta_int(meta, "image_num_x", 0) >= 1


def _safe_remove(path: str | None) -> None:
    if not path:
        return
    try:
        os.remove(path)
    except OSError:
        pass


def _extension_of(path: str) -> str:
    """Lowercase file extension without the dot; tolerates series/page-suffixed names
    (e.g. ``scan.czi_3`` -> ``czi``)."""
    base = os.path.basename(path).lower()
    match = re.search(r"\.([a-z0-9]+)(?:_\d+)?$", base)
    return match.group(1) if match else ""


def _prefer_bioio_extensions() -> frozenset[str]:
    """Extensions ALWAYS routed through the bioio transcode, even when libbioimage could
    decode them — bioio assembles these better (Zeiss .czi mosaics/scenes, which
    libbioimage rendered blocky/unstitched) or reads them at all (Leica .lif, DeltaVision
    .dv). It is a SOFT preference: if bioio can't read a given file, the runner falls back
    to libbioimage. Override with ULTRA_IMGSVC_PREFER_BIOIO_EXTS (comma-separated, e.g.
    "czi,lif,nd2")."""
    raw = os.environ.get("ULTRA_IMGSVC_PREFER_BIOIO_EXTS")
    if raw is not None:
        return frozenset(e.strip().lstrip(".").lower() for e in raw.split(",") if e.strip())
    # .zarr is intentionally NOT here: OME-Zarr is served natively by the ngff-service straight
    # from its directory bundle (see the special-format / ngff routing), so it never enters the
    # bioio transcode lane at all — it needs no convert-to-pyramid step.
    return frozenset({"czi"})


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
    transcode_fn: TranscodeFn = transcode_to_ome_tiff,
) -> dict[str, Any]:
    """Run a derive-pyramid job and return a completion-event payload.

    ``convert_fn``, ``meta_fn`` and ``transcode_fn`` are injectable so this is testable
    without the engine. Raises whatever ``convert_fn``/``transcode_fn`` raises (the
    worker maps that to a failed job); metadata extraction is best-effort.
    """
    spec_job = job if isinstance(job, DerivePyramidJob) else DerivePyramidJob.from_dict(job)
    # Some formats are PREFERENTIALLY read by bioio even though libbioimage could decode
    # them (Zeiss .czi mosaics, OME-Zarr) — those skip the libbioimage meta/native-pyramid
    # path and go straight to the bioio transcode.
    prefer_bioio = _extension_of(spec_job.src_path) in _prefer_bioio_extensions()
    # Read the source metadata once (best-effort) — it drives the native-pyramid skip,
    # the bioio fallback, and the fmt="auto" container choice. Skipped when preferring
    # bioio (we won't serve libbioimage's read of the source).
    src_meta: dict[str, Any] | None = None
    # Whether meta_fn RAN (returned, didn't raise). Only a successful-but-empty result
    # means "engine recognized but can't decode" (-> transcode); a meta_fn exception is
    # a transient/unexpected error and must fall through to a normal convert, as before.
    meta_ran = False
    if meta_fn is not None and not prefer_bioio:
        try:
            src_meta = dict(meta_fn(spec_job.src_path))
            meta_ran = True
        except Exception:  # noqa: BLE001 - metadata is best-effort; fall through and convert
            src_meta = None
    # Native-pyramid fast path: a source that is already a tiled multi-resolution
    # pyramid is served tile-by-tile directly, so skip the (potentially huge) convert.
    if not prefer_bioio and _source_is_native_tiled_pyramid(src_meta):
        return {
            "resource_id": spec_job.resource_id,
            "derived_path": None,
            "status": "skipped_native_pyramid",
        }
    # bioio transcode: convert the source to an intermediate OME-TIFF via bioio (pure-
    # Python plugins, off the serving path), then pyramid THAT — the pyramid, not bioio,
    # serves tiles. Triggered when (a) we PREFER bioio for this format, or (b) libbioimage
    # recognized the container but reported no pixel geometry (a .lif and other
    # registered-but-non-functional readers).
    convert_src = spec_job.src_path
    intermediate_path: str | None = None
    transcode: TranscodeResult | None = None
    if prefer_bioio or (meta_ran and not _has_pixel_geometry(src_meta)):
        intermediate_path = spec_job.dst_path + ".transcode.ome.tif"
        try:
            transcode = transcode_fn(spec_job.src_path, intermediate_path)
        except Exception:  # noqa: BLE001
            # PREFER is soft: if bioio can't read it, fall back to a normal libbioimage
            # convert of the source (don't discard a working native render). The
            # undecodable case has NO native render, so re-raise -> failure marker.
            _safe_remove(intermediate_path)
            intermediate_path = None
            if not prefer_bioio:
                raise
            if meta_fn is not None:
                try:
                    src_meta = dict(meta_fn(spec_job.src_path))
                except Exception:  # noqa: BLE001
                    src_meta = None
        if transcode is not None:
            convert_src = intermediate_path
            # Re-read the readable intermediate so fmt="auto" + meta reflect real geometry.
            if meta_fn is not None:
                try:
                    src_meta = dict(meta_fn(convert_src))
                except Exception:  # noqa: BLE001 - best-effort; multichannel/volume hint below still guides fmt
                    src_meta = None
            # A transcoded multichannel or z-stack series must stay OME-BigTIFF so its
            # channels/planes survive (plain BigTIFF flattens them to one plane).
            if spec_job.fmt in ("auto", "bigtiff") and transcode.is_multichannel_or_volume:
                spec_job.fmt = "ome-bigtiff"
    # fmt="auto": pick the output container from the SOURCE dimensionality. A
    # z-stack/volume must keep its Z planes — plain BigTIFF flattens a multi-channel
    # OME hyperstack to one plane, so volumes derive to OME-BigTIFF (Z preserved,
    # pyramidal => fast bounded /slice + /atlas reads). Flat 2D slides stay BigTIFF,
    # because the OME wrapper breaks this engine's embedded -tile reader (and 2D
    # serving is tile-based). Without a real engine (no meta), fall back to BigTIFF.
    if spec_job.fmt == "auto":
        spec_job.fmt = _resolve_auto_fmt(src_meta)
    try:
        convert_fn(convert_src, spec_job.dst_path, spec=spec_job.spec())
    finally:
        # The intermediate OME-TIFF is redundant once the pyramid exists — reclaim disk.
        _safe_remove(intermediate_path)
    result: dict[str, Any] = {
        "resource_id": spec_job.resource_id,
        "derived_path": spec_job.dst_path,
        "status": "succeeded",
        "tile_size": spec_job.tile_size,
        "compression": spec_job.compression,
        "layout": spec_job.layout,
        "fmt": spec_job.fmt,
    }
    if transcode is not None:
        # Surface the bioio provenance so the control plane/UI can note "converted from
        # series N of M" — the other series of a multi-series source aren't lost, just
        # not the one rendered.
        result["source_reader"] = "bioio"
        result["transcoded"] = True
        result["series_count"] = transcode.series_count
        result["series_index"] = transcode.series_index
        result["series_name"] = transcode.series_name
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
