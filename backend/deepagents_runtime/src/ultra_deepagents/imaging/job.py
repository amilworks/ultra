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
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from ultra_deepagents.imaging.convert import (
    ConversionDependencyError,
    ConversionInputError,
    ConversionProcessError,
    ConversionResourceError,
    ConvertResult,
    PyramidSpec,
    derive_pyramid,
)
from ultra_deepagents.imaging.derivative_manifest import (
    DerivativeJobError,
    DeterministicDerivativeError,
    TransientDerivativeError,
    ViewerInfoFn,
    run_strict_publication,
)
from ultra_deepagents.imaging.transcode import (
    TranscodeDependencyError,
    TranscodeError,
    TranscodeInputError,
    TranscodeOperationalError,
    TranscodeResourceError,
    TranscodeResult,
    transcode_to_ome_tiff,
)

__all__ = ["ConvertFn", "DerivePyramidJob", "MetaFn", "TranscodeFn", "run_derive_pyramid_job"]

ConvertFn = Callable[..., ConvertResult]
MetaFn = Callable[[str], dict[str, Any]]
TranscodeFn = Callable[..., TranscodeResult]
NATIVE_PYRAMID_TERMINAL_STATUS = "skipped_native_pyramid_no_manifest"


@dataclass
class DerivePyramidJob:
    resource_id: str
    src_path: str
    dst_path: str
    tile_size: int = 512
    compression: str = "lzw"
    layout: str = "topdirs"
    fmt: str = "bigtiff"
    source_sha256: str | None = None
    source_size_bytes: int | None = None
    force: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DerivePyramidJob:
        return cls(
            resource_id=str(payload["resource_id"]),
            src_path=str(payload["src_path"]),
            dst_path=str(payload["dst_path"]),
            tile_size=int(payload.get("tile_size", 512)),
            compression=str(payload.get("compression", "lzw")),
            layout=str(payload.get("layout", "topdirs")),
            fmt=str(payload.get("fmt", "bigtiff")),
            source_sha256=(
                str(payload["source_sha256"]) if payload.get("source_sha256") is not None else None
            ),
            source_size_bytes=(
                int(payload["source_size_bytes"])
                if payload.get("source_size_bytes") is not None
                else None
            ),
            force=payload.get("force", False) is True,
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
    if not meta:
        return False
    return _meta_int(meta, "image_num_x", 0) >= 1


def _safe_remove(path: str | None) -> None:
    if not path:
        return
    with suppress(OSError):
        os.remove(path)


def _extension_of(path: str) -> str:
    """Lowercase file extension without the dot; tolerates series/page-suffixed names
    (e.g. ``scan.czi_3`` -> ``czi``)."""
    base = os.path.basename(path).lower()
    match = re.search(r"\.([a-z0-9]+)(?:_\d+)?$", base)
    return match.group(1) if match else ""


def _prefer_bioio_extensions() -> frozenset[str]:
    """Proprietary microscopy formats ALWAYS routed through the bioio transcode (pure-Python
    readers), even when libbioimage nominally recognizes them — bioio is the more reliable
    reader here: it assembles Zeiss .czi mosaics/scenes (libbioimage rendered them
    blocky/unstitched), reads Nikon .nd2 (no libbioimage reader at all), and reads Leica
    .lif / DeltaVision .dv/.r3d that libbioimage handles poorly or not at all. Routing them
    here is DETERMINISTIC — it skips the wasted libbioimage meta probe that just returns
    empty for these — and the matching bioio plugin is baked into the image
    (bioio-czi/-nd2/-lif/-dv). Still a SOFT preference: if bioio cannot read a given file the
    runner falls back to libbioimage. Override with ULTRA_IMGSVC_PREFER_BIOIO_EXTS
    (comma-separated, e.g. "czi,nd2,lif,dv").

    NOT here: plain TIFF/OME-TIFF/SVS slides (libbioimage decodes those natively; bioio would
    be pure redundancy) and .zarr — OME-Zarr is served NATIVELY by the ngff-service straight
    from its directory bundle (native chunk serving, never transcode; see the special-format /
    ngff routing), so it never enters the bioio pyramid lane at all.
    """
    raw = os.environ.get("ULTRA_IMGSVC_PREFER_BIOIO_EXTS")
    if raw is not None:
        return frozenset(e.strip().lstrip(".").lower() for e in raw.split(",") if e.strip())
    return frozenset({"czi", "nd2", "lif", "dv", "r3d"})


def _source_is_native_tiled_pyramid(meta: dict[str, Any] | None) -> bool:
    """Whether the source ALREADY exposes a tiled multi-resolution pyramid the image
    service can serve tiles from directly — making a derived pyramid redundant (e.g. a
    4.7GB orthomosaic with 8 native levels + 256px tiles otherwise spawns a 21GB copy).
    Borrowed from BisQue's native-pyramid fast path."""
    if not meta:
        return False
    return (
        _meta_int(meta, "image_num_resolution_levels", 1) > 1
        and _meta_int(meta, "tile_num_x", 0) > 0
    )


def _run_derive_pyramid_job_legacy(
    job: dict[str, Any] | DerivePyramidJob,
    *,
    convert_fn: ConvertFn = derive_pyramid,
    meta_fn: MetaFn | None = None,
    transcode_fn: TranscodeFn = transcode_to_ome_tiff,
    strict_publication: bool = False,
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
        except Exception:  # Metadata is best-effort; fall through and convert.
            src_meta = None
    # Native-pyramid fast path: a source that is already a tiled multi-resolution
    # pyramid is served tile-by-tile directly, so skip the (potentially huge) convert.
    if not prefer_bioio and _source_is_native_tiled_pyramid(src_meta):
        return {
            "resource_id": spec_job.resource_id,
            "derived_path": None,
            "status": NATIVE_PYRAMID_TERMINAL_STATUS,
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
            transcode = transcode_fn(
                spec_job.src_path,
                intermediate_path,
                prefer="first" if strict_publication else "largest",
            )
        except DerivativeJobError:
            raise
        except (TranscodeDependencyError, TranscodeOperationalError, TranscodeResourceError) as exc:
            _safe_remove(intermediate_path)
            intermediate_path = None
            raise TransientDerivativeError("transcode_unavailable") from exc
        except TranscodeInputError as exc:
            # PREFER is soft: if bioio can't read it, fall back to a normal libbioimage
            # convert of the source (don't discard a working native render). The
            # undecodable case has NO native render, so re-raise -> failure marker.
            _safe_remove(intermediate_path)
            intermediate_path = None
            if not prefer_bioio or strict_publication:
                raise DeterministicDerivativeError("unsupported_source") from exc
            if meta_fn is not None:
                try:
                    src_meta = dict(meta_fn(spec_job.src_path))
                except Exception:
                    src_meta = None
        except TranscodeError as exc:
            _safe_remove(intermediate_path)
            intermediate_path = None
            raise TransientDerivativeError("transcode_unavailable") from exc
        if transcode is not None:
            if intermediate_path is None:
                raise RuntimeError("transcode completed without an intermediate artifact path")
            convert_src = intermediate_path
            # Re-read the readable intermediate so fmt="auto" + meta reflect real geometry.
            if meta_fn is not None:
                try:
                    src_meta = dict(meta_fn(convert_src))
                except Exception:  # Best-effort; transcode geometry still guides format.
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
        if not strict_publication:
            convert_fn(convert_src, spec_job.dst_path, spec=spec_job.spec())
        else:
            try:
                convert_fn(convert_src, spec_job.dst_path, spec=spec_job.spec())
            except DerivativeJobError:
                raise
            except ConversionInputError as exc:
                raise DeterministicDerivativeError("conversion_rejected") from exc
            except (
                ConversionDependencyError,
                ConversionProcessError,
                ConversionResourceError,
            ) as exc:
                raise TransientDerivativeError("conversion_unavailable") from exc
            except ValueError as exc:
                raise DeterministicDerivativeError("invalid_conversion_spec") from exc
            except Exception as exc:
                raise TransientDerivativeError("conversion_unavailable") from exc
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
        except Exception as exc:  # Metadata is best-effort.
            result["meta_warning"] = repr(exc)
    return result


def run_derive_pyramid_job(
    job: dict[str, Any] | DerivePyramidJob,
    *,
    convert_fn: ConvertFn = derive_pyramid,
    meta_fn: MetaFn | None = None,
    transcode_fn: TranscodeFn = transcode_to_ome_tiff,
    viewer_info_fn: ViewerInfoFn | None = None,
    source_viewer_info_fn: ViewerInfoFn | None = None,
    require_manifest: bool = False,
) -> dict[str, Any]:
    """Run a derive job, using strict source-bound publication when identity is present.

    ``require_manifest`` is used by the production worker to reject old queue payloads
    that cannot prove which catalog generation they were derived from. The legacy path
    remains solely for the existing direct runner API and its isolated unit tests.
    """
    spec_job = job if isinstance(job, DerivePyramidJob) else DerivePyramidJob.from_dict(job)
    source_sha256 = spec_job.source_sha256
    source_size_bytes = spec_job.source_size_bytes
    if source_sha256 is None or source_size_bytes is None:
        if require_manifest:
            raise ValueError("derive job is missing source_sha256 or source_size_bytes")
        return _run_derive_pyramid_job_legacy(
            spec_job, convert_fn=convert_fn, meta_fn=meta_fn, transcode_fn=transcode_fn
        )
    if viewer_info_fn is None:
        raise TransientDerivativeError("viewer_info_unavailable")
    prefer_bioio = _extension_of(spec_job.src_path) in _prefer_bioio_extensions()

    def produce(temp_path: str) -> dict[str, Any]:
        temp_job = DerivePyramidJob(
            resource_id=spec_job.resource_id,
            src_path=spec_job.src_path,
            dst_path=temp_path,
            tile_size=spec_job.tile_size,
            compression=spec_job.compression,
            layout=spec_job.layout,
            fmt=spec_job.fmt,
            source_sha256=spec_job.source_sha256,
            source_size_bytes=spec_job.source_size_bytes,
            force=spec_job.force,
        )
        return _run_derive_pyramid_job_legacy(
            temp_job,
            convert_fn=convert_fn,
            meta_fn=meta_fn,
            transcode_fn=transcode_fn,
            strict_publication=True,
        )

    return run_strict_publication(
        resource_id=spec_job.resource_id,
        src_path=spec_job.src_path,
        dst_path=spec_job.dst_path,
        source_sha256=source_sha256,
        source_size_bytes=source_size_bytes,
        viewer_info_fn=viewer_info_fn,
        source_viewer_info_fn=(source_viewer_info_fn if prefer_bioio else viewer_info_fn),
        source_reader=("bioio" if prefer_bioio else "libbioimage"),
        conversion_spec={
            "tile_size": spec_job.tile_size,
            "compression": spec_job.compression,
            "layout": spec_job.layout,
            "fmt": spec_job.fmt,
        },
        produce_fn=produce,
        force=spec_job.force,
    )
