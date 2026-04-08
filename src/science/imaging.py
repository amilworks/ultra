"""Universal scientific image loading and preview generation.

This module provides a single data-layer entrypoint for microscopy / medical
formats via bioio, with lightweight fallback for common 2D images.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import ExifTags, Image

from src.config import get_settings
from src.science.derivatives import get_cached_file_derivative

ArrayMode = Literal["plane", "volume", "tczyx"]

_NON_IMAGE_SUFFIX_PATTERNS = (
    ".txt",
    ".csv",
    ".tsv",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".md",
    ".rst",
    ".toml",
    ".ini",
    ".cfg",
    ".py",
    ".sh",
    ".env",
    ".pdf",
    ".h5",
    ".hdf5",
    ".he5",
    ".h5ebsd",
    ".dream3d",
)

_ORDINARY_DISPLAY_IMAGE_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".webp",
)

_FILENAME_CHANNEL_TOKENS = {
    "dapi",
    "gfp",
    "rfp",
    "cy3",
    "cy5",
    "fitc",
    "tritc",
    "hoechst",
    "brightfield",
    "phase",
}

_EXIF_TAG_ALLOWLIST = {
    "DateTime",
    "DateTimeOriginal",
    "Software",
    "Make",
    "Model",
    "ImageDescription",
    "Artist",
    "Copyright",
}

_HEADER_INFO_ALLOWLIST = {
    "compression": "Compression",
    "dpi": "DPI",
    "gamma": "Gamma",
}

_ORIENTATION_OPPOSITES = {
    "R": "L",
    "L": "R",
    "A": "P",
    "P": "A",
    "S": "I",
    "I": "S",
    "H": "F",
    "F": "H",
}


def _extract_filename_hints(source: Path) -> dict[str, Any]:
    stem = str(source.stem or "").strip()
    if not stem:
        return {}

    tokens = [token for token in re.split(r"[\s._-]+", stem) if token]
    lowered_tokens = [token.lower() for token in tokens]

    date_match = re.search(r"\b(20\d{2}[-_]?[01]\d[-_]?[0-3]\d)\b", stem)
    magnification_match = re.search(
        r"(?:^|[_\-\s])(\d{1,3})x(?:$|[_\-\s])",
        f"_{stem.lower()}_",
    )
    z_match = re.search(r"(?:^|[_-])z(\d{1,4})(?:$|[_-])", f"_{stem.lower()}_")
    t_match = re.search(r"(?:^|[_-])t(\d{1,4})(?:$|[_-])", f"_{stem.lower()}_")
    c_match = re.search(r"(?:^|[_-])c(\d{1,4})(?:$|[_-])", f"_{stem.lower()}_")
    detected_channels = sorted({token for token in lowered_tokens if token in _FILENAME_CHANNEL_TOKENS})

    hints: dict[str, Any] = {"tokens": tokens[:16]}
    if date_match:
        hints["acquisition_date_hint"] = date_match.group(1).replace("_", "-")
    if magnification_match:
        hints["magnification_hint"] = f"{magnification_match.group(1)}x"
    if z_match:
        hints["z_index_hint"] = int(z_match.group(1))
    if t_match:
        hints["t_index_hint"] = int(t_match.group(1))
    if c_match:
        hints["c_index_hint"] = int(c_match.group(1))
    if detected_channels:
        hints["channel_hints"] = detected_channels
    return hints


def _safe_scalar_exif_value(value: Any, *, max_length: int = 256) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) > max_length:
        return text[: max(1, max_length - 1)].rstrip() + "…"
    return text


def _extract_exif_metadata(source: Path) -> dict[str, str]:
    try:
        with Image.open(source) as image:
            exif = image.getexif()
    except Exception:
        return {}

    if not exif:
        return {}

    tags = getattr(ExifTags, "TAGS", {})
    output: dict[str, str] = {}
    for key, raw_value in dict(exif).items():
        tag_name = str(tags.get(key) or key)
        if tag_name not in _EXIF_TAG_ALLOWLIST:
            continue
        value = _safe_scalar_exif_value(raw_value)
        if value:
            output[tag_name] = value
    return output


def _exif_number_to_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        pass
    numerator = getattr(value, "numerator", None)
    denominator = getattr(value, "denominator", None)
    if numerator is not None and denominator not in (None, 0):
        try:
            return float(numerator) / float(denominator)
        except Exception:
            return None
    if isinstance(value, tuple) and len(value) == 2:
        try:
            denominator = float(value[1])
            if denominator == 0:
                return None
            return float(value[0]) / denominator
        except Exception:
            return None
    return None


def _gps_coordinate_to_decimal(values: Any, ref: str | None) -> float | None:
    if not isinstance(values, (list, tuple)) or len(values) < 3:
        return None
    degrees = _exif_number_to_float(values[0])
    minutes = _exif_number_to_float(values[1])
    seconds = _exif_number_to_float(values[2])
    if degrees is None or minutes is None or seconds is None:
        return None
    decimal = float(degrees) + float(minutes) / 60.0 + float(seconds) / 3600.0
    hemisphere = str(ref or "").strip().upper()
    if hemisphere in {"S", "W"}:
        decimal *= -1.0
    return round(decimal, 8)


def _extract_geo_metadata(source: Path) -> dict[str, float | str]:
    try:
        with Image.open(source) as image:
            exif = image.getexif()
    except Exception:
        return {}
    if not exif:
        return {}
    gps_tag_id = next(
        (key for key, value in ExifTags.TAGS.items() if str(value) == "GPSInfo"),
        None,
    )
    gps_raw = None
    if gps_tag_id is not None:
        try:
            gps_raw = exif.get_ifd(gps_tag_id)  # type: ignore[attr-defined]
        except Exception:
            gps_raw = exif.get(gps_tag_id)
    if not isinstance(gps_raw, dict):
        return {}
    gps_tags = getattr(ExifTags, "GPSTAGS", {})
    gps_info = {
        str(gps_tags.get(key) or key): value
        for key, value in gps_raw.items()
    }
    latitude = _gps_coordinate_to_decimal(
        gps_info.get("GPSLatitude"),
        str(gps_info.get("GPSLatitudeRef") or "").strip() or None,
    )
    longitude = _gps_coordinate_to_decimal(
        gps_info.get("GPSLongitude"),
        str(gps_info.get("GPSLongitudeRef") or "").strip() or None,
    )
    if latitude is None or longitude is None:
        return {}
    geo: dict[str, float | str] = {
        "latitude": latitude,
        "longitude": longitude,
        "source": "exif_gps",
    }
    altitude = _exif_number_to_float(gps_info.get("GPSAltitude"))
    if altitude is not None:
        geo["altitude_m"] = round(float(altitude), 3)
    return geo


def _format_header_value(value: Any) -> str:
    if isinstance(value, (tuple, list)):
        parts = [_safe_scalar_exif_value(item, max_length=64) for item in value]
        parts = [part for part in parts if part]
        return " × ".join(parts)
    return _safe_scalar_exif_value(value, max_length=128)


def _extract_image_header_metadata(source: Path) -> dict[str, str]:
    try:
        with Image.open(source) as image:
            header: dict[str, str] = {}
            if getattr(image, "format", None):
                header["Format"] = str(image.format)
            if getattr(image, "mode", None):
                header["Color mode"] = str(image.mode)
            bits = getattr(image, "bits", None)
            if bits is not None:
                header["Bit depth"] = str(bits)
            frame_count = int(getattr(image, "n_frames", 1) or 1)
            if frame_count > 1:
                header["Frame count"] = str(frame_count)
            info = getattr(image, "info", {}) or {}
            for key, label in _HEADER_INFO_ALLOWLIST.items():
                if key not in info:
                    continue
                value = _format_header_value(info.get(key))
                if value:
                    header[label] = value
            return header
    except Exception:
        return {}


def _attach_source_metadata_hints(result: dict[str, Any], source: Path) -> dict[str, Any]:
    metadata = result.get("metadata")
    metadata_payload = dict(metadata) if isinstance(metadata, dict) else {}
    metadata_payload["filename_hints"] = _extract_filename_hints(source)
    existing_header = dict(metadata_payload.get("header") or {}) if isinstance(metadata_payload.get("header"), dict) else {}
    header = _extract_image_header_metadata(source)
    if header:
        metadata_payload["header"] = {**existing_header, **header}
    elif existing_header:
        metadata_payload["header"] = existing_header
    exif = _extract_exif_metadata(source)
    if exif:
        metadata_payload["exif"] = exif
    geo = _extract_geo_metadata(source)
    if geo:
        metadata_payload["geo"] = geo
    result["metadata"] = metadata_payload
    return result


def extract_actionable_image_metadata(
    *,
    file_path: str,
    output_root: str | None = None,
) -> dict[str, Any]:
    """Return lightweight metadata context suitable for preprocessing and routing."""

    loaded = load_scientific_image(
        file_path=str(file_path),
        array_mode="plane",
        generate_preview=False,
        save_array=False,
        include_array=False,
        output_root=output_root,
    )
    if not isinstance(loaded, dict) or not bool(loaded.get("success")):
        return {
            "success": False,
            "error": str((loaded or {}).get("error") or "metadata_unavailable"),
        }

    metadata = loaded.get("metadata") if isinstance(loaded.get("metadata"), dict) else {}
    axis_sizes = loaded.get("axis_sizes") if isinstance(loaded.get("axis_sizes"), dict) else {}
    dimensions = {
        str(axis): int(value)
        for axis, value in axis_sizes.items()
        if str(axis).strip() and isinstance(value, (int, float))
    }
    exif = dict(metadata.get("exif") or {}) if isinstance(metadata.get("exif"), dict) else {}
    geo = dict(metadata.get("geo") or {}) if isinstance(metadata.get("geo"), dict) else {}
    filename_hints = (
        dict(metadata.get("filename_hints") or {})
        if isinstance(metadata.get("filename_hints"), dict)
        else {}
    )
    actionable_insights: list[str] = []
    if dimensions.get("X") and dimensions.get("Y"):
        actionable_insights.append(
            f"image_size={int(dimensions['X'])}x{int(dimensions['Y'])}"
        )
    captured_at = str(exif.get("DateTimeOriginal") or exif.get("DateTime") or "").strip()
    if captured_at:
        actionable_insights.append(f"captured_at={captured_at}")
    if geo.get("latitude") is not None and geo.get("longitude") is not None:
        actionable_insights.append(
            f"gps={geo.get('latitude')},{geo.get('longitude')}"
        )
    channel_hints = filename_hints.get("channel_hints")
    if isinstance(channel_hints, list) and channel_hints:
        actionable_insights.append(
            "channel_hints=" + ",".join(str(item) for item in channel_hints[:6] if str(item).strip())
        )
    summary: dict[str, Any] = {
        "success": True,
        "reader": str(loaded.get("reader") or "").strip() or None,
        "dims_order": str(loaded.get("dims_order") or "").strip() or None,
        "array_shape": list(loaded.get("array_shape") or [])
        if isinstance(loaded.get("array_shape"), list)
        else [],
        "dimensions": dimensions,
        "header": dict(metadata.get("header") or {}) if isinstance(metadata.get("header"), dict) else {},
        "exif": exif,
        "geo": geo,
        "filename_hints": filename_hints,
        "actionable_insights": actionable_insights,
    }
    if captured_at:
        summary["captured_at"] = captured_at
    return summary


def _negative_orientation_label(label: str | None, fallback: str) -> str:
    safe = str(label or "").strip().upper()
    if safe in _ORIENTATION_OPPOSITES:
        return _ORIENTATION_OPPOSITES[safe]
    if safe:
        return f"-{safe}"
    return str(fallback).strip()


def _orientation_axis_entry(positive_label: str | None, fallback_axis: str) -> dict[str, str]:
    positive = str(positive_label or fallback_axis).strip().upper()
    if not positive:
        positive = str(fallback_axis).strip().upper() or "X"
    return {
        "positive": positive,
        "negative": _negative_orientation_label(positive, f"-{positive}"),
    }


def _default_orientation_axis_labels() -> dict[str, dict[str, str]]:
    return {
        "x": _orientation_axis_entry("X", "X"),
        "y": _orientation_axis_entry("Y", "Y"),
        "z": _orientation_axis_entry("Z", "Z"),
    }


def _extract_nifti_orientation_metadata(nii: Any) -> dict[str, Any]:
    import nibabel as nib  # type: ignore

    affine = np.asarray(getattr(nii, "affine", np.eye(4)), dtype=np.float64)
    axis_codes = tuple(str(code or "").upper() for code in nib.orientations.aff2axcodes(affine))
    while len(axis_codes) < 3:
        axis_codes = axis_codes + ("",)
    axis_labels = {
        "x": _orientation_axis_entry(axis_codes[0], "X"),
        "y": _orientation_axis_entry(axis_codes[1], "Y"),
        "z": _orientation_axis_entry(axis_codes[2], "Z"),
    }
    try:
        xyz_unit, time_unit = nii.header.get_xyzt_units()
    except Exception:
        xyz_unit, time_unit = None, None
    return {
        "frame": "patient",
        "source": "nifti-affine",
        "axis_labels": axis_labels,
        "axis_codes": [axis_labels["x"]["positive"], axis_labels["y"]["positive"], axis_labels["z"]["positive"]],
        "affine": affine.tolist(),
        "space_units": {
            "spatial": str(xyz_unit) if xyz_unit else None,
            "time": str(time_unit) if time_unit else None,
        },
    }


def _extract_bioio_microscopy_metadata(bio: Any) -> dict[str, Any]:
    output: dict[str, Any] = {}

    try:
        channel_names = [str(name) for name in list(getattr(bio, "channel_names", []) or []) if str(name).strip()]
    except Exception:
        channel_names = []
    if channel_names:
        output["channel_names"] = channel_names

    standard = getattr(bio, "standard_metadata", None)
    if standard is not None:
        for field_name in (
            "dimensions_present",
            "objective",
            "imaging_datetime",
            "binning",
            "position_index",
            "row",
            "column",
            "timelapse_interval",
            "total_time_duration",
        ):
            value = getattr(standard, field_name, None)
            if value in (None, "", (), []):
                continue
            output[field_name] = str(value) if field_name in {"objective", "imaging_datetime", "binning", "dimensions_present"} else value

    return output


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _clamp_index(idx: int | None, size: int, default: int = 0) -> int:
    if size <= 0:
        return 0
    if idx is None:
        idx = default
    idx = int(idx)
    return max(0, min(size - 1, idx))


def _dims_axis_size(dims: Any, axis: str, fallback_order: str, fallback_shape: tuple[int, ...]) -> int:
    axis = str(axis).upper()
    value = getattr(dims, axis, None)
    if isinstance(value, (int, np.integer)):
        return int(value)

    order = str(getattr(dims, "order", "") or fallback_order).upper()
    if axis in order:
        pos = order.index(axis)
        if 0 <= pos < len(fallback_shape):
            return int(fallback_shape[pos])
    return 1


def _canonicalize_dims_order(order: str) -> str:
    normalized: list[str] = []
    for token in str(order or "").upper():
        axis = "C" if token == "S" else token
        if axis not in {"T", "C", "Z", "Y", "X"}:
            continue
        if axis in normalized:
            continue
        normalized.append(axis)
    return "".join(normalized) or "TCZYX"


def _normalize_uint8(image: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    arr = np.asarray(image)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)

    lo = float(np.percentile(finite, lo_pct))
    hi = float(np.percentile(finite, hi_pct))
    if hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)

    scaled = (arr.astype(np.float32) - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _render_preview_image(plane: np.ndarray) -> np.ndarray:
    arr = np.asarray(plane)
    if arr.ndim == 2:
        return _normalize_uint8(arr)

    if arr.ndim == 3:
        # Treat C,Y,X (preferred from bioio), fallback to Y,X,C.
        if arr.shape[0] <= 4 and arr.shape[-1] > 4:
            ch = min(3, arr.shape[0])
            rgb = np.stack([_normalize_uint8(arr[i]) for i in range(ch)], axis=-1)
            if ch == 1:
                rgb = np.repeat(rgb, 3, axis=-1)
            elif ch == 2:
                rgb = np.concatenate([rgb, rgb[..., :1]], axis=-1)
            return rgb
        if arr.shape[-1] in (3, 4):
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            return _normalize_uint8(arr)

    # Last resort: flatten leading dims and take first slice.
    flat = arr.reshape((-1,) + arr.shape[-2:])
    return _normalize_uint8(flat[0])


def _stable_artifact_stem(file_path: str, scene: str | None, array_mode: str) -> str:
    p = Path(file_path).expanduser()
    stat = p.stat() if p.exists() else None
    wire = "|".join(
        [
            str(p.resolve()) if p.exists() else str(p),
            str(stat.st_size if stat else 0),
            str(stat.st_mtime_ns if stat else 0),
            str(scene or ""),
            str(array_mode),
        ]
    )
    digest = hashlib.sha256(wire.encode("utf-8")).hexdigest()[:12]
    return f"{p.stem}_{digest}"


def _array_order_for_mode(
    mode: ArrayMode,
    *,
    has_t: bool,
    has_c: bool,
    has_z: bool,
) -> str:
    if mode == "tczyx":
        return "TCZYX"
    if mode == "volume":
        if has_c and has_z:
            return "CZYX"
        if has_z:
            return "ZYX"
        if has_c:
            return "CYX"
        return "YX"
    # plane
    if has_c:
        return "CYX"
    return "YX"


def _preferred_bioio_reader(source: Path) -> Any | None:
    """Best-effort reader selection to avoid noisy plugin auto-probing."""
    name = source.name.lower()

    try:
        if name.endswith(".czi"):
            import bioio_czi  # type: ignore

            return bioio_czi.Reader
        if name.endswith(".nd2"):
            import bioio_nd2  # type: ignore

            return bioio_nd2.Reader
        if name.endswith(".lif"):
            import bioio_lif  # type: ignore

            return bioio_lif.Reader
        if name.endswith(".dv") or name.endswith(".r3d"):
            import bioio_dv  # type: ignore

            return bioio_dv.Reader
        if name.endswith(".ome.zarr") or name.endswith(".zarr"):
            import bioio_ome_zarr  # type: ignore

            return bioio_ome_zarr.Reader
        if name.endswith(".ome.tif") or name.endswith(".ome.tiff"):
            import bioio_ome_tiff  # type: ignore

            return bioio_ome_tiff.Reader
        if name.endswith(".tif") or name.endswith(".tiff"):
            import bioio_tifffile  # type: ignore

            return bioio_tifffile.Reader
        if name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")):
            import bioio_imageio  # type: ignore

            return bioio_imageio.Reader
    except Exception:
        return None
    return None


def _fallback_bioio_readers(source: Path) -> list[Any]:
    name = source.name.lower()
    readers: list[Any] = []
    try:
        if name.endswith(".ome.tif") or name.endswith(".ome.tiff") or name.endswith(".tif") or name.endswith(".tiff"):
            import bioio_tifffile  # type: ignore

            readers.append(bioio_tifffile.Reader)
    except Exception:
        pass
    try:
        if name.endswith(".ome.tif") or name.endswith(".ome.tiff"):
            import bioio_imageio  # type: ignore

            readers.append(bioio_imageio.Reader)
    except Exception:
        pass
    return readers


def _bioio_reader_candidates(source: Path) -> list[Any | None]:
    candidates: list[Any | None] = []
    preferred = _preferred_bioio_reader(source)
    if preferred is not None:
        candidates.append(preferred)
    for candidate in _fallback_bioio_readers(source):
        if candidate is not None and candidate not in candidates:
            candidates.append(candidate)
    if not candidates:
        candidates.append(None)
    return candidates


def _bioio_reader_label(reader: Any | None) -> str:
    if reader is None:
        return "auto"
    module_name = str(getattr(reader, "__module__", "") or "").strip()
    qual_name = str(getattr(reader, "__qualname__", getattr(reader, "__name__", "Reader")) or "Reader").strip()
    if module_name:
        return f"{module_name}.{qual_name}"
    return qual_name or "Reader"


def _is_gzip_encoded(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(2) == b"\x1f\x8b"
    except Exception:
        return False


def _nifti_cache_dir(file_path: str | Path) -> Path:
    source = Path(file_path).expanduser().resolve()
    return source.parent / ".viewer-cache" / "nifti-source"


def _nifti_cache_path(file_path: str | Path) -> Path:
    source = Path(file_path).expanduser().resolve()
    digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:24]
    return _nifti_cache_dir(source) / f"{digest}.nii"


def _ensure_uncompressed_nifti_path(source_path: Path) -> tuple[Path, str | None]:
    resolved = source_path.expanduser().resolve()
    lower_name = resolved.name.lower()
    if not lower_name.endswith(".nii.gz"):
        return resolved, None

    target_path = _nifti_cache_path(resolved)
    if target_path.exists():
        return target_path, "Using cached uncompressed NIfTI source for random-access viewing."

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(f"{target_path}.{os.getpid()}.tmp")
    if _is_gzip_encoded(resolved):
        with gzip.open(resolved, "rb") as src, temp_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        shutil.copyfile(resolved, temp_path)
    os.replace(str(temp_path), str(target_path))
    return target_path, "Prepared uncompressed NIfTI source for random-access viewing."


def _schedule_nifti_uncompressed_warmup(source_path: Path) -> None:
    resolved = source_path.expanduser().resolve()
    if not resolved.name.lower().endswith(".nii.gz"):
        return
    target_path = _nifti_cache_path(resolved)
    if target_path.exists():
        return

    def _warm() -> None:
        try:
            _ensure_uncompressed_nifti_path(resolved)
        except Exception:
            return

    threading.Thread(target=_warm, daemon=True).start()


def _selector_for_order(
    order: str,
    *,
    has_t: bool,
    has_c: bool,
    has_z: bool,
    t_index: int,
    c_index: int,
    z_index: int,
    channel_axis: str = "C",
) -> dict[str, int]:
    selectors: dict[str, int] = {}
    if has_t and "T" not in order:
        selectors["T"] = int(t_index)
    channel_axis = str(channel_axis or "C").upper()
    if has_c and channel_axis not in order:
        selectors[channel_axis] = int(c_index)
    if has_z and "Z" not in order:
        selectors["Z"] = int(z_index)
    return selectors


def _native_order_with_channel_axis(order: str, channel_axis: str) -> str:
    native_axis = str(channel_axis or "C").upper()
    if native_axis == "C":
        return str(order).upper()
    return str(order).upper().replace("C", native_axis)


def _save_preview(
    preview_plane: np.ndarray,
    *,
    stem: str,
    output_root: Path,
) -> str:
    preview_dir = output_root / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / f"{stem}.png"
    image_u8 = _render_preview_image(preview_plane)
    Image.fromarray(image_u8).save(preview_path)
    return str(preview_path)


def _save_array(
    array: np.ndarray,
    *,
    stem: str,
    output_root: Path,
    max_saved_array_bytes: int,
) -> tuple[str | None, str | None]:
    if int(array.nbytes) > int(max_saved_array_bytes):
        warning = (
            "Array artifact skipped because it exceeds max_saved_array_bytes "
            f"({int(array.nbytes)} > {int(max_saved_array_bytes)})."
        )
        return None, warning
    array_dir = output_root / "arrays"
    array_dir.mkdir(parents=True, exist_ok=True)
    array_path = array_dir / f"{stem}.npy"
    np.save(array_path, array)
    return str(array_path), None


def _load_with_pillow(
    file_path: str,
    *,
    output_root: Path,
    array_mode: ArrayMode,
    generate_preview: bool,
    save_array: bool,
    include_array: bool,
    max_inline_elements: int,
    max_saved_array_bytes: int,
    return_array: bool,
) -> dict[str, Any]:
    img = Image.open(file_path)
    if img.mode in {"RGBA", "CMYK", "P", "LA", "PA", "HSV", "YCbCr"}:
        img = img.convert("RGB")
    array = np.asarray(img)
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        # Convert to C,Y,X for consistency with bioio pathway.
        array_cyx = np.transpose(array[..., :3], (2, 0, 1))
    else:
        array_cyx = array

    stem = _stable_artifact_stem(file_path, scene=None, array_mode=array_mode)
    preview_path = _save_preview(array, stem=stem, output_root=output_root) if generate_preview else None

    array_path = None
    warning = None
    if save_array:
        array_path, warning = _save_array(
            np.asarray(array_cyx),
            stem=stem,
            output_root=output_root,
            max_saved_array_bytes=max_saved_array_bytes,
        )

    result: dict[str, Any] = {
        "success": True,
        "reader": "pillow-fallback",
        "file_path": str(Path(file_path).resolve()),
        "scene": None,
        "scenes": [],
        "dims_order": "CYX" if np.asarray(array_cyx).ndim == 3 else "YX",
        "axis_sizes": {
            "T": 1,
            "C": int(array_cyx.shape[0]) if np.asarray(array_cyx).ndim == 3 else 1,
            "Z": 1,
            "Y": int(array.shape[0]),
            "X": int(array.shape[1]),
        },
        "array_order": "CYX" if np.asarray(array_cyx).ndim == 3 else "YX",
        "array_shape": list(np.asarray(array_cyx).shape),
        "array_dtype": str(np.asarray(array_cyx).dtype),
        "preview_path": preview_path,
        "array_path": array_path,
        "selected_indices": {"T": 0, "C": 0, "Z": 0},
        "is_volume": False,
        "is_timeseries": False,
        "is_multichannel": bool(np.asarray(array_cyx).ndim == 3 and array_cyx.shape[0] > 1),
        "warnings": [warning] if warning else [],
    }

    if include_array and int(np.asarray(array_cyx).size) <= int(max_inline_elements):
        result["array"] = np.asarray(array_cyx).tolist()

    if return_array:
        result["_array"] = np.asarray(array_cyx)
    return result


def _is_nifti_path(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _is_tiff_path(path: Path) -> bool:
    name = path.name.lower()
    return (
        name.endswith(".tif")
        or name.endswith(".tiff")
        or name.endswith(".ome.tif")
        or name.endswith(".ome.tiff")
    )


def _looks_like_non_image_file(path: Path) -> bool:
    lower = path.name.lower()
    return any(lower.endswith(suffix) for suffix in _NON_IMAGE_SUFFIX_PATTERNS)


def _is_ordinary_display_image_path(path: Path) -> bool:
    lower = path.name.lower()
    return any(lower.endswith(suffix) for suffix in _ORDINARY_DISPLAY_IMAGE_SUFFIXES)


def _pillow_mode_dtype(mode: str) -> str:
    safe_mode = str(mode or "").upper()
    if safe_mode.startswith("I;16"):
        return "uint16"
    if safe_mode == "I":
        return "int32"
    if safe_mode == "F":
        return "float32"
    return "uint8"


def _probe_with_pillow(file_path: str) -> dict[str, Any]:
    source = Path(str(file_path)).expanduser()
    with Image.open(source) as opened:
        width, height = opened.size
        mode = str(opened.mode or "")
        bands = tuple(str(band) for band in tuple(opened.getbands() or ()))
    channel_count = len(bands) if bands else (4 if "A" in mode else 3 if mode in {"RGB", "YCBCR", "LAB", "HSV", "CMYK"} else 1)
    multichannel = channel_count > 1
    if multichannel:
        dims_order = "CYX"
        array_shape = [int(channel_count), int(height), int(width)]
    else:
        dims_order = "YX"
        array_shape = [int(height), int(width)]

    return {
        "success": True,
        "reader": "pillow-probe",
        "file_path": str(source.resolve()),
        "scene": None,
        "scenes": [],
        "dims_order": dims_order,
        "axis_sizes": {
            "T": 1,
            "C": int(channel_count if multichannel else 1),
            "Z": 1,
            "Y": int(height),
            "X": int(width),
        },
        "selected_indices": {"T": 0, "C": 0, "Z": 0},
        "array_mode": "plane",
        "array_order": dims_order,
        "array_shape": array_shape,
        "array_dtype": _pillow_mode_dtype(mode),
        "preview_path": None,
        "array_path": None,
        "physical_spacing": None,
        "is_volume": False,
        "is_timeseries": False,
        "is_multichannel": multichannel,
        "warnings": [],
        "metadata": {
            "header": {
                "Format": source.suffix.lstrip(".").upper() if source.suffix else "Image",
                "Color mode": mode or "unknown",
            }
        },
    }


def _infer_tiff_axes(shape: tuple[int, ...]) -> str:
    ndim = len(shape)
    if ndim == 2:
        return "YX"
    if ndim == 3:
        return "YXS" if shape[-1] in (3, 4) else "ZYX"
    if ndim == 4:
        return "ZYXS" if shape[-1] in (3, 4) else "TZYX"
    if ndim == 5:
        return "TZYXS" if shape[-1] in (3, 4) else "TCZYX"
    if ndim >= 2:
        return ("Q" * (ndim - 2)) + "YX"
    return ""


def _load_with_tifffile(
    file_path: str,
    *,
    output_root: Path,
    array_mode: ArrayMode,
    t_index: int | None,
    c_index: int | None,
    z_index: int | None,
    generate_preview: bool,
    save_array: bool,
    include_array: bool,
    max_inline_elements: int,
    max_saved_array_bytes: int,
    return_array: bool,
) -> dict[str, Any]:
    import tifffile  # type: ignore

    warnings: list[str] = []
    with tifffile.TiffFile(file_path) as tif:
        if not tif.series:
            raise ValueError("TIFF has no readable image series.")
        series = tif.series[0]
        raw = np.asarray(series.asarray())
        axes = str(getattr(series, "axes", "") or "").upper()

    if raw.ndim < 2:
        raise ValueError(f"TIFF array must be at least 2D, received shape {raw.shape}")

    if not axes or len(axes) != raw.ndim:
        axes = _infer_tiff_axes(tuple(int(v) for v in raw.shape))
        warnings.append(
            "TIFF axis metadata missing/ambiguous; inferred axis order "
            f"as '{axes or 'unknown'}'."
        )

    arr = np.asarray(raw)
    axis_chars = list(axes)
    for idx in range(len(axis_chars) - 1, -1, -1):
        axis_name = axis_chars[idx]
        if axis_name not in {"T", "C", "Z", "Y", "X", "S"}:
            arr = np.take(arr, 0, axis=idx)
            warnings.append(
                f"Ignored unsupported TIFF axis '{axis_name}' by selecting index 0."
            )
            axis_chars.pop(idx)

    if "S" in axis_chars and "C" not in axis_chars:
        axis_chars[axis_chars.index("S")] = "C"
    elif "S" in axis_chars:
        channel_axis = axis_chars.index("C")
        sample_axis = axis_chars.index("S")
        channel_size = int(arr.shape[channel_axis]) if channel_axis < arr.ndim else 1
        sample_size = int(arr.shape[sample_axis]) if sample_axis < arr.ndim else 1
        if channel_size <= 1 and sample_size > 1:
            arr = np.take(arr, 0, axis=channel_axis)
            axis_chars.pop(channel_axis)
            if sample_axis > channel_axis:
                sample_axis -= 1
            axis_chars[sample_axis] = "C"
            warnings.append(
                "TIFF contained singleton 'C' and non-singleton 'S'; using sample axis for color channels."
            )
        else:
            arr = np.take(arr, 0, axis=sample_axis)
            warnings.append("Dropped TIFF sample axis because channel axis was already present.")
            axis_chars.pop(sample_axis)

    if "Y" not in axis_chars or "X" not in axis_chars:
        raise ValueError(f"TIFF axes must contain Y and X dimensions (axes='{''.join(axis_chars)}').")

    present_indices = [axis_chars.index(ax) for ax in "TCZYX" if ax in axis_chars]
    extra_indices = [i for i in range(len(axis_chars)) if i not in present_indices]
    for idx in sorted(extra_indices, reverse=True):
        arr = np.take(arr, 0, axis=idx)
        warnings.append(
            f"Dropped extra TIFF axis '{axis_chars[idx]}' by selecting index 0."
        )
        axis_chars.pop(idx)

    for axis_name in "TCZYX":
        if axis_name not in axis_chars:
            arr = np.expand_dims(arr, axis=0)
            axis_chars.insert(0, axis_name)

    perm = [axis_chars.index(axis_name) for axis_name in "TCZYX"]
    tczyx = np.transpose(arr, perm)
    t_size, c_size, z_size, y_size, x_size = [int(v) for v in tczyx.shape]

    t_sel = _clamp_index(t_index, t_size, default=0)
    c_sel = _clamp_index(c_index, c_size, default=0)
    z_mid = (z_size // 2) if z_size > 1 else 0
    z_sel = _clamp_index(z_index, z_size, default=z_mid)

    if c_size > 1:
        preview_plane = tczyx[t_sel, : min(3, c_size), z_sel]
    else:
        preview_plane = tczyx[t_sel, 0, z_sel]

    chosen_mode: ArrayMode = array_mode if array_mode in ("plane", "volume", "tczyx") else "plane"
    if chosen_mode == "tczyx":
        array = tczyx
        array_order = "TCZYX"
    elif chosen_mode == "volume":
        if c_size > 1:
            array = tczyx[t_sel]
            array_order = "CZYX"
        else:
            array = tczyx[t_sel, 0]
            array_order = "ZYX"
    else:
        if c_size > 1:
            array = tczyx[t_sel, :, z_sel]
            array_order = "CYX"
        else:
            array = tczyx[t_sel, 0, z_sel]
            array_order = "YX"

    stem = _stable_artifact_stem(file_path, scene=None, array_mode=chosen_mode)
    preview_path = _save_preview(preview_plane, stem=stem, output_root=output_root) if generate_preview else None

    array_path = None
    if save_array:
        array_path, warn = _save_array(
            np.asarray(array),
            stem=stem,
            output_root=output_root,
            max_saved_array_bytes=max_saved_array_bytes,
        )
        if warn:
            warnings.append(warn)

    dims_order = "".join(
        axis_name
        for axis_name, axis_size in (
            ("T", t_size),
            ("C", c_size),
            ("Z", z_size),
            ("Y", y_size),
            ("X", x_size),
        )
        if axis_name in {"Y", "X"} or axis_size > 1
    )

    result: dict[str, Any] = {
        "success": True,
        "reader": "tifffile-fallback",
        "file_path": str(Path(file_path).resolve()),
        "scene": None,
        "scenes": [],
        "dims_order": dims_order or "TCZYX",
        "axis_sizes": {
            "T": int(t_size),
            "C": int(c_size),
            "Z": int(z_size),
            "Y": int(y_size),
            "X": int(x_size),
        },
        "selected_indices": {"T": int(t_sel), "C": int(c_sel), "Z": int(z_sel)},
        "array_mode": chosen_mode,
        "array_order": array_order,
        "array_shape": list(np.asarray(array).shape),
        "array_dtype": str(np.asarray(array).dtype),
        "array_min": float(np.min(array)) if np.asarray(array).size else 0.0,
        "array_max": float(np.max(array)) if np.asarray(array).size else 0.0,
        "preview_path": preview_path,
        "array_path": array_path,
        "physical_spacing": None,
        "is_volume": bool(z_size > 1),
        "is_timeseries": bool(t_size > 1),
        "is_multichannel": bool(c_size > 1),
        "warnings": warnings,
    }

    if include_array and int(np.asarray(array).size) <= int(max_inline_elements):
        result["array"] = np.asarray(array).tolist()

    if return_array:
        result["_array"] = np.asarray(array)
    return result


def _load_with_nibabel(
    file_path: str,
    *,
    output_root: Path,
    array_mode: ArrayMode,
    t_index: int | None,
    z_index: int | None,
    generate_preview: bool,
    save_array: bool,
    include_array: bool,
    max_inline_elements: int,
    max_saved_array_bytes: int,
    return_array: bool,
) -> dict[str, Any]:
    import nibabel as nib  # type: ignore

    source_path = Path(file_path).expanduser()
    load_warning: str | None = None
    chosen_mode: ArrayMode = array_mode if array_mode in ("plane", "volume", "tczyx") else "plane"
    header_probe_only = bool(
        chosen_mode == "plane"
        and not generate_preview
        and not save_array
        and not include_array
        and not return_array
    )
    try:
        load_path = source_path
        if header_probe_only and source_path.name.lower().endswith(".nii.gz"):
            _schedule_nifti_uncompressed_warmup(source_path)
        if source_path.name.lower().endswith(".nii.gz") and not header_probe_only:
            load_path, load_warning = _ensure_uncompressed_nifti_path(source_path)
        try:
            nii = nib.load(str(load_path))
        except Exception as load_error:
            message = str(load_error).lower()
            if source_path.name.lower().endswith(".nii.gz") and "not a gzip file" in message:
                load_path, load_warning = _ensure_uncompressed_nifti_path(source_path)
                nii = nib.load(str(load_path))
                if load_warning is None:
                    load_warning = (
                        "Input file was named .nii.gz but payload was not gzip-compressed; "
                        "loaded using persistent .nii fallback."
                    )
            else:
                raise

        source_shape = tuple(int(v) for v in tuple(getattr(nii, "shape", ()) or ()))
        if len(source_shape) < 3:
            raise ValueError(f"NIfTI array must be at least 3D, received shape {source_shape}")

        collapsed_warning = None
        x_size = int(source_shape[0])
        y_size = int(source_shape[1])
        z_size = int(source_shape[2])
        trailing = tuple(int(v) for v in source_shape[3:])
        t_size = int(np.prod(trailing)) if trailing else 1

        t_sel = _clamp_index(t_index, t_size, default=0)
        z_mid = (z_size // 2) if z_size > 1 else 0
        z_sel = _clamp_index(z_index, z_size, default=z_mid)

        if header_probe_only:
            array = None
            array_order = "YX"
            preview_plane = None
            array_shape_value = [int(y_size), int(x_size)]
            try:
                raw_min = float(getattr(nii.header, "get", lambda *_args, **_kwargs: 0.0)("cal_min", 0.0) or 0.0)
                raw_max = float(getattr(nii.header, "get", lambda *_args, **_kwargs: 0.0)("cal_max", 0.0) or 0.0)
            except Exception:
                raw_min = 0.0
                raw_max = 0.0
            if not np.isfinite(raw_min):
                raw_min = 0.0
            if not np.isfinite(raw_max):
                raw_max = 0.0
        elif chosen_mode == "plane":
            if len(source_shape) == 3:
                plane = np.asarray(nii.dataobj[:, :, z_sel])
            else:
                trailing_index = np.unravel_index(t_sel, trailing) if trailing else ()
                plane = np.asarray(nii.dataobj[(slice(None), slice(None), z_sel, *trailing_index)])
                if len(source_shape) > 4:
                    collapsed_warning = (
                        "NIfTI has >4 dimensions; trailing dimensions were flattened into T for processing."
                    )
            array = np.transpose(plane, (1, 0))
            array_order = "YX"
            preview_plane = np.asarray(array)
            array_shape_value = list(np.asarray(array).shape)
            raw_min = float(np.min(array)) if np.asarray(array).size else 0.0
            raw_max = float(np.max(array)) if np.asarray(array).size else 0.0
        else:
            raw = np.asarray(nii.get_fdata())
            if raw.ndim < 3:
                raise ValueError(f"NIfTI array must be at least 3D, received shape {raw.shape}")

            # NIfTI canonical index order is typically X,Y,Z,(T...). Convert to T,Z,Y,X.
            if raw.ndim == 3:
                tzyx = np.transpose(raw, (2, 1, 0))[None, ...]
            else:
                x, y, z = raw.shape[:3]
                trailing = raw.shape[3:]
                t_size = int(np.prod(trailing)) if trailing else 1
                if raw.ndim > 4:
                    collapsed_warning = (
                        "NIfTI has >4 dimensions; trailing dimensions were flattened into T for processing."
                    )
                reshaped = np.reshape(raw, (x, y, z, t_size))
                tzyx = np.transpose(reshaped, (3, 2, 1, 0))

            if chosen_mode == "tczyx":
                array = tzyx[:, None, :, :, :]
                array_order = "TCZYX"
            else:
                array = tzyx[t_sel]
                array_order = "ZYX"
            preview_plane = np.asarray(tzyx[t_sel, z_sel])
            array_shape_value = list(np.asarray(array).shape)
            raw_min = float(np.min(array)) if np.asarray(array).size else 0.0
            raw_max = float(np.max(array)) if np.asarray(array).size else 0.0

        stem = _stable_artifact_stem(file_path, scene=None, array_mode=chosen_mode)
        preview_path = (
            _save_preview(np.asarray(preview_plane), stem=stem, output_root=output_root)
            if generate_preview and preview_plane is not None
            else None
        )

        warnings: list[str] = []
        if load_warning:
            warnings.append(load_warning)
        if collapsed_warning:
            warnings.append(collapsed_warning)

        array_path = None
        if save_array:
            array_path, warn = _save_array(
                np.asarray(array),
                stem=stem,
                output_root=output_root,
                max_saved_array_bytes=max_saved_array_bytes,
            )
            if warn:
                warnings.append(warn)

        zooms = tuple(float(v) for v in tuple(getattr(nii.header, "get_zooms", lambda: ())() or ()))
        physical_spacing = None
        if len(zooms) >= 3:
            physical_spacing = {
                "z": float(zooms[2]),
                "y": float(zooms[1]),
                "x": float(zooms[0]),
            }
        orientation_metadata = _extract_nifti_orientation_metadata(nii)
        header_metadata: dict[str, Any] = {
            "Format": "NIfTI",
            "Orientation": "".join(orientation_metadata.get("axis_codes") or []),
        }
        try:
            qform_code = int(nii.header.get("qform_code", 0))
            if qform_code > 0:
                header_metadata["QForm code"] = str(qform_code)
        except Exception:
            pass
        try:
            sform_code = int(nii.header.get("sform_code", 0))
            if sform_code > 0:
                header_metadata["SForm code"] = str(sform_code)
        except Exception:
            pass

        result: dict[str, Any] = {
            "success": True,
            "reader": "nibabel-fallback",
            "file_path": str(source_path.resolve()),
            "scene": None,
            "scenes": [],
            "dims_order": "TZYX",
            "axis_sizes": {
                "T": int(t_size),
                "C": 1,
                "Z": int(z_size),
                "Y": int(y_size),
                "X": int(x_size),
            },
            "selected_indices": {"T": int(t_sel), "C": 0, "Z": int(z_sel)},
            "array_mode": chosen_mode,
            "array_order": array_order,
            "array_shape": array_shape_value,
            "array_dtype": str(getattr(nii, "get_data_dtype", lambda: np.asarray(array).dtype)()),
            "array_min": raw_min,
            "array_max": raw_max,
            "preview_path": preview_path,
            "array_path": array_path,
            "physical_spacing": physical_spacing,
            "is_volume": bool(z_size > 1),
            "is_timeseries": bool(t_size > 1),
            "is_multichannel": False,
            "warnings": warnings,
            "metadata": {
                "header": header_metadata,
                "orientation": {
                    "frame": orientation_metadata.get("frame"),
                    "source": orientation_metadata.get("source"),
                    "axis_labels": orientation_metadata.get("axis_labels"),
                    "axis_codes": orientation_metadata.get("axis_codes"),
                },
                "coordinates": {
                    "space": "patient",
                    "axis_codes": orientation_metadata.get("axis_codes"),
                    "affine": orientation_metadata.get("affine"),
                    "space_units": orientation_metadata.get("space_units"),
                },
            },
        }

        if include_array and int(np.asarray(array).size) <= int(max_inline_elements):
            result["array"] = np.asarray(array).tolist()

        if return_array and array is not None:
            result["_array"] = np.asarray(array)
        return result
    finally:
        pass


def _load_with_bioio(
    file_path: str,
    *,
    scene: int | str | None,
    use_aicspylibczi: bool,
    array_mode: ArrayMode,
    t_index: int | None,
    c_index: int | None,
    z_index: int | None,
    generate_preview: bool,
    save_array: bool,
    include_array: bool,
    max_inline_elements: int,
    max_saved_array_bytes: int,
    output_root: Path,
    return_array: bool,
) -> dict[str, Any]:
    from bioio import BioImage  # type: ignore

    source = Path(str(file_path)).expanduser()
    bio: Any | None = None
    reader_used: Any | None = None
    reader_errors: list[str] = []
    for candidate in _bioio_reader_candidates(source):
        kwargs: dict[str, Any] = {}
        if candidate is not None:
            kwargs["reader"] = candidate
        if use_aicspylibczi and source.name.lower().endswith(".czi"):
            kwargs["use_aicspylibczi"] = True
        try:
            bio = BioImage(str(source), **kwargs)
            reader_used = candidate
            break
        except Exception as exc:
            reader_errors.append(f"{_bioio_reader_label(candidate)}: {exc}")
            continue

    if bio is None:
        raise RuntimeError("bioio open failed via all candidate readers: " + " | ".join(reader_errors))

    scenes = [str(s) for s in list(getattr(bio, "scenes", []) or [])]
    if scene is not None:
        target_scene = str(scene)
        if isinstance(scene, int):
            try:
                bio.set_scene(int(scene))
            except Exception:
                bio.set_scene(target_scene)
        else:
            bio.set_scene(target_scene)
    current_scene = str(getattr(bio, "current_scene", "") or "") or None

    dims = getattr(bio, "dims", None)
    shape = tuple(_safe_int(v, 1) for v in tuple(getattr(bio, "shape", ()) or ()))
    native_dims_order = str(getattr(dims, "order", "") or "TCZYX").upper()

    t_size = _dims_axis_size(dims, "T", native_dims_order, shape)
    c_axis_size = _dims_axis_size(dims, "C", native_dims_order, shape)
    s_axis_size = _dims_axis_size(dims, "S", native_dims_order, shape)
    z_size = _dims_axis_size(dims, "Z", native_dims_order, shape)
    y_size = _dims_axis_size(dims, "Y", native_dims_order, shape)
    x_size = _dims_axis_size(dims, "X", native_dims_order, shape)

    has_c_axis = "C" in native_dims_order and c_axis_size > 0
    has_s_axis = "S" in native_dims_order and s_axis_size > 0
    if has_s_axis and (not has_c_axis or (c_axis_size <= 1 and s_axis_size > 1)):
        channel_axis = "S"
    elif has_c_axis:
        channel_axis = "C"
    elif has_s_axis:
        channel_axis = "S"
    else:
        channel_axis = "C"

    if channel_axis == "S":
        c_size = int(s_axis_size)
    elif channel_axis == "C":
        c_size = int(c_axis_size)
    else:
        c_size = max(int(c_axis_size), int(s_axis_size), 1)

    has_t = "T" in native_dims_order and t_size > 0
    has_c = channel_axis in native_dims_order and c_size > 0
    has_z = "Z" in native_dims_order and z_size > 0

    t_sel = _clamp_index(t_index, t_size, default=0)
    c_sel = _clamp_index(c_index, c_size, default=0)
    z_mid = (z_size // 2) if z_size > 1 else 0
    z_sel = _clamp_index(z_index, z_size, default=z_mid)

    if has_c:
        preview_order = _native_order_with_channel_axis("CYX", channel_axis)
        preview_select = _selector_for_order(
            preview_order,
            has_t=has_t,
            has_c=has_c,
            has_z=has_z,
            t_index=t_sel,
            c_index=c_sel,
            z_index=z_sel,
            channel_axis=channel_axis,
        )
        preview_plane = np.asarray(bio.get_image_data(preview_order, **preview_select))
    else:
        preview_order = "YX"
        preview_select = _selector_for_order(
            preview_order,
            has_t=has_t,
            has_c=has_c,
            has_z=has_z,
            t_index=t_sel,
            c_index=c_sel,
            z_index=z_sel,
            channel_axis=channel_axis,
        )
        preview_plane = np.asarray(bio.get_image_data(preview_order, **preview_select))

    chosen_mode: ArrayMode = array_mode if array_mode in ("plane", "volume", "tczyx") else "plane"
    canonical_array_order = _array_order_for_mode(chosen_mode, has_t=has_t, has_c=has_c, has_z=has_z)
    native_array_order = _native_order_with_channel_axis(canonical_array_order, channel_axis)
    array_select = _selector_for_order(
        native_array_order,
        has_t=has_t,
        has_c=has_c,
        has_z=has_z,
        t_index=t_sel,
        c_index=c_sel,
        z_index=z_sel,
        channel_axis=channel_axis,
    )
    array = np.asarray(bio.get_image_data(native_array_order, **array_select))

    stem = _stable_artifact_stem(str(source), current_scene, chosen_mode)
    preview_path = _save_preview(preview_plane, stem=stem, output_root=output_root) if generate_preview else None

    warnings: list[str] = []
    if reader_used is not None and reader_errors:
        warnings.append(
            "Primary bioio reader failed; recovered with fallback reader "
            f"{_bioio_reader_label(reader_used)}."
        )
    if channel_axis == "S":
        if c_axis_size <= 1 and s_axis_size > 1:
            warnings.append(
                "Detected sample axis 'S' with RGB-like channels; using it as the display channel axis."
            )
        else:
            warnings.append("Normalized sample axis 'S' to channel axis 'C' for consistent color handling.")

    array_path = None
    if save_array:
        array_path, warn = _save_array(
            array,
            stem=stem,
            output_root=output_root,
            max_saved_array_bytes=max_saved_array_bytes,
        )
        if warn:
            warnings.append(warn)

    pixel_sizes = getattr(bio, "physical_pixel_sizes", None)
    physical_spacing = None
    if pixel_sizes is not None:
        physical_spacing = {
            "z": float(getattr(pixel_sizes, "Z", 0.0) or 0.0),
            "y": float(getattr(pixel_sizes, "Y", 0.0) or 0.0),
            "x": float(getattr(pixel_sizes, "X", 0.0) or 0.0),
        }
    microscopy_metadata = _extract_bioio_microscopy_metadata(bio)
    if current_scene:
        microscopy_metadata.setdefault("current_scene", current_scene)
    if scenes:
        microscopy_metadata.setdefault("scene_names", scenes)

    canonical_dims_order = _canonicalize_dims_order(native_dims_order)
    result: dict[str, Any] = {
        "success": True,
        "reader": "bioio",
        "file_path": str(source.resolve()),
        "scene": current_scene,
        "scenes": scenes,
        "dims_order": canonical_dims_order,
        "native_dims_order": native_dims_order,
        "axis_sizes": {
            "T": int(t_size),
            "C": int(c_size),
            "Z": int(z_size),
            "Y": int(y_size),
            "X": int(x_size),
        },
        "selected_indices": {"T": int(t_sel), "C": int(c_sel), "Z": int(z_sel)},
        "array_mode": chosen_mode,
        "array_order": canonical_array_order,
        "native_array_order": native_array_order,
        "array_shape": list(array.shape),
        "array_dtype": str(array.dtype),
        "array_min": float(np.min(array)) if array.size else 0.0,
        "array_max": float(np.max(array)) if array.size else 0.0,
        "preview_path": preview_path,
        "array_path": array_path,
        "physical_spacing": physical_spacing,
        "is_volume": bool(z_size > 1),
        "is_timeseries": bool(t_size > 1),
        "is_multichannel": bool(c_size > 1),
        "warnings": warnings,
        "metadata": {
            "header": {
                "Format": source.suffix.lstrip(".").upper() if source.suffix else "Scientific image",
                "Dimensions": canonical_dims_order,
            },
            "orientation": {
                "frame": "voxel",
                "source": "bioio-dims",
                "axis_labels": _default_orientation_axis_labels(),
            },
            "microscopy": microscopy_metadata,
        },
    }

    if include_array and int(array.size) <= int(max_inline_elements):
        result["array"] = array.tolist()

    if return_array:
        result["_array"] = array
    return result


def load_scientific_image(
    *,
    file_path: str,
    scene: int | str | None = None,
    use_aicspylibczi: bool = False,
    array_mode: ArrayMode = "plane",
    t_index: int | None = None,
    c_index: int | None = None,
    z_index: int | None = None,
    generate_preview: bool = True,
    save_array: bool = True,
    include_array: bool = False,
    max_inline_elements: int = 16384,
    max_saved_array_bytes: int = 1_000_000_000,
    output_root: str | None = None,
    return_array: bool = False,
) -> dict[str, Any]:
    """Load scientific image data through bioio and return normalized metadata/artifacts.

    The returned payload is API-safe by default. For internal model calls, set
    `return_array=True` to include a private `_array` numpy key.
    """
    if not file_path:
        return {"success": False, "error": "file_path is required"}

    source = Path(str(file_path)).expanduser()
    if not source.exists() or not source.is_file():
        return {"success": False, "error": f"File not found: {source}"}
    if _looks_like_non_image_file(source):
        return {
            "success": False,
            "error": (
                f"Unsupported file type for scientific image loading: {source.name}. "
                "Provide an image/volume/video file."
            ),
        }

    settings = get_settings()
    root = Path(output_root or getattr(settings, "science_data_root", "data/science"))
    root.mkdir(parents=True, exist_ok=True)

    if _is_ordinary_display_image_path(source):
        try:
            result = _load_with_pillow(
                str(source),
                output_root=root,
                array_mode=array_mode,
                generate_preview=generate_preview,
                save_array=save_array,
                include_array=include_array,
                max_inline_elements=max_inline_elements,
                max_saved_array_bytes=max_saved_array_bytes,
                return_array=return_array,
            )
            if bool(result.get("success")):
                return _attach_source_metadata_hints(result, source)
            return result
        except Exception as pil_error:
            return {
                "success": False,
                "error": f"Failed to load image with Pillow fast path: {pil_error}",
            }

    nifti_error: Exception | None = None
    if _is_nifti_path(source):
        try:
            result = _load_with_nibabel(
                str(source),
                output_root=root,
                array_mode=array_mode,
                t_index=t_index,
                z_index=z_index,
                generate_preview=generate_preview,
                save_array=save_array,
                include_array=include_array,
                max_inline_elements=max_inline_elements,
                max_saved_array_bytes=max_saved_array_bytes,
                return_array=return_array,
            )
            if bool(result.get("success")):
                return _attach_source_metadata_hints(result, source)
            return result
        except Exception as nib_exc:
            nifti_error = nib_exc

    bioio_error: Exception | None = None
    try:
        bioio_result = _load_with_bioio(
            str(source),
            scene=scene,
            use_aicspylibczi=use_aicspylibczi,
            array_mode=array_mode,
            t_index=t_index,
            c_index=c_index,
            z_index=z_index,
            generate_preview=generate_preview,
            save_array=save_array,
            include_array=include_array,
            max_inline_elements=max_inline_elements,
            max_saved_array_bytes=max_saved_array_bytes,
            output_root=root,
            return_array=return_array,
        )
        if bool(bioio_result.get("success")):
            return _attach_source_metadata_hints(bioio_result, source)
        return bioio_result
    except Exception as exc:
        bioio_error = exc

    is_tiff = _is_tiff_path(source)
    tiff_error: Exception | None = None
    if is_tiff:
        try:
            result = _load_with_tifffile(
                str(source),
                output_root=root,
                array_mode=array_mode,
                t_index=t_index,
                c_index=c_index,
                z_index=z_index,
                generate_preview=generate_preview,
                save_array=save_array,
                include_array=include_array,
                max_inline_elements=max_inline_elements,
                max_saved_array_bytes=max_saved_array_bytes,
                return_array=return_array,
            )
            if bioio_error is not None:
                result.setdefault("warnings", [])
                result["warnings"].append(f"bioio loader failed, using tifffile fallback: {bioio_error}")
            if bool(result.get("success")):
                return _attach_source_metadata_hints(result, source)
            return result
        except Exception as tifffile_exc:
            tiff_error = tifffile_exc

    try:
        fallback = _load_with_pillow(
            str(source),
            output_root=root,
            array_mode=array_mode,
            generate_preview=generate_preview,
            save_array=save_array,
            include_array=include_array,
            max_inline_elements=max_inline_elements,
            max_saved_array_bytes=max_saved_array_bytes,
            return_array=return_array,
        )
        fallback.setdefault("warnings", [])
        if bioio_error is not None:
            fallback["warnings"].append(f"bioio loader unavailable or failed: {bioio_error}")
        if is_tiff and tiff_error is not None:
            fallback["warnings"].append(f"tifffile fallback failed: {tiff_error}")
        if _is_nifti_path(source) and nifti_error is not None:
            fallback["warnings"].append(f"nibabel fallback failed: {nifti_error}")
        if bool(fallback.get("success")):
            return _attach_source_metadata_hints(fallback, source)
        return fallback
    except Exception as pil_error:
        detail_parts = [f"bioio_error={bioio_error}"]
        if tiff_error is not None:
            detail_parts.append(f"tifffile_error={tiff_error}")
        if nifti_error is not None:
            detail_parts.append(f"nibabel_error={nifti_error}")
        detail_parts.append(f"pillow_error={pil_error}")
        return {
            "success": False,
            "error": "Failed to load image with all loaders. " + "; ".join(detail_parts),
        }


def _probe_scientific_image_uncached(
    *,
    file_path: str,
    scene: int | str | None = None,
    use_aicspylibczi: bool = False,
    array_mode: ArrayMode = "plane",
    t_index: int | None = None,
    c_index: int | None = None,
    z_index: int | None = None,
    output_root: str | None = None,
) -> dict[str, Any]:
    source = Path(str(file_path)).expanduser()
    if _is_ordinary_display_image_path(source):
        try:
            return _attach_source_metadata_hints(_probe_with_pillow(str(source)), source)
        except Exception as exc:
            return {
                "success": False,
                "error": f"Failed to probe image with Pillow fast path: {exc}",
            }

    return load_scientific_image(
        file_path=str(source),
        scene=scene,
        use_aicspylibczi=use_aicspylibczi,
        array_mode=array_mode,
        t_index=t_index,
        c_index=c_index,
        z_index=z_index,
        generate_preview=False,
        save_array=False,
        include_array=False,
        output_root=output_root,
        return_array=False,
    )


def ensure_scientific_derivative(**kwargs: Any) -> dict[str, Any]:
    return load_scientific_image(**kwargs)

def probe_scientific_image(
    *,
    file_path: str,
    scene: int | str | None = None,
    use_aicspylibczi: bool = False,
    array_mode: ArrayMode = "plane",
    t_index: int | None = None,
    c_index: int | None = None,
    z_index: int | None = None,
    output_root: str | None = None,
) -> dict[str, Any]:
    return get_cached_file_derivative(
        derivative_kind="probe",
        file_path=file_path,
        factory=lambda: _probe_scientific_image_uncached(
            file_path=file_path,
            scene=scene,
            use_aicspylibczi=use_aicspylibczi,
            array_mode=array_mode,
            t_index=t_index,
            c_index=c_index,
            z_index=z_index,
            output_root=output_root,
        ),
        clone_result=True,
        scene=scene,
        use_aicspylibczi=bool(use_aicspylibczi),
        array_mode=array_mode,
        t_index=t_index,
        c_index=c_index,
        z_index=z_index,
    )


__all__ = ["ensure_scientific_derivative", "load_scientific_image", "probe_scientific_image"]
