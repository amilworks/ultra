"""Shared Megaseg DynUNet inference utilities used by the CLI runner and service."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import tifffile
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import DynUNet
from PIL import Image
from skimage.measure import label as sk_label

ROI_SIZE = [32, 128, 128]
STRIDES = [1, 2, 2, 2, 2]
KERNEL_SIZE = [3, 3, 3, 3, 3]
UPSAMPLE_KERNEL_SIZE = [2, 2, 2, 2]
REMOTE_SOURCE_SCHEMES = {"http", "https", "s3"}


def load_megaseg_request(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def looks_like_remote_source(source: str) -> bool:
    raw = str(source or "").strip()
    if not raw:
        return False
    parsed = urlparse(raw)
    return bool(parsed.scheme and parsed.netloc and parsed.scheme.lower() in REMOTE_SOURCE_SCHEMES)


def megaseg_source_name(source: str) -> str:
    raw = str(source or "").strip()
    if not raw:
        return ""
    if looks_like_remote_source(raw):
        parsed = urlparse(raw)
        name = Path(unquote(parsed.path.rstrip("/"))).name
        if name:
            return name
        return unquote(parsed.netloc)
    return Path(raw).expanduser().name


def _preferred_bioio_reader(source_name: str) -> Any | None:
    name = str(source_name or "").strip().lower()
    try:
        if name.endswith(".ome.zarr") or name.endswith(".zarr"):
            import bioio_ome_zarr  # type: ignore

            return bioio_ome_zarr.Reader
        if name.endswith(".ome.tif") or name.endswith(".ome.tiff"):
            import bioio_ome_tiff  # type: ignore

            return bioio_ome_tiff.Reader
        if name.endswith(".tif") or name.endswith(".tiff"):
            import bioio_tifffile  # type: ignore

            return bioio_tifffile.Reader
    except Exception:
        return None
    return None


def _fallback_bioio_readers(source_name: str) -> list[Any]:
    name = str(source_name or "").strip().lower()
    readers: list[Any] = []
    try:
        if (
            name.endswith(".ome.tif")
            or name.endswith(".ome.tiff")
            or name.endswith(".tif")
            or name.endswith(".tiff")
        ):
            import bioio_tifffile  # type: ignore

            readers.append(bioio_tifffile.Reader)
    except Exception:
        pass
    return readers


def _bioio_reader_candidates(source_name: str) -> list[Any | None]:
    candidates: list[Any | None] = []
    preferred = _preferred_bioio_reader(source_name)
    if preferred is not None:
        candidates.append(preferred)
    for candidate in _fallback_bioio_readers(source_name):
        if candidate is not None and candidate not in candidates:
            candidates.append(candidate)
    if not candidates:
        candidates.append(None)
    return candidates


def _bioio_reader_label(reader: Any | None) -> str:
    if reader is None:
        return "auto"
    module_name = str(getattr(reader, "__module__", "") or "").strip()
    qual_name = str(
        getattr(reader, "__qualname__", getattr(reader, "__name__", "Reader")) or "Reader"
    ).strip()
    if module_name:
        return f"{module_name}.{qual_name}"
    return qual_name or "Reader"


def load_image_source_with_bioio(source: str) -> tuple[np.ndarray, str, list[str]]:
    from bioio import BioImage  # type: ignore

    reader_errors: list[str] = []
    source_name = megaseg_source_name(source)
    for candidate in _bioio_reader_candidates(source_name):
        base_kwargs: dict[str, Any] = {}
        if candidate is not None:
            base_kwargs["reader"] = candidate
        attempts: list[tuple[dict[str, Any], str | None]] = [(base_kwargs, None)]
        if str(source).strip().lower().startswith("s3://"):
            attempts.append(
                (
                    {**base_kwargs, "fs_kwargs": {"anon": True}},
                    "Opened public S3 source with anon=True fallback.",
                )
            )
        for kwargs, attempt_warning in attempts:
            try:
                bio = BioImage(str(source), **kwargs)
                dims = getattr(bio, "dims", None)
                axes = str(getattr(dims, "order", "") or "").upper()
                if axes:
                    array = np.asarray(bio.get_image_data(axes))
                else:
                    array = np.asarray(getattr(bio, "data"))
                    axes = "TCZYX"[-array.ndim :]
                warnings: list[str] = []
                if attempt_warning:
                    warnings.append(attempt_warning)
                if candidate is not None and reader_errors:
                    warnings.append(
                        "Primary bioio reader failed; recovered with fallback reader "
                        f"{_bioio_reader_label(candidate)}."
                    )
                return array, axes, warnings
            except Exception as exc:
                reader_errors.append(f"{_bioio_reader_label(candidate)}: {exc}")
                continue
    raise RuntimeError("bioio open failed via all candidate readers: " + " | ".join(reader_errors))


def load_image_source(source: str) -> tuple[np.ndarray, str, list[str]]:
    bioio_error: Exception | None = None
    try:
        return load_image_source_with_bioio(source)
    except Exception as exc:
        bioio_error = exc

    local_path = Path(str(source)).expanduser()
    lower = megaseg_source_name(source).lower()
    if (
        not looks_like_remote_source(source)
        and local_path.exists()
        and local_path.is_file()
        and lower.endswith((".ome.tif", ".ome.tiff", ".tif", ".tiff"))
    ):
        with tifffile.TiffFile(str(local_path)) as tif:
            series = tif.series[0]
            array = series.asarray()
            axes = str(series.axes or "")
        warnings = []
        if bioio_error is not None:
            warnings.append(f"bioio loader failed, using tifffile fallback: {bioio_error}")
        return np.asarray(array), axes, warnings

    raise RuntimeError(f"Failed to load image source '{source}': {bioio_error}")


def resolve_megaseg_device(requested: str | None) -> torch.device:
    token = str(requested or "").strip().lower()
    if token and token not in {"auto", "default"}:
        return torch.device(token)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_megaseg_model(checkpoint_path: Path, device: torch.device) -> DynUNet:
    model = DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        strides=STRIDES,
        kernel_size=KERNEL_SIZE,
        upsample_kernel_size=UPSAMPLE_KERNEL_SIZE,
        dropout=0.0,
        res_block=True,
    )
    state = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    normalized_state_dict = {
        (key[len("backbone.") :] if str(key).startswith("backbone.") else str(key)): value
        for key, value in dict(state_dict).items()
    }
    model.load_state_dict(normalized_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def channel_to_index(channel_value: int | None, index_base: int) -> int | None:
    if channel_value is None:
        return None
    return int(channel_value) - int(index_base)


def _drop_axis(
    arr: np.ndarray, axes: str, axis_name: str, index: int = 0
) -> tuple[np.ndarray, str]:
    axis = axes.index(axis_name)
    updated = np.take(arr, index, axis=axis)
    return updated, axes[:axis] + axes[axis + 1 :]


def extract_channel_volume(array: np.ndarray, axes: str, channel_index: int | None) -> np.ndarray:
    arr = np.asarray(array)
    normalized_axes = str(axes or "").upper().strip()
    if len(normalized_axes) != arr.ndim:
        raise ValueError(
            f"Image axes metadata length ({len(normalized_axes)}) does not match ndim ({arr.ndim})."
        )

    for axis_name in ("S", "T"):
        if axis_name in normalized_axes:
            arr, normalized_axes = _drop_axis(arr, normalized_axes, axis_name, index=0)

    if "C" in normalized_axes:
        axis = normalized_axes.index("C")
        resolved_channel_index = int(channel_index or 0)
        if resolved_channel_index < 0 or resolved_channel_index >= arr.shape[axis]:
            raise IndexError(
                f"Requested channel index {resolved_channel_index} is outside the available channel range 0-{arr.shape[axis] - 1}."
            )
        arr, normalized_axes = _drop_axis(arr, normalized_axes, "C", index=resolved_channel_index)
    elif channel_index not in (0, None):
        raise IndexError("This image does not have a channel axis.")

    for axis_position in range(len(normalized_axes) - 1, -1, -1):
        axis_name = normalized_axes[axis_position]
        if axis_name in {"Z", "Y", "X"}:
            continue
        if arr.shape[axis_position] != 1:
            raise ValueError(
                f"Unsupported axis '{axis_name}' with size {arr.shape[axis_position]} for Megaseg inference."
            )
        arr = np.take(arr, 0, axis=axis_position)
        normalized_axes = normalized_axes[:axis_position] + normalized_axes[axis_position + 1 :]

    if "Y" not in normalized_axes or "X" not in normalized_axes:
        raise ValueError(f"Expected Y and X axes, but found axes='{normalized_axes}'.")
    if "Z" not in normalized_axes:
        arr = arr[None, ...]
        normalized_axes = "Z" + normalized_axes

    order = [normalized_axes.index("Z"), normalized_axes.index("Y"), normalized_axes.index("X")]
    return np.transpose(arr, order)


def zscore_normalize(volume: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    x = np.asarray(volume, dtype=np.float32)
    finite = x[np.isfinite(x)]
    if finite.size <= 0:
        return np.zeros_like(x, dtype=np.float32), {"mean": 0.0, "std": 0.0}
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    if std <= 1e-6:
        std = 1.0
    normalized = (x - mean) / std
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized, {"mean": mean, "std": std}


def percentile_uint8(image: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    finite = x[np.isfinite(x)]
    if finite.size <= 0:
        return np.zeros(x.shape, dtype=np.uint8)
    low = float(np.percentile(finite, lo))
    high = float(np.percentile(finite, hi))
    if high <= low:
        return np.zeros(x.shape, dtype=np.uint8)
    clipped = np.clip(x, low, high)
    scaled = (clipped - low) / (high - low)
    return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)


def mask_bbox(mask: np.ndarray) -> dict[str, list[int]] | None:
    coords = np.argwhere(mask)
    if coords.size <= 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return {
        "z": [int(mins[0]), int(maxs[0])],
        "y": [int(mins[1]), int(maxs[1])],
        "x": [int(mins[2]), int(maxs[2])],
    }


def component_metrics(mask: np.ndarray) -> dict[str, Any]:
    labels = sk_label(mask.astype(bool), connectivity=1)
    component_sizes = np.bincount(labels.ravel())[1:]
    if component_sizes.size <= 0:
        return {
            "object_count": 0,
            "largest_component_voxels": 0,
            "mean_component_size_voxels": 0.0,
            "median_component_size_voxels": 0.0,
            "component_size_voxels_values": [],
        }
    return {
        "object_count": int(component_sizes.size),
        "largest_component_voxels": int(component_sizes.max()),
        "mean_component_size_voxels": float(component_sizes.mean()),
        "median_component_size_voxels": float(np.median(component_sizes)),
        "component_size_voxels_values": [int(value) for value in component_sizes[:4096]],
    }


def inside_outside_stats(volume: np.ndarray, mask: np.ndarray, prefix: str) -> dict[str, Any]:
    inside = np.asarray(volume)[mask]
    outside = np.asarray(volume)[~mask]
    inside_mean = float(np.mean(inside)) if inside.size else 0.0
    outside_mean = float(np.mean(outside)) if outside.size else 0.0
    ratio = None
    if abs(outside_mean) > 1e-6:
        ratio = float(inside_mean / outside_mean)
    return {
        f"{prefix}_inside_mean": inside_mean,
        f"{prefix}_outside_mean": outside_mean,
        f"{prefix}_inside_outside_ratio": ratio,
    }


def save_overlay(
    *,
    structure_volume: np.ndarray,
    nucleus_volume: np.ndarray | None,
    mask_volume: np.ndarray,
    output_path: Path,
    projection: str,
) -> str:
    if projection == "mip":
        structure_image = percentile_uint8(np.max(structure_volume, axis=0))
        mask_image = np.max(mask_volume, axis=0).astype(bool)
        nucleus_image = (
            percentile_uint8(np.max(nucleus_volume, axis=0)) if nucleus_volume is not None else None
        )
    else:
        z_index = int(mask_volume.shape[0] // 2)
        structure_image = percentile_uint8(structure_volume[z_index])
        mask_image = mask_volume[z_index].astype(bool)
        nucleus_image = (
            percentile_uint8(nucleus_volume[z_index]) if nucleus_volume is not None else None
        )

    rgb = np.stack([structure_image, structure_image, structure_image], axis=-1)
    if nucleus_image is not None:
        rgb[..., 0] = np.maximum(rgb[..., 0], (0.6 * nucleus_image).astype(np.uint8))
        rgb[..., 2] = np.maximum(rgb[..., 2], nucleus_image)
    if np.any(mask_image):
        highlight = np.array([20, 235, 110], dtype=np.float32)
        rgb[mask_image] = np.clip(
            (0.55 * rgb[mask_image].astype(np.float32)) + (0.45 * highlight),
            0,
            255,
        ).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(output_path)
    return str(output_path)


def build_technical_summary(
    *,
    structure_name: str,
    mask_stats: dict[str, Any],
    structure_stats: dict[str, Any],
    nucleus_stats: dict[str, Any] | None,
) -> str:
    coverage = float(mask_stats.get("coverage_percent") or 0.0)
    segmented = int(mask_stats.get("segmented_voxels") or 0)
    total = int(mask_stats.get("total_voxels") or 0)
    objects = int(mask_stats.get("object_count") or 0)
    active_slices = int(mask_stats.get("active_slice_count") or 0)
    z_slices = int(mask_stats.get("z_slice_count") or 0)
    largest = int(mask_stats.get("largest_component_voxels") or 0)

    bits = [
        f"Megaseg segmented {structure_name} across {coverage:.3f}% of the volume ({segmented}/{total} voxels).",
        f"The mask spans {active_slices}/{z_slices} z-slices and resolves {objects} connected component(s); the largest component contains {largest} voxels.",
    ]

    structure_ratio = structure_stats.get("structure_inside_outside_ratio")
    if isinstance(structure_ratio, (int, float)):
        bits.append(
            f"Structure-channel mean intensity is {float(structure_ratio):.3f}x higher inside the mask than outside."
        )

    if nucleus_stats:
        nucleus_ratio = nucleus_stats.get("nucleus_inside_outside_ratio")
        if isinstance(nucleus_ratio, (int, float)):
            bits.append(
                f"Nucleus-channel mean intensity is {float(nucleus_ratio):.3f}x higher inside the mask than outside."
            )

    return " ".join(bits)


def write_megaseg_summary_json(path: Path, payload: dict[str, Any]) -> str:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def write_megaseg_summary_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    fieldnames = [
        "file",
        "success",
        "coverage_percent",
        "segmented_voxels",
        "object_count",
        "largest_component_voxels",
        "active_slice_count",
        "z_slice_count",
        "structure_inside_outside_ratio",
        "nucleus_inside_outside_ratio",
        "mask_path",
        "probability_path",
        "technical_summary",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return str(path)


def write_megaseg_report(path: Path, payload: dict[str, Any]) -> str:
    aggregate = payload.get("aggregate") if isinstance(payload.get("aggregate"), dict) else {}
    rows = payload.get("files") if isinstance(payload.get("files"), list) else []

    lines = [
        "# Megaseg Inference Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Aggregate",
        "",
        f"- Processed files: {int(aggregate.get('processed_files') or 0)}/{int(aggregate.get('total_files') or 0)}",
        f"- Mean coverage percent: {float(aggregate.get('mean_coverage_percent') or 0.0):.4f}",
        f"- Median coverage percent: {float(aggregate.get('median_coverage_percent') or 0.0):.4f}",
        f"- Mean object count: {float(aggregate.get('mean_object_count') or 0.0):.4f}",
        f"- Median object count: {float(aggregate.get('median_object_count') or 0.0):.4f}",
        "",
        "## Per-file Summary",
        "",
        "| File | Coverage % | Objects | Active Z | Largest Component |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        if not isinstance(row, dict):
            continue
        segmentation = row.get("segmentation") if isinstance(row.get("segmentation"), dict) else {}
        coverage = float(
            row.get("coverage_percent")
            if row.get("coverage_percent") is not None
            else segmentation.get("coverage_percent") or 0.0
        )
        objects = int(
            row.get("object_count")
            if row.get("object_count") is not None
            else segmentation.get("object_count") or 0
        )
        active = int(
            row.get("active_slice_count")
            if row.get("active_slice_count") is not None
            else segmentation.get("active_slice_count") or 0
        )
        z_slices = int(
            row.get("z_slice_count")
            if row.get("z_slice_count") is not None
            else segmentation.get("z_slice_count") or 0
        )
        largest = int(
            row.get("largest_component_voxels")
            if row.get("largest_component_voxels") is not None
            else segmentation.get("largest_component_voxels") or 0
        )
        lines.append(
            f"| {row.get('file') or 'unknown'} | {coverage:.4f} | {objects} | {active}/{z_slices} | {largest} |"
        )

    lines.extend(["", "## Technical Notes", ""])
    for row in rows:
        if not isinstance(row, dict) or not row.get("success"):
            continue
        lines.append(f"### {row.get('file')}")
        lines.append("")
        lines.append(str(row.get("technical_summary") or ""))
        lines.append("")

    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return str(path)


def run_megaseg_file(
    *,
    file_source: str,
    output_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    structure_channel_index: int,
    structure_channel_number: int,
    nucleus_channel_index: int | None,
    nucleus_channel_number: int | None,
    mask_threshold: float,
    structure_name: str,
    checkpoint_path: Path,
    save_visualizations: bool,
    amp_enabled: bool = False,
) -> dict[str, Any]:
    source_name = megaseg_source_name(file_source)
    base_name = Path(source_name).stem.replace(".ome", "")
    array, axes, load_warnings = load_image_source(file_source)

    structure_volume = extract_channel_volume(array, axes, structure_channel_index)
    nucleus_volume = (
        extract_channel_volume(array, axes, nucleus_channel_index)
        if nucleus_channel_index is not None
        else None
    )
    normalized_volume, normalization = zscore_normalize(structure_volume)

    x = torch.from_numpy(normalized_volume[None, None]).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        if amp_enabled and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = sliding_window_inference(
                    inputs=x,
                    predictor=model,
                    sw_batch_size=1,
                    roi_size=ROI_SIZE,
                    overlap=0.3,
                )
        else:
            logits = sliding_window_inference(
                inputs=x,
                predictor=model,
                sw_batch_size=1,
                roi_size=ROI_SIZE,
                overlap=0.3,
            )
    probability = torch.sigmoid(logits).detach().cpu().numpy().squeeze(0).squeeze(0)
    probability_u8 = np.clip(np.round(probability * 255.0), 0, 255).astype(np.uint8)
    probability = probability_u8.astype(np.float32) / 255.0
    mask_bool = probability >= float(mask_threshold)
    mask_u8 = (mask_bool.astype(np.uint8) * 255).astype(np.uint8)

    segmented_voxels = int(np.count_nonzero(mask_bool))
    total_voxels = int(mask_bool.size)
    coverage_fraction = float(segmented_voxels / total_voxels) if total_voxels else 0.0
    slice_activity = mask_bool.reshape(mask_bool.shape[0], -1).any(axis=1)
    per_slice_coverage = mask_bool.reshape(mask_bool.shape[0], -1).mean(axis=1) * 100.0
    mask_stats = {
        "segmented_voxels": segmented_voxels,
        "total_voxels": total_voxels,
        "coverage_fraction": coverage_fraction,
        "coverage_percent": coverage_fraction * 100.0,
        "active_slice_count": int(np.count_nonzero(slice_activity)),
        "inactive_slice_count": int(mask_bool.shape[0] - np.count_nonzero(slice_activity)),
        "z_slice_count": int(mask_bool.shape[0]),
        "slice_coverage_percent_mean": float(np.mean(per_slice_coverage)),
        "slice_coverage_percent_max": float(np.max(per_slice_coverage)),
        "bbox_zyx": mask_bbox(mask_bool),
    }
    mask_stats.update(component_metrics(mask_bool))

    structure_stats = inside_outside_stats(structure_volume, mask_bool, "structure")
    nucleus_stats = (
        inside_outside_stats(nucleus_volume, mask_bool, "nucleus")
        if nucleus_volume is not None
        else None
    )
    technical_summary = build_technical_summary(
        structure_name=structure_name,
        mask_stats=mask_stats,
        structure_stats=structure_stats,
        nucleus_stats=nucleus_stats,
    )

    file_dir = output_dir / base_name
    file_dir.mkdir(parents=True, exist_ok=True)
    mask_path = file_dir / f"{base_name}__megaseg_mask.tiff"
    probability_path = file_dir / f"{base_name}__megaseg_probability.tiff"
    tifffile.imwrite(mask_path, mask_u8, metadata={"axes": "ZYX"})
    tifffile.imwrite(probability_path, probability_u8.astype(np.uint8), metadata={"axes": "ZYX"})

    visualizations: list[dict[str, Any]] = []
    if save_visualizations:
        mid_path = file_dir / f"{base_name}__megaseg_overlay_midz.png"
        mip_path = file_dir / f"{base_name}__megaseg_overlay_mip.png"
        visualizations.append(
            {
                "path": save_overlay(
                    structure_volume=structure_volume,
                    nucleus_volume=nucleus_volume,
                    mask_volume=mask_bool,
                    output_path=mid_path,
                    projection="mid_z",
                ),
                "kind": "overlay_mid_z",
                "title": "Megaseg overlay (mid Z)",
            }
        )
        visualizations.append(
            {
                "path": save_overlay(
                    structure_volume=structure_volume,
                    nucleus_volume=nucleus_volume,
                    mask_volume=mask_bool,
                    output_path=mip_path,
                    projection="mip",
                ),
                "kind": "overlay_mip",
                "title": "Megaseg overlay (MIP)",
            }
        )

    summary_payload = {
        "file": source_name or file_source,
        "path": str(file_source),
        "success": True,
        "axes": axes,
        "source_volume_shape_zyx": [int(v) for v in structure_volume.shape],
        "structure_channel_index": int(structure_channel_index),
        "structure_channel_number": int(structure_channel_number),
        "nucleus_channel_index": int(nucleus_channel_index)
        if nucleus_channel_index is not None
        else None,
        "nucleus_channel_number": int(nucleus_channel_number)
        if nucleus_channel_number is not None
        else None,
        "normalization": normalization,
        "mask_threshold": float(mask_threshold),
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "mask_path": str(mask_path),
        "probability_path": str(probability_path),
        "visualizations": visualizations,
        "segmentation": mask_stats,
        "intensity_context": {
            **structure_stats,
            **(nucleus_stats or {}),
        },
        "technical_summary": technical_summary,
    }
    if load_warnings:
        summary_payload["warnings"] = load_warnings
    summary_json_path = file_dir / f"{base_name}__megaseg_summary.json"
    summary_payload["summary_json_path"] = write_megaseg_summary_json(
        summary_json_path, summary_payload
    )
    return summary_payload


def run_megaseg_batch(
    *,
    file_paths: list[str],
    output_dir: Path,
    checkpoint_path: Path,
    structure_channel: int = 4,
    nucleus_channel: int | None = 6,
    channel_index_base: int = 1,
    mask_threshold: float = 0.5,
    save_visualizations: bool = True,
    generate_report: bool = True,
    device: torch.device | str | None = None,
    structure_name: str = "structure",
    model: torch.nn.Module | None = None,
    amp_enabled: bool = False,
) -> dict[str, Any]:
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    resolved_device = (
        device if isinstance(device, torch.device) else resolve_megaseg_device(str(device or ""))
    )
    model_instance = model or build_megaseg_model(
        checkpoint_path=resolved_checkpoint_path,
        device=resolved_device,
    )
    structure_channel_index = channel_to_index(int(structure_channel), int(channel_index_base))
    nucleus_channel_index = channel_to_index(
        int(nucleus_channel) if nucleus_channel is not None else None,
        int(channel_index_base),
    )

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for raw_path in list(file_paths or []):
        file_source = str(raw_path or "").strip()
        source_name = megaseg_source_name(file_source) or file_source
        try:
            row = run_megaseg_file(
                file_source=file_source,
                output_dir=resolved_output_dir,
                model=model_instance,
                device=resolved_device,
                structure_channel_index=int(structure_channel_index or 0),
                structure_channel_number=int(structure_channel),
                nucleus_channel_index=nucleus_channel_index,
                nucleus_channel_number=int(nucleus_channel)
                if nucleus_channel is not None
                else None,
                mask_threshold=float(mask_threshold),
                structure_name=str(structure_name or "structure"),
                checkpoint_path=resolved_checkpoint_path,
                save_visualizations=bool(save_visualizations),
                amp_enabled=bool(amp_enabled),
            )
            rows.append(row)
            for item in list(row.get("warnings") or []):
                text = str(item or "").strip()
                if text and text not in warnings:
                    warnings.append(text)
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "file": source_name,
                    "path": file_source,
                    "success": False,
                    "error": str(exc),
                }
            )

    successful = [row for row in rows if row.get("success")]
    coverage_values = [
        float((row.get("segmentation") or {}).get("coverage_percent") or 0.0) for row in successful
    ]
    object_counts = [
        int((row.get("segmentation") or {}).get("object_count") or 0) for row in successful
    ]
    aggregate = {
        "processed_files": len(successful),
        "total_files": len(rows),
        "mean_coverage_percent": float(np.mean(coverage_values)) if coverage_values else 0.0,
        "median_coverage_percent": float(np.median(coverage_values)) if coverage_values else 0.0,
        "mean_object_count": float(np.mean(object_counts)) if object_counts else 0.0,
        "median_object_count": float(np.median(object_counts)) if object_counts else 0.0,
    }

    summary_rows: list[dict[str, Any]] = []
    for row in rows:
        segmentation = row.get("segmentation") if isinstance(row.get("segmentation"), dict) else {}
        intensity_context = (
            row.get("intensity_context") if isinstance(row.get("intensity_context"), dict) else {}
        )
        summary_rows.append(
            {
                "file": row.get("file"),
                "success": row.get("success"),
                "coverage_percent": segmentation.get("coverage_percent"),
                "segmented_voxels": segmentation.get("segmented_voxels"),
                "object_count": segmentation.get("object_count"),
                "largest_component_voxels": segmentation.get("largest_component_voxels"),
                "active_slice_count": segmentation.get("active_slice_count"),
                "z_slice_count": segmentation.get("z_slice_count"),
                "structure_inside_outside_ratio": intensity_context.get(
                    "structure_inside_outside_ratio"
                ),
                "nucleus_inside_outside_ratio": intensity_context.get(
                    "nucleus_inside_outside_ratio"
                ),
                "mask_path": row.get("mask_path"),
                "probability_path": row.get("probability_path"),
                "technical_summary": row.get("technical_summary"),
            }
        )

    payload: dict[str, Any] = {
        "success": len(successful) > 0,
        "processed": len(successful),
        "total_files": len(rows),
        "device": str(resolved_device),
        "checkpoint_path": str(resolved_checkpoint_path),
        "output_directory": str(resolved_output_dir),
        "files": rows,
        "aggregate": aggregate,
        "warnings": warnings,
    }

    summary_csv_path = resolved_output_dir / "megaseg_summary.csv"
    payload["summary_csv_path"] = write_megaseg_summary_csv(summary_csv_path, summary_rows)

    if bool(generate_report):
        report_path = resolved_output_dir / "megaseg_report.md"
        payload["report_path"] = write_megaseg_report(report_path, payload)

    return payload


__all__ = [
    "build_megaseg_model",
    "channel_to_index",
    "extract_channel_volume",
    "load_image_source",
    "load_image_source_with_bioio",
    "load_megaseg_request",
    "looks_like_remote_source",
    "megaseg_source_name",
    "resolve_megaseg_device",
    "run_megaseg_batch",
    "run_megaseg_file",
]
