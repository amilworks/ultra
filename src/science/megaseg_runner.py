#!/usr/bin/env python3
"""Minimal Megaseg DynUNet inference runner for multichannel microscopy volumes."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import tifffile
import torch
from monai.losses import GeneralizedDiceFocalLoss
from monai.networks.nets import DynUNet
from skimage.measure import label as sk_label

from cyto_dl.models.im2im import MultiTaskIm2Im
from cyto_dl.models.im2im.utils.postprocessing import ActThreshLabel
from cyto_dl.nn.head.base_head import BaseHead


_ROI_SIZE = [32, 128, 128]
_STRIDES = [1, 2, 2, 2, 2]
_KERNEL_SIZE = [3, 3, 3, 3, 3]
_UPSAMPLE_KERNEL_SIZE = [2, 2, 2, 2]


def _load_request(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_device(requested: str | None) -> torch.device:
    token = str(requested or "").strip().lower()
    if token and token not in {"auto", "default"}:
        return torch.device(token)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_model(checkpoint_path: Path, device: torch.device) -> MultiTaskIm2Im:
    backbone = DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        strides=_STRIDES,
        kernel_size=_KERNEL_SIZE,
        upsample_kernel_size=_UPSAMPLE_KERNEL_SIZE,
        dropout=0.0,
        res_block=True,
    )
    head = BaseHead(
        loss=GeneralizedDiceFocalLoss(sigmoid=True),
        postprocess={
            "input": ActThreshLabel(rescale_dtype=np.uint8),
            "prediction": ActThreshLabel(
                activation=torch.nn.Sigmoid(),
                rescale_dtype=np.uint8,
            ),
        },
    )
    model = MultiTaskIm2Im(
        backbone=backbone,
        task_heads={"seg": head},
        x_key="raw",
        inference_args={
            "sw_batch_size": 1,
            "roi_size": _ROI_SIZE,
            "overlap": 0.3,
        },
    )
    state = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _channel_to_index(channel_value: int | None, index_base: int) -> int | None:
    if channel_value is None:
        return None
    return int(channel_value) - int(index_base)


def _drop_axis(arr: np.ndarray, axes: str, axis_name: str, index: int = 0) -> tuple[np.ndarray, str]:
    axis = axes.index(axis_name)
    updated = np.take(arr, index, axis=axis)
    return updated, axes[:axis] + axes[axis + 1 :]


def _extract_channel_volume(array: np.ndarray, axes: str, channel_index: int) -> np.ndarray:
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
        if channel_index < 0 or channel_index >= arr.shape[axis]:
            raise IndexError(
                f"Requested channel index {channel_index} is outside the available channel range 0-{arr.shape[axis] - 1}."
            )
        arr, normalized_axes = _drop_axis(arr, normalized_axes, "C", index=channel_index)
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


def _zscore_normalize(volume: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
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


def _percentile_uint8(image: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
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


def _mask_bbox(mask: np.ndarray) -> dict[str, list[int]] | None:
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


def _component_metrics(mask: np.ndarray) -> dict[str, Any]:
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


def _inside_outside_stats(volume: np.ndarray, mask: np.ndarray, prefix: str) -> dict[str, Any]:
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


def _save_overlay(
    *,
    structure_volume: np.ndarray,
    nucleus_volume: np.ndarray | None,
    mask_volume: np.ndarray,
    output_path: Path,
    projection: str,
) -> str:
    if projection == "mip":
        structure_image = _percentile_uint8(np.max(structure_volume, axis=0))
        mask_image = np.max(mask_volume, axis=0).astype(bool)
        nucleus_image = (
            _percentile_uint8(np.max(nucleus_volume, axis=0))
            if nucleus_volume is not None
            else None
        )
    else:
        z_index = int(mask_volume.shape[0] // 2)
        structure_image = _percentile_uint8(structure_volume[z_index])
        mask_image = mask_volume[z_index].astype(bool)
        nucleus_image = (
            _percentile_uint8(nucleus_volume[z_index])
            if nucleus_volume is not None
            else None
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


def _build_technical_summary(
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


def _write_summary_json(path: Path, payload: dict[str, Any]) -> str:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> str:
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


def _write_report(path: Path, payload: dict[str, Any]) -> str:
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
            else segmentation.get("coverage_percent")
            or 0.0
        )
        objects = int(
            row.get("object_count")
            if row.get("object_count") is not None
            else segmentation.get("object_count")
            or 0
        )
        active = int(
            row.get("active_slice_count")
            if row.get("active_slice_count") is not None
            else segmentation.get("active_slice_count")
            or 0
        )
        z_slices = int(
            row.get("z_slice_count")
            if row.get("z_slice_count") is not None
            else segmentation.get("z_slice_count")
            or 0
        )
        largest = int(
            row.get("largest_component_voxels")
            if row.get("largest_component_voxels") is not None
            else segmentation.get("largest_component_voxels")
            or 0
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


def _run_file(
    *,
    file_path: Path,
    output_dir: Path,
    model: MultiTaskIm2Im,
    device: torch.device,
    structure_channel_index: int,
    structure_channel_number: int,
    nucleus_channel_index: int | None,
    nucleus_channel_number: int | None,
    mask_threshold: float,
    structure_name: str,
    checkpoint_path: Path,
    save_visualizations: bool,
) -> dict[str, Any]:
    base_name = file_path.stem.replace(".ome", "")
    with tifffile.TiffFile(str(file_path)) as tif:
        series = tif.series[0]
        array = series.asarray()
        axes = str(series.axes or "")

    structure_volume = _extract_channel_volume(array, axes, structure_channel_index)
    nucleus_volume = (
        _extract_channel_volume(array, axes, nucleus_channel_index)
        if nucleus_channel_index is not None
        else None
    )
    normalized_volume, normalization = _zscore_normalize(structure_volume)

    x = torch.from_numpy(normalized_volume[None, None]).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        outs = model._inference_forward({"raw": x}, "predict", 1, ["seg"])
    probability_u8 = np.asarray(outs["seg"]["pred"][0]).squeeze(0)
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
        "bbox_zyx": _mask_bbox(mask_bool),
    }
    mask_stats.update(_component_metrics(mask_bool))

    structure_stats = _inside_outside_stats(structure_volume, mask_bool, "structure")
    nucleus_stats = (
        _inside_outside_stats(nucleus_volume, mask_bool, "nucleus")
        if nucleus_volume is not None
        else None
    )
    technical_summary = _build_technical_summary(
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
                "path": _save_overlay(
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
                "path": _save_overlay(
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
        "file": file_path.name,
        "path": str(file_path),
        "success": True,
        "axes": axes,
        "source_volume_shape_zyx": [int(v) for v in structure_volume.shape],
        "structure_channel_index": int(structure_channel_index),
        "structure_channel_number": int(structure_channel_number),
        "nucleus_channel_index": int(nucleus_channel_index) if nucleus_channel_index is not None else None,
        "nucleus_channel_number": int(nucleus_channel_number) if nucleus_channel_number is not None else None,
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
    summary_json_path = file_dir / f"{base_name}__megaseg_summary.json"
    summary_payload["summary_json_path"] = _write_summary_json(summary_json_path, summary_payload)
    return summary_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Megaseg inference on microscopy images.")
    parser.add_argument("--request-json", required=True, help="Path to the runner request JSON.")
    args = parser.parse_args()

    request_path = Path(args.request_json).expanduser().resolve()
    request = _load_request(request_path)

    checkpoint_path = Path(str(request["checkpoint_path"])).expanduser().resolve()
    output_dir = Path(str(request["output_dir"])).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    structure_channel_number = int(request.get("structure_channel") or 4)
    nucleus_channel_raw = request.get("nucleus_channel")
    nucleus_channel_number = int(nucleus_channel_raw) if nucleus_channel_raw is not None else None
    channel_index_base = int(request.get("channel_index_base") or 1)
    structure_channel_index = _channel_to_index(structure_channel_number, channel_index_base)
    nucleus_channel_index = _channel_to_index(nucleus_channel_number, channel_index_base)

    device = _resolve_device(str(request.get("device") or ""))
    model = _build_model(checkpoint_path=checkpoint_path, device=device)

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for raw_path in list(request.get("file_paths") or []):
        file_path = Path(str(raw_path)).expanduser().resolve()
        try:
            row = _run_file(
                file_path=file_path,
                output_dir=output_dir,
                model=model,
                device=device,
                structure_channel_index=int(structure_channel_index or 0),
                structure_channel_number=structure_channel_number,
                nucleus_channel_index=nucleus_channel_index,
                nucleus_channel_number=nucleus_channel_number,
                mask_threshold=float(request.get("mask_threshold") or 0.5),
                structure_name=str(request.get("structure_name") or "structure"),
                checkpoint_path=checkpoint_path,
                save_visualizations=bool(request.get("save_visualizations", True)),
            )
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "file": file_path.name,
                    "path": str(file_path),
                    "success": False,
                    "error": str(exc),
                }
            )

    successful = [row for row in rows if row.get("success")]
    coverage_values = [
        float((row.get("segmentation") or {}).get("coverage_percent") or 0.0)
        for row in successful
    ]
    object_counts = [
        int((row.get("segmentation") or {}).get("object_count") or 0)
        for row in successful
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
            row.get("intensity_context")
            if isinstance(row.get("intensity_context"), dict)
            else {}
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
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "output_directory": str(output_dir),
        "files": rows,
        "aggregate": aggregate,
        "warnings": warnings,
    }

    summary_csv_path = output_dir / "megaseg_summary.csv"
    payload["summary_csv_path"] = _write_summary_csv(summary_csv_path, summary_rows)

    if bool(request.get("generate_report", True)):
        report_path = output_dir / "megaseg_report.md"
        payload["report_path"] = _write_report(report_path, payload)

    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
