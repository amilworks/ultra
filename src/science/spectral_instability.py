from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any
from uuid import uuid4

import matplotlib
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

from src.training.adapters import _resolve_yolov5_repo_path

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SPECTRAL_DEFAULT_CONF_THRES = 0.10
SPECTRAL_DEFAULT_IOU_THRES = 0.60
SPECTRAL_DEFAULT_MATCH_IOU = 0.50
SPECTRAL_DEFAULT_STABLE_IOU = 0.70
SPECTRAL_DEFAULT_PRESERVATION_RATIO = 0.90


@dataclass(frozen=True)
class SpectralInstabilityConfig:
    imgsz: int = 640
    batch_size: int = 4
    conf_thres: float = SPECTRAL_DEFAULT_CONF_THRES
    iou_thres: float = SPECTRAL_DEFAULT_IOU_THRES
    match_iou_thresh: float = SPECTRAL_DEFAULT_MATCH_IOU
    stable_iou_thresh: float = SPECTRAL_DEFAULT_STABLE_IOU
    preservation_ratio: float = SPECTRAL_DEFAULT_PRESERVATION_RATIO
    lost_weight: float = 2.0
    new_weight: float = 0.0
    jitter_weight: float = 0.5
    confidence_weight: float = 0.25


class _ImagePathDataset(Dataset):
    def __init__(self, image_paths: list[str], *, img_size: int, stride: int) -> None:
        self.image_paths = list(image_paths)
        self.img_size = int(img_size)
        self.stride = int(stride)
        self._load_images = _load_yolov5_load_images()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        path = str(self.image_paths[idx])
        _, image, _, _, _ = next(
            iter(self._load_images(path, img_size=self.img_size, stride=self.stride, auto=False))
        )
        return torch.from_numpy(np.ascontiguousarray(image)), path


def _ensure_yolov5_path() -> None:
    repo_path = _resolve_yolov5_repo_path()
    token = str(repo_path)
    if token not in sys.path:
        sys.path.insert(0, token)


def _load_yolov5_backend() -> Any:
    _ensure_yolov5_path()
    from models.common import DetectMultiBackend  # type: ignore

    return DetectMultiBackend


def _load_yolov5_load_images() -> Any:
    _ensure_yolov5_path()
    from utils.dataloaders import LoadImages  # type: ignore

    return LoadImages


def _xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def _nms_with_class_scores(
    prediction: torch.Tensor,
    *,
    conf_thres: float,
    iou_thres: float,
) -> torch.Tensor:
    num_classes = int(prediction.shape[1] - 5)
    candidates = prediction[prediction[:, 4] > float(conf_thres)]
    if candidates.shape[0] == 0:
        return torch.empty((0, 6 + num_classes), device=prediction.device)

    boxes_xyxy = _xywh2xyxy(candidates[:, :4])
    class_outputs = candidates[:, 5:]
    class_probs = torch.sigmoid(class_outputs)
    max_class_prob = class_probs.max(dim=1)[0]
    objectness = candidates[:, 4]
    winning_class_ids = class_outputs.argmax(dim=1)
    nms_scores = objectness * max_class_prob

    kept: list[int] = []
    for class_id in range(num_classes):
        mask = winning_class_ids == class_id
        if not bool(mask.any()):
            continue
        class_indices = torch.where(mask)[0]
        class_boxes = boxes_xyxy[class_indices]
        class_scores = nms_scores[class_indices]
        class_keep = torchvision.ops.nms(class_boxes, class_scores, float(iou_thres))
        kept.extend(class_indices[class_keep].tolist())

    if not kept:
        return torch.empty((0, 6 + num_classes), device=prediction.device)

    keep_indices = torch.tensor(sorted(kept), device=prediction.device)
    return torch.cat(
        [
            boxes_xyxy[keep_indices],
            objectness[keep_indices].unsqueeze(1),
            winning_class_ids[keep_indices].float().unsqueeze(1),
            class_outputs[keep_indices],
        ],
        dim=1,
    )


class SpectralInstabilityScorer:
    def __init__(self, config: SpectralInstabilityConfig) -> None:
        self.config = config

    @torch.no_grad()
    def create_energy_mask(self, energy_map: torch.Tensor) -> torch.Tensor:
        height, width = energy_map.shape
        y, x = torch.meshgrid(
            torch.arange(height, device=energy_map.device),
            torch.arange(width, device=energy_map.device),
            indexing="ij",
        )
        center_y, center_x = height // 2, width // 2
        dist = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
        max_radius = max(1.0, float(min(center_y, center_x)))
        ratio = float(self.config.preservation_ratio)
        return (dist / max_radius <= ratio).float()

    @torch.no_grad()
    def apply_adaptive_spectral_filter(self, feature_map: torch.Tensor) -> tuple[torch.Tensor, list[dict[str, float | int]]]:
        if feature_map.dtype == torch.float16:
            feature_map = feature_map.float()

        fft_tensor = torch.fft.fftshift(torch.fft.fft2(feature_map, dim=(-2, -1)), dim=(-2, -1))
        batch_size = int(fft_tensor.shape[0])
        filtered: list[torch.Tensor] = []
        diagnostics: list[dict[str, float | int]] = []
        for batch_index in range(batch_size):
            image_fft = fft_tensor[batch_index]
            channel_energy = torch.sum(torch.abs(image_fft) ** 2, dim=(1, 2))
            max_channel = int(torch.argmax(channel_energy).item())
            energy_map = torch.abs(image_fft[max_channel]) ** 2
            mask = self.create_energy_mask(energy_map)
            filtered_fft = image_fft * mask.unsqueeze(0)
            retained_energy_fraction = float(
                (energy_map * mask).sum().item() / max(float(energy_map.sum().item()), 1e-12)
            )
            restored = torch.fft.ifft2(
                torch.fft.ifftshift(filtered_fft, dim=(-2, -1)),
                dim=(-2, -1),
            ).real
            filtered.append(restored)
            diagnostics.append(
                {
                    "dominant_channel_index": max_channel,
                    "retained_energy_fraction": retained_energy_fraction,
                }
            )
        return torch.stack(filtered, dim=0), diagnostics

    @staticmethod
    def _greedy_iou_matches(
        ious: torch.Tensor,
        *,
        threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if ious.numel() == 0:
            device = ious.device
            empty = torch.empty((0,), device=device, dtype=torch.long)
            empty_float = torch.empty((0,), device=device, dtype=torch.float32)
            return empty, empty, empty_float

        row_idx, col_idx = torch.where(ious >= float(threshold))
        if row_idx.numel() == 0:
            device = ious.device
            empty = torch.empty((0,), device=device, dtype=torch.long)
            empty_float = torch.empty((0,), device=device, dtype=torch.float32)
            return empty, empty, empty_float

        pair_scores = ious[row_idx, col_idx]
        order = torch.argsort(pair_scores, descending=True)
        used_rows = torch.zeros(ious.shape[0], device=ious.device, dtype=torch.bool)
        used_cols = torch.zeros(ious.shape[1], device=ious.device, dtype=torch.bool)
        matched_rows: list[int] = []
        matched_cols: list[int] = []
        matched_scores: list[float] = []
        for pair_index in order.tolist():
            row = int(row_idx[pair_index].item())
            col = int(col_idx[pair_index].item())
            if bool(used_rows[row]) or bool(used_cols[col]):
                continue
            used_rows[row] = True
            used_cols[col] = True
            matched_rows.append(row)
            matched_cols.append(col)
            matched_scores.append(float(pair_scores[pair_index].item()))

        if not matched_rows:
            device = ious.device
            empty = torch.empty((0,), device=device, dtype=torch.long)
            empty_float = torch.empty((0,), device=device, dtype=torch.float32)
            return empty, empty, empty_float

        return (
            torch.tensor(matched_rows, device=ious.device, dtype=torch.long),
            torch.tensor(matched_cols, device=ious.device, dtype=torch.long),
            torch.tensor(matched_scores, device=ious.device, dtype=torch.float32),
        )

    @staticmethod
    def _winning_confidence(detections: torch.Tensor) -> torch.Tensor:
        if detections.numel() == 0:
            return torch.empty((0,), device=detections.device, dtype=torch.float32)
        objectness = detections[:, 4].float()
        if detections.shape[1] <= 6:
            return objectness
        class_ids = detections[:, 5].long().clamp(min=0)
        class_outputs = detections[:, 6:].float()
        max_class_index = max(0, int(class_outputs.shape[1]) - 1)
        class_ids = class_ids.clamp(max=max_class_index)
        winning_logits = class_outputs.gather(1, class_ids.unsqueeze(1)).squeeze(1)
        return objectness * torch.sigmoid(winning_logits)

    @torch.no_grad()
    def discrepancy_breakdown(self, pred_orig: torch.Tensor, pred_filt: torch.Tensor) -> dict[str, float]:
        if len(pred_orig) == 0 and len(pred_filt) == 0:
            return {
                "lost": 0.0,
                "new": 0.0,
                "class_jitter": 0.0,
                "spatial_jitter": 0.0,
                "confidence_jitter": 0.0,
                "matched": 0.0,
                "normalized_score": 0.0,
                "matched_iou_mean": 0.0,
                "matched_iou_median": 0.0,
                "class_consistency_rate": 1.0,
                "mean_confidence_shift": 0.0,
                "mean_confidence_drop": 0.0,
                "score": 0.0,
            }
        if len(pred_orig) == 0:
            new = float(len(pred_filt))
            return {
                "lost": 0.0,
                "new": new,
                "class_jitter": 0.0,
                "spatial_jitter": 0.0,
                "confidence_jitter": 0.0,
                "matched": 0.0,
                "normalized_score": float(self.config.new_weight * new),
                "matched_iou_mean": 0.0,
                "matched_iou_median": 0.0,
                "class_consistency_rate": 0.0,
                "mean_confidence_shift": 0.0,
                "mean_confidence_drop": 0.0,
                "score": self.config.new_weight * new,
            }
        if len(pred_filt) == 0:
            lost = float(len(pred_orig))
            return {
                "lost": lost,
                "new": 0.0,
                "class_jitter": 0.0,
                "spatial_jitter": 0.0,
                "confidence_jitter": 0.0,
                "matched": 0.0,
                "normalized_score": float((self.config.lost_weight * lost) / max(float(len(pred_orig)), 1.0)),
                "matched_iou_mean": 0.0,
                "matched_iou_median": 0.0,
                "class_consistency_rate": 0.0,
                "mean_confidence_shift": 0.0,
                "mean_confidence_drop": 0.0,
                "score": self.config.lost_weight * lost,
            }

        ious = torchvision.ops.box_iou(pred_orig[:, :4], pred_filt[:, :4])
        orig_idx, filt_idx, matched_ious = self._greedy_iou_matches(
            ious,
            threshold=self.config.match_iou_thresh,
        )
        matched_count = int(orig_idx.numel())
        lost = float(max(int(len(pred_orig)) - matched_count, 0))
        new = float(max(int(len(pred_filt)) - matched_count, 0))

        class_jitter = 0.0
        spatial_jitter = 0.0
        confidence_jitter = 0.0
        matched_iou_mean = 0.0
        matched_iou_median = 0.0
        class_consistency_rate = 0.0
        mean_confidence_shift = 0.0
        mean_confidence_drop = 0.0
        if len(orig_idx):
            class_mismatch = pred_orig[orig_idx, 5] != pred_filt[filt_idx, 5]
            class_jitter = float(class_mismatch.float().sum().item())
            stable_mask = (matched_ious < self.config.stable_iou_thresh) & (~class_mismatch)
            if bool(stable_mask.any()):
                spatial_jitter = float((1.0 - matched_ious[stable_mask]).sum().item())

            orig_conf = self._winning_confidence(pred_orig[orig_idx])
            filt_conf = self._winning_confidence(pred_filt[filt_idx])
            confidence_shift = (orig_conf - filt_conf).abs()
            confidence_drop = torch.clamp(orig_conf - filt_conf, min=0.0)
            confidence_jitter = float(confidence_shift.sum().item())
            mean_confidence_shift = float(confidence_shift.mean().item())
            mean_confidence_drop = float(confidence_drop.mean().item())
            matched_iou_mean = float(matched_ious.mean().item())
            matched_iou_median = float(matched_ious.median().item())
            class_consistency_rate = float((~class_mismatch).float().mean().item())

        score = (
            self.config.lost_weight * lost
            + self.config.new_weight * new
            + self.config.jitter_weight * (class_jitter + spatial_jitter)
            + self.config.confidence_weight * confidence_jitter
        )
        return {
            "lost": lost,
            "new": new,
            "class_jitter": class_jitter,
            "spatial_jitter": spatial_jitter,
            "confidence_jitter": confidence_jitter,
            "matched": float(matched_count),
            "normalized_score": float(score / max(float(len(pred_orig)), 1.0)),
            "matched_iou_mean": matched_iou_mean,
            "matched_iou_median": matched_iou_median,
            "class_consistency_rate": class_consistency_rate,
            "mean_confidence_shift": mean_confidence_shift,
            "mean_confidence_drop": mean_confidence_drop,
            "score": float(score),
        }


def _class_counts(detections: torch.Tensor, class_names: dict[int, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    if len(detections) == 0:
        return counts
    for class_id in detections[:, 5].tolist():
        label = str(class_names.get(int(class_id), str(int(class_id))))
        counts[label] = counts.get(label, 0) + 1
    return counts


def _safe_slug(value: str) -> str:
    token = str(value or "").strip()
    token = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in token)
    token = token.strip("-._")
    return token or "artifact"


def _render_ranking_plot(results: list[dict[str, Any]], *, output_dir: Path) -> dict[str, Any] | None:
    ranked = [item for item in list(results or []) if isinstance(item, dict)]
    if not ranked:
        return None
    top_rows = ranked[: min(12, len(ranked))]
    labels = [str(item.get("file_name") or "image") for item in reversed(top_rows)]
    values = [float(item.get("score") or 0.0) for item in reversed(top_rows)]
    fig_height = max(3.5, 0.48 * len(top_rows) + 1.8)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    bars = ax.barh(labels, values, color="#D97706")
    ax.set_title("Prediction stability ranking under spectral perturbation")
    ax.set_xlabel("Instability score")
    ax.set_ylabel("Image")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(
            float(bar.get_width()) + 0.01,
            float(bar.get_y()) + float(bar.get_height()) / 2.0,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=9,
        )
    fig.tight_layout()
    output_path = output_dir / "spectral_instability_ranking.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "path": str(output_path),
        "title": "Prediction stability ranking under spectral perturbation",
        "caption": (
            "Higher scores mean the detector changed more when low-level spectral content was perturbed. "
            "These are the strongest manual-review candidates."
        ),
        "kind": "image",
    }


def _render_breakdown_plot(
    results: list[dict[str, Any]],
    *,
    output_dir: Path,
    config: SpectralInstabilityConfig,
) -> dict[str, Any] | None:
    ranked = [item for item in list(results or []) if isinstance(item, dict)]
    if not ranked:
        return None
    top_rows = ranked[: min(10, len(ranked))]
    labels = [str(item.get("file_name") or "image") for item in reversed(top_rows)]
    lost = [float(item.get("lost_detection_count") or 0.0) * float(config.lost_weight) for item in reversed(top_rows)]
    new = [float(item.get("new_detection_count") or 0.0) * float(config.new_weight) for item in reversed(top_rows)]
    class_jitter = [float(item.get("class_jitter") or 0.0) * float(config.jitter_weight) for item in reversed(top_rows)]
    spatial_jitter = [float(item.get("spatial_jitter") or 0.0) * float(config.jitter_weight) for item in reversed(top_rows)]
    confidence_jitter = [float(item.get("confidence_jitter") or 0.0) * float(config.confidence_weight) for item in reversed(top_rows)]

    fig_height = max(3.5, 0.5 * len(top_rows) + 1.8)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    left = np.zeros(len(labels), dtype=float)
    for values, label, color in (
        (lost, "Lost detections contribution", "#B91C1C"),
        (class_jitter, "Class jitter contribution", "#D97706"),
        (spatial_jitter, "Spatial jitter contribution", "#2563EB"),
        (confidence_jitter, "Confidence jitter contribution", "#A855F7"),
        (new, "New detections contribution", "#6B7280"),
    ):
        if not any(float(value) > 0 for value in values):
            continue
        ax.barh(labels, values, left=left, label=label, color=color, alpha=0.9)
        left = left + np.asarray(values, dtype=float)
    ax.set_title("What drives the stability score")
    ax.set_xlabel("Weighted contribution to instability")
    ax.set_ylabel("Image")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    output_path = output_dir / "spectral_instability_breakdown.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "path": str(output_path),
        "title": "Why each image ranks as unstable",
        "caption": (
            "The total score is decomposed into lost detections, class changes, and spatial shifts. "
            "This helps distinguish missed objects from less severe box jitter."
        ),
        "kind": "image",
    }


def score_spectral_instability(
    *,
    image_paths: list[str],
    weights_path: str,
    output_dir: str | Path | None = None,
    config: SpectralInstabilityConfig | None = None,
) -> dict[str, Any]:
    resolved_paths = [str(Path(path).expanduser().resolve()) for path in image_paths if str(path).strip()]
    if not resolved_paths:
        return {"success": False, "error": "No image paths were provided."}

    config = config or SpectralInstabilityConfig()
    output_root = Path(output_dir).expanduser().resolve() if output_dir else (
        Path("data") / "spectral_instability" / f"run-{uuid4().hex[:8]}"
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_json_path = output_root / "spectral_scores.json"

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    DetectMultiBackend = _load_yolov5_backend()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda"
    model = DetectMultiBackend(str(Path(weights_path).expanduser().resolve()), device=device, fp16=use_fp16)
    model.eval()
    inner_model = model.model
    detect_layer = inner_model.model[-1]
    detect_sources = list(getattr(detect_layer, "f", []) or [])
    if len(detect_sources) < 1:
        return {"success": False, "error": "Could not determine YOLO detect-head feature sources."}

    feature_maps: dict[int, torch.Tensor] = {}
    hooks = []
    for layer_index in detect_sources:
        hooks.append(
            inner_model.model[int(layer_index)].register_forward_hook(
                lambda _module, _inputs, output, idx=int(layer_index): feature_maps.__setitem__(idx, output)
            )
        )

    try:
        stride = int(model.stride.max()) if hasattr(model.stride, "max") else int(model.stride)
        dataset = _ImagePathDataset(resolved_paths, img_size=config.imgsz, stride=stride)
        loader = DataLoader(dataset, batch_size=int(config.batch_size), shuffle=False)
        scorer = SpectralInstabilityScorer(config=config)
        image_results: list[dict[str, Any]] = []

        for imgs, paths in loader:
            imgs = imgs.to(device)
            imgs = imgs.half() if use_fp16 else imgs.float()
            imgs = imgs / 255.0
            feature_maps.clear()
            outputs = inner_model(imgs)
            preds = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            detect_inputs = [feature_maps[int(index)] for index in detect_sources]
            primary_feature, filter_diagnostics = scorer.apply_adaptive_spectral_filter(detect_inputs[0])
            filtered_inputs = [primary_feature, *detect_inputs[1:]]
            filtered_outputs = detect_layer(filtered_inputs)
            preds_filt = filtered_outputs[0] if isinstance(filtered_outputs, (tuple, list)) else filtered_outputs

            class_names = getattr(inner_model, "names", {}) or {}
            for batch_index, raw_path in enumerate(paths):
                original = _nms_with_class_scores(
                    preds[batch_index],
                    conf_thres=config.conf_thres,
                    iou_thres=config.iou_thres,
                )
                filtered = _nms_with_class_scores(
                    preds_filt[batch_index],
                    conf_thres=config.conf_thres,
                    iou_thres=config.iou_thres,
                )
                breakdown = scorer.discrepancy_breakdown(original, filtered)
                filter_diag = (
                    filter_diagnostics[batch_index]
                    if batch_index < len(filter_diagnostics)
                    else {}
                )
                image_results.append(
                    {
                        "file_name": Path(str(raw_path)).name,
                        "file_path": str(raw_path),
                        "score": float(breakdown["score"]),
                        "normalized_score": float(breakdown["normalized_score"]),
                        "original_detection_count": int(len(original)),
                        "filtered_detection_count": int(len(filtered)),
                        "lost_detection_count": int(breakdown["lost"]),
                        "new_detection_count": int(breakdown["new"]),
                        "matched_detection_count": int(breakdown["matched"]),
                        "class_jitter": float(breakdown["class_jitter"]),
                        "spatial_jitter": float(breakdown["spatial_jitter"]),
                        "confidence_jitter": float(breakdown["confidence_jitter"]),
                        "mean_confidence_shift": float(breakdown["mean_confidence_shift"]),
                        "mean_confidence_drop": float(breakdown["mean_confidence_drop"]),
                        "matched_iou_mean": float(breakdown["matched_iou_mean"]),
                        "matched_iou_median": float(breakdown["matched_iou_median"]),
                        "class_consistency_rate": float(breakdown["class_consistency_rate"]),
                        "retained_energy_fraction": float(filter_diag.get("retained_energy_fraction") or 0.0),
                        "dominant_channel_index": int(filter_diag.get("dominant_channel_index") or 0),
                        "original_class_counts": _class_counts(original, class_names),
                        "filtered_class_counts": _class_counts(filtered, class_names),
                    }
                )
    finally:
        for hook in hooks:
            hook.remove()

    image_results.sort(key=lambda item: (-float(item["score"]), str(item["file_name"])))
    scores = [float(item["score"]) for item in image_results]
    normalized_scores = [float(item.get("normalized_score") or 0.0) for item in image_results]
    nonzero_scores = [value for value in scores if value > 0]
    ranked = [
        {
            "file_name": item["file_name"],
            "score": item["score"],
        }
        for item in image_results[:10]
    ]
    visualization_paths: list[dict[str, Any]] = []
    ranking_plot = _render_ranking_plot(image_results, output_dir=output_root)
    if ranking_plot:
        visualization_paths.append(ranking_plot)
    breakdown_plot = _render_breakdown_plot(image_results, output_dir=output_root, config=config)
    if breakdown_plot:
        visualization_paths.append(breakdown_plot)
    payload = {
        "success": True,
        "method": "spectral_instability",
        "method_version": "block_1_repo_native_v1",
        "weights_path": str(Path(weights_path).expanduser().resolve()),
        "device": str(device),
        "output_json": str(output_json_path),
        "config": {
            "imgsz": int(config.imgsz),
            "batch_size": int(config.batch_size),
            "conf_thres": float(config.conf_thres),
            "iou_thres": float(config.iou_thres),
            "match_iou_thresh": float(config.match_iou_thresh),
            "stable_iou_thresh": float(config.stable_iou_thresh),
            "preservation_ratio": float(config.preservation_ratio),
            "lost_weight": float(config.lost_weight),
            "new_weight": float(config.new_weight),
            "jitter_weight": float(config.jitter_weight),
            "confidence_weight": float(config.confidence_weight),
        },
        "summary": {
            "image_count": len(image_results),
            "nonzero_score_count": len(nonzero_scores),
            "max_score": max(scores) if scores else 0.0,
            "mean_score": mean(scores) if scores else 0.0,
            "median_score": median(scores) if scores else 0.0,
            "max_normalized_score": max(normalized_scores) if normalized_scores else 0.0,
            "mean_normalized_score": mean(normalized_scores) if normalized_scores else 0.0,
            "median_normalized_score": median(normalized_scores) if normalized_scores else 0.0,
            "top_ranked": ranked,
        },
        "results": image_results,
        "visualization_paths": visualization_paths,
    }
    output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
