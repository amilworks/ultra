from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch


@dataclass(frozen=True)
class SpectralInstabilityConfig:
    imgsz: int = 512
    batch_size: int = 4
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    match_iou_thresh: float = 0.50
    stable_iou_thresh: float = 0.70
    preservation_ratio: float = 0.90
    lost_weight: float = 2.0
    new_weight: float = 0.0
    jitter_weight: float = 0.5
    confidence_weight: float = 0.25


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
        return (dist / max_radius <= float(self.config.preservation_ratio)).float()

    @torch.no_grad()
    def apply_adaptive_spectral_filter(
        self,
        feature_map: torch.Tensor,
    ) -> tuple[torch.Tensor, list[dict[str, float | int]]]:
        if feature_map.dtype == torch.float16:
            feature_map = feature_map.float()

        fft_tensor = torch.fft.fftshift(torch.fft.fft2(feature_map, dim=(-2, -1)), dim=(-2, -1))
        filtered: list[torch.Tensor] = []
        diagnostics: list[dict[str, float | int]] = []
        for batch_index in range(int(fft_tensor.shape[0])):
            image_fft = fft_tensor[batch_index]
            channel_energy = torch.sum(torch.abs(image_fft) ** 2, dim=(1, 2))
            dominant_channel = int(torch.argmax(channel_energy).item())
            energy_map = torch.abs(image_fft[dominant_channel]) ** 2
            mask = self.create_energy_mask(energy_map)
            filtered_fft = image_fft * mask.unsqueeze(0)
            retained_energy = float(
                (energy_map * mask).sum().item() / max(float(energy_map.sum().item()), 1e-12)
            )
            restored = torch.fft.ifft2(
                torch.fft.ifftshift(filtered_fft, dim=(-2, -1)),
                dim=(-2, -1),
            ).real
            filtered.append(restored)
            diagnostics.append(
                {
                    "dominant_channel_index": dominant_channel,
                    "retained_energy_fraction": retained_energy,
                }
            )
        return torch.stack(filtered, dim=0), diagnostics

    @staticmethod
    def _greedy_iou_matches(
        ious: torch.Tensor,
        *,
        threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = ious.device
        if ious.numel() == 0:
            empty = torch.empty((0,), device=device, dtype=torch.long)
            return empty, empty, torch.empty((0,), device=device, dtype=torch.float32)
        row_idx, col_idx = torch.where(ious >= float(threshold))
        if row_idx.numel() == 0:
            empty = torch.empty((0,), device=device, dtype=torch.long)
            return empty, empty, torch.empty((0,), device=device, dtype=torch.float32)
        pair_scores = ious[row_idx, col_idx]
        order = torch.argsort(pair_scores, descending=True)
        used_rows = torch.zeros(ious.shape[0], device=device, dtype=torch.bool)
        used_cols = torch.zeros(ious.shape[1], device=device, dtype=torch.bool)
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
            empty = torch.empty((0,), device=device, dtype=torch.long)
            return empty, empty, torch.empty((0,), device=device, dtype=torch.float32)
        return (
            torch.tensor(matched_rows, device=device, dtype=torch.long),
            torch.tensor(matched_cols, device=device, dtype=torch.long),
            torch.tensor(matched_scores, device=device, dtype=torch.float32),
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
        class_ids = class_ids.clamp(max=max(0, int(class_outputs.shape[1]) - 1))
        winning_logits = class_outputs.gather(1, class_ids.unsqueeze(1)).squeeze(1)
        return objectness * torch.sigmoid(winning_logits)

    @torch.no_grad()
    def discrepancy_breakdown(self, pred_orig: torch.Tensor, pred_filt: torch.Tensor) -> dict[str, float]:
        if len(pred_orig) == 0 and len(pred_filt) == 0:
            return _empty_breakdown(score=0.0, class_consistency_rate=1.0)
        if len(pred_orig) == 0:
            new = float(len(pred_filt))
            return _empty_breakdown(new=new, score=float(self.config.new_weight * new))
        if len(pred_filt) == 0:
            lost = float(len(pred_orig))
            score = float(self.config.lost_weight * lost)
            return _empty_breakdown(lost=lost, score=score, normalized_score=score / max(lost, 1.0))

        ious = box_iou(pred_orig[:, :4], pred_filt[:, :4])
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


def _empty_breakdown(
    *,
    lost: float = 0.0,
    new: float = 0.0,
    score: float = 0.0,
    normalized_score: float | None = None,
    class_consistency_rate: float = 0.0,
) -> dict[str, float]:
    return {
        "lost": lost,
        "new": new,
        "class_jitter": 0.0,
        "spatial_jitter": 0.0,
        "confidence_jitter": 0.0,
        "matched": 0.0,
        "normalized_score": float(score if normalized_score is None else normalized_score),
        "matched_iou_mean": 0.0,
        "matched_iou_median": 0.0,
        "class_consistency_rate": class_consistency_rate,
        "mean_confidence_shift": 0.0,
        "mean_confidence_drop": 0.0,
        "score": float(score),
    }


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device, dtype=torch.float32)
    inter_x1 = torch.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = torch.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = torch.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = torch.minimum(a[:, None, 3], b[None, :, 3])
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h
    area_a = torch.clamp(a[:, 2] - a[:, 0], min=0) * torch.clamp(a[:, 3] - a[:, 1], min=0)
    area_b = torch.clamp(b[:, 2] - b[:, 0], min=0) * torch.clamp(b[:, 3] - b[:, 1], min=0)
    denom = torch.clamp(area_a[:, None] + area_b[None, :] - inter_area, min=1e-12)
    return inter_area / denom


def nms(boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), device=boxes.device, dtype=torch.long)
    order = torch.argsort(scores, descending=True)
    keep: list[int] = []
    while int(order.numel()) > 0:
        current = int(order[0].item())
        keep.append(current)
        if int(order.numel()) == 1:
            break
        ious = box_iou(boxes[current].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        order = order[1:][ious <= float(threshold)]
    return torch.tensor(keep, device=boxes.device, dtype=torch.long)


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms_with_class_scores(
    prediction: torch.Tensor,
    *,
    conf_thres: float,
    iou_thres: float,
) -> torch.Tensor:
    num_classes = int(prediction.shape[1] - 5)
    candidates = prediction[prediction[:, 4] > float(conf_thres)]
    if candidates.shape[0] == 0:
        return torch.empty((0, 6 + num_classes), device=prediction.device)
    boxes_xyxy = xywh2xyxy(candidates[:, :4])
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
        class_keep = nms(boxes_xyxy[class_indices], nms_scores[class_indices], float(iou_thres))
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


def score_spectral_instability(
    *,
    image_paths: list[str],
    weights_path: str,
    yolov5_path: str,
    output_dir: str | Path,
    config: SpectralInstabilityConfig,
) -> dict[str, Any]:
    resolved_paths = [str(Path(path).expanduser().resolve()) for path in image_paths if str(path).strip()]
    if not resolved_paths:
        return {"success": False, "error": "No image paths were provided."}

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_json_path = output_root / "spectral_scores.json"

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    yolov5_repo = str(Path(yolov5_path).expanduser().resolve())
    if yolov5_repo not in sys.path:
        sys.path.insert(0, yolov5_repo)
    from models.common import DetectMultiBackend  # type: ignore
    from torch.utils.data import DataLoader, Dataset
    from utils.dataloaders import LoadImages  # type: ignore

    class ImagePathDataset(Dataset):
        def __init__(self, paths: list[str]) -> None:
            self.paths = paths

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
            path = self.paths[idx]
            _, image, _, _, _ = next(
                iter(LoadImages(path, img_size=int(config.imgsz), stride=32, auto=False))
            )
            return torch.from_numpy(image.copy()), path

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
    hooks = [
        inner_model.model[int(layer_index)].register_forward_hook(
            lambda _module, _inputs, output, idx=int(layer_index): feature_maps.__setitem__(idx, output)
        )
        for layer_index in detect_sources
    ]
    image_results: list[dict[str, Any]] = []
    try:
        loader = DataLoader(ImagePathDataset(resolved_paths), batch_size=int(config.batch_size), shuffle=False)
        scorer = SpectralInstabilityScorer(config=config)
        for imgs, paths in loader:
            imgs = imgs.to(device)
            imgs = imgs.half() if use_fp16 else imgs.float()
            imgs = imgs / 255.0
            feature_maps.clear()
            outputs = inner_model(imgs)
            preds = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            detect_inputs = [feature_maps[int(index)] for index in detect_sources]
            primary_feature, diagnostics = scorer.apply_adaptive_spectral_filter(detect_inputs[0])
            filtered_outputs = detect_layer([primary_feature, *detect_inputs[1:]])
            preds_filt = filtered_outputs[0] if isinstance(filtered_outputs, (tuple, list)) else filtered_outputs
            class_names = getattr(inner_model, "names", {}) or {}
            for batch_index, raw_path in enumerate(paths):
                original = nms_with_class_scores(
                    preds[batch_index],
                    conf_thres=config.conf_thres,
                    iou_thres=config.iou_thres,
                )
                filtered = nms_with_class_scores(
                    preds_filt[batch_index],
                    conf_thres=config.conf_thres,
                    iou_thres=config.iou_thres,
                )
                breakdown = scorer.discrepancy_breakdown(original, filtered)
                diag = diagnostics[batch_index] if batch_index < len(diagnostics) else {}
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
                        "retained_energy_fraction": float(diag.get("retained_energy_fraction") or 0.0),
                        "dominant_channel_index": int(diag.get("dominant_channel_index") or 0),
                        "original_class_counts": class_counts(original, class_names),
                        "filtered_class_counts": class_counts(filtered, class_names),
                    }
                )
    finally:
        for hook in hooks:
            hook.remove()

    image_results.sort(key=lambda item: (-float(item["score"]), str(item["file_name"])))
    visualization_paths = render_spectral_plots(image_results, output_root, config)
    scores = [float(item["score"]) for item in image_results]
    normalized_scores = [float(item.get("normalized_score") or 0.0) for item in image_results]
    nonzero_scores = [score for score in scores if score > 0]
    payload = {
        "success": True,
        "method": "spectral_instability",
        "method_version": "block_1_feature_hook_v1",
        "weights_path": str(Path(weights_path).expanduser().resolve()),
        "device": str(device),
        "output_json": str(output_json_path),
        "config": config.__dict__,
        "summary": {
            "image_count": len(image_results),
            "nonzero_score_count": len(nonzero_scores),
            "max_score": max(scores) if scores else 0.0,
            "mean_score": mean(scores) if scores else 0.0,
            "median_score": median(scores) if scores else 0.0,
            "max_normalized_score": max(normalized_scores) if normalized_scores else 0.0,
            "mean_normalized_score": mean(normalized_scores) if normalized_scores else 0.0,
            "median_normalized_score": median(normalized_scores) if normalized_scores else 0.0,
            "top_ranked": [{"file_name": item["file_name"], "score": item["score"]} for item in image_results[:10]],
        },
        "results": image_results,
        "visualization_paths": visualization_paths,
    }
    output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def class_counts(detections: torch.Tensor, class_names: dict[int, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    if len(detections) == 0:
        return counts
    for class_id in detections[:, 5].tolist():
        label = str(class_names.get(int(class_id), str(int(class_id))))
        counts[label] = counts.get(label, 0) + 1
    return counts


def render_spectral_plots(
    results: list[dict[str, Any]],
    output_dir: Path,
    config: SpectralInstabilityConfig,
) -> list[dict[str, Any]]:
    if not results:
        return []
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    visualizations: list[dict[str, Any]] = []
    top_rows = results[: min(12, len(results))]
    labels = [str(item.get("file_name") or "image") for item in reversed(top_rows)]
    values = [float(item.get("score") or 0.0) for item in reversed(top_rows)]
    fig, ax = plt.subplots(figsize=(11, max(3.5, 0.48 * len(top_rows) + 1.8)))
    ax.barh(labels, values, color="#D97706")
    ax.set_title("Prediction stability ranking under spectral perturbation")
    ax.set_xlabel("Instability score")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    fig.tight_layout()
    ranking_path = output_dir / "spectral_instability_ranking.png"
    fig.savefig(ranking_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    visualizations.append({"path": str(ranking_path), "kind": "image", "title": "Spectral ranking"})

    labels = [str(item.get("file_name") or "image") for item in reversed(results[: min(10, len(results))])]
    components = [
        ("lost", [float(item.get("lost_detection_count") or 0.0) * config.lost_weight for item in reversed(results[: min(10, len(results))])], "#B91C1C"),
        ("class jitter", [float(item.get("class_jitter") or 0.0) * config.jitter_weight for item in reversed(results[: min(10, len(results))])], "#D97706"),
        ("spatial jitter", [float(item.get("spatial_jitter") or 0.0) * config.jitter_weight for item in reversed(results[: min(10, len(results))])], "#2563EB"),
        ("confidence shift", [float(item.get("confidence_jitter") or 0.0) * config.confidence_weight for item in reversed(results[: min(10, len(results))])], "#A855F7"),
    ]
    fig, ax = plt.subplots(figsize=(11.5, max(3.5, 0.5 * len(labels) + 1.8)))
    left = [0.0] * len(labels)
    for label, values, color in components:
        if any(value > 0.0 for value in values):
            ax.barh(labels, values, left=left, label=label, color=color, alpha=0.9)
            left = [base + value for base, value in zip(left, values)]
    ax.set_title("What drives the stability score")
    ax.set_xlabel("Weighted instability contribution")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    breakdown_path = output_dir / "spectral_instability_breakdown.png"
    fig.savefig(breakdown_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    visualizations.append({"path": str(breakdown_path), "kind": "image", "title": "Spectral breakdown"})
    return visualizations
