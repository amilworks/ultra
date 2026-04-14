"""MedSAM2 inference adapters for 2D/3D scientific arrays.

This module prefers the official MedSAM2 checkpoint backend (local `.pt`
weights loaded via the `sam2` package) and falls back to a transformers
SAM2 pathway only when a Hugging Face model id is explicitly used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image

from src.config import get_settings

_MEDSAM2_HF_CACHE: dict[tuple[str, str, str], tuple[Any, Any, Any]] = {}
_MEDSAM2_OFFICIAL_CACHE: dict[tuple[str, str, str], Any] = {}

_MODEL_ALIAS_TO_CHECKPOINT = {
    "default": "MedSAM2_latest.pt",
    "latest": "MedSAM2_latest.pt",
    "medsam2_latest": "MedSAM2_latest.pt",
    "medsam2": "MedSAM2_latest.pt",
    "wanglab/medsam2": "MedSAM2_latest.pt",
    "2411": "MedSAM2_2411.pt",
    "base": "MedSAM2_2411.pt",
    "us_heart": "MedSAM2_US_Heart.pt",
    "heart_us": "MedSAM2_US_Heart.pt",
    "mri_liver_lesion": "MedSAM2_MRI_LiverLesion.pt",
    "liver": "MedSAM2_MRI_LiverLesion.pt",
    "ct_lesion": "MedSAM2_CTLesion.pt",
}


@dataclass(frozen=True)
class _ModelRuntime:
    backend: str
    resolved_ref: str
    display_name: str
    warnings: tuple[str, ...] = ()


def _torch_available() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch is not None)
    except Exception:
        return False


def _vendored_medsam2_root() -> Path | None:
    settings = get_settings()
    configured = str(getattr(settings, "medsam2_runtime_root", "") or "").strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(Path(__file__).resolve().parents[2] / "third_party" / "MedSAM2")
    candidates.append(Path(__file__).resolve().parents[2] / "data" / "runtime" / "MedSAM2")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _vendored_checkpoint_root() -> Path | None:
    root = _vendored_medsam2_root()
    if root is None:
        return None
    candidate = root / "checkpoints"
    if candidate.exists():
        return candidate
    return None


def _ensure_vendored_sam2_on_path() -> Path | None:
    root = _vendored_medsam2_root()
    if root is None:
        return None
    root_str = str(root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _transformers_available() -> bool:
    try:
        import transformers  # type: ignore

        return bool(transformers is not None)
    except Exception:
        return False


def _official_backend_available() -> bool:
    try:
        _ensure_vendored_sam2_on_path()
        import sam2  # type: ignore
        from sam2.build_sam import build_sam2  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

        return bool(sam2 is not None and build_sam2 is not None and SAM2ImagePredictor is not None)
    except Exception:
        return False


def _resolve_device(device: str | None) -> str:
    try:
        import torch  # type: ignore

        has_cuda = bool(torch.cuda.is_available())
    except Exception:
        has_cuda = False

    if not device:
        return "cuda" if has_cuda else "cpu"

    value = str(device).strip().lower()
    if value in {"cpu", "-1"}:
        return "cpu"
    if value in {"cuda", "gpu"}:
        if not has_cuda:
            raise ValueError("CUDA requested but not available.")
        return "cuda"
    if value.startswith("cuda:"):
        if not has_cuda:
            raise ValueError("CUDA requested but not available.")
        return value
    if value.isdigit():
        if not has_cuda:
            raise ValueError("CUDA requested but not available.")
        return f"cuda:{value}"
    raise ValueError(f"Invalid device value: {device!r}. Use cpu, cuda, cuda:N, or N.")


def _normalize_uint8(arr: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    x = np.asarray(arr)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros(x.shape, dtype=np.uint8)
    lo = float(np.percentile(finite, lo_pct))
    hi = float(np.percentile(finite, hi_pct))
    if hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        return np.zeros(x.shape, dtype=np.uint8)
    y = np.clip((x.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)


def _slice_to_rgb_uint8(slice_data: np.ndarray, order: str) -> np.ndarray:
    arr = np.asarray(slice_data)
    ord_chars = [ch for ch in str(order).upper() if ch.isalpha()]

    # Drop any axes we do not explicitly support for per-slice inference.
    for axis in list(ord_chars):
        if axis not in {"C", "Y", "X"}:
            idx = ord_chars.index(axis)
            arr = np.take(arr, 0, axis=idx)
            ord_chars.pop(idx)

    if "Y" not in ord_chars or "X" not in ord_chars:
        # Fallback: flatten all leading dimensions into one pseudo-channel.
        if arr.ndim >= 2:
            arr2d = arr.reshape((-1,) + arr.shape[-2:])[0]
            gray = _normalize_uint8(arr2d)
            return np.repeat(gray[..., None], 3, axis=-1)
        return np.zeros((1, 1, 3), dtype=np.uint8)

    if "C" in ord_chars:
        perm = [ord_chars.index("C"), ord_chars.index("Y"), ord_chars.index("X")]
        cyx = np.transpose(arr, perm)
        if cyx.shape[0] >= 3:
            rgb = np.stack([_normalize_uint8(cyx[i]) for i in range(3)], axis=-1)
            return rgb
        gray = _normalize_uint8(cyx[0])
        return np.repeat(gray[..., None], 3, axis=-1)

    perm = [ord_chars.index("Y"), ord_chars.index("X")]
    yx = np.transpose(arr, perm)
    gray = _normalize_uint8(yx)
    return np.repeat(gray[..., None], 3, axis=-1)


def _wrap_points(points: list[Any] | None) -> list[list[list[list[float]]]] | None:
    if not points:
        return None
    if (
        isinstance(points, list)
        and points
        and isinstance(points[0], (list, tuple))
        and len(points[0]) == 2
        and isinstance(points[0][0], (int, float))
    ):
        points_by_object = [points]
    else:
        points_by_object = points
    return [points_by_object]


def _wrap_labels(labels: list[Any] | None, points: list[Any] | None) -> list[list[list[int]]] | None:
    if labels:
        if isinstance(labels[0], list):
            return [labels]  # Already per-object labels.
        return [[labels]]
    if points:
        if (
            isinstance(points, list)
            and points
            and isinstance(points[0], (list, tuple))
            and len(points[0]) == 2
            and isinstance(points[0][0], (int, float))
        ):
            return [[[1 for _ in points]]]
        if isinstance(points[0], list):
            return [[[1 for _ in obj] for obj in points]]
    return None


def _wrap_boxes(boxes: list[Any] | None) -> list[list[list[float]]] | None:
    if not boxes:
        return None
    if (
        isinstance(boxes, list)
        and boxes
        and isinstance(boxes[0], (list, tuple))
        and len(boxes[0]) == 4
        and isinstance(boxes[0][0], (int, float))
    ):
        return [boxes]
    return boxes


def _sanitize_point_pair(point: Any, *, width: int, height: int) -> list[float] | None:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return None
    x = float(point[0])
    y = float(point[1])
    x = min(max(x, 0.0), float(max(width - 1, 0)))
    y = min(max(y, 0.0), float(max(height - 1, 0)))
    return [x, y]


def _sanitize_points(points: list[Any] | None, width: int, height: int) -> list[Any] | None:
    if not points:
        return None
    if (
        isinstance(points, list)
        and points
        and isinstance(points[0], (list, tuple))
        and len(points[0]) == 2
        and isinstance(points[0][0], (int, float))
    ):
        out: list[list[float]] = []
        for point in points:
            normalized = _sanitize_point_pair(point, width=width, height=height)
            if normalized is not None:
                out.append(normalized)
        return out or None

    grouped: list[list[list[float]]] = []
    for group in points:
        if not isinstance(group, (list, tuple)):
            continue
        normalized_group: list[list[float]] = []
        for point in group:
            normalized = _sanitize_point_pair(point, width=width, height=height)
            if normalized is not None:
                normalized_group.append(normalized)
        if normalized_group:
            grouped.append(normalized_group)
    return grouped or None


def _sanitize_labels(labels: list[Any] | None, points: list[Any] | None) -> list[Any] | None:
    if not labels:
        return None
    if not points:
        return None
    if (
        isinstance(points, list)
        and points
        and isinstance(points[0], (list, tuple))
        and len(points[0]) == 2
        and isinstance(points[0][0], (int, float))
    ):
        normalized: list[int] = []
        for value in labels:
            try:
                normalized.append(1 if int(value) > 0 else 0)
            except Exception:
                normalized.append(1)
        return normalized or None

    grouped: list[list[int]] = []
    for group in labels:
        if not isinstance(group, (list, tuple)):
            continue
        normalized_group: list[int] = []
        for value in group:
            try:
                normalized_group.append(1 if int(value) > 0 else 0)
            except Exception:
                normalized_group.append(1)
        if normalized_group:
            grouped.append(normalized_group)
    return grouped or None


def _sanitize_boxes(boxes: list[list[float]] | None, width: int, height: int) -> list[list[float]] | None:
    if not boxes:
        return None
    out: list[list[float]] = []
    for box in boxes:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        x1 = min(max(x1, 0.0), float(max(width - 1, 0)))
        x2 = min(max(x2, 0.0), float(max(width - 1, 0)))
        y1 = min(max(y1, 0.0), float(max(height - 1, 0)))
        y2 = min(max(y2, 0.0), float(max(height - 1, 0)))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        out.append([x1, y1, x2, y2])
    return out or None


def _is_grouped_point_list(points: list[Any] | None) -> bool:
    return bool(
        isinstance(points, list)
        and points
        and isinstance(points[0], (list, tuple))
        and points[0]
        and isinstance(points[0][0], (list, tuple))
    )


def _build_prompt_objects(
    *,
    points: list[Any] | None,
    labels: list[Any] | None,
    boxes: list[list[float]] | None,
) -> list[dict[str, Any]]:
    point_groups: list[list[list[float]]] = []
    label_groups: list[list[int]] = []

    if _is_grouped_point_list(points):
        point_groups = [
            [[float(point[0]), float(point[1])] for point in group]
            for group in points
            if isinstance(group, (list, tuple)) and group
        ]
        if isinstance(labels, list) and labels and isinstance(labels[0], list):
            label_groups = [
                [1 if int(value) > 0 else 0 for value in group]
                for group in labels
                if isinstance(group, (list, tuple))
            ]
    elif points:
        point_groups = [[ [float(point[0]), float(point[1])] for point in points if isinstance(point, (list, tuple)) and len(point) == 2 ]]
        if isinstance(labels, list) and labels:
            try:
                label_groups = [[1 if int(value) > 0 else 0 for value in labels]]
            except Exception:
                label_groups = []

    if point_groups and not label_groups:
        label_groups = [[1 for _ in group] for group in point_groups]

    normalized_boxes = list(boxes or [])
    prompt_objects: list[dict[str, Any]] = []

    if point_groups:
        if normalized_boxes and len(point_groups) == 1 and len(normalized_boxes) > 1:
            points_for_group = point_groups[0]
            labels_for_group = label_groups[0] if label_groups else [1 for _ in points_for_group]
            for box in normalized_boxes:
                prompt_objects.append(
                    {
                        "points": points_for_group,
                        "labels": labels_for_group,
                        "box": box,
                    }
                )
            return prompt_objects

        shared_box = normalized_boxes[0] if len(normalized_boxes) == 1 else None
        for idx, group in enumerate(point_groups):
            labels_for_group = (
                label_groups[idx]
                if idx < len(label_groups)
                else [1 for _ in group]
            )
            object_box = (
                normalized_boxes[idx]
                if len(normalized_boxes) == len(point_groups) and idx < len(normalized_boxes)
                else shared_box
            )
            prompt_objects.append(
                {
                    "points": group,
                    "labels": labels_for_group,
                    "box": object_box,
                }
            )
        return prompt_objects

    if normalized_boxes:
        for box in normalized_boxes:
            prompt_objects.append({"points": None, "labels": None, "box": box})
        return prompt_objects

    return []


def _compose_prompt_instance_label_map(
    object_masks: list[np.ndarray],
) -> tuple[np.ndarray, list[int]]:
    if not object_masks:
        return np.zeros((1, 1), dtype=np.uint8), []

    first_shape = tuple(np.asarray(object_masks[0]).shape)
    if len(first_shape) < 2:
        return np.zeros((1, 1), dtype=np.uint8), []

    label_dtype = np.uint16 if len(object_masks) > np.iinfo(np.uint8).max else np.uint8
    label_map = np.zeros(first_shape, dtype=label_dtype)
    instance_sizes: list[int] = []

    for index, candidate_mask in enumerate(object_masks, start=1):
        candidate = np.asarray(candidate_mask) > 0
        if candidate.shape != label_map.shape:
            continue
        assignable = candidate & (label_map == 0)
        assign_count = int(np.count_nonzero(assignable))
        if assign_count <= 0:
            continue
        label_map[assignable] = index
        instance_sizes.append(assign_count)

    return label_map, instance_sizes


def _checkpoint_root() -> Path:
    settings = get_settings()
    configured = Path(
        str(getattr(settings, "medsam2_checkpoint_dir", "data/models/medsam2/checkpoints"))
    ).expanduser()
    vendored = _vendored_checkpoint_root()
    if configured.exists():
        configured_has_weights = any(configured.glob("*.pt"))
        if configured_has_weights or vendored is None or not vendored.exists():
            return configured
    if vendored is not None and vendored.exists() and any(vendored.glob("*.pt")):
        return vendored
    return configured


def _default_checkpoint_name() -> str:
    settings = get_settings()
    value = str(getattr(settings, "medsam2_default_checkpoint", "MedSAM2_latest.pt")).strip()
    return value or "MedSAM2_latest.pt"


def _default_config_name() -> str:
    settings = get_settings()
    value = str(getattr(settings, "medsam2_config_file", "sam2.1_hiera_t512.yaml")).strip()
    return value or "sam2.1_hiera_t512.yaml"


def _candidate_config_names(config_name: str) -> list[str]:
    raw = str(config_name or "").strip()
    if not raw:
        raw = "sam2.1_hiera_t512.yaml"
    candidates: list[str] = [raw]
    basename = Path(raw).name
    if basename not in candidates:
        candidates.append(basename)
    prefixed = f"configs/{basename}"
    if prefixed not in candidates:
        candidates.append(prefixed)
    vendored_root = _ensure_vendored_sam2_on_path()
    if vendored_root is not None:
        for candidate in (
            vendored_root / "sam2" / "configs" / raw,
            vendored_root / "sam2" / "configs" / basename,
            vendored_root / "configs" / basename,
        ):
            candidate_str = str(candidate.resolve())
            if candidate.exists() and candidate_str not in candidates:
                candidates.append(candidate_str)
    return candidates


def _resolve_checkpoint_candidate(raw_model_id: str) -> tuple[str | None, list[str]]:
    warnings: list[str] = []
    root = _checkpoint_root()
    model_ref = str(raw_model_id or "").strip()
    normalized = model_ref.lower()

    def _resolve_path(candidate: Path) -> Path:
        candidate = candidate.expanduser()
        if candidate.is_absolute():
            return candidate
        local_candidate = (Path.cwd() / candidate).resolve()
        if local_candidate.exists():
            return local_candidate
        rooted = (root / candidate.name).resolve()
        return rooted

    if normalized in _MODEL_ALIAS_TO_CHECKPOINT:
        candidate = (root / _MODEL_ALIAS_TO_CHECKPOINT[normalized]).resolve()
        if candidate.exists():
            return str(candidate), warnings
        warnings.append(
            f"Requested MedSAM2 checkpoint alias '{model_ref}' but file not found at {candidate}."
        )
        return None, warnings

    if model_ref.lower().endswith(".pt"):
        candidate = _resolve_path(Path(model_ref))
        if candidate.exists():
            return str(candidate), warnings
        warnings.append(f"Requested MedSAM2 checkpoint path not found: {candidate}")
        return None, warnings

    return None, warnings


def _resolve_model_runtime(model_id: str | None) -> _ModelRuntime:
    settings = get_settings()
    requested = str(model_id or getattr(settings, "medsam2_model_id", "wanglab/MedSAM2")).strip()
    if not requested:
        requested = "wanglab/MedSAM2"

    normalized = requested.lower()
    if normalized in _MODEL_ALIAS_TO_CHECKPOINT:
        checkpoint_path, warnings = _resolve_checkpoint_candidate(requested)
        if checkpoint_path:
            return _ModelRuntime(
                backend="medsam2-official",
                resolved_ref=checkpoint_path,
                display_name=Path(checkpoint_path).name,
                warnings=tuple(warnings),
            )
        return _ModelRuntime(
            backend="medsam2-missing-checkpoint",
            resolved_ref=requested,
            display_name=requested,
            warnings=tuple(warnings),
        )

    checkpoint_path, warnings = _resolve_checkpoint_candidate(requested)
    if checkpoint_path:
        return _ModelRuntime(
            backend="medsam2-official",
            resolved_ref=checkpoint_path,
            display_name=Path(checkpoint_path).name,
            warnings=tuple(warnings),
        )

    # Non-checkpoint values are interpreted as Hugging Face model ids.
    return _ModelRuntime(
        backend="medsam2-transformers",
        resolved_ref=requested,
        display_name=requested,
        warnings=tuple(warnings),
    )


def _get_hf_predictor(model_id: str, device: str) -> tuple[Any, Any, Any]:
    import torch  # type: ignore
    from transformers import Sam2Model, Sam2Processor  # type: ignore

    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    cache_key = (str(model_id), str(device), str(dtype))
    if cache_key in _MEDSAM2_HF_CACHE:
        return _MEDSAM2_HF_CACHE[cache_key]

    model = Sam2Model.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=False,
    )
    model.to(device)
    model.eval()
    processor = Sam2Processor.from_pretrained(model_id)
    _MEDSAM2_HF_CACHE[cache_key] = (model, processor, dtype)
    return model, processor, dtype


def _get_official_predictor(checkpoint_path: str, config_name: str, device: str) -> Any:
    _ensure_vendored_sam2_on_path()
    from sam2.build_sam import build_sam2  # type: ignore
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

    errors: list[str] = []
    for cfg in _candidate_config_names(config_name):
        cache_key = (str(checkpoint_path), str(cfg), str(device))
        if cache_key in _MEDSAM2_OFFICIAL_CACHE:
            return _MEDSAM2_OFFICIAL_CACHE[cache_key]

        try:
            sam_model = build_sam2(config_file=cfg, ckpt_path=checkpoint_path, device=device)
            predictor = SAM2ImagePredictor(sam_model)
            _MEDSAM2_OFFICIAL_CACHE[cache_key] = predictor
            return predictor
        except Exception as exc:
            errors.append(f"{cfg}: {exc}")

    joined = "; ".join(errors)
    raise RuntimeError(f"Failed to initialize official MedSAM2 predictor. Tried configs: {joined}")


def _predict_single_slice_hf(
    image_rgb: np.ndarray,
    *,
    model_id: str,
    device: str,
    input_points: list[Any] | None,
    input_labels: list[Any] | None,
    input_boxes: list[list[float]] | None,
    multimask_output: bool,
) -> tuple[np.ndarray, float]:
    import torch  # type: ignore

    model, processor, dtype = _get_hf_predictor(model_id, device)

    height, width = int(image_rgb.shape[0]), int(image_rgb.shape[1])
    points = _sanitize_points(input_points, width=width, height=height)
    boxes = _sanitize_boxes(input_boxes, width=width, height=height)
    labels = _sanitize_labels(input_labels, points)

    if not points and not boxes:
        points = [[float(width // 2), float(height // 2)]]
        labels = [1]
    if points and not labels:
        labels = [1 for _ in points]

    prompt_objects = _build_prompt_objects(points=points, labels=labels, boxes=boxes)
    if not prompt_objects:
        prompt_objects = [{"points": points, "labels": labels, "box": None}]

    pil_img = Image.fromarray(np.asarray(image_rgb).astype(np.uint8), mode="RGB")
    selected_masks: list[np.ndarray] = []
    selected_scores: list[float] = []

    for prompt_object in prompt_objects:
        wrapped_points = _wrap_points(prompt_object.get("points"))
        wrapped_labels = _wrap_labels(
            prompt_object.get("labels"),
            prompt_object.get("points"),
        )
        object_box = prompt_object.get("box")
        wrapped_boxes = _wrap_boxes([object_box]) if object_box is not None else None

        inputs = processor(
            images=pil_img,
            input_points=wrapped_points,
            input_labels=wrapped_labels,
            input_boxes=wrapped_boxes,
            return_tensors="pt",
        )

        for key, value in list(inputs.items()):
            if key in ("original_sizes", "reshaped_input_sizes"):
                continue
            if hasattr(value, "to"):
                if torch.is_floating_point(value):
                    inputs[key] = value.to(device=device, dtype=dtype)
                else:
                    inputs[key] = value.to(device=device)

        with torch.no_grad():
            outputs = model(**inputs, multimask_output=bool(multimask_output))

        masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        scores = outputs.iou_scores.squeeze(0).cpu().numpy()

        masks_np = np.asarray(masks)
        if masks_np.ndim == 2:
            masks_np = masks_np[None, ...]
        scores_np = np.asarray(scores, dtype=np.float32).reshape(-1)
        best_idx = int(np.argmax(scores_np)) if scores_np.size else 0
        best_idx = min(max(best_idx, 0), masks_np.shape[0] - 1)
        selected_masks.append(np.asarray(masks_np[best_idx]) > 0)
        selected_scores.append(float(scores_np[best_idx]) if scores_np.size else 0.0)

    if not selected_masks:
        return np.zeros((height, width), dtype=np.uint8), 0.0

    combined, _ = _compose_prompt_instance_label_map(selected_masks)
    score = float(np.mean(selected_scores)) if selected_scores else 0.0
    return combined, score


def _official_predict_prompt(
    predictor: Any,
    *,
    points: list[list[float]] | None,
    labels: list[int] | None,
    box: list[float] | None,
    multimask_output: bool,
) -> tuple[np.ndarray, float]:
    point_coords = np.asarray(points, dtype=np.float32) if points else None
    point_labels = np.asarray(labels, dtype=np.int32) if points else None
    box_arr = np.asarray(box, dtype=np.float32) if box is not None else None

    masks, iou_scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box_arr,
        multimask_output=bool(multimask_output),
        return_logits=False,
        normalize_coords=True,
    )

    masks_np = np.asarray(masks)
    iou_np = np.asarray(iou_scores, dtype=np.float32).reshape(-1)

    if masks_np.ndim == 2:
        score = float(iou_np[0]) if iou_np.size else 0.0
        return (masks_np > 0).astype(np.uint8), score
    if masks_np.ndim != 3 or masks_np.shape[0] <= 0:
        return np.zeros((1, 1), dtype=np.uint8), 0.0

    best_idx = int(np.argmax(iou_np)) if iou_np.size else 0
    best_idx = min(max(best_idx, 0), masks_np.shape[0] - 1)
    score = float(iou_np[best_idx]) if iou_np.size else 0.0
    return (masks_np[best_idx] > 0).astype(np.uint8), score


def _predict_single_slice_official(
    image_rgb: np.ndarray,
    *,
    checkpoint_path: str,
    config_name: str,
    device: str,
    input_points: list[Any] | None,
    input_labels: list[Any] | None,
    input_boxes: list[list[float]] | None,
    multimask_output: bool,
) -> tuple[np.ndarray, float]:
    predictor = _get_official_predictor(checkpoint_path, config_name, device)

    height, width = int(image_rgb.shape[0]), int(image_rgb.shape[1])
    points = _sanitize_points(input_points, width=width, height=height)
    boxes = _sanitize_boxes(input_boxes, width=width, height=height)
    labels = _sanitize_labels(input_labels, points)

    if not points and not boxes:
        points = [[float(width // 2), float(height // 2)]]
        labels = [1]
    if points and not labels:
        labels = [1 for _ in points]

    predictor.set_image(np.asarray(image_rgb).astype(np.uint8))
    prompt_objects = _build_prompt_objects(points=points, labels=labels, boxes=boxes)
    if not prompt_objects:
        prompt_objects = [{"points": points, "labels": labels, "box": None}]

    masks_to_merge: list[np.ndarray] = []
    scores: list[float] = []

    for prompt_object in prompt_objects:
        mask, score = _official_predict_prompt(
            predictor,
            points=prompt_object.get("points"),
            labels=prompt_object.get("labels"),
            box=prompt_object.get("box"),
            multimask_output=multimask_output,
        )
        if mask.shape != (height, width):
            mask = np.array(
                Image.fromarray(mask.astype(np.uint8)).resize((width, height), Image.NEAREST)
            )
        masks_to_merge.append(mask > 0)
        scores.append(float(score))

    if not masks_to_merge:
        return np.zeros((height, width), dtype=np.uint8), 0.0

    combined, _ = _compose_prompt_instance_label_map(masks_to_merge)
    score = float(np.mean(scores)) if scores else 0.0
    return combined, score


def _canonical_slices(array: np.ndarray, order: str) -> tuple[np.ndarray, str]:
    arr = np.asarray(array)
    ord_chars = [ch for ch in str(order).upper() if ch.isalpha()]

    # Drop batch/time dimension by selecting first element.
    for axis in ("T", "B"):
        if axis in ord_chars:
            idx = ord_chars.index(axis)
            arr = np.take(arr, 0, axis=idx)
            ord_chars.pop(idx)

    if "Z" in ord_chars:
        z_axis = ord_chars.index("Z")
        arr = np.moveaxis(arr, z_axis, 0)
        ord_chars.pop(z_axis)
        slice_order = "".join(ord_chars)
        return arr, slice_order

    return arr[None, ...], "".join(ord_chars)


def segment_array_with_medsam2(
    array: np.ndarray,
    *,
    order: str,
    input_points: list[Any] | None = None,
    input_labels: list[Any] | None = None,
    input_boxes: list[list[float]] | None = None,
    multimask_output: bool = True,
    model_id: str | None = None,
    device: str | None = None,
    max_slices: int = 160,
) -> dict[str, Any]:
    """Segment 2D or 3D arrays with MedSAM2 (slice-wise for 3D volumes)."""
    if not _torch_available():
        return {
            "success": False,
            "error": "Torch runtime unavailable for MedSAM2 inference.",
        }

    runtime = _resolve_model_runtime(model_id)
    warnings: list[str] = list(runtime.warnings)

    try:
        resolved_device = _resolve_device(device)
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    if runtime.backend == "medsam2-missing-checkpoint":
        return {
            "success": False,
            "error": (
                f"Requested MedSAM2 alias '{runtime.resolved_ref}' but the local checkpoint is missing. "
                "Download/place the checkpoint under medsam2_checkpoint_dir or set MEDSAM2_MODEL_ID to a valid path/model id."
            ),
            "warnings": warnings,
        }

    config_name = _default_config_name()
    if runtime.backend == "medsam2-official":
        if not _official_backend_available():
            return {
                "success": False,
                "error": (
                    "Official MedSAM2 checkpoint backend is unavailable. Install the MedSAM2 package "
                    "(with sam2/hydra dependencies) or provide a Hugging Face SAM2 model id."
                ),
                "warnings": warnings,
            }
        try:
            _get_official_predictor(runtime.resolved_ref, config_name, resolved_device)
        except Exception as exc:
            return {
                "success": False,
                "error": (
                    f"Failed to load MedSAM2 checkpoint '{runtime.resolved_ref}' on {resolved_device}: {exc}"
                ),
                "warnings": warnings,
            }
    else:
        if not _transformers_available():
            return {
                "success": False,
                "error": (
                    "Transformers runtime unavailable for Hugging Face SAM2 fallback inference."
                ),
                "warnings": warnings,
            }
        try:
            _get_hf_predictor(runtime.resolved_ref, resolved_device)
        except Exception as exc:
            return {
                "success": False,
                "error": (
                    f"Failed to load SAM2 model '{runtime.resolved_ref}' on {resolved_device}: {exc}"
                ),
                "warnings": warnings,
            }

    arr = np.asarray(array)
    if arr.size == 0:
        return {"success": False, "error": "Input array is empty.", "warnings": warnings}

    slices, slice_order = _canonical_slices(arr, order=str(order))
    if slices.ndim < 3:
        return {
            "success": False,
            "error": f"Invalid canonical slice tensor shape: {slices.shape}",
            "warnings": warnings,
        }

    z_count = int(slices.shape[0])
    if z_count <= 0:
        return {"success": False, "error": "No slices available for segmentation.", "warnings": warnings}

    resolved_max_slices: int | None = None
    if max_slices is not None:
        try:
            resolved_max_slices = int(max_slices)
        except Exception:
            resolved_max_slices = None

    stride = 1
    if resolved_max_slices is not None and resolved_max_slices > 0 and z_count > resolved_max_slices:
        stride = int(np.ceil(z_count / float(resolved_max_slices)))
    indices = list(range(0, z_count, stride))
    processed_all_slices = stride == 1 and len(indices) == z_count

    mask_stack = np.zeros((z_count, int(slices.shape[-2]), int(slices.shape[-1])), dtype=np.uint16)
    score_by_slice: dict[int, float] = {}

    for z in indices:
        rgb = _slice_to_rgb_uint8(slices[z], slice_order)
        try:
            if runtime.backend == "medsam2-official":
                pred_mask, pred_score = _predict_single_slice_official(
                    rgb,
                    checkpoint_path=runtime.resolved_ref,
                    config_name=config_name,
                    device=resolved_device,
                    input_points=input_points,
                    input_labels=input_labels,
                    input_boxes=input_boxes,
                    multimask_output=multimask_output,
                )
            else:
                pred_mask, pred_score = _predict_single_slice_hf(
                    rgb,
                    model_id=runtime.resolved_ref,
                    device=resolved_device,
                    input_points=input_points,
                    input_labels=input_labels,
                    input_boxes=input_boxes,
                    multimask_output=multimask_output,
                )
            mask_stack[z] = pred_mask.astype(np.uint8)
            score_by_slice[z] = float(pred_score)
        except Exception as exc:
            warnings.append(f"slice {z}: {exc}")

    if not score_by_slice:
        return {
            "success": False,
            "error": "MedSAM2 inference failed for all slices.",
            "warnings": warnings,
        }

    # Nearest-slice fill for skipped slices when stride > 1.
    if stride > 1 and indices:
        warnings.append(
            "MedSAM2 processed "
            f"{len(indices)}/{z_count} slice(s) directly because max_slices={resolved_max_slices}; "
            "intermediate slices were filled from the nearest processed slice."
        )
        for z in range(z_count):
            if z in score_by_slice:
                continue
            nearest = min(indices, key=lambda k: abs(k - z))
            mask_stack[z] = mask_stack[nearest]
            score_by_slice[z] = score_by_slice.get(nearest, 0.0)

    mean_score = float(np.mean(list(score_by_slice.values()))) if score_by_slice else 0.0
    nonzero = int(np.count_nonzero(mask_stack))
    coverage = float(nonzero / mask_stack.size * 100.0) if mask_stack.size else 0.0

    # Return 2D mask for single-slice inputs.
    mask_out: np.ndarray
    if z_count == 1:
        mask_out = mask_stack[0]
    else:
        mask_out = mask_stack

    positive_labels = [int(value) for value in np.unique(mask_out) if int(value) > 0]
    instance_voxel_counts = [
        int(np.count_nonzero(mask_out == label_value)) for label_value in positive_labels
    ]
    instance_coverage_values = [
        round((float(count) / float(mask_out.size)) * 100.0, 6) if mask_out.size else 0.0
        for count in instance_voxel_counts
    ]
    instance_count = int(len(positive_labels))

    return {
        "success": True,
        "backend": runtime.backend,
        "model_id": runtime.display_name,
        "resolved_model_ref": runtime.resolved_ref,
        "requested_model_id": model_id,
        "device": resolved_device,
        "slice_count": z_count,
        "slice_stride": stride,
        "slices_processed": len(indices),
        "processed_all_slices": processed_all_slices,
        "requested_max_slices": resolved_max_slices,
        "mean_iou": mean_score,
        "mask_shape": list(mask_out.shape),
        "mask_dtype": str(mask_out.dtype),
        "segmented_voxels": nonzero,
        "coverage_percent": round(coverage, 4),
        "total_masks_generated": instance_count,
        "instance_count": instance_count,
        "instance_count_scope": "prompt_object_masks",
        "instance_voxel_counts": instance_voxel_counts,
        "instance_coverage_percent_values": instance_coverage_values,
        "instance_coverage_percent_mean": (
            round(float(np.mean(instance_coverage_values)), 6)
            if instance_coverage_values
            else None
        ),
        "instance_coverage_percent_min": (
            round(float(np.min(instance_coverage_values)), 6)
            if instance_coverage_values
            else None
        ),
        "instance_coverage_percent_max": (
            round(float(np.max(instance_coverage_values)), 6)
            if instance_coverage_values
            else None
        ),
        "warnings": warnings,
        "_mask": mask_out.astype(np.uint16),
        "_slice_scores": score_by_slice,
    }


__all__ = ["segment_array_with_medsam2"]
