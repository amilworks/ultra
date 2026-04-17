from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image

from .dataset import analyze_manifest_spatial_compatibility, normalize_spatial_dims

try:
    from sahi import AutoDetectionModel  # type: ignore
    from sahi.predict import get_sliced_prediction  # type: ignore

    _SAHI_AVAILABLE = True
except Exception:
    AutoDetectionModel = None  # type: ignore[assignment]
    get_sliced_prediction = None  # type: ignore[assignment]
    _SAHI_AVAILABLE = False


ProgressCallback = Callable[[dict[str, Any]], None]
ControlCallback = Callable[[], None]


def _safe_int(value: Any, default: int, minimum: int = 1, maximum: int = 10_000) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(minimum, min(maximum, parsed))


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value or "").strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _normalized_suffix(path: Path) -> str:
    lower = path.name.lower()
    if lower.endswith(".nii.gz"):
        return ".nii.gz"
    if lower.endswith(".ome.tiff"):
        return ".ome.tiff"
    if lower.endswith(".ome.tif"):
        return ".ome.tif"
    return path.suffix.lower()


def _load_image_array(path: Path) -> np.ndarray:
    suffix = _normalized_suffix(path)
    if suffix in {".npy", ".npz"}:
        loaded = np.load(path)  # type: ignore[call-overload]
        if isinstance(loaded, np.lib.npyio.NpzFile):
            first_key = next(iter(loaded.files), None)
            if first_key is None:
                raise ValueError(f"NPZ file has no arrays: {path}")
            array = np.asarray(loaded[first_key])
        else:
            array = np.asarray(loaded)
    elif suffix in {".nii", ".nii.gz"}:
        try:
            import nibabel as nib  # type: ignore
        except Exception as exc:
            raise ValueError(
                f"NIfTI input requires nibabel, but it is unavailable for {path}."
            ) from exc
        nii = nib.load(str(path))
        array = np.asarray(nii.get_fdata())
    elif suffix in {".tif", ".tiff", ".ome.tif", ".ome.tiff"}:
        try:
            import tifffile  # type: ignore

            array = np.asarray(tifffile.imread(str(path)))
        except Exception:
            with Image.open(path) as image:
                array = np.asarray(image.convert("L"))
    else:
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"))
    squeezed = np.squeeze(np.asarray(array))
    if squeezed.ndim < 2:
        raise ValueError(f"Unsupported array shape for inference: {squeezed.shape}")
    if squeezed.ndim > 3:
        leading_shape = squeezed.shape[:-2]
        leading_size = int(np.prod(leading_shape))
        squeezed = np.reshape(squeezed, (leading_size, *squeezed.shape[-2:]))
    if squeezed.ndim not in {2, 3}:
        raise ValueError(f"Unsupported array shape for inference: {squeezed.shape}")
    return squeezed.astype(np.float32)


def _load_mask_array(path: Path) -> np.ndarray:
    array = _load_image_array(path)
    finite = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    if np.issubdtype(finite.dtype, np.floating):
        finite = np.rint(finite)
    return finite.astype(np.int64)


def _save_mask_array(mask: np.ndarray, output_path_no_ext: Path) -> Path:
    output_path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    if mask.ndim == 2:
        output_path = output_path_no_ext.with_suffix(".png")
        Image.fromarray(mask.astype(np.uint8)).save(output_path, format="PNG")
        return output_path
    output_path = output_path_no_ext.with_suffix(".npy")
    np.save(output_path, mask.astype(np.uint8))
    return output_path


def _normalized_mean_intensity(array: np.ndarray) -> float:
    max_value = float(np.max(array)) if array.size > 0 else 0.0
    scale = max(1.0, max_value)
    return float(np.mean(array) / scale)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _torch_monai_available() -> bool:
    try:
        import monai  # type: ignore
        import torch  # type: ignore

        del monai, torch
        return True
    except Exception:
        return False


def _coerce_spatial_array(
    array: np.ndarray,
    *,
    required_spatial_dims: str,
    role: str,
    path: str,
) -> np.ndarray:
    squeezed = np.squeeze(np.asarray(array))
    expected = normalize_spatial_dims(required_spatial_dims, default="2d")

    if expected == "2d":
        if squeezed.ndim == 2:
            return squeezed
        if squeezed.ndim == 3:
            shape = squeezed.shape
            if shape[0] <= 4 and min(shape[1:]) > 16:
                return np.mean(squeezed, axis=0)
            if shape[-1] <= 4 and min(shape[:2]) > 16:
                return np.mean(squeezed, axis=-1)
        raise ValueError(
            f"Expected 2D {role} array but found shape {tuple(int(v) for v in squeezed.shape)} "
            f"for {path}."
        )

    if squeezed.ndim == 3:
        return squeezed
    if squeezed.ndim == 4:
        shape = squeezed.shape
        singleton_axes = [axis for axis, size in enumerate(shape) if int(size) == 1]
        if len(singleton_axes) == 1:
            return np.squeeze(squeezed, axis=singleton_axes[0])
        if shape[0] <= 4 and min(shape[1:]) > 4:
            return np.mean(squeezed, axis=0)
        if shape[-1] <= 4 and min(shape[:-1]) > 4:
            return np.mean(squeezed, axis=-1)
    raise ValueError(
        f"Expected 3D {role} array but found shape {tuple(int(v) for v in squeezed.shape)} "
        f"for {path}."
    )


def _normalize_image_for_model(array: np.ndarray) -> np.ndarray:
    normalized = np.nan_to_num(np.asarray(array, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(np.min(normalized)) if normalized.size else 0.0
    max_value = float(np.max(normalized)) if normalized.size else 0.0
    if max_value > min_value:
        normalized = (normalized - min_value) / (max_value - min_value)
    else:
        normalized = np.zeros_like(normalized, dtype=np.float32)
    return normalized.astype(np.float32)


def _normalize_mask_for_model(array: np.ndarray, *, num_classes: int) -> np.ndarray:
    mask = np.nan_to_num(np.asarray(array), nan=0.0, posinf=0.0, neginf=0.0)
    if np.issubdtype(mask.dtype, np.floating):
        mask = np.rint(mask)
    mask = mask.astype(np.int64)
    if np.min(mask) < 0:
        mask = np.clip(mask, 0, None)
    if num_classes > 1:
        mask = np.clip(mask, 0, num_classes - 1)
    return mask


def _resolve_target_shape(
    *,
    config: dict[str, Any],
    spatial_dims: str,
    sample_shape: tuple[int, ...],
) -> tuple[int, ...]:
    dims = 2 if spatial_dims == "2d" else 3
    raw_target = config.get("target_shape")
    parsed: list[int] = []

    if isinstance(raw_target, (list, tuple)):
        for item in raw_target:
            try:
                parsed.append(int(item))
            except Exception:
                continue
    elif isinstance(raw_target, str):
        tokens = [token.strip() for token in raw_target.split(",") if token.strip()]
        for token in tokens:
            try:
                parsed.append(int(token))
            except Exception:
                continue

    if len(parsed) >= dims:
        output = [max(8, int(parsed[index])) for index in range(dims)]
        return tuple(output)

    if spatial_dims == "2d":
        baseline = [128, 128]
        minimum = 16
    else:
        baseline = [32, 128, 128]
        minimum = 8
    if len(sample_shape) >= dims:
        adjusted: list[int] = []
        for index in range(dims):
            sample_size = int(sample_shape[index])
            adjusted.append(min(int(baseline[index]), max(minimum, sample_size)))
        return tuple(adjusted)
    return tuple(baseline)


def _resolve_dynunet_architecture(
    *,
    spatial_dims: int,
    target_shape: tuple[int, ...],
    checkpoint_architecture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(checkpoint_architecture, dict):
        kernel_size = checkpoint_architecture.get("kernel_size")
        strides = checkpoint_architecture.get("strides")
        upsample_kernel_size = checkpoint_architecture.get("upsample_kernel_size")
        filters = checkpoint_architecture.get("filters")
        if (
            isinstance(kernel_size, list)
            and isinstance(strides, list)
            and isinstance(upsample_kernel_size, list)
            and isinstance(filters, list)
        ):
            return {
                "kernel_size": kernel_size,
                "strides": strides,
                "upsample_kernel_size": upsample_kernel_size,
                "filters": filters,
            }

    stride_unit = [2] * spatial_dims
    strides: list[list[int]] = [[1] * spatial_dims]
    min_axis = int(min(target_shape)) if target_shape else 8
    while min_axis >= 16 and len(strides) < 4:
        strides.append(list(stride_unit))
        min_axis = max(1, min_axis // 2)
    if len(strides) < 2:
        strides.append(list(stride_unit))
    kernel_size = [[3] * spatial_dims for _ in strides]
    upsample_kernel_size = [list(step) for step in strides[1:]]
    filters = [min(32 * (2**index), 320) for index in range(len(strides))]
    return {
        "kernel_size": kernel_size,
        "strides": strides,
        "upsample_kernel_size": upsample_kernel_size,
        "filters": filters,
    }


def _mean_multiclass_dice(
    pred_labels: Any,
    target_labels: Any,
    *,
    num_classes: int,
) -> float:
    try:
        import torch  # type: ignore
    except Exception:
        return 0.0

    eps = 1e-6
    scores: list[float] = []
    class_range = range(1, max(2, int(num_classes)))
    for class_index in class_range:
        pred_mask = (pred_labels == class_index).float()
        target_mask = (target_labels == class_index).float()
        denom = float(torch.sum(pred_mask).item() + torch.sum(target_mask).item())
        if denom <= 0.0:
            continue
        intersection = float(torch.sum(pred_mask * target_mask).item())
        score = (2.0 * intersection + eps) / (denom + eps)
        scores.append(score)

    if not scores:
        pred_foreground = (pred_labels > 0).float()
        target_foreground = (target_labels > 0).float()
        denom = float(torch.sum(pred_foreground).item() + torch.sum(target_foreground).item())
        if denom <= 0.0:
            return 1.0
        intersection = float(torch.sum(pred_foreground * target_foreground).item())
        return (2.0 * intersection + eps) / (denom + eps)
    return float(sum(scores) / float(len(scores)))


def _collect_split_pairs(
    *,
    manifest: dict[str, Any],
    split: str,
) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    splits = manifest.get("splits") if isinstance(manifest.get("splits"), dict) else {}
    rows = splits.get(split) if isinstance(splits, dict) else []
    if not isinstance(rows, list):
        return pairs
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        image = row.get("image") if isinstance(row.get("image"), dict) else {}
        mask = row.get("mask") if isinstance(row.get("mask"), dict) else {}
        image_path = str(image.get("path") or "").strip()
        mask_path = str(mask.get("path") or "").strip()
        if not image_path or not mask_path:
            continue
        pairs.append(
            {
                "sample_id": str(row.get("sample_id") or f"{split}-{index}"),
                "image_path": image_path,
                "mask_path": mask_path,
            }
        )
    return pairs


def _load_resume_checkpoint(
    *,
    initial_checkpoint_path: str | None,
    epochs: int,
    model: Any,
    optimizer: Any,
    device: Any,
) -> tuple[int, str | None]:
    if not initial_checkpoint_path:
        return 1, None
    checkpoint_path = Path(initial_checkpoint_path).expanduser().resolve()
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        return 1, None

    if checkpoint_path.suffix.lower() == ".json":
        try:
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            nested = str(payload.get("checkpoint_path") or "").strip()
            if nested:
                nested_path = Path(nested).expanduser().resolve()
                if nested_path.exists() and nested_path.is_file():
                    checkpoint_path = nested_path
            epoch = _safe_int(payload.get("epoch"), default=1, minimum=1, maximum=epochs) + 1
            return min(max(1, epoch), max(1, epochs)), str(checkpoint_path)
        except Exception:
            return 1, None

    if checkpoint_path.suffix.lower() != ".pt":
        return 1, None

    try:
        import torch  # type: ignore

        state = torch.load(str(checkpoint_path), map_location=device)
        if isinstance(state, dict):
            model_state = state.get("model_state")
            optimizer_state = state.get("optimizer_state")
            if isinstance(model_state, dict):
                model.load_state_dict(model_state)
            if isinstance(optimizer_state, dict):
                optimizer.load_state_dict(optimizer_state)
            epoch = _safe_int(state.get("epoch"), default=1, minimum=1, maximum=epochs) + 1
            return min(max(1, epoch), max(1, epochs)), str(checkpoint_path)
    except Exception:
        return 1, None
    return 1, None


class BaseTrainingAdapter:
    key = "base"
    supports_training = False
    supports_finetune = False
    supports_inference = False

    def train(
        self,
        *,
        manifest: dict[str, Any],
        config: dict[str, Any],
        output_dir: Path,
        progress_callback: ProgressCallback,
        control_callback: ControlCallback,
        initial_checkpoint_path: str | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def infer(
        self,
        *,
        model_artifact_path: str | None,
        input_paths: list[str],
        output_dir: Path,
        config: dict[str, Any],
        progress_callback: ProgressCallback,
        control_callback: ControlCallback,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def benchmark(
        self,
        *,
        model_artifact_path: str | None,
        output_dir: Path,
        config: dict[str, Any],
        progress_callback: ProgressCallback,
        control_callback: ControlCallback,
    ) -> dict[str, Any]:
        raise ValueError(f"Model '{self.key}' does not support benchmark execution.")


class DynUNETAdapter(BaseTrainingAdapter):
    key = "dynunet"
    supports_training = True
    supports_finetune = True
    supports_inference = True

    def _resolve_backend(self, config: dict[str, Any]) -> str:
        requested = str(config.get("execution_backend") or "auto").strip().lower() or "auto"
        if requested not in {"auto", "simulated", "monai"}:
            requested = "auto"
        monai_available = _torch_monai_available()
        if requested == "simulated":
            return "simulated"
        if requested == "monai":
            if not monai_available:
                raise ValueError("execution_backend=monai requested, but MONAI is not installed.")
            return "monai"
        return "monai" if monai_available else "simulated"

    def train(
        self,
        *,
        manifest: dict[str, Any],
        config: dict[str, Any],
        output_dir: Path,
        progress_callback: ProgressCallback,
        control_callback: ControlCallback,
        initial_checkpoint_path: str | None = None,
    ) -> dict[str, Any]:
        requested_spatial_dims = normalize_spatial_dims(config.get("spatial_dims"), default="2d")
        spatial_profile = analyze_manifest_spatial_compatibility(
            manifest=manifest,
            required_spatial_dims=requested_spatial_dims,
            max_samples=1024,
        )
        violations = spatial_profile.get("violations")
        pair_mismatches = spatial_profile.get("pair_mismatches")
        if isinstance(violations, list) and violations:
            first = violations[0]
            raise ValueError(
                "Dataset dimensionality mismatch: "
                f"sample '{first.get('sample_id')}' ({first.get('role')}) was "
                f"{first.get('detected_spatial_dims')} but training requested "
                f"{first.get('required_spatial_dims')}."
            )
        if isinstance(pair_mismatches, list) and pair_mismatches:
            first = pair_mismatches[0]
            raise ValueError(
                "Dataset image/mask dimensionality mismatch: "
                f"sample '{first.get('sample_id')}' has image={first.get('image_dims')} "
                f"and mask={first.get('mask_dims')}."
            )

        backend = self._resolve_backend(config)
        if backend == "monai":
            result = self._train_monai(
                manifest=manifest,
                config=config,
                output_dir=output_dir,
                progress_callback=progress_callback,
                control_callback=control_callback,
                initial_checkpoint_path=initial_checkpoint_path,
                requested_spatial_dims=requested_spatial_dims,
            )
        else:
            result = self._train_simulated(
                manifest=manifest,
                config=config,
                output_dir=output_dir,
                progress_callback=progress_callback,
                control_callback=control_callback,
                initial_checkpoint_path=initial_checkpoint_path,
                requested_spatial_dims=requested_spatial_dims,
            )
        result["spatial_profile"] = spatial_profile
        return result

    def _train_simulated(
        self,
        *,
        manifest: dict[str, Any],
        config: dict[str, Any],
        output_dir: Path,
        progress_callback: ProgressCallback,
        control_callback: ControlCallback,
        initial_checkpoint_path: str | None,
        requested_spatial_dims: str,
    ) -> dict[str, Any]:
        del manifest
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        epochs = _safe_int(config.get("epochs"), default=10, minimum=1, maximum=200)
        seed = _safe_int(config.get("seed"), default=42, minimum=1, maximum=2_147_483_647)
        rng = random.Random(seed)
        start_epoch = 1
        if initial_checkpoint_path:
            try:
                raw = json.loads(Path(initial_checkpoint_path).read_text(encoding="utf-8"))
                start_epoch = _safe_int(raw.get("epoch"), default=1, minimum=1, maximum=epochs) + 1
            except Exception:
                start_epoch = 1

        progress_rows: list[dict[str, Any]] = []
        best_val_dice = 0.0
        final_loss = 1.0
        for epoch in range(start_epoch, epochs + 1):
            control_callback()
            final_loss = max(0.015, (0.92**epoch) + rng.uniform(-0.01, 0.01))
            val_dice = min(0.985, 0.42 + (0.055 * epoch) + rng.uniform(-0.015, 0.015))
            best_val_dice = max(best_val_dice, val_dice)
            row = {
                "epoch": epoch,
                "epochs": epochs,
                "train_loss": round(final_loss, 6),
                "val_dice": round(val_dice, 6),
            }
            progress_rows.append(row)
            checkpoint_path = checkpoints_dir / f"epoch-{epoch:03d}.json"
            _write_json(
                checkpoint_path,
                {
                    "model_key": self.key,
                    "epoch": epoch,
                    "epochs": epochs,
                    "metrics": row,
                    "config": config,
                },
            )
            progress_callback(
                {
                    "event": "epoch_complete",
                    "epoch": epoch,
                    "epochs": epochs,
                    "train_loss": row["train_loss"],
                    "val_dice": row["val_dice"],
                    "checkpoint_path": str(checkpoint_path),
                }
            )
            time.sleep(0.05)

        model_artifact = output_dir / "model-final.json"
        _write_json(
            model_artifact,
            {
                "model_key": self.key,
                "framework": "dynunet-fallback",
                "epochs": epochs,
                "spatial_dims": requested_spatial_dims,
                "metrics": {
                    "best_val_dice": round(best_val_dice, 6),
                    "final_loss": round(final_loss, 6),
                },
                "config": config,
            },
        )
        metrics_path = output_dir / "metrics.json"
        _write_json(
            metrics_path,
            {
                "epochs": epochs,
                "spatial_dims": requested_spatial_dims,
                "best_val_dice": round(best_val_dice, 6),
                "final_loss": round(final_loss, 6),
                "history": progress_rows,
            },
        )
        return {
            "framework": "dynunet-fallback",
            "execution_backend": "simulated",
            "epochs_completed": epochs,
            "spatial_dims": requested_spatial_dims,
            "model_artifact_path": str(model_artifact),
            "metrics_path": str(metrics_path),
            "checkpoint_paths": [
                str(path) for path in sorted(checkpoints_dir.glob("epoch-*.json"))
            ],
            "metrics": {
                "best_val_dice": round(best_val_dice, 6),
                "final_loss": round(final_loss, 6),
            },
        }

    def _train_monai(
        self,
        *,
        manifest: dict[str, Any],
        config: dict[str, Any],
        output_dir: Path,
        progress_callback: ProgressCallback,
        control_callback: ControlCallback,
        initial_checkpoint_path: str | None,
        requested_spatial_dims: str,
    ) -> dict[str, Any]:
        try:
            import monai  # type: ignore
            import torch  # type: ignore
            import torch.nn.functional as torch_f  # type: ignore
            from torch.utils.data import DataLoader, Dataset  # type: ignore
        except Exception as exc:
            raise ValueError("MONAI backend requested but dependencies are unavailable.") from exc

        epochs = _safe_int(config.get("epochs"), default=10, minimum=1, maximum=500)
        batch_size = _safe_int(config.get("batch_size"), default=1, minimum=1, maximum=16)
        learning_rate = _safe_float(config.get("learning_rate"), default=1e-3)
        seed = _safe_int(config.get("seed"), default=42, minimum=1, maximum=2_147_483_647)
        num_classes = _safe_int(config.get("num_classes"), default=2, minimum=2, maximum=256)
        spatial_ndim = 2 if requested_spatial_dims == "2d" else 3
        device_token = str(config.get("device") or "").strip().lower()
        if device_token in {"cpu", "cuda"}:
            if device_token == "cuda" and not torch.cuda.is_available():
                device = torch.device("cpu")
            else:
                device = torch.device(device_token)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_pairs = _collect_split_pairs(manifest=manifest, split="train")
        val_pairs = _collect_split_pairs(manifest=manifest, split="val")
        if not train_pairs:
            raise ValueError("Training manifest has no train image/mask pairs.")
        if not val_pairs:
            raise ValueError("Training manifest has no validation image/mask pairs.")

        probe_image = _coerce_spatial_array(
            _load_image_array(Path(train_pairs[0]["image_path"])),
            required_spatial_dims=requested_spatial_dims,
            role="image",
            path=train_pairs[0]["image_path"],
        )
        target_shape = _resolve_target_shape(
            config=config,
            spatial_dims=requested_spatial_dims,
            sample_shape=tuple(int(item) for item in probe_image.shape[:spatial_ndim]),
        )
        architecture = _resolve_dynunet_architecture(
            spatial_dims=spatial_ndim,
            target_shape=target_shape,
        )

        class SegPairDataset(Dataset):  # type: ignore[misc,valid-type]
            def __init__(self, pairs: list[dict[str, str]]) -> None:
                self._pairs = pairs

            def __len__(self) -> int:
                return len(self._pairs)

            def __getitem__(self, index: int) -> tuple[Any, Any]:
                row = self._pairs[index]
                image_array = _coerce_spatial_array(
                    _load_image_array(Path(row["image_path"])),
                    required_spatial_dims=requested_spatial_dims,
                    role="image",
                    path=row["image_path"],
                )
                mask_array = _coerce_spatial_array(
                    _load_mask_array(Path(row["mask_path"])),
                    required_spatial_dims=requested_spatial_dims,
                    role="mask",
                    path=row["mask_path"],
                )
                if image_array.shape != mask_array.shape:
                    raise ValueError(
                        f"Image/mask shape mismatch for sample '{row.get('sample_id')}': "
                        f"{tuple(int(v) for v in image_array.shape)} vs "
                        f"{tuple(int(v) for v in mask_array.shape)}."
                    )
                image = _normalize_image_for_model(image_array)
                mask = _normalize_mask_for_model(mask_array, num_classes=num_classes)
                image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
                mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
                image_mode = "bilinear" if spatial_ndim == 2 else "trilinear"
                image_tensor = torch_f.interpolate(
                    image_tensor,
                    size=target_shape,
                    mode=image_mode,
                    align_corners=False,
                )
                mask_tensor = torch_f.interpolate(
                    mask_tensor,
                    size=target_shape,
                    mode="nearest",
                )
                return image_tensor.squeeze(0), mask_tensor.squeeze(0).long()

        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        _seed_everything(seed)
        model = monai.networks.nets.DynUNet(
            spatial_dims=spatial_ndim,
            in_channels=1,
            out_channels=num_classes,
            kernel_size=architecture["kernel_size"],
            strides=architecture["strides"],
            upsample_kernel_size=architecture["upsample_kernel_size"],
            filters=architecture["filters"],
            deep_supervision=False,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = monai.losses.DiceCELoss(to_onehot_y=True, softmax=True)

        start_epoch, resume_checkpoint_path = _load_resume_checkpoint(
            initial_checkpoint_path=initial_checkpoint_path,
            epochs=epochs,
            model=model,
            optimizer=optimizer,
            device=device,
        )

        train_loader = DataLoader(
            SegPairDataset(train_pairs),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            SegPairDataset(val_pairs),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

        progress_rows: list[dict[str, Any]] = []
        best_val_dice = 0.0
        final_loss = 0.0
        epochs_completed = 0
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            total_loss = 0.0
            batch_count = 0
            for image_batch, mask_batch in train_loader:
                control_callback()
                image_batch = image_batch.to(device)
                mask_batch = mask_batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(image_batch)
                loss = loss_fn(logits, mask_batch)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().item())
                batch_count += 1

            train_loss = total_loss / float(max(1, batch_count))
            final_loss = train_loss

            model.eval()
            val_scores: list[float] = []
            with torch.no_grad():
                for image_batch, mask_batch in val_loader:
                    control_callback()
                    image_batch = image_batch.to(device)
                    mask_batch = mask_batch.to(device)
                    logits = model(image_batch)
                    prediction = torch.argmax(logits, dim=1)
                    target = mask_batch[:, 0, ...].long()
                    val_scores.append(
                        _mean_multiclass_dice(
                            prediction,
                            target,
                            num_classes=num_classes,
                        )
                    )
            val_dice = float(sum(val_scores) / float(len(val_scores))) if val_scores else 0.0
            best_val_dice = max(best_val_dice, val_dice)
            epochs_completed += 1

            row = {
                "epoch": epoch,
                "epochs": epochs,
                "train_loss": round(train_loss, 6),
                "val_dice": round(val_dice, 6),
            }
            progress_rows.append(row)

            checkpoint_path = checkpoints_dir / f"epoch-{epoch:03d}.pt"
            checkpoint_payload = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "epochs": epochs,
                "config": config,
                "metadata": {
                    "model_key": self.key,
                    "framework": "dynunet-monai",
                    "spatial_dims": requested_spatial_dims,
                    "target_shape": list(target_shape),
                    "num_classes": num_classes,
                    "architecture": architecture,
                },
            }
            torch.save(checkpoint_payload, str(checkpoint_path))
            checkpoint_meta_path = checkpoints_dir / f"epoch-{epoch:03d}.json"
            _write_json(
                checkpoint_meta_path,
                {
                    "model_key": self.key,
                    "framework": "dynunet-monai",
                    "epoch": epoch,
                    "epochs": epochs,
                    "checkpoint_path": str(checkpoint_path),
                    "metrics": row,
                },
            )
            progress_callback(
                {
                    "event": "epoch_complete",
                    "epoch": epoch,
                    "epochs": epochs,
                    "train_loss": row["train_loss"],
                    "val_dice": row["val_dice"],
                    "checkpoint_path": str(checkpoint_path),
                }
            )

        model_artifact = output_dir / "model-final.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": config,
                "metadata": {
                    "model_key": self.key,
                    "framework": "dynunet-monai",
                    "spatial_dims": requested_spatial_dims,
                    "target_shape": list(target_shape),
                    "num_classes": num_classes,
                    "architecture": architecture,
                    "best_val_dice": round(best_val_dice, 6),
                    "epochs_completed": epochs_completed,
                    "resume_checkpoint_path": resume_checkpoint_path,
                },
            },
            str(model_artifact),
        )
        model_summary_path = output_dir / "model-final.json"
        _write_json(
            model_summary_path,
            {
                "model_key": self.key,
                "framework": "dynunet-monai",
                "execution_backend": "monai",
                "epochs": epochs,
                "epochs_completed": epochs_completed,
                "spatial_dims": requested_spatial_dims,
                "target_shape": list(target_shape),
                "num_classes": num_classes,
                "device": str(device),
                "model_artifact_path": str(model_artifact),
                "resume_checkpoint_path": resume_checkpoint_path,
                "metrics": {
                    "best_val_dice": round(best_val_dice, 6),
                    "final_loss": round(final_loss, 6),
                },
            },
        )

        metrics_path = output_dir / "metrics.json"
        _write_json(
            metrics_path,
            {
                "epochs": epochs,
                "epochs_completed": epochs_completed,
                "spatial_dims": requested_spatial_dims,
                "target_shape": list(target_shape),
                "num_classes": num_classes,
                "best_val_dice": round(best_val_dice, 6),
                "final_loss": round(final_loss, 6),
                "history": progress_rows,
            },
        )

        return {
            "framework": "dynunet-monai",
            "execution_backend": "monai",
            "epochs_completed": epochs_completed,
            "spatial_dims": requested_spatial_dims,
            "target_shape": list(target_shape),
            "num_classes": num_classes,
            "device": str(device),
            "model_artifact_path": str(model_artifact),
            "model_summary_path": str(model_summary_path),
            "metrics_path": str(metrics_path),
            "checkpoint_paths": [str(path) for path in sorted(checkpoints_dir.glob("epoch-*.pt"))],
            "metrics": {
                "best_val_dice": round(best_val_dice, 6),
                "final_loss": round(final_loss, 6),
            },
        }

    def infer(
        self,
        *,
        model_artifact_path: str | None,
        input_paths: list[str],
        output_dir: Path,
        config: dict[str, Any],
        progress_callback: ProgressCallback,
        control_callback: ControlCallback,
    ) -> dict[str, Any]:
        backend = self._resolve_backend(config)
        if backend == "monai":
            if not model_artifact_path:
                raise ValueError(
                    "DynUNET MONAI inference requires model_artifact_path from a succeeded training job."
                )
            return self._infer_monai(
                model_artifact_path=model_artifact_path,
                input_paths=input_paths,
                output_dir=output_dir,
                config=config,
                progress_callback=progress_callback,
                control_callback=control_callback,
            )
        return self._infer_simulated(
            model_artifact_path=model_artifact_path,
            input_paths=input_paths,
            output_dir=output_dir,
            config=config,
            progress_callback=progress_callback,
            control_callback=control_callback,
        )

    def _infer_simulated(
        self,
        *,
        model_artifact_path: str | None,
        input_paths: list[str],
        output_dir: Path,
        config: dict[str, Any],
        progress_callback: ProgressCallback,
        control_callback: ControlCallback,
    ) -> dict[str, Any]:
        del model_artifact_path
        threshold_bias = _safe_float(config.get("threshold_bias"), 0.0)
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions: list[dict[str, Any]] = []
        drift_scores: list[float] = []

        for index, raw_path in enumerate(input_paths):
            control_callback()
            source = Path(raw_path)
            array = _load_image_array(source)
            threshold = float(np.mean(array)) + threshold_bias
            mask = (array >= threshold).astype(np.uint8) * 255
            output_path = _save_mask_array(mask, output_dir / f"{source.stem}__dynunet_mask")
            drift_score = abs(_normalized_mean_intensity(array) - 0.5)
            drift_scores.append(drift_score)
            row = {
                "input_path": str(source),
                "mask_path": str(output_path),
                "drift_score": round(drift_score, 6),
                "spatial_dims": f"{int(mask.ndim)}d",
            }
            predictions.append(row)
            progress_callback(
                {
                    "event": "inference_item_complete",
                    "index": index + 1,
                    "total": len(input_paths),
                    "input_path": str(source),
                    "mask_path": str(output_path),
                    "drift_score": row["drift_score"],
                }
            )

        return {
            "predictions": predictions,
            "prediction_count": len(predictions),
            "framework": "dynunet-fallback",
            "execution_backend": "simulated",
            "drift_score_mean": round(
                (sum(drift_scores) / float(len(drift_scores))) if drift_scores else 0.0,
                6,
            ),
        }

    def _infer_monai(
        self,
        *,
        model_artifact_path: str,
        input_paths: list[str],
        output_dir: Path,
        config: dict[str, Any],
        progress_callback: ProgressCallback,
        control_callback: ControlCallback,
    ) -> dict[str, Any]:
        try:
            import monai  # type: ignore
            import torch  # type: ignore
            import torch.nn.functional as torch_f  # type: ignore
        except Exception as exc:
            raise ValueError("MONAI backend requested but dependencies are unavailable.") from exc

        artifact_path = Path(model_artifact_path).expanduser().resolve()
        if not artifact_path.exists() or not artifact_path.is_file():
            raise ValueError(f"Model artifact not found: {artifact_path}")
        if artifact_path.suffix.lower() != ".pt":
            raise ValueError(
                f"MONAI inference expects a .pt checkpoint artifact, got: {artifact_path.name}"
            )

        state = torch.load(str(artifact_path), map_location="cpu")
        if not isinstance(state, dict):
            raise ValueError("Invalid model artifact payload.")
        metadata = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
        spatial_dims = normalize_spatial_dims(
            metadata.get("spatial_dims") or config.get("spatial_dims"),
            default="2d",
        )
        spatial_ndim = 2 if spatial_dims == "2d" else 3
        target_shape_values = metadata.get("target_shape")
        target_shape: tuple[int, ...]
        if (
            isinstance(target_shape_values, list)
            and len(target_shape_values) >= spatial_ndim
            and all(str(item).strip() for item in target_shape_values[:spatial_ndim])
        ):
            target_shape = tuple(
                max(8, int(target_shape_values[index])) for index in range(spatial_ndim)
            )
        else:
            target_shape = tuple([128, 128] if spatial_dims == "2d" else [32, 128, 128])
        num_classes = _safe_int(metadata.get("num_classes"), default=2, minimum=2, maximum=256)
        architecture = _resolve_dynunet_architecture(
            spatial_dims=spatial_ndim,
            target_shape=target_shape,
            checkpoint_architecture=metadata.get("architecture")
            if isinstance(metadata.get("architecture"), dict)
            else None,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = monai.networks.nets.DynUNet(
            spatial_dims=spatial_ndim,
            in_channels=1,
            out_channels=num_classes,
            kernel_size=architecture["kernel_size"],
            strides=architecture["strides"],
            upsample_kernel_size=architecture["upsample_kernel_size"],
            filters=architecture["filters"],
            deep_supervision=False,
        ).to(device)
        model_state = state.get("model_state")
        if not isinstance(model_state, dict):
            raise ValueError("Model artifact missing model_state.")
        model.load_state_dict(model_state, strict=True)
        model.eval()

        output_dir.mkdir(parents=True, exist_ok=True)
        predictions: list[dict[str, Any]] = []
        drift_scores: list[float] = []

        for index, raw_path in enumerate(input_paths):
            control_callback()
            source = Path(raw_path)
            raw_image = _coerce_spatial_array(
                _load_image_array(source),
                required_spatial_dims=spatial_dims,
                role="image",
                path=str(source),
            )
            image = _normalize_image_for_model(raw_image)
            original_shape = tuple(int(item) for item in image.shape[:spatial_ndim])
            image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)
            image_mode = "bilinear" if spatial_ndim == 2 else "trilinear"
            resized_input = torch_f.interpolate(
                image_tensor,
                size=target_shape,
                mode=image_mode,
                align_corners=False,
            )
            with torch.no_grad():
                logits = model(resized_input)
                prediction = torch.argmax(logits, dim=1, keepdim=True).float()
            restored = torch_f.interpolate(
                prediction,
                size=original_shape,
                mode="nearest",
            )
            mask = restored.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
            output_path = _save_mask_array(mask, output_dir / f"{source.stem}__dynunet_mask")
            drift_score = abs(_normalized_mean_intensity(image) - 0.5)
            drift_scores.append(drift_score)
            row = {
                "input_path": str(source),
                "mask_path": str(output_path),
                "drift_score": round(drift_score, 6),
                "spatial_dims": f"{int(mask.ndim)}d",
            }
            predictions.append(row)
            progress_callback(
                {
                    "event": "inference_item_complete",
                    "index": index + 1,
                    "total": len(input_paths),
                    "input_path": str(source),
                    "mask_path": str(output_path),
                    "drift_score": row["drift_score"],
                }
            )

        return {
            "predictions": predictions,
            "prediction_count": len(predictions),
            "framework": "dynunet-monai",
            "execution_backend": "monai",
            "spatial_dims": spatial_dims,
            "model_artifact_path": str(artifact_path),
            "drift_score_mean": round(
                (sum(drift_scores) / float(len(drift_scores))) if drift_scores else 0.0,
                6,
            ),
        }


class MedSAMAdapter(BaseTrainingAdapter):
    key = "medsam"
    supports_training = False
    supports_finetune = True
    supports_inference = True

    def train(
        self,
        *,
        manifest: dict[str, Any],
        config: dict[str, Any],
        output_dir: Path,
        progress_callback: ProgressCallback,
        control_callback: ControlCallback,
        initial_checkpoint_path: str | None = None,
    ) -> dict[str, Any]:
        # v1 MedSAM train path remains finetune-oriented with lightweight fallback execution.
        dynu = DynUNETAdapter()
        result = dynu.train(
            manifest=manifest,
            config={
                **config,
                "epochs": _safe_int(config.get("epochs"), 10, 1, 120),
                "execution_backend": "simulated",
            },
            output_dir=output_dir,
            progress_callback=progress_callback,
            control_callback=control_callback,
            initial_checkpoint_path=initial_checkpoint_path,
        )
        result["framework"] = "medsam-fallback-finetune"
        result["execution_backend"] = "simulated"
        return result

    def infer(
        self,
        *,
        model_artifact_path: str | None,
        input_paths: list[str],
        output_dir: Path,
        config: dict[str, Any],
        progress_callback: ProgressCallback,
        control_callback: ControlCallback,
    ) -> dict[str, Any]:
        del model_artifact_path
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions: list[dict[str, Any]] = []
        drift_scores: list[float] = []
        edge_weight = _safe_float(config.get("edge_weight"), 0.25)
        medsam_model_id = str(config.get("model_id") or "").strip() or None
        medsam_device = str(config.get("device") or "").strip() or None
        medsam_max_slices = _safe_int(
            config.get("max_slices"), default=160, minimum=1, maximum=2048
        )
        medsam_multimask = _safe_bool(config.get("multimask_output"), True)

        for index, raw_path in enumerate(input_paths):
            control_callback()
            source = Path(raw_path)
            array = _load_image_array(source)
            score_map = array.astype(np.float32)
            mask: np.ndarray | None = None
            medsam_backend = "medsam-fallback"
            medsam_warning: str | None = None

            try:
                from src.science.medsam2 import segment_array_with_medsam2

                medsam_result = segment_array_with_medsam2(
                    array,
                    order="YX" if array.ndim == 2 else "ZYX",
                    multimask_output=medsam_multimask,
                    model_id=medsam_model_id,
                    device=medsam_device,
                    max_slices=medsam_max_slices,
                )
                if bool(medsam_result.get("success")) and isinstance(
                    medsam_result.get("_mask"), np.ndarray
                ):
                    medsam_backend = str(medsam_result.get("backend") or "medsam2")
                    mask = (np.asarray(medsam_result["_mask"]).astype(np.uint8) > 0).astype(
                        np.uint8
                    ) * 255
                    score_map = array.astype(np.float32)
                else:
                    medsam_warning = str(medsam_result.get("error") or "MedSAM2 inference failed.")
            except Exception as exc:
                medsam_warning = str(exc)

            if mask is None:
                gradients = np.gradient(array.astype(np.float32))
                grad_magnitude = np.zeros_like(array, dtype=np.float32)
                for component in gradients:
                    grad_magnitude += np.abs(component.astype(np.float32))
                score_map = (1.0 - edge_weight) * array + edge_weight * grad_magnitude
                threshold = float(np.percentile(score_map, 60))
                mask = (score_map >= threshold).astype(np.uint8) * 255
                medsam_backend = "medsam-fallback"
            output_path = _save_mask_array(mask, output_dir / f"{source.stem}__medsam_mask")
            drift_score = abs(_normalized_mean_intensity(score_map) - 0.45)
            drift_scores.append(drift_score)
            row = {
                "input_path": str(source),
                "mask_path": str(output_path),
                "drift_score": round(drift_score, 6),
                "spatial_dims": f"{int(mask.ndim)}d",
                "backend": medsam_backend,
            }
            if medsam_warning:
                row["warning"] = medsam_warning
            predictions.append(row)
            progress_callback(
                {
                    "event": "inference_item_complete",
                    "index": index + 1,
                    "total": len(input_paths),
                    "input_path": str(source),
                    "mask_path": str(output_path),
                    "drift_score": row["drift_score"],
                    "backend": medsam_backend,
                }
            )

        framework = (
            "medsam2"
            if any(str(item.get("backend", "")).startswith("medsam2") for item in predictions)
            else "medsam-fallback"
        )
        return {
            "predictions": predictions,
            "prediction_count": len(predictions),
            "framework": framework,
            "execution_backend": "medsam2" if framework == "medsam2" else "simulated",
            "drift_score_mean": round(
                (sum(drift_scores) / float(len(drift_scores))) if drift_scores else 0.0,
                6,
            ),
        }


_YOLO_SUPPORTED_CLASSES: list[str] = ["prairie_dog", "burrow"]
_YOLO_CLASS_TO_ID = {name: index for index, name in enumerate(_YOLO_SUPPORTED_CLASSES)}
_YOLO_IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _slug_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip())
    token = re.sub(r"-+", "-", token).strip("-")
    return token or "sample"


def _package_root_path() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_package_relative_path(value: str | None, default: str) -> Path:
    raw = str(value or "").strip() or default
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = _package_root_path() / candidate
    return candidate.resolve()


def _resolve_yolov5_repo_path() -> Path:
    return _resolve_package_relative_path(os.getenv("YOLOV5_RUNTIME_PATH"), "third_party/yolov5")


def _resolve_rarespot_weights_path() -> Path:
    return _resolve_package_relative_path(
        os.getenv("YOLOV5_RARESPOT_WEIGHTS"), "RareSpotWeights.pt"
    )


def _ensure_yolov5_font_asset(config_dir: Path) -> None:
    target = config_dir / "Arial.ttf"
    if target.exists():
        return
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    ]
    try:
        from matplotlib import font_manager  # type: ignore

        resolved = Path(str(font_manager.findfont("DejaVu Sans"))).expanduser()
        candidates.insert(0, resolved)
    except Exception:
        pass
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, target)
                return
        except Exception:
            continue
    raise ValueError(
        "Unable to locate a local TTF font for YOLOv5 runtime. "
        "Install a system font (for example DejaVu Sans) or provide Arial.ttf."
    )


def _yolo_train_pairs(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    splits = manifest.get("splits") if isinstance(manifest.get("splits"), dict) else {}
    rows = splits.get(split) if isinstance(splits, dict) else []
    output: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return output
    for row in rows:
        if not isinstance(row, dict):
            continue
        image = row.get("image") if isinstance(row.get("image"), dict) else {}
        annotation = row.get("annotation") if isinstance(row.get("annotation"), dict) else {}
        image_path = str(image.get("path") or "").strip()
        annotation_path = str(annotation.get("path") or "").strip()
        if not image_path or not annotation_path:
            continue
        output.append(
            {
                "sample_id": str(row.get("sample_id") or Path(image_path).stem),
                "image_path": image_path,
                "annotation_path": annotation_path,
            }
        )
    return output


def _parse_bisque_rectangles(
    xml_path: Path, *, layer_name: str = "gt2"
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    top_layers = list(root.findall("./gobject"))
    selected_layers: list[ET.Element] = []
    for layer in top_layers:
        if str(layer.attrib.get("name") or "").strip() == layer_name:
            selected_layers.append(layer)
    if not selected_layers:
        raise ValueError(f"Missing required annotation layer '{layer_name}' in {xml_path}.")
    boxes: list[dict[str, Any]] = []
    unsupported: dict[str, int] = {}
    for layer in selected_layers:
        for class_node in layer.findall("./gobject"):
            class_name = str(class_node.attrib.get("name") or "").strip()
            if not class_name:
                continue
            for rect in class_node.findall("./rectangle"):
                verts = rect.findall("./vertex")
                if len(verts) < 2:
                    continue
                try:
                    x1 = float(verts[0].attrib.get("x"))
                    y1 = float(verts[0].attrib.get("y"))
                    x2 = float(verts[1].attrib.get("x"))
                    y2 = float(verts[1].attrib.get("y"))
                except Exception:
                    continue
                xyxy = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                if class_name not in _YOLO_CLASS_TO_ID:
                    unsupported[class_name] = unsupported.get(class_name, 0) + 1
                    continue
                boxes.append(
                    {
                        "class_name": class_name,
                        "class_id": _YOLO_CLASS_TO_ID[class_name],
                        "xyxy": xyxy,
                    }
                )
    return boxes, unsupported


def _xyxy_to_yolo_line(box: dict[str, Any], width: int, height: int) -> str:
    x1, y1, x2, y2 = [float(v) for v in box["xyxy"]]
    cx = ((x1 + x2) / 2.0) / float(max(1, width))
    cy = ((y1 + y2) / 2.0) / float(max(1, height))
    bw = (x2 - x1) / float(max(1, width))
    bh = (y2 - y1) / float(max(1, height))
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    bw = max(0.0, min(1.0, bw))
    bh = max(0.0, min(1.0, bh))
    return f"{int(box['class_id'])} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def _clamp_overlap(value: Any, *, default: float) -> float:
    parsed = _safe_float(value, default=default)
    if parsed < 0.0:
        return 0.0
    if parsed > 0.95:
        return 0.95
    return parsed


def _tile_start_positions(length: int, tile_size: int, stride: int) -> list[int]:
    size = max(1, int(length))
    tile = max(1, int(tile_size))
    step = max(1, int(stride))
    if size <= tile:
        return [0]
    starts = list(range(0, max(1, size - tile + 1), step))
    last = max(0, size - tile)
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def _build_sliding_tiles(
    *,
    width: int,
    height: int,
    tile_size: int,
    overlap: float,
) -> list[dict[str, int]]:
    tile = max(8, int(tile_size))
    overlap_clamped = _clamp_overlap(overlap, default=0.0)
    stride = max(1, round(tile * (1.0 - overlap_clamped)))
    x_starts = _tile_start_positions(width, tile, stride)
    y_starts = _tile_start_positions(height, tile, stride)
    tiles: list[dict[str, int]] = []
    for yi, y0 in enumerate(y_starts):
        for xi, x0 in enumerate(x_starts):
            x1 = min(int(width), int(x0 + tile))
            y1 = min(int(height), int(y0 + tile))
            tiles.append(
                {
                    "x0": int(x0),
                    "y0": int(y0),
                    "x1": int(x1),
                    "y1": int(y1),
                    "width": int(max(1, x1 - x0)),
                    "height": int(max(1, y1 - y0)),
                    "grid_x": int(xi),
                    "grid_y": int(yi),
                }
            )
    return tiles


def _clip_box_to_tile(
    *,
    xyxy: list[float],
    tile: dict[str, int],
    min_pixels: float,
) -> list[float] | None:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    tile_x0 = float(tile["x0"])
    tile_y0 = float(tile["y0"])
    tile_x1 = float(tile["x1"])
    tile_y1 = float(tile["y1"])
    clipped_x1 = max(tile_x0, x1)
    clipped_y1 = max(tile_y0, y1)
    clipped_x2 = min(tile_x1, x2)
    clipped_y2 = min(tile_y1, y2)
    if (clipped_x2 - clipped_x1) < float(min_pixels) or (clipped_y2 - clipped_y1) < float(
        min_pixels
    ):
        return None
    return [
        float(clipped_x1 - tile_x0),
        float(clipped_y1 - tile_y0),
        float(clipped_x2 - tile_x0),
        float(clipped_y2 - tile_y0),
    ]


def _box_center_in_tile(*, xyxy: list[float], tile: dict[str, int]) -> bool:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return bool(
        float(tile["x0"]) <= cx < float(tile["x1"]) and float(tile["y0"]) <= cy < float(tile["y1"])
    )


def _box_area(xyxy: list[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _build_training_tile_assignments(
    *,
    boxes: list[dict[str, Any]],
    tiles: list[dict[str, int]],
    min_pixels: float,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, float]]:
    assignments: dict[int, list[dict[str, Any]]] = {}
    assigned_objects = 0
    dropped_objects = 0
    clipped_area_sum = 0.0
    visible_area_sum = 0.0
    assignment_count = 0
    original_area_sum = 0.0

    for box in boxes:
        raw_xyxy = box.get("xyxy")
        if not isinstance(raw_xyxy, list) or len(raw_xyxy) != 4:
            continue
        xyxy = [float(v) for v in raw_xyxy]
        original_area = _box_area(xyxy)
        if original_area <= 0.0:
            dropped_objects += 1
            continue
        original_area_sum += original_area

        matched: list[tuple[int, list[float], float]] = []
        best_visible_area = 0.0
        for tile_index, tile in enumerate(tiles):
            clipped = _clip_box_to_tile(
                xyxy=xyxy,
                tile=tile,
                min_pixels=min_pixels,
            )
            if clipped is None:
                continue
            clipped_area = _box_area(clipped)
            if clipped_area <= 0.0:
                continue
            if clipped_area > best_visible_area:
                best_visible_area = clipped_area
            matched.append((tile_index, clipped, clipped_area))

        if not matched:
            dropped_objects += 1
            continue

        for tile_index, clipped, clipped_area in matched:
            assignments.setdefault(tile_index, []).append(
                {
                    "class_id": int(box["class_id"]),
                    "class_name": str(box["class_name"]),
                    "xyxy": clipped,
                }
            )
            assignment_count += 1
            visible_area_sum += clipped_area
        assigned_objects += 1
        clipped_area_sum += best_visible_area

    clipped_fraction = (
        max(0.0, min(1.0, clipped_area_sum / original_area_sum)) if original_area_sum > 0.0 else 0.0
    )
    return assignments, {
        "objects_total": float(len(boxes)),
        "objects_assigned": float(assigned_objects),
        "objects_dropped": float(dropped_objects),
        "assignment_count": float(assignment_count),
        "original_area_pixels_sum": float(round(original_area_sum, 6)),
        "clipped_area_pixels_sum": float(round(clipped_area_sum, 6)),
        "visible_area_pixels_sum": float(round(visible_area_sum, 6)),
        "clipped_area_fraction": float(round(clipped_fraction, 6)),
    }


def _deterministic_sample_rows(
    *,
    rows: list[dict[str, Any]],
    limit: int,
    seed_token: str,
) -> list[dict[str, Any]]:
    target = max(0, int(limit))
    if target <= 0 or not rows:
        return []
    if target >= len(rows):
        return list(rows)

    decorated: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        row_token = str(row.get("token") or row.get("sample_id") or "").strip() or "tile"
        digest = hashlib.sha256(f"{seed_token}|{row_token}".encode()).hexdigest()
        decorated.append((digest, row))
    decorated.sort(key=lambda item: item[0])
    return [row for _, row in decorated[:target]]


def _build_detection_size_histogram(boxes: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "area_lt_1024": 0,
        "area_1024_4096": 0,
        "area_4096_16384": 0,
        "area_ge_16384": 0,
    }
    for row in boxes:
        xyxy = row.get("xyxy")
        if not isinstance(xyxy, list) or len(xyxy) != 4:
            continue
        area = _box_area([float(value) for value in xyxy])
        if area < 1024.0:
            counts["area_lt_1024"] += 1
        elif area < 4096.0:
            counts["area_1024_4096"] += 1
        elif area < 16384.0:
            counts["area_4096_16384"] += 1
        else:
            counts["area_ge_16384"] += 1
    return counts


def _build_training_tile_boxes(
    *,
    boxes: list[dict[str, Any]],
    tile: dict[str, int],
    min_pixels: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for box in boxes:
        xyxy = box.get("xyxy")
        if not isinstance(xyxy, list) or len(xyxy) != 4:
            continue
        clipped = _clip_box_to_tile(
            xyxy=[float(v) for v in xyxy],
            tile=tile,
            min_pixels=min_pixels,
        )
        if clipped is None:
            continue
        output.append(
            {
                "class_id": int(box["class_id"]),
                "class_name": str(box["class_name"]),
                "xyxy": clipped,
            }
        )
    return output


def _parse_yolo_line_to_xyxy(
    *,
    line: str,
    width: int,
    height: int,
    min_pixels: float,
) -> dict[str, Any] | None:
    tokens = [token for token in str(line or "").strip().split() if token]
    if len(tokens) < 5:
        return None
    try:
        class_id = int(float(tokens[0]))
        cx = float(tokens[1])
        cy = float(tokens[2])
        bw = float(tokens[3])
        bh = float(tokens[4])
        conf_score = float(tokens[5]) if len(tokens) > 5 else None
    except Exception:
        return None
    x1 = (cx - bw / 2.0) * float(width)
    y1 = (cy - bh / 2.0) * float(height)
    x2 = (cx + bw / 2.0) * float(width)
    y2 = (cy + bh / 2.0) * float(height)
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if (x2 - x1) < float(min_pixels) or (y2 - y1) < float(min_pixels):
        return None
    class_name = (
        _YOLO_SUPPORTED_CLASSES[class_id]
        if 0 <= class_id < len(_YOLO_SUPPORTED_CLASSES)
        else str(class_id)
    )
    return {
        "class_id": int(class_id),
        "class_name": class_name,
        "confidence": conf_score,
        "xyxy": [float(x1), float(y1), float(x2), float(y2)],
    }


def _xyxy_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0
    return float(inter_area / denom)


def _classwise_nms(rows: list[dict[str, Any]], *, iou_threshold: float) -> list[dict[str, Any]]:
    if not rows:
        return []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("class_name") or row.get("class_id") or "").strip()
        if not key:
            continue
        grouped.setdefault(key, []).append(row)
    merged: list[dict[str, Any]] = []
    threshold = max(0.01, min(0.99, float(iou_threshold)))
    for _, group in grouped.items():
        ordered = sorted(
            group,
            key=lambda item: float(item.get("confidence") or 0.0),
            reverse=True,
        )
        kept: list[dict[str, Any]] = []
        while ordered:
            candidate = ordered.pop(0)
            kept.append(candidate)
            candidate_xyxy = candidate.get("xyxy")
            if not isinstance(candidate_xyxy, list) or len(candidate_xyxy) != 4:
                continue
            survivors: list[dict[str, Any]] = []
            for row in ordered:
                row_xyxy = row.get("xyxy")
                if not isinstance(row_xyxy, list) or len(row_xyxy) != 4:
                    continue
                if (
                    _xyxy_iou(
                        [float(v) for v in candidate_xyxy],
                        [float(v) for v in row_xyxy],
                    )
                    <= threshold
                ):
                    survivors.append(row)
            ordered = survivors
        merged.extend(kept)
    merged.sort(
        key=lambda item: (
            str(item.get("class_name") or ""),
            float(item.get("confidence") or 0.0),
        ),
        reverse=True,
    )
    return merged


def _write_yolo_prediction_labels(
    *,
    rows: list[dict[str, Any]],
    width: int,
    height: int,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for row in rows:
        xyxy = row.get("xyxy")
        if not isinstance(xyxy, list) or len(xyxy) != 4:
            continue
        class_id = int(row.get("class_id") or 0)
        converted = _xyxy_to_yolo_line(
            {"class_id": class_id, "xyxy": [float(v) for v in xyxy]},
            width,
            height,
        )
        confidence = row.get("confidence")
        if isinstance(confidence, (float, int)):
            lines.append(f"{converted} {float(confidence):.6f}")
        else:
            lines.append(converted)
    output_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
    return output_path


def _write_prediction_xml(
    *,
    image_path: Path,
    predictions: list[dict[str, Any]],
    output_path: Path,
    layer_name: str = "model_predictions",
) -> Path:
    root = ET.Element("image", name=image_path.name, value=image_path.name)
    ET.SubElement(root, "tag", name="source_image", value=str(image_path))
    layer = ET.SubElement(root, "gobject", name=layer_name)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        cls = str(row.get("class_name") or "").strip()
        if not cls:
            continue
        grouped.setdefault(cls, []).append(row)
    for class_name, rows in grouped.items():
        class_node = ET.SubElement(layer, "gobject", name=class_name)
        for row in rows:
            rect = ET.SubElement(class_node, "rectangle")
            x1, y1, x2, y2 = [float(v) for v in row.get("xyxy") or [0, 0, 0, 0]]
            ET.SubElement(rect, "vertex", index="0", x=f"{x1:.3f}", y=f"{y1:.3f}", z="0.0", t="0.0")
            ET.SubElement(rect, "vertex", index="1", x=f"{x2:.3f}", y=f"{y2:.3f}", z="0.0", t="0.0")
            if isinstance(row.get("confidence"), (float, int)):
                ET.SubElement(
                    rect, "tag", name="confidence", value=f"{float(row['confidence']):.6f}"
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    return output_path


def _find_best_weights(run_root: Path) -> Path | None:
    direct = run_root / "train" / "weights" / "best.pt"
    if direct.exists():
        return direct
    for path in sorted(run_root.rglob("best.pt")):
        if path.is_file():
            return path
    return None


def _read_map50(results_csv: Path) -> float | None:
    if not results_csv.exists():
        return None
    try:
        with results_csv.open("r", encoding="utf-8", errors="ignore") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return None
        last = rows[-1]
        normalized: dict[str, str] = {}
        for key, value in last.items():
            token = str(key or "").strip().lower().replace(" ", "")
            if not token:
                continue
            normalized[token] = str(value or "").strip()
        for key in (
            "metrics/map_0.5",
            "metrics/map50(b)",
            "map_0.5",
            "metrics/map50",
            "map50",
        ):
            value = normalized.get(key)
            if not value:
                continue
            try:
                return float(value)
            except Exception:
                continue
        for key, value in normalized.items():
            if "map" not in key:
                continue
            if "0.5" not in key and "50" not in key:
                continue
            try:
                return float(value)
            except Exception:
                continue
    except Exception:
        return None
    return None


def _find_results_csv_for_run(run_root: Path, run_name: str) -> Path | None:
    direct = run_root / run_name / "results.csv"
    if direct.exists() and direct.is_file():
        return direct
    candidates = []
    for path in run_root.rglob("results.csv"):
        if not path.is_file():
            continue
        if run_name and run_name not in str(path):
            continue
        candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates)[-1]


def _read_map50_from_console_text(text: str) -> float | None:
    raw = str(text or "")
    if not raw.strip():
        return None
    patterns = [
        r"^\s*all\s+\d+\s+\d+\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+([-+0-9.eE]+)\s+[-+0-9.eE]+",
        r"\bmAP50[:=\s]+([-+0-9.eE]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, raw, flags=re.MULTILINE)
        if not matches:
            continue
        token = str(matches[-1]).strip()
        try:
            return float(token)
        except Exception:
            continue
    return None


def _tail_text_file(path: Path, *, max_chars: int = 8000) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _parse_yaml_scalar(token: str) -> Any:
    value = str(token or "").strip()
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except Exception:
        return value


def _load_simple_yaml_mapping(path: Path) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    parent_key: str | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = str(raw_line).rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith((" ", "\t")):
            if not parent_key:
                continue
            stripped = line.strip()
            if ":" not in stripped:
                continue
            key, raw_value = stripped.split(":", 1)
            bucket = parsed.get(parent_key)
            if not isinstance(bucket, dict):
                bucket = {}
                parsed[parent_key] = bucket
            bucket[str(key).strip()] = _parse_yaml_scalar(raw_value)
            continue
        parent_key = None
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        token = str(key).strip()
        value = str(raw_value).strip()
        if not token:
            continue
        if value == "":
            parsed[token] = {}
            parent_key = token
        else:
            parsed[token] = _parse_yaml_scalar(value)
    return parsed


def _rewrite_text_token(text: str, *, find: str, replace: str) -> str:
    source = str(text or "")
    if not find:
        return source
    return source.replace(find, replace)


def _resolve_dataset_yaml_path(
    *,
    dataset_yaml: Path,
    dataset_path_entry: str,
    entry: str,
) -> Path:
    root = Path(str(dataset_path_entry or "")).expanduser()
    if not root.is_absolute():
        root = (dataset_yaml.parent / root).resolve()
    token = Path(str(entry or "").strip())
    if token.is_absolute():
        return token.resolve()
    return (root / token).resolve()


def _derive_labels_dir_from_images_dir(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    try:
        index = parts.index("images")
        parts[index] = "labels"
        return Path(*parts)
    except Exception:
        return images_dir.parent / "labels" / images_dir.name


def _parse_yolo_label_rows(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists() or not label_path.is_file():
        return rows
    for raw_line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        tokens = [token for token in str(raw_line).strip().split() if token]
        if len(tokens) < 5:
            continue
        try:
            class_id = int(float(tokens[0]))
            cx = float(tokens[1])
            cy = float(tokens[2])
            bw = float(tokens[3])
            bh = float(tokens[4])
        except Exception:
            continue
        rows.append((class_id, cx, cy, bw, bh))
    return rows


def _parse_val_table_rows(text: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    pattern = re.compile(
        r"^\s*(all|burrow|prairie_dog)\s+(\d+)\s+(\d+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$",
        flags=re.MULTILINE,
    )
    for match in pattern.finditer(str(text or "")):
        class_name = str(match.group(1) or "").strip()
        images = int(match.group(2))
        labels = int(match.group(3))
        precision = float(match.group(4))
        recall = float(match.group(5))
        map50 = float(match.group(6))
        map50_95 = float(match.group(7))
        tp = max(0.0, recall * float(labels))
        fn = max(0.0, float(labels) - tp)
        if precision > 0.0 and tp > 0.0:
            fp = max(0.0, (tp / precision) - tp)
        elif tp <= 0.0:
            fp = 0.0
        else:
            fp = float(labels)
        rows[class_name] = {
            "images": int(images),
            "labels": int(labels),
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "map50": round(float(map50), 6),
            "map50_95": round(float(map50_95), 6),
            "fp_per_image": round(float(fp) / float(max(1, images)), 6),
            "fn_per_image": round(float(fn) / float(max(1, images)), 6),
        }
    return rows


def _benchmark_metrics_complete(metrics: dict[str, Any]) -> bool:
    if not isinstance(metrics, dict):
        return False
    per_class = metrics.get("per_class")
    if not isinstance(per_class, dict):
        return False
    for class_name in _YOLO_SUPPORTED_CLASSES:
        row = per_class.get(class_name)
        if not isinstance(row, dict):
            return False
        required = (
            row.get("map50"),
            row.get("map50_95"),
            row.get("precision"),
            row.get("recall"),
            row.get("fp_per_image"),
            row.get("fn_per_image"),
        )
        if not all(isinstance(value, (int, float)) for value in required):
            return False
    return True


def _run_subprocess_with_heartbeat(
    *,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    progress_callback: ProgressCallback,
    control_callback: ControlCallback,
    heartbeat_event: str,
    heartbeat_phase: str,
    stdout_log_path: Path,
    stderr_log_path: Path,
    poll_interval_seconds: float = 1.0,
    heartbeat_interval_seconds: float = 15.0,
) -> dict[str, Any]:
    stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
    returncode = -1
    with (
        stdout_log_path.open("w", encoding="utf-8", errors="ignore") as stdout_handle,
        stderr_log_path.open(
            "w",
            encoding="utf-8",
            errors="ignore",
        ) as stderr_handle,
    ):
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        last_heartbeat = 0.0
        while True:
            try:
                control_callback()
            except Exception:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10.0)
                    except Exception:
                        process.kill()
                        with suppress(Exception):
                            process.wait(timeout=5.0)
                raise

            poll_code = process.poll()
            returncode = int(poll_code) if poll_code is not None else -1
            now = time.monotonic()
            if now - last_heartbeat >= max(1.0, float(heartbeat_interval_seconds)):
                progress_callback(
                    {
                        "event": heartbeat_event,
                        "phase": heartbeat_phase,
                        "pid": int(process.pid),
                    }
                )
                last_heartbeat = now
            if returncode >= 0:
                break
            time.sleep(max(0.1, float(poll_interval_seconds)))

    return {
        "returncode": returncode,
        "stdout_log_path": str(stdout_log_path),
        "stderr_log_path": str(stderr_log_path),
        "stdout_tail": _tail_text_file(stdout_log_path, max_chars=8000),
        "stderr_tail": _tail_text_file(stderr_log_path, max_chars=8000),
    }


class YOLOv5Adapter(BaseTrainingAdapter):
    key = "yolov5_rarespot"
    supports_training = True
    supports_finetune = True
    supports_inference = True

    def _require_runtime(self) -> Path:
        repo = _resolve_yolov5_repo_path()
        if (
            not (repo / "detect.py").exists()
            or not (repo / "train.py").exists()
            or not (repo / "val.py").exists()
        ):
            raise ValueError(
                "YOLOv5 runtime not found at "
                f"{repo}. Set YOLOV5_RUNTIME_PATH or add third_party/yolov5."
            )
        return repo

    def _load_canonical_benchmark_spec(self, *, config: dict[str, Any]) -> dict[str, Any]:
        spec_path_raw = str(config.get("canonical_benchmark_spec_path") or "").strip()
        if not spec_path_raw:
            raise ValueError("failed_pre_eval: canonical benchmark spec path is not configured.")
        spec_path = Path(spec_path_raw).expanduser().resolve()
        if not spec_path.exists() or not spec_path.is_file():
            raise ValueError(
                f"failed_pre_eval: canonical benchmark spec file was not found at {spec_path}."
            )
        spec = _load_simple_yaml_mapping(spec_path)
        dataset_yaml_raw = str(spec.get("dataset_yaml") or "").strip()
        if not dataset_yaml_raw:
            raise ValueError("failed_pre_eval: canonical benchmark spec missing dataset_yaml.")
        dataset_yaml = Path(dataset_yaml_raw).expanduser().resolve()
        if not dataset_yaml.exists() or not dataset_yaml.is_file():
            raise ValueError(
                "failed_pre_eval: canonical benchmark dataset yaml was not found at "
                f"{dataset_yaml}."
            )
        rewrite = (
            spec.get("yaml_path_rewrite") if isinstance(spec.get("yaml_path_rewrite"), dict) else {}
        )
        return {
            "spec_path": str(spec_path),
            "dataset_yaml": str(dataset_yaml),
            "rewrite_find": str(rewrite.get("find") or "").strip(),
            "rewrite_replace": str(rewrite.get("replace") or "").strip(),
            "val_task": str(spec.get("val_task") or "test").strip().lower() or "test",
            "imgsz": _safe_int(spec.get("imgsz"), default=512, minimum=128, maximum=4096),
            "batch_size": _safe_int(spec.get("batch_size"), default=16, minimum=1, maximum=256),
        }

    def _prepare_canonical_data_yaml(
        self,
        *,
        spec: dict[str, Any],
        output_dir: Path,
    ) -> Path:
        dataset_yaml = Path(str(spec.get("dataset_yaml") or "")).expanduser().resolve()
        content = dataset_yaml.read_text(encoding="utf-8", errors="ignore")
        rewrite_find = str(spec.get("rewrite_find") or "").strip()
        rewrite_replace = str(spec.get("rewrite_replace") or "").strip()
        if rewrite_find and rewrite_replace:
            content = _rewrite_text_token(content, find=rewrite_find, replace=rewrite_replace)
        canonical_yaml = output_dir / "canonical_benchmark_data.yaml"
        canonical_yaml.write_text(content, encoding="utf-8")
        return canonical_yaml

    def _collect_canonical_replay_pool(
        self,
        *,
        spec: dict[str, Any],
    ) -> list[dict[str, Any]]:
        dataset_yaml = Path(str(spec.get("dataset_yaml") or "")).expanduser().resolve()
        raw = _load_simple_yaml_mapping(dataset_yaml)
        dataset_path_entry = str(raw.get("path") or "").strip()
        train_entry = str(raw.get("train") or "").strip()
        if not dataset_path_entry or not train_entry:
            return []
        images_dir = _resolve_dataset_yaml_path(
            dataset_yaml=dataset_yaml,
            dataset_path_entry=dataset_path_entry,
            entry=train_entry,
        )
        if not images_dir.exists() or not images_dir.is_dir():
            return []
        labels_dir = _derive_labels_dir_from_images_dir(images_dir)
        candidates: list[dict[str, Any]] = []
        for image_path in sorted(images_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in _YOLO_IMG_SUFFIXES:
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists() or not label_path.is_file():
                continue
            label_rows = _parse_yolo_label_rows(label_path)
            if not label_rows:
                continue
            small_object_count = 0
            classes: set[str] = set()
            for class_id, _, _, bw, bh in label_rows:
                if 0 <= class_id < len(_YOLO_SUPPORTED_CLASSES):
                    classes.add(_YOLO_SUPPORTED_CLASSES[class_id])
                tile_area_fraction = max(0.0, float(bw)) * max(0.0, float(bh))
                if tile_area_fraction <= 0.0009:
                    small_object_count += 1
            candidates.append(
                {
                    "image_path": str(image_path.resolve()),
                    "label_path": str(label_path.resolve()),
                    "sample_id": image_path.stem,
                    "small_object_count": int(small_object_count),
                    "class_tags": sorted(classes),
                }
            )
        return candidates

    def _evaluate_dataset_metrics(
        self,
        *,
        repo: Path,
        runtime_env: dict[str, str],
        run_root: Path,
        run_name: str,
        weights_path: Path,
        data_yaml: Path,
        imgsz: int,
        batch_size: int,
        task: str,
        progress_callback: ProgressCallback,
        control_callback: ControlCallback,
        output_dir: Path,
    ) -> dict[str, Any]:
        command = [
            sys.executable,
            str(repo / "val.py"),
            "--weights",
            str(weights_path),
            "--data",
            str(data_yaml),
            "--img",
            str(imgsz),
            "--batch",
            str(batch_size),
            "--task",
            str(task or "val"),
            "--project",
            str(run_root),
            "--name",
            run_name,
            "--exist-ok",
        ]
        process_result = _run_subprocess_with_heartbeat(
            command=command,
            cwd=repo,
            env=runtime_env,
            progress_callback=progress_callback,
            control_callback=control_callback,
            heartbeat_event="validation_heartbeat",
            heartbeat_phase=run_name,
            stdout_log_path=output_dir / f"{run_name}.stdout.log",
            stderr_log_path=output_dir / f"{run_name}.stderr.log",
        )
        if int(process_result.get("returncode") or 0) != 0:
            stderr_tail = str(process_result.get("stderr_tail") or "").strip()
            stdout_tail = str(process_result.get("stdout_tail") or "").strip()
            raise ValueError(f"YOLOv5 validation failed: {stderr_tail or stdout_tail}")
        results_csv = _find_results_csv_for_run(run_root, run_name)
        map50 = _read_map50(results_csv) if results_csv is not None else None
        stdout_log_path = Path(str(process_result.get("stdout_log_path") or "")).expanduser()
        stderr_log_path = Path(str(process_result.get("stderr_log_path") or "")).expanduser()
        stdout_text = _tail_text_file(stdout_log_path, max_chars=500_000)
        stderr_text = _tail_text_file(stderr_log_path, max_chars=500_000)
        console_text = "\n".join([stderr_text, stdout_text]).strip()
        if map50 is None:
            map50 = _read_map50_from_console_text(console_text)
        table_rows = _parse_val_table_rows(console_text)
        all_row = table_rows.get("all") if isinstance(table_rows.get("all"), dict) else {}
        per_class: dict[str, dict[str, Any]] = {}
        for class_name in _YOLO_SUPPORTED_CLASSES:
            row = table_rows.get(class_name)
            per_class[class_name] = row if isinstance(row, dict) else {}
        map50_value = (
            float(map50)
            if isinstance(map50, (int, float))
            else float(all_row.get("map50"))
            if isinstance(all_row.get("map50"), (int, float))
            else None
        )
        map50_95_value = (
            float(all_row.get("map50_95"))
            if isinstance(all_row.get("map50_95"), (int, float))
            else None
        )
        return {
            "map50": round(float(map50_value), 6)
            if isinstance(map50_value, (float, int))
            else None,
            "map50_95": round(float(map50_95_value), 6)
            if isinstance(map50_95_value, (float, int))
            else None,
            "precision": round(float(all_row.get("precision")), 6)
            if isinstance(all_row.get("precision"), (float, int))
            else None,
            "recall": round(float(all_row.get("recall")), 6)
            if isinstance(all_row.get("recall"), (float, int))
            else None,
            "fp_per_image": round(float(all_row.get("fp_per_image")), 6)
            if isinstance(all_row.get("fp_per_image"), (float, int))
            else None,
            "fn_per_image": round(float(all_row.get("fn_per_image")), 6)
            if isinstance(all_row.get("fn_per_image"), (float, int))
            else None,
            "per_class": per_class,
            "task": str(task or "val"),
            "results_csv": str(results_csv) if results_csv is not None else None,
            "stdout_log_path": str(process_result.get("stdout_log_path") or "").strip() or None,
            "stderr_log_path": str(process_result.get("stderr_log_path") or "").strip() or None,
            "stdout_tail": str(process_result.get("stdout_tail") or "").strip() or None,
            "stderr_tail": str(process_result.get("stderr_tail") or "").strip() or None,
        }

    def _build_benchmark_comparison(
        self,
        *,
        baseline_before_train: dict[str, Any],
        candidate_after_train: dict[str, Any],
        thresholds: dict[str, float],
    ) -> dict[str, Any]:
        def _map50(row: dict[str, Any], class_name: str | None = None) -> float | None:
            if class_name:
                per_class = row.get("per_class")
                if not isinstance(per_class, dict):
                    return None
                class_row = per_class.get(class_name)
                if not isinstance(class_row, dict):
                    return None
                value = class_row.get("map50")
            else:
                value = row.get("map50")
            if not isinstance(value, (float, int)):
                return None
            return float(value)

        def _fp_image(row: dict[str, Any]) -> float | None:
            value = row.get("fp_per_image")
            if isinstance(value, (float, int)):
                return float(value)
            return None

        canonical_baseline = (
            baseline_before_train.get("canonical")
            if isinstance(baseline_before_train.get("canonical"), dict)
            else {}
        )
        canonical_candidate = (
            candidate_after_train.get("canonical")
            if isinstance(candidate_after_train.get("canonical"), dict)
            else {}
        )
        active_baseline = (
            baseline_before_train.get("active")
            if isinstance(baseline_before_train.get("active"), dict)
            else {}
        )
        active_candidate = (
            candidate_after_train.get("active")
            if isinstance(candidate_after_train.get("active"), dict)
            else {}
        )

        canonical_drop = None
        prairie_dog_drop = None
        active_drop = None
        fp_increase_ratio = None

        canonical_baseline_map50 = _map50(canonical_baseline)
        canonical_candidate_map50 = _map50(canonical_candidate)
        prairie_baseline_map50 = _map50(canonical_baseline, class_name="prairie_dog")
        prairie_candidate_map50 = _map50(canonical_candidate, class_name="prairie_dog")
        active_baseline_map50 = _map50(active_baseline)
        active_candidate_map50 = _map50(active_candidate)
        canonical_baseline_fp = _fp_image(canonical_baseline)
        canonical_candidate_fp = _fp_image(canonical_candidate)

        if canonical_baseline_map50 is not None and canonical_candidate_map50 is not None:
            canonical_drop = max(0.0, canonical_baseline_map50 - canonical_candidate_map50)
        if prairie_baseline_map50 is not None and prairie_candidate_map50 is not None:
            prairie_dog_drop = max(0.0, prairie_baseline_map50 - prairie_candidate_map50)
        if active_baseline_map50 is not None and active_candidate_map50 is not None:
            active_drop = max(0.0, active_baseline_map50 - active_candidate_map50)
        if (
            canonical_baseline_fp is not None
            and canonical_candidate_fp is not None
            and canonical_baseline_fp > 0.0
        ):
            fp_increase_ratio = max(
                0.0,
                (canonical_candidate_fp - canonical_baseline_fp) / canonical_baseline_fp,
            )

        gate_checks = {
            "canonical_map50_drop_ok": (
                canonical_drop is not None
                and canonical_drop <= float(thresholds.get("canonical_map50_drop_max", 0.02))
            ),
            "prairie_dog_map50_drop_ok": (
                prairie_dog_drop is not None
                and prairie_dog_drop <= float(thresholds.get("prairie_dog_map50_drop_max", 0.03))
            ),
            "active_map50_drop_ok": (
                active_drop is not None
                and active_drop <= float(thresholds.get("active_map50_drop_max", 0.02))
            ),
            "canonical_fp_increase_ok": (
                fp_increase_ratio is not None
                and fp_increase_ratio <= float(thresholds.get("canonical_fp_increase_max", 0.25))
            ),
        }
        required_packets = (
            _benchmark_metrics_complete(canonical_baseline)
            and _benchmark_metrics_complete(canonical_candidate)
            and _benchmark_metrics_complete(active_baseline)
            and _benchmark_metrics_complete(active_candidate)
        )
        passed = bool(required_packets and all(gate_checks.values()))
        reasons: list[str] = []
        if not required_packets:
            reasons.append("Missing benchmark packet metrics for canonical or active holdout.")
        if gate_checks["canonical_map50_drop_ok"] is False:
            reasons.append(
                "Canonical mAP50 regression exceeds threshold "
                f"({(canonical_drop or 0.0):.4f} > {thresholds.get('canonical_map50_drop_max', 0.02):.4f})."
            )
        if gate_checks["prairie_dog_map50_drop_ok"] is False:
            reasons.append(
                "prairie_dog class canonical mAP50 regression exceeds threshold "
                f"({(prairie_dog_drop or 0.0):.4f} > {thresholds.get('prairie_dog_map50_drop_max', 0.03):.4f})."
            )
        if gate_checks["active_map50_drop_ok"] is False:
            reasons.append(
                "Active holdout mAP50 regression exceeds threshold "
                f"({(active_drop or 0.0):.4f} > {thresholds.get('active_map50_drop_max', 0.02):.4f})."
            )
        if gate_checks["canonical_fp_increase_ok"] is False:
            reasons.append(
                "Canonical FP/image increase exceeds threshold "
                f"({(fp_increase_ratio or 0.0):.4f} > {thresholds.get('canonical_fp_increase_max', 0.25):.4f})."
            )
        return {
            "passed": passed,
            "required_packet_complete": required_packets,
            "checks": gate_checks,
            "reasons": reasons,
            "metrics": {
                "canonical_map50_drop": round(float(canonical_drop), 6)
                if canonical_drop is not None
                else None,
                "prairie_dog_map50_drop": round(float(prairie_dog_drop), 6)
                if prairie_dog_drop is not None
                else None,
                "active_map50_drop": round(float(active_drop), 6)
                if active_drop is not None
                else None,
                "canonical_fp_increase_ratio": round(float(fp_increase_ratio), 6)
                if fp_increase_ratio is not None
                else None,
            },
            "thresholds": {
                "canonical_map50_drop_max": float(thresholds.get("canonical_map50_drop_max", 0.02)),
                "prairie_dog_map50_drop_max": float(
                    thresholds.get("prairie_dog_map50_drop_max", 0.03)
                ),
                "active_map50_drop_max": float(thresholds.get("active_map50_drop_max", 0.02)),
                "canonical_fp_increase_max": float(
                    thresholds.get("canonical_fp_increase_max", 0.25)
                ),
            },
        }

    def train(
        self,
        *,
        manifest: dict[str, Any],
        config: dict[str, Any],
        output_dir: Path,
        progress_callback: ProgressCallback,
        control_callback: ControlCallback,
        initial_checkpoint_path: str | None = None,
    ) -> dict[str, Any]:
        control_callback()
        output_dir.mkdir(parents=True, exist_ok=True)
        repo = self._require_runtime()

        train_pairs = _yolo_train_pairs(manifest, "train")
        val_pairs = _yolo_train_pairs(manifest, "val")
        if not train_pairs or not val_pairs:
            raise ValueError("Detection training requires train and val image/annotation pairs.")

        tile_size = _safe_int(config.get("tile_size"), default=512, minimum=128, maximum=4096)
        train_tile_overlap = _clamp_overlap(config.get("train_tile_overlap"), default=0.25)
        include_empty_tiles = _safe_bool(config.get("include_empty_tiles"), default=True)
        empty_tile_ratio = max(0.0, min(2.0, _safe_float(config.get("empty_tile_ratio"), 1.0)))
        min_box_pixels = max(1.0, _safe_float(config.get("min_box_pixels"), 4.0))
        replay_new_ratio = max(0.01, min(0.99, _safe_float(config.get("replay_new_ratio"), 0.6)))
        replay_old_ratio = max(0.0, min(0.99, _safe_float(config.get("replay_old_ratio"), 0.4)))

        dataset_root = output_dir / "dataset"
        for split in ("train", "val"):
            (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)

        class_counts: dict[str, int] = {name: 0 for name in _YOLO_SUPPORTED_CLASSES}
        unsupported_counts: dict[str, int] = {}
        reviewed_count = 0
        tile_counts: dict[str, dict[str, int | float]] = {
            "train": {
                "source_images": 0,
                "tiles_generated": 0,
                "annotated_tiles": 0,
                "empty_tiles": 0,
                "tiles_selected": 0,
                "positive_tiles_all_intersections": 0,
                "positive_tiles_selected": 0,
                "hidden_positive_tiles": 0,
                "objects_assigned": 0,
                "objects_dropped": 0,
                "box_tile_assignments": 0,
            },
            "val": {
                "source_images": 0,
                "tiles_generated": 0,
                "annotated_tiles": 0,
                "empty_tiles": 0,
                "tiles_selected": 0,
                "positive_tiles_all_intersections": 0,
                "positive_tiles_selected": 0,
                "hidden_positive_tiles": 0,
                "objects_assigned": 0,
                "objects_dropped": 0,
                "box_tile_assignments": 0,
            },
        }
        object_size_histogram: dict[str, int] = {
            "area_lt_1024": 0,
            "area_1024_4096": 0,
            "area_4096_16384": 0,
            "area_ge_16384": 0,
        }
        clipping_stats = {
            "objects_total": 0.0,
            "objects_assigned": 0.0,
            "objects_dropped": 0.0,
            "original_area_pixels_sum": 0.0,
            "clipped_area_pixels_sum": 0.0,
        }
        deterministic_seed = hashlib.sha256(
            json.dumps(
                {
                    "dataset_id": manifest.get("dataset_id"),
                    "counts": manifest.get("counts"),
                    "splits": manifest.get("splits"),
                },
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        replay_manifest_rows: list[dict[str, Any]] = []
        replay_source_total = 0
        replay_tiles_injected = 0

        for split, pairs in (("train", train_pairs), ("val", val_pairs)):
            tile_rows_with_objects: list[dict[str, Any]] = []
            tile_rows_empty: list[dict[str, Any]] = []
            for idx, pair in enumerate(pairs):
                control_callback()
                image_path = Path(pair["image_path"]).expanduser().resolve()
                ann_path = Path(pair["annotation_path"]).expanduser().resolve()
                if not image_path.exists() or image_path.suffix.lower() not in _YOLO_IMG_SUFFIXES:
                    continue
                if not ann_path.exists():
                    continue
                with Image.open(image_path) as image:
                    width, height = image.size
                boxes, unsupported = _parse_bisque_rectangles(ann_path, layer_name="gt2")
                hist = _build_detection_size_histogram(boxes)
                for key, value in hist.items():
                    object_size_histogram[key] = object_size_histogram.get(key, 0) + int(value)
                reviewed_count += 1
                tile_counts[split]["source_images"] = int(tile_counts[split]["source_images"]) + 1
                for cls, count in unsupported.items():
                    unsupported_counts[cls] = unsupported_counts.get(cls, 0) + int(count)
                for box in boxes:
                    class_counts[str(box["class_name"])] = (
                        class_counts.get(str(box["class_name"]), 0) + 1
                    )
                tiles = _build_sliding_tiles(
                    width=width,
                    height=height,
                    tile_size=tile_size,
                    overlap=train_tile_overlap,
                )
                tile_assignments, assign_stats = _build_training_tile_assignments(
                    boxes=boxes,
                    tiles=tiles,
                    min_pixels=min_box_pixels,
                )
                clipping_stats["objects_total"] += float(assign_stats.get("objects_total") or 0.0)
                clipping_stats["objects_assigned"] += float(
                    assign_stats.get("objects_assigned") or 0.0
                )
                clipping_stats["objects_dropped"] += float(
                    assign_stats.get("objects_dropped") or 0.0
                )
                clipping_stats["original_area_pixels_sum"] += float(
                    assign_stats.get("original_area_pixels_sum") or 0.0
                )
                clipping_stats["clipped_area_pixels_sum"] += float(
                    assign_stats.get("clipped_area_pixels_sum") or 0.0
                )
                tile_counts[split]["objects_assigned"] = int(
                    tile_counts[split]["objects_assigned"]
                ) + int(assign_stats.get("objects_assigned") or 0)
                tile_counts[split]["objects_dropped"] = int(
                    tile_counts[split]["objects_dropped"]
                ) + int(assign_stats.get("objects_dropped") or 0)
                tile_counts[split]["box_tile_assignments"] = int(
                    tile_counts[split]["box_tile_assignments"]
                ) + int(assign_stats.get("assignment_count") or 0)
                tile_counts[split]["tiles_generated"] = int(
                    tile_counts[split]["tiles_generated"]
                ) + len(tiles)
                for tile_index, tile in enumerate(tiles):
                    tile_box_rows = tile_assignments.get(tile_index, [])
                    has_objects = bool(tile_box_rows)
                    if not has_objects and not include_empty_tiles:
                        continue
                    if has_objects:
                        tile_counts[split]["annotated_tiles"] = (
                            int(tile_counts[split]["annotated_tiles"]) + 1
                        )
                    else:
                        tile_counts[split]["empty_tiles"] = (
                            int(tile_counts[split]["empty_tiles"]) + 1
                        )
                    token = _slug_token(
                        f"{pair['sample_id']}-{idx:04d}-"
                        f"x{int(tile['x0']):05d}-y{int(tile['y0']):05d}"
                    )
                    row_payload = {
                        "token": token,
                        "tile": tile,
                        "tile_box_rows": tile_box_rows,
                        "image_path": str(image_path),
                    }
                    if has_objects:
                        tile_rows_with_objects.append(row_payload)
                    else:
                        tile_rows_empty.append(row_payload)

            tile_counts[split]["positive_tiles_all_intersections"] = len(tile_rows_with_objects)
            selected_rows = list(tile_rows_with_objects)
            if include_empty_tiles and tile_rows_empty:
                empty_target = round(float(len(tile_rows_with_objects)) * float(empty_tile_ratio))
                if len(tile_rows_with_objects) <= 0:
                    empty_target = min(
                        len(tile_rows_empty), max(1, int(len(tile_rows_empty) * 0.25))
                    )
                sampled_empty = _deterministic_sample_rows(
                    rows=tile_rows_empty,
                    limit=min(len(tile_rows_empty), max(0, empty_target)),
                    seed_token=f"{deterministic_seed}:{split}",
                )
                selected_rows.extend(sampled_empty)
            tile_counts[split]["tiles_selected"] = len(selected_rows)
            positive_tiles_selected = sum(
                1
                for row in selected_rows
                if isinstance(row.get("tile_box_rows"), list)
                and len(row.get("tile_box_rows") or []) > 0
            )
            tile_counts[split]["positive_tiles_selected"] = int(positive_tiles_selected)
            tile_counts[split]["hidden_positive_tiles"] = max(
                0,
                int(tile_counts[split]["positive_tiles_all_intersections"])
                - int(positive_tiles_selected),
            )

            if split == "train" and selected_rows:
                replay_ratio = replay_old_ratio / max(1e-9, replay_new_ratio)
                replay_target = max(0, round(float(len(selected_rows)) * replay_ratio))
                if replay_target > 0:
                    try:
                        canonical_spec = self._load_canonical_benchmark_spec(config=config)
                        replay_pool = self._collect_canonical_replay_pool(spec=canonical_spec)
                    except Exception:
                        replay_pool = []
                    replay_source_total = max(replay_source_total, len(replay_pool))
                    if replay_pool:
                        replay_rows = [
                            {
                                "token": _slug_token(f"replay-{row['sample_id']}-{index:06d}"),
                                "image_path": str(row["image_path"]),
                                "label_path": str(row["label_path"]),
                                "class_tags": row.get("class_tags")
                                if isinstance(row.get("class_tags"), list)
                                else [],
                                "small_object_count": int(row.get("small_object_count") or 0),
                            }
                            for index, row in enumerate(replay_pool)
                        ]
                        sampled_replay = _deterministic_sample_rows(
                            rows=replay_rows,
                            limit=min(len(replay_rows), replay_target),
                            seed_token=f"{deterministic_seed}:{split}:replay",
                        )
                        for replay_row in sampled_replay:
                            image_path_raw = str(replay_row.get("image_path") or "").strip()
                            label_path_raw = str(replay_row.get("label_path") or "").strip()
                            if not image_path_raw or not label_path_raw:
                                continue
                            image_path = Path(image_path_raw).expanduser().resolve()
                            label_path = Path(label_path_raw).expanduser().resolve()
                            if not image_path.exists() or not image_path.is_file():
                                continue
                            if not label_path.exists() or not label_path.is_file():
                                continue
                            token = str(replay_row.get("token") or "").strip() or _slug_token(
                                f"replay-{image_path.stem}"
                            )
                            target_image = dataset_root / "images" / split / f"{token}.jpg"
                            target_label = dataset_root / "labels" / split / f"{token}.txt"
                            shutil.copy2(image_path, target_image)
                            target_label.write_text(
                                label_path.read_text(encoding="utf-8", errors="ignore"),
                                encoding="utf-8",
                            )
                            replay_tiles_injected += 1
                            replay_manifest_rows.append(
                                {
                                    "token": token,
                                    "image_path": str(image_path),
                                    "label_path": str(label_path),
                                    "class_tags": list(replay_row.get("class_tags") or []),
                                    "small_object_count": int(
                                        replay_row.get("small_object_count") or 0
                                    ),
                                }
                            )

            image_cache: dict[str, Image.Image] = {}
            for row in selected_rows:
                control_callback()
                tile = row["tile"]
                token = str(row["token"])
                tile_box_rows = (
                    row.get("tile_box_rows") if isinstance(row.get("tile_box_rows"), list) else []
                )
                image_path_raw = str(row.get("image_path") or "").strip()
                if not image_path_raw:
                    continue
                image_ref = image_cache.get(image_path_raw)
                if image_ref is None:
                    source_path = Path(image_path_raw).expanduser().resolve()
                    if not source_path.exists() or not source_path.is_file():
                        continue
                    with Image.open(source_path) as source_image:
                        image_ref = source_image.convert("RGB")
                    image_cache[image_path_raw] = image_ref
                target_image = dataset_root / "images" / split / f"{token}.jpg"
                image_ref.crop(
                    (
                        int(tile["x0"]),
                        int(tile["y0"]),
                        int(tile["x1"]),
                        int(tile["y1"]),
                    )
                ).save(target_image, format="JPEG", quality=95)
                label_path = dataset_root / "labels" / split / f"{token}.txt"
                lines = [
                    _xyxy_to_yolo_line(
                        {
                            "class_id": int(box_row["class_id"]),
                            "xyxy": [float(v) for v in box_row["xyxy"]],
                        },
                        int(tile["width"]),
                        int(tile["height"]),
                    )
                    for box_row in tile_box_rows
                ]
                label_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
            for cached_image in image_cache.values():
                with suppress(Exception):
                    cached_image.close()

        if int(tile_counts["train"]["tiles_selected"]) <= 0:
            raise ValueError(
                "No train tiles were generated for YOLO training dataset. "
                "Check tile settings and dataset annotations."
            )
        if int(tile_counts["val"]["tiles_selected"]) <= 0:
            raise ValueError(
                "No val tiles were generated for YOLO training dataset. "
                "Check tile settings and dataset annotations."
            )

        data_yaml = dataset_root / "data.yaml"
        data_yaml.write_text(
            (
                f"path: {dataset_root.resolve()}\n"
                "train: images/train\n"
                "val: images/val\n"
                f"nc: {len(_YOLO_SUPPORTED_CLASSES)}\n"
                "names:\n"
                + "\n".join(
                    [
                        f"  {index}: {json.dumps(name)}"
                        for index, name in enumerate(_YOLO_SUPPORTED_CLASSES)
                    ]
                )
                + "\n"
            ),
            encoding="utf-8",
        )

        weights_path = _resolve_rarespot_weights_path()
        if not weights_path.exists():
            raise ValueError(f"RareSpot weights not found at {weights_path}")

        epochs = _safe_int(config.get("epochs"), default=20, minimum=1, maximum=300)
        imgsz = _safe_int(config.get("imgsz"), default=512, minimum=128, maximum=2048)
        batch_size = _safe_int(config.get("batch_size"), default=4, minimum=1, maximum=128)
        small_dataset_object_threshold = _safe_int(
            config.get("small_dataset_object_threshold"), default=300, minimum=1, maximum=1_000_000
        )
        small_dataset_epochs = _safe_int(
            config.get("small_dataset_epochs"), default=6, minimum=1, maximum=300
        )
        if int(sum(class_counts.values())) < small_dataset_object_threshold:
            epochs = min(epochs, small_dataset_epochs)
        run_root = output_dir / "runs"
        run_root.mkdir(parents=True, exist_ok=True)
        runtime_env = os.environ.copy()
        config_dir = (output_dir / ".yolov5-config").resolve()
        _ensure_yolov5_font_asset(config_dir)
        runtime_env["YOLOV5_CONFIG_DIR"] = str(config_dir)
        # Torch 2.6+ defaults torch.load(weights_only=True), which breaks older
        # YOLOv5 checkpoints that rely on module pickles.
        runtime_env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        hyp_path = str(config.get("hyp_path") or "").strip()
        if not hyp_path:
            hyp_path = str(
                (Path(__file__).resolve().parent / "hyp.prairie_finetune.yaml").resolve()
            )
        hyp_file = Path(hyp_path).expanduser().resolve()
        if not hyp_file.exists() or not hyp_file.is_file():
            raise ValueError(f"YOLOv5 finetune hyp profile not found at {hyp_file}")
        freeze_layers = _safe_int(config.get("freeze_layers"), default=10, minimum=0, maximum=64)
        patience = _safe_int(config.get("patience"), default=3, minimum=1, maximum=100)
        canonical_spec = self._load_canonical_benchmark_spec(config=config)
        canonical_data_yaml = self._prepare_canonical_data_yaml(
            spec=canonical_spec,
            output_dir=output_dir,
        )
        guardrail_thresholds = {
            "canonical_map50_drop_max": _safe_float(
                config.get("guardrail_canonical_map50_drop_max"), 0.02
            ),
            "prairie_dog_map50_drop_max": _safe_float(
                config.get("guardrail_prairie_dog_map50_drop_max"), 0.03
            ),
            "active_map50_drop_max": _safe_float(
                config.get("guardrail_active_map50_drop_max"), 0.02
            ),
            "canonical_fp_increase_max": _safe_float(
                config.get("guardrail_canonical_fp_image_increase_max"), 0.25
            ),
        }
        enable_hard_sample_bank = _safe_bool(config.get("enable_hard_sample_bank"), default=False)
        hard_sample_injection_ratio = max(
            0.0,
            min(1.0, _safe_float(config.get("hard_sample_injection_ratio"), 0.2)),
        )
        enable_small_object_weighting = _safe_bool(
            config.get("enable_small_object_weighting"),
            default=False,
        )

        start_weights = weights_path
        resume_checkpoint: Path | None = None
        if initial_checkpoint_path:
            candidate = Path(initial_checkpoint_path).expanduser().resolve()
            if candidate.exists() and candidate.is_file() and candidate.suffix.lower() == ".pt":
                resume_checkpoint = candidate
                start_weights = candidate

        baseline_active_eval = self._evaluate_dataset_metrics(
            repo=repo,
            runtime_env=runtime_env,
            run_root=run_root,
            run_name="active-baseline-before-train",
            weights_path=start_weights,
            data_yaml=data_yaml,
            imgsz=imgsz,
            batch_size=batch_size,
            task="val",
            progress_callback=progress_callback,
            control_callback=control_callback,
            output_dir=output_dir,
        )
        baseline_canonical_eval = self._evaluate_dataset_metrics(
            repo=repo,
            runtime_env=runtime_env,
            run_root=run_root,
            run_name="canonical-baseline-before-train",
            weights_path=start_weights,
            data_yaml=canonical_data_yaml,
            imgsz=_safe_int(canonical_spec.get("imgsz"), default=512, minimum=128, maximum=4096),
            batch_size=_safe_int(
                canonical_spec.get("batch_size"), default=16, minimum=1, maximum=256
            ),
            task=str(canonical_spec.get("val_task") or "test"),
            progress_callback=progress_callback,
            control_callback=control_callback,
            output_dir=output_dir,
        )
        baseline_before_train = {
            "active": baseline_active_eval,
            "canonical": baseline_canonical_eval,
            "model_artifact_path": str(start_weights),
        }
        if not _benchmark_metrics_complete(baseline_canonical_eval):
            baseline_path = output_dir / "baseline_before_train.json"
            _write_json(baseline_path, baseline_before_train)
            raise ValueError(
                "failed_pre_eval: canonical baseline benchmark metrics are incomplete; "
                "candidate training is blocked."
            )
        progress_callback(
            {
                "event": "training_started",
                "epochs": epochs,
                "imgsz": imgsz,
                "batch_size": batch_size,
                "tile_size": tile_size,
                "train_tile_overlap": train_tile_overlap,
                "include_empty_tiles": include_empty_tiles,
                "weights_path": str(start_weights),
                "resume_checkpoint_path": str(resume_checkpoint) if resume_checkpoint else None,
                "baseline_map50": baseline_active_eval.get("map50"),
            }
        )
        command = [
            sys.executable,
            str(repo / "train.py"),
            "--weights",
            str(start_weights),
            "--data",
            str(data_yaml),
            "--epochs",
            str(epochs),
            "--img",
            str(imgsz),
            "--batch",
            str(batch_size),
            "--project",
            str(run_root),
            "--name",
            "train",
            "--exist-ok",
            "--cos-lr",
            "--hyp",
            str(hyp_file),
            "--freeze",
            str(freeze_layers),
            "--patience",
            str(patience),
        ]
        run_search_root = run_root
        process_result = _run_subprocess_with_heartbeat(
            command=command,
            cwd=repo,
            env=runtime_env,
            progress_callback=progress_callback,
            control_callback=control_callback,
            heartbeat_event="training_heartbeat",
            heartbeat_phase="yolov5_train_subprocess",
            stdout_log_path=output_dir / "train.stdout.log",
            stderr_log_path=output_dir / "train.stderr.log",
        )
        if int(process_result.get("returncode") or 0) != 0:
            stderr_tail = str(process_result.get("stderr_tail") or "").strip()
            stdout_tail = str(process_result.get("stdout_tail") or "").strip()
            raise ValueError(f"YOLOv5 training failed: {stderr_tail or stdout_tail}")

        best_weights = _find_best_weights(run_search_root) or _find_best_weights(run_root)
        if best_weights is None:
            raise ValueError("YOLOv5 training finished but best.pt was not found.")
        model_artifact = output_dir / "model-final.pt"
        shutil.copy2(best_weights, model_artifact)
        metrics_csv = next(iter(sorted(run_search_root.rglob("results.csv"))), None)
        if metrics_csv is None:
            metrics_csv = next(iter(sorted(run_root.rglob("results.csv"))), None)
        map50_train = _read_map50(metrics_csv) if metrics_csv is not None else None
        candidate_active_eval = self._evaluate_dataset_metrics(
            repo=repo,
            runtime_env=runtime_env,
            run_root=run_root,
            run_name="active-candidate-after-train",
            weights_path=model_artifact,
            data_yaml=data_yaml,
            imgsz=imgsz,
            batch_size=batch_size,
            task="val",
            progress_callback=progress_callback,
            control_callback=control_callback,
            output_dir=output_dir,
        )
        candidate_canonical_eval = self._evaluate_dataset_metrics(
            repo=repo,
            runtime_env=runtime_env,
            run_root=run_root,
            run_name="canonical-candidate-after-train",
            weights_path=model_artifact,
            data_yaml=canonical_data_yaml,
            imgsz=_safe_int(canonical_spec.get("imgsz"), default=512, minimum=128, maximum=4096),
            batch_size=_safe_int(
                canonical_spec.get("batch_size"), default=16, minimum=1, maximum=256
            ),
            task=str(canonical_spec.get("val_task") or "test"),
            progress_callback=progress_callback,
            control_callback=control_callback,
            output_dir=output_dir,
        )
        candidate_after_train = {
            "active": candidate_active_eval,
            "canonical": candidate_canonical_eval,
            "model_artifact_path": str(model_artifact),
        }
        comparison = self._build_benchmark_comparison(
            baseline_before_train=baseline_before_train,
            candidate_after_train=candidate_after_train,
            thresholds=guardrail_thresholds,
        )
        benchmark_packet = {
            "ready": bool(
                _benchmark_metrics_complete(candidate_canonical_eval)
                and _benchmark_metrics_complete(candidate_active_eval)
                and bool(comparison.get("required_packet_complete"))
            ),
            "canonical_benchmark_ready": bool(
                _benchmark_metrics_complete(candidate_canonical_eval)
            ),
            "promotion_benchmark_ready": bool(
                _benchmark_metrics_complete(candidate_canonical_eval)
                and _benchmark_metrics_complete(candidate_active_eval)
                and bool(comparison.get("required_packet_complete"))
            ),
            "baseline_before_train": baseline_before_train,
            "candidate_after_train": candidate_after_train,
            "comparison": comparison,
            "last_benchmark_at": datetime.utcnow().isoformat(),
        }
        baseline_path = output_dir / "baseline_before_train.json"
        candidate_path = output_dir / "candidate_after_train.json"
        comparison_path = output_dir / "comparison.json"
        _write_json(baseline_path, baseline_before_train)
        _write_json(candidate_path, candidate_after_train)
        _write_json(comparison_path, benchmark_packet)

        clipped_fraction = (
            clipping_stats["clipped_area_pixels_sum"] / clipping_stats["original_area_pixels_sum"]
            if clipping_stats["original_area_pixels_sum"] > 0.0
            else 0.0
        )
        data_quality_report = {
            "reviewed_images": reviewed_count,
            "class_counts": class_counts,
            "unsupported_class_counts": unsupported_counts,
            "object_size_histogram": object_size_histogram,
            "tile_size": tile_size,
            "train_tile_overlap": train_tile_overlap,
            "empty_tile_ratio": empty_tile_ratio,
            "include_empty_tiles": include_empty_tiles,
            "min_box_pixels": min_box_pixels,
            "enable_hard_sample_bank": enable_hard_sample_bank,
            "hard_sample_injection_ratio": hard_sample_injection_ratio,
            "enable_small_object_weighting": enable_small_object_weighting,
            "replay_new_ratio": replay_new_ratio,
            "replay_old_ratio": replay_old_ratio,
            "replay_source_total": int(replay_source_total),
            "replay_tiles_injected": int(replay_tiles_injected),
            "replay_tile_fraction": round(
                float(replay_tiles_injected)
                / float(
                    max(1, int(tile_counts["train"]["tiles_selected"]) + int(replay_tiles_injected))
                ),
                6,
            ),
            "splits": tile_counts,
            "clipping": {
                **{key: round(float(value), 6) for key, value in clipping_stats.items()},
                "clipped_area_fraction": round(float(max(0.0, min(1.0, clipped_fraction))), 6),
            },
        }
        data_quality_path = output_dir / "data_quality_report.json"
        _write_json(data_quality_path, data_quality_report)
        replay_manifest_path = output_dir / "replay_manifest.json"
        _write_json(
            replay_manifest_path,
            {
                "replay_new_ratio": replay_new_ratio,
                "replay_old_ratio": replay_old_ratio,
                "replay_source_total": int(replay_source_total),
                "replay_tiles_injected": int(replay_tiles_injected),
                "items": replay_manifest_rows,
            },
        )

        map50_candidate = (
            float(candidate_active_eval["map50"])
            if isinstance(candidate_active_eval.get("map50"), (float, int))
            else float(map50_train)
            if isinstance(map50_train, (float, int))
            else None
        )
        map50_baseline = (
            float(baseline_active_eval["map50"])
            if isinstance(baseline_active_eval.get("map50"), (float, int))
            else None
        )
        map50_delta_vs_baseline = (
            round(map50_candidate - map50_baseline, 6)
            if isinstance(map50_candidate, float) and isinstance(map50_baseline, float)
            else None
        )
        metrics = {
            "map50": round(float(map50_candidate), 6)
            if isinstance(map50_candidate, float)
            else None,
            "map50_train": round(float(map50_train), 6)
            if isinstance(map50_train, (float, int))
            else None,
            "map50_baseline": round(float(map50_baseline), 6)
            if isinstance(map50_baseline, float)
            else None,
            "map50_candidate_eval": round(float(map50_candidate), 6)
            if isinstance(map50_candidate, float)
            else None,
            "map50_delta_vs_baseline": map50_delta_vs_baseline,
            "reviewed_images": reviewed_count,
            "class_counts": class_counts,
            "unsupported_class_counts": unsupported_counts,
            "tile_size": tile_size,
            "train_tile_overlap": train_tile_overlap,
            "include_empty_tiles": include_empty_tiles,
            "empty_tile_ratio": empty_tile_ratio,
            "min_box_pixels": min_box_pixels,
            "tile_counts": tile_counts,
            "data_quality_report_path": str(data_quality_path),
            "replay_manifest_path": str(replay_manifest_path),
            "replay_new_ratio": replay_new_ratio,
            "replay_old_ratio": replay_old_ratio,
            "replay_source_total": int(replay_source_total),
            "replay_tiles_injected": int(replay_tiles_injected),
            "benchmark": benchmark_packet,
            "benchmark_baseline": baseline_before_train,
            "benchmark_latest_candidate": candidate_after_train,
            "last_benchmark_at": str(benchmark_packet.get("last_benchmark_at") or ""),
            "benchmark_ready": bool(benchmark_packet.get("ready")),
            "canonical_benchmark_ready": bool(
                _benchmark_metrics_complete(candidate_canonical_eval)
            ),
            "promotion_benchmark_ready": bool(benchmark_packet.get("ready")),
            "enable_hard_sample_bank": enable_hard_sample_bank,
            "hard_sample_injection_ratio": hard_sample_injection_ratio,
            "enable_small_object_weighting": enable_small_object_weighting,
        }
        metrics_path = output_dir / "metrics.json"
        _write_json(
            metrics_path,
            {
                "framework": "yolov5",
                "execution_backend": "yolov5",
                "metrics": metrics,
                "stdout_tail": str(process_result.get("stdout_tail") or "").strip(),
                "stderr_tail": str(process_result.get("stderr_tail") or "").strip(),
                "stdout_log_path": str(process_result.get("stdout_log_path") or "").strip() or None,
                "stderr_log_path": str(process_result.get("stderr_log_path") or "").strip() or None,
                "baseline_eval": baseline_active_eval,
                "candidate_eval": candidate_active_eval,
                "baseline_before_train_path": str(baseline_path),
                "candidate_after_train_path": str(candidate_path),
                "comparison_path": str(comparison_path),
                "data_quality_report_path": str(data_quality_path),
            },
        )
        progress_callback(
            {
                "event": "training_complete",
                "model_artifact_path": str(model_artifact),
                "map50": metrics["map50"],
                "map50_baseline": metrics["map50_baseline"],
                "map50_delta_vs_baseline": metrics["map50_delta_vs_baseline"],
                "benchmark_ready": bool(metrics.get("benchmark_ready")),
            }
        )
        return {
            "framework": "yolov5",
            "execution_backend": "yolov5",
            "model_artifact_path": str(model_artifact),
            "metrics_path": str(metrics_path),
            "checkpoint_paths": [str(best_weights)],
            "metrics": metrics,
            "supported_classes": list(_YOLO_SUPPORTED_CLASSES),
            "unsupported_class_counts": unsupported_counts,
            "baseline_before_train_path": str(baseline_path),
            "candidate_after_train_path": str(candidate_path),
            "comparison_path": str(comparison_path),
            "data_quality_report_path": str(data_quality_path),
            "benchmark_ready": bool(metrics.get("benchmark_ready")),
            "stdout_log_path": str(process_result.get("stdout_log_path") or "").strip() or None,
            "stderr_log_path": str(process_result.get("stderr_log_path") or "").strip() or None,
        }

    def benchmark(
        self,
        *,
        model_artifact_path: str | None,
        output_dir: Path,
        config: dict[str, Any],
        progress_callback: ProgressCallback,
        control_callback: ControlCallback,
    ) -> dict[str, Any]:
        control_callback()
        output_dir.mkdir(parents=True, exist_ok=True)
        repo = self._require_runtime()
        runtime_env = os.environ.copy()
        config_dir = (output_dir / ".yolov5-config").resolve()
        _ensure_yolov5_font_asset(config_dir)
        runtime_env["YOLOV5_CONFIG_DIR"] = str(config_dir)
        runtime_env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        weights_path = (
            Path(model_artifact_path).expanduser().resolve()
            if model_artifact_path
            else _resolve_rarespot_weights_path()
        )
        if not weights_path.exists() or not weights_path.is_file():
            raise ValueError(f"Model artifact path not found for benchmark: {weights_path}")
        run_root = output_dir / "runs"
        run_root.mkdir(parents=True, exist_ok=True)
        benchmark_mode = str(config.get("benchmark_mode") or "canonical_only").strip().lower()
        if benchmark_mode not in {"canonical_only", "promotion_packet"}:
            benchmark_mode = "canonical_only"
        spec = self._load_canonical_benchmark_spec(config=config)
        canonical_data_yaml = self._prepare_canonical_data_yaml(spec=spec, output_dir=output_dir)
        benchmark_eval = self._evaluate_dataset_metrics(
            repo=repo,
            runtime_env=runtime_env,
            run_root=run_root,
            run_name="manual-canonical-benchmark",
            weights_path=weights_path,
            data_yaml=canonical_data_yaml,
            imgsz=_safe_int(spec.get("imgsz"), default=512, minimum=128, maximum=4096),
            batch_size=_safe_int(spec.get("batch_size"), default=16, minimum=1, maximum=256),
            task=str(spec.get("val_task") or "test"),
            progress_callback=progress_callback,
            control_callback=control_callback,
            output_dir=output_dir,
        )
        canonical_ready = bool(_benchmark_metrics_complete(benchmark_eval))
        promotion_ready = bool(canonical_ready)
        if benchmark_mode == "promotion_packet":
            promotion_ready = bool(canonical_ready and bool(config.get("_active_packet_ready")))
        report = {
            "mode": benchmark_mode,
            "benchmark_ready": bool(promotion_ready),
            "canonical_benchmark_ready": bool(canonical_ready),
            "promotion_benchmark_ready": bool(promotion_ready),
            "canonical": benchmark_eval,
            "weights_path": str(weights_path),
            "last_benchmark_at": datetime.utcnow().isoformat(),
        }
        report_path = output_dir / "manual_benchmark_report.json"
        _write_json(report_path, report)
        return {
            **report,
            "report_path": str(report_path),
        }

    def _infer_with_sahi(
        self,
        *,
        weights: Path,
        input_paths: list[str],
        output_dir: Path,
        tile_size: int,
        tile_overlap: float,
        conf: float,
        merge_iou: float,
        min_box_pixels: float,
        progress_callback: ProgressCallback,
        control_callback: ControlCallback,
    ) -> dict[str, Any]:
        if not _SAHI_AVAILABLE or AutoDetectionModel is None or get_sliced_prediction is None:
            raise ValueError("SAHI runtime is unavailable.")
        try:
            import torch  # type: ignore

            device = "cuda:0" if bool(torch.cuda.is_available()) else "cpu"
        except Exception:
            device = "cpu"

        model = AutoDetectionModel.from_pretrained(
            model_type="yolov5",
            model_path=str(weights),
            confidence_threshold=float(conf),
            device=device,
        )
        predictions: list[dict[str, Any]] = []
        class_counts: dict[str, int] = {}
        xml_paths: list[str] = []
        total_tile_count = 0
        for index, raw_path in enumerate(input_paths):
            control_callback()
            image_path = Path(raw_path).expanduser().resolve()
            if not image_path.exists() or image_path.suffix.lower() not in _YOLO_IMG_SUFFIXES:
                continue
            with Image.open(image_path) as image:
                width, height = image.size
            estimated_tiles = _build_sliding_tiles(
                width=width,
                height=height,
                tile_size=tile_size,
                overlap=tile_overlap,
            )
            total_tile_count += len(estimated_tiles)
            result = get_sliced_prediction(
                image=str(image_path),
                detection_model=model,
                slice_height=int(tile_size),
                slice_width=int(tile_size),
                overlap_height_ratio=float(tile_overlap),
                overlap_width_ratio=float(tile_overlap),
                postprocess_type="NMS",
                postprocess_match_metric="IOU",
                postprocess_match_threshold=float(merge_iou),
                perform_standard_pred=False,
                verbose=0,
            )
            merged_rows: list[dict[str, Any]] = []
            for prediction in list(result.object_prediction_list or []):
                bbox = getattr(prediction, "bbox", None)
                if bbox is None:
                    continue
                try:
                    x1, y1, x2, y2 = [float(value) for value in list(bbox.to_xyxy())[:4]]
                except Exception:
                    continue
                if (x2 - x1) < float(min_box_pixels) or (y2 - y1) < float(min_box_pixels):
                    continue
                category = getattr(prediction, "category", None)
                class_id = None
                class_name = None
                if category is not None:
                    if hasattr(category, "id"):
                        try:
                            class_id = int(category.id)
                        except Exception:
                            class_id = None
                    if hasattr(category, "name"):
                        class_name = str(category.name or "").strip() or None
                if class_name and class_name in _YOLO_CLASS_TO_ID:
                    class_id = _YOLO_CLASS_TO_ID[class_name]
                if class_id is None and class_name in _YOLO_CLASS_TO_ID:
                    class_id = _YOLO_CLASS_TO_ID[str(class_name)]
                if class_id is None:
                    continue
                if class_name is None:
                    class_name = (
                        _YOLO_SUPPORTED_CLASSES[class_id]
                        if 0 <= class_id < len(_YOLO_SUPPORTED_CLASSES)
                        else str(class_id)
                    )
                score = getattr(prediction, "score", None)
                confidence = None
                if score is not None and hasattr(score, "value"):
                    try:
                        confidence = float(score.value)
                    except Exception:
                        confidence = None
                merged_rows.append(
                    {
                        "class_id": int(class_id),
                        "class_name": str(class_name),
                        "confidence": confidence,
                        "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    }
                )
            merged_rows = _classwise_nms(merged_rows, iou_threshold=merge_iou)
            artifact_token = _slug_token(f"{image_path.stem}-{index:04d}")
            label_path = _write_yolo_prediction_labels(
                rows=merged_rows,
                width=int(width),
                height=int(height),
                output_path=output_dir / "prediction_labels" / f"{artifact_token}.txt",
            )
            xml_out = _write_prediction_xml(
                image_path=image_path,
                predictions=merged_rows,
                output_path=output_dir / "prediction_xml" / f"{artifact_token}.xml",
                layer_name="model_predictions",
            )
            xml_paths.append(str(xml_out))
            per_image_class_counts: dict[str, int] = {}
            for row in merged_rows:
                class_name = str(row.get("class_name") or "").strip()
                if not class_name:
                    continue
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                per_image_class_counts[class_name] = per_image_class_counts.get(class_name, 0) + 1
            predictions.append(
                {
                    "input_path": str(image_path),
                    "label_path": str(label_path),
                    "prediction_xml_path": str(xml_out),
                    "class_counts": per_image_class_counts,
                    "boxes": merged_rows,
                    "tile_count": len(estimated_tiles),
                }
            )
            progress_callback(
                {
                    "event": "inference_item_complete",
                    "input_path": str(image_path),
                    "detections": len(merged_rows),
                    "tile_count": len(estimated_tiles),
                    "inference_backend": "sahi",
                }
            )
        predictions_json = output_dir / "predictions.json"
        _write_json(
            predictions_json,
            {
                "framework": "yolov5",
                "model_artifact_path": str(weights),
                "counts_by_class": class_counts,
                "tile_size": tile_size,
                "tile_overlap": tile_overlap,
                "merge_iou": merge_iou,
                "min_box_pixels": min_box_pixels,
                "inference_backend": "sahi",
                "predictions": predictions,
            },
        )
        return {
            "framework": "yolov5",
            "execution_backend": "yolov5",
            "inference_backend": "sahi",
            "prediction_count": len(predictions),
            "predictions": predictions,
            "prediction_xml_paths": xml_paths,
            "predictions_json": str(predictions_json),
            "counts_by_class": class_counts,
            "tile_size": tile_size,
            "tile_overlap": tile_overlap,
            "merge_iou": merge_iou,
            "min_box_pixels": min_box_pixels,
            "tile_count": int(total_tile_count),
            "supported_classes": list(_YOLO_SUPPORTED_CLASSES),
        }

    def infer(
        self,
        *,
        model_artifact_path: str | None,
        input_paths: list[str],
        output_dir: Path,
        config: dict[str, Any],
        progress_callback: ProgressCallback,
        control_callback: ControlCallback,
    ) -> dict[str, Any]:
        control_callback()
        output_dir.mkdir(parents=True, exist_ok=True)
        repo = self._require_runtime()
        weights = (
            Path(model_artifact_path).expanduser().resolve()
            if model_artifact_path
            else _resolve_rarespot_weights_path()
        )
        if not weights.exists():
            raise ValueError(f"Model artifact path not found: {weights}")
        source_dir = output_dir / "inputs_tiled"
        source_dir.mkdir(parents=True, exist_ok=True)
        tile_size = _safe_int(config.get("tile_size"), default=512, minimum=128, maximum=4096)
        tile_overlap = _clamp_overlap(config.get("tile_overlap"), default=0.25)
        conf = _safe_float(config.get("conf"), 0.25)
        iou = _safe_float(config.get("iou"), 0.45)
        merge_iou = _clamp_overlap(config.get("merge_iou"), default=iou)
        imgsz = _safe_int(config.get("imgsz"), default=512, minimum=128, maximum=4096)
        min_box_pixels = max(1.0, _safe_float(config.get("min_box_pixels"), 2.0))
        prefer_sahi = _safe_bool(config.get("enable_sahi"), default=True)

        if prefer_sahi and _SAHI_AVAILABLE:
            try:
                progress_callback(
                    {
                        "event": "inference_backend_selected",
                        "inference_backend": "sahi",
                        "tile_size": tile_size,
                        "tile_overlap": tile_overlap,
                    }
                )
                return self._infer_with_sahi(
                    weights=weights,
                    input_paths=input_paths,
                    output_dir=output_dir,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    conf=conf,
                    merge_iou=merge_iou,
                    min_box_pixels=min_box_pixels,
                    progress_callback=progress_callback,
                    control_callback=control_callback,
                )
            except Exception as exc:
                progress_callback(
                    {
                        "event": "inference_backend_fallback",
                        "from_backend": "sahi",
                        "to_backend": "yolov5_detect_tiled",
                        "reason": str(exc),
                    }
                )

        prepared_inputs: list[dict[str, Any]] = []
        tile_manifest_by_stem: dict[str, dict[str, Any]] = {}
        for source_index, raw in enumerate(input_paths):
            path = Path(raw).expanduser().resolve()
            if not path.exists() or path.suffix.lower() not in _YOLO_IMG_SUFFIXES:
                continue
            with Image.open(path) as image:
                width, height = image.size
                rgb_image = image.convert("RGB")
                tiles = _build_sliding_tiles(
                    width=width,
                    height=height,
                    tile_size=tile_size,
                    overlap=tile_overlap,
                )
                prepared_inputs.append(
                    {
                        "input_path": path,
                        "width": int(width),
                        "height": int(height),
                        "tiles": len(tiles),
                        "rows": [],
                    }
                )
                prepared_index = len(prepared_inputs) - 1
                for tile in tiles:
                    control_callback()
                    stem = _slug_token(
                        f"{path.stem}-{source_index:04d}-"
                        f"x{int(tile['x0']):05d}-y{int(tile['y0']):05d}"
                    )
                    tile_path = source_dir / f"{stem}.jpg"
                    rgb_image.crop(
                        (
                            int(tile["x0"]),
                            int(tile["y0"]),
                            int(tile["x1"]),
                            int(tile["y1"]),
                        )
                    ).save(tile_path, format="JPEG", quality=95)
                    tile_manifest_by_stem[stem] = {
                        "prepared_index": prepared_index,
                        "x0": int(tile["x0"]),
                        "y0": int(tile["y0"]),
                        "width": int(tile["width"]),
                        "height": int(tile["height"]),
                    }

        if not prepared_inputs:
            raise ValueError("Inference requires at least one valid image input path.")

        command = [
            sys.executable,
            str(repo / "detect.py"),
            "--weights",
            str(weights),
            "--source",
            str(source_dir),
            "--imgsz",
            str(imgsz),
            "--project",
            str(output_dir),
            "--name",
            "predict",
            "--exist-ok",
            "--save-txt",
            "--save-conf",
            "--nosave",
            "--conf-thres",
            f"{conf}",
            "--iou-thres",
            f"{iou}",
        ]
        runtime_env = os.environ.copy()
        config_dir = (output_dir / ".yolov5-config").resolve()
        _ensure_yolov5_font_asset(config_dir)
        runtime_env["YOLOV5_CONFIG_DIR"] = str(config_dir)
        runtime_env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        progress_callback(
            {
                "event": "inference_started",
                "input_count": len(prepared_inputs),
                "tile_count": len(tile_manifest_by_stem),
                "tile_size": tile_size,
                "tile_overlap": tile_overlap,
                "imgsz": imgsz,
            }
        )
        process_result = _run_subprocess_with_heartbeat(
            command=command,
            cwd=repo,
            env=runtime_env,
            progress_callback=progress_callback,
            control_callback=control_callback,
            heartbeat_event="inference_heartbeat",
            heartbeat_phase="yolov5_detect_subprocess",
            stdout_log_path=output_dir / "infer.stdout.log",
            stderr_log_path=output_dir / "infer.stderr.log",
        )
        if int(process_result.get("returncode") or 0) != 0:
            stderr_tail = str(process_result.get("stderr_tail") or "").strip()
            stdout_tail = str(process_result.get("stdout_tail") or "").strip()
            raise ValueError(f"YOLOv5 inference failed: {stderr_tail or stdout_tail}")

        labels_dir = output_dir / "predict" / "labels"
        for stem, tile_meta in tile_manifest_by_stem.items():
            control_callback()
            txt_path = labels_dir / f"{stem}.txt"
            if not txt_path.exists():
                continue
            prepared_index = int(tile_meta["prepared_index"])
            source_item = prepared_inputs[prepared_index]
            for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parsed = _parse_yolo_line_to_xyxy(
                    line=line,
                    width=int(tile_meta["width"]),
                    height=int(tile_meta["height"]),
                    min_pixels=min_box_pixels,
                )
                if parsed is None:
                    continue
                local_xyxy = parsed["xyxy"]
                parsed["xyxy"] = [
                    float(local_xyxy[0]) + float(tile_meta["x0"]),
                    float(local_xyxy[1]) + float(tile_meta["y0"]),
                    float(local_xyxy[2]) + float(tile_meta["x0"]),
                    float(local_xyxy[3]) + float(tile_meta["y0"]),
                ]
                source_item["rows"].append(parsed)

        predictions: list[dict[str, Any]] = []
        class_counts: dict[str, int] = {}
        xml_paths: list[str] = []
        for prepared_index, source_item in enumerate(prepared_inputs):
            control_callback()
            image_path = Path(source_item["input_path"]).expanduser().resolve()
            width = int(source_item["width"])
            height = int(source_item["height"])
            artifact_token = _slug_token(f"{image_path.stem}-{prepared_index:04d}")
            merged_rows = _classwise_nms(
                list(source_item.get("rows") or []),
                iou_threshold=merge_iou,
            )
            label_path = _write_yolo_prediction_labels(
                rows=merged_rows,
                width=width,
                height=height,
                output_path=output_dir / "prediction_labels" / f"{artifact_token}.txt",
            )
            xml_out = _write_prediction_xml(
                image_path=image_path,
                predictions=merged_rows,
                output_path=output_dir / "prediction_xml" / f"{artifact_token}.xml",
                layer_name="model_predictions",
            )
            xml_paths.append(str(xml_out))
            per_image_class_counts: dict[str, int] = {}
            for row in merged_rows:
                class_name = str(row.get("class_name") or "").strip()
                if not class_name:
                    continue
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                per_image_class_counts[class_name] = per_image_class_counts.get(class_name, 0) + 1
            predictions.append(
                {
                    "input_path": str(image_path),
                    "label_path": str(label_path),
                    "prediction_xml_path": str(xml_out),
                    "class_counts": per_image_class_counts,
                    "boxes": merged_rows,
                    "tile_count": int(source_item["tiles"]),
                }
            )
            progress_callback(
                {
                    "event": "inference_item_complete",
                    "input_path": str(image_path),
                    "detections": len(merged_rows),
                    "tile_count": int(source_item["tiles"]),
                }
            )

        predictions_json = output_dir / "predictions.json"
        _write_json(
            predictions_json,
            {
                "framework": "yolov5",
                "model_artifact_path": str(weights),
                "counts_by_class": class_counts,
                "tile_size": tile_size,
                "tile_overlap": tile_overlap,
                "merge_iou": merge_iou,
                "min_box_pixels": min_box_pixels,
                "inference_backend": "yolov5_detect_tiled",
                "predictions": predictions,
            },
        )
        return {
            "framework": "yolov5",
            "execution_backend": "yolov5",
            "inference_backend": "yolov5_detect_tiled",
            "prediction_count": len(predictions),
            "predictions": predictions,
            "prediction_xml_paths": xml_paths,
            "predictions_json": str(predictions_json),
            "counts_by_class": class_counts,
            "tile_size": tile_size,
            "tile_overlap": tile_overlap,
            "merge_iou": merge_iou,
            "min_box_pixels": min_box_pixels,
            "tile_count": len(tile_manifest_by_stem),
            "supported_classes": list(_YOLO_SUPPORTED_CLASSES),
            "stdout_log_path": str(process_result.get("stdout_log_path") or "").strip() or None,
            "stderr_log_path": str(process_result.get("stderr_log_path") or "").strip() or None,
        }
