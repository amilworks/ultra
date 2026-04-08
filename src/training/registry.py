from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelDefinition:
    key: str
    name: str
    framework: str
    description: str
    supports_training: bool
    supports_finetune: bool
    supports_inference: bool
    task_type: str = "segmentation"
    enabled: bool = True
    default_config: dict[str, Any] = field(default_factory=dict)
    dimensions: list[str] = field(default_factory=list)


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


_MODEL_REGISTRY: dict[str, ModelDefinition] = {
    "dynunet": ModelDefinition(
        key="dynunet",
        name="DynUNET (MONAI)",
        framework="monai",
        description=(
            "Dynamic U-Net style medical segmentation model with configurable "
            "2D/3D support."
        ),
        task_type="segmentation",
        supports_training=True,
        supports_finetune=True,
        supports_inference=True,
        enabled=_env_flag("TRAINING_ENABLE_DYNUNET", False),
        default_config={
            "epochs": 10,
            "learning_rate": 1e-3,
            "batch_size": 2,
            "spatial_dims": "2d",
            "num_classes": 2,
        },
        dimensions=["2d", "3d"],
    ),
    "medsam": ModelDefinition(
        key="medsam",
        name="MedSAM",
        framework="medsam",
        description=(
            "Segment Anything variant for medical imaging. v1 supports inference "
            "and finetuning-oriented updates."
        ),
        task_type="segmentation",
        supports_training=False,
        supports_finetune=False,
        supports_inference=True,
        enabled=_env_flag("TRAINING_ENABLE_MEDSAM", True),
        default_config={
            "finetune": False,
        },
        dimensions=["2d", "3d"],
    ),
    "yolov5_rarespot": ModelDefinition(
        key="yolov5_rarespot",
        name="YOLOv5 RareSpot",
        framework="yolov5",
        description=(
            "Prairie dog detection model for burrow and prairie_dog classes using "
            "RareSpotWeights.pt."
        ),
        task_type="detection",
        supports_training=True,
        supports_finetune=True,
        supports_inference=True,
        enabled=_env_flag("TRAINING_ENABLE_YOLOV5_RARESPOT", True),
        default_config={
            "epochs": 10,
            "imgsz": 512,
            "batch_size": 4,
            "conf": 0.25,
            "iou": 0.45,
            "tile_size": 512,
            "train_tile_overlap": 0.25,
            "tile_overlap": 0.25,
            "merge_iou": 0.45,
            "include_empty_tiles": True,
            "min_box_pixels": 4.0,
            "empty_tile_ratio": 1.0,
        },
        dimensions=["2d"],
    ),
}


def list_model_definitions(*, include_disabled: bool = False) -> list[ModelDefinition]:
    rows = list(_MODEL_REGISTRY.values())
    if include_disabled:
        return rows
    return [row for row in rows if row.enabled]


def get_model_definition(model_key: str) -> ModelDefinition | None:
    model = _MODEL_REGISTRY.get(str(model_key or "").strip().lower())
    if model is None:
        return None
    if not model.enabled:
        return None
    return model
