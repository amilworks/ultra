from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


VALID_DATASET_SPLITS = ("train", "val", "test")
VALID_DATASET_ROLES = ("image", "mask", "annotation")
VALID_SPATIAL_DIMS = ("2d", "3d")


class DatasetValidationError(ValueError):
    pass


def _normalize_split(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token not in VALID_DATASET_SPLITS:
        raise DatasetValidationError(
            f"Invalid split '{value}'. Expected one of: {', '.join(VALID_DATASET_SPLITS)}."
        )
    return token


def _normalize_role(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token not in VALID_DATASET_ROLES:
        raise DatasetValidationError(
            f"Invalid role '{value}'. Expected one of: {', '.join(VALID_DATASET_ROLES)}."
        )
    return token


def _normalize_sample_id(value: Any, fallback_file_id: str, role: str) -> str:
    raw = str(value or "").strip()
    if raw:
        return raw
    if role == "image":
        suffix = "img"
    elif role == "mask":
        suffix = "mask"
    else:
        suffix = "ann"
    return f"{fallback_file_id}:{suffix}"


def normalize_spatial_dims(value: Any, *, default: str = "2d") -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        raw = str(default or "2d").strip().lower()
    aliases = {
        "2": "2d",
        "2d": "2d",
        "2-d": "2d",
        "2dim": "2d",
        "2dims": "2d",
        "2dimension": "2d",
        "2dimensions": "2d",
        "3": "3d",
        "3d": "3d",
        "3-d": "3d",
        "3dim": "3d",
        "3dims": "3d",
        "3dimension": "3d",
        "3dimensions": "3d",
    }
    token = aliases.get(raw, raw)
    if token not in VALID_SPATIAL_DIMS:
        raise DatasetValidationError(
            f"Invalid spatial_dims '{value}'. Expected one of: {', '.join(VALID_SPATIAL_DIMS)}."
        )
    return token


def _normalized_suffix(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    if name.endswith(".ome.tiff"):
        return ".ome.tiff"
    if name.endswith(".ome.tif"):
        return ".ome.tif"
    return path.suffix.lower()


def _shape_from_numpy(path: Path) -> tuple[int, ...] | None:
    try:
        loaded = np.load(path, mmap_mode="r")  # type: ignore[call-overload]
        if isinstance(loaded, np.lib.npyio.NpzFile):
            first_key = next(iter(loaded.files), None)
            if first_key is None:
                return None
            shape = tuple(int(item) for item in np.asarray(loaded[first_key]).shape)
        else:
            shape = tuple(int(item) for item in np.asarray(loaded).shape)
        return shape
    except Exception:
        return None


def _shape_from_nibabel(path: Path) -> tuple[int, ...] | None:
    try:
        import nibabel as nib  # type: ignore

        image = nib.load(str(path))
        return tuple(int(item) for item in image.shape)
    except Exception:
        return None


def _shape_from_tifffile(path: Path) -> tuple[int, ...] | None:
    try:
        import tifffile  # type: ignore

        with tifffile.TiffFile(str(path)) as tif:
            if not tif.series:
                return None
            shape = tuple(int(item) for item in tif.series[0].shape)
            return shape
    except Exception:
        return None


def _shape_from_pillow(path: Path) -> tuple[int, ...] | None:
    try:
        with Image.open(path) as image:
            width, height = image.size
            return (int(height), int(width))
    except Exception:
        return None


def _spatial_ndim_from_shape(shape: tuple[int, ...] | None) -> int | None:
    if not shape:
        return None
    dims = [int(item) for item in shape if int(item) > 1]
    if not dims:
        return None
    if len(dims) <= 2:
        return 2
    if len(dims) == 3:
        small_dims = [item for item in dims if item <= 4]
        large_dims = [item for item in dims if item > 4]
        if len(small_dims) == 1 and len(large_dims) >= 2:
            return 2
        return 3
    return 3


def inspect_image_spatial_dims(path_value: str | Path) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve()
    if not path.exists() or not path.is_file():
        return {
            "spatial_dims": None,
            "spatial_ndim": None,
            "shape": None,
            "source": "missing",
            "path": str(path),
        }
    suffix = _normalized_suffix(path)
    shape: tuple[int, ...] | None = None
    source = "unknown"

    if suffix in {".npy", ".npz"}:
        shape = _shape_from_numpy(path)
        source = "numpy"
    elif suffix in {".nii", ".nii.gz"}:
        shape = _shape_from_nibabel(path)
        source = "nibabel"
    elif suffix in {".tif", ".tiff", ".ome.tif", ".ome.tiff"}:
        shape = _shape_from_tifffile(path)
        source = "tifffile"
        if shape is None:
            shape = _shape_from_pillow(path)
            source = "pillow-fallback"
    else:
        shape = _shape_from_pillow(path)
        source = "pillow"
        if shape is None and suffix in {".nrrd", ".mha", ".mhd"}:
            # Fallback heuristic when optional volume loaders are unavailable.
            return {
                "spatial_dims": "3d",
                "spatial_ndim": 3,
                "shape": None,
                "source": "extension-heuristic",
                "path": str(path),
            }

    spatial_ndim = _spatial_ndim_from_shape(shape)
    spatial_dims = None if spatial_ndim is None else f"{spatial_ndim}d"
    return {
        "spatial_dims": spatial_dims,
        "spatial_ndim": spatial_ndim,
        "shape": list(shape) if shape else None,
        "source": source,
        "path": str(path),
    }


def analyze_manifest_spatial_compatibility(
    *,
    manifest: dict[str, Any],
    required_spatial_dims: str,
    max_samples: int = 512,
) -> dict[str, Any]:
    expected_dims = normalize_spatial_dims(required_spatial_dims, default="2d")
    expected_ndim = 2 if expected_dims == "2d" else 3
    max_samples_clamped = max(1, int(max_samples))

    per_role_counts: dict[str, dict[str, int]] = {
        "image": {"2d": 0, "3d": 0, "unknown": 0},
        "mask": {"2d": 0, "3d": 0, "unknown": 0},
    }
    violations: list[dict[str, Any]] = []
    pair_mismatches: list[dict[str, Any]] = []
    inspected_samples = 0
    total_samples = 0

    splits = manifest.get("splits") if isinstance(manifest.get("splits"), dict) else {}
    for split in ("train", "val", "test"):
        rows = splits.get(split) if isinstance(splits, dict) else None
        if not isinstance(rows, list):
            continue
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            total_samples += 1
            if inspected_samples >= max_samples_clamped:
                continue
            inspected_samples += 1
            sample_id = str(raw.get("sample_id") or f"{split}-{inspected_samples}")
            image_row = raw.get("image") if isinstance(raw.get("image"), dict) else {}
            mask_row = raw.get("mask") if isinstance(raw.get("mask"), dict) else {}
            image_info = inspect_image_spatial_dims(str(image_row.get("path") or ""))
            mask_info = (
                inspect_image_spatial_dims(str(mask_row.get("path") or ""))
                if mask_row
                else {"spatial_dims": None, "spatial_ndim": None, "shape": None, "source": "missing"}
            )

            for role, info in (("image", image_info), ("mask", mask_info)):
                dims_token = str(info.get("spatial_dims") or "").strip().lower()
                if dims_token in {"2d", "3d"}:
                    per_role_counts[role][dims_token] += 1
                else:
                    per_role_counts[role]["unknown"] += 1
                spatial_ndim = info.get("spatial_ndim")
                if spatial_ndim is None or not isinstance(spatial_ndim, int):
                    continue
                if role == "mask" and not mask_row:
                    continue
                if spatial_ndim != expected_ndim:
                    if len(violations) < 25:
                        violations.append(
                            {
                                "split": split,
                                "sample_id": sample_id,
                                "role": role,
                                "required_spatial_dims": expected_dims,
                                "detected_spatial_dims": f"{spatial_ndim}d",
                                "path": str(info.get("path") or ""),
                            }
                        )

            image_ndim = image_info.get("spatial_ndim")
            mask_ndim = mask_info.get("spatial_ndim")
            if isinstance(image_ndim, int) and isinstance(mask_ndim, int) and image_ndim != mask_ndim:
                if len(pair_mismatches) < 25:
                    pair_mismatches.append(
                        {
                            "split": split,
                            "sample_id": sample_id,
                            "image_dims": f"{image_ndim}d",
                            "mask_dims": f"{mask_ndim}d",
                            "image_path": str(image_info.get("path") or ""),
                            "mask_path": str(mask_info.get("path") or ""),
                        }
                    )

    warnings: list[str] = []
    unknown_images = int(per_role_counts["image"]["unknown"])
    unknown_masks = int(per_role_counts["mask"]["unknown"])
    if unknown_images > 0 or unknown_masks > 0:
        warnings.append(
            "Some files could not be dimension-profiled automatically "
            f"(unknown images={unknown_images}, unknown masks={unknown_masks})."
        )
    if inspected_samples < total_samples:
        warnings.append(
            f"Dimension checks were sampled ({inspected_samples}/{total_samples} samples)."
        )

    return {
        "required_spatial_dims": expected_dims,
        "expected_spatial_ndim": expected_ndim,
        "inspected_samples": inspected_samples,
        "total_samples": total_samples,
        "per_role_counts": per_role_counts,
        "violations": violations,
        "pair_mismatches": pair_mismatches,
        "warnings": warnings,
    }


def build_dataset_manifest(
    *,
    dataset_id: str,
    dataset_name: str,
    items: list[dict[str, Any]],
    task_type: str = "segmentation",
) -> dict[str, Any]:
    mode = str(task_type or "segmentation").strip().lower()
    if mode not in {"segmentation", "detection"}:
        raise DatasetValidationError(
            f"Unsupported dataset task_type '{task_type}'. Expected segmentation or detection."
        )
    normalized_items: list[dict[str, Any]] = []
    pairs_by_split: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        split: defaultdict(dict) for split in VALID_DATASET_SPLITS
    }

    for index, raw in enumerate(items):
        split = _normalize_split(raw.get("split"))
        role = _normalize_role(raw.get("role"))
        file_id = str(raw.get("file_id") or "").strip()
        if not file_id:
            raise DatasetValidationError(f"Item #{index + 1} is missing file_id.")
        sample_id = _normalize_sample_id(raw.get("sample_id"), file_id, role)
        path = str(raw.get("path") or "").strip()
        if not path:
            raise DatasetValidationError(
                f"Item #{index + 1} ({file_id}) has no resolved local file path."
            )
        if mode == "detection" and role == "mask":
            raise DatasetValidationError(
                "Detection dataset items cannot include role='mask'. Use role='annotation' for XML labels."
            )
        normalized = {
            "file_id": file_id,
            "split": split,
            "role": role,
            "sample_id": sample_id,
            "path": path,
            "original_name": str(raw.get("original_name") or Path(path).name),
            "sha256": str(raw.get("sha256") or "").strip() or None,
            "size_bytes": int(raw.get("size_bytes") or 0),
            "metadata": raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {},
        }
        pair_bucket = pairs_by_split[split][sample_id]
        if role in pair_bucket:
            raise DatasetValidationError(
                f"Duplicate {role} assignment for split='{split}', sample_id='{sample_id}'."
            )
        pair_bucket[role] = normalized
        normalized_items.append(normalized)

    for required_split in ("train", "val"):
        if len(pairs_by_split[required_split]) == 0:
            raise DatasetValidationError(
                f"Dataset requires at least one '{required_split}' sample."
            )

    split_manifests: dict[str, list[dict[str, Any]]] = {split: [] for split in VALID_DATASET_SPLITS}
    split_counts = {
        split: {"samples": 0, "pairs": 0, "images": 0, "masks": 0, "annotations": 0}
        for split in VALID_DATASET_SPLITS
    }

    for split in VALID_DATASET_SPLITS:
        for sample_id, roles in sorted(pairs_by_split[split].items()):
            image = roles.get("image")
            mask = roles.get("mask")
            annotation = roles.get("annotation")
            if mode == "segmentation":
                if split in {"train", "val"} and (image is None or mask is None):
                    raise DatasetValidationError(
                        f"Missing image/mask pair for split='{split}', sample_id='{sample_id}'."
                    )
            else:
                if split in {"train", "val"} and (image is None or annotation is None):
                    raise DatasetValidationError(
                        f"Missing image/annotation pair for split='{split}', sample_id='{sample_id}'."
                    )
            if image is None:
                # Test split can be inference-only.
                continue
            entry = {
                "sample_id": sample_id,
                "image": image,
                "mask": mask,
                "annotation": annotation,
            }
            split_manifests[split].append(entry)
            split_counts[split]["samples"] += 1
            split_counts[split]["images"] += 1
            if mode == "segmentation" and mask is not None:
                split_counts[split]["masks"] += 1
                split_counts[split]["pairs"] += 1
            if mode == "detection" and annotation is not None:
                split_counts[split]["annotations"] += 1
                split_counts[split]["pairs"] += 1

    manifest = {
        "dataset_id": dataset_id,
        "name": dataset_name,
        "task_type": mode,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "splits": split_manifests,
        "counts": split_counts,
        "item_count": len(normalized_items),
    }
    return manifest
