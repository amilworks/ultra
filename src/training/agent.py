from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from .dataset import (
    DatasetValidationError,
    analyze_manifest_spatial_compatibility,
    normalize_spatial_dims,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _yolov5_font_available() -> bool:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    ]
    try:
        from matplotlib import font_manager  # type: ignore

        candidates.insert(0, Path(str(font_manager.findfont("DejaVu Sans"))).expanduser())
    except Exception:
        pass
    return any(path.exists() and path.is_file() for path in candidates)


def _detection_annotation_profile(
    dataset_manifest: dict[str, Any],
    *,
    supported_classes: list[str],
    layer_name: str = "gt2",
) -> dict[str, Any]:
    class_counts: dict[str, int] = {name: 0 for name in supported_classes}
    unsupported_class_counts: dict[str, int] = {}
    parse_errors: list[str] = []
    reviewed_samples = 0
    total_samples = 0

    splits = (
        dataset_manifest.get("splits") if isinstance(dataset_manifest.get("splits"), dict) else {}
    )
    for split in ("train", "val", "test"):
        rows = splits.get(split) if isinstance(splits, dict) else None
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            total_samples += 1
            annotation_row = (
                row.get("annotation") if isinstance(row.get("annotation"), dict) else {}
            )
            annotation_path = Path(str(annotation_row.get("path") or "")).expanduser()
            if not annotation_path.exists():
                continue
            reviewed_samples += 1
            try:
                root = ET.parse(str(annotation_path)).getroot()
            except Exception as exc:
                parse_errors.append(f"{annotation_path.name}: {exc}")
                continue
            top_layers = list(root.findall("./gobject"))
            selected_layers = [
                layer
                for layer in top_layers
                if str(layer.attrib.get("name") or "").strip() == layer_name
            ] or top_layers[:1]
            for layer in selected_layers:
                for class_node in layer.findall("./gobject"):
                    class_name = str(class_node.attrib.get("name") or "").strip()
                    if not class_name:
                        continue
                    rectangles = class_node.findall("./rectangle")
                    if class_name in class_counts:
                        class_counts[class_name] += len(rectangles)
                    else:
                        unsupported_class_counts[class_name] = unsupported_class_counts.get(
                            class_name, 0
                        ) + len(rectangles)

    return {
        "class_counts": class_counts,
        "unsupported_class_counts": unsupported_class_counts,
        "parse_errors": parse_errors[:50],
        "reviewed_samples": reviewed_samples,
        "total_samples": total_samples,
    }


def build_preflight_report(
    *,
    model_key: str,
    dataset_manifest: dict[str, Any],
    config: dict[str, Any],
    artifact_root: Path,
    model_dimensions: list[str] | None = None,
    max_dimension_samples: int = 512,
) -> dict[str, Any]:
    model_token = str(model_key or "").strip().lower()
    epochs = max(1, int(config.get("epochs") or 10))
    train_samples = int(
        ((dataset_manifest.get("counts") or {}).get("train") or {}).get("samples") or 0
    )
    val_samples = int(((dataset_manifest.get("counts") or {}).get("val") or {}).get("samples") or 0)
    disk_usage = shutil.disk_usage(str(artifact_root))
    free_gb = disk_usage.free / float(1024**3)

    gpu_summary: dict[str, Any] = {"available": False, "count": 0, "devices": []}
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            gpu_summary["available"] = True
            gpu_summary["count"] = int(torch.cuda.device_count())
            devices: list[dict[str, Any]] = []
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                total_gb = float(props.total_memory) / float(1024**3)
                free_gb_device = None
                try:
                    free_mem, total_mem = torch.cuda.mem_get_info(idx)
                    free_gb_device = float(free_mem) / float(1024**3)
                    total_gb = float(total_mem) / float(1024**3)
                except Exception:
                    pass
                devices.append(
                    {
                        "index": idx,
                        "name": str(props.name),
                        "total_memory_gb": round(total_gb, 3),
                        "free_memory_gb": round(free_gb_device, 3)
                        if free_gb_device is not None
                        else None,
                    }
                )
            gpu_summary["devices"] = devices
    except Exception:
        gpu_summary = {"available": False, "count": 0, "devices": []}

    monai_available = False
    try:
        import monai  # type: ignore

        del monai
        monai_available = True
    except Exception:
        monai_available = False

    requested_backend = str(config.get("execution_backend") or "auto").strip().lower() or "auto"
    if requested_backend not in {"auto", "simulated", "monai"}:
        requested_backend = "auto"

    spatial_dims_error: str | None = None
    requested_spatial_dims = "2d"
    try:
        requested_spatial_dims = normalize_spatial_dims(
            config.get("spatial_dims"),
            default="2d",
        )
    except DatasetValidationError as exc:
        spatial_dims_error = str(exc)

    spatial_profile: dict[str, Any] = {}
    if spatial_dims_error is None:
        try:
            spatial_profile = analyze_manifest_spatial_compatibility(
                manifest=dataset_manifest,
                required_spatial_dims=requested_spatial_dims,
                max_samples=max_dimension_samples,
            )
        except Exception as exc:
            spatial_profile = {
                "required_spatial_dims": requested_spatial_dims,
                "inspected_samples": 0,
                "total_samples": 0,
                "violations": [],
                "pair_mismatches": [],
                "warnings": [f"Failed to profile dataset dimensionality: {exc}"],
            }

    estimated_minutes = max(
        1.0,
        ((epochs * max(1, train_samples)) / (8.0 if gpu_summary["available"] else 2.0)),
    )

    checks = [
        {
            "name": "manifest_train_split",
            "status": "pass" if train_samples > 0 else "fail",
            "detail": f"train samples={train_samples}",
        },
        {
            "name": "manifest_val_split",
            "status": "pass" if val_samples > 0 else "fail",
            "detail": f"val samples={val_samples}",
        },
        {
            "name": "storage_free",
            "status": "pass" if free_gb >= 2.0 else "warn",
            "detail": f"free storage={free_gb:.2f} GB",
        },
        {
            "name": "gpu_available",
            "status": "pass" if gpu_summary["available"] else "warn",
            "detail": (
                f"gpu count={gpu_summary['count']}"
                if gpu_summary["available"]
                else "GPU not detected; training will run on CPU."
            ),
        },
    ]

    if spatial_dims_error:
        checks.append(
            {
                "name": "config_spatial_dims",
                "status": "fail",
                "detail": spatial_dims_error,
            }
        )
    else:
        checks.append(
            {
                "name": "config_spatial_dims",
                "status": "pass",
                "detail": f"requested spatial_dims={requested_spatial_dims}",
            }
        )

    if model_dimensions:
        dims = [
            str(item or "").strip().lower() for item in model_dimensions if str(item or "").strip()
        ]
        if requested_spatial_dims not in dims:
            checks.append(
                {
                    "name": "model_dimension_support",
                    "status": "fail",
                    "detail": (
                        f"Model '{model_token}' supports dimensions {dims}; "
                        f"requested {requested_spatial_dims}."
                    ),
                }
            )
        else:
            checks.append(
                {
                    "name": "model_dimension_support",
                    "status": "pass",
                    "detail": f"Model supports requested {requested_spatial_dims} mode.",
                }
            )

    if spatial_profile:
        violations = spatial_profile.get("violations")
        pair_mismatches = spatial_profile.get("pair_mismatches")
        warnings = spatial_profile.get("warnings")
        violation_count = len(violations) if isinstance(violations, list) else 0
        pair_mismatch_count = len(pair_mismatches) if isinstance(pair_mismatches, list) else 0
        warning_count = len(warnings) if isinstance(warnings, list) else 0
        if violation_count > 0:
            checks.append(
                {
                    "name": "dataset_spatial_dims_compatible",
                    "status": "fail",
                    "detail": (
                        f"{violation_count} sample(s) do not match requested "
                        f"{requested_spatial_dims} dimensionality."
                    ),
                }
            )
        else:
            checks.append(
                {
                    "name": "dataset_spatial_dims_compatible",
                    "status": "pass",
                    "detail": f"All checked samples match requested {requested_spatial_dims}.",
                }
            )
        if pair_mismatch_count > 0:
            checks.append(
                {
                    "name": "dataset_image_mask_dim_match",
                    "status": "fail",
                    "detail": (
                        f"{pair_mismatch_count} sample(s) have image/mask dimensionality mismatch."
                    ),
                }
            )
        else:
            checks.append(
                {
                    "name": "dataset_image_mask_dim_match",
                    "status": "pass",
                    "detail": "Checked image/mask pairs are dimensionally aligned.",
                }
            )
        if warning_count > 0:
            checks.append(
                {
                    "name": "dataset_dimension_profile_coverage",
                    "status": "warn",
                    "detail": "; ".join(str(item) for item in warnings[:3]),
                }
            )

    if requested_backend == "monai" and not monai_available:
        checks.append(
            {
                "name": "monai_backend_available",
                "status": "fail",
                "detail": "execution_backend=monai requested, but MONAI is not installed.",
            }
        )
    else:
        checks.append(
            {
                "name": "monai_backend_available",
                "status": "pass" if monai_available else "warn",
                "detail": (
                    "MONAI is available."
                    if monai_available
                    else "MONAI not installed; adapter will use simulated fallback."
                ),
            }
        )

    if model_token == "dynunet" and not gpu_summary["available"]:
        checks.append(
            {
                "name": "dynunet_cpu_fallback",
                "status": "warn",
                "detail": "DynUNET will run in CPU fallback mode; expect slower epochs.",
            }
        )

    if model_token == "yolov5_rarespot":
        profile = _detection_annotation_profile(
            dataset_manifest=dataset_manifest,
            supported_classes=["prairie_dog", "burrow"],
            layer_name="gt2",
        )
        reviewed_samples = int(profile.get("reviewed_samples") or 0)
        class_counts = (
            profile.get("class_counts") if isinstance(profile.get("class_counts"), dict) else {}
        )
        unsupported = (
            profile.get("unsupported_class_counts")
            if isinstance(profile.get("unsupported_class_counts"), dict)
            else {}
        )
        parse_errors = (
            profile.get("parse_errors") if isinstance(profile.get("parse_errors"), list) else []
        )
        repo_path = (
            Path(str(config.get("runtime_repo_path") or "third_party/yolov5"))
            .expanduser()
            .resolve()
        )
        weights_path = (
            Path(str(config.get("weights_path") or "RareSpotWeights.pt")).expanduser().resolve()
        )
        checks.append(
            {
                "name": "yolov5_runtime_available",
                "status": (
                    "pass"
                    if (repo_path / "detect.py").exists()
                    and (repo_path / "train.py").exists()
                    and (repo_path / "val.py").exists()
                    else "fail"
                ),
                "detail": f"runtime_repo_path={repo_path}",
            }
        )
        checks.append(
            {
                "name": "rarespot_weights_available",
                "status": "pass" if weights_path.exists() else "fail",
                "detail": f"weights_path={weights_path}",
            }
        )
        has_font = _yolov5_font_available()
        checks.append(
            {
                "name": "yolov5_font_asset_available",
                "status": "pass" if has_font else "fail",
                "detail": (
                    "Found local TTF font for YOLOv5 plot/font bootstrap."
                    if has_font
                    else "No compatible local TTF font found. Install DejaVu Sans or Arial."
                ),
            }
        )
        for check_name, module_name, install_hint in (
            ("setuptools", "pkg_resources", "setuptools"),
            ("tensorboard", "tensorboard", "tensorboard"),
            ("thop", "thop", "thop>=0.1.1"),
            ("seaborn", "seaborn", "seaborn"),
        ):
            available = True
            if check_name == "setuptools":
                available = importlib.util.find_spec(module_name) is not None
            else:
                available = importlib.util.find_spec(module_name) is not None
            checks.append(
                {
                    "name": f"yolov5_dependency_{check_name}",
                    "status": "pass" if available else "fail",
                    "detail": (
                        f"python module '{module_name}' available."
                        if available
                        else f"Missing python module '{module_name}'. Install {install_hint}."
                    ),
                }
            )
        checks.append(
            {
                "name": "reviewed_annotation_coverage",
                "status": "pass" if reviewed_samples >= 1 else "fail",
                "detail": f"reviewed samples={reviewed_samples}",
            }
        )
        for cls in ("prairie_dog", "burrow"):
            checks.append(
                {
                    "name": f"class_coverage_{cls}",
                    "status": "pass" if int(class_counts.get(cls) or 0) >= 1 else "fail",
                    "detail": f"{cls} boxes={int(class_counts.get(cls) or 0)}",
                }
            )
        if unsupported:
            checks.append(
                {
                    "name": "unsupported_class_observations",
                    "status": "warn",
                    "detail": ", ".join(
                        [f"{key}={int(value)}" for key, value in sorted(unsupported.items())]
                    ),
                }
            )
        if parse_errors:
            checks.append(
                {
                    "name": "annotation_parse_errors",
                    "status": "fail",
                    "detail": "; ".join([str(item) for item in parse_errors[:3]]),
                }
            )

    return {
        "model_key": model_token,
        "checks": checks,
        "gpu": gpu_summary,
        "runtime": {
            "execution_backend_requested": requested_backend,
            "monai_available": monai_available,
        },
        "disk": {
            "total_gb": round(_safe_float(disk_usage.total) / float(1024**3), 3),
            "used_gb": round(_safe_float(disk_usage.used) / float(1024**3), 3),
            "free_gb": round(free_gb, 3),
        },
        "estimate": {
            "epochs": epochs,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "estimated_runtime_minutes": round(estimated_minutes, 2),
        },
        "spatial_profile": spatial_profile,
        "detection_profile": _detection_annotation_profile(
            dataset_manifest=dataset_manifest,
            supported_classes=["prairie_dog", "burrow"],
            layer_name="gt2",
        )
        if model_token == "yolov5_rarespot"
        else {},
        "recommended_launch": all(check["status"] != "fail" for check in checks),
    }
