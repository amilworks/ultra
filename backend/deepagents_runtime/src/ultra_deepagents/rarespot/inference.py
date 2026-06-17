from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from ultra_deepagents.rarespot.artifacts import (
    artifact_record,
    render_overlay,
    render_stability_overlay,
    slug_token,
    write_detections_csv,
    write_json,
    write_prediction_xml,
    write_report,
)
from ultra_deepagents.rarespot.config import RareSpotConfig
from ultra_deepagents.rarespot.geospatial import build_geospatial_summary
from ultra_deepagents.rarespot.stability import run_detection_stability
from ultra_deepagents.rarespot.tiling import (
    build_sliding_tiles,
    classwise_nms,
    remap_tile_box,
    yolo_line_to_xyxy,
)

CLASS_NAMES = ["prairie_dog", "burrow"]
ProgressCallback = Callable[[dict[str, Any]], None]


def run_rarespot_inference(
    *,
    image_paths: list[Path],
    run_id: str,
    thread_id: str,
    output_dir: Path,
    config: RareSpotConfig,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    if not image_paths:
        raise ValueError("RareSpot inference requires at least one image.")
    if not config.weights_path.exists():
        raise FileNotFoundError(f"RareSpot weights not found: {config.weights_path}")
    if not (config.yolov5_path / "detect.py").exists():
        raise FileNotFoundError(f"YOLOv5 runtime not found: {config.yolov5_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "tiles"
    source_dir.mkdir(parents=True, exist_ok=True)
    tile_manifest: dict[str, dict[str, Any]] = {}
    prepared_inputs = prepare_tiles(
        image_paths=image_paths,
        source_dir=source_dir,
        config=config,
        tile_manifest=tile_manifest,
    )
    emit(progress_callback, {"event": "inference_started", "image_count": len(prepared_inputs), "tile_count": len(tile_manifest)})
    labels_dir = run_yolov5_detect(source_dir=source_dir, output_dir=output_dir, config=config)
    predictions = parse_tile_predictions(
        prepared_inputs=prepared_inputs,
        tile_manifest=tile_manifest,
        labels_dir=labels_dir,
        output_dir=output_dir,
        config=config,
    )
    stability_summary: dict[str, Any] | None = None
    if config.stability:
        emit(progress_callback, {"event": "stability_started"})
        stability_summary = run_detection_stability(
            predictions=predictions,
            source_dir=source_dir,
            output_dir=output_dir,
            tile_manifest=tile_manifest,
            detect_fn=lambda **kwargs: run_yolov5_detect(output_dir=output_dir, config=config, **kwargs),
            match_iou=config.stability_match_iou,
            iou_threshold=config.iou,
        )
        emit(progress_callback, {"event": "stability_completed", **((stability_summary or {}).get("label_counts") or {})})
    geospatial_summary = build_geospatial_summary(predictions=predictions, output_dir=output_dir)
    counts_by_class: dict[str, int] = {}
    for prediction in predictions:
        for class_name, count in (prediction.get("class_counts") or {}).items():
            counts_by_class[str(class_name)] = counts_by_class.get(str(class_name), 0) + int(count)
    confidence_summary = summarize_detection_confidences(predictions)

    overlays: list[Path] = []
    stability_overlays: list[Path] = []
    for index, prediction in enumerate(predictions):
        source = Path(str(prediction["input_path"]))
        boxes = list(prediction.get("boxes") or [])
        overlay_path = output_dir / "overlays" / f"{index:04d}-{slug_token(source.stem)}.png"
        overlays.append(render_overlay(source_path=source, boxes=boxes, output_path=overlay_path))
        if stability_summary:
            stability_overlay_path = output_dir / "overlays" / f"{index:04d}-{slug_token(source.stem)}-stability.png"
            stability_overlays.append(
                render_stability_overlay(source_path=source, boxes=boxes, output_path=stability_overlay_path)
            )

    configuration = {
        "tile_size": config.tile_size,
        "tile_overlap": config.tile_overlap,
        "stride": config.stride,
        "conf": config.conf,
        "iou": config.iou,
        "spectral": config.spectral,
    }
    prediction_payload = {
        "run_id": run_id,
        "model_path": str(config.weights_path),
        "class_names": CLASS_NAMES,
        "configuration": configuration,
        "counts_by_class": counts_by_class,
        "confidence_summary": confidence_summary,
        "predictions": predictions,
    }
    predictions_json = write_json(output_dir / "predictions.json", prediction_payload)
    detections_csv = write_detections_csv(output_dir / "detections.csv", predictions)
    report_payload = {
        "summary": {
            "image_count": len(predictions),
            "total_detections": sum(counts_by_class.values()),
            "tile_overlap": config.tile_overlap,
            "stride": config.stride,
        },
        "counts_by_class": counts_by_class,
        "confidence_summary": confidence_summary,
        "configuration": configuration,
        "stability": stability_summary or {},
        "geospatial": geospatial_summary or {},
        "spectral": {},
    }
    artifacts = [
        artifact_record(
            run_id=run_id,
            thread_id=thread_id,
            run_root=output_dir,
            path=predictions_json,
            kind="json",
            title="RareSpot predictions",
            mime_type="application/json",
            category="prediction",
        ),
        artifact_record(
            run_id=run_id,
            thread_id=thread_id,
            run_root=output_dir,
            path=detections_csv,
            kind="csv",
            title="RareSpot detections",
            mime_type="text/csv",
            category="prediction",
        ),
    ]
    for prediction in predictions:
        xml_path = Path(str(prediction["prediction_xml_path"]))
        artifacts.append(
            artifact_record(
                run_id=run_id,
                thread_id=thread_id,
                run_root=output_dir,
                path=xml_path,
                kind="xml",
                title=f"Prediction XML: {Path(str(prediction['input_path'])).name}",
                mime_type="application/xml",
                category="prediction",
            )
        )
    for overlay in overlays:
        artifacts.append(
            artifact_record(
                run_id=run_id,
                thread_id=thread_id,
                run_root=output_dir,
                path=overlay,
                kind="image",
                title=f"Overlay: {overlay.name}",
                mime_type="image/png",
                category="overlay",
            )
        )
    for stability_overlay in stability_overlays:
        artifacts.append(
            artifact_record(
                run_id=run_id,
                thread_id=thread_id,
                run_root=output_dir,
                path=stability_overlay,
                kind="image",
                title=f"Stability overlay: {stability_overlay.name}",
                mime_type="image/png",
                category="overlay",
            )
        )

    spectral_payload: dict[str, Any] | None = None
    if config.spectral:
        spectral_payload = run_spectral(image_paths=image_paths, output_dir=output_dir, config=config)
        report_payload["spectral"] = spectral_payload
        spectral_json = Path(str(spectral_payload.get("output_json") or output_dir / "spectral_scores.json"))
        if spectral_json.exists():
            artifacts.append(
                artifact_record(
                    run_id=run_id,
                    thread_id=thread_id,
                    run_root=output_dir,
                    path=spectral_json,
                    kind="json",
                    title="Spectral instability scores",
                    mime_type="application/json",
                    category="spectral",
                )
            )
        for item in spectral_payload.get("visualization_paths") or []:
            path = Path(str(item.get("path") or ""))
            if path.exists():
                artifacts.append(
                    artifact_record(
                        run_id=run_id,
                        thread_id=thread_id,
                        run_root=output_dir,
                        path=path,
                        kind="image",
                        title=str(item.get("title") or path.name),
                        mime_type="image/png",
                        category="spectral",
                    )
                )

    if geospatial_summary and geospatial_summary.get("map_path"):
        map_path = Path(str(geospatial_summary["map_path"]))
        if map_path.exists():
            artifacts.append(
                artifact_record(
                    run_id=run_id,
                    thread_id=thread_id,
                    run_root=output_dir,
                    path=map_path,
                    kind="image",
                    title="Survey detection map",
                    mime_type="image/png",
                    category="geospatial",
                )
            )

    report_md = write_report(output_dir / "report.md", report_payload)
    artifacts.append(
        artifact_record(
            run_id=run_id,
            thread_id=thread_id,
            run_root=output_dir,
            path=report_md,
            kind="report",
            title="RareSpot ecology inference report",
            mime_type="text/markdown",
            category="report",
        )
    )
    emit(progress_callback, {"event": "inference_completed", "detections": sum(counts_by_class.values())})
    top_candidates = []
    if spectral_payload:
        top_candidates = list((spectral_payload.get("summary") or {}).get("top_ranked") or [])
    return {
        "configuration": configuration,
        "counts_by_class": counts_by_class,
        "confidence_summary": confidence_summary,
        "predictions": predictions,
        "stability_summary": stability_summary,
        "geospatial_summary": geospatial_summary,
        "top_spectral_review_candidates": top_candidates,
        "spectral": spectral_payload,
        "artifacts": artifacts,
        "output_dir": str(output_dir),
    }


def prepare_tiles(
    *,
    image_paths: list[Path],
    source_dir: Path,
    config: RareSpotConfig,
    tile_manifest: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    from PIL import Image

    prepared: list[dict[str, Any]] = []
    for source_index, image_path in enumerate(image_paths):
        with Image.open(image_path) as image:
            width, height = image.size
            rgb = image.convert("RGB")
            tiles = build_sliding_tiles(width=width, height=height, tile_size=config.tile_size, overlap=config.tile_overlap)
            item = {"input_path": str(image_path), "width": width, "height": height, "tile_count": len(tiles.tiles), "rows": []}
            prepared_index = len(prepared)
            prepared.append(item)
            for tile in tiles.tiles:
                stem = slug_token(f"{image_path.stem}-{source_index:04d}-x{tile.x0:05d}-y{tile.y0:05d}")
                tile_path = source_dir / f"{stem}.jpg"
                rgb.crop((tile.x0, tile.y0, tile.x1, tile.y1)).save(tile_path, format="JPEG", quality=95)
                tile_manifest[stem] = {
                    "prepared_index": prepared_index,
                    "x0": tile.x0,
                    "y0": tile.y0,
                    "width": tile.width,
                    "height": tile.height,
                    "image_width": width,
                    "image_height": height,
                }
    return prepared


def run_yolov5_detect(
    *,
    source_dir: Path,
    output_dir: Path,
    config: RareSpotConfig,
    project_subdir: str = "yolov5",
    name: str = "predict",
) -> Path:
    """Run YOLOv5 detect.py over a tile directory; returns the labels output dir.
    `project_subdir` lets the main and stability passes write to distinct trees."""
    env = os.environ.copy()
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    env["YOLOV5_CONFIG_DIR"] = str((output_dir / ".yolov5-config").resolve())
    env["PYTHONPATH"] = f"{config.yolov5_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
    command = [
        sys.executable,
        str(config.yolov5_path / "detect.py"),
        "--weights",
        str(config.weights_path),
        "--source",
        str(source_dir),
        "--imgsz",
        str(config.tile_size),
        "--project",
        str(output_dir / project_subdir),
        "--name",
        name,
        "--exist-ok",
        "--save-txt",
        "--save-conf",
        "--nosave",
        "--conf-thres",
        str(config.conf),
        "--iou-thres",
        str(config.iou),
    ]
    result = subprocess.run(
        command,
        cwd=str(config.yolov5_path),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    (output_dir / f"{project_subdir}.stdout.log").write_text(result.stdout or "", encoding="utf-8")
    (output_dir / f"{project_subdir}.stderr.log").write_text(result.stderr or "", encoding="utf-8")
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip()[-4000:]
        raise RuntimeError(f"YOLOv5 inference failed: {tail}")
    return output_dir / project_subdir / name / "labels"


def parse_tile_predictions(
    *,
    prepared_inputs: list[dict[str, Any]],
    tile_manifest: dict[str, dict[str, Any]],
    labels_dir: Path,
    output_dir: Path,
    config: RareSpotConfig,
) -> list[dict[str, Any]]:
    for stem, tile_meta in tile_manifest.items():
        txt_path = labels_dir / f"{stem}.txt"
        if not txt_path.exists():
            continue
        source_item = prepared_inputs[int(tile_meta["prepared_index"])]
        for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parsed = yolo_line_to_xyxy(
                line=line,
                width=int(tile_meta["width"]),
                height=int(tile_meta["height"]),
                class_names=CLASS_NAMES,
            )
            if parsed is None:
                continue
            parsed["xyxy"] = remap_tile_box(
                parsed["xyxy"],
                tile_x0=int(tile_meta["x0"]),
                tile_y0=int(tile_meta["y0"]),
                image_width=int(tile_meta["image_width"]),
                image_height=int(tile_meta["image_height"]),
            )
            source_item.setdefault("rows", []).append(parsed)

    predictions: list[dict[str, Any]] = []
    for index, source_item in enumerate(prepared_inputs):
        image_path = Path(str(source_item["input_path"]))
        rows = classwise_nms(list(source_item.get("rows") or []), iou_threshold=config.iou)
        token = slug_token(f"{image_path.stem}-{index:04d}")
        xml_path = write_prediction_xml(
            image_path=image_path,
            predictions=rows,
            output_path=output_dir / "prediction_xml" / f"{token}.xml",
        )
        class_counts: dict[str, int] = {}
        for row in rows:
            class_name = str(row.get("class_name") or "")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        predictions.append(
            {
                "input_path": str(image_path),
                "prediction_xml_path": str(xml_path),
                "class_counts": class_counts,
                "boxes": rows,
                "tile_count": source_item.get("tile_count", 0),
            }
        )
    return predictions


def summarize_detection_confidences(predictions: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    values_by_class: dict[str, list[float]] = {}
    for prediction in predictions:
        for box in prediction.get("boxes") or []:
            class_name = str(box.get("class_name") or "").strip()
            if not class_name:
                continue
            try:
                confidence = float(box.get("confidence"))
            except (TypeError, ValueError):
                continue
            values_by_class.setdefault(class_name, []).append(confidence)
    summary: dict[str, dict[str, float | int]] = {}
    for class_name in sorted(values_by_class):
        values = values_by_class[class_name]
        if not values:
            continue
        summary[class_name] = {
            "count": len(values),
            "min": round(min(values), 6),
            "mean": round(sum(values) / len(values), 6),
            "max": round(max(values), 6),
        }
    return summary


def run_spectral(*, image_paths: list[Path], output_dir: Path, config: RareSpotConfig) -> dict[str, Any]:
    from ultra_deepagents.rarespot.spectral import (
        SpectralInstabilityConfig,
        score_spectral_instability,
    )

    spectral_config = SpectralInstabilityConfig(
        imgsz=int(config.tile_size),
        conf_thres=float(config.conf),
        iou_thres=float(config.iou),
    )
    return score_spectral_instability(
        image_paths=[str(path) for path in image_paths],
        weights_path=str(config.weights_path),
        yolov5_path=str(config.yolov5_path),
        output_dir=output_dir / "spectral",
        config=spectral_config,
    )


def emit(callback: ProgressCallback | None, payload: dict[str, Any]) -> None:
    if callback is not None:
        callback(payload)
