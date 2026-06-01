from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


CLASS_COLORS = {
    "prairie_dog": (220, 38, 38),
    "burrow": (37, 99, 235),
}


def slug_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip())
    token = re.sub(r"-+", "-", token).strip("-._")
    return token or "artifact"


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def write_detections_csv(path: Path, predictions: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image_path",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "tile_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for prediction in predictions:
            for box in prediction.get("boxes") or []:
                xyxy = box.get("xyxy") or [0, 0, 0, 0]
                writer.writerow(
                    {
                        "image_path": prediction.get("input_path"),
                        "class_id": box.get("class_id"),
                        "class_name": box.get("class_name"),
                        "confidence": box.get("confidence"),
                        "x1": xyxy[0],
                        "y1": xyxy[1],
                        "x2": xyxy[2],
                        "y2": xyxy[3],
                        "tile_count": prediction.get("tile_count"),
                    }
                )
    return path


def write_prediction_xml(
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
        class_name = str(row.get("class_name") or "").strip()
        if class_name:
            grouped.setdefault(class_name, []).append(row)
    for class_name, rows in grouped.items():
        class_node = ET.SubElement(layer, "gobject", name=class_name)
        for row in rows:
            rect = ET.SubElement(class_node, "rectangle")
            x1, y1, x2, y2 = [float(value) for value in row.get("xyxy") or [0, 0, 0, 0]]
            ET.SubElement(rect, "vertex", index="0", x=f"{x1:.3f}", y=f"{y1:.3f}", z="0.0", t="0.0")
            ET.SubElement(rect, "vertex", index="1", x=f"{x2:.3f}", y=f"{y2:.3f}", z="0.0", t="0.0")
            if isinstance(row.get("confidence"), (float, int)):
                ET.SubElement(rect, "tag", name="confidence", value=f"{float(row['confidence']):.6f}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    return output_path


def render_overlay(*, source_path: Path, boxes: list[dict[str, Any]], output_path: Path) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        rgb = image.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    try:
        font = ImageFont.truetype("Arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for box in boxes:
        xyxy = box.get("xyxy") if isinstance(box.get("xyxy"), list) else []
        if len(xyxy) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in xyxy]
        class_name = str(box.get("class_name") or "det")
        color = CLASS_COLORS.get(class_name, (5, 5, 5))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        confidence = box.get("confidence")
        label = class_name
        if isinstance(confidence, (float, int)):
            label = f"{label} {float(confidence):.2f}"
        label_bbox = draw.textbbox((x1, max(0, y1 - 18)), label, font=font)
        draw.rectangle(label_bbox, fill=(5, 5, 5))
        draw.text((x1, max(0, y1 - 18)), label, fill=color, font=font)
    rgb.save(output_path, format="PNG")
    return output_path


def write_report(path: Path, payload: dict[str, Any]) -> Path:
    summary = payload.get("summary") or {}
    counts = payload.get("counts_by_class") or {}
    confidence_summary = payload.get("confidence_summary") or {}
    configuration = payload.get("configuration") or {}
    spectral = payload.get("spectral") or {}
    top_candidates = (spectral.get("summary") or {}).get("top_ranked") or []
    lines = [
        "# RareSpot Ecology Inference Report",
        "",
        f"- Images processed: {summary.get('image_count', 0)}",
        f"- Total detections: {summary.get('total_detections', 0)}",
        f"- Prairie dogs: {counts.get('prairie_dog', 0)}",
        f"- Burrows: {counts.get('burrow', 0)}",
        f"- Tile overlap: {summary.get('tile_overlap', 0.25):.2f}",
        f"- Tile stride: {summary.get('stride', '')}",
        f"- Confidence threshold: {configuration.get('conf', '')}",
        f"- IoU threshold: {configuration.get('iou', '')}",
        "",
        "## Confidence Summary",
    ]
    if confidence_summary:
        lines.extend(["", "| Class | Count | Mean | Min | Max |", "|---|---:|---:|---:|---:|"])
        for class_name, stats in sorted(confidence_summary.items()):
            lines.append(
                f"| {class_name} | {stats.get('count', 0)} | "
                f"{float(stats.get('mean') or 0.0):.3f} | "
                f"{float(stats.get('min') or 0.0):.3f} | "
                f"{float(stats.get('max') or 0.0):.3f} |"
            )
    else:
        lines.append("- No confidence values were reported.")
    lines.extend(
        [
            "",
            "## Spectral Review Candidates",
        ]
    )
    if top_candidates:
        for candidate in top_candidates[:10]:
            lines.append(f"- {candidate.get('file_name')}: {float(candidate.get('score') or 0.0):.3f}")
    else:
        lines.append("- No high-instability candidates were reported.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def artifact_record(
    *,
    run_id: str,
    thread_id: str,
    run_root: Path,
    path: Path,
    kind: str,
    title: str,
    mime_type: str,
    category: str,
    preview_path: Path | None = None,
) -> dict[str, Any]:
    size = path.stat().st_size if path.exists() else 0
    sha = ""
    if path.exists() and path.is_file():
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
    relative = path.relative_to(run_root).as_posix()
    record = {
        "run_id": run_id,
        "thread_id": thread_id,
        "kind": kind,
        "path": relative,
        "source_path": str(path),
        "title": title,
        "mime_type": mime_type,
        "size_bytes": size,
        "sha256": sha,
        "storage_uri": path.as_uri(),
        "tool_name": "rarespot_ecology_inference",
        "category": category,
        "metadata": {},
    }
    if preview_path is not None:
        record["preview_path"] = preview_path.relative_to(run_root).as_posix()
    return record
