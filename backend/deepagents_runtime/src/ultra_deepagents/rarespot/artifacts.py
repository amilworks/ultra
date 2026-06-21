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


STABILITY_COLORS = {
    "trusted": (22, 163, 74),     # green  — survives nearly every perturbation
    "borderline": (217, 119, 6),  # amber  — survives about half
    "unstable": (220, 38, 38),    # red    — flickers; likely a false positive, review it
}


def render_stability_overlay(*, source_path: Path, boxes: list[dict[str, Any]], output_path: Path) -> Path:
    """Annotated overlay coloured by per-detection STABILITY (green/amber/red) rather than
    class, so an ecologist can see at a glance which detections to trust vs review. Labels
    show class, confidence, and stability %."""
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
        color = STABILITY_COLORS.get(str(box.get("stability_label") or ""), (107, 114, 128))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        parts = [str(box.get("class_name") or "det")]
        confidence = box.get("confidence")
        if isinstance(confidence, (float, int)):
            parts.append(f"c{float(confidence):.2f}")
        stability = box.get("stability")
        if isinstance(stability, (float, int)):
            parts.append(f"s{float(stability):.0%}")
        label = " ".join(parts)
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

    stability = payload.get("stability") or {}
    label_counts = stability.get("label_counts") or {}
    lines.extend(["", "## Reliability & trust"])
    if label_counts:
        trials = int(stability.get("trials") or 0)
        perturbations = ", ".join(stability.get("perturbations") or [])
        trusted = int(label_counts.get("trusted", 0))
        borderline = int(label_counts.get("borderline", 0))
        unstable = int(label_counts.get("unstable", 0))
        total = trusted + borderline + unstable
        lines.extend(
            [
                "",
                f"Per-detection stability under {trials} perturbations ({perturbations}). A "
                "detection is **trusted** if a same-class box survives nearly every perturbation "
                "and **unstable** if it flickers (a likely false positive).",
                "",
                "| Reliability | Detections | Guidance |",
                "|---|---:|---|",
                f"| Trusted (survives ≥75%) | {trusted} | Use with confidence |",
                f"| Borderline (≥50%) | {borderline} | Spot-check |",
                f"| Unstable (<50%) | {unstable} | Review — likely false positive |",
                f"| Total | {total} | |",
            ]
        )
        by_class = stability.get("by_class") or {}
        if by_class:
            lines.extend(["", "| Class | Trusted | Borderline | Unstable |", "|---|---:|---:|---:|"])
            for class_name in sorted(by_class):
                counts_for_class = by_class[class_name]
                lines.append(
                    f"| {class_name} | {counts_for_class.get('trusted', 0)} | "
                    f"{counts_for_class.get('borderline', 0)} | {counts_for_class.get('unstable', 0)} |"
                )
    else:
        lines.append("")
        lines.append("- Per-detection stability was not computed for this run.")
    lines.extend(
        [
            "",
            "_Caveat: this detector has no held-out validation set (it was trained with mAP) and is "
            "known to over-detect (false positives). Confidence is the model's raw score, not a "
            "calibrated probability of correctness. Treat confidence and stability together as "
            "triage, not ground truth — to quantify accuracy, hand-verify the flagged "
            "(unstable/borderline) detections, which calibrates a precision estimate for this imagery._",
        ]
    )

    geospatial = payload.get("geospatial") or {}
    if geospatial.get("points"):
        metrics = geospatial.get("metrics") or {}
        nearest = metrics.get("nearest_neighbor_m") or {}
        totals = metrics.get("totals_by_class") or {}
        lines.extend(["", "## Geospatial survey", ""])
        lines.append(f"- Georeferenced images: {geospatial.get('georeferenced_image_count', 0)}")
        if metrics.get("survey_extent_m") is not None:
            lines.append(f"- Survey extent: {float(metrics['survey_extent_m']):.0f} m")
        if nearest:
            lines.append(
                f"- Image spacing (nearest-neighbor): mean {float(nearest.get('mean', 0)):.0f} m "
                f"(min {float(nearest.get('min', 0)):.0f}, max {float(nearest.get('max', 0)):.0f})"
            )
        if totals:
            lines.append("- Totals across survey: " + ", ".join(f"{name}: {count}" for name, count in sorted(totals.items())))
        if metrics.get("dog_per_burrow") is not None:
            lines.append(f"- Prairie dog : burrow ratio: {float(metrics['dog_per_burrow']):.2f}")
        lines.append("- See the survey detection map artifact (points sized by detections, coloured by reliability).")

    lines.extend(["", "## Spectral Review Candidates"])
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
