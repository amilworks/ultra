"""gt2 census + recovered-vs-gt2 label diff — GoldGate M1 freeze input #4 (part 2).

Answers two plan questions (section 5b) from the four GT-annotated frames:
1. The prairie_dog_in_burrow census at native resolution (box sizes, per frame).
2. Dropped-vs-remapped: the recovered YOLO labels contain zero class ids beyond
   {0,1} — did the original dataset builder DROP prairie_dog_in_burrow boxes or
   REMAP them onto prairie_dog/burrow? For every gt2 pdib box we look for a
   recovered tile-label box whose remapped frame-space center lands on it.

Layer selection follows the manifest hazard rules: names are trimmed, the first
layer matching gt_layer_priority wins, and duplicate same-priority layers fail.

Tile remap: recovered tiles are named <stem>_<row>_<col>.jpg on a 512px window
with stride 384 (verified against the production build_sliding_tiles grid), so
tile origin = (col*384, row*384) and a label box (cx,cy) in tile-normalized
coords maps to frame coords (col*384 + cx*512, row*384 + cy*512).
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

GT_LAYER_PRIORITY = ("gt2", "New Ground Truth")
TILE_NAME_RE = re.compile(r"^(?P<stem>.+)_(?P<row>\d+)_(?P<col>\d+)$")
TILE_PX = 512
STRIDE_PX = 384
CLASS_NAMES = {0: "prairie_dog", 1: "burrow"}
MATCH_RADIUS_PX = 24.0


def select_gt_layer(image_element: ET.Element) -> tuple[str, ET.Element]:
    layers: dict[str, list[ET.Element]] = {}
    for gobject in image_element.findall("gobject"):
        name = (gobject.get("name") or "").strip()
        layers.setdefault(name, []).append(gobject)
    for wanted in GT_LAYER_PRIORITY:
        matches = layers.get(wanted, [])
        if len(matches) > 1:
            raise ValueError(f"multiple GT layers named {wanted!r} after trimming")
        if matches:
            return wanted, matches[0]
    raise ValueError(f"no GT layer matches priority {GT_LAYER_PRIORITY}; found {sorted(layers)}")


def layer_boxes(layer: ET.Element) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for class_gobject in layer.findall("gobject"):
        class_name = (class_gobject.get("name") or "").strip()
        for rectangle in class_gobject.findall("rectangle"):
            vertices = rectangle.findall("vertex")
            if len(vertices) < 2:
                continue
            x0, y0 = float(vertices[0].get("x")), float(vertices[0].get("y"))
            x1, y1 = float(vertices[1].get("x")), float(vertices[1].get("y"))
            boxes.append(
                {
                    "class": class_name,
                    "cx": (x0 + x1) / 2.0,
                    "cy": (y0 + y1) / 2.0,
                    "w": abs(x1 - x0),
                    "h": abs(y1 - y0),
                }
            )
    return boxes


def recovered_frame_boxes(dataset_dir: Path, run: str, frame_stem: str) -> list[dict[str, Any]]:
    """Remap every recovered tile-label box for a frame into frame coords."""
    boxes: list[dict[str, Any]] = []
    labels_dir = dataset_dir / "labels" / run
    for label_path in sorted(labels_dir.glob(f"{frame_stem}_*.txt")):
        match = TILE_NAME_RE.match(label_path.stem)
        if not match or match.group("stem") != frame_stem:
            continue
        row, col = int(match.group("row")), int(match.group("col"))
        origin_x, origin_y = col * STRIDE_PX, row * STRIDE_PX
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            boxes.append(
                {
                    "class": CLASS_NAMES.get(class_id, f"class_{class_id}"),
                    "cx": origin_x + float(parts[1]) * TILE_PX,
                    "cy": origin_y + float(parts[2]) * TILE_PX,
                    "w": float(parts[3]) * TILE_PX,
                    "h": float(parts[4]) * TILE_PX,
                }
            )
    return boxes


def census(frames_dir: Path, dataset_dir: Path, run: str) -> dict[str, Any]:
    report: dict[str, Any] = {"frames": {}, "pdib_boxes": 0, "pdib_dispositions": Counter()}
    for xml_path in sorted(frames_dir.glob("*.JPG.xml")):
        frame_stem = xml_path.name[: -len(".JPG.xml")]
        image_element = ET.parse(xml_path).getroot()
        layer_name, layer = select_gt_layer(image_element)
        gt_boxes = layer_boxes(layer)
        recovered = recovered_frame_boxes(dataset_dir, run, frame_stem)

        gt_counts = Counter(box["class"] for box in gt_boxes)
        recovered_counts = Counter(box["class"] for box in recovered)

        pdib_rows = []
        for box in gt_boxes:
            if box["class"] != "prairie_dog_in_burrow":
                continue
            report["pdib_boxes"] += 1
            nearest = None
            nearest_distance = None
            for candidate in recovered:
                distance = ((candidate["cx"] - box["cx"]) ** 2 + (candidate["cy"] - box["cy"]) ** 2) ** 0.5
                if nearest_distance is None or distance < nearest_distance:
                    nearest, nearest_distance = candidate, distance
            if nearest is not None and nearest_distance is not None and nearest_distance <= MATCH_RADIUS_PX:
                disposition = f"remapped_to_{nearest['class']}"
            else:
                disposition = "dropped"
            report["pdib_dispositions"][disposition] += 1
            pdib_rows.append(
                {
                    "cx": box["cx"],
                    "cy": box["cy"],
                    "w": box["w"],
                    "h": box["h"],
                    "disposition": disposition,
                    "nearest_distance_px": round(nearest_distance, 1) if nearest_distance is not None else None,
                }
            )

        report["frames"][frame_stem] = {
            "gt_layer": layer_name,
            "gt2_counts": dict(gt_counts),
            "recovered_counts": dict(recovered_counts),
            "pdib": pdib_rows,
            "gt2_box_sizes_native_px": [
                {"class": box["class"], "w": round(box["w"], 1), "h": round(box["h"], 1)} for box in gt_boxes
            ],
        }
    report["pdib_dispositions"] = dict(report["pdib_dispositions"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames_dir", type=Path, help="dir with <frame>.JPG.xml GT files (test-prairie)")
    parser.add_argument("dataset_dir", type=Path, help="recovered dataset root")
    parser.add_argument("--run", default="run1_tiles")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = census(args.frames_dir, args.dataset_dir, args.run)
    payload = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
