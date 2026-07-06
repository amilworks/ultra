"""Per-detection stability via perturbation consensus.

The detector has no held-out validation (mAP-during-training only) and a known
false-positive problem, so raw confidence is only a *relative* signal. This pass turns it
into an actionable per-box reliability triage: re-run detection over a bank of
coordinate-preserving perturbations of the original tiles (Gaussian blur, brightness
up/down, JPEG recompression) and, for each original detection, record the fraction of
perturbations under which a SAME-CLASS box survives (Hungarian IoU match >= threshold).

Real objects survive almost every perturbation; texture/noise false positives flicker.
The perturbations are coordinate-preserving (no flips/scales), so each tile's boxes remap
to the full image exactly as the originals do — no coordinate gymnastics.
"""

from __future__ import annotations

import io
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from ultra_deepagents.rarespot._matching import matched_source_indices
from ultra_deepagents.rarespot.tiling import classwise_nms, remap_tile_box, yolo_line_to_xyxy

CLASS_NAMES = ["prairie_dog", "burrow"]

# detect_fn(source_dir, project_subdir, name) -> labels_dir
DetectFn = Callable[..., Path]


def _jpeg_recompress(image: Any) -> Any:
    from PIL import Image

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=40)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def perturbation_fns() -> list[tuple[str, Callable[[Any], Any]]]:
    """Coordinate-preserving tile perturbations. Box geometry is unchanged by all of
    these, so a detection's coordinates map back to the full image identically."""
    from PIL import ImageEnhance, ImageFilter

    return [
        ("blur", lambda image: image.filter(ImageFilter.GaussianBlur(radius=1.2))),
        ("bright_up", lambda image: ImageEnhance.Brightness(image).enhance(1.2)),
        ("bright_down", lambda image: ImageEnhance.Brightness(image).enhance(0.8)),
        ("jpeg", _jpeg_recompress),
    ]


def stability_label(stability: float) -> str:
    if stability >= 0.75:
        return "trusted"
    if stability >= 0.5:
        return "borderline"
    return "unstable"


def generate_perturbed_tiles(
    *,
    source_dir: Path,
    stability_dir: Path,
    perturbations: list[tuple[str, Callable[[Any], Any]]],
    only_stems: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Write ``<stem>__pert-<tag>.jpg`` for every original tile. Returns a manifest mapping
    each perturbed stem to its original stem + perturbation tag.

    When ``only_stems`` is given, only those tile stems are perturbed. Stability scores a
    detection by how often it *survives* perturbation, so a tile that had no detection has
    no box to match and contributes nothing — skipping it is result-identical and, on
    sparse aerial frames where the vast majority of tiles are empty, cuts the perturbation
    detector passes (the dominant cost) by an order of magnitude. ``None`` perturbs every
    tile (the original behaviour), so callers without the detection set are unaffected."""
    from PIL import Image

    stability_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, Any]] = {}
    for tile_path in sorted(source_dir.glob("*.jpg")):
        original_stem = tile_path.stem
        if only_stems is not None and original_stem not in only_stems:
            continue
        with Image.open(tile_path) as image:
            rgb = image.convert("RGB")
            for tag, fn in perturbations:
                perturbed_stem = f"{original_stem}__pert-{tag}"
                fn(rgb.copy()).convert("RGB").save(
                    stability_dir / f"{perturbed_stem}.jpg", format="JPEG", quality=95
                )
                manifest[perturbed_stem] = {"original_stem": original_stem, "tag": tag}
    return manifest


def _perturbed_boxes_by_image_tag(
    *,
    perturbed_manifest: dict[str, dict[str, Any]],
    tile_manifest: dict[str, dict[str, Any]],
    labels_dir: Path,
    iou_threshold: float,
) -> dict[tuple[int, str], list[dict[str, Any]]]:
    rows_by_key: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for perturbed_stem, meta in perturbed_manifest.items():
        txt_path = labels_dir / f"{perturbed_stem}.txt"
        if not txt_path.exists():
            continue
        tile_meta = tile_manifest.get(str(meta["original_stem"]))
        if not tile_meta:
            continue
        prepared_index = int(tile_meta["prepared_index"])
        tag = str(meta["tag"])
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
            rows_by_key[(prepared_index, tag)].append(parsed)
    return {key: classwise_nms(rows, iou_threshold=iou_threshold) for key, rows in rows_by_key.items()}


def assign_stability(
    *,
    predictions: list[dict[str, Any]],
    perturbed_boxes: dict[tuple[int, str], list[dict[str, Any]]],
    tags: list[str],
    match_iou: float,
) -> dict[str, Any]:
    """Attach `stability`, `stability_label`, `stability_survived`, `stability_trials` to
    every original box (mutates predictions). Returns a summary across all boxes."""
    trials = len(tags)
    label_counts: dict[str, int] = {"trusted": 0, "borderline": 0, "unstable": 0}
    stable_by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"trusted": 0, "borderline": 0, "unstable": 0})
    for prepared_index, prediction in enumerate(predictions):
        boxes = list(prediction.get("boxes") or [])
        survived = [0] * len(boxes)
        for tag in tags:
            pert = perturbed_boxes.get((prepared_index, tag), [])
            for matched_index in matched_source_indices(boxes, pert, iou_thresh=match_iou, class_key="class_name"):
                if 0 <= matched_index < len(survived):
                    survived[matched_index] += 1
        for box_index, box in enumerate(boxes):
            score = (survived[box_index] / trials) if trials else 1.0
            label = stability_label(score)
            box["stability"] = round(float(score), 4)
            box["stability_label"] = label
            box["stability_survived"] = int(survived[box_index])
            box["stability_trials"] = int(trials)
            label_counts[label] += 1
            stable_by_class[str(box.get("class_name") or "")][label] += 1
    total = sum(label_counts.values())
    return {
        "trials": trials,
        "perturbations": list(tags),
        "match_iou": float(match_iou),
        "total_detections": total,
        "label_counts": label_counts,
        "by_class": {name: dict(counts) for name, counts in stable_by_class.items()},
        "trusted_fraction": round(label_counts["trusted"] / total, 4) if total else 0.0,
    }


def run_detection_stability(
    *,
    predictions: list[dict[str, Any]],
    source_dir: Path,
    output_dir: Path,
    tile_manifest: dict[str, dict[str, Any]],
    detect_fn: DetectFn,
    match_iou: float = 0.5,
    iou_threshold: float = 0.45,
    detection_tile_stems: set[str] | None = None,
) -> dict[str, Any]:
    """Orchestrate the stability pass and annotate `predictions` in place. `detect_fn` runs
    the detector over a tile directory and returns its labels dir; it is injected so this
    is testable without the model.

    ``detection_tile_stems`` (the tiles that carried a raw detection in the initial pass)
    restricts perturbation to tiles that actually have a box to stabilise; ``None`` perturbs
    every tile (unchanged behaviour)."""
    perturbations = perturbation_fns()
    tags = [tag for tag, _ in perturbations]
    if not tags:
        return {"trials": 0}
    stability_dir = output_dir / "stability_tiles"
    perturbed_manifest = generate_perturbed_tiles(
        source_dir=source_dir,
        stability_dir=stability_dir,
        perturbations=perturbations,
        only_stems=detection_tile_stems,
    )
    if not perturbed_manifest:
        return {"trials": 0}
    labels_dir = detect_fn(source_dir=stability_dir, project_subdir="stability_yolov5", name="predict")
    perturbed_boxes = _perturbed_boxes_by_image_tag(
        perturbed_manifest=perturbed_manifest,
        tile_manifest=tile_manifest,
        labels_dir=Path(labels_dir),
        iou_threshold=iou_threshold,
    )
    return assign_stability(predictions=predictions, perturbed_boxes=perturbed_boxes, tags=tags, match_iou=match_iou)
