"""Tests for the torch-free per-detection stability + matching logic (the model/detector is
injected, so these run without torch)."""

from __future__ import annotations

import itertools
import random
import tempfile
from pathlib import Path

import numpy as np

from ultra_deepagents.rarespot._matching import hungarian_min_cost, matched_source_indices
from ultra_deepagents.rarespot.stability import (
    assign_stability,
    perturbation_fns,
    run_detection_stability,
    stability_label,
)


def _brute_min(cost: np.ndarray) -> float:
    n = cost.shape[0]
    return min(sum(cost[i, perm[i]] for i in range(n)) for perm in itertools.permutations(range(n)))


def test_hungarian_matches_brute_force():
    random.seed(1)
    rng = np.random.default_rng(1)
    for _ in range(150):
        n = random.randint(1, 6)
        cost = rng.random((n, n))
        rows, cols = hungarian_min_cost(cost)
        assert abs(sum(cost[r, c] for r, c in zip(rows, cols)) - _brute_min(cost)) < 1e-9


def test_matched_source_indices_is_class_aware_and_optimal():
    # class mismatch never survives, even at perfect overlap
    assert matched_source_indices(
        [{"class_name": "burrow", "xyxy": [0, 0, 10, 10]}],
        [{"class_name": "prairie_dog", "xyxy": [0, 0, 10, 10]}],
        iou_thresh=0.5,
    ) == set()
    # optimal assignment matches both (greedy-by-IoU would strand one)
    a = [{"class_name": "x", "xyxy": [0, 0, 10, 10]}, {"class_name": "x", "xyxy": [0, 0, 10, 9]}]
    b = [{"class_name": "x", "xyxy": [0, 0, 10, 10]}, {"class_name": "x", "xyxy": [0, 0, 10, 9]}]
    assert matched_source_indices(a, b, iou_thresh=0.5) == {0, 1}


def test_assign_stability_fractions_and_labels():
    preds = [{"input_path": "img", "boxes": [
        {"class_name": "prairie_dog", "xyxy": [0, 0, 10, 10]},
        {"class_name": "burrow", "xyxy": [50, 50, 60, 60]},
    ]}]
    perturbed = {
        (0, "blur"): [{"class_name": "prairie_dog", "xyxy": [0, 0, 10, 10]},
                      {"class_name": "burrow", "xyxy": [50, 50, 60, 60]}],
        (0, "jpeg"): [{"class_name": "prairie_dog", "xyxy": [1, 1, 11, 11]}],
    }
    summary = assign_stability(predictions=preds, perturbed_boxes=perturbed, tags=["blur", "jpeg"], match_iou=0.5)
    box0, box1 = preds[0]["boxes"]
    assert box0["stability"] == 1.0 and box0["stability_label"] == "trusted"
    assert box1["stability"] == 0.5 and box1["stability_label"] == "borderline"
    assert summary["label_counts"] == {"trusted": 1, "borderline": 1, "unstable": 0}


def test_stability_label_thresholds():
    assert stability_label(1.0) == "trusted"
    assert stability_label(0.75) == "trusted"
    assert stability_label(0.5) == "borderline"
    assert stability_label(0.49) == "unstable"


def test_run_detection_stability_end_to_end_with_injected_detector():
    from PIL import Image

    tmp = Path(tempfile.mkdtemp())
    source = tmp / "tiles"
    source.mkdir()
    Image.new("RGB", (64, 64), (120, 120, 120)).save(source / "t0.jpg")
    tile_manifest = {"t0": {"prepared_index": 0, "x0": 0, "y0": 0, "width": 64, "height": 64,
                            "image_width": 64, "image_height": 64}}
    predictions = [{"input_path": "img", "boxes": [
        {"class_id": 0, "class_name": "prairie_dog", "xyxy": [10, 10, 30, 30], "confidence": 0.6}]}]

    def fake_detect(*, source_dir, project_subdir, name):
        labels = Path(source_dir).parent / project_subdir / name / "labels"
        labels.mkdir(parents=True, exist_ok=True)
        for jpg in Path(source_dir).glob("*.jpg"):
            (labels / f"{jpg.stem}.txt").write_text("0 0.3125 0.3125 0.3125 0.3125 0.6\n")
        return labels

    run_detection_stability(
        predictions=predictions, source_dir=source, output_dir=tmp,
        tile_manifest=tile_manifest, detect_fn=fake_detect, match_iou=0.5, iou_threshold=0.45,
    )
    box = predictions[0]["boxes"][0]
    assert box["stability"] == 1.0 and box["stability_label"] == "trusted"
    assert box["stability_trials"] == len(perturbation_fns())
    assert len(list((source.parent / "stability_tiles").glob("*.jpg"))) == len(perturbation_fns())
