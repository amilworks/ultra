"""RareSpot adapter tests — real test-prairie fixtures, no network, no torch.

Materialization runs against one real GT frame (EnrNE_Day2_Run1__0211: the
frame that carries ONLY the gt2 layer, with 6 prairie_dog_in_burrow boxes);
the leakage checks run against a mini inventory the test writes itself; the
two-pass kernel is exercised through its pure functions (parse/match/assemble)
with synthetic inputs.
"""

from __future__ import annotations

import hashlib
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pytest
import yaml
from ultra_deepagents.rarespot.tiling import build_sliding_tiles
from ultra_deepagents.training.adapters import MaterializeContext, TrainContext
from ultra_deepagents.training.gt2_census import layer_boxes, select_gt_layer
from ultra_deepagents.training.phash import PHASH_KERNEL
from ultra_deepagents.training.rarespot_adapter import (
    GRAY_FILL_RGB,
    HYP_PATH,
    LeakageCheckRefused,
    LocalDirectoryFrameSource,
    RareSpotAdapter,
    assemble_detection_metrics,
    bn_freeze_patch_source,
    build_finetune_command,
    match_operating_point,
    parse_val_output,
    reconcile_class,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_PRAIRIE = REPO_ROOT / "test-prairie"
FRAME = "EnrNE_Day2_Run1__0211"

pytestmark = pytest.mark.skipif(
    not (TEST_PRAIRIE / f"{FRAME}.JPG").is_file(),
    reason="test-prairie fixtures not present",
)


def _gt2_counts() -> Counter:
    root = ET.parse(TEST_PRAIRIE / f"{FRAME}.JPG.xml").getroot()
    _, layer = select_gt_layer(root)
    return Counter(box["class"] for box in layer_boxes(layer))


def _gt2_pdib_boxes() -> list[dict]:
    root = ET.parse(TEST_PRAIRIE / f"{FRAME}.JPG.xml").getroot()
    _, layer = select_gt_layer(root)
    return [box for box in layer_boxes(layer) if box["class"] == "prairie_dog_in_burrow"]


@pytest.fixture(scope="module")
def materialized(tmp_path_factory) -> tuple[dict, Path]:
    source_root = tmp_path_factory.mktemp("frames")
    for suffix in (".JPG", ".JPG.xml"):
        os.symlink(TEST_PRAIRIE / f"{FRAME}{suffix}", source_root / f"{FRAME}{suffix}")
    workdir = tmp_path_factory.mktemp("workdir")
    adapter = RareSpotAdapter()
    ctx = MaterializeContext(
        purpose="gold_cut",
        params={"frame_source_dir": str(source_root), "site_id": "EnrNE"},
        workdir=workdir,
    )
    return adapter.materialize_dataset(ctx), workdir


# -------------------------------------------------------------- frame source


def test_local_directory_frame_source_lists_gt_frames() -> None:
    source = LocalDirectoryFrameSource(TEST_PRAIRIE)
    frames = source.list_frames()
    assert FRAME in frames
    assert "test_no_gt" not in frames  # no XML, no frame
    image_path, xml_path = source.fetch(FRAME)
    assert image_path.name == f"{FRAME}.JPG"
    assert xml_path.name == f"{FRAME}.JPG.xml"


def test_bisque_frame_source_is_an_honest_stub() -> None:
    from ultra_deepagents.training.rarespot_adapter import BisqueFrameSource

    with pytest.raises(NotImplementedError, match="fleet credentials"):
        BisqueFrameSource()


# ------------------------------------------------------------ materialization


def test_materialize_emits_standard_item_with_exif_and_identity(materialized) -> None:
    manifest, _ = materialized
    assert manifest["model_key"] == "yolov5_rarespot"
    assert manifest["purpose"] == "gold_cut"
    assert manifest["tile_spec"] == {"tile_size": 512, "overlap": 0.25, "stride": 384}
    items = manifest["items"]
    assert len(items) == 1
    item = items[0]
    assert item["item_id"] == FRAME
    assert item["label_kind"] == "boxes"
    assert item["width"] == 6000 and item["height"] == 4000
    # EXIF extracted BEFORE tiling (step 1a): GPS present on the source frame.
    assert item["exif_present"] is True
    geom = item["footprint_geom"]
    assert 47.0 < geom["center_lat"] < 48.0
    assert -108.0 < geom["center_lon"] < -107.0
    assert geom["alt_msl"] is not None and geom["alt_msl"] > 700
    assert geom["footprint_polygon"] is None and geom["agl_m"] is None
    # Identity: sha256 + pHash per frame.
    assert len(item["content_sha256"]) == 64
    assert len(item["gt_label_sha256"]) == 64
    assert len(item["phash"]) == 16
    int(item["phash"], 16)
    # source_ref carries the BisQue resource_uniq parsed from the GObject XML.
    assert item["source_ref"]["bisque_image_id"].startswith("00-")
    assert item["source_ref"]["frame_stem"] == FRAME


def test_materialize_selects_gt2_and_matches_census_counts(materialized) -> None:
    manifest, _ = materialized
    item = manifest["items"][0]
    assert item["metadata"]["gt_layer"] == "gt2"
    expected = _gt2_counts()
    per_class = item["label_stats"]["per_class_box_count"]
    assert per_class["prairie_dog"] == expected["prairie_dog"]
    assert per_class["burrow"] == expected["burrow"]
    assert item["label_stats"]["box_count"] == expected["prairie_dog"] + expected["burrow"]
    # pdib is counted with its disposition, never as a label.
    pdib = manifest["unsupported_class_counts"]["prairie_dog_in_burrow"]
    assert pdib == {"count": expected["prairie_dog_in_burrow"], "disposition": "ignore"}
    assert len(item["metadata"]["ignore_boxes"]) == expected["prairie_dog_in_burrow"]


def test_materialize_tiles_full_frame_and_labels_only_model_classes(materialized) -> None:
    manifest, workdir = materialized
    item = manifest["items"][0]
    grid = build_sliding_tiles(width=6000, height=4000, tile_size=512, overlap=0.25)
    assert item["label_stats"]["tile_count"] == len(grid.tiles)
    frames_dir = workdir / "frames" / FRAME
    assert len(list(frames_dir.glob("*.jpg"))) == len(grid.tiles)
    labels_dir = Path(item["label_uri"])
    label_files = sorted(labels_dir.glob("*.txt"))
    assert label_files, "the GT frame must produce labeled tiles"
    class_ids = set()
    for label_file in label_files:
        for line in label_file.read_text().splitlines():
            class_ids.add(int(line.split()[0]))
    assert class_ids <= {0, 1}  # nc=2; no third class ever reaches the trainer


def test_pdib_boxes_became_gray_ignore_regions_with_no_label_rows(materialized) -> None:
    from PIL import Image

    manifest, workdir = materialized
    item = manifest["items"][0]
    grid = build_sliding_tiles(width=6000, height=4000, tile_size=512, overlap=0.25)
    pdib_boxes = _gt2_pdib_boxes()
    assert pdib_boxes
    checked = 0
    for box in pdib_boxes:
        cx, cy = box["cx"], box["cy"]
        tile = next(
            (
                candidate
                for candidate in grid.tiles
                if candidate.x0 + 4 <= cx < candidate.x1 - 4
                and candidate.y0 + 4 <= cy < candidate.y1 - 4
            ),
            None,
        )
        if tile is None:
            continue
        checked += 1
        tile_name = f"{FRAME}_{tile.grid_y}_{tile.grid_x}"
        with Image.open(workdir / "frames" / FRAME / f"{tile_name}.jpg") as tile_image:
            pixel = tile_image.convert("RGB").getpixel((int(cx) - tile.x0, int(cy) - tile.y0))
        # JPEG q95 ringing tolerance around the exact letterbox gray fill.
        assert all(abs(channel - fill) <= 8 for channel, fill in zip(pixel, GRAY_FILL_RGB)), (
            f"pdib box at ({cx:.0f},{cy:.0f}) was not gray-filled on {tile_name}: {pixel}"
        )
        # No label row was emitted for the ignore box.
        label_path = Path(item["label_uri"]) / f"{tile_name}.txt"
        if label_path.is_file():
            for line in label_path.read_text().splitlines():
                parts = line.split()
                row_cx = float(parts[1]) * tile.width + tile.x0
                row_cy = float(parts[2]) * tile.height + tile.y0
                assert not (abs(row_cx - cx) < 4 and abs(row_cy - cy) < 4), (
                    f"ignore box leaked a label row on {tile_name}"
                )
    assert checked > 0, "no pdib box landed strictly inside a tile - fixture assumption broke"


def test_ignore_sidecar_is_folded_into_gt_label_sha(materialized) -> None:
    manifest, _ = materialized
    item = manifest["items"][0]
    labels_dir = Path(item["label_uri"])
    sidecar = labels_dir / "ignore_boxes.json"
    assert sidecar.is_file()
    ignore_json = sidecar.read_bytes()
    label_bytes = b"".join(
        f"{path.stem}\n".encode() + path.read_bytes() for path in sorted(labels_dir.glob("*.txt"))
    )
    expected = hashlib.sha256(label_bytes + b"\0" + ignore_json).hexdigest()
    assert item["gt_label_sha256"] == expected


def test_taxonomy_reconcile_trims_and_maps() -> None:
    assert reconcile_class(" prairie_dog ") == ("prairie_dog", "map")
    assert reconcile_class("burrow") == ("burrow", "map")
    assert reconcile_class(" prairie_dog_in_burrow") == ("prairie_dog_in_burrow", "ignore")
    assert reconcile_class("coyote") == ("coyote", "dropped")


# -------------------------------------------------------------- leakage extras


def _write_inventory(tmp_path: Path, *, kernel: str = PHASH_KERNEL) -> tuple[str, str]:
    inventory = {
        "corpus": "recovered-train-v0",
        "phash_kernel": kernel,
        "frame_stems": {"run1_tiles": ["frameA"], "run2_tiles": []},
        "tiles": [
            {
                "relpath": "images/run1_tiles/frameA_0_0.jpg",
                "run": "run1_tiles",
                "stem": "frameA",
                "content_sha256": "c" * 64,
                "phash": "0000000000000000",
                "label_sha256": None,
                "per_class_box_count": {},
            }
        ],
        "exif_frame_centers": [{"stem": "frameA", "lat": 47.7520, "lon": -107.7670}],
    }
    path = tmp_path / "inventory.json"
    raw = json.dumps(inventory, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    return str(path), hashlib.sha256(raw).hexdigest()


def _gold_item(
    item_id: str,
    *,
    slice_name: str,
    phash: str = "ffffffffffffffff",
    sha: str = "d" * 64,
    center=None,
) -> dict:
    item = {
        "item_id": item_id,
        "slice": slice_name,
        "source_ref": {"frame_stem": item_id},
        "content_sha256": sha,
        "gt_label_sha256": "e" * 64,
        "phash": phash,
    }
    if center is not None:
        item["footprint_geom"] = {"center_lat": center[0], "center_lon": center[1]}
    return item


def test_leakage_stem_identity_violates_held_out_but_exempts_prior(tmp_path) -> None:
    uri, sha = _write_inventory(tmp_path)
    adapter = RareSpotAdapter()
    params = {"exclusion_inventory_uri": uri, "inventory_sha256": sha}
    held_out = adapter.extra_leakage_checks(
        [], [_gold_item("frameA", slice_name="held_out_test")], params=params
    )
    assert any(violation["check"] == "identity_stem" for violation in held_out)
    prior = adapter.extra_leakage_checks(
        [], [_gold_item("frameA", slice_name="prior_train")], params=params
    )
    assert prior == []


def test_leakage_content_sha_identity(tmp_path) -> None:
    uri, sha = _write_inventory(tmp_path)
    adapter = RareSpotAdapter()
    params = {"exclusion_inventory_uri": uri, "inventory_sha256": sha}
    violations = adapter.extra_leakage_checks(
        [], [_gold_item("fresh", slice_name="held_out_test", sha="c" * 64)], params=params
    )
    assert any(violation["check"] == "identity_content_sha256" for violation in violations)


def test_leakage_phash_near_dup_held_out_only(tmp_path) -> None:
    uri, sha = _write_inventory(tmp_path)
    adapter = RareSpotAdapter()
    params = {"exclusion_inventory_uri": uri, "inventory_sha256": sha}
    near = "0000000000000003"  # Hamming 2 from the inventory tile
    violations = adapter.extra_leakage_checks(
        [], [_gold_item("fresh", slice_name="held_out_test", phash=near)], params=params
    )
    assert any(violation["check"] == "phash_near_dup" for violation in violations)
    assert (
        adapter.extra_leakage_checks(
            [], [_gold_item("fresh", slice_name="prior_train", phash=near)], params=params
        )
        == []
    )
    far = adapter.extra_leakage_checks(
        [],
        [_gold_item("fresh", slice_name="held_out_test", phash="ffffffffffffffff")],
        params=params,
    )
    assert far == []


def test_leakage_geo_radius_against_inventory_and_prior_gold(tmp_path) -> None:
    uri, sha = _write_inventory(tmp_path)
    adapter = RareSpotAdapter()
    params = {"exclusion_inventory_uri": uri, "inventory_sha256": sha, "geo_radius_m": 200}
    # ~11m from the inventory EXIF center -> violation.
    close = adapter.extra_leakage_checks(
        [],
        [_gold_item("fresh", slice_name="held_out_test", center=(47.7521, -107.7670))],
        params=params,
    )
    assert any(violation["check"] == "aerial_geospatial_overlap" for violation in close)
    # ~100km away -> clean.
    far = adapter.extra_leakage_checks(
        [], [_gold_item("fresh", slice_name="held_out_test", center=(48.5, -106.5))], params=params
    )
    assert far == []
    # Within the radius of a prior_train GOLD center (no inventory center near).
    items = [
        _gold_item("prior", slice_name="prior_train", center=(48.5000, -106.5000)),
        _gold_item("candidate", slice_name="held_out_test", center=(48.5005, -106.5000)),  # ~55m
    ]
    mixed = adapter.extra_leakage_checks([], items, params=params)
    assert any(
        violation["check"] == "aerial_geospatial_overlap" and violation["item_id"] == "candidate"
        for violation in mixed
    )


def test_leakage_refuses_kernel_mismatch(tmp_path) -> None:
    uri, sha = _write_inventory(tmp_path, kernel="phash64/other-kernel@9/params")
    adapter = RareSpotAdapter()
    with pytest.raises(LeakageCheckRefused, match="kernel mismatch"):
        adapter.extra_leakage_checks(
            [],
            [_gold_item("fresh", slice_name="held_out_test")],
            params={"exclusion_inventory_uri": uri, "inventory_sha256": sha},
        )


def test_leakage_refuses_sha_mismatch(tmp_path) -> None:
    uri, _ = _write_inventory(tmp_path)
    adapter = RareSpotAdapter()
    with pytest.raises(LeakageCheckRefused, match="sha mismatch"):
        adapter.extra_leakage_checks(
            [],
            [_gold_item("fresh", slice_name="held_out_test")],
            params={"exclusion_inventory_uri": uri, "inventory_sha256": "0" * 64},
        )


def test_leakage_refuses_missing_inventory_config() -> None:
    adapter = RareSpotAdapter()
    with pytest.raises(LeakageCheckRefused, match="not configured"):
        adapter.extra_leakage_checks(
            [], [_gold_item("fresh", slice_name="held_out_test")], params={}
        )


# ---------------------------------------------------------- two-pass (pure)


def test_parse_val_output_reads_the_vendored_table() -> None:
    text = """
val: data=/tmp/gold.yaml, weights=['w.pt'], conf_thres=0.001
                 Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95
                   all        176         29      0.905      0.897      0.937      0.554
           prairie_dog        176         10        0.9        0.9       0.95        0.6
                burrow        176         19       0.91       0.89       0.92       0.51
Speed: 0.2ms pre-process, 5.1ms inference, 1.0ms NMS per image
"""
    parsed = parse_val_output(text, class_names=["prairie_dog", "burrow"])
    assert parsed["map50"] == pytest.approx(0.937)
    assert parsed["map50_95"] == pytest.approx(0.554)
    assert parsed["labels"] == 29
    assert parsed["per_class"]["prairie_dog"]["ap50"] == pytest.approx(0.95)
    assert parsed["per_class"]["burrow"]["label_count"] == 19
    # max-F1 P/R are recorded under explicit names, never as op-point numbers.
    assert parsed["per_class"]["burrow"]["precision_max_f1"] == pytest.approx(0.91)
    with pytest.raises(ValueError, match="no 'all' summary row"):
        parse_val_output("garbage\n", class_names=["prairie_dog", "burrow"])


def test_match_operating_point_counts_tp_fp_fn_and_empty_frames() -> None:
    gt = {
        "s/tile_a": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "confidence": None}],
        "s/tile_empty": [],
    }
    preds = {
        "s/tile_a": [
            {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "confidence": 0.9},  # tp
            {
                "class_id": 1,
                "cx": 0.1,
                "cy": 0.1,
                "w": 0.1,
                "h": 0.1,
                "confidence": 0.6,
            },  # fp (burrow)
        ],
        "s/tile_empty": [
            {
                "class_id": 0,
                "cx": 0.3,
                "cy": 0.3,
                "w": 0.1,
                "h": 0.1,
                "confidence": 0.5,
            },  # fp on empty
        ],
    }
    result = match_operating_point(
        gt_by_tile=gt, pred_by_tile=preds, class_names=["prairie_dog", "burrow"], iou_threshold=0.5
    )
    dog = result["per_class"]["prairie_dog"]
    assert (dog["tp"], dog["fp"], dog["fn"]) == (1, 1, 0)
    assert dog["predicted_count"] == 2
    assert dog["label_count"] == 1
    burrow = result["per_class"]["burrow"]
    assert (burrow["tp"], burrow["fp"], burrow["fn"]) == (0, 1, 0)
    assert result["empty_frame_count"] == 1
    assert result["fp_per_empty_frame"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["precision"] == pytest.approx(1 / 3)


def test_assemble_detection_metrics_empty_held_out_contract() -> None:
    """Plan 7.3 emission contract: per_slice.held_out_test is ALWAYS present —
    label_count 0 with null metric values, never an omitted key — and the
    per-class predicted_count/label_count keys are always present (conv. 3)."""
    metrics = assemble_detection_metrics(
        class_names=["prairie_dog", "burrow"],
        slices={
            "prior_train": {
                "label_count": 29,
                "per_class_label_count": {"prairie_dog": 10, "burrow": 19},
            },
            "held_out_test": {"label_count": 0},
        },
        pass1_by_slice={
            "prior_train": {
                "map50": 0.9,
                "map50_95": 0.55,
                "per_class": {
                    "prairie_dog": {"ap50": 0.95, "ap50_95": 0.6, "label_count": 10},
                    "burrow": {"ap50": 0.85, "ap50_95": 0.5, "label_count": 19},
                },
            }
        },
        op_metrics={
            "precision": 0.9,
            "recall": 0.88,
            "fp_per_empty_frame": 0.02,
            "per_class": {
                "prairie_dog": {
                    "predicted_count": 11,
                    "label_count": 10,
                    "precision": 0.91,
                    "recall": 0.9,
                },
                "burrow": {
                    "predicted_count": 20,
                    "label_count": 19,
                    "precision": 0.89,
                    "recall": 0.87,
                },
            },
        },
        operating_point={"conf": 0.25, "iou": 0.45},
        passes=3,
        wall_clock_s=12.5,
    )
    assert metrics["schema"] == "detection.v1"
    held_out = metrics["per_slice"]["held_out_test"]
    assert held_out["label_count"] == 0
    assert held_out["map50"] is None and held_out["map50_95"] is None
    assert held_out["per_class"]["prairie_dog"] == {"ap50": None, "ap50_95": None, "label_count": 0}
    prior = metrics["per_slice"]["prior_train"]
    assert prior["map50"] == 0.9 and prior["label_count"] == 29
    # Convention 3: predicted_count/label_count under exactly those key names.
    dog = metrics["per_class"]["prairie_dog"]
    assert dog["predicted_count"] == 11 and dog["label_count"] == 10
    assert dog["recall_at_op"] == 0.9 and dog["precision_at_op"] == 0.91
    aggregate = metrics["aggregate"]
    assert aggregate["map50"] == pytest.approx(0.9)  # single populated slice
    assert aggregate["fp_per_empty_frame"] == 0.02
    assert aggregate["operating_point"] == {"conf": 0.25, "iou": 0.45}
    assert metrics["eval"] == {"passes": 3, "kernel": "yolov5_two_pass/v1", "wall_clock_s": 12.5}


def test_assemble_detection_metrics_fully_empty_gold() -> None:
    metrics = assemble_detection_metrics(
        class_names=["prairie_dog", "burrow"],
        slices={},
        pass1_by_slice={},
        op_metrics=None,
        operating_point={"conf": 0.25, "iou": 0.45},
        wall_clock_s=0.0,
    )
    assert metrics["aggregate"]["map50"] is None
    for slice_name in ("prior_train", "held_out_test"):
        assert metrics["per_slice"][slice_name]["label_count"] == 0
    for class_name in ("prairie_dog", "burrow"):
        row = metrics["per_class"][class_name]
        assert row["predicted_count"] == 0 and row["label_count"] == 0
    assert metrics["eval"]["passes"] == 0


# ------------------------------------------------------------- finetune (M3)


def test_build_finetune_command_golden_argv() -> None:
    params = {
        "python": "python",
        "weights_uri": "/weights/active.pt",
        "data_yaml": "/ds/assembled_finetune.yaml",
        "yolov5_path": "/opt/rarespot/yolov5",
        "run_dir": "/runs/job-1",
        "new_tile_count": 500,
    }
    argv = build_finetune_command(params)
    assert argv == [
        "python",
        "/opt/rarespot/yolov5/train.py",
        "--weights",
        "/weights/active.pt",
        "--data",
        "/ds/assembled_finetune.yaml",
        "--hyp",
        str(HYP_PATH),
        "--imgsz",
        "512",
        "--epochs",
        "40",
        "--batch-size",
        "16",
        "--freeze",
        "10",
        "--patience",
        "15",
        "--seed",
        "0",
        "--project",
        "/runs/job-1",
        "--name",
        "finetune",
        "--exist-ok",
        "--save-period",
        "5",
    ]
    # 60-epoch rule above 1000 new tiles.
    argv_large = build_finetune_command({**params, "new_tile_count": 1001})
    assert argv_large[argv_large.index("--epochs") + 1] == "60"
    # tesla overflow batch size override.
    argv_tesla = build_finetune_command({**params, "batch_size": 8})
    assert argv_tesla[argv_tesla.index("--batch-size") + 1] == "8"
    # Forbidden knobs never appear (no --close-mosaic arg in the vendored tree;
    # --image-weights destabilizes 2-class).
    assert not any("close" in token for token in argv)
    assert "--image-weights" not in argv


def test_build_finetune_command_refuses_scratch_training() -> None:
    with pytest.raises(ValueError, match="never train from scratch"):
        build_finetune_command({"weights_uri": "", "data_yaml": "/d.yaml", "run_dir": "/r"})


def test_train_raises_the_m3_not_implemented_branch(tmp_path) -> None:
    adapter = RareSpotAdapter()
    ctx = TrainContext(
        params={
            "weights_uri": "/weights/active.pt",
            "data_yaml": "/ds/assembled.yaml",
            "run_dir": str(tmp_path),
        },
        workdir=tmp_path,
    )
    with pytest.raises(NotImplementedError, match="lambda worker deployment"):
        adapter.train(ctx)


def test_bn_freeze_patch_source_freezes_batchnorm() -> None:
    source = bn_freeze_patch_source()
    assert "_module.eval()" in source
    assert "track_running_stats = False" in source
    assert "range(10)" in source  # layers 0-9, matching --freeze 10
    compile(source, "<bn-freeze-patch>", "exec")  # must be valid Python


def test_hyp_file_matches_the_plan_recipe() -> None:
    hyp = yaml.safe_load(HYP_PATH.read_text(encoding="utf-8"))
    assert hyp["lr0"] == 0.0032
    assert hyp["lrf"] == 0.12
    assert hyp["momentum"] == 0.937
    assert hyp["weight_decay"] == 0.0005
    assert hyp["warmup_epochs"] == 2.0
    assert hyp["hsv_h"] == 0.015 and hyp["hsv_s"] == 0.7 and hyp["hsv_v"] == 0.4
    assert hyp["fliplr"] == 0.5 and hyp["flipud"] == 0.5  # nadir aerial: flipud ON
    assert hyp["degrees"] == 0.0 and hyp["shear"] == 0.0 and hyp["perspective"] == 0.0
    assert hyp["mixup"] == 0.0 and hyp["copy_paste"] == 0.0
    assert hyp["mosaic"] == 1.0
    # Every key the vendored train.py/loss.py actually consumes must exist —
    # a missing key is a KeyError mid-run.
    required = {
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "warmup_momentum",
        "warmup_bias_lr",
        "box",
        "cls",
        "cls_pw",
        "obj",
        "obj_pw",
        "iou_t",
        "anchor_t",
        "fl_gamma",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "mosaic",
        "mixup",
        "copy_paste",
    }
    assert required <= set(hyp)
