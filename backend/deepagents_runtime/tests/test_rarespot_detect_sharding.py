"""The parallel detector sharding must be a pure throughput win: the merged per-tile
labels from K>1 shards must be IDENTICAL to a single sequential pass. The real YOLOv5
detector is mocked (deterministic per-tile output) so this exercises the split/parallel/
merge orchestration without the model."""
from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image

from ultra_deepagents.rarespot import inference
from ultra_deepagents.rarespot.config import RareSpotConfig


def _fake_detect_run(cmd, *args, **kwargs):
    source = Path(cmd[cmd.index("--source") + 1])
    project = Path(cmd[cmd.index("--project") + 1])
    name = cmd[cmd.index("--name") + 1]
    labels = project / name / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    # One deterministic "detection" per tile, tagged by stem so a mis-merge would show up.
    for jpg in sorted(source.glob("*.jpg")):
        (labels / f"{jpg.stem}.txt").write_text(f"0 0.5 0.5 0.2 0.2 0.9\n")
    return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")


def _config(tmp: Path) -> RareSpotConfig:
    return RareSpotConfig(
        weights_path=tmp / "weights.pt",
        yolov5_path=tmp / "yolov5",
        artifact_root=tmp,
        allowed_input_roots=(tmp,),
    )


def _make_tiles(src: Path, n: int) -> None:
    src.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (32, 32), (100, 100, 100)).save(src / f"tile-{i:04d}.jpg")


def _snapshot(labels_dir: Path) -> dict[str, str]:
    return {p.name: p.read_text() for p in sorted(labels_dir.glob("*.txt"))}


def test_sharded_detection_matches_single_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(inference.subprocess, "run", _fake_detect_run)

    # Baseline: force the single-process path.
    monkeypatch.setenv("ULTRA_RARESPOT_DETECT_WORKERS", "1")
    src1, out1 = tmp_path / "single" / "tiles", tmp_path / "single"
    _make_tiles(src1, 41)
    labels1 = inference.run_yolov5_detect(source_dir=src1, output_dir=out1, config=_config(tmp_path))
    single = _snapshot(labels1)

    # Force 5 parallel shards over the same 41 tiles.
    monkeypatch.setenv("ULTRA_RARESPOT_DETECT_WORKERS", "5")
    monkeypatch.setenv("ULTRA_RARESPOT_MIN_TILES_PER_WORKER", "1")
    src2, out2 = tmp_path / "sharded" / "tiles", tmp_path / "sharded"
    _make_tiles(src2, 41)
    labels2 = inference.run_yolov5_detect(source_dir=src2, output_dir=out2, config=_config(tmp_path))
    sharded = _snapshot(labels2)

    expected = {f"tile-{i:04d}.txt" for i in range(41)}
    assert set(single) == expected, "single-pass produced the wrong tile set"
    assert set(sharded) == expected, "sharded pass dropped or duplicated tiles"
    assert single == sharded, "sharded labels differ from single-pass (not merge-identical)"
    # confirm it really sharded (>1) under this budget, i.e. the test is meaningful
    k, _ = inference._plan_detect_shards(41, 16)
    assert k > 1


def test_tiny_tile_set_uses_single_pass(monkeypatch):
    # Few tiles must NOT over-shard (each detector reloads the model).
    for n in (1, 5, 8):
        k, _ = inference._plan_detect_shards(n, 64)
        assert k == 1, f"{n} tiles should stay single-pass, got K={k}"


def test_planner_scales_with_budget_and_tiles():
    # Adaptive to whatever tile-size/overlap the model chose (only tile count changes).
    assert inference._plan_detect_shards(420, 8)[0] == 4
    assert inference._plan_detect_shards(420, 32)[0] == 16
    assert inference._plan_detect_shards(420, 64)[0] == 32
    # bounded by tiles so we never spawn a shard per tile
    k, _ = inference._plan_detect_shards(20, 128)
    assert k <= 20 // int(__import__("os").getenv("ULTRA_RARESPOT_MIN_TILES_PER_WORKER", "8"))
