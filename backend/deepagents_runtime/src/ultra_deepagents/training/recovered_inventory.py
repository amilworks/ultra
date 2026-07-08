"""Build the recovered-train-v0 inventory — GoldGate M1 freeze input #1.

The inventory is the DEFINITIVE trained-on exclusion list (plan section 5e):
every gold freeze consumes it through the RareSpot adapter's identity checks,
and the M3 replay pool seeds from it. Exclusion by identity beats exclusion by
review-timestamp. The output is content-hashed (inventory_sha256) and destined
for WORM storage at
the shared training store (corpus/<model_key>/recovered-train-v0/)
— the local run produces the artifact; the copy to the store is a deploy step.

Per-tile rows: {relpath, run, stem, content_sha256, phash, label_sha256|null,
per_class_box_count}. Headers: phash_kernel (the freeze-time hasher asserts
kernel equality and refuses on mismatch), ENUMERATED stem sets for run1/run2
(never assumed ranges), the run3 block-coordinate list, per-run trained_on
(presumed until the checkpoint-opt verification writes a verdict), and a
box-size scan (freeze input #4: per-class quantiles that resolve the strata
cut points, plan section 4.2).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultra_deepagents.training.phash import PHASH_KERNEL, phash64_file

RUNS = ("run1_tiles", "run2_tiles", "run3_tiles")
CLASS_NAMES = {0: "prairie_dog", 1: "burrow"}
TILE_SUFFIX_RE = re.compile(r"^(?P<stem>.+)_(?P<row>\d+)_(?P<col>\d+)$")
RUN3_BLOCK_RE = re.compile(r"^tile_(?P<a>\d+)_(?P<b>\d+)$")


@dataclass
class TileRow:
    relpath: str
    run: str
    stem: str
    content_sha256: str
    phash: str
    label_sha256: str | None
    per_class_box_count: dict[str, int]
    box_sizes_px: list[tuple[str, float, float]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_label(path: Path, tile_px: int) -> tuple[dict[str, int], list[tuple[str, float, float]]]:
    counts: Counter[str] = Counter()
    sizes: list[tuple[str, float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            width = float(parts[3]) * tile_px
            height = float(parts[4]) * tile_px
        except ValueError:
            continue
        name = CLASS_NAMES.get(class_id, f"class_{class_id}")
        counts[name] += 1
        sizes.append((name, width, height))
    return dict(counts), sizes


def _tile_stem(filename: str) -> str:
    match = TILE_SUFFIX_RE.match(Path(filename).stem)
    return match.group("stem") if match else Path(filename).stem


def _process_tile(args: tuple[Path, Path, str, int]) -> TileRow:
    image_path, dataset_dir, run, tile_px = args
    relpath = str(image_path.relative_to(dataset_dir))
    label_path = dataset_dir / "labels" / run / (image_path.stem + ".txt")
    label_sha: str | None = None
    counts: dict[str, int] = {}
    sizes: list[tuple[str, float, float]] = []
    if label_path.is_file():
        label_sha = _sha256_file(label_path)
        counts, sizes = _parse_label(label_path, tile_px)
    return TileRow(
        relpath=relpath,
        run=run,
        stem=_tile_stem(image_path.name),
        content_sha256=_sha256_file(image_path),
        phash=phash64_file(image_path),
        label_sha256=label_sha,
        per_class_box_count=counts,
        box_sizes_px=sizes,
    )


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def pick(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
        return round(ordered[index], 2)

    return {
        "count": len(ordered),
        "min": round(ordered[0], 2),
        "q10": pick(0.10),
        "q25": pick(0.25),
        "median": pick(0.50),
        "q75": pick(0.75),
        "q90": pick(0.90),
        "max": round(ordered[-1], 2),
    }


def build_inventory(dataset_dir: Path, tile_px: int = 512, workers: int = 8) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    jobs: list[tuple[Path, Path, str, int]] = []
    for run in RUNS:
        run_dir = dataset_dir / "images" / run
        if not run_dir.is_dir():
            raise FileNotFoundError(f"expected run directory missing: {run_dir}")
        for image_path in sorted(run_dir.glob("*.jpg")):
            jobs.append((image_path, dataset_dir, run, tile_px))

    rows: list[TileRow] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(_process_tile, jobs, chunksize=64):
            rows.append(row)
    rows.sort(key=lambda row: row.relpath)

    stems: dict[str, list[str]] = {run: [] for run in RUNS}
    seen_stems: dict[str, set[str]] = {run: set() for run in RUNS}
    per_run_counts: dict[str, Counter[str]] = {run: Counter() for run in RUNS}
    per_run_tiles: Counter[str] = Counter()
    per_run_labeled: Counter[str] = Counter()
    box_widths: dict[str, list[float]] = {}
    box_heights: dict[str, list[float]] = {}
    for row in rows:
        per_run_tiles[row.run] += 1
        if row.label_sha256 is not None:
            per_run_labeled[row.run] += 1
        if row.stem not in seen_stems[row.run]:
            seen_stems[row.run].add(row.stem)
            stems[row.run].append(row.stem)
        for name, count in row.per_class_box_count.items():
            per_run_counts[row.run][name] += count
        for name, width, height in row.box_sizes_px:
            box_widths.setdefault(name, []).append(width)
            box_heights.setdefault(name, []).append(height)

    run3_blocks = []
    for stem in stems["run3_tiles"]:
        match = RUN3_BLOCK_RE.match(stem)
        if match:
            run3_blocks.append({"a": int(match.group("a")), "b": int(match.group("b"))})

    inventory: dict[str, Any] = {
        "corpus": "recovered-train-v0",
        "model_key": "yolov5_rarespot",
        "source_dir": str(dataset_dir),
        "phash_kernel": PHASH_KERNEL,
        "tile_px": tile_px,
        # The archival destination is deployment config (the shared training
        # store URI); the local run only produces the artifact.
        "worm_destination": os.getenv(
            "ULTRA_TRAINING_CORPUS_WORM_URI",
            "store:/training/corpus/yolov5_rarespot/recovered-train-v0/",
        ),
        "runs": {
            run: {
                # presumed = treat as trained-on; the checkpoint-opt
                # verification (freeze input #2) upgrades this to
                # verified_true/verified_false where it can.
                "trained_on": "presumed",
                "tile_count": per_run_tiles[run],
                "labeled_tile_count": per_run_labeled[run],
                "frame_stem_count": len(stems[run]),
                "per_class_box_count": dict(per_run_counts[run]),
            }
            for run in RUNS
        },
        "frame_stems": {run: stems[run] for run in ("run1_tiles", "run2_tiles")},
        "run3_blocks": run3_blocks,
        "strata_scan": {
            "note": "box sizes in pixels at the 512px tile scale (native crops - no rescale); resolves the section-4.2 strata cut points at freeze",
            "box_width_px": {name: _quantiles(values) for name, values in sorted(box_widths.items())},
            "box_height_px": {name: _quantiles(values) for name, values in sorted(box_heights.items())},
        },
        "exif_frame_centers": [],
        "tiles": [
            {
                "relpath": row.relpath,
                "run": row.run,
                "stem": row.stem,
                "content_sha256": row.content_sha256,
                "phash": row.phash,
                "label_sha256": row.label_sha256,
                "per_class_box_count": row.per_class_box_count,
            }
            for row in rows
        ],
    }
    return inventory


def write_inventory(inventory: dict[str, Any], out_dir: Path) -> tuple[Path, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "inventory.json"
    payload = json.dumps(inventory, indent=1, sort_keys=True)
    out_path.write_text(payload, encoding="utf-8")
    sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    (out_dir / "inventory.sha256").write_text(sha + "\n", encoding="utf-8")
    return out_path, sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, help="recovered dataset root (images/ + labels/)")
    parser.add_argument("out_dir", type=Path, help="output directory for inventory.json + inventory.sha256")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    inventory = build_inventory(args.dataset_dir, workers=args.workers)
    out_path, sha = write_inventory(inventory, args.out_dir)
    summary = {
        run: {
            "tiles": inventory["runs"][run]["tile_count"],
            "labeled": inventory["runs"][run]["labeled_tile_count"],
            "stems": inventory["runs"][run]["frame_stem_count"],
        }
        for run in RUNS
    }
    print(json.dumps({"inventory": str(out_path), "inventory_sha256": sha, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
