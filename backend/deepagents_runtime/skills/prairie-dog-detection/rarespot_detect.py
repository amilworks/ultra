#!/usr/bin/env python3
"""RareSpot prairie-dog / burrow detection CLI for the code sandbox.

This is a THIN wrapper over the production trust pipeline
(``rarespot.inference.run_rarespot_inference``) — it does NOT reimplement tiling,
NMS, the perturbation-consensus stability scoring, the spectral mask, or the
survey-map logic. The calibrated pipeline is the single source of truth; this CLI
only resolves paths, runs it over the staged images, and prints a compact summary.

In the prairie-dog-detection sandbox image the model, runtime and package are baked
at ``/opt/rarespot`` (weights ``/opt/rarespot/RareSpotWeights.pt``, yolov5
``/opt/rarespot/yolov5``, package on ``PYTHONPATH=/opt/rarespot``). The agent stages
the image(s) into ``/workspace`` and runs:

    YOLOv5_AUTOINSTALL=false python /opt/rarespot/rarespot_detect.py \
        --images /workspace/staged_resources --weights /opt/rarespot/RareSpotWeights.pt \
        --yolov5 /opt/rarespot/yolov5 --out /outputs/rarespot_run

Researcher artifacts (overlays, detections.csv, predictions.json, reliability
report, survey map) land under ``--out`` (collected by the backend's normal
artifact path); scratch (tiles, logs) stays under ``--out/tiles`` etc.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def _collect_images(images_args: list[str]) -> list[Path]:
    """Resolve --images entries (files or directories) into a sorted image list.

    Sorted for determinism so the tile manifest order is reproducible (the parity
    gate depends on this)."""
    found: list[Path] = []
    seen: set[str] = set()
    for raw in images_args:
        path = Path(raw).expanduser()
        candidates: list[Path]
        if path.is_dir():
            candidates = [p for p in path.rglob("*") if p.is_file()]
        else:
            candidates = [path]
        for candidate in candidates:
            if candidate.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            resolved = candidate.resolve()
            key = str(resolved)
            if key in seen or not resolved.is_file():
                continue
            seen.add(key)
            found.append(resolved)
    return sorted(found, key=lambda p: str(p))


def _summary(result: dict, out_dir: Path) -> dict:
    """Compact, model-facing summary: counts, trust buckets, spectral review picks,
    and researcher-facing artifact paths (relative to the output dir)."""
    stability = result.get("stability_summary") or {}
    artifacts = []
    for art in result.get("artifacts") or []:
        rel = str(art.get("relative_path") or art.get("path") or "").strip()
        if rel:
            artifacts.append(
                {"title": art.get("title"), "kind": art.get("kind"), "path": rel}
            )
    return {
        "ok": True,
        "image_count": len(result.get("predictions") or []),
        "counts_by_class": result.get("counts_by_class") or {},
        "total_detections": sum((result.get("counts_by_class") or {}).values()),
        "stability": {
            "label_counts": stability.get("label_counts") or {},
            "trusted_fraction": stability.get("trusted_fraction"),
            "trials": stability.get("trials"),
        },
        "top_spectral_review_candidates": result.get("top_spectral_review_candidates") or [],
        "geospatial": {
            "georeferenced_images": (result.get("geospatial_summary") or {}).get(
                "georeferenced_image_count"
            ),
        },
        "configuration": result.get("configuration") or {},
        "output_dir": str(out_dir),
        "artifacts": artifacts,
        "caveat": (
            "RareSpot over-detects and has no held-out validation set; confidence "
            "and stability are TRIAGE signals, not ground truth. Hand-verify "
            "borderline/unstable detections."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RareSpot prairie-dog detection (sandbox).")
    parser.add_argument("--images", nargs="+", required=True, help="Image file(s) or directory(ies) to scan.")
    parser.add_argument("--weights", default="/opt/rarespot/RareSpotWeights.pt")
    parser.add_argument("--yolov5", default="/opt/rarespot/yolov5")
    parser.add_argument("--out", default="/outputs/rarespot_run")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--stability", dest="stability", action="store_true", default=True)
    parser.add_argument("--no-stability", dest="stability", action="store_false")
    parser.add_argument("--spectral", dest="spectral", action="store_true", default=True)
    parser.add_argument("--no-spectral", dest="spectral", action="store_false")
    args = parser.parse_args(argv)

    # Pre-baked deps; never let YOLOv5 try to pip-install under --network none.
    os.environ.setdefault("YOLOv5_AUTOINSTALL", "false")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    # The minimal package is baked at /opt/rarespot/ultra_deepagents (preserving the
    # ultra_deepagents.rarespot import path); add /opt/rarespot to the path so the
    # CLI also runs from a real checkout (the parity test) without env wiring.
    sys.path.insert(0, str(Path(args.yolov5).resolve().parent))
    from ultra_deepagents.rarespot.config import RareSpotConfig  # noqa: E402
    from ultra_deepagents.rarespot.inference import run_rarespot_inference  # noqa: E402

    images = _collect_images(args.images)
    if not images:
        print(json.dumps({"ok": False, "error": "no_images_found", "searched": args.images}))
        return 2

    out_dir = Path(args.out).expanduser().resolve()
    # Allow reading staged images from anywhere under /workspace and the image dirs.
    allowed_roots = tuple({str(p.parent) for p in images} | {"/workspace"})
    config = RareSpotConfig.from_paths(
        weights_path=args.weights,
        yolov5_path=args.yolov5,
        artifact_root=out_dir,
        allowed_input_roots=allowed_roots,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        conf=args.conf,
        iou=args.iou,
        spectral=args.spectral,
        stability=args.stability,
    )
    result = run_rarespot_inference(
        image_paths=images,
        run_id=f"rarespot-skill-{uuid.uuid4().hex[:12]}",
        thread_id="rarespot-skill",
        output_dir=out_dir,
        config=config,
    )
    print(json.dumps(_summary(result, out_dir), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
