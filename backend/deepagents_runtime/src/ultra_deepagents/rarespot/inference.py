from __future__ import annotations

import concurrent.futures
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

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
        # Only tiles that carried a raw detection can contribute a stability score, so
        # restrict the perturbation passes (the dominant cost) to them. YOLOv5 --save-txt
        # writes a label file only for a tile with >=1 detection; this is the same labels
        # dir the predictions were parsed from, so the set is consistent with them.
        detection_tile_stems = {
            txt.stem
            for txt in Path(labels_dir).glob("*.txt")
            if txt.is_file() and txt.stat().st_size > 0
        }
        stability_summary = run_detection_stability(
            predictions=predictions,
            source_dir=source_dir,
            output_dir=output_dir,
            tile_manifest=tile_manifest,
            detect_fn=lambda **kwargs: run_yolov5_detect(output_dir=output_dir, config=config, **kwargs),
            match_iou=config.stability_match_iou,
            iou_threshold=config.iou,
            detection_tile_stems=detection_tile_stems,
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


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default))).strip() or default)
    except (TypeError, ValueError):
        return default


def _effective_cpu_budget() -> int:
    """Cores this process may actually use — the cgroup CPU quota, NOT ``os.cpu_count()``,
    which reports every host core even inside a ``docker run --cpus=N`` container. Sizing
    parallelism off ``os.cpu_count()`` on a 384-core host under an 8-core quota would spawn
    hundreds of workers thrashing 8 cores of CPU time. Falls back to the affinity mask, then
    the host count, when no quota is set (e.g. the bare analysis worker)."""
    try:  # cgroup v2
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts and parts[0] not in ("", "max"):
            quota = int(parts[0])
            period = int(parts[1]) if len(parts) > 1 else 100000
            if quota > 0 and period > 0:
                return max(1, quota // period)
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text().strip())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip())
        if quota > 0 and period > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _plan_detect_shards(num_tiles: int, cpu_budget: int) -> tuple[int, int]:
    """Plan how many parallel detector processes (K) to fan the tiles across, and the CPU
    threads each gets. Detection over independent tiles is embarrassingly parallel, while a
    single 512px YOLOv5s inference plateaus at ~1-2 CPU threads (memory-bandwidth bound), so
    the budget buys MORE shards (parallel tiles) rather than more threads-per-inference —
    that scales near-linearly where per-inference threads do not. Bounds:
      - K is capped so each shard gets >= MIN_TILES_PER_WORKER tiles: every detector reloads
        the model (a fixed cost), so a shard-per-tile would waste it (and its memory).
      - K*threads ~ the CPU budget, so shards never oversubscribe the quota.
      - K==1 (few tiles, or a 1-core quota) => the original single-process pass, unchanged.
    Fully adaptive: any tile-size/overlap the model chose only changes ``num_tiles``; an
    operator can pin K via ULTRA_RARESPOT_DETECT_WORKERS or the threads knob."""
    threads_per = max(1, _env_int("ULTRA_RARESPOT_DETECT_THREADS_PER_WORKER", 2))
    min_tiles_per = max(1, _env_int("ULTRA_RARESPOT_MIN_TILES_PER_WORKER", 8))
    forced = _env_int("ULTRA_RARESPOT_DETECT_WORKERS", 0)  # 0 = auto from the CPU quota
    k_cap = forced if forced > 0 else max(1, max(1, cpu_budget) // threads_per)
    k_by_tiles = max(1, num_tiles // min_tiles_per)
    k = max(1, min(k_cap, k_by_tiles, max(1, num_tiles)))
    return k, threads_per


def _detect_command(*, source: Path, project: Path, name: str, config: RareSpotConfig) -> list[str]:
    return [
        sys.executable,
        str(config.yolov5_path / "detect.py"),
        "--weights", str(config.weights_path),
        "--source", str(source),
        "--imgsz", str(config.tile_size),
        "--project", str(project),
        "--name", name,
        "--exist-ok", "--save-txt", "--save-conf", "--nosave",
        "--conf-thres", str(config.conf),
        "--iou-thres", str(config.iou),
    ]


def run_yolov5_detect(
    *,
    source_dir: Path,
    output_dir: Path,
    config: RareSpotConfig,
    project_subdir: str = "yolov5",
    name: str = "predict",
) -> Path:
    """Run YOLOv5 detect.py over a tile directory; returns the labels output dir.
    `project_subdir` lets the main and stability passes write to distinct trees.

    The tile set is fanned across K parallel detector processes sized to the sandbox's CPU
    quota (see ``_plan_detect_shards``). Per-tile detection is independent + deterministic
    and cross-tile NMS runs later in image coordinates, so the merged per-tile labels are
    IDENTICAL to a single sequential pass — a pure throughput win, not a behaviour change.
    K==1 (few tiles, or a 1-core quota) runs the original single pass unchanged, which is
    what the parity-golden small-image test exercises."""
    base_env = os.environ.copy()
    base_env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    # YOLOv5's detect.py runs check_requirements() at startup, which pip-installs any missing
    # requirement. That dies in the code sandbox (--network none), so disable it; the runtime
    # deps (torch, torchvision, ...) are pre-baked into the sandbox image.
    base_env["YOLOv5_AUTOINSTALL"] = "false"
    base_env["YOLOV5_CONFIG_DIR"] = str((output_dir / ".yolov5-config").resolve())
    base_env["PYTHONPATH"] = f"{config.yolov5_path}{os.pathsep}{base_env.get('PYTHONPATH', '')}"

    labels_dir = output_dir / project_subdir / name / "labels"
    tiles = sorted(source_dir.glob("*.jpg"))
    k, threads_per = _plan_detect_shards(len(tiles), _effective_cpu_budget())

    if k <= 1:
        result = subprocess.run(
            _detect_command(source=source_dir, project=output_dir / project_subdir, name=name, config=config),
            cwd=str(config.yolov5_path), env=base_env, text=True, capture_output=True, check=False,
        )
        (output_dir / f"{project_subdir}.stdout.log").write_text(result.stdout or "", encoding="utf-8")
        (output_dir / f"{project_subdir}.stderr.log").write_text(result.stderr or "", encoding="utf-8")
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip()[-4000:]
            raise RuntimeError(f"YOLOv5 inference failed: {tail}")
        return labels_dir

    # --- sharded path: split tiles into K balanced groups, detect in parallel, merge -------
    shard_root = output_dir / f"{project_subdir}.shards"
    shard_inputs: list[Path] = []
    for i in range(k):
        d = shard_root / f"in_{i:03d}"
        d.mkdir(parents=True, exist_ok=True)
        shard_inputs.append(d)
    # Round-robin so groups are balanced regardless of tile ordering; deterministic (tiles
    # are sorted) so a rerun is reproducible.
    for idx, tile in enumerate(tiles):
        link = shard_inputs[idx % k] / tile.name
        if link.exists():
            continue
        try:
            os.symlink(tile.resolve(), link)
        except OSError:
            try:
                os.link(tile, link)  # same-filesystem hardlink, no data copy
            except OSError:
                shutil.copy2(tile, link)

    def _run_shard(i: int) -> subprocess.CompletedProcess[str]:
        env = dict(base_env)
        # Cap each shard's math threads so K shards ~ the CPU quota (no oversubscription),
        # and give each its own YOLOv5 config dir so concurrent startups don't race.
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[var] = str(threads_per)
        # Per-shard config dir so concurrent YOLOv5 startups don't race on settings.yaml.
        # Keep it directly under shard_root (which exists): YOLOv5 does mkdir(exist_ok=True)
        # WITHOUT parents, so the parent must already exist.
        env["YOLOV5_CONFIG_DIR"] = str((shard_root / f".yolov5-config-{i:03d}").resolve())
        return subprocess.run(
            _detect_command(source=shard_inputs[i], project=shard_root / f"out_{i:03d}", name=name, config=config),
            cwd=str(config.yolov5_path), env=env, text=True, capture_output=True, check=False,
        )

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=k) as pool:
        for i, result in enumerate(pool.map(_run_shard, range(k))):
            stdout_parts.append(f"# shard {i}\n{result.stdout or ''}")
            stderr_parts.append(f"# shard {i}\n{result.stderr or ''}")
            if result.returncode != 0:
                failures.append((result.stderr or result.stdout or "").strip()[-1000:])

    (output_dir / f"{project_subdir}.stdout.log").write_text("\n".join(stdout_parts), encoding="utf-8")
    (output_dir / f"{project_subdir}.stderr.log").write_text("\n".join(stderr_parts), encoding="utf-8")
    if failures:
        raise RuntimeError(f"YOLOv5 sharded inference failed in {len(failures)}/{k} shard(s): {failures[0]}")

    # Merge per-tile labels into the canonical dir. Each tile is in exactly one shard, so
    # stems never collide — the merged set equals a single sequential pass.
    labels_dir.mkdir(parents=True, exist_ok=True)
    for i in range(k):
        shard_labels = shard_root / f"out_{i:03d}" / name / "labels"
        if not shard_labels.is_dir():
            continue
        for txt in shard_labels.glob("*.txt"):
            shutil.move(str(txt), str(labels_dir / txt.name))
    return labels_dir


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
