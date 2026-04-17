#!/usr/bin/env python3
"""Benchmark the dedicated code execution service under concurrent load."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from src.tooling.code_execution_jobs import build_service_submission_bundle
from src.tooling.code_execution_service_client import CodeExecutionServiceClient


@dataclass(slots=True)
class JobResult:
    job_id: str
    success: bool
    submit_latency_seconds: float
    queue_wait_seconds: float
    runtime_seconds: float
    end_to_end_seconds: float
    artifact_download_seconds: float
    status: str
    exit_code: int | None
    error_class: str | None


@dataclass(slots=True)
class PreparedJob:
    job_id: str
    temp_dir: tempfile.TemporaryDirectory[str]
    bundle_path: Path
    local_input_paths: list[Path]
    request_payload: dict[str, Any]
    artifact_name: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--workload",
        choices=("rf_csv", "image_otsu"),
        default="rf_csv",
    )
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--wait-timeout", type=float, default=1800.0)
    parser.add_argument("--health-interval", type=float, default=0.5)
    parser.add_argument("--rows", type=int, default=24000)
    parser.add_argument("--features", type=int, default=24)
    parser.add_argument("--estimators", type=int, default=160)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--cpu-limit", type=float, default=2.0)
    parser.add_argument("--memory-mb", type=int, default=4096)
    return parser.parse_args()


def _iso_delta_seconds(later: str | None, earlier: str | None) -> float:
    if not later or not earlier:
        return 0.0
    return max(
        0.0,
        (
            datetime.fromisoformat(str(later))
            - datetime.fromisoformat(str(earlier))
        ).total_seconds(),
    )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _build_rf_job(
    *, tmp_dir: Path, job_id: str, rows: int, features: int, estimators: int
) -> tuple[Path, Path, list[Path], dict[str, Any], str]:
    csv_path = tmp_dir / f"{job_id}.csv"
    source_path = tmp_dir / "job" / "source"
    source_path.mkdir(parents=True)
    (source_path / "main.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import json",
                "import matplotlib.pyplot as plt",
                "import pandas as pd",
                "from sklearn.ensemble import RandomForestClassifier",
                "",
                "df = pd.read_csv('/inputs/input_01_dataset.csv')",
                "feature_columns = [col for col in df.columns if col.startswith('x')]",
                "X = df[feature_columns]",
                "y = df['label']",
                f"model = RandomForestClassifier(n_estimators={estimators}, random_state=7, n_jobs=-1)",
                "model.fit(X, y)",
                "accuracy = float(model.score(X, y))",
                "Path('outputs').mkdir(parents=True, exist_ok=True)",
                "Path('result.json').write_text(",
                "    json.dumps({'accuracy': accuracy, 'rows': int(len(df)), 'features': int(len(feature_columns))}, indent=2),",
                "    encoding='utf-8',",
                ")",
                "plt.figure(figsize=(6, 4))",
                "plt.bar(feature_columns[: min(12, len(feature_columns))], model.feature_importances_[: min(12, len(feature_columns))])",
                "plt.xticks(rotation=45, ha='right')",
                "plt.tight_layout()",
                "plt.savefig('outputs/feature_importance.png', dpi=160)",
                "print(json.dumps({'accuracy': accuracy, 'rows': int(len(df)), 'features': int(len(feature_columns))}))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rng = random.Random(7)
    columns = [f"x{i:02d}" for i in range(features)]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([*columns, "label"])
        for index in range(rows):
            label = 1 if index % 2 else 0
            center = 1.25 if label else -1.25
            row = [
                round(rng.gauss(center + (feature_index * 0.025), 0.55), 6)
                for feature_index in range(features)
            ]
            writer.writerow([*row, label])
    job_dir = tmp_dir / "job"
    request_payload = {
        "job_id": job_id,
        "timeout_seconds": 1800,
        "cpu_limit": 2.0,
        "memory_mb": 4096,
        "expected_outputs": ["result.json", "outputs/feature_importance.png"],
        "inputs": [
            {
                "path": str(csv_path),
                "kind": "file",
                "name": csv_path.name,
                "sandbox_path": "/inputs/input_01_dataset.csv",
                "description": "Synthetic classification dataset for code execution benchmarking.",
            }
        ],
    }
    (job_dir / "job_spec.json").write_text(
        json.dumps(
            {
                **request_payload,
                "entrypoint": "main.py",
                "command": "python main.py",
                "files": [{"path": "main.py"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    bundle_path, _remote_inputs, local_inputs = build_service_submission_bundle(job_dir)
    return (
        job_dir,
        bundle_path,
        [Path(str(item["path"])) for item in local_inputs],
        request_payload,
        "result.json",
    )


def _build_image_job(
    *, tmp_dir: Path, job_id: str, image_size: int
) -> tuple[Path, Path, list[Path], dict[str, Any], str]:
    image_path = tmp_dir / f"{job_id}.pgm"
    source_path = tmp_dir / "job" / "source"
    source_path.mkdir(parents=True)
    (source_path / "main.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import imageio.v3 as iio",
                "import json",
                "import matplotlib.pyplot as plt",
                "import numpy as np",
                "from skimage import color, filters, measure",
                "",
                "image = iio.imread('/inputs/input_01_image.pgm').astype('float32') / 255.0",
                "smoothed = filters.gaussian(image, sigma=2.0, preserve_range=True)",
                "threshold = float(filters.threshold_otsu(smoothed))",
                "mask = smoothed >= threshold",
                "labels = measure.label(mask)",
                "regions = measure.regionprops(labels)",
                "Path('outputs').mkdir(parents=True, exist_ok=True)",
                "summary = {",
                "    'threshold': threshold,",
                "    'region_count': int(len(regions)),",
                "    'foreground_fraction': float(mask.mean()),",
                "}",
                "Path('result.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')",
                "overlay = color.label2rgb(labels, image=image, bg_label=0, alpha=0.24, colors=[(1.0, 0.1, 0.75)])",
                "plt.figure(figsize=(6, 6))",
                "plt.imshow(overlay)",
                "plt.axis('off')",
                "plt.tight_layout()",
                "plt.savefig('outputs/overlay.png', dpi=160)",
                "print(json.dumps(summary))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rng = random.Random(7)
    size = image_size
    image: list[list[int]] = []
    spots: list[tuple[int, int, int, int]] = []
    for _ in range(48):
        spots.append(
            (
                rng.randint(64, size - 64),
                rng.randint(64, size - 64),
                rng.randint(18, 56),
                rng.randint(96, 220),
            )
        )
    for y_index in range(size):
        row: list[int] = []
        for x_index in range(size):
            value = 18 + int(max(-4.0, min(4.0, rng.gauss(0.0, 1.0))) * 6)
            for center_y, center_x, radius, intensity in spots:
                if (y_index - center_y) ** 2 + (x_index - center_x) ** 2 <= radius**2:
                    value += intensity
            row.append(max(0, min(255, value)))
        image.append(row)
    with image_path.open("wb") as handle:
        handle.write(f"P5\n{size} {size}\n255\n".encode("ascii"))
        for row in image:
            handle.write(bytes(row))
    job_dir = tmp_dir / "job"
    request_payload = {
        "job_id": job_id,
        "timeout_seconds": 1800,
        "cpu_limit": 2.0,
        "memory_mb": 4096,
        "expected_outputs": ["result.json", "outputs/overlay.png"],
        "inputs": [
            {
                "path": str(image_path),
                "kind": "file",
                "name": image_path.name,
                "sandbox_path": "/inputs/input_01_image.pgm",
                "description": "Synthetic microscopy-like PGM for image-processing benchmarking.",
            }
        ],
    }
    (job_dir / "job_spec.json").write_text(
        json.dumps(
            {
                **request_payload,
                "entrypoint": "main.py",
                "command": "python main.py",
                "files": [{"path": "main.py"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    bundle_path, _remote_inputs, local_inputs = build_service_submission_bundle(job_dir)
    return (
        job_dir,
        bundle_path,
        [Path(str(item["path"])) for item in local_inputs],
        request_payload,
        "result.json",
    )


def _build_job(
    *,
    workload: str,
    tmp_dir: Path,
    job_id: str,
    rows: int,
    features: int,
    estimators: int,
    image_size: int,
) -> tuple[Path, Path, list[Path], dict[str, Any], str]:
    if workload == "image_otsu":
        return _build_image_job(tmp_dir=tmp_dir, job_id=job_id, image_size=image_size)
    return _build_rf_job(
        tmp_dir=tmp_dir,
        job_id=job_id,
        rows=rows,
        features=features,
        estimators=estimators,
    )


def _run_health_probe(
    *,
    base_url: str,
    api_key: str,
    interval_seconds: float,
    stop_event: threading.Event,
    samples: list[float],
    failures: list[str],
) -> None:
    headers = {"Authorization": f"Bearer {api_key}"}
    while not stop_event.wait(max(0.1, interval_seconds)):
        started = time.perf_counter()
        try:
            response = httpx.get(
                f"{base_url.rstrip('/')}/health",
                headers=headers,
                timeout=max(1.0, interval_seconds * 2.0),
            )
            response.raise_for_status()
            samples.append(time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{type(exc).__name__}: {exc}")


def _run_one_job(
    *,
    client: CodeExecutionServiceClient,
    prepared_job: PreparedJob,
    poll_interval: float,
    wait_timeout: float,
    cpu_limit: float,
    memory_mb: int,
) -> JobResult:
    tmp_dir = Path(prepared_job.temp_dir.name)
    request_payload = dict(prepared_job.request_payload)
    request_payload["cpu_limit"] = cpu_limit
    request_payload["memory_mb"] = memory_mb
    started = time.perf_counter()
    submit_started = time.perf_counter()
    try:
        submitted = client.submit_job(
            request_payload=request_payload,
            bundle_path=prepared_job.bundle_path,
            local_input_paths=prepared_job.local_input_paths,
        )
    finally:
        prepared_job.bundle_path.unlink(missing_ok=True)
    submit_latency = time.perf_counter() - submit_started
    job_token = str(submitted["job_id"])
    terminal = client.wait_for_job(
        job_id=job_token,
        poll_interval_seconds=poll_interval,
        wait_timeout_seconds=wait_timeout,
    )
    download_started = time.perf_counter()
    destination = tmp_dir / f"{prepared_job.job_id}-{Path(prepared_job.artifact_name).name}"
    client.download_artifact(
        job_id=job_token,
        artifact_name=prepared_job.artifact_name,
        destination=destination,
    )
    download_latency = time.perf_counter() - download_started
    result_payload = dict(terminal.get("result") or {})
    return JobResult(
        job_id=job_token,
        success=str(terminal.get("status") or "").strip().lower() == "succeeded",
        submit_latency_seconds=round(submit_latency, 4),
        queue_wait_seconds=round(
            _iso_delta_seconds(terminal.get("started_at"), terminal.get("created_at")),
            4,
        ),
        runtime_seconds=round(float(result_payload.get("runtime_seconds") or 0.0), 4),
        end_to_end_seconds=round(time.perf_counter() - started, 4),
        artifact_download_seconds=round(download_latency, 4),
        status=str(terminal.get("status") or ""),
        exit_code=(
            int(result_payload["exit_code"])
            if result_payload.get("exit_code") is not None
            else None
        ),
        error_class=(
            str(result_payload.get("error_class"))
            if result_payload.get("error_class") is not None
            else None
        ),
    )


def main() -> int:
    args = _parse_args()
    client = CodeExecutionServiceClient(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout_seconds=120.0,
    )

    prepared_jobs: list[PreparedJob] = []
    for index in range(max(1, int(args.jobs))):
        temp_dir = tempfile.TemporaryDirectory(prefix=f"codeexec-bench-{index:03d}-")
        tmp_path = Path(temp_dir.name)
        job_id = f"bench_{args.workload}_{index:03d}_{int(time.time())}"
        _job_dir, bundle_path, local_inputs, request_payload, artifact_name = _build_job(
            workload=args.workload,
            tmp_dir=tmp_path,
            job_id=job_id,
            rows=args.rows,
            features=args.features,
            estimators=args.estimators,
            image_size=args.image_size,
        )
        prepared_jobs.append(
            PreparedJob(
                job_id=job_id,
                temp_dir=temp_dir,
                bundle_path=bundle_path,
                local_input_paths=local_inputs,
                request_payload=request_payload,
                artifact_name=artifact_name,
            )
        )

    health_samples: list[float] = []
    health_failures: list[str] = []
    stop_event = threading.Event()
    health_thread = threading.Thread(
        target=_run_health_probe,
        kwargs={
            "base_url": args.base_url,
            "api_key": args.api_key,
            "interval_seconds": args.health_interval,
            "stop_event": stop_event,
            "samples": health_samples,
            "failures": health_failures,
        },
        daemon=True,
    )
    health_thread.start()

    benchmark_started = time.perf_counter()
    results: list[JobResult] = []
    try:
        with ThreadPoolExecutor(max_workers=max(1, int(args.concurrency))) as pool:
            futures = [
                pool.submit(
                    _run_one_job,
                    client=client,
                    prepared_job=prepared_job,
                    poll_interval=args.poll_interval,
                    wait_timeout=args.wait_timeout,
                    cpu_limit=args.cpu_limit,
                    memory_mb=args.memory_mb,
                )
                for prepared_job in prepared_jobs
            ]
            for future in as_completed(futures):
                results.append(future.result())
    finally:
        stop_event.set()
        health_thread.join(timeout=2.0)
        for prepared_job in prepared_jobs:
            prepared_job.bundle_path.unlink(missing_ok=True)
            prepared_job.temp_dir.cleanup()

    duration_seconds = time.perf_counter() - benchmark_started
    successful = [item for item in results if item.success]
    failed = [item for item in results if not item.success]

    def summarize(field: str) -> dict[str, float]:
        values = [float(getattr(item, field)) for item in results]
        return {
            "mean": round(statistics.fmean(values), 4) if values else 0.0,
            "p50": round(_percentile(values, 0.50), 4),
            "p95": round(_percentile(values, 0.95), 4),
            "max": round(max(values), 4) if values else 0.0,
        }

    summary = {
        "workload": args.workload,
        "jobs": len(results),
        "concurrency": int(args.concurrency),
        "success_count": len(successful),
        "failure_count": len(failed),
        "success_rate": round(len(successful) / len(results), 4) if results else 0.0,
        "total_duration_seconds": round(duration_seconds, 4),
        "throughput_jobs_per_minute": round((len(results) / duration_seconds) * 60.0, 4)
        if duration_seconds > 0
        else 0.0,
        "submit_latency_seconds": summarize("submit_latency_seconds"),
        "queue_wait_seconds": summarize("queue_wait_seconds"),
        "runtime_seconds": summarize("runtime_seconds"),
        "end_to_end_seconds": summarize("end_to_end_seconds"),
        "artifact_download_seconds": summarize("artifact_download_seconds"),
        "health_latency_seconds": {
            "samples": len(health_samples),
            "mean": round(statistics.fmean(health_samples), 4) if health_samples else 0.0,
            "p95": round(_percentile(health_samples, 0.95), 4),
            "max": round(max(health_samples), 4) if health_samples else 0.0,
            "failure_count": len(health_failures),
        },
        "failed_jobs": [asdict(item) for item in failed[:10]],
        "health_failures": health_failures[:10],
        "job_results": [asdict(item) for item in sorted(results, key=lambda item: item.job_id)],
    }
    print(json.dumps(summary, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
