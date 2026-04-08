#!/usr/bin/env python3
"""Run BisQue modules end-to-end using bqapi from a local Python client.

This script is intended to mirror notebook/local-script usage:
- authenticate with BQSession
- upload input resources
- execute a module through module_service
- poll MEX status until completion
- resolve output image URI(s)
- download output image blobs to local disk
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import gzip
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Iterable

from lxml import etree
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
BQAPI_PATH = REPO_ROOT / "source" / "bqapi"
if str(BQAPI_PATH) not in sys.path:
    sys.path.insert(0, str(BQAPI_PATH))

from bqapi import BQSession  # noqa: E402
from bqapi.util import fetch_blob  # noqa: E402


@dataclasses.dataclass
class RunResult:
    task_id: str
    image_path: str
    upload_path: str | None = None
    upload_uri: str | None = None
    mex_uri: str | None = None
    statuses: list[str] = dataclasses.field(default_factory=list)
    final_status: str | None = None
    output_image_uri: str | None = None
    output_local_path: str | None = None
    seconds: float = 0.0
    success: bool = False
    error: str | None = None


def _normalize_root(root: str) -> str:
    return root.rstrip("/")


def _parse_upload_uri(upload_xml_bytes: bytes | str) -> str:
    tree = etree.XML(upload_xml_bytes)
    if tree.get("uri"):
        return tree.get("uri")
    uploaded = tree.find("./*")
    if uploaded is None or uploaded.get("uri") is None:
        errors = [node.get("value", "") for node in tree.xpath('.//tag[@name="error"]')]
        raise RuntimeError(f"Upload did not return a resource URI; errors={errors}")
    return uploaded.get("uri")


def _prepare_upload_path(image_path: Path, prep_dir: Path, task_id: str) -> Path:
    """Normalize compressed-looking inputs to avoid downstream client gzip bugs.

    Behavior:
    - non-.gz files: upload as-is
    - .gz files:
      - if valid gzip stream: decompress to prep_dir/<task_id>_<stem>
      - otherwise: byte-copy to prep_dir/<task_id>_<stem> (strip .gz only)
    """
    if image_path.suffix.lower() != ".gz":
        return image_path

    prep_dir.mkdir(parents=True, exist_ok=True)
    normalized = prep_dir / f"{task_id}_{image_path.stem}"
    try:
        with gzip.open(image_path, "rb") as src, normalized.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    except OSError:
        shutil.copyfile(image_path, normalized)
    return normalized


def _binary_fallback_png(input_path: Path, prep_dir: Path, task_id: str, side: int = 512) -> Path:
    """Create a deterministic grayscale PNG from raw bytes for upload fallback."""
    prep_dir.mkdir(parents=True, exist_ok=True)
    raw = input_path.read_bytes()
    if not raw:
        raw = b"\x00"
    total = side * side
    repeats = (total + len(raw) - 1) // len(raw)
    pixels = (raw * repeats)[:total]
    out = prep_dir / f"{task_id}_{input_path.stem}_fallback.png"
    Image.frombytes("L", (side, side), pixels).save(out, format="PNG")
    return out


def _execute_module(session: BQSession, module_name: str, input_uri: str) -> str:
    mex = etree.Element("mex")
    inputs = etree.SubElement(mex, "tag", name="inputs")
    etree.SubElement(inputs, "tag", name="Input Image", type="resource", value=input_uri)
    execute_response = session.postxml(f"/module_service/{module_name}/execute", mex)
    if execute_response is None or execute_response.get("uri") is None:
        raise RuntimeError(f"Module execute for {module_name} did not return a MEX URI")
    return execute_response.get("uri")


def _poll_mex(session: BQSession, mex_uri: str, timeout: int, poll_interval: float) -> tuple[etree._Element, list[str]]:
    deadline = time.time() + timeout
    statuses: list[str] = []
    last = None
    final_mex = None
    while time.time() < deadline:
        final_mex = session.fetchxml(mex_uri, view="deep")
        status = final_mex.get("value", "")
        if status != last:
            statuses.append(status)
            last = status
        if status in {"FINISHED", "FAILED"}:
            return final_mex, statuses
        time.sleep(poll_interval)
    raise TimeoutError(f"MEX timeout after {timeout}s (last_status={last})")


def _extract_output_image_uri(final_mex: etree._Element) -> str:
    # Preferred: output tag with type=image and a value URI.
    for node in final_mex.xpath('./tag[@name="outputs"]//tag[@type="image"]'):
        value = node.get("value", "")
        if value:
            return value
        uri = node.get("uri", "")
        if uri:
            return uri

    # Alternate: concrete image nodes.
    for node in final_mex.xpath('./tag[@name="outputs"]//image'):
        uri = node.get("uri", "")
        if uri:
            return uri
        value = node.get("value", "")
        if value:
            return value

    raise RuntimeError("No output image URI found in MEX outputs")


def _download_output_blob(session: BQSession, output_image_uri: str, output_dir: Path, task_id: str) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = fetch_blob(session, output_image_uri, dest=str(output_dir))
    local_path = downloaded.get(output_image_uri) if isinstance(downloaded, dict) else None
    if not local_path:
        raise RuntimeError(f"Failed to download output blob for {output_image_uri}")
    src = Path(local_path)
    dst = output_dir / f"{task_id}_{src.name}"
    if src.resolve() != dst.resolve():
        dst.write_bytes(src.read_bytes())
    return str(dst)


def _collect_error_messages(final_mex: etree._Element) -> list[str]:
    return [tag.get("value", "") for tag in final_mex.xpath('.//tag[@name="error_message"]')]


def run_single_task(
    bisque_root: str,
    user: str,
    password: str,
    token: str | None,
    module_name: str,
    image_path: Path,
    output_dir: Path,
    timeout: int,
    poll_interval: float,
    task_id: str,
    prep_dir: Path,
) -> RunResult:
    start = time.time()
    result = RunResult(task_id=task_id, image_path=str(image_path))
    session = None
    try:
        if token:
            session = BQSession().init_token(token, bisque_root=bisque_root, create_mex=False)
        else:
            session = BQSession().init_local(user, password, bisque_root=bisque_root, create_mex=False)
        upload_path = _prepare_upload_path(image_path, prep_dir=prep_dir, task_id=task_id)
        result.upload_path = str(upload_path)
        resource_xml = etree.Element("image", name=upload_path.name)
        upload_response = session.postblob(str(upload_path), xml=resource_xml)
        try:
            upload_uri = _parse_upload_uri(upload_response)
        except RuntimeError:
            # Some binary inputs are rejected by ingest; convert bytes to PNG for upload.
            fallback_path = _binary_fallback_png(upload_path, prep_dir=prep_dir, task_id=task_id)
            resource_xml = etree.Element("image", name=fallback_path.name)
            upload_response = session.postblob(str(fallback_path), xml=resource_xml)
            upload_uri = _parse_upload_uri(upload_response)
            result.upload_path = str(fallback_path)
        result.upload_uri = upload_uri

        mex_uri = _execute_module(session, module_name, upload_uri)
        result.mex_uri = mex_uri

        final_mex, statuses = _poll_mex(session, mex_uri, timeout=timeout, poll_interval=poll_interval)
        result.statuses = statuses
        result.final_status = final_mex.get("value", "")

        if result.final_status != "FINISHED":
            errors = _collect_error_messages(final_mex)
            raise RuntimeError(f"MEX finished with status={result.final_status}; errors={errors}")

        output_image_uri = _extract_output_image_uri(final_mex)
        result.output_image_uri = output_image_uri
        result.output_local_path = _download_output_blob(session, output_image_uri, output_dir, task_id)
        result.success = True
        return result
    except Exception as exc:  # noqa: BLE001
        result.error = f"{exc}\n{traceback.format_exc()}"
        return result
    finally:
        result.seconds = round(time.time() - start, 3)
        if session is not None:
            session.close()


def _iter_tasks(images: Iterable[Path], repeat: int) -> list[tuple[str, Path]]:
    tasks: list[tuple[str, Path]] = []
    for image_idx, image in enumerate(images, start=1):
        for rep in range(1, repeat + 1):
            task_id = f"img{image_idx:02d}_run{rep:03d}"
            tasks.append((task_id, image))
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BisQue module(s) via bqapi and download outputs")
    parser.add_argument("--bisque-root", default="http://127.0.0.1:8080", help="BisQue root URL")
    parser.add_argument("--user", default="admin")
    parser.add_argument("--password", default="admin")
    parser.add_argument("--token", default=None, help="Bearer access token (overrides user/password)")
    parser.add_argument("--module", default="EdgeDetection", help="Registered module name")
    parser.add_argument("--images", nargs="+", required=True, help="Input image/file paths")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "artifacts" / "bqapi_outputs"))
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--repeat", type=int, default=1, help="Repetitions per input")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--summary-json", default=None, help="Path to write JSON summary")
    parser.add_argument("--allow-failures", action="store_true", help="Exit 0 even if some tasks fail")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bisque_root = _normalize_root(args.bisque_root)

    image_paths = [Path(p).resolve() for p in args.images]
    missing = [str(p) for p in image_paths if not p.exists()]
    if missing:
        print(json.dumps({"ok": False, "error": f"Missing input paths: {missing}"}, indent=2))
        return 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir).resolve() / f"{args.module}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    prep_dir = run_dir / "prepared_inputs"
    prep_dir.mkdir(parents=True, exist_ok=True)

    tasks = _iter_tasks(image_paths, repeat=args.repeat)
    results: list[RunResult] = []

    print(f"bqapi_runner_start module={args.module} tasks={len(tasks)} concurrency={args.concurrency} root={bisque_root}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        future_map = {
            executor.submit(
                run_single_task,
                bisque_root=bisque_root,
                user=args.user,
                password=args.password,
                token=args.token,
                module_name=args.module,
                image_path=image,
                output_dir=run_dir,
                timeout=args.timeout_seconds,
                poll_interval=args.poll_interval,
                task_id=task_id,
                prep_dir=prep_dir,
            ): (task_id, image)
            for task_id, image in tasks
        }
        for future in concurrent.futures.as_completed(future_map):
            result = future.result()
            results.append(result)
            status_chain = " -> ".join(result.statuses)
            if result.success:
                print(
                    f"task={result.task_id} ok=1 mex={result.mex_uri} "
                    f"status_chain=[{status_chain}] output={result.output_local_path} sec={result.seconds}"
                )
            else:
                print(
                    f"task={result.task_id} ok=0 mex={result.mex_uri} "
                    f"status_chain=[{status_chain}] error={result.error.splitlines()[0] if result.error else 'unknown'} sec={result.seconds}"
                )

    results.sort(key=lambda r: r.task_id)
    success_count = sum(1 for r in results if r.success)
    failure_count = len(results) - success_count
    avg_seconds = round(sum(r.seconds for r in results) / len(results), 3) if results else 0.0

    summary = {
        "ok": failure_count == 0,
        "module": args.module,
        "bisque_root": bisque_root,
        "run_dir": str(run_dir),
        "tasks_total": len(results),
        "tasks_success": success_count,
        "tasks_failed": failure_count,
        "avg_seconds": avg_seconds,
        "results": [dataclasses.asdict(r) for r in results],
    }

    summary_path = Path(args.summary_json).resolve() if args.summary_json else (run_dir / "summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"summary_json={summary_path}")
    print(json.dumps({k: summary[k] for k in ["ok", "tasks_total", "tasks_success", "tasks_failed", "avg_seconds"]}, indent=2))

    if failure_count > 0 and not args.allow_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
