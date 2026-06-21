"""HTTP clients for the batch inference worker: the GPU MegaSeg service and the control
plane's output-registration / job-status endpoints. All calls are synchronous (the worker
invokes them via asyncio.to_thread)."""

from __future__ import annotations

import json
import os
import tarfile
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any

import requests

_CHUNK = 1024 * 1024


def _safe_extract(archive: tarfile.TarFile, dest_dir: Path) -> None:
    dest_resolved = dest_dir.resolve()
    for member in archive.getmembers():
        target = (dest_dir / member.name).resolve()
        if target != dest_resolved and not str(target).startswith(f"{dest_resolved}{os.sep}"):
            raise RuntimeError(f"refusing to extract unsafe archive member: {member.name}")
    archive.extractall(dest_dir)


def run_megaseg_infer(
    *,
    service_url: str,
    api_key: str,
    image_path: Path,
    params: dict[str, Any],
    timeout: float,
    dest_dir: Path,
) -> dict[str, Any]:
    """POST one image to the GPU /v1/infer, stream the result tarball into dest_dir, and
    return the parsed result.json (run_megaseg_batch output with archive-relative paths).
    Raises RuntimeError on a non-200 response."""
    url = f"{service_url.rstrip('/')}/v1/infer"
    headers = {"Authorization": f"Bearer {api_key}"}
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp_tar_fd, tmp_tar_name = tempfile.mkstemp(prefix="megaseg-", suffix=".tar.gz")
    os.close(tmp_tar_fd)
    tmp_tar = Path(tmp_tar_name)
    try:
        with open(image_path, "rb") as image_handle:
            files = {
                "params": (None, json.dumps(params), "application/json"),
                "file": (Path(image_path).name, image_handle, "application/octet-stream"),
            }
            with requests.post(url, headers=headers, files=files, stream=True, timeout=timeout) as resp:
                if resp.status_code != 200:
                    detail = resp.text[:500] if resp.content is not None else ""
                    raise RuntimeError(f"megaseg /v1/infer failed: HTTP {resp.status_code} {detail}")
                with tmp_tar.open("wb") as out:
                    for chunk in resp.iter_content(chunk_size=_CHUNK):
                        if chunk:
                            out.write(chunk)
        with tarfile.open(tmp_tar, mode="r:gz") as archive:
            _safe_extract(archive, dest_dir)
    finally:
        tmp_tar.unlink(missing_ok=True)
    result_path = dest_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    return {}


def register_outputs(
    *,
    control_base_url: str,
    job_id: str,
    principal_headers: dict[str, str],
    outputs: list[dict[str, Any]],
    timeout: float,
    collection_id: str = "",
) -> dict[str, Any]:
    """Register produced output files with the control plane (POST .../outputs)."""
    quoted = urllib.parse.quote(job_id, safe="")
    url = f"{control_base_url.rstrip('/')}/v2/data-agent/jobs/{quoted}/outputs"
    headers = {"Content-Type": "application/json", "Accept": "application/json", **principal_headers}
    body: dict[str, Any] = {"outputs": outputs}
    if collection_id:
        body["collection_id"] = collection_id
    resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=timeout)
    resp.raise_for_status()
    if not resp.content:
        return {}
    data = resp.json()
    return data if isinstance(data, dict) else {}


def fetch_job(
    *,
    control_base_url: str,
    job_id: str,
    principal_headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    """Fetch the current job record (for resume + cancellation polling). Returns {} on error."""
    quoted = urllib.parse.quote(job_id, safe="")
    url = f"{control_base_url.rstrip('/')}/v2/data-agent/jobs/{quoted}"
    headers = {"Accept": "application/json", **principal_headers}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except Exception:  # noqa: BLE001 - resume/cancel polling is best-effort
        return {}
    if not resp.content:
        return {}
    data = resp.json()
    if isinstance(data, dict) and isinstance(data.get("job"), dict):
        return data["job"]
    return {}
