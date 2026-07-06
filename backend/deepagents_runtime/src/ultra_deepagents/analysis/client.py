"""HTTP clients for the batch inference worker: the GPU MegaSeg service and the control
plane's output-registration / job-status endpoints. All calls are synchronous (the worker
invokes them via asyncio.to_thread)."""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
import urllib.parse
from pathlib import Path, PurePosixPath
from typing import Any

import requests

_CHUNK = 1024 * 1024
MAX_EXTRACTED_TAR_MEMBERS = 10_000
MAX_EXTRACTED_TAR_UNCOMPRESSED_BYTES = 10 * 1024**3


def _safe_tar_member_parts(member_name: str) -> tuple[str, ...]:
    raw_name = str(member_name or "")
    if raw_name in {"", "."}:
        return ()
    member_path = PurePosixPath(raw_name)
    if member_path.is_absolute():
        raise RuntimeError(f"refusing to extract absolute archive member: {member_name}")
    parts = tuple(part for part in member_path.parts if part not in {"", "."})
    if any(part == ".." for part in parts):
        raise RuntimeError(f"refusing to extract unsafe archive member: {member_name}")
    return parts


def _safe_tar_member_target(dest_dir: Path, member: tarfile.TarInfo) -> Path:
    dest_resolved = dest_dir.resolve()
    parts = _safe_tar_member_parts(member.name)
    candidate = dest_dir.joinpath(*parts) if parts else dest_dir
    target = candidate.resolve(strict=False)
    try:
        target.relative_to(dest_resolved)
    except ValueError as exc:
        raise RuntimeError(f"refusing to extract unsafe archive member: {member.name}") from exc
    if target == dest_resolved and not member.isdir():
        raise RuntimeError(f"refusing to extract archive member over destination root: {member.name}")
    return target


def _validate_safe_tar_member(member: tarfile.TarInfo, dest_dir: Path) -> Path:
    target = _safe_tar_member_target(dest_dir, member)
    if member.issym() or member.islnk():
        raise RuntimeError(f"refusing to extract archive link member: {member.name}")
    if member.ischr() or member.isblk() or member.isfifo():
        raise RuntimeError(f"refusing to extract special archive member: {member.name}")
    if not (member.isdir() or member.isfile()):
        raise RuntimeError(f"refusing to extract unsupported archive member: {member.name}")
    return target


def _validated_tar_member(
    member: tarfile.TarInfo,
    dest_dir: Path,
    *,
    member_count: int,
    total_uncompressed_bytes: int,
) -> tuple[Path, int]:
    if member_count > MAX_EXTRACTED_TAR_MEMBERS:
        raise RuntimeError(
            f"refusing to extract archive with too many members: "
            f"{member_count} > {MAX_EXTRACTED_TAR_MEMBERS}"
        )
    if member.isfile():
        size = int(member.size or 0)
        if size < 0:
            raise RuntimeError(f"refusing to extract archive member with negative size: {member.name}")
        total_uncompressed_bytes += size
        if total_uncompressed_bytes > MAX_EXTRACTED_TAR_UNCOMPRESSED_BYTES:
            raise RuntimeError(
                f"refusing to extract archive exceeding uncompressed size limit: "
                f"{total_uncompressed_bytes} > {MAX_EXTRACTED_TAR_UNCOMPRESSED_BYTES}"
            )
    return _validate_safe_tar_member(member, dest_dir), total_uncompressed_bytes


def _safe_extract(archive: tarfile.TarFile, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    member_count = 0
    total_uncompressed_bytes = 0
    for member in archive:
        member_count += 1
        target, total_uncompressed_bytes = _validated_tar_member(
            member,
            dest_dir,
            member_count=member_count,
            total_uncompressed_bytes=total_uncompressed_bytes,
        )
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        source = archive.extractfile(member)
        if source is None:
            raise RuntimeError(f"unable to read archive member: {member.name}")
        with source, target.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=_CHUNK)


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
