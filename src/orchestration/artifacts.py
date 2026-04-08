from __future__ import annotations

import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import get_settings


def ensure_run_artifact_dir(run_id: str, *, artifact_root: str | None = None) -> Path:
    settings = get_settings()
    root = Path(str(artifact_root or getattr(settings, "artifact_root", "data/artifacts"))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = (root / str(run_id).strip()).resolve()
    if root not in run_dir.parents:
        raise ValueError("Invalid run_id path")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _safe_artifact_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(name or "artifact"))
    return cleaned.strip("._") or "artifact"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_entry(
    run_id: str,
    path: Path,
    *,
    source_path: str | None = None,
    artifact_root: str | None = None,
) -> dict[str, Any]:
    stat = path.stat()
    run_dir = ensure_run_artifact_dir(run_id, artifact_root=artifact_root)
    entry = {
        "path": str(path.relative_to(run_dir)),
        "size_bytes": int(stat.st_size),
        "mime_type": None,
        "modified_at": datetime.utcfromtimestamp(stat.st_mtime).isoformat() + "Z",
    }
    if source_path:
        entry["source_path"] = str(source_path)
    return entry


def snapshot_tool_invocation_artifacts(
    *,
    run_id: str,
    tool_invocations: list[dict[str, Any]],
    artifact_root: str | None = None,
) -> list[dict[str, Any]]:
    run_dir = ensure_run_artifact_dir(run_id, artifact_root=artifact_root)
    target_dir = run_dir / "tool_outputs"
    target_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for invocation in tool_invocations:
        if not isinstance(invocation, dict):
            continue
        tool_name = str(invocation.get("tool") or "").strip() or "tool"
        envelope = invocation.get("output_envelope")
        if not isinstance(envelope, dict):
            continue
        for artifact_group in ("ui_artifacts", "download_artifacts"):
            raw_group = envelope.get(artifact_group)
            if not isinstance(raw_group, list):
                continue
            for item in raw_group[:64]:
                if not isinstance(item, dict):
                    continue
                source_path = Path(str(item.get("path") or "")).expanduser()
                if not source_path.exists() or not source_path.is_file():
                    continue
                source_token = str(source_path.resolve())
                if source_token in seen:
                    continue
                seen.add(source_token)
                sha256 = _sha256_file(source_path)
                safe_name = _safe_artifact_name(source_path.name)
                destination = target_dir / f"{sha256[:12]}__{safe_name}"
                if not destination.exists():
                    shutil.copy2(source_path, destination)
                entry = artifact_entry(
                    run_id,
                    destination,
                    source_path=str(source_path),
                    artifact_root=artifact_root,
                )
                entry["tool"] = tool_name
                entry["category"] = "ui" if artifact_group == "ui_artifacts" else "download"
                entry["title"] = str(item.get("title") or "").strip() or safe_name
                entries.append(entry)
    return entries
