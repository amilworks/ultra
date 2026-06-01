from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable

from ultra_deepagents.rarespot.staging import IMAGE_SUFFIXES, ensure_allowed_path

SAFE_FILE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")


@dataclass(frozen=True)
class UploadedFileResolution:
    image_paths: list[Path]
    missing_file_ids: list[str]


async def resolve_uploaded_file_ids(
    file_ids: Iterable[str],
    *,
    upload_roots: tuple[Path, ...],
    allowed_roots: tuple[Path, ...] = (),
    database_url: str = "",
) -> UploadedFileResolution:
    requested = unique_file_ids(file_ids)
    if not requested:
        return UploadedFileResolution(image_paths=[], missing_file_ids=[])

    resolved: dict[str, Path] = {}
    rows = await load_upload_rows(database_url, requested) if database_url else {}
    effective_allowed_roots = allowed_roots or upload_roots

    for file_id in requested:
        for candidate in candidate_paths_from_row(rows.get(file_id), upload_roots=upload_roots):
            path = usable_image_path(candidate, allowed_roots=effective_allowed_roots)
            if path is not None:
                resolved[file_id] = path
                break

    for file_id in requested:
        if file_id in resolved:
            continue
        path = scan_upload_roots(file_id, upload_roots=upload_roots, allowed_roots=effective_allowed_roots)
        if path is not None:
            resolved[file_id] = path

    return UploadedFileResolution(
        image_paths=[resolved[file_id] for file_id in requested if file_id in resolved],
        missing_file_ids=[file_id for file_id in requested if file_id not in resolved],
    )


def unique_file_ids(file_ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for raw in file_ids:
        file_id = str(raw or "").strip()
        if not file_id or file_id in seen:
            continue
        seen.add(file_id)
        unique.append(file_id)
    return unique


async def load_upload_rows(database_url: str, file_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not database_url or not file_ids:
        return {}
    if not database_url.startswith(("postgres://", "postgresql://")):
        return {}
    try:
        import psycopg
        from psycopg.rows import dict_row
    except Exception:
        return {}

    try:
        async with await psycopg.AsyncConnection.connect(database_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT file_id, stored_path, staging_path, cache_path, original_name, content_type
                    FROM uploads
                    WHERE file_id = ANY(%s)
                    """,
                    (file_ids,),
                )
                rows = await cursor.fetchall()
    except Exception:
        return {}
    return {str(row.get("file_id")): dict(row) for row in rows if row.get("file_id")}


def candidate_paths_from_row(row: dict[str, Any] | None, *, upload_roots: tuple[Path, ...]) -> list[Path]:
    if not row:
        return []
    candidates: list[Path] = []
    for key in ("stored_path", "staging_path", "cache_path"):
        value = str(row.get(key) or "").strip()
        if not value:
            continue
        raw = Path(value).expanduser()
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.extend(root / raw for root in upload_roots)
    return candidates


def scan_upload_roots(
    file_id: str,
    *,
    upload_roots: tuple[Path, ...],
    allowed_roots: tuple[Path, ...],
) -> Path | None:
    if not SAFE_FILE_ID_RE.fullmatch(file_id):
        return None
    for root in upload_roots:
        resolved_root = root.expanduser().resolve()
        if not resolved_root.exists():
            continue
        for pattern in (file_id, f"{file_id}__*", f"{file_id}.*"):
            for candidate in sorted(resolved_root.rglob(pattern)):
                path = usable_image_path(candidate, allowed_roots=allowed_roots)
                if path is not None:
                    return path
    return None


def usable_image_path(path: Path, *, allowed_roots: tuple[Path, ...]) -> Path | None:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    try:
        ensure_allowed_path(resolved, allowed_roots)
    except ValueError:
        return None
    return resolved
