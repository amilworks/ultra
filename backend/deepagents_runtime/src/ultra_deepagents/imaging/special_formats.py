"""Special-format registry (Python side).

Mirrors ``backend/shared/special_formats.json`` and the Go registry
(``internal/httpapi/special_formats.go``); ``tests/test_special_formats_parity.py``
asserts this matches the shared JSON. Special formats are directory/archive shaped or
need a non-default serving backend (e.g. OME-Zarr → the ngff-service).

TEMPLATE — to add a special format: append a row to the shared JSON, mirror it here and
in the Go registry, and wire its reader/serve hook. The parity test enforces the mirror.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

__all__ = ["SpecialFormat", "SPECIAL_FORMATS", "detect_by_name", "detect_by_dir", "detect"]


@dataclass(frozen=True)
class SpecialFormat:
    id: str
    extensions: tuple[str, ...]  # lowercase, with leading dot
    shape: str  # "file" | "directory" | "archive"
    dir_markers: tuple[str, ...]
    serve: str  # "ngff" | "libbioimage" | "transcode"
    reader: str
    resource_kind: str


SPECIAL_FORMATS: list[SpecialFormat] = [
    SpecialFormat(
        id="ome-zarr",
        extensions=(".ome.zarr", ".zarr"),
        shape="directory",
        dir_markers=(".zattrs", ".zgroup"),
        serve="ngff",
        reader="ngff",
        resource_kind="image",
    ),
]


def detect_by_name(name: str) -> SpecialFormat | None:
    """Match a resource name against the registry extensions (tolerating a series/page
    suffix like ``img.zarr_3``)."""
    lower = (name or "").strip().lower()
    for sf in SPECIAL_FORMATS:
        for ext in sf.extensions:
            if lower.endswith(ext) or (ext + "_") in lower or re.search(re.escape(ext) + r"_\d+$", lower):
                return sf
    return None


def detect_by_dir(path: str) -> SpecialFormat | None:
    """Match a directory against directory-format marker files (e.g. an OME-Zarr group
    has .zattrs + .zgroup at its root)."""
    if not os.path.isdir(path):
        return None
    for sf in SPECIAL_FORMATS:
        if sf.shape != "directory" or not sf.dir_markers:
            continue
        if all(os.path.exists(os.path.join(path, m)) for m in sf.dir_markers):
            return sf
    return None


def detect(name: str, path: str | None = None) -> SpecialFormat | None:
    """Resolve by name first, then (if a path is given) by on-disk directory markers."""
    sf = detect_by_name(name)
    if sf is not None:
        return sf
    return detect_by_dir(path) if path else None
