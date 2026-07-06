from __future__ import annotations

import re
from typing import Iterable

SAFE_FILE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")


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
