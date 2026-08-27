"""Fail-closed parsing for the control-plane-authored Notes run scope."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

_MAX_SELECTED_NOTES = 20
_MAX_NOTE_ID_CHARS = 512
_MAX_REVISION = (1 << 63) - 1
_NOTE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


@dataclass(frozen=True, slots=True)
class NoteReference:
    note_id: str
    revision: int | None = None


@dataclass(frozen=True, slots=True)
class NoteAccessScope:
    mode: Literal["none", "selected", "search"] = "none"
    notes: tuple[NoteReference, ...] = ()
    allow_append_proposal: bool = False

    @property
    def enabled(self) -> bool:
        return self.mode in {"selected", "search"}

    @property
    def allows_search(self) -> bool:
        return self.mode == "search"

    def allows_note(self, note_id: str) -> bool:
        normalized = _note_id(note_id)
        if not normalized:
            return False
        if self.mode == "search":
            return True
        return any(reference.note_id == normalized for reference in self.notes)

    def to_mapping(self) -> dict[str, Any]:
        if not self.enabled:
            return {"mode": "none", "notes": []}
        notes: list[dict[str, Any]] = []
        for reference in self.notes:
            item: dict[str, Any] = {"note_id": reference.note_id}
            if reference.revision is not None:
                item["revision"] = reference.revision
            notes.append(item)
        return {
            "mode": self.mode,
            "notes": notes,
            "allow_append_proposal": self.allow_append_proposal,
        }


def note_access_from_selection_context(value: Any) -> NoteAccessScope:
    """Read only the typed ``selection_context.note_access`` capability.

    Free-form siblings never grant Notes access. Unknown modes, malformed note
    references, and an empty selected scope all fail closed.
    """

    if not isinstance(value, Mapping):
        return NoteAccessScope()
    return normalize_note_access(value.get("note_access"))


def normalize_note_access(value: Any) -> NoteAccessScope:
    if not isinstance(value, Mapping):
        return NoteAccessScope()
    mode = value.get("mode")
    if not isinstance(mode, str):
        return NoteAccessScope()
    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"selected", "search"}:
        return NoteAccessScope()

    raw_allow_append_proposal = value.get("allow_append_proposal", False)
    if not isinstance(raw_allow_append_proposal, bool):
        return NoteAccessScope()

    raw_notes = value.get("notes")
    if raw_notes is None:
        raw_notes = []
    if not isinstance(raw_notes, list | tuple):
        return NoteAccessScope()
    if len(raw_notes) > _MAX_SELECTED_NOTES:
        return NoteAccessScope()

    references: list[NoteReference] = []
    seen: set[str] = set()
    for raw_note in raw_notes:
        if not isinstance(raw_note, Mapping):
            return NoteAccessScope()
        note_id = _note_id(raw_note.get("note_id"))
        if not note_id or note_id in seen:
            return NoteAccessScope()
        revision = _revision(raw_note.get("revision"))
        if raw_note.get("revision") is not None and revision is None:
            return NoteAccessScope()
        seen.add(note_id)
        references.append(NoteReference(note_id=note_id, revision=revision))

    if normalized_mode == "selected" and not references:
        return NoteAccessScope()
    return NoteAccessScope(
        mode=cast(Literal["selected", "search"], normalized_mode),
        notes=tuple(references),
        allow_append_proposal=raw_allow_append_proposal,
    )


def normalized_selection_context(value: Any) -> dict[str, Any]:
    """Preserve unrelated selection hints while canonicalizing Notes authority."""

    if not isinstance(value, Mapping):
        return {}
    normalized = dict(value)
    scope = note_access_from_selection_context(normalized)
    if scope.enabled:
        normalized["note_access"] = scope.to_mapping()
    else:
        normalized.pop("note_access", None)
    return normalized


def _note_id(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.strip()
    if not normalized or len(normalized) > _MAX_NOTE_ID_CHARS:
        return ""
    return normalized if _NOTE_ID_RE.fullmatch(normalized) else ""


def _revision(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 1 <= value <= _MAX_REVISION else None
