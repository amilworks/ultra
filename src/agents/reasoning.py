"""Generic reasoning helpers used by routing, verification, and domain prompts."""

from __future__ import annotations

import re
from typing import Any

_MCQ_OPTION_LINE_RE = re.compile(r"(?mi)^\s*([A-D])\s*[\.\)]\s*(.+?)\s*$")
_FINAL_ANSWER_RE = re.compile(r"FINAL_ANSWER\s*:\s*([ABCD])\b", re.IGNORECASE)
_ANSWER_HINT_RE = re.compile(
    r"\b(?:answer|option|choice)\s*(?:is|:)?\s*\(?([ABCD])\)?\b",
    re.IGNORECASE,
)
_STANDALONE_LETTER_RE = re.compile(r"\b([ABCD])\b")
_MCQ_PROMPT_CUES = (
    "which of the following",
    "choose the correct",
    "multiple choice",
    "options:",
)


def normalize_text(value: Any) -> str:
    """Normalize free text for robust matching."""

    lowered = str(value or "").lower()
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return re.sub(r"[^a-z0-9 ]+", "", lowered)


def extract_mcq_options(user_text: str) -> dict[str, str]:
    """Extract option mapping from a multiple-choice prompt.

    Parameters
    ----------
    user_text
        Prompt text potentially containing options like ``A. ...``.

    Returns
    -------
    dict[str, str]
        Mapping from option letter to option text.
    """

    out: dict[str, str] = {}
    for match in _MCQ_OPTION_LINE_RE.finditer(str(user_text or "")):
        letter = str(match.group(1) or "").upper()
        text = str(match.group(2) or "").strip()
        if letter and text:
            out[letter] = text
    return out


def is_mcq_prompt(user_text: str) -> bool:
    """Return True when prompt appears to be multiple-choice."""

    text = str(user_text or "")
    options = extract_mcq_options(text)
    if len(options) >= 3:
        return True
    lowered = text.lower()
    return any(cue in lowered for cue in _MCQ_PROMPT_CUES)


def parse_mcq_answer_letter(text: str, *, options: dict[str, str] | None = None) -> str | None:
    """Parse a selected MCQ option letter from model output."""

    body = str(text or "")
    for pattern in (_FINAL_ANSWER_RE, _ANSWER_HINT_RE):
        match = pattern.search(body)
        if match:
            return str(match.group(1)).upper()

    if options:
        normalized_response = normalize_text(body)
        hits: list[str] = []
        for letter, option_text in options.items():
            token = normalize_text(option_text)
            if token and token in normalized_response:
                hits.append(str(letter).upper())
        unique_hits = sorted(set(hits))
        if len(unique_hits) == 1:
            return unique_hits[0]

    letters = [m.group(1).upper() for m in _STANDALONE_LETTER_RE.finditer(body)]
    if letters:
        return letters[-1]
    return None


def has_hesitation_language(text: str) -> bool:
    """Return True when answer text signals low commitment."""

    return bool(
        re.search(
            r"\b(maybe|perhaps|possibly|likely|i think|i guess|uncertain|not sure)\b",
            str(text or ""),
            flags=re.IGNORECASE,
        )
    )
