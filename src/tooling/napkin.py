from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

_CORRECTIONS_HEADING = "Corrections"
_USER_PREFERENCES_HEADING = "User Preferences"
_PATTERNS_WORK_HEADING = "Patterns That Work"
_PATTERNS_DONT_WORK_HEADING = "Patterns That Don't Work"
_GUARDRAILS_HEADING = "Tool-Call Guardrails"
_DOMAIN_NOTES_HEADING = "Domain Notes"

_DEFAULT_NAPKIN_TEMPLATE = """# Napkin

## Corrections
| Date | Source | What Went Wrong | What To Do Instead |
|------|--------|----------------|-------------------|

## User Preferences
- Prioritize user-facing clarity over internal execution details.

## Patterns That Work
- Prefer deterministic tool evidence (measurements, artifact names, and completion state) in the final response.

## Patterns That Don't Work
- One-line completion-only summaries without concrete metrics or artifacts.

## Tool-Call Guardrails
- Never expose internal tool/job IDs (for example codejob_* or exec_*) in user-facing output.
- When retries occur, keep only the best successful result in the final response and avoid duplicate cards.
- For code execution, include concrete quantitative metrics and produced artifact names.
- If expected outputs are missing, report partial completion and list missing deliverables.

## Domain Notes
- Keep outputs scientifically interpretable and reproducible.
"""

_DEFAULT_SKILL_GUARDRAILS = [
    "Read persistent corrections before selecting tool strategy.",
    "Do not expose internal job IDs, tool IDs, or scratchpad internals to users.",
    "When retries happen, dedupe tool outputs and surface only the final successful evidence.",
    "For code execution, report concrete metrics, artifacts, and explicit limitations.",
]

_INTERNAL_TOKEN_RE = re.compile(
    r"\b(?:codejob_[a-z0-9_-]{6,64}|exec_[a-z0-9_-]{6,64}|tool_[0-9]+_[0-9]+)\b",
    flags=re.IGNORECASE,
)
_HEADING_RE = re.compile(r"^##\s+(?P<title>.+?)\s*$")
_NAPKIN_LOCK = Lock()


def ensure_napkin_file(path: Path) -> None:
    with _NAPKIN_LOCK:
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_DEFAULT_NAPKIN_TEMPLATE, encoding="utf-8")


def build_tool_call_napkin_system_message(
    *,
    napkin_path: Path,
    skill_path: Path | None = None,
    max_chars: int = 3200,
) -> str | None:
    ensure_napkin_file(napkin_path)
    try:
        with _NAPKIN_LOCK:
            napkin_text = napkin_path.read_text(encoding="utf-8")
    except Exception:
        return None

    skill_text = ""
    if skill_path and skill_path.exists():
        try:
            skill_text = skill_path.read_text(encoding="utf-8")
        except Exception:
            skill_text = ""

    guardrails = _extract_section_bullets(napkin_text, _GUARDRAILS_HEADING)
    skill_guardrails = _extract_section_bullets(skill_text, "Bisque Ultra Tool-Call Additions")
    if not skill_guardrails:
        skill_guardrails = list(_DEFAULT_SKILL_GUARDRAILS)

    corrections = _extract_correction_rows(napkin_text)
    preferences = _extract_section_bullets(napkin_text, _USER_PREFERENCES_HEADING)
    patterns_work = _extract_section_bullets(napkin_text, _PATTERNS_WORK_HEADING)
    patterns_dont = _extract_section_bullets(napkin_text, _PATTERNS_DONT_WORK_HEADING)

    lines: list[str] = [
        "Persistent tool-calling memory (napkin). Apply these constraints before selecting tools:",
    ]
    for item in _dedup_items([*guardrails, *skill_guardrails])[:8]:
        lines.append(f"- {item}")

    if corrections:
        lines.append("Recent corrections:")
        for row in corrections[-4:]:
            wrong = _sanitize_fragment(row.get("wrong") or "")
            fix = _sanitize_fragment(row.get("fix") or "")
            if wrong and fix:
                lines.append(f"- Avoid: {wrong} | Do instead: {fix}")

    if preferences:
        lines.append("User preferences:")
        for item in preferences[:5]:
            lines.append(f"- {item}")

    if patterns_work:
        lines.append("Patterns that work:")
        for item in patterns_work[:5]:
            lines.append(f"- {item}")

    if patterns_dont:
        lines.append("Patterns to avoid:")
        for item in patterns_dont[:5]:
            lines.append(f"- {item}")

    rendered = "\n".join(lines).strip()
    if not rendered:
        return None
    if len(rendered) <= max_chars:
        return rendered
    return rendered[: max(0, max_chars - 1)].rstrip() + "…"


def update_tool_call_napkin(
    *,
    napkin_path: Path,
    progress_events: list[dict[str, Any]] | None,
    response_text: str | None = None,
) -> dict[str, int]:
    ensure_napkin_file(napkin_path)
    try:
        with _NAPKIN_LOCK:
            content = napkin_path.read_text(encoding="utf-8")
    except Exception:
        return {"rows_added": 0, "bullets_added": 0}

    today = datetime.utcnow().strftime("%Y-%m-%d")
    correction_rows: list[tuple[str, str, str, str]] = []
    work_bullets: list[str] = []
    dont_work_bullets: list[str] = []
    guardrail_bullets: list[str] = []

    completed_fingerprints: set[str] = set()
    duplicate_success_count = 0

    for event in progress_events or []:
        if not isinstance(event, dict):
            continue
        event_name = str(event.get("event") or "").strip().lower()
        tool_name = str(event.get("tool") or "").strip()
        if event_name == "error" and tool_name:
            error_text = _sanitize_fragment(
                str(event.get("error") or event.get("message") or "tool execution failed")
            )
            if not error_text:
                continue
            correction_rows.append(
                (
                    today,
                    "self",
                    f"{tool_name} failed ({error_text}).",
                    "Inspect tool inputs/constraints, retry with repaired arguments, and report partial completion if unresolved.",
                )
            )
            continue

        if event_name != "completed" or not tool_name:
            continue
        summary = event.get("summary")
        summary_blob = str(summary) if isinstance(summary, dict) else ""
        fingerprint = f"{tool_name}|{summary_blob[:180]}"
        if fingerprint in completed_fingerprints:
            duplicate_success_count += 1
        else:
            completed_fingerprints.add(fingerprint)

        if tool_name == "execute_python_job" and isinstance(summary, dict):
            measurements = summary.get("measurements")
            output_files = summary.get("output_files")
            missing_outputs = summary.get("missing_expected_outputs")
            measurement_count = len(measurements) if isinstance(measurements, list) else 0
            output_count = len(output_files) if isinstance(output_files, list) else 0
            missing_count = len(missing_outputs) if isinstance(missing_outputs, list) else 0
            if measurement_count > 0 and output_count > 0:
                work_bullets.append(
                    "For execute_python_job, use extracted quantitative metrics + output artifact names in the final answer."
                )
            if missing_count > 0:
                dont_work_bullets.append(
                    "Do not claim full completion when expected outputs are missing; label the result as partial and list missing deliverables."
                )
        if tool_name in {"search_bisque_resources", "bisque_advanced_search"}:
            work_bullets.append(
                "For BisQue searches, aggregate repeated retries into one consolidated user-facing summary/card."
            )

    if duplicate_success_count > 0:
        correction_rows.append(
            (
                today,
                "self",
                "Multiple successful retries were surfaced as duplicate tool outputs.",
                "Deduplicate by tool + output fingerprint and keep only the best successful result in user-facing output.",
            )
        )

    raw_response = str(response_text or "").strip()
    if raw_response and _INTERNAL_TOKEN_RE.search(raw_response):
        correction_rows.append(
            (
                today,
                "self",
                "User-facing response leaked internal execution identifiers.",
                "Redact internal IDs and refer generically to prior analysis or execution attempt.",
            )
        )
        guardrail_bullets.append(
            "Before finalizing a response, redact internal identifiers (codejob_*, exec_*, tool_*)."
        )

    updated = content
    rows_added = 0
    bullets_added = 0
    if correction_rows:
        updated, rows_added = _append_correction_rows(updated, correction_rows)
    if work_bullets:
        updated, added = _append_unique_bullets(updated, _PATTERNS_WORK_HEADING, work_bullets)
        bullets_added += added
    if dont_work_bullets:
        updated, added = _append_unique_bullets(
            updated, _PATTERNS_DONT_WORK_HEADING, dont_work_bullets
        )
        bullets_added += added
    if guardrail_bullets:
        updated, added = _append_unique_bullets(updated, _GUARDRAILS_HEADING, guardrail_bullets)
        bullets_added += added

    if updated != content:
        with _NAPKIN_LOCK:
            napkin_path.write_text(updated, encoding="utf-8")
    return {"rows_added": rows_added, "bullets_added": bullets_added}


def _extract_section_bullets(markdown: str, heading: str) -> list[str]:
    body = _extract_section_body(markdown, heading)
    if not body:
        return []
    output: list[str] = []
    for raw_line in body.splitlines():
        line = str(raw_line or "").strip()
        if not line.startswith("- "):
            continue
        candidate = _sanitize_fragment(line[2:])
        if candidate:
            output.append(candidate)
    return output


def _extract_correction_rows(markdown: str) -> list[dict[str, str]]:
    body = _extract_section_body(markdown, _CORRECTIONS_HEADING)
    if not body:
        return []
    rows: list[dict[str, str]] = []
    for raw_line in body.splitlines():
        line = str(raw_line or "").strip()
        if "|" not in line or line.startswith("|------"):
            continue
        columns = [item.strip() for item in line.strip("|").split("|")]
        if len(columns) < 4:
            continue
        if columns[0].strip().lower() == "date":
            continue
        date_value = _sanitize_fragment(columns[0])
        source = _sanitize_fragment(columns[1])
        wrong = _sanitize_fragment(columns[2])
        fix = _sanitize_fragment(columns[3])
        if not wrong or not fix:
            continue
        rows.append({"date": date_value, "source": source, "wrong": wrong, "fix": fix})
    return rows


def _append_correction_rows(
    markdown: str,
    rows: list[tuple[str, str, str, str]],
) -> tuple[str, int]:
    if not rows:
        return markdown, 0
    lines = markdown.splitlines()
    section_start, section_end = _section_bounds(lines, _CORRECTIONS_HEADING)
    if section_start < 0:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(
            [
                f"## {_CORRECTIONS_HEADING}",
                "| Date | Source | What Went Wrong | What To Do Instead |",
                "|------|--------|----------------|-------------------|",
                "",
            ]
        )
        section_start, section_end = _section_bounds(lines, _CORRECTIONS_HEADING)
    section_text = "\n".join(lines[section_start:section_end]).lower()
    insert_at = section_end
    added = 0
    for date_value, source, wrong, fix in rows:
        row_text = (
            f"| {date_value} | {source} | {wrong} | {fix} |"
        )
        row_fingerprint = f"{wrong.lower()}|{fix.lower()}"
        if row_fingerprint in section_text:
            continue
        lines.insert(insert_at, row_text)
        insert_at += 1
        added += 1
        section_text += f"\n{row_fingerprint}"
    if added == 0:
        return markdown, 0
    rendered = "\n".join(lines).rstrip() + "\n"
    return rendered, added


def _append_unique_bullets(markdown: str, heading: str, bullets: list[str]) -> tuple[str, int]:
    candidates = _dedup_items([_sanitize_fragment(item) for item in bullets if _sanitize_fragment(item)])
    if not candidates:
        return markdown, 0
    lines = markdown.splitlines()
    section_start, section_end = _section_bounds(lines, heading)
    if section_start < 0:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend([f"## {heading}", ""])
        section_start, section_end = _section_bounds(lines, heading)
    section_text = "\n".join(lines[section_start:section_end]).lower()
    insert_at = section_end
    added = 0
    for item in candidates:
        fingerprint = item.lower()
        if fingerprint in section_text:
            continue
        lines.insert(insert_at, f"- {item}")
        insert_at += 1
        added += 1
        section_text += "\n" + fingerprint
    if added == 0:
        return markdown, 0
    rendered = "\n".join(lines).rstrip() + "\n"
    return rendered, added


def _section_bounds(lines: list[str], heading: str) -> tuple[int, int]:
    section_start = -1
    section_end = -1
    target = str(heading or "").strip().lower()
    for idx, line in enumerate(lines):
        match = _HEADING_RE.match(str(line or "").strip())
        if not match:
            continue
        title = str(match.group("title") or "").strip().lower()
        if section_start < 0 and title == target:
            section_start = idx
            continue
        if section_start >= 0:
            section_end = idx
            break
    if section_start >= 0 and section_end < 0:
        section_end = len(lines)
    return section_start, section_end


def _extract_section_body(markdown: str, heading: str) -> str:
    if not markdown:
        return ""
    lines = markdown.splitlines()
    start, end = _section_bounds(lines, heading)
    if start < 0 or end <= start:
        return ""
    body = "\n".join(lines[start + 1 : end]).strip()
    return body


def _sanitize_fragment(value: str, max_len: int = 260) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = _INTERNAL_TOKEN_RE.sub("internal_execution_id", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    return text[: max(1, max_len - 1)].rstrip() + "…"


def _dedup_items(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value or "").strip()
        if not token:
            continue
        lower = token.lower()
        if lower in seen:
            continue
        seen.add(lower)
        output.append(token)
    return output
