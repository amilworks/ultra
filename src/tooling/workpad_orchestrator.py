from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Literal


_PHASE_EVENT_NAME = "phase"


@dataclass(frozen=True)
class PromptWorkpadConfig:
    enabled: bool = False
    # BRUHHH LOOK AT THIS LATER: "legacy" is still a live compatibility mode,
    # not the removed Code Mode feature. Decide whether phased can be the only
    # production path before deleting it.
    mode: Literal["legacy", "phased"] = "legacy"
    retain_on_success: bool = False
    retain_on_failure: bool = True


@dataclass(frozen=True)
class PromptWorkpadFinalizeResult:
    action: str
    source_path: str | None = None
    retained_path: str | None = None
    manifest_category: str | None = None


def upsert_markdown_section(
    *,
    path: Path | str,
    title: str,
    body: str | None,
) -> bool:
    """Insert or replace a named markdown section in the run scratchpad."""
    section_title = str(title or "").strip()
    section_body = str(body or "").strip()
    if not section_title or not section_body:
        return False
    target_path = Path(path).expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    current = ""
    if target_path.exists():
        try:
            current = target_path.read_text(encoding="utf-8")
        except Exception:
            current = ""
    if not current:
        current = "# Prompt Workpad\n\n"

    section_re = re.compile(
        rf"(?ms)^## {re.escape(section_title)}\n.*?(?=^## |\Z)"
    )
    replacement = f"## {section_title}\n{section_body}\n\n"
    if section_re.search(current):
        updated = section_re.sub(replacement, current, count=1)
    else:
        updated = current.rstrip() + "\n\n" + replacement
    target_path.write_text(updated.rstrip() + "\n", encoding="utf-8")
    return True


def extract_prompt_workpad_phase_trace(
    progress_events: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Summarize phase progress events into a compact trace payload."""
    if not isinstance(progress_events, list):
        return None
    phase_rows: list[dict[str, Any]] = []
    total_durations: dict[str, float] = {}
    for event in progress_events:
        if not isinstance(event, dict):
            continue
        if str(event.get("event") or "").strip().lower() != _PHASE_EVENT_NAME:
            continue
        phase = str(event.get("phase") or "").strip()
        status = str(event.get("status") or "").strip().lower()
        if not phase:
            continue
        row: dict[str, Any] = {"phase": phase}
        if status:
            row["status"] = status
        mode = str(event.get("mode") or "").strip()
        if mode:
            row["mode"] = mode
        ts = str(event.get("ts") or "").strip()
        if ts:
            row["ts"] = ts
        reason = str(event.get("reason") or "").strip()
        if reason:
            row["reason"] = reason
        message = str(event.get("message") or "").strip()
        if message:
            row["message"] = message
        duration = event.get("duration_seconds")
        if isinstance(duration, (int, float)):
            duration_value = round(float(duration), 3)
            row["duration_seconds"] = duration_value
            if status == "completed":
                total_durations[phase] = round(
                    total_durations.get(phase, 0.0) + duration_value, 3
                )
        phase_rows.append(row)
        if len(phase_rows) >= 120:
            break
    if not phase_rows:
        return None
    return {
        "count": len(phase_rows),
        "phases": phase_rows,
        "total_phase_durations": total_durations,
    }


class PromptWorkpadOrchestrator:
    """Manage the optional per-run scratchpad that captures planning traces."""

    def __init__(
        self,
        *,
        run_id: str,
        scratchpad_path: str | None,
        run_artifact_dir: Path,
        config: PromptWorkpadConfig,
    ) -> None:
        self.run_id = str(run_id or "").strip()
        self.config = config
        self.run_artifact_dir = Path(run_artifact_dir)
        self.path = (
            Path(str(scratchpad_path)).expanduser().resolve()
            if str(scratchpad_path or "").strip()
            else None
        )

    @property
    def active(self) -> bool:
        return self.path is not None

    def mode_payload(self, *, scratchpad_requested: bool) -> dict[str, Any]:
        return {
            "enabled": bool(self.config.enabled),
            "mode": str(self.config.mode),
            "scratchpad_requested": bool(scratchpad_requested),
            "scratchpad_active": bool(self.active),
            "retain_on_success": bool(self.config.retain_on_success),
            "retain_on_failure": bool(self.config.retain_on_failure),
        }

    def initialize(self, *, prompt_text: str | None = None) -> dict[str, Any] | None:
        if not self.active:
            return None
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = ""
        if self.path.exists():
            try:
                existing = self.path.read_text(encoding="utf-8")
            except Exception:
                existing = ""
        if existing.strip():
            return {
                "created": False,
                "path": str(self.path),
                "mode": str(self.config.mode),
            }
        self.path.write_text(
            self._initial_workpad_template(prompt_text=prompt_text),
            encoding="utf-8",
        )
        return {
            "created": True,
            "path": str(self.path),
            "mode": str(self.config.mode),
        }

    def append_section(self, title: str, body: str | None) -> bool:
        if not self.active:
            return False
        assert self.path is not None
        text = str(body or "").strip()
        if not text:
            return False
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n## {str(title).strip()}\n{text}\n")
        return True

    def set_section(self, title: str, body: str | None) -> bool:
        if not self.active:
            return False
        assert self.path is not None
        return upsert_markdown_section(path=self.path, title=title, body=body)

    def append_outcome_success(self, *, duration_seconds: float, tool_calls: int) -> bool:
        return self.set_section(
            "Outcome",
            (
                f"Status: succeeded\n"
                f"Duration seconds: {round(float(duration_seconds), 3)}\n"
                f"Tool calls: {int(tool_calls)}\n"
                f"Completed at: {datetime.utcnow().isoformat()}Z"
            ),
        )

    def append_outcome_failure(self, *, error_text: str) -> bool:
        return self.set_section(
            "Outcome",
            (
                f"Status: failed\n"
                f"Error: {str(error_text or '').strip()}\n"
                f"Failed at: {datetime.utcnow().isoformat()}Z"
            ),
        )

    def finalize_success(self) -> PromptWorkpadFinalizeResult:
        if not self.active:
            return PromptWorkpadFinalizeResult(action="no_scratchpad")
        assert self.path is not None
        if not self.path.exists():
            return PromptWorkpadFinalizeResult(action="scratchpad_missing")
        if self.config.enabled:
            if self.config.retain_on_success:
                destination = self.run_artifact_dir / "scratchpad.md"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(self.path.read_bytes())
                return PromptWorkpadFinalizeResult(
                    action="retained_on_success",
                    source_path=str(self.path),
                    retained_path=str(destination),
                    manifest_category="scratchpad",
                )
            self.path.unlink(missing_ok=True)
            return PromptWorkpadFinalizeResult(
                action="deleted_on_success",
                source_path=str(self.path),
            )

        destination = self.run_artifact_dir / "scratchpad.md"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.path.read_bytes())
        return PromptWorkpadFinalizeResult(
            action="legacy_copied",
            source_path=str(self.path),
            retained_path=str(destination),
            manifest_category="scratchpad",
        )

    def finalize_failure(self) -> PromptWorkpadFinalizeResult:
        if not self.active:
            return PromptWorkpadFinalizeResult(action="no_scratchpad")
        assert self.path is not None
        if not self.path.exists():
            return PromptWorkpadFinalizeResult(action="scratchpad_missing")
        if not self.config.enabled:
            return PromptWorkpadFinalizeResult(action="legacy_noop")

        if self.config.retain_on_failure:
            destination = self.run_artifact_dir / "scratchpad_failed.md"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(self.path.read_bytes())
            return PromptWorkpadFinalizeResult(
                action="retained_on_failure",
                source_path=str(self.path),
                retained_path=str(destination),
                manifest_category="scratchpad_failure",
            )

        self.path.unlink(missing_ok=True)
        return PromptWorkpadFinalizeResult(
            action="deleted_on_failure",
            source_path=str(self.path),
        )

    def _initial_workpad_template(self, *, prompt_text: str | None) -> str:
        normalized_prompt = str(prompt_text or "").strip()
        lines: list[str] = [
            "# Prompt Workpad",
            "",
            f"Run ID: {self.run_id or 'unknown'}",
            f"Mode: {self.config.mode}",
            f"Created: {datetime.utcnow().isoformat()}Z",
            "",
            "## Prompt (original)",
            normalized_prompt or "(empty)",
            "",
            "## Prompt (reframed - low effort)",
            "",
            "## Plan (medium effort)",
            "",
            "## Tool Calls",
            "",
            "## Tool Outputs (normalized)",
            "",
            "## Final Answer Draft (high effort)",
            "",
            "## Final Contract",
            "",
            "## Outcome",
            "Status: running",
            "",
        ]
        return "\n".join(lines)
