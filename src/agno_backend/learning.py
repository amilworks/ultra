from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agentic.repositories import ScientificNoteRepository


@dataclass
class ScientificLearningOutcome:
    promoted_notes: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)

    def metadata(self) -> dict[str, Any]:
        return {
            "promoted_count": len(self.promoted_notes),
            "promoted_notes": list(self.promoted_notes),
            "skipped": list(self.skipped),
        }


class ScientificLearningJournal:
    def __init__(self, *, notes: ScientificNoteRepository) -> None:
        self.notes = notes

    def evaluate_and_promote(
        self,
        *,
        user_id: str | None,
        session_id: str | None,
        project_id: str | None,
        run_id: str | None,
        query: str,
        response_text: str,
        selected_domains: list[str] | None,
        tool_invocations: list[dict[str, Any]] | None,
        interrupted: bool,
        error: str | None = None,
    ) -> ScientificLearningOutcome:
        outcome = ScientificLearningOutcome()
        resolved_user_id = str(user_id or "").strip()
        resolved_session_id = str(session_id or "").strip() or None
        resolved_project_id = str(project_id or "").strip() or None
        if not resolved_user_id:
            outcome.skipped.append({"reason": "missing_user"})
            return outcome
        if interrupted:
            outcome.skipped.append({"reason": "approval_pending"})
            return outcome
        if str(error or "").strip():
            outcome.skipped.append({"reason": "runtime_error"})
            return outcome
        final_response = str(response_text or "").strip()
        if not final_response:
            outcome.skipped.append({"reason": "empty_response"})
            return outcome

        successful_tools = [
            invocation
            for invocation in list(tool_invocations or [])
            if str(invocation.get("status") or "").strip().lower() == "completed"
            and str(invocation.get("tool") or "").strip().lower() not in {"update_user_memory"}
            and isinstance(invocation.get("output_summary"), dict)
            and bool(invocation.get("output_summary"))
            and str((invocation.get("output_summary") or {}).get("preview") or "").strip().lower()
            != "no response from model"
        ]
        if not successful_tools:
            outcome.skipped.append({"reason": "no_artifact_backed_outputs"})
            return outcome

        note_scope = "project_notes" if resolved_project_id else "session_notes"
        title = self._candidate_title(query=query, tool_invocations=successful_tools)
        body = self._candidate_body(
            response_text=final_response,
            tool_invocations=successful_tools,
        )
        row = self.notes.upsert_note(
            user_id=resolved_user_id,
            session_id=resolved_session_id,
            project_id=resolved_project_id,
            scope=note_scope,
            title=title,
            body=body,
            tags=self._candidate_tags(
                selected_domains=selected_domains,
                tool_invocations=successful_tools,
                note_scope=note_scope,
            ),
            provenance={
                "source": "artifact_backed_run",
                "run_id": run_id,
                "tool_names": [
                    str(item.get("tool") or "").strip()
                    for item in successful_tools
                    if str(item.get("tool") or "").strip()
                ],
            },
            score=min(0.95, 0.58 + 0.08 * len(successful_tools)),
        )
        outcome.promoted_notes.append(
            {
                "note_id": row.get("note_id"),
                "title": row.get("title"),
                "scope": row.get("scope"),
                "project_id": row.get("project_id"),
            }
        )
        return outcome

    @staticmethod
    def _candidate_title(*, query: str, tool_invocations: list[dict[str, Any]]) -> str:
        if tool_invocations:
            tool_names = [
                str(item.get("tool") or "").strip()
                for item in tool_invocations
                if str(item.get("tool") or "").strip()
            ]
            if tool_names:
                if len(tool_names) == 1:
                    return f"Reusable finding from {tool_names[0]}"
                return "Reusable finding from tool-backed run"
        clean_query = str(query or "").strip()
        if clean_query:
            return clean_query[:120]
        return "Reusable scientific finding"

    @staticmethod
    def _candidate_body(
        *,
        response_text: str,
        tool_invocations: list[dict[str, Any]],
    ) -> str:
        lines = ["Artifact-backed reusable finding:"]
        lines.append(str(response_text or "").strip()[:480])
        lines.append("")
        lines.append("Grounding:")
        for invocation in tool_invocations[:3]:
            tool_name = str(invocation.get("tool") or "tool").strip()
            summary = invocation.get("output_summary")
            if isinstance(summary, dict):
                rendered = ", ".join(
                    f"{key}={value}"
                    for key, value in summary.items()
                    if value not in (None, "", [], {})
                )
            else:
                rendered = ""
            if not rendered:
                rendered = str(invocation.get("output_preview") or "").strip()[:180]
            if rendered:
                lines.append(f"- {tool_name}: {rendered}")
        return "\n".join(lines).strip()[:1400]

    @staticmethod
    def _candidate_tags(
        *,
        selected_domains: list[str] | None,
        tool_invocations: list[dict[str, Any]],
        note_scope: str,
    ) -> list[str]:
        ordered: list[str] = [note_scope, "scientific_learning"]
        seen = set(ordered)
        for raw in list(selected_domains or []):
            token = str(raw or "").strip().lower()
            if token and token not in seen:
                seen.add(token)
                ordered.append(token)
        for invocation in tool_invocations[:4]:
            token = str(invocation.get("tool") or "").strip().lower()
            if token and token not in seen:
                seen.add(token)
                ordered.append(token)
        return ordered
