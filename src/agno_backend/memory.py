from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from src.agentic.repositories import ScientificNoteRepository, SessionRepository


class ScientificMemoryPolicy(BaseModel):
    mode: Literal["off", "conservative"] = "conservative"
    session_summary: bool = True
    user_memory: bool = True
    project_notebook: bool = True


@dataclass
class ScientificMemoryContext:
    policy: ScientificMemoryPolicy
    system_messages: list[dict[str, str]] = field(default_factory=list)
    session_summary_used: bool = False
    hit_count: int = 0
    hits: list[dict[str, Any]] = field(default_factory=list)
    context_chars: int = 0
    agno_features: dict[str, Any] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        return {
            "policy": self.policy.model_dump(mode="json"),
            "session_summary_used": self.session_summary_used,
            "hit_count": int(self.hit_count),
            "hits": list(self.hits),
            "context_chars": int(self.context_chars),
            "writes": [],
            "agno": dict(self.agno_features),
        }


@dataclass
class ScientificMemoryUpdate:
    summary_updated: bool = False
    summary: str | None = None
    writes: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def metadata(self) -> dict[str, Any]:
        return {
            "summary_updated": bool(self.summary_updated),
            "summary": str(self.summary or "").strip() or None,
            "writes": list(self.writes),
            "skipped": list(self.skipped),
        }


class ScientificMemoryService:
    def __init__(
        self,
        *,
        sessions: SessionRepository,
        notes: ScientificNoteRepository,
    ) -> None:
        self.sessions = sessions
        self.notes = notes

    @staticmethod
    def normalize_policy(
        raw: dict[str, Any] | ScientificMemoryPolicy | None,
    ) -> ScientificMemoryPolicy:
        if isinstance(raw, ScientificMemoryPolicy):
            return raw
        if isinstance(raw, dict):
            return ScientificMemoryPolicy.model_validate(raw)
        return ScientificMemoryPolicy()

    @classmethod
    def agno_feature_settings(
        cls,
        *,
        policy: ScientificMemoryPolicy,
        history_message_count: int,
    ) -> dict[str, Any]:
        if policy.mode == "off":
            return {
                "add_history_to_context": False,
                "search_past_sessions": False,
                "enable_session_summaries": False,
                "enable_agentic_memory": False,
                "update_memory_on_run": False,
            }
        return {
            # The stable path keeps memory app-owned and explicit; Agno's built-in
            # session/history tooling adds extra hidden model work after the visible
            # answer is ready, which hurts latency and makes failure recovery harder.
            "add_history_to_context": False,
            "search_past_sessions": False,
            "enable_session_summaries": False,
            "enable_agentic_memory": False,
            "update_memory_on_run": False,
        }

    def ensure_session(
        self,
        *,
        session_id: str | None,
        user_id: str | None,
        title: str,
        memory_policy: ScientificMemoryPolicy,
        knowledge_scope: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        resolved_session_id = str(session_id or "").strip()
        resolved_user_id = str(user_id or "").strip()
        if not resolved_session_id or not resolved_user_id:
            return None
        return self.sessions.ensure_session(
            session_id=resolved_session_id,
            user_id=resolved_user_id,
            title=title,
            memory_policy=memory_policy.model_dump(mode="json"),
            knowledge_scope=dict(knowledge_scope or {}),
            metadata={"runtime": "agno"},
        )

    def retrieve_context(
        self,
        *,
        session_id: str | None,
        user_id: str | None,
        query: str,
        policy: ScientificMemoryPolicy,
        knowledge_scope: dict[str, Any] | None = None,
    ) -> ScientificMemoryContext:
        agno_features = self.agno_feature_settings(
            policy=policy,
            history_message_count=1,
        )
        context = ScientificMemoryContext(
            policy=policy,
            agno_features=agno_features,
        )
        if policy.mode == "off":
            return context

        resolved_session_id = str(session_id or "").strip()
        resolved_user_id = str(user_id or "").strip()
        sections: list[str] = []
        if resolved_session_id and resolved_user_id and policy.session_summary:
            session_row = self.sessions.get_session(
                session_id=resolved_session_id, user_id=resolved_user_id
            )
            session_summary = str((session_row or {}).get("summary") or "").strip()
            if session_summary:
                sections.append(f"Current session summary:\n{session_summary}")
                context.session_summary_used = True
                context.hit_count += 1
                context.hits.append(
                    {
                        "scope": "session_summary",
                        "title": "Current session summary",
                        "score": 1.0,
                    }
                )

        if resolved_user_id and policy.user_memory:
            note_rows = self.notes.search_notes(
                query,
                user_id=resolved_user_id,
                scopes=["user_memory"],
                limit=3,
            )
            if note_rows:
                rendered = self._render_note_block("User memory", note_rows)
                if rendered:
                    sections.append(rendered)
                context.hit_count += len(note_rows)
                context.hits.extend(self._note_metadata_rows(note_rows))

        if not sections:
            return context

        body = "\n\n".join(section for section in sections if str(section or "").strip()).strip()
        if not body:
            return context
        context.context_chars = len(body)
        context.system_messages.append({"role": "system", "content": body})
        return context

    def update_after_run(
        self,
        *,
        session_id: str | None,
        user_id: str | None,
        title: str,
        latest_user_text: str,
        response_text: str,
        policy: ScientificMemoryPolicy,
        knowledge_scope: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> ScientificMemoryUpdate:
        update = ScientificMemoryUpdate()
        if policy.mode == "off":
            update.skipped.append("memory_disabled")
            return update

        resolved_session_id = str(session_id or "").strip()
        resolved_user_id = str(user_id or "").strip()
        session_row = self.ensure_session(
            session_id=resolved_session_id or None,
            user_id=resolved_user_id or None,
            title=title,
            memory_policy=policy,
            knowledge_scope=knowledge_scope,
        )
        if policy.session_summary and resolved_session_id and resolved_user_id:
            existing_summary = str((session_row or {}).get("summary") or "").strip()
            next_summary = self._compose_session_summary(
                existing_summary=existing_summary,
                latest_user_text=latest_user_text,
                response_text=response_text,
            )
            if next_summary and next_summary != existing_summary:
                self.sessions.update_session(
                    session_id=resolved_session_id,
                    user_id=resolved_user_id,
                    summary=next_summary,
                    memory_policy=policy.model_dump(mode="json"),
                    knowledge_scope=dict(knowledge_scope or {}),
                )
                update.summary_updated = True
                update.summary = next_summary

        if policy.user_memory and resolved_user_id:
            candidate = self._extract_user_memory_candidate(
                latest_user_text=latest_user_text,
                user_id=resolved_user_id,
                session_id=resolved_session_id or None,
                run_id=run_id,
            )
            if candidate is not None:
                row = self.notes.upsert_note(**candidate)
                update.writes.append(
                    {
                        "kind": "user_memory",
                        "note_id": row.get("note_id"),
                        "title": row.get("title"),
                        "scope": row.get("scope"),
                    }
                )
        return update

    @staticmethod
    def _compose_session_summary(
        *,
        existing_summary: str,
        latest_user_text: str,
        response_text: str,
    ) -> str:
        focus = str(latest_user_text or "").strip()
        response = str(response_text or "").strip()
        summary_parts: list[str] = []
        if existing_summary:
            summary_parts.append(existing_summary[:900].strip())
        if focus:
            summary_parts.append(f"Latest question: {focus[:240].strip()}")
        if response:
            summary_parts.append(f"Most recent answer: {response[:360].strip()}")
        combined = "\n".join(part for part in summary_parts if part).strip()
        return combined[:1400].strip()

    @staticmethod
    def _render_note_block(label: str, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return ""
        lines = [f"{label}:"]
        for row in rows:
            title = str(row.get("title") or "Note").strip()
            body = str(row.get("body") or "").strip()
            if not body:
                continue
            lines.append(f"{title}")
            lines.append(body[:280].strip())
        return "\n".join(lines).strip()

    @staticmethod
    def _note_metadata_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "note_id": row.get("note_id"),
                "scope": row.get("scope"),
                "title": row.get("title"),
                "body": row.get("body"),
                "score": row.get("search_score") or row.get("score"),
                "provenance": dict(row.get("provenance") or {}),
            }
            for row in rows
        ]

    @staticmethod
    def fallback_response(
        *,
        latest_user_text: str,
        memory_context: dict[str, Any],
    ) -> str | None:
        question = str(latest_user_text or "").strip().lower()
        if not question:
            return None
        recall_prompt = (
            "what" in question
            and ("favorite" in question or "prefer" in question)
            and (
                "reply with only" in question
                or "what did i just say" in question
                or "what reagent" in question
            )
        )
        if not recall_prompt:
            return None
        for hit in list(memory_context.get("hits") or []):
            if not isinstance(hit, dict):
                continue
            if str(hit.get("scope") or "").strip().lower() != "user_memory":
                continue
            body = str(hit.get("body") or "").strip()
            favorite_match = re.search(
                r"favorite [a-z0-9 _-]{2,64} is ([a-z0-9 ,._+-]{2,120})",
                body,
                flags=re.IGNORECASE,
            )
            if favorite_match:
                return str(favorite_match.group(1) or "").strip().rstrip(".")
            prefer_match = re.search(
                r"prefers ([a-z0-9 ,._+-]{2,120})",
                body,
                flags=re.IGNORECASE,
            )
            if prefer_match:
                return str(prefer_match.group(1) or "").strip().rstrip(".")
        return None

    @staticmethod
    def _extract_user_memory_candidate(
        *,
        latest_user_text: str,
        user_id: str,
        session_id: str | None,
        run_id: str | None,
    ) -> dict[str, Any] | None:
        text = str(latest_user_text or "").strip()
        if not text:
            return None

        favorite_match = re.search(
            r"\bmy favorite ([a-z0-9 _-]{2,64}) is ([a-z0-9 ,._+-]{2,120})",
            text,
            flags=re.IGNORECASE,
        )
        if favorite_match:
            subject = str(favorite_match.group(1) or "").strip()
            value = str(favorite_match.group(2) or "").strip().rstrip(".")
            return {
                "user_id": user_id,
                "session_id": session_id,
                "scope": "user_memory",
                "title": f"User preference: favorite {subject}",
                "body": f"The user said their favorite {subject} is {value}.",
                "tags": ["user_memory", "preference", subject.replace(" ", "_")],
                "provenance": {
                    "source": "user_message",
                    "run_id": run_id,
                },
                "score": 0.78,
            }

        prefer_match = re.search(
            r"\bi prefer ([a-z0-9 ,._+-]{3,120})",
            text,
            flags=re.IGNORECASE,
        )
        if prefer_match:
            value = str(prefer_match.group(1) or "").strip().rstrip(".")
            return {
                "user_id": user_id,
                "session_id": session_id,
                "scope": "user_memory",
                "title": "User preference",
                "body": f"The user explicitly prefers {value}.",
                "tags": ["user_memory", "preference"],
                "provenance": {
                    "source": "user_message",
                    "run_id": run_id,
                },
                "score": 0.72,
            }
        return None
