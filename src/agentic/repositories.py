from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from typing import Any
from uuid import uuid4

from .db import AgenticDb


def _utcnow() -> str:
    return datetime.utcnow().isoformat()


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, default=str)


def _decode(value: Any) -> Any:
    if value in (None, "", b""):
        return {}
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return {}


def _normalized_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _tokenize_search_text(value: Any) -> list[str]:
    return re.findall(r"[a-z0-9]{2,}", _normalized_text(value))


def _hashed_embedding(text: Any, *, dimensions: int = 192) -> list[float]:
    vector = [0.0] * dimensions
    tokens = _tokenize_search_text(text)
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        weight = 1.0 + min(len(token) / 16.0, 1.5)
        vector[index] += sign * weight
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [round(value / norm, 6) for value in vector]


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{value:.6f}" for value in vector) + "]"


def _note_search_score(query: str, row: dict[str, Any]) -> float:
    query_text = _normalized_text(query)
    if not query_text:
        return 0.0
    haystack = " ".join(
        [
            _normalized_text(row.get("title")),
            _normalized_text(row.get("body")),
            " ".join(str(item or "").strip().lower() for item in list(row.get("tags") or [])),
        ]
    ).strip()
    if not haystack:
        return 0.0
    score = 0.0
    if query_text in haystack:
        score += 0.8
    query_tokens = _tokenize_search_text(query_text)
    if not query_tokens:
        return score
    matched = sum(1 for token in query_tokens if token in haystack)
    score += matched / max(1, len(query_tokens))
    return round(score, 4)


class SessionRepository:
    def __init__(self, db: AgenticDb) -> None:
        self.db = db

    @staticmethod
    def _session_user_matches(existing_user_id: Any, requested_user_id: str | None) -> bool:
        existing = _optional_text(existing_user_id)
        requested = _optional_text(requested_user_id)
        return requested is None or existing is None or existing == requested

    @staticmethod
    def _decode_session_row(row: dict[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        payload["memory_policy"] = _decode(payload.pop("memory_policy_json", None))
        payload["knowledge_scope"] = _decode(payload.pop("knowledge_scope_json", None))
        payload["metadata"] = _decode(payload.pop("metadata_json", None))
        return payload

    def _claim_session_user(self, *, session_id: str, user_id: str | None) -> None:
        resolved_user_id = _optional_text(user_id)
        if not resolved_user_id:
            return
        self.db.write(
            """
            UPDATE agentic_sessions
            SET user_id=?, updated_at=?
            WHERE session_id=? AND (user_id IS NULL OR user_id='')
            """,
            (resolved_user_id, _utcnow(), session_id),
        )

    def create_session(
        self,
        *,
        user_id: str,
        title: str,
        session_id: str | None = None,
        status: str = "active",
        summary: str | None = None,
        memory_policy: dict[str, Any] | None = None,
        knowledge_scope: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = _utcnow()
        session_id = str(session_id or uuid4().hex)
        try:
            self.db.write(
                """
                INSERT INTO agentic_sessions(
                    session_id, user_id, title, status, summary,
                    memory_policy_json, knowledge_scope_json, metadata_json,
                    created_at, updated_at
                )
                VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    session_id,
                    user_id,
                    title,
                    status,
                    summary,
                    _json(memory_policy),
                    _json(knowledge_scope),
                    _json(metadata),
                    now,
                    now,
                ),
            )
        except Exception:
            existing = self.get_session_any(session_id=session_id)
            if existing is not None and self._session_user_matches(existing.get("user_id"), user_id):
                self._claim_session_user(session_id=session_id, user_id=user_id)
                return self.get_session_any(session_id=session_id) or existing
            raise
        return self.get_session(session_id=session_id, user_id=user_id) or {}

    def ensure_session(
        self,
        *,
        session_id: str,
        user_id: str,
        title: str,
        status: str = "active",
        summary: str | None = None,
        memory_policy: dict[str, Any] | None = None,
        knowledge_scope: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        existing = self.get_session(session_id=session_id, user_id=user_id)
        if existing is not None:
            return existing
        existing_any = self.get_session_any(session_id=session_id)
        if existing_any is not None:
            if not self._session_user_matches(existing_any.get("user_id"), user_id):
                raise ValueError(f"Session {session_id} already belongs to another user.")
            self._claim_session_user(session_id=session_id, user_id=user_id)
            return self.get_session_any(session_id=session_id) or existing_any
        return self.create_session(
            session_id=session_id,
            user_id=user_id,
            title=title,
            status=status,
            summary=summary,
            memory_policy=memory_policy,
            knowledge_scope=knowledge_scope,
            metadata=metadata,
        )

    def get_session(self, *, session_id: str, user_id: str) -> dict[str, Any] | None:
        row = self.db.fetchone(
            """
            SELECT session_id, user_id, title, status, summary, memory_policy_json,
                   knowledge_scope_json, metadata_json, created_at, updated_at
            FROM agentic_sessions
            WHERE session_id=? AND user_id=?
            """,
            (session_id, user_id),
        )
        if row is None:
            return None
        return self._decode_session_row(row)

    def get_session_any(self, *, session_id: str) -> dict[str, Any] | None:
        row = self.db.fetchone(
            """
            SELECT session_id, user_id, title, status, summary, memory_policy_json,
                   knowledge_scope_json, metadata_json, created_at, updated_at
            FROM agentic_sessions
            WHERE session_id=?
            """,
            (session_id,),
        )
        if row is None:
            return None
        return self._decode_session_row(row)

    def list_sessions(self, *, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.db.fetchall(
            """
            SELECT session_id, user_id, title, status, summary, memory_policy_json,
                   knowledge_scope_json, metadata_json, created_at, updated_at
            FROM agentic_sessions
            WHERE user_id=?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (user_id, int(limit)),
        )
        for row in rows:
            row["memory_policy"] = _decode(row.pop("memory_policy_json", None))
            row["knowledge_scope"] = _decode(row.pop("knowledge_scope_json", None))
            row["metadata"] = _decode(row.pop("metadata_json", None))
        return rows

    def update_session(
        self,
        *,
        session_id: str,
        user_id: str,
        title: str | None = None,
        status: str | None = None,
        summary: str | None = None,
        memory_policy: dict[str, Any] | None = None,
        knowledge_scope: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        now = _utcnow()
        existing = self.get_session(session_id=session_id, user_id=user_id)
        if existing is None:
            return None
        self.db.write(
            """
            UPDATE agentic_sessions
            SET title=?, status=?, summary=?,
                memory_policy_json=?, knowledge_scope_json=?, metadata_json=?,
                updated_at=?
            WHERE session_id=? AND user_id=?
            """,
            (
                title if title is not None else existing.get("title"),
                status if status is not None else existing.get("status"),
                summary if summary is not None else existing.get("summary"),
                _json(memory_policy if memory_policy is not None else existing.get("memory_policy")),
                _json(knowledge_scope if knowledge_scope is not None else existing.get("knowledge_scope")),
                _json(metadata if metadata is not None else existing.get("metadata")),
                now,
                session_id,
                user_id,
            ),
        )
        return self.get_session(session_id=session_id, user_id=user_id)

    def append_message(
        self,
        *,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        attachments: list[dict[str, Any]] | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        message_id: str | None = None,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        payload_created_at = str(created_at or _utcnow())
        resolved_message_id = str(message_id or uuid4().hex)
        self.db.write(
            """
            INSERT INTO agentic_messages(
                message_id, session_id, user_id, role, content,
                attachments_json, run_id, metadata_json, created_at
            )
            VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                resolved_message_id,
                session_id,
                user_id,
                role,
                content,
                _json(attachments or []),
                run_id,
                _json(metadata),
                payload_created_at,
            ),
        )
        self.update_session(session_id=session_id, user_id=user_id)
        return {
            "message_id": resolved_message_id,
            "session_id": session_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "attachments": attachments or [],
            "run_id": run_id,
            "metadata": metadata or {},
            "created_at": payload_created_at,
        }

    def list_messages(self, *, session_id: str, user_id: str, limit: int = 500) -> list[dict[str, Any]]:
        rows = self.db.fetchall(
            """
            SELECT message_id, session_id, user_id, role, content,
                   attachments_json, run_id, metadata_json, created_at
            FROM agentic_messages
            WHERE session_id=? AND user_id=?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (session_id, user_id, int(limit)),
        )
        for row in rows:
            row["attachments"] = _decode(row.pop("attachments_json", None))
            row["metadata"] = _decode(row.pop("metadata_json", None))
        return rows


class RunRepository:
    def __init__(self, db: AgenticDb) -> None:
        self.db = db

    def create_run(
        self,
        *,
        run_id: str,
        session_id: str,
        user_id: str,
        workflow_name: str,
        status: str,
        current_step: str,
        checkpoint_state: dict[str, Any] | None = None,
        budget_state: dict[str, Any] | None = None,
        response_text: str | None = None,
        trace_group_id: str | None = None,
        metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        now = _utcnow()
        self.db.write(
            """
            INSERT INTO agentic_runs(
                run_id, session_id, user_id, workflow_name, status, current_step,
                checkpoint_state_json, budget_state_json, response_text, trace_group_id,
                metrics_json, error, metadata_json, created_at, updated_at
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run_id,
                session_id,
                user_id,
                workflow_name,
                status,
                current_step,
                _json(checkpoint_state),
                _json(budget_state),
                response_text,
                trace_group_id,
                _json(metrics),
                error,
                _json(metadata),
                now,
                now,
            ),
        )
        return self.get_run(run_id=run_id, user_id=user_id) or {}

    def get_run(self, *, run_id: str, user_id: str) -> dict[str, Any] | None:
        row = self.db.fetchone(
            """
            SELECT run_id, session_id, user_id, workflow_name, status, current_step,
                   checkpoint_state_json, budget_state_json, response_text, trace_group_id,
                   metrics_json, error, metadata_json, created_at, updated_at
            FROM agentic_runs
            WHERE run_id=? AND user_id=?
            """,
            (run_id, user_id),
        )
        if row is None:
            return None
        row["checkpoint_state"] = _decode(row.pop("checkpoint_state_json", None))
        row["budget_state"] = _decode(row.pop("budget_state_json", None))
        row["metrics"] = _decode(row.pop("metrics_json", None))
        row["metadata"] = _decode(row.pop("metadata_json", None))
        return row

    def update_run(
        self,
        *,
        run_id: str,
        user_id: str,
        status: str | None = None,
        current_step: str | None = None,
        checkpoint_state: dict[str, Any] | None = None,
        budget_state: dict[str, Any] | None = None,
        response_text: str | None = None,
        metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any] | None:
        existing = self.get_run(run_id=run_id, user_id=user_id)
        if existing is None:
            return None
        self.db.write(
            """
            UPDATE agentic_runs
            SET status=?, current_step=?, checkpoint_state_json=?, budget_state_json=?,
                response_text=?, metrics_json=?, error=?, metadata_json=?, updated_at=?
            WHERE run_id=? AND user_id=?
            """,
            (
                status if status is not None else existing.get("status"),
                current_step if current_step is not None else existing.get("current_step"),
                _json(checkpoint_state if checkpoint_state is not None else existing.get("checkpoint_state")),
                _json(budget_state if budget_state is not None else existing.get("budget_state")),
                response_text if response_text is not None else existing.get("response_text"),
                _json(metrics if metrics is not None else existing.get("metrics")),
                error if error is not None else existing.get("error"),
                _json(metadata if metadata is not None else existing.get("metadata")),
                _utcnow(),
                run_id,
                user_id,
            ),
        )
        return self.get_run(run_id=run_id, user_id=user_id)


class ScientificNoteRepository:
    def __init__(self, db: AgenticDb) -> None:
        self.db = db

    def upsert_note(
        self,
        *,
        user_id: str,
        scope: str,
        title: str,
        body: str,
        note_id: str | None = None,
        session_id: str | None = None,
        project_id: str | None = None,
        tags: list[str] | None = None,
        provenance: dict[str, Any] | None = None,
        score: float = 0.5,
    ) -> dict[str, Any]:
        now = _utcnow()
        resolved_note_id = str(note_id or uuid4().hex)
        self.db.write(
            """
            INSERT INTO agentic_notes(
                note_id, user_id, session_id, project_id, scope, title, body,
                tags_json, provenance_json, score, created_at, updated_at
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(note_id) DO UPDATE SET
                user_id=excluded.user_id,
                session_id=excluded.session_id,
                project_id=excluded.project_id,
                scope=excluded.scope,
                title=excluded.title,
                body=excluded.body,
                tags_json=excluded.tags_json,
                provenance_json=excluded.provenance_json,
                score=excluded.score,
                updated_at=excluded.updated_at
            """,
            (
                resolved_note_id,
                user_id,
                session_id,
                project_id,
                scope,
                title,
                body,
                _json(tags or []),
                _json(provenance),
                float(score),
                now,
                now,
            ),
        )
        self._upsert_vector(
            note_id=resolved_note_id,
            title=title,
            body=body,
            tags=list(tags or []),
        )
        return self.get_note(note_id=resolved_note_id, user_id=user_id) or {}

    def get_note(self, *, note_id: str, user_id: str) -> dict[str, Any] | None:
        row = self.db.fetchone(
            """
            SELECT note_id, user_id, session_id, project_id, scope, title, body,
                   tags_json, provenance_json, score, created_at, updated_at
            FROM agentic_notes
            WHERE note_id=? AND user_id=?
            """,
            (note_id, user_id),
        )
        if row is None:
            return None
        row["tags"] = _decode(row.pop("tags_json", None))
        row["provenance"] = _decode(row.pop("provenance_json", None))
        return row

    def list_notes(
        self,
        *,
        user_id: str,
        scopes: list[str] | None = None,
        session_id: str | None = None,
        project_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT note_id, user_id, session_id, project_id, scope, title, body,
                   tags_json, provenance_json, score, created_at, updated_at
            FROM agentic_notes
            WHERE user_id=?
        """
        params: list[Any] = [user_id]
        normalized_scopes = [str(item or "").strip() for item in list(scopes or []) if str(item or "").strip()]
        if normalized_scopes:
            query += " AND scope IN (" + ",".join("?" for _ in normalized_scopes) + ")"
            params.extend(normalized_scopes)
        if session_id:
            query += " AND session_id=?"
            params.append(session_id)
        if project_id:
            query += " AND project_id=?"
            params.append(project_id)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.db.fetchall(query, tuple(params))
        for row in rows:
            row["tags"] = _decode(row.pop("tags_json", None))
            row["provenance"] = _decode(row.pop("provenance_json", None))
        return rows

    def search_notes(
        self,
        query: str,
        *,
        user_id: str,
        scopes: list[str] | None = None,
        session_id: str | None = None,
        project_id: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        rows = self.list_notes(
            user_id=user_id,
            scopes=scopes,
            session_id=session_id,
            project_id=project_id,
            limit=max(25, int(limit) * 12),
        )
        if not rows:
            return []
        vector_scores = self._vector_scores(query=query, note_ids=[str(row.get("note_id") or "") for row in rows])
        ranked: list[dict[str, Any]] = []
        for row in rows:
            lexical_score = _note_search_score(query, row)
            vector_score = float(vector_scores.get(str(row.get("note_id") or ""), 0.0))
            combined_score = max(lexical_score, 0.6 * lexical_score + 0.4 * vector_score)
            if combined_score <= 0.0:
                continue
            ranked.append(
                {
                    **row,
                    "search_score": round(combined_score, 4),
                }
            )
        ranked.sort(
            key=lambda item: (
                -float(item.get("search_score") or 0.0),
                -float(item.get("score") or 0.0),
                str(item.get("updated_at") or ""),
            )
        )
        return ranked[: max(1, int(limit))]

    def _upsert_vector(
        self,
        *,
        note_id: str,
        title: str,
        body: str,
        tags: list[str],
    ) -> None:
        if self.db.backend != "postgres":
            return
        embedding = _vector_literal(
            _hashed_embedding("\n".join([str(title or ""), str(body or ""), " ".join(tags)]))
        )
        try:
            self.db.write(
                """
                INSERT INTO agentic_note_vectors(note_id, embedding, updated_at)
                VALUES(?, ?::vector, ?)
                ON CONFLICT(note_id) DO UPDATE SET
                    embedding=excluded.embedding,
                    updated_at=excluded.updated_at
                """,
                (note_id, embedding, _utcnow()),
            )
        except Exception:
            return

    def _vector_scores(self, *, query: str, note_ids: list[str]) -> dict[str, float]:
        if self.db.backend != "postgres":
            return {}
        filtered_note_ids = [str(item or "").strip() for item in note_ids if str(item or "").strip()]
        if not filtered_note_ids:
            return {}
        embedding = _vector_literal(_hashed_embedding(query))
        placeholders = ",".join("?" for _ in filtered_note_ids)
        try:
            rows = self.db.fetchall(
                f"""
                SELECT note_id, GREATEST(0, 1 - (embedding <=> ?::vector)) AS similarity
                FROM agentic_note_vectors
                WHERE note_id IN ({placeholders})
                ORDER BY embedding <=> ?::vector
                LIMIT ?
                """,
                (embedding, *filtered_note_ids, embedding, len(filtered_note_ids)),
            )
        except Exception:
            return {}
        return {
            str(row.get("note_id") or ""): round(float(row.get("similarity") or 0.0), 4)
            for row in rows
            if str(row.get("note_id") or "").strip()
        }


class EventRepository:
    def __init__(self, db: AgenticDb) -> None:
        self.db = db

    def append_event(
        self,
        *,
        run_id: str,
        session_id: str,
        user_id: str,
        event_kind: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        agent_name: str | None = None,
        tool_name: str | None = None,
        level: str = "info",
        created_at: str | None = None,
    ) -> dict[str, Any]:
        event_id = uuid4().hex
        ts = str(created_at or _utcnow())
        self.db.write(
            """
            INSERT INTO agentic_events(
                event_id, run_id, session_id, user_id, event_kind, event_type,
                agent_name, tool_name, level, payload_json, created_at
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                event_id,
                run_id,
                session_id,
                user_id,
                event_kind,
                event_type,
                agent_name,
                tool_name,
                level,
                _json(payload),
                ts,
            ),
        )
        return {
            "event_id": event_id,
            "run_id": run_id,
            "session_id": session_id,
            "user_id": user_id,
            "event_kind": event_kind,
            "event_type": event_type,
            "agent_name": agent_name,
            "tool_name": tool_name,
            "level": level,
            "payload": payload or {},
            "created_at": ts,
        }

    def list_events(self, *, run_id: str, user_id: str, limit: int = 500) -> list[dict[str, Any]]:
        rows = self.db.fetchall(
            """
            SELECT event_id, run_id, session_id, user_id, event_kind, event_type,
                   agent_name, tool_name, level, payload_json, created_at
            FROM agentic_events
            WHERE run_id=? AND user_id=?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (run_id, user_id, int(limit)),
        )
        for row in rows:
            row["payload"] = _decode(row.pop("payload_json", None))
        return rows


class ArtifactRepository:
    def __init__(self, db: AgenticDb) -> None:
        self.db = db

    def upsert_artifact(
        self,
        *,
        artifact_id: str,
        run_id: str,
        session_id: str,
        user_id: str,
        kind: str,
        title: str | None,
        path: str | None,
        source_path: str | None = None,
        preview_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = _utcnow()
        self.db.write(
            """
            INSERT INTO agentic_artifacts(
                artifact_id, run_id, session_id, user_id, kind, title,
                path, source_path, preview_path, metadata_json, created_at, updated_at
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(artifact_id) DO UPDATE SET
                kind=excluded.kind,
                title=excluded.title,
                path=excluded.path,
                source_path=excluded.source_path,
                preview_path=excluded.preview_path,
                metadata_json=excluded.metadata_json,
                updated_at=excluded.updated_at
            """,
            (
                artifact_id,
                run_id,
                session_id,
                user_id,
                kind,
                title,
                path,
                source_path,
                preview_path,
                _json(metadata),
                now,
                now,
            ),
        )
        return {
            "artifact_id": artifact_id,
            "run_id": run_id,
            "session_id": session_id,
            "user_id": user_id,
            "kind": kind,
            "title": title,
            "path": path,
            "source_path": source_path,
            "preview_path": preview_path,
            "metadata": metadata or {},
            "created_at": now,
            "updated_at": now,
        }

    def list_artifacts(self, *, run_id: str, user_id: str, limit: int = 500) -> list[dict[str, Any]]:
        rows = self.db.fetchall(
            """
            SELECT artifact_id, run_id, session_id, user_id, kind, title,
                   path, source_path, preview_path, metadata_json, created_at, updated_at
            FROM agentic_artifacts
            WHERE run_id=? AND user_id=?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (run_id, user_id, int(limit)),
        )
        for row in rows:
            row["metadata"] = _decode(row.pop("metadata_json", None))
        return rows


class ApprovalRepository:
    def __init__(self, db: AgenticDb) -> None:
        self.db = db

    def create_approval(
        self,
        *,
        run_id: str,
        session_id: str,
        user_id: str,
        action_type: str,
        tool_name: str | None,
        request_payload: dict[str, Any] | None = None,
        status: str = "pending",
    ) -> dict[str, Any]:
        approval_id = uuid4().hex
        now = _utcnow()
        self.db.write(
            """
            INSERT INTO agentic_approvals(
                approval_id, run_id, session_id, user_id, action_type, tool_name,
                status, request_payload_json, resolution_json, created_at, updated_at
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                approval_id,
                run_id,
                session_id,
                user_id,
                action_type,
                tool_name,
                status,
                _json(request_payload),
                _json({}),
                now,
                now,
            ),
        )
        return self.get_approval(approval_id=approval_id, user_id=user_id) or {}

    def get_approval(self, *, approval_id: str, user_id: str) -> dict[str, Any] | None:
        row = self.db.fetchone(
            """
            SELECT approval_id, run_id, session_id, user_id, action_type, tool_name,
                   status, request_payload_json, resolution_json, created_at, updated_at
            FROM agentic_approvals
            WHERE approval_id=? AND user_id=?
            """,
            (approval_id, user_id),
        )
        if row is None:
            return None
        row["request_payload"] = _decode(row.pop("request_payload_json", None))
        row["resolution"] = _decode(row.pop("resolution_json", None))
        return row

    def list_run_approvals(self, *, run_id: str, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.db.fetchall(
            """
            SELECT approval_id, run_id, session_id, user_id, action_type, tool_name,
                   status, request_payload_json, resolution_json, created_at, updated_at
            FROM agentic_approvals
            WHERE run_id=? AND user_id=?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (run_id, user_id, int(limit)),
        )
        for row in rows:
            row["request_payload"] = _decode(row.pop("request_payload_json", None))
            row["resolution"] = _decode(row.pop("resolution_json", None))
        return rows

    def update_approval(
        self,
        *,
        approval_id: str,
        user_id: str,
        status: str,
        resolution: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        existing = self.get_approval(approval_id=approval_id, user_id=user_id)
        if existing is None:
            return None
        self.db.write(
            """
            UPDATE agentic_approvals
            SET status=?, resolution_json=?, updated_at=?
            WHERE approval_id=? AND user_id=?
            """,
            (status, _json(resolution), _utcnow(), approval_id, user_id),
        )
        return self.get_approval(approval_id=approval_id, user_id=user_id)
