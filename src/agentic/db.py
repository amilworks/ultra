from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator


class AgenticDb:
    """Small SQL helper for the v3 agentic persistence layer."""

    def __init__(self, db_target: str) -> None:
        self.db_target = str(db_target or "").strip() or os.path.join("data", "runs.db")
        self.backend = self._detect_backend(self.db_target)
        if self.backend == "sqlite":
            os.makedirs(os.path.dirname(self.db_target) or ".", exist_ok=True)
            self._postgres_module = None
        else:
            try:
                import psycopg  # type: ignore
            except Exception as exc:  # pragma: no cover - env-specific
                raise RuntimeError(
                    "Postgres agentic backend requested but psycopg is not installed."
                ) from exc
            self._postgres_module = psycopg
            self.db_target = self._normalize_postgres_connect_target(self.db_target)
        self._init_schema()

    @staticmethod
    def _detect_backend(target: str) -> str:
        lowered = str(target or "").strip().lower()
        if lowered.startswith(("postgres://", "postgresql://", "postgresql+psycopg://")):
            return "postgres"
        return "sqlite"

    @staticmethod
    def _normalize_postgres_connect_target(target: str) -> str:
        normalized = str(target or "").strip()
        if normalized.startswith("postgresql+psycopg://"):
            return "postgresql://" + normalized[len("postgresql+psycopg://") :]
        return normalized

    @contextmanager
    def conn(self) -> Iterator[Any]:
        if self.backend == "sqlite":
            conn = sqlite3.connect(self.db_target, check_same_thread=False)
            try:
                yield conn
            finally:
                conn.close()
            return

        assert self._postgres_module is not None
        conn = self._postgres_module.connect(self.db_target, autocommit=False)
        try:
            yield conn
        finally:
            conn.close()

    def _rewrite_query(self, query: str) -> str:
        if self.backend == "postgres":
            return query.replace("?", "%s")
        return query

    def execute(self, conn: Any, query: str, params: tuple[Any, ...] | list[Any] = ()) -> Any:
        return conn.execute(self._rewrite_query(query), params)

    def fetchone(self, query: str, params: tuple[Any, ...] | list[Any] = ()) -> dict[str, Any] | None:
        with self.conn() as conn:
            cur = self.execute(conn, query, params)
            row = cur.fetchone()
            if row is None:
                return None
            columns = [col[0] for col in cur.description or []]
            return {columns[idx]: row[idx] for idx in range(len(columns))}

    def fetchall(self, query: str, params: tuple[Any, ...] | list[Any] = ()) -> list[dict[str, Any]]:
        with self.conn() as conn:
            cur = self.execute(conn, query, params)
            rows = cur.fetchall()
            columns = [col[0] for col in cur.description or []]
            return [{columns[idx]: row[idx] for idx in range(len(columns))} for row in rows]

    def write(self, query: str, params: tuple[Any, ...] | list[Any] = ()) -> None:
        with self.conn() as conn:
            self.execute(conn, query, params)
            conn.commit()

    def _init_schema(self) -> None:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS agentic_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                title TEXT,
                status TEXT,
                summary TEXT,
                memory_policy_json TEXT,
                knowledge_scope_json TEXT,
                metadata_json TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS agentic_messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_id TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                attachments_json TEXT,
                run_id TEXT,
                metadata_json TEXT,
                created_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS agentic_runs (
                run_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_id TEXT,
                workflow_name TEXT,
                status TEXT,
                current_step TEXT,
                checkpoint_state_json TEXT,
                budget_state_json TEXT,
                response_text TEXT,
                trace_group_id TEXT,
                metrics_json TEXT,
                error TEXT,
                metadata_json TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS agentic_events (
                event_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                session_id TEXT,
                user_id TEXT,
                event_kind TEXT,
                event_type TEXT,
                agent_name TEXT,
                tool_name TEXT,
                level TEXT,
                payload_json TEXT,
                created_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS agentic_artifacts (
                artifact_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                session_id TEXT,
                user_id TEXT,
                kind TEXT,
                title TEXT,
                path TEXT,
                source_path TEXT,
                preview_path TEXT,
                metadata_json TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS agentic_approvals (
                approval_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                session_id TEXT,
                user_id TEXT,
                action_type TEXT,
                tool_name TEXT,
                status TEXT,
                request_payload_json TEXT,
                resolution_json TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS agentic_notes (
                note_id TEXT PRIMARY KEY,
                user_id TEXT,
                session_id TEXT,
                project_id TEXT,
                scope TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                tags_json TEXT,
                provenance_json TEXT,
                score REAL NOT NULL DEFAULT 0.5,
                created_at TEXT,
                updated_at TEXT
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_agentic_sessions_user_updated ON agentic_sessions(user_id, updated_at)",
            "CREATE INDEX IF NOT EXISTS idx_agentic_messages_session_created ON agentic_messages(session_id, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_agentic_runs_session_created ON agentic_runs(session_id, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_agentic_events_run_created ON agentic_events(run_id, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_agentic_artifacts_run_created ON agentic_artifacts(run_id, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_agentic_approvals_run_created ON agentic_approvals(run_id, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_agentic_notes_user_scope_updated ON agentic_notes(user_id, scope, updated_at)",
            "CREATE INDEX IF NOT EXISTS idx_agentic_notes_project_scope_updated ON agentic_notes(project_id, scope, updated_at)",
            "CREATE INDEX IF NOT EXISTS idx_agentic_notes_session_scope_updated ON agentic_notes(session_id, scope, updated_at)",
        ]
        with self.conn() as conn:
            if self.backend == "postgres":
                vector_enabled = True
                try:
                    self.execute(conn, "CREATE EXTENSION IF NOT EXISTS vector")
                except Exception:
                    vector_enabled = False
                    conn.rollback()
            for statement in statements:
                self.execute(conn, statement)
            if self.backend == "postgres" and vector_enabled:
                try:
                    self.execute(
                        conn,
                        """
                        CREATE TABLE IF NOT EXISTS agentic_note_vectors (
                            note_id TEXT PRIMARY KEY REFERENCES agentic_notes(note_id) ON DELETE CASCADE,
                            embedding vector(192),
                            updated_at TEXT
                        )
                        """,
                    )
                    self.execute(
                        conn,
                        """
                        CREATE INDEX IF NOT EXISTS idx_agentic_note_vectors_embedding
                        ON agentic_note_vectors
                        USING hnsw (embedding vector_cosine_ops)
                        """,
                    )
                except Exception:
                    conn.rollback()
                    pass
            conn.commit()
