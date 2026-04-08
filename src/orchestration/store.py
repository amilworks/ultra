from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Iterator
from uuid import uuid4

from src.orchestration.models import RunStatus, WorkflowPlan, WorkflowRun


class _NoopLock:
    def __enter__(self) -> "_NoopLock":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


class RunStore:
    """Persistent store for runs, events, uploads, and user conversation state.
    
    Notes
    -----
    Supports SQLite path targets and Postgres URLs.
    """

    _CONVERSATION_SCOPE_SEPARATOR = "::"

    def __init__(self, db_target: str):
        self.db_target = str(db_target or "").strip() or os.path.join("data", "runs.db")
        self._backend = self._detect_backend(self.db_target)
        self._lock: Any = threading.RLock() if self._backend == "sqlite" else _NoopLock()
        self._sqlite_fts_enabled = False
        if self._backend == "sqlite":
            os.makedirs(os.path.dirname(self.db_target) or ".", exist_ok=True)
            self._postgres_module = None
        else:
            try:
                import psycopg  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    "Postgres backend requested but psycopg is not installed. "
                    "Install `psycopg[binary]`."
                ) from exc
            self._postgres_module = psycopg
        self._init_db()

    @staticmethod
    def _detect_backend(target: str) -> str:
        lowered = target.lower()
        if lowered.startswith("postgres://") or lowered.startswith("postgresql://"):
            return "postgres"
        return "sqlite"

    def _rewrite_query(self, query: str) -> str:
        if self._backend == "postgres":
            return query.replace("?", "%s")
        return query

    def _execute(self, conn: Any, query: str, params: tuple[Any, ...] | list[Any] = ()) -> Any:
        return conn.execute(self._rewrite_query(query), params)

    def _scoped_conversation_id(self, conversation_id: str, user_id: str) -> str:
        raw_conversation_id = str(conversation_id or "").strip()
        raw_user_id = str(user_id or "").strip()
        if not raw_conversation_id:
            return raw_conversation_id
        prefix = f"{raw_user_id}{self._CONVERSATION_SCOPE_SEPARATOR}"
        if raw_conversation_id.startswith(prefix):
            return raw_conversation_id
        return f"{prefix}{raw_conversation_id}"

    def _conversation_lookup_ids(self, conversation_id: str, user_id: str) -> tuple[str, ...]:
        raw_conversation_id = str(conversation_id or "").strip()
        scoped_conversation_id = self._scoped_conversation_id(raw_conversation_id, user_id)
        if scoped_conversation_id == raw_conversation_id:
            return (raw_conversation_id,)
        return (scoped_conversation_id, raw_conversation_id)

    def _external_conversation_id(self, stored_conversation_id: str, user_id: str) -> str:
        raw_stored = str(stored_conversation_id or "").strip()
        raw_user = str(user_id or "").strip()
        if not raw_stored:
            return raw_stored
        prefix = f"{raw_user}{self._CONVERSATION_SCOPE_SEPARATOR}"
        if raw_stored.startswith(prefix):
            unscoped = raw_stored[len(prefix) :].strip()
            if unscoped:
                return unscoped
        return raw_stored

    @contextmanager
    def _conn(self) -> Iterator[Any]:
        if self._backend == "sqlite":
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

    def _sqlite_column_exists(self, conn: sqlite3.Connection, table: str, column: str) -> bool:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        for row in rows:
            if str(row[1]) == column:
                return True
        return False

    def _postgres_column_exists(self, conn: Any, table: str, column: str) -> bool:
        row = self._execute(
            conn,
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = ? AND column_name = ?
            LIMIT 1
            """,
            (table, column),
        ).fetchone()
        return row is not None

    def _ensure_runs_metadata_columns(self, conn: Any) -> None:
        required: dict[str, str] = {
            "user_id": "TEXT",
            "conversation_id": "TEXT",
            "workflow_kind": "TEXT",
            "mode": "TEXT",
            "parent_run_id": "TEXT",
            "planner_version": "TEXT",
            "agent_role": "TEXT",
            "checkpoint_state_json": "TEXT",
            "budget_state_json": "TEXT",
            "trace_group_id": "TEXT",
            "workflow_profile": "TEXT",
            "pedagogy_level": "TEXT",
        }
        if self._backend == "sqlite":
            assert isinstance(conn, sqlite3.Connection)
            for column, ddl in required.items():
                if not self._sqlite_column_exists(conn, "runs", column):
                    conn.execute(f"ALTER TABLE runs ADD COLUMN {column} {ddl}")
            return

        for column, ddl in required.items():
            if not self._postgres_column_exists(conn, "runs", column):
                self._execute(conn, f"ALTER TABLE runs ADD COLUMN {column} {ddl}")

    def _ensure_upload_columns(self, conn: Any) -> None:
        required: dict[str, str] = {
            "user_id": "TEXT",
            "source_type": "TEXT",
            "source_uri": "TEXT",
            "client_view_url": "TEXT",
            "image_service_url": "TEXT",
            "resource_kind": "TEXT",
            "thumbnail_path": "TEXT",
            "metadata_json": "TEXT",
            "updated_at": "TEXT",
            "deleted_at": "TEXT",
            "staging_path": "TEXT",
            "cache_path": "TEXT",
            "canonical_resource_uniq": "TEXT",
            "canonical_resource_uri": "TEXT",
            "sync_status": "TEXT",
            "sync_error": "TEXT",
            "sync_retry_count": "INTEGER",
            "sync_started_at": "TEXT",
            "sync_completed_at": "TEXT",
            "sync_run_id": "TEXT",
        }
        if self._backend == "sqlite":
            assert isinstance(conn, sqlite3.Connection)
            for column, ddl in required.items():
                if not self._sqlite_column_exists(conn, "uploads", column):
                    conn.execute(f"ALTER TABLE uploads ADD COLUMN {column} {ddl}")
            return

        for column, ddl in required.items():
            if not self._postgres_column_exists(conn, "uploads", column):
                self._execute(conn, f"ALTER TABLE uploads ADD COLUMN {column} {ddl}")

    def _ensure_conversation_message_time_column(self, conn: Any) -> None:
        if self._backend != "postgres":
            return
        row = self._execute(
            conn,
            """
            SELECT data_type
            FROM information_schema.columns
            WHERE table_name = 'conversation_messages'
              AND column_name = 'created_at_ms'
            LIMIT 1
            """,
        ).fetchone()
        if not row:
            return
        data_type = str(row[0] or "").strip().lower()
        if data_type in {"integer", "smallint"}:
            self._execute(
                conn,
                """
                ALTER TABLE conversation_messages
                ALTER COLUMN created_at_ms TYPE BIGINT
                USING created_at_ms::bigint
                """,
            )

    def _init_conversation_search_indexes(self, conn: Any) -> None:
        if self._backend == "postgres":
            self._execute(
                conn,
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_messages_fts
                ON conversation_messages
                USING GIN (to_tsvector('simple', COALESCE(content, '')))
                """,
            )
            return

        assert isinstance(conn, sqlite3.Connection)
        self._sqlite_fts_enabled = False
        try:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS conversation_messages_fts
                USING fts5(
                    conversation_id UNINDEXED,
                    user_id UNINDEXED,
                    message_id UNINDEXED,
                    role,
                    content,
                    run_id UNINDEXED,
                    created_at_ms UNINDEXED
                )
                """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS conversation_messages_ai
                AFTER INSERT ON conversation_messages
                BEGIN
                    INSERT INTO conversation_messages_fts(
                        rowid,
                        conversation_id,
                        user_id,
                        message_id,
                        role,
                        content,
                        run_id,
                        created_at_ms
                    ) VALUES (
                        NEW.rowid,
                        NEW.conversation_id,
                        NEW.user_id,
                        NEW.message_id,
                        NEW.role,
                        NEW.content,
                        NEW.run_id,
                        COALESCE(CAST(NEW.created_at_ms AS TEXT), '')
                    );
                END
                """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS conversation_messages_ad
                AFTER DELETE ON conversation_messages
                BEGIN
                    DELETE FROM conversation_messages_fts WHERE rowid = OLD.rowid;
                END
                """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS conversation_messages_au
                AFTER UPDATE ON conversation_messages
                BEGIN
                    DELETE FROM conversation_messages_fts WHERE rowid = OLD.rowid;
                    INSERT INTO conversation_messages_fts(
                        rowid,
                        conversation_id,
                        user_id,
                        message_id,
                        role,
                        content,
                        run_id,
                        created_at_ms
                    ) VALUES (
                        NEW.rowid,
                        NEW.conversation_id,
                        NEW.user_id,
                        NEW.message_id,
                        NEW.role,
                        NEW.content,
                        NEW.run_id,
                        COALESCE(CAST(NEW.created_at_ms AS TEXT), '')
                    );
                END
                """)
            fts_count_row = conn.execute(
                "SELECT COUNT(*) FROM conversation_messages_fts"
            ).fetchone()
            message_count_row = conn.execute(
                "SELECT COUNT(*) FROM conversation_messages"
            ).fetchone()
            fts_count = int((fts_count_row or [0])[0] or 0)
            message_count = int((message_count_row or [0])[0] or 0)
            if fts_count != message_count:
                conn.execute("DELETE FROM conversation_messages_fts")
                conn.execute("""
                    INSERT INTO conversation_messages_fts(
                        rowid,
                        conversation_id,
                        user_id,
                        message_id,
                        role,
                        content,
                        run_id,
                        created_at_ms
                    )
                    SELECT
                        rowid,
                        conversation_id,
                        user_id,
                        message_id,
                        role,
                        content,
                        run_id,
                        COALESCE(CAST(created_at_ms AS TEXT), '')
                    FROM conversation_messages
                    """)
            self._sqlite_fts_enabled = True
        except sqlite3.OperationalError:
            self._sqlite_fts_enabled = False

    def _init_db(self) -> None:
        with self._conn() as conn:
            if self._backend == "sqlite":
                assert isinstance(conn, sqlite3.Connection)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA temp_store=MEMORY")
                conn.execute("PRAGMA foreign_keys=ON")
            else:
                # Serialize schema initialization across concurrent processes/workers.
                # This avoids postgres catalog races during CREATE TABLE IF NOT EXISTS.
                self._execute(
                    conn,
                    "SELECT pg_advisory_xact_lock(hashtext(?))",
                    ("bisque_ultra_runstore_schema_init",),
                )
            if self._backend == "postgres":
                self._execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS runs (
                        run_id TEXT PRIMARY KEY,
                        goal TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        plan_json TEXT,
                        error TEXT,
                        user_id TEXT,
                        conversation_id TEXT,
                        workflow_kind TEXT,
                        mode TEXT,
                        parent_run_id TEXT,
                        planner_version TEXT,
                        agent_role TEXT,
                        checkpoint_state_json TEXT,
                        budget_state_json TEXT,
                        trace_group_id TEXT,
                        workflow_profile TEXT,
                        pedagogy_level TEXT
                    )
                    """,
                )
                self._execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS events (
                        event_id BIGSERIAL PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        ts TEXT NOT NULL,
                        level TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    )
                    """,
                )
            else:
                self._execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS runs (
                        run_id TEXT PRIMARY KEY,
                        goal TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        plan_json TEXT,
                        error TEXT,
                        user_id TEXT,
                        conversation_id TEXT,
                        workflow_kind TEXT,
                        mode TEXT,
                        parent_run_id TEXT,
                        planner_version TEXT,
                        agent_role TEXT,
                        checkpoint_state_json TEXT,
                        budget_state_json TEXT,
                        trace_group_id TEXT,
                        workflow_profile TEXT,
                        pedagogy_level TEXT
                    )
                    """,
                )
                self._execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS events (
                        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        ts TEXT NOT NULL,
                        level TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    )
                    """,
                )

            self._ensure_runs_metadata_columns(conn)

            self._execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS idempotency_keys (
                    endpoint TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (endpoint, idempotency_key)
                )
                """,
            )
            self._execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS uploads (
                    file_id TEXT PRIMARY KEY,
                    original_name TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    content_type TEXT,
                    size_bytes INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    user_id TEXT,
                    source_type TEXT,
                    source_uri TEXT,
                    client_view_url TEXT,
                    image_service_url TEXT,
                    resource_kind TEXT,
                    thumbnail_path TEXT,
                    metadata_json TEXT,
                    updated_at TEXT,
                    deleted_at TEXT,
                    staging_path TEXT,
                    cache_path TEXT,
                    canonical_resource_uniq TEXT,
                    canonical_resource_uri TEXT,
                    sync_status TEXT,
                    sync_error TEXT,
                    sync_retry_count INTEGER,
                    sync_started_at TEXT,
                    sync_completed_at TEXT,
                    sync_run_id TEXT
                )
                """,
            )
            self._ensure_upload_columns(conn)
            self._execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS run_metadata (
                    run_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    conversation_id TEXT,
                    created_at TEXT NOT NULL
                )
                """,
            )
            self._execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    state_json TEXT NOT NULL
                )
                """,
            )
            if self._backend == "postgres":
                self._execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS conversation_messages (
                        conversation_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        message_id TEXT NOT NULL,
                        message_index INTEGER NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at_ms BIGINT,
                        run_id TEXT,
                        payload_json TEXT,
                        PRIMARY KEY (conversation_id, message_id)
                    )
                    """,
                )
            else:
                self._execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS conversation_messages (
                        conversation_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        message_id TEXT NOT NULL,
                        message_index INTEGER NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at_ms INTEGER,
                        run_id TEXT,
                        payload_json TEXT,
                        PRIMARY KEY (conversation_id, message_id)
                    )
                    """,
                )
            self._ensure_conversation_message_time_column(conn)
            self._execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS conversation_context_memory (
                    conversation_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    source_message_count INTEGER NOT NULL,
                    source_hash TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    summary_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (conversation_id, user_id)
                )
                """,
            )
            self._execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS upload_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    content_type TEXT,
                    size_bytes INTEGER NOT NULL,
                    temp_path TEXT NOT NULL,
                    bytes_received INTEGER NOT NULL,
                    chunk_size_bytes INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    file_id TEXT,
                    sha256 TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """,
            )
            self._execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS threads (
                    thread_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    conversation_id TEXT,
                    title TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    latest_run_id TEXT,
                    checkpoint_id TEXT,
                    summary TEXT,
                    metadata_json TEXT
                )
                """,
            )
            self._execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS thread_messages (
                    thread_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    message_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    run_id TEXT,
                    metadata_json TEXT,
                    PRIMARY KEY (thread_id, message_id)
                )
                """,
            )
            self._execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS thread_summaries (
                    thread_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """,
            )
            if self._backend == "postgres":
                self._execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS task_attempts (
                        attempt_id BIGSERIAL PRIMARY KEY,
                        task_id TEXT NOT NULL,
                        attempt INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        finished_at TEXT,
                        error TEXT,
                        payload_json TEXT
                    )
                    """,
                )
                self._execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS task_heartbeats (
                        heartbeat_id BIGSERIAL PRIMARY KEY,
                        task_id TEXT NOT NULL,
                        lease_id TEXT NOT NULL,
                        heartbeat_at TEXT NOT NULL,
                        payload_json TEXT
                    )
                    """,
                )
            else:
                self._execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS task_attempts (
                        attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id TEXT NOT NULL,
                        attempt INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        finished_at TEXT,
                        error TEXT,
                        payload_json TEXT
                    )
                    """,
                )
                self._execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS task_heartbeats (
                        heartbeat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id TEXT NOT NULL,
                        lease_id TEXT NOT NULL,
                        heartbeat_at TEXT NOT NULL,
                        payload_json TEXT
                    )
                    """,
                )
            self._execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    thread_id TEXT,
                    user_id TEXT,
                    queue_name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    idempotency_key TEXT,
                    lease_owner TEXT,
                    lease_expires_at TEXT,
                    attempt_count INTEGER NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    next_attempt_at TEXT,
                    result_ref TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """,
            )
            self._execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS leases (
                    lease_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    scope TEXT,
                    expires_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT
                )
                """,
            )
            self._execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS resource_computations (
                    run_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    conversation_id TEXT,
                    tool_name TEXT NOT NULL,
                    file_sha256 TEXT NOT NULL,
                    file_id TEXT,
                    file_name TEXT NOT NULL,
                    file_name_norm TEXT NOT NULL,
                    source_path TEXT,
                    run_goal TEXT,
                    run_status TEXT NOT NULL,
                    run_created_at TEXT,
                    run_updated_at TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, tool_name, file_sha256)
                )
                """,
            )
            self._execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS training_managed_sources (
                    source_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    training_dataset_id TEXT,
                    remote_uri TEXT,
                    status TEXT NOT NULL,
                    metadata_json TEXT,
                    stats_json TEXT,
                    error TEXT,
                    last_sync_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(user_id, source_type, name)
                )
                """,
            )
            self._execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS training_datasets (
                    dataset_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """,
            )
            self._execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS training_dataset_items (
                    item_id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    split TEXT NOT NULL,
                    role TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    file_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    original_name TEXT,
                    sha256 TEXT,
                    size_bytes INTEGER NOT NULL,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(dataset_id, split, role, sample_id)
                )
                """,
            )
            self._execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS training_jobs (
                    job_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    job_type TEXT NOT NULL,
                    dataset_id TEXT,
                    model_key TEXT NOT NULL,
                    model_version TEXT,
                    status TEXT NOT NULL,
                    request_json TEXT,
                    result_json TEXT,
                    control_json TEXT,
                    artifact_run_id TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    last_heartbeat_at TEXT
                )
                """,
            )
            if self._backend == "postgres":
                self._execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS training_job_events (
                        event_id BIGSERIAL PRIMARY KEY,
                        job_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        ts TEXT NOT NULL,
                        payload_json TEXT
                    )
                    """,
                )
            else:
                self._execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS training_job_events (
                        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        job_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        ts TEXT NOT NULL,
                        payload_json TEXT
                    )
                    """,
                )
            self._execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS training_domains (
                    domain_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    owner_scope TEXT NOT NULL,
                    owner_user_id TEXT NOT NULL,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """,
            )
            self._execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS training_lineages (
                    lineage_id TEXT PRIMARY KEY,
                    domain_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    owner_user_id TEXT NOT NULL,
                    model_key TEXT NOT NULL,
                    parent_lineage_id TEXT,
                    active_version_id TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """,
            )
            self._execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS training_model_versions (
                    version_id TEXT PRIMARY KEY,
                    lineage_id TEXT NOT NULL,
                    source_job_id TEXT,
                    artifact_run_id TEXT,
                    status TEXT NOT NULL,
                    metrics_json TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """,
            )
            self._execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS training_update_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    lineage_id TEXT NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    trigger_snapshot_json TEXT,
                    dataset_snapshot_json TEXT,
                    config_json TEXT,
                    status TEXT NOT NULL,
                    idempotency_key TEXT,
                    approved_by TEXT,
                    rejected_by TEXT,
                    linked_job_id TEXT,
                    candidate_version_id TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    approved_at TEXT,
                    rejected_at TEXT,
                    started_at TEXT,
                    finished_at TEXT
                )
                """,
            )
            self._execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS training_replay_items (
                    replay_item_id TEXT PRIMARY KEY,
                    lineage_id TEXT NOT NULL,
                    file_id TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    weight REAL NOT NULL,
                    class_tag TEXT,
                    pinned INTEGER NOT NULL,
                    last_seen_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(lineage_id, file_id, sample_id)
                )
                """,
            )
            self._execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS training_merge_requests (
                    merge_id TEXT PRIMARY KEY,
                    source_lineage_id TEXT NOT NULL,
                    target_lineage_id TEXT NOT NULL,
                    candidate_version_id TEXT NOT NULL,
                    requested_by TEXT NOT NULL,
                    status TEXT NOT NULL,
                    decision_by TEXT,
                    notes TEXT,
                    evaluation_json TEXT,
                    linked_proposal_id TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    decided_at TEXT,
                    executed_at TEXT
                )
                """,
            )
            self._execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS training_scheduler_leases (
                    lease_name TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """,
            )

            self._execute(conn, "CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id)")
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_idempotency_run_id ON idempotency_keys(run_id)",
            )
            self._execute(
                conn, "CREATE INDEX IF NOT EXISTS idx_uploads_created_at ON uploads(created_at)"
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_uploads_user_created ON uploads(user_id, created_at)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_uploads_user_deleted ON uploads(user_id, deleted_at)",
            )
            self._execute(
                conn, "CREATE INDEX IF NOT EXISTS idx_runs_conversation ON runs(conversation_id)"
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_runs_user_updated ON runs(user_id, updated_at)",
            )
            self._execute(
                conn, "CREATE INDEX IF NOT EXISTS idx_run_metadata_user ON run_metadata(user_id)"
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_conversations_user_updated "
                "ON conversations(user_id, updated_at)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_conversation_messages_user_created "
                "ON conversation_messages(user_id, created_at_ms)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_conversation_context_memory_user_updated "
                "ON conversation_context_memory(user_id, updated_at)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_upload_sessions_user_fingerprint "
                "ON upload_sessions(user_id, fingerprint, updated_at)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_threads_user_updated "
                "ON threads(user_id, updated_at)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_thread_messages_thread_index "
                "ON thread_messages(thread_id, message_index)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_tasks_queue_status_next "
                "ON tasks(queue_name, status, next_attempt_at)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_tasks_run_id ON tasks(run_id)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_task_attempts_task_id ON task_attempts(task_id, attempt)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_task_heartbeats_task_id "
                "ON task_heartbeats(task_id, heartbeat_at)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_leases_owner_expires "
                "ON leases(owner_id, expires_at)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_resource_computations_sha_lookup "
                "ON resource_computations(user_id, tool_name, file_sha256, run_updated_at)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_resource_computations_name_lookup "
                "ON resource_computations(user_id, tool_name, file_name_norm, run_updated_at)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_resource_computations_run_id "
                "ON resource_computations(run_id)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_resource_computations_user_run "
                "ON resource_computations(user_id, run_id, run_updated_at)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_training_managed_sources_user_updated "
                "ON training_managed_sources(user_id, source_type, updated_at)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_training_datasets_user_updated "
                "ON training_datasets(user_id, updated_at)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_training_dataset_items_dataset "
                "ON training_dataset_items(dataset_id, user_id, split, role, sample_id)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_training_jobs_user_status_updated "
                "ON training_jobs(user_id, status, updated_at)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_training_jobs_type_status_updated "
                "ON training_jobs(job_type, status, updated_at)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_training_jobs_model_version "
                "ON training_jobs(model_key, model_version, updated_at)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_training_job_events_job_id_event "
                "ON training_job_events(job_id, event_id)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_training_domains_owner_updated "
                "ON training_domains(owner_user_id, updated_at)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_training_lineages_domain_scope "
                "ON training_lineages(domain_id, scope, updated_at)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_training_lineages_owner_model "
                "ON training_lineages(owner_user_id, model_key, updated_at)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_training_model_versions_lineage_status "
                "ON training_model_versions(lineage_id, status, updated_at)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_training_model_versions_source_job "
                "ON training_model_versions(source_job_id, updated_at)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_training_update_proposals_lineage_status "
                "ON training_update_proposals(lineage_id, status, updated_at)",
            )
            self._execute(
                conn,
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_training_update_proposals_idempotency "
                "ON training_update_proposals(idempotency_key)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_training_replay_items_lineage "
                "ON training_replay_items(lineage_id, pinned, updated_at)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_training_merge_requests_target_status "
                "ON training_merge_requests(target_lineage_id, status, updated_at)",
            )
            self._execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_training_scheduler_leases_updated "
                "ON training_scheduler_leases(updated_at)",
            )
            self._init_conversation_search_indexes(conn)
            conn.commit()

    def create_run(self, run: WorkflowRun) -> WorkflowRun:
        """Persist a new workflow run row and return the stored run object.
        
        Parameters
        ----------
        run : WorkflowRun
            Input argument.
        
        Returns
        -------
        WorkflowRun
            Computed result.
        """
        plan_json = json.dumps(_plan_to_dict(run.plan)) if run.plan else None
        checkpoint_state_json = (
            json.dumps(run.checkpoint_state, default=str) if isinstance(run.checkpoint_state, dict) else None
        )
        budget_state_json = (
            json.dumps(run.budget_state, default=str) if isinstance(run.budget_state, dict) else None
        )
        with self._lock, self._conn() as conn:
            self._execute(
                conn,
                "INSERT INTO runs("
                "run_id, goal, status, created_at, updated_at, plan_json, error, "
                "workflow_kind, mode, parent_run_id, planner_version, agent_role, "
                "checkpoint_state_json, budget_state_json, trace_group_id, workflow_profile, pedagogy_level"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run.run_id,
                    run.goal,
                    run.status.value,
                    run.created_at.isoformat(),
                    run.updated_at.isoformat(),
                    plan_json,
                    run.error,
                    run.workflow_kind,
                    run.mode,
                    run.parent_run_id,
                    run.planner_version,
                    run.agent_role,
                    checkpoint_state_json,
                    budget_state_json,
                    run.trace_group_id,
                    run.workflow_profile,
                    run.pedagogy_level,
                ),
            )
            conn.commit()
        return run

    def get_run(self, run_id: str) -> WorkflowRun | None:
        """Fetch a run by ID, including deserialized plan payload when present.
        
        Parameters
        ----------
        run_id : str
            Workflow run identifier.
        
        Returns
        -------
        WorkflowRun | None
            Computed result.
        """
        with self._lock, self._conn() as conn:
            row = self._execute(
                conn,
                "SELECT run_id, goal, status, created_at, updated_at, plan_json, error, "
                "workflow_kind, mode, parent_run_id, planner_version, agent_role, "
                "checkpoint_state_json, budget_state_json, trace_group_id, workflow_profile, pedagogy_level "
                "FROM runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
        if not row:
            return None
        plan = _plan_from_json(row[5])
        return WorkflowRun(
            run_id=row[0],
            goal=row[1],
            status=RunStatus(row[2]),
            created_at=datetime.fromisoformat(row[3]),
            updated_at=datetime.fromisoformat(row[4]),
            plan=plan,
            error=row[6],
            workflow_kind=str(row[7] or "workflow_plan"),
            mode=str(row[8] or "durable"),
            parent_run_id=row[9],
            planner_version=row[10],
            agent_role=row[11],
            checkpoint_state=(json.loads(row[12]) if row[12] else None),
            budget_state=(json.loads(row[13]) if row[13] else None),
            trace_group_id=row[14],
            workflow_profile=row[15],
            pedagogy_level=row[16],
        )

    def list_runs_for_user(self, *, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """List recent runs for one user, ordered by most recently updated.
        
        Parameters
        ----------
        user_id : str
            User identifier.
        limit : int, optional
            Maximum number of records to return.
        
        Returns
        -------
        list[dict[str, Any]]
            Computed result.
        """
        with self._lock, self._conn() as conn:
            rows = self._execute(
                conn,
                """
                SELECT run_id, goal, status, created_at, updated_at, error, conversation_id, user_id,
                       workflow_kind, mode, workflow_profile
                FROM runs
                WHERE user_id=?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (user_id, int(limit)),
            ).fetchall()
        return [
            {
                "run_id": row[0],
                "goal": row[1],
                "status": row[2],
                "created_at": row[3],
                "updated_at": row[4],
                "error": row[5],
                "conversation_id": row[6],
                "user_id": row[7],
                "workflow_kind": row[8],
                "mode": row[9],
                "workflow_profile": row[10],
            }
            for row in rows
        ]

    def update_status(self, run_id: str, status: RunStatus, error: str | None = None) -> None:
        """Update run lifecycle status and optional terminal error text.
        
        Parameters
        ----------
        run_id : str
            Workflow run identifier.
        status : RunStatus
            Status filter or update value.
        error : str | None, optional
            Error message text.
        
        Returns
        -------
        None
            No return value.
        """
        now = datetime.utcnow().isoformat()
        with self._lock, self._conn() as conn:
            self._execute(
                conn,
                "UPDATE runs SET status=?, updated_at=?, error=? WHERE run_id=?",
                (status.value, now, error, run_id),
            )
            conn.commit()

    def set_run_metadata(
        self,
        run_id: str,
        user_id: str | None,
        conversation_id: str | None,
        *,
        workflow_kind: str | None = None,
        mode: str | None = None,
        parent_run_id: str | None = None,
        planner_version: str | None = None,
        agent_role: str | None = None,
        checkpoint_state: dict[str, Any] | None = None,
        budget_state: dict[str, Any] | None = None,
        trace_group_id: str | None = None,
        workflow_profile: str | None = None,
        pedagogy_level: str | None = None,
    ) -> None:
        """Upsert run-to-user/conversation metadata and mirror it into `runs`.
        
        Parameters
        ----------
        run_id : str
            Workflow run identifier.
        user_id : str | None
            User identifier.
        conversation_id : str | None
            Conversation identifier.
        
        Returns
        -------
        None
            No return value.
        """
        now = datetime.utcnow().isoformat()
        checkpoint_state_json = (
            json.dumps(checkpoint_state, default=str) if isinstance(checkpoint_state, dict) else None
        )
        budget_state_json = (
            json.dumps(budget_state, default=str) if isinstance(budget_state, dict) else None
        )
        with self._lock, self._conn() as conn:
            self._execute(
                conn,
                """
                INSERT INTO run_metadata(run_id, user_id, conversation_id, created_at)
                VALUES(?,?,?,?)
                ON CONFLICT(run_id) DO UPDATE SET
                    user_id=COALESCE(excluded.user_id, run_metadata.user_id),
                    conversation_id=COALESCE(excluded.conversation_id, run_metadata.conversation_id)
                """,
                (run_id, user_id, conversation_id, now),
            )
            self._execute(
                conn,
                """
                UPDATE runs
                SET user_id=COALESCE(?, user_id),
                    conversation_id=COALESCE(?, conversation_id),
                    workflow_kind=COALESCE(?, workflow_kind),
                    mode=COALESCE(?, mode),
                    parent_run_id=COALESCE(?, parent_run_id),
                    planner_version=COALESCE(?, planner_version),
                    agent_role=COALESCE(?, agent_role),
                    checkpoint_state_json=COALESCE(?, checkpoint_state_json),
                    budget_state_json=COALESCE(?, budget_state_json),
                    trace_group_id=COALESCE(?, trace_group_id),
                    workflow_profile=COALESCE(?, workflow_profile),
                    pedagogy_level=COALESCE(?, pedagogy_level)
                WHERE run_id=?
                """,
                (
                    user_id,
                    conversation_id,
                    workflow_kind,
                    mode,
                    parent_run_id,
                    planner_version,
                    agent_role,
                    checkpoint_state_json,
                    budget_state_json,
                    trace_group_id,
                    workflow_profile,
                    pedagogy_level,
                    run_id,
                ),
            )
            conn.commit()

    def get_run_metadata(self, run_id: str) -> dict[str, Any] | None:
        """Return persisted metadata linkage for a run, if available.
        
        Parameters
        ----------
        run_id : str
            Workflow run identifier.
        
        Returns
        -------
        dict[str, Any] | None
            Result payload.
        """
        with self._lock, self._conn() as conn:
            row = self._execute(
                conn,
                "SELECT run_id, user_id, conversation_id, created_at "
                "FROM run_metadata WHERE run_id=?",
                (run_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "run_id": row[0],
            "user_id": row[1],
            "conversation_id": row[2],
            "created_at": row[3],
        }

    def upsert_thread(
        self,
        *,
        thread_id: str,
        user_id: str | None,
        conversation_id: str | None = None,
        title: str | None = None,
        status: str = "active",
        latest_run_id: str | None = None,
        checkpoint_id: str | None = None,
        summary: str | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
    ) -> dict[str, Any]:
        created = str(created_at or datetime.utcnow().isoformat())
        updated = str(updated_at or created)
        metadata_json = json.dumps(metadata or {}, default=str)
        with self._lock, self._conn() as conn:
            self._execute(
                conn,
                """
                INSERT INTO threads(
                    thread_id,
                    user_id,
                    conversation_id,
                    title,
                    status,
                    created_at,
                    updated_at,
                    latest_run_id,
                    checkpoint_id,
                    summary,
                    metadata_json
                )
                VALUES(?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    user_id=COALESCE(excluded.user_id, threads.user_id),
                    conversation_id=COALESCE(excluded.conversation_id, threads.conversation_id),
                    title=COALESCE(excluded.title, threads.title),
                    status=COALESCE(excluded.status, threads.status),
                    updated_at=excluded.updated_at,
                    latest_run_id=COALESCE(excluded.latest_run_id, threads.latest_run_id),
                    checkpoint_id=COALESCE(excluded.checkpoint_id, threads.checkpoint_id),
                    summary=COALESCE(excluded.summary, threads.summary),
                    metadata_json=COALESCE(excluded.metadata_json, threads.metadata_json),
                    created_at=COALESCE(threads.created_at, excluded.created_at)
                """,
                (
                    str(thread_id or "").strip(),
                    str(user_id or "").strip() or None,
                    str(conversation_id or "").strip() or None,
                    str(title or "").strip() or None,
                    str(status or "active").strip() or "active",
                    created,
                    updated,
                    str(latest_run_id or "").strip() or None,
                    str(checkpoint_id or "").strip() or None,
                    str(summary or "").strip() or None,
                    metadata_json,
                ),
            )
            conn.commit()
        payload = self.get_thread(thread_id=thread_id, user_id=user_id)
        if payload is None:
            raise RuntimeError(f"Failed to persist thread: {thread_id}")
        return payload

    def get_thread(self, *, thread_id: str, user_id: str | None = None) -> dict[str, Any] | None:
        with self._lock, self._conn() as conn:
            if user_id:
                row = self._execute(
                    conn,
                    """
                    SELECT
                        thread_id,
                        user_id,
                        conversation_id,
                        title,
                        status,
                        created_at,
                        updated_at,
                        latest_run_id,
                        checkpoint_id,
                        summary,
                        metadata_json
                    FROM threads
                    WHERE thread_id=? AND (user_id=? OR user_id IS NULL)
                    """,
                    (str(thread_id or "").strip(), str(user_id or "").strip()),
                ).fetchone()
            else:
                row = self._execute(
                    conn,
                    """
                    SELECT
                        thread_id,
                        user_id,
                        conversation_id,
                        title,
                        status,
                        created_at,
                        updated_at,
                        latest_run_id,
                        checkpoint_id,
                        summary,
                        metadata_json
                    FROM threads
                    WHERE thread_id=?
                    """,
                    (str(thread_id or "").strip(),),
                ).fetchone()
        if not row:
            return None
        metadata = self._decode_json_payload(row[10])
        return {
            "thread_id": row[0],
            "user_id": row[1],
            "conversation_id": row[2],
            "title": row[3],
            "status": row[4],
            "created_at": row[5],
            "updated_at": row[6],
            "latest_run_id": row[7],
            "checkpoint_id": row[8],
            "summary": row[9],
            "metadata": metadata if isinstance(metadata, dict) else {},
        }

    def list_threads(
        self,
        *,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        params: tuple[Any, ...]
        query = (
            "SELECT thread_id, user_id, conversation_id, title, status, created_at, updated_at, "
            "latest_run_id, checkpoint_id, summary, metadata_json "
            "FROM threads "
        )
        if user_id:
            query += "WHERE user_id=? ORDER BY updated_at DESC LIMIT ?"
            params = (str(user_id or "").strip(), max(1, int(limit)))
        else:
            query += "ORDER BY updated_at DESC LIMIT ?"
            params = (max(1, int(limit)),)
        with self._lock, self._conn() as conn:
            rows = self._execute(conn, query, params).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            metadata = self._decode_json_payload(row[10])
            output.append(
                {
                    "thread_id": row[0],
                    "user_id": row[1],
                    "conversation_id": row[2],
                    "title": row[3],
                    "status": row[4],
                    "created_at": row[5],
                    "updated_at": row[6],
                    "latest_run_id": row[7],
                    "checkpoint_id": row[8],
                    "summary": row[9],
                    "metadata": metadata if isinstance(metadata, dict) else {},
                }
            )
        return output

    def replace_thread_messages(
        self,
        *,
        thread_id: str,
        user_id: str,
        messages: list[dict[str, Any]],
    ) -> None:
        thread_token = str(thread_id or "").strip()
        user_token = str(user_id or "").strip()
        now = datetime.utcnow().isoformat()
        with self._lock, self._conn() as conn:
            self._execute(
                conn,
                "DELETE FROM thread_messages WHERE thread_id=? AND user_id=?",
                (thread_token, user_token),
            )
            for index, message in enumerate(messages):
                payload = dict(message or {})
                message_id = str(payload.get("id") or f"{thread_token}:{index}").strip()
                role = str(payload.get("role") or "assistant").strip() or "assistant"
                content = str(payload.get("content") or "")
                created_at = str(payload.get("created_at") or payload.get("createdAt") or now)
                run_id = str(payload.get("run_id") or payload.get("runId") or "").strip() or None
                metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
                self._execute(
                    conn,
                    """
                    INSERT INTO thread_messages(
                        thread_id,
                        user_id,
                        message_id,
                        message_index,
                        role,
                        content,
                        created_at,
                        run_id,
                        metadata_json
                    )
                    VALUES(?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        thread_token,
                        user_token,
                        message_id,
                        int(index),
                        role,
                        content,
                        created_at,
                        run_id,
                        json.dumps(metadata, default=str),
                    ),
                )
            conn.commit()

    def list_thread_messages(
        self,
        *,
        thread_id: str,
        user_id: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        params: tuple[Any, ...]
        query = (
            "SELECT thread_id, user_id, message_id, message_index, role, content, created_at, run_id, "
            "metadata_json FROM thread_messages WHERE thread_id=?"
        )
        params_list: list[Any] = [str(thread_id or "").strip()]
        if user_id:
            query += " AND user_id=?"
            params_list.append(str(user_id or "").strip())
        query += " ORDER BY message_index ASC LIMIT ?"
        params_list.append(max(1, int(limit)))
        params = tuple(params_list)
        with self._lock, self._conn() as conn:
            rows = self._execute(conn, query, params).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            metadata = self._decode_json_payload(row[8])
            output.append(
                {
                    "message_id": row[2],
                    "thread_id": row[0],
                    "role": row[4],
                    "content": row[5],
                    "created_at": row[6],
                    "metadata": metadata if isinstance(metadata, dict) else {},
                    "run_id": row[7],
                }
            )
        return output

    def upsert_thread_summary(
        self,
        *,
        thread_id: str,
        summary: str,
        updated_at: str | None = None,
    ) -> None:
        now = str(updated_at or datetime.utcnow().isoformat())
        with self._lock, self._conn() as conn:
            self._execute(
                conn,
                """
                INSERT INTO thread_summaries(thread_id, summary, updated_at)
                VALUES(?,?,?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    summary=excluded.summary,
                    updated_at=excluded.updated_at
                """,
                (str(thread_id or "").strip(), str(summary or ""), now),
            )
            conn.commit()

    def append_event(
        self,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
        level: str = "info",
    ) -> None:
        """Append a durable run event for streaming, audit, and replay.
        
        Parameters
        ----------
        run_id : str
            Workflow run identifier.
        event_type : str
            Input argument.
        payload : dict[str, Any]
            Input payload for the operation.
        level : str, optional
            Input argument.
        
        Returns
        -------
        None
            No return value.
        """
        now = datetime.utcnow().isoformat()
        payload_json = json.dumps(payload, default=str)
        with self._lock, self._conn() as conn:
            self._execute(
                conn,
                "INSERT INTO events(run_id, ts, level, event_type, payload_json) VALUES(?,?,?,?,?)",
                (run_id, now, level, event_type, payload_json),
            )
            conn.commit()

    def list_events(self, run_id: str, limit: int = 200) -> list[dict[str, Any]]:
        """Return run events in chronological order up to `limit`.
        
        Parameters
        ----------
        run_id : str
            Workflow run identifier.
        limit : int, optional
            Maximum number of records to return.
        
        Returns
        -------
        list[dict[str, Any]]
            Computed result.
        """
        with self._lock, self._conn() as conn:
            rows = self._execute(
                conn,
                "SELECT ts, level, event_type, payload_json FROM events "
                "WHERE run_id=? ORDER BY event_id DESC LIMIT ?",
                (run_id, int(limit)),
            ).fetchall()
        events: list[dict[str, Any]] = []
        for ts, level, event_type, payload_json in rows[::-1]:
            events.append(
                {
                    "ts": ts,
                    "level": level,
                    "event_type": event_type,
                    "payload": json.loads(payload_json),
                }
            )
        return events

    def get_latest_event(self, run_id: str, event_type: str) -> dict[str, Any] | None:
        """Fetch the latest event of a given type for a run.
        
        Parameters
        ----------
        run_id : str
            Workflow run identifier.
        event_type : str
            Input argument.
        
        Returns
        -------
        dict[str, Any] | None
            Result payload.
        """
        with self._lock, self._conn() as conn:
            row = self._execute(
                conn,
                "SELECT ts, level, event_type, payload_json FROM events "
                "WHERE run_id=? AND event_type=? ORDER BY event_id DESC LIMIT 1",
                (run_id, event_type),
            ).fetchone()
        if not row:
            return None
        return {
            "ts": row[0],
            "level": row[1],
            "event_type": row[2],
            "payload": json.loads(row[3]),
        }

    def get_idempotency(self, endpoint: str, idempotency_key: str) -> dict[str, Any] | None:
        """Lookup previously stored idempotency mapping for an endpoint/request key.
        
        Parameters
        ----------
        endpoint : str
            Input argument.
        idempotency_key : str
            Input argument.
        
        Returns
        -------
        dict[str, Any] | None
            Result payload.
        """
        with self._lock, self._conn() as conn:
            row = self._execute(
                conn,
                """
                SELECT endpoint, idempotency_key, request_hash, run_id, created_at
                FROM idempotency_keys
                WHERE endpoint=? AND idempotency_key=?
                """,
                (endpoint, idempotency_key),
            ).fetchone()
        if not row:
            return None
        return {
            "endpoint": row[0],
            "idempotency_key": row[1],
            "request_hash": row[2],
            "run_id": row[3],
            "created_at": row[4],
        }

    def put_idempotency(
        self,
        endpoint: str,
        idempotency_key: str,
        request_hash: str,
        run_id: str,
    ) -> None:
        """Store an idempotency key record if it does not already exist.
        
        Parameters
        ----------
        endpoint : str
            Input argument.
        idempotency_key : str
            Input argument.
        request_hash : str
            Input argument.
        run_id : str
            Workflow run identifier.
        
        Returns
        -------
        None
            No return value.
        """
        now = datetime.utcnow().isoformat()
        with self._lock, self._conn() as conn:
            self._execute(
                conn,
                """
                INSERT INTO idempotency_keys(endpoint, idempotency_key, request_hash, run_id, created_at)
                VALUES(?,?,?,?,?)
                ON CONFLICT(endpoint, idempotency_key) DO NOTHING
                """,
                (endpoint, idempotency_key, request_hash, run_id, now),
            )
            conn.commit()

    def put_upload(
        self,
        *,
        file_id: str,
        original_name: str,
        stored_path: str,
        content_type: str | None,
        size_bytes: int,
        sha256: str,
        created_at: str | None = None,
        user_id: str | None = None,
        source_type: str | None = None,
        source_uri: str | None = None,
        client_view_url: str | None = None,
        image_service_url: str | None = None,
        resource_kind: str | None = None,
        thumbnail_path: str | None = None,
        metadata: dict[str, Any] | None = None,
        deleted_at: str | None = None,
        staging_path: str | None = None,
        cache_path: str | None = None,
        canonical_resource_uniq: str | None = None,
        canonical_resource_uri: str | None = None,
        sync_status: str | None = None,
        sync_error: str | None = None,
        sync_retry_count: int | None = None,
        sync_started_at: str | None = None,
        sync_completed_at: str | None = None,
        sync_run_id: str | None = None,
    ) -> None:
        """Upsert upload metadata, preserving provenance and soft-delete state.
        
        Parameters
        ----------
        file_id : str
            Upload file identifier.
        original_name : str
            Input argument.
        stored_path : str
            Input argument.
        content_type : str | None
            Input argument.
        size_bytes : int
            Input argument.
        sha256 : str
            Input argument.
        created_at : str | None, optional
            Timestamp value.
        user_id : str | None, optional
            User identifier.
        source_type : str | None, optional
            Input argument.
        source_uri : str | None, optional
            Input argument.
        client_view_url : str | None, optional
            Input argument.
        image_service_url : str | None, optional
            Input argument.
        resource_kind : str | None, optional
            Input argument.
        thumbnail_path : str | None, optional
            Input argument.
        metadata : dict[str, Any] | None, optional
            Input argument.
        deleted_at : str | None, optional
            Timestamp value.
        
        Returns
        -------
        None
            No return value.
        
        Notes
        -----
        Keep `sha256` and `size_bytes` accurate for downstream reproducibility checks.
        Use `source_type/source_uri` for imported assets so UI links remain discoverable.
        """
        created = created_at or datetime.utcnow().isoformat()
        updated = datetime.utcnow().isoformat()
        metadata_json = json.dumps(metadata, default=str) if metadata else None
        with self._lock, self._conn() as conn:
            self._execute(
                conn,
                """
                INSERT INTO uploads(
                    file_id,
                    original_name,
                    stored_path,
                    content_type,
                    size_bytes,
                    sha256,
                    created_at,
                    user_id,
                    source_type,
                    source_uri,
                    client_view_url,
                    image_service_url,
                    resource_kind,
                    thumbnail_path,
                    metadata_json,
                    updated_at,
                    deleted_at,
                    staging_path,
                    cache_path,
                    canonical_resource_uniq,
                    canonical_resource_uri,
                    sync_status,
                    sync_error,
                    sync_retry_count,
                    sync_started_at,
                    sync_completed_at,
                    sync_run_id
                )
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(file_id) DO UPDATE SET
                    original_name=excluded.original_name,
                    stored_path=excluded.stored_path,
                    content_type=excluded.content_type,
                    size_bytes=excluded.size_bytes,
                    sha256=excluded.sha256,
                    created_at=excluded.created_at,
                    user_id=excluded.user_id,
                    source_type=excluded.source_type,
                    source_uri=excluded.source_uri,
                    client_view_url=excluded.client_view_url,
                    image_service_url=excluded.image_service_url,
                    resource_kind=excluded.resource_kind,
                    thumbnail_path=excluded.thumbnail_path,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at,
                    deleted_at=excluded.deleted_at,
                    staging_path=excluded.staging_path,
                    cache_path=excluded.cache_path,
                    canonical_resource_uniq=excluded.canonical_resource_uniq,
                    canonical_resource_uri=excluded.canonical_resource_uri,
                    sync_status=excluded.sync_status,
                    sync_error=excluded.sync_error,
                    sync_retry_count=excluded.sync_retry_count,
                    sync_started_at=excluded.sync_started_at,
                    sync_completed_at=excluded.sync_completed_at,
                    sync_run_id=excluded.sync_run_id
                """,
                (
                    file_id,
                    original_name,
                    stored_path,
                    content_type,
                    int(size_bytes),
                    sha256,
                    created,
                    user_id,
                    source_type,
                    source_uri,
                    client_view_url,
                    image_service_url,
                    resource_kind,
                    thumbnail_path,
                    metadata_json,
                    updated,
                    deleted_at,
                    staging_path,
                    cache_path,
                    canonical_resource_uniq,
                    canonical_resource_uri,
                    sync_status,
                    sync_error,
                    sync_retry_count,
                    sync_started_at,
                    sync_completed_at,
                    sync_run_id,
                ),
            )
            conn.commit()

    def get_upload(
        self,
        file_id: str,
        *,
        user_id: str | None = None,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        """Fetch one upload record with optional user scoping and soft-delete visibility.
        
        Parameters
        ----------
        file_id : str
            Upload file identifier.
        user_id : str | None, optional
            User identifier.
        include_deleted : bool, optional
            Whether to include optional values in the result.
        
        Returns
        -------
        dict[str, Any] | None
            Result payload.
        """
        identifier = str(file_id or "").strip()
        clauses = ["(file_id=? OR canonical_resource_uniq=? OR canonical_resource_uri=? OR source_uri=?)"]
        params: list[Any] = [identifier, identifier, identifier, identifier]
        if user_id is not None:
            clauses.append("(user_id=? OR user_id IS NULL)")
            params.append(user_id)
        if not include_deleted:
            clauses.append("deleted_at IS NULL")
        with self._lock, self._conn() as conn:
            row = self._execute(
                conn,
                """
                SELECT
                    file_id,
                    original_name,
                    stored_path,
                    content_type,
                    size_bytes,
                    sha256,
                    created_at,
                    user_id,
                    source_type,
                    source_uri,
                    client_view_url,
                    image_service_url,
                    resource_kind,
                    thumbnail_path,
                    metadata_json,
                    updated_at,
                    deleted_at,
                    staging_path,
                    cache_path,
                    canonical_resource_uniq,
                    canonical_resource_uri,
                    sync_status,
                    sync_error,
                    sync_retry_count,
                    sync_started_at,
                    sync_completed_at,
                    sync_run_id
                FROM uploads
                WHERE {where_clause}
                """.replace("{where_clause}", " AND ".join(clauses)),
                tuple(params),
            ).fetchone()
        if not row:
            return None
        metadata_raw = row[14]
        try:
            metadata_value = json.loads(metadata_raw) if metadata_raw else {}
        except Exception:
            metadata_value = {}
        return {
            "file_id": row[0],
            "original_name": row[1],
            "stored_path": row[2],
            "content_type": row[3],
            "size_bytes": int(row[4]),
            "sha256": row[5],
            "created_at": row[6],
            "user_id": row[7],
            "source_type": row[8],
            "source_uri": row[9],
            "client_view_url": row[10],
            "image_service_url": row[11],
            "resource_kind": row[12],
            "thumbnail_path": row[13],
            "metadata": metadata_value,
            "updated_at": row[15],
            "deleted_at": row[16],
            "staging_path": row[17],
            "cache_path": row[18],
            "canonical_resource_uniq": row[19],
            "canonical_resource_uri": row[20],
            "sync_status": row[21],
            "sync_error": row[22],
            "sync_retry_count": int(row[23] or 0),
            "sync_started_at": row[24],
            "sync_completed_at": row[25],
            "sync_run_id": row[26],
        }

    def find_upload_by_sha256(
        self,
        *,
        sha256: str,
        user_id: str | None = None,
        include_deleted: bool = False,
        require_canonical: bool = False,
    ) -> dict[str, Any] | None:
        """Fetch the most recent upload row that matches an exact SHA256 digest.

        Parameters
        ----------
        sha256 : str
            File digest to match.
        user_id : str | None, optional
            Optional user scoping. When provided, prefer rows owned by this user.
        include_deleted : bool, optional
            Whether soft-deleted rows are eligible.
        require_canonical : bool, optional
            When true, only return rows that already resolved to a canonical BisQue resource.

        Returns
        -------
        dict[str, Any] | None
            Upload row or None.
        """
        digest = str(sha256 or "").strip().lower()
        if not digest:
            return None

        clauses = ["LOWER(COALESCE(sha256, ''))=?"]
        params: list[Any] = [digest]
        if user_id is not None:
            clauses.append("(user_id=? OR user_id IS NULL)")
            params.append(user_id)
        if not include_deleted:
            clauses.append("deleted_at IS NULL")
        if require_canonical:
            clauses.append("canonical_resource_uri IS NOT NULL")

        with self._lock, self._conn() as conn:
            row = self._execute(
                conn,
                """
                SELECT
                    file_id,
                    original_name,
                    stored_path,
                    content_type,
                    size_bytes,
                    sha256,
                    created_at,
                    user_id,
                    source_type,
                    source_uri,
                    client_view_url,
                    image_service_url,
                    resource_kind,
                    thumbnail_path,
                    metadata_json,
                    updated_at,
                    deleted_at,
                    staging_path,
                    cache_path,
                    canonical_resource_uniq,
                    canonical_resource_uri,
                    sync_status,
                    sync_error,
                    sync_retry_count,
                    sync_started_at,
                    sync_completed_at,
                    sync_run_id
                FROM uploads
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT 1
                """.replace("{where_clause}", " AND ".join(clauses)),
                tuple(params),
            ).fetchone()
        if not row:
            return None
        metadata_raw = row[14]
        try:
            metadata_value = json.loads(metadata_raw) if metadata_raw else {}
        except Exception:
            metadata_value = {}
        return {
            "file_id": row[0],
            "original_name": row[1],
            "stored_path": row[2],
            "content_type": row[3],
            "size_bytes": int(row[4]),
            "sha256": row[5],
            "created_at": row[6],
            "user_id": row[7],
            "source_type": row[8],
            "source_uri": row[9],
            "client_view_url": row[10],
            "image_service_url": row[11],
            "resource_kind": row[12],
            "thumbnail_path": row[13],
            "metadata": metadata_value,
            "updated_at": row[15],
            "deleted_at": row[16],
            "staging_path": row[17],
            "cache_path": row[18],
            "canonical_resource_uniq": row[19],
            "canonical_resource_uri": row[20],
            "sync_status": row[21],
            "sync_error": row[22],
            "sync_retry_count": int(row[23] or 0),
            "sync_started_at": row[24],
            "sync_completed_at": row[25],
            "sync_run_id": row[26],
        }

    def list_uploads(
        self,
        *,
        user_id: str,
        limit: int = 200,
        offset: int = 0,
        query: str | None = None,
        kind: str | None = None,
        source_type: str | None = None,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        """List uploads for a user with optional text/source/kind filters.
        
        Parameters
        ----------
        user_id : str
            User identifier.
        limit : int, optional
            Maximum number of records to return.
        offset : int, optional
            Pagination offset.
        query : str | None, optional
            Free-text query string.
        kind : str | None, optional
            Input argument.
        source_type : str | None, optional
            Input argument.
        include_deleted : bool, optional
            Whether to include optional values in the result.
        
        Returns
        -------
        list[dict[str, Any]]
            Computed result.
        """
        clauses = ["(user_id=? OR user_id IS NULL)"]
        params: list[Any] = [user_id]
        if not include_deleted:
            clauses.append("deleted_at IS NULL")
        query_text = str(query or "").strip()
        if query_text:
            like = f"%{query_text}%"
            clauses.append(
                "(LOWER(original_name) LIKE LOWER(?) "
                "OR LOWER(COALESCE(source_uri, '')) LIKE LOWER(?) "
                "OR LOWER(COALESCE(client_view_url, '')) LIKE LOWER(?) "
                "OR LOWER(COALESCE(canonical_resource_uniq, '')) LIKE LOWER(?) "
                "OR LOWER(COALESCE(canonical_resource_uri, '')) LIKE LOWER(?))"
            )
            params.extend([like, like, like, like, like])
        kind_text = str(kind or "").strip().lower()
        if kind_text:
            clauses.append("LOWER(COALESCE(resource_kind, 'file'))=?")
            params.append(kind_text)
        source_text = str(source_type or "").strip().lower()
        if source_text:
            clauses.append("LOWER(COALESCE(source_type, 'upload'))=?")
            params.append(source_text)

        params.extend([int(limit), int(offset)])
        with self._lock, self._conn() as conn:
            rows = self._execute(
                conn,
                """
                SELECT
                    file_id,
                    original_name,
                    stored_path,
                    content_type,
                    size_bytes,
                    sha256,
                    created_at,
                    user_id,
                    source_type,
                    source_uri,
                    client_view_url,
                    image_service_url,
                    resource_kind,
                    thumbnail_path,
                    metadata_json,
                    updated_at,
                    deleted_at,
                    staging_path,
                    cache_path,
                    canonical_resource_uniq,
                    canonical_resource_uri,
                    sync_status,
                    sync_error,
                    sync_retry_count,
                    sync_started_at,
                    sync_completed_at,
                    sync_run_id
                FROM uploads
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ?
                OFFSET ?
                """.replace("{where_clause}", " AND ".join(clauses)),
                tuple(params),
            ).fetchall()

        output: list[dict[str, Any]] = []
        for row in rows:
            metadata_raw = row[14]
            try:
                metadata_value = json.loads(metadata_raw) if metadata_raw else {}
            except Exception:
                metadata_value = {}
            output.append(
                {
                    "file_id": row[0],
                    "original_name": row[1],
                    "stored_path": row[2],
                    "content_type": row[3],
                    "size_bytes": int(row[4]),
                    "sha256": row[5],
                    "created_at": row[6],
                    "user_id": row[7],
                    "source_type": row[8],
                    "source_uri": row[9],
                    "client_view_url": row[10],
                    "image_service_url": row[11],
                    "resource_kind": row[12],
                    "thumbnail_path": row[13],
                    "metadata": metadata_value,
                    "updated_at": row[15],
                    "deleted_at": row[16],
                    "staging_path": row[17],
                    "cache_path": row[18],
                    "canonical_resource_uniq": row[19],
                    "canonical_resource_uri": row[20],
                    "sync_status": row[21],
                    "sync_error": row[22],
                    "sync_retry_count": int(row[23] or 0),
                    "sync_started_at": row[24],
                    "sync_completed_at": row[25],
                    "sync_run_id": row[26],
                }
            )
        return output

    def update_upload_thumbnail(
        self,
        *,
        file_id: str,
        user_id: str,
        thumbnail_path: str | None,
    ) -> bool:
        """Update thumbnail path for a visible upload and return whether a row changed.
        
        Parameters
        ----------
        file_id : str
            Upload file identifier.
        user_id : str
            User identifier.
        thumbnail_path : str | None
            Input argument.
        
        Returns
        -------
        bool
            Computed result.
        """
        now = datetime.utcnow().isoformat()
        identifier = str(file_id or "").strip()
        with self._lock, self._conn() as conn:
            result = self._execute(
                conn,
                """
                UPDATE uploads
                SET thumbnail_path=?, updated_at=?
                WHERE (file_id=? OR canonical_resource_uniq=? OR canonical_resource_uri=?)
                  AND (user_id=? OR user_id IS NULL)
                """,
                (thumbnail_path, now, identifier, identifier, identifier, user_id),
            )
            conn.commit()
            rowcount = int(getattr(result, "rowcount", 0) or 0)
        return rowcount > 0

    def soft_delete_upload(
        self,
        *,
        file_id: str,
        user_id: str,
        deleted_at: str | None = None,
    ) -> bool:
        """Soft-delete an upload by setting `deleted_at` and `updated_at` timestamps.
        
        Parameters
        ----------
        file_id : str
            Upload file identifier.
        user_id : str
            User identifier.
        deleted_at : str | None, optional
            Timestamp value.
        
        Returns
        -------
        bool
            Computed result.
        """
        ts = str(deleted_at or datetime.utcnow().isoformat())
        identifier = str(file_id or "").strip()
        with self._lock, self._conn() as conn:
            result = self._execute(
                conn,
                """
                UPDATE uploads
                SET deleted_at=?, updated_at=?
                WHERE (file_id=? OR canonical_resource_uniq=? OR canonical_resource_uri=?)
                  AND (user_id=? OR user_id IS NULL)
                  AND deleted_at IS NULL
                """,
                (ts, ts, identifier, identifier, identifier, user_id),
            )
            conn.commit()
            rowcount = int(getattr(result, "rowcount", 0) or 0)
        return rowcount > 0

    def upsert_conversation(
        self,
        *,
        conversation_id: str,
        user_id: str,
        title: str,
        created_at: str,
        updated_at: str,
        state: dict[str, Any],
    ) -> None:
        """Create or update a conversation envelope with user-scoped conversation IDs.
        
        Parameters
        ----------
        conversation_id : str
            Conversation identifier.
        user_id : str
            User identifier.
        title : str
            Input argument.
        created_at : str
            Timestamp value.
        updated_at : str
            Timestamp value.
        state : dict[str, Any]
            Input argument.
        
        Returns
        -------
        None
            No return value.
        
        Notes
        -----
        Persist only sanitized serializable state payloads.
        Use scoped IDs to avoid cross-user collisions when IDs overlap.
        """
        scoped_conversation_id = self._scoped_conversation_id(conversation_id, user_id)
        legacy_conversation_id = str(conversation_id or "").strip()
        payload_json = json.dumps(state, default=str)
        with self._lock, self._conn() as conn:
            if legacy_conversation_id and legacy_conversation_id != scoped_conversation_id:
                self._execute(
                    conn,
                    "DELETE FROM conversations WHERE conversation_id=? AND user_id=?",
                    (legacy_conversation_id, user_id),
                )
            self._execute(
                conn,
                """
                INSERT INTO conversations(conversation_id, user_id, title, created_at, updated_at, state_json)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    user_id=excluded.user_id,
                    title=excluded.title,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    state_json=excluded.state_json
                """,
                (
                    scoped_conversation_id,
                    user_id,
                    title,
                    created_at,
                    updated_at,
                    payload_json,
                ),
            )
            conn.commit()

    def replace_conversation_messages(
        self,
        *,
        conversation_id: str,
        user_id: str,
        messages: list[dict[str, Any]],
    ) -> None:
        """Replace all stored messages for a conversation with a new ordered snapshot.
        
        Parameters
        ----------
        conversation_id : str
            Conversation identifier.
        user_id : str
            User identifier.
        messages : list[dict[str, Any]]
            Input argument.
        
        Returns
        -------
        None
            No return value.
        """
        lookup_ids = self._conversation_lookup_ids(conversation_id, user_id)
        scoped_conversation_id = lookup_ids[0]
        placeholders = ",".join(["?"] * len(lookup_ids))
        delete_params: tuple[Any, ...] = tuple([*lookup_ids, user_id])
        with self._lock, self._conn() as conn:
            self._execute(
                conn,
                f"DELETE FROM conversation_messages WHERE conversation_id IN ({placeholders}) AND user_id=?",
                delete_params,
            )
            for index, message in enumerate(messages):
                message_id = str(message.get("id") or f"{conversation_id}:{index}")
                role = str(message.get("role") or "assistant")
                content = str(message.get("content") or "")
                created_at_ms = message.get("createdAt")
                try:
                    created_at_ms_int = int(created_at_ms) if created_at_ms is not None else None
                except Exception:
                    created_at_ms_int = None
                run_id = str(message.get("runId") or "").strip() or None
                self._execute(
                    conn,
                    """
                    INSERT INTO conversation_messages(
                        conversation_id,
                        user_id,
                        message_id,
                        message_index,
                        role,
                        content,
                        created_at_ms,
                        run_id,
                        payload_json
                    )
                    VALUES(?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(conversation_id, message_id) DO UPDATE SET
                        user_id=excluded.user_id,
                        message_index=excluded.message_index,
                        role=excluded.role,
                        content=excluded.content,
                        created_at_ms=excluded.created_at_ms,
                        run_id=excluded.run_id,
                        payload_json=excluded.payload_json
                    """,
                    (
                        scoped_conversation_id,
                        user_id,
                        message_id,
                        int(index),
                        role,
                        content,
                        created_at_ms_int,
                        run_id,
                        json.dumps(message, default=str),
                    ),
                )
            conn.commit()

    def get_conversation(self, *, conversation_id: str, user_id: str) -> dict[str, Any] | None:
        """Fetch one conversation envelope for a user and return externalized ID form.
        
        Parameters
        ----------
        conversation_id : str
            Conversation identifier.
        user_id : str
            User identifier.
        
        Returns
        -------
        dict[str, Any] | None
            Result payload.
        """
        lookup_ids = self._conversation_lookup_ids(conversation_id, user_id)
        placeholders = ",".join(["?"] * len(lookup_ids))
        params: tuple[Any, ...] = tuple([*lookup_ids, user_id])
        with self._lock, self._conn() as conn:
            row = self._execute(
                conn,
                f"""
                SELECT conversation_id, user_id, title, created_at, updated_at, state_json
                FROM conversations
                WHERE conversation_id IN ({placeholders}) AND user_id=?
                ORDER BY updated_at DESC
                """,
                params,
            ).fetchone()
        if not row:
            return None
        external_conversation_id = self._external_conversation_id(
            stored_conversation_id=str(row[0]),
            user_id=user_id,
        )
        return {
            "conversation_id": external_conversation_id,
            "user_id": row[1],
            "title": row[2],
            "created_at": row[3],
            "updated_at": row[4],
            "state": json.loads(row[5]),
        }

    def count_conversations(self, *, user_id: str) -> int:
        """Count conversation envelopes for a user."""
        with self._lock, self._conn() as conn:
            row = self._execute(
                conn,
                "SELECT COUNT(*) FROM conversations WHERE user_id=?",
                (user_id,),
            ).fetchone()
        return int(row[0]) if row else 0

    def list_conversations(
        self,
        *,
        user_id: str,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List conversation envelopes for a user, newest first.
        
        Parameters
        ----------
        user_id : str
            User identifier.
        limit : int, optional
            Maximum number of records to return.
        offset : int, optional
            Number of records to skip before applying the page limit.
        
        Returns
        -------
        list[dict[str, Any]]
            Computed result.
        """
        with self._lock, self._conn() as conn:
            rows = self._execute(
                conn,
                """
                SELECT conversation_id, user_id, title, created_at, updated_at, state_json
                FROM conversations
                WHERE user_id=?
                ORDER BY updated_at DESC
                LIMIT ?
                OFFSET ?
                """,
                (user_id, int(limit), int(offset)),
            ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            output.append(
                {
                    "conversation_id": self._external_conversation_id(
                        stored_conversation_id=str(row[0]),
                        user_id=user_id,
                    ),
                    "user_id": row[1],
                    "title": row[2],
                    "created_at": row[3],
                    "updated_at": row[4],
                    "state": json.loads(row[5]),
                }
            )
        return output

    def delete_conversation(self, *, conversation_id: str, user_id: str) -> None:
        """Delete conversation envelope, messages, and context-memory rows for a user.
        
        Parameters
        ----------
        conversation_id : str
            Conversation identifier.
        user_id : str
            User identifier.
        
        Returns
        -------
        None
            No return value.
        """
        lookup_ids = self._conversation_lookup_ids(conversation_id, user_id)
        placeholders = ",".join(["?"] * len(lookup_ids))
        delete_params: tuple[Any, ...] = tuple([*lookup_ids, user_id])
        with self._lock, self._conn() as conn:
            self._execute(
                conn,
                f"DELETE FROM conversation_messages WHERE conversation_id IN ({placeholders}) AND user_id=?",
                delete_params,
            )
            self._execute(
                conn,
                f"DELETE FROM conversation_context_memory WHERE conversation_id IN ({placeholders}) AND user_id=?",
                delete_params,
            )
            self._execute(
                conn,
                f"DELETE FROM conversations WHERE conversation_id IN ({placeholders}) AND user_id=?",
                delete_params,
            )
            conn.commit()

    def get_conversation_context_memory(
        self,
        *,
        conversation_id: str,
        user_id: str,
    ) -> dict[str, Any] | None:
        """Return stored context-compaction memory for a conversation if present.
        
        Parameters
        ----------
        conversation_id : str
            Conversation identifier.
        user_id : str
            User identifier.
        
        Returns
        -------
        dict[str, Any] | None
            Result payload.
        """
        lookup_ids = self._conversation_lookup_ids(conversation_id, user_id)
        placeholders = ",".join(["?"] * len(lookup_ids))
        params: tuple[Any, ...] = tuple([*lookup_ids, user_id])
        with self._lock, self._conn() as conn:
            row = self._execute(
                conn,
                f"""
                SELECT
                    conversation_id,
                    user_id,
                    source_message_count,
                    source_hash,
                    summary_text,
                    summary_json,
                    created_at,
                    updated_at
                FROM conversation_context_memory
                WHERE conversation_id IN ({placeholders}) AND user_id=?
                ORDER BY updated_at DESC
                """,
                params,
            ).fetchone()
        if not row:
            return None
        summary_json_raw = row[5]
        try:
            summary_payload = json.loads(summary_json_raw) if summary_json_raw else {}
        except Exception:
            summary_payload = {}
        return {
            "conversation_id": self._external_conversation_id(
                stored_conversation_id=str(row[0]),
                user_id=user_id,
            ),
            "user_id": row[1],
            "source_message_count": int(row[2] or 0),
            "source_hash": str(row[3] or ""),
            "summary_text": str(row[4] or ""),
            "summary": summary_payload,
            "created_at": str(row[6] or ""),
            "updated_at": str(row[7] or ""),
        }

    def upsert_conversation_context_memory(
        self,
        *,
        conversation_id: str,
        user_id: str,
        source_message_count: int,
        source_hash: str,
        summary_text: str,
        summary: dict[str, Any] | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
    ) -> None:
        """Upsert compacted conversation memory used for context compression.
        
        Parameters
        ----------
        conversation_id : str
            Conversation identifier.
        user_id : str
            User identifier.
        source_message_count : int
            Input argument.
        source_hash : str
            Input argument.
        summary_text : str
            Input argument.
        summary : dict[str, Any] | None, optional
            Input argument.
        created_at : str | None, optional
            Timestamp value.
        updated_at : str | None, optional
            Timestamp value.
        
        Returns
        -------
        None
            No return value.
        """
        now = datetime.utcnow().isoformat()
        created_at_iso = str(created_at or "").strip() or now
        updated_at_iso = str(updated_at or "").strip() or now
        summary_json = json.dumps(summary or {}, default=str)
        scoped_conversation_id = self._scoped_conversation_id(conversation_id, user_id)
        with self._lock, self._conn() as conn:
            self._execute(
                conn,
                """
                INSERT INTO conversation_context_memory(
                    conversation_id,
                    user_id,
                    source_message_count,
                    source_hash,
                    summary_text,
                    summary_json,
                    created_at,
                    updated_at
                )
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(conversation_id, user_id) DO UPDATE SET
                    source_message_count=excluded.source_message_count,
                    source_hash=excluded.source_hash,
                    summary_text=excluded.summary_text,
                    summary_json=excluded.summary_json,
                    updated_at=excluded.updated_at
                """,
                (
                    scoped_conversation_id,
                    user_id,
                    int(max(0, source_message_count)),
                    str(source_hash or "").strip(),
                    str(summary_text or "").strip(),
                    summary_json,
                    created_at_iso,
                    updated_at_iso,
                ),
            )
            conn.commit()

    def search_conversation_messages(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 50,
        include_conversation_ids: list[str] | None = None,
        exclude_conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search a user's conversation messages using FTS when available.
        
        Parameters
        ----------
        user_id : str
            User identifier.
        query : str
            Free-text query string.
        limit : int, optional
            Maximum number of records to return.
        include_conversation_ids : list[str] | None, optional
            Collection of identifier values.
        exclude_conversation_id : str | None, optional
            Identifier value.
        
        Returns
        -------
        list[dict[str, Any]]
            Computed result.
        
        Notes
        -----
        Prefer scoped include/exclude conversation IDs to reduce noisy matches.
        Keep query text user-originated and non-empty before issuing DB scans.
        """
        query_text = str(query or "").strip()
        if not query_text:
            return []
        include_lookup_ids: list[str] = []
        include_seen: set[str] = set()
        for raw_id in [
            str(item or "").strip()
            for item in (include_conversation_ids or [])
            if str(item or "").strip()
        ]:
            for candidate in self._conversation_lookup_ids(raw_id, user_id):
                candidate_id = str(candidate or "").strip()
                if not candidate_id or candidate_id in include_seen:
                    continue
                include_seen.add(candidate_id)
                include_lookup_ids.append(candidate_id)

        exclude_lookup_ids: list[str] = []
        exclude_raw_id = str(exclude_conversation_id or "").strip()
        if exclude_raw_id:
            for candidate in self._conversation_lookup_ids(exclude_raw_id, user_id):
                candidate_id = str(candidate or "").strip()
                if candidate_id and candidate_id not in exclude_lookup_ids:
                    exclude_lookup_ids.append(candidate_id)

        def _map_rows(rows: list[Any], has_score: bool = False) -> list[dict[str, Any]]:
            output: list[dict[str, Any]] = []
            for row in rows:
                record = {
                    "conversation_id": self._external_conversation_id(
                        stored_conversation_id=str(row[0]),
                        user_id=user_id,
                    ),
                    "message_id": row[1],
                    "role": row[2],
                    "content": row[3],
                    "created_at_ms": row[4],
                    "run_id": row[5],
                }
                if has_score:
                    try:
                        score_value = float(row[6])
                    except Exception:
                        score_value = 0.0
                    record["search_score"] = score_value
                output.append(record)
            return output

        token_candidates = _tokenize_message_search_terms(query_text, max_terms=16)

        def _build_message_filters() -> tuple[list[str], list[Any]]:
            clauses = ["user_id=?"]
            params: list[Any] = [user_id]
            if exclude_lookup_ids:
                placeholders = ",".join(["?"] * len(exclude_lookup_ids))
                clauses.append(f"conversation_id NOT IN ({placeholders})")
                params.extend(exclude_lookup_ids)
            if include_lookup_ids:
                placeholders = ",".join(["?"] * len(include_lookup_ids))
                clauses.append(f"conversation_id IN ({placeholders})")
                params.extend(include_lookup_ids)
            return clauses, params

        with self._lock, self._conn() as conn:
            if self._backend == "sqlite" and self._sqlite_fts_enabled and token_candidates:
                try:
                    fts_query = " ".join(f"{token}*" for token in token_candidates)
                    where_clauses, where_params = _build_message_filters()
                    rows = self._execute(
                        conn,
                        """
                        SELECT
                            conversation_id,
                            message_id,
                            role,
                            content,
                            CAST(COALESCE(created_at_ms, '0') AS INTEGER) AS created_at_ms,
                            run_id,
                            bm25(conversation_messages_fts) AS score
                        FROM conversation_messages_fts
                        WHERE conversation_messages_fts MATCH ? AND {where_clause}
                        ORDER BY score ASC, created_at_ms DESC
                        LIMIT ?
                        """.replace("{where_clause}", " AND ".join(where_clauses)),
                        tuple([fts_query, *where_params, int(limit)]),
                    ).fetchall()
                    if rows:
                        mapped = _map_rows(rows, has_score=True)
                        for item in mapped:
                            raw = float(item.get("search_score") or 0.0)
                            item["search_score"] = -raw
                        return mapped
                except sqlite3.OperationalError:
                    pass

            if self._backend == "postgres" and token_candidates:
                try:
                    where_clauses, where_params = _build_message_filters()
                    ts_query = " ".join(token_candidates)
                    rows = self._execute(
                        conn,
                        """
                        SELECT
                            conversation_id,
                            message_id,
                            role,
                            content,
                            created_at_ms,
                            run_id,
                            ts_rank_cd(
                                to_tsvector('simple', COALESCE(content, '')),
                                websearch_to_tsquery('simple', ?)
                            ) AS score
                        FROM conversation_messages
                        WHERE {where_clause}
                          AND to_tsvector('simple', COALESCE(content, ''))
                              @@ websearch_to_tsquery('simple', ?)
                        ORDER BY score DESC, created_at_ms DESC
                        LIMIT ?
                        """.replace("{where_clause}", " AND ".join(where_clauses)),
                        tuple([ts_query, *where_params, ts_query, int(limit)]),
                    ).fetchall()
                    if rows:
                        return _map_rows(rows, has_score=True)
                except Exception:
                    pass

            where_clauses, where_params = _build_message_filters()
            where_clauses.append("LOWER(content) LIKE LOWER(?)")
            like_query_text = f"%{query_text}%"
            rows = self._execute(
                conn,
                """
                SELECT conversation_id, message_id, role, content, created_at_ms, run_id
                FROM conversation_messages
                WHERE {where_clause}
                ORDER BY created_at_ms DESC
                LIMIT ?
                """.replace("{where_clause}", " AND ".join(where_clauses)),
                tuple([*where_params, like_query_text, int(limit)]),
            ).fetchall()
        return _map_rows(rows, has_score=False)

    def upsert_resource_computations(self, rows: list[dict[str, Any]]) -> None:
        """Upsert tool-to-resource computation records for provenance lookups.
        
        Parameters
        ----------
        rows : list[dict[str, Any]]
            Input argument.
        
        Returns
        -------
        None
            No return value.
        """
        if not rows:
            return

        now = datetime.utcnow().isoformat()

        def _normalize_name(value: str) -> str:
            return " ".join(str(value or "").strip().lower().split())

        with self._lock, self._conn() as conn:
            for row in rows:
                run_id = str(row.get("run_id") or "").strip()
                user_id = str(row.get("user_id") or "").strip()
                tool_name = str(row.get("tool_name") or "").strip()
                file_sha256 = str(row.get("file_sha256") or "").strip().lower()
                file_name = str(row.get("file_name") or "").strip()
                if not (run_id and user_id and tool_name and file_sha256 and file_name):
                    continue

                file_name_norm = _normalize_name(file_name)
                if not file_name_norm:
                    continue

                metadata = row.get("metadata")
                metadata_json = (
                    json.dumps(metadata, default=str)
                    if isinstance(metadata, (dict, list))
                    else None
                )
                self._execute(
                    conn,
                    """
                    INSERT INTO resource_computations(
                        run_id,
                        user_id,
                        conversation_id,
                        tool_name,
                        file_sha256,
                        file_id,
                        file_name,
                        file_name_norm,
                        source_path,
                        run_goal,
                        run_status,
                        run_created_at,
                        run_updated_at,
                        metadata_json,
                        created_at,
                        updated_at
                    )
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(run_id, tool_name, file_sha256) DO UPDATE SET
                        user_id=excluded.user_id,
                        conversation_id=excluded.conversation_id,
                        file_id=excluded.file_id,
                        file_name=excluded.file_name,
                        file_name_norm=excluded.file_name_norm,
                        source_path=excluded.source_path,
                        run_goal=excluded.run_goal,
                        run_status=excluded.run_status,
                        run_created_at=excluded.run_created_at,
                        run_updated_at=excluded.run_updated_at,
                        metadata_json=excluded.metadata_json,
                        updated_at=excluded.updated_at
                    """,
                    (
                        run_id,
                        user_id,
                        str(row.get("conversation_id") or "").strip() or None,
                        tool_name,
                        file_sha256,
                        str(row.get("file_id") or "").strip() or None,
                        file_name,
                        file_name_norm,
                        str(row.get("source_path") or "").strip() or None,
                        str(row.get("run_goal") or "").strip() or None,
                        str(row.get("run_status") or "").strip() or "unknown",
                        str(row.get("run_created_at") or "").strip() or None,
                        str(row.get("run_updated_at") or "").strip() or None,
                        metadata_json,
                        str(row.get("created_at") or "").strip() or now,
                        str(row.get("updated_at") or "").strip() or now,
                    ),
                )
            conn.commit()

    def list_resource_computations(
        self,
        *,
        user_id: str,
        tool_names: list[str] | None = None,
        file_sha256s: list[str] | None = None,
        file_name_norms: list[str] | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """List resource computation rows filtered by tool and file identifiers.
        
        Parameters
        ----------
        user_id : str
            User identifier.
        tool_names : list[str] | None, optional
            Input argument.
        file_sha256s : list[str] | None, optional
            Input argument.
        file_name_norms : list[str] | None, optional
            Input argument.
        limit : int, optional
            Maximum number of records to return.
        
        Returns
        -------
        list[dict[str, Any]]
            Computed result.
        """
        clauses = ["user_id=?"]
        params: list[Any] = [user_id]

        normalized_tools = [
            str(item or "").strip() for item in (tool_names or []) if str(item or "").strip()
        ]
        if normalized_tools:
            placeholders = ",".join(["?"] * len(normalized_tools))
            clauses.append(f"tool_name IN ({placeholders})")
            params.extend(normalized_tools)

        normalized_sha256s = [
            str(item or "").strip().lower()
            for item in (file_sha256s or [])
            if str(item or "").strip()
        ]
        normalized_names = [
            " ".join(str(item or "").strip().lower().split())
            for item in (file_name_norms or [])
            if str(item or "").strip()
        ]
        match_parts: list[str] = []
        if normalized_sha256s:
            placeholders = ",".join(["?"] * len(normalized_sha256s))
            match_parts.append(f"file_sha256 IN ({placeholders})")
            params.extend(normalized_sha256s)
        if normalized_names:
            placeholders = ",".join(["?"] * len(normalized_names))
            match_parts.append(f"file_name_norm IN ({placeholders})")
            params.extend(normalized_names)
        if match_parts:
            clauses.append("(" + " OR ".join(match_parts) + ")")

        params.append(max(1, int(limit)))
        with self._lock, self._conn() as conn:
            rows = self._execute(
                conn,
                """
                SELECT
                    run_id,
                    user_id,
                    conversation_id,
                    tool_name,
                    file_sha256,
                    file_id,
                    file_name,
                    file_name_norm,
                    source_path,
                    run_goal,
                    run_status,
                    run_created_at,
                    run_updated_at,
                    metadata_json,
                    created_at,
                    updated_at
                FROM resource_computations
                WHERE {where_clause}
                ORDER BY run_updated_at DESC, updated_at DESC
                LIMIT ?
                """.replace("{where_clause}", " AND ".join(clauses)),
                tuple(params),
            ).fetchall()

        output: list[dict[str, Any]] = []
        for row in rows:
            metadata_raw = row[13]
            try:
                metadata = json.loads(metadata_raw) if metadata_raw else {}
            except Exception:
                metadata = {}
            output.append(
                {
                    "run_id": row[0],
                    "user_id": row[1],
                    "conversation_id": row[2],
                    "tool_name": row[3],
                    "file_sha256": row[4],
                    "file_id": row[5],
                    "file_name": row[6],
                    "file_name_norm": row[7],
                    "source_path": row[8],
                    "run_goal": row[9],
                    "run_status": row[10],
                    "run_created_at": row[11],
                    "run_updated_at": row[12],
                    "metadata": metadata,
                    "created_at": row[14],
                    "updated_at": row[15],
                }
            )
        return output

    def summarize_resource_computations_for_runs(
        self,
        *,
        user_id: str,
        run_ids: list[str],
        max_files_per_run: int = 12,
        max_tools_per_run: int = 8,
    ) -> dict[str, dict[str, list[str]]]:
        """Summarize distinct tools/file names linked to each run ID.
        
        Parameters
        ----------
        user_id : str
            User identifier.
        run_ids : list[str]
            Collection of identifier values.
        max_files_per_run : int, optional
            Maximum allowed value.
        max_tools_per_run : int, optional
            Maximum allowed value.
        
        Returns
        -------
        dict[str, dict[str, list[str]]]
            Result payload.
        """
        normalized_run_ids = [
            str(run_id or "").strip() for run_id in run_ids if str(run_id or "").strip()
        ]
        if not normalized_run_ids:
            return {}

        placeholders = ",".join(["?"] * len(normalized_run_ids))
        params: list[Any] = [user_id, *normalized_run_ids]
        with self._lock, self._conn() as conn:
            rows = self._execute(
                conn,
                """
                SELECT run_id, tool_name, file_name
                FROM resource_computations
                WHERE user_id=? AND run_id IN ({placeholders})
                ORDER BY COALESCE(run_updated_at, updated_at, created_at) DESC, updated_at DESC
                """.replace("{placeholders}", placeholders),
                tuple(params),
            ).fetchall()

        summary: dict[str, dict[str, list[str]]] = {}
        seen_tools: dict[str, set[str]] = {}
        seen_files: dict[str, set[str]] = {}
        for run_id_raw, tool_name_raw, file_name_raw in rows:
            run_id = str(run_id_raw or "").strip()
            if not run_id:
                continue
            bucket = summary.setdefault(run_id, {"tools": [], "file_names": []})
            tool_seen = seen_tools.setdefault(run_id, set())
            file_seen = seen_files.setdefault(run_id, set())

            tool_name = str(tool_name_raw or "").strip()
            if (
                tool_name
                and tool_name not in tool_seen
                and len(bucket["tools"]) < max(1, int(max_tools_per_run))
            ):
                bucket["tools"].append(tool_name)
                tool_seen.add(tool_name)

            file_name = str(file_name_raw or "").strip()
            normalized_file = file_name.lower()
            if (
                file_name
                and normalized_file not in file_seen
                and len(bucket["file_names"]) < max(1, int(max_files_per_run))
            ):
                bucket["file_names"].append(file_name)
                file_seen.add(normalized_file)
        return summary

    @staticmethod
    def _decode_json_payload(raw: Any) -> Any:
        if not raw:
            return None
        try:
            return json.loads(str(raw))
        except Exception:
            return None

    def create_training_dataset(
        self,
        *,
        dataset_id: str,
        user_id: str,
        name: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = datetime.utcnow().isoformat()
        metadata_json = json.dumps(metadata, default=str) if metadata is not None else None
        with self._lock, self._conn() as conn:
            self._execute(
                conn,
                """
                INSERT INTO training_datasets(
                    dataset_id,
                    user_id,
                    name,
                    description,
                    metadata_json,
                    created_at,
                    updated_at
                )
                VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(dataset_id) DO UPDATE SET
                    user_id=excluded.user_id,
                    name=excluded.name,
                    description=excluded.description,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    dataset_id,
                    user_id,
                    str(name or "").strip(),
                    str(description or "").strip() or None,
                    metadata_json,
                    now,
                    now,
                ),
            )
            conn.commit()
        payload = self.get_training_dataset(dataset_id=dataset_id, user_id=user_id)
        if payload is None:
            raise RuntimeError(f"Failed to persist training dataset: {dataset_id}")
        return payload

    def upsert_training_managed_source(
        self,
        *,
        source_id: str,
        user_id: str,
        source_type: str,
        name: str,
        training_dataset_id: str | None = None,
        remote_uri: str | None = None,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
        stats: dict[str, Any] | None = None,
        error: str | None = None,
        last_sync_at: str | None = None,
    ) -> dict[str, Any]:
        now = datetime.utcnow().isoformat()
        metadata_json = json.dumps(metadata, default=str) if metadata is not None else None
        stats_json = json.dumps(stats, default=str) if stats is not None else None
        with self._lock, self._conn() as conn:
            self._execute(
                conn,
                """
                INSERT INTO training_managed_sources(
                    source_id,
                    user_id,
                    source_type,
                    name,
                    training_dataset_id,
                    remote_uri,
                    status,
                    metadata_json,
                    stats_json,
                    error,
                    last_sync_at,
                    created_at,
                    updated_at
                )
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(source_id) DO UPDATE SET
                    user_id=excluded.user_id,
                    source_type=excluded.source_type,
                    name=excluded.name,
                    training_dataset_id=COALESCE(excluded.training_dataset_id, training_managed_sources.training_dataset_id),
                    remote_uri=COALESCE(excluded.remote_uri, training_managed_sources.remote_uri),
                    status=excluded.status,
                    metadata_json=COALESCE(excluded.metadata_json, training_managed_sources.metadata_json),
                    stats_json=COALESCE(excluded.stats_json, training_managed_sources.stats_json),
                    error=excluded.error,
                    last_sync_at=COALESCE(excluded.last_sync_at, training_managed_sources.last_sync_at),
                    updated_at=excluded.updated_at
                """,
                (
                    str(source_id or "").strip(),
                    str(user_id or "").strip(),
                    str(source_type or "").strip().lower(),
                    str(name or "").strip(),
                    str(training_dataset_id or "").strip() or None,
                    str(remote_uri or "").strip() or None,
                    str(status or "").strip().lower() or "idle",
                    metadata_json,
                    stats_json,
                    str(error or "").strip() or None,
                    str(last_sync_at or "").strip() or None,
                    now,
                    now,
                ),
            )
            conn.commit()
        payload = self.get_training_managed_source(source_id=source_id, user_id=user_id)
        if payload is None:
            raise RuntimeError(f"Failed to persist training managed source: {source_id}")
        return payload

    def update_training_managed_source(
        self,
        *,
        source_id: str,
        user_id: str | None = None,
        training_dataset_id: str | None = None,
        remote_uri: str | None = None,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
        stats: dict[str, Any] | None = None,
        error: str | None = None,
        last_sync_at: str | None = None,
    ) -> dict[str, Any] | None:
        assignments: list[str] = []
        params: list[Any] = []
        if training_dataset_id is not None:
            assignments.append("training_dataset_id=?")
            params.append(str(training_dataset_id or "").strip() or None)
        if remote_uri is not None:
            assignments.append("remote_uri=?")
            params.append(str(remote_uri or "").strip() or None)
        if status is not None:
            assignments.append("status=?")
            params.append(str(status or "").strip().lower())
        if metadata is not None:
            assignments.append("metadata_json=?")
            params.append(json.dumps(metadata, default=str))
        if stats is not None:
            assignments.append("stats_json=?")
            params.append(json.dumps(stats, default=str))
        if error is not None:
            assignments.append("error=?")
            params.append(str(error or "").strip() or None)
        if last_sync_at is not None:
            assignments.append("last_sync_at=?")
            params.append(str(last_sync_at or "").strip() or None)
        if not assignments:
            return self.get_training_managed_source(source_id=source_id, user_id=user_id)
        assignments.append("updated_at=?")
        params.append(datetime.utcnow().isoformat())
        where_clause = "source_id=?"
        params.append(str(source_id or "").strip())
        user_token = str(user_id or "").strip()
        if user_token:
            where_clause += " AND user_id=?"
            params.append(user_token)
        with self._lock, self._conn() as conn:
            self._execute(
                conn,
                """
                UPDATE training_managed_sources
                SET {assignments}
                WHERE {where_clause}
                """
                .replace("{assignments}", ", ".join(assignments))
                .replace("{where_clause}", where_clause),
                tuple(params),
            )
            conn.commit()
        return self.get_training_managed_source(source_id=source_id, user_id=user_id)

    def get_training_managed_source(
        self,
        *,
        source_id: str | None = None,
        user_id: str | None = None,
        source_type: str | None = None,
        name: str | None = None,
    ) -> dict[str, Any] | None:
        clauses: list[str] = []
        params: list[Any] = []
        source_id_token = str(source_id or "").strip()
        if source_id_token:
            clauses.append("source_id=?")
            params.append(source_id_token)
        source_type_token = str(source_type or "").strip().lower()
        if source_type_token:
            clauses.append("source_type=?")
            params.append(source_type_token)
        name_token = str(name or "").strip()
        if name_token:
            clauses.append("name=?")
            params.append(name_token)
        user_token = str(user_id or "").strip()
        if user_token:
            clauses.append("user_id=?")
            params.append(user_token)
        where_clause = " AND ".join(clauses) if clauses else "1=1"
        with self._lock, self._conn() as conn:
            row = self._execute(
                conn,
                """
                SELECT
                    source_id,
                    user_id,
                    source_type,
                    name,
                    training_dataset_id,
                    remote_uri,
                    status,
                    metadata_json,
                    stats_json,
                    error,
                    last_sync_at,
                    created_at,
                    updated_at
                FROM training_managed_sources
                WHERE {where_clause}
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 1
                """.replace("{where_clause}", where_clause),
                tuple(params),
            ).fetchone()
        if not row:
            return None
        metadata = self._decode_json_payload(row[7])
        stats = self._decode_json_payload(row[8])
        return {
            "source_id": row[0],
            "user_id": row[1],
            "source_type": row[2],
            "name": row[3],
            "training_dataset_id": row[4],
            "remote_uri": row[5],
            "status": row[6],
            "metadata": metadata if isinstance(metadata, dict) else {},
            "stats": stats if isinstance(stats, dict) else {},
            "error": row[9],
            "last_sync_at": row[10],
            "created_at": row[11],
            "updated_at": row[12],
        }

    def list_training_managed_sources(
        self,
        *,
        user_id: str | None = None,
        source_type: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        user_token = str(user_id or "").strip()
        if user_token:
            clauses.append("user_id=?")
            params.append(user_token)
        source_type_token = str(source_type or "").strip().lower()
        if source_type_token:
            clauses.append("source_type=?")
            params.append(source_type_token)
        where_clause = " AND ".join(clauses) if clauses else "1=1"
        params.append(max(1, int(limit)))
        with self._lock, self._conn() as conn:
            rows = self._execute(
                conn,
                """
                SELECT
                    source_id,
                    user_id,
                    source_type,
                    name,
                    training_dataset_id,
                    remote_uri,
                    status,
                    metadata_json,
                    stats_json,
                    error,
                    last_sync_at,
                    created_at,
                    updated_at
                FROM training_managed_sources
                WHERE {where_clause}
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ?
                """.replace("{where_clause}", where_clause),
                tuple(params),
            ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            metadata = self._decode_json_payload(row[7])
            stats = self._decode_json_payload(row[8])
            output.append(
                {
                    "source_id": row[0],
                    "user_id": row[1],
                    "source_type": row[2],
                    "name": row[3],
                    "training_dataset_id": row[4],
                    "remote_uri": row[5],
                    "status": row[6],
                    "metadata": metadata if isinstance(metadata, dict) else {},
                    "stats": stats if isinstance(stats, dict) else {},
                    "error": row[9],
                    "last_sync_at": row[10],
                    "created_at": row[11],
                    "updated_at": row[12],
                }
            )
        return output

    def get_training_dataset(
        self,
        *,
        dataset_id: str,
        user_id: str | None = None,
    ) -> dict[str, Any] | None:
        clauses = ["dataset_id=?"]
        params: list[Any] = [dataset_id]
        if user_id is not None:
            clauses.append("user_id=?")
            params.append(user_id)
        with self._lock, self._conn() as conn:
            row = self._execute(
                conn,
                """
                SELECT
                    dataset_id,
                    user_id,
                    name,
                    description,
                    metadata_json,
                    created_at,
                    updated_at
                FROM training_datasets
                WHERE {where_clause}
                """.replace("{where_clause}", " AND ".join(clauses)),
                tuple(params),
            ).fetchone()
        if not row:
            return None
        metadata = self._decode_json_payload(row[4])
        return {
            "dataset_id": row[0],
            "user_id": row[1],
            "name": row[2],
            "description": row[3],
            "metadata": metadata if isinstance(metadata, dict) else {},
            "created_at": row[5],
            "updated_at": row[6],
        }

    def list_training_datasets(
        self,
        *,
        user_id: str,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = self._execute(
                conn,
                """
                SELECT
                    dataset_id,
                    user_id,
                    name,
                    description,
                    metadata_json,
                    created_at,
                    updated_at
                FROM training_datasets
                WHERE user_id=?
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ?
                """,
                (user_id, int(max(1, limit))),
            ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            metadata = self._decode_json_payload(row[4])
            output.append(
                {
                    "dataset_id": row[0],
                    "user_id": row[1],
                    "name": row[2],
                    "description": row[3],
                    "metadata": metadata if isinstance(metadata, dict) else {},
                    "created_at": row[5],
                    "updated_at": row[6],
                }
            )
        return output

    def upsert_training_dataset_items(
        self,
        *,
        dataset_id: str,
        user_id: str,
        items: list[dict[str, Any]],
        replace: bool = False,
    ) -> int:
        now = datetime.utcnow().isoformat()
        with self._lock, self._conn() as conn:
            if replace:
                self._execute(
                    conn,
                    "DELETE FROM training_dataset_items WHERE dataset_id=? AND user_id=?",
                    (dataset_id, user_id),
                )
            upserted = 0
            for raw in items:
                split = str(raw.get("split") or "").strip().lower()
                role = str(raw.get("role") or "").strip().lower()
                sample_id = str(raw.get("sample_id") or "").strip()
                file_id = str(raw.get("file_id") or "").strip()
                path = str(raw.get("path") or "").strip()
                if not (split and role and sample_id and file_id and path):
                    continue
                metadata = raw.get("metadata")
                metadata_json = json.dumps(metadata, default=str) if isinstance(metadata, dict) else None
                size_bytes = int(raw.get("size_bytes") or 0)
                item_id = (
                    str(raw.get("item_id") or "").strip()
                    or f"{dataset_id}:{split}:{role}:{sample_id}:{uuid4().hex[:8]}"
                )
                self._execute(
                    conn,
                    """
                    INSERT INTO training_dataset_items(
                        item_id,
                        dataset_id,
                        user_id,
                        split,
                        role,
                        sample_id,
                        file_id,
                        path,
                        original_name,
                        sha256,
                        size_bytes,
                        metadata_json,
                        created_at,
                        updated_at
                    )
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(dataset_id, split, role, sample_id) DO UPDATE SET
                        item_id=excluded.item_id,
                        user_id=excluded.user_id,
                        file_id=excluded.file_id,
                        path=excluded.path,
                        original_name=excluded.original_name,
                        sha256=excluded.sha256,
                        size_bytes=excluded.size_bytes,
                        metadata_json=excluded.metadata_json,
                        updated_at=excluded.updated_at
                    """,
                    (
                        item_id,
                        dataset_id,
                        user_id,
                        split,
                        role,
                        sample_id,
                        file_id,
                        path,
                        str(raw.get("original_name") or "").strip() or None,
                        str(raw.get("sha256") or "").strip() or None,
                        size_bytes,
                        metadata_json,
                        now,
                        now,
                    ),
                )
                upserted += 1
            self._execute(
                conn,
                "UPDATE training_datasets SET updated_at=? WHERE dataset_id=? AND user_id=?",
                (now, dataset_id, user_id),
            )
            conn.commit()
        return upserted

    def list_training_dataset_items(
        self,
        *,
        dataset_id: str,
        user_id: str,
    ) -> list[dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = self._execute(
                conn,
                """
                SELECT
                    item_id,
                    dataset_id,
                    user_id,
                    split,
                    role,
                    sample_id,
                    file_id,
                    path,
                    original_name,
                    sha256,
                    size_bytes,
                    metadata_json,
                    created_at,
                    updated_at
                FROM training_dataset_items
                WHERE dataset_id=? AND user_id=?
                ORDER BY split ASC, sample_id ASC, role ASC
                """,
                (dataset_id, user_id),
            ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            metadata = self._decode_json_payload(row[11])
            output.append(
                {
                    "item_id": row[0],
                    "dataset_id": row[1],
                    "user_id": row[2],
                    "split": row[3],
                    "role": row[4],
                    "sample_id": row[5],
                    "file_id": row[6],
                    "path": row[7],
                    "original_name": row[8],
                    "sha256": row[9],
                    "size_bytes": int(row[10] or 0),
                    "metadata": metadata if isinstance(metadata, dict) else {},
                    "created_at": row[12],
                    "updated_at": row[13],
                }
            )
        return output

    def create_training_job(
        self,
        *,
        job_id: str,
        user_id: str,
        job_type: str,
        dataset_id: str | None,
        model_key: str,
        model_version: str | None,
        status: str,
        request: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        control: dict[str, Any] | None = None,
        artifact_run_id: str | None = None,
        error: str | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
        last_heartbeat_at: str | None = None,
    ) -> dict[str, Any]:
        now = datetime.utcnow().isoformat()
        request_json = json.dumps(request, default=str) if request is not None else None
        result_json = json.dumps(result, default=str) if result is not None else None
        control_json = json.dumps(control, default=str) if control is not None else None
        with self._lock, self._conn() as conn:
            self._execute(
                conn,
                """
                INSERT INTO training_jobs(
                    job_id,
                    user_id,
                    job_type,
                    dataset_id,
                    model_key,
                    model_version,
                    status,
                    request_json,
                    result_json,
                    control_json,
                    artifact_run_id,
                    error,
                    created_at,
                    updated_at,
                    started_at,
                    finished_at,
                    last_heartbeat_at
                )
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(job_id) DO UPDATE SET
                    user_id=excluded.user_id,
                    job_type=excluded.job_type,
                    dataset_id=excluded.dataset_id,
                    model_key=excluded.model_key,
                    model_version=COALESCE(excluded.model_version, training_jobs.model_version),
                    status=excluded.status,
                    request_json=COALESCE(excluded.request_json, training_jobs.request_json),
                    result_json=COALESCE(excluded.result_json, training_jobs.result_json),
                    control_json=COALESCE(excluded.control_json, training_jobs.control_json),
                    artifact_run_id=COALESCE(excluded.artifact_run_id, training_jobs.artifact_run_id),
                    error=excluded.error,
                    updated_at=excluded.updated_at,
                    started_at=COALESCE(excluded.started_at, training_jobs.started_at),
                    finished_at=COALESCE(excluded.finished_at, training_jobs.finished_at),
                    last_heartbeat_at=COALESCE(excluded.last_heartbeat_at, training_jobs.last_heartbeat_at)
                """,
                (
                    job_id,
                    user_id,
                    str(job_type or "").strip().lower(),
                    str(dataset_id or "").strip() or None,
                    str(model_key or "").strip().lower(),
                    str(model_version or "").strip() or None,
                    str(status or "").strip().lower(),
                    request_json,
                    result_json,
                    control_json,
                    str(artifact_run_id or "").strip() or None,
                    str(error or "").strip() or None,
                    now,
                    now,
                    started_at,
                    finished_at,
                    last_heartbeat_at,
                ),
            )
            conn.commit()
        payload = self.get_training_job(job_id=job_id, user_id=user_id)
        if payload is None:
            raise RuntimeError(f"Failed to persist training job: {job_id}")
        return payload

    def update_training_job(
        self,
        *,
        job_id: str,
        user_id: str | None = None,
        status: str | None = None,
        model_version: str | None = None,
        request: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        control: dict[str, Any] | None = None,
        artifact_run_id: str | None = None,
        error: str | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
        heartbeat_at: str | None = None,
    ) -> dict[str, Any] | None:
        assignments: list[str] = []
        params: list[Any] = []
        if status is not None:
            assignments.append("status=?")
            params.append(str(status))
        if model_version is not None:
            assignments.append("model_version=?")
            params.append(str(model_version))
        if request is not None:
            assignments.append("request_json=?")
            params.append(json.dumps(request, default=str))
        if result is not None:
            assignments.append("result_json=?")
            params.append(json.dumps(result, default=str))
        if control is not None:
            assignments.append("control_json=?")
            params.append(json.dumps(control, default=str))
        if artifact_run_id is not None:
            assignments.append("artifact_run_id=?")
            params.append(str(artifact_run_id))
        if error is not None:
            assignments.append("error=?")
            params.append(str(error))
        if started_at is not None:
            assignments.append("started_at=?")
            params.append(started_at)
        if finished_at is not None:
            assignments.append("finished_at=?")
            params.append(finished_at)
        if heartbeat_at is not None:
            assignments.append("last_heartbeat_at=?")
            params.append(heartbeat_at)
        if not assignments:
            return self.get_training_job(job_id=job_id, user_id=user_id)
        assignments.append("updated_at=?")
        params.append(datetime.utcnow().isoformat())
        where_clause = "job_id=?"
        params.append(job_id)
        if user_id is not None:
            where_clause += " AND user_id=?"
            params.append(user_id)
        with self._lock, self._conn() as conn:
            self._execute(
                conn,
                """
                UPDATE training_jobs
                SET {assignments}
                WHERE {where_clause}
                """
                .replace("{assignments}", ", ".join(assignments))
                .replace("{where_clause}", where_clause),
                tuple(params),
            )
            conn.commit()
        return self.get_training_job(job_id=job_id, user_id=user_id)

    def get_training_job(
        self,
        *,
        job_id: str,
        user_id: str | None = None,
    ) -> dict[str, Any] | None:
        clauses = ["job_id=?"]
        params: list[Any] = [job_id]
        if user_id is not None:
            clauses.append("user_id=?")
            params.append(user_id)
        with self._lock, self._conn() as conn:
            row = self._execute(
                conn,
                """
                SELECT
                    job_id,
                    user_id,
                    job_type,
                    dataset_id,
                    model_key,
                    model_version,
                    status,
                    request_json,
                    result_json,
                    control_json,
                    artifact_run_id,
                    error,
                    created_at,
                    updated_at,
                    started_at,
                    finished_at,
                    last_heartbeat_at
                FROM training_jobs
                WHERE {where_clause}
                """.replace("{where_clause}", " AND ".join(clauses)),
                tuple(params),
            ).fetchone()
        if not row:
            return None
        request_payload = self._decode_json_payload(row[7])
        result_payload = self._decode_json_payload(row[8])
        control_payload = self._decode_json_payload(row[9])
        return {
            "job_id": row[0],
            "user_id": row[1],
            "job_type": row[2],
            "dataset_id": row[3],
            "model_key": row[4],
            "model_version": row[5],
            "status": row[6],
            "request": request_payload if isinstance(request_payload, dict) else {},
            "result": result_payload if isinstance(result_payload, dict) else {},
            "control": control_payload if isinstance(control_payload, dict) else {},
            "artifact_run_id": row[10],
            "error": row[11],
            "created_at": row[12],
            "updated_at": row[13],
            "started_at": row[14],
            "finished_at": row[15],
            "last_heartbeat_at": row[16],
        }

    def list_training_jobs(
        self,
        *,
        user_id: str | None = None,
        statuses: list[str] | None = None,
        job_type: str | None = None,
        model_key: str | None = None,
        model_version: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if user_id is not None:
            clauses.append("user_id=?")
            params.append(user_id)
        normalized_statuses = [
            str(item or "").strip().lower() for item in (statuses or []) if str(item or "").strip()
        ]
        if normalized_statuses:
            placeholders = ",".join(["?"] * len(normalized_statuses))
            clauses.append(f"LOWER(status) IN ({placeholders})")
            params.extend(normalized_statuses)
        job_type_token = str(job_type or "").strip().lower()
        if job_type_token:
            clauses.append("LOWER(job_type)=?")
            params.append(job_type_token)
        model_key_token = str(model_key or "").strip().lower()
        if model_key_token:
            clauses.append("LOWER(model_key)=?")
            params.append(model_key_token)
        model_version_token = str(model_version or "").strip()
        if model_version_token:
            clauses.append("model_version=?")
            params.append(model_version_token)
        where_clause = " AND ".join(clauses) if clauses else "1=1"
        params.append(max(1, int(limit)))
        with self._lock, self._conn() as conn:
            rows = self._execute(
                conn,
                """
                SELECT
                    job_id,
                    user_id,
                    job_type,
                    dataset_id,
                    model_key,
                    model_version,
                    status,
                    request_json,
                    result_json,
                    control_json,
                    artifact_run_id,
                    error,
                    created_at,
                    updated_at,
                    started_at,
                    finished_at,
                    last_heartbeat_at
                FROM training_jobs
                WHERE {where_clause}
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ?
                """.replace("{where_clause}", where_clause),
                tuple(params),
            ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            request_payload = self._decode_json_payload(row[7])
            result_payload = self._decode_json_payload(row[8])
            control_payload = self._decode_json_payload(row[9])
            output.append(
                {
                    "job_id": row[0],
                    "user_id": row[1],
                    "job_type": row[2],
                    "dataset_id": row[3],
                    "model_key": row[4],
                    "model_version": row[5],
                    "status": row[6],
                    "request": request_payload if isinstance(request_payload, dict) else {},
                    "result": result_payload if isinstance(result_payload, dict) else {},
                    "control": control_payload if isinstance(control_payload, dict) else {},
                    "artifact_run_id": row[10],
                    "error": row[11],
                    "created_at": row[12],
                    "updated_at": row[13],
                    "started_at": row[14],
                    "finished_at": row[15],
                    "last_heartbeat_at": row[16],
                }
            )
        return output

    def append_training_job_event(
        self,
        *,
        job_id: str,
        user_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> int:
        ts = datetime.utcnow().isoformat()
        payload_json = json.dumps(payload, default=str) if payload is not None else None
        with self._lock, self._conn() as conn:
            if self._backend == "postgres":
                row = self._execute(
                    conn,
                    """
                    INSERT INTO training_job_events(job_id, user_id, event_type, ts, payload_json)
                    VALUES(?,?,?,?,?)
                    RETURNING event_id
                    """,
                    (job_id, user_id, event_type, ts, payload_json),
                ).fetchone()
                event_id = int((row or [0])[0] or 0)
            else:
                cursor = self._execute(
                    conn,
                    """
                    INSERT INTO training_job_events(job_id, user_id, event_type, ts, payload_json)
                    VALUES(?,?,?,?,?)
                    """,
                    (job_id, user_id, event_type, ts, payload_json),
                )
                event_id = int(getattr(cursor, "lastrowid", 0) or 0)
            conn.commit()
        return event_id

    def list_training_job_events(
        self,
        *,
        job_id: str,
        user_id: str | None = None,
        after_event_id: int = 0,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        clauses = ["job_id=?", "event_id>?"]
        params: list[Any] = [job_id, int(max(0, after_event_id))]
        if user_id is not None:
            clauses.append("user_id=?")
            params.append(user_id)
        params.append(max(1, int(limit)))
        with self._lock, self._conn() as conn:
            rows = self._execute(
                conn,
                """
                SELECT event_id, job_id, user_id, event_type, ts, payload_json
                FROM training_job_events
                WHERE {where_clause}
                ORDER BY event_id ASC
                LIMIT ?
                """.replace("{where_clause}", " AND ".join(clauses)),
                tuple(params),
            ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            payload = self._decode_json_payload(row[5])
            output.append(
                {
                    "event_id": int(row[0]),
                    "job_id": row[1],
                    "user_id": row[2],
                    "event_type": row[3],
                    "ts": row[4],
                    "payload": payload if isinstance(payload, dict) else {},
                }
            )
        return output

    def create_training_domain(
        self,
        *,
        domain_id: str,
        name: str,
        description: str | None,
        owner_scope: str,
        owner_user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = datetime.utcnow().isoformat()
        metadata_json = json.dumps(metadata, default=str) if metadata is not None else None
        with self._lock, self._conn() as conn:
            self._execute(
                conn,
                """
                INSERT INTO training_domains(
                    domain_id,
                    name,
                    description,
                    owner_scope,
                    owner_user_id,
                    metadata_json,
                    created_at,
                    updated_at
                )
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(domain_id) DO UPDATE SET
                    name=excluded.name,
                    description=excluded.description,
                    owner_scope=excluded.owner_scope,
                    owner_user_id=excluded.owner_user_id,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    str(domain_id or "").strip(),
                    str(name or "").strip(),
                    str(description or "").strip() or None,
                    str(owner_scope or "").strip().lower() or "shared",
                    str(owner_user_id or "").strip(),
                    metadata_json,
                    now,
                    now,
                ),
            )
            conn.commit()
        payload = self.get_training_domain(domain_id=domain_id)
        if payload is None:
            raise RuntimeError(f"Failed to persist training domain: {domain_id}")
        return payload

    def get_training_domain(
        self,
        *,
        domain_id: str,
        owner_user_id: str | None = None,
        include_shared: bool = True,
    ) -> dict[str, Any] | None:
        clauses = ["domain_id=?"]
        params: list[Any] = [str(domain_id or "").strip()]
        owner_token = str(owner_user_id or "").strip()
        if owner_token:
            if include_shared:
                clauses.append("(owner_user_id=? OR owner_scope='shared')")
            else:
                clauses.append("owner_user_id=?")
            params.append(owner_token)
        with self._lock, self._conn() as conn:
            row = self._execute(
                conn,
                """
                SELECT
                    domain_id,
                    name,
                    description,
                    owner_scope,
                    owner_user_id,
                    metadata_json,
                    created_at,
                    updated_at
                FROM training_domains
                WHERE {where_clause}
                """.replace("{where_clause}", " AND ".join(clauses)),
                tuple(params),
            ).fetchone()
        if not row:
            return None
        metadata = self._decode_json_payload(row[5])
        return {
            "domain_id": row[0],
            "name": row[1],
            "description": row[2],
            "owner_scope": row[3],
            "owner_user_id": row[4],
            "metadata": metadata if isinstance(metadata, dict) else {},
            "created_at": row[6],
            "updated_at": row[7],
        }

    def list_training_domains(
        self,
        *,
        owner_user_id: str | None = None,
        include_shared: bool = True,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        owner_token = str(owner_user_id or "").strip()
        if owner_token:
            if include_shared:
                clauses.append("(owner_user_id=? OR owner_scope='shared')")
            else:
                clauses.append("owner_user_id=?")
            params.append(owner_token)
        where_clause = " AND ".join(clauses) if clauses else "1=1"
        params.append(max(1, int(limit)))
        with self._lock, self._conn() as conn:
            rows = self._execute(
                conn,
                """
                SELECT
                    domain_id,
                    name,
                    description,
                    owner_scope,
                    owner_user_id,
                    metadata_json,
                    created_at,
                    updated_at
                FROM training_domains
                WHERE {where_clause}
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ?
                """.replace("{where_clause}", where_clause),
                tuple(params),
            ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            metadata = self._decode_json_payload(row[5])
            output.append(
                {
                    "domain_id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "owner_scope": row[3],
                    "owner_user_id": row[4],
                    "metadata": metadata if isinstance(metadata, dict) else {},
                    "created_at": row[6],
                    "updated_at": row[7],
                }
            )
        return output

    def create_training_lineage(
        self,
        *,
        lineage_id: str,
        domain_id: str,
        scope: str,
        owner_user_id: str,
        model_key: str,
        parent_lineage_id: str | None = None,
        active_version_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = datetime.utcnow().isoformat()
        metadata_json = json.dumps(metadata, default=str) if metadata is not None else None
        with self._lock, self._conn() as conn:
            self._execute(
                conn,
                """
                INSERT INTO training_lineages(
                    lineage_id,
                    domain_id,
                    scope,
                    owner_user_id,
                    model_key,
                    parent_lineage_id,
                    active_version_id,
                    metadata_json,
                    created_at,
                    updated_at
                )
                VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(lineage_id) DO UPDATE SET
                    domain_id=excluded.domain_id,
                    scope=excluded.scope,
                    owner_user_id=excluded.owner_user_id,
                    model_key=excluded.model_key,
                    parent_lineage_id=COALESCE(excluded.parent_lineage_id, training_lineages.parent_lineage_id),
                    active_version_id=COALESCE(excluded.active_version_id, training_lineages.active_version_id),
                    metadata_json=COALESCE(excluded.metadata_json, training_lineages.metadata_json),
                    updated_at=excluded.updated_at
                """,
                (
                    str(lineage_id or "").strip(),
                    str(domain_id or "").strip(),
                    str(scope or "").strip().lower() or "shared",
                    str(owner_user_id or "").strip(),
                    str(model_key or "").strip().lower(),
                    str(parent_lineage_id or "").strip() or None,
                    str(active_version_id or "").strip() or None,
                    metadata_json,
                    now,
                    now,
                ),
            )
            conn.commit()
        payload = self.get_training_lineage(lineage_id=lineage_id)
        if payload is None:
            raise RuntimeError(f"Failed to persist training lineage: {lineage_id}")
        return payload

    def update_training_lineage(
        self,
        *,
        lineage_id: str,
        owner_user_id: str | None = None,
        active_version_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        model_key: str | None = None,
    ) -> dict[str, Any] | None:
        assignments: list[str] = []
        params: list[Any] = []
        if active_version_id is not None:
            assignments.append("active_version_id=?")
            params.append(str(active_version_id or "").strip() or None)
        if metadata is not None:
            assignments.append("metadata_json=?")
            params.append(json.dumps(metadata, default=str))
        if model_key is not None:
            assignments.append("model_key=?")
            params.append(str(model_key or "").strip().lower())
        if not assignments:
            return self.get_training_lineage(
                lineage_id=lineage_id,
                owner_user_id=owner_user_id,
            )
        assignments.append("updated_at=?")
        params.append(datetime.utcnow().isoformat())
        where_clause = "lineage_id=?"
        params.append(str(lineage_id or "").strip())
        owner_token = str(owner_user_id or "").strip()
        if owner_token:
            where_clause += " AND owner_user_id=?"
            params.append(owner_token)
        with self._lock, self._conn() as conn:
            self._execute(
                conn,
                """
                UPDATE training_lineages
                SET {assignments}
                WHERE {where_clause}
                """
                .replace("{assignments}", ", ".join(assignments))
                .replace("{where_clause}", where_clause),
                tuple(params),
            )
            conn.commit()
        return self.get_training_lineage(lineage_id=lineage_id)

    def get_training_lineage(
        self,
        *,
        lineage_id: str,
        owner_user_id: str | None = None,
        include_shared: bool = True,
    ) -> dict[str, Any] | None:
        clauses = ["lineage_id=?"]
        params: list[Any] = [str(lineage_id or "").strip()]
        owner_token = str(owner_user_id or "").strip()
        if owner_token:
            if include_shared:
                clauses.append("(owner_user_id=? OR scope='shared')")
            else:
                clauses.append("owner_user_id=?")
            params.append(owner_token)
        with self._lock, self._conn() as conn:
            row = self._execute(
                conn,
                """
                SELECT
                    lineage_id,
                    domain_id,
                    scope,
                    owner_user_id,
                    model_key,
                    parent_lineage_id,
                    active_version_id,
                    metadata_json,
                    created_at,
                    updated_at
                FROM training_lineages
                WHERE {where_clause}
                """.replace("{where_clause}", " AND ".join(clauses)),
                tuple(params),
            ).fetchone()
        if not row:
            return None
        metadata = self._decode_json_payload(row[7])
        return {
            "lineage_id": row[0],
            "domain_id": row[1],
            "scope": row[2],
            "owner_user_id": row[3],
            "model_key": row[4],
            "parent_lineage_id": row[5],
            "active_version_id": row[6],
            "metadata": metadata if isinstance(metadata, dict) else {},
            "created_at": row[8],
            "updated_at": row[9],
        }

    def list_training_lineages(
        self,
        *,
        domain_id: str | None = None,
        owner_user_id: str | None = None,
        include_shared: bool = True,
        scope: str | None = None,
        model_key: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        domain_token = str(domain_id or "").strip()
        if domain_token:
            clauses.append("domain_id=?")
            params.append(domain_token)
        owner_token = str(owner_user_id or "").strip()
        if owner_token:
            if include_shared:
                clauses.append("(owner_user_id=? OR scope='shared')")
            else:
                clauses.append("owner_user_id=?")
            params.append(owner_token)
        scope_token = str(scope or "").strip().lower()
        if scope_token:
            clauses.append("scope=?")
            params.append(scope_token)
        model_token = str(model_key or "").strip().lower()
        if model_token:
            clauses.append("model_key=?")
            params.append(model_token)
        where_clause = " AND ".join(clauses) if clauses else "1=1"
        params.append(max(1, int(limit)))
        with self._lock, self._conn() as conn:
            rows = self._execute(
                conn,
                """
                SELECT
                    lineage_id,
                    domain_id,
                    scope,
                    owner_user_id,
                    model_key,
                    parent_lineage_id,
                    active_version_id,
                    metadata_json,
                    created_at,
                    updated_at
                FROM training_lineages
                WHERE {where_clause}
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ?
                """.replace("{where_clause}", where_clause),
                tuple(params),
            ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            metadata = self._decode_json_payload(row[7])
            output.append(
                {
                    "lineage_id": row[0],
                    "domain_id": row[1],
                    "scope": row[2],
                    "owner_user_id": row[3],
                    "model_key": row[4],
                    "parent_lineage_id": row[5],
                    "active_version_id": row[6],
                    "metadata": metadata if isinstance(metadata, dict) else {},
                    "created_at": row[8],
                    "updated_at": row[9],
                }
            )
        return output

    def create_training_model_version(
        self,
        *,
        version_id: str,
        lineage_id: str,
        source_job_id: str | None = None,
        artifact_run_id: str | None = None,
        status: str,
        metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = datetime.utcnow().isoformat()
        metrics_json = json.dumps(metrics, default=str) if metrics is not None else None
        metadata_json = json.dumps(metadata, default=str) if metadata is not None else None
        with self._lock, self._conn() as conn:
            self._execute(
                conn,
                """
                INSERT INTO training_model_versions(
                    version_id,
                    lineage_id,
                    source_job_id,
                    artifact_run_id,
                    status,
                    metrics_json,
                    metadata_json,
                    created_at,
                    updated_at
                )
                VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(version_id) DO UPDATE SET
                    lineage_id=excluded.lineage_id,
                    source_job_id=COALESCE(excluded.source_job_id, training_model_versions.source_job_id),
                    artifact_run_id=COALESCE(excluded.artifact_run_id, training_model_versions.artifact_run_id),
                    status=excluded.status,
                    metrics_json=COALESCE(excluded.metrics_json, training_model_versions.metrics_json),
                    metadata_json=COALESCE(excluded.metadata_json, training_model_versions.metadata_json),
                    updated_at=excluded.updated_at
                """,
                (
                    str(version_id or "").strip(),
                    str(lineage_id or "").strip(),
                    str(source_job_id or "").strip() or None,
                    str(artifact_run_id or "").strip() or None,
                    str(status or "").strip().lower(),
                    metrics_json,
                    metadata_json,
                    now,
                    now,
                ),
            )
            conn.commit()
        payload = self.get_training_model_version(version_id=version_id)
        if payload is None:
            raise RuntimeError(f"Failed to persist training model version: {version_id}")
        return payload

    def update_training_model_version(
        self,
        *,
        version_id: str,
        status: str | None = None,
        metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        assignments: list[str] = []
        params: list[Any] = []
        if status is not None:
            assignments.append("status=?")
            params.append(str(status or "").strip().lower())
        if metrics is not None:
            assignments.append("metrics_json=?")
            params.append(json.dumps(metrics, default=str))
        if metadata is not None:
            assignments.append("metadata_json=?")
            params.append(json.dumps(metadata, default=str))
        if not assignments:
            return self.get_training_model_version(version_id=version_id)
        assignments.append("updated_at=?")
        params.append(datetime.utcnow().isoformat())
        params.append(str(version_id or "").strip())
        with self._lock, self._conn() as conn:
            self._execute(
                conn,
                """
                UPDATE training_model_versions
                SET {assignments}
                WHERE version_id=?
                """
                .replace("{assignments}", ", ".join(assignments)),
                tuple(params),
            )
            conn.commit()
        return self.get_training_model_version(version_id=version_id)

    def get_training_model_version(self, *, version_id: str) -> dict[str, Any] | None:
        with self._lock, self._conn() as conn:
            row = self._execute(
                conn,
                """
                SELECT
                    version_id,
                    lineage_id,
                    source_job_id,
                    artifact_run_id,
                    status,
                    metrics_json,
                    metadata_json,
                    created_at,
                    updated_at
                FROM training_model_versions
                WHERE version_id=?
                """,
                (str(version_id or "").strip(),),
            ).fetchone()
        if not row:
            return None
        metrics = self._decode_json_payload(row[5])
        metadata = self._decode_json_payload(row[6])
        return {
            "version_id": row[0],
            "lineage_id": row[1],
            "source_job_id": row[2],
            "artifact_run_id": row[3],
            "status": row[4],
            "metrics": metrics if isinstance(metrics, dict) else {},
            "metadata": metadata if isinstance(metadata, dict) else {},
            "created_at": row[7],
            "updated_at": row[8],
        }

    def list_training_model_versions(
        self,
        *,
        lineage_id: str,
        statuses: list[str] | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses = ["lineage_id=?"]
        params: list[Any] = [str(lineage_id or "").strip()]
        normalized_statuses = [
            str(item or "").strip().lower() for item in (statuses or []) if str(item or "").strip()
        ]
        if normalized_statuses:
            placeholders = ",".join(["?"] * len(normalized_statuses))
            clauses.append(f"LOWER(status) IN ({placeholders})")
            params.extend(normalized_statuses)
        params.append(max(1, int(limit)))
        with self._lock, self._conn() as conn:
            rows = self._execute(
                conn,
                """
                SELECT
                    version_id,
                    lineage_id,
                    source_job_id,
                    artifact_run_id,
                    status,
                    metrics_json,
                    metadata_json,
                    created_at,
                    updated_at
                FROM training_model_versions
                WHERE {where_clause}
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ?
                """.replace("{where_clause}", " AND ".join(clauses)),
                tuple(params),
            ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            metrics = self._decode_json_payload(row[5])
            metadata = self._decode_json_payload(row[6])
            output.append(
                {
                    "version_id": row[0],
                    "lineage_id": row[1],
                    "source_job_id": row[2],
                    "artifact_run_id": row[3],
                    "status": row[4],
                    "metrics": metrics if isinstance(metrics, dict) else {},
                    "metadata": metadata if isinstance(metadata, dict) else {},
                    "created_at": row[7],
                    "updated_at": row[8],
                }
            )
        return output

    def create_training_update_proposal(
        self,
        *,
        proposal_id: str,
        lineage_id: str,
        trigger_reason: str,
        trigger_snapshot: dict[str, Any] | None = None,
        dataset_snapshot: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        status: str,
        idempotency_key: str | None = None,
        approved_by: str | None = None,
        rejected_by: str | None = None,
        linked_job_id: str | None = None,
        candidate_version_id: str | None = None,
        error: str | None = None,
        approved_at: str | None = None,
        rejected_at: str | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
    ) -> dict[str, Any]:
        now = datetime.utcnow().isoformat()
        trigger_snapshot_json = (
            json.dumps(trigger_snapshot, default=str) if trigger_snapshot is not None else None
        )
        dataset_snapshot_json = (
            json.dumps(dataset_snapshot, default=str) if dataset_snapshot is not None else None
        )
        config_json = json.dumps(config, default=str) if config is not None else None
        idempotency_token = str(idempotency_key or "").strip() or None
        try:
            with self._lock, self._conn() as conn:
                self._execute(
                    conn,
                    """
                    INSERT INTO training_update_proposals(
                        proposal_id,
                        lineage_id,
                        trigger_reason,
                        trigger_snapshot_json,
                        dataset_snapshot_json,
                        config_json,
                        status,
                        idempotency_key,
                        approved_by,
                        rejected_by,
                        linked_job_id,
                        candidate_version_id,
                        error,
                        created_at,
                        updated_at,
                        approved_at,
                        rejected_at,
                        started_at,
                        finished_at
                    )
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(proposal_id) DO UPDATE SET
                        lineage_id=excluded.lineage_id,
                        trigger_reason=excluded.trigger_reason,
                        trigger_snapshot_json=COALESCE(excluded.trigger_snapshot_json, training_update_proposals.trigger_snapshot_json),
                        dataset_snapshot_json=COALESCE(excluded.dataset_snapshot_json, training_update_proposals.dataset_snapshot_json),
                        config_json=COALESCE(excluded.config_json, training_update_proposals.config_json),
                        status=excluded.status,
                        idempotency_key=COALESCE(excluded.idempotency_key, training_update_proposals.idempotency_key),
                        approved_by=COALESCE(excluded.approved_by, training_update_proposals.approved_by),
                        rejected_by=COALESCE(excluded.rejected_by, training_update_proposals.rejected_by),
                        linked_job_id=COALESCE(excluded.linked_job_id, training_update_proposals.linked_job_id),
                        candidate_version_id=COALESCE(excluded.candidate_version_id, training_update_proposals.candidate_version_id),
                        error=COALESCE(excluded.error, training_update_proposals.error),
                        updated_at=excluded.updated_at,
                        approved_at=COALESCE(excluded.approved_at, training_update_proposals.approved_at),
                        rejected_at=COALESCE(excluded.rejected_at, training_update_proposals.rejected_at),
                        started_at=COALESCE(excluded.started_at, training_update_proposals.started_at),
                        finished_at=COALESCE(excluded.finished_at, training_update_proposals.finished_at)
                    """,
                    (
                        str(proposal_id or "").strip(),
                        str(lineage_id or "").strip(),
                        str(trigger_reason or "").strip().lower() or "manual",
                        trigger_snapshot_json,
                        dataset_snapshot_json,
                        config_json,
                        str(status or "").strip().lower(),
                        idempotency_token,
                        str(approved_by or "").strip() or None,
                        str(rejected_by or "").strip() or None,
                        str(linked_job_id or "").strip() or None,
                        str(candidate_version_id or "").strip() or None,
                        str(error or "").strip() or None,
                        now,
                        now,
                        approved_at,
                        rejected_at,
                        started_at,
                        finished_at,
                    ),
                )
                conn.commit()
        except Exception:
            if idempotency_token:
                existing = self.get_training_update_proposal_by_idempotency_key(
                    idempotency_key=idempotency_token
                )
                if existing is not None:
                    return existing
            raise
        payload = self.get_training_update_proposal(proposal_id=proposal_id)
        if payload is None:
            raise RuntimeError(f"Failed to persist training update proposal: {proposal_id}")
        return payload

    def update_training_update_proposal(
        self,
        *,
        proposal_id: str,
        status: str | None = None,
        trigger_snapshot: dict[str, Any] | None = None,
        dataset_snapshot: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        approved_by: str | None = None,
        rejected_by: str | None = None,
        linked_job_id: str | None = None,
        candidate_version_id: str | None = None,
        error: str | None = None,
        approved_at: str | None = None,
        rejected_at: str | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
    ) -> dict[str, Any] | None:
        assignments: list[str] = []
        params: list[Any] = []
        if status is not None:
            assignments.append("status=?")
            params.append(str(status or "").strip().lower())
        if trigger_snapshot is not None:
            assignments.append("trigger_snapshot_json=?")
            params.append(json.dumps(trigger_snapshot, default=str))
        if dataset_snapshot is not None:
            assignments.append("dataset_snapshot_json=?")
            params.append(json.dumps(dataset_snapshot, default=str))
        if config is not None:
            assignments.append("config_json=?")
            params.append(json.dumps(config, default=str))
        if approved_by is not None:
            assignments.append("approved_by=?")
            params.append(str(approved_by or "").strip() or None)
        if rejected_by is not None:
            assignments.append("rejected_by=?")
            params.append(str(rejected_by or "").strip() or None)
        if linked_job_id is not None:
            assignments.append("linked_job_id=?")
            params.append(str(linked_job_id or "").strip() or None)
        if candidate_version_id is not None:
            assignments.append("candidate_version_id=?")
            params.append(str(candidate_version_id or "").strip() or None)
        if error is not None:
            assignments.append("error=?")
            params.append(str(error or "").strip() or None)
        if approved_at is not None:
            assignments.append("approved_at=?")
            params.append(approved_at)
        if rejected_at is not None:
            assignments.append("rejected_at=?")
            params.append(rejected_at)
        if started_at is not None:
            assignments.append("started_at=?")
            params.append(started_at)
        if finished_at is not None:
            assignments.append("finished_at=?")
            params.append(finished_at)
        if not assignments:
            return self.get_training_update_proposal(proposal_id=proposal_id)
        assignments.append("updated_at=?")
        params.append(datetime.utcnow().isoformat())
        params.append(str(proposal_id or "").strip())
        with self._lock, self._conn() as conn:
            self._execute(
                conn,
                """
                UPDATE training_update_proposals
                SET {assignments}
                WHERE proposal_id=?
                """
                .replace("{assignments}", ", ".join(assignments)),
                tuple(params),
            )
            conn.commit()
        return self.get_training_update_proposal(proposal_id=proposal_id)

    def get_training_update_proposal(self, *, proposal_id: str) -> dict[str, Any] | None:
        with self._lock, self._conn() as conn:
            row = self._execute(
                conn,
                """
                SELECT
                    proposal_id,
                    lineage_id,
                    trigger_reason,
                    trigger_snapshot_json,
                    dataset_snapshot_json,
                    config_json,
                    status,
                    idempotency_key,
                    approved_by,
                    rejected_by,
                    linked_job_id,
                    candidate_version_id,
                    error,
                    created_at,
                    updated_at,
                    approved_at,
                    rejected_at,
                    started_at,
                    finished_at
                FROM training_update_proposals
                WHERE proposal_id=?
                """,
                (str(proposal_id or "").strip(),),
            ).fetchone()
        if not row:
            return None
        trigger_snapshot = self._decode_json_payload(row[3])
        dataset_snapshot = self._decode_json_payload(row[4])
        config = self._decode_json_payload(row[5])
        return {
            "proposal_id": row[0],
            "lineage_id": row[1],
            "trigger_reason": row[2],
            "trigger_snapshot": trigger_snapshot if isinstance(trigger_snapshot, dict) else {},
            "dataset_snapshot": dataset_snapshot if isinstance(dataset_snapshot, dict) else {},
            "config": config if isinstance(config, dict) else {},
            "status": row[6],
            "idempotency_key": row[7],
            "approved_by": row[8],
            "rejected_by": row[9],
            "linked_job_id": row[10],
            "candidate_version_id": row[11],
            "error": row[12],
            "created_at": row[13],
            "updated_at": row[14],
            "approved_at": row[15],
            "rejected_at": row[16],
            "started_at": row[17],
            "finished_at": row[18],
        }

    def get_training_update_proposal_by_idempotency_key(
        self,
        *,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        token = str(idempotency_key or "").strip()
        if not token:
            return None
        with self._lock, self._conn() as conn:
            row = self._execute(
                conn,
                """
                SELECT proposal_id
                FROM training_update_proposals
                WHERE idempotency_key=?
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 1
                """,
                (token,),
            ).fetchone()
        if not row:
            return None
        return self.get_training_update_proposal(proposal_id=str(row[0] or "").strip())

    def list_training_update_proposals(
        self,
        *,
        owner_user_id: str | None = None,
        lineage_id: str | None = None,
        statuses: list[str] | None = None,
        include_shared: bool = True,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        lineage_token = str(lineage_id or "").strip()
        if lineage_token:
            clauses.append("p.lineage_id=?")
            params.append(lineage_token)
        owner_token = str(owner_user_id or "").strip()
        if owner_token:
            if include_shared:
                clauses.append("(l.owner_user_id=? OR l.scope='shared')")
            else:
                clauses.append("l.owner_user_id=?")
            params.append(owner_token)
        normalized_statuses = [
            str(item or "").strip().lower() for item in (statuses or []) if str(item or "").strip()
        ]
        if normalized_statuses:
            placeholders = ",".join(["?"] * len(normalized_statuses))
            clauses.append(f"LOWER(p.status) IN ({placeholders})")
            params.extend(normalized_statuses)
        where_clause = " AND ".join(clauses) if clauses else "1=1"
        params.append(max(1, int(limit)))
        with self._lock, self._conn() as conn:
            rows = self._execute(
                conn,
                """
                SELECT
                    p.proposal_id,
                    p.lineage_id,
                    p.trigger_reason,
                    p.trigger_snapshot_json,
                    p.dataset_snapshot_json,
                    p.config_json,
                    p.status,
                    p.idempotency_key,
                    p.approved_by,
                    p.rejected_by,
                    p.linked_job_id,
                    p.candidate_version_id,
                    p.error,
                    p.created_at,
                    p.updated_at,
                    p.approved_at,
                    p.rejected_at,
                    p.started_at,
                    p.finished_at
                FROM training_update_proposals p
                LEFT JOIN training_lineages l ON l.lineage_id = p.lineage_id
                WHERE {where_clause}
                ORDER BY p.updated_at DESC, p.created_at DESC
                LIMIT ?
                """.replace("{where_clause}", where_clause),
                tuple(params),
            ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            trigger_snapshot = self._decode_json_payload(row[3])
            dataset_snapshot = self._decode_json_payload(row[4])
            config = self._decode_json_payload(row[5])
            output.append(
                {
                    "proposal_id": row[0],
                    "lineage_id": row[1],
                    "trigger_reason": row[2],
                    "trigger_snapshot": trigger_snapshot if isinstance(trigger_snapshot, dict) else {},
                    "dataset_snapshot": dataset_snapshot if isinstance(dataset_snapshot, dict) else {},
                    "config": config if isinstance(config, dict) else {},
                    "status": row[6],
                    "idempotency_key": row[7],
                    "approved_by": row[8],
                    "rejected_by": row[9],
                    "linked_job_id": row[10],
                    "candidate_version_id": row[11],
                    "error": row[12],
                    "created_at": row[13],
                    "updated_at": row[14],
                    "approved_at": row[15],
                    "rejected_at": row[16],
                    "started_at": row[17],
                    "finished_at": row[18],
                }
            )
        return output

    def upsert_training_replay_items(
        self,
        *,
        lineage_id: str,
        items: list[dict[str, Any]],
        replace: bool = False,
    ) -> int:
        now = datetime.utcnow().isoformat()
        with self._lock, self._conn() as conn:
            if replace:
                self._execute(
                    conn,
                    "DELETE FROM training_replay_items WHERE lineage_id=?",
                    (str(lineage_id or "").strip(),),
                )
            upserted = 0
            for raw in items:
                file_id = str(raw.get("file_id") or "").strip()
                sample_id = str(raw.get("sample_id") or "").strip()
                if not file_id or not sample_id:
                    continue
                replay_item_id = (
                    str(raw.get("replay_item_id") or "").strip()
                    or f"{str(lineage_id or '').strip()}:{sample_id}:{uuid4().hex[:8]}"
                )
                try:
                    weight = float(raw.get("weight") if raw.get("weight") is not None else 1.0)
                except Exception:
                    weight = 1.0
                if weight < 0.0:
                    weight = 0.0
                pinned_value = raw.get("pinned")
                pinned = 1 if bool(pinned_value) else 0
                last_seen_at = str(raw.get("last_seen_at") or "").strip() or now
                self._execute(
                    conn,
                    """
                    INSERT INTO training_replay_items(
                        replay_item_id,
                        lineage_id,
                        file_id,
                        sample_id,
                        weight,
                        class_tag,
                        pinned,
                        last_seen_at,
                        created_at,
                        updated_at
                    )
                    VALUES(?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(lineage_id, file_id, sample_id) DO UPDATE SET
                        replay_item_id=excluded.replay_item_id,
                        weight=excluded.weight,
                        class_tag=excluded.class_tag,
                        pinned=excluded.pinned,
                        last_seen_at=excluded.last_seen_at,
                        updated_at=excluded.updated_at
                    """,
                    (
                        replay_item_id,
                        str(lineage_id or "").strip(),
                        file_id,
                        sample_id,
                        weight,
                        str(raw.get("class_tag") or "").strip() or None,
                        pinned,
                        last_seen_at,
                        now,
                        now,
                    ),
                )
                upserted += 1
            conn.commit()
        return upserted

    def list_training_replay_items(
        self,
        *,
        lineage_id: str,
        limit: int = 2000,
    ) -> list[dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = self._execute(
                conn,
                """
                SELECT
                    replay_item_id,
                    lineage_id,
                    file_id,
                    sample_id,
                    weight,
                    class_tag,
                    pinned,
                    last_seen_at,
                    created_at,
                    updated_at
                FROM training_replay_items
                WHERE lineage_id=?
                ORDER BY pinned DESC, weight DESC, updated_at DESC
                LIMIT ?
                """,
                (str(lineage_id or "").strip(), max(1, int(limit))),
            ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            output.append(
                {
                    "replay_item_id": row[0],
                    "lineage_id": row[1],
                    "file_id": row[2],
                    "sample_id": row[3],
                    "weight": float(row[4] or 0.0),
                    "class_tag": row[5],
                    "pinned": bool(int(row[6] or 0)),
                    "last_seen_at": row[7],
                    "created_at": row[8],
                    "updated_at": row[9],
                }
            )
        return output

    def create_training_merge_request(
        self,
        *,
        merge_id: str,
        source_lineage_id: str,
        target_lineage_id: str,
        candidate_version_id: str,
        requested_by: str,
        status: str,
        decision_by: str | None = None,
        notes: str | None = None,
        evaluation: dict[str, Any] | None = None,
        linked_proposal_id: str | None = None,
        error: str | None = None,
        decided_at: str | None = None,
        executed_at: str | None = None,
    ) -> dict[str, Any]:
        now = datetime.utcnow().isoformat()
        evaluation_json = json.dumps(evaluation, default=str) if evaluation is not None else None
        with self._lock, self._conn() as conn:
            self._execute(
                conn,
                """
                INSERT INTO training_merge_requests(
                    merge_id,
                    source_lineage_id,
                    target_lineage_id,
                    candidate_version_id,
                    requested_by,
                    status,
                    decision_by,
                    notes,
                    evaluation_json,
                    linked_proposal_id,
                    error,
                    created_at,
                    updated_at,
                    decided_at,
                    executed_at
                )
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(merge_id) DO UPDATE SET
                    source_lineage_id=excluded.source_lineage_id,
                    target_lineage_id=excluded.target_lineage_id,
                    candidate_version_id=excluded.candidate_version_id,
                    requested_by=excluded.requested_by,
                    status=excluded.status,
                    decision_by=COALESCE(excluded.decision_by, training_merge_requests.decision_by),
                    notes=COALESCE(excluded.notes, training_merge_requests.notes),
                    evaluation_json=COALESCE(excluded.evaluation_json, training_merge_requests.evaluation_json),
                    linked_proposal_id=COALESCE(excluded.linked_proposal_id, training_merge_requests.linked_proposal_id),
                    error=COALESCE(excluded.error, training_merge_requests.error),
                    updated_at=excluded.updated_at,
                    decided_at=COALESCE(excluded.decided_at, training_merge_requests.decided_at),
                    executed_at=COALESCE(excluded.executed_at, training_merge_requests.executed_at)
                """,
                (
                    str(merge_id or "").strip(),
                    str(source_lineage_id or "").strip(),
                    str(target_lineage_id or "").strip(),
                    str(candidate_version_id or "").strip(),
                    str(requested_by or "").strip(),
                    str(status or "").strip().lower(),
                    str(decision_by or "").strip() or None,
                    str(notes or "").strip() or None,
                    evaluation_json,
                    str(linked_proposal_id or "").strip() or None,
                    str(error or "").strip() or None,
                    now,
                    now,
                    decided_at,
                    executed_at,
                ),
            )
            conn.commit()
        payload = self.get_training_merge_request(merge_id=merge_id)
        if payload is None:
            raise RuntimeError(f"Failed to persist training merge request: {merge_id}")
        return payload

    def update_training_merge_request(
        self,
        *,
        merge_id: str,
        status: str | None = None,
        decision_by: str | None = None,
        notes: str | None = None,
        evaluation: dict[str, Any] | None = None,
        linked_proposal_id: str | None = None,
        error: str | None = None,
        decided_at: str | None = None,
        executed_at: str | None = None,
    ) -> dict[str, Any] | None:
        assignments: list[str] = []
        params: list[Any] = []
        if status is not None:
            assignments.append("status=?")
            params.append(str(status or "").strip().lower())
        if decision_by is not None:
            assignments.append("decision_by=?")
            params.append(str(decision_by or "").strip() or None)
        if notes is not None:
            assignments.append("notes=?")
            params.append(str(notes or "").strip() or None)
        if evaluation is not None:
            assignments.append("evaluation_json=?")
            params.append(json.dumps(evaluation, default=str))
        if linked_proposal_id is not None:
            assignments.append("linked_proposal_id=?")
            params.append(str(linked_proposal_id or "").strip() or None)
        if error is not None:
            assignments.append("error=?")
            params.append(str(error or "").strip() or None)
        if decided_at is not None:
            assignments.append("decided_at=?")
            params.append(decided_at)
        if executed_at is not None:
            assignments.append("executed_at=?")
            params.append(executed_at)
        if not assignments:
            return self.get_training_merge_request(merge_id=merge_id)
        assignments.append("updated_at=?")
        params.append(datetime.utcnow().isoformat())
        params.append(str(merge_id or "").strip())
        with self._lock, self._conn() as conn:
            self._execute(
                conn,
                """
                UPDATE training_merge_requests
                SET {assignments}
                WHERE merge_id=?
                """
                .replace("{assignments}", ", ".join(assignments)),
                tuple(params),
            )
            conn.commit()
        return self.get_training_merge_request(merge_id=merge_id)

    def get_training_merge_request(self, *, merge_id: str) -> dict[str, Any] | None:
        with self._lock, self._conn() as conn:
            row = self._execute(
                conn,
                """
                SELECT
                    merge_id,
                    source_lineage_id,
                    target_lineage_id,
                    candidate_version_id,
                    requested_by,
                    status,
                    decision_by,
                    notes,
                    evaluation_json,
                    linked_proposal_id,
                    error,
                    created_at,
                    updated_at,
                    decided_at,
                    executed_at
                FROM training_merge_requests
                WHERE merge_id=?
                """,
                (str(merge_id or "").strip(),),
            ).fetchone()
        if not row:
            return None
        evaluation = self._decode_json_payload(row[8])
        return {
            "merge_id": row[0],
            "source_lineage_id": row[1],
            "target_lineage_id": row[2],
            "candidate_version_id": row[3],
            "requested_by": row[4],
            "status": row[5],
            "decision_by": row[6],
            "notes": row[7],
            "evaluation": evaluation if isinstance(evaluation, dict) else {},
            "linked_proposal_id": row[9],
            "error": row[10],
            "created_at": row[11],
            "updated_at": row[12],
            "decided_at": row[13],
            "executed_at": row[14],
        }

    def list_training_merge_requests(
        self,
        *,
        owner_user_id: str | None = None,
        source_lineage_id: str | None = None,
        target_lineage_id: str | None = None,
        statuses: list[str] | None = None,
        include_shared: bool = True,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        source_token = str(source_lineage_id or "").strip()
        if source_token:
            clauses.append("m.source_lineage_id=?")
            params.append(source_token)
        target_token = str(target_lineage_id or "").strip()
        if target_token:
            clauses.append("m.target_lineage_id=?")
            params.append(target_token)
        owner_token = str(owner_user_id or "").strip()
        if owner_token:
            if include_shared:
                clauses.append(
                    "(src.owner_user_id=? OR dst.owner_user_id=? OR src.scope='shared' OR dst.scope='shared')"
                )
                params.extend([owner_token, owner_token])
            else:
                clauses.append("(src.owner_user_id=? OR dst.owner_user_id=?)")
                params.extend([owner_token, owner_token])
        normalized_statuses = [
            str(item or "").strip().lower() for item in (statuses or []) if str(item or "").strip()
        ]
        if normalized_statuses:
            placeholders = ",".join(["?"] * len(normalized_statuses))
            clauses.append(f"LOWER(m.status) IN ({placeholders})")
            params.extend(normalized_statuses)
        where_clause = " AND ".join(clauses) if clauses else "1=1"
        params.append(max(1, int(limit)))
        with self._lock, self._conn() as conn:
            rows = self._execute(
                conn,
                """
                SELECT
                    m.merge_id,
                    m.source_lineage_id,
                    m.target_lineage_id,
                    m.candidate_version_id,
                    m.requested_by,
                    m.status,
                    m.decision_by,
                    m.notes,
                    m.evaluation_json,
                    m.linked_proposal_id,
                    m.error,
                    m.created_at,
                    m.updated_at,
                    m.decided_at,
                    m.executed_at
                FROM training_merge_requests m
                LEFT JOIN training_lineages src ON src.lineage_id = m.source_lineage_id
                LEFT JOIN training_lineages dst ON dst.lineage_id = m.target_lineage_id
                WHERE {where_clause}
                ORDER BY m.updated_at DESC, m.created_at DESC
                LIMIT ?
                """.replace("{where_clause}", where_clause),
                tuple(params),
            ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            evaluation = self._decode_json_payload(row[8])
            output.append(
                {
                    "merge_id": row[0],
                    "source_lineage_id": row[1],
                    "target_lineage_id": row[2],
                    "candidate_version_id": row[3],
                    "requested_by": row[4],
                    "status": row[5],
                    "decision_by": row[6],
                    "notes": row[7],
                    "evaluation": evaluation if isinstance(evaluation, dict) else {},
                    "linked_proposal_id": row[9],
                    "error": row[10],
                    "created_at": row[11],
                    "updated_at": row[12],
                    "decided_at": row[13],
                    "executed_at": row[14],
                }
            )
        return output

    def acquire_training_scheduler_lease(
        self,
        *,
        lease_name: str,
        owner_id: str,
        ttl_seconds: int,
    ) -> bool:
        now_dt = datetime.utcnow()
        now = now_dt.isoformat()
        ttl = max(5, int(ttl_seconds))
        expires_at = (now_dt + timedelta(seconds=ttl)).isoformat()
        lease_token = str(lease_name or "").strip()
        owner_token = str(owner_id or "").strip()
        if not lease_token or not owner_token:
            return False

        def _parse_ts(raw: Any) -> datetime:
            token = str(raw or "").strip()
            if not token:
                return datetime.utcfromtimestamp(0)
            try:
                return datetime.fromisoformat(token.replace("Z", "+00:00"))
            except Exception:
                return datetime.utcfromtimestamp(0)

        with self._lock, self._conn() as conn:
            row = self._execute(
                conn,
                """
                SELECT owner_id, expires_at
                FROM training_scheduler_leases
                WHERE lease_name=?
                """,
                (lease_token,),
            ).fetchone()
            if row:
                existing_owner = str(row[0] or "").strip()
                existing_expires_at = _parse_ts(row[1])
                if existing_owner != owner_token and existing_expires_at > now_dt:
                    return False
                self._execute(
                    conn,
                    """
                    UPDATE training_scheduler_leases
                    SET owner_id=?, expires_at=?, updated_at=?
                    WHERE lease_name=?
                    """,
                    (owner_token, expires_at, now, lease_token),
                )
            else:
                self._execute(
                    conn,
                    """
                    INSERT INTO training_scheduler_leases(
                        lease_name,
                        owner_id,
                        expires_at,
                        updated_at
                    )
                    VALUES(?,?,?,?)
                    """,
                    (lease_token, owner_token, expires_at, now),
                )
            conn.commit()
        return True

    def release_training_scheduler_lease(self, *, lease_name: str, owner_id: str) -> None:
        with self._lock, self._conn() as conn:
            self._execute(
                conn,
                "DELETE FROM training_scheduler_leases WHERE lease_name=? AND owner_id=?",
                (str(lease_name or "").strip(), str(owner_id or "").strip()),
            )
            conn.commit()

    def create_or_resume_upload_session(
        self,
        *,
        session_id: str,
        user_id: str,
        fingerprint: str,
        original_name: str,
        content_type: str | None,
        size_bytes: int,
        temp_path: str,
        chunk_size_bytes: int,
    ) -> dict[str, Any]:
        """Create resumable upload session or return latest matching active/completed one.
        
        Parameters
        ----------
        session_id : str
            Upload session identifier.
        user_id : str
            User identifier.
        fingerprint : str
            Input argument.
        original_name : str
            Input argument.
        content_type : str | None
            Input argument.
        size_bytes : int
            Input argument.
        temp_path : str
            Input argument.
        chunk_size_bytes : int
            Input argument.
        
        Returns
        -------
        dict[str, Any]
            Result payload.
        """
        now = datetime.utcnow().isoformat()
        with self._lock, self._conn() as conn:
            existing = self._execute(
                conn,
                """
                SELECT session_id, user_id, fingerprint, original_name, content_type, size_bytes, temp_path,
                       bytes_received, chunk_size_bytes, status, file_id, sha256, error, created_at, updated_at
                FROM upload_sessions
                WHERE user_id=? AND fingerprint=? AND status IN ('active', 'completed')
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (user_id, fingerprint),
            ).fetchone()
            if existing:
                return {
                    "session_id": existing[0],
                    "user_id": existing[1],
                    "fingerprint": existing[2],
                    "original_name": existing[3],
                    "content_type": existing[4],
                    "size_bytes": int(existing[5]),
                    "temp_path": existing[6],
                    "bytes_received": int(existing[7]),
                    "chunk_size_bytes": int(existing[8]),
                    "status": existing[9],
                    "file_id": existing[10],
                    "sha256": existing[11],
                    "error": existing[12],
                    "created_at": existing[13],
                    "updated_at": existing[14],
                }

            self._execute(
                conn,
                """
                INSERT INTO upload_sessions(
                    session_id, user_id, fingerprint, original_name, content_type,
                    size_bytes, temp_path, bytes_received, chunk_size_bytes,
                    status, file_id, sha256, error, created_at, updated_at
                )
                VALUES(?,?,?,?,?,?,?,?,?,'active',NULL,NULL,NULL,?,?)
                """,
                (
                    session_id,
                    user_id,
                    fingerprint,
                    original_name,
                    content_type,
                    int(size_bytes),
                    temp_path,
                    0,
                    int(chunk_size_bytes),
                    now,
                    now,
                ),
            )
            conn.commit()

        return {
            "session_id": session_id,
            "user_id": user_id,
            "fingerprint": fingerprint,
            "original_name": original_name,
            "content_type": content_type,
            "size_bytes": int(size_bytes),
            "temp_path": temp_path,
            "bytes_received": 0,
            "chunk_size_bytes": int(chunk_size_bytes),
            "status": "active",
            "file_id": None,
            "sha256": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
        }

    def get_upload_session(self, *, session_id: str, user_id: str) -> dict[str, Any] | None:
        """Fetch resumable upload session state for one user session ID.
        
        Parameters
        ----------
        session_id : str
            Upload session identifier.
        user_id : str
            User identifier.
        
        Returns
        -------
        dict[str, Any] | None
            Result payload.
        """
        with self._lock, self._conn() as conn:
            row = self._execute(
                conn,
                """
                SELECT session_id, user_id, fingerprint, original_name, content_type, size_bytes, temp_path,
                       bytes_received, chunk_size_bytes, status, file_id, sha256, error, created_at, updated_at
                FROM upload_sessions
                WHERE session_id=? AND user_id=?
                """,
                (session_id, user_id),
            ).fetchone()
        if not row:
            return None
        return {
            "session_id": row[0],
            "user_id": row[1],
            "fingerprint": row[2],
            "original_name": row[3],
            "content_type": row[4],
            "size_bytes": int(row[5]),
            "temp_path": row[6],
            "bytes_received": int(row[7]),
            "chunk_size_bytes": int(row[8]),
            "status": row[9],
            "file_id": row[10],
            "sha256": row[11],
            "error": row[12],
            "created_at": row[13],
            "updated_at": row[14],
        }

    def update_upload_session_progress(
        self,
        *,
        session_id: str,
        user_id: str,
        bytes_received: int,
        status: str = "active",
        error: str | None = None,
    ) -> None:
        """Update bytes received and status for an in-progress upload session.
        
        Parameters
        ----------
        session_id : str
            Upload session identifier.
        user_id : str
            User identifier.
        bytes_received : int
            Input argument.
        status : str, optional
            Status filter or update value.
        error : str | None, optional
            Error message text.
        
        Returns
        -------
        None
            No return value.
        """
        now = datetime.utcnow().isoformat()
        with self._lock, self._conn() as conn:
            self._execute(
                conn,
                """
                UPDATE upload_sessions
                SET bytes_received=?, status=?, error=?, updated_at=?
                WHERE session_id=? AND user_id=?
                """,
                (
                    int(bytes_received),
                    status,
                    error,
                    now,
                    session_id,
                    user_id,
                ),
            )
            conn.commit()

    def complete_upload_session(
        self,
        *,
        session_id: str,
        user_id: str,
        file_id: str,
        sha256: str,
    ) -> None:
        """Mark resumable upload session complete and attach canonical file identifiers.
        
        Parameters
        ----------
        session_id : str
            Upload session identifier.
        user_id : str
            User identifier.
        file_id : str
            Upload file identifier.
        sha256 : str
            Input argument.
        
        Returns
        -------
        None
            No return value.
        """
        now = datetime.utcnow().isoformat()
        with self._lock, self._conn() as conn:
            self._execute(
                conn,
                """
                UPDATE upload_sessions
                SET status='completed', file_id=?, sha256=?, updated_at=?
                WHERE session_id=? AND user_id=?
                """,
                (file_id, sha256, now, session_id, user_id),
            )
            conn.commit()

    def fail_upload_session(self, *, session_id: str, user_id: str, error: str) -> None:
        """Mark resumable upload session as failed with an error message.
        
        Parameters
        ----------
        session_id : str
            Upload session identifier.
        user_id : str
            User identifier.
        error : str
            Error message text.
        
        Returns
        -------
        None
            No return value.
        """
        now = datetime.utcnow().isoformat()
        with self._lock, self._conn() as conn:
            self._execute(
                conn,
                """
                UPDATE upload_sessions
                SET status='failed', error=?, updated_at=?
                WHERE session_id=? AND user_id=?
                """,
                (error, now, session_id, user_id),
            )
            conn.commit()

    @staticmethod
    def _parse_iso_datetime(value: Any) -> datetime | None:
        raw = str(value or "").strip()
        if not raw:
            return None
        normalized = raw.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except Exception:
            return None

    def list_admin_run_tools(self, *, run_ids: list[str]) -> dict[str, list[str]]:
        """Return unique tool names observed for each run ID in admin views.
        
        Parameters
        ----------
        run_ids : list[str]
            Collection of identifier values.
        
        Returns
        -------
        dict[str, list[str]]
            Result payload.
        """
        normalized_run_ids = [
            str(run_id or "").strip() for run_id in run_ids if str(run_id or "").strip()
        ]
        if not normalized_run_ids:
            return {}
        placeholders = ",".join(["?"] * len(normalized_run_ids))
        with self._lock, self._conn() as conn:
            rows = self._execute(
                conn,
                """
                SELECT run_id, tool_name
                FROM resource_computations
                WHERE run_id IN ({placeholders})
                GROUP BY run_id, tool_name
                ORDER BY run_id, tool_name
                """.replace("{placeholders}", placeholders),
                tuple(normalized_run_ids),
            ).fetchall()
        by_run: dict[str, list[str]] = {}
        for run_id, tool_name in rows:
            run_key = str(run_id or "").strip()
            tool_value = str(tool_name or "").strip()
            if not run_key or not tool_value:
                continue
            by_run.setdefault(run_key, [])
            if tool_value not in by_run[run_key]:
                by_run[run_key].append(tool_value)
        return by_run

    def list_admin_runs(
        self,
        *,
        limit: int = 200,
        offset: int = 0,
        status: str | None = None,
        user_id: str | None = None,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        """List runs for admin dashboards with filters and derived duration/tool metadata.
        
        Parameters
        ----------
        limit : int, optional
            Maximum number of records to return.
        offset : int, optional
            Pagination offset.
        status : str | None, optional
            Status filter or update value.
        user_id : str | None, optional
            User identifier.
        query : str | None, optional
            Free-text query string.
        
        Returns
        -------
        list[dict[str, Any]]
            Computed result.
        """
        clauses: list[str] = ["1=1"]
        params: list[Any] = []

        status_value = str(status or "").strip().lower()
        if status_value:
            clauses.append("LOWER(status)=?")
            params.append(status_value)

        user_value = str(user_id or "").strip()
        if user_value:
            clauses.append("user_id=?")
            params.append(user_value)

        query_text = str(query or "").strip()
        if query_text:
            like = f"%{query_text}%"
            clauses.append(
                "(LOWER(COALESCE(goal, '')) LIKE LOWER(?) "
                "OR LOWER(COALESCE(run_id, '')) LIKE LOWER(?) "
                "OR LOWER(COALESCE(conversation_id, '')) LIKE LOWER(?) "
                "OR LOWER(COALESCE(user_id, '')) LIKE LOWER(?))"
            )
            params.extend([like, like, like, like])

        params.extend([max(1, int(limit)), max(0, int(offset))])
        with self._lock, self._conn() as conn:
            rows = self._execute(
                conn,
                """
                SELECT run_id, goal, status, created_at, updated_at, error, conversation_id, user_id
                FROM runs
                WHERE {where_clause}
                ORDER BY updated_at DESC
                LIMIT ?
                OFFSET ?
                """.replace("{where_clause}", " AND ".join(clauses)),
                tuple(params),
            ).fetchall()

        run_ids = [str(row[0] or "").strip() for row in rows]
        tool_map = self.list_admin_run_tools(run_ids=run_ids)
        output: list[dict[str, Any]] = []
        for row in rows:
            created_at = str(row[3] or "")
            updated_at = str(row[4] or "")
            created_dt = self._parse_iso_datetime(created_at)
            updated_dt = self._parse_iso_datetime(updated_at)
            duration_seconds: float | None = None
            if created_dt and updated_dt:
                duration_seconds = max(0.0, (updated_dt - created_dt).total_seconds())
            run_id = str(row[0] or "")
            output.append(
                {
                    "run_id": run_id,
                    "goal": str(row[1] or ""),
                    "status": str(row[2] or ""),
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "error": str(row[5] or "").strip() or None,
                    "conversation_id": str(row[6] or "").strip() or None,
                    "user_id": str(row[7] or "").strip() or None,
                    "duration_seconds": duration_seconds,
                    "tool_names": tool_map.get(run_id, []),
                }
            )
        return output

    def list_admin_users_summary(
        self,
        *,
        limit: int = 200,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        """Aggregate per-user counts and activity timestamps for admin analytics.
        
        Parameters
        ----------
        limit : int, optional
            Maximum number of records to return.
        query : str | None, optional
            Free-text query string.
        
        Returns
        -------
        list[dict[str, Any]]
            Computed result.
        """
        conversations: dict[str, int] = {}
        messages: dict[str, int] = {}
        runs: dict[str, dict[str, int]] = {}
        uploads: dict[str, dict[str, int]] = {}
        last_activity: dict[str, str] = {}

        with self._lock, self._conn() as conn:
            convo_rows = self._execute(
                conn,
                """
                SELECT user_id, COUNT(*)
                FROM conversations
                WHERE user_id IS NOT NULL AND user_id <> ''
                GROUP BY user_id
                """,
            ).fetchall()
            for user_id, count in convo_rows:
                key = str(user_id or "").strip()
                if key:
                    conversations[key] = int(count or 0)

            message_rows = self._execute(
                conn,
                """
                SELECT user_id, COUNT(*)
                FROM conversation_messages
                WHERE user_id IS NOT NULL AND user_id <> ''
                GROUP BY user_id
                """,
            ).fetchall()
            for user_id, count in message_rows:
                key = str(user_id or "").strip()
                if key:
                    messages[key] = int(count or 0)

            run_rows = self._execute(
                conn,
                """
                SELECT
                    user_id,
                    COUNT(*) AS runs_total,
                    SUM(CASE WHEN LOWER(status) IN ('pending', 'running') THEN 1 ELSE 0 END) AS runs_running,
                    SUM(CASE WHEN LOWER(status)='failed' THEN 1 ELSE 0 END) AS runs_failed,
                    SUM(CASE WHEN LOWER(status)='succeeded' THEN 1 ELSE 0 END) AS runs_succeeded
                FROM runs
                WHERE user_id IS NOT NULL AND user_id <> ''
                GROUP BY user_id
                """,
            ).fetchall()
            for row in run_rows:
                key = str(row[0] or "").strip()
                if not key:
                    continue
                runs[key] = {
                    "runs_total": int(row[1] or 0),
                    "runs_running": int(row[2] or 0),
                    "runs_failed": int(row[3] or 0),
                    "runs_succeeded": int(row[4] or 0),
                }

            upload_rows = self._execute(
                conn,
                """
                SELECT user_id, COUNT(*), COALESCE(SUM(size_bytes), 0)
                FROM uploads
                WHERE user_id IS NOT NULL AND user_id <> '' AND deleted_at IS NULL
                GROUP BY user_id
                """,
            ).fetchall()
            for user_id, count, size_bytes in upload_rows:
                key = str(user_id or "").strip()
                if not key:
                    continue
                uploads[key] = {
                    "uploads": int(count or 0),
                    "storage_bytes": int(size_bytes or 0),
                }

            activity_rows = self._execute(
                conn,
                """
                SELECT user_id, MAX(ts) AS last_activity
                FROM (
                    SELECT user_id, updated_at AS ts FROM runs
                    UNION ALL
                    SELECT user_id, updated_at AS ts FROM conversations
                    UNION ALL
                    SELECT user_id, COALESCE(updated_at, created_at) AS ts FROM uploads
                ) activity
                WHERE user_id IS NOT NULL AND user_id <> '' AND ts IS NOT NULL AND ts <> ''
                GROUP BY user_id
                """,
            ).fetchall()
            for user_id, ts in activity_rows:
                key = str(user_id or "").strip()
                stamp = str(ts or "").strip()
                if key and stamp:
                    last_activity[key] = stamp

        user_ids = (
            set(conversations) | set(messages) | set(runs) | set(uploads) | set(last_activity)
        )
        query_text = str(query or "").strip().lower()
        rows: list[dict[str, Any]] = []
        for user_id in user_ids:
            if query_text and query_text not in user_id.lower():
                continue
            row_runs = runs.get(user_id, {})
            row_uploads = uploads.get(user_id, {})
            rows.append(
                {
                    "user_id": user_id,
                    "conversations": int(conversations.get(user_id, 0)),
                    "messages": int(messages.get(user_id, 0)),
                    "runs_total": int(row_runs.get("runs_total", 0)),
                    "runs_running": int(row_runs.get("runs_running", 0)),
                    "runs_failed": int(row_runs.get("runs_failed", 0)),
                    "runs_succeeded": int(row_runs.get("runs_succeeded", 0)),
                    "uploads": int(row_uploads.get("uploads", 0)),
                    "storage_bytes": int(row_uploads.get("storage_bytes", 0)),
                    "last_activity_at": last_activity.get(user_id),
                }
            )

        def _sort_key(item: dict[str, Any]) -> tuple[int, int, int, str]:
            activity_value = str(item.get("last_activity_at") or "")
            activity_dt = self._parse_iso_datetime(activity_value)
            activity_sort = int(activity_dt.timestamp()) if activity_dt else 0
            return (
                activity_sort,
                int(item.get("runs_total") or 0),
                int(item.get("uploads") or 0),
                str(item.get("user_id") or ""),
            )

        rows.sort(key=_sort_key, reverse=True)
        return rows[: max(1, int(limit))]

    def list_admin_tool_usage(
        self,
        *,
        since_hours: int = 24 * 7,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """Summarize recent tool usage counts and success/failure splits for admins.
        
        Parameters
        ----------
        since_hours : int, optional
            Input argument.
        limit : int, optional
            Maximum number of records to return.
        
        Returns
        -------
        list[dict[str, Any]]
            Computed result.
        """
        threshold = (datetime.utcnow() - timedelta(hours=max(1, int(since_hours)))).isoformat()
        with self._lock, self._conn() as conn:
            rows = self._execute(
                conn,
                """
                SELECT
                    tool_name,
                    COUNT(*) AS total_count,
                    SUM(CASE WHEN LOWER(COALESCE(run_status, ''))='succeeded' THEN 1 ELSE 0 END) AS succeeded_count,
                    SUM(CASE WHEN LOWER(COALESCE(run_status, ''))='failed' THEN 1 ELSE 0 END) AS failed_count
                FROM resource_computations
                WHERE COALESCE(run_updated_at, updated_at, created_at) >= ?
                GROUP BY tool_name
                ORDER BY total_count DESC, tool_name ASC
                LIMIT ?
                """,
                (threshold, max(1, int(limit))),
            ).fetchall()
        return [
            {
                "tool_name": str(row[0] or "unknown"),
                "count": int(row[1] or 0),
                "succeeded": int(row[2] or 0),
                "failed": int(row[3] or 0),
            }
            for row in rows
        ]

    def list_admin_issues(self, *, limit: int = 25) -> list[dict[str, Any]]:
        """Collect recent operational issues (failed/stalled runs and failed uploads).
        
        Parameters
        ----------
        limit : int, optional
            Maximum number of records to return.
        
        Returns
        -------
        list[dict[str, Any]]
            Computed result.
        """
        now = datetime.utcnow()
        stalled_threshold = (now - timedelta(minutes=45)).isoformat()
        issues: list[dict[str, Any]] = []

        def _compact_issue_message(value: str, max_chars: int = 240) -> str:
            collapsed = re.sub(r"\s+", " ", str(value or "").strip())
            if not collapsed:
                return ""
            if len(collapsed) <= max_chars:
                return collapsed
            return collapsed[: max(1, max_chars - 1)].rstrip() + "…"

        with self._lock, self._conn() as conn:
            failed_runs = self._execute(
                conn,
                """
                SELECT run_id, user_id, conversation_id, goal, updated_at, error
                FROM runs
                WHERE LOWER(status)='failed'
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
            for run_id, user_id, conversation_id, goal, updated_at, error in failed_runs:
                raw_error_message = str(error or "").strip()
                display_error = raw_error_message
                if "[Errno 63]" in raw_error_message and "File name too long" in raw_error_message:
                    display_error = (
                        "Artifact filename exceeded the filesystem limit. "
                        "Use shorter output naming or sanitize generated filenames."
                    )
                issues.append(
                    {
                        "issue_type": "failed_run",
                        "severity": "high",
                        "user_id": str(user_id or "").strip() or None,
                        "run_id": str(run_id or "").strip() or None,
                        "upload_id": None,
                        "conversation_id": str(conversation_id or "").strip() or None,
                        "message": _compact_issue_message(
                            display_error or f"Run failed for goal: {goal}"
                        ),
                        "occurred_at": str(updated_at or datetime.utcnow().isoformat()),
                        "metadata": {
                            "goal": str(goal or "").strip(),
                            "full_error": raw_error_message or None,
                        },
                    }
                )

            failed_uploads = self._execute(
                conn,
                """
                SELECT session_id, user_id, original_name, updated_at, error
                FROM upload_sessions
                WHERE LOWER(status)='failed'
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
            for session_id, user_id, original_name, updated_at, error in failed_uploads:
                raw_upload_error = str(error or "").strip()
                issues.append(
                    {
                        "issue_type": "failed_upload_session",
                        "severity": "medium",
                        "user_id": str(user_id or "").strip() or None,
                        "run_id": None,
                        "upload_id": str(session_id or "").strip() or None,
                        "conversation_id": None,
                        "message": _compact_issue_message(
                            raw_upload_error or f"Upload failed for {str(original_name or 'file')}"
                        ),
                        "occurred_at": str(updated_at or datetime.utcnow().isoformat()),
                        "metadata": {
                            "file_name": str(original_name or "").strip(),
                            "full_error": raw_upload_error or None,
                        },
                    }
                )

            stalled_runs = self._execute(
                conn,
                """
                SELECT run_id, user_id, conversation_id, goal, status, updated_at
                FROM runs
                WHERE LOWER(status) IN ('pending', 'running') AND updated_at <= ?
                ORDER BY updated_at ASC
                LIMIT ?
                """,
                (stalled_threshold, max(1, int(limit))),
            ).fetchall()
            for run_id, user_id, conversation_id, goal, status, updated_at in stalled_runs:
                issues.append(
                    {
                        "issue_type": "stalled_run",
                        "severity": "medium",
                        "user_id": str(user_id or "").strip() or None,
                        "run_id": str(run_id or "").strip() or None,
                        "upload_id": None,
                        "conversation_id": str(conversation_id or "").strip() or None,
                        "message": f"Run remains {status} longer than expected.",
                        "occurred_at": str(updated_at or datetime.utcnow().isoformat()),
                        "metadata": {
                            "goal": str(goal or "").strip(),
                            "status": str(status or "").strip(),
                        },
                    }
                )

        def _issue_sort_key(item: dict[str, Any]) -> float:
            dt = self._parse_iso_datetime(item.get("occurred_at"))
            return dt.timestamp() if dt else 0.0

        issues.sort(key=_issue_sort_key, reverse=True)
        return issues[: max(1, int(limit))]

    def admin_usage_timeseries_24h(self) -> list[dict[str, Any]]:
        """Build hourly platform usage buckets for the trailing 24 hours.
        
        Returns
        -------
        list[dict[str, Any]]
            Computed result.
        """
        now = datetime.utcnow()
        start = (now - timedelta(hours=23)).replace(minute=0, second=0, microsecond=0)
        buckets: dict[str, dict[str, Any]] = {}
        for index in range(24):
            bucket_start = start + timedelta(hours=index)
            key = bucket_start.isoformat()
            buckets[key] = {
                "bucket_start": key,
                "runs_total": 0,
                "runs_succeeded": 0,
                "runs_failed": 0,
                "uploads": 0,
                "new_users": 0,
            }

        threshold_iso = start.isoformat()
        with self._lock, self._conn() as conn:
            run_rows = self._execute(
                conn,
                """
                SELECT updated_at, status
                FROM runs
                WHERE updated_at >= ?
                """,
                (threshold_iso,),
            ).fetchall()
            for updated_at, status in run_rows:
                dt = self._parse_iso_datetime(updated_at)
                if not dt:
                    continue
                bucket_key = dt.replace(minute=0, second=0, microsecond=0).isoformat()
                bucket = buckets.get(bucket_key)
                if not bucket:
                    continue
                bucket["runs_total"] += 1
                normalized_status = str(status or "").strip().lower()
                if normalized_status == "succeeded":
                    bucket["runs_succeeded"] += 1
                elif normalized_status == "failed":
                    bucket["runs_failed"] += 1

            upload_rows = self._execute(
                conn,
                """
                SELECT created_at
                FROM uploads
                WHERE created_at >= ? AND deleted_at IS NULL
                """,
                (threshold_iso,),
            ).fetchall()
            for (created_at,) in upload_rows:
                dt = self._parse_iso_datetime(created_at)
                if not dt:
                    continue
                bucket_key = dt.replace(minute=0, second=0, microsecond=0).isoformat()
                bucket = buckets.get(bucket_key)
                if bucket:
                    bucket["uploads"] += 1

            new_user_rows = self._execute(
                conn,
                """
                SELECT user_id, MIN(ts) AS first_seen
                FROM (
                    SELECT user_id, created_at AS ts FROM runs
                    UNION ALL
                    SELECT user_id, created_at AS ts FROM conversations
                    UNION ALL
                    SELECT user_id, created_at AS ts FROM uploads
                ) activity
                WHERE user_id IS NOT NULL AND user_id <> ''
                GROUP BY user_id
                HAVING MIN(ts) >= ?
                """,
                (threshold_iso,),
            ).fetchall()
            for _user_id, first_seen in new_user_rows:
                dt = self._parse_iso_datetime(first_seen)
                if not dt:
                    continue
                bucket_key = dt.replace(minute=0, second=0, microsecond=0).isoformat()
                bucket = buckets.get(bucket_key)
                if bucket:
                    bucket["new_users"] += 1

        ordered_keys = sorted(buckets.keys())
        return [buckets[key] for key in ordered_keys]

    def admin_overview(self, *, top_user_limit: int = 8, issue_limit: int = 12) -> dict[str, Any]:
        """Return top-level admin KPI snapshot with usage, users, tools, and issues.
        
        Parameters
        ----------
        top_user_limit : int, optional
            Input argument.
        issue_limit : int, optional
            Input argument.
        
        Returns
        -------
        dict[str, Any]
            Result payload.
        """
        now = datetime.utcnow()
        threshold_24h_dt = now - timedelta(hours=24)
        threshold_24h = threshold_24h_dt.isoformat()
        threshold_24h_ms = int(threshold_24h_dt.timestamp() * 1000)
        with self._lock, self._conn() as conn:
            total_runs = int(
                (self._execute(conn, "SELECT COUNT(*) FROM runs").fetchone() or [0])[0] or 0
            )
            running_runs = int(
                (
                    self._execute(
                        conn,
                        "SELECT COUNT(*) FROM runs WHERE LOWER(status) IN ('pending', 'running')",
                    ).fetchone()
                    or [0]
                )[0]
                or 0
            )
            runs_24h = int(
                (
                    self._execute(
                        conn,
                        "SELECT COUNT(*) FROM runs WHERE updated_at >= ?",
                        (threshold_24h,),
                    ).fetchone()
                    or [0]
                )[0]
                or 0
            )
            succeeded_24h = int(
                (
                    self._execute(
                        conn,
                        "SELECT COUNT(*) FROM runs WHERE LOWER(status)='succeeded' AND updated_at >= ?",
                        (threshold_24h,),
                    ).fetchone()
                    or [0]
                )[0]
                or 0
            )
            failed_24h = int(
                (
                    self._execute(
                        conn,
                        "SELECT COUNT(*) FROM runs WHERE LOWER(status)='failed' AND updated_at >= ?",
                        (threshold_24h,),
                    ).fetchone()
                    or [0]
                )[0]
                or 0
            )
            total_conversations = int(
                (self._execute(conn, "SELECT COUNT(*) FROM conversations").fetchone() or [0])[0]
                or 0
            )
            conversations_started_24h = int(
                (
                    self._execute(
                        conn,
                        "SELECT COUNT(*) FROM conversations WHERE created_at >= ?",
                        (threshold_24h,),
                    ).fetchone()
                    or [0]
                )[0]
                or 0
            )
            total_messages = int(
                (
                    self._execute(conn, "SELECT COUNT(*) FROM conversation_messages").fetchone()
                    or [0]
                )[0]
                or 0
            )
            message_volume_row = (
                self._execute(
                    conn,
                    """
                    SELECT
                        COUNT(*) AS total_count,
                        SUM(CASE WHEN LOWER(COALESCE(role, ''))='user' THEN 1 ELSE 0 END) AS user_count,
                        SUM(CASE WHEN LOWER(COALESCE(role, ''))='assistant' THEN 1 ELSE 0 END) AS assistant_count
                    FROM conversation_messages
                    WHERE COALESCE(created_at_ms, 0) >= ?
                    """,
                    (threshold_24h_ms,),
                ).fetchone()
                or [0, 0, 0]
            )
            messages_last_24h = int(message_volume_row[0] or 0)
            user_messages_last_24h = int(message_volume_row[1] or 0)
            assistant_messages_last_24h = int(message_volume_row[2] or 0)
            total_uploads = int(
                (
                    self._execute(
                        conn,
                        "SELECT COUNT(*) FROM uploads WHERE deleted_at IS NULL",
                    ).fetchone()
                    or [0]
                )[0]
                or 0
            )
            soft_deleted_uploads = int(
                (
                    self._execute(
                        conn,
                        "SELECT COUNT(*) FROM uploads WHERE deleted_at IS NOT NULL",
                    ).fetchone()
                    or [0]
                )[0]
                or 0
            )
            total_storage_bytes = int(
                (
                    self._execute(
                        conn,
                        "SELECT COALESCE(SUM(size_bytes), 0) FROM uploads WHERE deleted_at IS NULL",
                    ).fetchone()
                    or [0]
                )[0]
                or 0
            )
            total_users = int(
                (
                    self._execute(
                        conn,
                        """
                        SELECT COUNT(*)
                        FROM (
                            SELECT user_id FROM runs WHERE user_id IS NOT NULL AND user_id <> ''
                            UNION
                            SELECT user_id FROM conversations WHERE user_id IS NOT NULL AND user_id <> ''
                            UNION
                            SELECT user_id FROM uploads WHERE user_id IS NOT NULL AND user_id <> ''
                        ) users
                        """,
                    ).fetchone()
                    or [0]
                )[0]
                or 0
            )
            active_users_24h = int(
                (
                    self._execute(
                        conn,
                        """
                        SELECT COUNT(*)
                        FROM (
                            SELECT user_id FROM runs
                            WHERE user_id IS NOT NULL AND user_id <> '' AND updated_at >= ?
                            UNION
                            SELECT user_id FROM conversations
                            WHERE user_id IS NOT NULL AND user_id <> '' AND updated_at >= ?
                            UNION
                            SELECT user_id FROM uploads
                            WHERE user_id IS NOT NULL AND user_id <> '' AND COALESCE(updated_at, created_at) >= ?
                        ) users
                        """,
                        (threshold_24h, threshold_24h, threshold_24h),
                    ).fetchone()
                    or [0]
                )[0]
                or 0
            )

        success_rate_24h = float((succeeded_24h / runs_24h) * 100.0) if runs_24h > 0 else 0.0
        avg_messages = (
            float(total_messages) / float(total_conversations) if total_conversations > 0 else 0.0
        )
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "kpis": {
                "total_users": total_users,
                "active_users_24h": active_users_24h,
                "total_conversations": total_conversations,
                "conversations_started_24h": conversations_started_24h,
                "total_messages": total_messages,
                "messages_last_24h": messages_last_24h,
                "user_messages_last_24h": user_messages_last_24h,
                "assistant_messages_last_24h": assistant_messages_last_24h,
                "total_runs": total_runs,
                "runs_last_24h": runs_24h,
                "success_rate_last_24h": round(success_rate_24h, 2),
                "running_runs": running_runs,
                "failed_runs_24h": failed_24h,
                "total_uploads": total_uploads,
                "soft_deleted_uploads": soft_deleted_uploads,
                "total_storage_bytes": total_storage_bytes,
                "avg_messages_per_conversation": round(avg_messages, 2),
            },
            "usage_last_24h": self.admin_usage_timeseries_24h(),
            "tool_usage_7d": self.list_admin_tool_usage(since_hours=24 * 7, limit=8),
            "top_users": self.list_admin_users_summary(limit=max(1, int(top_user_limit))),
            "recent_issues": self.list_admin_issues(limit=max(1, int(issue_limit))),
        }

    def admin_cancel_run(self, *, run_id: str, reason: str | None = None) -> dict[str, Any] | None:
        """Cancel a pending/running run and return transition metadata.
        
        Parameters
        ----------
        run_id : str
            Workflow run identifier.
        reason : str | None, optional
            Input argument.
        
        Returns
        -------
        dict[str, Any] | None
            Result payload.
        """
        normalized_run_id = str(run_id or "").strip()
        if not normalized_run_id:
            return None
        now = datetime.utcnow().isoformat()
        with self._lock, self._conn() as conn:
            row = self._execute(
                conn,
                "SELECT status FROM runs WHERE run_id=?",
                (normalized_run_id,),
            ).fetchone()
            if not row:
                return None
            previous_status = str(row[0] or "").strip().lower()
            if previous_status in {"pending", "running"}:
                self._execute(
                    conn,
                    "UPDATE runs SET status=?, updated_at=?, error=? WHERE run_id=?",
                    (
                        "canceled",
                        now,
                        str(reason or "").strip() or "Canceled by admin",
                        normalized_run_id,
                    ),
                )
                conn.commit()
                return {
                    "run_id": normalized_run_id,
                    "previous_status": previous_status,
                    "status": "canceled",
                    "updated": True,
                }
            return {
                "run_id": normalized_run_id,
                "previous_status": previous_status,
                "status": previous_status,
                "updated": False,
            }

    def admin_delete_conversation_for_user(self, *, conversation_id: str, user_id: str) -> bool:
        """Delete a specific user's conversation and return whether deletion occurred.
        
        Parameters
        ----------
        conversation_id : str
            Conversation identifier.
        user_id : str
            User identifier.
        
        Returns
        -------
        bool
            Computed result.
        """
        normalized_conversation_id = str(conversation_id or "").strip()
        normalized_user_id = str(user_id or "").strip()
        if not normalized_conversation_id or not normalized_user_id:
            return False
        current = self.get_conversation(
            conversation_id=normalized_conversation_id,
            user_id=normalized_user_id,
        )
        if current is None:
            return False
        self.delete_conversation(
            conversation_id=normalized_conversation_id,
            user_id=normalized_user_id,
        )
        return True


def _tokenize_message_search_terms(query: str, max_terms: int = 12) -> list[str]:
    raw = str(query or "").strip().lower()
    if not raw:
        return []
    quoted = [
        part.strip()
        for group in re.findall(r'"([^"]+)"|\'([^\']+)\'', raw)
        for part in group
        if part and part.strip()
    ]
    candidates = quoted + re.findall(r"[a-z0-9_./:-]{2,}", raw)
    tokens: list[str] = []
    seen: set[str] = set()
    for token in candidates:
        normalized = token.strip("._:-")
        if not normalized:
            continue
        if len(normalized) <= 1:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        tokens.append(normalized)
        if len(tokens) >= max(1, int(max_terms)):
            break
    return tokens


def _plan_to_dict(plan: WorkflowPlan | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    return {
        "goal": plan.goal,
        "steps": [
            {
                "id": step.id,
                "tool_name": step.tool_name,
                "arguments": step.arguments,
                "description": step.description,
                "timeout_seconds": step.timeout_seconds,
                "retries": step.retries,
            }
            for step in plan.steps
        ],
    }


def _plan_from_json(plan_json: str | None) -> WorkflowPlan | None:
    if not plan_json:
        return None
    data = json.loads(plan_json)
    return WorkflowPlan.from_dict(data)
