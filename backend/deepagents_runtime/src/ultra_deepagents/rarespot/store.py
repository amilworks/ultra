from __future__ import annotations

from datetime import UTC, datetime
import json
from typing import Any

from ultra_deepagents.config import RuntimeSettings


class EventPublishError(RuntimeError):
    pass


class NATSRunEventStore:
    def __init__(self, js: Any, settings: RuntimeSettings) -> None:
        self.js = js
        self.settings = settings
        self._sequences: dict[str, int] = {}
        self._artifact_indexes: dict[str, int] = {}

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def get_run(self, run_id: str) -> dict[str, Any]:
        return {"run_id": run_id, "status": "queued", "metadata": {}}

    async def update_run_status(
        self,
        run_id: str,
        status: str,
        *,
        response_text: str = "",
        error: str = "",
    ) -> None:
        _ = run_id, status, response_text, error
        return None

    async def append_run_event(
        self,
        run_id: str,
        thread_id: str,
        event_kind: str,
        message: str,
        payload: dict[str, Any] | None = None,
        *,
        event_id: str | None = None,
    ) -> None:
        event_payload = dict(payload or {})
        if event_kind == "run.completed" and not event_payload.get("response_text"):
            event_payload["response_text"] = _rarespot_response_text(event_payload, message)
        sequence = self._next_sequence(run_id)
        event = {
            "event_id": event_id or f"evt_{run_id}_rarespot_{sequence:06d}",
            "sequence": sequence,
            "run_id": run_id,
            "thread_id": thread_id,
            "event_kind": event_kind,
            "event_type": _event_type(event_kind),
            "node_name": "rarespot-worker",
            "agent_role": "rarespot",
            "level": _event_level(event_kind),
            "ts": datetime.now(UTC).isoformat(),
            "message": message,
            "payload": event_payload,
        }
        await self._publish_event(event)

    async def create_artifact(self, artifact: dict[str, Any]) -> dict[str, Any]:
        created = dict(artifact)
        run_id = str(created["run_id"])
        thread_id = str(created.get("thread_id") or "")
        artifact_id = created.get("artifact_id")
        if not artifact_id:
            artifact_id = f"artifact_{run_id}_rarespot_{self._next_artifact_index(run_id):06d}"
        created["artifact_id"] = str(artifact_id)
        created["tool_name"] = created.get("tool_name") or "rarespot_ecology_inference"
        created["kind"] = created.get("kind") or "artifact"
        if created.get("relative_path") is None and created.get("path") is not None:
            created["relative_path"] = created["path"]
        await self.append_run_event(
            run_id,
            thread_id,
            "artifact.created",
            str(created.get("title") or "RareSpot artifact created."),
            created,
        )
        return created

    def _next_sequence(self, run_id: str) -> int:
        sequence = self._sequences.get(run_id, 0) + 1
        self._sequences[run_id] = sequence
        return sequence

    def _next_artifact_index(self, run_id: str) -> int:
        index = self._artifact_indexes.get(run_id, 0) + 1
        self._artifact_indexes[run_id] = index
        return index

    async def _publish_event(self, event: dict[str, Any]) -> None:
        headers = {}
        message_id = _event_message_id(event)
        if message_id:
            headers["Nats-Msg-Id"] = message_id
        try:
            await self.js.publish(
                self.settings.rarespot_nats_events_subject,
                json.dumps(event, default=str).encode("utf-8"),
                headers=headers or None,
            )
        except Exception as exc:
            raise EventPublishError(str(exc)) from exc


def _event_type(event_kind: str) -> str:
    if event_kind.startswith("run."):
        return "run"
    if event_kind.startswith("artifact."):
        return "artifact"
    if event_kind.startswith("tool."):
        return "tool"
    return "worker"


def _event_level(event_kind: str) -> str:
    if event_kind == "run.failed":
        return "error"
    return "info"


def _event_message_id(event: dict[str, Any]) -> str:
    event_id = str(event.get("event_id") or "").strip()
    if event_id:
        return f"event:{event_id}"
    run_id = str(event.get("run_id") or "").strip()
    try:
        sequence = int(event.get("sequence"))
    except (TypeError, ValueError):
        sequence = 0
    if run_id and sequence > 0:
        return f"event:{run_id}:{sequence}"
    return ""


def _rarespot_response_text(payload: dict[str, Any], fallback: str) -> str:
    counts = payload.get("counts_by_class")
    artifact_ids = payload.get("artifact_ids")
    if isinstance(counts, dict):
        count_parts = [f"{name}: {value}" for name, value in sorted(counts.items())]
        count_text = ", ".join(count_parts) if count_parts else "no detections"
        artifact_count = len(artifact_ids) if isinstance(artifact_ids, list) else 0
        return (
            "RareSpot ecology inference completed. "
            f"Detection counts by class: {count_text}. "
            f"Saved {artifact_count} durable output{'s' if artifact_count != 1 else ''}."
        )
    return fallback


class PostgresRunStore:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self._conn: Any | None = None

    async def connect(self) -> None:
        import psycopg

        self._conn = await psycopg.AsyncConnection.connect(self.database_url)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> Any:
        if self._conn is None:
            raise RuntimeError("PostgresRunStore is not connected.")
        return self._conn

    async def get_run(self, run_id: str) -> dict[str, Any]:
        async with self.conn.cursor() as cursor:
            await cursor.execute("SELECT run_id, thread_id, status, metadata FROM control_runs WHERE run_id = %s", (run_id,))
            row = await cursor.fetchone()
        if row is None:
            raise KeyError(run_id)
        return {"run_id": row[0], "thread_id": row[1], "status": row[2], "metadata": row[3] or {}}

    async def update_run_status(
        self,
        run_id: str,
        status: str,
        *,
        response_text: str = "",
        error: str = "",
    ) -> None:
        async with self.conn.cursor() as cursor:
            await cursor.execute(
                """
                UPDATE control_runs
                SET status = %s,
                    response_text = NULLIF(%s, ''),
                    error = NULLIF(%s, ''),
                    updated_at = now(),
                    started_at = CASE WHEN %s = 'running' AND started_at IS NULL THEN now() ELSE started_at END,
                    completed_at = CASE WHEN %s IN ('succeeded', 'failed', 'canceled') THEN now() ELSE completed_at END
                WHERE run_id = %s
                """,
                (status, response_text, error, status, status, run_id),
            )
        await self.conn.commit()

    async def append_run_event(
        self,
        run_id: str,
        thread_id: str,
        event_kind: str,
        message: str,
        payload: dict[str, Any] | None = None,
        *,
        event_id: str | None = None,
    ) -> None:
        async with self.conn.cursor() as cursor:
            await cursor.execute(
                "SELECT COALESCE(MAX(sequence_number), 0) + 1 FROM control_run_events WHERE run_id = %s",
                (run_id,),
            )
            sequence = int((await cursor.fetchone())[0])
            await cursor.execute(
                """
                INSERT INTO control_run_events
                    (event_id, sequence_number, run_id, thread_id, event_kind, ts, message, payload)
                VALUES
                    (%s, %s, %s, %s, %s, now(), %s, %s::jsonb)
                ON CONFLICT (event_id) DO NOTHING
                """,
                (
                    event_id or f"event_{run_id}_{sequence}",
                    sequence,
                    run_id,
                    thread_id or None,
                    event_kind,
                    message,
                    json.dumps(payload or {}),
                ),
            )
        await self.conn.commit()

    async def create_artifact(self, artifact: dict[str, Any]) -> dict[str, Any]:
        artifact_id = artifact.get("artifact_id") or f"artifact_{artifact['run_id']}_{abs(hash((artifact['path'], artifact.get('kind'))))}"
        async with self.conn.cursor() as cursor:
            await cursor.execute(
                """
                INSERT INTO control_artifacts
                    (artifact_id, run_id, thread_id, kind, path, source_path, preview_path, title,
                     result_group_id, mime_type, size_bytes, sha256, storage_uri, tool_name, category,
                     created_at, updated_at, metadata)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now(), now(), %s::jsonb)
                RETURNING artifact_id
                """,
                (
                    artifact_id,
                    artifact["run_id"],
                    artifact.get("thread_id") or None,
                    artifact["kind"],
                    artifact.get("path") or None,
                    artifact.get("source_path") or None,
                    artifact.get("preview_path") or None,
                    artifact.get("title") or None,
                    artifact.get("result_group_id") or None,
                    artifact.get("mime_type") or None,
                    artifact.get("size_bytes") or None,
                    artifact.get("sha256") or None,
                    artifact.get("storage_uri") or None,
                    artifact.get("tool_name") or "rarespot_ecology_inference",
                    artifact.get("category") or None,
                    json.dumps(artifact.get("metadata") or {}),
                ),
            )
            artifact["artifact_id"] = (await cursor.fetchone())[0]
        await self.conn.commit()
        return artifact
