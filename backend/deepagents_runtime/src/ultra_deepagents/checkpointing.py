"""Durable LangGraph checkpointing for resumable autonomous runs.

A long autonomous run that loses its worker/replica mid-trajectory must resume
from its last completed super-step instead of re-executing every model call and
tool invocation from scratch. LangGraph already checkpoints graph state per
super-step when a checkpointer is configured; the gap is durability across a
process restart.

The official ``langgraph-checkpoint-postgres`` saver is incompatible with the
LangGraph 1.x line used here (it pins ``langgraph-checkpoint`` 2.x while 1.x
requires 4.x). Rather than reimplement the full, correctness-critical checkpoint
protocol, this module reuses LangGraph's own :class:`InMemorySaver` logic and
adds durability around it: after every checkpoint write it mirrors the run's
thread slice to a pluggable state store, and on resume it hydrates that slice
back into memory before the graph runs. The state store is Postgres in
production and an in-process dict in tests.

Each run is checkpointed under ``thread_id == run_id`` (one autonomous
trajectory per run), so a second run in the same conversation never resumes a
previous run's graph state.
"""

from __future__ import annotations

import asyncio
import logging
import pickle
from typing import Any, Protocol

from langgraph.checkpoint.memory import InMemorySaver

logger = logging.getLogger(__name__)

# Bumped if the persisted slice layout changes so stale blobs are ignored
# instead of mis-deserialized.
_STATE_BLOB_VERSION = 1

_CHECKPOINT_TABLE = "deepagents_checkpoint_threads"


class CheckpointStateStore(Protocol):
    """Durable key/value home for a run's serialized checkpoint slice."""

    async def load(self, thread_id: str) -> bytes | None: ...

    async def save(self, thread_id: str, blob: bytes) -> None: ...

    async def close(self) -> None: ...


class InMemoryCheckpointStateStore:
    """Process-local state store. Two checkpointers sharing one instance model
    two worker processes sharing the same durable backend."""

    def __init__(self) -> None:
        self._blobs: dict[str, bytes] = {}

    async def load(self, thread_id: str) -> bytes | None:
        return self._blobs.get(thread_id)

    async def save(self, thread_id: str, blob: bytes) -> None:
        self._blobs[thread_id] = blob

    async def close(self) -> None:
        return None


class PostgresCheckpointStateStore:
    """Stores each run's checkpoint slice as a single row in the control-plane
    Postgres. One row per ``thread_id`` (== run_id); upserted on every write."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn: Any | None = None
        self._lock = asyncio.Lock()
        self._ready = False

    async def _connection(self) -> Any:
        import psycopg

        if self._conn is not None and not self._conn.closed:
            return self._conn
        self._conn = await psycopg.AsyncConnection.connect(self._dsn, autocommit=True)
        return self._conn

    async def ensure_schema(self) -> None:
        async with self._lock:
            if self._ready:
                return
            conn = await self._connection()
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_CHECKPOINT_TABLE} (
                    thread_id text PRIMARY KEY,
                    state bytea NOT NULL,
                    updated_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            self._ready = True

    async def load(self, thread_id: str) -> bytes | None:
        await self.ensure_schema()
        async with self._lock:
            conn = await self._connection()
            cursor = await conn.execute(
                f"SELECT state FROM {_CHECKPOINT_TABLE} WHERE thread_id = %s",
                (thread_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        value = row[0]
        return bytes(value) if value is not None else None

    async def save(self, thread_id: str, blob: bytes) -> None:
        await self.ensure_schema()
        async with self._lock:
            conn = await self._connection()
            await conn.execute(
                f"""
                INSERT INTO {_CHECKPOINT_TABLE} (thread_id, state, updated_at)
                VALUES (%s, %s, now())
                ON CONFLICT (thread_id)
                DO UPDATE SET state = EXCLUDED.state, updated_at = now()
                """,
                (thread_id, blob),
            )

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None and not self._conn.closed:
                await self._conn.close()
            self._conn = None


class DurableCheckpointer(InMemorySaver):
    """:class:`InMemorySaver` that mirrors each thread's checkpoint slice to a
    durable :class:`CheckpointStateStore`, enabling cross-process resume.

    All checkpoint correctness (blob versioning, pending writes, channel
    reassembly) is inherited unchanged from LangGraph; this subclass only adds
    persist-after-write and hydrate-before-read.
    """

    def __init__(self, store: CheckpointStateStore, *, serde: Any | None = None) -> None:
        super().__init__(serde=serde)
        # Guard against a future LangGraph layout change silently breaking
        # persistence: we depend on these in-memory attributes existing.
        for attribute in ("storage", "blobs", "writes"):
            if not hasattr(self, attribute):
                raise RuntimeError(
                    f"InMemorySaver no longer exposes '{attribute}'; durable "
                    "checkpointing must be updated for this LangGraph version."
                )
        self._store = store
        self._hydrated: set[str] = set()

    async def hydrate(self, thread_id: str) -> bool:
        """Load a run's prior checkpoint slice into memory. Returns True when
        durable state was found and restored. Safe to call repeatedly."""
        if thread_id in self._hydrated:
            return False
        self._hydrated.add(thread_id)
        try:
            blob = await self._store.load(thread_id)
        except Exception:
            logger.warning(
                "Checkpoint hydrate failed; treating run as fresh.",
                extra={"thread_id": thread_id},
                exc_info=True,
            )
            return False
        if not blob:
            return False
        try:
            slice_ = _decode_thread_slice(blob)
        except Exception:
            logger.warning(
                "Checkpoint slice could not be decoded; treating run as fresh.",
                extra={"thread_id": thread_id},
                exc_info=True,
            )
            return False
        self._restore_thread_slice(thread_id, slice_)
        return True

    async def _persist(self, thread_id: str) -> None:
        try:
            blob = _encode_thread_slice(self._collect_thread_slice(thread_id))
            await self._store.save(thread_id, blob)
        except Exception:
            # Never let a durability hiccup crash an in-flight run; the run
            # still streams and completes, it just may not resume on restart.
            logger.warning(
                "Checkpoint persist failed; resume may be unavailable for this run.",
                extra={"thread_id": thread_id},
                exc_info=True,
            )

    async def aput(self, config: Any, checkpoint: Any, metadata: Any, new_versions: Any) -> Any:
        result = await super().aput(config, checkpoint, metadata, new_versions)
        await self._persist(_thread_id_from_config(config))
        return result

    async def aput_writes(
        self, config: Any, writes: Any, task_id: str, task_path: str = ""
    ) -> None:
        await super().aput_writes(config, writes, task_id, task_path)
        await self._persist(_thread_id_from_config(config))

    def _collect_thread_slice(self, thread_id: str) -> dict[str, Any]:
        storage = {ns: dict(by_id) for ns, by_id in self.storage.get(thread_id, {}).items()}
        blobs = {key: value for key, value in self.blobs.items() if key[0] == thread_id}
        writes = {key: dict(value) for key, value in self.writes.items() if key[0] == thread_id}
        return {"storage": storage, "blobs": blobs, "writes": writes}

    def clear_thread(self, thread_id: str) -> None:
        """Drop only this run's in-memory checkpoint slice.

        The durable state store is intentionally left untouched so a later
        redelivery or inspection can hydrate the run again if needed.
        """
        thread_id = str(thread_id or "").strip()
        if not thread_id:
            return
        self.storage.pop(thread_id, None)
        for key in list(self.blobs.keys()):
            if key and key[0] == thread_id:
                self.blobs.pop(key, None)
        for key in list(self.writes.keys()):
            if key and key[0] == thread_id:
                self.writes.pop(key, None)
        self._hydrated.discard(thread_id)

    def _restore_thread_slice(self, thread_id: str, slice_: dict[str, Any]) -> None:
        for ns, by_id in slice_.get("storage", {}).items():
            self.storage[thread_id][ns].update(by_id)
        for key, value in slice_.get("blobs", {}).items():
            self.blobs[key] = value
        for key, value in slice_.get("writes", {}).items():
            self.writes[key].update(value)


def _thread_id_from_config(config: Any) -> str:
    try:
        return str(config["configurable"]["thread_id"])
    except (KeyError, TypeError):
        return ""


def _encode_thread_slice(slice_: dict[str, Any]) -> bytes:
    return pickle.dumps({"version": _STATE_BLOB_VERSION, "slice": slice_})


def _decode_thread_slice(blob: bytes) -> dict[str, Any]:
    payload = pickle.loads(blob)
    if not isinstance(payload, dict) or payload.get("version") != _STATE_BLOB_VERSION:
        raise ValueError("unsupported checkpoint slice version")
    slice_ = payload.get("slice")
    if not isinstance(slice_, dict):
        raise ValueError("malformed checkpoint slice")
    return slice_


def run_graph_config(run_id: str, *, recursion_limit: int) -> dict[str, Any]:
    """Config that scopes graph checkpoints to a single run and bounds depth."""
    return {
        "recursion_limit": max(1, int(recursion_limit)),
        "configurable": {"thread_id": run_id, "checkpoint_ns": ""},
    }


async def checkpoint_has_pending_work(agent: Any, config: dict[str, Any]) -> bool:
    """True when a prior checkpoint exists for this run with unfinished work, so
    the graph should resume rather than start over."""
    getter = getattr(agent, "aget_state", None)
    if getter is None:
        return False
    try:
        snapshot = await getter(config)
    except Exception:
        logger.warning(
            "Could not read checkpoint state; starting run fresh.",
            extra={"thread_id": _thread_id_from_config(config)},
            exc_info=True,
        )
        return False
    pending = bool(getattr(snapshot, "next", ()) or ())
    return pending


def build_checkpoint_state_store(database_url: str) -> CheckpointStateStore | None:
    """Build the durable Postgres state store, or None when no DB is configured."""
    dsn = (database_url or "").strip()
    if not dsn:
        return None
    return PostgresCheckpointStateStore(dsn)
