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
import contextlib
import logging
import pickle
import time
import zlib
from enum import Enum
from typing import Any, Protocol

from langgraph.checkpoint.memory import InMemorySaver

logger = logging.getLogger(__name__)

# Bumped if the persisted slice layout changes so stale blobs are ignored
# instead of mis-deserialized. Version 1 was a raw pickle payload; version 2 is
# a magic-prefixed compressed envelope whose inner payload remains the v1 slice.
_STATE_BLOB_VERSION = 2
_LEGACY_STATE_BLOB_VERSION = 1
_STATE_BLOB_MAGIC = b"ULTRA_DEEPAGENTS_CKPT\x00"
_COMPRESSION_ZSTD = "zstd"
_COMPRESSION_ZLIB = "zlib"

_CHECKPOINT_TABLE = "deepagents_checkpoint_threads"
_CHECKPOINT_UPDATED_AT_INDEX = f"{_CHECKPOINT_TABLE}_updated_at_idx"


class CheckpointRunState(str, Enum):
    """Authoritative durable state for one run before graph invocation."""

    ABSENT = "absent"
    PENDING = "pending"
    COMPLETED = "completed"


class CheckpointStateUnavailableError(RuntimeError):
    """Durable checkpoint state could not be classified safely."""


class CheckpointStateStore(Protocol):
    """Durable key/value home for a run's serialized checkpoint slice."""

    async def load(self, thread_id: str) -> bytes | None: ...

    async def save(self, thread_id: str, blob: bytes) -> None: ...

    async def delete(self, thread_id: str) -> None: ...

    async def close(self) -> None: ...


class InMemoryCheckpointStateStore:
    """Process-local state store. Two checkpointers sharing one instance model
    two worker processes sharing the same durable backend."""

    def __init__(self) -> None:
        self._blobs: dict[str, bytes] = {}
        self._saved_at: dict[str, float] = {}

    async def load(self, thread_id: str) -> bytes | None:
        return self._blobs.get(thread_id)

    async def save(self, thread_id: str, blob: bytes) -> None:
        self._blobs[thread_id] = blob
        self._saved_at[thread_id] = time.monotonic()

    async def delete(self, thread_id: str) -> None:
        self._blobs.pop(thread_id, None)
        self._saved_at.pop(thread_id, None)

    async def delete_older_than(
        self, retention_seconds: float, *, now_seconds: float | None = None
    ) -> int:
        if retention_seconds <= 0:
            return 0
        now = time.monotonic() if now_seconds is None else now_seconds
        expires_before = now - retention_seconds
        stale = [tid for tid, ts in self._saved_at.items() if ts < expires_before]
        for tid in stale:
            self._blobs.pop(tid, None)
            self._saved_at.pop(tid, None)
        return len(stale)

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
                    thread_id text PRIMARY KEY
                      REFERENCES control_runs(run_id) ON DELETE CASCADE,
                    state bytea NOT NULL,
                    updated_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            # CREATE INDEX IF NOT EXISTS still takes relation locks. Issuing it
            # every time a fresh worker process starts can therefore block live
            # checkpoint upserts even though the index already exists. Inspect
            # the catalog first and reserve DDL for genuine first-time setup.
            cursor = await conn.execute(
                f"""
                SELECT
                  pg_catalog.to_regclass(
                    pg_catalog.current_schema() || '.{_CHECKPOINT_UPDATED_AT_INDEX}'
                  ) IS NOT NULL AS index_exists,
                  EXISTS (
                    SELECT 1
                    FROM pg_catalog.pg_class AS index_relation
                    JOIN pg_catalog.pg_index AS index_info
                      ON index_info.indexrelid = index_relation.oid
                    JOIN pg_catalog.pg_class AS table_relation
                      ON table_relation.oid = index_info.indrelid
                    JOIN pg_catalog.pg_namespace AS table_namespace
                      ON table_namespace.oid = table_relation.relnamespace
                    JOIN pg_catalog.pg_attribute AS indexed_column
                      ON indexed_column.attrelid = table_relation.oid
                     AND indexed_column.attnum = index_info.indkey[0]
                    WHERE index_relation.relname = '{_CHECKPOINT_UPDATED_AT_INDEX}'
                      AND table_relation.relname = '{_CHECKPOINT_TABLE}'
                      AND table_namespace.nspname = pg_catalog.current_schema()
                      AND index_info.indisvalid
                      AND index_info.indisready
                      AND index_info.indislive
                      AND NOT index_info.indisunique
                      AND index_info.indnkeyatts = 1
                      AND index_info.indnatts = 1
                      AND index_info.indpred IS NULL
                      AND index_info.indexprs IS NULL
                      AND index_info.indoption[0] = 0
                      AND indexed_column.attname = 'updated_at'
                      AND index_relation.relam = (
                        SELECT access_method.oid
                        FROM pg_catalog.pg_am AS access_method
                        WHERE access_method.amname = 'btree'
                      )
                  ) AS index_is_current
                """
            )
            row = await cursor.fetchone()
            index_exists = bool(row and row[0])
            index_is_current = bool(row and row[1])
            if index_exists and not index_is_current:
                raise RuntimeError(
                    f"checkpoint index {_CHECKPOINT_UPDATED_AT_INDEX} has an incompatible definition"
                )
            if not index_is_current:
                await conn.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {_CHECKPOINT_UPDATED_AT_INDEX}
                    ON {_CHECKPOINT_TABLE} (updated_at)
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
                DO UPDATE SET
                  state = CASE
                    WHEN {_CHECKPOINT_TABLE}.state IS DISTINCT FROM EXCLUDED.state
                    THEN EXCLUDED.state
                    ELSE {_CHECKPOINT_TABLE}.state
                  END,
                  updated_at = now()
                """,
                (thread_id, blob),
            )

    async def delete(self, thread_id: str) -> None:
        await self.ensure_schema()
        async with self._lock:
            conn = await self._connection()
            await conn.execute(
                f"DELETE FROM {_CHECKPOINT_TABLE} WHERE thread_id = %s",
                (thread_id,),
            )

    async def delete_older_than(self, retention_seconds: float) -> int:
        """Reap old checkpoint rows only after their run is terminal.

        Age alone is not evidence that a queued/running/waiting run is
        abandoned: long-lived jobs and wedged partitions still need their only
        resume point. Normal terminal handling deletes immediately; this is the
        bounded privacy backstop for cleanup failures.
        """
        if retention_seconds <= 0:
            return 0
        await self.ensure_schema()
        async with self._lock:
            conn = await self._connection()
            cursor = await conn.execute(
                f"DELETE FROM {_CHECKPOINT_TABLE} AS checkpoint_row "
                "USING control_runs AS control_run "
                "WHERE checkpoint_row.thread_id = control_run.run_id "
                "AND control_run.status IN ('succeeded', 'failed', 'canceled') "
                "AND checkpoint_row.updated_at < now() - make_interval(secs => %s)",
                (float(retention_seconds),),
            )
        return cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0

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

    Persistence is debounced per thread: a burst of writes within a super-step
    (one ``aput`` plus several ``aput_writes`` for fan-out tasks) coalesces into a
    single trailing slice upload instead of one full re-serialization per write,
    and the graph no longer blocks on the durable write. The window is small
    (sub-second) relative to LLM/tool super-steps, so a crash still resumes from
    the last flushed super-step; LangGraph re-runs only the in-flight one.
    """

    def __init__(
        self,
        store: CheckpointStateStore,
        *,
        serde: Any | None = None,
        persist_debounce_seconds: float = 0.25,
    ) -> None:
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
        self._persist_debounce_seconds = max(0.0, persist_debounce_seconds)
        self._dirty: set[str] = set()
        self._persist_tasks: dict[str, asyncio.Task[bool]] = {}
        # Generation fencing lets flush prove that the freshest in-memory
        # state, not merely some earlier successful snapshot, reached the
        # durable store before worker-local state is released.
        self._mutation_generation: dict[str, int] = {}
        self._persisted_generation: dict[str, int] = {}

    async def hydrate(self, thread_id: str) -> bool:
        """Load a run's prior checkpoint slice into memory. Returns True when
        durable state was found and restored. Safe to call repeatedly."""
        if thread_id in self._hydrated:
            # A run that started without a durable row can acquire in-memory
            # state before an ACK/flush failure redelivers it to the same
            # worker. Treat that retained state as existing checkpoint
            # authority; returning False here would restart the graph.
            return self._thread_has_runtime_state(thread_id)
        try:
            blob = await self._store.load(thread_id)
        except Exception as exc:
            logger.warning(
                "Checkpoint hydrate failed; refusing to classify the run as fresh.",
                extra={"thread_id": thread_id},
                exc_info=True,
            )
            raise CheckpointStateUnavailableError("durable checkpoint could not be loaded") from exc
        if blob is None:
            self._hydrated.add(thread_id)
            return False
        try:
            slice_ = _decode_thread_slice(blob)
        except Exception as exc:
            logger.warning(
                "Checkpoint slice could not be decoded; refusing to restart the run.",
                extra={"thread_id": thread_id},
                exc_info=True,
            )
            raise CheckpointStateUnavailableError(
                "durable checkpoint could not be decoded"
            ) from exc
        self._restore_thread_slice(thread_id, slice_)
        self._hydrated.add(thread_id)
        return True

    async def _persist(self, thread_id: str) -> bool:
        generation = self._mutation_generation.get(thread_id, 0)
        try:
            # Snapshot the slice synchronously on the loop (consistent view of
            # the saver's in-memory dicts), then pickle+compress it off-loop:
            # the encode is pure CPU over the run's entire cumulative slice and
            # would otherwise starve NATS acks/heartbeats on every super-step.
            slice_ = self._collect_thread_slice(thread_id)
            blob = await asyncio.to_thread(_encode_thread_slice, slice_)
            await self._store.save(thread_id, blob)
            self._persisted_generation[thread_id] = max(
                self._persisted_generation.get(thread_id, 0),
                generation,
            )
            return True
        except Exception:
            # The graph may keep running through a transient durability
            # hiccup, but flush reports failure at the delivery boundary so the
            # worker retains its freshest in-memory recovery state.
            logger.warning(
                "Checkpoint persist failed; resume may be unavailable for this run.",
                extra={"thread_id": thread_id},
                exc_info=True,
            )
            return False

    def _schedule_persist(self, thread_id: str) -> None:
        """Mark the thread dirty and ensure exactly one trailing persist task is
        in flight for it; a task already running will pick up the latest slice."""
        if not thread_id:
            return
        self._mutation_generation[thread_id] = self._mutation_generation.get(thread_id, 0) + 1
        self._dirty.add(thread_id)
        existing = self._persist_tasks.get(thread_id)
        if existing is not None and not existing.done():
            return
        try:
            self._persist_tasks[thread_id] = asyncio.ensure_future(
                self._debounced_persist(thread_id)
            )
        except RuntimeError:
            # No running loop (shouldn't happen on the async write path); fall
            # back to leaving the thread dirty so a later flush persists it.
            self._dirty.add(thread_id)

    async def _debounced_persist(self, thread_id: str) -> bool:
        persisted = True
        try:
            if self._persist_debounce_seconds > 0:
                await asyncio.sleep(self._persist_debounce_seconds)
            if thread_id in self._dirty:
                self._dirty.discard(thread_id)
                persisted = await self._persist(thread_id)
        finally:
            if self._persist_tasks.get(thread_id) is asyncio.current_task():
                self._persist_tasks.pop(thread_id, None)
        # A write that arrived during the persist re-dirties the thread; chain
        # one more trailing persist so the very last super-step is never lost.
        if thread_id in self._dirty:
            self._schedule_persist(thread_id)
        return persisted

    async def flush(self, thread_id: str | None = None) -> bool:
        """Force any pending debounced persist(s) to complete.

        Awaited at graceful boundaries (and in tests) so the latest committed
        super-step is durable before another process may hydrate it. Returns
        True only when every requested thread's freshest generation is known to
        be durable.
        """
        targets = (
            [thread_id]
            if thread_id
            else list(set(self._persist_tasks) | self._dirty | set(self._mutation_generation))
        )
        all_persisted = True
        for tid in targets:
            if not tid:
                continue
            while True:
                task = self._persist_tasks.get(tid)
                if task is not None:
                    try:
                        await task
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        # _persist reports ordinary save failures as False, but
                        # retain fail-closed behavior for an unexpected task
                        # exception too.
                        logger.warning(
                            "Checkpoint persist task failed; retaining worker-local state.",
                            extra={"thread_id": tid},
                            exc_info=True,
                        )
                    continue
                if tid in self._dirty:
                    self._dirty.discard(tid)
                    if not await self._persist(tid):
                        all_persisted = False
                        break
                    continue
                generation = self._mutation_generation.get(tid, 0)
                if self._persisted_generation.get(tid, 0) < generation:
                    # A prior debounced save failed after consuming the dirty
                    # bit. Make one synchronous boundary retry; failure is
                    # returned to the worker rather than hidden.
                    if not await self._persist(tid):
                        all_persisted = False
                    break
                break
        return all_persisted

    async def aput(self, config: Any, checkpoint: Any, metadata: Any, new_versions: Any) -> Any:
        result = await super().aput(config, checkpoint, metadata, new_versions)
        self._schedule_persist(_thread_id_from_config(config))
        return result

    async def aput_writes(
        self, config: Any, writes: Any, task_id: str, task_path: str = ""
    ) -> None:
        await super().aput_writes(config, writes, task_id, task_path)
        self._schedule_persist(_thread_id_from_config(config))

    def _collect_thread_slice(self, thread_id: str) -> dict[str, Any]:
        storage = {ns: dict(by_id) for ns, by_id in self.storage.get(thread_id, {}).items()}
        blobs = {key: value for key, value in self.blobs.items() if key[0] == thread_id}
        writes = {key: dict(value) for key, value in self.writes.items() if key[0] == thread_id}
        return {"storage": storage, "blobs": blobs, "writes": writes}

    def _thread_has_runtime_state(self, thread_id: str) -> bool:
        return bool(
            self.storage.get(thread_id)
            or any(key and key[0] == thread_id for key in self.blobs)
            or any(key and key[0] == thread_id for key in self.writes)
        )

    def clear_thread(self, thread_id: str) -> None:
        """Drop only this run's in-memory checkpoint slice.

        The durable state store is intentionally left untouched so a later
        redelivery or inspection can hydrate the run again if needed. To also
        drop the durable row (a terminal run that will never resume), use
        :meth:`delete_thread`.
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
        self._mutation_generation.pop(thread_id, None)
        self._persisted_generation.pop(thread_id, None)

    async def delete_thread(self, thread_id: str) -> None:
        """Drop a terminal run's in-memory slice AND its durable row.

        Called on terminal ack (succeeded/failed/canceled): the run will never be
        redelivered or resumed, so leaving its (potentially large) row in the
        shared Postgres only leaks disk. Cancels any pending debounced persist
        first so a late trailing write cannot resurrect the row after deletion.
        """
        thread_id = str(thread_id or "").strip()
        if not thread_id:
            return
        self._dirty.discard(thread_id)
        task = self._persist_tasks.pop(thread_id, None)
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self.clear_thread(thread_id)
        try:
            await self._store.delete(thread_id)
        except Exception:
            logger.warning(
                "Durable checkpoint delete failed; row may linger until GC.",
                extra={"thread_id": thread_id},
                exc_info=True,
            )

    async def gc(self, retention_seconds: float) -> int:
        """Reap old terminal-run rows after immediate cleanup failed.

        Production Postgres keeps resumable nonterminal rows regardless of age.
        Test/local stores without run-status metadata retain their simple
        time-based behavior.
        """
        reaper = getattr(self._store, "delete_older_than", None)
        if reaper is None or retention_seconds <= 0:
            return 0
        try:
            return int(await reaper(retention_seconds))
        except Exception:
            logger.warning("Durable checkpoint GC failed.", exc_info=True)
            return 0

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
    raw = pickle.dumps(
        {"version": _LEGACY_STATE_BLOB_VERSION, "slice": slice_},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    algorithm, compressed = _compress_checkpoint_payload(raw)
    envelope = {
        "version": _STATE_BLOB_VERSION,
        "compression": algorithm,
        "raw_size": len(raw),
        "payload": compressed,
    }
    return _STATE_BLOB_MAGIC + pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)


def _decode_thread_slice(blob: bytes) -> dict[str, Any]:
    if blob.startswith(_STATE_BLOB_MAGIC):
        envelope = pickle.loads(blob[len(_STATE_BLOB_MAGIC) :])
        if not isinstance(envelope, dict) or envelope.get("version") != _STATE_BLOB_VERSION:
            raise ValueError("unsupported checkpoint envelope version")
        payload = envelope.get("payload")
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise ValueError("malformed checkpoint envelope payload")
        raw = _decompress_checkpoint_payload(
            str(envelope.get("compression") or ""),
            bytes(payload),
        )
        return _decode_legacy_thread_slice(raw)
    return _decode_legacy_thread_slice(blob)


def _decode_legacy_thread_slice(blob: bytes) -> dict[str, Any]:
    payload = pickle.loads(blob)
    if not isinstance(payload, dict) or payload.get("version") != _LEGACY_STATE_BLOB_VERSION:
        raise ValueError("unsupported checkpoint slice version")
    slice_ = payload.get("slice")
    if not isinstance(slice_, dict):
        raise ValueError("malformed checkpoint slice")
    return slice_


def _compress_checkpoint_payload(raw: bytes) -> tuple[str, bytes]:
    try:
        import zstandard as zstd

        return _COMPRESSION_ZSTD, zstd.ZstdCompressor(level=3).compress(raw)
    except Exception:
        return _COMPRESSION_ZLIB, zlib.compress(raw, level=6)


def _decompress_checkpoint_payload(algorithm: str, payload: bytes) -> bytes:
    if algorithm == _COMPRESSION_ZSTD:
        import zstandard as zstd

        return zstd.ZstdDecompressor().decompress(payload)
    if algorithm == _COMPRESSION_ZLIB:
        return zlib.decompress(payload)
    raise ValueError("unsupported checkpoint compression")


def run_graph_config(run_id: str, *, recursion_limit: int) -> dict[str, Any]:
    """Config that scopes graph checkpoints to a single run and bounds depth."""
    return {
        "recursion_limit": max(1, int(recursion_limit)),
        "configurable": {"thread_id": run_id, "checkpoint_ns": ""},
    }


async def checkpoint_run_state(
    agent: Any,
    config: dict[str, Any],
    *,
    hydrated: bool,
) -> CheckpointRunState:
    """Classify a run without collapsing completed state into absence.

    Only an absent durable row authorizes a fresh graph invocation. A restored
    snapshot with pending tasks resumes; a restored snapshot with no pending
    tasks is completed and must wait for terminal-event reconciliation instead
    of replaying the original request.
    """
    if not hydrated:
        return CheckpointRunState.ABSENT
    getter = getattr(agent, "aget_state", None)
    if getter is None:
        raise CheckpointStateUnavailableError(
            "compiled graph cannot classify restored checkpoint state"
        )
    try:
        snapshot = await getter(config)
    except Exception as exc:
        logger.warning(
            "Could not read restored checkpoint state; refusing to restart the run.",
            extra={"thread_id": _thread_id_from_config(config)},
            exc_info=True,
        )
        raise CheckpointStateUnavailableError(
            "restored checkpoint state could not be classified"
        ) from exc
    pending = bool(getattr(snapshot, "next", ()) or ())
    return CheckpointRunState.PENDING if pending else CheckpointRunState.COMPLETED


def build_checkpoint_state_store(database_url: str) -> CheckpointStateStore | None:
    """Build the durable Postgres state store, or None when no DB is configured."""
    dsn = (database_url or "").strip()
    if not dsn:
        return None
    return PostgresCheckpointStateStore(dsn)
