from __future__ import annotations

import os
import pickle
import uuid

import pytest
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict
from ultra_deepagents.checkpointing import (
    CheckpointRunState,
    DurableCheckpointer,
    InMemoryCheckpointStateStore,
    PostgresCheckpointStateStore,
    _decode_thread_slice,
    _encode_thread_slice,
    checkpoint_run_state,
    run_graph_config,
)


class _State(TypedDict):
    steps: list


def _build_crashing_graph(side_effects: list[str], crash_armed: dict[str, bool]):
    """A 3-node graph whose middle node crashes once, recording every node
    execution so a test can prove completed nodes are not re-run on resume."""

    def node_a(state: _State) -> _State:
        side_effects.append("A")
        return {"steps": state.get("steps", []) + ["A"]}

    def node_b(state: _State) -> _State:
        side_effects.append("B")
        if crash_armed["v"]:
            crash_armed["v"] = False
            raise RuntimeError("simulated worker kill mid-run")
        return {"steps": state["steps"] + ["B"]}

    def node_c(state: _State) -> _State:
        side_effects.append("C")
        return {"steps": state["steps"] + ["C"]}

    graph = StateGraph(_State)
    graph.add_node("a", node_a)
    graph.add_node("b", node_b)
    graph.add_node("c", node_c)
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.add_edge("c", END)
    return graph


def test_durable_checkpointer_resumes_across_process_restart_without_repeating_completed_nodes():
    # One shared store models the durable Postgres backend; two checkpointer
    # instances model two worker processes (the second has empty memory).
    store = InMemoryCheckpointStateStore()
    side_effects: list[str] = []
    crash_armed = {"v": True}
    run_id = "run_resume_1"
    config = run_graph_config(run_id, recursion_limit=50)

    import asyncio

    async def scenario() -> dict:
        # Process 1: run until the middle node crashes.
        saver1 = DurableCheckpointer(store)
        app1 = _build_crashing_graph(side_effects, crash_armed).compile(checkpointer=saver1)
        crashed = False
        try:
            await app1.ainvoke({"steps": []}, config)
        except RuntimeError:
            crashed = True
        # Persistence is debounced; in a real run the sub-second window elapses
        # between super-steps, so the completed node's checkpoint is durable
        # before a crash. Force the trailing persist to model that.
        await saver1.flush()

        # Process 2: fresh memory; hydrate durable state, then resume with None.
        saver2 = DurableCheckpointer(store)
        hydrated = await saver2.hydrate(run_id)
        app2 = _build_crashing_graph(side_effects, crash_armed).compile(checkpointer=saver2)
        result = await app2.ainvoke(None, config)
        return {"crashed": crashed, "hydrated": hydrated, "result": result}

    outcome = asyncio.run(scenario())

    assert outcome["crashed"] is True
    assert outcome["hydrated"] is True, "second process must restore durable checkpoint state"
    assert outcome["result"]["steps"] == ["A", "B", "C"]
    # The completed node A ran exactly once across both processes — its work was
    # not wasted on restart. Only the in-flight node B re-ran (once per process).
    assert side_effects.count("A") == 1
    assert side_effects.count("B") == 2
    assert side_effects.count("C") == 1


def test_durable_checkpointer_reports_no_pending_work_for_unknown_run():
    import asyncio

    store = InMemoryCheckpointStateStore()
    saver = DurableCheckpointer(store)

    async def scenario() -> bool:
        return await saver.hydrate("run_never_seen")

    assert asyncio.run(scenario()) is False


def test_checkpoint_run_state_distinguishes_absent_pending_and_completed():
    import asyncio

    class _Agent:
        def __init__(self, pending: tuple[str, ...]) -> None:
            self.pending = pending
            self.calls = 0

        async def aget_state(self, _config):
            self.calls += 1

            class _Snapshot:
                next = self.pending

            return _Snapshot()

    async def scenario() -> tuple[CheckpointRunState, CheckpointRunState, CheckpointRunState, int]:
        absent_agent = _Agent(("must-not-be-read",))
        pending_agent = _Agent(("tools",))
        completed_agent = _Agent(())
        config = run_graph_config("run-state", recursion_limit=50)
        absent = await checkpoint_run_state(absent_agent, config, hydrated=False)
        pending = await checkpoint_run_state(pending_agent, config, hydrated=True)
        completed = await checkpoint_run_state(completed_agent, config, hydrated=True)
        return absent, pending, completed, absent_agent.calls

    absent, pending, completed, absent_calls = asyncio.run(scenario())

    assert absent is CheckpointRunState.ABSENT
    assert pending is CheckpointRunState.PENDING
    assert completed is CheckpointRunState.COMPLETED
    assert absent_calls == 0


def test_durable_checkpointer_clear_thread_removes_only_runtime_state_and_preserves_durable_store():
    import asyncio

    store = InMemoryCheckpointStateStore()
    saver = DurableCheckpointer(store)
    config_a = run_graph_config("run-clear-a", recursion_limit=50)
    config_b = run_graph_config("run-clear-b", recursion_limit=50)

    def has_runtime_state(thread_id: str) -> bool:
        return (
            bool(saver.storage.get(thread_id))
            or any(key[0] == thread_id for key in saver.blobs)
            or any(key[0] == thread_id for key in saver.writes)
        )

    async def scenario() -> dict:
        graph = _build_crashing_graph([], {"v": False})
        app = graph.compile(checkpointer=saver)
        await app.ainvoke({"steps": []}, config_a)
        await app.ainvoke({"steps": []}, config_b)
        await saver.flush()  # debounced persists land before we inspect durable state
        before = {
            "a_runtime": has_runtime_state("run-clear-a"),
            "b_runtime": has_runtime_state("run-clear-b"),
            "a_durable": await store.load("run-clear-a"),
            "b_durable": await store.load("run-clear-b"),
        }

        saver.clear_thread("run-clear-a")
        after_clear = {
            "a_runtime": has_runtime_state("run-clear-a"),
            "b_runtime": has_runtime_state("run-clear-b"),
            "a_durable": await store.load("run-clear-a"),
            "b_durable": await store.load("run-clear-b"),
        }
        rehydrated = await saver.hydrate("run-clear-a")
        return {
            "before": before,
            "after_clear": after_clear,
            "rehydrated": rehydrated,
            "a_runtime_after_hydrate": has_runtime_state("run-clear-a"),
        }

    outcome = asyncio.run(scenario())

    assert outcome["before"]["a_runtime"] is True
    assert outcome["before"]["b_runtime"] is True
    assert outcome["before"]["a_durable"]
    assert outcome["before"]["b_durable"]
    assert outcome["after_clear"]["a_runtime"] is False
    assert outcome["after_clear"]["b_runtime"] is True
    assert outcome["after_clear"]["a_durable"] == outcome["before"]["a_durable"]
    assert outcome["after_clear"]["b_durable"] == outcome["before"]["b_durable"]
    assert outcome["rehydrated"] is True
    assert outcome["a_runtime_after_hydrate"] is True


def test_run_graph_config_scopes_checkpoints_to_the_run():
    config = run_graph_config("run_xyz", recursion_limit=1000)
    assert config["recursion_limit"] == 1000
    assert config["configurable"]["thread_id"] == "run_xyz"
    assert config["configurable"]["checkpoint_ns"] == ""


def test_checkpoint_thread_slice_is_compressed_and_backward_compatible():
    large_message = "deterministic checkpoint payload " * 10000
    slice_ = {
        "storage": {
            "": {
                "checkpoint-1": {
                    "channel_values": {
                        "messages": [{"role": "assistant", "content": large_message}]
                    }
                }
            }
        },
        "blobs": {},
        "writes": {},
    }
    legacy_blob = pickle.dumps({"version": 1, "slice": slice_}, protocol=pickle.HIGHEST_PROTOCOL)

    encoded = _encode_thread_slice(slice_)

    assert encoded.startswith(b"ULTRA_DEEPAGENTS_CKPT")
    assert len(encoded) < len(legacy_blob) * 0.35
    assert _decode_thread_slice(encoded) == slice_
    assert _decode_thread_slice(legacy_blob) == slice_


def test_postgres_checkpoint_store_schema_indexes_updated_at_for_gc():
    import asyncio

    statements: list[str] = []

    class _Cursor:
        async def fetchone(self):
            return False, False

    class _FakeConn:
        closed = False

        async def execute(self, sql, _params=None):
            statements.append(sql)
            return _Cursor()

    class _FakeStore(PostgresCheckpointStateStore):
        async def _connection(self):
            return _FakeConn()

    asyncio.run(_FakeStore("postgresql://unused").ensure_schema())

    joined = "\n".join(statements)
    assert "CREATE TABLE IF NOT EXISTS deepagents_checkpoint_threads" in joined
    assert "REFERENCES control_runs(run_id) ON DELETE CASCADE" in joined
    assert "CREATE INDEX IF NOT EXISTS deepagents_checkpoint_threads_updated_at_idx" in joined
    assert "ON deepagents_checkpoint_threads (updated_at)" in joined


def test_postgres_checkpoint_store_skips_index_ddl_when_catalog_definition_is_current():
    import asyncio

    statements: list[str] = []

    class _Cursor:
        async def fetchone(self):
            return True, True

    class _FakeConn:
        closed = False

        async def execute(self, sql, _params=None):
            statements.append(sql)
            return _Cursor()

    class _FakeStore(PostgresCheckpointStateStore):
        async def _connection(self):
            return _FakeConn()

    asyncio.run(_FakeStore("postgresql://unused").ensure_schema())

    catalog_queries = [
        sql for sql in statements if "FROM pg_catalog.pg_class AS index_relation" in sql
    ]
    assert len(catalog_queries) == 1
    assert "index_info.indislive" in catalog_queries[0]
    assert not any("CREATE INDEX IF NOT EXISTS" in sql for sql in statements)


def test_postgres_checkpoint_gc_preserves_nonterminal_resume_state():
    import asyncio

    calls: list[tuple[str, tuple | None]] = []

    class _Cursor:
        rowcount = 2

    class _FakeConn:
        closed = False

        async def execute(self, sql, params=None):
            calls.append((sql, params))
            return _Cursor()

    class _FakeStore(PostgresCheckpointStateStore):
        async def _connection(self):
            return _FakeConn()

    async def scenario() -> int:
        store = _FakeStore("postgresql://unused")
        store._ready = True
        return await store.delete_older_than(72 * 3600)

    assert asyncio.run(scenario()) == 2
    sql, params = calls[-1]
    assert "USING control_runs AS control_run" in sql
    assert "checkpoint_row.thread_id = control_run.run_id" in sql
    assert "control_run.status IN ('succeeded', 'failed', 'canceled')" in sql
    assert "queued" not in sql
    assert "running" not in sql
    assert "waiting_for_input" not in sql
    assert params == (float(72 * 3600),)


def test_postgres_checkpoint_store_skips_identical_state_rewrites():
    import asyncio

    statements: list[str] = []

    class _FakeConn:
        closed = False

        async def execute(self, sql, _params=None):
            statements.append(sql)

    class _FakeStore(PostgresCheckpointStateStore):
        async def _connection(self):
            return _FakeConn()

    async def scenario() -> None:
        store = _FakeStore("postgresql://unused")
        store._ready = True
        await store.save("run-1", b"state")

    asyncio.run(scenario())

    save_sql = "\n".join(statements)
    assert "ON CONFLICT (thread_id)" in save_sql
    assert "state IS DISTINCT FROM EXCLUDED.state" in save_sql


def test_run_job_resumes_with_none_payload_and_seeded_sequencer_when_checkpoint_pending(tmp_path):
    """run_job must drive a redelivered run into LangGraph resume: invoke with
    a None payload (continue from checkpoint) and seed event ids above the
    run's already-persisted events so resumed events are not deduped away."""
    import asyncio

    from ultra_deepagents.config import RuntimeSettings
    from ultra_deepagents.runner import run_job
    from ultra_deepagents.schemas import RunJobEnvelope

    captured: dict = {}

    class _FakeResumingAgent:
        async def aget_state(self, config):
            # Report pending work so run_job chooses resume.
            class _Snapshot:
                next = ("tools",)

            return _Snapshot()

        async def astream_events(self, payload, config=None, *, context=None, version=None):
            captured["payload"] = payload
            captured["config"] = config
            yield {
                "type": "event",
                "method": "values",
                "params": {
                    "namespace": [],
                    "data": {
                        "messages": [
                            {"role": "user", "content": "resume me"},
                            {"role": "assistant", "content": "resumed answer"},
                        ]
                    },
                },
            }

    def fake_agent_factory(settings, **kwargs):
        captured["factory_checkpointer"] = kwargs.get("checkpointer")
        return _FakeResumingAgent()

    published: list[dict] = []
    event_emission_ready_at_event_counts: list[int] = []

    async def publish(event):
        published.append(event)

    def on_event_emission_ready() -> None:
        event_emission_ready_at_event_counts.append(len(published))

    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        workspace_root=str(tmp_path / "ws"),
        artifact_root=str(tmp_path / "art"),
        memory_root=str(tmp_path / "mem"),
        title_generation_enabled=False,
    )
    job = RunJobEnvelope(run_id="run-resume-7", thread_id="thread-1", user_id="u", goal="resume me")

    store = InMemoryCheckpointStateStore()

    class _HydrateTrackingCheckpointer(DurableCheckpointer):
        hydrated_runs: list[str] = []

        async def hydrate(self, thread_id: str) -> bool:
            # A fresh worker must hydrate durable state before deciding to
            # resume; record the call so the ordering is pinned.
            type(self).hydrated_runs.append(thread_id)
            return True

    checkpointer = _HydrateTrackingCheckpointer(store)

    asyncio.run(
        run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=fake_agent_factory,
            checkpointer=checkpointer,
            sequence_floor=42,
            on_event_emission_ready=on_event_emission_ready,
        )
    )

    # run_job must hydrate the run's durable checkpoint before checking pending
    # work, or a restarted worker always sees empty state and restarts the run.
    assert "run-resume-7" in _HydrateTrackingCheckpointer.hydrated_runs
    # Resume payload is None (continue from checkpoint), not the original messages.
    assert captured["payload"] is None
    assert captured["config"]["configurable"]["thread_id"] == "run-resume-7"
    assert captured["factory_checkpointer"] is checkpointer
    assert event_emission_ready_at_event_counts == [0]
    # The first stamped event sits above the seeded floor so it cannot collide
    # with the original partial run's persisted event ids.
    first_sequence = min(event["sequence"] for event in published)
    assert first_sequence > 42
    assert any(event["event_kind"] == "run.resumed" for event in published)


def test_run_job_starts_fresh_when_no_checkpoint_pending(tmp_path):
    import asyncio

    from ultra_deepagents.config import RuntimeSettings
    from ultra_deepagents.runner import run_job
    from ultra_deepagents.schemas import RunJobEnvelope

    captured: dict = {}

    class _FakeFreshAgent:
        async def aget_state(self, config):
            class _Snapshot:
                next = ()  # no pending work -> fresh start

            return _Snapshot()

        async def astream_events(self, payload, config=None, *, context=None, version=None):
            captured["payload"] = payload
            yield {
                "type": "event",
                "method": "values",
                "params": {
                    "namespace": [],
                    "data": {"messages": [{"role": "assistant", "content": "fresh"}]},
                },
            }

    published: list[dict] = []
    event_emission_ready_at_event_counts: list[int] = []

    async def publish(event):
        published.append(event)

    def on_event_emission_ready() -> None:
        event_emission_ready_at_event_counts.append(len(published))

    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        workspace_root=str(tmp_path / "ws"),
        artifact_root=str(tmp_path / "art"),
        memory_root=str(tmp_path / "mem"),
        title_generation_enabled=False,
    )
    job = RunJobEnvelope(
        run_id="run-fresh-1",
        thread_id="thread-1",
        user_id="u",
        goal="hello",
        messages=[{"role": "user", "content": "hello"}],
    )

    asyncio.run(
        run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda settings, **kwargs: _FakeFreshAgent(),
            checkpointer=DurableCheckpointer(InMemoryCheckpointStateStore()),
            sequence_floor=0,
            on_event_emission_ready=on_event_emission_ready,
        )
    )

    assert captured["payload"] == {"messages": [{"role": "user", "content": "hello"}]}
    assert event_emission_ready_at_event_counts == [0]


def test_run_job_completed_checkpoint_waits_without_publishing_or_recomputing(tmp_path):
    import asyncio

    from ultra_deepagents.config import RuntimeSettings
    from ultra_deepagents.runner import CheckpointReconciliationPendingError, run_job
    from ultra_deepagents.schemas import RunJobEnvelope

    compute_started = False
    published: list[dict] = []
    event_emission_ready_calls = 0

    class _CompletedAgent:
        async def aget_state(self, _config):
            class _Snapshot:
                next = ()

            return _Snapshot()

        async def astream_events(self, *_args, **_kwargs):
            nonlocal compute_started
            compute_started = True
            raise AssertionError("completed checkpoint must not restart graph compute")
            yield  # pragma: no cover

    class _CompletedCheckpointer(DurableCheckpointer):
        async def hydrate(self, thread_id: str) -> bool:
            assert thread_id == "run-completed-lag"
            return True

    async def publish(event):
        published.append(event)

    def on_event_emission_ready() -> None:
        nonlocal event_emission_ready_calls
        event_emission_ready_calls += 1

    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        workspace_root=str(tmp_path / "ws-completed"),
        artifact_root=str(tmp_path / "art-completed"),
        memory_root=str(tmp_path / "mem-completed"),
        title_generation_enabled=False,
    )
    job = RunJobEnvelope(
        run_id="run-completed-lag",
        thread_id="thread-1",
        user_id="u",
        goal="do not replay",
        messages=[{"role": "user", "content": "do not replay"}],
    )
    lease_path = tmp_path / "ws-completed" / "run-completed-lag" / "lease.json"
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    original_lease = '{"status":"succeeded","sentinel":"original-terminal-lease"}'
    lease_path.write_text(original_lease)

    async def scenario() -> None:
        with pytest.raises(CheckpointReconciliationPendingError):
            await run_job(
                job,
                settings,
                publish_event=publish,
                agent_factory=lambda settings, **kwargs: _CompletedAgent(),
                checkpointer=_CompletedCheckpointer(InMemoryCheckpointStateStore()),
                sequence_floor=41,
                on_event_emission_ready=on_event_emission_ready,
            )

    asyncio.run(scenario())

    assert compute_started is False
    assert published == []
    assert event_emission_ready_calls == 0
    assert lease_path.read_text() == original_lease


_POSTGRES_DSN = os.getenv(
    "ULTRA_CHECKPOINT_TEST_DSN",
    os.getenv("ULTRA_CONTROL_TEST_DATABASE_URL", ""),
)


@pytest.mark.skipif(not _POSTGRES_DSN, reason="no Postgres DSN configured for checkpoint test")
def test_postgres_checkpoint_store_round_trips_durable_resume():
    import asyncio

    side_effects: list[str] = []
    crash_armed = {"v": True}
    suffix = uuid.uuid4().hex
    thread_id = f"thread_pg_resume_{suffix}"
    run_id = f"run_pg_resume_{suffix}"
    config = run_graph_config(run_id, recursion_limit=50)

    async def scenario() -> dict:
        store1 = PostgresCheckpointStateStore(_POSTGRES_DSN)
        try:
            await store1.ensure_schema()
            conn = await store1._connection()
            await conn.execute(
                """
                INSERT INTO control_threads (
                  thread_id, user_id, title, status, created_at, updated_at, metadata
                ) VALUES (%s, %s, %s, 'active', now(), now(), '{}'::jsonb)
                """,
                (thread_id, f"user_{suffix}", "Checkpoint resume integration"),
            )
            await conn.execute(
                """
                INSERT INTO control_runs (
                  run_id, thread_id, user_id, goal, status, workflow_kind,
                  created_at, updated_at, metadata
                ) VALUES (%s, %s, %s, %s, 'running', 'deep_agents', now(), now(), '{}'::jsonb)
                """,
                (run_id, thread_id, f"user_{suffix}", "Checkpoint resume integration"),
            )
            saver1 = DurableCheckpointer(store1)
            app1 = _build_crashing_graph(side_effects, crash_armed).compile(checkpointer=saver1)
            crashed = False
            try:
                await app1.ainvoke({"steps": []}, config)
            except RuntimeError:
                crashed = True
            assert await saver1.flush(run_id) is True
            await store1.close()

            # Brand-new store + saver: only Postgres carries the state across.
            store2 = PostgresCheckpointStateStore(_POSTGRES_DSN)
            saver2 = DurableCheckpointer(store2)
            try:
                hydrated = await saver2.hydrate(run_id)
                app2 = _build_crashing_graph(side_effects, crash_armed).compile(checkpointer=saver2)
                result = await app2.ainvoke(None, config)
                await saver2.delete_thread(run_id)
                return {"crashed": crashed, "hydrated": hydrated, "result": result}
            finally:
                await store2.close()
        finally:
            if not store1._conn or store1._conn.closed:
                cleanup = PostgresCheckpointStateStore(_POSTGRES_DSN)
            else:
                cleanup = store1
            try:
                cleanup_conn = await cleanup._connection()
                await cleanup_conn.execute(
                    "DELETE FROM control_threads WHERE thread_id = %s", (thread_id,)
                )
            finally:
                await cleanup.close()

    outcome = asyncio.run(scenario())
    assert outcome["crashed"] is True
    assert outcome["hydrated"] is True
    assert outcome["result"]["steps"] == ["A", "B", "C"]
    assert side_effects.count("A") == 1
    assert side_effects.count("C") == 1


def test_durable_checkpointer_delete_thread_removes_runtime_and_durable_row():
    import asyncio

    store = InMemoryCheckpointStateStore()
    saver = DurableCheckpointer(store, persist_debounce_seconds=0.0)
    config = run_graph_config("run-del", recursion_limit=50)

    async def scenario() -> dict:
        app = _build_crashing_graph([], {"v": False}).compile(checkpointer=saver)
        await app.ainvoke({"steps": []}, config)
        await saver.flush()
        durable_before = await store.load("run-del")
        await saver.delete_thread("run-del")
        return {
            "durable_before": durable_before,
            "durable_after": await store.load("run-del"),
            "runtime_after": bool(saver.storage.get("run-del")),
            "task_after": "run-del" in saver._persist_tasks,
            "dirty_after": "run-del" in saver._dirty,
        }

    outcome = asyncio.run(scenario())
    assert outcome["durable_before"] is not None  # was durably checkpointed
    assert outcome["durable_after"] is None  # terminal delete removed the row
    assert outcome["runtime_after"] is False  # in-memory slice cleared
    assert outcome["task_after"] is False
    assert outcome["dirty_after"] is False


def test_durable_checkpointer_delete_thread_cancels_pending_persist_no_resurrection():
    import asyncio

    # A long debounce means the scheduled persist is still pending when we delete;
    # delete_thread must cancel it so a late write can't recreate the row.
    store = InMemoryCheckpointStateStore()
    saver = DurableCheckpointer(store, persist_debounce_seconds=5.0)
    config = run_graph_config("run-race", recursion_limit=50)

    async def scenario() -> bytes | None:
        app = _build_crashing_graph([], {"v": False}).compile(checkpointer=saver)
        await app.ainvoke({"steps": []}, config)
        assert "run-race" in saver._persist_tasks  # a persist is pending
        await saver.delete_thread("run-race")
        await asyncio.sleep(0.05)  # give any uncancelled task a chance to fire
        return await store.load("run-race")

    assert asyncio.run(scenario()) is None


def test_durable_checkpointer_flush_reports_failure_until_freshest_state_is_persisted():
    import asyncio

    class _FailingStore(InMemoryCheckpointStateStore):
        fail_saves = True

        async def save(self, thread_id: str, blob: bytes) -> None:
            if self.fail_saves:
                raise RuntimeError("simulated durable save failure")
            await super().save(thread_id, blob)

    async def scenario() -> dict:
        store = _FailingStore()
        saver = DurableCheckpointer(store, persist_debounce_seconds=5.0)
        config = run_graph_config("run-flush-fence", recursion_limit=50)
        app = _build_crashing_graph([], {"v": False}).compile(checkpointer=saver)
        await app.ainvoke({"steps": []}, config)
        first = await saver.flush("run-flush-fence")
        runtime_after_failure = saver._thread_has_runtime_state("run-flush-fence")
        store.fail_saves = False
        second = await saver.flush("run-flush-fence")
        return {
            "first": first,
            "second": second,
            "runtime_after_failure": runtime_after_failure,
            "durable": await store.load("run-flush-fence"),
        }

    outcome = asyncio.run(scenario())

    assert outcome["first"] is False
    assert outcome["runtime_after_failure"] is True
    assert outcome["second"] is True
    assert outcome["durable"] is not None


def test_checkpoint_store_gc_reaps_only_rows_older_than_retention():
    import asyncio

    async def scenario() -> dict:
        store = InMemoryCheckpointStateStore()
        await store.save("old", b"x")
        await store.save("fresh", b"y")
        # Deterministic clock: now=1000, old saved at 0, fresh at 900, retention 500.
        store._saved_at["old"] = 0.0
        store._saved_at["fresh"] = 900.0
        reaped = await store.delete_older_than(500, now_seconds=1000.0)
        return {
            "reaped": reaped,
            "old": await store.load("old"),
            "fresh": await store.load("fresh"),
        }

    outcome = asyncio.run(scenario())
    assert outcome["reaped"] == 1
    assert outcome["old"] is None
    assert outcome["fresh"] is not None


def test_durable_checkpointer_gc_wires_through_to_store():
    import asyncio

    async def scenario() -> int:
        store = InMemoryCheckpointStateStore()
        await store.save("abandoned", b"x")
        store._saved_at["abandoned"] = store._saved_at["abandoned"] - 10_000
        saver = DurableCheckpointer(store, persist_debounce_seconds=0.0)
        return await saver.gc(retention_seconds=3600)

    assert asyncio.run(scenario()) == 1
