from __future__ import annotations

import os

import pytest
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict
from ultra_deepagents.checkpointing import (
    DurableCheckpointer,
    InMemoryCheckpointStateStore,
    PostgresCheckpointStateStore,
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

    async def publish(event):
        published.append(event)

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
            return await super().hydrate(thread_id)

    checkpointer = _HydrateTrackingCheckpointer(store)

    asyncio.run(
        run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=fake_agent_factory,
            checkpointer=checkpointer,
            sequence_floor=42,
        )
    )

    # run_job must hydrate the run's durable checkpoint before checking pending
    # work, or a restarted worker always sees empty state and restarts the run.
    assert "run-resume-7" in _HydrateTrackingCheckpointer.hydrated_runs
    # Resume payload is None (continue from checkpoint), not the original messages.
    assert captured["payload"] is None
    assert captured["config"]["configurable"]["thread_id"] == "run-resume-7"
    assert captured["factory_checkpointer"] is checkpointer
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

    async def publish(event):
        return None

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
        )
    )

    assert captured["payload"] == {"messages": [{"role": "user", "content": "hello"}]}


_POSTGRES_DSN = os.getenv(
    "ULTRA_CHECKPOINT_TEST_DSN",
    os.getenv("ULTRA_CONTROL_TEST_DATABASE_URL", ""),
)


@pytest.mark.skipif(not _POSTGRES_DSN, reason="no Postgres DSN configured for checkpoint test")
def test_postgres_checkpoint_store_round_trips_durable_resume():
    import asyncio

    side_effects: list[str] = []
    crash_armed = {"v": True}
    run_id = "run_pg_resume_1"
    config = run_graph_config(run_id, recursion_limit=50)

    async def scenario() -> dict:
        store1 = PostgresCheckpointStateStore(_POSTGRES_DSN)
        await store1.ensure_schema()
        # Clean any prior row for a deterministic test.
        conn = await store1._connection()
        await conn.execute(
            "DELETE FROM deepagents_checkpoint_threads WHERE thread_id = %s", (run_id,)
        )
        saver1 = DurableCheckpointer(store1)
        app1 = _build_crashing_graph(side_effects, crash_armed).compile(checkpointer=saver1)
        crashed = False
        try:
            await app1.ainvoke({"steps": []}, config)
        except RuntimeError:
            crashed = True
        await store1.close()

        # Brand-new store + saver: only Postgres carries the state across.
        store2 = PostgresCheckpointStateStore(_POSTGRES_DSN)
        saver2 = DurableCheckpointer(store2)
        hydrated = await saver2.hydrate(run_id)
        app2 = _build_crashing_graph(side_effects, crash_armed).compile(checkpointer=saver2)
        result = await app2.ainvoke(None, config)
        conn2 = await store2._connection()
        await conn2.execute(
            "DELETE FROM deepagents_checkpoint_threads WHERE thread_id = %s", (run_id,)
        )
        await store2.close()
        return {"crashed": crashed, "hydrated": hydrated, "result": result}

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
