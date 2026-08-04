"""map_task: deterministic batched subagent dispatch, specified test-first.

The headline scenario runs the SAME 12-item workload twice — serial `task`
calls versus one `map_task` — and asserts properties the serial path cannot
achieve: bounded coordinator context (results held out of context), ~2
coordinator turns instead of 13+, real bounded concurrency, and per-dispatch
event integrity (the guard-visibility property that justified building this
instead of adopting the interpreter).

Subagent turns run on the same scripted model (subagents inherit the loop
model); policies distinguish them by the ITEM marker in the incoming context.
"""

from __future__ import annotations

import json
import threading
import time

from longhorizon_harness import (
    CompressedConfig,
    LongHorizonWorld,
    TurnDecision,
    TurnRequest,
)
from longhorizon_invariants import (
    assert_event_stream_integrity,
    assert_terminal_success,
    turn_calls,
)

GOAL = "Analyze the twelve prepared samples and state the combined outcome plainly."
ITEM_MARKER = "SAMPLE-ITEM"


def _items(count: int) -> list[str]:
    return [f"{ITEM_MARKER} {index}: analyze this sample" for index in range(1, count + 1)]


def _is_subagent_turn(request: TurnRequest) -> bool:
    # Subagent contexts never contain the coordinator goal; matching on the
    # last message alone misfires when the COORDINATOR reads tool results that
    # quote item markers.
    return GOAL not in request.full_text


def _coordinator_calls(world: LongHorizonWorld) -> list:
    return [record for record in turn_calls(world) if GOAL in record.prompt_text]


def _subagent_reply(request: TurnRequest, *, size_chars: int = 220) -> TurnDecision:
    item = request.last_text.split(":", 1)[0].strip()
    text = (f"finding for {item}: nominal signal detected. " * max(1, size_chars // 40))[
        :size_chars
    ]
    return TurnDecision(text=text)


def test_map_task_beats_serial_on_turns_and_context(tmp_path) -> None:
    """THE decisive head-to-head. Serial dispatch cannot pass these bounds:
    every serial result re-enters coordinator context and each dispatch costs a
    coordinator model call; map_task pays one dispatch turn and one synthesis
    turn with results aggregated out-of-context."""
    items = _items(12)

    def run(world: LongHorizonWorld, use_map: bool) -> None:
        dispatched = {"count": 0}

        def policy(request: TurnRequest) -> TurnDecision:
            if _is_subagent_turn(request):
                return _subagent_reply(request)
            if use_map:
                if not request.saw("map_task result") and dispatched["count"] == 0:
                    dispatched["count"] += 1
                    return TurnDecision(
                        tool_name="map_task",
                        tool_args={
                            "subagent_type": "general-purpose",
                            "items": items,
                            "max_concurrency": 4,
                        },
                    )
                return TurnDecision(text="Combined outcome: all twelve samples nominal.")
            if dispatched["count"] < len(items):
                dispatched["count"] += 1
                return TurnDecision(
                    tool_name="task",
                    tool_args={
                        "subagent_type": "general-purpose",
                        "description": items[dispatched["count"] - 1],
                    },
                )
            return TurnDecision(text="Combined outcome: all twelve samples nominal.")

        world.run_sync(policy, goal=GOAL, config=CompressedConfig(context_window_tokens=60_000))

    serial_world = LongHorizonWorld(tmp=tmp_path / "serial", run_id="run-map-serial")
    run(serial_world, use_map=False)
    map_world = LongHorizonWorld(tmp=tmp_path / "mapped", run_id="run-map-batched")
    run(map_world, use_map=True)

    for world in (serial_world, map_world):
        assert_terminal_success(world.log)
        assert_event_stream_integrity(world.log)

    serial_coordinator = _coordinator_calls(serial_world)
    map_coordinator = _coordinator_calls(map_world)
    # Serial: one coordinator call per dispatch plus synthesis. Map: dispatch +
    # synthesis (+1 slack for framework variance).
    assert len(serial_coordinator) >= 13
    assert len(map_coordinator) <= 3, (
        f"map_task path used {len(map_coordinator)} coordinator calls — "
        f"the fan-out is not deterministic"
    )
    # Context economics: cumulative coordinator input must be dramatically
    # smaller when results do not re-enter context on every subsequent call.
    serial_tokens = sum(r.approx_conversation_tokens for r in serial_coordinator)
    map_tokens = sum(r.approx_conversation_tokens for r in map_coordinator)
    assert map_tokens < serial_tokens * 0.5, (
        f"map_task coordinator context ({map_tokens} tokens) is not materially "
        f"below serial ({serial_tokens} tokens) — aggregation is leaking into context"
    )
    # Both paths must actually have produced twelve subagent analyses.
    for world in (serial_world, map_world):
        subagent_calls = [
            record for record in turn_calls(world) if ITEM_MARKER in record.prompt_text
        ]
        assert len(subagent_calls) >= 12


def test_map_task_respects_concurrency_cap_with_real_parallelism(tmp_path) -> None:
    world = LongHorizonWorld(tmp=tmp_path, run_id="run-map-conc")
    lock = threading.Lock()
    active = {"now": 0, "max": 0}

    def policy(request: TurnRequest) -> TurnDecision:
        if _is_subagent_turn(request):
            with lock:
                active["now"] += 1
                active["max"] = max(active["max"], active["now"])
            time.sleep(0.25)
            with lock:
                active["now"] -= 1
            return _subagent_reply(request)
        if not request.saw("map_task result"):
            return TurnDecision(
                tool_name="map_task",
                tool_args={
                    "subagent_type": "general-purpose",
                    "items": _items(9),
                    "max_concurrency": 3,
                },
            )
        return TurnDecision(text="Combined outcome: nine samples nominal.")

    started = time.monotonic()
    world.run_sync(policy, goal=GOAL, config=CompressedConfig(context_window_tokens=60_000))
    elapsed = time.monotonic() - started

    assert_terminal_success(world.log)
    assert active["max"] <= 3, f"concurrency cap violated: {active['max']} simultaneous"
    assert active["max"] >= 2, "no real parallelism observed under the cap"
    # 9 items x 0.25s serially is ~2.3s of sleep alone; capped fan-out at 3
    # should land well under the serial floor even with framework overhead.
    assert elapsed < 60


def test_map_task_isolates_failures_and_preserves_order(tmp_path) -> None:
    world = LongHorizonWorld(tmp=tmp_path, run_id="run-map-errors")
    seen_result = {"payload": ""}

    def policy(request: TurnRequest) -> TurnDecision:
        if _is_subagent_turn(request):
            if f"{ITEM_MARKER} 3:" in request.last_text:
                raise RuntimeError("scripted subagent failure for item 3")
            return _subagent_reply(request)
        if not request.saw("map_task result"):
            return TurnDecision(
                tool_name="map_task",
                tool_args={
                    "subagent_type": "general-purpose",
                    "items": _items(6),
                    "max_concurrency": 2,
                },
            )
        seen_result["payload"] = request.last_text
        return TurnDecision(text="Combined outcome: five nominal, one failed sample.")

    world.run_sync(policy, goal=GOAL, config=CompressedConfig(context_window_tokens=60_000))

    assert_terminal_success(world.log)  # one bad item must never fail the run
    payload = seen_result["payload"]
    assert payload, "coordinator never saw the map_task result"
    # The aggregated result reports six slots in order with the failure marked.
    for index in (1, 2, 4, 5, 6):
        assert f'"index": {index}' in payload or f"item {index}" in payload.lower()
    assert "error" in payload.lower()
    assert "scripted subagent failure" in payload


def test_map_task_dispatches_are_visible_events(tmp_path) -> None:
    """The guard-visibility property: every dispatch produces normal-path
    events (usage per subagent model call, gapless sequencing) — the reason
    this exists instead of interpreter-side dispatch."""
    world = LongHorizonWorld(tmp=tmp_path, run_id="run-map-events")

    def policy(request: TurnRequest) -> TurnDecision:
        if _is_subagent_turn(request):
            return _subagent_reply(request)
        if not request.saw("map_task result"):
            return TurnDecision(
                tool_name="map_task",
                tool_args={
                    "subagent_type": "general-purpose",
                    "items": _items(5),
                    "max_concurrency": 2,
                },
            )
        return TurnDecision(text="Combined outcome: five samples nominal.")

    world.run_sync(policy, goal=GOAL, config=CompressedConfig(context_window_tokens=60_000))

    assert_terminal_success(world.log)
    assert_event_stream_integrity(world.log)
    usage_events = world.log.of_kind("run.token_usage")
    # Coordinator calls plus at least one model call per dispatched item.
    assert len(usage_events) >= 7
    map_tool_events = [
        event
        for event in world.log.tool_events("map_task")
        if (event.get("payload") or {}).get("status") in {"started", "completed"}
    ]
    assert len(map_tool_events) >= 2, "map_task itself must surface as a normal tool call"


def test_map_task_offloads_bulk_results_to_workspace_jsonl(tmp_path) -> None:
    world = LongHorizonWorld(tmp=tmp_path, run_id="run-map-offload")
    seen_result = {"payload": ""}

    def policy(request: TurnRequest) -> TurnDecision:
        if _is_subagent_turn(request):
            return _subagent_reply(request, size_chars=3000)
        if not request.saw("map_task result"):
            return TurnDecision(
                tool_name="map_task",
                tool_args={
                    "subagent_type": "general-purpose",
                    "items": _items(8),
                    "max_concurrency": 4,
                },
            )
        seen_result["payload"] = request.last_text
        return TurnDecision(text="Combined outcome: eight samples nominal.")

    world.run_sync(policy, goal=GOAL, config=CompressedConfig(context_window_tokens=60_000))

    assert_terminal_success(world.log)
    workspace = tmp_path / "workspaces" / world.run_id
    jsonl_files = list(workspace.glob("map_task/*.jsonl"))
    assert jsonl_files, "bulk results were not offloaded to a workspace JSONL"
    lines = jsonl_files[0].read_text().strip().splitlines()
    assert len(lines) == 8
    assert all("text" in json.loads(line) for line in lines)
    # The in-context payload stays compact: pointers + previews, not 8x3000 chars.
    assert len(seen_result["payload"]) < 6000, (
        f"map_task returned {len(seen_result['payload'])} chars into context — "
        f"bulk results must live in the JSONL, not the transcript"
    )
