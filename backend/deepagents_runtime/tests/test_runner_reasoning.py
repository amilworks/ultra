from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from ultra_deepagents.code_execution.progress import ExecuteProgressEvent
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.reasoning_stream import (
    ReasoningDegenerationError,
    ReasoningEventStreamer,
    ReasoningQualityGuard,
)
from ultra_deepagents.runner import (
    ExecuteProgressEventStreamer,
    ModelProtocolLeakError,
    RunEventSequencer,
    _stream_agent_attempt,
)


def _context() -> AgentRunContext:
    return AgentRunContext(
        assistant_id="assistant_test",
        org_id="org_test",
        user_id="user_test",
        project_id="project_test",
        thread_id="thread_reasoning_test",
        run_id="run_reasoning_test",
        goal="Explain the result.",
        workspace_root="/tmp/ws",
        artifact_root="/tmp/art",
    )


def _reasoning_chunk(text: str) -> ChatGenerationChunk:
    return ChatGenerationChunk(
        message=AIMessageChunk(content="", additional_kwargs={"reasoning_content": text})
    )


def _content_chunk(text: str) -> ChatGenerationChunk:
    return ChatGenerationChunk(message=AIMessageChunk(content=text))


def _streamer(published: list[dict[str, Any]]) -> ReasoningEventStreamer:
    async def publish_event(event: dict[str, Any]) -> None:
        published.append(event)

    return ReasoningEventStreamer(
        context=_context(),
        sequencer=RunEventSequencer("run_reasoning_test"),
        publish_event=publish_event,
    )


def test_streamer_coalesces_thinking_and_closes_on_first_answer_token():
    published: list[dict[str, Any]] = []
    streamer = _streamer(published)
    long_thought = "Weighing the estimator variance against bias. " * 5  # > 160 chars

    async def scenario() -> None:
        run_id = uuid4()
        await streamer.on_chat_model_start({}, [], run_id=run_id, metadata={})
        await streamer.on_llm_new_token("", chunk=_reasoning_chunk(long_thought), run_id=run_id)
        await streamer.on_llm_new_token(
            "", chunk=_reasoning_chunk("Settling on the answer."), run_id=run_id
        )
        await streamer.on_llm_new_token("The", chunk=_content_chunk("The"), run_id=run_id)
        await streamer.on_llm_end(None, run_id=run_id)

    asyncio.run(scenario())

    kinds = [event["event_kind"] for event in published]
    assert kinds == ["trace.reasoning.delta", "trace.reasoning.delta"]
    first, second = published
    assert first["payload"]["status"] == "running"
    assert first["payload"]["text"] == long_thought
    assert second["payload"]["status"] == "completed"
    assert second["payload"]["text"] == "Settling on the answer."
    assert [event["sequence"] for event in published] == [1, 2]
    assert streamer.observed_chars == len(long_thought) + len("Settling on the answer.")


def test_streamer_closes_round_on_model_end_before_tools():
    published: list[dict[str, Any]] = []
    streamer = _streamer(published)

    async def scenario() -> None:
        run_id = uuid4()
        await streamer.on_chat_model_start({}, [], run_id=run_id, metadata={})
        await streamer.on_llm_new_token(
            "", chunk=_reasoning_chunk("Need to check the profile first."), run_id=run_id
        )
        # Tool-calling rounds end without any visible content tokens.
        await streamer.on_llm_end(None, run_id=run_id)

    asyncio.run(scenario())

    assert [event["event_kind"] for event in published] == ["trace.reasoning.delta"]
    assert published[0]["payload"]["status"] == "completed"
    assert published[0]["payload"]["text"] == "Need to check the profile first."


def test_streamer_accepts_real_coordinator_callback_metadata():
    published: list[dict[str, Any]] = []
    streamer = _streamer(published)

    async def scenario() -> None:
        run_id = uuid4()
        # Metadata shape observed live from the deepagents coordinator graph.
        await streamer.on_chat_model_start(
            {},
            [],
            run_id=run_id,
            metadata={
                "lc_agent_name": "ultra-research-agent",
                "langgraph_node": "model",
                "langgraph_checkpoint_ns": "model:deddceff-29e3-3a52-0140-a1a58a08ca3b",
            },
        )
        await streamer.on_llm_new_token(
            "", chunk=_reasoning_chunk("Coordinator thinking."), run_id=run_id
        )
        await streamer.on_llm_end(None, run_id=run_id)

    asyncio.run(scenario())

    assert [event["event_kind"] for event in published] == ["trace.reasoning.delta"]


def test_streamer_ignores_subagent_model_calls():
    published: list[dict[str, Any]] = []
    streamer = _streamer(published)

    async def scenario() -> None:
        run_id = uuid4()
        await streamer.on_chat_model_start(
            {},
            [],
            run_id=run_id,
            metadata={"lc_agent_name": "code-runner"},
        )
        await streamer.on_llm_new_token(
            "", chunk=_reasoning_chunk("Subagent private thinking."), run_id=run_id
        )
        await streamer.on_llm_end(None, run_id=run_id)

    asyncio.run(scenario())

    assert published == []


def test_streamer_never_splices_overlapping_coordinator_calls():
    published: list[dict[str, Any]] = []
    streamer = _streamer(published)

    async def scenario() -> None:
        first_run = uuid4()
        second_run = uuid4()
        await streamer.on_chat_model_start({}, [], run_id=first_run, metadata={})
        await streamer.on_chat_model_start({}, [], run_id=second_run, metadata={})
        await streamer.on_llm_new_token(
            "", chunk=_reasoning_chunk("First model thought."), run_id=first_run
        )
        await streamer.on_llm_new_token(
            "", chunk=_reasoning_chunk("Second model thought."), run_id=second_run
        )
        await streamer.on_llm_end(None, run_id=second_run)
        await streamer.on_llm_end(None, run_id=first_run)

    asyncio.run(scenario())

    assert [event["payload"]["text"] for event in published] == [
        "Second model thought.",
        "First model thought.",
    ]
    assert all(event["payload"]["status"] == "completed" for event in published)


def test_reasoning_quality_guard_rejects_observed_dash_token_loop():
    guard = ReasoningQualityGuard()
    bad_fragment = "the — wait — the — — continue — " * 24

    async def scenario() -> None:
        run_id = uuid4()
        await guard.on_chat_model_start({}, [], run_id=run_id, metadata={})
        for _ in range(8):
            await guard.on_llm_new_token(
                "",
                chunk=_reasoning_chunk(bad_fragment),
                run_id=run_id,
            )

    with pytest.raises(ReasoningDegenerationError) as exc_info:
        asyncio.run(scenario())

    assert exc_info.value.verdict.signal == "dashlike_density"
    assert exc_info.value.verdict.window_chars >= 512
    assert exc_info.value.verdict.dashlike_count >= 32
    assert exc_info.value.reasoning_chars >= 512


def test_reasoning_quality_guard_rejects_observed_retrieval_token_loop():
    guard = ReasoningQualityGuard()
    repeated = ("Retrieval Retrieval Retrieval Retrieval | Delta " * 180).strip()

    async def scenario() -> None:
        run_id = uuid4()
        await guard.on_chat_model_start({}, [], run_id=run_id, metadata={})
        await guard.on_llm_new_token(
            "",
            chunk=_reasoning_chunk(repeated),
            run_id=run_id,
        )

    with pytest.raises(ReasoningDegenerationError) as exc_info:
        asyncio.run(scenario())

    assert exc_info.value.verdict.signal == "lexical_repetition"
    assert exc_info.value.verdict.max_repeated_trigram >= 48
    assert exc_info.value.verdict.token_diversity <= 0.08
    assert exc_info.value.reasoning_chars >= 2048


def test_reasoning_quality_guard_allows_coherent_scientific_reasoning():
    guard = ReasoningQualityGuard()
    coherent = (
        "Compare the measured estimate with the independent calculation, retain units, "
        "and report the uncertainty. The first result—while preliminary—agrees with the "
        "replicate. "
    ) * 30

    async def scenario() -> None:
        run_id = uuid4()
        await guard.on_chat_model_start({}, [], run_id=run_id, metadata={})
        await guard.on_llm_new_token(
            "",
            chunk=_reasoning_chunk(coherent),
            run_id=run_id,
        )
        await guard.on_llm_end(None, run_id=run_id)

    asyncio.run(scenario())


def test_streamer_swallows_publish_failures():
    events_seen = 0

    async def failing_publish(_event: dict[str, Any]) -> None:
        nonlocal events_seen
        events_seen += 1
        raise RuntimeError("nats down")

    streamer = ReasoningEventStreamer(
        context=_context(),
        sequencer=RunEventSequencer("run_reasoning_test"),
        publish_event=failing_publish,
    )

    async def scenario() -> None:
        run_id = uuid4()
        await streamer.on_chat_model_start({}, [], run_id=run_id, metadata={})
        await streamer.on_llm_new_token("", chunk=_reasoning_chunk("Thought."), run_id=run_id)
        await streamer.on_llm_end(None, run_id=run_id)

    asyncio.run(scenario())

    assert events_seen == 1


def test_execute_progress_streamer_publishes_threaded_tool_progress():
    published: list[dict[str, Any]] = []

    async def publish_event(event: dict[str, Any]) -> None:
        published.append(event)

    async def scenario() -> None:
        streamer = ExecuteProgressEventStreamer(
            context=_context(),
            sequencer=RunEventSequencer("run_reasoning_test"),
            publish_event=publish_event,
        )
        streamer.bind_tool_event(
            {
                "event_kind": "tool_call.started",
                "payload": {
                    "tool_name": "execute",
                    "status": "started",
                    "tool_call_id": "call-exec-1",
                },
                "agent_role": "builder",
                "scope_id": "builder:execute",
            }
        )
        await asyncio.to_thread(
            streamer.emit_sync,
            ExecuteProgressEvent(
                command="python run_full_experiment.py",
                stream="stdout",
                text="[17/192] condition ok",
                elapsed_seconds=42.5,
                output_size_chars=2048,
                progress_index=3,
                suppressed_line_count=2,
            ),
        )

    asyncio.run(scenario())

    assert [event["event_kind"] for event in published] == ["tool_call.progress"]
    event = published[0]
    assert event["sequence"] == 1
    assert event["payload"]["tool_name"] == "execute"
    assert event["payload"]["status"] == "progress"
    assert event["payload"]["tool_call_id"] == "call-exec-1"
    assert event["payload"]["text"] == "[17/192] condition ok"
    assert event["payload"]["stream"] == "stdout"
    assert event["payload"]["command_preview"] == "python run_full_experiment.py"
    assert event["payload"]["progress_index"] == 3
    assert event["payload"]["suppressed_line_count"] == 2
    assert event["agent_role"] == "builder"
    assert event["scope_id"] == "builder:execute"


class CallbackDrivingAgent:
    """Fake agent that drives the config callbacks the way LangGraph does:
    model-chunk callbacks fire while the graph stream yields message events."""

    def __init__(self) -> None:
        self.seen_callbacks: list[Any] = []

    def astream_events(self, _payload: Any, *, config: Any = None, **_kwargs: Any) -> Any:
        callbacks = list((config or {}).get("callbacks") or [])
        self.seen_callbacks = callbacks

        async def generate() -> Any:
            run_id = uuid4()
            for handler in callbacks:
                if hasattr(handler, "on_chat_model_start"):
                    await handler.on_chat_model_start({}, [], run_id=run_id, metadata={})
            for handler in callbacks:
                if hasattr(handler, "on_llm_new_token"):
                    await handler.on_llm_new_token(
                        "",
                        chunk=_reasoning_chunk("Thinking about the request."),
                        run_id=run_id,
                    )
            for handler in callbacks:
                if hasattr(handler, "on_llm_end"):
                    await handler.on_llm_end(None, run_id=run_id)
            yield {
                "event": "on_chat_model_stream",
                "data": {"chunk": AIMessageChunk(content="The answer is 42.")},
                "metadata": {},
            }

        return generate()


def test_stream_agent_attempt_publishes_reasoning_before_answer():
    published: list[dict[str, Any]] = []

    async def publish_event(event: dict[str, Any]) -> None:
        published.append(event)

    agent = CallbackDrivingAgent()

    async def scenario() -> None:
        await _stream_agent_attempt(
            agent,
            messages=[{"role": "user", "content": "Explain the result."}],
            context=_context(),
            sequencer=RunEventSequencer("run_reasoning_test"),
            publish_event=publish_event,
        )

    asyncio.run(scenario())

    assert any(isinstance(handler, ReasoningEventStreamer) for handler in agent.seen_callbacks)
    kinds = [event["event_kind"] for event in published]
    assert kinds == ["trace.reasoning.delta", "message.delta"]
    assert published[0]["payload"]["status"] == "completed"
    assert published[0]["payload"]["text"] == "Thinking about the request."
    assert published[1]["payload"]["text"] == "The answer is 42."
    # Sequences stay strictly increasing across both publish paths.
    assert [event["sequence"] for event in published] == [1, 2]


def test_stream_agent_attempt_rejects_split_deepseek_protocol_before_it_reaches_ui():
    published: list[dict[str, Any]] = []

    async def publish_event(event: dict[str, Any]) -> None:
        published.append(event)

    class _MalformedDeepSeekProtocolAgent:
        def astream_events(self, _payload: Any, **_kwargs: Any) -> Any:
            async def generate() -> Any:
                for text in (
                    "Computation verified. Now the figures.\n\n</｜DS",
                    'ML｜tool_calls>\n<｜DSML｜invoke name="execute">',
                ):
                    yield {
                        "event": "on_chat_model_stream",
                        "data": {"chunk": AIMessageChunk(content=text)},
                        "metadata": {},
                    }

            return generate()

    async def scenario() -> None:
        await _stream_agent_attempt(
            _MalformedDeepSeekProtocolAgent(),
            messages=[{"role": "user", "content": "Explain delta attention."}],
            context=_context(),
            sequencer=RunEventSequencer("run_protocol_leak_test"),
            publish_event=publish_event,
        )

    with pytest.raises(ModelProtocolLeakError) as exc_info:
        asyncio.run(scenario())

    assert exc_info.value.protocol == "deepseek_dsml"
    visible = "".join(
        str(event.get("payload", {}).get("text") or "")
        for event in published
        if event.get("event_kind") == "message.delta"
    )
    assert visible == "Computation verified. Now the figures.\n\n"
    assert "DSML" not in visible
    assert "tool_calls" not in visible


def test_stream_agent_attempt_rejects_protocol_found_only_in_final_state():
    published: list[dict[str, Any]] = []

    async def publish_event(event: dict[str, Any]) -> None:
        published.append(event)

    class _MalformedFinalStateAgent:
        def astream_events(self, _payload: Any, **_kwargs: Any) -> Any:
            async def generate() -> Any:
                yield {
                    "type": "event",
                    "method": "values",
                    "params": {
                        "namespace": [],
                        "data": {
                            "messages": [
                                {"role": "user", "content": "Return the calculation."},
                                {
                                    "role": "assistant",
                                    "content": "</｜DSML｜tool_calls>",
                                },
                            ]
                        },
                    },
                }

            return generate()

    async def scenario() -> None:
        await _stream_agent_attempt(
            _MalformedFinalStateAgent(),
            messages=[{"role": "user", "content": "Return the calculation."}],
            context=_context(),
            sequencer=RunEventSequencer("run_protocol_final_state_test"),
            publish_event=publish_event,
        )

    with pytest.raises(ModelProtocolLeakError):
        asyncio.run(scenario())

    assert not any(event.get("event_kind") == "message.delta" for event in published)


def test_idle_watchdog_fires_during_an_active_tool_call():
    """Regression for the 1h43m hang: a tool that STARTS but never finishes (the live
    33-started/32-completed signature) must still trip the idle watchdog. The old
    `and not active_tool_calls` gate disarmed the watchdog for exactly this case."""
    import pytest
    from ultra_deepagents.runner import AgentStreamIdleTimeoutError

    published: list[dict[str, Any]] = []

    async def publish_event(event: dict[str, Any]) -> None:
        published.append(event)

    class _ToolThenHangAgent:
        def astream_events(self, _payload: Any, *, config: Any = None, **_kwargs: Any) -> Any:
            async def generate() -> Any:
                # a tool call STARTS -> active_tool_calls becomes non-empty
                yield {
                    "event": "on_tool_start",
                    "name": "read_file",
                    "run_id": str(uuid4()),
                    "data": {"input": {"path": "/workspace/x"}},
                    "metadata": {},
                }
                # ...then the tool never returns: the stream goes permanently silent
                # (an uncancellable to_thread blocked in httpx, exactly the live incident).
                await asyncio.Event().wait()
                yield {}  # unreachable

            return generate()

    async def scenario() -> None:
        await _stream_agent_attempt(
            _ToolThenHangAgent(),
            messages=[{"role": "user", "content": "read it"}],
            context=_context(),
            sequencer=RunEventSequencer("run_hang_test"),
            publish_event=publish_event,
            model_stream_idle_timeout_seconds=0.5,  # short for the test; prod is 3600
        )

    with pytest.raises(AgentStreamIdleTimeoutError):
        asyncio.run(scenario())

    # The tool genuinely STARTED (active_tool_calls was populated), proving the watchdog
    # fired DURING an active tool call — not merely on an already-empty stream.
    assert any(e.get("event_kind") == "tool_call.started" for e in published)


def test_model_output_idle_watchdog_recovers_reasoning_only_dead_open_stream():
    """Reasoning callbacks are activity even though v3 does not yield them.

    Once that output stops, a provider connection that stays open must recover
    on the short model-output bound instead of waiting for the one-hour hard
    stream ceiling.
    """
    import pytest
    from ultra_deepagents.runner import AgentStreamIdleTimeoutError

    published: list[dict[str, Any]] = []

    async def publish_event(event: dict[str, Any]) -> None:
        published.append(event)

    class _ReasoningThenDeadOpenAgent:
        def astream_events(self, _payload: Any, *, config: Any = None, **_kwargs: Any) -> Any:
            callbacks = list((config or {}).get("callbacks") or [])

            async def generate() -> Any:
                run_id = uuid4()
                for handler in callbacks:
                    if hasattr(handler, "on_chat_model_start"):
                        await handler.on_chat_model_start({}, [], run_id=run_id, metadata={})
                for handler in callbacks:
                    if hasattr(handler, "on_llm_new_token"):
                        await handler.on_llm_new_token(
                            "",
                            chunk=_reasoning_chunk("Reasoning without a final answer. " * 8),
                            run_id=run_id,
                        )
                await asyncio.Event().wait()
                yield {}  # unreachable

            return generate()

    async def scenario() -> None:
        await _stream_agent_attempt(
            _ReasoningThenDeadOpenAgent(),
            messages=[{"role": "user", "content": "Explain delta attention."}],
            context=_context(),
            sequencer=RunEventSequencer("run_dead_open_reasoning_test"),
            publish_event=publish_event,
            model_stream_idle_timeout_seconds=1.0,
            model_output_idle_timeout_seconds=0.05,
        )

    with pytest.raises(AgentStreamIdleTimeoutError) as exc_info:
        asyncio.run(scenario())

    assert exc_info.value.timeout_seconds == pytest.approx(0.05)
    assert any(event.get("event_kind") == "trace.reasoning.delta" for event in published)


def test_model_output_idle_watchdog_does_not_cut_off_active_tool():
    published: list[dict[str, Any]] = []

    async def publish_event(event: dict[str, Any]) -> None:
        published.append(event)

    class _ReasoningToolThenAnswerAgent:
        def astream_events(self, _payload: Any, *, config: Any = None, **_kwargs: Any) -> Any:
            callbacks = list((config or {}).get("callbacks") or [])

            async def generate() -> Any:
                run_id = uuid4()
                for handler in callbacks:
                    if hasattr(handler, "on_chat_model_start"):
                        await handler.on_chat_model_start({}, [], run_id=run_id, metadata={})
                for handler in callbacks:
                    if hasattr(handler, "on_llm_new_token"):
                        await handler.on_llm_new_token(
                            "",
                            chunk=_reasoning_chunk("I should run the calculation. " * 8),
                            run_id=run_id,
                        )
                yield {
                    "event": "on_tool_start",
                    "name": "execute",
                    "run_id": "tool-call-1",
                    "data": {"input": {"command": "python calculation.py"}},
                    "metadata": {},
                }
                await asyncio.sleep(0.08)
                yield {
                    "event": "on_tool_end",
                    "name": "execute",
                    "run_id": "tool-call-1",
                    "data": {"output": "42"},
                    "metadata": {},
                }
                yield {
                    "event": "on_chat_model_stream",
                    "data": {"chunk": AIMessageChunk(content="The computed answer is 42.")},
                    "metadata": {},
                }

            return generate()

    async def scenario() -> Any:
        return await _stream_agent_attempt(
            _ReasoningToolThenAnswerAgent(),
            messages=[{"role": "user", "content": "Compute the answer."}],
            context=_context(),
            sequencer=RunEventSequencer("run_active_tool_output_idle_test"),
            publish_event=publish_event,
            model_stream_idle_timeout_seconds=1.0,
            model_output_idle_timeout_seconds=0.02,
        )

    result = asyncio.run(scenario())

    assert result.streamed_response_text == "The computed answer is 42."
    assert [
        event["event_kind"]
        for event in published
        if str(event.get("event_kind") or "").startswith("tool_call.")
    ] == ["tool_call.started", "tool_call.completed"]
    assert not [event for event in published if event.get("event_kind") == "run.failed"]


def test_model_output_idle_watchdog_rearms_after_last_tool_completes():
    import pytest
    from ultra_deepagents.runner import AgentStreamIdleTimeoutError

    published: list[dict[str, Any]] = []

    async def publish_event(event: dict[str, Any]) -> None:
        published.append(event)

    class _LongToolThenDeadOpenAgent:
        def astream_events(self, _payload: Any, *, config: Any = None, **_kwargs: Any) -> Any:
            callbacks = list((config or {}).get("callbacks") or [])

            async def generate() -> Any:
                run_id = uuid4()
                for handler in callbacks:
                    if hasattr(handler, "on_chat_model_start"):
                        await handler.on_chat_model_start({}, [], run_id=run_id, metadata={})
                for handler in callbacks:
                    if hasattr(handler, "on_llm_new_token"):
                        await handler.on_llm_new_token(
                            "",
                            chunk=_reasoning_chunk("I should run the long calculation. " * 8),
                            run_id=run_id,
                        )
                yield {
                    "event": "on_tool_start",
                    "name": "execute",
                    "run_id": "tool-call-then-dead-open",
                    "data": {"input": {"command": "python calculation.py"}},
                    "metadata": {},
                }
                # This exceeds the short deadline but remains protected while
                # the tool is active.
                await asyncio.sleep(0.08)
                yield {
                    "event": "on_tool_end",
                    "name": "execute",
                    "run_id": "tool-call-then-dead-open",
                    "data": {"output": "42"},
                    "metadata": {},
                }
                # The provider never resumes after receiving the tool result.
                await asyncio.Event().wait()
                yield {}  # unreachable

            return generate()

    async def scenario() -> None:
        await _stream_agent_attempt(
            _LongToolThenDeadOpenAgent(),
            messages=[{"role": "user", "content": "Compute the answer."}],
            context=_context(),
            sequencer=RunEventSequencer("run_post_tool_dead_open_test"),
            publish_event=publish_event,
            model_stream_idle_timeout_seconds=1.0,
            model_output_idle_timeout_seconds=0.02,
        )

    with pytest.raises(AgentStreamIdleTimeoutError) as exc_info:
        asyncio.run(scenario())

    assert exc_info.value.timeout_seconds == pytest.approx(0.02)
    assert exc_info.value.idle_scope == "model_output"
    assert [
        event["event_kind"]
        for event in published
        if str(event.get("event_kind") or "").startswith("tool_call.")
    ] == ["tool_call.started", "tool_call.completed"]


def test_model_output_idle_watchdog_tracks_continuing_raw_reasoning():
    published: list[dict[str, Any]] = []

    async def publish_event(event: dict[str, Any]) -> None:
        published.append(event)

    class _LongReasoningThenAnswerAgent:
        def astream_events(self, _payload: Any, *, config: Any = None, **_kwargs: Any) -> Any:
            callbacks = list((config or {}).get("callbacks") or [])

            async def generate() -> Any:
                run_id = uuid4()
                for handler in callbacks:
                    if hasattr(handler, "on_chat_model_start"):
                        await handler.on_chat_model_start({}, [], run_id=run_id, metadata={})
                for index in range(6):
                    for handler in callbacks:
                        if hasattr(handler, "on_llm_new_token"):
                            await handler.on_llm_new_token(
                                "",
                                chunk=_reasoning_chunk(f"reasoning chunk {index} "),
                                run_id=run_id,
                            )
                    await asyncio.sleep(0.02)
                for handler in callbacks:
                    if hasattr(handler, "on_llm_end"):
                        await handler.on_llm_end(None, run_id=run_id)
                yield {
                    "event": "on_chat_model_stream",
                    "data": {"chunk": AIMessageChunk(content="Reasoning completed safely.")},
                    "metadata": {},
                }

            return generate()

    async def scenario() -> Any:
        return await _stream_agent_attempt(
            _LongReasoningThenAnswerAgent(),
            messages=[{"role": "user", "content": "Think carefully."}],
            context=_context(),
            sequencer=RunEventSequencer("run_continuing_reasoning_test"),
            publish_event=publish_event,
            model_stream_idle_timeout_seconds=1.0,
            model_output_idle_timeout_seconds=0.04,
        )

    result = asyncio.run(scenario())

    assert result.streamed_response_text == "Reasoning completed safely."
    assert any(event.get("event_kind") == "trace.reasoning.delta" for event in published)


def test_thinking_fallback_contextvar_survives_child_tasks_and_resets():
    """The runner arms the fallback in the recovery handler and the retry
    attempt's child tasks / to_thread hops must observe it (contextvar
    propagation), while a fresh run's reset must clear it."""
    import asyncio

    from ultra_deepagents.model import (
        arm_thinking_fallback,
        reset_thinking_fallback,
        thinking_fallback_armed,
    )

    async def scenario() -> tuple[bool, bool, bool]:
        reset_thinking_fallback()
        before = thinking_fallback_armed()
        arm_thinking_fallback()

        async def child() -> bool:
            return thinking_fallback_armed()

        in_child_task = await asyncio.create_task(child())
        in_thread = await asyncio.to_thread(thinking_fallback_armed)
        reset_thinking_fallback()
        return before, in_child_task and in_thread, thinking_fallback_armed()

    before, propagated, after_reset = asyncio.run(scenario())
    assert before is False
    assert propagated is True
    assert after_reset is False


def test_bare_tool_markup_flood_trips_protocol_guard():
    """The production leak variant WITHOUT the DSML prefix — floods of bare
    invoke/tool_calls tags — must trip the guard even though no DSML marker
    ever appears."""
    from ultra_deepagents.runner import ModelProtocolLeakError, _VisibleModelProtocolGuard

    guard = _VisibleModelProtocolGuard()
    # Condensed verbatim shape from the observed production response.
    flood = (
        '<invoke name="read_file">\n</invoke>\n' * 6
        + "</tool_calls>\n<tool>\n</tool>\n</invoke>\n"
    )
    with pytest.raises(ModelProtocolLeakError) as excinfo:
        for chunk in [flood[i : i + 37] for i in range(0, len(flood), 37)]:
            guard.feed(chunk)
    assert excinfo.value.protocol == "bare_tool_markup"
    assert excinfo.value.signal == "tool_markup_flood"


def test_bare_tool_markup_validate_catches_trailing_flood_in_long_answer():
    from ultra_deepagents.runner import ModelProtocolLeakError, _VisibleModelProtocolGuard

    long_valid_answer = ("The delta rule updates W by an outer product correction. " * 200)
    flood = '<invoke name="read_file">\n</invoke>\n' * 10
    with pytest.raises(ModelProtocolLeakError):
        _VisibleModelProtocolGuard.validate(long_valid_answer + flood)


def test_isolated_tool_markup_mentions_do_not_trip_guard():
    """Prose that legitimately QUOTES a tag or two must stream unharmed."""
    from ultra_deepagents.runner import _VisibleModelProtocolGuard

    guard = _VisibleModelProtocolGuard()
    prose = (
        "In Anthropic's dialect a call looks like <invoke name=\"read_file\"> "
        "followed by </invoke>, whereas OpenAI uses JSON tool_calls. "
        "The retrieval MAE was 4.07 for the delta rule versus 5.62 additive, "
        "a ratio of 1.38, computed over 64 probes with d=32. " * 8
    )
    released = "".join(guard.feed(prose[i : i + 53]) for i in range(0, len(prose), 53))
    released += guard.finish()
    assert released == prose
    _VisibleModelProtocolGuard.validate(prose)


def test_scientific_html_and_inequalities_do_not_trip_guard():
    from ultra_deepagents.runner import _VisibleModelProtocolGuard

    guard = _VisibleModelProtocolGuard()
    prose = (
        "For d<32 the bound tightens; note x < y > z comparisons and HTML like "
        "<table><tr><td>4.07</td></tr></table> or <b>bold</b> markup. " * 20
    )
    released = "".join(guard.feed(prose[i : i + 41]) for i in range(0, len(prose), 41))
    released += guard.finish()
    assert released == prose
    _VisibleModelProtocolGuard.validate(prose)
