from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.reasoning_stream import ReasoningEventStreamer
from ultra_deepagents.runner import RunEventSequencer, _stream_agent_attempt


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
        await streamer.on_llm_new_token(
            "", chunk=_reasoning_chunk(long_thought), run_id=run_id
        )
        await streamer.on_llm_new_token(
            "", chunk=_reasoning_chunk("Settling on the answer."), run_id=run_id
        )
        await streamer.on_llm_new_token(
            "The", chunk=_content_chunk("The"), run_id=run_id
        )
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
        await streamer.on_llm_new_token(
            "", chunk=_reasoning_chunk("Thought."), run_id=run_id
        )
        await streamer.on_llm_end(None, run_id=run_id)

    asyncio.run(scenario())

    assert events_seen == 1


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

    assert any(
        isinstance(handler, ReasoningEventStreamer) for handler in agent.seen_callbacks
    )
    kinds = [event["event_kind"] for event in published]
    assert kinds == ["trace.reasoning.delta", "message.delta"]
    assert published[0]["payload"]["status"] == "completed"
    assert published[0]["payload"]["text"] == "Thinking about the request."
    assert published[1]["payload"]["text"] == "The answer is 42."
    # Sequences stay strictly increasing across both publish paths.
    assert [event["sequence"] for event in published] == [1, 2]


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
