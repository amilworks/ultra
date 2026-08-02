"""Regression tests for final-response extraction.

Pins the fix for the truncated-response incident: a turn shaped
``report -> write_todos (bookkeeping) -> short coda`` must persist the report,
not just the coda (previously only the last AIMessage / the post-tool stream
buffer survived, both of which held only the coda).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.runner import (
    _choose_response_text,
    _response_text_from_final_state,
    run_job,
)
from ultra_deepagents.schemas import RunJobEnvelope

REPORT = "## Results\n\nThe full report body with figure links and analysis."
CODA = "The outputs are at the paths listed above."


def test_final_state_joins_report_and_coda_across_write_todos_dicts():
    state = {
        "messages": [
            {"role": "user", "content": "Write the report."},
            {"role": "assistant", "content": "Working on it."},
            {"role": "tool", "name": "execute", "content": "ok"},
            {"role": "assistant", "content": REPORT},
            {"role": "tool", "name": "write_todos", "content": "Updated todo list"},
            {"role": "assistant", "content": CODA},
        ]
    }
    assert _response_text_from_final_state(state) == f"{REPORT}\n\n{CODA}"


def test_final_state_joins_report_and_coda_across_write_todos_langchain():
    state = {
        "messages": [
            HumanMessage(content="Write the report."),
            AIMessage(content="Working on it.", tool_calls=[]),
            ToolMessage(content="ok", name="execute", tool_call_id="call-1"),
            AIMessage(content=REPORT),
            ToolMessage(content="Updated todo list", name="write_todos", tool_call_id="call-2"),
            AIMessage(content=CODA),
        ]
    }
    assert _response_text_from_final_state(state) == f"{REPORT}\n\n{CODA}"


def test_final_state_stops_at_substantive_tool_boundary():
    state = {
        "messages": [
            {"role": "user", "content": "Do the thing."},
            {"role": "assistant", "content": "Let me run the script first."},
            {"role": "tool", "name": "execute", "content": "script output"},
            {"role": "assistant", "content": CODA},
        ]
    }
    assert _response_text_from_final_state(state) == CODA


def test_final_state_single_assistant_message_unchanged():
    state = {
        "messages": [
            {"role": "user", "content": "Hello?"},
            {"role": "assistant", "content": "Hi there."},
        ]
    }
    assert _response_text_from_final_state(state) == "Hi there."


def test_final_state_trailing_bookkeeping_without_coda_keeps_report():
    state = {
        "messages": [
            {"role": "user", "content": "Write the report."},
            {"role": "assistant", "content": REPORT},
            {"role": "tool", "name": "write_todos", "content": "Updated todo list"},
        ]
    }
    assert _response_text_from_final_state(state) == REPORT


def test_final_state_empty_returns_empty():
    assert _response_text_from_final_state(None) == ""
    assert _response_text_from_final_state({"messages": []}) == ""


def test_choose_response_prefers_fuller_post_tool_when_final_is_short_suffix():
    chosen = _choose_response_text(
        final_response_text=CODA,
        streamed_response_text=f"early narration {REPORT} {CODA}",
        post_tool_streamed_response_text=f"{REPORT}\n\n{CODA}",
        artifact_events=[],
    )
    assert chosen == f"{REPORT}\n\n{CODA}"


def test_choose_response_keeps_final_when_not_a_suffix():
    chosen = _choose_response_text(
        final_response_text="A distinct final answer.",
        streamed_response_text="something else entirely that is much longer",
        post_tool_streamed_response_text="something else entirely that is much longer",
        artifact_events=[],
    )
    assert chosen == "A distinct final answer."


def test_choose_response_keeps_final_when_comparable_length():
    final = f"{REPORT}\n\n{CODA}"
    chosen = _choose_response_text(
        final_response_text=final,
        streamed_response_text=final,
        post_tool_streamed_response_text=f"{REPORT} {CODA}",
        artifact_events=[],
    )
    assert chosen == final


def test_choose_response_empty_final_falls_back_in_priority_order():
    chosen = _choose_response_text(
        final_response_text="  ",
        streamed_response_text="full stream",
        post_tool_streamed_response_text="post tool text",
        artifact_events=[],
    )
    assert chosen == "post tool text"
    chosen = _choose_response_text(
        final_response_text="",
        streamed_response_text="full stream",
        post_tool_streamed_response_text="",
        artifact_events=[],
    )
    assert chosen == "full stream"


class FakeReportTodoCodaAgent:
    """Streams the exact turn shape from the truncated-response incident."""

    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"

        def text_delta(text: str):
            return {
                "type": "event",
                "method": "messages",
                "params": {
                    "namespace": [],
                    "data": [
                        {
                            "event": "content-block-delta",
                            "index": 0,
                            "delta": {"type": "text-delta", "text": text},
                        },
                        {"lc_agent_name": "ultra-research-agent"},
                    ],
                },
            }

        yield text_delta(REPORT)
        for status in ("started", "completed"):
            yield {
                "type": "event",
                "method": "tools",
                "params": {
                    "namespace": [],
                    "data": {
                        "event": status,
                        "tool_call_id": "todos-1",
                        "tool_name": "write_todos",
                        "input": {"todos": []},
                        **({"output": "Updated todo list"} if status == "completed" else {}),
                    },
                },
            }
        yield text_delta(CODA)
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        {"role": "user", "content": payload["messages"][0]["content"]},
                        {"role": "assistant", "content": REPORT},
                        {"role": "tool", "name": "write_todos", "content": "Updated todo list"},
                        {"role": "assistant", "content": CODA},
                    ]
                },
            },
        }


def test_run_job_persists_report_across_trailing_write_todos(tmp_path: Path):
    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        workspace_root=str(tmp_path / "workspaces"),
        artifact_root=str(tmp_path / "artifacts"),
    )
    job = RunJobEnvelope(
        run_id="run-1",
        thread_id="thread-1",
        user_id="researcher-1",
        goal="Write the report.",
        messages=[{"role": "user", "content": "Write the report."}],
    )
    published = []

    async def publish(event):
        published.append(event)

    asyncio.run(
        run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: FakeReportTodoCodaAgent(),
        )
    )

    completed = published[-1]
    assert completed["event_kind"] == "run.completed"
    response_text = completed["payload"]["response_text"]
    assert REPORT in response_text
    assert CODA in response_text
    tool_events = [e for e in published if e.get("event_type") == "tool_call"]
    assert [e["payload"]["tool_name"] for e in tool_events] == ["write_todos", "write_todos"]


ANALYSIS = (
    "## Scientific analysis\n\n"
    "The Lyapunov spectrum indicates chaotic dynamics across the full "
    "parameter sweep, with detailed per-regime interpretation."
)
GUARD_CODA = "Added the requested figure."


class FakeAnalysisThenFigureDeltaAgent:
    """Attempt 1 answers with a full analysis but forgets the requested figure;
    the completion-guard continuation (attempt 2) saves it and replies with
    only the delta. Models the cross-attempt truncation incident."""

    def __init__(self) -> None:
        self.calls = 0

    async def astream_events(self, payload, *, context=None, version=None):
        assert version == "v3"
        self.calls += 1
        if self.calls == 1:
            yield {
                "type": "event",
                "method": "values",
                "params": {
                    "namespace": [],
                    "data": {
                        "messages": [
                            {"role": "user", "content": payload["messages"][0]["content"]},
                            {"role": "assistant", "content": ANALYSIS},
                        ]
                    },
                },
            }
            return
        workspace = Path(context.workspace_root)
        (workspace / "lyapunov.png").write_bytes(b"\x89PNG\r\n\x1a\nspectrum")
        yield {
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "data": {
                    "messages": [
                        {"role": "user", "content": payload["messages"][-1]["content"]},
                        {"role": "assistant", "content": GUARD_CODA},
                    ]
                },
            },
        }


def test_run_job_keeps_earlier_attempt_answer_across_completion_guard(tmp_path: Path):
    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        workspace_root=str(tmp_path / "workspaces"),
        artifact_root=str(tmp_path / "artifacts"),
        completion_max_continuations=1,
    )
    goal = "Analyze the system dynamics and save a plot of the Lyapunov spectrum."
    job = RunJobEnvelope(
        run_id="run-1",
        thread_id="thread-1",
        user_id="researcher-1",
        goal=goal,
        messages=[{"role": "user", "content": goal}],
    )
    fake_agent = FakeAnalysisThenFigureDeltaAgent()
    published = []

    async def publish(event):
        published.append(event)

    result = asyncio.run(
        run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: fake_agent,
        )
    )

    assert fake_agent.calls == 2
    completed = published[-1]
    assert completed["event_kind"] == "run.completed"
    response_text = completed["payload"]["response_text"]
    # The UI must receive attempt 1's analysis AND attempt 2's delta, in order.
    assert ANALYSIS in response_text
    assert GUARD_CODA in response_text
    assert response_text.index(ANALYSIS) < response_text.index(GUARD_CODA)
    assert result == response_text
