from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.events import (
    normalize_artifact_created,
    normalize_message_delta,
    normalize_run_completed,
    normalize_run_failed,
    normalize_subagent_message_delta,
    normalize_tool_call,
)


def _context() -> AgentRunContext:
    return AgentRunContext(
        assistant_id="ultra-research",
        org_id="allen",
        user_id="researcher-1",
        project_id="project-1",
        thread_id="thread-1",
        run_id="run-1",
    )


def test_normalize_message_delta_matches_go_event_shape():
    event = normalize_message_delta(_context(), "hello", source="coordinator")

    assert event["run_id"] == "run-1"
    assert event["thread_id"] == "thread-1"
    assert event["event_kind"] == "message.delta"
    assert event["message"] == "hello"
    assert event["node_name"] == "coordinator"
    assert event["agent_role"] == "coordinator"
    assert event["level"] == "info"
    assert event["payload"] == {"text": "hello", "source": "coordinator"}
    assert event["ts"].endswith("Z")


def test_normalize_tool_call_includes_tool_status_payload():
    event = normalize_tool_call(
        _context(),
        tool_name="execute",
        status="started",
        payload={"command": "python analysis.py"},
    )

    assert event["event_kind"] == "tool_call.started"
    assert event["node_name"] == "tool:execute"
    assert event["payload"]["tool_name"] == "execute"
    assert event["payload"]["status"] == "started"
    assert event["payload"]["command"] == "python analysis.py"


def test_normalize_subagent_message_delta_keeps_specialist_text_separate():
    event = normalize_subagent_message_delta(
        _context(),
        name="literature-reviewer",
        text="Found the methods section.",
        task_id="task-paper-1",
        payload={"namespace": ["literature-reviewer"]},
    )

    assert event["event_kind"] == "subagent.message.delta"
    assert event["event_type"] == "subagent_message"
    assert event["node_name"] == "literature-reviewer"
    assert event["task_id"] == "task-paper-1"
    assert event["agent_role"] == "literature-reviewer"
    assert event["message"] == "Found the methods section."
    assert event["payload"]["text"] == "Found the methods section."
    assert event["payload"]["source"] == "literature-reviewer"
    assert event["payload"]["namespace"] == ["literature-reviewer"]


def test_normalize_artifact_created_is_go_compatible():
    event = normalize_artifact_created(
        _context(),
        artifact_id="artifact-1",
        path="/workspace/outputs/plot.png",
        title="QC Plot",
        payload={"mime_type": "image/png"},
    )

    assert event["event_kind"] == "artifact.created"
    assert event["node_name"] == "artifact"
    assert event["payload"]["artifact_id"] == "artifact-1"
    assert event["payload"]["path"] == "/workspace/outputs/plot.png"
    assert event["payload"]["title"] == "QC Plot"
    assert event["payload"]["mime_type"] == "image/png"


def test_normalize_run_completed_and_failed_set_levels():
    completed = normalize_run_completed(_context(), "final answer")
    failed = normalize_run_failed(_context(), RuntimeError("boom"))

    assert completed["event_kind"] == "run.completed"
    assert completed["message"] == "final answer"
    assert completed["level"] == "info"
    assert completed["payload"] == {"response_text": "final answer"}

    assert failed["event_kind"] == "run.failed"
    assert failed["message"] == "boom"
    assert failed["level"] == "error"
    assert failed["payload"] == {"error": "boom", "error_type": "RuntimeError"}
