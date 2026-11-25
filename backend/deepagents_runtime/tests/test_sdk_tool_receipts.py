from langchain_core.messages import ToolMessage
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.runner import _tool_event_from_stream_event


def test_child_tool_lifecycles_keep_names_failures_and_private_deltas_separate():
    context = AgentRunContext(
        run_id="run",
        thread_id="thread",
        user_id="user",
        org_id="org",
        project_id="project",
        assistant_id="agent",
        goal="inspect",
    )
    names = {}

    def event(namespace, status, **extra):
        raw = {
            "method": "tools",
            "params": {
                "namespace": [namespace],
                "data": {
                    "event": status,
                    "tool_call_id": "same",
                    **extra,
                },
            },
        }
        return _tool_event_from_stream_event(context, raw, names)

    assert event("child-a", "tool-started", name="execute")["event_kind"] == "tool_call.started"
    assert event("child-b", "tool-started", name="read_file")["event_kind"] == "tool_call.started"
    assert event("child-b", "unrecognized") is None
    assert event("child-a", "tool-output-delta", delta="private partial result") is None
    error = ToolMessage(content="read failed", status="error", tool_call_id="same")
    failed = event("child-b", "tool-finished", output=error)
    assert (failed["event_kind"], failed["payload"]["tool_name"]) == (
        "tool_call.failed",
        "read_file",
    )
    done = event("child-a", "tool-finished", output={"status": "error"})
    assert (done["event_kind"], done["payload"]["tool_name"]) == ("tool_call.completed", "execute")
    assert event("child-b", "tool-output-delta") is None
