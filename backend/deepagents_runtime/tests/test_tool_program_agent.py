from __future__ import annotations

import asyncio
import json
from dataclasses import replace

from deepagents.backends import StateBackend
from langchain.tools import ToolRuntime, tool
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import Field
from ultra_deepagents.agent import build_research_agent
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.context_tools import build_context_tools
from ultra_deepagents.harness_plugins import (
    HarnessPlugin,
    HarnessPluginRegistry,
    ProgramToolPolicy,
)
from ultra_deepagents.tool_program import ToolProgramLimits, build_tool_program_tool


def _context(run_id: str) -> AgentRunContext:
    return AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="org-1",
        user_id="user-1",
        project_id="project-1",
        thread_id="thread-1",
        run_id=run_id,
        goal="Compare the selected paper measurements.",
        runtime_facts={"current_datetime_utc": f"2026-08-15T00:00:0{run_id[-1]}Z"},
    )


class _ToolCallingModel(FakeMessagesListChatModel):
    bound_tool_names: list[list[str]] = Field(default_factory=list)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        _ = tool_choice, kwargs
        self.bound_tool_names.append(
            [str(getattr(tool_object, "name", "") or "") for tool_object in tools]
        )
        return self

    def _get_ls_params(self, stop=None, **kwargs):
        _ = stop, kwargs
        return {"ls_provider": "openai", "ls_model_name": "fake-tool-program"}


def _capture_agent(monkeypatch, settings: RuntimeSettings, context: AgentRunContext):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "compiled"

    monkeypatch.setattr("ultra_deepagents.agent.create_deep_agent", fake_create_deep_agent)
    result = build_research_agent(
        settings,
        model=object(),
        backend=object(),
        context=context,
    )
    assert result == "compiled"
    return captured


def test_agent_registers_tool_program_and_projects_only_cache_stable_capabilities(
    monkeypatch,
):
    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        tool_program_enabled=True,
    )

    first = _capture_agent(monkeypatch, settings, _context("run-1"))
    second = _capture_agent(monkeypatch, settings, _context("run-2"))

    first_names = [str(getattr(item, "name", "")) for item in first["tools"]]
    assert "run_tool_program" in first_names
    assert "artifact_manifest" in first["system_prompt"]
    assert '"concurrency":"parallel"' in first["system_prompt"]
    assert "run-1" not in first["system_prompt"]
    assert "run-2" not in second["system_prompt"]
    assert first["system_prompt"] == second["system_prompt"]


def test_agent_keeps_tool_program_absent_when_rollout_flag_is_disabled(monkeypatch):
    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        tool_program_enabled=False,
    )

    captured = _capture_agent(monkeypatch, settings, _context("run-1"))

    names = [str(getattr(item, "name", "")) for item in captured["tools"]]
    assert "run_tool_program" not in names
    assert "Typed tool-program SDK" not in captured["system_prompt"]


def test_tool_program_surface_excludes_remote_bisque_mutations_and_secrets(monkeypatch):
    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        tool_program_enabled=True,
    )
    context = replace(
        _context("run-3"),
        allowed_tool_packs=("bisque",),
        run_metadata={"bisque_session_id": "must-never-enter-the-prompt"},
    )

    captured = _capture_agent(monkeypatch, settings, context)
    sdk = json.loads(captured["system_prompt"].split("SDK_JSON=", 1)[1].strip())
    names = {entry["name"] for entry in sdk}

    assert "bisque_search_resources" in names
    assert "bisque_download_resource" in names
    assert "bisque_upload_files" not in names
    assert "bisque_upload_workspace_files" not in names
    assert "bisque_create_dataset" not in names
    assert "must-never-enter-the-prompt" not in captured["system_prompt"]


class _ToolEvents(AsyncCallbackHandler):
    def __init__(self) -> None:
        self.names: list[str] = []
        self.inputs: list[dict | None] = []

    async def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        _ = input_str, run_id
        self.names.append(str(kwargs.get("name") or serialized.get("name") or ""))
        value = kwargs.get("inputs")
        self.inputs.append(value if isinstance(value, dict) else None)


@tool
def scoped_probe(runtime: ToolRuntime[AgentRunContext], value: int) -> dict[str, object]:
    """Return the current run identity and a supplied value."""
    return {
        "run_id": runtime.context.run_id,
        "tool_call_id": runtime.tool_call_id,
        "value": value,
    }


def test_program_invokes_original_tool_with_run_context_and_standard_nested_events():
    surface = HarnessPluginRegistry(
        (
            HarnessPlugin(
                name="probe",
                tools=(scoped_probe,),
                program_tools=(ProgramToolPolicy(tool_name="scoped_probe"),),
            ),
        )
    ).freeze()
    program_tool = build_tool_program_tool(
        surface.program_tools,
        limits=ToolProgramLimits(max_operations=4, max_concurrency=2),
    )
    events = _ToolEvents()
    context = _context("run-7")
    runtime = ToolRuntime(
        state={},
        context=context,
        config={"callbacks": [events]},
        stream_writer=lambda _: None,
        tool_call_id="outer-call",
        store=None,
        tools=[program_tool, scoped_probe],
    )

    result = asyncio.run(
        program_tool.ainvoke(
            {
                "name": "run_tool_program",
                "args": {
                    "runtime": runtime,
                    "operations": [
                        {
                            "kind": "call",
                            "id": "probe",
                            "tool": "scoped_probe",
                            "arguments": {"value": 7},
                        }
                    ],
                    "outputs": [
                        {
                            "name": "run_id",
                            "source": {"step": "probe", "pointer": "/run_id"},
                        },
                        {
                            "name": "child_call_id",
                            "source": {
                                "step": "probe",
                                "pointer": "/tool_call_id",
                            },
                        },
                    ],
                },
                "id": "outer-call",
                "type": "tool_call",
            },
            config=runtime.config,
        )
    )

    assert isinstance(result, ToolMessage)
    payload = json.loads(str(result.content))
    assert payload["status"] == "succeeded"
    assert payload["outputs"]["run_id"] == "run-7"
    assert payload["outputs"]["child_call_id"] == "outer-call:probe"
    assert events.names == ["run_tool_program", "scoped_probe"]
    assert all(item is None or "runtime" not in item for item in events.inputs)
    assert "runtime" not in program_tool.tool_call_schema.model_json_schema()["properties"]


def test_program_parses_existing_json_tool_results_for_later_projection():
    artifact_tool = next(item for item in build_context_tools() if item.name == "artifact_manifest")
    surface = HarnessPluginRegistry(
        (
            HarnessPlugin(
                name="context",
                tools=(artifact_tool,),
                program_tools=(ProgramToolPolicy(tool_name="artifact_manifest"),),
            ),
        )
    ).freeze()
    program_tool = build_tool_program_tool(surface.program_tools)
    context = replace(
        _context("run-8"),
        resource_descriptors=(
            {
                "type": "artifact",
                "artifact_id": "artifact-8",
                "run_id": "prior-run",
                "path": "outputs/table.csv",
            },
        ),
    )
    runtime = ToolRuntime(
        state={},
        context=context,
        config={},
        stream_writer=lambda _: None,
        tool_call_id="outer-artifact",
        store=None,
        tools=[program_tool, artifact_tool],
    )

    result = asyncio.run(
        program_tool.ainvoke(
            {
                "runtime": runtime,
                "operations": [
                    {
                        "kind": "call",
                        "id": "manifest",
                        "tool": "artifact_manifest",
                    }
                ],
                "outputs": [
                    {
                        "name": "artifact_id",
                        "source": {
                            "step": "manifest",
                            "pointer": "/prior_artifacts/0/artifact_id",
                        },
                    }
                ],
            }
        )
    )

    parsed = json.loads(result)
    assert parsed["status"] == "succeeded"
    assert parsed["outputs"] == {"artifact_id": "artifact-8"}


def test_program_tool_schema_errors_are_safe_tool_results():
    surface = HarnessPluginRegistry(
        (
            HarnessPlugin(
                name="probe",
                tools=(scoped_probe,),
                program_tools=(ProgramToolPolicy(tool_name="scoped_probe"),),
            ),
        )
    ).freeze()
    program_tool = build_tool_program_tool(surface.program_tools)
    runtime = ToolRuntime(
        state={},
        context=_context("run-invalid"),
        config={},
        stream_writer=lambda _: None,
        tool_call_id="invalid-program",
        store=None,
        tools=[program_tool, scoped_probe],
    )

    result = program_tool.invoke(
        {
            "name": "run_tool_program",
            "args": {"runtime": runtime, "operations": [], "outputs": []},
            "id": "invalid-program",
            "type": "tool_call",
        }
    )

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert result.content == "Invalid tool program schema."


def test_real_deepagents_graph_executes_program_and_returns_complete_final_message():
    model = _ToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_tool_program",
                        "args": {
                            "operations": [
                                {
                                    "kind": "call",
                                    "id": "manifest",
                                    "tool": "artifact_manifest",
                                }
                            ],
                            "outputs": [
                                {
                                    "name": "run_id",
                                    "source": {
                                        "step": "manifest",
                                        "pointer": "/run_id",
                                    },
                                }
                            ],
                        },
                        "id": "program-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The tool program completed and the full answer is intact."),
        ]
    )
    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="fake-tool-program",
        tool_program_enabled=True,
        builder_enabled=False,
        todo_reminders_enabled=False,
    )
    context = _context("run-9")
    agent = build_research_agent(
        settings,
        model=model,
        backend=StateBackend(),
        context=context,
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Read the manifest, then answer."}]},
        context=context,
    )

    assert any("run_tool_program" in names for names in model.bound_tool_names)
    program_messages = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage) and message.name == "run_tool_program"
    ]
    assert len(program_messages) == 1
    payload = json.loads(str(program_messages[0].content))
    assert payload["outputs"] == {"run_id": "run-9"}
    assert result["messages"][-1].content == (
        "The tool program completed and the full answer is intact."
    )
