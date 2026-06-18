import json
from pathlib import Path

import pytest
from deepagents.backends import CompositeBackend, StateBackend
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field
from ultra_deepagents.agent import (
    SKILLS_SOURCES,
    build_agent_backend,
    build_research_agent,
    build_run_context_brief,
    build_sandbox_backend,
    build_system_prompt,
    resolve_memory_permissions,
    resolve_skills_root,
    resolve_user_memory_root,
)
from ultra_deepagents.code_execution.docker import DockerSandboxBackend
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.context_tools import (
    artifact_manifest_text,
    build_tool_capability_manifest,
    stage_artifact,
    stage_artifact_for_analysis_text,
    stage_uploaded_files,
    stage_uploaded_files_for_analysis_text,
)
from ultra_deepagents.multimodal import TextOnlyMultimodalMiddleware


class ToolCallingFakeOpenAIModel(FakeMessagesListChatModel):
    bound_tool_names: list[list[str]] = Field(default_factory=list)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        _ = tool_choice, kwargs
        self.bound_tool_names.append(
            [str(getattr(tool, "name", "") or "") for tool in tools]
        )
        return self

    def _get_ls_params(self, stop=None, **kwargs):
        _ = stop, kwargs
        return {"ls_provider": "openai", "ls_model_name": "fake-ultra-model"}


def test_resolve_user_memory_root_is_scoped_and_path_safe(tmp_path):
    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        memory_root=str(tmp_path),
    )

    root_a = resolve_user_memory_root(settings, "researcher-1")
    root_b = resolve_user_memory_root(settings, "researcher-2")
    assert root_a == tmp_path / "users" / "researcher-1"
    assert root_a != root_b
    # Anonymous/empty users must not share the global memory root.
    assert resolve_user_memory_root(settings, None) == tmp_path / "anonymous" / "unscoped"
    assert resolve_user_memory_root(settings, "  ") == tmp_path / "anonymous" / "unscoped"
    assert (
        resolve_user_memory_root(settings, None, thread_id="thread-7")
        == tmp_path / "anonymous_threads" / "thread-7"
    )
    # A hostile user id can never escape the memory root.
    traversal = resolve_user_memory_root(settings, "../../etc/passwd")
    assert ".." not in traversal.parts
    assert str(traversal).startswith(str(tmp_path))


def test_build_agent_backend_routes_memories_to_per_user_directory(tmp_path):
    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        memory_root=str(tmp_path / "memory"),
        artifact_root=str(tmp_path / "artifacts"),
    )
    backend = build_agent_backend(
        settings,
        workspace_dir=str(tmp_path / "workspace"),
        artifact_dir=str(tmp_path / "artifacts" / "run-1"),
        user_id="researcher-7",
    )
    assert isinstance(backend, CompositeBackend)
    expected = tmp_path / "memory" / "users" / "researcher-7"
    assert expected.exists()


def test_build_research_agent_passes_current_deepagents_contract(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "compiled-agent"

    monkeypatch.setattr("ultra_deepagents.agent.create_deep_agent", fake_create_deep_agent)
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
    )
    fake_model = object()
    fake_backend = object()
    fake_tool = object()

    agent = build_research_agent(
        settings,
        model=fake_model,
        backend=fake_backend,
        tools=[fake_tool],
    )

    assert agent == "compiled-agent"
    assert captured["name"] == "ultra-research-agent"
    assert captured["model"] is fake_model
    assert captured["backend"] is fake_backend
    assert not callable(captured["backend"])
    assert captured["tools"][0] is fake_tool
    tool_names = {getattr(tool, "name", "") for tool in captured["tools"]}
    assert "tool_capability_manifest" in tool_names
    assert "rarespot_ecology_inference" in tool_names
    manifest_tool = next(tool for tool in captured["tools"] if getattr(tool, "name", "") == "tool_capability_manifest")
    manifest = manifest_tool.invoke({})
    assert "execute" in manifest
    assert "artifact_manifest" in manifest
    assert "rarespot_ecology_inference" in manifest
    assert "selected_tool_names" in manifest
    rarespot_tool = next(tool for tool in captured["tools"] if getattr(tool, "name", "") == "rarespot_ecology_inference")
    assert "512 px tiles" in rarespot_tool.description
    assert "25%" in rarespot_tool.description
    assert "queued" in rarespot_tool.description.lower()
    assert "runtime context" in rarespot_tool.description.lower()
    assert captured["context_schema"] is AgentRunContext
    assert captured["memory"] == [
        "/memories/user_profile.md",
        "/memories/preferences.md",
        "/memories/research_context/INDEX.md",
        "/policies/lab_policy.md",
    ]
    # App-owned and org-owned memory are write-protected (read-only to the agent).
    deny_paths = {
        path
        for rule in captured["permissions"]
        if rule.mode == "deny"
        for path in rule.paths
    }
    assert "/memories/user_profile.md" in deny_paths
    assert "/policies/**" in deny_paths
    assert "/skills/**" in deny_paths
    assert "/memories/" in captured["system_prompt"]
    assert "user_profile.md" in captured["system_prompt"]
    assert "preferences.md" in captured["system_prompt"]
    assert "research_context/" in captured["system_prompt"]
    assert "token" not in captured["system_prompt"].lower()
    assert "/outputs/" in captured["system_prompt"]
    assert "when code-runner is available" in captured["system_prompt"].lower()
    assert "prefer delegating" in captured["system_prompt"].lower()
    assert "code execution" in captured["system_prompt"].lower()
    assert "data inspection" in captured["system_prompt"].lower()
    assert "sandbox execution" in captured["system_prompt"].lower()
    assert "caption immediately after each figure" in captured["system_prompt"].lower()
    assert "do not call read_file on image" in captured["system_prompt"].lower()
    assert any(
        isinstance(item, TextOnlyMultimodalMiddleware)
        for item in captured["middleware"]
    )

    assert captured["subagents"] == []


def test_tool_capability_manifest_describes_builtin_storage_and_registered_tools():
    class FakeTool:
        def __init__(self, name: str) -> None:
            self.name = name

    manifest = build_tool_capability_manifest(
        [
            FakeTool("artifact_manifest"),
            FakeTool("stage_uploaded_files_for_analysis"),
            FakeTool("rarespot_ecology_inference"),
            FakeTool(""),
        ]
    )

    builtin_names = {
        str(tool.get("name"))
        for tool in manifest["deepagents_builtin_tools"]
        if isinstance(tool, dict)
    }
    assert {"execute", "write_file", "read_file", "edit_file", "ls", "glob", "grep", "write_todos"}.issubset(
        builtin_names
    )
    assert "task" not in builtin_names
    assert manifest["available_subagents"] == []
    assert manifest["available_async_subagents"] == []
    assert manifest["registered_tools"] == [
        "artifact_manifest",
        "rarespot_ecology_inference",
        "stage_uploaded_files_for_analysis",
    ]
    assert manifest["storage"] == {
        "workspace": "/workspace",
        "outputs": "/outputs",
        "memories": "/memories",
        "staged_uploads": "/workspace/staged_uploads",
        "staged_artifacts": "/workspace/staged_artifacts",
        "staged_repos": "/workspace/staged_repos",
        "staged_resources": "/workspace/staged_resources",
    }
    assert "selected_tool_names" in manifest


def test_tool_capability_manifest_does_not_advertise_undescribed_async_subagents():
    manifest = build_tool_capability_manifest(
        [],
        available_async_subagents=[
            {
                "name": "remote-training-runner",
                "description": " ",
                "graph_id": "ultra-training-agent",
            }
        ],
    )

    builtin_names = {
        str(tool.get("name"))
        for tool in manifest["deepagents_builtin_tools"]
        if isinstance(tool, dict)
    }
    assert "start_async_task" not in builtin_names
    assert "list_async_tasks" not in builtin_names
    assert manifest["available_async_subagents"] == []


def test_build_sandbox_backend_uses_runtime_settings(tmp_path: Path):
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
        sandbox_image="bisque-ultra-codeexec:test",
        sandbox_network="none",
        sandbox_cpus=3.5,
        sandbox_memory="8g",
        sandbox_pids_limit=512,
        sandbox_timeout_seconds=1800,
        sandbox_output_limit_bytes=500_000,
    )

    backend = build_sandbox_backend(
        settings,
        workspace_dir=tmp_path / "workspace",
        outputs_dir=tmp_path / "artifacts",
    )

    assert isinstance(backend, DockerSandboxBackend)
    assert backend.workspace_dir == tmp_path / "workspace"
    assert backend.outputs_dir == tmp_path / "artifacts"
    matplotlibrc = (tmp_path / "workspace" / "matplotlibrc").read_text()
    mpl_config_rc = (tmp_path / "workspace" / ".cache" / "matplotlib" / "matplotlibrc").read_text()
    assert "savefig.dpi: 300" in matplotlibrc
    assert "savefig.dpi: 300" in mpl_config_rc
    assert backend.config.image == "bisque-ultra-codeexec:test"
    assert backend.config.network == "none"
    assert backend.config.cpus == 3.5
    assert backend.config.memory == "8g"
    assert backend.config.pids_limit == 512
    assert backend.config.timeout_seconds == 1800
    assert backend.config.output_limit_bytes == 500_000


def test_build_agent_backend_routes_memory_outside_sandbox(tmp_path: Path):
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
        memory_root=str(tmp_path / "memory"),
        artifact_root=str(tmp_path / "artifacts"),
    )

    run_artifact_dir = tmp_path / "artifacts" / "run-1"
    backend = build_agent_backend(
        settings,
        workspace_dir=tmp_path / "workspace",
        artifact_dir=run_artifact_dir,
    )

    assert isinstance(backend, CompositeBackend)
    assert isinstance(backend.default, DockerSandboxBackend)
    assert backend.default.outputs_dir == run_artifact_dir
    docker_command = backend.default.build_docker_command("python /outputs/report.py")
    assert f"{run_artifact_dir.resolve()}:/outputs:rw" in docker_command
    memory_response = backend.download_files(["/memories/preferences.md"])[0]
    assert memory_response.error == "file_not_found"
    assert memory_response.error != "permission_denied"
    output_response = backend.upload_files([("/outputs/report.md", b"# Report")])[0]
    assert output_response.error is None
    assert (run_artifact_dir / "report.md").read_bytes() == b"# Report"
    workspace_response = backend.upload_files([("/workspace/data.txt", b"ok")])[0]
    assert workspace_response.error is None
    assert (tmp_path / "workspace" / "data.txt").read_bytes() == b"ok"


def test_build_research_agent_skips_text_only_sanitizer_for_multimodal_model(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "compiled-agent"

    monkeypatch.setattr("ultra_deepagents.agent.create_deep_agent", fake_create_deep_agent)
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="gpt-4o",
        model_supports_multimodal=True,
    )

    agent = build_research_agent(settings, model=object(), backend=object())

    assert agent == "compiled-agent"
    assert all(
        not isinstance(item, TextOnlyMultimodalMiddleware)
        for item in captured["middleware"]
    )
    assert captured["subagents"] == []
    assert "do not call read_file on image" not in captured["system_prompt"].lower()


def test_research_agent_registers_configured_async_subagents_and_manifest(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "compiled-agent"

    monkeypatch.setattr("ultra_deepagents.agent.create_deep_agent", fake_create_deep_agent)
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
        async_subagents=(
            {
                "name": "remote-training-runner",
                "description": "Runs long model training jobs on an Ultra-owned remote worker.",
                "graph_id": "ultra-training-agent",
                "url": "https://langgraph.example.test",
                "headers": {"X-Ultra-Async-Token": "secret-ref"},
            },
        ),
    )
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="researcher-1",
        project_id="local-project",
        thread_id="thread-async",
        run_id="run-async",
        goal="Launch async work.",
    )

    agent = build_research_agent(settings, model=object(), backend=object(), context=context)

    assert agent == "compiled-agent"
    assert captured["subagents"] == [
        {
            "name": "remote-training-runner",
            "description": "Runs long model training jobs on an Ultra-owned remote worker.",
            "graph_id": "ultra-training-agent",
            "url": "https://langgraph.example.test",
            "headers": {"X-Ultra-Async-Token": "secret-ref"},
        }
    ]
    manifest_tool = next(
        tool for tool in captured["tools"] if getattr(tool, "name", "") == "tool_capability_manifest"
    )
    manifest = json.loads(manifest_tool.invoke({}))
    builtin_names = {
        str(tool.get("name"))
        for tool in manifest["deepagents_builtin_tools"]
        if isinstance(tool, dict)
    }
    assert {
        "start_async_task",
        "check_async_task",
        "update_async_task",
        "cancel_async_task",
        "list_async_tasks",
    }.issubset(builtin_names)
    assert "task" not in builtin_names
    assert manifest["available_subagents"] == []
    assert manifest["available_async_subagents"] == [
        {
            "name": "remote-training-runner",
            "description": "Runs long model training jobs on an Ultra-owned remote worker.",
        }
    ]


def test_research_agent_rejects_async_subagent_without_description(monkeypatch):
    def fake_create_deep_agent(**kwargs):
        raise AssertionError("invalid async subagent specs should fail before graph creation")

    monkeypatch.setattr("ultra_deepagents.agent.create_deep_agent", fake_create_deep_agent)
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
        async_subagents=(
            {
                "name": "remote-training-runner",
                "description": " ",
                "graph_id": "ultra-training-agent",
                "url": "https://langgraph.example.test",
            },
        ),
    )
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="researcher-1",
        project_id="local-project",
        thread_id="thread-async",
        run_id="run-async",
        goal="Launch async work.",
    )

    with pytest.raises(ValueError, match="description"):
        build_research_agent(settings, model=object(), backend=object(), context=context)


def test_configured_async_subagent_tools_persist_task_state_across_followup(monkeypatch):
    import deepagents.middleware.async_subagents as async_subagents_module

    class FakeThreads:
        def __init__(self) -> None:
            self.created: list[dict[str, object]] = []

        def create(self, **kwargs):
            thread_id = f"async-thread-{len(self.created) + 1}"
            self.created.append({"thread_id": thread_id, **kwargs})
            return {"thread_id": thread_id}

        def get(self, thread_id: str):
            assert thread_id == "async-thread-1"
            return {
                "values": {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "Remote worker finished the analysis.",
                        }
                    ]
                }
            }

    class FakeRuns:
        def __init__(self) -> None:
            self.created: list[dict[str, object]] = []
            self.checked: list[dict[str, str]] = []

        def create(self, **kwargs):
            self.created.append(kwargs)
            return {"run_id": "async-run-1"}

        def get(self, **kwargs):
            self.checked.append(kwargs)
            return {"status": "success"}

    class FakeAgentProtocolClient:
        def __init__(self) -> None:
            self.threads = FakeThreads()
            self.runs = FakeRuns()

    clients: list[FakeAgentProtocolClient] = []

    def fake_get_sync_client(**kwargs):
        assert kwargs == {
            "url": "http://agent-protocol.test",
            "headers": {"x-auth-scheme": "langsmith"},
        }
        if not clients:
            clients.append(FakeAgentProtocolClient())
        return clients[0]

    monkeypatch.setattr("ultra_deepagents.async_delegation.get_sync_client", fake_get_sync_client)
    monkeypatch.setattr(async_subagents_module, "get_sync_client", fake_get_sync_client)
    model = ToolCallingFakeOpenAIModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "start_async_task",
                        "args": {
                            "description": "Run a concise analysis.",
                            "subagent_type": "remote-training-runner",
                        },
                        "id": "call-start-1",
                    }
                ],
            ),
            AIMessage(content="Launched async-thread-1."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "check_async_task",
                        "args": {"task_id": "async-thread-1"},
                        "id": "call-check-1",
                    }
                ],
            ),
            AIMessage(content="Checked async-thread-1."),
        ]
    )
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="fake-ultra-model",
        async_subagents=(
            {
                "name": "remote-training-runner",
                "description": "Runs long model training jobs on an Ultra-owned remote worker.",
                "graph_id": "ultra-training-agent",
                "url": "http://agent-protocol.test",
            },
        ),
    )
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="researcher-1",
        project_id="local-project",
        thread_id="thread-async",
        run_id="run-async",
        goal="Launch async work.",
        selected_file_ids=("file-1",),
        allowed_tool_packs=("workspace", "code"),
        resource_descriptors=(
            {
                "type": "artifact",
                "artifact_id": "artifact-1",
                "run_id": "prior-run",
                "path": "outputs/result.csv",
            },
        ),
        run_metadata={"bisque_session_id": "session-secret"},
        auth_claims={"access_token": "secret-token"},
    )
    agent = build_research_agent(
        settings,
        model=model,
        backend=StateBackend(),
        context=context,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": context.thread_id}}

    first = agent.invoke(
        {"messages": [{"role": "user", "content": "Launch async work."}]},
        config=config,
        context=context,
    )
    second = agent.invoke(
        {"messages": [{"role": "user", "content": "Check async-thread-1."}]},
        config=config,
        context=context,
    )

    assert first["async_tasks"]["async-thread-1"]["status"] == "running"
    assert second["async_tasks"]["async-thread-1"]["status"] == "success"
    assert second["async_tasks"]["async-thread-1"]["agent_name"] == "remote-training-runner"
    assert clients[0].threads.created == [
        {
            "thread_id": "async-thread-1",
            "metadata": {
                "ultra_delegation": "async_subagent",
                "ultra_parent_run_id": "run-async",
                "ultra_parent_thread_id": "thread-async",
                "ultra_subagent_name": "remote-training-runner",
                "ultra_org_id": "local-org",
                "ultra_user_id": "researcher-1",
                "ultra_project_id": "local-project",
                "ultra_subagent_graph_id": "ultra-training-agent",
            },
            "graph_id": "ultra-training-agent",
        }
    ]
    assert clients[0].runs.created == [
        {
            "thread_id": "async-thread-1",
            "assistant_id": "ultra-training-agent",
            "input": {
                "messages": [{"role": "user", "content": "Run a concise analysis."}]
            },
            "context": {
                "assistant_id": "ultra-research-agent",
                "org_id": "local-org",
                "user_id": "researcher-1",
                "project_id": "local-project",
                "thread_id": "thread-async",
                "run_id": "run-async",
                "goal": "Launch async work.",
                "model_profile": "vllm",
                "selected_file_ids": ["file-1"],
                "selected_resource_uris": [],
                "selected_dataset_uris": [],
                "allowed_tool_packs": ["workspace", "code"],
                "knowledge_context": {},
                "selection_context": {},
                "workflow_hint": {},
                "reasoning_mode": "auto",
                "benchmark": {},
                "run_metadata": {
                    "delegation": {
                        "mode": "async_subagent",
                        "parent_run_id": "run-async",
                        "parent_thread_id": "thread-async",
                        "subagent_name": "remote-training-runner",
                    }
                },
                "resource_descriptors": [
                    {
                        "type": "artifact",
                        "artifact_id": "artifact-1",
                        "run_id": "prior-run",
                        "path": "outputs/result.csv",
                    },
                ],
                "response_contract": {},
                "budget": {},
                "auth_claims": {},
                "artifact_root": "/outputs",
                "workspace_root": "/workspace",
                "sandbox_policy": {},
            },
            "metadata": {
                "ultra_delegation": "async_subagent",
                "ultra_parent_run_id": "run-async",
                "ultra_parent_thread_id": "thread-async",
                "ultra_subagent_name": "remote-training-runner",
                "ultra_org_id": "local-org",
                "ultra_user_id": "researcher-1",
                "ultra_project_id": "local-project",
                "ultra_subagent_graph_id": "ultra-training-agent",
            },
            "durability": "async",
        }
    ]
    assert clients[0].runs.checked == [
        {"thread_id": "async-thread-1", "run_id": "async-run-1"}
    ]
    assert any("start_async_task" in tool_names for tool_names in model.bound_tool_names)
    assert any("check_async_task" in tool_names for tool_names in model.bound_tool_names)
    assert all("task" not in tool_names for tool_names in model.bound_tool_names)
    check_tool_message = next(
        message
        for message in second["messages"]
        if getattr(message, "name", "") == "check_async_task"
    )
    assert "Remote worker finished the analysis." in check_tool_message.content


def test_research_agent_registers_scoped_subagents_for_code_execution_context(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "compiled-agent"

    monkeypatch.setattr("ultra_deepagents.agent.create_deep_agent", fake_create_deep_agent)
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
    )
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-code",
        goal=(
            "Train a tiny UNet model, run code, debug the script, save weights, "
            "plots, metrics, and a report."
        ),
    )

    build_research_agent(settings, model=object(), backend=object(), context=context)

    subagent_names = {subagent["name"] for subagent in captured["subagents"]}
    assert "code-runner" in subagent_names
    # data-analyst was dropped (runtime-identical to code-runner); no duplicate delegate.
    assert "data-analyst" not in subagent_names
    assert "general-purpose" not in subagent_names

    code_runner = next(
        subagent for subagent in captured["subagents"] if subagent["name"] == "code-runner"
    )
    assert "sandbox execution" in code_runner["description"].lower()
    code_response_format = code_runner["response_format"]
    assert code_response_format["type"] == "object"
    assert code_response_format["required"] == [
        "summary",
        "key_findings",
        "artifacts",
        "failures",
        "confidence",
        "confidence_basis",
    ]
    assert code_response_format["properties"]["artifacts"]["items"]["required"] == [
        "path",
        "description",
    ]
    code_tool_names = {getattr(tool, "name", "") for tool in code_runner["tools"]}
    assert code_tool_names == {
        "artifact_manifest",
        "stage_artifact_for_analysis",
        "stage_uploaded_files_for_analysis",
    }
    assert "stage selected uploads" in code_runner["system_prompt"].lower()
    assert "stage prior artifacts" in code_runner["system_prompt"].lower()
    assert "preserve the user's requested compute scope" in code_runner["system_prompt"].lower()
    assert "longer durations, finer step sizes, more seeds" in code_runner["system_prompt"].lower()
    # code-runner absorbed the data-inspection role and carries an anti-bloat instruction.
    assert "data-inspection" in code_runner["system_prompt"].lower()
    assert "do not paste raw stdout" in code_runner["system_prompt"].lower()
    assert all(
        isinstance(item, TextOnlyMultimodalMiddleware)
        for item in code_runner.get("middleware", [])
    )
    manifest_tool = next(
        tool for tool in captured["tools"] if getattr(tool, "name", "") == "tool_capability_manifest"
    )
    manifest = json.loads(manifest_tool.invoke({}))
    builtin_names = {
        str(tool.get("name"))
        for tool in manifest["deepagents_builtin_tools"]
        if isinstance(tool, dict)
    }
    assert "task" in builtin_names
    structured_handoff = {
        "type": "object",
        "required": [
            "summary",
            "key_findings",
            "artifacts",
            "failures",
            "confidence",
            "confidence_basis",
        ],
        "properties": [
            "artifacts",
            "confidence",
            "confidence_basis",
            "failures",
            "key_findings",
            "summary",
        ],
    }
    assert manifest["available_subagents"] == [
        {
            "name": "code-runner",
            "description": code_runner["description"],
            "response_format": structured_handoff,
            "tool_names": [
                "artifact_manifest",
                "stage_artifact_for_analysis",
                "stage_uploaded_files_for_analysis",
            ],
        },
    ]


def test_text_only_system_prompt_guides_inline_plot_captions_and_updates():
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
    )

    prompt = build_system_prompt(settings)

    assert "caption immediately after each figure" in prompt.lower()
    assert "not all at the end" in prompt.lower()
    assert "savefig.dpi" in prompt
    assert "dpi=300" in prompt
    assert "publication-quality" in prompt.lower()
    assert "error bars" in prompt.lower()
    assert "do not call read_file on image" in prompt.lower()


def test_system_prompt_discourages_analysis_wall_clock_timeout_wrappers():
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
    )

    prompt = " ".join(build_system_prompt(settings).lower().split())

    assert "do not wrap sandbox commands with shell timeout" in prompt
    assert "execute timeout" in prompt
    assert "long-running analysis is allowed" in prompt


def test_system_prompt_requires_delegation_for_complex_code_workflows():
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
    )

    prompt = " ".join(build_system_prompt(settings).lower().split())

    assert "for complex code, simulation, model-training, or multi-file implementation work" in prompt
    assert "call tool_capability_manifest early" in prompt
    assert "delegate at least one focused verification" in prompt
    assert "bounded by the user's requested seeds, durations, data size, and artifact scope" in prompt
    assert "run a small smoke check before any expensive cross-check" in prompt
    assert "do not expand into exhaustive convergence sweeps" in prompt
    assert "code-runner" in prompt
    assert "start_async_task" in prompt
    assert "configured async subagents" in prompt
    assert "before the final answer" in prompt


def test_system_prompt_guides_rarespot_without_filesystem_hunting():
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
    )

    prompt = " ".join(build_system_prompt(settings).lower().split())

    assert "rarespot_ecology_inference" in prompt
    assert "512 px tiles with 25% overlap" in prompt
    assert "do not search the sandbox filesystem" in prompt
    assert "do not rerun the same rarespot configuration" in prompt
    assert "report-only or synthesis-only follow-ups" in prompt
    assert "artifact_manifest and stage_artifact_for_analysis" in prompt
    assert "do not create stub" in prompt


def test_system_prompt_surfaces_prior_artifacts_from_runtime_context():
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
    )
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-2",
        goal="Add a dashed reference line.",
        selected_file_ids=("file-prairie",),
        resource_descriptors=(
            {
                "artifact_id": "artifact-plot",
                "run_id": "run-1",
                "kind": "figure",
                "title": "Squared Plot",
                "path": "outputs/plot_squared.png",
                "mime_type": "image/png",
            },
        ),
    )

    brief = build_run_context_brief(context)
    prompt = build_system_prompt(settings, context)

    assert "Add a dashed reference line." in brief
    assert "file-prairie" in brief
    assert "stage_uploaded_files_for_analysis" in brief
    assert "artifact-plot" in brief
    assert "/outputs/run-1/outputs/plot_squared.png" not in brief
    assert "use stage_artifact_for_analysis" in brief
    assert "artifact_manifest" in prompt
    assert "stage_artifact_for_analysis" in prompt
    assert "stage_uploaded_files_for_analysis" in prompt
    assert "artifact-plot" in prompt


def test_research_agent_registers_prior_artifact_context_tools(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "compiled-agent"

    monkeypatch.setattr("ultra_deepagents.agent.create_deep_agent", fake_create_deep_agent)
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
    )

    build_research_agent(settings, model=object(), backend=object())

    tool_names = {getattr(tool, "name", "") for tool in captured["tools"]}
    assert "artifact_manifest" in tool_names
    assert "stage_artifact_for_analysis" in tool_names
    assert "stage_uploaded_files_for_analysis" in tool_names


def test_research_agent_narrows_tools_for_rarespot_report_only_context(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "compiled-agent"

    monkeypatch.setattr("ultra_deepagents.agent.create_deep_agent", fake_create_deep_agent)
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
    )
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-report",
        goal=(
            "Write a combined quantitative report across the RareSpot runs in this chat. "
            "Do not rerun inference; use the prior RareSpot outputs."
        ),
        selected_file_ids=("file-prairie",),
    )

    build_research_agent(settings, model=object(), backend=object(), context=context)

    tool_names = {getattr(tool, "name", "") for tool in captured["tools"]}
    assert {"artifact_manifest", "stage_artifact_for_analysis"}.issubset(tool_names)
    assert "read_paper_pages" not in tool_names
    assert "search_paper" not in tool_names
    assert "render_paper_page" not in tool_names
    assert "rarespot_ecology_inference" not in tool_names
    assert captured["subagents"] == []


def test_research_agent_does_not_expose_async_subagents_for_rarespot_report_only_context(
    monkeypatch,
):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "compiled-agent"

    monkeypatch.setattr("ultra_deepagents.agent.create_deep_agent", fake_create_deep_agent)
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
        async_subagents=(
            {
                "name": "remote-training-runner",
                "description": "Runs long model training jobs on an Ultra-owned remote worker.",
                "graph_id": "ultra-training-agent",
                "url": "https://langgraph.example.test",
            },
        ),
    )
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-report",
        goal=(
            "Write a combined quantitative report across the RareSpot runs in this chat. "
            "Do not rerun inference; use the prior RareSpot outputs."
        ),
        selected_file_ids=("file-prairie",),
    )

    build_research_agent(settings, model=object(), backend=object(), context=context)

    assert captured["subagents"] == []
    manifest_tool = next(
        tool for tool in captured["tools"] if getattr(tool, "name", "") == "tool_capability_manifest"
    )
    manifest = json.loads(manifest_tool.invoke({}))
    builtin_names = {
        str(tool.get("name"))
        for tool in manifest["deepagents_builtin_tools"]
        if isinstance(tool, dict)
    }
    assert "start_async_task" not in builtin_names
    assert "list_async_tasks" not in builtin_names
    assert manifest["available_async_subagents"] == []


def test_research_agent_keeps_paper_tools_when_paper_context_exists(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "compiled-agent"

    monkeypatch.setattr("ultra_deepagents.agent.create_deep_agent", fake_create_deep_agent)
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
    )
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-paper",
        goal="Explain equation 3 from the uploaded paper.",
        knowledge_context={
            "ingested_papers": [
                {"paper_id": "attention", "page_count": 15, "extraction_status": "ok"}
            ]
        },
    )

    build_research_agent(settings, model=object(), backend=object(), context=context)

    tool_names = {getattr(tool, "name", "") for tool in captured["tools"]}
    assert "read_paper_pages" in tool_names
    assert "search_paper" in tool_names
    assert [subagent["name"] for subagent in captured["subagents"]] == ["literature-reviewer"]
    literature = captured["subagents"][0]
    literature_tool_names = {getattr(tool, "name", "") for tool in literature["tools"]}
    assert "read_paper_pages" in literature_tool_names
    assert all(
        isinstance(item, TextOnlyMultimodalMiddleware)
        for item in literature.get("middleware", [])
    )


def test_stage_artifact_copies_prior_output_into_current_workspace(tmp_path: Path):
    artifact_store = tmp_path / "artifacts"
    source = artifact_store / "run-1" / "outputs" / "plot_squared.py"
    source.parent.mkdir(parents=True)
    source.write_text("print('old plot')\n")
    workspace = tmp_path / "workspaces" / "run-2"
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-2",
        artifact_root=str(artifact_store / "run-2"),
        workspace_root=str(workspace),
        resource_descriptors=(
            {
                "type": "artifact",
                "artifact_id": "artifact-code",
                "run_id": "run-1",
                "kind": "code",
                "path": "outputs/plot_squared.py",
            },
        ),
    )

    result = stage_artifact(context, artifact_id="artifact-code")

    assert result["ok"] is True
    staged = Path(result["staged_path"])
    assert staged.read_text() == "print('old plot')\n"
    assert result["sandbox_path"] == "/workspace/staged_artifacts/run-1/plot_squared.py"


def test_artifact_manifest_text_uses_public_virtual_paths(tmp_path: Path):
    source_path = tmp_path / "artifacts" / "run-1" / "outputs" / "result.csv"
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-2",
        artifact_root=str(tmp_path / "artifacts" / "run-2"),
        workspace_root=str(tmp_path / "workspaces" / "run-2"),
        resource_descriptors=(
            {
                "type": "artifact",
                "artifact_id": "artifact-data",
                "run_id": "run-1",
                "kind": "table",
                "path": "outputs/result.csv",
                "source_path": str(source_path),
                "storage_uri": f"file://{source_path}",
                "sha256": "abc123",
            },
        ),
    )

    payload = json.loads(artifact_manifest_text(context))

    assert payload["workspace_root"] == "/workspace"
    assert payload["artifact_root"] == "/outputs"
    artifact = payload["prior_artifacts"][0]
    assert artifact["artifact_id"] == "artifact-data"
    assert artifact["path"] == "outputs/result.csv"
    assert artifact["access"] == "stage_artifact_for_analysis"
    assert "source_path" not in artifact
    assert "storage_uri" not in artifact
    assert str(tmp_path) not in json.dumps(payload)


def test_artifact_manifest_text_exposes_remote_safe_storage_uri():
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-2",
        resource_descriptors=(
            {
                "type": "artifact",
                "artifact_id": "artifact-object",
                "run_id": "run-1",
                "kind": "table",
                "title": "Remote table",
                "remote_storage_uri": "s3://ultra-artifacts/local-org/run-1/result.csv",
                "storage_uri": "file:///home/scientist/private/result.csv",
                "source_path": "/home/scientist/private/result.csv",
            },
        ),
    )

    payload = json.loads(artifact_manifest_text(context))

    artifact = payload["prior_artifacts"][0]
    assert artifact["artifact_id"] == "artifact-object"
    assert artifact["remote_storage_uri"] == (
        "s3://ultra-artifacts/local-org/run-1/result.csv"
    )
    assert artifact["access"] == "remote_storage_uri"
    assert "storage_uri" not in artifact
    assert "source_path" not in artifact


def test_stage_artifact_for_analysis_text_returns_sandbox_paths_only(tmp_path: Path):
    artifact_store = tmp_path / "artifacts"
    source = artifact_store / "run-1" / "outputs" / "plot_squared.py"
    source.parent.mkdir(parents=True)
    source.write_text("print('old plot')\n")
    workspace = tmp_path / "workspaces" / "run-2"
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-2",
        artifact_root=str(artifact_store / "run-2"),
        workspace_root=str(workspace),
        resource_descriptors=(
            {
                "type": "artifact",
                "artifact_id": "artifact-code",
                "run_id": "run-1",
                "kind": "code",
                "path": "outputs/plot_squared.py",
                "source_path": str(source),
                "storage_uri": f"file://{source}",
            },
        ),
    )

    payload = json.loads(stage_artifact_for_analysis_text(context, artifact_id="artifact-code"))

    assert payload["ok"] is True
    assert payload["staged_path"] == "/workspace/staged_artifacts/run-1/plot_squared.py"
    assert payload["sandbox_path"] == "/workspace/staged_artifacts/run-1/plot_squared.py"
    assert "source_path" not in payload
    assert "storage_uri" not in payload["artifact"]
    assert str(tmp_path) not in json.dumps(payload)
    assert (workspace / "staged_artifacts" / "run-1" / "plot_squared.py").read_text() == (
        "print('old plot')\n"
    )


def test_stage_uploaded_files_copies_selected_uploads_into_current_workspace(tmp_path: Path):
    upload_root = tmp_path / "uploads"
    source = upload_root / "file-1__prairie.jpg"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"image-bytes")
    workspace = tmp_path / "workspaces" / "run-1"
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-1",
        workspace_root=str(workspace),
        selected_file_ids=("file-1",),
    )

    result = stage_uploaded_files(context, upload_roots=(upload_root,))

    assert result["ok"] is True
    assert result["missing_file_ids"] == []
    assert len(result["staged_files"]) == 1
    staged = Path(result["staged_files"][0]["staged_path"])
    assert staged.read_bytes() == b"image-bytes"
    assert result["staged_files"][0]["sandbox_path"] == "/workspace/staged_uploads/file-1/prairie.jpg"


def test_stage_uploaded_files_for_analysis_text_returns_sandbox_paths_only(tmp_path: Path):
    upload_root = tmp_path / "uploads"
    source = upload_root / "file-1__prairie.jpg"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"image-bytes")
    workspace = tmp_path / "workspaces" / "run-1"
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-1",
        workspace_root=str(workspace),
        selected_file_ids=("file-1",),
    )

    payload = json.loads(
        stage_uploaded_files_for_analysis_text(context, upload_roots=(upload_root,))
    )

    assert payload["ok"] is True
    staged = payload["staged_files"][0]
    assert staged["file_id"] == "file-1"
    assert staged["staged_path"] == "/workspace/staged_uploads/file-1/prairie.jpg"
    assert staged["sandbox_path"] == "/workspace/staged_uploads/file-1/prairie.jpg"
    assert "source_path" not in staged
    assert str(tmp_path) not in json.dumps(payload)
    assert (workspace / "staged_uploads" / "file-1" / "prairie.jpg").read_bytes() == b"image-bytes"


def test_stage_uploaded_files_accepts_json_encoded_file_ids(tmp_path: Path):
    upload_root = tmp_path / "uploads"
    source = upload_root / "file-1__prairie.jpg"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"image-bytes")
    workspace = tmp_path / "workspaces" / "run-1"
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-1",
        workspace_root=str(workspace),
    )

    result = stage_uploaded_files(
        context,
        upload_roots=(upload_root,),
        file_ids='["file-1"]',
    )

    assert result["ok"] is True
    assert result["missing_file_ids"] == []
    assert result["staged_files"][0]["sandbox_path"] == "/workspace/staged_uploads/file-1/prairie.jpg"


def _code_execution_context(run_id: str = "run-skills") -> AgentRunContext:
    return AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id=run_id,
        goal="Run a simulation, analyze metrics, and plot the results.",
    )


def test_resolve_skills_root_defaults_to_shipped_skills():
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
    )

    root = resolve_skills_root(settings)

    assert root is not None
    names = {path.parent.name for path in root.glob("*/SKILL.md")}
    assert {"computational-experiment-rigor", "scientific-reporting"}.issubset(names)


def test_resolve_skills_root_rejects_empty_or_missing_dirs(tmp_path: Path):
    empty = tmp_path / "empty-skills"
    empty.mkdir()
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
        skills_root=str(empty),
    )
    assert resolve_skills_root(settings) is None

    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
        skills_root=str(tmp_path / "does-not-exist"),
    )
    assert resolve_skills_root(settings) is None


def test_research_agent_wires_skills_with_built_backend(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "compiled-agent"

    monkeypatch.setattr("ultra_deepagents.agent.create_deep_agent", fake_create_deep_agent)
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
        workspace_root=str(tmp_path / "workspaces"),
        memory_root=str(tmp_path / "memory"),
        artifact_root=str(tmp_path / "artifacts"),
    )

    build_research_agent(
        settings,
        model=object(),
        workspace_dir=tmp_path / "workspaces" / "run-skills",
        context=_code_execution_context(),
    )

    assert captured["skills"] == SKILLS_SOURCES
    subagents = captured["subagents"]
    assert subagents, "scoped delegation subagents should register for this goal"
    for subagent in subagents:
        assert subagent["skills"] == SKILLS_SOURCES


def test_research_agent_skips_skills_with_caller_backend(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "compiled-agent"

    monkeypatch.setattr("ultra_deepagents.agent.create_deep_agent", fake_create_deep_agent)
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
    )

    build_research_agent(
        settings,
        model=object(),
        backend=object(),
        context=_code_execution_context(run_id="run-skills-custom-backend"),
    )

    assert captured["skills"] is None
    for subagent in captured["subagents"]:
        assert "skills" not in subagent


def test_shipped_skill_files_have_valid_frontmatter():
    import re

    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
    )
    root = resolve_skills_root(settings)
    assert root is not None

    for skill_md in sorted(root.glob("*/SKILL.md")):
        text = skill_md.read_text(encoding="utf-8")
        match = re.match(r"^---\n(.*?)\n---\n", text, flags=re.DOTALL)
        assert match, f"{skill_md} is missing YAML frontmatter"
        frontmatter = match.group(1)
        name_match = re.search(r"^name:\s*(\S+)\s*$", frontmatter, flags=re.MULTILINE)
        desc_match = re.search(r"^description:\s*(.+)$", frontmatter, flags=re.MULTILINE)
        assert name_match, f"{skill_md} frontmatter is missing name"
        assert desc_match, f"{skill_md} frontmatter is missing description"
        name = name_match.group(1)
        assert re.fullmatch(r"[a-z0-9-]{1,64}", name), name
        assert name == skill_md.parent.name
        assert len(desc_match.group(1)) <= 1024


def test_is_rigor_intelligence_maps_pro_mode_hint():
    from ultra_deepagents.agent import is_rigor_intelligence

    pro = _code_execution_context()
    pro = AgentRunContext(**{**pro.to_payload(), "workflow_hint": {"id": "pro_mode"}})
    assert is_rigor_intelligence(pro) is True

    high = _code_execution_context()
    assert is_rigor_intelligence(high) is False
    assert is_rigor_intelligence(None) is False

    other = AgentRunContext(
        **{**_code_execution_context().to_payload(), "workflow_hint": {"id": "sam3"}}
    )
    assert is_rigor_intelligence(other) is False


def test_looks_scoped_delegation_goal_shared_gate():
    from ultra_deepagents.agent import looks_scoped_delegation_goal

    assert looks_scoped_delegation_goal("Run a simulation and analyze metrics.") is True
    assert looks_scoped_delegation_goal("Analyze this Python code and debug the workflow.") is True
    assert looks_scoped_delegation_goal("Detect prairie dog burrows in this image.") is False
    assert looks_scoped_delegation_goal("Say hello.") is False


def test_quantitative_rigor_goal_is_narrower_than_scoped_delegation():
    from ultra_deepagents.agent import (
        looks_quantitative_rigor_goal,
        looks_scoped_delegation_goal,
    )

    non_quantitative = "Analyze this Python code and debug the workflow."
    schema_inspection = "Analyze this dataset schema and summarize its columns."
    negated_live_debug = """Pro mode debug request. Analyze this Python function for correctness.
Do not run a numerical experiment and do not create plots or CSVs.

```python
def normalize_rows(rows):
    totals = [sum(row) for row in rows]
    return [[x / totals[i] for x in row] for i, row in enumerate(rows)]
```

Please answer with findings and a corrected implementation."""
    quantitative = "Run a simulation and classify chaotic regimes from Lyapunov metrics."

    assert looks_scoped_delegation_goal(non_quantitative) is True
    assert looks_quantitative_rigor_goal(non_quantitative) is False
    assert looks_scoped_delegation_goal(schema_inspection) is True
    assert looks_quantitative_rigor_goal(schema_inspection) is False
    assert looks_scoped_delegation_goal(negated_live_debug) is True
    assert looks_quantitative_rigor_goal(negated_live_debug) is False
    assert looks_scoped_delegation_goal(quantitative) is True
    assert looks_quantitative_rigor_goal(quantitative) is True


def test_runtime_prompt_suffix_appends_contract_only_for_pro_intelligence():
    from ultra_deepagents.agent import build_runtime_prompt_suffix

    base = _code_execution_context().to_payload()

    pro = AgentRunContext(**{**base, "workflow_hint": {"id": "pro_mode"}})
    suffix = build_runtime_prompt_suffix(pro, elapsed_seconds=95)
    assert "Results contract" in suffix
    assert "Active run context:" in suffix
    assert "1m35s" in suffix
    assert "wall-clock" in suffix

    high = AgentRunContext(**base)
    high_suffix = build_runtime_prompt_suffix(high, elapsed_seconds=5)
    assert "Results contract" not in high_suffix
    assert "Active run context:" in high_suffix

    chat = AgentRunContext(
        **{**base, "goal": "Say hello.", "workflow_hint": {"id": "pro_mode"}}
    )
    assert "Results contract" not in build_runtime_prompt_suffix(chat, elapsed_seconds=5)

    code_debug = AgentRunContext(
        **{
            **base,
            "goal": "Analyze this Python code and debug the workflow.",
            "workflow_hint": {"id": "pro_mode"},
        }
    )
    code_suffix = build_runtime_prompt_suffix(code_debug, elapsed_seconds=5)
    assert "Results contract" not in code_suffix
    assert "Active run context:" in code_suffix

    assert build_runtime_prompt_suffix(None) == ""


def _settings_factory() -> RuntimeSettings:
    return RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
    )


def test_domain_guidance_is_gated_to_relevant_runs():
    """Paper/RareSpot guidance must not load on runs whose tools are absent
    (no ~650-tok phantom-tool guidance), while report-only RareSpot runs — which
    have no inference tool but need the report workflow — still get it."""
    settings = _settings_factory()

    generic = AgentRunContext(
        assistant_id="a", org_id="o", user_id="u", project_id="p", thread_id="t",
        run_id="r1", goal="Explain what a Lyapunov exponent measures in 3 sentences.",
    )
    prompt = build_system_prompt(settings, generic)
    assert "rarespot_ecology_inference" not in prompt
    assert "render_paper_page" not in prompt
    assert "search_paper" not in prompt

    paper_ctx = AgentRunContext(
        assistant_id="a", org_id="o", user_id="u", project_id="p", thread_id="t",
        run_id="r2", goal="Summarize this https://arxiv.org/abs/1706.03762 paper.",
    )
    assert "render_paper_page" in build_system_prompt(settings, paper_ctx)
    assert "rarespot_ecology_inference" not in build_system_prompt(settings, paper_ctx)

    rarespot_ctx = AgentRunContext(
        assistant_id="a", org_id="o", user_id="u", project_id="p", thread_id="t",
        run_id="r3", goal="Detect prairie dog burrows in these survey images.",
    )
    assert "rarespot_ecology_inference" in build_system_prompt(settings, rarespot_ctx)

    # Report-only RareSpot: no inference tool registered, but the report-only
    # workflow guidance must still load (gate on domain, not on tool predicate).
    report_only_ctx = AgentRunContext(
        assistant_id="a", org_id="o", user_id="u", project_id="p", thread_id="t",
        run_id="r4",
        goal="Write a combined report across all my RareSpot runs in this chat.",
    )
    assert "rarespot" in build_system_prompt(settings, report_only_ctx).lower()


def test_literature_reviewer_subagent_does_not_carry_skills(monkeypatch, tmp_path: Path):
    """Skills go to the computational subagents only; literature-reviewer is
    page-grounded paper review and should not re-pay the SkillsMiddleware cost."""
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "compiled-agent"

    monkeypatch.setattr("ultra_deepagents.agent.create_deep_agent", fake_create_deep_agent)
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
        workspace_root=str(tmp_path / "workspaces"),
        memory_root=str(tmp_path / "memory"),
        artifact_root=str(tmp_path / "artifacts"),
    )
    # Goal triggers BOTH paper review (arxiv) and computational delegation.
    context = AgentRunContext(
        assistant_id="ultra-research-agent", org_id="o", user_id="u", project_id="p",
        thread_id="t", run_id="run-lit-skills",
        goal="Reproduce the simulation from https://arxiv.org/abs/1706.03762 and analyze the metrics.",
    )
    build_research_agent(
        settings, model=object(),
        workspace_dir=tmp_path / "workspaces" / "run-lit-skills", context=context,
    )

    by_name = {s["name"]: s for s in captured["subagents"]}
    assert "literature-reviewer" in by_name
    assert "code-runner" in by_name
    assert "data-analyst" not in by_name
    assert by_name["code-runner"]["skills"] == SKILLS_SOURCES
    assert "skills" not in by_name["literature-reviewer"]


def test_subagents_never_set_permissions_so_they_inherit_parent_denies(monkeypatch, tmp_path: Path):
    """Subagents must NOT carry their own `permissions` key: deepagents REPLACES
    (not merges) parent permissions when a subagent sets them, which would silently
    drop the /skills/** and /policies/** write-denies on the subagent. Inheriting is
    the only way the read-only guards survive into delegated work."""
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "compiled-agent"

    monkeypatch.setattr("ultra_deepagents.agent.create_deep_agent", fake_create_deep_agent)
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
        workspace_root=str(tmp_path / "workspaces"),
        memory_root=str(tmp_path / "memory"),
        artifact_root=str(tmp_path / "artifacts"),
    )
    context = AgentRunContext(
        assistant_id="ultra-research-agent", org_id="o", user_id="u", project_id="p",
        thread_id="t", run_id="run-perm",
        goal="Reproduce the simulation from https://arxiv.org/abs/1706.03762 and analyze metrics.",
    )
    build_research_agent(
        settings, model=object(),
        workspace_dir=tmp_path / "workspaces" / "run-perm", context=context,
    )
    assert captured["subagents"], "expected scoped + paper subagents to register"
    for subagent in captured["subagents"]:
        assert "permissions" not in subagent, (
            f"{subagent['name']} sets its own permissions, which would REPLACE the "
            "parent /skills/** and /policies/** write-denies"
        )


def _write_skill(skills_root: Path, name: str = "demo") -> Path:
    skill_dir = skills_root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: A demo skill for tests.\n---\n\nDo the thing.\n"
    )
    return skills_root


def _build_real_research_agent(tmp_path: Path, skills_root: Path):
    """Build the REAL agent (no monkeypatched create_deep_agent) so the
    FilesystemMiddleware permission/route guard actually runs."""
    model = ToolCallingFakeOpenAIModel(responses=[AIMessage(content="ok")])
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
        workspace_root=str(tmp_path / "ws"),
        memory_root=str(tmp_path / "mem"),
        artifact_root=str(tmp_path / "art"),
        skills_root=str(skills_root),
    )
    context = AgentRunContext(
        assistant_id="ultra-research-agent", org_id="o", user_id="u",
        project_id="p", thread_id="t", run_id="run-skills",
        goal="Summarize findings.",
    )
    return build_research_agent(
        settings, model=model,
        workspace_dir=tmp_path / "ws" / "run-skills", context=context,
    )


def test_resolve_memory_permissions_drops_skills_deny_only_when_skills_absent(tmp_path: Path):
    empty = tmp_path / "noskills"
    empty.mkdir()
    with_skills = _write_skill(tmp_path / "skills")
    base = dict(openai_base_url="http://x/v1", openai_model="deepseek_v4")
    deny_absent = {
        path
        for rule in resolve_memory_permissions(RuntimeSettings(**base, skills_root=str(empty)))
        for path in rule.paths
    }
    deny_present = {
        path
        for rule in resolve_memory_permissions(RuntimeSettings(**base, skills_root=str(with_skills)))
        for path in rule.paths
    }
    # The /skills/** deny must drop in lockstep with the (absent) /skills/ route...
    assert "/skills/**" not in deny_absent
    # ...while the always-registered /memories/ and /policies/ routes keep their denies.
    assert "/policies/**" in deny_absent
    assert "/memories/user_profile.md" in deny_absent
    # With skills present, the full read-only deny set applies.
    assert "/skills/**" in deny_present


def test_build_research_agent_with_real_sandbox_backend_builds_with_and_without_skills(tmp_path: Path):
    # Regression: an unconditional /skills/** deny against a conditional /skills/
    # route made FilesystemMiddleware raise NotImplementedError at agent
    # construction (execute-capable sandbox default) whenever skills were absent.
    # Building the real agent both ways must not raise.
    with_skills = _write_skill(tmp_path / "with" / "skills")
    without_skills = tmp_path / "without"
    without_skills.mkdir()
    assert _build_real_research_agent(tmp_path / "with", with_skills) is not None
    assert _build_real_research_agent(tmp_path / "without", without_skills) is not None
