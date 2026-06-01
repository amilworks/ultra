from pathlib import Path

from deepagents.backends import CompositeBackend

from ultra_deepagents.agent import (
    build_run_context_brief,
    build_agent_backend,
    build_research_agent,
    build_sandbox_backend,
    build_system_prompt,
)
from ultra_deepagents.code_execution.docker import DockerSandboxBackend
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.context_tools import (
    build_tool_capability_manifest,
    stage_artifact,
    stage_uploaded_files,
)
from ultra_deepagents.multimodal import TextOnlyMultimodalMiddleware


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
    assert captured["memory"] == ["/memories/preferences.md", "/memories/research_context.md"]
    assert "/memories/" in captured["system_prompt"]
    assert "/outputs/" in captured["system_prompt"]
    assert "when subagents are available" in captured["system_prompt"].lower()
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
    assert {"execute", "write_file", "read_file", "edit_file", "ls", "glob", "grep"}.issubset(
        builtin_names
    )
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
    }
    assert "selected_tool_names" in manifest


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


def test_research_agent_disables_sync_subagents_for_code_execution_context(monkeypatch):
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

    assert captured["subagents"] == []


def test_text_only_system_prompt_guides_inline_plot_captions_and_updates():
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
    )

    prompt = build_system_prompt(settings)

    assert "caption immediately after each figure" in prompt.lower()
    assert "not all at the end" in prompt.lower()
    assert "error bars" in prompt.lower()
    assert "do not call read_file on image" in prompt.lower()


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
