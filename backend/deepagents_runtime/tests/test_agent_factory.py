from pathlib import Path

from ultra_deepagents.agent import build_research_agent, build_sandbox_backend
from ultra_deepagents.code_execution.docker import DockerSandboxBackend
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext


def test_build_research_agent_passes_current_deepagents_contract(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "compiled-agent"

    monkeypatch.setattr("ultra_deepagents.agent.create_deep_agent", fake_create_deep_agent)
    settings = RuntimeSettings(
        openai_base_url="http://vrl-h200.ece.ucsb.edu:9393/v1",
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
    assert captured["tools"] == [fake_tool]
    assert captured["context_schema"] is AgentRunContext
    assert captured["memory"] == ["/memories/preferences.md", "/memories/research_context.md"]
    assert "/memories/" in captured["system_prompt"]
    assert "/outputs/" in captured["system_prompt"]
    assert "subagents" in captured["system_prompt"].lower()
    assert "sandbox execution" in captured["system_prompt"].lower()

    subagent_names = {subagent["name"] for subagent in captured["subagents"]}
    assert {
        "literature-reviewer",
        "methods-critic",
        "imaging-analyst",
        "statistics-analyst",
    }.issubset(subagent_names)


def test_build_sandbox_backend_uses_runtime_settings(tmp_path: Path):
    settings = RuntimeSettings(
        openai_base_url="http://vrl-h200.ece.ucsb.edu:9393/v1",
        openai_model="deepseek_v4",
        sandbox_image="bisque-ultra-codeexec:test",
        sandbox_network="none",
        sandbox_cpus=3.5,
        sandbox_memory="8g",
        sandbox_pids_limit=512,
        sandbox_timeout_seconds=1800,
        sandbox_output_limit_bytes=500_000,
    )

    backend = build_sandbox_backend(settings, workspace_dir=tmp_path / "workspace")

    assert isinstance(backend, DockerSandboxBackend)
    assert backend.workspace_dir == tmp_path / "workspace"
    assert backend.config.image == "bisque-ultra-codeexec:test"
    assert backend.config.network == "none"
    assert backend.config.cpus == 3.5
    assert backend.config.memory == "8g"
    assert backend.config.pids_limit == 512
    assert backend.config.timeout_seconds == 1800
    assert backend.config.output_limit_bytes == 500_000
