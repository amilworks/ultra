"""Consolidation agent scaffold tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.consolidation import (
    CONSOLIDATION_SYSTEM_PROMPT,
    build_consolidation_agent,
)
from ultra_deepagents.context import AgentRunContext


def _settings(tmp_path: Path) -> RuntimeSettings:
    return RuntimeSettings(
        openai_base_url="http://127.0.0.1:9/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
        control_worker_token="trace-worker-secret",
        workspace_root=str(tmp_path / "ws"),
        memory_root=str(tmp_path / "mem"),
        artifact_root=str(tmp_path / "art"),
    )


def _context() -> AgentRunContext:
    return AgentRunContext(
        assistant_id="ultra-consolidation-agent",
        org_id="org-1",
        user_id="researcher-7",
        project_id="proj",
        thread_id="thread-consolidation",
        run_id="run-consolidation",
        goal="Consolidate recent research sessions.",
    )


def test_build_consolidation_agent_compiles_with_episodic_tool(monkeypatch):
    captured: dict = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "compiled-consolidation-agent"

    monkeypatch.setattr("ultra_deepagents.consolidation.create_deep_agent", fake_create_deep_agent)
    tmp = Path(tempfile.mkdtemp())
    settings = _settings(tmp)

    agent = build_consolidation_agent(
        settings, context=_context(), workspace_dir=tmp / "ws" / "run-consolidation"
    )

    assert agent == "compiled-consolidation-agent"
    assert captured["name"] == "ultra-consolidation-agent"
    tool_names = {getattr(t, "name", "") for t in captured["tools"]}
    assert tool_names == {"search_past_research"}
    # Reuses the per-user memory contract: same memory files + write protections.
    assert "/memories/research_context/INDEX.md" in captured["memory"]
    deny_paths = {p for rule in captured["permissions"] if rule.mode == "deny" for p in rule.paths}
    assert "/memories/user_profile.md" in deny_paths
    assert "research_context" in captured["system_prompt"]
    assert "search_past_research" in captured["system_prompt"]


def test_consolidation_prompt_is_non_user_facing_and_merge_oriented():
    prompt = CONSOLIDATION_SYSTEM_PROMPT.lower()
    assert "you only read recent history" in prompt
    assert "merge" in prompt
    assert "do not fabricate" in prompt
