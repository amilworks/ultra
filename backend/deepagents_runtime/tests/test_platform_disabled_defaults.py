"""The materials platform is OFF by default: a non-materials deployment must
carry no materials tools, no materials skill listing, and no materials prompt
framing — even for a prompt that trips the shared (dual-use) materials tokens.

This module is intentionally NOT named for a materials domain, so the suite's
autouse fixture leaves the platform disabled here.
"""

from __future__ import annotations

import pytest

from ultra_deepagents import agent as agent_mod
from ultra_deepagents.agent import _MaterialsFilteredSkillsBackend
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext


def _ctx(goal: str) -> AgentRunContext:
    return AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="u",
        project_id="p",
        thread_id="t",
        run_id="r",
        goal=goal,
    )


_MATERIALS_GOAL = "Compute the equilibrium phase fractions for Al-Cu at 800K using CALPHAD."


def test_disabled_registers_no_materials_tools(monkeypatch):
    monkeypatch.delenv("ULTRA_DEEPAGENTS_MATERIALS_ENABLED", raising=False)
    captured: dict = {}
    monkeypatch.setattr(agent_mod, "create_deep_agent", lambda **kw: captured.update(kw) or "compiled")
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:9/v1",
        openai_model="deepseek_v4",
        control_base_url="http://127.0.0.1:9",
    )
    agent_mod.build_research_agent(settings, model=object(), backend=object(), context=_ctx(_MATERIALS_GOAL))
    names = {str(getattr(t, "name", "")) for t in captured["tools"]}
    materials_like = {
        n for n in names if n.startswith(("calphad_", "materials_", "kinetics_")) or "crystal_plasticity" in n
    }
    assert not materials_like, f"materials tools registered while disabled: {materials_like}"


def test_disabled_suppresses_materials_brief_but_enabled_shows_it():
    goal = "Analyze this sensor data waveform for the alloy microstructure."
    # Disabled (default): no materials framing even though the prompt trips tokens.
    import os

    os.environ.pop("ULTRA_DEEPAGENTS_MATERIALS_ENABLED", None)
    brief_off = agent_mod.build_run_context_brief(_ctx(goal))
    assert "suggested_domain: materials" not in brief_off

    # Enabled: the materials routing brief appears.
    os.environ["ULTRA_DEEPAGENTS_MATERIALS_ENABLED"] = "true"
    try:
        brief_on = agent_mod.build_run_context_brief(_ctx(goal))
        assert "suggested_domain: materials" in brief_on
    finally:
        os.environ.pop("ULTRA_DEEPAGENTS_MATERIALS_ENABLED", None)


def test_materials_skills_hidden_from_listing(tmp_path):
    (tmp_path / "web-research").mkdir()
    (tmp_path / "web-research" / "SKILL.md").write_text("---\nname: web-research\n---\n")
    (tmp_path / "materials-structure-thermo").mkdir()
    (tmp_path / "materials-structure-thermo" / "SKILL.md").write_text("---\nname: materials-structure-thermo\n---\n")

    backend = _MaterialsFilteredSkillsBackend(str(tmp_path), virtual_mode=True)
    listed = {
        str(entry.get("path", "")).rstrip("/").rsplit("/", 1)[-1]
        for entry in (backend.ls("/").entries or [])
    }
    assert "web-research" in listed
    assert "materials-structure-thermo" not in listed
