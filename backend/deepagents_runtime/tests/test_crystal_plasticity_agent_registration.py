"""Natural-prompt routing for the bounded crystal-plasticity tool surface."""

from __future__ import annotations

from typing import Any

import pytest
import ultra_deepagents.agent as agent_module
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.evaluation_profiles import MATERIALS_CLEANROOM_PROFILE


def _context(goal: str, *, evaluation_profile: str = "") -> AgentRunContext:
    return AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="org",
        user_id="user",
        project_id="project",
        thread_id="thread",
        run_id="run-crystal-plasticity",
        goal=goal,
        evaluation_profile=evaluation_profile,
        selection_context={"suggested_domain": "materials_science"},
    )


def _capture_agent(
    monkeypatch: pytest.MonkeyPatch,
    context: AgentRunContext,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        agent_module,
        "create_deep_agent",
        lambda **kwargs: captured.update(kwargs) or "compiled-agent",
    )
    monkeypatch.setattr(agent_module, "build_builder_subagent", lambda *args, **kwargs: None)
    agent_module.build_research_agent(
        RuntimeSettings(
            openai_base_url="http://127.0.0.1:9/v1",
            openai_model="deepseek_v4",
        ),
        model=object(),
        backend=object(),
        context=context,
    )
    return captured


@pytest.mark.parametrize(
    "goal",
    [
        (
            "Calculate the crystal-plasticity geometry for an FCC grain using only "
            "fcc-{111}<110>; report every Schmid factor and resolved shear stress."
        ),
        (
            "Generate the HCP first-order pyramidal c+a family without assuming c/a, "
            "and report whether a numerical slip-system calculation is supportable."
        ),
        (
            "Build and validate a schema-v1 FCC CPFE input contract, then attempt "
            "execution without substituting a toy constitutive model."
        ),
        "Calculate CRSS-normalized resolved shear for this single-phase BCC grain.",
        "Evaluate HCP basal slip for these measured alpha-phase orientations.",
        "Compare HCP prismatic slip and pyramidal slip geometry using the measured c/a.",
        "Enumerate the FCC octahedral slip family for this gamma grain.",
    ],
)
def test_cp_natural_prompts_register_first_class_typed_surface(goal: str) -> None:
    assert agent_module._should_register_crystal_plasticity_tools(_context(goal)) is True


@pytest.mark.parametrize(
    "goal",
    [
        "Fit a Paris relation to the fatigue crack-growth rows.",
        "Run binary KWN precipitation kinetics for the selected TDB.",
        "Calculate CALPHAD equilibrium phase fractions at 1173 K.",
        "Explain plastic recycling systems for this policy memo.",
        "Characterize the facets of these pyramidal nanoparticles.",
    ],
)
def test_unrelated_materials_prompts_do_not_carry_crystal_plasticity_schema(goal: str) -> None:
    assert agent_module._should_register_crystal_plasticity_tools(_context(goal)) is False


def test_agent_registers_typed_cp_tools_for_parent_and_delegate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_agent(
        monkeypatch,
        _context(
            "Calculate resolved shear and Schmid factors for this FCC grain, then validate "
            "the CPFE input contract."
        ),
    )

    parent_tool_names = {str(getattr(tool, "name", "")) for tool in captured["tools"]}
    assert {
        "materials_analyze_crystal_slip",
        "materials_validate_cpfe_contract",
    } <= parent_tool_names
    code_runner = next(item for item in captured["subagents"] if item["name"] == "code-runner")
    delegated_tool_names = {str(getattr(tool, "name", "")) for tool in code_runner.get("tools", [])}
    assert {
        "materials_analyze_crystal_slip",
        "materials_validate_cpfe_contract",
    } <= delegated_tool_names
    prompt = captured["system_prompt"]
    assert "/skills/materials-crystal-plasticity/SKILL.md" in prompt
    assert "/skills/materials-mechanics-degradation/SKILL.md" not in prompt
    assert "call materials_analyze_crystal_slip" in prompt
    assert "materials_validate_cpfe_contract" in prompt
    assert "directly before considering execute or code-runner delegation" in prompt
    assert "Copy analysis_artifact.canonical_json" in prompt
    assert "do not discover or reconstruct the validation API" in prompt
    assert "deterministic typed input rejection is terminal" in prompt
    assert "Do not create unrequested output files" in prompt


@pytest.mark.parametrize(
    "goal",
    [
        (
            "Calculate the crystal-plasticity geometry for an FCC grain using only "
            "fcc-{111}<110>. The active crystal-to-sample rotation is the identity matrix. "
            "Apply a sample-frame Cauchy stress of diag(0,0,100) MPa and a uniaxial load "
            "axis [0,0,1]. Use materials_analyze_crystal_slip directly, save every system "
            "ID, Schmid factor, and resolved shear stress, run a 123 MPa hydrostatic "
            "zero-shear control, and do not claim that slip occurred or CPFE was solved."
        ),
        (
            "Build a schema-v1 FCC CPFE input contract for phase gamma with m-3m, identity "
            "crystal-to-sample orientation, SI units, fcc-{111}<110>, CRSS 45 MPa, and a "
            "structurally complete Voce hardening block. Use content hashes made of 64 a, "
            "b, and c characters for phase, CRSS, and hardening provenance. Call "
            "materials_validate_cpfe_contract with attempt_execution=true. Report contract "
            "validity separately from execution support and do not substitute a toy model."
        ),
    ],
    ids=["CP-01", "CP-03"],
)
def test_cp_acceptance_prompts_route_directly_without_a_new_subagent(
    monkeypatch: pytest.MonkeyPatch,
    goal: str,
) -> None:
    captured = _capture_agent(monkeypatch, _context(goal))

    tool_names = {str(getattr(tool, "name", "")) for tool in captured["tools"]}
    assert {
        "materials_analyze_crystal_slip",
        "materials_validate_cpfe_contract",
    } <= tool_names
    assert "/skills/materials-crystal-plasticity/SKILL.md" in captured["system_prompt"]
    assert "/skills/materials-mechanics-degradation/SKILL.md" not in captured["system_prompt"]
    assert [subagent["name"] for subagent in captured["subagents"]] == ["code-runner"]


def test_non_cp_materials_run_does_not_expand_tool_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_agent(
        monkeypatch,
        _context("Fit a rigid registration and score held-out EBSD-to-APT fiducials."),
    )

    tool_names = {str(getattr(tool, "name", "")) for tool in captured["tools"]}
    assert "materials_analyze_crystal_slip" not in tool_names
    assert "materials_validate_cpfe_contract" not in tool_names
    assert "call materials_analyze_crystal_slip" not in captured["system_prompt"]


def test_cleanroom_profile_never_registers_cp_tool_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_agent(
        monkeypatch,
        _context(
            "Calculate resolved shear on every FCC slip system.",
            evaluation_profile=MATERIALS_CLEANROOM_PROFILE,
        ),
    )

    tool_names = {str(getattr(tool, "name", "")) for tool in captured["tools"]}
    assert "materials_analyze_crystal_slip" not in tool_names
    assert "materials_validate_cpfe_contract" not in tool_names
    assert "call materials_analyze_crystal_slip" not in captured["system_prompt"]
