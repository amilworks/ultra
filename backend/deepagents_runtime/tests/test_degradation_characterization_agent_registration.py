"""Selective natural-prompt routing for bounded materials analysis tools."""

from __future__ import annotations

from typing import Any

import pytest
import ultra_deepagents.agent as agent_module
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.evaluation_profiles import MATERIALS_CLEANROOM_PROFILE

DEGRADATION_TOOL_NAMES = {
    "materials_convert_uniform_corrosion",
    "materials_evaluate_mode_i_lefm",
    "materials_evaluate_norton_arrhenius_creep",
    "materials_evaluate_oxidation_mass_gain",
    "materials_fit_paris_law",
}
CHARACTERIZATION_TOOL_NAMES = {
    "materials_calculate_diffraction_profile_metrics",
    "materials_fit_held_out_rigid_registration",
}
PROCESSING_SUPPORT_TOOL_NAME = "materials_processing_method_support"
ALL_TOOL_NAMES = {
    *DEGRADATION_TOOL_NAMES,
    *CHARACTERIZATION_TOOL_NAMES,
    PROCESSING_SUPPORT_TOOL_NAME,
}


def _context(
    goal: str,
    *,
    evaluation_profile: str = "",
    selection_context: dict[str, Any] | None = None,
) -> AgentRunContext:
    return AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="org",
        user_id="user",
        project_id="project",
        thread_id="thread",
        run_id="run-degradation-characterization",
        goal=goal,
        evaluation_profile=evaluation_profile,
        selection_context=(
            {"suggested_domain": "materials_science"}
            if selection_context is None
            else selection_context
        ),
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
        "Run a bounded Mode-I LEFM screen for this crack geometry.",
        "Fit a Paris law to these fatigue crack-growth rows and score the holdout.",
        "Evaluate Norton-Arrhenius secondary creep inside this calibration domain.",
        "Calculate parabolic oxidation mass gain for this isothermal exposure.",
        "Convert corrosion current density into uniform corrosion penetration with Faraday's law.",
        "Fit the Paris equation to these crack-growth measurements.",
        "Calculate the Norton law creep rate at the requested stress and temperature.",
        "Evaluate this calibrated oxidation mass-gain law at 1073 K.",
        "Use Faraday law to calculate corrosion penetration from the measured current.",
    ],
)
def test_degradation_calculations_register_bounded_tool_group(goal: str) -> None:
    assert agent_module._should_register_degradation_tools(_context(goal)) is True


@pytest.mark.parametrize("tool_name", sorted(DEGRADATION_TOOL_NAMES))
def test_exact_degradation_tool_requests_register_bounded_tool_group(tool_name: str) -> None:
    goal = f"Call {tool_name} exactly once and capture its typed response."
    assert agent_module._should_register_degradation_tools(_context(goal)) is True


@pytest.mark.parametrize(
    "goal",
    [
        "Explain materials_evaluate_oxidation_mass_gain and its schema.",
        "How do I use materials_evaluate_oxidation_mass_gain?",
        "What happens if I call materials_evaluate_oxidation_mass_gain?",
        "Do not call materials_evaluate_oxidation_mass_gain; summarize oxidation instead.",
        "```\nCall materials_evaluate_oxidation_mass_gain\n```",
        "Call materials_evaluate_oxidation_mass_gain_preview.",
        "Call preview_materials_evaluate_oxidation_mass_gain.",
    ],
)
def test_incidental_or_nonexact_degradation_tool_mentions_do_not_register(goal: str) -> None:
    assert agent_module._should_register_degradation_tools(_context(goal)) is False


def test_exact_live_oxidation_prompt_routes_without_materials_ui_hint() -> None:
    context = _context(
        "Call materials_evaluate_oxidation_mass_gain exactly once for a linear "
        "areal-mass-gain model and capture its typed response.",
        selection_context={},
    )

    assert agent_module._should_register_degradation_tools(context) is True
    assert agent_module.is_materials_context(context) is True


@pytest.mark.parametrize(
    "goal",
    [
        "Compute Rp, Rwp, and Rexp for these observed and calculated diffraction profiles.",
        "Fit a rigid registration and score held-out EBSD-to-APT fiducials.",
        "Register these 4D-STEM and TEM landmarks with a calibration/holdout split.",
    ],
)
def test_characterization_validation_registers_only_bounded_validators(goal: str) -> None:
    assert agent_module._should_register_characterization_validation_tools(_context(goal)) is True


@pytest.mark.parametrize(
    "goal",
    [
        "Analyze executable support for Scheil, back diffusion, mobility, precipitation, and phase field.",
        "Can the platform run a phase-field solidification simulation?",
        "Evaluate readiness for coupled solidification and back diffusion.",
        "Call materials_processing_method_support and report the exact boundary.",
    ],
)
def test_processing_boundary_questions_register_zero_argument_support(goal: str) -> None:
    assert agent_module._should_register_processing_support_tool(_context(goal)) is True


@pytest.mark.parametrize(
    "goal",
    [
        "Explain fatigue mechanisms in nickel superalloys.",
        "Calculate resolved shear and Schmid factors for this FCC grain.",
        "Run binary KWN precipitation kinetics for the selected TDB.",
        "Calculate CALPHAD equilibrium phase fractions at 1173 K.",
        "Refine this measured XRD pattern with a Rietveld model.",
        "Characterize the facets of these pyramidal nanoparticles.",
    ],
)
def test_unrelated_or_explanatory_prompts_do_not_register_new_numerical_groups(
    goal: str,
) -> None:
    context = _context(goal)
    assert agent_module._should_register_degradation_tools(context) is False
    assert agent_module._should_register_characterization_validation_tools(context) is False


@pytest.mark.parametrize(
    "goal",
    [
        "Run binary KWN precipitation kinetics for the selected TDB.",
        "Run a Scheil solidification calculation using this governed database.",
        "Calculate a one-dimensional diffusion profile for this couple.",
    ],
)
def test_real_processing_execution_does_not_get_static_support_tool(goal: str) -> None:
    assert agent_module._should_register_processing_support_tool(_context(goal)) is False


def _tool_names(captured: dict[str, Any]) -> set[str]:
    return {str(getattr(tool, "name", "")) for tool in captured["tools"]}


def _delegate_tool_names(captured: dict[str, Any]) -> set[str]:
    code_runner = next(item for item in captured["subagents"] if item["name"] == "code-runner")
    return {str(getattr(tool, "name", "")) for tool in code_runner.get("tools", [])}


def test_agent_registers_only_degradation_group_for_parent_and_delegate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_agent(
        monkeypatch,
        _context("Fit a Paris law to fatigue crack-growth rows with held-out interpolation."),
    )

    assert _tool_names(captured) & ALL_TOOL_NAMES == DEGRADATION_TOOL_NAMES
    assert _delegate_tool_names(captured) & ALL_TOOL_NAMES == DEGRADATION_TOOL_NAMES
    prompt = captured["system_prompt"]
    assert "selected degradation skill" in prompt
    assert "use the matching bounded typed tool directly" in prompt
    assert "Never invent placeholder hashes" in prompt
    assert "deterministic typed input rejection is terminal" in prompt
    assert "do not predict fracture/fatigue/creep" in prompt
    assert "advanced-characterization validation" not in prompt


def test_exact_oxidation_tool_prompt_registers_typed_surface_without_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_agent(
        monkeypatch,
        _context(
            "Call materials_evaluate_oxidation_mass_gain exactly once for a linear "
            "areal-mass-gain model and capture its typed response."
        ),
    )

    assert _tool_names(captured) & ALL_TOOL_NAMES == DEGRADATION_TOOL_NAMES
    assert _delegate_tool_names(captured) & ALL_TOOL_NAMES == DEGRADATION_TOOL_NAMES
    assert "selected degradation skill" in captured["system_prompt"]


def test_agent_registers_only_characterization_group_for_parent_and_delegate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_agent(
        monkeypatch,
        _context("Fit a rigid registration and score held-out EBSD-to-APT fiducials."),
    )

    assert _tool_names(captured) & ALL_TOOL_NAMES == CHARACTERIZATION_TOOL_NAMES
    assert _delegate_tool_names(captured) & ALL_TOOL_NAMES == CHARACTERIZATION_TOOL_NAMES
    prompt = captured["system_prompt"]
    assert "advanced-characterization validation" in prompt
    assert "known correspondences" in prompt
    assert "selected degradation skill" not in prompt


def test_agent_registers_only_zero_argument_processing_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_agent(
        monkeypatch,
        _context("Can the platform run phase-field solidification? Report capability readiness."),
    )

    assert _tool_names(captured) & ALL_TOOL_NAMES == {PROCESSING_SUPPORT_TOOL_NAME}
    assert _delegate_tool_names(captured) & ALL_TOOL_NAMES == {PROCESSING_SUPPORT_TOOL_NAME}
    support_tool = next(
        tool
        for tool in captured["tools"]
        if getattr(tool, "name", "") == PROCESSING_SUPPORT_TOOL_NAME
    )
    assert support_tool.args == {}
    prompt = captured["system_prompt"]
    assert "processing-method boundary" in prompt
    assert "zero-argument static support discovery" in prompt
    assert "Never replace unsupported phase-field" in prompt


def test_generic_materials_selection_does_not_eagerly_add_eight_schemas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_agent(
        monkeypatch,
        _context("Summarize the supplied alloy-development notes."),
    )

    assert not (_tool_names(captured) & ALL_TOOL_NAMES)


def test_cleanroom_profile_never_registers_new_tool_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_agent(
        monkeypatch,
        _context(
            "Fit a Paris law and a held-out rigid registration, then assess phase-field support.",
            evaluation_profile=MATERIALS_CLEANROOM_PROFILE,
        ),
    )

    assert not (_tool_names(captured) & ALL_TOOL_NAMES)
    prompt = captured["system_prompt"]
    assert "selected degradation skill" not in prompt
    assert "advanced-characterization validation" not in prompt
    assert "processing-method boundary" not in prompt
