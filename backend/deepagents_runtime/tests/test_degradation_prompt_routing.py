from __future__ import annotations

from pathlib import Path

import pytest
from ultra_deepagents.agent import (
    build_run_context_brief,
    build_subagents,
    looks_materials_computational_goal,
    looks_scoped_delegation_goal,
)
from ultra_deepagents.context import AgentRunContext


def _context(goal: str) -> AgentRunContext:
    return AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-degradation-routing",
        run_id="run-degradation-routing",
        goal=goal,
        selection_context={"suggested_domain": "materials"},
        workflow_hint={"id": "pro_mode"},
    )


@pytest.mark.parametrize(
    "prompt",
    [
        (
            "Run a bounded Mode-I fracture LEFM screen with an explicit geometry-factor "
            "calibration and small-scale-yielding audit."
        ),
        (
            "Fit this fatigue crack-growth Paris relation using only the calibration rows and "
            "score the held-out interpolation rows."
        ),
        (
            "Evaluate the calibrated Norton Arrhenius secondary creep rate at 200 MPa and "
            "1000 K without predicting rupture life."
        ),
        (
            "Calculate linear and parabolic oxidation areal mass gain inside the measured "
            "isothermal exposure domain."
        ),
        (
            "Convert this corrosion current density to average uniform penetration using the "
            "explicit equivalent mass, density, and current efficiency."
        ),
    ],
)
def test_natural_degradation_prompts_route_to_materials_code_runner(prompt: str) -> None:
    context = _context(prompt)

    assert looks_materials_computational_goal(prompt) is True
    assert looks_scoped_delegation_goal(prompt) is True
    brief = build_run_context_brief(context)
    assert "/skills/materials-mechanics-degradation/SKILL.md" in brief
    assert "/skills/materials-crystal-plasticity/SKILL.md" not in brief
    subagents = build_subagents(context=context, skills_sources=["/skills/"])
    assert [subagent["name"] for subagent in subagents] == ["code-runner"]
    assert subagents[0]["skills"] == ["/skills/"]


def test_degradation_skill_forbids_placeholder_evidence_and_retry() -> None:
    text = (
        Path(__file__).parents[1] / "skills" / "materials-mechanics-degradation" / "SKILL.md"
    ).read_text(encoding="utf-8")

    assert "Never fabricate placeholder or all-zero hashes" in text
    assert "deterministic typed input rejection as terminal" in text
    assert "do not retry with substitute inputs" in text
