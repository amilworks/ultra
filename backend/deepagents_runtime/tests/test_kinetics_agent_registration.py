"""Natural-prompt routing for the separately pinned Kawin tool surface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import ultra_deepagents.agent as agent_module
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.materials.processing_kinetics import processing_method_support


def _context(goal: str) -> AgentRunContext:
    return AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="org",
        user_id="user",
        project_id="project",
        thread_id="thread",
        run_id="run-kinetics",
        goal=goal,
        selection_context={"suggested_domain": "materials_science"},
    )


def test_processing_support_matrix_matches_the_qualified_isolated_tool_scope() -> None:
    support = processing_method_support()

    assert support["back_diffusion"]["status"] == "qualified_isolated_runtime"
    assert support["back_diffusion"]["scope"] == "post_solidification_single_phase_1d_only"
    assert support["mobility_diffusion"]["tools"] == [
        "materials_transport_coefficients",
        "materials_run_diffusion_1d",
    ]
    assert support["precipitation"]["tool"] == "materials_run_binary_precipitation_kwn"
    assert "binary isothermal spherical KWN" in support["precipitation"]["scope"]
    assert support["phase_field"]["status"] == "requires_external_hpc_solver"


@pytest.mark.parametrize(
    "goal",
    [
        (
            "Using my selected Al-Zr TDB, calculate tracer diffusivity and the "
            "interdiffusion coefficient in FCC_A1 at 723.15 K and X(ZR)=0.004."
        ),
        (
            "Run post-solidification back diffusion for my selected mobility assessment "
            "on a 10 micrometre dendrite-arm domain and report mass closure."
        ),
        (
            "Run binary KWN precipitation kinetics with Kawin for my selected Al-Zr "
            "database and retain the particle-size distribution."
        ),
        (
            "Use Kawin to run a three-dimensional phase-field simulation with coupled "
            "moving-interface solidification and back diffusion."
        ),
        (
            "With my selected kinetic TDB, run isothermal single-phase diffusion in FCC_A1 "
            "and return the final concentration profile."
        ),
        ("Calculate diffusion coefficients from my selected mobility database at 900 K."),
        (
            "Run a binary precipitation simulation for one matrix and one precipitate and "
            "test particle-bin convergence."
        ),
    ],
)
def test_natural_materials_kinetics_prompts_route_to_typed_surface(goal: str) -> None:
    assert agent_module._should_register_kinetics_tools(_context(goal)) is True


@pytest.mark.parametrize(
    "goal",
    [
        "Analyze diffusion of information through this social network.",
        "Calculate CALPHAD equilibrium phase fractions at 1173 K.",
        "Index this EBSD map and calculate grain boundaries.",
    ],
)
def test_unrelated_prompts_do_not_carry_kinetics_tool_schema(goal: str) -> None:
    assert agent_module._should_register_kinetics_tools(_context(goal)) is False


def test_agent_registers_three_tools_with_separate_immutable_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    built_backends: list[Any] = []
    real_builder = agent_module.build_kinetics_tools

    def capture_builder(settings: Any, *, backend: Any, upload_roots: Any) -> list[Any]:
        built_backends.append(backend)
        return real_builder(settings, backend=backend, upload_roots=upload_roots)

    kinetics_image = "sha256:" + "2" * 64
    monkeypatch.setenv("ULTRA_MATERIALS_KINETICS_RUNTIME_IMAGE", "ultra-materials-kinetics:py311")
    monkeypatch.setenv("ULTRA_MATERIALS_KINETICS_RUNTIME_IMAGE_ID", kinetics_image)
    monkeypatch.setattr(
        agent_module,
        "resolve_docker_image_id",
        lambda image: (
            kinetics_image if image == "ultra-materials-kinetics:py311" else "sha256:" + "1" * 64
        ),
    )
    monkeypatch.setattr(agent_module, "build_kinetics_tools", capture_builder)
    monkeypatch.setattr(
        agent_module,
        "create_deep_agent",
        lambda **kwargs: captured.update(kwargs) or "compiled-agent",
    )
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:9/v1",
        openai_model="deepseek_v4",
        sandbox_cpus=4,
        sandbox_memory="16g",
        sandbox_pids_limit=1024,
    )

    result = agent_module.build_research_agent(
        settings,
        model=object(),
        workspace_dir=tmp_path / "workspace",
        artifact_dir=tmp_path / "outputs",
        context=_context("Calculate tracer diffusivity with Kawin from my selected TDB."),
    )

    assert result == "compiled-agent"
    tool_names = {getattr(item, "name", "") for item in captured["tools"]}
    assert {
        "materials_transport_coefficients",
        "materials_run_diffusion_1d",
        "materials_run_binary_precipitation_kwn",
    } <= tool_names
    assert len(built_backends) == 1
    backend = built_backends[0]
    assert backend.config.image == kinetics_image
    assert backend.config.image != "sha256:" + "1" * 64
    assert backend.config.network == "none"
    assert backend.config.no_new_privileges is True
    assert backend.config.cpus == 2.0
    assert backend.config.memory == "8g"
    assert backend.config.pids_limit == 256
    assert backend.config.timeout_seconds == 30
    assert backend.config.gpus == ""


def test_mutable_or_missing_kinetics_image_disables_product_tool_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setenv("ULTRA_MATERIALS_KINETICS_RUNTIME_IMAGE", "ultra-materials-kinetics:py311")
    monkeypatch.setenv(
        "ULTRA_MATERIALS_KINETICS_RUNTIME_IMAGE_ID", "ultra-materials-kinetics:latest"
    )
    monkeypatch.setattr(
        agent_module,
        "resolve_docker_image_id",
        lambda _image: "sha256:" + "1" * 64,
    )
    monkeypatch.setattr(
        agent_module,
        "create_deep_agent",
        lambda **kwargs: captured.update(kwargs) or "compiled-agent",
    )
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:9/v1",
        openai_model="deepseek_v4",
    )

    agent_module.build_research_agent(
        settings,
        model=object(),
        workspace_dir=tmp_path / "workspace",
        artifact_dir=tmp_path / "outputs",
        context=_context("Calculate tracer diffusivity with Kawin from my selected TDB."),
    )

    tool_names = {getattr(item, "name", "") for item in captured["tools"]}
    assert (
        not {
            "materials_transport_coefficients",
            "materials_run_diffusion_1d",
            "materials_run_binary_precipitation_kwn",
        }
        & tool_names
    )


@pytest.mark.parametrize("reference", ["", "ultra-materials-kinetics:py311"])
def test_missing_or_mismatched_operator_image_binding_disables_kinetics_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reference: str,
) -> None:
    captured: dict[str, Any] = {}
    if reference:
        monkeypatch.setenv("ULTRA_MATERIALS_KINETICS_RUNTIME_IMAGE", reference)
    else:
        monkeypatch.delenv("ULTRA_MATERIALS_KINETICS_RUNTIME_IMAGE", raising=False)
    monkeypatch.setenv("ULTRA_MATERIALS_KINETICS_RUNTIME_IMAGE_ID", "sha256:" + "2" * 64)
    monkeypatch.setattr(
        agent_module,
        "resolve_docker_image_id",
        lambda _image: "sha256:" + "3" * 64,
    )
    monkeypatch.setattr(
        agent_module,
        "create_deep_agent",
        lambda **kwargs: captured.update(kwargs) or "compiled-agent",
    )

    agent_module.build_research_agent(
        RuntimeSettings(
            openai_base_url="http://127.0.0.1:9/v1",
            openai_model="deepseek_v4",
        ),
        model=object(),
        workspace_dir=tmp_path / "workspace",
        artifact_dir=tmp_path / "outputs",
        context=_context("Calculate tracer diffusivity with Kawin from my selected TDB."),
    )

    tool_names = {getattr(item, "name", "") for item in captured["tools"]}
    assert (
        not {
            "materials_transport_coefficients",
            "materials_run_diffusion_1d",
            "materials_run_binary_precipitation_kwn",
        }
        & tool_names
    )
