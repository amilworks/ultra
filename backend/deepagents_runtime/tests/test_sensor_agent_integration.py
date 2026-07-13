from __future__ import annotations

from typing import Any

import pytest
from ultra_deepagents.agent import build_research_agent
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.evaluation_profiles import MATERIALS_CLEANROOM_PROFILE


def _context(
    *,
    goal: str,
    original_name: str,
    evaluation_profile: str = "",
    typed_sensor: bool = False,
) -> AgentRunContext:
    digest = "a" * 64
    descriptor: dict[str, Any] = {
        "type": "selected_resource",
        "binding_schema": "ultra.selected_resource.v1",
        "authority": "control_resource_catalog",
        "resource_id": "file-selected",
        "file_id": "file-selected",
        "original_name": original_name,
        "sha256": digest,
        "size_bytes": 1024,
    }
    if typed_sensor:
        descriptor["sensor_format"] = {
            "schema": "ultra.sensor-format-binding.v1",
            "authority": "control_resource_catalog",
            "container": "zarr",
            "sensor_schema": "ultra.sensor-series.v1",
            "resource_sha256": digest,
            "detection": "bounded_root_attributes",
        }
    return AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="org",
        user_id="user",
        project_id="project",
        thread_id="thread",
        run_id="run-sensor-agent",
        goal=goal,
        evaluation_profile=evaluation_profile,
        selected_file_ids=("file-selected",),
        resource_descriptors=(descriptor,),
    )


def _capture_agent(monkeypatch: pytest.MonkeyPatch, context: AgentRunContext) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_create_deep_agent(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "compiled-agent"

    monkeypatch.setattr("ultra_deepagents.agent.create_deep_agent", fake_create_deep_agent)
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
    )
    build_research_agent(settings, model=object(), backend=object(), context=context)
    return captured


@pytest.mark.parametrize(
    ("goal", "original_name", "typed_sensor"),
    [
        (
            "Validate clocks, calibration, uncertainty, and saturation in this "
            "acoustic-emission sensor series, then plot a bounded waveform envelope.",
            "ae-run.sensor.zarr",
            False,
        ),
        ("Inspect the selected Zarr.", "thermal-process.zarr", True),
    ],
)
def test_selected_sensor_runs_register_typed_tool_for_parent_and_code_runner(
    monkeypatch: pytest.MonkeyPatch,
    goal: str,
    original_name: str,
    typed_sensor: bool,
) -> None:
    captured = _capture_agent(
        monkeypatch,
        _context(goal=goal, original_name=original_name, typed_sensor=typed_sensor),
    )

    parent_tool_names = {str(getattr(tool, "name", "")) for tool in captured["tools"]}
    assert "inspect_selected_sensor_series" in parent_tool_names

    code_runner = next(item for item in captured["subagents"] if item["name"] == "code-runner")
    delegated_tool_names = {str(getattr(tool, "name", "")) for tool in code_runner.get("tools", [])}
    assert "inspect_selected_sensor_series" in delegated_tool_names
    assert "call inspect_selected_sensor_series before execute" in captured["system_prompt"]
    assert (
        "lineage as unbound unless the tool explicitly reports tree_verified"
        in captured["system_prompt"]
    )


@pytest.mark.parametrize(
    ("goal", "original_name"),
    [
        ("Summarize the selected paper.", "paper.pdf"),
        ("Inspect this CALPHAD database.", "alloy.tdb"),
    ],
)
def test_non_sensor_selected_resources_do_not_expand_sensor_tool_surface(
    monkeypatch: pytest.MonkeyPatch,
    goal: str,
    original_name: str,
) -> None:
    captured = _capture_agent(
        monkeypatch,
        _context(goal=goal, original_name=original_name),
    )

    tool_names = {str(getattr(tool, "name", "")) for tool in captured["tools"]}
    assert "inspect_selected_sensor_series" not in tool_names
    assert "call inspect_selected_sensor_series before execute" not in captured["system_prompt"]


def test_biology_ome_ngff_zarr_does_not_expand_sensor_tool_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_agent(
        monkeypatch,
        _context(
            goal="Inspect the selected Zarr and summarize its microscopy channels.",
            original_name="organoid.ome.zarr",
        ),
    )

    tool_names = {str(getattr(tool, "name", "")) for tool in captured["tools"]}
    assert "inspect_selected_sensor_series" not in tool_names
    assert "call inspect_selected_sensor_series before execute" not in captured["system_prompt"]


def test_cleanroom_never_registers_selected_sensor_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_agent(
        monkeypatch,
        _context(
            goal="Validate this acoustic-emission sensor waveform.",
            original_name="ae-run.sensor.zarr",
            evaluation_profile=MATERIALS_CLEANROOM_PROFILE,
        ),
    )

    tool_names = {str(getattr(tool, "name", "")) for tool in captured["tools"]}
    assert "inspect_selected_sensor_series" not in tool_names
