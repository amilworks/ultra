from __future__ import annotations

import asyncio
from pathlib import Path

import src.agno_backend.codeexec_reasoning as codeexec_reasoning_module
from src.agno_backend.codeexec_reasoning import build_codeexec_reasoning_agent
from src.agno_backend.pro_mode import ProModeIntakeDecision
from src.agno_backend.runtime import AgnoChatRuntime
from src.config import Settings
from src.tooling.domains.code_execution import CODEGEN_PYTHON_PLAN_TOOL, EXECUTE_PYTHON_JOB_TOOL


def _make_runtime(tmp_path: Path, **overrides: object) -> AgnoChatRuntime:
    settings = Settings(
        _env_file=None,
        environment="development",
        run_store_path=str(tmp_path / "runs.db"),
        artifact_root=str(tmp_path / "artifacts"),
        session_upload_root=str(tmp_path / "sessions"),
        science_data_root=str(tmp_path / "science"),
        upload_store_root=str(tmp_path / "uploads"),
        **overrides,
    )
    return AgnoChatRuntime(settings=settings)


def test_hard_codeexec_prompts_route_to_codeexec_reasoning_regime(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path, code_execution_enabled=True)
    prompt = (
        "Compare nonlinear model families on this dataset, bootstrap the coefficients, "
        "generate diagnostic plots, and explain why the best model is preferable."
    )

    stabilized = runtime._stabilize_pro_mode_intake_decision(
        decision=ProModeIntakeDecision(
            route="deep_reasoning",
            execution_regime="reasoning_solver",
            reason="Initial intake suggested a hard technical prompt.",
        ),
        messages=[{"role": "user", "content": prompt}],
        latest_user_text=prompt,
        uploaded_files=["/tmp/input.csv"],
        selected_tool_names=None,
        selection_context=None,
        prior_pro_mode_state=None,
    )

    assert stabilized.route == "deep_reasoning"
    assert stabilized.execution_regime == "reasoning_solver"
    assert stabilized.tool_plan_category == "code_execution"
    assert "codegen_python_plan" in stabilized.selected_tool_names
    assert "execute_python_job" in stabilized.selected_tool_names


def test_build_codeexec_reasoning_agent_enables_agno_reasoning(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    class FakeAgent:
        def __init__(self, **kwargs) -> None:
            captured.append(kwargs)

    def fake_model_builder(**kwargs):
        return {"builder": "fake", **kwargs}

    monkeypatch.setattr(codeexec_reasoning_module, "Agent", FakeAgent)

    build_codeexec_reasoning_agent(
        model_builder=fake_model_builder,
        tools=[CODEGEN_PYTHON_PLAN_TOOL, EXECUTE_PYTHON_JOB_TOOL],
        max_runtime_seconds=900,
    )

    assert len(captured) == 1
    kwargs = captured[0]
    assert kwargs["reasoning"] is True
    assert kwargs["reasoning_max_steps"] == 10
    assert kwargs["reasoning_min_steps"] == 2
    assert kwargs["tools"] == [CODEGEN_PYTHON_PLAN_TOOL, EXECUTE_PYTHON_JOB_TOOL]
    assert any(
        "executing the code tools is mandatory" in str(item or "")
        for item in list(kwargs["instructions"] or [])
    )


def test_codeexec_reasoning_solver_runs_required_tools_when_agent_skips_execution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = _make_runtime(tmp_path, code_execution_enabled=True)
    executed_tools: list[str] = []

    async def fake_arun_with_optional_pro_mode_fallback(**_kwargs):
        return "Random forest outperformed logistic regression with stable performance.", {
            "transport": "openai",
            "active_model": "gpt-oss-120b",
            "fallback_used": False,
        }

    async def fake_execute_tool_program_actions(**kwargs):
        actions = list(kwargs.get("actions") or [])
        executed_tools.extend(str(action.tool_name or "") for action in actions)
        return [
            {
                "tool": "codegen_python_plan",
                "status": "completed",
                "output_summary": {
                    "success": True,
                    "job_id": "job-123",
                    "summary": "Prepared a deterministic Python plan.",
                },
            },
            {
                "tool": "execute_python_job",
                "status": "completed",
                "output_summary": {
                    "success": True,
                    "status": "completed",
                    "artifacts": ["result.json", "outputs/feature_importance.png"],
                },
            },
        ]

    monkeypatch.setattr(
        runtime,
        "_arun_with_optional_pro_mode_fallback",
        fake_arun_with_optional_pro_mode_fallback,
    )
    monkeypatch.setattr(runtime, "_execute_tool_program_actions", fake_execute_tool_program_actions)

    result = asyncio.run(
        runtime._run_pro_mode_codeexec_reasoning_solver(
            latest_user_text=(
                "Write and run Python to compare random forest and logistic regression, "
                "save a figure, and report measured results."
            ),
            task_regime="programmatic_experiment",
            shared_context={},
            uploaded_files=["/tmp/input.csv"],
            selection_context=None,
            selected_tool_names=["codegen_python_plan", "execute_python_job"],
            conversation_id="conv-1",
            run_id="run-1",
            user_id="user-1",
            max_runtime_seconds=120,
            debug=False,
            event_callback=None,
        )
    )

    assert result.runtime_status == "completed"
    assert executed_tools == ["codegen_python_plan", "execute_python_job"]
    assert [item["tool"] for item in result.metadata["tool_invocations"]] == [
        "codegen_python_plan",
        "execute_python_job",
    ]
    assert (
        result.metadata["pro_mode"]["deterministic_required_tool_fallback"][
            "missing_required_tool_names"
        ]
        == ["codegen_python_plan", "execute_python_job"]
    )
