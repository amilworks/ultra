from __future__ import annotations

import asyncio
from pathlib import Path

from src.agno_backend.pro_mode import ProModeIntakeDecision
from src.agno_backend.runtime import AgnoChatRuntime, ProModeToolPlan
from src.config import Settings


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


def test_pro_mode_settings_default_to_global_llm_path() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="openai",
        llm_base_url="https://gateway.example/v1",
        llm_api_key="global-key",
        llm_model="gpt-oss-120b",
        openai_timeout=75,
    )

    assert settings.resolved_pro_mode_base_url == "https://gateway.example/v1"
    assert settings.resolved_pro_mode_api_key == "global-key"
    assert settings.resolved_pro_mode_model == "gpt-oss-120b"
    assert settings.resolved_pro_mode_timeout_seconds == 75


def test_pro_mode_settings_prefer_dedicated_gateway_values() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="openai",
        llm_base_url="https://gateway.example/v1",
        llm_api_key="global-key",
        llm_model="gpt-oss-120b",
        pro_mode_base_url="https://opus-gateway.example/v1",
        pro_mode_api_key="opus-key",
        pro_mode_model="claude-opus",
        pro_mode_timeout_seconds=180,
    )

    assert settings.resolved_pro_mode_base_url == "https://opus-gateway.example/v1"
    assert settings.resolved_pro_mode_api_key == "opus-key"
    assert settings.resolved_pro_mode_model == "claude-opus"
    assert settings.resolved_pro_mode_timeout_seconds == 180


def test_tool_workflow_default_execution_regime_stays_validated_tool(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)

    plan = ProModeToolPlan(
        category="research_program",
        selected_tool_names=["search_bisque_resources"],
        strict_validation=False,
        reason="Need broad evidence gathering.",
    )

    assert (
        runtime._execution_regime_for_decision(route="tool_workflow", tool_plan=plan)
        == "validated_tool"
    )


def test_report_like_turns_stay_on_reasoning_solver_by_default(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path, pro_mode_expert_council_enabled=True)
    prompt = "Write a report comparing the main approaches to segmentation in fluorescence microscopy."

    decision = ProModeIntakeDecision(
        route="deep_reasoning",
        execution_regime="focused_team",
        reason="Initial intake suggested a report workflow.",
    )
    stabilized = runtime._stabilize_pro_mode_intake_decision(
        decision=decision,
        messages=[{"role": "user", "content": prompt}],
        latest_user_text=prompt,
        uploaded_files=[],
        selected_tool_names=None,
        selection_context=None,
        prior_pro_mode_state=None,
    )

    assert stabilized.execution_regime == "reasoning_solver"
    assert stabilized.task_regime == "conceptual_high_uncertainty"


def test_proof_like_turns_stay_on_reasoning_solver_by_default(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path, pro_mode_expert_council_enabled=True)
    prompt = "Give a rigorous proof that the sequence converges under the stated assumptions."

    decision = ProModeIntakeDecision(
        route="deep_reasoning",
        execution_regime="proof_workflow",
        reason="Initial intake suggested a proof workflow.",
    )
    stabilized = runtime._stabilize_pro_mode_intake_decision(
        decision=decision,
        messages=[{"role": "user", "content": prompt}],
        latest_user_text=prompt,
        uploaded_files=[],
        selected_tool_names=None,
        selection_context=None,
        prior_pro_mode_state=None,
    )

    assert stabilized.execution_regime == "reasoning_solver"
    assert stabilized.task_regime == "rigorous_proof"


def test_benchmark_force_can_still_select_benchmark_only_regimes(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    base_decision = ProModeIntakeDecision(route="deep_reasoning", execution_regime="reasoning_solver")

    overridden = runtime._force_pro_mode_execution_regime(
        decision=base_decision,
        forced_execution_regime="expert_council",
        latest_user_text="Think collectively and compare competing explanations.",
        uploaded_files=[],
        selection_context=None,
        prior_pro_mode_state=None,
    )

    assert overridden.execution_regime == "expert_council"
    assert overridden.route == "deep_reasoning"


def test_pro_mode_helper_retries_once_on_gateway_failure(tmp_path: Path) -> None:
    runtime = _make_runtime(
        tmp_path,
        llm_provider="openai",
        llm_base_url="https://global-gateway.example/v1",
        llm_api_key="global-key",
        llm_model="gpt-oss-120b",
        pro_mode_base_url="https://opus-gateway.example/v1",
        pro_mode_api_key="opus-key",
        pro_mode_model="claude-opus",
        pro_mode_fallback_enabled=True,
    )

    class FakeAgent:
        def __init__(self, *, fail: bool) -> None:
            self.fail = fail

        async def arun(self, *_args, **_kwargs) -> str:
            if self.fail:
                raise RuntimeError("API connection error")
            return "fallback-ok"

    def build_agent(model_builder):
        return FakeAgent(fail=getattr(model_builder, "__name__", "") == "_build_pro_mode_model")

    result, metadata = asyncio.run(
        runtime._arun_with_optional_pro_mode_fallback(
            phase_name="unit_test_phase",
            prompt="Hello",
            build_agent=build_agent,
            conversation_id="conv-1",
            run_id="run-1",
            user_id="user-1",
            debug=False,
        )
    )

    assert result == "fallback-ok"
    assert metadata["fallback_used"] is True
    assert metadata["fallback_reason"] == "transport_or_availability_failure"
    assert metadata["active_model"] == runtime.model
