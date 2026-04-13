from __future__ import annotations

import asyncio
from pathlib import Path

from src.agno_backend.pro_mode import ProModeIntakeDecision, ProModeWorkflowResult
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


def test_pro_mode_settings_accept_custom_gateway_headers() -> None:
    settings = Settings(
        _env_file=None,
        pro_mode_transport="bedrock_published_api",
        pro_mode_api_key_header="X-API-Key",
        pro_mode_api_key_prefix="",
        pro_mode_default_headers={"anthropic-version": "bedrock-2023-05-31"},
        pro_mode_default_query={"profile": "opus"},
    )

    assert settings.pro_mode_transport == "bedrock_published_api"
    assert settings.pro_mode_api_key_header == "X-API-Key"
    assert settings.pro_mode_api_key_prefix == ""
    assert settings.pro_mode_default_headers == {"anthropic-version": "bedrock-2023-05-31"}
    assert settings.pro_mode_default_query == {"profile": "opus"}


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


def test_pro_mode_gateway_can_use_api_gateway_style_header_auth(tmp_path: Path) -> None:
    runtime = _make_runtime(
        tmp_path,
        llm_provider="openai",
        llm_base_url="https://global-gateway.example/v1",
        llm_api_key="global-key",
        llm_model="gpt-oss-120b",
        pro_mode_base_url="https://mpa10fscde.execute-api.us-east-1.amazonaws.com/api",
        pro_mode_api_key="gateway-key",
        pro_mode_api_key_header="X-API-Key",
        pro_mode_api_key_prefix="",
        pro_mode_default_headers={"anthropic-version": "bedrock-2023-05-31"},
        pro_mode_default_query={"profile": "opus"},
        pro_mode_model="claude-opus-4-5",
    )

    model = runtime._build_pro_mode_model()

    assert model.base_url == "https://mpa10fscde.execute-api.us-east-1.amazonaws.com/api"
    assert model.api_key == "EMPTY"
    assert model.default_headers["X-API-Key"] == "gateway-key"
    assert model.default_headers["anthropic-version"] == "bedrock-2023-05-31"
    assert model.default_query == {"profile": "opus"}
    assert model.id == "claude-opus-4-5"


def test_published_api_transport_extracts_text_blocks() -> None:
    payload = {
        "conversationId": "conv-1",
        "message": {
            "content": [
                {"contentType": "reasoning", "text": "hidden", "signature": "sig", "redactedContent": ""},
                {"contentType": "text", "body": "First paragraph."},
                {"contentType": "text", "body": "Second paragraph."},
            ]
        },
    }

    assert (
        AgnoChatRuntime._published_api_extract_text(payload)
        == "First paragraph.\nSecond paragraph."
    )


def test_published_api_text_phase_uses_dedicated_transport(tmp_path: Path) -> None:
    runtime = _make_runtime(
        tmp_path,
        llm_provider="openai",
        llm_base_url="https://global-gateway.example/v1",
        llm_api_key="global-key",
        llm_model="gpt-oss-120b",
        pro_mode_transport="bedrock_published_api",
        pro_mode_base_url="https://published.example/api",
        pro_mode_api_key="published-key",
        pro_mode_api_key_header="X-API-Key",
        pro_mode_api_key_prefix="",
        pro_mode_model="claude-v4.5-opus",
    )

    async def fake_published(**_kwargs):
        return "published-answer", {"conversation_id": "conv-1", "message_id": "msg-1"}

    runtime._run_published_pro_mode_prompt = fake_published  # type: ignore[method-assign]

    class FakeAgent:
        async def arun(self, *_args, **_kwargs) -> str:
            return "should-not-be-used"

    def build_agent(_model_builder):
        return FakeAgent()

    result, metadata = asyncio.run(
        runtime._arun_text_phase_with_optional_pro_mode_transport(
            phase_name="reasoning_solver",
            prompt="Hello",
            build_agent=build_agent,
            conversation_id="conv-1",
            run_id="run-1",
            user_id="user-1",
            debug=False,
            reasoning_mode="deep",
            reasoning_effort_override="high",
            max_runtime_seconds=30,
        )
    )

    assert result == "published-answer"
    assert metadata["transport"] == "bedrock_published_api"
    assert metadata["active_model"] == "claude-v4.5-opus"
    assert metadata["published_api"]["conversation_id"] == "conv-1"


def test_published_api_keeps_structured_agent_phases_on_tool_capable_model(tmp_path: Path) -> None:
    runtime = _make_runtime(
        tmp_path,
        llm_provider="openai",
        llm_base_url="https://global-gateway.example/v1",
        llm_api_key="global-key",
        llm_model="gpt-oss-120b",
        pro_mode_transport="bedrock_published_api",
        pro_mode_base_url="https://published.example/api",
        pro_mode_api_key="published-key",
        pro_mode_api_key_header="X-API-Key",
        pro_mode_api_key_prefix="",
        pro_mode_model="claude-v4.5-opus",
    )

    model = runtime._build_pro_mode_agent_model(reasoning_mode="deep")

    assert model.id == "gpt-oss-120b"
    assert model.base_url == "https://global-gateway.example/v1"


def test_conceptual_tool_discussion_does_not_force_tool_workflow() -> None:
    assert (
        AgnoChatRuntime._requires_tool_workflow(
            user_text=(
                "Compare the assumptions behind Otsu thresholding and watershed segmentation "
                "for fluorescence microscopy, and explain when each breaks down."
            ),
            uploaded_files=[],
            selected_tool_names=[],
            selection_context=None,
            inferred_tool_names=["segment_image_sam3", "quantify_segmentation_masks"],
        )
        is False
    )


def test_operational_bisque_request_still_uses_tool_workflow() -> None:
    assert (
        AgnoChatRuntime._requires_tool_workflow(
            user_text="Search BisQue for datasets and summarize what is available.",
            uploaded_files=[],
            selected_tool_names=[],
            selection_context=None,
            inferred_tool_names=["search_bisque_resources"],
        )
        is True
    )


def test_pro_mode_direct_response_path_uses_dedicated_model_branch(tmp_path: Path) -> None:
    runtime = _make_runtime(
        tmp_path,
        llm_provider="openai",
        llm_base_url="https://global-gateway.example/v1",
        llm_api_key="global-key",
        llm_model="gpt-oss-120b",
        pro_mode_transport="bedrock_published_api",
        pro_mode_base_url="https://published.example/api",
        pro_mode_api_key="published-key",
        pro_mode_api_key_header="X-API-Key",
        pro_mode_api_key_prefix="",
        pro_mode_model="claude-v4.5-opus",
    )

    async def fake_intake(**_kwargs):
        return ProModeIntakeDecision(
            route="direct_response",
            execution_regime="fast_dialogue",
            reason="Simple conceptual answer.",
            direct_response="Draft answer from intake.",
        )

    async def fake_fast_dialogue(**_kwargs):
        return ProModeWorkflowResult(
            response_text="Final answer from dedicated Pro Mode model.",
            metadata={
                "pro_mode": {
                    "execution_path": "direct_response",
                    "runtime_status": "completed",
                    "model_route": {
                        "active_model": "claude-v4.5-opus",
                        "transport": "bedrock_published_api",
                        "fallback_used": False,
                    },
                }
            },
            runtime_status="completed",
            runtime_error=None,
        )

    runtime.pro_mode.intake = fake_intake  # type: ignore[method-assign]
    runtime._run_pro_mode_fast_dialogue = fake_fast_dialogue  # type: ignore[method-assign]
    runtime._persist_analysis_state = lambda **_kwargs: None  # type: ignore[method-assign]

    async def _collect() -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        async for event in runtime.stream(
            messages=[
                {
                    "role": "user",
                    "content": "Explain the tradeoffs between Otsu thresholding and watershed segmentation.",
                }
            ],
            uploaded_files=[],
            conversation_id="conv-1",
            max_tool_calls=8,
            max_runtime_seconds=120,
            workflow_hint={"id": "pro_mode", "source": "slash_menu"},
            reasoning_mode="deep",
            user_id="user-1",
            run_id="run-1",
            debug=True,
        ):
            events.append(event)
        return events

    events = asyncio.run(_collect())
    done_event = next(event for event in events if event.get("event") == "done")
    payload = dict(done_event.get("data") or {})
    metadata = dict(payload.get("metadata") or {})

    assert payload["response_text"] == "Final answer from dedicated Pro Mode model."
    assert payload["model"] == "claude-v4.5-opus"
    assert metadata["pro_mode"]["execution_path"] == "direct_response"
    assert metadata["pro_mode"]["model_route"]["transport"] == "bedrock_published_api"
