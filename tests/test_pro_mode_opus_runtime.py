from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import types

import src.agno_backend.runtime as runtime_module
import src.agno_backend.pro_mode as pro_mode_module
from src.agno_backend.pro_mode import ProModeIntakeDecision, ProModeVerifierReport, ProModeWorkflowResult
from src.agno_backend.runtime import AgnoChatRuntime, ProModeToolPlan, ToolProgramSynthesis
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


def test_pro_mode_settings_accept_native_bedrock_values() -> None:
    settings = Settings(
        _env_file=None,
        pro_mode_transport="aws_bedrock_claude",
        pro_mode_model="global.anthropic.claude-opus-4-1-20250805-v1:0",
        pro_mode_aws_region="us-east-1",
        pro_mode_aws_profile="ucsb-sandbox",
        pro_mode_aws_sso_auth=True,
    )

    assert settings.pro_mode_transport == "aws_bedrock_claude"
    assert settings.resolved_pro_mode_model == "global.anthropic.claude-opus-4-1-20250805-v1:0"
    assert settings.resolved_pro_mode_aws_region == "us-east-1"
    assert settings.resolved_pro_mode_aws_profile == "ucsb-sandbox"
    assert settings.pro_mode_aws_sso_auth is True


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
        pro_mode_base_url="https://gateway.example/api",
        pro_mode_api_key="gateway-key",
        pro_mode_api_key_header="X-API-Key",
        pro_mode_api_key_prefix="",
        pro_mode_default_headers={"anthropic-version": "bedrock-2023-05-31"},
        pro_mode_default_query={"profile": "opus"},
        pro_mode_model="claude-opus-4-5",
    )

    model = runtime._build_pro_mode_model()

    assert model.base_url == "https://gateway.example/api"
    assert model.api_key == "EMPTY"
    assert model.default_headers["X-API-Key"] == "gateway-key"
    assert model.default_headers["anthropic-version"] == "bedrock-2023-05-31"
    assert model.default_query == {"profile": "opus"}
    assert model.id == "claude-opus-4-5"


def test_native_bedrock_transport_builds_claude_model(tmp_path: Path, monkeypatch) -> None:
    fake_module = types.ModuleType("agno.models.aws")

    class FakeClaude:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.id = kwargs["id"]

    fake_module.Claude = FakeClaude
    monkeypatch.setitem(sys.modules, "agno.models.aws", fake_module)

    runtime = _make_runtime(
        tmp_path,
        pro_mode_transport="aws_bedrock_claude",
        pro_mode_model="global.anthropic.claude-opus-4-1-20250805-v1:0",
        pro_mode_aws_region="us-east-1",
        pro_mode_aws_access_key_id="AKIA_TEST",
        pro_mode_aws_secret_access_key="secret-test",
    )

    model = runtime._build_pro_mode_model(max_runtime_seconds=45)

    assert isinstance(model, FakeClaude)
    assert model.kwargs["id"] == "global.anthropic.claude-opus-4-1-20250805-v1:0"
    assert model.kwargs["aws_region"] == "us-east-1"
    assert model.kwargs["aws_access_key"] == "AKIA_TEST"
    assert model.kwargs["aws_secret_key"] == "secret-test"
    assert model.kwargs["timeout"] >= 60


def test_native_bedrock_transport_uses_boto3_session_for_sso(tmp_path: Path, monkeypatch) -> None:
    fake_aws_module = types.ModuleType("agno.models.aws")

    class FakeClaude:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    fake_aws_module.Claude = FakeClaude
    monkeypatch.setitem(sys.modules, "agno.models.aws", fake_aws_module)

    fake_boto3_session_module = types.ModuleType("boto3.session")

    class FakeSession:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    fake_boto3_session_module.Session = FakeSession
    monkeypatch.setitem(sys.modules, "boto3.session", fake_boto3_session_module)

    runtime = _make_runtime(
        tmp_path,
        pro_mode_transport="aws_bedrock_claude",
        pro_mode_model="global.anthropic.claude-opus-4-1-20250805-v1:0",
        pro_mode_aws_region="us-east-1",
        pro_mode_aws_profile="ucsb-sandbox",
        pro_mode_aws_sso_auth=True,
    )

    model = runtime._build_pro_mode_model(max_runtime_seconds=45)

    assert isinstance(model, FakeClaude)
    assert isinstance(model.kwargs["session"], FakeSession)
    assert model.kwargs["session"].kwargs == {
        "profile_name": "ucsb-sandbox",
        "region_name": "us-east-1",
    }
    assert model.kwargs["aws_region"] == "us-east-1"


def test_native_bedrock_transport_classifies_missing_credentials_for_fallback(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path, pro_mode_transport="aws_bedrock_claude")

    assert (
        runtime._classify_pro_mode_failure("AWS credentials not found for Bedrock Claude request")
        == "transport_or_availability_failure"
    )


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


def test_structured_phase_uses_hybrid_reasoning_for_deep_phases(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    class FakeAgent:
        def __init__(self, **kwargs) -> None:
            captured.append(kwargs)
            self.output_schema = kwargs["output_schema"]

        async def arun(self, *_args, **_kwargs):
            return self.output_schema()

    def fake_builder(**kwargs):
        return {"builder": "pro_mode", **kwargs}

    monkeypatch.setattr(pro_mode_module, "Agent", FakeAgent)

    runner = pro_mode_module.ProModeWorkflowRunner(model_builder=fake_builder)
    result = asyncio.run(
        runner._run_structured_phase(
            role="Verifier",
            phase_name="verifier",
            schema=ProModeVerifierReport,
            prompt="Check the answer.",
            fallback=ProModeVerifierReport(),
            conversation_id="conv-1",
            run_id="run-1",
            user_id="user-1",
            reasoning_mode="deep",
            max_runtime_seconds=120,
            debug=False,
        )
    )

    assert isinstance(result, ProModeVerifierReport)
    assert len(captured) == 1
    kwargs = captured[0]
    assert kwargs["reasoning"] is True
    assert kwargs["reasoning_min_steps"] == 3
    assert kwargs["reasoning_max_steps"] == 8
    assert kwargs["model"] == {
        "builder": "pro_mode",
        "reasoning_mode": "deep",
        "reasoning_effort_override": "high",
        "max_runtime_seconds": 120,
    }
    assert kwargs["reasoning_model"] == {
        "builder": "pro_mode",
        "reasoning_mode": "deep",
        "reasoning_effort_override": "high",
        "max_runtime_seconds": 120,
    }


def test_structured_phase_keeps_intake_on_lightweight_path(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    class FakeAgent:
        def __init__(self, **kwargs) -> None:
            captured.append(kwargs)
            self.output_schema = kwargs["output_schema"]

        async def arun(self, *_args, **_kwargs):
            return self.output_schema()

    def fake_builder(**kwargs):
        return {"builder": "pro_mode", **kwargs}

    monkeypatch.setattr(pro_mode_module, "Agent", FakeAgent)

    runner = pro_mode_module.ProModeWorkflowRunner(model_builder=fake_builder)
    result = asyncio.run(
        runner._run_structured_phase(
            role="intake",
            phase_name="intake",
            schema=ProModeIntakeDecision,
            prompt="Route this turn.",
            fallback=ProModeIntakeDecision(),
            conversation_id="conv-1",
            run_id="run-1",
            user_id="user-1",
            reasoning_mode="deep",
            max_runtime_seconds=60,
            debug=False,
        )
    )

    assert isinstance(result, ProModeIntakeDecision)
    assert len(captured) == 1
    kwargs = captured[0]
    assert "reasoning_model" not in kwargs
    assert "reasoning" not in kwargs
    assert kwargs["model"] == {
        "builder": "pro_mode",
        "reasoning_mode": "fast",
        "reasoning_effort_override": "low",
        "max_runtime_seconds": 60,
    }


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


def test_explicit_bisque_upload_turn_prefers_upload_tool_over_research_bundle() -> None:
    plan = AgnoChatRuntime._build_pro_mode_tool_plan(
        user_text="Upload this image to BisQue.",
        uploaded_files=["/tmp/test-image.png"],
        selected_tool_names=["upload_to_bisque", "search_bisque_resources"],
        selection_context=None,
        inferred_tool_names=["bioio_load_image", "segment_image_sam2", "search_bisque_resources"],
        prior_pro_mode_state=None,
    )

    assert plan is not None
    assert plan.category == "bisque_management"
    assert plan.selected_tool_names == ["upload_to_bisque"]
    assert plan.strict_validation is True


def test_explicit_bisque_upload_turn_does_not_trigger_iterative_research_program() -> None:
    assert (
        AgnoChatRuntime._requires_iterative_research_program(
            user_text="Upload this image to BisQue.",
            uploaded_files=["/tmp/test-image.png"],
            selection_context=None,
            prior_pro_mode_state=None,
        )
        is False
    )


def test_megaseg_request_prefers_megaseg_tool_plan(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)

    plan = runtime._build_pro_mode_tool_plan(
        user_text="Use only MegaSeg to segment this microscopy image and quantify the mask.",
        uploaded_files=["/tmp/test-image.ome.tiff"],
        selected_tool_names=[],
        selection_context=None,
        inferred_tool_names=["segment_image_sam2", "quantify_segmentation_masks"],
        prior_pro_mode_state=None,
    )

    assert plan is not None
    assert plan.category == "segmentation"
    assert plan.selected_tool_names == ["segment_image_megaseg", "quantify_segmentation_masks"]
    assert plan.strict_validation is True


def test_megaseg_request_requires_megaseg_in_strict_tool_workflow(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)

    required, strict = runtime._tool_workflow_required_tools(
        user_text="Run MegaSeg on this image and quantify the result.",
        uploaded_files=["/tmp/test-image.ome.tiff"],
        selection_context=None,
        selected_tool_names=["segment_image_megaseg", "quantify_segmentation_masks", "segment_image_sam2"],
    )

    assert strict is True
    assert required == ["segment_image_megaseg", "quantify_segmentation_masks"]


def test_strict_tool_workflow_executes_required_upload_tool_when_model_skips_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = _make_runtime(tmp_path)

    async def fake_stream(**_kwargs):
        yield {
            "event": "done",
            "data": {
                "response_text": "Please attach the file here or tell me where it lives.",
                "metadata": {"tool_invocations": []},
            },
        }

    def fake_execute_tool_call(tool_name, args, **kwargs):
        assert tool_name == "upload_to_bisque"
        assert args == {}
        assert kwargs["uploaded_files"] == ["/tmp/test-image.png"]
        return {
            "success": True,
            "results": [
                {
                    "file": "test-image.png",
                    "resource_uri": "https://ultra.example.com/data_service/01-upload",
                    "client_view_url": "https://ultra.example.com/client_service/view?resource=https://ultra.example.com/data_service/01-upload",
                }
            ],
        }

    runtime.stream = fake_stream  # type: ignore[method-assign]
    monkeypatch.setattr(runtime_module, "execute_tool_call", fake_execute_tool_call)

    result = asyncio.run(
        runtime._run_pro_mode_tool_workflow(
            messages=[{"role": "user", "content": "Upload this to BisQue."}],
            latest_user_text="Upload this to BisQue.",
            uploaded_files=["/tmp/test-image.png"],
            max_tool_calls=8,
            max_runtime_seconds=120,
            reasoning_mode="deep",
            conversation_id="conv-1",
            run_id="run-1",
            user_id="user-1",
            event_callback=None,
            selected_tool_names=["upload_to_bisque"],
            tool_plan_category="bisque_management",
            strict_tool_validation=True,
            selection_context=None,
            knowledge_context=None,
            shared_context={},
            conversation_state_seed=None,
            debug=False,
        )
    )

    assert result["runtime_status"] == "completed"
    assert [item["tool"] for item in result["tool_invocations"]] == ["upload_to_bisque"]
    assert result["tool_invocations"][0]["status"] == "completed"
    assert result["response_text"] != "Please attach the file here or tell me where it lives."


def test_strict_tool_workflow_executes_required_search_tool_when_model_skips_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = _make_runtime(tmp_path)

    async def fake_stream(**_kwargs):
        yield {
            "event": "done",
            "data": {
                "response_text": "I can help search BisQue if you want.",
                "metadata": {"tool_invocations": []},
            },
        }

    def fake_execute_tool_call(tool_name, args, **_kwargs):
        assert tool_name == "search_bisque_resources"
        assert args == {}
        return {
            "success": True,
            "results": [
                {
                    "name": "Prairie_Dog_Active_Learning",
                    "resource_type": "dataset",
                    "client_view_url": "https://ultra.example.com/client_service/view?resource=https://ultra.example.com/data_service/00-dataset",
                }
            ],
            "count": 1,
        }

    runtime.stream = fake_stream  # type: ignore[method-assign]
    monkeypatch.setattr(runtime_module, "execute_tool_call", fake_execute_tool_call)

    result = asyncio.run(
        runtime._run_pro_mode_tool_workflow(
            messages=[{"role": "user", "content": "Search BisQue for datasets."}],
            latest_user_text="Search BisQue for datasets.",
            uploaded_files=[],
            max_tool_calls=8,
            max_runtime_seconds=120,
            reasoning_mode="deep",
            conversation_id="conv-1",
            run_id="run-1",
            user_id="user-1",
            event_callback=None,
            selected_tool_names=["search_bisque_resources"],
            tool_plan_category="catalog_lookup",
            strict_tool_validation=True,
            selection_context=None,
            knowledge_context=None,
            shared_context={},
            conversation_state_seed=None,
            debug=False,
        )
    )

    assert result["runtime_status"] == "completed"
    assert [item["tool"] for item in result["tool_invocations"]] == ["search_bisque_resources"]
    assert result["tool_invocations"][0]["status"] == "completed"
    assert result["response_text"] != "I can help search BisQue if you want."


def test_strict_tool_workflow_executes_required_code_tools_when_model_skips_them(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = _make_runtime(tmp_path)
    executed_tools: list[str] = []

    async def fake_stream(**_kwargs):
        yield {
            "event": "done",
            "data": {
                "response_text": "I can write some Python if you want.",
                "metadata": {"tool_invocations": []},
            },
        }

    def fake_execute_tool_call(tool_name, args, **_kwargs):
        executed_tools.append(tool_name)
        assert args == {}
        if tool_name == "codegen_python_plan":
            return {
                "success": True,
                "job_id": "job-123",
                "summary": "Prepared a deterministic Python analysis job.",
            }
        assert tool_name == "execute_python_job"
        return {
            "success": True,
            "status": "completed",
            "job_id": "job-123",
            "summary": "Executed the prepared Python analysis job successfully.",
            "artifacts": ["plots/result.png"],
        }

    runtime.stream = fake_stream  # type: ignore[method-assign]
    monkeypatch.setattr(runtime_module, "execute_tool_call", fake_execute_tool_call)

    result = asyncio.run(
        runtime._run_pro_mode_tool_workflow(
            messages=[{"role": "user", "content": "Write Python to analyze this image and run it."}],
            latest_user_text="Write Python to analyze this image and run it.",
            uploaded_files=["/tmp/test-image.png"],
            max_tool_calls=8,
            max_runtime_seconds=120,
            reasoning_mode="deep",
            conversation_id="conv-1",
            run_id="run-1",
            user_id="user-1",
            event_callback=None,
            selected_tool_names=["codegen_python_plan", "execute_python_job"],
            tool_plan_category="programmatic_experiment",
            strict_tool_validation=True,
            selection_context=None,
            knowledge_context=None,
            shared_context={},
            conversation_state_seed=None,
            debug=False,
        )
    )

    assert result["runtime_status"] == "completed"
    assert executed_tools == ["codegen_python_plan", "execute_python_job"]
    assert [item["tool"] for item in result["tool_invocations"]] == [
        "codegen_python_plan",
        "execute_python_job",
    ]
    assert result["response_text"] != "I can write some Python if you want."


def test_tool_program_phase_records_native_bedrock_model_route_in_compression_stats(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = _make_runtime(
        tmp_path,
        pro_mode_transport="aws_bedrock_claude",
        pro_mode_model="anthropic.claude-opus-4-5-20251101-v1:0",
        pro_mode_aws_region="us-east-1",
        pro_mode_aws_profile="ucsb-sandbox",
        pro_mode_aws_sso_auth=True,
    )
    session_state: dict[str, object] = {}

    async def fake_arun_with_optional_pro_mode_fallback(**_kwargs):
        return types.SimpleNamespace(
            final_output=ToolProgramSynthesis(
                response_text="Tool synthesis complete.",
                evidence_basis=["measured evidence"],
                unresolved_points=[],
                confidence="high",
            )
        ), {
            "transport": "aws_bedrock_claude",
            "active_model": "anthropic.claude-opus-4-5-20251101-v1:0",
            "fallback_used": False,
        }

    monkeypatch.setattr(
        runtime,
        "_arun_with_optional_pro_mode_fallback",
        fake_arun_with_optional_pro_mode_fallback,
    )

    result = asyncio.run(
        runtime._run_tool_program_phase(
            phase_name="unit_test_phase",
            schema=ToolProgramSynthesis,
            prompt="Return the final synthesis.",
            fallback=ToolProgramSynthesis(
                response_text="fallback",
                evidence_basis=[],
                unresolved_points=[],
                confidence="low",
            ),
            session_state=session_state,
            conversation_id="conv-1",
            run_id="run-1",
            user_id="user-1",
            reasoning_mode="deep",
            max_runtime_seconds=60,
            debug=False,
        )
    )

    assert result.response_text == "Tool synthesis complete."
    assert session_state["compression_stats"]["unit_test_phase"]["model_route"]["transport"] == (
        "aws_bedrock_claude"
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


def test_pro_mode_direct_response_failure_does_not_claim_completed_opus_turn(tmp_path: Path) -> None:
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
        pro_mode_fallback_enabled=False,
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
            response_text="Draft answer from intake.",
            metadata={
                "pro_mode": {
                    "execution_path": "direct_response",
                    "runtime_status": "failed",
                    "model_route": {
                        "active_model": "gpt-oss-120b",
                        "transport": "bedrock_published_api",
                        "fallback_used": False,
                        "fallback_reason": "published_api_failure",
                    },
                }
            },
            runtime_status="failed",
            runtime_error="403 Forbidden",
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

    assert payload["response_text"] == "Draft answer from intake."
    assert payload["model"] == "gpt-oss-120b"
    assert metadata["pro_mode"]["runtime_status"] == "failed"
    assert metadata["pro_mode"]["summary"] == (
        "Returned the intake draft because the dedicated Pro Mode reasoning model path did not complete."
    )
    assert metadata["pro_mode"]["verifier"]["passed"] is False
    assert metadata["pro_mode"]["model_route"]["active_model"] == "gpt-oss-120b"


def test_tool_workflow_metadata_exposes_structured_phase_model_routes(tmp_path: Path) -> None:
    runtime = _make_runtime(
        tmp_path,
        pro_mode_transport="aws_bedrock_claude",
        pro_mode_model="anthropic.claude-opus-4-5-20251101-v1:0",
        pro_mode_aws_region="us-east-1",
        pro_mode_aws_profile="ucsb-sandbox",
        pro_mode_aws_sso_auth=True,
    )

    async def fake_intake(**_kwargs):
        return ProModeIntakeDecision(
            route="tool_workflow",
            execution_regime="validated_tool",
            reason="Need deterministic tool use.",
            selected_tool_names=["execute_python_job"],
        )

    async def fake_tool_workflow(**_kwargs):
        return {
            "response_text": "Measured output is ready.",
            "tool_invocations": [
                {"tool": "execute_python_job", "status": "completed", "output_summary": {"success": True}}
            ],
            "metadata": {
                "research_program": {
                    "iterations": 1,
                    "evidence_summaries": ["Executed the Python analysis job."],
                    "handles": {"analysis_table_paths": ["/tmp/result.json"]},
                    "requirements": {"required_families": ["code"]},
                    "executed_families": ["code"],
                    "compression_stats": {
                        "research_program_synthesis": {
                            "model_route": {
                                "transport": "aws_bedrock_claude",
                                "active_model": "anthropic.claude-opus-4-5-20251101-v1:0",
                                "fallback_used": False,
                            }
                        }
                    },
                }
            },
            "model": "gpt-oss-120b",
            "selected_domains": ["core"],
            "runtime_status": "completed",
            "runtime_error": None,
            "selected_tool_names": ["execute_python_job"],
            "attempted_tool_sets": [["execute_python_job"]],
        }

    async def fake_final_writer(**_kwargs):
        return "Measured output is ready.", {
            "model_route": {
                "transport": "aws_bedrock_claude",
                "active_model": "anthropic.claude-opus-4-5-20251101-v1:0",
                "fallback_used": False,
            }
        }

    runtime.pro_mode.intake = fake_intake  # type: ignore[method-assign]
    runtime._run_pro_mode_tool_workflow = fake_tool_workflow  # type: ignore[method-assign]
    runtime._run_pro_mode_final_writer = fake_final_writer  # type: ignore[method-assign]
    runtime._persist_analysis_state = lambda **_kwargs: None  # type: ignore[method-assign]

    async def _collect() -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        async for event in runtime.stream(
            messages=[
                {
                    "role": "user",
                    "content": "Run Python analysis on this file and summarize the result.",
                }
            ],
            uploaded_files=["/tmp/test-image.png"],
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

    assert payload["model"] == "gpt-oss-120b"
    assert metadata["pro_mode"]["tool_runtime_model"] == "gpt-oss-120b"
    assert metadata["pro_mode"]["model_routes"]["research_program_synthesis"]["transport"] == (
        "aws_bedrock_claude"
    )
    assert metadata["pro_mode"]["model_routes"]["pro_mode_final_writer"]["transport"] == (
        "aws_bedrock_claude"
    )
