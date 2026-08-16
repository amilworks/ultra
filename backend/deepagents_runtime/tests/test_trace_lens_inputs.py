from __future__ import annotations

import asyncio
import copy
import json
from uuid import UUID, uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from ultra_deepagents.agent import build_research_agent
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.runner import RunEventSequencer, _stream_agent_attempt
from ultra_deepagents.trace_lens_inputs import (
    TRACE_LENS_MAX_PENDING_CALLS,
    TRACE_LENS_MAX_SERIALIZED_BYTES,
    TraceLensInputCallback,
    attach_trace_lens_metadata,
    build_agent_configuration,
)

CALLBACK_RUN_ID = UUID("11111111-2222-3333-4444-555555555555")


def _finish_event(checkpoint_ns: str) -> dict[str, object]:
    return {
        "type": "event",
        "method": "messages",
        "params": {
            "namespace": [],
            "data": [
                {
                    "event": "message-finish",
                    "usage": {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10},
                    "metadata": {},
                },
                {
                    "langgraph_node": "model",
                    "langgraph_checkpoint_ns": checkpoint_ns,
                    "ls_model_name": "deepseek_v4",
                },
            ],
        },
    }


def _context() -> AgentRunContext:
    return AgentRunContext(
        assistant_id="assistant-trace-lens",
        org_id="org-trace-lens",
        run_id="run-trace-lens",
        thread_id="thread-trace-lens",
        user_id="user-trace-lens",
        project_id="project-trace-lens",
        goal="Test structural tracing.",
    )


def _invoke(
    callback: TraceLensInputCallback,
    checkpoint_ns: str,
    *,
    run_id: UUID = CALLBACK_RUN_ID,
    messages: object | None = None,
    invocation_params: object | None = None,
    include_invocation_params: bool = True,
    options: object | None = None,
) -> None:
    kwargs: dict[str, object] = {}
    if include_invocation_params:
        kwargs["invocation_params"] = invocation_params if invocation_params is not None else {}
    if options is not None:
        kwargs["options"] = options
    asyncio.run(
        callback.on_chat_model_start(
            {},
            messages if messages is not None else [[HumanMessage(content="SECRET_PROMPT")]],
            run_id=run_id,
            metadata={
                "langgraph_node": "model",
                "langgraph_checkpoint_ns": checkpoint_ns,
            },
            **kwargs,
        )
    )


def test_runtime_setting_is_off_by_default_and_env_gated(monkeypatch) -> None:
    monkeypatch.delenv("ULTRA_DEEPAGENTS_TRACE_LENS_INPUTS", raising=False)
    assert RuntimeSettings.from_env().trace_lens_inputs_enabled is False

    monkeypatch.setenv("ULTRA_DEEPAGENTS_TRACE_LENS_INPUTS", "1")
    assert RuntimeSettings.from_env().trace_lens_inputs_enabled is True


def test_real_langchain_messages_keep_only_structural_counts_and_real_markers() -> None:
    callback = TraceLensInputCallback()
    messages = [
        [
            SystemMessage(content="SECRET_SYSTEM_CANARY"),
            HumanMessage(content=[{"type": "text", "text": "SECRET_USER_CANARY"}]),
            ToolMessage(content="SECRET_TOOL_CANARY", tool_call_id="tool-call-1"),
            AIMessage(
                content="SECRET_SUMMARY_CANARY",
                additional_kwargs={
                    "lc_source": "summarization",
                    "reasoning_content": "SECRET_REASONING_CANARY",
                    "summary": False,
                    "offloaded": True,
                },
            ),
        ]
    ]
    _invoke(
        callback,
        "model:exact",
        messages=messages,
        invocation_params={
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "execute",
                        "description": "SECRET_DESCRIPTION_CANARY",
                        "parameters": {"default": "SECRET_DEFAULT_CANARY"},
                    },
                },
                {
                    "type": "function",
                    "function": {"name": "response_format_SECRET_GENERATED_SUFFIX"},
                },
                {"type": "web_search", "filters": {"url": "https://secret.example"}},
                {"type": "custom", "headers": {"Authorization": "SECRET_TOKEN"}},
            ],
            "tool_choice": {"type": "function", "function": {"name": "execute"}},
            "api_key": "SECRET_API_KEY_CANARY",
        },
        options={"ls_structured_output_format": {"schema": "SECRET_SCHEMA_CANARY"}},
    )
    record = callback.take_exact_match(
        _finish_event("model:exact"),
        logical_call_id="run-trace-lens:model:stable-usage-id",
    )

    assert record is not None
    assert record["schema_version"] == 2
    assert record["capture_scope"] == "logical_model_invocation"
    assert record["message_count"] == 4
    assert record["role_counts"] == {
        "system": 1,
        "user": 1,
        "assistant": 1,
        "tool": 1,
        "unknown": 0,
    }
    assert record["summary_message_count"] == 1
    assert record["reasoning_continuity_message_count"] == 1
    assert "offload_message_count" not in record
    tools = record["provider_bound_tools"]
    assert tools["availability"] == "observed"
    assert tools["tool_count"] == 4
    assert tools["function_tool_count"] == 2
    assert tools["function_tool_names"] == ["execute", "structured_response"]
    assert tools["non_function_classifications"] == {
        "custom": 1,
        "web_search": 1,
        "file_search": 0,
        "code_interpreter": 0,
        "other": 0,
    }
    assert tools["structured_response_tool_count"] == 1
    assert tools["tool_choice_mode"] == "specific_function"
    encoded = json.dumps(record)
    for canary in (
        "SECRET_",
        "secret.example",
        "description",
        "parameters",
        "headers",
        "api_key",
        "SECRET_SCHEMA_CANARY",
        "offload",
    ):
        assert canary not in encoded


def test_provider_bound_tools_distinguishes_unavailable_malformed_and_observed_empty() -> None:
    cases = [
        ({"include_invocation_params": False}, "unavailable"),
        ({"invocation_params": "not-a-mapping"}, "malformed"),
        ({"invocation_params": {"tools": "not-a-list"}}, "malformed"),
        ({"invocation_params": {"tools": [{"type": {"unsafe": "SECRET"}}]}}, "malformed"),
        ({"invocation_params": {"tools": []}}, "observed"),
    ]
    for index, (kwargs, availability) in enumerate(cases):
        callback = TraceLensInputCallback()
        checkpoint_ns = f"model:availability-{index}"
        _invoke(callback, checkpoint_ns, **kwargs)
        record = callback.take_exact_match(
            _finish_event(checkpoint_ns), logical_call_id=f"run:model:{index}"
        )
        assert record is not None
        tools = record["provider_bound_tools"]
        assert tools["availability"] == availability
        if availability == "observed":
            assert tools["tool_count"] == 0


def test_synthetic_callback_options_do_not_invent_structured_response_tools() -> None:
    callback = TraceLensInputCallback()
    _invoke(
        callback,
        "model:no-synthetic-structured-output",
        invocation_params={"tools": []},
        options={"ls_structured_output_format": {"schema": "SECRET_SCHEMA_CANARY"}},
    )
    record = callback.take_exact_match(
        _finish_event("model:no-synthetic-structured-output"),
        logical_call_id="run:model:no-synthetic-structured-output",
    )
    assert record is not None
    assert record["provider_bound_tools"]["structured_response_tool_count"] == 0
    assert "SECRET" not in json.dumps(record)


def test_callback_matches_concurrent_calls_in_reverse_completion_order() -> None:
    callback = TraceLensInputCallback()
    first_id = UUID("00000000-0000-0000-0000-000000000001")
    second_id = UUID("00000000-0000-0000-0000-000000000002")
    _invoke(callback, "model:first", run_id=first_id)
    _invoke(callback, "model:second", run_id=second_id)

    second = callback.take_exact_match(
        _finish_event("model:second"), logical_call_id="run:model:second"
    )
    first = callback.take_exact_match(
        _finish_event("model:first"), logical_call_id="run:model:first"
    )
    assert second is not None
    assert first is not None
    assert second["logical_call_id"] == "run:model:second"
    assert first["logical_call_id"] == "run:model:first"


def test_callback_fails_closed_on_mismatch_ambiguity_and_saturation() -> None:
    callback = TraceLensInputCallback()
    _invoke(callback, "model:one")
    assert callback.take_exact_match(_finish_event("model:two"), logical_call_id="x") is None

    _invoke(callback, "model:one")
    _invoke(callback, "model:one")
    assert callback.take_exact_match(_finish_event("model:one"), logical_call_id="x") is None

    saturated = TraceLensInputCallback()
    for index in range(TRACE_LENS_MAX_PENDING_CALLS + 1):
        _invoke(saturated, f"model:bounded-{index}", run_id=uuid4())
    assert (
        saturated.take_exact_match(
            _finish_event("model:bounded-0"), logical_call_id="run:model:bounded"
        )
        is None
    )
    assert saturated._records == {}
    assert saturated._ambiguous == set()


def test_callback_matches_v2_only_on_exact_callback_run_id() -> None:
    callback = TraceLensInputCallback()
    asyncio.run(
        callback.on_chat_model_start(
            {},
            [[HumanMessage(content="SECRET")]],
            run_id=CALLBACK_RUN_ID,
            metadata={},
            invocation_params={"tools": []},
        )
    )
    assert (
        callback.take_exact_match(
            {"event": "on_chat_model_end", "run_id": str(CALLBACK_RUN_ID)},
            logical_call_id="run:model:v2-call",
        )
        is not None
    )


def test_agent_configuration_is_honest_bounded_and_content_free() -> None:
    configuration = build_agent_configuration(
        model_id="openai/deepseek_v4",
        provider_id="vllm",
        registered_tools=[
            {"name": "execute"},
            {"name": "artifact_manifest"},
            {"name": "not safe <SECRET>"},
        ],
        subagents=[
            {"name": "code-runner"},
            {"name": "operator-secret-agent"},
            {"name": "builder"},
        ],
        async_subagent_count=4,
        builder_enabled=True,
        declared_memory_count=3,
        declared_policy_count=1,
        declared_skill_count=1,
    )

    assert configuration["schema_version"] == 2
    assert configuration["capture_scope"] == "configuration"
    assert configuration["model_id"] == "openai/deepseek_v4"
    assert configuration["provider_id"] == "vllm"
    assert configuration["domain_tool_declaration_count"] == 3
    assert configuration["domain_tool_names"] == ["artifact_manifest", "execute"]
    assert configuration["declared_local_subagent_count"] == 3
    assert configuration["code_owned_subagent_names"] == ["builder", "code-runner"]
    assert configuration["configured_async_subagent_count"] == 4
    assert configuration["declared_memory_count"] == 3
    assert configuration["declared_policy_count"] == 1
    assert "operator-secret-agent" not in json.dumps(configuration)
    assert "SECRET" not in json.dumps(configuration)


def test_agent_construction_isolates_a_raising_trace_sink(monkeypatch) -> None:
    monkeypatch.setattr("ultra_deepagents.agent.build_builder_subagent", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "ultra_deepagents.agent.create_deep_agent",
        lambda **_kwargs: "compiled-agent",
    )

    def raising_sink(_configuration: dict[str, object]) -> None:
        raise RuntimeError("SECRET_SINK_FAILURE_CANARY")

    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
    )
    agent = build_research_agent(
        settings,
        model=object(),
        backend=object(),
        trace_surface_sink=raising_sink,
    )
    assert agent == "compiled-agent"


class _CallbackDrivingUsageAgent:
    def __init__(
        self,
        *,
        finish_checkpoint_ns: str = "model:exact",
        leave_pending_record: bool = False,
    ) -> None:
        self.finish_checkpoint_ns = finish_checkpoint_ns
        self.leave_pending_record = leave_pending_record
        self.saw_trace_callback = False
        self.trace_callback: TraceLensInputCallback | None = None

    def astream_events(self, _payload, *, config=None, **_kwargs):
        callbacks = list((config or {}).get("callbacks") or [])
        self.saw_trace_callback = any(
            isinstance(callback, TraceLensInputCallback) for callback in callbacks
        )
        self.trace_callback = next(
            (callback for callback in callbacks if isinstance(callback, TraceLensInputCallback)),
            None,
        )

        async def generate():
            for callback in callbacks:
                if isinstance(callback, TraceLensInputCallback):
                    await callback.on_chat_model_start(
                        {},
                        [[HumanMessage(content="SECRET_PROMPT_CANARY")]],
                        run_id=CALLBACK_RUN_ID,
                        metadata={
                            "langgraph_node": "model",
                            "langgraph_checkpoint_ns": "model:exact",
                        },
                        invocation_params={
                            "tools": [{"type": "function", "function": {"name": "execute"}}],
                            "tool_choice": "auto",
                        },
                    )
                    if self.leave_pending_record:
                        await callback.on_chat_model_start(
                            {},
                            [[HumanMessage(content="SECRET_PENDING_CANARY")]],
                            run_id=UUID("99999999-8888-7777-6666-555555555555"),
                            metadata={
                                "langgraph_node": "model",
                                "langgraph_checkpoint_ns": "model:pending",
                            },
                            invocation_params={"tools": []},
                        )
            yield _finish_event(self.finish_checkpoint_ns)
            yield {
                "type": "event",
                "method": "values",
                "params": {
                    "namespace": [],
                    "data": {"messages": [{"role": "assistant", "content": "done"}]},
                },
            }

        return generate()


def _stream(agent: object, published: list[dict[str, object]], **kwargs: object) -> None:
    async def scenario() -> None:
        async def publish(event):
            published.append(event)

        await _stream_agent_attempt(
            agent,
            messages=[{"role": "user", "content": "request"}],
            context=_context(),
            sequencer=RunEventSequencer("run-trace-lens"),
            publish_event=publish,
            **kwargs,
        )

    asyncio.run(scenario())


def test_stream_enriches_one_existing_usage_event_with_schema_v2() -> None:
    published: list[dict[str, object]] = []
    agent = _CallbackDrivingUsageAgent()
    configuration = {
        "schema_version": 2,
        "capture_scope": "configuration",
        "content_captured": False,
        "domain_tool_declaration_count": 2,
    }
    _stream(
        agent,
        published,
        trace_lens_inputs_enabled=True,
        trace_surface=configuration,
    )

    assert agent.saw_trace_callback is True
    assert len(published) == 1
    event = published[0]
    assert event["event_kind"] == "run.token_usage"
    assert event["sequence"] == 1
    trace_lens = event["payload"]["trace_lens"]
    assert trace_lens["schema_version"] == 2
    assert trace_lens["capture_scope"] == "attempt"
    assert (
        trace_lens["logical_model_input"]["logical_call_id"] == event["payload"]["usage_event_id"]
    )
    assert trace_lens["agent_configuration"]["domain_tool_declaration_count"] == 2
    assert trace_lens["logical_model_input"]["provider_bound_tools"]["function_tool_names"] == [
        "execute"
    ]
    assert "SECRET_PROMPT_CANARY" not in json.dumps(event)
    assert "checkpoint" not in json.dumps(trace_lens).lower()


def test_stream_finalization_clears_unmatched_callback_records() -> None:
    published: list[dict[str, object]] = []
    agent = _CallbackDrivingUsageAgent(leave_pending_record=True)
    _stream(agent, published, trace_lens_inputs_enabled=True)
    assert len(published) == 1
    assert agent.trace_callback is not None
    assert agent.trace_callback._records == {}
    assert agent.trace_callback._ambiguous == set()


def test_stream_omits_trace_on_mismatch_and_when_disabled() -> None:
    mismatched: list[dict[str, object]] = []
    _stream(
        _CallbackDrivingUsageAgent(finish_checkpoint_ns="model:different"),
        mismatched,
        trace_lens_inputs_enabled=True,
    )
    assert len(mismatched) == 1
    assert "trace_lens" not in mismatched[0]["payload"]

    disabled: list[dict[str, object]] = []
    agent = _CallbackDrivingUsageAgent()
    _stream(agent, disabled)
    assert agent.saw_trace_callback is False
    assert len(disabled) == 1
    assert "trace_lens" not in disabled[0]["payload"]


def test_disabled_path_is_deep_equal_to_the_uninstrumented_usage_event() -> None:
    first: list[dict[str, object]] = []
    second: list[dict[str, object]] = []
    _stream(_CallbackDrivingUsageAgent(), first, trace_lens_inputs_enabled=False)
    _stream(_CallbackDrivingUsageAgent(), second)
    for event in (*first, *second):
        event.pop("ts", None)
    assert first == second


def test_runner_isolates_a_raising_attachment(monkeypatch) -> None:
    import ultra_deepagents.runner as runner

    def raising_attach(*_args, **_kwargs):
        raise RuntimeError("SECRET_ATTACH_FAILURE_CANARY")

    monkeypatch.setattr(runner, "attach_trace_lens_metadata", raising_attach)
    published: list[dict[str, object]] = []
    _stream(
        _CallbackDrivingUsageAgent(),
        published,
        trace_lens_inputs_enabled=True,
    )
    assert len(published) == 1
    assert "trace_lens" not in published[0]["payload"]
    assert "SECRET" not in json.dumps(published)


def test_runner_isolates_a_raising_callback_configuration(monkeypatch) -> None:
    import ultra_deepagents.runner as runner

    class RaisingCallback:
        def __init__(self) -> None:
            raise RuntimeError("SECRET_CALLBACK_CONFIGURATION_CANARY")

    monkeypatch.setattr(runner, "TraceLensInputCallback", RaisingCallback)
    published: list[dict[str, object]] = []
    _stream(
        _CallbackDrivingUsageAgent(),
        published,
        trace_lens_inputs_enabled=True,
    )
    assert len(published) == 1
    assert "trace_lens" not in published[0]["payload"]


def test_attach_helper_is_bounded_atomic_and_does_not_stamp() -> None:
    event = {"payload": {"usage_event_id": "run:model:call"}}
    attached = attach_trace_lens_metadata(
        event,
        logical_model_input={
            "schema_version": 2,
            "capture_scope": "logical_model_invocation",
            "content_captured": False,
        },
    )
    assert attached is True
    assert "event_id" not in event
    assert "sequence" not in event
    encoded = json.dumps(event["payload"]["trace_lens"], default=str).encode()
    assert len(encoded) <= TRACE_LENS_MAX_SERIALIZED_BYTES

    oversized = {"payload": {"usage_event_id": "run:model:large"}}
    before = copy.deepcopy(oversized)
    assert (
        attach_trace_lens_metadata(
            oversized,
            logical_model_input={"safe_but_large": "x" * TRACE_LENS_MAX_SERIALIZED_BYTES},
        )
        is False
    )
    assert oversized == before


def test_maximum_valid_names_compact_without_losing_logical_input() -> None:
    def maximum_name(prefix: str, index: int) -> str:
        return (f"{prefix}_{index:02d}_" + "x" * 96)[:96]

    configuration = build_agent_configuration(
        model_id="provider/" + "m" * 63,
        provider_id="provider_id",
        registered_tools=[{"name": maximum_name("domain", index)} for index in range(80)],
        subagents=[{"name": "code-runner"}, {"name": "builder"}],
        async_subagent_count=8,
        builder_enabled=True,
        declared_memory_count=3,
        declared_policy_count=1,
        declared_skill_count=1,
    )
    callback = TraceLensInputCallback()
    _invoke(
        callback,
        "model:maximum-valid-names",
        invocation_params={
            "tools": [
                {
                    "type": "function",
                    "function": {"name": maximum_name("function", index)},
                }
                for index in range(40)
            ]
        },
    )
    logical_input = callback.take_exact_match(
        _finish_event("model:maximum-valid-names"),
        logical_call_id="run:model:maximum-valid-names",
    )
    assert logical_input is not None
    event = {"payload": {"usage_event_id": "run:model:maximum-valid-names"}}
    assert (
        attach_trace_lens_metadata(
            event,
            logical_model_input=logical_input,
            agent_configuration=configuration,
        )
        is True
    )
    trace = event["payload"]["trace_lens"]
    assert "logical_model_input" in trace
    assert len(json.dumps(trace, default=str).encode()) <= TRACE_LENS_MAX_SERIALIZED_BYTES
    tools = trace["logical_model_input"]["provider_bound_tools"]
    assert tools["function_names_omitted_due_to_size"] is True
    assert tools["omitted_function_name_count"] == 40
    assert tools.get("function_tool_names") is None
    if "agent_configuration" in trace:
        compact_configuration = trace["agent_configuration"]
        assert compact_configuration["domain_tool_names_omitted_due_to_size"] is True
        assert compact_configuration["domain_tool_name_omitted_count"] == 80
    else:
        assert trace["agent_configuration_omitted_due_to_size"] is True


def test_transport_bound_uses_default_spaced_publisher_serialization() -> None:
    padding = None
    for length in range(3500, TRACE_LENS_MAX_SERIALIZED_BYTES + 1):
        candidate = {
            "schema_version": 2,
            "capture_scope": "attempt",
            "content_captured": False,
            "logical_model_input": {"safe_padding": "x" * length},
        }
        compact_size = len(json.dumps(candidate, separators=(",", ":")).encode())
        spaced_size = len(json.dumps(candidate, default=str).encode())
        if compact_size <= TRACE_LENS_MAX_SERIALIZED_BYTES < spaced_size:
            padding = length
            break
    assert padding is not None
    event = {"payload": {"usage_event_id": "run:model:spaced-boundary"}}
    assert (
        attach_trace_lens_metadata(
            event,
            logical_model_input={"safe_padding": "x" * padding},
        )
        is False
    )
    assert "trace_lens" not in event["payload"]
