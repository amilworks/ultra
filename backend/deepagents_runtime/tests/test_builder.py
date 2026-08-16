"""Tests for the Builder - the model-agnostic autonomous-coding sub-coordinator.

See planning/2026-06-21-autonomous-builder-subagent.md. Covers the GoalLoop termination
logic (the "/goal until met" engine), the flag-gated registration, and the model-agnostic
fallback.
"""

from __future__ import annotations

from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from ultra_deepagents.builder import (
    BUILDER_SYSTEM_PROMPT,
    GOAL_LOOP_INJECTIONS_KEY,
    GOAL_LOOP_SOURCE,
    GoalLoopMiddleware,
    StripImagesForTextModelMiddleware,
    _strip_image_content,
    build_builder_subagent,
)
from ultra_deepagents.config import RuntimeSettings


def _settings(**over) -> RuntimeSettings:
    base = dict(openai_base_url="http://127.0.0.1:8003/v1", openai_model="deepseek_v4")
    base.update(over)
    return RuntimeSettings(**base)


def _state(*messages):
    return {"messages": list(messages)}


def _nudge(n: int):
    return [
        HumanMessage(
            content="keep going",
            name=GOAL_LOOP_SOURCE,
            additional_kwargs={"lc_source": GOAL_LOOP_SOURCE},
        )
        for _ in range(n)
    ]


def test_goal_loop_finishes_when_goal_verified_met():
    gl = GoalLoopMiddleware(max_iterations=6)
    state = _state(
        AIMessage(content='Done.\nGOAL_OUTCOME: {"met": true, "metric": 0.71, "target": "> 0.65"}')
    )
    assert gl.after_agent(state, None) is None  # met -> finish, no jump


def test_goal_loop_jumps_back_when_goal_not_met():
    gl = GoalLoopMiddleware(max_iterations=6)
    state = _state(
        AIMessage(content='GOAL_OUTCOME: {"met": false, "metric": 0.60, "target": "> 0.65"}')
    )
    out = gl.after_agent(state, None)
    assert out is not None and out["jump_to"] == "model"
    msg = out["messages"][0]
    assert msg.additional_kwargs["lc_source"] == GOAL_LOOP_SOURCE
    assert "not met" in msg.content.lower() and "budget" in msg.content.lower()


def test_goal_loop_jumps_when_no_verdict_emitted():
    gl = GoalLoopMiddleware(max_iterations=6)
    state = _state(AIMessage(content="I think the model is trained now."))  # no GOAL_OUTCOME
    out = gl.after_agent(state, None)
    assert out is not None and out["jump_to"] == "model"
    assert "without a verdict" in out["messages"][0].content.lower()


def test_goal_loop_budget_exhausted_finishes_even_if_unmet():
    gl = GoalLoopMiddleware(max_iterations=2)
    # 2 prior nudges already injected -> at the cap -> finish regardless of unmet verdict
    state = _state(*_nudge(2), AIMessage(content='GOAL_OUTCOME: {"met": false, "metric": 0.6}'))
    assert gl.after_agent(state, None) is None  # pathology cap fires, never a depth cap


def test_goal_loop_cap_holds_when_summarization_prunes_nudges():
    """Regression for the #1 prod-enable blocker: SummarizationMiddleware rewrites `messages`
    with RemoveMessage(REMOVE_ALL_MESSAGES), pruning the goal_loop nudge tags. A message-derived
    budget would reset to 0 and loop forever; the DURABLE STATE counter must still fire the cap."""
    gl = GoalLoopMiddleware(max_iterations=6)
    # Post-compaction: a fresh summary message + an unmet verdict; ZERO goal_loop nudges survive.
    state = {
        "messages": [
            HumanMessage(content="<summary of prior work>", additional_kwargs={"lc_source": "summarization"}),
            AIMessage(content='GOAL_OUTCOME: {"met": false, "metric": 0.6, "target": "> 0.65"}'),
        ],
        GOAL_LOOP_INJECTIONS_KEY: 6,  # the count the summarizer cannot touch
    }
    assert gl.after_agent(state, None) is None  # cap fires from state, not from messages


def test_goal_loop_persists_incremented_count_in_state():
    """Each injected nudge bumps the durable counter so the cap survives history compaction."""
    gl = GoalLoopMiddleware(max_iterations=6)
    state = {
        "messages": [AIMessage(content='GOAL_OUTCOME: {"met": false, "metric": 0.6}')],
        GOAL_LOOP_INJECTIONS_KEY: 3,
    }
    out = gl.after_agent(state, None)
    assert out is not None and out["jump_to"] == "model"
    assert out[GOAL_LOOP_INJECTIONS_KEY] == 4  # bumped + persisted to durable state


def test_goal_loop_cap_enforced_in_compiled_graph_via_durable_state():
    """End-to-end: the durable counter registers as a graph channel and enforces the cap
    through a REAL compiled deep-agent graph WITHOUT the message-scan fallback -- the exact
    path that holds after SummarizationMiddleware prunes the nudges from history."""
    from deepagents import create_deep_agent
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    class _FakeBind(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):  # the deep-agent stack binds tools
            return self

    class _StateOnly(GoalLoopMiddleware):
        @staticmethod
        def _message_injections(messages):  # disable the lower-bound fallback -> state only
            return 0

    agent = create_deep_agent(
        model=_FakeBind(messages=(f"working {i}" for i in range(2000))),
        tools=[],
        system_prompt=BUILDER_SYSTEM_PROMPT,
        middleware=[_StateOnly(max_iterations=3)],
    ).with_config({"recursion_limit": 60})
    assert GOAL_LOOP_INJECTIONS_KEY in (getattr(agent, "channels", {}) or {})  # channel registered
    result = agent.invoke({"messages": [("user", "GOAL: X. DONE-WHEN: metric>0.9.")]})
    nudges = sum(
        1 for m in result["messages"]
        if (getattr(m, "additional_kwargs", {}) or {}).get("lc_source") == GOAL_LOOP_SOURCE
    )
    assert nudges == 3  # capped purely by the durable state counter, no message fallback


def test_goal_loop_malformed_verdict_treated_as_missing():
    gl = GoalLoopMiddleware(max_iterations=6)
    state = _state(AIMessage(content="GOAL_OUTCOME: {not valid json"))
    out = gl.after_agent(state, None)
    assert out is not None and out["jump_to"] == "model"  # can't parse -> keep going


def test_goal_loop_parses_real_verdict_after_echoed_template():
    """Regression: the prompt shows the literal GOAL_OUTCOME template, so the model sometimes
    echoes it earlier then emits the REAL verdict last. A greedy first-tag match would span both
    and capture invalid JSON -> a met goal looks unmet -> one wasted iteration. Anchor on the LAST tag."""
    gl = GoalLoopMiddleware(max_iterations=6)
    content = (
        'I will finish by emitting GOAL_OUTCOME: {"met": ...} as instructed.\n'
        'Here is some prose between the template and the real verdict.\n'
        'GOAL_OUTCOME: {"met": true, "metric": 0.97, "target": "> 0.9"}'
    )
    state = _state(AIMessage(content=content))
    assert gl.after_agent(state, None) is None  # parses the LAST verdict -> met -> finish (no extra loop)


def test_goal_loop_handles_list_content_blocks():
    """Regression: via the worker's streaming path AIMessage.content is a LIST of blocks, not a
    str; a naive regex raised 'expected string ... got list' and failed the live run."""
    gl = GoalLoopMiddleware(max_iterations=6)
    blocks = [
        {"type": "text", "text": "done "},
        {"type": "text", "text": 'GOAL_OUTCOME: {"met": true}'},
    ]
    state = _state(AIMessage(content=blocks))
    assert gl.after_agent(state, None) is None  # parses the verdict out of list content -> finish


def test_builder_disabled_returns_none():
    assert (
        build_builder_subagent(_settings(builder_enabled=False), tools=[], backend=object()) is None
    )


def test_builder_registered_when_enabled(monkeypatch):
    captured = {}
    bound = {}

    class FakeRunnable:
        def with_config(self, config):
            bound.update(config)
            return self

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return FakeRunnable()

    monkeypatch.setattr("ultra_deepagents.builder.create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr("ultra_deepagents.builder.build_builder_model", lambda s: "builder-model")
    monkeypatch.setattr("ultra_deepagents.builder.build_builder_worker_model", lambda s: "worker-model")
    backend = object()
    sub = build_builder_subagent(
        _settings(
            builder_enabled=True,
            builder_goal_max_iterations=4,
            builder_recursion_limit=123,
        ),
        tools=["execute"],
        backend=backend,
        permissions=["DENY-/policies"],
    )
    assert sub is not None and sub["name"] == "builder"
    assert (
        "build-until-goal" in sub["description"].lower()
        or "owns the whole loop" in sub["description"].lower()
    )
    # the GoalLoop middleware is attached to the LEAD with the configured cap
    mw = captured["middleware"][0]
    assert isinstance(mw, GoalLoopMiddleware) and mw.max_iterations == 4
    assert captured["model"] == "builder-model"
    assert bound == {"recursion_limit": 123}
    # SECURITY: the caller's filesystem denies are passed to the inner Builder's create_deep_agent
    # (a pre-compiled CompiledSubAgent does NOT inherit the parent's permissions, so this is the
    # only thing stopping the Builder + worker from writing /policies, /skills, /memories).
    assert captured["permissions"] == ["DENY-/policies"]
    # the Builder gets an explicit "general-purpose" sub-worker on the WORKER model so the lead can
    # delegate focused sub-runs to a (possibly different) executor endpoint.
    worker = captured["subagents"][0]
    assert worker["name"] == "general-purpose"  # replaces the globally-disabled auto-added worker
    assert worker["model"] == "worker-model"
    # worker omits its own permissions key, so it INHERITS the lead's denies (which now exist)
    assert "permissions" not in worker
    # the worker carries image-stripping (text-only by construction) but NOT the GoalLoop (lead-only)
    assert any(isinstance(m, StripImagesForTextModelMiddleware) for m in worker["middleware"])
    assert not any(isinstance(m, GoalLoopMiddleware) for m in worker["middleware"])
    lead_todos = [m for m in captured["middleware"] if isinstance(m, TodoListMiddleware)]
    lead_filesystems = [m for m in captured["middleware"] if isinstance(m, FilesystemMiddleware)]
    worker_todos = [m for m in worker["middleware"] if isinstance(m, TodoListMiddleware)]
    worker_filesystems = [m for m in worker["middleware"] if isinstance(m, FilesystemMiddleware)]
    assert len(lead_todos) == len(lead_filesystems) == 1
    assert len(worker_todos) == len(worker_filesystems) == 1
    assert lead_todos[0] is not worker_todos[0]
    assert lead_filesystems[0] is not worker_filesystems[0]
    assert lead_filesystems[0].backend is worker_filesystems[0].backend is backend
    assert (
        lead_filesystems[0]._permissions == worker_filesystems[0]._permissions == ["DENY-/policies"]
    )
    assert "delete" not in {tool.name for tool in lead_filesystems[0].tools}
    assert "delete" not in {tool.name for tool in worker_filesystems[0].tools}


def test_strip_image_content_drops_images_keeps_text():
    """A text-only Builder model 400s on any image block; the guard must drop them (leaving a
    breadcrumb) while leaving plain-text messages untouched. Regression: the Builder read its
    own rendered .png via read_file and hard-failed with 'not a multimodal model'."""
    msgs = [
        HumanMessage(content="plain text stays"),
        ToolMessage(
            content=[
                {"type": "text", "text": "fit_log.csv contents"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
            tool_call_id="t1",
        ),
    ]
    out = _strip_image_content(msgs)
    assert out[0].content == "plain text stays"  # untouched
    blocks = out[1].content
    assert all("image_url" not in b and "image" not in str(b.get("type", "")) for b in blocks)
    assert any(b.get("text", "").startswith("[image omitted") for b in blocks)
    assert any(b.get("text") == "fit_log.csv contents" for b in blocks)  # text block preserved


def test_builder_attaches_image_strip_when_text_only(monkeypatch):
    captured = {}

    class FakeRunnable:
        def with_config(self, config):
            return self

    monkeypatch.setattr(
        "ultra_deepagents.builder.create_deep_agent",
        lambda **kw: captured.update(kw) or FakeRunnable(),
    )
    monkeypatch.setattr("ultra_deepagents.builder.build_builder_model", lambda s: "m")
    # text-only (default multimodal=False) -> strip guard attached
    build_builder_subagent(_settings(builder_enabled=True), tools=["execute"], backend=object())
    assert any(isinstance(mw, StripImagesForTextModelMiddleware) for mw in captured["middleware"])
    # multimodal builder (can SEE) -> no strip guard
    captured.clear()
    build_builder_subagent(
        _settings(builder_enabled=True, builder_multimodal=True),
        tools=["execute"],
        backend=object(),
        vision_tools=["inspect_images"],
    )
    assert not any(isinstance(mw, StripImagesForTextModelMiddleware) for mw in captured["middleware"])


def test_builder_multimodal_adds_vision_tools(monkeypatch):
    captured = {}

    class FakeRunnable:
        def with_config(self, config):
            return self

    monkeypatch.setattr(
        "ultra_deepagents.builder.create_deep_agent",
        lambda **kw: captured.update(kw) or FakeRunnable(),
    )
    monkeypatch.setattr("ultra_deepagents.builder.build_builder_model", lambda s: "m")
    monkeypatch.setattr("ultra_deepagents.builder.build_builder_worker_model", lambda s: "w")
    build_builder_subagent(
        _settings(builder_enabled=True, builder_multimodal=True),
        tools=["execute"],
        backend=object(),
        vision_tools=["inspect_images"],
    )
    assert "inspect_images" in captured["tools"]  # lead self-check capability wired in
    # the worker is TEXT-ONLY by construction even when the LEAD is multimodal: it must NOT get the
    # vision tools (only the base tools), so a stray image can never 400 the Qwen worker.
    worker = captured["subagents"][0]
    assert "inspect_images" not in worker["tools"]
    assert any(isinstance(m, StripImagesForTextModelMiddleware) for m in worker["middleware"])


def test_builder_delegation_guidance_tracks_builder_enabled():
    """The coordinator advertises the Builder's delegation discipline exactly when the
    Builder is registered (builder_enabled). Re-enabled after the 6.7M-token livelock
    trace: iterative build/debug loops must be steered into the Builder's ISOLATED
    context instead of accumulating inline in the coordinator. Disabled must stay
    silent so the model is never told to delegate to an absent subagent."""
    from ultra_deepagents.agent import build_system_prompt
    from ultra_deepagents.builder import BUILDER_DELEGATION_GUIDANCE

    on = build_system_prompt(_settings(builder_enabled=True))
    off = build_system_prompt(_settings(builder_enabled=False))
    assert BUILDER_DELEGATION_GUIDANCE in on
    assert "DELEGATE IT TO THE BUILDER EARLY" in on
    assert BUILDER_DELEGATION_GUIDANCE not in off
    assert "DELEGATE IT TO THE BUILDER EARLY" not in off


def test_builder_disabled_by_default():
    """builder_enabled defaults OFF (cleanup candidate): a live A/B found the Builder
    concentrates ~92% of a run's tokens for little net gain over the plain coordinator +
    code-runner path, which the progress-stall guard already protects. The wiring is kept
    behind the flag for a future A/B; ULTRA_DEEPAGENTS_BUILDER_ENABLED=1 re-enables it."""
    assert _settings().builder_enabled is False


def test_builder_prompt_steers_parallel_fanout():
    """Concurrent fan-out is a prompt-steering feature: the async path already runs same-turn task()
    calls concurrently, so the Builder must be told to emit K independent sub-units as K calls in ONE
    turn (with distinct output paths). Without this the lead serializes and the speedup is lost."""
    p = BUILDER_SYSTEM_PROMPT
    assert "FAN OUT IN PARALLEL" in p
    assert "SINGLE turn" in p and "CONCURRENTLY" in p
    assert "DISTINCT output path" in p  # output-collision guard for concurrent workers


def test_builder_prompt_requires_pilot_checkpoint_and_budget_gate_for_long_runs():
    """Long scientific experiments must not become one opaque execute() call: the Builder
    should pilot timing, checkpoint intermediate outputs, and shrink/chunk oversized grids."""
    p = BUILDER_SYSTEM_PROMPT
    assert "pilot timing pass" in p
    assert "checkpoint" in p and "/outputs" in p
    assert "after each condition or batch" in p
    assert "shrink or chunk" in p
    assert "estimated runtime" in p


def test_build_builder_model_falls_back_to_coordinator_when_unconfigured():
    from ultra_deepagents.model import build_builder_model

    # no builder_base_url/model -> uses the coordinator's model (single-model deployments still work)
    m = build_builder_model(_settings())
    assert m.model_name == "deepseek_v4"


def test_build_builder_worker_model_falls_back_to_lead_when_unconfigured():
    from ultra_deepagents.model import build_builder_worker_model

    # no worker block + no builder block -> worker == lead == coordinator model (single-model default)
    assert build_builder_worker_model(_settings()).model_name == "deepseek_v4"
    # worker unset but the lead points at its own endpoint -> worker inherits the LEAD endpoint
    m = build_builder_worker_model(
        _settings(builder_base_url="http://tesla:8000/v1", builder_model="Qwen3.6-27B")
    )
    assert m.model_name == "Qwen3.6-27B"


def test_build_builder_worker_model_uses_worker_endpoint_when_set():
    from ultra_deepagents.model import UltraChatOpenAI, build_builder_worker_model

    m = build_builder_worker_model(
        _settings(
            builder_base_url="http://h200:9393/v1",
            builder_model="deepseek_v4",
            builder_worker_base_url="http://tesla:8000/v1",
            builder_worker_model="Qwen3.6-27B",
            builder_worker_max_input_tokens=262144,
        )
    )
    # must be an UltraChatOpenAI INSTANCE (not a 'provider:model' string routed through
    # init_chat_model, which would silently lose the reasoning-delta lift + stream_usage + profile)
    assert isinstance(m, UltraChatOpenAI)
    assert m.model_name == "Qwen3.6-27B"  # the WORKER model, not the deepseek lead
    assert m.profile == {"max_input_tokens": 262144}


def test_build_builder_worker_model_partial_config_falls_back_to_lead():
    from ultra_deepagents.model import build_builder_worker_model

    # only base_url set (no model) -> operator mistake -> fall back to the lead model, do not crash
    m = build_builder_worker_model(
        _settings(
            builder_base_url="http://h200:9393/v1",
            builder_model="deepseek_v4",
            builder_worker_base_url="http://tesla:8000/v1",
        )
    )
    assert m.model_name == "deepseek_v4"  # fell back to the lead (deepseek), not a broken worker
