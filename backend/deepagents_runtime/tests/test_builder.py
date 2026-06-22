"""Tests for the Builder - the model-agnostic autonomous-coding sub-coordinator.

See planning/2026-06-21-autonomous-builder-subagent.md. Covers the GoalLoop termination
logic (the "/goal until met" engine), the flag-gated registration, and the model-agnostic
fallback.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from ultra_deepagents.builder import (
    GOAL_LOOP_SOURCE,
    GoalLoopMiddleware,
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


def test_goal_loop_malformed_verdict_treated_as_missing():
    gl = GoalLoopMiddleware(max_iterations=6)
    state = _state(AIMessage(content="GOAL_OUTCOME: {not valid json"))
    out = gl.after_agent(state, None)
    assert out is not None and out["jump_to"] == "model"  # can't parse -> keep going


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
    sub = build_builder_subagent(
        _settings(
            builder_enabled=True,
            builder_goal_max_iterations=4,
            builder_recursion_limit=123,
        ),
        tools=["execute"],
        backend=object(),
    )
    assert sub is not None and sub["name"] == "builder"
    assert (
        "build-until-goal" in sub["description"].lower()
        or "owns the whole loop" in sub["description"].lower()
    )
    # the GoalLoop middleware is attached with the configured cap
    mw = captured["middleware"][0]
    assert isinstance(mw, GoalLoopMiddleware) and mw.max_iterations == 4
    assert captured["model"] == "builder-model"
    assert bound == {"recursion_limit": 123}


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
    build_builder_subagent(
        _settings(builder_enabled=True, builder_multimodal=True),
        tools=["execute"],
        backend=object(),
        vision_tools=["inspect_images"],
    )
    assert "inspect_images" in captured["tools"]  # self-check capability wired in


def test_builder_delegation_guidance_in_prompt_only_when_enabled():
    """The coordinator gets the 'delegate heavy coding to the Builder early' discipline only when
    the Builder is enabled; otherwise it would steer toward an absent subagent."""
    from ultra_deepagents.agent import build_system_prompt
    from ultra_deepagents.builder import BUILDER_DELEGATION_GUIDANCE

    on = build_system_prompt(_settings(builder_enabled=True))
    off = build_system_prompt(_settings(builder_enabled=False))
    assert BUILDER_DELEGATION_GUIDANCE in on
    assert "DELEGATE IT TO THE BUILDER EARLY" in on
    assert BUILDER_DELEGATION_GUIDANCE not in off


def test_build_builder_model_falls_back_to_coordinator_when_unconfigured():
    from ultra_deepagents.model import build_builder_model

    # no builder_base_url/model -> uses the coordinator's model (single-model deployments still work)
    m = build_builder_model(_settings())
    assert m.model_name == "deepseek_v4"
