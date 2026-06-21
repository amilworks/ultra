"""Episodic memory tool tests."""

from __future__ import annotations

from ultra_deepagents.agent import build_research_agent
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.episodic.tools import build_episodic_tools, search_past_research_history


def _settings() -> RuntimeSettings:
    return RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
        control_worker_token="trace-worker-secret",
    )


def _context(*, user_id: str = "user-1", run_id: str = "run-now") -> AgentRunContext:
    return AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="org-1",
        user_id=user_id,
        project_id="proj",
        thread_id="thread-now",
        run_id=run_id,
        goal="What did we conclude about the prairie dog density last time?",
    )


def _fake_httpx(monkeypatch, captured: dict, payload: dict):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    class FakeClient:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json, headers=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers or {}
            return FakeResponse()

    monkeypatch.setattr("httpx.Client", FakeClient)


def test_search_past_research_posts_run_anchored_request(monkeypatch):
    captured: dict = {}
    _fake_httpx(
        monkeypatch,
        captured,
        {
            "hits": [
                {
                    "run_id": "run-old",
                    "thread_id": "thread-old",
                    "title": "Prairie dog density 2024",
                    "goal": "Compare 2024 burrow density",
                    "response_snippet": "Mean burrow density 12.3 per hectare.",
                    "completed_at": "2026-05-14T10:00:00Z",
                }
            ]
        },
    )

    result = search_past_research_history(
        _settings(), context=_context(), query="prairie dog density", since_days=30, limit=5
    )

    # Run-anchored, worker-authenticated POST to the episodic endpoint.
    assert captured["url"] == "http://control.test/v2/runs/run-now/episodic-search"
    assert captured["headers"]["X-Ultra-Worker-Token"] == "trace-worker-secret"
    assert captured["headers"]["X-Ultra-Run-Id"] == "run-now"
    assert captured["headers"]["X-Ultra-User-Id"] == "user-1"
    assert captured["json"] == {"query": "prairie dog density", "limit": 5, "since_days": 30}
    assert result["ok"] is True
    assert result["hits"][0]["response_snippet"].startswith("Mean burrow density")


def test_search_past_research_handles_errors_gracefully(monkeypatch):
    class BoomClient:
        def __init__(self, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json, headers=None):
            raise RuntimeError("control plane down")

    monkeypatch.setattr("httpx.Client", BoomClient)

    result = search_past_research_history(_settings(), context=_context(), query="x")
    assert result["ok"] is False
    assert "control plane down" in result["error"]
    assert result["hits"] == []


def test_search_past_research_requires_run_id():
    result = search_past_research_history(
        _settings(), context=_context(run_id=""), query="x"
    )
    assert result["ok"] is False
    assert result["error"] == "no_run_id"


def test_build_episodic_tools_exposes_search_tool():
    tools = build_episodic_tools(_settings())
    names = {getattr(t, "name", "") for t in tools}
    assert names == {"search_past_research"}


def test_agent_registers_episodic_tool_for_authenticated_user(monkeypatch):
    captured: dict = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "compiled-agent"

    monkeypatch.setattr("ultra_deepagents.agent.create_deep_agent", fake_create_deep_agent)
    build_research_agent(_settings(), model=object(), backend=object(), context=_context())

    tool_names = {getattr(t, "name", "") for t in captured["tools"]}
    assert "search_past_research" in tool_names
    assert "search_past_research" in captured["system_prompt"]


def test_agent_omits_episodic_tool_for_anonymous_context(monkeypatch):
    captured: dict = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "compiled-agent"

    monkeypatch.setattr("ultra_deepagents.agent.create_deep_agent", fake_create_deep_agent)
    build_research_agent(
        _settings(), model=object(), backend=object(), context=_context(user_id="")
    )

    tool_names = {getattr(t, "name", "") for t in captured["tools"]}
    assert "search_past_research" not in tool_names
    assert "search_past_research" not in captured["system_prompt"]
