"""Run-scoped private Notes tool and trace contracts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from langchain.agents.middleware import ModelRequest
from langchain.agents.middleware.types import ModelResponse
from langchain.tools import ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import Field
from ultra_deepagents.agent import (
    UltraRunContextPromptMiddleware,
    build_notes_run_context_brief,
    build_research_agent,
    build_run_context_brief,
    build_system_prompt,
)
from ultra_deepagents.async_delegation import async_subagent_context_payload
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.evaluation_profiles import EvaluationProfilePolicy
from ultra_deepagents.notes.access import note_access_from_selection_context
from ultra_deepagents.notes.tools import (
    NOTE_TOOL_NAMES,
    build_notes_tools,
    create_note_append_proposal,
    note_append_proposal_goal_authorized,
    read_user_note,
    search_user_notes,
)
from ultra_deepagents.runner import (
    _subagent_message_event_from_stream_event,
    _tool_event_from_stream_event,
)
from ultra_deepagents.schemas import RunJobEnvelope


class _ToolCallingFakeModel(FakeMessagesListChatModel):
    bound_tool_names: list[list[str]] = Field(default_factory=list)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        _ = tool_choice, kwargs
        self.bound_tool_names.append([str(getattr(tool, "name", "") or "") for tool in tools])
        return self

    def _get_ls_params(self, stop=None, **kwargs):
        _ = stop, kwargs
        return {"ls_provider": "openai", "ls_model_name": "fake-ultra-model"}


def _settings(**overrides) -> RuntimeSettings:
    values = {
        "openai_base_url": "http://localhost:8001/v1",
        "openai_model": "deepseek_v4",
        "control_base_url": "http://control.test",
        "control_worker_token": "worker-secret",
    }
    values.update(overrides)
    return RuntimeSettings(**values)


def _context(
    *,
    mode: str = "search",
    goal: str = "Find my calibration note and add this result to the note.",
    notes: list[dict] | None = None,
    evaluation_profile: str = "",
    allow_append_proposal: bool = True,
    proposal_feature_enabled: bool = True,
) -> AgentRunContext:
    return AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="org-1",
        user_id="user-1",
        project_id="project-1",
        thread_id="thread-1",
        run_id="run-1",
        goal=goal,
        evaluation_profile=evaluation_profile,
        selection_context={
            "source": "chat",
            "note_access": {
                "mode": mode,
                "notes": notes or [],
                "allow_append_proposal": allow_append_proposal,
            },
        },
        run_metadata={"model_notes_proposals_enabled": proposal_feature_enabled},
        run_lease_worker_id="worker-1",
        run_lease_token="lease-secret",
    )


def _fake_httpx(monkeypatch, captured: dict, payload: dict) -> None:
    class FakeResponse:
        status_code = 200

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
            captured.setdefault("requests", []).append(
                {"url": url, "json": json, "headers": headers or {}}
            )
            return FakeResponse()

    monkeypatch.setattr("httpx.Client", FakeClient)


def test_note_access_parser_is_typed_bounded_and_fail_closed():
    scope = note_access_from_selection_context(
        {
            "note_access": {
                "mode": "SEARCH",
                "notes": [
                    {"note_id": "note-a", "revision": 4},
                    *({"note_id": f"note-{index}", "revision": index + 1} for index in range(19)),
                ],
                "attacker_mode": "all_users",
                "allow_append_proposal": True,
            },
            "notes": {"mode": "search_all"},
        }
    )

    assert scope.mode == "search"
    assert scope.notes[0].note_id == "note-a"
    assert scope.notes[0].revision == 4
    assert len(scope.notes) == 20
    assert scope.allow_append_proposal is True

    assert not note_access_from_selection_context({"note_access": {"mode": "all"}}).enabled
    assert not note_access_from_selection_context(
        {"note_access": {"mode": "selected", "notes": []}}
    ).enabled
    assert not note_access_from_selection_context(
        {
            "note_access": {
                "mode": "search",
                "notes": [],
                "allow_append_proposal": "yes",
            }
        }
    ).enabled
    assert not note_access_from_selection_context(
        {
            "note_access": {
                "mode": "search",
                "notes": [
                    {"note_id": "note-a", "revision": 4},
                    {"note_id": "note-a", "revision": 99},
                ],
            }
        }
    ).enabled
    assert not note_access_from_selection_context(
        {
            "note_access": {
                "mode": "search",
                "notes": [
                    {"note_id": f"note-{index}", "revision": index + 1} for index in range(21)
                ],
            }
        }
    ).enabled


def test_job_normalizes_note_scope_and_cleanroom_strips_it(monkeypatch, tmp_path: Path):
    import ultra_deepagents.evaluation_profiles as evaluation_profiles

    monkeypatch.setitem(
        evaluation_profiles._SUPPORTED_PROFILES,
        "test_cleanroom",
        EvaluationProfilePolicy(
            name="test_cleanroom",
            disabled_capabilities=("notes",),
            goal_only_messages=True,
            run_scoped_memory=True,
            run_scoped_workspace=True,
        ),
    )
    job = RunJobEnvelope.from_dict(
        {
            "run_id": "run-1",
            "thread_id": "thread-1",
            "user_id": "user-1",
            "goal": "Use the attached note.",
            "selection_context": {
                "source": "chat",
                "note_access": {
                    "mode": "selected",
                    "notes": [{"note_id": "note-a", "revision": 7, "title": "private"}],
                    "allow_append_proposal": True,
                },
            },
        }
    )
    assert job.selection_context["note_access"] == {
        "mode": "selected",
        "notes": [{"note_id": "note-a", "revision": 7}],
        "allow_append_proposal": True,
    }

    cleanroom = RunJobEnvelope.from_dict(
        {
            "run_id": "run-clean",
            "thread_id": "thread-clean",
            "user_id": "user-1",
            "goal": "Evaluate the goal only.",
            "evaluation_profile": "test_cleanroom",
            "selection_context": job.selection_context,
        }
    ).to_context(
        artifact_root=str(tmp_path / "artifacts"),
        workspace_root=str(tmp_path / "workspace"),
        run_lease_worker_id="worker-1",
        run_lease_token="lease-secret",
    )
    assert cleanroom.selection_context == {}


def test_search_notes_posts_exact_lease_headers_and_bounds_results(monkeypatch):
    captured: dict = {}
    _fake_httpx(
        monkeypatch,
        captured,
        {
            "notes": [
                {
                    "note_id": f"note-{index}",
                    "title": f"Calibration {index}",
                    "snippet": "Needle drift was 2 μm.",
                    "pinned": index == 0,
                    "revision": index + 1,
                    "updated_at": "2026-08-25T12:00:00Z",
                    "content_updated_at": "2026-08-24T12:00:00Z",
                }
                for index in range(20)
            ],
            "has_more": True,
            "next_cursor": "search-cursor-page-2",
        },
    )

    result = search_user_notes(
        _settings(),
        context=_context(),
        query="  needle drift  ",
        limit=999,
    )

    request = captured["requests"][0]
    assert request["url"] == "http://control.test/v2/runs/run-1/note-search"
    assert request["json"] == {
        "query": "needle drift",
        "sort": "relevance",
        "limit": 20,
    }
    assert request["headers"] == {
        "X-Ultra-Worker-Token": "worker-secret",
        "X-Ultra-Run-Id": "run-1",
        "X-Ultra-Worker-Id": "worker-1",
        "X-Ultra-Run-Lease-Token": "lease-secret",
    }
    assert result["ok"] is True
    assert result["result_count"] == 20
    assert result["has_more"] is True
    assert result["next_cursor"] == "search-cursor-page-2"
    assert result["content_trust"] == "untrusted_user_data"
    assert result["notes"][0]["pinned"] is True
    assert result["notes"][0]["content_updated_at"] == "2026-08-24T12:00:00Z"
    assert result["notes"][0]["updated_at"] == "2026-08-25T12:00:00Z"


def test_search_notes_recent_allows_empty_query_and_forwards_canonical_sort(monkeypatch):
    captured: dict = {}
    _fake_httpx(
        monkeypatch,
        captured,
        {
            "notes": [
                {
                    "note_id": "note-newest",
                    "title": "Latest observation",
                    "snippet": "The newest entry.",
                    "pinned": False,
                    "revision": 9,
                    "updated_at": "2026-08-27T12:00:00Z",
                    "content_updated_at": "2026-08-26T12:00:00Z",
                }
            ],
            "has_more": True,
            "next_cursor": "recent-cursor-page-2",
        },
    )

    result = search_user_notes(
        _settings(),
        context=_context(goal="What is my most recent Note?", allow_append_proposal=False),
        query="   ",
        sort=" RECENT ",
        limit=1,
    )

    assert captured["requests"][0]["json"] == {
        "query": "",
        "sort": "recent",
        "limit": 1,
    }
    assert result["ok"] is True
    assert result["notes"][0]["note_id"] == "note-newest"
    assert result["notes"][0]["content_updated_at"] == "2026-08-26T12:00:00Z"
    assert result["next_cursor"] == "recent-cursor-page-2"

    continued = search_user_notes(
        _settings(),
        context=_context(goal="What are my latest Notes?", allow_append_proposal=False),
        query="",
        sort="recent",
        cursor=result["next_cursor"],
        limit=1,
    )

    assert captured["requests"][1]["json"] == {
        "query": "",
        "sort": "recent",
        "limit": 1,
        "cursor": "recent-cursor-page-2",
    }
    assert continued["next_cursor"] == "recent-cursor-page-2"


def test_search_notes_rejects_malformed_sort_or_query_without_http(monkeypatch):
    def fail_client(*args, **kwargs):
        raise AssertionError("HTTP must not be called")

    monkeypatch.setattr("httpx.Client", fail_client)
    assert search_user_notes(_settings(), context=_context(), query="")["error"] == (
        "note_search_query_required"
    )
    assert (
        search_user_notes(_settings(), context=_context(), query="", sort="newest")["error"]
        == "note_search_sort_invalid"
    )
    assert (
        search_user_notes(
            _settings(), context=_context(), query={"private": "query"}, sort="recent"
        )["error"]
        == "note_search_query_invalid"
    )
    assert search_user_notes(_settings(), context=_context(), query="x" * 513)["error"] == (
        "note_search_query_too_long"
    )
    assert (
        search_user_notes(_settings(), context=_context(), query="x" * 513, sort="recent")["error"]
        == "note_search_query_too_long"
    )
    assert (
        search_user_notes(
            _settings(), context=_context(), query="calibration", cursor={"offset": 20}
        )["error"]
        == "invalid_note_search_cursor"
    )
    assert (
        search_user_notes(_settings(), context=_context(), query="calibration", cursor="x" * 2049)[
            "error"
        ]
        == "invalid_note_search_cursor"
    )


@pytest.mark.parametrize(
    "response",
    [
        {"notes": [], "has_more": True},
        {"notes": [], "has_more": False, "next_cursor": "unexpected-cursor"},
        {"notes": [], "has_more": "yes", "next_cursor": "cursor-page-2"},
        {"notes": [], "has_more": True, "next_cursor": {"offset": 20}},
        {
            "notes": [
                {
                    "note_id": "note-old-server",
                    "title": "Missing recency contract",
                    "snippet": "old shape",
                    "revision": 1,
                    "updated_at": "2026-08-27T12:00:00Z",
                }
            ],
            "has_more": False,
            "next_cursor": "",
        },
        {
            "notes": [
                {
                    "note_id": "note-malformed-pin",
                    "title": "Malformed",
                    "snippet": "bad pin type",
                    "pinned": "false",
                    "revision": 1,
                    "updated_at": "2026-08-27T12:00:00Z",
                    "content_updated_at": "2026-08-27T12:00:00Z",
                }
            ],
            "has_more": False,
            "next_cursor": "",
        },
    ],
)
def test_search_notes_rejects_malformed_pagination_response(monkeypatch, response):
    captured: dict = {}
    _fake_httpx(monkeypatch, captured, response)

    result = search_user_notes(_settings(), context=_context(), query="calibration")

    assert result == {"ok": False, "error": "invalid_notes_response", "notes": []}


def test_notes_agent_fake_model_searches_recent_then_reads_before_answering(monkeypatch):
    captured: list[dict] = []
    body = "The most recent observation was collected today."

    class FakeResponse:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, timeout):
            assert timeout == 30.0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json, headers=None):
            captured.append({"url": url, "json": json, "headers": headers or {}})
            if url.endswith("/note-search"):
                return FakeResponse(
                    {
                        "notes": [
                            {
                                "note_id": "note-newest",
                                "title": "Today's observation",
                                "snippet": "The most recent observation...",
                                "pinned": True,
                                "revision": 9,
                                "updated_at": "2026-08-27T12:00:00Z",
                                "content_updated_at": "2026-08-26T12:00:00Z",
                            }
                        ],
                        "has_more": False,
                    }
                )
            assert url.endswith("/note-read")
            return FakeResponse(
                {
                    "note_id": "note-newest",
                    "title": "Today's observation",
                    "revision": 9,
                    "content_digest": "sha256:" + "a" * 64,
                    "body_markdown": body,
                    "start_byte": 0,
                    "end_byte": len(body.encode("utf-8")),
                    "next_cursor": "",
                    "has_more": False,
                    "read_token": "read-token",
                    "updated_at": "2026-08-27T12:00:00Z",
                }
            )

    monkeypatch.setattr("httpx.Client", FakeClient)
    model = _ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_notes",
                        "args": {"sort": "recent", "limit": 1},
                        "id": "search-recent-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_note",
                        "args": {"note_id": "note-newest"},
                        "id": "read-newest-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Your most recent Note records today's observation."),
        ]
    )
    context = _context(
        goal="What is my most recent Note?",
        allow_append_proposal=False,
    )
    agent = build_research_agent(_settings(), model=model, context=context)

    result = agent.invoke(
        {"messages": [HumanMessage(content=context.goal)]},
        context=context,
    )

    assert [request["json"] for request in captured] == [
        {"query": "", "sort": "recent", "limit": 1},
        {"note_id": "note-newest", "max_chars": 8_000},
    ]
    assert model.bound_tool_names
    assert all(set(names) == {"search_notes", "read_note"} for names in model.bound_tool_names)
    assert result["messages"][-1].content == ("Your most recent Note records today's observation.")


def test_read_note_is_selected_scope_bound_and_validates_utf8_byte_range(monkeypatch):
    body = "Measured 2 μm after calibration."
    captured: dict = {}
    _fake_httpx(
        monkeypatch,
        captured,
        {
            "note_id": "note-a",
            "title": "Calibration log",
            "revision": 7,
            "content_digest": "sha256:" + "a" * 64,
            "body_markdown": body,
            "start_byte": 0,
            "end_byte": len(body.encode("utf-8")),
            "next_cursor": "cursor-next",
            "has_more": True,
            "read_token": "read-token",
            "updated_at": "2026-08-25T12:00:00Z",
        },
    )
    context = _context(mode="selected", notes=[{"note_id": "note-a", "revision": 7}])

    result = read_user_note(
        _settings(),
        context=context,
        note_id="note-a",
        cursor="cursor-current",
        max_chars=999_999,
    )

    assert captured["requests"][0]["json"] == {
        "note_id": "note-a",
        "cursor": "cursor-current",
        "max_chars": 16_000,
    }
    assert result["body_markdown"] == body
    assert result["returned_bytes"] == len(body.encode("utf-8"))
    assert result["read_token"] == "read-token"
    assert (
        read_user_note(_settings(), context=context, note_id="note-other")["error"]
        == "note_outside_selected_scope"
    )
    assert len(captured["requests"]) == 1


def test_read_note_rejects_response_without_required_read_capability(monkeypatch):
    body = "Bounded note content."
    captured: dict = {}
    _fake_httpx(
        monkeypatch,
        captured,
        {
            "note_id": "note-a",
            "title": "Calibration log",
            "revision": 7,
            "content_digest": "sha256:" + "a" * 64,
            "body_markdown": body,
            "start_byte": 0,
            "end_byte": len(body),
            "next_cursor": "",
            "has_more": False,
        },
    )
    context = _context(mode="selected", notes=[{"note_id": "note-a", "revision": 7}])

    result = read_user_note(_settings(), context=context, note_id="note-a")

    assert result == {"ok": False, "error": "invalid_notes_response"}


def test_proposal_uses_host_tool_call_id_for_idempotency_and_never_echoes_body(monkeypatch):
    captured: dict = {}
    _fake_httpx(
        monkeypatch,
        captured,
        {
            "proposal_id": "proposal-1",
            "note_id": "note-a",
            "base_revision": 7,
            "expires_at": "2026-08-25T12:10:00Z",
            "status": "pending",
            "body_markdown": "server-must-not-echo-this",
        },
    )
    context = _context(mode="selected", notes=[{"note_id": "note-a", "revision": 7}])
    addition = "## New result\n\nDrift was 2 μm."

    first = create_note_append_proposal(
        _settings(),
        context=context,
        note_id="note-a",
        expected_revision=7,
        body_markdown=addition,
        read_token="read-token",
        tool_call_id="tool-call-42",
    )
    second = create_note_append_proposal(
        _settings(),
        context=context,
        note_id="note-a",
        expected_revision=7,
        body_markdown=addition,
        read_token="read-token",
        tool_call_id="tool-call-42",
    )

    request = captured["requests"][0]["json"]
    assert request["body_markdown"] == addition
    assert request["read_token"] == "read-token"
    assert re.fullmatch(r"[0-9a-f]{64}", request["idempotency_key"])
    assert captured["requests"][1]["json"]["idempotency_key"] == request["idempotency_key"]
    assert "tool-call-42" not in request["idempotency_key"]
    assert first == second
    assert addition not in json.dumps(first)
    assert "server-must-not-echo-this" not in json.dumps(first)
    assert "idempotency" not in json.dumps(first)


def test_proposal_tool_injects_runtime_identity_outside_model_schema(monkeypatch):
    captured: dict = {}
    _fake_httpx(
        monkeypatch,
        captured,
        {
            "proposal_id": "proposal-2",
            "note_id": "note-a",
            "expected_revision": 7,
            "expires_at": "2026-08-25T12:10:00Z",
            "status": "pending",
        },
    )
    context = _context(mode="selected", notes=[{"note_id": "note-a", "revision": 7}])
    tools = build_notes_tools(_settings(), context=context)
    proposal_tool = next(tool for tool in tools if tool.name == "propose_note_append")
    runtime = ToolRuntime(
        state={},
        context=context,
        config={},
        stream_writer=lambda _: None,
        tool_call_id="host-call-9",
        store=None,
        tools=tools,
    )

    result = proposal_tool.invoke(
        {
            "name": "propose_note_append",
            "args": {
                "runtime": runtime,
                "note_id": "note-a",
                "expected_revision": 7,
                "body_markdown": "Exact proposed addition.",
                "read_token": "read-token",
            },
            "id": "host-call-9",
            "type": "tool_call",
        }
    )

    payload = json.loads(str(result.content))
    assert payload["proposal_id"] == "proposal-2"
    request = captured["requests"][0]["json"]
    expected_key = request["idempotency_key"]
    assert re.fullmatch(r"[0-9a-f]{64}", expected_key)
    assert "host-call-9" not in json.dumps(payload)
    schema_fields = proposal_tool.tool_call_schema.model_json_schema()["properties"]
    assert "runtime" not in schema_fields
    assert "tool_call_id" not in schema_fields
    assert "idempotency_key" not in schema_fields


def test_proposal_requires_host_identity_and_enforces_utf8_byte_limit(monkeypatch):
    def fail_client(*args, **kwargs):
        raise AssertionError("HTTP must not be called")

    monkeypatch.setattr("httpx.Client", fail_client)
    context = _context(mode="selected", notes=[{"note_id": "note-a", "revision": 7}])
    base = {
        "settings": _settings(),
        "context": context,
        "note_id": "note-a",
        "expected_revision": 7,
        "read_token": "read-token",
    }
    assert (
        create_note_append_proposal(
            **base,
            body_markdown="result",
            tool_call_id="",
        )["error"]
        == "proposal_identity_unavailable"
    )
    assert (
        create_note_append_proposal(
            **base,
            body_markdown="μ" * 16_385,
            tool_call_id="call-1",
        )["error"]
        == "note_append_body_too_large"
    )
    oversized_revision = {**base, "expected_revision": 1 << 63}
    assert (
        create_note_append_proposal(
            **oversized_revision,
            body_markdown="result",
            tool_call_id="call-2",
        )["error"]
        == "invalid_expected_revision"
    )


def test_notes_errors_are_content_free(monkeypatch):
    class BoomClient:
        def __init__(self, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json, headers=None):
            raise RuntimeError("private diagnosis and worker-secret")

    monkeypatch.setattr("httpx.Client", BoomClient)
    result = search_user_notes(_settings(), context=_context(), query="private diagnosis")
    assert result == {"ok": False, "error": "notes_service_unavailable", "notes": []}
    assert "private diagnosis" not in json.dumps(result)
    assert "worker-secret" not in json.dumps(result)


def test_notes_http_error_body_is_never_forwarded(monkeypatch):
    class DeniedClient:
        def __init__(self, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json, headers=None):
            return httpx.Response(
                403,
                json={
                    "error": "private diagnosis",
                    "body_markdown": "secret treatment details",
                },
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr("httpx.Client", DeniedClient)
    result = search_user_notes(_settings(), context=_context(), query="private diagnosis")
    assert result == {
        "ok": False,
        "error": "notes_access_denied",
        "status_code": 403,
        "notes": [],
    }
    assert "private diagnosis" not in json.dumps(result)
    assert "secret treatment" not in json.dumps(result)


def test_notes_http_error_retains_only_allowlisted_typed_code(monkeypatch):
    server_code = {"value": "note_read_required"}

    class ConflictClient:
        def __init__(self, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json, headers=None):
            return httpx.Response(
                503 if server_code["value"] == "note_search_timeout" else 409,
                json={
                    "code": server_code["value"],
                    "error": "private diagnosis must never be retained",
                    "body_markdown": "secret treatment details",
                },
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr("httpx.Client", ConflictClient)
    result = search_user_notes(_settings(), context=_context(), query="private diagnosis")
    assert result == {
        "ok": False,
        "error": "note_read_required",
        "status_code": 409,
        "notes": [],
    }
    assert "private diagnosis" not in json.dumps(result)
    assert "secret treatment" not in json.dumps(result)

    for allowed_code, status in (
        ("note_append_idempotency_conflict", 409),
        ("note_search_timeout", 503),
    ):
        server_code["value"] = allowed_code
        typed = search_user_notes(_settings(), context=_context(), query="private diagnosis")
        assert typed == {
            "ok": False,
            "error": allowed_code,
            "status_code": status,
            "notes": [],
        }

    server_code["value"] = "private_diagnosis"
    fallback = search_user_notes(_settings(), context=_context(), query="private diagnosis")
    assert fallback == {
        "ok": False,
        "error": "note_revision_conflict",
        "status_code": 409,
        "notes": [],
    }


def test_proposal_goal_gate_allows_concrete_requests_and_rejects_questions_or_negation():
    for goal in (
        "Add this result to my calibration note.",
        "Can you add this to my lab note?",
        "Could you add this result to my lab note?",
        "Please record today's measurements in my notes.",
        "Update my protocol note with the following observation.",
        "Write this to my lab log.",
        "Record this in my notebook.",
        "Jot this in my lab note.",
        "Find my calibration note and add today's result.",
        "Add this to the selected Note but don't search other Notes.",
        "Don't search other Notes, add this to the selected Note.",
        "Don’t search other Notes, add this to the selected Note.",
    ):
        assert note_append_proposal_goal_authorized(goal)

    for goal in (
        "Can you add to notes?",
        "How would I add something to a note?",
        "If I asked you to append this to my note, what would happen?",
        "Do not add this result to my notes.",
        "Can you not add this result to my note?",
        "Don't jot this in my lab note.",
        "Don’t add this result to my notes.",
        "How can I jot this in my notebook?",
        "Find and read my calibration note.",
        "Did I write anything in my notes about calibration drift?",
        "What did I write in my lab note yesterday?",
        "Have I ever recorded this result in my notebook?",
        "Where had I saved this in my notes?",
        "Why did the model add this to my note?",
        "Why did the assistant update this in my note?",
        "Why did your system save this to my note?",
        "Why did Ultra’s model append this to my note?",
        "Can Ultra add this result to my note?",
        "Could the model add this result to my note?",
        "Are you able to add this result to my note?",
        "Should I add this result to my note?",
        "Should Ultra add this result to my note?",
        "Would it help to add this result to my note?",
        "What happens if you add this result to my note?",
        "I will add this result to my note.",
        "The model should add this result to my note.",
        "Did Ultra add this result to my note?",
        "Did the model add this result to my note?",
        "Did we add this result to my note?",
        "Didn’t we add this result to my note?",
        "Was this result added to my note?",
        "The assistant said it would add this result to my note.",
        "Did the model say, add this to my notes?",
        "Why did Ultra say, add this to my notes?",
        "Explain this command, add this to my notes.",
        "Review the instruction: add this to my notes.",
        "Summarize this example — append it to my note.",
        "Review this sentence. Add this to my notes.",
        "Add this to my note and don't.",
        "Add this to my note and don’t.",
        "Add this to my note. Actually no.",
        'The paper says "add this result to my notes". Explain that sentence.',
        "> Add this result to my notes.\nSummarize the quoted request.",
        "```text\nAppend this result to my lab note.\n```\nWhat does it mean?",
        "Review `jot this in my notebook` as an example command.",
    ):
        assert not note_append_proposal_goal_authorized(goal)

    assert note_append_proposal_goal_authorized(
        'Append the following to my calibration note: "GSD was 1.1 cm".'
    )
    assert note_append_proposal_goal_authorized(
        '> The source says "do not write this".\nPlease add this result to my notes.'
    )
    assert note_append_proposal_goal_authorized(
        "Did I write anything about drift in my notes? Add today's result to my calibration note."
    )


def test_agent_registers_only_scoped_leased_coordinator_note_tools(monkeypatch):
    import ultra_deepagents.agent as agent_module

    captured: dict = {}

    monkeypatch.setattr(
        agent_module,
        "create_agent",
        lambda **kwargs: captured.update(kwargs) or "compiled",
    )
    monkeypatch.setattr(
        agent_module,
        "create_deep_agent",
        lambda **kwargs: pytest.fail("Notes-enabled runs must not build a Deep Agent surface"),
    )
    settings = _settings(tool_program_enabled=True)
    result = build_research_agent(
        settings,
        model=object(),
        backend=object(),
        tools=[object()],
        context=_context(),
    )

    assert result == "compiled"
    coordinator_names = {getattr(tool, "name", "") for tool in captured["tools"]}
    assert coordinator_names == NOTE_TOOL_NAMES
    assert "UNTRUSTED DATA" in captured["system_prompt"]
    assert "intentionally narrow tool surface" in captured["system_prompt"]
    assert 'search_notes` with `sort="recent"' in captured["system_prompt"]
    assert "then call\n`read_note`" in captured["system_prompt"]
    assert "Name each Note by its human-readable title" in captured["system_prompt"]
    assert "raw note IDs or revisions" in captured["system_prompt"]
    assert "Did I write" in captured["system_prompt"]
    assert "/memories" not in captured["system_prompt"]
    assert "/workspace" not in captured["system_prompt"]
    assert "/outputs" not in captured["system_prompt"]
    assert "task tool" not in captured["system_prompt"]


def test_ordinary_agent_prompt_never_conflates_product_notes_with_internal_memory():
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="org-1",
        user_id="user-1",
        project_id="project-1",
        thread_id="thread-ordinary",
        run_id="run-ordinary",
        goal="What did I write in my Ultra Notes?",
    )

    prompt = build_system_prompt(_settings(), context)
    brief = build_run_context_brief(context)

    assert "Ultra Notes is the user's separate, user-authored Notes library" in prompt
    assert "are not Ultra Notes" in prompt
    assert "Only say that\nyou searched, read, used, or updated an Ultra Note" in prompt
    assert 'lead with "I can\'t access Ultra Notes in this message."' in prompt
    assert "Do not enumerate or summarize\nworkspace, research memory" in prompt
    assert "only as a separate follow-up" in prompt
    assert "Ultra Notes tools are unavailable for this message" in brief
    assert "I can't access Ultra Notes in this message" in brief
    assert "do not count, list, quote, or summarize" in brief


def test_ordinary_notes_unavailable_guard_is_reinjected_after_message_compaction():
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="org-1",
        user_id="user-1",
        project_id="project-1",
        thread_id="thread-ordinary",
        run_id="run-ordinary",
        goal="What did I write in my Ultra Notes?",
    )
    middleware = UltraRunContextPromptMiddleware()

    def model_prompt_for(messages: list[HumanMessage]) -> str:
        captured: list[SystemMessage] = []

        def handler(request: ModelRequest) -> ModelResponse:
            if isinstance(request.system_message, SystemMessage):
                captured.append(request.system_message)
            return ModelResponse(result=[AIMessage(content="ok")])

        request = ModelRequest(
            model=cast(BaseChatModel, object()),
            messages=messages,
            system_message=SystemMessage(content="base prompt"),
            runtime=cast(Any, SimpleNamespace(context=context)),
        )
        middleware.wrap_model_call(request, handler)
        assert captured
        return str(captured[0].content)

    before = model_prompt_for([HumanMessage(content="What did I write in my Ultra Notes?")])
    after = model_prompt_for(
        [HumanMessage(content="Compacted conversation summary with prior messages replaced.")]
    )

    for prompt in (before, after):
        assert "I can't access Ultra Notes in this message" in prompt
        assert "do not count, list, quote, or summarize" in prompt


def test_notes_request_without_typed_scope_gets_a_zero_tool_fail_closed_agent(monkeypatch):
    import ultra_deepagents.agent as agent_module

    captured: dict = {}
    monkeypatch.setattr(
        agent_module,
        "create_agent",
        lambda **kwargs: captured.update(kwargs) or "compiled",
    )
    monkeypatch.setattr(
        agent_module,
        "create_deep_agent",
        lambda **kwargs: pytest.fail(
            "A Notes request without typed scope must not inherit ordinary corpora or tools"
        ),
    )
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="org-1",
        user_id="user-1",
        project_id="project-1",
        thread_id="thread-ordinary",
        run_id="run-ordinary",
        goal="What about my notes on NPH?",
    )

    assert build_research_agent(_settings(), model=object(), context=context) == "compiled"
    assert captured["tools"] == []
    assert "## Ultra Notes unavailable" in captured["system_prompt"]
    assert "/memories" not in captured["system_prompt"]
    assert "/workspace" not in captured["system_prompt"]
    assert "I can't access Ultra Notes" in captured["system_prompt"]
    dynamic_brief = captured["middleware"][0]._brief(
        SimpleNamespace(runtime=SimpleNamespace(context=context))
    )
    assert "Notes access for this run" not in dynamic_brief
    assert "Ultra Notes tools are unavailable" in dynamic_brief


@pytest.mark.parametrize(
    "goal",
    (
        "What about my notes on NPH?",
        "And what about my NPH notes?",
        "And my notes on NPH?",
        "And my NPH notes?",
        "What about NPH in my notes?",
        "What about the NPH notes I wrote?",
        "How about the note I wrote about calibration?",
        "Search my notes for NPH.",
        "Search Notes for NPH.",
        "Can you search my Notes?",
        "Read my Field Protocol note.",
        "Did I write anything in my most recent note?",
        "What do my Notes say about p53?",
    ),
)
def test_notes_no_scope_fallback_recognizes_high_confidence_content_requests(goal):
    import ultra_deepagents.agent as agent_module

    assert agent_module._goal_requests_product_notes_access(goal)


@pytest.mark.parametrize(
    "goal",
    (
        "Read my notes in this PDF.",
        "Use my notes in this PDF.",
        "What about my notes on NPH in this PDF?",
        "What about my notes below?",
        "Review my meeting notes.",
        "Search Notes in report.pdf.",
        "How can I search my notes?",
        "Can the model search my notes?",
        "Are you able to search my notes?",
        "Tell me whether you can search my notes.",
        "What about Ultra Notes?",
        "What about my Notes app?",
        "What about my notes privacy?",
        "What would happen if I asked what about my notes on NPH?",
        "Review the question: What about my notes on NPH?",
        "The assistant said it would search my notes.",
        "Did you search my notes?",
        'The prompt says, "search my notes for p53."',
        '"Search my notes for p53."',
        "> Search my notes for p53.",
        "```text\nSearch my notes for p53.\n```",
        "What about my notes on NPH? Actually, don't search.",
        "What about my notes on NPH, but don't access them.",
        "Don't search my notes.",
        "Answer without reading my notes.",
        "Search my notes. Stop searching.",
        "Search files, not my Notes.",
    ),
)
def test_notes_no_scope_fallback_leaves_non_content_or_contextual_requests_ordinary(goal):
    import ultra_deepagents.agent as agent_module

    assert not agent_module._goal_requests_product_notes_access(goal)


def test_contextual_notes_request_without_scope_keeps_the_ordinary_agent(monkeypatch):
    import ultra_deepagents.agent as agent_module

    captured: dict = {}
    monkeypatch.setattr(
        agent_module,
        "create_deep_agent",
        lambda **kwargs: captured.update(kwargs) or "compiled",
    )
    monkeypatch.setattr(
        agent_module,
        "create_agent",
        lambda **kwargs: pytest.fail("Document notes are not an Ultra Notes content run"),
    )
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="org-1",
        user_id="user-1",
        project_id="project-1",
        thread_id="thread-ordinary",
        run_id="run-ordinary",
        goal="Read my notes in this PDF.",
    )

    assert build_research_agent(_settings(), model=object(), backend=object(), context=context) == (
        "compiled"
    )
    assert "## Ultra Notes unavailable" not in captured["system_prompt"]


def test_notes_product_privacy_question_does_not_masquerade_as_content_access(monkeypatch):
    import ultra_deepagents.agent as agent_module

    captured: dict = {}
    monkeypatch.setattr(
        agent_module,
        "create_deep_agent",
        lambda **kwargs: captured.update(kwargs) or "compiled",
    )
    monkeypatch.setattr(
        agent_module,
        "create_agent",
        lambda **kwargs: pytest.fail("A product privacy question is not a Notes content run"),
    )
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="org-1",
        user_id="user-1",
        project_id="project-1",
        thread_id="thread-ordinary",
        run_id="run-ordinary",
        goal="Tell me whether my notes are private.",
    )

    assert build_research_agent(_settings(), model=object(), backend=object(), context=context) == (
        "compiled"
    )
    assert "Ultra Notes is the user's separate" in captured["system_prompt"]


def test_notes_agent_does_not_construct_a_durable_backend_or_inherit_caller_tools(
    monkeypatch,
    tmp_path,
):
    import ultra_deepagents.agent as agent_module

    captured: dict = {}
    monkeypatch.setattr(
        agent_module,
        "create_agent",
        lambda **kwargs: captured.update(kwargs) or "compiled",
    )
    monkeypatch.setattr(
        agent_module,
        "build_agent_backend",
        lambda *args, **kwargs: pytest.fail("Notes runs must not mount durable agent storage"),
    )

    caller_tool = type("CallerTool", (), {"name": "write_private_copy"})()
    result = build_research_agent(
        _settings(),
        model=object(),
        workspace_dir=tmp_path / "workspace",
        artifact_dir=tmp_path / "artifacts",
        tools=[caller_tool],
        context=_context(),
    )

    assert result == "compiled"
    assert {tool.name for tool in captured["tools"]} == NOTE_TOOL_NAMES
    assert all(
        middleware.__class__.__name__
        not in {
            "FilesystemMiddleware",
            "SubAgentMiddleware",
            "AsyncSubAgentMiddleware",
            "MemoryMiddleware",
        }
        for middleware in captured["middleware"]
    )


def test_selected_scope_omits_search_and_read_only_goal_omits_proposal():
    selected = build_notes_tools(
        _settings(),
        context=_context(
            mode="selected",
            notes=[{"note_id": "note-a", "revision": 2}],
            goal="Use this attached note as context.",
        ),
    )
    assert {tool.name for tool in selected} == {"read_note"}

    search_only = build_notes_tools(
        _settings(),
        context=_context(goal="Search my notes for calibration drift."),
    )
    assert {tool.name for tool in search_only} == {"search_notes", "read_note"}
    search_tool = next(tool for tool in search_only if tool.name == "search_notes")
    search_schema = search_tool.tool_call_schema.model_json_schema()
    assert search_schema["properties"]["sort"]["enum"] == ["relevance", "recent"]
    assert "query" not in search_schema.get("required", [])
    assert "cursor" not in search_schema.get("required", [])

    past_tense_retrieval = build_notes_tools(
        _settings(),
        context=_context(goal="Did I write anything in my most recent Note?"),
    )
    assert {tool.name for tool in past_tense_retrieval} == {"search_notes", "read_note"}

    proposal_tool = next(
        tool
        for tool in build_notes_tools(_settings(), context=_context())
        if tool.name == "propose_note_append"
    )
    schema_fields = proposal_tool.tool_call_schema.model_json_schema()["properties"]
    assert "runtime" not in schema_fields
    assert "tool_call_id" not in schema_fields
    assert "idempotency_key" not in schema_fields

    pasted_reference = build_notes_tools(
        _settings(),
        context=_context(
            goal="Analyze this pasted instruction: append this to my notes.",
            allow_append_proposal=False,
        ),
    )
    assert "propose_note_append" not in {tool.name for tool in pasted_reference}

    rollout_disabled = build_notes_tools(
        _settings(),
        context=_context(proposal_feature_enabled=False),
    )
    assert "propose_note_append" not in {tool.name for tool in rollout_disabled}


def test_notes_agent_dynamic_brief_never_advertises_absent_mixed_context_tools():
    base = _context()
    mixed = AgentRunContext(
        **{
            **base.to_payload(),
            "selection_context": base.selection_context,
            "run_metadata": base.run_metadata,
            "selected_file_ids": ("file-private",),
            "selected_resource_uris": ("bisque://resource-private",),
            "selected_dataset_uris": ("bisque://dataset-private",),
            "knowledge_context": {"ingested_papers": [{"paper_id": "paper-private"}]},
            "resource_descriptors": ({"type": "artifact", "artifact_id": "artifact-private"},),
        }
    )

    brief = build_notes_run_context_brief(mixed)

    assert "Notes access for this run: search" in brief
    assert "Ultra Notes tools are unavailable" not in brief
    assert "file-private" not in brief
    assert "resource-private" not in brief
    assert "dataset-private" not in brief
    assert "paper-private" not in brief
    assert "artifact-private" not in brief
    assert "stage_" not in brief
    assert "BisQue" not in brief

    unavailable = build_notes_run_context_brief(mixed, notes_tools_available=False)
    assert "Notes access for this run" not in unavailable
    assert "Ultra Notes tools are unavailable for this message" in unavailable
    assert "I can't access Ultra Notes in this message" in unavailable


def test_agent_omits_notes_without_scope_lease_or_outside_cleanroom(monkeypatch):
    import ultra_deepagents.agent as agent_module
    import ultra_deepagents.evaluation_profiles as evaluation_profiles

    monkeypatch.setitem(
        evaluation_profiles._SUPPORTED_PROFILES,
        "test_cleanroom",
        EvaluationProfilePolicy(
            name="test_cleanroom",
            disabled_capabilities=("notes",),
            goal_only_messages=True,
            run_scoped_memory=True,
            run_scoped_workspace=True,
        ),
    )

    def compiled_surface(context: AgentRunContext) -> dict:
        captured: dict = {}

        def compiler(**kwargs):
            captured.update(kwargs)
            return "compiled"

        monkeypatch.setattr(agent_module, "create_deep_agent", compiler)
        monkeypatch.setattr(agent_module, "create_agent", compiler)
        build_research_agent(_settings(), model=object(), backend=object(), context=context)
        return captured

    def tool_names(surface: dict) -> set[str]:
        return {getattr(tool, "name", "") for tool in surface["tools"]}

    without_scope = AgentRunContext(
        assistant_id="a",
        org_id="o",
        user_id="u",
        project_id="p",
        thread_id="t",
        run_id="r",
        run_lease_worker_id="worker-1",
        run_lease_token="lease-secret",
    )
    assert not (NOTE_TOOL_NAMES & tool_names(compiled_surface(without_scope)))
    missing_lease = AgentRunContext(
        **{
            **_context().to_payload(),
            "selection_context": _context().selection_context,
        }
    )
    for degraded_context in (
        missing_lease,
        _context(evaluation_profile="test_cleanroom"),
    ):
        surface = compiled_surface(degraded_context)
        assert not (NOTE_TOOL_NAMES & tool_names(surface))
        assert "## Ultra Notes unavailable" in surface["system_prompt"]
        assert "I can't access Ultra Notes" in surface["system_prompt"]
        notes_middleware = surface["middleware"][0]
        dynamic_brief = notes_middleware._brief(
            SimpleNamespace(runtime=SimpleNamespace(context=degraded_context))
        )
        assert "Notes access for this run" not in dynamic_brief
        assert "Ultra Notes tools are unavailable" in dynamic_brief


def test_async_delegation_strips_note_scope_and_lease_authority():
    payload = async_subagent_context_payload(_context(), subagent_name="remote-worker")
    assert payload["selection_context"] == {"source": "chat"}
    assert "run_lease_worker_id" not in payload
    assert "run_lease_token" not in payload


def test_notes_tool_events_redact_private_inputs_and_outputs():
    context = _context()
    cases = [
        (
            {
                "method": "tools",
                "params": {
                    "namespace": [],
                    "data": {
                        "event": "started",
                        "tool_call_id": "call-search",
                        "tool_name": "search_notes",
                        "input": {
                            "query": "private diagnosis",
                            "sort": "recent",
                            "cursor": "search-cursor-secret",
                            "limit": 8,
                        },
                    },
                },
            },
            {"tool_name", "status", "tool_call_id"},
        ),
        (
            {
                "method": "tools",
                "params": {
                    "namespace": [],
                    "data": {
                        "event": "completed",
                        "tool_call_id": "call-search",
                        "tool_name": "search_notes",
                        "output": json.dumps(
                            {
                                "ok": True,
                                "result_count": 1,
                                "has_more": True,
                                "next_cursor": "next-search-cursor-secret",
                                "notes": [
                                    {
                                        "note_id": "note-private",
                                        "title": "Private diagnosis",
                                        "snippet": "Secret treatment details",
                                    }
                                ],
                            }
                        ),
                    },
                },
            },
            {"tool_name", "status", "tool_call_id", "ok", "result_count", "has_more"},
        ),
        (
            {
                "method": "tools",
                "params": {
                    "namespace": [],
                    "data": {
                        "event": "completed",
                        "tool_call_id": "call-read",
                        "tool_name": "read_note",
                        "output": json.dumps(
                            {
                                "ok": True,
                                "note_id": "note-private",
                                "title": "Private diagnosis",
                                "revision": 4,
                                "content_digest": "sha256:" + "a" * 64,
                                "body_markdown": "Secret treatment details",
                                "returned_bytes": 24,
                                "has_more": True,
                                "read_token": "read-secret",
                                "next_cursor": "cursor-secret",
                            }
                        ),
                    },
                },
            },
            {
                "tool_name",
                "status",
                "tool_call_id",
                "ok",
                "note_id",
                "revision",
                "returned_bytes",
                "has_more",
            },
        ),
        (
            {
                "method": "tools",
                "params": {
                    "namespace": [],
                    "data": {
                        "event": "started",
                        "tool_call_id": "call-proposal",
                        "tool_name": "propose_note_append",
                        "input": {
                            "note_id": "note-private",
                            "expected_revision": 4,
                            "body_markdown": "Secret addition",
                            "read_token": "read-secret",
                            "idempotency_key": "idempotency-secret",
                        },
                    },
                },
            },
            {"tool_name", "status", "tool_call_id", "note_id", "expected_revision"},
        ),
        (
            {
                "method": "tools",
                "params": {
                    "namespace": [],
                    "data": {
                        "event": "completed",
                        "tool_call_id": "call-proposal",
                        "tool_name": "propose_note_append",
                        "output": json.dumps(
                            {
                                "ok": True,
                                "proposal_id": "proposal-1",
                                "note_id": "note-private",
                                "expected_revision": 4,
                                "expires_at": "2026-08-25T12:10:00Z",
                                "status": "pending",
                                "message": "Secret addition was proposed",
                            }
                        ),
                    },
                },
            },
            {
                "tool_name",
                "status",
                "tool_call_id",
                "ok",
                "proposal_id",
                "note_id",
                "expected_revision",
                "expires_at",
                "proposal_status",
            },
        ),
    ]

    serialized_events: list[str] = []
    for stream_event, expected_keys in cases:
        event = _tool_event_from_stream_event(context, stream_event, {})
        assert event is not None
        assert set(event["payload"]) == expected_keys
        serialized_events.append(json.dumps(event))

    serialized = "\n".join(serialized_events)
    for private_value in (
        "private diagnosis",
        "recent",
        "Private diagnosis",
        "Secret treatment details",
        "read-secret",
        "cursor-secret",
        "search-cursor-secret",
        "next-search-cursor-secret",
        "Secret addition",
        "idempotency-secret",
        "sha256:" + "a" * 64,
    ):
        assert private_value not in serialized
    assert "input" not in serialized
    assert "output_preview" not in serialized


def test_legacy_notes_tool_events_and_failures_are_redacted():
    context = _context()
    completed = _tool_event_from_stream_event(
        context,
        {
            "event": "on_tool_end",
            "name": "read_note",
            "run_id": "call-read",
            "data": {
                "output": json.dumps(
                    {
                        "note_id": "note-a",
                        "revision": 3,
                        "content_digest": "a" * 64,
                        "body_markdown": "private body",
                        "returned_bytes": 12,
                        "has_more": False,
                    }
                )
            },
        },
        {},
    )
    failed = _tool_event_from_stream_event(
        context,
        {
            "event": "on_tool_error",
            "name": "read_note",
            "run_id": "call-read",
            "data": {"error": "private body and worker-secret"},
        },
        {},
    )

    assert completed is not None
    assert completed["payload"] == {
        "tool_name": "read_note",
        "status": "completed",
        "tool_call_id": "call-read",
        "note_id": "note-a",
        "revision": 3,
        "returned_bytes": 12,
        "has_more": False,
    }
    assert failed is not None
    assert failed["payload"] == {
        "tool_name": "read_note",
        "status": "failed",
        "tool_call_id": "call-read",
        "error": "notes_tool_failed",
    }
    assert "private body" not in json.dumps((completed, failed))
    assert "worker-secret" not in json.dumps((completed, failed))


def test_notes_enabled_run_redacts_every_non_notes_tool_event():
    sentinel = "NOTE_SENTINEL_COPIED_INTO_EXECUTE"
    started = _tool_event_from_stream_event(
        _context(),
        {
            "event": "on_tool_start",
            "name": "execute",
            "run_id": "call-execute",
            "data": {"input": {"command": f"printf {sentinel}"}},
        },
        {},
    )
    completed = _tool_event_from_stream_event(
        _context(),
        {
            "event": "on_tool_end",
            "name": "write_file",
            "run_id": "call-write",
            "data": {"output": f"wrote {sentinel} to /outputs/private.txt"},
        },
        {},
    )

    assert started is not None
    assert completed is not None
    assert started["payload"] == {
        "tool_name": "execute",
        "status": "started",
        "redacted": True,
        "tool_call_id": "call-execute",
    }
    assert completed["payload"] == {
        "tool_name": "write_file",
        "status": "completed",
        "redacted": True,
        "tool_call_id": "call-write",
    }
    assert sentinel not in json.dumps([started, completed])


def test_notes_enabled_run_redacts_unexpected_subagent_delta():
    sentinel = "NOTE_SENTINEL_COPIED_INTO_SUBAGENT"
    event = _subagent_message_event_from_stream_event(
        _context(),
        {
            "type": "event",
            "method": "messages",
            "params": {
                "namespace": ["general-purpose"],
                "data": [
                    {
                        "event": "content-block-delta",
                        "index": 0,
                        "delta": {"type": "text-delta", "text": sentinel},
                    },
                    {"lc_agent_name": "general-purpose"},
                ],
            },
        },
    )

    assert event is not None
    assert event["message"] is None
    assert event["payload"] == {"source": "general-purpose", "redacted": True}
    assert sentinel not in json.dumps(event)
