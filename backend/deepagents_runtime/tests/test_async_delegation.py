import ast
import asyncio
import json
from pathlib import Path

import pytest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from ultra_deepagents.async_delegation import (
    UltraAsyncSubagentContextMiddleware,
    async_subagent_context_payload,
)
from ultra_deepagents.context import AgentRunContext


def test_async_delegation_middleware_uses_public_sdk_instead_of_private_deepagents_helpers():
    source_path = Path(__file__).parents[1] / "src" / "ultra_deepagents" / "async_delegation.py"
    tree = ast.parse(source_path.read_text())

    private_deepagents_imports = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != "deepagents.middleware.async_subagents":
            continue
        private_deepagents_imports.extend(
            alias.name for alias in node.names if alias.name.startswith("_")
        )

    assert private_deepagents_imports == []


def test_async_delegation_middleware_rejects_case_insensitive_duplicate_subagent_names():
    with pytest.raises(ValueError, match="Duplicate async subagent name"):
        UltraAsyncSubagentContextMiddleware(
            [
                {
                    "name": "remote-training-runner",
                    "description": "Runs long model training jobs.",
                    "graph_id": "ultra-training-agent",
                },
                {
                    "name": "Remote-Training-Runner",
                    "description": "Runs another remote job.",
                    "graph_id": "ultra-other-agent",
                },
            ]
        )


@pytest.mark.parametrize(
    "name",
    ["", 123, None],
)
def test_async_delegation_middleware_rejects_invalid_subagent_names(name):
    spec = {
        "name": name,
        "description": "Runs long model training jobs.",
        "graph_id": "ultra-training-agent",
        "url": "https://langgraph.example.test",
    }
    with pytest.raises(ValueError, match="name"):
        UltraAsyncSubagentContextMiddleware([spec])


def test_async_delegation_middleware_rejects_missing_subagent_name():
    with pytest.raises(ValueError, match="name"):
        UltraAsyncSubagentContextMiddleware(
            [
                {
                    "description": "Runs long model training jobs.",
                    "graph_id": "ultra-training-agent",
                    "url": "https://langgraph.example.test",
                },
            ]
        )


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/langgraph.sock",
        "langgraph.example.test",
        "ftp://langgraph.example.test",
    ],
)
def test_async_delegation_middleware_rejects_non_http_subagent_urls(url):
    with pytest.raises(ValueError, match=r"url.*http:// or https://"):
        UltraAsyncSubagentContextMiddleware(
            [
                {
                    "name": "remote-training-runner",
                    "description": "Runs long model training jobs.",
                    "graph_id": "ultra-training-agent",
                    "url": url,
                },
            ]
        )


@pytest.mark.parametrize(
    "url",
    [
        "https://runner-token@langgraph.example.test",
        "https://ultra:secret@langgraph.example.test",
    ],
)
def test_async_delegation_middleware_rejects_subagent_urls_with_credentials(url):
    with pytest.raises(ValueError, match=r"url.*credentials"):
        UltraAsyncSubagentContextMiddleware(
            [
                {
                    "name": "remote-training-runner",
                    "description": "Runs long model training jobs.",
                    "graph_id": "ultra-training-agent",
                    "url": url,
                },
            ]
        )


def test_async_delegation_middleware_rejects_case_insensitive_duplicate_headers():
    with pytest.raises(ValueError, match="Duplicate async subagent header name"):
        UltraAsyncSubagentContextMiddleware(
            [
                {
                    "name": "remote-training-runner",
                    "description": "Runs long model training jobs.",
                    "graph_id": "ultra-training-agent",
                    "url": "https://langgraph.example.test",
                    "headers": {
                        "X-Auth-Scheme": "ultra-workos",
                        "x-auth-scheme": "langsmith",
                    },
                },
            ]
        )


@pytest.mark.parametrize(
    "headers",
    [
        ["Authorization: Bearer token-ref"],
        {123: "token-ref"},
        {"": "token-ref"},
        {"Authorization": ""},
        {"Authorization": 123},
        {"Authorization": None},
    ],
)
def test_async_delegation_middleware_rejects_invalid_headers(headers):
    with pytest.raises(ValueError, match="header"):
        UltraAsyncSubagentContextMiddleware(
            [
                {
                    "name": "remote-training-runner",
                    "description": "Runs long model training jobs.",
                    "graph_id": "ultra-training-agent",
                    "url": "https://langgraph.example.test",
                    "headers": headers,
                },
            ]
        )


@pytest.mark.parametrize(
    "graph_id",
    ["", 123, None],
)
def test_async_delegation_middleware_rejects_invalid_graph_id(graph_id):
    spec = {
        "name": "remote-training-runner",
        "description": "Runs long model training jobs.",
        "graph_id": graph_id,
        "url": "https://langgraph.example.test",
    }
    with pytest.raises(ValueError, match="graph_id"):
        UltraAsyncSubagentContextMiddleware([spec])


def test_async_delegation_middleware_rejects_missing_graph_id():
    with pytest.raises(ValueError, match="graph_id"):
        UltraAsyncSubagentContextMiddleware(
            [
                {
                    "name": "remote-training-runner",
                    "description": "Runs long model training jobs.",
                    "url": "https://langgraph.example.test",
                },
            ]
        )


def test_update_async_task_returns_tool_message_for_retired_subagent_type():
    middleware = _middleware()
    request = _stale_task_request(
        "update_async_task",
        args={
            "task_id": "stale-task",
            "message": "Continue the remote run with new constraints.",
        },
    )

    def unreachable_handler(request):
        raise AssertionError("stale async task should be handled before fallback")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-update-async-task-stale"
    assert "Unknown async subagent type 'retired-runner'" in str(result.content)
    assert "remote-training-runner" in str(result.content)


def test_aupdate_async_task_returns_tool_message_for_retired_subagent_type():
    middleware = _middleware()
    request = _stale_task_request(
        "update_async_task",
        args={
            "task_id": "stale-task",
            "message": "Continue the remote run with new constraints.",
        },
    )

    async def unreachable_handler(request):
        raise AssertionError("stale async task should be handled before fallback")

    result = asyncio.run(middleware.awrap_tool_call(request, unreachable_handler))

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-update-async-task-stale"
    assert "Unknown async subagent type 'retired-runner'" in str(result.content)
    assert "remote-training-runner" in str(result.content)


def test_start_async_task_requires_ultra_run_context():
    middleware = _middleware()
    request = _start_task_request_without_context()

    def unreachable_handler(request):
        raise AssertionError("context-free async launch should be blocked before fallback")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-start-no-context"
    assert "AgentRunContext is required" in str(result.content)


@pytest.mark.parametrize("description", ["", "   ", None])
def test_start_async_task_rejects_blank_description_before_remote_call(
    monkeypatch,
    description,
):
    created_threads: list[dict[str, object]] = []
    created_runs: list[dict[str, object]] = []

    class FakeThreads:
        def create(self, **kwargs):
            created_threads.append(kwargs)
            return {"thread_id": "async-thread-1"}

    class FakeRuns:
        def create(self, **kwargs):
            created_runs.append(kwargs)
            return {"run_id": "async-run-1"}

    class FakeClient:
        threads = FakeThreads()
        runs = FakeRuns()

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_sync_client",
        lambda **kwargs: FakeClient(),
    )
    middleware = _middleware()
    request = _start_task_request()
    args = {"subagent_type": "remote-training-runner"}
    if description is not None:
        args["description"] = description
    request = ToolCallRequest(
        tool_call={
            "name": "start_async_task",
            "args": args,
            "id": "call-start-async-task-blank-description",
        },
        tool=request.tool,
        state=request.state,
        runtime=request.runtime,
    )

    def unreachable_handler(request):
        raise AssertionError("Ultra should handle async launch validation")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-start-async-task-blank-description"
    assert "description is required" in str(result.content)
    assert created_threads == []
    assert created_runs == []


@pytest.mark.parametrize("description", ["", "   ", None])
def test_astart_async_task_rejects_blank_description_before_remote_call(
    monkeypatch,
    description,
):
    created_threads: list[dict[str, object]] = []
    created_runs: list[dict[str, object]] = []

    class FakeThreads:
        async def create(self, **kwargs):
            created_threads.append(kwargs)
            return {"thread_id": "async-thread-1"}

    class FakeRuns:
        async def create(self, **kwargs):
            created_runs.append(kwargs)
            return {"run_id": "async-run-1"}

    class FakeClient:
        threads = FakeThreads()
        runs = FakeRuns()

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_client",
        lambda **kwargs: FakeClient(),
    )
    middleware = _middleware()
    request = _start_task_request()
    args = {"subagent_type": "remote-training-runner"}
    if description is not None:
        args["description"] = description
    request = ToolCallRequest(
        tool_call={
            "name": "start_async_task",
            "args": args,
            "id": "call-start-async-task-blank-description",
        },
        tool=request.tool,
        state=request.state,
        runtime=request.runtime,
    )

    async def unreachable_handler(request):
        raise AssertionError("Ultra should handle async launch validation")

    result = asyncio.run(middleware.awrap_tool_call(request, unreachable_handler))

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-start-async-task-blank-description"
    assert "description is required" in str(result.content)
    assert created_threads == []
    assert created_runs == []


def test_start_async_task_creates_remote_thread_with_ultra_metadata(monkeypatch):
    class FakeThreads:
        def __init__(self) -> None:
            self.created: list[dict[str, object]] = []

        def create(self, **kwargs):
            self.created.append(kwargs)
            return {"thread_id": "async-thread-1"}

    class FakeRuns:
        def create(self, **kwargs):
            return {"run_id": "async-run-1"}

    class FakeClient:
        def __init__(self) -> None:
            self.threads = FakeThreads()
            self.runs = FakeRuns()

    clients: list[FakeClient] = []

    def fake_get_sync_client(**kwargs):
        if not clients:
            clients.append(FakeClient())
        return clients[0]

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_sync_client",
        fake_get_sync_client,
    )
    middleware = _middleware()
    request = _start_task_request()

    def unreachable_handler(request):
        raise AssertionError("Ultra should handle async launch with scoped context")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, Command)
    assert clients[0].threads.created == [
        {
            "metadata": {
                "ultra_delegation": "async_subagent",
                "ultra_parent_run_id": "run-async",
                "ultra_parent_thread_id": "thread-async",
                "ultra_subagent_name": "remote-training-runner",
                "ultra_org_id": "local-org",
                "ultra_user_id": "researcher-1",
                "ultra_project_id": "local-project",
                "ultra_subagent_graph_id": "ultra-training-agent",
            },
            "graph_id": "ultra-training-agent",
        }
    ]


def test_start_async_task_creates_remote_run_with_ultra_tenant_metadata(monkeypatch):
    class FakeThreads:
        def create(self, **kwargs):
            return {"thread_id": "async-thread-1"}

    class FakeRuns:
        def __init__(self) -> None:
            self.created: list[dict[str, object]] = []

        def create(self, **kwargs):
            self.created.append(kwargs)
            return {"run_id": "async-run-1"}

    class FakeClient:
        def __init__(self) -> None:
            self.threads = FakeThreads()
            self.runs = FakeRuns()

    clients: list[FakeClient] = []

    def fake_get_sync_client(**kwargs):
        if not clients:
            clients.append(FakeClient())
        return clients[0]

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_sync_client",
        fake_get_sync_client,
    )
    middleware = _middleware()
    request = _start_task_request()

    def unreachable_handler(request):
        raise AssertionError("Ultra should handle async launch with scoped context")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, Command)
    assert clients[0].runs.created[0]["metadata"] == {
        "ultra_delegation": "async_subagent",
        "ultra_parent_run_id": "run-async",
        "ultra_parent_thread_id": "thread-async",
        "ultra_subagent_name": "remote-training-runner",
        "ultra_org_id": "local-org",
        "ultra_user_id": "researcher-1",
        "ultra_project_id": "local-project",
        "ultra_subagent_graph_id": "ultra-training-agent",
    }


def test_async_subagent_headers_respect_operator_auth_scheme_case_insensitively(
    monkeypatch,
):
    class FakeThreads:
        def create(self, **kwargs):
            return {"thread_id": "async-thread-1"}

    class FakeRuns:
        def create(self, **kwargs):
            return {"run_id": "async-run-1"}

    class FakeClient:
        threads = FakeThreads()
        runs = FakeRuns()

    captured_clients: list[dict[str, object]] = []

    def fake_get_sync_client(**kwargs):
        captured_clients.append(kwargs)
        return FakeClient()

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_sync_client",
        fake_get_sync_client,
    )
    middleware = UltraAsyncSubagentContextMiddleware(
        [
            {
                "name": "remote-training-runner",
                "description": "Runs long model training jobs.",
                "graph_id": "ultra-training-agent",
                "url": "http://agent-protocol.test",
                "headers": {
                    "X-Auth-Scheme": "ultra-workos",
                    "Authorization": "Bearer opaque-token-ref",
                },
            }
        ]
    )
    request = _start_task_request()

    def unreachable_handler(request):
        raise AssertionError("Ultra should handle async launch with scoped context")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, Command)
    assert captured_clients == [
        {
            "url": "http://agent-protocol.test",
            "headers": {
                "X-Auth-Scheme": "ultra-workos",
                "Authorization": "Bearer opaque-token-ref",
            },
        }
    ]


def test_start_async_task_persists_failed_launch_after_thread_created(monkeypatch):
    class FakeThreads:
        def create(self, **kwargs):
            return {"thread_id": "async-thread-1"}

    class FakeRuns:
        def create(self, **kwargs):
            raise RuntimeError("503 Service Unavailable")

    class FakeClient:
        threads = FakeThreads()
        runs = FakeRuns()

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_sync_client",
        lambda **kwargs: FakeClient(),
    )
    middleware = _middleware()
    request = _start_task_request()

    def unreachable_handler(request):
        raise AssertionError("Ultra should handle async launch with scoped context")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, Command)
    message = result.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert "Failed to launch async subagent 'remote-training-runner'" in str(message.content)
    assert "task_id: async-thread-1" in str(message.content)
    task = result.update["async_tasks"]["async-thread-1"]
    assert task["task_id"] == "async-thread-1"
    assert task["agent_name"] == "remote-training-runner"
    assert task["thread_id"] == "async-thread-1"
    assert task["status"] == "error"
    assert task["last_error"] == "503 Service Unavailable"


def test_start_async_task_persists_malformed_run_response_after_thread_created(monkeypatch):
    class FakeThreads:
        def create(self, **kwargs):
            return {"thread_id": "async-thread-1"}

    class FakeRuns:
        def create(self, **kwargs):
            return {}

    class FakeClient:
        threads = FakeThreads()
        runs = FakeRuns()

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_sync_client",
        lambda **kwargs: FakeClient(),
    )
    middleware = _middleware()
    request = _start_task_request()

    def unreachable_handler(request):
        raise AssertionError("Ultra should handle async launch with scoped context")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, Command)
    message = result.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert "Failed to launch async subagent 'remote-training-runner'" in str(message.content)
    assert "missing run_id" in str(message.content)
    task = result.update["async_tasks"]["async-thread-1"]
    assert task["task_id"] == "async-thread-1"
    assert task["agent_name"] == "remote-training-runner"
    assert task["thread_id"] == "async-thread-1"
    assert task["run_id"] == "__ultra_async_launch_failed__"
    assert task["status"] == "error"
    assert "missing run_id" in task["last_error"]


def test_astart_async_task_persists_malformed_run_response_after_thread_created(monkeypatch):
    class FakeThreads:
        async def create(self, **kwargs):
            return {"thread_id": "async-thread-1"}

    class FakeRuns:
        async def create(self, **kwargs):
            return {}

    class FakeClient:
        threads = FakeThreads()
        runs = FakeRuns()

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_client",
        lambda **kwargs: FakeClient(),
    )
    middleware = _middleware()
    request = _start_task_request()

    async def unreachable_handler(request):
        raise AssertionError("Ultra should handle async launch with scoped context")

    result = asyncio.run(middleware.awrap_tool_call(request, unreachable_handler))

    assert isinstance(result, Command)
    message = result.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert "Failed to launch async subagent 'remote-training-runner'" in str(message.content)
    assert "missing run_id" in str(message.content)
    task = result.update["async_tasks"]["async-thread-1"]
    assert task["run_id"] == "__ultra_async_launch_failed__"
    assert task["status"] == "error"
    assert "missing run_id" in task["last_error"]


def test_start_async_task_rejects_malformed_thread_response_before_task_state(
    monkeypatch,
):
    class FakeThreads:
        def create(self, **kwargs):
            return {"thread_id": ""}

    class FakeRuns:
        def create(self, **kwargs):
            return {"run_id": "async-run-1"}

    class FakeClient:
        threads = FakeThreads()
        runs = FakeRuns()

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_sync_client",
        lambda **kwargs: FakeClient(),
    )
    middleware = _middleware()
    request = _start_task_request()

    def unreachable_handler(request):
        raise AssertionError("Ultra should handle async launch with scoped context")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-start-async-task"
    assert "Failed to launch async subagent 'remote-training-runner'" in str(result.content)
    assert "missing thread_id" in str(result.content)


def test_astart_async_task_rejects_malformed_thread_response_before_task_state(
    monkeypatch,
):
    class FakeThreads:
        async def create(self, **kwargs):
            return {"thread_id": ""}

    class FakeRuns:
        async def create(self, **kwargs):
            return {"run_id": "async-run-1"}

    class FakeClient:
        threads = FakeThreads()
        runs = FakeRuns()

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_client",
        lambda **kwargs: FakeClient(),
    )
    middleware = _middleware()
    request = _start_task_request()

    async def unreachable_handler(request):
        raise AssertionError("Ultra should handle async launch with scoped context")

    result = asyncio.run(middleware.awrap_tool_call(request, unreachable_handler))

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-start-async-task"
    assert "Failed to launch async subagent 'remote-training-runner'" in str(result.content)
    assert "missing thread_id" in str(result.content)


def test_check_async_task_reports_persisted_failed_launch_without_remote_call():
    middleware = _middleware()
    request = _failed_launch_task_request("check_async_task")

    def unreachable_handler(request):
        raise AssertionError("failed launch task should not call remote status")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-check-async-task-failed-launch"
    assert "async-thread-1" in str(result.content)
    assert "failed before a remote run was created" in str(result.content)
    assert "503 Service Unavailable" in str(result.content)


def test_check_async_task_reports_cached_terminal_status_without_remote_call():
    middleware = _middleware()
    request = _terminal_task_request(
        "check_async_task",
        task_id="failed-thread-1",
        run_id="failed-run-1",
        status="failed",
        last_error="training worker exhausted retry budget",
    )

    def unreachable_handler(request):
        raise AssertionError("terminal cached check should not call remote status")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-check-async-task-terminal"
    assert "task_id: failed-thread-1" in str(result.content)
    assert "agent: remote-training-runner" in str(result.content)
    assert "status: failed" in str(result.content)
    assert "error: training worker exhausted retry budget" in str(result.content)


def test_acheck_async_task_reports_cached_terminal_status_without_remote_call():
    middleware = _middleware()
    request = _terminal_task_request(
        "check_async_task",
        task_id="succeeded-thread-1",
        run_id="succeeded-run-1",
        status="succeeded",
    )

    async def unreachable_handler(request):
        raise AssertionError("terminal cached check should not call remote status")

    result = asyncio.run(middleware.awrap_tool_call(request, unreachable_handler))

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-check-async-task-terminal"
    assert "task_id: succeeded-thread-1" in str(result.content)
    assert "agent: remote-training-runner" in str(result.content)
    assert "status: succeeded" in str(result.content)


def test_cancel_async_task_reports_cached_terminal_status_without_remote_call():
    middleware = _middleware()
    request = _terminal_task_request(
        "cancel_async_task",
        task_id="canceled-thread-1",
        run_id="canceled-run-1",
        status="canceled",
    )

    def unreachable_handler(request):
        raise AssertionError("terminal cached cancel should not call remote status")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-cancel-async-task-terminal"
    assert "task_id: canceled-thread-1" in str(result.content)
    assert "agent: remote-training-runner" in str(result.content)
    assert "status: canceled" in str(result.content)


def test_acancel_async_task_reports_cached_terminal_status_without_remote_call():
    middleware = _middleware()
    request = _terminal_task_request(
        "cancel_async_task",
        task_id="failed-thread-1",
        run_id="failed-run-1",
        status="failed",
        last_error="training worker exhausted retry budget",
    )

    async def unreachable_handler(request):
        raise AssertionError("terminal cached cancel should not call remote status")

    result = asyncio.run(middleware.awrap_tool_call(request, unreachable_handler))

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-cancel-async-task-terminal"
    assert "task_id: failed-thread-1" in str(result.content)
    assert "agent: remote-training-runner" in str(result.content)
    assert "status: failed" in str(result.content)
    assert "error: training worker exhausted retry budget" in str(result.content)


def test_update_async_task_reports_cached_terminal_status_without_remote_call(monkeypatch):
    class FakeRuns:
        def create(self, **kwargs):
            raise AssertionError("terminal cached update should not create remote run")

    class FakeClient:
        runs = FakeRuns()

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_sync_client",
        lambda **kwargs: FakeClient(),
    )
    middleware = _middleware()
    request = _terminal_task_request(
        "update_async_task",
        task_id="succeeded-thread-1",
        run_id="succeeded-run-1",
        status="succeeded",
    )

    def unreachable_handler(request):
        raise AssertionError("Ultra handles update_async_task directly")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-update-async-task-terminal"
    assert "task_id: succeeded-thread-1" in str(result.content)
    assert "agent: remote-training-runner" in str(result.content)
    assert "status: succeeded" in str(result.content)


def test_aupdate_async_task_reports_cached_terminal_status_without_remote_call(monkeypatch):
    class FakeRuns:
        async def create(self, **kwargs):
            raise AssertionError("terminal cached update should not create remote run")

    class FakeClient:
        runs = FakeRuns()

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_client",
        lambda **kwargs: FakeClient(),
    )
    middleware = _middleware()
    request = _terminal_task_request(
        "update_async_task",
        task_id="failed-thread-1",
        run_id="failed-run-1",
        status="failed",
        last_error="training worker exhausted retry budget",
    )

    async def unreachable_handler(request):
        raise AssertionError("Ultra handles update_async_task directly")

    result = asyncio.run(middleware.awrap_tool_call(request, unreachable_handler))

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-update-async-task-terminal"
    assert "task_id: failed-thread-1" in str(result.content)
    assert "agent: remote-training-runner" in str(result.content)
    assert "status: failed" in str(result.content)
    assert "error: training worker exhausted retry budget" in str(result.content)


@pytest.mark.parametrize("message", ["", "   ", None])
def test_update_async_task_rejects_blank_message_before_remote_call(monkeypatch, message):
    created_runs: list[dict[str, object]] = []

    class FakeRuns:
        def create(self, **kwargs):
            created_runs.append(kwargs)
            return {"run_id": "async-run-2"}

    class FakeClient:
        runs = FakeRuns()

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_sync_client",
        lambda **kwargs: FakeClient(),
    )
    middleware = _middleware()
    args = {"task_id": "async-thread-1"}
    if message is not None:
        args["message"] = message
    request = _tracked_task_request("update_async_task", args=args)

    def unreachable_handler(request):
        raise AssertionError("Ultra should handle async update validation")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-update-async-task"
    assert "message is required" in str(result.content)
    assert created_runs == []


@pytest.mark.parametrize("message", ["", "   ", None])
def test_aupdate_async_task_rejects_blank_message_before_remote_call(monkeypatch, message):
    created_runs: list[dict[str, object]] = []

    class FakeRuns:
        async def create(self, **kwargs):
            created_runs.append(kwargs)
            return {"run_id": "async-run-2"}

    class FakeClient:
        runs = FakeRuns()

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_client",
        lambda **kwargs: FakeClient(),
    )
    middleware = _middleware()
    args = {"task_id": "async-thread-1"}
    if message is not None:
        args["message"] = message
    request = _tracked_task_request("update_async_task", args=args)

    async def unreachable_handler(request):
        raise AssertionError("Ultra should handle async update validation")

    result = asyncio.run(middleware.awrap_tool_call(request, unreachable_handler))

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-update-async-task"
    assert "message is required" in str(result.content)
    assert created_runs == []


def test_list_async_tasks_reports_persisted_failed_launch_error_without_remote_call():
    middleware = _middleware()
    request = _failed_launch_task_request(
        "list_async_tasks",
        args={"status_filter": "all"},
    )

    def unreachable_handler(request):
        raise AssertionError("failed launch list should not call remote status")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-list-async-tasks-failed-launch"
    assert "1 tracked task(s):" in str(result.content)
    assert "task_id: async-thread-1" in str(result.content)
    assert "agent: remote-training-runner" in str(result.content)
    assert "status: error" in str(result.content)
    assert "503 Service Unavailable" in str(result.content)


def test_alist_async_tasks_reports_persisted_failed_launch_error_without_remote_call():
    middleware = _middleware()
    request = _failed_launch_task_request(
        "list_async_tasks",
        args={"status_filter": "all"},
    )

    async def unreachable_handler(request):
        raise AssertionError("failed launch list should not call remote status")

    result = asyncio.run(middleware.awrap_tool_call(request, unreachable_handler))

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-list-async-tasks-failed-launch"
    assert "1 tracked task(s):" in str(result.content)
    assert "task_id: async-thread-1" in str(result.content)
    assert "agent: remote-training-runner" in str(result.content)
    assert "status: error" in str(result.content)
    assert "503 Service Unavailable" in str(result.content)


def test_list_async_tasks_reports_mixed_terminal_cached_tasks_without_remote_call():
    middleware = _middleware()
    request = _mixed_terminal_task_request("list_async_tasks")

    def unreachable_handler(request):
        raise AssertionError("terminal cached list should not call remote status")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-list-async-tasks-mixed-terminal"
    assert "2 tracked task(s):" in str(result.content)
    assert "task_id: async-thread-1" in str(result.content)
    assert "status: error" in str(result.content)
    assert "503 Service Unavailable" in str(result.content)
    assert "task_id: cancelled-thread-1" in str(result.content)
    assert "status: cancelled" in str(result.content)


def test_alist_async_tasks_reports_mixed_terminal_cached_tasks_without_remote_call():
    middleware = _middleware()
    request = _mixed_terminal_task_request("list_async_tasks")

    async def unreachable_handler(request):
        raise AssertionError("terminal cached list should not call remote status")

    result = asyncio.run(middleware.awrap_tool_call(request, unreachable_handler))

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-list-async-tasks-mixed-terminal"
    assert "2 tracked task(s):" in str(result.content)
    assert "task_id: async-thread-1" in str(result.content)
    assert "status: error" in str(result.content)
    assert "503 Service Unavailable" in str(result.content)
    assert "task_id: cancelled-thread-1" in str(result.content)
    assert "status: cancelled" in str(result.content)


def test_list_async_tasks_treats_agent_protocol_terminal_aliases_as_cached(
    monkeypatch,
):
    class FakeRuns:
        def get(self, **kwargs):
            raise AssertionError("terminal cached list should not call remote status")

    class FakeClient:
        runs = FakeRuns()

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_sync_client",
        lambda **kwargs: FakeClient(),
    )
    middleware = _middleware()
    request = _agent_protocol_terminal_task_request("list_async_tasks")

    def unreachable_handler(request):
        raise AssertionError("terminal cached list should not call fallback handler")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-list-async-tasks-agent-protocol-terminal"
    assert "3 tracked task(s):" in str(result.content)
    assert "task_id: succeeded-thread-1" in str(result.content)
    assert "status: succeeded" in str(result.content)
    assert "task_id: failed-thread-1" in str(result.content)
    assert "status: failed" in str(result.content)
    assert "error: training worker exhausted retry budget" in str(result.content)
    assert "task_id: canceled-thread-1" in str(result.content)
    assert "status: canceled" in str(result.content)


def test_alist_async_tasks_treats_agent_protocol_terminal_aliases_as_cached(
    monkeypatch,
):
    class FakeRuns:
        async def get(self, **kwargs):
            raise AssertionError("terminal cached list should not call remote status")

    class FakeClient:
        runs = FakeRuns()

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_client",
        lambda **kwargs: FakeClient(),
    )
    middleware = _middleware()
    request = _agent_protocol_terminal_task_request("list_async_tasks")

    async def unreachable_handler(request):
        raise AssertionError("terminal cached list should not call fallback handler")

    result = asyncio.run(middleware.awrap_tool_call(request, unreachable_handler))

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-list-async-tasks-agent-protocol-terminal"
    assert "3 tracked task(s):" in str(result.content)
    assert "status: succeeded" in str(result.content)
    assert "status: failed" in str(result.content)
    assert "error: training worker exhausted retry budget" in str(result.content)
    assert "status: canceled" in str(result.content)


def test_update_async_task_persists_update_failure_without_marking_task_failed(monkeypatch):
    class FakeRuns:
        def create(self, **kwargs):
            raise RuntimeError("409 update conflict")

    class FakeClient:
        runs = FakeRuns()

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_sync_client",
        lambda **kwargs: FakeClient(),
    )
    middleware = _middleware()
    request = _tracked_task_request(
        "update_async_task",
        args={
            "task_id": "async-thread-1",
            "message": "Use the corrected learning rate.",
        },
    )

    def unreachable_handler(request):
        raise AssertionError("Ultra should handle async update with scoped context")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, Command)
    message = result.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert "Failed to update async subagent" in str(message.content)
    assert "task_id: async-thread-1" in str(message.content)
    task = result.update["async_tasks"]["async-thread-1"]
    assert task["task_id"] == "async-thread-1"
    assert task["agent_name"] == "remote-training-runner"
    assert task["thread_id"] == "async-thread-1"
    assert task["run_id"] == "async-run-1"
    assert task["status"] == "running"
    assert task["created_at"] == "2026-06-10T00:00:00Z"
    assert task["last_checked_at"] == "2026-06-10T00:01:00Z"
    assert task["last_error"] == "409 update conflict"


def test_aupdate_async_task_persists_update_failure_without_marking_task_failed(monkeypatch):
    class FakeRuns:
        async def create(self, **kwargs):
            raise RuntimeError("409 async update conflict")

    class FakeClient:
        runs = FakeRuns()

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_client",
        lambda **kwargs: FakeClient(),
    )
    middleware = _middleware()
    request = _tracked_task_request(
        "update_async_task",
        args={
            "task_id": "async-thread-1",
            "message": "Use the corrected learning rate.",
        },
    )

    async def unreachable_handler(request):
        raise AssertionError("Ultra should handle async update with scoped context")

    result = asyncio.run(middleware.awrap_tool_call(request, unreachable_handler))

    assert isinstance(result, Command)
    message = result.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert "Failed to update async subagent" in str(message.content)
    assert "task_id: async-thread-1" in str(message.content)
    task = result.update["async_tasks"]["async-thread-1"]
    assert task["run_id"] == "async-run-1"
    assert task["status"] == "running"
    assert task["created_at"] == "2026-06-10T00:00:00Z"
    assert task["last_checked_at"] == "2026-06-10T00:01:00Z"
    assert task["last_error"] == "409 async update conflict"


def test_update_async_task_preserves_remote_thread_id_when_task_id_differs(monkeypatch):
    created_runs: list[dict[str, object]] = []

    class FakeRuns:
        def create(self, **kwargs):
            created_runs.append(kwargs)
            return {"run_id": "async-run-2"}

    class FakeClient:
        runs = FakeRuns()

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_sync_client",
        lambda **kwargs: FakeClient(),
    )
    middleware = _middleware()
    request = _tracked_task_request(
        "update_async_task",
        args={
            "task_id": "async-task-alias-1",
            "message": "Use the corrected learning rate.",
        },
        task_id="async-task-alias-1",
        thread_id="remote-thread-1",
    )

    def unreachable_handler(request):
        raise AssertionError("Ultra should handle async update with scoped context")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, Command)
    assert created_runs[0]["thread_id"] == "remote-thread-1"
    task = result.update["async_tasks"]["async-task-alias-1"]
    assert task["task_id"] == "async-task-alias-1"
    assert task["thread_id"] == "remote-thread-1"
    assert task["run_id"] == "async-run-2"
    assert task["created_at"] == "2026-06-10T00:00:00Z"
    assert task["last_checked_at"] == "2026-06-10T00:01:00Z"


def test_aupdate_async_task_preserves_remote_thread_id_when_task_id_differs(monkeypatch):
    created_runs: list[dict[str, object]] = []

    class FakeRuns:
        async def create(self, **kwargs):
            created_runs.append(kwargs)
            return {"run_id": "async-run-2"}

    class FakeClient:
        runs = FakeRuns()

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_client",
        lambda **kwargs: FakeClient(),
    )
    middleware = _middleware()
    request = _tracked_task_request(
        "update_async_task",
        args={
            "task_id": "async-task-alias-1",
            "message": "Use the corrected learning rate.",
        },
        task_id="async-task-alias-1",
        thread_id="remote-thread-1",
    )

    async def unreachable_handler(request):
        raise AssertionError("Ultra should handle async update with scoped context")

    result = asyncio.run(middleware.awrap_tool_call(request, unreachable_handler))

    assert isinstance(result, Command)
    assert created_runs[0]["thread_id"] == "remote-thread-1"
    task = result.update["async_tasks"]["async-task-alias-1"]
    assert task["task_id"] == "async-task-alias-1"
    assert task["thread_id"] == "remote-thread-1"
    assert task["run_id"] == "async-run-2"
    assert task["created_at"] == "2026-06-10T00:00:00Z"
    assert task["last_checked_at"] == "2026-06-10T00:01:00Z"


def test_update_async_task_persists_malformed_run_response_without_marking_task_failed(
    monkeypatch,
):
    class FakeRuns:
        def create(self, **kwargs):
            return {}

    class FakeClient:
        runs = FakeRuns()

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_sync_client",
        lambda **kwargs: FakeClient(),
    )
    middleware = _middleware()
    request = _tracked_task_request(
        "update_async_task",
        args={
            "task_id": "async-thread-1",
            "message": "Use the corrected learning rate.",
        },
    )

    def unreachable_handler(request):
        raise AssertionError("Ultra should handle async update with scoped context")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, Command)
    message = result.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert "Failed to update async subagent" in str(message.content)
    assert "missing run_id" in str(message.content)
    task = result.update["async_tasks"]["async-thread-1"]
    assert task["run_id"] == "async-run-1"
    assert task["status"] == "running"
    assert task["created_at"] == "2026-06-10T00:00:00Z"
    assert task["last_checked_at"] == "2026-06-10T00:01:00Z"
    assert "missing run_id" in task["last_error"]


def test_aupdate_async_task_persists_malformed_run_response_without_marking_task_failed(
    monkeypatch,
):
    class FakeRuns:
        async def create(self, **kwargs):
            return {}

    class FakeClient:
        runs = FakeRuns()

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_client",
        lambda **kwargs: FakeClient(),
    )
    middleware = _middleware()
    request = _tracked_task_request(
        "update_async_task",
        args={
            "task_id": "async-thread-1",
            "message": "Use the corrected learning rate.",
        },
    )

    async def unreachable_handler(request):
        raise AssertionError("Ultra should handle async update with scoped context")

    result = asyncio.run(middleware.awrap_tool_call(request, unreachable_handler))

    assert isinstance(result, Command)
    message = result.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert "Failed to update async subagent" in str(message.content)
    assert "missing run_id" in str(message.content)
    task = result.update["async_tasks"]["async-thread-1"]
    assert task["run_id"] == "async-run-1"
    assert task["status"] == "running"
    assert task["created_at"] == "2026-06-10T00:00:00Z"
    assert task["last_checked_at"] == "2026-06-10T00:01:00Z"
    assert "missing run_id" in task["last_error"]


def test_check_async_task_persists_status_failure_without_marking_task_failed():
    middleware = _middleware()
    request = _tracked_task_request("check_async_task")

    def failed_handler(request):
        return ToolMessage(
            "Failed to get run status: 502 Bad Gateway",
            tool_call_id=request.tool_call["id"],
        )

    result = middleware.wrap_tool_call(request, failed_handler)

    assert isinstance(result, Command)
    message = result.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert "Failed to get run status" in str(message.content)
    assert "task_id: async-thread-1" in str(message.content)
    task = result.update["async_tasks"]["async-thread-1"]
    assert task["task_id"] == "async-thread-1"
    assert task["agent_name"] == "remote-training-runner"
    assert task["thread_id"] == "async-thread-1"
    assert task["run_id"] == "async-run-1"
    assert task["status"] == "running"
    assert task["created_at"] == "2026-06-10T00:00:00Z"
    assert task["last_checked_at"] == "2026-06-10T00:01:00Z"
    assert task["last_error"] == "502 Bad Gateway"


def test_acheck_async_task_persists_status_failure_without_marking_task_failed():
    middleware = _middleware()
    request = _tracked_task_request("check_async_task")

    async def failed_handler(request):
        return ToolMessage(
            "Failed to get run status: 502 Bad Gateway",
            tool_call_id=request.tool_call["id"],
        )

    result = asyncio.run(middleware.awrap_tool_call(request, failed_handler))

    assert isinstance(result, Command)
    message = result.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert "Failed to get run status" in str(message.content)
    assert "task_id: async-thread-1" in str(message.content)
    task = result.update["async_tasks"]["async-thread-1"]
    assert task["run_id"] == "async-run-1"
    assert task["status"] == "running"
    assert task["created_at"] == "2026-06-10T00:00:00Z"
    assert task["last_checked_at"] == "2026-06-10T00:01:00Z"
    assert task["last_error"] == "502 Bad Gateway"


def test_check_async_task_persists_remote_run_error_detail():
    middleware = _middleware()
    request = _tracked_task_request("check_async_task")
    errored_task = dict(request.state["async_tasks"]["async-thread-1"])
    errored_task["status"] = "error"

    def failed_run_handler(request):
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        '{"status": "error", "error": "CUDA out of memory", "thread_id": "async-thread-1"}',
                        tool_call_id=request.tool_call["id"],
                    )
                ],
                "async_tasks": {"async-thread-1": errored_task},
            }
        )

    result = middleware.wrap_tool_call(request, failed_run_handler)

    assert isinstance(result, Command)
    message = result.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert "CUDA out of memory" in str(message.content)
    task = result.update["async_tasks"]["async-thread-1"]
    assert task["status"] == "error"
    assert task["last_error"] == "CUDA out of memory"


def test_acheck_async_task_persists_remote_run_error_detail():
    middleware = _middleware()
    request = _tracked_task_request("check_async_task")
    errored_task = dict(request.state["async_tasks"]["async-thread-1"])
    errored_task["status"] = "error"

    async def failed_run_handler(request):
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        '{"status": "error", "error": "CUDA out of memory", "thread_id": "async-thread-1"}',
                        tool_call_id=request.tool_call["id"],
                    )
                ],
                "async_tasks": {"async-thread-1": errored_task},
            }
        )

    result = asyncio.run(middleware.awrap_tool_call(request, failed_run_handler))

    assert isinstance(result, Command)
    task = result.update["async_tasks"]["async-thread-1"]
    assert task["status"] == "error"
    assert task["last_error"] == "CUDA out of memory"


def test_cancel_async_task_persists_cancel_failure_without_marking_task_cancelled():
    middleware = _middleware()
    request = _tracked_task_request("cancel_async_task")

    def failed_handler(request):
        return ToolMessage(
            "Failed to cancel run: 503 Service Unavailable",
            tool_call_id=request.tool_call["id"],
        )

    result = middleware.wrap_tool_call(request, failed_handler)

    assert isinstance(result, Command)
    message = result.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert "Failed to cancel run" in str(message.content)
    assert "task_id: async-thread-1" in str(message.content)
    task = result.update["async_tasks"]["async-thread-1"]
    assert task["task_id"] == "async-thread-1"
    assert task["agent_name"] == "remote-training-runner"
    assert task["thread_id"] == "async-thread-1"
    assert task["run_id"] == "async-run-1"
    assert task["status"] == "running"
    assert task["created_at"] == "2026-06-10T00:00:00Z"
    assert task["last_checked_at"] == "2026-06-10T00:01:00Z"
    assert task["last_error"] == "503 Service Unavailable"


def test_acancel_async_task_persists_cancel_failure_without_marking_task_cancelled():
    middleware = _middleware()
    request = _tracked_task_request("cancel_async_task")

    async def failed_handler(request):
        return ToolMessage(
            "Failed to cancel run: 503 Service Unavailable",
            tool_call_id=request.tool_call["id"],
        )

    result = asyncio.run(middleware.awrap_tool_call(request, failed_handler))

    assert isinstance(result, Command)
    message = result.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert "Failed to cancel run" in str(message.content)
    assert "task_id: async-thread-1" in str(message.content)
    task = result.update["async_tasks"]["async-thread-1"]
    assert task["run_id"] == "async-run-1"
    assert task["status"] == "running"
    assert task["created_at"] == "2026-06-10T00:00:00Z"
    assert task["last_checked_at"] == "2026-06-10T00:01:00Z"
    assert task["last_error"] == "503 Service Unavailable"


def test_list_async_tasks_persists_status_fetch_failure_without_hiding_outage(monkeypatch):
    class FakeRuns:
        def get(self, **kwargs):
            assert kwargs == {"thread_id": "async-thread-1", "run_id": "async-run-1"}
            raise RuntimeError("502 Bad Gateway")

    class FakeClient:
        runs = FakeRuns()

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_sync_client",
        lambda **kwargs: FakeClient(),
    )
    middleware = _middleware()
    request = _tracked_task_request("list_async_tasks", args={"status_filter": "all"})

    def unreachable_handler(request):
        raise AssertionError("Ultra should handle async task list status checks")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, Command)
    message = result.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert "task_id: async-thread-1" in str(message.content)
    assert "status: running" in str(message.content)
    assert "error: 502 Bad Gateway" in str(message.content)
    task = result.update["async_tasks"]["async-thread-1"]
    assert task["task_id"] == "async-thread-1"
    assert task["agent_name"] == "remote-training-runner"
    assert task["thread_id"] == "async-thread-1"
    assert task["run_id"] == "async-run-1"
    assert task["status"] == "running"
    assert task["created_at"] == "2026-06-10T00:00:00Z"
    assert task["last_checked_at"] == "2026-06-10T00:01:00Z"
    assert task["last_error"] == "502 Bad Gateway"


def test_list_async_tasks_persists_remote_run_error_detail(monkeypatch):
    class FakeRuns:
        def get(self, **kwargs):
            assert kwargs == {"thread_id": "async-thread-1", "run_id": "async-run-1"}
            return {"status": "error", "error": "CUDA out of memory"}

    class FakeClient:
        runs = FakeRuns()

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_sync_client",
        lambda **kwargs: FakeClient(),
    )
    middleware = _middleware()
    request = _tracked_task_request("list_async_tasks", args={"status_filter": "all"})

    def unreachable_handler(request):
        raise AssertionError("Ultra should handle async task list status checks")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, Command)
    message = result.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert "task_id: async-thread-1" in str(message.content)
    assert "status: error" in str(message.content)
    assert "error: CUDA out of memory" in str(message.content)
    task = result.update["async_tasks"]["async-thread-1"]
    assert task["status"] == "error"
    assert task["created_at"] == "2026-06-10T00:00:00Z"
    assert task["last_error"] == "CUDA out of memory"


def test_list_async_tasks_preserves_failed_alias_error_in_mixed_list(monkeypatch):
    calls = []

    class FakeRuns:
        def get(self, **kwargs):
            calls.append(kwargs)
            assert kwargs == {"thread_id": "async-thread-1", "run_id": "async-run-1"}
            return {"status": "running"}

    class FakeClient:
        runs = FakeRuns()

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_sync_client",
        lambda **kwargs: FakeClient(),
    )
    middleware = _middleware()
    request = _mixed_failed_alias_and_running_task_request("list_async_tasks")

    def unreachable_handler(request):
        raise AssertionError("Ultra should handle async task list status checks")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, Command)
    assert calls == [{"thread_id": "async-thread-1", "run_id": "async-run-1"}]
    message = result.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert "task_id: failed-thread-1" in str(message.content)
    assert "status: failed" in str(message.content)
    assert "error: training worker exhausted retry budget" in str(message.content)
    failed_task = result.update["async_tasks"]["failed-thread-1"]
    assert failed_task["status"] == "failed"
    assert failed_task["last_error"] == "training worker exhausted retry budget"
    running_task = result.update["async_tasks"]["async-thread-1"]
    assert running_task["status"] == "running"


def test_list_async_tasks_clears_stale_error_after_successful_status_fetch(monkeypatch):
    class FakeRuns:
        def get(self, **kwargs):
            assert kwargs == {"thread_id": "async-thread-1", "run_id": "async-run-1"}
            return {"status": "running"}

    class FakeClient:
        runs = FakeRuns()

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_sync_client",
        lambda **kwargs: FakeClient(),
    )
    middleware = _middleware()
    request = _failed_launch_task_request(
        "list_async_tasks",
        args={"status_filter": "all"},
    )
    task = dict(request.state["async_tasks"]["async-thread-1"])
    task["run_id"] = "async-run-1"
    task["status"] = "running"
    task["last_error"] = "502 Bad Gateway"
    request.state["async_tasks"]["async-thread-1"] = task

    def unreachable_handler(request):
        raise AssertionError("Ultra should handle async task list status checks")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, Command)
    task = result.update["async_tasks"]["async-thread-1"]
    assert task["status"] == "running"
    assert "last_error" not in task


def test_list_async_tasks_clears_stale_error_when_remote_run_succeeds(monkeypatch):
    class FakeRuns:
        def get(self, **kwargs):
            assert kwargs == {"thread_id": "async-thread-1", "run_id": "async-run-1"}
            return {"status": "success"}

    class FakeClient:
        runs = FakeRuns()

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_sync_client",
        lambda **kwargs: FakeClient(),
    )
    middleware = _middleware()
    request = _failed_launch_task_request(
        "list_async_tasks",
        args={"status_filter": "all"},
    )
    task = dict(request.state["async_tasks"]["async-thread-1"])
    task["run_id"] = "async-run-1"
    task["status"] = "running"
    task["last_error"] = "502 Bad Gateway"
    request.state["async_tasks"]["async-thread-1"] = task

    def unreachable_handler(request):
        raise AssertionError("Ultra should handle async task list status checks")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, Command)
    task = result.update["async_tasks"]["async-thread-1"]
    assert task["status"] == "success"
    assert "last_error" not in task


def test_alist_async_tasks_persists_status_fetch_failure_without_hiding_outage(monkeypatch):
    class FakeRuns:
        async def get(self, **kwargs):
            assert kwargs == {"thread_id": "async-thread-1", "run_id": "async-run-1"}
            raise RuntimeError("502 Bad Gateway")

    class FakeClient:
        runs = FakeRuns()

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_client",
        lambda **kwargs: FakeClient(),
    )
    middleware = _middleware()
    request = _tracked_task_request("list_async_tasks", args={"status_filter": "all"})

    async def unreachable_handler(request):
        raise AssertionError("Ultra should handle async task list status checks")

    result = asyncio.run(middleware.awrap_tool_call(request, unreachable_handler))

    assert isinstance(result, Command)
    message = result.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert "task_id: async-thread-1" in str(message.content)
    assert "status: running" in str(message.content)
    assert "error: 502 Bad Gateway" in str(message.content)
    task = result.update["async_tasks"]["async-thread-1"]
    assert task["run_id"] == "async-run-1"
    assert task["status"] == "running"
    assert task["created_at"] == "2026-06-10T00:00:00Z"
    assert task["last_checked_at"] == "2026-06-10T00:01:00Z"
    assert task["last_error"] == "502 Bad Gateway"


def test_alist_async_tasks_persists_remote_run_error_detail(monkeypatch):
    class FakeRuns:
        async def get(self, **kwargs):
            assert kwargs == {"thread_id": "async-thread-1", "run_id": "async-run-1"}
            return {"status": "error", "error": "CUDA out of memory"}

    class FakeClient:
        runs = FakeRuns()

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_client",
        lambda **kwargs: FakeClient(),
    )
    middleware = _middleware()
    request = _tracked_task_request("list_async_tasks", args={"status_filter": "all"})

    async def unreachable_handler(request):
        raise AssertionError("Ultra should handle async task list status checks")

    result = asyncio.run(middleware.awrap_tool_call(request, unreachable_handler))

    assert isinstance(result, Command)
    message = result.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert "status: error" in str(message.content)
    assert "error: CUDA out of memory" in str(message.content)
    task = result.update["async_tasks"]["async-thread-1"]
    assert task["status"] == "error"
    assert task["last_error"] == "CUDA out of memory"


def test_alist_async_tasks_preserves_failed_alias_error_in_mixed_list(monkeypatch):
    calls = []

    class FakeRuns:
        async def get(self, **kwargs):
            calls.append(kwargs)
            assert kwargs == {"thread_id": "async-thread-1", "run_id": "async-run-1"}
            return {"status": "running"}

    class FakeClient:
        runs = FakeRuns()

    monkeypatch.setattr(
        "ultra_deepagents.async_delegation.get_client",
        lambda **kwargs: FakeClient(),
    )
    middleware = _middleware()
    request = _mixed_failed_alias_and_running_task_request("list_async_tasks")

    async def unreachable_handler(request):
        raise AssertionError("Ultra should handle async task list status checks")

    result = asyncio.run(middleware.awrap_tool_call(request, unreachable_handler))

    assert isinstance(result, Command)
    assert calls == [{"thread_id": "async-thread-1", "run_id": "async-run-1"}]
    message = result.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert "task_id: failed-thread-1" in str(message.content)
    assert "status: failed" in str(message.content)
    assert "error: training worker exhausted retry budget" in str(message.content)
    failed_task = result.update["async_tasks"]["failed-thread-1"]
    assert failed_task["status"] == "failed"
    assert failed_task["last_error"] == "training worker exhausted retry budget"


def test_astart_async_task_requires_ultra_run_context():
    middleware = _middleware()
    request = _start_task_request_without_context()

    async def unreachable_handler(request):
        raise AssertionError("context-free async launch should be blocked before fallback")

    result = asyncio.run(middleware.awrap_tool_call(request, unreachable_handler))

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-start-no-context"
    assert "AgentRunContext is required" in str(result.content)


def test_async_subagent_context_payload_strips_parent_local_artifact_paths():
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="researcher-1",
        project_id="local-project",
        thread_id="thread-async",
        run_id="run-async",
        goal="Launch async work.",
        resource_descriptors=(
            {
                "type": "artifact",
                "artifact_id": "artifact-1",
                "run_id": "prior-run",
                "kind": "table",
                "title": "Prior result",
                "path": "/home/scientist/ultra/artifacts/prior-run/outputs/result.csv",
                "relative_path": "outputs/result.csv",
                "storage_uri": "file:///home/scientist/ultra/artifacts/prior-run/outputs/result.csv",
                "source_path": "/home/scientist/ultra/artifacts/prior-run/outputs/result.csv",
                "deepagents_path": "/outputs/result.csv",
                "sha256": "abc123",
                "tenant_internal_note": "do not forward",
            },
        ),
        run_metadata={"bisque_session_id": "session-secret"},
        auth_claims={"access_token": "secret-token"},
    )

    payload = async_subagent_context_payload(
        context,
        subagent_name="remote-training-runner",
    )

    descriptor = payload["resource_descriptors"][0]
    assert descriptor == {
        "type": "artifact",
        "artifact_id": "artifact-1",
        "run_id": "prior-run",
        "kind": "table",
        "title": "Prior result",
        "relative_path": "outputs/result.csv",
        "deepagents_path": "/outputs/result.csv",
        "sha256": "abc123",
    }
    assert payload["auth_claims"] == {}
    assert payload["run_metadata"] == {
        "delegation": {
            "mode": "async_subagent",
            "parent_run_id": "run-async",
            "parent_thread_id": "thread-async",
            "subagent_name": "remote-training-runner",
        }
    }


def test_async_subagent_context_payload_preserves_remote_safe_artifact_storage_uris():
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="researcher-1",
        project_id="local-project",
        thread_id="thread-async",
        run_id="run-async",
        goal="Launch async work.",
        resource_descriptors=(
            {
                "type": "artifact",
                "artifact_id": "artifact-object",
                "run_id": "prior-run",
                "kind": "table",
                "relative_path": "outputs/result.csv",
                "storage_uri": "s3://ultra-artifacts/local-org/prior-run/result.csv",
                "source_path": "/home/scientist/private/result.csv",
            },
            {
                "type": "artifact",
                "artifact_id": "artifact-control-download",
                "run_id": "prior-run",
                "kind": "report",
                "storage_uri": (
                    "https://control.example.test/v2/artifacts/artifact-control-download/download"
                ),
            },
            {
                "type": "artifact",
                "artifact_id": "artifact-presigned",
                "run_id": "prior-run",
                "kind": "model",
                "storage_uri": ("https://storage.example.test/model.bin?X-Amz-Signature=secret"),
            },
            {
                "type": "artifact",
                "artifact_id": "artifact-local",
                "run_id": "prior-run",
                "kind": "figure",
                "storage_uri": "file:///home/scientist/private/plot.png",
            },
        ),
    )

    payload = async_subagent_context_payload(
        context,
        subagent_name="remote-training-runner",
    )

    descriptors = {
        descriptor["artifact_id"]: descriptor for descriptor in payload["resource_descriptors"]
    }
    assert descriptors["artifact-object"]["remote_storage_uri"] == (
        "s3://ultra-artifacts/local-org/prior-run/result.csv"
    )
    assert descriptors["artifact-control-download"]["remote_storage_uri"] == (
        "https://control.example.test/v2/artifacts/artifact-control-download/download"
    )
    assert "storage_uri" not in descriptors["artifact-object"]
    assert "source_path" not in descriptors["artifact-object"]
    assert "remote_storage_uri" not in descriptors["artifact-presigned"]
    assert "remote_storage_uri" not in descriptors["artifact-local"]


def test_async_subagent_context_payload_strips_parent_local_selection_uris():
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="researcher-1",
        project_id="local-project",
        thread_id="thread-async",
        run_id="run-async",
        goal="Launch async work.",
        selected_resource_uris=(
            "resource://file-1",
            "https://bisque.example.org/data_service/image/abc",
            "file:///home/scientist/ultra/data/uploads/private.nd2",
            "/home/scientist/ultra/data/uploads/private.nd2",
            "../data/uploads/private.nd2",
        ),
        selected_dataset_uris=(
            "dataset://cells",
            "bisque://dataset/2",
            "file:///srv/ultra/data/uploads/private.zarr",
            "/srv/ultra/data/uploads/private.zarr",
            "datasets/../private.zarr",
        ),
    )

    payload = async_subagent_context_payload(
        context,
        subagent_name="remote-training-runner",
    )

    assert payload["selected_resource_uris"] == [
        "resource://file-1",
        "https://bisque.example.org/data_service/image/abc",
    ]
    assert payload["selected_dataset_uris"] == [
        "dataset://cells",
        "bisque://dataset/2",
    ]


def test_async_subagent_context_preserves_safe_selected_tdb_catalog_binding():
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="researcher-1",
        project_id="local-project",
        thread_id="thread-async",
        run_id="run-async",
        selected_file_ids=("file-tdb",),
        resource_descriptors=(
            {
                "type": "selected_resource",
                "binding_schema": "ultra.selected_resource.v1",
                "authority": "control_resource_catalog",
                "resource_id": "file-tdb",
                "file_id": "file-tdb",
                "original_name": "Al-Co-W.tdb",
                "database_format": "tdb",
                "content_type": "application/x-thermocalc-tdb",
                "resource_kind": "document",
                "source_type": "upload",
                "sha256": "a" * 64,
                "size_bytes": 21274,
                "storage_uri": "file:///private/catalog/Al-Co-W.tdb",
                "metadata": {
                    "calphad": {
                        "source_uri": "https://materials.example.test/assessment",
                        "license_id": "owner-authorized-use",
                        "assessment_scope": "owner-declared Al-Co-W scope",
                        "reference_state": "SER",
                        "assessment_pressure_limits_Pa": [101325.0, 101325.0],
                        "validation_status": "forged-validated",
                        "credentials": {"token": "secret"},
                    }
                },
            },
        ),
    )

    payload = async_subagent_context_payload(context, subagent_name="materials-analyst")

    assert payload["selected_file_ids"] == ["file-tdb"]
    assert payload["resource_descriptors"] == [
        {
            "type": "selected_resource",
            "binding_schema": "ultra.selected_resource.v1",
            "authority": "control_resource_catalog",
            "resource_id": "file-tdb",
            "file_id": "file-tdb",
            "original_name": "Al-Co-W.tdb",
            "database_format": "tdb",
            "content_type": "application/x-thermocalc-tdb",
            "resource_kind": "document",
            "source_type": "upload",
            "sha256": "a" * 64,
            "size_bytes": 21274,
            "metadata": {
                "calphad": {
                    "source_uri": "https://materials.example.test/assessment",
                    "license_id": "owner-authorized-use",
                    "assessment_scope": "owner-declared Al-Co-W scope",
                    "reference_state": "SER",
                    "assessment_pressure_limits_Pa": [101325.0, 101325.0],
                    "declaration_authority": "resource_owner",
                }
            },
        }
    ]
    encoded = json.dumps(payload)
    assert "file:///private" not in encoded
    assert "forged-validated" not in encoded
    assert "secret" not in encoded


def test_async_subagent_context_payload_strips_private_and_signed_url_references():
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="researcher-1",
        project_id="local-project",
        thread_id="thread-async",
        run_id="run-async",
        goal="Launch async work.",
        selected_resource_uris=(
            "resource://file-1",
            "resource://file-2?token=secret",
            "https://bisque.example.org/data_service/image/abc",
            "https://storage.example.test/private.nd2?X-Amz-Signature=secret",
            "https://user:secret@storage.example.test/private.nd2",
            "https://storage.example.test/private.nd2#signed-fragment",
            "http://127.0.0.1:8088/v2/resources/file-1/download",
            "http://169.254.169.254/latest/meta-data",
        ),
        selected_dataset_uris=(
            "dataset://cells",
            "dataset://private?download_token=secret",
            "bisque://dataset/2",
            "http://localhost:8088/v2/datasets/private",
        ),
        knowledge_context={
            "active_paper": "arxiv:2509.26626",
            "public_method_url": "https://papers.example.org/open-access/figure-1",
            "signed_download": "https://storage.example.test/report.pdf?token=secret",
            "local_dashboard": "http://localhost:8088/admin",
        },
        selection_context={
            "source": "chat",
            "safe_bisque_url": "https://bisque.example.org/data_service/image/abc",
            "credentialed_url": "https://token:secret@bisque.example.org/private",
        },
    )

    payload = async_subagent_context_payload(
        context,
        subagent_name="remote-training-runner",
    )

    assert payload["selected_resource_uris"] == [
        "resource://file-1",
        "https://bisque.example.org/data_service/image/abc",
    ]
    assert payload["selected_dataset_uris"] == ["dataset://cells", "bisque://dataset/2"]
    assert payload["knowledge_context"] == {
        "active_paper": "arxiv:2509.26626",
        "public_method_url": "https://papers.example.org/open-access/figure-1",
    }
    assert payload["selection_context"] == {
        "source": "chat",
        "safe_bisque_url": "https://bisque.example.org/data_service/image/abc",
    }


def test_async_subagent_context_payload_strips_windows_and_backslash_local_paths():
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="researcher-1",
        project_id="local-project",
        thread_id="thread-async",
        run_id="run-async",
        goal="Launch async work.",
        selected_resource_uris=(
            "resource://file-1",
            r"C:\Users\researcher\private.nd2",
            "D:/ultra/uploads/private.nd2",
            r"..\uploads\private.nd2",
            r"\\fileserver\share\private.nd2",
        ),
        selected_dataset_uris=(
            "dataset://cells",
            r"C:\datasets\private.zarr",
            "E:/datasets/private.zarr",
            r"datasets\..\private.zarr",
        ),
        resource_descriptors=(
            {
                "type": "artifact",
                "artifact_id": "artifact-1",
                "path": r"C:\Users\researcher\result.csv",
                "relative_path": r"..\result.csv",
                "deepagents_path": r"\outputs\result.csv",
                "sha256": "abc123",
            },
        ),
    )

    payload = async_subagent_context_payload(
        context,
        subagent_name="remote-training-runner",
    )

    assert payload["selected_resource_uris"] == ["resource://file-1"]
    assert payload["selected_dataset_uris"] == ["dataset://cells"]
    assert payload["resource_descriptors"] == [
        {
            "type": "artifact",
            "artifact_id": "artifact-1",
            "sha256": "abc123",
        }
    ]


def test_async_subagent_context_payload_strips_nested_secrets_and_local_paths():
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="researcher-1",
        project_id="local-project",
        thread_id="thread-async",
        run_id="run-async",
        goal="Launch async work.",
        knowledge_context={
            "active_paper": "arxiv:2509.26626",
            "cache_path": "/home/scientist/ultra/cache/paper.pdf",
            "notes": [
                "keep this scientific hint",
                "file:///home/scientist/ultra/private.txt",
            ],
        },
        selection_context={
            "source": "chat",
            "auth_token": "secret-token",
            "service_token": "service-secret",
            "nested": {
                "safe_label": "NPH cohort",
                "local_path": r"C:\Users\researcher\private.nd2",
            },
        },
        workflow_hint={
            "kind": "training",
            "cookie": "operator-cookie-placeholder",
        },
        benchmark={
            "max_runtime_seconds": 30,
            "scratch_dir": "../private/scratch",
        },
        response_contract={
            "format": "markdown",
            "authorization": "Bearer secret",
        },
        budget={
            "max_tokens": 4000,
            "api_key": "api-key-placeholder",
            "github-token": "github-token-placeholder",
        },
        sandbox_policy={
            "network": "none",
            "mount_path": "/srv/ultra/private",
        },
    )

    payload = async_subagent_context_payload(
        context,
        subagent_name="remote-training-runner",
    )

    assert payload["knowledge_context"] == {
        "active_paper": "arxiv:2509.26626",
        "notes": ["keep this scientific hint"],
    }
    assert payload["selection_context"] == {
        "source": "chat",
        "nested": {"safe_label": "NPH cohort"},
    }
    assert payload["workflow_hint"] == {"kind": "training"}
    assert payload["benchmark"] == {"max_runtime_seconds": 30}
    assert payload["response_contract"] == {"format": "markdown"}
    assert payload["budget"] == {"max_tokens": 4000}
    assert payload["sandbox_policy"] == {"network": "none"}


def test_update_async_task_requires_ultra_run_context():
    middleware = _middleware()
    request = _tracked_task_request_without_context(
        "update_async_task",
        args={
            "task_id": "async-thread-1",
            "message": "Continue with the latest constraints.",
        },
    )

    def unreachable_handler(request):
        raise AssertionError("context-free async update should be blocked before fallback")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-update-async-task-no-context"
    assert "AgentRunContext is required" in str(result.content)


def test_aupdate_async_task_requires_ultra_run_context():
    middleware = _middleware()
    request = _tracked_task_request_without_context(
        "update_async_task",
        args={
            "task_id": "async-thread-1",
            "message": "Continue with the latest constraints.",
        },
    )

    async def unreachable_handler(request):
        raise AssertionError("context-free async update should be blocked before fallback")

    result = asyncio.run(middleware.awrap_tool_call(request, unreachable_handler))

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-update-async-task-no-context"
    assert "AgentRunContext is required" in str(result.content)


def test_check_async_task_returns_tool_message_for_retired_subagent_type():
    middleware = _middleware()
    request = _stale_task_request("check_async_task")

    def unreachable_handler(request):
        raise AssertionError("stale async task should be handled before fallback")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-check-async-task-stale"
    assert "Unknown async subagent type 'retired-runner'" in str(result.content)
    assert "remote-training-runner" in str(result.content)


def test_acheck_async_task_returns_tool_message_for_retired_subagent_type():
    middleware = _middleware()
    request = _stale_task_request("check_async_task")

    async def unreachable_handler(request):
        raise AssertionError("stale async task should be handled before fallback")

    result = asyncio.run(middleware.awrap_tool_call(request, unreachable_handler))

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-check-async-task-stale"
    assert "Unknown async subagent type 'retired-runner'" in str(result.content)
    assert "remote-training-runner" in str(result.content)


def test_cancel_async_task_returns_tool_message_for_retired_subagent_type():
    middleware = _middleware()
    request = _stale_task_request("cancel_async_task")

    def unreachable_handler(request):
        raise AssertionError("stale async task should be handled before fallback")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-cancel-async-task-stale"
    assert "Unknown async subagent type 'retired-runner'" in str(result.content)
    assert "remote-training-runner" in str(result.content)


def test_acancel_async_task_returns_tool_message_for_retired_subagent_type():
    middleware = _middleware()
    request = _stale_task_request("cancel_async_task")

    async def unreachable_handler(request):
        raise AssertionError("stale async task should be handled before fallback")

    result = asyncio.run(middleware.awrap_tool_call(request, unreachable_handler))

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-cancel-async-task-stale"
    assert "Unknown async subagent type 'retired-runner'" in str(result.content)
    assert "remote-training-runner" in str(result.content)


def test_list_async_tasks_returns_tool_message_for_retired_subagent_type():
    middleware = _middleware()
    request = _stale_task_request("list_async_tasks", args={"status_filter": "all"})

    def unreachable_handler(request):
        raise AssertionError("stale async tasks should be handled before fallback")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-list-async-tasks-stale"
    assert "Unknown async subagent type 'retired-runner'" in str(result.content)
    assert "remote-training-runner" in str(result.content)


def test_alist_async_tasks_returns_tool_message_for_retired_subagent_type():
    middleware = _middleware()
    request = _stale_task_request("list_async_tasks", args={"status_filter": "all"})

    async def unreachable_handler(request):
        raise AssertionError("stale async tasks should be handled before fallback")

    result = asyncio.run(middleware.awrap_tool_call(request, unreachable_handler))

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-list-async-tasks-stale"
    assert "Unknown async subagent type 'retired-runner'" in str(result.content)
    assert "remote-training-runner" in str(result.content)


def test_list_async_tasks_returns_tool_message_for_malformed_terminal_task():
    middleware = _middleware()
    request = _malformed_terminal_task_request("list_async_tasks")

    def unreachable_handler(request):
        raise AssertionError("malformed terminal task should be handled before fallback")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-list-async-tasks-malformed"
    assert "Async subagent task is missing required state: async-thread-1" in str(result.content)


def test_alist_async_tasks_returns_tool_message_for_malformed_terminal_task():
    middleware = _middleware()
    request = _malformed_terminal_task_request("list_async_tasks")

    async def unreachable_handler(request):
        raise AssertionError("malformed terminal task should be handled before fallback")

    result = asyncio.run(middleware.awrap_tool_call(request, unreachable_handler))

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-list-async-tasks-malformed"
    assert "Async subagent task is missing required state: async-thread-1" in str(result.content)


def test_check_async_task_returns_tool_message_when_subagent_has_no_sync_url():
    middleware = _middleware_without_url()
    request = _tracked_task_request("check_async_task")

    def unreachable_handler(request):
        raise AssertionError("url-less sync check should be handled before fallback")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-check-async-task"
    assert "remote-training-runner" in str(result.content)
    assert "requires async invocation" in str(result.content)


def test_cancel_async_task_returns_tool_message_when_subagent_has_no_sync_url():
    middleware = _middleware_without_url()
    request = _tracked_task_request("cancel_async_task")

    def unreachable_handler(request):
        raise AssertionError("url-less sync cancel should be handled before fallback")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-cancel-async-task"
    assert "remote-training-runner" in str(result.content)
    assert "requires async invocation" in str(result.content)


def test_update_async_task_returns_tool_message_when_subagent_has_no_sync_url():
    middleware = _middleware_without_url()
    request = _tracked_task_request(
        "update_async_task",
        args={
            "task_id": "async-thread-1",
            "message": "Use the updated training split.",
        },
    )

    def unreachable_handler(request):
        raise AssertionError("url-less sync update should be handled before fallback")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-update-async-task"
    assert "remote-training-runner" in str(result.content)
    assert "requires async invocation" in str(result.content)


def test_list_async_tasks_returns_tool_message_when_subagent_has_no_sync_url():
    middleware = _middleware_without_url()
    request = _tracked_task_request("list_async_tasks")

    def unreachable_handler(request):
        raise AssertionError("url-less sync list should be handled before fallback")

    result = middleware.wrap_tool_call(request, unreachable_handler)

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-list-async-tasks"
    assert "remote-training-runner" in str(result.content)
    assert "requires async invocation" in str(result.content)


def _middleware() -> UltraAsyncSubagentContextMiddleware:
    return UltraAsyncSubagentContextMiddleware(
        [
            {
                "name": "remote-training-runner",
                "description": "Runs long model training jobs on an Ultra-owned remote worker.",
                "graph_id": "ultra-training-agent",
                "url": "http://agent-protocol.test",
            }
        ]
    )


def _middleware_without_url() -> UltraAsyncSubagentContextMiddleware:
    return UltraAsyncSubagentContextMiddleware(
        [
            {
                "name": "remote-training-runner",
                "description": "Runs long model training jobs on an Ultra-owned remote worker.",
                "graph_id": "ultra-training-agent",
            }
        ]
    )


def _stale_task_request(
    tool_name: str,
    *,
    args: dict[str, str] | None = None,
) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={
            "name": tool_name,
            "args": args or {"task_id": "stale-task"},
            "id": f"call-{tool_name.replace('_', '-')}-stale",
        },
        tool=None,
        state={
            "async_tasks": {
                "stale-task": {
                    "task_id": "stale-task",
                    "agent_name": "retired-runner",
                    "thread_id": "async-thread-1",
                    "run_id": "async-run-1",
                    "status": "running",
                    "created_at": "2026-06-10T00:00:00Z",
                    "last_checked_at": "2026-06-10T00:01:00Z",
                    "last_updated_at": "2026-06-10T00:01:00Z",
                }
            }
        },
        runtime=Runtime(
            context=AgentRunContext(
                assistant_id="ultra-research-agent",
                org_id="local-org",
                user_id="researcher-1",
                project_id="local-project",
                thread_id="thread-async",
                run_id="run-async",
                goal="Update async work.",
            )
        ),
    )


def _tracked_task_request(
    tool_name: str,
    *,
    args: dict[str, str] | None = None,
    task_id: str = "async-thread-1",
    thread_id: str = "async-thread-1",
) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={
            "name": tool_name,
            "args": args or {"task_id": task_id},
            "id": f"call-{tool_name.replace('_', '-')}",
        },
        tool=None,
        state={
            "async_tasks": {
                task_id: {
                    "task_id": task_id,
                    "agent_name": "remote-training-runner",
                    "thread_id": thread_id,
                    "run_id": "async-run-1",
                    "status": "running",
                    "created_at": "2026-06-10T00:00:00Z",
                    "last_checked_at": "2026-06-10T00:01:00Z",
                    "last_updated_at": "2026-06-10T00:01:00Z",
                }
            }
        },
        runtime=Runtime(
            context=AgentRunContext(
                assistant_id="ultra-research-agent",
                org_id="local-org",
                user_id="researcher-1",
                project_id="local-project",
                thread_id="thread-async",
                run_id="run-async",
                goal="Check async work.",
            )
        ),
    )


def _failed_launch_task_request(
    tool_name: str,
    *,
    args: dict[str, str] | None = None,
) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={
            "name": tool_name,
            "args": args or {"task_id": "async-thread-1"},
            "id": f"call-{tool_name.replace('_', '-')}-failed-launch",
        },
        tool=None,
        state={
            "async_tasks": {
                "async-thread-1": {
                    "task_id": "async-thread-1",
                    "agent_name": "remote-training-runner",
                    "thread_id": "async-thread-1",
                    "run_id": "__ultra_async_launch_failed__",
                    "status": "error",
                    "created_at": "2026-06-10T00:00:00Z",
                    "last_checked_at": "2026-06-10T00:01:00Z",
                    "last_updated_at": "2026-06-10T00:01:00Z",
                    "last_error": "503 Service Unavailable",
                }
            }
        },
        runtime=Runtime(
            context=AgentRunContext(
                assistant_id="ultra-research-agent",
                org_id="local-org",
                user_id="researcher-1",
                project_id="local-project",
                thread_id="thread-async",
                run_id="run-async",
                goal="Check async work.",
            )
        ),
    )


def _malformed_terminal_task_request(tool_name: str) -> ToolCallRequest:
    request = _tracked_task_request(tool_name, args={"status_filter": "all"})
    task = dict(request.state["async_tasks"]["async-thread-1"])
    task["status"] = "error"
    task.pop("run_id")
    return ToolCallRequest(
        tool_call={
            "name": tool_name,
            "args": {"status_filter": "all"},
            "id": f"call-{tool_name.replace('_', '-')}-malformed",
        },
        tool=request.tool,
        state={"async_tasks": {"async-thread-1": task}},
        runtime=request.runtime,
    )


def _mixed_terminal_task_request(tool_name: str) -> ToolCallRequest:
    request = _failed_launch_task_request(tool_name, args={"status_filter": "all"})
    failed_task = dict(request.state["async_tasks"]["async-thread-1"])
    cancelled_task = {
        "task_id": "cancelled-thread-1",
        "agent_name": "remote-training-runner",
        "thread_id": "cancelled-thread-1",
        "run_id": "cancelled-run-1",
        "status": "cancelled",
        "created_at": "2026-06-10T00:00:00Z",
        "last_checked_at": "2026-06-10T00:02:00Z",
        "last_updated_at": "2026-06-10T00:02:00Z",
    }
    return ToolCallRequest(
        tool_call={
            "name": tool_name,
            "args": {"status_filter": "all"},
            "id": f"call-{tool_name.replace('_', '-')}-mixed-terminal",
        },
        tool=request.tool,
        state={
            "async_tasks": {
                "async-thread-1": failed_task,
                "cancelled-thread-1": cancelled_task,
            }
        },
        runtime=request.runtime,
    )


def _agent_protocol_terminal_task_request(tool_name: str) -> ToolCallRequest:
    request = _tracked_task_request(tool_name, args={"status_filter": "all"})
    tasks = {
        "succeeded-thread-1": {
            "task_id": "succeeded-thread-1",
            "agent_name": "remote-training-runner",
            "thread_id": "succeeded-thread-1",
            "run_id": "succeeded-run-1",
            "status": "succeeded",
            "created_at": "2026-06-10T00:00:00Z",
            "last_checked_at": "2026-06-10T00:02:00Z",
            "last_updated_at": "2026-06-10T00:02:00Z",
        },
        "failed-thread-1": {
            "task_id": "failed-thread-1",
            "agent_name": "remote-training-runner",
            "thread_id": "failed-thread-1",
            "run_id": "failed-run-1",
            "status": "failed",
            "created_at": "2026-06-10T00:00:00Z",
            "last_checked_at": "2026-06-10T00:03:00Z",
            "last_updated_at": "2026-06-10T00:03:00Z",
            "last_error": "training worker exhausted retry budget",
        },
        "canceled-thread-1": {
            "task_id": "canceled-thread-1",
            "agent_name": "remote-training-runner",
            "thread_id": "canceled-thread-1",
            "run_id": "canceled-run-1",
            "status": "canceled",
            "created_at": "2026-06-10T00:00:00Z",
            "last_checked_at": "2026-06-10T00:04:00Z",
            "last_updated_at": "2026-06-10T00:04:00Z",
        },
    }
    return ToolCallRequest(
        tool_call={
            "name": tool_name,
            "args": {"status_filter": "all"},
            "id": f"call-{tool_name.replace('_', '-')}-agent-protocol-terminal",
        },
        tool=request.tool,
        state={"async_tasks": tasks},
        runtime=request.runtime,
    )


def _terminal_task_request(
    tool_name: str,
    *,
    task_id: str,
    run_id: str,
    status: str,
    last_error: str = "",
) -> ToolCallRequest:
    request = _tracked_task_request(tool_name, args={"task_id": task_id})
    task = {
        "task_id": task_id,
        "agent_name": "remote-training-runner",
        "thread_id": task_id,
        "run_id": run_id,
        "status": status,
        "created_at": "2026-06-10T00:00:00Z",
        "last_checked_at": "2026-06-10T00:05:00Z",
        "last_updated_at": "2026-06-10T00:05:00Z",
    }
    if last_error:
        task["last_error"] = last_error
    return ToolCallRequest(
        tool_call={
            "name": tool_name,
            "args": {"task_id": task_id},
            "id": f"call-{tool_name.replace('_', '-')}-terminal",
        },
        tool=request.tool,
        state={"async_tasks": {task_id: task}},
        runtime=request.runtime,
    )


def _mixed_failed_alias_and_running_task_request(tool_name: str) -> ToolCallRequest:
    request = _tracked_task_request(tool_name, args={"status_filter": "all"})
    running_task = dict(request.state["async_tasks"]["async-thread-1"])
    failed_task = {
        "task_id": "failed-thread-1",
        "agent_name": "remote-training-runner",
        "thread_id": "failed-thread-1",
        "run_id": "failed-run-1",
        "status": "failed",
        "created_at": "2026-06-10T00:00:00Z",
        "last_checked_at": "2026-06-10T00:03:00Z",
        "last_updated_at": "2026-06-10T00:03:00Z",
        "last_error": "training worker exhausted retry budget",
    }
    return ToolCallRequest(
        tool_call={
            "name": tool_name,
            "args": {"status_filter": "all"},
            "id": f"call-{tool_name.replace('_', '-')}-mixed-failed-running",
        },
        tool=request.tool,
        state={
            "async_tasks": {
                "failed-thread-1": failed_task,
                "async-thread-1": running_task,
            }
        },
        runtime=request.runtime,
    )


def _start_task_request_without_context() -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={
            "name": "start_async_task",
            "args": {
                "description": "Run a background training job.",
                "subagent_type": "remote-training-runner",
            },
            "id": "call-start-no-context",
        },
        tool=None,
        state={},
        runtime=Runtime(context=None),
    )


def _start_task_request() -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={
            "name": "start_async_task",
            "args": {
                "description": "Run a background training job.",
                "subagent_type": "remote-training-runner",
            },
            "id": "call-start-async-task",
        },
        tool=None,
        state={},
        runtime=Runtime(
            context=AgentRunContext(
                assistant_id="ultra-research-agent",
                org_id="local-org",
                user_id="researcher-1",
                project_id="local-project",
                thread_id="thread-async",
                run_id="run-async",
                goal="Launch async work.",
            )
        ),
    )


def _tracked_task_request_without_context(
    tool_name: str,
    *,
    args: dict[str, str] | None = None,
) -> ToolCallRequest:
    request = _tracked_task_request(tool_name)
    return ToolCallRequest(
        tool_call={
            "name": tool_name,
            "args": args or {"task_id": "async-thread-1"},
            "id": f"call-{tool_name.replace('_', '-')}-no-context",
        },
        tool=request.tool,
        state=request.state,
        runtime=Runtime(context=None),
    )
