import httpx

from ultra_deepagents.agent import build_research_agent, build_run_context_brief
from ultra_deepagents.bisque.tools import (
    bisque_mutation_goal_authorized,
    describe_bisque_module,
    list_bisque_modules,
    run_bisque_module,
)
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.schemas import _remote_mutation_intents

MEX_URI = "https://bisque.example.org/data_service/00-NEW"
IMAGE_URI = "https://bisque.example.org/data_service/00-IMG"


def _settings() -> RuntimeSettings:
    return RuntimeSettings(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
        control_base_url="http://control.test",
        openai_api_key="should-not-leak",
    )


def _context(*intents: str, goal: str = "Run prairie dog detection on BisQue") -> AgentRunContext:
    return AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-bisque-modules",
        goal=goal,
        remote_mutation_intents=tuple(intents),
        run_metadata={"bisque_session_id": "bisque_session_opaque"},
    )


class _FakeResponse:
    def __init__(self, url: str, payload: dict, status_code: int = 200):
        self._url = url
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", self._url)
            response = httpx.Response(self.status_code, json=self._payload, request=request)
            raise httpx.HTTPStatusError("control plane error", request=request, response=response)

    def json(self):
        return self._payload


def _fake_client(routes: dict, calls: list):
    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json, headers=None):
            calls.append({"url": url, "json": json, "headers": headers or {}})
            path = url.removeprefix("http://control.test")
            handler = routes[path]
            payload, status_code = handler(json) if callable(handler) else (handler, 200)
            return _FakeResponse(url, payload, status_code)

    return FakeClient


def test_list_and_describe_modules_call_control_plane_without_credentials(monkeypatch):
    calls: list = []
    routes = {
        "/v2/bisque/modules": (
            {"ok": True, "count": 1, "total": 30, "modules": [{"name": "PrairieDogDetection"}]}
        ),
        "/v2/bisque/module": (
            {"ok": True, "module": {"name": "PrairieDogDetection", "help": "YOLO-v5"}}
        ),
    }
    monkeypatch.setattr("httpx.Client", _fake_client(routes, calls))

    listed = list_bisque_modules(
        _settings(), task="detect prairie dogs", limit=5, context=_context()
    )
    described = describe_bisque_module(
        _settings(), name="PrairieDogDetection", context=_context()
    )

    assert listed["ok"] is True and listed["modules"][0]["name"] == "PrairieDogDetection"
    assert described["module"]["help"] == "YOLO-v5"
    assert calls[0]["json"] == {"query": "detect prairie dogs", "limit": 5}
    assert calls[1]["json"] == {"name": "PrairieDogDetection"}
    for call in calls:
        assert call["headers"]["X-Ultra-Bisque-Session-Id"] == "bisque_session_opaque"
        assert "Authorization" not in call["headers"]
        assert "should-not-leak" not in str(call["headers"])


def test_run_module_is_blocked_without_the_run_module_intent(monkeypatch):
    calls: list = []
    monkeypatch.setattr("httpx.Client", _fake_client({}, calls))

    result = run_bisque_module(
        _settings(),
        module_name="PrairieDogDetection",
        inputs={"Input Image": IMAGE_URI},
        context=_context("bisque.upload"),
    )

    assert result["ok"] is False
    assert result["blocked"] is True
    assert result["action"] == "run_module"
    assert calls == []


def test_run_module_passes_references_and_polls_until_finished(monkeypatch):
    calls: list = []
    statuses = ["RUNNING", "FINISHED"]

    def module_run(payload):
        status = statuses.pop(0)
        return (
            {
                "resource_uri": MEX_URI,
                "status": status,
                "outputs": [{"name": "csv_file", "client_view_url": "view-csv"}],
            },
            200,
        )

    routes = {
        "/v2/bisque/module-execute": {
            "ok": True,
            "module_name": "PrairieDogDetection",
            "status": "DISPATCH",
            "mex_uri": MEX_URI,
            "mex_uniq": "00-NEW",
            "client_view_url": "view-mex",
            "run": {"resource_uri": MEX_URI, "status": "DISPATCH"},
        },
        "/v2/bisque/module-run": module_run,
    }
    monkeypatch.setattr("httpx.Client", _fake_client(routes, calls))
    sleeps: list = []

    result = run_bisque_module(
        _settings(),
        module_name="PrairieDogDetection",
        inputs='{"Input Image": "%s", "weights": "00-W", "Annotation Name": "ultra_demo"}'
        % IMAGE_URI,
        wait_seconds=60,
        poll_interval=1,
        context=_context("bisque.run_module"),
        sleep=sleeps.append,
    )

    assert calls[0]["url"] == "http://control.test/v2/bisque/module-execute"
    assert calls[0]["json"] == {
        "module_name": "PrairieDogDetection",
        "inputs": {"Input Image": IMAGE_URI, "weights": "00-W", "Annotation Name": "ultra_demo"},
    }
    assert [call["json"] for call in calls[1:]] == [{"mex_uri": MEX_URI}, {"mex_uri": MEX_URI}]
    assert len(sleeps) == 2
    assert result["ok"] is True
    assert result["status"] == "FINISHED"
    assert result["finished"] is True
    assert result["client_view_url"] == "view-mex"
    assert result["run"]["outputs"][0]["client_view_url"] == "view-csv"


def test_run_module_returns_immediately_when_not_waiting(monkeypatch):
    calls: list = []
    routes = {
        "/v2/bisque/module-execute": {
            "ok": True,
            "status": "DISPATCH",
            "mex_uri": MEX_URI,
            "client_view_url": "view-mex",
        }
    }
    monkeypatch.setattr("httpx.Client", _fake_client(routes, calls))

    result = run_bisque_module(
        _settings(),
        module_name="PrairieDogDetection",
        inputs={"Input Image": IMAGE_URI},
        wait_seconds=0,
        context=_context("bisque.run_module"),
    )

    assert len(calls) == 1
    assert result["finished"] is False
    assert "bisque_module_runs" in result["next_step"]


def test_run_module_surfaces_control_plane_validation_errors(monkeypatch):
    calls: list = []
    routes = {
        "/v2/bisque/module-execute": lambda payload: (
            {"error": "missing required resource inputs for PrairieDogDetection: weights"},
            400,
        )
    }
    monkeypatch.setattr("httpx.Client", _fake_client(routes, calls))

    result = run_bisque_module(
        _settings(),
        module_name="PrairieDogDetection",
        inputs={"Input Image": IMAGE_URI},
        context=_context("bisque.run_module"),
    )

    assert result["ok"] is False
    assert result["status_code"] == 400
    assert "weights" in result["error"]


def test_agent_registers_module_tools_and_gates_run_on_intent(monkeypatch):
    def tool_names_for(context):
        captured = {}

        def fake_create_deep_agent(**kwargs):
            captured.update(kwargs)
            return "compiled-agent"

        monkeypatch.setattr("ultra_deepagents.agent.create_deep_agent", fake_create_deep_agent)
        build_research_agent(_settings(), model=object(), backend=object(), context=context)
        return {getattr(tool, "name", "") for tool in captured["tools"]}

    read_only = tool_names_for(_context())
    assert {"bisque_list_modules", "bisque_describe_module"} <= read_only
    assert "bisque_run_module" not in read_only

    authorized = tool_names_for(_context("bisque.run_module"))
    assert "bisque_run_module" in authorized

    brief = build_run_context_brief(_context("bisque.run_module"))
    assert "bisque_run_module" in brief
    assert "bisque_list_modules" in brief


def test_run_module_intent_survives_worker_job_parsing():
    assert _remote_mutation_intents(["bisque.run_module", "bisque.upload"]) == (
        "bisque.upload",
        "bisque.run_module",
    )
    assert _remote_mutation_intents(["bisque.run_module", "bisque.delete_everything"]) == ()


def test_run_module_goal_heuristic():
    for goal in (
        "Run the PrairieDogDetection module on my survey image.",
        "Please execute PrairieDogDetection on BisQue for 00-IMG.",
        "Launch the prairie dog module in my BisQue account.",
    ):
        assert bisque_mutation_goal_authorized(goal, "run_module") is True
    for goal in (
        "Which BisQue module should I run for prairie dogs?",
        "How do I run a module on BisQue?",
        "Can you run a module on BisQue?",
        "Do not run any module on BisQue.",
        "Run the analysis locally.",
    ):
        assert bisque_mutation_goal_authorized(goal, "run_module") is False
