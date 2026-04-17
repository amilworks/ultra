from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.api.v3 import build_v3_router


def _iso_now() -> str:
    return datetime.utcnow().isoformat()


@dataclass
class FakeSessionStore:
    sessions: dict[str, dict[str, Any]] = field(default_factory=dict)
    messages: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def create_session(self, **kwargs: Any) -> dict[str, Any]:
        session_id = f"session-{len(self.sessions) + 1}"
        record = {
            "session_id": session_id,
            "user_id": kwargs["user_id"],
            "title": kwargs["title"],
            "status": kwargs["status"],
            "summary": kwargs["summary"],
            "memory_policy": kwargs["memory_policy"],
            "knowledge_scope": kwargs["knowledge_scope"],
            "metadata": kwargs["metadata"],
            "created_at": _iso_now(),
            "updated_at": _iso_now(),
        }
        self.sessions[session_id] = record
        self.messages[session_id] = []
        return record

    def list_sessions(self, *, user_id: str, limit: int) -> list[dict[str, Any]]:
        del limit
        return [row for row in self.sessions.values() if row["user_id"] == user_id]

    def get_session(self, *, session_id: str, user_id: str) -> dict[str, Any] | None:
        record = self.sessions.get(session_id)
        if record is None or record["user_id"] != user_id:
            return None
        return record

    def list_messages(self, *, session_id: str, user_id: str, limit: int) -> list[dict[str, Any]]:
        del user_id
        return self.messages.get(session_id, [])[:limit]


@dataclass
class FakeRunStore:
    runs: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_run(self, *, run_id: str, user_id: str) -> dict[str, Any] | None:
        record = self.runs.get(run_id)
        if record is None or record["user_id"] != user_id:
            return None
        return record


@dataclass
class FakeEventStore:
    events_by_run: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def list_events(self, *, run_id: str, user_id: str, limit: int) -> list[dict[str, Any]]:
        del user_id
        return self.events_by_run.get(run_id, [])[:limit]


@dataclass
class FakeArtifactStore:
    artifacts_by_run: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def list_artifacts(self, *, run_id: str, user_id: str, limit: int) -> list[dict[str, Any]]:
        del user_id
        return self.artifacts_by_run.get(run_id, [])[:limit]


@dataclass
class FakeApprovalStore:
    approvals: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_approval(self, *, approval_id: str, user_id: str) -> dict[str, Any] | None:
        record = self.approvals.get(approval_id)
        if record is None or record["user_id"] != user_id:
            return None
        return record


class FakeOrchestrator:
    def __init__(self, services: SimpleNamespace) -> None:
        self._services = services

    def start_run(self, *, session_id: str, user_id: str, request: Any) -> dict[str, Any]:
        run_id = f"run-{len(self._services.runs.runs) + 1}"
        run = {
            "run_id": run_id,
            "session_id": session_id,
            "user_id": user_id,
            "workflow_name": "scientist_workflow",
            "status": "queued",
            "current_step": "planning",
            "checkpoint_state": {},
            "budget_state": request.budgets.model_dump(mode="json"),
            "response_text": None,
            "trace_group_id": None,
            "metrics": {},
            "error": None,
            "metadata": request.metadata,
            "created_at": _iso_now(),
            "updated_at": _iso_now(),
        }
        self._services.runs.runs[run_id] = run
        self._services.sessions.messages.setdefault(session_id, []).extend(
            [
                {
                    "message_id": f"message-{index + 1}",
                    "session_id": session_id,
                    "role": message.role,
                    "content": message.content,
                    "attachments": [item.model_dump(mode="json") for item in message.attachments],
                    "run_id": run_id,
                    "metadata": message.metadata,
                    "created_at": _iso_now(),
                }
                for index, message in enumerate(request.messages)
            ]
        )
        self._services.events.events_by_run[run_id] = [
            {
                "event_id": "event-1",
                "run_id": run_id,
                "session_id": session_id,
                "user_id": user_id,
                "event_kind": "phase",
                "event_type": "planning.started",
                "agent_name": "scientist",
                "tool_name": None,
                "level": "info",
                "payload": {"status": "started"},
                "created_at": _iso_now(),
            }
        ]
        self._services.artifacts.artifacts_by_run[run_id] = [
            {
                "artifact_id": "artifact-1",
                "run_id": run_id,
                "session_id": session_id,
                "user_id": user_id,
                "kind": "report",
                "title": "Run report",
                "path": "reports/run-report.json",
                "source_path": None,
                "preview_path": None,
                "metadata": {},
                "created_at": _iso_now(),
                "updated_at": _iso_now(),
            }
        ]
        self._services.approvals.approvals["approval-1"] = {
            "approval_id": "approval-1",
            "run_id": run_id,
            "session_id": session_id,
            "user_id": user_id,
            "action_type": "tool_execution",
            "tool_name": "sandboxed_python",
            "status": "pending",
            "request_payload": {"tool_name": "sandboxed_python"},
            "resolution": {},
            "created_at": _iso_now(),
            "updated_at": _iso_now(),
        }
        return run

    def resolve_approval(
        self,
        *,
        approval_id: str,
        user_id: str,
        decision: str,
        note: str | None,
        metadata: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        approval = dict(self._services.approvals.approvals[approval_id])
        approval["status"] = "approved" if decision == "approve" else "rejected"
        approval["resolution"] = {
            "decision": decision,
            "note": note,
            "metadata": metadata,
        }
        approval["updated_at"] = _iso_now()
        self._services.approvals.approvals[approval_id] = approval
        run = dict(self._services.runs.runs[approval["run_id"]])
        run["status"] = "running" if decision == "approve" else "canceled"
        run["updated_at"] = _iso_now()
        self._services.runs.runs[approval["run_id"]] = run
        assert run["user_id"] == user_id
        return approval, run


def _build_test_client() -> TestClient:
    services = SimpleNamespace(
        sessions=FakeSessionStore(),
        runs=FakeRunStore(),
        events=FakeEventStore(),
        artifacts=FakeArtifactStore(),
        approvals=FakeApprovalStore(),
    )
    services.orchestrator = FakeOrchestrator(services)

    app = FastAPI()
    app.state.agentic_v3 = services
    app.include_router(
        build_v3_router(
            current_user_id=lambda bisque_auth, allow_anonymous=False: str(
                (bisque_auth or {}).get("user_id") or ("anonymous" if allow_anonymous else "")
            ),
            optional_auth_dependency=lambda: {"user_id": "user-123"},
            require_api_key_dependency=lambda: None,
        )
    )
    return TestClient(app)


def test_v3_session_run_artifact_flow() -> None:
    client = _build_test_client()

    session_response = client.post("/v3/sessions", json={"title": "Cell segmentation"})
    assert session_response.status_code == 200
    session_id = session_response.json()["session_id"]

    run_response = client.post(
        f"/v3/sessions/{session_id}/runs",
        json={
            "messages": [{"role": "user", "content": "Quantify nuclei coverage."}],
            "budgets": {"max_tool_calls": 3, "max_runtime_seconds": 120},
        },
    )
    assert run_response.status_code == 200
    run_id = run_response.json()["run_id"]

    messages_response = client.get(f"/v3/sessions/{session_id}/messages")
    events_response = client.get(f"/v3/runs/{run_id}/events")
    artifacts_response = client.get(f"/v3/runs/{run_id}/artifacts")

    assert messages_response.status_code == 200
    assert messages_response.json()["count"] == 1
    assert events_response.status_code == 200
    assert events_response.json()["count"] == 1
    assert artifacts_response.status_code == 200
    assert artifacts_response.json()["count"] == 1


def test_v3_approval_resolution_updates_run_state() -> None:
    client = _build_test_client()
    session_response = client.post("/v3/sessions", json={"title": "Approval flow"})
    session_id = session_response.json()["session_id"]
    run_response = client.post(
        f"/v3/sessions/{session_id}/runs",
        json={"messages": [{"role": "user", "content": "Execute the next step."}]},
    )
    run_id = run_response.json()["run_id"]

    approval_response = client.post(
        f"/v3/runs/{run_id}/approvals/approval-1",
        json={"decision": "approve", "note": "Looks good."},
    )

    assert approval_response.status_code == 200
    assert approval_response.json()["approval"]["status"] == "approved"
    assert approval_response.json()["run"]["status"] == "running"
