from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi import Request as FastAPIRequest

from src.agentic.models import (
    AgenticAttachment,
    AgenticMessageInput,
    AgenticRunBudget,
    AgenticRunRequest,
)
from src.agno_backend.knowledge import ScientificKnowledgeScope
from src.agno_backend.memory import ScientificMemoryPolicy
from src.api.schemas import (
    V3ApprovalRecord,
    V3ApprovalResolveRequest,
    V3ApprovalResolveResponse,
    V3ArtifactListResponse,
    V3ArtifactRecord,
    V3AttachmentRecord,
    V3MessageRecord,
    V3RunCreateRequest,
    V3RunEventListResponse,
    V3RunEventRecord,
    V3RunRecord,
    V3SessionCreateRequest,
    V3SessionListResponse,
    V3SessionMessageListResponse,
    V3SessionRecord,
)


def _dt(value: str | None) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        return datetime.utcnow()
    return datetime.fromisoformat(raw)


def _session_record(row: dict[str, Any]) -> V3SessionRecord:
    return V3SessionRecord(
        session_id=str(row.get("session_id") or ""),
        user_id=str(row.get("user_id") or "").strip() or None,
        title=str(row.get("title") or "New session"),
        status=str(row.get("status") or "active"),
        summary=str(row.get("summary") or "").strip() or None,
        memory_policy=ScientificMemoryPolicy.model_validate(dict(row.get("memory_policy") or {})),
        knowledge_scope=ScientificKnowledgeScope.model_validate(
            dict(row.get("knowledge_scope") or {})
        ),
        metadata=dict(row.get("metadata") or {}),
        created_at=_dt(row.get("created_at")),
        updated_at=_dt(row.get("updated_at")),
    )


def _message_record(row: dict[str, Any]) -> V3MessageRecord:
    return V3MessageRecord(
        message_id=str(row.get("message_id") or ""),
        session_id=str(row.get("session_id") or ""),
        role=str(row.get("role") or "user"),  # type: ignore[arg-type]
        content=str(row.get("content") or ""),
        attachments=[
            V3AttachmentRecord.model_validate(item) for item in list(row.get("attachments") or [])
        ],
        run_id=str(row.get("run_id") or "").strip() or None,
        metadata=dict(row.get("metadata") or {}),
        created_at=_dt(row.get("created_at")),
    )


def _run_record(row: dict[str, Any]) -> V3RunRecord:
    return V3RunRecord(
        run_id=str(row.get("run_id") or ""),
        session_id=str(row.get("session_id") or ""),
        user_id=str(row.get("user_id") or "").strip() or None,
        workflow_name=str(row.get("workflow_name") or "scientist_workflow"),
        status=str(row.get("status") or "queued"),
        current_step=str(row.get("current_step") or "").strip() or None,
        checkpoint_state=dict(row.get("checkpoint_state") or {}),
        budget_state=dict(row.get("budget_state") or {}),
        response_text=str(row.get("response_text") or "").strip() or None,
        trace_group_id=str(row.get("trace_group_id") or "").strip() or None,
        metrics=dict(row.get("metrics") or {}),
        error=str(row.get("error") or "").strip() or None,
        metadata=dict(row.get("metadata") or {}),
        created_at=_dt(row.get("created_at")),
        updated_at=_dt(row.get("updated_at")),
    )


def _event_record(row: dict[str, Any]) -> V3RunEventRecord:
    return V3RunEventRecord(
        event_id=str(row.get("event_id") or ""),
        run_id=str(row.get("run_id") or ""),
        session_id=str(row.get("session_id") or "").strip() or None,
        user_id=str(row.get("user_id") or "").strip() or None,
        event_kind=str(row.get("event_kind") or ""),
        event_type=str(row.get("event_type") or ""),
        agent_name=str(row.get("agent_name") or "").strip() or None,
        tool_name=str(row.get("tool_name") or "").strip() or None,
        level=str(row.get("level") or "").strip() or None,
        payload=dict(row.get("payload") or {}),
        created_at=_dt(row.get("created_at")),
    )


def _artifact_record(row: dict[str, Any]) -> V3ArtifactRecord:
    return V3ArtifactRecord(
        artifact_id=str(row.get("artifact_id") or ""),
        run_id=str(row.get("run_id") or ""),
        session_id=str(row.get("session_id") or "").strip() or None,
        user_id=str(row.get("user_id") or "").strip() or None,
        kind=str(row.get("kind") or "artifact"),
        title=str(row.get("title") or "").strip() or None,
        path=str(row.get("path") or "").strip() or None,
        source_path=str(row.get("source_path") or "").strip() or None,
        preview_path=str(row.get("preview_path") or "").strip() or None,
        metadata=dict(row.get("metadata") or {}),
        created_at=_dt(row.get("created_at")),
        updated_at=_dt(row.get("updated_at")),
    )


def _approval_record(row: dict[str, Any]) -> V3ApprovalRecord:
    return V3ApprovalRecord(
        approval_id=str(row.get("approval_id") or ""),
        run_id=str(row.get("run_id") or ""),
        session_id=str(row.get("session_id") or "").strip() or None,
        user_id=str(row.get("user_id") or "").strip() or None,
        action_type=str(row.get("action_type") or ""),
        tool_name=str(row.get("tool_name") or "").strip() or None,
        status=str(row.get("status") or "pending"),
        request_payload=dict(row.get("request_payload") or {}),
        resolution=dict(row.get("resolution") or {}),
        created_at=_dt(row.get("created_at")),
        updated_at=_dt(row.get("updated_at")),
    )


def build_v3_router(
    current_user_id: Callable[..., str],
    optional_auth_dependency: Callable[..., Any],
    require_api_key_dependency: Callable[..., Any],
) -> APIRouter:
    router = APIRouter(prefix="/v3")

    def _services(request: FastAPIRequest) -> Any:
        services = getattr(request.app.state, "agentic_v3", None)
        if services is None:
            raise HTTPException(status_code=500, detail="v3 services not initialized")
        return services

    @router.post("/sessions", response_model=V3SessionRecord)
    def create_session_v3(
        req: V3SessionCreateRequest,
        request: FastAPIRequest,
        bisque_auth: dict[str, Any] | None = Depends(optional_auth_dependency),
        _auth: None = Depends(require_api_key_dependency),
    ) -> V3SessionRecord:
        del _auth
        user_id = current_user_id(bisque_auth, allow_anonymous=True)
        services = _services(request)
        title = str(req.title or "").strip() or "New scientist session"
        record = services.sessions.create_session(
            user_id=user_id,
            title=title,
            status=str(req.status or "active"),
            summary=req.summary,
            memory_policy=req.memory_policy.model_dump(mode="json"),
            knowledge_scope=req.knowledge_scope.model_dump(mode="json"),
            metadata=dict(req.metadata or {}),
        )
        return _session_record(record)

    @router.get("/sessions", response_model=V3SessionListResponse)
    def list_sessions_v3(
        request: FastAPIRequest,
        limit: int = 100,
        bisque_auth: dict[str, Any] | None = Depends(optional_auth_dependency),
        _auth: None = Depends(require_api_key_dependency),
    ) -> V3SessionListResponse:
        del _auth
        user_id = current_user_id(bisque_auth, allow_anonymous=True)
        services = _services(request)
        rows = services.sessions.list_sessions(user_id=user_id, limit=max(1, min(int(limit), 500)))
        sessions = [_session_record(row) for row in rows]
        return V3SessionListResponse(count=len(sessions), sessions=sessions)

    @router.get("/sessions/{session_id}", response_model=V3SessionRecord)
    def get_session_v3(
        session_id: str,
        request: FastAPIRequest,
        bisque_auth: dict[str, Any] | None = Depends(optional_auth_dependency),
        _auth: None = Depends(require_api_key_dependency),
    ) -> V3SessionRecord:
        del _auth
        user_id = current_user_id(bisque_auth, allow_anonymous=True)
        services = _services(request)
        row = services.sessions.get_session(session_id=session_id, user_id=user_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return _session_record(row)

    @router.get("/sessions/{session_id}/messages", response_model=V3SessionMessageListResponse)
    def get_session_messages_v3(
        session_id: str,
        request: FastAPIRequest,
        limit: int = 500,
        bisque_auth: dict[str, Any] | None = Depends(optional_auth_dependency),
        _auth: None = Depends(require_api_key_dependency),
    ) -> V3SessionMessageListResponse:
        del _auth
        user_id = current_user_id(bisque_auth, allow_anonymous=True)
        services = _services(request)
        if services.sessions.get_session(session_id=session_id, user_id=user_id) is None:
            raise HTTPException(status_code=404, detail="Session not found")
        rows = services.sessions.list_messages(
            session_id=session_id, user_id=user_id, limit=max(1, min(int(limit), 5000))
        )
        messages = [_message_record(row) for row in rows]
        return V3SessionMessageListResponse(
            session_id=session_id, count=len(messages), messages=messages
        )

    @router.post("/sessions/{session_id}/runs", response_model=V3RunRecord)
    def create_run_v3(
        session_id: str,
        req: V3RunCreateRequest,
        request: FastAPIRequest,
        bisque_auth: dict[str, Any] | None = Depends(optional_auth_dependency),
        _auth: None = Depends(require_api_key_dependency),
    ) -> V3RunRecord:
        del _auth
        user_id = current_user_id(bisque_auth, allow_anonymous=True)
        services = _services(request)
        if services.sessions.get_session(session_id=session_id, user_id=user_id) is None:
            raise HTTPException(status_code=404, detail="Session not found")
        messages = [
            AgenticMessageInput(
                role=message.role,
                content=message.content,
                attachments=[
                    AgenticAttachment.model_validate(
                        item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                    )
                    for item in message.attachments
                ],
                metadata=dict(message.metadata or {}),
            )
            for message in req.messages
        ]
        if not messages and str(req.goal or "").strip():
            messages = [
                AgenticMessageInput(
                    role="user",
                    content=str(req.goal or "").strip(),
                    attachments=[],
                    metadata={},
                )
            ]
        if not messages:
            raise HTTPException(
                status_code=400, detail="At least one message or a goal is required"
            )
        run_request = AgenticRunRequest(
            goal=req.goal,
            messages=messages,
            file_ids=list(req.file_ids or []),
            resource_uris=list(req.resource_uris or []),
            dataset_uris=list(req.dataset_uris or []),
            selected_tool_names=list(req.selected_tool_names or []),
            knowledge_context=(
                req.knowledge_context.model_dump(mode="json")
                if req.knowledge_context is not None
                else {}
            ),
            selection_context=(
                req.selection_context.model_dump(mode="json")
                if req.selection_context is not None
                else {}
            ),
            workflow_hint=(
                req.workflow_hint.model_dump(mode="json") if req.workflow_hint is not None else {}
            ),
            reasoning_mode=req.reasoning_mode,
            budgets=AgenticRunBudget.model_validate(req.budgets.model_dump(mode="json")),
            metadata={
                **dict(req.metadata or {}),
                "debug": bool(req.debug),
            },
        )
        row = services.orchestrator.start_run(
            session_id=session_id, user_id=user_id, request=run_request
        )
        return _run_record(row)

    @router.get("/runs/{run_id}", response_model=V3RunRecord)
    def get_run_v3(
        run_id: str,
        request: FastAPIRequest,
        bisque_auth: dict[str, Any] | None = Depends(optional_auth_dependency),
        _auth: None = Depends(require_api_key_dependency),
    ) -> V3RunRecord:
        del _auth
        user_id = current_user_id(bisque_auth, allow_anonymous=True)
        services = _services(request)
        row = services.runs.get_run(run_id=run_id, user_id=user_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return _run_record(row)

    @router.get("/runs/{run_id}/events", response_model=V3RunEventListResponse)
    def get_run_events_v3(
        run_id: str,
        request: FastAPIRequest,
        limit: int = 500,
        bisque_auth: dict[str, Any] | None = Depends(optional_auth_dependency),
        _auth: None = Depends(require_api_key_dependency),
    ) -> V3RunEventListResponse:
        del _auth
        user_id = current_user_id(bisque_auth, allow_anonymous=True)
        services = _services(request)
        if services.runs.get_run(run_id=run_id, user_id=user_id) is None:
            raise HTTPException(status_code=404, detail="Run not found")
        rows = services.events.list_events(
            run_id=run_id, user_id=user_id, limit=max(1, min(int(limit), 5000))
        )
        events = [_event_record(row) for row in rows]
        return V3RunEventListResponse(run_id=run_id, count=len(events), events=events)

    @router.get("/runs/{run_id}/artifacts", response_model=V3ArtifactListResponse)
    def get_run_artifacts_v3(
        run_id: str,
        request: FastAPIRequest,
        limit: int = 500,
        bisque_auth: dict[str, Any] | None = Depends(optional_auth_dependency),
        _auth: None = Depends(require_api_key_dependency),
    ) -> V3ArtifactListResponse:
        del _auth
        user_id = current_user_id(bisque_auth, allow_anonymous=True)
        services = _services(request)
        if services.runs.get_run(run_id=run_id, user_id=user_id) is None:
            raise HTTPException(status_code=404, detail="Run not found")
        rows = services.artifacts.list_artifacts(
            run_id=run_id, user_id=user_id, limit=max(1, min(int(limit), 5000))
        )
        artifacts = [_artifact_record(row) for row in rows]
        return V3ArtifactListResponse(run_id=run_id, count=len(artifacts), artifacts=artifacts)

    @router.post("/runs/{run_id}/approvals/{approval_id}", response_model=V3ApprovalResolveResponse)
    def resolve_run_approval_v3(
        run_id: str,
        approval_id: str,
        req: V3ApprovalResolveRequest,
        request: FastAPIRequest,
        bisque_auth: dict[str, Any] | None = Depends(optional_auth_dependency),
        _auth: None = Depends(require_api_key_dependency),
    ) -> V3ApprovalResolveResponse:
        del _auth
        user_id = current_user_id(bisque_auth, allow_anonymous=True)
        services = _services(request)
        run = services.runs.get_run(run_id=run_id, user_id=user_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        approval = services.approvals.get_approval(approval_id=approval_id, user_id=user_id)
        if approval is None or str(approval.get("run_id") or "") != run_id:
            raise HTTPException(status_code=404, detail="Approval not found")
        approval_row, run_row = services.orchestrator.resolve_approval(
            approval_id=approval_id,
            user_id=user_id,
            decision=req.decision,
            note=req.note,
            metadata=dict(req.metadata or {}),
        )
        return V3ApprovalResolveResponse(
            approval=_approval_record(approval_row),
            run=_run_record(run_row),
        )

    return router
