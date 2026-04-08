"""Agno-native v3 services built on the shared Agno chat runtime."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.agentic.db import AgenticDb
from src.agentic.models import AgenticAttachment, AgenticMessageInput, AgenticRunRequest
from src.agentic.repositories import (
    ApprovalRepository,
    ArtifactRepository,
    EventRepository,
    RunRepository,
    ScientificNoteRepository,
    SessionRepository,
)
from src.config import Settings

from .knowledge import ScientificKnowledgeScope
from .memory import ScientificMemoryPolicy
from .runtime import AgnoChatRuntime, AgnoChatRuntimeResult


@dataclass(frozen=True)
class AgnoV3Services:
    db: AgenticDb
    sessions: SessionRepository
    runs: RunRepository
    events: EventRepository
    artifacts: ArtifactRepository
    approvals: ApprovalRepository
    notes: ScientificNoteRepository
    orchestrator: "AgnoV3WorkflowService"


class AgnoV3WorkflowService:
    """Persisted Session -> Run -> Event / Artifact / Approval service."""

    def __init__(
        self,
        *,
        settings: Settings,
        sessions: SessionRepository,
        runs: RunRepository,
        events: EventRepository,
        artifacts: ArtifactRepository,
        approvals: ApprovalRepository,
        artifact_root: Path,
        runtime: AgnoChatRuntime | None = None,
    ) -> None:
        self.settings = settings
        self.sessions = sessions
        self.runs = runs
        self.events = events
        self.artifacts = artifacts
        self.approvals = approvals
        self.artifact_root = artifact_root.resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.runtime = runtime or AgnoChatRuntime(settings=settings)

    def start_run(
        self,
        *,
        session_id: str,
        user_id: str,
        request: AgenticRunRequest,
        run_id: str | None = None,
        session_title: str | None = None,
    ) -> dict[str, Any]:
        title = str(
            session_title or request.goal or self._latest_user_text(request.messages) or "New session"
        ).strip() or "New session"
        existing_session = self.sessions.get_session(session_id=session_id, user_id=user_id) or {}
        resolved_memory_policy = ScientificMemoryPolicy.model_validate(
            dict(existing_session.get("memory_policy") or {})
        )
        resolved_knowledge_scope = ScientificKnowledgeScope.model_validate(
            dict(existing_session.get("knowledge_scope") or {})
        )
        self.sessions.ensure_session(
            session_id=session_id,
            user_id=user_id,
            title=title,
            memory_policy=resolved_memory_policy.model_dump(mode="json"),
            knowledge_scope=resolved_knowledge_scope.model_dump(mode="json"),
            metadata={"runtime": "agno"},
        )
        self._sync_request_messages(session_id=session_id, user_id=user_id, messages=request.messages)

        resolved_run_id = str(run_id or uuid4().hex)
        route_context = {
            "selected_tool_names": list(request.selected_tool_names or []),
            "workflow_hint": dict(request.workflow_hint or {}),
            "selection_context": dict(request.selection_context or {}),
            "knowledge_context": dict(request.knowledge_context or {}),
            "memory_policy": resolved_memory_policy.model_dump(mode="json"),
            "knowledge_scope": resolved_knowledge_scope.model_dump(mode="json"),
            "reasoning_mode": str(request.reasoning_mode or "deep"),
        }
        workflow_id = str((request.workflow_hint or {}).get("id") or "").strip().lower()
        workflow_name = "agno_pro_mode" if workflow_id == "pro_mode" else "agno_single_agent"
        row = self.runs.create_run(
            run_id=resolved_run_id,
            session_id=session_id,
            user_id=user_id,
            workflow_name=workflow_name,
            status="queued",
            current_step="queued",
            checkpoint_state={"phase": "queued", **route_context},
            budget_state=self._budget_state(request),
            trace_group_id=session_id,
            metadata={"runtime": "agno", "planner_version": "agno_v1", **dict(request.metadata or {})},
        )
        self.events.append_event(
            run_id=resolved_run_id,
            session_id=session_id,
            user_id=user_id,
            event_kind="run",
            event_type="run.started",
            payload={"status": "queued", "workflow_name": workflow_name},
        )
        return asyncio.run(
            self._execute_runtime_run(
                run_id=resolved_run_id,
                session_id=session_id,
                user_id=user_id,
                request=request,
                existing_run=row,
                hitl_resume=None,
            )
        )

    def resolve_approval(
        self,
        *,
        approval_id: str,
        user_id: str,
        decision: str,
        note: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        approval = self.approvals.get_approval(approval_id=approval_id, user_id=user_id)
        if approval is None:
            raise ValueError(f"Approval not found: {approval_id}")
        run_id = str(approval.get("run_id") or "").strip()
        run = self.runs.get_run(run_id=run_id, user_id=user_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")
        checkpoint_state = dict(run.get("checkpoint_state") or {}) if isinstance(run.get("checkpoint_state"), dict) else {}
        pending_hitl = dict(checkpoint_state.get("pending_hitl") or {}) if isinstance(checkpoint_state.get("pending_hitl"), dict) else {}
        if not pending_hitl:
            raise ValueError("Run is missing pending approval state")

        normalized_decision = str(decision or "").strip().lower() or "approve"
        approval = self.approvals.update_approval(
            approval_id=approval_id,
            user_id=user_id,
            status=("approved" if normalized_decision == "approve" else "rejected"),
            resolution={
                "decision": normalized_decision,
                "note": str(note or "").strip() or None,
                "metadata": dict(metadata or {}),
            },
        ) or approval

        request_payload = dict(approval.get("request_payload") or {}) if isinstance(approval.get("request_payload"), dict) else {}
        request_model = AgenticRunRequest.model_validate(
            dict(request_payload.get("request") or {})
            if isinstance(request_payload.get("request"), dict)
            else {
                "messages": [],
                "selected_tool_names": list(pending_hitl.get("selected_tool_names") or []),
                "workflow_hint": dict(pending_hitl.get("workflow_hint") or {}),
                "reasoning_mode": str((run.get("budget_state") or {}).get("reasoning_mode") or "deep"),
                "budgets": self._budget_state_from_row(run),
            }
        )
        updated_run = asyncio.run(
            self._execute_runtime_run(
                run_id=run_id,
                session_id=str(run.get("session_id") or ""),
                user_id=user_id,
                request=request_model,
                existing_run=run,
                hitl_resume={
                    "decision": normalized_decision,
                    "note": str(note or "").strip() or None,
                    "pending_hitl": pending_hitl,
                },
            )
        )
        return approval, updated_run

    async def _execute_runtime_run(
        self,
        *,
        run_id: str,
        session_id: str,
        user_id: str,
        request: AgenticRunRequest,
        existing_run: dict[str, Any],
        hitl_resume: dict[str, Any] | None,
    ) -> dict[str, Any]:
        del existing_run
        self.runs.update_run(
            run_id=run_id,
            user_id=user_id,
            status="running",
            current_step="running",
        )

        def _persist_runtime_event(payload: dict[str, Any]) -> None:
            event_kind = str(payload.get("kind") or "graph")
            explicit_type = str(payload.get("event_type") or "").strip()
            phase = str(payload.get("phase") or "").strip().lower()
            status = str(payload.get("status") or "").strip().lower()
            event_type = explicit_type or (f"{phase}.{status}" if phase and status else f"{event_kind}.updated")
            self.events.append_event(
                run_id=run_id,
                session_id=session_id,
                user_id=user_id,
                event_kind=event_kind,
                event_type=event_type,
                tool_name=str(payload.get("tool_name") or payload.get("tool") or "").strip() or None,
                payload=payload,
            )

        session_row = self.sessions.get_session(session_id=session_id, user_id=user_id) or {}
        resolved_memory_policy = ScientificMemoryPolicy.model_validate(
            dict(session_row.get("memory_policy") or {})
        )
        resolved_knowledge_scope = ScientificKnowledgeScope.model_validate(
            dict(session_row.get("knowledge_scope") or {})
        )
        result = await self.runtime.run(
            messages=self._runtime_messages(
                request=request,
                session_id=session_id,
                user_id=user_id,
            ),
            uploaded_files=[],
            max_tool_calls=int(request.budgets.max_tool_calls),
            max_runtime_seconds=int(request.budgets.max_runtime_seconds),
            reasoning_mode=str(request.reasoning_mode or "deep"),
            conversation_id=session_id,
            run_id=run_id,
            user_id=user_id,
            event_callback=_persist_runtime_event,
            selected_tool_names=list(request.selected_tool_names or []),
            workflow_hint=dict(request.workflow_hint or {}),
            selection_context=dict(request.selection_context or {}),
            knowledge_context=dict(request.knowledge_context or {}),
            memory_policy=resolved_memory_policy.model_dump(mode="json"),
            knowledge_scope=resolved_knowledge_scope.model_dump(mode="json"),
            hitl_resume=hitl_resume,
            debug=bool((request.metadata or {}).get("debug")),
        )
        return self._finalize_runtime_result(
            run_id=run_id,
            session_id=session_id,
            user_id=user_id,
            request=request,
            result=result,
        )

    def _finalize_runtime_result(
        self,
        *,
        run_id: str,
        session_id: str,
        user_id: str,
        request: AgenticRunRequest,
        result: AgnoChatRuntimeResult,
    ) -> dict[str, Any]:
        metadata = dict(result.metadata or {})
        debug = dict(metadata.get("debug") or {}) if isinstance(metadata.get("debug"), dict) else {}
        interrupted = bool(metadata.get("interrupted"))
        resume_decision = str(metadata.get("resume_decision") or "").strip().lower()
        pending_hitl = dict(metadata.get("pending_hitl") or {}) if isinstance(metadata.get("pending_hitl"), dict) else {}
        checkpoint_state = {
            "selected_tool_names": list(request.selected_tool_names or []),
            "workflow_hint": dict(request.workflow_hint or {}),
            "selection_context": dict(request.selection_context or {}),
            "knowledge_context": dict(request.knowledge_context or {}),
            "selected_domains": list(result.selected_domains or []),
        }
        metrics = {
            "selected_domains": list(result.selected_domains or []),
            "tool_calls": int(result.tool_calls),
        }

        if interrupted:
            approval = self.approvals.create_approval(
                run_id=run_id,
                session_id=session_id,
                user_id=user_id,
                action_type="tool_execution",
                tool_name=", ".join(
                    str(item.get("tool_name") or "").strip()
                    for item in list(pending_hitl.get("interruptions") or [])
                    if isinstance(item, dict) and str(item.get("tool_name") or "").strip()
                )
                or None,
                request_payload={
                    "request": request.model_dump(mode="json"),
                    "pending_hitl": pending_hitl,
                    "debug": debug,
                },
            )
            pending_hitl["approval_id"] = str(approval.get("approval_id") or "").strip() or None
            self.runs.update_run(
                run_id=run_id,
                user_id=user_id,
                status="waiting_for_input",
                current_step="approval_pending",
                checkpoint_state={
                    **checkpoint_state,
                    "phase": "pending_approval",
                    "pending_hitl": pending_hitl,
                },
                budget_state=self._budget_state(request),
                response_text=result.response_text,
                metrics=metrics,
                metadata={"runtime": "agno", "planner_version": "agno_v1", **metadata},
            )
            self.events.append_event(
                run_id=run_id,
                session_id=session_id,
                user_id=user_id,
                event_kind="approval",
                event_type="approval.requested",
                tool_name=str(approval.get("tool_name") or "").strip() or None,
                payload={
                    "approval_id": approval.get("approval_id"),
                    "pending_hitl": pending_hitl,
                },
                level="warning",
            )
        elif resume_decision == "reject":
            self.runs.update_run(
                run_id=run_id,
                user_id=user_id,
                status="canceled",
                current_step="approval_rejected",
                checkpoint_state={
                    **checkpoint_state,
                    "phase": "approval_rejected",
                    "pending_hitl": None,
                    "resume_decision": "reject",
                },
                budget_state=self._budget_state(request),
                response_text=result.response_text,
                metrics=metrics,
                metadata={"runtime": "agno", "planner_version": "agno_v1", **metadata},
                error=str(result.response_text or "").strip() or "Approval rejected by user.",
            )
            self.events.append_event(
                run_id=run_id,
                session_id=session_id,
                user_id=user_id,
                event_kind="run",
                event_type="run.completed",
                payload={"status": "canceled", "resume_decision": "reject"},
                level="warning",
            )
        else:
            self.runs.update_run(
                run_id=run_id,
                user_id=user_id,
                status="succeeded",
                current_step="completed",
                checkpoint_state={
                    **checkpoint_state,
                    "phase": "completed",
                    "pending_hitl": None,
                    "resume_decision": resume_decision or None,
                },
                budget_state=self._budget_state(request),
                response_text=result.response_text,
                metrics=metrics,
                metadata={"runtime": "agno", "planner_version": "agno_v1", **metadata},
            )
            self.events.append_event(
                run_id=run_id,
                session_id=session_id,
                user_id=user_id,
                event_kind="run",
                event_type="run.completed",
                payload={"status": "succeeded"},
            )
            self.artifacts.upsert_artifact(
                artifact_id=f"{run_id}:final_response",
                run_id=run_id,
                session_id=session_id,
                user_id=user_id,
                kind="response",
                title="Final response",
                path=None,
                metadata={"response_text": result.response_text},
            )

        assistant_text = str(result.response_text or "").strip()
        if assistant_text:
            self.sessions.append_message(
                session_id=session_id,
                user_id=user_id,
                role="assistant",
                content=assistant_text,
                attachments=[],
                run_id=run_id,
                metadata={"runtime": "agno", **metadata},
            )

        updated = self.runs.get_run(run_id=run_id, user_id=user_id)
        return updated or {}

    def _sync_request_messages(
        self,
        *,
        session_id: str,
        user_id: str,
        messages: list[AgenticMessageInput],
    ) -> None:
        for message in messages:
            self.sessions.append_message(
                session_id=session_id,
                user_id=user_id,
                role=str(message.role or "user"),
                content=str(message.content or ""),
                attachments=[item.model_dump(mode="json") for item in list(message.attachments or [])],
                metadata=dict(message.metadata or {}),
            )

    @staticmethod
    def _latest_user_text(messages: list[AgenticMessageInput]) -> str:
        for message in reversed(messages):
            if str(message.role or "").strip().lower() != "user":
                continue
            text = str(message.content or "").strip()
            if text:
                return text
        return ""

    @staticmethod
    def _budget_state(request: AgenticRunRequest) -> dict[str, Any]:
        return {
            "max_tool_calls": int(request.budgets.max_tool_calls),
            "max_runtime_seconds": int(request.budgets.max_runtime_seconds),
            "reasoning_mode": str(request.reasoning_mode or "deep"),
            "selected_tool_names": list(request.selected_tool_names or []),
            "workflow_hint": dict(request.workflow_hint or {}),
            "selection_context": dict(request.selection_context or {}),
            "knowledge_context": dict(request.knowledge_context or {}),
        }

    @staticmethod
    def _budget_state_from_row(run: dict[str, Any]) -> dict[str, Any]:
        return dict(run.get("budget_state") or {}) if isinstance(run.get("budget_state"), dict) else {}

    @staticmethod
    def _context_message(label: str, payload: dict[str, Any] | list[str]) -> dict[str, str] | None:
        if isinstance(payload, dict):
            if not payload:
                return None
            body = json.dumps(payload, ensure_ascii=False, indent=2)
        elif isinstance(payload, list):
            if not payload:
                return None
            body = json.dumps(payload, ensure_ascii=False, indent=2)
        else:
            return None
        return {"role": "system", "content": f"{label}:\n{body}"}

    def _runtime_messages(
        self,
        *,
        request: AgenticRunRequest,
        session_id: str,
        user_id: str,
    ) -> list[dict[str, str]]:
        stored_messages = self.sessions.list_messages(session_id=session_id, user_id=user_id, limit=200)
        messages = [
            {
                "role": str(message.get("role") or "user"),
                "content": str(message.get("content") or ""),
            }
            for message in stored_messages
            if str(message.get("content") or "").strip()
        ]
        attachments = [item.model_dump(mode="json") for item in self._attachments_for_request(request)]
        context_messages = [
            self._context_message("Attachments", attachments),
        ]
        hydrated = [msg for msg in context_messages if msg is not None]
        hydrated.extend(messages)
        return hydrated

    @staticmethod
    def _attachments_for_request(request: AgenticRunRequest) -> list[AgenticAttachment]:
        attachments: list[AgenticAttachment] = []
        for file_id in list(request.file_ids or []):
            attachments.append(AgenticAttachment(kind="file_id", value=str(file_id)))
        for resource_uri in list(request.resource_uris or []):
            attachments.append(AgenticAttachment(kind="resource_uri", value=str(resource_uri)))
        for dataset_uri in list(request.dataset_uris or []):
            attachments.append(AgenticAttachment(kind="dataset_uri", value=str(dataset_uri)))
        return attachments


def build_agno_v3_services(
    *,
    settings: Settings,
    artifact_root: Path,
    runtime: AgnoChatRuntime | None = None,
) -> AgnoV3Services:
    db_target = str(getattr(settings, "run_store_path", "data/runs.db") or "data/runs.db").strip() or "data/runs.db"
    db = AgenticDb(db_target)
    sessions = SessionRepository(db)
    runs = RunRepository(db)
    events = EventRepository(db)
    artifacts = ArtifactRepository(db)
    approvals = ApprovalRepository(db)
    notes = ScientificNoteRepository(db)
    orchestrator = AgnoV3WorkflowService(
        settings=settings,
        sessions=sessions,
        runs=runs,
        events=events,
        artifacts=artifacts,
        approvals=approvals,
        artifact_root=artifact_root,
        runtime=runtime,
    )
    return AgnoV3Services(
        db=db,
        sessions=sessions,
        runs=runs,
        events=events,
        artifacts=artifacts,
        approvals=approvals,
        notes=notes,
        orchestrator=orchestrator,
    )
