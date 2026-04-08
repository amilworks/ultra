"""Bounded workgraph scheduler for chat-centric agent execution."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from .contracts import (
    AgentResult,
    GraphEventRecord,
    RouteDecision,
    VerificationIssue,
    VerificationReport,
    WorkGraphExecution,
)


RouteFn = Callable[[str, Any], Awaitable[RouteDecision]]
SolveFn = Callable[..., Awaitable[AgentResult]]
VerifyFn = Callable[..., Awaitable[VerificationReport]]
GraphEventFn = Callable[[GraphEventRecord], None]


class WorkGraphScheduler:
    """Execute a bounded route->solve->verify graph with one optional retry pass."""

    def __init__(self, *, max_retries: int = 1) -> None:
        self.max_retries = max(0, int(max_retries))

    async def execute(
        self,
        *,
        user_text: str,
        messages: list[dict[str, str]],
        tool_state: Any,
        run_config: Any,
        mcp_servers: list[Any],
        route_fn: RouteFn,
        solve_fn: SolveFn,
        verify_fn: VerifyFn,
        workflow_kind: str = "interactive_chat",
        event_callback: GraphEventFn | None = None,
    ) -> WorkGraphExecution:
        events: list[dict[str, Any]] = []
        route_node = "triage"
        solve_node = "solve"
        route_role = "triage"

        self._emit(
            events,
            GraphEventRecord(
                workflow_kind=workflow_kind,
                phase=route_node,
                status="started",
                agent_role=route_role,
                node=route_node,
                message="Selecting the specialist agents for this run.",
            ),
            event_callback=event_callback,
        )
        route = await route_fn(user_text, run_config)
        selected_domains = [str(token).strip() for token in route.selected_domains if str(token).strip()]
        if not selected_domains:
            selected_domains = ["core"]
            route = RouteDecision(
                selected_domains=selected_domains,
                confidence=max(0.0, float(route.confidence)),
                reason=str(route.reason or "empty_selection_fallback"),
                used_model_classifier=bool(route.used_model_classifier),
            )
        self._emit(
            events,
            GraphEventRecord(
                workflow_kind=workflow_kind,
                phase=route_node,
                status="completed",
                agent_role=route_role,
                node=route_node,
                message="Specialists selected.",
                payload={
                    "selected_domains": list(selected_domains),
                    "confidence": float(route.confidence),
                    "reason": str(route.reason or ""),
                },
            ),
            event_callback=event_callback,
        )

        solve_results = await self._run_solve_domains(
            selected_domains=selected_domains,
            user_text=user_text,
            messages=messages,
            tool_state=tool_state,
            run_config=run_config,
            mcp_servers=mcp_servers,
            solve_fn=solve_fn,
            events=events,
            retry_feedback_by_domain=None,
            workflow_kind=workflow_kind,
            solve_node=solve_node,
            event_callback=event_callback,
        )
        if any(
            bool(dict(result.metadata or {}).get("interrupted"))
            for result in solve_results.values()
            if isinstance(result, AgentResult)
        ):
            self._emit(
                events,
                GraphEventRecord(
                    workflow_kind=workflow_kind,
                    phase="verify",
                    status="completed",
                    agent_role="verifier",
                    node="verify",
                    message="Execution paused for approval before verification.",
                    payload={"paused_for_approval": True},
                ),
                event_callback=event_callback,
            )
            return WorkGraphExecution(
                route=route,
                agent_results=solve_results,
                verification=VerificationReport(passed=True, notes=["paused_for_approval"]),
                retry_count=0,
                events=events,
            )

        verification = await self._safe_verify(
            verify_fn=verify_fn,
            user_text=user_text,
            route=route,
            agent_results=solve_results,
            run_config=run_config,
            events=events,
            workflow_kind=workflow_kind,
            event_callback=event_callback,
        )
        retry_count = 0

        if self.max_retries > 0 and verification.retry_domains:
            retry_targets = [
                domain_id
                for domain_id in selected_domains
                if domain_id in {str(token).strip() for token in verification.retry_domains}
            ]
            if retry_targets:
                retry_count += 1
                self._emit(
                    events,
                    GraphEventRecord(
                        workflow_kind=workflow_kind,
                        phase="repair",
                        status="started",
                        agent_role="verifier",
                        node="retry",
                        message="Retrying flagged specialist tasks.",
                        payload={"retry_domains": list(retry_targets)},
                    ),
                    event_callback=event_callback,
                )
                retry_feedback_by_domain: dict[str, str] = {}
                for issue in verification.issues:
                    domain_token = str(issue.domain_id or "").strip()
                    if not domain_token or domain_token not in retry_targets:
                        continue
                    feedback = str(issue.message or "").strip()
                    if not feedback:
                        continue
                    existing = retry_feedback_by_domain.get(domain_token, "")
                    retry_feedback_by_domain[domain_token] = (
                        f"{existing}\n- {feedback}" if existing else f"- {feedback}"
                    )
                retried = await self._run_solve_domains(
                    selected_domains=retry_targets,
                    user_text=user_text,
                    messages=messages,
                    tool_state=tool_state,
                    run_config=run_config,
                    mcp_servers=mcp_servers,
                    solve_fn=solve_fn,
                    events=events,
                    retry_feedback_by_domain=retry_feedback_by_domain,
                    workflow_kind=workflow_kind,
                    solve_node=solve_node,
                    event_callback=event_callback,
                )
                solve_results.update(retried)
                verification = await self._safe_verify(
                    verify_fn=verify_fn,
                    user_text=user_text,
                    route=route,
                    agent_results=solve_results,
                    run_config=run_config,
                    events=events,
                    workflow_kind=workflow_kind,
                    event_callback=event_callback,
                )
                self._emit(
                    events,
                    GraphEventRecord(
                        workflow_kind=workflow_kind,
                        phase="repair",
                        status="completed",
                        agent_role="verifier",
                        node="retry",
                        message="Repair pass finished.",
                        payload={"retry_domains": list(retry_targets)},
                    ),
                    event_callback=event_callback,
                )

        return WorkGraphExecution(
            route=route,
            agent_results=solve_results,
            verification=verification,
            retry_count=retry_count,
            events=events,
        )

    async def _run_solve_domains(
        self,
        *,
        selected_domains: list[str],
        user_text: str,
        messages: list[dict[str, str]],
        tool_state: Any,
        run_config: Any,
        mcp_servers: list[Any],
        solve_fn: SolveFn,
        events: list[dict[str, Any]],
        retry_feedback_by_domain: dict[str, str] | None,
        workflow_kind: str,
        solve_node: str,
        event_callback: GraphEventFn | None,
    ) -> dict[str, AgentResult]:
        async def _solve_one(domain_id: str) -> tuple[str, AgentResult]:
            started = time.monotonic()
            scoped_tool_state = (
                tool_state.spawn_scope(f"{workflow_kind}:{domain_id}")
                if hasattr(tool_state, "spawn_scope")
                else tool_state
            )
            self._emit(
                events,
                GraphEventRecord(
                    workflow_kind=workflow_kind,
                    phase=solve_node,
                    status="started",
                    agent_role="domain_specialist",
                    node=solve_node,
                    domain_id=str(domain_id),
                    scope_id=str(getattr(scoped_tool_state, "scope_id", "") or "") or None,
                    message=f"Running {domain_id} specialist.",
                ),
                event_callback=event_callback,
            )
            try:
                result = await solve_fn(
                    domain_id=domain_id,
                    user_text=user_text,
                    messages=messages,
                    tool_state=scoped_tool_state,
                    run_config=run_config,
                    mcp_servers=mcp_servers,
                    retry_feedback=(
                        retry_feedback_by_domain.get(domain_id)
                        if isinstance(retry_feedback_by_domain, dict)
                        else None
                    ),
                )
            except Exception as exc:
                result = AgentResult(
                    domain_id=str(domain_id),
                    success=False,
                    summary="",
                    raw_output="",
                    error=str(exc),
                )
                self._emit(
                    events,
                    GraphEventRecord(
                        workflow_kind=workflow_kind,
                        phase=solve_node,
                        status="failed",
                        agent_role="domain_specialist",
                        node=solve_node,
                        domain_id=str(domain_id),
                        scope_id=str(getattr(scoped_tool_state, "scope_id", "") or "") or None,
                        message=str(exc),
                        payload={"error": str(exc)},
                    ),
                    event_callback=event_callback,
                )
                return str(domain_id), result

            elapsed = round(time.monotonic() - started, 6)
            model_route = (
                dict(result.metadata.get("model_route") or {})
                if isinstance(result.metadata, dict) and isinstance(result.metadata.get("model_route"), dict)
                else {}
            )
            payload = {
                "success": bool(result.success),
                "duration_seconds": elapsed,
                "tool_calls": int(result.tool_calls or 0),
            }
            if model_route:
                payload["model_route"] = model_route
                for key in (
                    "requested_provider",
                    "requested_model",
                    "actual_provider",
                    "actual_model",
                    "fallback_used",
                    "fallback_reason",
                    "multimodal_enabled",
                    "multimodal_image_count",
                    "available_multimodal_image_count",
                    "breaker_state",
                ):
                    if key in model_route:
                        payload[key] = model_route.get(key)
            self._emit(
                events,
                GraphEventRecord(
                    workflow_kind=workflow_kind,
                    phase=solve_node,
                    status=("completed" if result.success else "failed"),
                    agent_role="domain_specialist",
                    node=solve_node,
                    domain_id=str(domain_id),
                    scope_id=str(getattr(scoped_tool_state, "scope_id", "") or "") or None,
                    message=(result.summary[:240] if result.summary else result.error or None),
                    payload=payload,
                ),
                event_callback=event_callback,
            )
            return str(domain_id), result

        tasks = [_solve_one(domain_id) for domain_id in selected_domains]
        outputs: dict[str, AgentResult] = {}
        for domain_id, result in await asyncio.gather(*tasks):
            outputs[domain_id] = result
        return outputs

    async def _safe_verify(
        self,
        *,
        verify_fn: VerifyFn,
        user_text: str,
        route: RouteDecision,
        agent_results: dict[str, AgentResult],
        run_config: Any,
        events: list[dict[str, Any]],
        workflow_kind: str,
        event_callback: GraphEventFn | None,
    ) -> VerificationReport:
        self._emit(
            events,
            GraphEventRecord(
                workflow_kind=workflow_kind,
                phase="verify",
                status="started",
                agent_role="verifier",
                node="verify",
                message="Checking specialist outputs for conflicts and missing evidence.",
            ),
            event_callback=event_callback,
        )
        try:
            report = await verify_fn(
                user_text=user_text,
                route=route,
                agent_results=agent_results,
                run_config=run_config,
            )
        except Exception as exc:
            report = VerificationReport(
                passed=False,
                issues=[
                    VerificationIssue(
                        code="verifier_runtime_error",
                        severity="high",
                        message=str(exc),
                        correctable=False,
                    )
                ],
                retry_domains=[],
                notes=["Verifier failed; continuing with available domain outputs."],
            )
            self._emit(
                events,
                GraphEventRecord(
                    workflow_kind=workflow_kind,
                    phase="verify",
                    status="failed",
                    agent_role="verifier",
                    node="verify",
                    message=str(exc),
                    payload={"error": str(exc)},
                ),
                event_callback=event_callback,
            )
            return report

        self._emit(
            events,
            GraphEventRecord(
                workflow_kind=workflow_kind,
                phase="verify",
                status="completed",
                agent_role="verifier",
                node="verify",
                message=(
                    "Verification passed."
                    if bool(report.passed)
                    else "Verification requested revisions."
                ),
                payload={
                    "passed": bool(report.passed),
                    "issue_count": len(report.issues),
                    "retry_domains": list(report.retry_domains),
                },
            ),
            event_callback=event_callback,
        )
        return report

    @staticmethod
    def _emit(
        events: list[dict[str, Any]],
        record: GraphEventRecord,
        *,
        event_callback: GraphEventFn | None,
    ) -> None:
        payload = record.model_dump(mode="json")
        payload.setdefault("event", WorkGraphScheduler._legacy_event_name(record))
        payload.setdefault("ts", datetime.utcnow().isoformat() + "Z")
        events.append(payload)
        if event_callback is not None:
            event_callback(record)

    @staticmethod
    def _legacy_event_name(record: GraphEventRecord) -> str:
        """Preserve legacy event names for tests and existing consumers."""

        phase = str(record.phase or "").strip().lower()
        status = str(record.status or "").strip().lower()
        if phase == "verify":
            return f"agent_verification_{status or 'progress'}"
        if phase in {"triage", "planner"}:
            return f"agent_routing_{status or 'progress'}"
        if phase in {"solve", "task"}:
            return f"agent_solve_{status or 'progress'}"
        if phase == "repair":
            return f"agent_repair_{status or 'progress'}"
        return f"agent_{phase or 'graph'}_{status or 'progress'}"
