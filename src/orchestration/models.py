from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass(frozen=True)
class ToolStep:
    id: str
    tool_name: str
    arguments: dict[str, Any]
    description: str | None = None
    timeout_seconds: int | None = None
    retries: int = 0

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ToolStep:
        """Create a normalized `ToolStep` from loosely typed input data.

        Parameters
        ----------
        data : dict[str, Any]
            Input argument.

        Returns
        -------
        'ToolStep'
            Computed result.
        """
        return ToolStep(
            id=str(data.get("id") or uuid4()),
            tool_name=str(data["tool_name"]),
            arguments=dict(data.get("arguments") or {}),
            description=data.get("description"),
            timeout_seconds=data.get("timeout_seconds"),
            retries=int(data.get("retries") or 0),
        )


@dataclass(frozen=True)
class WorkflowPlan:
    goal: str
    steps: list[ToolStep]

    @staticmethod
    def from_dict(data: dict[str, Any]) -> WorkflowPlan:
        """Build a `WorkflowPlan` and recursively coerce each step payload.

        Parameters
        ----------
        data : dict[str, Any]
            Input argument.

        Returns
        -------
        'WorkflowPlan'
            Computed result.
        """
        steps_in = data.get("steps") or []
        return WorkflowPlan(
            goal=str(data.get("goal") or ""),
            steps=[ToolStep.from_dict(s) for s in steps_in],
        )


@dataclass
class WorkflowRun:
    run_id: str
    goal: str
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    plan: WorkflowPlan | None = None
    error: str | None = None
    workflow_kind: str = "workflow_plan"
    mode: str = "durable"
    parent_run_id: str | None = None
    planner_version: str | None = None
    agent_role: str | None = None
    checkpoint_state: dict[str, Any] | None = None
    budget_state: dict[str, Any] | None = None
    trace_group_id: str | None = None
    workflow_profile: str | None = None
    pedagogy_level: str | None = None

    @staticmethod
    def new(
        goal: str,
        plan: WorkflowPlan | None = None,
        *,
        workflow_kind: str = "workflow_plan",
        mode: str = "durable",
        parent_run_id: str | None = None,
        planner_version: str | None = None,
        agent_role: str | None = None,
        checkpoint_state: dict[str, Any] | None = None,
        budget_state: dict[str, Any] | None = None,
        trace_group_id: str | None = None,
        workflow_profile: str | None = None,
        pedagogy_level: str | None = None,
    ) -> WorkflowRun:
        """Create a new pending run with generated ID and synchronized timestamps.

        Parameters
        ----------
        goal : str
            Input argument.
        plan : WorkflowPlan | None, optional
            Input argument.

        Returns
        -------
        'WorkflowRun'
            Computed result.
        """
        now = datetime.utcnow()
        return WorkflowRun(
            run_id=str(uuid4()),
            goal=goal,
            status=RunStatus.PENDING,
            created_at=now,
            updated_at=now,
            plan=plan,
            workflow_kind=str(workflow_kind or "workflow_plan"),
            mode=str(mode or "durable"),
            parent_run_id=(str(parent_run_id).strip() if parent_run_id else None),
            planner_version=(str(planner_version).strip() if planner_version else None),
            agent_role=(str(agent_role).strip() if agent_role else None),
            checkpoint_state=dict(checkpoint_state or {}) or None,
            budget_state=dict(budget_state or {}) or None,
            trace_group_id=(str(trace_group_id).strip() if trace_group_id else None),
            workflow_profile=(str(workflow_profile).strip() if workflow_profile else None),
            pedagogy_level=(str(pedagogy_level).strip() if pedagogy_level else None),
        )
