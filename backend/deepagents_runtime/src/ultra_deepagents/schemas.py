from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ultra_deepagents.context import AgentRunContext


@dataclass(frozen=True)
class RunJobEnvelope:
    run_id: str
    thread_id: str
    user_id: str
    goal: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    file_ids: list[str] = field(default_factory=list)
    resource_uris: list[str] = field(default_factory=list)
    dataset_uris: list[str] = field(default_factory=list)
    selected_tool_names: list[str] = field(default_factory=list)
    knowledge_context: dict[str, Any] = field(default_factory=dict)
    selection_context: dict[str, Any] = field(default_factory=dict)
    workflow_hint: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    benchmark: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    resource_descriptors: list[dict[str, Any]] = field(default_factory=list)
    reasoning_mode: str = "auto"
    response_contract: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunJobEnvelope":
        return cls(
            run_id=_string(payload.get("run_id")),
            thread_id=_string(payload.get("thread_id")),
            user_id=_string(payload.get("user_id")),
            goal=_string(payload.get("goal")),
            messages=_message_list(payload.get("messages")),
            file_ids=_string_list(payload.get("file_ids")),
            resource_uris=_string_list(payload.get("resource_uris")),
            dataset_uris=_string_list(payload.get("dataset_uris")),
            selected_tool_names=_string_list(payload.get("selected_tool_names")),
            knowledge_context=_dict(payload.get("knowledge_context")),
            selection_context=_dict(payload.get("selection_context")),
            workflow_hint=_dict(payload.get("workflow_hint")),
            budgets=_dict(payload.get("budgets")),
            benchmark=_dict(payload.get("benchmark")),
            metadata=_dict(payload.get("metadata")),
            resource_descriptors=_dict_list(payload.get("resource_descriptors")),
            reasoning_mode=_string(payload.get("reasoning_mode")) or "auto",
            response_contract=_dict(payload.get("response_contract")),
        )

    def to_context(self, *, artifact_root: str, workspace_root: str) -> AgentRunContext:
        principal = _dict(self.metadata.get("principal"))
        org_id = _string(principal.get("org_id")) or _string(self.metadata.get("org_id")) or "local-org"
        role = _string(principal.get("role")) or _string(self.metadata.get("role")) or "researcher"
        auth_claims = dict(principal)
        auth_claims.setdefault("role", role)
        return AgentRunContext(
            assistant_id=_string(self.metadata.get("assistant_id")) or "ultra-research-agent",
            org_id=org_id,
            user_id=self.user_id or _string(principal.get("user_id")) or "local-user",
            project_id=_string(self.metadata.get("project_id")) or "local-project",
            thread_id=self.thread_id,
            run_id=self.run_id,
            goal=self.goal,
            model_profile=_string(self.metadata.get("model_profile")) or "vllm",
            selected_file_ids=tuple(self.file_ids),
            selected_resource_uris=tuple(self.resource_uris),
            selected_dataset_uris=tuple(self.dataset_uris),
            allowed_tool_packs=tuple(self.selected_tool_names),
            knowledge_context=dict(self.knowledge_context),
            selection_context=dict(self.selection_context),
            workflow_hint=dict(self.workflow_hint),
            reasoning_mode=self.reasoning_mode,
            benchmark=dict(self.benchmark),
            run_metadata=dict(self.metadata),
            resource_descriptors=tuple(dict(item) for item in self.resource_descriptors),
            response_contract=dict(self.response_contract),
            budget=dict(self.budgets),
            auth_claims=auth_claims,
            artifact_root=artifact_root,
            workspace_root=workspace_root,
        )


def _string(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [str(item) for item in value if item is not None]


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list | tuple):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _message_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list | tuple):
        return []
    messages: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            messages.append(dict(item))
    return messages
