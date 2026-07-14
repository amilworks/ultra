from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentRunContext:
    assistant_id: str
    org_id: str
    user_id: str
    project_id: str
    thread_id: str
    run_id: str
    goal: str = ""
    model_profile: str = "vllm"
    evaluation_profile: str = ""
    remote_mutation_intents: tuple[str, ...] = field(default_factory=tuple)
    selected_file_ids: tuple[str, ...] = field(default_factory=tuple)
    selected_resource_uris: tuple[str, ...] = field(default_factory=tuple)
    selected_dataset_uris: tuple[str, ...] = field(default_factory=tuple)
    allowed_tool_packs: tuple[str, ...] = field(default_factory=tuple)
    knowledge_context: dict[str, Any] = field(default_factory=dict)
    selection_context: dict[str, Any] = field(default_factory=dict)
    workflow_hint: dict[str, Any] = field(default_factory=dict)
    reasoning_mode: str = "auto"
    benchmark: dict[str, Any] = field(default_factory=dict)
    runtime_facts: dict[str, Any] = field(default_factory=dict)
    run_metadata: dict[str, Any] = field(default_factory=dict)
    resource_descriptors: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    response_contract: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)
    auth_claims: dict[str, Any] = field(default_factory=dict)
    artifact_root: str = "/outputs"
    workspace_root: str = "/workspace"
    sandbox_policy: dict[str, Any] = field(default_factory=dict)
    run_lease_worker_id: str = field(default="", repr=False, compare=False)
    run_lease_token: str = field(default="", repr=False, compare=False)

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        # Preserve the ordinary context wire shape. The profile is only relevant
        # when the trusted envelope explicitly selected one.
        if not self.evaluation_profile:
            payload.pop("evaluation_profile", None)
        # Remote mutation authority is local to the leased coordinator run and
        # must never be delegated to an async subagent payload.
        payload.pop("remote_mutation_intents", None)
        payload["selected_file_ids"] = list(self.selected_file_ids)
        payload["selected_resource_uris"] = list(self.selected_resource_uris)
        payload["selected_dataset_uris"] = list(self.selected_dataset_uris)
        payload["allowed_tool_packs"] = list(self.allowed_tool_packs)
        payload["resource_descriptors"] = list(self.resource_descriptors)
        payload.pop("run_lease_worker_id", None)
        payload.pop("run_lease_token", None)
        return payload
