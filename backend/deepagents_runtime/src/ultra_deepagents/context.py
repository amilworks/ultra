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
    model_profile: str = "vllm"
    selected_file_ids: tuple[str, ...] = field(default_factory=tuple)
    selected_resource_uris: tuple[str, ...] = field(default_factory=tuple)
    selected_dataset_uris: tuple[str, ...] = field(default_factory=tuple)
    allowed_tool_packs: tuple[str, ...] = field(default_factory=tuple)
    budget: dict[str, Any] = field(default_factory=dict)
    auth_claims: dict[str, Any] = field(default_factory=dict)
    artifact_root: str = "/outputs"
    sandbox_policy: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selected_file_ids"] = list(self.selected_file_ids)
        payload["selected_resource_uris"] = list(self.selected_resource_uris)
        payload["selected_dataset_uris"] = list(self.selected_dataset_uris)
        payload["allowed_tool_packs"] = list(self.allowed_tool_packs)
        return payload
