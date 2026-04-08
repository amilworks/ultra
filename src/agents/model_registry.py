"""Role-based model registry for agent orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import Settings

from .contracts import AgentRole


@dataclass(frozen=True)
class AgentModelRegistry:
    """Resolve models by agent role instead of hardcoding one global model."""

    settings: Settings

    def default_model(self) -> str:
        return str(
            getattr(self.settings, "resolved_llm_model", None)
            or getattr(self.settings, "openai_model", "gpt-4.1")
        ).strip()

    def for_role(self, role: AgentRole) -> str:
        resolved_default = self.default_model()
        overrides = {
            "triage": getattr(self.settings, "agents_model_triage", None),
            "planner": getattr(self.settings, "agents_model_planner", None),
            "domain_specialist": getattr(self.settings, "agents_model_domain_specialist", None),
            "biology": getattr(self.settings, "agents_model_biology", None),
            "verifier": getattr(self.settings, "agents_model_verifier", None),
            "synthesizer": getattr(self.settings, "agents_model_synthesizer", None),
            "coder": getattr(self.settings, "agents_model_coder", None),
            "vision": getattr(self.settings, "agents_model_vision", None),
            "medical": getattr(self.settings, "agents_model_medical", None),
            "safety_governor": getattr(self.settings, "agents_model_safety_governor", None),
        }
        override = str(overrides.get(role) or "").strip()
        return override or resolved_default

    def for_domain(self, domain_id: str) -> str:
        token = str(domain_id or "").strip().lower()
        if token == "medical":
            return self.for_role("medical")
        if token == "bio":
            return self.for_role("biology")
        if token == "materials":
            return self.for_role("vision")
        return self.for_role("domain_specialist")
