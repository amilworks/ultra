"""Memory correctness guards.

These cover the parts of the memory contract that previously had no test and
that the live review exercised by hand: app/org-owned files are read-only to the
agent, learned preferences persist and are re-loaded on a later run for the same
user, and org policy memory is shared within an org but isolated across orgs.
"""

from __future__ import annotations

from pathlib import Path

from deepagents.middleware.filesystem import _check_fs_permission
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pydantic import Field
from ultra_deepagents.agent import (
    MEMORY_PERMISSIONS,
    build_research_agent,
    resolve_org_policies_root,
    resolve_user_memory_root,
)
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext


class _PromptCapturingModel(FakeMessagesListChatModel):
    """Fake model that records the system prompt of the first model call."""

    system_prompts: list[str] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN002, ANN003
        return self

    def _get_ls_params(self, stop=None, **kwargs):  # noqa: ANN001, ANN002, ANN003
        return {"ls_provider": "openai", "ls_model_name": "fake"}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if messages:
            self.system_prompts.append(str(messages[0].content))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _settings(tmp_path: Path) -> RuntimeSettings:
    return RuntimeSettings(
        openai_base_url="http://127.0.0.1:9/v1",
        openai_model="deepseek_v4",
        workspace_root=str(tmp_path / "ws"),
        memory_root=str(tmp_path / "mem"),
        artifact_root=str(tmp_path / "art"),
    )


def _context(*, user_id: str, org_id: str, run_id: str, goal: str = "Analyze a dataset.") -> AgentRunContext:
    return AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id=org_id,
        user_id=user_id,
        project_id="proj",
        thread_id=f"thread-{run_id}",
        run_id=run_id,
        goal=goal,
    )


# --- C1: app/org-owned memory is read-only to the agent ----------------------


def test_memory_permissions_deny_app_and_org_owned_writes():
    deny = lambda path: _check_fs_permission(MEMORY_PERMISSIONS, "write", path)  # noqa: E731
    allow = lambda path: _check_fs_permission(MEMORY_PERMISSIONS, "read", path)  # noqa: E731

    # App-owned profile, all org policy files, and developer-owned skills: writes denied.
    assert deny("/memories/user_profile.md") == "deny"
    assert deny("/policies/lab_policy.md") == "deny"
    assert deny("/policies/nested/anything.md") == "deny"
    assert deny("/skills/computational-experiment-rigor/SKILL.md") == "deny"
    assert deny("/skills/scientific-reporting/references/anything.md") == "deny"
    # Agent-owned memory stays writable.
    assert deny("/memories/preferences.md") == "allow"
    assert deny("/memories/research_context/attention.md") == "allow"
    # Reads are never blocked (the agent must load these).
    assert allow("/memories/user_profile.md") == "allow"
    assert allow("/policies/lab_policy.md") == "allow"
    assert allow("/skills/computational-experiment-rigor/SKILL.md") == "allow"


# --- C2: learned preferences persist and reload on a later run ---------------


def test_user_memory_roundtrip_reloads_persisted_preference(tmp_path: Path):
    settings = _settings(tmp_path)
    user_id = "researcher-roundtrip"
    memory_root = resolve_user_memory_root(settings, user_id)
    memory_root.mkdir(parents=True, exist_ok=True)
    preference = "Always report uncertainty as 95% confidence intervals."
    (memory_root / "preferences.md").write_text(
        f"# Preferences\n\n## Uncertainty\n{preference}\n", encoding="utf-8"
    )

    model = _PromptCapturingModel(responses=[AIMessage(content="ok")])
    ctx = _context(user_id=user_id, org_id="org-1", run_id="run-read")
    agent = build_research_agent(
        settings, model=model, workspace_dir=tmp_path / "ws" / "run-read", context=ctx
    )
    agent.invoke({"messages": [{"role": "user", "content": "hi"}]}, context=ctx)

    assert model.system_prompts, "model was never called"
    assert preference in model.system_prompts[0], "persisted preference not reloaded into prompt"


# --- W3: org policy is shared within an org and isolated across orgs ---------


def test_org_policy_loaded_and_isolated_across_orgs(tmp_path: Path):
    settings = _settings(tmp_path)
    policy_text = "Lab policy: cite primary sources; never disclose raw subject identifiers."
    org_a_root = resolve_org_policies_root(settings, "org-alpha")
    org_a_root.mkdir(parents=True, exist_ok=True)
    (org_a_root / "lab_policy.md").write_text(policy_text, encoding="utf-8")

    # User in org-alpha sees the policy.
    model_a = _PromptCapturingModel(responses=[AIMessage(content="ok")])
    ctx_a = _context(user_id="alice", org_id="org-alpha", run_id="run-a")
    build_research_agent(
        settings, model=model_a, workspace_dir=tmp_path / "ws" / "run-a", context=ctx_a
    ).invoke({"messages": [{"role": "user", "content": "hi"}]}, context=ctx_a)
    assert policy_text in model_a.system_prompts[0]

    # User in a different org does not.
    model_b = _PromptCapturingModel(responses=[AIMessage(content="ok")])
    ctx_b = _context(user_id="bob", org_id="org-beta", run_id="run-b")
    build_research_agent(
        settings, model=model_b, workspace_dir=tmp_path / "ws" / "run-b", context=ctx_b
    ).invoke({"messages": [{"role": "user", "content": "hi"}]}, context=ctx_b)
    assert policy_text not in model_b.system_prompts[0]
