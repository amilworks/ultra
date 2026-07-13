from __future__ import annotations

import asyncio
import hashlib
import json
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ultra_deepagents import agent as agent_module
from ultra_deepagents import runner as runner_module
from ultra_deepagents.agent import (
    build_agent_backend,
    build_research_agent,
    build_system_prompt,
)
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.evaluation_profiles import (
    EVALUATION_PROFILE_EVENT_KIND,
    MATERIALS_CLEANROOM_PROFILE,
    EvaluationProfileError,
    build_evaluation_profile_attestation,
    build_evaluation_surface_attestation,
    evaluation_artifact_dir,
    evaluation_memory_dir,
    evaluation_policy_dir,
    evaluation_state_root,
    evaluation_workspace_dir,
    materialize_evaluation_profile_attestation,
    materialize_evaluation_surface_attestation,
)
from ultra_deepagents.nats_worker import NATSDeepAgentsWorker, _should_load_user_profile
from ultra_deepagents.runner import run_job
from ultra_deepagents.schemas import RunJobEnvelope

EXPECTED_DISABLED_CAPABILITIES = [
    "benchmark_identity_context",
    "durable_user_memory",
    "episodic_memory_tools",
    "external_async_subagents",
    "linked_account_tools",
    "organization_policy_memory",
    "preloaded_knowledge_context",
    "prior_run_artifact_tools",
    "prior_thread_messages",
    "selected_file_context",
    "user_profile_context",
    "user_resource_catalog_tools",
]
CLEANROOM_PROFILES = (MATERIALS_CLEANROOM_PROFILE,)


def _job_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_id": "run-clean-1",
        "thread_id": "thread-clean-1",
        "user_id": "researcher-1",
        "goal": "Return the word isolated.",
        "messages": [{"role": "user", "content": "Return the word isolated."}],
    }
    payload.update(overrides)
    return payload


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def test_only_typed_top_level_field_enables_cleanroom() -> None:
    spoofed = RunJobEnvelope.from_dict(
        _job_payload(
            metadata={
                "evaluation_profile": MATERIALS_CLEANROOM_PROFILE,
                "runtime_facts": {"benchmark_name": "hidden-suite"},
            },
            selection_context={"evaluation_profile": MATERIALS_CLEANROOM_PROFILE},
            workflow_hint={"evaluation_profile": MATERIALS_CLEANROOM_PROFILE},
            benchmark={"evaluation_profile": MATERIALS_CLEANROOM_PROFILE},
        )
    )

    assert spoofed.evaluation_profile == ""
    assert _should_load_user_profile(spoofed) is True
    spoofed_context = spoofed.to_context(
        artifact_root="/tmp/artifacts",
        workspace_root="/tmp/workspace",
    )
    assert "evaluation_profile" not in spoofed_context.run_metadata
    assert spoofed_context.selection_context
    assert spoofed_context.workflow_hint
    assert spoofed_context.benchmark

    trusted = RunJobEnvelope.from_dict(
        _job_payload(
            evaluation_profile=MATERIALS_CLEANROOM_PROFILE,
            selected_tool_names=["spoofed-tool-pack"],
            metadata={
                "evaluation_profile": "unknown_nested_profile",
                "runtime_facts": {"benchmark_name": "hidden-suite"},
                "principal": {"org_id": "lab-1", "role": "researcher"},
            },
            selection_context={"benchmark_name": "hidden-suite"},
            workflow_hint={"id": "pro_mode"},
            benchmark={"suite": "hidden-suite"},
            budgets={"max_runtime_seconds": 123},
            response_contract={"must_name_suite": True},
        )
    )
    trusted_context = trusted.to_context(
        artifact_root="/tmp/artifacts",
        workspace_root="/tmp/workspace",
    )

    assert trusted.evaluation_profile == MATERIALS_CLEANROOM_PROFILE
    assert _should_load_user_profile(trusted) is False
    assert trusted_context.evaluation_profile == MATERIALS_CLEANROOM_PROFILE
    assert trusted_context.allowed_tool_packs == ()
    assert trusted_context.knowledge_context == {}
    assert trusted_context.selection_context == {}
    assert trusted_context.workflow_hint == {}
    assert trusted_context.reasoning_mode == "auto"
    assert trusted_context.benchmark == {}
    assert trusted_context.runtime_facts == {}
    assert trusted_context.run_metadata == {}
    assert trusted_context.resource_descriptors == ()
    assert trusted_context.response_contract == {}
    assert trusted_context.budget == {}
    assert trusted_context.auth_claims["role"] == "researcher"

    for unknown in ("materials_cleanroom_v2", f" {MATERIALS_CLEANROOM_PROFILE} "):
        with pytest.raises(EvaluationProfileError, match="unsupported evaluation_profile"):
            RunJobEnvelope.from_dict(_job_payload(evaluation_profile=unknown))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("file_ids", ["file-from-earlier-thread"]),
        ("resource_uris", ["resource://prior-result"]),
        ("dataset_uris", ["dataset://prior-result"]),
        ("resource_descriptors", [{"type": "artifact", "path": "prior.txt"}]),
        ("knowledge_context", {"ingested_papers": [{"paper_id": "prior-paper"}]}),
        ("remote_mutation_intents", ["bisque.upload"]),
    ],
)
@pytest.mark.parametrize("profile", CLEANROOM_PROFILES)
def test_cleanroom_rejects_selected_or_preloaded_context(
    profile: str,
    field: str,
    value: Any,
) -> None:
    with pytest.raises(EvaluationProfileError, match=field):
        RunJobEnvelope.from_dict(
            _job_payload(
                evaluation_profile=profile,
                **{field: value},
            )
        )


def test_worker_attestation_is_write_once_and_exact_on_retry(tmp_path: Path) -> None:
    memory_root = tmp_path / "memory"
    workspace = evaluation_workspace_dir(
        tmp_path / "workspaces",
        MATERIALS_CLEANROOM_PROFILE,
        "run-1",
    )
    artifact_dir = evaluation_artifact_dir(
        tmp_path / "artifacts",
        MATERIALS_CLEANROOM_PROFILE,
        "run-1",
    )
    attestation = build_evaluation_profile_attestation(
        profile=MATERIALS_CLEANROOM_PROFILE,
        run_id="run-1",
        thread_id="thread-1",
        user_id="user-1",
        goal="current goal",
        provided_message_count=4,
    )

    path, created = materialize_evaluation_profile_attestation(
        memory_root=memory_root,
        workspace_dir=workspace,
        artifact_dir=artifact_dir,
        attestation=attestation,
    )
    first_bytes = path.read_bytes()
    workspace.mkdir(parents=True)
    retry_path, retry_created = materialize_evaluation_profile_attestation(
        memory_root=memory_root,
        workspace_dir=workspace,
        artifact_dir=artifact_dir,
        attestation=attestation,
    )

    assert created is True
    assert retry_created is False
    assert retry_path == path
    assert retry_path.read_bytes() == first_bytes
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.parent == evaluation_state_root(
        memory_root,
        MATERIALS_CLEANROOM_PROFILE,
        "run-1",
    )
    assert (
        evaluation_memory_dir(
            memory_root,
            MATERIALS_CLEANROOM_PROFILE,
            "run-1",
        )
        not in path.parents
    )

    changed_goal = build_evaluation_profile_attestation(
        profile=MATERIALS_CLEANROOM_PROFILE,
        run_id="run-1",
        thread_id="thread-1",
        user_id="user-1",
        goal="changed goal",
        provided_message_count=4,
    )
    with pytest.raises(EvaluationProfileError, match="differs from this delivery"):
        materialize_evaluation_profile_attestation(
            memory_root=memory_root,
            workspace_dir=workspace,
            artifact_dir=artifact_dir,
            attestation=changed_goal,
        )
    assert path.read_bytes() == first_bytes


def test_worker_surface_attestation_is_bound_to_profile_and_write_once(
    tmp_path: Path,
) -> None:
    profile = MATERIALS_CLEANROOM_PROFILE
    memory_root = tmp_path / "memory"
    workspace = evaluation_workspace_dir(tmp_path / "workspaces", profile, "run-surface")
    artifact_dir = evaluation_artifact_dir(tmp_path / "artifacts", profile, "run-surface")
    attestation = build_evaluation_profile_attestation(
        profile=profile,
        run_id="run-surface",
        thread_id="thread-surface",
        user_id="user-surface",
        goal="sealed surface",
        provided_message_count=1,
    )
    materialize_evaluation_profile_attestation(
        memory_root=memory_root,
        workspace_dir=workspace,
        artifact_dir=artifact_dir,
        attestation=attestation,
    )
    surface = build_evaluation_surface_attestation(
        profile_attestation=attestation,
        runtime_image_digest="sha256:" + "1" * 64,
        surface={
            "surface_source": "build_research_agent",
            "domain_tool_manifest_sha256": "2" * 64,
            "full_tool_manifest_sha256": "3" * 64,
            "system_prompt_sha256": "5" * 64,
        },
        model_id="deepseek_v4",
        provider_id="openai-compatible",
    )
    path, created = materialize_evaluation_surface_attestation(
        memory_root=memory_root,
        profile_attestation=attestation,
        surface_attestation=surface,
    )
    replay_path, replay_created = materialize_evaluation_surface_attestation(
        memory_root=memory_root,
        profile_attestation=attestation,
        surface_attestation=surface,
    )
    assert created is True
    assert replay_created is False
    assert replay_path == path
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert surface["profile_attestation_sha256"] == attestation["attestation_sha256"]
    assert len(surface["surface_attestation_sha256"]) == 64

    changed = dict(surface)
    changed["system_prompt_sha256"] = "6" * 64
    unsigned = dict(changed)
    unsigned.pop("surface_attestation_sha256")
    changed["surface_attestation_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    with pytest.raises(EvaluationProfileError, match="surface attestation differs"):
        materialize_evaluation_surface_attestation(
            memory_root=memory_root,
            profile_attestation=attestation,
            surface_attestation=changed,
        )


def test_worker_surface_attestation_rejects_mutable_runtime_image() -> None:
    attestation = build_evaluation_profile_attestation(
        profile=MATERIALS_CLEANROOM_PROFILE,
        run_id="run-surface-reject",
        thread_id="thread-surface-reject",
        user_id="user-surface-reject",
        goal="sealed materials surface",
        provided_message_count=1,
    )
    with pytest.raises(EvaluationProfileError, match="runtime image"):
        build_evaluation_surface_attestation(
            profile_attestation=attestation,
            runtime_image_digest="mutable:latest",
            surface={
                "surface_source": "build_research_agent",
                "domain_tool_manifest_sha256": "2" * 64,
                "full_tool_manifest_sha256": "3" * 64,
                "system_prompt_sha256": "5" * 64,
            },
            model_id="deepseek_v4",
            provider_id="openai-compatible",
        )


@pytest.mark.parametrize("existing_kind", ["workspace", "artifact"])
def test_worker_refuses_unsealed_preexisting_cleanroom_storage(
    tmp_path: Path,
    existing_kind: str,
) -> None:
    workspace = evaluation_workspace_dir(
        tmp_path / "workspaces",
        MATERIALS_CLEANROOM_PROFILE,
        "run-1",
    )
    artifact_dir = evaluation_artifact_dir(
        tmp_path / "artifacts",
        MATERIALS_CLEANROOM_PROFILE,
        "run-1",
    )
    (workspace if existing_kind == "workspace" else artifact_dir).mkdir(parents=True)
    attestation = build_evaluation_profile_attestation(
        profile=MATERIALS_CLEANROOM_PROFILE,
        run_id="run-1",
        thread_id="thread-1",
        user_id="user-1",
        goal="current goal",
        provided_message_count=1,
    )

    with pytest.raises(EvaluationProfileError, match="unsealed pre-existing"):
        materialize_evaluation_profile_attestation(
            memory_root=tmp_path / "memory",
            workspace_dir=workspace,
            artifact_dir=artifact_dir,
            attestation=attestation,
        )


class _CapturingAgent:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []
        self.contexts: list[Any] = []

    async def astream_events(
        self,
        payload: dict[str, Any],
        *,
        config: dict[str, Any] | None = None,
        context: Any = None,
        version: str | None = None,
    ):
        assert config is not None
        assert version == "v3"
        self.payloads.append(payload)
        self.contexts.append(context)
        yield {
            "event": "on_chat_model_stream",
            "data": {"chunk": SimpleNamespace(content="isolated")},
            "metadata": {"langgraph_node": "coordinator"},
        }


def test_run_job_enforces_goal_only_isolation_and_worker_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        workspace_root=str(tmp_path / "workspaces"),
        artifact_root=str(tmp_path / "artifacts"),
        memory_root=str(tmp_path / "memory"),
        workspace_retention_seconds=0,
    )
    normal_workspace = Path(settings.workspace_root) / "run-clean-1"
    normal_workspace.mkdir(parents=True)
    (normal_workspace / "prior-thread.txt").write_text(
        "secret-benchmark-name",
        encoding="utf-8",
    )
    normal_artifacts = Path(settings.artifact_root) / "run-clean-1"
    normal_artifacts.mkdir(parents=True)
    (normal_artifacts / "prior-output.txt").write_text(
        "secret-benchmark-name",
        encoding="utf-8",
    )
    durable_memory = Path(settings.memory_root) / "users" / "researcher-1"
    durable_memory.mkdir(parents=True)
    (durable_memory / "preferences.md").write_text(
        "Always mention secret-benchmark-name.",
        encoding="utf-8",
    )

    def fail_if_called(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("durable or preloaded context path was consulted")

    monkeypatch.setattr(runner_module, "_seed_user_profile_memory", fail_if_called)
    monkeypatch.setattr(runner_module, "_preload_arxiv_papers_for_context", fail_if_called)
    monkeypatch.setattr(
        runner_module,
        "_preload_uploaded_pdf_papers_for_context",
        fail_if_called,
    )
    monkeypatch.setattr(
        runner_module,
        "resolve_docker_image_id",
        lambda _image: "sha256:" + "a" * 64,
    )

    job = RunJobEnvelope.from_dict(
        _job_payload(
            evaluation_profile=MATERIALS_CLEANROOM_PROFILE,
            messages=[
                {"role": "user", "content": "Run secret-benchmark-name."},
                {"role": "assistant", "content": "I remember the earlier answer."},
                {"role": "user", "content": "Return the word isolated."},
            ],
            benchmark={"suite": "secret-benchmark-name"},
            selection_context={"suite": "secret-benchmark-name"},
            workflow_hint={"id": "pro_mode"},
            metadata={
                "evaluation_profile": "metadata-spoof",
                "runtime_facts": {"benchmark_name": "secret-benchmark-name"},
                "principal": {"org_id": "lab-1", "role": "researcher"},
            },
        )
    )
    agent = _CapturingAgent()
    published: list[dict[str, Any]] = []

    async def publish(event: dict[str, Any]) -> None:
        published.append(event)

    result = asyncio.run(
        run_job(
            job,
            settings,
            publish_event=publish,
            agent_factory=lambda *_args, **_kwargs: agent,
            user_profile={"bio": "secret-benchmark-name"},
        )
    )

    assert result == "isolated"
    assert agent.payloads == [
        {"messages": [{"role": "user", "content": "Return the word isolated."}]}
    ]
    context = agent.contexts[0]
    expected_workspace = evaluation_workspace_dir(
        settings.workspace_root,
        MATERIALS_CLEANROOM_PROFILE,
        job.run_id,
    )
    expected_artifacts = evaluation_artifact_dir(
        settings.artifact_root,
        MATERIALS_CLEANROOM_PROFILE,
        job.run_id,
    )
    assert Path(context.workspace_root) == expected_workspace
    assert Path(context.artifact_root) == expected_artifacts
    assert not (expected_workspace / "prior-thread.txt").exists()
    assert not (expected_artifacts / "prior-output.txt").exists()
    assert context.evaluation_profile == MATERIALS_CLEANROOM_PROFILE
    assert context.knowledge_context == {}
    assert context.selection_context == {}
    assert context.workflow_hint == {}
    assert context.benchmark == {}
    assert context.runtime_facts == {}
    assert context.run_metadata == {}
    assert (normal_workspace / "prior-thread.txt").exists()
    assert (
        (durable_memory / "preferences.md").read_text(encoding="utf-8").startswith("Always mention")
    )

    assert [event["event_kind"] for event in published[:2]] == [
        EVALUATION_PROFILE_EVENT_KIND,
        "run.started",
    ]
    attestation_event = published[0]
    attestation = attestation_event["payload"]
    assert attestation_event["node_name"] == "worker"
    assert attestation_event["agent_role"] == "worker"
    assert attestation["evaluation_profile"] == MATERIALS_CLEANROOM_PROFILE
    assert attestation["profile_source"] == "typed_job_envelope"
    assert attestation["input_policy"] == "goal_only"
    assert attestation["provided_message_count"] == 3
    assert attestation["effective_message_count"] == 1
    assert attestation["prior_thread_context_discarded"] is True
    assert attestation["run_scoped_workspace"] is True
    assert attestation["run_scoped_memory"] is True
    assert attestation["disabled_capabilities"] == EXPECTED_DISABLED_CAPABILITIES
    assert set(attestation) == {
        "schema_version",
        "attestation_kind",
        "worker_owned",
        "evaluation_profile",
        "profile_source",
        "trusted_envelope_field",
        "namespace_id",
        "run_id_sha256",
        "thread_id_sha256",
        "user_id_sha256",
        "goal_sha256",
        "input_policy",
        "provided_message_count",
        "effective_message_count",
        "prior_thread_context_discarded",
        "same_run_retry_state_allowed",
        "run_scoped_workspace",
        "run_scoped_memory",
        "disabled_capabilities",
        "attestation_sha256",
    }
    unsigned = dict(attestation)
    declared_digest = unsigned.pop("attestation_sha256")
    assert declared_digest == hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    assert "secret-benchmark-name" not in json.dumps(published, sort_keys=True)

    state_root = evaluation_state_root(
        settings.memory_root,
        MATERIALS_CLEANROOM_PROFILE,
        job.run_id,
    )
    seal = state_root / "worker-attestation.json"
    assert json.loads(seal.read_text(encoding="utf-8")) == attestation
    assert stat.S_IMODE(seal.stat().st_mode) == 0o600
    assert not any(
        evaluation_memory_dir(
            settings.memory_root,
            MATERIALS_CLEANROOM_PROFILE,
            job.run_id,
        ).iterdir()
    )


def test_cleanroom_agent_has_no_durable_or_external_context_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_create_deep_agent(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "compiled-cleanroom-agent"

    def forbidden_builder(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("disabled context tool builder was called")

    monkeypatch.setattr(agent_module, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(agent_module, "build_context_tools", forbidden_builder)
    monkeypatch.setattr(agent_module, "build_episodic_tools", forbidden_builder)
    monkeypatch.setattr(agent_module, "build_resource_tools", forbidden_builder)
    monkeypatch.setattr(agent_module, "build_bisque_tools", forbidden_builder)
    monkeypatch.setattr(agent_module, "build_builder_subagent", lambda *_a, **_k: None)

    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        async_subagents=(
            {
                "name": "remote-researcher",
                "description": "remote",
                "graph_id": "remote-graph",
            },
        ),
    )
    context = RunJobEnvelope.from_dict(
        _job_payload(evaluation_profile=MATERIALS_CLEANROOM_PROFILE)
    ).to_context(artifact_root="/tmp/artifacts", workspace_root="/tmp/workspace")

    result = build_research_agent(
        settings,
        model=object(),
        backend=object(),
        context=context,
    )

    assert result == "compiled-cleanroom-agent"
    assert captured["memory"] == []
    assert captured["subagents"] == []
    tool_names = {str(getattr(tool, "name", "")) for tool in captured["tools"]}
    assert "artifact_manifest" not in tool_names
    assert "stage_artifact_for_analysis" not in tool_names
    assert "stage_uploaded_files_for_analysis" not in tool_names
    assert "search_past_research" not in tool_names
    assert "search_resources" not in tool_names
    assert "stage_resource_for_analysis" not in tool_names
    assert "start_async_task" not in tool_names
    assert "check_async_task" not in tool_names

    prompt = build_system_prompt(settings, context)
    assert "isolated evaluation context" in prompt
    assert "/memories/user_profile.md" not in prompt
    assert "/memories/preferences.md" not in prompt
    assert "research_context/INDEX.md" not in prompt
    assert "search their catalog" not in prompt
    assert "search prior sessions" in prompt
    assert "secret-benchmark-name" not in prompt


def test_cleanroom_backend_routes_memory_and_policy_to_run_namespace(
    tmp_path: Path,
) -> None:
    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        memory_root=str(tmp_path / "memory"),
        artifact_root=str(tmp_path / "artifacts"),
    )
    durable_user_root = Path(settings.memory_root) / "users" / "researcher-1"
    durable_user_root.mkdir(parents=True)
    (durable_user_root / "preferences.md").write_text("durable", encoding="utf-8")
    run_id = "run-clean-backend"
    backend = build_agent_backend(
        settings,
        workspace_dir=tmp_path / "workspace",
        artifact_dir=tmp_path / "artifacts" / "clean",
        user_id="researcher-1",
        org_id="lab-1",
        run_id=run_id,
        evaluation_profile=MATERIALS_CLEANROOM_PROFILE,
    )

    assert (
        backend.routes["/memories/"].cwd.resolve()
        == evaluation_memory_dir(
            settings.memory_root,
            MATERIALS_CLEANROOM_PROFILE,
            run_id,
        ).resolve()
    )
    assert (
        backend.routes["/policies/"].cwd.resolve()
        == evaluation_policy_dir(
            settings.memory_root,
            MATERIALS_CLEANROOM_PROFILE,
            run_id,
        ).resolve()
    )
    assert backend.routes["/memories/"].cwd.resolve() != durable_user_root.resolve()
    assert (durable_user_root / "preferences.md").read_text(encoding="utf-8") == "durable"


class _FakeNATSMessage:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.data = _canonical_json(payload)
        self.acked = 0
        self.naked = 0
        self.in_progress_calls = 0

    async def ack(self) -> None:
        self.acked += 1

    async def nak(self, delay: float | None = None) -> None:
        _ = delay
        self.naked += 1

    async def in_progress(self) -> None:
        self.in_progress_calls += 1


class _CapturingJetStream:
    def __init__(self) -> None:
        self.published: list[tuple[str, bytes, dict[str, str]]] = []

    async def publish(
        self,
        subject: str,
        payload: bytes,
        **kwargs: Any,
    ) -> None:
        self.published.append((subject, payload, kwargs.get("headers") or {}))


class _NoCheckpointWorker(NATSDeepAgentsWorker):
    async def _ensure_checkpointer(self):
        return None

    async def _run_events_snapshot(self, run_id: str):
        _ = run_id
        return 0, None


def test_worker_transport_honors_only_typed_profile_and_skips_profile_fetch(
    tmp_path: Path,
) -> None:
    profile_fetches: list[str] = []
    delivered: list[tuple[str, str, dict[str, Any]]] = []

    async def run_job_stub(
        job: RunJobEnvelope,
        _settings: RuntimeSettings,
        **kwargs: Any,
    ) -> str:
        delivered.append((job.run_id, job.evaluation_profile, kwargs))
        return "done"

    async def run_status_stub(
        _run_id: str,
        _settings: RuntimeSettings,
    ) -> None:
        return None

    async def run_lease_stub(
        _run_id: str,
        _settings: RuntimeSettings,
    ) -> None:
        return None

    async def user_profile_stub(
        run_id: str,
        _settings: RuntimeSettings,
    ) -> dict[str, str]:
        profile_fetches.append(run_id)
        return {"bio": "durable profile"}

    async def heartbeat_stub(
        _settings: RuntimeSettings,
        **_kwargs: Any,
    ) -> None:
        return None

    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        workspace_root=str(tmp_path / "workspaces"),
        worker_ack_progress_interval_seconds=30.0,
        worker_heartbeat_interval_seconds=0,
        control_status_poll_interval_seconds=0,
    )
    worker = _NoCheckpointWorker(
        settings,
        run_job_func=run_job_stub,
        run_status_func=run_status_stub,
        run_lease_func=run_lease_stub,
        worker_heartbeat_func=heartbeat_stub,
        user_profile_func=user_profile_stub,
    )
    js = _CapturingJetStream()
    cleanroom_message = _FakeNATSMessage(
        _job_payload(evaluation_profile=MATERIALS_CLEANROOM_PROFILE)
    )
    spoofed_message = _FakeNATSMessage(
        _job_payload(
            run_id="run-spoofed",
            metadata={"evaluation_profile": MATERIALS_CLEANROOM_PROFILE},
        )
    )
    unknown_message = _FakeNATSMessage(
        _job_payload(
            run_id="run-unknown",
            evaluation_profile="unknown_profile",
        )
    )

    async def scenario() -> None:
        await worker._process_message(cleanroom_message, js)
        await worker._process_message(spoofed_message, js)
        await worker._process_message(unknown_message, js)

    asyncio.run(scenario())

    assert cleanroom_message.acked == 1
    assert spoofed_message.acked == 1
    assert unknown_message.acked == 1
    assert cleanroom_message.naked == 0
    assert spoofed_message.naked == 0
    assert unknown_message.naked == 0
    assert [item[:2] for item in delivered] == [
        ("run-clean-1", MATERIALS_CLEANROOM_PROFILE),
        ("run-spoofed", ""),
    ]
    assert "user_profile" not in delivered[0][2]
    assert delivered[1][2]["user_profile"] == {"bio": "durable profile"}
    assert profile_fetches == ["run-spoofed"]
