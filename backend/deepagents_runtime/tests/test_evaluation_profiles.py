from __future__ import annotations

from typing import Any

import pytest
from ultra_deepagents.evaluation_profiles import (
    EvaluationProfileError,
    evaluation_profile_policy,
    is_cleanroom_evaluation_profile,
    normalize_evaluation_profile,
)
from ultra_deepagents.nats_worker import _should_load_user_profile
from ultra_deepagents.schemas import RunJobEnvelope

# No evaluation profile is registered (see evaluation_profiles._SUPPORTED_PROFILES),
# which mirrors the control plane's closed enum in
# backend/controlplane/internal/domain/evaluation_profile.go. These tests pin the
# surface that must keep holding while the registry is empty: an ordinary run is
# never treated as protected, and no free-form field can invent a profile.


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


def test_absent_profile_is_a_normal_unprotected_run() -> None:
    assert normalize_evaluation_profile("") == ""
    assert normalize_evaluation_profile(None) == ""
    assert evaluation_profile_policy("") is None
    assert is_cleanroom_evaluation_profile("") is False

    job = RunJobEnvelope.from_dict(_job_payload())
    assert job.evaluation_profile == ""
    assert _should_load_user_profile(job) is True


def test_every_unregistered_profile_is_rejected() -> None:
    for unknown in (
        "materials_cleanroom_v1",
        "cleanroom_v1",
        " materials_cleanroom_v1 ",
        "unknown_profile",
    ):
        with pytest.raises(EvaluationProfileError, match="unsupported evaluation_profile"):
            normalize_evaluation_profile(unknown)
        with pytest.raises(EvaluationProfileError, match="unsupported evaluation_profile"):
            RunJobEnvelope.from_dict(_job_payload(evaluation_profile=unknown))


def test_profile_must_be_a_string() -> None:
    with pytest.raises(EvaluationProfileError, match="must be a string"):
        normalize_evaluation_profile(123)


def test_no_untyped_field_can_invent_a_protected_profile() -> None:
    # The trusted typed top-level field is the ONLY channel. Free-form metadata,
    # selection_context, workflow_hint, and benchmark must never promote a run.
    spoofed = RunJobEnvelope.from_dict(
        _job_payload(
            metadata={
                "evaluation_profile": "materials_cleanroom_v1",
                "runtime_facts": {"benchmark_name": "hidden-suite"},
            },
            selection_context={"evaluation_profile": "materials_cleanroom_v1"},
            workflow_hint={"evaluation_profile": "materials_cleanroom_v1"},
            benchmark={"evaluation_profile": "materials_cleanroom_v1"},
        )
    )

    assert spoofed.evaluation_profile == ""
    assert _should_load_user_profile(spoofed) is True

    spoofed_context = spoofed.to_context(
        artifact_root="/tmp/artifacts",
        workspace_root="/tmp/workspace",
    )
    assert "evaluation_profile" not in spoofed_context.run_metadata
    assert spoofed_context.evaluation_profile == ""
    assert is_cleanroom_evaluation_profile(spoofed_context.evaluation_profile) is False
    # An unprotected run keeps its ordinary context surface.
    assert spoofed_context.selection_context
    assert spoofed_context.workflow_hint
    assert spoofed_context.benchmark
