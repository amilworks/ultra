from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EVALUATION_PROFILE_ATTESTATION_SCHEMA_VERSION = "1"
EVALUATION_PROFILE_EVENT_KIND = "run.evaluation_profile_attested"
EVALUATION_SURFACE_EVENT_KIND = "run.evaluation_surface_attested"
_MAX_ATTESTATION_BYTES = 64 * 1024


class EvaluationProfileError(ValueError):
    """The trusted evaluation profile envelope is invalid or inconsistent."""


@dataclass(frozen=True)
class EvaluationProfilePolicy:
    name: str
    disabled_capabilities: tuple[str, ...]
    goal_only_messages: bool
    run_scoped_memory: bool
    run_scoped_workspace: bool


# No evaluation profile is registered. Build an EvaluationProfilePolicy and add it
# here to enable one; every unregistered name is rejected by normalization below.
_SUPPORTED_PROFILES: dict[str, EvaluationProfilePolicy] = {}


def normalize_evaluation_profile(value: Any) -> str:
    """Normalize the trusted typed field and reject every unknown profile."""

    if value is None:
        return ""
    if not isinstance(value, str):
        raise EvaluationProfileError("evaluation_profile must be a string")
    profile = value.strip()
    if not profile:
        return ""
    if value != profile or profile not in _SUPPORTED_PROFILES:
        raise EvaluationProfileError("unsupported evaluation_profile")
    return profile


def evaluation_profile_policy(value: Any) -> EvaluationProfilePolicy | None:
    profile = normalize_evaluation_profile(value)
    return _SUPPORTED_PROFILES.get(profile)


def is_cleanroom_evaluation_profile(value: Any) -> bool:
    """Return whether the trusted profile requires the shared clean-room boundary."""

    return evaluation_profile_policy(value) is not None


def evaluation_namespace_id(profile: str, run_id: str) -> str:
    normalized = normalize_evaluation_profile(profile)
    if not normalized:
        raise EvaluationProfileError("an evaluation namespace requires a profile")
    run_id = str(run_id or "").strip()
    if not run_id:
        raise EvaluationProfileError("an evaluation namespace requires run_id")
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    return f"{normalized}-{digest}"


def evaluation_workspace_dir(root: str | Path, profile: str, run_id: str) -> Path:
    namespace = evaluation_namespace_id(profile, run_id)
    return Path(root).expanduser() / f".evaluation-{namespace}"


def evaluation_artifact_dir(root: str | Path, profile: str, run_id: str) -> Path:
    namespace = evaluation_namespace_id(profile, run_id)
    return Path(root).expanduser() / f".evaluation-{namespace}"


def evaluation_state_root(root: str | Path, profile: str, run_id: str) -> Path:
    namespace = evaluation_namespace_id(profile, run_id)
    return Path(root).expanduser() / ".evaluation_profiles" / namespace


def evaluation_memory_dir(root: str | Path, profile: str, run_id: str) -> Path:
    return evaluation_state_root(root, profile, run_id) / "memory"


def evaluation_policy_dir(root: str | Path, profile: str, run_id: str) -> Path:
    return evaluation_state_root(root, profile, run_id) / "policies"


def goal_only_messages(profile: str, goal: str) -> list[dict[str, str]]:
    policy = evaluation_profile_policy(profile)
    if policy is None or not policy.goal_only_messages:
        raise EvaluationProfileError("goal-only messages require a supported profile")
    goal = str(goal or "").strip()
    if not goal:
        raise EvaluationProfileError(f"{profile} requires a non-empty goal")
    return [{"role": "user", "content": goal}]


def build_evaluation_profile_attestation(
    *,
    profile: str,
    run_id: str,
    thread_id: str,
    user_id: str,
    goal: str,
    provided_message_count: int,
) -> dict[str, Any]:
    policy = evaluation_profile_policy(profile)
    if policy is None:
        raise EvaluationProfileError("cannot attest an empty evaluation profile")
    namespace = evaluation_namespace_id(policy.name, run_id)
    payload: dict[str, Any] = {
        "schema_version": EVALUATION_PROFILE_ATTESTATION_SCHEMA_VERSION,
        "attestation_kind": "worker_evaluation_profile",
        "worker_owned": True,
        "evaluation_profile": policy.name,
        "profile_source": "typed_job_envelope",
        "trusted_envelope_field": "evaluation_profile",
        "namespace_id": namespace,
        "run_id_sha256": _sha256_text(run_id),
        "thread_id_sha256": _sha256_text(thread_id),
        "user_id_sha256": _sha256_text(user_id),
        "goal_sha256": _sha256_text(goal),
        "input_policy": "goal_only",
        "provided_message_count": max(0, int(provided_message_count)),
        "effective_message_count": 1,
        "prior_thread_context_discarded": True,
        "same_run_retry_state_allowed": True,
        "run_scoped_workspace": policy.run_scoped_workspace,
        "run_scoped_memory": policy.run_scoped_memory,
        "disabled_capabilities": list(policy.disabled_capabilities),
    }
    payload["attestation_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def materialize_evaluation_profile_attestation(
    *,
    memory_root: str | Path,
    workspace_dir: str | Path,
    artifact_dir: str | Path | None = None,
    attestation: dict[str, Any],
) -> tuple[Path, bool]:
    """Create or exact-verify the worker-owned profile seal.

    The seal lives beside, not inside, the model-visible run memory directory.
    A same-run retry may reuse an exact seal; any changed profile, principal,
    thread, or goal fails before the model or sandbox starts.
    """

    profile = normalize_evaluation_profile(attestation.get("evaluation_profile"))
    policy = evaluation_profile_policy(profile)
    if policy is None:
        raise EvaluationProfileError("worker attestation requires a supported evaluation profile")
    namespace = str(attestation.get("namespace_id") or "")
    prefix = f"{profile}-"
    namespace_digest = namespace.removeprefix(prefix)
    if (
        not namespace.startswith(prefix)
        or len(namespace_digest) != 64
        or any(character not in "0123456789abcdef" for character in namespace_digest)
    ):
        raise EvaluationProfileError("evaluation profile namespace is malformed")
    run_id_sha = _required_attested_digest(attestation, "run_id_sha256")
    if namespace_digest != run_id_sha:
        raise EvaluationProfileError("evaluation profile namespace/run binding is inconsistent")
    for key in ("thread_id_sha256", "user_id_sha256", "goal_sha256"):
        _required_attested_digest(attestation, key)
    declared_attestation_sha = _required_attested_digest(
        attestation,
        "attestation_sha256",
    )
    unsigned = dict(attestation)
    unsigned.pop("attestation_sha256", None)
    observed_attestation_sha = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    if declared_attestation_sha != observed_attestation_sha:
        raise EvaluationProfileError("evaluation profile attestation digest is inconsistent")
    state_root = Path(memory_root).expanduser() / ".evaluation_profiles" / namespace
    memory_dir = state_root / "memory"
    policy_dir = state_root / "policies"
    attestation_path = state_root / "worker-attestation.json"
    expected = _canonical_json(attestation) + b"\n"
    if len(expected) > _MAX_ATTESTATION_BYTES:
        raise EvaluationProfileError("evaluation profile attestation is too large")

    state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    created = False
    if attestation_path.exists() or attestation_path.is_symlink():
        observed = _read_regular_file(attestation_path)
        if observed != expected:
            raise EvaluationProfileError(
                "existing worker evaluation profile attestation differs from this delivery"
            )
    else:
        artifact_exists = artifact_dir is not None and Path(artifact_dir).exists()
        if (
            Path(workspace_dir).exists()
            or artifact_exists
            or memory_dir.exists()
            or policy_dir.exists()
        ):
            raise EvaluationProfileError(
                "refusing an unsealed pre-existing evaluation workspace, artifact, "
                "or memory namespace"
            )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(attestation_path, flags, 0o600)
        except FileExistsError:
            observed = _read_regular_file(attestation_path)
            if observed != expected:
                raise EvaluationProfileError(
                    "concurrent worker evaluation profile attestation differs"
                ) from None
        else:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(expected)
                handle.flush()
                os.fsync(handle.fileno())
            created = True
    memory_dir.mkdir(mode=0o700, exist_ok=True)
    policy_dir.mkdir(mode=0o700, exist_ok=True)
    return attestation_path, created


def build_evaluation_surface_attestation(
    *,
    profile_attestation: dict[str, Any],
    runtime_image_digest: str,
    surface: dict[str, str],
    model_id: str,
    provider_id: str,
) -> dict[str, Any]:
    """Bind the actual constructed model/tool/prompt surface to the profile seal."""

    profile = normalize_evaluation_profile(profile_attestation.get("evaluation_profile"))
    policy = evaluation_profile_policy(profile)
    if policy is None:
        raise EvaluationProfileError("surface attestation requires a supported profile")
    profile_attestation_sha256 = _required_attested_digest(
        profile_attestation, "attestation_sha256"
    )
    if (
        not isinstance(runtime_image_digest, str)
        or len(runtime_image_digest) != 71
        or not runtime_image_digest.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in runtime_image_digest[7:])
    ):
        raise EvaluationProfileError("surface attestation runtime image digest is invalid")
    if surface.get("surface_source") != "build_research_agent":
        raise EvaluationProfileError("surface attestation was not derived from the research agent")
    domain_tool_manifest_sha256 = _surface_digest(surface, "domain_tool_manifest_sha256")
    full_tool_manifest_sha256 = _surface_digest(surface, "full_tool_manifest_sha256")
    system_prompt_sha256 = _surface_digest(surface, "system_prompt_sha256")
    normalized_model_id = _bounded_surface_identity(model_id, "model_id")
    normalized_provider_id = _bounded_surface_identity(provider_id, "provider_id")
    payload: dict[str, Any] = {
        "schema_version": EVALUATION_PROFILE_ATTESTATION_SCHEMA_VERSION,
        "attestation_kind": "worker_evaluation_surface",
        "worker_owned": True,
        "evaluation_profile": profile,
        "profile_attestation_sha256": profile_attestation_sha256,
        "surface_source": "build_research_agent",
        "runtime_image_digest": runtime_image_digest,
        "domain_tool_manifest_sha256": domain_tool_manifest_sha256,
        "full_tool_manifest_sha256": full_tool_manifest_sha256,
        "system_prompt_sha256": system_prompt_sha256,
        "model_id": normalized_model_id,
        "provider_id": normalized_provider_id,
    }
    payload["surface_attestation_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def materialize_evaluation_surface_attestation(
    *,
    memory_root: str | Path,
    profile_attestation: dict[str, Any],
    surface_attestation: dict[str, Any],
) -> tuple[Path, bool]:
    """Create or exact-verify the post-construction surface seal."""

    profile = normalize_evaluation_profile(profile_attestation.get("evaluation_profile"))
    namespace = str(profile_attestation.get("namespace_id") or "")
    if not namespace.startswith(f"{profile}-"):
        raise EvaluationProfileError("surface attestation namespace is inconsistent")
    profile_sha256 = _required_attested_digest(profile_attestation, "attestation_sha256")
    if (
        surface_attestation.get("evaluation_profile") != profile
        or surface_attestation.get("profile_attestation_sha256") != profile_sha256
    ):
        raise EvaluationProfileError("surface attestation differs from the profile seal")
    declared_surface_sha256 = _required_attested_digest(
        surface_attestation, "surface_attestation_sha256"
    )
    unsigned = dict(surface_attestation)
    unsigned.pop("surface_attestation_sha256", None)
    if hashlib.sha256(_canonical_json(unsigned)).hexdigest() != declared_surface_sha256:
        raise EvaluationProfileError("surface attestation digest is inconsistent")
    state_root = Path(memory_root).expanduser() / ".evaluation_profiles" / namespace
    profile_path = state_root / "worker-attestation.json"
    expected_profile = _canonical_json(profile_attestation) + b"\n"
    try:
        observed_profile = _read_regular_file(profile_path)
    except OSError as exc:
        raise EvaluationProfileError("surface attestation profile seal is missing") from exc
    if observed_profile != expected_profile:
        raise EvaluationProfileError("surface attestation profile seal is missing or changed")
    surface_path = state_root / "worker-surface-attestation.json"
    expected_surface = _canonical_json(surface_attestation) + b"\n"
    if len(expected_surface) > _MAX_ATTESTATION_BYTES:
        raise EvaluationProfileError("evaluation surface attestation is too large")
    created = False
    if surface_path.exists() or surface_path.is_symlink():
        if _read_regular_file(surface_path) != expected_surface:
            raise EvaluationProfileError("existing worker surface attestation differs")
    else:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(surface_path, flags, 0o600)
        except FileExistsError:
            if _read_regular_file(surface_path) != expected_surface:
                raise EvaluationProfileError(
                    "concurrent worker surface attestation differs"
                ) from None
        else:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(expected_surface)
                handle.flush()
                os.fsync(handle.fileno())
            created = True
    return surface_path, created


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_text(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _read_regular_file(path: Path) -> bytes:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_ATTESTATION_BYTES:
        raise EvaluationProfileError("worker evaluation profile attestation is not regular/bounded")
    return path.read_bytes()


def _required_attested_digest(attestation: dict[str, Any], key: str) -> str:
    value = str(attestation.get(key) or "")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise EvaluationProfileError(f"evaluation profile attestation has invalid {key}")
    return value


def _surface_digest(surface: dict[str, str], key: str) -> str:
    value = surface.get(key)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EvaluationProfileError(f"evaluation surface has invalid {key}")
    return value


def _bounded_surface_identity(value: Any, key: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or len(value.encode("utf-8")) > 512
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise EvaluationProfileError(f"evaluation surface has invalid {key}")
    return value
