#!/usr/bin/env python3
"""Create and verify the trusted boundary for materials production promotion.

The readiness gate emits an evidence-qualified candidate.  This module closes
the restricted evidence tree, creates a sanitized public envelope, and permits
a full production-ready decision only after exact GitHub/Sigstore attestation
verification and independent revalidation of the retained bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

EVIDENCE_ROOT_SCHEMA = "ultra.materials.evidence-root.v1"
RELEASE_ENVELOPE_SCHEMA = "ultra.materials.release-envelope.v1"
FINAL_VERIFICATION_SCHEMA = "ultra.materials.production-attestation-verification.v1"
EVIDENCE_AGGREGATE_DOMAIN = b"ultra.materials.evidence-files.v1\0"
GITHUB_OIDC_ISSUER = "https://token.actions.githubusercontent.com"
SLSA_PREDICATE_TYPE = "https://slsa.dev/provenance/v1"
TRUSTED_REPOSITORY = "amilworks/ultra"
TRUSTED_REPOSITORY_ID = "1204778765"
TRUSTED_OWNER_ID = "22850980"
TRUSTED_SOURCE_REF = "refs/heads/main"
TRUSTED_WORKFLOW_PATH = ".github/workflows/materials-production-qualification.yml"
TRUSTED_ENVIRONMENT = "materials-production-qualification"

RUNNABLE_DENOMINATOR = 147
RUNNABLE_MINIMUM = 118
SCIENTIFIC_DENOMINATOR = 414
SCIENTIFIC_MINIMUM = 249
EXPECTED_TRIAL_COUNT = 3
PARENTS_PER_TRIAL = 49
SCIENTIFIC_SUBTASKS_PER_TRIAL = 138
PER_TRIAL_RUNNABLE_MINIMUM = 40
PER_TRIAL_SCIENTIFIC_MINIMUM = 83
MATERIALS_CLEANROOM_PROFILE = "materials_cleanroom_v1"
EXPECTED_CANDIDATE_FIXTURE_FILE_COUNT = 141
EXPECTED_CANDIDATE_FIXTURE_MANIFEST_SHA256 = (
    "296b5b55a5c1640999dd46556c2cd1a1487ae9de3e0f050fa601d3c5236bf308"
)
EXPECTED_CANDIDATE_VISIBLE_SOURCE_POLICY = "input-fixtures-only"
WORKER_CLEANROOM_DISABLED_CAPABILITIES = (
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
)
WORKER_EVALUATION_ATTESTATION_FIELDS = (
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
)

REQUIRED_READINESS_HARD_GATES = frozenset(
    {
        "mattools_server_authorized_cleanroom_profile",
        "mattools_worker_enforced_cleanroom_profile",
        "mattools_per_trial_function_runnable",
        "mattools_per_trial_strict_scientific_correctness",
    }
)
REQUIRED_MATTOOLS_HARD_GATES = frozenset(
    {
        "server_authorized_cleanroom_profile",
        "worker_enforced_cleanroom_profile",
        "per_trial_mattools_function_runnable",
        "per_trial_strict_scientific_task_success",
    }
)

REQUIRED_SINGLE_ROLES = frozenset(
    {
        "readiness_report",
        "readiness_manifest",
        "deterministic_report",
        "production_parity_report",
        "calphad_ledger_report",
        "calphad_cross_language_report",
        "calphad_cross_language_manifest",
        "mattools_report",
        "mattools_manifest",
        "release_tarball",
        "release_manifest",
    }
)
ROLE_RE = re.compile(r"^[a-z][a-z0-9_]*(?::[1-9][0-9]*)?$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DECIMAL_RE = re.compile(r"^[1-9][0-9]*$")
REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
WORKFLOW_PATH_RE = re.compile(r"^\.github/workflows/[A-Za-z0-9_.-]+\.ya?ml$")
FORBIDDEN_KEY_PARTS = frozenset(
    {
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "dsn",
        "passwd",
        "password",
        "private_key",
        "secret",
        "token",
    }
)
URI_USERINFO_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s/@:]+:[^\s/@]+@")
BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]+=*")
PRIVATE_KEY_RE = re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----")
KNOWN_TOKEN_RE = re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{20,})\b")
QUERY_SECRET_RE = re.compile(
    r"(?i)(?:[?&])(?:api[_-]?key|password|secret|signature|sig|token)=[^&#\s]+"
)
AWS_ACCESS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_EVIDENCE_FILES = 2_000_000


class PromotionEnvelopeError(RuntimeError):
    """An input failed a production-promotion trust requirement."""


def _fail(message: str) -> None:
    raise PromotionEnvelopeError(message)


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _reject_json_constant(value: str) -> None:
    _fail(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _strict_json_bytes(data: bytes, label: str) -> Any:
    if not data or len(data) > MAX_JSON_BYTES:
        _fail(f"{label} JSON size is outside 1..{MAX_JSON_BYTES} bytes")
    try:
        text = data.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"{label} is not strict UTF-8 JSON: {exc}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object")
    return value


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{label} must be a JSON array")
    return value


def _strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{label} must be an integer >= {minimum}")
    return value


def _strict_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be a boolean")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        _fail(f"{label} keys differ (missing={missing}, extra={extra})")


def _plain_sha256(value: Any, label: str) -> str:
    text = str(value or "")
    if not SHA256_RE.fullmatch(text):
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return text


def _image_digest(value: Any, label: str) -> str:
    text = str(value or "")
    if not IMAGE_DIGEST_RE.fullmatch(text):
        _fail(f"{label} must be sha256:<64 lowercase hex>")
    return text


def _git_sha(value: Any, label: str) -> str:
    text = str(value or "")
    if not GIT_SHA_RE.fullmatch(text):
        _fail(f"{label} must be a 40-character lowercase Git SHA")
    return text


def _positive_decimal(value: Any, label: str) -> str:
    text = str(value or "")
    if not DECIMAL_RE.fullmatch(text):
        _fail(f"{label} must be a canonical positive decimal string")
    return text


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _worker_cleanroom_attestation_valid(attempt: Mapping[str, Any]) -> bool:
    trace = attempt.get("trace_summary")
    if not isinstance(trace, dict):
        return False
    attestations = trace.get("worker_cleanroom_attestations")
    if not isinstance(attestations, list) or len(attestations) != 1:
        return False
    record = attestations[0]
    if not isinstance(record, dict):
        return False
    payload = record.get("payload")
    source_keys = record.get("source_payload_keys")
    if not isinstance(payload, dict) or source_keys != sorted(WORKER_EVALUATION_ATTESTATION_FIELDS):
        return False
    if set(payload) != set(WORKER_EVALUATION_ATTESTATION_FIELDS):
        return False
    unsigned = dict(payload)
    declared_attestation_sha = unsigned.pop("attestation_sha256", None)
    run_id = str(attempt.get("run_id") or "")
    thread_id = str(attempt.get("thread_id") or "")
    run_sha = _hash_text(run_id)
    thread_sha = _hash_text(thread_id)
    digest_names = ("run_id_sha256", "thread_id_sha256", "user_id_sha256", "goal_sha256")
    digests_valid = all(
        isinstance(payload.get(name), str) and SHA256_RE.fullmatch(payload[name])
        for name in digest_names
    )
    expected_attestation_sha = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    provided_count = payload.get("provided_message_count")
    binding = attempt.get("cleanroom_binding")
    if not isinstance(binding, dict):
        return False
    return all(
        (
            record.get("valid") is True,
            payload.get("schema_version") == "1",
            payload.get("attestation_kind") == "worker_evaluation_profile",
            payload.get("worker_owned") is True,
            payload.get("evaluation_profile") == MATERIALS_CLEANROOM_PROFILE,
            payload.get("profile_source") == "typed_job_envelope",
            payload.get("trusted_envelope_field") == "evaluation_profile",
            payload.get("namespace_id") == f"{MATERIALS_CLEANROOM_PROFILE}-{run_sha}",
            digests_valid,
            payload.get("run_id_sha256") == run_sha,
            payload.get("thread_id_sha256") == thread_sha,
            payload.get("input_policy") == "goal_only",
            type(provided_count) is int and provided_count >= 0,
            payload.get("effective_message_count") == 1,
            payload.get("prior_thread_context_discarded") is True,
            payload.get("same_run_retry_state_allowed") is True,
            payload.get("run_scoped_workspace") is True,
            payload.get("run_scoped_memory") is True,
            payload.get("disabled_capabilities") == list(WORKER_CLEANROOM_DISABLED_CAPABILITIES),
            isinstance(declared_attestation_sha, str),
            SHA256_RE.fullmatch(str(declared_attestation_sha)) is not None,
            declared_attestation_sha == expected_attestation_sha,
            binding.get("evaluation_profile") == MATERIALS_CLEANROOM_PROFILE,
            binding.get("worker_event_count") == 1,
            binding.get("worker_attestation_valid") is True,
            binding.get("server_attestation_valid") is True,
            binding.get("identity_hash_checks")
            == {
                "run_id_sha256": True,
                "thread_id_sha256": True,
                "goal_sha256": True,
                "user_id_sha256": True,
            },
            binding.get("user_identity_independently_bound") is True,
            binding.get("valid") is True,
        )
    )


def _ensure_regular_path(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        _fail(f"{label} cannot be inspected: {exc}")
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        _fail(f"{label} must be a non-symlink regular file")
    if info.st_nlink != 1:
        _fail(f"{label} must not be hard-linked")
    return info


def _read_regular_bytes(path: Path, label: str, *, maximum: int | None = None) -> bytes:
    before = _ensure_regular_path(path, label)
    if maximum is not None and (before.st_size <= 0 or before.st_size > maximum):
        _fail(f"{label} size is outside 1..{maximum} bytes")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        _fail(f"{label} cannot be opened safely: {exc}")
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            _fail(f"{label} changed to a non-regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if maximum is not None and total > maximum:
                _fail(f"{label} exceeds {maximum} bytes")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_opened = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if (
        identity_before != identity_opened
        or identity_opened != identity_after
        or total != after.st_size
    ):
        _fail(f"{label} changed while it was being read")
    return b"".join(chunks)


def _hash_regular_file(path: Path, label: str) -> tuple[str, int]:
    before = _ensure_regular_path(path, label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        _fail(f"{label} cannot be opened safely: {exc}")
    digest = hashlib.sha256()
    total = 0
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            _fail(f"{label} changed to a non-regular file")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_opened = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if (
        identity_before != identity_opened
        or identity_opened != identity_after
        or total != after.st_size
    ):
        _fail(f"{label} changed while it was being hashed")
    return digest.hexdigest(), total


def _strict_json_path(path: Path, label: str) -> Mapping[str, Any]:
    return _mapping(
        _strict_json_bytes(_read_regular_bytes(path, label, maximum=MAX_JSON_BYTES), label),
        label,
    )


def _write_once(path: Path, data: bytes, label: str) -> None:
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        existing = _read_regular_bytes(path, label)
        if existing != data:
            _fail(f"{label} already exists with different bytes")
        return
    except OSError as exc:
        _fail(f"{label} cannot be created safely: {exc}")
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _fail(f"{label} write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _relative_posix_path(value: str, label: str) -> str:
    if not value or "\\" in value or "\x00" in value:
        _fail(f"{label} must be a non-empty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        _fail(f"{label} must not be absolute, empty, dot, or traversing")
    normalized = path.as_posix()
    if normalized != value:
        _fail(f"{label} must already be normalized")
    return normalized


def _root_path(path: Path, label: str) -> Path:
    expanded = path.expanduser().absolute()
    try:
        info = expanded.lstat()
    except OSError as exc:
        _fail(f"{label} cannot be inspected: {exc}")
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        _fail(f"{label} must be a non-symlink directory")
    return expanded


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.expanduser().absolute().relative_to(root)
        return True
    except ValueError:
        return False


def _scan_evidence_tree(root: Path) -> list[dict[str, Any]]:
    root = _root_path(root, "evidence root")
    records: list[dict[str, Any]] = []

    def visit(directory: Path, prefix: PurePosixPath | None) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            _fail(f"evidence directory cannot be scanned: {exc}")
        for entry in entries:
            if entry.name in {".", ".."} or "/" in entry.name or "\\" in entry.name:
                _fail("evidence entry has an unsafe name")
            relative = PurePosixPath(entry.name) if prefix is None else prefix / entry.name
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                _fail(f"evidence entry {relative.as_posix()} cannot be inspected: {exc}")
            mode = info.st_mode
            if stat.S_ISLNK(mode):
                _fail(f"evidence entry {relative.as_posix()} is a symlink")
            if stat.S_ISDIR(mode):
                visit(Path(entry.path), relative)
                continue
            if not stat.S_ISREG(mode):
                _fail(f"evidence entry {relative.as_posix()} is not a regular file")
            digest, size = _hash_regular_file(
                Path(entry.path), f"evidence file {relative.as_posix()}"
            )
            records.append({"path": relative.as_posix(), "sha256": digest, "size_bytes": size})
            if len(records) > MAX_EVIDENCE_FILES:
                _fail(f"evidence root exceeds {MAX_EVIDENCE_FILES} files")

    visit(root, None)
    records.sort(key=lambda item: item["path"])
    return records


def _parse_roles(values: Sequence[str]) -> dict[str, str]:
    roles: dict[str, str] = {}
    paths: set[str] = set()
    for value in values:
        name, separator, raw_path = value.partition("=")
        if separator != "=" or not ROLE_RE.fullmatch(name):
            _fail("each --role must be role_name=relative/posix/path")
        relative = _relative_posix_path(raw_path, f"role {name} path")
        if name in roles:
            _fail(f"duplicate evidence role: {name}")
        if relative in paths:
            _fail(f"duplicate evidence role path: {relative}")
        roles[name] = relative
        paths.add(relative)
    missing = sorted(REQUIRED_SINGLE_ROLES - set(roles))
    if missing:
        _fail(f"required evidence roles are missing: {missing}")
    live_indexes: list[int] = []
    for name in roles:
        if name.startswith("live_trace:"):
            live_indexes.append(int(name.split(":", 1)[1]))
    live_indexes.sort()
    if not live_indexes or live_indexes != list(range(1, len(live_indexes) + 1)):
        _fail("live_trace roles must be contiguous from live_trace:1")
    return roles


def _aggregate_files(files: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(EVIDENCE_AGGREGATE_DOMAIN)
    digest.update(_canonical_json_bytes(files))
    return digest.hexdigest()


def _source_context(args: argparse.Namespace, workflow_sha256: str) -> dict[str, Any]:
    repository = str(args.repository or "")
    if repository != TRUSTED_REPOSITORY or not REPOSITORY_RE.fullmatch(repository):
        _fail(f"repository must be the trusted {TRUSTED_REPOSITORY} repository")
    repository_id = _positive_decimal(args.repository_id, "repository ID")
    owner_id = _positive_decimal(args.owner_id, "owner ID")
    if repository_id != TRUSTED_REPOSITORY_ID or owner_id != TRUSTED_OWNER_ID:
        _fail("repository or owner ID differs from the trusted stable identity")
    git_sha = _git_sha(args.source_git_sha, "source Git SHA")
    source_ref = str(args.source_ref or "")
    if source_ref != TRUSTED_SOURCE_REF:
        _fail(f"source ref must be {TRUSTED_SOURCE_REF}")
    workflow_path = _relative_posix_path(str(args.workflow_path or ""), "workflow path")
    if workflow_path != TRUSTED_WORKFLOW_PATH or not WORKFLOW_PATH_RE.fullmatch(workflow_path):
        _fail(f"workflow path must be {TRUSTED_WORKFLOW_PATH}")
    signer_digest = _git_sha(args.workflow_signer_digest, "workflow signer digest")
    run_id = _positive_decimal(args.run_id, "GitHub run ID")
    run_attempt = _strict_int(args.run_attempt, "GitHub run attempt", minimum=1)
    event_name = str(args.event_name or "")
    if event_name != "workflow_dispatch":
        _fail("production qualification must be triggered by workflow_dispatch")
    environment = str(args.environment or "")
    if environment != TRUSTED_ENVIRONMENT:
        _fail("unexpected protected environment name")
    return {
        "repository": repository,
        "repository_id": repository_id,
        "owner_id": owner_id,
        "git_sha": git_sha,
        "ref": source_ref,
        "workflow_path": workflow_path,
        "workflow_file_sha256": workflow_sha256,
        "workflow_signer_digest": signer_digest,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "event_name": event_name,
        "environment": environment,
    }


def _role_records(files: list[dict[str, Any]], roles: Mapping[str, str]) -> list[dict[str, Any]]:
    by_path = {str(item["path"]): item for item in files}
    records: list[dict[str, Any]] = []
    for name, relative in sorted(roles.items()):
        file_record = by_path.get(relative)
        if file_record is None:
            _fail(f"evidence role {name} does not identify a retained file")
        records.append(
            {
                "name": name,
                "path": relative,
                "sha256": file_record["sha256"],
                "size_bytes": file_record["size_bytes"],
            }
        )
    return records


def _build_root_manifest(
    *, root: Path, roles: Mapping[str, str], context: Mapping[str, Any]
) -> dict[str, Any]:
    files = _scan_evidence_tree(root)
    if not files:
        _fail("evidence root is empty")
    total_size = sum(_strict_int(item["size_bytes"], "evidence file size") for item in files)
    return {
        "schema_version": EVIDENCE_ROOT_SCHEMA,
        "hash_algorithm": "sha256",
        "source": dict(context),
        "file_count": len(files),
        "total_size_bytes": total_size,
        "aggregate_sha256": _aggregate_files(files),
        "files": files,
        "roles": _role_records(files, roles),
    }


def _validate_root_manifest_shape(manifest: Mapping[str, Any]) -> None:
    _exact_keys(
        manifest,
        {
            "schema_version",
            "hash_algorithm",
            "source",
            "file_count",
            "total_size_bytes",
            "aggregate_sha256",
            "files",
            "roles",
        },
        "evidence root manifest",
    )
    if manifest.get("schema_version") != EVIDENCE_ROOT_SCHEMA:
        _fail("unexpected evidence root schema")
    if manifest.get("hash_algorithm") != "sha256":
        _fail("evidence root hash algorithm must be sha256")
    _mapping(manifest.get("source"), "evidence root source")
    _strict_int(manifest.get("file_count"), "evidence root file count", minimum=1)
    _strict_int(manifest.get("total_size_bytes"), "evidence root total size")
    _plain_sha256(manifest.get("aggregate_sha256"), "evidence aggregate")
    files = _sequence(manifest.get("files"), "evidence root files")
    roles = _sequence(manifest.get("roles"), "evidence root roles")
    if len(files) > MAX_EVIDENCE_FILES:
        _fail("evidence root manifest has too many files")
    previous_path = ""
    seen_paths: set[str] = set()
    for index, raw in enumerate(files):
        item = _mapping(raw, f"evidence file {index}")
        _exact_keys(item, {"path", "sha256", "size_bytes"}, f"evidence file {index}")
        relative = _relative_posix_path(str(item.get("path") or ""), f"evidence file {index}")
        if relative in seen_paths or (previous_path and relative <= previous_path):
            _fail("evidence file paths must be unique and sorted")
        seen_paths.add(relative)
        previous_path = relative
        _plain_sha256(item.get("sha256"), f"evidence file {index} digest")
        _strict_int(item.get("size_bytes"), f"evidence file {index} size")
    previous_role = ""
    seen_roles: set[str] = set()
    role_paths: set[str] = set()
    for index, raw in enumerate(roles):
        item = _mapping(raw, f"evidence role {index}")
        _exact_keys(
            item,
            {"name", "path", "sha256", "size_bytes"},
            f"evidence role {index}",
        )
        name = str(item.get("name") or "")
        if (
            not ROLE_RE.fullmatch(name)
            or name in seen_roles
            or (previous_role and name <= previous_role)
        ):
            _fail("evidence roles must have unique sorted names")
        path = _relative_posix_path(str(item.get("path") or ""), f"evidence role {name}")
        if path in role_paths:
            _fail("evidence roles must identify unique files")
        seen_roles.add(name)
        role_paths.add(path)
        previous_role = name
        _plain_sha256(item.get("sha256"), f"evidence role {name} digest")
        _strict_int(item.get("size_bytes"), f"evidence role {name} size")


def verify_evidence_root(
    evidence_root: Path, evidence_root_manifest: Path
) -> tuple[Mapping[str, Any], str, int]:
    root = _root_path(evidence_root, "evidence root")
    manifest_path = evidence_root_manifest.expanduser().absolute()
    if _is_within(manifest_path, root):
        _fail("evidence root manifest must be outside the evidence root")
    manifest_bytes = _read_regular_bytes(
        manifest_path, "evidence root manifest", maximum=MAX_JSON_BYTES
    )
    manifest = _mapping(
        _strict_json_bytes(manifest_bytes, "evidence root manifest"),
        "evidence root manifest",
    )
    if manifest_bytes != _canonical_json_bytes(manifest):
        _fail("evidence root manifest is not canonical JSON")
    _validate_root_manifest_shape(manifest)
    observed_files = _scan_evidence_tree(root)
    if observed_files != manifest.get("files"):
        _fail("retained evidence tree differs from the exact closure manifest")
    expected_count = _strict_int(manifest.get("file_count"), "evidence root file count", minimum=1)
    expected_total = _strict_int(manifest.get("total_size_bytes"), "evidence root total size")
    if expected_count != len(observed_files):
        _fail("evidence root file count is inconsistent")
    observed_total = sum(int(item["size_bytes"]) for item in observed_files)
    if expected_total != observed_total:
        _fail("evidence root total size is inconsistent")
    if manifest.get("aggregate_sha256") != _aggregate_files(observed_files):
        _fail("evidence root aggregate digest is inconsistent")
    by_path = {str(item["path"]): item for item in observed_files}
    roles = _sequence(manifest.get("roles"), "evidence root roles")
    role_names: dict[str, str] = {}
    for raw in roles:
        role = _mapping(raw, "evidence role")
        name = str(role["name"])
        path = str(role["path"])
        bound = by_path.get(path)
        if (
            bound is None
            or role.get("sha256") != bound["sha256"]
            or role.get("size_bytes") != bound["size_bytes"]
        ):
            _fail(f"evidence role {name} is not bound to the closed file")
        role_names[name] = path
    _parse_roles([f"{name}={path}" for name, path in role_names.items()])
    return manifest, hashlib.sha256(manifest_bytes).hexdigest(), len(manifest_bytes)


def _role_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(item["name"]): _mapping(item, "evidence role")
        for item in _sequence(manifest.get("roles"), "evidence root roles")
    }


def _role_path(root: Path, roles: Mapping[str, Mapping[str, Any]], name: str) -> Path:
    role = roles.get(name)
    if role is None:
        _fail(f"required role is absent: {name}")
    relative = _relative_posix_path(str(role.get("path") or ""), f"role {name} path")
    return root / Path(*PurePosixPath(relative).parts)


def _validate_readiness_candidate(report: Mapping[str, Any]) -> dict[str, Any]:
    if (
        report.get("schema_version") != "1"
        or report.get("gate") != "materials-production-readiness"
    ):
        _fail("readiness report has the wrong schema or gate")
    if report.get("scope") != "full-materials-production-readiness":
        _fail("readiness report has the wrong scope")
    if report.get("status") != "candidate_for_attestation":
        _fail("readiness report is not a candidate for attestation")
    hard_gates = _mapping(report.get("hard_gates"), "readiness hard gates")
    if not hard_gates or any(value is not True for value in hard_gates.values()):
        _fail("every readiness hard gate must be exactly true")
    if not REQUIRED_READINESS_HARD_GATES.issubset(hard_gates):
        _fail("readiness report lacks the clean-room or per-trial hard gates")
    promotion = _mapping(report.get("promotion"), "readiness promotion")
    required_promotion = {
        "passed": True,
        "evidence_passed": True,
        "attestation_required": True,
        "distribution_ready": False,
        "full_materials_production_ready": False,
    }
    for name, expected in required_promotion.items():
        if promotion.get(name) is not expected:
            _fail(f"readiness promotion {name} must be {expected}")
    if promotion.get("reasons") != []:
        _fail("readiness candidate must have no blocking reasons")
    counts = _mapping(report.get("counts"), "readiness counts")
    parity = _mapping(counts.get("production_parity"), "production parity counts")
    if parity.get("scope") != "production-full" or parity.get("passed") is not True:
        _fail("full production image parity did not pass")
    ledger = _mapping(counts.get("calphad_ledger"), "CALPHAD ledger counts")
    if ledger.get("passed") is not True:
        _fail("CALPHAD PostgreSQL ledger qualification did not pass")
    cross = _mapping(counts.get("calphad_cross_language"), "CALPHAD cross-language counts")
    if any(
        cross.get(name) is not True for name in ("passed", "live_http_callback", "live_postgres")
    ):
        _fail("CALPHAD typed CLI/HTTP/PostgreSQL qualification did not pass")
    deterministic = _mapping(counts.get("deterministic"), "deterministic counts")
    _strict_int(deterministic.get("total"), "deterministic total count", minimum=13)
    if (
        deterministic.get("passed") is not True
        or _strict_int(deterministic.get("skipped"), "deterministic skipped count") != 0
    ):
        _fail("deterministic materials evidence is incomplete or skipped")
    traces = _strict_int(counts.get("designated_live_traces"), "designated live traces")
    if traces < 1:
        _fail("at least one designated live trace is required")
    mattools = _mapping(counts.get("mattools"), "MatTools counts")
    runnable = _strict_int(mattools.get("runnable"), "MatTools runnable count")
    runnable_denominator = _strict_int(
        mattools.get("runnable_denominator"), "MatTools runnable denominator", minimum=1
    )
    runnable_minimum = _strict_int(
        mattools.get("runnable_minimum"), "MatTools runnable minimum", minimum=1
    )
    per_trial_runnable_minimum = _strict_int(
        mattools.get("per_trial_runnable_minimum"),
        "MatTools per-trial runnable minimum",
        minimum=1,
    )
    official = _strict_int(mattools.get("scientific_pass"), "MatTools official scientific count")
    strict = _strict_int(mattools.get("strict_scientific_pass"), "MatTools strict scientific count")
    scientific_denominator = _strict_int(
        mattools.get("scientific_denominator"), "MatTools scientific denominator", minimum=1
    )
    scientific_minimum = _strict_int(
        mattools.get("scientific_minimum"), "MatTools scientific minimum", minimum=1
    )
    per_trial_scientific_minimum = _strict_int(
        mattools.get("per_trial_scientific_minimum"),
        "MatTools per-trial scientific minimum",
        minimum=1,
    )
    per_trial = _sequence(mattools.get("per_trial"), "readiness MatTools per-trial counts")
    if len(per_trial) != EXPECTED_TRIAL_COUNT:
        _fail("readiness MatTools counts must retain exactly three trial summaries")
    recomputed_runnable = 0
    recomputed_official = 0
    recomputed_strict = 0
    for expected_trial, value in enumerate(per_trial, start=1):
        trial = _mapping(value, f"readiness MatTools trial {expected_trial}")
        trial_runnable = _strict_int(
            trial.get("runnable"), f"readiness MatTools trial {expected_trial} runnable"
        )
        trial_official = _strict_int(
            trial.get("scientific_pass"),
            f"readiness MatTools trial {expected_trial} official scientific",
        )
        trial_strict = _strict_int(
            trial.get("strict_scientific_pass"),
            f"readiness MatTools trial {expected_trial} strict scientific",
        )
        if (
            trial.get("trial") != expected_trial
            or trial_runnable < PER_TRIAL_RUNNABLE_MINIMUM
            or trial_runnable > PARENTS_PER_TRIAL
            or trial_strict < PER_TRIAL_SCIENTIFIC_MINIMUM
            or trial_strict > SCIENTIFIC_SUBTASKS_PER_TRIAL
            or trial_official < 0
            or trial_official > SCIENTIFIC_SUBTASKS_PER_TRIAL
        ):
            _fail("readiness MatTools per-trial counts violate the fixed policy")
        recomputed_runnable += trial_runnable
        recomputed_official += trial_official
        recomputed_strict += trial_strict
    if (
        runnable_denominator != RUNNABLE_DENOMINATOR
        or runnable_minimum != RUNNABLE_MINIMUM
        or per_trial_runnable_minimum != PER_TRIAL_RUNNABLE_MINIMUM
        or scientific_denominator != SCIENTIFIC_DENOMINATOR
        or scientific_minimum != SCIENTIFIC_MINIMUM
        or per_trial_scientific_minimum != PER_TRIAL_SCIENTIFIC_MINIMUM
        or recomputed_runnable != runnable
        or recomputed_official != official
        or recomputed_strict != strict
        or runnable < RUNNABLE_MINIMUM
        or runnable > RUNNABLE_DENOMINATOR
        or official < SCIENTIFIC_MINIMUM
        or official > SCIENTIFIC_DENOMINATOR
        or strict < SCIENTIFIC_MINIMUM
        or strict > SCIENTIFIC_DENOMINATOR
    ):
        _fail("MatTools counts do not meet the fixed published thresholds")
    rates = _mapping(report.get("rates"), "readiness rates")
    expected_rates = {
        "mattools_function_runnable": runnable / RUNNABLE_DENOMINATOR,
        "mattools_official_task_success": official / SCIENTIFIC_DENOMINATOR,
        "mattools_strict_task_success": strict / SCIENTIFIC_DENOMINATOR,
    }
    for name, expected in expected_rates.items():
        value = rates.get(name)
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not math.isclose(float(value), expected, rel_tol=0, abs_tol=1e-12)
        ):
            _fail(f"readiness rate {name} is inconsistent")
    return {
        "counts": {
            "production_parity": dict(parity),
            "calphad_ledger": dict(ledger),
            "calphad_cross_language": dict(cross),
            "deterministic": dict(deterministic),
            "mattools": dict(mattools),
            "designated_live_traces": traces,
        },
        "rates": {name: rates[name] for name in sorted(expected_rates)},
    }


def _validate_readiness_manifest(
    manifest: Mapping[str, Any], report_record: Mapping[str, Any], report_path: Path
) -> None:
    if manifest.get("schema_version") != "1":
        _fail("readiness manifest has the wrong schema")
    for name, expected in {
        "promotion_passed": True,
        "evidence_passed": True,
        "attestation_required": True,
        "full_materials_production_ready": False,
    }.items():
        if manifest.get(name) is not expected:
            _fail(f"readiness manifest {name} must be {expected}")
    report = _mapping(manifest.get("report"), "readiness manifest report")
    expected_name = (
        "materials-production-readiness-"
        f"{_plain_sha256(report_record.get('sha256'), 'readiness report digest')}.json"
    )
    if (
        report_path.name != expected_name
        or report.get("path") != expected_name
        or report.get("sha256") != report_record.get("sha256")
        or report.get("size_bytes") != report_record.get("size_bytes")
    ):
        _fail("readiness manifest does not bind the content-addressed retained readiness report")


def _validate_release_manifest(
    manifest: Mapping[str, Any], git_sha: str, expected_run_id: str
) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("release_sha") != git_sha:
        _fail("release manifest schema or source SHA is inconsistent")
    source = _mapping(manifest.get("source"), "release source")
    if (
        source.get("repository") != TRUSTED_REPOSITORY
        or source.get("ref") not in {"main", TRUSTED_SOURCE_REF}
        or str(source.get("github_run_id") or "") != expected_run_id
    ):
        _fail("release manifest source repository, ref, or run ID is inconsistent")
    materials = _mapping(manifest.get("materials"), "release materials policy")
    if materials.get("full_materials_production_ready") is not False:
        _fail("source release manifest must not claim materials production readiness")
    if materials.get("required_post_image_gate") != "materials-production-readiness":
        _fail("release manifest has the wrong required post-image gate")
    required = _mapping(materials.get("required_evidence"), "release required materials evidence")
    if (
        required.get("production_parity_scope") != "production-full"
        or required.get("calphad_cross_language_requires_production_runtime_image") is not True
        or required.get("mattools_runnable_minimum") != RUNNABLE_MINIMUM
        or required.get("mattools_scientific_minimum") != SCIENTIFIC_MINIMUM
    ):
        _fail("release manifest does not encode the production materials policy")


def _validate_release_tarball(
    tarball_path: Path,
    release_manifest_path: Path,
    release_manifest: Mapping[str, Any],
) -> None:
    release_name = str(release_manifest.get("release_name") or "")
    if (
        not release_name
        or "/" in release_name
        or "\\" in release_name
        or release_name in {".", ".."}
    ):
        _fail("release manifest has an unsafe release name")
    expected_manifest_name = f"{release_name}/release-manifest.json"
    expected_manifest_bytes = _read_regular_bytes(
        release_manifest_path, "release manifest", maximum=MAX_JSON_BYTES
    )
    names: set[str] = set()
    embedded_manifest: bytes | None = None
    total_regular_size = 0
    try:
        with tarfile.open(tarball_path, mode="r:*") as archive:
            member_count = 0
            for member in archive:
                member_count += 1
                if member_count > MAX_EVIDENCE_FILES:
                    _fail("release tarball has too many members")
                name = member.name.rstrip("/")
                normalized = _relative_posix_path(name, "release tar member")
                if normalized in names:
                    _fail("release tarball contains duplicate members")
                names.add(normalized)
                if not (normalized == release_name or normalized.startswith(f"{release_name}/")):
                    _fail("release tarball contains a member outside its release prefix")
                if member.isdir():
                    continue
                if not member.isreg():
                    _fail("release tarball contains a link, device, or other non-regular member")
                total_regular_size += member.size
                if total_regular_size > 20 * 1024**3:
                    _fail("release tarball expands beyond the 20 GiB policy limit")
                if normalized == expected_manifest_name:
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        _fail("embedded release manifest cannot be read")
                    embedded_manifest = extracted.read(MAX_JSON_BYTES + 1)
            if member_count == 0:
                _fail("release tarball is empty")
    except (OSError, tarfile.TarError) as exc:
        _fail(f"release tarball cannot be parsed safely: {type(exc).__name__}")
    if embedded_manifest != expected_manifest_bytes:
        _fail("release tarball does not contain the byte-identical release manifest")


def _validate_direct_lane_reports(
    *,
    root: Path,
    root_manifest: Mapping[str, Any],
    roles: Mapping[str, Mapping[str, Any]],
    git_sha: str,
    runtime_image_id: str,
    readiness: Mapping[str, Any],
) -> None:
    deterministic = _strict_json_path(
        _role_path(root, roles, "deterministic_report"), "deterministic materials report"
    )
    if (
        deterministic.get("schema_version") != 1
        or deterministic.get("gate") != "materials-domain-gate"
        or deterministic.get("scope") != "deterministic-domain-invariants"
        or deterministic.get("status") != "passed"
        or deterministic.get("failures") != []
    ):
        _fail("deterministic materials role is not a passing domain gate report")

    parity = _strict_json_path(
        _role_path(root, roles, "production_parity_report"), "production parity report"
    )
    parity_bundle = _mapping(parity.get("evidence_bundle"), "production parity bundle")
    executed_image = _mapping(parity.get("executed_image"), "production executed image")
    if (
        parity.get("schema_version") != 1
        or parity.get("gate") != "production-materials-sandbox-parity"
        or parity.get("scope") != "production-full"
        or parity.get("status") != "passed"
        or parity.get("failures") != []
        or parity.get("expected_git_sha") != git_sha
        or parity.get("full_production_image_parity") is not True
        or parity_bundle.get("promotable") is not True
        or executed_image.get("image_id") != runtime_image_id
        or executed_image.get("revision") != git_sha
    ):
        _fail("production parity role is not a promotable full-image report")

    ledger = _strict_json_path(
        _role_path(root, roles, "calphad_ledger_report"), "CALPHAD ledger report"
    )
    if (
        ledger.get("schema_version") != "1"
        or ledger.get("gate") != "calphad-ledger-postgres-qualification"
        or ledger.get("status") != "passed"
        or ledger.get("qualification_database") is not True
        or ledger.get("production_database_used") is not False
        or ledger.get("git_sha") != git_sha
        or ledger.get("repository_clean") is not True
        or ledger.get("failures") != []
    ):
        _fail("CALPHAD ledger role is not a clean dedicated-PostgreSQL qualification")

    cross_path = _role_path(root, roles, "calphad_cross_language_report")
    cross = _strict_json_path(cross_path, "CALPHAD cross-language report")
    backend = _mapping(cross.get("backend"), "CALPHAD cross-language backend")
    generation = _mapping(cross.get("generation"), "CALPHAD cross-language generation")
    if (
        cross.get("schema_version") != "ultra.calphad.cross-language-gate.v1"
        or cross.get("gate") != "calphad-typed-cli-http-postgres-cross-language"
        or cross.get("status") != "qualified"
        or cross.get("production_live_qualified") is not True
        or cross.get("promotable") is not True
        or cross.get("expected_git_sha") != git_sha
        or backend.get("live_http_callback") is not True
        or backend.get("live_postgres") is not True
        or generation.get("mode") != "pinned-image"
        or generation.get("image_inspected") is not True
        or generation.get("runtime_image_id") != runtime_image_id
        or backend.get("runtime_image_id") != runtime_image_id
    ):
        _fail("CALPHAD cross-language role is not a live production qualification")
    cross_manifest = _strict_json_path(
        _role_path(root, roles, "calphad_cross_language_manifest"),
        "CALPHAD cross-language manifest",
    )
    cross_bound = _mapping(cross_manifest.get("report"), "CALPHAD report binding")
    cross_record = roles["calphad_cross_language_report"]
    if (
        cross_manifest.get("schema_version") != "ultra.calphad.cross-language-report-manifest.v1"
        or cross_manifest.get("production_live_qualified") is not True
        or cross_manifest.get("expected_git_sha") != git_sha
        or cross_manifest.get("runtime_image_id") != runtime_image_id
        or PurePosixPath(str(cross_bound.get("path") or "")).name != cross_path.name
        or cross_bound.get("sha256") != cross_record.get("sha256")
        or cross_bound.get("size_bytes") != cross_record.get("size_bytes")
    ):
        _fail("CALPHAD cross-language manifest does not bind the retained live report")

    inputs = _mapping(readiness.get("inputs"), "readiness input identities")
    input_roles = {
        "deterministic_report": "deterministic_report",
        "production_parity_report": "production_parity_report",
        "calphad_ledger_report": "calphad_ledger_report",
        "calphad_cross_language_report": "calphad_cross_language_report",
        "calphad_cross_language_report_manifest": "calphad_cross_language_manifest",
        "mattools_report": "mattools_report",
        "mattools_report_manifest": "mattools_manifest",
    }
    for input_name, role_name in input_roles.items():
        metadata = _mapping(inputs.get(input_name), f"readiness input {input_name}")
        role = roles[role_name]
        expected_path = _role_path(root, roles, role_name).absolute()
        observed_path = Path(str(metadata.get("path") or "")).expanduser().absolute()
        if (
            observed_path != expected_path
            or metadata.get("sha256") != role.get("sha256")
            or metadata.get("size_bytes") != role.get("size_bytes")
        ):
            _fail(f"readiness input {input_name} does not bind its retained role")
    trace_metadata = _sequence(inputs.get("live_trace_reports"), "readiness live trace inputs")
    live_roles = [name for name in sorted(roles) if name.startswith("live_trace:")]
    if len(trace_metadata) != len(live_roles):
        _fail("readiness live trace inputs differ from the retained live trace roles")
    for metadata_value, role_name in zip(trace_metadata, live_roles):
        metadata = _mapping(metadata_value, f"readiness input {role_name}")
        role = roles[role_name]
        if (
            Path(str(metadata.get("path") or "")).expanduser().absolute()
            != _role_path(root, roles, role_name).absolute()
            or metadata.get("sha256") != role.get("sha256")
            or metadata.get("size_bytes") != role.get("size_bytes")
        ):
            _fail(f"readiness input {role_name} does not bind its retained role")

    files_by_path = {
        str(item["path"]): item
        for item in _sequence(root_manifest.get("files"), "evidence root files")
    }
    mattools_path = _role_path(root, roles, "mattools_report")
    mattools_report = _strict_json_path(mattools_path, "MatTools report")
    mattools_manifest = _strict_json_path(
        _role_path(root, roles, "mattools_manifest"), "MatTools report manifest"
    )
    if (
        mattools_manifest.get("schema_version") != "2"
        or mattools_manifest.get("manifest_kind") != "ultra.mattools.report_bundle.v2"
        or mattools_manifest.get("campaign_id") != mattools_report.get("campaign_id")
    ):
        _fail("MatTools report manifest has the wrong regeneration contract")
    regeneration = _mapping(mattools_manifest.get("regeneration"), "MatTools regeneration")
    if dict(regeneration) != {
        "helper": "revalidate_report_bundle",
        "cli_subcommand": "verify-report",
        "comparison": "byte_exact",
        "task_execution_performed": False,
    }:
        _fail("MatTools report manifest is not byte-exact/no-task regeneration")
    for key in ("results_json", "results_markdown", "checkpoint"):
        binding = _mapping(mattools_manifest.get(key), f"MatTools manifest {key}")
        absolute = Path(str(binding.get("path") or "")).expanduser().absolute()
        try:
            relative = absolute.relative_to(root).as_posix()
        except ValueError:
            _fail(f"MatTools manifest {key} path escapes the retained evidence root")
        closed = files_by_path.get(relative)
        if closed is None or binding.get("sha256") != closed.get("sha256"):
            _fail(f"MatTools manifest {key} is not in the exact retained closure")
        if key == "results_json" and absolute != mattools_path.absolute():
            _fail("MatTools results binding differs from the retained report role")


def _validate_mattools_identity(
    report: Mapping[str, Any],
    *,
    source_git_sha: str,
    expected_runtime_image: str,
    license_basis: str | None,
    license_purpose: str | None,
    license_evidence_sha256: str | None,
    model_identity: str | None,
    provider_identity: str | None,
) -> dict[str, Any]:
    if report.get("schema_version") != "1":
        _fail("MatTools report has the wrong schema")
    if not str(report.get("campaign_id") or "").strip():
        _fail("MatTools report lacks a campaign identity")
    promotion = _mapping(report.get("promotion"), "MatTools promotion")
    if (
        promotion.get("scope") != "MatTools benchmark lane only"
        or promotion.get("passed") is not True
        or promotion.get("full_materials_production_ready") is not False
        or promotion.get("reasons") != []
    ):
        _fail("MatTools benchmark lane did not pass")
    hard_gates = _mapping(report.get("hard_gates"), "MatTools hard gates")
    if not hard_gates or any(value is not True for value in hard_gates.values()):
        _fail("every MatTools hard gate must be exactly true")
    if not REQUIRED_MATTOOLS_HARD_GATES.issubset(hard_gates):
        _fail("MatTools report lacks the clean-room or per-trial hard gates")
    counts = _mapping(report.get("counts"), "MatTools counts")
    expected_count_policy = {
        "runnable_denominator": RUNNABLE_DENOMINATOR,
        "runnable_minimum": RUNNABLE_MINIMUM,
        "per_trial_runnable_minimum": PER_TRIAL_RUNNABLE_MINIMUM,
        "scientific_denominator": SCIENTIFIC_DENOMINATOR,
        "scientific_minimum": SCIENTIFIC_MINIMUM,
        "per_trial_scientific_minimum": PER_TRIAL_SCIENTIFIC_MINIMUM,
        "terminal_attempts": RUNNABLE_DENOMINATOR,
        "expected_attempts_for_configured_run": RUNNABLE_DENOMINATOR,
    }
    for name, expected_value in expected_count_policy.items():
        if _strict_int(counts.get(name), f"MatTools {name}") != expected_value:
            _fail(f"MatTools {name} differs from the fixed benchmark policy")
    for name, minimum, denominator in (
        ("runnable", RUNNABLE_MINIMUM, RUNNABLE_DENOMINATOR),
        ("scientific_pass", SCIENTIFIC_MINIMUM, SCIENTIFIC_DENOMINATOR),
        ("strict_scientific_pass", SCIENTIFIC_MINIMUM, SCIENTIFIC_DENOMINATOR),
    ):
        count = _strict_int(counts.get(name), f"MatTools {name}")
        if count < minimum or count > denominator:
            _fail(f"MatTools {name} does not meet the fixed benchmark threshold")
    rates = _mapping(report.get("rates"), "MatTools rates")
    expected_rates = {
        "function_runnable": counts["runnable"] / RUNNABLE_DENOMINATOR,
        "task_success": counts["scientific_pass"] / SCIENTIFIC_DENOMINATOR,
        "strict_task_success": counts["strict_scientific_pass"] / SCIENTIFIC_DENOMINATOR,
    }
    for name, expected_rate in expected_rates.items():
        observed_rate = rates.get(name)
        if (
            not isinstance(observed_rate, int | float)
            or isinstance(observed_rate, bool)
            or not math.isfinite(float(observed_rate))
            or not math.isclose(float(observed_rate), expected_rate, rel_tol=0, abs_tol=1e-12)
        ):
            _fail(f"MatTools {name} is inconsistent with its retained counts")
    ultra = _mapping(report.get("ultra"), "MatTools Ultra provenance")
    if ultra.get("commit") != source_git_sha or ultra.get("dirty") is not False:
        _fail("MatTools report is not bound to the clean source commit")
    runtime = _mapping(report.get("runtime_environment"), "MatTools runtime environment")
    if (
        runtime.get("image_digest") != expected_runtime_image
        or runtime.get("evaluation_profile") != MATERIALS_CLEANROOM_PROFILE
    ):
        _fail("MatTools report is not bound to the qualified runtime image")
    observed_model = str(runtime.get("operator_declared_model_id") or "").strip()
    observed_provider = str(runtime.get("operator_declared_provider_id") or "").strip()
    if (
        not observed_model
        or not observed_provider
        or runtime.get("observed_model_ids") != [observed_model]
        or runtime.get("observed_provider_ids") != [observed_provider]
        or runtime.get("actual_model_provider_provenance_validated") is not True
    ):
        _fail("MatTools report lacks model/provider identity")
    if model_identity is not None and model_identity != observed_model:
        _fail("expected model identity differs from the MatTools report")
    if provider_identity is not None and provider_identity != observed_provider:
        _fail("expected provider identity differs from the MatTools report")
    attestation = _mapping(report.get("license_attestation"), "MatTools license attestation")
    observed_basis = str(attestation.get("use_basis") or "")
    observed_purpose = str(attestation.get("use_purpose") or "").strip()
    observed_evidence = attestation.get("separate_license_evidence_sha256")
    if observed_evidence in {None, ""}:
        observed_evidence = None
    else:
        observed_evidence = _plain_sha256(observed_evidence, "MatTools license evidence")
    if observed_basis not in {"noncommercial", "separately_licensed"} or len(observed_purpose) < 12:
        _fail("MatTools license basis or purpose is invalid")
    if (
        attestation.get("accepted") is not True
        or attestation.get("repository_license") != "Apache-2.0"
        or attestation.get("dataset_card_license") != "CC-BY-NC-4.0"
        or not str(attestation.get("attested_at") or "").strip()
    ):
        _fail("MatTools license attestation is incomplete")
    if (observed_basis == "noncommercial" and observed_evidence is not None) or (
        observed_basis == "separately_licensed" and observed_evidence is None
    ):
        _fail("MatTools license evidence is inconsistent with the use basis")
    if license_basis is not None and license_basis != observed_basis:
        _fail("expected license basis differs from the MatTools report")
    if license_purpose is not None and license_purpose != observed_purpose:
        _fail("expected license purpose differs from the MatTools report")
    normalized_expected_evidence = license_evidence_sha256 or None
    if normalized_expected_evidence is not None:
        normalized_expected_evidence = _plain_sha256(
            normalized_expected_evidence, "expected license evidence"
        )
    if license_evidence_sha256 is not None and normalized_expected_evidence != observed_evidence:
        _fail("expected license evidence differs from the MatTools report")

    official = _mapping(
        report.get("official_evaluator_environment"),
        "MatTools evaluator environment",
    )
    approved_lock = _mapping(
        official.get("approved_lock"),
        "MatTools approved evaluator lock",
    )
    build = _mapping(approved_lock.get("build"), "MatTools evaluator build")
    upstream = _mapping(approved_lock.get("upstream"), "MatTools evaluator upstream")
    platform = _mapping(approved_lock.get("platform"), "MatTools evaluator platform")
    harness = _mapping(report.get("harness"), "MatTools harness")
    semantic_repairs_sha256 = _plain_sha256(
        build.get("semantic_repairs_sha256"),
        "MatTools semantic-repair build input",
    )
    if (
        build.get("candidate_fixture_file_count") != EXPECTED_CANDIDATE_FIXTURE_FILE_COUNT
        or build.get("candidate_fixture_manifest_sha256")
        != EXPECTED_CANDIDATE_FIXTURE_MANIFEST_SHA256
        or build.get("candidate_visible_source_policy") != EXPECTED_CANDIDATE_VISIBLE_SOURCE_POLICY
        or build.get("semantic_repairs_path") != "scripts/mattools_semantic_repairs.py"
        or harness.get("semantic_repairs_path") != "scripts/mattools_semantic_repairs.py"
        or harness.get("semantic_repairs_sha256") != semantic_repairs_sha256
    ):
        _fail("MatTools evaluator does not bind the fixture-only semantic-repair build")
    expected_labels = {
        "io.ultra.mattools.adapted-requirements-sha256": build.get("adapted_requirements_sha256"),
        "io.ultra.mattools.base-image": build.get("base_image"),
        "io.ultra.mattools.environment-kind": approved_lock.get("environment_kind"),
        "io.ultra.mattools.official-artifact": "false",
        "io.ultra.mattools.snapshot-manifest-sha256": upstream.get("manifest_sha256"),
        "io.ultra.mattools.safe-parser-sha256": build.get("safe_parser_sha256"),
        "io.ultra.mattools.runner-wrapper-sha256": build.get("runner_wrapper_sha256"),
        "io.ultra.mattools.semantic-repairs-sha256": semantic_repairs_sha256,
        "io.ultra.mattools.strict-shadow-sha256": build.get("strict_shadow_sha256"),
        "io.ultra.mattools.supplemental-requirements-sha256": build.get(
            "supplemental_requirements_sha256"
        ),
        "io.ultra.mattools.target-platform": platform.get("docker"),
        "io.ultra.mattools.tool-source-manifest-sha256": build.get("tool_source_manifest_sha256"),
        "io.ultra.mattools.candidate-fixture-file-count": str(
            build.get("candidate_fixture_file_count")
        ),
        "io.ultra.mattools.candidate-fixture-manifest-sha256": build.get(
            "candidate_fixture_manifest_sha256"
        ),
        "io.ultra.mattools.candidate-visible-source-policy": build.get(
            "candidate_visible_source_policy"
        ),
        "io.ultra.mattools.upstream-requirements-sha256": upstream.get("requirements_sha256"),
        "org.opencontainers.image.revision": upstream.get("revision"),
    }
    expected_embedded_inputs = {
        "candidate_fixture_file_count": EXPECTED_CANDIDATE_FIXTURE_FILE_COUNT,
        "candidate_fixture_manifest_sha256": EXPECTED_CANDIDATE_FIXTURE_MANIFEST_SHA256,
        "candidate_visible_non_fixture_paths": [],
        "candidate_visible_executable_source_paths": [],
        "candidate_visible_dependency_test_paths": {
            "pymatgen": [],
            "pymatgen-analysis-defects": [],
        },
        "upstream_requirements_sha256": upstream.get("requirements_sha256"),
        "adapted_requirements_sha256": build.get("adapted_requirements_sha256"),
        "supplemental_requirements_sha256": build.get("supplemental_requirements_sha256"),
    }
    trials = _sequence(report.get("trials"), "MatTools trials")
    if len(trials) != EXPECTED_TRIAL_COUNT:
        _fail("MatTools report must contain exactly three trials")
    total_runnable = 0
    total_official = 0
    total_strict = 0
    per_trial_counts: list[dict[str, int]] = []
    server_cleanroom = True
    worker_cleanroom = True
    for expected_trial, trial in enumerate(trials, start=1):
        trial_record = _mapping(trial, f"MatTools trial {expected_trial}")
        trial_runnable = _strict_int(
            trial_record.get("runnable"), f"MatTools trial {expected_trial} runnable"
        )
        trial_official = _strict_int(
            trial_record.get("scientific_pass"),
            f"MatTools trial {expected_trial} official scientific",
        )
        trial_strict = _strict_int(
            trial_record.get("strict_scientific_pass"),
            f"MatTools trial {expected_trial} strict scientific",
        )
        attempts = _sequence(
            trial_record.get("attempts"), f"MatTools trial {expected_trial} attempts"
        )
        environment = _mapping(
            trial_record.get("evaluator_environment"),
            f"MatTools trial {expected_trial} evaluator environment",
        )
        if (
            trial_record.get("trial") != expected_trial
            or trial_record.get("status") != "complete"
            or len(attempts) != PARENTS_PER_TRIAL
            or trial_record.get("runnable_denominator") != PARENTS_PER_TRIAL
            or trial_record.get("scientific_denominator") != SCIENTIFIC_SUBTASKS_PER_TRIAL
            or trial_runnable < PER_TRIAL_RUNNABLE_MINIMUM
            or trial_runnable > PARENTS_PER_TRIAL
            or trial_strict < PER_TRIAL_SCIENTIFIC_MINIMUM
            or trial_strict > SCIENTIFIC_SUBTASKS_PER_TRIAL
            or trial_official < 0
            or trial_official > SCIENTIFIC_SUBTASKS_PER_TRIAL
        ):
            _fail("MatTools trial identity, shape, or score floor is inconsistent")
        if (
            environment.get("approved_environment_lock") != approved_lock
            or environment.get("labels_match_approved_lock") is not True
            or environment.get("embedded_inputs_match_approved_lock") is not True
            or environment.get("full_environment_lock_matches") is not True
            or environment.get("comparable") is not True
            or _mapping(environment.get("image_labels"), "MatTools evaluator labels")
            != dict(sorted(expected_labels.items()))
            or _mapping(
                environment.get("embedded_inputs"),
                "MatTools evaluator embedded inputs",
            )
            != expected_embedded_inputs
        ):
            _fail("MatTools evaluator labels or fixture-only inputs are inconsistent")
        for attempt in attempts:
            attempt_record = _mapping(attempt, "MatTools attempt")
            trace = _mapping(attempt_record.get("trace_summary"), "MatTools trace summary")
            binding = _mapping(
                attempt_record.get("cleanroom_binding"),
                "MatTools worker clean-room binding",
            )
            server_cleanroom = server_cleanroom and all(
                (
                    trace.get("server_cleanroom_profile_attested") is True,
                    trace.get("server_evaluation_profiles") == [MATERIALS_CLEANROOM_PROFILE],
                )
            )
            worker_cleanroom = worker_cleanroom and all(
                (
                    _worker_cleanroom_attestation_valid(attempt_record),
                    binding.get("valid") is True,
                    binding.get("user_identity_independently_bound") is True,
                )
            )
        total_runnable += trial_runnable
        total_official += trial_official
        total_strict += trial_strict
        per_trial_counts.append(
            {
                "trial": expected_trial,
                "attempts": len(attempts),
                "runnable": trial_runnable,
                "published_runner_runnable": _strict_int(
                    trial_record.get("published_runner_runnable"),
                    f"MatTools trial {expected_trial} published runnable",
                ),
                "scientific_pass": trial_official,
                "strict_scientific_pass": trial_strict,
            }
        )
    if not server_cleanroom or not worker_cleanroom:
        _fail("MatTools attempts lack independently bound server/worker clean-room proof")
    if (
        total_runnable != counts.get("runnable")
        or total_official != counts.get("scientific_pass")
        or total_strict != counts.get("strict_scientific_pass")
    ):
        _fail("MatTools aggregate counts differ from the recomputed per-trial counts")
    return {
        "license": {
            "use_basis": observed_basis,
            "use_purpose_sha256": _hash_text(observed_purpose),
            "separate_license_evidence_sha256": observed_evidence,
        },
        "runtime_identity": {
            "model_identity_sha256": _hash_text(observed_model),
            "provider_identity_sha256": _hash_text(observed_provider),
            "provenance_validated": True,
        },
        "per_trial_counts": per_trial_counts,
    }


def _public_artifacts(roles: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        name: {
            "sha256": _plain_sha256(role.get("sha256"), f"role {name} digest"),
            "size_bytes": _strict_int(role.get("size_bytes"), f"role {name} size"),
        }
        for name, role in sorted(roles.items())
    }


def _validate_bound_evidence(
    *,
    root: Path,
    root_manifest: Mapping[str, Any],
    source_git_sha: str,
    expected_run_id: str,
    license_basis: str | None = None,
    license_purpose: str | None = None,
    license_evidence_sha256: str | None = None,
    model_identity: str | None = None,
    provider_identity: str | None = None,
) -> dict[str, Any]:
    roles = _role_map(root_manifest)
    readiness_path = _role_path(root, roles, "readiness_report")
    readiness = _strict_json_path(readiness_path, "readiness report")
    summary = _validate_readiness_candidate(readiness)
    expected_provenance = _mapping(
        readiness.get("expected_provenance"), "readiness expected provenance"
    )
    runtime_image_id = _image_digest(
        expected_provenance.get("runtime_image"), "readiness runtime image"
    )
    readiness_manifest = _strict_json_path(
        _role_path(root, roles, "readiness_manifest"), "readiness manifest"
    )
    _validate_readiness_manifest(readiness_manifest, roles["readiness_report"], readiness_path)
    release_manifest = _strict_json_path(
        _role_path(root, roles, "release_manifest"), "release manifest"
    )
    _validate_release_manifest(release_manifest, source_git_sha, expected_run_id)
    _validate_release_tarball(
        _role_path(root, roles, "release_tarball"),
        _role_path(root, roles, "release_manifest"),
        release_manifest,
    )
    _validate_direct_lane_reports(
        root=root,
        root_manifest=root_manifest,
        roles=roles,
        git_sha=source_git_sha,
        runtime_image_id=runtime_image_id,
        readiness=readiness,
    )
    mattools = _strict_json_path(_role_path(root, roles, "mattools_report"), "MatTools report")
    privacy = _validate_mattools_identity(
        mattools,
        source_git_sha=source_git_sha,
        expected_runtime_image=runtime_image_id,
        license_basis=license_basis,
        license_purpose=license_purpose,
        license_evidence_sha256=license_evidence_sha256,
        model_identity=model_identity,
        provider_identity=provider_identity,
    )
    published_mattools_counts = _mapping(mattools.get("counts"), "MatTools counts")
    readiness_counts = _mapping(summary.get("counts"), "readiness count summary")
    readiness_mattools_counts = _mapping(
        readiness_counts.get("mattools"), "readiness MatTools count summary"
    )
    for name in (
        "runnable",
        "runnable_denominator",
        "runnable_minimum",
        "per_trial_runnable_minimum",
        "scientific_pass",
        "strict_scientific_pass",
        "scientific_denominator",
        "scientific_minimum",
        "per_trial_scientific_minimum",
    ):
        if published_mattools_counts.get(name) != readiness_mattools_counts.get(name):
            _fail(f"MatTools report and readiness candidate disagree on {name}")
    if readiness_mattools_counts.get("per_trial") != privacy["per_trial_counts"]:
        _fail("MatTools report and readiness candidate disagree on per-trial counts")
    separate_license_digest = privacy["license"]["separate_license_evidence_sha256"]
    license_role = roles.get("license_evidence")
    if separate_license_digest is None:
        if license_role is not None:
            _fail("noncommercial MatTools use must not retain a separate-license role")
    elif license_role is None or license_role.get("sha256") != separate_license_digest:
        _fail("separately licensed MatTools use requires the exact license evidence role")
    return {
        "readiness": summary,
        "privacy": privacy,
        "expected_provenance": dict(expected_provenance),
        "release_sha": str(release_manifest.get("release_sha") or ""),
        "artifacts": _public_artifacts(roles),
    }


def _assert_secret_free(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            if any(
                normalized == part
                or normalized.startswith(part + "_")
                or normalized.endswith("_" + part)
                for part in FORBIDDEN_KEY_PARTS
            ):
                _fail(f"public envelope contains a forbidden secret-like field at {path}")
            _assert_secret_free(nested, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_secret_free(nested, f"{path}[{index}]")
        return
    if isinstance(value, str) and any(
        pattern.search(value)
        for pattern in (
            URI_USERINFO_RE,
            BEARER_RE,
            PRIVATE_KEY_RE,
            KNOWN_TOKEN_RE,
            QUERY_SECRET_RE,
            AWS_ACCESS_KEY_RE,
        )
    ):
        _fail(f"public envelope contains credential-like text at {path}")


def create_release_envelope(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    root = _root_path(args.evidence_root, "evidence root")
    root_manifest_path = args.evidence_root_manifest.expanduser().absolute()
    envelope_path = args.envelope.expanduser().absolute()
    for output, label in (
        (root_manifest_path, "evidence root manifest"),
        (envelope_path, "release envelope"),
    ):
        if _is_within(output, root):
            _fail(f"{label} must be outside the evidence root")
    workflow_path = args.workflow_file.expanduser().absolute()
    workflow_digest, _ = _hash_regular_file(workflow_path, "trusted workflow file")
    context = _source_context(args, workflow_digest)
    roles = _parse_roles(args.role)
    root_manifest = _build_root_manifest(root=root, roles=roles, context=context)
    bound = _validate_bound_evidence(
        root=root,
        root_manifest=root_manifest,
        source_git_sha=context["git_sha"],
        expected_run_id=context["run_id"],
        license_basis=args.license_basis,
        license_purpose=args.license_purpose,
        license_evidence_sha256=args.license_evidence_sha256,
        model_identity=args.model_identity,
        provider_identity=args.provider_identity,
    )
    expected = _mapping(bound["expected_provenance"], "readiness expected provenance")
    domain_image = _image_digest(args.domain_image_id, "domain image config ID")
    runtime_config = _image_digest(args.runtime_config_id, "runtime image config ID")
    evaluator_image = _image_digest(args.evaluator_image_id, "evaluator image config ID")
    if (
        expected.get("git_sha") != context["git_sha"]
        or expected.get("domain_image") != domain_image
        or expected.get("runtime_image") != runtime_config
        or expected.get("evaluator_image") != evaluator_image
    ):
        _fail("runtime image/source identities differ from the readiness candidate")
    runtime_oci = _image_digest(args.runtime_oci_digest, "runtime OCI manifest digest")
    locator_digest = _plain_sha256(
        args.restricted_store_locator_sha256, "restricted store locator digest"
    )
    root_bytes = _canonical_json_bytes(root_manifest)
    root_digest = hashlib.sha256(root_bytes).hexdigest()
    artifacts = _mapping(bound["artifacts"], "public artifact summaries")
    envelope = {
        "schema_version": RELEASE_ENVELOPE_SCHEMA,
        "claim": {
            "status": "candidate_for_attestation",
            "evidence_passed": True,
            "attestation_required": True,
            "distribution_ready": False,
            "full_materials_production_ready": False,
        },
        "source": {
            "repository": context["repository"],
            "repository_id": context["repository_id"],
            "owner_id": context["owner_id"],
            "git_sha": context["git_sha"],
            "ref": context["ref"],
        },
        "workflow": {
            "path": context["workflow_path"],
            "file_sha256": context["workflow_file_sha256"],
            "signer_digest": context["workflow_signer_digest"],
            "run_id": context["run_id"],
            "run_attempt": context["run_attempt"],
            "event": context["event_name"],
            "environment": context["environment"],
            "qualification_runner": {
                "os": "linux",
                "arch": "arm64",
                "class": "self-hosted-ephemeral",
            },
        },
        "evidence_root": {
            "manifest_sha256": root_digest,
            "manifest_size_bytes": len(root_bytes),
            "aggregate_sha256": root_manifest["aggregate_sha256"],
            "file_count": root_manifest["file_count"],
            "total_size_bytes": root_manifest["total_size_bytes"],
            "restricted_store_locator_sha256": locator_digest,
        },
        "readiness": {
            "report_sha256": artifacts["readiness_report"]["sha256"],
            "report_size_bytes": artifacts["readiness_report"]["size_bytes"],
            "manifest_sha256": artifacts["readiness_manifest"]["sha256"],
            "manifest_size_bytes": artifacts["readiness_manifest"]["size_bytes"],
            "counts": bound["readiness"]["counts"],
            "rates": bound["readiness"]["rates"],
        },
        "release": {
            "tarball_sha256": artifacts["release_tarball"]["sha256"],
            "tarball_size_bytes": artifacts["release_tarball"]["size_bytes"],
            "manifest_sha256": artifacts["release_manifest"]["sha256"],
            "manifest_size_bytes": artifacts["release_manifest"]["size_bytes"],
            "release_sha": bound["release_sha"],
        },
        "images": {
            "domain_config_id": domain_image,
            "runtime_config_id": runtime_config,
            "runtime_oci_manifest_digest": runtime_oci,
            "evaluator_config_id": evaluator_image,
        },
        "license": bound["privacy"]["license"],
        "runtime_identity": bound["privacy"]["runtime_identity"],
        "artifacts": artifacts,
    }
    _assert_secret_free(envelope)
    envelope_bytes = _canonical_json_bytes(envelope)
    _write_once(root_manifest_path, root_bytes, "evidence root manifest")
    _write_once(envelope_path, envelope_bytes, "release envelope")
    verify_evidence_root(root, root_manifest_path)
    return root_manifest, envelope


def _validate_envelope_shape(envelope: Mapping[str, Any]) -> None:
    _exact_keys(
        envelope,
        {
            "schema_version",
            "claim",
            "source",
            "workflow",
            "evidence_root",
            "readiness",
            "release",
            "images",
            "license",
            "runtime_identity",
            "artifacts",
        },
        "release envelope",
    )
    if envelope.get("schema_version") != RELEASE_ENVELOPE_SCHEMA:
        _fail("unexpected release envelope schema")
    claim = _mapping(envelope.get("claim"), "release envelope claim")
    expected_claim = {
        "status": "candidate_for_attestation",
        "evidence_passed": True,
        "attestation_required": True,
        "distribution_ready": False,
        "full_materials_production_ready": False,
    }
    if dict(claim) != expected_claim:
        _fail("release envelope must contain only a non-distributable candidate claim")
    _assert_secret_free(envelope)


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        _fail(f"{label} differs from trusted policy")


def _certificate_policy(
    result: Mapping[str, Any],
    *,
    repository: str,
    repository_id: str,
    owner_id: str,
    workflow_identity: str,
    signer_digest: str,
    source_digest: str,
    source_ref: str,
    run_id: str,
    run_attempt: int,
    event_name: str,
    issuer: str,
    envelope_sha256: str,
) -> None:
    verification = _mapping(result.get("verificationResult"), "gh verification result")
    signature = _mapping(verification.get("signature"), "gh signature")
    certificate = _mapping(signature.get("certificate"), "gh signing certificate")
    owner = repository.split("/", 1)[0]
    expected_values = {
        "subjectAlternativeName": workflow_identity,
        "issuer": issuer,
        "runnerEnvironment": "github-hosted",
        "sourceRepositoryURI": f"https://github.com/{repository}",
        "sourceRepositoryDigest": source_digest,
        "sourceRepositoryRef": source_ref,
        "sourceRepositoryIdentifier": repository_id,
        "sourceRepositoryOwnerURI": f"https://github.com/{owner}",
        "sourceRepositoryOwnerIdentifier": owner_id,
        "buildSignerURI": workflow_identity,
        "buildSignerDigest": signer_digest,
        "buildConfigURI": workflow_identity,
        "buildConfigDigest": signer_digest,
        "buildTrigger": event_name,
        "runInvocationURI": (
            f"https://github.com/{repository}/actions/runs/{run_id}/attempts/{run_attempt}"
        ),
        "sourceRepositoryVisibilityAtSigning": "public",
        "githubWorkflowRepository": repository,
        "githubWorkflowSHA": signer_digest,
        "githubWorkflowRef": source_ref,
        "githubWorkflowTrigger": event_name,
    }
    for name, expected in expected_values.items():
        _assert_equal(certificate.get(name), expected, f"certificate {name}")
    timestamps = _sequence(verification.get("verifiedTimestamps"), "verified timestamps")
    if not timestamps:
        _fail("GitHub attestation has no verified transparency timestamp")
    for index, timestamp in enumerate(timestamps):
        timestamp_record = _mapping(timestamp, f"verified timestamp {index}")
        if not str(timestamp_record.get("timestamp") or "").strip():
            _fail("GitHub attestation has a timestamp without a witnessed time")
    statement = _mapping(verification.get("statement"), "attestation statement")
    if (
        statement.get("_type") != "https://in-toto.io/Statement/v1"
        or statement.get("predicateType") != SLSA_PREDICATE_TYPE
    ):
        _fail("attestation predicate type is not SLSA provenance v1")
    subjects = _sequence(statement.get("subject"), "attestation subjects")
    if len(subjects) != 1:
        _fail("attestation must contain exactly one subject")
    subject = _mapping(subjects[0], "attestation subject")
    subject_digest = _mapping(subject.get("digest"), "attestation subject digest")
    if subject_digest.get("sha256") != envelope_sha256:
        _fail("attestation subject does not bind the exact envelope bytes")


def _verify_public_bindings(
    *,
    envelope: Mapping[str, Any],
    root_manifest: Mapping[str, Any],
    root_manifest_sha256: str,
    root_manifest_size: int,
    bound: Mapping[str, Any],
) -> None:
    envelope_source = _mapping(envelope.get("source"), "envelope source")
    envelope_workflow = _mapping(envelope.get("workflow"), "envelope workflow")
    manifest_source = _mapping(root_manifest.get("source"), "evidence root source")
    expected_manifest_source = {
        "repository": envelope_source.get("repository"),
        "repository_id": envelope_source.get("repository_id"),
        "owner_id": envelope_source.get("owner_id"),
        "git_sha": envelope_source.get("git_sha"),
        "ref": envelope_source.get("ref"),
        "workflow_path": envelope_workflow.get("path"),
        "workflow_file_sha256": envelope_workflow.get("file_sha256"),
        "workflow_signer_digest": envelope_workflow.get("signer_digest"),
        "run_id": envelope_workflow.get("run_id"),
        "run_attempt": envelope_workflow.get("run_attempt"),
        "event_name": envelope_workflow.get("event"),
        "environment": envelope_workflow.get("environment"),
    }
    _assert_equal(dict(manifest_source), expected_manifest_source, "evidence root source context")
    root_claim = _mapping(envelope.get("evidence_root"), "envelope evidence root")
    expected_root = {
        "manifest_sha256": root_manifest_sha256,
        "manifest_size_bytes": root_manifest_size,
        "aggregate_sha256": root_manifest.get("aggregate_sha256"),
        "file_count": root_manifest.get("file_count"),
        "total_size_bytes": root_manifest.get("total_size_bytes"),
    }
    for name, expected in expected_root.items():
        _assert_equal(root_claim.get(name), expected, f"envelope evidence root {name}")
    _plain_sha256(
        root_claim.get("restricted_store_locator_sha256"),
        "restricted store locator digest",
    )
    artifacts = _mapping(envelope.get("artifacts"), "envelope artifact identities")
    _assert_equal(dict(artifacts), bound["artifacts"], "public artifact identities")
    readiness = _mapping(envelope.get("readiness"), "envelope readiness")
    expected_readiness = {
        "report_sha256": artifacts["readiness_report"]["sha256"],
        "report_size_bytes": artifacts["readiness_report"]["size_bytes"],
        "manifest_sha256": artifacts["readiness_manifest"]["sha256"],
        "manifest_size_bytes": artifacts["readiness_manifest"]["size_bytes"],
        "counts": bound["readiness"]["counts"],
        "rates": bound["readiness"]["rates"],
    }
    _assert_equal(dict(readiness), expected_readiness, "readiness public summary")
    release = _mapping(envelope.get("release"), "envelope release")
    expected_release = {
        "tarball_sha256": artifacts["release_tarball"]["sha256"],
        "tarball_size_bytes": artifacts["release_tarball"]["size_bytes"],
        "manifest_sha256": artifacts["release_manifest"]["sha256"],
        "manifest_size_bytes": artifacts["release_manifest"]["size_bytes"],
        "release_sha": bound["release_sha"],
    }
    _assert_equal(dict(release), expected_release, "release public summary")
    _assert_equal(envelope.get("license"), bound["privacy"]["license"], "license summary")
    _assert_equal(
        envelope.get("runtime_identity"),
        bound["privacy"]["runtime_identity"],
        "runtime identity summary",
    )


def verify_attestation(args: argparse.Namespace) -> dict[str, Any]:
    root = _root_path(args.evidence_root, "evidence root")
    root_manifest, root_digest, root_size = verify_evidence_root(root, args.evidence_root_manifest)
    envelope_path = args.envelope.expanduser().absolute()
    if _is_within(envelope_path, root):
        _fail("release envelope must be outside the evidence root")
    envelope_bytes = _read_regular_bytes(envelope_path, "release envelope", maximum=MAX_JSON_BYTES)
    envelope = _mapping(_strict_json_bytes(envelope_bytes, "release envelope"), "release envelope")
    if envelope_bytes != _canonical_json_bytes(envelope):
        _fail("release envelope is not canonical JSON")
    _validate_envelope_shape(envelope)
    source = _mapping(envelope.get("source"), "release envelope source")
    workflow = _mapping(envelope.get("workflow"), "release envelope workflow")
    repository = str(args.repository or "")
    signer_repo = str(args.signer_repo or "")
    signer_workflow = str(args.signer_workflow or "")
    signer_digest = _git_sha(args.signer_digest, "trusted signer digest")
    source_digest = _git_sha(args.source_digest, "trusted source digest")
    source_ref = str(args.source_ref or "")
    run_id = _positive_decimal(args.expected_run_id, "expected run ID")
    run_attempt = _strict_int(args.expected_run_attempt, "expected run attempt", minimum=1)
    environment = str(args.expected_environment or "")
    event_name = str(args.expected_event_name or "")
    workflow_sha256 = _plain_sha256(args.expected_workflow_sha256, "trusted workflow file digest")
    repository_id = _positive_decimal(args.repository_id, "trusted repository ID")
    owner_id = _positive_decimal(args.owner_id, "trusted owner ID")
    if repository != signer_repo or not REPOSITORY_RE.fullmatch(repository):
        _fail("repository and signer repository must be the same exact owner/name")
    if (
        repository != TRUSTED_REPOSITORY
        or repository_id != TRUSTED_REPOSITORY_ID
        or owner_id != TRUSTED_OWNER_ID
        or source_ref != TRUSTED_SOURCE_REF
        or environment != TRUSTED_ENVIRONMENT
        or event_name != "workflow_dispatch"
        or args.cert_oidc_issuer != GITHUB_OIDC_ISSUER
        or signer_digest != source_digest
    ):
        _fail("verification policy differs from the compiled production trust policy")
    if args.gh_command != "gh":
        _fail("production attestation verification must use the reviewed gh executable")
    gh_timeout_seconds = _strict_int(args.gh_timeout_seconds, "gh verification timeout", minimum=1)
    expected_workflow_path = str(workflow.get("path") or "")
    if expected_workflow_path != TRUSTED_WORKFLOW_PATH:
        _fail("envelope does not name the trusted production workflow")
    expected_signer_workflow = f"{repository}/{expected_workflow_path}"
    if signer_workflow != expected_signer_workflow:
        _fail("signer workflow must exactly match repository plus envelope workflow path")
    expected_identity = f"https://github.com/{signer_workflow}@{source_ref}"
    policy = {
        "source": {
            "repository": repository,
            "repository_id": repository_id,
            "owner_id": owner_id,
            "git_sha": source_digest,
            "ref": source_ref,
        },
        "workflow": {
            "path": expected_workflow_path,
            "file_sha256": workflow_sha256,
            "signer_digest": signer_digest,
            "run_id": run_id,
            "run_attempt": run_attempt,
            "event": event_name,
            "environment": environment,
        },
    }
    for name, expected in policy["source"].items():
        _assert_equal(source.get(name), expected, f"envelope source {name}")
    for name, expected in policy["workflow"].items():
        _assert_equal(workflow.get(name), expected, f"envelope workflow {name}")
    runner = _mapping(workflow.get("qualification_runner"), "qualification runner")
    if dict(runner) != {
        "os": "linux",
        "arch": "arm64",
        "class": "self-hosted-ephemeral",
    }:
        _fail("qualification runner declaration is not the protected ARM64 class")
    bound = _validate_bound_evidence(
        root=root,
        root_manifest=root_manifest,
        source_git_sha=source_digest,
        expected_run_id=run_id,
    )
    _verify_public_bindings(
        envelope=envelope,
        root_manifest=root_manifest,
        root_manifest_sha256=root_digest,
        root_manifest_size=root_size,
        bound=bound,
    )
    expected = _mapping(bound["expected_provenance"], "readiness expected provenance")
    images = _mapping(envelope.get("images"), "envelope image identities")
    for name in (
        "domain_config_id",
        "runtime_config_id",
        "runtime_oci_manifest_digest",
        "evaluator_config_id",
    ):
        _image_digest(images.get(name), f"envelope image {name}")
    if (
        expected.get("git_sha") != source_digest
        or expected.get("domain_image") != images.get("domain_config_id")
        or expected.get("runtime_image") != images.get("runtime_config_id")
        or expected.get("evaluator_image") != images.get("evaluator_config_id")
    ):
        _fail("envelope images differ from readiness provenance")
    bundle_path = args.bundle.expanduser().absolute()
    bundle_digest, bundle_size = _hash_regular_file(bundle_path, "attestation bundle")
    envelope_digest = hashlib.sha256(envelope_bytes).hexdigest()
    command = [
        args.gh_command,
        "attestation",
        "verify",
        str(envelope_path),
        "--bundle",
        str(bundle_path),
        "--repo",
        repository,
        "--cert-identity",
        expected_identity,
        "--signer-digest",
        signer_digest,
        "--source-digest",
        source_digest,
        "--source-ref",
        source_ref,
        "--deny-self-hosted-runners",
        "--cert-oidc-issuer",
        args.cert_oidc_issuer,
        "--predicate-type",
        SLSA_PREDICATE_TYPE,
        "--format",
        "json",
    ]
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            check=False,
            timeout=gh_timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _fail(f"gh attestation verification could not complete: {type(exc).__name__}")
    if process.returncode != 0:
        _fail(f"gh attestation verification failed with exit status {process.returncode}")
    verified = _strict_json_bytes(process.stdout, "gh attestation verification output")
    results = _sequence(verified, "gh attestation verification output")
    if len(results) != 1:
        _fail("exactly one GitHub attestation must verify")
    result = _mapping(results[0], "gh attestation result")
    _certificate_policy(
        result,
        repository=repository,
        repository_id=repository_id,
        owner_id=owner_id,
        workflow_identity=expected_identity,
        signer_digest=signer_digest,
        source_digest=source_digest,
        source_ref=source_ref,
        run_id=run_id,
        run_attempt=run_attempt,
        event_name=event_name,
        issuer=args.cert_oidc_issuer,
        envelope_sha256=envelope_digest,
    )
    final = {
        "schema_version": FINAL_VERIFICATION_SCHEMA,
        "status": "verified",
        "envelope": {"sha256": envelope_digest, "size_bytes": len(envelope_bytes)},
        "evidence_root": {
            "closure_verified": True,
            "manifest_sha256": root_digest,
            "aggregate_sha256": root_manifest["aggregate_sha256"],
            "file_count": root_manifest["file_count"],
            "total_size_bytes": root_manifest["total_size_bytes"],
        },
        "release": dict(_mapping(envelope.get("release"), "envelope release")),
        "images": dict(_mapping(envelope.get("images"), "envelope images")),
        "qualification_metrics": {
            "counts": dict(
                _mapping(
                    _mapping(envelope.get("readiness"), "envelope readiness").get("counts"),
                    "envelope readiness counts",
                )
            ),
            "rates": dict(
                _mapping(
                    _mapping(envelope.get("readiness"), "envelope readiness").get("rates"),
                    "envelope readiness rates",
                )
            ),
        },
        "github_attestation": {
            "verified": True,
            "bundle_sha256": bundle_digest,
            "bundle_size_bytes": bundle_size,
            "verification_result_sha256": hashlib.sha256(process.stdout).hexdigest(),
            "repository": repository,
            "repository_id": repository_id,
            "owner_id": owner_id,
            "source_sha": source_digest,
            "source_ref": source_ref,
            "signer_workflow": signer_workflow,
            "signer_digest": signer_digest,
            "run_id": run_id,
            "run_attempt": run_attempt,
            "event": event_name,
            "qualification_environment": environment,
            "runner_environment": "github-hosted",
            "oidc_issuer": args.cert_oidc_issuer,
        },
        "decision": {
            "distribution_ready": True,
            "full_materials_production_ready": True,
            "reasons": [],
        },
    }
    _assert_secret_free(final)
    output_path = args.output.expanduser().absolute()
    if _is_within(output_path, root):
        _fail("final verification report must be outside the sealed evidence root")
    _write_once(output_path, _canonical_json_bytes(final), "final report")
    return final


def _add_create_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--evidence-root-manifest", required=True, type=Path)
    parser.add_argument("--envelope", required=True, type=Path)
    parser.add_argument("--role", required=True, action="append")
    parser.add_argument("--repository", required=True)
    parser.add_argument("--repository-id", required=True)
    parser.add_argument("--owner-id", required=True)
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--workflow-path", required=True)
    parser.add_argument("--workflow-file", required=True, type=Path)
    parser.add_argument("--workflow-signer-digest", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-attempt", required=True, type=int)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--runtime-oci-digest", required=True)
    parser.add_argument("--runtime-config-id", required=True)
    parser.add_argument("--domain-image-id", required=True)
    parser.add_argument("--evaluator-image-id", required=True)
    parser.add_argument(
        "--license-basis", required=True, choices=("noncommercial", "separately_licensed")
    )
    parser.add_argument("--license-purpose")
    parser.add_argument("--license-evidence-sha256")
    parser.add_argument("--model-identity")
    parser.add_argument("--provider-identity")
    parser.add_argument("--restricted-store-locator-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="close evidence and emit a sanitized candidate")
    _add_create_arguments(create)
    verify_root = subparsers.add_parser("verify-root", help="rehash the exact restricted closure")
    verify_root.add_argument("--evidence-root", required=True, type=Path)
    verify_root.add_argument("--evidence-root-manifest", required=True, type=Path)
    verify = subparsers.add_parser(
        "verify-attestation", help="verify closure plus exact GitHub attestation policy"
    )
    verify.add_argument("--evidence-root", required=True, type=Path)
    verify.add_argument("--evidence-root-manifest", required=True, type=Path)
    verify.add_argument("--envelope", required=True, type=Path)
    verify.add_argument("--bundle", required=True, type=Path)
    verify.add_argument("--output", required=True, type=Path)
    verify.add_argument("--repository", required=True)
    verify.add_argument("--repository-id", required=True)
    verify.add_argument("--owner-id", required=True)
    verify.add_argument("--signer-repo", required=True)
    verify.add_argument("--signer-workflow", required=True)
    verify.add_argument("--signer-digest", required=True)
    verify.add_argument("--source-digest", required=True)
    verify.add_argument("--source-ref", required=True)
    verify.add_argument("--expected-run-id", required=True)
    verify.add_argument("--expected-run-attempt", required=True, type=int)
    verify.add_argument("--expected-environment", required=True)
    verify.add_argument("--expected-event-name", required=True)
    verify.add_argument("--expected-workflow-sha256", required=True)
    verify.add_argument("--cert-oidc-issuer", default=GITHUB_OIDC_ISSUER)
    verify.add_argument("--gh-command", default="gh")
    verify.add_argument("--gh-timeout-seconds", type=int, default=120)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "create":
            manifest, envelope = create_release_envelope(args)
            result = {
                "status": "candidate_for_attestation",
                "evidence_root_manifest_sha256": hashlib.sha256(
                    _canonical_json_bytes(manifest)
                ).hexdigest(),
                "release_envelope_sha256": hashlib.sha256(
                    _canonical_json_bytes(envelope)
                ).hexdigest(),
                "full_materials_production_ready": False,
            }
        elif args.command == "verify-root":
            manifest, digest, size = verify_evidence_root(
                args.evidence_root, args.evidence_root_manifest
            )
            result = {
                "status": "verified",
                "manifest_sha256": digest,
                "manifest_size_bytes": size,
                "aggregate_sha256": manifest["aggregate_sha256"],
                "file_count": manifest["file_count"],
                "full_materials_production_ready": False,
            }
        else:
            result = verify_attestation(args)
    except PromotionEnvelopeError as exc:
        print(f"materials promotion boundary blocked: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
