#!/usr/bin/env python3
"""Run the pinned MatTools real-world benchmark through Ultra.

This harness deliberately separates generation from scientific evaluation:

* Ultra sees only each upstream ``question.txt`` through the v2 control plane.
* Candidate source is captured without executing it on the host.
* The unmodified, pinned upstream ``result_analysis.py`` performs scoring in
  a separately reviewed, immutable ``mat-tool-ben`` evaluator image.

The benchmark data is not vendored.  Supply a checkout of the official
MatTools repository at the pinned revision.  A short diagnostic subset may be
submitted, but only three complete 49-question trials can produce a comparable
promotion score.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import copy
import dataclasses
import datetime as dt
import hashlib
import http.cookiejar
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any
from urllib import error, parse, request

SCHEMA_VERSION = "1"
REPLAY_TERMINAL_RECORD_SCHEMA_VERSION = "1"
REPORT_MANIFEST_SCHEMA_VERSION = "2"
REPORT_MANIFEST_KIND = "ultra.mattools.report_bundle.v2"
REPORT_REVALIDATION_SCHEMA_VERSION = "1"
REPORT_REVALIDATION_KIND = "ultra.mattools.report_revalidation.v1"
BENCHMARK_NAME = "MatTools-real-world"
OFFICIAL_REPOSITORY_URL = "https://github.com/Grenzlinie/MatTools"
OFFICIAL_DATASET_URL = "https://huggingface.co/datasets/SiyuLiu/MatTools"
OFFICIAL_DATASET_DOI = "10.57967/hf/5486"
OFFICIAL_REVISION = "1803a6abfe23a9da56c894076c59117873b758ff"
OFFICIAL_MANIFEST_SHA256 = "c70c9c5b1d085643372728e4017c28282e190cd452afa2f5e7fd3366e1a9528e"
OFFICIAL_RUNNER_SHA256 = "4004d0a9d7b103a0a29ada96d7ac7b7977f7cb6fdd73d2cee774b8fd62cc4d70"
OFFICIAL_UNSAFE_UTILS_SHA256 = (
    "ee41f88d71f11997d8160294fb5ad200b294de823968a75d1ac1cc1352d9ec29"
)
OFFICIAL_SANDBOX_SHA256 = "35dea5539537faeffac4f9148de37743190dc4bef67ba9c17e76a7f9b82db426"
OFFICIAL_IMAGE_TAG = "mat-tool-ben"
OFFICIAL_PACKAGE_VERSIONS = {
    "pymatgen": "2024.8.9",
    "pymatgen-analysis-defects": "2024.7.19",
}
PARENT_TASKS_PER_TRIAL = 49
SCIENTIFIC_SUBTASKS_PER_TRIAL = 138
PROMOTION_TRIALS = 3
RUNNABLE_DENOMINATOR = PARENT_TASKS_PER_TRIAL * PROMOTION_TRIALS
SCIENTIFIC_DENOMINATOR = SCIENTIFIC_SUBTASKS_PER_TRIAL * PROMOTION_TRIALS
RUNNABLE_THRESHOLD = 0.80
SCIENTIFIC_THRESHOLD = 0.60
RUNNABLE_MINIMUM = math.ceil(RUNNABLE_DENOMINATOR * RUNNABLE_THRESHOLD)
SCIENTIFIC_MINIMUM = math.ceil(SCIENTIFIC_DENOMINATOR * SCIENTIFIC_THRESHOLD)
PER_TRIAL_RUNNABLE_MINIMUM = math.ceil(PARENT_TASKS_PER_TRIAL * RUNNABLE_THRESHOLD)
PER_TRIAL_SCIENTIFIC_MINIMUM = math.ceil(
    SCIENTIFIC_SUBTASKS_PER_TRIAL * SCIENTIFIC_THRESHOLD
)
SOLUTION_FILENAME = "materials_solution.py"
SIDECAR_FILENAME = "materials_submission.json"
SOLUTION_FUNCTION_NAME = "solve_materials_task"
MAX_SOLUTION_BYTES = 2 * 1024 * 1024
TERMINAL_RUN_STATUSES = {"succeeded", "failed", "canceled"}
HOST_VALIDATOR_REQUIREMENTS_INPUT = Path(__file__).with_name("mattools-validator-requirements.txt")
HOST_VALIDATOR_REQUIREMENTS = Path(__file__).with_name("mattools-validator-requirements.lock.txt")
HOST_VALIDATOR_REQUIRED_VERSIONS = {
    "docker": "7.1.0",
    "numpy": "1.26.4",
    "openpyxl": "3.1.5",
    "pandas": "2.2.2",
    "pymatgen": "2024.8.9",
}
HOST_VALIDATOR_PYTHON_VERSION = "3.11.9"
SHA256_HEX_RE = re.compile(r"[0-9a-f]{64}")
STRICT_SHADOW_SCRIPT = Path(__file__).with_name("mattools_strict_shadow.py")
SAFE_PARSER_SCRIPT = Path(__file__).with_name("mattools_safe_parser.py")
RUNNER_WRAPPER_SCRIPT = Path(__file__).with_name("mattools_runner_wrapper.py")
SEMANTIC_REPAIRS_SCRIPT = Path(__file__).with_name("mattools_semantic_repairs.py")
EVALUATOR_ENVIRONMENT_KIND = "reviewed-reconstruction-variant"
MATERIALS_CLEANROOM_PROFILE = "materials_cleanroom_v1"
WORKER_EVALUATION_ATTESTATION_EVENT = "run.evaluation_profile_attested"
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
EVALUATOR_DEFAULT_LOCK = Path("deploy/docker/mattools-evaluator-linux-arm64-lock.json")
EVALUATOR_BUILDER = Path("scripts/build_mattools_evaluator.py")
EVALUATOR_DOCKERFILE = Path("deploy/docker/mattools-evaluator.Dockerfile")
EVALUATOR_SUPPLEMENTAL_REQUIREMENTS = Path(
    "deploy/docker/mattools-evaluator-supplemental-requirements.txt"
)
EVALUATOR_BASE_IMAGE = (
    "python:3.11.8@sha256:61d662f6d52206ab2290af4258257b5369573b6a4bbd904896699cc909221334"
)
UPSTREAM_REQUIREMENTS_SHA256 = "2c33bd9d99fedaf24bc99aebe60a2e64fa1bcc01d62eb6ae6e013790d3d60122"
ADAPTED_REQUIREMENTS_SHA256 = "d21b1a54095b8e60fc6631fc50c3a345bbbfdb0066cdd0a75513ba040aa3ee91"
TOOL_SOURCE_FILE_COUNT = 2756
TOOL_SOURCE_MANIFEST_SHA256 = "26c1b9224e58bdf7d8aeafbeca74a059d5551a66c85d0fa05b203dff410389d6"
CANDIDATE_FIXTURE_FILE_COUNT = 141
CANDIDATE_FIXTURE_MANIFEST_SHA256 = (
    "296b5b55a5c1640999dd46556c2cd1a1487ae9de3e0f050fa601d3c5236bf308"
)
EVALUATOR_PLATFORM = {
    "docker": "linux/arm64",
    "machine": "aarch64",
    "python_implementation": "CPython",
    "system": "Linux",
}

EVALUATOR_PROBE_SCRIPT = r"""
import hashlib
import json
import os
import platform
from importlib.metadata import distribution as metadata_distribution
from importlib.metadata import distributions, version
from pathlib import Path

def hash_path(path):
    if path.is_symlink():
        return hashlib.sha256(os.readlink(path).encode("utf-8")).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

root = Path("/app/tool_source_code")
file_hashes = {}
for path in sorted(root.rglob("*")):
    if not path.is_symlink() and not path.is_file():
        continue
    relative = "src/tool_source_code/" + path.relative_to(root).as_posix()
    file_hashes[relative] = hash_path(path)
manifest_lines = "".join(
    f"{file_hashes[name]}  {name}\n" for name in sorted(file_hashes)
)
packages = {
    distribution.metadata.get("Name", "").lower(): distribution.version
    for distribution in distributions()
    if distribution.metadata.get("Name")
}
dependency_test_paths = {}
for package_name in ("pymatgen", "pymatgen-analysis-defects"):
    paths = []
    for entry in metadata_distribution(package_name).files or ():
        if any(
            part.casefold() in {"test", "tests"}
            or part.casefold().startswith("test_")
            for part in entry.parts
        ):
            paths.append(str(entry))
    dependency_test_paths[package_name] = sorted(paths)
print(json.dumps({
    "python": platform.python_version(),
    "platform": {
        "docker": "linux/arm64",
        "machine": platform.machine(),
        "python_implementation": platform.python_implementation(),
        "system": platform.system(),
    },
    "packages": {name: version(name) for name in (
        "pymatgen", "pymatgen-analysis-defects"
    )},
    "resolved_packages": dict(sorted(packages.items())),
    "candidate_fixture_file_count": len(file_hashes),
    "candidate_fixture_manifest_sha256": hashlib.sha256(
        manifest_lines.encode("utf-8")
    ).hexdigest(),
    "candidate_visible_non_fixture_paths": sorted(
        name for name in file_hashes
        if not name.startswith(
            "src/tool_source_code/pymatgen-analysis-defects/tests/test_files/"
        )
    ),
    "candidate_visible_executable_source_paths": sorted(
        name for name in file_hashes
        if Path(name).suffix.lower() in {".py", ".pyc", ".pyo", ".ipynb"}
    ),
    "candidate_visible_dependency_test_paths": dependency_test_paths,
    "upstream_requirements_sha256": hash_path(
        Path("/app/upstream-requirements.txt")
    ),
    "adapted_requirements_sha256": hash_path(
        Path("/app/evaluator-requirements.txt")
    ),
    "supplemental_requirements_sha256": hash_path(
        Path("/app/supplemental-requirements.txt")
    ),
    "task_execution_performed": False,
}, sort_keys=True))
"""

# The upstream runner zips raw os.listdir() order against a hard-coded subtask
# vector.  This is the order in the pinned official checkout and its published
# result JSONL files.  Refuse reordered snapshots instead of silently assigning
# verifier counts to the wrong question.
OFFICIAL_TASK_ORDER = (
    "test_vacancy",
    "test_parsing_and_grouping_NamedDefects",
    "test_pchip_eval",
    "test_formation_energy_diagram_shape_fixed",
    "test_substitution",
    "test_vacancy_generators",
    "test_defect_finder",
    "test_get_avg_chg",
    "test_get_SRH_coef",
    "test_supercells",
    "test_freysoldt",
    "test_cluster_nodes",
    "test_defect_entry_grouping",
    "test_get_localized_states",
    "test_charge_interstitial_generator",
    "test_formation_energy_diagram_using_atomic_entries",
    "test_lower_envelope",
    "test_formation_energy_diagram_numerical",
    "test_multi",
    "test_fed_plot",
    "test_get_local_extrema",
    "test_adsorbate",
    "test_get_vibronic_matrix_elements",
    "test_complex",
    "test_get_Rad_coef",
    "test_group_docs",
    "test_ensure_stable_bulk",
    "test_SRHCapture",
    "test_antisite_generator",
    "test_ase_supercells",
    "test_interstitial",
    "test_defect_band_raises",
    "test_dielectric_func",
    "test_interstitial_generator",
    "test_chgcar_insertion",
    "test_generate_all_native_defects",
    "test_competing_phases",
    "test_kumagai",
    "test_HarmonicDefect",
    "test_formation_from_directory",
    "test_plane_spacing",
    "test_formation_energy_diagram_using_bulk_entry",
    "test_topography_analyzer",
    "test_boltzmann",
    "test_voronoi_interstitial_generator",
    "test_closest_sc_mat",
    "test_substitution_generators",
    "test_defect_entry",
    "test_wswq_slope",
)

SNAPSHOT_CONTROL_FILES = (
    "LICENSE",
    "Dockerfile",
    "poetry.lock",
    "pyproject.toml",
    "requirements.txt",
    "src/docker_sandbox.py",
    "src/result_analysis.py",
    "src/utils.py",
)


class GateError(RuntimeError):
    """A fail-closed benchmark or campaign validation error."""


@dataclasses.dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    ordinal: int
    subtask_count: int
    question_path: Path
    properties_path: Path
    verifier_path: Path
    question_sha256: str
    properties_sha256: str
    verifier_sha256: str
    question_text: str = dataclasses.field(repr=False)

    def provenance_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "ordinal": self.ordinal,
            "subtask_count": self.subtask_count,
            "question": {
                "path": self.question_path.name,
                "sha256": self.question_sha256,
            },
            "expected_values": {
                "path": self.properties_path.name,
                "sha256": self.properties_sha256,
                "isolated_from_ultra": True,
            },
            "verifier": {
                "path": self.verifier_path.name,
                "sha256": self.verifier_sha256,
                "isolated_from_ultra": True,
            },
        }


@dataclasses.dataclass(frozen=True)
class BenchmarkSnapshot:
    root: Path
    src_root: Path
    question_root: Path
    revision: str
    revision_source: str
    manifest_sha256: str
    file_hashes: dict[str, str]
    tasks: tuple[BenchmarkTask, ...]
    runner_parent_count: int
    runner_subtask_count: int
    runner_subtask_vector: tuple[int, ...]
    strict_official: bool

    def provenance_record(self) -> dict[str, Any]:
        return {
            "name": BENCHMARK_NAME,
            "repository_url": OFFICIAL_REPOSITORY_URL,
            "dataset_url": OFFICIAL_DATASET_URL,
            "dataset_doi": OFFICIAL_DATASET_DOI,
            "revision": self.revision,
            "revision_source": self.revision_source,
            "sha256": self.manifest_sha256,
            "official_manifest_sha256": OFFICIAL_MANIFEST_SHA256,
            "strict_official": self.strict_official,
            "tracked_file_count": len(self.file_hashes),
            "full_git_tree_hashed": self.strict_official,
            "git_checkout_clean": self.strict_official,
            "parent_count": self.runner_parent_count,
            "scientific_subtask_count": self.runner_subtask_count,
            "licenses": {
                "repository": "Apache-2.0",
                "dataset_card": "CC-BY-NC-4.0",
                "note": (
                    "The repository and dataset card state different licenses; "
                    "the operator must confirm the intended use is permitted."
                ),
            },
            "control_file_hashes": {
                path: self.file_hashes[path]
                for path in SNAPSHOT_CONTROL_FILES
                if path in self.file_hashes
            },
            "tracked_file_hashes": dict(sorted(self.file_hashes.items())),
            "tasks": [task.provenance_record() for task in self.tasks],
        }


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tracked_path(path: Path) -> str:
    if path.is_symlink():
        return sha256_bytes(os.readlink(path).encode("utf-8"))
    return sha256_file(path)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


_LOCAL_MODULE_CACHE: dict[str, Any] = {}


def _load_local_reviewed_module(path: Path, module_name: str) -> Any:
    cached = _LOCAL_MODULE_CACHE.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise GateError(f"could not load reviewed local module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _LOCAL_MODULE_CACHE[module_name] = module
    return module


def _parse_candidate_stdout_safely(stdout: str) -> dict[str, Any] | None:
    module = _load_local_reviewed_module(
        SAFE_PARSER_SCRIPT,
        "ultra_mattools_safe_parser_for_promotion",
    )
    parsed = module.SafeComplexDictParser().parse(stdout)
    return parsed if isinstance(parsed, dict) and parsed else None


def _repair_semantic_score(
    *,
    task_id: str,
    generated: dict[str, Any] | None,
    upstream_strict_scientific_pass: int,
    subtask_count: int,
) -> dict[str, Any]:
    module = _load_local_reviewed_module(
        SEMANTIC_REPAIRS_SCRIPT,
        "ultra_mattools_semantic_repairs_for_promotion",
    )
    return module.repair_task_score(
        task_id=task_id,
        generated=generated,
        upstream_strict_scientific_pass=upstream_strict_scientific_pass,
        subtask_count=subtask_count,
    )


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise GateError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def read_json_file_strict(path: Path, *, label: str) -> Any:
    """Read JSON while rejecting duplicate keys and non-UTF-8 evidence."""

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, GateError) as exc:
        raise GateError(f"invalid {label}: {exc}") from exc


def atomic_write_bytes(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str, *, mode: int = 0o600) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), mode=mode)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, pretty_json_bytes(value))


def write_once_bytes(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    """Create immutable-by-contract evidence without overwriting an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    except FileExistsError:
        if path.read_bytes() != data:
            raise GateError(f"write-once evidence path already contains different bytes: {path}")
        return
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _run_capture(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: float = 30.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def _git_revision(root: Path) -> str | None:
    result = _run_capture(("git", "rev-parse", "HEAD"), cwd=root)
    if result.returncode != 0:
        return None
    revision = result.stdout.strip()
    return revision if re.fullmatch(r"[0-9a-fA-F]{40}", revision) else None


def _clean_tracked_snapshot_files(root: Path) -> list[str]:
    if not (root / ".git").exists():
        raise GateError("strict official snapshot must be a Git checkout, not an unpacked archive")
    status = _run_capture(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=root,
        timeout=60,
    )
    if status.returncode != 0:
        raise GateError("could not verify benchmark checkout cleanliness")
    if status.stdout.strip():
        raise GateError("benchmark checkout is dirty; tracked or untracked changes are forbidden")
    listed = _run_capture(("git", "ls-files", "-z"), cwd=root, timeout=60)
    if listed.returncode != 0:
        raise GateError("could not enumerate benchmark tracked files")
    files = [value for value in listed.stdout.split("\0") if value]
    if not files:
        raise GateError("benchmark Git checkout contains no tracked files")
    return sorted(files)


def _literal_assignments(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assignments: dict[str, Any] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            assignments[target.id] = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            continue
    return assignments


def _snapshot_manifest_files(root: Path, task_ids: Iterable[str]) -> list[str]:
    paths = [path for path in SNAPSHOT_CONTROL_FILES if (root / path).is_file()]
    for task_id in task_ids:
        base = f"src/question_segments/pymatgen_analysis_defects/{task_id}"
        paths.extend(
            (
                f"{base}/new_unit_test.py",
                f"{base}/properties.json",
                f"{base}/question.txt",
            )
        )
    return sorted(paths)


def _manifest_hash(file_hashes: dict[str, str]) -> str:
    lines = "".join(f"{file_hashes[path]}  {path}\n" for path in sorted(file_hashes))
    return sha256_bytes(lines.encode("utf-8"))


def load_benchmark_snapshot(
    root: str | Path,
    *,
    strict_official: bool = True,
    expected_parent_count: int = PARENT_TASKS_PER_TRIAL,
    expected_subtask_count: int = SCIENTIFIC_SUBTASKS_PER_TRIAL,
    expected_task_order: Sequence[str] | None = OFFICIAL_TASK_ORDER,
    expected_manifest_sha256: str | None = OFFICIAL_MANIFEST_SHA256,
) -> BenchmarkSnapshot:
    """Validate and load a MatTools repository snapshot without importing it."""

    repository_root = Path(root).expanduser().resolve()
    src_root = repository_root / "src"
    runner_path = src_root / "result_analysis.py"
    question_root = src_root / "question_segments" / "pymatgen_analysis_defects"
    for required in (repository_root / "LICENSE", runner_path, question_root):
        if not required.exists():
            raise GateError(f"benchmark snapshot is missing required path: {required}")

    license_text = (repository_root / "LICENSE").read_text(encoding="utf-8")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise GateError("benchmark repository LICENSE is not recognizable as Apache-2.0")

    git_revision = _git_revision(repository_root)
    if strict_official:
        if git_revision != OFFICIAL_REVISION:
            raise GateError(
                f"benchmark git revision is {git_revision!r}; expected {OFFICIAL_REVISION}"
            )
        manifest_paths = _clean_tracked_snapshot_files(repository_root)
    else:
        manifest_paths = []

    discovered_order = tuple(
        name for name in os.listdir(question_root) if (question_root / name).is_dir()
    )
    if len(discovered_order) != expected_parent_count:
        raise GateError(
            f"benchmark has {len(discovered_order)} parent tasks; expected {expected_parent_count}"
        )
    if expected_task_order is None:
        raise GateError("an explicit expected task order is required")
    pinned_order = tuple(expected_task_order)
    if len(pinned_order) != expected_parent_count or set(pinned_order) != set(discovered_order):
        raise GateError("benchmark task directories differ from the expected task set")
    if strict_official and pinned_order != discovered_order:
        raise GateError(
            "benchmark directory order differs from the pinned official order; "
            "use a git checkout whose os.listdir order matches the upstream runner"
        )
    task_order = discovered_order if strict_official else pinned_order

    assignments = _literal_assignments(runner_path)
    runner_parent_count = assignments.get("total_tasks_number")
    runner_subtask_count = assignments.get("total_sub_tasks")
    runner_vector = assignments.get("ref_sub_tasks_list")
    if runner_parent_count != expected_parent_count:
        raise GateError(
            f"upstream runner parent denominator is {runner_parent_count!r}; "
            f"expected {expected_parent_count}"
        )
    if runner_subtask_count != expected_subtask_count:
        raise GateError(
            f"upstream runner subtask denominator is {runner_subtask_count!r}; "
            f"expected {expected_subtask_count}"
        )
    if not isinstance(runner_vector, list) or len(runner_vector) != expected_parent_count:
        raise GateError("upstream runner ref_sub_tasks_list has an invalid shape")
    if any(not isinstance(value, int) or value < 1 for value in runner_vector):
        raise GateError("upstream runner ref_sub_tasks_list contains an invalid value")
    if sum(runner_vector) != expected_subtask_count:
        raise GateError(
            f"upstream runner subtask vector sums to {sum(runner_vector)}; "
            f"expected {expected_subtask_count}"
        )

    if not strict_official:
        manifest_paths = _snapshot_manifest_files(repository_root, task_order)
    missing = [
        path
        for path in manifest_paths
        if not (repository_root / path).exists() and not (repository_root / path).is_symlink()
    ]
    if missing:
        raise GateError(f"benchmark snapshot is missing files: {', '.join(missing)}")
    file_hashes = {path: sha256_tracked_path(repository_root / path) for path in manifest_paths}
    manifest_sha256 = _manifest_hash(file_hashes)
    if expected_manifest_sha256 and manifest_sha256 != expected_manifest_sha256:
        raise GateError(
            "benchmark snapshot manifest does not match the pinned official commit: "
            f"got {manifest_sha256}, expected {expected_manifest_sha256}"
        )
    if strict_official:
        if file_hashes.get("src/result_analysis.py") != OFFICIAL_RUNNER_SHA256:
            raise GateError("upstream result_analysis.py is not the pinned unmodified runner")
        if file_hashes.get("src/utils.py") != OFFICIAL_UNSAFE_UTILS_SHA256:
            raise GateError("upstream utils.py differs from the reviewed unsafe-parser snapshot")
        if file_hashes.get("src/docker_sandbox.py") != OFFICIAL_SANDBOX_SHA256:
            raise GateError("upstream docker_sandbox.py is not the pinned unmodified sandbox")

    tasks: list[BenchmarkTask] = []
    for index, (task_id, runner_count) in enumerate(
        zip(task_order, runner_vector, strict=True), start=1
    ):
        task_root = question_root / task_id
        question_path = task_root / "question.txt"
        properties_path = task_root / "properties.json"
        verifier_path = task_root / "new_unit_test.py"
        try:
            properties = json.loads(properties_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise GateError(f"invalid properties JSON for {task_id}: {exc}") from exc
        expected_values = properties.get("properties") if isinstance(properties, dict) else None
        if not isinstance(expected_values, dict):
            raise GateError(f"properties.json for {task_id} has no properties object")
        if len(expected_values) != runner_count:
            raise GateError(
                f"{task_id} has {len(expected_values)} expected properties but the "
                f"runner assigns {runner_count}"
            )
        prefix = f"src/question_segments/pymatgen_analysis_defects/{task_id}"
        question_text = question_path.read_text(encoding="utf-8").strip()
        if not question_text:
            raise GateError(f"question.txt for {task_id} is empty")
        tasks.append(
            BenchmarkTask(
                task_id=task_id,
                ordinal=index,
                subtask_count=runner_count,
                question_path=question_path,
                properties_path=properties_path,
                verifier_path=verifier_path,
                question_sha256=file_hashes[f"{prefix}/question.txt"],
                properties_sha256=file_hashes[f"{prefix}/properties.json"],
                verifier_sha256=file_hashes[f"{prefix}/new_unit_test.py"],
                question_text=question_text,
            )
        )

    revision = git_revision or (OFFICIAL_REVISION if strict_official else manifest_sha256)
    revision_source = "git" if git_revision else "content-manifest"
    return BenchmarkSnapshot(
        root=repository_root,
        src_root=src_root,
        question_root=question_root,
        revision=revision,
        revision_source=revision_source,
        manifest_sha256=manifest_sha256,
        file_hashes=file_hashes,
        tasks=tuple(tasks),
        runner_parent_count=int(runner_parent_count),
        runner_subtask_count=int(runner_subtask_count),
        runner_subtask_vector=tuple(runner_vector),
        strict_official=strict_official,
    )


class CampaignCheckpoint:
    """Thread-safe, atomic campaign state with immutable terminal attempts."""

    def __init__(self, path: Path, data: dict[str, Any]) -> None:
        self.path = path
        self.data = data
        self._lock = threading.RLock()

    @classmethod
    def open_or_create(
        cls,
        path: Path,
        *,
        snapshot: BenchmarkSnapshot,
        config: dict[str, Any],
    ) -> CampaignCheckpoint:
        if path.exists():
            data = read_json_file_strict(path, label="MatTools checkpoint")
            if not isinstance(data, dict):
                raise GateError("MatTools checkpoint must be a JSON object")
            checkpoint = cls(path, data)
            checkpoint._validate_resume(snapshot=snapshot, config=config)
            return checkpoint
        data = {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": config["campaign_id"],
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "benchmark": snapshot.provenance_record(),
            "config": config,
            "attempts": {},
            "evaluations": {},
        }
        checkpoint = cls(path, data)
        checkpoint.save()
        return checkpoint

    def _validate_resume(
        self,
        *,
        snapshot: BenchmarkSnapshot,
        config: dict[str, Any],
    ) -> None:
        if self.data.get("schema_version") != SCHEMA_VERSION:
            raise GateError("checkpoint schema version does not match this harness")
        benchmark = self.data.get("benchmark", {})
        if benchmark.get("sha256") != snapshot.manifest_sha256:
            raise GateError("checkpoint benchmark digest differs from supplied snapshot")
        existing_config = self.data.get("config", {})
        immutable_fields = (
            "campaign_id",
            "trial_count",
            "selected_ordinals",
            "evaluation_mode",
            "validator_replays",
            "control_plane_base_url",
            "evaluation_profile",
            "model_id",
            "provider_id",
            "runtime_image_digest",
            "runtime_pymatgen_version",
            "runtime_defects_version",
            "reasoning_mode",
            "seed",
            "seed_supported",
            "budgets",
            "harness_sha256",
            "host_validator_requirements_sha256",
            "host_validator_input_requirements_sha256",
            "host_validator_environment",
            "safe_parser_sha256",
            "runner_wrapper_sha256",
            "strict_shadow_sha256",
            "semantic_repairs_sha256",
            "evaluator_environment_lock",
            "expected_evaluator_image_id",
            "ultra",
        )
        mismatches = [
            field for field in immutable_fields if existing_config.get(field) != config.get(field)
        ]
        if mismatches:
            raise GateError(
                "resume configuration differs for immutable fields: " + ", ".join(mismatches)
            )
        existing_license = dict(existing_config.get("license_attestation") or {})
        requested_license = dict(config.get("license_attestation") or {})
        existing_license.pop("attested_at", None)
        requested_license.pop("attested_at", None)
        if existing_license != requested_license:
            raise GateError("resume configuration differs for benchmark license attestation")

    def save(self) -> None:
        with self._lock:
            self.data["updated_at"] = utc_now()
            atomic_write_json(self.path, self.data)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self.data)

    def get_attempt(self, key: str) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self.data.get("attempts", {}).get(key, {}))

    def update_attempt(self, key: str, fields: dict[str, Any]) -> None:
        with self._lock:
            attempt = self.data.setdefault("attempts", {}).setdefault(key, {})
            terminal = attempt.get("submission_status") in {
                "captured",
                "terminal_failure",
                "missing_code",
            }
            if terminal:
                changed = [
                    field
                    for field, value in fields.items()
                    if field not in attempt or attempt.get(field) != value
                ]
                if changed:
                    raise GateError(
                        f"refusing to replace terminal attempt {key}; changed fields: "
                        + ", ".join(changed)
                    )
            attempt.update(fields)
            attempt["updated_at"] = utc_now()
            if attempt.get("submission_status") in {
                "captured",
                "terminal_failure",
                "missing_code",
            } and not attempt.get("terminal_record_sha256"):
                terminal_payload = {
                    "schema_version": "1",
                    "attempt_key": key,
                    "attempt": copy.deepcopy(attempt),
                }
                terminal_bytes = canonical_json_bytes(terminal_payload) + b"\n"
                terminal_sha = sha256_bytes(terminal_bytes)
                terminal_path = self.path.parent / "terminal-attempts" / f"{terminal_sha}.json"
                write_once_bytes(terminal_path, terminal_bytes)
                attempt["terminal_record_path"] = str(terminal_path.relative_to(self.path.parent))
                attempt["terminal_record_sha256"] = terminal_sha
            self.save()

    def set_evaluation(self, trial_key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self.data.setdefault("evaluations", {})[trial_key] = value
            self.save()

    def update_config(self, fields: dict[str, Any]) -> None:
        with self._lock:
            self.data.setdefault("config", {}).update(fields)
            self.save()


def _unwrap(payload: dict[str, Any], key: str) -> dict[str, Any]:
    nested = payload.get(key)
    if isinstance(nested, dict):
        return nested
    return payload


def validate_base_url(value: str) -> str:
    parsed = parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise GateError("control-plane base URL must be an absolute http(s) URL")
    if parsed.username or parsed.password:
        raise GateError("credentials are forbidden in the control-plane base URL")
    if parsed.query or parsed.fragment:
        raise GateError("control-plane base URL cannot contain a query or fragment")
    return value.rstrip("/")


class ControlPlaneClient:
    """Minimal v2 client; it never talks to a model endpoint directly."""

    def __init__(self, base_url: str, *, headers: dict[str, str], timeout: float) -> None:
        self.base_url = validate_base_url(base_url)
        self.headers = dict(headers)
        self.timeout = timeout
        jar = http.cookiejar.CookieJar()
        self.opener = request.build_opener(request.HTTPCookieProcessor(jar))

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        body = None if payload is None else canonical_json_bytes(payload)
        req = request.Request(f"{self.base_url}{path}", data=body, method=method)
        req.add_header("Accept", "application/json")
        if body is not None:
            req.add_header("Content-Type", "application/json")
        for key, value in self.headers.items():
            req.add_header(key, value)
        for key, value in (headers or {}).items():
            req.add_header(key, value)
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                raw = response.read()
        except error.HTTPError as exc:
            raise GateError(f"control plane returned HTTP {exc.code} for {method} {path}") from exc
        except error.URLError as exc:
            raise GateError(
                f"control-plane request failed for {method} {path}: {exc.reason}"
            ) from exc
        if not raw:
            return {}
        decoded = json.loads(raw.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise GateError(f"control plane returned non-object JSON for {method} {path}")
        return decoded

    def _request_bytes(self, path: str) -> bytes:
        req = request.Request(f"{self.base_url}{path}", method="GET")
        for key, value in self.headers.items():
            req.add_header(key, value)
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                return response.read(MAX_SOLUTION_BYTES + 1)
        except error.HTTPError as exc:
            raise GateError(f"control plane returned HTTP {exc.code} for GET {path}") from exc
        except error.URLError as exc:
            raise GateError(f"control-plane request failed for GET {path}: {exc.reason}") from exc

    def create_thread(self, title: str) -> dict[str, Any]:
        return _unwrap(
            self._request("POST", "/v2/threads", {"title": title, "messages": []}), "thread"
        )

    def create_run(
        self,
        *,
        thread_id: str,
        prompt: str,
        idempotency_key: str,
        reasoning_mode: str,
        budgets: dict[str, int],
    ) -> dict[str, Any]:
        # Only generic materials routing and the problem statement cross the
        # runtime boundary. Benchmark identity/order remain evaluator-side.
        payload = {
            "goal": prompt,
            "messages": [{"role": "user", "content": prompt}],
            "idempotency_key": idempotency_key,
            "evaluation_profile": MATERIALS_CLEANROOM_PROFILE,
            "file_ids": [],
            "selection_context": {
                "suggested_domain": "materials",
            },
            "reasoning_mode": reasoning_mode,
            "budgets": budgets,
        }
        quoted = parse.quote(thread_id, safe="")
        response = self._request(
            "POST",
            f"/v2/threads/{quoted}/runs",
            payload,
            headers={"Idempotency-Key": idempotency_key},
        )
        return _unwrap(response, "run")

    def get_run(self, run_id: str) -> dict[str, Any]:
        quoted = parse.quote(run_id, safe="")
        return _unwrap(self._request("GET", f"/v2/runs/{quoted}"), "run")

    def list_run_events(self, run_id: str, *, limit: int = 2000) -> list[dict[str, Any]]:
        quoted = parse.quote(run_id, safe="")
        after = 0
        records: list[dict[str, Any]] = []
        while True:
            payload = self._request(
                "GET",
                f"/v2/runs/{quoted}/events?limit={limit}&after_sequence={after}",
            )
            page = payload.get("events", [])
            if not isinstance(page, list):
                break
            items = [item for item in page if isinstance(item, dict)]
            records.extend(items)
            sequences = [int(item.get("sequence") or 0) for item in items]
            next_after = max([after, *sequences])
            if len(items) < limit or next_after <= after:
                break
            after = next_after
        return records

    def list_run_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        quoted = parse.quote(run_id, safe="")
        payload = self._request("GET", f"/v2/runs/{quoted}/artifacts?limit=200")
        artifacts = payload.get("artifacts", [])
        return artifacts if isinstance(artifacts, list) else []

    def download_artifact(self, artifact_id: str) -> bytes:
        quoted = parse.quote(artifact_id, safe="")
        raw = self._request_bytes(f"/v2/artifacts/{quoted}/download")
        if len(raw) > MAX_SOLUTION_BYTES:
            raise GateError(f"artifact {artifact_id} exceeds {MAX_SOLUTION_BYTES} bytes")
        return raw


def auth_headers_from_environment(args: argparse.Namespace) -> tuple[dict[str, str], list[str]]:
    headers: dict[str, str] = {
        "X-Ultra-User-Id": os.environ.get(args.user_id_env, "local-user"),
        "X-Ultra-Org-Id": os.environ.get(args.org_id_env, "local-org"),
        "X-Ultra-Role": os.environ.get(args.role_env, "researcher"),
    }
    used_names = [args.user_id_env, args.org_id_env, args.role_env]
    authorization = os.environ.get(args.authorization_env, "").strip()
    if authorization:
        headers["Authorization"] = authorization
        used_names.append(args.authorization_env)
    cookie = os.environ.get(args.cookie_env, "").strip()
    if cookie:
        headers["Cookie"] = cookie
        used_names.append(args.cookie_env)
    return headers, sorted(set(used_names))


def build_ultra_prompt(question_text: str) -> str:
    """Build a prompt from the problem statement only (never expected values/tests)."""

    return f"""Solve the materials-science programming task below through Ultra's normal tools and runtime.

Return a zero-argument Python function named `{SOLUTION_FUNCTION_NAME}`. The function must compute the requested properties and return one non-empty Python dictionary. Do not merely narrate an answer. You may use the libraries available in the ordinary Ultra runtime.

Save the exact complete Python source as `/{"outputs"}/{SOLUTION_FILENAME}`. Also save `/{"outputs"}/{SIDECAR_FILENAME}` containing exactly a JSON object with `function_name` set to `{SOLUTION_FUNCTION_NAME}`. Preserve all imports and helper definitions in the source artifact.

Problem statement:

{question_text}
"""


def attempt_key(trial: int, task_id: str) -> str:
    return f"trial-{trial:02d}/{task_id}"


def opaque_idempotency_key(campaign_id: str, trial: int, task_id: str) -> str:
    raw = f"{campaign_id}\0{trial}\0{task_id}".encode()
    return "evaluation-" + hashlib.sha256(raw).hexdigest()[:40]


def model_visible_thread_title(idempotency_key: str) -> str:
    opaque = hashlib.sha256(idempotency_key.encode()).hexdigest()[:16]
    return f"Materials analysis {opaque}"


def _artifact_record(artifact: dict[str, Any]) -> dict[str, Any]:
    # Deliberately omit storage_uri and arbitrary metadata, which may contain
    # transient credentials or connector details.
    allowed = (
        "artifact_id",
        "category",
        "created_at",
        "kind",
        "mime_type",
        "path",
        "run_id",
        "sha256",
        "size_bytes",
        "source_path",
        "thread_id",
        "title",
        "tool_name",
        "updated_at",
    )
    return {key: artifact.get(key) for key in allowed if artifact.get(key) is not None}


def _safe_worker_attestation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Project the nonsecret typed worker seal and retain its exact source-key set."""

    projected: dict[str, Any] = {}
    for key in WORKER_EVALUATION_ATTESTATION_FIELDS:
        value = payload.get(key)
        if isinstance(value, str | bool | int):
            projected[key] = value
        elif key == "disabled_capabilities" and isinstance(value, list) and all(
            isinstance(item, str) for item in value
        ):
            projected[key] = list(value)
    projected["_ultra_source_payload_keys"] = sorted(str(key) for key in payload)
    return projected


def _validated_worker_attestation(payload: dict[str, Any], *, run_id: str) -> dict[str, Any]:
    source_keys = payload.get("_ultra_source_payload_keys")
    if isinstance(source_keys, list) and all(isinstance(item, str) for item in source_keys):
        observed_keys = source_keys
    else:
        observed_keys = sorted(
            str(key) for key in payload if key != "_ultra_source_payload_keys"
        )
    record = {key: payload.get(key) for key in WORKER_EVALUATION_ATTESTATION_FIELDS}
    unsigned = dict(record)
    declared_digest = unsigned.pop("attestation_sha256", None)
    digest_fields = ("run_id_sha256", "thread_id_sha256", "user_id_sha256", "goal_sha256")
    digest_fields_valid = all(
        isinstance(record.get(key), str) and SHA256_HEX_RE.fullmatch(str(record[key]))
        for key in digest_fields
    )
    run_digest = sha256_bytes(str(run_id or "").encode("utf-8"))
    expected_namespace = f"{MATERIALS_CLEANROOM_PROFILE}-{run_digest}"
    expected_attestation_digest = sha256_bytes(canonical_json_bytes(unsigned))
    valid = all(
        (
            observed_keys == sorted(WORKER_EVALUATION_ATTESTATION_FIELDS),
            record.get("schema_version") == "1",
            record.get("attestation_kind") == "worker_evaluation_profile",
            record.get("worker_owned") is True,
            record.get("evaluation_profile") == MATERIALS_CLEANROOM_PROFILE,
            record.get("profile_source") == "typed_job_envelope",
            record.get("trusted_envelope_field") == "evaluation_profile",
            record.get("namespace_id") == expected_namespace,
            digest_fields_valid,
            record.get("run_id_sha256") == run_digest,
            record.get("input_policy") == "goal_only",
            isinstance(record.get("provided_message_count"), int),
            not isinstance(record.get("provided_message_count"), bool),
            isinstance(record.get("provided_message_count"), int)
            and int(record["provided_message_count"]) >= 0,
            record.get("effective_message_count") == 1,
            record.get("prior_thread_context_discarded") is True,
            record.get("same_run_retry_state_allowed") is True,
            record.get("run_scoped_workspace") is True,
            record.get("run_scoped_memory") is True,
            record.get("disabled_capabilities")
            == list(WORKER_CLEANROOM_DISABLED_CAPABILITIES),
            isinstance(declared_digest, str),
            declared_digest == expected_attestation_digest,
        )
    )
    return {"valid": valid, "payload": record, "source_payload_keys": observed_keys}


def _event_record(event: dict[str, Any]) -> dict[str, Any]:
    # Keep trace ordering and a payload hash, not raw tool payloads.
    record = {
        key: event.get(key)
        for key in ("event_id", "run_id", "sequence", "source_sequence", "event_kind", "created_at")
        if event.get(key) is not None
    }
    kind = str(event.get("event_kind") or "")
    payload = event.get("payload")
    payload_dict = payload if isinstance(payload, dict) else {}
    safe_payload: dict[str, Any] = {}
    if kind == WORKER_EVALUATION_ATTESTATION_EVENT:
        safe_payload.update(_safe_worker_attestation_payload(payload_dict))
    elif kind in {"run.accepted", "run.requeued"}:
        value = payload_dict.get("evaluation_profile")
        if isinstance(value, str) and value:
            safe_payload["evaluation_profile"] = value
    elif kind.startswith("tool_call."):
        for key in (
            "tool_name",
            "tool_call_id",
            "status",
            "runtime_image_digest",
            "image_digest",
        ):
            value = payload_dict.get(key)
            if isinstance(value, str) and value:
                safe_payload[key] = value
    elif kind == "run.token_usage":
        for key in (
            "model",
            "provider",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_tokens",
        ):
            value = payload_dict.get(key)
            if isinstance(value, str | int):
                safe_payload[key] = value
    if safe_payload:
        record["payload"] = safe_payload
    record["raw_event_sha256"] = sha256_bytes(canonical_json_bytes(event))
    return record


def _payload_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _payload_strings(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _payload_strings(item)


def _trace_summary(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    event_counts: dict[str, int] = {}
    tool_names: list[str] = []
    skill_reads: set[str] = set()
    failed_tool_calls = 0
    visible_failure_signals = 0
    token_totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
    }
    observed_models: set[str] = set()
    observed_providers: set[str] = set()
    server_evaluation_profiles: set[str] = set()
    server_profile_attestation_events = 0
    worker_profile_attestations: list[dict[str, Any]] = []
    execute_started: set[str] = set()
    execute_terminal: set[str] = set()
    execute_completed: set[str] = set()
    execute_image_digests: set[str] = set()
    skill_pattern = re.compile(r"/skills/([A-Za-z0-9_.-]+)/SKILL\.md\b")
    failure_pattern = re.compile(
        r"\b(traceback|exception|command failed|exit code [1-9]|status code [1-9])\b",
        flags=re.IGNORECASE,
    )
    for event in events:
        kind = str(event.get("event_kind") or "")
        if kind:
            event_counts[kind] = event_counts.get(kind, 0) + 1
        payload = event.get("payload")
        payload_dict = payload if isinstance(payload, dict) else {}
        if kind in {"run.accepted", "run.requeued"}:
            profile = str(payload_dict.get("evaluation_profile") or "").strip()
            if profile:
                server_evaluation_profiles.add(profile)
                server_profile_attestation_events += 1
        if kind == WORKER_EVALUATION_ATTESTATION_EVENT:
            worker_profile_attestations.append(
                _validated_worker_attestation(
                    payload_dict,
                    run_id=str(event.get("run_id") or ""),
                )
            )
        if kind == "tool_call.started":
            name = str(
                payload_dict.get("tool_name")
                or payload_dict.get("name")
                or payload_dict.get("tool")
                or ""
            ).strip()
            if name:
                tool_names.append(name)
                if name == "execute":
                    call_id = str(payload_dict.get("tool_call_id") or "").strip()
                    if call_id:
                        execute_started.add(call_id)
            for text in _payload_strings(payload_dict):
                skill_reads.update(match.group(1) for match in skill_pattern.finditer(text))
        if kind == "tool_call.failed":
            failed_tool_calls += 1
        if kind in {"tool_call.completed", "tool_call.failed"}:
            name = str(payload_dict.get("tool_name") or "").strip()
            call_id = str(payload_dict.get("tool_call_id") or "").strip()
            if name == "execute" and call_id:
                execute_terminal.add(call_id)
                if kind == "tool_call.completed":
                    execute_completed.add(call_id)
            if name == "execute":
                for key in ("runtime_image_digest", "image_digest"):
                    digest = _normalize_sha256(str(payload_dict.get(key) or ""))
                    if digest:
                        execute_image_digests.add(digest)
        if kind in {"tool_call.completed", "tool_call.failed"}:
            combined = "\n".join(_payload_strings(payload_dict))
            if failure_pattern.search(combined):
                visible_failure_signals += 1
        if kind == "run.token_usage":
            model = str(payload_dict.get("model") or "").strip()
            provider = str(payload_dict.get("provider") or "").strip()
            if model:
                observed_models.add(model)
            if provider:
                observed_providers.add(provider)
            for key in token_totals:
                value = payload_dict.get(key)
                if isinstance(value, int) and value >= 0:
                    token_totals[key] += value
    return {
        "event_count": len(events),
        "event_counts": dict(sorted(event_counts.items())),
        "tool_call_count": len(tool_names),
        "tool_names": tool_names,
        "failed_tool_call_count": failed_tool_calls,
        "visible_failure_signal_count": visible_failure_signals,
        "skills_read": sorted(skill_reads),
        "token_usage": token_totals,
        "observed_models": sorted(observed_models),
        "observed_providers": sorted(observed_providers),
        "server_evaluation_profiles": sorted(server_evaluation_profiles),
        "server_cleanroom_profile_attested": (
            server_profile_attestation_events >= 1
            and server_evaluation_profiles == {MATERIALS_CLEANROOM_PROFILE}
        ),
        "worker_cleanroom_attestation_count": len(worker_profile_attestations),
        "worker_cleanroom_profile_attested": (
            len(worker_profile_attestations) == 1
            and worker_profile_attestations[0].get("valid") is True
        ),
        "worker_cleanroom_attestations": worker_profile_attestations,
        "production_execute_started_count": len(execute_started),
        "production_execute_terminal_count": len(execute_terminal),
        "production_execute_completed_count": len(execute_completed),
        "production_execute_tool_evidence": bool(execute_started & execute_completed)
        and execute_completed.issubset(execute_started)
        and execute_started.issubset(execute_terminal),
        "observed_execute_image_digests": sorted(execute_image_digests),
    }


def _actual_runtime_provenance(
    trace_summary: dict[str, Any],
    *,
    declared_model_id: str,
    declared_provider_id: str,
) -> dict[str, Any]:
    models = [str(value) for value in trace_summary.get("observed_models", []) if value]
    providers = [str(value) for value in trace_summary.get("observed_providers", []) if value]
    model_observable = bool(models)
    provider_observable = bool(providers)
    return {
        "operator_declared_model_id": declared_model_id,
        "operator_declared_provider_id": declared_provider_id,
        "observed_model_ids": models,
        "observed_provider_ids": providers,
        "model_observable": model_observable,
        "provider_observable": provider_observable,
        "model_matches_declaration": model_observable and declared_model_id in models,
        "provider_matches_declaration": provider_observable and declared_provider_id in providers,
        "validated": (
            model_observable
            and provider_observable
            and declared_model_id in models
            and declared_provider_id in providers
        ),
        "note": (
            "CLI model/provider values are operator declarations, not run-selection controls. "
            "A comparable campaign requires matching observable runtime events."
        ),
    }


def _worker_cleanroom_binding(
    trace_summary: dict[str, Any],
    *,
    run_id: str,
    thread_id: str,
    goal_sha256: str,
    user_id_sha256: str | None = None,
) -> dict[str, Any]:
    attestations = trace_summary.get("worker_cleanroom_attestations")
    record = attestations[0] if isinstance(attestations, list) and len(attestations) == 1 else {}
    payload = record.get("payload") if isinstance(record, dict) else {}
    payload = payload if isinstance(payload, dict) else {}
    expected = {
        "run_id_sha256": sha256_bytes(str(run_id).encode("utf-8")),
        "thread_id_sha256": sha256_bytes(str(thread_id).encode("utf-8")),
        "goal_sha256": goal_sha256,
    }
    if user_id_sha256 is not None:
        expected["user_id_sha256"] = user_id_sha256
    checks = {key: payload.get(key) == value for key, value in expected.items()}
    return {
        "evaluation_profile": MATERIALS_CLEANROOM_PROFILE,
        "worker_event_count": len(attestations) if isinstance(attestations, list) else 0,
        "worker_attestation_valid": isinstance(record, dict) and record.get("valid") is True,
        "server_attestation_valid": trace_summary.get("server_cleanroom_profile_attested") is True,
        "identity_hash_checks": checks,
        "user_identity_independently_bound": user_id_sha256 is not None,
        "valid": (
            isinstance(record, dict)
            and record.get("valid") is True
            and trace_summary.get("server_cleanroom_profile_attested") is True
            and user_id_sha256 is not None
            and all(checks.values())
        ),
    }


def _run_record(run: dict[str, Any]) -> dict[str, Any]:
    record = {
        key: run.get(key)
        for key in (
            "run_id",
            "thread_id",
            "status",
            "created_at",
            "updated_at",
            "started_at",
            "completed_at",
            "model",
            "provider",
            "usage",
        )
        if run.get(key) is not None
    }
    error_text = str(run.get("error") or "")
    user_id = str(run.get("user_id") or "").strip()
    if user_id:
        record["user_id_sha256"] = sha256_bytes(user_id.encode("utf-8"))
    if error_text:
        lowered = error_text.lower()
        if "out of memory" in lowered or "oom" in lowered:
            error_class = "oom"
        elif "timeout" in lowered or "timed out" in lowered:
            error_class = "timeout"
        else:
            error_class = "run_error"
        record["error_class"] = error_class
        record["error_sha256"] = sha256_bytes(error_text.encode("utf-8"))
    return record


def extract_fenced_python(response_text: str) -> str | None:
    matches = re.findall(
        r"```(?:python|py)\s*\n(.*?)```", response_text, flags=re.DOTALL | re.IGNORECASE
    )
    return matches[0] if len(matches) == 1 else None


def select_submission_function(source: str, sidecar_name: str | None = None) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise GateError(f"candidate source is not valid Python: {exc}") from exc
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    by_name = {node.name: node for node in functions}
    candidates = [name for name in (sidecar_name, SOLUTION_FUNCTION_NAME) if name]
    for name in candidates:
        if not isinstance(name, str) or not name.isidentifier():
            raise GateError("submission function_name is not a Python identifier")
        if name in by_name:
            return name
    zero_arg = []
    for node in functions:
        positional = [*node.args.posonlyargs, *node.args.args]
        required = len(positional) - len(node.args.defaults)
        if required == 0 and not node.args.kwonlyargs and not node.name.startswith("_"):
            zero_arg.append(node.name)
    if len(zero_arg) == 1:
        return zero_arg[0]
    raise GateError(
        f"candidate must define {SOLUTION_FUNCTION_NAME} or one unambiguous zero-argument function"
    )


def _parse_sidecar(raw: bytes) -> str | None:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError(f"invalid {SIDECAR_FILENAME}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"function_name"}:
        raise GateError(f"{SIDECAR_FILENAME} must contain only function_name")
    name = payload.get("function_name")
    if not isinstance(name, str) or not name.isidentifier():
        raise GateError(f"{SIDECAR_FILENAME} function_name is invalid")
    return name


def _placeholder_submission() -> tuple[bytes, str]:
    source = (
        b"def ultra_mattools_missing_submission():\n"
        b"    raise RuntimeError('Ultra attempt did not yield an evaluable submission')\n"
    )
    return source, "ultra_mattools_missing_submission"


def capture_submission(
    *,
    client: ControlPlaneClient,
    run: dict[str, Any],
    artifacts: list[dict[str, Any]],
    attempt_dir: Path,
) -> dict[str, Any]:
    response_text = str(run.get("response_text") or "")
    response_bytes = response_text.encode("utf-8")
    atomic_write_bytes(attempt_dir / "response.txt", response_bytes)
    candidates = [
        artifact
        for artifact in artifacts
        if Path(str(artifact.get("path") or "")).name == SOLUTION_FILENAME
    ]
    sidecars = [
        artifact
        for artifact in artifacts
        if Path(str(artifact.get("path") or "")).name == SIDECAR_FILENAME
    ]
    source: bytes | None = None
    source_kind: str | None = None
    artifact_id: str | None = None
    sidecar_name: str | None = None
    issues: list[str] = []

    if len(candidates) == 1:
        artifact_id = str(candidates[0].get("artifact_id") or "")
        if not artifact_id:
            issues.append("solution artifact has no artifact_id")
        else:
            source = client.download_artifact(artifact_id)
            source_kind = "artifact"
            declared_hash = str(candidates[0].get("sha256") or "").removeprefix("sha256:")
            if declared_hash and declared_hash != sha256_bytes(source):
                raise GateError("downloaded solution artifact hash does not match control plane")
    elif len(candidates) > 1:
        issues.append(f"expected one {SOLUTION_FILENAME} artifact, found {len(candidates)}")
    else:
        fenced = extract_fenced_python(response_text)
        if fenced is not None:
            source = fenced.encode("utf-8")
            source_kind = "response_fallback"
            issues.append(f"required {SOLUTION_FILENAME} artifact was missing")
        else:
            issues.append("no solution artifact or unique Python response fence")

    if len(sidecars) == 1:
        sidecar_id = str(sidecars[0].get("artifact_id") or "")
        if sidecar_id:
            sidecar_raw = client.download_artifact(sidecar_id)
            atomic_write_bytes(attempt_dir / SIDECAR_FILENAME, sidecar_raw)
            try:
                sidecar_name = _parse_sidecar(sidecar_raw)
            except GateError as exc:
                issues.append(str(exc))
        else:
            issues.append("submission sidecar has no artifact_id")
    elif len(sidecars) > 1:
        issues.append(f"expected at most one {SIDECAR_FILENAME}, found {len(sidecars)}")

    function_name: str | None = None
    if source is not None:
        try:
            decoded = source.decode("utf-8")
            function_name = select_submission_function(decoded, sidecar_name)
        except (UnicodeDecodeError, GateError) as exc:
            issues.append(str(exc))
            source = None
    if source is None or function_name is None:
        source, function_name = _placeholder_submission()
        source_kind = "failure_placeholder"
        status = "missing_code"
    else:
        status = "captured"

    code_path = attempt_dir / "submission.py"
    atomic_write_bytes(code_path, source)
    return {
        "submission_status": status,
        "code_path": str(code_path),
        "code_sha256": sha256_bytes(source),
        "function_name": function_name,
        "source_kind": source_kind,
        "solution_artifact_id": artifact_id,
        "required_artifact_present": source_kind == "artifact",
        "response_path": str(attempt_dir / "response.txt"),
        "response_sha256": sha256_bytes(response_bytes),
        "capture_issues": issues,
    }


def _persist_trace(
    *,
    attempt_dir: Path,
    run: dict[str, Any],
    events: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    paths = {
        "run": attempt_dir / "run.json",
        "events": attempt_dir / "events.json",
        "artifacts": attempt_dir / "artifacts.json",
    }
    atomic_write_json(paths["run"], _run_record(run))
    atomic_write_json(paths["events"], [_event_record(event) for event in events])
    atomic_write_json(paths["artifacts"], [_artifact_record(item) for item in artifacts])
    return (
        {f"{name}_record_path": str(path) for name, path in paths.items()}
        | {f"{name}_record_sha256": sha256_file(path) for name, path in paths.items()}
        | {"trace_summary": _trace_summary(events)}
    )


def submit_attempt(
    *,
    snapshot: BenchmarkSnapshot,
    task: BenchmarkTask,
    trial: int,
    checkpoint: CampaignCheckpoint,
    output_dir: Path,
    client_factory: Any,
    poll_interval: float,
    poll_timeout: float,
) -> dict[str, Any]:
    key = attempt_key(trial, task.task_id)
    existing = checkpoint.get_attempt(key)
    if existing.get("submission_status") in {"captured", "terminal_failure", "missing_code"}:
        return existing

    config = checkpoint.snapshot()["config"]
    client: ControlPlaneClient = client_factory()
    attempt_dir = output_dir / "attempts" / f"trial-{trial:02d}" / f"{task.ordinal:02d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_ultra_prompt(task.question_text)
    prompt_path = attempt_dir / "prompt.txt"
    if not prompt_path.exists():
        atomic_write_text(prompt_path, prompt)
    idempotency_key = opaque_idempotency_key(config["campaign_id"], trial, task.task_id)
    base_fields = {
        "trial": trial,
        "task_id": task.task_id,
        "ordinal": task.ordinal,
        "subtask_count": task.subtask_count,
        "question_sha256": task.question_sha256,
        "prompt_path": str(prompt_path),
        "prompt_sha256": sha256_file(prompt_path),
        "prompt_inputs": ["question.txt"],
        "expected_values_exposed": False,
        "verifier_exposed": False,
        "selection_context": {
            "suggested_domain": "materials",
        },
        "evaluation_profile": MATERIALS_CLEANROOM_PROFILE,
        "idempotency_key_sha256": sha256_bytes(idempotency_key.encode("utf-8")),
    }
    checkpoint.update_attempt(key, base_fields)
    existing = checkpoint.get_attempt(key)

    thread_id = str(existing.get("thread_id") or "")
    if not thread_id:
        thread = client.create_thread(model_visible_thread_title(idempotency_key))
        thread_id = str(thread.get("thread_id") or "")
        if not thread_id:
            raise GateError(f"control plane did not return a thread_id for {key}")
        checkpoint.update_attempt(key, {"thread_id": thread_id})

    run_id = str(existing.get("run_id") or "")
    if not run_id:
        run = client.create_run(
            thread_id=thread_id,
            prompt=prompt,
            idempotency_key=idempotency_key,
            reasoning_mode=config["reasoning_mode"],
            budgets=config["budgets"],
        )
        run_id = str(run.get("run_id") or "")
        if not run_id:
            raise GateError(f"control plane did not return a run_id for {key}")
        checkpoint.update_attempt(
            key,
            {"run_id": run_id, "last_observed_run_status": run.get("status")},
        )

    started = time.monotonic()
    while True:
        run = client.get_run(run_id)
        status = str(run.get("status") or "")
        checkpoint.update_attempt(key, {"last_observed_run_status": status})
        if status in TERMINAL_RUN_STATUSES:
            break
        if time.monotonic() - started > poll_timeout:
            raise TimeoutError(
                f"Ultra run {run_id} is still {status or 'non-terminal'} after {poll_timeout:.1f}s; "
                "resume the same checkpoint to continue polling"
            )
        time.sleep(poll_interval)

    events = client.list_run_events(run_id)
    artifacts = client.list_run_artifacts(run_id)
    trace_fields = _persist_trace(
        attempt_dir=attempt_dir,
        run=run,
        events=events,
        artifacts=artifacts,
    )
    actual_runtime = _actual_runtime_provenance(
        trace_fields["trace_summary"],
        declared_model_id=str(config["model_id"]),
        declared_provider_id=str(config["provider_id"]),
    )
    run_user_id = str(run.get("user_id") or "").strip()
    cleanroom_binding = _worker_cleanroom_binding(
        trace_fields["trace_summary"],
        run_id=run_id,
        thread_id=thread_id,
        goal_sha256=sha256_file(prompt_path),
        user_id_sha256=(sha256_bytes(run_user_id.encode("utf-8")) if run_user_id else None),
    )
    artifact_records = [_artifact_record(item) for item in artifacts]
    terminal_fields: dict[str, Any] = {
        **trace_fields,
        "run_id": run_id,
        "thread_id": thread_id,
        "run_status": status,
        "artifact_ids": [
            str(item.get("artifact_id")) for item in artifact_records if item.get("artifact_id")
        ],
        "artifact_records": artifact_records,
        "actual_runtime_provenance": actual_runtime,
        "cleanroom_binding": cleanroom_binding,
    }
    if status == "succeeded":
        terminal_fields.update(
            capture_submission(
                client=client,
                run=run,
                artifacts=artifacts,
                attempt_dir=attempt_dir,
            )
        )
    else:
        response_text = str(run.get("response_text") or "")
        response_path = attempt_dir / "response.txt"
        atomic_write_text(response_path, response_text)
        placeholder, function_name = _placeholder_submission()
        code_path = attempt_dir / "submission.py"
        atomic_write_bytes(code_path, placeholder)
        terminal_fields.update(
            {
                "submission_status": "terminal_failure",
                "code_path": str(code_path),
                "code_sha256": sha256_bytes(placeholder),
                "function_name": function_name,
                "source_kind": "failure_placeholder",
                "solution_artifact_id": None,
                "required_artifact_present": False,
                "response_path": str(response_path),
                "response_sha256": sha256_file(response_path),
                "capture_issues": [f"Ultra run ended with status {status}"],
            }
        )
    checkpoint.update_attempt(key, terminal_fields)
    return checkpoint.get_attempt(key)


def submit_campaign(
    *,
    snapshot: BenchmarkSnapshot,
    checkpoint: CampaignCheckpoint,
    output_dir: Path,
    selected_tasks: Sequence[BenchmarkTask],
    trial_count: int,
    headers: dict[str, str],
    concurrency: int,
    http_timeout: float,
    poll_interval: float,
    poll_timeout: float,
) -> None:
    config = checkpoint.snapshot()["config"]

    def client_factory() -> ControlPlaneClient:
        return ControlPlaneClient(
            config["control_plane_base_url"],
            headers=headers,
            timeout=http_timeout,
        )

    jobs = [(trial, task) for trial in range(1, trial_count + 1) for task in selected_tasks]
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map = {
            executor.submit(
                submit_attempt,
                snapshot=snapshot,
                task=task,
                trial=trial,
                checkpoint=checkpoint,
                output_dir=output_dir,
                client_factory=client_factory,
                poll_interval=poll_interval,
                poll_timeout=poll_timeout,
            ): (trial, task)
            for trial, task in jobs
        }
        for future in concurrent.futures.as_completed(future_map):
            trial, task = future_map[future]
            try:
                future.result()
            except Exception as exc:  # retained in checkpoint; never replace a terminal run
                failures.append(f"trial {trial}, ordinal {task.ordinal}: {exc}")
    if failures:
        raise GateError("submission campaign was interrupted:\n" + "\n".join(failures))


def _normalize_sha256(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"sha256:([0-9a-fA-F]{64})", value)
    if match:
        return "sha256:" + match.group(1).lower()
    if re.fullmatch(r"[0-9a-fA-F]{64}", value):
        return "sha256:" + value.lower()
    return None


def _is_git_tracked_unchanged(path: Path, repository_root: Path) -> bool:
    if not path.is_relative_to(repository_root):
        return False
    relative = str(path.relative_to(repository_root))
    tracked = _run_capture(
        ("git", "ls-files", "--error-unmatch", relative),
        cwd=repository_root,
    )
    unchanged = _run_capture(
        ("git", "diff", "--quiet", "HEAD", "--", relative),
        cwd=repository_root,
    )
    return tracked.returncode == 0 and unchanged.returncode == 0


def load_approved_evaluator_environment_lock(path: Path | None) -> dict[str, Any]:
    if path is None:
        raise GateError(
            "comparable scoring requires a reviewed, Git-tracked evaluator environment lock"
        )
    resolved = path.expanduser().resolve()
    repository_root = Path(__file__).resolve().parents[1]
    if not resolved.is_relative_to(repository_root):
        raise GateError("evaluator environment lock must live in the Ultra repository")
    relative = str(resolved.relative_to(repository_root))
    if not _is_git_tracked_unchanged(resolved, repository_root):
        raise GateError(
            "evaluator environment lock must be reviewed, committed, and unchanged from HEAD"
        )
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"invalid evaluator environment lock: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "1":
        raise GateError("evaluator environment lock must use schema_version 1")
    python_version = str(payload.get("python_version") or "").strip()
    packages = payload.get("packages")
    if not python_version or not isinstance(packages, dict) or not packages:
        raise GateError("evaluator environment lock lacks Python or resolved packages")
    normalized_packages = {
        str(name).lower(): str(version)
        for name, version in packages.items()
        if str(name).strip() and str(version).strip()
    }
    if len(normalized_packages) != len(packages):
        raise GateError("evaluator environment lock contains invalid package entries")
    if payload.get("environment_kind") != EVALUATOR_ENVIRONMENT_KIND:
        raise GateError("evaluator environment lock is not the reviewed reconstruction variant")
    if payload.get("official_artifact") is not False:
        raise GateError("evaluator reconstruction lock must not claim an official image artifact")
    variant_reason = str(payload.get("variant_reason") or "").strip()
    if not variant_reason:
        raise GateError("evaluator reconstruction lock must document why it is a variant")
    platform_record = payload.get("platform")
    if platform_record != EVALUATOR_PLATFORM:
        raise GateError("evaluator environment lock targets an unreviewed platform")
    upstream = payload.get("upstream")
    expected_upstream = {
        "revision": OFFICIAL_REVISION,
        "manifest_sha256": OFFICIAL_MANIFEST_SHA256,
        "dockerfile_python": "3.11.8",
        "project_python": ">=3.13,<4.0",
        "requirements_sha256": UPSTREAM_REQUIREMENTS_SHA256,
    }
    if upstream != expected_upstream:
        raise GateError("evaluator environment lock has stale upstream provenance")
    build = payload.get("build")
    if not isinstance(build, dict):
        raise GateError("evaluator environment lock lacks build provenance")
    expected_build_values = {
        "base_image": EVALUATOR_BASE_IMAGE,
        "builder_path": EVALUATOR_BUILDER.as_posix(),
        "dockerfile_path": EVALUATOR_DOCKERFILE.as_posix(),
        "adapted_requirements_sha256": ADAPTED_REQUIREMENTS_SHA256,
        "supplemental_requirements_path": EVALUATOR_SUPPLEMENTAL_REQUIREMENTS.as_posix(),
        "tool_source_file_count": TOOL_SOURCE_FILE_COUNT,
        "tool_source_manifest_sha256": TOOL_SOURCE_MANIFEST_SHA256,
        "candidate_fixture_file_count": CANDIDATE_FIXTURE_FILE_COUNT,
        "candidate_fixture_manifest_sha256": CANDIDATE_FIXTURE_MANIFEST_SHA256,
        "candidate_visible_source_policy": "input-fixtures-only",
        "strict_shadow_path": str(STRICT_SHADOW_SCRIPT.relative_to(repository_root)),
        "safe_parser_path": str(SAFE_PARSER_SCRIPT.relative_to(repository_root)),
        "runner_wrapper_path": str(RUNNER_WRAPPER_SCRIPT.relative_to(repository_root)),
        "semantic_repairs_path": str(SEMANTIC_REPAIRS_SCRIPT.relative_to(repository_root)),
    }
    mismatched_build_values = [
        key for key, value in expected_build_values.items() if build.get(key) != value
    ]
    if mismatched_build_values:
        raise GateError(
            "evaluator environment lock has stale build inputs: "
            + ", ".join(mismatched_build_values)
        )
    tracked_build_inputs = (
        (EVALUATOR_BUILDER, "builder_sha256"),
        (EVALUATOR_DOCKERFILE, "dockerfile_sha256"),
        (EVALUATOR_SUPPLEMENTAL_REQUIREMENTS, "supplemental_requirements_sha256"),
        (Path(str(build["strict_shadow_path"])), "strict_shadow_sha256"),
        (Path(str(build["safe_parser_path"])), "safe_parser_sha256"),
        (Path(str(build["runner_wrapper_path"])), "runner_wrapper_sha256"),
        (Path(str(build["semantic_repairs_path"])), "semantic_repairs_sha256"),
    )
    for relative_input, digest_key in tracked_build_inputs:
        absolute_input = (repository_root / relative_input).resolve()
        if not absolute_input.is_relative_to(repository_root) or not _is_git_tracked_unchanged(
            absolute_input, repository_root
        ):
            raise GateError(
                f"evaluator build input must be committed and unchanged: {relative_input}"
            )
        if build.get(digest_key) != sha256_file(absolute_input):
            raise GateError(f"evaluator build input hash changed: {relative_input}")
    expected_package_hash = sha256_bytes(
        canonical_json_bytes(dict(sorted(normalized_packages.items())))
    )
    if payload.get("package_map_sha256") != expected_package_hash:
        raise GateError("evaluator package-map hash is inconsistent")
    mismatched_pins = {
        name: normalized_packages.get(name)
        for name, expected_version in OFFICIAL_PACKAGE_VERSIONS.items()
        if normalized_packages.get(name) != expected_version
    }
    if mismatched_pins:
        raise GateError("evaluator environment lock has incorrect scientific package pins")
    return {
        "schema_version": "1",
        "path": relative,
        "sha256": sha256_file(resolved),
        "environment_kind": EVALUATOR_ENVIRONMENT_KIND,
        "official_artifact": False,
        "variant_reason": variant_reason,
        "python_version": python_version,
        "platform": platform_record,
        "upstream": upstream,
        "build": build,
        "package_map_sha256": expected_package_hash,
        "packages": dict(sorted(normalized_packages.items())),
        "approved_from_git_head": True,
    }


def inspect_evaluator_image(
    *,
    runtime_image_digest: str,
    expected_image_id: str | None = None,
    environment_lock: dict[str, Any] | None = None,
) -> dict[str, Any]:
    inspect = _run_capture(("docker", "image", "inspect", OFFICIAL_IMAGE_TAG), timeout=60)
    if inspect.returncode != 0:
        raise GateError(f"reviewed evaluator image {OFFICIAL_IMAGE_TAG!r} is unavailable")
    try:
        image_data = json.loads(inspect.stdout)[0]
    except (json.JSONDecodeError, IndexError, TypeError) as exc:
        raise GateError("could not parse docker image inspection output") from exc
    image_id = _normalize_sha256(str(image_data.get("Id") or ""))
    if image_id is None:
        raise GateError("evaluator image has no immutable sha256 image ID")
    expected = _normalize_sha256(expected_image_id)
    if expected_image_id and expected is None:
        raise GateError("expected evaluator image ID is not a sha256 digest")
    if expected and image_id != expected:
        raise GateError(f"evaluator image ID is {image_id}; checkpoint pins {expected}")
    runtime_digest = _normalize_sha256(runtime_image_digest)
    if runtime_digest is None:
        raise GateError("Ultra runtime image must be recorded as an immutable sha256 digest")
    repo_digests = [str(value) for value in image_data.get("RepoDigests") or []]
    evaluator_digests = {
        image_id,
        *filter(None, (_normalize_sha256(value) for value in repo_digests)),
    }
    if runtime_digest in evaluator_digests:
        raise GateError(
            "refusing to score in the production Ultra runtime image; the evaluator "
            "must be an independent image"
        )
    image_labels = image_data.get("Config", {}).get("Labels") or {}
    if not isinstance(image_labels, dict):
        image_labels = {}
    probe = _run_capture(
        (
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            OFFICIAL_IMAGE_TAG,
            "python",
            "-c",
            EVALUATOR_PROBE_SCRIPT,
        ),
        timeout=600,
    )
    if probe.returncode != 0:
        raise GateError(
            "reviewed evaluator provenance probe failed; no comparable score will be produced"
        )
    try:
        environment = json.loads(probe.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise GateError("could not parse evaluator package-version probe") from exc
    packages = environment.get("packages", {})
    resolved_packages = environment.get("resolved_packages", packages)
    if not isinstance(resolved_packages, dict):
        raise GateError("evaluator package probe returned an invalid resolved environment")
    mismatches = {
        name: {"expected": expected_version, "observed": packages.get(name)}
        for name, expected_version in OFFICIAL_PACKAGE_VERSIONS.items()
        if packages.get(name) != expected_version
    }
    if mismatches:
        raise GateError(
            "evaluator dependency versions differ from the official MatTools stack: "
            + json.dumps(mismatches, sort_keys=True)
        )
    resolved_packages = dict(sorted(resolved_packages.items()))
    lock_matches = False
    labels_match = False
    embedded_inputs_match = False
    platform_match = False
    if environment_lock is not None:
        build = environment_lock.get("build", {})
        upstream = environment_lock.get("upstream", {})
        platform_record = environment_lock.get("platform")
        expected_labels = {
            "io.ultra.mattools.adapted-requirements-sha256": build.get(
                "adapted_requirements_sha256"
            ),
            "io.ultra.mattools.base-image": build.get("base_image"),
            "io.ultra.mattools.environment-kind": environment_lock.get("environment_kind"),
            "io.ultra.mattools.official-artifact": "false",
            "io.ultra.mattools.snapshot-manifest-sha256": upstream.get("manifest_sha256"),
            "io.ultra.mattools.safe-parser-sha256": build.get("safe_parser_sha256"),
            "io.ultra.mattools.runner-wrapper-sha256": build.get("runner_wrapper_sha256"),
            "io.ultra.mattools.semantic-repairs-sha256": build.get(
                "semantic_repairs_sha256"
            ),
            "io.ultra.mattools.strict-shadow-sha256": build.get("strict_shadow_sha256"),
            "io.ultra.mattools.supplemental-requirements-sha256": build.get(
                "supplemental_requirements_sha256"
            ),
            "io.ultra.mattools.target-platform": platform_record.get("docker")
            if isinstance(platform_record, dict)
            else None,
            "io.ultra.mattools.tool-source-manifest-sha256": build.get(
                "tool_source_manifest_sha256"
            ),
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
        labels_match = all(image_labels.get(key) == value for key, value in expected_labels.items())
        embedded_inputs_match = (
            environment.get("candidate_fixture_file_count")
            == build.get("candidate_fixture_file_count")
            and environment.get("candidate_fixture_manifest_sha256")
            == build.get("candidate_fixture_manifest_sha256")
            and environment.get("candidate_visible_non_fixture_paths") == []
            and environment.get("candidate_visible_executable_source_paths") == []
            and environment.get("candidate_visible_dependency_test_paths")
            == {"pymatgen": [], "pymatgen-analysis-defects": []}
            and environment.get("upstream_requirements_sha256")
            == upstream.get("requirements_sha256")
            and environment.get("adapted_requirements_sha256")
            == build.get("adapted_requirements_sha256")
            and environment.get("supplemental_requirements_sha256")
            == build.get("supplemental_requirements_sha256")
            and environment.get("task_execution_performed") is False
        )
        platform_match = (
            platform_record == environment.get("platform")
            and image_data.get("Os") == "linux"
            and image_data.get("Architecture") == "arm64"
        )
        lock_matches = (
            environment_lock.get("approved_from_git_head") is True
            and environment_lock.get("environment_kind") == EVALUATOR_ENVIRONMENT_KIND
            and environment_lock.get("official_artifact") is False
            and environment_lock.get("python_version") == environment.get("python")
            and environment_lock.get("packages") == resolved_packages
            and environment_lock.get("package_map_sha256")
            == sha256_bytes(canonical_json_bytes(resolved_packages))
            and labels_match
            and embedded_inputs_match
            and platform_match
        )
        if not lock_matches:
            raise GateError(
                "evaluator image, source, platform, labels, or full package map differs from "
                "the approved reconstruction lock"
            )
    return {
        "image_tag": OFFICIAL_IMAGE_TAG,
        "image_id": image_id,
        "repo_digests": repo_digests,
        "environment_kind": environment_lock.get("environment_kind") if environment_lock else None,
        "official_artifact": False,
        "image_labels": dict(sorted(image_labels.items())),
        "labels_match_approved_lock": labels_match,
        "python_version": environment.get("python"),
        "platform": environment.get("platform"),
        "platform_matches_approved_lock": platform_match,
        "packages": packages,
        "required_packages": OFFICIAL_PACKAGE_VERSIONS,
        "resolved_packages": resolved_packages,
        "resolved_environment_sha256": sha256_bytes(canonical_json_bytes(resolved_packages)),
        "embedded_inputs": {
            "candidate_fixture_file_count": environment.get("candidate_fixture_file_count"),
            "candidate_fixture_manifest_sha256": environment.get(
                "candidate_fixture_manifest_sha256"
            ),
            "candidate_visible_non_fixture_paths": environment.get(
                "candidate_visible_non_fixture_paths"
            ),
            "candidate_visible_executable_source_paths": environment.get(
                "candidate_visible_executable_source_paths"
            ),
            "candidate_visible_dependency_test_paths": environment.get(
                "candidate_visible_dependency_test_paths"
            ),
            "upstream_requirements_sha256": environment.get("upstream_requirements_sha256"),
            "adapted_requirements_sha256": environment.get("adapted_requirements_sha256"),
            "supplemental_requirements_sha256": environment.get("supplemental_requirements_sha256"),
        },
        "embedded_inputs_match_approved_lock": embedded_inputs_match,
        "task_execution_performed": environment.get("task_execution_performed"),
        "approved_environment_lock": environment_lock,
        "full_environment_lock_matches": lock_matches,
        "production_runtime_image_digest": runtime_digest,
        "independent_from_production_runtime": True,
        "comparable": lock_matches,
        "inspected_at": utc_now(),
    }


def validate_sandbox_attestation(
    path: Path | None,
    *,
    image_id: str,
    signature_path: Path | None,
    public_key_path: Path | None,
) -> dict[str, Any]:
    """Verify signed evidence for isolation enforced outside the upstream runner.

    MatTools' pinned DockerSandbox creates default-network containers and sets
    no resource limits. This harness does not alter that source and therefore
    cannot truthfully claim that it enforces isolation. A promotion decision
    requires separate host/daemon-policy evidence, its hash, and a detached
    operator signature. We verify integrity/signature here, not the underlying
    policy's real-world effectiveness.
    """

    base: dict[str, Any] = {
        "valid": False,
        "harness_enforces_isolation": False,
        "upstream_runner_declares_network_isolation": False,
        "upstream_runner_declares_resource_limits": False,
        "note": (
            "The harness verifies signed external evidence but does not itself enforce or "
            "independently prove sandbox isolation."
        ),
    }
    if path is None or signature_path is None or public_key_path is None:
        return {
            **base,
            "issues": [
                "sandbox policy JSON, detached signature, and operator public key are all required"
            ],
        }
    resolved = path.expanduser().resolve()
    resolved_signature = signature_path.expanduser().resolve()
    resolved_public_key = public_key_path.expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"invalid sandbox policy attestation: {exc}") from exc
    if not isinstance(payload, dict):
        raise GateError("sandbox policy attestation must be a JSON object")

    issues: list[str] = []
    required_true = (
        "network_egress_denied",
        "host_access_denied",
        "resource_limits_enforced",
        "external_enforcement",
    )
    issues.extend(key for key in required_true if payload.get(key) is not True)
    if payload.get("attestation_kind") != "external_sandbox_isolation":
        issues.append("attestation_kind")
    mechanism = str(payload.get("enforcement_mechanism") or "").strip()
    if not mechanism:
        issues.append("enforcement_mechanism")
    signed_by = str(payload.get("signed_by") or "").strip()
    signed_at = str(payload.get("signed_at") or "").strip()
    if not signed_by:
        issues.append("signed_by")
    if not signed_at:
        issues.append("signed_at")
    attested_image = _normalize_sha256(str(payload.get("evaluator_image_id") or ""))
    if attested_image != image_id:
        issues.append("evaluator_image_id")

    evidence_value = str(payload.get("isolation_evidence_path") or "").strip()
    evidence_path = (resolved.parent / evidence_value).resolve() if evidence_value else None
    declared_evidence_sha = _normalize_sha256(str(payload.get("isolation_evidence_sha256") or ""))
    observed_evidence_sha: str | None = None
    evidence_semantics_valid = False
    evidence_summary: dict[str, Any] = {}
    if evidence_path is None or not evidence_path.is_file():
        issues.append("isolation_evidence_path")
    else:
        observed_evidence_sha = "sha256:" + sha256_file(evidence_path)
        if declared_evidence_sha != observed_evidence_sha:
            issues.append("isolation_evidence_sha256")
        try:
            evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            evidence_payload = None
        if isinstance(evidence_payload, dict):
            network_probe = evidence_payload.get("network_egress_probe")
            host_probe = evidence_payload.get("host_access_probe")
            limits = evidence_payload.get("resource_limits")
            network_blocked = (
                isinstance(network_probe, dict)
                and network_probe.get("attempted") is True
                and network_probe.get("result") == "blocked"
            )
            host_blocked = (
                isinstance(host_probe, dict)
                and host_probe.get("host_mount_count") == 0
                and host_probe.get("docker_socket_mounted") is False
            )
            limits_present = (
                isinstance(limits, dict)
                and isinstance(limits.get("memory_bytes"), int)
                and limits["memory_bytes"] > 0
                and isinstance(limits.get("pids_limit"), int)
                and limits["pids_limit"] > 0
                and (
                    (isinstance(limits.get("nano_cpus"), int) and limits["nano_cpus"] > 0)
                    or (isinstance(limits.get("cpu_quota"), int) and limits["cpu_quota"] > 0)
                )
            )
            evidence_image = _normalize_sha256(
                str(evidence_payload.get("evaluator_image_id") or "")
            )
            evidence_semantics_valid = (
                evidence_payload.get("schema_version") == "1"
                and evidence_image == image_id
                and bool(str(evidence_payload.get("observed_at") or "").strip())
                and bool(str(evidence_payload.get("observed_container_id") or "").strip())
                and network_blocked
                and host_blocked
                and limits_present
            )
            evidence_summary = {
                "schema_version": evidence_payload.get("schema_version"),
                "evaluator_image_id": evidence_image,
                "observed_at": evidence_payload.get("observed_at"),
                "observed_container_id": evidence_payload.get("observed_container_id"),
                "network_egress_blocked": network_blocked,
                "host_access_blocked": host_blocked,
                "resource_limits_present": limits_present,
            }
        if not evidence_semantics_valid:
            issues.append("isolation_evidence_semantics")

    signature_verified = False
    signature_error = ""
    repository_root = Path(__file__).resolve().parents[1]
    public_key_trusted = _is_git_tracked_unchanged(
        resolved_public_key,
        repository_root,
    )
    if not public_key_trusted:
        issues.append("operator_public_key_not_anchored_in_git_head")
    if not resolved_signature.is_file():
        issues.append("detached_signature")
    elif not resolved_public_key.is_file():
        issues.append("operator_public_key")
    else:
        openssl = shutil.which("openssl")
        if openssl is None:
            issues.append("openssl_unavailable")
        else:
            verification = _run_capture(
                (
                    openssl,
                    "dgst",
                    "-sha256",
                    "-verify",
                    str(resolved_public_key),
                    "-signature",
                    str(resolved_signature),
                    str(resolved),
                ),
                timeout=30,
            )
            signature_verified = verification.returncode == 0
            if not signature_verified:
                signature_error = "detached signature verification failed"
                issues.append("signature_verification")

    return {
        **base,
        "valid": not issues and signature_verified,
        "issues": sorted(set(issues)),
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "detached_signature_path": str(resolved_signature),
        "detached_signature_sha256": (
            sha256_file(resolved_signature) if resolved_signature.is_file() else None
        ),
        "operator_public_key_path": str(resolved_public_key),
        "operator_public_key_sha256": (
            sha256_file(resolved_public_key) if resolved_public_key.is_file() else None
        ),
        "operator_public_key_trusted_from_git_head": public_key_trusted,
        "public_key_trust_anchor": "current Ultra Git HEAD" if public_key_trusted else None,
        "operator_signature_verified": signature_verified,
        "signature_error": signature_error or None,
        "signed_by": signed_by,
        "signed_at": signed_at,
        "attestation_kind": payload.get("attestation_kind"),
        "evaluator_image_id": attested_image,
        "network_egress_denied": payload.get("network_egress_denied") is True,
        "host_access_denied": payload.get("host_access_denied") is True,
        "resource_limits_enforced": payload.get("resource_limits_enforced") is True,
        "external_enforcement": payload.get("external_enforcement") is True,
        "enforcement_mechanism": mechanism,
        "isolation_evidence_path": str(evidence_path) if evidence_path else None,
        "isolation_evidence_sha256": observed_evidence_sha,
        "declared_isolation_evidence_sha256": declared_evidence_sha,
        "external_isolation_evidence_semantics_valid": evidence_semantics_valid,
        "external_isolation_evidence_summary": evidence_summary,
    }


def prepare_official_jsonl(
    *,
    snapshot: BenchmarkSnapshot,
    checkpoint_data: dict[str, Any],
    trial: int,
    destination: Path,
) -> str:
    content = official_jsonl_content(
        snapshot=snapshot,
        checkpoint_data=checkpoint_data,
        trial=trial,
    )
    atomic_write_text(destination, content)
    return sha256_file(destination)


def official_jsonl_content(
    *,
    snapshot: BenchmarkSnapshot,
    checkpoint_data: dict[str, Any],
    trial: int,
) -> str:
    records: list[str] = []
    attempts = checkpoint_data.get("attempts", {})
    for task in snapshot.tasks:
        key = attempt_key(trial, task.task_id)
        attempt = attempts.get(key, {})
        if attempt.get("submission_status") not in {
            "captured",
            "terminal_failure",
            "missing_code",
        }:
            raise GateError(f"cannot evaluate incomplete attempt {key}")
        code_path = Path(str(attempt.get("code_path") or ""))
        if not code_path.is_file():
            raise GateError(f"attempt {key} is missing captured source")
        code_bytes = code_path.read_bytes()
        if sha256_bytes(code_bytes) != attempt.get("code_sha256"):
            raise GateError(f"attempt {key} captured source hash changed")
        try:
            source = code_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GateError(f"attempt {key} source is not UTF-8") from exc
        record = {
            "question_file_path": task.task_id,
            "function": source,
            "function_name": attempt.get("function_name"),
        }
        records.append(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(records) + "\n"


EVALUATION_LINE = re.compile(r"Evaluation result for function \| ([^|]+) \| (.*)$")
SUMMARY_TASK_LINE = re.compile(
    r"Total tasks: (\d+), Correct: (\d+), Partially Correct: (\d+), "
    r"Incorrect: (\d+), Function Errors: (\d+), Result Errors: (\d+), "
    r"Successes: (\d+), Accuracy: ([0-9.]+)%, Function Runnable Rate: ([0-9.]+)%"
)
SUMMARY_SUBTASK_LINE = re.compile(
    r"Total sub-tasks: (\d+), Correct: (\d+), Incorrect: (\d+), Accuracy: ([0-9.]+)%"
)


def _parse_evaluation_value(raw: str) -> str | list[Any]:
    stripped = raw.strip()
    if stripped.startswith("["):
        try:
            value = ast.literal_eval(stripped)
        except (SyntaxError, ValueError) as exc:
            raise GateError(f"cannot parse upstream evaluator list result: {stripped}") from exc
        if not isinstance(value, list):
            raise GateError("upstream evaluator emitted a non-list literal")
        return value
    return stripped


def classify_official_result(value: str | list[Any], expected_subtasks: int) -> dict[str, Any]:
    """Reproduce upstream branch ordering, then fail closed on invalid list totals."""

    strict_classification = "strict_failure"
    strict_scientific_pass = 0
    strict_semantics_valid = True
    strict_verifiable_from_official_log = not isinstance(value, str)
    if isinstance(value, str):
        # Upstream run_test() has already normalized any raw string containing
        # "ok" to exact "ok". Even an exact logged ok is not strict evidence.
        strict_classification = "strict_unverifiable_normalized_string"
        strict_semantics_valid = False
    elif isinstance(value, list):
        if (
            len(value) >= 2
            and all(isinstance(item, int) for item in value[-2:])
            and value[-1] == expected_subtasks
            and 0 <= value[-2] <= value[-1]
        ):
            strict_scientific_pass = value[-1] - value[-2]
            strict_classification = "strict_partial"
        else:
            strict_semantics_valid = False

    strict_fields = {
        "strict_classification": strict_classification,
        "strict_scientific_pass": strict_scientific_pass,
        "strict_semantics_valid": strict_semantics_valid,
        "strict_verifiable_from_official_log": strict_verifiable_from_official_log,
    }
    if "ok" in value:
        return {
            "classification": "success",
            "runnable": True,
            "scientific_pass": expected_subtasks,
            "scientific_fail": 0,
            **strict_fields,
        }
    if isinstance(value, list):
        if len(value) < 2 or not all(isinstance(item, int) for item in value[-2:]):
            raise GateError("upstream partial result lacks integer incorrect/total counters")
        incorrect, total = value[-2:]
        if total != expected_subtasks or incorrect < 0 or incorrect > total:
            raise GateError(
                f"upstream partial result counters are {incorrect}/{total}; "
                f"expected a total of {expected_subtasks}"
            )
        return {
            "classification": "partial",
            "runnable": True,
            "scientific_pass": total - incorrect,
            "scientific_fail": incorrect,
            **strict_fields,
        }
    return {
        "classification": "function_error",
        "runnable": False,
        "scientific_pass": 0,
        "scientific_fail": expected_subtasks,
        **strict_fields,
    }


def parse_official_evaluation_log(
    log_path: Path,
    tasks: Sequence[BenchmarkTask],
) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    raw_results: list[tuple[str, str]] = []
    task_summary_match: re.Match[str] | None = None
    subtask_summary_match: re.Match[str] | None = None
    for line in text.splitlines():
        match = EVALUATION_LINE.search(line)
        if match:
            raw_results.append((match.group(1).strip(), match.group(2).strip()))
        task_match = SUMMARY_TASK_LINE.search(line)
        if task_match:
            task_summary_match = task_match
        subtask_match = SUMMARY_SUBTASK_LINE.search(line)
        if subtask_match:
            subtask_summary_match = subtask_match
    if len(raw_results) != len(tasks):
        raise GateError(
            f"upstream evaluator log contains {len(raw_results)} results; expected {len(tasks)}"
        )
    if task_summary_match is None or subtask_summary_match is None:
        raise GateError("upstream evaluator log is missing its final accuracy summary")

    results: list[dict[str, Any]] = []
    for task, (function_name, raw) in zip(tasks, raw_results, strict=True):
        value = _parse_evaluation_value(raw)
        classified = classify_official_result(value, task.subtask_count)
        results.append(
            {
                "task_id": task.task_id,
                "ordinal": task.ordinal,
                "subtask_count": task.subtask_count,
                "function_name": function_name,
                "upstream_result": value,
                **classified,
            }
        )

    runnable = sum(1 for result in results if result["runnable"])
    scientific_pass = sum(int(result["scientific_pass"]) for result in results)
    strict_scientific_pass = sum(int(result["strict_scientific_pass"]) for result in results)
    success = sum(1 for result in results if result["classification"] == "success")
    partial = sum(1 for result in results if result["classification"] == "partial")
    function_errors = len(results) - runnable
    task_groups = [int(value) for value in task_summary_match.groups()[:7]]
    subtask_groups = [int(value) for value in subtask_summary_match.groups()[:3]]
    upstream = {
        "total_tasks": task_groups[0],
        "correct_tasks": task_groups[1],
        "partial_tasks": task_groups[2],
        "incorrect_tasks": task_groups[3],
        "function_errors": task_groups[4],
        "result_errors": task_groups[5],
        "success_count": task_groups[6],
        "total_subtasks": subtask_groups[0],
        "correct_subtasks": subtask_groups[1],
        "incorrect_subtasks": subtask_groups[2],
    }
    expected_summary = {
        "total_tasks": len(tasks),
        "correct_tasks": success,
        "partial_tasks": partial,
        "incorrect_tasks": function_errors,
        "function_errors": function_errors,
        "result_errors": partial,
        "success_count": success,
        "total_subtasks": sum(task.subtask_count for task in tasks),
        "correct_subtasks": scientific_pass,
        "incorrect_subtasks": sum(task.subtask_count for task in tasks) - scientific_pass,
    }
    if upstream != expected_summary:
        raise GateError(
            "parsed per-task results disagree with the unmodified upstream summary: "
            f"parsed={expected_summary}, upstream={upstream}"
        )
    return {
        "status": "complete",
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
        "runnable": runnable,
        "runnable_denominator": len(tasks),
        "scientific_pass": scientific_pass,
        "scientific_denominator": sum(task.subtask_count for task in tasks),
        "strict_scientific_pass": strict_scientific_pass,
        "strict_semantics_valid": all(result["strict_semantics_valid"] for result in results),
        "full_question_success": success,
        "results": results,
        "upstream_summary": upstream,
    }


def parse_strict_shadow_report(
    path: Path,
    tasks: Sequence[BenchmarkTask],
    *,
    expected_submission_sha256: str,
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"invalid strict shadow report: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "1":
        raise GateError("strict shadow report has an invalid schema")
    if (
        payload.get("purpose")
        != "pre-normalization strict shadow; not the published MatTools score"
    ):
        raise GateError("strict shadow report has an invalid purpose")
    if payload.get("submission_sha256") != expected_submission_sha256:
        raise GateError("strict shadow report submission hash mismatch")
    raw_results = payload.get("results")
    if not isinstance(raw_results, list) or len(raw_results) != len(tasks):
        raise GateError("strict shadow report result count mismatch")
    if payload.get("result_count") != len(raw_results):
        raise GateError("strict shadow report declared result count mismatch")
    results: list[dict[str, Any]] = []
    for task, raw_record in zip(tasks, raw_results, strict=True):
        if not isinstance(raw_record, dict):
            raise GateError("strict shadow result is not an object")
        if (
            raw_record.get("question_file_path") != task.task_id
            or raw_record.get("ordinal") != task.ordinal
        ):
            raise GateError("strict shadow task order/identity mismatch")
        raw_output = raw_record.get("raw_verifier_output")
        upstream_scientific_pass = 0
        upstream_classification = "strict_function_error"
        upstream_scoring_valid = False
        exact_ok = False
        runnable = raw_record.get("runnable") is True
        generated: dict[str, Any] | None = None
        if runnable:
            code_stdout = raw_record.get("code_stdout")
            if not isinstance(code_stdout, str) or not code_stdout:
                raise GateError(f"strict shadow lacks candidate stdout for {task.task_id}")
            if raw_record.get("code_stdout_truncated") is True:
                raise GateError(f"strict shadow candidate stdout is truncated for {task.task_id}")
            if sha256_bytes(code_stdout.encode("utf-8")) != raw_record.get("code_stdout_sha256"):
                raise GateError(f"strict shadow candidate stdout hash mismatch for {task.task_id}")
            try:
                generated = _parse_candidate_stdout_safely(code_stdout)
            except Exception as exc:
                raise GateError(
                    f"strict shadow candidate stdout no longer parses for {task.task_id}"
                ) from exc
            if generated is None:
                raise GateError(f"strict shadow candidate stdout is not a non-empty map for {task.task_id}")
        if runnable and isinstance(raw_output, str):
            if raw_record.get("raw_verifier_output_truncated") is True:
                upstream_classification = "strict_unverifiable_truncated"
            elif sha256_bytes(raw_output.encode("utf-8")) != raw_record.get(
                "raw_verifier_output_sha256"
            ):
                raise GateError(f"strict shadow raw verifier hash mismatch for {task.task_id}")
            else:
                try:
                    parsed = json.loads(raw_output)
                except json.JSONDecodeError:
                    parsed = None
                if parsed == "ok":
                    upstream_classification = "strict_success"
                    upstream_scientific_pass = task.subtask_count
                    upstream_scoring_valid = True
                    exact_ok = True
                elif isinstance(parsed, list) and len(parsed) >= 2:
                    incorrect, total = parsed[-2:]
                    if (
                        type(incorrect) is int
                        and type(total) is int
                        and total == task.subtask_count
                        and 0 <= incorrect <= total
                    ):
                        upstream_classification = "strict_partial"
                        upstream_scientific_pass = total - incorrect
                        upstream_scoring_valid = True
                    else:
                        upstream_classification = "strict_invalid_counters"
                else:
                    upstream_classification = "strict_failure"
        repair = _repair_semantic_score(
            task_id=task.task_id,
            generated=generated,
            upstream_strict_scientific_pass=upstream_scientific_pass,
            subtask_count=task.subtask_count,
        )
        scientific_pass = (
            int(repair["repaired_scientific_pass"]) if upstream_scoring_valid else 0
        )
        classification = upstream_classification
        if upstream_scoring_valid and repair["repair_applied"] is True:
            if scientific_pass == task.subtask_count:
                classification = "strict_repaired_success"
            elif scientific_pass > 0:
                classification = "strict_repaired_partial"
            else:
                classification = "strict_repaired_failure"
        results.append(
            {
                "task_id": task.task_id,
                "ordinal": task.ordinal,
                "subtask_count": task.subtask_count,
                "runnable": runnable,
                "classification": classification,
                "scientific_pass": scientific_pass,
                "scientific_fail": task.subtask_count - scientific_pass,
                "exact_ok": exact_ok,
                "raw_verifier_output_sha256": raw_record.get("raw_verifier_output_sha256"),
                "code_stdout_sha256": raw_record.get("code_stdout_sha256"),
                "upstream_strict_classification": upstream_classification,
                "upstream_strict_scientific_pass": upstream_scientific_pass,
                "upstream_strict_scoring_valid": upstream_scoring_valid,
                "semantic_repair": repair,
            }
        )
    runnable = sum(1 for result in results if result["runnable"])
    scientific_pass = sum(result["scientific_pass"] for result in results)
    upstream_strict_scientific_pass = sum(
        result["upstream_strict_scientific_pass"] for result in results
    )
    return {
        "status": "complete",
        "path": str(path),
        "sha256": sha256_file(path),
        "runnable": runnable,
        "runnable_denominator": len(tasks),
        "scientific_pass": scientific_pass,
        "upstream_strict_scientific_pass": upstream_strict_scientific_pass,
        "scientific_denominator": sum(task.subtask_count for task in tasks),
        "results": results,
        "pre_normalization_captured": True,
        "semantic_repairs_applied": True,
        "semantic_repair_spec_sha256": sha256_file(SEMANTIC_REPAIRS_SCRIPT),
        "published_score_unchanged": True,
    }


def replay_classifications_match(replays: Sequence[dict[str, Any]]) -> bool:
    if len(replays) < 2:
        return False

    def fingerprint(replay: dict[str, Any]) -> dict[str, Any]:
        return {
            "published": [
                (
                    item["task_id"],
                    item["classification"],
                    item["runnable"],
                    item["scientific_pass"],
                )
                for item in replay.get("results", [])
            ],
            "strict_shadow": [
                (
                    item["task_id"],
                    item["runnable"],
                    item["classification"],
                    item["scientific_pass"],
                )
                for item in replay.get("strict_shadow", {}).get("results", [])
            ],
        }

    baseline = fingerprint(replays[0])
    return all(fingerprint(replay) == baseline for replay in replays[1:])


def replay_terminal_record_payload(
    trial_key: str,
    replay: dict[str, Any],
) -> dict[str, Any]:
    """Return the exact write-once payload for one completed evaluator replay."""

    unsealed = {
        key: copy.deepcopy(value)
        for key, value in replay.items()
        if key not in {"terminal_record_path", "terminal_record_sha256"}
    }
    replay_number = unsealed.get("replay")
    if not isinstance(replay_number, int) or isinstance(replay_number, bool) or replay_number < 1:
        raise GateError("completed evaluator replay lacks a positive integer replay number")
    return {
        "schema_version": REPLAY_TERMINAL_RECORD_SCHEMA_VERSION,
        "record_kind": "ultra.mattools.evaluator_replay_terminal.v1",
        "trial_key": trial_key,
        "replay_number": replay_number,
        "replay": unsealed,
    }


def seal_terminal_replay(
    checkpoint: CampaignCheckpoint,
    trial_key: str,
    replay: dict[str, Any],
) -> dict[str, Any]:
    """Content-address and seal all replay evidence before checkpointing it."""

    payload = replay_terminal_record_payload(trial_key, replay)
    terminal_bytes = canonical_json_bytes(payload) + b"\n"
    terminal_sha = sha256_bytes(terminal_bytes)
    terminal_path = checkpoint.path.parent / "terminal-replays" / f"{terminal_sha}.json"
    write_once_bytes(terminal_path, terminal_bytes)
    sealed = copy.deepcopy(replay)
    sealed["terminal_record_path"] = str(terminal_path.relative_to(checkpoint.path.parent))
    sealed["terminal_record_sha256"] = terminal_sha
    return sealed


def verify_terminal_replay_seal(
    campaign_root: Path,
    trial_key: str,
    replay: dict[str, Any],
) -> Path:
    """Fail closed unless a replay exactly matches its content-addressed seal."""

    raw_path = str(replay.get("terminal_record_path") or "").strip()
    expected_sha = str(replay.get("terminal_record_sha256") or "").strip()
    relative = Path(raw_path)
    root = campaign_root.expanduser().resolve()
    if not raw_path or relative.is_absolute():
        raise GateError(f"{trial_key} replay terminal seal must use a relative campaign path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise GateError(f"{trial_key} replay terminal seal is missing or escapes campaign root")
    if not SHA256_HEX_RE.fullmatch(expected_sha) or sha256_file(path) != expected_sha:
        raise GateError(f"{trial_key} replay terminal seal SHA-256 mismatch")
    payload = read_json_file_strict(path, label=f"{trial_key} replay terminal seal")
    if payload != replay_terminal_record_payload(trial_key, replay):
        raise GateError(f"{trial_key} replay differs from its terminal seal")
    return path


def _next_replay_dir(base: Path, replay_number: int) -> Path:
    preferred = base / f"replay-{replay_number:02d}"
    if not preferred.exists():
        return preferred
    return base / f"replay-{replay_number:02d}-recovery-{uuid.uuid4().hex[:8]}"


def _record_failed_evaluator_replay(
    *,
    evaluation: dict[str, Any],
    checkpoint: CampaignCheckpoint,
    trial_key: str,
    replay_number: int,
    reason: str,
    failure_status: str,
    started_at: str,
    replay_dir: Path,
    input_path: Path,
    input_sha256: str,
) -> None:
    record: dict[str, Any] = {
        "replay": replay_number,
        "status": failure_status,
        "reason": reason,
        "started_at": started_at,
        "failed_at": utc_now(),
        "directory": str(replay_dir),
        "input_jsonl_path": str(input_path),
        "input_jsonl_sha256": input_sha256,
    }
    for filename in ("runner.stdout.log", "runner.stderr.log", "timeout.txt"):
        artifact_path = replay_dir / filename
        if artifact_path.is_file():
            key = filename.replace(".", "_")
            record[f"{key}_path"] = str(artifact_path)
            record[f"{key}_sha256"] = sha256_file(artifact_path)
    failure_artifacts = [
        {"path": str(path), "sha256": sha256_file(path)}
        for path in sorted(replay_dir.rglob("*"))
        if path.is_file()
    ]
    record["failure_artifacts"] = failure_artifacts
    record["failure_artifact_manifest_sha256"] = sha256_bytes(
        canonical_json_bytes(failure_artifacts)
    )
    record = seal_terminal_replay(checkpoint, trial_key, record)
    evaluation.setdefault("failed_replays", []).append(record)
    evaluation["status"] = "infrastructure_error"
    checkpoint.set_evaluation(trial_key, evaluation)


def run_official_trial_evaluation(
    *,
    snapshot: BenchmarkSnapshot,
    checkpoint: CampaignCheckpoint,
    output_dir: Path,
    trial: int,
    validator_command: Sequence[str],
    replay_count: int,
    evaluator_timeout: float,
    runtime_image_digest: str,
    expected_image_id: str | None,
    evaluator_environment_lock: dict[str, Any],
    sandbox_attestation_path: Path | None,
    sandbox_attestation_signature_path: Path | None,
    sandbox_attestation_public_key_path: Path | None,
) -> dict[str, Any]:
    trial_key = f"trial-{trial:02d}"
    existing = checkpoint.snapshot().get("evaluations", {}).get(trial_key, {})
    if not isinstance(existing, dict):
        raise GateError(f"{trial_key} checkpoint evaluation must be an object")
    stored_replays = existing.get("replays", [])
    if not isinstance(stored_replays, list):
        raise GateError(f"{trial_key} checkpoint replays must be a list")
    failed_replays = existing.get("failed_replays", [])
    if not isinstance(failed_replays, list):
        raise GateError(f"{trial_key} checkpoint failed_replays must be a list")
    for failed_replay in failed_replays:
        if not isinstance(failed_replay, dict):
            raise GateError(f"{trial_key} checkpoint contains an invalid failed replay")
        verify_terminal_replay_seal(checkpoint.path.parent, trial_key, failed_replay)
    if failed_replays:
        raise GateError(
            f"{trial_key} has terminal failed replay evidence; start a fresh campaign instead "
            "of cherry-picking a later evaluator replay"
        )
    replays = list(stored_replays)
    if len(replays) > replay_count:
        raise GateError(f"{trial_key} checkpoint contains more replays than configured")
    for replay_number, replay in enumerate(replays, start=1):
        if not isinstance(replay, dict) or replay.get("replay") != replay_number:
            raise GateError(f"{trial_key} checkpoint replay order/identity is invalid")
        verify_terminal_replay_seal(checkpoint.path.parent, trial_key, replay)
    pinned_image_id = existing.get("evaluator_environment", {}).get("image_id") or expected_image_id
    environment = inspect_evaluator_image(
        runtime_image_digest=runtime_image_digest,
        expected_image_id=pinned_image_id,
        environment_lock=evaluator_environment_lock,
    )
    attestation = validate_sandbox_attestation(
        sandbox_attestation_path,
        image_id=environment["image_id"],
        signature_path=sandbox_attestation_signature_path,
        public_key_path=sandbox_attestation_public_key_path,
    )
    if attestation.get("valid") is not True:
        raise GateError(
            "refusing to execute MatTools candidate code without valid signed external "
            "sandbox-isolation evidence bound to the evaluator image"
        )
    evaluation: dict[str, Any] = {
        "trial": trial,
        "status": "running",
        "evaluator_environment": environment,
        "sandbox_policy_attestation": attestation,
        "runner": {
            "path": str(snapshot.src_root / "result_analysis.py"),
            "sha256": snapshot.file_hashes["src/result_analysis.py"],
            "unmodified_official": snapshot.file_hashes["src/result_analysis.py"]
            == OFFICIAL_RUNNER_SHA256,
            "host_validator_command": list(validator_command),
            "host_validator_executable_sha256": sha256_file(Path(validator_command[0]).resolve()),
            "host_requirements_path": str(HOST_VALIDATOR_REQUIREMENTS),
            "host_requirements_sha256": sha256_file(HOST_VALIDATOR_REQUIREMENTS),
            "host_input_requirements_path": str(HOST_VALIDATOR_REQUIREMENTS_INPUT),
            "host_input_requirements_sha256": sha256_file(HOST_VALIDATOR_REQUIREMENTS_INPUT),
            "host_validator_environment": checkpoint.snapshot()
            .get("config", {})
            .get("host_validator_environment"),
            "safe_parser_path": str(SAFE_PARSER_SCRIPT),
            "safe_parser_sha256": sha256_file(SAFE_PARSER_SCRIPT),
            "runner_wrapper_path": str(RUNNER_WRAPPER_SCRIPT),
            "runner_wrapper_sha256": sha256_file(RUNNER_WRAPPER_SCRIPT),
            "candidate_host_eval_removed": True,
            "strict_shadow_path": str(STRICT_SHADOW_SCRIPT),
            "strict_shadow_sha256": sha256_file(STRICT_SHADOW_SCRIPT),
            "semantic_repairs_path": str(SEMANTIC_REPAIRS_SCRIPT),
            "semantic_repairs_sha256": sha256_file(SEMANTIC_REPAIRS_SCRIPT),
        },
        "replays": replays,
        "failed_replays": list(existing.get("failed_replays", [])),
    }
    checkpoint.set_evaluation(trial_key, evaluation)
    base = output_dir / "evaluations" / trial_key
    base.mkdir(parents=True, exist_ok=True)

    for replay_number in range(len(replays) + 1, replay_count + 1):
        replay_dir = _next_replay_dir(base, replay_number)
        replay_dir.mkdir(parents=True, exist_ok=False)
        input_path = replay_dir / "function_generation_results.jsonl"
        input_sha = prepare_official_jsonl(
            snapshot=snapshot,
            checkpoint_data=checkpoint.snapshot(),
            trial=trial,
            destination=input_path,
        )
        started_at = utc_now()
        try:
            pre_environment = inspect_evaluator_image(
                runtime_image_digest=runtime_image_digest,
                expected_image_id=environment["image_id"],
                environment_lock=evaluator_environment_lock,
            )
        except GateError as exc:
            _record_failed_evaluator_replay(
                evaluation=evaluation,
                checkpoint=checkpoint,
                trial_key=trial_key,
                replay_number=replay_number,
                reason=str(exc),
                failure_status="evaluator_image_verification_failed",
                started_at=started_at,
                replay_dir=replay_dir,
                input_path=input_path,
                input_sha256=input_sha,
            )
            raise
        command = (
            *validator_command,
            str(RUNNER_WRAPPER_SCRIPT),
            "--snapshot-src",
            str(snapshot.src_root),
            "--expected-runner-sha256",
            OFFICIAL_RUNNER_SHA256,
            "--expected-utils-sha256",
            OFFICIAL_UNSAFE_UTILS_SHA256,
            "--generated-function-path",
            str(replay_dir),
        )
        try:
            process = _run_capture(command, cwd=snapshot.src_root, timeout=evaluator_timeout)
        except subprocess.TimeoutExpired as exc:
            atomic_write_text(replay_dir / "timeout.txt", str(exc))
            _record_failed_evaluator_replay(
                evaluation=evaluation,
                checkpoint=checkpoint,
                trial_key=trial_key,
                replay_number=replay_number,
                reason="unmodified upstream evaluator exceeded the outer timeout",
                failure_status="timeout",
                started_at=started_at,
                replay_dir=replay_dir,
                input_path=input_path,
                input_sha256=input_sha,
            )
            raise GateError(
                f"pinned upstream evaluator timed out for trial {trial}, replay {replay_number}; "
                "the upstream runner produced no comparable classification"
            ) from exc
        atomic_write_text(replay_dir / "runner.stdout.log", process.stdout)
        atomic_write_text(replay_dir / "runner.stderr.log", process.stderr)
        if process.returncode != 0:
            _record_failed_evaluator_replay(
                evaluation=evaluation,
                checkpoint=checkpoint,
                trial_key=trial_key,
                replay_number=replay_number,
                reason=f"unmodified upstream evaluator exited {process.returncode}",
                failure_status="nonzero_exit",
                started_at=started_at,
                replay_dir=replay_dir,
                input_path=input_path,
                input_sha256=input_sha,
            )
            raise GateError(
                f"unmodified upstream evaluator exited {process.returncode} for trial {trial}, "
                f"replay {replay_number}; no score will be inferred"
            )
        strict_shadow_path = replay_dir / "strict-shadow-results.json"
        strict_shadow_stdout = replay_dir / "strict-shadow.stdout.log"
        strict_shadow_stderr = replay_dir / "strict-shadow.stderr.log"
        strict_command = (
            *validator_command,
            str(STRICT_SHADOW_SCRIPT),
            "--snapshot-src",
            str(snapshot.src_root),
            "--submissions",
            str(input_path),
            "--output",
            str(strict_shadow_path),
        )
        try:
            strict_process = _run_capture(
                strict_command,
                cwd=snapshot.src_root,
                timeout=evaluator_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            atomic_write_text(replay_dir / "strict-shadow-timeout.txt", str(exc))
            _record_failed_evaluator_replay(
                evaluation=evaluation,
                checkpoint=checkpoint,
                trial_key=trial_key,
                replay_number=replay_number,
                reason="strict pre-normalization shadow exceeded the outer timeout",
                failure_status="strict_shadow_timeout",
                started_at=started_at,
                replay_dir=replay_dir,
                input_path=input_path,
                input_sha256=input_sha,
            )
            raise GateError(
                f"strict shadow timed out for trial {trial}, replay {replay_number}"
            ) from exc
        atomic_write_text(strict_shadow_stdout, strict_process.stdout)
        atomic_write_text(strict_shadow_stderr, strict_process.stderr)
        if strict_process.returncode != 0 or not strict_shadow_path.is_file():
            _record_failed_evaluator_replay(
                evaluation=evaluation,
                checkpoint=checkpoint,
                trial_key=trial_key,
                replay_number=replay_number,
                reason=f"strict shadow exited {strict_process.returncode}",
                failure_status="strict_shadow_nonzero_exit",
                started_at=started_at,
                replay_dir=replay_dir,
                input_path=input_path,
                input_sha256=input_sha,
            )
            raise GateError(f"strict shadow failed for trial {trial}, replay {replay_number}")
        strict_shadow = parse_strict_shadow_report(
            strict_shadow_path,
            snapshot.tasks,
            expected_submission_sha256=input_sha,
        )
        try:
            post_environment = inspect_evaluator_image(
                runtime_image_digest=runtime_image_digest,
                expected_image_id=environment["image_id"],
                environment_lock=evaluator_environment_lock,
            )
        except GateError as exc:
            _record_failed_evaluator_replay(
                evaluation=evaluation,
                checkpoint=checkpoint,
                trial_key=trial_key,
                replay_number=replay_number,
                reason=str(exc),
                failure_status="evaluator_image_changed",
                started_at=started_at,
                replay_dir=replay_dir,
                input_path=input_path,
                input_sha256=input_sha,
            )
            raise
        logs = sorted(replay_dir.glob("evaluation_logs_*.log"))
        if len(logs) != 1:
            _record_failed_evaluator_replay(
                evaluation=evaluation,
                checkpoint=checkpoint,
                trial_key=trial_key,
                replay_number=replay_number,
                reason=f"upstream evaluator produced {len(logs)} log files",
                failure_status="missing_or_ambiguous_log",
                started_at=started_at,
                replay_dir=replay_dir,
                input_path=input_path,
                input_sha256=input_sha,
            )
            raise GateError(
                f"pinned upstream evaluator produced {len(logs)} log files for trial {trial}, "
                f"replay {replay_number}"
            )
        try:
            replay = parse_official_evaluation_log(logs[0], snapshot.tasks)
        except GateError as exc:
            _record_failed_evaluator_replay(
                evaluation=evaluation,
                checkpoint=checkpoint,
                trial_key=trial_key,
                replay_number=replay_number,
                reason=str(exc),
                failure_status="unparseable_upstream_result",
                started_at=started_at,
                replay_dir=replay_dir,
                input_path=input_path,
                input_sha256=input_sha,
            )
            raise
        replay.update(
            {
                "replay": replay_number,
                "started_at": started_at,
                "completed_at": utc_now(),
                "input_jsonl_path": str(input_path),
                "input_jsonl_sha256": input_sha,
                "runner_stdout_path": str(replay_dir / "runner.stdout.log"),
                "runner_stdout_sha256": sha256_file(replay_dir / "runner.stdout.log"),
                "runner_stderr_path": str(replay_dir / "runner.stderr.log"),
                "runner_stderr_sha256": sha256_file(replay_dir / "runner.stderr.log"),
                "evaluator_image_id_before": pre_environment["image_id"],
                "evaluator_image_id_after": post_environment["image_id"],
                "strict_shadow": strict_shadow,
                "strict_shadow_stdout_path": str(strict_shadow_stdout),
                "strict_shadow_stdout_sha256": sha256_file(strict_shadow_stdout),
                "strict_shadow_stderr_path": str(strict_shadow_stderr),
                "strict_shadow_stderr_sha256": sha256_file(strict_shadow_stderr),
            }
        )
        summary_xlsx = replay_dir / "accuracy_summary.xlsx"
        if summary_xlsx.exists():
            replay["upstream_summary_xlsx_path"] = str(summary_xlsx)
            replay["upstream_summary_xlsx_sha256"] = sha256_file(summary_xlsx)
        replay = seal_terminal_replay(checkpoint, trial_key, replay)
        replays.append(replay)
        evaluation["replays"] = replays
        checkpoint.set_evaluation(trial_key, evaluation)

    evaluation["reproducible"] = replay_classifications_match(replays)
    evaluation["status"] = "complete" if len(replays) == replay_count else "incomplete"
    evaluation["completed_at"] = utc_now()
    checkpoint.set_evaluation(trial_key, evaluation)
    return evaluation


def evaluate_campaign(
    *,
    snapshot: BenchmarkSnapshot,
    checkpoint: CampaignCheckpoint,
    output_dir: Path,
    validator_command: Sequence[str],
    replay_count: int,
    evaluator_timeout: float,
    expected_image_id: str | None,
    evaluator_environment_lock: dict[str, Any],
    sandbox_attestation_path: Path | None,
    sandbox_attestation_signature_path: Path | None,
    sandbox_attestation_public_key_path: Path | None,
) -> None:
    data = checkpoint.snapshot()
    config = data["config"]
    configured_replays = int(config.get("validator_replays") or 0)
    if replay_count != configured_replays or configured_replays not in range(2, 5):
        raise GateError("validator replay count differs from the immutable campaign configuration")
    normalized_evaluator_id = _normalize_sha256(expected_image_id)
    if normalized_evaluator_id is None:
        raise GateError(
            "comparable scoring requires --expected-evaluator-image-id=sha256:...; "
            "automatic pinning of a mutable tag is not accepted"
        )
    full_task_ordinals = list(range(1, PARENT_TASKS_PER_TRIAL + 1))
    if config.get("selected_ordinals") != full_task_ordinals:
        raise GateError(
            "scientific evaluation requires all 49 tasks; diagnostic subsets are submission-only"
        )
    evaluation_mode = config.get("evaluation_mode")
    if evaluation_mode == "promotion":
        expected_trials = PROMOTION_TRIALS
    elif evaluation_mode == "diagnostic_full_trial":
        expected_trials = 1
    else:
        raise GateError(f"unsupported scientific evaluation mode: {evaluation_mode!r}")
    if config.get("trial_count") != expected_trials:
        raise GateError(
            f"{evaluation_mode} evaluation requires exactly {expected_trials} complete trial(s)"
        )
    runtime_image_digest = str(config.get("runtime_image_digest") or "")
    for trial in range(1, expected_trials + 1):
        run_official_trial_evaluation(
            snapshot=snapshot,
            checkpoint=checkpoint,
            output_dir=output_dir,
            trial=trial,
            validator_command=validator_command,
            replay_count=replay_count,
            evaluator_timeout=evaluator_timeout,
            runtime_image_digest=runtime_image_digest,
            expected_image_id=normalized_evaluator_id,
            evaluator_environment_lock=evaluator_environment_lock,
            sandbox_attestation_path=sandbox_attestation_path,
            sandbox_attestation_signature_path=sandbox_attestation_signature_path,
            sandbox_attestation_public_key_path=sandbox_attestation_public_key_path,
        )


def threshold_counts() -> dict[str, int]:
    return {
        "runnable_minimum": RUNNABLE_MINIMUM,
        "scientific_minimum": SCIENTIFIC_MINIMUM,
        "per_trial_runnable_minimum": PER_TRIAL_RUNNABLE_MINIMUM,
        "per_trial_scientific_minimum": PER_TRIAL_SCIENTIFIC_MINIMUM,
    }


def _campaign_evidence_path(
    value: Any,
    *,
    campaign_root: Path | None,
    issues: list[str],
    label: str,
) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        issues.append(f"{label}: missing path")
        return None
    raw_path = Path(raw).expanduser()
    root = campaign_root.expanduser().resolve() if campaign_root is not None else None
    path = (
        (root / raw_path) if root is not None and not raw_path.is_absolute() else raw_path
    ).resolve()
    if root is not None and not path.is_relative_to(root):
        issues.append(f"{label}: path escapes campaign root")
        return None
    if not path.is_file():
        issues.append(f"{label}: file is missing")
        return None
    return path


def _verify_evidence_file(
    value: Any,
    expected_sha256: Any,
    *,
    campaign_root: Path | None,
    issues: list[str],
    label: str,
) -> Path | None:
    path = _campaign_evidence_path(
        value,
        campaign_root=campaign_root,
        issues=issues,
        label=label,
    )
    if path is None:
        return None
    observed = sha256_file(path)
    expected = str(expected_sha256 or "").removeprefix("sha256:")
    if observed != expected:
        issues.append(f"{label}: SHA-256 mismatch")
        return None
    return path


def _read_verified_json(
    value: Any,
    expected_sha256: Any,
    *,
    campaign_root: Path | None,
    issues: list[str],
    label: str,
) -> Any:
    path = _verify_evidence_file(
        value,
        expected_sha256,
        campaign_root=campaign_root,
        issues=issues,
        label=label,
    )
    if path is None:
        return None
    try:
        return read_json_file_strict(path, label=label)
    except GateError as exc:
        issues.append(str(exc))
        return None


def _register_terminal_seal_path(
    path_value: Any,
    sha_value: Any,
    *,
    directory: str,
    referenced_paths: set[str],
    issues: list[str],
    label: str,
) -> None:
    sha256 = str(sha_value or "")
    expected = (Path(directory) / f"{sha256}.json").as_posix()
    actual = Path(str(path_value or "")).as_posix()
    if not SHA256_HEX_RE.fullmatch(sha256) or actual != expected:
        issues.append(f"{label}: terminal seal path is not content-addressed")
        return
    if expected in referenced_paths:
        issues.append(f"{label}: duplicate terminal seal reference")
        return
    referenced_paths.add(expected)


def _terminal_seal_directory_exact(
    campaign_root: Path | None,
    *,
    directory: str,
    referenced_paths: set[str],
    issues: list[str],
) -> bool:
    if campaign_root is None:
        if referenced_paths:
            issues.append(f"{directory}: campaign root is required to enumerate terminal seals")
            return False
        return True
    root = campaign_root.expanduser().resolve()
    seal_root = root / directory
    if not seal_root.exists():
        actual_paths: set[str] = set()
        structure_valid = True
    elif not seal_root.is_dir() or seal_root.is_symlink():
        actual_paths = set()
        structure_valid = False
    else:
        entries = list(seal_root.rglob("*"))
        structure_valid = all(
            entry.parent == seal_root
            and entry.is_file()
            and not entry.is_symlink()
            and re.fullmatch(r"[0-9a-f]{64}\.json", entry.name) is not None
            for entry in entries
        )
        actual_paths = {
            entry.relative_to(root).as_posix()
            for entry in entries
            if entry.is_file() and not entry.is_symlink()
        }
    exact = structure_valid and actual_paths == referenced_paths
    if not exact:
        issues.append(
            f"{directory}: terminal seal directory differs from checkpoint references "
            f"(referenced={len(referenced_paths)}, observed={len(actual_paths)})"
        )
    return exact


def revalidate_checkpoint_evidence(
    snapshot: BenchmarkSnapshot,
    state: dict[str, Any],
    *,
    campaign_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute promotion-relevant facts from hashed files, never state booleans."""

    audited = copy.deepcopy(state)
    issues: list[str] = []
    if audited.get("schema_version") != SCHEMA_VERSION:
        issues.append("checkpoint schema version mismatch")
    benchmark_record = audited.get("benchmark")
    if not isinstance(benchmark_record, dict):
        benchmark_record = {}
        issues.append("checkpoint benchmark record must be an object")
    if benchmark_record.get("sha256") != snapshot.manifest_sha256:
        issues.append("checkpoint benchmark digest mismatch")

    config = audited.get("config")
    if not isinstance(config, dict):
        config = {}
        audited["config"] = config
        issues.append("checkpoint config must be an object")
    configured_environment_lock = config.get("evaluator_environment_lock")
    if isinstance(configured_environment_lock, dict):
        lock_path = Path(__file__).resolve().parents[1] / str(
            configured_environment_lock.get("path") or ""
        )
        try:
            recomputed_lock = load_approved_evaluator_environment_lock(lock_path)
        except GateError as exc:
            issues.append(f"evaluator environment lock revalidation failed ({exc})")
            recomputed_lock = None
        if recomputed_lock != configured_environment_lock:
            issues.append("forged, stale, or unapproved evaluator environment lock")
        config["evaluator_environment_lock"] = recomputed_lock
    declared_model = str(config.get("model_id") or "")
    declared_provider = str(config.get("provider_id") or "")
    if config.get("evaluation_profile") != MATERIALS_CLEANROOM_PROFILE:
        issues.append("invalid or missing immutable materials clean-room evaluation profile")
    trial_count_value = config.get("trial_count")
    valid_trial_count = (
        isinstance(trial_count_value, int)
        and not isinstance(trial_count_value, bool)
        and 0 <= trial_count_value <= PROMOTION_TRIALS
    )
    trial_count = trial_count_value if valid_trial_count else 0
    if not valid_trial_count:
        issues.append("invalid immutable trial count")
    configured_replays_value = config.get("validator_replays")
    configured_replays = (
        configured_replays_value
        if isinstance(configured_replays_value, int)
        and not isinstance(configured_replays_value, bool)
        else 0
    )
    if configured_replays not in range(2, 5):
        issues.append("invalid or missing immutable validator replay count")
    raw_selected_ordinals = config.get("selected_ordinals")
    if not isinstance(raw_selected_ordinals, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in raw_selected_ordinals
    ):
        issues.append("invalid immutable selected task ordinals")
        raw_selected_ordinals = []
    selected_ordinals = set(raw_selected_ordinals)
    valid_snapshot_ordinals = {task.ordinal for task in snapshot.tasks}
    if len(selected_ordinals) != len(raw_selected_ordinals) or not selected_ordinals.issubset(
        valid_snapshot_ordinals
    ):
        issues.append("selected task ordinals are duplicated or outside the benchmark")
    selected_tasks = [task for task in snapshot.tasks if task.ordinal in selected_ordinals]
    expected_keys = {
        attempt_key(trial, task.task_id)
        for trial in range(1, trial_count + 1)
        for task in selected_tasks
    }
    attempts = audited.setdefault("attempts", {})
    if not isinstance(attempts, dict):
        issues.append("checkpoint attempts must be an object")
        attempts = {}
        audited["attempts"] = attempts
    unexpected = sorted(set(attempts) - expected_keys)
    if unexpected:
        issues.append(f"unexpected checkpoint attempts: {', '.join(unexpected)}")

    verified_attempts = 0
    verified_terminal_records = 0
    referenced_attempt_terminal_paths: set[str] = set()
    for trial in range(1, trial_count + 1):
        for task in selected_tasks:
            key = attempt_key(trial, task.task_id)
            attempt = attempts.get(key)
            label = f"attempt {key}"
            if not isinstance(attempt, dict):
                issues.append(f"{label}: missing")
                continue
            if attempt.get("submission_status") not in {
                "captured",
                "terminal_failure",
                "missing_code",
            }:
                issues.append(f"{label}: non-terminal submission")
                continue

            sealed_attempt = {
                key: value
                for key, value in attempt.items()
                if key not in {"terminal_record_path", "terminal_record_sha256"}
            }
            _register_terminal_seal_path(
                attempt.get("terminal_record_path"),
                attempt.get("terminal_record_sha256"),
                directory="terminal-attempts",
                referenced_paths=referenced_attempt_terminal_paths,
                issues=issues,
                label=f"{label} terminal record",
            )
            terminal_record = _read_verified_json(
                attempt.get("terminal_record_path"),
                attempt.get("terminal_record_sha256"),
                campaign_root=campaign_root,
                issues=issues,
                label=f"{label} terminal record",
            )
            expected_terminal_record = {
                "schema_version": "1",
                "attempt_key": key,
                "attempt": sealed_attempt,
            }
            if terminal_record != expected_terminal_record:
                issues.append(f"{label}: terminal record does not match checkpoint attempt")
            else:
                verified_terminal_records += 1

            prompt_path = _verify_evidence_file(
                attempt.get("prompt_path"),
                attempt.get("prompt_sha256"),
                campaign_root=campaign_root,
                issues=issues,
                label=f"{label} prompt",
            )
            prompt_exact = False
            if prompt_path is not None:
                prompt_exact = prompt_path.read_text(encoding="utf-8") == build_ultra_prompt(
                    task.question_text
                )
                if not prompt_exact:
                    issues.append(f"{label}: prompt is not the benchmark-neutral question prompt")
            attempt["prompt_inputs"] = ["question.txt"] if prompt_exact else []
            attempt["expected_values_exposed"] = not prompt_exact
            attempt["verifier_exposed"] = not prompt_exact

            code_path = _verify_evidence_file(
                attempt.get("code_path"),
                attempt.get("code_sha256"),
                campaign_root=campaign_root,
                issues=issues,
                label=f"{label} code",
            )
            if code_path is not None:
                try:
                    source = code_path.read_text(encoding="utf-8")
                    selected_name = select_submission_function(
                        source, str(attempt.get("function_name") or "")
                    )
                    if selected_name != attempt.get("function_name"):
                        issues.append(f"{label}: function name changed")
                except (OSError, UnicodeDecodeError, GateError) as exc:
                    issues.append(f"{label}: invalid captured code ({exc})")

            run_record = _read_verified_json(
                attempt.get("run_record_path"),
                attempt.get("run_record_sha256"),
                campaign_root=campaign_root,
                issues=issues,
                label=f"{label} run record",
            )
            event_records = _read_verified_json(
                attempt.get("events_record_path"),
                attempt.get("events_record_sha256"),
                campaign_root=campaign_root,
                issues=issues,
                label=f"{label} event record",
            )
            artifact_records = _read_verified_json(
                attempt.get("artifacts_record_path"),
                attempt.get("artifacts_record_sha256"),
                campaign_root=campaign_root,
                issues=issues,
                label=f"{label} artifact record",
            )
            _verify_evidence_file(
                attempt.get("response_path"),
                attempt.get("response_sha256"),
                campaign_root=campaign_root,
                issues=issues,
                label=f"{label} response",
            )

            if isinstance(run_record, dict):
                for field in ("run_id", "thread_id", "status"):
                    expected_value = (
                        attempt.get("run_status") if field == "status" else attempt.get(field)
                    )
                    if run_record.get(field) != expected_value:
                        issues.append(f"{label}: run record {field} mismatch")
            else:
                run_record = {}
            if not isinstance(event_records, list) or not all(
                isinstance(item, dict) for item in event_records
            ):
                issues.append(f"{label}: event record must be an object list")
                event_records = []
            recomputed_trace = _trace_summary(event_records)
            claimed_trace = attempt.get("trace_summary", {})
            for field in (
                "observed_models",
                "observed_providers",
                "production_execute_tool_evidence",
                "observed_execute_image_digests",
                "server_evaluation_profiles",
                "server_cleanroom_profile_attested",
                "worker_cleanroom_attestation_count",
                "worker_cleanroom_profile_attested",
                "worker_cleanroom_attestations",
            ):
                if claimed_trace.get(field) != recomputed_trace.get(field):
                    issues.append(f"{label}: forged or stale trace summary field {field}")
            attempt["trace_summary"] = recomputed_trace
            attempt["actual_runtime_provenance"] = _actual_runtime_provenance(
                recomputed_trace,
                declared_model_id=declared_model,
                declared_provider_id=declared_provider,
            )
            recomputed_cleanroom = _worker_cleanroom_binding(
                recomputed_trace,
                run_id=str(attempt.get("run_id") or ""),
                thread_id=str(attempt.get("thread_id") or ""),
                goal_sha256=(sha256_file(prompt_path) if prompt_path is not None else ""),
                user_id_sha256=(
                    str(run_record.get("user_id_sha256"))
                    if SHA256_HEX_RE.fullmatch(str(run_record.get("user_id_sha256") or ""))
                    else None
                ),
            )
            if attempt.get("cleanroom_binding") != recomputed_cleanroom:
                issues.append(f"{label}: forged or stale clean-room identity binding")
            attempt["cleanroom_binding"] = recomputed_cleanroom

            if not isinstance(artifact_records, list) or not all(
                isinstance(item, dict) for item in artifact_records
            ):
                issues.append(f"{label}: artifact record must be an object list")
                artifact_records = []
            attempt["artifact_records"] = artifact_records
            attempt["artifact_ids"] = [
                str(item.get("artifact_id")) for item in artifact_records if item.get("artifact_id")
            ]
            artifact_ok = False
            if attempt.get("source_kind") == "artifact" and code_path is not None:
                matching = [
                    item
                    for item in artifact_records
                    if item.get("artifact_id") == attempt.get("solution_artifact_id")
                    and Path(str(item.get("path") or "")).name == SOLUTION_FILENAME
                ]
                if len(matching) == 1:
                    declared_sha = _normalize_sha256(str(matching[0].get("sha256") or ""))
                    artifact_ok = declared_sha == "sha256:" + sha256_file(code_path)
                if not artifact_ok:
                    issues.append(f"{label}: solution artifact linkage/hash is invalid")
            attempt["required_artifact_present"] = artifact_ok
            verified_attempts += 1

    evaluations = audited.setdefault("evaluations", {})
    if not isinstance(evaluations, dict):
        issues.append("checkpoint evaluations must be an object")
        evaluations = {}
        audited["evaluations"] = evaluations
    expected_evaluation_keys = {
        f"trial-{trial:02d}"
        for trial in range(1, trial_count + 1)
        if config.get("evaluation_mode") != "submission_only"
    }
    unexpected_evaluations = sorted(set(evaluations) - expected_evaluation_keys)
    missing_evaluations = sorted(expected_evaluation_keys - set(evaluations))
    if unexpected_evaluations:
        issues.append("unexpected checkpoint evaluations: " + ", ".join(unexpected_evaluations))
    if missing_evaluations:
        issues.append("missing checkpoint evaluations: " + ", ".join(missing_evaluations))
    recomputed_replay_count = 0
    verified_replay_terminal_records = 0
    verified_failed_replay_terminal_records = 0
    referenced_replay_terminal_paths: set[str] = set()
    for trial in range(1, trial_count + 1):
        trial_key = f"trial-{trial:02d}"
        evaluation = evaluations.get(trial_key)
        if not isinstance(evaluation, dict):
            continue
        environment = evaluation.get("evaluator_environment")
        if not isinstance(environment, dict):
            issues.append(f"{trial_key}: missing evaluator environment provenance")
            environment = {}
        resolved_packages = environment.get("resolved_packages")
        approved_lock = config.get("evaluator_environment_lock")
        resolved_hash_valid = isinstance(resolved_packages, dict) and environment.get(
            "resolved_environment_sha256"
        ) == sha256_bytes(canonical_json_bytes(resolved_packages))
        required_pins_valid = isinstance(resolved_packages, dict) and all(
            resolved_packages.get(name) == version
            for name, version in OFFICIAL_PACKAGE_VERSIONS.items()
        )
        approved_build = approved_lock.get("build", {}) if isinstance(approved_lock, dict) else {}
        approved_upstream = (
            approved_lock.get("upstream", {}) if isinstance(approved_lock, dict) else {}
        )
        approved_platform = (
            approved_lock.get("platform", {}) if isinstance(approved_lock, dict) else {}
        )
        expected_labels = {
            "io.ultra.mattools.adapted-requirements-sha256": approved_build.get(
                "adapted_requirements_sha256"
            ),
            "io.ultra.mattools.base-image": approved_build.get("base_image"),
            "io.ultra.mattools.environment-kind": approved_lock.get("environment_kind")
            if isinstance(approved_lock, dict)
            else None,
            "io.ultra.mattools.official-artifact": "false",
            "io.ultra.mattools.snapshot-manifest-sha256": approved_upstream.get("manifest_sha256"),
            "io.ultra.mattools.safe-parser-sha256": approved_build.get("safe_parser_sha256"),
            "io.ultra.mattools.runner-wrapper-sha256": approved_build.get(
                "runner_wrapper_sha256"
            ),
            "io.ultra.mattools.semantic-repairs-sha256": approved_build.get(
                "semantic_repairs_sha256"
            ),
            "io.ultra.mattools.strict-shadow-sha256": approved_build.get("strict_shadow_sha256"),
            "io.ultra.mattools.supplemental-requirements-sha256": approved_build.get(
                "supplemental_requirements_sha256"
            ),
            "io.ultra.mattools.target-platform": approved_platform.get("docker"),
            "io.ultra.mattools.tool-source-manifest-sha256": approved_build.get(
                "tool_source_manifest_sha256"
            ),
            "io.ultra.mattools.candidate-fixture-file-count": str(
                approved_build.get("candidate_fixture_file_count")
            ),
            "io.ultra.mattools.candidate-fixture-manifest-sha256": approved_build.get(
                "candidate_fixture_manifest_sha256"
            ),
            "io.ultra.mattools.candidate-visible-source-policy": approved_build.get(
                "candidate_visible_source_policy"
            ),
            "io.ultra.mattools.upstream-requirements-sha256": approved_upstream.get(
                "requirements_sha256"
            ),
            "org.opencontainers.image.revision": approved_upstream.get("revision"),
        }
        labels_match = environment.get("image_labels") == dict(sorted(expected_labels.items()))
        expected_embedded_inputs = {
            "candidate_fixture_file_count": approved_build.get("candidate_fixture_file_count"),
            "candidate_fixture_manifest_sha256": approved_build.get(
                "candidate_fixture_manifest_sha256"
            ),
            "candidate_visible_non_fixture_paths": [],
            "candidate_visible_executable_source_paths": [],
            "upstream_requirements_sha256": approved_upstream.get("requirements_sha256"),
            "adapted_requirements_sha256": approved_build.get("adapted_requirements_sha256"),
            "supplemental_requirements_sha256": approved_build.get(
                "supplemental_requirements_sha256"
            ),
        }
        embedded_inputs_match = environment.get("embedded_inputs") == expected_embedded_inputs
        platform_match = environment.get("platform") == approved_platform
        if environment.get("labels_match_approved_lock") is not labels_match:
            issues.append(f"{trial_key}: forged or stale evaluator image-label provenance")
        if environment.get("embedded_inputs_match_approved_lock") is not embedded_inputs_match:
            issues.append(f"{trial_key}: forged or stale evaluator embedded-input provenance")
        if environment.get("platform_matches_approved_lock") is not platform_match:
            issues.append(f"{trial_key}: forged or stale evaluator platform provenance")
        full_lock_matches = (
            isinstance(approved_lock, dict)
            and approved_lock.get("approved_from_git_head") is True
            and approved_lock.get("environment_kind") == EVALUATOR_ENVIRONMENT_KIND
            and approved_lock.get("official_artifact") is False
            and approved_lock.get("python_version") == environment.get("python_version")
            and approved_lock.get("packages") == resolved_packages
            and approved_lock.get("package_map_sha256")
            == sha256_bytes(canonical_json_bytes(resolved_packages))
            and environment.get("approved_environment_lock") == approved_lock
            and environment.get("environment_kind") == EVALUATOR_ENVIRONMENT_KIND
            and environment.get("official_artifact") is False
            and environment.get("task_execution_performed") is False
            and labels_match
            and embedded_inputs_match
            and platform_match
        )
        expected_evaluator_id = config.get("expected_evaluator_image_id")
        image_identity_valid = _normalize_sha256(
            str(environment.get("image_id") or "")
        ) == expected_evaluator_id and expected_evaluator_id != _normalize_sha256(
            str(config.get("runtime_image_digest") or "")
        )
        recomputed_comparable = (
            resolved_hash_valid
            and required_pins_valid
            and full_lock_matches
            and image_identity_valid
        )
        if environment.get("comparable") is not recomputed_comparable:
            issues.append(f"{trial_key}: forged or stale evaluator comparability")
        environment["comparable"] = recomputed_comparable
        environment["full_environment_lock_matches"] = full_lock_matches
        environment["labels_match_approved_lock"] = labels_match
        environment["embedded_inputs_match_approved_lock"] = embedded_inputs_match
        environment["platform_matches_approved_lock"] = platform_match
        evaluation["evaluator_environment"] = environment

        runner_record = evaluation.get("runner")
        if not isinstance(runner_record, dict):
            issues.append(f"{trial_key}: missing runner provenance")
        else:
            try:
                expected_validator_command = list(pinned_validator_command())
                expected_validator_sha = sha256_file(Path(expected_validator_command[0]).resolve())
            except GateError as exc:
                issues.append(f"{trial_key}: validator command unavailable ({exc})")
                expected_validator_command = []
                expected_validator_sha = ""
            runner_expectations = {
                "sha256": OFFICIAL_RUNNER_SHA256,
                "unmodified_official": True,
                "host_validator_command": expected_validator_command,
                "host_validator_executable_sha256": expected_validator_sha,
                "host_requirements_sha256": sha256_file(HOST_VALIDATOR_REQUIREMENTS),
                "host_input_requirements_sha256": sha256_file(HOST_VALIDATOR_REQUIREMENTS_INPUT),
                "host_validator_environment": state.get("config", {}).get(
                    "host_validator_environment"
                ),
                "safe_parser_path": str(SAFE_PARSER_SCRIPT),
                "safe_parser_sha256": sha256_file(SAFE_PARSER_SCRIPT),
                "runner_wrapper_path": str(RUNNER_WRAPPER_SCRIPT),
                "runner_wrapper_sha256": sha256_file(RUNNER_WRAPPER_SCRIPT),
                "candidate_host_eval_removed": True,
                "strict_shadow_sha256": sha256_file(STRICT_SHADOW_SCRIPT),
                "semantic_repairs_sha256": sha256_file(SEMANTIC_REPAIRS_SCRIPT),
            }
            for field, expected_value in runner_expectations.items():
                if runner_record.get(field) != expected_value:
                    issues.append(f"{trial_key}: runner provenance mismatch for {field}")

        stored_attestation = evaluation.get("sandbox_policy_attestation")
        if isinstance(stored_attestation, dict) and environment.get("image_id"):
            try:
                recomputed_attestation = validate_sandbox_attestation(
                    Path(str(stored_attestation.get("path") or "")),
                    image_id=str(environment["image_id"]),
                    signature_path=Path(
                        str(stored_attestation.get("detached_signature_path") or "")
                    ),
                    public_key_path=Path(
                        str(stored_attestation.get("operator_public_key_path") or "")
                    ),
                )
            except GateError as exc:
                issues.append(f"{trial_key}: sandbox attestation revalidation failed ({exc})")
                recomputed_attestation = {"valid": False, "issues": [str(exc)]}
            if stored_attestation.get("valid") != recomputed_attestation.get("valid"):
                issues.append(f"{trial_key}: forged or stale sandbox attestation")
            evaluation["sandbox_policy_attestation"] = recomputed_attestation
        else:
            evaluation["sandbox_policy_attestation"] = {"valid": False}
        failed_replays = evaluation.get("failed_replays", [])
        if not isinstance(failed_replays, list):
            issues.append(f"{trial_key}: failed_replays must be a list")
            failed_replays = []
        if failed_replays:
            issues.append(
                f"{trial_key}: failed evaluator replay history makes the campaign non-comparable"
            )
        for failed_index, failed_replay in enumerate(failed_replays, start=1):
            failed_label = f"{trial_key} failed replay record {failed_index}"
            if not isinstance(failed_replay, dict):
                issues.append(f"{failed_label}: invalid record")
                continue
            _register_terminal_seal_path(
                failed_replay.get("terminal_record_path"),
                failed_replay.get("terminal_record_sha256"),
                directory="terminal-replays",
                referenced_paths=referenced_replay_terminal_paths,
                issues=issues,
                label=f"{failed_label} terminal record",
            )
            failed_terminal = _read_verified_json(
                failed_replay.get("terminal_record_path"),
                failed_replay.get("terminal_record_sha256"),
                campaign_root=campaign_root,
                issues=issues,
                label=f"{failed_label} terminal record",
            )
            try:
                expected_failed_terminal = replay_terminal_record_payload(
                    trial_key,
                    failed_replay,
                )
            except GateError as exc:
                issues.append(f"{failed_label}: invalid terminal record payload ({exc})")
                continue
            if failed_terminal != expected_failed_terminal:
                issues.append(f"{failed_label}: replay differs from terminal record")
                continue
            verified_failed_replay_terminal_records += 1
            failure_artifacts = failed_replay.get("failure_artifacts")
            if not isinstance(failure_artifacts, list) or failed_replay.get(
                "failure_artifact_manifest_sha256"
            ) != sha256_bytes(canonical_json_bytes(failure_artifacts)):
                issues.append(f"{failed_label}: invalid failure artifact manifest")
                failure_artifacts = []
            observed_failure_paths: set[str] = set()
            for artifact_index, artifact in enumerate(failure_artifacts, start=1):
                if not isinstance(artifact, dict):
                    issues.append(f"{failed_label}: failure artifact {artifact_index} is invalid")
                    continue
                artifact_path = str(artifact.get("path") or "")
                if artifact_path in observed_failure_paths:
                    issues.append(f"{failed_label}: duplicate failure artifact path")
                    continue
                observed_failure_paths.add(artifact_path)
                _verify_evidence_file(
                    artifact_path,
                    artifact.get("sha256"),
                    campaign_root=campaign_root,
                    issues=issues,
                    label=f"{failed_label} artifact {artifact_index}",
                )
        stored_replays = evaluation.get("replays")
        if not isinstance(stored_replays, list):
            issues.append(f"{trial_key}: replays must be a list")
            stored_replays = []
        verified_replays: list[dict[str, Any]] = []
        for replay_index, stored_replay in enumerate(stored_replays, start=1):
            replay_label = f"{trial_key} replay {replay_index}"
            if not isinstance(stored_replay, dict):
                issues.append(f"{replay_label}: invalid record")
                continue
            if stored_replay.get("replay") != replay_index:
                issues.append(f"{replay_label}: replay order/identity mismatch")
                continue
            _register_terminal_seal_path(
                stored_replay.get("terminal_record_path"),
                stored_replay.get("terminal_record_sha256"),
                directory="terminal-replays",
                referenced_paths=referenced_replay_terminal_paths,
                issues=issues,
                label=f"{replay_label} terminal record",
            )
            terminal_record = _read_verified_json(
                stored_replay.get("terminal_record_path"),
                stored_replay.get("terminal_record_sha256"),
                campaign_root=campaign_root,
                issues=issues,
                label=f"{replay_label} terminal record",
            )
            try:
                expected_terminal_record = replay_terminal_record_payload(
                    trial_key,
                    stored_replay,
                )
            except GateError as exc:
                issues.append(f"{replay_label}: invalid terminal record payload ({exc})")
                continue
            if terminal_record != expected_terminal_record:
                issues.append(f"{replay_label}: replay differs from terminal record")
                continue
            verified_replay_terminal_records += 1
            input_path = _verify_evidence_file(
                stored_replay.get("input_jsonl_path"),
                stored_replay.get("input_jsonl_sha256"),
                campaign_root=campaign_root,
                issues=issues,
                label=f"{replay_label} input",
            )
            input_ok = False
            if input_path is not None:
                try:
                    expected_input = official_jsonl_content(
                        snapshot=snapshot,
                        checkpoint_data=audited,
                        trial=trial,
                    )
                    input_ok = input_path.read_text(encoding="utf-8") == expected_input
                except GateError as exc:
                    issues.append(f"{replay_label}: cannot reconstruct input ({exc})")
                if not input_ok:
                    issues.append(f"{replay_label}: input JSONL differs from captured code")
            log_path = _verify_evidence_file(
                stored_replay.get("log_path"),
                stored_replay.get("log_sha256"),
                campaign_root=campaign_root,
                issues=issues,
                label=f"{replay_label} upstream log",
            )
            for prefix in ("runner_stdout", "runner_stderr"):
                _verify_evidence_file(
                    stored_replay.get(f"{prefix}_path")
                    or Path(str(stored_replay.get("input_jsonl_path") or "")).parent
                    / f"{prefix.replace('_', '.')}.log",
                    stored_replay.get(f"{prefix}_sha256"),
                    campaign_root=campaign_root,
                    issues=issues,
                    label=f"{replay_label} {prefix}",
                )
            if log_path is None or not input_ok:
                continue
            try:
                parsed = parse_official_evaluation_log(log_path, snapshot.tasks)
            except GateError as exc:
                issues.append(f"{replay_label}: official log recomputation failed ({exc})")
                continue
            for field in (
                "runnable",
                "runnable_denominator",
                "scientific_pass",
                "scientific_denominator",
                "strict_scientific_pass",
                "strict_semantics_valid",
                "full_question_success",
                "results",
                "upstream_summary",
            ):
                if stored_replay.get(field) != parsed.get(field):
                    issues.append(f"{replay_label}: forged or stale {field}")
            stored_shadow = stored_replay.get("strict_shadow")
            if not isinstance(stored_shadow, dict):
                issues.append(f"{replay_label}: missing strict pre-normalization shadow")
                continue
            shadow_path = _verify_evidence_file(
                stored_shadow.get("path"),
                stored_shadow.get("sha256"),
                campaign_root=campaign_root,
                issues=issues,
                label=f"{replay_label} strict shadow",
            )
            for prefix in ("strict_shadow_stdout", "strict_shadow_stderr"):
                _verify_evidence_file(
                    stored_replay.get(f"{prefix}_path"),
                    stored_replay.get(f"{prefix}_sha256"),
                    campaign_root=campaign_root,
                    issues=issues,
                    label=f"{replay_label} {prefix}",
                )
            if shadow_path is None:
                continue
            try:
                strict_shadow = parse_strict_shadow_report(
                    shadow_path,
                    snapshot.tasks,
                    expected_submission_sha256=sha256_file(input_path),
                )
            except GateError as exc:
                issues.append(f"{replay_label}: strict shadow recomputation failed ({exc})")
                continue
            for field in (
                "status",
                "runnable",
                "runnable_denominator",
                "scientific_pass",
                "upstream_strict_scientific_pass",
                "scientific_denominator",
                "results",
                "pre_normalization_captured",
                "semantic_repairs_applied",
                "semantic_repair_spec_sha256",
                "published_score_unchanged",
            ):
                if stored_shadow.get(field) != strict_shadow.get(field):
                    issues.append(f"{replay_label}: forged or stale strict shadow {field}")
            expected_image_id = environment.get("image_id")
            if (
                stored_replay.get("evaluator_image_id_before") != expected_image_id
                or stored_replay.get("evaluator_image_id_after") != expected_image_id
            ):
                issues.append(f"{replay_label}: evaluator image changed across replay")
            verified = {**stored_replay, **parsed, "strict_shadow": strict_shadow}
            verified_replays.append(verified)
            recomputed_replay_count += 1
        evaluation["replays"] = verified_replays
        evaluation["reproducible"] = replay_classifications_match(verified_replays)
        if len(verified_replays) != configured_replays:
            issues.append(
                f"{trial_key}: verified replay count {len(verified_replays)} does not match "
                f"configured count {configured_replays}"
            )
        evaluation["status"] = (
            "complete"
            if len(verified_replays) == configured_replays and evaluation["reproducible"]
            else "incomplete"
        )

    attempt_terminal_directory_exact = _terminal_seal_directory_exact(
        campaign_root,
        directory="terminal-attempts",
        referenced_paths=referenced_attempt_terminal_paths,
        issues=issues,
    )
    replay_terminal_directory_exact = _terminal_seal_directory_exact(
        campaign_root,
        directory="terminal-replays",
        referenced_paths=referenced_replay_terminal_paths,
        issues=issues,
    )
    expected_replay_terminal_records = len(expected_evaluation_keys) * configured_replays
    audit = {
        "valid": not issues,
        "issues": issues,
        "verified_attempt_count": verified_attempts,
        "recomputed_replay_count": recomputed_replay_count,
        "audited_at": str(audited.get("updated_at") or audited.get("created_at") or ""),
        "trusted_state_booleans": False,
        "expected_attempt_count": len(expected_keys),
        "actual_attempt_count": len(attempts),
        "attempt_key_set_exact": set(attempts) == expected_keys,
        "verified_terminal_record_count": verified_terminal_records,
        "terminal_attempts_non_replaced": verified_terminal_records == len(expected_keys),
        "terminal_attempt_directory_exact": attempt_terminal_directory_exact,
        "verified_replay_terminal_record_count": verified_replay_terminal_records,
        "expected_replay_terminal_record_count": expected_replay_terminal_records,
        "terminal_replays_non_replaced": (
            verified_replay_terminal_records == expected_replay_terminal_records
        ),
        "terminal_replay_directory_exact": replay_terminal_directory_exact,
        "failed_replay_terminal_record_count": verified_failed_replay_terminal_records,
    }
    return audited, audit


def _single_task_result(
    results: Any,
    task_id: str,
) -> dict[str, Any] | None:
    if not isinstance(results, list):
        return None
    matches = [
        item for item in results if isinstance(item, dict) and item.get("task_id") == task_id
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def attempt_scoring_evidence(
    task: BenchmarkTask,
    replays: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Emit independently summable semantic, strict, and published task facts."""

    records: list[dict[str, Any]] = []
    for replay_index, replay in enumerate(replays, start=1):
        published = _single_task_result(replay.get("results"), task.task_id)
        shadow = replay.get("strict_shadow")
        strict = (
            _single_task_result(shadow.get("results"), task.task_id)
            if isinstance(shadow, dict)
            else None
        )
        if published is None or strict is None:
            continue
        records.append(
            {
                "replay": replay.get("replay", replay_index),
                "replay_terminal_record_sha256": replay.get("terminal_record_sha256"),
                "published_upstream": {
                    "classification": published.get("classification"),
                    "runnable": published.get("runnable"),
                    "scientific_pass": published.get("scientific_pass"),
                    "scientific_fail": published.get("scientific_fail"),
                },
                "strict_shadow": {
                    "semantic_runnable": strict.get("runnable"),
                    "strict_scientific_classification": strict.get("classification"),
                    "strict_scientific_pass": strict.get("scientific_pass"),
                    "strict_scientific_fail": strict.get("scientific_fail"),
                    "strict_exact_ok": strict.get("exact_ok"),
                    "raw_verifier_output_sha256": strict.get("raw_verifier_output_sha256"),
                },
            }
        )

    def fingerprint(record: dict[str, Any]) -> bytes:
        strict = record["strict_shadow"]
        return canonical_json_bytes(
            {
                "published_upstream": record["published_upstream"],
                "strict_shadow": {
                    "semantic_runnable": strict["semantic_runnable"],
                    "strict_scientific_classification": strict["strict_scientific_classification"],
                    "strict_scientific_pass": strict["strict_scientific_pass"],
                    "strict_scientific_fail": strict["strict_scientific_fail"],
                    "strict_exact_ok": strict["strict_exact_ok"],
                },
            }
        )

    replay_consistent = len(records) >= 2 and all(
        fingerprint(record) == fingerprint(records[0]) for record in records[1:]
    )
    return {
        "schema_version": "1",
        "task_id": task.task_id,
        "ordinal": task.ordinal,
        "subtask_count": task.subtask_count,
        "expected_replay_count": len(replays),
        "replay_count": len(records),
        "complete": bool(replays) and len(records) == len(replays),
        "replay_consistent": replay_consistent,
        "primary": copy.deepcopy(records[0]) if records else None,
        "replays": records,
    }


def _trial_report(
    trial: int,
    snapshot: BenchmarkSnapshot,
    state: dict[str, Any],
) -> dict[str, Any]:
    trial_key = f"trial-{trial:02d}"
    evaluation = state.get("evaluations", {}).get(trial_key, {})
    replays = evaluation.get("replays", [])
    primary = replays[0] if replays else None
    attempts = state.get("attempts", {})
    linkage = []
    for task in snapshot.tasks:
        attempt = attempts.get(attempt_key(trial, task.task_id), {})
        result = _single_task_result(primary.get("results"), task.task_id) if primary else None
        scoring_evidence = attempt_scoring_evidence(task, replays)
        linkage.append(
            {
                "task_id": task.task_id,
                "ordinal": task.ordinal,
                "run_id": attempt.get("run_id"),
                "thread_id": attempt.get("thread_id"),
                "run_status": attempt.get("run_status"),
                "submission_status": attempt.get("submission_status"),
                "artifact_ids": attempt.get("artifact_ids", []),
                "solution_artifact_id": attempt.get("solution_artifact_id"),
                "code_sha256": attempt.get("code_sha256"),
                "actual_runtime_provenance": attempt.get("actual_runtime_provenance"),
                "trace_summary": attempt.get("trace_summary"),
                "cleanroom_binding": attempt.get("cleanroom_binding"),
                "evaluation": result,
                "scoring_evidence": scoring_evidence,
            }
        )
    scoring_complete = bool(replays) and all(
        item["scoring_evidence"]["complete"] is True
        and item["scoring_evidence"]["replay_consistent"] is True
        and isinstance(item["scoring_evidence"]["primary"], dict)
        for item in linkage
    )
    primary_scoring = [item["scoring_evidence"]["primary"] for item in linkage]
    semantic_runnable = (
        sum(int(record["strict_shadow"]["semantic_runnable"] is True) for record in primary_scoring)
        if scoring_complete
        else None
    )
    published_runnable = (
        sum(int(record["published_upstream"]["runnable"] is True) for record in primary_scoring)
        if scoring_complete
        else None
    )
    published_scientific = (
        sum(int(record["published_upstream"]["scientific_pass"]) for record in primary_scoring)
        if scoring_complete
        else None
    )
    strict_scientific = (
        sum(int(record["strict_shadow"]["strict_scientific_pass"]) for record in primary_scoring)
        if scoring_complete
        else None
    )
    return {
        "trial": trial,
        "status": evaluation.get("status", "not_evaluated"),
        "scoring_evidence_complete": scoring_complete,
        "runnable": semantic_runnable,
        "published_runner_runnable": published_runnable,
        "runnable_denominator": PARENT_TASKS_PER_TRIAL,
        "function_runnable_rate": (
            semantic_runnable / PARENT_TASKS_PER_TRIAL if semantic_runnable is not None else None
        ),
        "published_runner_function_runnable_rate": (
            published_runnable / PARENT_TASKS_PER_TRIAL if published_runnable is not None else None
        ),
        "scientific_pass": published_scientific,
        "strict_scientific_pass": strict_scientific,
        "scientific_denominator": SCIENTIFIC_SUBTASKS_PER_TRIAL,
        "task_success_rate": (
            published_scientific / SCIENTIFIC_SUBTASKS_PER_TRIAL
            if published_scientific is not None
            else None
        ),
        "strict_task_success_rate": (
            strict_scientific / SCIENTIFIC_SUBTASKS_PER_TRIAL
            if strict_scientific is not None
            else None
        ),
        "full_question_success": primary.get("full_question_success") if primary else None,
        "replay_count": len(replays),
        "reproducible": evaluation.get("reproducible", False),
        "evaluator_environment": evaluation.get("evaluator_environment"),
        "sandbox_policy_attestation": evaluation.get("sandbox_policy_attestation"),
        "attempts": linkage,
    }


def build_report(
    snapshot: BenchmarkSnapshot,
    state: dict[str, Any],
    *,
    campaign_root: Path | None = None,
) -> dict[str, Any]:
    state, evidence_audit = revalidate_checkpoint_evidence(
        snapshot,
        state,
        campaign_root=campaign_root,
    )
    config = state.get("config", {})
    trials = [_trial_report(trial, snapshot, state) for trial in range(1, PROMOTION_TRIALS + 1)]
    attempts = state.get("attempts", {})
    selected_ordinals = config.get("selected_ordinals", [])
    expected_attempt_count = int(config.get("trial_count") or 0) * len(selected_ordinals)
    terminal_attempts = [
        attempt
        for attempt in attempts.values()
        if attempt.get("submission_status") in {"captured", "terminal_failure", "missing_code"}
    ]
    complete_evaluations = all(
        trial["status"] == "complete" and trial["scoring_evidence_complete"] is True
        for trial in trials
    )
    complete_promotion_shape = (
        config.get("trial_count") == PROMOTION_TRIALS
        and selected_ordinals == list(range(1, PARENT_TASKS_PER_TRIAL + 1))
        and len(terminal_attempts) == RUNNABLE_DENOMINATOR
        and complete_evaluations
    )
    runnable = sum(int(trial["runnable"] or 0) for trial in trials)
    published_runner_runnable = sum(
        int(trial["published_runner_runnable"] or 0) for trial in trials
    )
    scientific_pass = sum(int(trial["scientific_pass"] or 0) for trial in trials)
    strict_scientific_pass = sum(int(trial["strict_scientific_pass"] or 0) for trial in trials)
    all_attempts_linked = len(terminal_attempts) == expected_attempt_count and all(
        attempt.get("run_id")
        and attempt.get("thread_id")
        and attempt.get("run_record_sha256")
        and attempt.get("events_record_sha256")
        and attempt.get("artifacts_record_sha256")
        for attempt in terminal_attempts
    )
    artifact_contract = len(terminal_attempts) == expected_attempt_count and all(
        attempt.get("run_status") != "succeeded" or attempt.get("required_artifact_present") is True
        for attempt in terminal_attempts
    )
    evaluator_exact = complete_evaluations and all(
        trial.get("evaluator_environment", {}).get("comparable") is True for trial in trials
    )
    reproducible = complete_evaluations and all(trial["reproducible"] for trial in trials)
    sandbox_security = complete_evaluations and all(
        trial.get("sandbox_policy_attestation", {}).get("valid") is True for trial in trials
    )
    actual_runtime_provenance = (
        bool(expected_attempt_count)
        and len(terminal_attempts) == expected_attempt_count
        and all(
            attempt.get("actual_runtime_provenance", {}).get("validated") is True
            for attempt in terminal_attempts
        )
    )
    production_execute_evidence = (
        bool(expected_attempt_count)
        and len(terminal_attempts) == expected_attempt_count
        and all(
            attempt.get("trace_summary", {}).get("production_execute_tool_evidence") is True
            for attempt in terminal_attempts
        )
    )
    server_authorized_cleanroom = (
        bool(expected_attempt_count)
        and len(terminal_attempts) == expected_attempt_count
        and all(
            attempt.get("trace_summary", {}).get("server_cleanroom_profile_attested") is True
            and attempt.get("trace_summary", {}).get("server_evaluation_profiles")
            == [MATERIALS_CLEANROOM_PROFILE]
            for attempt in terminal_attempts
        )
    )
    worker_enforced_cleanroom = (
        bool(expected_attempt_count)
        and len(terminal_attempts) == expected_attempt_count
        and all(
            attempt.get("cleanroom_binding", {}).get("valid") is True
            and attempt.get("cleanroom_binding", {}).get("user_identity_independently_bound") is True
            for attempt in terminal_attempts
        )
    )
    declared_runtime_digest = _normalize_sha256(config.get("runtime_image_digest"))
    execute_runtime_image_attested = (
        declared_runtime_digest is not None
        and bool(expected_attempt_count)
        and len(terminal_attempts) == expected_attempt_count
        and all(
            declared_runtime_digest
            in attempt.get("trace_summary", {}).get("observed_execute_image_digests", [])
            for attempt in terminal_attempts
        )
    )
    observed_models = sorted(
        {
            model
            for attempt in terminal_attempts
            for model in attempt.get("actual_runtime_provenance", {}).get("observed_model_ids", [])
        }
    )
    observed_providers = sorted(
        {
            provider
            for attempt in terminal_attempts
            for provider in attempt.get("actual_runtime_provenance", {}).get(
                "observed_provider_ids", []
            )
        }
    )
    provenance = all(
        (
            config.get("model_id"),
            config.get("provider_id"),
            _normalize_sha256(config.get("runtime_image_digest")),
            config.get("ultra", {}).get("commit"),
            config.get("ultra", {}).get("skills_sha256"),
            config.get("harness_sha256"),
            config.get("host_validator_requirements_sha256"),
            config.get("host_validator_input_requirements_sha256"),
            config.get("host_validator_environment", {}).get("resolved_packages_sha256"),
            config.get("safe_parser_sha256"),
            config.get("runner_wrapper_sha256"),
            config.get("strict_shadow_sha256"),
            config.get("semantic_repairs_sha256"),
            actual_runtime_provenance,
        )
    )
    official_snapshot = (
        snapshot.strict_official
        and snapshot.revision == OFFICIAL_REVISION
        and snapshot.manifest_sha256 == OFFICIAL_MANIFEST_SHA256
    )
    license_attested = license_attestation_valid(config.get("license_attestation"))
    checkpoint_immutable = (
        evidence_audit["valid"]
        and evidence_audit["attempt_key_set_exact"]
        and evidence_audit["terminal_attempts_non_replaced"]
        and evidence_audit["terminal_replays_non_replaced"]
    )
    expected_values_isolated = len(terminal_attempts) == expected_attempt_count and all(
        attempt.get("expected_values_exposed") is False
        and attempt.get("verifier_exposed") is False
        and attempt.get("prompt_inputs") == ["question.txt"]
        for attempt in terminal_attempts
    )
    evaluator_independent = evaluator_exact and all(
        trial.get("evaluator_environment", {}).get("independent_from_production_runtime") is True
        for trial in trials
    )
    worktree_clean = config.get("ultra", {}).get("dirty") is False
    score_comparable = all(
        (
            complete_promotion_shape,
            official_snapshot,
            license_attested,
            evidence_audit["valid"],
            checkpoint_immutable,
            all_attempts_linked,
            actual_runtime_provenance,
            production_execute_evidence,
            server_authorized_cleanroom,
            worker_enforced_cleanroom,
            execute_runtime_image_attested,
            artifact_contract,
            expected_values_isolated,
            evaluator_exact,
            evaluator_independent,
            reproducible,
            sandbox_security,
            provenance,
            worktree_clean,
        )
    )
    hard_gates = {
        "official_snapshot": official_snapshot,
        "license_attested": license_attested,
        "checkpoint_evidence_integrity": evidence_audit["valid"],
        "checkpoint_non_erasure_integrity": checkpoint_immutable,
        "replay_terminal_evidence_integrity": (
            evidence_audit["terminal_replays_non_replaced"]
            and evidence_audit["terminal_replay_directory_exact"]
        ),
        "three_trial_completeness": complete_promotion_shape,
        "actual_ultra_control_plane_path": all_attempts_linked,
        "actual_model_provider_provenance": actual_runtime_provenance,
        "production_execute_tool_evidence": production_execute_evidence,
        "server_authorized_cleanroom_profile": server_authorized_cleanroom,
        "worker_enforced_cleanroom_profile": worker_enforced_cleanroom,
        "production_execute_runtime_image_attestation": execute_runtime_image_attested,
        "required_solution_artifacts": artifact_contract,
        "expected_values_and_verifiers_isolated": expected_values_isolated,
        "official_evaluator_environment_exact": evaluator_exact,
        "evaluator_independent_from_production": evaluator_independent,
        "immediate_replay_reproducible": reproducible,
        "external_sandbox_isolation_evidence": sandbox_security,
        "provenance_complete": bool(provenance),
        "ultra_worktree_clean": worktree_clean,
        "per_trial_mattools_function_runnable": score_comparable
        and all(
            isinstance(trial.get("runnable"), int)
            and not isinstance(trial.get("runnable"), bool)
            and int(trial["runnable"]) >= PER_TRIAL_RUNNABLE_MINIMUM
            for trial in trials
        ),
        "per_trial_strict_scientific_task_success": score_comparable
        and all(
            isinstance(trial.get("strict_scientific_pass"), int)
            and not isinstance(trial.get("strict_scientific_pass"), bool)
            and int(trial["strict_scientific_pass"]) >= PER_TRIAL_SCIENTIFIC_MINIMUM
            for trial in trials
        ),
        "mattools_function_runnable": score_comparable and runnable >= RUNNABLE_MINIMUM,
        "mattools_task_success": score_comparable and scientific_pass >= SCIENTIFIC_MINIMUM,
        "strict_scientific_task_success": score_comparable
        and strict_scientific_pass >= SCIENTIFIC_MINIMUM,
    }
    reasons = [name for name, passed in hard_gates.items() if not passed]
    lane_passed = not reasons
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": str(state.get("updated_at") or state.get("created_at") or ""),
        "campaign_id": state.get("campaign_id"),
        "benchmark": snapshot.provenance_record(),
        "ultra": config.get("ultra", {}),
        "harness": {
            "path": "scripts/mattools_promotion_gate.py",
            "sha256": config.get("harness_sha256"),
            "host_validator_requirements_path": (
                "scripts/mattools-validator-requirements.lock.txt"
            ),
            "host_validator_requirements_sha256": config.get("host_validator_requirements_sha256"),
            "host_validator_input_requirements_path": (
                "scripts/mattools-validator-requirements.txt"
            ),
            "host_validator_input_requirements_sha256": config.get(
                "host_validator_input_requirements_sha256"
            ),
            "host_validator_environment": config.get("host_validator_environment"),
            "safe_parser_path": "scripts/mattools_safe_parser.py",
            "safe_parser_sha256": config.get("safe_parser_sha256"),
            "runner_wrapper_path": "scripts/mattools_runner_wrapper.py",
            "runner_wrapper_sha256": config.get("runner_wrapper_sha256"),
            "strict_shadow_path": "scripts/mattools_strict_shadow.py",
            "strict_shadow_sha256": config.get("strict_shadow_sha256"),
            "semantic_repairs_path": "scripts/mattools_semantic_repairs.py",
            "semantic_repairs_sha256": config.get("semantic_repairs_sha256"),
        },
        "runtime_environment": {
            "role": "Ultra generation/runtime only; never used for pinned evaluator scoring",
            "image_digest": config.get("runtime_image_digest"),
            "pymatgen_version": config.get("runtime_pymatgen_version"),
            "pymatgen_analysis_defects_version": config.get("runtime_defects_version"),
            "operator_declared_model_id": config.get("model_id"),
            "operator_declared_provider_id": config.get("provider_id"),
            "observed_model_ids": observed_models,
            "observed_provider_ids": observed_providers,
            "actual_model_provider_provenance_validated": actual_runtime_provenance,
            "provenance_note": (
                "CLI model/provider values do not select the runtime. Matching observable "
                "run.token_usage provenance is required for a comparable campaign."
            ),
            "reasoning_mode": config.get("reasoning_mode"),
            "seed": config.get("seed"),
            "seed_supported": config.get("seed_supported", False),
            "budgets": config.get("budgets"),
            "selection_context": {
                "suggested_domain": "materials",
            },
            "evaluation_profile": MATERIALS_CLEANROOM_PROFILE,
        },
        "official_evaluator_environment": {
            "role": "independent scientific scoring only",
            "environment_kind": config.get("evaluator_environment_lock", {}).get(
                "environment_kind"
            ),
            "official_artifact": False,
            "naming_note": (
                "The key name is retained for readiness-contract compatibility; the reviewed "
                "image is explicitly a reconstruction variant, not an official artifact."
            ),
            "required_packages": OFFICIAL_PACKAGE_VERSIONS,
            "source_revision": OFFICIAL_REVISION,
            "runner_sha256": OFFICIAL_RUNNER_SHA256,
            "approved_lock": config.get("evaluator_environment_lock"),
            "observed_trials": [trial.get("evaluator_environment") for trial in trials],
        },
        "license_attestation": config.get("license_attestation"),
        "checkpoint_evidence_audit": evidence_audit,
        "trials": trials,
        "counts": {
            "runnable": runnable if score_comparable else None,
            "runnable_observed_in_completed_trials": runnable,
            "published_runner_runnable": (published_runner_runnable if score_comparable else None),
            "published_runner_runnable_observed_in_completed_trials": (published_runner_runnable),
            "runnable_denominator": RUNNABLE_DENOMINATOR,
            "runnable_minimum": RUNNABLE_MINIMUM,
            "per_trial_runnable_minimum": PER_TRIAL_RUNNABLE_MINIMUM,
            "scientific_pass": scientific_pass if score_comparable else None,
            "scientific_pass_observed_in_completed_trials": scientific_pass,
            "scientific_denominator": SCIENTIFIC_DENOMINATOR,
            "scientific_minimum": SCIENTIFIC_MINIMUM,
            "per_trial_scientific_minimum": PER_TRIAL_SCIENTIFIC_MINIMUM,
            "strict_scientific_pass": (strict_scientific_pass if score_comparable else None),
            "strict_scientific_pass_observed_in_completed_trials": strict_scientific_pass,
            "terminal_attempts": len(terminal_attempts),
            "expected_attempts_for_configured_run": expected_attempt_count,
        },
        "rates": {
            "function_runnable": (runnable / RUNNABLE_DENOMINATOR if score_comparable else None),
            "published_runner_function_runnable": (
                published_runner_runnable / RUNNABLE_DENOMINATOR if score_comparable else None
            ),
            "task_success": (
                scientific_pass / SCIENTIFIC_DENOMINATOR if score_comparable else None
            ),
            "strict_task_success": (
                strict_scientific_pass / SCIENTIFIC_DENOMINATOR if score_comparable else None
            ),
        },
        "hard_gates": hard_gates,
        "external_hard_gates": {
            "status": "not_evaluated_by_this_mattools_lane",
            "includes": [
                "deterministic_domain_suite",
                "critical_invariants",
                "silent_success_rate",
                "full_materials_release_validation",
            ],
        },
        "promotion": {
            "scope": "MatTools benchmark lane only",
            "passed": lane_passed,
            "full_materials_production_ready": False,
            "reasons": reasons,
        },
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    promotion = report["promotion"]
    status = "PASS" if promotion["passed"] else "BLOCKED"
    benchmark = report["benchmark"]
    runtime = report["runtime_environment"]
    evaluator = report["official_evaluator_environment"]
    lines = [
        "# MatTools promotion-gate report",
        "",
        f"MatTools lane status: **{status}**.",
        "",
        "> This report covers the MatTools benchmark lane only. It never asserts full materials "
        "production readiness; deterministic, critical-invariant, silent-success, and release "
        "validation gates are evaluated elsewhere.",
        "",
        "## Provenance",
        "",
        f"- Benchmark: `{benchmark['name']}` at `{benchmark['revision']}`",
        f"- Benchmark manifest SHA-256: `{benchmark['sha256']}`",
        f"- Source: {benchmark['repository_url']}",
        f"- Dataset card/DOI: {benchmark['dataset_url']} / `{benchmark['dataset_doi']}`",
        "- Licenses recorded: repository Apache-2.0; dataset card CC-BY-NC-4.0",
        f"- Ultra runtime image (generation only): `{runtime.get('image_digest')}`",
        "- Operator-declared Ultra model/provider: "
        f"`{runtime.get('operator_declared_model_id')}` / "
        f"`{runtime.get('operator_declared_provider_id')}`",
        "- Runtime-observed model/provider IDs: "
        f"`{runtime.get('observed_model_ids')}` / `{runtime.get('observed_provider_ids')}`",
        f"- Pinned evaluator source revision: `{evaluator['source_revision']}`",
        f"- Upstream scientific pins: `{json.dumps(evaluator['required_packages'], sort_keys=True)}`",
        f"- Evaluator environment kind: `{evaluator.get('environment_kind')}`",
        f"- Upstream-published image artifact claimed: `{evaluator.get('official_artifact')}`",
        "",
        "The production runtime and reviewed evaluator reconstruction are intentionally separate. A score is "
        "not comparable if the evaluator package pins differ, or if the evaluator image is the "
        "production image.",
        "",
        "## Scores",
        "",
        "| Trial | Runnable | FRR | Published scientific pass | Published TSR | Strict shadow TSR | Replays agree |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for trial in report["trials"]:
        runnable = trial["runnable"]
        scientific = trial["scientific_pass"]
        frr = trial["function_runnable_rate"]
        tsr = trial["task_success_rate"]
        strict_tsr = trial["strict_task_success_rate"]
        lines.append(
            "| {trial} | {runnable} | {frr} | {scientific} | {tsr} | {strict_tsr} | {repro} |".format(
                trial=trial["trial"],
                runnable=f"{runnable}/49" if runnable is not None else "not scored",
                frr=f"{frr:.4f}" if frr is not None else "—",
                scientific=f"{scientific}/138" if scientific is not None else "not scored",
                tsr=f"{tsr:.4f}" if tsr is not None else "—",
                strict_tsr=f"{strict_tsr:.4f}" if strict_tsr is not None else "—",
                repro="yes" if trial["reproducible"] else "no",
            )
        )
    counts = report["counts"]
    rates = report["rates"]
    lines.extend(
        [
            "",
            "Aggregate promotion denominator: 147 parent functions and 414 scientific "
            "subtasks. Passing requires at least 118 runnable parents and 249 accepted subtasks.",
            "",
            (
                f"Comparable aggregate FRR: `{rates['function_runnable']:.4f}` "
                f"({counts['runnable']}/147)."
                if rates["function_runnable"] is not None
                else "Comparable aggregate FRR: **not available** (campaign is incomplete or non-comparable)."
            ),
            (
                f"Comparable published aggregate TSR: `{rates['task_success']:.4f}` "
                f"({counts['scientific_pass']}/414)."
                if rates["task_success"] is not None
                else "Comparable published aggregate TSR: **not available** (campaign is incomplete or non-comparable)."
            ),
            (
                f"Comparable strict-shadow aggregate TSR: `{rates['strict_task_success']:.4f}` "
                f"({counts['strict_scientific_pass']}/414)."
                if rates["strict_task_success"] is not None
                else "Comparable strict-shadow aggregate TSR: **not available** (campaign is incomplete or non-comparable)."
            ),
            "",
            "## Hard gates",
            "",
            "| Gate | Result |",
            "| --- | --- |",
        ]
    )
    for name, passed in report["hard_gates"].items():
        lines.append(f"| `{name}` | {'pass' if passed else 'BLOCKED'} |")
    if promotion["reasons"]:
        lines.extend(
            ["", "Blocked by: " + ", ".join(f"`{item}`" for item in promotion["reasons"]) + "."]
        )
    lines.append("")
    return "\n".join(lines)


def diagnostic_evaluation_completed(report: dict[str, Any]) -> bool:
    """Validate one full diagnostic trial without borrowing three-trial gates."""

    trials = report.get("trials")
    hard = report.get("hard_gates")
    if not isinstance(trials, list) or not trials or not isinstance(hard, dict):
        return False
    trial = trials[0]
    if not isinstance(trial, dict):
        return False
    evaluator = trial.get("evaluator_environment")
    attestation = trial.get("sandbox_policy_attestation")
    shared_requirements = (
        "official_snapshot",
        "license_attested",
        "checkpoint_evidence_integrity",
        "actual_ultra_control_plane_path",
        "actual_model_provider_provenance",
        "production_execute_tool_evidence",
        "production_execute_runtime_image_attestation",
        "required_solution_artifacts",
        "expected_values_and_verifiers_isolated",
        "provenance_complete",
    )
    return (
        trial.get("status") == "complete"
        and trial.get("reproducible") is True
        and isinstance(evaluator, dict)
        and evaluator.get("comparable") is True
        and evaluator.get("independent_from_production_runtime") is True
        and isinstance(attestation, dict)
        and attestation.get("valid") is True
        and all(hard.get(name) is True for name in shared_requirements)
    )


def report_manifest_payload(
    output_dir: Path,
    snapshot: BenchmarkSnapshot,
    report: dict[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Build the deterministic manifest expected for a report/checkpoint bundle."""

    root = output_dir.expanduser().resolve()
    expected_checkpoint = root / "state.json"
    resolved_checkpoint = checkpoint_path.expanduser().resolve()
    if resolved_checkpoint != expected_checkpoint or not resolved_checkpoint.is_file():
        raise GateError("report checkpoint must be the campaign root state.json")
    results_json = pretty_json_bytes(report)
    results_markdown = render_markdown_report(report).encode("utf-8")
    return {
        "schema_version": REPORT_MANIFEST_SCHEMA_VERSION,
        "manifest_kind": REPORT_MANIFEST_KIND,
        "generated_at": report.get("generated_at"),
        "campaign_id": report.get("campaign_id"),
        "benchmark_sha256": snapshot.manifest_sha256,
        "checkpoint_evidence_audit_sha256": sha256_bytes(
            canonical_json_bytes(report.get("checkpoint_evidence_audit"))
        ),
        "regeneration": {
            "helper": "revalidate_report_bundle",
            "cli_subcommand": "verify-report",
            "comparison": "byte_exact",
            "task_execution_performed": False,
        },
        "results_json": {
            "path": str(root / "results.json"),
            "sha256": sha256_bytes(results_json),
        },
        "results_markdown": {
            "path": str(root / "results.md"),
            "sha256": sha256_bytes(results_markdown),
        },
        "checkpoint": {
            "path": str(resolved_checkpoint),
            "sha256": sha256_file(resolved_checkpoint),
        },
    }


def _failed_report_revalidation(issues: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": REPORT_REVALIDATION_SCHEMA_VERSION,
        "revalidation_kind": REPORT_REVALIDATION_KIND,
        "valid": False,
        "bundle_exact": False,
        "manifest_integrity_valid": False,
        "checkpoint_evidence_valid": False,
        "checkpoint_exact": False,
        "results_json_exact": False,
        "results_markdown_exact": False,
        "manifest_exact": False,
        "task_execution_performed": False,
        "promotion_passed": False,
        "issues": list(issues),
    }


def revalidate_report_bundle(
    snapshot: BenchmarkSnapshot,
    report_manifest_path: Path,
) -> dict[str, Any]:
    """Read-only exact regeneration of a MatTools report bundle.

    This helper never submits or executes candidate code. It verifies manifest
    hashes, reparses and revalidates sealed checkpoint evidence through ``build_report``,
    regenerates JSON/Markdown, and exact-compares a fresh manifest payload.
    """

    manifest_path = report_manifest_path.expanduser().resolve()
    root = manifest_path.parent
    if manifest_path.name != "report_manifest.json" or not manifest_path.is_file():
        return _failed_report_revalidation(
            ["report manifest must be an existing campaign-root report_manifest.json"]
        )
    try:
        manifest = read_json_file_strict(manifest_path, label="MatTools report manifest")
    except GateError as exc:
        return _failed_report_revalidation([str(exc)])
    if not isinstance(manifest, dict):
        return _failed_report_revalidation(["MatTools report manifest must be a JSON object"])

    issues: list[str] = []
    schema_valid = (
        manifest.get("schema_version") == REPORT_MANIFEST_SCHEMA_VERSION
        and manifest.get("manifest_kind") == REPORT_MANIFEST_KIND
        and manifest.get("benchmark_sha256") == snapshot.manifest_sha256
        and manifest.get("regeneration")
        == {
            "helper": "revalidate_report_bundle",
            "cli_subcommand": "verify-report",
            "comparison": "byte_exact",
            "task_execution_performed": False,
        }
    )
    if not schema_valid:
        issues.append("report manifest schema, benchmark, or regeneration contract mismatch")

    expected_paths = {
        "results_json": root / "results.json",
        "results_markdown": root / "results.md",
        "checkpoint": root / "state.json",
    }
    record_hashes_valid = True
    for key, expected_path in expected_paths.items():
        record = manifest.get(key)
        if not isinstance(record, dict):
            issues.append(f"report manifest {key} record is missing")
            record_hashes_valid = False
            continue
        declared_path = Path(str(record.get("path") or "")).expanduser().resolve()
        declared_sha = str(record.get("sha256") or "")
        if declared_path != expected_path.resolve():
            issues.append(f"report manifest {key} path is not the exact campaign artifact")
            record_hashes_valid = False
            continue
        if not declared_path.is_file() or not SHA256_HEX_RE.fullmatch(declared_sha):
            issues.append(f"report manifest {key} path/hash is invalid")
            record_hashes_valid = False
            continue
        if sha256_file(declared_path) != declared_sha:
            issues.append(f"report manifest {key} SHA-256 mismatch")
            record_hashes_valid = False

    checkpoint_path = expected_paths["checkpoint"]
    try:
        state = read_json_file_strict(checkpoint_path, label="MatTools checkpoint")
    except GateError as exc:
        return {
            **_failed_report_revalidation([*issues, str(exc)]),
            "manifest_integrity_valid": schema_valid and record_hashes_valid,
        }
    if not isinstance(state, dict):
        return {
            **_failed_report_revalidation([*issues, "MatTools checkpoint must be an object"]),
            "manifest_integrity_valid": schema_valid and record_hashes_valid,
        }
    checkpoint_exact = checkpoint_path.read_bytes() == pretty_json_bytes(state)
    if not checkpoint_exact:
        issues.append("state.json is not in the exact canonical checkpoint encoding")

    try:
        regenerated_report = build_report(snapshot, state, campaign_root=root)
        regenerated_manifest = report_manifest_payload(
            root,
            snapshot,
            regenerated_report,
            checkpoint_path,
        )
    except (GateError, TypeError, ValueError, KeyError, AttributeError) as exc:
        return {
            **_failed_report_revalidation([*issues, f"report regeneration failed ({exc})"]),
            "manifest_integrity_valid": schema_valid and record_hashes_valid,
        }

    expected_json = pretty_json_bytes(regenerated_report)
    expected_markdown = render_markdown_report(regenerated_report).encode("utf-8")
    expected_manifest_bytes = pretty_json_bytes(regenerated_manifest)
    results_json_exact = (
        expected_paths["results_json"].is_file()
        and expected_paths["results_json"].read_bytes() == expected_json
    )
    results_markdown_exact = (
        expected_paths["results_markdown"].is_file()
        and expected_paths["results_markdown"].read_bytes() == expected_markdown
    )
    manifest_exact = (
        manifest == regenerated_manifest and manifest_path.read_bytes() == expected_manifest_bytes
    )
    if not results_json_exact:
        issues.append("results.json is not the exact checkpoint regeneration")
    if not results_markdown_exact:
        issues.append("results.md is not the exact checkpoint regeneration")
    if not manifest_exact:
        issues.append("report_manifest.json is not the exact regenerated manifest")

    manifest_integrity_valid = schema_valid and record_hashes_valid
    bundle_exact = all(
        (
            manifest_integrity_valid,
            checkpoint_exact,
            results_json_exact,
            results_markdown_exact,
            manifest_exact,
        )
    )
    audit = regenerated_report.get("checkpoint_evidence_audit")
    checkpoint_evidence_valid = isinstance(audit, dict) and audit.get("valid") is True
    if not checkpoint_evidence_valid:
        issues.append("checkpoint evidence audit is not valid")
    promotion_passed = regenerated_report.get("promotion", {}).get("passed") is True
    return {
        "schema_version": REPORT_REVALIDATION_SCHEMA_VERSION,
        "revalidation_kind": REPORT_REVALIDATION_KIND,
        "valid": bundle_exact and checkpoint_evidence_valid,
        "bundle_exact": bundle_exact,
        "manifest_integrity_valid": manifest_integrity_valid,
        "checkpoint_evidence_valid": checkpoint_evidence_valid,
        "checkpoint_exact": checkpoint_exact,
        "results_json_exact": results_json_exact,
        "results_markdown_exact": results_markdown_exact,
        "manifest_exact": manifest_exact,
        "task_execution_performed": False,
        "promotion_passed": promotion_passed,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "regenerated_results_json_sha256": sha256_bytes(expected_json),
        "regenerated_results_markdown_sha256": sha256_bytes(expected_markdown),
        "regenerated_manifest_sha256": sha256_bytes(expected_manifest_bytes),
        "report_manifest_sha256": sha256_file(manifest_path),
        "checkpoint_evidence_audit": audit,
        "issues": sorted(set(issues)),
    }


def write_reports(
    output_dir: Path, snapshot: BenchmarkSnapshot, checkpoint: CampaignCheckpoint
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if checkpoint.path.expanduser().resolve() != output_dir / "state.json":
        raise GateError("report checkpoint must be output_dir/state.json")
    persisted_state = read_json_file_strict(checkpoint.path, label="MatTools checkpoint")
    if persisted_state != checkpoint.snapshot():
        raise GateError("in-memory checkpoint differs from persisted state.json")
    report = build_report(
        snapshot,
        persisted_state,
        campaign_root=output_dir,
    )
    json_path = output_dir / "results.json"
    markdown_path = output_dir / "results.md"
    atomic_write_bytes(json_path, pretty_json_bytes(report))
    atomic_write_bytes(markdown_path, render_markdown_report(report).encode("utf-8"))
    manifest = report_manifest_payload(output_dir, snapshot, report, checkpoint.path)
    atomic_write_json(output_dir / "report_manifest.json", manifest)
    return report


def repository_provenance(repository_root: Path) -> dict[str, Any]:
    commit = _git_revision(repository_root)
    status = _run_capture(("git", "status", "--porcelain"), cwd=repository_root)
    dirty = status.returncode != 0 or bool(status.stdout.strip())
    skills_root = repository_root / "backend" / "deepagents_runtime" / "skills"
    skill_files = sorted(path for path in skills_root.rglob("*") if path.is_file())
    skill_hashes = {
        str(path.relative_to(repository_root)): sha256_file(path) for path in skill_files
    }
    return {
        "commit": commit,
        "dirty": dirty,
        "skills_sha256": _manifest_hash(skill_hashes),
        "skills_file_count": len(skill_hashes),
    }


def _selected_tasks(
    snapshot: BenchmarkSnapshot, task_limit: int | None
) -> tuple[BenchmarkTask, ...]:
    if task_limit is None:
        return snapshot.tasks
    if task_limit < 1 or task_limit > len(snapshot.tasks):
        raise GateError(f"--task-limit must be between 1 and {len(snapshot.tasks)}")
    return snapshot.tasks[:task_limit]


def _campaign_config(
    args: argparse.Namespace,
    *,
    snapshot: BenchmarkSnapshot,
    selected_tasks: Sequence[BenchmarkTask],
    auth_env_names: list[str],
) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[1]
    campaign_id = args.campaign_id or (
        f"mattools-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{snapshot.manifest_sha256[:10]}"
    )
    runtime_digest = _normalize_sha256(args.runtime_image_digest)
    if runtime_digest is None:
        raise GateError("--runtime-image-digest must be an immutable sha256 digest")
    return {
        "campaign_id": campaign_id,
        "trial_count": args.trials,
        "selected_ordinals": [task.ordinal for task in selected_tasks],
        "evaluation_mode": args.evaluation_mode,
        "validator_replays": args.validator_replays,
        "diagnostic_subset": len(selected_tasks) != PARENT_TASKS_PER_TRIAL or args.trials != 3,
        "control_plane_base_url": validate_base_url(args.base_url),
        "evaluation_profile": MATERIALS_CLEANROOM_PROFILE,
        "model_id": args.model_id,
        "provider_id": args.provider_id,
        "runtime_image_digest": runtime_digest,
        "runtime_pymatgen_version": args.runtime_pymatgen_version,
        "runtime_defects_version": args.runtime_defects_version,
        "reasoning_mode": args.reasoning_mode,
        "seed": None,
        "seed_supported": False,
        "budgets": {
            "max_runtime_seconds": args.max_runtime_seconds,
            "max_tool_calls": args.max_tool_calls,
        },
        "license_attestation": build_license_attestation(args),
        "auth_environment_variable_names": auth_env_names,
        "secrets_persisted": False,
        "harness_sha256": sha256_file(Path(__file__).resolve()),
        "host_validator_requirements_sha256": sha256_file(HOST_VALIDATOR_REQUIREMENTS),
        "host_validator_input_requirements_sha256": sha256_file(HOST_VALIDATOR_REQUIREMENTS_INPUT),
        "safe_parser_sha256": sha256_file(SAFE_PARSER_SCRIPT),
        "runner_wrapper_sha256": sha256_file(RUNNER_WRAPPER_SCRIPT),
        "strict_shadow_sha256": sha256_file(STRICT_SHADOW_SCRIPT),
        "semantic_repairs_sha256": sha256_file(SEMANTIC_REPAIRS_SCRIPT),
        "ultra": repository_provenance(repository_root),
    }


def _add_snapshot_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--benchmark-root",
        required=True,
        type=Path,
        help="Pinned official MatTools repository checkout; benchmark data is not vendored.",
    )


def license_attestation_valid(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("accepted") is not True:
        return False
    if value.get("repository_license") != "Apache-2.0":
        return False
    if value.get("dataset_card_license") != "CC-BY-NC-4.0":
        return False
    purpose = str(value.get("use_purpose") or "").strip()
    normalized_purpose = re.sub(r"[^a-z0-9]+", " ", purpose.lower()).strip()
    if len(purpose) < 12 or normalized_purpose in {
        "test",
        "unit test",
        "unknown",
        "placeholder",
        "todo",
    }:
        return False
    basis = value.get("use_basis")
    if basis == "noncommercial":
        return value.get("separate_license_evidence_sha256") in {None, ""}
    if basis == "separately_licensed":
        return _normalize_sha256(value.get("separate_license_evidence_sha256")) is not None
    return False


def build_license_attestation(args: argparse.Namespace) -> dict[str, Any]:
    evidence_sha = _normalize_sha256(args.benchmark_license_evidence_sha256)
    value = {
        "accepted": args.accept_benchmark_license,
        "use_basis": args.benchmark_license_basis,
        "use_purpose": str(args.benchmark_use_purpose or "").strip(),
        "repository_license": "Apache-2.0",
        "dataset_card_license": "CC-BY-NC-4.0",
        "separate_license_evidence_sha256": evidence_sha,
        "attested_at": utc_now(),
    }
    if not license_attestation_valid(value):
        raise GateError(
            "benchmark license declaration is incomplete: choose noncommercial use or bind a "
            "separately licensed basis to an immutable evidence SHA-256 and state a concrete purpose"
        )
    return value


def _add_campaign_arguments(parser: argparse.ArgumentParser) -> None:
    _add_snapshot_argument(parser)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--campaign-id")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--provider-id", required=True)
    parser.add_argument("--runtime-image-digest", required=True)
    parser.add_argument("--runtime-pymatgen-version", required=True)
    parser.add_argument("--runtime-defects-version", required=True)
    parser.add_argument("--reasoning-mode", default="deep")
    parser.add_argument("--max-runtime-seconds", type=int, default=1200)
    parser.add_argument("--max-tool-calls", type=int, default=80)
    parser.add_argument("--trials", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument(
        "--task-limit",
        type=int,
        help="Submit the first N tasks for diagnostics. Subsets are never scored as comparable.",
    )
    parser.add_argument("--concurrency", type=int, default=1, choices=range(1, 9))
    parser.add_argument("--http-timeout", type=float, default=30.0)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--poll-timeout", type=float, default=1800.0)
    parser.add_argument("--authorization-env", default="ULTRA_LIVE_TRACE_AUTHORIZATION")
    parser.add_argument("--cookie-env", default="ULTRA_LIVE_TRACE_COOKIE")
    parser.add_argument("--user-id-env", default="ULTRA_LIVE_TRACE_USER_ID")
    parser.add_argument("--org-id-env", default="ULTRA_LIVE_TRACE_ORG_ID")
    parser.add_argument("--role-env", default="ULTRA_LIVE_TRACE_ROLE")
    parser.add_argument("--accept-benchmark-license", action="store_true")
    parser.add_argument(
        "--benchmark-license-basis",
        required=True,
        choices=("noncommercial", "separately_licensed"),
        help="Operator-declared legal basis for using the CC-BY-NC-4.0 benchmark dataset.",
    )
    parser.add_argument(
        "--benchmark-license-evidence-sha256",
        help=(
            "Required immutable evidence hash when "
            "--benchmark-license-basis=separately_licensed; reports use canonical "
            "64-character lowercase hex without a sha256: prefix."
        ),
    )
    parser.add_argument(
        "--benchmark-use-purpose",
        required=True,
        help="Recorded operator declaration of the licensed use purpose.",
    )


def _add_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--validator-replays", type=int, default=2, choices=range(2, 5))
    parser.add_argument("--evaluator-timeout", type=float, default=7200.0)
    parser.add_argument("--expected-evaluator-image-id")
    parser.add_argument("--evaluator-environment-lock", type=Path)
    parser.add_argument("--sandbox-policy-attestation", type=Path)
    parser.add_argument("--sandbox-attestation-signature", type=Path)
    parser.add_argument("--sandbox-attestation-public-key", type=Path)


def resolve_evaluation_mode(args: argparse.Namespace) -> str:
    """Classify execution without allowing a diagnostic to masquerade as promotion."""

    if args.submit_only and args.diagnostic_evaluate:
        raise GateError("--submit-only and --diagnostic-evaluate are mutually exclusive")
    full_task_set = args.task_limit in {None, PARENT_TASKS_PER_TRIAL}
    if args.submit_only:
        return "submission_only"
    if args.diagnostic_evaluate:
        if args.trials != 1 or not full_task_set:
            raise GateError(
                "--diagnostic-evaluate requires one complete 49-task trial; subsets remain "
                "submission-only"
            )
        return "diagnostic_full_trial"
    if args.trials != PROMOTION_TRIALS or not full_task_set:
        raise GateError(
            "automatic promotion evaluation requires exactly three complete 49-task trials; "
            "use --submit-only for a subset or --diagnostic-evaluate for one full diagnostic trial"
        )
    return "promotion"


def pinned_validator_command() -> tuple[str, ...]:
    uv_path = shutil.which("uv")
    if uv_path is None:
        raise GateError("uv is required for the pinned host validator environment")
    for required_path in (
        HOST_VALIDATOR_REQUIREMENTS_INPUT,
        HOST_VALIDATOR_REQUIREMENTS,
        SAFE_PARSER_SCRIPT,
        RUNNER_WRAPPER_SCRIPT,
        STRICT_SHADOW_SCRIPT,
        SEMANTIC_REPAIRS_SCRIPT,
    ):
        if not required_path.is_file():
            raise GateError(f"host validator requirements are missing: {required_path}")
    return (
        uv_path,
        "run",
        "--isolated",
        "--no-project",
        "--python",
        HOST_VALIDATOR_PYTHON_VERSION,
        "--with-requirements",
        str(HOST_VALIDATOR_REQUIREMENTS),
        "python",
    )


def inspect_host_validator_environment(
    snapshot_src: Path,
    validator_command: Sequence[str],
) -> dict[str, Any]:
    """Prove the exact no-task host parser can import before any Ultra submission."""

    probe_script = """
import json
import hashlib
import platform
import sys
from importlib.metadata import distributions, version

import docker_sandbox  # noqa: F401

packages = {
    distribution.metadata.get("Name", "").lower(): distribution.version
    for distribution in distributions()
    if distribution.metadata.get("Name")
}
required = {name: version(name) for name in __REQUIRED_PACKAGES__}
python_digest = hashlib.sha256()
with open(sys.executable, "rb") as executable:
    for chunk in iter(lambda: executable.read(1024 * 1024), b""):
        python_digest.update(chunk)
print(json.dumps({
    "schema_version": "1",
    "python_version": platform.python_version(),
    "python_implementation": platform.python_implementation(),
    "python_executable_sha256": python_digest.hexdigest(),
    "platform": platform.platform(),
    "task_execution_performed": False,
    "required_packages": required,
    "resolved_packages": dict(sorted(packages.items())),
}, sort_keys=True))
""".replace(
        "__REQUIRED_PACKAGES__",
        repr(tuple(HOST_VALIDATOR_REQUIRED_VERSIONS)),
        1,
    )
    process = _run_capture(
        (*validator_command, "-c", probe_script),
        cwd=snapshot_src,
        timeout=600,
    )
    if process.returncode != 0:
        raise GateError(
            "pinned MatTools host validator cannot import the unmodified runner/parser: "
            f"exit {process.returncode}"
        )
    try:
        payload = json.loads(process.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError, TypeError) as exc:
        raise GateError("could not parse MatTools host-validator environment probe") from exc
    if not isinstance(payload, dict) or payload.get("task_execution_performed") is not False:
        raise GateError("MatTools host-validator probe returned an invalid no-task record")
    if payload.get("python_version") != HOST_VALIDATOR_PYTHON_VERSION:
        raise GateError(
            "MatTools host-validator Python version differs from the reviewed interpreter: "
            f"{payload.get('python_version')!r}"
        )
    python_sha = str(payload.get("python_executable_sha256") or "")
    if not SHA256_HEX_RE.fullmatch(python_sha):
        raise GateError("MatTools host-validator probe returned an invalid Python binary hash")
    required = payload.get("required_packages")
    resolved = payload.get("resolved_packages")
    if not isinstance(required, dict) or not isinstance(resolved, dict):
        raise GateError("MatTools host-validator probe returned an invalid package map")
    mismatches = {
        name: {"expected": expected, "observed": required.get(name)}
        for name, expected in HOST_VALIDATOR_REQUIRED_VERSIONS.items()
        if required.get(name) != expected
    }
    if mismatches:
        raise GateError(
            "MatTools host-validator package versions differ from the reviewed lock: "
            + json.dumps(mismatches, sort_keys=True)
        )
    safe_preflight = _run_capture(
        (
            *validator_command,
            str(RUNNER_WRAPPER_SCRIPT),
            "--snapshot-src",
            str(snapshot_src),
            "--expected-runner-sha256",
            OFFICIAL_RUNNER_SHA256,
            "--expected-utils-sha256",
            OFFICIAL_UNSAFE_UTILS_SHA256,
            "--preflight",
        ),
        cwd=snapshot_src,
        timeout=600,
    )
    if safe_preflight.returncode != 0:
        raise GateError("MatTools safe-parser runner preflight failed")
    try:
        safe_payload = json.loads(safe_preflight.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError, TypeError) as exc:
        raise GateError("could not parse MatTools safe-parser preflight") from exc
    if safe_payload != {
        "official_runner_sha256": OFFICIAL_RUNNER_SHA256,
        "safe_parser_bound": True,
        "snapshot_utils_imported": False,
        "task_execution_performed": False,
    }:
        raise GateError("MatTools safe-parser preflight returned an invalid binding record")
    normalized_resolved = {
        str(name).lower(): str(value) for name, value in sorted(resolved.items())
    }
    return {
        "schema_version": "1",
        "python_version": HOST_VALIDATOR_PYTHON_VERSION,
        "python_implementation": payload.get("python_implementation"),
        "python_executable_sha256": python_sha,
        "platform": payload.get("platform"),
        "task_execution_performed": False,
        "required_packages": dict(sorted(required.items())),
        "resolved_packages": normalized_resolved,
        "resolved_packages_sha256": sha256_bytes(canonical_json_bytes(normalized_resolved)),
        "requirements_input_path": str(HOST_VALIDATOR_REQUIREMENTS_INPUT),
        "requirements_input_sha256": sha256_file(HOST_VALIDATOR_REQUIREMENTS_INPUT),
        "requirements_lock_path": str(HOST_VALIDATOR_REQUIREMENTS),
        "requirements_lock_sha256": sha256_file(HOST_VALIDATOR_REQUIREMENTS),
        "safe_parser_path": str(SAFE_PARSER_SCRIPT),
        "safe_parser_sha256": sha256_file(SAFE_PARSER_SCRIPT),
        "runner_wrapper_path": str(RUNNER_WRAPPER_SCRIPT),
        "runner_wrapper_sha256": sha256_file(RUNNER_WRAPPER_SCRIPT),
        "safe_parser_preflight": safe_payload,
        "validator_command": list(validator_command),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser(
        "inspect", help="Verify a pinned snapshot without running it."
    )
    _add_snapshot_argument(inspect_parser)

    run_parser = subparsers.add_parser(
        "run", help="Submit through Ultra and, when complete, evaluate."
    )
    _add_campaign_arguments(run_parser)
    _add_evaluation_arguments(run_parser)
    run_parser.add_argument("--submit-only", action="store_true")
    run_parser.add_argument(
        "--diagnostic-evaluate",
        action="store_true",
        help=(
            "Evaluate one complete 49-task trial with the exact pinned scorer and security "
            "requirements. The report remains explicitly non-comparable and cannot promote."
        ),
    )

    report_parser = subparsers.add_parser("report", help="Regenerate reports from a checkpoint.")
    _add_snapshot_argument(report_parser)
    report_parser.add_argument("--output-dir", required=True, type=Path)
    verify_parser = subparsers.add_parser(
        "verify-report",
        help="Read-only exact regeneration and verification of a report bundle.",
    )
    _add_snapshot_argument(verify_parser)
    verify_parser.add_argument("--report-manifest", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    snapshot: BenchmarkSnapshot | None = None
    checkpoint: CampaignCheckpoint | None = None
    output_dir: Path | None = None
    try:
        snapshot = load_benchmark_snapshot(args.benchmark_root)
        if args.command == "inspect":
            print(json.dumps(snapshot.provenance_record(), indent=2, sort_keys=True))
            return 0
        if args.command == "verify-report":
            verification = revalidate_report_bundle(snapshot, args.report_manifest)
            print(json.dumps(verification, indent=2, sort_keys=True))
            return 0 if verification["valid"] else 2

        output_dir = args.output_dir.expanduser().resolve()
        state_path = output_dir / "state.json"
        if args.command == "report":
            if not state_path.is_file():
                raise GateError(f"checkpoint does not exist: {state_path}")
            state = read_json_file_strict(state_path, label="MatTools checkpoint")
            if not isinstance(state, dict):
                raise GateError("MatTools checkpoint must be a JSON object")
            checkpoint = CampaignCheckpoint(state_path, state)
            report = write_reports(output_dir, snapshot, checkpoint)
            print(json.dumps(report["promotion"], indent=2, sort_keys=True))
            return 0 if report["promotion"]["passed"] else 2

        if not args.accept_benchmark_license:
            raise GateError(
                "running requires --accept-benchmark-license after reviewing Apache-2.0 and "
                "CC-BY-NC-4.0 applicability"
            )
        args.evaluation_mode = resolve_evaluation_mode(args)
        evaluator_environment_lock: dict[str, Any] | None = None
        validator_command: tuple[str, ...] | None = None
        host_validator_environment: dict[str, Any] | None = None
        if args.evaluation_mode != "submission_only":
            if _normalize_sha256(args.expected_evaluator_image_id) is None:
                raise GateError(
                    "scientific evaluation requires an explicit immutable "
                    "--expected-evaluator-image-id=sha256:... before submission"
                )
            evaluator_environment_lock = load_approved_evaluator_environment_lock(
                args.evaluator_environment_lock
            )
            validator_command = pinned_validator_command()
            host_validator_environment = inspect_host_validator_environment(
                snapshot.src_root,
                validator_command,
            )
        selected_tasks = _selected_tasks(snapshot, args.task_limit)
        headers, auth_env_names = auth_headers_from_environment(args)
        if state_path.is_file() and not args.campaign_id:
            existing_state = read_json_file_strict(state_path, label="MatTools checkpoint")
            if not isinstance(existing_state, dict):
                raise GateError("MatTools checkpoint must be a JSON object")
            existing_campaign_id = str(existing_state.get("campaign_id") or "").strip()
            if not existing_campaign_id:
                raise GateError("existing checkpoint has no campaign_id")
            args.campaign_id = existing_campaign_id
        config = _campaign_config(
            args,
            snapshot=snapshot,
            selected_tasks=selected_tasks,
            auth_env_names=auth_env_names,
        )
        config["evaluator_environment_lock"] = evaluator_environment_lock
        config["expected_evaluator_image_id"] = _normalize_sha256(args.expected_evaluator_image_id)
        config["host_validator_environment"] = host_validator_environment
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = CampaignCheckpoint.open_or_create(
            state_path,
            snapshot=snapshot,
            config=config,
        )
        submit_campaign(
            snapshot=snapshot,
            checkpoint=checkpoint,
            output_dir=output_dir,
            selected_tasks=selected_tasks,
            trial_count=args.trials,
            headers=headers,
            concurrency=args.concurrency,
            http_timeout=args.http_timeout,
            poll_interval=args.poll_interval,
            poll_timeout=args.poll_timeout,
        )
        if args.evaluation_mode != "submission_only":
            if validator_command is None:
                raise GateError("host validator preflight was not completed")
            evaluate_campaign(
                snapshot=snapshot,
                checkpoint=checkpoint,
                output_dir=output_dir,
                validator_command=validator_command,
                replay_count=args.validator_replays,
                evaluator_timeout=args.evaluator_timeout,
                expected_image_id=args.expected_evaluator_image_id,
                evaluator_environment_lock=evaluator_environment_lock,
                sandbox_attestation_path=args.sandbox_policy_attestation,
                sandbox_attestation_signature_path=args.sandbox_attestation_signature,
                sandbox_attestation_public_key_path=args.sandbox_attestation_public_key,
            )
        report = write_reports(output_dir, snapshot, checkpoint)
        print(json.dumps(report["promotion"], indent=2, sort_keys=True))
        if args.evaluation_mode == "submission_only":
            return 0
        if args.evaluation_mode == "diagnostic_full_trial":
            return 0 if diagnostic_evaluation_completed(report) else 2
        return 0 if report["promotion"]["passed"] else 2
    except (GateError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        if checkpoint is not None and snapshot is not None and output_dir is not None:
            try:
                checkpoint.update_config(
                    {"last_harness_error": str(exc), "last_harness_error_at": utc_now()}
                )
                write_reports(output_dir, snapshot, checkpoint)
            except Exception:
                # Preserve the original failure and never risk printing secrets
                # from a secondary reporting exception.
                pass
        print(f"MatTools promotion gate blocked: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
