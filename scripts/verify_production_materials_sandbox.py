#!/usr/bin/env python3
"""Verify deterministic materials support through Ultra's real Docker sandbox.

Two deliberately different scopes are supported:

``ci-pinned-materials``
    Runs through :class:`DockerSandboxBackend` in the small, pinned materials
    gate image.  This is an enforceable PR/release source contract, not a claim
    about the much larger production image.

``production-full``
    Runs the same contract through the exact production sandbox image after it
    is built during deployment.  This is the full-image parity claim.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

SCHEMA_VERSION = 1
EVIDENCE_BUNDLE_SCHEMA_VERSION = 1
SCOPES = {"ci-pinned-materials", "production-full"}
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
IMAGE_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
SAFE_IMAGE_REF_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,255}$")
EXPECTED_IMAGE_TITLES = {
    "ci-pinned-materials": "Ultra deterministic materials domain gate",
    "production-full": "Ultra Deep Agents scientific sandbox",
}
DOMAIN_REPORT = Path("domain/materials-domain-gate.json")
CALPHAD_EXPERIMENTAL_REPORT = Path("domain/calphad-experimental-benchmark.json")
CALPHAD_REPORT = Path("calphad-embedded-probe.json")
CALPHAD_RUNTIME_JUNIT = Path("calphad-runtime-junit.xml")
CALPHAD_TOOLS_JUNIT = Path("calphad-tools-junit.xml")
REQUIRED_DOMAIN_INVARIANT_COUNT = 13
REQUIRED_CALPHAD_RUNTIME_TEST_COUNT = 39
REQUIRED_CALPHAD_CORE_TEST_COUNT = 36
REQUIRED_TYPED_CALPHAD_CLI_TEST_COUNT = 3
REQUIRED_CALPHAD_TOOLS_TEST_COUNT = 56
REQUIRED_CALPHAD_MANIFEST_SHA256 = (
    "a5dfb3aac68f119a8fe0ee751255a16f31fe8e8515cfa9739320ba21aa28fb09"
)
REQUIRED_CALPHAD_RELEASE_INPUT_SHA256S = {
    "backend/deepagents_runtime/materials_data/calphad/experimental_benchmark_manifest.json": (
        "741e8a78b568155a877301c618e88bbda41d719d647502c11e1c917df292519c"
    ),
    "backend/deepagents_runtime/materials_data/calphad/manifest.json": (
        REQUIRED_CALPHAD_MANIFEST_SHA256
    ),
    "backend/deepagents_runtime/src/ultra_deepagents/context_tools.py": (
        "98cb9a21d9f135e0f7968f54dcd9608c936e6ea552f061d46d684a7038358aa3"
    ),
    "backend/deepagents_runtime/src/ultra_deepagents/live_trace.py": (
        "1844d54c05af03e753c759c78f560bac287c118297d869c2b919e5ed66a402d0"
    ),
    "backend/deepagents_runtime/src/ultra_deepagents/materials/trace_binding.py": (
        "ac1bfff2b8fd6ae94270f6018c81cd09c4aa92e95d0518729071bd4e53ac0149"
    ),
    "backend/deepagents_runtime/src/ultra_deepagents/materials/calphad.py": (
        "c17c9158457c4aa236fa58865372f59325d479226418f28c1c7e998ce68cdc85"
    ),
    "backend/deepagents_runtime/src/ultra_deepagents/materials/calphad_cli.py": (
        "9a917c3650768ccc89f3dcd1fce048eeee72f4fb28a2ac701e3defb03cfc962c"
    ),
    "backend/deepagents_runtime/src/ultra_deepagents/materials/calphad_tools.py": (
        "5bce33c39548e566bdcbdd6f67367451e9aa391a310bcb19c221f1e7c78e9c4b"
    ),
    "backend/deepagents_runtime/tests/test_calphad_cli.py": (
        "a6b2c1cdbb5ad7de3f1336bac86e43b588d855db5a63a1cc36c99478a5b9ad0f"
    ),
    "backend/deepagents_runtime/tests/test_calphad_runtime.py": (
        "b9a3ef8b6729ff9cd3a50a0469aeeca0015fef01bd82631f47e7dc8ae488f268"
    ),
    "backend/deepagents_runtime/tests/test_calphad_tools.py": (
        "edc972afd420424a6a9253dd284424f86f5e5a6c2fc357de93035bbba887b257"
    ),
    "backend/deepagents_runtime/tests/test_materials_live_trace.py": (
        "fd474a548df96460868ee84d7e9a5657d1b9e0e037c79184f9a70aca2b05b585"
    ),
    "scripts/calphad_cross_language_gate.py": (
        "18439683fcef25a6cecfb727fe0d26a3f6a7625764d11b5e48fa729d809e8528"
    ),
    "scripts/calphad_experimental_benchmark.py": (
        "248ac0f741afa4ff76224dda8e8dc77d29f3472afaee518efa0b43cdcdf34f78"
    ),
    "tests/fixtures/materials/calphad_experimental_benchmark_expected.json": (
        "b39a5d50df6ff89201f5421381a6754446ea17366cfa6a9aaba974e27e46d58b"
    ),
    "tests/test_calphad_cross_language_gate.py": (
        "dc143d199b0959256d39de1512a6dadf4c417c027eab95c45f1d2ddfcf66d276"
    ),
    "tests/test_calphad_experimental_benchmark.py": (
        "b934ab54c49f1a0c8483d8436c60d1abcdf464cea2a27fb2c134b597a4fac405"
    ),
}
REQUIRED_CALPHAD_TOOL_TEST_NAMES = (
    "test_selected_resource_inspection_uses_fixed_command_and_content_addressed_evidence",
    "test_selected_dat_resource_preserves_descriptor_format_through_evidence",
    "test_selected_descriptor_rejects_db_even_with_tdb_mime_type",
    "test_selected_descriptor_database_format_must_match_original_name",
    "test_selected_descriptor_requires_server_database_format",
    "test_selected_descriptor_and_staged_source_suffix_must_match",
    "test_catalog_validation_callback_is_run_anchored_and_response_verified",
    "test_catalog_validation_callback_persists_failure_tuple_as_nonpromotable",
    "test_governance_persistence_failure_fails_closed_after_artifact_creation",
    "test_embedded_calphad_result_uses_release_registry_not_tenant_ledger",
    "test_resource_missing_complete_owner_provenance_never_executes",
    "test_resource_missing_owner_pressure_scope_never_executes",
    "test_resource_catalog_byte_mismatch_never_executes",
    "test_nonselected_readable_catalog_resource_is_rejected",
    "test_backend_timeout_and_nonfinite_input_fail_closed",
    "test_equilibrium_summary_exposes_v2_mu_and_phase_compositions",
    "test_equilibrium_rejects_pressure_outside_owner_scope_before_execution",
    "test_wrong_tool_or_runtime_evidence_schema_is_rejected[wrong_schema]",
    "test_wrong_tool_or_runtime_evidence_schema_is_rejected[wrong_runtime_schema]",
    "test_result_manifest_format_mismatch_is_rejected_by_host_verifier",
    "test_v2_schema_and_residuals_are_validated_across_full_artifact[missing_residual]",
    "test_v2_schema_and_residuals_are_validated_across_full_artifact[invalid_hidden_point]",
    "test_host_rejects_artifact_binding_or_request_forgery[wrong_binding]",
    "test_host_rejects_artifact_binding_or_request_forgery[wrong_request]",
    "test_host_rejects_artifact_hash_mismatch",
    "test_content_addressed_request_symlink_collision_fails_closed",
    "test_isolated_cli_bootstrap_cannot_be_shadowed_from_workspace",
    "test_cli_requires_fixed_or_bounded_resource_pressure_scope",
    "test_cli_requires_exact_resource_format_and_path_suffix",
    "test_cli_rejects_result_manifest_format_mismatch[dat-.dat-does not match the resource binding]",
    "test_cli_rejects_result_manifest_format_mismatch[tdb-.dat-path suffix does not match]",
    "test_cli_retains_exact_bounded_parse_failure_without_raw_diagnostics[CalphadInputError-expected_outcome0]",
    "test_cli_retains_exact_bounded_parse_failure_without_raw_diagnostics[CalphadTimeoutError-expected_outcome1]",
    "test_cli_retains_exact_bounded_parse_failure_without_raw_diagnostics[CalphadUnsupportedError-expected_outcome2]",
    "test_host_verifier_rejects_individually_valid_but_mismatched_failure_tuple",
    "test_equilibrium_unsupported_is_pre_solver_and_retained_with_solver_started_false",
    "test_cli_inspection_chain_rejects_wrong_hash_binding_and_inventory",
    "test_cli_rejects_resource_byte_mismatch_and_symlink_path",
    "test_scheil_typed_tool_retains_va_and_returns_mass_closed_bounded_summary",
    "test_scheil_host_rejects_complete_hash_bound_but_scientifically_invalid_artifact[scheil_missing_phase_component]",
    "test_scheil_host_rejects_complete_hash_bound_but_scientifically_invalid_artifact[scheil_mass_closure_forgery]",
    "test_scheil_invalid_scientific_conditions_fail_before_sandbox_execution[overrides0-invalid_typed_input]",
    "test_scheil_invalid_scientific_conditions_fail_before_sandbox_execution[overrides1-invalid_typed_input]",
    "test_scheil_invalid_scientific_conditions_fail_before_sandbox_execution[overrides2-invalid_typed_input]",
    "test_scheil_invalid_scientific_conditions_fail_before_sandbox_execution[overrides3-invalid_typed_input]",
    "test_cli_scheil_uses_fixed_kernel_limits_and_retains_inspection_lineage",
    "test_equilibrium_catalog_callback_requires_exact_inspection_lineage_response[True]",
    "test_equilibrium_catalog_callback_requires_exact_inspection_lineage_response[False]",
    "test_shared_calphad_resource_returns_nonpromoting_read_only_artifact",
    "test_agent_registers_typed_calphad_tools_in_manifest_and_code_runner",
    "test_typed_calphad_backend_has_immutable_nonextensible_outer_cap[0]",
    "test_typed_calphad_backend_has_immutable_nonextensible_outer_cap[21600]",
    "test_typed_calphad_backend_has_immutable_nonextensible_outer_cap[45]",
    "test_typed_calphad_backend_rejects_unbounded_resources[0-4g-256]",
    "test_typed_calphad_backend_rejects_unbounded_resources[2--256]",
    "test_typed_calphad_backend_rejects_unbounded_resources[2-4g-0]",
)
REQUIRED_CALPHAD_CORE_TEST_NAMES = (
    "test_parser_uses_the_validated_database_format",
    "test_database_input_rejects_unregistered_db_suffix",
    "test_pinned_pycalphad_database_corpus_parses_all_registered_text_formats",
    "test_dat_inspection_records_the_actual_parser_format",
    "test_inspection_emits_bounded_database_and_phase_model_manifest",
    "test_catalog_inspection_discovers_components_and_phases_without_solver_request",
    "test_embedded_nist_manifest_directory_json_and_tdb_are_verified",
    "test_manifest_hash_size_and_path_traversal_fail_closed",
    "test_self_authored_sibling_manifest_cannot_mint_verified_provenance",
    "test_source_license_and_catalog_binding_are_fail_closed",
    "test_assessment_pressure_limits_are_finite_bounded_and_nondecreasing[limits0]",
    "test_assessment_pressure_limits_are_finite_bounded_and_nondecreasing[limits1]",
    "test_assessment_pressure_limits_are_finite_bounded_and_nondecreasing[limits2]",
    "test_assessment_pressure_limits_are_finite_bounded_and_nondecreasing[limits3]",
    "test_package_fixture_and_byte_identical_copy_are_rejected",
    "test_symlink_nonregular_oversize_and_nonfinite_tdb_are_rejected",
    "test_database_domain_count_limits_are_enforced[MAX_DATABASE_ELEMENTS-3-element limit]",
    "test_database_domain_count_limits_are_enforced[MAX_DATABASE_SPECIES-3-species limit]",
    "test_database_domain_count_limits_are_enforced[MAX_DATABASE_PHASES-0-phase limit]",
    "test_database_domain_count_limits_are_enforced[MAX_DATABASE_PARAMETERS-3-parameter limit]",
    "test_equilibrium_returns_phase_compositions_mu_np_gm_and_canonical_evidence",
    "test_equilibrium_rejects_nonfinite_request_values[temperatures_K-nan]",
    "test_equilibrium_rejects_nonfinite_request_values[pressures_Pa-inf]",
    "test_equilibrium_rejects_nonfinite_request_values[total_amount_mol--inf]",
    "test_equilibrium_rejects_nonfinite_request_values[independent_compositions-value3]",
    "test_equilibrium_rejects_nonfinite_request_values[wall_time_seconds-nan]",
    "test_composition_closure_domain_subset_grid_and_temperature_bounds",
    "test_result_size_and_wall_time_are_hard_limits",
    "test_database_parse_has_an_independent_wall_time_limit",
    "test_nonfinite_solver_output_is_rejected",
    "test_nonfinite_thermodynamic_output_is_rejected[MU-non-finite MU]",
    "test_nonfinite_thermodynamic_output_is_rejected[X-non-finite X]",
    "test_solver_component_coordinate_must_match_request",
    "test_solver_phase_vertices_must_reconstruct_requested_bulk_composition",
    "test_solver_chemical_potentials_must_satisfy_gibbs_euler_relation",
    "test_off_main_thread_execution_fails_closed",
)
REQUIRED_TYPED_CALPHAD_CLI_TEST_NAMES = (
    "test_typed_cli_real_pycalphad_0_11_2_inspection_and_equilibrium",
    "test_typed_cli_real_dat_resource_format_binding",
    "test_typed_cli_real_scheil_alcocrni_is_mass_closed_and_retains_va",
)
REQUIRED_CALPHAD_ADVERSARIAL_TEST_NAMES = frozenset(
    {
        "test_parser_uses_the_validated_database_format",
        "test_database_input_rejects_unregistered_db_suffix",
        "test_pinned_pycalphad_database_corpus_parses_all_registered_text_formats",
        "test_dat_inspection_records_the_actual_parser_format",
        "test_embedded_nist_manifest_directory_json_and_tdb_are_verified",
        "test_assessment_pressure_limits_are_finite_bounded_and_nondecreasing[limits0]",
        "test_assessment_pressure_limits_are_finite_bounded_and_nondecreasing[limits1]",
        "test_assessment_pressure_limits_are_finite_bounded_and_nondecreasing[limits2]",
        "test_assessment_pressure_limits_are_finite_bounded_and_nondecreasing[limits3]",
        "test_composition_closure_domain_subset_grid_and_temperature_bounds",
        "test_typed_cli_real_pycalphad_0_11_2_inspection_and_equilibrium",
        "test_typed_cli_real_dat_resource_format_binding",
        "test_typed_cli_real_scheil_alcocrni_is_mass_closed_and_retains_va",
    }
)
REQUIRED_EQUILIBRIUM_SCHEMA_VERSION = "ultra.calphad.equilibrium.v2"
MAX_JUNIT_BYTES = 4 * 1024 * 1024
_DOCKER_SIZE_PATTERN = re.compile(r"^(?P<value>[0-9]+(?:\.[0-9]+)?)(?:[kmgtpe]i?b?|b)?$", re.I)

RELEASE_CRITICAL_FIXED_FILES = (
    Path(".github/workflows/autonomy-gate.yml"),
    Path(".github/workflows/materials-domain-gate.yml"),
    Path(".github/workflows/materials-production-qualification.yml"),
    Path(".github/workflows/release-artifacts.yml"),
    Path("AGENTS.md"),
    Path("Makefile"),
    Path("backend/controlplane/.dockerignore"),
    Path("backend/controlplane/Dockerfile"),
    Path("backend/controlplane/Makefile"),
    Path("backend/controlplane/integration/calphad_cross_language_http_test.go"),
    Path("backend/controlplane/integration/calphad_cross_language_test.go"),
    Path("backend/controlplane/api/openapi.yaml"),
    Path("backend/controlplane/go.mod"),
    Path("backend/controlplane/go.sum"),
    Path("backend/controlplane/internal/domain/calphad.go"),
    Path("backend/controlplane/internal/eventbus/bus.go"),
    Path("backend/controlplane/internal/httpapi/handlers.go"),
    Path("backend/controlplane/internal/httpapi/handlers_calphad.go"),
    Path("backend/controlplane/internal/httpapi/resource_staging_security.go"),
    Path("backend/controlplane/internal/openapi/generated.gen.go"),
    Path("backend/controlplane/internal/runcontrol/service.go"),
    Path("backend/controlplane/internal/store/calphad_ledger.go"),
    Path("backend/controlplane/internal/store/memory.go"),
    Path("backend/controlplane/internal/store/schema.sql"),
    Path("backend/controlplane/internal/store/schema_apply.go"),
    Path("backend/controlplane/internal/store/schema_check.go"),
    Path("backend/controlplane/migrations/000008_calphad_revision_ledger.down.sql"),
    Path("backend/controlplane/migrations/000008_calphad_revision_ledger.up.sql"),
    Path("backend/deepagents_runtime/MATERIALS_PRODUCTION_READINESS.md"),
    Path("backend/deepagents_runtime/MATTOOLS_PROMOTION_GATE.md"),
    Path("backend/deepagents_runtime/.dockerignore"),
    Path("backend/deepagents_runtime/Dockerfile.worker"),
    Path("backend/deepagents_runtime/materials_data/calphad/experimental_benchmark_manifest.json"),
    Path("backend/deepagents_runtime/pyproject.toml"),
    Path("backend/deepagents_runtime/uv.lock"),
    Path("backend/deepagents_runtime/src/ultra_deepagents/agent.py"),
    Path("backend/deepagents_runtime/src/ultra_deepagents/async_delegation.py"),
    Path("backend/deepagents_runtime/src/ultra_deepagents/config.py"),
    Path("backend/deepagents_runtime/src/ultra_deepagents/context.py"),
    Path("backend/deepagents_runtime/src/ultra_deepagents/context_tools.py"),
    Path("backend/deepagents_runtime/src/ultra_deepagents/events.py"),
    Path("backend/deepagents_runtime/src/ultra_deepagents/imaging/hdf5.py"),
    Path("backend/deepagents_runtime/src/ultra_deepagents/live_trace.py"),
    Path("backend/deepagents_runtime/src/ultra_deepagents/materials/__init__.py"),
    Path("backend/deepagents_runtime/src/ultra_deepagents/materials/calphad.py"),
    Path("backend/deepagents_runtime/src/ultra_deepagents/materials/calphad_cli.py"),
    Path("backend/deepagents_runtime/src/ultra_deepagents/materials/calphad_tools.py"),
    Path("backend/deepagents_runtime/src/ultra_deepagents/materials/trace_binding.py"),
    Path("backend/deepagents_runtime/src/ultra_deepagents/materials/validation.py"),
    Path("backend/deepagents_runtime/src/ultra_deepagents/nats_worker.py"),
    Path("backend/deepagents_runtime/src/ultra_deepagents/runner.py"),
    Path("backend/deepagents_runtime/src/ultra_deepagents/schemas.py"),
    Path("backend/deepagents_runtime/src/ultra_deepagents/resources/tools.py"),
    Path("backend/deepagents_runtime/src/ultra_deepagents/code_execution/docker.py"),
    Path("backend/deepagents_runtime/src/ultra_deepagents/code_execution/git_staging.py"),
    Path("backend/deepagents_runtime/src/ultra_deepagents/code_execution/paths.py"),
    Path("backend/deepagents_runtime/src/ultra_deepagents/code_execution/progress.py"),
    Path("backend/deepagents_runtime/tests/test_worker_transport.py"),
    Path("backend/deepagents_runtime/tests/test_bisque_tools.py"),
    Path("deploy/docker/deepagents-sandbox.Dockerfile"),
    Path("deploy/docker/materials-domain-gate.Dockerfile"),
    Path("deploy/docker/materials-requirements.txt"),
    Path("deploy/docker/mattools-evaluator-linux-arm64-lock.json"),
    Path("deploy/docker/mattools-evaluator-supplemental-requirements.txt"),
    Path("deploy/docker/mattools-evaluator.Dockerfile"),
    Path("deploy/docker/mattools-upstream-published-linux-arm64-audit.json"),
    Path("frontend/src/components/viewer/hdf5/Hdf5Overview.tsx"),
    Path("frontend/src/components/viewer/hdf5/PhaseMetadataSummary.test.tsx"),
    Path("frontend/src/components/viewer/hdf5/PhaseMetadataSummary.tsx"),
    Path("frontend/src/components/viewer/hdf5/formatters.test.ts"),
    Path("frontend/src/components/viewer/hdf5/formatters.ts"),
    Path("frontend/src/components/viewer/hdf5/hdf5-viewer.css"),
    Path("frontend/src/App.tsx"),
    Path("frontend/src/lib/api.ts"),
    Path("frontend/src/lib/bisqueMutationIntent.test.ts"),
    Path("frontend/src/lib/bisqueMutationIntent.ts"),
    Path("frontend/src/lib/viewerManifest.test.ts"),
    Path("frontend/src/lib/viewerManifest.ts"),
    Path("frontend/src/types.ts"),
    Path("scripts/build_mattools_evaluator.py"),
    Path("scripts/build_ultra_release_artifact.sh"),
    Path("scripts/calphad_ledger_gate.py"),
    Path("scripts/calphad_cross_language_gate.py"),
    Path("scripts/calphad_experimental_benchmark.py"),
    Path("scripts/deploy_ultra_control_stack.sh"),
    Path("scripts/materials_domain_gate.py"),
    Path("scripts/materials_promotion_envelope.py"),
    Path("scripts/materials_readiness_gate.py"),
    Path("scripts/mattools-validator-requirements.lock.txt"),
    Path("scripts/mattools-validator-requirements.txt"),
    Path("scripts/mattools_promotion_gate.py"),
    Path("scripts/mattools_strict_shadow.py"),
    Path("scripts/run_materials_domain_gate.sh"),
    Path("scripts/verify_production_materials_sandbox.py"),
    Path("tests/test_control_stack_deploy_assets.py"),
    Path("tests/test_calphad_ledger_gate.py"),
    Path("tests/test_calphad_cross_language_gate.py"),
    Path("tests/fixtures/materials/calphad_experimental_benchmark_expected.json"),
    Path("tests/test_calphad_experimental_benchmark.py"),
    Path("tests/test_production_materials_sandbox.py"),
)

RELEASE_CRITICAL_TREES = (
    Path("backend/controlplane/api"),
    Path("backend/controlplane/internal/domain"),
    Path("backend/controlplane/internal/eventbus"),
    Path("backend/controlplane/internal/httpapi"),
    Path("backend/controlplane/internal/openapi"),
    Path("backend/controlplane/internal/runcontrol"),
    Path("backend/controlplane/internal/store"),
    Path("backend/controlplane/migrations"),
    Path("backend/deepagents_runtime/materials_data/calphad"),
    Path("backend/deepagents_runtime/skills/computational-materials"),
    Path("backend/deepagents_runtime/skills/materials-characterization"),
    Path("backend/deepagents_runtime/skills/materials-characterization-advanced"),
    Path("backend/deepagents_runtime/skills/materials-crystal-plasticity"),
    Path("backend/deepagents_runtime/skills/materials-mechanics-degradation"),
    Path("backend/deepagents_runtime/skills/materials-processing-kinetics"),
    Path("backend/deepagents_runtime/skills/materials-sensor-data"),
    Path("backend/deepagents_runtime/skills/materials-structure-thermo"),
    Path("backend/deepagents_runtime/src/ultra_deepagents"),
    Path("backend/deepagents_runtime/src/ultra_deepagents/materials"),
    Path("backend/deepagents_runtime/tests/domain_correctness"),
    Path("backend/deepagents_runtime/tests/kinetics_runtime"),
)

RELEASE_CRITICAL_GLOBS = (
    "backend/deepagents_runtime/tests/test_calphad_*.py",
    "backend/deepagents_runtime/tests/test_crystal_plasticity_*.py",
    "backend/deepagents_runtime/tests/test_degradation_*.py",
    "backend/deepagents_runtime/tests/test_kinetics_*.py",
    "backend/deepagents_runtime/tests/test_materials_*.py",
    "backend/deepagents_runtime/tests/test_materials_natural_prompt_fixtures.py",
    "backend/deepagents_runtime/tests/test_ngff.py",
    "backend/deepagents_runtime/tests/test_paper_*.py",
    "backend/deepagents_runtime/tests/test_runner_paper_preload.py",
    "backend/deepagents_runtime/tests/test_sensor_*.py",
    "backend/deepagents_runtime/tests/test_vision_subagent.py",
    "backend/deepagents_runtime/tests/test_zarr_tree_identity_contract.py",
    "tests/test_materials_*.py",
    "tests/test_mattools_*.py",
)

RUNTIME_SCRATCH_ROOTS = (Path(".cache"), Path(".tmp"))


class VerificationError(RuntimeError):
    """A fail-closed production-parity error."""


class BackendResponse(Protocol):
    output: str
    exit_code: int
    truncated: bool


class SandboxBackend(Protocol):
    config: Any

    def build_docker_command(self, command: str) -> list[str]: ...

    def execute(self, command: str, *, timeout: int | None = None) -> BackendResponse: ...


@dataclass(frozen=True)
class ImageInspection:
    ref: str
    image_id: str
    revision: str
    title: str
    entrypoint: tuple[str, ...]
    labels: Mapping[str, str] = dataclass_field(default_factory=dict)
    os: str = ""
    architecture: str = ""
    raw_inspect: Mapping[str, Any] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class SandboxPolicy:
    network: str = "none"
    cpus: float = 2.0
    memory: str = "8g"
    pids_limit: int = 512
    shm_size: str = "1g"
    timeout_seconds: int = 1200
    output_limit_bytes: int = 8 * 1024 * 1024
    gpus: str = ""
    max_concurrency: int = 1
    no_new_privileges: bool = True
    source: str = "ci_fixed_limits"


BackendFactory = Callable[[Path, Path, str, SandboxPolicy], SandboxBackend]
ImageInspector = Callable[[str], ImageInspection]
HostSuiteRunner = Callable[[Path, Path], Mapping[str, Any]]


def _environment_bool(value: str, *, name: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise VerificationError(f"{name} must be a boolean; got {value!r}")


def _positive_docker_size(value: str, *, name: str) -> str:
    normalized = str(value).strip().lower()
    match = _DOCKER_SIZE_PATTERN.fullmatch(normalized)
    if match is None or float(match.group("value")) <= 0:
        raise VerificationError(f"{name} must be a positive bounded Docker size; got {value!r}")
    return normalized


def _validated_policy(policy: SandboxPolicy) -> SandboxPolicy:
    if policy.network.strip().lower() != "none":
        raise VerificationError(
            f"materials parity requires sandbox network=none; got {policy.network!r}"
        )
    if not math.isfinite(policy.cpus) or policy.cpus <= 0:
        raise VerificationError("materials parity requires a positive finite CPU limit")
    _positive_docker_size(policy.memory, name="sandbox memory")
    _positive_docker_size(policy.shm_size, name="sandbox shared memory")
    if policy.pids_limit <= 0:
        raise VerificationError("materials parity requires a positive PID limit")
    if policy.timeout_seconds <= 0:
        raise VerificationError("materials parity requires a positive wall-clock limit")
    if policy.output_limit_bytes <= 0:
        raise VerificationError("materials parity requires a positive output limit")
    if policy.max_concurrency <= 0:
        raise VerificationError("materials parity requires bounded worker sandbox concurrency")
    if not policy.no_new_privileges:
        raise VerificationError("materials parity requires no-new-privileges")
    return policy


def policy_for_scope(
    args: argparse.Namespace,
    *,
    environ: Mapping[str, str] | None = None,
) -> SandboxPolicy:
    """Use fixed CI limits or the production worker's exported effective policy."""

    if args.scope != "production-full":
        return _validated_policy(
            SandboxPolicy(
                cpus=float(args.cpus),
                memory=str(args.memory),
                pids_limit=int(args.pids_limit),
                shm_size=str(args.shm_size),
                timeout_seconds=int(args.timeout_seconds),
                output_limit_bytes=int(args.output_limit_bytes),
            )
        )

    values = os.environ if environ is None else environ
    worker_image = values.get("ULTRA_DEEPAGENTS_SANDBOX_IMAGE", "bisque-ultra-codeexec:py311")
    raw_network = values.get("ULTRA_DEEPAGENTS_SANDBOX_NETWORK", "none")
    raw_memory = values.get("ULTRA_DEEPAGENTS_SANDBOX_MEMORY", "")
    for name, value in (
        ("ULTRA_DEEPAGENTS_SANDBOX_IMAGE", worker_image),
        ("ULTRA_DEEPAGENTS_SANDBOX_NETWORK", raw_network),
        ("ULTRA_DEEPAGENTS_SANDBOX_MEMORY", raw_memory),
    ):
        if value != value.strip():
            raise VerificationError(f"{name} contains unsafe surrounding whitespace")
    if worker_image != args.image:
        raise VerificationError(
            "production parity image does not match ULTRA_DEEPAGENTS_SANDBOX_IMAGE: "
            f"{args.image!r} != {worker_image!r}"
        )
    try:
        policy = SandboxPolicy(
            network=raw_network,
            cpus=float(values.get("ULTRA_DEEPAGENTS_SANDBOX_CPUS", "0")),
            memory=raw_memory,
            pids_limit=int(values.get("ULTRA_DEEPAGENTS_SANDBOX_PIDS_LIMIT", "0")),
            shm_size=values.get("ULTRA_DEEPAGENTS_SANDBOX_SHM_SIZE", "").strip(),
            timeout_seconds=int(values.get("ULTRA_DEEPAGENTS_SANDBOX_TIMEOUT_SECONDS", "21600")),
            output_limit_bytes=int(values.get("ULTRA_DEEPAGENTS_SANDBOX_OUTPUT_LIMIT_BYTES", "0")),
            gpus=values.get("ULTRA_DEEPAGENTS_SANDBOX_GPUS", "").strip(),
            max_concurrency=int(values.get("ULTRA_DEEPAGENTS_SANDBOX_MAX_CONCURRENCY", "0")),
            no_new_privileges=_environment_bool(
                values.get("ULTRA_DEEPAGENTS_SANDBOX_NO_NEW_PRIVILEGES", "true"),
                name="ULTRA_DEEPAGENTS_SANDBOX_NO_NEW_PRIVILEGES",
            ),
            source="exported_worker_environment",
        )
    except ValueError as exc:
        raise VerificationError(f"invalid exported production sandbox policy: {exc}") from exc
    return _validated_policy(policy)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
        ensure_ascii=False,
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def _safe_relative_path(value: str | Path, *, label: str) -> Path:
    relative = Path(value)
    if (
        not relative.parts
        or relative == Path(".")
        or relative.is_absolute()
        or ".." in relative.parts
    ):
        raise VerificationError(f"{label} must be a safe report-relative path")
    return relative


def _evidence_directory(output_dir: Path, relative: Path) -> Path:
    safe = _safe_relative_path(relative, label="retained evidence directory")
    current = output_dir
    for part in safe.parts:
        current = current / part
        if current.exists() or current.is_symlink():
            try:
                mode = os.lstat(current).st_mode
            except OSError as exc:
                raise VerificationError(f"cannot inspect evidence directory {current}") from exc
            if not stat.S_ISDIR(mode) or current.is_symlink():
                raise VerificationError(
                    f"retained evidence directory contains a non-directory or symlink: {current}"
                )
        else:
            current.mkdir()
    return current


def _require_regular_file(path: Path, *, label: str) -> None:
    try:
        mode = os.lstat(path).st_mode
    except OSError as exc:
        raise VerificationError(f"{label} is missing: {path}") from exc
    if not stat.S_ISREG(mode):
        raise VerificationError(f"{label} must be a regular file: {path}")


def _file_evidence(path: Path, output_dir: Path) -> dict[str, Any]:
    _require_regular_file(path, label="retained evidence")
    try:
        relative = path.relative_to(output_dir)
    except ValueError as exc:
        raise VerificationError("retained evidence path escapes the report directory") from exc
    return {
        "schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "relative_path": relative.as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _retain_bytes(
    payload: bytes,
    output_dir: Path,
    *,
    directory: Path,
    stem: str,
    suffix: str,
) -> dict[str, Any]:
    relative_directory = _safe_relative_path(directory, label="retained evidence directory")
    digest = _sha256_bytes(payload)
    parent = _evidence_directory(output_dir, relative_directory)
    destination = parent / f"{stem}-{digest}{suffix}"
    if destination.exists() or destination.is_symlink():
        _require_regular_file(destination, label="content-addressed evidence collision")
        if destination.read_bytes() != payload:
            raise VerificationError(
                f"content-addressed evidence collision has different bytes: {destination}"
            )
    else:
        destination.write_bytes(payload)
    return _file_evidence(destination, output_dir)


def _retain_file(
    source: Path,
    output_dir: Path,
    *,
    directory: Path,
    stem: str,
    suffix: str = "",
) -> dict[str, Any]:
    _require_regular_file(source, label="evidence source")
    digest = _sha256_file(source)
    parent = _evidence_directory(output_dir, directory)
    destination = parent / f"{stem}-{digest}{suffix}"
    if destination.exists() or destination.is_symlink():
        _require_regular_file(destination, label="content-addressed evidence collision")
        if (
            destination.stat().st_size != source.stat().st_size
            or _sha256_file(destination) != digest
        ):
            raise VerificationError(
                f"content-addressed evidence collision has different bytes: {destination}"
            )
    else:
        shutil.copyfile(source, destination)
    return _file_evidence(destination, output_dir)


def _strict_tree_regular_files(root: Path, *, label: str) -> list[Path]:
    try:
        root_mode = os.lstat(root).st_mode
    except OSError as exc:
        raise VerificationError(f"{label} directory is missing: {root}") from exc
    if not stat.S_ISDIR(root_mode) or root.is_symlink():
        raise VerificationError(f"{label} must be a real directory: {root}")
    files: list[Path] = []
    unsafe: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        try:
            mode = os.lstat(path).st_mode
        except OSError:
            unsafe.append(relative.as_posix())
            continue
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            unsafe.append(relative.as_posix())
            continue
        files.append(relative)
    if unsafe:
        raise VerificationError(f"{label} contains non-regular paths: " + ", ".join(unsafe))
    return files


def _strict_tree_manifest(root: Path, *, label: str) -> dict[str, Any]:
    relatives = _strict_tree_regular_files(root, label=label)
    return _manifest_for_files(root, relatives)


def _retain_tree(
    source: Path,
    output_dir: Path,
    *,
    directory: Path,
    stem: str,
    label: str,
    manifest_path_prefix: Path | None = None,
) -> dict[str, Any]:
    source_manifest = _strict_tree_manifest(source, label=label)
    manifest = source_manifest
    if manifest_path_prefix is not None:
        prefix = _safe_relative_path(manifest_path_prefix, label=f"{label} manifest prefix")
        entries = [
            {**entry, "path": (prefix / str(entry["path"])).as_posix()}
            for entry in source_manifest["files"]
        ]
        manifest = {
            "aggregate_sha256": _sha256_bytes(
                json.dumps(entries, separators=(",", ":"), sort_keys=True).encode()
            ),
            "file_count": len(entries),
            "files": entries,
        }
    parent = _evidence_directory(output_dir, directory)
    destination = parent / f"{stem}-{manifest['aggregate_sha256']}"
    if destination.exists() or destination.is_symlink():
        observed = _strict_tree_manifest(destination, label="content-addressed tree collision")
        if observed != source_manifest:
            raise VerificationError(
                f"content-addressed tree collision has different bytes: {destination}"
            )
    else:
        shutil.copytree(source, destination, symlinks=True)
        observed = _strict_tree_manifest(destination, label="retained evidence tree")
        if observed != source_manifest:
            raise VerificationError("retained evidence tree changed while it was copied")
    try:
        relative = destination.relative_to(output_dir)
    except ValueError as exc:
        raise VerificationError("retained evidence tree escapes the report directory") from exc
    record = {
        "schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "relative_path": relative.as_posix(),
        **manifest,
    }
    if manifest_path_prefix is not None:
        record["manifest_path_prefix"] = manifest_path_prefix.as_posix()
    return record


def _load_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise VerificationError(f"cannot read finite JSON report {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise VerificationError(f"JSON report root must be an object: {path}")
    return payload


def inspect_image(image_ref: str) -> ImageInspection:
    """Inspect a local image without executing it."""

    result = subprocess.run(
        ["docker", "image", "inspect", image_ref],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise VerificationError(
            f"docker image inspect failed for {image_ref!r}: {result.stderr.strip()}"
        )
    try:
        records = json.loads(result.stdout)
        record = records[0]
        config = record["Config"]
    except (json.JSONDecodeError, IndexError, KeyError, TypeError) as exc:
        raise VerificationError(f"malformed docker image inspection for {image_ref!r}") from exc
    if not isinstance(config, Mapping):
        raise VerificationError(f"malformed docker image configuration for {image_ref!r}")
    labels = config.get("Labels") or {}
    entrypoint = config.get("Entrypoint") or []
    if not isinstance(labels, Mapping) or not isinstance(entrypoint, list):
        raise VerificationError(f"malformed docker image metadata for {image_ref!r}")
    return ImageInspection(
        ref=image_ref,
        image_id=str(record.get("Id") or "").lower(),
        revision=str(labels.get("org.opencontainers.image.revision") or "").lower(),
        title=str(labels.get("org.opencontainers.image.title") or ""),
        entrypoint=tuple(str(part) for part in entrypoint),
        labels={str(key): str(value) for key, value in sorted(labels.items())},
        os=str(record.get("Os") or ""),
        architecture=str(record.get("Architecture") or ""),
        raw_inspect=dict(record),
    )


def _image_summary(inspection: ImageInspection) -> dict[str, Any]:
    return {
        "ref": inspection.ref,
        "image_id": inspection.image_id,
        "revision": inspection.revision,
        "title": inspection.title,
        "entrypoint": list(inspection.entrypoint),
        "labels": dict(sorted((str(key), str(value)) for key, value in inspection.labels.items())),
        "os": inspection.os,
        "architecture": inspection.architecture,
    }


def _docker_config_environment(config: Mapping[str, Any]) -> dict[str, str]:
    raw_environment = config.get("Env") or []
    if not isinstance(raw_environment, list):
        raise VerificationError("Docker inspect Config.Env is malformed")
    environment: dict[str, str] = {}
    for raw in raw_environment:
        name, separator, value = str(raw).partition("=")
        if not separator or not name or name in environment:
            raise VerificationError("Docker inspect Config.Env has a duplicate or malformed entry")
        environment[name] = value
    return environment


def _retain_image_inspection(
    inspection: ImageInspection,
    output_dir: Path,
    *,
    role: str,
) -> dict[str, Any]:
    if not inspection.raw_inspect:
        raise VerificationError(f"{role} image is missing raw Docker inspect evidence")
    raw = dict(inspection.raw_inspect)
    config = raw.get("Config")
    if not isinstance(config, Mapping):
        raise VerificationError(f"{role} image raw Docker inspect lacks Config")
    raw_labels = config.get("Labels") or {}
    if not isinstance(raw_labels, Mapping):
        raise VerificationError(f"{role} image raw Docker inspect labels are malformed")
    normalized_labels = {str(key): str(value) for key, value in sorted(raw_labels.items())}
    if str(raw.get("Id") or "").lower() != inspection.image_id:
        raise VerificationError(f"{role} image raw Docker inspect ID disagrees with summary")
    if normalized_labels != dict(inspection.labels):
        raise VerificationError(f"{role} image raw Docker inspect labels disagree with summary")
    if (
        normalized_labels.get("org.opencontainers.image.revision", "").lower()
        != inspection.revision
        or normalized_labels.get("org.opencontainers.image.title", "") != inspection.title
    ):
        raise VerificationError(
            f"{role} image raw Docker inspect OCI identity labels disagree with summary"
        )
    raw_entrypoint = config.get("Entrypoint") or []
    if not isinstance(raw_entrypoint, list) or tuple(str(part) for part in raw_entrypoint) != (
        inspection.entrypoint
    ):
        raise VerificationError(
            f"{role} image raw Docker inspect entrypoint disagrees with summary"
        )
    if str(raw.get("Os") or "") != inspection.os:
        raise VerificationError(f"{role} image raw Docker inspect OS disagrees with summary")
    if str(raw.get("Architecture") or "") != inspection.architecture:
        raise VerificationError(
            f"{role} image raw Docker inspect architecture disagrees with summary"
        )
    record = _retain_bytes(
        _canonical_json_bytes(raw, newline=True),
        output_dir,
        directory=Path("bundle/image"),
        stem=f"docker-inspect-{role}",
        suffix=".json",
    )
    return {
        "schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "summary": _image_summary(inspection),
        "docker_inspect": record,
    }


def validate_image(
    inspection: ImageInspection,
    *,
    expected_git_sha: str,
    scope: str,
    allow_entrypoint: bool,
) -> list[str]:
    failures: list[str] = []
    if IMAGE_ID_PATTERN.fullmatch(inspection.image_id) is None:
        failures.append(
            f"image is not pinned by an immutable configuration ID: {inspection.image_id!r}"
        )
    if inspection.revision != expected_git_sha:
        failures.append(
            "image OCI revision does not match expected Git SHA: "
            f"{inspection.revision or '<missing>'} != {expected_git_sha}"
        )
    expected_title = EXPECTED_IMAGE_TITLES[scope]
    if inspection.title != expected_title:
        failures.append(
            f"wrong image title for {scope}: {inspection.title!r} != {expected_title!r}"
        )
    if inspection.entrypoint and not allow_entrypoint:
        failures.append(
            "image entrypoint would intercept DockerSandboxBackend's bash command: "
            + repr(list(inspection.entrypoint))
        )
    if scope == "production-full":
        config = (
            inspection.raw_inspect.get("Config")
            if isinstance(inspection.raw_inspect.get("Config"), Mapping)
            else {}
        )
        try:
            environment = _docker_config_environment(config)
        except VerificationError as exc:
            failures.append(str(exc))
        else:
            if environment.get("PYTHONPATH") != "/opt/ultra-runtime":
                failures.append(
                    "production image PYTHONPATH must select the baked /opt/ultra-runtime"
                )
            if environment.get("PYTHONHOME"):
                failures.append("production image must not override PYTHONHOME")
    return failures


def load_required_domain_validator_ids(repo_root: Path) -> tuple[str, ...]:
    """Read the readiness gate's canonical validator tuple without executing it."""

    path = repo_root / "scripts" / "materials_readiness_gate.py"
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise VerificationError(f"cannot parse readiness validator contract {path}: {exc}") from exc
    assignments: list[ast.expr] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(
            isinstance(target, ast.Name) and target.id == "REQUIRED_DOMAIN_VALIDATORS"
            for target in targets
        ):
            assignments.append(node.value)
    if len(assignments) != 1:
        raise VerificationError(
            "readiness REQUIRED_DOMAIN_VALIDATORS must have exactly one assignment"
        )
    try:
        raw: Any = ast.literal_eval(assignments[0])
    except (ValueError, TypeError) as exc:
        raise VerificationError(
            "readiness REQUIRED_DOMAIN_VALIDATORS must be a literal sequence"
        ) from exc
    if not isinstance(raw, (tuple, list)):
        raise VerificationError("readiness REQUIRED_DOMAIN_VALIDATORS is missing")
    validators = tuple(str(value) for value in raw)
    if len(validators) != REQUIRED_DOMAIN_INVARIANT_COUNT:
        raise VerificationError(
            "readiness validator contract must contain exactly "
            f"{REQUIRED_DOMAIN_INVARIANT_COUNT} IDs; found {len(validators)}"
        )
    if len(set(validators)) != len(validators):
        raise VerificationError("readiness validator contract contains duplicate IDs")
    if any(
        re.fullmatch(r"materials\.[a-z0-9_.-]+\.v[0-9]+", value) is None for value in validators
    ):
        raise VerificationError("readiness validator contract contains an invalid validator ID")
    return validators


def load_staged_matplotlibrc(staged_root: Path) -> str:
    path = staged_root / "backend/deepagents_runtime/src/ultra_deepagents/code_execution/docker.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise VerificationError(
            f"cannot parse staged Docker sandbox contract {path}: {exc}"
        ) from exc
    assignments = [
        node.value
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == "MATPLOTLIBRC"
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
    ]
    if len(assignments) != 1:
        raise VerificationError("staged Docker MATPLOTLIBRC must have one literal assignment")
    try:
        value = ast.literal_eval(assignments[0])
    except (TypeError, ValueError) as exc:
        raise VerificationError("staged Docker MATPLOTLIBRC must be a literal string") from exc
    if not isinstance(value, str) or not value:
        raise VerificationError("staged Docker MATPLOTLIBRC must be a non-empty string")
    return value


def _is_ignored_release_path(path: Path) -> bool:
    return (
        "__pycache__" in path.parts or path.name == ".DS_Store" or path.suffix in {".pyc", ".pyo"}
    )


def _require_regular_files(repo_root: Path, relatives: Sequence[Path], *, label: str) -> list[Path]:
    unique = sorted(set(relatives), key=lambda path: path.as_posix())
    unsafe: list[str] = []
    for relative in unique:
        try:
            mode = os.lstat(repo_root / relative).st_mode
        except OSError:
            unsafe.append(relative.as_posix())
            continue
        if not stat.S_ISREG(mode):
            unsafe.append(relative.as_posix())
    if unsafe:
        raise VerificationError(f"{label} is missing or non-regular: " + ", ".join(unsafe))
    return unique


def _manifest_for_files(repo_root: Path, relatives: Sequence[Path]) -> dict[str, Any]:
    entries = [
        {
            "path": relative.as_posix(),
            "sha256": _sha256_file(repo_root / relative),
            "size_bytes": (repo_root / relative).stat().st_size,
        }
        for relative in relatives
    ]
    canonical = json.dumps(entries, separators=(",", ":"), sort_keys=True).encode()
    return {
        "aggregate_sha256": _sha256_bytes(canonical),
        "file_count": len(entries),
        "files": entries,
    }


def _manifest_for_entries(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized = [
        {
            "path": str(entry.get("path") or ""),
            "sha256": str(entry.get("sha256") or ""),
            "size_bytes": int(entry.get("size_bytes", -1)),
        }
        for entry in entries
    ]
    normalized.sort(key=lambda entry: entry["path"])
    canonical = json.dumps(normalized, separators=(",", ":"), sort_keys=True).encode()
    return {
        "aggregate_sha256": _sha256_bytes(canonical),
        "file_count": len(normalized),
        "files": normalized,
    }


def validate_declared_file_manifest(
    manifest: Mapping[str, Any],
    *,
    label: str,
) -> list[str]:
    failures: list[str] = []
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        return [f"{label} file list is malformed"]
    normalized: list[dict[str, Any]] = []
    paths: list[str] = []
    try:
        for raw in raw_files:
            if not isinstance(raw, Mapping):
                raise VerificationError(f"{label} contains a malformed file row")
            record = {
                "path": str(raw.get("path") or ""),
                "sha256": str(raw.get("sha256") or ""),
                "size_bytes": int(raw.get("size_bytes", -1)),
            }
            _safe_relative_path(record["path"], label=f"{label} file path")
            if (
                dict(raw) != record
                or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
                or record["size_bytes"] < 0
            ):
                raise VerificationError(f"{label} contains an invalid file identity")
            normalized.append(record)
            paths.append(record["path"])
        if len(set(paths)) != len(paths) or paths != sorted(paths):
            failures.append(f"{label} paths are duplicate or noncanonical")
        if _manifest_for_entries(normalized) != {
            "aggregate_sha256": manifest.get("aggregate_sha256"),
            "file_count": manifest.get("file_count"),
            "files": raw_files,
        }:
            failures.append(f"{label} aggregate or file count is inconsistent")
    except (TypeError, ValueError, VerificationError) as exc:
        failures.append(str(exc))
    return failures


def _bind_copied_source_to_release(
    copied_source: Mapping[str, Any],
    required_materials: Mapping[str, Any],
) -> dict[str, Any]:
    copied_files = copied_source.get("files")
    release_files = required_materials.get("files")
    if not isinstance(copied_files, list) or not isinstance(release_files, list):
        raise VerificationError("staged or release source manifest is malformed")
    release_by_path = {
        str(record.get("path") or ""): record
        for record in release_files
        if isinstance(record, Mapping)
    }
    mismatches: list[str] = []
    for copied in copied_files:
        if not isinstance(copied, Mapping):
            mismatches.append("<malformed>")
            continue
        path = str(copied.get("path") or "")
        declared = release_by_path.get(path)
        if declared is None or dict(copied) != {
            "path": path,
            "sha256": declared.get("sha256"),
            "size_bytes": declared.get("size_bytes"),
        }:
            mismatches.append(path or "<blank>")
    if mismatches:
        raise VerificationError(
            "staged source differs from the verified release manifest subset: "
            + ", ".join(sorted(mismatches))
        )
    return {
        "schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "status": "exact_release_manifest_subset",
        "file_count": len(copied_files),
        "release_aggregate_sha256": required_materials.get("aggregate_sha256"),
    }


def _post_execution_workspace_evidence(
    workspace: Path,
    staged_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    observed = _strict_tree_manifest(workspace, label="post-execution sandbox workspace")
    baseline_files = staged_manifest.get("files")
    if not isinstance(baseline_files, list):
        raise VerificationError("staged-source baseline manifest is malformed")
    baseline_paths = {
        str(record.get("path") or "") for record in baseline_files if isinstance(record, Mapping)
    }

    comparable: list[Mapping[str, Any]] = []
    runtime_scratch: list[Mapping[str, Any]] = []
    for record in observed["files"]:
        path = Path(str(record["path"]))
        is_runtime_scratch = any(
            path == root or root in path.parents for root in RUNTIME_SCRATCH_ROOTS
        )
        if str(record["path"]) not in baseline_paths and is_runtime_scratch:
            runtime_scratch.append(record)
        else:
            comparable.append(record)
    return _manifest_for_entries(comparable), _manifest_for_entries(runtime_scratch)


def _retained_path(output_dir: Path, raw: Any, *, label: str) -> Path:
    relative = _safe_relative_path(str(raw or ""), label=label)
    current = output_dir
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except OSError as exc:
            raise VerificationError(f"{label} is missing from the report directory") from exc
        if stat.S_ISLNK(mode):
            raise VerificationError(f"{label} contains a symlinked report-relative component")
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(mode):
            raise VerificationError(f"{label} contains a non-directory parent component")
    try:
        current.resolve(strict=True).relative_to(output_dir.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise VerificationError(f"{label} escapes or is missing from the report directory") from exc
    return current


def validate_retained_file_record(
    output_dir: Path,
    record: Mapping[str, Any],
    *,
    label: str,
) -> list[str]:
    failures: list[str] = []
    try:
        if int(record.get("schema_version", -1)) != EVIDENCE_BUNDLE_SCHEMA_VERSION:
            raise VerificationError(f"{label} has the wrong evidence schema version")
        path = _retained_path(output_dir, record.get("relative_path"), label=label)
        _require_regular_file(path, label=label)
        expected_size = int(record.get("size_bytes", -1))
        expected_hash = str(record.get("sha256") or "")
        if expected_size < 0 or path.stat().st_size != expected_size:
            failures.append(f"{label} retained size does not match its record")
        if re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
            failures.append(f"{label} retained SHA-256 is malformed")
        elif _sha256_file(path) != expected_hash:
            failures.append(f"{label} retained SHA-256 does not match its bytes")
        if expected_hash and expected_hash not in path.name:
            failures.append(f"{label} retained filename is not content-addressed")
    except (OSError, TypeError, ValueError, VerificationError) as exc:
        failures.append(str(exc))
    return failures


def validate_retained_tree_record(
    output_dir: Path,
    record: Mapping[str, Any],
    *,
    label: str,
) -> list[str]:
    failures: list[str] = []
    try:
        if int(record.get("schema_version", -1)) != EVIDENCE_BUNDLE_SCHEMA_VERSION:
            raise VerificationError(f"{label} has the wrong evidence schema version")
        root = _retained_path(output_dir, record.get("relative_path"), label=label)
        expected = {
            "aggregate_sha256": str(record.get("aggregate_sha256") or ""),
            "file_count": int(record.get("file_count", -1)),
            "files": record.get("files"),
        }
        if not isinstance(expected["files"], list):
            raise VerificationError(f"{label} retained file manifest is malformed")
        observed = _strict_tree_manifest(root, label=label)
        raw_prefix = record.get("manifest_path_prefix")
        if raw_prefix is not None:
            prefix = _safe_relative_path(str(raw_prefix), label=f"{label} manifest prefix")
            entries = [
                {**entry, "path": (prefix / str(entry["path"])).as_posix()}
                for entry in observed["files"]
            ]
            observed = {
                "aggregate_sha256": _sha256_bytes(
                    json.dumps(entries, separators=(",", ":"), sort_keys=True).encode()
                ),
                "file_count": len(entries),
                "files": entries,
            }
        if observed != expected:
            failures.append(f"{label} retained tree closure or aggregate does not match its record")
        if expected["aggregate_sha256"] not in root.name:
            failures.append(f"{label} retained directory is not content-addressed")
    except (OSError, TypeError, ValueError, VerificationError) as exc:
        failures.append(str(exc))
    return failures


def _tree_regular_files(repo_root: Path, relative_root: Path, *, label: str) -> list[Path]:
    absolute_root = repo_root / relative_root
    try:
        root_mode = os.lstat(absolute_root).st_mode
    except OSError as exc:
        raise VerificationError(f"{label} directory is missing: {relative_root}") from exc
    if not stat.S_ISDIR(root_mode) or absolute_root.is_symlink():
        raise VerificationError(f"{label} must be a real directory: {relative_root}")
    files: list[Path] = []
    unsafe: list[str] = []
    for path in sorted(absolute_root.rglob("*")):
        relative = path.relative_to(repo_root)
        if _is_ignored_release_path(relative):
            continue
        try:
            mode = os.lstat(path).st_mode
        except OSError:
            unsafe.append(relative.as_posix())
            continue
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            unsafe.append(relative.as_posix())
            continue
        files.append(relative)
    if unsafe:
        raise VerificationError(f"{label} contains non-regular paths: " + ", ".join(unsafe))
    return files


def _release_critical_source_files(repo_root: Path) -> list[Path]:
    """Return the complete source/evidence set that can affect materials promotion."""

    discovered = list(RELEASE_CRITICAL_FIXED_FILES)
    empty_trees: list[str] = []
    for relative_root in RELEASE_CRITICAL_TREES:
        tree_files = _tree_regular_files(
            repo_root,
            relative_root,
            label="release-critical materials source",
        )
        if not tree_files:
            empty_trees.append(relative_root.as_posix())
        discovered.extend(tree_files)
    for pattern in RELEASE_CRITICAL_GLOBS:
        matches = [
            path.relative_to(repo_root)
            for path in sorted(repo_root.glob(pattern))
            if not _is_ignored_release_path(path.relative_to(repo_root))
            and path.is_file()
            and not path.is_symlink()
        ]
        if not matches:
            empty_trees.append(pattern)
        discovered.extend(matches)
    if empty_trees:
        raise VerificationError(
            "release-critical materials source groups are empty: " + ", ".join(empty_trees)
        )
    return _require_regular_files(
        repo_root,
        discovered,
        label="release-critical materials source",
    )


def build_required_source_manifest(repo_root: Path) -> dict[str, Any]:
    """Hash all production-critical materials/CALPHAD/MatTools release inputs."""

    mismatches: list[str] = []
    for raw_path, expected_sha256 in sorted(REQUIRED_CALPHAD_RELEASE_INPUT_SHA256S.items()):
        relative = Path(raw_path)
        path = repo_root / relative
        try:
            _require_regular_file(path, label="release-critical CALPHAD input")
            observed_sha256 = _sha256_file(path)
        except VerificationError:
            mismatches.append(raw_path)
            continue
        if observed_sha256 != expected_sha256:
            mismatches.append(raw_path)
    if mismatches:
        raise VerificationError(
            "release-critical CALPHAD input hash drift: " + ", ".join(mismatches)
        )
    return _manifest_for_files(repo_root, _release_critical_source_files(repo_root))


def build_release_artifact_identities(repo_root: Path) -> dict[str, Any]:
    """Hash the exact deployable control binary and every frontend-dist file."""

    control_relative = Path("bin/ultra-control")
    control_path = repo_root / control_relative
    _require_regular_files(repo_root, (control_relative,), label="release control binary")
    if control_path.stat().st_size <= 0:
        raise VerificationError("release control binary is empty")

    frontend_root = repo_root / "frontend" / "dist"
    try:
        frontend_mode = os.lstat(frontend_root).st_mode
    except OSError as exc:
        raise VerificationError("release frontend/dist is missing") from exc
    if not stat.S_ISDIR(frontend_mode) or frontend_root.is_symlink():
        raise VerificationError("release frontend/dist must be a real directory")
    frontend_files = _tree_regular_files(
        repo_root,
        Path("frontend/dist"),
        label="release frontend",
    )
    frontend_files = _require_regular_files(
        repo_root,
        frontend_files,
        label="release frontend files",
    )
    index_relative = Path("frontend/dist/index.html")
    if index_relative not in frontend_files or (repo_root / index_relative).stat().st_size <= 0:
        raise VerificationError("release frontend identity requires a non-empty index.html")
    frontend_manifest = _manifest_for_files(repo_root, frontend_files)
    return {
        "control_binary": {
            "path": control_relative.as_posix(),
            "sha256": _sha256_file(control_path),
            "size_bytes": control_path.stat().st_size,
        },
        "frontend_dist": {
            "path": "frontend/dist",
            **frontend_manifest,
        },
    }


def _verify_source_revision(repo_root: Path, expected_git_sha: str) -> dict[str, Any]:
    """Require a clean checkout or content-hashed git-archive release."""

    git_marker = repo_root / ".git"
    if git_marker.exists():
        git_probe = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--is-inside-work-tree"],
            check=False,
            capture_output=True,
            text=True,
        )
        if git_probe.returncode != 0 or git_probe.stdout.strip() != "true":
            raise VerificationError(f"invalid Git checkout marker at {git_marker}")
        head = (
            subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .lower()
        )
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignore-submodules=none",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if head != expected_git_sha:
            raise VerificationError(f"checked-out Git SHA {head} != expected {expected_git_sha}")
        if status.returncode != 0:
            raise VerificationError(f"git status failed: {status.stderr.strip()}")
        if status.stdout:
            raise VerificationError(
                "source is dirty (tracked, staged, or untracked); parity requires a clean Git SHA"
            )
        required_materials = build_required_source_manifest(repo_root)
        return {
            "kind": "clean_git_checkout",
            "expected_git_sha": expected_git_sha,
            "observed_git_sha": head,
            "required_materials": required_materials,
            "tracked_worktree_clean": True,
            "staged_index_clean": True,
            "untracked_files_clean": True,
        }

    manifest_path = repo_root / "release-manifest.json"
    manifest = _load_json(manifest_path)
    observed = str(manifest.get("release_sha") or "").lower()
    if observed != expected_git_sha:
        raise VerificationError(
            f"release manifest SHA {observed or '<missing>'} != expected {expected_git_sha}"
        )
    source = manifest.get("source")
    required_materials = source.get("required_materials") if isinstance(source, Mapping) else None
    expected_materials = build_required_source_manifest(repo_root)
    if not isinstance(required_materials, Mapping):
        raise VerificationError(
            "release manifest source.required_materials content-hash contract is missing"
        )
    if dict(required_materials) != expected_materials:
        raise VerificationError(
            "release required-materials content hashes do not match the extracted source"
        )
    targets = manifest.get("targets")
    if not isinstance(targets, Mapping):
        raise VerificationError("release manifest deployable target identities are missing")
    expected_artifacts = build_release_artifact_identities(repo_root)
    declared_artifacts = {
        "control_binary": targets.get("control_binary_identity"),
        "frontend_dist": targets.get("frontend_dist_identity"),
    }
    if targets.get("control_binary") != "bin/ultra-control":
        raise VerificationError("release manifest control binary path is not canonical")
    if targets.get("frontend_dist") != "frontend/dist":
        raise VerificationError("release manifest frontend path is not canonical")
    if declared_artifacts != expected_artifacts:
        raise VerificationError(
            "release control binary or frontend content identities do not match extracted bytes"
        )
    return {
        "kind": "git_archive_release_manifest",
        "expected_git_sha": expected_git_sha,
        "observed_git_sha": observed,
        "manifest_path": "release-manifest.json",
        "manifest_sha256": _sha256_file(manifest_path),
        "required_materials": expected_materials,
        "release_artifacts": expected_artifacts,
        "tracked_worktree_clean": True,
        "staged_index_clean": True,
        "untracked_files_clean": True,
    }


def _retain_release_bundle(
    repo_root: Path,
    output_dir: Path,
    source_evidence: Mapping[str, Any],
    *,
    scope: str,
) -> dict[str, Any]:
    if scope != "production-full":
        return {
            "schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
            "promotable": False,
            "reason": "ci-pinned source contract is not an extracted production release",
            "manifest": None,
            "control_binary": None,
            "frontend_dist": None,
        }
    if source_evidence.get("kind") != "git_archive_release_manifest":
        raise VerificationError(
            "production-full parity requires an extracted immutable release manifest, not Git"
        )
    artifacts = source_evidence.get("release_artifacts")
    if not isinstance(artifacts, Mapping):
        raise VerificationError("production release artifact identities are missing")
    manifest_record = _retain_file(
        repo_root / "release-manifest.json",
        output_dir,
        directory=Path("bundle/release"),
        stem="release-manifest",
        suffix=".json",
    )
    control_record = _retain_file(
        repo_root / "bin/ultra-control",
        output_dir,
        directory=Path("bundle/release"),
        stem="ultra-control",
    )
    frontend_record = _retain_tree(
        repo_root / "frontend/dist",
        output_dir,
        directory=Path("bundle/release"),
        stem="frontend-dist",
        label="release frontend distribution",
        manifest_path_prefix=Path("frontend/dist"),
    )
    declared_control = artifacts.get("control_binary")
    declared_frontend = artifacts.get("frontend_dist")
    if not isinstance(declared_control, Mapping) or not isinstance(declared_frontend, Mapping):
        raise VerificationError("production release artifact identity records are malformed")
    if control_record["sha256"] != declared_control.get("sha256") or control_record[
        "size_bytes"
    ] != declared_control.get("size_bytes"):
        raise VerificationError("retained control binary differs from the release identity")
    if {key: frontend_record[key] for key in ("aggregate_sha256", "file_count", "files")} != {
        key: declared_frontend.get(key) for key in ("aggregate_sha256", "file_count", "files")
    }:
        raise VerificationError("retained frontend tree differs from the release identity")
    if manifest_record["sha256"] != source_evidence.get("manifest_sha256"):
        raise VerificationError("retained release manifest differs from verified source evidence")
    return {
        "schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "promotable": True,
        "source_kind": source_evidence.get("kind"),
        "manifest": manifest_record,
        "control_binary": control_record,
        "frontend_dist": frontend_record,
    }


def _staged_sandbox_source_files(repo_root: Path) -> list[Path]:
    """Return only the files mounted into the isolated image-parity workspace."""

    fixed = [
        Path("scripts/materials_domain_gate.py"),
        Path("scripts/calphad_experimental_benchmark.py"),
        Path("scripts/materials_readiness_gate.py"),
        Path("scripts/verify_production_materials_sandbox.py"),
        Path("deploy/docker/deepagents-sandbox.Dockerfile"),
        Path("deploy/docker/materials-requirements.txt"),
        Path("backend/deepagents_runtime/tests/domain_correctness/conftest.py"),
        Path("backend/deepagents_runtime/tests/domain_correctness/test_materials_invariants.py"),
        Path("backend/deepagents_runtime/tests/test_calphad_runtime.py"),
        Path("backend/deepagents_runtime/tests/test_calphad_cli.py"),
        Path("backend/deepagents_runtime/tests/test_calphad_tools.py"),
        Path("backend/deepagents_runtime/src/ultra_deepagents/agent.py"),
        Path("backend/deepagents_runtime/src/ultra_deepagents/crystal_plasticity_tools.py"),
        Path("backend/deepagents_runtime/src/ultra_deepagents/context_tools.py"),
        Path("backend/deepagents_runtime/src/ultra_deepagents/code_execution/docker.py"),
        Path("backend/deepagents_runtime/src/ultra_deepagents/imaging/hdf5.py"),
        Path("backend/deepagents_runtime/materials_data/calphad/manifest.json"),
        Path(
            "backend/deepagents_runtime/materials_data/calphad/experimental_benchmark_manifest.json"
        ),
    ]
    materials = Path("backend/deepagents_runtime/src/ultra_deepagents/materials")
    fixed.extend(
        path.relative_to(repo_root) for path in sorted((repo_root / materials).glob("*.py"))
    )
    calphad_data = Path("backend/deepagents_runtime/materials_data/calphad")
    fixed.extend(
        path.relative_to(repo_root) for path in sorted((repo_root / calphad_data).glob("*.tdb"))
    )
    return _require_regular_files(repo_root, fixed, label="required parity source")


def _stage_source(repo_root: Path, workspace: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for relative in _staged_sandbox_source_files(repo_root):
        source = repo_root / relative
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        entries.append(
            {
                "path": relative.as_posix(),
                "sha256": _sha256_file(target),
                "size_bytes": target.stat().st_size,
            }
        )

    # The repository package initializers import the networked worker runtime.
    # Empty adapter initializers expose only the staged, sandbox-safe modules.
    generated = [
        Path("backend/deepagents_runtime/src/ultra_deepagents/__init__.py"),
        Path("backend/deepagents_runtime/src/ultra_deepagents/imaging/__init__.py"),
    ]
    for relative in generated:
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")

    canonical = json.dumps(entries, separators=(",", ":"), sort_keys=True).encode()
    return {
        "file_count": len(entries),
        "aggregate_sha256": _sha256_bytes(canonical),
        "files": entries,
        "generated_import_shims": [path.as_posix() for path in generated],
    }


def _stage_host_tool_source(
    repo_root: Path,
    destination: Path,
    required_materials: Mapping[str, Any],
) -> dict[str, Any]:
    raw_files = required_materials.get("files")
    if not isinstance(raw_files, list):
        raise VerificationError("verified release source manifest is malformed")
    fixed = {
        "backend/deepagents_runtime/pyproject.toml",
        "backend/deepagents_runtime/uv.lock",
        "backend/deepagents_runtime/tests/test_calphad_tools.py",
    }
    source_prefix = "backend/deepagents_runtime/src/ultra_deepagents/"
    selected = [
        record
        for record in raw_files
        if isinstance(record, Mapping)
        and (
            str(record.get("path") or "") in fixed
            or str(record.get("path") or "").startswith(source_prefix)
        )
    ]
    selected_paths = {str(record.get("path") or "") for record in selected}
    if not fixed <= selected_paths or not any(
        path.startswith(source_prefix) for path in selected_paths
    ):
        raise VerificationError("verified release manifest lacks host CALPHAD tool inputs")
    relatives = [Path(path) for path in sorted(selected_paths)]
    for relative in relatives:
        source = repo_root / relative
        _require_regular_file(source, label="host CALPHAD source")
        declared = next(record for record in selected if record.get("path") == relative.as_posix())
        if _sha256_file(source) != declared.get("sha256") or source.stat().st_size != declared.get(
            "size_bytes"
        ):
            raise VerificationError(
                f"host CALPHAD source changed after release verification: {relative.as_posix()}"
            )
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    observed = _manifest_for_files(destination, relatives)
    expected = _manifest_for_entries(selected)
    if observed != expected:
        raise VerificationError("host CALPHAD source snapshot differs from the release manifest")
    return observed


_CALPHAD_PROBE = r"""from __future__ import annotations
import hashlib
import json
import math
import os
import sys
from pathlib import Path

from pycalphad import Database
import ultra_deepagents.materials as baked_materials
from ultra_deepagents.materials import calphad as baked_calphad
from ultra_deepagents.materials import inspect_calphad_input

EXPECTED_MANIFEST_SHA256 = "a5dfb3aac68f119a8fe0ee751255a16f31fe8e8515cfa9739320ba21aa28fb09"
EXPECTED_CALPHAD_SOURCE_HASHES = {
    "calphad.py": "c17c9158457c4aa236fa58865372f59325d479226418f28c1c7e998ce68cdc85",
    "calphad_cli.py": "68cfbb3ca78560c686f43c7bb790e8f97a3e770ca5fc802bbf13eecc56ad466e",
    "calphad_tools.py": "51587074ea751e0af1a07b4b4d9e169a1b4067ff8d11b356a6f2ac92f4f5fd82",
}

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

workspace = Path("/workspace")
source_package = workspace / "backend/deepagents_runtime/src/ultra_deepagents/materials"
baked_package = Path(baked_materials.__file__).resolve().parent
expected_baked_package = Path(os.environ["ULTRA_EXPECTED_BAKED_MATERIALS_PATH"])
if baked_package != expected_baked_package or workspace in baked_package.parents:
    raise RuntimeError("materials runtime was imported from staged source instead of the baked image")
source_files = {path.name: sha(path) for path in source_package.glob("*.py")}
baked_files = {path.name: sha(path) for path in baked_package.glob("*.py")}
if source_files != baked_files:
    raise RuntimeError("baked materials runtime does not match checked-out source")
if any(source_files.get(name) != digest for name, digest in EXPECTED_CALPHAD_SOURCE_HASHES.items()):
    raise RuntimeError("baked CALPHAD source hashes differ from the reviewed release contract")
if baked_calphad.EQUILIBRIUM_SCHEMA_VERSION != "ultra.calphad.equilibrium.v2":
    raise RuntimeError("embedded CALPHAD equilibrium schema is not v2")

source_root = workspace / "backend/deepagents_runtime/materials_data/calphad"
embedded_root = Path("/opt/ultra-calphad")
source_manifest_path = source_root / "manifest.json"
embedded_manifest_path = embedded_root / "manifest.json"
if source_manifest_path.read_bytes() != embedded_manifest_path.read_bytes():
    raise RuntimeError("embedded CALPHAD manifest differs from checked-out source")
if sha(source_manifest_path) != EXPECTED_MANIFEST_SHA256:
    raise RuntimeError("embedded CALPHAD manifest hash differs from the reviewed release contract")
manifest = json.loads(embedded_manifest_path.read_text(encoding="utf-8"))
if manifest.get("schema_version") != "1" or not manifest.get("databases"):
    raise RuntimeError("embedded CALPHAD manifest is empty or unsupported")

records = []
for entry in manifest["databases"]:
    source_path = source_root / entry["filename"]
    embedded_path = embedded_root / entry["filename"]
    source_hash = sha(source_path)
    embedded_hash = sha(embedded_path)
    if source_hash != entry["sha256"] or embedded_hash != entry["sha256"]:
        raise RuntimeError("embedded CALPHAD database hash mismatch")
    if source_path.stat().st_size != entry["size_bytes"] or embedded_path.stat().st_size != entry["size_bytes"]:
        raise RuntimeError("embedded CALPHAD database size mismatch")
    declared_format = str(entry.get("format") or "").casefold()
    if declared_format not in {"tdb", "dat"} or embedded_path.suffix.casefold() != f".{declared_format}":
        raise RuntimeError("embedded CALPHAD format is unsupported or disagrees with its suffix")
    pressure_limits = entry.get("assessment_pressure_limits_Pa")
    if (
        not isinstance(pressure_limits, list)
        or len(pressure_limits) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in pressure_limits)
        or any(not math.isfinite(float(value)) for value in pressure_limits)
        or float(pressure_limits[0]) < 1e-9
        or float(pressure_limits[0]) > float(pressure_limits[1])
        or float(pressure_limits[1]) > 1e12
    ):
        raise RuntimeError("embedded CALPHAD assessment pressure limits are invalid")
    pressure_limits = [float(value) for value in pressure_limits]
    database = Database.from_file(str(embedded_path), fmt=declared_format)
    elements = sorted(item for item in database.elements if item != "/-")
    phases = sorted(database.phases)
    if elements != sorted(entry["elements"]) or phases != sorted(entry["phases"]):
        raise RuntimeError("embedded CALPHAD parse domain differs from manifest")
    inspection = inspect_calphad_input(
        embedded_root,
        database_id=entry["database_id"],
        components=entry["elements"],
        phases=entry["phases"],
    )
    if inspection["sha256"] != entry["sha256"]:
        raise RuntimeError("Ultra CALPHAD inspection hash mismatch")
    if inspection.get("format") != declared_format:
        raise RuntimeError("Ultra CALPHAD inspection format disagrees with the manifest")
    if inspection.get("assessment_pressure_limits_Pa") != pressure_limits:
        raise RuntimeError("Ultra CALPHAD inspection pressure scope disagrees with the manifest")
    records.append({
        "database_id": entry["database_id"],
        "filename": entry["filename"],
        "sha256": embedded_hash,
        "size_bytes": embedded_path.stat().st_size,
        "format": declared_format,
        "assessment_pressure_limits_Pa": pressure_limits,
        "elements": elements,
        "phases": phases,
        "pycalphad_parse_supported": True,
        "ultra_inspection_supported": True,
    })

payload = {
    "schema_version": 1,
    "status": "passed",
    "equilibrium_schema_version": baked_calphad.EQUILIBRIUM_SCHEMA_VERSION,
    "baked_materials_path": str(baked_package),
    "materials_source_hashes": source_files,
    "materials_baked_hashes": baked_files,
    "source_manifest_sha256": sha(source_manifest_path),
    "embedded_manifest_sha256": sha(embedded_manifest_path),
    "database_count": len(records),
    "databases": records,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
"""


def _write_probe(workspace: Path) -> Path:
    path = workspace / ".ultra-parity" / "calphad_probe.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    compile(_CALPHAD_PROBE, str(path), "exec")
    path.write_text(_CALPHAD_PROBE, encoding="utf-8")
    return path


def _prepare_entrypoint_adapter(
    base: ImageInspection,
    *,
    inspector: ImageInspector,
) -> ImageInspection:
    if not base.entrypoint:
        return base
    if SAFE_IMAGE_REF_PATTERN.fullmatch(base.ref) is None:
        raise VerificationError(f"unsafe image reference for adapter build: {base.ref!r}")
    identifier = base.image_id.removeprefix("sha256:")[:24]
    pinned_base_ref = "bisque-ultra-materials-parity-base:" + identifier
    adapter_ref = "bisque-ultra-materials-parity-adapter:" + identifier
    tag_result = subprocess.run(
        ["docker", "image", "tag", base.image_id, pinned_base_ref],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if tag_result.returncode != 0:
        raise VerificationError(f"could not pin adapter base tag: {tag_result.stderr.strip()}")
    if inspector(pinned_base_ref).image_id != base.image_id:
        raise VerificationError("entrypoint adapter base tag did not resolve to inspected image ID")
    dockerfile = f"FROM {pinned_base_ref}\nENTRYPOINT []\n"
    with tempfile.TemporaryDirectory(prefix="ultra-materials-adapter-") as context:
        result = subprocess.run(
            ["docker", "build", "--file", "-", "--tag", adapter_ref, context],
            input=dockerfile,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    if result.returncode != 0:
        raise VerificationError(f"could not build entrypoint adapter: {result.stderr.strip()}")
    return inspector(adapter_ref)


def _real_backend_factory(
    workspace: Path,
    outputs: Path,
    image_id: str,
    policy: SandboxPolicy,
) -> SandboxBackend:
    repo_root = Path(__file__).resolve().parents[1]
    runtime_source = repo_root / "backend" / "deepagents_runtime" / "src"
    sys.path.insert(0, str(runtime_source))
    from ultra_deepagents.code_execution.docker import (  # noqa: PLC0415
        DockerSandboxBackend,
        DockerSandboxConfig,
    )

    return DockerSandboxBackend(
        workspace_dir=workspace,
        outputs_dir=outputs,
        config=DockerSandboxConfig(
            image=image_id,
            network=policy.network,
            cpus=policy.cpus,
            memory=policy.memory,
            pids_limit=policy.pids_limit,
            shm_size=policy.shm_size,
            gpus=policy.gpus,
            timeout_seconds=policy.timeout_seconds,
            output_limit_bytes=policy.output_limit_bytes,
            worker_id="materials-parity",
            run_id="production-sandbox-verification",
        ),
    )


def validate_backend_command(
    command: Sequence[str],
    *,
    image_id: str,
    policy: SandboxPolicy,
    workspace: Path,
    outputs: Path,
    expected_command: str,
) -> list[str]:
    """Validate the effective Docker argv, rejecting duplicates and overrides."""

    failures: list[str] = []
    tokens = list(command)
    if tokens[:2] != ["docker", "run"]:
        return ["DockerSandboxBackend command must start with exactly 'docker run'"]
    image_positions = [index for index, token in enumerate(tokens) if token == image_id]
    if len(image_positions) != 1:
        return ["DockerSandboxBackend command must contain the immutable image ID exactly once"]
    image_index = image_positions[0]
    expected_tail_length = 4
    if len(tokens) != image_index + expected_tail_length or tokens[
        image_index + 1 : image_index + 3
    ] != ["bash", "-lc"]:
        failures.append("DockerSandboxBackend image must be followed only by 'bash -lc <command>'")
    elif tokens[image_index + 3] != expected_command:
        failures.append("DockerSandboxBackend shell payload differs from the verified command")

    boolean_options = {"--read-only", "--rm"}
    value_options = {
        "--cap-drop",
        "--cpus",
        "--env",
        "--gpus",
        "--label",
        "--memory",
        "--network",
        "--pids-limit",
        "--security-opt",
        "--shm-size",
        "--tmpfs",
        "--volume",
        "--workdir",
    }
    parsed: dict[str, list[str | None]] = {}
    index = 2
    while index < image_index:
        token = tokens[index]
        if not token.startswith("--"):
            failures.append(f"forbidden short/positional Docker option before image: {token!r}")
            index += 1
            continue
        name, separator, inline_value = token.partition("=")
        if name in boolean_options:
            parsed.setdefault(name, []).append(inline_value if separator else None)
            index += 1
            continue
        if name not in value_options:
            failures.append(f"forbidden or unknown Docker option: {name}")
            index += 1
            continue
        if separator:
            value = inline_value
            index += 1
        elif index + 1 < image_index:
            value = tokens[index + 1]
            index += 2
        else:
            failures.append(f"Docker option {name} is missing its value")
            index += 1
            continue
        parsed.setdefault(name, []).append(value)

    def require_unique(name: str, expected: str | None) -> None:
        values = parsed.get(name, [])
        if len(values) != 1:
            failures.append(
                f"DockerSandboxBackend requires exactly one {name}; found {len(values)}"
            )
        elif values[0] != expected:
            failures.append(
                f"DockerSandboxBackend has wrong {name}: {values[0]!r}; expected {expected!r}"
            )

    require_unique("--rm", None)
    require_unique("--network", policy.network)
    require_unique("--cap-drop", "ALL")
    require_unique("--security-opt", "no-new-privileges")
    require_unique("--read-only", None)
    require_unique("--cpus", str(policy.cpus))
    require_unique("--memory", policy.memory)
    require_unique("--pids-limit", str(policy.pids_limit))
    require_unique("--shm-size", policy.shm_size)
    require_unique("--tmpfs", "/tmp:rw,nosuid,nodev,size=512m")
    require_unique("--workdir", "/workspace")
    if policy.gpus:
        require_unique("--gpus", policy.gpus)
    elif parsed.get("--gpus"):
        failures.append("DockerSandboxBackend unexpectedly enables GPU device passthrough")

    volumes = [str(value) for value in parsed.get("--volume", [])]
    expected_volumes = {
        f"{workspace.resolve()}:/workspace:rw",
        f"{outputs.resolve()}:/outputs:rw",
    }
    if len(volumes) != 2 or set(volumes) != expected_volumes:
        failures.append(
            "DockerSandboxBackend volumes do not bind the exact verified workspace and outputs"
        )

    expected_environment = {
        "HOME": "/workspace",
        "MPLCONFIGDIR": "/workspace/.cache/matplotlib",
        "NUMBA_CACHE_DIR": "/workspace/.cache/numba",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": "/workspace/.tmp",
        "XDG_CACHE_HOME": "/workspace/.cache",
    }
    observed_environment: dict[str, str] = {}
    for raw in parsed.get("--env", []):
        name, separator, value = str(raw).partition("=")
        if not separator or name in observed_environment:
            failures.append(f"duplicate or malformed Docker environment override: {raw!r}")
            continue
        observed_environment[name] = value
    if observed_environment != expected_environment:
        failures.append("DockerSandboxBackend environment overrides differ from the safe contract")

    labels: dict[str, str] = {}
    for raw in parsed.get("--label", []):
        name, separator, value = str(raw).partition("=")
        if not separator or name in labels:
            failures.append(f"duplicate or malformed Docker label override: {raw!r}")
            continue
        labels[name] = value
    if labels.get("ultra.sandbox") != "1":
        failures.append("DockerSandboxBackend must label the container ultra.sandbox=1")
    if labels.get("ultra.sandbox.cap") != str(policy.timeout_seconds):
        failures.append("DockerSandboxBackend timeout label disagrees with the policy")

    try:
        _validated_policy(policy)
    except VerificationError as exc:
        failures.append(str(exc))
    return failures


def _gate_environment(
    *,
    expected_git_sha: str,
    image: ImageInspection,
    requirements_sha256: str,
    scope: str,
) -> dict[str, str]:
    return {
        "MATERIALS_DOMAIN_GATE_REQUIRE_CLEAN_PROVENANCE": "1",
        "ULTRA_MATERIALS_GATE_GIT_DIRTY": "false",
        "ULTRA_MATERIALS_GATE_GIT_REF": expected_git_sha,
        "ULTRA_MATERIALS_GATE_GIT_SHA": expected_git_sha,
        "ULTRA_MATERIALS_GATE_IMAGE_DIGEST": "",
        "ULTRA_MATERIALS_GATE_IMAGE_ID": image.image_id,
        "ULTRA_MATERIALS_GATE_IMAGE_REF": image.ref,
        "ULTRA_MATERIALS_GATE_REQUIRE_CALPHAD_RUNTIME_JUNIT": "1",
        "ULTRA_MATERIALS_GATE_REQUIREMENTS_SHA256": requirements_sha256,
        "ULTRA_EXPECTED_BAKED_MATERIALS_PATH": (
            "/opt/ultra-runtime/ultra_deepagents/materials"
            if scope == "production-full"
            else "/opt/ultra/src/ultra_deepagents/materials"
        ),
        "PYTEST_ADDOPTS": "",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTEST_PLUGINS": "",
    }


def _execution_command(
    *,
    expected_git_sha: str,
    image: ImageInspection,
    requirements_sha256: str,
    scope: str,
) -> str:
    exports = "\n".join(
        f"export {name}={shlex.quote(value)}"
        for name, value in sorted(
            _gate_environment(
                expected_git_sha=expected_git_sha,
                image=image,
                requirements_sha256=requirements_sha256,
                scope=scope,
            ).items()
        )
    )
    return f"""set -euo pipefail
{exports}
mkdir -p /outputs/domain
python /workspace/.ultra-parity/calphad_probe.py /outputs/{CALPHAD_REPORT.as_posix()}
python -m pytest \
  /workspace/backend/deepagents_runtime/tests/test_calphad_runtime.py \
  /workspace/backend/deepagents_runtime/tests/test_calphad_cli.py \
  -q -ra --color=no --tb=short -p no:cacheprovider \
  -c /dev/null --rootdir=/workspace/backend/deepagents_runtime --noconftest \
  -o junit_family=legacy \
  --junitxml=/outputs/{CALPHAD_RUNTIME_JUNIT.as_posix()}
python /workspace/scripts/materials_domain_gate.py \\
  --repo-root /workspace \\
  --requirements /workspace/deploy/docker/materials-requirements.txt \\
  --test-path backend/deepagents_runtime/tests/domain_correctness/test_materials_invariants.py \\
  --calphad-runtime-junit /outputs/{CALPHAD_RUNTIME_JUNIT.as_posix()} \\
  --output-dir /outputs/domain
"""


def run_host_calphad_tools_suite(source_root: Path, junit_path: Path) -> dict[str, Any]:
    """Run the focused worker-orchestration contract outside the untrusted sandbox."""

    uv = shutil.which("uv")
    if uv is None:
        raise VerificationError("uv is required for the CALPHAD host-tool orchestration suite")
    runtime_root = source_root / "backend" / "deepagents_runtime"
    test_path = runtime_root / "tests" / "test_calphad_tools.py"
    command = [
        uv,
        "run",
        "--frozen",
        "--no-sync",
        "--project",
        str(runtime_root),
        "--python",
        "3.11",
        "--with",
        "pytest==8.4.2",
        "pytest",
        "-q",
        str(test_path),
        "-c",
        "/dev/null",
        f"--rootdir={runtime_root}",
        "--noconftest",
        "-p",
        "no:cacheprovider",
        "-o",
        "junit_family=legacy",
        f"--junitxml={junit_path}",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTEST_ADDOPTS": "",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTEST_PLUGINS": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(runtime_root / "src"),
            "UV_PROJECT_ENVIRONMENT": sys.prefix,
        }
    )
    environment.pop("PYTHONHOME", None)
    process = subprocess.run(
        command,
        cwd=source_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    stdout = process.stdout or ""
    stderr = process.stderr or ""
    return {
        "runner": "uv-frozen-project-with-pytest-8.4.2",
        "source_isolation": {
            "config": "/dev/null",
            "conftest_loading": False,
            "plugin_autoload": False,
            "pytest_plugins": "",
            "pythonpath": "retained-host-source/backend/deepagents_runtime/src",
            "uv_sync": False,
        },
        "exit_code": process.returncode,
        "stdout_text": stdout,
        "stderr_text": stderr,
        "stdout_size_bytes": len(stdout.encode("utf-8", "replace")),
        "stdout_sha256": _sha256_bytes(stdout.encode("utf-8", "replace")),
        "stderr_size_bytes": len(stderr.encode("utf-8", "replace")),
        "stderr_sha256": _sha256_bytes(stderr.encode("utf-8", "replace")),
    }


def _junit_summary(path: Path) -> tuple[ET.Element, list[ET.Element], dict[str, int], list[str]]:
    failures: list[str] = []
    try:
        size = path.stat().st_size
        if size <= 0 or size > MAX_JUNIT_BYTES:
            raise VerificationError(f"JUnit size {size} is outside 1..{MAX_JUNIT_BYTES} bytes")
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError, VerificationError) as exc:
        raise VerificationError(str(exc)) from exc
    testcases = [element for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "testcase"]
    outcomes = {"failures": 0, "errors": 0, "skipped": 0}
    for testcase in testcases:
        children = {child.tag.rsplit("}", 1)[-1] for child in testcase}
        outcomes["failures"] += int("failure" in children)
        outcomes["errors"] += int("error" in children)
        outcomes["skipped"] += int("skipped" in children)
    summary = {"tests": len(testcases), **outcomes}
    leaf_suites = [
        suite
        for suite in root.iter()
        if suite.tag.rsplit("}", 1)[-1] == "testsuite"
        and not any(child.tag.rsplit("}", 1)[-1] == "testsuite" for child in suite)
    ]
    declared = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    try:
        for suite in leaf_suites:
            for field in declared:
                value = int(suite.attrib.get(field, "0"))
                if value < 0:
                    raise ValueError(f"negative {field}")
                declared[field] += value
    except ValueError as exc:
        failures.append(f"JUnit has invalid suite counters: {exc}")
    if not leaf_suites or declared != summary:
        failures.append("JUnit suite counters disagree with testcase outcomes")
    return root, testcases, summary, failures


def validate_calphad_runtime_junit(path: Path) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    try:
        size = path.stat().st_size
        if size <= 0 or size > MAX_JUNIT_BYTES:
            raise VerificationError(
                f"CALPHAD runtime JUnit size {size} is outside 1..{MAX_JUNIT_BYTES} bytes"
            )
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError, VerificationError) as exc:
        return ({"tests": 0, "failures": 0, "errors": 0, "skipped": 0}, [str(exc)])

    testcases = [element for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "testcase"]
    outcomes = {"failures": 0, "errors": 0, "skipped": 0}
    identities: list[str] = []
    core_names: list[str] = []
    typed_cli_names: list[str] = []
    for testcase in testcases:
        classname = str(testcase.attrib.get("classname") or "")
        name = str(testcase.attrib.get("name") or "")
        if classname == "tests.test_calphad_runtime":
            core_names.append(name)
        elif classname == "tests.test_calphad_cli":
            typed_cli_names.append(name)
        else:
            failures.append("CALPHAD runtime JUnit contains an unrelated or unnamed testcase")
        if not name:
            failures.append("CALPHAD runtime JUnit contains an unrelated or unnamed testcase")
        identities.append(f"{classname}::{name}")
        children = {child.tag.rsplit("}", 1)[-1] for child in testcase}
        outcomes["failures"] += int("failure" in children)
        outcomes["errors"] += int("error" in children)
        outcomes["skipped"] += int("skipped" in children)
    summary = {"tests": len(testcases), **outcomes}
    leaf_suites = [
        suite
        for suite in root.iter()
        if suite.tag.rsplit("}", 1)[-1] == "testsuite"
        and not any(child.tag.rsplit("}", 1)[-1] == "testsuite" for child in suite)
    ]
    declared = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    try:
        for suite in leaf_suites:
            for field in declared:
                value = int(suite.attrib.get(field, "0"))
                if value < 0:
                    raise ValueError(f"negative {field}")
                declared[field] += value
    except ValueError as exc:
        failures.append(f"CALPHAD runtime JUnit has invalid suite counters: {exc}")
    if not leaf_suites or declared != summary:
        failures.append("CALPHAD runtime JUnit suite counters disagree with testcase outcomes")
    if len(testcases) != REQUIRED_CALPHAD_RUNTIME_TEST_COUNT:
        failures.append(
            "CALPHAD runtime JUnit must contain exactly "
            f"{REQUIRED_CALPHAD_RUNTIME_TEST_COUNT} tests; found {len(testcases)}"
        )
    if len(core_names) != REQUIRED_CALPHAD_CORE_TEST_COUNT:
        failures.append(
            "CALPHAD runtime JUnit must contain exactly "
            f"{REQUIRED_CALPHAD_CORE_TEST_COUNT} core runtime tests; found {len(core_names)}"
        )
    if set(core_names) != set(REQUIRED_CALPHAD_CORE_TEST_NAMES):
        failures.append(
            "CALPHAD runtime JUnit does not contain the exact required core test identities"
        )
    if len(typed_cli_names) != REQUIRED_TYPED_CALPHAD_CLI_TEST_COUNT:
        failures.append(
            "CALPHAD runtime JUnit must contain exactly "
            f"{REQUIRED_TYPED_CALPHAD_CLI_TEST_COUNT} real typed CLI tests; "
            f"found {len(typed_cli_names)}"
        )
    if set(typed_cli_names) != set(REQUIRED_TYPED_CALPHAD_CLI_TEST_NAMES):
        failures.append(
            "CALPHAD runtime JUnit does not contain the exact required typed CLI test identity"
        )
    if len(set(identities)) != len(identities):
        failures.append("CALPHAD runtime JUnit contains duplicate testcase identities")
    for field in ("failures", "errors", "skipped"):
        if summary[field] != 0:
            failures.append(f"CALPHAD runtime JUnit {field} is nonzero")
    return summary, failures


def validate_calphad_tools_junit(path: Path) -> tuple[dict[str, Any], list[str]]:
    """Require the reviewed host/worker orchestration subset with no skips."""

    try:
        _, testcases, summary, failures = _junit_summary(path)
    except VerificationError as exc:
        return ({"tests": 0, "failures": 0, "errors": 0, "skipped": 0}, [str(exc)])
    identities: list[str] = []
    for testcase in testcases:
        classname = str(testcase.attrib.get("classname") or "")
        name = str(testcase.attrib.get("name") or "")
        if classname != "tests.test_calphad_tools" or not name:
            failures.append("CALPHAD tools JUnit contains an unrelated or unnamed testcase")
        identities.append(f"{classname}::{name}")
    if len(testcases) != REQUIRED_CALPHAD_TOOLS_TEST_COUNT:
        failures.append(
            "CALPHAD tools JUnit must contain exactly "
            f"{REQUIRED_CALPHAD_TOOLS_TEST_COUNT} tests; found {len(testcases)}"
        )
    if {identity.rsplit("::", 1)[-1] for identity in identities} != set(
        REQUIRED_CALPHAD_TOOL_TEST_NAMES
    ):
        failures.append("CALPHAD tools JUnit does not contain the exact required test identities")
    if len(set(identities)) != len(identities):
        failures.append("CALPHAD tools JUnit contains duplicate testcase identities")
    for field in ("failures", "errors", "skipped"):
        if summary[field] != 0:
            failures.append(f"CALPHAD tools JUnit {field} is nonzero")
    return summary, failures


def validate_domain_junit_binding(
    path: Path,
    domain_report: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    try:
        _, testcases, summary, junit_failures = _junit_summary(path)
    except VerificationError as exc:
        return [str(exc)]
    failures.extend(junit_failures)
    declared_summary = domain_report.get("junit")
    if not isinstance(declared_summary, Mapping) or any(
        int(declared_summary.get(key, -1)) != value for key, value in summary.items()
    ):
        failures.append("retained domain JUnit summary differs from the domain report")
    records: list[dict[str, Any]] = []
    for testcase in testcases:
        name = str(testcase.attrib.get("name") or "").strip()
        property_values: list[str] = []
        for child in testcase:
            if child.tag.rsplit("}", 1)[-1] != "properties":
                continue
            for prop in child:
                if (
                    prop.tag.rsplit("}", 1)[-1] == "property"
                    and prop.attrib.get("name") == "materials_invariant_evidence"
                ):
                    property_values.append(str(prop.attrib.get("value") or prop.text or ""))
        if len(property_values) != 1:
            failures.append(f"domain JUnit testcase {name!r} lacks one invariant record")
            continue
        try:
            payload = json.loads(
                property_values[0],
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant {value!r}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            failures.append(f"domain JUnit testcase {name!r} has malformed evidence: {exc}")
            continue
        if not isinstance(payload, dict) or payload.get("test_id") != name:
            failures.append(f"domain JUnit testcase {name!r} has a mismatched evidence test ID")
            continue
        records.append(payload)
    records.sort(key=lambda record: (str(record.get("test_id")), str(record.get("validator_id"))))
    declared_records = [
        dict(record)
        for record in domain_report.get("invariants", [])
        if isinstance(record, Mapping)
    ]
    declared_records.sort(
        key=lambda record: (str(record.get("test_id")), str(record.get("validator_id")))
    )
    if records != declared_records:
        failures.append("retained domain JUnit invariant evidence differs from the JSON report")
    return failures


def validate_domain_calphad_experimental_benchmark(
    domain_report: Mapping[str, Any],
    *,
    retained_path: Path | None = None,
) -> list[str]:
    failures: list[str] = []
    wrapper = domain_report.get("calphad_experimental_benchmark")
    if not isinstance(wrapper, Mapping):
        return ["required CALPHAD experimental benchmark evidence is missing"]
    report = wrapper.get("report")
    if not isinstance(report, Mapping):
        return ["required CALPHAD experimental benchmark report is malformed"]
    if wrapper.get("relative_path") != "calphad-experimental-benchmark.json":
        failures.append("CALPHAD experimental benchmark retained path is unexpected")
    if re.fullmatch(r"[0-9a-f]{64}", str(wrapper.get("sha256") or "")) is None:
        failures.append("CALPHAD experimental benchmark hash is missing or malformed")
    if not isinstance(wrapper.get("size_bytes"), int) or int(wrapper["size_bytes"]) <= 0:
        failures.append("CALPHAD experimental benchmark size is missing or invalid")
    if retained_path is not None:
        if not retained_path.is_file() or retained_path.is_symlink():
            failures.append("retained CALPHAD experimental benchmark file is missing")
        else:
            try:
                retained = _load_json(retained_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                failures.append(f"retained CALPHAD experimental benchmark is unreadable: {exc}")
            else:
                if retained != report:
                    failures.append(
                        "retained CALPHAD experimental benchmark differs from domain report"
                    )
                if wrapper.get("sha256") != _sha256_file(retained_path):
                    failures.append("retained CALPHAD experimental benchmark hash differs")
                if wrapper.get("size_bytes") != retained_path.stat().st_size:
                    failures.append("retained CALPHAD experimental benchmark size differs")

    lanes = report.get("lanes") if isinstance(report.get("lanes"), Mapping) else {}
    calibration = lanes.get("calibration") if isinstance(lanes.get("calibration"), Mapping) else {}
    held_out = lanes.get("held_out") if isinstance(lanes.get("held_out"), Mapping) else {}
    calibration_metrics = (
        calibration.get("metrics") if isinstance(calibration.get("metrics"), Mapping) else {}
    )
    held_out_metrics = (
        held_out.get("metrics") if isinstance(held_out.get("metrics"), Mapping) else {}
    )

    def finite_number(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    calibration_rms = finite_number(calibration_metrics.get("weighted_rms_z"))
    calibration_max = finite_number(calibration_metrics.get("max_abs_z"))
    held_out_mae = finite_number(held_out_metrics.get("mae_K"))
    held_out_max = finite_number(held_out_metrics.get("max_abs_error_K"))
    observations = held_out.get("observations")
    observations = observations if isinstance(observations, list) else []
    valid = all(
        (
            report.get("schema_version") == "ultra.calphad.experimental_benchmark.v1",
            report.get("benchmark_id") == "materials.calphad.al_co_w_experimental_two_lane.v1",
            report.get("status") == "passed",
            report.get("required_independent_invariant") is True,
            report.get("production_promotion_blocked") is False,
            calibration.get("classification") == "calibration",
            calibration.get("independent_validation") is False,
            calibration.get("required") is True,
            calibration.get("status") == "passed",
            calibration.get("observation_count") == 6,
            calibration_rms is not None and calibration_rms <= 1.0,
            calibration_max is not None and calibration_max <= 2.0,
            calibration_metrics.get("weighted_rms_z_max") == 1.0,
            calibration_metrics.get("max_abs_z_max") == 2.0,
            held_out.get("classification") == "held_out",
            held_out.get("independent_validation") is True,
            held_out.get("required") is True,
            held_out.get("status") == "passed",
            held_out.get("observation_count") == 4,
            len(observations) == 4,
            all(
                isinstance(observation, Mapping)
                and observation.get("reported_uncertainty_K") is None
                and observation.get("uncertainty_status") == "not_reported_numerically"
                for observation in observations
            ),
            held_out_mae is not None and held_out_mae <= 20.0,
            held_out_max is not None and held_out_max <= 30.0,
            held_out_metrics.get("mae_K_max") == 20.0,
            held_out_metrics.get("max_abs_error_K_max") == 30.0,
        )
    )
    if not valid:
        failures.append(
            "required CALPHAD calibration/independent holdout metrics did not pass locked policy"
        )
    return failures


def validate_domain_report(
    report: Mapping[str, Any],
    *,
    image_id: str,
    required_validator_ids: Sequence[str],
) -> list[str]:
    failures: list[str] = []
    junit = report.get("junit") if isinstance(report.get("junit"), Mapping) else {}
    evidence = (
        report.get("invariant_evidence")
        if isinstance(report.get("invariant_evidence"), Mapping)
        else {}
    )
    pytest_record = report.get("pytest") if isinstance(report.get("pytest"), Mapping) else {}
    image = report.get("image") if isinstance(report.get("image"), Mapping) else {}
    invariants = report.get("invariants") if isinstance(report.get("invariants"), list) else []
    if report.get("gate") != "materials-domain-gate":
        failures.append("deterministic materials report has the wrong gate identity")
    if report.get("scope") != "deterministic-domain-invariants":
        failures.append("deterministic materials report has the wrong scope")
    if report.get("status") != "passed":
        failures.append("deterministic materials domain report did not pass")
    if int(pytest_record.get("exit_code", -1)) != 0:
        failures.append("deterministic materials pytest exit was nonzero")
    tests = int(junit.get("tests", 0))
    if tests != REQUIRED_DOMAIN_INVARIANT_COUNT:
        failures.append(
            "deterministic materials JUnit must contain exactly "
            f"{REQUIRED_DOMAIN_INVARIANT_COUNT} tests; found {tests}"
        )
    for field in ("failures", "errors", "skipped"):
        if int(junit.get(field, -1)) != 0:
            failures.append(f"deterministic materials JUnit {field} is nonzero")
    if report.get("version_drift") not in ([], ()):
        failures.append("deterministic materials direct dependency versions drifted")
    if evidence.get("complete") is not True:
        failures.append("deterministic scientific invariant evidence is incomplete")
    if int(evidence.get("record_count", -1)) != REQUIRED_DOMAIN_INVARIANT_COUNT:
        failures.append("deterministic evidence count is not exactly 13")
    if int(evidence.get("passed", -1)) != REQUIRED_DOMAIN_INVARIANT_COUNT:
        failures.append("deterministic evidence does not contain exactly 13 passes")
    if int(evidence.get("failed", -1)) != 0:
        failures.append("deterministic scientific invariant evidence contains failures")
    if evidence.get("errors") not in ([], ()):
        failures.append("deterministic scientific invariant evidence contains schema errors")
    expected_validators = tuple(required_validator_ids)
    observed_validators = [
        str(record.get("validator_id") or "")
        for record in invariants
        if isinstance(record, Mapping)
    ]
    observed_test_ids = [
        str(record.get("test_id") or "").strip()
        for record in invariants
        if isinstance(record, Mapping)
    ]
    if len(invariants) != REQUIRED_DOMAIN_INVARIANT_COUNT:
        failures.append("deterministic report does not contain exactly 13 invariant records")
    if len(observed_validators) != len(invariants):
        failures.append("deterministic report contains malformed invariant records")
    if len(set(observed_validators)) != len(observed_validators):
        failures.append("deterministic report contains duplicate validator IDs")
    if (
        len(observed_test_ids) != len(invariants)
        or any(not test_id for test_id in observed_test_ids)
        or len(set(observed_test_ids)) != len(observed_test_ids)
    ):
        failures.append("deterministic report contains blank or duplicate test IDs")
    if set(observed_validators) != set(expected_validators):
        failures.append("deterministic report does not contain the exact readiness validator set")
    if any(
        record.get("required") is not True or record.get("outcome") != "pass"
        for record in invariants
        if isinstance(record, Mapping)
    ):
        failures.append("deterministic report contains a non-required or non-passing invariant")
    if str(image.get("id") or "").lower() != image_id:
        failures.append("domain report image ID does not match executed immutable image")
    provenance = report.get("provenance_policy")
    if not isinstance(provenance, Mapping) or provenance.get("status") != "enforced":
        failures.append("domain report did not enforce clean promotion provenance")
    failures.extend(validate_domain_calphad_experimental_benchmark(report))
    return failures


def validate_domain_report_against_staged_source(
    report: Mapping[str, Any],
    staged_root: Path,
    *,
    expected_git_sha: str,
    image: ImageInspection,
    expected_baked_materials_path: str,
) -> list[str]:
    failures: list[str] = []
    requirements_path = staged_root / "deploy/docker/materials-requirements.txt"
    invariant_path = (
        staged_root
        / "backend/deepagents_runtime/tests/domain_correctness/test_materials_invariants.py"
    )
    validation_path = (
        staged_root / "backend/deepagents_runtime/src/ultra_deepagents/materials/validation.py"
    )
    calphad_benchmark_validator_path = staged_root / "scripts/calphad_experimental_benchmark.py"
    calphad_benchmark_manifest_path = (
        staged_root / "backend/deepagents_runtime/materials_data/calphad/"
        "experimental_benchmark_manifest.json"
    )
    try:
        requirements_hash = _sha256_file(requirements_path)
        invariant_hash = _sha256_file(invariant_path)
        validation_hash = _sha256_file(validation_path)
        calphad_benchmark_validator_hash = _sha256_file(calphad_benchmark_validator_path)
        calphad_benchmark_manifest_hash = _sha256_file(calphad_benchmark_manifest_path)
    except OSError as exc:
        return [f"cannot bind domain evidence to staged source: {exc}"]
    requirements = report.get("requirements")
    test_source = report.get("test_source")
    git = report.get("git")
    image_record = report.get("image")
    runtime = report.get("runtime")
    pytest_record = report.get("pytest")
    if not all(
        isinstance(record, Mapping)
        for record in (requirements, test_source, git, image_record, runtime, pytest_record)
    ):
        return ["domain report source/provenance binding records are malformed"]
    assert isinstance(requirements, Mapping)
    assert isinstance(test_source, Mapping)
    assert isinstance(git, Mapping)
    assert isinstance(image_record, Mapping)
    assert isinstance(runtime, Mapping)
    assert isinstance(pytest_record, Mapping)
    if (
        requirements.get("path") != "/workspace/deploy/docker/materials-requirements.txt"
        or requirements.get("sha256") != requirements_hash
        or requirements.get("source_sha256") != requirements_hash
    ):
        failures.append("domain requirements evidence differs from retained staged source")
    if (
        test_source.get("path")
        != "/workspace/backend/deepagents_runtime/tests/domain_correctness/test_materials_invariants.py"
        or test_source.get("sha256") != invariant_hash
    ):
        failures.append("domain invariant source evidence differs from retained staged source")
    if (
        git.get("sha") != expected_git_sha
        or git.get("ref") != expected_git_sha
        or git.get("dirty") is not False
    ):
        failures.append("domain Git provenance differs from the expected immutable release")
    if image_record.get("id") != image.image_id or image_record.get("ref") != image.ref:
        failures.append("domain image provenance differs from the executed image")
    validation = runtime.get("materials_validation")
    if (
        not isinstance(validation, Mapping)
        or validation.get("path") != f"{expected_baked_materials_path}/validation.py"
        or validation.get("sha256") != validation_hash
    ):
        failures.append("domain validation module differs from retained staged source")
    calphad_benchmark_wrapper = report.get("calphad_experimental_benchmark")
    calphad_benchmark_report = (
        calphad_benchmark_wrapper.get("report")
        if isinstance(calphad_benchmark_wrapper, Mapping)
        else None
    )
    calphad_benchmark_validator = (
        calphad_benchmark_wrapper.get("validator")
        if isinstance(calphad_benchmark_wrapper, Mapping)
        else None
    )
    calphad_benchmark_manifest = (
        calphad_benchmark_report.get("source_manifest")
        if isinstance(calphad_benchmark_report, Mapping)
        else None
    )
    if (
        not isinstance(calphad_benchmark_validator, Mapping)
        or calphad_benchmark_validator.get("path")
        != "/workspace/scripts/calphad_experimental_benchmark.py"
        or calphad_benchmark_validator.get("sha256") != calphad_benchmark_validator_hash
    ):
        failures.append("domain CALPHAD experimental validator differs from retained staged source")
    if (
        not isinstance(calphad_benchmark_manifest, Mapping)
        or calphad_benchmark_manifest.get("relative_path")
        != (
            "backend/deepagents_runtime/materials_data/calphad/experimental_benchmark_manifest.json"
        )
        or calphad_benchmark_manifest.get("sha256") != calphad_benchmark_manifest_hash
        or calphad_benchmark_manifest.get("size_bytes")
        != calphad_benchmark_manifest_path.stat().st_size
    ):
        failures.append("domain CALPHAD experimental manifest differs from retained staged source")
    calphad_preflight = runtime.get("calphad_runtime_preflight")
    calphad_junit = (
        calphad_preflight.get("junit") if isinstance(calphad_preflight, Mapping) else None
    )
    if (
        not isinstance(calphad_preflight, Mapping)
        or calphad_preflight.get("required") is not True
        or calphad_preflight.get("validated") is not True
        or calphad_preflight.get("path") != "/outputs/calphad-runtime-junit.xml"
        or calphad_preflight.get("core_tests") != REQUIRED_CALPHAD_CORE_TEST_COUNT
        or calphad_preflight.get("typed_cli_tests") != REQUIRED_TYPED_CALPHAD_CLI_TEST_COUNT
        or calphad_preflight.get("required_adversarial_test_names")
        != sorted(REQUIRED_CALPHAD_ADVERSARIAL_TEST_NAMES)
        or not isinstance(calphad_junit, Mapping)
        or calphad_junit.get("tests") != REQUIRED_CALPHAD_RUNTIME_TEST_COUNT
        or any(calphad_junit.get(field) != 0 for field in ("failures", "errors", "skipped"))
    ):
        failures.append("domain CALPHAD runtime preflight evidence is incomplete or stale")
    command = pytest_record.get("command")
    expected_tail = [
        "-m",
        "pytest",
        "/workspace/backend/deepagents_runtime/tests/domain_correctness/test_materials_invariants.py",
        "-q",
        "-ra",
        "--color=no",
        "--tb=short",
        "-p",
        "no:cacheprovider",
        "-o",
        "junit_family=legacy",
        "--junitxml=/outputs/domain/materials-junit.xml",
    ]
    if not isinstance(command, list) or [str(value) for value in command[1:]] != expected_tail:
        failures.append("domain pytest command differs from the fixed isolated contract")
    return failures


def validate_calphad_report(report: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    databases = report.get("databases") if isinstance(report.get("databases"), list) else []
    if report.get("status") != "passed":
        failures.append("embedded CALPHAD probe did not pass")
    if report.get("equilibrium_schema_version") != REQUIRED_EQUILIBRIUM_SCHEMA_VERSION:
        failures.append("embedded CALPHAD runtime does not expose equilibrium schema v2")
    if report.get("source_manifest_sha256") != report.get("embedded_manifest_sha256"):
        failures.append("embedded CALPHAD manifest hash differs from checked-out source")
    if report.get("source_manifest_sha256") != REQUIRED_CALPHAD_MANIFEST_SHA256:
        failures.append("embedded CALPHAD manifest hash differs from the reviewed release contract")
    source_hashes = report.get("materials_source_hashes")
    baked_hashes = report.get("materials_baked_hashes")
    expected_source_hashes = {
        Path(path).name: digest
        for path, digest in REQUIRED_CALPHAD_RELEASE_INPUT_SHA256S.items()
        if path.startswith("backend/deepagents_runtime/src/ultra_deepagents/materials/calphad")
    }
    if not isinstance(source_hashes, Mapping) or not isinstance(baked_hashes, Mapping):
        failures.append("embedded CALPHAD probe lacks source/baked module hashes")
    elif any(
        source_hashes.get(name) != digest or baked_hashes.get(name) != digest
        for name, digest in expected_source_hashes.items()
    ):
        failures.append("embedded CALPHAD source hashes differ from the reviewed release contract")
    if not databases or int(report.get("database_count", 0)) != len(databases):
        failures.append("embedded CALPHAD manifest has no validated databases")
    for record in databases:
        if not isinstance(record, Mapping):
            failures.append("embedded CALPHAD database record is malformed")
            continue
        if re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256") or "")) is None:
            failures.append("embedded CALPHAD database lacks a valid SHA-256")
        if int(record.get("size_bytes", 0)) <= 0:
            failures.append("embedded CALPHAD database is empty")
        if record.get("pycalphad_parse_supported") is not True:
            failures.append("embedded CALPHAD database did not parse with pycalphad")
        if record.get("ultra_inspection_supported") is not True:
            failures.append("embedded CALPHAD database did not parse through Ultra's bounded API")
        declared_format = str(record.get("format") or "").casefold()
        filename = str(record.get("filename") or "")
        if declared_format not in {"tdb", "dat"} or Path(filename).suffix.casefold() != (
            f".{declared_format}"
        ):
            failures.append("embedded CALPHAD database format is unsupported or mismatched")
        pressure_limits = record.get("assessment_pressure_limits_Pa")
        if (
            not isinstance(pressure_limits, list)
            or len(pressure_limits) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in pressure_limits
            )
            or any(not math.isfinite(float(value)) for value in pressure_limits)
            or float(pressure_limits[0]) < 1e-9
            or float(pressure_limits[0]) > float(pressure_limits[1])
            or float(pressure_limits[1]) > 1e12
        ):
            failures.append("embedded CALPHAD database pressure scope is invalid")
    return failures


def validate_calphad_report_against_staged_source(
    report: Mapping[str, Any],
    staged_root: Path,
    *,
    expected_baked_path: str,
) -> list[str]:
    failures: list[str] = []
    try:
        manifest_path = (
            staged_root / "backend/deepagents_runtime/materials_data/calphad/manifest.json"
        )
        manifest = _load_json(manifest_path)
        manifest_hash = _sha256_file(manifest_path)
        if (
            report.get("source_manifest_sha256") != manifest_hash
            or report.get("embedded_manifest_sha256") != manifest_hash
        ):
            failures.append("CALPHAD probe manifest hashes differ from retained staged source")
        if report.get("baked_materials_path") != expected_baked_path:
            failures.append("CALPHAD probe imported materials from a noncanonical image path")

        materials_root = staged_root / "backend/deepagents_runtime/src/ultra_deepagents/materials"
        material_files = sorted(materials_root.glob("*.py"))
        if not material_files:
            raise VerificationError("retained staged materials package is empty")
        material_hashes = {path.name: _sha256_file(path) for path in material_files}
        if (
            report.get("materials_source_hashes") != material_hashes
            or report.get("materials_baked_hashes") != material_hashes
        ):
            failures.append("CALPHAD probe material module hashes differ from staged source")

        manifest_databases = manifest.get("databases")
        probe_databases = report.get("databases")
        if not isinstance(manifest_databases, list) or not isinstance(probe_databases, list):
            raise VerificationError("CALPHAD staged manifest or probe database list is malformed")
        expected_records: dict[str, dict[str, Any]] = {}
        for entry in manifest_databases:
            if not isinstance(entry, Mapping):
                raise VerificationError("CALPHAD staged manifest database row is malformed")
            filename = str(entry.get("filename") or "")
            database_id = str(entry.get("database_id") or "")
            relative_filename = _safe_relative_path(
                filename,
                label="CALPHAD staged database filename",
            )
            if len(relative_filename.parts) != 1 or not database_id:
                raise VerificationError("CALPHAD staged database identity is unsafe")
            database_path = manifest_path.parent / relative_filename
            _require_regular_file(database_path, label="retained staged CALPHAD database")
            actual_hash = _sha256_file(database_path)
            actual_size = database_path.stat().st_size
            if entry.get("sha256") != actual_hash or entry.get("size_bytes") != actual_size:
                failures.append(
                    f"retained staged CALPHAD database differs from manifest: {database_id}"
                )
            if database_id in expected_records:
                raise VerificationError("CALPHAD staged manifest has duplicate database IDs")
            expected_records[database_id] = {
                "database_id": database_id,
                "filename": filename,
                "sha256": actual_hash,
                "size_bytes": actual_size,
                "format": str(entry.get("format") or "").casefold(),
                "assessment_pressure_limits_Pa": entry.get("assessment_pressure_limits_Pa"),
                "elements": sorted(str(value) for value in entry.get("elements", [])),
                "phases": sorted(str(value) for value in entry.get("phases", [])),
                "pycalphad_parse_supported": True,
                "ultra_inspection_supported": True,
            }
        observed_records = {
            str(record.get("database_id") or ""): {
                "database_id": str(record.get("database_id") or ""),
                "filename": str(record.get("filename") or ""),
                "sha256": str(record.get("sha256") or ""),
                "size_bytes": int(record.get("size_bytes", -1)),
                "format": str(record.get("format") or "").casefold(),
                "assessment_pressure_limits_Pa": record.get("assessment_pressure_limits_Pa"),
                "elements": sorted(str(value) for value in record.get("elements", [])),
                "phases": sorted(str(value) for value in record.get("phases", [])),
                "pycalphad_parse_supported": record.get("pycalphad_parse_supported"),
                "ultra_inspection_supported": record.get("ultra_inspection_supported"),
            }
            for record in probe_databases
            if isinstance(record, Mapping)
        }
        if (
            len(observed_records) != len(probe_databases)
            or observed_records != expected_records
            or report.get("database_count") != len(expected_records)
        ):
            failures.append("CALPHAD probe database evidence differs from staged registry bytes")
    except (OSError, TypeError, ValueError, VerificationError) as exc:
        failures.append(str(exc))
    return failures


def _parse_pip_freeze(path: Path) -> dict[str, str]:
    _require_regular_file(path, label="pip freeze")
    packages: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        normalized = re.sub(r"[-_.]+", "-", name.strip().lower())
        if (
            not separator
            or not normalized
            or not version.strip()
            or re.fullmatch(r"[a-z0-9][a-z0-9-]*", normalized) is None
            or normalized in packages
        ):
            raise VerificationError(
                f"pip freeze line {line_number} is not a unique exact package identity"
            )
        packages[normalized] = version.strip()
    if not packages:
        raise VerificationError("pip freeze contains no exact package identities")
    return dict(sorted(packages.items()))


def _normalize_installed_package_inventory(raw_packages: Any) -> list[dict[str, str]]:
    if not isinstance(raw_packages, list) or not raw_packages:
        raise VerificationError("domain report lacks a complete installed package inventory")
    packages: list[dict[str, str]] = []
    normalized_names: list[str] = []
    for raw in raw_packages:
        if not isinstance(raw, Mapping):
            raise VerificationError("domain installed package inventory contains a malformed row")
        name = str(raw.get("name") or "").strip()
        record = {
            "name": name,
            "normalized_name": str(raw.get("normalized_name") or "").strip(),
            "version": str(raw.get("version") or "").strip(),
        }
        expected_normalized = re.sub(r"[-_.]+", "-", name.lower())
        if (
            dict(raw) != record
            or not all(record.values())
            or record["normalized_name"] != expected_normalized
        ):
            raise VerificationError(
                "domain installed package inventory contains invalid identity data"
            )
        packages.append(record)
        normalized_names.append(record["normalized_name"])
    expected_order = sorted(
        packages,
        key=lambda record: (record["normalized_name"], record["name"].lower()),
    )
    if packages != expected_order:
        raise VerificationError("domain installed package inventory is not canonically sorted")
    if len(set(normalized_names)) != len(normalized_names):
        raise VerificationError("domain installed package inventory has duplicate normalized names")
    return packages


def _retain_package_identity(
    domain_report: Mapping[str, Any],
    domain_output: Path,
    output_dir: Path,
    domain_tree: Mapping[str, Any],
) -> dict[str, Any]:
    packages = _normalize_installed_package_inventory(domain_report.get("installed_packages"))
    package_record = _retain_bytes(
        _canonical_json_bytes(packages, newline=True),
        output_dir,
        directory=Path("bundle/environment"),
        stem="installed-packages",
        suffix=".json",
    )
    freeze_record = _retain_file(
        domain_output / "materials-pip-freeze.txt",
        output_dir,
        directory=Path("bundle/environment"),
        stem="materials-pip-freeze",
        suffix=".txt",
    )
    if freeze_record["size_bytes"] <= 0:
        raise VerificationError("retained pip freeze is empty")
    freeze_packages = _parse_pip_freeze(domain_output / "materials-pip-freeze.txt")
    domain_freeze_entries = [
        record
        for record in domain_tree.get("files", [])
        if isinstance(record, Mapping) and record.get("path") == "materials-pip-freeze.txt"
    ]
    if len(domain_freeze_entries) != 1 or (
        domain_freeze_entries[0].get("sha256") != freeze_record["sha256"]
        or domain_freeze_entries[0].get("size_bytes") != freeze_record["size_bytes"]
    ):
        raise VerificationError("standalone pip freeze differs from retained domain evidence")
    installed_direct = domain_report.get("installed_direct")
    expected_pins = domain_report.get("expected_pins")
    runtime = domain_report.get("runtime")
    if not isinstance(installed_direct, Mapping) or not isinstance(expected_pins, Mapping):
        raise VerificationError("domain report lacks direct package pin identity")
    canonical_expected = {
        str(name): str(version) for name, version in sorted(expected_pins.items())
    }
    canonical_direct = {
        str(name): str(version) for name, version in sorted(installed_direct.items())
    }
    if not canonical_expected or canonical_direct != canonical_expected:
        raise VerificationError("domain direct package identities do not match exact release pins")
    inventory_versions = {record["normalized_name"]: record["version"] for record in packages}
    if any(inventory_versions.get(name) != version for name, version in canonical_expected.items()):
        raise VerificationError("complete package inventory disagrees with exact direct pins")
    if any(inventory_versions.get(name) != version for name, version in freeze_packages.items()):
        raise VerificationError("pip freeze disagrees with complete installed package inventory")
    if any(freeze_packages.get(name) != version for name, version in canonical_expected.items()):
        raise VerificationError("pip freeze omits or changes an exact direct release pin")
    if not isinstance(runtime, Mapping):
        raise VerificationError("domain report lacks Python runtime identity")
    return {
        "schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "runtime_image_id": (
            domain_report.get("image", {}).get("id")
            if isinstance(domain_report.get("image"), Mapping)
            else None
        ),
        "package_count": len(packages),
        "installed_packages": packages,
        "installed_packages_file": package_record,
        "pip_freeze": freeze_record,
        "pip_freeze_packages": freeze_packages,
        "domain_tree_pip_freeze": dict(domain_freeze_entries[0]),
        "expected_direct_pins": canonical_expected,
        "installed_direct": canonical_direct,
        "runtime": dict(runtime),
    }


def _required_mapping(container: Mapping[str, Any], key: str, *, label: str) -> Mapping[str, Any]:
    value = container.get(key)
    if not isinstance(value, Mapping):
        raise VerificationError(f"{label}.{key} record is missing or malformed")
    return value


def validate_retained_evidence_bundle(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[str]:
    """Independently rehash every production-full report-relative retained byte."""

    if report.get("scope") != "production-full":
        bundle = report.get("evidence_bundle")
        if isinstance(bundle, Mapping) and bundle.get("promotable") is False:
            return []
        return ["non-production parity evidence must be explicitly nonpromotable"]

    failures: list[str] = []
    try:
        bundle = _required_mapping(report, "evidence_bundle", label="report")
        if int(bundle.get("schema_version", -1)) != EVIDENCE_BUNDLE_SCHEMA_VERSION:
            raise VerificationError("evidence bundle has the wrong schema version")
        if bundle.get("promotable") is not True:
            raise VerificationError("production-full evidence bundle is not marked promotable")
        if (
            report.get("gate") != "production-materials-sandbox-parity"
            or report.get("status") != "passed"
            or report.get("full_production_image_parity") is not True
            or report.get("failures") not in ([], ())
            or SHA_PATTERN.fullmatch(str(report.get("expected_git_sha") or "")) is None
        ):
            raise VerificationError("production-full report identity or success claim is invalid")
        if report.get("calphad_release_contract") != {
            "manifest_sha256": REQUIRED_CALPHAD_MANIFEST_SHA256,
            "release_input_sha256s": dict(sorted(REQUIRED_CALPHAD_RELEASE_INPUT_SHA256S.items())),
            "runtime_test_count": REQUIRED_CALPHAD_RUNTIME_TEST_COUNT,
            "core_runtime_test_count": REQUIRED_CALPHAD_CORE_TEST_COUNT,
            "typed_cli_test_count": REQUIRED_TYPED_CALPHAD_CLI_TEST_COUNT,
            "calphad_tools_test_count": REQUIRED_CALPHAD_TOOLS_TEST_COUNT,
            "required_adversarial_test_names": sorted(REQUIRED_CALPHAD_ADVERSARIAL_TEST_NAMES),
        }:
            raise VerificationError("production CALPHAD release contract is missing or stale")

        release = _required_mapping(bundle, "release", label="evidence_bundle")
        if release.get("promotable") is not True:
            raise VerificationError("retained release evidence is not promotable")
        release_manifest = _required_mapping(release, "manifest", label="release")
        control_binary = _required_mapping(release, "control_binary", label="release")
        frontend_dist = _required_mapping(release, "frontend_dist", label="release")
        failures.extend(
            validate_retained_file_record(
                output_dir, release_manifest, label="retained release manifest"
            )
        )
        failures.extend(
            validate_retained_file_record(
                output_dir, control_binary, label="retained control binary"
            )
        )
        failures.extend(
            validate_retained_tree_record(
                output_dir, frontend_dist, label="retained frontend distribution"
            )
        )
        source = _required_mapping(report, "source", label="report")
        if source.get("kind") != "git_archive_release_manifest":
            failures.append("production-full evidence source is not an extracted release manifest")
        source_artifacts = _required_mapping(source, "release_artifacts", label="source")
        declared_control = _required_mapping(
            source_artifacts, "control_binary", label="source.release_artifacts"
        )
        declared_frontend = _required_mapping(
            source_artifacts, "frontend_dist", label="source.release_artifacts"
        )
        if (
            release_manifest.get("sha256") != source.get("manifest_sha256")
            or control_binary.get("sha256") != declared_control.get("sha256")
            or control_binary.get("size_bytes") != declared_control.get("size_bytes")
        ):
            failures.append("retained release files are not bound to verified source identities")
        if {key: frontend_dist.get(key) for key in ("aggregate_sha256", "file_count", "files")} != {
            key: declared_frontend.get(key) for key in ("aggregate_sha256", "file_count", "files")
        }:
            failures.append("retained frontend closure is not bound to the release manifest")
        manifest_path = _retained_path(
            output_dir,
            release_manifest.get("relative_path"),
            label="retained release manifest",
        )
        manifest_payload = _load_json(manifest_path)
        manifest_source = (
            manifest_payload.get("source")
            if isinstance(manifest_payload.get("source"), Mapping)
            else {}
        )
        manifest_targets = (
            manifest_payload.get("targets")
            if isinstance(manifest_payload.get("targets"), Mapping)
            else {}
        )
        if (
            manifest_payload.get("schema_version") != 1
            or manifest_payload.get("release_sha") != report.get("expected_git_sha")
            or manifest_payload.get("release_sha") != source.get("observed_git_sha")
        ):
            failures.append("retained release manifest is not bound to the expected Git SHA")
        if manifest_source.get("required_materials") != source.get("required_materials"):
            failures.append("retained release manifest source closure differs from the report")
        source_required_materials = _required_mapping(
            source,
            "required_materials",
            label="source",
        )
        failures.extend(
            validate_declared_file_manifest(
                source_required_materials,
                label="retained release required-materials manifest",
            )
        )
        if (
            manifest_targets.get("control_binary") != "bin/ultra-control"
            or manifest_targets.get("frontend_dist") != "frontend/dist"
            or manifest_targets.get("control_binary_identity") != declared_control
            or manifest_targets.get("frontend_dist_identity") != declared_frontend
        ):
            failures.append("retained release manifest target identities differ from the bundle")

        staged_source = _required_mapping(bundle, "staged_source", label="evidence_bundle")
        failures.extend(
            validate_retained_tree_record(output_dir, staged_source, label="retained staged source")
        )
        staged_root = _retained_path(
            output_dir,
            staged_source.get("relative_path"),
            label="retained staged source",
        )
        staged_report = _required_mapping(report, "staged_source", label="report")
        if {key: staged_source.get(key) for key in ("aggregate_sha256", "file_count", "files")} != {
            key: staged_report.get(key) for key in ("aggregate_sha256", "file_count", "files")
        }:
            failures.append("retained staged-source closure differs from the executed manifest")
        staged_paths = {
            str(record.get("path") or "")
            for record in staged_source.get("files", [])
            if isinstance(record, Mapping)
        }
        required_generated = {
            ".ultra-parity/calphad_probe.py",
            "backend/deepagents_runtime/src/ultra_deepagents/__init__.py",
            "backend/deepagents_runtime/src/ultra_deepagents/imaging/__init__.py",
        }
        if not required_generated <= staged_paths:
            failures.append("retained staged source omits its probe or generated import shims")
        if staged_report.get("post_execution_declared_files_unchanged") is not True:
            failures.append("executed staged-source bytes were not proven unchanged")
        copied_release_source = _required_mapping(
            staged_report,
            "copied_release_source",
            label="staged_source",
        )
        required_materials = _required_mapping(source, "required_materials", label="source")
        expected_binding = _bind_copied_source_to_release(
            copied_release_source,
            required_materials,
        )
        if staged_report.get("copied_release_source_binding") != expected_binding:
            failures.append("staged release-source binding record is inconsistent")
        staged_by_path = {
            str(record.get("path") or ""): record
            for record in staged_source.get("files", [])
            if isinstance(record, Mapping)
        }
        for record in copied_release_source.get("files", []):
            if (
                not isinstance(record, Mapping)
                or staged_by_path.get(str(record.get("path") or "")) != record
            ):
                failures.append("retained staged source differs from its copied release subset")
                break
        copied_paths = {
            str(record.get("path") or "")
            for record in copied_release_source.get("files", [])
            if isinstance(record, Mapping)
        }
        matplotlib_paths = {
            "matplotlibrc",
            ".cache/matplotlib/matplotlibrc",
        }
        if staged_paths != copied_paths | required_generated | matplotlib_paths:
            failures.append("retained staged source contains an undeclared input path")
        matplotlibrc = load_staged_matplotlibrc(staged_root).encode("utf-8")
        for path in matplotlib_paths:
            record = staged_by_path.get(path)
            if (
                not isinstance(record, Mapping)
                or record.get("sha256") != _sha256_bytes(matplotlibrc)
                or record.get("size_bytes") != len(matplotlibrc)
            ):
                failures.append("retained staged matplotlib configuration is not release-bound")
        probe_record = staged_by_path.get(".ultra-parity/calphad_probe.py")
        if (
            not isinstance(probe_record, Mapping)
            or probe_record.get("sha256") != _sha256_bytes(_CALPHAD_PROBE.encode("utf-8"))
            or probe_record.get("size_bytes") != len(_CALPHAD_PROBE.encode("utf-8"))
        ):
            failures.append("retained staged CALPHAD probe bytes differ from verifier source")
        for shim in (
            "backend/deepagents_runtime/src/ultra_deepagents/__init__.py",
            "backend/deepagents_runtime/src/ultra_deepagents/imaging/__init__.py",
        ):
            shim_record = staged_by_path.get(shim)
            if (
                not isinstance(shim_record, Mapping)
                or shim_record.get("sha256") != _sha256_bytes(b"")
                or shim_record.get("size_bytes") != 0
            ):
                failures.append("retained staged import shim is not the exact empty adapter")
        post_execution_manifest = staged_report.get("post_execution_manifest")
        if post_execution_manifest != {
            key: staged_source.get(key) for key in ("aggregate_sha256", "file_count", "files")
        }:
            failures.append("post-execution staged-source manifest differs from retained bytes")

        host_source = _required_mapping(bundle, "host_source", label="evidence_bundle")
        failures.extend(
            validate_retained_tree_record(
                output_dir,
                host_source,
                label="retained host CALPHAD source",
            )
        )
        host_source_report = _required_mapping(report, "host_source", label="report")
        if {key: host_source.get(key) for key in ("aggregate_sha256", "file_count", "files")} != {
            key: host_source_report.get(key) for key in ("aggregate_sha256", "file_count", "files")
        }:
            failures.append("retained host source differs from its release-bound manifest")
        host_paths = {
            str(record.get("path") or "")
            for record in host_source.get("files", [])
            if isinstance(record, Mapping)
        }
        required_host_paths = {
            "backend/deepagents_runtime/pyproject.toml",
            "backend/deepagents_runtime/uv.lock",
            "backend/deepagents_runtime/tests/test_calphad_tools.py",
        }
        if not required_host_paths <= host_paths or any(
            path.endswith((".pyc", ".pyo"))
            or "__pycache__" in Path(path).parts
            or Path(path).name == "conftest.py"
            for path in host_paths
        ):
            failures.append("retained host source is incomplete or contains executable extras")
        release_files = required_materials.get("files")
        if not isinstance(release_files, list):
            raise VerificationError("release required-materials file manifest is malformed")
        expected_host_files = [
            record
            for record in release_files
            if isinstance(record, Mapping)
            and (
                str(record.get("path") or "") in required_host_paths
                or str(record.get("path") or "").startswith(
                    "backend/deepagents_runtime/src/ultra_deepagents/"
                )
            )
        ]
        if _manifest_for_entries(expected_host_files) != {
            key: host_source.get(key) for key in ("aggregate_sha256", "file_count", "files")
        }:
            failures.append("retained host source is not the exact release-manifest subset")

        images = _required_mapping(bundle, "image_identity", label="evidence_bundle")
        retained_image_summaries: dict[str, Mapping[str, Any]] = {}
        for role in ("base", "executed"):
            image_record = _required_mapping(images, role, label="image_identity")
            inspect_record = _required_mapping(
                image_record, "docker_inspect", label=f"image_identity.{role}"
            )
            role_failures = validate_retained_file_record(
                output_dir, inspect_record, label=f"retained {role} Docker inspect"
            )
            failures.extend(role_failures)
            if not role_failures:
                raw_path = _retained_path(
                    output_dir,
                    inspect_record.get("relative_path"),
                    label=f"retained {role} Docker inspect",
                )
                raw = _load_json(raw_path)
                summary = _required_mapping(image_record, "summary", label=role)
                retained_image_summaries[role] = summary
                config = raw.get("Config") if isinstance(raw.get("Config"), Mapping) else {}
                labels = config.get("Labels") if isinstance(config.get("Labels"), Mapping) else {}
                normalized_labels = {str(key): str(value) for key, value in sorted(labels.items())}
                if str(raw.get("Id") or "").lower() != summary.get("image_id"):
                    failures.append(f"retained {role} Docker inspect ID differs from summary")
                if normalized_labels != summary.get("labels"):
                    failures.append(f"retained {role} Docker inspect labels differ from summary")
                if normalized_labels.get(
                    "org.opencontainers.image.revision", ""
                ).lower() != summary.get("revision") or normalized_labels.get(
                    "org.opencontainers.image.title", ""
                ) != summary.get("title"):
                    failures.append(
                        f"retained {role} Docker inspect OCI labels differ from summary identity"
                    )
                raw_entrypoint = config.get("Entrypoint") or []
                if not isinstance(raw_entrypoint, list) or [
                    str(part) for part in raw_entrypoint
                ] != summary.get("entrypoint"):
                    failures.append(
                        f"retained {role} Docker inspect entrypoint differs from summary"
                    )
                if str(raw.get("Os") or "") != summary.get("os"):
                    failures.append(f"retained {role} Docker inspect OS differs from summary")
                if str(raw.get("Architecture") or "") != summary.get("architecture"):
                    failures.append(
                        f"retained {role} Docker inspect architecture differs from summary"
                    )
                report_summary = _required_mapping(report, f"{role}_image", label="report")
                comparable_report_summary = {
                    key: report_summary.get(key)
                    for key in (
                        "ref",
                        "image_id",
                        "revision",
                        "title",
                        "entrypoint",
                        "labels",
                        "os",
                        "architecture",
                    )
                }
                if dict(summary) != comparable_report_summary:
                    failures.append(f"retained {role} image summary differs from the report")
                retained_inspection = ImageInspection(
                    ref=str(summary.get("ref") or ""),
                    image_id=str(summary.get("image_id") or ""),
                    revision=str(summary.get("revision") or ""),
                    title=str(summary.get("title") or ""),
                    entrypoint=tuple(str(part) for part in summary.get("entrypoint", [])),
                    labels=normalized_labels,
                    os=str(summary.get("os") or ""),
                    architecture=str(summary.get("architecture") or ""),
                    raw_inspect=raw,
                )
                failures.extend(
                    validate_image(
                        retained_inspection,
                        expected_git_sha=str(report.get("expected_git_sha") or ""),
                        scope="production-full",
                        allow_entrypoint=False,
                    )
                )

        base_summary = retained_image_summaries.get("base", {})
        executed_summary = retained_image_summaries.get("executed", {})
        executed_image_report = _required_mapping(report, "executed_image", label="report")
        if (
            base_summary.get("image_id") != executed_summary.get("image_id")
            or base_summary.get("ref") != executed_summary.get("ref")
            or executed_image_report.get("entrypoint_adapter") is not False
            or executed_image_report.get("base_image_id") != base_summary.get("image_id")
        ):
            failures.append("production base/executed image identity chain is inconsistent")
        sandbox_report = _required_mapping(report, "sandbox", label="report")
        if (
            sandbox_report.get("immutable_image_id") != executed_summary.get("image_id")
            or sandbox_report.get("network_none") is not True
            or sandbox_report.get("rootfs_read_only") is not True
            or sandbox_report.get("capabilities_dropped") is not True
            or sandbox_report.get("no_new_privileges") is not True
            or sandbox_report.get("policy_source") != "exported_worker_environment"
            or sandbox_report.get("pytest_isolation")
            != {
                "config": "/dev/null",
                "conftest_loading": False,
                "plugin_autoload": False,
                "pytest_addopts": "",
                "pytest_plugins": "",
                "rootdir": "/workspace/backend/deepagents_runtime",
            }
        ):
            failures.append("production sandbox/image/isolation policy evidence is inconsistent")

        execution = _required_mapping(bundle, "execution_output", label="evidence_bundle")
        if execution.get("schema_version") != EVIDENCE_BUNDLE_SCHEMA_VERSION:
            failures.append("sandbox output has the wrong evidence schema version")
        if execution.get("capture_mode") != "docker_sandbox_backend_combined_stdout_stderr":
            failures.append("sandbox output capture mode is missing or misleading")
        if execution.get("separate_streams_available") is not False:
            failures.append("sandbox output must truthfully declare combined stream capture")
        combined = _required_mapping(execution, "combined", label="execution_output")
        failures.extend(
            validate_retained_file_record(
                output_dir, combined, label="retained sandbox combined stdout/stderr"
            )
        )
        execution_report = _required_mapping(report, "execution", label="report")
        if (
            execution_report.get("output_sha256") != combined.get("sha256")
            or execution_report.get("output_size_bytes") != combined.get("size_bytes")
            or int(execution_report.get("exit_code", -1)) != 0
            or execution_report.get("truncated") is not False
        ):
            failures.append("sandbox execution summary differs from retained output or success")

        host_output = _required_mapping(bundle, "host_output", label="evidence_bundle")
        if (
            host_output.get("schema_version") != EVIDENCE_BUNDLE_SCHEMA_VERSION
            or host_output.get("capture_mode") != "subprocess_separate_stdout_stderr"
        ):
            failures.append("host output capture mode or schema is invalid")
        for stream in ("stdout", "stderr"):
            record = _required_mapping(host_output, stream, label="host_output")
            failures.extend(
                validate_retained_file_record(output_dir, record, label=f"retained host {stream}")
            )
        host_report = _required_mapping(
            _required_mapping(report, "calphad_tool_orchestration", label="report"),
            "execution",
            label="calphad_tool_orchestration",
        )
        if int(host_report.get("exit_code", -1)) != 0:
            failures.append("host CALPHAD orchestration did not report success")
        if host_report.get("source_isolation") != {
            "config": "/dev/null",
            "conftest_loading": False,
            "plugin_autoload": False,
            "pytest_plugins": "",
            "pythonpath": "retained-host-source/backend/deepagents_runtime/src",
            "uv_sync": False,
        }:
            failures.append("host CALPHAD orchestration source/config isolation is invalid")
        for stream in ("stdout", "stderr"):
            retained_stream = _required_mapping(host_output, stream, label="host_output")
            if host_report.get(f"{stream}_sha256") != retained_stream.get(
                "sha256"
            ) or host_report.get(f"{stream}_size_bytes") != retained_stream.get("size_bytes"):
                failures.append(f"host {stream} summary differs from retained bytes")

        environment = _required_mapping(bundle, "environment", label="evidence_bundle")
        package_file = _required_mapping(
            environment, "installed_packages_file", label="environment"
        )
        freeze_file = _required_mapping(environment, "pip_freeze", label="environment")
        failures.extend(
            validate_retained_file_record(
                output_dir, package_file, label="retained installed package inventory"
            )
        )
        failures.extend(
            validate_retained_file_record(output_dir, freeze_file, label="retained pip freeze")
        )
        package_path = _retained_path(
            output_dir,
            package_file.get("relative_path"),
            label="retained installed package inventory",
        )
        packages = json.loads(package_path.read_text(encoding="utf-8"))
        normalized_packages = _normalize_installed_package_inventory(packages)
        freeze_path = _retained_path(
            output_dir,
            freeze_file.get("relative_path"),
            label="retained pip freeze",
        )
        freeze_packages = _parse_pip_freeze(freeze_path)
        if normalized_packages != environment.get("installed_packages"):
            failures.append("retained installed package inventory differs from its report record")
        if len(packages) != int(environment.get("package_count", -1)):
            failures.append("retained installed package inventory count is inconsistent")
        if not packages or int(freeze_file.get("size_bytes", 0)) <= 0:
            failures.append("retained package inventory or pip freeze is empty")
        expected_direct = environment.get("expected_direct_pins")
        installed_direct = environment.get("installed_direct")
        if (
            not isinstance(expected_direct, Mapping)
            or not expected_direct
            or installed_direct != expected_direct
        ):
            failures.append("retained direct package identities differ from exact release pins")
        inventory_versions = {
            str(record.get("normalized_name") or ""): str(record.get("version") or "")
            for record in normalized_packages
        }
        if isinstance(expected_direct, Mapping) and any(
            inventory_versions.get(str(name)) != str(version)
            for name, version in expected_direct.items()
        ):
            failures.append("retained package inventory disagrees with direct release pins")
        if freeze_packages != environment.get("pip_freeze_packages"):
            failures.append("retained pip freeze parse differs from its package identity record")
        if any(
            inventory_versions.get(name) != version for name, version in freeze_packages.items()
        ):
            failures.append("retained pip freeze disagrees with installed package inventory")
        if isinstance(expected_direct, Mapping) and any(
            freeze_packages.get(str(name)) != str(version)
            for name, version in expected_direct.items()
        ):
            failures.append("retained pip freeze omits or changes a direct release pin")
        executed_image = _required_mapping(report, "executed_image", label="report")
        if environment.get("runtime_image_id") != executed_image.get("image_id"):
            failures.append("installed package inventory is not bound to the executed image")

        domain_tree = _required_mapping(bundle, "domain_tree", label="evidence_bundle")
        failures.extend(
            validate_retained_tree_record(output_dir, domain_tree, label="retained domain evidence")
        )
        domain_paths = {
            str(record.get("path") or "")
            for record in domain_tree.get("files", [])
            if isinstance(record, Mapping)
        }
        required_domain_paths = {
            "calphad-experimental-benchmark.json",
            "materials-domain-gate.json",
            "materials-domain-gate.md",
            "materials-junit.xml",
            "materials-pip-freeze.txt",
            "materials-pytest.stderr.txt",
            "materials-pytest.stdout.txt",
        }
        if domain_paths != required_domain_paths:
            failures.append("retained domain evidence does not have the exact artifact closure")
        domain_freeze_entries = [
            record
            for record in domain_tree.get("files", [])
            if isinstance(record, Mapping) and record.get("path") == "materials-pip-freeze.txt"
        ]
        declared_domain_freeze = environment.get("domain_tree_pip_freeze")
        if (
            len(domain_freeze_entries) != 1
            or not isinstance(declared_domain_freeze, Mapping)
            or dict(domain_freeze_entries[0]) != dict(declared_domain_freeze)
            or domain_freeze_entries[0].get("sha256") != freeze_file.get("sha256")
            or domain_freeze_entries[0].get("size_bytes") != freeze_file.get("size_bytes")
        ):
            failures.append("retained standalone and domain-tree pip freeze bytes differ")
        domain_root = _retained_path(
            output_dir,
            domain_tree.get("relative_path"),
            label="retained domain evidence",
        )
        retained_domain_path = domain_root / "materials-domain-gate.json"
        _require_regular_file(retained_domain_path, label="retained domain report")
        retained_domain = _load_json(retained_domain_path)
        failures.extend(
            validate_domain_junit_binding(domain_root / "materials-junit.xml", retained_domain)
        )
        reported_validators = report.get("required_domain_validators")
        if not isinstance(reported_validators, list):
            raise VerificationError("report required-domain validator identities are malformed")
        required_validators = list(load_required_domain_validator_ids(staged_root))
        if reported_validators != required_validators:
            failures.append("reported domain validator identities differ from retained source")
        failures.extend(
            validate_domain_report(
                retained_domain,
                image_id=str(executed_image.get("image_id") or ""),
                required_validator_ids=required_validators,
            )
        )
        failures.extend(
            validate_domain_report_against_staged_source(
                retained_domain,
                staged_root,
                expected_git_sha=str(report.get("expected_git_sha") or ""),
                image=ImageInspection(
                    ref=str(executed_image.get("ref") or ""),
                    image_id=str(executed_image.get("image_id") or ""),
                    revision=str(executed_image.get("revision") or ""),
                    title=str(executed_image.get("title") or ""),
                    entrypoint=tuple(str(part) for part in executed_image.get("entrypoint", [])),
                ),
                expected_baked_materials_path=("/opt/ultra-runtime/ultra_deepagents/materials"),
            )
        )
        domain_report_record = _required_mapping(report, "domain_gate", label="report")
        if (
            retained_domain != domain_report_record.get("report")
            or _sha256_file(retained_domain_path) != domain_report_record.get("sha256")
            or retained_domain.get("installed_packages") != environment.get("installed_packages")
            or retained_domain.get("installed_direct") != environment.get("installed_direct")
            or retained_domain.get("expected_pins") != environment.get("expected_direct_pins")
        ):
            failures.append("retained domain/package evidence differs from the parity report")

        results = _required_mapping(bundle, "results", label="evidence_bundle")
        result_records: dict[str, Mapping[str, Any]] = {}
        for name in ("calphad_probe", "calphad_runtime_junit", "calphad_tools_junit"):
            record = _required_mapping(results, name, label="results")
            result_records[name] = record
            failures.extend(
                validate_retained_file_record(output_dir, record, label=f"retained {name}")
            )
        calphad_probe_path = _retained_path(
            output_dir,
            result_records["calphad_probe"].get("relative_path"),
            label="retained CALPHAD probe",
        )
        retained_calphad = _load_json(calphad_probe_path)
        failures.extend(validate_calphad_report(retained_calphad))
        failures.extend(
            validate_calphad_report_against_staged_source(
                retained_calphad,
                staged_root,
                expected_baked_path="/opt/ultra-runtime/ultra_deepagents/materials",
            )
        )
        calphad_record = _required_mapping(report, "calphad", label="report")
        if retained_calphad != calphad_record.get("report") or _sha256_file(
            calphad_probe_path
        ) != calphad_record.get("sha256"):
            failures.append("retained CALPHAD probe differs from the parity report")

        runtime_junit_path = _retained_path(
            output_dir,
            result_records["calphad_runtime_junit"].get("relative_path"),
            label="retained CALPHAD runtime JUnit",
        )
        runtime_summary, runtime_failures = validate_calphad_runtime_junit(runtime_junit_path)
        failures.extend(runtime_failures)
        runtime_record = _required_mapping(report, "calphad_runtime", label="report")
        if runtime_summary != runtime_record.get("junit") or _sha256_file(
            runtime_junit_path
        ) != runtime_record.get("sha256"):
            failures.append("retained CALPHAD runtime JUnit differs from the parity report")

        tools_junit_path = _retained_path(
            output_dir,
            result_records["calphad_tools_junit"].get("relative_path"),
            label="retained CALPHAD tools JUnit",
        )
        tools_summary, tools_failures = validate_calphad_tools_junit(tools_junit_path)
        failures.extend(tools_failures)
        tools_record = _required_mapping(report, "calphad_tool_orchestration", label="report")
        if tools_summary != tools_record.get("junit") or _sha256_file(
            tools_junit_path
        ) != tools_record.get("sha256"):
            failures.append("retained CALPHAD tools JUnit differs from the parity report")
    except (
        json.JSONDecodeError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        VerificationError,
    ) as exc:
        failures.append(str(exc))
    return failures


def write_content_addressed_report(output_dir: Path, report: Mapping[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("production-materials-sandbox-parity-*.json"):
        stale.unlink()
    payload = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False).encode(
            "utf-8"
        )
        + b"\n"
    )
    digest = _sha256_bytes(payload)
    path = output_dir / f"production-materials-sandbox-parity-{digest}.json"
    path.write_bytes(payload)
    return path


def run_verification(
    args: argparse.Namespace,
    *,
    inspector: ImageInspector = inspect_image,
    backend_factory: BackendFactory = _real_backend_factory,
    adapter_builder: Callable[..., ImageInspection] = _prepare_entrypoint_adapter,
    host_suite_runner: HostSuiteRunner = run_host_calphad_tools_suite,
) -> tuple[int, Path]:
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if args.scope == "production-full" and (
        output_dir == repo_root
        or output_dir.is_relative_to(repo_root)
        or repo_root.is_relative_to(output_dir)
    ):
        raise VerificationError(
            "production release root and parity output directory must be disjoint"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_domain = output_dir / "domain"
    if stale_domain.exists():
        shutil.rmtree(stale_domain)
    (output_dir / CALPHAD_REPORT).unlink(missing_ok=True)
    (output_dir / CALPHAD_RUNTIME_JUNIT).unlink(missing_ok=True)
    (output_dir / CALPHAD_TOOLS_JUNIT).unlink(missing_ok=True)
    failures: list[str] = []
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "gate": "production-materials-sandbox-parity",
        "scope": args.scope,
        "claim": (
            "full production DockerSandboxBackend image parity"
            if args.scope == "production-full"
            else "pinned lean materials source contract; not a production-image claim"
        ),
        "generated_at_utc": _utc_now(),
        "expected_git_sha": args.expected_git_sha,
        "calphad_release_contract": {
            "manifest_sha256": REQUIRED_CALPHAD_MANIFEST_SHA256,
            "release_input_sha256s": dict(sorted(REQUIRED_CALPHAD_RELEASE_INPUT_SHA256S.items())),
            "runtime_test_count": REQUIRED_CALPHAD_RUNTIME_TEST_COUNT,
            "core_runtime_test_count": REQUIRED_CALPHAD_CORE_TEST_COUNT,
            "typed_cli_test_count": REQUIRED_TYPED_CALPHAD_CLI_TEST_COUNT,
            "calphad_tools_test_count": REQUIRED_CALPHAD_TOOLS_TEST_COUNT,
            "required_adversarial_test_names": sorted(REQUIRED_CALPHAD_ADVERSARIAL_TEST_NAMES),
        },
        "failures": failures,
    }
    evidence_bundle: dict[str, Any] = {
        "schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "promotable": False,
    }
    report["evidence_bundle"] = evidence_bundle

    try:
        source_evidence = _verify_source_revision(repo_root, args.expected_git_sha)
        report["source"] = source_evidence
        report["verified_release_artifacts"] = source_evidence.get("release_artifacts")
        retained_release = _retain_release_bundle(
            repo_root,
            output_dir,
            source_evidence,
            scope=args.scope,
        )
        evidence_bundle["release"] = retained_release
        required_validator_ids = load_required_domain_validator_ids(repo_root)
        report["required_domain_validators"] = list(required_validator_ids)
        base = inspector(args.image)
        report["base_image"] = _image_summary(base)
        failures.extend(
            validate_image(
                base,
                expected_git_sha=args.expected_git_sha,
                scope=args.scope,
                allow_entrypoint=(
                    args.scope == "ci-pinned-materials" and args.prepare_entrypoint_adapter
                ),
            )
        )
        if failures:
            raise VerificationError("image/source preconditions failed")

        executed = base
        if base.entrypoint:
            if not args.prepare_entrypoint_adapter or args.scope != "ci-pinned-materials":
                raise VerificationError("only CI's pinned lean image may use an entrypoint adapter")
            executed = adapter_builder(base, inspector=inspector)
            adapter_failures = validate_image(
                executed,
                expected_git_sha=args.expected_git_sha,
                scope=args.scope,
                allow_entrypoint=False,
            )
            failures.extend(adapter_failures)
            if failures:
                raise VerificationError("entrypoint adapter validation failed")
        report["executed_image"] = {
            **_image_summary(executed),
            "entrypoint_adapter": executed.image_id != base.image_id,
            "base_image_id": base.image_id,
        }
        image_identity = {
            "schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
            "base": _retain_image_inspection(base, output_dir, role="base"),
            "executed": _retain_image_inspection(executed, output_dir, role="executed"),
        }
        report["image_identity"] = image_identity
        evidence_bundle["image_identity"] = image_identity

        policy = policy_for_scope(args)

        with tempfile.TemporaryDirectory(prefix="ultra-materials-parity-") as temp:
            workspace = Path(temp) / "workspace"
            outputs = Path(temp) / "outputs"
            workspace.mkdir()
            outputs.mkdir()
            copied_source = _stage_source(repo_root, workspace)
            probe_path = _write_probe(workspace)
            required_materials = source_evidence.get("required_materials")
            if not isinstance(required_materials, Mapping):
                raise VerificationError("verified source manifest is missing")
            copied_source_binding = _bind_copied_source_to_release(
                copied_source,
                required_materials,
            )
            requirements_sha256 = _sha256_file(
                workspace / "deploy/docker/materials-requirements.txt"
            )
            backend = backend_factory(workspace, outputs, executed.image_id, policy)
            host_source_workspace = Path(temp) / "host-source"
            host_source_workspace.mkdir()
            host_source_manifest = _stage_host_tool_source(
                repo_root,
                host_source_workspace,
                required_materials,
            )
            retained_host_source = _retain_tree(
                host_source_workspace,
                output_dir,
                directory=Path("bundle/host-source"),
                stem="host-source",
                label="host CALPHAD tool source",
            )
            host_source_root = _retained_path(
                output_dir,
                retained_host_source["relative_path"],
                label="retained host CALPHAD source",
            )
            report["host_source"] = {
                "schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
                **host_source_manifest,
                "release_source_binding": "exact_manifest_subset",
                "retained_tree": retained_host_source,
            }
            evidence_bundle["host_source"] = retained_host_source
            staged_manifest = _strict_tree_manifest(
                workspace,
                label="complete staged sandbox source",
            )
            retained_staged_source = _retain_tree(
                workspace,
                output_dir,
                directory=Path("bundle/staged"),
                stem="staged-source",
                label="complete staged sandbox source",
            )
            report["staged_source"] = {
                "schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
                **staged_manifest,
                "copied_release_source": copied_source,
                "copied_release_source_binding": copied_source_binding,
                "probe_path": probe_path.relative_to(workspace).as_posix(),
                "retained_tree": retained_staged_source,
            }
            evidence_bundle["staged_source"] = retained_staged_source
            command = _execution_command(
                expected_git_sha=args.expected_git_sha,
                image=executed,
                requirements_sha256=requirements_sha256,
                scope=args.scope,
            )
            docker_command = backend.build_docker_command(command)
            failures.extend(
                validate_backend_command(
                    docker_command,
                    image_id=executed.image_id,
                    policy=policy,
                    workspace=workspace,
                    outputs=outputs,
                    expected_command=command,
                )
            )
            if failures:
                raise VerificationError("DockerSandboxBackend security contract failed")
            response = backend.execute(command)
            combined_output = response.output.encode("utf-8", "replace")
            retained_combined_output = _retain_bytes(
                combined_output,
                output_dir,
                directory=Path("bundle/logs"),
                stem="sandbox-combined-stdout-stderr",
                suffix=".log",
            )
            execution_output = {
                "schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
                "capture_mode": "docker_sandbox_backend_combined_stdout_stderr",
                "separate_streams_available": False,
                "combined": retained_combined_output,
            }
            evidence_bundle["execution_output"] = execution_output
            report["sandbox"] = {
                **asdict(policy),
                "backend": "DockerSandboxBackend",
                "network_none": policy.network == "none",
                "rootfs_read_only": "--read-only" in docker_command,
                "capabilities_dropped": "--cap-drop" in docker_command,
                "no_new_privileges": "no-new-privileges" in docker_command,
                "immutable_image_id": executed.image_id,
                "policy_source": policy.source,
                "pytest_isolation": {
                    "config": "/dev/null",
                    "conftest_loading": False,
                    "plugin_autoload": False,
                    "pytest_addopts": "",
                    "pytest_plugins": "",
                    "rootdir": "/workspace/backend/deepagents_runtime",
                },
            }
            report["execution"] = {
                "exit_code": int(response.exit_code),
                "truncated": bool(response.truncated),
                "output_size_bytes": len(combined_output),
                "output_sha256": _sha256_bytes(combined_output),
                "output_evidence": execution_output,
            }
            if int(response.exit_code) != 0:
                failures.append(f"DockerSandboxBackend execution exited {response.exit_code}")
            if response.truncated:
                failures.append("DockerSandboxBackend execution output was truncated")
            post_execution_staged, runtime_scratch = _post_execution_workspace_evidence(
                workspace,
                staged_manifest,
            )
            staged_unchanged = post_execution_staged == staged_manifest
            report["staged_source"]["post_execution_declared_files_unchanged"] = staged_unchanged
            report["staged_source"]["post_execution_manifest"] = post_execution_staged
            report["staged_source"]["runtime_scratch"] = {
                "excluded_from_source_closure": True,
                **runtime_scratch,
            }
            if not staged_unchanged:
                failures.append(
                    "sandbox execution changed the staged-source closure outside declared scratch"
                )

            domain_path = outputs / DOMAIN_REPORT
            calphad_path = outputs / CALPHAD_REPORT
            calphad_runtime_junit_path = outputs / CALPHAD_RUNTIME_JUNIT
            result_evidence: dict[str, Any] = {}
            if not calphad_runtime_junit_path.is_file():
                failures.append("non-skipping CALPHAD runtime JUnit evidence is missing")
            else:
                calphad_runtime_summary, calphad_runtime_failures = validate_calphad_runtime_junit(
                    calphad_runtime_junit_path
                )
                failures.extend(calphad_runtime_failures)
                retained_runtime_junit = _retain_file(
                    calphad_runtime_junit_path,
                    output_dir,
                    directory=Path("bundle/results"),
                    stem="calphad-runtime-junit",
                    suffix=".xml",
                )
                result_evidence["calphad_runtime_junit"] = retained_runtime_junit
                report["calphad_runtime"] = {
                    "relative_path": CALPHAD_RUNTIME_JUNIT.as_posix(),
                    "sha256": _sha256_file(calphad_runtime_junit_path),
                    "junit": calphad_runtime_summary,
                    "required_core_test_names": list(REQUIRED_CALPHAD_CORE_TEST_NAMES),
                    "required_typed_cli_test_names": list(REQUIRED_TYPED_CALPHAD_CLI_TEST_NAMES),
                    "retained": retained_runtime_junit,
                }
            if not domain_path.is_file():
                failures.append("ordinary deterministic domain-gate evidence is missing")
            else:
                domain = _load_json(domain_path)
                failures.extend(
                    validate_domain_report(
                        domain,
                        image_id=executed.image_id,
                        required_validator_ids=required_validator_ids,
                    )
                )
                failures.extend(
                    validate_domain_calphad_experimental_benchmark(
                        domain,
                        retained_path=outputs / CALPHAD_EXPERIMENTAL_REPORT,
                    )
                )
                failures.extend(
                    validate_domain_report_against_staged_source(
                        domain,
                        workspace,
                        expected_git_sha=args.expected_git_sha,
                        image=executed,
                        expected_baked_materials_path=(
                            "/opt/ultra-runtime/ultra_deepagents/materials"
                            if args.scope == "production-full"
                            else "/opt/ultra/src/ultra_deepagents/materials"
                        ),
                    )
                )
                retained_domain_tree = _retain_tree(
                    outputs / "domain",
                    output_dir,
                    directory=Path("bundle/domain"),
                    stem="domain-evidence",
                    label="deterministic materials domain evidence",
                )
                package_identity = _retain_package_identity(
                    domain,
                    outputs / "domain",
                    output_dir,
                    retained_domain_tree,
                )
                evidence_bundle["domain_tree"] = retained_domain_tree
                evidence_bundle["environment"] = package_identity
                report["domain_gate"] = {
                    "relative_path": DOMAIN_REPORT.as_posix(),
                    "sha256": _sha256_file(domain_path),
                    "report": domain,
                    "retained_tree": retained_domain_tree,
                    "package_identity": package_identity,
                }
            if not calphad_path.is_file():
                failures.append("embedded CALPHAD manifest/hash/parse evidence is missing")
            else:
                calphad = _load_json(calphad_path)
                failures.extend(validate_calphad_report(calphad))
                failures.extend(
                    validate_calphad_report_against_staged_source(
                        calphad,
                        workspace,
                        expected_baked_path=(
                            "/opt/ultra-runtime/ultra_deepagents/materials"
                            if args.scope == "production-full"
                            else "/opt/ultra/src/ultra_deepagents/materials"
                        ),
                    )
                )
                retained_calphad = _retain_file(
                    calphad_path,
                    output_dir,
                    directory=Path("bundle/results"),
                    stem="calphad-embedded-probe",
                    suffix=".json",
                )
                result_evidence["calphad_probe"] = retained_calphad
                report["calphad"] = {
                    "relative_path": CALPHAD_REPORT.as_posix(),
                    "sha256": _sha256_file(calphad_path),
                    "report": calphad,
                    "retained": retained_calphad,
                }

            retained = output_dir / "domain"
            if retained.exists():
                shutil.rmtree(retained)
            if (outputs / "domain").is_dir():
                shutil.copytree(outputs / "domain", retained)
            if calphad_path.is_file():
                shutil.copyfile(calphad_path, output_dir / CALPHAD_REPORT)
            if calphad_runtime_junit_path.is_file():
                shutil.copyfile(
                    calphad_runtime_junit_path,
                    output_dir / CALPHAD_RUNTIME_JUNIT,
                )
            evidence_bundle["results"] = result_evidence

        calphad_tools_junit_path = output_dir / CALPHAD_TOOLS_JUNIT
        host_execution = dict(host_suite_runner(host_source_root, calphad_tools_junit_path))
        stdout_text = host_execution.pop("stdout_text", None)
        stderr_text = host_execution.pop("stderr_text", None)
        if not isinstance(stdout_text, str) or not isinstance(stderr_text, str):
            raise VerificationError(
                "CALPHAD host-tool runner did not return retained stdout/stderr bytes"
            )
        retained_host_stdout = _retain_bytes(
            stdout_text.encode("utf-8", "replace"),
            output_dir,
            directory=Path("bundle/logs"),
            stem="calphad-tools-host-stdout",
            suffix=".log",
        )
        retained_host_stderr = _retain_bytes(
            stderr_text.encode("utf-8", "replace"),
            output_dir,
            directory=Path("bundle/logs"),
            stem="calphad-tools-host-stderr",
            suffix=".log",
        )
        host_output = {
            "schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
            "capture_mode": "subprocess_separate_stdout_stderr",
            "stdout": retained_host_stdout,
            "stderr": retained_host_stderr,
        }
        evidence_bundle["host_output"] = host_output
        for stream, retained in (
            ("stdout", retained_host_stdout),
            ("stderr", retained_host_stderr),
        ):
            if (
                int(host_execution.get(f"{stream}_size_bytes", -1)) != retained["size_bytes"]
                or host_execution.get(f"{stream}_sha256") != retained["sha256"]
            ):
                failures.append(f"CALPHAD host-tool {stream} summary differs from retained bytes")
        if int(host_execution.get("exit_code", -1)) != 0:
            failures.append("CALPHAD host-tool orchestration pytest exit was nonzero")
        if not calphad_tools_junit_path.is_file():
            failures.append("CALPHAD host-tool orchestration JUnit evidence is missing")
        else:
            calphad_tools_summary, calphad_tools_failures = validate_calphad_tools_junit(
                calphad_tools_junit_path
            )
            failures.extend(calphad_tools_failures)
            retained_tools_junit = _retain_file(
                calphad_tools_junit_path,
                output_dir,
                directory=Path("bundle/results"),
                stem="calphad-tools-junit",
                suffix=".xml",
            )
            results = evidence_bundle.get("results")
            if not isinstance(results, dict):
                raise VerificationError("CALPHAD result evidence bundle is missing")
            results["calphad_tools_junit"] = retained_tools_junit
            report["calphad_tool_orchestration"] = {
                "scope": "host-worker-runtime-orchestration-contract",
                "relative_path": CALPHAD_TOOLS_JUNIT.as_posix(),
                "sha256": _sha256_file(calphad_tools_junit_path),
                "junit": calphad_tools_summary,
                "execution": {**host_execution, "output_evidence": host_output},
                "required_test_names": list(REQUIRED_CALPHAD_TOOL_TEST_NAMES),
                "retained": retained_tools_junit,
                "binding": {
                    "git_sha": args.expected_git_sha,
                    "runtime_image_id": executed.image_id,
                    "source_kind": source_evidence.get("kind"),
                    "release_artifacts": source_evidence.get("release_artifacts"),
                },
            }
    except (
        ImportError,
        KeyError,
        OSError,
        subprocess.SubprocessError,
        TypeError,
        VerificationError,
        ValueError,
    ) as exc:
        message = str(exc).strip() or type(exc).__name__
        if message and message not in failures:
            failures.append(message)

    production_candidate = args.scope == "production-full" and not failures
    evidence_bundle["promotable"] = production_candidate
    report["status"] = "passed" if not failures else "failed"
    # This is only one input to the final readiness aggregator.  The lean CI
    # source contract must never be interpreted as production-image parity.
    report["full_production_image_parity"] = production_candidate
    if production_candidate:
        failures.extend(validate_retained_evidence_bundle(report, output_dir))
        evidence_bundle["promotable"] = not failures
        report["status"] = "passed" if not failures else "failed"
        report["full_production_image_parity"] = not failures
    path = write_content_addressed_report(output_dir, report)
    return (0 if not failures else 1), path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--image", required=True)
    parser.add_argument("--expected-git-sha", required=True, type=str.lower)
    parser.add_argument("--scope", choices=sorted(SCOPES), required=True)
    parser.add_argument("--output-dir", default=".tmp/materials-production-parity")
    parser.add_argument("--prepare-entrypoint-adapter", action="store_true")
    parser.add_argument("--cpus", type=float, default=2.0)
    parser.add_argument("--memory", default="8g")
    parser.add_argument("--pids-limit", type=int, default=512)
    parser.add_argument("--shm-size", default="1g")
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--output-limit-bytes", type=int, default=8 * 1024 * 1024)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if SHA_PATTERN.fullmatch(args.expected_git_sha) is None:
        parser.error("--expected-git-sha must be a lowercase 40-character Git SHA")
    if args.scope != "production-full":
        if args.cpus <= 0 or args.pids_limit <= 0 or args.timeout_seconds <= 0:
            parser.error("CPU, PID, and timeout bounds must be positive")
        if args.output_limit_bytes <= 0 or not str(args.memory).strip():
            parser.error("memory and output bounds must be positive/non-empty")
    try:
        status, path = run_verification(args)
    except VerificationError as exc:
        print(f"Production materials sandbox parity refused: {exc}", file=sys.stderr)
        return 1
    print(f"Production materials sandbox parity report: {path}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
