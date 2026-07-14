#!/usr/bin/env python3
"""Aggregate every materials release lane into one fail-closed decision.

This script deliberately does not run either scientific evaluator. It consumes
their final reports, requires the exact production-sandbox parity attestation,
recomputes all score/coverage arithmetic exposed by those reports, rehashes the
checked-out benchmark and repository evidence, validates the designated live
traces, and emits an evidence-qualified promotion candidate. A separate
GitHub/Sigstore-attested sanitized promotion envelope is required before any
artifact may claim full materials production readiness.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

SCHEMA_VERSION = "1"
OFFICIAL_REVISION = "1803a6abfe23a9da56c894076c59117873b758ff"
OFFICIAL_MANIFEST_SHA256 = "c70c9c5b1d085643372728e4017c28282e190cd452afa2f5e7fd3366e1a9528e"
PARENTS_PER_TRIAL = 49
SUBTASKS_PER_TRIAL = 138
TRIAL_COUNT = 3
RUNNABLE_DENOMINATOR = PARENTS_PER_TRIAL * TRIAL_COUNT
SCIENTIFIC_DENOMINATOR = SUBTASKS_PER_TRIAL * TRIAL_COUNT
RUNNABLE_MINIMUM = math.ceil(RUNNABLE_DENOMINATOR * 0.80)
SCIENTIFIC_MINIMUM = math.ceil(SCIENTIFIC_DENOMINATOR * 0.60)
PER_TRIAL_RUNNABLE_MINIMUM = math.ceil(PARENTS_PER_TRIAL * 0.80)
PER_TRIAL_SCIENTIFIC_MINIMUM = math.ceil(SUBTASKS_PER_TRIAL * 0.60)
MATERIALS_CLEANROOM_PROFILE = "materials_cleanroom_v1"
MAX_REPORT_BYTES = 128 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

REQUIRED_MATTOOLS_HARD_GATES = (
    "official_snapshot",
    "license_attested",
    "checkpoint_evidence_integrity",
    "checkpoint_non_erasure_integrity",
    "replay_terminal_evidence_integrity",
    "three_trial_completeness",
    "actual_ultra_control_plane_path",
    "actual_model_provider_provenance",
    "production_execute_tool_evidence",
    "server_authorized_cleanroom_profile",
    "worker_enforced_cleanroom_profile",
    "production_execute_runtime_image_attestation",
    "required_solution_artifacts",
    "expected_values_and_verifiers_isolated",
    "official_evaluator_environment_exact",
    "evaluator_independent_from_production",
    "immediate_replay_reproducible",
    "external_sandbox_isolation_evidence",
    "provenance_complete",
    "ultra_worktree_clean",
    "per_trial_mattools_function_runnable",
    "per_trial_strict_scientific_task_success",
    "mattools_function_runnable",
    "mattools_task_success",
    "strict_scientific_task_success",
)

REQUIRED_LIVE_SIGNALS = (
    "code_execution",
    "durable_validation_artifact",
    "hashed_evidence",
    "materials_skill",
    "no_silent_success",
    "remote_mutation_aligned",
    "remote_mutation_scope_valid",
    "first_party_scientific_record_valid",
    "terminal_ok",
    "validation_present",
    "validation_valid",
)

REMOTE_MUTATION_TOKENS = (
    "upload",
    "create_dataset",
    "delete",
    "update",
    "publish",
    "move",
    "copy",
    "write",
    "patch",
)

EXPECTED_EVALUATOR_PACKAGES = {
    "pymatgen": "2024.8.9",
    "pymatgen-analysis-defects": "2024.7.19",
}
EXPECTED_HOST_VALIDATOR_PACKAGES = {
    "docker": "7.1.0",
    "numpy": "1.26.4",
    "openpyxl": "3.1.5",
    "pandas": "2.2.2",
    "pymatgen": "2024.8.9",
}
EXPECTED_HOST_VALIDATOR_PYTHON = "3.11.9"
EXPECTED_HOST_VALIDATOR_IMPLEMENTATION = "CPython"
EXPECTED_HOST_VALIDATOR_LOCK_SHA256 = (
    "285c60dd8c4ef501fb7515d4bb28cb7072b55f63c074e53e990ce99d97658b1c"
)
EXPECTED_HOST_VALIDATOR_INPUT_SHA256 = (
    "1b61090c00858d09f5d34852425807873ff7ce759babd685eb1d67f0ce23cacd"
)
EXPECTED_EVALUATOR_LOCK_PATH = "deploy/docker/mattools-evaluator-linux-arm64-lock.json"
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

REQUIRED_CALPHAD_LEDGER_SOURCE_FILES = (
    ".github/workflows/materials-domain-gate.yml",
    ".github/workflows/release-artifacts.yml",
    "Makefile",
    "backend/controlplane/go.mod",
    "backend/controlplane/go.sum",
    "backend/controlplane/api/openapi.yaml",
    "backend/controlplane/internal/app/app.go",
    "backend/controlplane/internal/app/app_test.go",
    "backend/controlplane/internal/config/config.go",
    "backend/controlplane/internal/config/config_test.go",
    "backend/controlplane/internal/domain/calphad.go",
    "backend/controlplane/internal/httpapi/calphad_evidence.go",
    "backend/controlplane/internal/httpapi/calphad_evidence_test.go",
    "backend/controlplane/internal/httpapi/calphad_scientific_evidence.go",
    "backend/controlplane/internal/httpapi/calphad_resource_test.go",
    "backend/controlplane/internal/httpapi/handlers.go",
    "backend/controlplane/internal/httpapi/handlers_calphad.go",
    "backend/controlplane/internal/httpapi/handlers_calphad_test.go",
    "backend/controlplane/internal/httpapi/resource_staging_security.go",
    "backend/controlplane/internal/httpapi/resource_staging_security_test.go",
    "backend/controlplane/internal/openapi/generated.gen.go",
    "backend/controlplane/internal/openapi/calphad_failure_test.go",
    "backend/controlplane/internal/store/calphad_ledger.go",
    "backend/controlplane/internal/store/calphad_ledger_test.go",
    "backend/controlplane/internal/store/memory.go",
    "backend/controlplane/internal/store/schema.sql",
    "backend/controlplane/internal/store/schema_apply.go",
    "backend/controlplane/internal/store/schema_apply_test.go",
    "backend/controlplane/internal/store/schema_check.go",
    "backend/controlplane/internal/store/schema_check_test.go",
    "backend/controlplane/internal/runcontrol/service.go",
    "backend/controlplane/internal/runcontrol/service_test.go",
    "backend/controlplane/migrations/000008_calphad_revision_ledger.down.sql",
    "backend/controlplane/migrations/000008_calphad_revision_ledger.up.sql",
    "deploy/env/ultra-backend.env.example",
    "deploy/env/ultra-migration.env.example",
    "backend/deepagents_runtime/pyproject.toml",
    "backend/deepagents_runtime/uv.lock",
    "backend/deepagents_runtime/materials_data/calphad/README.md",
    "backend/deepagents_runtime/materials_data/calphad/alcow_CALPHAD-2017-Wang.tdb",
    "backend/deepagents_runtime/materials_data/calphad/manifest.json",
    "backend/deepagents_runtime/src/ultra_deepagents/agent.py",
    "backend/deepagents_runtime/src/ultra_deepagents/code_execution/docker.py",
    "backend/deepagents_runtime/src/ultra_deepagents/context.py",
    "backend/deepagents_runtime/src/ultra_deepagents/materials/calphad.py",
    "backend/deepagents_runtime/src/ultra_deepagents/materials/calphad_cli.py",
    "backend/deepagents_runtime/src/ultra_deepagents/materials/calphad_tools.py",
    "backend/deepagents_runtime/src/ultra_deepagents/nats_worker.py",
    "backend/deepagents_runtime/src/ultra_deepagents/runner.py",
    "backend/deepagents_runtime/src/ultra_deepagents/schemas.py",
    "backend/deepagents_runtime/tests/test_calphad_tools.py",
    "backend/deepagents_runtime/tests/test_calphad_cli.py",
    "backend/deepagents_runtime/tests/test_calphad_runtime.py",
    "backend/deepagents_runtime/tests/test_worker_transport.py",
    "scripts/calphad_ledger_gate.py",
    "scripts/deploy_ultra_control_stack.sh",
    "scripts/materials_readiness_gate.py",
    "tests/test_control_stack_deploy_assets.py",
)
REQUIRED_CALPHAD_CROSS_LANGUAGE_SOURCE_FILES = (
    ".github/workflows/materials-domain-gate.yml",
    "Makefile",
    "backend/controlplane/integration/calphad_cross_language_http_test.go",
    "backend/controlplane/integration/calphad_cross_language_test.go",
    "backend/controlplane/internal/domain/calphad.go",
    "backend/controlplane/internal/httpapi/calphad_evidence.go",
    "backend/controlplane/internal/httpapi/calphad_scientific_evidence.go",
    "backend/controlplane/internal/httpapi/handlers_calphad.go",
    "backend/controlplane/internal/store/calphad_ledger.go",
    "backend/controlplane/internal/store/schema.sql",
    "backend/controlplane/internal/store/schema_apply.go",
    "backend/controlplane/internal/store/schema_check.go",
    "backend/deepagents_runtime/src/ultra_deepagents/materials/calphad.py",
    "backend/deepagents_runtime/src/ultra_deepagents/materials/calphad_cli.py",
    "backend/deepagents_runtime/materials_data/calphad/manifest.json",
    "backend/deepagents_runtime/materials_data/calphad/alcow_CALPHAD-2017-Wang.tdb",
    "deploy/docker/deepagents-sandbox.Dockerfile",
    "deploy/docker/materials-requirements.txt",
    "scripts/calphad_cross_language_gate.py",
    "tests/test_calphad_cross_language_gate.py",
)
REQUIRED_CALPHAD_CROSS_LANGUAGE_CHECKS = (
    "actual_typed_cli_artifacts",
    "pycalphad_0_11_2",
    "live_go_http_callback",
    "live_postgres",
    "role_separated_postgres",
    "exact_retained_evidence_bytes",
    "database_inventory_lineage",
    "inspection_equilibrium_lineage",
    "distinct_typed_request_hashes",
    "immutable_runtime_image_inspected",
    "docker_image_inspection_retained",
    "pinned_sandbox_policy_enforced",
    "clean_repository",
    "image_revision_matches_git",
)
REQUIRED_CALPHAD_RELEASE_INPUT_FILES = (
    "backend/deepagents_runtime/materials_data/calphad/experimental_benchmark_manifest.json",
    "backend/deepagents_runtime/materials_data/calphad/manifest.json",
    "backend/deepagents_runtime/src/ultra_deepagents/context_tools.py",
    "backend/deepagents_runtime/src/ultra_deepagents/live_trace.py",
    "backend/deepagents_runtime/src/ultra_deepagents/materials/calphad.py",
    "backend/deepagents_runtime/src/ultra_deepagents/materials/calphad_cli.py",
    "backend/deepagents_runtime/src/ultra_deepagents/materials/calphad_tools.py",
    "backend/deepagents_runtime/tests/test_calphad_cli.py",
    "backend/deepagents_runtime/tests/test_calphad_runtime.py",
    "backend/deepagents_runtime/tests/test_calphad_tools.py",
    "backend/deepagents_runtime/tests/test_materials_live_trace.py",
    "scripts/calphad_cross_language_gate.py",
    "scripts/calphad_experimental_benchmark.py",
    "tests/fixtures/materials/calphad_experimental_benchmark_expected.json",
    "tests/test_calphad_cross_language_gate.py",
    "tests/test_calphad_experimental_benchmark.py",
)
REQUIRED_CALPHAD_LEDGER_TESTS = (
    "TestPostgresStoreCalphadLedgerIsAppendOnlyTenantScopedAndContentBound",
    "TestCalphadLedgerSchemaEncodesGovernanceWithoutRelationalizingGibbsModels",
    "TestCalphadGovernanceHTTPIsOwnerReadableAndWorkerWritable",
    "TestCalphadWorkerValidationRouteIsExplicitlyAllowlisted",
    "TestCalphadCanonicalJSONMatchesPythonUTF8Contract",
    "TestCalphadDatabaseInventoryFingerprintExcludesOnlySelection",
    "TestVerifyCalphadEvidenceBindsExactBytesAndRejectsForgery",
    "TestVerifyCalphadFailureEvidenceAcceptsOnlyExactBoundedTerminalTuples",
    "TestVerifyCalphadInspectionEvidenceRejectsResealedScientificForgeries",
    "TestVerifyCalphadEquilibriumEvidenceAcceptsOnlyBoundedTypedRequest",
    "TestVerifyCalphadEquilibriumEvidenceRejectsResealedScientificForgeries",
    "TestVerifyCalphadEquilibriumEvidenceRejectsDeclaredTemperatureExtrapolation",
    "TestRunSelectedCalphadBindingPinsOwnerDeclarations",
    "TestCalphadEvidenceRejectsDuplicateKeysTrailingMembersAndZipBombs",
    "TestCalphadValidationEnvelopeRejectsDuplicateAndUnknownFields",
)
CALPHAD_LEDGER_GO_TEST_PATTERN = (
    "^(" + "|".join(re.escape(name) for name in REQUIRED_CALPHAD_LEDGER_TESTS) + ")$"
)
CALPHAD_LEDGER_GO_COMMAND = (
    "go",
    "test",
    "-json",
    "-count=1",
    "./internal/store",
    "./internal/httpapi",
    "-run",
    CALPHAD_LEDGER_GO_TEST_PATTERN,
)
CALPHAD_LEDGER_MODULE = "github.com/amilworks/bisque-ultra/backend/controlplane"
CALPHAD_LEDGER_TEST_PACKAGES = {
    "TestPostgresStoreCalphadLedgerIsAppendOnlyTenantScopedAndContentBound": (
        f"{CALPHAD_LEDGER_MODULE}/internal/store"
    ),
    "TestCalphadLedgerSchemaEncodesGovernanceWithoutRelationalizingGibbsModels": (
        f"{CALPHAD_LEDGER_MODULE}/internal/store"
    ),
    "TestCalphadGovernanceHTTPIsOwnerReadableAndWorkerWritable": (
        f"{CALPHAD_LEDGER_MODULE}/internal/httpapi"
    ),
    "TestCalphadWorkerValidationRouteIsExplicitlyAllowlisted": (
        f"{CALPHAD_LEDGER_MODULE}/internal/httpapi"
    ),
    "TestCalphadCanonicalJSONMatchesPythonUTF8Contract": (
        f"{CALPHAD_LEDGER_MODULE}/internal/httpapi"
    ),
    "TestCalphadDatabaseInventoryFingerprintExcludesOnlySelection": (
        f"{CALPHAD_LEDGER_MODULE}/internal/httpapi"
    ),
    "TestVerifyCalphadEvidenceBindsExactBytesAndRejectsForgery": (
        f"{CALPHAD_LEDGER_MODULE}/internal/httpapi"
    ),
    "TestVerifyCalphadFailureEvidenceAcceptsOnlyExactBoundedTerminalTuples": (
        f"{CALPHAD_LEDGER_MODULE}/internal/httpapi"
    ),
    "TestVerifyCalphadInspectionEvidenceRejectsResealedScientificForgeries": (
        f"{CALPHAD_LEDGER_MODULE}/internal/httpapi"
    ),
    "TestVerifyCalphadEquilibriumEvidenceAcceptsOnlyBoundedTypedRequest": (
        f"{CALPHAD_LEDGER_MODULE}/internal/httpapi"
    ),
    "TestVerifyCalphadEquilibriumEvidenceRejectsResealedScientificForgeries": (
        f"{CALPHAD_LEDGER_MODULE}/internal/httpapi"
    ),
    "TestVerifyCalphadEquilibriumEvidenceRejectsDeclaredTemperatureExtrapolation": (
        f"{CALPHAD_LEDGER_MODULE}/internal/httpapi"
    ),
    "TestRunSelectedCalphadBindingPinsOwnerDeclarations": (
        f"{CALPHAD_LEDGER_MODULE}/internal/httpapi"
    ),
    "TestCalphadEvidenceRejectsDuplicateKeysTrailingMembersAndZipBombs": (
        f"{CALPHAD_LEDGER_MODULE}/internal/httpapi"
    ),
    "TestCalphadValidationEnvelopeRejectsDuplicateAndUnknownFields": (
        f"{CALPHAD_LEDGER_MODULE}/internal/httpapi"
    ),
}
CALPHAD_POSTGRES_TEST = "TestPostgresStoreCalphadLedgerIsAppendOnlyTenantScopedAndContentBound"
CALPHAD_HTTP_TEST = "TestCalphadGovernanceHTTPIsOwnerReadableAndWorkerWritable"
CALPHAD_POSTGRES_IDENTITY_MARKER = "CALPHAD_POSTGRES_IDENTITY "
CALPHAD_POSTGRES_INVARIANT_TEST_EVIDENCE: dict[str, tuple[str, ...]] = {
    "append_only_update_delete": tuple(
        f"{CALPHAD_POSTGRES_TEST}/{suffix}"
        for suffix in (
            "append_only_revision_update",
            "append_only_revision_delete",
            "append_only_validation_update",
            "append_only_validation_delete",
            "append_only_evidence_update",
            "append_only_evidence_delete",
        )
    ),
    "append_only_truncate": tuple(
        f"{CALPHAD_POSTGRES_TEST}/{suffix}"
        for suffix in (
            "append_only_revision_truncate",
            "append_only_validation_truncate",
            "append_only_evidence_truncate",
        )
    ),
    "database_bytes_revision_bound": tuple(
        f"{CALPHAD_POSTGRES_TEST}/{suffix}"
        for suffix in ("database_revision_binding", "database_digest_binding")
    ),
    "evidence_bytes_server_verified": (
        f"{CALPHAD_POSTGRES_TEST}/evidence_blob_content_bound",
        CALPHAD_HTTP_TEST,
    ),
    "immutable_runtime_image_required": (
        f"{CALPHAD_POSTGRES_TEST}/immutable_runtime_image",
        f"{CALPHAD_POSTGRES_TEST}/runtime_policy_authorized",
    ),
    "retry_idempotent": (f"{CALPHAD_POSTGRES_TEST}/retry_idempotent",),
    "multiple_equilibria_idempotent": (f"{CALPHAD_POSTGRES_TEST}/multiple_equilibria_idempotent",),
    "run_lease_authorized": (f"{CALPHAD_POSTGRES_TEST}/run_lease_authorized",),
    "tenant_scoped": (f"{CALPHAD_POSTGRES_TEST}/parent_same_tenant",),
    "inspection_lineage_bound": (f"{CALPHAD_POSTGRES_TEST}/inspection_lineage_required",),
    "inspection_inventory_bound": (f"{CALPHAD_POSTGRES_TEST}/inspection_inventory_bound",),
    "retained_evidence_contract_bound": (f"{CALPHAD_POSTGRES_TEST}/evidence_blob_content_bound",),
    "retained_failure_evidence": (
        f"{CALPHAD_POSTGRES_TEST}/retained_terminal_statuses",
        "TestVerifyCalphadFailureEvidenceAcceptsOnlyExactBoundedTerminalTuples",
    ),
    "retained_timeout_evidence": (
        f"{CALPHAD_POSTGRES_TEST}/retained_terminal_statuses",
        CALPHAD_HTTP_TEST,
    ),
    "retained_unsupported_evidence": (
        f"{CALPHAD_POSTGRES_TEST}/retained_terminal_statuses",
        "TestVerifyCalphadFailureEvidenceAcceptsOnlyExactBoundedTerminalTuples",
    ),
    "terminal_failure_nonpromotable": (
        f"{CALPHAD_POSTGRES_TEST}/retained_terminal_statuses",
        CALPHAD_HTTP_TEST,
    ),
    "schema_fingerprint_verified": (f"{CALPHAD_POSTGRES_TEST}/schema_fingerprint_verified",),
    "trigger_search_path_pinned": (
        f"{CALPHAD_POSTGRES_TEST}/temporary_schema_guarded",
        f"{CALPHAD_POSTGRES_TEST}/schema_fingerprint_verified",
    ),
    "serving_role_separated": (f"{CALPHAD_POSTGRES_TEST}/serving_role_separated",),
    "public_and_unexpected_acl_grantees_rejected": (
        f"{CALPHAD_POSTGRES_TEST}/public_and_unexpected_acl_grantees_rejected",
    ),
    "unexpected_writer_overload_revoked_and_rejected": (
        f"{CALPHAD_POSTGRES_TEST}/unexpected_writer_overload_revoked_and_rejected",
    ),
    "equilibrium_reads_require_retained_inspection_event": (
        f"{CALPHAD_POSTGRES_TEST}/equilibrium_reads_require_retained_inspection_event",
    ),
}
REQUIRED_CALPHAD_LEDGER_INVARIANTS = (
    "append_only_update_delete",
    "append_only_truncate",
    "database_bytes_revision_bound",
    "evidence_bytes_server_verified",
    "immutable_runtime_image_required",
    "retry_idempotent",
    "multiple_equilibria_idempotent",
    "run_lease_authorized",
    "tenant_scoped",
    "inspection_lineage_bound",
    "inspection_inventory_bound",
    "retained_evidence_contract_bound",
    "retained_failure_evidence",
    "retained_timeout_evidence",
    "retained_unsupported_evidence",
    "terminal_failure_nonpromotable",
    "schema_fingerprint_verified",
    "trigger_search_path_pinned",
    "serving_role_separated",
    "public_and_unexpected_acl_grantees_rejected",
    "unexpected_writer_overload_revoked_and_rejected",
    "equilibrium_reads_require_retained_inspection_event",
)

REQUIRED_DOMAIN_VALIDATORS = (
    "materials.atomistics.ase_emt_cu_eos_smoke.v1",
    "materials.calphad.input_domain_axes_fixture_rejection.v1",
    "materials.calphad.nist_al_co_w_phase_field_checkpoints.v2",
    "materials.defects.nacl_generator_stoichiometry.v1",
    "materials.dream3d.geometry_feature_sentinel.v1",
    "materials.ebsd.ipf_cubic_color_coverage.v1",
    "materials.ebsd.ipf_cubic_tsl_corners.v1",
    "materials.ebsd.mackenzie_cubic_distribution.v1",
    "materials.informatics.magpie_ni3al_schema.v1",
    "materials.microstructure.anisotropic_stereology_volume.v1",
    "materials.porosity.porespy_true_void_local_radius.v1",
    "materials.structure.ordering_sensitive_space_group.v1",
    "materials.xrd.fcc_ni_cuka_peak_and_extinctions.v1",
)
REQUIRED_DOMAIN_INVARIANT_COUNT = len(REQUIRED_DOMAIN_VALIDATORS)
PRODUCTION_PARITY_GATE = "production-materials-sandbox-parity"
PRODUCTION_PARITY_SCOPE = "production-full"
PRODUCTION_PARITY_CLAIM = "full production DockerSandboxBackend image parity"
PRODUCTION_IMAGE_TITLE = "Ultra Deep Agents scientific sandbox"
PRODUCTION_PARITY_REPORT_PREFIX = "production-materials-sandbox-parity-"
CALPHAD_RUNTIME_TEST_COUNT = 39
CALPHAD_RUNTIME_CORE_TEST_COUNT = 36
CALPHAD_RUNTIME_CLI_TEST_COUNT = 3
REQUIRED_CALPHAD_RUNTIME_CORE_TEST_NAMES = (
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
REQUIRED_CALPHAD_RUNTIME_CLI_TEST_NAMES = (
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
CALPHAD_TOOLS_TEST_COUNT = 56
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
CALPHAD_EQUILIBRIUM_SCHEMA = "ultra.calphad.equilibrium.v2"
MAX_JUNIT_BYTES = 4 * 1024 * 1024
MAX_CALPHAD_LEDGER_GO_LOG_BYTES = 32 * 1024 * 1024
CALPHAD_REAL_HTTP_REVALIDATION_TEST = "TestCalphadTypedCLIArtifactsPassRealHTTPVerifier"
CALPHAD_REAL_HTTP_REVALIDATION_PACKAGE = (
    "github.com/amilworks/bisque-ultra/backend/controlplane/integration"
)
CALPHAD_REAL_HTTP_REVALIDATION_COMMAND = (
    "go",
    "test",
    "-json",
    "-count=1",
    "./integration",
    "-run",
    f"^{CALPHAD_REAL_HTTP_REVALIDATION_TEST}$",
)


class ReadinessPolicy(NamedTuple):
    official_revision: str = OFFICIAL_REVISION
    official_manifest_sha256: str = OFFICIAL_MANIFEST_SHA256


class ExpectedProvenance(NamedTuple):
    git_sha: str
    domain_image: str
    runtime_image: str
    evaluator_image: str


class GateInputError(RuntimeError):
    """An input could not be read safely enough to make a gate decision."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r} is forbidden")
        value[key] = item
    return value


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _strict_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _plain_sha256(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text[7:]
    return text if SHA256_RE.fullmatch(text) else None


def _immutable_sha256(value: Any) -> str | None:
    digest = _plain_sha256(value)
    return f"sha256:{digest}" if digest else None


def _git_sha(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    return text if GIT_SHA_RE.fullmatch(text) else None


def _is_remote_mutation_tool(name: Any) -> bool:
    normalized = str(name or "").strip().lower()
    return normalized.startswith("bisque_") and any(
        token in normalized for token in REMOTE_MUTATION_TOKENS
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tracked_path(path: Path) -> str:
    if path.is_symlink():
        return hashlib.sha256(os.readlink(path).encode("utf-8")).hexdigest()
    return sha256_file(path)


def manifest_hash(file_hashes: Mapping[str, str]) -> str:
    content = "".join(f"{file_hashes[path]}  {path}\n" for path in sorted(file_hashes)).encode(
        "utf-8"
    )
    return hashlib.sha256(content).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _worker_cleanroom_attestation_valid(attempt: Mapping[str, Any]) -> bool:
    """Recompute the generic worker seal from the sanitized trace projection."""

    trace = _mapping(attempt.get("trace_summary"))
    attestations = _sequence(trace.get("worker_cleanroom_attestations"))
    if len(attestations) != 1:
        return False
    record = _mapping(attestations[0])
    payload = _mapping(record.get("payload"))
    source_keys = _sequence(record.get("source_payload_keys"))
    if source_keys != sorted(WORKER_EVALUATION_ATTESTATION_FIELDS):
        return False
    if set(payload) != set(WORKER_EVALUATION_ATTESTATION_FIELDS):
        return False
    unsigned = dict(payload)
    declared_attestation_sha = unsigned.pop("attestation_sha256", None)
    run_id = str(attempt.get("run_id") or "")
    thread_id = str(attempt.get("thread_id") or "")
    run_sha = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    thread_sha = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()
    digests_valid = all(
        _plain_sha256(payload.get(name)) is not None
        for name in ("run_id_sha256", "thread_id_sha256", "user_id_sha256", "goal_sha256")
    )
    provided_count = payload.get("provided_message_count")
    payload_valid = all(
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
            _plain_sha256(declared_attestation_sha) is not None,
            declared_attestation_sha == canonical_json_sha256(unsigned),
        )
    )
    binding = _mapping(attempt.get("cleanroom_binding"))
    binding_valid = all(
        (
            binding.get("evaluation_profile") == MATERIALS_CLEANROOM_PROFILE,
            binding.get("worker_event_count") == 1,
            binding.get("worker_attestation_valid") is True,
            binding.get("server_attestation_valid") is True,
            _mapping(binding.get("identity_hash_checks"))
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
    return payload_valid and binding_valid


def load_json_report(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.expanduser().resolve()
    try:
        size = resolved.stat().st_size
    except OSError as exc:
        raise GateInputError(f"cannot stat report {resolved}: {exc}") from exc
    if size <= 0 or size > MAX_REPORT_BYTES:
        raise GateInputError(
            f"report {resolved} has invalid size {size}; maximum is {MAX_REPORT_BYTES}"
        )
    try:
        payload = json.loads(
            resolved.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise GateInputError(f"cannot parse report {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GateInputError(f"report {resolved} is not a JSON object")
    return payload, {
        "path": str(resolved),
        "size_bytes": size,
        "sha256": sha256_file(resolved),
    }


def inspect_repository(repository_root: Path) -> dict[str, Any]:
    root = repository_root.expanduser().resolve()
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    return {
        "commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": status.returncode != 0 or bool(status.stdout.strip()),
        "inspection_ok": revision.returncode == 0 and status.returncode == 0,
    }


def inspect_benchmark_checkout(benchmark_root: Path) -> dict[str, Any]:
    root = benchmark_root.expanduser().resolve()
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    tracked = subprocess.run(
        ("git", "ls-files", "-z"),
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    return {
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": status.returncode != 0 or bool(status.stdout.strip()),
        "tracked_files": sorted(value for value in tracked.stdout.split("\0") if value),
        "inspection_ok": all(process.returncode == 0 for process in (revision, status, tracked)),
    }


def _safe_relative_path(root: Path, relative: Any) -> Path | None:
    text = str(relative or "").strip()
    if not text or Path(text).is_absolute():
        return None
    candidate = root / text
    resolved_parent = candidate.parent.resolve()
    try:
        resolved_parent.relative_to(root.resolve())
    except ValueError:
        return None
    # Preserve a final symlink: the MatTools manifest hashes its link text, not
    # the target bytes. Resolving the parent still prevents traversal through a
    # symlinked directory outside the evidence root.
    return resolved_parent / candidate.name


def _rehash(*, label: str, path: Path | None, expected_sha256: Any, issues: list[str]) -> bool:
    expected = _plain_sha256(expected_sha256)
    if path is None:
        issues.append(f"{label}: path is missing or escapes its evidence root")
        return False
    if expected is None:
        issues.append(f"{label}: expected SHA-256 is missing or malformed")
        return False
    if not path.is_file() and not path.is_symlink():
        issues.append(f"{label}: evidence file is missing")
        return False
    try:
        observed = sha256_tracked_path(path)
    except OSError as exc:
        issues.append(f"{label}: cannot hash evidence ({exc})")
        return False
    if observed != expected:
        issues.append(f"{label}: SHA-256 mismatch")
        return False
    return True


def _repository_evidence(
    deterministic: Mapping[str, Any],
    mattools: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    issues: list[str] = []
    checks = 0
    verified = 0

    test_source = _mapping(deterministic.get("test_source"))
    raw_test_path = str(test_source.get("path") or "")
    test_path = None
    if raw_test_path.startswith("/workspace/"):
        test_path = _safe_relative_path(root, raw_test_path.removeprefix("/workspace/"))
    checks += 1
    verified += int(
        _rehash(
            label="deterministic test source",
            path=test_path,
            expected_sha256=test_source.get("sha256"),
            issues=issues,
        )
    )

    requirements = _mapping(deterministic.get("requirements"))
    checks += 1
    verified += int(
        _rehash(
            label="deterministic materials requirements",
            path=_safe_relative_path(root, "deploy/docker/materials-requirements.txt"),
            expected_sha256=requirements.get("source_sha256"),
            issues=issues,
        )
    )
    image = _mapping(deterministic.get("image"))
    checks += 1
    verified += int(
        _rehash(
            label="deterministic gate Dockerfile",
            path=_safe_relative_path(root, "deploy/docker/materials-domain-gate.Dockerfile"),
            expected_sha256=image.get("dockerfile_sha256"),
            issues=issues,
        )
    )

    harness = _mapping(mattools.get("harness"))
    for label, path_key, hash_key in (
        ("MatTools harness", "path", "sha256"),
        (
            "MatTools host validator lock",
            "host_validator_requirements_path",
            "host_validator_requirements_sha256",
        ),
        (
            "MatTools host validator reviewed input",
            "host_validator_input_requirements_path",
            "host_validator_input_requirements_sha256",
        ),
        ("MatTools strict shadow", "strict_shadow_path", "strict_shadow_sha256"),
        (
            "MatTools semantic repairs",
            "semantic_repairs_path",
            "semantic_repairs_sha256",
        ),
    ):
        checks += 1
        verified += int(
            _rehash(
                label=label,
                path=_safe_relative_path(root, harness.get(path_key)),
                expected_sha256=harness.get(hash_key),
                issues=issues,
            )
        )

    skills_root = root / "backend/deepagents_runtime/skills"
    skills = sorted(path for path in skills_root.rglob("*") if path.is_file())
    observed_skill_hashes = {
        str(path.relative_to(root)): sha256_tracked_path(path) for path in skills
    }
    ultra = _mapping(mattools.get("ultra"))
    checks += 1
    expected_skills = _plain_sha256(ultra.get("skills_sha256"))
    observed_skills = manifest_hash(observed_skill_hashes)
    skill_count = _strict_int(ultra.get("skills_file_count"))
    skills_ok = (
        expected_skills == observed_skills
        and skill_count == len(observed_skill_hashes)
        and len(observed_skill_hashes) > 0
    )
    if not skills_ok:
        issues.append("Ultra skills manifest hash or file count changed after MatTools reporting")
    verified += int(skills_ok)

    return {
        "valid": checks > 0 and verified == checks,
        "checks": checks,
        "verified": verified,
        "issues": issues,
    }


def _canonical_package_name(name: Any) -> str:
    return re.sub(r"[-_.]+", "-", str(name or "").strip().lower())


def _parse_hashed_requirements_lock(path: Path) -> tuple[dict[str, str], list[str]]:
    """Parse the reviewed uv lock and require hashes for every exact pin."""

    issues: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        return {}, [f"cannot read host validator lock: {exc}"]
    packages: dict[str, str] = {}
    current: str | None = None
    hashes: dict[str, int] = {}
    for line in lines:
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)", line)
        if match:
            current = _canonical_package_name(match.group(1))
            if not current or current in packages:
                issues.append(f"duplicate or invalid locked package {match.group(1)!r}")
                continue
            packages[current] = match.group(2)
            hashes[current] = 0
            continue
        digest = re.search(r"--hash=sha256:([0-9a-f]{64})(?:\s|\\|$)", line)
        if digest and current is not None:
            hashes[current] += 1
    unhashed = sorted(name for name in packages if hashes.get(name, 0) == 0)
    if unhashed:
        issues.append("host validator lock has unhashed pins: " + ", ".join(unhashed))
    if not packages:
        issues.append("host validator lock contains no exact package pins")
    return packages, issues


_HOST_VALIDATOR_PROBE_CACHE: dict[tuple[str, ...], dict[str, Any]] = {}


def _run_host_validator_no_task_probe(command: Sequence[str], snapshot_src: Path) -> dict[str, Any]:
    """Re-run the parser import in the exact cached lock without executing a task."""

    key = (*command, str(snapshot_src.resolve()))
    cached = _HOST_VALIDATOR_PROBE_CACHE.get(key)
    if cached is not None:
        return dict(cached)
    probe = """
import hashlib
import json
import platform
import sys
from importlib.metadata import distributions, version

import result_analysis  # noqa: F401
from utils import ComplexDictParser  # noqa: F401

packages = {
    distribution.metadata.get("Name", "").lower(): distribution.version
    for distribution in distributions()
    if distribution.metadata.get("Name")
}
required_names = ("docker", "numpy", "openpyxl", "pandas", "pymatgen")
required = {name: version(name) for name in required_names}
digest = hashlib.sha256()
with open(sys.executable, "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
print(json.dumps({
    "schema_version": "1",
    "python_version": platform.python_version(),
    "python_implementation": platform.python_implementation(),
    "python_executable_sha256": digest.hexdigest(),
    "platform": platform.platform(),
    "task_execution_performed": False,
    "required_packages": required,
    "resolved_packages": dict(sorted(packages.items())),
}, sort_keys=True))
"""
    environment = os.environ.copy()
    environment["UV_OFFLINE"] = "1"
    process = subprocess.run(
        (*command, "-c", probe),
        cwd=snapshot_src,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=600,
    )
    if process.returncode != 0:
        raise GateInputError(f"offline host-validator no-task probe exited {process.returncode}")
    try:
        payload = json.loads(
            process.stdout.strip().splitlines()[-1],
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (IndexError, json.JSONDecodeError, ValueError) as exc:
        raise GateInputError("offline host-validator no-task probe returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise GateInputError("offline host-validator no-task probe returned a non-object")
    _HOST_VALIDATOR_PROBE_CACHE[key] = payload
    return dict(payload)


def _host_validator_environment_evidence(
    mattools: Mapping[str, Any], repository_root: Path, benchmark_root: Path
) -> dict[str, Any]:
    harness = _mapping(mattools.get("harness"))
    environment = _mapping(harness.get("host_validator_environment"))
    required = _mapping(environment.get("required_packages"))
    resolved = {
        str(name).lower(): str(version)
        for name, version in _mapping(environment.get("resolved_packages")).items()
        if str(name).strip() and str(version).strip()
    }
    command = [str(value) for value in _sequence(environment.get("validator_command"))]
    issues: list[str] = []
    root = repository_root.expanduser().resolve()
    lock_path = root / "scripts/mattools-validator-requirements.lock.txt"
    input_path = root / "scripts/mattools-validator-requirements.txt"
    regular_inputs = all(
        path.is_file() and not path.is_symlink() for path in (lock_path, input_path)
    )
    observed_lock_sha = sha256_file(lock_path) if regular_inputs else None
    observed_input_sha = sha256_file(input_path) if regular_inputs else None
    lock_packages, lock_issues = (
        _parse_hashed_requirements_lock(lock_path) if regular_inputs else ({}, ["lock is missing"])
    )
    issues.extend(lock_issues)
    try:
        input_packages = {
            _canonical_package_name(match.group(1)): match.group(2)
            for line in input_path.read_text(encoding="utf-8").splitlines()
            if (match := re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", line.strip()))
        }
    except (OSError, UnicodeDecodeError):
        input_packages = {}
    reviewed_lock_valid = all(
        (
            regular_inputs,
            observed_lock_sha == EXPECTED_HOST_VALIDATOR_LOCK_SHA256,
            observed_input_sha == EXPECTED_HOST_VALIDATOR_INPUT_SHA256,
            _plain_sha256(harness.get("host_validator_requirements_sha256"))
            == EXPECTED_HOST_VALIDATOR_LOCK_SHA256,
            _plain_sha256(harness.get("host_validator_input_requirements_sha256"))
            == EXPECTED_HOST_VALIDATOR_INPUT_SHA256,
            _plain_sha256(environment.get("requirements_lock_sha256"))
            == EXPECTED_HOST_VALIDATOR_LOCK_SHA256,
            _plain_sha256(environment.get("requirements_input_sha256"))
            == EXPECTED_HOST_VALIDATOR_INPUT_SHA256,
            input_packages == EXPECTED_HOST_VALIDATOR_PACKAGES,
            len(lock_packages) >= 40,
            all(lock_packages.get(name) == version for name, version in input_packages.items()),
        )
    )
    if not reviewed_lock_valid:
        issues.append("host validator lock/input is not the complete reviewed hashed environment")

    def reported_path_matches(value: Any, expected_path: Path) -> bool:
        text = str(value or "").strip()
        if not text:
            return False
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        return candidate.resolve() == expected_path.resolve()

    uv_observed = shutil.which("uv")
    uv_reported = Path(command[0]).expanduser() if command else None
    uv_valid = all(
        (
            uv_observed is not None,
            uv_reported is not None,
            uv_reported is not None and uv_reported.resolve() == Path(str(uv_observed)).resolve(),
            uv_reported is not None
            and uv_reported.resolve().is_file()
            and not uv_reported.resolve().is_symlink(),
        )
    )
    expected_command = (
        [
            command[0],
            "run",
            "--isolated",
            "--no-project",
            "--python",
            EXPECTED_HOST_VALIDATOR_PYTHON,
            "--with-requirements",
            str(lock_path),
            "python",
        ]
        if command
        else []
    )
    command_valid = all(
        (
            uv_valid,
            command == expected_command,
            reported_path_matches(environment.get("requirements_lock_path"), lock_path),
            reported_path_matches(environment.get("requirements_input_path"), input_path),
        )
    )
    actual_uv_sha = sha256_file(uv_reported.resolve()) if uv_valid and uv_reported else None
    runner_records = [
        _mapping(_mapping(trial).get("runner")) for trial in _sequence(mattools.get("trials"))
    ]
    runner_binding_valid = len(runner_records) == TRIAL_COUNT and all(
        _sequence(runner.get("host_validator_command")) == command
        and _plain_sha256(runner.get("host_validator_executable_sha256")) == actual_uv_sha
        and _plain_sha256(runner.get("host_requirements_sha256"))
        == EXPECTED_HOST_VALIDATOR_LOCK_SHA256
        and _plain_sha256(runner.get("host_input_requirements_sha256"))
        == EXPECTED_HOST_VALIDATOR_INPUT_SHA256
        and _mapping(runner.get("host_validator_environment")) == environment
        for runner in runner_records
    )
    command_valid = command_valid and runner_binding_valid
    resolved_hash = canonical_json_sha256(dict(sorted(resolved.items()))) if resolved else None
    resolved_canonical = {
        _canonical_package_name(name): version for name, version in resolved.items()
    }
    direct_versions_valid = required == EXPECTED_HOST_VALIDATOR_PACKAGES and all(
        resolved_canonical.get(name) == version
        for name, version in EXPECTED_HOST_VALIDATOR_PACKAGES.items()
    )
    full_resolved_lock_valid = bool(resolved) and resolved_canonical == lock_packages
    hashes_valid = all(
        (
            reviewed_lock_valid,
            _plain_sha256(environment.get("resolved_packages_sha256")) == resolved_hash,
            _plain_sha256(environment.get("python_executable_sha256")) is not None,
            full_resolved_lock_valid,
        )
    )
    identity_valid = all(
        (
            environment.get("schema_version") == "1",
            environment.get("python_version") == EXPECTED_HOST_VALIDATOR_PYTHON,
            environment.get("python_implementation") == EXPECTED_HOST_VALIDATOR_IMPLEMENTATION,
            bool(str(environment.get("platform") or "").strip()),
            environment.get("task_execution_performed") is False,
        )
    )
    probe: dict[str, Any] = {}
    if command_valid and reviewed_lock_valid:
        try:
            probe = _run_host_validator_no_task_probe(command, benchmark_root / "src")
        except (GateInputError, OSError, subprocess.SubprocessError) as exc:
            issues.append(f"host validator offline no-task probe failed: {exc}")
    probe_valid = all(
        (
            probe.get("schema_version") == "1",
            probe.get("python_version") == environment.get("python_version"),
            probe.get("python_implementation") == environment.get("python_implementation"),
            probe.get("python_executable_sha256") == environment.get("python_executable_sha256"),
            probe.get("platform") == environment.get("platform"),
            probe.get("task_execution_performed") is False,
            _mapping(probe.get("required_packages")) == required,
            _mapping(probe.get("resolved_packages")) == resolved,
        )
    )
    if not probe_valid:
        issues.append("host validator report differs from the independent offline no-task probe")
    valid = all(
        (
            direct_versions_valid,
            full_resolved_lock_valid,
            command_valid,
            hashes_valid,
            identity_valid,
            probe_valid,
        )
    )
    if not direct_versions_valid:
        issues.append("host validator direct pins or resolved direct versions differ")
    if not command_valid:
        issues.append("host validator command is not the isolated CPython 3.11.9 lock command")
    if not hashes_valid:
        issues.append("host validator binary, lock, input, or resolved-map hash is inconsistent")
    if not identity_valid:
        issues.append("host validator interpreter identity or no-task preflight is incomplete")
    return {
        "valid": valid,
        "python_version": environment.get("python_version"),
        "python_implementation": environment.get("python_implementation"),
        "python_executable_sha256": _plain_sha256(environment.get("python_executable_sha256")),
        "resolved_package_count": len(resolved),
        "resolved_packages_sha256": resolved_hash,
        "direct_versions_valid": direct_versions_valid,
        "full_resolved_lock_valid": full_resolved_lock_valid,
        "reviewed_lock_valid": reviewed_lock_valid,
        "offline_no_task_probe_valid": probe_valid,
        "command_valid": command_valid,
        "hashes_valid": hashes_valid,
        "issues": issues,
    }


def _evaluator_lock_evidence(
    mattools: Mapping[str, Any], repository_root: Path, expected: ExpectedProvenance
) -> dict[str, Any]:
    official = _mapping(mattools.get("official_evaluator_environment"))
    approved = _mapping(official.get("approved_lock"))
    issues: list[str] = []
    lock_path = _safe_relative_path(repository_root, approved.get("path"))
    lock_regular = lock_path is not None and lock_path.is_file() and not lock_path.is_symlink()
    lock_rehashed = lock_regular and _rehash(
        label="MatTools approved evaluator lock",
        path=lock_path,
        expected_sha256=approved.get("sha256"),
        issues=issues,
    )
    if not lock_regular:
        issues.append("MatTools approved evaluator lock is not a regular file")
    raw_lock: dict[str, Any] = {}
    if lock_rehashed and lock_path is not None:
        try:
            payload = json.loads(
                lock_path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
            )
            raw_lock = _mapping(payload)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            issues.append(f"MatTools approved evaluator lock cannot be parsed: {exc}")
    report_lock_payload = {
        key: value
        for key, value in approved.items()
        if key not in {"path", "sha256", "approved_from_git_head"}
    }
    lock_semantics_valid = all(
        (
            approved.get("path") == EXPECTED_EVALUATOR_LOCK_PATH,
            approved.get("approved_from_git_head") is True,
            raw_lock == report_lock_payload,
            raw_lock.get("schema_version") == "1",
            raw_lock.get("environment_kind") == "reviewed-reconstruction-variant",
            raw_lock.get("official_artifact") is False,
            bool(str(raw_lock.get("variant_reason") or "").strip()),
        )
    )
    if not lock_semantics_valid:
        issues.append("approved evaluator lock differs from the reviewed repository payload")

    packages = {
        str(name).lower(): str(version)
        for name, version in _mapping(raw_lock.get("packages")).items()
        if str(name).strip() and str(version).strip()
    }
    package_hash = canonical_json_sha256(dict(sorted(packages.items()))) if packages else None
    packages_valid = bool(packages) and all(
        (
            _plain_sha256(raw_lock.get("package_map_sha256")) == package_hash,
            all(
                packages.get(name) == version
                for name, version in EXPECTED_EVALUATOR_PACKAGES.items()
            ),
        )
    )
    if not packages_valid:
        issues.append("approved evaluator full resolved package map is inconsistent")

    build = _mapping(raw_lock.get("build"))
    fixture_boundary_valid = all(
        (
            _strict_int(build.get("candidate_fixture_file_count"))
            == EXPECTED_CANDIDATE_FIXTURE_FILE_COUNT,
            _plain_sha256(build.get("candidate_fixture_manifest_sha256"))
            == EXPECTED_CANDIDATE_FIXTURE_MANIFEST_SHA256,
            build.get("candidate_visible_source_policy")
            == EXPECTED_CANDIDATE_VISIBLE_SOURCE_POLICY,
        )
    )
    if not fixture_boundary_valid:
        issues.append("approved evaluator lock does not bind the fixture-only candidate boundary")
    build_inputs = (
        ("dockerfile_path", "dockerfile_sha256", "deploy/docker/mattools-evaluator.Dockerfile"),
        (
            "supplemental_requirements_path",
            "supplemental_requirements_sha256",
            "deploy/docker/mattools-evaluator-supplemental-requirements.txt",
        ),
        ("strict_shadow_path", "strict_shadow_sha256", "scripts/mattools_strict_shadow.py"),
        ("safe_parser_path", "safe_parser_sha256", "scripts/mattools_safe_parser.py"),
        (
            "runner_wrapper_path",
            "runner_wrapper_sha256",
            "scripts/mattools_runner_wrapper.py",
        ),
        (
            "semantic_repairs_path",
            "semantic_repairs_sha256",
            "scripts/mattools_semantic_repairs.py",
        ),
        ("builder_path", "builder_sha256", "scripts/build_mattools_evaluator.py"),
    )
    verified_build_inputs = 0
    for path_key, hash_key, expected_path in build_inputs:
        declared_path = str(build.get(path_key) or "")
        path = _safe_relative_path(repository_root, declared_path)
        path_valid = all(
            (
                declared_path == expected_path,
                path is not None and path.is_file() and not path.is_symlink(),
                _rehash(
                    label=f"evaluator build input {expected_path}",
                    path=path,
                    expected_sha256=build.get(hash_key),
                    issues=issues,
                ),
            )
        )
        verified_build_inputs += int(path_valid)
    build_inputs_valid = verified_build_inputs == len(build_inputs)

    trial_environments = [
        _mapping(_mapping(trial).get("evaluator_environment"))
        for trial in _sequence(mattools.get("trials"))
    ]
    trials_valid = len(trial_environments) == TRIAL_COUNT
    upstream = _mapping(raw_lock.get("upstream"))
    platform = _mapping(raw_lock.get("platform"))
    expected_labels = {
        "io.ultra.mattools.adapted-requirements-sha256": build.get("adapted_requirements_sha256"),
        "io.ultra.mattools.base-image": build.get("base_image"),
        "io.ultra.mattools.environment-kind": raw_lock.get("environment_kind"),
        "io.ultra.mattools.official-artifact": "false",
        "io.ultra.mattools.snapshot-manifest-sha256": upstream.get("manifest_sha256"),
        "io.ultra.mattools.safe-parser-sha256": build.get("safe_parser_sha256"),
        "io.ultra.mattools.runner-wrapper-sha256": build.get("runner_wrapper_sha256"),
        "io.ultra.mattools.semantic-repairs-sha256": build.get("semantic_repairs_sha256"),
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
        "candidate_fixture_file_count": build.get("candidate_fixture_file_count"),
        "candidate_fixture_manifest_sha256": build.get("candidate_fixture_manifest_sha256"),
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
    for index, environment in enumerate(trial_environments, start=1):
        resolved = {
            str(name).lower(): str(version)
            for name, version in _mapping(environment.get("resolved_packages")).items()
            if str(name).strip() and str(version).strip()
        }
        embedded = _mapping(environment.get("embedded_inputs"))
        environment_valid = all(
            (
                resolved == packages,
                _plain_sha256(environment.get("resolved_environment_sha256")) == package_hash,
                environment.get("approved_environment_lock") == approved,
                environment.get("environment_kind") == raw_lock.get("environment_kind"),
                environment.get("official_artifact") is False,
                environment.get("python_version") == raw_lock.get("python_version"),
                environment.get("platform") == raw_lock.get("platform"),
                environment.get("task_execution_performed") is False,
                environment.get("labels_match_approved_lock") is True,
                environment.get("embedded_inputs_match_approved_lock") is True,
                environment.get("platform_matches_approved_lock") is True,
                environment.get("full_environment_lock_matches") is True,
                environment.get("comparable") is True,
                _immutable_sha256(environment.get("image_id")) == expected.evaluator_image,
                _immutable_sha256(environment.get("production_runtime_image_digest"))
                == expected.runtime_image,
                embedded.get("upstream_requirements_sha256") == upstream.get("requirements_sha256"),
                embedded.get("adapted_requirements_sha256")
                == build.get("adapted_requirements_sha256"),
                embedded.get("supplemental_requirements_sha256")
                == build.get("supplemental_requirements_sha256"),
                _mapping(environment.get("image_labels")) == dict(sorted(expected_labels.items())),
                embedded == expected_embedded_inputs,
            )
        )
        if not environment_valid:
            issues.append(
                f"trial {index} evaluator environment differs from the full approved lock"
            )
        trials_valid = trials_valid and environment_valid
    observed_trials_valid = _sequence(official.get("observed_trials")) == trial_environments
    if not observed_trials_valid:
        issues.append("top-level evaluator observations differ from trial evidence")
    valid = all(
        (
            lock_rehashed,
            lock_semantics_valid,
            packages_valid,
            fixture_boundary_valid,
            build_inputs_valid,
            trials_valid,
            observed_trials_valid,
            official.get("required_packages") == EXPECTED_EVALUATOR_PACKAGES,
        )
    )
    return {
        "valid": valid,
        "lock_rehashed": lock_rehashed,
        "package_count": len(packages),
        "package_map_sha256": package_hash,
        "fixture_boundary_valid": fixture_boundary_valid,
        "verified_build_inputs": verified_build_inputs,
        "expected_build_inputs": len(build_inputs),
        "trials_valid": trials_valid,
        "issues": issues,
    }


def _revalidate_mattools_report_bundle(
    mattools: Mapping[str, Any],
    repository_root: Path,
    benchmark_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Invoke the reviewed harness's read-only exact bundle regeneration."""

    harness = repository_root / "scripts/mattools_promotion_gate.py"
    declared_harness = _mapping(mattools.get("harness"))
    if (
        not harness.is_file()
        or harness.is_symlink()
        or _plain_sha256(declared_harness.get("sha256")) != sha256_file(harness)
    ):
        return {"valid": False, "issues": ["reviewed MatTools harness is unavailable"]}
    process = subprocess.run(
        (
            os.sys.executable,
            str(harness),
            "verify-report",
            "--benchmark-root",
            str(benchmark_root),
            "--report-manifest",
            str(manifest_path),
        ),
        cwd=repository_root,
        text=True,
        capture_output=True,
        check=False,
        timeout=1200,
    )
    try:
        payload = json.loads(
            process.stdout,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (json.JSONDecodeError, ValueError):
        payload = {}
    result = _mapping(payload)
    result["process_exit_code"] = process.returncode
    result["process_stderr_sha256"] = hashlib.sha256(
        process.stderr.encode("utf-8", "replace")
    ).hexdigest()
    return result


def _mattools_manifest_evidence(
    manifest: Mapping[str, Any],
    metadata: Mapping[str, Any],
    mattools: Mapping[str, Any],
    repository_root: Path,
    benchmark_root: Path,
) -> dict[str, Any]:
    issues: list[str] = []
    manifest_meta = _mapping(metadata.get("mattools_report_manifest"))
    report_meta = _mapping(metadata.get("mattools_report"))
    manifest_path_text = str(manifest_meta.get("path") or "").strip()
    manifest_path = Path(manifest_path_text).expanduser().resolve() if manifest_path_text else None
    manifest_size = _strict_int(manifest_meta.get("size_bytes"))
    input_integrity = (
        manifest_path is not None
        and manifest_path.is_file()
        and not manifest_path.is_symlink()
        and manifest_path.name == "report_manifest.json"
        and manifest_size is not None
        and 0 < manifest_size <= MAX_REPORT_BYTES
        and manifest_path.stat().st_size == manifest_size
    ) and _rehash(
        label="MatTools report manifest",
        path=manifest_path,
        expected_sha256=manifest_meta.get("sha256"),
        issues=issues,
    )
    if input_integrity and manifest_path is not None:
        try:
            on_disk_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
            input_integrity = on_disk_manifest == _mapping(manifest)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            input_integrity = False
    root = manifest_path.parent if manifest_path is not None else None
    verified = 0
    schema_valid = all(
        (
            manifest.get("schema_version") == "2",
            manifest.get("manifest_kind") == "ultra.mattools.report_bundle.v2",
            bool(str(manifest.get("generated_at") or "").strip()),
            manifest.get("campaign_id") == mattools.get("campaign_id"),
            _plain_sha256(manifest.get("benchmark_sha256"))
            == _plain_sha256(_mapping(mattools.get("benchmark")).get("sha256")),
            _plain_sha256(manifest.get("checkpoint_evidence_audit_sha256"))
            == canonical_json_sha256(mattools.get("checkpoint_evidence_audit")),
            _mapping(manifest.get("regeneration"))
            == {
                "helper": "revalidate_report_bundle",
                "cli_subcommand": "verify-report",
                "comparison": "byte_exact",
                "task_execution_performed": False,
            },
        )
    )
    records_valid = input_integrity and root is not None and schema_valid
    for key in ("results_json", "results_markdown", "checkpoint"):
        record = _mapping(manifest.get(key))
        path_text = str(record.get("path") or "").strip()
        path = Path(path_text).expanduser().resolve() if path_text else None
        expected_name = {
            "results_json": "results.json",
            "results_markdown": "results.md",
            "checkpoint": "state.json",
        }[key]
        inside_root = (
            path is not None
            and root is not None
            and path.parent == root
            and path.name == expected_name
            and path.is_file()
            and not path.is_symlink()
        )
        hash_ok = bool(inside_root) and _rehash(
            label=f"MatTools manifest {key}",
            path=path,
            expected_sha256=record.get("sha256"),
            issues=issues,
        )
        if key == "results_json":
            expected_report_path = str(report_meta.get("path") or "").strip()
            hash_ok = (
                bool(hash_ok)
                and path is not None
                and path == Path(expected_report_path).expanduser().resolve()
                and _plain_sha256(record.get("sha256")) == _plain_sha256(report_meta.get("sha256"))
            )
        verified += int(bool(hash_ok))
        records_valid = records_valid and bool(hash_ok)
    bundle = (
        _revalidate_mattools_report_bundle(
            mattools,
            repository_root,
            benchmark_root,
            manifest_path,
        )
        if manifest_path is not None and records_valid
        else {"valid": False, "issues": ["manifest records failed before regeneration"]}
    )
    bundle_valid = all(
        (
            _strict_int(bundle.get("process_exit_code")) == 0,
            bundle.get("schema_version") == "1",
            bundle.get("revalidation_kind") == "ultra.mattools.report_revalidation.v1",
            bundle.get("valid") is True,
            bundle.get("bundle_exact") is True,
            bundle.get("manifest_integrity_valid") is True,
            bundle.get("checkpoint_evidence_valid") is True,
            bundle.get("checkpoint_exact") is True,
            bundle.get("results_json_exact") is True,
            bundle.get("results_markdown_exact") is True,
            bundle.get("manifest_exact") is True,
            bundle.get("task_execution_performed") is False,
            bundle.get("promotion_passed")
            is (_mapping(mattools.get("promotion")).get("passed") is True),
            _mapping(bundle.get("checkpoint_evidence_audit"))
            == _mapping(mattools.get("checkpoint_evidence_audit")),
        )
    )
    if not schema_valid:
        issues.append("MatTools report manifest schema/regeneration binding is invalid")
    if not bundle_valid:
        issues.append("MatTools report/checkpoint bundle failed exact read-only regeneration")
        issues.extend(str(issue) for issue in _sequence(bundle.get("issues")))
    return {
        "valid": records_valid and verified == 3 and bundle_valid,
        "input_integrity": input_integrity,
        "schema_valid": schema_valid,
        "verified_records": verified,
        "expected_records": 3,
        "bundle_revalidation": bundle,
        "issues": issues,
    }


def _calphad_go_test_log_evidence(path: Path | None) -> dict[str, Any]:
    """Recompute exact test outcomes from the retained ``go test -json`` log."""

    issues: list[str] = []
    if path is None:
        return {"valid": False, "records": [], "issues": ["Go test log path is missing"]}
    try:
        payload = path.read_bytes()
    except OSError as exc:
        return {"valid": False, "records": [], "issues": [str(exc)]}
    if not payload or len(payload) > MAX_CALPHAD_LEDGER_GO_LOG_BYTES:
        return {
            "valid": False,
            "records": [],
            "issues": ["Go test log is empty or exceeds its fixed evidence bound"],
        }

    outcomes: dict[str, str] = {}
    packages: dict[str, str] = {}
    event_packages: dict[str, str] = {}
    observed_database: dict[str, Any] = {}
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        try:
            event = json.loads(
                raw_line,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            issues.append(f"Go test log line {line_number} is invalid JSON: {exc}")
            continue
        if not isinstance(event, dict):
            issues.append(f"Go test log line {line_number} is not an object")
            continue
        test_name = str(event.get("Test") or "")
        action = str(event.get("Action") or "")
        package = str(event.get("Package") or "")
        if test_name and action in {"pass", "fail", "skip"}:
            outcomes[test_name] = action
            event_packages[test_name] = package
        if test_name in REQUIRED_CALPHAD_LEDGER_TESTS and action in {"pass", "fail", "skip"}:
            packages[test_name] = package
        if not test_name and action == "fail" and package:
            issues.append(f"Go package failed: {package}")
        output_text = str(event.get("Output") or "")
        marker_at = output_text.find(CALPHAD_POSTGRES_IDENTITY_MARKER)
        if marker_at >= 0:
            encoded = output_text[marker_at + len(CALPHAD_POSTGRES_IDENTITY_MARKER) :].strip()
            try:
                identity = json.loads(
                    encoded,
                    parse_constant=_reject_json_constant,
                    object_pairs_hook=_reject_duplicate_json_keys,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                issues.append(f"PostgreSQL identity output is invalid JSON: {exc}")
            else:
                observed_database = _mapping(identity)

    records = [
        {
            "name": name,
            "package": packages.get(name, ""),
            "passed": outcomes.get(name) == "pass",
            "skipped": outcomes.get(name) == "skip",
        }
        for name in REQUIRED_CALPHAD_LEDGER_TESTS
    ]
    if any(
        not record["passed"] or record["package"] != CALPHAD_LEDGER_TEST_PACKAGES[record["name"]]
        for record in records
    ):
        issues.append("Go test log does not prove every exact required test passed")
    invariant_outcomes: dict[str, bool] = {}
    invariant_records: list[dict[str, Any]] = []
    for invariant in REQUIRED_CALPHAD_LEDGER_INVARIANTS:
        evidence_tests = CALPHAD_POSTGRES_INVARIANT_TEST_EVIDENCE.get(invariant, ())
        passed = bool(evidence_tests) and all(
            outcomes.get(test_name) == "pass"
            and event_packages.get(test_name)
            == (
                CALPHAD_LEDGER_TEST_PACKAGES[CALPHAD_POSTGRES_TEST]
                if test_name.startswith(CALPHAD_POSTGRES_TEST)
                else CALPHAD_LEDGER_TEST_PACKAGES[CALPHAD_HTTP_TEST]
            )
            for test_name in evidence_tests
        )
        invariant_outcomes[invariant] = passed
        invariant_records.append(
            {
                "name": invariant,
                "passed": passed,
                "test_evidence": [
                    {"name": test_name, "outcome": outcomes.get(test_name, "missing")}
                    for test_name in evidence_tests
                ],
            }
        )
        if not passed:
            issues.append(f"PostgreSQL invariant lacks passing test evidence: {invariant}")
    observed_database_valid = all(
        (
            set(observed_database)
            == {
                "database",
                "server_address",
                "server_port",
                "connection_target_host",
                "connection_target_port",
                "transaction_read_only",
                "role",
                "role_superuser",
                "role_create_role",
                "role_create_database",
                "role_replication",
                "role_bypass_rls",
                "calphad_owned_tables",
                "calphad_owned_functions",
                "calphad_owner_roles",
                "calphad_reachable_roles",
                "calphad_owner_role_reachable",
                "public_schema_owner",
                "public_owner_role_reachable",
                "can_create_public_schema",
                "calphad_select_all",
                "calphad_insert_all",
                "calphad_insert_any",
                "calphad_execute_create_revision",
                "calphad_execute_append_validation",
                "calphad_writer_functions_exact",
                "calphad_execute_unexpected_writer",
                "calphad_execute_internal",
                "calphad_public_execute",
                "calphad_unexpected_table_acl_grantees",
                "calphad_unexpected_function_acl_grantees",
                "calphad_mutation_privilege",
            },
            bool(str(observed_database.get("database") or "").strip()),
            bool(str(observed_database.get("server_address") or "").strip()),
            _strict_int(observed_database.get("server_port")) is not None,
            0 <= int(observed_database.get("server_port", -1)) <= 65535,
            bool(str(observed_database.get("role") or "").strip()),
            observed_database.get("transaction_read_only") == "off",
            observed_database.get("role_superuser") is False,
            observed_database.get("role_create_role") is False,
            observed_database.get("role_create_database") is False,
            observed_database.get("role_replication") is False,
            observed_database.get("role_bypass_rls") is False,
            _sequence(observed_database.get("calphad_owned_tables")) == [],
            _sequence(observed_database.get("calphad_owned_functions")) == [],
            bool(_sequence(observed_database.get("calphad_owner_roles"))),
            _sequence(observed_database.get("calphad_reachable_roles")) == [],
            observed_database.get("calphad_owner_role_reachable") is False,
            bool(str(observed_database.get("public_schema_owner") or "").strip()),
            observed_database.get("public_owner_role_reachable") is False,
            observed_database.get("can_create_public_schema") is False,
            observed_database.get("calphad_select_all") is True,
            observed_database.get("calphad_insert_all") is False,
            observed_database.get("calphad_insert_any") is False,
            observed_database.get("calphad_execute_create_revision") is True,
            observed_database.get("calphad_execute_append_validation") is True,
            observed_database.get("calphad_writer_functions_exact") is True,
            observed_database.get("calphad_execute_unexpected_writer") is False,
            observed_database.get("calphad_execute_internal") is False,
            observed_database.get("calphad_public_execute") is False,
            _sequence(observed_database.get("calphad_unexpected_table_acl_grantees")) == [],
            _sequence(observed_database.get("calphad_unexpected_function_acl_grantees")) == [],
            observed_database.get("calphad_mutation_privilege") is False,
            bool(str(observed_database.get("connection_target_host") or "").strip()),
            _strict_int(observed_database.get("connection_target_port")) is not None,
            0 < int(observed_database.get("connection_target_port", 0)) <= 65535,
        )
    )
    if not observed_database_valid:
        issues.append("Go test log lacks a valid writable PostgreSQL identity")
    return {
        "valid": not issues,
        "records": records,
        "invariant_outcomes": invariant_outcomes,
        "invariant_records": invariant_records,
        "observed_database": observed_database,
        "observed_database_valid": observed_database_valid,
        "issues": issues,
    }


def _calphad_ledger_evidence(
    report: Mapping[str, Any], metadata: Mapping[str, Any], repository_root: Path, git_sha: str
) -> dict[str, Any]:
    issues: list[str] = []
    report_meta = _mapping(metadata.get("calphad_ledger_report"))
    report_path_text = str(report_meta.get("path") or "").strip()
    report_path = Path(report_path_text).expanduser().resolve() if report_path_text else None
    report_digest = _plain_sha256(report_meta.get("sha256"))
    report_size = _strict_int(report_meta.get("size_bytes"))
    input_integrity = all(
        (
            report_path is not None,
            report_digest is not None,
            report_size is not None and 0 < report_size <= MAX_REPORT_BYTES,
            report_path is not None
            and report_digest is not None
            and report_path.name == f"calphad-ledger-postgres-qualification-{report_digest}.json",
            report_path is not None and report_path.is_file() and not report_path.is_symlink(),
            report_path is not None
            and report_size is not None
            and report_path.stat().st_size == report_size,
        )
    ) and _rehash(
        label="CALPHAD PostgreSQL ledger qualification report",
        path=report_path,
        expected_sha256=report_digest,
        issues=issues,
    )
    if input_integrity and report_path is not None:
        try:
            on_disk = json.loads(
                report_path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
            )
            input_integrity = isinstance(on_disk, dict) and on_disk == _mapping(report)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            input_integrity = False
    if not input_integrity:
        issues.append("CALPHAD ledger report is not the exact content-addressed regular file")
    source = _mapping(report.get("source_manifest"))
    files = [_mapping(item) for item in _sequence(source.get("files"))]
    declared_paths = {str(item.get("path") or "") for item in files}
    exact_source_set = declared_paths == set(REQUIRED_CALPHAD_LEDGER_SOURCE_FILES)
    verified_files = 0
    for item in files:
        relative = str(item.get("path") or "")
        path = _safe_relative_path(repository_root, relative)
        try:
            size_ok = (
                path is not None and _strict_int(item.get("size_bytes")) == path.stat().st_size
            )
        except OSError:
            size_ok = False
        hash_ok = _rehash(
            label=f"CALPHAD ledger source {relative}",
            path=path,
            expected_sha256=item.get("sha256"),
            issues=issues,
        )
        verified_files += int(size_ok and hash_ok)
    source_valid = all(
        (
            exact_source_set,
            _strict_int(source.get("file_count")) == len(files),
            len(files) == len(REQUIRED_CALPHAD_LEDGER_SOURCE_FILES),
            verified_files == len(files),
            _plain_sha256(source.get("aggregate_sha256"))
            == manifest_hash(
                {
                    str(item.get("path")): str(_plain_sha256(item.get("sha256")) or "")
                    for item in files
                }
            ),
        )
    )
    if not source_valid:
        issues.append("CALPHAD ledger source manifest is incomplete or changed")

    test_records = [_mapping(item) for item in _sequence(report.get("tests"))]
    test_names = {str(item.get("name") or "") for item in test_records}
    tests_valid = all(
        (
            test_names == set(REQUIRED_CALPHAD_LEDGER_TESTS),
            len(test_records) == len(REQUIRED_CALPHAD_LEDGER_TESTS),
            all(
                item.get("passed") is True and item.get("skipped") is False for item in test_records
            ),
            _strict_int(_mapping(report.get("summary")).get("passed")) == len(test_records),
            _strict_int(_mapping(report.get("summary")).get("failed")) == 0,
            _strict_int(_mapping(report.get("summary")).get("skipped")) == 0,
        )
    )

    runner = _mapping(report.get("runner"))
    log_record = _mapping(runner.get("go_test_log"))
    log_digest = _plain_sha256(log_record.get("sha256"))
    log_size = _strict_int(log_record.get("size_bytes"))
    log_relative = str(log_record.get("path") or "")
    log_path = (
        _safe_relative_path(report_path.parent, log_relative) if report_path is not None else None
    )
    log_integrity = all(
        (
            log_digest is not None,
            log_size is not None and 0 < log_size <= MAX_CALPHAD_LEDGER_GO_LOG_BYTES,
            log_digest is not None and log_relative == f"calphad-ledger-go-test-{log_digest}.jsonl",
            log_path is not None and log_path.is_file() and not log_path.is_symlink(),
            log_path is not None and log_size is not None and log_path.stat().st_size == log_size,
        )
    ) and _rehash(
        label="CALPHAD ledger Go test log",
        path=log_path,
        expected_sha256=log_digest,
        issues=issues,
    )
    go_log = _calphad_go_test_log_evidence(log_path if log_integrity else None)
    if not go_log["valid"]:
        issues.extend(f"CALPHAD ledger Go test log: {issue}" for issue in go_log["issues"])
    runner_valid = all(
        (
            _sequence(runner.get("command")) == list(CALPHAD_LEDGER_GO_COMMAND),
            runner.get("database_credentials_recorded") is False,
            log_integrity,
            go_log["valid"],
            test_records == go_log["records"],
        )
    )
    if not runner_valid:
        issues.append("CALPHAD ledger runner command/log evidence is incomplete or changed")

    database = _mapping(report.get("database"))
    database_name = str(database.get("database") or "")
    database_tokens = {token.lower() for token in database_name.split("_")}
    observed_database = _mapping(report.get("observed_database"))
    database_valid = all(
        (
            set(database)
            == {
                "scheme",
                "host",
                "port",
                "database",
                "serving_role",
                "migration_role",
                "credentials_recorded",
            },
            database.get("scheme") == "postgresql",
            bool(str(database.get("host") or "").strip()),
            _strict_int(database.get("port")) is not None,
            0 < int(database.get("port", 0)) <= 65535,
            re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", database_name, re.I) is not None,
            bool(database_tokens & {"test", "testing", "ci", "qualification", "sandbox"}),
            not bool(database_tokens & {"prod", "production", "live", "primary", "critical"}),
            database.get("credentials_recorded") is False,
            go_log.get("observed_database_valid") is True,
            observed_database == _mapping(go_log.get("observed_database")),
            observed_database.get("database") == database_name,
            observed_database.get("connection_target_host") == database.get("host"),
            observed_database.get("connection_target_port") == database.get("port"),
            observed_database.get("role") == database.get("serving_role"),
            bool(str(database.get("migration_role") or "").strip()),
            database.get("migration_role") != database.get("serving_role"),
            _sequence(observed_database.get("calphad_owner_roles"))
            == [database.get("migration_role")],
        )
    )
    if not database_valid:
        issues.append("CALPHAD ledger qualification database identity is unsafe or incomplete")
    invariants = _mapping(report.get("postgres_invariants"))
    invariant_records = [
        _mapping(item) for item in _sequence(report.get("postgres_invariant_evidence"))
    ]
    invariants_valid = all(
        (
            set(invariants) == set(REQUIRED_CALPHAD_LEDGER_INVARIANTS),
            invariants == _mapping(go_log.get("invariant_outcomes")),
            all(invariants.get(name) is True for name in REQUIRED_CALPHAD_LEDGER_INVARIANTS),
            invariant_records == _sequence(go_log.get("invariant_records")),
        )
    )
    shape_valid = all(
        (
            report.get("schema_version") == "1",
            report.get("gate") == "calphad-ledger-postgres-qualification",
            report.get("status") == "passed",
            report.get("qualification_database") is True,
            report.get("production_database_used") is False,
            _git_sha(report.get("git_sha")) == git_sha,
            report.get("repository_clean") is True,
            _sequence(report.get("failures")) == [],
        )
    )
    valid = all(
        (
            input_integrity,
            source_valid,
            tests_valid,
            invariants_valid,
            runner_valid,
            database_valid,
            shape_valid,
        )
    )
    return {
        "valid": valid,
        "input_integrity": input_integrity,
        "source_valid": source_valid,
        "verified_source_files": verified_files,
        "tests_valid": tests_valid,
        "invariants_valid": invariants_valid,
        "runner_valid": runner_valid,
        "database_valid": database_valid,
        "go_test_log": go_log,
        "issues": issues,
    }


def _run_calphad_real_http_revalidation(
    repository_root: Path,
    database_input_path: Path,
    inspection_path: Path,
    equilibrium_path: Path,
    runtime_image_id: str,
) -> dict[str, Any]:
    """Post retained Python artifacts through the real Go HTTP verifier again."""

    environment = os.environ.copy()
    environment.update(
        {
            "ULTRA_CALPHAD_DATABASE_INPUT_ARTIFACT": str(database_input_path.resolve()),
            "ULTRA_CALPHAD_INSPECTION_ARTIFACT": str(inspection_path.resolve()),
            "ULTRA_CALPHAD_EQUILIBRIUM_ARTIFACT": str(equilibrium_path.resolve()),
            "ULTRA_CALPHAD_RUNTIME_IMAGE_ID": runtime_image_id,
        }
    )
    try:
        completed = subprocess.run(
            CALPHAD_REAL_HTTP_REVALIDATION_COMMAND,
            cwd=repository_root / "backend/controlplane",
            env=environment,
            capture_output=True,
            check=False,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "valid": False,
            "command": list(CALPHAD_REAL_HTTP_REVALIDATION_COMMAND),
            "issues": [f"real Go HTTP revalidation could not run: {exc}"],
        }
    terminal_actions: list[tuple[str, str]] = []
    package_actions: list[str] = []
    disallowed_action = False
    issues: list[str] = []
    if (
        len(completed.stdout) > MAX_CALPHAD_LEDGER_GO_LOG_BYTES
        or len(completed.stderr) > MAX_CALPHAD_LEDGER_GO_LOG_BYTES
    ):
        issues.append("real Go HTTP revalidation output exceeded its fixed bound")
    else:
        try:
            for line_number, raw_line in enumerate(completed.stdout.splitlines(), start=1):
                if not raw_line.strip():
                    continue
                event = json.loads(
                    raw_line,
                    parse_constant=_reject_json_constant,
                    object_pairs_hook=_reject_duplicate_json_keys,
                )
                if not isinstance(event, dict):
                    raise ValueError(f"Go JSON line {line_number} is not an object")
                if event.get("Test") == CALPHAD_REAL_HTTP_REVALIDATION_TEST and event.get(
                    "Action"
                ) in {"pass", "fail", "skip"}:
                    terminal_actions.append(
                        (str(event.get("Action")), str(event.get("Package") or ""))
                    )
                if event.get("Package") == CALPHAD_REAL_HTTP_REVALIDATION_PACKAGE:
                    if event.get("Test") in {None, ""} and event.get("Action") in {
                        "pass",
                        "fail",
                    }:
                        package_actions.append(str(event.get("Action")))
                    if event.get("Action") in {"fail", "skip"}:
                        disallowed_action = True
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            issues.append(f"real Go HTTP revalidation output is invalid: {exc}")
    valid = all(
        (
            completed.returncode == 0,
            not issues,
            terminal_actions == [("pass", CALPHAD_REAL_HTTP_REVALIDATION_PACKAGE)],
            package_actions == ["pass"],
            disallowed_action is False,
        )
    )
    if not valid and not issues:
        issues.append("real Go HTTP revalidation did not pass exactly once without skips")
    return {
        "valid": valid,
        "command": list(CALPHAD_REAL_HTTP_REVALIDATION_COMMAND),
        "test": CALPHAD_REAL_HTTP_REVALIDATION_TEST,
        "package": CALPHAD_REAL_HTTP_REVALIDATION_PACKAGE,
        "exit_code": completed.returncode,
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "issues": issues,
    }


def _calphad_cross_language_evidence(
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    metadata: Mapping[str, Any],
    repository_root: Path,
    expected: ExpectedProvenance,
) -> dict[str, Any]:
    """Revalidate the typed-CLI -> Go HTTP -> PostgreSQL qualification bundle."""

    issues: list[str] = []
    report_meta = _mapping(metadata.get("calphad_cross_language_report"))
    manifest_meta = _mapping(metadata.get("calphad_cross_language_report_manifest"))

    report_path_text = str(report_meta.get("path") or "").strip()
    report_path = Path(report_path_text).expanduser().resolve() if report_path_text else None
    report_digest = _plain_sha256(report_meta.get("sha256"))
    report_size = _strict_int(report_meta.get("size_bytes"))
    report_input_integrity = all(
        (
            report_path is not None,
            report_digest is not None,
            report_size is not None and 0 < report_size <= MAX_REPORT_BYTES,
            report_path is not None
            and report_digest is not None
            and report_path.name == f"calphad-cross-language-qualification-{report_digest}.json",
            report_path is not None and report_path.is_file() and not report_path.is_symlink(),
            report_path is not None
            and report_size is not None
            and report_path.stat().st_size == report_size,
        )
    ) and _rehash(
        label="CALPHAD cross-language qualification report",
        path=report_path,
        expected_sha256=report_digest,
        issues=issues,
    )
    if report_input_integrity and report_path is not None:
        try:
            on_disk_report = json.loads(
                report_path.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
            report_input_integrity = on_disk_report == _mapping(report)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            report_input_integrity = False
    if not report_input_integrity:
        issues.append("CALPHAD cross-language report is not the exact content-addressed file")

    manifest_path_text = str(manifest_meta.get("path") or "").strip()
    manifest_path = Path(manifest_path_text).expanduser().resolve() if manifest_path_text else None
    manifest_digest = _plain_sha256(manifest_meta.get("sha256"))
    manifest_size = _strict_int(manifest_meta.get("size_bytes"))
    manifest_input_integrity = all(
        (
            manifest_path is not None,
            manifest_digest is not None,
            manifest_size is not None and 0 < manifest_size <= MAX_REPORT_BYTES,
            manifest_path is not None and manifest_path.name == "report_manifest.json",
            manifest_path is not None
            and manifest_path.is_file()
            and not manifest_path.is_symlink(),
            manifest_path is not None
            and manifest_size is not None
            and manifest_path.stat().st_size == manifest_size,
        )
    ) and _rehash(
        label="CALPHAD cross-language report manifest",
        path=manifest_path,
        expected_sha256=manifest_digest,
        issues=issues,
    )
    if manifest_input_integrity and manifest_path is not None:
        try:
            on_disk_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
            manifest_input_integrity = on_disk_manifest == _mapping(manifest)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            manifest_input_integrity = False
    manifest_report = _mapping(manifest.get("report"))
    declared_report_path_text = str(manifest_report.get("path") or "").strip()
    declared_report_path = (
        (manifest_path.parent / declared_report_path_text).resolve()
        if manifest_path is not None
        and declared_report_path_text
        and not Path(declared_report_path_text).is_absolute()
        else None
    )
    manifest_binding_valid = all(
        (
            set(manifest)
            == {
                "schema_version",
                "report",
                "production_live_qualified",
                "runtime_image_id",
                "expected_git_sha",
            },
            manifest.get("schema_version") == "ultra.calphad.cross-language-report-manifest.v1",
            manifest.get("production_live_qualified") is True,
            _immutable_sha256(manifest.get("runtime_image_id")) == expected.runtime_image,
            _git_sha(manifest.get("expected_git_sha")) == expected.git_sha,
            set(manifest_report) == {"path", "sha256", "size_bytes"},
            report_path is not None and declared_report_path_text == report_path.name,
            declared_report_path == report_path,
            _plain_sha256(manifest_report.get("sha256")) == report_digest,
            _strict_int(manifest_report.get("size_bytes")) == report_size,
        )
    )
    if not manifest_input_integrity or not manifest_binding_valid:
        issues.append("CALPHAD cross-language report manifest is invalid or unbound")

    source_files = [_mapping(item) for item in _sequence(report.get("source_manifest"))]
    source_paths = {str(item.get("path") or "") for item in source_files}
    verified_source_files = 0
    for item in source_files:
        relative = str(item.get("path") or "")
        path = _safe_relative_path(repository_root, relative)
        try:
            size_ok = path is not None and path.stat().st_size == _strict_int(
                item.get("size_bytes")
            )
        except OSError:
            size_ok = False
        hash_ok = _rehash(
            label=f"CALPHAD cross-language source {relative}",
            path=path,
            expected_sha256=item.get("sha256"),
            issues=issues,
        )
        verified_source_files += int(size_ok and hash_ok)
    source_valid = all(
        (
            source_paths == set(REQUIRED_CALPHAD_CROSS_LANGUAGE_SOURCE_FILES),
            len(source_files) == len(REQUIRED_CALPHAD_CROSS_LANGUAGE_SOURCE_FILES),
            verified_source_files == len(REQUIRED_CALPHAD_CROSS_LANGUAGE_SOURCE_FILES),
        )
    )
    if not source_valid:
        issues.append("CALPHAD cross-language source manifest is incomplete or changed")

    repository = _mapping(report.get("repository"))
    generation = _mapping(report.get("generation"))
    sandbox_policy = _mapping(generation.get("sandbox_policy"))
    image_inspect = _mapping(generation.get("docker_image_inspect"))
    image_inspect_digest = _plain_sha256(image_inspect.get("sha256"))
    image_inspect_size = _strict_int(image_inspect.get("size_bytes"))
    image_inspect_relative = str(image_inspect.get("path") or "").strip()
    image_inspect_path = (
        report_path.parent / f"docker-image-inspect-{image_inspect_digest}.json"
        if report_path is not None and image_inspect_digest is not None
        else None
    )
    image_inspect_integrity = all(
        (
            image_inspect_digest is not None,
            image_inspect_size is not None and 0 < image_inspect_size <= MAX_REPORT_BYTES,
            image_inspect_relative == f"docker-image-inspect-{image_inspect_digest}.json",
            image_inspect_path is not None
            and image_inspect_path.is_file()
            and not image_inspect_path.is_symlink(),
            image_inspect_path is not None
            and image_inspect_size is not None
            and image_inspect_path.stat().st_size == image_inspect_size,
        )
    ) and _rehash(
        label="CALPHAD cross-language Docker image inspection",
        path=image_inspect_path,
        expected_sha256=image_inspect_digest,
        issues=issues,
    )
    inspected_image_valid = False
    if image_inspect_integrity and image_inspect_path is not None:
        try:
            raw_inspection = json.loads(
                image_inspect_path.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
            inspected = raw_inspection[0] if isinstance(raw_inspection, list) else None
            config = _mapping(_mapping(inspected).get("Config"))
            labels = _mapping(config.get("Labels"))
            environment = {
                item.split("=", 1)[0]: item.split("=", 1)[1]
                for item in _sequence(config.get("Env"))
                if isinstance(item, str) and "=" in item
            }
            inspected_image_valid = all(
                (
                    isinstance(raw_inspection, list) and len(raw_inspection) == 1,
                    _immutable_sha256(_mapping(inspected).get("Id")) == expected.runtime_image,
                    _git_sha(labels.get("org.opencontainers.image.revision")) == expected.git_sha,
                    labels.get("org.opencontainers.image.title") == generation.get("image_title"),
                    environment.get("PYTHONPATH") == generation.get("pythonpath"),
                )
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, IndexError):
            inspected_image_valid = False
    generation_valid = all(
        (
            set(generation)
            == {
                "mode",
                "runtime_identity_kind",
                "image_ref",
                "runtime_image_id",
                "image_title",
                "image_revision",
                "pythonpath",
                "image_inspected",
                "docker_image_inspect",
                "pycalphad_version",
                "sandbox_policy",
            },
            repository.get("clean") is True,
            _git_sha(repository.get("head_sha")) == expected.git_sha,
            generation.get("mode") == "pinned_image",
            generation.get("runtime_identity_kind") == "immutable_oci_image",
            generation.get("image_inspected") is True,
            _immutable_sha256(generation.get("runtime_image_id")) == expected.runtime_image,
            _git_sha(generation.get("image_revision")) == expected.git_sha,
            generation.get("image_title") == "Ultra Deep Agents scientific sandbox",
            generation.get("pythonpath") == "/opt/ultra-runtime",
            generation.get("pycalphad_version") == "0.11.2",
            set(sandbox_policy)
            == {
                "enforced_by_gate",
                "network",
                "read_only_root_filesystem",
                "no_new_privileges",
                "cap_drop_all",
                "cpus_at_most",
                "memory_bytes_at_most",
                "pids_at_most",
            },
            sandbox_policy.get("enforced_by_gate") is True,
            sandbox_policy.get("network") == "none",
            sandbox_policy.get("read_only_root_filesystem") is True,
            sandbox_policy.get("no_new_privileges") is True,
            sandbox_policy.get("cap_drop_all") is True,
            _strict_int(sandbox_policy.get("cpus_at_most")) == 8,
            _strict_int(sandbox_policy.get("memory_bytes_at_most")) == 32 * 1024**3,
            _strict_int(sandbox_policy.get("pids_at_most")) == 4096,
            image_inspect_integrity,
            inspected_image_valid,
        )
    )
    if not generation_valid:
        issues.append("CALPHAD cross-language generation is not bound to the expected image/Git")

    resource = _mapping(report.get("resource"))
    resource_sha = _plain_sha256(resource.get("database_sha256"))
    resource_size = _strict_int(resource.get("database_size_bytes"))
    resource_format = str(resource.get("database_format") or "").strip().lower()
    resource_pressure_limits = _sequence(resource.get("assessment_pressure_limits_Pa"))
    resource_valid = all(
        (
            set(resource)
            == {
                "resource_id",
                "database_id",
                "database_sha256",
                "database_size_bytes",
                "database_format",
                "assessment_pressure_limits_Pa",
                "license_id",
                "source",
            },
            bool(str(resource.get("resource_id") or "").strip()),
            bool(str(resource.get("database_id") or "").strip()),
            resource_sha is not None,
            resource_size is not None and resource_size > 0,
            resource_format in {"tdb", "dat"},
            resource_pressure_limits == [101325.0, 101325.0],
            bool(str(resource.get("license_id") or "").strip()),
            bool(str(resource.get("source") or "").strip()),
        )
    )
    if not resource_valid:
        issues.append("CALPHAD cross-language resource identity is incomplete")

    typed_artifacts = _mapping(report.get("typed_cli_artifacts"))
    artifact_evidence: dict[str, dict[str, Any]] = {}
    artifact_paths: dict[str, Path] = {}
    artifacts_valid = report_path is not None and set(typed_artifacts) == {
        "database_input",
        "inspect",
        "equilibrium",
    }
    database_record = _mapping(typed_artifacts.get("database_input"))
    database_digest = _plain_sha256(database_record.get("sha256"))
    database_size = _strict_int(database_record.get("size_bytes"))
    database_format = str(database_record.get("format") or "").strip().lower()
    database_expected_relative = (
        f"artifacts/database/{database_digest}.{database_format}"
        if database_digest is not None and database_format in {"tdb", "dat"}
        else ""
    )
    database_expected_path = (
        report_path.parent / database_expected_relative
        if report_path is not None and database_expected_relative
        else None
    )
    database_declared_path_text = str(database_record.get("path") or "").strip()
    database_declared_path = (
        (report_path.parent / database_declared_path_text).resolve()
        if report_path is not None
        and database_declared_path_text
        and not Path(database_declared_path_text).is_absolute()
        else None
    )
    database_artifact_valid = all(
        (
            set(database_record) == {"path", "sha256", "size_bytes", "format"},
            database_digest is not None and database_digest == resource_sha,
            database_size is not None and database_size == resource_size,
            database_size is not None and 0 < database_size <= 64 * 1024 * 1024,
            database_format in {"tdb", "dat"} and database_format == resource_format,
            database_declared_path_text == database_expected_relative,
            database_declared_path == database_expected_path,
            database_expected_path is not None
            and database_expected_path.is_file()
            and not database_expected_path.is_symlink(),
            database_expected_path is not None
            and database_size is not None
            and database_expected_path.stat().st_size == database_size,
        )
    ) and _rehash(
        label="CALPHAD cross-language database input artifact",
        path=database_expected_path,
        expected_sha256=database_digest,
        issues=issues,
    )
    artifacts_valid = artifacts_valid and database_artifact_valid
    if database_artifact_valid and database_expected_path is not None:
        artifact_paths["database_input"] = database_expected_path
    artifact_evidence["database_input"] = {
        "valid": database_artifact_valid,
        "sha256": database_digest,
        "size_bytes": database_size,
        "format": database_format,
    }
    for operation, directory in (("inspect", "inspection"), ("equilibrium", "equilibrium")):
        record = _mapping(typed_artifacts.get(operation))
        expected_record_keys = {"path", "sha256", "size_bytes"}
        if operation == "equilibrium":
            expected_record_keys.add("inspection_artifact_sha256")
        digest = _plain_sha256(record.get("sha256"))
        size = _strict_int(record.get("size_bytes"))
        expected_path = (
            report_path.parent / "artifacts" / directory / f"{digest}.json"
            if report_path is not None and digest is not None
            else None
        )
        expected_relative = f"artifacts/{directory}/{digest}.json" if digest is not None else ""
        declared_path_text = str(record.get("path") or "").strip()
        declared_path = (
            (report_path.parent / declared_path_text).resolve()
            if report_path is not None
            and declared_path_text
            and not Path(declared_path_text).is_absolute()
            else None
        )
        artifact_valid = all(
            (
                set(record) == expected_record_keys,
                digest is not None,
                size is not None and 0 < size <= 32 * 1024 * 1024,
                declared_path_text == expected_relative,
                declared_path == expected_path,
                expected_path is not None
                and expected_path.is_file()
                and not expected_path.is_symlink(),
                expected_path is not None
                and size is not None
                and expected_path.stat().st_size == size,
            )
        ) and _rehash(
            label=f"CALPHAD cross-language {operation} artifact",
            path=expected_path,
            expected_sha256=digest,
            issues=issues,
        )
        evidence: dict[str, Any] = {}
        if artifact_valid and expected_path is not None:
            try:
                decoded = json.loads(
                    expected_path.read_text(encoding="utf-8"),
                    parse_constant=_reject_json_constant,
                    object_pairs_hook=_reject_duplicate_json_keys,
                )
                evidence = _mapping(decoded)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
                artifact_valid = False
        request = _mapping(evidence.get("request"))
        binding = _mapping(evidence.get("database_binding"))
        result = _mapping(evidence.get("result"))
        execution = _mapping(evidence.get("execution_contract"))
        persistence = _mapping(evidence.get("validation_persistence"))
        binding_valid = all(
            (
                set(binding)
                == {
                    "kind",
                    "database_id",
                    "resource_id",
                    "database_format",
                    "sha256",
                    "size_bytes",
                    "source",
                    "license_id",
                    "assessment_scope",
                    "reference_state",
                    "temperature_limits_K",
                    "assessment_pressure_limits_Pa",
                    "binding_schema",
                    "binding_authority",
                    "declaration_authority",
                },
                binding.get("kind") == "resource",
                binding.get("database_id") == resource.get("database_id"),
                binding.get("resource_id") == resource.get("resource_id"),
                binding.get("database_format") in {"tdb", "dat"},
                _plain_sha256(binding.get("sha256")) == resource_sha,
                _strict_int(binding.get("size_bytes")) == resource_size,
                binding.get("source") == resource.get("source"),
                binding.get("license_id") == resource.get("license_id"),
                bool(str(binding.get("assessment_scope") or "").strip()),
                bool(str(binding.get("reference_state") or "").strip()),
                len(_sequence(binding.get("temperature_limits_K"))) == 2,
                binding.get("assessment_pressure_limits_Pa") == [101325.0, 101325.0],
                binding.get("binding_schema") == "ultra.selected_resource.v1",
                binding.get("binding_authority") == "control_resource_catalog",
                binding.get("declaration_authority") == "resource_owner",
            )
        )
        execution_valid = all(
            (
                set(execution)
                == {
                    "interface",
                    "caller_code_accepted",
                    "caller_models_or_solver_options_accepted",
                    "network",
                    "no_new_privileges",
                    "read_only_root_filesystem",
                    "cap_drop_all",
                    "cpus_at_most",
                    "memory_bytes_at_most",
                    "pids_at_most",
                    "runtime_image_id",
                    "max_components",
                    "max_phases",
                    "max_axis_values",
                    "max_grid_points",
                    "wall_time_seconds",
                    "max_result_bytes",
                },
                execution.get("interface")
                == "fixed ultra_deepagents.materials.calphad public surface",
                execution.get("caller_code_accepted") is False,
                execution.get("caller_models_or_solver_options_accepted") is False,
                execution.get("network") == "none",
                execution.get("no_new_privileges") is True,
                execution.get("read_only_root_filesystem") is True,
                execution.get("cap_drop_all") is True,
                _finite_number(execution.get("cpus_at_most")) == 8,
                _strict_int(execution.get("memory_bytes_at_most")) == 32 * 1024**3,
                _strict_int(execution.get("pids_at_most")) == 4096,
                _immutable_sha256(execution.get("runtime_image_id")) == expected.runtime_image,
                _strict_int(execution.get("max_components")) == 32,
                _strict_int(execution.get("max_phases")) == 128,
                _strict_int(execution.get("max_axis_values")) == 64,
                _strict_int(execution.get("max_grid_points")) == 256,
                _finite_number(execution.get("wall_time_seconds")) == 30,
                _strict_int(execution.get("max_result_bytes")) == 16 * 1024 * 1024,
            )
        )
        persistence_valid = all(
            (
                set(persistence) == {"catalog_status", "catalog_metadata_updated", "mode", "note"},
                persistence.get("catalog_status") == "pending",
                persistence.get("catalog_metadata_updated") is False,
                persistence.get("mode") == "immutable_per_run_evidence",
                bool(str(persistence.get("note") or "").strip()),
            )
        )
        artifact_valid = artifact_valid and all(
            (
                set(evidence)
                == {
                    "schema_version",
                    "operation",
                    "database_binding",
                    "request",
                    "result",
                    "execution_contract",
                    "validation_persistence",
                },
                evidence.get("schema_version") == "ultra.calphad.tool-evidence.v3",
                evidence.get("operation") == operation,
                request.get("operation") == operation,
                _immutable_sha256(request.get("runtime_image_id")) == expected.runtime_image,
                binding_valid,
                execution_valid,
                persistence_valid,
            )
        )
        result_database = result if operation == "inspect" else _mapping(result.get("database"))
        result_name = str(result_database.get("name") or "")
        result_path = str(result_database.get("path") or "")
        result_suffix = Path(result_name).suffix.lower()
        result_identity_valid = all(
            (
                result_database.get("schema_version") == "1",
                _plain_sha256(result_database.get("sha256")) == resource_sha,
                _strict_int(result_database.get("size_bytes")) == resource_size,
                result_database.get("pycalphad_version") == "0.11.2",
                _plain_sha256(result_database.get("manifest_sha256")) is not None,
                result_database.get("format") == binding.get("database_format"),
                result_suffix == f".{binding.get('database_format')}",
                result_name == f"{resource_sha}{result_suffix}",
                result_path == f"/workspace/.ultra/calphad/staged/{result_name}",
            )
        )
        artifact_valid = artifact_valid and result_identity_valid
        if operation == "equilibrium":
            artifact_valid = artifact_valid and all(
                (
                    set(request)
                    == {
                        "operation",
                        "runtime_image_id",
                        "selection",
                        "inspection_artifact_sha256",
                        "conditions",
                    },
                    result.get("schema_version") == "ultra.calphad.equilibrium.v2",
                    isinstance(result.get("request"), dict),
                    isinstance(result.get("result"), dict),
                    isinstance(result.get("warnings"), list),
                    isinstance(result.get("evidence"), dict),
                    _plain_sha256(request.get("inspection_artifact_sha256"))
                    == _plain_sha256(_mapping(typed_artifacts.get("inspect")).get("sha256")),
                    _plain_sha256(record.get("inspection_artifact_sha256"))
                    == _plain_sha256(_mapping(typed_artifacts.get("inspect")).get("sha256")),
                )
            )
        else:
            artifact_valid = artifact_valid and set(request) == {
                "operation",
                "runtime_image_id",
                "selection",
            }
        artifacts_valid = artifacts_valid and artifact_valid
        if expected_path is not None:
            artifact_paths[operation] = expected_path
        artifact_evidence[operation] = {
            "valid": artifact_valid,
            "sha256": digest,
            "size_bytes": size,
        }
    if not artifacts_valid:
        issues.append("CALPHAD typed CLI artifacts are missing, changed, or unbound")
    if artifacts_valid and {"database_input", "inspect", "equilibrium"}.issubset(artifact_paths):
        real_http_revalidation = _run_calphad_real_http_revalidation(
            repository_root,
            artifact_paths["database_input"],
            artifact_paths["inspect"],
            artifact_paths["equilibrium"],
            expected.runtime_image,
        )
    else:
        real_http_revalidation = {
            "valid": False,
            "issues": ["retained artifacts are unavailable for real Go HTTP revalidation"],
        }
    if real_http_revalidation.get("valid") is not True:
        issues.extend(
            f"CALPHAD real HTTP revalidation: {issue}"
            for issue in _sequence(real_http_revalidation.get("issues"))
        )

    backend = _mapping(report.get("backend"))
    backend_database = _mapping(backend.get("database"))
    inspect_backend = _mapping(backend.get("inspect"))
    equilibrium_backend = _mapping(backend.get("equilibrium"))
    inspect_artifact = _mapping(typed_artifacts.get("inspect"))
    equilibrium_artifact = _mapping(typed_artifacts.get("equilibrium"))
    inventory_sha = _plain_sha256(backend.get("database_inventory_sha256"))
    inspect_request_sha = _plain_sha256(inspect_backend.get("request_sha256"))
    equilibrium_request_sha = _plain_sha256(equilibrium_backend.get("request_sha256"))
    backend_server_port = _strict_int(backend_database.get("server_port"))
    backend_connection_port = _strict_int(backend_database.get("connection_target_port"))
    backend_database_name = str(backend_database.get("name") or "")
    backend_database_tokens = {
        token.lower() for token in re.split(r"[_-]+", backend_database_name) if token
    }
    backend_valid = all(
        (
            set(backend)
            == {
                "command",
                "test",
                "go_test_log",
                "schema_version",
                "live_http_callback",
                "live_postgres",
                "database",
                "resource_id",
                "revision_id",
                "run_id",
                "runtime_image_id",
                "pycalphad_version",
                "database_sha256",
                "database_size_bytes",
                "database_format",
                "assessment_pressure_limits_Pa",
                "database_inventory_sha256",
                "inspect",
                "equilibrium",
            },
            backend.get("schema_version") == "ultra.calphad.cross-language-qualification.v1",
            _sequence(backend.get("command"))
            == [
                "go",
                "test",
                "-json",
                "-count=1",
                "./integration",
                "-run",
                "^TestCalphadTypedCLIHTTPPostgresQualification$",
            ],
            _mapping(backend.get("test"))
            == {
                "name": "TestCalphadTypedCLIHTTPPostgresQualification",
                "package": "github.com/amilworks/bisque-ultra/backend/controlplane/integration",
                "action": "pass",
            },
            backend.get("live_http_callback") is True,
            backend.get("live_postgres") is True,
            _immutable_sha256(backend.get("runtime_image_id")) == expected.runtime_image,
            backend.get("pycalphad_version") == "0.11.2",
            backend.get("resource_id") == resource.get("resource_id"),
            _plain_sha256(backend.get("database_sha256")) == resource_sha,
            _strict_int(backend.get("database_size_bytes")) == resource_size,
            backend.get("database_format") == resource_format,
            _sequence(backend.get("assessment_pressure_limits_Pa")) == resource_pressure_limits,
            inventory_sha is not None,
            set(inspect_backend)
            == {
                "evidence_sha256",
                "evidence_size_bytes",
                "request_sha256",
                "evidence_retention",
                "promotable",
                "postgres_bytes_exact",
            },
            set(equilibrium_backend)
            == {
                "evidence_sha256",
                "evidence_size_bytes",
                "request_sha256",
                "inspection_evidence_sha256",
                "evidence_retention",
                "promotable",
                "postgres_bytes_exact",
            },
            _plain_sha256(inspect_backend.get("evidence_sha256"))
            == _plain_sha256(inspect_artifact.get("sha256")),
            _strict_int(inspect_backend.get("evidence_size_bytes"))
            == _strict_int(inspect_artifact.get("size_bytes")),
            _plain_sha256(equilibrium_backend.get("evidence_sha256"))
            == _plain_sha256(equilibrium_artifact.get("sha256")),
            _strict_int(equilibrium_backend.get("evidence_size_bytes"))
            == _strict_int(equilibrium_artifact.get("size_bytes")),
            inspect_request_sha is not None,
            equilibrium_request_sha is not None,
            inspect_request_sha != equilibrium_request_sha,
            _plain_sha256(equilibrium_backend.get("inspection_evidence_sha256"))
            == _plain_sha256(inspect_artifact.get("sha256")),
            inspect_backend.get("evidence_retention") == "retained",
            equilibrium_backend.get("evidence_retention") == "retained",
            inspect_backend.get("promotable") is True,
            equilibrium_backend.get("promotable") is True,
            inspect_backend.get("postgres_bytes_exact") is True,
            equilibrium_backend.get("postgres_bytes_exact") is True,
            set(backend_database)
            == {
                "name",
                "server_address",
                "server_port",
                "connection_target_host",
                "connection_target_port",
                "transaction_read_only",
                "serving_role",
                "migration_role",
                "serving_role_superuser",
                "serving_role_create_role",
                "serving_role_create_database",
                "serving_role_replication",
                "serving_role_bypass_rls",
                "serving_role_owned_tables",
                "serving_role_owned_functions",
                "calphad_owner_roles",
                "calphad_reachable_roles",
                "calphad_owner_role_reachable",
                "public_schema_owner",
                "public_owner_role_reachable",
                "can_create_public_schema",
                "serving_role_select_all",
                "serving_role_insert_all",
                "serving_role_insert_any",
                "serving_role_execute_create_revision",
                "serving_role_execute_append_validation",
                "serving_writer_functions_exact",
                "serving_execute_unexpected_writer",
                "serving_role_execute_internal",
                "serving_role_public_execute",
                "serving_unexpected_table_acl_grantees",
                "serving_unexpected_function_acl_grantees",
                "serving_role_mutation_privilege",
            },
            re.fullmatch(r"[A-Za-z0-9_-]{1,63}", backend_database_name) is not None,
            bool(
                backend_database_tokens
                & {"ci", "test", "tests", "testing", "qualification", "qual", "sandbox"}
            ),
            not bool(
                backend_database_tokens & {"prod", "production", "live", "primary", "critical"}
            ),
            bool(str(backend_database.get("server_address") or "").strip()),
            backend_server_port is not None,
            backend_server_port is not None and 0 < backend_server_port <= 65535,
            bool(str(backend_database.get("connection_target_host") or "").strip()),
            backend_connection_port is not None,
            backend_connection_port is not None and 0 < backend_connection_port <= 65535,
            backend_database.get("transaction_read_only") == "off",
            bool(str(backend_database.get("serving_role") or "").strip()),
            bool(str(backend_database.get("migration_role") or "").strip()),
            backend_database.get("serving_role") != backend_database.get("migration_role"),
            backend_database.get("serving_role_superuser") is False,
            backend_database.get("serving_role_create_database") is False,
            backend_database.get("serving_role_replication") is False,
            backend_database.get("serving_role_owned_tables") == [],
            backend_database.get("serving_role_owned_functions") == [],
            backend_database.get("serving_role_select_all") is True,
            backend_database.get("serving_role_insert_all") is False,
            backend_database.get("serving_role_insert_any") is False,
            backend_database.get("serving_role_execute_create_revision") is True,
            backend_database.get("serving_role_execute_append_validation") is True,
            backend_database.get("serving_writer_functions_exact") is True,
            backend_database.get("serving_execute_unexpected_writer") is False,
            backend_database.get("serving_role_execute_internal") is False,
            backend_database.get("serving_role_public_execute") is False,
            backend_database.get("serving_role_mutation_privilege") is False,
            backend_database.get("serving_role_create_role") is False,
            backend_database.get("serving_role_bypass_rls") is False,
            isinstance(backend_database.get("calphad_owner_roles"), list),
            bool(backend_database.get("calphad_owner_roles")),
            all(
                isinstance(role, str) and bool(role.strip())
                for role in backend_database.get("calphad_owner_roles", [])
            ),
            backend_database.get("calphad_reachable_roles") == [],
            backend_database.get("calphad_owner_role_reachable") is False,
            bool(str(backend_database.get("public_schema_owner") or "").strip()),
            backend_database.get("public_owner_role_reachable") is False,
            backend_database.get("can_create_public_schema") is False,
            backend_database.get("serving_unexpected_table_acl_grantees") == [],
            backend_database.get("serving_unexpected_function_acl_grantees") == [],
        )
    )
    if not backend_valid:
        issues.append("CALPHAD live Go HTTP/PostgreSQL evidence is incomplete or unsafe")

    go_log = _mapping(backend.get("go_test_log"))
    go_log_digest = _plain_sha256(go_log.get("sha256"))
    go_log_size = _strict_int(go_log.get("size_bytes"))
    go_log_path = (
        report_path.parent / f"go-test-{go_log_digest}.jsonl"
        if report_path is not None and go_log_digest is not None
        else None
    )
    declared_go_log_text = str(go_log.get("path") or "").strip()
    declared_go_log = (
        (report_path.parent / declared_go_log_text).resolve()
        if report_path is not None
        and declared_go_log_text
        and not Path(declared_go_log_text).is_absolute()
        else None
    )
    go_log_valid = all(
        (
            go_log_digest is not None,
            go_log_size is not None and 0 < go_log_size <= MAX_CALPHAD_LEDGER_GO_LOG_BYTES,
            declared_go_log_text == f"go-test-{go_log_digest}.jsonl",
            declared_go_log == go_log_path,
            go_log_path is not None and go_log_path.is_file() and not go_log_path.is_symlink(),
            go_log_path is not None
            and go_log_size is not None
            and go_log_path.stat().st_size == go_log_size,
        )
    ) and _rehash(
        label="CALPHAD cross-language Go test log",
        path=go_log_path,
        expected_sha256=go_log_digest,
        issues=issues,
    )
    go_log_semantics_valid = False
    if go_log_valid and go_log_path is not None:
        terminal_actions: list[tuple[str, str]] = []
        package_terminal_actions: list[str] = []
        package_fail_or_skip = False
        markers: list[dict[str, Any]] = []
        try:
            for line_number, raw_line in enumerate(
                go_log_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not raw_line.strip():
                    continue
                event = json.loads(
                    raw_line,
                    parse_constant=_reject_json_constant,
                    object_pairs_hook=_reject_duplicate_json_keys,
                )
                if not isinstance(event, dict):
                    raise ValueError(f"Go JSON line {line_number} is not an object")
                if event.get(
                    "Test"
                ) == "TestCalphadTypedCLIHTTPPostgresQualification" and event.get("Action") in {
                    "pass",
                    "fail",
                    "skip",
                }:
                    terminal_actions.append(
                        (str(event.get("Action")), str(event.get("Package") or ""))
                    )
                if event.get("Package") == (
                    "github.com/amilworks/bisque-ultra/backend/controlplane/integration"
                ):
                    if event.get("Test") in {None, ""} and event.get("Action") in {
                        "pass",
                        "fail",
                    }:
                        package_terminal_actions.append(str(event.get("Action")))
                    if event.get("Action") in {"fail", "skip"}:
                        package_fail_or_skip = True
                output = event.get("Output")
                marker_prefix = "CALPHAD_CROSS_LANGUAGE_EVIDENCE "
                if isinstance(output, str) and marker_prefix in output:
                    encoded_marker = output.split(marker_prefix, 1)[1].strip()
                    marker = json.loads(
                        encoded_marker,
                        parse_constant=_reject_json_constant,
                        object_pairs_hook=_reject_duplicate_json_keys,
                    )
                    if not isinstance(marker, dict):
                        raise ValueError("Go evidence marker is not an object")
                    markers.append(marker)
            expected_marker = {
                key: value
                for key, value in backend.items()
                if key not in {"command", "test", "go_test_log"}
            }
            go_log_semantics_valid = all(
                (
                    terminal_actions
                    == [
                        (
                            "pass",
                            "github.com/amilworks/bisque-ultra/backend/controlplane/integration",
                        )
                    ],
                    package_terminal_actions == ["pass"],
                    package_fail_or_skip is False,
                    len(markers) == 1,
                    len(markers) == 1 and markers[0] == expected_marker,
                )
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            go_log_semantics_valid = False
    if not go_log_valid or not go_log_semantics_valid:
        issues.append("CALPHAD cross-language Go test log is missing or changed")

    checks = _mapping(report.get("checks"))
    checks_valid = set(checks) == set(REQUIRED_CALPHAD_CROSS_LANGUAGE_CHECKS) and all(
        checks.get(name) is True for name in REQUIRED_CALPHAD_CROSS_LANGUAGE_CHECKS
    )
    shape_valid = all(
        (
            set(report)
            == {
                "schema_version",
                "gate",
                "generated_at_utc",
                "expected_git_sha",
                "repository",
                "source_manifest",
                "generation",
                "resource",
                "typed_cli_artifacts",
                "backend",
                "checks",
                "production_live_qualified",
                "promotable",
                "status",
            },
            report.get("schema_version") == "ultra.calphad.cross-language-gate.v1",
            report.get("gate") == "calphad-typed-cli-http-postgres-cross-language",
            bool(str(report.get("generated_at_utc") or "").strip()),
            report.get("status") == "qualified",
            report.get("production_live_qualified") is True,
            report.get("promotable") is True,
            _git_sha(report.get("expected_git_sha")) == expected.git_sha,
            set(repository) == {"head_sha", "clean"},
            checks_valid,
        )
    )
    if not shape_valid:
        issues.append("CALPHAD cross-language report is not a live promotable qualification")

    valid = all(
        (
            report_input_integrity,
            manifest_input_integrity,
            manifest_binding_valid,
            source_valid,
            generation_valid,
            resource_valid,
            artifacts_valid,
            real_http_revalidation.get("valid") is True,
            backend_valid,
            go_log_valid,
            go_log_semantics_valid,
            shape_valid,
        )
    )
    return {
        "valid": valid,
        "input_integrity": report_input_integrity,
        "manifest_integrity": manifest_input_integrity and manifest_binding_valid,
        "source_files_verified": verified_source_files,
        "expected_source_files": len(REQUIRED_CALPHAD_CROSS_LANGUAGE_SOURCE_FILES),
        "runtime_image_id": generation.get("runtime_image_id"),
        "live_http_callback": backend.get("live_http_callback") is True,
        "live_postgres": backend.get("live_postgres") is True,
        "go_test_log_semantics_valid": go_log_semantics_valid,
        "artifacts": artifact_evidence,
        "real_http_revalidation": real_http_revalidation,
        "database_inventory_sha256": inventory_sha,
        "issues": issues,
    }


def _deterministic_invariant_evidence(
    deterministic: Mapping[str, Any], junit: Mapping[str, Any]
) -> dict[str, Any]:
    records = [_mapping(item) for item in _sequence(deterministic.get("invariants"))]
    summary = _mapping(deterministic.get("invariant_evidence"))
    validator_ids = [str(record.get("validator_id") or "") for record in records]
    test_ids = [str(record.get("test_id") or "") for record in records]
    record_shape_valid = all(
        (
            record.get("schema_version") == "1",
            record.get("required") is True,
            record.get("outcome") == "pass",
            bool(str(record.get("validator_id") or "").strip()),
            bool(str(record.get("test_id") or "").strip()),
            bool(_mapping(record.get("observed"))),
            bool(_mapping(record.get("expected"))),
            bool(str(record.get("tolerance_rationale") or "").strip()),
            bool(str(record.get("units") or "").strip()),
            bool(str(record.get("convention") or "").strip()),
            bool(_mapping(record.get("library_versions"))),
            all(
                str(name).strip() and str(version).strip()
                for name, version in _mapping(record.get("library_versions")).items()
            ),
        )
        for record in records
    )
    exact_validators = (
        len(records) == REQUIRED_DOMAIN_INVARIANT_COUNT
        and len(set(validator_ids)) == REQUIRED_DOMAIN_INVARIANT_COUNT
        and set(validator_ids) == set(REQUIRED_DOMAIN_VALIDATORS)
        and len(set(test_ids)) == REQUIRED_DOMAIN_INVARIANT_COUNT
    )
    complete = all(
        (
            summary.get("schema_version") == "1",
            summary.get("junit_property") == "materials_invariant_evidence",
            _strict_int(summary.get("record_count")) == REQUIRED_DOMAIN_INVARIANT_COUNT,
            _strict_int(summary.get("passed")) == REQUIRED_DOMAIN_INVARIANT_COUNT,
            _strict_int(summary.get("failed")) == 0,
            _sequence(summary.get("errors")) == [],
            summary.get("complete") is True,
            _strict_int(junit.get("tests")) == REQUIRED_DOMAIN_INVARIANT_COUNT,
            record_shape_valid,
            exact_validators,
        )
    )
    return {
        "valid": complete,
        "record_count": len(records),
        "passed": sum(record.get("outcome") == "pass" for record in records),
        "validator_ids": sorted(validator_ids),
        "exact_required_validator_set": exact_validators,
        "record_shape_valid": record_shape_valid,
        "summary": summary,
    }


def _positive_finite_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _calphad_runtime_junit_evidence(path: Path | None) -> dict[str, Any]:
    summary = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    issues: list[str] = []
    if path is None:
        return {"valid": False, "summary": summary, "issues": ["JUnit path is missing"]}
    try:
        size = path.stat().st_size
        if size <= 0 or size > MAX_JUNIT_BYTES:
            raise ValueError(f"JUnit size {size} is outside 1..{MAX_JUNIT_BYTES}")
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError, ValueError) as exc:
        return {"valid": False, "summary": summary, "issues": [str(exc)]}

    testcases = [element for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "testcase"]
    identities: list[str] = []
    core_names: list[str] = []
    cli_names: list[str] = []
    outcomes = {"failures": 0, "errors": 0, "skipped": 0}
    for testcase in testcases:
        classname = str(testcase.attrib.get("classname") or "")
        name = str(testcase.attrib.get("name") or "")
        if classname == "tests.test_calphad_runtime":
            core_names.append(name)
        elif classname == "tests.test_calphad_cli":
            cli_names.append(name)
        else:
            issues.append("CALPHAD runtime JUnit contains an unrelated testcase")
        if not name:
            issues.append("CALPHAD runtime JUnit contains an unnamed testcase")
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
        issues.append(f"CALPHAD runtime JUnit has invalid suite counters: {exc}")
    if not leaf_suites or declared != summary:
        issues.append("CALPHAD runtime JUnit counters disagree with testcase outcomes")
    if len(testcases) != CALPHAD_RUNTIME_TEST_COUNT:
        issues.append(
            f"CALPHAD runtime JUnit does not contain exactly {CALPHAD_RUNTIME_TEST_COUNT} tests"
        )
    if len(core_names) != CALPHAD_RUNTIME_CORE_TEST_COUNT:
        issues.append(
            "CALPHAD runtime JUnit does not contain exactly "
            f"{CALPHAD_RUNTIME_CORE_TEST_COUNT} core tests"
        )
    if set(core_names) != set(REQUIRED_CALPHAD_RUNTIME_CORE_TEST_NAMES):
        issues.append("CALPHAD runtime JUnit does not contain the exact required core tests")
    if len(cli_names) != CALPHAD_RUNTIME_CLI_TEST_COUNT:
        issues.append(
            "CALPHAD runtime JUnit does not contain exactly "
            f"{CALPHAD_RUNTIME_CLI_TEST_COUNT} typed CLI tests"
        )
    if set(cli_names) != set(REQUIRED_CALPHAD_RUNTIME_CLI_TEST_NAMES):
        issues.append("CALPHAD runtime JUnit does not contain the exact required CLI test")
    if len(set(identities)) != len(identities):
        issues.append("CALPHAD runtime JUnit contains duplicate testcase identities")
    if any(summary[field] != 0 for field in ("failures", "errors", "skipped")):
        issues.append("CALPHAD runtime JUnit contains a failure, error, or skip")
    return {"valid": not issues, "summary": summary, "issues": issues}


def _calphad_tools_junit_evidence(path: Path | None) -> dict[str, Any]:
    """Independently parse the exact host/worker CALPHAD orchestration suite."""

    summary = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    issues: list[str] = []
    if path is None:
        return {"valid": False, "summary": summary, "issues": ["JUnit path is missing"]}
    try:
        size = path.stat().st_size
        if size <= 0 or size > MAX_JUNIT_BYTES:
            raise ValueError(f"JUnit size {size} is outside 1..{MAX_JUNIT_BYTES}")
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError, ValueError) as exc:
        return {"valid": False, "summary": summary, "issues": [str(exc)]}

    testcases = [element for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "testcase"]
    identities: list[str] = []
    outcomes = {"failures": 0, "errors": 0, "skipped": 0}
    for testcase in testcases:
        classname = str(testcase.attrib.get("classname") or "")
        name = str(testcase.attrib.get("name") or "")
        if classname != "tests.test_calphad_tools":
            issues.append("CALPHAD tools JUnit contains an unrelated testcase")
        if not name:
            issues.append("CALPHAD tools JUnit contains an unnamed testcase")
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
        issues.append(f"CALPHAD tools JUnit has invalid suite counters: {exc}")
    if not leaf_suites or declared != summary:
        issues.append("CALPHAD tools JUnit counters disagree with testcase outcomes")
    if len(testcases) != CALPHAD_TOOLS_TEST_COUNT:
        issues.append(
            f"CALPHAD tools JUnit does not contain exactly {CALPHAD_TOOLS_TEST_COUNT} tests"
        )
    if set(identity.rsplit("::", 1)[-1] for identity in identities) != set(
        REQUIRED_CALPHAD_TOOL_TEST_NAMES
    ):
        issues.append("CALPHAD tools JUnit does not contain the exact required test identities")
    if len(set(identities)) != len(identities):
        issues.append("CALPHAD tools JUnit contains duplicate testcase identities")
    if any(summary[field] != 0 for field in ("failures", "errors", "skipped")):
        issues.append("CALPHAD tools JUnit contains a failure, error, or skip")
    return {"valid": not issues, "summary": summary, "issues": issues}


def _content_manifest_valid(manifest: Mapping[str, Any], *, root: str | None = None) -> bool:
    """Validate a verifier-style per-file content manifest, including its aggregate."""

    value = _mapping(manifest)
    files = [_mapping(item) for item in _sequence(value.get("files"))]
    paths = [str(item.get("path") or "") for item in files]
    records_valid = all(
        (
            len(files) > 0,
            _strict_int(value.get("file_count")) == len(files),
            paths == sorted(paths),
            len(set(paths)) == len(paths),
            all(
                bool(path)
                and (root is None or path == root or path.startswith(f"{root}/"))
                and not Path(path).is_absolute()
                and ".." not in Path(path).parts
                and _plain_sha256(item.get("sha256")) is not None
                and _strict_int(item.get("size_bytes")) is not None
                and int(item["size_bytes"]) >= 0
                for path, item in zip(paths, files, strict=True)
            ),
        )
    )
    return records_valid and _plain_sha256(value.get("aggregate_sha256")) == canonical_json_sha256(
        files
    )


def _repository_file_hashes(
    repository_root: Path,
    relative_paths: Sequence[str],
    *,
    label: str,
) -> tuple[dict[str, str], list[str]]:
    root = repository_root.expanduser().resolve()
    hashes: dict[str, str] = {}
    issues: list[str] = []
    for relative in relative_paths:
        path = _safe_relative_path(root, relative)
        if path is None or not path.is_file() or path.is_symlink():
            issues.append(f"{label}: required regular file is missing: {relative}")
            continue
        try:
            hashes[relative] = sha256_file(path)
        except OSError as exc:
            issues.append(f"{label}: cannot hash {relative} ({exc})")
    return hashes, issues


def _calphad_release_contract_evidence(
    parity: Mapping[str, Any], repository_root: Path
) -> dict[str, Any]:
    hashes, issues = _repository_file_hashes(
        repository_root,
        REQUIRED_CALPHAD_RELEASE_INPUT_FILES,
        label="production CALPHAD release contract",
    )
    manifest_relative = "backend/deepagents_runtime/materials_data/calphad/manifest.json"
    expected = {
        "manifest_sha256": hashes.get(manifest_relative),
        "release_input_sha256s": dict(sorted(hashes.items())),
        "runtime_test_count": CALPHAD_RUNTIME_TEST_COUNT,
        "core_runtime_test_count": CALPHAD_RUNTIME_CORE_TEST_COUNT,
        "typed_cli_test_count": CALPHAD_RUNTIME_CLI_TEST_COUNT,
        "calphad_tools_test_count": CALPHAD_TOOLS_TEST_COUNT,
        "required_adversarial_test_names": sorted(REQUIRED_CALPHAD_ADVERSARIAL_TEST_NAMES),
    }
    exact = (
        len(hashes) == len(REQUIRED_CALPHAD_RELEASE_INPUT_FILES)
        and _mapping(parity.get("calphad_release_contract")) == expected
    )
    if not exact:
        issues.append("production CALPHAD release contract is missing, stale, or not source-bound")
    return {"valid": not issues and exact, "expected": expected, "issues": issues}


def _calphad_pressure_limits(value: Any) -> list[float] | None:
    values = _sequence(value)
    if len(values) != 2:
        return None
    minimum = _finite_number(values[0])
    maximum = _finite_number(values[1])
    if minimum is None or maximum is None or minimum < 1e-9 or minimum > maximum or maximum > 1e12:
        return None
    return [minimum, maximum]


def _calphad_probe_source_evidence(
    report: Mapping[str, Any], repository_root: Path
) -> dict[str, Any]:
    root = repository_root.expanduser().resolve()
    issues: list[str] = []
    manifest_relative = "backend/deepagents_runtime/materials_data/calphad/manifest.json"
    manifest_path = _safe_relative_path(root, manifest_relative)
    manifest: Mapping[str, Any] = {}
    manifest_sha: str | None = None
    if manifest_path is None or not manifest_path.is_file() or manifest_path.is_symlink():
        issues.append("embedded CALPHAD source manifest is missing or not a regular file")
    else:
        try:
            manifest_sha = sha256_file(manifest_path)
            loaded = json.loads(
                manifest_path.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
            )
            if isinstance(loaded, dict):
                manifest = loaded
            else:
                issues.append("embedded CALPHAD source manifest is not an object")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            issues.append(f"embedded CALPHAD source manifest cannot be verified ({exc})")

    materials_root = root / "backend/deepagents_runtime/src/ultra_deepagents/materials"
    material_paths = sorted(materials_root.glob("*.py")) if materials_root.is_dir() else []
    material_hashes: dict[str, str] = {}
    if not material_paths:
        issues.append("embedded CALPHAD materials source package is empty")
    for path in material_paths:
        if not path.is_file() or path.is_symlink():
            issues.append(f"embedded CALPHAD materials source is not regular: {path.name}")
            continue
        try:
            material_hashes[path.name] = sha256_file(path)
        except OSError as exc:
            issues.append(
                f"embedded CALPHAD materials source cannot be hashed: {path.name} ({exc})"
            )

    if (
        report.get("materials_source_hashes") != material_hashes
        or report.get("materials_baked_hashes") != material_hashes
    ):
        issues.append("embedded CALPHAD source/baked module hashes differ from repository bytes")
    if (
        manifest_sha is None
        or report.get("source_manifest_sha256") != manifest_sha
        or report.get("embedded_manifest_sha256") != manifest_sha
    ):
        issues.append("embedded CALPHAD source/embedded manifest hashes are not repository-bound")

    raw_entries = manifest.get("databases")
    raw_records = report.get("databases")
    entries = raw_entries if isinstance(raw_entries, list) else []
    records = raw_records if isinstance(raw_records, list) else []
    if not entries or not records:
        issues.append("embedded CALPHAD manifest/probe has no database records")

    expected_records: dict[str, dict[str, Any]] = {}
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping):
            issues.append("embedded CALPHAD manifest contains a malformed database record")
            continue
        database_id = str(raw_entry.get("database_id") or "")
        filename = str(raw_entry.get("filename") or "")
        relative = Path(filename)
        declared_format = str(raw_entry.get("format") or "")
        pressure_limits = _calphad_pressure_limits(raw_entry.get("assessment_pressure_limits_Pa"))
        if (
            not database_id
            or database_id in expected_records
            or not filename
            or relative.is_absolute()
            or len(relative.parts) != 1
            or ".." in relative.parts
            or declared_format not in {"tdb", "dat"}
            or relative.suffix.casefold() != f".{declared_format}"
            or pressure_limits is None
        ):
            issues.append("embedded CALPHAD manifest database identity/format/pressure is invalid")
            continue
        database_path = manifest_path.parent / relative if manifest_path is not None else None
        if database_path is None or not database_path.is_file() or database_path.is_symlink():
            issues.append(f"embedded CALPHAD database is missing or nonregular: {database_id}")
            continue
        try:
            database_sha = sha256_file(database_path)
            database_size = database_path.stat().st_size
        except OSError as exc:
            issues.append(f"embedded CALPHAD database cannot be hashed: {database_id} ({exc})")
            continue
        if (
            raw_entry.get("sha256") != database_sha
            or _strict_int(raw_entry.get("size_bytes")) != database_size
            or database_size <= 0
        ):
            issues.append(f"embedded CALPHAD manifest differs from database bytes: {database_id}")
        expected_records[database_id] = {
            "database_id": database_id,
            "filename": filename,
            "sha256": database_sha,
            "size_bytes": database_size,
            "format": declared_format,
            "assessment_pressure_limits_Pa": pressure_limits,
            "elements": sorted(str(item) for item in _sequence(raw_entry.get("elements"))),
            "phases": sorted(str(item) for item in _sequence(raw_entry.get("phases"))),
            "pycalphad_parse_supported": True,
            "ultra_inspection_supported": True,
        }

    observed_records: dict[str, dict[str, Any]] = {}
    for raw_record in records:
        if not isinstance(raw_record, Mapping):
            issues.append("embedded CALPHAD probe contains a malformed database record")
            continue
        database_id = str(raw_record.get("database_id") or "")
        if not database_id or database_id in observed_records:
            issues.append("embedded CALPHAD probe has a missing or duplicate database identity")
            continue
        observed_records[database_id] = {
            "database_id": database_id,
            "filename": str(raw_record.get("filename") or ""),
            "sha256": str(raw_record.get("sha256") or ""),
            "size_bytes": _strict_int(raw_record.get("size_bytes")),
            "format": str(raw_record.get("format") or ""),
            "assessment_pressure_limits_Pa": _calphad_pressure_limits(
                raw_record.get("assessment_pressure_limits_Pa")
            ),
            "elements": sorted(str(item) for item in _sequence(raw_record.get("elements"))),
            "phases": sorted(str(item) for item in _sequence(raw_record.get("phases"))),
            "pycalphad_parse_supported": raw_record.get("pycalphad_parse_supported"),
            "ultra_inspection_supported": raw_record.get("ultra_inspection_supported"),
        }
    if (
        len(observed_records) != len(records)
        or observed_records != expected_records
        or _strict_int(report.get("database_count")) != len(expected_records)
    ):
        issues.append("embedded CALPHAD probe database evidence differs from repository registry")

    shape_valid = all(
        (
            _strict_int(report.get("schema_version")) == 1,
            report.get("status") == "passed",
            report.get("equilibrium_schema_version") == CALPHAD_EQUILIBRIUM_SCHEMA,
            report.get("baked_materials_path") == "/opt/ultra-runtime/ultra_deepagents/materials",
            manifest.get("schema_version") == "1",
        )
    )
    if not shape_valid:
        issues.append("embedded CALPHAD probe schema, status, or baked package path is stale")
    return {
        "valid": not issues and shape_valid,
        "manifest_sha256": manifest_sha,
        "material_source_hashes": material_hashes,
        "database_count": len(expected_records),
        "issues": issues,
    }


def _domain_calphad_preflight_valid(domain_report: Mapping[str, Any]) -> bool:
    runtime = _mapping(domain_report.get("runtime"))
    preflight = _mapping(runtime.get("calphad_runtime_preflight"))
    junit = _mapping(preflight.get("junit"))
    time_seconds = _finite_number(junit.get("time_seconds"))
    return all(
        (
            preflight.get("path") == "/outputs/calphad-runtime-junit.xml",
            preflight.get("required") is True,
            preflight.get("validated") is True,
            _strict_int(preflight.get("core_tests")) == CALPHAD_RUNTIME_CORE_TEST_COUNT,
            _strict_int(preflight.get("typed_cli_tests")) == CALPHAD_RUNTIME_CLI_TEST_COUNT,
            _sequence(preflight.get("required_adversarial_test_names"))
            == sorted(REQUIRED_CALPHAD_ADVERSARIAL_TEST_NAMES),
            _strict_int(junit.get("tests")) == CALPHAD_RUNTIME_TEST_COUNT,
            _strict_int(junit.get("failures")) == 0,
            _strict_int(junit.get("errors")) == 0,
            _strict_int(junit.get("skipped")) == 0,
            time_seconds is not None and time_seconds >= 0,
        )
    )


def _domain_calphad_experimental_benchmark_valid(
    domain_report: Mapping[str, Any],
) -> bool:
    wrapper = _mapping(domain_report.get("calphad_experimental_benchmark"))
    report = _mapping(wrapper.get("report"))
    lanes = _mapping(report.get("lanes"))
    calibration = _mapping(lanes.get("calibration"))
    held_out = _mapping(lanes.get("held_out"))
    calibration_metrics = _mapping(calibration.get("metrics"))
    held_out_metrics = _mapping(held_out.get("metrics"))
    calibration_rms = _finite_number(calibration_metrics.get("weighted_rms_z"))
    calibration_max = _finite_number(calibration_metrics.get("max_abs_z"))
    held_out_mae = _finite_number(held_out_metrics.get("mae_K"))
    held_out_max = _finite_number(held_out_metrics.get("max_abs_error_K"))
    observations = [_mapping(item) for item in _sequence(held_out.get("observations"))]
    source_manifest = _mapping(report.get("source_manifest"))
    database_binding = _mapping(report.get("database_binding"))
    return all(
        (
            wrapper.get("relative_path") == "calphad-experimental-benchmark.json",
            _plain_sha256(wrapper.get("sha256")) is not None,
            _strict_int(wrapper.get("size_bytes")) is not None and int(wrapper["size_bytes"]) > 0,
            report.get("schema_version") == "ultra.calphad.experimental_benchmark.v1",
            report.get("benchmark_id") == "materials.calphad.al_co_w_experimental_two_lane.v1",
            report.get("status") == "passed",
            report.get("required_independent_invariant") is True,
            report.get("production_promotion_blocked") is False,
            _sequence(report.get("blocking_reasons")) == [],
            database_binding.get("database_id") == "nist-al-co-w-wang-2017",
            _plain_sha256(database_binding.get("sha256")) is not None,
            source_manifest.get("relative_path")
            == (
                "backend/deepagents_runtime/materials_data/calphad/"
                "experimental_benchmark_manifest.json"
            ),
            _plain_sha256(source_manifest.get("sha256")) is not None,
            calibration.get("classification") == "calibration",
            calibration.get("independent_validation") is False,
            calibration.get("required") is True,
            calibration.get("status") == "passed",
            _strict_int(calibration.get("observation_count")) == 6,
            calibration_rms is not None and calibration_rms <= 1.0,
            calibration_max is not None and calibration_max <= 2.0,
            calibration_metrics.get("weighted_rms_z_max") == 1.0,
            calibration_metrics.get("max_abs_z_max") == 2.0,
            held_out.get("classification") == "held_out",
            held_out.get("independent_validation") is True,
            held_out.get("required") is True,
            held_out.get("status") == "passed",
            _strict_int(held_out.get("observation_count")) == 4,
            len(observations) == 4,
            all(
                observation.get("reported_uncertainty_K") is None
                and observation.get("uncertainty_status") == "not_reported_numerically"
                for observation in observations
            ),
            held_out_mae is not None and held_out_mae <= 20.0,
            held_out_max is not None and held_out_max <= 30.0,
            held_out_metrics.get("mae_K_max") == 20.0,
            held_out_metrics.get("max_abs_error_K_max") == 30.0,
        )
    )


def _release_artifacts_valid(value: Mapping[str, Any]) -> bool:
    artifacts = _mapping(value)
    control = _mapping(artifacts.get("control_binary"))
    frontend = _mapping(artifacts.get("frontend_dist"))
    frontend_files = [_mapping(item) for item in _sequence(frontend.get("files"))]
    index = next(
        (item for item in frontend_files if item.get("path") == "frontend/dist/index.html"),
        None,
    )
    return all(
        (
            set(artifacts) == {"control_binary", "frontend_dist"},
            set(control) == {"path", "sha256", "size_bytes"},
            control.get("path") == "bin/ultra-control",
            _plain_sha256(control.get("sha256")) is not None,
            (_strict_int(control.get("size_bytes")) or 0) > 0,
            frontend.get("path") == "frontend/dist",
            _content_manifest_valid(frontend, root="frontend/dist"),
            index is not None and (_strict_int(index.get("size_bytes")) or 0) > 0,
        )
    )


def _production_bundle_evidence(
    parity: Mapping[str, Any], report_path: Path | None, repository_root: Path
) -> dict[str, Any]:
    """Run the reviewed retained-byte validator and bind it to the release source hash."""

    issues: list[str] = []
    verifier = repository_root / "scripts/verify_production_materials_sandbox.py"
    source_files = [
        _mapping(item)
        for item in _sequence(
            _mapping(_mapping(parity.get("source")).get("required_materials")).get("files")
        )
    ]
    verifier_records = [
        item
        for item in source_files
        if item.get("path") == "scripts/verify_production_materials_sandbox.py"
    ]
    source_bound = all(
        (
            verifier.is_file() and not verifier.is_symlink(),
            len(verifier_records) == 1,
            len(verifier_records) == 1
            and _plain_sha256(verifier_records[0].get("sha256")) == sha256_file(verifier),
            len(verifier_records) == 1
            and _strict_int(verifier_records[0].get("size_bytes")) == verifier.stat().st_size,
        )
    )
    if not source_bound:
        issues.append("production bundle validator is not bound to the release source manifest")
    failures: list[str] = []
    if source_bound and report_path is not None:
        module_name = "_ultra_materials_production_bundle_validator"
        try:
            spec = importlib.util.spec_from_file_location(module_name, verifier)
            if spec is None or spec.loader is None:
                raise ImportError("could not load retained-evidence validator")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            validator = getattr(module, "validate_retained_evidence_bundle")
            raw_failures = validator(parity, report_path.parent)
            failures = [str(value) for value in raw_failures]
        except (AttributeError, ImportError, OSError, TypeError, ValueError) as exc:
            failures = [f"retained-evidence validator failed: {exc}"]
        finally:
            sys.modules.pop(module_name, None)
    else:
        failures.append("retained-evidence validator could not run")
    issues.extend(failures)
    bundle = _mapping(parity.get("evidence_bundle"))
    shape_valid = all(
        (
            _strict_int(bundle.get("schema_version")) == 1,
            bundle.get("promotable") is True,
            _mapping(bundle.get("image_identity")) == _mapping(parity.get("image_identity")),
            not failures,
        )
    )
    if not shape_valid:
        issues.append("production retained-evidence bundle is incomplete or changed")
    return {
        "valid": source_bound and shape_valid,
        "source_bound": source_bound,
        "validator_failures": failures,
        "issues": issues,
    }


def _production_parity_evidence(
    parity_report: Mapping[str, Any],
    expected: ExpectedProvenance,
    metadata: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    """Revalidate the content-addressed full production-image parity report."""

    parity = _mapping(parity_report)
    meta = _mapping(metadata)
    issues: list[str] = []

    report_digest = _plain_sha256(meta.get("sha256"))
    report_path_text = str(meta.get("path") or "").strip()
    report_path = Path(report_path_text).expanduser().resolve() if report_path_text else None
    report_size = _strict_int(meta.get("size_bytes"))
    input_integrity = all(
        (
            report_digest is not None,
            report_path is not None,
            report_size is not None and 0 < report_size <= MAX_REPORT_BYTES,
            report_path is not None
            and report_digest is not None
            and report_path.name == f"{PRODUCTION_PARITY_REPORT_PREFIX}{report_digest}.json",
        )
    )
    if input_integrity and report_path is not None and report_digest is not None:
        try:
            on_disk = json.loads(
                report_path.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
            )
            input_integrity = all(
                (
                    report_path.is_file(),
                    not report_path.is_symlink(),
                    report_path.stat().st_size == report_size,
                    sha256_file(report_path) == report_digest,
                    isinstance(on_disk, dict),
                    on_disk == parity,
                )
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            input_integrity = False
    if not input_integrity:
        issues.append("production parity report is not the exact content-addressed input")

    retained_bundle = _production_bundle_evidence(
        parity,
        report_path,
        repository_root.expanduser().resolve(),
    )
    issues.extend(retained_bundle["issues"])

    calphad_release_contract = _calphad_release_contract_evidence(parity, repository_root)
    issues.extend(calphad_release_contract["issues"])

    source = _mapping(parity.get("source"))
    required_materials = _mapping(source.get("required_materials"))
    release_artifacts = _mapping(source.get("release_artifacts"))
    verified_release_artifacts = _mapping(parity.get("verified_release_artifacts"))
    release_manifest_valid = all(
        (
            source.get("kind") == "git_archive_release_manifest",
            source.get("manifest_path") == "release-manifest.json",
            _plain_sha256(source.get("manifest_sha256")) is not None,
            _content_manifest_valid(required_materials),
            _release_artifacts_valid(release_artifacts),
            verified_release_artifacts == release_artifacts,
        )
    )
    source_valid = all(
        (
            source.get("kind") == "git_archive_release_manifest",
            _git_sha(parity.get("expected_git_sha")) == expected.git_sha,
            _git_sha(source.get("expected_git_sha")) == expected.git_sha,
            _git_sha(source.get("observed_git_sha")) == expected.git_sha,
            source.get("tracked_worktree_clean") is True,
            source.get("staged_index_clean") is True,
            source.get("untracked_files_clean") is True,
            release_manifest_valid,
        )
    )
    if not source_valid:
        issues.append(
            "production parity source is not the exact content-hashed release artifact revision"
        )

    base_image = _mapping(parity.get("base_image"))
    executed_image = _mapping(parity.get("executed_image"))
    image_valid = all(
        (
            _immutable_sha256(base_image.get("image_id")) == expected.runtime_image,
            _immutable_sha256(executed_image.get("image_id")) == expected.runtime_image,
            _immutable_sha256(executed_image.get("base_image_id")) == expected.runtime_image,
            _git_sha(base_image.get("revision")) == expected.git_sha,
            _git_sha(executed_image.get("revision")) == expected.git_sha,
            base_image.get("title") == PRODUCTION_IMAGE_TITLE,
            executed_image.get("title") == PRODUCTION_IMAGE_TITLE,
            _sequence(base_image.get("entrypoint")) == [],
            _sequence(executed_image.get("entrypoint")) == [],
            executed_image.get("entrypoint_adapter") is False,
            bool(str(base_image.get("ref") or "").strip()),
            base_image.get("ref") == executed_image.get("ref"),
        )
    )
    if not image_valid:
        issues.append("production parity did not execute the expected immutable release image")

    sandbox = _mapping(parity.get("sandbox"))
    sandbox_valid = all(
        (
            sandbox.get("backend") == "DockerSandboxBackend",
            sandbox.get("source") == "exported_worker_environment",
            sandbox.get("policy_source") == "exported_worker_environment",
            sandbox.get("network") == "none",
            sandbox.get("network_none") is True,
            sandbox.get("rootfs_read_only") is True,
            sandbox.get("capabilities_dropped") is True,
            sandbox.get("no_new_privileges") is True,
            _immutable_sha256(sandbox.get("immutable_image_id")) == expected.runtime_image,
            _positive_finite_number(sandbox.get("cpus")),
            bool(str(sandbox.get("memory") or "").strip()),
            _strict_int(sandbox.get("pids_limit")) is not None and int(sandbox["pids_limit"]) > 0,
            bool(str(sandbox.get("shm_size") or "").strip()),
            _strict_int(sandbox.get("timeout_seconds")) is not None
            and int(sandbox["timeout_seconds"]) > 0,
            _strict_int(sandbox.get("output_limit_bytes")) is not None
            and int(sandbox["output_limit_bytes"]) > 0,
            _strict_int(sandbox.get("max_concurrency")) is not None
            and int(sandbox["max_concurrency"]) > 0,
        )
    )
    if not sandbox_valid:
        issues.append("production parity sandbox policy is incomplete, mutable, or unbounded")

    execution = _mapping(parity.get("execution"))
    execution_valid = all(
        (
            _strict_int(execution.get("exit_code")) == 0,
            execution.get("truncated") is False,
            _strict_int(execution.get("output_size_bytes")) is not None
            and int(execution["output_size_bytes"]) > 0,
            _plain_sha256(execution.get("output_sha256")) is not None,
        )
    )
    if not execution_valid:
        issues.append("production parity DockerSandboxBackend execution did not finish cleanly")

    staged = _mapping(parity.get("staged_source"))
    staged_files = [_mapping(item) for item in _sequence(staged.get("files"))]
    staged_valid = all(
        (
            _plain_sha256(staged.get("aggregate_sha256")) is not None,
            _strict_int(staged.get("file_count")) == len(staged_files),
            len(staged_files) > 0,
            all(
                bool(str(item.get("path") or "").strip())
                and _plain_sha256(item.get("sha256")) is not None
                and _strict_int(item.get("size_bytes")) is not None
                and int(item["size_bytes"]) >= 0
                for item in staged_files
            ),
        )
    )
    if not staged_valid:
        issues.append("production parity staged-source manifest is incomplete")

    domain_wrapper = _mapping(parity.get("domain_gate"))
    embedded_domain = _mapping(domain_wrapper.get("report"))
    domain_junit = _mapping(embedded_domain.get("junit"))
    domain_invariants = _deterministic_invariant_evidence(embedded_domain, domain_junit)
    embedded_domain_valid = all(
        (
            embedded_domain.get("schema_version") == 1,
            embedded_domain.get("gate") == "materials-domain-gate",
            embedded_domain.get("scope") == "deterministic-domain-invariants",
            embedded_domain.get("status") == "passed",
            _sequence(embedded_domain.get("failures")) == [],
            _strict_int(domain_junit.get("tests")) == REQUIRED_DOMAIN_INVARIANT_COUNT,
            _strict_int(domain_junit.get("failures")) == 0,
            _strict_int(domain_junit.get("errors")) == 0,
            _strict_int(domain_junit.get("skipped")) == 0,
            _strict_int(_mapping(embedded_domain.get("pytest")).get("exit_code")) == 0,
            _sequence(embedded_domain.get("version_drift")) == [],
            domain_invariants["valid"],
            _immutable_sha256(_mapping(embedded_domain.get("image")).get("id"))
            == expected.runtime_image,
            _mapping(embedded_domain.get("provenance_policy")).get("status") == "enforced",
            _sequence(parity.get("required_domain_validators")) == list(REQUIRED_DOMAIN_VALIDATORS),
            _domain_calphad_preflight_valid(embedded_domain),
            _domain_calphad_experimental_benchmark_valid(embedded_domain),
        )
    )
    if not embedded_domain_valid:
        issues.append("full production image did not reproduce the exact 13 domain invariants")

    runtime_wrapper = _mapping(parity.get("calphad_runtime"))
    runtime_junit = _mapping(runtime_wrapper.get("junit"))
    calphad_runtime_valid = all(
        (
            runtime_wrapper.get("relative_path") == "calphad-runtime-junit.xml",
            _plain_sha256(runtime_wrapper.get("sha256")) is not None,
            _strict_int(runtime_junit.get("tests")) == CALPHAD_RUNTIME_TEST_COUNT,
            _strict_int(runtime_junit.get("failures")) == 0,
            _strict_int(runtime_junit.get("errors")) == 0,
            _strict_int(runtime_junit.get("skipped")) == 0,
            _sequence(runtime_wrapper.get("required_core_test_names"))
            == list(REQUIRED_CALPHAD_RUNTIME_CORE_TEST_NAMES),
            _sequence(runtime_wrapper.get("required_typed_cli_test_names"))
            == list(REQUIRED_CALPHAD_RUNTIME_CLI_TEST_NAMES),
        )
    )
    if not calphad_runtime_valid:
        issues.append("full production image CALPHAD runtime/CLI suite is incomplete or skipped")

    tools_wrapper = _mapping(parity.get("calphad_tool_orchestration"))
    tools_junit = _mapping(tools_wrapper.get("junit"))
    tools_execution = _mapping(tools_wrapper.get("execution"))
    tools_binding = _mapping(tools_wrapper.get("binding"))
    calphad_tools_valid = all(
        (
            tools_wrapper.get("scope") == "host-worker-runtime-orchestration-contract",
            tools_wrapper.get("relative_path") == "calphad-tools-junit.xml",
            _plain_sha256(tools_wrapper.get("sha256")) is not None,
            _strict_int(tools_junit.get("tests")) == CALPHAD_TOOLS_TEST_COUNT,
            _strict_int(tools_junit.get("failures")) == 0,
            _strict_int(tools_junit.get("errors")) == 0,
            _strict_int(tools_junit.get("skipped")) == 0,
            tools_execution.get("runner") == "uv-frozen-project-with-pytest-8.4.2",
            _strict_int(tools_execution.get("exit_code")) == 0,
            _strict_int(tools_execution.get("stdout_size_bytes")) is not None,
            int(tools_execution.get("stdout_size_bytes", -1)) >= 0,
            _plain_sha256(tools_execution.get("stdout_sha256")) is not None,
            _strict_int(tools_execution.get("stderr_size_bytes")) is not None,
            int(tools_execution.get("stderr_size_bytes", -1)) >= 0,
            _plain_sha256(tools_execution.get("stderr_sha256")) is not None,
            _git_sha(tools_binding.get("git_sha")) == expected.git_sha,
            _immutable_sha256(tools_binding.get("runtime_image_id")) == expected.runtime_image,
            tools_binding.get("source_kind") == "git_archive_release_manifest",
            _mapping(tools_binding.get("release_artifacts")) == release_artifacts,
            _sequence(tools_wrapper.get("required_test_names"))
            == list(REQUIRED_CALPHAD_TOOL_TEST_NAMES),
        )
    )
    if not calphad_tools_valid:
        issues.append("host/worker CALPHAD tool orchestration is incomplete or not release-bound")

    calphad_wrapper = _mapping(parity.get("calphad"))
    calphad = _mapping(calphad_wrapper.get("report"))
    calphad_source_evidence = _calphad_probe_source_evidence(calphad, repository_root)
    issues.extend(calphad_source_evidence["issues"])
    calphad_valid = all(
        (
            calphad_wrapper.get("relative_path") == "calphad-embedded-probe.json",
            _plain_sha256(calphad_wrapper.get("sha256")) is not None,
            calphad_source_evidence["valid"],
        )
    )
    if not calphad_valid:
        issues.append("full production image embedded CALPHAD manifest/parse probe did not pass")

    companion_valid = report_path is not None
    runtime_xml_evidence: dict[str, Any] = {
        "valid": False,
        "summary": {"tests": 0, "failures": 0, "errors": 0, "skipped": 0},
        "issues": ["runtime JUnit was not revalidated"],
    }
    tools_xml_evidence: dict[str, Any] = {
        "valid": False,
        "summary": {"tests": 0, "failures": 0, "errors": 0, "skipped": 0},
        "issues": ["tools JUnit was not revalidated"],
    }
    if report_path is not None:
        for label, wrapper, expected_relative, expected_payload in (
            (
                "production parity domain report",
                domain_wrapper,
                "domain/materials-domain-gate.json",
                embedded_domain,
            ),
            (
                "production parity CALPHAD probe",
                calphad_wrapper,
                "calphad-embedded-probe.json",
                calphad,
            ),
        ):
            relative_ok = wrapper.get("relative_path") == expected_relative
            companion_path = (
                _safe_relative_path(report_path.parent, expected_relative) if relative_ok else None
            )
            hash_ok = _rehash(
                label=label,
                path=companion_path,
                expected_sha256=wrapper.get("sha256"),
                issues=issues,
            )
            payload_ok = False
            if hash_ok and companion_path is not None:
                try:
                    payload = json.loads(
                        companion_path.read_text(encoding="utf-8"),
                        parse_constant=_reject_json_constant,
                    )
                    payload_ok = isinstance(payload, dict) and payload == expected_payload
                except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    payload_ok = False
            if not payload_ok:
                issues.append(f"{label}: retained JSON differs from the parity attestation")
            companion_valid = companion_valid and relative_ok and hash_ok and payload_ok

        runtime_path = _safe_relative_path(report_path.parent, "calphad-runtime-junit.xml")
        runtime_hash_ok = _rehash(
            label="production parity CALPHAD runtime JUnit",
            path=runtime_path,
            expected_sha256=runtime_wrapper.get("sha256"),
            issues=issues,
        )
        runtime_xml_evidence = _calphad_runtime_junit_evidence(
            runtime_path if runtime_hash_ok else None
        )
        runtime_summary_matches = runtime_xml_evidence["summary"] == runtime_junit
        if not runtime_xml_evidence["valid"]:
            issues.extend(
                f"production parity CALPHAD runtime JUnit: {issue}"
                for issue in runtime_xml_evidence["issues"]
            )
        if not runtime_summary_matches:
            issues.append(
                "production parity CALPHAD runtime JUnit summary differs from retained XML"
            )
        companion_valid = (
            companion_valid
            and runtime_hash_ok
            and runtime_xml_evidence["valid"]
            and runtime_summary_matches
        )

        tools_path = _safe_relative_path(report_path.parent, "calphad-tools-junit.xml")
        tools_hash_ok = _rehash(
            label="production parity CALPHAD tools JUnit",
            path=tools_path,
            expected_sha256=tools_wrapper.get("sha256"),
            issues=issues,
        )
        tools_xml_evidence = _calphad_tools_junit_evidence(tools_path if tools_hash_ok else None)
        tools_summary_matches = tools_xml_evidence["summary"] == tools_junit
        if not tools_xml_evidence["valid"]:
            issues.extend(
                f"production parity CALPHAD tools JUnit: {issue}"
                for issue in tools_xml_evidence["issues"]
            )
        if not tools_summary_matches:
            issues.append("production parity CALPHAD tools JUnit summary differs from retained XML")
        companion_valid = (
            companion_valid
            and tools_hash_ok
            and tools_xml_evidence["valid"]
            and tools_summary_matches
        )
    if not companion_valid:
        issues.append("production parity retained companion evidence is incomplete or changed")

    report_shape_valid = all(
        (
            parity.get("schema_version") == 1,
            parity.get("gate") == PRODUCTION_PARITY_GATE,
            parity.get("scope") == PRODUCTION_PARITY_SCOPE,
            parity.get("claim") == PRODUCTION_PARITY_CLAIM,
            parity.get("status") == "passed",
            parity.get("full_production_image_parity") is True,
            _sequence(parity.get("failures")) == [],
        )
    )
    if not report_shape_valid:
        issues.append("production parity report is not a passing production-full attestation")

    checks = {
        "content_addressed_input": input_integrity,
        "retained_evidence_bundle": retained_bundle["valid"],
        "calphad_release_contract": calphad_release_contract["valid"],
        "report_shape": report_shape_valid,
        "same_clean_source": source_valid,
        "immutable_release_image": image_valid,
        "production_sandbox_policy": sandbox_valid,
        "execution": execution_valid,
        "staged_source_manifest": staged_valid,
        "domain_invariants": embedded_domain_valid,
        "calphad_runtime": calphad_runtime_valid,
        "calphad_tool_orchestration": calphad_tools_valid,
        "embedded_calphad": calphad_valid,
        "retained_companion_evidence": companion_valid,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "issues": issues,
        "scope": parity.get("scope"),
        "runtime_image": _immutable_sha256(executed_image.get("image_id")),
        "git_sha": _git_sha(executed_image.get("revision")),
        "domain_invariants": domain_invariants,
        "calphad_runtime_junit": runtime_xml_evidence,
        "calphad_tools_junit": tools_xml_evidence,
        "calphad_release_contract": calphad_release_contract,
        "calphad_probe_source": calphad_source_evidence,
        "verified_release_artifacts": release_artifacts,
        "retained_evidence_bundle": retained_bundle,
    }


def _benchmark_evidence(
    benchmark: Mapping[str, Any],
    benchmark_root: Path,
    policy: ReadinessPolicy,
    checkout_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = benchmark_root.expanduser().resolve()
    issues: list[str] = []
    raw_hashes = _mapping(benchmark.get("tracked_file_hashes"))
    declared: dict[str, str] = {}
    for relative, raw_digest in raw_hashes.items():
        digest = _plain_sha256(raw_digest)
        if not isinstance(relative, str) or digest is None:
            issues.append("benchmark tracked-file manifest contains an invalid entry")
            continue
        declared[relative] = digest

    observed_checkout = (
        _mapping(checkout_state) if checkout_state is not None else inspect_benchmark_checkout(root)
    )
    checkout_clean = all(
        (
            observed_checkout.get("inspection_ok", True) is True,
            observed_checkout.get("dirty") is False,
            observed_checkout.get("revision") == policy.official_revision,
            _sequence(observed_checkout.get("tracked_files")) == sorted(declared),
        )
    )
    if not checkout_clean:
        issues.append("benchmark checkout is not the exact clean pinned revision/tracked-file set")

    verified = 0
    for relative, expected_digest in declared.items():
        path = _safe_relative_path(root, relative)
        if _rehash(
            label=f"MatTools tracked file {relative}",
            path=path,
            expected_sha256=expected_digest,
            issues=issues,
        ):
            verified += 1

    declared_count = _strict_int(benchmark.get("tracked_file_count"))
    if declared_count != len(declared) or not declared:
        issues.append("benchmark tracked-file count is missing or inconsistent")
    recomputed_manifest = manifest_hash(declared) if declared else None
    if recomputed_manifest != policy.official_manifest_sha256:
        issues.append("benchmark tracked-file manifest is not the pinned official manifest")
    if _plain_sha256(benchmark.get("sha256")) != recomputed_manifest:
        issues.append("benchmark report manifest SHA-256 is internally inconsistent")
    control_hashes = _mapping(benchmark.get("control_file_hashes"))
    if not control_hashes or any(
        _plain_sha256(digest) != declared.get(str(path)) for path, digest in control_hashes.items()
    ):
        issues.append("benchmark control-file hashes are absent or inconsistent")

    return {
        "valid": not issues and verified == len(declared) and bool(declared),
        "declared_file_count": len(declared),
        "verified_file_count": verified,
        "recomputed_manifest_sha256": recomputed_manifest,
        "checkout": observed_checkout,
        "checkout_clean": checkout_clean,
        "issues": issues,
    }


def _benchmark_shape(benchmark: Mapping[str, Any], policy: ReadinessPolicy) -> dict[str, Any]:
    tasks = [_mapping(item) for item in _sequence(benchmark.get("tasks"))]
    tracked_hashes = {
        str(path): _plain_sha256(digest)
        for path, digest in _mapping(benchmark.get("tracked_file_hashes")).items()
    }
    task_ids = [str(task.get("task_id") or "") for task in tasks]
    ordinals = [_strict_int(task.get("ordinal")) for task in tasks]
    subtask_counts = [_strict_int(task.get("subtask_count")) for task in tasks]
    isolation = all(
        _mapping(task.get("expected_values")).get("isolated_from_ultra") is True
        and _mapping(task.get("verifier")).get("isolated_from_ultra") is True
        for task in tasks
    )
    task_hash_linkage = True
    for task in tasks:
        task_id = str(task.get("task_id") or "")
        base = f"src/question_segments/pymatgen_analysis_defects/{task_id}"
        for record_name, filename in (
            ("question", "question.txt"),
            ("expected_values", "properties.json"),
            ("verifier", "new_unit_test.py"),
        ):
            record = _mapping(task.get(record_name))
            task_hash_linkage = task_hash_linkage and (
                record.get("path") == filename
                and _plain_sha256(record.get("sha256")) == tracked_hashes.get(f"{base}/{filename}")
            )
    valid = all(
        (
            benchmark.get("name") == "MatTools-real-world",
            benchmark.get("revision") == policy.official_revision,
            _plain_sha256(benchmark.get("sha256")) == policy.official_manifest_sha256,
            _plain_sha256(benchmark.get("official_manifest_sha256"))
            == policy.official_manifest_sha256,
            benchmark.get("strict_official") is True,
            benchmark.get("full_git_tree_hashed") is True,
            benchmark.get("git_checkout_clean") is True,
            _strict_int(benchmark.get("parent_count")) == PARENTS_PER_TRIAL,
            _strict_int(benchmark.get("scientific_subtask_count")) == SUBTASKS_PER_TRIAL,
            len(tasks) == PARENTS_PER_TRIAL,
            len(set(task_ids)) == PARENTS_PER_TRIAL,
            ordinals == list(range(1, PARENTS_PER_TRIAL + 1)),
            all(value is not None and value > 0 for value in subtask_counts),
            sum(value or 0 for value in subtask_counts) == SUBTASKS_PER_TRIAL,
            isolation,
            task_hash_linkage,
        )
    )
    return {
        "valid": valid,
        "task_ids": task_ids,
        "subtask_counts": {
            task_id: int(count or 0) for task_id, count in zip(task_ids, subtask_counts)
        },
        "answers_and_verifiers_isolated": isolation,
        "task_hash_linkage_valid": task_hash_linkage,
    }


def _trial_and_attempt_evidence(
    mattools: Mapping[str, Any],
    benchmark_shape: Mapping[str, Any],
    expected: ExpectedProvenance,
) -> dict[str, Any]:
    trials = [_mapping(item) for item in _sequence(mattools.get("trials"))]
    expected_task_ids = list(benchmark_shape.get("task_ids", []))
    subtask_counts = _mapping(benchmark_shape.get("subtask_counts"))
    runtime = _mapping(mattools.get("runtime_environment"))
    declared_model = str(runtime.get("operator_declared_model_id") or "").strip()
    declared_provider = str(runtime.get("operator_declared_provider_id") or "").strip()

    complete = len(trials) == TRIAL_COUNT
    execute_ok = complete
    execute_image_ok = complete
    runtime_provenance_ok = complete and bool(declared_model) and bool(declared_provider)
    remote_mutation_free = complete
    evaluator_ok = complete
    evaluator_independent = complete
    replay_ok = complete
    server_cleanroom_ok = complete
    worker_cleanroom_ok = complete
    per_trial_runnable_floor = complete
    per_trial_strict_scientific_floor = complete
    unique_runs: set[str] = set()
    unique_threads: set[str] = set()
    total_attempts = 0
    runnable = 0
    published_runner_runnable = 0
    official_scientific = 0
    strict_scientific = 0
    per_trial: list[dict[str, Any]] = []

    for expected_trial, trial in enumerate(trials, start=1):
        attempts = [_mapping(item) for item in _sequence(trial.get("attempts"))]
        trial_replay_count = _strict_int(trial.get("replay_count"))
        trial_ids = [str(item.get("task_id") or "") for item in attempts]
        trial_ordinals = [_strict_int(item.get("ordinal")) for item in attempts]
        trial_complete = (
            _strict_int(trial.get("trial")) == expected_trial
            and trial.get("status") == "complete"
            and len(attempts) == PARENTS_PER_TRIAL
            and trial_ids == expected_task_ids
            and trial_ordinals == list(range(1, PARENTS_PER_TRIAL + 1))
            and _strict_int(trial.get("runnable_denominator")) == PARENTS_PER_TRIAL
            and _strict_int(trial.get("scientific_denominator")) == SUBTASKS_PER_TRIAL
            and trial_replay_count is not None
            and 2 <= trial_replay_count <= 4
        )
        complete = complete and trial_complete
        trial_runnable = 0
        trial_published_runnable = 0
        trial_scientific = 0
        trial_strict_scientific = 0

        for attempt in attempts:
            total_attempts += 1
            run_id = str(attempt.get("run_id") or "").strip()
            thread_id = str(attempt.get("thread_id") or "").strip()
            if not run_id or run_id in unique_runs or not thread_id or thread_id in unique_threads:
                complete = False
            unique_runs.add(run_id)
            unique_threads.add(thread_id)
            if attempt.get("submission_status") not in {
                "captured",
                "terminal_failure",
                "missing_code",
            }:
                complete = False
            if attempt.get("run_status") not in {"succeeded", "failed", "canceled"}:
                complete = False
            if _plain_sha256(attempt.get("code_sha256")) is None:
                complete = False
            artifact_ids = [str(value) for value in _sequence(attempt.get("artifact_ids")) if value]
            solution_artifact_id = str(attempt.get("solution_artifact_id") or "")
            if attempt.get("run_status") == "succeeded" and (
                not solution_artifact_id or solution_artifact_id not in artifact_ids
            ):
                complete = False

            evaluation = _mapping(attempt.get("evaluation"))
            task_id = str(attempt.get("task_id") or "")
            task_total = _strict_int(subtask_counts.get(task_id))
            scoring = _mapping(attempt.get("scoring_evidence"))
            scoring_replays = [_mapping(item) for item in _sequence(scoring.get("replays"))]
            primary = _mapping(scoring.get("primary"))
            published = _mapping(primary.get("published_upstream"))
            strict = _mapping(primary.get("strict_shadow"))
            published_pass = _strict_int(published.get("scientific_pass"))
            published_fail = _strict_int(published.get("scientific_fail"))
            strict_pass = _strict_int(strict.get("strict_scientific_pass"))
            strict_fail = _strict_int(strict.get("strict_scientific_fail"))
            raw_verifier_digest = strict.get("raw_verifier_output_sha256")
            raw_verifier_digest_valid = _plain_sha256(raw_verifier_digest) is not None or (
                raw_verifier_digest is None
                and strict.get("semantic_runnable") is False
                and strict_pass == 0
            )
            terminal_hashes = [
                _plain_sha256(record.get("replay_terminal_record_sha256"))
                for record in scoring_replays
            ]
            replay_fingerprints = [
                canonical_json_sha256(
                    {
                        "published_upstream": _mapping(record.get("published_upstream")),
                        "strict_shadow": _mapping(record.get("strict_shadow")),
                    }
                )
                for record in scoring_replays
            ]
            scoring_valid = all(
                (
                    task_total is not None,
                    scoring.get("schema_version") == "1",
                    scoring.get("task_id") == task_id,
                    _strict_int(scoring.get("ordinal")) == _strict_int(attempt.get("ordinal")),
                    _strict_int(scoring.get("subtask_count")) == task_total,
                    _strict_int(scoring.get("expected_replay_count")) == trial_replay_count,
                    _strict_int(scoring.get("replay_count")) == trial_replay_count,
                    scoring.get("complete") is True,
                    scoring.get("replay_consistent") is True,
                    len(scoring_replays) == trial_replay_count,
                    primary == (scoring_replays[0] if scoring_replays else {}),
                    [record.get("replay") for record in scoring_replays]
                    == list(range(1, int(trial_replay_count or 0) + 1)),
                    all(value is not None for value in terminal_hashes),
                    len(set(terminal_hashes)) == len(terminal_hashes),
                    len(set(replay_fingerprints)) == 1,
                    set(published)
                    == {"classification", "runnable", "scientific_pass", "scientific_fail"},
                    isinstance(published.get("runnable"), bool),
                    bool(str(published.get("classification") or "").strip()),
                    published_pass is not None,
                    published_fail is not None,
                    published_pass is not None
                    and published_fail is not None
                    and task_total is not None
                    and 0 <= published_pass <= task_total
                    and published_pass + published_fail == task_total,
                    set(strict)
                    == {
                        "semantic_runnable",
                        "strict_scientific_classification",
                        "strict_scientific_pass",
                        "strict_scientific_fail",
                        "strict_exact_ok",
                        "raw_verifier_output_sha256",
                    },
                    isinstance(strict.get("semantic_runnable"), bool),
                    isinstance(strict.get("strict_exact_ok"), bool),
                    bool(str(strict.get("strict_scientific_classification") or "").strip()),
                    raw_verifier_digest_valid,
                    strict_pass is not None,
                    strict_fail is not None,
                    strict_pass is not None
                    and strict_fail is not None
                    and task_total is not None
                    and 0 <= strict_pass <= task_total
                    and strict_pass + strict_fail == task_total,
                    evaluation.get("task_id") == task_id,
                    _strict_int(evaluation.get("ordinal")) == _strict_int(attempt.get("ordinal")),
                    evaluation.get("classification") == published.get("classification"),
                    evaluation.get("runnable") is published.get("runnable"),
                    _strict_int(evaluation.get("scientific_pass")) == published_pass,
                    _strict_int(evaluation.get("scientific_fail")) == published_fail,
                )
            )
            if not scoring_valid:
                complete = False
                published_pass = 0
                strict_pass = 0
            if strict.get("semantic_runnable") is True and scoring_valid:
                trial_runnable += 1
            if published.get("runnable") is True and scoring_valid:
                trial_published_runnable += 1
            trial_scientific += int(published_pass or 0)
            trial_strict_scientific += int(strict_pass or 0)

            trace = _mapping(attempt.get("trace_summary"))
            server_cleanroom_ok = server_cleanroom_ok and all(
                (
                    trace.get("server_cleanroom_profile_attested") is True,
                    _sequence(trace.get("server_evaluation_profiles"))
                    == [MATERIALS_CLEANROOM_PROFILE],
                )
            )
            cleanroom_binding = _mapping(attempt.get("cleanroom_binding"))
            worker_cleanroom_ok = worker_cleanroom_ok and all(
                (
                    _worker_cleanroom_attestation_valid(attempt),
                    cleanroom_binding.get("valid") is True,
                    cleanroom_binding.get("user_identity_independently_bound") is True,
                )
            )
            remote_mutation_free = remote_mutation_free and not any(
                _is_remote_mutation_tool(name) for name in _sequence(trace.get("tool_names"))
            )
            started = _strict_int(trace.get("production_execute_started_count"))
            terminal = _strict_int(trace.get("production_execute_terminal_count"))
            completed = _strict_int(trace.get("production_execute_completed_count"))
            execute_ok = execute_ok and (
                trace.get("production_execute_tool_evidence") is True
                and started is not None
                and terminal is not None
                and completed is not None
                and started > 0
                and terminal >= started
                and 0 < completed <= started
            )
            observed_images = {
                _immutable_sha256(value)
                for value in _sequence(trace.get("observed_execute_image_digests"))
            }
            execute_image_ok = execute_image_ok and expected.runtime_image in observed_images

            actual = _mapping(attempt.get("actual_runtime_provenance"))
            models = {str(value) for value in _sequence(actual.get("observed_model_ids")) if value}
            providers = {
                str(value) for value in _sequence(actual.get("observed_provider_ids")) if value
            }
            runtime_provenance_ok = runtime_provenance_ok and all(
                (
                    actual.get("validated") is True,
                    actual.get("model_observable") is True,
                    actual.get("provider_observable") is True,
                    actual.get("model_matches_declaration") is True,
                    actual.get("provider_matches_declaration") is True,
                    declared_model in models,
                    declared_provider in providers,
                )
            )

        reported_runnable = _strict_int(trial.get("runnable"))
        reported_published_runnable = _strict_int(trial.get("published_runner_runnable"))
        reported_scientific = _strict_int(trial.get("scientific_pass"))
        reported_strict = _strict_int(trial.get("strict_scientific_pass"))
        complete = complete and (
            reported_runnable == trial_runnable
            and reported_published_runnable == trial_published_runnable
            and reported_scientific == trial_scientific
            and reported_strict == trial_strict_scientific
        )
        per_trial_runnable_floor = (
            per_trial_runnable_floor and trial_runnable >= PER_TRIAL_RUNNABLE_MINIMUM
        )
        per_trial_strict_scientific_floor = (
            per_trial_strict_scientific_floor
            and trial_strict_scientific >= PER_TRIAL_SCIENTIFIC_MINIMUM
        )
        runnable += trial_runnable
        published_runner_runnable += trial_published_runnable
        official_scientific += trial_scientific
        strict_scientific += trial_strict_scientific

        replay_ok = replay_ok and (
            trial.get("reproducible") is True and (_strict_int(trial.get("replay_count")) or 0) >= 2
        )
        environment = _mapping(trial.get("evaluator_environment"))
        packages = _mapping(environment.get("packages"))
        evaluator_ok = evaluator_ok and all(
            (
                environment.get("comparable") is True,
                environment.get("full_environment_lock_matches") is True,
                _immutable_sha256(environment.get("image_id")) == expected.evaluator_image,
                _immutable_sha256(environment.get("production_runtime_image_digest"))
                == expected.runtime_image,
                packages == EXPECTED_EVALUATOR_PACKAGES,
                _plain_sha256(environment.get("resolved_environment_sha256")) is not None,
            )
        )
        evaluator_independent = evaluator_independent and (
            environment.get("independent_from_production_runtime") is True
            and expected.evaluator_image != expected.runtime_image
        )
        per_trial.append(
            {
                "trial": expected_trial,
                "attempts": len(attempts),
                "runnable": trial_runnable,
                "published_runner_runnable": trial_published_runnable,
                "scientific_pass": trial_scientific,
                "strict_scientific_pass": trial_strict_scientific,
            }
        )

    return {
        "complete": complete and total_attempts == RUNNABLE_DENOMINATOR,
        "attempt_count": total_attempts,
        "runnable": runnable,
        "published_runner_runnable": published_runner_runnable,
        "scientific_pass": official_scientific,
        "strict_scientific_pass": strict_scientific,
        "execute_evidence": execute_ok and total_attempts == RUNNABLE_DENOMINATOR,
        "execute_image_attestation": execute_image_ok and total_attempts == RUNNABLE_DENOMINATOR,
        "runtime_provenance": runtime_provenance_ok and total_attempts == RUNNABLE_DENOMINATOR,
        "server_authorized_cleanroom_profile": server_cleanroom_ok
        and total_attempts == RUNNABLE_DENOMINATOR,
        "worker_enforced_cleanroom_profile": worker_cleanroom_ok
        and total_attempts == RUNNABLE_DENOMINATOR,
        "per_trial_runnable_floor": per_trial_runnable_floor and len(per_trial) == TRIAL_COUNT,
        "per_trial_strict_scientific_floor": per_trial_strict_scientific_floor
        and len(per_trial) == TRIAL_COUNT,
        "remote_mutation_free": remote_mutation_free and total_attempts == RUNNABLE_DENOMINATOR,
        "evaluator_exact": evaluator_ok,
        "evaluator_independent": evaluator_independent,
        "reproducible": replay_ok,
        "per_trial": per_trial,
    }


def _published_counts_consistent(
    mattools: Mapping[str, Any], recomputed: Mapping[str, Any]
) -> bool:
    counts = _mapping(mattools.get("counts"))
    rates = _mapping(mattools.get("rates"))
    runnable = _strict_int(counts.get("runnable"))
    scientific = _strict_int(counts.get("scientific_pass"))
    strict_scientific = _strict_int(counts.get("strict_scientific_pass"))
    if not all(
        (
            runnable == recomputed.get("runnable"),
            scientific == recomputed.get("scientific_pass"),
            strict_scientific == recomputed.get("strict_scientific_pass"),
            _strict_int(counts.get("runnable_denominator")) == RUNNABLE_DENOMINATOR,
            _strict_int(counts.get("scientific_denominator")) == SCIENTIFIC_DENOMINATOR,
            _strict_int(counts.get("runnable_minimum")) == RUNNABLE_MINIMUM,
            _strict_int(counts.get("scientific_minimum")) == SCIENTIFIC_MINIMUM,
            _strict_int(counts.get("per_trial_runnable_minimum")) == PER_TRIAL_RUNNABLE_MINIMUM,
            _strict_int(counts.get("per_trial_scientific_minimum")) == PER_TRIAL_SCIENTIFIC_MINIMUM,
            _strict_int(counts.get("terminal_attempts")) == RUNNABLE_DENOMINATOR,
            _strict_int(counts.get("expected_attempts_for_configured_run")) == RUNNABLE_DENOMINATOR,
        )
    ):
        return False
    expected_rates = {
        "function_runnable": runnable / RUNNABLE_DENOMINATOR,
        "task_success": scientific / SCIENTIFIC_DENOMINATOR,
        "strict_task_success": strict_scientific / SCIENTIFIC_DENOMINATOR,
    }
    return all(
        isinstance(rates.get(name), int | float)
        and not isinstance(rates.get(name), bool)
        and math.isclose(float(rates[name]), expected_value, rel_tol=0, abs_tol=1e-12)
        for name, expected_value in expected_rates.items()
    )


def _verify_detached_signature(attestation: Mapping[str, Any]) -> tuple[bool, str | None]:
    openssl = shutil.which("openssl")
    if not openssl:
        return False, "openssl is unavailable for independent signature verification"
    command = (
        openssl,
        "dgst",
        "-sha256",
        "-verify",
        str(attestation.get("operator_public_key_path") or ""),
        "-signature",
        str(attestation.get("detached_signature_path") or ""),
        str(attestation.get("path") or ""),
    )
    process = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if process.returncode != 0:
        return False, "detached sandbox-attestation signature verification failed"
    return True, None


def _public_key_git_anchored(public_key: Path, repository_root: Path) -> bool:
    root = repository_root.expanduser().resolve()
    resolved_key = public_key.expanduser().resolve()
    try:
        relative = resolved_key.relative_to(root)
    except ValueError:
        return False
    tracked = subprocess.run(
        ("git", "ls-files", "--error-unmatch", "--", str(relative)),
        cwd=root,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if tracked.returncode != 0:
        return False
    committed = subprocess.run(
        ("git", "show", f"HEAD:{relative.as_posix()}"),
        cwd=root,
        capture_output=True,
        check=False,
        timeout=30,
    )
    return (
        committed.returncode == 0
        and resolved_key.is_file()
        and hashlib.sha256(committed.stdout).hexdigest() == sha256_file(resolved_key)
    )


def _external_isolation_evidence(path: Path | None, expected_image: str) -> dict[str, Any]:
    """Parse the operator probe rather than trusting its report-side summary."""

    summary: dict[str, Any] = {}
    issues: list[str] = []
    if path is None or path.is_symlink() or not path.is_file():
        return {"valid": False, "summary": summary, "issues": ["evidence is not a regular file"]}
    try:
        size = path.stat().st_size
        if size <= 0 or size > 1024 * 1024:
            raise ValueError("evidence size is outside 1..1048576 bytes")
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return {"valid": False, "summary": summary, "issues": [str(exc)]}
    if not isinstance(payload, dict):
        return {"valid": False, "summary": summary, "issues": ["evidence is not an object"]}

    network_probe = _mapping(payload.get("network_egress_probe"))
    host_probe = _mapping(payload.get("host_access_probe"))
    limits = _mapping(payload.get("resource_limits"))
    memory = _strict_int(limits.get("memory_bytes"))
    pids = _strict_int(limits.get("pids_limit"))
    nano_cpus = _strict_int(limits.get("nano_cpus"))
    cpu_quota = _strict_int(limits.get("cpu_quota"))
    network_blocked = all(
        (network_probe.get("attempted") is True, network_probe.get("result") == "blocked")
    )
    host_blocked = all(
        (
            _strict_int(host_probe.get("host_mount_count")) == 0,
            host_probe.get("docker_socket_mounted") is False,
        )
    )
    limits_present = all(
        (
            memory is not None and memory > 0,
            pids is not None and pids > 0,
            (nano_cpus is not None and nano_cpus > 0) or (cpu_quota is not None and cpu_quota > 0),
        )
    )
    evidence_image = _immutable_sha256(payload.get("evaluator_image_id"))
    summary = {
        "schema_version": payload.get("schema_version"),
        "evaluator_image_id": evidence_image,
        "observed_at": payload.get("observed_at"),
        "observed_container_id": payload.get("observed_container_id"),
        "network_egress_blocked": network_blocked,
        "host_access_blocked": host_blocked,
        "resource_limits_present": limits_present,
    }
    valid = all(
        (
            payload.get("schema_version") == "1",
            evidence_image == expected_image,
            bool(str(payload.get("observed_at") or "").strip()),
            bool(str(payload.get("observed_container_id") or "").strip()),
            network_blocked,
            host_blocked,
            limits_present,
        )
    )
    if not valid:
        issues.append("isolation probe does not prove blocked egress/host access and fixed limits")
    return {"valid": valid, "summary": summary, "issues": issues}


def _isolation_attestations(
    mattools: Mapping[str, Any], expected: ExpectedProvenance, repository_root: Path
) -> dict[str, Any]:
    issues: list[str] = []
    trials = [_mapping(item) for item in _sequence(mattools.get("trials"))]
    valid = len(trials) == TRIAL_COUNT
    signature_cache: dict[tuple[str, str, str], tuple[bool, str | None]] = {}
    verified_files = 0
    expected_files = TRIAL_COUNT * 4

    for index, trial in enumerate(trials, start=1):
        attestation = _mapping(trial.get("sandbox_policy_attestation"))
        prefix = f"trial {index} sandbox attestation"
        semantic = all(
            (
                attestation.get("valid") is True,
                _sequence(attestation.get("issues")) == [],
                attestation.get("attestation_kind") == "external_sandbox_isolation",
                _immutable_sha256(attestation.get("evaluator_image_id"))
                == expected.evaluator_image,
                attestation.get("network_egress_denied") is True,
                attestation.get("host_access_denied") is True,
                attestation.get("resource_limits_enforced") is True,
                attestation.get("external_enforcement") is True,
                bool(str(attestation.get("enforcement_mechanism") or "").strip()),
                bool(str(attestation.get("signed_by") or "").strip()),
                bool(str(attestation.get("signed_at") or "").strip()),
                attestation.get("operator_signature_verified") is True,
                attestation.get("operator_public_key_trusted_from_git_head") is True,
            )
        )
        if not semantic:
            issues.append(f"{prefix}: semantic/signature declarations are incomplete")
        valid = valid and semantic

        file_fields = (
            ("JSON", "path", "sha256"),
            ("detached signature", "detached_signature_path", "detached_signature_sha256"),
            ("operator public key", "operator_public_key_path", "operator_public_key_sha256"),
            ("isolation evidence", "isolation_evidence_path", "isolation_evidence_sha256"),
        )
        file_paths: dict[str, Path | None] = {}
        for label, path_key, hash_key in file_fields:
            path_text = str(attestation.get(path_key) or "").strip()
            path = Path(path_text).expanduser().resolve() if path_text else None
            file_paths[path_key] = path
            regular = path is not None and path.is_file() and not path.is_symlink()
            passed = regular and _rehash(
                label=f"{prefix} {label}",
                path=path,
                expected_sha256=attestation.get(hash_key),
                issues=issues,
            )
            if not regular:
                issues.append(f"{prefix} {label}: evidence is not a regular file")
            verified_files += int(passed)
            valid = valid and passed

        declared_evidence = _immutable_sha256(attestation.get("declared_isolation_evidence_sha256"))
        observed_evidence = _immutable_sha256(attestation.get("isolation_evidence_sha256"))
        if declared_evidence != observed_evidence or declared_evidence is None:
            issues.append(f"{prefix}: declared and observed isolation evidence hashes differ")
            valid = False

        signed_payload: dict[str, Any] = {}
        signed_path = file_paths.get("path")
        if signed_path is not None:
            try:
                raw = json.loads(
                    signed_path.read_text(encoding="utf-8"),
                    parse_constant=_reject_json_constant,
                    object_pairs_hook=_reject_duplicate_json_keys,
                )
                signed_payload = _mapping(raw)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                issues.append(f"{prefix}: signed JSON cannot be parsed ({exc})")

        evidence_reference = str(signed_payload.get("isolation_evidence_path") or "").strip()
        reference_path = Path(evidence_reference)
        referenced_evidence: Path | None = None
        if signed_path is not None and evidence_reference and not reference_path.is_absolute():
            candidate = (signed_path.parent / reference_path).resolve()
            try:
                candidate.relative_to(signed_path.parent.resolve())
                referenced_evidence = candidate
            except ValueError:
                referenced_evidence = None
        evidence_path = file_paths.get("isolation_evidence_path")
        parsed_evidence = _external_isolation_evidence(evidence_path, expected.evaluator_image)
        signed_semantics = all(
            (
                signed_payload.get("attestation_kind") == "external_sandbox_isolation",
                _immutable_sha256(signed_payload.get("evaluator_image_id"))
                == expected.evaluator_image,
                signed_payload.get("network_egress_denied") is True,
                signed_payload.get("host_access_denied") is True,
                signed_payload.get("resource_limits_enforced") is True,
                signed_payload.get("external_enforcement") is True,
                str(signed_payload.get("enforcement_mechanism") or "").strip()
                == str(attestation.get("enforcement_mechanism") or "").strip(),
                str(signed_payload.get("signed_by") or "").strip()
                == str(attestation.get("signed_by") or "").strip(),
                str(signed_payload.get("signed_at") or "").strip()
                == str(attestation.get("signed_at") or "").strip(),
                referenced_evidence is not None and referenced_evidence == evidence_path,
                _immutable_sha256(signed_payload.get("isolation_evidence_sha256"))
                == declared_evidence,
                parsed_evidence["valid"],
                attestation.get("external_isolation_evidence_semantics_valid") is True,
                _mapping(attestation.get("external_isolation_evidence_summary"))
                == parsed_evidence["summary"],
                attestation.get("harness_enforces_isolation") is False,
                attestation.get("upstream_runner_declares_network_isolation") is False,
                attestation.get("upstream_runner_declares_resource_limits") is False,
                attestation.get("signature_error") is None,
                attestation.get("public_key_trust_anchor") == "current Ultra Git HEAD",
            )
        )
        if not signed_semantics:
            issues.append(f"{prefix}: signed JSON/probe semantics differ from the report")
            issues.extend(f"{prefix}: {issue}" for issue in parsed_evidence["issues"])
            valid = False

        public_key_text = str(attestation.get("operator_public_key_path") or "").strip()
        if not public_key_text or not _public_key_git_anchored(
            Path(public_key_text), repository_root
        ):
            issues.append(f"{prefix}: operator public key is not anchored unchanged in Git HEAD")
            valid = False

        signature_key = tuple(
            str(attestation.get(key) or "")
            for key in ("path", "detached_signature_path", "operator_public_key_path")
        )
        if signature_key not in signature_cache:
            signature_cache[signature_key] = _verify_detached_signature(attestation)
        signature_result = signature_cache[signature_key]
        if not signature_result[0]:
            issues.append(f"{prefix}: {signature_result[1]}")
            valid = False

    return {
        "valid": valid and verified_files == expected_files,
        "trial_count": len(trials),
        "verified_file_references": verified_files,
        "expected_file_references": expected_files,
        "signatures_reverified": True,
        "issues": issues,
    }


def _license_attestation(mattools: Mapping[str, Any]) -> dict[str, Any]:
    attestation = _mapping(mattools.get("license_attestation"))
    purpose = str(attestation.get("use_purpose") or "").strip()
    normalized_purpose = re.sub(r"[^a-z0-9]+", " ", purpose.lower()).strip()
    placeholder = len(purpose) < 12 or normalized_purpose in {
        "",
        "unknown",
        "todo",
        "n a",
        "none",
        "test",
        "unit test",
        "placeholder",
    }
    basis = str(attestation.get("use_basis") or "")
    separate_evidence = _immutable_sha256(attestation.get("separate_license_evidence_sha256"))
    legal_basis_valid = (basis == "noncommercial" and separate_evidence is None) or (
        basis == "separately_licensed" and separate_evidence is not None
    )
    valid = all(
        (
            attestation.get("accepted") is True,
            not placeholder,
            legal_basis_valid,
            attestation.get("repository_license") == "Apache-2.0",
            attestation.get("dataset_card_license") == "CC-BY-NC-4.0",
            bool(str(attestation.get("attested_at") or "").strip()),
        )
    )
    return {
        "valid": valid,
        "use_basis": basis,
        "use_purpose": purpose,
        "separate_license_evidence_sha256": separate_evidence,
        "attested_at": attestation.get("attested_at"),
    }


def _retained_materials_validation_evidence(
    validation: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    issues: list[str] = []
    retained_path_text = str(validation.get("retained_path") or "").strip()
    retained_path = Path(retained_path_text).expanduser().resolve() if retained_path_text else None
    record_sha = _plain_sha256(validation.get("record_sha256"))
    canonical_sha = _plain_sha256(validation.get("canonical_sha256"))
    retained_sha = _plain_sha256(validation.get("retained_sha256"))
    size = _strict_int(validation.get("size_bytes"))
    retained_size = _strict_int(validation.get("retained_size_bytes"))
    regular = all(
        (
            retained_path is not None,
            retained_path is not None and retained_path.is_file(),
            retained_path is not None and not retained_path.is_symlink(),
            record_sha is not None,
            canonical_sha is not None,
            retained_sha == record_sha,
            size is not None and 0 < size <= 1_000_000,
            retained_size == size,
            retained_path is not None
            and record_sha is not None
            and retained_path.name == f"materials-validation-{record_sha}.json",
        )
    )
    if not regular or retained_path is None or size is None or record_sha is None:
        return {
            "valid": False,
            "path": retained_path_text or None,
            "sha256": record_sha,
            "size_bytes": size,
            "issues": ["retained materials validation file binding is incomplete"],
        }
    try:
        payload_bytes = retained_path.read_bytes()
    except OSError as exc:
        return {
            "valid": False,
            "path": str(retained_path),
            "sha256": record_sha,
            "size_bytes": size,
            "issues": [f"retained materials validation could not be read: {exc}"],
        }
    observed_sha = hashlib.sha256(payload_bytes).hexdigest()
    if len(payload_bytes) != size or observed_sha != record_sha:
        issues.append("retained materials validation bytes differ from the trace record")
    try:
        payload = json.loads(
            payload_bytes.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        if not isinstance(payload, dict):
            raise ValueError("materials validation record is not an object")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        issues.append(f"retained materials validation JSON is invalid: {exc}")
        payload = None

    module_name = "_ultra_materials_validation_for_readiness"
    validation_source = (
        repository_root.expanduser().resolve()
        / "backend/deepagents_runtime/src/ultra_deepagents/materials/validation.py"
    )
    if payload is not None:
        try:
            if validation_source.is_symlink() or not validation_source.is_file():
                raise ImportError("materials validation source is unavailable")
            spec = importlib.util.spec_from_file_location(module_name, validation_source)
            if spec is None or spec.loader is None:
                raise ImportError("materials validation source could not be loaded")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            assessment = module.parse_assessment_record(payload)
            canonical_bytes = module.canonical_record_json(assessment).encode("utf-8")
            decision_matches = all(
                (
                    assessment.run_status == validation.get("run_status"),
                    assessment.scientific_status.value == validation.get("scientific_status"),
                    assessment.verified is validation.get("verified"),
                    assessment.silent_success is validation.get("silent_success"),
                    hashlib.sha256(canonical_bytes).hexdigest() == canonical_sha,
                )
            )
            if not decision_matches:
                issues.append(
                    "retained materials validation decisions/canonical hash differ from the trace"
                )
        except (AttributeError, ImportError, OSError, TypeError, ValueError) as exc:
            issues.append(f"retained materials validation revalidation failed: {exc}")
        finally:
            sys.modules.pop(module_name, None)
    return {
        "valid": not issues,
        "path": str(retained_path),
        "sha256": observed_sha,
        "size_bytes": len(payload_bytes),
        "issues": issues,
    }


def _live_trace_evidence(
    reports: Sequence[Mapping[str, Any]],
    repository_root: Path,
) -> dict[str, Any]:
    designated = len(reports) > 0
    first_party_records_valid = designated
    silent_success_free = designated
    remote_mutation_aligned = designated
    evidence_integrity = designated
    retained_validation_artifacts = designated
    run_ids: set[str] = set()
    summaries: list[dict[str, Any]] = []

    for index, report in enumerate(reports, start=1):
        quality = _mapping(report.get("materials_quality") or report.get("quality"))
        signals = _mapping(quality.get("signals"))
        prompt = _mapping(report.get("prompt"))
        validation = _mapping(prompt.get("materials_validation") or report.get("validation"))
        artifacts = [_mapping(item) for item in _sequence(prompt.get("artifacts"))]
        run_id = str(prompt.get("run_id") or "").strip()
        thread_id = str(prompt.get("thread_id") or report.get("thread_id") or "").strip()
        tools = [str(value) for value in _sequence(prompt.get("tool_names"))]

        signal_set_ok = all(signals.get(name) is True for name in REQUIRED_LIVE_SIGNALS)
        first_party_record_ok = all(
            (
                quality.get("passed") is True,
                _sequence(quality.get("issues")) == [],
                signal_set_ok,
                quality.get("quality_scope") == "trace_and_first_party_validation_record",
                quality.get("independent_scientific_verification") is False,
                quality.get("scientific_conclusion_verified") is False,
                prompt.get("status") == "succeeded",
                validation.get("valid") is True,
                validation.get("verified") is True,
                validation.get("evidence_verified") is True,
                validation.get("scientific_status") == "verified",
                validation.get("run_status") == "succeeded",
                _sequence(validation.get("critical_failures")) == [],
                _sequence(validation.get("contradiction_failures")) == [],
                _sequence(validation.get("evidence_errors")) == [],
                "execute" in tools,
                bool(run_id),
                bool(thread_id),
                run_id not in run_ids,
            )
        )
        run_ids.add(run_id)
        no_silent = (
            signals.get("no_silent_success") is True and validation.get("silent_success") is False
        )
        remote_ok = (
            signals.get("remote_mutation_aligned") is True
            and signals.get("remote_mutation_scope_valid") is True
        )
        retained = _retained_materials_validation_evidence(validation, repository_root)

        canonical_id = str(validation.get("artifact_id") or "")
        canonical_hash = _plain_sha256(validation.get("canonical_sha256"))
        record_hash = _plain_sha256(validation.get("record_sha256"))
        canonical_size = _strict_int(validation.get("size_bytes"))
        candidates = [item for item in artifacts if item.get("artifact_id") == canonical_id]
        canonical = candidates[0] if len(candidates) == 1 else {}
        integrity_ok = all(
            (
                len(candidates) == 1,
                canonical_hash is not None,
                canonical_hash == record_hash,
                _plain_sha256(canonical.get("sha256")) == canonical_hash,
                canonical.get("run_id") == run_id,
                canonical.get("tool_name") == "outputs_collector",
                canonical.get("download_ok") is True,
                canonical.get("path") == "materials_validation.json",
                validation.get("durable_path") == "outputs/materials_validation.json",
                canonical_size is not None and 0 < canonical_size <= 1_000_000,
                _strict_int(canonical.get("size_bytes")) == canonical_size,
            )
        )
        first_party_records_valid = first_party_records_valid and first_party_record_ok
        silent_success_free = silent_success_free and no_silent
        remote_mutation_aligned = remote_mutation_aligned and remote_ok
        retained_ok = retained["valid"] is True
        evidence_integrity = evidence_integrity and integrity_ok and retained_ok
        retained_validation_artifacts = retained_validation_artifacts and retained_ok
        summaries.append(
            {
                "index": index,
                "run_id": run_id,
                "thread_id": thread_id,
                "first_party_scientific_record_valid": first_party_record_ok,
                "independent_scientific_verification": False,
                "silent_success": not no_silent,
                "remote_mutation_authorized": remote_ok,
                "evidence_integrity": integrity_ok and retained_ok,
                "retained_validation_artifact": retained,
                "validation_sha256": canonical_hash,
            }
        )

    return {
        "designated": designated,
        "first_party_records_valid": first_party_records_valid,
        "silent_success_free": silent_success_free,
        "remote_mutation_aligned": remote_mutation_aligned,
        "evidence_integrity": evidence_integrity,
        "retained_validation_artifacts": retained_validation_artifacts,
        "reports": summaries,
    }


def evaluate_readiness(
    *,
    deterministic_report: Mapping[str, Any],
    production_parity_report: Mapping[str, Any],
    calphad_ledger_report: Mapping[str, Any],
    calphad_cross_language_report: Mapping[str, Any],
    calphad_cross_language_report_manifest: Mapping[str, Any],
    mattools_report: Mapping[str, Any],
    mattools_report_manifest: Mapping[str, Any],
    live_trace_reports: Sequence[Mapping[str, Any]],
    repository_root: Path,
    benchmark_root: Path,
    expected: ExpectedProvenance,
    policy: ReadinessPolicy = ReadinessPolicy(),
    repository_state: Mapping[str, Any] | None = None,
    benchmark_state: Mapping[str, Any] | None = None,
    input_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute every full-readiness gate from the supplied evidence."""

    deterministic = _mapping(deterministic_report)
    production_parity = _mapping(production_parity_report)
    calphad_ledger = _mapping(calphad_ledger_report)
    calphad_cross_language = _mapping(calphad_cross_language_report)
    calphad_cross_language_manifest = _mapping(calphad_cross_language_report_manifest)
    mattools = _mapping(mattools_report)
    mattools_manifest = _mapping(mattools_report_manifest)
    traces = [_mapping(item) for item in live_trace_reports]
    observed_repo = (
        _mapping(repository_state)
        if repository_state is not None
        else inspect_repository(repository_root)
    )

    expected_git = _git_sha(expected.git_sha)
    expected_domain = _immutable_sha256(expected.domain_image)
    expected_runtime = _immutable_sha256(expected.runtime_image)
    expected_evaluator = _immutable_sha256(expected.evaluator_image)
    normalized_expected = ExpectedProvenance(
        git_sha=expected_git or "",
        domain_image=expected_domain or "",
        runtime_image=expected_runtime or "",
        evaluator_image=expected_evaluator or "",
    )
    metadata = _mapping(input_metadata)
    production_parity_evidence = _production_parity_evidence(
        production_parity,
        normalized_expected,
        _mapping(metadata.get("production_parity_report")),
        repository_root.expanduser().resolve(),
    )
    calphad_ledger_evidence = _calphad_ledger_evidence(
        calphad_ledger,
        metadata,
        repository_root.expanduser().resolve(),
        normalized_expected.git_sha,
    )
    calphad_cross_language_evidence = _calphad_cross_language_evidence(
        calphad_cross_language,
        calphad_cross_language_manifest,
        metadata,
        repository_root.expanduser().resolve(),
        normalized_expected,
    )

    domain_git = _mapping(deterministic.get("git"))
    mattools_ultra = _mapping(mattools.get("ultra"))
    parity_source = _mapping(production_parity.get("source"))
    parity_image = _mapping(production_parity.get("executed_image"))
    report_git_values = {
        _git_sha(domain_git.get("sha")),
        _git_sha(production_parity.get("expected_git_sha")),
        _git_sha(parity_source.get("observed_git_sha")),
        _git_sha(parity_image.get("revision")),
        _git_sha(mattools_ultra.get("commit")),
        _git_sha(observed_repo.get("commit")),
        expected_git,
    }
    aggregator_clean = all(
        (
            observed_repo.get("dirty") is False,
            observed_repo.get("inspection_ok", True) is True,
            _git_sha(observed_repo.get("commit")) == expected_git,
            expected_git is not None,
        )
    )
    same_clean_git = all(
        (
            None not in report_git_values,
            len(report_git_values) == 1,
            domain_git.get("dirty") is False,
            mattools_ultra.get("dirty") is False,
        )
    )

    provenance_policy = _mapping(deterministic.get("provenance_policy"))
    junit = _mapping(deterministic.get("junit"))
    deterministic_provenance = all(
        (
            provenance_policy.get("required") is True,
            provenance_policy.get("status") == "enforced",
            provenance_policy.get("promotion_provenance_enforced") is True,
            provenance_policy.get("would_pass_if_enforced") is True,
            _sequence(provenance_policy.get("issues")) == [],
        )
    )
    deterministic_required_invariants = all(
        (
            deterministic.get("schema_version") == 1,
            deterministic.get("gate") == "materials-domain-gate",
            deterministic.get("status") == "passed",
            _sequence(deterministic.get("failures")) == [],
            _strict_int(junit.get("tests")) == REQUIRED_DOMAIN_INVARIANT_COUNT,
            _strict_int(junit.get("failures")) == 0,
            _strict_int(junit.get("errors")) == 0,
            _strict_int(_mapping(deterministic.get("pytest")).get("exit_code")) == 0,
            _sequence(deterministic.get("version_drift")) == [],
            _domain_calphad_experimental_benchmark_valid(deterministic),
        )
    )
    deterministic_zero_skip = _strict_int(junit.get("skipped")) == 0
    invariant_evidence = _deterministic_invariant_evidence(deterministic, junit)

    domain_image_values = {
        value
        for value in (
            _immutable_sha256(_mapping(deterministic.get("image")).get("id")),
            _immutable_sha256(_mapping(deterministic.get("image")).get("digest")),
        )
        if value
    }
    policy_image_values = {
        _immutable_sha256(item.get("value") or item.get("digest"))
        for item in (
            _mapping(value)
            for value in _sequence(provenance_policy.get("immutable_image_identifiers"))
        )
    }
    runtime_image = _immutable_sha256(
        _mapping(mattools.get("runtime_environment")).get("image_digest")
    )
    expected_images_ok = all(
        (
            expected_domain is not None,
            expected_runtime is not None,
            expected_evaluator is not None,
            expected_domain in domain_image_values,
            expected_domain in policy_image_values,
            runtime_image == expected_runtime,
            expected_runtime != expected_evaluator,
        )
    )

    repository_evidence = _repository_evidence(
        deterministic, mattools, repository_root.expanduser().resolve()
    )
    host_validator_evidence = _host_validator_environment_evidence(
        mattools,
        repository_root.expanduser().resolve(),
        benchmark_root.expanduser().resolve(),
    )
    evaluator_lock_evidence = _evaluator_lock_evidence(
        mattools, repository_root.expanduser().resolve(), normalized_expected
    )
    mattools_manifest_evidence = _mattools_manifest_evidence(
        mattools_manifest,
        metadata,
        mattools,
        repository_root.expanduser().resolve(),
        benchmark_root.expanduser().resolve(),
    )
    benchmark = _mapping(mattools.get("benchmark"))
    benchmark_shape = _benchmark_shape(benchmark, policy)
    benchmark_evidence = _benchmark_evidence(
        benchmark, benchmark_root, policy, checkout_state=benchmark_state
    )
    recomputed = _trial_and_attempt_evidence(mattools, benchmark_shape, normalized_expected)
    published_counts_ok = _published_counts_consistent(mattools, recomputed)
    hard = _mapping(mattools.get("hard_gates"))
    evaluator_top = _mapping(mattools.get("official_evaluator_environment"))
    mattools_schema_ok = all(
        (
            mattools.get("schema_version") == SCHEMA_VERSION,
            _mapping(mattools.get("promotion")).get("scope") == "MatTools benchmark lane only",
            _mapping(mattools.get("runtime_environment")).get("evaluation_profile")
            == MATERIALS_CLEANROOM_PROFILE,
            evaluator_top.get("source_revision") == policy.official_revision,
            _mapping(evaluator_top.get("required_packages")) == EXPECTED_EVALUATOR_PACKAGES,
            len(_sequence(evaluator_top.get("observed_trials"))) == TRIAL_COUNT,
        )
    )
    all_mattools_gates = all(hard.get(name) is True for name in REQUIRED_MATTOOLS_HARD_GATES)
    audit = _mapping(mattools.get("checkpoint_evidence_audit"))
    replay_counts = [
        _strict_int(_mapping(trial).get("replay_count"))
        for trial in _sequence(mattools.get("trials"))
    ]
    replay_shape_valid = len(replay_counts) == TRIAL_COUNT and all(
        value is not None and 2 <= value <= 4 for value in replay_counts
    )
    expected_replay_records = sum(int(value or 0) for value in replay_counts)
    checkpoint_integrity = all(
        (
            hard.get("checkpoint_evidence_integrity") is True,
            hard.get("replay_terminal_evidence_integrity") is True,
            audit.get("valid") is True,
            _sequence(audit.get("issues")) == [],
            audit.get("trusted_state_booleans") is False,
            audit.get("terminal_attempt_directory_exact") is True,
            audit.get("terminal_replay_directory_exact") is True,
            _strict_int(audit.get("verified_attempt_count")) == RUNNABLE_DENOMINATOR,
            replay_shape_valid,
            _strict_int(audit.get("recomputed_replay_count")) == expected_replay_records,
            _strict_int(audit.get("expected_attempt_count")) == RUNNABLE_DENOMINATOR,
            _strict_int(audit.get("actual_attempt_count")) == RUNNABLE_DENOMINATOR,
            _strict_int(audit.get("verified_replay_terminal_record_count"))
            == expected_replay_records,
            _strict_int(audit.get("expected_replay_terminal_record_count"))
            == expected_replay_records,
            audit.get("terminal_replays_non_replaced") is True,
            _strict_int(audit.get("failed_replay_terminal_record_count")) == 0,
        )
    )
    checkpoint_non_erasure = all(
        (
            hard.get("checkpoint_non_erasure_integrity") is True,
            audit.get("attempt_key_set_exact") is True,
            audit.get("terminal_attempts_non_replaced") is True,
            audit.get("terminal_replays_non_replaced") is True,
            audit.get("terminal_attempt_directory_exact") is True,
            audit.get("terminal_replay_directory_exact") is True,
        )
    )
    answer_isolation = (
        hard.get("expected_values_and_verifiers_isolated") is True
        and benchmark_shape["answers_and_verifiers_isolated"] is True
    )
    license_evidence = _license_attestation(mattools)
    isolation_evidence = _isolation_attestations(mattools, normalized_expected, repository_root)
    live_evidence = _live_trace_evidence(traces, repository_root)
    counts = _mapping(mattools.get("counts"))
    runnable = _strict_int(counts.get("runnable"))
    official_scientific = _strict_int(counts.get("scientific_pass"))
    strict_scientific = _strict_int(counts.get("strict_scientific_pass"))

    calphad_experimental_wrapper = _mapping(deterministic.get("calphad_experimental_benchmark"))
    calphad_experimental_report = _mapping(calphad_experimental_wrapper.get("report"))
    calphad_experimental_lanes = _mapping(calphad_experimental_report.get("lanes"))
    hard_gates = {
        "aggregator_checkout_clean": aggregator_clean,
        "same_clean_git_sha": same_clean_git,
        "production_full_image_parity": production_parity_evidence["valid"],
        "calphad_postgres_ledger_qualified": calphad_ledger_evidence["valid"],
        "calphad_typed_cli_http_postgres_cross_language_qualified": (
            calphad_cross_language_evidence["valid"]
        ),
        "expected_immutable_images": expected_images_ok,
        "repository_evidence_rehashed": repository_evidence["valid"],
        "mattools_report_manifest_integrity": mattools_manifest_evidence["valid"],
        "mattools_host_validator_environment_exact": host_validator_evidence["valid"],
        "deterministic_clean_provenance_enforced": deterministic_provenance,
        "deterministic_required_invariants": deterministic_required_invariants,
        "calphad_experimental_two_lane_benchmark": (
            _domain_calphad_experimental_benchmark_valid(deterministic)
        ),
        "deterministic_zero_skip": deterministic_zero_skip,
        "deterministic_invariant_evidence_complete": invariant_evidence["valid"],
        "mattools_report_schema": mattools_schema_ok,
        "mattools_official_snapshot": benchmark_shape["valid"]
        and hard.get("official_snapshot") is True,
        "benchmark_evidence_rehashed": benchmark_evidence["valid"],
        "mattools_three_trial_coverage": recomputed["complete"],
        "mattools_published_counts_consistent": published_counts_ok,
        "mattools_function_runnable_rate": runnable is not None and runnable >= RUNNABLE_MINIMUM,
        "mattools_official_scientific_correctness": official_scientific is not None
        and official_scientific >= SCIENTIFIC_MINIMUM,
        "mattools_strict_scientific_correctness": strict_scientific is not None
        and strict_scientific >= SCIENTIFIC_MINIMUM,
        "mattools_checkpoint_integrity": checkpoint_integrity,
        "mattools_checkpoint_non_erasure": checkpoint_non_erasure,
        "mattools_replay_terminal_evidence": checkpoint_integrity
        and audit.get("terminal_replays_non_replaced") is True,
        "mattools_answer_isolation": answer_isolation,
        "mattools_evaluator_environment_exact": recomputed["evaluator_exact"]
        and evaluator_lock_evidence["valid"]
        and hard.get("official_evaluator_environment_exact") is True,
        "mattools_evaluator_independence": recomputed["evaluator_independent"]
        and hard.get("evaluator_independent_from_production") is True,
        "mattools_immediate_replay_reproducible": recomputed["reproducible"]
        and hard.get("immediate_replay_reproducible") is True,
        "mattools_production_execute_evidence": recomputed["execute_evidence"]
        and hard.get("production_execute_tool_evidence") is True,
        "mattools_server_authorized_cleanroom_profile": recomputed[
            "server_authorized_cleanroom_profile"
        ]
        and hard.get("server_authorized_cleanroom_profile") is True,
        "mattools_worker_enforced_cleanroom_profile": recomputed[
            "worker_enforced_cleanroom_profile"
        ]
        and hard.get("worker_enforced_cleanroom_profile") is True,
        "mattools_execute_image_attestation": recomputed["execute_image_attestation"]
        and hard.get("production_execute_runtime_image_attestation") is True,
        "mattools_observable_model_provider": recomputed["runtime_provenance"]
        and hard.get("actual_model_provider_provenance") is True,
        "mattools_no_unauthorized_remote_mutation": recomputed["remote_mutation_free"],
        "mattools_required_solution_artifacts": hard.get("required_solution_artifacts") is True,
        "mattools_per_trial_function_runnable": recomputed["complete"]
        and recomputed["per_trial_runnable_floor"]
        and hard.get("per_trial_mattools_function_runnable") is True,
        "mattools_per_trial_strict_scientific_correctness": recomputed["complete"]
        and recomputed["per_trial_strict_scientific_floor"]
        and hard.get("per_trial_strict_scientific_task_success") is True,
        "mattools_lane_hard_gates": all_mattools_gates
        and _mapping(mattools.get("promotion")).get("passed") is True,
        "external_license_attestation": license_evidence["valid"],
        "external_isolation_attestations": isolation_evidence["valid"]
        and hard.get("external_sandbox_isolation_evidence") is True,
        "live_traces_designated": live_evidence["designated"],
        "live_traces_first_party_records_valid": live_evidence["first_party_records_valid"],
        "live_traces_no_silent_success": live_evidence["silent_success_free"],
        "live_traces_remote_mutation_authorized": live_evidence["remote_mutation_aligned"],
        "live_traces_evidence_integrity": live_evidence["evidence_integrity"],
        "live_traces_retained_validation_artifacts": live_evidence["retained_validation_artifacts"],
    }
    reasons = [name for name, passed in hard_gates.items() if passed is not True]
    passed = not reasons

    return {
        "schema_version": SCHEMA_VERSION,
        "gate": "materials-production-readiness",
        "scope": "full-materials-production-readiness",
        "generated_at_utc": utc_now(),
        "status": "candidate_for_attestation" if passed else "blocked",
        "inputs": dict(input_metadata or {}),
        "expected_provenance": normalized_expected._asdict(),
        "observed_provenance": {
            "aggregator_repository": observed_repo,
            "deterministic_git": domain_git,
            "production_parity_git": production_parity_evidence["git_sha"],
            "production_parity_runtime_image": production_parity_evidence["runtime_image"],
            "calphad_cross_language_runtime_image": calphad_cross_language_evidence[
                "runtime_image_id"
            ],
            "mattools_ultra": mattools_ultra,
            "deterministic_images": sorted(domain_image_values),
            "runtime_image": runtime_image,
            "evaluator_image": expected_evaluator,
        },
        "counts": {
            "production_parity": {
                "scope": production_parity_evidence["scope"],
                "passed": production_parity_evidence["valid"],
            },
            "calphad_ledger": {
                "passed": calphad_ledger_evidence["valid"],
                "tests": len(REQUIRED_CALPHAD_LEDGER_TESTS),
                "source_files": len(REQUIRED_CALPHAD_LEDGER_SOURCE_FILES),
            },
            "calphad_cross_language": {
                "passed": calphad_cross_language_evidence["valid"],
                "live_http_callback": calphad_cross_language_evidence["live_http_callback"],
                "live_postgres": calphad_cross_language_evidence["live_postgres"],
                "source_files": len(REQUIRED_CALPHAD_CROSS_LANGUAGE_SOURCE_FILES),
            },
            "calphad_experimental": {
                "passed": _domain_calphad_experimental_benchmark_valid(deterministic),
                "calibration_observations": _strict_int(
                    _mapping(calphad_experimental_lanes.get("calibration")).get("observation_count")
                ),
                "held_out_observations": _strict_int(
                    _mapping(calphad_experimental_lanes.get("held_out")).get("observation_count")
                ),
            },
            "deterministic": {
                "passed": invariant_evidence["passed"],
                "total": _strict_int(junit.get("tests")),
                "skipped": _strict_int(junit.get("skipped")),
            },
            "mattools": {
                "runnable": runnable,
                "runnable_denominator": RUNNABLE_DENOMINATOR,
                "runnable_minimum": RUNNABLE_MINIMUM,
                "per_trial_runnable_minimum": PER_TRIAL_RUNNABLE_MINIMUM,
                "scientific_pass": official_scientific,
                "strict_scientific_pass": strict_scientific,
                "scientific_denominator": SCIENTIFIC_DENOMINATOR,
                "scientific_minimum": SCIENTIFIC_MINIMUM,
                "per_trial_scientific_minimum": PER_TRIAL_SCIENTIFIC_MINIMUM,
                "per_trial": recomputed["per_trial"],
                "recomputed_attempts": recomputed["attempt_count"],
            },
            "designated_live_traces": len(traces),
        },
        "rates": {
            "mattools_function_runnable": (
                runnable / RUNNABLE_DENOMINATOR if runnable is not None else None
            ),
            "mattools_official_task_success": (
                official_scientific / SCIENTIFIC_DENOMINATOR
                if official_scientific is not None
                else None
            ),
            "mattools_strict_task_success": (
                strict_scientific / SCIENTIFIC_DENOMINATOR
                if strict_scientific is not None
                else None
            ),
        },
        "evidence_revalidation": {
            "repository": repository_evidence,
            "production_parity": production_parity_evidence,
            "calphad_ledger": calphad_ledger_evidence,
            "calphad_cross_language": calphad_cross_language_evidence,
            "calphad_experimental": calphad_experimental_wrapper,
            "deterministic_invariants": invariant_evidence,
            "benchmark": benchmark_evidence,
            "mattools_recomputed": recomputed,
            "mattools_report_manifest": mattools_manifest_evidence,
            "mattools_host_validator": host_validator_evidence,
            "mattools_evaluator_lock": evaluator_lock_evidence,
            "license": license_evidence,
            "isolation": isolation_evidence,
            "live_traces": live_evidence,
        },
        "hard_gates": hard_gates,
        "promotion": {
            "passed": passed,
            "evidence_passed": passed,
            "attestation_required": True,
            "distribution_ready": False,
            "full_materials_production_ready": False,
            "product_label": (
                "materials science promotion candidate"
                if passed
                else "materials science research preview"
            ),
            "reasons": reasons,
        },
    }


def _md(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ") or "—"


def render_markdown(report: Mapping[str, Any]) -> str:
    promotion = _mapping(report.get("promotion"))
    counts = _mapping(report.get("counts"))
    production_parity = _mapping(counts.get("production_parity"))
    calphad_ledger = _mapping(counts.get("calphad_ledger"))
    calphad_cross_language = _mapping(counts.get("calphad_cross_language"))
    calphad_experimental = _mapping(counts.get("calphad_experimental"))
    deterministic = _mapping(counts.get("deterministic"))
    mattools = _mapping(counts.get("mattools"))
    rates = _mapping(report.get("rates"))
    status = "CANDIDATE — ATTESTATION REQUIRED" if promotion.get("passed") is True else "BLOCKED"
    lines = [
        "# Materials Production-Readiness Gate",
        "",
        f"**Status: {status}**",
        "",
        f"Product label: **{_md(promotion.get('product_label'))}**",
        f"Distribution ready: **{_md(promotion.get('distribution_ready'))}**",
        "",
        "## Decisive evidence",
        "",
        "| Lane | Result |",
        "|---|---|",
        (
            "| Full production image parity | "
            f"{'PASS' if production_parity.get('passed') is True else 'BLOCK'} "
            f"({_md(production_parity.get('scope'))}) |"
        ),
        (
            "| CALPHAD PostgreSQL ledger | "
            f"{'PASS' if calphad_ledger.get('passed') is True else 'BLOCK'} "
            f"({_md(calphad_ledger.get('tests'))} tests; "
            f"{_md(calphad_ledger.get('source_files'))} source files) |"
        ),
        (
            "| CALPHAD typed CLI → HTTP → PostgreSQL | "
            f"{'PASS' if calphad_cross_language.get('passed') is True else 'BLOCK'} "
            f"(HTTP {_md(calphad_cross_language.get('live_http_callback'))}; "
            f"PostgreSQL {_md(calphad_cross_language.get('live_postgres'))}) |"
        ),
        (
            "| CALPHAD experimental calibration + holdout | "
            f"{'PASS' if calphad_experimental.get('passed') is True else 'BLOCK'} "
            f"({_md(calphad_experimental.get('calibration_observations'))} calibration; "
            f"{_md(calphad_experimental.get('held_out_observations'))} held-out) |"
        ),
        (
            "| Deterministic domain suite | "
            f"{_md(deterministic.get('passed'))}/{_md(deterministic.get('total'))}; "
            f"skipped {_md(deterministic.get('skipped'))} |"
        ),
        (
            "| MatTools runnable | "
            f"{_md(mattools.get('runnable'))}/{_md(mattools.get('runnable_denominator'))} "
            f"({_md(rates.get('mattools_function_runnable'))}) |"
        ),
        (
            "| MatTools official scientific | "
            f"{_md(mattools.get('scientific_pass'))}/{_md(mattools.get('scientific_denominator'))} "
            f"({_md(rates.get('mattools_official_task_success'))}) |"
        ),
        (
            "| MatTools strict-shadow scientific | "
            f"{_md(mattools.get('strict_scientific_pass'))}/"
            f"{_md(mattools.get('scientific_denominator'))} "
            f"({_md(rates.get('mattools_strict_task_success'))}) |"
        ),
        f"| Designated live traces | {_md(counts.get('designated_live_traces'))} |",
        "",
        "## Hard gates",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    for name, passed in _mapping(report.get("hard_gates")).items():
        lines.append(f"| `{_md(name)}` | {'PASS' if passed is True else 'BLOCK'} |")
    reasons = [str(value) for value in _sequence(promotion.get("reasons"))]
    lines.extend(["", "## Decision", ""])
    if reasons:
        lines.append("Promotion is blocked by:")
        lines.append("")
        lines.extend(f"- `{_md(reason)}`" for reason in reasons)
    else:
        lines.append(
            "Full-image parity plus all deterministic, MatTools, provenance, isolation, and "
            "designated live-trace requirements passed."
        )
    lines.append("")
    return "\n".join(lines)


def _atomic_write(path: Path, content: bytes) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=resolved.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, resolved)


def _write_once_bytes(path: Path, content: bytes) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(resolved, flags, 0o600)
    except FileExistsError as exc:
        if resolved.read_bytes() != content:
            raise GateInputError(
                f"refusing to replace content-addressed readiness evidence {resolved}"
            ) from exc
        return
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise GateInputError(f"could not finish readiness evidence {resolved}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_outputs(
    report: Mapping[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    json_bytes = (json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    markdown_bytes = render_markdown(report).encode("utf-8")
    json_digest = hashlib.sha256(json_bytes).hexdigest()
    immutable_json_path = (
        json_path.expanduser().resolve().parent
        / f"materials-production-readiness-{json_digest}.json"
    )
    _write_once_bytes(immutable_json_path, json_bytes)
    _atomic_write(json_path, json_bytes)
    _atomic_write(markdown_path, markdown_bytes)
    manifest = {
        "schema_version": "1",
        "generated_at_utc": utc_now(),
        "report": {
            "path": immutable_json_path.name,
            "sha256": json_digest,
            "size_bytes": len(json_bytes),
        },
        "markdown": {
            "path": str(markdown_path.expanduser().resolve()),
            "sha256": hashlib.sha256(markdown_bytes).hexdigest(),
            "size_bytes": len(markdown_bytes),
        },
        "expected_provenance": _mapping(report.get("expected_provenance")),
        "promotion_passed": _mapping(report.get("promotion")).get("passed") is True,
        "evidence_passed": _mapping(report.get("promotion")).get("evidence_passed") is True,
        "attestation_required": True,
        "full_materials_production_ready": False,
    }
    if manifest_path is not None:
        _atomic_write(
            manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deterministic-report", required=True, type=Path)
    parser.add_argument("--production-parity-report", required=True, type=Path)
    parser.add_argument("--calphad-ledger-report", required=True, type=Path)
    parser.add_argument("--calphad-cross-language-report", required=True, type=Path)
    parser.add_argument("--calphad-cross-language-report-manifest", required=True, type=Path)
    parser.add_argument("--mattools-report", required=True, type=Path)
    parser.add_argument("--mattools-report-manifest", required=True, type=Path)
    parser.add_argument("--live-trace", required=True, action="append", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--expected-domain-image", required=True)
    parser.add_argument("--expected-runtime-image", required=True)
    parser.add_argument("--expected-evaluator-image", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    deterministic, deterministic_meta = load_json_report(args.deterministic_report)
    production_parity, production_parity_meta = load_json_report(args.production_parity_report)
    calphad_ledger, calphad_ledger_meta = load_json_report(args.calphad_ledger_report)
    calphad_cross_language, calphad_cross_language_meta = load_json_report(
        args.calphad_cross_language_report
    )
    calphad_cross_language_manifest, calphad_cross_language_manifest_meta = load_json_report(
        args.calphad_cross_language_report_manifest
    )
    mattools, mattools_meta = load_json_report(args.mattools_report)
    mattools_manifest, mattools_manifest_meta = load_json_report(args.mattools_report_manifest)
    traces: list[dict[str, Any]] = []
    trace_meta: list[dict[str, Any]] = []
    for path in args.live_trace:
        report, metadata = load_json_report(path)
        traces.append(report)
        trace_meta.append(metadata)
    report = evaluate_readiness(
        deterministic_report=deterministic,
        production_parity_report=production_parity,
        calphad_ledger_report=calphad_ledger,
        calphad_cross_language_report=calphad_cross_language,
        calphad_cross_language_report_manifest=calphad_cross_language_manifest,
        mattools_report=mattools,
        mattools_report_manifest=mattools_manifest,
        live_trace_reports=traces,
        repository_root=args.repository_root,
        benchmark_root=args.benchmark_root,
        expected=ExpectedProvenance(
            git_sha=args.expected_git_sha,
            domain_image=args.expected_domain_image,
            runtime_image=args.expected_runtime_image,
            evaluator_image=args.expected_evaluator_image,
        ),
        input_metadata={
            "deterministic_report": deterministic_meta,
            "production_parity_report": production_parity_meta,
            "calphad_ledger_report": calphad_ledger_meta,
            "calphad_cross_language_report": calphad_cross_language_meta,
            "calphad_cross_language_report_manifest": calphad_cross_language_manifest_meta,
            "mattools_report": mattools_meta,
            "mattools_report_manifest": mattools_manifest_meta,
            "live_trace_reports": trace_meta,
        },
    )
    write_outputs(
        report,
        json_path=args.output_json,
        markdown_path=args.output_markdown,
        manifest_path=args.output_manifest,
    )
    print(json.dumps(report["promotion"], indent=2, sort_keys=True))
    return 0 if report["promotion"]["passed"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateInputError as exc:
        print(f"materials readiness gate configuration error: {exc}", file=os.sys.stderr)
        raise SystemExit(2) from exc
