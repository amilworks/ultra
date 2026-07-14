package store

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"slices"
	"strings"

	"github.com/jackc/pgx/v5"
)

type schemaQuerier interface {
	QueryRow(ctx context.Context, sql string, args ...any) pgx.Row
}

var requiredPostgresControlTables = []string{
	"control_threads",
	"control_organizations",
	"control_users",
	"control_user_token_usage_daily",
	"control_user_token_usage_lifetime",
	"control_run_token_usage",
	"control_run_token_usage_finalized",
	"control_thread_messages",
	"control_runs",
	"control_run_events",
	"control_run_event_sequences",
	"control_run_leases",
	"control_worker_heartbeats",
	"control_artifacts",
	"control_resources",
	"control_resource_search_documents",
	"control_resource_search_facts",
	"control_resource_events",
	"control_calphad_input_blobs",
	"control_calphad_revisions",
	"control_calphad_evidence_blobs",
	"control_calphad_validation_events",
	"control_calphad_tenant_capacity",
	"control_resource_share_grants",
	"control_resource_collections",
	"control_resource_collection_share_grants",
	"control_resource_collection_members",
	"control_dataset_snapshots",
	"control_dataset_snapshot_resources",
	"control_dataset_snapshot_share_grants",
	"control_dataset_snapshot_events",
	"control_data_agent_jobs",
	"control_data_agent_job_resources",
	"control_data_agent_job_events",
	"control_data_agent_job_leases",
	"control_upload_sessions",
	"control_upload_session_files",
	"control_upload_session_events",
	"control_upload_chunks",
	"control_bisque_credentials",
	"control_training_models",
	"control_training_domains",
	"control_training_lineages",
	"control_training_model_versions",
	"control_training_model_status",
	"control_training_gate_policies",
	"control_training_guardrail_clauses",
	"control_training_gate_config_events",
	"control_training_gold_sets",
	"control_training_gold_items",
	"control_training_replay_pool",
	"control_training_benchmark_runs",
	"control_training_canary_observations",
	"control_training_retrain_requests",
	"control_training_model_version_events",
	"control_training_jobs",
	"control_training_job_events",
	"control_training_job_leases",
}

var requiredCalphadTables = []string{
	"control_calphad_evidence_blobs",
	"control_calphad_input_blobs",
	"control_calphad_revisions",
	"control_calphad_tenant_capacity",
	"control_calphad_validation_events",
}

const calphadCreateRevisionFunctionSignature = "public.ultra_create_calphad_revision_v1(text,text,text,text,text,bigint,text,double precision,double precision,bytea,jsonb)"

const calphadAppendValidationFunctionSignature = "public.ultra_append_calphad_validation_v1(text,text,text,text,bigint,text,jsonb,double precision,double precision,text,text,text,text,text,text,text,text,text,bigint,bytea,text,text,text,text,text,text,jsonb)"

var requiredCalphadInternalFunctions = []string{
	"ultra_reject_calphad_ledger_mutation",
	"ultra_validate_calphad_equilibrium_inspection_lineage",
	"ultra_validate_calphad_input_retention",
	"ultra_validate_calphad_pressure_binding",
	"ultra_validate_calphad_revision_parent",
	"ultra_validate_calphad_validation_run_authority",
}

type calphadSchemaCatalog struct {
	Columns                 []string
	Constraints             []string
	Indexes                 []string
	Functions               []string
	Triggers                []string
	WriterFunctionsExact    bool
	CapacityAccountingValid bool
}

type CalphadServingRoleStatus struct {
	Role                       string
	Superuser                  bool
	CreateRole                 bool
	CreateDB                   bool
	Replication                bool
	BypassRLS                  bool
	OwnedTables                []string
	OwnedFunctions             []string
	OwnerRoles                 []string
	ReachableRoles             []string
	OwnerRoleReachable         bool
	PublicSchemaOwner          string
	PublicOwnerReachable       bool
	CanCreatePublicSchema      bool
	CanSelectAll               bool
	CanInsertAll               bool
	CanInsertAny               bool
	CanMutateCalphad           bool
	CanExecuteCreateRevision   bool
	CanExecuteAppendValidation bool
	WriterFunctionsExact       bool
	CanExecuteUnexpectedWriter bool
	CanExecuteInternal         bool
	PublicCanExecute           bool
	UnexpectedTableACLGrantees []string
	UnexpectedFuncACLGrantees  []string
}

var requiredCalphadColumns = []string{
	"control_calphad_evidence_blobs.1.evidence_sha256:text:NO",
	"control_calphad_evidence_blobs.2.evidence_size_bytes:int8:NO",
	"control_calphad_evidence_blobs.3.encoding:text:NO",
	"control_calphad_evidence_blobs.4.payload:bytea:NO",
	"control_calphad_evidence_blobs.5.created_at:timestamptz:NO",
	"control_calphad_input_blobs.1.input_sha256:text:NO",
	"control_calphad_input_blobs.2.input_size_bytes:int8:NO",
	"control_calphad_input_blobs.3.encoding:text:NO",
	"control_calphad_input_blobs.4.payload:bytea:NO",
	"control_calphad_input_blobs.5.created_at:timestamptz:NO",
	"control_calphad_tenant_capacity.1.owner_user_id:text:NO",
	"control_calphad_tenant_capacity.2.owner_org_id:text:NO",
	"control_calphad_tenant_capacity.3.max_retained_bytes:int8:NO",
	"control_calphad_tenant_capacity.4.max_validation_events:int8:NO",
	"control_calphad_tenant_capacity.5.retained_input_bytes:int8:NO",
	"control_calphad_tenant_capacity.6.retained_evidence_bytes:int8:NO",
	"control_calphad_tenant_capacity.7.validation_events:int8:NO",
	"control_calphad_tenant_capacity.8.updated_at:timestamptz:NO",
	"control_calphad_revisions.1.revision_id:text:NO",
	"control_calphad_revisions.2.resource_id:text:NO",
	"control_calphad_revisions.3.owner_user_id:text:NO",
	"control_calphad_revisions.4.owner_org_id:text:YES",
	"control_calphad_revisions.5.sha256:text:NO",
	"control_calphad_revisions.6.size_bytes:int8:NO",
	"control_calphad_revisions.7.database_format:text:NO",
	"control_calphad_revisions.8.assessment_pressure_min_pa:float8:YES",
	"control_calphad_revisions.9.assessment_pressure_max_pa:float8:YES",
	"control_calphad_revisions.10.parent_revision_id:text:YES",
	"control_calphad_revisions.11.created_by_user_id:text:YES",
	"control_calphad_revisions.12.created_at:timestamptz:NO",
	"control_calphad_revisions.13.metadata:jsonb:NO",
	"control_calphad_validation_events.1.validation_id:text:NO",
	"control_calphad_validation_events.2.revision_id:text:NO",
	"control_calphad_validation_events.3.resource_id:text:NO",
	"control_calphad_validation_events.4.database_sha256:text:NO",
	"control_calphad_validation_events.5.database_size_bytes:int8:NO",
	"control_calphad_validation_events.6.database_format:text:NO",
	"control_calphad_validation_events.7.assessment_pressure_min_pa:float8:YES",
	"control_calphad_validation_events.8.assessment_pressure_max_pa:float8:YES",
	"control_calphad_validation_events.9.database_inventory_sha256:text:YES",
	"control_calphad_validation_events.10.request_sha256:text:YES",
	"control_calphad_validation_events.11.status:text:NO",
	"control_calphad_validation_events.12.operation:text:NO",
	"control_calphad_validation_events.13.failure_domain:text:YES",
	"control_calphad_validation_events.14.failure_stage:text:YES",
	"control_calphad_validation_events.15.failure_code:text:YES",
	"control_calphad_validation_events.16.evidence_path:text:YES",
	"control_calphad_validation_events.17.evidence_sha256:text:YES",
	"control_calphad_validation_events.18.evidence_size_bytes:int8:YES",
	"control_calphad_validation_events.19.runtime_image_id:text:YES",
	"control_calphad_validation_events.20.pycalphad_version:text:YES",
	"control_calphad_validation_events.21.run_id:text:YES",
	"control_calphad_validation_events.22.inspection_evidence_sha256:text:YES",
	"control_calphad_validation_events.23.evidence_contract_version:text:YES",
	"control_calphad_validation_events.24.created_by_authority:text:NO",
	"control_calphad_validation_events.25.created_at:timestamptz:NO",
	"control_calphad_validation_events.26.metadata:jsonb:NO",
}

var requiredCalphadConstraintFingerprints = map[string][]string{
	"control_calphad_tenant_capacity_pkey": {
		"p true", "primary key (owner_user_id, owner_org_id)",
	},
	"control_calphad_tenant_capacity_limits_check": {
		"c true", "max_retained_bytes > 0", "max_validation_events > 0",
	},
	"control_calphad_tenant_capacity_counters_check": {
		"c true", "retained_input_bytes >= 0", "retained_evidence_bytes >= 0",
		"validation_events >= 0", "max_retained_bytes", "max_validation_events",
	},
	"control_calphad_input_blob_binding_unique": {
		"u", "unique (input_sha256, input_size_bytes)",
	},
	"control_calphad_input_blob_size_check": {
		"c true", "input_size_bytes >= 1", "input_size_bytes <= 67108864",
	},
	"control_calphad_input_blob_encoding_check": {
		"c true", "encoding = 'raw'",
	},
	"control_calphad_input_blob_payload_sha_check": {
		"c true", "encode(sha256(payload), 'hex'::text) = input_sha256",
	},
	"control_calphad_input_blob_payload_size_check": {
		"c true", "octet_length(payload) = input_size_bytes",
	},
	"control_calphad_revisions_binding_unique": {
		"u", "unique (revision_id, resource_id, sha256, size_bytes, database_format)",
	},
	"control_calphad_revisions_database_format_check": {
		"c", "database_format", "tdb", "dat",
	},
	"control_calphad_revisions_pressure_binding_unique": {
		"u", "unique (revision_id, assessment_pressure_min_pa, assessment_pressure_max_pa)",
	},
	"control_calphad_revisions_pressure_limits_check": {
		"c", "assessment_pressure_min_pa is not null", "assessment_pressure_max_pa is not null",
		"assessment_pressure_min_pa >=", "assessment_pressure_max_pa <=",
		"assessment_pressure_min_pa <= assessment_pressure_max_pa",
	},
	"control_calphad_revisions_pressure_metadata_check": {
		"c", "assessment_pressure_limits_Pa", "jsonb_typeof", "jsonb_array_length",
		"assessment_pressure_min_pa", "assessment_pressure_max_pa",
	},
	"control_calphad_revisions_owner_declaration_check": {
		"c true", "? 'owner_declaration'", "owner_declaration", "ultra.calphad.owner-declaration.v1", "resource_owner",
		"database_id", "source", "license_id", "assessment_scope", "reference_state",
		"assessment_temperature_limits_K", "assessment_pressure_limits_Pa", "database_format",
		"jsonb_build_object", "jsonb_typeof",
	},
	"control_calphad_revisions_input_blob_fkey": {
		"f", "foreign key (sha256, size_bytes)",
		"references control_calphad_input_blobs(input_sha256, input_size_bytes)",
	},
	"control_calphad_evidence_blob_binding_unique": {
		"u", "unique (evidence_sha256, evidence_size_bytes)",
	},
	"control_calphad_evidence_blob_payload_sha_check": {
		"c true", "encode(sha256(payload), 'hex'::text) = evidence_sha256",
	},
	"control_calphad_validation_revision_binding_fkey": {
		"f true", "foreign key (revision_id, resource_id, database_sha256, database_size_bytes, database_format)",
		"references control_calphad_revisions(revision_id, resource_id, sha256, size_bytes, database_format)",
	},
	"control_calphad_validation_pressure_binding_fkey": {
		"f", "foreign key (revision_id, assessment_pressure_min_pa, assessment_pressure_max_pa)",
		"references control_calphad_revisions(revision_id, assessment_pressure_min_pa, assessment_pressure_max_pa)",
	},
	"control_calphad_validation_evidence_blob_fkey": {
		"f", "foreign key (evidence_sha256, evidence_size_bytes)",
		"references control_calphad_evidence_blobs(evidence_sha256, evidence_size_bytes)",
	},
	"control_calphad_validation_run_fkey": {
		"f", "foreign key (run_id) references control_runs(run_id)",
	},
	"control_calphad_validation_inspection_blob_fkey": {
		"f true", "foreign key (inspection_evidence_sha256)",
		"references control_calphad_evidence_blobs(evidence_sha256)",
	},
	"control_calphad_validation_inspection_lineage_check": {
		"c true", "operation = any", "equilibrium", "scheil", "inspection_evidence_sha256 is not null",
	},
	"control_calphad_validation_pycalphad_version_check": {
		"c true", "operation = 'registration'", "pycalphad_version = '0.11.2'",
	},
	"control_calphad_validation_database_sha_check": {
		"c true", "database_sha256 ~ '^[0-9a-f]{64}$'",
	},
	"control_calphad_validation_database_size_check": {
		"c true", "database_size_bytes > 0",
	},
	"control_calphad_validation_database_format_check": {
		"c", "database_format", "tdb", "dat",
	},
	"control_calphad_validation_pressure_limits_check": {
		"c", "assessment_pressure_min_pa is not null", "assessment_pressure_max_pa is not null",
		"assessment_pressure_min_pa >=", "assessment_pressure_max_pa <=",
		"assessment_pressure_min_pa <= assessment_pressure_max_pa",
	},
	"control_calphad_validation_pressure_metadata_check": {
		"c", "assessment_pressure_limits_Pa", "jsonb_typeof", "jsonb_array_length",
		"assessment_pressure_min_pa", "assessment_pressure_max_pa",
	},
	"control_calphad_validation_status_check": {
		"c true", "status = any", "pending", "input_validated", "equilibrium_completed", "scheil_completed", "failed", "timeout", "unsupported",
	},
	"control_calphad_validation_operation_check": {
		"c true", "operation = any", "registration", "inspect", "equilibrium", "scheil",
	},
	"control_calphad_validation_evidence_sha_check": {
		"c true", "evidence_sha256 is null", "evidence_sha256 ~ '^[0-9a-f]{64}$'",
	},
	"control_calphad_validation_evidence_size_check": {
		"c true", "evidence_size_bytes is null", "evidence_size_bytes >= 1",
		"evidence_size_bytes <= 33554432",
	},
	"control_calphad_validation_runtime_image_check": {
		"c true", "runtime_image_id is null", "runtime_image_id ~ '^sha256:[0-9a-f]{64}$'",
	},
	"control_calphad_validation_inspection_sha_check": {
		"c true", "inspection_evidence_sha256 is null", "inspection_evidence_sha256 ~ '^[0-9a-f]{64}$'",
	},
	"control_calphad_validation_inventory_sha_check": {
		"c true", "database_inventory_sha256 is null", "database_inventory_sha256 ~ '^[0-9a-f]{64}$'",
	},
	"control_calphad_validation_request_sha_check": {
		"c true", "request_sha256 is null", "request_sha256 ~ '^[0-9a-f]{64}$'",
	},
	"control_calphad_validation_evidence_contract_check": {
		"c", "evidence_contract_version is null", "ultra.calphad.retained-evidence.v2",
	},
	"control_calphad_validation_authority_check": {
		"c true", "created_by_authority = any", "control_plane", "trusted_worker",
	},
	"control_calphad_validation_metadata_check": {
		"c true", "jsonb_typeof(metadata) = 'object'",
	},
	"control_calphad_validation_registration_status_check": {
		"c true", "operation = 'registration'", "status = 'pending'",
	},
	"control_calphad_validation_registration_authority_check": {
		"c true", "operation = 'registration'", "created_by_authority = 'control_plane'",
	},
	"control_calphad_validation_input_operation_check": {
		"c true", "status <> 'input_validated'", "operation = 'inspect'",
	},
	"control_calphad_validation_equilibrium_operation_check": {
		"c true", "status <> 'equilibrium_completed'", "operation = 'equilibrium'",
	},
	"control_calphad_validation_scheil_operation_check": {
		"c true", "status <> 'scheil_completed'", "operation = 'scheil'",
	},
	"control_calphad_validation_evidence_tuple_check": {
		"c true", "evidence_path is null", "evidence_sha256 is null", "evidence_size_bytes is null",
		"evidence_path is not null", "evidence_sha256 is not null", "evidence_size_bytes is not null",
	},
	"control_calphad_validation_retained_evidence_check": {
		"c true", "operation = 'registration'", "evidence_path is not null",
	},
	"control_calphad_validation_failure_tuple_check": {
		"c true", "failed", "timeout", "unsupported",
		"failure_domain is not null", "failure_stage is not null", "failure_code is not null",
		"input", "scientific", "platform", "parse", "solver", "result_validation", "sandbox_runtime",
		"calphad_parse_failed", "calphad_parse_timeout", "calphad_parse_unsupported",
		"calphad_solver_failed", "calphad_solver_timeout", "calphad_solver_unsupported",
		"calphad_result_invalid", "calphad_runtime_internal_failure",
		"calphad_sandbox_failed", "calphad_sandbox_timeout", "scheil",
	},
	"control_calphad_validation_evidence_path_check": {
		"c true", "/outputs/calphad/", "inspection", "operation", "evidence_sha256",
	},
	"control_calphad_validation_runtime_binding_check": {
		"c true", "operation = 'registration'", "runtime_image_id is null", "pycalphad_version is null",
		"operation <> 'registration'", "runtime_image_id is not null", "pycalphad_version is not null",
		"run_id is not null", "char_length(btrim(pycalphad_version)) >= 1",
		"char_length(btrim(pycalphad_version)) <= 128", "char_length(btrim(run_id)) >= 1",
		"char_length(btrim(run_id)) <= 512",
	},
	"control_calphad_validation_worker_identity_check": {
		"c", "operation = 'registration'", "database_inventory_sha256 is null", "request_sha256 is null",
		"evidence_contract_version is null", "operation <> 'registration'",
		"database_inventory_sha256 is not null", "operation = 'inspect'", "failed", "timeout", "unsupported",
		"request_sha256 is not null",
		"ultra.calphad.retained-evidence.v2",
	},
}

var requiredCalphadIndexFingerprints = map[string][]string{
	"control_calphad_revisions_owner_created_idx": {
		"control_calphad_revisions using btree (owner_user_id, owner_org_id, created_at desc)",
	},
	"control_calphad_revisions_parent_idx": {
		"control_calphad_revisions using btree (parent_revision_id)", "where (parent_revision_id is not null)",
	},
	"control_calphad_validation_revision_created_idx": {
		"control_calphad_validation_events using btree (revision_id, created_at desc, validation_id desc)",
	},
	"control_calphad_validation_run_idx": {
		"control_calphad_validation_events using btree (run_id)", "where (run_id is not null)",
	},
	"control_calphad_validation_evidence_uidx": {
		"create unique index", "control_calphad_validation_events using btree (revision_id, run_id, operation, evidence_sha256)",
		"run_id is not null", "evidence_sha256 is not null",
	},
	"control_calphad_validation_request_idx": {
		"control_calphad_validation_events using btree (revision_id, run_id, operation, request_sha256, created_at desc)",
		"request_sha256 is not null",
	},
	"control_calphad_validation_inspection_lineage_idx": {
		"control_calphad_validation_events using btree (revision_id, run_id, database_format, runtime_image_id, database_inventory_sha256, evidence_sha256)",
		"operation = 'inspect'", "status = 'input_validated'",
	},
}

var requiredCalphadFunctionFingerprints = map[string][]string{
	"ultra_create_calphad_revision_v1": {
		"security_definer=true", "search_path=pg_catalog", "set search_path to 'pg_catalog'",
		"public.control_resources", "public.control_calphad_input_blobs",
		"public.control_calphad_revisions", "public.control_calphad_validation_events",
		"public.control_calphad_tenant_capacity", "for update", "gen_random_uuid()",
		"calphad_tenant_capacity_exceeded", "owner_declaration", "database_format",
		"on conflict (input_sha256) do nothing",
	},
	"ultra_append_calphad_validation_v1": {
		"security_definer=true", "search_path=pg_catalog", "set search_path to 'pg_catalog'",
		"public.control_runs", "public.control_run_leases", "public.control_resources",
		"public.control_calphad_revisions", "public.control_calphad_input_blobs",
		"public.control_calphad_evidence_blobs", "public.control_calphad_validation_events",
		"public.control_calphad_tenant_capacity", "for update", "gen_random_uuid()",
		"calphad_run_lease_invalid", "calphad_runtime_policy_invalid",
		"calphad_selected_resource_invalid", "calphad_tenant_capacity_exceeded",
		"calphad_inspection_required", "ultra.calphad.retained-evidence.v2",
		"ultra.calphad.tool-evidence.v3", "ultra.calphad.failure-evidence.v1",
		"caller_code_accepted", "caller_models_or_solver_options_accepted",
		"catalog_metadata_updated", "immutable_per_run_evidence", "?& array",
		"lease_record.lease_expires_at <= clock_timestamp()", "scheil_completed", "p_operation = 'scheil'",
	},
	"ultra_validate_calphad_pressure_binding": {
		"security_definer=false", "search_path=pg_catalog", "set search_path to 'pg_catalog'",
		"public.control_calphad_revisions", "assessment_pressure_min_pa",
		"assessment_pressure_max_pa", "assessment_pressure_limits_Pa",
		"database_format", "owner_declaration", "ultra.calphad.owner-declaration.v1",
		"jsonb_build_array", "calphad_pressure_binding_invalid",
	},
	"ultra_validate_calphad_input_retention": {
		"security_definer=false", "search_path=pg_catalog", "set search_path to 'pg_catalog'",
		"public.control_calphad_revisions", "public.control_calphad_input_blobs",
		"blob.input_sha256 = revision.sha256", "blob.input_size_bytes = revision.size_bytes",
		"revision.database_format = new.database_format",
		"encode(sha256(blob.payload), 'hex') = revision.sha256", "calphad_input_retention_required",
	},
	"ultra_validate_calphad_revision_parent": {
		"security_definer=false", "search_path=pg_catalog", "set search_path to 'pg_catalog'", "public.control_calphad_revisions",
		"parent.resource_id <> new.resource_id", "parent.owner_user_id = new.owner_user_id",
	},
	"ultra_validate_calphad_validation_run_authority": {
		"security_definer=false", "search_path=pg_catalog", "set search_path to 'pg_catalog'", "public.control_runs", "public.control_run_leases",
		"public.control_calphad_revisions", "clock_timestamp()", "calphad_run_lease_invalid", "errcode = '28000'",
		"calphad_runtime_policy", "ultra.calphad.runtime-policy.v2", "pycalphad_version', '0.11.2'",
		"network', 'none'", "no_new_privileges', true", "read_only_root_filesystem', true",
		"cap_drop_all', true", "cpus_at_most', 8", "memory_bytes_at_most', 34359738368",
		"pids_at_most', 4096",
	},
	"ultra_validate_calphad_equilibrium_inspection_lineage": {
		"security_definer=false", "search_path=pg_catalog", "set search_path to 'pg_catalog'", "public.control_calphad_validation_events",
		"public.control_calphad_evidence_blobs", "calphad_inspection_required",
		"inspection.revision_id = new.revision_id",
		"inspection.run_id = new.run_id", "inspection.runtime_image_id = new.runtime_image_id",
		"inspection.database_format = new.database_format",
		"inspection.evidence_sha256 = new.inspection_evidence_sha256",
		"inspection.database_inventory_sha256 = new.database_inventory_sha256",
		"inspection.evidence_contract_version = 'ultra.calphad.retained-evidence.v2'",
		"encode(sha256(blob.payload), 'hex') = inspection.evidence_sha256",
		"new.operation in ('equilibrium', 'scheil')",
	},
	"ultra_reject_calphad_ledger_mutation": {
		"security_definer=false", "search_path=pg_catalog", "set search_path to 'pg_catalog'", "calphad governance ledger is append-only",
	},
}

var requiredCalphadTriggerFingerprints = map[string][]string{
	"control_calphad_input_blobs_append_only":        {"before delete or update", "control_calphad_input_blobs", "ultra_reject_calphad_ledger_mutation"},
	"control_calphad_input_blobs_no_truncate":        {"before truncate", "control_calphad_input_blobs", "ultra_reject_calphad_ledger_mutation"},
	"control_calphad_revisions_parent_guard":         {"before insert", "control_calphad_revisions", "ultra_validate_calphad_revision_parent"},
	"control_calphad_validation_input_retention":     {"before insert", "control_calphad_validation_events", "ultra_validate_calphad_input_retention"},
	"control_calphad_validation_run_authority":       {"before insert", "control_calphad_validation_events", "ultra_validate_calphad_validation_run_authority"},
	"control_calphad_validation_z_pressure_binding":  {"before insert", "control_calphad_validation_events", "ultra_validate_calphad_pressure_binding"},
	"control_calphad_equilibrium_inspection_lineage": {"before insert", "control_calphad_validation_events", "ultra_validate_calphad_equilibrium_inspection_lineage"},
	"control_calphad_revisions_append_only":          {"before delete or update", "control_calphad_revisions", "ultra_reject_calphad_ledger_mutation"},
	"control_calphad_validation_append_only":         {"before delete or update", "control_calphad_validation_events", "ultra_reject_calphad_ledger_mutation"},
	"control_calphad_evidence_blobs_append_only":     {"before delete or update", "control_calphad_evidence_blobs", "ultra_reject_calphad_ledger_mutation"},
	"control_calphad_revisions_no_truncate":          {"before truncate", "control_calphad_revisions", "ultra_reject_calphad_ledger_mutation"},
	"control_calphad_validation_no_truncate":         {"before truncate", "control_calphad_validation_events", "ultra_reject_calphad_ledger_mutation"},
	"control_calphad_evidence_blobs_no_truncate":     {"before truncate", "control_calphad_evidence_blobs", "ultra_reject_calphad_ledger_mutation"},
}

func VerifyPostgresSchema(ctx context.Context, db schemaQuerier) error {
	var presentTables []string
	err := db.QueryRow(ctx, `
SELECT COALESCE(array_agg(table_name::text ORDER BY table_name), ARRAY[]::text[])
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name = ANY($1::text[])
`, requiredPostgresControlTables).Scan(&presentTables)
	if err != nil {
		return fmt.Errorf("verify postgres schema: %w", err)
	}

	present := map[string]struct{}{}
	for _, table := range presentTables {
		present[table] = struct{}{}
	}
	missing := make([]string, 0)
	for _, table := range requiredPostgresControlTables {
		if _, ok := present[table]; !ok {
			missing = append(missing, table)
		}
	}
	if len(missing) > 0 {
		slices.Sort(missing)
		return fmt.Errorf("postgres control schema is not ready; apply migrations before starting: missing tables %s", strings.Join(missing, ", "))
	}
	return verifyCalphadPostgresSchema(ctx, db)
}

func verifyCalphadPostgresSchema(ctx context.Context, db schemaQuerier) error {
	var catalog calphadSchemaCatalog
	err := db.QueryRow(ctx, `
/* calphad_schema_catalog_v1 */
SELECT
  ARRAY(
    SELECT format('%s.%s.%s:%s:%s', table_name, ordinal_position, column_name, udt_name, is_nullable)
    FROM information_schema.columns
    WHERE table_schema='public' AND table_name=ANY($1::text[])
    ORDER BY table_name, ordinal_position
  ),
  ARRAY(
    SELECT constraint_record
    FROM (
      SELECT constraint_record.conname || E'\n' || constraint_record.contype::text || E'\n' ||
             constraint_record.convalidated::text || E'\n' ||
             pg_get_constraintdef(constraint_record.oid, true) AS constraint_record
      FROM pg_constraint constraint_record
      JOIN pg_class relation ON relation.oid=constraint_record.conrelid
      JOIN pg_namespace namespace ON namespace.oid=relation.relnamespace
      WHERE namespace.nspname='public'
        AND constraint_record.conname=ANY($2::text[])
      ORDER BY constraint_record.conname
    ) records
  ),
  ARRAY(
    SELECT indexname || E'\n' || indexdef
    FROM pg_indexes
    WHERE schemaname='public' AND indexname=ANY($3::text[])
    ORDER BY indexname
  ),
  ARRAY(
    SELECT routine.proname || E'\nsecurity_definer=' || routine.prosecdef::text || E'\n' ||
           COALESCE(array_to_string(routine.proconfig, E'\n'), '') ||
           E'\n' || pg_get_functiondef(routine.oid)
    FROM pg_proc routine
    JOIN pg_namespace namespace ON namespace.oid=routine.pronamespace
    WHERE namespace.nspname='public' AND routine.proname=ANY($4::text[])
    ORDER BY routine.proname
  ),
  ARRAY(
    SELECT trigger_record.tgname || E'\n' || pg_get_triggerdef(trigger_record.oid, true)
    FROM pg_trigger trigger_record
    JOIN pg_class relation ON relation.oid=trigger_record.tgrelid
    JOIN pg_namespace namespace ON namespace.oid=relation.relnamespace
    WHERE namespace.nspname='public' AND NOT trigger_record.tgisinternal
      AND relation.relname=ANY($1::text[])
    ORDER BY trigger_record.tgname
  ),
  (
    SELECT count(*)=2 AND count(*) FILTER (
      WHERE routine.oid=to_regprocedure($5) OR routine.oid=to_regprocedure($6)
    )=2
    FROM pg_proc routine
    JOIN pg_namespace namespace ON namespace.oid=routine.pronamespace
    WHERE namespace.nspname='public'
      AND routine.proname IN ('ultra_create_calphad_revision_v1',
                              'ultra_append_calphad_validation_v1')
  ),
  NOT EXISTS (
    SELECT 1
    FROM public.control_calphad_tenant_capacity capacity
    WHERE capacity.owner_user_id <> btrim(capacity.owner_user_id)
       OR capacity.owner_org_id <> COALESCE(NULLIF(btrim(capacity.owner_org_id), ''), '')
       OR capacity.retained_input_bytes <> COALESCE((
         SELECT sum(revision.size_bytes)::bigint
         FROM public.control_calphad_revisions revision
         WHERE btrim(revision.owner_user_id)=capacity.owner_user_id
           AND COALESCE(NULLIF(btrim(revision.owner_org_id), ''), '')=capacity.owner_org_id
       ), 0)
       OR capacity.retained_evidence_bytes <> COALESCE((
         SELECT sum(validation.evidence_size_bytes)::bigint
         FROM public.control_calphad_validation_events validation
         JOIN public.control_calphad_revisions revision
           ON revision.revision_id=validation.revision_id
         WHERE btrim(revision.owner_user_id)=capacity.owner_user_id
           AND COALESCE(NULLIF(btrim(revision.owner_org_id), ''), '')=capacity.owner_org_id
       ), 0)
       OR capacity.validation_events <> COALESCE((
         SELECT count(*)::bigint
         FROM public.control_calphad_validation_events validation
         JOIN public.control_calphad_revisions revision
           ON revision.revision_id=validation.revision_id
         WHERE btrim(revision.owner_user_id)=capacity.owner_user_id
           AND COALESCE(NULLIF(btrim(revision.owner_org_id), ''), '')=capacity.owner_org_id
       ), 0)
  ) AND NOT EXISTS (
    SELECT 1
    FROM public.control_calphad_revisions revision
    WHERE NOT EXISTS (
      SELECT 1 FROM public.control_calphad_tenant_capacity capacity
      WHERE capacity.owner_user_id=btrim(revision.owner_user_id)
        AND capacity.owner_org_id=COALESCE(NULLIF(btrim(revision.owner_org_id), ''), '')
    )
  )
`, requiredCalphadTables, sortedMapKeys(requiredCalphadConstraintFingerprints),
		sortedMapKeys(requiredCalphadIndexFingerprints), sortedMapKeys(requiredCalphadFunctionFingerprints),
		calphadCreateRevisionFunctionSignature, calphadAppendValidationFunctionSignature).Scan(
		&catalog.Columns, &catalog.Constraints, &catalog.Indexes, &catalog.Functions,
		&catalog.Triggers, &catalog.WriterFunctionsExact, &catalog.CapacityAccountingValid,
	)
	if err != nil {
		return fmt.Errorf("verify postgres CALPHAD schema catalog: %w", err)
	}
	issues := validateCalphadSchemaCatalog(catalog)
	if len(issues) > 0 {
		return fmt.Errorf(
			"postgres CALPHAD schema fingerprint %s is not ready; apply migrations before starting: %s",
			calphadSchemaFingerprint(catalog), strings.Join(issues, "; "),
		)
	}
	return nil
}
func sortedMapKeys(values map[string][]string) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	return keys
}

func normalizedSchemaDefinition(value string) string {
	return strings.Join(strings.Fields(strings.ToLower(value)), " ")
}

func schemaObjectDefinitions(records []string) map[string]string {
	out := make(map[string]string, len(records))
	for _, record := range records {
		parts := strings.SplitN(record, "\n", 2)
		if len(parts) != 2 || strings.TrimSpace(parts[0]) == "" {
			continue
		}
		out[strings.TrimSpace(parts[0])] = normalizedSchemaDefinition(parts[1])
	}
	return out
}

func validateSchemaObjectFingerprints(
	label string,
	records []string,
	required map[string][]string,
	exactNames bool,
) []string {
	definitions := schemaObjectDefinitions(records)
	issues := make([]string, 0)
	if exactNames {
		actualNames := make([]string, 0, len(definitions))
		for name := range definitions {
			actualNames = append(actualNames, name)
		}
		slices.Sort(actualNames)
		expectedNames := sortedMapKeys(required)
		if !slices.Equal(actualNames, expectedNames) {
			issues = append(issues, fmt.Sprintf("%s names=%v want=%v", label, actualNames, expectedNames))
		}
	}
	for name, fragments := range required {
		definition, ok := definitions[name]
		if !ok {
			issues = append(issues, fmt.Sprintf("missing %s %s", label, name))
			continue
		}
		for _, fragment := range fragments {
			if !strings.Contains(definition, normalizedSchemaDefinition(fragment)) {
				issues = append(issues, fmt.Sprintf("%s %s changed", label, name))
				break
			}
		}
	}
	return issues
}

func validateCalphadSchemaCatalog(catalog calphadSchemaCatalog) []string {
	canonicalColumns := func(values []string) []string {
		out := make([]string, 0, len(values))
		for _, value := range values {
			parts := strings.SplitN(value, ".", 3)
			if len(parts) == 3 {
				value = parts[0] + "." + parts[2]
			}
			out = append(out, value)
		}
		return out
	}
	// Column order differs on safely upgraded databases because immutable
	// binding fields were added after the original draft. Exact names, types,
	// and nullability are the serving contract; physical ordinal is not.
	actualColumns := canonicalColumns(catalog.Columns)
	expectedColumns := canonicalColumns(requiredCalphadColumns)
	slices.Sort(actualColumns)
	slices.Sort(expectedColumns)
	issues := make([]string, 0)
	if !catalog.WriterFunctionsExact {
		issues = append(issues, "CALPHAD writer function catalog contains a missing or unexpected overload")
	}
	if !catalog.CapacityAccountingValid {
		issues = append(issues, "CALPHAD tenant capacity accounting differs from the immutable ledger")
	}
	if !slices.Equal(actualColumns, expectedColumns) {
		issues = append(issues, "exact CALPHAD columns/types/nullability changed")
	}
	issues = append(issues, validateSchemaObjectFingerprints(
		"constraint", catalog.Constraints, requiredCalphadConstraintFingerprints, true,
	)...)
	issues = append(issues, validateSchemaObjectFingerprints(
		"index", catalog.Indexes, requiredCalphadIndexFingerprints, true,
	)...)
	issues = append(issues, validateSchemaObjectFingerprints(
		"function", catalog.Functions, requiredCalphadFunctionFingerprints, true,
	)...)
	issues = append(issues, validateSchemaObjectFingerprints(
		"trigger", catalog.Triggers, requiredCalphadTriggerFingerprints, true,
	)...)
	slices.Sort(issues)
	return issues
}

func calphadSchemaFingerprint(catalog calphadSchemaCatalog) string {
	sections := [][]string{
		catalog.Columns, catalog.Constraints, catalog.Indexes, catalog.Functions, catalog.Triggers,
	}
	digest := sha256.New()
	for _, section := range sections {
		values := append([]string(nil), section...)
		slices.Sort(values)
		for _, value := range values {
			_, _ = digest.Write([]byte(normalizedSchemaDefinition(value)))
			_, _ = digest.Write([]byte{0})
		}
		_, _ = digest.Write([]byte{0xff})
	}
	_, _ = fmt.Fprintf(digest, "writer_functions_exact=%t\x00capacity_accounting_valid=%t",
		catalog.WriterFunctionsExact, catalog.CapacityAccountingValid)
	return hex.EncodeToString(digest.Sum(nil))
}

func InspectCalphadServingRole(ctx context.Context, db schemaQuerier) (CalphadServingRoleStatus, error) {
	var status CalphadServingRoleStatus
	err := db.QueryRow(ctx, `
/* calphad_serving_role_v1 */
WITH owner_roles AS (
  SELECT DISTINCT pg_get_userbyid(owner_oid)::text AS owner_role
  FROM (
    SELECT relation.relowner AS owner_oid
    FROM pg_class relation
    JOIN pg_namespace namespace ON namespace.oid=relation.relnamespace
    WHERE namespace.nspname='public' AND relation.relname=ANY($1::text[])
    UNION
    SELECT routine.proowner AS owner_oid
    FROM pg_proc routine
    JOIN pg_namespace namespace ON namespace.oid=routine.pronamespace
    WHERE namespace.nspname='public' AND routine.proname=ANY($2::text[])
  ) owners
)
SELECT current_user, role_record.rolsuper, role_record.rolcreaterole,
  role_record.rolcreatedb, role_record.rolreplication, role_record.rolbypassrls,
  ARRAY(
    SELECT relation.relname::text
    FROM pg_class relation
    JOIN pg_namespace namespace ON namespace.oid=relation.relnamespace
    WHERE namespace.nspname='public' AND relation.relname=ANY($1::text[])
      AND pg_get_userbyid(relation.relowner)=current_user
    ORDER BY relation.relname
  ),
  ARRAY(
    SELECT routine.proname::text
    FROM pg_proc routine
    JOIN pg_namespace namespace ON namespace.oid=routine.pronamespace
    WHERE namespace.nspname='public' AND routine.proname=ANY($2::text[])
      AND pg_get_userbyid(routine.proowner)=current_user
    ORDER BY routine.proname
  ),
  ARRAY(
    SELECT owner_role
    FROM owner_roles
    WHERE owner_role <> current_user
    ORDER BY owner_role
  ),
  ARRAY(
    SELECT candidate.rolname::text
    FROM pg_roles candidate
    WHERE candidate.rolname <> current_user AND (
      pg_has_role(current_user, candidate.oid, 'MEMBER') OR
      pg_has_role(current_user, candidate.oid, 'SET')
    )
    ORDER BY candidate.rolname
  ),
  EXISTS (
    SELECT 1 FROM owner_roles
    WHERE owner_role <> current_user AND (
      pg_has_role(current_user, owner_role, 'MEMBER') OR
      pg_has_role(current_user, owner_role, 'SET')
    )
  ),
  COALESCE((
    SELECT pg_get_userbyid(namespace.nspowner)::text
    FROM pg_namespace namespace
    WHERE namespace.nspname='public'
  ), ''),
  EXISTS (
    SELECT 1
    FROM pg_namespace namespace
    WHERE namespace.nspname='public' AND (
      pg_get_userbyid(namespace.nspowner)=current_user OR
      pg_has_role(current_user, pg_get_userbyid(namespace.nspowner), 'MEMBER') OR
      pg_has_role(current_user, pg_get_userbyid(namespace.nspowner), 'SET')
    )
  ),
  has_schema_privilege(current_user, 'public', 'CREATE'),
  (SELECT bool_and(has_table_privilege(current_user, 'public.' || table_name, 'SELECT'))
   FROM unnest($1::text[]) table_name),
  (SELECT bool_and(has_table_privilege(current_user, 'public.' || table_name, 'INSERT'))
   FROM unnest($1::text[]) table_name),
  (SELECT bool_or(has_table_privilege(current_user, 'public.' || table_name, 'INSERT'))
   FROM unnest($1::text[]) table_name),
  (SELECT bool_or(
     has_table_privilege(current_user, 'public.' || table_name, 'UPDATE') OR
     has_any_column_privilege(current_user, 'public.' || table_name, 'UPDATE') OR
     has_table_privilege(current_user, 'public.' || table_name, 'DELETE') OR
     has_table_privilege(current_user, 'public.' || table_name, 'TRUNCATE')
   ) FROM unnest($1::text[]) table_name),
  has_function_privilege(current_user, $3::text, 'EXECUTE'),
  has_function_privilege(current_user, $4::text, 'EXECUTE'),
  (
    SELECT count(*)=2 AND count(*) FILTER (
      WHERE routine.oid=to_regprocedure($3) OR routine.oid=to_regprocedure($4)
    )=2
    FROM pg_proc routine
    JOIN pg_namespace namespace ON namespace.oid=routine.pronamespace
    WHERE namespace.nspname='public'
      AND routine.proname IN ('ultra_create_calphad_revision_v1',
                              'ultra_append_calphad_validation_v1')
  ),
  EXISTS (
    SELECT 1
    FROM pg_proc routine
    JOIN pg_namespace namespace ON namespace.oid=routine.pronamespace
    WHERE namespace.nspname='public'
      AND routine.proname IN ('ultra_create_calphad_revision_v1',
                              'ultra_append_calphad_validation_v1')
      AND routine.oid IS DISTINCT FROM to_regprocedure($3)
      AND routine.oid IS DISTINCT FROM to_regprocedure($4)
      AND has_function_privilege(current_user, routine.oid, 'EXECUTE')
  ),
  EXISTS (
    SELECT 1
    FROM pg_proc routine
    JOIN pg_namespace namespace ON namespace.oid=routine.pronamespace
    WHERE namespace.nspname='public' AND routine.proname=ANY($5::text[])
      AND has_function_privilege(current_user, routine.oid, 'EXECUTE')
  ),
  EXISTS (
    SELECT 1
    FROM pg_proc routine
    JOIN pg_namespace namespace ON namespace.oid=routine.pronamespace
    CROSS JOIN LATERAL aclexplode(COALESCE(routine.proacl, acldefault('f', routine.proowner))) acl
    WHERE namespace.nspname='public' AND routine.proname=ANY($2::text[])
      AND acl.grantee=0 AND acl.privilege_type='EXECUTE'
  ),
  ARRAY(
    SELECT DISTINCT CASE WHEN acl.grantee=0 THEN 'PUBLIC'::text
                         ELSE pg_get_userbyid(acl.grantee)::text END
    FROM pg_class relation
    JOIN pg_namespace namespace ON namespace.oid=relation.relnamespace
    CROSS JOIN LATERAL aclexplode(
      COALESCE(relation.relacl, acldefault('r', relation.relowner))
    ) acl
    WHERE namespace.nspname='public' AND relation.relname=ANY($1::text[])
      AND (acl.grantee=0 OR
           (acl.grantee <> relation.relowner AND pg_get_userbyid(acl.grantee) <> current_user))
    ORDER BY 1
  ),
  ARRAY(
    SELECT DISTINCT CASE WHEN acl.grantee=0 THEN 'PUBLIC'::text
                         ELSE pg_get_userbyid(acl.grantee)::text END
    FROM pg_proc routine
    JOIN pg_namespace namespace ON namespace.oid=routine.pronamespace
    CROSS JOIN LATERAL aclexplode(
      COALESCE(routine.proacl, acldefault('f', routine.proowner))
    ) acl
    WHERE namespace.nspname='public' AND routine.proname=ANY($2::text[])
      AND (acl.grantee=0 OR
           (acl.grantee <> routine.proowner AND pg_get_userbyid(acl.grantee) <> current_user))
    ORDER BY 1
  )
FROM pg_roles role_record
WHERE role_record.rolname=current_user
`, requiredCalphadTables, sortedMapKeys(requiredCalphadFunctionFingerprints),
		calphadCreateRevisionFunctionSignature, calphadAppendValidationFunctionSignature,
		requiredCalphadInternalFunctions).Scan(
		&status.Role, &status.Superuser, &status.CreateRole, &status.CreateDB,
		&status.Replication, &status.BypassRLS,
		&status.OwnedTables, &status.OwnedFunctions, &status.OwnerRoles, &status.ReachableRoles,
		&status.OwnerRoleReachable, &status.PublicSchemaOwner, &status.PublicOwnerReachable,
		&status.CanCreatePublicSchema,
		&status.CanSelectAll, &status.CanInsertAll, &status.CanInsertAny, &status.CanMutateCalphad,
		&status.CanExecuteCreateRevision, &status.CanExecuteAppendValidation,
		&status.WriterFunctionsExact, &status.CanExecuteUnexpectedWriter,
		&status.CanExecuteInternal, &status.PublicCanExecute,
		&status.UnexpectedTableACLGrantees, &status.UnexpectedFuncACLGrantees,
	)
	if err != nil {
		return CalphadServingRoleStatus{}, fmt.Errorf("inspect postgres CALPHAD serving role: %w", err)
	}
	return status, nil
}

func VerifyCalphadServingRole(ctx context.Context, db schemaQuerier) error {
	status, err := InspectCalphadServingRole(ctx, db)
	if err != nil {
		return err
	}
	if status.Role == "" || status.Superuser || status.CreateRole || status.CreateDB ||
		status.Replication || status.BypassRLS || len(status.ReachableRoles) != 0 ||
		len(status.OwnedTables) != 0 ||
		len(status.OwnedFunctions) != 0 || !status.CanSelectAll || status.CanInsertAll || status.CanInsertAny ||
		len(status.OwnerRoles) != 1 || status.OwnerRoleReachable || status.PublicSchemaOwner == "" ||
		status.PublicOwnerReachable || status.CanCreatePublicSchema ||
		status.CanMutateCalphad || !status.CanExecuteCreateRevision ||
		!status.CanExecuteAppendValidation || !status.WriterFunctionsExact ||
		status.CanExecuteUnexpectedWriter || status.CanExecuteInternal || status.PublicCanExecute ||
		len(status.UnexpectedTableACLGrantees) != 0 || len(status.UnexpectedFuncACLGrantees) != 0 {
		return fmt.Errorf(
			"postgres CALPHAD serving role is unsafe: role=%q superuser=%t createrole=%t createdb=%t replication=%t bypassrls=%t owned_tables=%v owned_functions=%v owner_roles=%v reachable_roles=%v owner_reachable=%t public_owner=%q public_owner_reachable=%t create_public=%t select=%t insert_all=%t insert_any=%t mutate=%t execute_create=%t execute_append=%t writer_catalog_exact=%t execute_unexpected_writer=%t execute_internal=%t public_execute=%t unexpected_table_acl_grantees=%v unexpected_function_acl_grantees=%v",
			status.Role, status.Superuser, status.CreateRole, status.CreateDB,
			status.Replication, status.BypassRLS,
			status.OwnedTables, status.OwnedFunctions, status.OwnerRoles, status.ReachableRoles,
			status.OwnerRoleReachable, status.PublicSchemaOwner, status.PublicOwnerReachable,
			status.CanCreatePublicSchema,
			status.CanSelectAll, status.CanInsertAll, status.CanInsertAny, status.CanMutateCalphad,
			status.CanExecuteCreateRevision, status.CanExecuteAppendValidation,
			status.WriterFunctionsExact, status.CanExecuteUnexpectedWriter,
			status.CanExecuteInternal, status.PublicCanExecute,
			status.UnexpectedTableACLGrantees, status.UnexpectedFuncACLGrantees,
		)
	}
	return nil
}
