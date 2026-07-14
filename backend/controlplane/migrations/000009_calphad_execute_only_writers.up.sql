-- CALPHAD writes are admitted only through two fixed migration-owner
-- SECURITY DEFINER functions. The serving role receives their exact EXECUTE
-- signatures from GrantPostgresServingPrivileges and no raw table INSERT.

CREATE TABLE IF NOT EXISTS control_calphad_tenant_capacity (
  owner_user_id text NOT NULL,
  owner_org_id text NOT NULL DEFAULT '',
  max_retained_bytes bigint NOT NULL DEFAULT 4294967296,
  max_validation_events bigint NOT NULL DEFAULT 100000,
  retained_input_bytes bigint NOT NULL DEFAULT 0,
  retained_evidence_bytes bigint NOT NULL DEFAULT 0,
  validation_events bigint NOT NULL DEFAULT 0,
  updated_at timestamptz NOT NULL,
  PRIMARY KEY (owner_user_id, owner_org_id),
  CONSTRAINT control_calphad_tenant_capacity_limits_check
    CHECK (max_retained_bytes > 0 AND max_validation_events > 0),
  CONSTRAINT control_calphad_tenant_capacity_counters_check
    CHECK (retained_input_bytes >= 0 AND retained_evidence_bytes >= 0 AND
           validation_events >= 0 AND
           retained_input_bytes::numeric + retained_evidence_bytes::numeric <= max_retained_bytes::numeric AND
           validation_events <= max_validation_events)
);

DO $$
DECLARE
  inconsistent_count bigint;
BEGIN
  SELECT count(*) INTO inconsistent_count
  FROM public.control_calphad_revisions revision
  LEFT JOIN public.control_calphad_input_blobs blob
    ON blob.input_sha256 = revision.sha256
   AND blob.input_size_bytes = revision.size_bytes
   AND octet_length(blob.payload) = revision.size_bytes
   AND encode(sha256(blob.payload), 'hex') = revision.sha256
  WHERE blob.input_sha256 IS NULL;
  IF inconsistent_count <> 0 THEN
    RAISE EXCEPTION 'CALPHAD_INPUT_RETENTION_REQUIRED: capacity backfill found % inconsistent revisions', inconsistent_count
      USING ERRCODE = '23514';
  END IF;

  SELECT count(*) INTO inconsistent_count
  FROM public.control_calphad_validation_events validation
  LEFT JOIN public.control_calphad_evidence_blobs blob
    ON blob.evidence_sha256 = validation.evidence_sha256
   AND blob.evidence_size_bytes = validation.evidence_size_bytes
   AND octet_length(blob.payload) = validation.evidence_size_bytes
   AND encode(sha256(blob.payload), 'hex') = validation.evidence_sha256
  WHERE validation.operation <> 'registration' AND blob.evidence_sha256 IS NULL;
  IF inconsistent_count <> 0 THEN
    RAISE EXCEPTION 'CALPHAD_EVIDENCE_RETENTION_REQUIRED: capacity backfill found % inconsistent events', inconsistent_count
      USING ERRCODE = '23514';
  END IF;
END;
$$;

CREATE OR REPLACE FUNCTION public.ultra_append_calphad_validation_v1(
  p_resource_id text,
  p_owner_user_id text,
  p_owner_org_id text,
  p_database_sha256 text,
  p_database_size_bytes bigint,
  p_database_format text,
  p_owner_declaration jsonb,
  p_assessment_pressure_min_pa double precision,
  p_assessment_pressure_max_pa double precision,
  p_database_inventory_sha256 text,
  p_request_sha256 text,
  p_status text,
  p_operation text,
  p_failure_domain text,
  p_failure_stage text,
  p_failure_code text,
  p_evidence_path text,
  p_evidence_sha256 text,
  p_evidence_size_bytes bigint,
  p_evidence_payload bytea,
  p_runtime_image_id text,
  p_pycalphad_version text,
  p_run_id text,
  p_inspection_evidence_sha256 text,
  p_lease_worker_id text,
  p_lease_token text,
  p_metadata jsonb
)
RETURNS TABLE (
  validation_id text,
  revision_id text,
  resource_id text,
  database_sha256 text,
  database_size_bytes bigint,
  database_format text,
  assessment_pressure_min_pa double precision,
  assessment_pressure_max_pa double precision,
  database_inventory_sha256 text,
  request_sha256 text,
  status text,
  operation text,
  failure_domain text,
  failure_stage text,
  failure_code text,
  evidence_path text,
  evidence_sha256 text,
  evidence_size_bytes bigint,
  runtime_image_id text,
  pycalphad_version text,
  run_id text,
  inspection_evidence_sha256 text,
  evidence_contract_version text,
  created_by_authority text,
  created_at timestamptz,
  metadata jsonb,
  evidence_blob_retained boolean
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog
AS $$
DECLARE
  run_record public.control_runs%ROWTYPE;
  lease_record public.control_run_leases%ROWTYPE;
  resource_record public.control_resources%ROWTYPE;
  revision_record public.control_calphad_revisions%ROWTYPE;
  existing_validation public.control_calphad_validation_events%ROWTYPE;
  inserted_validation public.control_calphad_validation_events%ROWTYPE;
  capacity_record public.control_calphad_tenant_capacity%ROWTYPE;
  normalized_org text;
  derived_format text;
  calphad_metadata jsonb;
  declared_temperature jsonb;
  expected_declaration jsonb;
  evidence_json jsonb;
  validation_metadata jsonb;
  stored_size bigint;
  stored_payload bytea;
  selected_file_count bigint;
  descriptor_candidate_count bigint;
  descriptor_exact_count bigint;
  capacity_updated bigint;
  new_validation_id text;
  created_timestamp timestamptz;
BEGIN
  p_resource_id := btrim(p_resource_id);
  p_owner_user_id := btrim(p_owner_user_id);
  normalized_org := COALESCE(NULLIF(btrim(p_owner_org_id), ''), '');
  p_database_sha256 := lower(btrim(p_database_sha256));
  p_database_format := btrim(p_database_format);
  p_database_inventory_sha256 := NULLIF(lower(btrim(p_database_inventory_sha256)), '');
  p_request_sha256 := lower(btrim(p_request_sha256));
  p_status := btrim(p_status);
  p_operation := btrim(p_operation);
  p_failure_domain := NULLIF(btrim(p_failure_domain), '');
  p_failure_stage := NULLIF(btrim(p_failure_stage), '');
  p_failure_code := NULLIF(btrim(p_failure_code), '');
  p_evidence_path := btrim(p_evidence_path);
  p_evidence_sha256 := lower(btrim(p_evidence_sha256));
  p_runtime_image_id := lower(btrim(p_runtime_image_id));
  p_pycalphad_version := btrim(p_pycalphad_version);
  p_run_id := btrim(p_run_id);
  p_inspection_evidence_sha256 := NULLIF(lower(btrim(p_inspection_evidence_sha256)), '');
  p_lease_worker_id := btrim(p_lease_worker_id);

  IF p_resource_id = '' OR p_owner_user_id = '' OR p_run_id = '' OR
     p_lease_worker_id = '' OR p_lease_token IS NULL OR btrim(p_lease_token) = '' OR
     p_database_sha256 !~ '^[0-9a-f]{64}$' OR p_database_size_bytes <= 0 OR
     p_database_format NOT IN ('tdb', 'dat') OR
     p_assessment_pressure_min_pa < 1e-9 OR
     p_assessment_pressure_max_pa > 1e12 OR
     p_assessment_pressure_min_pa > p_assessment_pressure_max_pa OR
     p_request_sha256 !~ '^[0-9a-f]{64}$' OR
     p_operation NOT IN ('inspect', 'equilibrium', 'scheil') OR
     p_status NOT IN ('input_validated', 'equilibrium_completed', 'scheil_completed', 'failed', 'timeout', 'unsupported') OR
     p_evidence_sha256 !~ '^[0-9a-f]{64}$' OR
     p_evidence_size_bytes NOT BETWEEN 1 AND 33554432 OR
     p_evidence_payload IS NULL OR octet_length(p_evidence_payload) <> p_evidence_size_bytes OR
     encode(sha256(p_evidence_payload), 'hex') <> p_evidence_sha256 OR
     p_evidence_path <> ('/outputs/calphad/' ||
       CASE p_operation WHEN 'inspect' THEN 'inspection' ELSE p_operation END ||
       '/' || p_evidence_sha256 || '.json') OR
     p_runtime_image_id !~ '^sha256:[0-9a-f]{64}$' OR
     p_pycalphad_version <> '0.11.2' OR
     jsonb_typeof(p_owner_declaration) <> 'object' OR
     jsonb_typeof(p_metadata) <> 'object' OR
     octet_length(convert_to(p_metadata::text, 'UTF8')) > 65536 THEN
    RAISE EXCEPTION 'CALPHAD_VALIDATION_BINDING_INVALID: validation request is not canonical or content-bound'
      USING ERRCODE = '23514';
  END IF;

  BEGIN
    evidence_json := convert_from(p_evidence_payload, 'UTF8')::jsonb;
  EXCEPTION WHEN others THEN
    RAISE EXCEPTION 'CALPHAD_EVIDENCE_SCHEMA_INVALID: retained evidence is not UTF-8 JSON'
      USING ERRCODE = '23514';
  END;
  IF jsonb_typeof(evidence_json) IS DISTINCT FROM 'object' OR
     (SELECT count(*) FROM jsonb_object_keys(evidence_json)) <> 7 OR
     jsonb_typeof(evidence_json->'schema_version') IS DISTINCT FROM 'string' OR
     jsonb_typeof(evidence_json->'operation') IS DISTINCT FROM 'string' OR
     evidence_json->>'operation' IS DISTINCT FROM p_operation OR
     jsonb_typeof(evidence_json->'database_binding') IS DISTINCT FROM 'object' OR
     jsonb_typeof(evidence_json->'request') IS DISTINCT FROM 'object' OR
     jsonb_typeof(evidence_json->'execution_contract') IS DISTINCT FROM 'object' OR
     jsonb_typeof(evidence_json->'validation_persistence') IS DISTINCT FROM 'object' OR
     NOT (
       (evidence_json->>'schema_version' IS NOT DISTINCT FROM 'ultra.calphad.tool-evidence.v3' AND
        ((p_operation = 'inspect' AND p_status = 'input_validated') OR
         (p_operation = 'equilibrium' AND p_status = 'equilibrium_completed') OR
         (p_operation = 'scheil' AND p_status = 'scheil_completed')) AND
        evidence_json ?& ARRAY[
          'schema_version', 'operation', 'database_binding', 'request', 'result',
          'execution_contract', 'validation_persistence'
        ] AND jsonb_typeof(evidence_json->'result') IS NOT DISTINCT FROM 'object') OR
       (evidence_json->>'schema_version' IS NOT DISTINCT FROM 'ultra.calphad.failure-evidence.v1' AND
        p_status IN ('failed', 'timeout', 'unsupported') AND
        evidence_json ?& ARRAY[
          'schema_version', 'operation', 'database_binding', 'request', 'outcome',
          'execution_contract', 'validation_persistence'
        ] AND jsonb_typeof(evidence_json->'outcome') IS NOT DISTINCT FROM 'object')
     ) OR
     (SELECT count(*) FROM jsonb_object_keys(evidence_json->'database_binding')) <> 15 OR
     NOT (evidence_json->'database_binding' ?& ARRAY[
       'kind', 'database_id', 'resource_id', 'sha256', 'size_bytes', 'database_format',
       'source', 'license_id', 'assessment_scope', 'reference_state',
       'temperature_limits_K', 'assessment_pressure_limits_Pa', 'binding_schema',
       'binding_authority', 'declaration_authority'
     ]) OR
     evidence_json#>>'{database_binding,kind}' IS DISTINCT FROM 'resource' OR
     evidence_json#>>'{database_binding,resource_id}' IS DISTINCT FROM p_resource_id OR
     lower(btrim(evidence_json#>>'{database_binding,sha256}')) IS DISTINCT FROM p_database_sha256 OR
     evidence_json#>'{database_binding,size_bytes}' IS DISTINCT FROM to_jsonb(p_database_size_bytes) OR
     evidence_json#>>'{database_binding,database_format}' IS DISTINCT FROM p_database_format OR
     evidence_json#>>'{database_binding,binding_schema}' IS DISTINCT FROM 'ultra.selected_resource.v1' OR
     evidence_json#>>'{database_binding,binding_authority}' IS DISTINCT FROM 'control_resource_catalog' OR
     evidence_json#>>'{database_binding,declaration_authority}' IS DISTINCT FROM 'resource_owner' OR
     evidence_json#>>'{database_binding,database_id}' IS DISTINCT FROM p_owner_declaration->>'database_id' OR
     evidence_json#>>'{database_binding,source}' IS DISTINCT FROM p_owner_declaration->>'source' OR
     evidence_json#>>'{database_binding,license_id}' IS DISTINCT FROM p_owner_declaration->>'license_id' OR
     evidence_json#>>'{database_binding,assessment_scope}' IS DISTINCT FROM
       p_owner_declaration->>'assessment_scope' OR
     evidence_json#>>'{database_binding,reference_state}' IS DISTINCT FROM
       p_owner_declaration->>'reference_state' OR
     evidence_json#>'{database_binding,temperature_limits_K}' IS DISTINCT FROM
       p_owner_declaration->'assessment_temperature_limits_K' OR
     evidence_json#>'{database_binding,assessment_pressure_limits_Pa}' IS DISTINCT FROM
       jsonb_build_array(p_assessment_pressure_min_pa, p_assessment_pressure_max_pa) OR
     evidence_json#>>'{request,runtime_image_id}' IS DISTINCT FROM p_runtime_image_id OR
     evidence_json->'execution_contract' IS DISTINCT FROM jsonb_build_object(
       'interface', 'fixed ultra_deepagents.materials.calphad public surface',
       'caller_code_accepted', false,
       'caller_models_or_solver_options_accepted', false,
       'network', 'none',
       'no_new_privileges', true,
       'read_only_root_filesystem', true,
       'cap_drop_all', true,
       'cpus_at_most', 8,
       'memory_bytes_at_most', 34359738368,
       'pids_at_most', 4096,
       'runtime_image_id', p_runtime_image_id,
       'max_components', 32,
       'max_phases', 128,
       'max_axis_values', 64,
       'max_grid_points', 256,
       'wall_time_seconds', 30,
       'max_result_bytes', 16777216
     ) OR
     (SELECT count(*) FROM jsonb_object_keys(evidence_json->'validation_persistence')) <> 4 OR
     NOT (evidence_json->'validation_persistence' ?& ARRAY[
       'catalog_status', 'catalog_metadata_updated', 'mode', 'note'
     ]) OR
     evidence_json#>>'{validation_persistence,catalog_status}' IS DISTINCT FROM 'pending' OR
     evidence_json#>'{validation_persistence,catalog_metadata_updated}' IS DISTINCT FROM 'false'::jsonb OR
     evidence_json#>>'{validation_persistence,mode}' IS DISTINCT FROM 'immutable_per_run_evidence' OR
     jsonb_typeof(evidence_json#>'{validation_persistence,note}') IS DISTINCT FROM 'string' OR
     char_length(btrim(evidence_json#>>'{validation_persistence,note}')) NOT BETWEEN 1 AND 1024 THEN
    RAISE EXCEPTION 'CALPHAD_EVIDENCE_SCHEMA_INVALID: retained evidence root or database binding is inconsistent'
      USING ERRCODE = '23514';
  END IF;

  SELECT run.* INTO run_record
  FROM public.control_runs run
  WHERE run.run_id = p_run_id
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'CALPHAD_RUN_LEASE_INVALID: active run is missing'
      USING ERRCODE = '28000';
  END IF;
  SELECT lease.* INTO lease_record
  FROM public.control_run_leases lease
  WHERE lease.run_id = p_run_id
  FOR UPDATE;
  IF NOT FOUND OR run_record.status <> 'running' OR
     lease_record.lease_expires_at <= clock_timestamp() OR
     lease_record.worker_id <> p_lease_worker_id OR
     sha256(convert_to(lease_record.lease_token, 'UTF8')) <>
       sha256(convert_to(p_lease_token, 'UTF8')) THEN
    RAISE EXCEPTION 'CALPHAD_RUN_LEASE_INVALID: worker identity or active lease does not match'
      USING ERRCODE = '28000';
  END IF;
  IF run_record.metadata->'calphad_runtime_policy' <> jsonb_build_object(
       'schema_version', 'ultra.calphad.runtime-policy.v2',
       'authority', 'control_plane',
       'runtime_image_id', p_runtime_image_id,
       'pycalphad_version', '0.11.2',
       'network', 'none',
       'no_new_privileges', true,
       'read_only_root_filesystem', true,
       'cap_drop_all', true,
       'cpus_at_most', 8,
       'memory_bytes_at_most', 34359738368,
       'pids_at_most', 4096
     ) THEN
    RAISE EXCEPTION 'CALPHAD_RUNTIME_POLICY_INVALID: run policy does not authorize this runtime'
      USING ERRCODE = '28000';
  END IF;

  SELECT resource.* INTO resource_record
  FROM public.control_resources resource
  WHERE resource.resource_id = p_resource_id
    AND resource.owner_user_id = p_owner_user_id
    AND (COALESCE(NULLIF(btrim(resource.owner_org_id), ''), '') = '' OR
         COALESCE(NULLIF(btrim(resource.owner_org_id), ''), '') = normalized_org)
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'CALPHAD_RESOURCE_NOT_FOUND' USING ERRCODE = 'P0002';
  END IF;
  derived_format := CASE
    WHEN lower(btrim(resource_record.original_name)) ~ '\.tdb$' THEN 'tdb'
    WHEN lower(btrim(resource_record.original_name)) ~ '\.dat$' THEN 'dat'
    ELSE ''
  END;
  IF resource_record.status <> 'active' OR derived_format <> p_database_format OR
     lower(btrim(resource_record.sha256)) <> p_database_sha256 OR
     resource_record.size_bytes <> p_database_size_bytes THEN
    RAISE EXCEPTION 'CALPHAD_RESOURCE_BINDING_INVALID: live catalog binding changed'
      USING ERRCODE = '23514';
  END IF;

  SELECT revision.* INTO revision_record
  FROM public.control_calphad_revisions revision
  WHERE revision.resource_id = p_resource_id;
  IF NOT FOUND OR revision_record.owner_user_id <> run_record.user_id OR
     (COALESCE(NULLIF(btrim(revision_record.owner_org_id), ''), '') <> '' AND
      COALESCE(NULLIF(btrim(run_record.metadata->>'org_id'), ''), '') <>
        COALESCE(NULLIF(btrim(revision_record.owner_org_id), ''), '')) OR
     revision_record.sha256 <> p_database_sha256 OR
     revision_record.size_bytes <> p_database_size_bytes OR
     revision_record.database_format <> p_database_format OR
     revision_record.assessment_pressure_min_pa <> p_assessment_pressure_min_pa OR
     revision_record.assessment_pressure_max_pa <> p_assessment_pressure_max_pa OR
     revision_record.metadata->'owner_declaration' <> p_owner_declaration THEN
    RAISE EXCEPTION 'CALPHAD_REVISION_BINDING_INVALID: immutable revision does not match callback authority'
      USING ERRCODE = '23514';
  END IF;

  calphad_metadata := resource_record.metadata->'calphad';
  declared_temperature := COALESCE(
    calphad_metadata->'assessment_temperature_limits_K',
    calphad_metadata->'tdb_temperature_limits_K'
  );
  expected_declaration := jsonb_build_object(
    'schema_version', 'ultra.calphad.owner-declaration.v1',
    'authority', 'resource_owner',
    'database_id', COALESCE(NULLIF(btrim(calphad_metadata->>'database_id'), ''), p_resource_id),
    'source', btrim(calphad_metadata->>'source'),
    'license_id', COALESCE(NULLIF(btrim(calphad_metadata->>'license_id'), ''),
                           btrim(calphad_metadata->>'license_identifier')),
    'assessment_scope', btrim(calphad_metadata->>'assessment_scope'),
    'reference_state', btrim(calphad_metadata->>'reference_state'),
    'assessment_temperature_limits_K', declared_temperature,
    'assessment_pressure_limits_Pa', calphad_metadata->'assessment_pressure_limits_Pa',
    'database_format', derived_format
  );
  IF p_owner_declaration <> expected_declaration OR
     (calphad_metadata ? 'assessment_temperature_limits_K' AND
      calphad_metadata ? 'tdb_temperature_limits_K' AND
      calphad_metadata->'assessment_temperature_limits_K' <>
        calphad_metadata->'tdb_temperature_limits_K') THEN
    RAISE EXCEPTION 'CALPHAD_OWNER_DECLARATION_INVALID: live and immutable provenance differ'
      USING ERRCODE = '23514';
  END IF;

  IF jsonb_typeof(run_record.metadata->'file_ids') <> 'array' OR
     jsonb_typeof(run_record.metadata->'resource_descriptors') <> 'array' THEN
    RAISE EXCEPTION 'CALPHAD_SELECTED_RESOURCE_INVALID: run lacks server-selected resource authority'
      USING ERRCODE = '23514';
  END IF;
  SELECT count(*) INTO selected_file_count
  FROM jsonb_array_elements_text(run_record.metadata->'file_ids') selected(value)
  WHERE selected.value = p_resource_id;
  SELECT count(*) INTO descriptor_candidate_count
  FROM jsonb_array_elements(run_record.metadata->'resource_descriptors') descriptor(value)
  WHERE descriptor.value->>'resource_id' = p_resource_id OR
        descriptor.value->>'file_id' = p_resource_id;
  SELECT count(*) INTO descriptor_exact_count
  FROM jsonb_array_elements(run_record.metadata->'resource_descriptors') descriptor(value)
  WHERE descriptor.value->>'type' = 'selected_resource'
    AND descriptor.value->>'binding_schema' = 'ultra.selected_resource.v1'
    AND descriptor.value->>'authority' = 'control_resource_catalog'
    AND descriptor.value->>'resource_id' = p_resource_id
    AND descriptor.value->>'file_id' = p_resource_id
    AND lower(btrim(descriptor.value->>'sha256')) = p_database_sha256
    AND descriptor.value->'size_bytes' = to_jsonb(p_database_size_bytes)
    AND descriptor.value->>'database_format' = p_database_format
    AND CASE
          WHEN lower(btrim(descriptor.value->>'original_name')) ~ '\.tdb$' THEN 'tdb'
          WHEN lower(btrim(descriptor.value->>'original_name')) ~ '\.dat$' THEN 'dat'
          ELSE ''
        END = p_database_format
    AND descriptor.value->>'calphad_governance_scope' = 'owner_validation'
    AND jsonb_build_object(
      'schema_version', 'ultra.calphad.owner-declaration.v1',
      'authority', 'resource_owner',
      'database_id', COALESCE(NULLIF(btrim(descriptor.value#>>'{metadata,calphad,database_id}'), ''), p_resource_id),
      'source', btrim(descriptor.value#>>'{metadata,calphad,source}'),
      'license_id', COALESCE(NULLIF(btrim(descriptor.value#>>'{metadata,calphad,license_id}'), ''),
                             btrim(descriptor.value#>>'{metadata,calphad,license_identifier}')),
      'assessment_scope', btrim(descriptor.value#>>'{metadata,calphad,assessment_scope}'),
      'reference_state', btrim(descriptor.value#>>'{metadata,calphad,reference_state}'),
      'assessment_temperature_limits_K', COALESCE(
        descriptor.value#>'{metadata,calphad,assessment_temperature_limits_K}',
        descriptor.value#>'{metadata,calphad,tdb_temperature_limits_K}'
      ),
      'assessment_pressure_limits_Pa',
        descriptor.value#>'{metadata,calphad,assessment_pressure_limits_Pa}',
      'database_format', descriptor.value->>'database_format'
    ) = p_owner_declaration;
  IF selected_file_count <> 1 OR descriptor_candidate_count <> 1 OR descriptor_exact_count <> 1 THEN
    RAISE EXCEPTION 'CALPHAD_SELECTED_RESOURCE_INVALID: descriptor is missing, ambiguous, or content-mismatched'
      USING ERRCODE = '23514';
  END IF;

  SELECT blob.input_size_bytes, blob.payload INTO stored_size, stored_payload
  FROM public.control_calphad_input_blobs blob
  WHERE blob.input_sha256 = revision_record.sha256;
  IF NOT FOUND OR stored_size <> revision_record.size_bytes OR
     octet_length(stored_payload) <> revision_record.size_bytes OR
     encode(sha256(stored_payload), 'hex') <> revision_record.sha256 THEN
    RAISE EXCEPTION 'CALPHAD_INPUT_RETENTION_REQUIRED: exact revision bytes are missing'
      USING ERRCODE = '23514';
  END IF;

  IF p_operation IN ('equilibrium', 'scheil') AND NOT EXISTS (
    SELECT 1
    FROM public.control_calphad_validation_events inspection
    JOIN public.control_calphad_evidence_blobs blob
      ON blob.evidence_sha256 = inspection.evidence_sha256
     AND blob.evidence_size_bytes = inspection.evidence_size_bytes
     AND octet_length(blob.payload) = inspection.evidence_size_bytes
     AND encode(sha256(blob.payload), 'hex') = inspection.evidence_sha256
    WHERE inspection.revision_id = revision_record.revision_id
      AND inspection.run_id = p_run_id
      AND inspection.operation = 'inspect'
      AND inspection.status = 'input_validated'
      AND inspection.runtime_image_id = p_runtime_image_id
      AND inspection.database_format = p_database_format
      AND inspection.database_inventory_sha256 = p_database_inventory_sha256
      AND inspection.assessment_pressure_min_pa = p_assessment_pressure_min_pa
      AND inspection.assessment_pressure_max_pa = p_assessment_pressure_max_pa
      AND inspection.evidence_contract_version = 'ultra.calphad.retained-evidence.v2'
      AND inspection.evidence_sha256 = p_inspection_evidence_sha256
  ) THEN
    RAISE EXCEPTION 'CALPHAD_INSPECTION_REQUIRED: exact retained inspection lineage is missing'
      USING ERRCODE = '23514';
  END IF;

  SELECT validation.* INTO existing_validation
  FROM public.control_calphad_validation_events validation
  WHERE validation.revision_id = revision_record.revision_id
    AND validation.run_id = p_run_id
    AND validation.operation = p_operation
    AND validation.evidence_sha256 = p_evidence_sha256;
  IF FOUND THEN
    IF existing_validation.resource_id <> p_resource_id OR
       existing_validation.database_sha256 <> p_database_sha256 OR
       existing_validation.database_size_bytes <> p_database_size_bytes OR
       existing_validation.database_format <> p_database_format OR
       existing_validation.assessment_pressure_min_pa <> p_assessment_pressure_min_pa OR
       existing_validation.assessment_pressure_max_pa <> p_assessment_pressure_max_pa OR
       existing_validation.database_inventory_sha256 IS DISTINCT FROM p_database_inventory_sha256 OR
       existing_validation.request_sha256 <> p_request_sha256 OR
       existing_validation.status <> p_status OR
       existing_validation.failure_domain IS DISTINCT FROM p_failure_domain OR
       existing_validation.failure_stage IS DISTINCT FROM p_failure_stage OR
       existing_validation.failure_code IS DISTINCT FROM p_failure_code OR
       existing_validation.evidence_path <> p_evidence_path OR
       existing_validation.evidence_size_bytes <> p_evidence_size_bytes OR
       existing_validation.runtime_image_id <> p_runtime_image_id OR
       existing_validation.pycalphad_version <> p_pycalphad_version OR
       existing_validation.inspection_evidence_sha256 IS DISTINCT FROM p_inspection_evidence_sha256 OR
       existing_validation.evidence_contract_version <> 'ultra.calphad.retained-evidence.v2' OR
       existing_validation.created_by_authority <> 'trusted_worker' THEN
      RAISE EXCEPTION 'CALPHAD_VALIDATION_CONFLICT: evidence identity was reused with different authority fields'
        USING ERRCODE = '23505';
    END IF;
    SELECT blob.evidence_size_bytes, blob.payload INTO stored_size, stored_payload
    FROM public.control_calphad_evidence_blobs blob
    WHERE blob.evidence_sha256 = p_evidence_sha256;
    IF NOT FOUND OR stored_size <> p_evidence_size_bytes OR stored_payload <> p_evidence_payload THEN
      RAISE EXCEPTION 'CALPHAD_EVIDENCE_RETENTION_REQUIRED: replay evidence bytes are missing or inconsistent'
        USING ERRCODE = '23514';
    END IF;
    IF lease_record.lease_expires_at <= clock_timestamp() THEN
      RAISE EXCEPTION 'CALPHAD_RUN_LEASE_INVALID: lease expired before replay completion'
        USING ERRCODE = '28000';
    END IF;
    RETURN QUERY SELECT existing_validation.validation_id, existing_validation.revision_id,
      existing_validation.resource_id, existing_validation.database_sha256,
      existing_validation.database_size_bytes, existing_validation.database_format,
      existing_validation.assessment_pressure_min_pa, existing_validation.assessment_pressure_max_pa,
      existing_validation.database_inventory_sha256, existing_validation.request_sha256,
      existing_validation.status, existing_validation.operation, existing_validation.failure_domain,
      existing_validation.failure_stage, existing_validation.failure_code, existing_validation.evidence_path,
      existing_validation.evidence_sha256, existing_validation.evidence_size_bytes,
      existing_validation.runtime_image_id, existing_validation.pycalphad_version,
      existing_validation.run_id, existing_validation.inspection_evidence_sha256,
      existing_validation.evidence_contract_version, existing_validation.created_by_authority,
      existing_validation.created_at, existing_validation.metadata, true;
    RETURN;
  END IF;

  INSERT INTO public.control_calphad_evidence_blobs
    (evidence_sha256, evidence_size_bytes, encoding, payload, created_at)
  VALUES (p_evidence_sha256, p_evidence_size_bytes, 'raw', p_evidence_payload, clock_timestamp())
  ON CONFLICT ON CONSTRAINT control_calphad_evidence_blobs_pkey DO NOTHING;
  SELECT blob.evidence_size_bytes, blob.payload INTO stored_size, stored_payload
  FROM public.control_calphad_evidence_blobs blob
  WHERE blob.evidence_sha256 = p_evidence_sha256;
  IF stored_size <> p_evidence_size_bytes OR stored_payload <> p_evidence_payload THEN
    RAISE EXCEPTION 'CALPHAD_EVIDENCE_RETENTION_REQUIRED: retained evidence conflicts with callback bytes'
      USING ERRCODE = '23514';
  END IF;

  SELECT capacity.* INTO capacity_record
  FROM public.control_calphad_tenant_capacity capacity
  WHERE capacity.owner_user_id = btrim(revision_record.owner_user_id)
    AND capacity.owner_org_id = COALESCE(NULLIF(btrim(revision_record.owner_org_id), ''), '')
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'CALPHAD_CAPACITY_STATE_INVALID: tenant capacity row is missing'
      USING ERRCODE = '23514';
  END IF;
  UPDATE public.control_calphad_tenant_capacity capacity
  SET retained_evidence_bytes = capacity.retained_evidence_bytes + p_evidence_size_bytes,
      validation_events = capacity.validation_events + 1,
      updated_at = clock_timestamp()
  WHERE capacity.owner_user_id = capacity_record.owner_user_id
    AND capacity.owner_org_id = capacity_record.owner_org_id
    AND capacity.retained_input_bytes::numeric + capacity.retained_evidence_bytes::numeric +
        p_evidence_size_bytes::numeric <= capacity.max_retained_bytes::numeric
    AND capacity.validation_events < capacity.max_validation_events
  RETURNING 1 INTO capacity_updated;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'CALPHAD_TENANT_CAPACITY_EXCEEDED: retained-byte or validation-event capacity exhausted'
      USING ERRCODE = '23514';
  END IF;

  IF run_record.status <> 'running' OR lease_record.lease_expires_at <= clock_timestamp() OR
     lease_record.worker_id <> p_lease_worker_id OR
     sha256(convert_to(lease_record.lease_token, 'UTF8')) <>
       sha256(convert_to(p_lease_token, 'UTF8')) THEN
    RAISE EXCEPTION 'CALPHAD_RUN_LEASE_INVALID: lease expired or changed before event insert'
      USING ERRCODE = '28000';
  END IF;

  new_validation_id := 'calphad_validation_' || replace(gen_random_uuid()::text, '-', '');
  created_timestamp := clock_timestamp();
  validation_metadata := jsonb_build_object(
      'server_managed', true,
      'revision_id', revision_record.revision_id,
      'assessment_pressure_limits_Pa',
        jsonb_build_array(p_assessment_pressure_min_pa, p_assessment_pressure_max_pa)
    );
  INSERT INTO public.control_calphad_validation_events
    (validation_id, revision_id, resource_id, database_sha256, database_size_bytes,
     database_format, assessment_pressure_min_pa, assessment_pressure_max_pa,
     database_inventory_sha256, request_sha256, status, operation,
     failure_domain, failure_stage, failure_code, evidence_path,
     evidence_sha256, evidence_size_bytes, runtime_image_id, pycalphad_version, run_id,
     inspection_evidence_sha256, evidence_contract_version, created_by_authority,
     created_at, metadata)
  VALUES (new_validation_id, revision_record.revision_id, p_resource_id,
          p_database_sha256, p_database_size_bytes, p_database_format,
          p_assessment_pressure_min_pa, p_assessment_pressure_max_pa,
          p_database_inventory_sha256, p_request_sha256, p_status, p_operation,
          p_failure_domain, p_failure_stage, p_failure_code, p_evidence_path,
          p_evidence_sha256, p_evidence_size_bytes, p_runtime_image_id,
          p_pycalphad_version, p_run_id, p_inspection_evidence_sha256,
          'ultra.calphad.retained-evidence.v2', 'trusted_worker',
          created_timestamp, validation_metadata)
  ON CONFLICT DO NOTHING
  RETURNING * INTO inserted_validation;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'CALPHAD_VALIDATION_CONFLICT: concurrent evidence identity conflict'
      USING ERRCODE = '23505';
  END IF;

  RETURN QUERY SELECT inserted_validation.validation_id, inserted_validation.revision_id,
    inserted_validation.resource_id, inserted_validation.database_sha256,
    inserted_validation.database_size_bytes, inserted_validation.database_format,
    inserted_validation.assessment_pressure_min_pa, inserted_validation.assessment_pressure_max_pa,
    inserted_validation.database_inventory_sha256, inserted_validation.request_sha256,
    inserted_validation.status, inserted_validation.operation, inserted_validation.failure_domain,
    inserted_validation.failure_stage, inserted_validation.failure_code, inserted_validation.evidence_path,
    inserted_validation.evidence_sha256, inserted_validation.evidence_size_bytes,
    inserted_validation.runtime_image_id, inserted_validation.pycalphad_version,
    inserted_validation.run_id, inserted_validation.inspection_evidence_sha256,
    inserted_validation.evidence_contract_version, inserted_validation.created_by_authority,
    inserted_validation.created_at, inserted_validation.metadata, true;
END;
$$;

WITH revision_usage AS (
  SELECT btrim(owner_user_id) AS owner_user_id,
         COALESCE(NULLIF(btrim(owner_org_id), ''), '') AS owner_org_id,
         sum(size_bytes)::bigint AS retained_input_bytes
  FROM public.control_calphad_revisions
  GROUP BY btrim(owner_user_id), COALESCE(NULLIF(btrim(owner_org_id), ''), '')
), event_usage AS (
  SELECT btrim(revision.owner_user_id) AS owner_user_id,
         COALESCE(NULLIF(btrim(revision.owner_org_id), ''), '') AS owner_org_id,
         COALESCE(sum(validation.evidence_size_bytes), 0)::bigint AS retained_evidence_bytes,
         count(*)::bigint AS validation_events
  FROM public.control_calphad_validation_events validation
  JOIN public.control_calphad_revisions revision
    ON revision.revision_id = validation.revision_id
  GROUP BY btrim(revision.owner_user_id), COALESCE(NULLIF(btrim(revision.owner_org_id), ''), '')
), usage AS (
  SELECT COALESCE(revision_usage.owner_user_id, event_usage.owner_user_id) AS owner_user_id,
         COALESCE(revision_usage.owner_org_id, event_usage.owner_org_id) AS owner_org_id,
         COALESCE(revision_usage.retained_input_bytes, 0) AS retained_input_bytes,
         COALESCE(event_usage.retained_evidence_bytes, 0) AS retained_evidence_bytes,
         COALESCE(event_usage.validation_events, 0) AS validation_events
  FROM revision_usage
  FULL OUTER JOIN event_usage USING (owner_user_id, owner_org_id)
)
INSERT INTO public.control_calphad_tenant_capacity
 (owner_user_id, owner_org_id, max_retained_bytes, max_validation_events,
  retained_input_bytes, retained_evidence_bytes, validation_events, updated_at)
SELECT owner_user_id, owner_org_id,
       GREATEST(4294967296::bigint, retained_input_bytes + retained_evidence_bytes),
       GREATEST(100000::bigint, validation_events),
       retained_input_bytes, retained_evidence_bytes, validation_events, clock_timestamp()
FROM usage
ON CONFLICT (owner_user_id, owner_org_id) DO UPDATE
SET retained_input_bytes = EXCLUDED.retained_input_bytes,
    retained_evidence_bytes = EXCLUDED.retained_evidence_bytes,
    validation_events = EXCLUDED.validation_events,
    max_retained_bytes = GREATEST(
      public.control_calphad_tenant_capacity.max_retained_bytes,
      EXCLUDED.retained_input_bytes + EXCLUDED.retained_evidence_bytes
    ),
    max_validation_events = GREATEST(
      public.control_calphad_tenant_capacity.max_validation_events,
      EXCLUDED.validation_events
    ),
    updated_at = EXCLUDED.updated_at;

CREATE OR REPLACE FUNCTION public.ultra_create_calphad_revision_v1(
  p_resource_id text,
  p_owner_user_id text,
  p_owner_org_id text,
  p_parent_revision_id text,
  p_expected_sha256 text,
  p_expected_size_bytes bigint,
  p_database_format text,
  p_assessment_pressure_min_pa double precision,
  p_assessment_pressure_max_pa double precision,
  p_input_payload bytea,
  p_metadata jsonb
)
RETURNS TABLE (
  revision_id text,
  resource_id text,
  owner_user_id text,
  owner_org_id text,
  sha256 text,
  size_bytes bigint,
  database_format text,
  assessment_pressure_min_pa double precision,
  assessment_pressure_max_pa double precision,
  parent_revision_id text,
  created_by_user_id text,
  created_at timestamptz,
  metadata jsonb
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog
AS $$
DECLARE
  resource_record public.control_resources%ROWTYPE;
  existing_revision public.control_calphad_revisions%ROWTYPE;
  parent_revision public.control_calphad_revisions%ROWTYPE;
  inserted_revision public.control_calphad_revisions%ROWTYPE;
  capacity_updated bigint;
  stored_size bigint;
  stored_payload bytea;
  normalized_org text;
  normalized_parent text;
  derived_format text;
  calphad_metadata jsonb;
  declared_temperature jsonb;
  expected_declaration jsonb;
  supplied_declaration jsonb;
  revision_metadata jsonb;
  created_timestamp timestamptz;
  new_revision_id text;
  new_validation_id text;
BEGIN
  p_resource_id := btrim(p_resource_id);
  p_owner_user_id := btrim(p_owner_user_id);
  normalized_org := COALESCE(NULLIF(btrim(p_owner_org_id), ''), '');
  normalized_parent := NULLIF(btrim(p_parent_revision_id), '');
  p_expected_sha256 := lower(btrim(p_expected_sha256));
  p_database_format := btrim(p_database_format);

  IF p_resource_id = '' OR p_owner_user_id = '' OR
     p_expected_sha256 !~ '^[0-9a-f]{64}$' OR
     p_expected_size_bytes NOT BETWEEN 1 AND 67108864 OR
     p_database_format NOT IN ('tdb', 'dat') OR
     p_assessment_pressure_min_pa < 1e-9 OR
     p_assessment_pressure_max_pa > 1e12 OR
     p_assessment_pressure_min_pa > p_assessment_pressure_max_pa OR
     p_input_payload IS NULL OR octet_length(p_input_payload) <> p_expected_size_bytes OR
     encode(sha256(p_input_payload), 'hex') <> p_expected_sha256 OR
     jsonb_typeof(p_metadata) <> 'object' OR
     octet_length(convert_to(p_metadata::text, 'UTF8')) > 65536 THEN
    RAISE EXCEPTION 'CALPHAD_REVISION_BINDING_INVALID: revision request is not canonical or content-bound'
      USING ERRCODE = '23514';
  END IF;

  SELECT resource.* INTO resource_record
  FROM public.control_resources resource
  WHERE resource.resource_id = p_resource_id
    AND resource.owner_user_id = p_owner_user_id
    AND (COALESCE(NULLIF(btrim(resource.owner_org_id), ''), '') = '' OR
         COALESCE(NULLIF(btrim(resource.owner_org_id), ''), '') = normalized_org)
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'CALPHAD_RESOURCE_NOT_FOUND' USING ERRCODE = 'P0002';
  END IF;

  derived_format := CASE
    WHEN lower(btrim(resource_record.original_name)) ~ '\.tdb$' THEN 'tdb'
    WHEN lower(btrim(resource_record.original_name)) ~ '\.dat$' THEN 'dat'
    ELSE ''
  END;
  IF btrim(resource_record.status) <> 'active' OR
     derived_format = '' OR derived_format <> p_database_format OR
     lower(btrim(COALESCE(resource_record.content_type, ''))) NOT IN
       ('', 'application/octet-stream', 'text/plain', 'application/x-thermocalc-tdb') OR
     lower(btrim(resource_record.sha256)) <> p_expected_sha256 OR
     resource_record.size_bytes <> p_expected_size_bytes THEN
    RAISE EXCEPTION 'CALPHAD_RESOURCE_BINDING_INVALID: live catalog binding changed or format is unsupported'
      USING ERRCODE = '23514';
  END IF;

  calphad_metadata := resource_record.metadata->'calphad';
  declared_temperature := COALESCE(
    calphad_metadata->'assessment_temperature_limits_K',
    calphad_metadata->'tdb_temperature_limits_K'
  );
  IF jsonb_typeof(calphad_metadata) <> 'object' OR
     (calphad_metadata ? 'assessment_temperature_limits_K' AND
      calphad_metadata ? 'tdb_temperature_limits_K' AND
      calphad_metadata->'assessment_temperature_limits_K' <>
        calphad_metadata->'tdb_temperature_limits_K') THEN
    RAISE EXCEPTION 'CALPHAD_OWNER_DECLARATION_INVALID: owner declaration is missing or contradictory'
      USING ERRCODE = '23514';
  END IF;
  expected_declaration := jsonb_build_object(
    'schema_version', 'ultra.calphad.owner-declaration.v1',
    'authority', 'resource_owner',
    'database_id', COALESCE(NULLIF(btrim(calphad_metadata->>'database_id'), ''), p_resource_id),
    'source', btrim(calphad_metadata->>'source'),
    'license_id', COALESCE(NULLIF(btrim(calphad_metadata->>'license_id'), ''),
                           btrim(calphad_metadata->>'license_identifier')),
    'assessment_scope', btrim(calphad_metadata->>'assessment_scope'),
    'reference_state', btrim(calphad_metadata->>'reference_state'),
    'assessment_temperature_limits_K', declared_temperature,
    'assessment_pressure_limits_Pa', calphad_metadata->'assessment_pressure_limits_Pa',
    'database_format', derived_format
  );
  supplied_declaration := p_metadata->'owner_declaration';
  IF supplied_declaration IS NULL OR supplied_declaration <> expected_declaration OR
     supplied_declaration->'assessment_pressure_limits_Pa' <>
       jsonb_build_array(p_assessment_pressure_min_pa, p_assessment_pressure_max_pa) THEN
    RAISE EXCEPTION 'CALPHAD_OWNER_DECLARATION_INVALID: immutable owner declaration does not match the live resource'
      USING ERRCODE = '23514';
  END IF;
  revision_metadata := jsonb_build_object(
      'server_managed', true,
      'owner_declaration', expected_declaration,
      'assessment_pressure_limits_Pa',
        jsonb_build_array(p_assessment_pressure_min_pa, p_assessment_pressure_max_pa)
    );

  SELECT revision.* INTO existing_revision
  FROM public.control_calphad_revisions revision
  WHERE revision.resource_id = p_resource_id;
  IF FOUND THEN
    IF existing_revision.owner_user_id <> resource_record.owner_user_id OR
       COALESCE(NULLIF(btrim(existing_revision.owner_org_id), ''), '') <>
         COALESCE(NULLIF(btrim(resource_record.owner_org_id), ''), '') OR
       existing_revision.sha256 <> p_expected_sha256 OR
       existing_revision.size_bytes <> p_expected_size_bytes OR
       existing_revision.database_format <> p_database_format OR
       existing_revision.assessment_pressure_min_pa <> p_assessment_pressure_min_pa OR
       existing_revision.assessment_pressure_max_pa <> p_assessment_pressure_max_pa OR
       existing_revision.metadata->'owner_declaration' <> expected_declaration OR
       existing_revision.parent_revision_id IS DISTINCT FROM normalized_parent THEN
      RAISE EXCEPTION 'CALPHAD_REVISION_CONFLICT: existing revision differs from the immutable request'
        USING ERRCODE = '23505';
    END IF;
    INSERT INTO public.control_calphad_input_blobs
      (input_sha256, input_size_bytes, encoding, payload, created_at)
    VALUES (p_expected_sha256, p_expected_size_bytes, 'raw', p_input_payload, clock_timestamp())
    ON CONFLICT (input_sha256) DO NOTHING;
    SELECT blob.input_size_bytes, blob.payload INTO stored_size, stored_payload
    FROM public.control_calphad_input_blobs blob
    WHERE blob.input_sha256 = p_expected_sha256;
    IF stored_size <> p_expected_size_bytes OR stored_payload <> p_input_payload THEN
      RAISE EXCEPTION 'CALPHAD_INPUT_RETENTION_REQUIRED: retained input conflicts with the request'
        USING ERRCODE = '23514';
    END IF;
    RETURN QUERY SELECT existing_revision.revision_id, existing_revision.resource_id,
      existing_revision.owner_user_id, existing_revision.owner_org_id,
      existing_revision.sha256, existing_revision.size_bytes, existing_revision.database_format,
      existing_revision.assessment_pressure_min_pa, existing_revision.assessment_pressure_max_pa,
      existing_revision.parent_revision_id, existing_revision.created_by_user_id,
      existing_revision.created_at, existing_revision.metadata;
    RETURN;
  END IF;

  IF normalized_parent IS NOT NULL THEN
    SELECT revision.* INTO parent_revision
    FROM public.control_calphad_revisions revision
    WHERE revision.revision_id = normalized_parent;
    IF NOT FOUND OR parent_revision.resource_id = p_resource_id OR
       parent_revision.owner_user_id <> resource_record.owner_user_id OR
       COALESCE(NULLIF(btrim(parent_revision.owner_org_id), ''), '') <>
         COALESCE(NULLIF(btrim(resource_record.owner_org_id), ''), '') THEN
      RAISE EXCEPTION 'CALPHAD_PARENT_NOT_FOUND' USING ERRCODE = 'P0002';
    END IF;
  END IF;

  INSERT INTO public.control_calphad_input_blobs
    (input_sha256, input_size_bytes, encoding, payload, created_at)
  VALUES (p_expected_sha256, p_expected_size_bytes, 'raw', p_input_payload, clock_timestamp())
  ON CONFLICT (input_sha256) DO NOTHING;
  SELECT blob.input_size_bytes, blob.payload INTO stored_size, stored_payload
  FROM public.control_calphad_input_blobs blob
  WHERE blob.input_sha256 = p_expected_sha256;
  IF stored_size <> p_expected_size_bytes OR stored_payload <> p_input_payload THEN
    RAISE EXCEPTION 'CALPHAD_INPUT_RETENTION_REQUIRED: retained input conflicts with the request'
      USING ERRCODE = '23514';
  END IF;

  INSERT INTO public.control_calphad_tenant_capacity
    (owner_user_id, owner_org_id, updated_at)
  VALUES (btrim(resource_record.owner_user_id),
          COALESCE(NULLIF(btrim(resource_record.owner_org_id), ''), ''), clock_timestamp())
  ON CONFLICT ON CONSTRAINT control_calphad_tenant_capacity_pkey DO NOTHING;
  UPDATE public.control_calphad_tenant_capacity capacity
  SET retained_input_bytes = capacity.retained_input_bytes + p_expected_size_bytes,
      validation_events = capacity.validation_events + 1,
      updated_at = clock_timestamp()
  WHERE capacity.owner_user_id = btrim(resource_record.owner_user_id)
    AND capacity.owner_org_id = COALESCE(NULLIF(btrim(resource_record.owner_org_id), ''), '')
    AND capacity.retained_input_bytes::numeric + capacity.retained_evidence_bytes::numeric +
        p_expected_size_bytes::numeric <= capacity.max_retained_bytes::numeric
    AND capacity.validation_events < capacity.max_validation_events
  RETURNING 1 INTO capacity_updated;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'CALPHAD_TENANT_CAPACITY_EXCEEDED: retained-byte or validation-event capacity exhausted'
      USING ERRCODE = '23514';
  END IF;

  created_timestamp := clock_timestamp();
  new_revision_id := 'calphad_revision_' || replace(gen_random_uuid()::text, '-', '');
  new_validation_id := 'calphad_validation_' || replace(gen_random_uuid()::text, '-', '');
  INSERT INTO public.control_calphad_revisions
    (revision_id, resource_id, owner_user_id, owner_org_id, sha256, size_bytes,
     database_format, assessment_pressure_min_pa, assessment_pressure_max_pa,
     parent_revision_id, created_by_user_id, created_at, metadata)
  VALUES (new_revision_id, p_resource_id, resource_record.owner_user_id,
          NULLIF(COALESCE(NULLIF(btrim(resource_record.owner_org_id), ''), ''), ''),
          p_expected_sha256, p_expected_size_bytes, p_database_format,
          p_assessment_pressure_min_pa, p_assessment_pressure_max_pa,
          normalized_parent, p_owner_user_id, created_timestamp, revision_metadata)
  RETURNING * INTO inserted_revision;

  INSERT INTO public.control_calphad_validation_events
    (validation_id, revision_id, resource_id, database_sha256, database_size_bytes,
     database_format, assessment_pressure_min_pa, assessment_pressure_max_pa,
     status, operation, created_by_authority, created_at, metadata)
  VALUES (new_validation_id, new_revision_id, p_resource_id, p_expected_sha256,
          p_expected_size_bytes, p_database_format, p_assessment_pressure_min_pa,
          p_assessment_pressure_max_pa, 'pending', 'registration', 'control_plane',
          created_timestamp, jsonb_build_object(
            'server_managed', true,
            'assessment_pressure_limits_Pa',
              jsonb_build_array(p_assessment_pressure_min_pa, p_assessment_pressure_max_pa)
          ));

  RETURN QUERY SELECT inserted_revision.revision_id, inserted_revision.resource_id,
    inserted_revision.owner_user_id, inserted_revision.owner_org_id,
    inserted_revision.sha256, inserted_revision.size_bytes, inserted_revision.database_format,
    inserted_revision.assessment_pressure_min_pa, inserted_revision.assessment_pressure_max_pa,
    inserted_revision.parent_revision_id, inserted_revision.created_by_user_id,
    inserted_revision.created_at, inserted_revision.metadata;
END;
$$;

REVOKE ALL ON FUNCTION public.ultra_create_calphad_revision_v1(
  text, text, text, text, text, bigint, text, double precision,
  double precision, bytea, jsonb
) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.ultra_append_calphad_validation_v1(
  text, text, text, text, bigint, text, jsonb, double precision,
  double precision, text, text, text, text, text, text, text, text,
  text, bigint, bytea, text, text, text, text, text, text, jsonb
) FROM PUBLIC;

REVOKE EXECUTE ON FUNCTION public.ultra_validate_calphad_revision_parent() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION public.ultra_validate_calphad_validation_run_authority() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION public.ultra_validate_calphad_pressure_binding() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION public.ultra_validate_calphad_input_retention() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION public.ultra_validate_calphad_equilibrium_inspection_lineage() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION public.ultra_reject_calphad_ledger_mutation() FROM PUBLIC;
