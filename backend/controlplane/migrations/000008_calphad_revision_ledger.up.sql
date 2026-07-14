CREATE TABLE IF NOT EXISTS control_calphad_input_blobs (
  input_sha256 text PRIMARY KEY CHECK (input_sha256 ~ '^[0-9a-f]{64}$'),
  input_size_bytes bigint NOT NULL,
  encoding text NOT NULL,
  payload bytea NOT NULL,
  created_at timestamptz NOT NULL,
  CONSTRAINT control_calphad_input_blob_binding_unique
    UNIQUE (input_sha256, input_size_bytes),
  CONSTRAINT control_calphad_input_blob_size_check
    CHECK (input_size_bytes BETWEEN 1 AND 67108864),
  CONSTRAINT control_calphad_input_blob_encoding_check
    CHECK (encoding = 'raw'),
  CONSTRAINT control_calphad_input_blob_payload_sha_check
    CHECK (encode(sha256(payload), 'hex') = input_sha256),
  CONSTRAINT control_calphad_input_blob_payload_size_check
    CHECK (octet_length(payload) = input_size_bytes)
);

CREATE TABLE IF NOT EXISTS control_calphad_revisions (
  revision_id text PRIMARY KEY,
  resource_id text NOT NULL UNIQUE,
  owner_user_id text NOT NULL,
  owner_org_id text,
  sha256 text NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
  size_bytes bigint NOT NULL CHECK (size_bytes > 0),
  database_format text NOT NULL,
  assessment_pressure_min_pa double precision,
  assessment_pressure_max_pa double precision,
  parent_revision_id text REFERENCES control_calphad_revisions(revision_id),
  created_by_user_id text,
  created_at timestamptz NOT NULL,
  metadata jsonb NOT NULL DEFAULT '{}' CHECK (jsonb_typeof(metadata) = 'object'),
  CONSTRAINT control_calphad_revisions_binding_unique
    UNIQUE (revision_id, resource_id, sha256, size_bytes, database_format),
  CONSTRAINT control_calphad_revisions_database_format_check
    CHECK (database_format IN ('tdb', 'dat')),
  CONSTRAINT control_calphad_revisions_pressure_binding_unique
    UNIQUE (revision_id, assessment_pressure_min_pa, assessment_pressure_max_pa),
  CONSTRAINT control_calphad_revisions_pressure_limits_check
    CHECK (assessment_pressure_min_pa IS NOT NULL AND assessment_pressure_max_pa IS NOT NULL AND
           assessment_pressure_min_pa >= 1e-9 AND assessment_pressure_max_pa <= 1e12 AND
           assessment_pressure_min_pa <= assessment_pressure_max_pa),
  CONSTRAINT control_calphad_revisions_pressure_metadata_check
    CHECK (jsonb_typeof(metadata->'assessment_pressure_limits_Pa') = 'array' AND
           jsonb_array_length(metadata->'assessment_pressure_limits_Pa') = 2 AND
           jsonb_typeof(metadata->'assessment_pressure_limits_Pa'->0) = 'number' AND
           jsonb_typeof(metadata->'assessment_pressure_limits_Pa'->1) = 'number' AND
           (metadata->'assessment_pressure_limits_Pa'->>0)::double precision = assessment_pressure_min_pa AND
           (metadata->'assessment_pressure_limits_Pa'->>1)::double precision = assessment_pressure_max_pa),
  CONSTRAINT control_calphad_revisions_owner_declaration_check
    CHECK (
      metadata ? 'owner_declaration' AND
      jsonb_typeof(metadata->'owner_declaration') = 'object' AND
      metadata->'owner_declaration' = jsonb_build_object(
        'schema_version', metadata->'owner_declaration'->'schema_version',
        'authority', metadata->'owner_declaration'->'authority',
        'database_id', metadata->'owner_declaration'->'database_id',
        'source', metadata->'owner_declaration'->'source',
        'license_id', metadata->'owner_declaration'->'license_id',
        'assessment_scope', metadata->'owner_declaration'->'assessment_scope',
        'reference_state', metadata->'owner_declaration'->'reference_state',
        'assessment_temperature_limits_K', metadata->'owner_declaration'->'assessment_temperature_limits_K',
        'assessment_pressure_limits_Pa', metadata->'owner_declaration'->'assessment_pressure_limits_Pa',
        'database_format', metadata->'owner_declaration'->'database_format'
      ) AND
      jsonb_typeof(metadata->'owner_declaration'->'schema_version') = 'string' AND
      jsonb_typeof(metadata->'owner_declaration'->'authority') = 'string' AND
      jsonb_typeof(metadata->'owner_declaration'->'database_id') = 'string' AND
      jsonb_typeof(metadata->'owner_declaration'->'source') = 'string' AND
      jsonb_typeof(metadata->'owner_declaration'->'license_id') = 'string' AND
      jsonb_typeof(metadata->'owner_declaration'->'assessment_scope') = 'string' AND
      jsonb_typeof(metadata->'owner_declaration'->'reference_state') = 'string' AND
      jsonb_typeof(metadata->'owner_declaration'->'database_format') = 'string' AND
      metadata->'owner_declaration'->>'schema_version' = 'ultra.calphad.owner-declaration.v1' AND
      metadata->'owner_declaration'->>'authority' = 'resource_owner' AND
      metadata->'owner_declaration'->>'database_format' = database_format AND
      char_length(btrim(metadata->'owner_declaration'->>'database_id')) BETWEEN 1 AND 512 AND
      char_length(btrim(metadata->'owner_declaration'->>'source')) BETWEEN 1 AND 1024 AND
      char_length(btrim(metadata->'owner_declaration'->>'license_id')) BETWEEN 1 AND 128 AND
      char_length(btrim(metadata->'owner_declaration'->>'assessment_scope')) BETWEEN 1 AND 1024 AND
      char_length(btrim(metadata->'owner_declaration'->>'reference_state')) BETWEEN 1 AND 512 AND
      jsonb_typeof(metadata->'owner_declaration'->'assessment_temperature_limits_K') = 'array' AND
      jsonb_array_length(metadata->'owner_declaration'->'assessment_temperature_limits_K') = 2 AND
      jsonb_typeof(metadata->'owner_declaration'->'assessment_temperature_limits_K'->0) = 'number' AND
      jsonb_typeof(metadata->'owner_declaration'->'assessment_temperature_limits_K'->1) = 'number' AND
      (metadata->'owner_declaration'->'assessment_temperature_limits_K'->>0)::double precision >= 1 AND
      (metadata->'owner_declaration'->'assessment_temperature_limits_K'->>0)::double precision <
        (metadata->'owner_declaration'->'assessment_temperature_limits_K'->>1)::double precision AND
      (metadata->'owner_declaration'->'assessment_temperature_limits_K'->>1)::double precision <= 10000 AND
      metadata->'owner_declaration'->'assessment_pressure_limits_Pa' =
        jsonb_build_array(assessment_pressure_min_pa, assessment_pressure_max_pa)
    ),
  CONSTRAINT control_calphad_revisions_input_blob_fkey
    FOREIGN KEY (sha256, size_bytes)
    REFERENCES control_calphad_input_blobs(input_sha256, input_size_bytes),
  CHECK (parent_revision_id IS NULL OR parent_revision_id <> revision_id)
);

CREATE INDEX IF NOT EXISTS control_calphad_revisions_owner_created_idx
  ON control_calphad_revisions(owner_user_id, owner_org_id, created_at DESC);
CREATE INDEX IF NOT EXISTS control_calphad_revisions_parent_idx
  ON control_calphad_revisions(parent_revision_id) WHERE parent_revision_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS control_calphad_evidence_blobs (
  evidence_sha256 text PRIMARY KEY CHECK (evidence_sha256 ~ '^[0-9a-f]{64}$'),
  evidence_size_bytes bigint NOT NULL CHECK (evidence_size_bytes BETWEEN 1 AND 33554432),
  encoding text NOT NULL CHECK (encoding = 'raw'),
  payload bytea NOT NULL,
  created_at timestamptz NOT NULL,
  CONSTRAINT control_calphad_evidence_blob_binding_unique
    UNIQUE (evidence_sha256, evidence_size_bytes),
  CONSTRAINT control_calphad_evidence_blob_payload_sha_check
    CHECK (encode(sha256(payload), 'hex') = evidence_sha256),
  CHECK (octet_length(payload) = evidence_size_bytes)
);

CREATE TABLE IF NOT EXISTS control_calphad_validation_events (
  validation_id text PRIMARY KEY,
  revision_id text NOT NULL,
  resource_id text NOT NULL,
  database_sha256 text NOT NULL,
  database_size_bytes bigint NOT NULL,
  database_format text NOT NULL,
  assessment_pressure_min_pa double precision,
  assessment_pressure_max_pa double precision,
  database_inventory_sha256 text,
  request_sha256 text,
  status text NOT NULL,
  operation text NOT NULL,
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
  created_by_authority text NOT NULL,
  created_at timestamptz NOT NULL,
  metadata jsonb NOT NULL DEFAULT '{}',
  CONSTRAINT control_calphad_validation_status_check
    CHECK (status IN ('pending', 'input_validated', 'equilibrium_completed', 'scheil_completed', 'failed', 'timeout', 'unsupported')),
  CONSTRAINT control_calphad_validation_operation_check
    CHECK (operation IN ('registration', 'inspect', 'equilibrium', 'scheil')),
  CONSTRAINT control_calphad_validation_evidence_sha_check
    CHECK (evidence_sha256 IS NULL OR evidence_sha256 ~ '^[0-9a-f]{64}$'),
  CONSTRAINT control_calphad_validation_evidence_size_check
    CHECK (evidence_size_bytes IS NULL OR evidence_size_bytes BETWEEN 1 AND 33554432),
  CONSTRAINT control_calphad_validation_runtime_image_check
    CHECK (runtime_image_id IS NULL OR runtime_image_id ~ '^sha256:[0-9a-f]{64}$'),
  CONSTRAINT control_calphad_validation_inspection_sha_check
    CHECK (inspection_evidence_sha256 IS NULL OR inspection_evidence_sha256 ~ '^[0-9a-f]{64}$'),
  CONSTRAINT control_calphad_validation_inventory_sha_check
    CHECK (database_inventory_sha256 IS NULL OR database_inventory_sha256 ~ '^[0-9a-f]{64}$'),
  CONSTRAINT control_calphad_validation_request_sha_check
    CHECK (request_sha256 IS NULL OR request_sha256 ~ '^[0-9a-f]{64}$'),
  CONSTRAINT control_calphad_validation_evidence_contract_check
    CHECK (evidence_contract_version IS NULL OR
           evidence_contract_version = 'ultra.calphad.retained-evidence.v2'),
  CONSTRAINT control_calphad_validation_authority_check
    CHECK (created_by_authority IN ('control_plane', 'trusted_worker')),
  CONSTRAINT control_calphad_validation_metadata_check
    CHECK (jsonb_typeof(metadata) = 'object'),
  CONSTRAINT control_calphad_validation_database_sha_check
    CHECK (database_sha256 ~ '^[0-9a-f]{64}$'),
  CONSTRAINT control_calphad_validation_database_size_check
    CHECK (database_size_bytes > 0),
  CONSTRAINT control_calphad_validation_database_format_check
    CHECK (database_format IN ('tdb', 'dat')),
  CONSTRAINT control_calphad_validation_pressure_limits_check
    CHECK (assessment_pressure_min_pa IS NOT NULL AND assessment_pressure_max_pa IS NOT NULL AND
           assessment_pressure_min_pa >= 1e-9 AND assessment_pressure_max_pa <= 1e12 AND
           assessment_pressure_min_pa <= assessment_pressure_max_pa),
  CONSTRAINT control_calphad_validation_pressure_metadata_check
    CHECK (jsonb_typeof(metadata->'assessment_pressure_limits_Pa') = 'array' AND
           jsonb_array_length(metadata->'assessment_pressure_limits_Pa') = 2 AND
           jsonb_typeof(metadata->'assessment_pressure_limits_Pa'->0) = 'number' AND
           jsonb_typeof(metadata->'assessment_pressure_limits_Pa'->1) = 'number' AND
           (metadata->'assessment_pressure_limits_Pa'->>0)::double precision = assessment_pressure_min_pa AND
           (metadata->'assessment_pressure_limits_Pa'->>1)::double precision = assessment_pressure_max_pa),
  CONSTRAINT control_calphad_validation_revision_binding_fkey
  FOREIGN KEY (revision_id, resource_id, database_sha256, database_size_bytes, database_format)
    REFERENCES control_calphad_revisions(revision_id, resource_id, sha256, size_bytes, database_format),
  CONSTRAINT control_calphad_validation_pressure_binding_fkey
  FOREIGN KEY (revision_id, assessment_pressure_min_pa, assessment_pressure_max_pa)
    REFERENCES control_calphad_revisions
      (revision_id, assessment_pressure_min_pa, assessment_pressure_max_pa),
  CONSTRAINT control_calphad_validation_evidence_blob_fkey
  FOREIGN KEY (evidence_sha256, evidence_size_bytes)
    REFERENCES control_calphad_evidence_blobs(evidence_sha256, evidence_size_bytes),
  CONSTRAINT control_calphad_validation_run_fkey
  FOREIGN KEY (run_id) REFERENCES control_runs(run_id),
  CONSTRAINT control_calphad_validation_inspection_blob_fkey
  FOREIGN KEY (inspection_evidence_sha256) REFERENCES control_calphad_evidence_blobs(evidence_sha256),
  CONSTRAINT control_calphad_validation_inspection_lineage_check
    CHECK ((operation IN ('equilibrium', 'scheil')) = (inspection_evidence_sha256 IS NOT NULL)),
  CONSTRAINT control_calphad_validation_pycalphad_version_check
    CHECK (operation = 'registration' OR pycalphad_version = '0.11.2'),
  CONSTRAINT control_calphad_validation_worker_identity_check
    CHECK ((operation = 'registration' AND database_inventory_sha256 IS NULL AND
            request_sha256 IS NULL AND evidence_contract_version IS NULL) OR
           (operation <> 'registration' AND request_sha256 IS NOT NULL AND
            (database_inventory_sha256 IS NOT NULL OR
             (operation = 'inspect' AND status IN ('failed', 'timeout', 'unsupported'))) AND
            evidence_contract_version = 'ultra.calphad.retained-evidence.v2')),
  CONSTRAINT control_calphad_validation_registration_status_check
    CHECK ((operation = 'registration') = (status = 'pending')),
  CONSTRAINT control_calphad_validation_registration_authority_check
    CHECK ((operation = 'registration') = (created_by_authority = 'control_plane')),
  CONSTRAINT control_calphad_validation_input_operation_check
    CHECK (status <> 'input_validated' OR operation = 'inspect'),
  CONSTRAINT control_calphad_validation_equilibrium_operation_check
    CHECK (status <> 'equilibrium_completed' OR operation = 'equilibrium'),
  CONSTRAINT control_calphad_validation_scheil_operation_check
    CHECK (status <> 'scheil_completed' OR operation = 'scheil'),
  CONSTRAINT control_calphad_validation_evidence_tuple_check
    CHECK ((evidence_path IS NULL AND evidence_sha256 IS NULL AND evidence_size_bytes IS NULL) OR
         (evidence_path IS NOT NULL AND evidence_sha256 IS NOT NULL AND evidence_size_bytes IS NOT NULL)),
  CONSTRAINT control_calphad_validation_retained_evidence_check
    CHECK (operation = 'registration' OR evidence_path IS NOT NULL),
  CONSTRAINT control_calphad_validation_failure_tuple_check
    CHECK (
      (status NOT IN ('failed', 'timeout', 'unsupported') AND
       failure_domain IS NULL AND failure_stage IS NULL AND failure_code IS NULL) OR
      (status IN ('failed', 'timeout', 'unsupported') AND
       failure_domain IS NOT NULL AND failure_stage IS NOT NULL AND failure_code IS NOT NULL AND
       failure_domain IN ('input', 'scientific', 'platform') AND
       failure_stage IN ('parse', 'solver', 'result_validation', 'sandbox_runtime') AND
       failure_code IN (
         'calphad_parse_failed', 'calphad_parse_timeout', 'calphad_parse_unsupported',
         'calphad_solver_failed', 'calphad_solver_timeout', 'calphad_solver_unsupported',
         'calphad_result_invalid', 'calphad_runtime_internal_failure',
         'calphad_sandbox_failed', 'calphad_sandbox_timeout'
       ) AND (
         (failure_code = 'calphad_parse_failed' AND status = 'failed' AND
          failure_domain IN ('input', 'scientific') AND failure_stage = 'parse' AND operation = 'inspect') OR
         (failure_code = 'calphad_parse_timeout' AND status = 'timeout' AND
          failure_domain = 'scientific' AND failure_stage = 'parse' AND operation = 'inspect') OR
         (failure_code = 'calphad_parse_unsupported' AND status = 'unsupported' AND
          failure_domain = 'input' AND failure_stage = 'parse' AND operation = 'inspect') OR
         (failure_code = 'calphad_solver_failed' AND status = 'failed' AND
          failure_domain IN ('input', 'scientific') AND failure_stage = 'solver' AND
          operation IN ('equilibrium', 'scheil')) OR
         (failure_code = 'calphad_solver_timeout' AND status = 'timeout' AND
          failure_domain = 'scientific' AND failure_stage = 'solver' AND
          operation IN ('equilibrium', 'scheil')) OR
         (failure_code = 'calphad_solver_unsupported' AND status = 'unsupported' AND
          failure_domain = 'scientific' AND failure_stage = 'solver' AND
          operation IN ('equilibrium', 'scheil')) OR
         (failure_code = 'calphad_result_invalid' AND status = 'failed' AND
          failure_domain = 'scientific' AND failure_stage = 'result_validation') OR
         (failure_code = 'calphad_runtime_internal_failure' AND status = 'failed' AND
          failure_domain = 'platform' AND
          ((operation = 'inspect' AND failure_stage = 'parse') OR
           (operation IN ('equilibrium', 'scheil') AND failure_stage = 'solver'))) OR
         (failure_code = 'calphad_sandbox_failed' AND status = 'failed' AND
          failure_domain = 'platform' AND failure_stage = 'sandbox_runtime') OR
         (failure_code = 'calphad_sandbox_timeout' AND status = 'timeout' AND
          failure_domain = 'platform' AND failure_stage = 'sandbox_runtime')
       ))
    ),
  CONSTRAINT control_calphad_validation_evidence_path_check
    CHECK (evidence_path IS NULL OR evidence_path = '/outputs/calphad/' ||
    CASE operation WHEN 'inspect' THEN 'inspection' ELSE operation END ||
    '/' || evidence_sha256 || '.json'),
  CONSTRAINT control_calphad_validation_runtime_binding_check
    CHECK ((operation = 'registration' AND evidence_path IS NULL AND evidence_sha256 IS NULL AND
          evidence_size_bytes IS NULL AND runtime_image_id IS NULL AND pycalphad_version IS NULL AND run_id IS NULL) OR
         (operation <> 'registration' AND runtime_image_id IS NOT NULL AND
          pycalphad_version IS NOT NULL AND run_id IS NOT NULL AND
          char_length(btrim(pycalphad_version)) BETWEEN 1 AND 128 AND
          char_length(btrim(run_id)) BETWEEN 1 AND 512))
);

-- Compatibility for a database on which an earlier IF-NOT-EXISTS draft of
-- this schema was applied. Backfill only immutable revision facts; evidence
-- bytes cannot be reconstructed. The input-blob FK remains NOT VALID for
-- historical revisions while still being enforced for every new insert;
-- ledger reads and promotion fail closed until exact input is retained.
DROP TRIGGER IF EXISTS control_calphad_validation_append_only ON control_calphad_validation_events;
ALTER TABLE control_calphad_revisions
  ADD COLUMN IF NOT EXISTS assessment_pressure_min_pa double precision,
  ADD COLUMN IF NOT EXISTS assessment_pressure_max_pa double precision,
  ADD COLUMN IF NOT EXISTS database_format text;
ALTER TABLE control_calphad_validation_events
  ADD COLUMN IF NOT EXISTS database_sha256 text,
  ADD COLUMN IF NOT EXISTS database_size_bytes bigint,
  ADD COLUMN IF NOT EXISTS inspection_evidence_sha256 text,
  ADD COLUMN IF NOT EXISTS database_inventory_sha256 text,
  ADD COLUMN IF NOT EXISTS request_sha256 text,
  ADD COLUMN IF NOT EXISTS evidence_contract_version text,
  ADD COLUMN IF NOT EXISTS failure_domain text,
  ADD COLUMN IF NOT EXISTS failure_stage text,
  ADD COLUMN IF NOT EXISTS failure_code text,
  ADD COLUMN IF NOT EXISTS assessment_pressure_min_pa double precision,
  ADD COLUMN IF NOT EXISTS assessment_pressure_max_pa double precision,
  ADD COLUMN IF NOT EXISTS database_format text;
UPDATE control_calphad_validation_events validation
SET database_sha256 = revision.sha256,
    database_size_bytes = revision.size_bytes
FROM control_calphad_revisions revision
WHERE validation.revision_id = revision.revision_id
  AND (validation.database_sha256 IS NULL OR validation.database_size_bytes IS NULL);
ALTER TABLE control_calphad_validation_events
  ALTER COLUMN database_sha256 SET NOT NULL,
  ALTER COLUMN database_size_bytes SET NOT NULL;

ALTER TABLE control_calphad_validation_events
  DROP CONSTRAINT IF EXISTS control_calphad_validation_revision_binding_fkey,
  DROP CONSTRAINT IF EXISTS control_calphad_validation_evidence_contract_check,
  DROP CONSTRAINT IF EXISTS control_calphad_validation_worker_identity_check,
  DROP CONSTRAINT IF EXISTS control_calphad_validation_status_check,
  DROP CONSTRAINT IF EXISTS control_calphad_validation_operation_check,
  DROP CONSTRAINT IF EXISTS control_calphad_validation_success_evidence_check,
  DROP CONSTRAINT IF EXISTS control_calphad_validation_inspection_lineage_check,
  DROP CONSTRAINT IF EXISTS control_calphad_validation_equilibrium_operation_check,
  DROP CONSTRAINT IF EXISTS control_calphad_validation_scheil_operation_check,
  DROP CONSTRAINT IF EXISTS control_calphad_validation_evidence_path_check,
  DROP CONSTRAINT IF EXISTS control_calphad_validation_retained_evidence_check,
  DROP CONSTRAINT IF EXISTS control_calphad_validation_failure_tuple_check;
ALTER TABLE control_calphad_revisions
  DROP CONSTRAINT IF EXISTS control_calphad_revisions_owner_declaration_check,
  DROP CONSTRAINT IF EXISTS control_calphad_revisions_binding_unique;

DO $$
DECLARE
  check_record record;
BEGIN
  FOR check_record IN
    SELECT * FROM (VALUES
      ('control_calphad_validation_status_check', $check$CHECK (status IN ('pending', 'input_validated', 'equilibrium_completed', 'scheil_completed', 'failed', 'timeout', 'unsupported'))$check$),
      ('control_calphad_validation_operation_check', $check$CHECK (operation IN ('registration', 'inspect', 'equilibrium', 'scheil'))$check$),
      ('control_calphad_validation_evidence_sha_check', $check$CHECK (evidence_sha256 IS NULL OR evidence_sha256 ~ '^[0-9a-f]{64}$')$check$),
      ('control_calphad_validation_evidence_size_check', $check$CHECK (evidence_size_bytes IS NULL OR evidence_size_bytes BETWEEN 1 AND 33554432)$check$),
      ('control_calphad_validation_runtime_image_check', $check$CHECK (runtime_image_id IS NULL OR runtime_image_id ~ '^sha256:[0-9a-f]{64}$')$check$),
      ('control_calphad_validation_inspection_sha_check', $check$CHECK (inspection_evidence_sha256 IS NULL OR inspection_evidence_sha256 ~ '^[0-9a-f]{64}$')$check$),
      ('control_calphad_validation_inventory_sha_check', $check$CHECK (database_inventory_sha256 IS NULL OR database_inventory_sha256 ~ '^[0-9a-f]{64}$')$check$),
      ('control_calphad_validation_request_sha_check', $check$CHECK (request_sha256 IS NULL OR request_sha256 ~ '^[0-9a-f]{64}$')$check$),
      ('control_calphad_validation_evidence_contract_check', $check$CHECK (evidence_contract_version IS NULL OR evidence_contract_version = 'ultra.calphad.retained-evidence.v2')$check$),
      ('control_calphad_validation_authority_check', $check$CHECK (created_by_authority IN ('control_plane', 'trusted_worker'))$check$),
      ('control_calphad_validation_metadata_check', $check$CHECK (jsonb_typeof(metadata) = 'object')$check$),
      ('control_calphad_validation_registration_status_check', $check$CHECK ((operation = 'registration') = (status = 'pending'))$check$),
      ('control_calphad_validation_registration_authority_check', $check$CHECK ((operation = 'registration') = (created_by_authority = 'control_plane'))$check$),
      ('control_calphad_validation_input_operation_check', $check$CHECK (status <> 'input_validated' OR operation = 'inspect')$check$),
      ('control_calphad_validation_equilibrium_operation_check', $check$CHECK (status <> 'equilibrium_completed' OR operation = 'equilibrium')$check$),
      ('control_calphad_validation_scheil_operation_check', $check$CHECK (status <> 'scheil_completed' OR operation = 'scheil')$check$),
      ('control_calphad_validation_evidence_tuple_check', $check$CHECK ((evidence_path IS NULL AND evidence_sha256 IS NULL AND evidence_size_bytes IS NULL) OR (evidence_path IS NOT NULL AND evidence_sha256 IS NOT NULL AND evidence_size_bytes IS NOT NULL))$check$),
      ('control_calphad_validation_retained_evidence_check', $check$CHECK (operation = 'registration' OR evidence_path IS NOT NULL)$check$),
      ('control_calphad_validation_evidence_path_check', $check$CHECK (evidence_path IS NULL OR evidence_path = '/outputs/calphad/' || CASE operation WHEN 'inspect' THEN 'inspection' ELSE operation END || '/' || evidence_sha256 || '.json')$check$),
      ('control_calphad_validation_runtime_binding_check', $check$CHECK ((operation = 'registration' AND evidence_path IS NULL AND evidence_sha256 IS NULL AND evidence_size_bytes IS NULL AND runtime_image_id IS NULL AND pycalphad_version IS NULL AND run_id IS NULL) OR (operation <> 'registration' AND runtime_image_id IS NOT NULL AND pycalphad_version IS NOT NULL AND run_id IS NOT NULL AND char_length(btrim(pycalphad_version)) BETWEEN 1 AND 128 AND char_length(btrim(run_id)) BETWEEN 1 AND 512))$check$),
      ('control_calphad_validation_worker_identity_check', $check$CHECK ((operation = 'registration' AND database_inventory_sha256 IS NULL AND request_sha256 IS NULL AND evidence_contract_version IS NULL) OR (operation <> 'registration' AND request_sha256 IS NOT NULL AND (database_inventory_sha256 IS NOT NULL OR (operation = 'inspect' AND status IN ('failed', 'timeout', 'unsupported'))) AND evidence_contract_version = 'ultra.calphad.retained-evidence.v2'))$check$),
      ('control_calphad_validation_failure_tuple_check', $check$CHECK ((status NOT IN ('failed', 'timeout', 'unsupported') AND failure_domain IS NULL AND failure_stage IS NULL AND failure_code IS NULL) OR (status IN ('failed', 'timeout', 'unsupported') AND failure_domain IS NOT NULL AND failure_stage IS NOT NULL AND failure_code IS NOT NULL AND failure_domain IN ('input', 'scientific', 'platform') AND failure_stage IN ('parse', 'solver', 'result_validation', 'sandbox_runtime') AND failure_code IN ('calphad_parse_failed', 'calphad_parse_timeout', 'calphad_parse_unsupported', 'calphad_solver_failed', 'calphad_solver_timeout', 'calphad_solver_unsupported', 'calphad_result_invalid', 'calphad_runtime_internal_failure', 'calphad_sandbox_failed', 'calphad_sandbox_timeout') AND ((failure_code = 'calphad_parse_failed' AND status = 'failed' AND failure_domain IN ('input', 'scientific') AND failure_stage = 'parse' AND operation = 'inspect') OR (failure_code = 'calphad_parse_timeout' AND status = 'timeout' AND failure_domain = 'scientific' AND failure_stage = 'parse' AND operation = 'inspect') OR (failure_code = 'calphad_parse_unsupported' AND status = 'unsupported' AND failure_domain = 'input' AND failure_stage = 'parse' AND operation = 'inspect') OR (failure_code = 'calphad_solver_failed' AND status = 'failed' AND failure_domain IN ('input', 'scientific') AND failure_stage = 'solver' AND operation IN ('equilibrium', 'scheil')) OR (failure_code = 'calphad_solver_timeout' AND status = 'timeout' AND failure_domain = 'scientific' AND failure_stage = 'solver' AND operation IN ('equilibrium', 'scheil')) OR (failure_code = 'calphad_solver_unsupported' AND status = 'unsupported' AND failure_domain = 'scientific' AND failure_stage = 'solver' AND operation IN ('equilibrium', 'scheil')) OR (failure_code = 'calphad_result_invalid' AND status = 'failed' AND failure_domain = 'scientific' AND failure_stage = 'result_validation') OR (failure_code = 'calphad_runtime_internal_failure' AND status = 'failed' AND failure_domain = 'platform' AND ((operation = 'inspect' AND failure_stage = 'parse') OR (operation IN ('equilibrium', 'scheil') AND failure_stage = 'solver'))) OR (failure_code = 'calphad_sandbox_failed' AND status = 'failed' AND failure_domain = 'platform' AND failure_stage = 'sandbox_runtime') OR (failure_code = 'calphad_sandbox_timeout' AND status = 'timeout' AND failure_domain = 'platform' AND failure_stage = 'sandbox_runtime'))))$check$)
      ,('control_calphad_validation_pressure_limits_check', $check$CHECK (assessment_pressure_min_pa IS NOT NULL AND assessment_pressure_max_pa IS NOT NULL AND assessment_pressure_min_pa >= 1e-9 AND assessment_pressure_max_pa <= 1e12 AND assessment_pressure_min_pa <= assessment_pressure_max_pa)$check$)
      ,('control_calphad_validation_pressure_metadata_check', $check$CHECK (jsonb_typeof(metadata->'assessment_pressure_limits_Pa') = 'array' AND jsonb_array_length(metadata->'assessment_pressure_limits_Pa') = 2 AND jsonb_typeof(metadata->'assessment_pressure_limits_Pa'->0) = 'number' AND jsonb_typeof(metadata->'assessment_pressure_limits_Pa'->1) = 'number' AND (metadata->'assessment_pressure_limits_Pa'->>0)::double precision = assessment_pressure_min_pa AND (metadata->'assessment_pressure_limits_Pa'->>1)::double precision = assessment_pressure_max_pa)$check$)
      ,('control_calphad_validation_database_format_check', $check$CHECK (database_format IS NOT NULL AND database_format IN ('tdb', 'dat'))$check$)
    ) AS checks(constraint_name, definition)
  LOOP
    IF NOT EXISTS (
      SELECT 1 FROM pg_constraint
      WHERE conrelid = 'control_calphad_validation_events'::regclass
        AND conname = check_record.constraint_name
    ) THEN
      EXECUTE format(
        'ALTER TABLE control_calphad_validation_events ADD CONSTRAINT %I %s NOT VALID',
        check_record.constraint_name,
        check_record.definition
      );
    END IF;
  END LOOP;
  IF NOT EXISTS (
	SELECT 1 FROM pg_constraint
	WHERE conrelid = 'control_calphad_revisions'::regclass
	  AND conname = 'control_calphad_revisions_pressure_binding_unique'
  ) THEN
	ALTER TABLE control_calphad_revisions
	  ADD CONSTRAINT control_calphad_revisions_pressure_binding_unique
	  UNIQUE (revision_id, assessment_pressure_min_pa, assessment_pressure_max_pa);
  END IF;
  IF NOT EXISTS (
	SELECT 1 FROM pg_constraint
	WHERE conrelid = 'control_calphad_revisions'::regclass
	  AND conname = 'control_calphad_revisions_pressure_limits_check'
  ) THEN
	ALTER TABLE control_calphad_revisions
	  ADD CONSTRAINT control_calphad_revisions_pressure_limits_check
	  CHECK (assessment_pressure_min_pa IS NOT NULL AND assessment_pressure_max_pa IS NOT NULL AND
	         assessment_pressure_min_pa >= 1e-9 AND assessment_pressure_max_pa <= 1e12 AND
	         assessment_pressure_min_pa <= assessment_pressure_max_pa) NOT VALID;
  END IF;
  IF NOT EXISTS (
	SELECT 1 FROM pg_constraint
	WHERE conrelid = 'control_calphad_revisions'::regclass
	  AND conname = 'control_calphad_revisions_pressure_metadata_check'
  ) THEN
	ALTER TABLE control_calphad_revisions
	  ADD CONSTRAINT control_calphad_revisions_pressure_metadata_check
	  CHECK (jsonb_typeof(metadata->'assessment_pressure_limits_Pa') = 'array' AND
	         jsonb_array_length(metadata->'assessment_pressure_limits_Pa') = 2 AND
	         jsonb_typeof(metadata->'assessment_pressure_limits_Pa'->0) = 'number' AND
	         jsonb_typeof(metadata->'assessment_pressure_limits_Pa'->1) = 'number' AND
	         (metadata->'assessment_pressure_limits_Pa'->>0)::double precision = assessment_pressure_min_pa AND
	         (metadata->'assessment_pressure_limits_Pa'->>1)::double precision = assessment_pressure_max_pa) NOT VALID;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'control_calphad_revisions'::regclass
      AND conname = 'control_calphad_revisions_database_format_check'
  ) THEN
    ALTER TABLE control_calphad_revisions
      ADD CONSTRAINT control_calphad_revisions_database_format_check
      CHECK (database_format IS NOT NULL AND database_format IN ('tdb', 'dat')) NOT VALID;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'control_calphad_revisions'::regclass
      AND conname = 'control_calphad_revisions_owner_declaration_check'
  ) THEN
    ALTER TABLE control_calphad_revisions
      ADD CONSTRAINT control_calphad_revisions_owner_declaration_check
      CHECK (
        metadata ? 'owner_declaration' AND
        jsonb_typeof(metadata->'owner_declaration') = 'object' AND
        metadata->'owner_declaration' = jsonb_build_object(
          'schema_version', metadata->'owner_declaration'->'schema_version',
          'authority', metadata->'owner_declaration'->'authority',
          'database_id', metadata->'owner_declaration'->'database_id',
          'source', metadata->'owner_declaration'->'source',
          'license_id', metadata->'owner_declaration'->'license_id',
          'assessment_scope', metadata->'owner_declaration'->'assessment_scope',
          'reference_state', metadata->'owner_declaration'->'reference_state',
          'assessment_temperature_limits_K', metadata->'owner_declaration'->'assessment_temperature_limits_K',
          'assessment_pressure_limits_Pa', metadata->'owner_declaration'->'assessment_pressure_limits_Pa',
          'database_format', metadata->'owner_declaration'->'database_format'
        ) AND
        jsonb_typeof(metadata->'owner_declaration'->'schema_version') = 'string' AND
        jsonb_typeof(metadata->'owner_declaration'->'authority') = 'string' AND
        jsonb_typeof(metadata->'owner_declaration'->'database_id') = 'string' AND
        jsonb_typeof(metadata->'owner_declaration'->'source') = 'string' AND
        jsonb_typeof(metadata->'owner_declaration'->'license_id') = 'string' AND
        jsonb_typeof(metadata->'owner_declaration'->'assessment_scope') = 'string' AND
        jsonb_typeof(metadata->'owner_declaration'->'reference_state') = 'string' AND
        jsonb_typeof(metadata->'owner_declaration'->'database_format') = 'string' AND
        metadata->'owner_declaration'->>'schema_version' = 'ultra.calphad.owner-declaration.v1' AND
        metadata->'owner_declaration'->>'authority' = 'resource_owner' AND
        metadata->'owner_declaration'->>'database_format' = database_format AND
        char_length(btrim(metadata->'owner_declaration'->>'database_id')) BETWEEN 1 AND 512 AND
        char_length(btrim(metadata->'owner_declaration'->>'source')) BETWEEN 1 AND 1024 AND
        char_length(btrim(metadata->'owner_declaration'->>'license_id')) BETWEEN 1 AND 128 AND
        char_length(btrim(metadata->'owner_declaration'->>'assessment_scope')) BETWEEN 1 AND 1024 AND
        char_length(btrim(metadata->'owner_declaration'->>'reference_state')) BETWEEN 1 AND 512 AND
        jsonb_typeof(metadata->'owner_declaration'->'assessment_temperature_limits_K') = 'array' AND
        jsonb_array_length(metadata->'owner_declaration'->'assessment_temperature_limits_K') = 2 AND
        jsonb_typeof(metadata->'owner_declaration'->'assessment_temperature_limits_K'->0) = 'number' AND
        jsonb_typeof(metadata->'owner_declaration'->'assessment_temperature_limits_K'->1) = 'number' AND
        (metadata->'owner_declaration'->'assessment_temperature_limits_K'->>0)::double precision >= 1 AND
        (metadata->'owner_declaration'->'assessment_temperature_limits_K'->>0)::double precision <
          (metadata->'owner_declaration'->'assessment_temperature_limits_K'->>1)::double precision AND
        (metadata->'owner_declaration'->'assessment_temperature_limits_K'->>1)::double precision <= 10000 AND
        metadata->'owner_declaration'->'assessment_pressure_limits_Pa' =
          jsonb_build_array(assessment_pressure_min_pa, assessment_pressure_max_pa)
      ) NOT VALID;
  END IF;
  IF NOT EXISTS (
	SELECT 1 FROM pg_constraint
	WHERE conrelid = 'control_calphad_input_blobs'::regclass
	  AND conname = 'control_calphad_input_blob_binding_unique'
  ) THEN
	ALTER TABLE control_calphad_input_blobs
	  ADD CONSTRAINT control_calphad_input_blob_binding_unique
	  UNIQUE (input_sha256, input_size_bytes);
  END IF;
  IF NOT EXISTS (
	SELECT 1 FROM pg_constraint
	WHERE conrelid = 'control_calphad_input_blobs'::regclass
	  AND conname = 'control_calphad_input_blob_size_check'
  ) THEN
	ALTER TABLE control_calphad_input_blobs
	  ADD CONSTRAINT control_calphad_input_blob_size_check
	  CHECK (input_size_bytes BETWEEN 1 AND 67108864) NOT VALID;
  END IF;
  IF NOT EXISTS (
	SELECT 1 FROM pg_constraint
	WHERE conrelid = 'control_calphad_input_blobs'::regclass
	  AND conname = 'control_calphad_input_blob_encoding_check'
  ) THEN
	ALTER TABLE control_calphad_input_blobs
	  ADD CONSTRAINT control_calphad_input_blob_encoding_check
	  CHECK (encoding = 'raw') NOT VALID;
  END IF;
  IF NOT EXISTS (
	SELECT 1 FROM pg_constraint
	WHERE conrelid = 'control_calphad_input_blobs'::regclass
	  AND conname = 'control_calphad_input_blob_payload_sha_check'
  ) THEN
	ALTER TABLE control_calphad_input_blobs
	  ADD CONSTRAINT control_calphad_input_blob_payload_sha_check
	  CHECK (encode(sha256(payload), 'hex') = input_sha256) NOT VALID;
  END IF;
  IF NOT EXISTS (
	SELECT 1 FROM pg_constraint
	WHERE conrelid = 'control_calphad_input_blobs'::regclass
	  AND conname = 'control_calphad_input_blob_payload_size_check'
  ) THEN
	ALTER TABLE control_calphad_input_blobs
	  ADD CONSTRAINT control_calphad_input_blob_payload_size_check
	  CHECK (octet_length(payload) = input_size_bytes) NOT VALID;
  END IF;
  IF NOT EXISTS (
	SELECT 1 FROM pg_constraint
	WHERE conrelid = 'control_calphad_validation_events'::regclass
	  AND conname = 'control_calphad_validation_pressure_binding_fkey'
  ) THEN
	ALTER TABLE control_calphad_validation_events
	  ADD CONSTRAINT control_calphad_validation_pressure_binding_fkey
	  FOREIGN KEY (revision_id, assessment_pressure_min_pa, assessment_pressure_max_pa)
	  REFERENCES control_calphad_revisions
	    (revision_id, assessment_pressure_min_pa, assessment_pressure_max_pa)
	  NOT VALID;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'control_calphad_evidence_blobs'::regclass
      AND conname = 'control_calphad_evidence_blob_payload_sha_check'
  ) THEN
    ALTER TABLE control_calphad_evidence_blobs
      ADD CONSTRAINT control_calphad_evidence_blob_payload_sha_check
      CHECK (encode(sha256(payload), 'hex') = evidence_sha256) NOT VALID;
  END IF;
  IF NOT EXISTS (
	SELECT 1 FROM pg_constraint
	WHERE conrelid = 'control_calphad_revisions'::regclass
	  AND conname = 'control_calphad_revisions_input_blob_fkey'
  ) THEN
	ALTER TABLE control_calphad_revisions
	  ADD CONSTRAINT control_calphad_revisions_input_blob_fkey
	  FOREIGN KEY (sha256, size_bytes)
	  REFERENCES control_calphad_input_blobs(input_sha256, input_size_bytes)
	  NOT VALID;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'control_calphad_revisions'::regclass
      AND conname = 'control_calphad_revisions_binding_unique'
  ) THEN
    ALTER TABLE control_calphad_revisions
      ADD CONSTRAINT control_calphad_revisions_binding_unique
      UNIQUE (revision_id, resource_id, sha256, size_bytes, database_format);
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'control_calphad_validation_events'::regclass
      AND conname = 'control_calphad_validation_database_sha_check'
  ) THEN
    ALTER TABLE control_calphad_validation_events
      ADD CONSTRAINT control_calphad_validation_database_sha_check
      CHECK (database_sha256 ~ '^[0-9a-f]{64}$') NOT VALID;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'control_calphad_validation_events'::regclass
      AND conname = 'control_calphad_validation_database_size_check'
  ) THEN
    ALTER TABLE control_calphad_validation_events
      ADD CONSTRAINT control_calphad_validation_database_size_check
      CHECK (database_size_bytes > 0) NOT VALID;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'control_calphad_validation_events'::regclass
      AND conname = 'control_calphad_validation_revision_binding_fkey'
  ) THEN
    ALTER TABLE control_calphad_validation_events
      ADD CONSTRAINT control_calphad_validation_revision_binding_fkey
      FOREIGN KEY (revision_id, resource_id, database_sha256, database_size_bytes, database_format)
      REFERENCES control_calphad_revisions(revision_id, resource_id, sha256, size_bytes, database_format)
      NOT VALID;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'control_calphad_validation_events'::regclass
      AND conname = 'control_calphad_validation_evidence_blob_fkey'
  ) THEN
    ALTER TABLE control_calphad_validation_events
      ADD CONSTRAINT control_calphad_validation_evidence_blob_fkey
      FOREIGN KEY (evidence_sha256, evidence_size_bytes)
      REFERENCES control_calphad_evidence_blobs(evidence_sha256, evidence_size_bytes)
      NOT VALID;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'control_calphad_validation_events'::regclass
      AND conname = 'control_calphad_validation_run_fkey'
  ) THEN
    ALTER TABLE control_calphad_validation_events
      ADD CONSTRAINT control_calphad_validation_run_fkey
      FOREIGN KEY (run_id) REFERENCES control_runs(run_id) NOT VALID;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'control_calphad_validation_events'::regclass
      AND conname = 'control_calphad_validation_inspection_blob_fkey'
  ) THEN
    ALTER TABLE control_calphad_validation_events
      ADD CONSTRAINT control_calphad_validation_inspection_blob_fkey
      FOREIGN KEY (inspection_evidence_sha256)
      REFERENCES control_calphad_evidence_blobs(evidence_sha256) NOT VALID;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'control_calphad_validation_events'::regclass
      AND conname = 'control_calphad_validation_inspection_lineage_check'
  ) THEN
    ALTER TABLE control_calphad_validation_events
      ADD CONSTRAINT control_calphad_validation_inspection_lineage_check
      CHECK ((operation IN ('equilibrium', 'scheil')) =
             (inspection_evidence_sha256 IS NOT NULL)) NOT VALID;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'control_calphad_validation_events'::regclass
      AND conname = 'control_calphad_validation_pycalphad_version_check'
  ) THEN
    ALTER TABLE control_calphad_validation_events
      ADD CONSTRAINT control_calphad_validation_pycalphad_version_check
      CHECK (operation = 'registration' OR pycalphad_version = '0.11.2') NOT VALID;
  END IF;
END;
$$;
ALTER TABLE control_calphad_revisions
  VALIDATE CONSTRAINT control_calphad_revisions_owner_declaration_check;
ALTER TABLE control_calphad_validation_events
  VALIDATE CONSTRAINT control_calphad_validation_status_check,
  VALIDATE CONSTRAINT control_calphad_validation_operation_check,
  VALIDATE CONSTRAINT control_calphad_validation_evidence_sha_check,
  VALIDATE CONSTRAINT control_calphad_validation_evidence_size_check,
  VALIDATE CONSTRAINT control_calphad_validation_runtime_image_check,
  VALIDATE CONSTRAINT control_calphad_validation_inspection_sha_check,
  VALIDATE CONSTRAINT control_calphad_validation_inventory_sha_check,
  VALIDATE CONSTRAINT control_calphad_validation_request_sha_check,
  VALIDATE CONSTRAINT control_calphad_validation_authority_check,
  VALIDATE CONSTRAINT control_calphad_validation_metadata_check,
  VALIDATE CONSTRAINT control_calphad_validation_registration_status_check,
  VALIDATE CONSTRAINT control_calphad_validation_registration_authority_check,
  VALIDATE CONSTRAINT control_calphad_validation_input_operation_check,
  VALIDATE CONSTRAINT control_calphad_validation_equilibrium_operation_check,
  VALIDATE CONSTRAINT control_calphad_validation_scheil_operation_check,
  VALIDATE CONSTRAINT control_calphad_validation_evidence_tuple_check,
  VALIDATE CONSTRAINT control_calphad_validation_retained_evidence_check,
  VALIDATE CONSTRAINT control_calphad_validation_failure_tuple_check,
  VALIDATE CONSTRAINT control_calphad_validation_worker_identity_check,
  VALIDATE CONSTRAINT control_calphad_validation_evidence_contract_check,
  VALIDATE CONSTRAINT control_calphad_validation_evidence_path_check,
  VALIDATE CONSTRAINT control_calphad_validation_runtime_binding_check,
  VALIDATE CONSTRAINT control_calphad_validation_database_sha_check,
  VALIDATE CONSTRAINT control_calphad_validation_database_size_check,
  VALIDATE CONSTRAINT control_calphad_validation_revision_binding_fkey,
  VALIDATE CONSTRAINT control_calphad_validation_inspection_blob_fkey,
  VALIDATE CONSTRAINT control_calphad_validation_inspection_lineage_check,
  VALIDATE CONSTRAINT control_calphad_validation_pycalphad_version_check;
ALTER TABLE control_calphad_evidence_blobs
  VALIDATE CONSTRAINT control_calphad_evidence_blob_payload_sha_check;
ALTER TABLE control_calphad_input_blobs
  VALIDATE CONSTRAINT control_calphad_input_blob_size_check,
  VALIDATE CONSTRAINT control_calphad_input_blob_encoding_check,
  VALIDATE CONSTRAINT control_calphad_input_blob_payload_sha_check,
  VALIDATE CONSTRAINT control_calphad_input_blob_payload_size_check;

CREATE INDEX IF NOT EXISTS control_calphad_validation_revision_created_idx
  ON control_calphad_validation_events(revision_id, created_at DESC, validation_id DESC);
CREATE INDEX IF NOT EXISTS control_calphad_validation_run_idx
  ON control_calphad_validation_events(run_id) WHERE run_id IS NOT NULL;
DROP INDEX IF EXISTS control_calphad_validation_run_operation_uidx;
DROP INDEX IF EXISTS control_calphad_validation_request_uidx;
CREATE UNIQUE INDEX IF NOT EXISTS control_calphad_validation_evidence_uidx
  ON control_calphad_validation_events(revision_id, run_id, operation, evidence_sha256)
  WHERE run_id IS NOT NULL AND evidence_sha256 IS NOT NULL;
CREATE INDEX IF NOT EXISTS control_calphad_validation_request_idx
  ON control_calphad_validation_events
    (revision_id, run_id, operation, request_sha256, created_at DESC)
  WHERE request_sha256 IS NOT NULL;
DROP INDEX IF EXISTS control_calphad_validation_inspection_lineage_idx;
CREATE INDEX control_calphad_validation_inspection_lineage_idx
  ON control_calphad_validation_events
    (revision_id, run_id, database_format, runtime_image_id, database_inventory_sha256, evidence_sha256)
  WHERE operation = 'inspect' AND status = 'input_validated';

CREATE OR REPLACE FUNCTION public.ultra_validate_calphad_revision_parent()
RETURNS trigger LANGUAGE plpgsql SET search_path = pg_catalog AS $$
BEGIN
  IF NEW.parent_revision_id IS NOT NULL AND NOT EXISTS (
	SELECT 1 FROM public.control_calphad_revisions parent
    WHERE parent.revision_id = NEW.parent_revision_id
      AND parent.resource_id <> NEW.resource_id
      AND parent.owner_user_id = NEW.owner_user_id
      AND COALESCE(parent.owner_org_id, '') = COALESCE(NEW.owner_org_id, '')
  ) THEN
    RAISE EXCEPTION 'CALPHAD parent revision must be a different resource in the same owner tenant'
      USING ERRCODE = '23503';
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS control_calphad_revisions_parent_guard ON control_calphad_revisions;
CREATE TRIGGER control_calphad_revisions_parent_guard
BEFORE INSERT ON control_calphad_revisions
FOR EACH ROW EXECUTE FUNCTION public.ultra_validate_calphad_revision_parent();

CREATE OR REPLACE FUNCTION public.ultra_validate_calphad_validation_run_authority()
RETURNS trigger LANGUAGE plpgsql SET search_path = pg_catalog AS $$
BEGIN
  IF NEW.created_by_authority = 'trusted_worker' AND NOT EXISTS (
    SELECT 1
	FROM public.control_runs run_record
	JOIN public.control_run_leases lease ON lease.run_id = run_record.run_id
	JOIN public.control_calphad_revisions revision ON revision.revision_id = NEW.revision_id
    WHERE run_record.run_id = NEW.run_id
      AND run_record.status = 'running'
      AND lease.lease_expires_at > clock_timestamp()
      AND run_record.user_id = revision.owner_user_id
      AND (COALESCE(revision.owner_org_id, '') = '' OR
           COALESCE(run_record.metadata->>'org_id', '') = revision.owner_org_id)
  ) THEN
    RAISE EXCEPTION 'CALPHAD_RUN_LEASE_INVALID: trusted CALPHAD validation requires the owner run and its active unexpired lease'
      USING ERRCODE = '28000';
  END IF;
  IF NEW.created_by_authority = 'trusted_worker' AND NOT EXISTS (
    SELECT 1
	FROM public.control_runs run_record
    WHERE run_record.run_id = NEW.run_id
      AND jsonb_typeof(run_record.metadata->'calphad_runtime_policy') = 'object'
      AND run_record.metadata->'calphad_runtime_policy' = jsonb_build_object(
		'schema_version', 'ultra.calphad.runtime-policy.v2',
        'authority', 'control_plane',
        'runtime_image_id', NEW.runtime_image_id,
		'pycalphad_version', '0.11.2',
		'network', 'none',
		'no_new_privileges', true,
		'read_only_root_filesystem', true,
		'cap_drop_all', true,
		'cpus_at_most', 8,
		'memory_bytes_at_most', 34359738368,
		'pids_at_most', 4096
      )
      AND NEW.pycalphad_version = '0.11.2'
  ) THEN
    RAISE EXCEPTION 'CALPHAD_RUNTIME_POLICY_INVALID: validation runtime is not authorized by server-stamped run metadata'
      USING ERRCODE = '28000';
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS control_calphad_validation_run_authority ON control_calphad_validation_events;
CREATE TRIGGER control_calphad_validation_run_authority
BEFORE INSERT ON control_calphad_validation_events
FOR EACH ROW EXECUTE FUNCTION public.ultra_validate_calphad_validation_run_authority();

CREATE OR REPLACE FUNCTION public.ultra_validate_calphad_pressure_binding()
RETURNS trigger LANGUAGE plpgsql SET search_path = pg_catalog AS $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM public.control_calphad_revisions revision
    WHERE revision.revision_id = NEW.revision_id
      AND revision.resource_id = NEW.resource_id
      AND revision.sha256 = NEW.database_sha256
      AND revision.size_bytes = NEW.database_size_bytes
      AND revision.database_format = NEW.database_format
      AND revision.assessment_pressure_min_pa = NEW.assessment_pressure_min_pa
      AND revision.assessment_pressure_max_pa = NEW.assessment_pressure_max_pa
      AND revision.metadata->'assessment_pressure_limits_Pa' = jsonb_build_array(
        revision.assessment_pressure_min_pa, revision.assessment_pressure_max_pa
      )
      AND revision.metadata->'owner_declaration'->>'schema_version' = 'ultra.calphad.owner-declaration.v1'
      AND revision.metadata->'owner_declaration'->>'authority' = 'resource_owner'
      AND revision.metadata->'owner_declaration'->>'database_format' = revision.database_format
      AND revision.metadata->'owner_declaration'->'assessment_pressure_limits_Pa' = jsonb_build_array(
        revision.assessment_pressure_min_pa, revision.assessment_pressure_max_pa
      )
      AND NEW.metadata->'assessment_pressure_limits_Pa' = jsonb_build_array(
        NEW.assessment_pressure_min_pa, NEW.assessment_pressure_max_pa
      )
  ) THEN
    RAISE EXCEPTION 'CALPHAD_PRESSURE_BINDING_INVALID: validation pressure limits must match the immutable owner declaration'
      USING ERRCODE = '23514';
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS control_calphad_validation_pressure_binding ON control_calphad_validation_events;
DROP TRIGGER IF EXISTS control_calphad_validation_z_pressure_binding ON control_calphad_validation_events;
CREATE TRIGGER control_calphad_validation_z_pressure_binding
BEFORE INSERT ON control_calphad_validation_events
FOR EACH ROW EXECUTE FUNCTION public.ultra_validate_calphad_pressure_binding();

CREATE OR REPLACE FUNCTION public.ultra_validate_calphad_input_retention()
RETURNS trigger LANGUAGE plpgsql SET search_path = pg_catalog AS $$
BEGIN
  IF NOT EXISTS (
	SELECT 1
	FROM public.control_calphad_revisions revision
	JOIN public.control_calphad_input_blobs blob
	  ON blob.input_sha256 = revision.sha256
	 AND blob.input_size_bytes = revision.size_bytes
	 AND octet_length(blob.payload) = revision.size_bytes
	 AND encode(sha256(blob.payload), 'hex') = revision.sha256
	WHERE revision.revision_id = NEW.revision_id
	  AND revision.resource_id = NEW.resource_id
	  AND revision.sha256 = NEW.database_sha256
	  AND revision.size_bytes = NEW.database_size_bytes
	  AND revision.database_format = NEW.database_format
  ) THEN
	RAISE EXCEPTION 'CALPHAD_INPUT_RETENTION_REQUIRED: validation requires exact retained CALPHAD database bytes'
	  USING ERRCODE = '23514';
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS control_calphad_validation_input_retention ON control_calphad_validation_events;
CREATE TRIGGER control_calphad_validation_input_retention
BEFORE INSERT ON control_calphad_validation_events
FOR EACH ROW EXECUTE FUNCTION public.ultra_validate_calphad_input_retention();

CREATE OR REPLACE FUNCTION public.ultra_validate_calphad_equilibrium_inspection_lineage()
RETURNS trigger LANGUAGE plpgsql SET search_path = pg_catalog AS $$
BEGIN
  IF NEW.operation IN ('equilibrium', 'scheil') AND NOT EXISTS (
    SELECT 1
	FROM public.control_calphad_validation_events inspection
	JOIN public.control_calphad_evidence_blobs blob
      ON blob.evidence_sha256 = inspection.evidence_sha256
     AND blob.evidence_size_bytes = inspection.evidence_size_bytes
     AND octet_length(blob.payload) = inspection.evidence_size_bytes
     AND encode(sha256(blob.payload), 'hex') = inspection.evidence_sha256
    WHERE inspection.revision_id = NEW.revision_id
      AND inspection.run_id = NEW.run_id
      AND inspection.operation = 'inspect'
      AND inspection.status = 'input_validated'
      AND inspection.database_format = NEW.database_format
      AND inspection.runtime_image_id = NEW.runtime_image_id
      AND inspection.database_inventory_sha256 = NEW.database_inventory_sha256
      AND inspection.evidence_contract_version = 'ultra.calphad.retained-evidence.v2'
      AND inspection.evidence_sha256 = NEW.inspection_evidence_sha256
  ) THEN
    RAISE EXCEPTION 'CALPHAD_INSPECTION_REQUIRED: solver operation requires exact retained inspection evidence and database inventory for the same revision, run, and runtime image'
      USING ERRCODE = '23514';
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS control_calphad_equilibrium_inspection_lineage ON control_calphad_validation_events;
CREATE TRIGGER control_calphad_equilibrium_inspection_lineage
BEFORE INSERT ON control_calphad_validation_events
FOR EACH ROW EXECUTE FUNCTION public.ultra_validate_calphad_equilibrium_inspection_lineage();

CREATE OR REPLACE FUNCTION public.ultra_reject_calphad_ledger_mutation()
RETURNS trigger LANGUAGE plpgsql SET search_path = pg_catalog AS $$
BEGIN
  RAISE EXCEPTION 'CALPHAD governance ledger is append-only';
END;
$$;

DROP TRIGGER IF EXISTS control_calphad_revisions_append_only ON control_calphad_revisions;
CREATE TRIGGER control_calphad_revisions_append_only
BEFORE UPDATE OR DELETE ON control_calphad_revisions
FOR EACH ROW EXECUTE FUNCTION public.ultra_reject_calphad_ledger_mutation();

DROP TRIGGER IF EXISTS control_calphad_validation_append_only ON control_calphad_validation_events;
CREATE TRIGGER control_calphad_validation_append_only
BEFORE UPDATE OR DELETE ON control_calphad_validation_events
FOR EACH ROW EXECUTE FUNCTION public.ultra_reject_calphad_ledger_mutation();

DROP TRIGGER IF EXISTS control_calphad_revisions_no_truncate ON control_calphad_revisions;
CREATE TRIGGER control_calphad_revisions_no_truncate
BEFORE TRUNCATE ON control_calphad_revisions
FOR EACH STATEMENT EXECUTE FUNCTION public.ultra_reject_calphad_ledger_mutation();

DROP TRIGGER IF EXISTS control_calphad_validation_no_truncate ON control_calphad_validation_events;
CREATE TRIGGER control_calphad_validation_no_truncate
BEFORE TRUNCATE ON control_calphad_validation_events
FOR EACH STATEMENT EXECUTE FUNCTION public.ultra_reject_calphad_ledger_mutation();

DROP TRIGGER IF EXISTS control_calphad_evidence_blobs_append_only ON control_calphad_evidence_blobs;
CREATE TRIGGER control_calphad_evidence_blobs_append_only
BEFORE UPDATE OR DELETE ON control_calphad_evidence_blobs
FOR EACH ROW EXECUTE FUNCTION public.ultra_reject_calphad_ledger_mutation();

DROP TRIGGER IF EXISTS control_calphad_evidence_blobs_no_truncate ON control_calphad_evidence_blobs;
CREATE TRIGGER control_calphad_evidence_blobs_no_truncate
BEFORE TRUNCATE ON control_calphad_evidence_blobs
FOR EACH STATEMENT EXECUTE FUNCTION public.ultra_reject_calphad_ledger_mutation();

DROP TRIGGER IF EXISTS control_calphad_input_blobs_append_only ON control_calphad_input_blobs;
CREATE TRIGGER control_calphad_input_blobs_append_only
BEFORE UPDATE OR DELETE ON control_calphad_input_blobs
FOR EACH ROW EXECUTE FUNCTION public.ultra_reject_calphad_ledger_mutation();

DROP TRIGGER IF EXISTS control_calphad_input_blobs_no_truncate ON control_calphad_input_blobs;
CREATE TRIGGER control_calphad_input_blobs_no_truncate
BEFORE TRUNCATE ON control_calphad_input_blobs
FOR EACH STATEMENT EXECUTE FUNCTION public.ultra_reject_calphad_ledger_mutation();
