-- GoldGate M0: model-agnostic training persistence (registry, versions/lineages,
-- gold sets, gate policy + guardrail clauses as data, benchmark/canary/audit,
-- job backbone) + the documented seed rows. Mirror of the schema.sql block;
-- schema.sql remains the source of truth.
-- GoldGate training subsystem (M0): model-agnostic persistence for gold-gated
-- continual finetuning. The registry row + guardrail clause rows are DATA so a
-- new model onboards with zero DDL/enum/route changes (the M5 acceptance test).
-- Design: planning/2026-07-07-goldgate-continual-finetuning-plan.md
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS control_training_models (
  model_key text PRIMARY KEY,
  task_type text NOT NULL,
  display_name text NOT NULL,
  dataset_format text NOT NULL,
  metric_schema text NOT NULL,
  requires_phash boolean NOT NULL DEFAULT false,
  capabilities jsonb NOT NULL DEFAULT '[]',
  executor jsonb NOT NULL DEFAULT '{}',
  classes jsonb,
  leakage_defenses_extra jsonb NOT NULL DEFAULT '[]',
  metadata jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS control_training_domains (
  domain_id text PRIMARY KEY,
  name text NOT NULL,
  description text,
  metadata jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS control_training_lineages (
  lineage_id text PRIMARY KEY,
  domain_id text NOT NULL REFERENCES control_training_domains(domain_id),
  model_key text NOT NULL REFERENCES control_training_models(model_key),
  scope text NOT NULL DEFAULT 'shared',
  owner_user_id text,
  parent_lineage_id text,
  active_version_id text,
  metadata jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS control_training_model_versions (
  version_id text PRIMARY KEY,
  lineage_id text NOT NULL REFERENCES control_training_lineages(lineage_id),
  model_key text NOT NULL REFERENCES control_training_models(model_key),
  status text NOT NULL CHECK (status IN ('candidate','canary','active','retired','rejected')),
  is_frozen boolean NOT NULL DEFAULT false,
  weights_uri text,
  source_job_id text,
  artifact_run_id text,
  metrics jsonb NOT NULL DEFAULT '{}',
  metadata jsonb NOT NULL DEFAULT '{}',
  activated_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

-- Sync-job-owned CACHE row (single writer: the training.sync worker from M1 on).
-- active_model_version mirrors control_training_lineages.active_version_id and
-- retrain_gate_thresholds mirrors the control_training_gate_policies row; any
-- writer that updates the canonical tables MUST refresh this row in the same
-- transaction (the seed-parity tests pin the seed copies to one fixture).
CREATE TABLE IF NOT EXISTS control_training_model_status (
  model_key text PRIMARY KEY REFERENCES control_training_models(model_key),
  dataset_name text,
  dataset_id text,
  model_health text,
  reviewed_images bigint NOT NULL DEFAULT 0,
  unreviewed_images bigint NOT NULL DEFAULT 0,
  class_counts jsonb NOT NULL DEFAULT '{}',
  per_class_new_objects jsonb NOT NULL DEFAULT '{}',
  unsupported_class_counts jsonb NOT NULL DEFAULT '{}',
  last_sync_at timestamptz,
  last_retrain_at timestamptz,
  active_model_version text,
  retrain_gate boolean NOT NULL DEFAULT false,
  retrain_gate_reasons jsonb NOT NULL DEFAULT '[]',
  retrain_gate_counts jsonb NOT NULL DEFAULT '{}',
  retrain_gate_thresholds jsonb NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS control_training_gate_policies (
  model_key text PRIMARY KEY REFERENCES control_training_models(model_key),
  min_reviewed bigint NOT NULL,
  min_new_objects bigint NOT NULL,
  min_per_class_objects jsonb NOT NULL DEFAULT '{}',
  min_days bigint NOT NULL
);

CREATE TABLE IF NOT EXISTS control_training_guardrail_clauses (
  model_key text NOT NULL REFERENCES control_training_models(model_key),
  clause_key text NOT NULL,
  metric_path text NOT NULL,
  comparator text NOT NULL CHECK (comparator IN ('max_drop_vs_active','abs_floor','max_rise_vs_active','abs_ceiling')),
  value real NOT NULL,
  slice text,
  params jsonb NOT NULL DEFAULT '{}',
  enabled boolean NOT NULL DEFAULT true,
  required boolean NOT NULL DEFAULT false,
  PRIMARY KEY (model_key, clause_key)
);

CREATE TABLE IF NOT EXISTS control_training_gate_config_events (
  event_id text PRIMARY KEY,
  model_key text NOT NULL,
  table_name text NOT NULL,
  change jsonb NOT NULL DEFAULT '{}',
  actor_user_id text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS control_training_gold_sets (
  gold_set_id text PRIMARY KEY,
  model_key text NOT NULL REFERENCES control_training_models(model_key),
  version bigint NOT NULL,
  content_hash text UNIQUE,
  item_count bigint NOT NULL DEFAULT 0,
  label_stats jsonb NOT NULL DEFAULT '{}',
  strata_summary jsonb NOT NULL DEFAULT '{}',
  split_manifest_uri text,
  provenance jsonb NOT NULL DEFAULT '{}',
  status text NOT NULL CHECK (status IN ('draft','freezing','frozen','failed','retired')),
  created_at timestamptz NOT NULL DEFAULT now(),
  created_by_user_id text NOT NULL,
  frozen_at timestamptz,
  UNIQUE (model_key, version)
);

CREATE TABLE IF NOT EXISTS control_training_gold_items (
  gold_set_id text NOT NULL REFERENCES control_training_gold_sets(gold_set_id) ON DELETE CASCADE,
  item_id text NOT NULL,
  source_ref jsonb NOT NULL DEFAULT '{}',
  slice text NOT NULL CHECK (slice IN ('prior_train','held_out_test')),
  label_kind text NOT NULL CHECK (label_kind IN ('boxes','mask','class')),
  content_sha256 text NOT NULL,
  phash text,
  gt_label_sha256 text NOT NULL,
  gt_label_uri text NOT NULL,
  width bigint,
  height bigint,
  metadata jsonb NOT NULL DEFAULT '{}',
  footprint_geom jsonb,
  strata_tags jsonb NOT NULL DEFAULT '{}',
  PRIMARY KEY (gold_set_id, item_id)
);

CREATE TABLE IF NOT EXISTS control_training_replay_pool (
  model_key text NOT NULL REFERENCES control_training_models(model_key),
  content_sha256 text NOT NULL,
  source_ref jsonb,
  label_stats jsonb,
  site_id text,
  forgetting_risk real,
  last_used_epoch bigint,
  priority real,
  PRIMARY KEY (model_key, content_sha256)
);

CREATE TABLE IF NOT EXISTS control_training_benchmark_runs (
  run_id text PRIMARY KEY,
  model_version_id text NOT NULL,
  gold_set_id text NOT NULL,
  gold_set_content_hash text NOT NULL,
  metric_schema text NOT NULL,
  kernel_version text NOT NULL,
  metrics jsonb NOT NULL DEFAULT '{}',
  guardrails_passed boolean NOT NULL,
  guardrails_reasons jsonb NOT NULL DEFAULT '[]',
  report_uri text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS control_training_canary_observations (
  observation_id text PRIMARY KEY,
  model_key text,
  canary_version_id text NOT NULL,
  active_version_id text NOT NULL,
  run_id text NOT NULL,
  canary_metrics jsonb,
  active_metrics jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);
-- model_key scopes the UI drift-echo list route; nullable because rows written
-- before the column existed cannot be backfilled (new writes always set it).
ALTER TABLE control_training_canary_observations ADD COLUMN IF NOT EXISTS model_key text;

CREATE TABLE IF NOT EXISTS control_training_retrain_requests (
  request_id text PRIMARY KEY,
  model_key text NOT NULL REFERENCES control_training_models(model_key),
  training_job_id text,
  status text NOT NULL DEFAULT 'queued',
  note text,
  error text,
  model_version text,
  gating_summary jsonb NOT NULL DEFAULT '{}',
  benchmark_report_artifact_id text,
  requested_by_user_id text,
  created_at timestamptz NOT NULL DEFAULT now(),
  started_at timestamptz,
  finished_at timestamptz
);

CREATE TABLE IF NOT EXISTS control_training_model_version_events (
  event_id text PRIMARY KEY,
  version_id text NOT NULL,
  event_type text NOT NULL,
  actor_user_id text NOT NULL,
  from_status text,
  to_status text,
  benchmark_run_id text,
  gold_set_content_hash text,
  reason text,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS control_training_jobs (
  job_id text PRIMARY KEY,
  model_key text NOT NULL REFERENCES control_training_models(model_key),
  job_type text NOT NULL,
  status text NOT NULL DEFAULT 'queued',
  gpu_pool text,
  params jsonb NOT NULL DEFAULT '{}',
  progress_completed bigint NOT NULL DEFAULT 0,
  progress_total bigint NOT NULL DEFAULT 0,
  error text,
  owner_user_id text NOT NULL,
  owner_org_id text,
  created_by_user_id text,
  created_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL,
  started_at timestamptz,
  completed_at timestamptz,
  output_summary jsonb NOT NULL DEFAULT '{}',
  metadata jsonb NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS control_training_job_events (
  event_id text PRIMARY KEY,
  job_id text NOT NULL REFERENCES control_training_jobs(job_id) ON DELETE CASCADE,
  sequence bigint NOT NULL,
  event_type text NOT NULL,
  actor_user_id text,
  actor_org_id text,
  ts timestamptz NOT NULL,
  message text,
  metadata jsonb NOT NULL DEFAULT '{}',
  UNIQUE (job_id, sequence)
);

CREATE TABLE IF NOT EXISTS control_training_job_leases (
  job_id text PRIMARY KEY REFERENCES control_training_jobs(job_id) ON DELETE CASCADE,
  worker_id text NOT NULL,
  lease_token text NOT NULL UNIQUE,
  lease_expires_at timestamptz NOT NULL,
  created_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL
);

CREATE INDEX IF NOT EXISTS control_training_lineages_model_scope_idx ON control_training_lineages(model_key, scope, updated_at DESC);
CREATE INDEX IF NOT EXISTS control_training_lineages_domain_idx ON control_training_lineages(domain_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS control_training_model_versions_lineage_idx ON control_training_model_versions(lineage_id, created_at DESC);
CREATE INDEX IF NOT EXISTS control_training_model_versions_model_status_idx ON control_training_model_versions(model_key, status, created_at DESC);
CREATE INDEX IF NOT EXISTS control_training_gold_items_set_slice_idx ON control_training_gold_items(gold_set_id, slice);
CREATE INDEX IF NOT EXISTS control_training_gold_sets_model_status_idx ON control_training_gold_sets(model_key, status, created_at DESC);
CREATE INDEX IF NOT EXISTS control_training_benchmark_runs_version_idx ON control_training_benchmark_runs(model_version_id, created_at DESC);
CREATE INDEX IF NOT EXISTS control_training_benchmark_runs_gold_hash_idx ON control_training_benchmark_runs(gold_set_content_hash, created_at DESC);
CREATE INDEX IF NOT EXISTS control_training_replay_pool_priority_idx ON control_training_replay_pool(model_key, priority DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS control_training_retrain_requests_model_idx ON control_training_retrain_requests(model_key, created_at DESC);
CREATE INDEX IF NOT EXISTS control_training_model_version_events_version_idx ON control_training_model_version_events(version_id, created_at DESC);
CREATE INDEX IF NOT EXISTS control_training_canary_observations_canary_idx ON control_training_canary_observations(canary_version_id, created_at DESC);
CREATE INDEX IF NOT EXISTS control_training_canary_observations_model_idx ON control_training_canary_observations(model_key, created_at DESC);
CREATE INDEX IF NOT EXISTS control_training_jobs_model_status_idx ON control_training_jobs(model_key, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS control_training_jobs_type_idx ON control_training_jobs(job_type, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS control_training_job_events_job_ts_idx ON control_training_job_events(job_id, ts DESC);

-- GoldGate M0 seed: the documented direct-DB writes (registry + version 0 +
-- lineage + gate policy + guardrail clauses). model_key 'yolov5_rarespot' is
-- the ONE canonical spelling (FE, Go, DB, NATS, barrel dirs). The baked
-- RareSpotWeights.pt becomes version 0: active, frozen, unbenchmarked — every
-- future weight must beat it on a frozen gold set to replace it.
INSERT INTO control_training_models (model_key, task_type, display_name, dataset_format, metric_schema, requires_phash, capabilities, executor, classes, leakage_defenses_extra, metadata)
VALUES (
  'yolov5_rarespot', 'detection', 'RareSpot prairie-dog/burrow', 'yolo_txt_tiles_512', 'detection.v1', true,
  '["SYNC","ASSEMBLE","FINETUNE","BENCHMARK"]'::jsonb,
  '{"gpu_pool":"titan","min_vram_gb":8,"shm_gb":8,"wall_clock_budget_s":21600,"cpu_only":false}'::jsonb,
  '{"0":"prairie_dog","1":"burrow"}'::jsonb,
  '["aerial_geospatial_overlap"]'::jsonb,
  '{"framework":"PyTorch/YOLOv5","description":"Prairie dog and burrow detection on aerial survey imagery (512px tiles). Gold-gated continual finetuning.","dimensions":["2d"],"workflow":"rarespot_ecology","gt_layer_priority":["gt2","New Ground Truth"]}'::jsonb
)
ON CONFLICT (model_key) DO NOTHING;

INSERT INTO control_training_domains (domain_id, name, description)
VALUES ('ecology', 'Ecology', 'Field-survey detection and segmentation models.')
ON CONFLICT (domain_id) DO NOTHING;

INSERT INTO control_training_lineages (lineage_id, domain_id, model_key, scope, active_version_id)
VALUES ('yolov5_rarespot-shared', 'ecology', 'yolov5_rarespot', 'shared', 'yolov5_rarespot-v0')
ON CONFLICT (lineage_id) DO NOTHING;

-- Guarded on "the model has NO versions at all", not just the v0 PK: schema.sql
-- re-runs on every deploy, and a plain ON CONFLICT would resurrect a deleted v0
-- as a SECOND status='active' row after later promotions (single-active invariant).
INSERT INTO control_training_model_versions (version_id, lineage_id, model_key, status, is_frozen, weights_uri, metrics, metadata, activated_at)
SELECT
  'yolov5_rarespot-v0', 'yolov5_rarespot-shared', 'yolov5_rarespot', 'active', true,
  'data/models/yolo/RareSpotWeights.pt',
  '{}'::jsonb,
  '{"is_baked":true,"provenance":"pre-GoldGate checkpoint; trained --noval on all_overfit.yaml (lineage focuswin_generalize4 -> allfullneg_calibrate2); no held-out validation existed at training time"}'::jsonb,
  now()
WHERE NOT EXISTS (SELECT 1 FROM control_training_model_versions WHERE model_key = 'yolov5_rarespot');

INSERT INTO control_training_model_status (model_key, dataset_name, model_health, active_model_version, retrain_gate, retrain_gate_reasons, retrain_gate_thresholds)
VALUES (
  'yolov5_rarespot', 'Prairie_Dog_Active_Learning', 'watch', 'yolov5_rarespot-v0', false,
  '["No reviewed training data has been synced yet - the sync path ships with M1.","Cannot check the gold-set precondition - no gold set has been frozen yet."]'::jsonb,
  '{"min_reviewed":50,"min_new_objects":200,"min_per_class_objects":{"prairie_dog":20,"burrow":20},"min_days":3}'::jsonb
)
ON CONFLICT (model_key) DO NOTHING;

INSERT INTO control_training_gate_policies (model_key, min_reviewed, min_new_objects, min_per_class_objects, min_days)
VALUES ('yolov5_rarespot', 50, 200, '{"prairie_dog":20,"burrow":20}'::jsonb, 3)
ON CONFLICT (model_key) DO NOTHING;

INSERT INTO control_training_guardrail_clauses (model_key, clause_key, metric_path, comparator, value, slice, params, enabled, required) VALUES
  ('yolov5_rarespot', 'agg_map50',           'aggregate.map50',                'max_drop_vs_active', 0.005, NULL,            '{}'::jsonb,                      true, true),
  ('yolov5_rarespot', 'agg_map50_95',        'aggregate.map50_95',             'max_drop_vs_active', 0.005, NULL,            '{}'::jsonb,                      true, true),
  ('yolov5_rarespot', 'class_recall_delta',  'per_class.*.recall_at_op',       'max_drop_vs_active', 0.02,  NULL,            '{}'::jsonb,                      true, true),
  ('yolov5_rarespot', 'class_recall_abs',    'per_class.*.recall_at_op',       'abs_floor',          0.50,  NULL,            '{}'::jsonb,                      true, true),
  ('yolov5_rarespot', 'slice_prior_map50',   'per_slice.prior_train.map50',    'max_drop_vs_active', 0.02,  'prior_train',   '{"min_label_count":10}'::jsonb,  true, true),
  ('yolov5_rarespot', 'slice_held_map50',    'per_slice.held_out_test.map50',  'max_drop_vs_active', 0.005, 'held_out_test', '{"min_label_count":10}'::jsonb,  true, true),
  ('yolov5_rarespot', 'class_ap50_collapse', 'per_class.*.ap50',               'max_drop_vs_active', 0.05,  NULL,            '{}'::jsonb,                      true, true),
  ('yolov5_rarespot', 'class_ap50_abs',      'per_class.*.ap50',               'abs_floor',          0.10,  NULL,            '{"strict":true}'::jsonb,         true, true),
  ('yolov5_rarespot', 'fp_empty_ceiling',    'aggregate.fp_per_empty_frame',   'max_rise_vs_active', 0.10,  NULL,            '{}'::jsonb,                      true, true),
  ('yolov5_rarespot', 'precision_delta',     'aggregate.precision_at_op',      'max_drop_vs_active', 0.03,  NULL,            '{}'::jsonb,                      true, true)
ON CONFLICT (model_key, clause_key) DO NOTHING;
