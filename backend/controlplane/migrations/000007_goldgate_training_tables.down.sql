-- GoldGate M0 rollback: drop the training subsystem tables in reverse
-- dependency order. Seeds and indexes go with their tables.
DROP TABLE IF EXISTS control_training_job_leases;
DROP TABLE IF EXISTS control_training_job_events;
DROP TABLE IF EXISTS control_training_jobs;
DROP TABLE IF EXISTS control_training_model_version_events;
DROP TABLE IF EXISTS control_training_retrain_requests;
DROP TABLE IF EXISTS control_training_canary_observations;
DROP TABLE IF EXISTS control_training_benchmark_runs;
DROP TABLE IF EXISTS control_training_replay_pool;
DROP TABLE IF EXISTS control_training_gold_items;
DROP TABLE IF EXISTS control_training_gold_sets;
DROP TABLE IF EXISTS control_training_gate_config_events;
DROP TABLE IF EXISTS control_training_guardrail_clauses;
DROP TABLE IF EXISTS control_training_gate_policies;
DROP TABLE IF EXISTS control_training_model_status;
DROP TABLE IF EXISTS control_training_model_versions;
DROP TABLE IF EXISTS control_training_lineages;
DROP TABLE IF EXISTS control_training_domains;
DROP TABLE IF EXISTS control_training_models;
