package store

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

// GoldGate training reads (M0). Rows are written by the schema.sql seed today;
// the sync/benchmark/finetune workers take over the writes from M1.

func (s *PostgresStore) ListTrainingModels(ctx context.Context) ([]domain.TrainingModelRecord, error) {
	rows, err := s.pool.Query(ctx, `
SELECT model_key, task_type, display_name, dataset_format, metric_schema, requires_phash,
       capabilities, executor, classes, leakage_defenses_extra, metadata, created_at
FROM control_training_models
ORDER BY model_key ASC`)
	if err != nil {
		return nil, mapPgError(err)
	}
	defer rows.Close()
	records := []domain.TrainingModelRecord{}
	for rows.Next() {
		record, err := scanTrainingModelRow(rows)
		if err != nil {
			return nil, mapPgError(err)
		}
		records = append(records, record)
	}
	if err := rows.Err(); err != nil {
		return nil, mapPgError(err)
	}
	return records, nil
}

func (s *PostgresStore) GetTrainingModel(ctx context.Context, modelKey string) (domain.TrainingModelRecord, error) {
	row := s.pool.QueryRow(ctx, `
SELECT model_key, task_type, display_name, dataset_format, metric_schema, requires_phash,
       capabilities, executor, classes, leakage_defenses_extra, metadata, created_at
FROM control_training_models
WHERE model_key = $1`, strings.TrimSpace(modelKey))
	record, err := scanTrainingModelRow(row)
	if err != nil {
		return domain.TrainingModelRecord{}, mapPgError(err)
	}
	return record, nil
}

func (s *PostgresStore) GetTrainingModelStatus(ctx context.Context, modelKey string) (domain.TrainingModelStatusRecord, error) {
	var record domain.TrainingModelStatusRecord
	var classCounts, perClassNew, unsupported, gateCounts, gateThresholds, gateReasons []byte
	var lastSync, lastRetrain *time.Time
	err := s.pool.QueryRow(ctx, `
SELECT model_key, COALESCE(dataset_name, ''), COALESCE(dataset_id, ''), COALESCE(model_health, ''),
       reviewed_images, unreviewed_images, class_counts, per_class_new_objects,
       unsupported_class_counts, last_sync_at, last_retrain_at,
       COALESCE(active_model_version, ''), retrain_gate, retrain_gate_reasons,
       retrain_gate_counts, retrain_gate_thresholds
FROM control_training_model_status
WHERE model_key = $1`, strings.TrimSpace(modelKey)).Scan(
		&record.ModelKey,
		&record.DatasetName,
		&record.DatasetID,
		&record.ModelHealth,
		&record.ReviewedImages,
		&record.UnreviewedImages,
		&classCounts,
		&perClassNew,
		&unsupported,
		&lastSync,
		&lastRetrain,
		&record.ActiveModelVersion,
		&record.RetrainGate,
		&gateReasons,
		&gateCounts,
		&gateThresholds,
	)
	if err != nil {
		return domain.TrainingModelStatusRecord{}, mapPgError(err)
	}
	record.ClassCounts = jsonMap(classCounts)
	record.PerClassNewObjects = jsonMap(perClassNew)
	record.UnsupportedClassCounts = jsonMap(unsupported)
	record.RetrainGateCounts = jsonMap(gateCounts)
	record.RetrainGateThresholds = jsonMap(gateThresholds)
	record.RetrainGateReasons = jsonStringSlice(gateReasons)
	record.LastSyncAt = utcTimePtr(lastSync)
	record.LastRetrainAt = utcTimePtr(lastRetrain)
	return record, nil
}

func (s *PostgresStore) ListTrainingDomains(ctx context.Context, limit int, offset int) ([]domain.TrainingDomainRecord, error) {
	rows, err := s.pool.Query(ctx, `
SELECT domain_id, name, COALESCE(description, ''), metadata, created_at, updated_at
FROM control_training_domains
ORDER BY domain_id ASC
LIMIT $1 OFFSET $2`, limit32(limit, 200), int32(max(0, offset)))
	if err != nil {
		return nil, mapPgError(err)
	}
	defer rows.Close()
	records := []domain.TrainingDomainRecord{}
	for rows.Next() {
		var record domain.TrainingDomainRecord
		var metadata []byte
		if err := rows.Scan(&record.DomainID, &record.Name, &record.Description, &metadata, &record.CreatedAt, &record.UpdatedAt); err != nil {
			return nil, mapPgError(err)
		}
		record.Metadata = jsonMap(metadata)
		record.CreatedAt = record.CreatedAt.UTC()
		record.UpdatedAt = record.UpdatedAt.UTC()
		records = append(records, record)
	}
	if err := rows.Err(); err != nil {
		return nil, mapPgError(err)
	}
	return records, nil
}

// Unscoped by design at M0: only scope='shared' lineage rows exist until the
// fork route ships. The fork milestone MUST add ForUser tenant variants here
// before user-scoped lineages are ever written (see the go-nats skill's
// new-table checklist).
func (s *PostgresStore) ListTrainingLineages(ctx context.Context, domainID string, limit int, offset int) ([]domain.TrainingLineageRecord, error) {
	rows, err := s.pool.Query(ctx, `
SELECT lineage_id, domain_id, model_key, scope, COALESCE(owner_user_id, ''),
       COALESCE(parent_lineage_id, ''), COALESCE(active_version_id, ''), metadata, created_at, updated_at
FROM control_training_lineages
WHERE domain_id = $1
ORDER BY lineage_id ASC
LIMIT $2 OFFSET $3`, strings.TrimSpace(domainID), limit32(limit, 200), int32(max(0, offset)))
	if err != nil {
		return nil, mapPgError(err)
	}
	defer rows.Close()
	records := []domain.TrainingLineageRecord{}
	for rows.Next() {
		record, err := scanTrainingLineageRow(rows)
		if err != nil {
			return nil, mapPgError(err)
		}
		records = append(records, record)
	}
	if err := rows.Err(); err != nil {
		return nil, mapPgError(err)
	}
	return records, nil
}

func (s *PostgresStore) ListTrainingModelVersions(ctx context.Context, lineageID string, limit int, offset int) ([]domain.TrainingModelVersionRecord, error) {
	rows, err := s.pool.Query(ctx, `
SELECT version_id, lineage_id, model_key, status, is_frozen, COALESCE(weights_uri, ''),
       COALESCE(source_job_id, ''), COALESCE(artifact_run_id, ''), metrics, metadata,
       activated_at, created_at, updated_at
FROM control_training_model_versions
WHERE lineage_id = $1
ORDER BY created_at DESC, version_id DESC
LIMIT $2 OFFSET $3`, strings.TrimSpace(lineageID), limit32(limit, 200), int32(max(0, offset)))
	if err != nil {
		return nil, mapPgError(err)
	}
	defer rows.Close()
	records := []domain.TrainingModelVersionRecord{}
	for rows.Next() {
		record, err := scanTrainingModelVersionRow(rows)
		if err != nil {
			return nil, mapPgError(err)
		}
		records = append(records, record)
	}
	if err := rows.Err(); err != nil {
		return nil, mapPgError(err)
	}
	return records, nil
}

func (s *PostgresStore) GetTrainingModelVersion(ctx context.Context, versionID string) (domain.TrainingModelVersionRecord, error) {
	row := s.pool.QueryRow(ctx, `
SELECT version_id, lineage_id, model_key, status, is_frozen, COALESCE(weights_uri, ''),
       COALESCE(source_job_id, ''), COALESCE(artifact_run_id, ''), metrics, metadata,
       activated_at, created_at, updated_at
FROM control_training_model_versions
WHERE version_id = $1`, strings.TrimSpace(versionID))
	record, err := scanTrainingModelVersionRow(row)
	if err != nil {
		return domain.TrainingModelVersionRecord{}, mapPgError(err)
	}
	return record, nil
}

func (s *PostgresStore) ListTrainingRetrainRequests(ctx context.Context, modelKey string, limit int) ([]domain.TrainingRetrainRequestRecord, error) {
	rows, err := s.pool.Query(ctx, `
SELECT request_id, model_key, COALESCE(training_job_id, ''), status, COALESCE(note, ''),
       COALESCE(error, ''), COALESCE(model_version, ''), gating_summary,
       COALESCE(benchmark_report_artifact_id, ''), COALESCE(requested_by_user_id, ''),
       created_at, started_at, finished_at
FROM control_training_retrain_requests
WHERE model_key = $1
ORDER BY created_at DESC
LIMIT $2`, strings.TrimSpace(modelKey), limit32(limit, 200))
	if err != nil {
		return nil, mapPgError(err)
	}
	defer rows.Close()
	records := []domain.TrainingRetrainRequestRecord{}
	for rows.Next() {
		var record domain.TrainingRetrainRequestRecord
		var gating []byte
		var started, finished *time.Time
		if err := rows.Scan(
			&record.RequestID,
			&record.ModelKey,
			&record.TrainingJobID,
			&record.Status,
			&record.Note,
			&record.Error,
			&record.ModelVersion,
			&gating,
			&record.BenchmarkReportArtifactID,
			&record.RequestedByUserID,
			&record.CreatedAt,
			&started,
			&finished,
		); err != nil {
			return nil, mapPgError(err)
		}
		record.GatingSummary = jsonMap(gating)
		record.CreatedAt = record.CreatedAt.UTC()
		record.StartedAt = utcTimePtr(started)
		record.FinishedAt = utcTimePtr(finished)
		records = append(records, record)
	}
	if err := rows.Err(); err != nil {
		return nil, mapPgError(err)
	}
	return records, nil
}

func scanTrainingModelRow(row scanner) (domain.TrainingModelRecord, error) {
	var record domain.TrainingModelRecord
	var capabilities, executor, classes, defenses, metadata []byte
	if err := row.Scan(
		&record.ModelKey,
		&record.TaskType,
		&record.DisplayName,
		&record.DatasetFormat,
		&record.MetricSchema,
		&record.RequiresPhash,
		&capabilities,
		&executor,
		&classes,
		&defenses,
		&metadata,
		&record.CreatedAt,
	); err != nil {
		return domain.TrainingModelRecord{}, err
	}
	record.Capabilities = jsonStringSlice(capabilities)
	record.Executor = jsonMap(executor)
	record.Classes = jsonMap(classes)
	record.LeakageDefensesExtra = jsonStringSlice(defenses)
	record.Metadata = jsonMap(metadata)
	record.CreatedAt = record.CreatedAt.UTC()
	return record, nil
}

func scanTrainingLineageRow(row pgx.Rows) (domain.TrainingLineageRecord, error) {
	var record domain.TrainingLineageRecord
	var metadata []byte
	if err := row.Scan(
		&record.LineageID,
		&record.DomainID,
		&record.ModelKey,
		&record.Scope,
		&record.OwnerUserID,
		&record.ParentLineageID,
		&record.ActiveVersionID,
		&metadata,
		&record.CreatedAt,
		&record.UpdatedAt,
	); err != nil {
		return domain.TrainingLineageRecord{}, err
	}
	record.Metadata = jsonMap(metadata)
	record.CreatedAt = record.CreatedAt.UTC()
	record.UpdatedAt = record.UpdatedAt.UTC()
	return record, nil
}

func scanTrainingModelVersionRow(row scanner) (domain.TrainingModelVersionRecord, error) {
	var record domain.TrainingModelVersionRecord
	var metrics, metadata []byte
	var activated *time.Time
	if err := row.Scan(
		&record.VersionID,
		&record.LineageID,
		&record.ModelKey,
		&record.Status,
		&record.IsFrozen,
		&record.WeightsURI,
		&record.SourceJobID,
		&record.ArtifactRunID,
		&metrics,
		&metadata,
		&activated,
		&record.CreatedAt,
		&record.UpdatedAt,
	); err != nil {
		return domain.TrainingModelVersionRecord{}, err
	}
	record.Metrics = jsonMap(metrics)
	record.Metadata = jsonMap(metadata)
	record.ActivatedAt = utcTimePtr(activated)
	record.CreatedAt = record.CreatedAt.UTC()
	record.UpdatedAt = record.UpdatedAt.UTC()
	return record, nil
}

// ---------------------------------------------------------------------------
// GoldGate training writes (M1). Jobs/leases/events clone the data-agent job
// family (CreateDataAgentJob / AcquireDataAgentJobLease / ...); gold sets,
// versions, benchmark runs, and guardrail clause reads feed the freeze + gate
// milestones.
// ---------------------------------------------------------------------------

const trainingJobSelectColumns = `job_id, model_key, job_type, status, COALESCE(gpu_pool, ''), params,
       progress_completed, progress_total, COALESCE(error, ''), owner_user_id,
       COALESCE(owner_org_id, ''), COALESCE(created_by_user_id, ''), created_at, updated_at,
       started_at, completed_at, output_summary, metadata`

const trainingGoldSetSelectColumns = `gold_set_id, model_key, version, COALESCE(content_hash, ''), item_count,
       label_stats, strata_summary, COALESCE(split_manifest_uri, ''), provenance, status,
       created_at, created_by_user_id, frozen_at`

const trainingRetrainRequestSelectColumns = `request_id, model_key, COALESCE(training_job_id, ''), status, COALESCE(note, ''),
       COALESCE(error, ''), COALESCE(model_version, ''), gating_summary,
       COALESCE(benchmark_report_artifact_id, ''), COALESCE(requested_by_user_id, ''),
       created_at, started_at, finished_at`

// trainingRowQuerier lets the job-event append run on the pool or inside a
// transaction with one implementation (both satisfy QueryRow).
type trainingRowQuerier interface {
	QueryRow(ctx context.Context, sql string, args ...any) pgx.Row
}

func (s *PostgresStore) CreateTrainingJob(ctx context.Context, input domain.CreateTrainingJobInput) (domain.TrainingJobRecord, error) {
	modelKey := strings.TrimSpace(input.ModelKey)
	if modelKey == "" {
		return domain.TrainingJobRecord{}, errors.New("training job model key is required")
	}
	jobType := strings.ToLower(strings.TrimSpace(input.JobType))
	if jobType == "" {
		return domain.TrainingJobRecord{}, errors.New("training job type is required")
	}
	jobID := strings.TrimSpace(input.JobID)
	if jobID == "" {
		jobID = domain.NewID("training_job")
	}
	ownerUserID := strings.TrimSpace(input.OwnerUserID)
	if ownerUserID == "" {
		ownerUserID = "local-user"
	}
	createdByUserID := strings.TrimSpace(input.CreatedByUserID)
	if createdByUserID == "" {
		createdByUserID = ownerUserID
	}
	now := domain.Now()

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.TrainingJobRecord{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()

	var modelExists bool
	if err := tx.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM control_training_models WHERE model_key = $1)`, modelKey).Scan(&modelExists); err != nil {
		return domain.TrainingJobRecord{}, mapPgError(err)
	}
	if !modelExists {
		return domain.TrainingJobRecord{}, ErrNotFound
	}

	job, err := scanTrainingJobRow(tx.QueryRow(ctx, `
INSERT INTO control_training_jobs (
  job_id, model_key, job_type, status, gpu_pool, params, progress_completed, progress_total,
  error, owner_user_id, owner_org_id, created_by_user_id, created_at, updated_at,
  started_at, completed_at, output_summary, metadata
)
VALUES ($1, $2, $3, 'queued', NULLIF($4, ''), $5, 0, 0, NULL, $6, NULLIF($7, ''),
        NULLIF($8, ''), $9, $9, NULL, NULL, '{}', $10)
RETURNING `+trainingJobSelectColumns,
		jobID,
		modelKey,
		jobType,
		strings.TrimSpace(input.GPUPool),
		jsonBytes(input.Params),
		ownerUserID,
		strings.TrimSpace(input.OwnerOrgID),
		createdByUserID,
		now,
		jsonBytes(input.Metadata),
	))
	if err != nil {
		return domain.TrainingJobRecord{}, mapPgError(err)
	}

	if _, err := appendTrainingJobEventRow(ctx, tx, domain.AppendTrainingJobEventInput{
		JobID:       job.JobID,
		EventType:   "training.job.created",
		ActorUserID: createdByUserID,
		ActorOrgID:  job.OwnerOrgID,
		TS:          job.CreatedAt,
		Message:     "Training job queued.",
		Metadata: domain.JSONMap{
			"model_key": job.ModelKey,
			"job_type":  job.JobType,
		},
	}); err != nil {
		return domain.TrainingJobRecord{}, err
	}

	if err := tx.Commit(ctx); err != nil {
		return domain.TrainingJobRecord{}, err
	}
	return job, nil
}

func (s *PostgresStore) GetTrainingJob(ctx context.Context, jobID string) (domain.TrainingJobRecord, error) {
	job, err := scanTrainingJobRow(s.pool.QueryRow(ctx, `
SELECT `+trainingJobSelectColumns+`
FROM control_training_jobs
WHERE job_id = $1`, strings.TrimSpace(jobID)))
	if err != nil {
		return domain.TrainingJobRecord{}, mapPgError(err)
	}
	return job, nil
}

func (s *PostgresStore) ListTrainingJobs(ctx context.Context, modelKey string, status string, limit int) ([]domain.TrainingJobRecord, error) {
	rows, err := s.pool.Query(ctx, `
SELECT `+trainingJobSelectColumns+`
FROM control_training_jobs
WHERE ($1::text = '' OR model_key = $1)
  AND ($2::text = '' OR status = $2)
ORDER BY created_at DESC, job_id ASC
LIMIT $3`,
		strings.TrimSpace(modelKey),
		strings.ToLower(strings.TrimSpace(status)),
		limit32(limit, 200),
	)
	if err != nil {
		return nil, mapPgError(err)
	}
	defer rows.Close()
	records := []domain.TrainingJobRecord{}
	for rows.Next() {
		record, err := scanTrainingJobRow(rows)
		if err != nil {
			return nil, mapPgError(err)
		}
		records = append(records, record)
	}
	if err := rows.Err(); err != nil {
		return nil, mapPgError(err)
	}
	return records, nil
}

func (s *PostgresStore) UpdateTrainingJobStatus(ctx context.Context, input domain.UpdateTrainingJobStatusInput) (domain.TrainingJobRecord, error) {
	status := strings.ToLower(strings.TrimSpace(input.Status))
	if !validTrainingJobStatus(status) {
		return domain.TrainingJobRecord{}, errors.New("invalid training job status")
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.TrainingJobRecord{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()

	existing, err := scanTrainingJobRow(tx.QueryRow(ctx, `
SELECT `+trainingJobSelectColumns+`
FROM control_training_jobs
WHERE job_id = $1
FOR UPDATE`, strings.TrimSpace(input.JobID)))
	if err != nil {
		return domain.TrainingJobRecord{}, mapPgError(err)
	}
	now := domain.Now()
	progressCompleted := clampTrainingJobProgress(input.ProgressCompleted, input.ProgressTotal)
	progressTotal := input.ProgressTotal
	if progressTotal <= 0 {
		progressTotal = existing.ProgressTotal
	}
	if progressTotal < progressCompleted {
		progressTotal = progressCompleted
	}
	startedAt := existing.StartedAt
	if status == "running" && startedAt.IsZero() {
		startedAt = now
	}
	completedAt := existing.CompletedAt
	if trainingJobStatusIsTerminal(status) {
		completedAt = now
	} else {
		completedAt = time.Time{}
	}
	outputSummary := existing.OutputSummary
	if input.OutputSummary != nil {
		outputSummary = input.OutputSummary
	}
	job, err := scanTrainingJobRow(tx.QueryRow(ctx, `
UPDATE control_training_jobs
SET status = $2,
    progress_completed = $3,
    progress_total = $4,
    error = NULLIF($5, ''),
    updated_at = $6,
    started_at = $7,
    completed_at = $8,
    output_summary = $9
WHERE job_id = $1
RETURNING `+trainingJobSelectColumns,
		existing.JobID,
		status,
		progressCompleted,
		progressTotal,
		strings.TrimSpace(input.Error),
		now,
		nullableTimestamptz(startedAt),
		nullableTimestamptz(completedAt),
		jsonBytes(outputSummary),
	))
	if err != nil {
		return domain.TrainingJobRecord{}, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.TrainingJobRecord{}, err
	}
	return job, nil
}

func (s *PostgresStore) AppendTrainingJobEvent(ctx context.Context, input domain.AppendTrainingJobEventInput) (domain.TrainingJobEventRecord, error) {
	return appendTrainingJobEventRow(ctx, s.pool, input)
}

func (s *PostgresStore) ListTrainingJobEvents(ctx context.Context, jobID string, limit int) ([]domain.TrainingJobEventRecord, error) {
	rows, err := s.pool.Query(ctx, `
SELECT event_id, job_id, sequence, event_type, COALESCE(actor_user_id, ''),
       COALESCE(actor_org_id, ''), ts, COALESCE(message, ''), metadata
FROM control_training_job_events
WHERE job_id = $1
ORDER BY sequence ASC, event_id ASC
LIMIT $2`,
		strings.TrimSpace(jobID),
		limit32(limit, 200),
	)
	if err != nil {
		return nil, mapPgError(err)
	}
	defer rows.Close()
	records := []domain.TrainingJobEventRecord{}
	for rows.Next() {
		record, err := scanTrainingJobEventRow(rows)
		if err != nil {
			return nil, mapPgError(err)
		}
		records = append(records, record)
	}
	if err := rows.Err(); err != nil {
		return nil, mapPgError(err)
	}
	return records, nil
}

func (s *PostgresStore) AcquireTrainingJobLease(ctx context.Context, input domain.AcquireTrainingJobLeaseInput) (domain.TrainingJobLeaseRecord, domain.TrainingJobRecord, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.TrainingJobLeaseRecord{}, domain.TrainingJobRecord{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()

	job, err := scanTrainingJobRow(tx.QueryRow(ctx, `
SELECT `+trainingJobSelectColumns+`
FROM control_training_jobs
WHERE job_id = $1
FOR UPDATE`, strings.TrimSpace(input.JobID)))
	if err != nil {
		return domain.TrainingJobLeaseRecord{}, domain.TrainingJobRecord{}, mapPgError(err)
	}
	if trainingJobStatusIsTerminal(job.Status) {
		return domain.TrainingJobLeaseRecord{}, domain.TrainingJobRecord{}, ErrConflict
	}
	now := leaseNow(input.Now)
	existingLease, err := scanTrainingJobLeaseRow(tx.QueryRow(ctx, `
SELECT job_id, worker_id, lease_token, lease_expires_at, created_at, updated_at
FROM control_training_job_leases
WHERE job_id = $1
FOR UPDATE`, job.JobID))
	if err == nil && existingLease.LeaseExpiresAt.After(now) {
		return domain.TrainingJobLeaseRecord{}, domain.TrainingJobRecord{}, ErrConflict
	}
	if err != nil && !errors.Is(err, ErrNotFound) {
		return domain.TrainingJobLeaseRecord{}, domain.TrainingJobRecord{}, err
	}
	lease, err := scanTrainingJobLeaseRow(tx.QueryRow(ctx, `
INSERT INTO control_training_job_leases (job_id, worker_id, lease_token, lease_expires_at, created_at, updated_at)
VALUES ($1, $2, $3, $4, $5, $5)
ON CONFLICT (job_id) DO UPDATE
SET worker_id = EXCLUDED.worker_id,
    lease_token = EXCLUDED.lease_token,
    lease_expires_at = EXCLUDED.lease_expires_at,
    updated_at = EXCLUDED.updated_at
RETURNING job_id, worker_id, lease_token, lease_expires_at, created_at, updated_at`,
		job.JobID,
		strings.TrimSpace(input.WorkerID),
		domain.NewID("training_lease"),
		now.Add(positiveLeaseTTL(input.TTL)),
		now,
	))
	if err != nil {
		return domain.TrainingJobLeaseRecord{}, domain.TrainingJobRecord{}, mapPgError(err)
	}
	job, err = scanTrainingJobRow(tx.QueryRow(ctx, `
UPDATE control_training_jobs
SET status = 'running',
    updated_at = $2,
    started_at = COALESCE(started_at, $2)
WHERE job_id = $1
RETURNING `+trainingJobSelectColumns,
		job.JobID,
		now,
	))
	if err != nil {
		return domain.TrainingJobLeaseRecord{}, domain.TrainingJobRecord{}, mapPgError(err)
	}
	if _, err := appendTrainingJobEventRow(ctx, tx, domain.AppendTrainingJobEventInput{
		JobID:     job.JobID,
		EventType: "training.job.leased",
		TS:        now,
		Message:   "Training job leased.",
		Metadata: domain.JSONMap{
			"worker_id":        lease.WorkerID,
			"lease_expires_at": lease.LeaseExpiresAt.UTC().Format(time.RFC3339Nano),
		},
	}); err != nil {
		return domain.TrainingJobLeaseRecord{}, domain.TrainingJobRecord{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.TrainingJobLeaseRecord{}, domain.TrainingJobRecord{}, err
	}
	return lease, job, nil
}

func (s *PostgresStore) RenewTrainingJobLease(ctx context.Context, input domain.RenewTrainingJobLeaseInput) (domain.TrainingJobLeaseRecord, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.TrainingJobLeaseRecord{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()

	var status string
	if err := tx.QueryRow(ctx, `
SELECT status
FROM control_training_jobs
WHERE job_id = $1
FOR UPDATE`, strings.TrimSpace(input.JobID)).Scan(&status); err != nil {
		return domain.TrainingJobLeaseRecord{}, mapPgError(err)
	}
	if trainingJobStatusIsTerminal(status) {
		return domain.TrainingJobLeaseRecord{}, ErrConflict
	}
	existing, err := scanTrainingJobLeaseRow(tx.QueryRow(ctx, `
SELECT job_id, worker_id, lease_token, lease_expires_at, created_at, updated_at
FROM control_training_job_leases
WHERE job_id = $1
FOR UPDATE`, strings.TrimSpace(input.JobID)))
	if err != nil {
		return domain.TrainingJobLeaseRecord{}, err
	}
	// Token match alone authorizes renewal — even of an EXPIRED lease (the
	// run-lease revival rule): a trainer that survived a control-plane outage
	// must be able to revive its lease and keep its (expensive, GPU-hours)
	// computation. If release/recovery already removed the lease row, the
	// lookup above returns the authoritative not-found instead.
	if existing.LeaseToken != strings.TrimSpace(input.LeaseToken) {
		return domain.TrainingJobLeaseRecord{}, ErrConflict
	}
	now := leaseNow(input.Now)
	lease, err := scanTrainingJobLeaseRow(tx.QueryRow(ctx, `
UPDATE control_training_job_leases
SET lease_expires_at = $3,
    updated_at = $4
WHERE job_id = $1 AND lease_token = $2
RETURNING job_id, worker_id, lease_token, lease_expires_at, created_at, updated_at`,
		strings.TrimSpace(input.JobID),
		strings.TrimSpace(input.LeaseToken),
		now.Add(positiveLeaseTTL(input.TTL)),
		now,
	))
	if err != nil {
		return domain.TrainingJobLeaseRecord{}, mapPgError(err)
	}
	if _, err := tx.Exec(ctx, `UPDATE control_training_jobs SET updated_at = $2 WHERE job_id = $1`, strings.TrimSpace(input.JobID), now); err != nil {
		return domain.TrainingJobLeaseRecord{}, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.TrainingJobLeaseRecord{}, err
	}
	return lease, nil
}

func (s *PostgresStore) ReleaseTrainingJobLease(ctx context.Context, input domain.ReleaseTrainingJobLeaseInput) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	jobID := strings.TrimSpace(input.JobID)
	if err := tx.QueryRow(ctx, `SELECT job_id FROM control_training_jobs WHERE job_id = $1`, jobID).Scan(&jobID); err != nil {
		return mapPgError(err)
	}
	tag, err := tx.Exec(ctx, `DELETE FROM control_training_job_leases WHERE job_id = $1 AND lease_token = $2`, jobID, strings.TrimSpace(input.LeaseToken))
	if err != nil {
		return mapPgError(err)
	}
	if tag.RowsAffected() > 0 {
		return tx.Commit(ctx)
	}
	var activeToken string
	err = tx.QueryRow(ctx, `SELECT lease_token FROM control_training_job_leases WHERE job_id = $1`, jobID).Scan(&activeToken)
	if errors.Is(err, pgx.ErrNoRows) {
		return tx.Commit(ctx)
	}
	if err != nil {
		return mapPgError(err)
	}
	return ErrConflict
}

func (s *PostgresStore) CreateTrainingGoldSetDraft(ctx context.Context, input domain.CreateTrainingGoldSetInput) (domain.TrainingGoldSetRecord, error) {
	modelKey := strings.TrimSpace(input.ModelKey)
	if modelKey == "" {
		return domain.TrainingGoldSetRecord{}, errors.New("gold set model key is required")
	}
	goldSetID := strings.TrimSpace(input.GoldSetID)
	if goldSetID == "" {
		goldSetID = domain.NewID("gold_set")
	}
	createdByUserID := strings.TrimSpace(input.CreatedByUserID)
	if createdByUserID == "" {
		createdByUserID = "local-user"
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.TrainingGoldSetRecord{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()

	var modelExists bool
	if err := tx.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM control_training_models WHERE model_key = $1)`, modelKey).Scan(&modelExists); err != nil {
		return domain.TrainingGoldSetRecord{}, mapPgError(err)
	}
	if !modelExists {
		return domain.TrainingGoldSetRecord{}, ErrNotFound
	}
	var openExists bool
	if err := tx.QueryRow(ctx, `
SELECT EXISTS(
  SELECT 1 FROM control_training_gold_sets
  WHERE model_key = $1 AND status IN ('draft', 'freezing')
)`, modelKey).Scan(&openExists); err != nil {
		return domain.TrainingGoldSetRecord{}, mapPgError(err)
	}
	if openExists {
		return domain.TrainingGoldSetRecord{}, ErrConflict
	}
	record, err := scanTrainingGoldSetRow(tx.QueryRow(ctx, `
INSERT INTO control_training_gold_sets (
  gold_set_id, model_key, version, content_hash, item_count, label_stats, strata_summary,
  split_manifest_uri, provenance, status, created_at, created_by_user_id, frozen_at
)
VALUES ($1, $2,
        (SELECT COALESCE(MAX(version), 0) + 1 FROM control_training_gold_sets WHERE model_key = $2),
        NULL, 0, '{}', '{}', NULL, '{}', 'draft', $3, $4, NULL)
RETURNING `+trainingGoldSetSelectColumns,
		goldSetID,
		modelKey,
		domain.Now(),
		createdByUserID,
	))
	if err != nil {
		return domain.TrainingGoldSetRecord{}, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.TrainingGoldSetRecord{}, err
	}
	return record, nil
}

func (s *PostgresStore) GetTrainingGoldSet(ctx context.Context, goldSetID string) (domain.TrainingGoldSetRecord, error) {
	record, err := scanTrainingGoldSetRow(s.pool.QueryRow(ctx, `
SELECT `+trainingGoldSetSelectColumns+`
FROM control_training_gold_sets
WHERE gold_set_id = $1`, strings.TrimSpace(goldSetID)))
	if err != nil {
		return domain.TrainingGoldSetRecord{}, mapPgError(err)
	}
	return record, nil
}

func (s *PostgresStore) GetCurrentTrainingGoldSet(ctx context.Context, modelKey string) (domain.TrainingGoldSetRecord, error) {
	record, err := scanTrainingGoldSetRow(s.pool.QueryRow(ctx, `
SELECT `+trainingGoldSetSelectColumns+`
FROM control_training_gold_sets
WHERE model_key = $1 AND status = 'frozen'
ORDER BY version DESC
LIMIT 1`, strings.TrimSpace(modelKey)))
	if err != nil {
		return domain.TrainingGoldSetRecord{}, mapPgError(err)
	}
	return record, nil
}

func (s *PostgresStore) ListTrainingGoldSets(ctx context.Context, modelKey string, limit int) ([]domain.TrainingGoldSetRecord, error) {
	rows, err := s.pool.Query(ctx, `
SELECT `+trainingGoldSetSelectColumns+`
FROM control_training_gold_sets
WHERE model_key = $1
ORDER BY version DESC
LIMIT $2`, strings.TrimSpace(modelKey), limit32(limit, 200))
	if err != nil {
		return nil, mapPgError(err)
	}
	defer rows.Close()
	records := []domain.TrainingGoldSetRecord{}
	for rows.Next() {
		record, err := scanTrainingGoldSetRow(rows)
		if err != nil {
			return nil, mapPgError(err)
		}
		records = append(records, record)
	}
	if err := rows.Err(); err != nil {
		return nil, mapPgError(err)
	}
	return records, nil
}

func (s *PostgresStore) UpdateTrainingGoldSetState(ctx context.Context, input domain.UpdateTrainingGoldSetStateInput) (domain.TrainingGoldSetRecord, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.TrainingGoldSetRecord{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()

	record, err := scanTrainingGoldSetRow(tx.QueryRow(ctx, `
SELECT `+trainingGoldSetSelectColumns+`
FROM control_training_gold_sets
WHERE gold_set_id = $1
FOR UPDATE`, strings.TrimSpace(input.GoldSetID)))
	if err != nil {
		return domain.TrainingGoldSetRecord{}, mapPgError(err)
	}
	status := strings.ToLower(strings.TrimSpace(input.Status))
	if status != "" && status != record.Status {
		if !trainingGoldSetTransitionAllowed(record.Status, status) {
			return domain.TrainingGoldSetRecord{}, ErrConflict
		}
		record.Status = status
	}
	if contentHash := strings.TrimSpace(input.ContentHash); contentHash != "" {
		record.ContentHash = contentHash
	}
	if input.ItemCount > 0 {
		record.ItemCount = input.ItemCount
	}
	if input.LabelStats != nil {
		record.LabelStats = input.LabelStats
	}
	if input.StrataSummary != nil {
		record.StrataSummary = input.StrataSummary
	}
	if input.Provenance != nil {
		record.Provenance = input.Provenance
	}
	if uri := strings.TrimSpace(input.SplitManifestURI); uri != "" {
		record.SplitManifestURI = uri
	}
	if input.FrozenAt != nil {
		record.FrozenAt = utcTimePtr(input.FrozenAt)
	}
	if record.Status == "frozen" && record.FrozenAt == nil {
		now := domain.Now()
		record.FrozenAt = &now
	}
	record, err = scanTrainingGoldSetRow(tx.QueryRow(ctx, `
UPDATE control_training_gold_sets
SET content_hash = NULLIF($2, ''),
    item_count = $3,
    label_stats = $4,
    strata_summary = $5,
    split_manifest_uri = NULLIF($6, ''),
    provenance = $7,
    status = $8,
    frozen_at = $9
WHERE gold_set_id = $1
RETURNING `+trainingGoldSetSelectColumns,
		record.GoldSetID,
		record.ContentHash,
		record.ItemCount,
		jsonBytes(record.LabelStats),
		jsonBytes(record.StrataSummary),
		record.SplitManifestURI,
		jsonBytes(record.Provenance),
		record.Status,
		record.FrozenAt,
	))
	if err != nil {
		return domain.TrainingGoldSetRecord{}, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.TrainingGoldSetRecord{}, err
	}
	return record, nil
}

func (s *PostgresStore) InsertTrainingGoldItems(ctx context.Context, goldSetID string, items []domain.TrainingGoldItemInput) (int, error) {
	goldSetID = strings.TrimSpace(goldSetID)
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return 0, err
	}
	defer func() { _ = tx.Rollback(ctx) }()

	var status string
	if err := tx.QueryRow(ctx, `
SELECT status
FROM control_training_gold_sets
WHERE gold_set_id = $1
FOR UPDATE`, goldSetID).Scan(&status); err != nil {
		return 0, mapPgError(err)
	}
	if !trainingGoldSetAcceptsItems(status) {
		return 0, ErrConflict
	}
	for _, item := range items {
		itemID := strings.TrimSpace(item.ItemID)
		if itemID == "" {
			return 0, errors.New("gold item id is required")
		}
		if _, err := tx.Exec(ctx, `
INSERT INTO control_training_gold_items (
  gold_set_id, item_id, source_ref, slice, label_kind, content_sha256, phash,
  gt_label_sha256, gt_label_uri, width, height, metadata, footprint_geom, strata_tags
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)`,
			goldSetID,
			itemID,
			jsonBytes(item.SourceRef),
			strings.TrimSpace(item.Slice),
			strings.TrimSpace(item.LabelKind),
			strings.TrimSpace(item.ContentSHA256),
			item.Phash,
			strings.TrimSpace(item.GTLabelSHA256),
			strings.TrimSpace(item.GTLabelURI),
			item.Width,
			item.Height,
			jsonBytes(item.Metadata),
			jsonBytesOrNil(item.FootprintGeom),
			jsonBytes(item.StrataTags),
		); err != nil {
			return 0, mapPgError(err)
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return 0, err
	}
	return len(items), nil
}

func (s *PostgresStore) ListTrainingGoldItems(ctx context.Context, goldSetID string, limit int, offset int) ([]domain.TrainingGoldItemRecord, error) {
	rows, err := s.pool.Query(ctx, `
SELECT gold_set_id, item_id, source_ref, slice, label_kind, content_sha256, phash,
       gt_label_sha256, gt_label_uri, width, height, metadata, footprint_geom, strata_tags
FROM control_training_gold_items
WHERE gold_set_id = $1
ORDER BY item_id ASC
LIMIT $2 OFFSET $3`,
		strings.TrimSpace(goldSetID),
		limit32(limit, 500),
		int32(max(0, offset)),
	)
	if err != nil {
		return nil, mapPgError(err)
	}
	defer rows.Close()
	records := []domain.TrainingGoldItemRecord{}
	for rows.Next() {
		record, err := scanTrainingGoldItemRow(rows)
		if err != nil {
			return nil, mapPgError(err)
		}
		records = append(records, record)
	}
	if err := rows.Err(); err != nil {
		return nil, mapPgError(err)
	}
	return records, nil
}

func (s *PostgresStore) CreateTrainingModelVersion(ctx context.Context, input domain.CreateTrainingModelVersionInput) (domain.TrainingModelVersionRecord, error) {
	lineageID := strings.TrimSpace(input.LineageID)
	modelKey := strings.TrimSpace(input.ModelKey)
	if lineageID == "" {
		return domain.TrainingModelVersionRecord{}, errors.New("model version lineage id is required")
	}
	if modelKey == "" {
		return domain.TrainingModelVersionRecord{}, errors.New("model version model key is required")
	}
	status := strings.ToLower(strings.TrimSpace(input.Status))
	if status == "" {
		status = "candidate"
	}
	if !validTrainingModelVersionStatus(status) {
		return domain.TrainingModelVersionRecord{}, errors.New("invalid training model version status")
	}
	versionID := strings.TrimSpace(input.VersionID)
	if versionID == "" {
		versionID = domain.NewID("version")
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.TrainingModelVersionRecord{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()

	var refsExist bool
	if err := tx.QueryRow(ctx, `
SELECT EXISTS(SELECT 1 FROM control_training_lineages WHERE lineage_id = $1)
   AND EXISTS(SELECT 1 FROM control_training_models WHERE model_key = $2)`,
		lineageID, modelKey).Scan(&refsExist); err != nil {
		return domain.TrainingModelVersionRecord{}, mapPgError(err)
	}
	if !refsExist {
		return domain.TrainingModelVersionRecord{}, ErrNotFound
	}
	now := domain.Now()
	record, err := scanTrainingModelVersionRow(tx.QueryRow(ctx, `
INSERT INTO control_training_model_versions (
  version_id, lineage_id, model_key, status, is_frozen, weights_uri, source_job_id,
  artifact_run_id, metrics, metadata, activated_at, created_at, updated_at
)
VALUES ($1, $2, $3, $4, false, NULLIF($5, ''), NULLIF($6, ''), NULL, $7, $8, NULL, $9, $9)
RETURNING version_id, lineage_id, model_key, status, is_frozen, COALESCE(weights_uri, ''),
          COALESCE(source_job_id, ''), COALESCE(artifact_run_id, ''), metrics, metadata,
          activated_at, created_at, updated_at`,
		versionID,
		lineageID,
		modelKey,
		status,
		strings.TrimSpace(input.WeightsURI),
		strings.TrimSpace(input.SourceJobID),
		jsonBytes(input.Metrics),
		jsonBytes(input.Metadata),
		now,
	))
	if err != nil {
		return domain.TrainingModelVersionRecord{}, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.TrainingModelVersionRecord{}, err
	}
	return record, nil
}

func (s *PostgresStore) UpdateTrainingModelVersion(ctx context.Context, input domain.UpdateTrainingModelVersionInput) (domain.TrainingModelVersionRecord, error) {
	status := strings.ToLower(strings.TrimSpace(input.Status))
	if status != "" && !validTrainingModelVersionStatus(status) {
		return domain.TrainingModelVersionRecord{}, errors.New("invalid training model version status")
	}
	record, err := scanTrainingModelVersionRow(s.pool.QueryRow(ctx, `
UPDATE control_training_model_versions
SET status = COALESCE(NULLIF($2, ''), status),
    metrics = COALESCE($3, metrics),
    metadata = COALESCE($4, metadata),
    activated_at = COALESCE($5, activated_at),
    updated_at = $6
WHERE version_id = $1
RETURNING version_id, lineage_id, model_key, status, is_frozen, COALESCE(weights_uri, ''),
          COALESCE(source_job_id, ''), COALESCE(artifact_run_id, ''), metrics, metadata,
          activated_at, created_at, updated_at`,
		strings.TrimSpace(input.VersionID),
		status,
		jsonBytesOrNil(input.Metrics),
		jsonBytesOrNil(input.Metadata),
		input.ActivatedAt,
		domain.Now(),
	))
	if err != nil {
		return domain.TrainingModelVersionRecord{}, mapPgError(err)
	}
	return record, nil
}

func (s *PostgresStore) AppendTrainingModelVersionEvent(ctx context.Context, input domain.AppendTrainingModelVersionEventInput) (domain.TrainingModelVersionEventRecord, error) {
	versionID := strings.TrimSpace(input.VersionID)
	if versionID == "" {
		return domain.TrainingModelVersionEventRecord{}, errors.New("version event version id is required")
	}
	eventID := strings.TrimSpace(input.EventID)
	if eventID == "" {
		eventID = domain.NewID("version_event")
	}
	record, err := scanTrainingModelVersionEventRow(s.pool.QueryRow(ctx, `
INSERT INTO control_training_model_version_events (
  event_id, version_id, event_type, actor_user_id, from_status, to_status,
  benchmark_run_id, gold_set_content_hash, reason, created_at
)
VALUES ($1, $2, $3, $4, NULLIF($5, ''), NULLIF($6, ''), NULLIF($7, ''), NULLIF($8, ''), NULLIF($9, ''), $10)
RETURNING event_id, version_id, event_type, actor_user_id, COALESCE(from_status, ''),
          COALESCE(to_status, ''), COALESCE(benchmark_run_id, ''),
          COALESCE(gold_set_content_hash, ''), COALESCE(reason, ''), created_at`,
		eventID,
		versionID,
		strings.TrimSpace(input.EventType),
		strings.TrimSpace(input.ActorUserID),
		strings.TrimSpace(input.FromStatus),
		strings.TrimSpace(input.ToStatus),
		strings.TrimSpace(input.BenchmarkRunID),
		strings.TrimSpace(input.GoldSetContentHash),
		strings.TrimSpace(input.Reason),
		domain.Now(),
	))
	if err != nil {
		return domain.TrainingModelVersionEventRecord{}, mapPgError(err)
	}
	return record, nil
}

func (s *PostgresStore) ListTrainingModelVersionEvents(ctx context.Context, versionID string, limit int) ([]domain.TrainingModelVersionEventRecord, error) {
	rows, err := s.pool.Query(ctx, `
SELECT event_id, version_id, event_type, actor_user_id, COALESCE(from_status, ''),
       COALESCE(to_status, ''), COALESCE(benchmark_run_id, ''),
       COALESCE(gold_set_content_hash, ''), COALESCE(reason, ''), created_at
FROM control_training_model_version_events
WHERE version_id = $1
ORDER BY created_at DESC, event_id DESC
LIMIT $2`,
		strings.TrimSpace(versionID),
		limit32(limit, 200),
	)
	if err != nil {
		return nil, mapPgError(err)
	}
	defer rows.Close()
	records := []domain.TrainingModelVersionEventRecord{}
	for rows.Next() {
		record, err := scanTrainingModelVersionEventRow(rows)
		if err != nil {
			return nil, mapPgError(err)
		}
		records = append(records, record)
	}
	if err := rows.Err(); err != nil {
		return nil, mapPgError(err)
	}
	return records, nil
}

func (s *PostgresStore) CreateTrainingRetrainRequest(ctx context.Context, input domain.CreateTrainingRetrainRequestInput) (domain.TrainingRetrainRequestRecord, error) {
	modelKey := strings.TrimSpace(input.ModelKey)
	if modelKey == "" {
		return domain.TrainingRetrainRequestRecord{}, errors.New("retrain request model key is required")
	}
	requestID := strings.TrimSpace(input.RequestID)
	if requestID == "" {
		requestID = domain.NewID("retrain_request")
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.TrainingRetrainRequestRecord{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()

	var modelExists bool
	if err := tx.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM control_training_models WHERE model_key = $1)`, modelKey).Scan(&modelExists); err != nil {
		return domain.TrainingRetrainRequestRecord{}, mapPgError(err)
	}
	if !modelExists {
		return domain.TrainingRetrainRequestRecord{}, ErrNotFound
	}
	record, err := scanTrainingRetrainRequestRow(tx.QueryRow(ctx, `
INSERT INTO control_training_retrain_requests (
  request_id, model_key, training_job_id, status, note, gating_summary,
  requested_by_user_id, created_at
)
VALUES ($1, $2, NULLIF($3, ''), 'queued', NULLIF($4, ''), $5, NULLIF($6, ''), $7)
RETURNING `+trainingRetrainRequestSelectColumns,
		requestID,
		modelKey,
		strings.TrimSpace(input.TrainingJobID),
		strings.TrimSpace(input.Note),
		jsonBytes(input.GatingSummary),
		strings.TrimSpace(input.RequestedByUserID),
		domain.Now(),
	))
	if err != nil {
		return domain.TrainingRetrainRequestRecord{}, mapPgError(err)
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.TrainingRetrainRequestRecord{}, err
	}
	return record, nil
}

func (s *PostgresStore) UpdateTrainingRetrainRequest(ctx context.Context, input domain.UpdateTrainingRetrainRequestInput) (domain.TrainingRetrainRequestRecord, error) {
	record, err := scanTrainingRetrainRequestRow(s.pool.QueryRow(ctx, `
UPDATE control_training_retrain_requests
SET status = COALESCE(NULLIF($2, ''), status),
    error = COALESCE(NULLIF($3, ''), error),
    model_version = COALESCE(NULLIF($4, ''), model_version),
    training_job_id = COALESCE(NULLIF($5, ''), training_job_id),
    started_at = COALESCE($6, started_at),
    finished_at = COALESCE($7, finished_at)
WHERE request_id = $1
RETURNING `+trainingRetrainRequestSelectColumns,
		strings.TrimSpace(input.RequestID),
		strings.ToLower(strings.TrimSpace(input.Status)),
		strings.TrimSpace(input.Error),
		strings.TrimSpace(input.ModelVersion),
		strings.TrimSpace(input.TrainingJobID),
		input.StartedAt,
		input.FinishedAt,
	))
	if err != nil {
		return domain.TrainingRetrainRequestRecord{}, mapPgError(err)
	}
	return record, nil
}

func (s *PostgresStore) InsertTrainingBenchmarkRun(ctx context.Context, input domain.InsertTrainingBenchmarkRunInput) (domain.TrainingBenchmarkRunRecord, error) {
	modelVersionID := strings.TrimSpace(input.ModelVersionID)
	if modelVersionID == "" {
		return domain.TrainingBenchmarkRunRecord{}, errors.New("benchmark run model version id is required")
	}
	goldSetID := strings.TrimSpace(input.GoldSetID)
	if goldSetID == "" {
		return domain.TrainingBenchmarkRunRecord{}, errors.New("benchmark run gold set id is required")
	}
	runID := strings.TrimSpace(input.RunID)
	if runID == "" {
		runID = domain.NewID("benchmark_run")
	}
	record, err := scanTrainingBenchmarkRunRow(s.pool.QueryRow(ctx, `
INSERT INTO control_training_benchmark_runs (
  run_id, model_version_id, gold_set_id, gold_set_content_hash, metric_schema,
  kernel_version, metrics, guardrails_passed, guardrails_reasons, report_uri, created_at
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
RETURNING run_id, model_version_id, gold_set_id, gold_set_content_hash, metric_schema,
          kernel_version, metrics, guardrails_passed, guardrails_reasons, report_uri, created_at`,
		runID,
		modelVersionID,
		goldSetID,
		strings.TrimSpace(input.GoldSetContentHash),
		strings.TrimSpace(input.MetricSchema),
		strings.TrimSpace(input.KernelVersion),
		jsonBytes(input.Metrics),
		input.GuardrailsPassed,
		jsonStringsBytes(input.GuardrailsReasons),
		strings.TrimSpace(input.ReportURI),
		domain.Now(),
	))
	if err != nil {
		return domain.TrainingBenchmarkRunRecord{}, mapPgError(err)
	}
	return record, nil
}

func (s *PostgresStore) GetLatestTrainingBenchmarkRun(ctx context.Context, modelVersionID string, goldSetContentHash string) (domain.TrainingBenchmarkRunRecord, error) {
	record, err := scanTrainingBenchmarkRunRow(s.pool.QueryRow(ctx, `
SELECT run_id, model_version_id, gold_set_id, gold_set_content_hash, metric_schema,
       kernel_version, metrics, guardrails_passed, guardrails_reasons, report_uri, created_at
FROM control_training_benchmark_runs
WHERE model_version_id = $1
  AND ($2::text = '' OR gold_set_content_hash = $2)
ORDER BY created_at DESC, run_id DESC
LIMIT 1`,
		strings.TrimSpace(modelVersionID),
		strings.TrimSpace(goldSetContentHash),
	))
	if err != nil {
		return domain.TrainingBenchmarkRunRecord{}, mapPgError(err)
	}
	return record, nil
}

func (s *PostgresStore) UpsertTrainingModelStatus(ctx context.Context, record domain.TrainingModelStatusRecord) (domain.TrainingModelStatusRecord, error) {
	modelKey := strings.TrimSpace(record.ModelKey)
	if modelKey == "" {
		return domain.TrainingModelStatusRecord{}, errors.New("model status model key is required")
	}
	var modelExists bool
	if err := s.pool.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM control_training_models WHERE model_key = $1)`, modelKey).Scan(&modelExists); err != nil {
		return domain.TrainingModelStatusRecord{}, mapPgError(err)
	}
	if !modelExists {
		return domain.TrainingModelStatusRecord{}, ErrNotFound
	}
	stored, err := scanTrainingModelStatusRow(s.pool.QueryRow(ctx, `
INSERT INTO control_training_model_status (
  model_key, dataset_name, dataset_id, model_health, reviewed_images, unreviewed_images,
  class_counts, per_class_new_objects, unsupported_class_counts, last_sync_at,
  last_retrain_at, active_model_version, retrain_gate, retrain_gate_reasons,
  retrain_gate_counts, retrain_gate_thresholds
)
VALUES ($1, NULLIF($2, ''), NULLIF($3, ''), NULLIF($4, ''), $5, $6, $7, $8, $9, $10, $11,
        NULLIF($12, ''), $13, $14, $15, $16)
ON CONFLICT (model_key) DO UPDATE
SET dataset_name = EXCLUDED.dataset_name,
    dataset_id = EXCLUDED.dataset_id,
    model_health = EXCLUDED.model_health,
    reviewed_images = EXCLUDED.reviewed_images,
    unreviewed_images = EXCLUDED.unreviewed_images,
    class_counts = EXCLUDED.class_counts,
    per_class_new_objects = EXCLUDED.per_class_new_objects,
    unsupported_class_counts = EXCLUDED.unsupported_class_counts,
    last_sync_at = EXCLUDED.last_sync_at,
    last_retrain_at = EXCLUDED.last_retrain_at,
    active_model_version = EXCLUDED.active_model_version,
    retrain_gate = EXCLUDED.retrain_gate,
    retrain_gate_reasons = EXCLUDED.retrain_gate_reasons,
    retrain_gate_counts = EXCLUDED.retrain_gate_counts,
    retrain_gate_thresholds = EXCLUDED.retrain_gate_thresholds
RETURNING model_key, COALESCE(dataset_name, ''), COALESCE(dataset_id, ''), COALESCE(model_health, ''),
          reviewed_images, unreviewed_images, class_counts, per_class_new_objects,
          unsupported_class_counts, last_sync_at, last_retrain_at,
          COALESCE(active_model_version, ''), retrain_gate, retrain_gate_reasons,
          retrain_gate_counts, retrain_gate_thresholds`,
		modelKey,
		strings.TrimSpace(record.DatasetName),
		strings.TrimSpace(record.DatasetID),
		strings.TrimSpace(record.ModelHealth),
		record.ReviewedImages,
		record.UnreviewedImages,
		jsonBytes(record.ClassCounts),
		jsonBytes(record.PerClassNewObjects),
		jsonBytes(record.UnsupportedClassCounts),
		record.LastSyncAt,
		record.LastRetrainAt,
		strings.TrimSpace(record.ActiveModelVersion),
		record.RetrainGate,
		jsonStringsBytes(record.RetrainGateReasons),
		jsonBytes(record.RetrainGateCounts),
		jsonBytes(record.RetrainGateThresholds),
	))
	if err != nil {
		return domain.TrainingModelStatusRecord{}, mapPgError(err)
	}
	return stored, nil
}

func (s *PostgresStore) ListTrainingGuardrailClauses(ctx context.Context, modelKey string) ([]domain.TrainingGuardrailClauseRecord, error) {
	rows, err := s.pool.Query(ctx, `
SELECT model_key, clause_key, metric_path, comparator, value::float8, COALESCE(slice, ''),
       params, enabled, required
FROM control_training_guardrail_clauses
WHERE model_key = $1
ORDER BY clause_key ASC`, strings.TrimSpace(modelKey))
	if err != nil {
		return nil, mapPgError(err)
	}
	defer rows.Close()
	records := []domain.TrainingGuardrailClauseRecord{}
	for rows.Next() {
		var record domain.TrainingGuardrailClauseRecord
		var params []byte
		if err := rows.Scan(
			&record.ModelKey,
			&record.ClauseKey,
			&record.MetricPath,
			&record.Comparator,
			&record.Value,
			&record.Slice,
			&params,
			&record.Enabled,
			&record.Required,
		); err != nil {
			return nil, mapPgError(err)
		}
		record.Params = jsonMap(params)
		records = append(records, record)
	}
	if err := rows.Err(); err != nil {
		return nil, mapPgError(err)
	}
	return records, nil
}

func (s *PostgresStore) GetTrainingGatePolicy(ctx context.Context, modelKey string) (domain.TrainingGatePolicyRecord, error) {
	var record domain.TrainingGatePolicyRecord
	var perClass []byte
	if err := s.pool.QueryRow(ctx, `
SELECT model_key, min_reviewed, min_new_objects, min_per_class_objects, min_days
FROM control_training_gate_policies
WHERE model_key = $1`, strings.TrimSpace(modelKey)).Scan(
		&record.ModelKey,
		&record.MinReviewed,
		&record.MinNewObjects,
		&perClass,
		&record.MinDays,
	); err != nil {
		return domain.TrainingGatePolicyRecord{}, mapPgError(err)
	}
	record.MinPerClassObjects = jsonMap(perClass)
	return record, nil
}

func appendTrainingJobEventRow(ctx context.Context, q trainingRowQuerier, input domain.AppendTrainingJobEventInput) (domain.TrainingJobEventRecord, error) {
	eventID := strings.TrimSpace(input.EventID)
	if eventID == "" {
		eventID = domain.NewID("training_job_event")
	}
	ts := input.TS
	if ts.IsZero() {
		ts = domain.Now()
	}
	event, err := scanTrainingJobEventRow(q.QueryRow(ctx, `
INSERT INTO control_training_job_events (
  event_id, job_id, sequence, event_type, actor_user_id, actor_org_id, ts, message, metadata
)
VALUES (
  $1, $2,
  CASE WHEN $3::bigint > 0 THEN $3::bigint ELSE (SELECT COALESCE(MAX(sequence), 0) + 1 FROM control_training_job_events WHERE job_id = $2) END,
  $4, NULLIF($5, ''), NULLIF($6, ''), $7, NULLIF($8, ''), $9
)
RETURNING event_id, job_id, sequence, event_type, COALESCE(actor_user_id, ''),
          COALESCE(actor_org_id, ''), ts, COALESCE(message, ''), metadata`,
		eventID,
		strings.TrimSpace(input.JobID),
		input.Sequence,
		strings.TrimSpace(input.EventType),
		strings.TrimSpace(input.ActorUserID),
		strings.TrimSpace(input.ActorOrgID),
		ts.UTC(),
		strings.TrimSpace(input.Message),
		jsonBytes(input.Metadata),
	))
	if err != nil {
		return domain.TrainingJobEventRecord{}, mapPgError(err)
	}
	return event, nil
}

func scanTrainingJobRow(row scanner) (domain.TrainingJobRecord, error) {
	var job domain.TrainingJobRecord
	var startedAt, completedAt pgtype.Timestamptz
	var params, outputSummary, metadata []byte
	if err := row.Scan(
		&job.JobID,
		&job.ModelKey,
		&job.JobType,
		&job.Status,
		&job.GPUPool,
		&params,
		&job.ProgressCompleted,
		&job.ProgressTotal,
		&job.Error,
		&job.OwnerUserID,
		&job.OwnerOrgID,
		&job.CreatedByUserID,
		&job.CreatedAt,
		&job.UpdatedAt,
		&startedAt,
		&completedAt,
		&outputSummary,
		&metadata,
	); err != nil {
		return domain.TrainingJobRecord{}, err
	}
	job.Params = jsonMap(params)
	job.OutputSummary = jsonMap(outputSummary)
	job.Metadata = jsonMap(metadata)
	job.CreatedAt = job.CreatedAt.UTC()
	job.UpdatedAt = job.UpdatedAt.UTC()
	job.StartedAt = timeValue(startedAt)
	job.CompletedAt = timeValue(completedAt)
	return job, nil
}

func scanTrainingJobEventRow(row scanner) (domain.TrainingJobEventRecord, error) {
	var event domain.TrainingJobEventRecord
	var metadata []byte
	if err := row.Scan(
		&event.EventID,
		&event.JobID,
		&event.Sequence,
		&event.EventType,
		&event.ActorUserID,
		&event.ActorOrgID,
		&event.TS,
		&event.Message,
		&metadata,
	); err != nil {
		return domain.TrainingJobEventRecord{}, err
	}
	event.TS = event.TS.UTC()
	event.Metadata = jsonMap(metadata)
	return event, nil
}

func scanTrainingJobLeaseRow(row scanner) (domain.TrainingJobLeaseRecord, error) {
	var lease domain.TrainingJobLeaseRecord
	if err := row.Scan(
		&lease.JobID,
		&lease.WorkerID,
		&lease.LeaseToken,
		&lease.LeaseExpiresAt,
		&lease.CreatedAt,
		&lease.UpdatedAt,
	); err != nil {
		return domain.TrainingJobLeaseRecord{}, mapPgError(err)
	}
	lease.LeaseExpiresAt = lease.LeaseExpiresAt.UTC()
	lease.CreatedAt = lease.CreatedAt.UTC()
	lease.UpdatedAt = lease.UpdatedAt.UTC()
	return lease, nil
}

func scanTrainingGoldSetRow(row scanner) (domain.TrainingGoldSetRecord, error) {
	var record domain.TrainingGoldSetRecord
	var labelStats, strataSummary, provenance []byte
	var frozenAt *time.Time
	if err := row.Scan(
		&record.GoldSetID,
		&record.ModelKey,
		&record.Version,
		&record.ContentHash,
		&record.ItemCount,
		&labelStats,
		&strataSummary,
		&record.SplitManifestURI,
		&provenance,
		&record.Status,
		&record.CreatedAt,
		&record.CreatedByUserID,
		&frozenAt,
	); err != nil {
		return domain.TrainingGoldSetRecord{}, err
	}
	record.LabelStats = jsonMap(labelStats)
	record.StrataSummary = jsonMap(strataSummary)
	record.Provenance = jsonMap(provenance)
	record.CreatedAt = record.CreatedAt.UTC()
	record.FrozenAt = utcTimePtr(frozenAt)
	return record, nil
}

func scanTrainingGoldItemRow(row scanner) (domain.TrainingGoldItemRecord, error) {
	var record domain.TrainingGoldItemRecord
	var sourceRef, metadata, footprintGeom, strataTags []byte
	if err := row.Scan(
		&record.GoldSetID,
		&record.ItemID,
		&sourceRef,
		&record.Slice,
		&record.LabelKind,
		&record.ContentSHA256,
		&record.Phash,
		&record.GTLabelSHA256,
		&record.GTLabelURI,
		&record.Width,
		&record.Height,
		&metadata,
		&footprintGeom,
		&strataTags,
	); err != nil {
		return domain.TrainingGoldItemRecord{}, err
	}
	record.SourceRef = jsonMap(sourceRef)
	record.Metadata = jsonMap(metadata)
	record.FootprintGeom = jsonMap(footprintGeom)
	record.StrataTags = jsonMap(strataTags)
	return record, nil
}

func scanTrainingModelVersionEventRow(row scanner) (domain.TrainingModelVersionEventRecord, error) {
	var record domain.TrainingModelVersionEventRecord
	if err := row.Scan(
		&record.EventID,
		&record.VersionID,
		&record.EventType,
		&record.ActorUserID,
		&record.FromStatus,
		&record.ToStatus,
		&record.BenchmarkRunID,
		&record.GoldSetContentHash,
		&record.Reason,
		&record.CreatedAt,
	); err != nil {
		return domain.TrainingModelVersionEventRecord{}, err
	}
	record.CreatedAt = record.CreatedAt.UTC()
	return record, nil
}

func scanTrainingRetrainRequestRow(row scanner) (domain.TrainingRetrainRequestRecord, error) {
	var record domain.TrainingRetrainRequestRecord
	var gating []byte
	var started, finished *time.Time
	if err := row.Scan(
		&record.RequestID,
		&record.ModelKey,
		&record.TrainingJobID,
		&record.Status,
		&record.Note,
		&record.Error,
		&record.ModelVersion,
		&gating,
		&record.BenchmarkReportArtifactID,
		&record.RequestedByUserID,
		&record.CreatedAt,
		&started,
		&finished,
	); err != nil {
		return domain.TrainingRetrainRequestRecord{}, err
	}
	record.GatingSummary = jsonMap(gating)
	record.CreatedAt = record.CreatedAt.UTC()
	record.StartedAt = utcTimePtr(started)
	record.FinishedAt = utcTimePtr(finished)
	return record, nil
}

func scanTrainingBenchmarkRunRow(row scanner) (domain.TrainingBenchmarkRunRecord, error) {
	var record domain.TrainingBenchmarkRunRecord
	var metrics, reasons []byte
	if err := row.Scan(
		&record.RunID,
		&record.ModelVersionID,
		&record.GoldSetID,
		&record.GoldSetContentHash,
		&record.MetricSchema,
		&record.KernelVersion,
		&metrics,
		&record.GuardrailsPassed,
		&reasons,
		&record.ReportURI,
		&record.CreatedAt,
	); err != nil {
		return domain.TrainingBenchmarkRunRecord{}, err
	}
	record.Metrics = jsonMap(metrics)
	record.GuardrailsReasons = jsonStringSlice(reasons)
	record.CreatedAt = record.CreatedAt.UTC()
	return record, nil
}

func scanTrainingModelStatusRow(row scanner) (domain.TrainingModelStatusRecord, error) {
	var record domain.TrainingModelStatusRecord
	var classCounts, perClassNew, unsupported, gateCounts, gateThresholds, gateReasons []byte
	var lastSync, lastRetrain *time.Time
	if err := row.Scan(
		&record.ModelKey,
		&record.DatasetName,
		&record.DatasetID,
		&record.ModelHealth,
		&record.ReviewedImages,
		&record.UnreviewedImages,
		&classCounts,
		&perClassNew,
		&unsupported,
		&lastSync,
		&lastRetrain,
		&record.ActiveModelVersion,
		&record.RetrainGate,
		&gateReasons,
		&gateCounts,
		&gateThresholds,
	); err != nil {
		return domain.TrainingModelStatusRecord{}, err
	}
	record.ClassCounts = jsonMap(classCounts)
	record.PerClassNewObjects = jsonMap(perClassNew)
	record.UnsupportedClassCounts = jsonMap(unsupported)
	record.RetrainGateCounts = jsonMap(gateCounts)
	record.RetrainGateThresholds = jsonMap(gateThresholds)
	record.RetrainGateReasons = jsonStringSlice(gateReasons)
	record.LastSyncAt = utcTimePtr(lastSync)
	record.LastRetrainAt = utcTimePtr(lastRetrain)
	return record, nil
}

func jsonBytesOrNil(value domain.JSONMap) []byte {
	if value == nil {
		return nil
	}
	return jsonBytes(value)
}

func jsonStringsBytes(values []string) []byte {
	if values == nil {
		values = []string{}
	}
	data, _ := json.Marshal(values)
	return data
}

func jsonStringSlice(raw []byte) []string {
	values := []string{}
	if len(raw) == 0 {
		return values
	}
	if err := json.Unmarshal(raw, &values); err != nil {
		return []string{}
	}
	return values
}

func utcTimePtr(value *time.Time) *time.Time {
	if value == nil {
		return nil
	}
	utc := value.UTC()
	return &utc
}
