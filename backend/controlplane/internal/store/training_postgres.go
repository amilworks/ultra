package store

import (
	"context"
	"encoding/json"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"

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
