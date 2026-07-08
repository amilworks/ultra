package store

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

// M4 lifecycle transactions, Postgres side: everything happens inside ONE DB
// transaction with the version rows locked (plan section 8.1's "the
// transaction is the law") - re-reading the benchmark verdict and the gold
// hash under the lock defends against a stale-passing candidate sneaking
// through after gold changed.

func (s *PostgresStore) PromoteTrainingModelVersion(ctx context.Context, input domain.PromoteTrainingModelVersionInput) (domain.TrainingModelVersionRecord, error) {
	versionID := strings.TrimSpace(input.VersionID)
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.TrainingModelVersionRecord{}, mapPgError(err)
	}
	defer func() { _ = tx.Rollback(ctx) }()

	version, err := lockTrainingVersionTx(ctx, tx, versionID)
	if err != nil {
		return domain.TrainingModelVersionRecord{}, err
	}
	var fromStatus, toStatus string
	switch version.Status {
	case "candidate":
		fromStatus, toStatus = "candidate", "canary"
	case "canary":
		fromStatus, toStatus = "canary", "active"
	default:
		return domain.TrainingModelVersionRecord{}, ErrConflict
	}

	gold, goldOK, err := currentFrozenGoldTx(ctx, tx, version.ModelKey)
	if err != nil {
		return domain.TrainingModelVersionRecord{}, err
	}
	reasons := []string{}
	heldOutPending := false
	if !goldOK {
		reasons = append(reasons, "no frozen gold set exists - nothing can be promoted without the gate")
	} else {
		heldOutPending = goldHeldOutState(gold) == "pending_new_survey"
		run, runOK, runErr := latestBenchmarkRunTx(ctx, tx, versionID, gold.ContentHash)
		if runErr != nil {
			return domain.TrainingModelVersionRecord{}, runErr
		}
		switch {
		case !runOK:
			reasons = append(reasons, "no benchmark run on the current gold content hash - benchmark the version first (section 7.5)")
		case !run.GuardrailsPassed:
			reasons = append(reasons, "the latest benchmark on the current gold hash did not pass the gate")
			reasons = append(reasons, run.GuardrailsReasons...)
		default:
			if ready, _ := version.Metrics["promotion_benchmark_ready"].(bool); !ready {
				reasons = append(reasons, "promotion_benchmark_ready is not set on the version metrics")
			}
			if input.RequiredMetricKeysCheck != nil {
				if keysErr := input.RequiredMetricKeysCheck(run.Metrics); keysErr != nil {
					reasons = append(reasons, "metrics blob failed the required-keys check: "+keysErr.Error())
				}
			}
		}
	}
	if toStatus == "active" && heldOutPending && strings.TrimSpace(input.OverrideReason) == "" {
		reasons = append(reasons,
			"held-out slice is pending new survey data - promoting to active requires an audited override_reason")
	}
	if len(reasons) > 0 {
		return domain.TrainingModelVersionRecord{}, &domain.TrainingGateFailedError{Reasons: reasons}
	}

	now := input.Now
	if now.IsZero() {
		now = domain.Now()
	}
	if toStatus == "active" {
		rows, demoteErr := tx.Query(ctx, `
UPDATE control_training_model_versions
SET status = 'retired', updated_at = $1
WHERE model_key = $2 AND status = 'active' AND version_id <> $3
RETURNING version_id`, now.UTC(), version.ModelKey, versionID)
		if demoteErr != nil {
			return domain.TrainingModelVersionRecord{}, mapPgError(demoteErr)
		}
		demoted := []string{}
		for rows.Next() {
			var id string
			if scanErr := rows.Scan(&id); scanErr != nil {
				rows.Close()
				return domain.TrainingModelVersionRecord{}, mapPgError(scanErr)
			}
			demoted = append(demoted, id)
		}
		rows.Close()
		if rowsErr := rows.Err(); rowsErr != nil {
			return domain.TrainingModelVersionRecord{}, mapPgError(rowsErr)
		}
		for _, id := range demoted {
			if eventErr := appendTrainingVersionEventTx(ctx, tx, id, "retired", input.ActorUserID, "active", "retired", "", now); eventErr != nil {
				return domain.TrainingModelVersionRecord{}, eventErr
			}
		}
		if _, pointerErr := tx.Exec(ctx, `
UPDATE control_training_lineages SET active_version_id = $1, updated_at = $2
WHERE model_key = $3 AND scope = 'shared'`, versionID, now.UTC(), version.ModelKey); pointerErr != nil {
			return domain.TrainingModelVersionRecord{}, mapPgError(pointerErr)
		}
		if _, statusErr := tx.Exec(ctx, `
UPDATE control_training_model_status SET active_model_version = $1 WHERE model_key = $2`,
			versionID, version.ModelKey); statusErr != nil {
			return domain.TrainingModelVersionRecord{}, mapPgError(statusErr)
		}
		if _, flipErr := tx.Exec(ctx, `
UPDATE control_training_model_versions SET status = 'active', activated_at = $1, updated_at = $1
WHERE version_id = $2`, now.UTC(), versionID); flipErr != nil {
			return domain.TrainingModelVersionRecord{}, mapPgError(flipErr)
		}
	} else {
		if _, flipErr := tx.Exec(ctx, `
UPDATE control_training_model_versions SET status = 'canary', updated_at = $1
WHERE version_id = $2`, now.UTC(), versionID); flipErr != nil {
			return domain.TrainingModelVersionRecord{}, mapPgError(flipErr)
		}
	}
	eventType := "promoted_" + toStatus
	if toStatus == "active" && heldOutPending {
		eventType = "promote_with_pending_held_out"
	}
	if eventErr := appendTrainingVersionEventTx(ctx, tx, versionID, eventType, input.ActorUserID, fromStatus, toStatus, strings.TrimSpace(input.OverrideReason), now); eventErr != nil {
		return domain.TrainingModelVersionRecord{}, eventErr
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.TrainingModelVersionRecord{}, mapPgError(err)
	}
	return s.GetTrainingModelVersion(ctx, versionID)
}

func (s *PostgresStore) RollbackTrainingModelVersion(ctx context.Context, input domain.RollbackTrainingModelVersionInput) (domain.TrainingModelVersionRecord, error) {
	activeID := strings.TrimSpace(input.VersionID)
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return domain.TrainingModelVersionRecord{}, mapPgError(err)
	}
	defer func() { _ = tx.Rollback(ctx) }()

	active, err := lockTrainingVersionTx(ctx, tx, activeID)
	if err != nil {
		return domain.TrainingModelVersionRecord{}, err
	}
	if active.Status != "active" {
		return domain.TrainingModelVersionRecord{}, ErrConflict
	}
	targetID := strings.TrimSpace(input.TargetVersionID)
	if targetID == "" {
		row := tx.QueryRow(ctx, `
SELECT version_id FROM control_training_model_versions
WHERE model_key = $1 AND status = 'retired'
ORDER BY updated_at DESC LIMIT 1`, active.ModelKey)
		if scanErr := row.Scan(&targetID); scanErr != nil {
			if errors.Is(scanErr, pgx.ErrNoRows) {
				return domain.TrainingModelVersionRecord{}, ErrNotFound
			}
			return domain.TrainingModelVersionRecord{}, mapPgError(scanErr)
		}
	}
	target, err := lockTrainingVersionTx(ctx, tx, targetID)
	if err != nil {
		return domain.TrainingModelVersionRecord{}, err
	}
	if target.ModelKey != active.ModelKey {
		return domain.TrainingModelVersionRecord{}, ErrNotFound
	}

	now := domain.Now()
	if _, execErr := tx.Exec(ctx, `
UPDATE control_training_model_versions SET status = 'retired', updated_at = $1 WHERE version_id = $2`,
		now.UTC(), activeID); execErr != nil {
		return domain.TrainingModelVersionRecord{}, mapPgError(execErr)
	}
	if _, execErr := tx.Exec(ctx, `
UPDATE control_training_model_versions SET status = 'active', activated_at = $1, updated_at = $1 WHERE version_id = $2`,
		now.UTC(), targetID); execErr != nil {
		return domain.TrainingModelVersionRecord{}, mapPgError(execErr)
	}
	if _, execErr := tx.Exec(ctx, `
UPDATE control_training_lineages SET active_version_id = $1, updated_at = $2
WHERE model_key = $3 AND scope = 'shared'`, targetID, now.UTC(), target.ModelKey); execErr != nil {
		return domain.TrainingModelVersionRecord{}, mapPgError(execErr)
	}
	if _, execErr := tx.Exec(ctx, `
UPDATE control_training_model_status SET active_model_version = $1 WHERE model_key = $2`,
		targetID, target.ModelKey); execErr != nil {
		return domain.TrainingModelVersionRecord{}, mapPgError(execErr)
	}
	if eventErr := appendTrainingVersionEventTx(ctx, tx, activeID, "rolled_back", input.ActorUserID, "active", "retired", strings.TrimSpace(input.Reason), now); eventErr != nil {
		return domain.TrainingModelVersionRecord{}, eventErr
	}
	if eventErr := appendTrainingVersionEventTx(ctx, tx, targetID, "restored_active", input.ActorUserID, "retired", "active", strings.TrimSpace(input.Reason), now); eventErr != nil {
		return domain.TrainingModelVersionRecord{}, eventErr
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.TrainingModelVersionRecord{}, mapPgError(err)
	}
	return s.GetTrainingModelVersion(ctx, targetID)
}

// --- tx helpers ---------------------------------------------------------------

func lockTrainingVersionTx(ctx context.Context, tx pgx.Tx, versionID string) (domain.TrainingModelVersionRecord, error) {
	row := tx.QueryRow(ctx, `
SELECT version_id, lineage_id, model_key, status, is_frozen, COALESCE(weights_uri, ''),
       COALESCE(source_job_id, ''), COALESCE(artifact_run_id, ''), metrics, metadata,
       activated_at, created_at, updated_at
FROM control_training_model_versions
WHERE version_id = $1
FOR UPDATE`, strings.TrimSpace(versionID))
	record, err := scanTrainingModelVersionRow(row)
	if err != nil {
		return domain.TrainingModelVersionRecord{}, mapPgError(err)
	}
	return record, nil
}

func currentFrozenGoldTx(ctx context.Context, tx pgx.Tx, modelKey string) (domain.TrainingGoldSetRecord, bool, error) {
	row := tx.QueryRow(ctx, `
SELECT gold_set_id, model_key, version, COALESCE(content_hash, ''), item_count, label_stats,
       strata_summary, COALESCE(split_manifest_uri, ''), provenance, status, created_at,
       COALESCE(created_by_user_id, ''), frozen_at
FROM control_training_gold_sets
WHERE model_key = $1 AND status = 'frozen'
ORDER BY version DESC LIMIT 1`, strings.TrimSpace(modelKey))
	record, err := scanTrainingGoldSetRow(row)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) || errors.Is(mapPgError(err), ErrNotFound) {
			return domain.TrainingGoldSetRecord{}, false, nil
		}
		return domain.TrainingGoldSetRecord{}, false, mapPgError(err)
	}
	return record, true, nil
}

func latestBenchmarkRunTx(ctx context.Context, tx pgx.Tx, versionID string, goldHash string) (domain.TrainingBenchmarkRunRecord, bool, error) {
	row := tx.QueryRow(ctx, `
SELECT run_id, model_version_id, gold_set_id, gold_set_content_hash, metric_schema,
       kernel_version, metrics, guardrails_passed, guardrails_reasons, report_uri, created_at
FROM control_training_benchmark_runs
WHERE model_version_id = $1 AND gold_set_content_hash = $2
ORDER BY created_at DESC LIMIT 1`, strings.TrimSpace(versionID), strings.TrimSpace(goldHash))
	record, err := scanTrainingBenchmarkRunRow(row)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) || errors.Is(mapPgError(err), ErrNotFound) {
			return domain.TrainingBenchmarkRunRecord{}, false, nil
		}
		return domain.TrainingBenchmarkRunRecord{}, false, mapPgError(err)
	}
	return record, true, nil
}

func appendTrainingVersionEventTx(ctx context.Context, tx pgx.Tx, versionID, eventType, actor, fromStatus, toStatus, reason string, now time.Time) error {
	_, err := tx.Exec(ctx, `
INSERT INTO control_training_model_version_events (
  event_id, version_id, event_type, actor_user_id, from_status, to_status, reason, created_at
) VALUES ($1, $2, $3, $4, NULLIF($5, ''), NULLIF($6, ''), NULLIF($7, ''), $8)`,
		domain.NewID("training_version_event"), versionID, eventType, actor, fromStatus, toStatus, reason, now.UTC())
	if err != nil {
		return mapPgError(err)
	}
	return nil
}
