package httpapi

import (
	"context"
	"errors"
	"net/http"
	"strings"

	"github.com/go-chi/chi/v5"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

// GoldGate training reads (M0). These replace the static/empty stubs: every
// value now comes from control_training_* rows, so the dashboard reports the
// truthful state (real zeros until the M1 sync, a real registry row, a real
// version 0). Write routes stay 501 until their milestones land.

type trainingStore interface {
	ListTrainingModels(ctx context.Context) ([]domain.TrainingModelRecord, error)
	GetTrainingModel(ctx context.Context, modelKey string) (domain.TrainingModelRecord, error)
	GetTrainingModelStatus(ctx context.Context, modelKey string) (domain.TrainingModelStatusRecord, error)
	ListTrainingDomains(ctx context.Context, limit int, offset int) ([]domain.TrainingDomainRecord, error)
	ListTrainingLineages(ctx context.Context, domainID string, limit int, offset int) ([]domain.TrainingLineageRecord, error)
	ListTrainingModelVersions(ctx context.Context, lineageID string, limit int, offset int) ([]domain.TrainingModelVersionRecord, error)
	GetTrainingModelVersion(ctx context.Context, versionID string) (domain.TrainingModelVersionRecord, error)
	ListTrainingRetrainRequests(ctx context.Context, modelKey string, limit int) ([]domain.TrainingRetrainRequestRecord, error)
}

// Both stores must satisfy the capability interface at compile time — a silent
// type-assertion failure would degrade every training read to empty/503 (the
// facade failure mode this milestone exists to kill).
var (
	_ trainingStore = (*store.PostgresStore)(nil)
	_ trainingStore = (*store.MemoryStore)(nil)
)

func (deps ServerDeps) trainingStore() (trainingStore, bool) {
	if deps.Store == nil {
		return nil, false
	}
	training, ok := deps.Store.(trainingStore)
	return training, ok
}

func (deps ServerDeps) handleTrainingModels(w http.ResponseWriter, r *http.Request) {
	training, ok := deps.trainingStore()
	if !ok {
		writeJSON(w, http.StatusOK, trainingModelsResponse{Count: 0, Models: []trainingModelRecord{}})
		return
	}
	models, err := training.ListTrainingModels(r.Context())
	if err != nil {
		writeStoreError(w, err)
		return
	}
	records := make([]trainingModelRecord, 0, len(models))
	for _, model := range models {
		records = append(records, trainingModelRecordFromDomain(model))
	}
	writeJSON(w, http.StatusOK, trainingModelsResponse{Count: len(records), Models: records})
}

func trainingModelRecordFromDomain(model domain.TrainingModelRecord) trainingModelRecord {
	// Exact-match capability semantics, mirroring the Python manifests
	// (CAPABILITY_ALLOWED_PHASES): a malformed registry row should fail
	// visibly, not be forgiven into a capability the worker won't honor.
	capabilities := map[string]bool{}
	for _, capability := range model.Capabilities {
		capabilities[capability] = true
	}
	dimensions := metadataStringSlice(model.Metadata["dimensions"])
	if len(dimensions) == 0 {
		dimensions = []string{"2d"}
	}
	return trainingModelRecord{
		Key:               model.ModelKey,
		Name:              model.DisplayName,
		Framework:         jsonMapString(model.Metadata, "framework"),
		TaskType:          model.TaskType,
		Description:       jsonMapString(model.Metadata, "description"),
		SupportsTraining:  capabilities["FINETUNE"],
		SupportsFinetune:  capabilities["FINETUNE"],
		SupportsInference: true,
		Dimensions:        dimensions,
		DefaultConfig: domain.JSONMap{
			"workflow": jsonMapString(model.Metadata, "workflow"),
			// Honest until the M1 job spine ships the write paths: the model
			// is REGISTERED (capabilities say what it supports by design) but
			// the training pipeline is not configured on this deployment yet.
			"training_state": "not_configured",
			"capabilities":   model.Capabilities,
			"metric_schema":  model.MetricSchema,
			"dataset_format": model.DatasetFormat,
			"requires_phash": model.RequiresPhash,
			"classes":        model.Classes,
		},
	}
}

// handleTrainingModelStatus serves the generic model-scoped status read.
func (deps ServerDeps) handleTrainingModelStatus(w http.ResponseWriter, r *http.Request) {
	deps.writeTrainingModelStatus(w, r, strings.TrimSpace(chi.URLParam(r, "model_key")))
}

// The benchmark readiness fields derive from the ACTIVE version's pinned
// benchmark provenance (metadata.guardrails.benchmarked_at, the plan-pinned
// location the M2 engine stamps) — schema-agnostic, so a segmentation model's
// benchmark lights them up identically. promotion_benchmark_ready is the flat
// metrics key the plan pins for the promote gate. They can never report a
// readiness no benchmark produced.
func (deps ServerDeps) writeTrainingModelStatus(w http.ResponseWriter, r *http.Request, modelKey string) {
	training, ok := deps.trainingStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "training reads are not configured"})
		return
	}
	status, err := training.GetTrainingModelStatus(r.Context(), modelKey)
	if errors.Is(err, store.ErrNotFound) {
		// A registered model without a status row (a freshly onboarded model
		// before its first sync) reads as honest zeros, not a 404 — the M5
		// onboarding walk depends on this. Unknown model keys still 404.
		model, modelErr := training.GetTrainingModel(r.Context(), modelKey)
		if modelErr != nil {
			writeStoreError(w, modelErr)
			return
		}
		status = domain.TrainingModelStatusRecord{
			ModelKey:               model.ModelKey,
			ClassCounts:            domain.JSONMap{},
			PerClassNewObjects:     domain.JSONMap{},
			UnsupportedClassCounts: domain.JSONMap{},
			RetrainGateReasons:     []string{"No data has been synced for this model yet."},
			RetrainGateCounts:      domain.JSONMap{},
			RetrainGateThresholds:  domain.JSONMap{},
		}
	} else if err != nil {
		writeStoreError(w, err)
		return
	}
	activeMetrics := domain.JSONMap{}
	activeGuardrails := domain.JSONMap{}
	var activatedAt any
	if strings.TrimSpace(status.ActiveModelVersion) != "" {
		active, activeErr := training.GetTrainingModelVersion(r.Context(), status.ActiveModelVersion)
		switch {
		case activeErr == nil:
			activeMetrics = active.Metrics
			if guardrails, ok := active.Metadata["guardrails"].(map[string]any); ok {
				activeGuardrails = guardrails
			}
			if active.ActivatedAt != nil {
				activatedAt = active.ActivatedAt
			}
		case errors.Is(activeErr, store.ErrNotFound):
			// Dangling pointer: degrade to unbenchmarked rather than 500, but
			// this is the silent-null class — surface it in the reasons.
			status.RetrainGateReasons = append(status.RetrainGateReasons,
				"active_model_version "+status.ActiveModelVersion+" has no version row - the lineage needs repair.")
		default:
			// A transient store error must not masquerade as "unbenchmarked".
			writeStoreError(w, activeErr)
			return
		}
	}
	lastBenchmarkAt := activeGuardrails["benchmarked_at"]
	canonicalReady := lastBenchmarkAt != nil
	promotionReady := false
	if value, ok := activeMetrics["promotion_benchmark_ready"].(bool); ok {
		promotionReady = value
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"model_key":                   status.ModelKey,
		"dataset_name":                status.DatasetName,
		"dataset_id":                  nullableString(status.DatasetID),
		"last_sync_at":                status.LastSyncAt,
		"next_sync_at":                nil,
		"last_retrain_at":             status.LastRetrainAt,
		"active_model_version":        nullableString(status.ActiveModelVersion),
		"active_version_activated_at": activatedAt,
		"model_health":                status.ModelHealth,
		"reviewed_images":             status.ReviewedImages,
		"unreviewed_images":           status.UnreviewedImages,
		"class_counts":                status.ClassCounts,
		"per_class_new_objects":       status.PerClassNewObjects,
		"unsupported_class_counts":    status.UnsupportedClassCounts,
		"detection_counts":            domain.JSONMap{},
		"latest_metrics":              activeMetrics,
		"benchmark_baseline":          activeMetrics,
		"benchmark_latest_candidate":  domain.JSONMap{},
		"last_benchmark_at":           lastBenchmarkAt,
		"benchmark_ready":             canonicalReady,
		"canonical_benchmark_ready":   canonicalReady,
		"promotion_benchmark_ready":   promotionReady,
		"retrain_gate":                status.RetrainGate,
		"retrain_gate_reasons":        status.RetrainGateReasons,
		"retrain_gate_counts":         status.RetrainGateCounts,
		"retrain_gate_thresholds":     status.RetrainGateThresholds,
	})
}

func (deps ServerDeps) handleTrainingDomains(w http.ResponseWriter, r *http.Request) {
	training, ok := deps.trainingStore()
	if !ok {
		writeJSON(w, http.StatusOK, map[string]any{"count": 0, "domains": []any{}})
		return
	}
	domains, err := training.ListTrainingDomains(r.Context(), clampLimit(parseLimit(r, 200), 1000), parseOffset(r))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"count": len(domains), "domains": domains})
}

func (deps ServerDeps) handleTrainingLineages(w http.ResponseWriter, r *http.Request) {
	training, ok := deps.trainingStore()
	if !ok {
		writeJSON(w, http.StatusOK, map[string]any{"count": 0, "lineages": []any{}})
		return
	}
	domainID := strings.TrimSpace(chi.URLParam(r, "domain_id"))
	lineages, err := training.ListTrainingLineages(r.Context(), domainID, clampLimit(parseLimit(r, 200), 1000), parseOffset(r))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"count": len(lineages), "lineages": lineages})
}

func (deps ServerDeps) handleTrainingVersions(w http.ResponseWriter, r *http.Request) {
	training, ok := deps.trainingStore()
	if !ok {
		writeJSON(w, http.StatusOK, map[string]any{"count": 0, "versions": []any{}})
		return
	}
	lineageID := strings.TrimSpace(chi.URLParam(r, "lineage_id"))
	versions, err := training.ListTrainingModelVersions(r.Context(), lineageID, clampLimit(parseLimit(r, 200), 1000), parseOffset(r))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"count": len(versions), "versions": versions})
}

func (deps ServerDeps) handleTrainingRetrainRequests(w http.ResponseWriter, r *http.Request) {
	training, ok := deps.trainingStore()
	if !ok {
		writeJSON(w, http.StatusOK, map[string]any{"count": 0, "requests": []any{}})
		return
	}
	modelKey := strings.TrimSpace(chi.URLParam(r, "model_key"))
	requests, err := training.ListTrainingRetrainRequests(r.Context(), modelKey, clampLimit(parseLimit(r, 200), 1000))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"count": len(requests), "requests": requests})
}

func nullableString(value string) any {
	if strings.TrimSpace(value) == "" {
		return nil
	}
	return value
}
