package httpapi

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/gateengine"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

// GoldGate training WRITE surface (M1 dispatch + M2 stamping + M4 lifecycle).
// Every route is generic and model-scoped; the capability->phase table gates
// dispatch BEFORE publish (409 unsupported_capability); the promote/graduate
// transactions re-validate server-side - the UI gate is convenience, the
// transaction is the law.

// capabilityAllowedPhases mirrors ultra_deepagents.training.manifests
// CAPABILITY_ALLOWED_PHASES exactly (the cross-language contract): a
// benchmark-only model materializes and freezes gold through the gold_freeze
// job - no hand INSERT.
var capabilityAllowedPhases = map[string][]string{
	"SYNC":      {"training.sync"},
	"ASSEMBLE":  {"training.assemble"},
	"FINETUNE":  {"training.finetune"},
	"BENCHMARK": {"training.benchmark", "training.gold_freeze"},
}

type trainingWriteStore interface {
	trainingStore
	CreateTrainingJob(ctx context.Context, input domain.CreateTrainingJobInput) (domain.TrainingJobRecord, error)
	GetTrainingJob(ctx context.Context, jobID string) (domain.TrainingJobRecord, error)
	UpdateTrainingJobStatus(ctx context.Context, input domain.UpdateTrainingJobStatusInput) (domain.TrainingJobRecord, error)
	AppendTrainingJobEvent(ctx context.Context, input domain.AppendTrainingJobEventInput) (domain.TrainingJobEventRecord, error)
	AcquireTrainingJobLease(ctx context.Context, input domain.AcquireTrainingJobLeaseInput) (domain.TrainingJobLeaseRecord, domain.TrainingJobRecord, error)
	RenewTrainingJobLease(ctx context.Context, input domain.RenewTrainingJobLeaseInput) (domain.TrainingJobLeaseRecord, error)
	ReleaseTrainingJobLease(ctx context.Context, input domain.ReleaseTrainingJobLeaseInput) error
	CreateTrainingGoldSetDraft(ctx context.Context, input domain.CreateTrainingGoldSetInput) (domain.TrainingGoldSetRecord, error)
	GetTrainingGoldSet(ctx context.Context, goldSetID string) (domain.TrainingGoldSetRecord, error)
	GetCurrentTrainingGoldSet(ctx context.Context, modelKey string) (domain.TrainingGoldSetRecord, error)
	UpdateTrainingGoldSetState(ctx context.Context, input domain.UpdateTrainingGoldSetStateInput) (domain.TrainingGoldSetRecord, error)
	InsertTrainingGoldItems(ctx context.Context, goldSetID string, items []domain.TrainingGoldItemInput) (int, error)
	CreateTrainingModelVersion(ctx context.Context, input domain.CreateTrainingModelVersionInput) (domain.TrainingModelVersionRecord, error)
	UpdateTrainingModelVersion(ctx context.Context, input domain.UpdateTrainingModelVersionInput) (domain.TrainingModelVersionRecord, error)
	AppendTrainingModelVersionEvent(ctx context.Context, input domain.AppendTrainingModelVersionEventInput) (domain.TrainingModelVersionEventRecord, error)
	CreateTrainingRetrainRequest(ctx context.Context, input domain.CreateTrainingRetrainRequestInput) (domain.TrainingRetrainRequestRecord, error)
	UpdateTrainingRetrainRequest(ctx context.Context, input domain.UpdateTrainingRetrainRequestInput) (domain.TrainingRetrainRequestRecord, error)
	InsertTrainingBenchmarkRun(ctx context.Context, input domain.InsertTrainingBenchmarkRunInput) (domain.TrainingBenchmarkRunRecord, error)
	GetLatestTrainingBenchmarkRun(ctx context.Context, modelVersionID string, goldSetContentHash string) (domain.TrainingBenchmarkRunRecord, error)
	UpsertTrainingModelStatus(ctx context.Context, record domain.TrainingModelStatusRecord) (domain.TrainingModelStatusRecord, error)
	ListTrainingGuardrailClauses(ctx context.Context, modelKey string) ([]domain.TrainingGuardrailClauseRecord, error)
	GetTrainingGatePolicy(ctx context.Context, modelKey string) (domain.TrainingGatePolicyRecord, error)
	PromoteTrainingModelVersion(ctx context.Context, input domain.PromoteTrainingModelVersionInput) (domain.TrainingModelVersionRecord, error)
	RollbackTrainingModelVersion(ctx context.Context, input domain.RollbackTrainingModelVersionInput) (domain.TrainingModelVersionRecord, error)
}

func (deps ServerDeps) trainingWriteStore() (trainingWriteStore, bool) {
	if deps.Store == nil {
		return nil, false
	}
	training, ok := deps.Store.(trainingWriteStore)
	return training, ok
}

var (
	_ trainingWriteStore = (*store.PostgresStore)(nil)
	_ trainingWriteStore = (*store.MemoryStore)(nil)
)

func modelAllowsPhase(model domain.TrainingModelRecord, phase string) bool {
	for _, capability := range model.Capabilities {
		for _, allowed := range capabilityAllowedPhases[capability] {
			if allowed == phase {
				return true
			}
		}
	}
	return false
}

// dispatchTrainingJob validates the capability->phase table, persists the job
// row, publishes the envelope, and 202s. Unknown model_key -> 404; phase
// outside the manifest's capabilities -> 409 unsupported_capability (BEFORE
// publish); no publisher configured -> 503.
func (deps ServerDeps) dispatchTrainingJob(w http.ResponseWriter, r *http.Request, modelKey string, phase string, params domain.JSONMap) (domain.TrainingJobRecord, bool) {
	training, ok := deps.trainingWriteStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "training writes are not configured"})
		return domain.TrainingJobRecord{}, false
	}
	if !domain.IsTrainingJobPhase(phase) {
		writeError(w, http.StatusBadRequest, errors.New("unknown training job phase "+phase))
		return domain.TrainingJobRecord{}, false
	}
	model, err := training.GetTrainingModel(r.Context(), modelKey)
	if err != nil {
		writeStoreError(w, err)
		return domain.TrainingJobRecord{}, false
	}
	if !modelAllowsPhase(model, phase) {
		writeJSON(w, http.StatusConflict, map[string]any{
			"error":  "unsupported_capability",
			"detail": "model " + model.ModelKey + " does not declare a capability for " + phase,
		})
		return domain.TrainingJobRecord{}, false
	}
	if deps.TrainingJobs == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "training job dispatch is not configured (no NATS publisher)"})
		return domain.TrainingJobRecord{}, false
	}
	principal := deps.principalFromRequest(r, "")
	gpuPool, _ := model.Executor["gpu_pool"].(string)
	job, err := training.CreateTrainingJob(r.Context(), domain.CreateTrainingJobInput{
		ModelKey:        model.ModelKey,
		JobType:         phase,
		GPUPool:         gpuPool,
		Params:          params,
		OwnerUserID:     principal.UserID,
		OwnerOrgID:      principal.OrgID,
		CreatedByUserID: principal.UserID,
	})
	if err != nil {
		writeStoreError(w, err)
		return domain.TrainingJobRecord{}, false
	}
	if err := deps.TrainingJobs.PublishTrainingJob(r.Context(), eventbus.TrainingJob{
		JobID:       job.JobID,
		ModelKey:    job.ModelKey,
		JobType:     job.JobType,
		GPUPool:     job.GPUPool,
		OwnerUserID: job.OwnerUserID,
		OwnerOrgID:  job.OwnerOrgID,
		Params:      job.Params,
	}); err != nil {
		_, _ = training.UpdateTrainingJobStatus(r.Context(), domain.UpdateTrainingJobStatusInput{
			JobID: job.JobID, Status: "failed", Error: "publish failed: " + err.Error(),
		})
		writeError(w, http.StatusServiceUnavailable, err)
		return domain.TrainingJobRecord{}, false
	}
	return job, true
}

func trainingDispatchResponse(w http.ResponseWriter, job domain.TrainingJobRecord) {
	writeJSON(w, http.StatusAccepted, map[string]any{
		"job_id":    job.JobID,
		"job_type":  job.JobType,
		"model_key": job.ModelKey,
		"status":    job.Status,
	})
}

func (deps ServerDeps) handleDispatchTrainingSync(w http.ResponseWriter, r *http.Request) {
	modelKey := strings.TrimSpace(chi.URLParam(r, "model_key"))
	job, ok := deps.dispatchTrainingJob(w, r, modelKey, "training.sync", domain.JSONMap{"purpose": "sync"})
	if !ok {
		return
	}
	trainingDispatchResponse(w, job)
}

func (deps ServerDeps) handleCreateTrainingGoldSetDraft(w http.ResponseWriter, r *http.Request) {
	training, ok := deps.trainingWriteStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "training writes are not configured"})
		return
	}
	modelKey := strings.TrimSpace(chi.URLParam(r, "model_key"))
	if _, err := training.GetTrainingModel(r.Context(), modelKey); err != nil {
		writeStoreError(w, err)
		return
	}
	principal := deps.principalFromRequest(r, "")
	draft, err := training.CreateTrainingGoldSetDraft(r.Context(), domain.CreateTrainingGoldSetInput{
		ModelKey:        modelKey,
		CreatedByUserID: principal.UserID,
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, draft)
}

// handleFreezeTrainingGoldSet: the async worker-executed freeze (section 4.6).
// Go validates (draft status, BENCHMARK capability, operator confirm), flips
// to 'freezing', and dispatches training.gold_freeze; the worker runs the
// Python leakage hooks fail-closed and reports back; Go finalizes.
func (deps ServerDeps) handleFreezeTrainingGoldSet(w http.ResponseWriter, r *http.Request) {
	training, ok := deps.trainingWriteStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "training writes are not configured"})
		return
	}
	modelKey := strings.TrimSpace(chi.URLParam(r, "model_key"))
	goldSetID := strings.TrimSpace(chi.URLParam(r, "gold_set_id"))
	goldSet, err := training.GetTrainingGoldSet(r.Context(), goldSetID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if goldSet.ModelKey != modelKey {
		writeError(w, http.StatusBadRequest, errors.New("gold set does not belong to this model"))
		return
	}
	if goldSet.Status != "draft" && goldSet.Status != "failed" {
		writeJSON(w, http.StatusConflict, map[string]any{"error": "gold set is not freezable from status " + goldSet.Status})
		return
	}
	if _, err := training.UpdateTrainingGoldSetState(r.Context(), domain.UpdateTrainingGoldSetStateInput{
		GoldSetID: goldSetID, Status: "freezing",
	}); err != nil {
		writeStoreError(w, err)
		return
	}
	job, ok := deps.dispatchTrainingJob(w, r, modelKey, "training.gold_freeze", domain.JSONMap{
		"gold_set_id": goldSetID,
	})
	if !ok {
		// Roll the state back so the operator can retry.
		_, _ = training.UpdateTrainingGoldSetState(r.Context(), domain.UpdateTrainingGoldSetStateInput{
			GoldSetID: goldSetID, Status: "failed",
		})
		return
	}
	trainingDispatchResponse(w, job)
}

func (deps ServerDeps) handleDispatchTrainingBenchmark(w http.ResponseWriter, r *http.Request) {
	training, ok := deps.trainingWriteStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "training writes are not configured"})
		return
	}
	modelKey := strings.TrimSpace(chi.URLParam(r, "model_key"))
	var request struct {
		VersionID string `json:"version_id"`
	}
	if r.Body != nil && r.ContentLength != 0 {
		if !decodeJSON(w, r, &request) {
			return
		}
	}
	gold, err := training.GetCurrentTrainingGoldSet(r.Context(), modelKey)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			writeJSON(w, http.StatusConflict, map[string]any{"error": "no frozen gold set - freeze gold before benchmarking"})
			return
		}
		writeStoreError(w, err)
		return
	}
	// Default target: the active version (baseline pinning, section 7.5).
	versionID := strings.TrimSpace(request.VersionID)
	if versionID == "" {
		status, statusErr := training.GetTrainingModelStatus(r.Context(), modelKey)
		if statusErr != nil {
			writeStoreError(w, statusErr)
			return
		}
		versionID = strings.TrimSpace(status.ActiveModelVersion)
		if versionID == "" {
			writeJSON(w, http.StatusConflict, map[string]any{"error": "no active version to baseline"})
			return
		}
	}
	job, ok := deps.dispatchTrainingJob(w, r, modelKey, "training.benchmark", domain.JSONMap{
		"version_id":            versionID,
		"gold_set_id":           gold.GoldSetID,
		"gold_set_content_hash": gold.ContentHash,
		"split_manifest_uri":    gold.SplitManifestURI,
	})
	if !ok {
		return
	}
	trainingDispatchResponse(w, job)
}

// handleDispatchTrainingRetrain validates Gate A server-side (the trigger
// gate, section 7.1) and dispatches training.assemble; the worker chains to
// training.finetune and the benchmark auto-runs on completion (M3).
func (deps ServerDeps) handleDispatchTrainingRetrain(w http.ResponseWriter, r *http.Request) {
	training, ok := deps.trainingWriteStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "training writes are not configured"})
		return
	}
	modelKey := strings.TrimSpace(chi.URLParam(r, "model_key"))
	status, err := training.GetTrainingModelStatus(r.Context(), modelKey)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if !status.RetrainGate {
		writeJSON(w, http.StatusConflict, map[string]any{
			"error":   "gate_blocked",
			"reasons": status.RetrainGateReasons,
		})
		return
	}
	var request struct {
		Note string `json:"note"`
	}
	if r.Body != nil && r.ContentLength != 0 {
		if !decodeJSON(w, r, &request) {
			return
		}
	}
	job, ok := deps.dispatchTrainingJob(w, r, modelKey, "training.assemble", domain.JSONMap{
		"purpose": "finetune_mix",
		"note":    strings.TrimSpace(request.Note),
	})
	if !ok {
		return
	}
	principal := deps.principalFromRequest(r, "")
	if _, err := training.CreateTrainingRetrainRequest(r.Context(), domain.CreateTrainingRetrainRequestInput{
		ModelKey:          modelKey,
		TrainingJobID:     job.JobID,
		Note:              strings.TrimSpace(request.Note),
		RequestedByUserID: principal.UserID,
		GatingSummary:     domain.JSONMap{"counts": status.RetrainGateCounts, "thresholds": status.RetrainGateThresholds},
	}); err != nil {
		writeStoreError(w, err)
		return
	}
	trainingDispatchResponse(w, job)
}

// handleRegisterTrainingModelVersion: external checkpoint registration
// (section 3.6) - the route whose absence made benchmark-only onboarding
// depend on hand INSERTs. The version starts life guilty: candidate,
// unbenchmarked. Provenance is REQUIRED.
func (deps ServerDeps) handleRegisterTrainingModelVersion(w http.ResponseWriter, r *http.Request) {
	training, ok := deps.trainingWriteStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "training writes are not configured"})
		return
	}
	modelKey := strings.TrimSpace(chi.URLParam(r, "model_key"))
	var request struct {
		VersionID  string         `json:"version_id"`
		WeightsURI string         `json:"weights_uri"`
		Provenance domain.JSONMap `json:"provenance"`
		Metadata   domain.JSONMap `json:"metadata"`
	}
	if !decodeJSON(w, r, &request) {
		return
	}
	if strings.TrimSpace(request.WeightsURI) == "" {
		writeError(w, http.StatusBadRequest, errors.New("weights_uri is required"))
		return
	}
	if len(request.Provenance) == 0 {
		writeError(w, http.StatusBadRequest, errors.New("provenance is required for externally-trained checkpoints"))
		return
	}
	if _, err := training.GetTrainingModel(r.Context(), modelKey); err != nil {
		writeStoreError(w, err)
		return
	}
	lineage := trainingLineageForModel(r.Context(), training, modelKey)
	if lineage == "" {
		writeJSON(w, http.StatusConflict, map[string]any{"error": "no lineage exists for model " + modelKey})
		return
	}
	metadata := domain.JSONMap{}
	for key, value := range request.Metadata {
		metadata[key] = value
	}
	metadata["provenance"] = request.Provenance
	version, err := training.CreateTrainingModelVersion(r.Context(), domain.CreateTrainingModelVersionInput{
		VersionID:  strings.TrimSpace(request.VersionID),
		LineageID:  lineage,
		ModelKey:   modelKey,
		Status:     "candidate",
		WeightsURI: strings.TrimSpace(request.WeightsURI),
		Metadata:   metadata,
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	principal := deps.principalFromRequest(r, "")
	_, _ = training.AppendTrainingModelVersionEvent(r.Context(), domain.AppendTrainingModelVersionEventInput{
		VersionID: version.VersionID, EventType: "registered_external",
		ActorUserID: principal.UserID, ToStatus: "candidate",
	})
	writeJSON(w, http.StatusCreated, version)
}

func trainingLineageForModel(ctx context.Context, training trainingWriteStore, modelKey string) string {
	domains, err := training.ListTrainingDomains(ctx, 500, 0)
	if err != nil {
		return ""
	}
	for _, trainingDomain := range domains {
		lineages, err := training.ListTrainingLineages(ctx, trainingDomain.DomainID, 500, 0)
		if err != nil {
			continue
		}
		for _, lineage := range lineages {
			if lineage.ModelKey == modelKey && lineage.Scope == "shared" {
				return lineage.LineageID
			}
		}
	}
	return ""
}

func (deps ServerDeps) handleRejectTrainingModelVersion(w http.ResponseWriter, r *http.Request) {
	training, ok := deps.trainingWriteStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "training writes are not configured"})
		return
	}
	versionID := strings.TrimSpace(chi.URLParam(r, "version_id"))
	var request struct {
		Reason string `json:"reason"`
	}
	if r.Body != nil && r.ContentLength != 0 {
		if !decodeJSON(w, r, &request) {
			return
		}
	}
	version, err := training.GetTrainingModelVersion(r.Context(), versionID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if version.Status != "candidate" && version.Status != "canary" {
		writeJSON(w, http.StatusConflict, map[string]any{"error": "only candidate or canary versions can be dismissed (status " + version.Status + ")"})
		return
	}
	updated, err := training.UpdateTrainingModelVersion(r.Context(), domain.UpdateTrainingModelVersionInput{
		VersionID: versionID, Status: "rejected",
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	principal := deps.principalFromRequest(r, "")
	_, _ = training.AppendTrainingModelVersionEvent(r.Context(), domain.AppendTrainingModelVersionEventInput{
		VersionID: versionID, EventType: "rejected", ActorUserID: principal.UserID,
		FromStatus: version.Status, ToStatus: "rejected", Reason: strings.TrimSpace(request.Reason),
	})
	writeJSON(w, http.StatusOK, updated)
}

// handlePromoteTrainingModelVersionReal replaces the 501 stub: candidate ->
// canary and canary -> active, each re-validated INSIDE the store transaction
// (section 8.1/8.2). A client bypassing the disabled button still cannot
// promote a regressed weight.
func (deps ServerDeps) handlePromoteTrainingModelVersionReal(w http.ResponseWriter, r *http.Request) {
	training, ok := deps.trainingWriteStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "training writes are not configured"})
		return
	}
	versionID := strings.TrimSpace(chi.URLParam(r, "version_id"))
	var request struct {
		Note           string `json:"note"`
		OverrideReason string `json:"override_reason"`
	}
	if r.Body != nil && r.ContentLength != 0 {
		if !decodeJSON(w, r, &request) {
			return
		}
	}
	principal := deps.principalFromRequest(r, "")
	version, err := training.PromoteTrainingModelVersion(r.Context(), domain.PromoteTrainingModelVersionInput{
		VersionID:      versionID,
		ActorUserID:    principal.UserID,
		OverrideReason: strings.TrimSpace(request.OverrideReason),
		Now:            time.Now().UTC(),
		RequiredMetricKeysCheck: func(metrics domain.JSONMap) error {
			return trainingRequiredMetricKeysCheck(metrics)
		},
	})
	if err != nil {
		var gateErr *domain.TrainingGateFailedError
		if errors.As(err, &gateErr) {
			writeJSON(w, http.StatusConflict, map[string]any{"error": "gate_failed", "reasons": gateErr.Reasons})
			return
		}
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, version)
}

// trainingRequiredMetricKeysCheck is the cheap Go half of the two-layer
// enforcement (section 7.4): the veto never trusts a hand-inserted or
// corrupted metrics blob even if guardrails_passed says true.
func trainingRequiredMetricKeysCheck(metrics domain.JSONMap) error {
	if metrics == nil {
		return errors.New("metrics blob is empty")
	}
	perSlice, ok := metrics["per_slice"].(map[string]any)
	if !ok || len(perSlice) == 0 {
		return errors.New("metrics blob missing per_slice")
	}
	perClass, ok := metrics["per_class"].(map[string]any)
	if !ok || len(perClass) == 0 {
		return errors.New("metrics blob missing per_class")
	}
	for name, raw := range perClass {
		entry, ok := raw.(map[string]any)
		if !ok {
			return errors.New("metrics per_class entry malformed for " + name)
		}
		if _, ok := entry["predicted_count"]; !ok {
			return errors.New("metrics per_class." + name + " missing predicted_count")
		}
		if _, ok := entry["label_count"]; !ok {
			return errors.New("metrics per_class." + name + " missing label_count")
		}
	}
	return nil
}

func (deps ServerDeps) handleRollbackTrainingModelVersionReal(w http.ResponseWriter, r *http.Request) {
	training, ok := deps.trainingWriteStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "training writes are not configured"})
		return
	}
	versionID := strings.TrimSpace(chi.URLParam(r, "version_id"))
	var request struct {
		TargetVersionID string `json:"target_version_id"`
		Note            string `json:"note"`
	}
	if r.Body != nil && r.ContentLength != 0 {
		if !decodeJSON(w, r, &request) {
			return
		}
	}
	principal := deps.principalFromRequest(r, "")
	version, err := training.RollbackTrainingModelVersion(r.Context(), domain.RollbackTrainingModelVersionInput{
		VersionID:       versionID,
		TargetVersionID: strings.TrimSpace(request.TargetVersionID),
		ActorUserID:     principal.UserID,
		Reason:          strings.TrimSpace(request.Note),
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, version)
}

// --- Worker endpoints (the Python training worker's HTTP contract) ----------

func (deps ServerDeps) handleAcquireTrainingJobLease(w http.ResponseWriter, r *http.Request) {
	training, ok := deps.trainingWriteStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "training writes are not configured"})
		return
	}
	jobID := strings.TrimSpace(chi.URLParam(r, "job_id"))
	var request struct {
		WorkerID   string `json:"worker_id"`
		TTLSeconds int    `json:"ttl_seconds"`
	}
	if !decodeJSON(w, r, &request) {
		return
	}
	lease, job, err := training.AcquireTrainingJobLease(r.Context(), domain.AcquireTrainingJobLeaseInput{
		JobID: jobID, WorkerID: strings.TrimSpace(request.WorkerID), TTL: trainingLeaseTTL(request.TTLSeconds),
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	// Flat lease fields at the top level - the worker's contract - with the
	// job nested for context.
	writeJSON(w, http.StatusOK, map[string]any{
		"job_id":           lease.JobID,
		"worker_id":        lease.WorkerID,
		"lease_token":      lease.LeaseToken,
		"lease_expires_at": lease.LeaseExpiresAt,
		"job":              job,
	})
}

func (deps ServerDeps) handleRenewTrainingJobLease(w http.ResponseWriter, r *http.Request) {
	training, ok := deps.trainingWriteStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "training writes are not configured"})
		return
	}
	jobID := strings.TrimSpace(chi.URLParam(r, "job_id"))
	var request struct {
		LeaseToken string `json:"lease_token"`
		TTLSeconds int    `json:"ttl_seconds"`
	}
	if !decodeJSON(w, r, &request) {
		return
	}
	lease, err := training.RenewTrainingJobLease(r.Context(), domain.RenewTrainingJobLeaseInput{
		JobID: jobID, LeaseToken: strings.TrimSpace(request.LeaseToken), TTL: trainingLeaseTTL(request.TTLSeconds),
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, lease)
}

func (deps ServerDeps) handleReleaseTrainingJobLease(w http.ResponseWriter, r *http.Request) {
	training, ok := deps.trainingWriteStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "training writes are not configured"})
		return
	}
	jobID := strings.TrimSpace(chi.URLParam(r, "job_id"))
	var request struct {
		LeaseToken string `json:"lease_token"`
	}
	if !decodeJSON(w, r, &request) {
		return
	}
	if err := training.ReleaseTrainingJobLease(r.Context(), domain.ReleaseTrainingJobLeaseInput{
		JobID: jobID, LeaseToken: strings.TrimSpace(request.LeaseToken),
	}); err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "released"})
}

func trainingLeaseTTL(seconds int) time.Duration {
	if seconds <= 0 {
		// Training runs are HOURS long; the default sits far above the
		// data-agent 300/600s defaults (section 9.2) and the OS-thread
		// keepalive renews at TTL/3 regardless.
		return 30 * time.Minute
	}
	return time.Duration(seconds) * time.Second
}

func (deps ServerDeps) handleAppendTrainingJobEventHTTP(w http.ResponseWriter, r *http.Request) {
	training, ok := deps.trainingWriteStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "training writes are not configured"})
		return
	}
	jobID := strings.TrimSpace(chi.URLParam(r, "job_id"))
	var request struct {
		EventType string         `json:"event_type"`
		Message   string         `json:"message"`
		Metadata  domain.JSONMap `json:"metadata"`
	}
	if !decodeJSON(w, r, &request) {
		return
	}
	event, err := training.AppendTrainingJobEvent(r.Context(), domain.AppendTrainingJobEventInput{
		JobID: jobID, EventType: strings.TrimSpace(request.EventType),
		Message: request.Message, Metadata: request.Metadata, TS: time.Now().UTC(),
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, event)
}

func (deps ServerDeps) handleUpdateTrainingJobStatusHTTP(w http.ResponseWriter, r *http.Request) {
	training, ok := deps.trainingWriteStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "training writes are not configured"})
		return
	}
	jobID := strings.TrimSpace(chi.URLParam(r, "job_id"))
	var request struct {
		Status        string         `json:"status"`
		Error         string         `json:"error"`
		OutputSummary domain.JSONMap `json:"output_summary"`
	}
	if !decodeJSON(w, r, &request) {
		return
	}
	job, err := training.UpdateTrainingJobStatus(r.Context(), domain.UpdateTrainingJobStatusInput{
		JobID: jobID, Status: strings.TrimSpace(request.Status),
		Error: request.Error, OutputSummary: request.OutputSummary,
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, job)
}

// handleTrainingBenchmarkResult is where the M2 engine meets the wire: the
// worker posts raw metrics; the CONTROL PLANE evaluates Gate B against the
// pinned baseline and stamps guardrails + promotion_benchmark_ready onto the
// version - the worker never self-grades.
func (deps ServerDeps) handleTrainingBenchmarkResult(w http.ResponseWriter, r *http.Request) {
	training, ok := deps.trainingWriteStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "training writes are not configured"})
		return
	}
	jobID := strings.TrimSpace(chi.URLParam(r, "job_id"))
	job, err := training.GetTrainingJob(r.Context(), jobID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	var request struct {
		ModelVersionID     string         `json:"model_version_id"`
		GoldSetID          string         `json:"gold_set_id"`
		GoldSetContentHash string         `json:"gold_set_content_hash"`
		MetricSchema       string         `json:"metric_schema"`
		KernelVersion      string         `json:"kernel_version"`
		Metrics            domain.JSONMap `json:"metrics"`
		ReportURI          string         `json:"report_uri"`
	}
	if !decodeJSON(w, r, &request) {
		return
	}
	result, evalErr := deps.evaluateAndStampBenchmark(r.Context(), training, job.ModelKey, request.ModelVersionID,
		request.GoldSetID, request.GoldSetContentHash, request.MetricSchema, request.KernelVersion,
		request.Metrics, request.ReportURI)
	if evalErr != nil {
		writeStoreError(w, evalErr)
		return
	}
	writeJSON(w, http.StatusOK, result)
}

func (deps ServerDeps) evaluateAndStampBenchmark(
	ctx context.Context,
	training trainingWriteStore,
	modelKey string,
	versionID string,
	goldSetID string,
	goldHash string,
	metricSchema string,
	kernelVersion string,
	metrics domain.JSONMap,
	reportURI string,
) (map[string]any, error) {
	model, err := training.GetTrainingModel(ctx, modelKey)
	if err != nil {
		return nil, err
	}
	clauseRecords, err := training.ListTrainingGuardrailClauses(ctx, modelKey)
	if err != nil {
		return nil, err
	}
	clauses := make([]gateengine.Clause, 0, len(clauseRecords))
	for _, record := range clauseRecords {
		clauses = append(clauses, gateengine.Clause{
			ClauseKey: record.ClauseKey, MetricPath: record.MetricPath, Comparator: record.Comparator,
			Value: record.Value, Slice: record.Slice, Params: record.Params,
			Enabled: record.Enabled, Required: record.Required,
		})
	}
	classNames := make([]string, 0, len(model.Classes))
	for _, name := range model.Classes {
		if text, ok := name.(string); ok {
			classNames = append(classNames, text)
		}
	}
	candidate := gateengine.Run{Metrics: metrics, MetricSchema: metricSchema, KernelVersion: kernelVersion, GoldHash: goldHash}

	// The pinned baseline: the ACTIVE version's latest run on this hash. A
	// baseline benchmarking ITSELF compares against itself (must pass).
	status, err := training.GetTrainingModelStatus(ctx, modelKey)
	if err != nil {
		return nil, err
	}
	baselineVersionID := strings.TrimSpace(status.ActiveModelVersion)
	baseline := candidate
	isBaselineRun := versionID == baselineVersionID
	if !isBaselineRun && baselineVersionID != "" {
		baselineRun, baselineErr := training.GetLatestTrainingBenchmarkRun(ctx, baselineVersionID, goldHash)
		if baselineErr != nil {
			if errors.Is(baselineErr, store.ErrNotFound) {
				// Fail closed toward re-baselining (section 7.5).
				baseline = gateengine.Run{MetricSchema: "missing", KernelVersion: "missing", GoldHash: goldHash}
			} else {
				return nil, baselineErr
			}
		} else {
			baseline = gateengine.Run{
				Metrics: baselineRun.Metrics, MetricSchema: baselineRun.MetricSchema,
				KernelVersion: baselineRun.KernelVersion, GoldHash: baselineRun.GoldSetContentHash,
			}
		}
	}

	verdict := gateengine.Evaluate(clauses, candidate, baseline, classNames)
	run, err := training.InsertTrainingBenchmarkRun(ctx, domain.InsertTrainingBenchmarkRunInput{
		ModelVersionID: versionID, GoldSetID: goldSetID, GoldSetContentHash: goldHash,
		MetricSchema: metricSchema, KernelVersion: kernelVersion, Metrics: metrics,
		GuardrailsPassed: verdict.Passed, GuardrailsReasons: verdict.Reasons, ReportURI: reportURI,
	})
	if err != nil {
		return nil, err
	}

	// Stamp the version: metrics summary + the pinned guardrails wire shape.
	version, err := training.GetTrainingModelVersion(ctx, versionID)
	if err != nil {
		return nil, err
	}
	gold, goldErr := training.GetTrainingGoldSet(ctx, goldSetID)
	goldVersion := int64(0)
	if goldErr == nil {
		goldVersion = gold.Version
	}
	stampedMetrics := domain.JSONMap{}
	for key, value := range metrics {
		stampedMetrics[key] = value
	}
	if aggregate, ok := metrics["aggregate"].(map[string]any); ok {
		if map50, ok := aggregate["map50"]; ok {
			stampedMetrics["map50"] = map50
		}
	}
	stampedMetrics["promotion_benchmark_ready"] = verdict.Passed
	metadata := domain.JSONMap{}
	for key, value := range version.Metadata {
		metadata[key] = value
	}
	metadata["guardrails"] = map[string]any{
		"passed":                verdict.Passed,
		"reasons":               verdict.Reasons,
		"clauses":               verdict.Clauses,
		"benchmarked_at":        time.Now().UTC().Format(time.RFC3339),
		"gold_set_version":      goldVersion,
		"gold_set_content_hash": goldHash,
		"report_uri":            reportURI,
	}
	if _, err := training.UpdateTrainingModelVersion(ctx, domain.UpdateTrainingModelVersionInput{
		VersionID: versionID, Metrics: stampedMetrics, Metadata: metadata,
	}); err != nil {
		return nil, err
	}
	return map[string]any{
		"run_id":            run.RunID,
		"guardrails_passed": verdict.Passed,
		"reasons":           verdict.Reasons,
		"baseline_run":      isBaselineRun,
	}, nil
}

// handleTrainingGoldResult finalizes the async freeze (section 4.6 step 3):
// frozen on success (with the sealed manifest facts), failed with recorded
// reasons on any violation. The job can only fail the draft, never expand it.
func (deps ServerDeps) handleTrainingGoldResult(w http.ResponseWriter, r *http.Request) {
	training, ok := deps.trainingWriteStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "training writes are not configured"})
		return
	}
	jobID := strings.TrimSpace(chi.URLParam(r, "job_id"))
	if _, err := training.GetTrainingJob(r.Context(), jobID); err != nil {
		writeStoreError(w, err)
		return
	}
	var request struct {
		GoldSetID        string         `json:"gold_set_id"`
		Status           string         `json:"status"` // frozen | failed
		ContentHash      string         `json:"content_hash"`
		LabelStats       domain.JSONMap `json:"label_stats"`
		StrataSummary    domain.JSONMap `json:"strata_summary"`
		Provenance       domain.JSONMap `json:"provenance"`
		SplitManifestURI string         `json:"split_manifest_uri"`
		// The worker reports structured violations: [{check, item_id, reason, defense?}].
		FailureReasons []domain.JSONMap `json:"failure_reasons"`
		Items          []goldItemWire   `json:"items"`
	}
	if !decodeJSON(w, r, &request) {
		return
	}
	if request.Status != "frozen" && request.Status != "failed" {
		writeError(w, http.StatusBadRequest, errors.New("gold result status must be frozen or failed"))
		return
	}
	itemCount := 0
	if len(request.Items) > 0 {
		inputs := make([]domain.TrainingGoldItemInput, 0, len(request.Items))
		for _, item := range request.Items {
			inputs = append(inputs, item.toInput())
		}
		inserted, err := training.InsertTrainingGoldItems(r.Context(), request.GoldSetID, inputs)
		if err != nil {
			writeStoreError(w, err)
			return
		}
		itemCount = inserted
	}
	provenance := request.Provenance
	if request.Status == "failed" {
		if provenance == nil {
			provenance = domain.JSONMap{}
		}
		provenance["freeze_failure_reasons"] = request.FailureReasons
	}
	now := time.Now().UTC()
	var frozenAt *time.Time
	if request.Status == "frozen" {
		frozenAt = &now
	}
	goldSet, err := training.UpdateTrainingGoldSetState(r.Context(), domain.UpdateTrainingGoldSetStateInput{
		GoldSetID: request.GoldSetID, Status: request.Status, ContentHash: request.ContentHash,
		ItemCount: int64(itemCount), LabelStats: request.LabelStats, StrataSummary: request.StrataSummary,
		Provenance: provenance, SplitManifestURI: request.SplitManifestURI, FrozenAt: frozenAt,
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, goldSet)
}

// handleTrainingStatusResult: the sync worker's status write + server-side
// Gate A evaluation (the trigger gate is computed HERE, from the policy row +
// the fixed active-passes-gold precondition, never by the worker).
func (deps ServerDeps) handleTrainingStatusResult(w http.ResponseWriter, r *http.Request) {
	training, ok := deps.trainingWriteStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "training writes are not configured"})
		return
	}
	jobID := strings.TrimSpace(chi.URLParam(r, "job_id"))
	job, err := training.GetTrainingJob(r.Context(), jobID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	// The worker's flat contract: status fields at the top level plus the
	// Gate A counts under retrain_gate_counts.
	var request struct {
		DatasetName            string         `json:"dataset_name"`
		DatasetID              string         `json:"dataset_id"`
		ModelHealth            string         `json:"model_health"`
		ReviewedImages         int64          `json:"reviewed_images"`
		UnreviewedImages       int64          `json:"unreviewed_images"`
		ClassCounts            domain.JSONMap `json:"class_counts"`
		PerClassNewObjects     domain.JSONMap `json:"per_class_new_objects"`
		UnsupportedClassCounts domain.JSONMap `json:"unsupported_class_counts"`
		RetrainGateCounts      struct {
			ReviewedImages int64            `json:"reviewed_images"`
			TotalObjects   int64            `json:"total_objects"`
			PerClass       map[string]int64 `json:"per_class"`
		} `json:"retrain_gate_counts"`
	}
	if !decodeJSON(w, r, &request) {
		return
	}
	previous, _ := training.GetTrainingModelStatus(r.Context(), job.ModelKey)
	record := previous
	record.ModelKey = job.ModelKey
	record.DatasetName = request.DatasetName
	if strings.TrimSpace(request.DatasetID) != "" {
		record.DatasetID = request.DatasetID
	}
	if strings.TrimSpace(request.ModelHealth) != "" {
		record.ModelHealth = request.ModelHealth
	}
	record.ReviewedImages = request.ReviewedImages
	record.UnreviewedImages = request.UnreviewedImages
	record.ClassCounts = request.ClassCounts
	record.PerClassNewObjects = request.PerClassNewObjects
	record.UnsupportedClassCounts = request.UnsupportedClassCounts
	now := time.Now().UTC()
	record.LastSyncAt = &now
	record.RetrainGateCounts = domain.JSONMap{
		"reviewed_images": request.RetrainGateCounts.ReviewedImages,
		"total_objects":   request.RetrainGateCounts.TotalObjects,
		"per_class":       request.RetrainGateCounts.PerClass,
	}
	daysSinceLastTrain := float64(-1)
	if previous.LastRetrainAt != nil {
		daysSinceLastTrain = now.Sub(*previous.LastRetrainAt).Hours() / 24
	}

	policyRecord, policyErr := training.GetTrainingGatePolicy(r.Context(), job.ModelKey)
	if policyErr == nil {
		var activePassesGold *bool
		if gold, goldErr := training.GetCurrentTrainingGoldSet(r.Context(), job.ModelKey); goldErr == nil {
			passes := false
			if record.ActiveModelVersion != "" {
				if run, runErr := training.GetLatestTrainingBenchmarkRun(r.Context(), record.ActiveModelVersion, gold.ContentHash); runErr == nil {
					passes = run.GuardrailsPassed
				}
			}
			activePassesGold = &passes
		}
		perClassPolicy := map[string]int64{}
		for name, raw := range policyRecord.MinPerClassObjects {
			if value, ok := raw.(float64); ok {
				perClassPolicy[name] = int64(value)
			}
		}
		gateA := gateengine.EvaluateGateA(gateengine.Policy{
			MinReviewed: policyRecord.MinReviewed, MinNewObjects: policyRecord.MinNewObjects,
			MinPerClassObjects: perClassPolicy, MinDays: policyRecord.MinDays,
		}, gateengine.GateACounts{
			ReviewedImagesSinceActive: request.RetrainGateCounts.ReviewedImages,
			NewTotalObjects:           request.RetrainGateCounts.TotalObjects,
			NewPerClassObjects:        request.RetrainGateCounts.PerClass,
			DaysSinceLastTrain:        daysSinceLastTrain,
			ActivePassesGold:          activePassesGold,
		})
		record.RetrainGate = gateA.Ready
		record.RetrainGateReasons = gateA.Reasons
		record.RetrainGateThresholds = domain.JSONMap{
			"min_reviewed": policyRecord.MinReviewed, "min_new_objects": policyRecord.MinNewObjects,
			"min_per_class_objects": policyRecord.MinPerClassObjects, "min_days": policyRecord.MinDays,
		}
	}
	updated, err := training.UpsertTrainingModelStatus(r.Context(), record)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, updated)
}

// goldItemWire is the training worker's gold-item wire shape; label_uri,
// exif_present, and per-item label_stats ride into the stored input.
type goldItemWire struct {
	ItemID        string         `json:"item_id"`
	SourceRef     domain.JSONMap `json:"source_ref"`
	Slice         string         `json:"slice"`
	LabelKind     string         `json:"label_kind"`
	ContentSHA256 string         `json:"content_sha256"`
	Phash         *string        `json:"phash"`
	GTLabelSHA256 string         `json:"gt_label_sha256"`
	LabelURI      string         `json:"label_uri"`
	GTLabelURI    string         `json:"gt_label_uri"`
	Width         *int64         `json:"width"`
	Height        *int64         `json:"height"`
	ExifPresent   *bool          `json:"exif_present"`
	LabelStats    domain.JSONMap `json:"label_stats"`
	Metadata      domain.JSONMap `json:"metadata"`
	FootprintGeom domain.JSONMap `json:"footprint_geom"`
	StrataTags    domain.JSONMap `json:"strata_tags"`
}

func (w goldItemWire) toInput() domain.TrainingGoldItemInput {
	labelURI := w.GTLabelURI
	if strings.TrimSpace(labelURI) == "" {
		labelURI = w.LabelURI
	}
	metadata := domain.JSONMap{}
	for key, value := range w.Metadata {
		metadata[key] = value
	}
	if w.ExifPresent != nil {
		metadata["exif_present"] = *w.ExifPresent
	}
	if len(w.LabelStats) > 0 {
		metadata["label_stats"] = w.LabelStats
	}
	return domain.TrainingGoldItemInput{
		ItemID:        w.ItemID,
		SourceRef:     w.SourceRef,
		Slice:         w.Slice,
		LabelKind:     w.LabelKind,
		ContentSHA256: w.ContentSHA256,
		Phash:         w.Phash,
		GTLabelSHA256: w.GTLabelSHA256,
		GTLabelURI:    labelURI,
		Width:         w.Width,
		Height:        w.Height,
		Metadata:      metadata,
		FootprintGeom: w.FootprintGeom,
		StrataTags:    w.StrataTags,
	}
}
