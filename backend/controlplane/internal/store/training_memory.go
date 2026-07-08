package store

import (
	"context"
	"errors"
	"sort"
	"strings"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

// In-memory mirror of the GoldGate training reads, seeded with the same rows
// as the schema.sql seed so dev-without-Postgres and handler tests see the
// truthful M0 state (registry + version 0 + lineage + status).

type memoryTrainingState struct {
	models           map[string]domain.TrainingModelRecord
	domains          map[string]domain.TrainingDomainRecord
	lineages         map[string]domain.TrainingLineageRecord
	versions         map[string]domain.TrainingModelVersionRecord
	statuses         map[string]domain.TrainingModelStatusRecord
	retrainRequests  map[string][]domain.TrainingRetrainRequestRecord
	jobs             map[string]domain.TrainingJobRecord
	jobEvents        map[string][]domain.TrainingJobEventRecord
	jobLeases        map[string]domain.TrainingJobLeaseRecord
	goldSets         map[string]domain.TrainingGoldSetRecord
	goldItems        map[string][]domain.TrainingGoldItemRecord
	versionEvents    map[string][]domain.TrainingModelVersionEventRecord
	benchmarkRuns    map[string][]domain.TrainingBenchmarkRunRecord
	gatePolicies     map[string]domain.TrainingGatePolicyRecord
	guardrailClauses map[string][]domain.TrainingGuardrailClauseRecord
}

func newMemoryTrainingState() *memoryTrainingState {
	now := domain.Now()
	model := trainingSeedModel(now)
	trainingDomain := trainingSeedDomain(now)
	lineage := trainingSeedLineage(now)
	version := trainingSeedVersion(now)
	status := trainingSeedStatus()
	gatePolicy := trainingSeedGatePolicy()
	return &memoryTrainingState{
		models:           map[string]domain.TrainingModelRecord{model.ModelKey: model},
		domains:          map[string]domain.TrainingDomainRecord{trainingDomain.DomainID: trainingDomain},
		lineages:         map[string]domain.TrainingLineageRecord{lineage.LineageID: lineage},
		versions:         map[string]domain.TrainingModelVersionRecord{version.VersionID: version},
		statuses:         map[string]domain.TrainingModelStatusRecord{status.ModelKey: status},
		retrainRequests:  map[string][]domain.TrainingRetrainRequestRecord{},
		jobs:             map[string]domain.TrainingJobRecord{},
		jobEvents:        map[string][]domain.TrainingJobEventRecord{},
		jobLeases:        map[string]domain.TrainingJobLeaseRecord{},
		goldSets:         map[string]domain.TrainingGoldSetRecord{},
		goldItems:        map[string][]domain.TrainingGoldItemRecord{},
		versionEvents:    map[string][]domain.TrainingModelVersionEventRecord{},
		benchmarkRuns:    map[string][]domain.TrainingBenchmarkRunRecord{},
		gatePolicies:     map[string]domain.TrainingGatePolicyRecord{gatePolicy.ModelKey: gatePolicy},
		guardrailClauses: map[string][]domain.TrainingGuardrailClauseRecord{model.ModelKey: trainingSeedGuardrailClauses()},
	}
}

func (s *MemoryStore) ListTrainingModels(ctx context.Context) ([]domain.TrainingModelRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	records := make([]domain.TrainingModelRecord, 0, len(s.training.models))
	for _, record := range s.training.models {
		records = append(records, cloneTrainingModel(record))
	}
	sort.Slice(records, func(i, j int) bool { return records[i].ModelKey < records[j].ModelKey })
	return records, nil
}

func (s *MemoryStore) GetTrainingModel(ctx context.Context, modelKey string) (domain.TrainingModelRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	record, ok := s.training.models[strings.TrimSpace(modelKey)]
	if !ok {
		return domain.TrainingModelRecord{}, ErrNotFound
	}
	return cloneTrainingModel(record), nil
}

func (s *MemoryStore) GetTrainingModelStatus(ctx context.Context, modelKey string) (domain.TrainingModelStatusRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	record, ok := s.training.statuses[strings.TrimSpace(modelKey)]
	if !ok {
		return domain.TrainingModelStatusRecord{}, ErrNotFound
	}
	return cloneTrainingStatus(record), nil
}

func (s *MemoryStore) ListTrainingDomains(ctx context.Context, limit int, offset int) ([]domain.TrainingDomainRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	records := make([]domain.TrainingDomainRecord, 0, len(s.training.domains))
	for _, record := range s.training.domains {
		record.Metadata = cloneJSONMap(record.Metadata)
		records = append(records, record)
	}
	sort.Slice(records, func(i, j int) bool { return records[i].DomainID < records[j].DomainID })
	return paginateTraining(records, limit, offset), nil
}

func (s *MemoryStore) ListTrainingLineages(ctx context.Context, domainID string, limit int, offset int) ([]domain.TrainingLineageRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	records := []domain.TrainingLineageRecord{}
	for _, record := range s.training.lineages {
		if record.DomainID == strings.TrimSpace(domainID) {
			record.Metadata = cloneJSONMap(record.Metadata)
			records = append(records, record)
		}
	}
	sort.Slice(records, func(i, j int) bool { return records[i].LineageID < records[j].LineageID })
	return paginateTraining(records, limit, offset), nil
}

func (s *MemoryStore) ListTrainingModelVersions(ctx context.Context, lineageID string, limit int, offset int) ([]domain.TrainingModelVersionRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	records := []domain.TrainingModelVersionRecord{}
	for _, record := range s.training.versions {
		if record.LineageID == strings.TrimSpace(lineageID) {
			records = append(records, cloneTrainingVersion(record))
		}
	}
	sort.Slice(records, func(i, j int) bool {
		if records[i].CreatedAt.Equal(records[j].CreatedAt) {
			return records[i].VersionID > records[j].VersionID
		}
		return records[i].CreatedAt.After(records[j].CreatedAt)
	})
	return paginateTraining(records, limit, offset), nil
}

func (s *MemoryStore) GetTrainingModelVersion(ctx context.Context, versionID string) (domain.TrainingModelVersionRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	record, ok := s.training.versions[strings.TrimSpace(versionID)]
	if !ok {
		return domain.TrainingModelVersionRecord{}, ErrNotFound
	}
	return cloneTrainingVersion(record), nil
}

func (s *MemoryStore) ListTrainingRetrainRequests(ctx context.Context, modelKey string, limit int) ([]domain.TrainingRetrainRequestRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	requests := s.training.retrainRequests[strings.TrimSpace(modelKey)]
	records := make([]domain.TrainingRetrainRequestRecord, 0, len(requests))
	for _, request := range requests {
		request.GatingSummary = cloneJSONMap(request.GatingSummary)
		records = append(records, request)
	}
	sort.Slice(records, func(i, j int) bool { return records[i].CreatedAt.After(records[j].CreatedAt) })
	return take(records, limit), nil
}

func paginateTraining[T any](records []T, limit int, offset int) []T {
	if offset < 0 {
		offset = 0
	}
	if offset >= len(records) {
		return []T{}
	}
	return take(records[offset:], limit)
}

// Records leave the store as copies: struct copy + cloned maps/slices, the
// same defensive boundary every other MemoryStore record type maintains via
// cloneJSONMap — a caller mutation must never write through to the seed.
func cloneTrainingModel(record domain.TrainingModelRecord) domain.TrainingModelRecord {
	record.Capabilities = append([]string(nil), record.Capabilities...)
	record.LeakageDefensesExtra = append([]string(nil), record.LeakageDefensesExtra...)
	record.Executor = cloneJSONMap(record.Executor)
	record.Classes = cloneJSONMap(record.Classes)
	record.Metadata = cloneJSONMap(record.Metadata)
	return record
}

func cloneTrainingVersion(record domain.TrainingModelVersionRecord) domain.TrainingModelVersionRecord {
	record.Metrics = cloneJSONMap(record.Metrics)
	record.Metadata = cloneJSONMap(record.Metadata)
	record.ActivatedAt = utcTimePtr(record.ActivatedAt)
	return record
}

func cloneTrainingStatus(record domain.TrainingModelStatusRecord) domain.TrainingModelStatusRecord {
	record.ClassCounts = cloneJSONMap(record.ClassCounts)
	record.PerClassNewObjects = cloneJSONMap(record.PerClassNewObjects)
	record.UnsupportedClassCounts = cloneJSONMap(record.UnsupportedClassCounts)
	record.RetrainGateCounts = cloneJSONMap(record.RetrainGateCounts)
	record.RetrainGateThresholds = cloneJSONMap(record.RetrainGateThresholds)
	record.RetrainGateReasons = append([]string(nil), record.RetrainGateReasons...)
	record.LastSyncAt = utcTimePtr(record.LastSyncAt)
	record.LastRetrainAt = utcTimePtr(record.LastRetrainAt)
	return record
}

// ---------------------------------------------------------------------------
// GoldGate training writes (M1) — in-memory mirror of the Postgres methods.
// Jobs/leases/events clone the data-agent job family semantics.
// ---------------------------------------------------------------------------

func (s *MemoryStore) CreateTrainingJob(ctx context.Context, input domain.CreateTrainingJobInput) (domain.TrainingJobRecord, error) {
	_ = ctx
	modelKey := strings.TrimSpace(input.ModelKey)
	if modelKey == "" {
		return domain.TrainingJobRecord{}, errors.New("training job model key is required")
	}
	jobType := strings.ToLower(strings.TrimSpace(input.JobType))
	if jobType == "" {
		return domain.TrainingJobRecord{}, errors.New("training job type is required")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.training.models[modelKey]; !ok {
		return domain.TrainingJobRecord{}, ErrNotFound
	}
	jobID := strings.TrimSpace(input.JobID)
	if jobID == "" {
		jobID = domain.NewID("training_job")
	}
	if _, exists := s.training.jobs[jobID]; exists {
		return domain.TrainingJobRecord{}, ErrConflict
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
	job := domain.TrainingJobRecord{
		JobID:           jobID,
		ModelKey:        modelKey,
		JobType:         jobType,
		Status:          "queued",
		GPUPool:         strings.TrimSpace(input.GPUPool),
		Params:          cloneJSONMap(input.Params),
		OwnerUserID:     ownerUserID,
		OwnerOrgID:      strings.TrimSpace(input.OwnerOrgID),
		CreatedByUserID: createdByUserID,
		CreatedAt:       now,
		UpdatedAt:       now,
		OutputSummary:   domain.JSONMap{},
		Metadata:        cloneJSONMap(input.Metadata),
	}
	s.training.jobs[job.JobID] = job
	if _, err := s.appendTrainingJobEventLocked(domain.AppendTrainingJobEventInput{
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
	return cloneTrainingJob(job), nil
}

func (s *MemoryStore) GetTrainingJob(ctx context.Context, jobID string) (domain.TrainingJobRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	job, ok := s.training.jobs[strings.TrimSpace(jobID)]
	if !ok {
		return domain.TrainingJobRecord{}, ErrNotFound
	}
	return cloneTrainingJob(job), nil
}

func (s *MemoryStore) ListTrainingJobs(ctx context.Context, modelKey string, status string, limit int) ([]domain.TrainingJobRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	modelKey = strings.TrimSpace(modelKey)
	status = strings.ToLower(strings.TrimSpace(status))
	records := []domain.TrainingJobRecord{}
	for _, job := range s.training.jobs {
		if modelKey != "" && job.ModelKey != modelKey {
			continue
		}
		if status != "" && job.Status != status {
			continue
		}
		records = append(records, cloneTrainingJob(job))
	}
	sort.Slice(records, func(i, j int) bool {
		if records[i].CreatedAt.Equal(records[j].CreatedAt) {
			return records[i].JobID < records[j].JobID
		}
		return records[i].CreatedAt.After(records[j].CreatedAt)
	})
	return take(records, limit), nil
}

func (s *MemoryStore) UpdateTrainingJobStatus(ctx context.Context, input domain.UpdateTrainingJobStatusInput) (domain.TrainingJobRecord, error) {
	_ = ctx
	status := strings.ToLower(strings.TrimSpace(input.Status))
	if !validTrainingJobStatus(status) {
		return domain.TrainingJobRecord{}, errors.New("invalid training job status")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	job, ok := s.training.jobs[strings.TrimSpace(input.JobID)]
	if !ok {
		return domain.TrainingJobRecord{}, ErrNotFound
	}
	now := domain.Now()
	job.Status = status
	job.ProgressCompleted = clampTrainingJobProgress(input.ProgressCompleted, input.ProgressTotal)
	if input.ProgressTotal > 0 {
		job.ProgressTotal = input.ProgressTotal
	}
	if job.ProgressTotal < job.ProgressCompleted {
		job.ProgressTotal = job.ProgressCompleted
	}
	job.Error = strings.TrimSpace(input.Error)
	job.UpdatedAt = now
	if status == "running" && job.StartedAt.IsZero() {
		job.StartedAt = now
	}
	if trainingJobStatusIsTerminal(status) {
		job.CompletedAt = now
	} else {
		job.CompletedAt = time.Time{}
	}
	if input.OutputSummary != nil {
		job.OutputSummary = cloneJSONMap(input.OutputSummary)
	}
	s.training.jobs[job.JobID] = job
	return cloneTrainingJob(job), nil
}

func (s *MemoryStore) AppendTrainingJobEvent(ctx context.Context, input domain.AppendTrainingJobEventInput) (domain.TrainingJobEventRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.appendTrainingJobEventLocked(input)
}

func (s *MemoryStore) ListTrainingJobEvents(ctx context.Context, jobID string, limit int) ([]domain.TrainingJobEventRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	events := s.training.jobEvents[strings.TrimSpace(jobID)]
	records := make([]domain.TrainingJobEventRecord, 0, len(events))
	for _, event := range events {
		records = append(records, cloneTrainingJobEvent(event))
	}
	sort.Slice(records, func(i, j int) bool {
		if records[i].Sequence == records[j].Sequence {
			return records[i].EventID < records[j].EventID
		}
		return records[i].Sequence < records[j].Sequence
	})
	return take(records, limit), nil
}

func (s *MemoryStore) AcquireTrainingJobLease(ctx context.Context, input domain.AcquireTrainingJobLeaseInput) (domain.TrainingJobLeaseRecord, domain.TrainingJobRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	job, ok := s.training.jobs[strings.TrimSpace(input.JobID)]
	if !ok {
		return domain.TrainingJobLeaseRecord{}, domain.TrainingJobRecord{}, ErrNotFound
	}
	if trainingJobStatusIsTerminal(job.Status) {
		return domain.TrainingJobLeaseRecord{}, domain.TrainingJobRecord{}, ErrConflict
	}
	now := leaseNow(input.Now)
	if existing, ok := s.training.jobLeases[job.JobID]; ok && existing.LeaseExpiresAt.After(now) {
		return domain.TrainingJobLeaseRecord{}, domain.TrainingJobRecord{}, ErrConflict
	}
	lease := domain.TrainingJobLeaseRecord{
		JobID:          job.JobID,
		WorkerID:       strings.TrimSpace(input.WorkerID),
		LeaseToken:     domain.NewID("training_lease"),
		LeaseExpiresAt: now.Add(positiveLeaseTTL(input.TTL)),
		CreatedAt:      now,
		UpdatedAt:      now,
	}
	s.training.jobLeases[job.JobID] = lease
	job.Status = "running"
	job.UpdatedAt = now
	if job.StartedAt.IsZero() {
		job.StartedAt = now
	}
	s.training.jobs[job.JobID] = job
	if _, err := s.appendTrainingJobEventLocked(domain.AppendTrainingJobEventInput{
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
	return lease, cloneTrainingJob(job), nil
}

func (s *MemoryStore) RenewTrainingJobLease(ctx context.Context, input domain.RenewTrainingJobLeaseInput) (domain.TrainingJobLeaseRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	job, ok := s.training.jobs[strings.TrimSpace(input.JobID)]
	if !ok {
		return domain.TrainingJobLeaseRecord{}, ErrNotFound
	}
	if trainingJobStatusIsTerminal(job.Status) {
		return domain.TrainingJobLeaseRecord{}, ErrConflict
	}
	lease, ok := s.training.jobLeases[job.JobID]
	if !ok {
		return domain.TrainingJobLeaseRecord{}, ErrNotFound
	}
	// Token match alone authorizes renewal — even of an EXPIRED lease (the
	// run-lease revival rule): a trainer that survived a control-plane outage
	// must be able to revive its lease and keep its GPU-hours computation. If
	// the lease row was removed (release/recovery), the lookup above misses
	// and the worker gets the authoritative not-found instead.
	if lease.LeaseToken != strings.TrimSpace(input.LeaseToken) {
		return domain.TrainingJobLeaseRecord{}, ErrConflict
	}
	now := leaseNow(input.Now)
	lease.LeaseExpiresAt = now.Add(positiveLeaseTTL(input.TTL))
	lease.UpdatedAt = now
	s.training.jobLeases[job.JobID] = lease
	job.UpdatedAt = now
	s.training.jobs[job.JobID] = job
	return lease, nil
}

func (s *MemoryStore) ReleaseTrainingJobLease(ctx context.Context, input domain.ReleaseTrainingJobLeaseInput) error {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	job, ok := s.training.jobs[strings.TrimSpace(input.JobID)]
	if !ok {
		return ErrNotFound
	}
	lease, ok := s.training.jobLeases[job.JobID]
	if !ok {
		return nil
	}
	if lease.LeaseToken != strings.TrimSpace(input.LeaseToken) {
		return ErrConflict
	}
	delete(s.training.jobLeases, job.JobID)
	return nil
}

func (s *MemoryStore) appendTrainingJobEventLocked(input domain.AppendTrainingJobEventInput) (domain.TrainingJobEventRecord, error) {
	jobID := strings.TrimSpace(input.JobID)
	if _, ok := s.training.jobs[jobID]; !ok {
		return domain.TrainingJobEventRecord{}, ErrNotFound
	}
	eventID := strings.TrimSpace(input.EventID)
	if eventID == "" {
		eventID = domain.NewID("training_job_event")
	}
	for _, existing := range s.training.jobEvents[jobID] {
		if strings.TrimSpace(existing.EventID) == eventID {
			return domain.TrainingJobEventRecord{}, ErrConflict
		}
	}
	sequence := input.Sequence
	if sequence <= 0 {
		sequence = int64(len(s.training.jobEvents[jobID]) + 1)
	}
	ts := input.TS
	if ts.IsZero() {
		ts = domain.Now()
	}
	record := domain.TrainingJobEventRecord{
		EventID:     eventID,
		JobID:       jobID,
		Sequence:    sequence,
		EventType:   strings.TrimSpace(input.EventType),
		ActorUserID: strings.TrimSpace(input.ActorUserID),
		ActorOrgID:  strings.TrimSpace(input.ActorOrgID),
		TS:          ts.UTC(),
		Message:     strings.TrimSpace(input.Message),
		Metadata:    cloneJSONMap(input.Metadata),
	}
	s.training.jobEvents[jobID] = append(s.training.jobEvents[jobID], record)
	return cloneTrainingJobEvent(record), nil
}

func (s *MemoryStore) CreateTrainingGoldSetDraft(ctx context.Context, input domain.CreateTrainingGoldSetInput) (domain.TrainingGoldSetRecord, error) {
	_ = ctx
	modelKey := strings.TrimSpace(input.ModelKey)
	if modelKey == "" {
		return domain.TrainingGoldSetRecord{}, errors.New("gold set model key is required")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.training.models[modelKey]; !ok {
		return domain.TrainingGoldSetRecord{}, ErrNotFound
	}
	goldSetID := strings.TrimSpace(input.GoldSetID)
	if goldSetID == "" {
		goldSetID = domain.NewID("gold_set")
	}
	if _, exists := s.training.goldSets[goldSetID]; exists {
		return domain.TrainingGoldSetRecord{}, ErrConflict
	}
	var version int64
	for _, set := range s.training.goldSets {
		if set.ModelKey != modelKey {
			continue
		}
		if set.Status == "draft" || set.Status == "freezing" {
			return domain.TrainingGoldSetRecord{}, ErrConflict
		}
		if set.Version > version {
			version = set.Version
		}
	}
	createdByUserID := strings.TrimSpace(input.CreatedByUserID)
	if createdByUserID == "" {
		createdByUserID = "local-user"
	}
	record := domain.TrainingGoldSetRecord{
		GoldSetID:       goldSetID,
		ModelKey:        modelKey,
		Version:         version + 1,
		LabelStats:      domain.JSONMap{},
		StrataSummary:   domain.JSONMap{},
		Provenance:      domain.JSONMap{},
		Status:          "draft",
		CreatedAt:       domain.Now(),
		CreatedByUserID: createdByUserID,
	}
	s.training.goldSets[record.GoldSetID] = record
	return cloneTrainingGoldSet(record), nil
}

func (s *MemoryStore) GetTrainingGoldSet(ctx context.Context, goldSetID string) (domain.TrainingGoldSetRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	record, ok := s.training.goldSets[strings.TrimSpace(goldSetID)]
	if !ok {
		return domain.TrainingGoldSetRecord{}, ErrNotFound
	}
	return cloneTrainingGoldSet(record), nil
}

func (s *MemoryStore) GetCurrentTrainingGoldSet(ctx context.Context, modelKey string) (domain.TrainingGoldSetRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	modelKey = strings.TrimSpace(modelKey)
	var current domain.TrainingGoldSetRecord
	found := false
	for _, set := range s.training.goldSets {
		if set.ModelKey != modelKey || set.Status != "frozen" {
			continue
		}
		if !found || set.Version > current.Version {
			current = set
			found = true
		}
	}
	if !found {
		return domain.TrainingGoldSetRecord{}, ErrNotFound
	}
	return cloneTrainingGoldSet(current), nil
}

func (s *MemoryStore) ListTrainingGoldSets(ctx context.Context, modelKey string, limit int) ([]domain.TrainingGoldSetRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	modelKey = strings.TrimSpace(modelKey)
	records := []domain.TrainingGoldSetRecord{}
	for _, set := range s.training.goldSets {
		if set.ModelKey != modelKey {
			continue
		}
		records = append(records, cloneTrainingGoldSet(set))
	}
	sort.Slice(records, func(i, j int) bool { return records[i].Version > records[j].Version })
	return take(records, limit), nil
}

func (s *MemoryStore) UpdateTrainingGoldSetState(ctx context.Context, input domain.UpdateTrainingGoldSetStateInput) (domain.TrainingGoldSetRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	record, ok := s.training.goldSets[strings.TrimSpace(input.GoldSetID)]
	if !ok {
		return domain.TrainingGoldSetRecord{}, ErrNotFound
	}
	status := strings.ToLower(strings.TrimSpace(input.Status))
	if status != "" && status != record.Status {
		if !trainingGoldSetTransitionAllowed(record.Status, status) {
			return domain.TrainingGoldSetRecord{}, ErrConflict
		}
		record.Status = status
	}
	if contentHash := strings.TrimSpace(input.ContentHash); contentHash != "" {
		for _, other := range s.training.goldSets {
			if other.GoldSetID != record.GoldSetID && other.ContentHash == contentHash {
				return domain.TrainingGoldSetRecord{}, ErrConflict
			}
		}
		record.ContentHash = contentHash
	}
	if input.ItemCount > 0 {
		record.ItemCount = input.ItemCount
	}
	if input.LabelStats != nil {
		record.LabelStats = cloneJSONMap(input.LabelStats)
	}
	if input.StrataSummary != nil {
		record.StrataSummary = cloneJSONMap(input.StrataSummary)
	}
	if input.Provenance != nil {
		record.Provenance = cloneJSONMap(input.Provenance)
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
	s.training.goldSets[record.GoldSetID] = record
	return cloneTrainingGoldSet(record), nil
}

func (s *MemoryStore) InsertTrainingGoldItems(ctx context.Context, goldSetID string, items []domain.TrainingGoldItemInput) (int, error) {
	_ = ctx
	goldSetID = strings.TrimSpace(goldSetID)
	s.mu.Lock()
	defer s.mu.Unlock()
	set, ok := s.training.goldSets[goldSetID]
	if !ok {
		return 0, ErrNotFound
	}
	if !trainingGoldSetAcceptsItems(set.Status) {
		return 0, ErrConflict
	}
	seen := map[string]struct{}{}
	for _, existing := range s.training.goldItems[goldSetID] {
		seen[existing.ItemID] = struct{}{}
	}
	records := make([]domain.TrainingGoldItemRecord, 0, len(items))
	for _, item := range items {
		itemID := strings.TrimSpace(item.ItemID)
		if itemID == "" {
			return 0, errors.New("gold item id is required")
		}
		if _, dup := seen[itemID]; dup {
			return 0, ErrConflict
		}
		seen[itemID] = struct{}{}
		records = append(records, cloneTrainingGoldItem(domain.TrainingGoldItemRecord{
			GoldSetID:     goldSetID,
			ItemID:        itemID,
			SourceRef:     item.SourceRef,
			Slice:         strings.TrimSpace(item.Slice),
			LabelKind:     strings.TrimSpace(item.LabelKind),
			ContentSHA256: strings.TrimSpace(item.ContentSHA256),
			Phash:         item.Phash,
			GTLabelSHA256: strings.TrimSpace(item.GTLabelSHA256),
			GTLabelURI:    strings.TrimSpace(item.GTLabelURI),
			Width:         item.Width,
			Height:        item.Height,
			Metadata:      item.Metadata,
			FootprintGeom: item.FootprintGeom,
			StrataTags:    item.StrataTags,
		}))
	}
	s.training.goldItems[goldSetID] = append(s.training.goldItems[goldSetID], records...)
	return len(records), nil
}

func (s *MemoryStore) ListTrainingGoldItems(ctx context.Context, goldSetID string, limit int, offset int) ([]domain.TrainingGoldItemRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	items := s.training.goldItems[strings.TrimSpace(goldSetID)]
	records := make([]domain.TrainingGoldItemRecord, 0, len(items))
	for _, item := range items {
		records = append(records, cloneTrainingGoldItem(item))
	}
	sort.Slice(records, func(i, j int) bool { return records[i].ItemID < records[j].ItemID })
	return paginateTraining(records, limit, offset), nil
}

func (s *MemoryStore) CreateTrainingModelVersion(ctx context.Context, input domain.CreateTrainingModelVersionInput) (domain.TrainingModelVersionRecord, error) {
	_ = ctx
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
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.training.lineages[lineageID]; !ok {
		return domain.TrainingModelVersionRecord{}, ErrNotFound
	}
	if _, ok := s.training.models[modelKey]; !ok {
		return domain.TrainingModelVersionRecord{}, ErrNotFound
	}
	versionID := strings.TrimSpace(input.VersionID)
	if versionID == "" {
		versionID = domain.NewID("version")
	}
	if _, exists := s.training.versions[versionID]; exists {
		return domain.TrainingModelVersionRecord{}, ErrConflict
	}
	now := domain.Now()
	record := domain.TrainingModelVersionRecord{
		VersionID:   versionID,
		LineageID:   lineageID,
		ModelKey:    modelKey,
		Status:      status,
		WeightsURI:  strings.TrimSpace(input.WeightsURI),
		SourceJobID: strings.TrimSpace(input.SourceJobID),
		Metrics:     cloneJSONMap(input.Metrics),
		Metadata:    cloneJSONMap(input.Metadata),
		CreatedAt:   now,
		UpdatedAt:   now,
	}
	s.training.versions[record.VersionID] = record
	return cloneTrainingVersion(record), nil
}

func (s *MemoryStore) UpdateTrainingModelVersion(ctx context.Context, input domain.UpdateTrainingModelVersionInput) (domain.TrainingModelVersionRecord, error) {
	_ = ctx
	status := strings.ToLower(strings.TrimSpace(input.Status))
	if status != "" && !validTrainingModelVersionStatus(status) {
		return domain.TrainingModelVersionRecord{}, errors.New("invalid training model version status")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	record, ok := s.training.versions[strings.TrimSpace(input.VersionID)]
	if !ok {
		return domain.TrainingModelVersionRecord{}, ErrNotFound
	}
	if status != "" {
		record.Status = status
	}
	if input.Metrics != nil {
		record.Metrics = cloneJSONMap(input.Metrics)
	}
	if input.Metadata != nil {
		record.Metadata = cloneJSONMap(input.Metadata)
	}
	if input.ActivatedAt != nil {
		record.ActivatedAt = utcTimePtr(input.ActivatedAt)
	}
	record.UpdatedAt = domain.Now()
	s.training.versions[record.VersionID] = record
	return cloneTrainingVersion(record), nil
}

func (s *MemoryStore) AppendTrainingModelVersionEvent(ctx context.Context, input domain.AppendTrainingModelVersionEventInput) (domain.TrainingModelVersionEventRecord, error) {
	_ = ctx
	versionID := strings.TrimSpace(input.VersionID)
	if versionID == "" {
		return domain.TrainingModelVersionEventRecord{}, errors.New("version event version id is required")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	eventID := strings.TrimSpace(input.EventID)
	if eventID == "" {
		eventID = domain.NewID("version_event")
	}
	for _, events := range s.training.versionEvents {
		for _, existing := range events {
			if existing.EventID == eventID {
				return domain.TrainingModelVersionEventRecord{}, ErrConflict
			}
		}
	}
	record := domain.TrainingModelVersionEventRecord{
		EventID:            eventID,
		VersionID:          versionID,
		EventType:          strings.TrimSpace(input.EventType),
		ActorUserID:        strings.TrimSpace(input.ActorUserID),
		FromStatus:         strings.TrimSpace(input.FromStatus),
		ToStatus:           strings.TrimSpace(input.ToStatus),
		BenchmarkRunID:     strings.TrimSpace(input.BenchmarkRunID),
		GoldSetContentHash: strings.TrimSpace(input.GoldSetContentHash),
		Reason:             strings.TrimSpace(input.Reason),
		CreatedAt:          domain.Now(),
	}
	s.training.versionEvents[versionID] = append(s.training.versionEvents[versionID], record)
	return record, nil
}

func (s *MemoryStore) ListTrainingModelVersionEvents(ctx context.Context, versionID string, limit int) ([]domain.TrainingModelVersionEventRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	events := s.training.versionEvents[strings.TrimSpace(versionID)]
	// Reverse insertion order first so the stable sort keeps newest-first
	// within identical timestamps.
	records := make([]domain.TrainingModelVersionEventRecord, 0, len(events))
	for i := len(events) - 1; i >= 0; i-- {
		records = append(records, events[i])
	}
	sort.SliceStable(records, func(i, j int) bool { return records[i].CreatedAt.After(records[j].CreatedAt) })
	return take(records, limit), nil
}

func (s *MemoryStore) CreateTrainingRetrainRequest(ctx context.Context, input domain.CreateTrainingRetrainRequestInput) (domain.TrainingRetrainRequestRecord, error) {
	_ = ctx
	modelKey := strings.TrimSpace(input.ModelKey)
	if modelKey == "" {
		return domain.TrainingRetrainRequestRecord{}, errors.New("retrain request model key is required")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.training.models[modelKey]; !ok {
		return domain.TrainingRetrainRequestRecord{}, ErrNotFound
	}
	requestID := strings.TrimSpace(input.RequestID)
	if requestID == "" {
		requestID = domain.NewID("retrain_request")
	}
	for _, requests := range s.training.retrainRequests {
		for _, existing := range requests {
			if existing.RequestID == requestID {
				return domain.TrainingRetrainRequestRecord{}, ErrConflict
			}
		}
	}
	record := domain.TrainingRetrainRequestRecord{
		RequestID:         requestID,
		ModelKey:          modelKey,
		TrainingJobID:     strings.TrimSpace(input.TrainingJobID),
		Status:            "queued",
		Note:              strings.TrimSpace(input.Note),
		GatingSummary:     cloneJSONMap(input.GatingSummary),
		RequestedByUserID: strings.TrimSpace(input.RequestedByUserID),
		CreatedAt:         domain.Now(),
	}
	s.training.retrainRequests[modelKey] = append(s.training.retrainRequests[modelKey], record)
	return cloneTrainingRetrainRequest(record), nil
}

func (s *MemoryStore) UpdateTrainingRetrainRequest(ctx context.Context, input domain.UpdateTrainingRetrainRequestInput) (domain.TrainingRetrainRequestRecord, error) {
	_ = ctx
	requestID := strings.TrimSpace(input.RequestID)
	s.mu.Lock()
	defer s.mu.Unlock()
	for modelKey, requests := range s.training.retrainRequests {
		for i, record := range requests {
			if record.RequestID != requestID {
				continue
			}
			if status := strings.ToLower(strings.TrimSpace(input.Status)); status != "" {
				record.Status = status
			}
			if errText := strings.TrimSpace(input.Error); errText != "" {
				record.Error = errText
			}
			if modelVersion := strings.TrimSpace(input.ModelVersion); modelVersion != "" {
				record.ModelVersion = modelVersion
			}
			if trainingJobID := strings.TrimSpace(input.TrainingJobID); trainingJobID != "" {
				record.TrainingJobID = trainingJobID
			}
			if input.StartedAt != nil {
				record.StartedAt = utcTimePtr(input.StartedAt)
			}
			if input.FinishedAt != nil {
				record.FinishedAt = utcTimePtr(input.FinishedAt)
			}
			s.training.retrainRequests[modelKey][i] = record
			return cloneTrainingRetrainRequest(record), nil
		}
	}
	return domain.TrainingRetrainRequestRecord{}, ErrNotFound
}

func (s *MemoryStore) InsertTrainingBenchmarkRun(ctx context.Context, input domain.InsertTrainingBenchmarkRunInput) (domain.TrainingBenchmarkRunRecord, error) {
	_ = ctx
	modelVersionID := strings.TrimSpace(input.ModelVersionID)
	if modelVersionID == "" {
		return domain.TrainingBenchmarkRunRecord{}, errors.New("benchmark run model version id is required")
	}
	goldSetID := strings.TrimSpace(input.GoldSetID)
	if goldSetID == "" {
		return domain.TrainingBenchmarkRunRecord{}, errors.New("benchmark run gold set id is required")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	runID := strings.TrimSpace(input.RunID)
	if runID == "" {
		runID = domain.NewID("benchmark_run")
	}
	for _, runs := range s.training.benchmarkRuns {
		for _, existing := range runs {
			if existing.RunID == runID {
				return domain.TrainingBenchmarkRunRecord{}, ErrConflict
			}
		}
	}
	record := domain.TrainingBenchmarkRunRecord{
		RunID:              runID,
		ModelVersionID:     modelVersionID,
		GoldSetID:          goldSetID,
		GoldSetContentHash: strings.TrimSpace(input.GoldSetContentHash),
		MetricSchema:       strings.TrimSpace(input.MetricSchema),
		KernelVersion:      strings.TrimSpace(input.KernelVersion),
		Metrics:            cloneJSONMap(input.Metrics),
		GuardrailsPassed:   input.GuardrailsPassed,
		GuardrailsReasons:  append([]string{}, input.GuardrailsReasons...),
		ReportURI:          strings.TrimSpace(input.ReportURI),
		CreatedAt:          domain.Now(),
	}
	s.training.benchmarkRuns[modelVersionID] = append(s.training.benchmarkRuns[modelVersionID], record)
	return cloneTrainingBenchmarkRun(record), nil
}

func (s *MemoryStore) GetLatestTrainingBenchmarkRun(ctx context.Context, modelVersionID string, goldSetContentHash string) (domain.TrainingBenchmarkRunRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	goldSetContentHash = strings.TrimSpace(goldSetContentHash)
	var latest domain.TrainingBenchmarkRunRecord
	found := false
	for _, run := range s.training.benchmarkRuns[strings.TrimSpace(modelVersionID)] {
		if goldSetContentHash != "" && run.GoldSetContentHash != goldSetContentHash {
			continue
		}
		// >= so later insertions win created_at ties.
		if !found || !run.CreatedAt.Before(latest.CreatedAt) {
			latest = run
			found = true
		}
	}
	if !found {
		return domain.TrainingBenchmarkRunRecord{}, ErrNotFound
	}
	return cloneTrainingBenchmarkRun(latest), nil
}

func (s *MemoryStore) UpsertTrainingModelStatus(ctx context.Context, record domain.TrainingModelStatusRecord) (domain.TrainingModelStatusRecord, error) {
	_ = ctx
	modelKey := strings.TrimSpace(record.ModelKey)
	if modelKey == "" {
		return domain.TrainingModelStatusRecord{}, errors.New("model status model key is required")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.training.models[modelKey]; !ok {
		return domain.TrainingModelStatusRecord{}, ErrNotFound
	}
	record.ModelKey = modelKey
	stored := cloneTrainingStatus(record)
	s.training.statuses[modelKey] = stored
	return cloneTrainingStatus(stored), nil
}

func (s *MemoryStore) ListTrainingGuardrailClauses(ctx context.Context, modelKey string) ([]domain.TrainingGuardrailClauseRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	clauses := s.training.guardrailClauses[strings.TrimSpace(modelKey)]
	records := make([]domain.TrainingGuardrailClauseRecord, 0, len(clauses))
	for _, clause := range clauses {
		clause.Params = cloneJSONMap(clause.Params)
		records = append(records, clause)
	}
	sort.Slice(records, func(i, j int) bool { return records[i].ClauseKey < records[j].ClauseKey })
	return records, nil
}

func (s *MemoryStore) GetTrainingGatePolicy(ctx context.Context, modelKey string) (domain.TrainingGatePolicyRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	policy, ok := s.training.gatePolicies[strings.TrimSpace(modelKey)]
	if !ok {
		return domain.TrainingGatePolicyRecord{}, ErrNotFound
	}
	policy.MinPerClassObjects = cloneJSONMap(policy.MinPerClassObjects)
	return policy, nil
}

func validTrainingJobStatus(status string) bool {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "queued", "running", "succeeded", "failed", "canceled":
		return true
	default:
		return false
	}
}

func trainingJobStatusIsTerminal(status string) bool {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "succeeded", "failed", "canceled":
		return true
	default:
		return false
	}
}

func clampTrainingJobProgress(completed int64, total int64) int64 {
	if completed < 0 {
		return 0
	}
	if total > 0 && completed > total {
		return total
	}
	return completed
}

func validTrainingModelVersionStatus(status string) bool {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "candidate", "canary", "active", "retired", "rejected":
		return true
	default:
		return false
	}
}

// trainingGoldSetTransitionAllowed encodes the gold-set lifecycle:
// draft→freezing→frozen|failed, failed→freezing (retry), frozen→retired.
// Once frozen, ONLY →retired is allowed — frozen gold sets are immutable.
func trainingGoldSetTransitionAllowed(from string, to string) bool {
	switch from {
	case "draft":
		return to == "freezing"
	case "freezing":
		return to == "frozen" || to == "failed"
	case "failed":
		return to == "freezing"
	case "frozen":
		return to == "retired"
	default:
		return false
	}
}

func trainingGoldSetAcceptsItems(status string) bool {
	return status == "draft" || status == "freezing"
}

func cloneTrainingJob(record domain.TrainingJobRecord) domain.TrainingJobRecord {
	record.Params = cloneJSONMap(record.Params)
	record.OutputSummary = cloneJSONMap(record.OutputSummary)
	record.Metadata = cloneJSONMap(record.Metadata)
	return record
}

func cloneTrainingJobEvent(record domain.TrainingJobEventRecord) domain.TrainingJobEventRecord {
	record.Metadata = cloneJSONMap(record.Metadata)
	return record
}

func cloneTrainingGoldSet(record domain.TrainingGoldSetRecord) domain.TrainingGoldSetRecord {
	record.LabelStats = cloneJSONMap(record.LabelStats)
	record.StrataSummary = cloneJSONMap(record.StrataSummary)
	record.Provenance = cloneJSONMap(record.Provenance)
	record.FrozenAt = utcTimePtr(record.FrozenAt)
	return record
}

func cloneTrainingGoldItem(record domain.TrainingGoldItemRecord) domain.TrainingGoldItemRecord {
	record.SourceRef = cloneJSONMap(record.SourceRef)
	record.Metadata = cloneJSONMap(record.Metadata)
	record.FootprintGeom = cloneJSONMap(record.FootprintGeom)
	record.StrataTags = cloneJSONMap(record.StrataTags)
	record.Phash = cloneStringPtr(record.Phash)
	record.Width = cloneInt64Ptr(record.Width)
	record.Height = cloneInt64Ptr(record.Height)
	return record
}

func cloneTrainingBenchmarkRun(record domain.TrainingBenchmarkRunRecord) domain.TrainingBenchmarkRunRecord {
	record.Metrics = cloneJSONMap(record.Metrics)
	record.GuardrailsReasons = append([]string{}, record.GuardrailsReasons...)
	return record
}

func cloneTrainingRetrainRequest(record domain.TrainingRetrainRequestRecord) domain.TrainingRetrainRequestRecord {
	record.GatingSummary = cloneJSONMap(record.GatingSummary)
	record.StartedAt = utcTimePtr(record.StartedAt)
	record.FinishedAt = utcTimePtr(record.FinishedAt)
	return record
}

func cloneStringPtr(value *string) *string {
	if value == nil {
		return nil
	}
	out := *value
	return &out
}

func cloneInt64Ptr(value *int64) *int64 {
	if value == nil {
		return nil
	}
	out := *value
	return &out
}
