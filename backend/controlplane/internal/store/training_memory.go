package store

import (
	"context"
	"sort"
	"strings"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

// In-memory mirror of the GoldGate training reads, seeded with the same rows
// as the schema.sql seed so dev-without-Postgres and handler tests see the
// truthful M0 state (registry + version 0 + lineage + status).

type memoryTrainingState struct {
	models          map[string]domain.TrainingModelRecord
	domains         map[string]domain.TrainingDomainRecord
	lineages        map[string]domain.TrainingLineageRecord
	versions        map[string]domain.TrainingModelVersionRecord
	statuses        map[string]domain.TrainingModelStatusRecord
	retrainRequests map[string][]domain.TrainingRetrainRequestRecord
}

func newMemoryTrainingState() *memoryTrainingState {
	now := domain.Now()
	model := trainingSeedModel(now)
	trainingDomain := trainingSeedDomain(now)
	lineage := trainingSeedLineage(now)
	version := trainingSeedVersion(now)
	status := trainingSeedStatus()
	return &memoryTrainingState{
		models:          map[string]domain.TrainingModelRecord{model.ModelKey: model},
		domains:         map[string]domain.TrainingDomainRecord{trainingDomain.DomainID: trainingDomain},
		lineages:        map[string]domain.TrainingLineageRecord{lineage.LineageID: lineage},
		versions:        map[string]domain.TrainingModelVersionRecord{version.VersionID: version},
		statuses:        map[string]domain.TrainingModelStatusRecord{status.ModelKey: status},
		retrainRequests: map[string][]domain.TrainingRetrainRequestRecord{},
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
	return record
}

func cloneTrainingStatus(record domain.TrainingModelStatusRecord) domain.TrainingModelStatusRecord {
	record.ClassCounts = cloneJSONMap(record.ClassCounts)
	record.PerClassNewObjects = cloneJSONMap(record.PerClassNewObjects)
	record.UnsupportedClassCounts = cloneJSONMap(record.UnsupportedClassCounts)
	record.RetrainGateCounts = cloneJSONMap(record.RetrainGateCounts)
	record.RetrainGateThresholds = cloneJSONMap(record.RetrainGateThresholds)
	record.RetrainGateReasons = append([]string(nil), record.RetrainGateReasons...)
	return record
}
