package worker

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

var (
	ErrDataAgentJobLeaseConflict = errors.New("data agent job lease conflict")
	ErrDataAgentJobLeaseLost     = errors.New("data agent job lease lost")
)

type DataAgentWorkerStore interface {
	GetDataAgentJobForUser(context.Context, string, string, string) (domain.DataAgentJobRecord, error)
	AcquireDataAgentJobLease(context.Context, domain.AcquireDataAgentJobLeaseInput) (domain.DataAgentJobLeaseRecord, domain.DataAgentJobRecord, domain.DataAgentJobEventRecord, error)
	RenewDataAgentJobLease(context.Context, domain.RenewDataAgentJobLeaseInput) (domain.DataAgentJobLeaseRecord, error)
	ReleaseDataAgentJobLease(context.Context, domain.ReleaseDataAgentJobLeaseInput) error
	UpdateDataAgentJob(context.Context, domain.UpdateDataAgentJobInput) (domain.DataAgentJobRecord, domain.DataAgentJobEventRecord, error)
	GetResourceForUser(context.Context, string, string, string) (domain.ResourceRecord, error)
	ListResourcesForUser(context.Context, domain.ResourceListInput) (domain.ResourceListPage, error)
	ListResourcesForCollectionForUser(context.Context, domain.ResourceCollectionResourceListInput) (domain.ResourceListPage, error)
	BulkTagResourcesForUser(context.Context, domain.BulkTagResourcesInput) (domain.BulkTagResourcesResult, error)
	CreateDatasetSnapshot(context.Context, domain.CreateDatasetSnapshotInput) (domain.DatasetSnapshotRecord, []domain.DatasetSnapshotResourceRecord, error)
	MergeResourceMetadataForUser(context.Context, domain.MergeResourceMetadataInput) (domain.ResourceRecord, error)
	CreateResourceEvent(context.Context, domain.AppendResourceEventInput) (domain.ResourceEventRecord, error)
	UpsertWorkerHeartbeat(context.Context, domain.UpsertWorkerHeartbeatInput) (domain.WorkerHeartbeatRecord, error)
}

const (
	dataAgentQueryResourcePageSize = 1000
)

type DataAgentWorkerConfig struct {
	WorkerID           string
	LeaseTTL           time.Duration
	LeaseRenewInterval time.Duration
	Processor          DataAgentProcessor
	Now                func() time.Time
}

type DataAgentWorker struct {
	store              DataAgentWorkerStore
	workerID           string
	leaseTTL           time.Duration
	leaseRenewInterval time.Duration
	processor          DataAgentProcessor
	now                func() time.Time
	hostname           string
}

type DataAgentWork struct {
	Envelope       eventbus.DataAgentJob
	Job            domain.DataAgentJobRecord
	Resources      []domain.ResourceRecord
	ReportProgress func(context.Context, int, string, domain.JSONMap) error
}

type DataAgentJobResult struct {
	Message       string
	OutputSummary domain.JSONMap
	Metadata      domain.JSONMap
}

type DataAgentProcessor interface {
	ProcessDataAgentJob(context.Context, DataAgentWork) (DataAgentJobResult, error)
}

type DataAgentProcessorFunc func(context.Context, DataAgentWork) (DataAgentJobResult, error)

func (fn DataAgentProcessorFunc) ProcessDataAgentJob(ctx context.Context, work DataAgentWork) (DataAgentJobResult, error) {
	return fn(ctx, work)
}

func NewDataAgentWorker(store DataAgentWorkerStore, cfg DataAgentWorkerConfig) *DataAgentWorker {
	workerID := strings.TrimSpace(cfg.WorkerID)
	if workerID == "" {
		workerID = "data-agent-worker-local"
	}
	leaseTTL := cfg.LeaseTTL
	if leaseTTL <= 0 {
		leaseTTL = time.Minute
	}
	renewInterval := cfg.LeaseRenewInterval
	if renewInterval <= 0 || renewInterval >= leaseTTL {
		renewInterval = leaseTTL / 2
	}
	if renewInterval <= 0 {
		renewInterval = 30 * time.Second
	}
	now := cfg.Now
	if now == nil {
		now = domain.Now
	}
	processor := cfg.Processor
	if processor == nil {
		processor = MetadataSummaryDataAgentProcessor{}
	}
	hostname, _ := os.Hostname()
	return &DataAgentWorker{
		store:              store,
		workerID:           workerID,
		leaseTTL:           leaseTTL,
		leaseRenewInterval: renewInterval,
		processor:          processor,
		now:                now,
		hostname:           hostname,
	}
}

func (w *DataAgentWorker) RunJob(ctx context.Context, envelope eventbus.DataAgentJob) error {
	if w == nil || w.store == nil {
		return errors.New("data agent worker store is required")
	}
	jobID := strings.TrimSpace(envelope.JobID)
	if jobID == "" {
		return errors.New("data agent job id is required")
	}
	ownerUserID := strings.TrimSpace(envelope.OwnerUserID)
	ownerOrgID := strings.TrimSpace(envelope.OwnerOrgID)
	current, err := w.store.GetDataAgentJobForUser(ctx, jobID, ownerUserID, ownerOrgID)
	if errors.Is(err, store.ErrNotFound) {
		return nil
	}
	if err != nil {
		return err
	}
	if dataAgentJobTerminal(current.Status) {
		return nil
	}
	lease, leasedJob, _, err := w.store.AcquireDataAgentJobLease(ctx, domain.AcquireDataAgentJobLeaseInput{
		JobID:       jobID,
		OwnerUserID: ownerUserID,
		OwnerOrgID:  ownerOrgID,
		WorkerID:    w.workerID,
		TTL:         w.leaseTTL,
		Now:         w.now(),
	})
	if errors.Is(err, store.ErrConflict) {
		latest, latestErr := w.store.GetDataAgentJobForUser(ctx, jobID, ownerUserID, ownerOrgID)
		if latestErr == nil && dataAgentJobTerminal(latest.Status) {
			return nil
		}
		return ErrDataAgentJobLeaseConflict
	}
	if errors.Is(err, store.ErrNotFound) {
		return nil
	}
	if err != nil {
		return err
	}
	_ = w.workerHeartbeat(ctx, "busy", jobID)
	renewCtx, stopRenew := context.WithCancel(ctx)
	leaseLost := make(chan error, 1)
	renewDone := make(chan struct{})
	go func() {
		defer close(renewDone)
		if err := w.renewLeaseLoop(renewCtx, lease, ownerUserID, ownerOrgID); err != nil {
			select {
			case leaseLost <- err:
			default:
			}
		}
	}()
	defer func() {
		stopRenew()
		<-renewDone
		_ = w.workerHeartbeat(context.Background(), "idle", "")
	}()

	resources, err := w.loadResources(ctx, envelope, leasedJob)
	if err != nil {
		return w.failAndRelease(ctx, leasedJob, lease, err)
	}
	progressTotal := dataAgentProgressTotal(envelope, leasedJob, resources)
	reportProgress := func(progressCtx context.Context, completed int, message string, metadata domain.JSONMap) error {
		return w.reportProgress(progressCtx, leasedJob, completed, progressTotal, message, metadata)
	}
	work := DataAgentWork{
		Envelope:       envelope,
		Job:            leasedJob,
		Resources:      resources,
		ReportProgress: reportProgress,
	}
	var result DataAgentJobResult
	switch normalizedDataAgentJobType(leasedJob.JobType) {
	case "batch_tag_resources":
		result, resources, err = w.processBatchTagResources(ctx, envelope, leasedJob, resources, reportProgress)
	case "create_dataset_snapshot":
		result, resources, err = w.processCreateDatasetSnapshot(ctx, envelope, leasedJob, resources, reportProgress)
	default:
		result, err = w.processor.ProcessDataAgentJob(ctx, work)
	}
	if progressTotal <= 0 {
		progressTotal = dataAgentResultResourceCount(result)
	}
	if err != nil {
		if ctx.Err() != nil || errors.Is(err, context.Canceled) {
			_ = w.releaseLease(context.Background(), leasedJob, lease)
			if ctx.Err() != nil {
				return ctx.Err()
			}
			return err
		}
		return w.failAndRelease(ctx, leasedJob, lease, err)
	}
	select {
	case err := <-leaseLost:
		return fmt.Errorf("%w: %v", ErrDataAgentJobLeaseLost, err)
	default:
	}
	if ctx.Err() != nil {
		_ = w.releaseLease(context.Background(), leasedJob, lease)
		return ctx.Err()
	}
	completedAt := w.now()
	if err := w.persistResourceDataAgentStatuses(ctx, leasedJob, resources, result, completedAt); err != nil {
		return w.failAndRelease(ctx, leasedJob, lease, err)
	}
	if err := w.completeAndRelease(ctx, leasedJob, lease, progressTotal, result, completedAt); err != nil {
		return err
	}
	return nil
}

func (w *DataAgentWorker) processBatchTagResources(
	ctx context.Context,
	envelope eventbus.DataAgentJob,
	job domain.DataAgentJobRecord,
	resources []domain.ResourceRecord,
	reportProgress func(context.Context, int, string, domain.JSONMap) error,
) (DataAgentJobResult, []domain.ResourceRecord, error) {
	tags := dataAgentJobTags(envelope, job)
	if len(tags) == 0 {
		return DataAgentJobResult{}, resources, errors.New("batch_tag_resources requires tags")
	}
	resourceIDs := make([]string, 0, len(resources))
	for _, resource := range resources {
		resourceIDs = append(resourceIDs, resource.ResourceID)
	}
	ts := w.now()
	tagged, err := w.store.BulkTagResourcesForUser(ctx, domain.BulkTagResourcesInput{
		OwnerUserID: job.OwnerUserID,
		OwnerOrgID:  job.OwnerOrgID,
		ActorUserID: job.OwnerUserID,
		ActorOrgID:  job.OwnerOrgID,
		ResourceIDs: resourceIDs,
		Tags:        tags,
		TS:          ts,
		Metadata: domain.JSONMap{
			"job_id":    job.JobID,
			"job_type":  "batch_tag_resources",
			"worker_id": w.workerID,
		},
	})
	if err != nil {
		return DataAgentJobResult{}, resources, err
	}
	for index, resource := range tagged.Resources {
		if reportProgress != nil {
			if err := reportProgress(ctx, index+1, "Tagged "+resource.OriginalName+".", domain.JSONMap{
				"resource_id": resource.ResourceID,
				"tags_added":  append([]string(nil), tags...),
			}); err != nil {
				return DataAgentJobResult{}, tagged.Resources, err
			}
		}
	}
	summary := domain.JSONMap{
		"summary_kind":          "batch_tagging",
		"summary":               fmt.Sprintf("Applied %d tags to %d resources.", len(tags), tagged.UpdatedCount),
		"job_type":              "batch_tag_resources",
		"resource_count":        len(resources),
		"tagged_resource_count": tagged.UpdatedCount,
		"tags_added":            append([]string(nil), tags...),
		"resources":             dataAgentResourceSummaries(tagged.Resources),
	}
	return DataAgentJobResult{
		Message:       "Data Agent batch tagging completed.",
		OutputSummary: summary,
		Metadata: domain.JSONMap{
			"processor":  "batch_tag_resources",
			"tags_added": append([]string(nil), tags...),
		},
	}, tagged.Resources, nil
}

func (w *DataAgentWorker) processCreateDatasetSnapshot(
	ctx context.Context,
	envelope eventbus.DataAgentJob,
	job domain.DataAgentJobRecord,
	resources []domain.ResourceRecord,
	reportProgress func(context.Context, int, string, domain.JSONMap) error,
) (DataAgentJobResult, []domain.ResourceRecord, error) {
	resourceQuery := dataAgentJobResourceQuery(envelope, job)
	sourceCollectionID := dataAgentJobString(envelope, job, "source_collection_id", "collection_id")
	if len(resources) == 0 && sourceCollectionID == "" && resourceQuery == nil {
		return DataAgentJobResult{}, resources, errors.New("create_dataset_snapshot requires resources, source collection, or resource query")
	}
	resourceIDs := make([]string, 0, len(resources))
	for _, resource := range resources {
		resourceIDs = append(resourceIDs, resource.ResourceID)
	}
	ts := w.now()
	name := dataAgentJobString(envelope, job, "snapshot_name", "name", "dataset_name")
	if name == "" {
		name = "Data Agent dataset " + ts.UTC().Format("2006-01-02 15:04")
	}
	description := dataAgentJobString(envelope, job, "description", "snapshot_description", "dataset_description")
	metadata := cloneWorkerJSONMap(job.Metadata)
	for key, value := range cloneWorkerJSONMap(envelope.Metadata) {
		if _, exists := metadata[key]; !exists {
			metadata[key] = value
		}
	}
	metadata["job_id"] = job.JobID
	metadata["job_type"] = "create_dataset_snapshot"
	metadata["worker_id"] = w.workerID
	if len(resourceIDs) > 0 {
		metadata["resource_count"] = len(resourceIDs)
	}
	if sourceCollectionID != "" {
		metadata["source_collection_id"] = sourceCollectionID
	}
	if resourceQuery != nil {
		metadata["resource_query"] = dataAgentDatasetSnapshotResourceQueryMetadata(resourceQuery)
	}
	snapshot, entries, err := w.store.CreateDatasetSnapshot(ctx, domain.CreateDatasetSnapshotInput{
		OwnerUserID:        job.OwnerUserID,
		OwnerOrgID:         job.OwnerOrgID,
		OwnerRole:          job.OwnerRole,
		ProjectID:          strings.TrimSpace(job.ProjectID),
		SourceCollectionID: sourceCollectionID,
		Name:               name,
		Description:        description,
		ResourceIDs:        resourceIDs,
		ResourceQuery:      resourceQuery,
		CreatedByUserID:    job.OwnerUserID,
		CreatedAt:          ts,
		Metadata:           metadata,
	})
	if err != nil {
		return DataAgentJobResult{}, resources, err
	}
	snapshotResourceIDs := dataAgentSnapshotResourceIDs(entries)
	for index, entry := range entries {
		if reportProgress != nil {
			if err := reportProgress(ctx, index+1, "Snapshotted "+entry.OriginalName+".", domain.JSONMap{
				"resource_id":   entry.ResourceID,
				"snapshot_id":   snapshot.SnapshotID,
				"snapshot_name": snapshot.Name,
			}); err != nil {
				return DataAgentJobResult{}, resources, err
			}
		}
		if _, err := w.store.CreateResourceEvent(ctx, domain.AppendResourceEventInput{
			ResourceID:  entry.ResourceID,
			ActorUserID: job.OwnerUserID,
			ActorOrgID:  job.OwnerOrgID,
			EventType:   "resource.dataset_snapshotted",
			TS:          ts,
			Metadata: domain.JSONMap{
				"snapshot_id":          snapshot.SnapshotID,
				"snapshot_name":        snapshot.Name,
				"source_collection_id": snapshot.SourceCollectionID,
				"job_id":               job.JobID,
				"job_type":             "create_dataset_snapshot",
				"worker_id":            w.workerID,
			},
		}); err != nil {
			return DataAgentJobResult{}, resources, fmt.Errorf("record dataset snapshot resource event %s: %w", entry.ResourceID, err)
		}
	}
	summary := domain.JSONMap{
		"summary_kind":          "dataset_snapshot",
		"summary":               fmt.Sprintf("Created dataset snapshot %q with %d resources.", snapshot.Name, snapshot.ResourceCount),
		"job_type":              "create_dataset_snapshot",
		"snapshot_id":           snapshot.SnapshotID,
		"snapshot_name":         snapshot.Name,
		"source_collection_id":  snapshot.SourceCollectionID,
		"resource_count":        snapshot.ResourceCount,
		"total_size_bytes":      snapshot.TotalBytes,
		"snapshot_created_at":   snapshot.CreatedAt.UTC().Format(time.RFC3339Nano),
		"snapshot_resource_ids": snapshotResourceIDs,
		"resources":             dataAgentDatasetSnapshotResourceSummaries(entries),
	}
	if resourceQuery != nil {
		summary["resource_query"] = dataAgentDatasetSnapshotResourceQueryMetadata(resourceQuery)
	}
	return DataAgentJobResult{
		Message:       "Data Agent dataset snapshot created.",
		OutputSummary: summary,
		Metadata: domain.JSONMap{
			"processor":   "create_dataset_snapshot",
			"snapshot_id": snapshot.SnapshotID,
		},
	}, resources, nil
}

func (w *DataAgentWorker) RunLoop(ctx context.Context, jobs <-chan eventbus.DataAgentJob) {
	for {
		select {
		case <-ctx.Done():
			return
		case job, ok := <-jobs:
			if !ok {
				return
			}
			_ = w.RunJob(ctx, job)
		}
	}
}

func (w *DataAgentWorker) renewLeaseLoop(ctx context.Context, lease domain.DataAgentJobLeaseRecord, ownerUserID string, ownerOrgID string) error {
	ticker := time.NewTicker(w.leaseRenewInterval)
	defer ticker.Stop()
	current := lease
	for {
		select {
		case <-ctx.Done():
			return nil
		case <-ticker.C:
			renewed, err := w.store.RenewDataAgentJobLease(ctx, domain.RenewDataAgentJobLeaseInput{
				JobID:       current.JobID,
				OwnerUserID: ownerUserID,
				OwnerOrgID:  ownerOrgID,
				LeaseToken:  current.LeaseToken,
				TTL:         w.leaseTTL,
				Now:         w.now(),
			})
			if err != nil {
				return err
			}
			current = renewed
		}
	}
}

func (w *DataAgentWorker) loadResources(ctx context.Context, envelope eventbus.DataAgentJob, job domain.DataAgentJobRecord) ([]domain.ResourceRecord, error) {
	resourceIDs := dataAgentEnvelopeResourceIDs(envelope)
	if len(resourceIDs) > 0 {
		resources := make([]domain.ResourceRecord, 0, len(resourceIDs))
		for _, resourceID := range resourceIDs {
			resource, err := w.store.GetResourceForUser(ctx, resourceID, job.OwnerUserID, job.OwnerOrgID)
			if err != nil {
				return nil, fmt.Errorf("load data agent resource %s: %w", resourceID, err)
			}
			resources = append(resources, resource)
		}
		return resources, nil
	}
	query := dataAgentJobResourceQuery(envelope, job)
	if query == nil {
		return nil, nil
	}
	sourceCollectionID := dataAgentJobString(envelope, job, "source_collection_id", "collection_id")
	resources := []domain.ResourceRecord{}
	offset := 0
	for {
		remaining := domain.DataAgentQueryResourceHardLimit - len(resources)
		if remaining <= 0 {
			return nil, fmt.Errorf("data agent query matched at least %d resources, above one-job limit %d", len(resources), domain.DataAgentQueryResourceHardLimit)
		}
		limit := dataAgentQueryResourcePageSize
		if remaining < limit {
			limit = remaining
		}
		page, err := w.listResourceQueryPage(ctx, job, sourceCollectionID, query, limit, offset)
		if err != nil {
			return nil, fmt.Errorf("load data agent resources from query: %w", err)
		}
		if page.TotalCount > domain.DataAgentQueryResourceHardLimit {
			return nil, fmt.Errorf("data agent query matched %d resources, above one-job limit %d", page.TotalCount, domain.DataAgentQueryResourceHardLimit)
		}
		resources = append(resources, page.Resources...)
		if len(resources) >= page.TotalCount {
			return resources, nil
		}
		if len(page.Resources) == 0 {
			return nil, fmt.Errorf("data agent query returned %d resources before reported total %d", len(resources), page.TotalCount)
		}
		offset += len(page.Resources)
	}
}

func (w *DataAgentWorker) listResourceQueryPage(
	ctx context.Context,
	job domain.DataAgentJobRecord,
	sourceCollectionID string,
	query *domain.DatasetSnapshotResourceQuery,
	limit int,
	offset int,
) (domain.ResourceListPage, error) {
	if sourceCollectionID != "" {
		return w.store.ListResourcesForCollectionForUser(ctx, domain.ResourceCollectionResourceListInput{
			CollectionID:     sourceCollectionID,
			UserID:           job.OwnerUserID,
			OrgID:            job.OwnerOrgID,
			Query:            query.Query,
			Kind:             query.Kind,
			Source:           query.Source,
			ProjectID:        query.ProjectID,
			Sharing:          query.Sharing,
			Tags:             query.Tags,
			Descriptors:      query.Descriptors,
			MetadataFilters:  query.MetadataFilters,
			CreatedAfter:     query.CreatedAfter,
			CreatedBefore:    query.CreatedBefore,
			ProcessingStatus: query.ProcessingStatus,
			Limit:            limit,
			Offset:           offset,
		})
	}
	return w.store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID:           job.OwnerUserID,
		OrgID:            job.OwnerOrgID,
		Query:            query.Query,
		Kind:             query.Kind,
		Source:           query.Source,
		ProjectID:        query.ProjectID,
		Sharing:          query.Sharing,
		Tags:             query.Tags,
		Descriptors:      query.Descriptors,
		MetadataFilters:  query.MetadataFilters,
		CreatedAfter:     query.CreatedAfter,
		CreatedBefore:    query.CreatedBefore,
		ProcessingStatus: query.ProcessingStatus,
		Limit:            limit,
		Offset:           offset,
	})
}

func (w *DataAgentWorker) reportProgress(ctx context.Context, job domain.DataAgentJobRecord, completed int, total int, message string, metadata domain.JSONMap) error {
	eventMetadata := cloneWorkerJSONMap(metadata)
	eventMetadata["worker_id"] = w.workerID
	if completed < 0 {
		completed = 0
	}
	if total < completed {
		total = completed
	}
	_, _, err := w.store.UpdateDataAgentJob(ctx, domain.UpdateDataAgentJobInput{
		JobID:             job.JobID,
		OwnerUserID:       job.OwnerUserID,
		OwnerOrgID:        job.OwnerOrgID,
		Status:            "running",
		ProgressCompleted: completed,
		ProgressTotal:     total,
		ActorUserID:       job.OwnerUserID,
		ActorOrgID:        job.OwnerOrgID,
		Message:           strings.TrimSpace(message),
		UpdatedAt:         w.now(),
		EventMetadata:     eventMetadata,
	})
	return err
}

func (w *DataAgentWorker) completeAndRelease(ctx context.Context, job domain.DataAgentJobRecord, lease domain.DataAgentJobLeaseRecord, progressTotal int, result DataAgentJobResult, completedAt time.Time) error {
	metadata := cloneWorkerJSONMap(result.Metadata)
	metadata["worker_id"] = w.workerID
	message := strings.TrimSpace(result.Message)
	if message == "" {
		message = "Data Agent job completed."
	}
	outputSummary := cloneWorkerJSONMap(result.OutputSummary)
	if len(outputSummary) == 0 {
		outputSummary = domain.JSONMap{"summary": message}
	}
	if _, _, err := w.store.UpdateDataAgentJob(ctx, domain.UpdateDataAgentJobInput{
		JobID:             job.JobID,
		OwnerUserID:       job.OwnerUserID,
		OwnerOrgID:        job.OwnerOrgID,
		Status:            "succeeded",
		ProgressCompleted: progressTotal,
		ProgressTotal:     progressTotal,
		ActorUserID:       job.OwnerUserID,
		ActorOrgID:        job.OwnerOrgID,
		Message:           message,
		UpdatedAt:         completedAt,
		CompletedAt:       completedAt,
		OutputSummary:     outputSummary,
		EventMetadata:     metadata,
	}); err != nil {
		return err
	}
	return w.releaseLease(ctx, job, lease)
}

func (w *DataAgentWorker) persistResourceDataAgentStatuses(ctx context.Context, job domain.DataAgentJobRecord, resources []domain.ResourceRecord, result DataAgentJobResult, completedAt time.Time) error {
	if len(resources) == 0 {
		return nil
	}
	jobType := normalizedDataAgentJobType(job.JobType)
	summaryKind := strings.TrimSpace(fmt.Sprint(result.OutputSummary["summary_kind"]))
	summaryText := strings.TrimSpace(fmt.Sprint(result.OutputSummary["summary"]))
	captions := dataAgentCaptionsByResourceID(result.OutputSummary)
	timestamp := completedAt.UTC().Format(time.RFC3339Nano)
	for _, resource := range resources {
		status := domain.JSONMap{
			"status":       "succeeded",
			"job_id":       job.JobID,
			"job_type":     jobType,
			"completed_at": timestamp,
			"updated_at":   timestamp,
			"worker_id":    w.workerID,
		}
		if summaryKind != "" && summaryKind != "<nil>" {
			status["summary_kind"] = summaryKind
		}
		if summaryText != "" && summaryText != "<nil>" {
			status["summary"] = summaryText
		}
		if caption, ok := captions[resource.ResourceID]; ok {
			if text := strings.TrimSpace(fmt.Sprint(caption["caption"])); text != "" && text != "<nil>" {
				status["caption"] = text
			}
			if source := strings.TrimSpace(fmt.Sprint(caption["caption_source"])); source != "" && source != "<nil>" {
				status["caption_source"] = source
			}
		}
		if _, err := w.store.MergeResourceMetadataForUser(ctx, domain.MergeResourceMetadataInput{
			ResourceID: resource.ResourceID,
			UserID:     job.OwnerUserID,
			OrgID:      job.OwnerOrgID,
			UpdatedAt:  completedAt,
			Patch: domain.JSONMap{
				"data_agent": domain.JSONMap{
					jobType: status,
				},
			},
		}); err != nil {
			return fmt.Errorf("persist data agent resource status %s: %w", resource.ResourceID, err)
		}
		if _, err := w.store.CreateResourceEvent(ctx, domain.AppendResourceEventInput{
			ResourceID:  resource.ResourceID,
			ActorUserID: job.OwnerUserID,
			ActorOrgID:  job.OwnerOrgID,
			EventType:   "resource.data_agent_job_completed",
			TS:          completedAt,
			Metadata: domain.JSONMap{
				"job_id":       job.JobID,
				"job_type":     jobType,
				"status":       "succeeded",
				"summary_kind": summaryKind,
				"worker_id":    w.workerID,
			},
		}); err != nil {
			return fmt.Errorf("record data agent resource event %s: %w", resource.ResourceID, err)
		}
	}
	return nil
}

func (w *DataAgentWorker) failAndRelease(ctx context.Context, job domain.DataAgentJobRecord, lease domain.DataAgentJobLeaseRecord, cause error) error {
	if cause == nil {
		cause = errors.New("data agent job failed")
	}
	_, _, err := w.store.UpdateDataAgentJob(ctx, domain.UpdateDataAgentJobInput{
		JobID:             job.JobID,
		OwnerUserID:       job.OwnerUserID,
		OwnerOrgID:        job.OwnerOrgID,
		Status:            "failed",
		ProgressCompleted: job.ProgressCompleted,
		ProgressTotal:     dataAgentProgressTotal(eventbus.DataAgentJob{}, job, nil),
		Error:             cause.Error(),
		ActorUserID:       job.OwnerUserID,
		ActorOrgID:        job.OwnerOrgID,
		Message:           "Data Agent job failed.",
		UpdatedAt:         w.now(),
		CompletedAt:       w.now(),
		EventMetadata: domain.JSONMap{
			"worker_id": w.workerID,
			"error":     cause.Error(),
		},
	})
	releaseErr := w.releaseLease(ctx, job, lease)
	if err != nil {
		return err
	}
	return releaseErr
}

func (w *DataAgentWorker) releaseLease(ctx context.Context, job domain.DataAgentJobRecord, lease domain.DataAgentJobLeaseRecord) error {
	return w.store.ReleaseDataAgentJobLease(ctx, domain.ReleaseDataAgentJobLeaseInput{
		JobID:       job.JobID,
		OwnerUserID: job.OwnerUserID,
		OwnerOrgID:  job.OwnerOrgID,
		LeaseToken:  lease.LeaseToken,
	})
}

func (w *DataAgentWorker) workerHeartbeat(ctx context.Context, status string, currentJobID string) error {
	_, err := w.store.UpsertWorkerHeartbeat(ctx, domain.UpsertWorkerHeartbeatInput{
		WorkerID:        w.workerID,
		WorkerKind:      "data_agent",
		Status:          status,
		CurrentRunID:    currentJobID,
		Hostname:        w.hostname,
		LastHeartbeatAt: w.now(),
		Metadata:        domain.JSONMap{"job_kind": "data_agent"},
	})
	return err
}

type MetadataSummaryDataAgentProcessor struct{}

func (MetadataSummaryDataAgentProcessor) ProcessDataAgentJob(ctx context.Context, work DataAgentWork) (DataAgentJobResult, error) {
	resourceSummaries := dataAgentResourceSummaries(work.Resources)
	for index, resource := range work.Resources {
		select {
		case <-ctx.Done():
			return DataAgentJobResult{}, ctx.Err()
		default:
		}
		if work.ReportProgress != nil {
			if err := work.ReportProgress(ctx, index+1, "Summarized "+resource.OriginalName+".", domain.JSONMap{"resource_id": resource.ResourceID}); err != nil {
				return DataAgentJobResult{}, err
			}
		}
	}
	outputSummary := dataAgentTemplateSummary(work.Envelope.JobType, work.Resources, resourceSummaries)
	return DataAgentJobResult{
		Message:       "Data Agent " + dataAgentTemplateDisplayName(work.Envelope.JobType) + " completed.",
		OutputSummary: outputSummary,
		Metadata:      domain.JSONMap{"processor": "metadata_summary"},
	}, nil
}

func dataAgentResourceSummaries(resources []domain.ResourceRecord) []domain.JSONMap {
	resourceSummaries := make([]domain.JSONMap, 0, len(resources))
	for _, resource := range resources {
		resourceSummaries = append(resourceSummaries, domain.JSONMap{
			"resource_id":   resource.ResourceID,
			"original_name": resource.OriginalName,
			"resource_kind": resource.ResourceKind,
			"source_type":   resource.SourceType,
			"content_type":  resource.ContentType,
			"size_bytes":    resource.SizeBytes,
			"sha256":        resource.SHA256,
			"project_id":    resource.ProjectID,
			"status":        resource.Status,
			"metadata":      cloneWorkerJSONMap(resource.Metadata),
		})
	}
	return resourceSummaries
}

func dataAgentDatasetSnapshotResourceSummaries(entries []domain.DatasetSnapshotResourceRecord) []domain.JSONMap {
	resourceSummaries := make([]domain.JSONMap, 0, len(entries))
	for _, entry := range entries {
		resourceSummaries = append(resourceSummaries, domain.JSONMap{
			"resource_id":   entry.ResourceID,
			"original_name": entry.OriginalName,
			"resource_kind": entry.ResourceKind,
			"source_type":   entry.SourceType,
			"content_type":  entry.ContentType,
			"size_bytes":    entry.SizeBytes,
			"sha256":        entry.SHA256,
			"project_id":    entry.ProjectID,
			"position":      entry.Position,
		})
	}
	return resourceSummaries
}

func dataAgentSnapshotResourceIDs(entries []domain.DatasetSnapshotResourceRecord) []string {
	resourceIDs := make([]string, 0, len(entries))
	for _, entry := range entries {
		resourceIDs = append(resourceIDs, entry.ResourceID)
	}
	return resourceIDs
}

func dataAgentTemplateSummary(jobType string, resources []domain.ResourceRecord, resourceSummaries []domain.JSONMap) domain.JSONMap {
	normalizedJobType := normalizedDataAgentJobType(jobType)
	if normalizedJobType == "" {
		normalizedJobType = "extract_metadata"
	}
	base := domain.JSONMap{
		"job_type":       normalizedJobType,
		"resource_count": len(resources),
		"resources":      resourceSummaries,
		"summary":        dataAgentSummaryText(normalizedJobType, resources),
	}
	switch normalizedJobType {
	case "caption_resources":
		return mergeWorkerJSONMap(base, dataAgentCaptionSummary(resources))
	case "organize_resources":
		return mergeWorkerJSONMap(base, dataAgentOrganizationSummary(resources))
	case "deduplicate_resources":
		return mergeWorkerJSONMap(base, dataAgentDeduplicateSummary(resources))
	case "quality_check_resources":
		return mergeWorkerJSONMap(base, dataAgentQualitySummary(resources))
	case "extract_metadata":
		return mergeWorkerJSONMap(base, dataAgentMetadataExtractionSummary(resources))
	default:
		return mergeWorkerJSONMap(base, dataAgentMetadataExtractionSummary(resources))
	}
}

func normalizedDataAgentJobType(jobType string) string {
	return strings.ToLower(strings.TrimSpace(jobType))
}

func dataAgentCaptionsByResourceID(outputSummary domain.JSONMap) map[string]domain.JSONMap {
	out := map[string]domain.JSONMap{}
	rawCaptions, ok := outputSummary["captions"]
	if !ok {
		return out
	}
	switch captions := rawCaptions.(type) {
	case []domain.JSONMap:
		for _, caption := range captions {
			resourceID := strings.TrimSpace(fmt.Sprint(caption["resource_id"]))
			if resourceID != "" && resourceID != "<nil>" {
				out[resourceID] = cloneWorkerJSONMap(caption)
			}
		}
	case []any:
		for _, item := range captions {
			if caption, ok := workerJSONMap(item); ok {
				resourceID := strings.TrimSpace(fmt.Sprint(caption["resource_id"]))
				if resourceID != "" && resourceID != "<nil>" {
					out[resourceID] = cloneWorkerJSONMap(caption)
				}
			}
		}
	}
	return out
}

func dataAgentMetadataExtractionSummary(resources []domain.ResourceRecord) domain.JSONMap {
	metadataKeys := map[string]struct{}{}
	totalSizeBytes := int64(0)
	for _, resource := range resources {
		totalSizeBytes += resource.SizeBytes
		for key := range resource.Metadata {
			metadataKeys[key] = struct{}{}
		}
	}
	return domain.JSONMap{
		"summary_kind":         "metadata_extraction",
		"total_size_bytes":     totalSizeBytes,
		"resource_kind_counts": dataAgentCountBy(resources, func(resource domain.ResourceRecord) string { return resource.ResourceKind }),
		"content_type_counts":  dataAgentCountBy(resources, func(resource domain.ResourceRecord) string { return resource.ContentType }),
		"source_type_counts":   dataAgentCountBy(resources, func(resource domain.ResourceRecord) string { return resource.SourceType }),
		"project_counts":       dataAgentCountBy(resources, func(resource domain.ResourceRecord) string { return resource.ProjectID }),
		"metadata_keys":        sortedWorkerSet(metadataKeys),
	}
}

func dataAgentCaptionSummary(resources []domain.ResourceRecord) domain.JSONMap {
	captions := make([]domain.JSONMap, 0, len(resources))
	for _, resource := range resources {
		captions = append(captions, domain.JSONMap{
			"resource_id":    resource.ResourceID,
			"original_name":  resource.OriginalName,
			"caption":        dataAgentResourceCaption(resource),
			"caption_source": "deterministic_metadata",
		})
	}
	return domain.JSONMap{
		"summary_kind": "caption_generation",
		"captions":     captions,
	}
}

func dataAgentOrganizationSummary(resources []domain.ResourceRecord) domain.JSONMap {
	groups := map[string]domain.JSONMap{}
	addGroup := func(rule string, name string, resource domain.ResourceRecord) {
		name = strings.TrimSpace(name)
		if name == "" {
			name = "Unspecified"
		}
		key := rule + "\x00" + strings.ToLower(name)
		group, ok := groups[key]
		if !ok {
			group = domain.JSONMap{
				"name":         name,
				"rule":         rule,
				"reason":       dataAgentOrganizationReason(rule, name),
				"resource_ids": []string{},
			}
		}
		group["resource_ids"] = append(group["resource_ids"].([]string), resource.ResourceID)
		groups[key] = group
	}
	for _, resource := range resources {
		addGroup("project", resource.ProjectID, resource)
		addGroup("resource_kind", resource.ResourceKind, resource)
		addGroup("source_type", resource.SourceType, resource)
		addGroup("file_extension", dataAgentFileExtension(resource.OriginalName), resource)
		if label := strings.TrimSpace(fmt.Sprint(resource.Metadata["label"])); label != "" && label != "<nil>" {
			addGroup("metadata_label", label, resource)
		}
	}
	suggestions := make([]domain.JSONMap, 0, len(groups))
	for _, group := range groups {
		ids := append([]string(nil), group["resource_ids"].([]string)...)
		sort.Strings(ids)
		group["resource_ids"] = ids
		group["resource_count"] = len(ids)
		suggestions = append(suggestions, group)
	}
	sort.Slice(suggestions, func(i, j int) bool {
		leftRule := fmt.Sprint(suggestions[i]["rule"])
		rightRule := fmt.Sprint(suggestions[j]["rule"])
		if leftRule != rightRule {
			return leftRule < rightRule
		}
		return fmt.Sprint(suggestions[i]["name"]) < fmt.Sprint(suggestions[j]["name"])
	})
	return domain.JSONMap{
		"summary_kind":           "organization_plan",
		"collection_suggestions": suggestions,
	}
}

func dataAgentDeduplicateSummary(resources []domain.ResourceRecord) domain.JSONMap {
	byChecksum := map[string][]domain.ResourceRecord{}
	for _, resource := range resources {
		sha := strings.TrimSpace(resource.SHA256)
		if sha == "" {
			continue
		}
		byChecksum[sha] = append(byChecksum[sha], resource)
	}
	groups := make([]domain.JSONMap, 0)
	duplicateCount := 0
	for sha, grouped := range byChecksum {
		if len(grouped) < 2 {
			continue
		}
		resourceIDs := make([]string, 0, len(grouped))
		originalNames := make([]string, 0, len(grouped))
		for _, resource := range grouped {
			resourceIDs = append(resourceIDs, resource.ResourceID)
			originalNames = append(originalNames, resource.OriginalName)
		}
		sort.Strings(resourceIDs)
		sort.Strings(originalNames)
		duplicateCount += len(grouped)
		groups = append(groups, domain.JSONMap{
			"sha256":         sha,
			"resource_ids":   resourceIDs,
			"original_names": originalNames,
			"resource_count": len(grouped),
		})
	}
	sort.Slice(groups, func(i, j int) bool {
		return fmt.Sprint(groups[i]["sha256"]) < fmt.Sprint(groups[j]["sha256"])
	})
	return domain.JSONMap{
		"summary_kind":             "duplicate_detection",
		"duplicate_groups":         groups,
		"duplicate_group_count":    len(groups),
		"duplicate_resource_count": duplicateCount,
		"unique_resource_count":    len(resources) - duplicateCount + len(groups),
	}
}

func dataAgentQualitySummary(resources []domain.ResourceRecord) domain.JSONMap {
	warnings := make([]domain.JSONMap, 0)
	for _, resource := range resources {
		if strings.TrimSpace(resource.SHA256) == "" {
			warnings = append(warnings, dataAgentQualityWarning(resource, "missing_checksum", "warning", "Resource is missing a committed SHA-256 checksum."))
		}
		if resource.SizeBytes <= 0 {
			warnings = append(warnings, dataAgentQualityWarning(resource, "empty_file", "error", "Resource has zero committed bytes."))
		}
		if strings.TrimSpace(resource.ContentType) == "" {
			warnings = append(warnings, dataAgentQualityWarning(resource, "missing_content_type", "warning", "Resource is missing a content type."))
		}
		if strings.TrimSpace(resource.ProjectID) == "" {
			warnings = append(warnings, dataAgentQualityWarning(resource, "missing_project_id", "info", "Resource is not assigned to a project."))
		}
		if status := strings.ToLower(strings.TrimSpace(resource.Status)); status != "" && status != "active" {
			warnings = append(warnings, dataAgentQualityWarning(resource, "non_active_status", "warning", "Resource is not active."))
		}
	}
	return domain.JSONMap{
		"summary_kind":  "quality_check",
		"warning_count": len(warnings),
		"warnings":      warnings,
	}
}

func dataAgentEnvelopeResourceIDs(envelope eventbus.DataAgentJob) []string {
	if len(envelope.ResourceIDs) > 0 {
		return uniqueWorkerStrings(envelope.ResourceIDs)
	}
	values, ok := envelope.InputSelector["resource_ids"]
	if !ok {
		return nil
	}
	switch typed := values.(type) {
	case []string:
		return uniqueWorkerStrings(typed)
	case []any:
		out := make([]string, 0, len(typed))
		for _, value := range typed {
			if str, ok := value.(string); ok {
				out = append(out, str)
			}
		}
		return uniqueWorkerStrings(out)
	default:
		return nil
	}
}

func dataAgentJobTags(envelope eventbus.DataAgentJob, job domain.DataAgentJobRecord) []string {
	for _, value := range []any{
		envelope.InputSelector["tags"],
		job.InputSelector["tags"],
		envelope.Metadata["tags"],
		job.Metadata["tags"],
	} {
		if tags := dataAgentStringSlice(value); len(tags) > 0 {
			return normalizeDataAgentTags(tags)
		}
	}
	return nil
}

func dataAgentJobResourceQuery(envelope eventbus.DataAgentJob, job domain.DataAgentJobRecord) *domain.DatasetSnapshotResourceQuery {
	for _, value := range []any{
		envelope.InputSelector["resource_query"],
		job.InputSelector["resource_query"],
	} {
		query := dataAgentDatasetSnapshotResourceQuery(value)
		if query != nil {
			return query
		}
	}
	return nil
}

func dataAgentDatasetSnapshotResourceQuery(value any) *domain.DatasetSnapshotResourceQuery {
	queryMap, ok := workerJSONMap(value)
	if !ok {
		return nil
	}
	query := domain.DatasetSnapshotResourceQuery{
		Query:            dataAgentStringValue(queryMap["q"]),
		Kind:             dataAgentStringValue(queryMap["kind"]),
		Source:           dataAgentStringValue(queryMap["source"]),
		ProjectID:        dataAgentStringValue(queryMap["project_id"]),
		Sharing:          dataAgentStringValue(queryMap["sharing"]),
		Tags:             normalizeDataAgentTags(dataAgentStringSlice(queryMap["tags"])),
		Descriptors:      normalizeDataAgentTags(dataAgentStringSlice(queryMap["descriptors"])),
		MetadataFilters:  dataAgentResourceMetadataFilters(queryMap["metadata_filters"]),
		CreatedAfter:     dataAgentTimeValue(queryMap["created_after"]),
		CreatedBefore:    dataAgentTimeValue(queryMap["created_before"]),
		ProcessingStatus: dataAgentStringValue(queryMap["processing_status"]),
	}
	if query.Query == "" {
		query.Query = dataAgentStringValue(queryMap["query"])
	}
	if query.ProjectID == "" {
		query.ProjectID = dataAgentStringValue(queryMap["projectId"])
	}
	if query.Query == "" &&
		query.Kind == "" &&
		query.Source == "" &&
		query.ProjectID == "" &&
		query.Sharing == "" &&
		len(query.Tags) == 0 &&
		len(query.Descriptors) == 0 &&
		len(query.MetadataFilters) == 0 &&
		query.CreatedAfter.IsZero() &&
		query.CreatedBefore.IsZero() &&
		strings.TrimSpace(query.ProcessingStatus) == "" {
		return nil
	}
	return &query
}

func dataAgentTimeValue(value any) time.Time {
	text := dataAgentStringValue(value)
	if text == "" {
		return time.Time{}
	}
	for _, layout := range []string{time.RFC3339Nano, time.RFC3339, "2006-01-02"} {
		if ts, err := time.Parse(layout, text); err == nil {
			return ts.UTC()
		}
	}
	return time.Time{}
}

func dataAgentResourceMetadataFilters(value any) []domain.ResourceMetadataFilter {
	items, ok := value.([]any)
	if !ok {
		return nil
	}
	filters := make([]domain.ResourceMetadataFilter, 0, len(items))
	for _, item := range items {
		record, ok := workerJSONMap(item)
		if !ok {
			continue
		}
		path := dataAgentStringValue(record["path"])
		operator := strings.ToLower(dataAgentStringValue(record["operator"]))
		value := dataAgentStringValue(record["value"])
		if path == "" || operator == "" || (operator != "exists" && value == "") {
			continue
		}
		filters = append(filters, domain.ResourceMetadataFilter{
			Path:     path,
			Operator: operator,
			Value:    value,
		})
	}
	return filters
}

func dataAgentDatasetSnapshotResourceQueryMetadata(query *domain.DatasetSnapshotResourceQuery) domain.JSONMap {
	if query == nil {
		return nil
	}
	metadata := domain.JSONMap{}
	if strings.TrimSpace(query.Query) != "" {
		metadata["q"] = strings.TrimSpace(query.Query)
	}
	if strings.TrimSpace(query.Kind) != "" {
		metadata["kind"] = strings.TrimSpace(query.Kind)
	}
	if strings.TrimSpace(query.Source) != "" {
		metadata["source"] = strings.TrimSpace(query.Source)
	}
	if strings.TrimSpace(query.ProjectID) != "" {
		metadata["project_id"] = strings.TrimSpace(query.ProjectID)
	}
	if strings.TrimSpace(query.Sharing) != "" {
		metadata["sharing"] = strings.TrimSpace(query.Sharing)
	}
	if len(query.Tags) > 0 {
		metadata["tags"] = append([]string(nil), query.Tags...)
	}
	if len(query.Descriptors) > 0 {
		metadata["descriptors"] = append([]string(nil), query.Descriptors...)
	}
	if len(query.MetadataFilters) > 0 {
		filters := make([]domain.JSONMap, 0, len(query.MetadataFilters))
		for _, filter := range query.MetadataFilters {
			filters = append(filters, domain.JSONMap{
				"path":     filter.Path,
				"operator": filter.Operator,
				"value":    filter.Value,
			})
		}
		metadata["metadata_filters"] = filters
	}
	if !query.CreatedAfter.IsZero() {
		metadata["created_after"] = query.CreatedAfter.UTC().Format(time.RFC3339Nano)
	}
	if !query.CreatedBefore.IsZero() {
		metadata["created_before"] = query.CreatedBefore.UTC().Format(time.RFC3339Nano)
	}
	if strings.TrimSpace(query.ProcessingStatus) != "" {
		metadata["processing_status"] = strings.TrimSpace(query.ProcessingStatus)
	}
	return metadata
}

func dataAgentJobString(envelope eventbus.DataAgentJob, job domain.DataAgentJobRecord, keys ...string) string {
	for _, key := range keys {
		for _, value := range []any{
			envelope.InputSelector[key],
			job.InputSelector[key],
			envelope.Metadata[key],
			job.Metadata[key],
		} {
			text := strings.TrimSpace(fmt.Sprint(value))
			if text != "" && text != "<nil>" {
				return text
			}
		}
	}
	return ""
}

func dataAgentStringValue(value any) string {
	text := strings.TrimSpace(fmt.Sprint(value))
	if text == "" || text == "<nil>" {
		return ""
	}
	return text
}

func dataAgentStringSlice(value any) []string {
	switch typed := value.(type) {
	case []string:
		return append([]string(nil), typed...)
	case []any:
		out := make([]string, 0, len(typed))
		for _, item := range typed {
			text := strings.TrimSpace(fmt.Sprint(item))
			if text != "" && text != "<nil>" {
				out = append(out, text)
			}
		}
		return out
	case string:
		return []string{typed}
	default:
		return nil
	}
}

func normalizeDataAgentTags(tags []string) []string {
	out := make([]string, 0, len(tags))
	seen := map[string]bool{}
	for _, tag := range tags {
		trimmed := strings.TrimSpace(tag)
		if trimmed == "" {
			continue
		}
		key := strings.ToLower(trimmed)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, trimmed)
	}
	return out
}

func dataAgentProgressTotal(envelope eventbus.DataAgentJob, job domain.DataAgentJobRecord, resources []domain.ResourceRecord) int {
	if job.ProgressTotal > 0 {
		return job.ProgressTotal
	}
	if envelope.ResourceCount > 0 {
		return envelope.ResourceCount
	}
	if job.ResourceCount > 0 {
		return job.ResourceCount
	}
	return len(resources)
}

func dataAgentResultResourceCount(result DataAgentJobResult) int {
	if result.OutputSummary == nil {
		return 0
	}
	for _, key := range []string{"resource_count", "tagged_resource_count"} {
		switch count := result.OutputSummary[key].(type) {
		case int:
			if count > 0 {
				return count
			}
		case int64:
			if count > 0 {
				return int(count)
			}
		case float64:
			if count > 0 {
				return int(count)
			}
		}
	}
	return 0
}

func dataAgentJobTerminal(status string) bool {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "succeeded", "failed", "canceled":
		return true
	default:
		return false
	}
}

func dataAgentSummaryText(jobType string, resources []domain.ResourceRecord) string {
	jobType = strings.TrimSpace(jobType)
	if jobType == "" {
		jobType = "data_agent_job"
	}
	if len(resources) == 0 {
		return "Prepared an empty " + jobType + " resource summary."
	}
	names := make([]string, 0, len(resources))
	for _, resource := range resources {
		name := strings.TrimSpace(resource.OriginalName)
		if name == "" {
			name = resource.ResourceID
		}
		names = append(names, name)
	}
	sort.Strings(names)
	return fmt.Sprintf("Prepared %s metadata summary for %d resources: %s.", jobType, len(resources), strings.Join(names, ", "))
}

func dataAgentTemplateDisplayName(jobType string) string {
	switch strings.ToLower(strings.TrimSpace(jobType)) {
	case "caption_resources":
		return "caption generation"
	case "extract_metadata":
		return "metadata extraction"
	case "organize_resources":
		return "organization plan"
	case "deduplicate_resources":
		return "duplicate detection"
	case "quality_check_resources":
		return "quality check"
	default:
		return "metadata summary"
	}
}

func dataAgentResourceCaption(resource domain.ResourceRecord) string {
	name := strings.TrimSpace(resource.OriginalName)
	if name == "" {
		name = resource.ResourceID
	}
	kind := strings.TrimSpace(resource.ResourceKind)
	if kind == "" {
		kind = "resource"
	}
	source := strings.TrimSpace(resource.SourceType)
	if source == "" {
		source = "unknown source"
	}
	contentType := strings.TrimSpace(resource.ContentType)
	if contentType == "" {
		contentType = "unspecified content type"
	}
	return fmt.Sprintf("%s is a %s from %s with %s content, %d bytes, and checksum %s.",
		name,
		kind,
		source,
		contentType,
		resource.SizeBytes,
		dataAgentChecksumLabel(resource.SHA256),
	)
}

func dataAgentChecksumLabel(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return "missing"
	}
	if len(value) > 12 {
		return value[:12]
	}
	return value
}

func dataAgentOrganizationReason(rule string, name string) string {
	switch rule {
	case "project":
		return "Groups resources that belong to the same project."
	case "resource_kind":
		return "Groups resources with the same catalog kind."
	case "source_type":
		return "Groups resources from the same acquisition or import source."
	case "file_extension":
		return "Groups resources with matching file extensions for batch tooling."
	case "metadata_label":
		return "Groups resources sharing the same metadata label."
	default:
		return "Groups resources by " + name + "."
	}
}

func dataAgentFileExtension(name string) string {
	normalized := strings.ToLower(strings.TrimSpace(name))
	if strings.HasSuffix(normalized, ".nii.gz") {
		return "nii.gz"
	}
	ext := strings.TrimPrefix(filepath.Ext(normalized), ".")
	if ext == "" {
		return "unknown"
	}
	return ext
}

func dataAgentQualityWarning(resource domain.ResourceRecord, code string, severity string, message string) domain.JSONMap {
	return domain.JSONMap{
		"resource_id":   resource.ResourceID,
		"original_name": resource.OriginalName,
		"code":          code,
		"severity":      severity,
		"message":       message,
	}
}

func dataAgentCountBy(resources []domain.ResourceRecord, keyFn func(domain.ResourceRecord) string) domain.JSONMap {
	counts := domain.JSONMap{}
	for _, resource := range resources {
		key := strings.TrimSpace(keyFn(resource))
		if key == "" {
			key = "unspecified"
		}
		current, _ := counts[key].(int)
		counts[key] = current + 1
	}
	return counts
}

func sortedWorkerSet(values map[string]struct{}) []string {
	out := make([]string, 0, len(values))
	for value := range values {
		if trimmed := strings.TrimSpace(value); trimmed != "" {
			out = append(out, trimmed)
		}
	}
	sort.Strings(out)
	return out
}

func mergeWorkerJSONMap(base domain.JSONMap, overlay domain.JSONMap) domain.JSONMap {
	out := cloneWorkerJSONMap(base)
	for key, value := range overlay {
		out[key] = value
	}
	return out
}

func uniqueWorkerStrings(values []string) []string {
	seen := map[string]struct{}{}
	out := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	return out
}

func cloneWorkerJSONMap(value domain.JSONMap) domain.JSONMap {
	out := domain.JSONMap{}
	for key, item := range value {
		out[key] = item
	}
	return out
}

func workerJSONMap(value any) (domain.JSONMap, bool) {
	switch typed := value.(type) {
	case domain.JSONMap:
		return typed, true
	case map[string]any:
		out := make(domain.JSONMap, len(typed))
		for key, item := range typed {
			out[key] = item
		}
		return out, true
	default:
		return nil, false
	}
}
