package worker

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

func TestDataAgentWorkerRunJobLeasesRenewsCompletesAndReleases(t *testing.T) {
	ctx := context.Background()
	mem := store.NewMemoryStore()
	rec := &recordingDataAgentWorkerStore{MemoryStore: mem}
	now := time.Date(2026, 6, 8, 16, 0, 0, 0, time.UTC)
	seedDataAgentWorkerResource(t, ctx, mem, "file_worker_a", "nph-a.nii.gz", "sha-a", now)
	seedDataAgentWorkerResource(t, ctx, mem, "file_worker_b", "nph-b.nii.gz", "sha-b", now.Add(time.Second))
	job := seedDataAgentWorkerJob(t, ctx, mem, "data_agent_job_worker", []string{"file_worker_a", "file_worker_b"}, now.Add(2*time.Second))

	processor := DataAgentProcessorFunc(func(ctx context.Context, work DataAgentWork) (DataAgentJobResult, error) {
		if work.Job.JobID != job.JobID || work.Job.Status != "running" {
			t.Fatalf("work job = %+v, want leased running job", work.Job)
		}
		if len(work.Resources) != 2 || work.Resources[0].OriginalName != "nph-a.nii.gz" || work.Resources[1].OriginalName != "nph-b.nii.gz" {
			t.Fatalf("work resources = %+v, want ordered envelope resources", work.Resources)
		}
		if err := work.ReportProgress(ctx, 1, "Captioned nph-a.nii.gz.", domain.JSONMap{"resource_id": "file_worker_a"}); err != nil {
			return DataAgentJobResult{}, err
		}
		time.Sleep(35 * time.Millisecond)
		return DataAgentJobResult{
			Message: "Caption summaries generated.",
			OutputSummary: domain.JSONMap{
				"summary":        "Generated calm metadata captions for two NIfTI files.",
				"resource_count": float64(2),
			},
			Metadata: domain.JSONMap{"processor": "unit-test"},
		}, nil
	})
	worker := NewDataAgentWorker(rec, DataAgentWorkerConfig{
		WorkerID:           "data-agent-worker-test",
		LeaseTTL:           100 * time.Millisecond,
		LeaseRenewInterval: 10 * time.Millisecond,
		Processor:          processor,
	})

	err := worker.RunJob(ctx, eventbus.DataAgentJob{
		JobID:         job.JobID,
		DispatchID:    "dispatch-worker-test",
		OwnerUserID:   "alice",
		OwnerOrgID:    "org-a",
		ProjectID:     "nph-study",
		JobType:       "caption_resources",
		ResourceIDs:   []string{"file_worker_a", "file_worker_b"},
		ResourceCount: 2,
	})
	if err != nil {
		t.Fatalf("RunJob: %v", err)
	}
	if rec.renewCount == 0 {
		t.Fatalf("renewCount = 0, want worker to renew lease while processor is active")
	}
	if rec.releaseCount != 1 {
		t.Fatalf("releaseCount = %d, want one release after terminal update", rec.releaseCount)
	}
	loaded, err := mem.GetDataAgentJobForUser(ctx, job.JobID, "alice", "org-a")
	if err != nil {
		t.Fatalf("GetDataAgentJobForUser: %v", err)
	}
	if loaded.Status != "succeeded" || loaded.ProgressCompleted != 2 || loaded.ProgressTotal != 2 || loaded.CompletedAt.IsZero() {
		t.Fatalf("loaded job = %+v, want terminal succeeded progress", loaded)
	}
	if loaded.OutputSummary["summary"] != "Generated calm metadata captions for two NIfTI files." {
		t.Fatalf("output summary = %+v, want processor summary", loaded.OutputSummary)
	}
	events, err := mem.ListDataAgentJobEvents(ctx, job.JobID, "alice", "org-a", 20)
	if err != nil {
		t.Fatalf("ListDataAgentJobEvents: %v", err)
	}
	gotTypes := make([]string, 0, len(events))
	for _, event := range events {
		gotTypes = append(gotTypes, event.EventType)
	}
	wantTypes := []string{
		"data_agent.job.created",
		"data_agent.job.leased",
		"data_agent.job.progressed",
		"data_agent.job.completed",
	}
	if !sameStringSlice(gotTypes, wantTypes) {
		t.Fatalf("event types = %+v, want %+v", gotTypes, wantTypes)
	}
}

func TestDataAgentWorkerLoadsResourcesFromQuerySelector(t *testing.T) {
	ctx := context.Background()
	mem := store.NewMemoryStore()
	now := time.Date(2026, 6, 8, 16, 5, 0, 0, time.UTC)
	for _, input := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_worker_query_caption_a",
			OriginalName: "NPH_001_68yo.nii.gz",
			ContentType:  "application/x-nifti",
			SizeBytes:    128,
			SHA256:       "sha-query-caption-a",
			ResourceKind: "file",
			SourceType:   "upload",
			ProjectID:    "nph-study",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now,
			UpdatedAt:    now,
			Tags:         []string{"NPH", "Under 70"},
			Metadata:     domain.JSONMap{"label": "NPH", "age": 68},
		},
		{
			ResourceID:   "file_worker_query_caption_b",
			OriginalName: "NPH_002_64yo.nii.gz",
			ContentType:  "application/x-nifti",
			SizeBytes:    256,
			SHA256:       "sha-query-caption-b",
			ResourceKind: "file",
			SourceType:   "upload",
			ProjectID:    "nph-study",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			UpdatedAt:    now.Add(time.Second),
			Tags:         []string{"NPH", "Under 70"},
			Metadata:     domain.JSONMap{"label": "NPH", "age": 64},
		},
		{
			ResourceID:   "file_worker_query_caption_over70",
			OriginalName: "NPH_003_74yo.nii.gz",
			ContentType:  "application/x-nifti",
			SizeBytes:    512,
			SHA256:       "sha-query-caption-over70",
			ResourceKind: "file",
			SourceType:   "upload",
			ProjectID:    "nph-study",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now.Add(2 * time.Second),
			UpdatedAt:    now.Add(2 * time.Second),
			Tags:         []string{"NPH", "Over 70"},
			Metadata:     domain.JSONMap{"label": "NPH", "age": 74},
		},
	} {
		if _, err := mem.UpsertResource(ctx, input); err != nil {
			t.Fatalf("UpsertResource(%s): %v", input.ResourceID, err)
		}
	}
	inputSelector := domain.JSONMap{
		"resource_query": domain.JSONMap{
			"q":          "NPH",
			"kind":       "file",
			"source":     "upload",
			"project_id": "nph-study",
			"tags":       []any{"Under 70"},
		},
	}
	job, err := mem.CreateDataAgentJob(ctx, domain.CreateDataAgentJobInput{
		JobID:           "data_agent_job_query_caption",
		OwnerUserID:     "alice",
		OwnerOrgID:      "org-a",
		ProjectID:       "nph-study",
		JobType:         "caption_resources",
		InputSelector:   inputSelector,
		CreatedByUserID: "alice",
		CreatedAt:       now.Add(3 * time.Second),
		UpdatedAt:       now.Add(3 * time.Second),
	})
	if err != nil {
		t.Fatalf("CreateDataAgentJob: %v", err)
	}
	processor := DataAgentProcessorFunc(func(ctx context.Context, work DataAgentWork) (DataAgentJobResult, error) {
		gotIDs := make([]string, 0, len(work.Resources))
		for _, resource := range work.Resources {
			gotIDs = append(gotIDs, resource.ResourceID)
		}
		if !sameStringSlice(gotIDs, []string{"file_worker_query_caption_b", "file_worker_query_caption_a"}) {
			t.Fatalf("work resources = %+v, want newest matching query resources only", gotIDs)
		}
		if err := work.ReportProgress(ctx, len(work.Resources), "Captioned query cohort.", domain.JSONMap{"resource_count": len(work.Resources)}); err != nil {
			return DataAgentJobResult{}, err
		}
		return DataAgentJobResult{
			Message: "Captioned query cohort.",
			OutputSummary: domain.JSONMap{
				"summary_kind":   "caption_generation",
				"resource_count": len(work.Resources),
				"resources":      dataAgentResourceSummaries(work.Resources),
			},
			Metadata: domain.JSONMap{"processor": "query-unit-test"},
		}, nil
	})
	worker := NewDataAgentWorker(mem, DataAgentWorkerConfig{
		WorkerID:  "data-agent-worker-test",
		Processor: processor,
		Now:       func() time.Time { return now.Add(2 * time.Minute) },
	})

	if err := worker.RunJob(ctx, eventbus.DataAgentJob{
		JobID:         job.JobID,
		OwnerUserID:   "alice",
		OwnerOrgID:    "org-a",
		ProjectID:     "nph-study",
		JobType:       "caption_resources",
		InputSelector: inputSelector,
	}); err != nil {
		t.Fatalf("RunJob: %v", err)
	}
	loaded, err := mem.GetDataAgentJobForUser(ctx, job.JobID, "alice", "org-a")
	if err != nil {
		t.Fatalf("GetDataAgentJobForUser: %v", err)
	}
	if loaded.Status != "succeeded" || loaded.ProgressTotal != 2 || loaded.ProgressCompleted != 2 {
		t.Fatalf("loaded job = %+v, want succeeded query job with resolved progress", loaded)
	}
}

func TestDataAgentWorkerLoadsQueryResourcesAcrossPages(t *testing.T) {
	ctx := context.Background()
	mem := store.NewMemoryStore()
	paged := &pagingDataAgentWorkerStore{MemoryStore: mem, pageSize: 2}
	now := time.Date(2026, 6, 8, 16, 10, 0, 0, time.UTC)
	for index, resourceID := range []string{
		"file_worker_paged_query_a",
		"file_worker_paged_query_b",
		"file_worker_paged_query_c",
	} {
		if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
			ResourceID:   resourceID,
			OriginalName: "NPH_paged_query.nii.gz",
			ContentType:  "application/x-nifti",
			SizeBytes:    int64(128 + index),
			SHA256:       "sha-paged-query-" + resourceID,
			ResourceKind: "file",
			SourceType:   "upload",
			ProjectID:    "nph-study",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now.Add(time.Duration(index) * time.Second),
			UpdatedAt:    now.Add(time.Duration(index) * time.Second),
			Tags:         []string{"NPH", "Under 70"},
			Metadata:     domain.JSONMap{"label": "NPH", "batch": "paged-query"},
		}); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resourceID, err)
		}
	}
	inputSelector := domain.JSONMap{
		"resource_query": domain.JSONMap{
			"q":          "NPH",
			"kind":       "file",
			"source":     "upload",
			"project_id": "nph-study",
			"tags":       []any{"Under 70"},
		},
	}
	job, err := mem.CreateDataAgentJob(ctx, domain.CreateDataAgentJobInput{
		JobID:           "data_agent_job_paged_query",
		OwnerUserID:     "alice",
		OwnerOrgID:      "org-a",
		ProjectID:       "nph-study",
		JobType:         "caption_resources",
		ResourceCount:   3,
		InputSelector:   inputSelector,
		CreatedByUserID: "alice",
		CreatedAt:       now.Add(3 * time.Second),
		UpdatedAt:       now.Add(3 * time.Second),
	})
	if err != nil {
		t.Fatalf("CreateDataAgentJob: %v", err)
	}
	processor := DataAgentProcessorFunc(func(ctx context.Context, work DataAgentWork) (DataAgentJobResult, error) {
		if len(work.Resources) != 3 {
			t.Fatalf("work resources = %+v, want all three paged query resources", work.Resources)
		}
		if err := work.ReportProgress(ctx, len(work.Resources), "Captioned paged query cohort.", domain.JSONMap{"resource_count": len(work.Resources)}); err != nil {
			return DataAgentJobResult{}, err
		}
		return DataAgentJobResult{
			Message: "Captioned paged query cohort.",
			OutputSummary: domain.JSONMap{
				"summary_kind":   "caption_generation",
				"resource_count": len(work.Resources),
			},
		}, nil
	})
	worker := NewDataAgentWorker(paged, DataAgentWorkerConfig{
		WorkerID:  "data-agent-worker-test",
		Processor: processor,
		Now:       func() time.Time { return now.Add(2 * time.Minute) },
	})

	if err := worker.RunJob(ctx, eventbus.DataAgentJob{
		JobID:         job.JobID,
		OwnerUserID:   "alice",
		OwnerOrgID:    "org-a",
		ProjectID:     "nph-study",
		JobType:       "caption_resources",
		ResourceCount: 3,
		InputSelector: inputSelector,
	}); err != nil {
		t.Fatalf("RunJob: %v", err)
	}
	if len(paged.listCalls) < 2 {
		t.Fatalf("query list calls = %+v, want worker to page beyond first partial result set", paged.listCalls)
	}
	if paged.listCalls[0].Offset != 0 || paged.listCalls[1].Offset != 2 {
		t.Fatalf("query list offsets = %+v, want offsets 0 then 2", paged.listCalls)
	}
	loaded, err := mem.GetDataAgentJobForUser(ctx, job.JobID, "alice", "org-a")
	if err != nil {
		t.Fatalf("GetDataAgentJobForUser: %v", err)
	}
	if loaded.Status != "succeeded" || loaded.ProgressTotal != 3 || loaded.ProgressCompleted != 3 {
		t.Fatalf("loaded job = %+v, want succeeded paged query job with full progress", loaded)
	}
}

func TestDataAgentWorkerPersistsResourceProcessingStatus(t *testing.T) {
	ctx := context.Background()
	mem := store.NewMemoryStore()
	now := time.Date(2026, 6, 8, 16, 15, 0, 0, time.UTC)
	seedDataAgentWorkerResource(t, ctx, mem, "file_worker_caption", "NPH_001_68yo.nii.gz", "sha-caption", now)
	job := seedDataAgentWorkerJob(t, ctx, mem, "data_agent_job_resource_status", []string{"file_worker_caption"}, now.Add(time.Second))

	worker := NewDataAgentWorker(mem, DataAgentWorkerConfig{
		WorkerID: "data-agent-worker-test",
		Now:      func() time.Time { return now.Add(2 * time.Minute) },
	})

	if err := worker.RunJob(ctx, eventbus.DataAgentJob{
		JobID:         job.JobID,
		OwnerUserID:   "alice",
		OwnerOrgID:    "org-a",
		ProjectID:     "nph-study",
		JobType:       "caption_resources",
		ResourceIDs:   []string{"file_worker_caption"},
		ResourceCount: 1,
	}); err != nil {
		t.Fatalf("RunJob: %v", err)
	}
	resource, err := mem.GetResourceForUser(ctx, "file_worker_caption", "alice", "org-a")
	if err != nil {
		t.Fatalf("GetResourceForUser: %v", err)
	}
	dataAgentMetadata, ok := resource.Metadata["data_agent"].(domain.JSONMap)
	if !ok {
		t.Fatalf("resource metadata = %+v, want data_agent block", resource.Metadata)
	}
	captionStatus, ok := dataAgentMetadata["caption_resources"].(domain.JSONMap)
	if !ok {
		t.Fatalf("data_agent metadata = %+v, want caption_resources status", dataAgentMetadata)
	}
	if captionStatus["status"] != "succeeded" || captionStatus["job_id"] != job.JobID || captionStatus["summary_kind"] != "caption_generation" {
		t.Fatalf("caption status = %+v, want succeeded caption job metadata", captionStatus)
	}
	if captionStatus["caption"] == "" || captionStatus["caption_source"] != "deterministic_metadata" {
		t.Fatalf("caption status = %+v, want persisted resource caption", captionStatus)
	}
	if captionStatus["completed_at"] == "" || captionStatus["updated_at"] == "" {
		t.Fatalf("caption status = %+v, want audit timestamps", captionStatus)
	}
	if resource.Metadata["label"] != "NPH" {
		t.Fatalf("resource metadata = %+v, want existing metadata preserved", resource.Metadata)
	}
	events, err := mem.ListResourceEvents(ctx, "file_worker_caption", 20)
	if err != nil {
		t.Fatalf("ListResourceEvents: %v", err)
	}
	if len(events) == 0 || events[len(events)-1].EventType != "resource.data_agent_job_completed" {
		t.Fatalf("resource events = %+v, want completion event", events)
	}
	if events[len(events)-1].Metadata["job_id"] != job.JobID || events[len(events)-1].Metadata["job_type"] != "caption_resources" {
		t.Fatalf("resource completion event = %+v, want job metadata", events[len(events)-1])
	}
}

func TestDataAgentWorkerBatchTagResourcesAppliesTagsAndAudits(t *testing.T) {
	ctx := context.Background()
	mem := store.NewMemoryStore()
	now := time.Date(2026, 6, 8, 16, 25, 0, 0, time.UTC)
	for _, input := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_worker_batch_tag_a",
			OriginalName: "NPH_001_68yo.nii.gz",
			ContentType:  "application/x-nifti",
			SizeBytes:    128,
			SHA256:       "sha-batch-tag-a",
			ResourceKind: "file",
			SourceType:   "upload",
			ProjectID:    "nph-study",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now,
			UpdatedAt:    now,
			Tags:         []string{"raw"},
			Metadata:     domain.JSONMap{"label": "NPH"},
		},
		{
			ResourceID:   "file_worker_batch_tag_b",
			OriginalName: "NPH_002_72yo.nii.gz",
			ContentType:  "application/x-nifti",
			SizeBytes:    256,
			SHA256:       "sha-batch-tag-b",
			ResourceKind: "file",
			SourceType:   "upload",
			ProjectID:    "nph-study",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			UpdatedAt:    now.Add(time.Second),
			Metadata:     domain.JSONMap{"label": "NPH"},
		},
	} {
		if _, err := mem.UpsertResource(ctx, input); err != nil {
			t.Fatalf("UpsertResource(%s): %v", input.ResourceID, err)
		}
	}
	resourceIDs := []string{"file_worker_batch_tag_a", "file_worker_batch_tag_b"}
	job, err := mem.CreateDataAgentJob(ctx, domain.CreateDataAgentJobInput{
		JobID:           "data_agent_job_batch_tag",
		OwnerUserID:     "alice",
		OwnerOrgID:      "org-a",
		ProjectID:       "nph-study",
		JobType:         "batch_tag_resources",
		ResourceIDs:     resourceIDs,
		InputSelector:   domain.JSONMap{"resource_ids": stringSliceToAny(resourceIDs), "tags": []any{"NPH", "Under 70", "NPH"}},
		CreatedByUserID: "alice",
		CreatedAt:       now.Add(3 * time.Second),
		UpdatedAt:       now.Add(3 * time.Second),
	})
	if err != nil {
		t.Fatalf("CreateDataAgentJob: %v", err)
	}
	worker := NewDataAgentWorker(mem, DataAgentWorkerConfig{
		WorkerID: "data-agent-worker-test",
		Now:      func() time.Time { return now.Add(2 * time.Minute) },
	})

	if err := worker.RunJob(ctx, eventbus.DataAgentJob{
		JobID:         job.JobID,
		OwnerUserID:   "alice",
		OwnerOrgID:    "org-a",
		ProjectID:     "nph-study",
		JobType:       "batch_tag_resources",
		ResourceIDs:   resourceIDs,
		ResourceCount: len(resourceIDs),
		InputSelector: domain.JSONMap{"resource_ids": stringSliceToAny(resourceIDs), "tags": []any{"NPH", "Under 70", "NPH"}},
	}); err != nil {
		t.Fatalf("RunJob: %v", err)
	}
	taggedA, err := mem.GetResourceForUser(ctx, "file_worker_batch_tag_a", "alice", "org-a")
	if err != nil {
		t.Fatalf("GetResourceForUser A: %v", err)
	}
	if !sameStringSlice(taggedA.Tags, []string{"raw", "NPH", "Under 70"}) {
		t.Fatalf("resource A tags = %#v, want existing tag plus normalized job tags", taggedA.Tags)
	}
	taggedB, err := mem.GetResourceForUser(ctx, "file_worker_batch_tag_b", "alice", "org-a")
	if err != nil {
		t.Fatalf("GetResourceForUser B: %v", err)
	}
	if !sameStringSlice(taggedB.Tags, []string{"NPH", "Under 70"}) {
		t.Fatalf("resource B tags = %#v, want normalized job tags", taggedB.Tags)
	}
	loaded, err := mem.GetDataAgentJobForUser(ctx, job.JobID, "alice", "org-a")
	if err != nil {
		t.Fatalf("GetDataAgentJobForUser: %v", err)
	}
	if loaded.Status != "succeeded" || loaded.OutputSummary["summary_kind"] != "batch_tagging" || loaded.OutputSummary["tagged_resource_count"] != len(resourceIDs) {
		t.Fatalf("loaded job = %+v, want succeeded batch tag summary", loaded)
	}
	events, err := mem.ListResourceEvents(ctx, "file_worker_batch_tag_a", 20)
	if err != nil {
		t.Fatalf("ListResourceEvents: %v", err)
	}
	gotTypes := make([]string, 0, len(events))
	taggedEventSeen := false
	completedEventSeen := false
	for _, event := range events {
		gotTypes = append(gotTypes, event.EventType)
		if event.EventType == "resource.tagged" {
			taggedEventSeen = true
			if tags, ok := event.Metadata["tags_added"].([]string); !ok || !sameStringSlice(tags, []string{"NPH", "Under 70"}) {
				t.Fatalf("tagged event = %+v, want normalized tags_added metadata", event)
			}
		}
		if event.EventType == "resource.data_agent_job_completed" {
			completedEventSeen = true
		}
	}
	if !taggedEventSeen || !completedEventSeen {
		t.Fatalf("resource events = %+v, want tag and Data Agent completion audit events", gotTypes)
	}
}

func TestDataAgentWorkerCreateDatasetSnapshotFreezesResourcesAndAudits(t *testing.T) {
	ctx := context.Background()
	mem := store.NewMemoryStore()
	now := time.Date(2026, 6, 8, 16, 28, 0, 0, time.UTC)
	seedDataAgentWorkerResource(t, ctx, mem, "file_worker_snapshot_a", "NPH_001_68yo.nii.gz", "sha-snapshot-a", now)
	seedDataAgentWorkerResource(t, ctx, mem, "file_worker_snapshot_b", "NPH_002_72yo.nii.gz", "sha-snapshot-b", now.Add(time.Second))
	resourceIDs := []string{"file_worker_snapshot_a", "file_worker_snapshot_b"}
	job, err := mem.CreateDataAgentJob(ctx, domain.CreateDataAgentJobInput{
		JobID:           "data_agent_job_dataset_snapshot",
		OwnerUserID:     "alice",
		OwnerOrgID:      "org-a",
		ProjectID:       "nph-study",
		JobType:         "create_dataset_snapshot",
		ResourceIDs:     resourceIDs,
		InputSelector:   domain.JSONMap{"resource_ids": stringSliceToAny(resourceIDs), "snapshot_name": "NPH under-70 training cohort", "source_collection_id": "collection_nph_under_70"},
		CreatedByUserID: "alice",
		CreatedAt:       now.Add(3 * time.Second),
		UpdatedAt:       now.Add(3 * time.Second),
	})
	if err != nil {
		t.Fatalf("CreateDataAgentJob: %v", err)
	}
	worker := NewDataAgentWorker(mem, DataAgentWorkerConfig{
		WorkerID: "data-agent-worker-test",
		Now:      func() time.Time { return now.Add(2 * time.Minute) },
	})

	if err := worker.RunJob(ctx, eventbus.DataAgentJob{
		JobID:         job.JobID,
		OwnerUserID:   "alice",
		OwnerOrgID:    "org-a",
		ProjectID:     "nph-study",
		JobType:       "create_dataset_snapshot",
		ResourceIDs:   resourceIDs,
		ResourceCount: len(resourceIDs),
		InputSelector: domain.JSONMap{"resource_ids": stringSliceToAny(resourceIDs), "snapshot_name": "NPH under-70 training cohort", "source_collection_id": "collection_nph_under_70"},
	}); err != nil {
		t.Fatalf("RunJob: %v", err)
	}
	loaded, err := mem.GetDataAgentJobForUser(ctx, job.JobID, "alice", "org-a")
	if err != nil {
		t.Fatalf("GetDataAgentJobForUser: %v", err)
	}
	if loaded.Status != "succeeded" || loaded.OutputSummary["summary_kind"] != "dataset_snapshot" || loaded.OutputSummary["snapshot_name"] != "NPH under-70 training cohort" {
		t.Fatalf("loaded job = %+v, want succeeded dataset snapshot summary", loaded)
	}
	snapshotID, ok := loaded.OutputSummary["snapshot_id"].(string)
	if !ok || snapshotID == "" {
		t.Fatalf("output summary = %+v, want snapshot_id", loaded.OutputSummary)
	}
	snapshot, entries, err := mem.GetDatasetSnapshotForUser(ctx, snapshotID, "alice", "org-a")
	if err != nil {
		t.Fatalf("GetDatasetSnapshotForUser: %v", err)
	}
	if snapshot.Name != "NPH under-70 training cohort" || snapshot.SourceCollectionID != "collection_nph_under_70" || snapshot.ResourceCount != 2 || snapshot.TotalBytes != 256 {
		t.Fatalf("snapshot = %+v, want named frozen two-resource dataset", snapshot)
	}
	if len(entries) != 2 || entries[0].ResourceID != "file_worker_snapshot_a" || entries[1].ResourceID != "file_worker_snapshot_b" {
		t.Fatalf("snapshot entries = %+v, want resources in job order", entries)
	}
	if snapshot.Metadata["job_id"] != job.JobID || snapshot.Metadata["job_type"] != "create_dataset_snapshot" {
		t.Fatalf("snapshot metadata = %+v, want Data Agent provenance", snapshot.Metadata)
	}
	events, err := mem.ListResourceEvents(ctx, "file_worker_snapshot_a", 20)
	if err != nil {
		t.Fatalf("ListResourceEvents: %v", err)
	}
	snapshottedEventSeen := false
	completedEventSeen := false
	for _, event := range events {
		switch event.EventType {
		case "resource.dataset_snapshotted":
			snapshottedEventSeen = true
			if event.Metadata["snapshot_id"] != snapshotID || event.Metadata["job_id"] != job.JobID || event.Metadata["job_type"] != "create_dataset_snapshot" {
				t.Fatalf("snapshot event = %+v, want snapshot and job provenance", event)
			}
		case "resource.data_agent_job_completed":
			completedEventSeen = true
		}
	}
	if !snapshottedEventSeen || !completedEventSeen {
		t.Fatalf("resource events = %+v, want dataset snapshot and Data Agent completion audit events", events)
	}
}

func TestDataAgentWorkerCreateDatasetSnapshotFromResourceQuery(t *testing.T) {
	ctx := context.Background()
	mem := store.NewMemoryStore()
	now := time.Date(2026, 6, 8, 16, 29, 0, 0, time.UTC)
	for _, input := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_worker_query_snapshot_a",
			OriginalName: "NPH_001_68yo.nii.gz",
			ContentType:  "application/x-nifti",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    128,
			SHA256:       "sha-query-snapshot-a",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			ProjectID:    "nph-study",
			Status:       "active",
			CreatedAt:    now,
			UpdatedAt:    now,
			Tags:         []string{"NPH", "Under 70"},
			Metadata:     domain.JSONMap{"label": "NPH", "age": 68},
		},
		{
			ResourceID:   "file_worker_query_snapshot_b",
			OriginalName: "NPH_002_64yo.nii.gz",
			ContentType:  "application/x-nifti",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    256,
			SHA256:       "sha-query-snapshot-b",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			ProjectID:    "nph-study",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			UpdatedAt:    now.Add(time.Second),
			Tags:         []string{"NPH", "Under 70"},
			Metadata:     domain.JSONMap{"label": "NPH", "age": 64},
		},
		{
			ResourceID:   "file_worker_query_snapshot_over70",
			OriginalName: "NPH_003_74yo.nii.gz",
			ContentType:  "application/x-nifti",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    512,
			SHA256:       "sha-query-snapshot-over70",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			ProjectID:    "nph-study",
			Status:       "active",
			CreatedAt:    now.Add(2 * time.Second),
			UpdatedAt:    now.Add(2 * time.Second),
			Tags:         []string{"NPH", "Over 70"},
			Metadata:     domain.JSONMap{"label": "NPH", "age": 74},
		},
	} {
		if _, err := mem.UpsertResource(ctx, input); err != nil {
			t.Fatalf("UpsertResource(%s): %v", input.ResourceID, err)
		}
	}
	inputSelector := domain.JSONMap{
		"snapshot_name": "NPH under-70 query cohort",
		"resource_query": domain.JSONMap{
			"q":          "NPH",
			"kind":       "file",
			"source":     "upload",
			"project_id": "nph-study",
			"tags":       []any{"Under 70"},
		},
	}
	job, err := mem.CreateDataAgentJob(ctx, domain.CreateDataAgentJobInput{
		JobID:           "data_agent_job_query_dataset_snapshot",
		OwnerUserID:     "alice",
		OwnerOrgID:      "org-a",
		ProjectID:       "nph-study",
		JobType:         "create_dataset_snapshot",
		InputSelector:   inputSelector,
		CreatedByUserID: "alice",
		CreatedAt:       now.Add(3 * time.Second),
		UpdatedAt:       now.Add(3 * time.Second),
	})
	if err != nil {
		t.Fatalf("CreateDataAgentJob: %v", err)
	}
	worker := NewDataAgentWorker(mem, DataAgentWorkerConfig{
		WorkerID: "data-agent-worker-test",
		Now:      func() time.Time { return now.Add(2 * time.Minute) },
	})

	if err := worker.RunJob(ctx, eventbus.DataAgentJob{
		JobID:         job.JobID,
		OwnerUserID:   "alice",
		OwnerOrgID:    "org-a",
		ProjectID:     "nph-study",
		JobType:       "create_dataset_snapshot",
		InputSelector: inputSelector,
	}); err != nil {
		t.Fatalf("RunJob: %v", err)
	}
	loaded, err := mem.GetDataAgentJobForUser(ctx, job.JobID, "alice", "org-a")
	if err != nil {
		t.Fatalf("GetDataAgentJobForUser: %v", err)
	}
	if loaded.Status != "succeeded" || loaded.OutputSummary["summary_kind"] != "dataset_snapshot" {
		t.Fatalf("loaded job = %+v, want succeeded dataset snapshot query job", loaded)
	}
	snapshotID, ok := loaded.OutputSummary["snapshot_id"].(string)
	if !ok || snapshotID == "" {
		t.Fatalf("output summary = %+v, want snapshot_id", loaded.OutputSummary)
	}
	snapshot, entries, err := mem.GetDatasetSnapshotForUser(ctx, snapshotID, "alice", "org-a")
	if err != nil {
		t.Fatalf("GetDatasetSnapshotForUser: %v", err)
	}
	if snapshot.Name != "NPH under-70 query cohort" || snapshot.ResourceCount != 2 || snapshot.TotalBytes != 384 {
		t.Fatalf("snapshot = %+v, want two-resource query dataset", snapshot)
	}
	gotIDs := []string{entries[0].ResourceID, entries[1].ResourceID}
	if !sameStringSlice(gotIDs, []string{"file_worker_query_snapshot_b", "file_worker_query_snapshot_a"}) {
		t.Fatalf("snapshot entries = %+v, want newest matching query resources only", entries)
	}
	if gotSummaryIDs, ok := loaded.OutputSummary["snapshot_resource_ids"].([]string); !ok || !sameStringSlice(gotSummaryIDs, gotIDs) {
		t.Fatalf("output summary = %+v, want frozen query resource ids", loaded.OutputSummary)
	}
	events, err := mem.ListResourceEvents(ctx, "file_worker_query_snapshot_b", 20)
	if err != nil {
		t.Fatalf("ListResourceEvents: %v", err)
	}
	snapshottedEventSeen := false
	for _, event := range events {
		if event.EventType == "resource.dataset_snapshotted" && event.Metadata["snapshot_id"] == snapshotID && event.Metadata["job_id"] == job.JobID {
			snapshottedEventSeen = true
		}
	}
	if !snapshottedEventSeen {
		t.Fatalf("resource events = %+v, want query snapshot audit event", events)
	}
}

func TestDataAgentWorkerMarksFailedAndReleasesLease(t *testing.T) {
	ctx := context.Background()
	mem := store.NewMemoryStore()
	rec := &recordingDataAgentWorkerStore{MemoryStore: mem}
	now := time.Date(2026, 6, 8, 16, 30, 0, 0, time.UTC)
	seedDataAgentWorkerResource(t, ctx, mem, "file_worker_fail", "corrupt.nii.gz", "sha-fail", now)
	job := seedDataAgentWorkerJob(t, ctx, mem, "data_agent_job_worker_fail", []string{"file_worker_fail"}, now.Add(time.Second))
	expectedErr := errors.New("metadata extractor rejected corrupt header")
	worker := NewDataAgentWorker(rec, DataAgentWorkerConfig{
		WorkerID:           "data-agent-worker-test",
		LeaseTTL:           time.Minute,
		LeaseRenewInterval: time.Hour,
		Processor: DataAgentProcessorFunc(func(ctx context.Context, work DataAgentWork) (DataAgentJobResult, error) {
			return DataAgentJobResult{}, expectedErr
		}),
	})

	if err := worker.RunJob(ctx, eventbus.DataAgentJob{
		JobID:         job.JobID,
		OwnerUserID:   "alice",
		OwnerOrgID:    "org-a",
		JobType:       "extract_metadata",
		ResourceIDs:   []string{"file_worker_fail"},
		ResourceCount: 1,
	}); err != nil {
		t.Fatalf("RunJob returns %v, want terminal job failure to be ack-safe", err)
	}
	if rec.releaseCount != 1 {
		t.Fatalf("releaseCount = %d, want release after failed terminal update", rec.releaseCount)
	}
	loaded, err := mem.GetDataAgentJobForUser(ctx, job.JobID, "alice", "org-a")
	if err != nil {
		t.Fatalf("GetDataAgentJobForUser: %v", err)
	}
	if loaded.Status != "failed" || loaded.Error != expectedErr.Error() || loaded.CompletedAt.IsZero() {
		t.Fatalf("loaded failed job = %+v, want failed terminal error", loaded)
	}
	events, err := mem.ListDataAgentJobEvents(ctx, job.JobID, "alice", "org-a", 20)
	if err != nil {
		t.Fatalf("ListDataAgentJobEvents: %v", err)
	}
	if events[len(events)-1].EventType != "data_agent.job.failed" {
		t.Fatalf("last event = %+v, want data_agent.job.failed", events[len(events)-1])
	}
}

func TestDataAgentWorkerDoesNotFailJobWhenContextCanceled(t *testing.T) {
	parent := context.Background()
	ctx, cancel := context.WithCancel(parent)
	mem := store.NewMemoryStore()
	rec := &recordingDataAgentWorkerStore{MemoryStore: mem}
	now := time.Date(2026, 6, 8, 16, 45, 0, 0, time.UTC)
	seedDataAgentWorkerResource(t, parent, mem, "file_worker_shutdown", "shutdown.nii.gz", "sha-shutdown", now)
	job := seedDataAgentWorkerJob(t, parent, mem, "data_agent_job_worker_shutdown", []string{"file_worker_shutdown"}, now.Add(time.Second))
	worker := NewDataAgentWorker(rec, DataAgentWorkerConfig{
		WorkerID:           "data-agent-worker-test",
		LeaseTTL:           time.Minute,
		LeaseRenewInterval: time.Hour,
		Processor: DataAgentProcessorFunc(func(ctx context.Context, work DataAgentWork) (DataAgentJobResult, error) {
			cancel()
			<-ctx.Done()
			return DataAgentJobResult{}, ctx.Err()
		}),
	})

	err := worker.RunJob(ctx, eventbus.DataAgentJob{
		JobID:         job.JobID,
		OwnerUserID:   "alice",
		OwnerOrgID:    "org-a",
		JobType:       "extract_metadata",
		ResourceIDs:   []string{"file_worker_shutdown"},
		ResourceCount: 1,
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("RunJob err = %v, want context.Canceled", err)
	}
	if rec.releaseCount != 1 {
		t.Fatalf("releaseCount = %d, want release on shutdown without terminal failure", rec.releaseCount)
	}
	loaded, err := mem.GetDataAgentJobForUser(parent, job.JobID, "alice", "org-a")
	if err != nil {
		t.Fatalf("GetDataAgentJobForUser: %v", err)
	}
	if loaded.Status != "running" || loaded.Error != "" || !loaded.CompletedAt.IsZero() {
		t.Fatalf("loaded job after shutdown = %+v, want non-terminal running job without false failure", loaded)
	}
	events, err := mem.ListDataAgentJobEvents(parent, job.JobID, "alice", "org-a", 20)
	if err != nil {
		t.Fatalf("ListDataAgentJobEvents: %v", err)
	}
	if events[len(events)-1].EventType != "data_agent.job.leased" {
		t.Fatalf("last event = %+v, want no failed event on context cancellation", events[len(events)-1])
	}
}

func TestDataAgentWorkerSkipsTerminalStaleDelivery(t *testing.T) {
	ctx := context.Background()
	mem := store.NewMemoryStore()
	now := time.Date(2026, 6, 8, 17, 0, 0, 0, time.UTC)
	seedDataAgentWorkerResource(t, ctx, mem, "file_worker_canceled", "canceled.nii.gz", "sha-canceled", now)
	job := seedDataAgentWorkerJob(t, ctx, mem, "data_agent_job_worker_canceled", []string{"file_worker_canceled"}, now.Add(time.Second))
	if _, _, err := mem.ControlDataAgentJob(ctx, domain.ControlDataAgentJobInput{
		JobID:       job.JobID,
		OwnerUserID: "alice",
		OwnerOrgID:  "org-a",
		Action:      "cancel",
		Reason:      "Scientist canceled the batch.",
		ActorUserID: "alice",
		ActorOrgID:  "org-a",
		TS:          now.Add(2 * time.Second),
	}); err != nil {
		t.Fatalf("ControlDataAgentJob cancel: %v", err)
	}
	calls := 0
	worker := NewDataAgentWorker(mem, DataAgentWorkerConfig{
		WorkerID: "data-agent-worker-test",
		Processor: DataAgentProcessorFunc(func(ctx context.Context, work DataAgentWork) (DataAgentJobResult, error) {
			calls++
			return DataAgentJobResult{}, nil
		}),
	})

	if err := worker.RunJob(ctx, eventbus.DataAgentJob{
		JobID:         job.JobID,
		OwnerUserID:   "alice",
		OwnerOrgID:    "org-a",
		JobType:       "caption_resources",
		ResourceIDs:   []string{"file_worker_canceled"},
		ResourceCount: 1,
	}); err != nil {
		t.Fatalf("RunJob terminal stale delivery: %v", err)
	}
	if calls != 0 {
		t.Fatalf("processor calls = %d, want terminal delivery skipped", calls)
	}
	events, err := mem.ListDataAgentJobEvents(ctx, job.JobID, "alice", "org-a", 20)
	if err != nil {
		t.Fatalf("ListDataAgentJobEvents: %v", err)
	}
	last := events[len(events)-1]
	if last.EventType != "data_agent.job.skipped" {
		t.Fatalf("last event = %+v, want skipped audit without lease/progress after cancellation", last)
	}
	if last.EventID != "data_agent_job_event_data_agent_job_worker_canceled_dispatch_unknown_skipped_initial_status" {
		t.Fatalf("skipped event id = %q, want deterministic stale-delivery id", last.EventID)
	}
	if last.ActorUserID != "data-agent-worker-test" || last.ActorOrgID != "org-a" {
		t.Fatalf("skipped actor = %s/%s, want worker/org actor", last.ActorUserID, last.ActorOrgID)
	}
	if last.Metadata["delivery_action"] != "skip" || last.Metadata["control_status"] != "canceled" || last.Metadata["worker_id"] != "data-agent-worker-test" || last.Metadata["stage"] != "initial_status_check" {
		t.Fatalf("skipped metadata = %#v, want durable local-worker skip context", last.Metadata)
	}
}

func TestMetadataSummaryDataAgentProcessorBuildsTemplateSpecificOutputs(t *testing.T) {
	ctx := context.Background()
	now := time.Date(2026, 6, 8, 18, 0, 0, 0, time.UTC)
	resources := []domain.ResourceRecord{
		{
			ResourceID:   "file_a",
			OriginalName: "NPH_001_68yo.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    2048,
			SHA256:       "sha-duplicate",
			SourceType:   "upload",
			ResourceKind: "file",
			ProjectID:    "nph-study",
			Status:       "active",
			CreatedAt:    now,
			UpdatedAt:    now,
			Metadata:     domain.JSONMap{"label": "NPH", "age": float64(68)},
		},
		{
			ResourceID:   "file_b",
			OriginalName: "NPH_002_72yo.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    4096,
			SHA256:       "sha-duplicate",
			SourceType:   "upload",
			ResourceKind: "file",
			ProjectID:    "nph-study",
			Status:       "active",
			CreatedAt:    now.Add(time.Minute),
			UpdatedAt:    now.Add(time.Minute),
			Metadata:     domain.JSONMap{"label": "NPH", "age": float64(72)},
		},
		{
			ResourceID:   "file_c",
			OriginalName: "missing-checksum.csv",
			ContentType:  "text/csv",
			SizeBytes:    0,
			SourceType:   "upload",
			ResourceKind: "table",
			ProjectID:    "nph-study",
			Status:       "active",
			CreatedAt:    now.Add(2 * time.Minute),
			UpdatedAt:    now.Add(2 * time.Minute),
			Metadata:     domain.JSONMap{"label": "QC"},
		},
	}
	progressMessages := []string{}
	processor := MetadataSummaryDataAgentProcessor{}
	run := func(jobType string) domain.JSONMap {
		t.Helper()
		progressMessages = nil
		result, err := processor.ProcessDataAgentJob(ctx, DataAgentWork{
			Envelope: eventbus.DataAgentJob{
				JobType:       jobType,
				ResourceIDs:   []string{"file_a", "file_b", "file_c"},
				ResourceCount: 3,
			},
			Resources: resources,
			ReportProgress: func(_ context.Context, completed int, message string, metadata domain.JSONMap) error {
				progressMessages = append(progressMessages, message)
				if completed <= 0 || metadata["resource_id"] == "" {
					t.Fatalf("progress completed=%d metadata=%+v, want resource progress", completed, metadata)
				}
				return nil
			},
		})
		if err != nil {
			t.Fatalf("ProcessDataAgentJob(%s): %v", jobType, err)
		}
		if len(progressMessages) != len(resources) {
			t.Fatalf("progress messages for %s = %d, want %d", jobType, len(progressMessages), len(resources))
		}
		return result.OutputSummary
	}

	metadataSummary := run("extract_metadata")
	if metadataSummary["summary_kind"] != "metadata_extraction" || metadataSummary["total_size_bytes"] != int64(6144) {
		t.Fatalf("metadata summary = %+v, want metadata extraction totals", metadataSummary)
	}
	if metadataSummary["resource_count"] != len(resources) {
		t.Fatalf("metadata resource_count = %+v, want %d", metadataSummary["resource_count"], len(resources))
	}

	captionSummary := run("caption_resources")
	captions, ok := captionSummary["captions"].([]domain.JSONMap)
	if !ok || len(captions) != 3 || captions[0]["caption"] == "" {
		t.Fatalf("captions = %+v, want deterministic caption entries", captionSummary["captions"])
	}
	if captionSummary["summary_kind"] != "caption_generation" {
		t.Fatalf("caption summary kind = %+v", captionSummary["summary_kind"])
	}

	organizeSummary := run("organize_resources")
	suggestions, ok := organizeSummary["collection_suggestions"].([]domain.JSONMap)
	if !ok || len(suggestions) < 2 {
		t.Fatalf("collection suggestions = %+v, want grouped organization suggestions", organizeSummary["collection_suggestions"])
	}
	if organizeSummary["summary_kind"] != "organization_plan" {
		t.Fatalf("organize summary kind = %+v", organizeSummary["summary_kind"])
	}

	dedupeSummary := run("deduplicate_resources")
	groups, ok := dedupeSummary["duplicate_groups"].([]domain.JSONMap)
	if !ok || len(groups) != 1 || groups[0]["sha256"] != "sha-duplicate" {
		t.Fatalf("duplicate groups = %+v, want one checksum group", dedupeSummary["duplicate_groups"])
	}
	if dedupeSummary["duplicate_resource_count"] != 2 {
		t.Fatalf("duplicate_resource_count = %+v, want 2", dedupeSummary["duplicate_resource_count"])
	}

	qualitySummary := run("quality_check_resources")
	warnings, ok := qualitySummary["warnings"].([]domain.JSONMap)
	if !ok || len(warnings) < 2 {
		t.Fatalf("quality warnings = %+v, want missing checksum and empty file warnings", qualitySummary["warnings"])
	}
	if qualitySummary["summary_kind"] != "quality_check" {
		t.Fatalf("quality summary kind = %+v", qualitySummary["summary_kind"])
	}
}

type recordingDataAgentWorkerStore struct {
	*store.MemoryStore
	renewCount   int
	releaseCount int
}

func (s *recordingDataAgentWorkerStore) RenewDataAgentJobLease(ctx context.Context, input domain.RenewDataAgentJobLeaseInput) (domain.DataAgentJobLeaseRecord, error) {
	s.renewCount++
	return s.MemoryStore.RenewDataAgentJobLease(ctx, input)
}

func (s *recordingDataAgentWorkerStore) ReleaseDataAgentJobLease(ctx context.Context, input domain.ReleaseDataAgentJobLeaseInput) error {
	s.releaseCount++
	return s.MemoryStore.ReleaseDataAgentJobLease(ctx, input)
}

type pagingDataAgentWorkerStore struct {
	*store.MemoryStore
	pageSize  int
	listCalls []domain.ResourceListInput
}

func (s *pagingDataAgentWorkerStore) ListResourcesForUser(ctx context.Context, input domain.ResourceListInput) (domain.ResourceListPage, error) {
	s.listCalls = append(s.listCalls, input)
	fullInput := input
	fullInput.Limit = 0
	fullInput.Offset = 0
	page, err := s.MemoryStore.ListResourcesForUser(ctx, fullInput)
	if err != nil {
		return domain.ResourceListPage{}, err
	}
	offset := input.Offset
	if offset < 0 {
		offset = 0
	}
	if offset >= len(page.Resources) {
		page.Resources = nil
	} else {
		end := len(page.Resources)
		if s.pageSize > 0 && offset+s.pageSize < end {
			end = offset + s.pageSize
		}
		page.Resources = append([]domain.ResourceRecord(nil), page.Resources[offset:end]...)
	}
	page.Limit = input.Limit
	page.Offset = input.Offset
	return page, nil
}

func seedDataAgentWorkerResource(t *testing.T, ctx context.Context, mem *store.MemoryStore, resourceID string, name string, sha string, ts time.Time) {
	t.Helper()
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   resourceID,
		OriginalName: name,
		ContentType:  "application/x-nifti",
		SizeBytes:    128,
		SHA256:       sha,
		ResourceKind: "file",
		SourceType:   "upload",
		ProjectID:    "nph-study",
		OwnerUserID:  "alice",
		OwnerOrgID:   "org-a",
		Status:       "active",
		CreatedAt:    ts,
		UpdatedAt:    ts,
		Metadata:     domain.JSONMap{"label": "NPH"},
	}); err != nil {
		t.Fatalf("UpsertResource(%s): %v", resourceID, err)
	}
}

func seedDataAgentWorkerJob(t *testing.T, ctx context.Context, mem *store.MemoryStore, jobID string, resourceIDs []string, ts time.Time) domain.DataAgentJobRecord {
	t.Helper()
	job, err := mem.CreateDataAgentJob(ctx, domain.CreateDataAgentJobInput{
		JobID:           jobID,
		OwnerUserID:     "alice",
		OwnerOrgID:      "org-a",
		ProjectID:       "nph-study",
		JobType:         "caption_resources",
		ResourceIDs:     resourceIDs,
		InputSelector:   domain.JSONMap{"resource_ids": stringSliceToAny(resourceIDs)},
		CreatedByUserID: "alice",
		CreatedAt:       ts,
		UpdatedAt:       ts,
	})
	if err != nil {
		t.Fatalf("CreateDataAgentJob: %v", err)
	}
	return job
}

func stringSliceToAny(values []string) []any {
	out := make([]any, 0, len(values))
	for _, value := range values {
		out = append(out, value)
	}
	return out
}

func sameStringSlice(a []string, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
