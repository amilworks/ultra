package runcontrol

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

func TestCreateRunRejectsUnknownAndFreeFormEvaluationProfiles(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)
	thread, err := service.CreateThread(ctx, CreateThreadRequest{UserID: "evaluator", Title: "profile security"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}

	if _, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID:          thread.ThreadID,
		UserID:            "evaluator",
		Goal:              "unknown profile",
		EvaluationProfile: domain.EvaluationProfile("future_profile"),
	}); !errors.Is(err, ErrInvalidEvaluationProfile) {
		t.Fatalf("unknown profile err = %v, want ErrInvalidEvaluationProfile", err)
	}

	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "evaluator",
		Goal:     "metadata cannot grant a profile",
		Metadata: domain.JSONMap{
			domain.EvaluationProfileMetadataKey: string(domain.EvaluationProfileMaterialsCleanroomV1),
		},
		JobMetadata: domain.JSONMap{
			domain.EvaluationProfileMetadataKey: string(domain.EvaluationProfileMaterialsCleanroomV1),
		},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	if _, found := run.Metadata[domain.EvaluationProfileMetadataKey]; found {
		t.Fatalf("free-form run metadata granted evaluation profile: %#v", run.Metadata)
	}
	job := receiveEvaluationProfileJob(t, bus)
	if job.EvaluationProfile != "" {
		t.Fatalf("free-form job profile = %q, want empty", job.EvaluationProfile)
	}
	if _, found := job.Metadata[domain.EvaluationProfileMetadataKey]; found {
		t.Fatalf("free-form job metadata granted evaluation profile: %#v", job.Metadata)
	}
	events, err := mem.ListRunEvents(ctx, run.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("events = %d, want accepted event", len(events))
	}
	if _, found := events[0].Payload[domain.EvaluationProfileMetadataKey]; found {
		t.Fatalf("unprofiled accepted event claims profile: %#v", events[0].Payload)
	}
}

func TestMaterialsCleanroomProfilePersistsAndPropagatesFromStoredRun(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)
	thread, err := service.CreateThread(ctx, CreateThreadRequest{UserID: "evaluator", Title: "clean room"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}

	const idempotencyKey = "materials-cleanroom-evaluation"
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID:          thread.ThreadID,
		UserID:            "evaluator",
		Goal:              "evaluate materials analysis",
		EvaluationProfile: domain.EvaluationProfileMaterialsCleanroomV1,
		IdempotencyKey:    idempotencyKey,
		Metadata: domain.JSONMap{
			domain.EvaluationProfileMetadataKey: "caller_forgery",
		},
		JobMetadata: domain.JSONMap{
			domain.EvaluationProfileMetadataKey: "transient_forgery",
		},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	assertStoredMaterialsCleanroomProfile(t, run)
	initialJob := receiveEvaluationProfileJob(t, bus)
	assertMaterialsCleanroomJob(t, initialJob)

	events, err := mem.ListRunEvents(ctx, run.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunEvents accepted: %v", err)
	}
	if len(events) != 1 || events[0].EventKind != "run.accepted" {
		t.Fatalf("accepted events = %#v", events)
	}
	if got := events[0].Payload[domain.EvaluationProfileMetadataKey]; got != string(domain.EvaluationProfileMaterialsCleanroomV1) {
		t.Fatalf("accepted event profile = %#v", got)
	}

	if _, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID:       thread.ThreadID,
		UserID:         "evaluator",
		Goal:           "try to change the profile",
		IdempotencyKey: idempotencyKey,
	}); !errors.Is(err, store.ErrConflict) {
		t.Fatalf("idempotent profile change err = %v, want ErrConflict", err)
	}

	if _, err := service.RequeueRun(ctx, RequeueRunRequest{
		RunID:  run.RunID,
		Reason: "clean-room retry",
		Metadata: domain.JSONMap{
			domain.EvaluationProfileMetadataKey: "retry_forgery",
		},
	}); err != nil {
		t.Fatalf("RequeueRun: %v", err)
	}
	retryJob := receiveEvaluationProfileJob(t, bus)
	assertMaterialsCleanroomJob(t, retryJob)
	if retryJob.DispatchID == "" {
		t.Fatal("retry job missing dispatch id")
	}

	stored, err := mem.GetRun(ctx, run.RunID)
	if err != nil {
		t.Fatalf("GetRun: %v", err)
	}
	assertStoredMaterialsCleanroomProfile(t, stored)
	events, err = mem.ListRunEvents(ctx, run.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunEvents requeued: %v", err)
	}
	if len(events) != 2 || events[1].EventKind != "run.requeued" {
		t.Fatalf("events after requeue = %#v", events)
	}
	if got := events[1].Payload[domain.EvaluationProfileMetadataKey]; got != string(domain.EvaluationProfileMaterialsCleanroomV1) {
		t.Fatalf("requeue event profile = %#v", got)
	}
}

func receiveEvaluationProfileJob(t *testing.T, bus *eventbus.MemoryBus) eventbus.Job {
	t.Helper()
	select {
	case job := <-bus.Jobs():
		return job
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for job")
		return eventbus.Job{}
	}
}

func assertStoredMaterialsCleanroomProfile(t *testing.T, run domain.RunRecord) {
	t.Helper()
	if got := run.Metadata[domain.EvaluationProfileMetadataKey]; got != string(domain.EvaluationProfileMaterialsCleanroomV1) {
		t.Fatalf("stored evaluation profile = %#v", got)
	}
}

func assertMaterialsCleanroomJob(t *testing.T, job eventbus.Job) {
	t.Helper()
	if job.EvaluationProfile != domain.EvaluationProfileMaterialsCleanroomV1 {
		t.Fatalf("job evaluation profile = %q", job.EvaluationProfile)
	}
	if got := job.Metadata[domain.EvaluationProfileMetadataKey]; got != string(domain.EvaluationProfileMaterialsCleanroomV1) {
		t.Fatalf("job metadata evaluation profile = %#v", got)
	}
}
