package worker

import (
	"context"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

func TestStubWorkerRunJobCompletesRunAndCreatesArtifact(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	worker := NewStubWorker(mem, bus)

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{UserID: "user-1", Title: "worker"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "finish",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "finish"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	var job eventbus.Job
	select {
	case job = <-bus.Jobs():
	case <-time.After(time.Second):
		t.Fatalf("expected dispatched job")
	}

	if err := worker.RunJob(ctx, job); err != nil {
		t.Fatalf("RunJob: %v", err)
	}

	updated, err := mem.GetRun(ctx, run.RunID)
	if err != nil {
		t.Fatalf("GetRun: %v", err)
	}
	if updated.Status != domain.RunStatusSucceeded {
		t.Fatalf("run status = %s, want succeeded", updated.Status)
	}
	if updated.ResponseText == "" || updated.CompletedAt == nil {
		t.Fatalf("completed run missing response/completed_at: %+v", updated)
	}

	artifacts, err := mem.ListRunArtifacts(ctx, run.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunArtifacts: %v", err)
	}
	if len(artifacts) != 1 || artifacts[0].Path != "outputs/stub-report.md" {
		t.Fatalf("artifacts = %+v, want stub report", artifacts)
	}

	events, err := mem.ListRunEvents(ctx, run.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	if events[len(events)-1].EventKind != "run.completed" {
		t.Fatalf("last event = %+v, want run.completed", events[len(events)-1])
	}
}
