package runcontrol

import (
	"context"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

func TestServiceCreateRunEmitsAcceptedAndDispatches(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Test thread",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}

	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Run deterministic worker.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Run deterministic worker."}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	if run.Status != domain.RunStatusQueued {
		t.Fatalf("run status = %s, want queued", run.Status)
	}

	events, err := mem.ListRunEvents(ctx, run.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	if len(events) != 1 || events[0].EventKind != "run.accepted" {
		t.Fatalf("events = %+v, want run.accepted", events)
	}

	select {
	case job := <-bus.Jobs():
		if job.RunID != run.RunID || job.ThreadID != thread.ThreadID {
			t.Fatalf("job = %+v, want run/thread ids", job)
		}
	case <-time.After(time.Second):
		t.Fatalf("expected dispatched job")
	}
}
