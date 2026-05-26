package worker

import (
	"context"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
)

type StubWorker struct {
	store runcontrol.Store
	bus   eventbus.Bus
}

func NewStubWorker(store runcontrol.Store, bus eventbus.Bus) *StubWorker {
	return &StubWorker{store: store, bus: bus}
}

func (w *StubWorker) RunJob(ctx context.Context, job eventbus.Job) error {
	if _, err := w.store.UpdateRunStatus(ctx, job.RunID, domain.RunStatusRunning, "", ""); err != nil {
		return err
	}
	events := []domain.AppendRunEventInput{
		{RunID: job.RunID, ThreadID: job.ThreadID, EventKind: "run.started", Message: "Planning started.", Payload: domain.JSONMap{"phase": "planning"}},
		{RunID: job.RunID, ThreadID: job.ThreadID, EventKind: "message.delta", Message: "Analyzing request.", Payload: domain.JSONMap{"delta": "Analyzing request."}},
		{RunID: job.RunID, ThreadID: job.ThreadID, EventKind: "artifact.created", Message: "Created deterministic report.", Payload: domain.JSONMap{"path": "outputs/stub-report.md"}},
	}
	for _, input := range events {
		select {
		case <-ctx.Done():
			_, _ = w.store.UpdateRunStatus(context.Background(), job.RunID, domain.RunStatusCanceled, "", "canceled")
			return ctx.Err()
		default:
		}
		event, err := w.store.AppendRunEvent(ctx, input)
		if err != nil {
			return err
		}
		if err := w.bus.PublishRunEvent(ctx, event); err != nil {
			return err
		}
		time.Sleep(5 * time.Millisecond)
	}
	if _, err := w.store.CreateArtifact(ctx, domain.CreateArtifactInput{
		RunID:      job.RunID,
		ThreadID:   job.ThreadID,
		Kind:       "report",
		Path:       "outputs/stub-report.md",
		Title:      "Deterministic Stub Report",
		MimeType:   "text/markdown",
		SizeBytes:  128,
		SHA256:     "stub-sha256",
		StorageURI: "memory://outputs/stub-report.md",
		Metadata:   domain.JSONMap{"worker": "stub"},
	}); err != nil {
		return err
	}
	if _, err := w.store.UpdateRunStatus(ctx, job.RunID, domain.RunStatusSucceeded, "Deterministic worker completed.", ""); err != nil {
		return err
	}
	event, err := w.store.AppendRunEvent(ctx, domain.AppendRunEventInput{
		RunID:     job.RunID,
		ThreadID:  job.ThreadID,
		EventKind: "run.completed",
		Message:   "Run completed.",
		Payload:   domain.JSONMap{"status": "succeeded"},
	})
	if err != nil {
		return err
	}
	return w.bus.PublishRunEvent(ctx, event)
}
