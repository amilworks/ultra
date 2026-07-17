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

func TestCreateRunIgnoresRemoteMutationIntentMetadataSpoof(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)
	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "remote mutation metadata spoof",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}

	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Explain how BisQue upload works without changing anything.",
		Metadata: domain.JSONMap{
			domain.RemoteMutationIntentsMetadataKey: []any{string(domain.RemoteMutationIntentBisqueUpload)},
		},
		JobMetadata: domain.JSONMap{
			domain.RemoteMutationIntentsMetadataKey: []string{string(domain.RemoteMutationIntentBisqueCreateDataset)},
		},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	if _, found := run.Metadata[domain.RemoteMutationIntentsMetadataKey]; found {
		t.Fatalf("free-form run metadata granted mutation intent: %#v", run.Metadata)
	}

	job := receiveRemoteMutationJob(t, bus)
	assertRemoteMutationIntents(t, job.RemoteMutationIntents, nil)
	if _, found := job.Metadata[domain.RemoteMutationIntentsMetadataKey]; found {
		t.Fatalf("free-form job metadata granted mutation intent: %#v", job.Metadata)
	}
}

// No evaluation profile is currently supported, so a run that requests one
// alongside remote mutation intents is rejected before dispatch. The
// evaluation-profile/mutation exclusion in CreateRun stays as the guard that
// re-arms if a profile is reintroduced.
func TestCreateRunRejectsRemoteMutationIntentsForProtectedEvaluationProfile(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)
	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "evaluator",
		Title:  "protected evaluation",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}

	for _, intent := range []domain.RemoteMutationIntent{
		domain.RemoteMutationIntentBisqueUpload,
		domain.RemoteMutationIntentBisqueCreateDataset,
	} {
		t.Run(string(intent), func(t *testing.T) {
			_, err := service.CreateRun(ctx, CreateRunRequest{
				ThreadID:              thread.ThreadID,
				UserID:                "evaluator",
				Goal:                  "Evaluate analysis in a clean room.",
				EvaluationProfile:     domain.EvaluationProfile("protected_profile_v1"),
				RemoteMutationIntents: []domain.RemoteMutationIntent{intent},
			})
			if !errors.Is(err, ErrInvalidEvaluationProfile) {
				t.Fatalf("CreateRun err = %v, want ErrInvalidEvaluationProfile", err)
			}
		})
	}

	select {
	case job := <-bus.Jobs():
		t.Fatalf("protected evaluation published mutation-capable job: %+v", job)
	default:
	}
}

func TestCreateRunIdempotencyConflictsWhenRemoteMutationIntentChanges(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)
	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "immutable remote mutation intent",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}

	const idempotencyKey = "remote-mutation-contract-1"
	created, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID:              thread.ThreadID,
		UserID:                "user-1",
		Goal:                  "Upload the completed analysis to BisQue.",
		IdempotencyKey:        idempotencyKey,
		RemoteMutationIntents: []domain.RemoteMutationIntent{domain.RemoteMutationIntentBisqueUpload},
	})
	if err != nil {
		t.Fatalf("CreateRun initial: %v", err)
	}
	initialJob := receiveRemoteMutationJob(t, bus)
	assertRemoteMutationIntents(t, initialJob.RemoteMutationIntents, []domain.RemoteMutationIntent{
		domain.RemoteMutationIntentBisqueUpload,
	})

	for name, changed := range map[string][]domain.RemoteMutationIntent{
		"removed": nil,
		"replaced": {
			domain.RemoteMutationIntentBisqueCreateDataset,
		},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := service.CreateRun(ctx, CreateRunRequest{
				ThreadID:              thread.ThreadID,
				UserID:                "user-1",
				Goal:                  "Retry with a changed external mutation contract.",
				IdempotencyKey:        idempotencyKey,
				RemoteMutationIntents: changed,
			})
			if !errors.Is(err, store.ErrConflict) {
				t.Fatalf("CreateRun changed intent err = %v, want store.ErrConflict", err)
			}
		})
	}

	stored, err := mem.GetRun(ctx, created.RunID)
	if err != nil {
		t.Fatalf("GetRun: %v", err)
	}
	assertRemoteMutationMetadata(t, stored.Metadata, []domain.RemoteMutationIntent{
		domain.RemoteMutationIntentBisqueUpload,
	})
	select {
	case job := <-bus.Jobs():
		t.Fatalf("changed idempotent request published a second job: %+v", job)
	default:
	}
}

func TestRequeueRunPreservesStoredRemoteMutationIntentAndIgnoresOverride(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)
	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "remote mutation requeue",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}

	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID:              thread.ThreadID,
		UserID:                "user-1",
		Goal:                  "Upload the completed analysis to BisQue.",
		RemoteMutationIntents: []domain.RemoteMutationIntent{domain.RemoteMutationIntentBisqueUpload},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	receiveRemoteMutationJob(t, bus)

	if _, err := service.RequeueRun(ctx, RequeueRunRequest{
		RunID:  run.RunID,
		Reason: "worker lease expired",
		Metadata: domain.JSONMap{
			domain.RemoteMutationIntentsMetadataKey: []any{
				string(domain.RemoteMutationIntentBisqueCreateDataset),
			},
		},
	}); err != nil {
		t.Fatalf("RequeueRun: %v", err)
	}

	retryJob := receiveRemoteMutationJob(t, bus)
	assertRemoteMutationIntents(t, retryJob.RemoteMutationIntents, []domain.RemoteMutationIntent{
		domain.RemoteMutationIntentBisqueUpload,
	})
	assertRemoteMutationMetadata(t, retryJob.Metadata, []domain.RemoteMutationIntent{
		domain.RemoteMutationIntentBisqueUpload,
	})

	stored, err := mem.GetRun(ctx, run.RunID)
	if err != nil {
		t.Fatalf("GetRun: %v", err)
	}
	assertRemoteMutationMetadata(t, stored.Metadata, []domain.RemoteMutationIntent{
		domain.RemoteMutationIntentBisqueUpload,
	})
	events, err := mem.ListRunEvents(ctx, run.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	if len(events) != 2 || events[1].EventKind != "run.requeued" {
		t.Fatalf("events after requeue = %#v", events)
	}
	if _, found := events[1].Payload[domain.RemoteMutationIntentsMetadataKey]; found {
		t.Fatalf("requeue event retained mutation override: %#v", events[1].Payload)
	}
}

func receiveRemoteMutationJob(t *testing.T, bus *eventbus.MemoryBus) eventbus.Job {
	t.Helper()
	select {
	case job := <-bus.Jobs():
		return job
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for remote-mutation job")
		return eventbus.Job{}
	}
}

func assertRemoteMutationMetadata(
	t *testing.T,
	metadata domain.JSONMap,
	want []domain.RemoteMutationIntent,
) {
	t.Helper()
	got, valid := domain.RemoteMutationIntentsFromMetadata(metadata)
	if !valid {
		t.Fatalf("remote mutation metadata is invalid: %#v", metadata)
	}
	assertRemoteMutationIntents(t, got, want)
}

func assertRemoteMutationIntents(
	t *testing.T,
	got []domain.RemoteMutationIntent,
	want []domain.RemoteMutationIntent,
) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("remote mutation intents = %#v, want %#v", got, want)
	}
	for index := range want {
		if got[index] != want[index] {
			t.Fatalf("remote mutation intents = %#v, want %#v", got, want)
		}
	}
}
