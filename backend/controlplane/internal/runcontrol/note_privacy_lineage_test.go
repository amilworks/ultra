package runcontrol

import (
	"context"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

func TestServiceCreateRunCarriesNotePrivacyLineageIntoFollowupJob(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)
	thread, err := service.CreateThread(ctx, CreateThreadRequest{UserID: "user-1", Title: "private follow-up"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}

	noteRun, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Use my Note",
		SelectionContext: domain.CanonicalNoteAccessSelection(nil, domain.NoteAccessScope{
			Mode: domain.NoteAccessModeSearch,
		}),
	})
	if err != nil {
		t.Fatalf("CreateRun Note: %v", err)
	}
	if !domain.RunHasNotePrivacyLineage(noteRun) {
		t.Fatalf("Note run lacks privacy lineage: %+v", noteRun.Metadata)
	}
	select {
	case job := <-bus.Jobs():
		if job.Metadata[domain.NotePrivacyLineageMetadataKey] != true {
			t.Fatalf("Note job metadata = %+v, want privacy marker", job.Metadata)
		}
	case <-time.After(time.Second):
		t.Fatal("expected Note job")
	}

	followupRun, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Explain that answer",
		Metadata: domain.JSONMap{domain.NotePrivacyLineageMetadataKey: false},
	})
	if err != nil {
		t.Fatalf("CreateRun follow-up: %v", err)
	}
	if !domain.RunHasNotePrivacyLineage(followupRun) || domain.RunHasNoteAccessSelection(followupRun) {
		t.Fatalf("follow-up metadata = %+v, want lineage without Note scope", followupRun.Metadata)
	}
	select {
	case job := <-bus.Jobs():
		if job.Metadata[domain.NotePrivacyLineageMetadataKey] != true {
			t.Fatalf("follow-up job metadata = %+v, want inherited privacy marker", job.Metadata)
		}
	case <-time.After(time.Second):
		t.Fatal("expected follow-up job")
	}

	ordinaryThread, err := service.CreateThread(ctx, CreateThreadRequest{UserID: "user-1", Title: "ordinary"})
	if err != nil {
		t.Fatalf("CreateThread ordinary: %v", err)
	}
	ordinaryRun, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: ordinaryThread.ThreadID,
		UserID:   "user-1",
		Goal:     "ordinary",
		Metadata: domain.JSONMap{domain.NotePrivacyLineageMetadataKey: true},
	})
	if err != nil {
		t.Fatalf("CreateRun ordinary: %v", err)
	}
	if domain.RunHasNotePrivacyLineage(ordinaryRun) {
		t.Fatalf("service caller spoof minted privacy lineage: %+v", ordinaryRun.Metadata)
	}
}
