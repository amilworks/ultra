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

func TestRunIdempotencyKeyBindsImmutableNoteAccess(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	service := NewService(mem, eventbus.NewMemoryBus())
	thread, err := service.CreateThread(ctx, CreateThreadRequest{UserID: "ada", Title: "notes"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	request := CreateRunRequest{
		ThreadID:       thread.ThreadID,
		UserID:         "ada",
		Goal:           "use notes",
		IdempotencyKey: "notes-run-key",
		SelectionContext: domain.CanonicalNoteAccessSelection(nil, domain.NoteAccessScope{
			Mode:  domain.NoteAccessModeSelected,
			Notes: []domain.NoteReference{{NoteID: "note_a", Revision: 1}},
		}),
	}
	first, err := service.CreateRun(ctx, request)
	if err != nil {
		t.Fatalf("first CreateRun: %v", err)
	}
	replay, err := service.CreateRun(ctx, request)
	if err != nil || replay.RunID != first.RunID {
		t.Fatalf("same-scope replay = %+v err=%v", replay, err)
	}
	request.SelectionContext = domain.CanonicalNoteAccessSelection(nil, domain.NoteAccessScope{
		Mode:  domain.NoteAccessModeSelected,
		Notes: []domain.NoteReference{{NoteID: "note_b", Revision: 1}},
	})
	if _, err := service.CreateRun(ctx, request); !errors.Is(err, store.ErrConflict) {
		t.Fatalf("different note scope err = %v, want ErrConflict", err)
	}
}

func TestCreateRunStampsServerResolvedModelNotesProposalAvailability(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)
	thread, err := service.CreateThread(ctx, CreateThreadRequest{UserID: "ada", Title: "proposal flag"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "ada",
		Goal:     "search notes",
		SelectionContext: domain.CanonicalNoteAccessSelection(nil, domain.NoteAccessScope{
			Mode: domain.NoteAccessModeSearch,
		}),
		Metadata: domain.JSONMap{domain.ModelNotesProposalsEnabledMetadataKey: true},
		JobMetadata: domain.JSONMap{
			domain.ModelNotesProposalsEnabledMetadataKey: true,
		},
		ModelNotesProposalsEnabled: false,
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	if value, ok := run.Metadata[domain.ModelNotesProposalsEnabledMetadataKey].(bool); !ok || value {
		t.Fatalf("run proposal availability = %#v, want server-authored false", run.Metadata[domain.ModelNotesProposalsEnabledMetadataKey])
	}
	select {
	case job := <-bus.Jobs():
		if value, ok := job.Metadata[domain.ModelNotesProposalsEnabledMetadataKey].(bool); !ok || value {
			t.Fatalf("job proposal availability = %#v, want server-authored false", job.Metadata[domain.ModelNotesProposalsEnabledMetadataKey])
		}
	case <-time.After(time.Second):
		t.Fatal("expected dispatched job")
	}
}

func TestRunIdempotencyKeyReplaysRawNoteScopeButBindsExplicitFields(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	service := NewService(mem, eventbus.NewMemoryBus())
	thread, err := service.CreateThread(ctx, CreateThreadRequest{UserID: "ada", Title: "notes"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	canonical := CreateRunRequest{
		ThreadID:       thread.ThreadID,
		UserID:         "ada",
		Goal:           "use notes",
		IdempotencyKey: "notes-raw-replay-key",
		SelectionContext: domain.CanonicalNoteAccessSelection(nil, domain.NoteAccessScope{
			Mode:                domain.NoteAccessModeSelected,
			Notes:               []domain.NoteReference{{NoteID: "note_a", Revision: 7}},
			AllowAppendProposal: true,
		}),
	}
	first, err := service.CreateRun(ctx, canonical)
	if err != nil {
		t.Fatalf("first CreateRun: %v", err)
	}
	rawReplay := canonical
	rawReplay.SelectionContext = domain.JSONMap{domain.NoteAccessSelectionKey: domain.JSONMap{
		"mode": "selected", "notes": []any{domain.JSONMap{"note_id": "note_a"}}, "allow_append_proposal": true,
	}}
	replay, err := service.CreateRun(ctx, rawReplay)
	if err != nil || replay.RunID != first.RunID {
		t.Fatalf("raw replay = %+v err=%v", replay, err)
	}

	for name, mutate := range map[string]func(*CreateRunRequest){
		"different mode": func(req *CreateRunRequest) {
			req.SelectionContext[domain.NoteAccessSelectionKey].(domain.JSONMap)["mode"] = "search"
		},
		"different note": func(req *CreateRunRequest) {
			req.SelectionContext[domain.NoteAccessSelectionKey].(domain.JSONMap)["notes"] = []any{domain.JSONMap{"note_id": "note_b"}}
		},
		"different explicit revision": func(req *CreateRunRequest) {
			req.SelectionContext[domain.NoteAccessSelectionKey].(domain.JSONMap)["notes"] = []any{domain.JSONMap{"note_id": "note_a", "revision": 8}}
		},
		"different append flag": func(req *CreateRunRequest) {
			req.SelectionContext[domain.NoteAccessSelectionKey].(domain.JSONMap)["allow_append_proposal"] = false
		},
	} {
		t.Run(name, func(t *testing.T) {
			candidate := rawReplay
			candidate.SelectionContext = domain.JSONMap{domain.NoteAccessSelectionKey: domain.JSONMap{
				"mode": "selected", "notes": []any{domain.JSONMap{"note_id": "note_a"}}, "allow_append_proposal": true,
			}}
			mutate(&candidate)
			if _, err := service.CreateRun(ctx, candidate); !errors.Is(err, store.ErrConflict) {
				t.Fatalf("CreateRun err = %v, want ErrConflict", err)
			}
		})
	}
}
