package store

import (
	"context"
	"testing"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

func TestMemoryStoreThreadRunEventArtifactFlow(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	thread, err := store.CreateThread(ctx, domain.CreateThreadInput{
		UserID: "user-1",
		Title:  "Microscopy analysis",
		InitialMessages: []domain.ThreadMessage{{
			Role:    "user",
			Content: "Segment these images.",
		}},
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	if thread.ThreadID == "" || thread.Status != domain.ThreadStatusActive {
		t.Fatalf("unexpected thread: %+v", thread)
	}

	run, err := store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Segment these images.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Segment these images."}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	if run.Status != domain.RunStatusQueued {
		t.Fatalf("run status = %s, want queued", run.Status)
	}

	event, err := store.AppendRunEvent(ctx, domain.AppendRunEventInput{
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.started",
		Message:   "Run started.",
		Payload:   domain.JSONMap{"phase": "planning"},
	})
	if err != nil {
		t.Fatalf("AppendRunEvent: %v", err)
	}
	if event.EventID == "" {
		t.Fatalf("event id must be set")
	}

	artifact, err := store.CreateArtifact(ctx, domain.CreateArtifactInput{
		RunID:      run.RunID,
		ThreadID:   thread.ThreadID,
		Kind:       "report",
		Path:       "outputs/report.md",
		Title:      "Report",
		MimeType:   "text/markdown",
		SizeBytes:  42,
		SHA256:     "abc123",
		StorageURI: "file://outputs/report.md",
		Metadata:   domain.JSONMap{"source": "stub"},
	})
	if err != nil {
		t.Fatalf("CreateArtifact: %v", err)
	}
	if artifact.ArtifactID == "" {
		t.Fatalf("artifact id must be set")
	}

	events, err := store.ListRunEvents(ctx, run.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	if len(events) != 1 || events[0].EventKind != "run.started" {
		t.Fatalf("events = %+v, want one run.started", events)
	}

	artifacts, err := store.ListRunArtifacts(ctx, run.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunArtifacts: %v", err)
	}
	if len(artifacts) != 1 || artifacts[0].Path != "outputs/report.md" {
		t.Fatalf("artifacts = %+v, want report", artifacts)
	}
}
