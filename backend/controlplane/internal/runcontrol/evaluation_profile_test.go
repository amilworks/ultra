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

	// No evaluation profile is supported, so every non-empty profile — including
	// the former materials_cleanroom_v1 — must be rejected.
	for _, profile := range []domain.EvaluationProfile{"future_profile", "materials_cleanroom_v1"} {
		if _, err := service.CreateRun(ctx, CreateRunRequest{
			ThreadID:          thread.ThreadID,
			UserID:            "evaluator",
			Goal:              "unknown profile",
			EvaluationProfile: profile,
		}); !errors.Is(err, ErrInvalidEvaluationProfile) {
			t.Fatalf("profile %q err = %v, want ErrInvalidEvaluationProfile", profile, err)
		}
	}

	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "evaluator",
		Goal:     "metadata cannot grant a profile",
		Metadata: domain.JSONMap{
			domain.EvaluationProfileMetadataKey: "materials_cleanroom_v1",
		},
		JobMetadata: domain.JSONMap{
			domain.EvaluationProfileMetadataKey: "materials_cleanroom_v1",
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

// The run record is the only authority for a protected profile. With no profile
// currently supported, an unrecognized profile stored on the run must never be
// projected onto a job or an attested event payload.
func TestStoredEvaluationProfileNeverPropagatesUnrecognizedProfiles(t *testing.T) {
	t.Parallel()
	run := domain.RunRecord{Metadata: domain.JSONMap{
		domain.EvaluationProfileMetadataKey: "materials_cleanroom_v1",
	}}

	if profile := storedEvaluationProfile(run); profile != "" {
		t.Fatalf("storedEvaluationProfile = %q, want empty", profile)
	}
	if !storedEvaluationProfileMatches(run, "") {
		t.Fatal("storedEvaluationProfileMatches(run, \"\") = false, want true")
	}
	if storedEvaluationProfileMatches(run, "materials_cleanroom_v1") {
		t.Fatal("an unrecognized stored profile must not match a requested profile")
	}

	metadata := metadataWithStoredEvaluationProfile(run, domain.JSONMap{
		domain.EvaluationProfileMetadataKey: "forgery",
		"keep":                              "value",
	})
	if _, found := metadata[domain.EvaluationProfileMetadataKey]; found {
		t.Fatalf("job metadata retained an evaluation profile: %#v", metadata)
	}
	if metadata["keep"] != "value" {
		t.Fatalf("job metadata dropped unrelated keys: %#v", metadata)
	}

	payload := domain.JSONMap{domain.EvaluationProfileMetadataKey: "forgery", "keep": "value"}
	attestEvaluationProfile(payload, run)
	if _, found := payload[domain.EvaluationProfileMetadataKey]; found {
		t.Fatalf("attested payload retained an evaluation profile: %#v", payload)
	}
	if payload["keep"] != "value" {
		t.Fatalf("attested payload dropped unrelated keys: %#v", payload)
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
