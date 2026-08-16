package store

import (
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

func TestGeneratedThreadTitleMetadataPreservesServerOwnedProvenance(t *testing.T) {
	t.Parallel()

	now := time.Date(2026, time.August, 12, 9, 10, 11, 123456789, time.FixedZone("PDT", -7*60*60))
	metadata := generatedThreadTitleMetadata(
		domain.JSONMap{"unrelated": "preserved"},
		domain.ApplyGeneratedThreadTitleInput{
			RunID: "  run-authoritative  ",
			Generation: domain.JSONMap{
				"source":         "auto",
				"run_id":         "run-spoofed",
				"previous_title": "Spoofed previous title",
				"updated_at":     "2000-01-01T00:00:00Z",
				"strategy":       "llm",
				"model":          "deepseek_v4",
			},
		},
		"  Authoritative previous title  ",
		now,
	)

	if got := metadata["unrelated"]; got != "preserved" {
		t.Fatalf("unrelated metadata = %v, want preserved", got)
	}
	state, ok := metadata[threadTitleStateKey].(domain.JSONMap)
	if !ok {
		t.Fatalf("title state = %#v, want domain.JSONMap", metadata[threadTitleStateKey])
	}
	want := domain.JSONMap{
		"source":         "generated",
		"run_id":         "run-authoritative",
		"previous_title": "Authoritative previous title",
		"updated_at":     now.UTC().Format(time.RFC3339Nano),
		"strategy":       "llm",
		"model":          "deepseek_v4",
	}
	for key, wantValue := range want {
		if got := state[key]; got != wantValue {
			t.Errorf("title state %q = %v, want %v", key, got, wantValue)
		}
	}
	if got := threadTitleStateSource(metadata); got != "generated" {
		t.Errorf("thread title state source = %q, want generated", got)
	}
	if generatedThreadTitleEligible(domain.ThreadRecord{
		Title:    "Generated authoritative title",
		Metadata: metadata,
	}) {
		t.Error("non-placeholder generated title is eligible for replacement")
	}
}
