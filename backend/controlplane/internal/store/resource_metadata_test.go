package store

import (
	"context"
	"errors"
	"testing"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

func viewerCalibrationPatch(sha string, revision int, threshold int) domain.JSONMap {
	return domain.JSONMap{
		"ultra_viewer_calibration_v1": domain.JSONMap{
			"version":       1,
			"source_sha256": sha,
			"selections": domain.JSONMap{
				"c0:t0": domain.JSONMap{
					"revision":        revision,
					"threshold_value": threshold,
				},
			},
		},
	}
}

func TestMergeResourceMetadataReplacesViewerCalibrationForNewSourceSHA(t *testing.T) {
	existing := domain.JSONMap{
		"review": domain.JSONMap{"status": "draft", "reader": "bioio"},
		"ultra_viewer_calibration_v1": domain.JSONMap{
			"version":       1,
			"source_sha256": "old-sha",
			"stale":         true,
			"selections": domain.JSONMap{
				"c0:t0": domain.JSONMap{"threshold_value": 12},
			},
		},
	}
	snapshot := domain.JSONMap{
		"version":       1,
		"source_sha256": "new-sha",
		"selections": domain.JSONMap{
			"c0:t0": domain.JSONMap{"threshold_value": 120},
		},
	}

	merged := mergeResourceMetadata(existing, domain.JSONMap{
		"review":                      domain.JSONMap{"status": "approved"},
		"ultra_viewer_calibration_v1": snapshot,
	})

	review, ok := resourceMetadataMap(merged["review"])
	if !ok || review["status"] != "approved" || review["reader"] != "bioio" {
		t.Fatalf("ordinary metadata did not retain deep-merge behavior: %#v", merged["review"])
	}
	calibration, ok := resourceMetadataMap(merged["ultra_viewer_calibration_v1"])
	if !ok || calibration["source_sha256"] != "new-sha" {
		t.Fatalf("new-source calibration snapshot was not replaced: %#v", calibration)
	}
	if _, exists := calibration["stale"]; exists {
		t.Fatalf("stale calibration root survived replacement: %#v", calibration)
	}
}

func TestMergeResourceMetadataPreservesIndependentViewerCalibrationSelections(t *testing.T) {
	existing := domain.JSONMap{
		"version":       1,
		"source_sha256": "same-sha",
		"selections": domain.JSONMap{
			"c0:t0": domain.JSONMap{"threshold_value": 12, "stale": true},
			"c0:t2": domain.JSONMap{"threshold_value": 99},
		},
	}
	patch := domain.JSONMap{
		"version":       1,
		"source_sha256": "same-sha",
		"selections": domain.JSONMap{
			"c0:t0": domain.JSONMap{"threshold_value": 120},
		},
	}

	merged := mergeViewerCalibrationMetadata(existing, patch)
	calibration, ok := resourceMetadataMap(merged)
	if !ok {
		t.Fatalf("calibration type = %T", merged)
	}
	selections, ok := resourceMetadataMap(calibration["selections"])
	if !ok {
		t.Fatalf("selection map missing: %#v", calibration)
	}
	if _, exists := selections["c0:t2"]; !exists {
		t.Fatalf("independent C/T selection was lost: %#v", selections)
	}
	timeZero, _ := resourceMetadataMap(selections["c0:t0"])
	if _, exists := timeZero["stale"]; exists {
		t.Fatalf("patched selection retained stale nested keys: %#v", timeZero)
	}
	if timeZero["threshold_value"] != 120 {
		t.Fatalf("patched selection = %#v", timeZero)
	}
}

func TestMemoryViewerCalibrationCASRejectsReverseCommitOrder(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	memory := NewMemoryStore()
	_, err := memory.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   "file-cas",
		OriginalName: "mask.ome.tiff",
		SHA256:       "source-sha",
		OwnerUserID:  "user-1",
		OwnerOrgID:   "org-1",
		Status:       "active",
	})
	if err != nil {
		t.Fatal(err)
	}
	input := func(threshold int) domain.MergeResourceMetadataInput {
		return domain.MergeResourceMetadataInput{
			ResourceID:                 "file-cas",
			UserID:                     "user-1",
			OrgID:                      "org-1",
			Patch:                      viewerCalibrationPatch("source-sha", 1, threshold),
			ExpectedSourceSHA256:       "source-sha",
			SelectionExpectedRevisions: map[string]int{"c0:t0": 0},
		}
	}
	if _, err := memory.MergeResourceMetadataForUser(ctx, input(200)); err != nil {
		t.Fatalf("newer calibration commit failed: %v", err)
	}
	if _, err := memory.MergeResourceMetadataForUser(ctx, input(100)); !errors.Is(err, ErrConflict) {
		t.Fatalf("stale reverse-order commit error = %v, want conflict", err)
	}
	resource, err := memory.GetResourceForUser(ctx, "file-cas", "user-1", "org-1")
	if err != nil {
		t.Fatal(err)
	}
	calibration, _ := resourceMetadataMap(resource.Metadata["ultra_viewer_calibration_v1"])
	selections, _ := resourceMetadataMap(calibration["selections"])
	selection, _ := resourceMetadataMap(selections["c0:t0"])
	if selection["threshold_value"] != 200 || selection["revision"] != 1 {
		t.Fatalf("persisted calibration = %#v, want newer revision 1", selection)
	}
}

func TestMemoryViewerCalibrationCASRejectsMissingOrReplacedSourceSHA(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	for _, sourceSHA := range []string{"", "replacement-sha"} {
		memory := NewMemoryStore()
		_, err := memory.UpsertResource(ctx, domain.UpsertResourceInput{
			ResourceID:   "file-sha",
			OriginalName: "mask.ome.tiff",
			SHA256:       sourceSHA,
			OwnerUserID:  "user-1",
			OwnerOrgID:   "org-1",
			Status:       "active",
		})
		if err != nil {
			t.Fatal(err)
		}
		_, err = memory.MergeResourceMetadataForUser(
			ctx,
			domain.MergeResourceMetadataInput{
				ResourceID:                 "file-sha",
				UserID:                     "user-1",
				OrgID:                      "org-1",
				Patch:                      viewerCalibrationPatch("source-sha", 1, 120),
				ExpectedSourceSHA256:       "source-sha",
				SelectionExpectedRevisions: map[string]int{"c0:t0": 0},
			},
		)
		if !errors.Is(err, ErrConflict) {
			t.Fatalf("source SHA %q error = %v, want conflict", sourceSHA, err)
		}
	}
}
