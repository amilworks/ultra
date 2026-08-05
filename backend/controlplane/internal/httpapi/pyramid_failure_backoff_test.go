package httpapi

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func failureMarkerRecord(fileID string) resourceRecord {
	return resourceRecord{FileID: fileID, SHA256: strings.Repeat("a", 64), SizeBytes: 123}
}

func writeFailureMarker(t *testing.T, root string, record resourceRecord, modTime time.Time) string {
	t.Helper()
	p := derivedPyramidFailedMarkerPath(root, record.FileID)
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	payload, err := json.Marshal(derivativeFailureMarker{
		Schema: "ultra.image-derived-pyramid-failure.v1", ResourceID: record.FileID,
		SourceSHA256: record.SHA256, SourceSizeBytes: record.SizeBytes,
		ConversionSpec: derivativeConversionSpec{TileSize: 512, Compression: "lzw", Layout: "topdirs", Format: "auto"},
		Code:           "unsupported_source",
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, payload, 0o644); err != nil {
		t.Fatalf("write marker: %v", err)
	}
	if err := os.Chtimes(p, modTime, modTime); err != nil {
		t.Fatalf("chtimes: %v", err)
	}
	return p
}

func TestRecentPyramidFailureBackoff(t *testing.T) {
	root := t.TempDir()
	now := time.Now()

	// No marker -> not suppressed.
	if recentPyramidFailure(root, failureMarkerRecord("file_none"), now) {
		t.Fatal("no marker should not suppress derivation")
	}

	// Fresh marker (just now) -> within the 1h default window -> suppressed.
	fresh := failureMarkerRecord("file_fresh")
	writeFailureMarker(t, root, fresh, now)
	if !recentPyramidFailure(root, fresh, now) {
		t.Fatal("a fresh failure marker should suppress re-derivation")
	}

	// Stale marker (2h old) -> past the default window -> retried (not suppressed).
	stale := failureMarkerRecord("file_stale")
	writeFailureMarker(t, root, stale, now.Add(-2*time.Hour))
	if recentPyramidFailure(root, stale, now) {
		t.Fatal("a marker past the backoff window should allow a retry")
	}
}

func TestPyramidFailureBackoffWindowEnvAndDisable(t *testing.T) {
	root := t.TempDir()
	now := time.Now()
	recordX := failureMarkerRecord("file_x")
	writeFailureMarker(t, root, recordX, now)

	// Disabled (0) never suppresses, even for a fresh marker.
	t.Setenv("ULTRA_CONTROL_PYRAMID_FAILURE_BACKOFF_SECONDS", "0")
	if recentPyramidFailure(root, recordX, now) {
		t.Fatal("backoff window 0 must disable suppression")
	}

	// A custom short window is honored: a 10s-old marker is past a 5s window.
	t.Setenv("ULTRA_CONTROL_PYRAMID_FAILURE_BACKOFF_SECONDS", "5")
	recordY := failureMarkerRecord("file_y")
	writeFailureMarker(t, root, recordY, now.Add(-10*time.Second))
	if recentPyramidFailure(root, recordY, now) {
		t.Fatal("a marker older than the custom window should allow a retry")
	}
}

func TestPyramidFailureMarkerIgnoresDifferentSourceGeneration(t *testing.T) {
	root := t.TempDir()
	now := time.Now()
	record := failureMarkerRecord("file_generation")
	writeFailureMarker(t, root, record, now)
	record.SHA256 = strings.Repeat("b", 64)
	if recentPyramidFailure(root, record, now) {
		t.Fatal("failure marker from a different source generation suppressed derivation")
	}
}

func TestPyramidFailureMarkerRejectsNoncanonicalJSONAndSpecMismatch(t *testing.T) {
	t.Parallel()

	for _, corruption := range []string{"duplicate", "trailing", "unknown", "spec"} {
		t.Run(corruption, func(t *testing.T) {
			root := t.TempDir()
			record := failureMarkerRecord("file_corrupt_" + corruption)
			path := writeFailureMarker(t, root, record, time.Now())
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			switch corruption {
			case "duplicate":
				data = []byte(strings.Replace(string(data), `"code":`, `"code":"duplicate","code":`, 1))
			case "trailing":
				data = append(data, []byte("\n{}")...)
			default:
				var marker map[string]any
				if err := json.Unmarshal(data, &marker); err != nil {
					t.Fatal(err)
				}
				if corruption == "unknown" {
					marker["detail"] = "/private/source/path"
				} else {
					marker["conversion_spec"].(map[string]any)["tile_size"] = float64(256)
				}
				data, err = json.Marshal(marker)
				if err != nil {
					t.Fatal(err)
				}
			}
			if err := os.WriteFile(path, data, 0o644); err != nil {
				t.Fatal(err)
			}
			if recentPyramidFailure(root, record, time.Now()) {
				t.Fatalf("%s failure marker suppressed derivation", corruption)
			}
		})
	}
}

func TestClearPyramidFailureMarker(t *testing.T) {
	root := t.TempDir()
	p := writeFailureMarker(t, root, failureMarkerRecord("file_clear"), time.Now())
	clearPyramidFailureMarker(root, "file_clear")
	if _, err := os.Stat(p); !os.IsNotExist(err) {
		t.Fatalf("marker should be removed, stat err = %v", err)
	}
	// Idempotent: clearing a missing marker is a no-op (no panic/error).
	clearPyramidFailureMarker(root, "file_clear")
}

func TestDerivedPyramidFailedNameMirrorsWorker(t *testing.T) {
	// Must match imaging/worker.py _failure_marker_path: <fileID>__pyramid.failed,
	// next to the <fileID>__pyramid.tif the convert worker would have written.
	if got := derivedPyramidFailedName("file_abc"); got != "file_abc__pyramid.failed" {
		t.Fatalf("derivedPyramidFailedName = %q", got)
	}
}
