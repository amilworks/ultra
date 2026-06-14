package httpapi

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func writeFailureMarker(t *testing.T, root, fileID string, modTime time.Time) string {
	t.Helper()
	p := derivedPyramidFailedMarkerPath(root, fileID)
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(p, []byte(`{"error":"Input format is not supported"}`), 0o644); err != nil {
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
	if recentPyramidFailure(root, "file_none", now) {
		t.Fatal("no marker should not suppress derivation")
	}

	// Fresh marker (just now) -> within the 1h default window -> suppressed.
	writeFailureMarker(t, root, "file_fresh", now)
	if !recentPyramidFailure(root, "file_fresh", now) {
		t.Fatal("a fresh failure marker should suppress re-derivation")
	}

	// Stale marker (2h old) -> past the default window -> retried (not suppressed).
	writeFailureMarker(t, root, "file_stale", now.Add(-2*time.Hour))
	if recentPyramidFailure(root, "file_stale", now) {
		t.Fatal("a marker past the backoff window should allow a retry")
	}
}

func TestPyramidFailureBackoffWindowEnvAndDisable(t *testing.T) {
	root := t.TempDir()
	now := time.Now()
	writeFailureMarker(t, root, "file_x", now)

	// Disabled (0) never suppresses, even for a fresh marker.
	t.Setenv("ULTRA_CONTROL_PYRAMID_FAILURE_BACKOFF_SECONDS", "0")
	if recentPyramidFailure(root, "file_x", now) {
		t.Fatal("backoff window 0 must disable suppression")
	}

	// A custom short window is honored: a 10s-old marker is past a 5s window.
	t.Setenv("ULTRA_CONTROL_PYRAMID_FAILURE_BACKOFF_SECONDS", "5")
	writeFailureMarker(t, root, "file_y", now.Add(-10*time.Second))
	if recentPyramidFailure(root, "file_y", now) {
		t.Fatal("a marker older than the custom window should allow a retry")
	}
}

func TestClearPyramidFailureMarker(t *testing.T) {
	root := t.TempDir()
	p := writeFailureMarker(t, root, "file_clear", time.Now())
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
