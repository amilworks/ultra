package httpapi

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

// RetentionGCStore is the catalog surface the retention GC needs. It returns only
// soft-deleted resources past their undelete window, so the GC can never touch active or
// still-restorable data.
type RetentionGCStore interface {
	ListResourcesPastRetention(context.Context, time.Time, int) ([]domain.ResourceRecord, error)
	PurgeResource(context.Context, string) error
}

// RunEventRetentionStore is the surface the run-event delta retention sweep needs.
type RunEventRetentionStore interface {
	PruneRunEventDeltas(context.Context, time.Time, []string, int) (int64, error)
}

// prunableRunEventDeltaKinds are the per-token, text-stream-only event kinds that bloat the
// control_run_events trace table (a heavy run is ~96% these). They carry no durable meaning beyond
// the live stream — the final answer lives in control_thread_messages and the structural trace
// (run.*, tool_call.*, trace.reasoning.*, artifact.*, run.token_usage) is retained — so they are
// safe to prune for completed runs after a grace TTL.
var prunableRunEventDeltaKinds = []string{"message.delta", "subagent.message.delta"}

// RunRunEventDeltaRetentionGC periodically prunes per-token delta events for runs that completed
// more than `retention` ago. It only ever removes prunableRunEventDeltaKinds from TERMINAL runs, so
// it is safe to run alongside live streaming and reconnect/catch-up (active runs keep their full
// event prefix). Each tick drains the backlog in batches of `batch` rows to bound lock duration.
func RunRunEventDeltaRetentionGC(ctx context.Context, store RunEventRetentionStore, retention, interval time.Duration, batch int) {
	if retention <= 0 {
		return
	}
	if interval <= 0 {
		interval = time.Hour
	}
	if batch <= 0 {
		batch = 1000
	}
	slog.InfoContext(ctx, "run-event delta retention enabled",
		"retention", retention.String(), "interval", interval.String(), "batch", batch)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			cutoff := time.Now().Add(-retention)
			var total int64
			for {
				n, err := store.PruneRunEventDeltas(ctx, cutoff, prunableRunEventDeltaKinds, batch)
				if err != nil {
					slog.WarnContext(ctx, "run-event delta prune failed", "error", err)
					break
				}
				total += n
				if n < int64(batch) {
					break // partial batch => backlog drained for this cycle
				}
			}
			if total > 0 {
				slog.InfoContext(ctx, "run-event delta prune", "deleted_events", total)
			}
		}
	}
}

// resourceArtifactPaths returns every on-disk file that belongs to a resource: the source
// upload, its derived tiled pyramid, any decompressed NIfTI sidecar, and the upload
// metadata sidecar. Only paths UNDER root are returned, so the GC can never delete outside
// the upload store even if a stored path is malformed.
func resourceArtifactPaths(root string, resource domain.ResourceRecord) []string {
	paths := []string{}
	add := func(p string) {
		p = strings.TrimSpace(p)
		if p != "" && pathIsUnderRoot(root, p) {
			paths = append(paths, p)
		}
	}
	source := resourceSourcePath(root, resource)
	add(source)
	add(filepath.Join(root, "derived", derivedPyramidName(resource.ResourceID)))
	add(derivedPyramidFailedMarkerPath(root, resource.ResourceID))
	if source != "" {
		add(niftiDecompressedSidecarPath(source))
	}
	add(uploadMetadataPath(root, resource.ResourceID))
	return paths
}

func resourceSourcePath(root string, resource domain.ResourceRecord) string {
	if candidate := fileStoragePath(resource.StorageURI); candidate != "" && pathIsUnderRoot(root, candidate) {
		return candidate
	}
	storagePath := strings.TrimSpace(resource.StoragePath)
	if storagePath == "" {
		return ""
	}
	if filepath.IsAbs(storagePath) {
		if pathIsUnderRoot(root, storagePath) {
			return filepath.Clean(storagePath)
		}
		return ""
	}
	clean := filepath.Clean(storagePath)
	if clean == "." || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return ""
	}
	candidate := filepath.Join(root, clean)
	if pathIsUnderRoot(root, candidate) {
		return candidate
	}
	return ""
}

// ReclaimExpiredResources permanently removes resources whose undelete window has elapsed:
// it deletes each resource's on-disk artifacts (best-effort) and then purges the catalog
// row (FK cascades drop the related rows). Bounded by batch. Returns the count and bytes
// reclaimed. SAFE: the store returns only soft-deleted, past-retention resources, and only
// paths under root are deleted.
func ReclaimExpiredResources(ctx context.Context, store RetentionGCStore, root string, batch int) (reclaimed int, bytes int64, err error) {
	if batch <= 0 {
		batch = 100
	}
	expired, err := store.ListResourcesPastRetention(ctx, time.Now(), batch)
	if err != nil {
		return 0, 0, err
	}
	for _, resource := range expired {
		var freed int64
		removeFailed := false
		for _, path := range resourceArtifactPaths(root, resource) {
			removedBytes, rmErr := removeResourceArtifact(root, resource, path)
			if rmErr != nil {
				removeFailed = true
				slog.WarnContext(ctx, "retention gc: failed to remove artifact",
					"resource_id", resource.ResourceID, "path", path, "error", rmErr)
				continue
			}
			freed += removedBytes
		}
		if removeFailed {
			continue
		}
		if purgeErr := store.PurgeResource(ctx, resource.ResourceID); purgeErr != nil {
			slog.WarnContext(ctx, "retention gc: failed to purge resource row",
				"resource_id", resource.ResourceID, "error", purgeErr)
			continue
		}
		reclaimed++
		bytes += freed
		slog.InfoContext(ctx, "retention gc: reclaimed expired resource",
			"resource_id", resource.ResourceID, "bytes_freed", freed)
	}
	return reclaimed, bytes, nil
}

func removeResourceArtifact(root string, resource domain.ResourceRecord, path string) (int64, error) {
	info, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil
		}
		return 0, err
	}
	if info.IsDir() {
		if !resourceBundleArtifactDir(root, resource, path) {
			return 0, fmt.Errorf("refusing recursive delete for non-bundle artifact directory")
		}
		bytes := dirSizeBytes(path)
		return bytes, os.RemoveAll(path)
	}
	if err := os.Remove(path); err != nil {
		if os.IsNotExist(err) {
			return 0, nil
		}
		return 0, err
	}
	return info.Size(), nil
}

func resourceBundleArtifactDir(root string, resource domain.ResourceRecord, path string) bool {
	resourceID := strings.TrimSpace(resource.ResourceID)
	if resourceID == "" {
		return false
	}
	bundleRoot := filepath.Join(root, bundlesDirName, resourceID)
	return pathIsUnderRoot(bundleRoot, path)
}

// RunRetentionGC periodically reclaims expired resources until ctx is canceled. Wired from
// the app only when retention GC is explicitly enabled (it permanently deletes data past
// the undelete window).
func RunRetentionGC(ctx context.Context, store RetentionGCStore, root string, interval time.Duration, batch int) {
	if interval <= 0 {
		interval = time.Hour
	}
	slog.InfoContext(ctx, "retention gc enabled", "interval", interval.String(), "batch", batch)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			reclaimed, bytes, err := ReclaimExpiredResources(ctx, store, root, batch)
			if err != nil {
				slog.WarnContext(ctx, "retention gc run failed", "error", err)
				continue
			}
			if reclaimed > 0 {
				slog.InfoContext(ctx, "retention gc run", "reclaimed_resources", reclaimed, "reclaimed_bytes", bytes)
			}
		}
	}
}
