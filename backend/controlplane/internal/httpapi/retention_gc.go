package httpapi

import (
	"context"
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
	source := strings.TrimSpace(resource.StoragePath)
	add(source)
	add(filepath.Join(root, "derived", derivedPyramidName(resource.ResourceID)))
	add(derivedPyramidFailedMarkerPath(root, resource.ResourceID))
	if source != "" {
		add(niftiDecompressedSidecarPath(source))
	}
	add(uploadMetadataPath(root, resource.ResourceID))
	return paths
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
		for _, path := range resourceArtifactPaths(root, resource) {
			if info, statErr := os.Stat(path); statErr == nil {
				freed += info.Size()
			}
			if rmErr := os.Remove(path); rmErr != nil && !os.IsNotExist(rmErr) {
				slog.WarnContext(ctx, "retention gc: failed to remove artifact",
					"resource_id", resource.ResourceID, "path", path, "error", rmErr)
			}
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
