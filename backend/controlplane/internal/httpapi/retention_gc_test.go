package httpapi

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

func TestReclaimExpiredResources(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	outsideRoot := t.TempDir()
	mem := store.NewMemoryStore()
	ctx := context.Background()
	const user, org = "u", "o"

	// mkResource creates a resource + its on-disk source and derived pyramid, then
	// optionally soft-deletes it as of `deletedAgo` ago (so retention_expires_at lands in
	// the past or future relative to the 30-day window).
	mkResource := func(id, name, sourceDir string, deletedAgo time.Duration, active bool) string {
		source := filepath.Join(sourceDir, id+"__"+name)
		if err := os.WriteFile(source, []byte("source bytes for "+id), 0o644); err != nil {
			t.Fatal(err)
		}
		_ = os.MkdirAll(filepath.Join(root, "derived"), 0o755)
		if err := os.WriteFile(filepath.Join(root, "derived", derivedPyramidName(id)), []byte("pyramid "+id), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
			ResourceID:   id,
			OriginalName: name,
			SizeBytes:    100,
			StoragePath:  source,
			OwnerUserID:  user,
			OwnerOrgID:   org,
			Status:       "active",
		}); err != nil {
			t.Fatal(err)
		}
		if !active {
			if _, err := mem.SoftDeleteResourceForUser(ctx, id, user, org, time.Now().Add(-deletedAgo)); err != nil {
				t.Fatal(err)
			}
		}
		return source
	}

	expiredSrc := mkResource("file_expired", "old.tif", root, 31*24*time.Hour, false) // past the 30d window
	freshSrc := mkResource("file_fresh", "recent.tif", root, 1*24*time.Hour, false)   // still within the window
	activeSrc := mkResource("file_active", "live.tif", root, 0, true)                 // active
	// A past-window resource whose source lives OUTSIDE the upload root: the row is
	// purged, but the GC must NEVER delete a file outside root.
	outsideSrc := mkResource("file_outside", "ext.tif", outsideRoot, 31*24*time.Hour, false)

	// Backlog accounts for exactly the two past-window resources before reclaim.
	backlog, err := mem.RetentionBacklog(ctx, time.Now())
	if err != nil {
		t.Fatalf("backlog: %v", err)
	}
	if backlog.Count != 2 {
		t.Fatalf("retention backlog count = %d, want 2 (the two past-window resources)", backlog.Count)
	}

	reclaimed, bytes, err := ReclaimExpiredResources(ctx, mem, root, 100)
	if err != nil {
		t.Fatalf("reclaim: %v", err)
	}
	if reclaimed != 2 {
		t.Fatalf("reclaimed %d resources, want 2 (the past-window ones only)", reclaimed)
	}
	if bytes <= 0 {
		t.Fatalf("reclaimed bytes = %d, want > 0", bytes)
	}

	// Expired (in-root): artifacts deleted, row purged.
	if _, statErr := os.Stat(expiredSrc); !os.IsNotExist(statErr) {
		t.Fatal("expired source was not deleted")
	}
	if _, statErr := os.Stat(filepath.Join(root, "derived", derivedPyramidName("file_expired"))); !os.IsNotExist(statErr) {
		t.Fatal("expired derived pyramid was not deleted")
	}
	if _, getErr := mem.GetResourceForUser(ctx, "file_expired", user, org); getErr == nil {
		t.Fatal("expired resource row was not purged")
	}

	// SAFETY: the outside-root source must survive even though its row is purged.
	if _, statErr := os.Stat(outsideSrc); statErr != nil {
		t.Fatalf("GC deleted a file OUTSIDE the upload root: %v", statErr)
	}

	// Within-window + active resources are untouched.
	if _, statErr := os.Stat(freshSrc); statErr != nil {
		t.Fatal("within-window (still-restorable) source was wrongly deleted")
	}
	if _, statErr := os.Stat(activeSrc); statErr != nil {
		t.Fatal("active source was wrongly deleted")
	}

	// Backlog is empty after reclaim.
	if backlog, err = mem.RetentionBacklog(ctx, time.Now()); err != nil || backlog.Count != 0 {
		t.Fatalf("retention backlog after reclaim = %+v err=%v, want 0", backlog, err)
	}
}

// TestResourceUsageAggregatesScopeAndExcludeDeleted verifies the quota aggregates count
// only ACTIVE resources for the requested owner/project — so deleting frees quota and a
// quota check never mis-counts another user's data.
func TestResourceUsageAggregatesScopeAndExcludeDeleted(t *testing.T) {
	t.Parallel()
	mem := store.NewMemoryStore()
	ctx := context.Background()
	mk := func(id, user, project string, bytes int64, deleted bool) {
		if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
			ResourceID: id, OwnerUserID: user, ProjectID: project, SizeBytes: bytes, Status: "active",
		}); err != nil {
			t.Fatal(err)
		}
		if deleted {
			if _, err := mem.SoftDeleteResourceForUser(ctx, id, user, "", time.Now()); err != nil {
				t.Fatal(err)
			}
		}
	}
	mk("r1", "alice", "proj", 100, false)
	mk("r2", "alice", "proj", 200, false)
	mk("r3", "alice", "proj", 999, true) // deleted -> excluded from quota
	mk("r4", "bob", "other", 50, false)  // different owner + project

	if c, b, _ := mem.ResourceUsageForOwner(ctx, "alice"); c != 2 || b != 300 {
		t.Fatalf("owner usage = %d/%d, want 2/300 (active only, deleted excluded)", c, b)
	}
	if c, b, _ := mem.ResourceUsageForProject(ctx, "proj"); c != 2 || b != 300 {
		t.Fatalf("project usage = %d/%d, want 2/300", c, b)
	}
	if c, b, _ := mem.ResourceUsageForOwner(ctx, "bob"); c != 1 || b != 50 {
		t.Fatalf("bob usage = %d/%d, want 1/50 (scoped to owner)", c, b)
	}
}

func TestReclaimExpiredResourcesRespectsBatch(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	mem := store.NewMemoryStore()
	ctx := context.Background()
	for i := 0; i < 5; i++ {
		id := "file_" + string(rune('a'+i))
		if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
			ResourceID: id, OriginalName: "f.tif", SizeBytes: 10, OwnerUserID: "u", OwnerOrgID: "o", Status: "active",
		}); err != nil {
			t.Fatal(err)
		}
		if _, err := mem.SoftDeleteResourceForUser(ctx, id, "u", "o", time.Now().Add(-31*24*time.Hour)); err != nil {
			t.Fatal(err)
		}
	}
	reclaimed, _, err := ReclaimExpiredResources(ctx, mem, root, 2)
	if err != nil {
		t.Fatalf("reclaim: %v", err)
	}
	if reclaimed != 2 {
		t.Fatalf("reclaimed %d with batch=2, want 2", reclaimed)
	}
}
