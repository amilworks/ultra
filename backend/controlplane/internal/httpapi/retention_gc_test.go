package httpapi

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

func TestDerivativeNameMatcherUsesExactOwnedGrammar(t *testing.T) {
	t.Parallel()
	matcher, err := derivativeNameMatcher("file_a")
	if err != nil {
		t.Fatal(err)
	}
	digest := strings.Repeat("a", 64)
	positives := []string{
		"file_a__pyramid.tif",
		"file_a__pyramid.tif.transcode.ome.tif",
		"file_a__pyramid.failed",
		"file_a__pyramid.manifest.json",
		"file_a__pyramid.sha256-" + digest + ".tif",
		".file_a__pyramid.tmp-abcdefgh.tif",
		".file_a__pyramid.tmp-abcdefgh.tif.transcode.ome.tif",
		".file_a__pyramid.manifest.json.tmp-abcdefgh",
		".file_a__pyramid.manifest.json.rollback-abcdefgh",
		".file_a__pyramid.sha256-" + digest + ".tif.publish-0123456789abcdef01234567",
		".file_a__pyramid.sha256-" + digest + ".tif.recovery-0123456789abcdef01234567",
		".file_a__pyramid.failed.abcdefgh",
		"file_a__nifti.sha256-" + digest + ".nii",
		"file_a__nifti.sha256-" + digest + ".nii.tmp",
		".file_a__nifti.tmp-0123456789abcdef01234567.nii",
		"file_a__scene3d.sha256-" + digest,
		"file_a__scene3d.sha256-" + digest + ".failed",
		".file_a__scene3d.sha256-" + digest + ".tmp-abcdefgh",
		".file_a__scene3d.sha256-" + digest + ".failed.abcdefgh",
		"file_a__scene3d.v2.sha256-" + digest,
		"file_a__scene3d.v2.sha256-" + digest + ".failed",
		".file_a__scene3d.v2.sha256-" + digest + ".tmp-abcdefgh",
		".file_a__scene3d.v2.sha256-" + digest + ".failed.abcdefgh",
		"file_a__scene3d.v3.sha256-" + digest,
		"file_a__scene3d.v3.sha256-" + digest + ".failed",
		".file_a__scene3d.v3.sha256-" + digest + ".tmp-abcdefgh",
		".file_a__scene3d.v3.sha256-" + digest + ".failed.abcdefgh",
		"file_a__scene3d.v4.sha256-" + digest,
		"file_a__scene3d.v4.sha256-" + digest + ".failed",
		".file_a__scene3d.v4.sha256-" + digest + ".tmp-abcdefgh",
		".file_a__scene3d.v4.sha256-" + digest + ".failed.abcdefgh",
	}
	for _, name := range positives {
		if !matcher.MatchString(name) {
			t.Errorf("owned derivative %q did not match", name)
		}
	}
	negatives := []string{
		"file_a__pyramid_shadow.tif",
		"file_ab__pyramid.tif",
		"file_a__pyramid.sha256-" + strings.Repeat("g", 64) + ".tif",
		".file_a__pyramid.sha256-" + digest + ".tif.publish-short",
		"file_a__nifti.nii",
		"file_a__nifti.sha256-" + strings.Repeat("g", 64) + ".nii",
		"file_ab__nifti.sha256-" + digest + ".nii",
		"file_ab__scene3d.sha256-" + digest,
		"file_a__scene3d.sha256-" + strings.Repeat("g", 64),
		"file_a__scene3d.sha256-" + digest + ".tmp-abcdefgh",
		"file_a__scene3d.sha256-" + digest + ".manifest.json",
		"../file_a__pyramid.tif",
		resourceLifecycleLockName("file_a"),
		".file_a__pyramid.work.lock",
		".file_a__nifti.work.lock",
		".file_a__scene3d.work.lock",
	}
	for _, name := range negatives {
		if matcher.MatchString(name) {
			t.Errorf("near-miss derivative %q unexpectedly matched", name)
		}
	}
	for _, resourceID := range []string{".", ".."} {
		if _, err := derivativeNameMatcher(resourceID); err == nil {
			t.Errorf("derivative matcher accepted special path component %q", resourceID)
		}
		if _, err := resourceBundleRelativeRoot(resourceID); err == nil {
			t.Errorf("bundle target accepted special path component %q", resourceID)
		}
	}
}

func TestResourceLifecycleMarkersAreInjectiveForDottedResourceIDs(t *testing.T) {
	t.Parallel()
	rootPath := t.TempDir()
	root, err := os.OpenRoot(rootPath)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()
	const baseID = "file_collision"
	const dottedID = baseID + ".deleted"
	if err := ensureResourceFilesystemTombstone(root, dottedID); err != nil {
		t.Fatal(err)
	}
	if err := ensureResourceFilesystemDeleteFence(root, baseID); err != nil {
		t.Fatal(err)
	}
	if resourceFilesystemTombstoneName(dottedID) == resourceFilesystemDeleteFenceName(baseID) {
		t.Fatal("permanent and reversible marker names alias")
	}
	checks := []struct {
		name string
		got  func() (bool, error)
		want bool
	}{
		{name: "base permanent", got: func() (bool, error) { return resourceFilesystemPermanentlyTombstoned(root, baseID) }, want: false},
		{name: "base reversible", got: func() (bool, error) { return resourceFilesystemSoftDeleteFenced(root, baseID) }, want: true},
		{name: "dotted permanent", got: func() (bool, error) { return resourceFilesystemPermanentlyTombstoned(root, dottedID) }, want: true},
		{name: "dotted reversible", got: func() (bool, error) { return resourceFilesystemSoftDeleteFenced(root, dottedID) }, want: false},
	}
	for _, check := range checks {
		got, err := check.got()
		if err != nil || got != check.want {
			t.Fatalf("%s = %v err=%v, want %v", check.name, got, err, check.want)
		}
	}
	if err := removeResourceFilesystemDeleteFence(root, baseID); err != nil {
		t.Fatal(err)
	}
	if permanent, err := resourceFilesystemPermanentlyTombstoned(root, dottedID); err != nil || !permanent {
		t.Fatalf("removing base reversible fence changed dotted permanent marker: %v err=%v", permanent, err)
	}
}

func TestReconcileResourceLifecycleFencesBackfillsLegacyCatalogStates(t *testing.T) {
	t.Parallel()
	rootPath := t.TempDir()
	mem := store.NewMemoryStore()
	ctx := context.Background()
	for _, fixture := range []struct {
		id     string
		status string
	}{
		{id: "file_active_unfenced", status: domain.ResourceStatusActive},
		{id: "file_legacy_deleted", status: domain.ResourceStatusDeleted},
		{id: "file_legacy_purging", status: domain.ResourceStatusPurging},
		{id: "file_legacy_retention_blocked", status: domain.ResourceStatusRetentionBlocked},
	} {
		if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
			ResourceID: fixture.id, OwnerUserID: "u", OwnerOrgID: "o", Status: fixture.status,
		}); err != nil {
			t.Fatalf("seed %s: %v", fixture.id, err)
		}
	}

	for attempt := 1; attempt <= 2; attempt++ {
		reconciled, err := ReconcileResourceLifecycleFences(ctx, mem, rootPath, 2)
		if err != nil {
			t.Fatalf("reconcile attempt %d: %v", attempt, err)
		}
		if reconciled != 3 {
			t.Fatalf("reconcile attempt %d = %d, want three deletion states", attempt, reconciled)
		}
	}

	root, err := os.OpenRoot(rootPath)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()
	if fenced, err := resourceFilesystemSoftDeleteFenced(root, "file_legacy_deleted"); err != nil || !fenced {
		t.Fatalf("legacy deleted fence = %v err=%v, want reversible fence", fenced, err)
	}
	for _, resourceID := range []string{"file_legacy_purging", "file_legacy_retention_blocked"} {
		if fenced, err := resourceFilesystemPermanentlyTombstoned(root, resourceID); err != nil || !fenced {
			t.Fatalf("legacy terminal fence for %s = %v err=%v, want permanent fence", resourceID, fenced, err)
		}
	}
	if fenced, err := resourceFilesystemTombstoned(root, "file_active_unfenced"); err != nil || fenced {
		t.Fatalf("active resource fence = %v err=%v, want none", fenced, err)
	}
	if lock, err := acquireResourceLifecycleLock(ctx, root, "file_legacy_deleted", ""); !errors.Is(err, errResourceLifecycleTombstoned) {
		if lock != nil {
			_ = lock.release()
		}
		t.Fatalf("publisher lock after legacy backfill error = %v, want deletion fence", err)
	}
}

func TestReconcileResourceLifecycleFencesRechecksStateUnderLock(t *testing.T) {
	t.Parallel()
	rootPath := t.TempDir()
	mem := store.NewMemoryStore()
	ctx := context.Background()
	const resourceID = "file_restored_during_backfill"
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID: resourceID, OwnerUserID: "u", OwnerOrgID: "o", Status: domain.ResourceStatusActive,
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := mem.SoftDeleteResourceForUser(ctx, resourceID, "u", "o", time.Now()); err != nil {
		t.Fatal(err)
	}
	flipping := &restoreDuringLifecycleListStore{MemoryStore: mem, resourceID: resourceID}
	reconciled, err := ReconcileResourceLifecycleFences(ctx, flipping, rootPath, 10)
	if err != nil {
		t.Fatal(err)
	}
	if reconciled != 0 || !flipping.restored {
		t.Fatalf("reconciled = %d restored=%v, want stale candidate skipped after active re-read", reconciled, flipping.restored)
	}
	root, err := os.OpenRoot(rootPath)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()
	if fenced, err := resourceFilesystemTombstoned(root, resourceID); err != nil || fenced {
		t.Fatalf("restored resource fence = %v err=%v, want none", fenced, err)
	}
}

func TestReconcileResourceLifecycleFencesHealsActiveSoftDeleteCrashBoundaries(t *testing.T) {
	t.Parallel()
	for _, scenario := range []string{"delete_before_catalog_commit", "restore_after_catalog_commit"} {
		t.Run(scenario, func(t *testing.T) {
			t.Parallel()
			rootPath := t.TempDir()
			root, err := os.OpenRoot(rootPath)
			if err != nil {
				t.Fatal(err)
			}
			defer root.Close()
			mem := store.NewMemoryStore()
			ctx := context.Background()
			resourceID := "file_active_soft_fence_" + scenario
			if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
				ResourceID: resourceID, OwnerUserID: "u", OwnerOrgID: "o", Status: domain.ResourceStatusActive,
			}); err != nil {
				t.Fatal(err)
			}
			if scenario == "restore_after_catalog_commit" {
				if _, err := mem.SoftDeleteResourceForUser(ctx, resourceID, "u", "o", time.Now()); err != nil {
					t.Fatal(err)
				}
			}
			if err := ensureResourceFilesystemDeleteFence(root, resourceID); err != nil {
				t.Fatal(err)
			}
			if scenario == "restore_after_catalog_commit" {
				if _, err := mem.RestoreResourceForUser(ctx, resourceID, "u", "o", time.Now()); err != nil {
					t.Fatal(err)
				}
			}
			reconciled, err := ReconcileResourceLifecycleFences(ctx, mem, rootPath, 1)
			if err != nil {
				t.Fatal(err)
			}
			if reconciled != 1 {
				t.Fatalf("reconciled = %d, want one active soft-delete fence repair", reconciled)
			}
			if fenced, err := resourceFilesystemSoftDeleteFenced(root, resourceID); err != nil || fenced {
				t.Fatalf("active soft-delete fence = %v err=%v, want removed", fenced, err)
			}
		})
	}
}

func TestLifecycleFenceRepairHealsCrashBoundaryOnSurvivingReplica(t *testing.T) {
	t.Parallel()
	rootPath := t.TempDir()
	root, err := os.OpenRoot(rootPath)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()
	mem := store.NewMemoryStore()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	const resourceID = "file_surviving_replica_fence"
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID: resourceID, OwnerUserID: "u", OwnerOrgID: "o", Status: domain.ResourceStatusActive,
	}); err != nil {
		t.Fatal(err)
	}

	go RunResourceLifecycleFenceRepair(ctx, mem, rootPath, 5*time.Millisecond)
	if err := ensureResourceFilesystemDeleteFence(root, resourceID); err != nil {
		t.Fatal(err)
	}

	deadline := time.Now().Add(time.Second)
	for {
		fenced, err := resourceFilesystemSoftDeleteFenced(root, resourceID)
		if err != nil {
			t.Fatal(err)
		}
		if !fenced {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("surviving replica did not heal active soft-delete fence")
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func TestSoftDeleteFenceRepairContinuesPastBusyResource(t *testing.T) {
	rootPath := t.TempDir()
	root, err := os.OpenRoot(rootPath)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()
	mem := store.NewMemoryStore()
	ctx := context.Background()
	const busyID = "file_a_busy_soft_fence"
	const healID = "file_b_active_soft_fence"
	for _, resourceID := range []string{busyID, healID} {
		if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
			ResourceID:  resourceID,
			OwnerUserID: "u",
			OwnerOrgID:  "o",
			Status:      domain.ResourceStatusActive,
		}); err != nil {
			t.Fatal(err)
		}
		if err := ensureResourceFilesystemDeleteFence(root, resourceID); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := mem.SoftDeleteResourceForUser(ctx, busyID, "u", "o", time.Now()); err != nil {
		t.Fatal(err)
	}
	held, err := acquireResourceLifecycleMutationLock(ctx, root, busyID)
	if err != nil {
		t.Fatal(err)
	}
	defer held.release() //nolint:errcheck // test cleanup

	reconciled, reconcileErr := ReconcileResourceLifecycleSoftDeleteFences(
		ctx, mem, rootPath,
	)
	if reconcileErr == nil || !strings.Contains(reconcileErr.Error(), busyID) {
		t.Fatalf("reconciliation error = %v, want isolated busy-resource error", reconcileErr)
	}
	if reconciled != 1 {
		t.Fatalf("reconciled = %d, want later active resource healed", reconciled)
	}
	if fenced, err := resourceFilesystemSoftDeleteFenced(root, healID); err != nil || fenced {
		t.Fatalf("later active resource fence = %v err=%v, want removed", fenced, err)
	}
	if fenced, err := resourceFilesystemSoftDeleteFenced(root, busyID); err != nil || !fenced {
		t.Fatalf("busy deleted resource fence = %v err=%v, want retained", fenced, err)
	}
}

func TestLifecycleFenceStartupContinuesPastMalformedMarker(t *testing.T) {
	rootPath := t.TempDir()
	root, err := os.OpenRoot(rootPath)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()
	mem := store.NewMemoryStore()
	ctx := context.Background()
	const malformedID = "file_a_malformed_soft_fence"
	const validID = "file_b_valid_soft_fence"
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID: validID, OwnerUserID: "u", OwnerOrgID: "o", Status: domain.ResourceStatusActive,
	}); err != nil {
		t.Fatal(err)
	}
	if err := ensureResourceFilesystemDeleteFence(root, validID); err != nil {
		t.Fatal(err)
	}
	malformed := filepath.Join(
		rootPath, resourceTombstoneDir, resourceSoftDeleteFenceDir, malformedID,
	)
	if err := os.Mkdir(malformed, 0o700); err != nil {
		t.Fatal(err)
	}

	reconciled, reconcileErr := ReconcileResourceLifecycleFences(ctx, mem, rootPath, 1)

	if reconcileErr != nil {
		t.Fatalf("entry-local malformed marker made reconciliation fatal: %v", reconcileErr)
	}
	if reconciled != 1 {
		t.Fatalf("reconciled = %d, want valid marker repaired", reconciled)
	}
	if fenced, err := resourceFilesystemSoftDeleteFenced(root, validID); err != nil || fenced {
		t.Fatalf("valid active fence = %v err=%v, want removed", fenced, err)
	}
	if info, err := os.Stat(malformed); err != nil || !info.IsDir() {
		t.Fatalf("malformed marker changed: info=%v err=%v", info, err)
	}
}

func TestTerminalReconciliationUsesBoundedBatches(t *testing.T) {
	rootPath := t.TempDir()
	root, err := os.OpenRoot(rootPath)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()
	ctx := context.Background()
	for index := 0; index < resourceTerminalCleanupBatch+1; index++ {
		resourceID := fmt.Sprintf("file_terminal_batch_%03d", index)
		if err := ensureResourceFilesystemTombstone(root, resourceID); err != nil {
			t.Fatal(err)
		}
	}

	reconciled, reconcileErr := ReconcileResourceLifecyclePermanentTombstones(ctx, rootPath)
	if reconcileErr != nil {
		t.Fatal(reconcileErr)
	}
	if reconciled != resourceTerminalCleanupBatch {
		t.Fatalf("first terminal batch = %d, want %d", reconciled, resourceTerminalCleanupBatch)
	}
	remaining, err := os.ReadDir(filepath.Join(
		rootPath,
		resourceTombstoneDir,
		resourceTerminalCleanupDir,
	))
	if err != nil {
		t.Fatal(err)
	}
	if len(remaining) != 1 {
		t.Fatalf("terminal markers remaining = %d, want 1", len(remaining))
	}

	reconciled, reconcileErr = ReconcileResourceLifecyclePermanentTombstones(ctx, rootPath)
	if reconcileErr != nil || reconciled != 1 {
		t.Fatalf("second terminal batch = %d err=%v, want final marker", reconciled, reconcileErr)
	}
}

func TestTerminalReconciliationDrainsLatePublisherStages(t *testing.T) {
	rootPath := t.TempDir()
	root, err := os.OpenRoot(rootPath)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()
	ctx := context.Background()
	const resourceID = "file_terminal_late_stage"
	held, err := acquireResourceDerivationLock(ctx, root, resourceID, "pyramid", "")
	if err != nil {
		t.Fatal(err)
	}
	if err := ensureResourceFilesystemTombstone(root, resourceID); err != nil {
		_ = held.release()
		t.Fatal(err)
	}
	staging := filepath.Join(rootPath, resourceStagingDir, resourceID, "pyramid")
	if err := os.MkdirAll(staging, 0o700); err != nil {
		_ = held.release()
		t.Fatal(err)
	}
	latePyramid := filepath.Join(staging, "artifact.tif")
	if err := os.WriteFile(latePyramid, []byte("late-publisher-stage"), 0o600); err != nil {
		_ = held.release()
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(rootPath, resourceDerivedDir), 0o755); err != nil {
		_ = held.release()
		t.Fatal(err)
	}
	lateNifti := filepath.Join(rootPath, resourceDerivedDir, "."+resourceID+"__nifti.stage.nii")
	if err := os.WriteFile(lateNifti, []byte("late-nifti-stage"), 0o600); err != nil {
		_ = held.release()
		t.Fatal(err)
	}

	blockedCtx, cancel := context.WithTimeout(ctx, 50*time.Millisecond)
	reconciled, reconcileErr := ReconcileResourceLifecyclePermanentTombstones(
		blockedCtx, rootPath,
	)
	cancel()
	if reconciled != 0 || reconcileErr == nil {
		t.Fatalf("blocked reconciliation = %d err=%v, want retryable busy result", reconciled, reconcileErr)
	}
	for _, path := range []string{latePyramid, lateNifti} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("busy producer stage %q changed before drain: %v", path, err)
		}
	}
	if err := held.release(); err != nil {
		t.Fatal(err)
	}

	reconciled, reconcileErr = ReconcileResourceLifecyclePermanentTombstones(ctx, rootPath)
	if reconcileErr != nil || reconciled != 1 {
		t.Fatalf("terminal reconciliation = %d err=%v, want one cleanup", reconciled, reconcileErr)
	}
	for _, path := range []string{
		latePyramid,
		lateNifti,
		filepath.Join(rootPath, resourceStagingDir, resourceID),
		filepath.Join(rootPath, resourceTombstoneDir, resourceTerminalCleanupDir, resourceID),
	} {
		if _, err := os.Lstat(path); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("terminal path survived %q: %v", path, err)
		}
	}
	if _, err := os.Stat(filepath.Join(
		rootPath, resourceTombstoneDir, resourceFilesystemTombstoneName(resourceID),
	)); err != nil {
		t.Fatalf("durable permanent tombstone was removed: %v", err)
	}
	for _, kind := range []string{"pyramid", "nifti"} {
		name, _ := resourceDerivationLockName(resourceID, kind)
		if _, err := os.Lstat(filepath.Join(rootPath, resourceLockDir, name)); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("%s work lock survived terminal cleanup: %v", kind, err)
		}
	}
}

func TestConcurrentLifecycleFenceReconcilersHealActiveMarkerIdempotently(t *testing.T) {
	t.Parallel()
	rootPath := t.TempDir()
	root, err := os.OpenRoot(rootPath)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()
	mem := store.NewMemoryStore()
	ctx := context.Background()
	const resourceID = "file_concurrent_active_fence"
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID: resourceID, OwnerUserID: "u", OwnerOrgID: "o", Status: domain.ResourceStatusActive,
	}); err != nil {
		t.Fatal(err)
	}
	if err := ensureResourceFilesystemDeleteFence(root, resourceID); err != nil {
		t.Fatal(err)
	}
	start := make(chan struct{})
	results := make(chan error, 2)
	for range 2 {
		go func() {
			<-start
			_, err := ReconcileResourceLifecycleFences(ctx, mem, rootPath, 1)
			results <- err
		}()
	}
	close(start)
	for range 2 {
		if err := <-results; err != nil {
			t.Fatal(err)
		}
	}
	if fenced, err := resourceFilesystemSoftDeleteFenced(root, resourceID); err != nil || fenced {
		t.Fatalf("active soft-delete fence = %v err=%v, want removed", fenced, err)
	}
}

func TestReclaimExternalResourcePublishesPermanentFenceBeforeCatalogBlock(t *testing.T) {
	t.Parallel()
	rootPath := t.TempDir()
	outsideSource := filepath.Join(t.TempDir(), "external.tif")
	if err := os.WriteFile(outsideSource, []byte("external"), 0o644); err != nil {
		t.Fatal(err)
	}
	mem := store.NewMemoryStore()
	ctx := context.Background()
	const resourceID = "file_external_fence_order"
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID: resourceID, StorageURI: fileStorageURI(outsideSource), SizeBytes: 8,
		OwnerUserID: "u", OwnerOrgID: "o", Status: domain.ResourceStatusActive,
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := mem.SoftDeleteResourceForUser(ctx, resourceID, "u", "o", time.Now().Add(-31*24*time.Hour)); err != nil {
		t.Fatal(err)
	}
	observing := &fenceObservingRetentionStore{MemoryStore: mem, root: rootPath}
	reclaimed, bytes, err := ReclaimExpiredResources(ctx, observing, rootPath, 1)
	if err != nil {
		t.Fatal(err)
	}
	if reclaimed != 0 || bytes != 0 || !observing.permanentFenceObserved {
		t.Fatalf("external classification = %d/%d fence=%v, want blocked after durable fence", reclaimed, bytes, observing.permanentFenceObserved)
	}
	if payload, err := os.ReadFile(outsideSource); err != nil || string(payload) != "external" {
		t.Fatalf("external source changed: %q err=%v", payload, err)
	}
	root, err := os.OpenRoot(rootPath)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()
	if lock, err := acquireResourceLifecycleLock(ctx, root, resourceID, ""); !errors.Is(err, errResourceLifecycleTombstoned) {
		if lock != nil {
			_ = lock.release()
		}
		t.Fatalf("publisher lock after external block error = %v, want permanent fence", err)
	}
}

func TestScanOwnedDerivativeNamesForResourcesInventoriesBatchOnce(t *testing.T) {
	t.Parallel()
	rootPath := t.TempDir()
	if err := os.MkdirAll(filepath.Join(rootPath, resourceDerivedDir), 0o755); err != nil {
		t.Fatal(err)
	}
	digest := strings.Repeat("d", 64)
	want := map[string][]string{
		"file_a": {
			".file_a__pyramid.failed.abcdefgh",
			"file_a__pyramid.sha256-" + digest + ".tif",
			"file_a__scene3d.sha256-" + digest,
			"file_a__scene3d.v2.sha256-" + digest,
			"file_a__scene3d.v3.sha256-" + digest,
			"file_a__scene3d.v4.sha256-" + digest,
		},
		"file_a__pyramid_beta": {
			"file_a__pyramid_beta__pyramid.manifest.json",
		},
	}
	for _, names := range want {
		for _, name := range names {
			if err := os.WriteFile(filepath.Join(rootPath, resourceDerivedDir, name), []byte(name), 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
	if err := os.WriteFile(filepath.Join(rootPath, resourceDerivedDir, "file_a__pyramid_shadow.tif"), []byte("neighbor"), 0o644); err != nil {
		t.Fatal(err)
	}
	root, err := os.OpenRoot(rootPath)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()
	got, err := scanOwnedDerivativeNamesForResources(root, []string{"file_a", "file_a__pyramid_beta"})
	if err != nil {
		t.Fatal(err)
	}
	for resourceID, names := range want {
		if fmt.Sprint(got[resourceID]) != fmt.Sprint(names) {
			t.Fatalf("inventory[%s] = %v, want %v", resourceID, got[resourceID], names)
		}
	}
}

func TestReclaimExpiredResourcesRemovesLeadingDotResourceWithoutTouchingNeighbor(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	const resourceID = ".file_a"
	const neighborID = "file_a"
	source := filepath.Join(root, resourceID+"__scan.tif")
	neighborSource := filepath.Join(root, neighborID+"__scan.tif")
	for path, payload := range map[string][]byte{
		source:         []byte("owned source"),
		neighborSource: []byte("neighbor source"),
	} {
		if err := os.WriteFile(path, payload, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := writeUploadMetadata(root, resourceID, requestPrincipal{UserID: "u", OrgID: "o"}); err != nil {
		t.Fatal(err)
	}
	derived := filepath.Join(root, resourceDerivedDir)
	if err := os.MkdirAll(derived, 0o755); err != nil {
		t.Fatal(err)
	}
	digest := strings.Repeat("a", 64)
	owned := []string{
		resourceID + "__pyramid.tif",
		resourceID + "__pyramid.sha256-" + digest + ".tif",
		resourceID + "__pyramid.manifest.json",
	}
	neighbors := []string{
		neighborID + "__pyramid.tif",
		neighborID + "__pyramid.sha256-" + digest + ".tif",
		neighborID + "__pyramid.manifest.json",
	}
	for _, name := range append(append([]string{}, owned...), neighbors...) {
		if err := os.WriteFile(filepath.Join(derived, name), []byte(name), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	ctx := context.Background()
	mem := store.NewMemoryStore()
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   resourceID,
		OriginalName: "scan.tif",
		StorageURI:   fileStorageURI(source),
		StoragePath:  filepath.Base(source),
		OwnerUserID:  "u",
		OwnerOrgID:   "o",
		Status:       domain.ResourceStatusActive,
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := mem.SoftDeleteResourceForUser(ctx, resourceID, "u", "o", time.Now().Add(-31*24*time.Hour)); err != nil {
		t.Fatal(err)
	}
	reclaimed, bytes, err := ReclaimExpiredResources(ctx, mem, root, 10)
	if err != nil {
		t.Fatal(err)
	}
	if reclaimed != 1 || bytes <= 0 {
		t.Fatalf("reclaimed = %d/%d, want one leading-dot resource with positive bytes", reclaimed, bytes)
	}
	for _, path := range append([]string{source, uploadMetadataPath(root, resourceID)}, pathsInDir(derived, owned)...) {
		if _, statErr := os.Lstat(path); !errors.Is(statErr, os.ErrNotExist) {
			t.Errorf("owned leading-dot path survived cleanup: %s (err=%v)", path, statErr)
		}
	}
	for _, path := range append([]string{neighborSource}, pathsInDir(derived, neighbors)...) {
		if _, statErr := os.Lstat(path); statErr != nil {
			t.Errorf("neighbor path changed: %s (err=%v)", path, statErr)
		}
	}
}

func TestReclaimExpiredNiftiDoesNotDeleteNestedResourceAtLegacySidecarPath(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	const deletedID = "file_nested_gzip"
	const activeID = "file_nested_plain"
	nestedRoot := filepath.Join(root, "analysis", "job")
	deletedSource := filepath.Join(nestedRoot, "foo.nii.gz")
	activeSource := filepath.Join(nestedRoot, resourceDerivedDir, "foo.nii")
	if err := os.MkdirAll(filepath.Dir(activeSource), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(deletedSource, []byte("owned gzip source"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(activeSource, []byte("independent active source"), 0o644); err != nil {
		t.Fatal(err)
	}
	digest := strings.Repeat("c", 64)
	identity := niftiDecompressedSidecarIdentity{
		root: root, resourceID: deletedID, sourceSHA256: digest,
	}
	sidecar, ok := niftiDecompressedSidecarPath(identity)
	if !ok {
		t.Fatal("valid sidecar identity was rejected")
	}
	if sidecar == activeSource {
		t.Fatal("resource-bound sidecar aliases a nested authoritative source")
	}
	if err := os.MkdirAll(filepath.Dir(sidecar), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{sidecar, sidecar + ".tmp"} {
		if err := os.WriteFile(path, []byte("owned sidecar"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	ctx := context.Background()
	mem := store.NewMemoryStore()
	for _, input := range []domain.UpsertResourceInput{
		{
			ResourceID: deletedID, OriginalName: "foo.nii.gz", StorageURI: fileStorageURI(deletedSource),
			StoragePath: filepath.Join("analysis", "job", "foo.nii.gz"), OwnerUserID: "u", OwnerOrgID: "o",
			Status: domain.ResourceStatusActive,
		},
		{
			ResourceID: activeID, OriginalName: "foo.nii", StorageURI: fileStorageURI(activeSource),
			StoragePath: filepath.Join("analysis", "job", resourceDerivedDir, "foo.nii"), OwnerUserID: "u", OwnerOrgID: "o",
			Status: domain.ResourceStatusActive,
		},
	} {
		if _, err := mem.UpsertResource(ctx, input); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := mem.SoftDeleteResourceForUser(ctx, deletedID, "u", "o", time.Now().Add(-31*24*time.Hour)); err != nil {
		t.Fatal(err)
	}
	reclaimed, _, err := ReclaimExpiredResources(ctx, mem, root, 10)
	if err != nil {
		t.Fatal(err)
	}
	if reclaimed != 1 {
		t.Fatalf("reclaimed resources = %d, want 1", reclaimed)
	}
	for _, path := range []string{deletedSource, sidecar, sidecar + ".tmp"} {
		if _, statErr := os.Lstat(path); !errors.Is(statErr, os.ErrNotExist) {
			t.Errorf("deleted resource path survived cleanup: %s (err=%v)", path, statErr)
		}
	}
	if payload, err := os.ReadFile(activeSource); err != nil || string(payload) != "independent active source" {
		t.Fatalf("nested active resource changed: %q err=%v", payload, err)
	}
}

func TestReclaimExpiredResourcesRejectsSpecialResourceIDWithoutDeletingBundles(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	ctx := context.Background()
	source := filepath.Join(root, bundlesDirName, "victim", "scan.ome.zarr", "0", "0")
	neighbor := filepath.Join(root, bundlesDirName, "neighbor", "keep.txt")
	for path, payload := range map[string][]byte{source: []byte("victim"), neighbor: []byte("neighbor")} {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, payload, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	claimedAt := time.Now().UTC()
	stub := &retentionGCStub{claims: []domain.ResourceRecord{{
		ResourceID: ".", StorageURI: fileStorageURI(source), Status: domain.ResourceStatusPurging, UpdatedAt: claimedAt,
	}}}

	reclaimed, bytes, err := ReclaimExpiredResources(ctx, stub, root, 10)
	if err != nil {
		t.Fatal(err)
	}
	if reclaimed != 0 || bytes != 0 {
		t.Fatalf("reclaimed = %d/%d, want 0/0 for unsafe resource ID", reclaimed, bytes)
	}
	for _, path := range []string{source, neighbor} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("bundle sentinel %q was changed: %v", path, err)
		}
	}
	if len(stub.blocked) != 1 || stub.blocked[0] != "." {
		t.Fatalf("unsafe legacy resource was not durably blocked: %+v", stub.blocked)
	}
}

func TestRetentionSweepContinuesPastClassificationCapForManagedFileURI(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	const batch = 2
	poisonCount := resourceRetentionClassificationPages*batch + 1
	claims := make([]domain.ResourceRecord, 0, poisonCount+1)
	claimedAt := time.Now().UTC()
	for index := range poisonCount {
		resourceID := fmt.Sprintf("file_outside_%03d", index)
		claims = append(claims, domain.ResourceRecord{
			ResourceID:  resourceID,
			StorageURI:  fmt.Sprintf("file:///outside/%s__source.tif", resourceID),
			StoragePath: resourceID + "__misleading.tif",
			Status:      domain.ResourceStatusPurging,
			UpdatedAt:   claimedAt.Add(time.Duration(index) * time.Nanosecond),
		})
	}
	const managedID = "file_managed_after_cap"
	managedSource := filepath.Join(root, managedID+"__source.tif")
	if err := os.WriteFile(managedSource, []byte("managed"), 0o644); err != nil {
		t.Fatal(err)
	}
	claims = append(claims, domain.ResourceRecord{
		ResourceID:  managedID,
		StorageURI:  fileStorageURI(managedSource),
		StoragePath: filepath.Base(managedSource),
		SizeBytes:   int64(len("managed")),
		Status:      domain.ResourceStatusPurging,
		UpdatedAt:   claimedAt.Add(time.Duration(poisonCount) * time.Nanosecond),
	})
	stub := &retentionGCStub{claims: claims}
	reclaimed, bytes, err := reclaimRetentionBacklog(context.Background(), stub, root, batch)
	if err != nil {
		t.Fatal(err)
	}
	if reclaimed != 1 || bytes != int64(len("managed")) {
		t.Fatalf("retention sweep reclaimed %d/%d, want managed resource 1/%d", reclaimed, bytes, len("managed"))
	}
	if len(stub.blocked) != poisonCount {
		t.Fatalf("blocked outside-file resources = %d, want %d", len(stub.blocked), poisonCount)
	}
	if len(stub.purged) != 1 || stub.purged[0] != managedID {
		t.Fatalf("purged resources = %+v, want %s", stub.purged, managedID)
	}
	if _, err := os.Stat(managedSource); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("managed source survived sweep: %v", err)
	}
}

func TestReclaimExpiredResourcesDeletesStrictDerivativeNamespace(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	outside := filepath.Join(t.TempDir(), "outside-sentinel")
	if err := os.WriteFile(outside, []byte("survive"), 0o644); err != nil {
		t.Fatal(err)
	}
	const resourceID = "file_strict"
	source := filepath.Join(root, resourceID+"__scan.nii.gz")
	if err := os.WriteFile(source, []byte("source"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := writeUploadMetadata(root, resourceID, requestPrincipal{UserID: "u", OrgID: "o"}); err != nil {
		t.Fatal(err)
	}
	derived := filepath.Join(root, resourceDerivedDir)
	if err := os.MkdirAll(derived, 0o755); err != nil {
		t.Fatal(err)
	}
	digest := strings.Repeat("a", 64)
	artifactName := resourceID + "__pyramid.sha256-" + digest + ".tif"
	artifactPath := filepath.Join(derived, artifactName)
	if err := os.WriteFile(artifactPath, []byte("immutable-artifact"), 0o644); err != nil {
		t.Fatal(err)
	}
	ownedFiles := []string{
		resourceID + "__pyramid.tif",
		resourceID + "__pyramid.tif.transcode.ome.tif",
		resourceID + "__pyramid.failed",
		resourceID + "__pyramid.manifest.json",
		"." + resourceID + "__pyramid.tmp-abcdefgh.tif",
		"." + resourceID + "__pyramid.tmp-abcdefgh.tif.transcode.ome.tif",
		"." + resourceID + "__pyramid.manifest.json.tmp-abcdefgh",
		"." + resourceID + "__pyramid.manifest.json.rollback-abcdefgh",
		"." + resourceID + "__pyramid.failed.abcdefgh",
		"." + resourceID + "__nifti.tmp-0123456789abcdef01234567.nii",
	}
	for _, name := range ownedFiles {
		payload := []byte("owned")
		if strings.HasSuffix(name, ".manifest.json") {
			payload = []byte(`{"artifact":{"basename":"../../outside-sentinel"}}`)
		}
		if err := os.WriteFile(filepath.Join(derived, name), payload, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for _, name := range []string{
		"." + resourceID + "__pyramid.sha256-" + digest + ".tif.publish-0123456789abcdef01234567",
		"." + resourceID + "__pyramid.sha256-" + digest + ".tif.recovery-0123456789abcdef01234567",
	} {
		if err := os.Link(artifactPath, filepath.Join(derived, name)); err != nil {
			t.Fatal(err)
		}
	}
	niftiSidecar := filepath.Join(derived, resourceID+"__scan.nii")
	if err := os.WriteFile(niftiSidecar, []byte("sidecar"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(niftiSidecar+".tmp", []byte("partial"), 0o644); err != nil {
		t.Fatal(err)
	}
	leafSymlink := filepath.Join(derived, resourceID+"__pyramid.sha256-"+strings.Repeat("b", 64)+".tif")
	if err := os.Symlink(outside, leafSymlink); err != nil {
		t.Fatal(err)
	}
	neighbor := filepath.Join(derived, resourceID+"__pyramid_shadow.sha256-"+digest+".tif")
	if err := os.WriteFile(neighbor, []byte("neighbor"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, resourceLockDir), 0o700); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"pyramid", "nifti"} {
		name, err := resourceDerivationLockName(resourceID, kind)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, resourceLockDir, name), []byte("lock"), 0o600); err != nil {
			t.Fatal(err)
		}
	}

	ctx := context.Background()
	mem := store.NewMemoryStore()
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   resourceID,
		OriginalName: "scan.nii.gz",
		StorageURI:   fileStorageURI(source),
		StoragePath:  filepath.Base(source),
		OwnerUserID:  "u",
		OwnerOrgID:   "o",
		Status:       "active",
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := mem.SoftDeleteResourceForUser(ctx, resourceID, "u", "o", time.Now().Add(-31*24*time.Hour)); err != nil {
		t.Fatal(err)
	}
	reclaimed, bytes, err := ReclaimExpiredResources(ctx, mem, root, 10)
	if err != nil {
		t.Fatal(err)
	}
	if reclaimed != 1 || bytes <= 0 {
		t.Fatalf("reclaimed = %d/%d, want one resource with positive bytes", reclaimed, bytes)
	}
	for _, path := range append([]string{source, artifactPath, niftiSidecar, niftiSidecar + ".tmp", leafSymlink}, pathsInDir(derived, ownedFiles)...) {
		if _, statErr := os.Lstat(path); !errors.Is(statErr, os.ErrNotExist) {
			t.Errorf("owned path survived cleanup: %s (err=%v)", path, statErr)
		}
	}
	if _, err := os.Stat(neighbor); err != nil {
		t.Fatalf("prefix-collision neighbor was deleted: %v", err)
	}
	if _, err := os.Stat(outside); err != nil {
		t.Fatalf("symlink target or malicious-manifest target was deleted: %v", err)
	}
	if _, err := os.Stat(filepath.Join(root, resourceMetaDir, resourceID+".json")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("metadata survived cleanup: %v", err)
	}
	if _, err := os.Stat(filepath.Join(root, resourceLockDir, resourceLifecycleLockName(resourceID))); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("lifecycle lock survived successful purge: %v", err)
	}
	if reconciled, err := ReconcileResourceLifecyclePermanentTombstones(ctx, root); err != nil || reconciled != 1 {
		t.Fatalf("terminal cleanup = %d err=%v, want purged namespace finalized", reconciled, err)
	}
	for _, kind := range []string{"pyramid", "nifti"} {
		name, _ := resourceDerivationLockName(resourceID, kind)
		if _, err := os.Stat(filepath.Join(root, resourceLockDir, name)); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("%s work lock survived successful purge: %v", kind, err)
		}
	}
}

func pathsInDir(directory string, names []string) []string {
	paths := make([]string, 0, len(names))
	for _, name := range names {
		paths = append(paths, filepath.Join(directory, name))
	}
	return paths
}

func TestRemoveUploadedFileUsesExactSourceAndDerivativeIdentity(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	const resourceID = "file_a"
	source := filepath.Join(root, resourceID+"__cells.tif")
	neighborSource := filepath.Join(root, "file_a_shadow__cells.tif")
	for path, data := range map[string][]byte{
		source:         []byte("owned source"),
		neighborSource: []byte("neighbor source"),
	} {
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := writeUploadMetadata(root, resourceID, requestPrincipal{UserID: "u"}); err != nil {
		t.Fatal(err)
	}
	derived := filepath.Join(root, resourceDerivedDir)
	if err := os.MkdirAll(derived, 0o755); err != nil {
		t.Fatal(err)
	}
	digest := strings.Repeat("c", 64)
	ownedDerivative := filepath.Join(derived, resourceID+"__pyramid.sha256-"+digest+".tif")
	neighborDerivative := filepath.Join(derived, resourceID+"__pyramid_shadow.sha256-"+digest+".tif")
	if err := os.WriteFile(ownedDerivative, []byte("owned derivative"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(neighborDerivative, []byte("neighbor derivative"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := removeUploadedFile(context.Background(), root, resourceID, source); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{source, ownedDerivative, uploadMetadataPath(root, resourceID)} {
		if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
			t.Errorf("owned path survived rollback cleanup: %s (err=%v)", path, err)
		}
	}
	for _, path := range []string{neighborSource, neighborDerivative} {
		if _, err := os.Stat(path); err != nil {
			t.Errorf("neighbor path was deleted: %s (err=%v)", path, err)
		}
	}
	if _, err := os.Stat(filepath.Join(root, resourceTombstoneDir, resourceFilesystemTombstoneName(resourceID))); err != nil {
		t.Fatalf("permanent publication fence missing after uncataloged cleanup: %v", err)
	}
	uploadRoot, err := os.OpenRoot(root)
	if err != nil {
		t.Fatal(err)
	}
	defer uploadRoot.Close()
	lateLock, err := acquireResourceLifecycleLock(context.Background(), uploadRoot, resourceID, "")
	if lateLock != nil {
		_ = lateLock.release()
	}
	if !errors.Is(err, errResourceLifecycleTombstoned) {
		t.Fatalf("late publisher lock err = %v, want deletion fence", err)
	}
}

func TestUncatalogedDeletionFailureRemainsAuthorizedAndRetryable(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	const resourceID = "file_retry_delete"
	principal := requestPrincipal{UserID: "owner", OrgID: "org"}
	source := filepath.Join(root, resourceID+"__cells.tif")
	if err := os.WriteFile(source, []byte("source"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := writeUploadMetadata(root, resourceID, principal); err != nil {
		t.Fatal(err)
	}
	blockedDerivative := filepath.Join(root, resourceDerivedDir, derivedPyramidName(resourceID))
	if err := os.MkdirAll(filepath.Join(blockedDerivative, "unexpected-child"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := writeUploadDeletionIntent(root, resourceID, source, principal.record()); err != nil {
		t.Fatal(err)
	}

	if err := removeUploadedFile(context.Background(), root, resourceID, source); err == nil {
		t.Fatal("cleanup unexpectedly accepted a directory in the file-only derivative namespace")
	}
	if _, err := os.Stat(filepath.Join(root, resourceTombstoneDir, resourceFilesystemTombstoneName(resourceID))); err != nil {
		t.Fatalf("retryable cleanup did not preserve its permanent publication fence: %v", err)
	}
	for _, path := range []string{
		source,
		uploadMetadataPath(root, resourceID),
		filepath.Join(root, resourceMetaDir, resourceDeletionIntentName(resourceID)),
	} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("retry authority %q did not survive partial cleanup: %v", path, err)
		}
	}
	if _, err := pendingUploadDeletionSourceForRequest(root, resourceID, requestPrincipal{UserID: "other", OrgID: "org"}); !errors.Is(err, store.ErrNotFound) {
		t.Fatalf("foreign retry err = %v, want ErrNotFound", err)
	}
	retrySource, err := pendingUploadDeletionSourceForRequest(root, resourceID, principal)
	if err != nil || retrySource != source {
		t.Fatalf("owner retry source = %q err=%v, want %q", retrySource, err, source)
	}

	if err := os.RemoveAll(blockedDerivative); err != nil {
		t.Fatal(err)
	}
	if err := removeUploadedFile(context.Background(), root, resourceID, retrySource); err != nil {
		t.Fatalf("retry cleanup: %v", err)
	}
	for _, path := range []string{
		source,
		uploadMetadataPath(root, resourceID),
		filepath.Join(root, resourceMetaDir, resourceDeletionIntentName(resourceID)),
	} {
		if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("owned path survived successful retry %q: %v", path, err)
		}
	}
}

func TestRemoveUploadedFileRejectsSymlinkedSource(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	outside := filepath.Join(t.TempDir(), "outside-source.tif")
	if err := os.WriteFile(outside, []byte("authoritative"), 0o644); err != nil {
		t.Fatal(err)
	}
	const resourceID = "file_symlink_rollback"
	source := filepath.Join(root, resourceID+"__source.tif")
	if err := os.Symlink(outside, source); err != nil {
		t.Fatal(err)
	}
	if err := writeUploadMetadata(root, resourceID, requestPrincipal{UserID: "owner"}); err != nil {
		t.Fatal(err)
	}
	if err := removeUploadedFile(context.Background(), root, resourceID, source); err == nil {
		t.Fatal("symlinked authoritative source was accepted for uncataloged deletion")
	}
	if payload, err := os.ReadFile(outside); err != nil || string(payload) != "authoritative" {
		t.Fatalf("outside source changed: %q err=%v", payload, err)
	}
	for _, path := range []string{source, uploadMetadataPath(root, resourceID)} {
		if _, err := os.Lstat(path); err != nil {
			t.Fatalf("retry authority %q changed: %v", path, err)
		}
	}
}

func TestUploadDeletionIntentRejectsSymlinkedAuthority(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	const resourceID = "file_symlink_intent"
	source := filepath.Join(root, resourceID+"__source.tif")
	if err := os.WriteFile(source, []byte("source"), 0o600); err != nil {
		t.Fatal(err)
	}
	meta := filepath.Join(root, resourceMetaDir)
	if err := os.MkdirAll(meta, 0o700); err != nil {
		t.Fatal(err)
	}
	authority := filepath.Join(meta, "authority.json")
	if err := os.WriteFile(authority, []byte("do not trust through a link"), 0o600); err != nil {
		t.Fatal(err)
	}
	intentPath := filepath.Join(meta, resourceDeletionIntentName(resourceID))
	if err := os.Symlink(filepath.Base(authority), intentPath); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}

	if err := writeUploadDeletionIntent(root, resourceID, source, principalRecord{UserID: "owner"}); err == nil {
		t.Fatal("symlinked deletion authority was accepted")
	}
	if payload, err := os.ReadFile(authority); err != nil || string(payload) != "do not trust through a link" {
		t.Fatalf("symlink target changed: %q err=%v", payload, err)
	}
}

func TestGoLifecycleLockContendsWithPythonPublisher(t *testing.T) {
	t.Parallel()
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 is unavailable")
	}
	root := t.TempDir()
	const resourceID = "file_cross_language"
	source := filepath.Join(root, resourceID+"__source.tif")
	if err := os.WriteFile(source, []byte("source"), 0o644); err != nil {
		t.Fatal(err)
	}
	uploadRoot, err := os.OpenRoot(root)
	if err != nil {
		t.Fatal(err)
	}
	defer uploadRoot.Close()
	lock, err := acquireResourceLifecycleLock(context.Background(), uploadRoot, resourceID, "")
	if err != nil {
		t.Fatal(err)
	}
	lockReleased := false
	defer func() {
		if !lockReleased {
			_ = lock.release()
		}
	}()

	_, testFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve test source path")
	}
	pythonSource := filepath.Clean(filepath.Join(filepath.Dir(testFile), "..", "..", "..", "deepagents_runtime", "src"))
	destination := filepath.Join(root, resourceDerivedDir, resourceID+"__pyramid.tif")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	command := exec.CommandContext(ctx, python, "-c", `
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from ultra_deepagents.imaging import derivative_manifest
print("ready", flush=True)
with derivative_manifest._publication_lock(Path(sys.argv[2]), Path(sys.argv[3])):
    print("acquired", flush=True)
`, pythonSource, destination, source)
	stdout, err := command.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	var stderr strings.Builder
	command.Stderr = &stderr
	if err := command.Start(); err != nil {
		t.Fatal(err)
	}
	waited := make(chan error, 1)
	go func() { waited <- command.Wait() }()
	scanner := bufio.NewScanner(stdout)
	if !scanner.Scan() || scanner.Text() != "ready" {
		t.Fatalf("python publisher did not become ready: line=%q err=%v stderr=%s", scanner.Text(), scanner.Err(), stderr.String())
	}
	select {
	case err := <-waited:
		t.Fatalf("python publisher bypassed Go lifecycle lock: %v stderr=%s", err, stderr.String())
	case <-time.After(150 * time.Millisecond):
	}
	if err := lock.release(); err != nil {
		t.Fatal(err)
	}
	lockReleased = true
	if !scanner.Scan() || scanner.Text() != "acquired" {
		t.Fatalf("python publisher did not acquire released Go lock: line=%q err=%v stderr=%s", scanner.Text(), scanner.Err(), stderr.String())
	}
	if err := <-waited; err != nil {
		t.Fatalf("python publisher exited with error: %v stderr=%s", err, stderr.String())
	}
}

func TestOwnedDerivativeHardLinksCountReclaimedBytesOnce(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	const resourceID = "file_hardlink"
	derived := filepath.Join(root, resourceDerivedDir)
	if err := os.MkdirAll(derived, 0o755); err != nil {
		t.Fatal(err)
	}
	digest := strings.Repeat("d", 64)
	artifact := filepath.Join(derived, resourceID+"__pyramid.sha256-"+digest+".tif")
	payload := []byte("one physical artifact")
	if err := os.WriteFile(artifact, payload, 0o644); err != nil {
		t.Fatal(err)
	}
	for _, suffix := range []string{"publish", "recovery"} {
		link := filepath.Join(derived, "."+resourceID+"__pyramid.sha256-"+digest+".tif."+suffix+"-0123456789abcdef01234567")
		if err := os.Link(artifact, link); err != nil {
			t.Fatal(err)
		}
	}
	uploadRoot, err := os.OpenRoot(root)
	if err != nil {
		t.Fatal(err)
	}
	defer uploadRoot.Close()
	lock, err := acquireResourceLifecycleLock(context.Background(), uploadRoot, resourceID, "")
	if err != nil {
		t.Fatal(err)
	}
	defer lock.release()
	freed, err := removeOwnedResourceNamespace(uploadRoot, resourceID, "")
	if err != nil {
		t.Fatal(err)
	}
	if freed != int64(len(payload)) {
		t.Fatalf("hard-linked reclaimed bytes = %d, want %d", freed, len(payload))
	}
}

func TestOwnedScene3dDerivativeDirectoriesAreReclaimedExactly(t *testing.T) {
	t.Parallel()
	rootPath := t.TempDir()
	const resourceID = "file_scene"
	digest := strings.Repeat("c", 64)
	derivedPath := filepath.Join(rootPath, resourceDerivedDir)
	if err := os.MkdirAll(derivedPath, 0o755); err != nil {
		t.Fatal(err)
	}
	ownedNames := make([]string, 0, 8)
	for _, version := range []string{"", ".v2", ".v3"} {
		ownedDirectory := resourceID + "__scene3d" + version + ".sha256-" + digest
		ownedTemporary := "." + ownedDirectory + ".tmp-abcdefgh"
		for _, directory := range []string{ownedDirectory, ownedTemporary} {
			path := filepath.Join(derivedPath, directory)
			if err := os.Mkdir(path, 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(path, "chunk_00000.bin"), []byte("scene bytes"), 0o600); err != nil {
				t.Fatal(err)
			}
			ownedNames = append(ownedNames, directory)
		}
		ownedMarker := ownedDirectory + ".failed"
		ownedMarkerTemporary := "." + ownedMarker + ".abcdefgh"
		for _, name := range []string{ownedMarker, ownedMarkerTemporary} {
			if err := os.WriteFile(filepath.Join(derivedPath, name), []byte("failure"), 0o600); err != nil {
				t.Fatal(err)
			}
			ownedNames = append(ownedNames, name)
		}
	}
	ownedDirectory := resourceID + "__scene3d.sha256-" + digest
	neighborDirectory := "file_scene_neighbor__scene3d.sha256-" + digest
	if err := os.Mkdir(filepath.Join(derivedPath, neighborDirectory), 0o755); err != nil {
		t.Fatal(err)
	}
	nearMiss := ownedDirectory + ".manifest.json"
	if err := os.WriteFile(filepath.Join(derivedPath, nearMiss), []byte("not owned"), 0o600); err != nil {
		t.Fatal(err)
	}

	root, err := os.OpenRoot(rootPath)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()
	lock, err := acquireResourceLifecycleLock(context.Background(), root, resourceID, "")
	if err != nil {
		t.Fatal(err)
	}
	defer lock.release()
	if _, err := removeOwnedResourceNamespace(root, resourceID, ""); err != nil {
		t.Fatal(err)
	}
	for _, name := range ownedNames {
		if _, err := os.Lstat(filepath.Join(derivedPath, name)); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("owned scene derivative %q remains: %v", name, err)
		}
	}
	for _, name := range []string{neighborDirectory, nearMiss} {
		if _, err := os.Lstat(filepath.Join(derivedPath, name)); err != nil {
			t.Fatalf("neighbor/near-miss %q was changed: %v", name, err)
		}
	}
}

func TestOwnedAnalysisPublicationStagingIsReclaimedExactly(t *testing.T) {
	t.Parallel()
	rootPath := t.TempDir()
	const resourceID = "file_analysis_output"
	sourceRelative := filepath.Join("analysis", "job-a", "mask.tif")
	if err := os.MkdirAll(filepath.Join(rootPath, filepath.Dir(sourceRelative)), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(rootPath, sourceRelative), []byte("mask"), 0o644); err != nil {
		t.Fatal(err)
	}
	ownedTemp := filepath.Join(
		rootPath,
		filepath.Dir(sourceRelative),
		"."+resourceID+"__analysis.tmp-abcdefgh",
	)
	neighborTemp := filepath.Join(
		rootPath,
		filepath.Dir(sourceRelative),
		".file_analysis_output_neighbor__analysis.tmp-abcdefgh",
	)
	for path, payload := range map[string]string{ownedTemp: "owned", neighborTemp: "neighbor"} {
		if err := os.WriteFile(path, []byte(payload), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	root, err := os.OpenRoot(rootPath)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()
	freed, err := removeOwnedResourceNamespace(root, resourceID, sourceRelative)
	if err != nil {
		t.Fatal(err)
	}
	if freed != int64(len("mask")+len("owned")) {
		t.Fatalf("reclaimed bytes = %d, want source plus owned staging", freed)
	}
	if _, err := os.Stat(ownedTemp); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("owned analysis staging remains: %v", err)
	}
	if payload, err := os.ReadFile(neighborTemp); err != nil || string(payload) != "neighbor" {
		t.Fatalf("neighbor analysis staging changed: %q err=%v", payload, err)
	}
}

func TestReclaimExpiredResourcesFailsClosedOnEscapingDerivedSymlink(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	outside := t.TempDir()
	sentinel := filepath.Join(outside, "sentinel")
	if err := os.WriteFile(sentinel, []byte("survive"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(root, resourceDerivedDir)); err != nil {
		t.Fatal(err)
	}
	const resourceID = "file_escape"
	source := filepath.Join(root, resourceID+"__cells.tif")
	if err := os.WriteFile(source, []byte("source"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := writeUploadMetadata(root, resourceID, requestPrincipal{UserID: "u", OrgID: "o"}); err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	mem := store.NewMemoryStore()
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   resourceID,
		OriginalName: "cells.tif",
		StorageURI:   fileStorageURI(source),
		StoragePath:  filepath.Base(source),
		OwnerUserID:  "u",
		OwnerOrgID:   "o",
		Status:       "active",
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := mem.SoftDeleteResourceForUser(ctx, resourceID, "u", "o", time.Now().Add(-31*24*time.Hour)); err != nil {
		t.Fatal(err)
	}
	reclaimed, bytes, err := ReclaimExpiredResources(ctx, mem, root, 1)
	if err == nil || !strings.Contains(err.Error(), "inventory claimed derivative namespaces") {
		t.Fatalf("reclaim error = %v, want derived-root preflight failure", err)
	}
	if reclaimed != 0 || bytes != 0 {
		t.Fatalf("escaping derived root reclaimed = %d/%d, want 0/0", reclaimed, bytes)
	}
	if _, err := os.Stat(sentinel); err != nil {
		t.Fatalf("outside sentinel was changed: %v", err)
	}
	if _, err := os.Stat(source); err != nil {
		t.Fatalf("source was removed before namespace preflight completed: %v", err)
	}
	if _, err := os.Stat(uploadMetadataPath(root, resourceID)); err != nil {
		t.Fatalf("ownership metadata was removed after failed cleanup: %v", err)
	}
	if status, found, err := mem.GetResourceLifecycleStatus(ctx, resourceID); err != nil || !found || status != domain.ResourceStatusPurging {
		t.Fatalf("failed resource lifecycle = %q found=%v err=%v, want reclaimable purging", status, found, err)
	}
	reclaimedClaim, err := mem.ClaimResourcesPastRetention(ctx, time.Minute, 10)
	if err != nil || len(reclaimedClaim) != 1 || reclaimedClaim[0].ResourceID != resourceID {
		t.Fatalf("failed resource was not reclaimable by a later sweep: %+v err=%v", reclaimedClaim, err)
	}
	_, _ = mem.ReleaseResourceRetentionClaim(ctx, resourceID, reclaimedClaim[0].UpdatedAt)
}

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
	// A past-window resource whose authoritative source lives OUTSIDE the upload
	// root stays tombstoned until an external-storage deletion policy handles it.
	outsideSrc := mkResource("file_outside", "ext.tif", outsideRoot, 31*24*time.Hour, false)
	uploadRoot, err := os.OpenRoot(root)
	if err != nil {
		t.Fatal(err)
	}
	if err := ensureResourceFilesystemDeleteFence(uploadRoot, "file_expired"); err != nil {
		_ = uploadRoot.Close()
		t.Fatal(err)
	}
	if err := uploadRoot.Close(); err != nil {
		t.Fatal(err)
	}

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
	if reclaimed != 1 {
		t.Fatalf("reclaimed %d resources, want only the locally managed expired resource", reclaimed)
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
	if _, statErr := os.Stat(filepath.Join(root, resourceTombstoneDir, resourceFilesystemDeleteFenceName("file_expired"))); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("expired resource retained reversible delete fence: %v", statErr)
	}
	if _, statErr := os.Stat(filepath.Join(root, resourceTombstoneDir, resourceFilesystemTombstoneName("file_expired"))); statErr != nil {
		t.Fatalf("expired resource permanent tombstone: %v", statErr)
	}

	// SAFETY: the outside-root source and its local derivative survive together
	// with the retryable tombstone; successful purge would permit resurrection.
	if _, statErr := os.Stat(outsideSrc); statErr != nil {
		t.Fatalf("GC deleted a file OUTSIDE the upload root: %v", statErr)
	}
	if _, statErr := os.Stat(filepath.Join(root, "derived", derivedPyramidName("file_outside"))); statErr != nil {
		t.Fatalf("GC changed derivatives for externally managed storage: %v", statErr)
	}

	// Within-window + active resources are untouched.
	if _, statErr := os.Stat(freshSrc); statErr != nil {
		t.Fatal("within-window (still-restorable) source was wrongly deleted")
	}
	if _, statErr := os.Stat(activeSrc); statErr != nil {
		t.Fatal("active source was wrongly deleted")
	}

	// The external resource remains visible to retention operators as unresolved.
	if backlog, err = mem.RetentionBacklog(ctx, time.Now()); err != nil ||
		backlog.Count != 0 || backlog.BlockedCount != 1 || backlog.BlockedBytes != 100 {
		t.Fatalf("retention backlog after reclaim = %+v err=%v, want one blocked external resource", backlog, err)
	}
}

func TestReclaimExpiredResourcesDeletesCatalogFileStorageURI(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	outsideRoot := t.TempDir()
	mem := store.NewMemoryStore()
	ctx := context.Background()
	const user, org = "u", "o"

	source := filepath.Join(root, "file_catalog__cells.ome.tiff")
	sourceBytes := []byte("catalog source bytes")
	if err := os.WriteFile(source, sourceBytes, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   "file_catalog",
		OriginalName: "cells.ome.tiff",
		SizeBytes:    int64(len(sourceBytes)),
		StorageURI:   fileStorageURI(source),
		StoragePath:  filepath.Base(source),
		OwnerUserID:  user,
		OwnerOrgID:   org,
		Status:       "active",
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := mem.SoftDeleteResourceForUser(ctx, "file_catalog", user, org, time.Now().Add(-31*24*time.Hour)); err != nil {
		t.Fatal(err)
	}

	outsideSource := filepath.Join(outsideRoot, "file_outside__secret.tif")
	if err := os.WriteFile(outsideSource, []byte("must survive"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   "file_outside",
		OriginalName: "secret.tif",
		StorageURI:   fileStorageURI(outsideSource),
		StoragePath:  filepath.Base(outsideSource),
		OwnerUserID:  user,
		OwnerOrgID:   org,
		Status:       "active",
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := mem.SoftDeleteResourceForUser(ctx, "file_outside", user, org, time.Now().Add(-31*24*time.Hour)); err != nil {
		t.Fatal(err)
	}

	reclaimed, bytes, err := ReclaimExpiredResources(ctx, mem, root, 100)
	if err != nil {
		t.Fatalf("reclaim: %v", err)
	}
	if reclaimed != 1 {
		t.Fatalf("reclaimed %d resources, want only the upload-root file purged", reclaimed)
	}
	if bytes != int64(len(sourceBytes)) {
		t.Fatalf("reclaimed bytes = %d, want catalog source size %d", bytes, len(sourceBytes))
	}
	if _, statErr := os.Stat(source); !os.IsNotExist(statErr) {
		t.Fatalf("catalog source was not deleted, stat err = %v", statErr)
	}
	if _, statErr := os.Stat(outsideSource); statErr != nil {
		t.Fatalf("GC deleted file:// storage outside upload root: %v", statErr)
	}
	if backlog, err := mem.RetentionBacklog(ctx, time.Now()); err != nil ||
		backlog.Count != 0 || backlog.BlockedCount != 1 {
		t.Fatalf("external resource backlog = %+v err=%v, want one durably blocked row", backlog, err)
	}
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID: "file_outside", OwnerUserID: user, OwnerOrgID: org, Status: domain.ResourceStatusActive,
	}); !errors.Is(err, store.ErrConflict) {
		t.Fatalf("blocked external resource upsert error = %v, want conflict", err)
	}
}

func TestReclaimExpiredResourcesDeletesRelativeStoragePathUnderRoot(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	mem := store.NewMemoryStore()
	ctx := context.Background()
	const user, org = "u", "o"

	source := filepath.Join(root, "file_relative__legacy.tif")
	sourceBytes := []byte("legacy relative source")
	if err := os.WriteFile(source, sourceBytes, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   "file_relative",
		OriginalName: "legacy.tif",
		SizeBytes:    int64(len(sourceBytes)),
		StoragePath:  filepath.Base(source),
		OwnerUserID:  user,
		OwnerOrgID:   org,
		Status:       "active",
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := mem.SoftDeleteResourceForUser(ctx, "file_relative", user, org, time.Now().Add(-31*24*time.Hour)); err != nil {
		t.Fatal(err)
	}

	reclaimed, bytes, err := ReclaimExpiredResources(ctx, mem, root, 100)
	if err != nil {
		t.Fatalf("reclaim: %v", err)
	}
	if reclaimed != 1 || bytes != int64(len(sourceBytes)) {
		t.Fatalf("reclaimed = %d/%d, want 1/%d", reclaimed, bytes, len(sourceBytes))
	}
	if _, statErr := os.Stat(source); !os.IsNotExist(statErr) {
		t.Fatalf("relative storage source was not deleted, stat err = %v", statErr)
	}
}

func TestReclaimExpiredResourcesDoesNotPurgeWhenArtifactRemovalFails(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	mem := store.NewMemoryStore()
	ctx := context.Background()
	const user, org = "u", "o"

	source := filepath.Join(root, "file_blocked__directory")
	if err := os.MkdirAll(source, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(source, "child.txt"), []byte("not removable by os.Remove"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   "file_blocked",
		OriginalName: "directory",
		StorageURI:   fileStorageURI(source),
		StoragePath:  filepath.Base(source),
		OwnerUserID:  user,
		OwnerOrgID:   org,
		Status:       "active",
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := mem.SoftDeleteResourceForUser(ctx, "file_blocked", user, org, time.Now().Add(-31*24*time.Hour)); err != nil {
		t.Fatal(err)
	}

	reclaimed, bytes, err := ReclaimExpiredResources(ctx, mem, root, 100)
	if err != nil {
		t.Fatalf("reclaim: %v", err)
	}
	if reclaimed != 0 || bytes != 0 {
		t.Fatalf("reclaimed = %d/%d, want 0/0 when artifact deletion fails", reclaimed, bytes)
	}
	if _, statErr := os.Stat(filepath.Join(source, "child.txt")); statErr != nil {
		t.Fatalf("source child changed after failed delete: %v", statErr)
	}
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID: "file_blocked", OwnerUserID: user, OwnerOrgID: org, Status: domain.ResourceStatusActive,
	}); !errors.Is(err, store.ErrConflict) {
		t.Fatalf("failed cleanup upsert error = %v, want purging fence conflict", err)
	}
	if _, err := os.Stat(filepath.Join(root, resourceTombstoneDir, resourceFilesystemTombstoneName("file_blocked"))); err != nil {
		t.Fatalf("failed cleanup must retain permanent lifecycle fence: %v", err)
	}
	if _, err := os.Stat(filepath.Join(root, resourceLockDir, resourceLifecycleLockName("file_blocked"))); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("failed cleanup retained an abandoned lock path: %v", err)
	}
}

func TestReclaimPartialCleanupKeepsPurgingFence(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	mem := store.NewMemoryStore()
	ctx := context.Background()
	const resourceID = "file_partial_cleanup"
	source := filepath.Join(root, resourceID+"__source.tif")
	if err := os.WriteFile(source, []byte("source"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, resourceMetaDir, resourceID+".json"), 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID: resourceID, StorageURI: fileStorageURI(source), SizeBytes: 6,
		OwnerUserID: "u", OwnerOrgID: "o", Status: domain.ResourceStatusActive,
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := mem.SoftDeleteResourceForUser(ctx, resourceID, "u", "o", time.Now().Add(-31*24*time.Hour)); err != nil {
		t.Fatal(err)
	}
	reclaimed, bytes, err := ReclaimExpiredResources(ctx, mem, root, 1)
	if err != nil {
		t.Fatal(err)
	}
	if reclaimed != 0 || bytes != 0 {
		t.Fatalf("partial cleanup = %d/%d, want 0/0", reclaimed, bytes)
	}
	if _, err := os.Stat(source); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("fixture did not fail after source removal: %v", err)
	}
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID: resourceID, OwnerUserID: "u", OwnerOrgID: "o", Status: domain.ResourceStatusActive,
	}); !errors.Is(err, store.ErrConflict) {
		t.Fatalf("partial cleanup upsert error = %v, want purging fence conflict", err)
	}
	if _, err := os.Stat(filepath.Join(root, resourceTombstoneDir, resourceFilesystemTombstoneName(resourceID))); err != nil {
		t.Fatalf("partial cleanup lost permanent lifecycle fence: %v", err)
	}
	if _, err := os.Stat(filepath.Join(root, resourceLockDir, resourceLifecycleLockName(resourceID))); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("partial cleanup retained an abandoned lock path: %v", err)
	}
}

func TestReclaimPurgeFailureKeepsPurgingFence(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	mem := store.NewMemoryStore()
	ctx := context.Background()
	const resourceID = "file_purge_failure"
	source := filepath.Join(root, resourceID+"__source.tif")
	if err := os.WriteFile(source, []byte("source"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID: resourceID, StorageURI: fileStorageURI(source), SizeBytes: 6,
		OwnerUserID: "u", OwnerOrgID: "o", Status: domain.ResourceStatusActive,
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := mem.SoftDeleteResourceForUser(ctx, resourceID, "u", "o", time.Now().Add(-31*24*time.Hour)); err != nil {
		t.Fatal(err)
	}
	gcStore := &failingPurgeRetentionStore{MemoryStore: mem, err: errors.New("catalog unavailable")}
	reclaimed, bytes, err := ReclaimExpiredResources(ctx, gcStore, root, 1)
	if err != nil {
		t.Fatal(err)
	}
	if reclaimed != 0 || bytes != 0 {
		t.Fatalf("purge failure = %d/%d, want 0/0", reclaimed, bytes)
	}
	if _, err := os.Stat(source); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("fixture did not reach catalog purge after source removal: %v", err)
	}
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID: resourceID, OwnerUserID: "u", OwnerOrgID: "o", Status: domain.ResourceStatusActive,
	}); !errors.Is(err, store.ErrConflict) {
		t.Fatalf("purge failure upsert error = %v, want purging fence conflict", err)
	}
	if _, err := os.Stat(filepath.Join(root, resourceTombstoneDir, resourceFilesystemTombstoneName(resourceID))); err != nil {
		t.Fatalf("purge failure lost permanent lifecycle fence: %v", err)
	}
	if _, err := os.Stat(filepath.Join(root, resourceLockDir, resourceLifecycleLockName(resourceID))); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("purge failure retained an abandoned lock path: %v", err)
	}
	if backlog, err := mem.RetentionBacklog(ctx, time.Now()); err != nil ||
		backlog.PurgingCount != 1 || backlog.PurgingBytes != 6 {
		t.Fatalf("purge failure backlog = %+v err=%v, want one operator-visible purging resource", backlog, err)
	}
}

func TestReclaimRejectsSymlinkedManagedSource(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	outside := filepath.Join(t.TempDir(), "outside-source.tif")
	if err := os.WriteFile(outside, []byte("outside"), 0o644); err != nil {
		t.Fatal(err)
	}
	const resourceID = "file_symlink_gc"
	source := filepath.Join(root, resourceID+"__source.tif")
	if err := os.Symlink(outside, source); err != nil {
		t.Fatal(err)
	}
	mem := store.NewMemoryStore()
	ctx := context.Background()
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID: resourceID, StorageURI: fileStorageURI(source), SizeBytes: 7,
		OwnerUserID: "u", OwnerOrgID: "o", Status: domain.ResourceStatusActive,
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := mem.SoftDeleteResourceForUser(ctx, resourceID, "u", "o", time.Now().Add(-31*24*time.Hour)); err != nil {
		t.Fatal(err)
	}
	reclaimed, bytes, err := ReclaimExpiredResources(ctx, mem, root, 1)
	if err != nil {
		t.Fatal(err)
	}
	if reclaimed != 0 || bytes != 0 {
		t.Fatalf("symlinked source = %d/%d, want 0/0", reclaimed, bytes)
	}
	if payload, err := os.ReadFile(outside); err != nil || string(payload) != "outside" {
		t.Fatalf("outside source changed: %q err=%v", payload, err)
	}
	if _, err := os.Lstat(source); err != nil {
		t.Fatalf("source symlink changed: %v", err)
	}
	if backlog, err := mem.RetentionBacklog(ctx, time.Now()); err != nil || backlog.BlockedCount != 1 {
		t.Fatalf("symlinked source backlog = %+v err=%v, want blocked", backlog, err)
	}
}

func TestReclaimExpiredResourcesDeletesOwnedBundleDirectory(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	mem := store.NewMemoryStore()
	ctx := context.Background()
	const user, org = "u", "o"

	source := filepath.Join(root, bundlesDirName, "file_bundle", "scan.ome.zarr")
	if err := os.MkdirAll(filepath.Join(source, "0", "0"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(source, ".zgroup"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(source, "0", "0", "0"), []byte("chunk"), 0o644); err != nil {
		t.Fatal(err)
	}
	bundleBytes := int64(len("{}") + len("chunk"))
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   "file_bundle",
		OriginalName: "scan.ome.zarr",
		SizeBytes:    bundleBytes,
		StorageURI:   fileStorageURI(source),
		StoragePath:  filepath.Base(source),
		OwnerUserID:  user,
		OwnerOrgID:   org,
		Status:       "active",
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := mem.SoftDeleteResourceForUser(ctx, "file_bundle", user, org, time.Now().Add(-31*24*time.Hour)); err != nil {
		t.Fatal(err)
	}

	reclaimed, bytes, err := ReclaimExpiredResources(ctx, mem, root, 100)
	if err != nil {
		t.Fatalf("reclaim: %v", err)
	}
	if reclaimed != 1 {
		t.Fatalf("reclaimed %d resources, want bundle row purged", reclaimed)
	}
	if bytes != bundleBytes {
		t.Fatalf("reclaimed bytes = %d, want bundle file bytes", bytes)
	}
	if _, statErr := os.Stat(source); !os.IsNotExist(statErr) {
		t.Fatalf("bundle directory was not deleted, stat err = %v", statErr)
	}
	if expired, err := mem.ListResourcesPastRetention(ctx, time.Now(), 100); err != nil || len(expired) != 0 {
		t.Fatalf("expired resources after bundle reclaim = %+v err=%v, want none", expired, err)
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
		source := filepath.Join(root, id+"__f.tif")
		if err := os.WriteFile(source, []byte(id), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
			ResourceID: id, OriginalName: "f.tif", SizeBytes: 10, StorageURI: fileStorageURI(source), OwnerUserID: "u", OwnerOrgID: "o", Status: "active",
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

func TestBlockedExternalResourcesDoNotStarveLocalRetention(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	outsideRoot := t.TempDir()
	mem := store.NewMemoryStore()
	ctx := context.Background()
	for i := 0; i < 4; i++ {
		id := fmt.Sprintf("file_external_%d", i)
		source := filepath.Join(outsideRoot, id+".tif")
		if err := os.WriteFile(source, []byte(id), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
			ResourceID: id, StorageURI: fileStorageURI(source), StoragePath: filepath.Base(source), SizeBytes: int64(len(id)),
			OwnerUserID: "u", OwnerOrgID: "o", Status: domain.ResourceStatusActive,
		}); err != nil {
			t.Fatal(err)
		}
		if _, err := mem.SoftDeleteResourceForUser(ctx, id, "u", "o", time.Now().Add(-time.Duration(40-i)*24*time.Hour)); err != nil {
			t.Fatal(err)
		}
	}
	const localID = "file_local_after_poison"
	localSource := filepath.Join(root, localID+".tif")
	if err := os.WriteFile(localSource, []byte("local"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID: localID, StorageURI: fileStorageURI(localSource), StoragePath: filepath.Base(localSource), SizeBytes: 5,
		OwnerUserID: "u", OwnerOrgID: "o", Status: domain.ResourceStatusActive,
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := mem.SoftDeleteResourceForUser(ctx, localID, "u", "o", time.Now().Add(-31*24*time.Hour)); err != nil {
		t.Fatal(err)
	}
	reclaimed, _, err := ReclaimExpiredResources(ctx, mem, root, 2)
	if err != nil {
		t.Fatal(err)
	}
	if reclaimed != 1 {
		t.Fatalf("reclaimed local resources = %d, want 1 in the first bounded sweep", reclaimed)
	}
	if _, err := os.Stat(localSource); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("local resource was starved: %v", err)
	}
	if backlog, err := mem.RetentionBacklog(ctx, time.Now()); err != nil || backlog.Count != 0 || backlog.BlockedCount != 4 {
		t.Fatalf("post-sweep backlog = %+v err=%v, want all four external rows classified without starving local cleanup", backlog, err)
	}
}

func TestConcurrentRetentionGCClaimsEachResourceOnce(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	mem := store.NewMemoryStore()
	ctx := context.Background()
	const resourceCount = 12
	for i := 0; i < resourceCount; i++ {
		id := fmt.Sprintf("file_gc_%02d", i)
		source := filepath.Join(root, id+"__source.tif")
		if err := os.WriteFile(source, []byte(id), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
			ResourceID: id, StorageURI: fileStorageURI(source), OwnerUserID: "u", OwnerOrgID: "o", Status: "active",
		}); err != nil {
			t.Fatal(err)
		}
		if _, err := mem.SoftDeleteResourceForUser(ctx, id, "u", "o", time.Now().Add(-31*24*time.Hour)); err != nil {
			t.Fatal(err)
		}
	}

	start := make(chan struct{})
	results := make(chan int, 2)
	errs := make(chan error, 2)
	for range 2 {
		go func() {
			<-start
			reclaimed, _, err := ReclaimExpiredResources(ctx, mem, root, resourceCount)
			results <- reclaimed
			errs <- err
		}()
	}
	close(start)
	total := <-results + <-results
	for range 2 {
		if err := <-errs; err != nil {
			t.Fatal(err)
		}
	}
	if total != resourceCount {
		t.Fatalf("concurrent GC reclaimed %d resources, want %d exactly once", total, resourceCount)
	}
	if backlog, err := mem.RetentionBacklog(ctx, time.Now()); err != nil || backlog.Count != 0 {
		t.Fatalf("retention backlog = %+v err=%v, want empty", backlog, err)
	}
}

type retentionGCStub struct {
	claims   []domain.ResourceRecord
	released []string
	blocked  []string
	purged   []string
}

type renewingRetentionGCStub struct {
	retentionGCStub
	mu         sync.Mutex
	renewed    chan struct{}
	renewCalls int
	lose       bool
}

func (s *renewingRetentionGCStub) RenewResourceRetentionClaim(
	_ context.Context,
	_ string,
	claimedAt time.Time,
) (time.Time, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.renewCalls++
	select {
	case s.renewed <- struct{}{}:
	default:
	}
	if s.lose {
		return time.Time{}, false, nil
	}
	return claimedAt.Add(time.Nanosecond), true, nil
}

func TestResourceRetentionClaimGuardRenewsAndFreezesExactToken(t *testing.T) {
	claimedAt := time.Now().UTC()
	store := &renewingRetentionGCStub{renewed: make(chan struct{}, 1)}
	guard := startResourceRetentionClaimGuardWithInterval(
		context.Background(),
		store,
		"file_claim_heartbeat",
		claimedAt,
		time.Millisecond,
	)
	select {
	case <-store.renewed:
	case <-time.After(time.Second):
		t.Fatal("retention claim heartbeat did not renew")
	}
	token, lost, err := guard.stop()
	if err != nil || lost || !token.After(claimedAt) {
		t.Fatalf("claim guard stop = token %v lost=%v err=%v, want renewed exact token", token, lost, err)
	}
}

func TestResourceRetentionClaimGuardReportsLostTakeover(t *testing.T) {
	store := &renewingRetentionGCStub{renewed: make(chan struct{}, 1), lose: true}
	guard := startResourceRetentionClaimGuardWithInterval(
		context.Background(),
		store,
		"file_claim_lost",
		time.Now().UTC(),
		time.Millisecond,
	)
	select {
	case <-store.renewed:
	case <-time.After(time.Second):
		t.Fatal("retention claim heartbeat did not attempt renewal")
	}
	_, lost, err := guard.stop()
	if err != nil || !lost {
		t.Fatalf("claim guard stop lost=%v err=%v, want lost without store error", lost, err)
	}
}

func (s *retentionGCStub) ClaimResourcesPastRetention(_ context.Context, _ time.Duration, limit int) ([]domain.ResourceRecord, error) {
	if limit <= 0 || limit > len(s.claims) {
		limit = len(s.claims)
	}
	claims := append([]domain.ResourceRecord(nil), s.claims[:limit]...)
	s.claims = append([]domain.ResourceRecord(nil), s.claims[limit:]...)
	return claims, nil
}

func (s *retentionGCStub) ReleaseResourceRetentionClaim(_ context.Context, resourceID string, _ time.Time) (bool, error) {
	s.released = append(s.released, resourceID)
	return true, nil
}

func (s *retentionGCStub) BlockResourceRetentionClaim(_ context.Context, resourceID string, _ time.Time) (bool, error) {
	s.blocked = append(s.blocked, resourceID)
	return true, nil
}

func (s *retentionGCStub) PurgeClaimedResource(_ context.Context, resourceID string, _ time.Time) (bool, error) {
	s.purged = append(s.purged, resourceID)
	return true, nil
}

type failingPurgeRetentionStore struct {
	*store.MemoryStore
	err error
}

type fenceObservingRetentionStore struct {
	*store.MemoryStore
	root                   string
	permanentFenceObserved bool
}

type restoreDuringLifecycleListStore struct {
	*store.MemoryStore
	resourceID string
	restored   bool
}

func (s *restoreDuringLifecycleListStore) ListResourceLifecycleFenceCandidates(ctx context.Context, afterResourceID string, limit int) ([]domain.ResourceRecord, error) {
	resources, err := s.MemoryStore.ListResourceLifecycleFenceCandidates(ctx, afterResourceID, limit)
	if err != nil || s.restored || len(resources) == 0 {
		return resources, err
	}
	if _, err := s.MemoryStore.RestoreResourceForUser(ctx, s.resourceID, "u", "o", time.Now()); err != nil {
		return nil, err
	}
	s.restored = true
	return resources, nil
}

func (s *fenceObservingRetentionStore) BlockResourceRetentionClaim(ctx context.Context, resourceID string, claimedAt time.Time) (bool, error) {
	root, err := os.OpenRoot(s.root)
	if err != nil {
		return false, err
	}
	defer root.Close()
	fenced, err := resourceFilesystemPermanentlyTombstoned(root, resourceID)
	if err != nil {
		return false, err
	}
	if !fenced {
		return false, errors.New("catalog block attempted before permanent filesystem fence")
	}
	s.permanentFenceObserved = true
	return s.MemoryStore.BlockResourceRetentionClaim(ctx, resourceID, claimedAt)
}

func (s *failingPurgeRetentionStore) PurgeClaimedResource(context.Context, string, time.Time) (bool, error) {
	return false, s.err
}
