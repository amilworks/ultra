package httpapi

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

// RetentionGCStore is the catalog surface the retention GC needs. Claims are
// atomic, exact, and stale-reclaimable so multiple control-plane replicas cannot
// delete the same resource or race an undelete.
type RetentionGCStore interface {
	ClaimResourcesPastRetention(context.Context, time.Duration, int) ([]domain.ResourceRecord, error)
	ReleaseResourceRetentionClaim(context.Context, string, time.Time) (bool, error)
	BlockResourceRetentionClaim(context.Context, string, time.Time) (bool, error)
	PurgeClaimedResource(context.Context, string, time.Time) (bool, error)
}

// retentionClaimRenewalStore is implemented by production stores so long
// filesystem cleanup cannot silently outlive its catalog lease. Test doubles
// that do not model elapsed claims can continue implementing RetentionGCStore
// alone.
type retentionClaimRenewalStore interface {
	RenewResourceRetentionClaim(context.Context, string, time.Time) (time.Time, bool, error)
}

// ResourceLifecycleFenceStore supplies a stable, bounded catalog walk for
// restoring publication fences that predate the filesystem lifecycle contract.
type ResourceLifecycleFenceStore interface {
	ListResourceLifecycleFenceCandidates(context.Context, string, int) ([]domain.ResourceRecord, error)
	GetResourceLifecycleStatus(context.Context, string) (string, bool, error)
}

const (
	resourceRetentionClaimLease          = 15 * time.Minute
	resourceRetentionClaimHeartbeat      = resourceRetentionClaimLease / 3
	resourceRetentionClassificationPages = 32
	resourceRetentionMaxBatch            = 100
	resourceRetentionMaxSweepBatches     = 100
	resourceRetentionLockWait            = 2 * time.Second
	resourceLifecycleFenceBackfillBatch  = 500
	resourceTerminalCleanupBatch         = 64
	resourceTerminalCleanupLockWait      = 50 * time.Millisecond
)

type resourceRetentionClaimGuard struct {
	cancel context.CancelFunc
	done   chan struct{}

	mu    sync.Mutex
	token time.Time
	lost  bool
	err   error
}

func startResourceRetentionClaimGuard(
	ctx context.Context,
	store RetentionGCStore,
	resourceID string,
	claimedAt time.Time,
) *resourceRetentionClaimGuard {
	return startResourceRetentionClaimGuardWithInterval(
		ctx,
		store,
		resourceID,
		claimedAt,
		resourceRetentionClaimHeartbeat,
	)
}

func startResourceRetentionClaimGuardWithInterval(
	ctx context.Context,
	store RetentionGCStore,
	resourceID string,
	claimedAt time.Time,
	heartbeatInterval time.Duration,
) *resourceRetentionClaimGuard {
	guard := &resourceRetentionClaimGuard{token: claimedAt, done: make(chan struct{})}
	renewer, ok := store.(retentionClaimRenewalStore)
	if !ok {
		close(guard.done)
		return guard
	}
	if heartbeatInterval <= 0 {
		heartbeatInterval = resourceRetentionClaimHeartbeat
	}
	heartbeatCtx, cancel := context.WithCancel(ctx)
	guard.cancel = cancel
	go func() {
		defer close(guard.done)
		ticker := time.NewTicker(heartbeatInterval)
		defer ticker.Stop()
		for {
			select {
			case <-heartbeatCtx.Done():
				return
			case <-ticker.C:
				guard.mu.Lock()
				token := guard.token
				guard.mu.Unlock()
				renewedAt, renewed, err := renewer.RenewResourceRetentionClaim(
					heartbeatCtx,
					resourceID,
					token,
				)
				if err != nil || !renewed {
					guard.mu.Lock()
					guard.lost = true
					guard.err = err
					guard.mu.Unlock()
					return
				}
				guard.mu.Lock()
				guard.token = renewedAt
				guard.mu.Unlock()
			}
		}
	}()
	return guard
}

func (guard *resourceRetentionClaimGuard) stop() (time.Time, bool, error) {
	if guard == nil {
		return time.Time{}, true, errors.New("retention claim guard is unavailable")
	}
	if guard.cancel != nil {
		guard.cancel()
	}
	<-guard.done
	guard.mu.Lock()
	defer guard.mu.Unlock()
	return guard.token, guard.lost, guard.err
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

// resourceSourceRelativePath resolves only storage owned by the configured
// upload root. A configured external locator is authoritative data that this
// process cannot prove it deleted, so retention must preserve the catalog
// tombstone instead of purging the row and allowing stale publishers to run.
func resourceSourceRelativePath(root string, resource domain.ResourceRecord) (string, error) {
	storageURI := strings.TrimSpace(resource.StorageURI)
	if storageURI != "" {
		parsed, err := url.Parse(storageURI)
		if err != nil || parsed.Scheme != "file" || parsed.Host != "" || parsed.Path == "" {
			return "", fmt.Errorf("resource storage URI is not managed by the upload root")
		}
		if relative, ok := relativePathUnderRoot(root, parsed.Path); ok {
			return relative, nil
		}
		return "", fmt.Errorf("resource storage URI is outside the upload root")
	}
	storagePath := strings.TrimSpace(resource.StoragePath)
	if storagePath == "" {
		return "", fmt.Errorf("resource has no managed source locator")
	}
	var candidate string
	if filepath.IsAbs(storagePath) {
		candidate = filepath.Clean(storagePath)
	} else {
		clean := filepath.Clean(storagePath)
		if clean == "." || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
			return "", fmt.Errorf("resource storage path escapes the upload root")
		}
		candidate = filepath.Join(root, clean)
	}
	if relative, ok := relativePathUnderRoot(root, candidate); ok {
		return relative, nil
	}
	return "", fmt.Errorf("resource storage path is outside the upload root")
}

func reconcileResourceFilesystemFenceState(root *os.Root, resourceID string, status string) error {
	switch status {
	case domain.ResourceStatusActive:
		permanent, err := resourceFilesystemPermanentlyTombstoned(root, resourceID)
		if err != nil {
			return err
		}
		if permanent {
			return errors.New("active resource has a permanent lifecycle tombstone")
		}
		return removeResourceFilesystemDeleteFence(root, resourceID)
	case domain.ResourceStatusDeleted:
		permanent, err := resourceFilesystemPermanentlyTombstoned(root, resourceID)
		if err != nil {
			return err
		}
		if permanent {
			return errors.New("restorable resource has a permanent lifecycle tombstone")
		}
		return ensureResourceFilesystemDeleteFence(root, resourceID)
	case domain.ResourceStatusPurging, domain.ResourceStatusRetentionBlocked:
		if err := ensureResourceFilesystemTombstone(root, resourceID); err != nil {
			return err
		}
		return removeResourceFilesystemDeleteFence(root, resourceID)
	default:
		return fmt.Errorf("unexpected lifecycle fence status %q", status)
	}
}

// ReconcileResourceLifecycleSoftDeleteFences repairs only durable reversible
// markers. Unlike the full legacy backfill, this work is proportional to
// outstanding deletes and is safe to run periodically on every replica. It
// closes the HA crash boundary where one replica dies between the filesystem
// marker and catalog transition while another replica keeps serving.
func ReconcileResourceLifecycleSoftDeleteFences(
	ctx context.Context,
	store ResourceLifecycleFenceStore,
	root string,
) (int, error) {
	if store == nil {
		return 0, errors.New("resource lifecycle fence store is unavailable")
	}
	uploadRoot, err := os.OpenRoot(root)
	if errors.Is(err, os.ErrNotExist) {
		return 0, nil
	}
	if err != nil {
		return 0, fmt.Errorf("open upload root for soft-delete reconciliation: %w", err)
	}
	defer uploadRoot.Close()
	softFenceIDs, scanDiagnostics, scanErr := scanResourceFilesystemSoftDeleteFenceIDs(uploadRoot)
	reconciled := 0
	var reconciliationErrors []error
	if len(scanDiagnostics) > 0 {
		slog.WarnContext(
			ctx,
			"resource lifecycle soft-delete marker diagnostics",
			"error",
			errors.Join(scanDiagnostics...),
		)
	}
	if scanErr != nil {
		reconciliationErrors = append(reconciliationErrors, scanErr)
	}
	for _, resourceID := range softFenceIDs {
		if !domain.IsCanonicalResourceID(resourceID) {
			slog.WarnContext(ctx, "resource lifecycle reconciliation skipped non-canonical soft-delete fence",
				"resource_id", resourceID)
			continue
		}
		lockCtx, cancelLock := context.WithTimeout(ctx, resourceRetentionLockWait)
		lock, lockErr := acquireResourceLifecycleCleanupLock(lockCtx, uploadRoot, resourceID, "")
		cancelLock()
		if lockErr != nil {
			reconciliationErrors = append(
				reconciliationErrors,
				fmt.Errorf("acquire soft-delete fence reconciliation lock for %q: %w", resourceID, lockErr),
			)
			continue
		}
		status, found, stateErr := store.GetResourceLifecycleStatus(ctx, resourceID)
		if stateErr != nil {
			reconciliationErrors = append(
				reconciliationErrors,
				fmt.Errorf("re-read lifecycle state for soft-delete fence %q: %w", resourceID, stateErr),
			)
			if releaseErr := lock.release(); releaseErr != nil {
				reconciliationErrors = append(
					reconciliationErrors,
					fmt.Errorf("release failed soft-delete fence lock for %q: %w", resourceID, releaseErr),
				)
			}
			continue
		}
		if !found {
			if releaseErr := lock.release(); releaseErr != nil {
				reconciliationErrors = append(
					reconciliationErrors,
					fmt.Errorf("release orphan soft-delete fence lock for %q: %w", resourceID, releaseErr),
				)
			}
			continue
		}
		status = strings.TrimSpace(status)
		markerErr := reconcileResourceFilesystemFenceState(uploadRoot, resourceID, status)
		releaseErr := lock.release()
		if markerErr != nil {
			reconciliationErrors = append(
				reconciliationErrors,
				fmt.Errorf("reconcile existing soft-delete fence for %q: %w", resourceID, markerErr),
			)
		}
		if releaseErr != nil {
			reconciliationErrors = append(
				reconciliationErrors,
				fmt.Errorf("release soft-delete fence reconciliation lock for %q: %w", resourceID, releaseErr),
			)
		}
		if markerErr == nil && releaseErr == nil && status == domain.ResourceStatusActive {
			reconciled++
		}
	}
	return reconciled, errors.Join(reconciliationErrors...)
}

type resourceTerminalCleanupTarget struct {
	resourceID    string
	workLocks     []*resourceLifecycleLock
	lifecycleLock *resourceLifecycleLock
}

func releaseResourceTerminalCleanupTarget(target resourceTerminalCleanupTarget) error {
	var releaseErr error
	if target.lifecycleLock != nil {
		releaseErr = target.lifecycleLock.release()
	}
	for index := len(target.workLocks) - 1; index >= 0; index-- {
		releaseErr = errors.Join(releaseErr, target.workLocks[index].release())
	}
	return releaseErr
}

// reconcileResourceLifecyclePermanentTombstoneBatch drains one bounded marker
// batch. It acquires every successful target's publication locks before taking
// one shared derivative inventory, keeping the batch O(D+B) instead of O(D*B).
func reconcileResourceLifecyclePermanentTombstoneBatch(
	ctx context.Context,
	uploadRoot *os.Root,
	resourceIDs []string,
) (int, error) {
	reconciled := 0
	var reconciliationErrors []error
	targets := make([]resourceTerminalCleanupTarget, 0, len(resourceIDs))
	for _, resourceID := range resourceIDs {
		if !domain.IsCanonicalResourceID(resourceID) {
			reconciliationErrors = append(
				reconciliationErrors,
				fmt.Errorf("terminal reconciliation skipped non-canonical resource %q", resourceID),
			)
			continue
		}
		var workLocks []*resourceLifecycleLock
		workFailed := false
		for _, kind := range []string{"pyramid", "nifti"} {
			lockCtx, cancelLock := context.WithTimeout(ctx, resourceTerminalCleanupLockWait)
			lock, lockErr := acquireResourceDerivationCleanupLock(
				lockCtx, uploadRoot, resourceID, kind,
			)
			cancelLock()
			if lockErr != nil {
				reconciliationErrors = append(
					reconciliationErrors,
					fmt.Errorf("drain %s work for terminal resource %q: %w", kind, resourceID, lockErr),
				)
				workFailed = true
				break
			}
			workLocks = append(workLocks, lock)
		}
		if workFailed {
			for index := len(workLocks) - 1; index >= 0; index-- {
				_ = workLocks[index].release()
			}
			continue
		}
		lifecycleCtx, cancelLifecycle := context.WithTimeout(ctx, resourceTerminalCleanupLockWait)
		lifecycleLock, lifecycleErr := acquireResourceLifecycleCleanupLock(
			lifecycleCtx, uploadRoot, resourceID, "",
		)
		cancelLifecycle()
		if lifecycleErr != nil {
			reconciliationErrors = append(
				reconciliationErrors,
				fmt.Errorf("acquire terminal lifecycle lock for %q: %w", resourceID, lifecycleErr),
			)
			for index := len(workLocks) - 1; index >= 0; index-- {
				_ = workLocks[index].release()
			}
			continue
		}
		targets = append(targets, resourceTerminalCleanupTarget{
			resourceID:    resourceID,
			workLocks:     workLocks,
			lifecycleLock: lifecycleLock,
		})
	}
	if len(targets) == 0 {
		return 0, errors.Join(reconciliationErrors...)
	}
	lockedResourceIDs := make([]string, 0, len(targets))
	for _, target := range targets {
		lockedResourceIDs = append(lockedResourceIDs, target.resourceID)
	}
	derivativesByResource, inventoryErr := scanOwnedDerivativeNamesForResources(
		uploadRoot,
		lockedResourceIDs,
	)
	if inventoryErr != nil {
		for _, target := range targets {
			reconciliationErrors = append(
				reconciliationErrors,
				releaseResourceTerminalCleanupTarget(target),
			)
		}
		return 0, errors.Join(
			errors.Join(reconciliationErrors...),
			fmt.Errorf("inventory terminal derivative namespaces: %w", inventoryErr),
		)
	}
	for _, target := range targets {
		resourceID := target.resourceID
		_, cleanupErr := finalizeOwnedResourceNamespaceFromInventory(
			uploadRoot,
			resourceID,
			derivativesByResource[resourceID],
		)
		if cleanupErr == nil {
			cleanupErr = removeResourceFilesystemTerminalCleanupMarker(uploadRoot, resourceID)
		}
		if cleanupErr == nil {
			cleanupErr = target.lifecycleLock.removePath()
		}
		releaseErr := releaseResourceTerminalCleanupTarget(target)
		if cleanupErr != nil || releaseErr != nil {
			reconciliationErrors = append(
				reconciliationErrors,
				fmt.Errorf(
					"finalize terminal resource %q: %w",
					resourceID,
					errors.Join(cleanupErr, releaseErr),
				),
			)
			continue
		}
		reconciled++
	}
	return reconciled, errors.Join(reconciliationErrors...)
}

// ReconcileResourceLifecyclePermanentTombstones drains one bounded batch of
// producer work and reclaims stages created after the initial purge inventory.
// The permanent tombstone remains durable, so every skipped marker is retryable.
func ReconcileResourceLifecyclePermanentTombstones(
	ctx context.Context,
	root string,
) (int, error) {
	uploadRoot, err := os.OpenRoot(root)
	if errors.Is(err, os.ErrNotExist) {
		return 0, nil
	}
	if err != nil {
		return 0, fmt.Errorf("open upload root for terminal reconciliation: %w", err)
	}
	defer uploadRoot.Close()
	scanner, err := openResourceFilesystemMarkerScanner(
		uploadRoot,
		filepath.Join(resourceTombstoneDir, resourceTerminalCleanupDir),
		"terminal cleanup marker",
	)
	if err != nil || scanner == nil {
		return 0, err
	}
	defer scanner.close() //nolint:errcheck // scanner reads are reported directly
	resourceIDs, diagnostics, _, scanErr := scanner.next(resourceTerminalCleanupBatch)
	if len(diagnostics) > 0 {
		slog.WarnContext(
			ctx,
			"resource lifecycle terminal marker diagnostics",
			"error",
			errors.Join(diagnostics...),
		)
	}
	if scanErr != nil {
		return 0, scanErr
	}
	return reconcileResourceLifecyclePermanentTombstoneBatch(ctx, uploadRoot, resourceIDs)
}

// RunResourceLifecycleFenceRepair continuously heals reversible marker/catalog
// crash boundaries on surviving replicas without repeating the full terminal
// catalog backfill.
func RunResourceLifecycleFenceRepair(
	ctx context.Context,
	store ResourceLifecycleFenceStore,
	root string,
	interval time.Duration,
) {
	if interval <= 0 {
		interval = time.Minute
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			reconciled, err := ReconcileResourceLifecycleSoftDeleteFences(ctx, store, root)
			if err != nil {
				slog.WarnContext(ctx, "resource lifecycle soft-delete reconciliation failed", "error", err)
			}
			if reconciled > 0 {
				slog.InfoContext(ctx, "resource lifecycle soft-delete fences reconciled", "resources", reconciled)
			}
		}
	}
}

// RunResourceLifecycleTerminalRepair advances a retained directory cursor by
// one bounded batch per tick. It is intentionally independent of reversible
// soft-fence repair so busy terminal producers cannot starve restore safety.
func RunResourceLifecycleTerminalRepair(
	ctx context.Context,
	root string,
	interval time.Duration,
) {
	if interval <= 0 {
		interval = time.Minute
	}
	var uploadRoot *os.Root
	var scanner *resourceFilesystemMarkerScanner
	reset := func() {
		if scanner != nil {
			_ = scanner.close()
			scanner = nil
		}
		if uploadRoot != nil {
			_ = uploadRoot.Close()
			uploadRoot = nil
		}
	}
	defer reset()
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if uploadRoot == nil {
				opened, err := os.OpenRoot(root)
				if errors.Is(err, os.ErrNotExist) {
					continue
				}
				if err != nil {
					slog.WarnContext(ctx, "open upload root for terminal repair", "error", err)
					continue
				}
				uploadRoot = opened
			}
			if scanner == nil {
				opened, err := openResourceFilesystemMarkerScanner(
					uploadRoot,
					filepath.Join(resourceTombstoneDir, resourceTerminalCleanupDir),
					"terminal cleanup marker",
				)
				if err != nil {
					slog.WarnContext(ctx, "resource lifecycle terminal marker scan failed", "error", err)
					reset()
					continue
				}
				if opened == nil {
					reset()
					continue
				}
				scanner = opened
			}
			resourceIDs, diagnostics, exhausted, scanErr := scanner.next(resourceTerminalCleanupBatch)
			if len(diagnostics) > 0 {
				slog.WarnContext(
					ctx,
					"resource lifecycle terminal marker diagnostics",
					"error",
					errors.Join(diagnostics...),
				)
			}
			if scanErr != nil {
				slog.WarnContext(ctx, "resource lifecycle terminal marker scan failed", "error", scanErr)
				reset()
				continue
			}
			reconciled, reconcileErr := reconcileResourceLifecyclePermanentTombstoneBatch(
				ctx,
				uploadRoot,
				resourceIDs,
			)
			if reconcileErr != nil {
				slog.WarnContext(ctx, "resource lifecycle terminal reconciliation failed", "error", reconcileErr)
			}
			if reconciled > 0 {
				slog.InfoContext(ctx, "resource lifecycle terminal namespaces reconciled", "resources", reconciled)
			}
			if exhausted {
				reset()
			}
		}
	}
}

// ReconcileResourceLifecycleFences backfills the filesystem half of catalog
// deletion state before workers are allowed to publish. It is intentionally
// idempotent: every process startup may walk the same deleted rows, while the
// marker creation itself remains exclusive and durable under the shared lock.
func ReconcileResourceLifecycleFences(
	ctx context.Context,
	store ResourceLifecycleFenceStore,
	root string,
	batch int,
) (int, error) {
	if store == nil {
		return 0, errors.New("resource lifecycle fence store is unavailable")
	}
	if batch <= 0 {
		batch = resourceLifecycleFenceBackfillBatch
	}
	if batch > resourceLifecycleFenceBackfillBatch {
		batch = resourceLifecycleFenceBackfillBatch
	}
	var uploadRoot *os.Root
	defer func() {
		if uploadRoot != nil {
			_ = uploadRoot.Close()
		}
	}()
	ensureUploadRoot := func() error {
		if uploadRoot != nil {
			return nil
		}
		if err := os.MkdirAll(root, 0o755); err != nil {
			return fmt.Errorf("prepare upload root for lifecycle reconciliation: %w", err)
		}
		opened, err := os.OpenRoot(root)
		if err != nil {
			return fmt.Errorf("open upload root for lifecycle reconciliation: %w", err)
		}
		uploadRoot = opened
		return nil
	}
	reconciled, err := ReconcileResourceLifecycleSoftDeleteFences(ctx, store, root)
	if err != nil {
		return 0, err
	}

	afterResourceID := ""
	for {
		candidates, listErr := store.ListResourceLifecycleFenceCandidates(ctx, afterResourceID, batch)
		if listErr != nil {
			return reconciled, fmt.Errorf("list resource lifecycle fence candidates: %w", listErr)
		}
		if len(candidates) == 0 {
			return reconciled, nil
		}
		for _, resource := range candidates {
			resourceID := resource.ResourceID
			if resourceID <= afterResourceID {
				return reconciled, errors.New("resource lifecycle fence candidates are not in stable ascending order")
			}
			afterResourceID = resourceID
			if !domain.IsCanonicalResourceID(resourceID) {
				slog.WarnContext(ctx, "resource lifecycle fence backfill skipped non-canonical legacy resource",
					"resource_id", resourceID)
				continue
			}
			if err := ensureUploadRoot(); err != nil {
				return reconciled, err
			}

			lockCtx, cancelLock := context.WithTimeout(ctx, resourceRetentionLockWait)
			lock, lockErr := acquireResourceLifecycleCleanupLock(lockCtx, uploadRoot, resourceID, "")
			cancelLock()
			if lockErr != nil {
				return reconciled, fmt.Errorf("acquire lifecycle fence backfill lock for %q: %w", resourceID, lockErr)
			}

			status, found, stateErr := store.GetResourceLifecycleStatus(ctx, resourceID)
			if stateErr != nil {
				_ = lock.release()
				return reconciled, fmt.Errorf("re-read lifecycle state for %q: %w", resourceID, stateErr)
			}
			if !found {
				if releaseErr := lock.release(); releaseErr != nil {
					return reconciled, fmt.Errorf("release lifecycle fence backfill lock for removed %q: %w", resourceID, releaseErr)
				}
				continue
			}
			status = strings.TrimSpace(status)
			markerErr := reconcileResourceFilesystemFenceState(uploadRoot, resourceID, status)
			releaseErr := lock.release()
			if markerErr != nil {
				return reconciled, fmt.Errorf("publish lifecycle fence for %q: %w", resourceID, markerErr)
			}
			if releaseErr != nil {
				return reconciled, fmt.Errorf("release lifecycle fence backfill lock for %q: %w", resourceID, releaseErr)
			}
			if status != domain.ResourceStatusActive {
				reconciled++
			}
		}
		if len(candidates) < batch {
			return reconciled, nil
		}
	}
}

// ReclaimExpiredResources permanently removes resources whose undelete window has elapsed.
// One GC replica first claims each row, then holds the same cross-process lifecycle lock as
// derivative publishers from filesystem cleanup through conditional catalog purge.
func ReclaimExpiredResources(ctx context.Context, store RetentionGCStore, root string, batch int) (reclaimed int, bytes int64, err error) {
	return reclaimExpiredResourcesWithProgress(ctx, store, root, batch, nil)
}

func reclaimExpiredResourcesWithProgress(
	ctx context.Context,
	store RetentionGCStore,
	root string,
	batch int,
	processed *int,
) (reclaimed int, bytes int64, err error) {
	if batch <= 0 {
		batch = resourceRetentionMaxBatch
	}
	if batch > resourceRetentionMaxBatch {
		batch = resourceRetentionMaxBatch
	}
	uploadRoot, err := os.OpenRoot(root)
	if err != nil {
		return 0, 0, err
	}
	defer uploadRoot.Close()
	type cleanupTarget struct {
		resource       domain.ResourceRecord
		sourceRelative string
		lock           *resourceLifecycleLock
		claim          *resourceRetentionClaimGuard
	}
	releaseTargetClaim := func(target cleanupTarget) {
		token, lost, renewErr := target.claim.stop()
		if renewErr != nil {
			slog.WarnContext(ctx, "retention gc: claim heartbeat failed",
				"resource_id", target.resource.ResourceID, "error", renewErr)
		}
		if lost {
			return
		}
		if released, releaseErr := store.ReleaseResourceRetentionClaim(ctx, target.resource.ResourceID, token); releaseErr != nil {
			slog.WarnContext(ctx, "retention gc: failed to release resource claim",
				"resource_id", target.resource.ResourceID, "error", releaseErr)
		} else if !released {
			slog.WarnContext(ctx, "retention gc: resource claim was no longer current",
				"resource_id", target.resource.ResourceID)
		}
	}
	targets := make([]cleanupTarget, 0, batch)
	for page := 0; page < resourceRetentionClassificationPages && len(targets) < batch; page++ {
		claimLimit := batch - len(targets)
		claimed, claimErr := store.ClaimResourcesPastRetention(ctx, resourceRetentionClaimLease, claimLimit)
		if claimErr != nil {
			for _, target := range targets {
				releaseTargetClaim(target)
				_ = target.lock.removePath()
				_ = target.lock.release()
			}
			return 0, 0, claimErr
		}
		if len(claimed) == 0 {
			break
		}
		madeProgress := false
		for _, resource := range claimed {
			claimedAt := resource.UpdatedAt
			releaseUnmodifiedClaim := func() {
				if released, releaseErr := store.ReleaseResourceRetentionClaim(ctx, resource.ResourceID, claimedAt); releaseErr != nil {
					slog.WarnContext(ctx, "retention gc: failed to release resource claim",
						"resource_id", resource.ResourceID, "error", releaseErr)
				} else if !released {
					slog.WarnContext(ctx, "retention gc: resource claim was no longer current",
						"resource_id", resource.ResourceID)
				}
			}
			blockClaim := func(reason error) {
				blocked, blockErr := store.BlockResourceRetentionClaim(ctx, resource.ResourceID, claimedAt)
				if blockErr != nil {
					slog.WarnContext(ctx, "retention gc: failed to block unsafe resource claim",
						"resource_id", resource.ResourceID, "reason", reason, "error", blockErr)
				} else if !blocked {
					slog.WarnContext(ctx, "retention gc: unsafe resource claim was no longer current",
						"resource_id", resource.ResourceID, "reason", reason)
				}
				if blocked {
					madeProgress = true
					if processed != nil {
						(*processed)++
					}
				}
			}
			blockClaimWithPermanentFence := func(reason error, lock *resourceLifecycleLock) {
				if lock == nil {
					lockCtx, cancelLock := context.WithTimeout(ctx, resourceRetentionLockWait)
					var lockErr error
					lock, lockErr = acquireResourceLifecycleCleanupLock(lockCtx, uploadRoot, resource.ResourceID, "")
					cancelLock()
					if lockErr != nil {
						slog.WarnContext(ctx, "retention gc: failed to fence blocked resource",
							"resource_id", resource.ResourceID, "reason", reason, "error", lockErr)
						releaseUnmodifiedClaim()
						return
					}
				}
				if tombstoneErr := ensureResourceFilesystemTombstone(uploadRoot, resource.ResourceID); tombstoneErr != nil {
					slog.WarnContext(ctx, "retention gc: failed to publish blocked-resource tombstone",
						"resource_id", resource.ResourceID, "reason", reason, "error", tombstoneErr)
					releaseUnmodifiedClaim()
					_ = lock.removePath()
					_ = lock.release()
					return
				}
				// The permanent marker is durable before the catalog becomes terminal,
				// so an already-queued publisher cannot orphan a new generation after
				// an external or unsafe source is classified as retention-blocked.
				blockClaim(reason)
				if fenceErr := removeResourceFilesystemDeleteFence(uploadRoot, resource.ResourceID); fenceErr != nil {
					slog.WarnContext(ctx, "retention gc: failed to retire blocked-resource delete fence",
						"resource_id", resource.ResourceID, "error", fenceErr)
				}
				if unlinkErr := lock.removePath(); unlinkErr != nil {
					slog.WarnContext(ctx, "retention gc: failed to remove blocked-resource lifecycle lock",
						"resource_id", resource.ResourceID, "error", unlinkErr)
				}
				if releaseErr := lock.release(); releaseErr != nil {
					slog.WarnContext(ctx, "retention gc: failed to release blocked-resource lifecycle lock",
						"resource_id", resource.ResourceID, "error", releaseErr)
				}
			}
			if !domain.IsCanonicalResourceID(resource.ResourceID) {
				blockClaim(errors.New("resource id is not canonical"))
				continue
			}
			sourceRelative, sourceErr := resourceSourceRelativePath(root, resource)
			if sourceErr != nil {
				slog.WarnContext(ctx, "retention gc: resource storage is not locally managed",
					"resource_id", resource.ResourceID, "error", sourceErr)
				blockClaimWithPermanentFence(sourceErr, nil)
				continue
			}
			if sourceErr := validateManagedSourcePath(uploadRoot, sourceRelative, true); sourceErr != nil {
				slog.WarnContext(ctx, "retention gc: resource source is not exclusively managed",
					"resource_id", resource.ResourceID, "error", sourceErr)
				blockClaimWithPermanentFence(sourceErr, nil)
				continue
			}
			lockCtx, cancelLock := context.WithTimeout(ctx, resourceRetentionLockWait)
			lock, lockErr := acquireResourceLifecycleCleanupLock(lockCtx, uploadRoot, resource.ResourceID, "")
			cancelLock()
			if lockErr != nil {
				slog.WarnContext(ctx, "retention gc: failed to acquire resource lifecycle lock",
					"resource_id", resource.ResourceID, "error", lockErr)
				releaseUnmodifiedClaim()
				continue
			}
			if sourceErr := validateManagedSourcePath(uploadRoot, sourceRelative, true); sourceErr != nil {
				slog.WarnContext(ctx, "retention gc: resource source changed before cleanup",
					"resource_id", resource.ResourceID, "error", sourceErr)
				blockClaimWithPermanentFence(sourceErr, lock)
				continue
			}
			targets = append(targets, cleanupTarget{
				resource:       resource,
				sourceRelative: sourceRelative,
				lock:           lock,
				claim: startResourceRetentionClaimGuard(
					ctx,
					store,
					resource.ResourceID,
					claimedAt,
				),
			})
			madeProgress = true
		}
		if ctx.Err() != nil || len(claimed) < claimLimit || !madeProgress {
			break
		}
	}
	if ctx.Err() != nil {
		for _, target := range targets {
			releaseTargetClaim(target)
			_ = target.lock.removePath()
			_ = target.lock.release()
		}
		return 0, 0, ctx.Err()
	}
	if len(targets) == 0 {
		return 0, 0, nil
	}
	resourceIDs := make([]string, 0, len(targets))
	for _, target := range targets {
		resourceIDs = append(resourceIDs, target.resource.ResourceID)
	}
	derivativesByResource, inventoryErr := scanOwnedDerivativeNamesForResources(uploadRoot, resourceIDs)
	if inventoryErr != nil {
		for _, target := range targets {
			releaseTargetClaim(target)
			_ = target.lock.removePath()
			_ = target.lock.release()
		}
		return 0, 0, fmt.Errorf("inventory claimed derivative namespaces: %w", inventoryErr)
	}
	for _, target := range targets {
		resource := target.resource
		lock := target.lock
		if tombstoneErr := ensureResourceFilesystemTombstone(uploadRoot, resource.ResourceID); tombstoneErr != nil {
			slog.WarnContext(ctx, "retention gc: failed to publish resource tombstone",
				"resource_id", resource.ResourceID, "error", tombstoneErr)
			releaseTargetClaim(target)
			_ = lock.removePath()
			_ = lock.release()
			continue
		}
		if fenceErr := removeResourceFilesystemDeleteFence(uploadRoot, resource.ResourceID); fenceErr != nil {
			slog.WarnContext(ctx, "retention gc: failed to retire reversible delete fence",
				"resource_id", resource.ResourceID, "error", fenceErr)
			releaseTargetClaim(target)
			_ = lock.removePath()
			_ = lock.release()
			continue
		}
		freed, cleanupErr := removeOwnedResourceNamespaceFromInventory(
			uploadRoot,
			resource.ResourceID,
			target.sourceRelative,
			derivativesByResource[resource.ResourceID],
			resource.SizeBytes,
		)
		if cleanupErr != nil {
			slog.WarnContext(ctx, "retention gc: failed to remove resource namespace",
				"resource_id", resource.ResourceID, "error", cleanupErr)
			releaseTargetClaim(target)
			_ = lock.removePath()
			_ = lock.release()
			continue
		}
		claimedAt, claimLost, renewErr := target.claim.stop()
		if renewErr != nil {
			slog.WarnContext(ctx, "retention gc: claim heartbeat failed during cleanup",
				"resource_id", resource.ResourceID, "error", renewErr)
		}
		if claimLost {
			slog.WarnContext(ctx, "retention gc: claim changed during filesystem cleanup",
				"resource_id", resource.ResourceID)
			_ = lock.removePath()
			_ = lock.release()
			continue
		}
		purged, purgeErr := store.PurgeClaimedResource(ctx, resource.ResourceID, claimedAt)
		if purgeErr != nil {
			slog.WarnContext(ctx, "retention gc: failed to purge resource row",
				"resource_id", resource.ResourceID, "error", purgeErr)
			_, _ = store.ReleaseResourceRetentionClaim(ctx, resource.ResourceID, claimedAt)
			_ = lock.removePath()
			_ = lock.release()
			continue
		}
		if !purged {
			slog.WarnContext(ctx, "retention gc: resource claim changed before purge",
				"resource_id", resource.ResourceID)
			_ = lock.removePath()
			_ = lock.release()
			continue
		}
		if unlinkErr := lock.removePath(); unlinkErr != nil {
			slog.WarnContext(ctx, "retention gc: purged resource but failed to remove lifecycle lock",
				"resource_id", resource.ResourceID, "error", unlinkErr)
		}
		if releaseErr := lock.release(); releaseErr != nil {
			slog.WarnContext(ctx, "retention gc: failed to release resource lifecycle lock",
				"resource_id", resource.ResourceID, "error", releaseErr)
		}
		reclaimed++
		if processed != nil {
			(*processed)++
		}
		bytes += freed
		slog.InfoContext(ctx, "retention gc: reclaimed expired resource",
			"resource_id", resource.ResourceID, "bytes_freed", freed)
	}
	return reclaimed, bytes, nil
}

func reclaimRetentionBacklog(
	ctx context.Context,
	store RetentionGCStore,
	root string,
	batch int,
) (totalResources int, totalBytes int64, err error) {
	for sweepBatch := 0; sweepBatch < resourceRetentionMaxSweepBatches; sweepBatch++ {
		processed := 0
		reclaimed, bytes, reclaimErr := reclaimExpiredResourcesWithProgress(ctx, store, root, batch, &processed)
		if reclaimErr != nil {
			return totalResources, totalBytes, reclaimErr
		}
		totalResources += reclaimed
		totalBytes += bytes
		// Blocked poison rows are forward progress too. Continue draining when
		// they consumed a full classification window so managed resources just
		// beyond that window do not wait for the next hourly tick.
		if processed < batch {
			break
		}
	}
	return totalResources, totalBytes, nil
}

// RunRetentionGC periodically reclaims expired resources until ctx is canceled. Wired from
// the app only when retention GC is explicitly enabled (it permanently deletes data past
// the undelete window).
func RunRetentionGC(ctx context.Context, store RetentionGCStore, root string, interval time.Duration, batch int) {
	if interval <= 0 {
		interval = time.Hour
	}
	if batch <= 0 {
		batch = resourceRetentionMaxBatch
	}
	if batch > resourceRetentionMaxBatch {
		slog.WarnContext(ctx, "retention gc batch clamped", "requested_batch", batch, "max_batch", resourceRetentionMaxBatch)
		batch = resourceRetentionMaxBatch
	}
	slog.InfoContext(ctx, "retention gc enabled", "interval", interval.String(), "batch", batch)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	runSweep := func() {
		totalResources, totalBytes, err := reclaimRetentionBacklog(ctx, store, root, batch)
		if err != nil {
			slog.WarnContext(ctx, "retention gc run failed", "error", err)
		}
		if totalResources > 0 {
			slog.InfoContext(ctx, "retention gc run", "reclaimed_resources", totalResources, "reclaimed_bytes", totalBytes)
		}
	}
	// Reclaim an existing backlog immediately instead of waiting a full interval
	// after every process restart.
	runSweep()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			runSweep()
		}
	}
}
