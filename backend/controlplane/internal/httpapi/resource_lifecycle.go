package httpapi

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

const (
	resourceDerivedDir               = "derived"
	resourceMetaDir                  = ".meta"
	resourceLockDir                  = ".locks"
	resourceTombstoneDir             = ".tombstones"
	resourceStagingDir               = ".staging"
	resourcePermanentTombstoneDir    = "permanent"
	resourceSoftDeleteFenceDir       = "deleted"
	resourceTerminalCleanupDir       = "cleanup"
	resourceCatalogReconciliationDir = ".catalog-reconcile"
)

var errResourceLifecycleTombstoned = errors.New("resource lifecycle is deletion-fenced")
var errResourceLifecycleBusy = errors.New("resource lifecycle is busy; retry the request")

const resourceLifecycleMutationLockWait = time.Second
const resourceLifecycleBulkMutationTimeout = 30 * time.Second
const resourceLifecycleCommitResolutionTimeout = 2 * time.Second

type resourceLifecycleLock struct {
	directory *os.Root
	file      *os.File
	name      string
}

func resourceLifecycleLockName(resourceID string) string {
	return "." + resourceID + "__pyramid.lock"
}

func resourceDerivationLockName(resourceID, kind string) (string, error) {
	if !safeResourceLifecycleID(resourceID) {
		return "", errors.New("invalid resource derivation lock target")
	}
	switch kind {
	case "pyramid", "nifti", "scene3d":
		return "." + resourceID + "__" + kind + ".work.lock", nil
	default:
		return "", errors.New("invalid resource derivation lock kind")
	}
}

func resourceDeletionIntentName(resourceID string) string {
	return resourceID + ".delete.json"
}

func resourceCatalogReconciliationIntentName(resourceID string) string {
	return resourceID + ".catalog-reconcile.json"
}

func resourceFilesystemTombstoneName(resourceID string) string {
	return filepath.Join(resourcePermanentTombstoneDir, resourceID)
}

func resourceFilesystemDeleteFenceName(resourceID string) string {
	return filepath.Join(resourceSoftDeleteFenceDir, resourceID)
}

func resourceFilesystemTerminalCleanupName(resourceID string) string {
	return filepath.Join(resourceTerminalCleanupDir, resourceID)
}

// safeResourceLifecycleID is deliberately stricter than safeUploadID. Upload
// IDs are also used as path components by retention cleanup, where filepath's
// special "." and ".." components must never be accepted as literal IDs.
func safeResourceLifecycleID(resourceID string) bool {
	return safeUploadID(resourceID) &&
		resourceID != "." &&
		resourceID != ".." &&
		filepath.Clean(resourceID) == resourceID &&
		filepath.Base(resourceID) == resourceID
}

func resourceBundleRelativeRoot(resourceID string) (string, error) {
	if !safeResourceLifecycleID(resourceID) {
		return "", errors.New("unsafe resource bundle id")
	}
	bundleRoot := filepath.Join(bundlesDirName, resourceID)
	if filepath.Dir(bundleRoot) != bundlesDirName || filepath.Base(bundleRoot) != resourceID {
		return "", errors.New("resource bundle target is not an exact bundle child")
	}
	return bundleRoot, nil
}

func resourceStagingRelativeRoot(resourceID string) (string, error) {
	if !safeResourceLifecycleID(resourceID) {
		return "", errors.New("unsafe resource staging id")
	}
	return filepath.Join(resourceStagingDir, resourceID), nil
}

// acquireResourceLifecycleLock obtains the same cross-process advisory lock
// used by strict derivative publication. When requireSource is non-empty the
// source is checked before opening and again after locking; a publisher waiting
// behind deletion therefore cannot recreate the resource's namespace.
func acquireResourceLifecycleLock(
	ctx context.Context,
	root *os.Root,
	resourceID string,
	requireSource string,
) (*resourceLifecycleLock, error) {
	return acquireResourceLifecycleLockMode(
		ctx, root, resourceID, requireSource, resourceLifecycleLockName(resourceID), false, false,
	)
}

// acquireResourceDerivationLock serializes expensive work across processes.
// Deletion never waits on this lock; final visibility still requires the
// lifecycle lock and therefore fails closed after a concurrent deletion.
func acquireResourceDerivationLock(
	ctx context.Context,
	root *os.Root,
	resourceID string,
	kind string,
	requireSource string,
) (*resourceLifecycleLock, error) {
	lockName, err := resourceDerivationLockName(resourceID, kind)
	if err != nil {
		return nil, err
	}
	return acquireResourceLifecycleLockMode(
		ctx, root, resourceID, requireSource, lockName, false, false,
	)
}

// acquireResourceDerivationCleanupLock drains a terminal resource's producer
// work queue after its permanent tombstone is durable. Publishers acquire work
// before lifecycle, so terminal reconciliation must use the same ordering.
func acquireResourceDerivationCleanupLock(
	ctx context.Context,
	root *os.Root,
	resourceID string,
	kind string,
) (*resourceLifecycleLock, error) {
	lockName, err := resourceDerivationLockName(resourceID, kind)
	if err != nil {
		return nil, err
	}
	return acquireResourceLifecycleLockMode(
		ctx, root, resourceID, "", lockName, true, true,
	)
}

// acquireResourceLifecycleMutationLock serializes soft-delete and restore with
// every source/derivative publisher. It accepts the reversible soft-delete
// fence for idempotent retries, but never crosses a permanent purge tombstone.
func acquireResourceLifecycleMutationLock(
	ctx context.Context,
	root *os.Root,
	resourceID string,
) (*resourceLifecycleLock, error) {
	return acquireResourceLifecycleLockMode(
		ctx, root, resourceID, "", resourceLifecycleLockName(resourceID), false, true,
	)
}

// acquireResourceLifecycleCleanupLock is the deletion-side form of the shared
// lifecycle fence. It may reopen a lock for a stale purging retry after the
// durable filesystem tombstone has already been published.
func acquireResourceLifecycleCleanupLock(
	ctx context.Context,
	root *os.Root,
	resourceID string,
	requireSource string,
) (*resourceLifecycleLock, error) {
	return acquireResourceLifecycleLockMode(
		ctx, root, resourceID, requireSource, resourceLifecycleLockName(resourceID), true, true,
	)
}

func acquireResourceLifecycleLockMode(
	ctx context.Context,
	root *os.Root,
	resourceID string,
	requireSource string,
	lockName string,
	allowPermanentTombstone bool,
	allowDeleteFence bool,
) (*resourceLifecycleLock, error) {
	if root == nil || !safeResourceLifecycleID(resourceID) {
		return nil, errors.New("invalid resource lifecycle lock target")
	}
	if err := rejectResourceLifecycleFence(
		root,
		resourceID,
		allowPermanentTombstone,
		allowDeleteFence,
	); err != nil {
		return nil, err
	}
	if err := root.MkdirAll(resourceLockDir, 0o700); err != nil {
		return nil, fmt.Errorf("prepare derivative lock directory: %w", err)
	}
	lockDirectory, err := root.OpenRoot(resourceLockDir)
	if err != nil {
		return nil, fmt.Errorf("open derivative lock directory: %w", err)
	}
	for {
		if err := rejectResourceLifecycleFence(
			root,
			resourceID,
			allowPermanentTombstone,
			allowDeleteFence,
		); err != nil {
			_ = lockDirectory.Close()
			return nil, err
		}
		if requireSource != "" {
			if _, err := root.Lstat(requireSource); err != nil {
				_ = lockDirectory.Close()
				return nil, fmt.Errorf("resource source is unavailable: %w", err)
			}
		}
		file, err := lockDirectory.OpenFile(
			lockName,
			os.O_CREATE|os.O_RDWR|syscall.O_NOFOLLOW,
			0o600,
		)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				// Another lifecycle process may be creating or replacing the
				// shared lock directory during first-start reconciliation.
				// Re-open the anchored directory instead of treating that
				// transient namespace race as a permanent startup failure.
				_ = lockDirectory.Close()
				select {
				case <-ctx.Done():
					return nil, ctx.Err()
				case <-time.After(10 * time.Millisecond):
				}
				if err := root.MkdirAll(resourceLockDir, 0o700); err != nil {
					return nil, fmt.Errorf("reprepare derivative lock directory: %w", err)
				}
				lockDirectory, err = root.OpenRoot(resourceLockDir)
				if err != nil {
					return nil, fmt.Errorf("reopen derivative lock directory: %w", err)
				}
				continue
			}
			_ = lockDirectory.Close()
			return nil, fmt.Errorf("open resource lifecycle lock: %w", err)
		}
		locked, err := flockContext(ctx, file)
		if err != nil {
			_ = file.Close()
			_ = lockDirectory.Close()
			return nil, err
		}
		if !locked {
			_ = file.Close()
			_ = lockDirectory.Close()
			return nil, context.Canceled
		}
		openedInfo, openedErr := file.Stat()
		pathInfo, pathErr := lockDirectory.Lstat(lockName)
		if openedErr == nil && pathErr == nil && openedInfo.Mode().IsRegular() &&
			pathInfo.Mode().IsRegular() && os.SameFile(openedInfo, pathInfo) {
			if err := rejectResourceLifecycleFence(
				root,
				resourceID,
				allowPermanentTombstone,
				allowDeleteFence,
			); err != nil {
				_ = syscall.Flock(int(file.Fd()), syscall.LOCK_UN)
				_ = file.Close()
				_ = lockDirectory.Close()
				return nil, err
			}
			if requireSource != "" {
				if _, err := root.Lstat(requireSource); err != nil {
					_ = syscall.Flock(int(file.Fd()), syscall.LOCK_UN)
					_ = file.Close()
					_ = lockDirectory.Close()
					return nil, fmt.Errorf("resource source disappeared while waiting for lifecycle lock: %w", err)
				}
			}
			return &resourceLifecycleLock{directory: lockDirectory, file: file, name: lockName}, nil
		}
		_ = syscall.Flock(int(file.Fd()), syscall.LOCK_UN)
		_ = file.Close()
		if openedErr != nil {
			_ = lockDirectory.Close()
			return nil, fmt.Errorf("inspect resource lifecycle lock: %w", openedErr)
		}
		if pathErr != nil && !errors.Is(pathErr, os.ErrNotExist) {
			_ = lockDirectory.Close()
			return nil, fmt.Errorf("inspect resource lifecycle lock path: %w", pathErr)
		}
		select {
		case <-ctx.Done():
			_ = lockDirectory.Close()
			return nil, ctx.Err()
		case <-time.After(10 * time.Millisecond):
		}
	}
}

func rejectResourceLifecycleFence(
	root *os.Root,
	resourceID string,
	allowPermanentTombstone bool,
	allowDeleteFence bool,
) error {
	if !allowPermanentTombstone {
		tombstoned, err := resourceFilesystemPermanentlyTombstoned(root, resourceID)
		if err != nil {
			return err
		}
		if tombstoned {
			return errResourceLifecycleTombstoned
		}
	}
	if !allowDeleteFence {
		fenced, err := resourceFilesystemSoftDeleteFenced(root, resourceID)
		if err != nil {
			return err
		}
		if fenced {
			return errResourceLifecycleTombstoned
		}
	}
	return nil
}

func resourceFilesystemTombstoned(root *os.Root, resourceID string) (bool, error) {
	tombstoned, err := resourceFilesystemPermanentlyTombstoned(root, resourceID)
	if err != nil || tombstoned {
		return tombstoned, err
	}
	return resourceFilesystemSoftDeleteFenced(root, resourceID)
}

func resourceFilesystemPermanentlyTombstoned(root *os.Root, resourceID string) (bool, error) {
	return resourceFilesystemMarkerExists(root, resourceID, resourceFilesystemTombstoneName(resourceID))
}

func resourceFilesystemSoftDeleteFenced(root *os.Root, resourceID string) (bool, error) {
	return resourceFilesystemMarkerExists(root, resourceID, resourceFilesystemDeleteFenceName(resourceID))
}

type resourceFilesystemMarkerScanner struct {
	directory *os.Root
	opened    *os.File
	label     string
}

func openResourceFilesystemMarkerScanner(
	root *os.Root,
	relativeDirectory string,
	label string,
) (*resourceFilesystemMarkerScanner, error) {
	if root == nil {
		return nil, errors.New("upload root is unavailable")
	}
	directory, err := root.OpenRoot(relativeDirectory)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("open %s directory: %w", label, err)
	}
	opened, err := directory.Open(".")
	if err != nil {
		_ = directory.Close()
		return nil, fmt.Errorf("open %s listing: %w", label, err)
	}
	return &resourceFilesystemMarkerScanner{
		directory: directory,
		opened:    opened,
		label:     label,
	}, nil
}

func (scanner *resourceFilesystemMarkerScanner) close() error {
	if scanner == nil {
		return nil
	}
	return errors.Join(scanner.opened.Close(), scanner.directory.Close())
}

// next advances one retained directory cursor by at most limit physical
// entries. Entry-local corruption is reported separately from a directory-wide
// read failure so one malformed marker cannot block valid repair or startup.
func (scanner *resourceFilesystemMarkerScanner) next(
	limit int,
) (resourceIDs []string, diagnostics []error, exhausted bool, err error) {
	if scanner == nil || scanner.opened == nil || scanner.directory == nil {
		return nil, nil, true, nil
	}
	if limit <= 0 {
		return nil, nil, false, errors.New("resource marker scan limit must be positive")
	}
	entries, readErr := scanner.opened.ReadDir(limit)
	for _, entry := range entries {
		info, statErr := scanner.directory.Lstat(entry.Name())
		if statErr != nil {
			if !errors.Is(statErr, os.ErrNotExist) {
				diagnostics = append(
					diagnostics,
					fmt.Errorf("inspect %s %q: %w", scanner.label, entry.Name(), statErr),
				)
			}
			continue
		}
		if !info.Mode().IsRegular() {
			diagnostics = append(
				diagnostics,
				fmt.Errorf("%s %q is not a regular file", scanner.label, entry.Name()),
			)
			continue
		}
		resourceIDs = append(resourceIDs, entry.Name())
	}
	if errors.Is(readErr, io.EOF) {
		exhausted = true
	} else if readErr != nil {
		err = fmt.Errorf("read %s directory: %w", scanner.label, readErr)
	}
	return resourceIDs, diagnostics, exhausted, err
}

func scanResourceFilesystemSoftDeleteFenceIDs(
	root *os.Root,
) ([]string, []error, error) {
	scanner, err := openResourceFilesystemMarkerScanner(
		root,
		filepath.Join(resourceTombstoneDir, resourceSoftDeleteFenceDir),
		"soft-delete fence",
	)
	if err != nil || scanner == nil {
		return nil, nil, err
	}
	defer scanner.close() //nolint:errcheck // a read-only scan reports read failures directly
	resourceIDs := []string{}
	var diagnostics []error
	for {
		batch, batchDiagnostics, exhausted, scanErr := scanner.next(256)
		resourceIDs = append(resourceIDs, batch...)
		diagnostics = append(diagnostics, batchDiagnostics...)
		if scanErr != nil {
			return resourceIDs, diagnostics, scanErr
		}
		if exhausted {
			break
		}
	}
	sort.Strings(resourceIDs)
	return resourceIDs, diagnostics, nil
}

func resourceFilesystemMarkerExists(root *os.Root, resourceID string, name string) (bool, error) {
	if root == nil || !safeResourceLifecycleID(resourceID) {
		return false, errors.New("invalid resource filesystem tombstone target")
	}
	info, err := root.Lstat(filepath.Join(resourceTombstoneDir, name))
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("inspect resource filesystem tombstone: %w", err)
	}
	if !info.Mode().IsRegular() {
		return false, errors.New("resource filesystem tombstone is not a regular file")
	}
	return true, nil
}

// ensureResourceFilesystemTombstone publishes the cross-process deletion
// evidence before authoritative bytes are removed. Publishers check this file
// before and after acquiring the lifecycle lock, so a stale worker cannot
// recreate a source or derivative after retention cleanup starts.
func ensureResourceFilesystemTombstone(root *os.Root, resourceID string) error {
	if err := ensureResourceFilesystemMarker(
		root,
		resourceID,
		resourceFilesystemTombstoneName(resourceID),
		[]byte("deleted\n"),
	); err != nil {
		return err
	}
	return ensureResourceFilesystemMarker(
		root,
		resourceID,
		resourceFilesystemTerminalCleanupName(resourceID),
		[]byte("pending\n"),
	)
}

// ensureResourceFilesystemDeleteFence publishes the reversible half of a soft
// delete before the catalog transition. It remains for the restore window and
// blocks stale publishers without destroying the retained source generation.
func ensureResourceFilesystemDeleteFence(root *os.Root, resourceID string) error {
	return ensureResourceFilesystemMarker(
		root,
		resourceID,
		resourceFilesystemDeleteFenceName(resourceID),
		[]byte("soft-deleted\n"),
	)
}

func ensureResourceFilesystemMarker(root *os.Root, resourceID string, name string, contents []byte) error {
	if root == nil || !safeResourceLifecycleID(resourceID) {
		return errors.New("invalid resource filesystem tombstone target")
	}
	markerKind := filepath.Dir(name)
	markerName := filepath.Base(name)
	if markerName != resourceID || (markerKind != resourcePermanentTombstoneDir && markerKind != resourceSoftDeleteFenceDir && markerKind != resourceTerminalCleanupDir) {
		return errors.New("invalid resource filesystem marker name")
	}
	markerDirectory := filepath.Join(resourceTombstoneDir, markerKind)
	if err := root.MkdirAll(markerDirectory, 0o700); err != nil {
		return fmt.Errorf("prepare resource tombstone directory: %w", err)
	}
	// A durable marker is not sufficient if a newly-created ancestor can be
	// lost independently. Persist both directory entries before publishing the
	// marker that authorizes destructive cleanup.
	if err := syncRootDirectory(root); err != nil {
		return fmt.Errorf("sync resource tombstone root: %w", err)
	}
	if err := syncRootChildDirectory(root, resourceTombstoneDir); err != nil {
		return fmt.Errorf("sync resource tombstone hierarchy: %w", err)
	}
	tombstoneRoot, err := root.OpenRoot(markerDirectory)
	if err != nil {
		return fmt.Errorf("open resource tombstone directory: %w", err)
	}
	defer tombstoneRoot.Close()
	file, err := tombstoneRoot.OpenFile(markerName, os.O_CREATE|os.O_EXCL|os.O_WRONLY|syscall.O_NOFOLLOW, 0o600)
	if err != nil {
		if !errors.Is(err, os.ErrExist) {
			return fmt.Errorf("create resource filesystem tombstone: %w", err)
		}
		info, statErr := tombstoneRoot.Lstat(markerName)
		if statErr != nil || !info.Mode().IsRegular() {
			return errors.New("existing resource filesystem tombstone is invalid")
		}
		return nil
	}
	created := true
	defer func() {
		_ = file.Close()
		if created {
			_ = tombstoneRoot.Remove(markerName)
		}
	}()
	if _, err := file.Write(contents); err != nil {
		return fmt.Errorf("write resource filesystem tombstone: %w", err)
	}
	if err := file.Sync(); err != nil {
		return fmt.Errorf("sync resource filesystem tombstone: %w", err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close resource filesystem tombstone: %w", err)
	}
	if err := syncRootDirectory(tombstoneRoot); err != nil {
		return fmt.Errorf("sync resource tombstone directory: %w", err)
	}
	created = false
	return nil
}

func removeResourceFilesystemTerminalCleanupMarker(root *os.Root, resourceID string) error {
	if root == nil || !safeResourceLifecycleID(resourceID) {
		return errors.New("invalid resource terminal cleanup target")
	}
	directory, err := root.OpenRoot(filepath.Join(resourceTombstoneDir, resourceTerminalCleanupDir))
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("open terminal cleanup directory: %w", err)
	}
	defer directory.Close()
	info, err := directory.Lstat(resourceID)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil || !info.Mode().IsRegular() {
		return errors.New("resource terminal cleanup marker is invalid")
	}
	if err := directory.Remove(resourceID); err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("remove resource terminal cleanup marker: %w", err)
	}
	return syncRootDirectory(directory)
}

func removeResourceFilesystemDeleteFence(root *os.Root, resourceID string) error {
	if root == nil || !safeResourceLifecycleID(resourceID) {
		return errors.New("invalid resource filesystem delete fence target")
	}
	tombstoneRoot, err := root.OpenRoot(filepath.Join(resourceTombstoneDir, resourceSoftDeleteFenceDir))
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("open resource tombstone directory: %w", err)
	}
	defer tombstoneRoot.Close()
	name := resourceID
	info, err := tombstoneRoot.Lstat(name)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("inspect resource filesystem delete fence: %w", err)
	}
	if !info.Mode().IsRegular() {
		return errors.New("resource filesystem delete fence is not a regular file")
	}
	if err := tombstoneRoot.Remove(name); err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("remove resource filesystem delete fence: %w", err)
	}
	if err := syncRootDirectory(tombstoneRoot); err != nil {
		return fmt.Errorf("sync resource tombstone directory: %w", err)
	}
	return nil
}

// softDeleteCatalogResourceWithFence makes the restore window a cross-process
// invariant: once the catalog row is deleted, no publisher can replace its
// retained source generation. Ownership is checked before publishing the
// fence, and the state is re-read under the lifecycle lock to close races with
// concurrent delete/restore requests.
func softDeleteCatalogResourceWithFence(
	ctx context.Context,
	root *os.Root,
	mutations resourceLifecycleMutationStore,
	ownerLookup resourceOwnerLookupStore,
	input domain.ResourceLifecycleMutationInput,
) (domain.ResourceRecord, domain.ResourceEventRecord, bool, error) {
	resourceID := strings.TrimSpace(input.ResourceID)
	userID := strings.TrimSpace(input.OwnerUserID)
	orgID := strings.TrimSpace(input.OwnerOrgID)
	if !safeResourceLifecycleID(resourceID) {
		return domain.ResourceRecord{}, domain.ResourceEventRecord{}, false, store.ErrNotFound
	}
	if _, err := ownerLookup.GetResourceForOwner(ctx, resourceID, userID, orgID); err != nil {
		return domain.ResourceRecord{}, domain.ResourceEventRecord{}, false, err
	}
	lockCtx, cancelLock := context.WithTimeout(ctx, resourceLifecycleMutationLockWait)
	lock, err := acquireResourceLifecycleMutationLock(lockCtx, root, resourceID)
	cancelLock()
	if err != nil {
		if errors.Is(err, errResourceLifecycleTombstoned) {
			return domain.ResourceRecord{}, domain.ResourceEventRecord{}, false, store.ErrNotFound
		}
		if errors.Is(err, context.DeadlineExceeded) && ctx.Err() == nil {
			return domain.ResourceRecord{}, domain.ResourceEventRecord{}, false, errResourceLifecycleBusy
		}
		return domain.ResourceRecord{}, domain.ResourceEventRecord{}, false, err
	}
	defer lock.release() //nolint:errcheck // request result is already authoritative

	current, err := ownerLookup.GetResourceForOwner(ctx, resourceID, userID, orgID)
	if err != nil {
		return domain.ResourceRecord{}, domain.ResourceEventRecord{}, false, err
	}
	switch strings.TrimSpace(current.Status) {
	case domain.ResourceStatusDeleted:
		if err := ensureResourceFilesystemDeleteFence(root, resourceID); err != nil {
			return domain.ResourceRecord{}, domain.ResourceEventRecord{}, false, err
		}
		return current, domain.ResourceEventRecord{}, false, nil
	case domain.ResourceStatusActive:
		// Continue below.
	default:
		return domain.ResourceRecord{}, domain.ResourceEventRecord{}, false, store.ErrNotFound
	}

	fenceExisted, err := resourceFilesystemSoftDeleteFenced(root, resourceID)
	if err != nil {
		return domain.ResourceRecord{}, domain.ResourceEventRecord{}, false, err
	}
	if err := ensureResourceFilesystemDeleteFence(root, resourceID); err != nil {
		return domain.ResourceRecord{}, domain.ResourceEventRecord{}, false, err
	}
	result, transitionErr := mutations.SoftDeleteResourceForUserWithEvent(ctx, input)
	if transitionErr == nil {
		return result.Resource, result.Event, true, nil
	}

	// A store error may be an ambiguous commit. Preserve the fence if the row
	// became deleted; remove only a fence created by this request if the row is
	// provably still active.
	resolutionCtx, cancelResolution := context.WithTimeout(
		context.WithoutCancel(ctx), resourceLifecycleCommitResolutionTimeout,
	)
	defer cancelResolution()
	current, lookupErr := ownerLookup.GetResourceForOwner(
		resolutionCtx, resourceID, userID, orgID,
	)
	if lookupErr == nil {
		switch strings.TrimSpace(current.Status) {
		case domain.ResourceStatusDeleted:
			// This request observed active before issuing the transition, so a
			// deleted row proves the ambiguous write committed. Report the
			// transition to the handler so its audit event is not lost.
			return current, domain.ResourceEventRecord{}, true, nil
		case domain.ResourceStatusActive:
			if !fenceExisted {
				if removeErr := removeResourceFilesystemDeleteFence(root, resourceID); removeErr != nil {
					return domain.ResourceRecord{}, domain.ResourceEventRecord{}, false, errors.Join(transitionErr, removeErr)
				}
			}
		}
	}
	return domain.ResourceRecord{}, domain.ResourceEventRecord{}, false, transitionErr
}

// restoreCatalogResourceWithFence performs the inverse transition while still
// holding the publication lock. The row becomes active before the reversible
// fence is removed, so a publisher can never observe an unfenced deleted row.
// An active row with a leftover fence is treated as an interrupted restore and
// reconciled idempotently.
func restoreCatalogResourceWithFence(
	ctx context.Context,
	root *os.Root,
	mutations resourceLifecycleMutationStore,
	ownerLookup resourceOwnerLookupStore,
	input domain.ResourceLifecycleMutationInput,
) (domain.ResourceRecord, domain.ResourceEventRecord, bool, error) {
	return restoreCatalogResourceWithFenceMode(
		ctx, root, mutations, ownerLookup, input, false,
	)
}

func restoreCatalogResourceWithFenceMode(
	ctx context.Context,
	root *os.Root,
	mutations resourceLifecycleMutationStore,
	ownerLookup resourceOwnerLookupStore,
	input domain.ResourceLifecycleMutationInput,
	allowActiveNoFence bool,
) (domain.ResourceRecord, domain.ResourceEventRecord, bool, error) {
	resourceID := strings.TrimSpace(input.ResourceID)
	userID := strings.TrimSpace(input.OwnerUserID)
	orgID := strings.TrimSpace(input.OwnerOrgID)
	if !safeResourceLifecycleID(resourceID) {
		return domain.ResourceRecord{}, domain.ResourceEventRecord{}, false, store.ErrNotFound
	}
	current, err := ownerLookup.GetResourceForOwner(ctx, resourceID, userID, orgID)
	if err != nil {
		return domain.ResourceRecord{}, domain.ResourceEventRecord{}, false, err
	}
	if strings.TrimSpace(current.Status) == domain.ResourceStatusActive {
		fenced, fenceErr := resourceFilesystemSoftDeleteFenced(root, resourceID)
		if fenceErr != nil {
			return domain.ResourceRecord{}, domain.ResourceEventRecord{}, false, fenceErr
		}
		if !fenced {
			if allowActiveNoFence {
				return current, domain.ResourceEventRecord{}, false, nil
			}
			return domain.ResourceRecord{}, domain.ResourceEventRecord{}, false, store.ErrNotFound
		}
	} else if strings.TrimSpace(current.Status) != domain.ResourceStatusDeleted {
		return domain.ResourceRecord{}, domain.ResourceEventRecord{}, false, store.ErrNotFound
	}

	lockCtx, cancelLock := context.WithTimeout(ctx, resourceLifecycleMutationLockWait)
	lock, err := acquireResourceLifecycleMutationLock(lockCtx, root, resourceID)
	cancelLock()
	if err != nil {
		if errors.Is(err, errResourceLifecycleTombstoned) {
			return domain.ResourceRecord{}, domain.ResourceEventRecord{}, false, store.ErrNotFound
		}
		if errors.Is(err, context.DeadlineExceeded) && ctx.Err() == nil {
			return domain.ResourceRecord{}, domain.ResourceEventRecord{}, false, errResourceLifecycleBusy
		}
		return domain.ResourceRecord{}, domain.ResourceEventRecord{}, false, err
	}
	defer lock.release() //nolint:errcheck // request result is already authoritative

	current, err = ownerLookup.GetResourceForOwner(ctx, resourceID, userID, orgID)
	if err != nil {
		return domain.ResourceRecord{}, domain.ResourceEventRecord{}, false, err
	}
	switch strings.TrimSpace(current.Status) {
	case domain.ResourceStatusActive:
		fenced, fenceErr := resourceFilesystemSoftDeleteFenced(root, resourceID)
		if fenceErr != nil {
			return domain.ResourceRecord{}, domain.ResourceEventRecord{}, false, fenceErr
		}
		if !fenced {
			if allowActiveNoFence {
				return current, domain.ResourceEventRecord{}, false, nil
			}
			return domain.ResourceRecord{}, domain.ResourceEventRecord{}, false, store.ErrNotFound
		}
		if err := removeResourceFilesystemDeleteFence(root, resourceID); err != nil {
			return domain.ResourceRecord{}, domain.ResourceEventRecord{}, false, err
		}
		return current, domain.ResourceEventRecord{}, false, nil
	case domain.ResourceStatusDeleted:
		if err := ensureResourceFilesystemDeleteFence(root, resourceID); err != nil {
			return domain.ResourceRecord{}, domain.ResourceEventRecord{}, false, err
		}
	default:
		return domain.ResourceRecord{}, domain.ResourceEventRecord{}, false, store.ErrNotFound
	}

	result, transitionErr := mutations.RestoreResourceForUserWithEvent(ctx, input)
	if transitionErr == nil {
		if err := removeResourceFilesystemDeleteFence(root, resourceID); err != nil {
			return domain.ResourceRecord{}, domain.ResourceEventRecord{}, false, err
		}
		return result.Resource, result.Event, true, nil
	}

	resolutionCtx, cancelResolution := context.WithTimeout(
		context.WithoutCancel(ctx), resourceLifecycleCommitResolutionTimeout,
	)
	defer cancelResolution()
	current, lookupErr := ownerLookup.GetResourceForOwner(
		resolutionCtx, resourceID, userID, orgID,
	)
	if lookupErr == nil && strings.TrimSpace(current.Status) == domain.ResourceStatusActive {
		if removeErr := removeResourceFilesystemDeleteFence(root, resourceID); removeErr != nil {
			return domain.ResourceRecord{}, domain.ResourceEventRecord{}, false, errors.Join(transitionErr, removeErr)
		}
		// This request observed deleted before issuing the transition, so an
		// active row proves the ambiguous write committed. Preserve the audit
		// signal even though the store acknowledgement was lost.
		return current, domain.ResourceEventRecord{}, true, nil
	}
	return domain.ResourceRecord{}, domain.ResourceEventRecord{}, false, transitionErr
}

func flockContext(ctx context.Context, file *os.File) (bool, error) {
	for {
		err := syscall.Flock(int(file.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
		if err == nil {
			return true, nil
		}
		if !errors.Is(err, syscall.EWOULDBLOCK) && !errors.Is(err, syscall.EAGAIN) {
			return false, fmt.Errorf("lock resource lifecycle: %w", err)
		}
		timer := time.NewTimer(25 * time.Millisecond)
		select {
		case <-ctx.Done():
			if !timer.Stop() {
				<-timer.C
			}
			return false, ctx.Err()
		case <-timer.C:
		}
	}
}

// removePath unlinks the visible lock name while the descriptor remains
// locked. Existing waiters acquire the old inode and detect that its name has
// disappeared before retrying; publishers then fail their source recheck.
func (lock *resourceLifecycleLock) removePath() error {
	if lock == nil || lock.file == nil || lock.directory == nil {
		return errors.New("resource lifecycle lock is not held")
	}
	openedInfo, err := lock.file.Stat()
	if err != nil {
		return err
	}
	pathInfo, err := lock.directory.Lstat(lock.name)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	if !os.SameFile(openedInfo, pathInfo) {
		return errors.New("resource lifecycle lock path was replaced")
	}
	if err := lock.directory.Remove(lock.name); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return syncRootDirectory(lock.directory)
}

func (lock *resourceLifecycleLock) release() error {
	if lock == nil {
		return nil
	}
	var errs []error
	if lock.file != nil {
		if err := syscall.Flock(int(lock.file.Fd()), syscall.LOCK_UN); err != nil {
			errs = append(errs, err)
		}
		if err := lock.file.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if lock.directory != nil {
		if err := lock.directory.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

type reclaimedFileAccounting struct {
	seen map[string]struct{}
}

func newReclaimedFileAccounting() *reclaimedFileAccounting {
	return &reclaimedFileAccounting{seen: map[string]struct{}{}}
}

func (accounting *reclaimedFileAccounting) add(path string, info os.FileInfo) int64 {
	if accounting == nil || info == nil || !info.Mode().IsRegular() {
		return 0
	}
	key := "path:" + path
	if stat, ok := info.Sys().(*syscall.Stat_t); ok {
		key = fmt.Sprintf("inode:%d:%d", stat.Dev, stat.Ino)
	}
	if _, exists := accounting.seen[key]; exists {
		return 0
	}
	accounting.seen[key] = struct{}{}
	return info.Size()
}

func derivativeNameMatcher(resourceID string) (*regexp.Regexp, error) {
	if !safeResourceLifecycleID(resourceID) {
		return nil, errors.New("unsafe upload id")
	}
	pyramidBase := regexp.QuoteMeta(resourceID + "__pyramid")
	niftiBase := regexp.QuoteMeta(resourceID + "__nifti")
	scene3dBase := regexp.QuoteMeta(resourceID + "__scene3d")
	// Match every published scene derivative generation. Old generations remain owned
	// by the resource after a converter revision bump and must be reclaimed on purge.
	scene3dRevision := `(?:\.v[1-9][0-9]*)?`
	token := `[A-Za-z0-9_]{6,64}`
	digest := `[0-9a-f]{64}`
	pattern := `^(?:` +
		pyramidBase + `\.(?:tif(?:\.transcode\.ome\.tif)?|failed|manifest\.json|sha256-` + digest + `\.tif)` +
		`|\.` + pyramidBase + `\.tmp-` + token + `\.tif(?:\.transcode\.ome\.tif)?` +
		`|\.` + pyramidBase + `\.manifest\.json\.(?:tmp|rollback)-` + token +
		`|\.` + pyramidBase + `\.sha256-` + digest + `\.tif\.(?:publish|recovery)-[0-9a-f]{24}` +
		`|\.` + pyramidBase + `\.failed\.` + token +
		`|` + niftiBase + `\.sha256-` + digest + `\.nii(?:\.tmp)?` +
		`|\.` + niftiBase + `\.tmp-` + token + `\.nii` +
		`|\.` + niftiBase + `\.stage\.nii` +
		`|` + scene3dBase + scene3dRevision + `\.sha256-` + digest + `(?:\.failed)?` +
		`|\.` + scene3dBase + scene3dRevision + `\.sha256-` + digest + `\.tmp-` + token +
		`|\.` + scene3dBase + scene3dRevision + `\.sha256-` + digest + `\.failed\.` + token + `)$`
	return regexp.Compile(pattern)
}

func ownedScene3dDerivativeDirectoryName(resourceID, name string) bool {
	if !safeResourceLifecycleID(resourceID) {
		return false
	}
	digest := `[0-9a-f]{64}`
	token := `[A-Za-z0-9_]{6,64}`
	base := regexp.QuoteMeta(resourceID+"__scene3d") + `(?:\.v[1-9][0-9]*)?\.sha256-`
	pattern := `^(?:` + base + digest + `|\.` + base + digest + `\.tmp-` + token + `)$`
	return regexp.MustCompile(pattern).MatchString(name)
}

func analysisStagingNameMatcher(resourceID string) (*regexp.Regexp, error) {
	if !safeResourceLifecycleID(resourceID) {
		return nil, errors.New("unsafe upload id")
	}
	return regexp.Compile(
		`^\.` + regexp.QuoteMeta(resourceID) + `__analysis\.tmp-[A-Za-z0-9_]{6,64}$`,
	)
}

func legacyTopLevelNiftiSidecarNames(resourceID string, sourceRelative string) []string {
	if !safeResourceLifecycleID(resourceID) || filepath.Dir(sourceRelative) != "." {
		return nil
	}
	base := filepath.Base(sourceRelative)
	if !strings.HasPrefix(base, resourceID+"__") || !strings.HasSuffix(strings.ToLower(base), ".nii.gz") {
		return nil
	}
	name := strings.TrimSuffix(base, ".gz")
	return []string{name, name + ".tmp"}
}

func cleanRootRelativePath(path string) (string, error) {
	if path == "" || filepath.IsAbs(path) {
		return "", errors.New("path must be relative to the upload root")
	}
	clean := filepath.Clean(path)
	if clean == "." || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return "", errors.New("path escapes the upload root")
	}
	return clean, nil
}

func relativePathUnderRoot(root, path string) (string, bool) {
	if !pathIsUnderRoot(root, path) {
		return "", false
	}
	rel, err := filepath.Rel(filepath.Clean(root), filepath.Clean(path))
	if err != nil {
		return "", false
	}
	rel, err = cleanRootRelativePath(rel)
	return rel, err == nil
}

// validateManagedSourcePath rejects symlinked path components before a source
// is treated as storage owned by the upload root. Missing sources are allowed
// for idempotent retry after a prior cleanup removed bytes but failed before
// the catalog tombstone committed.
func validateManagedSourcePath(root *os.Root, relative string, allowMissing bool) error {
	if relative == "" {
		return errors.New("resource has no managed source locator")
	}
	clean, err := cleanRootRelativePath(relative)
	if err != nil {
		return err
	}
	current := ""
	parts := strings.Split(clean, string(filepath.Separator))
	for index, part := range parts {
		current = filepath.Join(current, part)
		info, statErr := root.Lstat(current)
		if statErr != nil {
			if allowMissing && errors.Is(statErr, os.ErrNotExist) {
				return nil
			}
			return statErr
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("resource source component %q is a symlink", current)
		}
		if index < len(parts)-1 && !info.IsDir() {
			return fmt.Errorf("resource source component %q is not a directory", current)
		}
	}
	return nil
}

func removeRootEntry(
	root *os.Root,
	name string,
	allowDirectory bool,
	accounting *reclaimedFileAccounting,
	directorySizeHint int64,
) (int64, error) {
	clean, err := cleanRootRelativePath(name)
	if err != nil {
		return 0, err
	}
	info, err := root.Lstat(clean)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return 0, nil
		}
		return 0, err
	}
	if info.IsDir() {
		if !allowDirectory {
			return 0, errors.New("refusing recursive delete outside an exact resource bundle")
		}
		if err := root.RemoveAll(clean); err != nil {
			return 0, err
		}
		// OME-Zarr bundles can contain millions of chunks. Use the catalog's
		// bounded size accounting instead of walking the tree once to count and
		// retaining an inode key per chunk before RemoveAll walks it again.
		if directorySizeHint < 0 {
			directorySizeHint = 0
		}
		return directorySizeHint, nil
	}
	var removeErr error
	for attempt := 0; attempt < 2; attempt++ {
		removeErr = root.Remove(clean)
		if removeErr == nil {
			return accounting.add(clean, info), nil
		}
		if errors.Is(removeErr, os.ErrNotExist) {
			return 0, nil
		}
	}
	return 0, removeErr
}

func syncRootDirectory(root *os.Root) error {
	directory, err := root.Open(".")
	if err != nil {
		return err
	}
	defer directory.Close()
	return directory.Sync()
}

func syncRootChildDirectory(root *os.Root, name string) error {
	directory, err := root.Open(name)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	defer directory.Close()
	return directory.Sync()
}

func scanOwnedDerivativeNames(derived *os.Root, matcher *regexp.Regexp) ([]string, error) {
	directory, err := derived.Open(".")
	if err != nil {
		return nil, err
	}
	defer directory.Close()
	owned := []string{}
	for {
		entries, readErr := directory.ReadDir(128)
		for _, entry := range entries {
			if matcher.MatchString(entry.Name()) {
				owned = append(owned, entry.Name())
			}
		}
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			return nil, readErr
		}
	}
	sort.Strings(owned)
	return owned, nil
}

func removeOwnedAnalysisStagingFiles(
	root *os.Root,
	resourceID string,
	sourceRelative string,
	accounting *reclaimedFileAccounting,
) (int64, error) {
	if sourceRelative == "" {
		return 0, nil
	}
	matcher, err := analysisStagingNameMatcher(resourceID)
	if err != nil {
		return 0, err
	}
	parent := filepath.Dir(sourceRelative)
	if parent == "" {
		parent = "."
	}
	parentRoot, err := root.OpenRoot(parent)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return 0, nil
		}
		return 0, err
	}
	defer parentRoot.Close()
	owned, err := scanOwnedDerivativeNames(parentRoot, matcher)
	if err != nil {
		return 0, err
	}
	var freed int64
	for _, name := range owned {
		removed, removeErr := removeRootEntry(parentRoot, name, false, accounting, 0)
		freed += removed
		if removeErr != nil {
			return freed, removeErr
		}
	}
	if err := syncRootDirectory(parentRoot); err != nil {
		return freed, err
	}
	return freed, nil
}

// scanOwnedDerivativeNamesForResources inventories the flat derived directory
// once for an entire claimed batch. Callers hold every resource lifecycle lock
// in the batch while this runs, so no matching publisher can appear between
// inventory and cleanup.
func scanOwnedDerivativeNamesForResources(root *os.Root, resourceIDs []string) (map[string][]string, error) {
	owned := make(map[string][]string, len(resourceIDs))
	matchers := make(map[string]*regexp.Regexp, len(resourceIDs))
	for _, resourceID := range resourceIDs {
		matcher, err := derivativeNameMatcher(resourceID)
		if err != nil {
			return nil, err
		}
		matchers[resourceID] = matcher
		owned[resourceID] = nil
	}
	derived, err := root.OpenRoot(resourceDerivedDir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return owned, nil
		}
		return nil, err
	}
	defer derived.Close()
	directory, err := derived.Open(".")
	if err != nil {
		return nil, err
	}
	defer directory.Close()
	for {
		entries, readErr := directory.ReadDir(256)
		for _, entry := range entries {
			name := entry.Name()
			candidateNames := []string{name}
			if strings.HasPrefix(name, ".") {
				// A leading dot can belong to a canonical resource ID, or it
				// can be the publisher's temporary-file prefix. Try both
				// interpretations and let the exact filename grammar decide.
				candidateNames = append(candidateNames, strings.TrimPrefix(name, "."))
			}
		entryCandidates:
			for _, candidateName := range candidateNames {
				for _, marker := range []string{"__pyramid", "__nifti", "__scene3d"} {
					searchFrom := 0
					for searchFrom < len(candidateName) {
						index := strings.Index(candidateName[searchFrom:], marker)
						if index < 0 {
							break
						}
						end := searchFrom + index
						resourceID := candidateName[:end]
						if matcher, ok := matchers[resourceID]; ok && matcher.MatchString(name) {
							owned[resourceID] = append(owned[resourceID], name)
							break entryCandidates
						}
						searchFrom = end + len(marker)
					}
				}
			}
		}
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			return nil, readErr
		}
	}
	for resourceID := range owned {
		sort.Strings(owned[resourceID])
	}
	return owned, nil
}

func removeResourceDerivationLocks(root *os.Root, resourceID string) error {
	locks, err := root.OpenRoot(resourceLockDir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	defer locks.Close()
	for _, kind := range []string{"pyramid", "nifti", "scene3d"} {
		name, nameErr := resourceDerivationLockName(resourceID, kind)
		if nameErr != nil {
			return nameErr
		}
		if removeErr := locks.Remove(name); removeErr != nil && !errors.Is(removeErr, os.ErrNotExist) {
			return removeErr
		}
	}
	return syncRootDirectory(locks)
}

func preflightResourceNamespace(root *os.Root, sourceRelative string) error {
	directories := []string{resourceDerivedDir, resourceMetaDir}
	if strings.HasPrefix(sourceRelative, bundlesDirName+string(filepath.Separator)) {
		directories = append(directories, bundlesDirName)
	}
	for _, name := range directories {
		child, err := root.OpenRoot(name)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				continue
			}
			return fmt.Errorf("preflight resource namespace %q: %w", name, err)
		}
		if err := child.Close(); err != nil {
			return err
		}
	}
	return nil
}

// removeOwnedResourceNamespace removes only paths derived from trusted catalog
// identity and the exact producer filename grammar. It never reads deletion
// targets from a manifest. The caller must hold the resource lifecycle lock.
func removeOwnedResourceNamespace(
	root *os.Root,
	resourceID string,
	sourceRelative string,
) (int64, error) {
	return removeOwnedResourceNamespaceCore(root, resourceID, sourceRelative, nil, false, 0, false)
}

func removeOwnedResourceNamespaceFromInventory(
	root *os.Root,
	resourceID string,
	sourceRelative string,
	ownedDerivativeNames []string,
	sourceSizeHint int64,
) (int64, error) {
	return removeOwnedResourceNamespaceCore(
		root,
		resourceID,
		sourceRelative,
		ownedDerivativeNames,
		true,
		sourceSizeHint,
		false,
	)
}

func finalizeOwnedResourceNamespace(root *os.Root, resourceID string) (int64, error) {
	return removeOwnedResourceNamespaceCore(root, resourceID, "", nil, false, 0, true)
}

func finalizeOwnedResourceNamespaceFromInventory(
	root *os.Root,
	resourceID string,
	ownedDerivativeNames []string,
) (int64, error) {
	return removeOwnedResourceNamespaceCore(
		root,
		resourceID,
		"",
		ownedDerivativeNames,
		true,
		0,
		true,
	)
}

func removeOwnedResourceNamespaceCore(
	root *os.Root,
	resourceID string,
	sourceRelative string,
	ownedDerivativeNames []string,
	inventoryProvided bool,
	sourceSizeHint int64,
	finalizeDerivation bool,
) (int64, error) {
	matcher, err := derivativeNameMatcher(resourceID)
	if err != nil {
		return 0, err
	}
	bundleRoot, err := resourceBundleRelativeRoot(resourceID)
	if err != nil {
		return 0, err
	}
	if err := preflightResourceNamespace(root, sourceRelative); err != nil {
		return 0, err
	}
	accounting := newReclaimedFileAccounting()
	var freed int64
	var errs []error
	var sourceRemoveName string
	var sourceIsBundle bool

	if sourceRelative != "" {
		sourceRelative, err = cleanRootRelativePath(sourceRelative)
		if err != nil {
			return 0, err
		}
		sourceRemoveName = sourceRelative
		if sourceRelative == bundleRoot || strings.HasPrefix(sourceRelative, bundleRoot+string(filepath.Separator)) {
			sourceRemoveName = bundleRoot
			sourceIsBundle = true
		}
	}

	derived, openErr := root.OpenRoot(resourceDerivedDir)
	if openErr != nil {
		if !errors.Is(openErr, os.ErrNotExist) {
			errs = append(errs, fmt.Errorf("open derivative directory: %w", openErr))
		}
	} else {
		for _, name := range legacyTopLevelNiftiSidecarNames(resourceID, sourceRelative) {
			removed, removeErr := removeRootEntry(derived, name, false, accounting, 0)
			freed += removed
			if removeErr != nil {
				errs = append(errs, fmt.Errorf("remove legacy NIfTI sidecar %q: %w", name, removeErr))
			}
		}
		manifestName := resourceID + "__pyramid.manifest.json"
		removed, removeErr := removeRootEntry(derived, manifestName, false, accounting, 0)
		freed += removed
		if removeErr != nil {
			errs = append(errs, fmt.Errorf("remove derivative manifest: %w", removeErr))
		}
		if syncErr := syncRootDirectory(derived); syncErr != nil {
			errs = append(errs, fmt.Errorf("sync derivative manifest removal: %w", syncErr))
		}
		owned := ownedDerivativeNames
		if !inventoryProvided {
			var scanErr error
			owned, scanErr = scanOwnedDerivativeNames(derived, matcher)
			if scanErr != nil {
				errs = append(errs, fmt.Errorf("scan derivative namespace: %w", scanErr))
			}
		}
		if len(errs) == 0 {
			for _, name := range owned {
				if name == manifestName {
					continue
				}
				if !matcher.MatchString(name) {
					errs = append(errs, fmt.Errorf("derivative inventory contains unowned name %q", name))
					continue
				}
				removed, removeErr := removeRootEntry(
					derived,
					name,
					ownedScene3dDerivativeDirectoryName(resourceID, name),
					accounting,
					0,
				)
				freed += removed
				if removeErr != nil {
					errs = append(errs, fmt.Errorf("remove derivative %q: %w", name, removeErr))
				}
			}
		}
		if syncErr := syncRootDirectory(derived); syncErr != nil {
			errs = append(errs, fmt.Errorf("sync derivative cleanup: %w", syncErr))
		}
		if closeErr := derived.Close(); closeErr != nil {
			errs = append(errs, closeErr)
		}
	}
	if len(errs) == 0 && sourceRelative != "" {
		removed, removeErr := removeOwnedAnalysisStagingFiles(
			root,
			resourceID,
			sourceRelative,
			accounting,
		)
		freed += removed
		if removeErr != nil {
			errs = append(errs, fmt.Errorf("remove analysis publication staging files: %w", removeErr))
		}
	}
	// Remove the authoritative source only after every derivative cleanup step
	// has succeeded. A no-catalog deletion can then always be retried from the
	// still-discoverable source when an unexpected derivative blocks cleanup.
	if len(errs) == 0 && sourceRemoveName != "" {
		removed, removeErr := removeRootEntry(root, sourceRemoveName, sourceIsBundle, accounting, sourceSizeHint)
		freed += removed
		if removeErr != nil {
			errs = append(errs, fmt.Errorf("remove source: %w", removeErr))
		}
		if syncErr := syncRootDirectory(root); syncErr != nil {
			errs = append(errs, fmt.Errorf("sync upload root after source removal: %w", syncErr))
		}
		if sourceIsBundle {
			if syncErr := syncRootChildDirectory(root, bundlesDirName); syncErr != nil {
				errs = append(errs, fmt.Errorf("sync bundle directory after source removal: %w", syncErr))
			}
		}
	}

	// Metadata and the deletion intent are no-catalog ownership records. The
	// intent is removed last so a partial metadata cleanup remains retryable.
	if len(errs) == 0 {
		removed, removeErr := removeRootEntry(
			root,
			filepath.Join(resourceMetaDir, resourceID+".json"),
			false,
			accounting,
			0,
		)
		freed += removed
		if removeErr != nil {
			errs = append(errs, fmt.Errorf("remove upload metadata: %w", removeErr))
		}
	}
	if len(errs) == 0 {
		removed, removeErr := removeRootEntry(
			root,
			filepath.Join(resourceCatalogReconciliationDir, resourceCatalogReconciliationIntentName(resourceID)),
			false,
			accounting,
			0,
		)
		freed += removed
		if removeErr != nil {
			errs = append(errs, fmt.Errorf("remove catalog reconciliation intent: %w", removeErr))
		}
	}
	if len(errs) == 0 {
		removed, removeErr := removeRootEntry(
			root,
			filepath.Join(resourceMetaDir, resourceDeletionIntentName(resourceID)),
			false,
			accounting,
			0,
		)
		freed += removed
		if removeErr != nil {
			errs = append(errs, fmt.Errorf("remove upload deletion intent: %w", removeErr))
		}
	}
	if len(errs) == 0 {
		if syncErr := syncRootChildDirectory(root, resourceMetaDir); syncErr != nil {
			errs = append(errs, fmt.Errorf("sync upload metadata removal: %w", syncErr))
		}
	}
	if len(errs) == 0 && finalizeDerivation {
		stagingRoot, stagingErr := resourceStagingRelativeRoot(resourceID)
		if stagingErr != nil {
			errs = append(errs, stagingErr)
		} else {
			removed, removeErr := removeRootEntry(root, stagingRoot, true, accounting, 0)
			freed += removed
			if removeErr != nil {
				errs = append(errs, fmt.Errorf("remove resource publication staging: %w", removeErr))
			}
			if syncErr := syncRootChildDirectory(root, resourceStagingDir); syncErr != nil {
				errs = append(errs, fmt.Errorf("sync publication staging removal: %w", syncErr))
			}
		}
	}
	if len(errs) == 0 && finalizeDerivation {
		if err := removeResourceDerivationLocks(root, resourceID); err != nil {
			errs = append(errs, fmt.Errorf("remove resource derivation locks: %w", err))
		}
	}
	return freed, errors.Join(errs...)
}
