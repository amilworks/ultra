package httpapi

import (
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"syscall"
)

type niftiSidecarStage struct {
	name             string
	destinationName  string
	sourceGeneration derivativeSourceStat
	outputGeneration derivativeSourceStat
}

type niftiSidecarCopyFunc func(io.Writer, io.Reader) (int64, error)

func sameFileGenerationAcrossOwnedRename(
	before derivativeSourceStat,
	after derivativeSourceStat,
) bool {
	// A same-directory rename preserves the file and its content but may update
	// ctime/link metadata. Device, inode, size, and mtime remain the stable
	// generation identity; the retained descriptor supplies the refreshed ctime.
	return before.Device == after.Device &&
		before.Inode == after.Inode &&
		before.SizeBytes == after.SizeBytes &&
		before.MtimeNS == after.MtimeNS
}

func niftiSidecarStageName(resourceID string) (string, error) {
	if !safeResourceLifecycleID(resourceID) {
		return "", errors.New("invalid NIfTI staging target")
	}
	// The per-resource NIfTI work lock makes one fixed private stage sufficient.
	// A fixed name makes abandoned cleanup O(1) instead of scanning every resource.
	return "." + resourceID + "__nifti.stage.nii", nil
}

func cleanupAbandonedNiftiStages(directory *os.Root, resourceID string) error {
	if directory == nil || !safeResourceLifecycleID(resourceID) {
		return errors.New("invalid NIfTI staging cleanup target")
	}
	stageName, err := niftiSidecarStageName(resourceID)
	if err != nil {
		return err
	}
	if err := directory.Remove(stageName); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	return syncRootDirectory(directory)
}

func buildDecompressedNiftiStage(
	ctx context.Context,
	uploadRoot *os.Root,
	resourceID string,
	sourceRelative string,
	sourceSHA256 string,
	destinationName string,
	copyFn niftiSidecarCopyFunc,
) (_ niftiSidecarStage, err error) {
	if uploadRoot == nil || !safeResourceLifecycleID(resourceID) || !isSHA256Hex(sourceSHA256) {
		return niftiSidecarStage{}, errors.New("invalid NIfTI sidecar identity")
	}
	expectedDestination := resourceID + "__nifti.sha256-" + sourceSHA256 + ".nii"
	if destinationName != expectedDestination || filepath.Base(destinationName) != destinationName {
		return niftiSidecarStage{}, errors.New("invalid NIfTI sidecar destination")
	}
	if copyFn == nil {
		copyFn = io.Copy
	}
	if err := validateManagedSourcePath(uploadRoot, sourceRelative, false); err != nil {
		return niftiSidecarStage{}, err
	}
	beforeInfo, err := uploadRoot.Lstat(sourceRelative)
	if err != nil {
		return niftiSidecarStage{}, err
	}
	beforeGeneration, ok := fileGeneration(beforeInfo)
	if !ok {
		return niftiSidecarStage{}, errors.New("NIfTI source generation is unavailable")
	}
	in, err := uploadRoot.Open(sourceRelative)
	if err != nil {
		return niftiSidecarStage{}, err
	}
	defer func() { _ = in.Close() }()
	openedInfo, err := in.Stat()
	if err != nil {
		return niftiSidecarStage{}, err
	}
	openedGeneration, ok := fileGeneration(openedInfo)
	if !ok || openedGeneration != beforeGeneration {
		return niftiSidecarStage{}, errors.New("NIfTI source generation changed while opening")
	}
	if err := uploadRoot.MkdirAll(resourceDerivedDir, 0o755); err != nil {
		return niftiSidecarStage{}, err
	}
	directory, err := uploadRoot.OpenRoot(resourceDerivedDir)
	if err != nil {
		return niftiSidecarStage{}, err
	}
	defer directory.Close()
	if err := cleanupAbandonedNiftiStages(directory, resourceID); err != nil {
		return niftiSidecarStage{}, err
	}
	stageName, err := niftiSidecarStageName(resourceID)
	if err != nil {
		return niftiSidecarStage{}, err
	}
	out, err := directory.OpenFile(
		stageName,
		os.O_CREATE|os.O_EXCL|os.O_WRONLY|syscall.O_NOFOLLOW,
		0o600,
	)
	if err != nil {
		return niftiSidecarStage{}, err
	}
	committed := false
	defer func() {
		_ = out.Close()
		if !committed {
			_ = directory.Remove(stageName)
		}
	}()
	stop := make(chan struct{})
	defer close(stop)
	go func() {
		select {
		case <-ctx.Done():
			_ = in.Close()
		case <-stop:
		}
	}()
	hasher := sha256.New()
	hashingSource := io.TeeReader(in, hasher)
	zr, err := gzip.NewReader(hashingSource)
	if err != nil {
		return niftiSidecarStage{}, err
	}
	if _, err := copyFn(out, zr); err != nil {
		_ = zr.Close()
		return niftiSidecarStage{}, err
	}
	if err := zr.Close(); err != nil {
		return niftiSidecarStage{}, err
	}
	if _, err := io.Copy(io.Discard, hashingSource); err != nil {
		return niftiSidecarStage{}, err
	}
	if hex.EncodeToString(hasher.Sum(nil)) != sourceSHA256 {
		return niftiSidecarStage{}, errors.New("NIfTI source digest does not match catalog")
	}
	if err := out.Sync(); err != nil {
		return niftiSidecarStage{}, err
	}
	writtenInfo, err := out.Stat()
	if err != nil {
		return niftiSidecarStage{}, err
	}
	outputGeneration, ok := fileGeneration(writtenInfo)
	if !ok {
		return niftiSidecarStage{}, errors.New("NIfTI stage generation is unavailable")
	}
	if err := out.Close(); err != nil {
		return niftiSidecarStage{}, err
	}
	stageInfo, err := directory.Lstat(stageName)
	if err != nil {
		return niftiSidecarStage{}, err
	}
	pathGeneration, ok := fileGeneration(stageInfo)
	if !ok || pathGeneration != outputGeneration {
		return niftiSidecarStage{}, errors.New("NIfTI stage changed after writing")
	}
	afterInfo, err := uploadRoot.Lstat(sourceRelative)
	if err != nil {
		return niftiSidecarStage{}, err
	}
	afterGeneration, ok := fileGeneration(afterInfo)
	if !ok || afterGeneration != beforeGeneration {
		return niftiSidecarStage{}, errors.New("NIfTI source generation changed during decompression")
	}
	committed = true
	return niftiSidecarStage{
		name:             stageName,
		destinationName:  destinationName,
		sourceGeneration: beforeGeneration,
		outputGeneration: outputGeneration,
	}, nil
}

func buildAndPublishDecompressedNiftiSidecar(
	ctx context.Context,
	root string,
	resourceID string,
	sourceSHA256 string,
	srcPath string,
	dst string,
) error {
	return buildAndPublishDecompressedNiftiSidecarWithCopy(
		ctx, root, resourceID, sourceSHA256, srcPath, dst, io.Copy,
	)
}

func buildAndPublishDecompressedNiftiSidecarWithCopy(
	ctx context.Context,
	root string,
	resourceID string,
	sourceSHA256 string,
	srcPath string,
	dst string,
	copyFn niftiSidecarCopyFunc,
) error {
	uploadRoot, err := os.OpenRoot(filepath.Clean(root))
	if err != nil {
		return err
	}
	defer uploadRoot.Close()
	sourceRelative, ok := relativePathUnderRoot(root, srcPath)
	if !ok {
		return errors.New("NIfTI source is outside the managed upload root")
	}
	if filepath.Dir(dst) != filepath.Join(filepath.Clean(root), resourceDerivedDir) {
		return errors.New("NIfTI destination is outside the managed derived namespace")
	}
	workLock, err := acquireResourceDerivationLock(
		ctx, uploadRoot, resourceID, "nifti", sourceRelative,
	)
	if err != nil {
		return err
	}
	defer workLock.release()
	if readyDecompressedNiftiSidecar(niftiDecompressedSidecarIdentity{
		root: root, resourceID: resourceID, sourceSHA256: sourceSHA256,
	}) != "" {
		return nil
	}
	stage, err := buildDecompressedNiftiStage(
		ctx,
		uploadRoot,
		resourceID,
		sourceRelative,
		sourceSHA256,
		filepath.Base(dst),
		copyFn,
	)
	if err != nil {
		return err
	}
	return publishDecompressedNiftiStage(
		ctx, uploadRoot, resourceID, sourceRelative, stage,
	)
}

func publishDecompressedNiftiStage(
	ctx context.Context,
	uploadRoot *os.Root,
	resourceID string,
	sourceRelative string,
	stage niftiSidecarStage,
) error {
	if uploadRoot == nil || !safeResourceLifecycleID(resourceID) {
		return errors.New("invalid NIfTI publication target")
	}
	expectedStage, err := niftiSidecarStageName(resourceID)
	if err != nil || stage.name != expectedStage || filepath.Base(stage.destinationName) != stage.destinationName {
		return errors.New("invalid NIfTI publication stage")
	}
	directory, err := uploadRoot.OpenRoot(resourceDerivedDir)
	if err != nil {
		return err
	}
	defer directory.Close()
	stageRetained := true
	defer func() {
		if stageRetained {
			_ = directory.Remove(stage.name)
		}
	}()
	lifecycleLock, err := acquireResourceLifecycleLock(
		ctx, uploadRoot, resourceID, sourceRelative,
	)
	if err != nil {
		return err
	}
	defer lifecycleLock.release()
	currentInfo, err := uploadRoot.Lstat(sourceRelative)
	if err != nil {
		return err
	}
	currentGeneration, ok := fileGeneration(currentInfo)
	if !ok || currentGeneration != stage.sourceGeneration {
		return errors.New("NIfTI source generation changed before publication")
	}
	if info, statErr := directory.Lstat(stage.destinationName); statErr == nil {
		if !info.Mode().IsRegular() {
			return errors.New("NIfTI sidecar destination is not a regular file")
		}
	} else if !errors.Is(statErr, os.ErrNotExist) {
		return statErr
	}
	stageFile, err := directory.OpenFile(stage.name, os.O_RDONLY|syscall.O_NOFOLLOW, 0)
	if err != nil {
		return fmt.Errorf("open NIfTI stage for publication: %w", err)
	}
	defer stageFile.Close()
	openedStage, err := stageFile.Stat()
	if err != nil {
		return fmt.Errorf("inspect opened NIfTI stage: %w", err)
	}
	openedGeneration, ok := fileGeneration(openedStage)
	if !ok || openedGeneration != stage.outputGeneration {
		return errors.New("NIfTI stage changed before publication")
	}
	pathStage, err := directory.Lstat(stage.name)
	if err != nil {
		return fmt.Errorf("inspect NIfTI stage path: %w", err)
	}
	pathGeneration, ok := fileGeneration(pathStage)
	if !ok || pathGeneration != stage.outputGeneration {
		return errors.New("NIfTI stage path changed before publication")
	}
	if err := directory.Rename(stage.name, stage.destinationName); err != nil {
		return fmt.Errorf("publish NIfTI sidecar: %w", err)
	}
	stageRetained = false
	publishedInfo, err := directory.Lstat(stage.destinationName)
	if err != nil {
		return fmt.Errorf("inspect published NIfTI sidecar: %w", err)
	}
	publishedGeneration, ok := fileGeneration(publishedInfo)
	openedPublishedInfo, openedErr := stageFile.Stat()
	if openedErr != nil {
		_ = directory.Remove(stage.destinationName)
		_ = syncRootDirectory(directory)
		return fmt.Errorf("inspect retained NIfTI stage descriptor: %w", openedErr)
	}
	openedPublishedGeneration, openedOK := fileGeneration(openedPublishedInfo)
	if !ok || !openedOK ||
		publishedGeneration != openedPublishedGeneration ||
		!sameFileGenerationAcrossOwnedRename(stage.outputGeneration, openedPublishedGeneration) {
		_ = directory.Remove(stage.destinationName)
		_ = syncRootDirectory(directory)
		return errors.New("published NIfTI sidecar does not match staged generation")
	}
	return syncRootDirectory(directory)
}
