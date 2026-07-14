package httpapi

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

// Bundle uploads: directory-shaped special formats (OME-Zarr) arrive as a SET of upload
// files, each carrying a `relative_path` (from the browser folder picker / webkitdirectory)
// whose top segment is the bundle root (e.g. "scan.ome.zarr"). Such files are written into
// a single bundle tree under {root}/bundles/{bundleID}/... and committed as ONE catalog
// resource by the finalize-bundle endpoint — instead of one resource per member file.
//
// The bundle map (top segment -> bundleInfo) is computed at session-create from the file
// list and stored on the session metadata, so the per-file complete handler and the
// finalize step agree on the bundle id + on-disk layout.

const bundlesDirName = "bundles"

const (
	resourceTreeIdentitySchema = "ultra.resource-tree-identity.v1"
	treeManifestSchema         = "ultra.tree-manifest.v1"
	canonicalJSONSchema        = "ultra.canonical-json.v1"
	treeManifestPath           = ".ultra/tree-manifest.json"
	treeIdentityScope          = "all_regular_files_except_tree_manifest"
)

// bundleTreeManifest uses lexicographically ordered JSON field declarations so
// encoding/json emits exactly the same compact bytes as ultra.canonical-json.v1
// for this closed, ASCII-only schema. Bundle upload paths are already restricted
// to [A-Za-z0-9._/-], and the entries are sorted below.
type bundleTreeManifest struct {
	Entries []bundleTreeManifestEntry `json:"entries"`
	Schema  string                    `json:"schema"`
}

type bundleTreeManifestEntry struct {
	Path      string `json:"path"`
	SHA256    string `json:"sha256"`
	SizeBytes int64  `json:"size_bytes"`
}

type bundleTreeIdentity struct {
	ManifestSHA256 string
	EntryCount     int
	SizeBytes      int64
}

type bundleTreeActualFile struct {
	SHA256    string
	SizeBytes int64
}

// bundleInfo identifies a directory-format bundle within an upload session.
type bundleInfo struct {
	ID       string // resource id for the finished bundle (file_…)
	Name     string // sanitized bundle root dir name (e.g. scan.ome.zarr)
	FormatID string // special-format id (e.g. ome-zarr)
}

// firstPathSegment returns the first forward-slash segment of a relative path.
func firstPathSegment(relPath string) string {
	p := strings.ReplaceAll(strings.TrimSpace(relPath), "\\", "/")
	p = strings.TrimLeft(p, "/")
	if i := strings.IndexByte(p, '/'); i >= 0 {
		return p[:i]
	}
	return p
}

// sanitizeBundleSegment validates ONE path segment for a bundle. Unlike safeOriginalFilename
// it PRESERVES leading dots (zarr's .zattrs/.zarray/.zgroup) and rejects — rather than
// silently rewrites — any segment with unexpected characters, because a zarr store is
// content-addressed and a rewritten chunk key would corrupt it. Allows the zarr key
// alphabet [A-Za-z0-9._-]; "/" is the separator (handled by the caller).
func sanitizeBundleSegment(seg string) (string, bool) {
	seg = strings.TrimSpace(seg)
	if seg == "" || seg == "." || seg == ".." {
		return "", false
	}
	for _, c := range seg {
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9', c == '.', c == '_', c == '-':
		default:
			return "", false
		}
	}
	return seg, true
}

// sanitizeBundleRelPath cleans a client relative path into a safe forward-slash relative
// path (each segment validated). Returns ("", false) on any traversal/empty/illegal segment.
func sanitizeBundleRelPath(relPath string) (string, bool) {
	p := strings.ReplaceAll(strings.TrimSpace(relPath), "\\", "/")
	// webkitRelativePath is always relative; an absolute path is anomalous — reject it.
	if p == "" || strings.HasPrefix(p, "/") {
		return "", false
	}
	segs := strings.Split(p, "/")
	out := make([]string, 0, len(segs))
	for _, s := range segs {
		safe, ok := sanitizeBundleSegment(s)
		if !ok {
			return "", false
		}
		out = append(out, safe)
	}
	return strings.Join(out, "/"), true
}

// detectSessionBundles groups upload-session files by relative-path top segment and assigns
// a bundle to each group whose top segment is a directory-shaped special format
// (e.g. *.ome.zarr). Returns rawTopSegment -> bundleInfo. newID mints the bundle resource id
// (injectable for tests).
func detectSessionBundles(files []domain.UpsertUploadSessionFileInput, newID func() string) map[string]bundleInfo {
	out := map[string]bundleInfo{}
	for _, f := range files {
		top := firstPathSegment(f.RelativePath)
		if top == "" {
			continue
		}
		if _, seen := out[top]; seen {
			continue
		}
		safeTop, ok := sanitizeBundleSegment(top)
		if !ok {
			continue
		}
		if sf := detectSpecialFormatByName(top); sf != nil && sf.Shape == "directory" {
			out[top] = bundleInfo{ID: newID(), Name: safeTop, FormatID: sf.ID}
		}
	}
	return out
}

// bundleMetadataValue serializes the bundle map for storage on the session metadata.
func bundleMetadataValue(bundles map[string]bundleInfo) map[string]any {
	m := map[string]any{}
	for top, b := range bundles {
		m[top] = map[string]any{"id": b.ID, "name": b.Name, "format": b.FormatID}
	}
	return m
}

// sessionBundles reads the bundle map stored on a session at create time.
func sessionBundles(session domain.UploadSessionRecord) map[string]bundleInfo {
	out := map[string]bundleInfo{}
	raw, ok := session.Metadata["bundles"]
	if !ok {
		return out
	}
	m, ok := raw.(map[string]any)
	if !ok {
		return out
	}
	for top, v := range m {
		vm, ok := v.(map[string]any)
		if !ok {
			continue
		}
		out[top] = bundleInfo{
			ID:       asMetaString(vm["id"]),
			Name:     asMetaString(vm["name"]),
			FormatID: asMetaString(vm["format"]),
		}
	}
	return out
}

func asMetaString(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

// bundleDirPath is the on-disk root of a bundle: {root}/bundles/{id}/{name}.
func bundleDirPath(root string, b bundleInfo) string {
	return filepath.Join(root, bundlesDirName, b.ID, b.Name)
}

// bundleMemberTarget resolves the on-disk destination for a session file that belongs to a
// bundle. Returns (dest, bundle, true) when the file's relative-path top segment maps to a
// session bundle and the sanitized path stays under the bundle root.
func bundleMemberTarget(root string, session domain.UploadSessionRecord, file domain.UploadSessionFileRecord) (string, bundleInfo, bool) {
	bundles := sessionBundles(session)
	if len(bundles) == 0 {
		return "", bundleInfo{}, false
	}
	b, ok := bundles[firstPathSegment(file.RelativePath)]
	if !ok {
		return "", bundleInfo{}, false
	}
	safeRel, ok := sanitizeBundleRelPath(file.RelativePath)
	if !ok {
		return "", bundleInfo{}, false
	}
	bundleRoot := filepath.Join(root, bundlesDirName, b.ID)
	dest := filepath.Join(bundleRoot, safeRel)
	if !pathIsUnderRoot(bundleRoot, dest) {
		return "", bundleInfo{}, false
	}
	return dest, b, true
}

// bundleFinalizeReHashMaxTotalBytes is the total bundle size at or below which
// finalize re-reads and re-hashes every member (integrity re-verification).
// Above it, finalize trusts the commit-time sha256 and checks only presence +
// size, so a large OME-Zarr does not block the request re-reading every chunk.
// A package var so tests can lower it; 2 GiB by default.
var bundleFinalizeReHashMaxTotalBytes int64 = 2 << 30

// finalizedBundleTreeIdentity builds and installs the catalog-authoritative
// identity for a completed directory-format upload. It rereads and hashes every
// final member under the bundle lock, compares those bytes with the SHA-256
// authored while assembling verified chunks, proves directory closure, and
// only then binds normalized relative paths, hashes, and sizes into a canonical
// manifest that is atomically installed at treeManifestPath.
//
// treeManifestPath is server-owned final state. A client may include that path
// in an upload (for round-trip compatibility), but its completed upload row is
// only an ingress record: mismatched bytes are replaced and exact bytes are
// adopted without rewriting. The manifest is excluded from its own entries to
// avoid a self-hash cycle. SizeBytes is the physical retained size, including
// the final server manifest; EntryCount and manifest entries exclude it.
func finalizedBundleTreeIdentity(
	dir string,
	bundleTop string,
	bundle bundleInfo,
	files []domain.UploadSessionFileRecord,
) (bundleTreeIdentity, error) {
	rootInfo, err := os.Lstat(dir)
	if err != nil {
		return bundleTreeIdentity{}, err
	}
	if rootInfo.Mode()&os.ModeSymlink != 0 || !rootInfo.IsDir() {
		return bundleTreeIdentity{}, errors.New("bundle root must be a real directory")
	}

	expected := make(map[string]domain.UploadSessionFileRecord)
	declared := make(map[string]struct{})
	for _, file := range files {
		if firstPathSegment(file.RelativePath) != bundleTop {
			continue
		}
		relative, ok := relativeBundleMemberPath(bundleTop, bundle, file.RelativePath)
		if !ok {
			return bundleTreeIdentity{}, fmt.Errorf("bundle member has an invalid relative path: %q", file.RelativePath)
		}
		if _, duplicate := declared[relative]; duplicate {
			return bundleTreeIdentity{}, fmt.Errorf("bundle contains duplicate member path %q", relative)
		}
		declared[relative] = struct{}{}
		if file.Status != "completed" || file.ResourceID != bundle.ID {
			return bundleTreeIdentity{}, fmt.Errorf("bundle member %q is not committed", relative)
		}
		computedSHA := strings.ToLower(strings.TrimSpace(file.ComputedSHA256))
		if !isSHA256Hex(computedSHA) {
			return bundleTreeIdentity{}, fmt.Errorf("bundle member %q has no verified sha256", relative)
		}
		if file.SizeBytes < 0 {
			return bundleTreeIdentity{}, fmt.Errorf("bundle member %q has a negative size", relative)
		}
		file.ComputedSHA256 = computedSHA
		if relative == treeManifestPath {
			continue
		}
		expected[relative] = file
	}
	if len(expected) == 0 {
		return bundleTreeIdentity{}, errors.New("bundle has no declared members")
	}

	// Re-hashing every file at finalize re-verifies on-disk content but costs
	// O(all bytes). For a large bundle (a multi-GB OME-Zarr with thousands of
	// chunks) that blocks the finalize request on NFS and risks an edge-proxy
	// timeout. Above a bounded total we trust the sha256 each member had verified
	// at commit time and check only presence + size, keeping finalize a stat-walk.
	// Normal (small) bundles still get the full content re-hash.
	var expectedTotal int64
	for _, file := range expected {
		if file.SizeBytes <= 0 {
			continue
		}
		if file.SizeBytes > math.MaxInt64-expectedTotal {
			expectedTotal = math.MaxInt64
			break
		}
		expectedTotal += file.SizeBytes
	}
	verifyContent := expectedTotal <= bundleFinalizeReHashMaxTotalBytes

	actual := make(map[string]bundleTreeActualFile, len(expected))
	err = filepath.WalkDir(dir, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if path == dir {
			return nil
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("bundle contains symbolic link %q", entry.Name())
		}
		relative, relErr := filepath.Rel(dir, path)
		if relErr != nil {
			return relErr
		}
		relative = filepath.ToSlash(relative)
		if relative == treeManifestPath {
			info, infoErr := entry.Info()
			if infoErr != nil {
				return infoErr
			}
			if !info.Mode().IsRegular() {
				return errors.New("reserved tree manifest path must be a regular file")
			}
			return nil
		}
		if entry.IsDir() {
			return nil
		}
		actualFile, hashErr := finalizedBundleFileIdentity(path, verifyContent)
		if hashErr != nil {
			return hashErr
		}
		if _, duplicate := actual[relative]; duplicate {
			return fmt.Errorf("bundle contains duplicate filesystem path %q", relative)
		}
		actual[relative] = actualFile
		return nil
	})
	if err != nil {
		return bundleTreeIdentity{}, err
	}
	if len(actual) != len(expected) {
		return bundleTreeIdentity{}, fmt.Errorf(
			"bundle tree closure mismatch: declared=%d actual=%d", len(expected), len(actual),
		)
	}

	entries := make([]bundleTreeManifestEntry, 0, len(expected))
	var totalSize int64
	for relative, file := range expected {
		actualFile, exists := actual[relative]
		if !exists {
			return bundleTreeIdentity{}, fmt.Errorf("bundle member %q is missing", relative)
		}
		if actualFile.SizeBytes != file.SizeBytes {
			return bundleTreeIdentity{}, fmt.Errorf(
				"bundle member %q size changed: declared=%d actual=%d",
				relative, file.SizeBytes, actualFile.SizeBytes,
			)
		}
		entrySHA := file.ComputedSHA256
		if verifyContent {
			if !strings.EqualFold(actualFile.SHA256, file.ComputedSHA256) {
				return bundleTreeIdentity{}, fmt.Errorf(
					"bundle member %q content changed after verified upload", relative,
				)
			}
			entrySHA = actualFile.SHA256
		}
		if actualFile.SizeBytes > math.MaxInt64-totalSize {
			return bundleTreeIdentity{}, errors.New("bundle byte size overflows int64")
		}
		totalSize += actualFile.SizeBytes
		entries = append(entries, bundleTreeManifestEntry{
			Path: relative, SHA256: entrySHA, SizeBytes: actualFile.SizeBytes,
		})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Path < entries[j].Path })
	canonical, err := json.Marshal(bundleTreeManifest{Entries: entries, Schema: treeManifestSchema})
	if err != nil {
		return bundleTreeIdentity{}, err
	}
	digest := sha256.Sum256(canonical)
	digestText := hex.EncodeToString(digest[:])
	if int64(len(canonical)) > math.MaxInt64-totalSize {
		return bundleTreeIdentity{}, errors.New("bundle byte size overflows int64")
	}
	manifestFile, err := authorCanonicalBundleTreeManifest(dir, canonical, digestText)
	if err != nil {
		return bundleTreeIdentity{}, err
	}
	totalSize += manifestFile.SizeBytes
	return bundleTreeIdentity{
		ManifestSHA256: digestText,
		EntryCount:     len(entries),
		SizeBytes:      totalSize,
	}, nil
}

// authorCanonicalBundleTreeManifest installs canonical as one atomic rename on
// the bundle filesystem. The temporary file lives beside (not inside) the
// verified tree, so neither readers nor a retry can observe it as an undeclared
// tree member. An already exact regular manifest is a non-mutating retry.
func authorCanonicalBundleTreeManifest(
	dir string,
	canonical []byte,
	digest string,
) (bundleTreeActualFile, error) {
	target := filepath.Join(dir, filepath.FromSlash(treeManifestPath))
	if info, err := os.Lstat(target); err == nil {
		if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
			return bundleTreeActualFile{}, errors.New("reserved tree manifest path must be a regular file")
		}
		identity, identityErr := finalizedBundleFileIdentity(target, true)
		if identityErr != nil {
			return bundleTreeActualFile{}, identityErr
		}
		if identity.SHA256 == digest && identity.SizeBytes == int64(len(canonical)) {
			return identity, nil
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return bundleTreeActualFile{}, err
	}

	manifestDir := filepath.Dir(target)
	if err := os.Mkdir(manifestDir, 0o755); err != nil && !errors.Is(err, os.ErrExist) {
		return bundleTreeActualFile{}, err
	}
	manifestDirInfo, err := os.Lstat(manifestDir)
	if err != nil {
		return bundleTreeActualFile{}, err
	}
	if manifestDirInfo.Mode()&os.ModeSymlink != 0 || !manifestDirInfo.IsDir() {
		return bundleTreeActualFile{}, errors.New("tree manifest parent must be a real directory")
	}

	temporary, err := os.CreateTemp(filepath.Dir(dir), ".tree-manifest-*")
	if err != nil {
		return bundleTreeActualFile{}, err
	}
	temporaryPath := temporary.Name()
	defer func() { _ = os.Remove(temporaryPath) }()
	if err := temporary.Chmod(0o644); err != nil {
		_ = temporary.Close()
		return bundleTreeActualFile{}, err
	}
	if _, err := temporary.Write(canonical); err != nil {
		_ = temporary.Close()
		return bundleTreeActualFile{}, err
	}
	if err := temporary.Sync(); err != nil {
		_ = temporary.Close()
		return bundleTreeActualFile{}, err
	}
	if err := temporary.Close(); err != nil {
		return bundleTreeActualFile{}, err
	}

	currentManifestDir, err := os.Lstat(manifestDir)
	if err != nil {
		return bundleTreeActualFile{}, err
	}
	if currentManifestDir.Mode()&os.ModeSymlink != 0 || !currentManifestDir.IsDir() ||
		!os.SameFile(manifestDirInfo, currentManifestDir) {
		return bundleTreeActualFile{}, errors.New("tree manifest parent changed while finalizing")
	}
	if current, statErr := os.Lstat(target); statErr == nil {
		if current.Mode()&os.ModeSymlink != 0 || !current.Mode().IsRegular() {
			return bundleTreeActualFile{}, errors.New("reserved tree manifest path must be a regular file")
		}
	} else if !errors.Is(statErr, os.ErrNotExist) {
		return bundleTreeActualFile{}, statErr
	}
	if err := os.Rename(temporaryPath, target); err != nil {
		return bundleTreeActualFile{}, err
	}
	manifestDirectory, err := os.Open(manifestDir)
	if err != nil {
		return bundleTreeActualFile{}, err
	}
	syncErr := manifestDirectory.Sync()
	closeErr := manifestDirectory.Close()
	if syncErr != nil {
		return bundleTreeActualFile{}, syncErr
	}
	if closeErr != nil {
		return bundleTreeActualFile{}, closeErr
	}
	identity, err := finalizedBundleFileIdentity(target, true)
	if err != nil {
		return bundleTreeActualFile{}, err
	}
	if identity.SHA256 != digest || identity.SizeBytes != int64(len(canonical)) {
		return bundleTreeActualFile{}, errors.New("server-authored tree manifest failed verification")
	}
	return identity, nil
}

func finalizedBundleFileIdentity(path string, verifyContent bool) (bundleTreeActualFile, error) {
	pathInfo, err := os.Lstat(path)
	if err != nil {
		return bundleTreeActualFile{}, err
	}
	if pathInfo.Mode()&os.ModeSymlink != 0 || !pathInfo.Mode().IsRegular() {
		return bundleTreeActualFile{}, fmt.Errorf("bundle contains non-regular entry %q", filepath.Base(path))
	}
	if !verifyContent {
		// Presence + size only: trust the sha256 verified at commit time. Used for
		// large bundles (e.g. multi-GB OME-Zarr with thousands of chunks) where
		// re-reading every file synchronously at finalize would block the request
		// and risk an edge-proxy timeout. SHA256 is left empty; the caller
		// substitutes the committed hash for the manifest.
		return bundleTreeActualFile{SizeBytes: pathInfo.Size()}, nil
	}
	file, err := os.Open(path)
	if err != nil {
		return bundleTreeActualFile{}, err
	}
	before, err := file.Stat()
	if err != nil {
		_ = file.Close()
		return bundleTreeActualFile{}, err
	}
	if !before.Mode().IsRegular() {
		_ = file.Close()
		return bundleTreeActualFile{}, fmt.Errorf("bundle contains non-regular entry %q", filepath.Base(path))
	}
	hasher := sha256.New()
	written, hashErr := copyWithPooledBuffer(hasher, file)
	after, statErr := file.Stat()
	closeErr := file.Close()
	if hashErr != nil {
		return bundleTreeActualFile{}, hashErr
	}
	if statErr != nil {
		return bundleTreeActualFile{}, statErr
	}
	if closeErr != nil {
		return bundleTreeActualFile{}, closeErr
	}
	if written != before.Size() || written != after.Size() ||
		!before.ModTime().Equal(after.ModTime()) {
		return bundleTreeActualFile{}, fmt.Errorf("bundle member %q changed while hashing", filepath.Base(path))
	}
	current, err := os.Lstat(path)
	if err != nil {
		return bundleTreeActualFile{}, err
	}
	if current.Mode()&os.ModeSymlink != 0 || !current.Mode().IsRegular() ||
		!os.SameFile(after, current) {
		return bundleTreeActualFile{}, fmt.Errorf("bundle member %q changed while hashing", filepath.Base(path))
	}
	return bundleTreeActualFile{
		SHA256: hex.EncodeToString(hasher.Sum(nil)), SizeBytes: written,
	}, nil
}

func relativeBundleMemberPath(bundleTop string, bundle bundleInfo, raw string) (string, bool) {
	safe, ok := sanitizeBundleRelPath(raw)
	if !ok {
		return "", false
	}
	parts := strings.Split(safe, "/")
	safeTop, ok := sanitizeBundleSegment(bundleTop)
	if !ok || safeTop != bundle.Name || len(parts) < 2 || parts[0] != safeTop {
		return "", false
	}
	return strings.Join(parts[1:], "/"), true
}

func resourceTreeIdentityDescriptor(manifestSHA256 string) domain.JSONMap {
	digest := strings.ToLower(strings.TrimSpace(manifestSHA256))
	if !isSHA256Hex(digest) {
		return nil
	}
	return domain.JSONMap{
		"schema":                resourceTreeIdentitySchema,
		"authority":             "control_resource_catalog",
		"canonical_json_schema": canonicalJSONSchema,
		"tree_manifest_schema":  treeManifestSchema,
		"tree_manifest_path":    treeManifestPath,
		"tree_manifest_sha256":  digest,
		"scope":                 treeIdentityScope,
	}
}

// dirSizeBytes sums the sizes of all regular files under dir (best-effort).
func dirSizeBytes(dir string) int64 {
	var total int64
	_ = filepath.Walk(dir, func(_ string, info os.FileInfo, err error) error {
		if err == nil && info != nil && info.Mode().IsRegular() {
			total += info.Size()
		}
		return nil
	})
	return total
}
