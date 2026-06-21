package httpapi

import (
	"os"
	"path/filepath"
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
