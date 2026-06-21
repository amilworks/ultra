package httpapi

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

func TestSanitizeBundleSegmentPreservesZarrNames(t *testing.T) {
	// zarr metadata + chunk keys must survive byte-for-byte.
	for _, ok := range []string{".zattrs", ".zgroup", ".zarray", ".zmetadata", "0", "0.0.0", "labels", "scan.ome.zarr"} {
		if got, valid := sanitizeBundleSegment(ok); !valid || got != ok {
			t.Errorf("sanitizeBundleSegment(%q) = %q,%v; want unchanged+valid", ok, got, valid)
		}
	}
	// Traversal / illegal segments rejected (not rewritten).
	for _, bad := range []string{"", ".", "..", "a/b", "a\x00b", "evil;rm", "../x", " "} {
		if got, valid := sanitizeBundleSegment(bad); valid {
			t.Errorf("sanitizeBundleSegment(%q) = %q,true; want rejected", bad, got)
		}
	}
}

func TestSanitizeBundleRelPathRejectsTraversal(t *testing.T) {
	ok, valid := sanitizeBundleRelPath("scan.ome.zarr/0/0/0")
	if !valid || ok != "scan.ome.zarr/0/0/0" {
		t.Fatalf("clean path mangled: %q,%v", ok, valid)
	}
	if _, valid := sanitizeBundleRelPath("scan.ome.zarr/.zattrs"); !valid {
		t.Fatalf(".zattrs should be allowed")
	}
	for _, bad := range []string{"../etc/passwd", "scan.ome.zarr/../../etc", "/abs/path", "", "scan.ome.zarr/../x"} {
		if _, valid := sanitizeBundleRelPath(bad); valid {
			t.Errorf("sanitizeBundleRelPath(%q) accepted a traversal/invalid path", bad)
		}
	}
}

func TestDetectSessionBundles(t *testing.T) {
	files := []domain.UpsertUploadSessionFileInput{
		{RelativePath: "scan.ome.zarr/.zattrs"},
		{RelativePath: "scan.ome.zarr/0/0/0"},
		{RelativePath: "notes.txt"}, // not a bundle
		{RelativePath: "plain.tif"}, // not a bundle
	}
	n := 0
	bundles := detectSessionBundles(files, func() string { n++; return "file_b" + string(rune('0'+n)) })
	if len(bundles) != 1 {
		t.Fatalf("expected 1 bundle, got %d: %+v", len(bundles), bundles)
	}
	b, ok := bundles["scan.ome.zarr"]
	if !ok || b.FormatID != "ome-zarr" || b.Name != "scan.ome.zarr" || b.ID == "" {
		t.Fatalf("unexpected bundle: %+v", bundles)
	}
}

func TestBundleMemberTargetContainment(t *testing.T) {
	root := t.TempDir()
	session := domain.UploadSessionRecord{
		Metadata: domain.JSONMap{"bundles": map[string]any{
			"scan.ome.zarr": map[string]any{"id": "file_bundle1", "name": "scan.ome.zarr", "format": "ome-zarr"},
		}},
	}
	// A legitimate member resolves under bundles/file_bundle1/.
	dest, b, ok := bundleMemberTarget(root, session, domain.UploadSessionFileRecord{RelativePath: "scan.ome.zarr/0/0/0"})
	if !ok || b.ID != "file_bundle1" {
		t.Fatalf("legit member not resolved: %v %+v", ok, b)
	}
	want := filepath.Join(root, "bundles", "file_bundle1", "scan.ome.zarr", "0", "0", "0")
	if dest != want {
		t.Fatalf("dest=%q want=%q", dest, want)
	}
	if !pathIsUnderRoot(filepath.Join(root, "bundles", "file_bundle1"), dest) {
		t.Fatalf("dest escaped the bundle root")
	}
	// A traversal member is rejected.
	if _, _, ok := bundleMemberTarget(root, session, domain.UploadSessionFileRecord{RelativePath: "scan.ome.zarr/../../evil"}); ok {
		t.Fatalf("traversal member should be rejected")
	}
	// A file not in any bundle (different top segment) is not a member.
	if _, _, ok := bundleMemberTarget(root, session, domain.UploadSessionFileRecord{RelativePath: "other.tif"}); ok {
		t.Fatalf("non-bundle file should not be a member")
	}
}

func TestSessionBundlesRoundTrip(t *testing.T) {
	in := map[string]bundleInfo{"a.ome.zarr": {ID: "file_x", Name: "a.ome.zarr", FormatID: "ome-zarr"}}
	session := domain.UploadSessionRecord{Metadata: domain.JSONMap{"bundles": bundleMetadataValue(in)}}
	out := sessionBundles(session)
	if len(out) != 1 || out["a.ome.zarr"].ID != "file_x" || !strings.HasSuffix(out["a.ome.zarr"].Name, ".ome.zarr") {
		t.Fatalf("round-trip mismatch: %+v", out)
	}
}
