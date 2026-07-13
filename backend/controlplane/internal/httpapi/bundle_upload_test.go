package httpapi

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
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

func TestFinalizedBundleTreeIdentityAuthorsCanonicalSensorManifest(t *testing.T) {
	root := t.TempDir()
	bundle := bundleInfo{ID: "file_tree", Name: "signals.zarr", FormatID: "ome-zarr"}
	dir := filepath.Join(root, bundle.Name)
	contents := map[string][]byte{
		".zattrs":     []byte(`{"ultra":{"sensor_series":{"schema":"ultra.sensor-series.v1"}}}`),
		".zgroup":     []byte(`{"zarr_format":2}`),
		"signals/a/0": []byte("01234567"),
	}
	files := make([]domain.UploadSessionFileRecord, 0, len(contents))
	manifestEntries := make([]bundleTreeManifestEntry, 0, len(contents))
	for relative, payload := range contents {
		path := filepath.Join(dir, filepath.FromSlash(relative))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, payload, 0o644); err != nil {
			t.Fatal(err)
		}
		digest := sha256.Sum256(payload)
		digestText := hex.EncodeToString(digest[:])
		manifestEntries = append(manifestEntries, bundleTreeManifestEntry{
			Path: relative, SHA256: digestText, SizeBytes: int64(len(payload)),
		})
		files = append(files, domain.UploadSessionFileRecord{
			RelativePath:   bundle.Name + "/" + relative,
			ResourceID:     bundle.ID,
			SizeBytes:      int64(len(payload)),
			ComputedSHA256: digestText,
			Status:         "completed",
		})
	}
	sort.Slice(manifestEntries, func(i, j int) bool { return manifestEntries[i].Path < manifestEntries[j].Path })
	manifestBytes, err := json.Marshal(bundleTreeManifest{
		Entries: manifestEntries,
		Schema:  treeManifestSchema,
	})
	if err != nil {
		t.Fatal(err)
	}
	manifestFile := filepath.Join(dir, filepath.FromSlash(treeManifestPath))
	identity, err := finalizedBundleTreeIdentity(dir, bundle.Name, bundle, files)
	if err != nil {
		t.Fatalf("finalizedBundleTreeIdentity: %v", err)
	}
	wantDigest := sha256.Sum256(manifestBytes)
	if identity.ManifestSHA256 != hex.EncodeToString(wantDigest[:]) {
		t.Fatalf("manifest sha256=%q want=%q", identity.ManifestSHA256, hex.EncodeToString(wantDigest[:]))
	}
	if identity.EntryCount != len(contents) {
		t.Fatalf("entry count=%d want=%d", identity.EntryCount, len(contents))
	}
	wantSize := int64(len(manifestBytes))
	for _, payload := range contents {
		wantSize += int64(len(payload))
	}
	if identity.SizeBytes != wantSize {
		t.Fatalf("size=%d want=%d", identity.SizeBytes, wantSize)
	}
	authored, err := os.ReadFile(manifestFile)
	if err != nil {
		t.Fatalf("read server-authored manifest: %v", err)
	}
	if string(authored) != string(manifestBytes) {
		t.Fatalf("server-authored manifest=%s want=%s", authored, manifestBytes)
	}
	firstManifestInfo, err := os.Stat(manifestFile)
	if err != nil {
		t.Fatal(err)
	}
	replayed, err := finalizedBundleTreeIdentity(dir, bundle.Name, bundle, files)
	if err != nil {
		t.Fatalf("retry finalizedBundleTreeIdentity: %v", err)
	}
	secondManifestInfo, err := os.Stat(manifestFile)
	if err != nil {
		t.Fatal(err)
	}
	if replayed != identity || !os.SameFile(firstManifestInfo, secondManifestInfo) {
		t.Fatalf("retry mutated identity or exact manifest: first=%#v replay=%#v", identity, replayed)
	}

	// A file outside the upload session's closed member set makes the tree
	// non-finalizable instead of silently escaping the identity.
	if err := os.WriteFile(filepath.Join(dir, "extra"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := finalizedBundleTreeIdentity(dir, bundle.Name, bundle, files); err == nil ||
		!strings.Contains(err.Error(), "closure mismatch") {
		t.Fatalf("extra-file result=%v, want closure mismatch", err)
	}
	if err := os.Remove(filepath.Join(dir, "extra")); err != nil {
		t.Fatal(err)
	}
	// A same-size mutation after member completion must not inherit the upload
	// hash. Finalization rereads exact bytes before authoring the tree identity.
	tamperedPath := filepath.Join(dir, filepath.FromSlash("signals/a/0"))
	if err := os.WriteFile(tamperedPath, []byte("76543210"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := finalizedBundleTreeIdentity(dir, bundle.Name, bundle, files); err == nil ||
		!strings.Contains(err.Error(), "content changed after verified upload") {
		t.Fatalf("same-size tamper result=%v, want content-change rejection", err)
	}
}

func TestFinalizedBundleTreeIdentityReplacesMismatchedClientManifest(t *testing.T) {
	root := t.TempDir()
	bundle := bundleInfo{ID: "file_tree", Name: "signals.zarr", FormatID: "ome-zarr"}
	dir := filepath.Join(root, bundle.Name)
	payload := []byte(`{"zarr_format":2}`)
	payloadPath := filepath.Join(dir, ".zgroup")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(payloadPath, payload, 0o644); err != nil {
		t.Fatal(err)
	}
	payloadDigest := sha256.Sum256(payload)
	clientManifest := []byte("{\"entries\":[],\"schema\":\"client-controlled\"}\n")
	manifestPath := filepath.Join(dir, filepath.FromSlash(treeManifestPath))
	if err := os.MkdirAll(filepath.Dir(manifestPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(manifestPath, clientManifest, 0o600); err != nil {
		t.Fatal(err)
	}
	clientInfo, err := os.Stat(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	clientDigest := sha256.Sum256(clientManifest)
	files := []domain.UploadSessionFileRecord{
		{
			RelativePath: bundle.Name + "/.zgroup", ResourceID: bundle.ID,
			SizeBytes: int64(len(payload)), ComputedSHA256: hex.EncodeToString(payloadDigest[:]), Status: "completed",
		},
		{
			RelativePath: bundle.Name + "/" + treeManifestPath, ResourceID: bundle.ID,
			SizeBytes: int64(len(clientManifest)), ComputedSHA256: hex.EncodeToString(clientDigest[:]), Status: "completed",
		},
	}
	wantManifest, err := json.Marshal(bundleTreeManifest{
		Entries: []bundleTreeManifestEntry{{
			Path: ".zgroup", SHA256: hex.EncodeToString(payloadDigest[:]), SizeBytes: int64(len(payload)),
		}},
		Schema: treeManifestSchema,
	})
	if err != nil {
		t.Fatal(err)
	}

	identity, err := finalizedBundleTreeIdentity(dir, bundle.Name, bundle, files)
	if err != nil {
		t.Fatalf("finalizedBundleTreeIdentity: %v", err)
	}
	authored, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(authored) != string(wantManifest) {
		t.Fatalf("mismatched client manifest was not replaced: got=%s want=%s", authored, wantManifest)
	}
	serverInfo, err := os.Stat(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	if os.SameFile(clientInfo, serverInfo) {
		t.Fatal("mismatched client manifest was modified in place instead of atomically replaced")
	}
	wantDigest := sha256.Sum256(wantManifest)
	if identity.ManifestSHA256 != hex.EncodeToString(wantDigest[:]) ||
		identity.SizeBytes != int64(len(payload)+len(wantManifest)) {
		t.Fatalf("server identity=%#v", identity)
	}
}

func TestFinalizedBundleTreeIdentityRejectsIncompleteDuplicateAndSymlinkMembers(t *testing.T) {
	root := t.TempDir()
	bundle := bundleInfo{ID: "file_tree", Name: "signals.zarr", FormatID: "ome-zarr"}
	dir := filepath.Join(root, bundle.Name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	payload := []byte("data")
	path := filepath.Join(dir, ".zattrs")
	if err := os.WriteFile(path, payload, 0o644); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(payload)
	file := domain.UploadSessionFileRecord{
		RelativePath:   bundle.Name + "/.zattrs",
		ResourceID:     bundle.ID,
		SizeBytes:      int64(len(payload)),
		ComputedSHA256: hex.EncodeToString(digest[:]),
		Status:         "uploading",
	}
	if _, err := finalizedBundleTreeIdentity(dir, bundle.Name, bundle, []domain.UploadSessionFileRecord{file}); err == nil ||
		!strings.Contains(err.Error(), "not committed") {
		t.Fatalf("incomplete result=%v, want not committed", err)
	}
	file.Status = "completed"
	if _, err := finalizedBundleTreeIdentity(dir, bundle.Name, bundle, []domain.UploadSessionFileRecord{file, file}); err == nil ||
		!strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("duplicate result=%v, want duplicate", err)
	}
	manifestPayload := []byte("client manifest")
	manifestDigest := sha256.Sum256(manifestPayload)
	incompleteManifest := domain.UploadSessionFileRecord{
		RelativePath: bundle.Name + "/" + treeManifestPath, ResourceID: bundle.ID,
		SizeBytes: int64(len(manifestPayload)), ComputedSHA256: hex.EncodeToString(manifestDigest[:]), Status: "uploading",
	}
	if _, err := finalizedBundleTreeIdentity(
		dir, bundle.Name, bundle, []domain.UploadSessionFileRecord{file, incompleteManifest},
	); err == nil || !strings.Contains(err.Error(), "not committed") {
		t.Fatalf("incomplete client manifest result=%v, want not committed", err)
	}
	link := filepath.Join(dir, "linked")
	if err := os.Symlink(path, link); err != nil {
		t.Fatal(err)
	}
	if _, err := finalizedBundleTreeIdentity(dir, bundle.Name, bundle, []domain.UploadSessionFileRecord{file}); err == nil ||
		!strings.Contains(err.Error(), "symbolic link") {
		t.Fatalf("symlink result=%v, want symbolic-link rejection", err)
	}
	if err := os.Remove(link); err != nil {
		t.Fatal(err)
	}
	manifestPath := filepath.Join(dir, filepath.FromSlash(treeManifestPath))
	if err := os.MkdirAll(filepath.Dir(manifestPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(path, manifestPath); err != nil {
		t.Fatal(err)
	}
	if _, err := finalizedBundleTreeIdentity(dir, bundle.Name, bundle, []domain.UploadSessionFileRecord{file}); err == nil ||
		!strings.Contains(err.Error(), "symbolic link") {
		t.Fatalf("manifest symlink result=%v, want symbolic-link rejection", err)
	}
}

func TestFinalizeUploadBundleCatalogsCanonicalTreeIdentityAndIsIdempotent(t *testing.T) {
	uploadRoot := t.TempDir()
	memory := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(memory, eventbus.NewMemoryBus()),
		Store:      memory,
		UploadRoot: uploadRoot,
	})
	const userID, orgID, bundleName = "sensor-user", "sensor-org", "signals.zarr"
	contents := map[string][]byte{
		".zattrs":     []byte(`{"ultra":{"sensor_series":{"schema":"ultra.sensor-series.v1"}}}`),
		".zgroup":     []byte(`{"zarr_format":2}`),
		"signals/a/0": []byte("01234567"),
	}
	entries := make([]bundleTreeManifestEntry, 0, len(contents))
	for relative, payload := range contents {
		digest := sha256.Sum256(payload)
		entries = append(entries, bundleTreeManifestEntry{
			Path: relative, SHA256: hex.EncodeToString(digest[:]), SizeBytes: int64(len(payload)),
		})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Path < entries[j].Path })
	manifestBytes, err := json.Marshal(bundleTreeManifest{Entries: entries, Schema: treeManifestSchema})
	if err != nil {
		t.Fatal(err)
	}

	create := createUploadSessionRequest{
		IdempotencyKey: "sensor-tree-finalize",
		Files:          make([]createUploadSessionFileRequest, 0, len(contents)),
	}
	fixtures := make([]uploadFileFixture, 0, len(contents))
	for relative, payload := range contents {
		digest := sha256.Sum256(payload)
		token := strings.NewReplacer("/", "-", ".", "_").Replace(relative)
		fixtures = append(fixtures, uploadFileFixture{
			token: token, name: filepath.Base(relative), path: bundleName + "/" + relative,
			payload: payload, sha: hex.EncodeToString(digest[:]),
		})
		create.TotalBytes += int64(len(payload))
		create.Files = append(create.Files, createUploadSessionFileRequest{
			FileToken: token, OriginalName: filepath.Base(relative),
			RelativePath: bundleName + "/" + relative,
			ContentType:  "application/octet-stream", SizeBytes: int64(len(payload)),
			DeclaredSHA256: hex.EncodeToString(digest[:]),
		})
	}
	sort.Slice(fixtures, func(i, j int) bool { return fixtures[i].path < fixtures[j].path })
	createBody, err := json.Marshal(create)
	if err != nil {
		t.Fatal(err)
	}
	created := createUploadSessionForTest(
		t, router, string(createBody), userID, orgID, http.StatusCreated,
	)
	finalizeURL := "/v2/upload-sessions/" + created.Session.SessionID + "/finalize-bundle"
	finalize := func(wantStatus int) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPost, finalizeURL, nil)
		req.Header.Set("X-Ultra-User-Id", userID)
		req.Header.Set("X-Ultra-Org-Id", orgID)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != wantStatus {
			t.Fatalf("finalize status=%d body=%s want=%d", rec.Code, rec.Body.String(), wantStatus)
		}
		return rec
	}
	finalize(http.StatusConflict)
	for _, fixture := range fixtures {
		uploadChunkForTest(
			t, router, created.Session.SessionID, fixture.token, 0, 0,
			fixture.payload, fixture.sha, userID, orgID,
		)
		completeUploadSessionFileForTest(
			t, router, created.Session.SessionID, fixture.token, userID, orgID, http.StatusOK,
		)
	}
	first := finalize(http.StatusOK)
	var finalized struct {
		Bundles []uploadedFileRecord `json:"bundles"`
	}
	if err := json.Unmarshal(first.Body.Bytes(), &finalized); err != nil {
		t.Fatal(err)
	}
	if len(finalized.Bundles) != 1 {
		t.Fatalf("finalized bundles=%#v", finalized.Bundles)
	}
	manifestDigest := sha256.Sum256(manifestBytes)
	wantDigest := hex.EncodeToString(manifestDigest[:])
	if finalized.Bundles[0].SHA256 != wantDigest {
		t.Fatalf("bundle digest=%q want=%q", finalized.Bundles[0].SHA256, wantDigest)
	}
	bundleDir := bundleDirPath(uploadRoot, bundleInfo{
		ID: finalized.Bundles[0].FileID, Name: bundleName, FormatID: "ome-zarr",
	})
	manifestPath := filepath.Join(bundleDir, filepath.FromSlash(treeManifestPath))
	authored, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read server-authored manifest: %v", err)
	}
	if string(authored) != string(manifestBytes) {
		t.Fatalf("server-authored manifest=%s want=%s", authored, manifestBytes)
	}
	firstManifestInfo, err := os.Stat(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	cataloged, err := memory.GetResourceForUser(
		context.Background(), finalized.Bundles[0].FileID, userID, orgID,
	)
	if err != nil {
		t.Fatal(err)
	}
	wantPhysicalSize := create.TotalBytes + int64(len(manifestBytes))
	if cataloged.SHA256 != wantDigest || cataloged.SizeBytes != wantPhysicalSize ||
		dirSizeBytes(bundleDir) != wantPhysicalSize {
		t.Fatalf("cataloged resource=%#v", cataloged)
	}
	second := finalize(http.StatusOK)
	var replayed struct {
		Bundles []uploadedFileRecord `json:"bundles"`
	}
	if err := json.Unmarshal(second.Body.Bytes(), &replayed); err != nil {
		t.Fatal(err)
	}
	if len(replayed.Bundles) != 1 || replayed.Bundles[0].FileID != finalized.Bundles[0].FileID ||
		replayed.Bundles[0].SHA256 != wantDigest ||
		replayed.Bundles[0].CreatedAt != finalized.Bundles[0].CreatedAt {
		t.Fatalf("idempotent finalize changed identity: first=%#v second=%#v", finalized, replayed)
	}
	secondManifestInfo, err := os.Stat(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	if !os.SameFile(firstManifestInfo, secondManifestInfo) {
		t.Fatal("idempotent finalize rewrote an already exact server manifest")
	}
}

func BenchmarkFinalizedBundleTreeIdentity10000Members(b *testing.B) {
	const memberCount = 10_000
	root := b.TempDir()
	bundle := bundleInfo{ID: "file_benchmark", Name: "signals.zarr", FormatID: "ome-zarr"}
	dir := filepath.Join(root, bundle.Name)
	files := make([]domain.UploadSessionFileRecord, 0, memberCount)
	payload := []byte("0123456789abcdef")
	digest := sha256.Sum256(payload)
	digestText := hex.EncodeToString(digest[:])
	for index := 0; index < memberCount; index++ {
		relative := fmt.Sprintf("signals/channel-%04d/%04d", index/100, index)
		path := filepath.Join(dir, filepath.FromSlash(relative))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			b.Fatal(err)
		}
		if err := os.WriteFile(path, payload, 0o644); err != nil {
			b.Fatal(err)
		}
		files = append(files, domain.UploadSessionFileRecord{
			RelativePath: bundle.Name + "/" + relative, ResourceID: bundle.ID,
			SizeBytes: int64(len(payload)), ComputedSHA256: digestText, Status: "completed",
		})
	}
	b.ReportAllocs()
	b.SetBytes(int64(memberCount * len(payload)))
	b.ResetTimer()
	for iteration := 0; iteration < b.N; iteration++ {
		identity, err := finalizedBundleTreeIdentity(dir, bundle.Name, bundle, files)
		if err != nil {
			b.Fatal(err)
		}
		if identity.EntryCount != memberCount {
			b.Fatalf("entry count=%d", identity.EntryCount)
		}
	}
	b.ReportMetric(memberCount, "entries/op")
}

func BenchmarkFinalizedBundleTreeIdentity64MiB(b *testing.B) {
	const payloadSize = 64 << 20
	root := b.TempDir()
	bundle := bundleInfo{ID: "file_large", Name: "signals.zarr", FormatID: "ome-zarr"}
	dir := filepath.Join(root, bundle.Name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		b.Fatal(err)
	}
	payload := make([]byte, payloadSize)
	path := filepath.Join(dir, "large-chunk")
	if err := os.WriteFile(path, payload, 0o644); err != nil {
		b.Fatal(err)
	}
	digest := sha256.Sum256(payload)
	files := []domain.UploadSessionFileRecord{{
		RelativePath: bundle.Name + "/large-chunk", ResourceID: bundle.ID,
		SizeBytes: payloadSize, ComputedSHA256: hex.EncodeToString(digest[:]), Status: "completed",
	}}
	payload = nil
	b.ReportAllocs()
	b.SetBytes(payloadSize)
	b.ResetTimer()
	for iteration := 0; iteration < b.N; iteration++ {
		if _, err := finalizedBundleTreeIdentity(dir, bundle.Name, bundle, files); err != nil {
			b.Fatal(err)
		}
	}
}
