package httpapi

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

func strictManifestFixture(t *testing.T) (string, string, resourceRecord, map[string]any, string) {
	t.Helper()
	root := t.TempDir()
	sourcePath := filepath.Join(root, "source.ome.tif")
	sourceBytes := []byte("source-generation-one")
	if err := os.WriteFile(sourcePath, sourceBytes, 0o644); err != nil {
		t.Fatal(err)
	}
	record := resourceRecord{
		FileID:    "file-strict-manifest",
		SHA256:    fmt.Sprintf("%x", sha256.Sum256(sourceBytes)),
		SizeBytes: int64(len(sourceBytes)),
	}
	viewerInfo := derivativeViewerInfoForTest(2, 2, 3, 8, 16, 512)
	capabilities := derivativeCapabilitiesForViewer(viewerInfo, func() derivativeSemantics {
		semantics, _ := viewerDerivativeSemantics(viewerInfo)
		return semantics
	}())
	artifactPath := writeStrictDerivativeForTest(
		t, root, sourcePath, record, viewerInfo,
		capabilities,
		[]byte("strict derivative artifact"),
	)
	return root, sourcePath, record, viewerInfo, artifactPath
}

func TestReadDerivativeManifestRejectsUncommittedOrCorruptGenerations(t *testing.T) {
	t.Parallel()

	for _, corruption := range []string{
		"legacy-only",
		"unknown-field",
		"trailing-json",
		"duplicate-key",
		"oversize",
		"traversal",
		"symlink-artifact",
		"artifact-digest",
		"source-catalog",
		"source-bytes",
		"producer-revision",
		"effective-spec",
	} {
		corruption := corruption
		t.Run(corruption, func(t *testing.T) {
			t.Parallel()
			root, sourcePath, record, _, artifactPath := strictManifestFixture(t)
			manifestPath := derivedPyramidManifestPath(root, record.FileID)
			manifestBytes, err := os.ReadFile(manifestPath)
			if err != nil {
				t.Fatal(err)
			}
			var manifest map[string]any
			if err := json.Unmarshal(manifestBytes, &manifest); err != nil {
				t.Fatal(err)
			}
			switch corruption {
			case "legacy-only":
				if err := os.Remove(manifestPath); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(root, "derived", derivedPyramidName(record.FileID)), []byte("legacy"), 0o644); err != nil {
					t.Fatal(err)
				}
			case "unknown-field":
				manifest["unexpected"] = true
				manifestBytes, _ = json.Marshal(manifest)
				if err := os.WriteFile(manifestPath, manifestBytes, 0o644); err != nil {
					t.Fatal(err)
				}
			case "trailing-json":
				if err := os.WriteFile(manifestPath, append(manifestBytes, []byte("\n{}")...), 0o644); err != nil {
					t.Fatal(err)
				}
			case "duplicate-key":
				duplicate := strings.Replace(string(manifestBytes), `"schema":`, `"schema":"duplicate","schema":`, 1)
				if err := os.WriteFile(manifestPath, []byte(duplicate), 0o644); err != nil {
					t.Fatal(err)
				}
			case "oversize":
				if err := os.WriteFile(manifestPath, []byte("{"+strings.Repeat(" ", maxDerivedPyramidManifestBytes)+"}"), 0o644); err != nil {
					t.Fatal(err)
				}
			case "traversal":
				manifest["artifact"].(map[string]any)["basename"] = "../outside.tif"
				manifestBytes, _ = json.Marshal(manifest)
				if err := os.WriteFile(manifestPath, manifestBytes, 0o644); err != nil {
					t.Fatal(err)
				}
			case "symlink-artifact":
				if err := os.Remove(artifactPath); err != nil {
					t.Fatal(err)
				}
				if err := os.Symlink(sourcePath, artifactPath); err != nil {
					t.Fatal(err)
				}
			case "artifact-digest":
				if err := os.WriteFile(artifactPath, []byte("tampered derivative artifact"), 0o644); err != nil {
					t.Fatal(err)
				}
			case "source-catalog":
				record.SHA256 = strings.Repeat("0", 64)
			case "source-bytes":
				if err := os.WriteFile(sourcePath, []byte("different-source-generation"), 0o644); err != nil {
					t.Fatal(err)
				}
			case "producer-revision":
				manifest["conversion_spec"].(map[string]any)["producer_revision"] = "mutable-build-label"
				manifestBytes, _ = json.Marshal(manifest)
				if err := os.WriteFile(manifestPath, manifestBytes, 0o644); err != nil {
					t.Fatal(err)
				}
			case "effective-spec":
				manifest["conversion_spec"].(map[string]any)["effective"].(map[string]any)["fmt"] = "auto"
				manifestBytes, _ = json.Marshal(manifest)
				if err := os.WriteFile(manifestPath, manifestBytes, 0o644); err != nil {
					t.Fatal(err)
				}
			}

			if manifest, artifact, admitted := readDerivativeManifest(root, record, sourcePath); admitted || artifact != "" || manifest.Schema != "" {
				t.Fatalf("corrupt generation admitted: manifest=%+v artifact=%q", manifest, artifact)
			}
		})
	}
}

func TestDerivativeAdmissionCacheRejectsSameSizeSameMtimeArtifactReplacement(t *testing.T) {
	t.Parallel()

	root, sourcePath, record, _, artifactPath := strictManifestFixture(t)
	if _, _, admitted := readDerivativeManifest(root, record, sourcePath); !admitted {
		t.Fatal("initial strict generation was not admitted")
	}
	before, err := os.Stat(artifactPath)
	if err != nil {
		t.Fatal(err)
	}
	replacement := artifactPath + ".replacement"
	original, err := os.ReadFile(artifactPath)
	if err != nil {
		t.Fatal(err)
	}
	mutated := append([]byte(nil), original...)
	mutated[0] ^= 0xff
	if err := os.WriteFile(replacement, mutated, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(replacement, before.ModTime(), before.ModTime()); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(replacement, artifactPath); err != nil {
		t.Fatal(err)
	}

	if _, _, admitted := readDerivativeManifest(root, record, sourcePath); admitted {
		t.Fatal("same-size/same-mtime replacement reused cached artifact admission")
	}
}

func TestReadDerivativeManifestAdmitsPortableSourceIdentity(t *testing.T) {
	t.Parallel()

	root, sourcePath, record, _, _ := strictManifestFixture(t)
	manifestPath := derivedPyramidManifestPath(root, record.FileID)
	manifestBytes, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	var manifest map[string]any
	if err := json.Unmarshal(manifestBytes, &manifest); err != nil {
		t.Fatal(err)
	}
	delete(manifest["source"].(map[string]any), "stat")
	manifestBytes, err = json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(manifestPath, manifestBytes, 0o644); err != nil {
		t.Fatal(err)
	}

	if _, _, admitted := readDerivativeManifest(root, record, sourcePath); !admitted {
		t.Fatal("portable SHA-256/size source identity was not admitted")
	}
}

func TestDerivativeAdmissionRejectsSameSizeSameMtimeSourceReplacement(t *testing.T) {
	t.Parallel()

	root, sourcePath, record, _, _ := strictManifestFixture(t)
	if _, _, admitted := readDerivativeManifest(root, record, sourcePath); !admitted {
		t.Fatal("initial strict generation was not admitted")
	}
	before, err := os.Stat(sourcePath)
	if err != nil {
		t.Fatal(err)
	}
	original, err := os.ReadFile(sourcePath)
	if err != nil {
		t.Fatal(err)
	}
	mutated := append([]byte(nil), original...)
	mutated[0] ^= 0xff
	replacement := sourcePath + ".replacement"
	if err := os.WriteFile(replacement, mutated, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(replacement, before.ModTime(), before.ModTime()); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(replacement, sourcePath); err != nil {
		t.Fatal(err)
	}

	if _, _, admitted := readDerivativeManifest(root, record, sourcePath); admitted {
		t.Fatal("same-size/same-mtime source replacement reused cached source admission")
	}
}

func TestDerivativeValidationCachesEvictIncrementallyByRecency(t *testing.T) {
	t.Parallel()

	digests := newBoundedDerivativeCache[string](3)
	digests.put("a", "digest-a")
	digests.put("b", "digest-b")
	digests.put("c", "digest-c")
	if _, ok := digests.get("a"); !ok {
		t.Fatal("failed to warm most-recently-used digest entry")
	}
	digests.put("d", "digest-d")
	if _, ok := digests.get("b"); ok {
		t.Fatal("least-recently-used digest entry survived bounded eviction")
	}
	for _, key := range []string{"a", "c", "d"} {
		if _, ok := digests.get(key); !ok {
			t.Fatalf("incremental digest eviction dropped newer entry %q", key)
		}
	}
	if digests.len() != 3 {
		t.Fatalf("digest cache length = %d, want 3", digests.len())
	}

	admissions := newBoundedDerivativeCache[derivativeAdmissionEntry](1)
	admissions.put("old", derivativeAdmissionEntry{artifactPath: "old"})
	admissions.put("new", derivativeAdmissionEntry{artifactPath: "new"})
	if _, ok := admissions.get("old"); ok {
		t.Fatal("bounded admission cache retained its evicted entry")
	}
	if current, ok := admissions.get("new"); !ok || current.artifactPath != "new" {
		t.Fatalf("bounded admission cache lost newest entry: %+v, %t", current, ok)
	}
}

func TestDerivativeCapabilitiesRequireRouteSpecificSelectorSupport(t *testing.T) {
	t.Parallel()

	capabilities := derivativeCapabilities{
		Tile: true, Slice: true, Atlas: true, Thumbnail: true, OrderedChannels: true, LUT: true,
	}
	for _, tc := range []struct {
		name string
		use  derivativeUse
		want bool
	}{
		{name: "tile base", use: derivativeUse{capability: "tile"}, want: true},
		{name: "tile t", use: derivativeUse{capability: "tile", requireT: true}},
		{name: "tile z", use: derivativeUse{capability: "tile", requireZ: true}},
		{name: "slice selectors", use: derivativeUse{capability: "slice", requireT: true, requireZ: true}, want: true},
		{name: "atlas base", use: derivativeUse{capability: "atlas"}, want: true},
		{name: "atlas t", use: derivativeUse{capability: "atlas", requireT: true}},
		{name: "lut", use: derivativeUse{capability: "slice", requireChannels: true, requireLUT: true}, want: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := capabilities.supports(tc.use); got != tc.want {
				t.Fatalf("supports(%+v) = %t, want %t", tc.use, got, tc.want)
			}
		})
	}
}

func TestDerivativeRouteFallsBackUnlessSelectorCapabilityIsProven(t *testing.T) {
	t.Parallel()

	viewerInfo := derivativeViewerInfoForTest(2, 2, 3, 8, 16, 512)
	paths := map[string]string{}
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/viewerinfo":
			_ = json.NewEncoder(w).Encode(viewerInfo)
		case "/tile", "/atlas", "/slice":
			paths[r.URL.Path] = r.URL.Query().Get("path")
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write([]byte("PNG"))
		default:
			http.NotFound(w, r)
		}
	}))
	defer imageSvc.Close()

	root := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:           mem,
		UploadRoot:      root,
		ImageServiceURL: imageSvc.URL,
	})
	fileID := uploadNamedFileForProxyTest(t, router, "selectors.ome.tif", testPNGBytes(t, 4, 4))
	record := uploadedResourceRecordForTest(t, mem, fileID)
	artifactPath := writeStrictDerivativeForTest(
		t, root, uploadedSourcePathForTest(t, root, fileID), record, viewerInfo,
		derivativeCapabilitiesForViewer(viewerInfo, func() derivativeSemantics {
			semantics, _ := viewerDerivativeSemantics(viewerInfo)
			return semantics
		}()),
		[]byte("selector-limited derivative"),
	)

	for _, endpoint := range []string{
		"/tiles/z/0/0/0?t=0",
		"/atlas?t=0",
		"/slice?t=0&z=0",
	} {
		req := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+endpoint, nil)
		setProxyOwnerHeaders(req)
		recorder := httptest.NewRecorder()
		router.ServeHTTP(recorder, req)
		if recorder.Code != http.StatusOK {
			t.Fatalf("%s status=%d body=%s", endpoint, recorder.Code, recorder.Body.String())
		}
	}
	if paths["/tile"] == artifactPath {
		t.Fatalf("selector-limited derivative used for unsupported tile T selector: %#v", paths)
	}
	if paths["/atlas"] != artifactPath || paths["/slice"] != artifactPath {
		t.Fatalf("selector-capable atlas/slice did not use derivative: %#v", paths)
	}
}

func TestDerivedSliceNonSuccessRetriesEquivalentSourceSlice(t *testing.T) {
	t.Parallel()

	viewerInfo := derivativeViewerInfoForTest(1, 1, 1, 8, 16, 512)
	var attemptedPaths []string
	root := t.TempDir()
	mem := store.NewMemoryStore()
	var artifactPath string
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path := r.URL.Query().Get("path")
		switch r.URL.Path {
		case "/viewerinfo":
			_ = json.NewEncoder(w).Encode(viewerInfo)
		case "/slice":
			attemptedPaths = append(attemptedPaths, path)
			if path == artifactPath {
				http.Error(w, "derived decoder rejected generation", http.StatusUnprocessableEntity)
				return
			}
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write([]byte("SOURCE-PNG"))
		default:
			http.NotFound(w, r)
		}
	}))
	defer imageSvc.Close()
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:           mem,
		UploadRoot:      root,
		ImageServiceURL: imageSvc.URL,
	})
	fileID := uploadNamedFileForProxyTest(t, router, "fallback.ome.tif", testPNGBytes(t, 4, 4))
	record := uploadedResourceRecordForTest(t, mem, fileID)
	artifactPath = writeStrictDerivativeForTest(
		t, root, uploadedSourcePathForTest(t, root, fileID), record, viewerInfo,
		derivativeCapabilitiesForViewer(viewerInfo, func() derivativeSemantics {
			semantics, _ := viewerDerivativeSemantics(viewerInfo)
			return semantics
		}()),
		[]byte("strict derivative"),
	)

	req := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/slice?axis=z&t=0&z=0", nil)
	setProxyOwnerHeaders(req)
	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, req)
	if recorder.Code != http.StatusOK || recorder.Body.String() != "SOURCE-PNG" {
		t.Fatalf("slice status=%d body=%q", recorder.Code, recorder.Body.String())
	}
	want := []string{artifactPath, uploadedSourcePathForTest(t, root, fileID)}
	if fmt.Sprint(attemptedPaths) != fmt.Sprint(want) {
		t.Fatalf("slice attempts=%v, want derived then source %v", attemptedPaths, want)
	}
}

func TestDerivedTileAndAtlasNonSuccessRetryEquivalentSource(t *testing.T) {
	viewerInfo := derivativeViewerInfoForTest(1, 2, 3, 8, 16, 512)
	attemptedPaths := map[string][]string{}
	root := t.TempDir()
	mem := store.NewMemoryStore()
	var artifactPath string
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path := r.URL.Query().Get("path")
		switch r.URL.Path {
		case "/viewerinfo":
			_ = json.NewEncoder(w).Encode(viewerInfo)
		case "/tile", "/atlas":
			attemptedPaths[r.URL.Path] = append(attemptedPaths[r.URL.Path], path)
			if path == artifactPath {
				http.Error(w, "derived decoder rejected generation", http.StatusUnprocessableEntity)
				return
			}
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write([]byte("SOURCE-PNG"))
		default:
			http.NotFound(w, r)
		}
	}))
	defer imageSvc.Close()
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:           mem,
		UploadRoot:      root,
		ImageServiceURL: imageSvc.URL,
	})
	fileID := uploadNamedFileForProxyTest(t, router, "fallback.ome.tif", testPNGBytes(t, 4, 4))
	record := uploadedResourceRecordForTest(t, mem, fileID)
	sourcePath := uploadedSourcePathForTest(t, root, fileID)
	semantics, ok := viewerDerivativeSemantics(viewerInfo)
	if !ok {
		t.Fatal("viewer info did not produce derivative semantics")
	}
	artifactPath = writeStrictDerivativeForTest(
		t, root, sourcePath, record, viewerInfo,
		derivativeCapabilitiesForViewer(viewerInfo, semantics),
		[]byte("strict derivative"),
	)

	for _, endpoint := range []string{"/tiles/z/0/0/0", "/atlas"} {
		req := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+endpoint, nil)
		setProxyOwnerHeaders(req)
		recorder := httptest.NewRecorder()
		router.ServeHTTP(recorder, req)
		if recorder.Code != http.StatusOK || recorder.Body.String() != "SOURCE-PNG" {
			t.Fatalf("%s status=%d body=%q", endpoint, recorder.Code, recorder.Body.String())
		}
	}
	want := []string{artifactPath, sourcePath}
	for _, route := range []string{"/tile", "/atlas"} {
		if fmt.Sprint(attemptedPaths[route]) != fmt.Sprint(want) {
			t.Fatalf("%s attempts=%v, want derived then source %v", route, attemptedPaths[route], want)
		}
	}
}

func TestIntrinsicDerivativeCapabilityLimitationFallsBackWithoutRepair(t *testing.T) {
	t.Parallel()

	viewerInfo := derivativeViewerInfoForTest(1, 1, 1, 8, 16, 512)
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/tile" {
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write([]byte("SOURCE-PNG"))
			return
		}
		_ = json.NewEncoder(w).Encode(viewerInfo)
	}))
	defer imageSvc.Close()
	root := t.TempDir()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, bus),
		Store:           mem,
		UploadRoot:      root,
		ImageServiceURL: imageSvc.URL,
		DataAgentJobs:   bus,
	})
	fileID := uploadNamedFileForProxyTest(t, router, "repair.bin", testPNGBytes(t, 4, 4))
	record := uploadedResourceRecordForTest(t, mem, fileID)
	writeStrictDerivativeForTest(
		t, root, uploadedSourcePathForTest(t, root, fileID), record, viewerInfo,
		derivativeCapabilities{Slice: true, Thumbnail: true},
		[]byte("committed but tile-incompatible derivative"),
	)

	req := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/tiles/z/0/0/0", nil)
	setProxyOwnerHeaders(req)
	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, req)
	if recorder.Code != http.StatusOK {
		t.Fatalf("tile status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	select {
	case job := <-bus.DataAgentJobs():
		t.Fatalf("intrinsic capability limitation scheduled futile repair: %+v", job)
	default:
	}
}

func TestDerivativeCapabilityContractCorruptionEnqueuesRepair(t *testing.T) {
	root, sourcePath, record, viewerInfo, _ := strictManifestFixture(t)
	record.OriginalName = "source.ome.tif"
	viewerInfo["tile_scheme"] = nil
	viewerInfo["viewer"].(map[string]any)["tile_scheme"] = nil
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(viewerInfo)
	}))
	defer imageSvc.Close()
	pub := &recordingDataAgentJobPublisher{}
	deps := ServerDeps{
		ImageServiceURL: imageSvc.URL,
		DataAgentJobs:   pub,
	}

	if path, _, compatible := deps.compatibleDerivedPyramid(
		context.Background(),
		root,
		record,
		sourcePath,
		nil,
		derivativeUse{capability: "tile"},
	); compatible || path != "" {
		t.Fatalf("capability-corrupt derivative admitted: path=%q compatible=%t", path, compatible)
	}
	if len(pub.jobs) != 1 || pub.jobs[0].Metadata["trigger"] != "repair-incompatible" {
		t.Fatalf("capability corruption repair jobs = %+v", pub.jobs)
	}
}
