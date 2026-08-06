package httpapi

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

// The fixtures below reproduce the property layouts of the two MEASURED reference
// files (backend/contracts/scene3d/CONTRACT.md Appendix A). Rendered at their real vertex
// counts the splat header is byte-identical to the source's 1512-byte header, so
// the stride and data-offset assertions here are ground truth, not self-agreement.
//
//	fused_model1_superpoint.ply           2,068,089 verts, stride 27
//	willaGlobalonlyDrone-deleted_env-1.ply 14,469,103 splats, stride 236, offset 1512
const (
	referencePointCloudVertices = int64(2_068_089)
	referenceSplatVertices      = int64(14_469_103)
	referencePointCloudStride   = 27
	referenceSplatStride        = 236
	referenceSplatDataOffset    = int64(1512)
)

// plySplatHeader writes Postshot's layout: x/y/z, DC colour, restCount f_rest_*
// coefficients, opacity, log scales and a rotation quat — and NO normals, which
// is why the real stride is 236 and not INRIA's 248.
func plySplatHeader(vertexCount int64, restCount int, comment string) string {
	var builder strings.Builder
	builder.WriteString("ply\nformat binary_little_endian 1.0\n")
	if comment != "" {
		builder.WriteString("comment " + comment + "\n")
	}
	fmt.Fprintf(&builder, "element vertex %d\n", vertexCount)
	for _, axis := range []string{"x", "y", "z"} {
		builder.WriteString("property float " + axis + "\n")
	}
	for index := 0; index < 3; index++ {
		fmt.Fprintf(&builder, "property float f_dc_%d\n", index)
	}
	for index := 0; index < restCount; index++ {
		fmt.Fprintf(&builder, "property float f_rest_%d\n", index)
	}
	builder.WriteString("property float opacity\n")
	for index := 0; index < 3; index++ {
		fmt.Fprintf(&builder, "property float scale_%d\n", index)
	}
	for index := 0; index < 4; index++ {
		fmt.Fprintf(&builder, "property float rot_%d\n", index)
	}
	builder.WriteString("end_header\n")
	return builder.String()
}

// plyPointCloudHeader writes the dense-point layout: coordinates, normals and
// uchar RGB — 27 bytes per vertex.
func plyPointCloudHeader(vertexCount int64) string {
	var builder strings.Builder
	builder.WriteString("ply\nformat binary_little_endian 1.0\n")
	fmt.Fprintf(&builder, "element vertex %d\n", vertexCount)
	for _, name := range []string{"x", "y", "z", "nx", "ny", "nz"} {
		builder.WriteString("property float " + name + "\n")
	}
	for _, name := range []string{"red", "green", "blue"} {
		builder.WriteString("property uchar " + name + "\n")
	}
	builder.WriteString("end_header\n")
	return builder.String()
}

// plyFixtureBytes renders a header at a SMALL vertex count and appends exactly
// that many zeroed records, so the file is internally consistent and tiny.
func plyFixtureBytes(header func(count int64) string, stride int, vertexCount int64) []byte {
	body := header(vertexCount)
	payload := make([]byte, int64(stride)*vertexCount)
	return append([]byte(body), payload...)
}

func splatFixtureBytes(vertexCount int64) []byte {
	return plyFixtureBytes(
		func(count int64) string { return plySplatHeader(count, 45, "postshot.anti_aliasing=1") },
		referenceSplatStride,
		vertexCount,
	)
}

func pointCloudFixtureBytes(vertexCount int64) []byte {
	return plyFixtureBytes(plyPointCloudHeader, referencePointCloudStride, vertexCount)
}

func writePlyFixture(t *testing.T, name string, content []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, content, 0o600); err != nil {
		t.Fatalf("write PLY fixture: %v", err)
	}
	return path
}

func TestParsePlyHeaderMatchesMeasuredReferenceLayouts(t *testing.T) {
	t.Parallel()

	splat, ok := parsePlyHeader([]byte(plySplatHeader(referenceSplatVertices, 45, "postshot.anti_aliasing=1")))
	if !ok {
		t.Fatal("parsePlyHeader rejected the reference splat header")
	}
	if splat.species != "splat" {
		t.Fatalf("splat species = %q, want splat", splat.species)
	}
	if splat.strideBytes != referenceSplatStride {
		t.Fatalf("splat stride = %d, want %d (Postshot omits nx/ny/nz)", splat.strideBytes, referenceSplatStride)
	}
	if splat.dataOffset != referenceSplatDataOffset {
		t.Fatalf("splat data offset = %d, want %d", splat.dataOffset, referenceSplatDataOffset)
	}
	if splat.vertexCount != referenceSplatVertices {
		t.Fatalf("splat vertex count = %d, want %d", splat.vertexCount, referenceSplatVertices)
	}
	if splat.propertyCount != 59 {
		t.Fatalf("splat property count = %d, want 59", splat.propertyCount)
	}
	if splat.declaredSHDegree != 3 || splat.restCount != 45 {
		t.Fatalf("splat declared SH = degree %d from %d f_rest_*, want 3 from 45", splat.declaredSHDegree, splat.restCount)
	}
	if splat.writer != "postshot" {
		t.Fatalf("splat writer = %q, want postshot", splat.writer)
	}

	points, ok := parsePlyHeader([]byte(plyPointCloudHeader(referencePointCloudVertices)))
	if !ok {
		t.Fatal("parsePlyHeader rejected the reference point-cloud header")
	}
	if points.species != "pointcloud" {
		t.Fatalf("point-cloud species = %q, want pointcloud", points.species)
	}
	if points.strideBytes != referencePointCloudStride {
		t.Fatalf("point-cloud stride = %d, want %d", points.strideBytes, referencePointCloudStride)
	}
	if points.vertexCount != referencePointCloudVertices {
		t.Fatalf("point-cloud vertex count = %d, want %d", points.vertexCount, referencePointCloudVertices)
	}
	if points.declaredSHDegree != 0 || points.writer != "" {
		t.Fatalf("point cloud = degree %d writer %q, want 0 and no writer", points.declaredSHDegree, points.writer)
	}
}

// A hardcoded layout misreads every field of a file whose writer emits a
// different property set, so stride and offset must move with the header.
func TestParsePlyHeaderDerivesLayoutFromTheHeader(t *testing.T) {
	t.Parallel()

	inria := "ply\nformat binary_little_endian 1.0\nelement vertex 7\n" +
		"property float x\nproperty float y\nproperty float z\n" +
		"property float nx\nproperty float ny\nproperty float nz\n"
	for index := 0; index < 3; index++ {
		inria += fmt.Sprintf("property float f_dc_%d\n", index)
	}
	for index := 0; index < 45; index++ {
		inria += fmt.Sprintf("property float f_rest_%d\n", index)
	}
	inria += "property float opacity\nproperty float scale_0\nproperty float scale_1\nproperty float scale_2\n" +
		"property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\nend_header\n"
	withNormals, ok := parsePlyHeader([]byte(inria))
	if !ok || withNormals.strideBytes != 248 {
		t.Fatalf("INRIA-style stride = %d (ok=%t), want 248", withNormals.strideBytes, ok)
	}

	// Degree 0 (no f_rest_* at all) is a real, shippable file, not an error.
	degreeZero, ok := parsePlyHeader([]byte(plySplatHeader(7, 0, "")))
	if !ok || degreeZero.strideBytes != referenceSplatStride-45*4 || degreeZero.declaredSHDegree != 0 {
		t.Fatalf("degree-0 splat = stride %d degree %d (ok=%t), want stride %d degree 0",
			degreeZero.strideBytes, degreeZero.declaredSHDegree, ok, referenceSplatStride-45*4)
	}

	// Degree 1 and 2 allocations round-trip through the (d+1)^2-1 identity.
	for restCount, wantDegree := range map[int]int{9: 1, 24: 2} {
		info, ok := parsePlyHeader([]byte(plySplatHeader(7, restCount, "")))
		if !ok || info.declaredSHDegree != wantDegree {
			t.Fatalf("%d f_rest_* -> degree %d (ok=%t), want %d", restCount, info.declaredSHDegree, ok, wantDegree)
		}
	}

	// A comment shifts the data offset by exactly its own length.
	base, _ := parsePlyHeader([]byte(plySplatHeader(7, 45, "")))
	commented, _ := parsePlyHeader([]byte(plySplatHeader(7, 45, "postshot.anti_aliasing=1")))
	if delta := commented.dataOffset - base.dataOffset; delta != int64(len("comment postshot.anti_aliasing=1\n")) {
		t.Fatalf("comment shifted the data offset by %d bytes, want %d", delta, len("comment postshot.anti_aliasing=1\n"))
	}

	// CRLF headers parse identically to LF ones.
	crlf, ok := parsePlyHeader([]byte(strings.ReplaceAll(plyPointCloudHeader(7), "\n", "\r\n")))
	if !ok || crlf.strideBytes != referencePointCloudStride {
		t.Fatalf("CRLF header stride = %d (ok=%t), want %d", crlf.strideBytes, ok, referencePointCloudStride)
	}
	if crlf.dataOffset != int64(len(plyPointCloudHeader(7))+strings.Count(plyPointCloudHeader(7), "\n")) {
		t.Fatalf("CRLF data offset = %d, want the byte length of the CRLF header", crlf.dataOffset)
	}
}

func TestParsePlyHeaderRejectsHeadersItCannotDescribeExactly(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		header string
	}{
		{name: "not-a-ply", header: "PK\x03\x04not a ply at all\n"},
		{name: "no-end-header", header: "ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\n"},
		{name: "unknown-scalar-type", header: "ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float16 x\nproperty float y\nproperty float z\nend_header\n"},
		{
			name: "vertex-list-property",
			header: "ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n" +
				"property list uchar int extra\nend_header\n",
		},
		{
			name: "vertex-is-not-the-first-element",
			header: "ply\nformat binary_little_endian 1.0\nelement camera 2\nproperty float focal\n" +
				"element vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n",
		},
		{
			name: "two-vertex-elements",
			header: "ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n" +
				"element vertex 4\nproperty float x\nend_header\n",
		},
		{name: "no-coordinates", header: "ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty uchar red\nend_header\n"},
		{name: "unknown-byte-order", header: "ply\nformat binary_middle_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n"},
		{name: "no-vertex-element", header: "ply\nformat binary_little_endian 1.0\nelement face 3\nproperty list uchar int vertex_indices\nend_header\n"},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if info, ok := parsePlyHeader([]byte(test.header)); ok {
				t.Fatalf("parsePlyHeader accepted %s: %+v", test.name, info)
			}
		})
	}

	// ASCII is a legitimate PLY encoding with no fixed stride; it is classified,
	// and its stride is reported as unknown rather than invented.
	ascii, ok := parsePlyHeader([]byte("ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n"))
	if !ok || ascii.strideBytes != 0 || ascii.species != "pointcloud" {
		t.Fatalf("ascii PLY = stride %d species %q (ok=%t), want stride 0 pointcloud", ascii.strideBytes, ascii.species, ok)
	}
}

func TestPlyPeekBoundsTheReadAndVerifiesThePayloadIsPresent(t *testing.T) {
	t.Parallel()

	complete := writePlyFixture(t, "scene.ply", splatFixtureBytes(4))
	info, ok := plyPeek(complete)
	if !ok || info.strideBytes != referenceSplatStride || info.vertexCount != 4 {
		t.Fatalf("plyPeek(complete) = %+v ok=%t", info, ok)
	}

	full := splatFixtureBytes(4)
	truncated := writePlyFixture(t, "truncated.ply", full[:len(full)-1])
	if _, ok := plyPeek(truncated); ok {
		t.Fatal("plyPeek accepted a truncated splat file")
	}

	// A header that does not end inside the peek window is rejected rather than
	// read further: the sniff must stay O(1) against a multi-gigabyte source.
	oversized := "ply\nformat binary_little_endian 1.0\n" +
		strings.Repeat("comment padding to push end_header past the sniff window\n", 2000) +
		"element vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
	if int64(len(oversized)) <= plyHeaderPeekMaxBytes {
		t.Fatalf("oversized fixture is %d bytes, expected more than the %d-byte window", len(oversized), plyHeaderPeekMaxBytes)
	}
	if _, ok := plyPeek(writePlyFixture(t, "oversized.ply", []byte(oversized))); ok {
		t.Fatal("plyPeek accepted a header longer than its bounded read")
	}

	if _, ok := plyPeek(filepath.Join(t.TempDir(), "missing.ply")); ok {
		t.Fatal("plyPeek accepted a missing file")
	}
}

// TestPlyPeekAgainstRealReferenceFiles validates the sniff against the actual
// measured sources (Appendix A) rather than against fixtures we wrote ourselves.
// Skipped unless the env vars point at them:
//
//	ULTRA_SCENE3D_SPLAT_FIXTURE=<dir>/willaGlobalonlyDrone-deleted_env-1.ply
//	ULTRA_SCENE3D_POINTS_FIXTURE=<dir>/fused_model1_superpoint.ply
func TestPlyPeekAgainstRealReferenceFiles(t *testing.T) {
	t.Parallel()

	if path := os.Getenv("ULTRA_SCENE3D_SPLAT_FIXTURE"); path != "" {
		info, ok := plyPeek(path)
		if !ok {
			t.Fatalf("plyPeek(%q) rejected the reference splat file", path)
		}
		if info.species != "splat" || info.strideBytes != referenceSplatStride ||
			info.dataOffset != referenceSplatDataOffset || info.vertexCount != referenceSplatVertices {
			t.Fatalf("reference splat = %+v, want splat/%d/%d/%d",
				info, referenceSplatStride, referenceSplatDataOffset, referenceSplatVertices)
		}
		if info.declaredSHDegree != 3 || info.writer != "postshot" {
			t.Fatalf("reference splat declared degree %d writer %q, want 3/postshot", info.declaredSHDegree, info.writer)
		}
	} else {
		t.Log("set ULTRA_SCENE3D_SPLAT_FIXTURE to validate against the real splat file")
	}

	if path := os.Getenv("ULTRA_SCENE3D_POINTS_FIXTURE"); path != "" {
		info, ok := plyPeek(path)
		if !ok {
			t.Fatalf("plyPeek(%q) rejected the reference point cloud", path)
		}
		if info.species != "pointcloud" || info.strideBytes != referencePointCloudStride ||
			info.vertexCount != referencePointCloudVertices {
			t.Fatalf("reference point cloud = %+v, want pointcloud/%d/%d",
				info, referencePointCloudStride, referencePointCloudVertices)
		}
		if info.dataOffset != 248 {
			t.Fatalf("reference point-cloud data offset = %d, want 248", info.dataOffset)
		}
	} else {
		t.Log("set ULTRA_SCENE3D_POINTS_FIXTURE to validate against the real point cloud")
	}
}

func TestScene3dNameClassification(t *testing.T) {
	t.Parallel()

	for _, name := range []string{"scene.ply", "SCENE.PLY", "cloud.splat", "drone.spz", "web.ksplat", "packed.sog"} {
		if !isScene3dName(name) {
			t.Fatalf("isScene3dName(%q) = false, want true", name)
		}
	}
	for _, name := range []string{"scan.tif", "volume.nii.gz", "notes.txt", "plywood.txt", "model.ply.gz"} {
		if isScene3dName(name) {
			t.Fatalf("isScene3dName(%q) = true, want false", name)
		}
	}

	// A compact container is classified by extension alone (no cheap header to
	// sniff); a .ply must additionally parse.
	compact := resourceRecord{FileID: "f1", OriginalName: "drone.spz"}
	info, ok := scene3dPeek(compact, filepath.Join(t.TempDir(), "nonexistent.spz"))
	if !ok || info.sceneKind != "splat" || info.hasPly {
		t.Fatalf("scene3dPeek(compact) = %+v ok=%t, want a splat with no PLY facts", info, ok)
	}
	notAPly := writePlyFixture(t, "fake.ply", []byte("this is not a PLY file at all"))
	if resourceIsScene3d(resourceRecord{FileID: "f2", OriginalName: "fake.ply"}, notAPly) {
		t.Fatal("resourceIsScene3d accepted a .ply whose header does not parse")
	}
}

func newScene3dTestRouter(t *testing.T, imageServiceURL string) (http.Handler, *store.MemoryStore, string, *recordingDataAgentJobPublisher) {
	t.Helper()
	mem := store.NewMemoryStore()
	root := t.TempDir()
	publisher := &recordingDataAgentJobPublisher{}
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:           mem,
		UploadRoot:      root,
		ImageServiceURL: imageServiceURL,
		DataAgentJobs:   publisher,
	})
	return router, mem, root, publisher
}

func getAsOwner(t *testing.T, router http.Handler, target string, header func(*http.Request)) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, target, nil)
	setProxyOwnerHeaders(req)
	if header != nil {
		header(req)
	}
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	return rec
}

func decodeScene3dJSON(t *testing.T, rec *httptest.ResponseRecorder) map[string]any {
	t.Helper()
	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode response %q: %v", rec.Body.String(), err)
	}
	return payload
}

// Both viewer entry points must emit the arm: handleGetUploadViewerService (image
// service configured) and handleGetUploadViewer (unconfigured — the production
// path when the sidecar is absent). Missing the second is the classic bug.
func TestScene3dViewerDescriptorFromBothDispatchArms(t *testing.T) {
	var imageServiceCalls atomic.Int64
	imageService := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		imageServiceCalls.Add(1)
		http.Error(w, "a 3D scene must never reach the image service", http.StatusInternalServerError)
	}))
	defer imageService.Close()

	for _, arm := range []struct {
		name            string
		imageServiceURL string
	}{
		{name: "image_service_configured", imageServiceURL: imageService.URL},
		{name: "image_service_unconfigured", imageServiceURL: ""},
	} {
		arm := arm
		t.Run(arm.name, func(t *testing.T) {
			router, _, _, publisher := newScene3dTestRouter(t, arm.imageServiceURL)
			fileID := uploadNamedFileForProxyTest(t, router, "drone.ply", splatFixtureBytes(6))

			rec := getAsOwner(t, router, "/v2/uploads/"+fileID+"/viewer", nil)
			if rec.Code != http.StatusOK {
				t.Fatalf("viewer status = %d body=%s", rec.Code, rec.Body.String())
			}
			payload := decodeScene3dJSON(t, rec)
			if payload["kind"] != "scene3d" || payload["scene_kind"] != "splat" {
				t.Fatalf("viewer kind/scene_kind = %v/%v, want scene3d/splat", payload["kind"], payload["scene_kind"])
			}
			if payload["status"] != "deriving" {
				t.Fatalf("status = %v, want deriving (nothing derived yet)", payload["status"])
			}
			source, sourceOK := payload["source"].(map[string]any)
			if !sourceOK {
				t.Fatalf("descriptor has no source object: %v", payload)
			}
			if source["stride_bytes"] != float64(referenceSplatStride) {
				t.Fatalf("source.stride_bytes = %v, want %d", source["stride_bytes"], referenceSplatStride)
			}
			if source["declared_sh_degree"] != float64(3) || source["vertex_count"] != float64(6) {
				t.Fatalf("source = %v, want declared_sh_degree 3 and vertex_count 6", source)
			}
			if source["writer"] != "postshot" {
				t.Fatalf("source.writer = %v, want postshot", source["writer"])
			}
			// The control plane must not claim a measured degree; only the derive job
			// scans the data (the reference file declares 3 and measures 0).
			if _, present := source["measured_sh_degree"]; present {
				t.Fatal("descriptor reported a measured SH degree the control plane never measured")
			}
			urls, urlsOK := payload["service_urls"].(map[string]any)
			if !urlsOK ||
				urls["manifest"] != "/v2/uploads/"+fileID+"/scene3d/manifest" ||
				urls["chunk"] != "/v2/uploads/"+fileID+"/scene3d/chunk/{index}" ||
				urls["download"] != "/v2/resources/"+fileID+"/download" {
				t.Fatalf("service_urls = %v", payload["service_urls"])
			}
			if limitations, ok := payload["limitations"].([]any); !ok || len(limitations) == 0 {
				t.Fatalf("limitations = %v, want the honesty field to be populated", payload["limitations"])
			}

			// A .ply must never be queued for an imgcnv pyramid transcode.
			sceneJobs := 0
			for _, job := range publisher.jobs {
				if job.JobType == "image.derive_pyramid" {
					t.Fatalf("a 3D scene enqueued a pyramid transcode: %+v", job.Metadata)
				}
				if job.JobType == "scene.derive" {
					sceneJobs++
					if job.Metadata["max_splats_per_chunk"] != scene3dMaxSplatsPerChunk ||
						job.Metadata["tier_count"] != scene3dTierCount ||
						job.Metadata["resource_id"] != fileID {
						t.Fatalf("scene.derive metadata = %+v", job.Metadata)
					}
					if !strings.HasSuffix(fmt.Sprint(job.Metadata["dst_dir"]), fileID+"__scene3d") {
						t.Fatalf("scene.derive dst_dir = %v", job.Metadata["dst_dir"])
					}
				}
			}
			if sceneJobs != 1 {
				t.Fatalf("scene.derive jobs = %d, want exactly 1", sceneJobs)
			}
			if imageServiceCalls.Load() != 0 {
				t.Fatalf("image service calls = %d, want zero for a 3D scene", imageServiceCalls.Load())
			}
		})
	}
}

func TestScene3dPointCloudDescriptorReportsItsOwnSpecies(t *testing.T) {
	router, _, _, _ := newScene3dTestRouter(t, "")
	fileID := uploadNamedFileForProxyTest(t, router, "fused_model1_superpoint.ply", pointCloudFixtureBytes(5))
	payload := decodeScene3dJSON(t, getAsOwner(t, router, "/v2/uploads/"+fileID+"/viewer", nil))
	if payload["scene_kind"] != "pointcloud" {
		t.Fatalf("scene_kind = %v, want pointcloud", payload["scene_kind"])
	}
	source := payload["source"].(map[string]any)
	if source["stride_bytes"] != float64(referencePointCloudStride) {
		t.Fatalf("source.stride_bytes = %v, want %d", source["stride_bytes"], referencePointCloudStride)
	}
	if _, present := source["writer"]; present {
		t.Fatal("descriptor invented a writer for a file whose comments name none")
	}
}

func TestScene3dManifestServesDerivedBytesWithStrongETag(t *testing.T) {
	router, _, root, _ := newScene3dTestRouter(t, "")
	fileID := uploadNamedFileForProxyTest(t, router, "drone.ply", splatFixtureBytes(6))
	manifest := []byte(`{"schema":"ultra.scene3d.v1","scene_kind":"splat"}`)
	if err := os.MkdirAll(derivedScene3dDir(root, fileID), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(scene3dManifestPath(root, fileID), manifest, 0o600); err != nil {
		t.Fatal(err)
	}

	rec := getAsOwner(t, router, "/v2/uploads/"+fileID+"/scene3d/manifest", nil)
	if rec.Code != http.StatusOK || rec.Body.String() != string(manifest) {
		t.Fatalf("manifest status = %d body=%q", rec.Code, rec.Body.String())
	}
	etag := rec.Header().Get("ETag")
	if !strings.HasPrefix(etag, `"`) || strings.HasPrefix(etag, `W/`) {
		t.Fatalf("manifest ETag = %q, want a strong validator", etag)
	}
	if cache := rec.Header().Get("Cache-Control"); !strings.Contains(cache, "private") {
		t.Fatalf("manifest Cache-Control = %q", cache)
	}
	if contentType := rec.Header().Get("Content-Type"); !strings.HasPrefix(contentType, "application/json") {
		t.Fatalf("manifest Content-Type = %q", contentType)
	}

	conditional := getAsOwner(t, router, "/v2/uploads/"+fileID+"/scene3d/manifest", func(req *http.Request) {
		req.Header.Set("If-None-Match", etag)
	})
	if conditional.Code != http.StatusNotModified {
		t.Fatalf("conditional manifest status = %d, want 304", conditional.Code)
	}

	// Once the manifest exists the viewer descriptor reports the scene as ready.
	payload := decodeScene3dJSON(t, getAsOwner(t, router, "/v2/uploads/"+fileID+"/viewer", nil))
	if payload["status"] != "ready" {
		t.Fatalf("viewer status = %v, want ready", payload["status"])
	}
}

func TestScene3dManifestAcceptsMissingDeriveAndHonoursTheFailureMarker(t *testing.T) {
	router, _, root, publisher := newScene3dTestRouter(t, "")
	fileID := uploadNamedFileForProxyTest(t, router, "drone.ply", splatFixtureBytes(6))

	rec := getAsOwner(t, router, "/v2/uploads/"+fileID+"/scene3d/manifest", nil)
	if rec.Code != http.StatusAccepted {
		t.Fatalf("missing-manifest status = %d body=%s, want 202", rec.Code, rec.Body.String())
	}
	if payload := decodeScene3dJSON(t, rec); payload["status"] != "deriving" {
		t.Fatalf("missing-manifest body = %v, want status deriving", payload)
	}
	enqueued := 0
	for _, job := range publisher.jobs {
		if job.JobType == "scene.derive" {
			enqueued++
		}
	}
	if enqueued != 1 {
		t.Fatalf("scene.derive jobs after the first poll = %d, want 1", enqueued)
	}

	// A permanent-failure marker suppresses re-enqueueing and is reported honestly.
	failed := uploadNamedFileForProxyTest(t, router, "broken.ply", splatFixtureBytes(6))
	if err := os.MkdirAll(filepath.Join(root, "derived"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(derivedScene3dFailedMarkerPath(root, failed), []byte("permanent"), 0o600); err != nil {
		t.Fatal(err)
	}
	before := len(publisher.jobs)
	failedRec := getAsOwner(t, router, "/v2/uploads/"+failed+"/scene3d/manifest", nil)
	if failedRec.Code != http.StatusAccepted {
		t.Fatalf("failed-derive status = %d, want 202", failedRec.Code)
	}
	if payload := decodeScene3dJSON(t, failedRec); payload["status"] != "failed" {
		t.Fatalf("failed-derive body = %v, want status failed", payload)
	}
	if len(publisher.jobs) != before {
		t.Fatalf("a fresh .failed marker still enqueued %d job(s)", len(publisher.jobs)-before)
	}
	// The viewer descriptor agrees.
	payload := decodeScene3dJSON(t, getAsOwner(t, router, "/v2/uploads/"+failed+"/viewer", nil))
	if payload["status"] != "failed" {
		t.Fatalf("viewer status = %v, want failed", payload["status"])
	}

	// An expired marker stops suppressing.
	stale := time.Now().Add(-2 * scene3dFailureBackoff)
	if err := os.Chtimes(derivedScene3dFailedMarkerPath(root, failed), stale, stale); err != nil {
		t.Fatal(err)
	}
	if recentScene3dFailure(root, failed, time.Now()) {
		t.Fatal("an expired .failed marker still suppresses re-derivation")
	}
}

func TestScene3dChunkIndexValidationRejectsTraversalAndJunk(t *testing.T) {
	router, _, root, _ := newScene3dTestRouter(t, "")
	fileID := uploadNamedFileForProxyTest(t, router, "drone.ply", splatFixtureBytes(6))
	if err := os.MkdirAll(derivedScene3dDir(root, fileID), 0o755); err != nil {
		t.Fatal(err)
	}
	secret := "SUPER-SECRET-DERIVED-BYTES"
	if err := os.WriteFile(filepath.Join(root, "derived", "secret.txt"), []byte(secret), 0o600); err != nil {
		t.Fatal(err)
	}

	for _, raw := range []string{
		"..", "..%2F..%2Fsecret.txt", "%2Fetc%2Fpasswd", "-1", "+1", "abc", "1e3", "0x10",
		"007", "1000000", "99999999999999999999", "1%20", "1.0",
	} {
		rec := getAsOwner(t, router, "/v2/uploads/"+fileID+"/scene3d/chunk/"+raw, nil)
		// Every one of these reaches the handler and is rejected by the index
		// parser — not merely missed by the router.
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("chunk index %q status = %d, want 400: %q", raw, rec.Code, rec.Body.String())
		}
		if strings.Contains(rec.Body.String(), secret) {
			t.Fatalf("chunk index %q leaked a file outside the derive directory", raw)
		}
	}

	// Syntactically valid but not derived yet -> an honest 404, not a 200 of nothing.
	if rec := getAsOwner(t, router, "/v2/uploads/"+fileID+"/scene3d/chunk/0", nil); rec.Code != http.StatusNotFound {
		t.Fatalf("undelivered chunk status = %d, want 404", rec.Code)
	}

	// Unit-level: "0" is the only accepted spelling of zero.
	if index, err := parseScene3dChunkIndex("0"); err != nil || index != 0 {
		t.Fatalf("parseScene3dChunkIndex(\"0\") = %d, %v", index, err)
	}
	if index, err := parseScene3dChunkIndex(strconv.Itoa(scene3dMaxChunkIndex)); err != nil || index != scene3dMaxChunkIndex {
		t.Fatalf("parseScene3dChunkIndex(max) = %d, %v", index, err)
	}
}

func TestScene3dChunkStreamsRangesWithImmutableCaching(t *testing.T) {
	router, _, root, _ := newScene3dTestRouter(t, "")
	fileID := uploadNamedFileForProxyTest(t, router, "drone.ply", splatFixtureBytes(6))
	if err := os.MkdirAll(derivedScene3dDir(root, fileID), 0o755); err != nil {
		t.Fatal(err)
	}
	chunk := append([]byte("USX1"), make([]byte, 124)...)
	chunkPath := filepath.Join(derivedScene3dDir(root, fileID), fmt.Sprintf(scene3dChunkNameFormat, 0))
	if err := os.WriteFile(chunkPath, chunk, 0o600); err != nil {
		t.Fatal(err)
	}

	rec := getAsOwner(t, router, "/v2/uploads/"+fileID+"/scene3d/chunk/0", nil)
	if rec.Code != http.StatusOK || rec.Body.Len() != len(chunk) {
		t.Fatalf("chunk status = %d length = %d, want 200 and %d bytes", rec.Code, rec.Body.Len(), len(chunk))
	}
	if cache := rec.Header().Get("Cache-Control"); !strings.Contains(cache, "immutable") {
		t.Fatalf("chunk Cache-Control = %q, want an immutable directive", cache)
	}
	etag := rec.Header().Get("ETag")
	if etag == "" || rec.Header().Get("Content-Type") != "application/octet-stream" {
		t.Fatalf("chunk ETag = %q content-type = %q", etag, rec.Header().Get("Content-Type"))
	}

	partial := getAsOwner(t, router, "/v2/uploads/"+fileID+"/scene3d/chunk/0", func(req *http.Request) {
		req.Header.Set("Range", "bytes=0-3")
	})
	if partial.Code != http.StatusPartialContent || partial.Body.String() != "USX1" {
		t.Fatalf("range status = %d body = %q, want 206 USX1", partial.Code, partial.Body.String())
	}

	conditional := getAsOwner(t, router, "/v2/uploads/"+fileID+"/scene3d/chunk/0", func(req *http.Request) {
		req.Header.Set("If-None-Match", etag)
	})
	if conditional.Code != http.StatusNotModified {
		t.Fatalf("conditional chunk status = %d, want 304", conditional.Code)
	}
}

func TestScene3dChunkAdmissionBudgetShedsUnderPressure(t *testing.T) {
	original := scene3dChunkInFlightBudget
	scene3dChunkInFlightBudget = newByteAdmissionBudget(0)
	defer func() { scene3dChunkInFlightBudget = original }()

	router, _, root, _ := newScene3dTestRouter(t, "")
	fileID := uploadNamedFileForProxyTest(t, router, "drone.ply", splatFixtureBytes(6))
	if err := os.MkdirAll(derivedScene3dDir(root, fileID), 0o755); err != nil {
		t.Fatal(err)
	}
	chunkPath := filepath.Join(derivedScene3dDir(root, fileID), fmt.Sprintf(scene3dChunkNameFormat, 0))
	if err := os.WriteFile(chunkPath, make([]byte, 128), 0o600); err != nil {
		t.Fatal(err)
	}
	rec := getAsOwner(t, router, "/v2/uploads/"+fileID+"/scene3d/chunk/0", nil)
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("saturated chunk status = %d, want 503", rec.Code)
	}
	if rec.Header().Get("Retry-After") == "" {
		t.Fatal("saturated chunk response has no Retry-After header")
	}
}

func TestScene3dRoutesRejectNonScenesAndForeignPrincipals(t *testing.T) {
	router, _, _, _ := newScene3dTestRouter(t, "")
	notes := uploadNamedFileForProxyTest(t, router, "notes.txt", []byte("not a scene"))
	for _, route := range []string{"/scene3d/manifest", "/scene3d/chunk/0"} {
		rec := getAsOwner(t, router, "/v2/uploads/"+notes+route, nil)
		if rec.Code != http.StatusUnsupportedMediaType {
			t.Fatalf("%s on a text file = %d, want 415", route, rec.Code)
		}
	}

	scene := uploadNamedFileForProxyTest(t, router, "drone.ply", splatFixtureBytes(6))
	for _, route := range []string{"/scene3d/manifest", "/scene3d/chunk/0"} {
		req := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+scene+route, nil)
		req.Header.Set("X-Ultra-User-Id", "other-researcher")
		req.Header.Set("X-Ultra-Org-Id", "smithsonian")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != http.StatusNotFound {
			t.Fatalf("%s for a foreign principal = %d, want a hidden 404", route, rec.Code)
		}
	}
}
