package httpapi

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

func scene3dFixtureSHA(payload []byte) string {
	return fmt.Sprintf("%x", sha256.Sum256(payload))
}

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

	// The Python worker can address a vertex element after a fixed-width prefix.
	// Admission must compute the same offset instead of rejecting a file the worker
	// can decode.
	prefixedHeader := "ply\nformat binary_little_endian 1.0\nelement camera 2\nproperty float focal\n" +
		"element vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
	prefixed, ok := parsePlyHeader([]byte(prefixedHeader))
	if !ok || prefixed.dataOffset != int64(len(prefixedHeader)+8) {
		t.Fatalf("fixed-prefix data offset = %d (ok=%t), want %d", prefixed.dataOffset, ok, len(prefixedHeader)+8)
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
			name: "variable-width-element-before-vertex",
			header: "ply\nformat binary_little_endian 1.0\nelement camera 2\nproperty list uchar float calibration\n" +
				"element vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n",
		},
		{
			name: "two-vertex-elements",
			header: "ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n" +
				"element vertex 4\nproperty float x\nend_header\n",
		},
		{name: "no-coordinates", header: "ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty uchar red\nend_header\n"},
		{name: "ascii", header: "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n"},
		{
			name: "partial-splat-schema",
			header: "ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n" +
				"property float f_dc_0\nproperty float opacity\nproperty float scale_0\nproperty float rot_0\nend_header\n",
		},
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

	for _, name := range []string{"scene.ply", "SCENE.PLY"} {
		if !isScene3dName(name) {
			t.Fatalf("isScene3dName(%q) = false, want true", name)
		}
	}
	for _, name := range []string{"scan.tif", "volume.nii.gz", "notes.txt", "plywood.txt", "model.ply.gz", "cloud.splat", "drone.spz", "web.ksplat", "packed.sog"} {
		if isScene3dName(name) {
			t.Fatalf("isScene3dName(%q) = true, want false", name)
		}
	}

	// Compact containers are not advertised until their derive path is implemented;
	// extension-only classification previously queued them into the PLY parser.
	if info, ok := scene3dPeek(resourceRecord{FileID: "f1", OriginalName: "drone.spz"}, filepath.Join(t.TempDir(), "nonexistent.spz")); ok {
		t.Fatalf("scene3dPeek(compact) = %+v ok=%t, want unsupported", info, ok)
	}
	notAPly := writePlyFixture(t, "fake.ply", []byte("this is not a PLY file at all"))
	info, ok := scene3dPeek(resourceRecord{FileID: "f2", OriginalName: "fake.ply"}, notAPly)
	if !ok || info.unsupportedReason == "" || scene3dCanDerive(resourceRecord{}, info) {
		t.Fatalf("malformed named PLY = %+v ok=%t, want a recognized unsupported scene", info, ok)
	}
}

// --- COLMAP fixtures ---------------------------------------------------------

// writeColmapModel materializes a reconstruction as a directory tree. Member
// CONTENT is deliberately meaningless: recognition is structural, and a probe
// that needed real records would have to walk a variable-stride images.bin.
func writeColmapModel(t *testing.T, root string, relatives ...string) string {
	t.Helper()
	for _, relative := range relatives {
		path := filepath.Join(root, filepath.FromSlash(relative))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatalf("mkdir %s: %v", relative, err)
		}
		if err := os.WriteFile(path, []byte("colmap"), 0o600); err != nil {
			t.Fatalf("write %s: %v", relative, err)
		}
	}
	return root
}

func newColmapModelDir(t *testing.T, relatives ...string) string {
	t.Helper()
	return writeColmapModel(t, t.TempDir(), relatives...)
}

func colmapZipBytes(t *testing.T, names ...string) []byte {
	t.Helper()
	var buffer bytes.Buffer
	writer := zip.NewWriter(&buffer)
	for _, name := range names {
		entry, err := writer.Create(name)
		if err != nil {
			t.Fatalf("zip create %s: %v", name, err)
		}
		if _, err := entry.Write([]byte("colmap")); err != nil {
			t.Fatalf("zip write %s: %v", name, err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("zip close: %v", err)
	}
	return buffer.Bytes()
}

func writeFixture(t *testing.T, name string, content []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, content, 0o600); err != nil {
		t.Fatalf("write fixture %s: %v", name, err)
	}
	return path
}

// colmapZipBombDeclaredBytes is what the bomb member CLAIMS to hold. Inflating it
// costs this much memory; reading the central directory costs nothing.
const colmapZipBombDeclaredBytes = int64(64 << 20)

// colmapZipBombBytes builds a real model archive whose first member declares 64
// MiB of zeroes (stored in ~64 KB) and whose compressed stream is then SHREDDED.
// Every header and the whole central directory stay intact, so a probe that reads
// only names is unaffected — while any implementation that inflates a member to
// decide what the archive is both burns 64 MiB and fails outright. The corruption
// is what turns "we do not inflate" from a claim into an assertion.
func colmapZipBombBytes(t *testing.T) []byte {
	t.Helper()
	var buffer bytes.Buffer
	writer := zip.NewWriter(&buffer)
	bomb, err := writer.Create("bomb.bin")
	if err != nil {
		t.Fatalf("zip create bomb: %v", err)
	}
	zeros := make([]byte, 1<<20)
	for written := int64(0); written < colmapZipBombDeclaredBytes; written += int64(len(zeros)) {
		if _, err := bomb.Write(zeros); err != nil {
			t.Fatalf("zip write bomb: %v", err)
		}
	}
	for _, name := range []string{"sparse/0/cameras.bin", "sparse/0/images.bin", "sparse/0/points3D.bin"} {
		entry, err := writer.Create(name)
		if err != nil {
			t.Fatalf("zip create %s: %v", name, err)
		}
		if _, err := entry.Write([]byte("colmap")); err != nil {
			t.Fatalf("zip write %s: %v", name, err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("zip close: %v", err)
	}
	raw := buffer.Bytes()
	// The bomb's deflate stream begins a few dozen bytes in and runs for ~64 KB,
	// so this lands squarely inside it and nowhere near a header.
	for offset := 128; offset < 1024 && offset < len(raw); offset++ {
		raw[offset] ^= 0xFF
	}
	return raw
}

func seedColmapDirectoryResource(t *testing.T, mem *store.MemoryStore, root, fileID, name string, relatives ...string) {
	t.Helper()
	// Staged in a subdirectory for the same reason seedTextResource does it: the
	// upload-catalog migration re-catalogs top-level upload-root entries and would
	// otherwise clobber the seeded ownership.
	storageRelative := filepath.Join("staged", fileID+"__"+safeOriginalFilename(name))
	writeColmapModel(t, filepath.Join(root, storageRelative), relatives...)
	if _, err := mem.UpsertResource(context.Background(), domain.UpsertResourceInput{
		ResourceID:   fileID,
		OriginalName: name,
		ContentType:  "application/octet-stream",
		SizeBytes:    4096,
		StoragePath:  storageRelative,
		SourceType:   "upload",
		ResourceKind: "dataset",
		OwnerUserID:  "field-researcher",
		OwnerOrgID:   "smithsonian",
		Status:       "active",
	}); err != nil {
		t.Fatalf("seed COLMAP resource: %v", err)
	}
}

func TestColmapPeekRecognizesModelsAtEveryKnownRoot(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name             string
		members          []string
		wantModelPath    string
		wantRecordFormat string
		wantPoints3D     bool
		wantCameras      string
		wantPoints3DName string
	}{
		{
			name:             "at-the-resource-root",
			members:          []string{"cameras.bin", "images.bin", "points3D.bin"},
			wantRecordFormat: "bin",
			wantPoints3D:     true,
			wantCameras:      "cameras.bin",
			wantPoints3DName: "points3D.bin",
		},
		{
			name:             "sparse-0",
			members:          []string{"sparse/0/cameras.bin", "sparse/0/images.bin", "sparse/0/points3D.bin"},
			wantModelPath:    "sparse/0",
			wantRecordFormat: "bin",
			wantPoints3D:     true,
			wantCameras:      "cameras.bin",
			wantPoints3DName: "points3D.bin",
		},
		{
			name:             "sparse",
			members:          []string{"sparse/cameras.txt", "sparse/images.txt", "sparse/points3D.txt"},
			wantModelPath:    "sparse",
			wantRecordFormat: "txt",
			wantPoints3D:     true,
			wantCameras:      "cameras.txt",
			wantPoints3DName: "points3D.txt",
		},
		{
			name:             "dense-sparse",
			members:          []string{"dense/sparse/cameras.bin", "dense/sparse/images.bin"},
			wantModelPath:    "dense/sparse",
			wantRecordFormat: "bin",
			wantCameras:      "cameras.bin",
		},
		{
			// A pose-only model is a real COLMAP output; it renders as frusta and must
			// be recognized, with the missing geometry stated rather than implied.
			name:             "cameras-and-images-only",
			members:          []string{"cameras.bin", "images.bin"},
			wantRecordFormat: "bin",
			wantCameras:      "cameras.bin",
		},
		{
			name:             "half-converted-model-is-still-a-model",
			members:          []string{"cameras.bin", "images.txt"},
			wantRecordFormat: "mixed",
			wantCameras:      "cameras.bin",
		},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			info, ok := colmapPeek(newColmapModelDir(t, test.members...))
			if !ok {
				t.Fatalf("colmapPeek rejected %v", test.members)
			}
			if info.variant != "directory" || info.modelPath != test.wantModelPath {
				t.Fatalf("variant/modelPath = %q/%q, want directory/%q", info.variant, info.modelPath, test.wantModelPath)
			}
			if info.recordFormat != test.wantRecordFormat || info.camerasName != test.wantCameras {
				t.Fatalf("recordFormat/cameras = %q/%q, want %q/%q",
					info.recordFormat, info.camerasName, test.wantRecordFormat, test.wantCameras)
			}
			if info.hasPoints3D != test.wantPoints3D || info.points3DName != test.wantPoints3DName {
				t.Fatalf("points3D = %t/%q, want %t/%q",
					info.hasPoints3D, info.points3DName, test.wantPoints3D, test.wantPoints3DName)
			}
			if info.hasRigs || info.hasFrames {
				t.Fatalf("legacy model reported rigs=%t frames=%t, want neither", info.hasRigs, info.hasFrames)
			}
		})
	}
}

// Modern COLMAP writes rigs.bin/frames.bin beside the model. A legacy model has
// neither, and their presence must never turn a model into a non-model.
func TestColmapPeekToleratesRigsAndFrames(t *testing.T) {
	t.Parallel()

	modern := newColmapModelDir(t,
		"sparse/0/cameras.bin", "sparse/0/images.bin", "sparse/0/points3D.bin",
		"sparse/0/rigs.bin", "sparse/0/frames.bin",
		"sparse/0/project.ini", "database.db", "images/DSC_0001.jpg",
	)
	info, ok := colmapPeek(modern)
	if !ok {
		t.Fatal("colmapPeek rejected a modern model carrying rigs.bin + frames.bin")
	}
	if !info.hasRigs || !info.hasFrames {
		t.Fatalf("rigs=%t frames=%t, want both reported", info.hasRigs, info.hasFrames)
	}
	if info.modelPath != "sparse/0" || !info.hasPoints3D {
		t.Fatalf("model = %+v, want sparse/0 with points3D", info)
	}
}

func TestColmapPeekRejectsTreesThatOnlyLookLikeModels(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		members []string
	}{
		{
			// The decoy: a file called images.txt buried in an unrelated tree. Probing
			// only the four roots COLMAP actually writes is what refuses this.
			name: "images-txt-in-a-deep-unrelated-tree",
			members: []string{
				"docs/reports/2024/field-notes/images.txt",
				"docs/reports/2024/field-notes/cameras.txt",
				"README.md",
			},
		},
		{name: "images-without-cameras", members: []string{"sparse/0/images.bin", "sparse/0/points3D.bin"}},
		{name: "cameras-without-images", members: []string{"cameras.bin", "points3D.bin"}},
		{name: "points3d-alone", members: []string{"sparse/0/points3D.bin"}},
		{
			// Split across two roots: neither root is a model, and the pair must not be
			// stitched into one.
			name:    "cameras-and-images-in-different-roots",
			members: []string{"sparse/cameras.bin", "dense/sparse/images.bin"},
		},
		{name: "one-level-too-deep", members: []string{"sparse/0/1/cameras.bin", "sparse/0/1/images.bin"}},
		{name: "empty-directory"},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if info, ok := colmapPeek(newColmapModelDir(t, test.members...)); ok {
				t.Fatalf("colmapPeek accepted %s: %+v", test.name, info)
			}
		})
	}

	// A directory whose model members are directories, not files, is not a model.
	nested := newColmapModelDir(t, "cameras.bin/inner", "images.bin/inner")
	if _, ok := colmapPeek(nested); ok {
		t.Fatal("colmapPeek accepted directories named cameras.bin / images.bin")
	}

	// Non-archive regular files and missing paths stop at the magic/stat check.
	if _, ok := colmapPeek(writeFixture(t, "scene.ply", splatFixtureBytes(4))); ok {
		t.Fatal("colmapPeek accepted a PLY file")
	}
	if _, ok := colmapPeek(writeFixture(t, "tiny.bin", []byte("PK"))); ok {
		t.Fatal("colmapPeek accepted a 2-byte file")
	}
	if _, ok := colmapPeek(filepath.Join(t.TempDir(), "missing")); ok {
		t.Fatal("colmapPeek accepted a missing path")
	}
}

func TestColmapZipPeekReadsTheCentralDirectoryOnly(t *testing.T) {
	t.Parallel()

	// A model at any depth inside the archive is found; the members must share a
	// directory, and exactly one model must exist before it can be derived.
	nested := writeFixture(t, "reconstruction.zip", colmapZipBytes(t,
		"office-scan/sparse/0/cameras.bin",
		"office-scan/sparse/0/images.bin",
		"office-scan/sparse/0/points3D.bin",
		"office-scan/database.db",
		"office-scan/images/DSC_0001.jpg",
	))
	info, ok := colmapPeek(nested)
	if !ok {
		t.Fatal("colmapPeek rejected a zipped model nested under a wrapper folder")
	}
	if info.variant != "zip" || info.modelPath != "office-scan/sparse/0" {
		t.Fatalf("variant/modelPath = %q/%q, want zip/office-scan/sparse/0", info.variant, info.modelPath)
	}
	if info.modelCount != 1 || len(info.modelPaths) != 1 || info.modelPaths[0] != info.modelPath {
		t.Fatalf("unique model inventory = count %d paths %v", info.modelCount, info.modelPaths)
	}
	if info.recordFormat != "bin" || !info.hasPoints3D || info.points3DName != "points3D.bin" {
		t.Fatalf("zip model = %+v, want bin records with points3D.bin", info)
	}

	atRoot := writeFixture(t, "model.zip", colmapZipBytes(t, "cameras.txt", "images.txt"))
	rootInfo, ok := colmapPeek(atRoot)
	if !ok || rootInfo.modelPath != "" || rootInfo.recordFormat != "txt" || rootInfo.hasPoints3D {
		t.Fatalf("archive-root model = %+v (ok=%t), want modelPath \"\" txt without points3D", rootInfo, ok)
	}

	ambiguous := writeFixture(t, "two-models.zip", colmapZipBytes(t,
		"deep/wrapper/sparse/0/cameras.bin", "deep/wrapper/sparse/0/images.bin",
		"sparse/cameras.bin", "sparse/images.bin",
	))
	if info, ok := colmapPeek(ambiguous); !ok || info.modelCount != 2 || info.modelPath != "" ||
		strings.Join(info.modelPaths, ",") != "sparse,deep/wrapper/sparse/0" {
		t.Fatalf("two-model archive = %+v (ok=%t), want an explicit two-model ambiguity", info, ok)
	}

	// Members split across directories are not a model, however suggestive the names.
	split := writeFixture(t, "split.zip", colmapZipBytes(t, "a/cameras.bin", "b/images.bin", "c/points3D.bin"))
	if info, ok := colmapPeek(split); ok {
		t.Fatalf("colmapPeek stitched a model out of members in different archive directories: %+v", info)
	}

	// A traversal member never becomes a model path handed to the derive job.
	traversal := writeFixture(t, "traversal.zip", colmapZipBytes(t, "../escape/cameras.bin", "../escape/images.bin"))
	if info, ok := colmapPeek(traversal); ok {
		t.Fatalf("colmapPeek accepted a traversal model path: %+v", info)
	}
}

// The zip-bomb guard: recognition must cost the central directory and nothing
// else. The archive's first member declares 64 MiB and its compressed stream is
// deliberately corrupt, so an implementation that inflates anything fails here.
func TestColmapZipPeekNeverInflatesAMember(t *testing.T) {
	t.Parallel()

	raw := colmapZipBombBytes(t)
	if int64(len(raw)) > (1 << 20) {
		t.Fatalf("bomb archive is %d bytes; the fixture must stay far smaller than the %d bytes it declares",
			len(raw), colmapZipBombDeclaredBytes)
	}
	path := writeFixture(t, "bomb.zip", raw)

	started := time.Now()
	info, ok := colmapPeek(path)
	elapsed := time.Since(started)
	if !ok {
		t.Fatal("colmapPeek rejected a valid model because of an unrelated bomb member")
	}
	if info.variant != "zip" || info.modelPath != "sparse/0" {
		t.Fatalf("bomb archive model = %+v, want the zip model at sparse/0", info)
	}
	if elapsed > 2*time.Second {
		t.Fatalf("colmapPeek took %s on a %d-byte archive; it is not reading names only", elapsed, len(raw))
	}

	// Proof that the assertion above has teeth: the bomb member genuinely cannot be
	// inflated, and its declared size is what an inflating probe would have paid.
	reader, err := zip.OpenReader(path)
	if err != nil {
		t.Fatalf("open bomb archive: %v", err)
	}
	defer func() { _ = reader.Close() }()
	var bomb *zip.File
	for _, entry := range reader.File {
		if entry.Name == "bomb.bin" {
			bomb = entry
		}
	}
	if bomb == nil {
		t.Fatal("bomb.bin is missing from the fixture")
	}
	if int64(bomb.UncompressedSize64) != colmapZipBombDeclaredBytes {
		t.Fatalf("bomb declares %d bytes, want %d", bomb.UncompressedSize64, colmapZipBombDeclaredBytes)
	}
	stream, err := bomb.Open()
	if err == nil {
		_, err = io.Copy(io.Discard, stream)
		_ = stream.Close()
	}
	if err == nil {
		t.Fatal("the bomb member inflated cleanly, so this test cannot prove colmapPeek avoided inflating it")
	}
}

// The archive probe refuses an archive with a hostile member count BEFORE
// archive/zip parses the central directory: that parse is eager and this code
// runs on every viewer, manifest and chunk request.
func TestColmapZipPeekRefusesArchivesPastItsBoundedProbe(t *testing.T) {
	t.Parallel()

	names := []string{"cameras.bin", "images.bin", "points3D.bin"}
	for index := 0; index <= colmapZipMaxEntries; index++ {
		names = append(names, fmt.Sprintf("images/frame_%06d.jpg", index))
	}
	oversized := writeFixture(t, "oversized.zip", colmapZipBytes(t, names...))
	if info, ok := colmapPeek(oversized); ok {
		t.Fatalf("colmapPeek parsed an archive with %d entries: %+v", len(names), info)
	}
	// The same model in a small archive is recognized, so the refusal above is the
	// bound doing its job and not a broken probe.
	if _, ok := colmapPeek(writeFixture(t, "small.zip", colmapZipBytes(t, "cameras.bin", "images.bin", "points3D.bin"))); !ok {
		t.Fatal("colmapPeek rejected the same model in a small archive")
	}
}

func TestResourceIsScene3dCoversPlyAndColmap(t *testing.T) {
	t.Parallel()

	ply := writePlyFixture(t, "drone.ply", splatFixtureBytes(4))
	if !resourceIsScene3d(resourceRecord{FileID: "f-ply", OriginalName: "drone.ply"}, ply) {
		t.Fatal("resourceIsScene3d rejected a PLY splat")
	}
	model := newColmapModelDir(t, "sparse/0/cameras.bin", "sparse/0/images.bin", "sparse/0/points3D.bin")
	if !resourceIsScene3d(resourceRecord{FileID: "f-colmap", OriginalName: "office-scan"}, model) {
		t.Fatal("resourceIsScene3d rejected a COLMAP directory model")
	}
	archive := writeFixture(t, "office.zip", colmapZipBytes(t, "sparse/0/cameras.bin", "sparse/0/images.bin"))
	if !resourceIsScene3d(resourceRecord{FileID: "f-zip", OriginalName: "office.zip"}, archive) {
		t.Fatal("resourceIsScene3d rejected a zipped COLMAP model")
	}
	decoy := newColmapModelDir(t, "docs/reports/field-notes/images.txt")
	if resourceIsScene3d(resourceRecord{FileID: "f-decoy", OriginalName: "docs"}, decoy) {
		t.Fatal("resourceIsScene3d accepted a tree that merely contains an images.txt")
	}

	// scene3dPeek reports COLMAP's own kind, not a splat/pointcloud guess.
	info, ok := scene3dPeek(resourceRecord{FileID: "f-colmap", OriginalName: "office-scan"}, model)
	if !ok || info.format != "colmap" || info.sceneKind != "colmap" || !info.hasColmap || info.hasPly {
		t.Fatalf("scene3dPeek(colmap) = %+v ok=%t", info, ok)
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
			fixture := splatFixtureBytes(6)
			sourceSHA256 := scene3dFixtureSHA(fixture)
			fileID := uploadNamedFileForProxyTest(t, router, "drone.ply", fixture)

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
				urls["lod"] != "/v2/uploads/"+fileID+"/scene3d/lod/{artifact}" ||
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
						job.Metadata["preview_splats"] != scene3dPreviewSplats ||
						job.Metadata["preview_points"] != scene3dPreviewPoints ||
						job.Metadata["splat_delivery"] != "spark-rad-v1" ||
						job.Metadata["source_sha256"] != sourceSHA256 ||
						job.Metadata["source_size_bytes"] != int64(len(fixture)) ||
						job.Metadata["resource_id"] != fileID {
						t.Fatalf("scene.derive metadata = %+v", job.Metadata)
					}
					if !strings.HasSuffix(
						fmt.Sprint(job.Metadata["dst_dir"]),
						fileID+"__scene3d.v4.sha256-"+sourceSHA256,
					) {
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

func TestScene3dViewerRejectsUnsupportedPlyBeforeImageOrWorkerDispatch(t *testing.T) {
	t.Parallel()

	var imageServiceCalls atomic.Int64
	imageService := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		imageServiceCalls.Add(1)
		http.Error(w, "unsupported PLY must not reach the image service", http.StatusInternalServerError)
	}))
	defer imageService.Close()
	router, _, _, publisher := newScene3dTestRouter(t, imageService.URL)
	ascii := []byte("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n")
	fileID := uploadNamedFileForProxyTest(t, router, "ascii-cloud.ply", ascii)

	rec := getAsOwner(t, router, "/v2/uploads/"+fileID+"/viewer", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("viewer status = %d body=%s", rec.Code, rec.Body.String())
	}
	payload := decodeScene3dJSON(t, rec)
	if payload["kind"] != "scene3d" || payload["status"] != "failed" || payload["decodable"] != false {
		t.Fatalf("unsupported PLY descriptor = %v", payload)
	}
	message := fmt.Sprint(payload["message"])
	if !strings.Contains(message, "ASCII PLY") || !strings.Contains(message, "binary") {
		t.Fatalf("unsupported PLY message = %q", message)
	}
	if len(publisher.jobs) != 0 {
		t.Fatalf("unsupported PLY queued jobs: %+v", publisher.jobs)
	}
	if imageServiceCalls.Load() != 0 {
		t.Fatalf("image service calls = %d, want zero", imageServiceCalls.Load())
	}
}

// A COLMAP DIRECTORY is recognized from both dispatch arms so it never reaches
// libbioimage or imgcnv, but it is not derivable: a mutable directory has no
// cataloged byte stream the worker can bind to one immutable source digest. The
// descriptor tells the scientist to upload a zip rather than pretending a job is
// running forever.
func TestScene3dColmapDirectoryDescriptorFromBothDispatchArms(t *testing.T) {
	var imageServiceCalls atomic.Int64
	imageService := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		imageServiceCalls.Add(1)
		http.Error(w, "a COLMAP model must never reach the image service", http.StatusInternalServerError)
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
			router, mem, root, publisher := newScene3dTestRouter(t, arm.imageServiceURL)
			fileID := "file_colmap_" + arm.name
			seedColmapDirectoryResource(t, mem, root, fileID, "office-scan",
				"sparse/0/cameras.bin", "sparse/0/images.bin", "sparse/0/points3D.bin",
				"sparse/0/rigs.bin", "sparse/0/frames.bin", "database.db",
			)

			rec := getAsOwner(t, router, "/v2/uploads/"+fileID+"/viewer", nil)
			if rec.Code != http.StatusOK {
				t.Fatalf("viewer status = %d body=%s", rec.Code, rec.Body.String())
			}
			payload := decodeScene3dJSON(t, rec)
			if payload["kind"] != "scene3d" || payload["scene_kind"] != "colmap" || payload["format"] != "colmap" {
				t.Fatalf("kind/scene_kind/format = %v/%v/%v, want scene3d/colmap/colmap",
					payload["kind"], payload["scene_kind"], payload["format"])
			}
			if payload["status"] != "failed" || payload["decodable"] != false {
				t.Fatalf("status/decodable = %v/%v, want failed/false", payload["status"], payload["decodable"])
			}
			source, sourceOK := payload["source"].(map[string]any)
			if !sourceOK {
				t.Fatalf("descriptor has no source object: %v", payload)
			}
			if source["variant"] != "directory" || source["model_path"] != "sparse/0" || source["record_format"] != "bin" {
				t.Fatalf("source layout = %v, want a directory model at sparse/0 with bin records", source)
			}
			if source["cameras_file"] != "cameras.bin" || source["images_file"] != "images.bin" ||
				source["has_points3d"] != true || source["points3d_file"] != "points3D.bin" {
				t.Fatalf("source members = %v", source)
			}
			if source["has_rigs"] != true || source["has_frames"] != true {
				t.Fatalf("source rigs/frames = %v/%v, want both true", source["has_rigs"], source["has_frames"])
			}
			// Counts require walking a variable-stride images.bin; the control plane
			// must not claim numbers it never measured.
			for _, invented := range []string{"vertex_count", "camera_count", "point_count", "measured_sh_degree"} {
				if _, present := source[invented]; present {
					t.Fatalf("descriptor reported %q, which this path never measures: %v", invented, source)
				}
			}
			// service_urls are advertised exactly as they are for the PLY path.
			urls, urlsOK := payload["service_urls"].(map[string]any)
			if !urlsOK ||
				urls["manifest"] != "/v2/uploads/"+fileID+"/scene3d/manifest" ||
				urls["chunk"] != "/v2/uploads/"+fileID+"/scene3d/chunk/{index}" ||
				urls["lod"] != "/v2/uploads/"+fileID+"/scene3d/lod/{artifact}" ||
				urls["download"] != "/v2/resources/"+fileID+"/download" {
				t.Fatalf("service_urls = %v", payload["service_urls"])
			}
			limitations := scene3dLimitationsFromPayload(t, payload)
			if !containsSubstring(limitations, "recognized from its layout alone") ||
				!containsSubstring(limitations, "archive this model as a zip") ||
				!containsSubstring(limitations, "2D feature observations are skipped") ||
				!containsSubstring(limitations, "source world frame") {
				t.Fatalf("limitations = %v", limitations)
			}
			if containsSubstring(limitations, "classified by file extension alone") {
				t.Fatalf("a structurally-recognized model claimed extension-only classification: %v", limitations)
			}

			for _, job := range publisher.jobs {
				if job.JobType == "image.derive_pyramid" {
					t.Fatalf("a COLMAP model enqueued a pyramid transcode: %+v", job.Metadata)
				}
				if job.JobType == "scene.derive" {
					t.Fatalf("a mutable COLMAP directory enqueued a source-bound derive: %+v", job.Metadata)
				}
			}
			if imageServiceCalls.Load() != 0 {
				t.Fatalf("image service calls = %d, want zero for a COLMAP model", imageServiceCalls.Load())
			}
			// The record itself is not pyramid-eligible either, so the upload-time
			// prewarm cannot queue a transcode behind the viewer's back.
			record := resourceRecord{FileID: fileID, OriginalName: "office-scan", ContentType: "application/octet-stream", SizeBytes: 4096}
			if shouldDerivePyramid(record) {
				t.Fatal("shouldDerivePyramid accepted a COLMAP model record")
			}
		})
	}
}

func TestScene3dColmapArchiveRefusesAmbiguousModelSelection(t *testing.T) {
	t.Parallel()

	router, _, _, publisher := newScene3dTestRouter(t, "")
	raw := colmapZipBytes(t,
		"sparse/0/cameras.bin", "sparse/0/images.bin", "sparse/0/points3D.bin",
		"sparse/1/cameras.bin", "sparse/1/images.bin", "sparse/1/points3D.bin",
	)
	fileID := uploadNamedFileForProxyTest(t, router, "reconstruction.zip", raw)

	rec := getAsOwner(t, router, "/v2/uploads/"+fileID+"/viewer", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("viewer status = %d body=%s", rec.Code, rec.Body.String())
	}
	payload := decodeScene3dJSON(t, rec)
	if payload["kind"] != "scene3d" || payload["scene_kind"] != "colmap" ||
		payload["status"] != "failed" || payload["decodable"] != false {
		t.Fatalf("ambiguous descriptor = %v", payload)
	}
	source, ok := payload["source"].(map[string]any)
	if !ok || source["model_count"] != float64(2) {
		t.Fatalf("ambiguous source inventory = %v", payload["source"])
	}
	if _, selected := source["model_path"]; selected {
		t.Fatalf("ambiguous source selected a model path: %v", source)
	}
	if !strings.Contains(fmt.Sprint(payload["message"]), "multiple COLMAP models") {
		t.Fatalf("ambiguous message = %v", payload["message"])
	}
	if !containsSubstring(scene3dLimitationsFromPayload(t, payload), "does not choose one") {
		t.Fatalf("ambiguous limitations = %v", payload["limitations"])
	}
	for _, queued := range publisher.jobs {
		if queued.JobType == "scene.derive" || queued.JobType == "image.derive_pyramid" {
			t.Fatalf("ambiguous COLMAP archive queued %q: %+v", queued.JobType, queued.Metadata)
		}
	}

	manifest := getAsOwner(t, router, "/v2/uploads/"+fileID+"/scene3d/manifest", nil)
	if manifest.Code != http.StatusAccepted {
		t.Fatalf("manifest status = %d body=%s", manifest.Code, manifest.Body.String())
	}
	manifestPayload := decodeScene3dJSON(t, manifest)
	if manifestPayload["status"] != "failed" ||
		!strings.Contains(fmt.Sprint(manifestPayload["error"]), "multiple COLMAP models") {
		t.Fatalf("manifest ambiguity response = %v", manifestPayload)
	}
}

// A pose-only model states the missing geometry instead of rendering an empty
// canvas, and the scene3d routes accept it like any other scene.
func TestScene3dColmapWithoutPoints3DIsHonestAndServed(t *testing.T) {
	router, mem, root, _ := newScene3dTestRouter(t, "")
	fileID := "file_colmap_poses"
	seedColmapDirectoryResource(t, mem, root, fileID, "poses-only", "cameras.txt", "images.txt")

	payload := decodeScene3dJSON(t, getAsOwner(t, router, "/v2/uploads/"+fileID+"/viewer", nil))
	source := payload["source"].(map[string]any)
	if source["has_points3d"] != false || source["record_format"] != "txt" || source["model_path"] != "" {
		t.Fatalf("source = %v, want a txt model at the root with no points3D", source)
	}
	if _, present := source["points3d_file"]; present {
		t.Fatalf("descriptor named a points3D file that does not exist: %v", source)
	}
	limitations := scene3dLimitationsFromPayload(t, payload)
	if !containsSubstring(limitations, "no points3D file") {
		t.Fatalf("limitations = %v, want the missing geometry stated", limitations)
	}
	if message, _ := payload["message"].(string); !strings.Contains(message, "COLMAP reconstruction") {
		t.Fatalf("message = %q, want it to name a COLMAP reconstruction", message)
	}

	// The derived-stream route recognizes the resource but reports that preparation
	// is unavailable, rather than polling forever for a job that cannot be source-bound.
	rec := getAsOwner(t, router, "/v2/uploads/"+fileID+"/scene3d/manifest", nil)
	if rec.Code != http.StatusAccepted {
		t.Fatalf("manifest status = %d body=%s, want 202", rec.Code, rec.Body.String())
	}
	if response := decodeScene3dJSON(t, rec); response["status"] != "failed" {
		t.Fatalf("manifest body = %v, want status failed", response)
	}
	if chunk := getAsOwner(t, router, "/v2/uploads/"+fileID+"/scene3d/chunk/0", nil); chunk.Code != http.StatusNotFound {
		t.Fatalf("undelivered chunk status = %d, want 404", chunk.Code)
	}
}

// A zipped model reports the archive variant and says plainly that nothing inside
// it has been decompressed.
func TestScene3dColmapZipDescriptorReportsTheArchiveVariant(t *testing.T) {
	router, mem, root, publisher := newScene3dTestRouter(t, "")
	fileID := "file_colmap_zip"
	storageRelative := filepath.Join("staged", fileID+"__office.zip")
	archive := filepath.Join(root, storageRelative)
	if err := os.MkdirAll(filepath.Dir(archive), 0o755); err != nil {
		t.Fatal(err)
	}
	raw := colmapZipBytes(t, "office/sparse/0/cameras.bin", "office/sparse/0/images.bin", "office/sparse/0/points3D.bin")
	if err := os.WriteFile(archive, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := mem.UpsertResource(context.Background(), domain.UpsertResourceInput{
		ResourceID: fileID, OriginalName: "office.zip", ContentType: "application/zip",
		SizeBytes: int64(len(raw)), SHA256: scene3dFixtureSHA(raw), StoragePath: storageRelative, SourceType: "upload",
		ResourceKind: "dataset", OwnerUserID: "field-researcher", OwnerOrgID: "smithsonian", Status: "active",
	}); err != nil {
		t.Fatal(err)
	}

	payload := decodeScene3dJSON(t, getAsOwner(t, router, "/v2/uploads/"+fileID+"/viewer", nil))
	if payload["scene_kind"] != "colmap" {
		t.Fatalf("scene_kind = %v, want colmap", payload["scene_kind"])
	}
	if payload["status"] != "deriving" {
		t.Fatalf("status = %v, want deriving", payload["status"])
	}
	source := payload["source"].(map[string]any)
	if source["variant"] != "zip" || source["model_path"] != "office/sparse/0" {
		t.Fatalf("source = %v, want a zip model at office/sparse/0", source)
	}
	if source["bytes"] != float64(len(raw)) {
		t.Fatalf("source.bytes = %v, want the archive's size on disk (%d)", source["bytes"], len(raw))
	}
	if limitations := scene3dLimitationsFromPayload(t, payload); !containsSubstring(limitations, "central directory") {
		t.Fatalf("limitations = %v, want the archive statement", limitations)
	}
	jobs := 0
	for _, job := range publisher.jobs {
		if job.JobType == "scene.derive" {
			jobs++
			if job.Metadata["source_sha256"] != scene3dFixtureSHA(raw) ||
				job.Metadata["source_size_bytes"] != int64(len(raw)) {
				t.Fatalf("scene.derive source identity = %+v", job.Metadata)
			}
		}
	}
	if jobs != 1 {
		t.Fatalf("scene.derive jobs = %d, want exactly 1", jobs)
	}
}

func scene3dLimitationsFromPayload(t *testing.T, payload map[string]any) []string {
	t.Helper()
	raw, ok := payload["limitations"].([]any)
	if !ok || len(raw) == 0 {
		t.Fatalf("limitations = %v, want the honesty field to be populated", payload["limitations"])
	}
	limitations := make([]string, 0, len(raw))
	for _, item := range raw {
		text, isText := item.(string)
		if !isText {
			t.Fatalf("limitation %v is not a sentence", item)
		}
		limitations = append(limitations, text)
	}
	return limitations
}

func containsSubstring(values []string, needle string) bool {
	for _, value := range values {
		if strings.Contains(value, needle) {
			return true
		}
	}
	return false
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
	fixture := splatFixtureBytes(6)
	sourceSHA256 := scene3dFixtureSHA(fixture)
	fileID := uploadNamedFileForProxyTest(t, router, "drone.ply", fixture)
	manifest := []byte(`{"schema":"ultra.scene3d.v1","scene_kind":"splat"}`)
	if err := os.MkdirAll(derivedScene3dDir(root, fileID, sourceSHA256), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(scene3dManifestPath(root, fileID, sourceSHA256), manifest, 0o600); err != nil {
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
	fixture := splatFixtureBytes(6)
	fileID := uploadNamedFileForProxyTest(t, router, "drone.ply", fixture)

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
	failedFixture := splatFixtureBytes(7)
	failedSHA256 := scene3dFixtureSHA(failedFixture)
	failed := uploadNamedFileForProxyTest(t, router, "broken.ply", failedFixture)
	if err := os.MkdirAll(filepath.Join(root, "derived"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(derivedScene3dFailedMarkerPath(root, failed, failedSHA256), []byte("permanent"), 0o600); err != nil {
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
	if err := os.Chtimes(derivedScene3dFailedMarkerPath(root, failed, failedSHA256), stale, stale); err != nil {
		t.Fatal(err)
	}
	if recentScene3dFailure(root, failed, failedSHA256, time.Now()) {
		t.Fatal("an expired .failed marker still suppresses re-derivation")
	}
}

func TestScene3dChunkIndexValidationRejectsTraversalAndJunk(t *testing.T) {
	router, _, root, _ := newScene3dTestRouter(t, "")
	fixture := splatFixtureBytes(6)
	sourceSHA256 := scene3dFixtureSHA(fixture)
	fileID := uploadNamedFileForProxyTest(t, router, "drone.ply", fixture)
	if err := os.MkdirAll(derivedScene3dDir(root, fileID, sourceSHA256), 0o755); err != nil {
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

func TestScene3dChunkStreamsRangesWithRevalidatedCaching(t *testing.T) {
	router, _, root, _ := newScene3dTestRouter(t, "")
	fixture := splatFixtureBytes(6)
	sourceSHA256 := scene3dFixtureSHA(fixture)
	fileID := uploadNamedFileForProxyTest(t, router, "drone.ply", fixture)
	directory := derivedScene3dDir(root, fileID, sourceSHA256)
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	chunk := append([]byte("USX1"), make([]byte, 124)...)
	chunkPath := filepath.Join(directory, fmt.Sprintf(scene3dChunkNameFormat, 0))
	if err := os.WriteFile(chunkPath, chunk, 0o600); err != nil {
		t.Fatal(err)
	}

	rec := getAsOwner(t, router, "/v2/uploads/"+fileID+"/scene3d/chunk/0", nil)
	if rec.Code != http.StatusOK || rec.Body.Len() != len(chunk) {
		t.Fatalf("chunk status = %d length = %d, want 200 and %d bytes", rec.Code, rec.Body.Len(), len(chunk))
	}
	if cache := rec.Header().Get("Cache-Control"); !strings.Contains(cache, "private") || strings.Contains(cache, "immutable") {
		t.Fatalf("chunk Cache-Control = %q, want a private revalidated policy", cache)
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

func TestScene3dLodArtifactValidationAndRangeDelivery(t *testing.T) {
	router, _, root, _ := newScene3dTestRouter(t, "")
	fixture := splatFixtureBytes(6)
	sourceSHA256 := scene3dFixtureSHA(fixture)
	fileID := uploadNamedFileForProxyTest(t, router, "drone.ply", fixture)
	directory := derivedScene3dDir(root, fileID, sourceSHA256)
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	header := []byte("RAD-HEADER-BYTES")
	page := []byte("RAD-PAGE-ZERO")
	if err := os.WriteFile(filepath.Join(directory, scene3dLodHeaderName), header, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "scene-lod-0.radc"), page, 0o600); err != nil {
		t.Fatal(err)
	}

	rec := getAsOwner(t, router, "/v2/uploads/"+fileID+"/scene3d/lod/scene-lod.rad", nil)
	if rec.Code != http.StatusOK || !bytes.Equal(rec.Body.Bytes(), header) {
		t.Fatalf("RAD header status = %d body=%q", rec.Code, rec.Body.String())
	}
	if rec.Header().Get("Content-Type") != "application/octet-stream" || rec.Header().Get("ETag") == "" {
		t.Fatalf("RAD headers content-type=%q etag=%q", rec.Header().Get("Content-Type"), rec.Header().Get("ETag"))
	}

	partial := getAsOwner(t, router, "/v2/uploads/"+fileID+"/scene3d/lod/scene-lod-0.radc", func(req *http.Request) {
		req.Header.Set("Range", "bytes=0-2")
	})
	if partial.Code != http.StatusPartialContent || partial.Body.String() != "RAD" {
		t.Fatalf("RAD page range status = %d body=%q", partial.Code, partial.Body.String())
	}

	for _, name := range []string{
		"scene-lod-00.radc", "scene-lod--1.radc", "scene-lod-1.bin", "manifest.json", "..%2Fsecret.txt",
	} {
		invalid := getAsOwner(t, router, "/v2/uploads/"+fileID+"/scene3d/lod/"+name, nil)
		if invalid.Code != http.StatusBadRequest {
			t.Fatalf("RAD artifact %q status = %d, want 400", name, invalid.Code)
		}
	}
}

func TestScene3dChunkAdmissionBudgetShedsUnderPressure(t *testing.T) {
	original := scene3dChunkInFlightBudget
	scene3dChunkInFlightBudget = newByteAdmissionBudget(0)
	defer func() { scene3dChunkInFlightBudget = original }()

	router, _, root, _ := newScene3dTestRouter(t, "")
	fixture := splatFixtureBytes(6)
	sourceSHA256 := scene3dFixtureSHA(fixture)
	fileID := uploadNamedFileForProxyTest(t, router, "drone.ply", fixture)
	directory := derivedScene3dDir(root, fileID, sourceSHA256)
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	chunkPath := filepath.Join(directory, fmt.Sprintf(scene3dChunkNameFormat, 0))
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
	for _, route := range []string{"/scene3d/manifest", "/scene3d/chunk/0", "/scene3d/lod/scene-lod.rad"} {
		rec := getAsOwner(t, router, "/v2/uploads/"+notes+route, nil)
		if rec.Code != http.StatusUnsupportedMediaType {
			t.Fatalf("%s on a text file = %d, want 415", route, rec.Code)
		}
	}

	scene := uploadNamedFileForProxyTest(t, router, "drone.ply", splatFixtureBytes(6))
	for _, route := range []string{"/scene3d/manifest", "/scene3d/chunk/0", "/scene3d/lod/scene-lod.rad"} {
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
