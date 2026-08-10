package httpapi

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"net/url"
	"os"
	pathpkg "path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/go-chi/chi/v5"
)

// scene3d: Gaussian-splat, point-cloud and COLMAP scenes (kind:"scene3d").
//
// A splat/point .ply is not an image — it has no pixel geometry, so the
// libbioimage ladder can only 415 it, and an imgcnv transcode would burn the
// engine for nothing. It is served instead from a DERIVED, chunked stream that
// the imaging worker builds once (see backend/contracts/scene3d/CONTRACT.md §7):
//
//	{root}/derived/{file_id}__scene3d.v5.sha256-{digest}/manifest.json
//	{root}/derived/{file_id}__scene3d.v5.sha256-{digest}/chunk_{n:05d}.bin
//	{root}/derived/{file_id}__scene3d.v5.sha256-{digest}/scene-lod.{rad,radc}
//	{root}/derived/{file_id}__scene3d.v5.sha256-{digest}.failed
//
// The control plane NEVER parses a scene file in the request path. The real
// files are 3.4 GB with 14.5M splats; everything here is a bounded 64 KiB header
// sniff plus os.Stat, and the derived bytes are streamed with http.ServeContent
// so Range and conditional requests cost nothing.
//
// Property offsets are ALWAYS derived from the PLY header. The two measured
// reference files disagree with the canonical layout — Postshot omits nx/ny/nz
// and writes a 236-byte stride where INRIA's writer writes 248 — so a hardcoded
// layout silently misreads every field.
//
// A COLMAP reconstruction is the third species and it is not a file at all: it
// is a DIRECTORY (or a zip of one) holding cameras/images/points3D. Recognition
// is structural — the model's member NAMES, at the four roots COLMAP itself
// writes — and never parses a record. It cannot: an images.bin record carries a
// NUL-terminated name of arbitrary length followed by a variable number of
// 24-byte 2D observations, so there is no stride and no way to reach image N
// without walking 0..N-1 (1000 images x 8000 keypoints is ~192 MB of
// observations we would read only to throw away). The derive job does that work
// once, off the request path.

const (
	// plyHeaderPeekMaxBytes bounds the header sniff. A PLY header is ASCII and
	// ends at "end_header"; the largest real one (degree-3 SH, 62 properties) is
	// ~1.5 KiB, so 64 KiB is generous and still O(1) against a 3.4 GB source.
	plyHeaderPeekMaxBytes = int64(64 << 10)

	// colmapZipMaxEntries and colmapZipMaxCentralDirectoryBytes bound the archive
	// probe. archive/zip parses the WHOLE central directory eagerly, and this probe
	// runs on every viewer, manifest and chunk request for the resource, so an
	// archive with a hostile — or merely enormous — member count has to be refused
	// BEFORE that parse rather than re-paid on every request. 20k members covers a
	// sparse model plus a few thousand source photographs; a larger archive is not
	// recognized here at all (it stays an ordinary file resource) rather than
	// turning the request path into an unbounded directory walk.
	colmapZipMaxEntries               = 20_000
	colmapZipMaxCentralDirectoryBytes = int64(8 << 20)

	scene3dManifestName    = "manifest.json"
	scene3dChunkNameFormat = "chunk_%05d.bin"
	scene3dLodHeaderName   = "scene-lod.rad"
	scene3dLodChunkPrefix  = "scene-lod-"
	scene3dLodChunkSuffix  = ".radc"
	scene3dCameraImageName = "camera-image_%05d.jpg"
	scene3dDerivativeRev   = "v5"

	// scene3dMaxChunkIndex caps the addressable chunk space. At the default
	// 50k elements per chunk this is 5e10 elements — far past any real scene —
	// so anything larger is a malformed or hostile request, not a deep scene.
	scene3dMaxChunkIndex = 999_999

	// Derive-job parameters (contract §5/§7). max_splats_per_chunk is the frozen
	// default; tier_count is not fixed by the contract, and this is the value the
	// control plane requests.
	scene3dMaxSplatsPerChunk = 50_000
	scene3dTierCount         = 4
	scene3dPreviewSplats     = 100_000
	scene3dPreviewPoints     = 280_000

	defaultScene3dChunkInFlightBytes = int64(256 << 20)

	// scene3dFailureBackoff mirrors pyramidFailureBackoff: long enough that a
	// poison source cannot re-queue a doomed derive on every viewer open, short
	// enough that a transient cause (a worker deploy, a since-fixed bug) retries
	// within the hour. Override with ULTRA_CONTROL_SCENE3D_FAILURE_BACKOFF_SECONDS.
	scene3dFailureBackoff = time.Hour
)

var errScene3dChunkAdmission = errors.New("scene chunk in-flight byte budget is exhausted")

func newScene3dChunkInFlightBudgetFromEnv() *byteAdmissionBudget {
	maxBytes := defaultScene3dChunkInFlightBytes
	if raw := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_SCENE3D_INFLIGHT_BYTES")); raw != "" {
		if parsed, err := strconv.ParseInt(raw, 10, 64); err == nil && parsed >= 0 {
			maxBytes = parsed
		}
	}
	return newByteAdmissionBudget(maxBytes)
}

// scene3dChunkInFlightBudget bounds concurrent chunk delivery so ten simultaneous
// scene opens (each streaming megabyte-class chunks) cannot exhaust the edge node.
var scene3dChunkInFlightBudget = newScene3dChunkInFlightBudgetFromEnv()

var scene3dDerivationThrottle = newDerivationThrottle(derivationThrottleWindow)

// --- header sniffing ---------------------------------------------------------

// plyProperty is one fixed-width scalar property of an element. List properties
// are not representable here: an element containing one has no fixed stride.
type plyProperty struct {
	name string
	size int
}

type plyElementInfo struct {
	name          string
	count         int64
	propertyNames []string
	strideBytes   int64
	hasList       bool
}

// plyInfo is everything the bounded header peek can PROVE about a PLY file.
// declaredSHDegree is what the header allocates, never what the data contains —
// the measured degree is scanned by the derive job and reported in the manifest
// (the reference splat file declares degree 3 and measures degree 0).
type plyInfo struct {
	species          string // "splat" | "pointcloud"
	byteOrder        string // "binary_little_endian" | "binary_big_endian" | "ascii"
	vertexCount      int64
	strideBytes      int // 0 for ascii, which has no fixed stride
	dataOffset       int64
	declaredSHDegree int
	restCount        int // f_rest_* property count the degree was derived from
	propertyCount    int
	propertyNames    []string
	elements         []plyElementInfo
	writer           string // parsed from PLY comments when recognizable
}

// scene3dInfo is the viewer-facing summary of a scene resource: the container
// format plus, for PLY, the header facts and, for COLMAP, the model layout.
type scene3dInfo struct {
	format    string // "ply" | "colmap"
	sceneKind string // "splat" | "pointcloud" | "colmap"
	ply       plyInfo
	hasPly    bool
	colmap    colmapInfo
	hasColmap bool
	// unsupportedReason is non-empty only for a named Scene3D container whose
	// schema Lens can identify but cannot decode without scientific ambiguity.
	// Such a file must stay on the Scene3D path so it never falls through to the
	// image service or a worker that will reject it later.
	unsupportedReason string
}

var (
	errPlyASCII              = errors.New("ASCII PLY is not supported; re-export it as binary_little_endian or binary_big_endian PLY")
	errPlyIncompleteSplat    = errors.New("incomplete Gaussian-splat schema")
	errPlyVariablePrefix     = errors.New("variable-width element precedes the vertex element")
	errPlyUnsupportedPayload = errors.New("PLY header or payload cannot be addressed safely")
)

// plyScalarSizes is the PLY scalar type table (both the modern and legacy names).
var plyScalarSizes = map[string]int{
	"char": 1, "int8": 1,
	"uchar": 1, "uint8": 1,
	"short": 2, "int16": 2,
	"ushort": 2, "uint16": 2,
	"int": 4, "int32": 4,
	"uint": 4, "uint32": 4,
	"float": 4, "float32": 4,
	"double": 8, "float64": 8,
}

// plyKnownWriters maps a marker found in a PLY comment to the writer name the
// manifest reports. Postshot's own marker is "postshot.anti_aliasing=1"; nothing
// is inferred from a comment we do not recognize (an unknown writer stays empty
// rather than becoming a guess presented as provenance).
var plyKnownWriters = []struct {
	marker string
	writer string
}{
	{marker: "postshot", writer: "postshot"},
	{marker: "supersplat", writer: "supersplat"},
	{marker: "playcanvas", writer: "playcanvas"},
	{marker: "nerfstudio", writer: "nerfstudio"},
	{marker: "opensplat", writer: "opensplat"},
	{marker: "colmap", writer: "colmap"},
	{marker: "meshlab", writer: "meshlab"},
	{marker: "open3d", writer: "open3d"},
}

// parsePlyHeader reads the ASCII header out of the peeked prefix and derives the
// vertex element's stride and data offset FROM THE HEADER. The boolean wrapper
// is retained for the existing bounded-probe API; the detailed parser carries a
// stable reason to the viewer when a named PLY is unsupported.
func parsePlyHeader(head []byte) (plyInfo, bool) {
	info, err := parsePlyHeaderDetailed(head)
	return info, err == nil
}

func parsePlyHeaderDetailed(head []byte) (plyInfo, error) {
	if !bytes.HasPrefix(head, []byte("ply")) {
		return plyInfo{}, fmt.Errorf("%w: missing PLY magic", errPlyUnsupportedPayload)
	}
	info := plyInfo{}
	var elements []plyElementInfo
	currentElement := -1
	vertexElementIndex := -1
	var properties []plyProperty
	sawMagic := false
	offset := 0
	for offset < len(head) {
		newline := bytes.IndexByte(head[offset:], '\n')
		if newline < 0 {
			return plyInfo{}, fmt.Errorf("%w: header exceeds the bounded peek", errPlyUnsupportedPayload)
		}
		line := strings.TrimRight(string(head[offset:offset+newline]), "\r")
		offset += newline + 1
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}
		switch fields[0] {
		case "ply":
			if sawMagic || len(fields) != 1 {
				return plyInfo{}, fmt.Errorf("%w: malformed PLY magic", errPlyUnsupportedPayload)
			}
			sawMagic = true
		case "format":
			if len(fields) != 3 || fields[2] != "1.0" {
				return plyInfo{}, fmt.Errorf("%w: unsupported PLY format declaration", errPlyUnsupportedPayload)
			}
			info.byteOrder = fields[1]
		case "comment":
			if writer := plyWriterFromComment(line); writer != "" && info.writer == "" {
				info.writer = writer
			}
		case "element":
			if len(fields) != 3 {
				return plyInfo{}, fmt.Errorf("%w: malformed element declaration", errPlyUnsupportedPayload)
			}
			if info.byteOrder == "" {
				return plyInfo{}, fmt.Errorf("%w: element declared before format", errPlyUnsupportedPayload)
			}
			count, err := strconv.ParseInt(fields[2], 10, 64)
			if err != nil || count < 0 {
				return plyInfo{}, fmt.Errorf("%w: invalid element count", errPlyUnsupportedPayload)
			}
			elements = append(elements, plyElementInfo{name: fields[1], count: count})
			currentElement = len(elements) - 1
			if fields[1] == "vertex" {
				if vertexElementIndex >= 0 {
					return plyInfo{}, fmt.Errorf("%w: duplicate vertex element", errPlyUnsupportedPayload)
				}
				vertexElementIndex = currentElement
				info.vertexCount = count
			}
		case "property":
			if currentElement < 0 {
				return plyInfo{}, fmt.Errorf("%w: property declared before an element", errPlyUnsupportedPayload)
			}
			element := &elements[currentElement]
			if len(fields) >= 2 && fields[1] == "list" {
				if len(fields) != 5 {
					return plyInfo{}, fmt.Errorf("%w: malformed list property", errPlyUnsupportedPayload)
				}
				if _, known := plyScalarSizes[fields[2]]; !known {
					return plyInfo{}, fmt.Errorf("%w: unknown list-count scalar type", errPlyUnsupportedPayload)
				}
				if _, known := plyScalarSizes[fields[3]]; !known {
					return plyInfo{}, fmt.Errorf("%w: unknown list-item scalar type", errPlyUnsupportedPayload)
				}
				for _, existing := range element.propertyNames {
					if existing == fields[4] {
						return plyInfo{}, fmt.Errorf("%w: duplicate property %q", errPlyUnsupportedPayload, fields[4])
					}
				}
				element.hasList = true
				element.propertyNames = append(element.propertyNames, fields[4])
				if element.name == "vertex" {
					return plyInfo{}, fmt.Errorf("%w: variable-width vertex records", errPlyUnsupportedPayload)
				}
				continue
			}
			if len(fields) != 3 {
				return plyInfo{}, fmt.Errorf("%w: malformed scalar property", errPlyUnsupportedPayload)
			}
			size, known := plyScalarSizes[fields[1]]
			if !known {
				return plyInfo{}, fmt.Errorf("%w: unknown scalar type", errPlyUnsupportedPayload)
			}
			for _, existing := range element.propertyNames {
				if existing == fields[2] {
					return plyInfo{}, fmt.Errorf("%w: duplicate property %q", errPlyUnsupportedPayload, fields[2])
				}
			}
			element.propertyNames = append(element.propertyNames, fields[2])
			element.strideBytes += int64(size)
			if element.name == "vertex" {
				properties = append(properties, plyProperty{name: fields[2], size: size})
			}
		case "end_header":
			if !sawMagic || vertexElementIndex < 0 || len(properties) == 0 {
				return plyInfo{}, fmt.Errorf("%w: no fixed-width vertex element", errPlyUnsupportedPayload)
			}
			if info.byteOrder == "ascii" {
				return plyInfo{}, errPlyASCII
			}
			if info.byteOrder != "binary_little_endian" && info.byteOrder != "binary_big_endian" {
				return plyInfo{}, fmt.Errorf("%w: unsupported byte order", errPlyUnsupportedPayload)
			}
			dataOffset := int64(offset)
			for index := 0; index < vertexElementIndex; index++ {
				prefix := elements[index]
				if prefix.hasList {
					return plyInfo{}, errPlyVariablePrefix
				}
				prefixBytes, exact := checkedNonNegativeProduct(prefix.count, prefix.strideBytes)
				if !exact || prefixBytes > math.MaxInt64-dataOffset {
					return plyInfo{}, fmt.Errorf("%w: element offsets overflow", errPlyUnsupportedPayload)
				}
				dataOffset += prefixBytes
			}
			info.dataOffset = dataOffset
			info.propertyCount = len(properties)
			for _, property := range properties {
				info.strideBytes += property.size
				info.propertyNames = append(info.propertyNames, property.name)
			}
			species, err := classifyPlyPropertiesDetailed(properties)
			if err != nil {
				return plyInfo{}, err
			}
			info.species = species
			info.restCount, info.declaredSHDegree = plyDeclaredSHDegree(properties)
			info.elements = elements
			return info, nil
		}
	}
	return plyInfo{}, fmt.Errorf("%w: missing end_header", errPlyUnsupportedPayload)
}

// classifyPlyProperties names the species from the property set. Splats carry
// the INRIA parameter block (DC colour, opacity, log scales, rotation quat);
// anything else with coordinates is a point cloud.
func classifyPlyProperties(properties []plyProperty) (string, bool) {
	species, err := classifyPlyPropertiesDetailed(properties)
	return species, err == nil
}

func classifyPlyPropertiesDetailed(properties []plyProperty) (string, error) {
	present := make(map[string]bool, len(properties))
	for _, property := range properties {
		present[property.name] = true
	}
	if !present["x"] || !present["y"] || !present["z"] {
		return "", fmt.Errorf("%w: missing x/y/z coordinates", errPlyUnsupportedPayload)
	}
	required := []string{
		"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
		"scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
	}
	completeSplat := true
	for _, name := range required {
		completeSplat = completeSplat && present[name]
	}
	if completeSplat {
		return "splat", nil
	}
	for name := range present {
		if name == "opacity" || strings.HasPrefix(name, "f_dc_") || strings.HasPrefix(name, "f_rest_") ||
			strings.HasPrefix(name, "scale_") || strings.HasPrefix(name, "rot_") {
			return "", errPlyIncompleteSplat
		}
	}
	return "pointcloud", nil
}

// plyDeclaredSHDegree derives the DECLARED spherical-harmonic degree from the
// f_rest_* allocation: 3 channels × ((degree+1)² − 1) coefficients. A count that
// does not fit that identity leaves the degree at 0 rather than rounding to a
// number the file does not actually describe.
func plyDeclaredSHDegree(properties []plyProperty) (restCount, degree int) {
	for _, property := range properties {
		if strings.HasPrefix(property.name, "f_rest_") {
			restCount++
		}
	}
	if restCount == 0 || restCount%3 != 0 {
		return restCount, 0
	}
	perChannel := restCount / 3
	for candidate := 1; candidate <= 8; candidate++ {
		if (candidate+1)*(candidate+1)-1 == perChannel {
			return restCount, candidate
		}
	}
	return restCount, 0
}

func plyWriterFromComment(line string) string {
	lower := strings.ToLower(line)
	for _, known := range plyKnownWriters {
		if strings.Contains(lower, known.marker) {
			return known.writer
		}
	}
	return ""
}

// plyPeek reads at most plyHeaderPeekMaxBytes and reports what the header proves.
// It also verifies the payload is actually present (size ≥ offset + count×stride):
// a truncated splat file must not be advertised as renderable.
func plyPeek(path string) (plyInfo, bool) {
	info, err := plyPeekDetailed(path)
	return info, err == nil
}

func plyPeekDetailed(path string) (plyInfo, error) {
	file, err := os.Open(path)
	if err != nil {
		return plyInfo{}, err
	}
	defer func() { _ = file.Close() }()
	stat, err := file.Stat()
	if err != nil || !stat.Mode().IsRegular() {
		return plyInfo{}, fmt.Errorf("%w: PLY is not a regular file", errPlyUnsupportedPayload)
	}
	head := make([]byte, plyHeaderPeekMaxBytes)
	read, err := io.ReadFull(file, head)
	if read == 0 || (err != nil && err != io.ErrUnexpectedEOF && err != io.EOF) {
		return plyInfo{}, fmt.Errorf("%w: PLY header cannot be read", errPlyUnsupportedPayload)
	}
	info, err := parsePlyHeaderDetailed(head[:read])
	if err != nil {
		return plyInfo{}, err
	}
	payloadBytes, exact := checkedNonNegativeProduct(info.vertexCount, int64(info.strideBytes))
	if !exact || info.dataOffset > stat.Size() || payloadBytes > stat.Size()-info.dataOffset {
		return plyInfo{}, fmt.Errorf("%w: truncated vertex payload", errPlyUnsupportedPayload)
	}
	return info, nil
}

// --- COLMAP model recognition ------------------------------------------------

// colmapInfo is everything the bounded probe can PROVE about a COLMAP
// reconstruction: which container it arrived in, where inside it the model root
// sits, whether the records are binary or text, and which optional products are
// present. Nothing here is parsed from a record — see the file header for why.
type colmapInfo struct {
	variant        string // "directory" | "zip" | "bundle"
	modelPath      string // slash-separated model root relative to the resource ("" = at the root)
	modelCount     int
	modelPaths     []string
	recordFormat   string // "bin" | "txt" | "mixed"
	camerasName    string // the member name as it actually appears
	imagesName     string
	points3DName   string
	hasPoints3D    bool
	hasRigs        bool // modern COLMAP writes rigs/frames; a legacy model has neither
	hasFrames      bool
	bundlePlyPath  string
	bundlePlyCount int
	imageMembers   int
}

// colmapModelRoots are the four places COLMAP itself puts a model, in the order
// a reconstruction is most likely to be laid out. A model is recognized ONLY at
// one of these roots (for a directory), which is what keeps an unrelated tree
// that merely happens to contain a file called images.txt from being mistaken
// for a reconstruction.
var colmapModelRoots = []string{"", "sparse/0", "sparse", "dense/sparse"}

var (
	colmapCamerasCandidates = []string{"cameras.bin", "cameras.txt"}
	colmapImagesCandidates  = []string{"images.bin", "images.txt"}
	// COLMAP capitalizes the D in points3D; some exporters do not, and the
	// difference decides nothing, so both spellings are probed.
	colmapPoints3DCandidates = []string{"points3D.bin", "points3D.txt", "points3d.bin", "points3d.txt"}
	colmapRigsCandidates     = []string{"rigs.bin", "rigs.txt"}
	colmapFramesCandidates   = []string{"frames.bin", "frames.txt"}
)

// colmapMemberNames is the lowercased union of every member the probe cares
// about. Zip entries outside this set are skipped by NAME, so an archive of any
// shape costs one map lookup per central-directory entry and nothing else.
var colmapMemberNames = func() map[string]bool {
	names := map[string]bool{}
	for _, group := range [][]string{
		colmapCamerasCandidates, colmapImagesCandidates,
		colmapPoints3DCandidates, colmapRigsCandidates, colmapFramesCandidates,
	} {
		for _, name := range group {
			names[strings.ToLower(name)] = true
		}
	}
	return names
}()

// colmapMemberLookup resolves one candidate member inside a model root and
// returns the name AS IT ACTUALLY APPEARS plus whether it is there.
type colmapMemberLookup func(candidate string) (string, bool)

func firstColmapMember(lookup colmapMemberLookup, candidates []string) string {
	for _, candidate := range candidates {
		if actual, present := lookup(candidate); present {
			return actual
		}
	}
	return ""
}

// colmapModelFromMembers decides whether one model root holds a reconstruction.
// cameras + images are REQUIRED (they are what makes it a posed reconstruction);
// points3D is optional, and so are the rigs/frames a modern COLMAP writes —
// their presence must never turn a model into a non-model.
func colmapModelFromMembers(modelPath string, lookup colmapMemberLookup) (colmapInfo, bool) {
	cameras := firstColmapMember(lookup, colmapCamerasCandidates)
	images := firstColmapMember(lookup, colmapImagesCandidates)
	if cameras == "" || images == "" {
		return colmapInfo{}, false
	}
	info := colmapInfo{
		modelPath:    modelPath,
		recordFormat: colmapRecordFormat(cameras, images),
		camerasName:  cameras,
		imagesName:   images,
		points3DName: firstColmapMember(lookup, colmapPoints3DCandidates),
		hasRigs:      firstColmapMember(lookup, colmapRigsCandidates) != "",
		hasFrames:    firstColmapMember(lookup, colmapFramesCandidates) != "",
	}
	info.hasPoints3D = info.points3DName != ""
	return info, true
}

// colmapRecordFormat reports the encoding of the two required members. A model
// half-converted between the binary and text serializations is still a model, so
// it is reported as "mixed" rather than rejected.
func colmapRecordFormat(cameras, images string) string {
	camerasBinary := strings.HasSuffix(strings.ToLower(cameras), ".bin")
	imagesBinary := strings.HasSuffix(strings.ToLower(images), ".bin")
	switch {
	case camerasBinary && imagesBinary:
		return "bin"
	case !camerasBinary && !imagesBinary:
		return "txt"
	default:
		return "mixed"
	}
}

// colmapPeek recognizes a COLMAP reconstruction with a bounded probe: at most a
// handful of os.Stat calls for a directory, or a 4-byte magic read plus the
// archive's central directory for a zip. It NEVER opens a model member and never
// inflates an archive member.
func colmapPeek(path string) (colmapInfo, bool) {
	stat, err := os.Stat(path)
	if err != nil {
		return colmapInfo{}, false
	}
	if stat.IsDir() {
		return colmapDirectoryPeek(path)
	}
	if !stat.Mode().IsRegular() {
		return colmapInfo{}, false
	}
	return colmapZipPeek(path, stat.Size())
}

// colmapDirectoryPeek probes the four known model roots. Worst case is 4 roots x
// 13 stats, and a root that is not a model costs 2 (cameras.bin, cameras.txt) —
// which is what makes this affordable in the request path.
func colmapDirectoryPeek(root string) (colmapInfo, bool) {
	models := []colmapInfo{}
	for _, modelPath := range colmapModelRoots {
		directory := root
		if modelPath != "" {
			directory = filepath.Join(root, filepath.FromSlash(modelPath))
		}
		// Only a REGULAR file counts. Following a symlink is safe here because no
		// byte of the target is read — the probe decides on names alone, and the
		// derive job, not this path, is what later opens the model.
		lookup := func(candidate string) (string, bool) {
			stat, err := os.Stat(filepath.Join(directory, candidate))
			return candidate, err == nil && stat.Mode().IsRegular()
		}
		if info, ok := colmapModelFromMembers(modelPath, lookup); ok {
			info.variant = "directory"
			models = append(models, info)
		}
	}
	return summarizeColmapModels("directory", models)
}

// colmapZipPeek recognizes a zipped model from the archive's CENTRAL DIRECTORY
// ALONE. zip.File.Open is never called, so no member is ever inflated: a 4 GB
// bomb declared inside a 40 KB archive costs exactly the same as an empty one.
// The two required members must live in the SAME directory — a cameras.txt in
// one branch of the tree and an images.txt in another is not a model.
func colmapZipPeek(path string, size int64) (colmapInfo, bool) {
	file, err := os.Open(path)
	if err != nil {
		return colmapInfo{}, false
	}
	defer func() { _ = file.Close() }()
	if !hasZipMagic(file) {
		return colmapInfo{}, false // cheap rejection: 4 bytes, and every non-archive stops here
	}
	entries, centralDirectoryBytes, ok := zipCentralDirectoryExtent(file, size)
	if !ok || entries <= 0 || entries > colmapZipMaxEntries || centralDirectoryBytes > colmapZipMaxCentralDirectoryBytes {
		return colmapInfo{}, false
	}
	reader, err := zip.NewReader(file, size)
	if err != nil {
		return colmapInfo{}, false
	}
	// Names only. Nothing below reads, decompresses or even opens a member.
	members := map[string]map[string]string{}
	plyPaths := []string{}
	imageMembers := 0
	for _, entry := range reader.File {
		directory, base, valid := colmapZipEntryPath(entry.Name)
		if !valid {
			continue
		}
		lower := strings.ToLower(base)
		if strings.HasSuffix(lower, ".ply") {
			plyPaths = append(plyPaths, pathpkg.Join(directory, base))
		}
		if scene3dBundleImageName(lower) {
			imageMembers++
		}
		if !colmapMemberNames[lower] {
			continue
		}
		if members[directory] == nil {
			members[directory] = map[string]string{}
		}
		if _, duplicate := members[directory][lower]; !duplicate {
			members[directory][lower] = base
		}
	}
	models := []colmapInfo{}
	for directory, names := range members {
		lookup := func(candidate string) (string, bool) {
			actual, present := names[strings.ToLower(candidate)]
			return actual, present
		}
		info, ok := colmapModelFromMembers(directory, lookup)
		if !ok {
			continue
		}
		models = append(models, info)
	}
	info, recognized := summarizeColmapModels("zip", models)
	if !recognized || len(plyPaths) == 0 {
		return info, recognized
	}
	sort.Strings(plyPaths)
	info.variant = "bundle"
	info.bundlePlyCount = len(plyPaths)
	info.bundlePlyPath = plyPaths[0]
	info.imageMembers = imageMembers
	return info, true
}

func scene3dBundleImageName(lowerBase string) bool {
	switch filepath.Ext(lowerBase) {
	case ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp":
		return true
	default:
		return false
	}
}

// summarizeColmapModels preserves every candidate root. A unique model exposes
// its detailed layout; an ambiguous resource deliberately exposes only the
// sorted candidate paths so neither the request path nor the worker silently
// decides which reconstruction represents the scientist's data.
func summarizeColmapModels(variant string, models []colmapInfo) (colmapInfo, bool) {
	if len(models) == 0 {
		return colmapInfo{}, false
	}
	sort.Slice(models, func(left, right int) bool {
		return colmapModelRootPrecedes(models[left].modelPath, models[right].modelPath)
	})
	paths := make([]string, 0, len(models))
	for _, model := range models {
		paths = append(paths, model.modelPath)
	}
	if len(models) == 1 {
		info := models[0]
		info.variant = variant
		info.modelCount = 1
		info.modelPaths = paths
		return info, true
	}
	return colmapInfo{
		variant:    variant,
		modelCount: len(models),
		modelPaths: paths,
	}, true
}

// colmapModelRootPrecedes orders candidate model roots: shallower first, then
// lexicographically.
func colmapModelRootPrecedes(candidate, incumbent string) bool {
	candidateDepth, incumbentDepth := colmapPathDepth(candidate), colmapPathDepth(incumbent)
	if candidateDepth != incumbentDepth {
		return candidateDepth < incumbentDepth
	}
	return candidate < incumbent
}

func colmapPathDepth(value string) int {
	if value == "" {
		return 0
	}
	return strings.Count(value, "/") + 1
}

// colmapZipEntryPath splits a zip entry name into its directory and base name,
// rejecting anything that is not a plain relative file path. A traversal or
// absolute member is refused outright: the model path it produced would be
// handed to the derive job, and a "../.." there is a path the worker should
// never be told to join.
func colmapZipEntryPath(name string) (directory, base string, ok bool) {
	if name == "" || strings.HasSuffix(name, "/") || strings.HasPrefix(name, "/") {
		return "", "", false
	}
	if strings.ContainsAny(name, "\\\x00") {
		return "", "", false
	}
	if cut := strings.LastIndexByte(name, '/'); cut >= 0 {
		directory, base = name[:cut], name[cut+1:]
	} else {
		base = name
	}
	if base == "" {
		return "", "", false
	}
	for _, segment := range strings.Split(directory, "/") {
		if segment == ".." || segment == "." {
			return "", "", false
		}
	}
	return directory, base, true
}

// zip end-of-central-directory constants (APPNOTE 4.3.16 / 4.3.14 / 4.3.15).
const (
	zipEOCDSignature       = uint32(0x06054b50)
	zip64LocatorSignature  = uint32(0x07064b50)
	zip64EOCDSignature     = uint32(0x06064b50)
	zipEOCDMinBytes        = 22
	zipEOCDMaxCommentBytes = 0xFFFF
	zip64LocatorBytes      = 20
	zip64EOCDBytes         = 56
)

// hasZipMagic reads the 4-byte local-file-header signature. This is the gate that
// keeps the archive path off every non-archive resource: a TIFF or a NIfTI pays
// one 4-byte read and stops.
func hasZipMagic(file *os.File) bool {
	magic := make([]byte, 4)
	if _, err := file.ReadAt(magic, 0); err != nil {
		return false
	}
	switch binary.LittleEndian.Uint32(magic) {
	case 0x04034b50, // PK\x03\x04 local file header
		0x06054b50, // PK\x05\x06 empty archive (EOCD only)
		0x08074b50: // PK\x07\x08 spanned marker
		return true
	}
	return false
}

// zipCentralDirectoryExtent reads the end-of-central-directory record from a
// BOUNDED tail read and reports how many entries the central directory holds and
// how many bytes it occupies — before archive/zip is asked to parse any of it.
// Without this, "cheap probe" would mean "parse an attacker-chosen number of
// central-directory records on every request".
func zipCentralDirectoryExtent(file *os.File, size int64) (entries, centralDirectoryBytes int64, ok bool) {
	if size < zipEOCDMinBytes {
		return 0, 0, false
	}
	window := int64(zipEOCDMinBytes + zipEOCDMaxCommentBytes)
	if window > size {
		window = size
	}
	tail := make([]byte, window)
	if _, err := file.ReadAt(tail, size-window); err != nil {
		return 0, 0, false
	}
	index := -1
	for offset := len(tail) - zipEOCDMinBytes; offset >= 0; offset-- {
		if binary.LittleEndian.Uint32(tail[offset:]) != zipEOCDSignature {
			continue
		}
		// The declared comment length must consume exactly the rest of the file, or
		// this is archive bytes that merely look like a signature.
		if int(binary.LittleEndian.Uint16(tail[offset+20:]))+offset+zipEOCDMinBytes != len(tail) {
			continue
		}
		index = offset
		break
	}
	if index < 0 {
		return 0, 0, false
	}
	entries = int64(binary.LittleEndian.Uint16(tail[index+10:]))
	centralDirectoryBytes = int64(binary.LittleEndian.Uint32(tail[index+12:]))
	centralDirectoryOffset := int64(binary.LittleEndian.Uint32(tail[index+16:]))
	if entries != 0xFFFF && centralDirectoryBytes != 0xFFFFFFFF && centralDirectoryOffset != 0xFFFFFFFF {
		return entries, centralDirectoryBytes, true
	}
	// A zip64 sentinel: the real counts live in the zip64 EOCD record, reached via
	// the 20-byte locator immediately before the EOCD.
	locatorAt := size - window + int64(index) - zip64LocatorBytes
	if locatorAt < 0 {
		return 0, 0, false
	}
	locator := make([]byte, zip64LocatorBytes)
	if _, err := file.ReadAt(locator, locatorAt); err != nil {
		return 0, 0, false
	}
	if binary.LittleEndian.Uint32(locator) != zip64LocatorSignature {
		return 0, 0, false
	}
	recordAt := binary.LittleEndian.Uint64(locator[8:])
	if recordAt > math.MaxInt64 || int64(recordAt)+zip64EOCDBytes > size {
		return 0, 0, false
	}
	record := make([]byte, zip64EOCDBytes)
	if _, err := file.ReadAt(record, int64(recordAt)); err != nil {
		return 0, 0, false
	}
	if binary.LittleEndian.Uint32(record) != zip64EOCDSignature {
		return 0, 0, false
	}
	entries64 := binary.LittleEndian.Uint64(record[32:])
	bytes64 := binary.LittleEndian.Uint64(record[40:])
	if entries64 > math.MaxInt64 || bytes64 > math.MaxInt64 {
		return 0, 0, false
	}
	return int64(entries64), int64(bytes64), true
}

// isScene3dName reports whether a resource name is one of the 3D scene
// containers. PLY is additionally header-verified by scene3dPeek. COLMAP is
// deliberately absent: a reconstruction has no distinguishing name at all (it is a folder, often just
// "sparse" or the dataset's own name), so it is recognized STRUCTURALLY by
// colmapPeek instead.
func isScene3dName(name string) bool {
	return scene3dFormatFromName(name) != ""
}

func scene3dFormatFromName(name string) string {
	lower := strings.ToLower(strings.TrimSpace(name))
	if strings.HasSuffix(lower, ".ply") {
		return "ply"
	}
	return ""
}

// scene3dPeek classifies a resource as a 3D scene. A named PLY stays on this
// path even when its schema is unsupported, so Lens can show the precise format
// failure instead of handing the file to the image service or drawing an empty
// canvas.
//
// A COLMAP reconstruction carries no name signal, so it is probed structurally
// once the name has proven to be none of the single-file containers. That order
// matters twice over: it keeps the per-request cost of every ordinary resource
// at one os.Stat plus (for a regular file) a 4-byte magic read, and it means a
// COLMAP DIRECTORY is classified here — before the libbioimage probe and before
// enqueuePyramidDerivation — so a directory of camera poses can never be handed
// to an imaging engine or queued for an imgcnv pyramid transcode.
func scene3dPeek(record resourceRecord, path string) (scene3dInfo, bool) {
	format := scene3dFormatFromName(record.OriginalName)
	if format == "" {
		if colmap, isColmap := colmapPeek(path); isColmap {
			if colmap.variant == "bundle" {
				return scene3dInfo{
					format: "reconstruction_bundle", sceneKind: "reconstruction",
					colmap: colmap, hasColmap: true,
				}, true
			}
			return scene3dInfo{format: "colmap", sceneKind: "colmap", colmap: colmap, hasColmap: true}, true
		}
		return scene3dInfo{}, false
	}
	ply, err := plyPeekDetailed(path)
	if err != nil {
		return scene3dInfo{
			format:            "ply",
			sceneKind:         "unknown",
			unsupportedReason: plyUnsupportedReason(err),
		}, true
	}
	return scene3dInfo{format: format, sceneKind: ply.species, ply: ply, hasPly: true}, true
}

func plyUnsupportedReason(err error) string {
	switch {
	case errors.Is(err, errPlyASCII):
		return errPlyASCII.Error() + "."
	case errors.Is(err, errPlyIncompleteSplat):
		return "The PLY declares Gaussian-splat properties but its schema is incomplete; export x/y/z, f_dc_0..2, opacity, scale_0..2, and rot_0..3."
	default:
		return "This PLY header or payload cannot be addressed safely by Lens."
	}
}

func resourceIsScene3d(record resourceRecord, path string) bool {
	_, ok := scene3dPeek(record, path)
	return ok
}

// scene3dCanDerive is the publication boundary, not merely a format check. The
// worker can source-bind regular uploads (PLY and zipped COLMAP models) to the
// catalog's immutable SHA-256 and byte count. A directory has no cataloged byte
// stream to hash or revalidate, so accepting one here would let a changing tree
// publish under a stale identity. Directory models remain recognizable — which
// keeps them away from the image service — but must be archived before deriving.
func scene3dCanDerive(record resourceRecord, info scene3dInfo) bool {
	if !isSHA256Hex(strings.ToLower(strings.TrimSpace(record.SHA256))) || record.SizeBytes < 0 {
		return false
	}
	if info.unsupportedReason != "" {
		return false
	}
	if info.hasColmap {
		if info.colmap.variant == "bundle" {
			return info.colmap.modelCount == 1 && info.colmap.bundlePlyCount == 1
		}
		return info.colmap.variant != "directory" && info.colmap.modelCount == 1
	}
	return true
}

// --- derived artifacts -------------------------------------------------------

// derivedScene3dName is the deterministic destination the derive job writes into,
// mirroring derivedPyramidName. It is a DIRECTORY (manifest + chunks + poster).
func derivedScene3dName(fileID, sourceSHA256 string) string {
	return fileID + "__scene3d." + scene3dDerivativeRev + ".sha256-" + strings.ToLower(strings.TrimSpace(sourceSHA256))
}

func derivedScene3dDir(root, fileID, sourceSHA256 string) string {
	return filepath.Join(root, "derived", derivedScene3dName(fileID, sourceSHA256))
}

// derivedScene3dFailedMarkerPath is "{dst_dir}.failed" (contract §7): the sidecar
// a permanently-failed derive writes, which the control plane honours as backoff.
func derivedScene3dFailedMarkerPath(root, fileID, sourceSHA256 string) string {
	return derivedScene3dDir(root, fileID, sourceSHA256) + ".failed"
}

func scene3dManifestPath(root, fileID, sourceSHA256 string) string {
	return filepath.Join(derivedScene3dDir(root, fileID, sourceSHA256), scene3dManifestName)
}

func scene3dFailureBackoffWindow() time.Duration {
	if raw := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_SCENE3D_FAILURE_BACKOFF_SECONDS")); raw != "" {
		if seconds, err := strconv.Atoi(raw); err == nil && seconds >= 0 {
			return time.Duration(seconds) * time.Second
		}
	}
	return scene3dFailureBackoff
}

// recentScene3dFailure reports whether a permanent derive failure was recorded
// within the backoff window. The contract freezes the marker's PATH only, so
// presence + mtime is all this can honour (the pyramid marker additionally
// carries a source-bound JSON body). Failure-isolated: any stat error falls
// through to false and never blocks serving.
func recentScene3dFailure(root, fileID, sourceSHA256 string, now time.Time) bool {
	window := scene3dFailureBackoffWindow()
	if window <= 0 {
		return false
	}
	info, err := regularFileInfo(derivedScene3dFailedMarkerPath(root, fileID, sourceSHA256))
	if err != nil {
		return false
	}
	return now.Sub(info.ModTime()) < window
}

// scene3dDeriveStatus is the descriptor's "status": "ready" once the manifest
// exists, "failed" while a permanent-failure marker is fresh, "deriving" otherwise.
func scene3dDeriveStatus(root, fileID, sourceSHA256 string, now time.Time) string {
	if _, err := regularFileInfo(scene3dManifestPath(root, fileID, sourceSHA256)); err == nil {
		return "ready"
	}
	if recentScene3dFailure(root, fileID, sourceSHA256, now) {
		return "failed"
	}
	return "deriving"
}

// enqueueScene3dDerivation publishes the scene.derive job (contract §7). It
// no-ops when the queue is absent, a recent derive permanently failed, or one was
// enqueued inside the throttle window — and, unlike the pyramid lane, it does NOT
// require the image service, because a scene never touches that sidecar and the
// unconfigured viewer path is exactly where this modality has to keep working.
// A failed publish is logged, not silent: continued viewing retries it.
func (deps ServerDeps) enqueueScene3dDerivation(ctx context.Context, root string, record resourceRecord, path, trigger string) {
	if deps.DataAgentJobs == nil {
		return
	}
	if !isSHA256Hex(strings.ToLower(strings.TrimSpace(record.SHA256))) || record.SizeBytes < 0 {
		slog.WarnContext(ctx, "scene derivation skipped without immutable source identity",
			"resource_id", record.FileID, "trigger", trigger)
		return
	}
	now := time.Now()
	if recentScene3dFailure(root, record.FileID, record.SHA256, now) {
		return
	}
	if !scene3dDerivationThrottle.reserve(record.FileID, now) {
		return // enqueued recently; assume a derive is already in flight
	}
	if err := deps.DataAgentJobs.PublishDataAgentJob(ctx, eventbus.DataAgentJob{
		JobID:         domain.NewID("imgjob"),
		OwnerUserID:   record.Principal.UserID,
		OwnerOrgID:    record.Principal.OrgID,
		JobType:       "scene.derive",
		ResourceIDs:   []string{record.FileID},
		ResourceCount: 1,
		Metadata: domain.JSONMap{
			"resource_id":          record.FileID,
			"src_path":             path,
			"dst_dir":              derivedScene3dDir(root, record.FileID, record.SHA256),
			"max_splats_per_chunk": scene3dMaxSplatsPerChunk,
			"tier_count":           scene3dTierCount,
			"preview_splats":       scene3dPreviewSplats,
			"preview_points":       scene3dPreviewPoints,
			"splat_delivery":       "spark-rad-v1",
			"source_sha256":        strings.ToLower(record.SHA256),
			"source_size_bytes":    record.SizeBytes,
		},
	}); err != nil {
		slog.WarnContext(ctx, "scene derivation enqueue failed; will retry on next view",
			"resource_id", record.FileID, "trigger", trigger, "error", err)
	}
}

// --- HTTP -------------------------------------------------------------------

// resolveScene3dRequest authorizes the upload exactly like the CIFTI handlers and
// then re-establishes that the resource really is a scene, so a derived directory
// can never be served for a resource whose source is something else.
func (deps ServerDeps) resolveScene3dRequest(w http.ResponseWriter, r *http.Request) (string, resourceRecord, string, bool) {
	root, record, path, ok := deps.resolveUploadServingRequest(w, r)
	if !ok {
		return "", resourceRecord{}, "", false
	}
	if !resourceIsScene3d(record, path) {
		writeError(w, http.StatusUnsupportedMediaType, errors.New("not a 3D scene file"))
		return "", resourceRecord{}, "", false
	}
	return root, record, path, true
}

// scene3dETag is a strong validator over the artifact's size + mtime. The derive
// writes each artifact once, so this changes only when the scene is re-derived.
func scene3dETag(generation derivativeSourceStat) string {
	return fmt.Sprintf("\"%x-%x\"", generation.SizeBytes, generation.MtimeNS)
}

// handleGetUploadScene3dManifest serves the derived manifest. A missing manifest
// is NOT an error: it means the derive has not landed yet, so the job is
// (re)enqueued and the poll gets 202 with the current status.
func (deps ServerDeps) handleGetUploadScene3dManifest(w http.ResponseWriter, r *http.Request) {
	root, record, path, ok := deps.resolveScene3dRequest(w, r)
	if !ok {
		return
	}
	info, _ := scene3dPeek(record, path)
	if !scene3dCanDerive(record, info) {
		w.Header().Set("Cache-Control", "no-store")
		writeJSON(w, http.StatusAccepted, map[string]any{
			"status":  "failed",
			"file_id": record.FileID,
			"error":   scene3dMessage(info, "failed"),
		})
		return
	}
	file, generation, err := openRegularNoFollow(scene3dManifestPath(root, record.FileID, record.SHA256))
	if err != nil {
		status := "deriving"
		if recentScene3dFailure(root, record.FileID, record.SHA256, time.Now()) {
			// A permanent failure is recorded: report it honestly instead of
			// telling the viewer to keep polling for something that never lands.
			status = "failed"
		} else {
			deps.enqueueScene3dDerivation(r.Context(), root, record, path, "manifest")
		}
		w.Header().Set("Cache-Control", "no-store")
		writeJSON(w, http.StatusAccepted, map[string]any{
			"status":  status,
			"file_id": record.FileID,
		})
		return
	}
	defer func() { _ = file.Close() }()
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("ETag", scene3dETag(generation))
	// Short freshness + a strong validator: a re-derive replaces the manifest, and
	// every chunk request resolves through the current catalog source digest.
	w.Header().Set("Cache-Control", "private, max-age=60")
	http.ServeContent(w, r, scene3dManifestName, time.Unix(0, generation.MtimeNS), file)
}

// parseScene3dChunkIndex accepts a plain non-negative decimal only. Leading zeros
// are rejected so one chunk has exactly one URL (and therefore one cache entry),
// and everything else — "..", "../x", "-1", "0x10", "1e3", " 1" — never becomes
// a path element at all.
func parseScene3dChunkIndex(raw string) (int, error) {
	invalid := errors.New("scene chunk index must be a plain non-negative integer")
	if raw == "" || len(raw) > len(strconv.Itoa(scene3dMaxChunkIndex)) {
		return 0, invalid
	}
	if raw != "0" && raw[0] == '0' {
		return 0, invalid
	}
	for _, char := range raw {
		if char < '0' || char > '9' {
			return 0, invalid
		}
	}
	index, err := strconv.Atoi(raw)
	if err != nil || index < 0 || index > scene3dMaxChunkIndex {
		return 0, invalid
	}
	return index, nil
}

// parseScene3dLodArtifact accepts only the fixed RAD header and the canonical chunk
// names Spark writes into that header. The filename is later joined to a private,
// digest-bound directory, so rejecting every other character is the path boundary.
func parseScene3dLodArtifact(raw string) (string, error) {
	invalid := errors.New("scene LoD artifact name is invalid")
	if raw == scene3dLodHeaderName {
		return raw, nil
	}
	if !strings.HasPrefix(raw, scene3dLodChunkPrefix) || !strings.HasSuffix(raw, scene3dLodChunkSuffix) {
		return "", invalid
	}
	middle := strings.TrimSuffix(strings.TrimPrefix(raw, scene3dLodChunkPrefix), scene3dLodChunkSuffix)
	index, err := parseScene3dChunkIndex(middle)
	if err != nil || raw != fmt.Sprintf("%s%d%s", scene3dLodChunkPrefix, index, scene3dLodChunkSuffix) {
		return "", invalid
	}
	return raw, nil
}

func serveScene3dArtifact(w http.ResponseWriter, r *http.Request, path, displayName string) {
	file, generation, err := openRegularNoFollow(path)
	if err != nil {
		writeError(w, http.StatusNotFound, fmt.Errorf("scene artifact %q is not derived", displayName))
		return
	}
	defer func() { _ = file.Close() }()
	if !scene3dChunkInFlightBudget.tryAcquire(generation.SizeBytes) {
		w.Header().Set("Retry-After", "1")
		writeError(w, http.StatusServiceUnavailable, errScene3dChunkAdmission)
		return
	}
	defer scene3dChunkInFlightBudget.release(generation.SizeBytes)
	contentType := "application/octet-stream"
	if strings.HasSuffix(strings.ToLower(displayName), ".jpg") {
		contentType = "image/jpeg"
	}
	w.Header().Set("Content-Type", contentType)
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("ETag", scene3dETag(generation))
	w.Header().Set("Cache-Control", "private, max-age=60, must-revalidate")
	http.ServeContent(w, r, displayName, time.Unix(0, generation.MtimeNS), file)
}

// handleGetUploadScene3dChunk streams one derived chunk. http.ServeContent gives
// Range and conditional requests for free, which is what makes a resumed or
// partially-cached scene load cheap.
func (deps ServerDeps) handleGetUploadScene3dChunk(w http.ResponseWriter, r *http.Request) {
	root, record, _, ok := deps.resolveScene3dRequest(w, r)
	if !ok {
		return
	}
	index, err := parseScene3dChunkIndex(chi.URLParam(r, "index"))
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	directory := derivedScene3dDir(root, record.FileID, record.SHA256)
	chunkPath := filepath.Join(directory, fmt.Sprintf(scene3dChunkNameFormat, index))
	// Defense in depth: the name is formatted from a validated integer and cannot
	// escape, but assert containment anyway so a future format change can never
	// silently become a traversal.
	if filepath.Dir(chunkPath) != filepath.Clean(directory) {
		writeError(w, http.StatusBadRequest, errors.New("scene chunk index resolves outside the derived scene directory"))
		return
	}
	// Admit the whole chunk even for a Range read: the budget bounds concurrent
	// delivery, and a partial read is a fraction of an admission we already sized.
	serveScene3dArtifact(w, r, chunkPath, filepath.Base(chunkPath))
}

func (deps ServerDeps) handleGetUploadScene3dCameraImage(w http.ResponseWriter, r *http.Request) {
	root, record, _, ok := deps.resolveScene3dRequest(w, r)
	if !ok {
		return
	}
	index, err := parseScene3dChunkIndex(chi.URLParam(r, "index"))
	if err != nil || index >= 64 {
		writeError(w, http.StatusBadRequest, errors.New("scene camera image index is invalid"))
		return
	}
	directory := derivedScene3dDir(root, record.FileID, record.SHA256)
	imagePath := filepath.Join(directory, fmt.Sprintf(scene3dCameraImageName, index))
	if filepath.Dir(imagePath) != filepath.Clean(directory) {
		writeError(w, http.StatusBadRequest, errors.New("scene camera image resolves outside the derived scene directory"))
		return
	}
	serveScene3dArtifact(w, r, imagePath, filepath.Base(imagePath))
}

// handleGetUploadScene3dLodArtifact serves the paged RAD header and the relative RADC
// files it names. Spark supplies Range requests and the same authenticated headers as
// the rest of Lens; this handler never serves a filename not admitted above.
func (deps ServerDeps) handleGetUploadScene3dLodArtifact(w http.ResponseWriter, r *http.Request) {
	root, record, _, ok := deps.resolveScene3dRequest(w, r)
	if !ok {
		return
	}
	name, err := parseScene3dLodArtifact(chi.URLParam(r, "artifact"))
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	directory := derivedScene3dDir(root, record.FileID, record.SHA256)
	artifactPath := filepath.Join(directory, name)
	if filepath.Dir(artifactPath) != filepath.Clean(directory) {
		writeError(w, http.StatusBadRequest, errors.New("scene LoD artifact resolves outside the derived scene directory"))
		return
	}
	serveScene3dArtifact(w, r, artifactPath, name)
}

// writeScene3dViewer returns the kind:"scene3d" descriptor: what the scene is,
// what the header declares, whether the derived stream is ready, and where to
// fetch it. It reports only header-declared facts — the MEASURED spherical-
// harmonic degree comes from the derive job, in the manifest, because measuring
// it means scanning gigabytes and this path never does that.
func (deps ServerDeps) writeScene3dViewer(w http.ResponseWriter, record resourceRecord, info scene3dInfo, path string) {
	fileIDSegment := url.PathEscape(record.FileID)
	status := "deriving"
	if !scene3dCanDerive(record, info) {
		status = "failed"
	} else if root, err := deps.resolvedUploadRoot(); err == nil {
		status = scene3dDeriveStatus(root, record.FileID, record.SHA256, time.Now())
	}
	// Prefer the size on disk over the catalog's copy: it is the number the derive
	// job and the manifest report.
	sizeBytes := record.SizeBytes
	if stat, err := os.Stat(path); err == nil && stat.Mode().IsRegular() {
		sizeBytes = stat.Size()
	}
	source := map[string]any{
		"format": info.format,
		"bytes":  sizeBytes,
	}
	if info.hasPly {
		source["vertex_count"] = info.ply.vertexCount
		source["declared_sh_degree"] = info.ply.declaredSHDegree
		source["property_count"] = info.ply.propertyCount
		source["byte_order"] = info.ply.byteOrder
		source["data_offset"] = info.ply.dataOffset
		if info.ply.strideBytes > 0 {
			source["stride_bytes"] = info.ply.strideBytes
		}
		if info.ply.writer != "" {
			source["writer"] = info.ply.writer
		}
		source["vertex_properties"] = append([]string(nil), info.ply.propertyNames...)
	}
	if info.hasColmap {
		// Layout facts only: which container, where the model root is, how its
		// records are encoded, and which products exist. Camera and point COUNTS are
		// absent on purpose — both require walking a variable-stride file, which is
		// the derive job's work and never this path's.
		source["variant"] = info.colmap.variant
		source["model_count"] = info.colmap.modelCount
		source["model_paths"] = append([]string(nil), info.colmap.modelPaths...)
		if info.colmap.modelCount == 1 {
			source["model_path"] = info.colmap.modelPath
			source["record_format"] = info.colmap.recordFormat
			source["cameras_file"] = info.colmap.camerasName
			source["images_file"] = info.colmap.imagesName
			source["has_points3d"] = info.colmap.hasPoints3D
			if info.colmap.hasPoints3D {
				source["points3d_file"] = info.colmap.points3DName
			}
			source["has_rigs"] = info.colmap.hasRigs
			source["has_frames"] = info.colmap.hasFrames
		}
		if info.colmap.variant == "bundle" {
			source["geometry_member"] = info.colmap.bundlePlyPath
			source["geometry_member_count"] = info.colmap.bundlePlyCount
			source["image_member_count"] = info.colmap.imageMembers
		}
	}
	decodable := scene3dCanDerive(record, info)
	response := map[string]any{
		"kind":          "scene3d",
		"decodable":     decodable,
		"file_id":       record.FileID,
		"original_name": record.OriginalName,
		"format":        info.format,
		"modality":      "scene3d",
		"scene_kind":    info.sceneKind,
		"status":        status,
		"source":        source,
		"limitations":   scene3dLimitations(info, status),
		"service_urls": map[string]any{
			"manifest":     "/v2/uploads/" + fileIDSegment + "/scene3d/manifest",
			"chunk":        "/v2/uploads/" + fileIDSegment + "/scene3d/chunk/{index}",
			"lod":          "/v2/uploads/" + fileIDSegment + "/scene3d/lod/{artifact}",
			"camera_image": "/v2/uploads/" + fileIDSegment + "/scene3d/image/{index}",
			"download":     "/v2/resources/" + fileIDSegment + "/download",
		},
		"message": scene3dMessage(info, status),
	}
	if calibration, ok := scene3dCalibrationForViewer(record); ok {
		response["calibration"] = calibration
	}
	writeJSON(w, http.StatusOK, response)
}

func validateScene3dCalibrationMetadata(value any, sourceSHA string) (map[string]any, int, error) {
	raw, ok := jsonObject(value)
	if !ok || !hasOnlyJSONKeys(
		raw,
		"version",
		"source_sha256",
		"expected_revision",
		"signed_up_axis",
		"handedness",
		"units",
		"units_per_source_unit",
	) {
		return nil, 0, errors.New("3D scene calibration metadata is malformed")
	}
	version, versionOK := jsonInt(raw["version"])
	expectedRevision, revisionOK := jsonInt(raw["expected_revision"])
	calibrationSHA, shaOK := raw["source_sha256"].(string)
	axis, axisOK := raw["signed_up_axis"].(string)
	handedness, handednessOK := raw["handedness"].(string)
	units, unitsOK := raw["units"].(string)
	scale, scaleOK := jsonFiniteFloat(raw["units_per_source_unit"])
	if !versionOK || version != 1 || !revisionOK || expectedRevision < 0 ||
		!shaOK || strings.TrimSpace(calibrationSHA) != strings.TrimSpace(sourceSHA) ||
		!axisOK || !scene3dSignedAxisAllowed(axis) ||
		!handednessOK || (handedness != "right" && handedness != "left") ||
		!unitsOK || !scene3dUnitAllowed(units) ||
		!scaleOK || scale < 1e-12 || scale > 1e12 {
		return nil, 0, errors.New("3D scene calibration metadata is invalid for this source")
	}
	return map[string]any{
		"version":               1,
		"source_sha256":         strings.TrimSpace(calibrationSHA),
		"revision":              expectedRevision + 1,
		"signed_up_axis":        axis,
		"handedness":            handedness,
		"units":                 units,
		"units_per_source_unit": scale,
	}, expectedRevision, nil
}

func scene3dCalibrationForViewer(record resourceRecord) (map[string]any, bool) {
	raw, ok := jsonObject(record.Metadata["ultra_scene3d_calibration_v1"])
	if !ok || !hasOnlyJSONKeys(
		raw,
		"version",
		"source_sha256",
		"revision",
		"signed_up_axis",
		"handedness",
		"units",
		"units_per_source_unit",
	) {
		return nil, false
	}
	version, versionOK := jsonInt(raw["version"])
	revision, revisionOK := jsonInt(raw["revision"])
	calibrationSHA, shaOK := raw["source_sha256"].(string)
	axis, axisOK := raw["signed_up_axis"].(string)
	handedness, handednessOK := raw["handedness"].(string)
	units, unitsOK := raw["units"].(string)
	scale, scaleOK := jsonFiniteFloat(raw["units_per_source_unit"])
	if !versionOK || version != 1 || !revisionOK || revision < 1 ||
		!shaOK || strings.TrimSpace(calibrationSHA) != strings.TrimSpace(record.SHA256) ||
		!axisOK || !scene3dSignedAxisAllowed(axis) ||
		!handednessOK || (handedness != "right" && handedness != "left") ||
		!unitsOK || !scene3dUnitAllowed(units) ||
		!scaleOK || scale < 1e-12 || scale > 1e12 {
		return nil, false
	}
	return map[string]any{
		"version":               1,
		"source_sha256":         strings.TrimSpace(calibrationSHA),
		"revision":              revision,
		"signed_up_axis":        axis,
		"handedness":            handedness,
		"units":                 units,
		"units_per_source_unit": scale,
	}, true
}

func scene3dSignedAxisAllowed(value string) bool {
	switch value {
	case "+x", "-x", "+y", "-y", "+z", "-z":
		return true
	default:
		return false
	}
}

func scene3dUnitAllowed(value string) bool {
	switch value {
	case "arbitrary", "m", "cm", "mm", "um", "nm":
		return true
	default:
		return false
	}
}

func scene3dMessage(info scene3dInfo, status string) string {
	if info.unsupportedReason != "" {
		return info.unsupportedReason
	}
	subject := "Point-cloud scene"
	switch info.sceneKind {
	case "splat":
		subject = "3D Gaussian-splat scene"
	case "colmap":
		subject = "COLMAP reconstruction"
	case "reconstruction":
		subject = "3D reconstruction bundle"
	}
	switch status {
	case "ready":
		return subject + " — streamed as derived, chunked geometry in the source world frame. " +
			"The original file is never sent to the browser; download it to open it in a desktop tool."
	case "failed":
		if info.hasColmap && info.colmap.modelCount > 1 {
			return subject + " — this archive contains multiple COLMAP models, so Lens does not choose one. " +
				"Export or upload one reconstruction per archive."
		}
		if info.hasColmap && info.colmap.variant == "bundle" && info.colmap.bundlePlyCount != 1 {
			return subject + " — this archive must contain exactly one primary PLY geometry member."
		}
		if info.hasColmap && info.colmap.variant == "directory" {
			return subject + " — archive the model directory as a zip before uploading it. " +
				"That gives every camera and point record one immutable source identity for accurate derivation."
		}
		return subject + " — preparing the streamable scene failed. Download the original to open it in a desktop tool."
	default:
		return subject + " — the streamable scene is still being prepared. It renders as soon as the derived manifest lands."
	}
}

// scene3dLimitations is the CIFTI honesty field: plain sentences stating what the
// viewer is NOT doing, rendered verbatim in the provenance panel.
func scene3dLimitations(info scene3dInfo, status string) []string {
	limitations := []string{}
	if info.unsupportedReason != "" {
		return []string{info.unsupportedReason}
	}
	switch status {
	case "deriving":
		limitations = append(limitations, "The streamable scene is still being derived; nothing is rendered until its manifest exists.")
	case "failed":
		if !(info.hasColmap && (info.colmap.variant == "directory" || info.colmap.modelCount > 1)) {
			limitations = append(limitations, "Deriving the streamable scene failed permanently for this file; nothing is rendered.")
		}
	}
	if info.hasColmap {
		if info.colmap.modelCount > 1 {
			limitations = append(limitations,
				fmt.Sprintf("This resource contains %d COLMAP models. Lens reports every model root and does not choose one because they may represent different reconstructions.", info.colmap.modelCount),
				"Upload one reconstruction per archive to derive and render it.",
			)
			return limitations
		}
		limitations = append(limitations,
			"The reconstruction is recognized from its layout alone; no camera, image or point record is read until the derive job runs.")
		if info.colmap.variant == "bundle" {
			limitations = append(limitations,
				"This reconstruction bundle was recognized from one COLMAP model and one PLY geometry member; member bytes are validated only by the derive job.",
				"Source-image previews are published only when an archive member uniquely matches a registered COLMAP image name; ambiguous names are never guessed.",
			)
			if info.colmap.bundlePlyCount != 1 {
				limitations = append(limitations,
					fmt.Sprintf("The archive contains %d PLY geometry members. Lens requires exactly one and does not choose among them.", info.colmap.bundlePlyCount),
				)
				return limitations
			}
		} else if info.colmap.variant == "zip" {
			limitations = append(limitations,
				"This model is a zip archive: only its central directory was read here, so nothing inside it has been decompressed or validated yet.")
		} else if info.colmap.variant == "directory" {
			limitations = append(limitations,
				"Directory models are not derived because a mutable directory has no single cataloged byte stream to source-bind; archive this model as a zip and upload the archive.")
		}
		limitations = append(limitations,
			"Per-image 2D feature observations are skipped entirely; only camera poses and, where present, the 3D points are used.")
		if info.colmap.hasPoints3D {
			limitations = append(limitations,
				"COLMAP points are drawn as points; no surface reconstruction, meshing or normal shading is performed.")
		} else {
			limitations = append(limitations,
				"This model has no points3D file, so only the posed cameras are drawn — there is no scene geometry to render.")
		}
	}
	if info.hasPly && info.ply.declaredSHDegree > 0 {
		limitations = append(limitations, fmt.Sprintf(
			"The header declares spherical-harmonic degree %d. That is an allocation, not a measurement — the derived manifest reports the measured degree, which is often lower.",
			info.ply.declaredSHDegree,
		))
	}
	if info.sceneKind == "pointcloud" {
		limitations = append(limitations, "Points are drawn from the source coordinates; no surface reconstruction, meshing or normal shading is performed.")
	}
	limitations = append(limitations, "The scene keeps its source world frame: geometry is never rotated, and \"up\" is only a hint applied to the camera.")
	return limitations
}
