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
	"path/filepath"
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
//	{root}/derived/{file_id}__scene3d/manifest.json      the ultra.scene3d.v1 manifest
//	{root}/derived/{file_id}__scene3d/chunk_{n:05d}.bin  USX1 (splats) / UPC1 (points)
//	{root}/derived/{file_id}__scene3d.failed             permanent-failure marker
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

	// scene3dMaxChunkIndex caps the addressable chunk space. At the default
	// 200k elements per chunk this is 2e11 elements — far past any real scene —
	// so anything larger is a malformed or hostile request, not a deep scene.
	scene3dMaxChunkIndex = 999_999

	// Derive-job parameters (contract §5/§7). max_splats_per_chunk is the frozen
	// default; tier_count is not fixed by the contract, and this is the value the
	// control plane requests.
	scene3dMaxSplatsPerChunk = 200_000
	scene3dTierCount         = 3

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
	writer           string // parsed from PLY comments when recognizable
}

// scene3dInfo is the viewer-facing summary of a scene resource: the container
// format plus, for PLY, the header facts and, for COLMAP, the model layout. The
// compact splat containers (.spz/.splat/.ksplat/.sog) carry no ASCII header we
// can sniff cheaply, so they are classified by extension and inspected only by
// the derive job.
type scene3dInfo struct {
	format    string // "ply" | "splat" | "spz" | "ksplat" | "sog" | "colmap"
	sceneKind string // "splat" | "pointcloud" | "colmap"
	ply       plyInfo
	hasPly    bool
	colmap    colmapInfo
	hasColmap bool
}

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
// vertex element's stride and data offset FROM THE HEADER. It returns ok=false
// for anything it cannot describe exactly: a non-PLY magic, a header that does
// not fit the peek window, a vertex element that is not first (its data would
// then start after elements whose sizes we refuse to assume), or a vertex list
// property (no fixed stride).
func parsePlyHeader(head []byte) (plyInfo, bool) {
	if !bytes.HasPrefix(head, []byte("ply")) {
		return plyInfo{}, false // cheap rejection: not a PLY at all
	}
	info := plyInfo{}
	var properties []plyProperty
	element := ""
	elementIndex := -1
	vertexElementIndex := -1
	sawMagic := false
	offset := 0
	for offset < len(head) {
		newline := bytes.IndexByte(head[offset:], '\n')
		if newline < 0 {
			return plyInfo{}, false // header incomplete inside the peek window
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
				return plyInfo{}, false
			}
			sawMagic = true
		case "format":
			if len(fields) < 2 {
				return plyInfo{}, false
			}
			info.byteOrder = fields[1]
		case "comment":
			if writer := plyWriterFromComment(line); writer != "" && info.writer == "" {
				info.writer = writer
			}
		case "element":
			if len(fields) != 3 {
				return plyInfo{}, false
			}
			count, err := strconv.ParseInt(fields[2], 10, 64)
			if err != nil || count < 0 {
				return plyInfo{}, false
			}
			elementIndex++
			element = fields[1]
			if element == "vertex" {
				if vertexElementIndex >= 0 {
					return plyInfo{}, false // two vertex elements: ambiguous
				}
				vertexElementIndex = elementIndex
				info.vertexCount = count
			}
		case "property":
			if element != "vertex" {
				continue // other elements (faces, …) do not affect the vertex layout
			}
			if len(fields) >= 2 && fields[1] == "list" {
				return plyInfo{}, false // variable-length vertex record: no stride
			}
			if len(fields) != 3 {
				return plyInfo{}, false
			}
			size, known := plyScalarSizes[fields[1]]
			if !known {
				return plyInfo{}, false
			}
			properties = append(properties, plyProperty{name: fields[2], size: size})
		case "end_header":
			if !sawMagic || vertexElementIndex != 0 || len(properties) == 0 {
				return plyInfo{}, false
			}
			info.dataOffset = int64(offset)
			info.propertyCount = len(properties)
			for _, property := range properties {
				info.strideBytes += property.size
			}
			if info.byteOrder == "ascii" {
				info.strideBytes = 0 // ascii records are not fixed-width
			} else if info.byteOrder != "binary_little_endian" && info.byteOrder != "binary_big_endian" {
				return plyInfo{}, false
			}
			species, classified := classifyPlyProperties(properties)
			if !classified {
				return plyInfo{}, false
			}
			info.species = species
			info.restCount, info.declaredSHDegree = plyDeclaredSHDegree(properties)
			return info, true
		}
	}
	return plyInfo{}, false
}

// classifyPlyProperties names the species from the property set. Splats carry
// the INRIA parameter block (DC colour, opacity, log scales, rotation quat);
// anything else with coordinates is a point cloud.
func classifyPlyProperties(properties []plyProperty) (string, bool) {
	present := make(map[string]bool, len(properties))
	for _, property := range properties {
		present[property.name] = true
	}
	if !present["x"] || !present["y"] || !present["z"] {
		return "", false // not a scene we can place in space
	}
	if present["f_dc_0"] && present["opacity"] && present["scale_0"] && present["rot_0"] {
		return "splat", true
	}
	return "pointcloud", true
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
	file, err := os.Open(path)
	if err != nil {
		return plyInfo{}, false
	}
	defer func() { _ = file.Close() }()
	stat, err := file.Stat()
	if err != nil || !stat.Mode().IsRegular() {
		return plyInfo{}, false
	}
	head := make([]byte, plyHeaderPeekMaxBytes)
	read, err := io.ReadFull(file, head)
	if read == 0 || (err != nil && err != io.ErrUnexpectedEOF && err != io.EOF) {
		return plyInfo{}, false
	}
	info, ok := parsePlyHeader(head[:read])
	if !ok {
		return plyInfo{}, false
	}
	if info.strideBytes > 0 {
		payloadBytes, exact := checkedNonNegativeProduct(info.vertexCount, int64(info.strideBytes))
		if !exact || payloadBytes > stat.Size()-info.dataOffset {
			return plyInfo{}, false // truncated: the declared vertices are not on disk
		}
	}
	return info, true
}

// --- COLMAP model recognition ------------------------------------------------

// colmapInfo is everything the bounded probe can PROVE about a COLMAP
// reconstruction: which container it arrived in, where inside it the model root
// sits, whether the records are binary or text, and which optional products are
// present. Nothing here is parsed from a record — see the file header for why.
type colmapInfo struct {
	variant      string // "directory" | "zip"
	modelPath    string // slash-separated model root relative to the resource ("" = at the root)
	recordFormat string // "bin" | "txt" | "mixed"
	camerasName  string // the member name as it actually appears
	imagesName   string
	points3DName string
	hasPoints3D  bool
	hasRigs      bool // modern COLMAP writes rigs/frames; a legacy model has neither
	hasFrames    bool
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
			return info, true
		}
	}
	return colmapInfo{}, false
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
	for _, entry := range reader.File {
		directory, base, valid := colmapZipEntryPath(entry.Name)
		if !valid {
			continue
		}
		lower := strings.ToLower(base)
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
	best, found := colmapInfo{}, false
	for directory, names := range members {
		lookup := func(candidate string) (string, bool) {
			actual, present := names[strings.ToLower(candidate)]
			return actual, present
		}
		info, ok := colmapModelFromMembers(directory, lookup)
		if !ok {
			continue
		}
		// Deterministic choice: the shallowest model wins (an archive that wraps a
		// model in one folder is the common case), ties broken lexicographically, so
		// the same archive always reports the same model root.
		if !found || colmapModelRootPrecedes(info.modelPath, best.modelPath) {
			best, found = info, true
		}
	}
	if !found {
		return colmapInfo{}, false
	}
	best.variant = "zip"
	return best, true
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
// containers. Extension is the only signal for the compact formats; PLY is
// additionally header-verified by scene3dPeek. COLMAP is deliberately absent: a
// reconstruction has no distinguishing name at all (it is a folder, often just
// "sparse" or the dataset's own name), so it is recognized STRUCTURALLY by
// colmapPeek instead.
func isScene3dName(name string) bool {
	return scene3dFormatFromName(name) != ""
}

func scene3dFormatFromName(name string) string {
	lower := strings.ToLower(strings.TrimSpace(name))
	for _, ext := range []string{".ply", ".splat", ".spz", ".ksplat", ".sog"} {
		if strings.HasSuffix(lower, ext) {
			return strings.TrimPrefix(ext, ".")
		}
	}
	return ""
}

// scene3dPeek classifies a resource as a 3D scene. PLY must parse — a file named
// .ply whose header we cannot describe falls through to the normal ladder and
// ends up as an honest "unsupported" descriptor rather than an empty canvas.
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
			return scene3dInfo{format: "colmap", sceneKind: "colmap", colmap: colmap, hasColmap: true}, true
		}
		return scene3dInfo{}, false
	}
	if format != "ply" {
		// .spz/.splat/.ksplat/.sog are compact binary splat containers; they are
		// always splats, and their contents are inspected only by the derive job.
		return scene3dInfo{format: format, sceneKind: "splat"}, true
	}
	ply, ok := plyPeek(path)
	if !ok {
		return scene3dInfo{}, false
	}
	return scene3dInfo{format: format, sceneKind: ply.species, ply: ply, hasPly: true}, true
}

func resourceIsScene3d(record resourceRecord, path string) bool {
	_, ok := scene3dPeek(record, path)
	return ok
}

// --- derived artifacts -------------------------------------------------------

// derivedScene3dName is the deterministic destination the derive job writes into,
// mirroring derivedPyramidName. It is a DIRECTORY (manifest + chunks + poster).
func derivedScene3dName(fileID string) string { return fileID + "__scene3d" }

func derivedScene3dDir(root, fileID string) string {
	return filepath.Join(root, "derived", derivedScene3dName(fileID))
}

// derivedScene3dFailedMarkerPath is "{dst_dir}.failed" (contract §7): the sidecar
// a permanently-failed derive writes, which the control plane honours as backoff.
func derivedScene3dFailedMarkerPath(root, fileID string) string {
	return derivedScene3dDir(root, fileID) + ".failed"
}

func scene3dManifestPath(root, fileID string) string {
	return filepath.Join(derivedScene3dDir(root, fileID), scene3dManifestName)
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
func recentScene3dFailure(root, fileID string, now time.Time) bool {
	window := scene3dFailureBackoffWindow()
	if window <= 0 {
		return false
	}
	info, err := regularFileInfo(derivedScene3dFailedMarkerPath(root, fileID))
	if err != nil {
		return false
	}
	return now.Sub(info.ModTime()) < window
}

// scene3dDeriveStatus is the descriptor's "status": "ready" once the manifest
// exists, "failed" while a permanent-failure marker is fresh, "deriving" otherwise.
func scene3dDeriveStatus(root, fileID string, now time.Time) string {
	if _, err := regularFileInfo(scene3dManifestPath(root, fileID)); err == nil {
		return "ready"
	}
	if recentScene3dFailure(root, fileID, now) {
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
	now := time.Now()
	if recentScene3dFailure(root, record.FileID, now) {
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
			"dst_dir":              derivedScene3dDir(root, record.FileID),
			"max_splats_per_chunk": scene3dMaxSplatsPerChunk,
			"tier_count":           scene3dTierCount,
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
	file, generation, err := openRegularNoFollow(scene3dManifestPath(root, record.FileID))
	if err != nil {
		status := "deriving"
		if recentScene3dFailure(root, record.FileID, time.Now()) {
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
	// the chunk URLs it points at are themselves immutable.
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
	directory := derivedScene3dDir(root, record.FileID)
	chunkPath := filepath.Join(directory, fmt.Sprintf(scene3dChunkNameFormat, index))
	// Defense in depth: the name is formatted from a validated integer and cannot
	// escape, but assert containment anyway so a future format change can never
	// silently become a traversal.
	if filepath.Dir(chunkPath) != filepath.Clean(directory) {
		writeError(w, http.StatusBadRequest, errors.New("scene chunk index resolves outside the derived scene directory"))
		return
	}
	file, generation, err := openRegularNoFollow(chunkPath)
	if err != nil {
		writeError(w, http.StatusNotFound, fmt.Errorf("scene chunk %d is not derived", index))
		return
	}
	defer func() { _ = file.Close() }()
	// Admit the whole chunk even for a Range read: the budget bounds concurrent
	// delivery, and a partial read is a fraction of an admission we already sized.
	if !scene3dChunkInFlightBudget.tryAcquire(generation.SizeBytes) {
		w.Header().Set("Retry-After", "1")
		writeError(w, http.StatusServiceUnavailable, errScene3dChunkAdmission)
		return
	}
	defer scene3dChunkInFlightBudget.release(generation.SizeBytes)
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("ETag", scene3dETag(generation))
	// Chunks are write-once artifacts of an immutable source: a re-derive writes a
	// new manifest, and the viewer re-reads the chunk list from it.
	w.Header().Set("Cache-Control", "private, max-age=31536000, immutable")
	http.ServeContent(w, r, filepath.Base(chunkPath), time.Unix(0, generation.MtimeNS), file)
}

// writeScene3dViewer returns the kind:"scene3d" descriptor: what the scene is,
// what the header declares, whether the derived stream is ready, and where to
// fetch it. It reports only header-declared facts — the MEASURED spherical-
// harmonic degree comes from the derive job, in the manifest, because measuring
// it means scanning gigabytes and this path never does that.
func (deps ServerDeps) writeScene3dViewer(w http.ResponseWriter, record resourceRecord, info scene3dInfo, path string) {
	fileIDSegment := url.PathEscape(record.FileID)
	status := "deriving"
	if root, err := deps.resolvedUploadRoot(); err == nil {
		status = scene3dDeriveStatus(root, record.FileID, time.Now())
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
	}
	if info.hasColmap {
		// Layout facts only: which container, where the model root is, how its
		// records are encoded, and which products exist. Camera and point COUNTS are
		// absent on purpose — both require walking a variable-stride file, which is
		// the derive job's work and never this path's.
		source["variant"] = info.colmap.variant
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
	writeJSON(w, http.StatusOK, map[string]any{
		"kind":          "scene3d",
		"decodable":     true,
		"file_id":       record.FileID,
		"original_name": record.OriginalName,
		"format":        info.format,
		"modality":      "scene3d",
		"scene_kind":    info.sceneKind,
		"status":        status,
		"source":        source,
		"limitations":   scene3dLimitations(info, status),
		"service_urls": map[string]any{
			"manifest": "/v2/uploads/" + fileIDSegment + "/scene3d/manifest",
			"chunk":    "/v2/uploads/" + fileIDSegment + "/scene3d/chunk/{index}",
			"download": "/v2/resources/" + fileIDSegment + "/download",
		},
		"message": scene3dMessage(info, status),
	})
}

func scene3dMessage(info scene3dInfo, status string) string {
	subject := "Point-cloud scene"
	switch info.sceneKind {
	case "splat":
		subject = "3D Gaussian-splat scene"
	case "colmap":
		subject = "COLMAP reconstruction"
	}
	switch status {
	case "ready":
		return subject + " — streamed as derived, chunked geometry in the source world frame. " +
			"The original file is never sent to the browser; download it to open it in a desktop tool."
	case "failed":
		return subject + " — preparing the streamable scene failed. Download the original to open it in a desktop tool."
	default:
		return subject + " — the streamable scene is still being prepared. It renders as soon as the derived manifest lands."
	}
}

// scene3dLimitations is the CIFTI honesty field: plain sentences stating what the
// viewer is NOT doing, rendered verbatim in the provenance panel.
func scene3dLimitations(info scene3dInfo, status string) []string {
	limitations := []string{}
	switch status {
	case "deriving":
		limitations = append(limitations, "The streamable scene is still being derived; nothing is rendered until its manifest exists.")
	case "failed":
		limitations = append(limitations, "Deriving the streamable scene failed permanently for this file; nothing is rendered.")
	}
	if !info.hasPly && !info.hasColmap {
		limitations = append(limitations, strings.ToUpper(info.format)+" is classified by file extension alone; its contents are inspected only by the derive job.")
	}
	if info.hasColmap {
		limitations = append(limitations,
			"The reconstruction is recognized from its layout alone; no camera, image or point record is read until the derive job runs.")
		if info.colmap.variant == "zip" {
			limitations = append(limitations,
				"This model is a zip archive: only its central directory was read here, so nothing inside it has been decompressed or validated yet.")
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
