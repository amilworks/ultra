package httpapi

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

// nifti2Config builds a configurable little-endian NIfTI-2 header. dims are the
// 1-based dimension sizes (dim[1..]); datatype defaults to uint16.
type nifti2Config struct {
	dim        []int64 // dim[1], dim[2], ...
	datatype   int16
	bitpix     int16
	voxOffset  int64
	sformCode  int32
	srow       [12]float64 // row-major 3x4
	intentCode int32
	xyztUnits  int32
}

func buildNifti2Header(cfg nifti2Config) []byte {
	if cfg.datatype == 0 {
		cfg.datatype = 512 // uint16
		cfg.bitpix = 16
	}
	if cfg.voxOffset == 0 {
		cfg.voxOffset = nifti2HeaderReadSize
	}
	// Size the payload from the SPATIAL dims (dim[1..3]) × bytes-per-voxel — one
	// timepoint's worth, which is all the volume path reads. CIFTI's spatial dims
	// are 1×1×1 so its file stays tiny (the huge matrix dims live in dim[5..6],
	// which the peek path never materializes).
	bytesPerVoxel := int64(2)
	switch cfg.datatype {
	case 2:
		bytesPerVoxel = 1
	case 16:
		bytesPerVoxel = 4
	}
	spatial := int64(1)
	for i := 0; i < len(cfg.dim) && i < 3; i++ {
		if cfg.dim[i] > 0 {
			spatial *= cfg.dim[i]
		}
	}
	total := cfg.voxOffset + spatial*bytesPerVoxel + 16
	raw := make([]byte, total)
	le := binary.LittleEndian
	putI16 := func(off int, v int16) { le.PutUint16(raw[off:off+2], uint16(v)) }
	putI32 := func(off int, v int32) { le.PutUint32(raw[off:off+4], uint32(v)) }
	putI64 := func(off int, v int64) { le.PutUint64(raw[off:off+8], uint64(v)) }
	putF64 := func(off int, v float64) { le.PutUint64(raw[off:off+8], math.Float64bits(v)) }

	putI32(0, nifti2HeaderSize)
	copy(raw[4:12], []byte("n+2\x00\r\n\x1a\n"))
	putI16(12, cfg.datatype)
	putI16(14, cfg.bitpix)
	dim0 := int64(len(cfg.dim))
	putI64(16, dim0)
	for i, d := range cfg.dim {
		putI64(16+(i+1)*8, d)
	}
	for i := 1; i <= 3; i++ {
		putF64(104+i*8, 1.0) // pixdim spacing = 1mm
	}
	putI64(168, cfg.voxOffset)
	putF64(176, 1.0) // scl_slope
	putF64(184, 0.0) // scl_inter
	putI32(348, cfg.sformCode)
	for i := 0; i < 4; i++ {
		putF64(400+i*8, cfg.srow[0*4+i])
		putF64(432+i*8, cfg.srow[1*4+i])
		putF64(464+i*8, cfg.srow[2*4+i])
	}
	putI32(500, cfg.xyztUnits)
	putI32(504, cfg.intentCode)
	return raw
}

func TestNiftiHeaderVersion(t *testing.T) {
	t.Parallel()
	// NIfTI-1: sizeof_hdr 348.
	h1 := make([]byte, 4)
	binary.LittleEndian.PutUint32(h1, 348)
	if _, v, err := niftiHeaderVersion(h1); err != nil || v != 1 {
		t.Fatalf("NIfTI-1 detection = (%d, %v), want (1, nil)", v, err)
	}
	// NIfTI-2: sizeof_hdr 540.
	h2 := make([]byte, 4)
	binary.LittleEndian.PutUint32(h2, 540)
	if _, v, err := niftiHeaderVersion(h2); err != nil || v != 2 {
		t.Fatalf("NIfTI-2 detection = (%d, %v), want (2, nil)", v, err)
	}
	// Big-endian NIfTI-2.
	h2be := make([]byte, 4)
	binary.BigEndian.PutUint32(h2be, 540)
	if order, v, err := niftiHeaderVersion(h2be); err != nil || v != 2 || order != binary.BigEndian {
		t.Fatalf("big-endian NIfTI-2 = (%v, %d, %v), want (BE, 2, nil)", order, v, err)
	}
	// Garbage.
	if _, _, err := niftiHeaderVersion([]byte{1, 2, 3, 4}); err == nil {
		t.Fatalf("garbage header should error")
	}
}

func TestParseNifti2SpatialGeometry(t *testing.T) {
	t.Parallel()
	cfg := nifti2Config{
		dim:       []int64{40, 48, 32}, // W, H, D
		sformCode: 1,
		srow:      [12]float64{2, 0, 0, -80, 0, 2, 0, -96, 0, 0, 3, -64},
		xyztUnits: 2, // mm
	}
	geom, err := parseNiftiGeometry(buildNifti2Header(cfg))
	if err != nil {
		t.Fatalf("parseNiftiGeometry(NIfTI-2) failed: %v", err)
	}
	if geom.width != 40 || geom.height != 48 || geom.depth != 32 {
		t.Fatalf("dims = %dx%dx%d, want 40x48x32", geom.width, geom.height, geom.depth)
	}
	if geom.dtype != "uint16" || geom.bytesPerVoxel != 2 {
		t.Fatalf("dtype = %s/%d, want uint16/2", geom.dtype, geom.bytesPerVoxel)
	}
	if geom.affineCode != 3 {
		t.Fatalf("affineCode = %d, want 3 (sform)", geom.affineCode)
	}
	// sform's diagonal spacing (2,2,3) must survive the 64-bit read.
	if geom.affine[0] != 2 || geom.affine[5] != 2 || geom.affine[10] != 3 {
		t.Fatalf("affine diagonal = %v %v %v, want 2 2 3", geom.affine[0], geom.affine[5], geom.affine[10])
	}
	if geom.spaceUnit != "mm" {
		t.Fatalf("spaceUnit = %q, want mm", geom.spaceUnit)
	}
}

func TestReadNiftiHeaderBytesConsumedPerVersion(t *testing.T) {
	t.Parallel()
	// NIfTI-2 stream consumes the full 544 bytes.
	h2 := buildNifti2Header(nifti2Config{dim: []int64{8, 8, 8}})
	_, consumed, err := readNiftiHeaderBytes(bytes.NewReader(h2))
	if err != nil || consumed != nifti2HeaderReadSize {
		t.Fatalf("NIfTI-2 consumed = (%d, %v), want (%d, nil)", consumed, err, nifti2HeaderReadSize)
	}
	// NIfTI-1 stream consumes only 352, so a small vox_offset is never over-read.
	h1 := buildNiftiTestHeader(niftiTestHeader{w: 2, h: 2, d: 2})
	_, consumed1, err := readNiftiHeaderBytes(bytes.NewReader(h1))
	if err != nil || consumed1 != niftiHeaderReadSize {
		t.Fatalf("NIfTI-1 consumed = (%d, %v), want (%d, nil)", consumed1, err, niftiHeaderReadSize)
	}
}

func TestCiftiDetection(t *testing.T) {
	t.Parallel()
	// A CIFTI dense timeseries: intent code 3002, trivial spatial dims, matrix in
	// the last dims.
	header := buildNifti2Header(nifti2Config{
		dim:        []int64{1, 1, 1, 1, 1200, 91282},
		intentCode: 3002,
	})
	order, _, _ := niftiHeaderVersion(header)
	info, ok := niftiCiftiFromHeader(order, header, "rfMRI_REST1.dtseries.nii")
	if !ok {
		t.Fatalf("dtseries not detected as CIFTI")
	}
	if info.label != "dense timeseries" {
		t.Fatalf("label = %q, want dense timeseries", info.label)
	}
	if len(info.matrixDims) != 2 || info.matrixDims[0] != 1200 || info.matrixDims[1] != 91282 {
		t.Fatalf("matrixDims = %v, want [1200 91282]", info.matrixDims)
	}
	// Filename fallback (no intent code) still catches it.
	plain := buildNifti2Header(nifti2Config{dim: []int64{1, 1, 1, 1, 5, 7}})
	if _, ok := niftiCiftiFromHeader(order, plain, "scan.dscalar.nii"); !ok {
		t.Fatalf("dscalar name should be detected as CIFTI")
	}
	// A real NIfTI-2 spatial volume is NOT CIFTI.
	vol := buildNifti2Header(nifti2Config{dim: []int64{40, 48, 32}})
	if _, ok := niftiCiftiFromHeader(order, vol, "brain.nii"); ok {
		t.Fatalf("plain NIfTI-2 volume misclassified as CIFTI")
	}
}

func TestCiftiPeekAndViewerDescriptor(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii")
	header := buildNifti2Header(nifti2Config{
		dim:        []int64{1, 1, 1, 1, 1200, 91282},
		intentCode: 3002,
	})
	if err := os.WriteFile(path, header, 0o600); err != nil {
		t.Fatalf("write CIFTI fixture: %v", err)
	}
	info, ok := niftiCiftiPeek(path, "rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii")
	if !ok {
		t.Fatalf("niftiCiftiPeek did not detect the CIFTI file")
	}

	// The viewer descriptor is kind:"cifti" (carpet/connectivity views). A
	// hand-built header has no CIFTI XML, so parseCiftiMeta degrades to the carpet
	// view; the richer type-aware descriptor is covered in cifti_test.go with real
	// nibabel fixtures.
	rec := httptest.NewRecorder()
	ServerDeps{}.writeCiftiViewer(rec, resourceRecord{FileID: "file_cifti", OriginalName: "rest.dtseries.nii"}, info, path)
	if rec.Code != http.StatusOK {
		t.Fatalf("viewer status = %d, want 200", rec.Code)
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode viewer body: %v", err)
	}
	if body["kind"] != "cifti" || body["decodable"] != true {
		t.Fatalf("viewer kind/decodable = %v/%v, want cifti/true", body["kind"], body["decodable"])
	}
	views, _ := body["views"].([]any)
	if len(views) == 0 {
		t.Fatalf("viewer must advertise at least one view, got %v", body["views"])
	}
	msg, _ := body["message"].(string)
	if !bytes.Contains([]byte(msg), []byte("CIFTI")) {
		t.Fatalf("viewer message not informative: %q", msg)
	}
}

// TestV2Nifti2VolumeAndCiftiViewer drives the real router: a NIfTI-2 spatial
// volume opens in the volume viewer; a CIFTI file returns the honest unsupported
// descriptor. Both previously failed the NIfTI-1-only header check.
func TestV2Nifti2VolumeAndCiftiViewer(t *testing.T) {
	t.Parallel()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: t.TempDir(),
	})

	upload := func(name string, data []byte) string {
		t.Helper()
		var body bytes.Buffer
		writer := multipart.NewWriter(&body)
		part, _ := writer.CreateFormFile("files", name)
		_, _ = part.Write(data)
		_ = writer.Close()
		req := httptest.NewRequest(http.MethodPost, "/v2/uploads", &body)
		req.Header.Set("Content-Type", writer.FormDataContentType())
		setProxyOwnerHeaders(req)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("upload %s: status=%d body=%s", name, rec.Code, rec.Body.String())
		}
		var resp struct {
			Uploaded []struct {
				FileID string `json:"file_id"`
			} `json:"uploaded"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil || len(resp.Uploaded) != 1 {
			t.Fatalf("upload %s: bad response %s", name, rec.Body.String())
		}
		return resp.Uploaded[0].FileID
	}
	viewer := func(fileID string) map[string]any {
		t.Helper()
		req := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/viewer", nil)
		setProxyOwnerHeaders(req)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("viewer status=%d body=%s", rec.Code, rec.Body.String())
		}
		var out map[string]any
		if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
			t.Fatalf("decode viewer: %v", err)
		}
		return out
	}

	volID := upload("brain_nifti2.nii", buildNifti2Header(nifti2Config{
		dim: []int64{40, 48, 32}, sformCode: 1,
		srow: [12]float64{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0},
	}))
	vol := viewer(volID)
	axes, _ := vol["axis_sizes"].(map[string]any)
	if axes == nil || axes["X"] != float64(40) || axes["Y"] != float64(48) || axes["Z"] != float64(32) {
		t.Fatalf("NIfTI-2 volume axis_sizes = %v, want X40 Y48 Z32", axes)
	}

	ciftiID := upload("rest.dtseries.nii", buildNifti2Header(nifti2Config{
		dim: []int64{1, 1, 1, 1, 100, 200}, intentCode: 3002,
	}))
	cifti := viewer(ciftiID)
	if cifti["kind"] != "cifti" {
		t.Fatalf("CIFTI viewer kind = %v, want cifti", cifti["kind"])
	}
}
