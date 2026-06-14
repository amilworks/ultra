package httpapi

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// writeUint16Nifti writes a NIfTI-1 file with the given dimension counts. ndim
// selects how many dims are declared (4 => time in dim[4], 5 => channels in
// dim[5]); values are laid out column-major (one contiguous slab per volume).
func writeUint16Nifti(t *testing.T, path string, ndim, w, h, d, timeCount, channelCount int, values []uint16, gzipped bool) {
	t.Helper()
	const voxOffset = 352
	raw := make([]byte, voxOffset+len(values)*2)
	binary.LittleEndian.PutUint32(raw[0:4], 348)
	binary.LittleEndian.PutUint16(raw[40:42], uint16(ndim))
	binary.LittleEndian.PutUint16(raw[42:44], uint16(w))
	binary.LittleEndian.PutUint16(raw[44:46], uint16(h))
	binary.LittleEndian.PutUint16(raw[46:48], uint16(d))
	binary.LittleEndian.PutUint16(raw[48:50], uint16(timeCount))
	binary.LittleEndian.PutUint16(raw[50:52], uint16(channelCount))
	binary.LittleEndian.PutUint16(raw[70:72], 512) // DT_UINT16
	binary.LittleEndian.PutUint16(raw[72:74], 16)
	for axis := 1; axis <= 3; axis++ {
		binary.LittleEndian.PutUint32(raw[76+axis*4:80+axis*4], math.Float32bits(1))
	}
	binary.LittleEndian.PutUint32(raw[108:112], math.Float32bits(float32(voxOffset)))
	copy(raw[344:348], []byte{'n', '+', '1', 0})
	for i, v := range values {
		binary.LittleEndian.PutUint16(raw[voxOffset+i*2:voxOffset+i*2+2], v)
	}
	if gzipped {
		var buf bytes.Buffer
		zw := gzip.NewWriter(&buf)
		if _, err := zw.Write(raw); err != nil {
			t.Fatalf("gzip write: %v", err)
		}
		if err := zw.Close(); err != nil {
			t.Fatalf("gzip close: %v", err)
		}
		raw = buf.Bytes()
	}
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatalf("write nifti: %v", err)
	}
}

func sliceUint16(b []byte) []uint16 {
	out := make([]uint16, len(b)/2)
	for i := range out {
		out[i] = binary.LittleEndian.Uint16(b[i*2 : i*2+2])
	}
	return out
}

func TestLoadNiftiScalarVolumeAtExtractsExactTimepoint(t *testing.T) {
	t.Setenv("ULTRA_CONTROL_NIFTI_DECOMPRESS_CACHE", "0")
	dir := t.TempDir()
	path := filepath.Join(dir, "series.nii")
	// 2x1x2 spatial, 3 timepoints; each timepoint is a distinct slab.
	values := []uint16{
		1, 2, 3, 4, // t0
		11, 12, 13, 14, // t1
		21, 22, 23, 24, // t2
	}
	writeUint16Nifti(t, path, 4, 2, 1, 2, 3, 1, values, false)

	for ti, want := range [][]uint16{{1, 2, 3, 4}, {11, 12, 13, 14}, {21, 22, 23, 24}} {
		vol, err := loadNiftiScalarVolumeAt(path, ti, 0)
		if err != nil {
			t.Fatalf("t=%d: %v", ti, err)
		}
		if vol.TimeCount != 3 || vol.TimeIndex != ti || vol.ChannelCount != 1 {
			t.Fatalf("t=%d geometry = count:%d idx:%d ch:%d", ti, vol.TimeCount, vol.TimeIndex, vol.ChannelCount)
		}
		if got := sliceUint16(vol.Data); !equalUint16(got, want) {
			t.Fatalf("t=%d data = %v, want %v", ti, got, want)
		}
	}

	// Out-of-range timepoints clamp to the last instead of reading past the file.
	vol, err := loadNiftiScalarVolumeAt(path, 99, 0)
	if err != nil {
		t.Fatalf("clamp: %v", err)
	}
	if vol.TimeIndex != 2 || !equalUint16(sliceUint16(vol.Data), []uint16{21, 22, 23, 24}) {
		t.Fatalf("clamp returned idx %d data %v", vol.TimeIndex, sliceUint16(vol.Data))
	}
}

func TestLoadNiftiScalarVolumeAtGzipMatchesUncompressed(t *testing.T) {
	t.Setenv("ULTRA_CONTROL_NIFTI_DECOMPRESS_CACHE", "0") // force the streaming path
	dir := t.TempDir()
	plain := filepath.Join(dir, "series.nii")
	gz := filepath.Join(dir, "series.nii.gz")
	values := make([]uint16, 2*2*2*5) // 5 timepoints
	for i := range values {
		values[i] = uint16(i)
	}
	writeUint16Nifti(t, plain, 4, 2, 2, 2, 5, 1, values, false)
	writeUint16Nifti(t, gz, 4, 2, 2, 2, 5, 1, values, true)

	// Includes the last timepoint, which exercises the streaming discard-to-offset
	// path on the gzip reader.
	for _, ti := range []int{0, 1, 4} {
		plainVol, err := loadNiftiScalarVolumeAt(plain, ti, 0)
		if err != nil {
			t.Fatalf("plain t=%d: %v", ti, err)
		}
		gzVol, err := loadNiftiScalarVolumeAt(gz, ti, 0)
		if err != nil {
			t.Fatalf("gzip t=%d: %v", ti, err)
		}
		if !bytes.Equal(plainVol.Data, gzVol.Data) {
			t.Fatalf("t=%d gzip slab != uncompressed slab", ti)
		}
		wantFirst := uint16(ti * 8) // 8 voxels per timepoint
		if got := binary.LittleEndian.Uint16(gzVol.Data[0:2]); got != wantFirst {
			t.Fatalf("t=%d first voxel = %d, want %d", ti, got, wantFirst)
		}
	}
}

func TestLoadNiftiScalarVolumeAtChannelAddressing(t *testing.T) {
	t.Setenv("ULTRA_CONTROL_NIFTI_DECOMPRESS_CACHE", "0")
	dir := t.TempDir()
	path := filepath.Join(dir, "multichannel.nii")
	// 5D: 2x1x2 spatial, T=1, C=2. Channel slabs are consecutive.
	values := []uint16{
		1, 2, 3, 4, // channel 0
		101, 102, 103, 104, // channel 1
	}
	writeUint16Nifti(t, path, 5, 2, 1, 2, 1, 2, values, false)

	for ci, want := range [][]uint16{{1, 2, 3, 4}, {101, 102, 103, 104}} {
		vol, err := loadNiftiScalarVolumeAt(path, 0, ci)
		if err != nil {
			t.Fatalf("c=%d: %v", ci, err)
		}
		if vol.ChannelCount != 2 || vol.ChannelIndex != ci || vol.TimeCount != 1 {
			t.Fatalf("c=%d geometry = ch:%d idx:%d t:%d", ci, vol.ChannelCount, vol.ChannelIndex, vol.TimeCount)
		}
		if got := sliceUint16(vol.Data); !equalUint16(got, want) {
			t.Fatalf("c=%d data = %v, want %v", ci, got, want)
		}
	}
}

func TestLoadNiftiScalarVolumeAtRejectsOversizeVolume(t *testing.T) {
	t.Setenv("ULTRA_CONTROL_NIFTI_DECOMPRESS_CACHE", "0")
	dir := t.TempDir()
	path := filepath.Join(dir, "huge.nii")
	// Declare a single-volume size far over the 2 GiB cap with a tiny file, so a
	// malformed/hostile header is rejected instead of attempting a giant alloc.
	const voxOffset = 352
	raw := make([]byte, voxOffset+8)
	binary.LittleEndian.PutUint32(raw[0:4], 348)
	binary.LittleEndian.PutUint16(raw[40:42], 3)
	binary.LittleEndian.PutUint16(raw[42:44], 2000)
	binary.LittleEndian.PutUint16(raw[44:46], 2000)
	binary.LittleEndian.PutUint16(raw[46:48], 2000) // 2000^3 * 2 bytes ~= 16 GB
	binary.LittleEndian.PutUint16(raw[70:72], 512)
	binary.LittleEndian.PutUint16(raw[72:74], 16)
	binary.LittleEndian.PutUint32(raw[108:112], math.Float32bits(float32(voxOffset)))
	copy(raw[344:348], []byte{'n', '+', '1', 0})
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, err := loadNiftiScalarVolumeAt(path, 0, 0); err == nil {
		t.Fatalf("expected oversize volume to be rejected")
	}
}

func TestNiftiDecompressSidecarServesRandomAccess(t *testing.T) {
	dir := t.TempDir()
	gz := filepath.Join(dir, "series.nii.gz")
	values := make([]uint16, 2*2*2*4) // 4 timepoints
	for i := range values {
		values[i] = uint16(i + 1)
	}
	writeUint16Nifti(t, gz, 4, 2, 2, 2, 4, 1, values, true)

	// No sidecar yet: streaming path still serves correctly.
	if readyDecompressedNiftiSidecar(gz) != "" {
		t.Fatalf("unexpected pre-existing sidecar")
	}

	dst := niftiDecompressedSidecarPath(gz)
	if err := buildDecompressedNiftiSidecar(context.Background(), gz, dst); err != nil {
		t.Fatalf("build sidecar: %v", err)
	}
	ready := readyDecompressedNiftiSidecar(gz)
	if ready != dst {
		t.Fatalf("sidecar not ready: %q", ready)
	}
	// The sidecar is the uncompressed volume (random-access friendly): its size
	// matches the raw NIfTI, larger than the gzipped source.
	rawInfo, _ := os.Stat(dst)
	gzInfo, _ := os.Stat(gz)
	if rawInfo.Size() <= gzInfo.Size() {
		t.Fatalf("sidecar size %d should exceed gzip size %d", rawInfo.Size(), gzInfo.Size())
	}

	// With the sidecar present, every timepoint serves via ReadAt and matches the
	// source data exactly.
	for ti := 0; ti < 4; ti++ {
		vol, err := loadNiftiScalarVolumeAt(gz, ti, 0)
		if err != nil {
			t.Fatalf("t=%d: %v", ti, err)
		}
		want := values[ti*8 : ti*8+8]
		if got := sliceUint16(vol.Data); !equalUint16(got, want) {
			t.Fatalf("t=%d via sidecar = %v, want %v", ti, got, want)
		}
	}
}

// niftiTestHeader builds a configurable NIfTI-1 header (uint16 data, 1 voxel)
// for affine/rescale/orientation tests.
type niftiTestHeader struct {
	w, h, d             int
	sformCode           int16
	srow                [12]float32 // row-major 3x4
	qformCode           int16
	quatB, quatC, quatD float32
	qoffX, qoffY, qoffZ float32
	qfac                float32
	sclSlope, sclInter  float32
	xyztUnits           byte
}

func buildNiftiTestHeader(cfg niftiTestHeader) []byte {
	const voxOffset = 352
	w, h, d := cfg.w, cfg.h, cfg.d
	if w == 0 {
		w, h, d = 1, 1, 1
	}
	n := w * h * d
	raw := make([]byte, voxOffset+n*2)
	put16 := func(off int, v uint16) { binary.LittleEndian.PutUint16(raw[off:off+2], v) }
	putf := func(off int, v float32) { binary.LittleEndian.PutUint32(raw[off:off+4], math.Float32bits(v)) }
	binary.LittleEndian.PutUint32(raw[0:4], 348)
	put16(40, 3)
	put16(42, uint16(w))
	put16(44, uint16(h))
	put16(46, uint16(d))
	put16(70, 512) // DT_UINT16
	put16(72, 16)
	putf(76, cfg.qfac) // pixdim[0] = qfac
	for axis := 1; axis <= 3; axis++ {
		putf(76+axis*4, 1) // pixdim spacing = 1mm
	}
	putf(108, voxOffset)
	putf(112, cfg.sclSlope)
	putf(116, cfg.sclInter)
	raw[123] = cfg.xyztUnits
	binary.LittleEndian.PutUint16(raw[252:254], uint16(cfg.qformCode))
	binary.LittleEndian.PutUint16(raw[254:256], uint16(cfg.sformCode))
	putf(256, cfg.quatB)
	putf(260, cfg.quatC)
	putf(264, cfg.quatD)
	putf(268, cfg.qoffX)
	putf(272, cfg.qoffY)
	putf(276, cfg.qoffZ)
	for i := 0; i < 4; i++ {
		putf(280+i*4, cfg.srow[0*4+i])
		putf(296+i*4, cfg.srow[1*4+i])
		putf(312+i*4, cfg.srow[2*4+i])
	}
	copy(raw[344:348], []byte{'n', '+', '1', 0})
	return raw
}

func parseTestHeader(t *testing.T, cfg niftiTestHeader) niftiGeometry {
	t.Helper()
	geom, err := parseNiftiGeometry(buildNiftiTestHeader(cfg))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	return geom
}

func TestNiftiSformOrientationRAS(t *testing.T) {
	geom := parseTestHeader(t, niftiTestHeader{
		w: 1, h: 1, d: 1, sformCode: 1,
		srow: [12]float32{2, 0, 0, -90, 0, 2, 0, -126, 0, 0, 2, -72},
	})
	if geom.affineCode != 3 {
		t.Fatalf("affineCode = %d, want 3 (sform)", geom.affineCode)
	}
	code, ends, planeAxis := niftiOrientation(geom.affine)
	if code != "RAS" {
		t.Fatalf("orientation = %q, want RAS", code)
	}
	if planeAxis["axial"] != 2 || planeAxis["coronal"] != 1 || planeAxis["sagittal"] != 0 {
		t.Fatalf("planeAxis = %v, want axial=2 coronal=1 sagittal=0", planeAxis)
	}
	if ends[0][1] != "R" || ends[0][0] != "L" {
		t.Fatalf("x-axis ends = %v, want [L R]", ends[0])
	}
}

func TestNiftiSformLeftHandedFlip(t *testing.T) {
	// Negative srow_x[0] flips the first voxel axis to point Left -> LAS.
	geom := parseTestHeader(t, niftiTestHeader{
		w: 1, h: 1, d: 1, sformCode: 1,
		srow: [12]float32{-2, 0, 0, 90, 0, 2, 0, -126, 0, 0, 2, -72},
	})
	code, ends, _ := niftiOrientation(geom.affine)
	if code != "LAS" {
		t.Fatalf("orientation = %q, want LAS", code)
	}
	if ends[0][1] != "L" {
		t.Fatalf("x positive end = %q, want L", ends[0][1])
	}
}

func TestNiftiQformQfacFlipsThirdAxis(t *testing.T) {
	// Identity rotation (a=1) with qfac=-1 flips the slice axis -> RAI.
	geom := parseTestHeader(t, niftiTestHeader{
		w: 1, h: 1, d: 1, qformCode: 1, qfac: -1,
	})
	if geom.affineCode != 2 {
		t.Fatalf("affineCode = %d, want 2 (qform)", geom.affineCode)
	}
	code, _, planeAxis := niftiOrientation(geom.affine)
	if code != "RAI" {
		t.Fatalf("orientation = %q, want RAI", code)
	}
	if planeAxis["axial"] != 2 {
		t.Fatalf("axial axis = %d, want 2", planeAxis["axial"])
	}
}

func TestNiftiRescaleHounsfieldDetectionAndWindow(t *testing.T) {
	// CT stored as unsigned codes with scl_inter=-1024: HU = code - 1024.
	geom := parseTestHeader(t, niftiTestHeader{
		w: 1, h: 1, d: 1, sclSlope: 1, sclInter: -1024, xyztUnits: 2,
	})
	if geom.sclInter != -1024 || geom.sclSlope != 1 {
		t.Fatalf("rescale = slope %v inter %v, want 1/-1024", geom.sclSlope, geom.sclInter)
	}
	if geom.spaceUnit != "mm" {
		t.Fatalf("space unit = %q, want mm", geom.spaceUnit)
	}
	vol := niftiScalarVolume{SclSlope: geom.sclSlope, SclInter: geom.sclInter, RawMin: 0, RawMax: 4000}
	if vol.physical(0) != -1024 || vol.physical(4000) != 2976 {
		t.Fatalf("physical range = [%v,%v], want [-1024,2976]", vol.physical(0), vol.physical(4000))
	}
	if !niftiScalarRangeLooksCTLike(vol) {
		t.Fatalf("unsigned CT with inter=-1024 should be detected as CT-like")
	}
	// A brain window WC=40 WW=80 (HU [0,80]) must map to codes [1024,1104].
	transform := uploadPreviewTransform{WindowActive: true, WindowIsPhysical: true, WindowMin: 0, WindowMax: 80}
	lo, hi := scalarPreviewWindow(vol, transform)
	if lo != 1024 || hi != 1104 {
		t.Fatalf("HU window mapped to codes [%v,%v], want [1024,1104]", lo, hi)
	}
}

func TestNiftiRescaleSlopeScalesPhysicalRange(t *testing.T) {
	// PET-style slope != 1.
	vol := niftiScalarVolume{SclSlope: 2.5, SclInter: 0, RawMin: 0, RawMax: 100}
	if vol.physical(100) != 250 {
		t.Fatalf("physical(100) = %v, want 250", vol.physical(100))
	}
	if vol.codeFromPhysical(250) != 100 {
		t.Fatalf("codeFromPhysical(250) = %v, want 100", vol.codeFromPhysical(250))
	}
}

func TestNiftiViewerManifestEmitsOrientationAndRescale(t *testing.T) {
	uploadRoot := t.TempDir()
	router := NewRouter(ServerDeps{Version: "test", UploadRoot: uploadRoot})
	header := buildNiftiTestHeader(niftiTestHeader{
		w: 2, h: 2, d: 2, sformCode: 1, sclSlope: 1, sclInter: -1024, xyztUnits: 2,
		srow: [12]float32{2, 0, 0, -90, 0, 2, 0, -126, 0, 0, 2, -72},
	})
	fileID := writeTestUploadFile(t, uploadRoot, "ct.nii", header)
	req := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/viewer", nil)
	req.Header.Set("X-Ultra-User-Id", "test-user")
	req.Header.Set("X-Ultra-Org-Id", "test-org")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	var resp struct {
		Metadata struct {
			OrientationCode     string      `json:"orientation_code"`
			RescaleIntercept    float64     `json:"rescale_intercept"`
			RescaleSlope        float64     `json:"rescale_slope"`
			PhysicalSpacingUnit string      `json:"physical_spacing_unit"`
			Affine              [][]float64 `json:"affine"`
			IntensityStatsPhys  struct {
				Min float64 `json:"min"`
			} `json:"intensity_stats_physical"`
		} `json:"metadata"`
		Viewer struct {
			Orientation struct {
				Frame      string `json:"frame"`
				Convention string `json:"convention"`
				Code       string `json:"code"`
				Known      bool   `json:"known"`
			} `json:"orientation"`
			DisplayCapabilities []string `json:"display_capabilities"`
		} `json:"viewer"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.Metadata.OrientationCode != "RAS" {
		t.Fatalf("orientation_code = %q, want RAS", resp.Metadata.OrientationCode)
	}
	if resp.Metadata.RescaleIntercept != -1024 || resp.Metadata.RescaleSlope != 1 {
		t.Fatalf("rescale = %v/%v, want 1/-1024", resp.Metadata.RescaleSlope, resp.Metadata.RescaleIntercept)
	}
	if resp.Metadata.PhysicalSpacingUnit != "mm" {
		t.Fatalf("unit = %q, want mm", resp.Metadata.PhysicalSpacingUnit)
	}
	if len(resp.Metadata.Affine) != 4 || resp.Metadata.Affine[0][0] != 2 || resp.Metadata.Affine[3][3] != 1 {
		t.Fatalf("affine = %v", resp.Metadata.Affine)
	}
	if resp.Viewer.Orientation.Frame != "patient" || resp.Viewer.Orientation.Convention != "neurological" || resp.Viewer.Orientation.Code != "RAS" || !resp.Viewer.Orientation.Known {
		t.Fatalf("orientation block = %+v", resp.Viewer.Orientation)
	}
	if !sliceContains(resp.Viewer.DisplayCapabilities, "orientation_markers") {
		t.Fatalf("missing orientation_markers capability: %v", resp.Viewer.DisplayCapabilities)
	}
}

// TestNiftiRealFileAffineMatchesNibabel cross-checks the Go affine/orientation
// parse against nibabel's reference output for the real HCP rfMRI:
// AXCODES=LAS, affine=[[-2,0,0,90],[0,2,0,-126],[0,0,2,-72]], scl=NaN (identity),
// units=mm. Opt in via ULTRA_NIFTI_REAL_FILE.
func TestNiftiRealFileAffineMatchesNibabel(t *testing.T) {
	path := os.Getenv("ULTRA_NIFTI_REAL_FILE")
	if path == "" {
		t.Skip("set ULTRA_NIFTI_REAL_FILE to the rfMRI .nii.gz to cross-check against nibabel")
	}
	t.Setenv("ULTRA_CONTROL_NIFTI_DECOMPRESS_CACHE", "0")
	vol, err := loadNiftiScalarVolume(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	code, _, planeAxis := niftiOrientation(vol.Affine)
	if code != "LAS" {
		t.Fatalf("orientation = %q, want LAS (nibabel ground truth)", code)
	}
	want := [12]float64{-2, 0, 0, 90, 0, 2, 0, -126, 0, 0, 2, -72}
	for i := range want {
		if math.Abs(vol.Affine[i]-want[i]) > 1e-3 {
			t.Fatalf("affine[%d] = %v, want %v", i, vol.Affine[i], want[i])
		}
	}
	// scl_slope/inter are NaN in this file -> must be treated as identity.
	if vol.SclSlope != 1 || vol.SclInter != 0 {
		t.Fatalf("rescale = %v/%v, want identity 1/0 (NaN guard)", vol.SclSlope, vol.SclInter)
	}
	if vol.SpaceUnit != "mm" {
		t.Fatalf("space unit = %q, want mm", vol.SpaceUnit)
	}
	// LAS: i->L (sagittal), j->A (coronal), k->S (axial).
	if planeAxis["sagittal"] != 0 || planeAxis["coronal"] != 1 || planeAxis["axial"] != 2 {
		t.Fatalf("planeAxis = %v, want sagittal=0 coronal=1 axial=2", planeAxis)
	}
	t.Logf("Go parser matches nibabel: orientation=%s affine ok rescale=identity unit=mm", code)
}

func TestScalarPreviewWindowFullRangeUsesDataRangeNotDatatypeSpan(t *testing.T) {
	// int16 MRI with values ~0..1000: "full range" must use the data range, not
	// the [-32768, 32767] datatype span (which rendered it near-black).
	vol := niftiScalarVolume{DType: "int16", RawMin: 0, RawMax: 1000}
	lo, hi := scalarPreviewWindow(vol, uploadPreviewTransform{FullRange: true})
	if lo != 0 || hi != 1000 {
		t.Fatalf("int16 full range = [%v,%v], want data range [0,1000]", lo, hi)
	}
	// uint8 display data still spans the full byte range.
	lo8, hi8 := scalarPreviewWindow(niftiScalarVolume{DType: "uint8", RawMin: 5, RawMax: 200}, uploadPreviewTransform{FullRange: true})
	if lo8 != 0 || hi8 != 255 {
		t.Fatalf("uint8 full range = [%v,%v], want [0,255]", lo8, hi8)
	}
}

func equalUint16(a, b []uint16) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestNiftiBoundedMemoryRealFile proves the bounded reader holds only one
// timepoint (~MB) in memory even for the real 4.33 GB 4D fMRI, including the
// worst-case last timepoint. Opt in by pointing ULTRA_NIFTI_REAL_FILE at the
// uploaded .nii.gz; skipped otherwise so CI stays hermetic.
func TestNiftiBoundedMemoryRealFile(t *testing.T) {
	path := os.Getenv("ULTRA_NIFTI_REAL_FILE")
	if path == "" {
		t.Skip("set ULTRA_NIFTI_REAL_FILE to the rfMRI .nii.gz to run the memory bound check")
	}
	t.Setenv("ULTRA_CONTROL_NIFTI_DECOMPRESS_CACHE", "0") // measure the streaming path

	const maxHeapDelta = int64(128) << 20 // 128 MiB ceiling for one ~3.6 MB slab
	for _, ti := range []int{0, 1199} {
		runtime.GC()
		var before runtime.MemStats
		runtime.ReadMemStats(&before)

		vol, err := loadNiftiScalarVolumeAt(path, ti, 0)
		if err != nil {
			t.Fatalf("t=%d: %v", ti, err)
		}

		var after runtime.MemStats
		runtime.ReadMemStats(&after)
		delta := int64(after.HeapAlloc) - int64(before.HeapAlloc)
		t.Logf("t=%d: timepoints=%d volume=%dx%dx%d slab=%d bytes heapDelta=%.1f MiB",
			ti, vol.TimeCount, vol.Width, vol.Height, vol.Depth, len(vol.Data), float64(delta)/(1<<20))
		if delta > maxHeapDelta {
			t.Fatalf("t=%d heap delta %.1f MiB exceeds %.0f MiB — not bounded to one timepoint",
				ti, float64(delta)/(1<<20), float64(maxHeapDelta)/(1<<20))
		}
		runtime.KeepAlive(vol)
	}
}
