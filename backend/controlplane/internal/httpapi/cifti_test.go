package httpapi

import (
	"encoding/base64"
	"encoding/binary"
	"math"
	"os"
	"testing"
)

// These tests validate the CIFTI reader against REAL nibabel-written fixtures
// (ground truth for the byte layout + XML). Skipped unless the env vars point at
// them; generate with scratchpad/gen_cifti_fixtures.py.
//   ULTRA_CIFTI_DTSERIES_FIXTURE=<dir>/hcp_like.dtseries.nii
//   ULTRA_CIFTI_PCONN_FIXTURE=<dir>/parcels.pconn.nii

func TestParseCiftiDtseriesMeta(t *testing.T) {
	path := os.Getenv("ULTRA_CIFTI_DTSERIES_FIXTURE")
	if path == "" {
		t.Skip("set ULTRA_CIFTI_DTSERIES_FIXTURE")
	}
	meta, err := parseCiftiMeta(path)
	if err != nil {
		t.Fatalf("parseCiftiMeta: %v", err)
	}
	if meta.kind != "timeseries" {
		t.Fatalf("kind = %q, want timeseries", meta.kind)
	}
	if meta.rows.role != "brain_models" || meta.cols.role != "series" {
		t.Fatalf("axes = rows:%q cols:%q, want brain_models/series", meta.rows.role, meta.cols.role)
	}
	if meta.rows.size != 2240 || meta.cols.size != 300 {
		t.Fatalf("dims = %d rows × %d cols, want 2240 × 300", meta.rows.size, meta.cols.size)
	}
	if meta.cols.step != 0.72 {
		t.Fatalf("TR = %v, want 0.72", meta.cols.step)
	}
	got := 0
	for _, s := range meta.rows.structures {
		got += s.Count
	}
	if len(meta.rows.structures) != 4 || got != 2240 {
		t.Fatalf("structures = %+v (sum %d), want 4 summing to 2240", meta.rows.structures, got)
	}
	if meta.rows.structures[0].Name != "Cortex Left" {
		t.Fatalf("first structure = %q, want Cortex Left", meta.rows.structures[0].Name)
	}
}

func TestCiftiCarpetBuild(t *testing.T) {
	path := os.Getenv("ULTRA_CIFTI_DTSERIES_FIXTURE")
	if path == "" {
		t.Skip("set ULTRA_CIFTI_DTSERIES_FIXTURE")
	}
	meta, err := parseCiftiMeta(path)
	if err != nil {
		t.Fatalf("parseCiftiMeta: %v", err)
	}
	mat, err := openCiftiMatrix(path, meta)
	if err != nil {
		t.Fatalf("openCiftiMatrix: %v", err)
	}
	defer mat.close()
	payload, err := buildCarpet(mat, meta, 800, 300)
	if err != nil {
		t.Fatalf("buildCarpet: %v", err)
	}
	rows := payload["rows"].(int)
	cols := payload["cols"].(int)
	if rows != 800 || cols != 300 {
		t.Fatalf("carpet %d×%d, want 800×300", rows, cols)
	}
	raw, err := base64.StdEncoding.DecodeString(payload["data"].(string))
	if err != nil || len(raw) != rows*cols {
		t.Fatalf("carpet data length %d, want %d", len(raw), rows*cols)
	}
	bands := payload["structures"].([]ciftiStructureBand)
	if len(bands) != 4 || bands[0].Start != 0 || bands[3].End != rows {
		t.Fatalf("structure bands wrong: %+v", bands)
	}
}

func TestCiftiConnectivityComputedFromTimeseries(t *testing.T) {
	path := os.Getenv("ULTRA_CIFTI_DTSERIES_FIXTURE")
	if path == "" {
		t.Skip("set ULTRA_CIFTI_DTSERIES_FIXTURE")
	}
	meta, _ := parseCiftiMeta(path)
	mat, err := openCiftiMatrix(path, meta)
	if err != nil {
		t.Fatalf("openCiftiMatrix: %v", err)
	}
	defer mat.close()
	payload, err := buildConnectivity(mat, meta, 120)
	if err != nil {
		t.Fatalf("buildConnectivity: %v", err)
	}
	n := payload["n"].(int)
	if n != 120 || payload["computed"] != true {
		t.Fatalf("connectivity n=%d computed=%v, want 120/true", n, payload["computed"])
	}
	vals := decodeFloat32(t, payload["data"].(string), n*n)
	// Diagonal is exactly 1; matrix symmetric; values in [-1, 1].
	for i := 0; i < n; i++ {
		if math.Abs(float64(vals[i*n+i])-1) > 1e-5 {
			t.Fatalf("diagonal[%d] = %v, want 1", i, vals[i*n+i])
		}
		for j := 0; j < n; j++ {
			if vals[i*n+j] < -1.001 || vals[i*n+j] > 1.001 {
				t.Fatalf("corr out of range at %d,%d: %v", i, j, vals[i*n+j])
			}
			if math.Abs(float64(vals[i*n+j]-vals[j*n+i])) > 1e-5 {
				t.Fatalf("not symmetric at %d,%d", i, j)
			}
		}
	}
}

func TestParseCiftiPconnMeta(t *testing.T) {
	path := os.Getenv("ULTRA_CIFTI_PCONN_FIXTURE")
	if path == "" {
		t.Skip("set ULTRA_CIFTI_PCONN_FIXTURE")
	}
	meta, err := parseCiftiMeta(path)
	if err != nil {
		t.Fatalf("parseCiftiMeta: %v", err)
	}
	if meta.kind != "connectivity" || meta.label != "parcellated connectivity" {
		t.Fatalf("kind/label = %q/%q, want connectivity/parcellated connectivity", meta.kind, meta.label)
	}
	if meta.rows.size != 100 || meta.cols.size != 100 {
		t.Fatalf("dims = %d × %d, want 100 × 100", meta.rows.size, meta.cols.size)
	}
	mat, _ := openCiftiMatrix(path, meta)
	defer mat.close()
	payload, err := buildConnectivity(mat, meta, 100)
	if err != nil {
		t.Fatalf("buildConnectivity: %v", err)
	}
	if payload["computed"] != false {
		t.Fatalf("pconn should serve the stored matrix directly (computed=false)")
	}
	vals := decodeFloat32(t, payload["data"].(string), 100*100)
	// It's a correlation matrix → unit diagonal.
	if math.Abs(float64(vals[0])-1) > 1e-4 {
		t.Fatalf("pconn[0,0] = %v, want ~1", vals[0])
	}
}

func decodeFloat32(t *testing.T, b64 string, n int) []float32 {
	t.Helper()
	raw, err := base64.StdEncoding.DecodeString(b64)
	if err != nil || len(raw) != n*4 {
		t.Fatalf("bad float payload: len %d want %d (err %v)", len(raw), n*4, err)
	}
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}
