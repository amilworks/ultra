package httpapi

import (
	"bytes"
	"compress/gzip"
	"encoding/base64"
	"encoding/binary"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// These tests validate the CIFTI reader against REAL nibabel-written fixtures
// (ground truth for the byte layout + XML). Skipped unless the env vars point at
// them; generate with scratchpad/gen_cifti_fixtures.py.
//   ULTRA_CIFTI_DTSERIES_FIXTURE=<dir>/hcp_like.dtseries.nii
//   ULTRA_CIFTI_PCONN_FIXTURE=<dir>/parcels.pconn.nii

type ciftiFixtureOptions struct {
	physicalRowRole        string
	physicalColRole        string
	dim7                   int
	unrelatedExtensionSize int
}

func writeCifti2Fixture(t *testing.T, name string, rows, cols int, value func(row, col int) float32) (string, []byte) {
	t.Helper()
	return writeCifti2FixtureWithOptions(t, name, rows, cols, ciftiFixtureOptions{
		physicalRowRole: "brain_models",
		physicalColRole: "series",
		dim7:            1,
	}, value)
}

func writeCifti2FixtureWithOptions(t *testing.T, name string, rows, cols int, options ciftiFixtureOptions, value func(row, col int) float32) (string, []byte) {
	t.Helper()
	if options.dim7 <= 0 {
		options.dim7 = 1
	}
	axisXML := func(dimension int, role string, size int) string {
		prefix := `<MatrixIndicesMap AppliesToMatrixDimension="` + strconv.Itoa(dimension) + `" IndicesMapToDataType="CIFTI_INDEX_TYPE_` + strings.ToUpper(role) + `"`
		switch role {
		case "series":
			return prefix + ` NumberOfSeriesPoints="` + strconv.Itoa(size) + `" SeriesStep="0.72" SeriesUnit="SECOND"/>`
		case "brain_models":
			return prefix + `><BrainModel IndexOffset="0" IndexCount="` + strconv.Itoa(size) + `" BrainStructure="CIFTI_STRUCTURE_CORTEX_LEFT"/></MatrixIndicesMap>`
		case "parcels":
			return prefix + `><Parcel Name="parcel"/></MatrixIndicesMap>`
		default:
			return prefix + `/>`
		}
	}
	xml := []byte(`<CIFTI Version="2.0"><Matrix>` +
		axisXML(0, options.physicalColRole, cols) +
		axisXML(1, options.physicalRowRole, rows) +
		`</Matrix></CIFTI>`)
	esize := (8 + len(xml) + 1 + 15) / 16 * 16
	unrelatedSize := options.unrelatedExtensionSize
	if unrelatedSize > 0 {
		unrelatedSize = max(16, (unrelatedSize+15)/16*16)
	}
	voxOffset := nifti2HeaderReadSize + unrelatedSize + esize
	payload := make([]byte, voxOffset+rows*cols*options.dim7*4)
	binary.LittleEndian.PutUint32(payload[0:4], nifti2HeaderSize)
	copy(payload[4:8], []byte("n+2\x00"))
	binary.LittleEndian.PutUint16(payload[12:14], 16)
	binary.LittleEndian.PutUint16(payload[14:16], 32)
	dimensionCount := int64(6)
	if options.dim7 > 1 {
		dimensionCount = 7
	}
	for index, dimension := range []int64{dimensionCount, 1, 1, 1, 1, int64(cols), int64(rows), int64(options.dim7)} {
		binary.LittleEndian.PutUint64(payload[16+index*8:24+index*8], uint64(dimension))
	}
	binary.LittleEndian.PutUint64(payload[168:176], uint64(voxOffset))
	binary.LittleEndian.PutUint64(payload[176:184], math.Float64bits(1))
	binary.LittleEndian.PutUint32(payload[504:508], 3002)
	payload[nifti2HeaderSize] = 1
	extensionOffset := nifti2HeaderReadSize
	if unrelatedSize > 0 {
		binary.LittleEndian.PutUint32(payload[extensionOffset:extensionOffset+4], uint32(unrelatedSize))
		binary.LittleEndian.PutUint32(payload[extensionOffset+4:extensionOffset+8], 6)
		extensionOffset += unrelatedSize
	}
	binary.LittleEndian.PutUint32(payload[extensionOffset:extensionOffset+4], uint32(esize))
	binary.LittleEndian.PutUint32(payload[extensionOffset+4:extensionOffset+8], 32)
	copy(payload[extensionOffset+8:voxOffset], xml)
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			offset := voxOffset + (row*cols+col)*4
			binary.LittleEndian.PutUint32(payload[offset:offset+4], math.Float32bits(value(row, col)))
		}
	}
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, payload, 0o600); err != nil {
		t.Fatalf("write CIFTI fixture: %v", err)
	}
	return path, payload
}

func TestRenderCiftiThumbnailPNGDeterministic(t *testing.T) {
	t.Parallel()
	path, _ := writeCifti2Fixture(t, "rest.dtseries.nii", 240, 300, func(row, col int) float32 {
		return float32(math.Sin(float64(col)/13) + math.Cos(float64(row)/17))
	})
	first, err := renderCiftiThumbnailPNG(path)
	if err != nil {
		t.Fatalf("render CIFTI thumbnail: %v", err)
	}
	second, err := renderCiftiThumbnailPNG(path)
	if err != nil {
		t.Fatalf("render repeated CIFTI thumbnail: %v", err)
	}
	if !bytes.Equal(first, second) {
		t.Fatal("repeated CIFTI thumbnails are not byte-identical")
	}
	config, err := png.DecodeConfig(bytes.NewReader(first))
	if err != nil {
		t.Fatalf("decode thumbnail config: %v", err)
	}
	if config.Width != ciftiThumbnailWidth || config.Height != ciftiThumbnailHeight {
		t.Fatalf("thumbnail dimensions = %dx%d, want %dx%d", config.Width, config.Height, ciftiThumbnailWidth, ciftiThumbnailHeight)
	}
	decoded, err := png.Decode(bytes.NewReader(first))
	if err != nil {
		t.Fatalf("decode thumbnail: %v", err)
	}
	firstPixel := color.RGBAModel.Convert(decoded.At(0, 0)).(color.RGBA)
	nonuniform := false
	for y := 0; y < decoded.Bounds().Dy() && !nonuniform; y++ {
		for x := 0; x < decoded.Bounds().Dx(); x++ {
			if color.RGBAModel.Convert(decoded.At(x, y)).(color.RGBA) != firstPixel {
				nonuniform = true
				break
			}
		}
	}
	if !nonuniform {
		t.Fatal("CIFTI thumbnail pixels are uniform")
	}
}

func TestRenderCiftiThumbnailNeutralForConstantAndNonfiniteRows(t *testing.T) {
	t.Parallel()
	path, _ := writeCifti2Fixture(t, "neutral.dtseries.nii", 3, 300, func(row, col int) float32 {
		switch row {
		case 0:
			return 7
		case 1:
			return float32(math.NaN())
		default:
			if col%2 == 0 {
				return float32(math.Inf(1))
			}
			return float32(math.Inf(-1))
		}
	})
	body, err := renderCiftiThumbnailPNG(path)
	if err != nil {
		t.Fatalf("render neutral CIFTI thumbnail: %v", err)
	}
	decoded, err := png.Decode(bytes.NewReader(body))
	if err != nil {
		t.Fatalf("decode neutral thumbnail: %v", err)
	}
	want := color.RGBA{R: 0xf7, G: 0xf7, B: 0xf7, A: 0xff}
	for y := 0; y < decoded.Bounds().Dy(); y++ {
		for x := 0; x < decoded.Bounds().Dx(); x++ {
			if got := color.RGBAModel.Convert(decoded.At(x, y)).(color.RGBA); got != want {
				t.Fatalf("neutral thumbnail pixel (%d,%d) = %#v, want %#v", x, y, got, want)
			}
		}
	}
}

func TestRenderCiftiThumbnailRejectsMalformedTruncatedAndGzip(t *testing.T) {
	t.Parallel()
	path, payload := writeCifti2Fixture(t, "matrix.dtseries.nii", 4, 300, func(row, col int) float32 {
		return float32(row + col)
	})
	truncated := filepath.Join(t.TempDir(), "truncated.dtseries.nii")
	if err := os.WriteFile(truncated, payload[:len(payload)-4], 0o600); err != nil {
		t.Fatalf("write truncated fixture: %v", err)
	}
	malformedBytes := append([]byte(nil), payload...)
	binary.LittleEndian.PutUint32(malformedBytes[nifti2HeaderReadSize:nifti2HeaderReadSize+4], 15)
	malformed := filepath.Join(t.TempDir(), "malformed.dtseries.nii")
	if err := os.WriteFile(malformed, malformedBytes, 0o600); err != nil {
		t.Fatalf("write malformed fixture: %v", err)
	}
	gzipPath := filepath.Join(t.TempDir(), "matrix.dtseries.nii.gz")
	gzipFile, err := os.Create(gzipPath)
	if err != nil {
		t.Fatalf("create gzip fixture: %v", err)
	}
	gzipWriter := gzip.NewWriter(gzipFile)
	if _, err := gzipWriter.Write(payload); err != nil {
		t.Fatalf("write gzip fixture: %v", err)
	}
	if err := gzipWriter.Close(); err != nil {
		t.Fatalf("close gzip writer: %v", err)
	}
	if err := gzipFile.Close(); err != nil {
		t.Fatalf("close gzip fixture: %v", err)
	}
	for _, candidate := range []string{truncated, malformed, gzipPath} {
		if _, err := renderCiftiThumbnailPNG(candidate); err == nil {
			t.Fatalf("renderCiftiThumbnailPNG(%q) succeeded, want clean failure", candidate)
		}
	}
	if _, err := renderCiftiThumbnailPNG(path); err != nil {
		t.Fatalf("valid control fixture failed: %v", err)
	}
}

func TestRenderCiftiThumbnailRejectsNonTimeseriesAndExtraPayloadAxis(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name    string
		rowRole string
		colRole string
		dim7    int
	}{
		{name: "dscalar", rowRole: "brain_models", colRole: "scalars", dim7: 1},
		{name: "dlabel", rowRole: "brain_models", colRole: "labels", dim7: 1},
		{name: "connectivity", rowRole: "brain_models", colRole: "brain_models", dim7: 1},
		{name: "transposed-timeseries", rowRole: "series", colRole: "brain_models", dim7: 1},
		{name: "extra-dim", rowRole: "brain_models", colRole: "series", dim7: 2},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			path, _ := writeCifti2FixtureWithOptions(t, test.name+".nii", 4, 8, ciftiFixtureOptions{
				physicalRowRole: test.rowRole,
				physicalColRole: test.colRole,
				dim7:            test.dim7,
			}, func(row, col int) float32 {
				return float32(row + col)
			})
			if _, err := renderCiftiThumbnailPNG(path); err == nil {
				t.Fatal("renderCiftiThumbnailPNG succeeded for unsupported matrix semantics")
			}
		})
	}
}

func TestRenderCiftiThumbnailRejectsIncompatibleNiftiDimensions(t *testing.T) {
	t.Parallel()
	_, payload := writeCifti2Fixture(t, "dimensions.dtseries.nii", 4, 8, func(row, col int) float32 {
		return float32(row + col)
	})
	tests := []struct {
		name  string
		index int
		value uint64
	}{
		{name: "dimension-count", index: 0, value: 5},
		{name: "non-singleton-prefix", index: 3, value: 2},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			malformed := append([]byte(nil), payload...)
			offset := 16 + test.index*8
			binary.LittleEndian.PutUint64(malformed[offset:offset+8], test.value)
			path := filepath.Join(t.TempDir(), test.name+".dtseries.nii")
			if err := os.WriteFile(path, malformed, 0o600); err != nil {
				t.Fatalf("write incompatible CIFTI fixture: %v", err)
			}
			if _, err := renderCiftiThumbnailPNG(path); err == nil {
				t.Fatal("renderCiftiThumbnailPNG accepted incompatible NIfTI dimensions")
			}
		})
	}
}

func TestRenderCiftiThumbnailRejectsXMLDimensionCountMismatch(t *testing.T) {
	t.Parallel()
	_, payload := writeCifti2Fixture(t, "counts.dtseries.nii", 4, 300, func(row, col int) float32 {
		return float32(row + col)
	})
	tests := []struct {
		name string
		old  string
		new  string
	}{
		{name: "series-points", old: `NumberOfSeriesPoints="300"`, new: `NumberOfSeriesPoints="299"`},
		{name: "brain-model-count", old: `IndexCount="4"`, new: `IndexCount="3"`},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			malformed := bytes.Replace(payload, []byte(test.old), []byte(test.new), 1)
			if bytes.Equal(malformed, payload) {
				t.Fatalf("fixture marker %q was not found", test.old)
			}
			path := filepath.Join(t.TempDir(), test.name+".dtseries.nii")
			if err := os.WriteFile(path, malformed, 0o600); err != nil {
				t.Fatalf("write mismatched CIFTI fixture: %v", err)
			}
			if _, err := renderCiftiThumbnailPNG(path); err == nil {
				t.Fatal("renderCiftiThumbnailPNG accepted mismatched XML dimensions")
			}
		})
	}
}

type countingReadSeeker struct {
	*bytes.Reader
	readBytes int64
}

func (reader *countingReadSeeker) Read(body []byte) (int, error) {
	read, err := reader.Reader.Read(body)
	reader.readBytes += int64(read)
	return read, err
}

func TestNiftiCiftiExtensionSeeksPastLargeUnrelatedExtension(t *testing.T) {
	t.Parallel()
	_, payload := writeCifti2FixtureWithOptions(t, "large-extension.dtseries.nii", 2, 3, ciftiFixtureOptions{
		physicalRowRole:        "brain_models",
		physicalColRole:        "series",
		dim7:                   1,
		unrelatedExtensionSize: 20 << 20,
	}, func(row, col int) float32 {
		return float32(row + col)
	})
	reader := &countingReadSeeker{Reader: bytes.NewReader(payload)}
	voxOffset := int64(binary.LittleEndian.Uint64(payload[168:176]))
	xml, err := niftiCiftiExtension(reader, binary.LittleEndian, voxOffset)
	if err != nil {
		t.Fatalf("niftiCiftiExtension: %v", err)
	}
	if !strings.Contains(xml, "CIFTI_INDEX_TYPE_SERIES") {
		t.Fatalf("unexpected CIFTI XML: %q", xml)
	}
	if reader.readBytes >= int64(1<<20) {
		t.Fatalf("extension scanner read %d bytes; large unrelated body was not seek-skipped", reader.readBytes)
	}
}

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
