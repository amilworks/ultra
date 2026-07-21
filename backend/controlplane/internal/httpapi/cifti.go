package httpapi

import (
	"compress/gzip"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strconv"
	"strings"
)

// CIFTI grayordinate viewer: carpet plot + connectivity matrix.
//
// A CIFTI file (.dtseries.nii etc.) is a NIfTI-2 container whose payload is a 2-D
// MATRIX over brain locations (grayordinates or parcels) and a second axis
// (time, scalars, or a second set of locations) — not a spatial image. It cannot
// be shown as slices without surface meshes, but the matrix itself is exactly
// what Connectome Workbench shows on the right of its window, and it needs no
// surfaces. This serves two standard views of it:
//
//   - CARPET (grayplot): rows = brain locations grouped by structure, columns =
//     time/scalar index, colour = signal z-scored per row.
//   - CONNECTIVITY matrix: for a *conn file the stored matrix directly; for a
//     timeseries, the correlation of a bounded set of downsampled node signals.
//
// Reads are BOUNDED: the matrix is stored grayordinate-contiguous
// (data[row*cols + col]) at vox_offset, so a strided subset of rows is a handful
// of small ReadAts regardless of the file's size (a real dtseries is ~400 MB).

const (
	ciftiCarpetMaxRows  = 1200 // rows sampled for the carpet (viewport-bounded)
	ciftiCarpetMaxCols  = 1200 // columns kept for the carpet
	ciftiConnMaxNodes   = 240  // N for the N×N connectivity matrix
	ciftiGzipDecompCap  = int64(3) << 30
	ciftiCarpetClipZ    = 3.0
	ciftiMaxParcelNames = 512
)

var (
	reMatrixMap    = regexp.MustCompile(`(?s)<MatrixIndicesMap\b[^>]*>`)
	reAppliesToDim = regexp.MustCompile(`AppliesToMatrixDimension="([^"]+)"`)
	reMapDataType  = regexp.MustCompile(`IndicesMapToDataType="([^"]+)"`)
	reSeriesPoints = regexp.MustCompile(`NumberOfSeriesPoints="([^"]+)"`)
	reSeriesStep   = regexp.MustCompile(`SeriesStep="([^"]+)"`)
	reSeriesUnit   = regexp.MustCompile(`SeriesUnit="([^"]+)"`)
	reBrainModel   = regexp.MustCompile(`(?s)<BrainModel\b[^>]*>`)
	reBrainStruct  = regexp.MustCompile(`BrainStructure="([^"]+)"`)
	reIndexCount   = regexp.MustCompile(`IndexCount="([^"]+)"`)
	reParcelName   = regexp.MustCompile(`<Parcel\b[^>]*Name="([^"]+)"`)
)

type ciftiBrainModel struct {
	Name  string
	Count int
}

type ciftiAxis struct {
	role       string // "series" | "brain_models" | "parcels" | "scalars" | "labels" | "unknown"
	size       int
	step       float64 // SERIES step (seconds); 0 otherwise
	structures []ciftiBrainModel
	names      []string
}

type ciftiMeta struct {
	kind      string // "timeseries" | "connectivity" | "scalar" | "matrix"
	label     string // human, e.g. "dense timeseries"
	rows      ciftiAxis
	cols      ciftiAxis
	voxOffset int64
	order     binary.ByteOrder
	dtype     string
	bytesPer  int
	slope     float64
	inter     float64
}

// niftiCiftiExtension pulls the CIFTI XML out of the NIfTI-2 extension list.
func niftiCiftiExtension(f io.ReadSeeker) (string, error) {
	if _, err := f.Seek(nifti2HeaderSize, io.SeekStart); err != nil {
		return "", err
	}
	flag := make([]byte, 4)
	if _, err := io.ReadFull(f, flag); err != nil {
		return "", err
	}
	if flag[0] == 0 {
		return "", errors.New("no NIfTI extension present")
	}
	// Walk extensions: [int32 esize][int32 ecode][esize-8 bytes]; CIFTI ecode = 32.
	for i := 0; i < 32; i++ {
		hdr := make([]byte, 8)
		if _, err := io.ReadFull(f, hdr); err != nil {
			return "", err
		}
		esize := int64(int32(binary.LittleEndian.Uint32(hdr[0:4])))
		ecode := int32(binary.LittleEndian.Uint32(hdr[4:8]))
		if esize < 8 || esize > (16<<20) {
			return "", errors.New("invalid NIfTI extension size")
		}
		body := make([]byte, esize-8)
		if _, err := io.ReadFull(f, body); err != nil {
			return "", err
		}
		if ecode == 32 {
			if idx := indexOfZero(body); idx >= 0 {
				body = body[:idx]
			}
			return string(body), nil
		}
	}
	return "", errors.New("no CIFTI extension found")
}

func indexOfZero(b []byte) int {
	for i, c := range b {
		if c == 0 {
			return i
		}
	}
	return -1
}

// parseCiftiAxes maps each CIFTI matrix dimension to a role from the XML. CIFTI
// matrix dimension i corresponds to NIfTI dim[5+i]; on disk NIfTI dim[5] is the
// fast (column) axis and dim[6] the slow (row) axis, so dim-0 → cols, dim-1 →
// rows. Defensive: an unparseable map leaves the axis role "unknown".
func parseCiftiAxes(xml string, dim5, dim6 int) (rows, cols ciftiAxis) {
	cols = ciftiAxis{role: "unknown", size: dim5}
	rows = ciftiAxis{role: "unknown", size: dim6}
	for _, mm := range reMatrixMap.FindAllString(xml, -1) {
		applies := ""
		if m := reAppliesToDim.FindStringSubmatch(mm); m != nil {
			applies = m[1]
		}
		role := "unknown"
		if m := reMapDataType.FindStringSubmatch(mm); m != nil {
			role = strings.ToLower(strings.TrimPrefix(m[1], "CIFTI_INDEX_TYPE_"))
		}
		// Slice the XML region belonging to this map (up to the next map).
		region := regionAfter(xml, mm)
		axis := ciftiAxis{role: role}
		switch role {
		case "series":
			if m := reSeriesStep.FindStringSubmatch(mm); m != nil {
				step, _ := strconv.ParseFloat(m[1], 64)
				axis.step = step
			}
			if m := reSeriesUnit.FindStringSubmatch(mm); m != nil && strings.EqualFold(m[1], "HZ") {
				axis.step = 0 // spectral axis; leave as index
			}
		case "brain_models":
			for _, bm := range reBrainModel.FindAllString(region, -1) {
				name, count := "", 0
				if s := reBrainStruct.FindStringSubmatch(bm); s != nil {
					name = prettyBrainStructure(s[1])
				}
				if s := reIndexCount.FindStringSubmatch(bm); s != nil {
					count, _ = strconv.Atoi(s[1])
				}
				if count > 0 {
					axis.structures = append(axis.structures, ciftiBrainModel{Name: name, Count: count})
				}
			}
		case "parcels":
			names := reParcelName.FindAllStringSubmatch(region, -1)
			for i, n := range names {
				if i >= ciftiMaxParcelNames {
					break
				}
				axis.names = append(axis.names, n[1])
			}
		}
		// "0,1" applies to BOTH dims (a symmetric connectivity map).
		for _, d := range strings.Split(applies, ",") {
			switch strings.TrimSpace(d) {
			case "0":
				axis.size = dim5
				cols = axis
			case "1":
				axis.size = dim6
				rows = axis
			}
		}
	}
	return rows, cols
}

// regionAfter returns the XML slice from the start of match `mm` up to the next
// MatrixIndicesMap open tag, so BrainModel/Parcel children are attributed to the
// right axis.
func regionAfter(xml, mm string) string {
	start := strings.Index(xml, mm)
	if start < 0 {
		return xml
	}
	rest := xml[start+len(mm):]
	if next := reMatrixMap.FindStringIndex(rest); next != nil {
		return rest[:next[0]]
	}
	return rest
}

func prettyBrainStructure(name string) string {
	n := strings.TrimPrefix(strings.ToUpper(name), "CIFTI_STRUCTURE_")
	return strings.Title(strings.ToLower(strings.ReplaceAll(n, "_", " ")))
}

// classifyCifti names the file's kind + label from its two axis roles.
func classifyCifti(rows, cols ciftiAxis) (kind, label string) {
	a, b := cols.role, rows.role
	has := func(x, y string) bool { return (a == x && b == y) || (a == y && b == x) }
	switch {
	case has("series", "brain_models"):
		return "timeseries", "dense timeseries"
	case has("series", "parcels"):
		return "timeseries", "parcellated timeseries"
	case a == "brain_models" && b == "brain_models":
		return "connectivity", "dense connectivity"
	case a == "parcels" && b == "parcels":
		return "connectivity", "parcellated connectivity"
	case has("scalars", "brain_models"):
		return "scalar", "dense scalar"
	case has("labels", "brain_models"):
		return "scalar", "dense label"
	case has("scalars", "parcels"):
		return "scalar", "parcellated scalar"
	}
	return "matrix", "connectivity"
}

// parseCiftiMeta reads the NIfTI-2 header + CIFTI XML and assembles the matrix
// metadata that drives the views. Rows are the on-disk outer axis (dim[6]),
// columns the inner axis (dim[5]).
func parseCiftiMeta(path string) (ciftiMeta, error) {
	f, err := os.Open(path)
	if err != nil {
		return ciftiMeta{}, err
	}
	defer func() { _ = f.Close() }()
	// CIFTI is stored uncompressed (.nii); a rare .gz is handled by the readers.
	var rs io.ReadSeeker = f
	if isGzipPath(path) {
		return ciftiMeta{}, errors.New("gzipped CIFTI headers are read via the matrix reader")
	}
	header, _, err := readNiftiHeaderBytes(rs)
	if err != nil {
		return ciftiMeta{}, err
	}
	order, version, err := niftiHeaderVersion(header)
	if err != nil || version != 2 {
		return ciftiMeta{}, errors.New("not a NIfTI-2/CIFTI file")
	}
	dim5 := nifti2Dimension(order, header, 5)
	dim6 := nifti2Dimension(order, header, 6)
	if dim5 <= 0 || dim6 <= 0 {
		return ciftiMeta{}, fmt.Errorf("CIFTI has no 2-D matrix (dims %d × %d)", dim6, dim5)
	}
	datatype := int(int16(order.Uint16(header[12:14])))
	dtype, bytesPer, err := ciftiScalarType(datatype)
	if err != nil {
		return ciftiMeta{}, err
	}
	voxOffset := niftiInt64(order, header[168:176])
	if voxOffset < nifti2HeaderReadSize {
		voxOffset = nifti2HeaderReadSize
	}
	slope, inter := nifti2RescaleFromHeader(order, header)
	xml, xerr := niftiCiftiExtension(f)
	rows, cols := ciftiAxis{role: "unknown", size: dim6}, ciftiAxis{role: "unknown", size: dim5}
	if xerr == nil {
		rows, cols = parseCiftiAxes(xml, dim5, dim6)
	}
	kind, label := classifyCifti(rows, cols)
	// Carpet convention: brain locations on rows. If the XML put the location
	// axis on columns (scalars/series on rows), swap so the reader is uniform.
	if isLocationRole(cols.role) && !isLocationRole(rows.role) {
		rows, cols = cols, rows
		// NOTE: after a swap the on-disk (row,col) indexing no longer matches;
		// only symmetric or location-on-rows files reach the views, so we keep the
		// physical layout and treat "rows" as display rows below.
	}
	return ciftiMeta{
		kind: kind, label: label, rows: rows, cols: cols,
		voxOffset: voxOffset, order: order, dtype: dtype, bytesPer: bytesPer,
		slope: slope, inter: inter,
	}, nil
}

func isLocationRole(role string) bool { return role == "brain_models" || role == "parcels" }

func ciftiScalarType(datatype int) (string, int, error) {
	switch datatype {
	case 16:
		return "float32", 4, nil
	case 64:
		return "float64", 8, nil
	case 8:
		return "int32", 4, nil
	default:
		return "", 0, fmt.Errorf("unsupported CIFTI datatype %d", datatype)
	}
}

func isGzipPath(path string) bool {
	return strings.HasSuffix(strings.ToLower(strings.TrimSpace(path)), ".gz")
}

// ciftiMatrix is a bounded reader over the on-disk float matrix. Uncompressed
// files read rows with ReadAt (O(1) per row); a gzip file is decompressed once
// (capped) into memory. The physical layout is data[row*physCols + col].
type ciftiMatrix struct {
	f        *os.File
	buf      []byte // decompressed payload (gz only), starting at vox_offset
	base     int64  // vox_offset for the ReadAt path
	physRows int
	physCols int
	bytesPer int
	dtype    string
	order    binary.ByteOrder
	slope    float64
	inter    float64
}

func openCiftiMatrix(path string, m ciftiMeta) (*ciftiMatrix, error) {
	// Physical (on-disk) rows/cols follow dim[6] (outer) × dim[5] (inner). The
	// display "rows/cols" in meta may be swapped for scalar files, but the
	// endpoints below only serve timeseries (location-on-rows) and symmetric
	// connectivity, where physical rows == display rows.
	physCols := m.cols.size
	physRows := m.rows.size
	mat := &ciftiMatrix{
		base: m.voxOffset, physRows: physRows, physCols: physCols,
		bytesPer: m.bytesPer, dtype: m.dtype, order: m.order, slope: m.slope, inter: m.inter,
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	if !isGzipPath(path) {
		mat.f = f
		return mat, nil
	}
	defer func() { _ = f.Close() }()
	zr, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer func() { _ = zr.Close() }()
	if _, err := io.CopyN(io.Discard, zr, m.voxOffset); err != nil {
		return nil, err
	}
	need := int64(physRows) * int64(physCols) * int64(m.bytesPer)
	if need > ciftiGzipDecompCap {
		return nil, errors.New("gzipped CIFTI is too large to preview; download the original")
	}
	buf := make([]byte, need)
	if _, err := io.ReadFull(zr, buf); err != nil {
		return nil, err
	}
	mat.buf = buf
	return mat, nil
}

func (m *ciftiMatrix) close() {
	if m.f != nil {
		_ = m.f.Close()
	}
}

// row reads physical row r (physCols values) as float64, applying the rescale.
func (m *ciftiMatrix) row(r int) ([]float64, error) {
	if r < 0 || r >= m.physRows {
		return nil, fmt.Errorf("row %d out of range", r)
	}
	rowBytes := m.physCols * m.bytesPer
	raw := make([]byte, rowBytes)
	if m.f != nil {
		if _, err := m.f.ReadAt(raw, m.base+int64(r)*int64(rowBytes)); err != nil && err != io.EOF {
			return nil, err
		}
	} else {
		off := r * rowBytes
		if off+rowBytes > len(m.buf) {
			return nil, io.ErrUnexpectedEOF
		}
		copy(raw, m.buf[off:off+rowBytes])
	}
	out := make([]float64, m.physCols)
	for c := 0; c < m.physCols; c++ {
		out[c] = m.slope*m.decode(raw[c*m.bytesPer:]) + m.inter
	}
	return out, nil
}

func (m *ciftiMatrix) decode(b []byte) float64 {
	switch m.dtype {
	case "float64":
		return math.Float64frombits(m.order.Uint64(b))
	case "int32":
		return float64(int32(m.order.Uint32(b)))
	default: // float32
		return float64(math.Float32frombits(m.order.Uint32(b)))
	}
}

// evenIndices returns up to `count` evenly spaced indices in [0, n).
func evenIndices(n, count int) []int {
	if n <= count {
		out := make([]int, n)
		for i := range out {
			out[i] = i
		}
		return out
	}
	out := make([]int, count)
	for i := 0; i < count; i++ {
		out[i] = int(int64(i) * int64(n) / int64(count))
	}
	return out
}

type ciftiStructureBand struct {
	Name  string `json:"name"`
	Start int    `json:"start"`
	End   int    `json:"end"`
}

// buildCarpet samples up to maxRows rows and maxCols columns, z-scores each row,
// clips to ±clipZ, and packs to uint8. Structure bands are remapped into the
// sampled-row space so the frontend can group and label them.
func buildCarpet(mat *ciftiMatrix, m ciftiMeta, maxRows, maxCols int) (map[string]any, error) {
	rowIdx := evenIndices(mat.physRows, maxRows)
	colIdx := evenIndices(mat.physCols, maxCols)
	R, C := len(rowIdx), len(colIdx)
	u8 := make([]byte, R*C)
	for ri, r := range rowIdx {
		full, err := mat.row(r)
		if err != nil {
			return nil, err
		}
		// mean/std over the SAMPLED columns.
		var sum, sumsq float64
		for _, c := range colIdx {
			v := full[c]
			sum += v
			sumsq += v * v
		}
		mean := sum / float64(C)
		std := math.Sqrt(math.Max(sumsq/float64(C)-mean*mean, 0))
		if std < 1e-9 {
			std = 1
		}
		for ci, c := range colIdx {
			z := (full[c] - mean) / std
			z = math.Max(-ciftiCarpetClipZ, math.Min(ciftiCarpetClipZ, z))
			u8[ri*C+ci] = byte((z + ciftiCarpetClipZ) / (2 * ciftiCarpetClipZ) * 255)
		}
	}
	bands := remapStructures(m.rows.structures, mat.physRows, R)
	colAxis := map[string]any{"role": m.cols.role, "size": m.cols.size, "sampled": C}
	if m.cols.step > 0 {
		colAxis["step"] = m.cols.step
		colAxis["unit"] = "s"
	}
	return map[string]any{
		"rows": R, "cols": C, "clip_z": ciftiCarpetClipZ,
		"source_rows": mat.physRows, "structures": bands, "column_axis": colAxis,
		"data": base64.StdEncoding.EncodeToString(u8),
	}, nil
}

func remapStructures(src []ciftiBrainModel, physRows, sampledRows int) []ciftiStructureBand {
	if len(src) == 0 || physRows <= 0 {
		return []ciftiStructureBand{}
	}
	out := make([]ciftiStructureBand, 0, len(src))
	acc := 0
	for _, s := range src {
		start := acc * sampledRows / physRows
		end := (acc + s.Count) * sampledRows / physRows
		if end > sampledRows {
			end = sampledRows
		}
		if end > start {
			out = append(out, ciftiStructureBand{Name: s.Name, Start: start, End: end})
		}
		acc += s.Count
	}
	return out
}

// buildConnectivity produces an N×N matrix. For a *conn file it samples the
// stored matrix directly; for a timeseries it correlates N downsampled node
// signals. Values are returned as float32 with their range for the colour scale.
func buildConnectivity(mat *ciftiMatrix, m ciftiMeta, nodes int) (map[string]any, error) {
	var conn []float32
	var n int
	var labels []string
	if m.kind == "connectivity" {
		idx := evenIndices(mat.physRows, nodes)
		colSel := evenIndices(mat.physCols, nodes)
		n = len(idx)
		nc := len(colSel)
		conn = make([]float32, n*nc)
		for ri, r := range idx {
			full, err := mat.row(r)
			if err != nil {
				return nil, err
			}
			for ci, c := range colSel {
				conn[ri*nc+ci] = float32(full[c])
			}
		}
		labels = sampleNames(m.rows.names, idx)
		n = nc // square display
	} else {
		// Timeseries → correlation of N evenly-spaced node signals (z-scored).
		idx := evenIndices(mat.physRows, nodes)
		n = len(idx)
		sig := make([][]float64, n)
		for i, r := range idx {
			full, err := mat.row(r)
			if err != nil {
				return nil, err
			}
			sig[i] = zscore(full)
		}
		conn = make([]float32, n*n)
		T := float64(mat.physCols)
		for i := 0; i < n; i++ {
			conn[i*n+i] = 1
			for j := i + 1; j < n; j++ {
				var dot float64
				for t := 0; t < mat.physCols; t++ {
					dot += sig[i][t] * sig[j][t]
				}
				r := float32(dot / T)
				conn[i*n+j], conn[j*n+i] = r, r
			}
		}
		labels = structureLabelsForRows(m.rows.structures, mat.physRows, idx)
	}
	lo, hi := float32(math.Inf(1)), float32(math.Inf(-1))
	for _, v := range conn {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	buf := make([]byte, len(conn)*4)
	for i, v := range conn {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	out := map[string]any{
		"n": n, "min": lo, "max": hi, "dtype": "float32",
		"computed": m.kind != "connectivity",
		"data":     base64.StdEncoding.EncodeToString(buf),
	}
	if len(labels) > 0 {
		out["labels"] = labels
	}
	return out, nil
}

func zscore(v []float64) []float64 {
	var sum, sumsq float64
	for _, x := range v {
		sum += x
		sumsq += x * x
	}
	mean := sum / float64(len(v))
	std := math.Sqrt(math.Max(sumsq/float64(len(v))-mean*mean, 0))
	if std < 1e-9 {
		std = 1
	}
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = (x - mean) / std
	}
	return out
}

func sampleNames(names []string, idx []int) []string {
	if len(names) == 0 {
		return nil
	}
	out := make([]string, 0, len(idx))
	for _, i := range idx {
		if i < len(names) {
			out = append(out, names[i])
		}
	}
	return out
}

// structureLabelsForRows tags each sampled node with the structure it falls in.
func structureLabelsForRows(src []ciftiBrainModel, physRows int, idx []int) []string {
	if len(src) == 0 {
		return nil
	}
	bounds := make([]int, 0, len(src)+1)
	acc := 0
	for _, s := range src {
		bounds = append(bounds, acc)
		acc += s.Count
	}
	bounds = append(bounds, acc)
	out := make([]string, len(idx))
	for i, r := range idx {
		out[i] = src[len(src)-1].Name
		for k := 0; k < len(src); k++ {
			if r >= bounds[k] && r < bounds[k+1] {
				out[i] = src[k].Name
				break
			}
		}
	}
	return out
}

// --- HTTP -------------------------------------------------------------------

func (deps ServerDeps) resolveCiftiRequest(w http.ResponseWriter, r *http.Request) (resourceRecord, string, ciftiMeta, bool) {
	_, record, path, ok := deps.resolveUploadServingRequest(w, r)
	if !ok {
		return resourceRecord{}, "", ciftiMeta{}, false
	}
	if info, isCifti := niftiCiftiPeek(path, record.OriginalName); !isCifti {
		_ = info
		writeError(w, http.StatusUnsupportedMediaType, errors.New("not a CIFTI file"))
		return resourceRecord{}, "", ciftiMeta{}, false
	}
	meta, err := parseCiftiMeta(path)
	if err != nil {
		writeError(w, http.StatusUnsupportedMediaType, err)
		return resourceRecord{}, "", ciftiMeta{}, false
	}
	return record, path, meta, true
}

func (deps ServerDeps) handleGetUploadCiftiCarpet(w http.ResponseWriter, r *http.Request) {
	_, path, meta, ok := deps.resolveCiftiRequest(w, r)
	if !ok {
		return
	}
	maxRows := int(clampInt64(parseInt64Query(r, "max_rows", ciftiCarpetMaxRows), 16, ciftiCarpetMaxRows))
	maxCols := int(clampInt64(parseInt64Query(r, "max_cols", ciftiCarpetMaxCols), 8, ciftiCarpetMaxCols))
	mat, err := openCiftiMatrix(path, meta)
	if err != nil {
		writeError(w, http.StatusUnsupportedMediaType, err)
		return
	}
	defer mat.close()
	payload, err := buildCarpet(mat, meta, maxRows, maxCols)
	if err != nil {
		writeError(w, http.StatusUnsupportedMediaType, err)
		return
	}
	payload["cifti_type"] = meta.label
	writeJSON(w, http.StatusOK, payload)
}

func (deps ServerDeps) handleGetUploadCiftiConnectivity(w http.ResponseWriter, r *http.Request) {
	_, path, meta, ok := deps.resolveCiftiRequest(w, r)
	if !ok {
		return
	}
	nodes := int(clampInt64(parseInt64Query(r, "nodes", ciftiConnMaxNodes), 8, ciftiConnMaxNodes))
	mat, err := openCiftiMatrix(path, meta)
	if err != nil {
		writeError(w, http.StatusUnsupportedMediaType, err)
		return
	}
	defer mat.close()
	payload, err := buildConnectivity(mat, meta, nodes)
	if err != nil {
		writeError(w, http.StatusUnsupportedMediaType, err)
		return
	}
	payload["cifti_type"] = meta.label
	writeJSON(w, http.StatusOK, payload)
}

// writeCiftiViewer returns the kind:"cifti" descriptor: what the file is, its
// matrix shape and structures, and which views (carpet / connectivity) apply.
func (deps ServerDeps) writeCiftiViewer(w http.ResponseWriter, record resourceRecord, info ciftiInfo, path string) {
	fileIDSegment := url.PathEscape(record.FileID)
	meta, err := parseCiftiMeta(path)
	views := []string{}
	structures := []map[string]any{}
	label := info.label
	var rowsN, colsN int
	colAxis := map[string]any{}
	if err == nil {
		label = meta.label
		rowsN, colsN = meta.rows.size, meta.cols.size
		switch meta.kind {
		case "timeseries":
			views = []string{"carpet", "connectivity"}
		case "connectivity":
			views = []string{"connectivity"}
		default:
			views = []string{"carpet"}
		}
		for _, s := range meta.rows.structures {
			structures = append(structures, map[string]any{"name": s.Name, "count": s.Count})
		}
		colAxis = map[string]any{"role": meta.cols.role, "size": meta.cols.size}
		if meta.cols.step > 0 {
			colAxis["step"] = meta.cols.step
			colAxis["unit"] = "s"
		}
	} else {
		// Header parsed but XML/meta failed: still offer the carpet of the raw matrix.
		views = []string{"carpet"}
	}
	message := fmt.Sprintf(
		"CIFTI-2 %s — brain data on grayordinates (cortical-surface vertices + subcortical "+
			"voxels). Shown as its data matrix; the anatomical surface render needs Connectome "+
			"Workbench.", label,
	)
	writeJSON(w, http.StatusOK, map[string]any{
		"kind":          "cifti",
		"decodable":     true,
		"file_id":       record.FileID,
		"original_name": record.OriginalName,
		"format":        "cifti",
		"modality":      "connectivity",
		"cifti_type":    label,
		"views":         views,
		"rows":          rowsN,
		"cols":          colsN,
		"structures":    structures,
		"column_axis":   colAxis,
		"service_urls": map[string]any{
			"carpet":       "/v2/uploads/" + fileIDSegment + "/cifti/carpet",
			"connectivity": "/v2/uploads/" + fileIDSegment + "/cifti/connectivity",
			"download":     "/v2/resources/" + fileIDSegment + "/download",
		},
		"message": message,
	})
}
