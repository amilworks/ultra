package httpapi

import (
	"compress/gzip"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
)

// NIfTI-2 support + CIFTI detection.
//
// The legacy reader (handlers.go) parses only NIfTI-1 (sizeof_hdr 348, magic
// "n+1"/"ni1", 16-bit dims, 32-bit floats). NIfTI-2 is a distinct binary layout
// (sizeof_hdr 540, magic "n+2"/"ni2", 64-bit dims, 64-bit floats) used by modern
// neuroimaging tools and — always — by CIFTI grayordinate files (.dtseries.nii
// et al.). Without this, every NIfTI-2 upload failed the header check and the
// viewer returned "cannot view".
//
// Two outcomes are added:
//   - A real NIfTI-2 *spatial* volume parses into the SAME niftiGeometry the rest
//     of the pipeline consumes, so it views exactly like a NIfTI-1 volume.
//   - A CIFTI file (NIfTI-2 whose data is a grayordinate/parcel matrix, not a
//     spatial image) is detected up front and the viewer returns an honest
//     "unsupported — needs a surface viewer" descriptor instead of a broken 1×1
//     canvas or a hard error.

const (
	nifti1HeaderSize = 348
	nifti2HeaderSize = 540
	// NIfTI-1 path reads 352 bytes: the 348-byte header + the 4-byte extension
	// flag that precedes the voxel data. niftiHeaderReadSize (handlers.go) holds
	// that value and doubles as the NIfTI-1 vox_offset floor and stream-skip base.
	// NIfTI-2's equivalent is 540 + 4 = 544.
	nifti2HeaderReadSize = nifti2HeaderSize + 4
)

// niftiHeaderVersion inspects sizeof_hdr (int32 at offset 0) to distinguish
// NIfTI-1 (348) from NIfTI-2 (540), tolerating either byte order. It is the
// single source of truth for which parser a header goes to.
func niftiHeaderVersion(data []byte) (binary.ByteOrder, int, error) {
	if len(data) < 4 {
		return binary.LittleEndian, 0, errors.New("NIfTI header is too small")
	}
	for _, order := range []binary.ByteOrder{binary.LittleEndian, binary.BigEndian} {
		switch order.Uint32(data[0:4]) {
		case nifti1HeaderSize:
			return order, 1, nil
		case nifti2HeaderSize:
			return order, 2, nil
		}
	}
	return binary.LittleEndian, 0, errors.New("NIfTI header size is invalid")
}

func niftiInt32(order binary.ByteOrder, data []byte) int {
	return int(int32(order.Uint32(data)))
}

func niftiInt64(order binary.ByteOrder, data []byte) int64 {
	return int64(order.Uint64(data))
}

func niftiFloat64(order binary.ByteOrder, data []byte) float64 {
	return math.Float64frombits(order.Uint64(data))
}

// nifti2Dimension reads dim[i] (a 64-bit dimension) and clamps a negative/absurd
// value to 0 so the caller's geometry guards reject it cleanly.
func nifti2Dimension(order binary.ByteOrder, header []byte, i int) int {
	off := 16 + i*8 // dim[] starts at offset 16, 8 bytes per entry
	v := niftiInt64(order, header[off:off+8])
	if v < 0 || v > math.MaxInt32 {
		return 0
	}
	return int(v)
}

func nifti2Spacing(order binary.ByteOrder, header []byte, i int) float64 {
	off := 104 + i*8 // pixdim[] starts at offset 104
	v := niftiFloat64(order, header[off:off+8])
	if !numberIsFinite(v) || v <= 0 {
		return 1
	}
	return v
}

// parseNifti2Geometry maps a NIfTI-2 header into the shared niftiGeometry. Field
// offsets follow nifti2.h exactly; the dim->W/H/D/T/C mapping and datatype table
// match the NIfTI-1 path so the two produce interchangeable geometry.
func parseNifti2Geometry(order binary.ByteOrder, header []byte) (niftiGeometry, error) {
	if len(header) < nifti2HeaderSize {
		return niftiGeometry{}, errors.New("NIfTI-2 file is too small")
	}
	magic := string(header[4:8])
	if magic != "n+2\x00" && magic != "ni2\x00" {
		return niftiGeometry{}, fmt.Errorf("unsupported NIfTI-2 magic %q", strings.TrimRight(magic, "\x00"))
	}
	dim0 := nifti2Dimension(order, header, 0)
	if dim0 < 2 {
		return niftiGeometry{}, fmt.Errorf("unsupported NIfTI dimension count %d", dim0)
	}
	width := nifti2Dimension(order, header, 1)
	height := nifti2Dimension(order, header, 2)
	depth := 1
	if dim0 >= 3 {
		depth = nifti2Dimension(order, header, 3)
	}
	timeCount := 1
	if dim0 >= 4 {
		timeCount = nifti2Dimension(order, header, 4)
	}
	channelCount := 1
	if dim0 >= 5 {
		channelCount = nifti2Dimension(order, header, 5)
	}
	if width <= 0 || height <= 0 || depth <= 0 {
		return niftiGeometry{}, fmt.Errorf("invalid NIfTI dimensions %dx%dx%d", width, height, depth)
	}
	if timeCount <= 0 {
		timeCount = 1
	}
	if channelCount <= 0 {
		channelCount = 1
	}
	datatype := int(int16(order.Uint16(header[12:14])))
	dtype, bytesPerVoxel, err := niftiScalarType(datatype)
	if err != nil {
		return niftiGeometry{}, err
	}
	voxOffset := niftiInt64(order, header[168:176])
	if voxOffset < nifti2HeaderReadSize {
		voxOffset = nifti2HeaderReadSize
	}
	spacingX := nifti2Spacing(order, header, 1)
	spacingY := nifti2Spacing(order, header, 2)
	spacingZ := nifti2Spacing(order, header, 3)
	affine, affineCode := nifti2AffineFromHeader(order, header, spacingX, spacingY, spacingZ)
	sclSlope, sclInter := nifti2RescaleFromHeader(order, header)
	return niftiGeometry{
		order:         order,
		width:         width,
		height:        height,
		depth:         depth,
		timeCount:     timeCount,
		channelCount:  channelCount,
		dtype:         dtype,
		bytesPerVoxel: bytesPerVoxel,
		voxOffset:     voxOffset,
		spacingX:      spacingX,
		spacingY:      spacingY,
		spacingZ:      spacingZ,
		affine:        affine,
		affineCode:    affineCode,
		sclSlope:      sclSlope,
		sclInter:      sclInter,
		spaceUnit:     nifti2SpaceUnitFromHeader(order, header),
	}, nil
}

// nifti2AffineFromHeader mirrors niftiAffineFromHeader (handlers.go) at NIfTI-2
// offsets and widths: srow float64 at 400/432/464, qform/sform int32 at 344/348,
// quaternion float64 at 352/360/368, qoffset at 376/384/392, qfac = pixdim[0].
func nifti2AffineFromHeader(order binary.ByteOrder, h []byte, sx, sy, sz float64) ([12]float64, int) {
	var affine [12]float64
	qformCode := niftiInt32(order, h[344:348])
	sformCode := niftiInt32(order, h[348:352])
	if sformCode > 0 {
		nonZero := false
		for c := 0; c < 4; c++ {
			affine[0*4+c] = niftiFloat64(order, h[400+c*8:408+c*8])
			affine[1*4+c] = niftiFloat64(order, h[432+c*8:440+c*8])
			affine[2*4+c] = niftiFloat64(order, h[464+c*8:472+c*8])
		}
		for i := 0; i < 9; i++ {
			if affine[(i/3)*4+(i%3)] != 0 {
				nonZero = true
				break
			}
		}
		if nonZero {
			return affine, 3
		}
		affine = [12]float64{}
	}
	if qformCode > 0 {
		b := niftiFloat64(order, h[352:360])
		c := niftiFloat64(order, h[360:368])
		d := niftiFloat64(order, h[368:376])
		a2 := 1.0 - (b*b + c*c + d*d)
		a := 0.0
		if a2 > 0 {
			a = math.Sqrt(a2)
		}
		qfac := niftiFloat64(order, h[104:112]) // pixdim[0]
		if qfac >= 0 {
			qfac = 1
		} else {
			qfac = -1
		}
		r := [3][3]float64{
			{a*a + b*b - c*c - d*d, 2 * (b*c - a*d), 2 * (b*d + a*c)},
			{2 * (b*c + a*d), a*a + c*c - b*b - d*d, 2 * (c*d - a*b)},
			{2 * (b*d - a*c), 2 * (c*d + a*b), a*a + d*d - b*b - c*c},
		}
		off := [3]float64{
			niftiFloat64(order, h[376:384]),
			niftiFloat64(order, h[384:392]),
			niftiFloat64(order, h[392:400]),
		}
		scale := [3]float64{sx, sy, sz * qfac}
		for i := 0; i < 3; i++ {
			affine[i*4+0] = r[i][0] * scale[0]
			affine[i*4+1] = r[i][1] * scale[1]
			affine[i*4+2] = r[i][2] * scale[2]
			affine[i*4+3] = off[i]
		}
		return affine, 2
	}
	affine[0] = sx
	affine[5] = sy
	affine[10] = sz
	return affine, 0
}

func nifti2RescaleFromHeader(order binary.ByteOrder, header []byte) (float64, float64) {
	slope := niftiFloat64(order, header[176:184])
	inter := niftiFloat64(order, header[184:192])
	if !numberIsFinite(slope) || slope == 0 {
		return 1, 0
	}
	if !numberIsFinite(inter) {
		inter = 0
	}
	return slope, inter
}

// nifti2SpaceUnitFromHeader reads the spatial bits (0x07) of xyzt_units (int32 at
// offset 500; the unit lives in the low byte), matching the NIfTI-1 decoder.
func nifti2SpaceUnitFromHeader(order binary.ByteOrder, header []byte) string {
	if len(header) < 504 {
		return ""
	}
	switch byte(order.Uint32(header[500:504])) & 0x07 {
	case 1:
		return "m"
	case 2:
		return "mm"
	case 3:
		return "um"
	default:
		return ""
	}
}

// readNiftiHeaderBytes reads a full NIfTI header from a stream, sized to its
// version, and reports how many bytes it consumed — which the streaming slab
// reader needs so its skip-to-voxels math stays correct (NIfTI-1: 352; NIfTI-2:
// 544). It always reads the NIfTI-1 amount first, then tops up for NIfTI-2, so a
// NIfTI-1 file with a small vox_offset is never over-read.
func readNiftiHeaderBytes(r io.Reader) (header []byte, consumed int, err error) {
	buf := make([]byte, niftiHeaderReadSize)
	if _, err = io.ReadFull(r, buf); err != nil {
		return nil, 0, err
	}
	if _, version, verr := niftiHeaderVersion(buf); verr == nil && version == 2 {
		more := make([]byte, nifti2HeaderReadSize-niftiHeaderReadSize)
		if _, err = io.ReadFull(r, more); err != nil {
			return nil, 0, fmt.Errorf("read NIfTI-2 header: %w", err)
		}
		return append(buf, more...), nifti2HeaderReadSize, nil
	}
	return buf, niftiHeaderReadSize, nil
}

// --- CIFTI --------------------------------------------------------------------
// CIFTI (Connectivity Informatics Technology Initiative) files are NIfTI-2
// containers whose payload is a grayordinate/parcel MATRIX, not a spatial volume:
// the spatial dims are 1×1×1 and the data maps to cortical-surface vertices +
// subcortical voxels described by an XML extension. They cannot be shown as
// slices without the accompanying surface meshes, so we detect them and return an
// honest "download to open in a surface viewer" descriptor.

// niftiCiftiIntentLabels maps the NIFTI_INTENT connectivity codes to a short
// human label. Anything in [3000, 3100) that is not listed is still treated as
// CIFTI (generic label).
var niftiCiftiIntentLabels = map[int]string{
	3001: "dense connectivity",
	3002: "dense timeseries",
	3003: "parcellated connectivity",
	3004: "parcellated timeseries",
	3006: "dense scalar",
	3007: "dense label",
	3008: "parcellated scalar",
	3009: "parcellated dense",
	3010: "dense parcellated",
	3011: "parcellated connectivity series",
	3012: "parcellated connectivity scalar",
}

type ciftiInfo struct {
	intentCode int
	label      string
	matrixDims []int64 // the non-trivial (>1) dims: the connectivity matrix shape
}

// niftiCiftiExtensions are the double-extension names CIFTI uses. Used as a
// fallback when the header is unavailable; the intent code is authoritative.
var niftiCiftiExtensions = []string{
	".dtseries.nii", ".dscalar.nii", ".dlabel.nii", ".dconn.nii",
	".pconn.nii", ".ptseries.nii", ".pscalar.nii", ".plabel.nii",
	".pdconn.nii", ".dpconn.nii", ".pconnseries.nii", ".pconnscalar.nii",
}

func isCiftiName(name string) bool {
	lower := strings.ToLower(strings.TrimSpace(name))
	lower = strings.TrimSuffix(lower, ".gz")
	for _, ext := range niftiCiftiExtensions {
		if strings.HasSuffix(lower, ext) {
			return true
		}
	}
	return false
}

// niftiCiftiFromHeader returns CIFTI info when the NIfTI-2 header carries a CIFTI
// connectivity intent code (or, as a fallback, the name matches a CIFTI double
// extension). Non-CIFTI headers return ok=false.
func niftiCiftiFromHeader(order binary.ByteOrder, header []byte, name string) (ciftiInfo, bool) {
	if len(header) < nifti2HeaderSize {
		if isCiftiName(name) {
			return ciftiInfo{label: "connectivity"}, true
		}
		return ciftiInfo{}, false
	}
	intent := niftiInt32(order, header[504:508])
	isCifti := (intent >= 3000 && intent < 3100) || isCiftiName(name)
	if !isCifti {
		return ciftiInfo{}, false
	}
	label := niftiCiftiIntentLabels[intent]
	if label == "" {
		label = "connectivity"
	}
	// Scan all seven possible dims (not just dim[0] of them): different CIFTI
	// types place the matrix in different slots, and the trivial ones are 1.
	var matrix []int64
	for i := 1; i <= 7; i++ {
		if size := nifti2Dimension(order, header, i); size > 1 {
			matrix = append(matrix, int64(size))
		}
	}
	return ciftiInfo{intentCode: intent, label: label, matrixDims: matrix}, true
}

// niftiCiftiPeek opens the file, reads its header, and reports CIFTI info when it
// is a CIFTI file. Errors (unreadable, not NIfTI-2) return ok=false so the caller
// falls through to the normal NIfTI-1/2 volume path.
func niftiCiftiPeek(path string, originalName string) (ciftiInfo, bool) {
	file, err := os.Open(path)
	if err != nil {
		return ciftiInfo{}, false
	}
	defer func() { _ = file.Close() }()
	var reader io.Reader = file
	if strings.HasSuffix(strings.ToLower(strings.TrimSpace(path)), ".gz") {
		gzr, gerr := gzip.NewReader(file)
		if gerr != nil {
			return ciftiInfo{}, false
		}
		defer func() { _ = gzr.Close() }()
		reader = gzr
	}
	header, _, err := readNiftiHeaderBytes(reader)
	if err != nil {
		return ciftiInfo{}, false
	}
	order, version, err := niftiHeaderVersion(header)
	if err != nil || version != 2 {
		return ciftiInfo{}, false
	}
	return niftiCiftiFromHeader(order, header, originalName)
}

// The CIFTI viewer descriptor (kind:"cifti") lives in cifti.go, which parses the
// full matrix metadata (axis roles, structures, TR) needed to drive the carpet /
// connectivity views.
