package httpapi

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"hash/crc32"
	"image"
	"image/color"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"
	"io"
	"log/slog"
	"math"
	"mime"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
	"github.com/go-chi/chi/v5"
	_ "golang.org/x/image/bmp"
	xdraw "golang.org/x/image/draw"
	_ "golang.org/x/image/webp"
	"golang.org/x/sync/singleflight"
)

// pyramidPlainImageMinBytes is the size past which a plain raster image (JPEG/PNG)
// is worth converting into a tiled pyramid. Below it, the direct path is faster.
const pyramidPlainImageMinBytes = 16 << 20 // 16 MiB

const (
	thumbnailMaxDimension    = 512
	thumbnailMaxEncodedBytes = int64(8 << 20)
	rasterThumbnailMaxInput  = int64(32 << 20)
	rasterThumbnailMaxPixels = int64(32_000_000)
	rasterThumbnailMaxAxis   = 32_768
	rasterThumbnailHeaderCap = int64(256 << 10)
	rasterThumbnailMaxChunks = 1_024

	niftiThumbnailMaxPlaneBytes   = int64(24 << 20)
	niftiThumbnailMaxAxis         = 16_384
	niftiThumbnailMaxGzipWork     = int64(256 << 20)
	thumbnailDecodeBytesPerPixel  = int64(8)
	defaultThumbnailInFlightBytes = int64(256 << 20)
)

var errThumbnailAdmission = errors.New("thumbnail in-flight byte budget is exhausted")

func newThumbnailInFlightBudgetFromEnv() *byteAdmissionBudget {
	maxBytes := defaultThumbnailInFlightBytes
	if raw := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_THUMBNAIL_INFLIGHT_BYTES")); raw != "" {
		if parsed, err := strconv.ParseInt(raw, 10, 64); err == nil && parsed >= 0 {
			maxBytes = parsed
		}
	}
	return newByteAdmissionBudget(maxBytes)
}

var thumbnailInFlightBudget = newThumbnailInFlightBudgetFromEnv()

// This file routes the viewer-facing image endpoints (/viewer, /slice,
// /scalar-volume) through the libbioimage image service when it is configured.
// Interactive viewer handlers retain their legacy fallback behavior. Resource
// thumbnails are deliberately fail-closed and never serve original source bytes
// when a renderer or sidecar is unavailable.

// derivedPyramidName is the deterministic destination hint sent to the worker.
// Publication never creates this mutable path: the worker commits an immutable,
// digest-named artifact through derivedPyramidManifestName instead.
func derivedPyramidName(fileID string) string { return fileID + "__pyramid.tif" }

// derivedPyramidFailedName is the sidecar the convert worker writes (see
// imaging/worker.py _failure_marker_path) when a resource's pyramid derivation
// PERMANENTLY fails — e.g. a source format this engine build cannot decode. Its
// presence + mtime let the control plane back off re-enqueueing a doomed conversion.
func derivedPyramidFailedName(fileID string) string { return fileID + "__pyramid.failed" }

func derivedPyramidFailedMarkerPath(root, fileID string) string {
	return filepath.Join(root, "derived", derivedPyramidFailedName(fileID))
}

// pyramidFailureBackoff is how long a recorded permanent-derivation-failure suppresses
// re-enqueueing. Long enough that a poison source (which always fails) can't keep
// burning the engine on every viewer open, short enough that a transient cause (an
// engine deploy, a since-fixed bug) is retried within the hour. Override with
// ULTRA_CONTROL_PYRAMID_FAILURE_BACKOFF_SECONDS.
const pyramidFailureBackoff = time.Hour

var derivativeFailureCodePattern = regexp.MustCompile(`^[a-z][a-z0-9_]{0,63}$`)

func pyramidFailureBackoffWindow() time.Duration {
	if raw := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_PYRAMID_FAILURE_BACKOFF_SECONDS")); raw != "" {
		if secs, err := strconv.Atoi(raw); err == nil && secs >= 0 {
			return time.Duration(secs) * time.Second
		}
	}
	return pyramidFailureBackoff
}

// recentPyramidFailure reports whether a permanent derivation failure was recorded for
// this resource within the backoff window. Failure-isolated: any stat error falls
// through to false (never blocks serving). A zero/disabled window never suppresses.
type derivativeFailureMarker struct {
	Schema          string                   `json:"schema"`
	ResourceID      string                   `json:"resource_id"`
	SourceSHA256    string                   `json:"source_sha256"`
	SourceSizeBytes int64                    `json:"source_size_bytes"`
	ConversionSpec  derivativeConversionSpec `json:"conversion_spec"`
	Code            string                   `json:"code"`
}

func decodeDerivativeFailureMarker(data []byte) (derivativeFailureMarker, error) {
	strict := json.NewDecoder(bytes.NewReader(data))
	strict.UseNumber()
	if err := consumeStrictJSONValue(strict); err != nil {
		return derivativeFailureMarker{}, err
	}
	if _, err := strict.Token(); err != io.EOF {
		if err == nil {
			return derivativeFailureMarker{}, errors.New("failure marker contains trailing JSON")
		}
		return derivativeFailureMarker{}, err
	}
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return derivativeFailureMarker{}, err
	}
	if err := exactJSONKeys(raw, "schema", "resource_id", "source_sha256", "source_size_bytes", "conversion_spec", "code"); err != nil {
		return derivativeFailureMarker{}, err
	}
	if err := exactJSONKeys(raw["conversion_spec"], "tile_size", "compression", "layout", "fmt"); err != nil {
		return derivativeFailureMarker{}, err
	}
	var marker derivativeFailureMarker
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&marker); err != nil {
		return derivativeFailureMarker{}, err
	}
	return marker, nil
}

func recentPyramidFailure(root string, record resourceRecord, now time.Time) bool {
	window := pyramidFailureBackoffWindow()
	if window <= 0 {
		return false
	}
	path := derivedPyramidFailedMarkerPath(root, record.FileID)
	file, generation, err := openRegularNoFollow(path)
	if err != nil || generation.SizeBytes <= 0 || generation.SizeBytes > 16<<10 {
		return false
	}
	data, readErr := io.ReadAll(io.LimitReader(file, (16<<10)+1))
	closeErr := file.Close()
	if readErr != nil || closeErr != nil || int64(len(data)) != generation.SizeBytes {
		return false
	}
	marker, markerErr := decodeDerivativeFailureMarker(data)
	if markerErr != nil || marker.Schema != "ultra.image-derived-pyramid-failure.v1" || marker.ResourceID != record.FileID || marker.SourceSHA256 != strings.ToLower(strings.TrimSpace(record.SHA256)) || marker.SourceSizeBytes != record.SizeBytes || marker.ConversionSpec != (derivativeConversionSpec{TileSize: 512, Compression: "lzw", Layout: "topdirs", Format: "auto"}) || !derivativeFailureCodePattern.MatchString(marker.Code) {
		return false
	}
	return now.Sub(time.Unix(0, generation.MtimeNS)) < window
}

// clearPyramidFailureMarker removes the failure sidecar so a derivation can be retried
// (called on an explicit re-derive request — the operator/serve-time escape hatch).
func clearPyramidFailureMarker(root, fileID string) {
	_ = os.Remove(derivedPyramidFailedMarkerPath(root, fileID))
}

// imageServiceStatusError carries the HTTP status of a non-200 image-service
// response so callers can distinguish "the engine recognized the file but cannot
// decode it" (415/422 — a permanent, format-level failure) from a transport error
// or sidecar outage (which should fall back to the legacy native path instead).
type imageServiceStatusError struct {
	status int
	msg    string
}

func (e *imageServiceStatusError) Error() string { return e.msg }

// imageServiceUndecodable reports whether err is an image-service response that
// means the source format cannot be decoded by this engine build (HTTP 415/422),
// as opposed to a dial/timeout error. build_viewer_info raises (-> 415) on a file
// with no pixel geometry, and the convert/render path returns 422 for the same.
func imageServiceUndecodable(err error) bool {
	var se *imageServiceStatusError
	if errors.As(err, &se) {
		return se.status == http.StatusUnsupportedMediaType || se.status == http.StatusUnprocessableEntity
	}
	return false
}

// imageServiceGetJSON fetches and decodes a JSON object from the image service.
func (deps ServerDeps) imageServiceGetJSON(ctx context.Context, endpoint string, query url.Values) (map[string]any, error) {
	base := strings.TrimRight(strings.TrimSpace(deps.ImageServiceURL), "/")
	target := base + endpoint
	if encoded := query.Encode(); encoded != "" {
		target += "?" + encoded
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	if err != nil {
		return nil, err
	}
	resp, err := imageServiceHTTPClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, &imageServiceStatusError{
			status: resp.StatusCode,
			msg:    fmt.Sprintf("image service %s -> %d: %s", endpoint, resp.StatusCode, strings.TrimSpace(string(body))),
		}
	}
	var out map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return out, nil
}

// cachedImageServiceViewerInfo fetches /viewerinfo through the in-process image
// response cache (keyed on path + file stat-stamp), so a repeated viewer open serves
// the metadata from this node's RAM instead of re-hitting the sidecar. On a large
// microscopy source over NFS, /viewerinfo is a multi-second cold decode; worse, the
// image service's per-worker caches don't share across its N worker processes, so a
// given open lands warm only by luck. A single shared cache here makes repeat opens
// reliably instant. Returns a freshly-decoded map each call so callers may mutate it.
func (deps ServerDeps) cachedImageServiceViewerInfo(ctx context.Context, path string) (map[string]any, error) {
	return deps.cachedImageServiceViewerInfoVia(ctx, path, deps.imageCache)
}

var imageViewerInfoFlights singleflight.Group

func (deps ServerDeps) cachedImageServiceViewerInfoVia(
	ctx context.Context,
	path string,
	cache *imageResponseCache,
) (map[string]any, error) {
	query := url.Values{"path": {path}}
	if cache == nil {
		return deps.imageServiceGetJSON(ctx, "/viewerinfo", query)
	}
	key, ok := imageCacheKey("/viewerinfo", query)
	if !ok {
		return deps.imageServiceGetJSON(ctx, "/viewerinfo", query)
	}
	if resp, hit := cache.get(key); hit {
		var out map[string]any
		if err := json.Unmarshal(resp.body, &out); err == nil {
			return out, nil
		}
		// Corrupt cached entry: fall through and recompute.
	}
	flightKey := strings.TrimRight(strings.TrimSpace(deps.ImageServiceURL), "/") + "|" + key
	value, err, _ := imageViewerInfoFlights.Do(flightKey, func() (any, error) {
		// The shared load must not inherit the first waiter's cancellation; otherwise
		// one disconnected client cancels metadata delivery for every joined viewer.
		out, fetchErr := deps.imageServiceGetJSON(context.WithoutCancel(ctx), "/viewerinfo", query)
		if fetchErr != nil {
			return nil, fetchErr
		}
		body, marshalErr := json.Marshal(out)
		if marshalErr != nil {
			return nil, marshalErr
		}
		cache.put(key, &cachedResponse{status: http.StatusOK, contentType: "application/json", body: body}, int64(len(body)))
		return body, nil
	})
	if err != nil {
		return nil, err
	}
	body, ok := value.([]byte)
	if !ok {
		return nil, errors.New("image service viewer-info flight returned an invalid result")
	}
	var out map[string]any
	if err := json.Unmarshal(body, &out); err != nil {
		return nil, err
	}
	return out, nil
}

func sourceViewerAxes(info map[string]any) (t, c, z int, ok bool) {
	axes, ok := info["axis_sizes"].(map[string]any)
	if !ok {
		return 0, 0, 0, false
	}
	t, tOK := jsonInt(axes["T"])
	c, cOK := jsonInt(axes["C"])
	z, zOK := jsonInt(axes["Z"])
	if !tOK || !cOK || !zOK || t < 1 || c < 1 || z < 1 {
		return 0, 0, 0, false
	}
	return t, c, z, true
}

var errMalformedImageViewerAxes = errors.New("image service returned malformed source axes")

// sourceImageServiceViewerInfo obtains T/C/Z from the original upload. A
// display pyramid is an acceleration artifact and must never become the
// authority for semantic channel or time selection.
func (deps ServerDeps) sourceImageServiceViewerInfo(
	ctx context.Context,
	sourcePath string,
) (info map[string]any, t, c, z int, err error) {
	info, err = deps.cachedImageServiceViewerInfo(ctx, sourcePath)
	if err != nil {
		return nil, 0, 0, 0, err
	}
	t, c, z, ok := sourceViewerAxes(info)
	if !ok {
		return nil, 0, 0, 0, errMalformedImageViewerAxes
	}
	return info, t, c, z, nil
}

// committedThumbnailDerivative performs the source-bound manifest admission needed
// to advertise or serve a thumbnail without doing a sidecar metadata read.
func committedThumbnailDerivative(
	root string,
	record resourceRecord,
	sourcePath string,
) (string, bool) {
	manifest, artifactPath, admitted := readDerivativeManifest(root, record, sourcePath)
	if !admitted || !manifest.Capabilities.Thumbnail {
		return "", false
	}
	if prefersBioioReader(record.OriginalName) && manifest.Producer.Reader != "bioio" {
		return "", false
	}
	return artifactPath, true
}

// compatibleDerivedPyramid returns a derived artifact only when a strict, source-
// bound manifest commits it; both source and derivative metadata match the exact
// manifest semantics; and the manifest declares the requested serving capability.
// Derived pyramids are accelerators, never semantic authority.
func (deps ServerDeps) compatibleDerivedPyramid(
	ctx context.Context,
	root string,
	record resourceRecord,
	sourcePath string,
	sourceInfo map[string]any,
	use derivativeUse,
) (string, map[string]any, bool) {
	manifest, derivedPath, admitted := readDerivativeManifest(root, record, sourcePath)
	if !admitted {
		return "", nil, false
	}
	preferredBioio := prefersBioioReader(record.OriginalName)
	if preferredBioio && manifest.Producer.Reader != "bioio" {
		return "", nil, false
	}
	repairIncompatible := func() {
		deps.enqueuePyramidDerivation(ctx, root, record, sourcePath, "repair-incompatible")
	}
	if !manifest.Capabilities.supports(use) {
		// A valid derivative may intentionally lack route-specific selectors (for
		// example T-aware tiles). That is an intrinsic delivery limitation, not
		// corruption; use the source fallback without scheduling futile repair.
		return "", nil, false
	}
	sourceStat, sourceErr := os.Stat(sourcePath)
	derivedStat, derivedErr := os.Stat(derivedPath)
	if sourceErr != nil || derivedErr != nil {
		return "", nil, false
	}
	if sourceInfo == nil && !preferredBioio {
		var err error
		sourceInfo, err = deps.cachedImageServiceViewerInfoVia(ctx, sourcePath, deps.pyramidInfoCache)
		if err != nil {
			return "", nil, false
		}
	}
	if !preferredBioio && !derivativeSemanticsMatch(sourceInfo, manifest.Semantics) {
		repairIncompatible()
		return "", nil, false
	}
	derivedInfo, err := deps.cachedImageServiceViewerInfoVia(ctx, derivedPath, deps.pyramidInfoCache)
	if err != nil {
		return "", nil, false
	}
	if !derivativeArtifactSemanticsMatch(derivedInfo, manifest.Semantics) {
		repairIncompatible()
		return "", nil, false
	}
	if manifest.Capabilities != derivativeCapabilitiesForViewer(derivedInfo, manifest.Semantics) {
		repairIncompatible()
		return "", nil, false
	}
	sourceAfter, sourceAfterErr := os.Stat(sourcePath)
	derivedAfter, derivedAfterErr := os.Stat(derivedPath)
	if sourceAfterErr != nil || derivedAfterErr != nil ||
		sourceAfter.Size() != sourceStat.Size() ||
		!sourceAfter.ModTime().Equal(sourceStat.ModTime()) ||
		derivedAfter.Size() != derivedStat.Size() ||
		!derivedAfter.ModTime().Equal(derivedStat.ModTime()) {
		return "", nil, false
	}
	return derivedPath, derivedInfo, true
}

// preferredBioioDerivedViewerInfo admits a preferred-reader derivative without
// consulting libbioimage for the proprietary source. The strict Go manifest is
// the source/scene semantic authority; the derivative viewer-info contributes
// only delivery fields after its pixel semantics and capabilities are verified.
func (deps ServerDeps) preferredBioioDerivedViewerInfo(
	ctx context.Context,
	root string,
	record resourceRecord,
	sourcePath string,
) (map[string]any, bool) {
	manifest, derivedPath, admitted := readDerivativeManifest(root, record, sourcePath)
	if !admitted || manifest.Producer.Reader != "bioio" || !manifest.Capabilities.Thumbnail {
		return nil, false
	}
	derivedInfo, err := deps.cachedImageServiceViewerInfoVia(ctx, derivedPath, deps.pyramidInfoCache)
	if err != nil || !derivativeArtifactSemanticsMatch(derivedInfo, manifest.Semantics) {
		return nil, false
	}
	if manifest.Capabilities != derivativeCapabilitiesForViewer(derivedInfo, manifest.Semantics) {
		return nil, false
	}
	return preferredReaderViewerInfo(manifest, derivedInfo), true
}

func preferredReaderViewerInfo(manifest derivativeManifest, delivery map[string]any) map[string]any {
	core := make(map[string]any)
	for _, key := range []string{
		"kind", "modality", "backend_mode", "decodable", "is_volume", "is_timeseries",
		"is_multichannel", "tile_scheme", "display_defaults", "selected_indices", "viewer",
	} {
		if value, present := delivery[key]; present {
			core[key] = value
		}
	}
	semantics := manifest.Semantics
	core["dims_order"] = semantics.DimsOrder
	core["dtype"] = semantics.DType
	core["axis_sizes"] = map[string]any{
		"T": semantics.AxisSizes.T,
		"C": semantics.AxisSizes.C,
		"Z": semantics.AxisSizes.Z,
		"Y": semantics.AxisSizes.Y,
		"X": semantics.AxisSizes.X,
	}
	channelNames := make([]string, len(semantics.Channels))
	for index, channel := range semantics.Channels {
		channelNames[index] = channel.Name
	}
	core["channel_names"] = channelNames
	core["physical_spacing"] = map[string]any{
		"x": semantics.Spacing.X.Value,
		"y": semantics.Spacing.Y.Value,
		"z": semantics.Spacing.Z.Value,
	}
	displayChannels := []int(nil)
	if phys, ok := jsonObject(delivery["phys"]); ok {
		displayChannels = preferredReaderDisplayChannels(
			phys["display_channels"],
			semantics.AxisSizes.C,
		)
	}
	if len(displayChannels) == 0 {
		if defaults, ok := jsonObject(delivery["display_defaults"]); ok {
			displayChannels = preferredReaderDisplayChannels(
				defaults["channels"],
				semantics.AxisSizes.C,
			)
		}
	}
	if len(displayChannels) == 0 {
		for channel := 0; channel < min(semantics.AxisSizes.C, 3); channel++ {
			displayChannels = append(displayChannels, channel)
		}
	}
	phys := map[string]any{
		"x":                semantics.AxisSizes.X,
		"y":                semantics.AxisSizes.Y,
		"z":                semantics.AxisSizes.Z,
		"t":                semantics.AxisSizes.T,
		"ch":               semantics.AxisSizes.C,
		"pixel_size":       []float64{semantics.Spacing.X.Value, semantics.Spacing.Y.Value, semantics.Spacing.Z.Value, 1},
		"pixel_units":      []string{semantics.Spacing.X.Unit, semantics.Spacing.Y.Unit, semantics.Spacing.Z.Unit, "frame"},
		"channel_names":    channelNames,
		"display_channels": displayChannels,
	}
	if pixelDepth, pixelFormat, ok := preferredReaderDType(semantics.DType); ok {
		phys["pixel_depth"] = pixelDepth
		phys["pixel_format"] = pixelFormat
	}
	core["phys"] = phys
	metadata := map[string]any{
		"reader":               manifest.Producer.Reader,
		"array_dtype":          semantics.DType,
		"scene_count":          semantics.Scene.Count,
		"selected_scene_index": semantics.Scene.Index,
		"spacing_units": map[string]any{
			"x": semantics.Spacing.X.Unit,
			"y": semantics.Spacing.Y.Unit,
			"z": semantics.Spacing.Z.Unit,
		},
	}
	if semantics.Scene.ID != nil {
		metadata["selected_scene_id"] = *semantics.Scene.ID
		core["selected_scene_id"] = *semantics.Scene.ID
	}
	core["scene_count"] = semantics.Scene.Count
	core["selected_scene_index"] = semantics.Scene.Index
	core["metadata"] = metadata
	return core
}

func preferredReaderDType(dtype string) (pixelDepth int, pixelFormat string, ok bool) {
	switch strings.ToLower(strings.TrimSpace(dtype)) {
	case "uint8":
		return 8, "u", true
	case "uint16":
		return 16, "u", true
	case "int16":
		return 16, "s", true
	case "float32":
		return 32, "f", true
	case "float64":
		return 64, "f", true
	default:
		return 0, "", false
	}
}

func preferredReaderDisplayChannels(value any, channelCount int) []int {
	var raw []any
	switch values := value.(type) {
	case []any:
		raw = values
	case []int:
		raw = make([]any, len(values))
		for index, channel := range values {
			raw[index] = channel
		}
	default:
		return nil
	}
	channels := make([]int, 0, len(raw))
	for _, item := range raw {
		channel, ok := jsonInt(item)
		if !ok || channel < 0 || channel >= channelCount || slices.Contains(channels, channel) {
			return nil
		}
		channels = append(channels, channel)
	}
	return channels
}

func writeImageSourceAuthorityError(w http.ResponseWriter, err error) {
	status := http.StatusUnprocessableEntity
	var serviceErr *imageServiceStatusError
	if errors.As(err, &serviceErr) {
		status = serviceErr.status
	} else if !errors.Is(err, errMalformedImageViewerAxes) {
		status = http.StatusBadGateway
	}
	writeError(w, status, errors.New("authoritative source image metadata is unavailable"))
}

// handleGetUploadViewerService backs /viewer with libbioimage metadata. The
// source drives axis sizes/channels/spacing (so z-scrub planes stay correct);
// when the source is not natively pyramidal, a derived pyramid's tile scheme is
// folded in so the viewer can use the bounded DeepZoom path.
func (deps ServerDeps) handleGetUploadViewerService(w http.ResponseWriter, r *http.Request) {
	if !deps.imageServiceConfigured() {
		deps.handleGetUploadViewer(w, r)
		return
	}
	root, record, path, ok := deps.resolveUploadServingRequest(w, r)
	if !ok {
		return
	}
	// NIfTI keeps the dedicated, volume-specialized medical viewer.
	if isNiftiUpload(record.OriginalName, record.ContentType) {
		deps.writeNiftiUploadViewer(w, root, record, path)
		return
	}
	// 3D scenes (Gaussian splats / point clouds) are served from a derived, chunked
	// stream and must be routed BEFORE the libbioimage probe below: a .ply has no
	// pixel geometry, so the engine can only 415 it and the undecodable path would
	// then enqueue a pointless imgcnv transcode of a multi-gigabyte scene. See scene3d.go.
	if info, isScene := scene3dPeek(record, path); isScene {
		deps.enqueueScene3dDerivation(r.Context(), root, record, path, "view")
		deps.writeScene3dViewer(w, record, info, path)
		return
	}
	// OME-Zarr (and other ngff-served special formats) is served natively by the
	// ngff-service — its viewer-info comes from the zarr store, and the Lens viewer
	// consumes it identically to a libbioimage image. Routed via an ngffDeps copy so the
	// edge tile cache + backpressure + graceful fallback all apply unchanged.
	if deps.servesViaNgff(record, path) {
		ngff := deps.ngffDeps()
		core, err := ngff.cachedImageServiceViewerInfo(r.Context(), path)
		if err != nil {
			if imageServiceUndecodable(err) {
				deps.writeUnsupportedFormatViewer(w, record)
				return
			}
			writeError(w, http.StatusBadGateway, fmt.Errorf("ngff viewer-info: %w", err))
			return
		}
		injectControlPlaneViewerFields(core, record)
		writeJSON(w, http.StatusOK, core)
		return
	}
	if deps.ngffServiceUnavailable(record, path) {
		writeError(w, http.StatusServiceUnavailable, errNgffServiceNotConfigured)
		return
	}
	if prefersBioioReader(record.OriginalName) {
		if preferred, admitted := deps.preferredBioioDerivedViewerInfo(r.Context(), root, record, path); admitted {
			injectControlPlaneViewerFields(preferred, record)
			writeJSON(w, http.StatusOK, preferred)
			return
		}
		deps.enqueuePyramidDerivation(r.Context(), root, record, path, "prefer-bioio")
		deps.writeUnsupportedFormatViewer(w, record)
		return
	}
	core, err := deps.cachedImageServiceViewerInfo(r.Context(), path)
	if err != nil {
		// The engine recognized the file but cannot decode it (415/422): a permanent,
		// format-level failure (e.g. a Leica .lif — registered but non-functional in
		// this libbioimage build). The legacy native Go viewer supports only a small
		// raster subset, so probe whether it can produce a real plane; if not, surface
		// an explicit "unsupported" descriptor so the viewer shows a clear message +
		// download instead of a broken 1x1 canvas with an endless spinner.
		if imageServiceUndecodable(err) {
			// Without authoritative source axes, an existing derivative cannot prove
			// that it preserved C/T/Z shape. The legacy native Go viewer reads only a
			// small raster subset; if it can produce a real plane, use it.
			if info := uploadImageDescriptorForPath(path, record.ContentType); info.Width >= 2 && info.Height >= 2 {
				deps.handleGetUploadViewer(w, r)
				return
			}
			// Kick off a bioio transcode->pyramid conversion so a later open renders it
			// (bypassing the extension allowlist — the engine already recognized but
			// couldn't decode this image, e.g. a series-suffixed ".lif_15" name), and
			// meanwhile show a calm "preview unavailable" card.
			deps.enqueuePyramidDerivation(r.Context(), root, record, path, "transcode")
			deps.writeUnsupportedFormatViewer(w, record)
			return
		}
		// Transport error / sidecar outage: graceful fallback to the legacy native viewer.
		deps.handleGetUploadViewer(w, r)
		return
	}
	derivedPath, derivedInfo, derivedCompatible := deps.compatibleDerivedPyramid(
		r.Context(), root, record, path, core, derivativeUse{capability: "thumbnail"},
	)
	// A slice_stack volume (microscopy z-stack) derives to an OME-BigTIFF whose
	// embedded -tile reader is broken (the OME wrapper); it serves 3D via /atlas
	// and 2D via /slice, so it must NOT advertise the derived pyramid's tile_scheme
	// (that would route the viewer to the failing deferred-multiscale tile path).
	if !viewerIsSliceStackVolume(core) {
		if derivedCompatible {
			// The derived pyramid serves the tile PIXELS (resolveUploadTilePathForImageService),
			// so the viewer must use the PYRAMID's tile_scheme — its tile size and level grid —
			// even when the source advertised its own. Otherwise the viewer fetches at the
			// source geometry (e.g. 256-px / 8 levels) while pixels come from the pyramid
			// (512-px / 11 levels): every pyramid tile is decoded 4x and the engine is
			// needlessly overloaded on deep zoom. Overriding here aligns grid with data.
			mergePyramidTileScheme(core, derivedInfo)
		} else if core["tile_scheme"] == nil {
			// No derived pyramid and the source is not directly tile-servable: kick off
			// derivation so a later open gets the bounded DeepZoom path; the direct/slice
			// path still works meanwhile.
			if derivedPath == "" {
				deps.ensurePyramidDerivation(r.Context(), root, record, path, "view")
			}
		}
	}
	injectControlPlaneViewerFields(core, record)
	writeJSON(w, http.StatusOK, core)
}

// writeUnsupportedFormatViewer emits an explicit "this engine build cannot decode
// this format" viewer descriptor. It returns HTTP 200 (not an error) so the frontend
// receives structured fields — kind:"unsupported", decodable:false, the format, and a
// download URL — and can render a calm "preview unavailable, download instead" card
// rather than a broken 1x1 canvas stuck on "Loading…". axis_sizes is kept (zeroed) so
// existing viewer-info parsers that read it unconditionally don't choke.
func (deps ServerDeps) writeUnsupportedFormatViewer(w http.ResponseWriter, record resourceRecord) {
	fileIDSegment := url.PathEscape(record.FileID)
	ext := strings.TrimPrefix(strings.ToLower(filepath.Ext(record.OriginalName)), ".")
	label := strings.ToUpper(ext)
	if label == "" {
		label = "this format"
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"kind":             "unsupported",
		"decodable":        false,
		"file_id":          record.FileID,
		"original_name":    record.OriginalName,
		"format":           ext,
		"modality":         "image",
		"backend_mode":     "none",
		"dims_order":       "YX",
		"axis_sizes":       map[string]int{"T": 1, "C": 1, "Z": 1, "Y": 0, "X": 0},
		"selected_indices": map[string]int{"T": 0, "C": 0, "Z": 0},
		"is_volume":        false,
		"is_timeseries":    false,
		"is_multichannel":  false,
		"service_urls": map[string]any{
			"download": "/v2/resources/" + fileIDSegment + "/download",
		},
		"message": fmt.Sprintf("%s files can't be previewed by the image engine yet. Download the original to open it in a desktop tool.", label),
	})
}

// handleServeUploadSliceService backs /slice with a real libbioimage plane read
// (honoring z/t/level), which is what makes z-scrub scrub actual planes. NIfTI
// and the unconfigured case keep the legacy behavior.
func (deps ServerDeps) handleServeUploadSliceService(w http.ResponseWriter, r *http.Request) {
	authorization, ok := deps.authorizeUploadServingRequest(w, r)
	if !ok {
		return
	}
	record := authorization.record
	// NIfTI owns an exact native MPR contract and must be routed before the
	// generic Z-only parser. Syntax and catalog-authoritative bounds are checked
	// before resolving/statting the source path.
	if isNiftiUpload(record.OriginalName, record.ContentType) {
		selection, selectionErr := parseNiftiMPRSelection(r.URL.Query())
		if selectionErr != nil {
			writeError(w, http.StatusUnprocessableEntity, selectionErr)
			return
		}
		maskRequest, maskErr := parseMaskSliceRequest(r)
		if maskErr != nil {
			writeError(w, http.StatusUnprocessableEntity, maskErr)
			return
		}
		if maskRequest.enabled {
			writeError(w, http.StatusUnprocessableEntity, errors.New("mask slices are unsupported for NIfTI sources"))
			return
		}
		catalogBounds, boundsErr := validateCatalogNiftiMPRBounds(record, selection)
		if boundsErr != nil {
			writeError(w, http.StatusUnprocessableEntity, boundsErr)
			return
		}
		path, resolved := resolveAuthorizedUploadStorage(w, authorization)
		if !resolved {
			return
		}
		if !catalogBounds {
			geometry, geometryErr := readNiftiHeaderGeometry(path)
			if geometryErr != nil {
				writeError(w, http.StatusUnsupportedMediaType, geometryErr)
				return
			}
			if boundsErr := validateNiftiMPRBounds(
				selection,
				geometry.width,
				geometry.height,
				geometry.depth,
				geometry.timeCount,
				geometry.channelCount,
			); boundsErr != nil {
				writeError(w, http.StatusUnprocessableEntity, boundsErr)
				return
			}
		}
		if err := serveNiftiSliceAsPNG(w, path, r, niftiDecompressedSidecarIdentity{
			root: authorization.root, resourceID: record.FileID, sourceSHA256: record.SHA256,
		}); err != nil {
			writeError(w, http.StatusUnsupportedMediaType, err)
		}
		return
	}
	selectors, selectorErr := parseScientificImageSelectors(r.URL.Query(), record)
	if selectorErr != nil {
		writeError(w, http.StatusUnprocessableEntity, selectorErr)
		return
	}
	sliceOptions, sliceOptionsErr := parseImageSliceOptions(r.URL.Query())
	if sliceOptionsErr != nil {
		writeError(w, http.StatusUnprocessableEntity, sliceOptionsErr)
		return
	}
	maskRequest, maskErr := parseMaskSliceRequest(r)
	if maskErr != nil {
		writeError(w, http.StatusUnprocessableEntity, maskErr)
		return
	}
	path, ok := resolveAuthorizedUploadStorage(w, authorization)
	if !ok {
		return
	}
	root := authorization.root
	if !deps.imageServiceConfigured() && strings.TrimSpace(deps.NgffServiceURL) == "" {
		if maskRequest.enabled {
			deps.handleNotConfigured("mask slices require the configured source image service")(w, r)
			return
		}
		if !legacyDisplayPhotoSliceAllowed(record, selectors, sliceOptions) {
			deps.handleNotConfigured("scientific slices require the configured image service")(w, r)
			return
		}
		deps.handleServeUpload(w, r)
		return
	}
	// OME-Zarr is rendered natively by the ngff-service from the store (bundle dir path).
	if deps.servesViaNgff(record, path) {
		if maskRequest.enabled {
			writeError(w, http.StatusUnprocessableEntity, errors.New("mask slices are unsupported for NGFF sources"))
			return
		}
		q := url.Values{"path": {path}}
		for key, values := range sliceOptions {
			q.Set(key, values[0])
		}
		selectors.apply(q)
		copyQueryValueIfPresent(q, r.URL.Query(), "cache_key")
		deps.ngffDeps().proxyImageServiceCached(w, r, "/slice", q)
		return
	}
	if deps.ngffServiceUnavailable(record, path) {
		writeError(w, http.StatusServiceUnavailable, errNgffServiceNotConfigured)
		return
	}
	if !deps.imageServiceConfigured() {
		if maskRequest.enabled {
			deps.handleNotConfigured("mask slices require the configured source image service")(w, r)
			return
		}
		if !legacyDisplayPhotoSliceAllowed(record, selectors, sliceOptions) {
			deps.handleNotConfigured("scientific slices require the configured image service")(w, r)
			return
		}
		deps.handleServeUpload(w, r)
		return
	}
	maskMode := maskRequest.enabled
	if maskMode {
		sourceInfo, timeCount, channelCount, _, authorityErr := deps.sourceImageServiceViewerInfo(
			r.Context(),
			path,
		)
		if authorityErr != nil {
			writeImageSourceAuthorityError(w, authorityErr)
			return
		}
		sanitizeScalarMaskCapability(sourceInfo, record)
		capability, capable := jsonObject(sourceInfo["scalar_mask_capability"])
		if !capable {
			writeError(w, http.StatusUnprocessableEntity, errors.New("mask slices are unsupported for this source"))
			return
		}
		if maskRequest.channel >= channelCount || maskRequest.time >= timeCount {
			writeError(w, http.StatusBadRequest, errors.New("mask slice channel/time selection is out of range"))
			return
		}
		threshold, parseErr := strconv.ParseFloat(maskRequest.thresholdRaw, 64)
		dtype, _ := capability["dtype"].(string)
		canonical, canonicalOK := canonicalIntegerMaskThreshold(threshold, dtype)
		if parseErr != nil || !canonicalOK {
			writeError(w, http.StatusBadRequest, errors.New("mask slice threshold is invalid for the source dtype"))
			return
		}
		maskRequest.thresholdRaw = strconv.FormatFloat(canonical, 'f', -1, 64)
	}
	// Prefer the derived tiled pyramid when one exists: its native level 0 is
	// pixel-identical to the source but a bounded (tiled) read, so a z-scrub plane
	// decodes ~8x faster than re-decoding a full plane from a non-pyramidal source
	// (1.9s -> 0.23s on the 575MB OME-TIFF). -slice works on the derived OME-BigTIFF
	// even though its -tile reader does not (atlas/thumbnail read it the same way).
	servePath := path
	if !maskMode {
		if dp, _, compatible := deps.compatibleDerivedPyramid(
			r.Context(), root, record, path, nil, derivativeUse{
				capability:      "slice",
				requireT:        selectors.tPresent,
				requireZ:        selectors.zPresent,
				requireChannels: selectors.channelsPresent,
				requireLUT:      selectors.colorsPresent,
			},
		); compatible {
			servePath = dp
		}
	}
	// channels/colors enable additive multi-channel LUT compositing for fluorescence
	// microscopy (libbioimage fuses the selected channels). full_resolution=false serves
	// a bounded pyramid level for fast scrub frames; true reads the native plane.
	buildSliceQuery := func(p string) url.Values {
		q := url.Values{"path": {p}}
		for key, values := range sliceOptions {
			q.Set(key, values[0])
		}
		copyQueryValueIfPresent(q, r.URL.Query(), "cache_key")
		if maskRequest.enabled {
			q.Set("channels", strconv.Itoa(maskRequest.channel))
			q.Set("t", strconv.Itoa(maskRequest.time))
			q.Set("scalar_render_mode", "mask")
			q.Set("scalar_threshold_value", maskRequest.thresholdRaw)
			q.Set("scalar_threshold_foreground", "above")
		} else {
			selectors.apply(q)
		}
		return q
	}
	// A failed derived scientific read may retry the authoritative source through the
	// same selector-aware /slice endpoint. It must never degrade to legacy /display,
	// which can flatten T/Z/C and silently return the wrong plane.
	var fallback http.HandlerFunc
	if servePath != path {
		fallback = func(w http.ResponseWriter, r *http.Request) {
			deps.proxyImageServiceSliceCached(w, r, "/slice", buildSliceQuery(path))
		}
	}
	// Route slices through the dedicated slice cache so a z-scrub burst can't evict
	// the DeepZoom viewer's tile/atlas working set from the main image cache.
	deps.proxyImageServiceSliceCached(w, r, "/slice", buildSliceQuery(servePath), fallback)
}

func legacyDisplayPhotoSliceAllowed(
	record resourceRecord,
	selectors scientificImageSelectors,
	sliceOptions url.Values,
) bool {
	if selectors.present() || len(sliceOptions) > 0 || !goNativeThumbnailable(record) {
		return false
	}
	if timeCount, _, depth, authoritative := catalogImageSelectorLimits(record); authoritative {
		return timeCount == 1 && depth == 1
	}
	// Common still-image decoders are intrinsically two-dimensional. Animated GIF
	// is intentionally excluded because serving its first frame is not an exact
	// time-series slice contract.
	return !strings.EqualFold(filepath.Ext(record.OriginalName), ".gif") &&
		!strings.EqualFold(strings.TrimSpace(record.ContentType), "image/gif")
}

type maskSliceRequest struct {
	enabled      bool
	thresholdRaw string
	channel      int
	time         int
}

func parseMaskSliceRequest(r *http.Request) (maskSliceRequest, error) {
	if r == nil {
		return maskSliceRequest{}, nil
	}
	query := r.URL.Query()
	mode, modePresent, err := exactRawQueryValue(
		query,
		[]string{"scalar_render_mode"},
		"scalar render mode",
	)
	if err != nil {
		return maskSliceRequest{}, err
	}
	thresholdRaw, thresholdPresent, thresholdErr := exactRawQueryValue(
		query,
		[]string{"scalar_threshold_value"},
		"scalar threshold value",
	)
	if thresholdErr != nil {
		return maskSliceRequest{}, thresholdErr
	}
	foreground, foregroundPresent, foregroundErr := exactRawQueryValue(
		query,
		[]string{"scalar_threshold_foreground"},
		"scalar threshold foreground",
	)
	if foregroundErr != nil {
		return maskSliceRequest{}, foregroundErr
	}
	if !modePresent {
		if thresholdPresent || foregroundPresent {
			return maskSliceRequest{}, errors.New("mask threshold selectors require scalar_render_mode=mask")
		}
		return maskSliceRequest{}, nil
	}
	if mode == "intensity" {
		if thresholdPresent || foregroundPresent {
			return maskSliceRequest{}, errors.New("intensity slices must not include mask threshold selectors")
		}
		return maskSliceRequest{}, nil
	}
	if mode != "mask" {
		return maskSliceRequest{}, errors.New("scalar_render_mode must be intensity or mask")
	}
	if !thresholdPresent || !foregroundPresent {
		return maskSliceRequest{}, errors.New("mask slices require one raw threshold and foreground selector")
	}
	threshold, err := strconv.ParseFloat(thresholdRaw, 64)
	if err != nil || math.IsNaN(threshold) || math.IsInf(threshold, 0) {
		return maskSliceRequest{}, errors.New("mask slice requires a finite raw threshold")
	}
	if foreground != "above" {
		return maskSliceRequest{}, errors.New("mask slice foreground must be above")
	}
	channelRaw, channelPresent, channelErr := exactRawQueryValue(
		query,
		[]string{"channel", "c", "channels"},
		"mask slice channel selector",
	)
	if channelErr != nil {
		return maskSliceRequest{}, channelErr
	}
	channel := 0
	if channelPresent {
		if strings.Contains(channelRaw, ",") {
			return maskSliceRequest{}, errors.New("mask slice requires exactly one channel")
		}
		channel, err = parseExactNonNegativeDecimal(channelRaw, "mask slice channel index")
		if err != nil {
			return maskSliceRequest{}, err
		}
	}
	timeRaw, timePresent, timeErr := exactRawQueryValue(
		query,
		[]string{"t", "time", "timepoint"},
		"mask slice time selector",
	)
	if timeErr != nil {
		return maskSliceRequest{}, timeErr
	}
	timeIndex := 0
	if timePresent {
		timeIndex, err = parseExactNonNegativeDecimal(timeRaw, "mask slice time index")
		if err != nil {
			return maskSliceRequest{}, err
		}
	}
	return maskSliceRequest{
		enabled:      true,
		thresholdRaw: thresholdRaw,
		channel:      channel,
		time:         timeIndex,
	}, nil
}

// handleGetUploadScalarVolumeService backs /scalar-volume for non-NIfTI volumes
// (microscopy z-stacks) by streaming the image service's raw scalar grid plus
// its x-volume-* headers. NIfTI keeps its native loader.
func (deps ServerDeps) handleGetUploadScalarVolumeService(w http.ResponseWriter, r *http.Request) {
	if !deps.imageServiceConfigured() {
		deps.handleGetUploadScalarVolume(w, r)
		return
	}
	authorization, ok := deps.authorizeUploadServingRequest(w, r)
	if !ok {
		return
	}
	record := authorization.record
	if isNiftiUpload(record.OriginalName, record.ContentType) {
		deps.serveAuthorizedNiftiScalarVolume(w, r, authorization)
		return
	}
	path, ok := resolveAuthorizedUploadStorage(w, authorization)
	if !ok {
		return
	}
	channelIndex, err := parseExactScalarIndex(r, []string{"channel", "c"})
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	timeIndex, err := parseExactScalarIndex(r, []string{"t", "time", "timepoint"})
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	_, timeCount, channelCount, _, authorityErr := deps.sourceImageServiceViewerInfo(
		r.Context(),
		path,
	)
	if authorityErr != nil {
		writeImageSourceAuthorityError(w, authorityErr)
		return
	}
	if channelIndex >= channelCount {
		writeError(
			w,
			http.StatusBadRequest,
			fmt.Errorf(
				"scalar volume channel index %d is out of range for C=%d",
				channelIndex,
				channelCount,
			),
		)
		return
	}
	if timeIndex >= timeCount {
		writeError(
			w,
			http.StatusBadRequest,
			fmt.Errorf(
				"scalar volume time index %d is out of range for T=%d",
				timeIndex,
				timeCount,
			),
		)
		return
	}
	query := url.Values{
		"path":    {path},
		"channel": {strconv.Itoa(channelIndex)},
		"t":       {strconv.Itoa(timeIndex)},
	}
	sampling, _, samplingErr := exactRawQueryValue(
		r.URL.Query(),
		[]string{"sampling"},
		"scalar volume sampling",
	)
	if samplingErr != nil {
		writeError(w, http.StatusBadRequest, samplingErr)
		return
	}
	if sampling == "" {
		sampling = "box"
	} else if sampling != "box" && sampling != "nearest" {
		writeError(w, http.StatusBadRequest, errors.New("scalar volume sampling must be box or nearest"))
		return
	}
	query.Set("sampling", sampling)
	deps.proxyImageService(w, r, "/scalar-volume", query, deps.handleGetUploadScalarVolume)
}

type niftiScalarVolumeSelection struct {
	time     int
	channel  int
	sampling string
}

func parseNiftiScalarVolumeSelection(query url.Values) (niftiScalarVolumeSelection, error) {
	allowed := map[string]bool{
		"t": true, "time": true, "timepoint": true,
		"channel": true, "c": true, "sampling": true,
	}
	for key := range query {
		if !allowed[key] {
			return niftiScalarVolumeSelection{}, fmt.Errorf("unsupported NIfTI scalar volume selector %q", key)
		}
	}
	selection := niftiScalarVolumeSelection{sampling: "box"}
	timeRaw, timePresent, err := exactRawQueryValue(
		query,
		[]string{"t", "time", "timepoint"},
		"NIfTI scalar volume time selector",
	)
	if err != nil {
		return niftiScalarVolumeSelection{}, err
	}
	if timePresent {
		selection.time, err = parseExactNonNegativeDecimal(
			timeRaw,
			"NIfTI scalar volume time selector",
		)
		if err != nil {
			return niftiScalarVolumeSelection{}, err
		}
	}
	channelRaw, channelPresent, err := exactRawQueryValue(
		query,
		[]string{"channel", "c"},
		"NIfTI scalar volume channel selector",
	)
	if err != nil {
		return niftiScalarVolumeSelection{}, err
	}
	if channelPresent {
		selection.channel, err = parseExactNonNegativeDecimal(
			channelRaw,
			"NIfTI scalar volume channel selector",
		)
		if err != nil {
			return niftiScalarVolumeSelection{}, err
		}
	}
	sampling, samplingPresent, err := exactRawQueryValue(
		query,
		[]string{"sampling"},
		"NIfTI scalar volume sampling",
	)
	if err != nil {
		return niftiScalarVolumeSelection{}, err
	}
	if samplingPresent {
		if sampling != "box" && sampling != "nearest" {
			return niftiScalarVolumeSelection{}, errors.New("NIfTI scalar volume sampling must be box or nearest")
		}
		selection.sampling = sampling
	}
	return selection, nil
}

func validateCatalogNiftiScalarVolumeBounds(
	record resourceRecord,
	selection niftiScalarVolumeSelection,
) (bool, error) {
	timeCount, channelCount, _, authoritative := catalogImageSelectorLimits(record)
	if !authoritative {
		return false, nil
	}
	if selection.time >= timeCount {
		return true, errors.New("NIfTI scalar volume time selector is out of range")
	}
	if selection.channel >= channelCount {
		return true, errors.New("NIfTI scalar volume channel selector is out of range")
	}
	return true, nil
}

func exactRawQueryValue(
	query url.Values,
	aliases []string,
	label string,
) (raw string, present bool, err error) {
	for _, alias := range aliases {
		values, exists := query[alias]
		if !exists {
			continue
		}
		if present || len(values) != 1 {
			return "", false, fmt.Errorf("%s must be supplied exactly once", label)
		}
		present = true
		raw = strings.TrimSpace(values[0])
		if raw == "" {
			return "", false, fmt.Errorf("%s must not be empty", label)
		}
	}
	return raw, present, nil
}

func parseExactNonNegativeDecimal(raw, label string) (int, error) {
	if raw == "" {
		return 0, fmt.Errorf("%s must be a non-negative integer", label)
	}
	for _, char := range raw {
		if char < '0' || char > '9' {
			return 0, fmt.Errorf("%s must be a non-negative integer", label)
		}
	}
	value, err := strconv.ParseUint(raw, 10, 64)
	maxInt := uint64(^uint(0) >> 1)
	if err != nil || value > maxInt {
		return 0, fmt.Errorf("%s must be a non-negative integer", label)
	}
	return int(value), nil
}

func parseExactScalarIndex(r *http.Request, aliases []string) (int, error) {
	if r == nil {
		return 0, nil
	}
	label := "scalar volume " + strings.Join(aliases, "/") + " index"
	raw, present, err := exactRawQueryValue(r.URL.Query(), aliases, label)
	if err != nil || !present {
		return 0, err
	}
	return parseExactNonNegativeDecimal(raw, label)
}

// isVideoUpload reports whether a resource is a video (rendered client-side with
// a <video> poster, not a server-side still).
func isVideoUpload(originalName string, contentType string) bool {
	if strings.HasPrefix(strings.ToLower(strings.TrimSpace(contentType)), "video/") {
		return true
	}
	switch strings.ToLower(filepath.Ext(originalName)) {
	case ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".ogv":
		return true
	}
	return false
}

// goNativeThumbnailable reports whether the native Go image decoder can render a
// thumbnail for this resource (the common web formats). Scientific containers —
// BigTIFF, OME-TIFF, CZI, ND2, LSM, … — are NOT in Go's stdlib decoders, so they
// route to the libbioimage thumbnail instead (this is what fixed the broken
// thumbnails for those formats while they still rendered in the viewer).
func goNativeThumbnailable(record resourceRecord) bool {
	switch strings.ToLower(strings.TrimSpace(record.ContentType)) {
	case "image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp", "image/bmp":
		return true
	}
	switch strings.ToLower(filepath.Ext(record.OriginalName)) {
	case ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp":
		return true
	}
	return false
}

func writeThumbnailPNG(w http.ResponseWriter, body []byte) error {
	if err := validateThumbnailPNG("image/png", body); err != nil {
		return err
	}
	w.Header().Set("Content-Type", "image/png")
	w.Header().Set("Cache-Control", "private, max-age=3600")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(body)
	return nil
}

func boundedRasterThumbnailConfig(file *os.File) (image.Config, string, error) {
	info, err := file.Stat()
	if err != nil {
		return image.Config{}, "", err
	}
	if !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > rasterThumbnailMaxInput {
		return image.Config{}, "", errors.New("raster thumbnail source exceeds the 32 MiB input limit")
	}
	width, height, format, err := boundedRasterThumbnailDimensions(file, info.Size())
	if err != nil {
		return image.Config{}, "", fmt.Errorf("raster thumbnail header could not be validated: %w", err)
	}
	if width <= 0 || height <= 0 || width > rasterThumbnailMaxAxis || height > rasterThumbnailMaxAxis {
		return image.Config{}, "", errors.New("raster thumbnail dimensions are invalid")
	}
	pixels, ok := checkedNonNegativeProduct(int64(width), int64(height))
	if !ok || pixels > rasterThumbnailMaxPixels {
		return image.Config{}, "", errors.New("raster thumbnail exceeds the 32 megapixel limit")
	}
	return image.Config{Width: width, Height: height}, format, nil
}

func boundedRasterThumbnailPreflight(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer func() { _ = file.Close() }()
	_, _, err = boundedRasterThumbnailConfig(file)
	return err
}

func boundedRasterThumbnailDimensions(file *os.File, size int64) (int, int, string, error) {
	prefixSize := int(min(size, int64(12)))
	prefix := make([]byte, prefixSize)
	if _, err := file.ReadAt(prefix, 0); err != nil && !errors.Is(err, io.EOF) {
		return 0, 0, "", err
	}
	switch {
	case len(prefix) >= 8 && bytes.Equal(prefix[:8], []byte("\x89PNG\r\n\x1a\n")):
		width, height, err := boundedPNGThumbnailDimensions(file, size)
		return width, height, "png", err
	case len(prefix) >= 2 && prefix[0] == 0xff && prefix[1] == 0xd8:
		width, height, err := boundedJPEGThumbnailDimensions(file, size)
		return width, height, "jpeg", err
	case len(prefix) >= 6 && (string(prefix[:6]) == "GIF87a" || string(prefix[:6]) == "GIF89a"):
		width, height, err := boundedGIFThumbnailDimensions(file, size)
		return width, height, "gif", err
	case len(prefix) >= 2 && string(prefix[:2]) == "BM":
		width, height, err := boundedBMPThumbnailDimensions(file, size)
		return width, height, "bmp", err
	case len(prefix) >= 12 && string(prefix[:4]) == "RIFF" && string(prefix[8:12]) == "WEBP":
		width, height, err := boundedWebPThumbnailDimensions(file, size)
		return width, height, "webp", err
	default:
		return 0, 0, "", errors.New("unsupported raster thumbnail format")
	}
}

func readRasterHeaderAt(file *os.File, offset int64, body []byte) error {
	if offset < 0 || int64(len(body)) > math.MaxInt64-offset {
		return errors.New("raster thumbnail header offset overflows")
	}
	if _, err := file.ReadAt(body, offset); err != nil {
		if errors.Is(err, io.EOF) {
			return io.ErrUnexpectedEOF
		}
		return err
	}
	return nil
}

func boundedPNGThumbnailDimensions(file *os.File, size int64) (int, int, error) {
	const signatureSize = int64(8)
	offset := signatureSize
	headerBytes := signatureSize
	sawHeader := false
	sawImageData := false
	width, height := 0, 0
	for chunk := 0; chunk < rasterThumbnailMaxChunks; chunk++ {
		if offset > size-12 || headerBytes > rasterThumbnailHeaderCap-8 {
			return 0, 0, errors.New("PNG chunk header is truncated or exceeds the structural scan limit")
		}
		var header [8]byte
		if err := readRasterHeaderAt(file, offset, header[:]); err != nil {
			return 0, 0, err
		}
		headerBytes += 8
		length := int64(binary.BigEndian.Uint32(header[0:4]))
		if length > math.MaxInt64-offset-12 {
			return 0, 0, errors.New("PNG chunk length overflows")
		}
		next := offset + 12 + length
		if next > size {
			return 0, 0, errors.New("PNG chunk is truncated")
		}
		chunkType := string(header[4:8])
		switch chunkType {
		case "IHDR":
			if sawHeader || chunk != 0 || length != 13 || headerBytes > rasterThumbnailHeaderCap-17 {
				return 0, 0, errors.New("PNG IHDR is invalid")
			}
			var body [13]byte
			if err := readRasterHeaderAt(file, offset+8, body[:]); err != nil {
				return 0, 0, err
			}
			var checksum [4]byte
			if err := readRasterHeaderAt(file, offset+8+int64(len(body)), checksum[:]); err != nil {
				return 0, 0, err
			}
			hash := crc32.NewIEEE()
			_, _ = hash.Write(header[4:8])
			_, _ = hash.Write(body[:])
			if hash.Sum32() != binary.BigEndian.Uint32(checksum[:]) {
				return 0, 0, errors.New("PNG IHDR checksum is invalid")
			}
			headerBytes += 17
			width64 := int64(binary.BigEndian.Uint32(body[0:4]))
			height64 := int64(binary.BigEndian.Uint32(body[4:8]))
			if width64 <= 0 || height64 <= 0 || width64 > math.MaxInt32 || height64 > math.MaxInt32 || !validPNGHeaderFields(body) {
				return 0, 0, errors.New("PNG IHDR fields are invalid")
			}
			width, height = int(width64), int(height64)
			sawHeader = true
		case "IDAT":
			if !sawHeader {
				return 0, 0, errors.New("PNG image data precedes IHDR")
			}
			if length > 0 {
				sawImageData = true
			}
		case "IEND":
			if length != 0 || !sawHeader || !sawImageData || next != size {
				return 0, 0, errors.New("PNG IEND is invalid or the source is truncated")
			}
			var checksum [4]byte
			if err := readRasterHeaderAt(file, offset+8, checksum[:]); err != nil {
				return 0, 0, err
			}
			if crc32.ChecksumIEEE(header[4:8]) != binary.BigEndian.Uint32(checksum[:]) {
				return 0, 0, errors.New("PNG IEND checksum is invalid")
			}
			return width, height, nil
		}
		offset = next
	}
	return 0, 0, errors.New("PNG has too many chunks")
}

func validPNGHeaderFields(body [13]byte) bool {
	bitDepth := body[8]
	colorType := body[9]
	validDepth := false
	switch colorType {
	case 0:
		validDepth = bitDepth == 1 || bitDepth == 2 || bitDepth == 4 || bitDepth == 8 || bitDepth == 16
	case 2, 4, 6:
		validDepth = bitDepth == 8 || bitDepth == 16
	case 3:
		validDepth = bitDepth == 1 || bitDepth == 2 || bitDepth == 4 || bitDepth == 8
	}
	return validDepth && body[10] == 0 && body[11] == 0 && body[12] <= 1
}

func boundedJPEGThumbnailDimensions(file *os.File, size int64) (int, int, error) {
	offset := int64(2)
	width, height := 0, 0
	sawStartOfFrame := false
	for markerCount := 0; markerCount < rasterThumbnailMaxChunks; markerCount++ {
		if offset >= size || offset >= rasterThumbnailHeaderCap {
			return 0, 0, errors.New("JPEG header is truncated or exceeds the structural scan limit")
		}
		var markerPrefix [1]byte
		if err := readRasterHeaderAt(file, offset, markerPrefix[:]); err != nil {
			return 0, 0, err
		}
		offset++
		if markerPrefix[0] != 0xff {
			return 0, 0, errors.New("JPEG marker prefix is invalid")
		}
		marker := byte(0xff)
		for marker == 0xff {
			if offset >= size || offset >= rasterThumbnailHeaderCap {
				return 0, 0, errors.New("JPEG marker is truncated")
			}
			if err := readRasterHeaderAt(file, offset, markerPrefix[:]); err != nil {
				return 0, 0, err
			}
			marker = markerPrefix[0]
			offset++
		}
		if marker == 0x00 || marker == 0xd8 || marker == 0xd9 {
			return 0, 0, errors.New("JPEG marker sequence is invalid")
		}
		if marker == 0x01 || (marker >= 0xd0 && marker <= 0xd7) {
			continue
		}
		var lengthBytes [2]byte
		if err := readRasterHeaderAt(file, offset, lengthBytes[:]); err != nil {
			return 0, 0, err
		}
		segmentLength := int64(binary.BigEndian.Uint16(lengthBytes[:]))
		if segmentLength < 2 || segmentLength > math.MaxInt64-offset {
			return 0, 0, errors.New("JPEG segment length is invalid")
		}
		segmentEnd := offset + segmentLength
		if segmentEnd > size || segmentEnd > rasterThumbnailHeaderCap {
			return 0, 0, errors.New("JPEG segment is truncated or exceeds the structural scan limit")
		}
		if jpegStartOfFrameMarker(marker) {
			if segmentLength < 8 {
				return 0, 0, errors.New("JPEG start-of-frame segment is truncated")
			}
			var dimensions [5]byte
			if err := readRasterHeaderAt(file, offset+2, dimensions[:]); err != nil {
				return 0, 0, err
			}
			height = int(binary.BigEndian.Uint16(dimensions[1:3]))
			width = int(binary.BigEndian.Uint16(dimensions[3:5]))
			sawStartOfFrame = width > 0 && height > 0
		}
		if marker == 0xda {
			if !sawStartOfFrame || segmentLength < 6 || size <= segmentEnd+2 {
				return 0, 0, errors.New("JPEG scan header is invalid")
			}
			var end [2]byte
			if err := readRasterHeaderAt(file, size-2, end[:]); err != nil {
				return 0, 0, err
			}
			if end != [2]byte{0xff, 0xd9} {
				return 0, 0, errors.New("JPEG end marker is missing")
			}
			return width, height, nil
		}
		offset = segmentEnd
	}
	return 0, 0, errors.New("JPEG has too many header segments")
}

func jpegStartOfFrameMarker(marker byte) bool {
	return marker >= 0xc0 && marker <= 0xcf && marker != 0xc4 && marker != 0xc8 && marker != 0xcc
}

func boundedGIFThumbnailDimensions(file *os.File, size int64) (int, int, error) {
	if size < 14 {
		return 0, 0, errors.New("GIF header is truncated")
	}
	var header [13]byte
	if err := readRasterHeaderAt(file, 0, header[:]); err != nil {
		return 0, 0, err
	}
	width := int(binary.LittleEndian.Uint16(header[6:8]))
	height := int(binary.LittleEndian.Uint16(header[8:10]))
	offset := int64(13)
	if header[10]&0x80 != 0 {
		offset += int64(3 * (1 << ((header[10] & 0x07) + 1)))
	}
	if offset >= size {
		return 0, 0, errors.New("GIF color table is truncated")
	}
	sawImage := false
	operations := 0
	for operations < rasterThumbnailMaxChunks {
		operations++
		var introducer [1]byte
		if err := readRasterHeaderAt(file, offset, introducer[:]); err != nil {
			return 0, 0, err
		}
		offset++
		switch introducer[0] {
		case 0x2c:
			var descriptor [9]byte
			if err := readRasterHeaderAt(file, offset, descriptor[:]); err != nil {
				return 0, 0, err
			}
			offset += int64(len(descriptor))
			if descriptor[8]&0x80 != 0 {
				offset += int64(3 * (1 << ((descriptor[8] & 0x07) + 1)))
			}
			if offset >= size {
				return 0, 0, errors.New("GIF image descriptor is truncated")
			}
			offset++ // LZW minimum code size.
			var err error
			offset, operations, err = skipGIFSubBlocks(file, size, offset, operations)
			if err != nil {
				return 0, 0, err
			}
			sawImage = true
		case 0x21:
			if offset >= size {
				return 0, 0, errors.New("GIF extension is truncated")
			}
			offset++ // Extension label.
			var err error
			offset, operations, err = skipGIFSubBlocks(file, size, offset, operations)
			if err != nil {
				return 0, 0, err
			}
		case 0x3b:
			if !sawImage || offset != size {
				return 0, 0, errors.New("GIF trailer is invalid or the source is truncated")
			}
			return width, height, nil
		default:
			return 0, 0, errors.New("GIF block introducer is invalid")
		}
	}
	return 0, 0, errors.New("GIF has too many structural blocks")
}

func skipGIFSubBlocks(file *os.File, size, offset int64, operations int) (int64, int, error) {
	for operations < rasterThumbnailMaxChunks {
		operations++
		var length [1]byte
		if err := readRasterHeaderAt(file, offset, length[:]); err != nil {
			return 0, operations, err
		}
		offset++
		if length[0] == 0 {
			return offset, operations, nil
		}
		if int64(length[0]) > size-offset {
			return 0, operations, errors.New("GIF data sub-block is truncated")
		}
		offset += int64(length[0])
	}
	return 0, operations, errors.New("GIF has too many data sub-blocks")
}

func boundedBMPThumbnailDimensions(file *os.File, size int64) (int, int, error) {
	if size < 26 || size > math.MaxUint32 {
		return 0, 0, errors.New("BMP header is truncated")
	}
	var prefix [26]byte
	if err := readRasterHeaderAt(file, 0, prefix[:]); err != nil {
		return 0, 0, err
	}
	if string(prefix[0:2]) != "BM" || int64(binary.LittleEndian.Uint32(prefix[2:6])) != size {
		return 0, 0, errors.New("BMP file-size declaration is invalid")
	}
	pixelOffset := int64(binary.LittleEndian.Uint32(prefix[10:14]))
	dibSize := int64(binary.LittleEndian.Uint32(prefix[14:18]))
	if dibSize < 12 || dibSize > rasterThumbnailHeaderCap || pixelOffset < 14+dibSize || pixelOffset >= size {
		return 0, 0, errors.New("BMP data offset or DIB header is invalid")
	}
	if dibSize == 12 {
		width := int(binary.LittleEndian.Uint16(prefix[18:20]))
		height := int(binary.LittleEndian.Uint16(prefix[20:22]))
		bitsPerPixel := binary.LittleEndian.Uint16(prefix[24:26])
		if binary.LittleEndian.Uint16(prefix[22:24]) != 1 || !validBMPBitsPerPixel(bitsPerPixel) || width <= 0 || height <= 0 {
			return 0, 0, errors.New("BMP core header is invalid")
		}
		return width, height, nil
	}
	if dibSize < 40 {
		return 0, 0, errors.New("unsupported BMP DIB header")
	}
	var info [40]byte
	if err := readRasterHeaderAt(file, 14, info[:]); err != nil {
		return 0, 0, err
	}
	width64 := int64(int32(binary.LittleEndian.Uint32(info[4:8])))
	height64 := int64(int32(binary.LittleEndian.Uint32(info[8:12])))
	if height64 == math.MinInt32 {
		return 0, 0, errors.New("BMP height overflows")
	}
	if height64 < 0 {
		height64 = -height64
	}
	if width64 <= 0 || height64 <= 0 || width64 > math.MaxInt32 || height64 > math.MaxInt32 || binary.LittleEndian.Uint16(info[12:14]) != 1 || !validBMPBitsPerPixel(binary.LittleEndian.Uint16(info[14:16])) {
		return 0, 0, errors.New("BMP info header dimensions are invalid")
	}
	return int(width64), int(height64), nil
}

func validBMPBitsPerPixel(bits uint16) bool {
	switch bits {
	case 1, 2, 4, 8, 16, 24, 32:
		return true
	default:
		return false
	}
}

func boundedWebPThumbnailDimensions(file *os.File, size int64) (int, int, error) {
	if size < 20 || size > math.MaxUint32+8 {
		return 0, 0, errors.New("WebP RIFF header is truncated")
	}
	var riff [12]byte
	if err := readRasterHeaderAt(file, 0, riff[:]); err != nil {
		return 0, 0, err
	}
	declaredSize := int64(binary.LittleEndian.Uint32(riff[4:8])) + 8
	if string(riff[0:4]) != "RIFF" || string(riff[8:12]) != "WEBP" || declaredSize != size {
		return 0, 0, errors.New("WebP RIFF size is invalid")
	}
	offset := int64(12)
	canvasWidth, canvasHeight := 0, 0
	imageWidth, imageHeight := 0, 0
	for chunk := 0; chunk < rasterThumbnailMaxChunks; chunk++ {
		if offset > size-8 || offset > rasterThumbnailHeaderCap {
			return 0, 0, errors.New("WebP chunk header is truncated or exceeds the structural scan limit")
		}
		var header [8]byte
		if err := readRasterHeaderAt(file, offset, header[:]); err != nil {
			return 0, 0, err
		}
		chunkSize := int64(binary.LittleEndian.Uint32(header[4:8]))
		dataOffset := offset + 8
		paddedSize := chunkSize + chunkSize%2
		if paddedSize > math.MaxInt64-dataOffset || dataOffset+paddedSize > size {
			return 0, 0, errors.New("WebP chunk is truncated")
		}
		switch string(header[0:4]) {
		case "VP8X":
			if chunkSize < 10 {
				return 0, 0, errors.New("WebP VP8X header is truncated")
			}
			var body [10]byte
			if err := readRasterHeaderAt(file, dataOffset, body[:]); err != nil {
				return 0, 0, err
			}
			if body[0]&0x02 != 0 {
				return 0, 0, errors.New("animated WebP thumbnails require a derived raster")
			}
			canvasWidth = 1 + int(uint24LittleEndian(body[4:7]))
			canvasHeight = 1 + int(uint24LittleEndian(body[7:10]))
		case "VP8 ":
			if chunkSize < 10 {
				return 0, 0, errors.New("WebP VP8 header is truncated")
			}
			var body [10]byte
			if err := readRasterHeaderAt(file, dataOffset, body[:]); err != nil {
				return 0, 0, err
			}
			if body[3] != 0x9d || body[4] != 0x01 || body[5] != 0x2a {
				return 0, 0, errors.New("WebP VP8 frame signature is invalid")
			}
			imageWidth = int(binary.LittleEndian.Uint16(body[6:8]) & 0x3fff)
			imageHeight = int(binary.LittleEndian.Uint16(body[8:10]) & 0x3fff)
		case "VP8L":
			if chunkSize < 5 {
				return 0, 0, errors.New("WebP VP8L header is truncated")
			}
			var body [5]byte
			if err := readRasterHeaderAt(file, dataOffset, body[:]); err != nil {
				return 0, 0, err
			}
			if body[0] != 0x2f {
				return 0, 0, errors.New("WebP VP8L signature is invalid")
			}
			imageWidth = 1 + int(body[1]) + int(body[2]&0x3f)<<8
			imageHeight = 1 + int(body[2]>>6) + int(body[3])<<2 + int(body[4]&0x0f)<<10
		}
		offset = dataOffset + paddedSize
		if offset == size {
			if imageWidth <= 0 || imageHeight <= 0 {
				return 0, 0, errors.New("WebP image payload is missing")
			}
			if canvasWidth > 0 || canvasHeight > 0 {
				if canvasWidth <= 0 || canvasHeight <= 0 || imageWidth > canvasWidth || imageHeight > canvasHeight {
					return 0, 0, errors.New("WebP canvas dimensions are invalid")
				}
				return canvasWidth, canvasHeight, nil
			}
			return imageWidth, imageHeight, nil
		}
	}
	return 0, 0, errors.New("WebP has too many chunks")
}

func uint24LittleEndian(body []byte) uint32 {
	return uint32(body[0]) | uint32(body[1])<<8 | uint32(body[2])<<16
}

func renderBoundedRasterThumbnailPNG(path string, budget *byteAdmissionBudget) ([]byte, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()
	config, format, err := boundedRasterThumbnailConfig(file)
	if err != nil {
		return nil, err
	}
	pixels, ok := checkedNonNegativeProduct(int64(config.Width), int64(config.Height))
	if !ok {
		return nil, errors.New("raster thumbnail pixel count overflows")
	}
	reservation, ok := checkedNonNegativeProduct(pixels, thumbnailDecodeBytesPerPixel)
	if !ok || !budget.tryAcquire(reservation) {
		return nil, errThumbnailAdmission
	}
	defer budget.release(reservation)
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}
	source, decodedFormat, err := image.Decode(io.LimitReader(file, rasterThumbnailMaxInput+1))
	if err != nil {
		return nil, fmt.Errorf("raster thumbnail pixels could not be decoded: %w", err)
	}
	if decodedFormat != format {
		return nil, errors.New("raster thumbnail format changed between header and pixel decode")
	}
	bounds := source.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	decodedPixels, ok := checkedNonNegativeProduct(int64(width), int64(height))
	if !ok || width <= 0 || height <= 0 || width > rasterThumbnailMaxAxis || height > rasterThumbnailMaxAxis || decodedPixels > rasterThumbnailMaxPixels {
		return nil, errors.New("decoded raster thumbnail dimensions are invalid")
	}
	targetWidth, targetHeight := boundedThumbnailDimensions(width, height)
	target := image.NewRGBA(image.Rect(0, 0, targetWidth, targetHeight))
	xdraw.ApproxBiLinear.Scale(target, target.Bounds(), source, bounds, xdraw.Src, nil)
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, target); err != nil {
		return nil, err
	}
	body := encoded.Bytes()
	if err := validateThumbnailPNG("image/png", body); err != nil {
		return nil, err
	}
	return body, nil
}

func boundedThumbnailDimensions(width, height int) (int, int) {
	if width <= 0 || height <= 0 {
		return 0, 0
	}
	if width <= thumbnailMaxDimension && height <= thumbnailMaxDimension {
		return width, height
	}
	if width >= height {
		return thumbnailMaxDimension, max(1, int(int64(height)*thumbnailMaxDimension/int64(width)))
	}
	return max(1, int(int64(width)*thumbnailMaxDimension/int64(height))), thumbnailMaxDimension
}

type niftiThumbnailPlan struct {
	path              string
	geometry          niftiGeometry
	gzipped           bool
	headerConsumed    int
	planeOffset       int64
	planeBytes        int64
	decompressionWork int64
	reservation       int64
}

func niftiThumbnailPreflight(path string) (niftiThumbnailPlan, error) {
	gzipped := isGzipPath(path)
	file, err := os.Open(path)
	if err != nil {
		return niftiThumbnailPlan{}, err
	}
	defer func() { _ = file.Close() }()
	var reader io.Reader = file
	var gzipReader *gzip.Reader
	if gzipped {
		gzipReader, err = gzip.NewReader(file)
		if err != nil {
			return niftiThumbnailPlan{}, err
		}
		defer func() { _ = gzipReader.Close() }()
		reader = gzipReader
	}
	header, consumed, err := readNiftiHeaderBytes(reader)
	if err != nil {
		return niftiThumbnailPlan{}, fmt.Errorf("read NIfTI thumbnail header: %w", err)
	}
	geometry, err := parseNiftiGeometry(header)
	if err != nil {
		return niftiThumbnailPlan{}, err
	}
	if geometry.width > niftiThumbnailMaxAxis || geometry.height > niftiThumbnailMaxAxis {
		return niftiThumbnailPlan{}, errors.New("NIfTI thumbnail plane dimensions exceed the limit")
	}
	planePixels, ok := checkedNonNegativeProduct(int64(geometry.width), int64(geometry.height))
	if !ok || planePixels <= 0 {
		return niftiThumbnailPlan{}, errors.New("NIfTI thumbnail plane dimensions overflow")
	}
	planeBytes, ok := checkedNonNegativeProduct(planePixels, int64(geometry.bytesPerVoxel))
	if !ok || planeBytes <= 0 || planeBytes > niftiThumbnailMaxPlaneBytes {
		return niftiThumbnailPlan{}, errors.New("NIfTI thumbnail plane exceeds the byte limit")
	}
	zOffset, ok := checkedNonNegativeProduct(int64(geometry.depth/2), planeBytes)
	if !ok || geometry.voxOffset > math.MaxInt64-zOffset {
		return niftiThumbnailPlan{}, errors.New("NIfTI thumbnail plane offset overflows")
	}
	planeOffset := geometry.voxOffset + zOffset
	if planeOffset > math.MaxInt64-planeBytes {
		return niftiThumbnailPlan{}, errors.New("NIfTI thumbnail work estimate overflows")
	}
	workBytes := planeOffset + planeBytes
	decompressionWork := int64(0)
	if gzipped && workBytes > niftiThumbnailMaxGzipWork {
		return niftiThumbnailPlan{}, errors.New("NIfTI thumbnail gzip work exceeds the limit")
	}
	if gzipped {
		decompressionWork = workBytes
	}
	if !gzipped {
		info, err := file.Stat()
		if err != nil {
			return niftiThumbnailPlan{}, err
		}
		if !info.Mode().IsRegular() || workBytes > info.Size() {
			return niftiThumbnailPlan{}, errors.New("NIfTI thumbnail plane is truncated")
		}
	}
	decodedEstimate, ok := checkedNonNegativeProduct(planePixels, thumbnailDecodeBytesPerPixel)
	if !ok {
		return niftiThumbnailPlan{}, errors.New("NIfTI thumbnail decode estimate overflows")
	}
	reservation, ok := checkedThumbnailReservation(decodedEstimate, planeBytes, decompressionWork)
	if !ok {
		return niftiThumbnailPlan{}, errors.New("NIfTI thumbnail admission estimate overflows")
	}
	return niftiThumbnailPlan{
		path: path, geometry: geometry, gzipped: gzipped, headerConsumed: consumed,
		planeOffset: planeOffset, planeBytes: planeBytes, decompressionWork: decompressionWork,
		reservation: reservation,
	}, nil
}

func checkedThumbnailReservation(costs ...int64) (int64, bool) {
	total := int64(0)
	for _, cost := range costs {
		if cost < 0 || cost > math.MaxInt64-total {
			return 0, false
		}
		total += cost
	}
	return total, total > 0
}

func readNiftiThumbnailPlane(plan niftiThumbnailPlan) ([]byte, error) {
	file, err := os.Open(plan.path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()
	if !plan.gzipped {
		plane := make([]byte, int(plan.planeBytes))
		if _, err := file.ReadAt(plane, plan.planeOffset); err != nil {
			return nil, fmt.Errorf("read NIfTI thumbnail plane: %w", err)
		}
		return plane, nil
	}
	gzipReader, err := gzip.NewReader(file)
	if err != nil {
		return nil, err
	}
	defer func() { _ = gzipReader.Close() }()
	header, consumed, err := readNiftiHeaderBytes(gzipReader)
	if err != nil {
		return nil, fmt.Errorf("read NIfTI thumbnail header: %w", err)
	}
	geometry, err := parseNiftiGeometry(header)
	if err != nil {
		return nil, err
	}
	if geometry.width != plan.geometry.width || geometry.height != plan.geometry.height || geometry.depth != plan.geometry.depth || geometry.bytesPerVoxel != plan.geometry.bytesPerVoxel || consumed != plan.headerConsumed {
		return nil, errors.New("NIfTI thumbnail geometry changed during read")
	}
	skip := plan.planeOffset - int64(consumed)
	if skip < 0 || plan.decompressionWork <= 0 || plan.decompressionWork > niftiThumbnailMaxGzipWork {
		return nil, errors.New("NIfTI thumbnail gzip work exceeds the limit")
	}
	if skip > 0 {
		if _, err := io.CopyN(io.Discard, gzipReader, skip); err != nil {
			return nil, fmt.Errorf("skip to NIfTI thumbnail plane: %w", err)
		}
	}
	plane := make([]byte, int(plan.planeBytes))
	if _, err := io.ReadFull(gzipReader, plane); err != nil {
		return nil, fmt.Errorf("read NIfTI thumbnail plane: %w", err)
	}
	return plane, nil
}

func renderNiftiThumbnailPNG(path string, request *http.Request, budget *byteAdmissionBudget) ([]byte, error) {
	plan, err := niftiThumbnailPreflight(path)
	if err != nil {
		return nil, err
	}
	if !budget.tryAcquire(plan.reservation) {
		return nil, errThumbnailAdmission
	}
	defer budget.release(plan.reservation)
	plane, err := readNiftiThumbnailPlane(plan)
	if err != nil {
		return nil, err
	}
	if plan.geometry.bytesPerVoxel > 1 && plan.geometry.order != binary.LittleEndian {
		normalizeScalarPayloadToLittleEndian(plane, plan.geometry.bytesPerVoxel)
	}
	minValue, maxValue := niftiScalarRange(plane, plan.geometry.dtype, plan.geometry.bytesPerVoxel)
	volume := niftiScalarVolume{
		Width: plan.geometry.width, Height: plan.geometry.height, Depth: 1,
		DType: plan.geometry.dtype, BytesPerVoxel: plan.geometry.bytesPerVoxel, Data: plane,
		RawMin: minValue, RawMax: maxValue, SclSlope: plan.geometry.sclSlope, SclInter: plan.geometry.sclInter,
	}
	transform := uploadPreviewTransformFromRequest(request)
	if !transform.WindowActive && !transform.FullRange && niftiScalarRangeLooksCTLike(volume) {
		transform.WindowMin = 0
		transform.WindowMax = 80
		transform.WindowActive = true
		transform.WindowIsPhysical = true
	}
	windowCodeMin, windowCodeMax := scalarPreviewWindow(volume, transform)
	windowMin := volume.physical(windowCodeMin)
	windowMax := volume.physical(windowCodeMax)
	if windowMax < windowMin {
		windowMin, windowMax = windowMax, windowMin
	}
	source := image.NewGray(image.Rect(0, 0, volume.Width, volume.Height))
	gamma := transform.Gamma
	if gamma <= 0 {
		gamma = 1
	}
	scale := windowMax - windowMin
	for y := 0; y < volume.Height; y++ {
		for x := 0; x < volume.Width; x++ {
			code := niftiScalarDataValue(plane, (y*volume.Width+x)*volume.BytesPerVoxel, volume.DType, volume.BytesPerVoxel)
			value := volume.physical(code)
			normalized := 0.0
			if scale > 0 && numberIsFinite(value) {
				normalized = (value - windowMin) / scale
			}
			normalized = math.Max(0, math.Min(1, normalized))
			if gamma != 1 {
				normalized = math.Pow(normalized, 1/gamma)
			}
			pixel := uint8(math.Round(normalized * 255))
			if transform.Negative {
				pixel = 255 - pixel
			}
			source.SetGray(x, y, color.Gray{Y: pixel})
		}
	}
	targetWidth, targetHeight := boundedThumbnailDimensions(volume.Width, volume.Height)
	var output image.Image = source
	if targetWidth != volume.Width || targetHeight != volume.Height {
		target := image.NewGray(image.Rect(0, 0, targetWidth, targetHeight))
		xdraw.ApproxBiLinear.Scale(target, target.Bounds(), source, source.Bounds(), xdraw.Src, nil)
		output = target
	}
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, output); err != nil {
		return nil, err
	}
	body := encoded.Bytes()
	if err := validateThumbnailPNG("image/png", body); err != nil {
		return nil, err
	}
	return body, nil
}

func validateThumbnailPNG(contentType string, body []byte) error {
	mediaType, _, err := mime.ParseMediaType(contentType)
	if err != nil || mediaType != "image/png" {
		return errors.New("thumbnail service returned a non-PNG response")
	}
	if len(body) == 0 || int64(len(body)) > thumbnailMaxEncodedBytes {
		return errors.New("thumbnail service response exceeds the 8 MiB encoded limit")
	}
	config, err := png.DecodeConfig(bytes.NewReader(body))
	if err != nil {
		return errors.New("thumbnail service returned malformed PNG data")
	}
	if config.Width <= 0 || config.Height <= 0 || config.Width > thumbnailMaxDimension || config.Height > thumbnailMaxDimension {
		return errors.New("thumbnail service returned out-of-bounds PNG dimensions")
	}
	if _, err := png.Decode(bytes.NewReader(body)); err != nil {
		return errors.New("thumbnail service returned malformed PNG data")
	}
	return nil
}

func writeResourceThumbnailError(w http.ResponseWriter, resourceID, renderer string, status int, err error) {
	slog.Error("resource thumbnail render failed", "resource_id", resourceID, "renderer", renderer, "status", status, "error", err)
	if errors.Is(err, errThumbnailAdmission) {
		w.Header().Set("Retry-After", "1")
	}
	message := "thumbnail is unavailable for this resource"
	if status == http.StatusServiceUnavailable {
		message = "thumbnail rendering is temporarily unavailable"
	}
	writeError(w, status, errors.New(message))
}

func writeRenderedThumbnailPNG(w http.ResponseWriter, resourceID, renderer string, body []byte) bool {
	if err := writeThumbnailPNG(w, body); err != nil {
		writeResourceThumbnailError(w, resourceID, renderer, http.StatusServiceUnavailable, err)
		return false
	}
	return true
}

func (deps ServerDeps) proxyBoundedThumbnailPNG(w http.ResponseWriter, r *http.Request, resourceID, renderer, endpoint string, query url.Values) {
	cacheKey, cacheable := imageCacheKey(endpoint, query)
	if deps.imageCache != nil && cacheable {
		if cached, hit := deps.imageCache.get(cacheKey); hit && cached.status == http.StatusOK && validateThumbnailPNG(cached.contentType, cached.body) == nil {
			writeCachedResponse(w, cached, "hit")
			return
		}
	}
	base := strings.TrimRight(strings.TrimSpace(deps.ImageServiceURL), "/")
	target := base + endpoint
	if encoded := query.Encode(); encoded != "" {
		target += "?" + encoded
	}
	request, err := http.NewRequestWithContext(r.Context(), http.MethodGet, target, nil)
	if err != nil {
		writeResourceThumbnailError(w, resourceID, renderer, http.StatusServiceUnavailable, err)
		return
	}
	response, err := imageServiceHTTPClient.Do(request)
	if err != nil {
		writeResourceThumbnailError(w, resourceID, renderer, http.StatusServiceUnavailable, err)
		return
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		status := http.StatusServiceUnavailable
		if response.StatusCode == http.StatusUnsupportedMediaType || response.StatusCode == http.StatusUnprocessableEntity {
			status = http.StatusUnsupportedMediaType
		}
		writeResourceThumbnailError(w, resourceID, renderer, status, fmt.Errorf("thumbnail service returned HTTP %d", response.StatusCode))
		return
	}
	if response.ContentLength > thumbnailMaxEncodedBytes {
		writeResourceThumbnailError(w, resourceID, renderer, http.StatusServiceUnavailable, errors.New("thumbnail service returned an oversized response"))
		return
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, thumbnailMaxEncodedBytes+1))
	if err != nil {
		writeResourceThumbnailError(w, resourceID, renderer, http.StatusServiceUnavailable, err)
		return
	}
	if err := validateThumbnailPNG(response.Header.Get("Content-Type"), body); err != nil {
		writeResourceThumbnailError(w, resourceID, renderer, http.StatusServiceUnavailable, err)
		return
	}
	w.Header().Set("X-Ultra-Cache", "miss")
	if !writeRenderedThumbnailPNG(w, resourceID, renderer, body) {
		return
	}
	if deps.imageCache != nil && cacheable {
		deps.imageCache.put(cacheKey, &cachedResponse{
			status: http.StatusOK, contentType: "image/png", body: body,
		}, int64(len(body)))
	}
}

func resourceIsCifti(record resourceRecord, path string) bool {
	if isCiftiName(record.OriginalName) {
		return true
	}
	if !isNiftiUpload(record.OriginalName, record.ContentType) {
		return false
	}
	_, ok := niftiCiftiPeek(path, record.OriginalName)
	return ok
}

func isScientificThumbnailCandidate(record resourceRecord) bool {
	return isTIFFUpload(record.OriginalName, record.ContentType) ||
		hasPyramidMicroscopyExtension(record.OriginalName) ||
		strings.EqualFold(strings.TrimSpace(record.ResourceKind), "image") ||
		strings.HasPrefix(strings.ToLower(strings.TrimSpace(record.ContentType)), "image/")
}

// handleServeResourceThumbnail authenticates and resolves the resource before
// dispatch. Every successful response is a bounded PNG; no failure path serves
// or falls back to the original upload bytes.
func (deps ServerDeps) handleServeResourceThumbnail(w http.ResponseWriter, r *http.Request) {
	resourceID := chi.URLParam(r, "file_id")
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeResourceThumbnailError(w, resourceID, "resource_resolution", http.StatusServiceUnavailable, err)
		return
	}
	record, path, err := deps.findUploadResourceForRequest(r.Context(), root, deps.principalFromRequest(r, ""), resourceID)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			writeError(w, http.StatusNotFound, errors.New("resource not found"))
			return
		}
		writeResourceThumbnailError(w, resourceID, "resource_resolution", http.StatusServiceUnavailable, err)
		return
	}
	if resourceIsCifti(record, path) {
		body, err := renderCiftiThumbnailPNG(path)
		if err != nil {
			writeResourceThumbnailError(w, record.FileID, "cifti", http.StatusUnsupportedMediaType, err)
			return
		}
		writeRenderedThumbnailPNG(w, record.FileID, "cifti", body)
		return
	}
	if isNiftiUpload(record.OriginalName, record.ContentType) {
		body, err := renderNiftiThumbnailPNG(path, r, thumbnailInFlightBudget)
		if err != nil {
			status := http.StatusUnsupportedMediaType
			if errors.Is(err, errThumbnailAdmission) {
				status = http.StatusServiceUnavailable
			}
			writeResourceThumbnailError(w, record.FileID, "nifti", status, err)
			return
		}
		writeRenderedThumbnailPNG(w, record.FileID, "nifti", body)
		return
	}
	if goNativeThumbnailable(record) {
		body, err := renderBoundedRasterThumbnailPNG(path, thumbnailInFlightBudget)
		if err == nil {
			writeRenderedThumbnailPNG(w, record.FileID, "native_raster", body)
			return
		}
		if deps.imageServiceConfigured() {
			if derivative, committed := committedThumbnailDerivative(root, record, path); committed {
				deps.proxyBoundedThumbnailPNG(w, r, record.FileID, "derived_raster", "/thumbnail", url.Values{"path": {derivative}, "max_size": {"512"}})
				return
			}
		}
		status := http.StatusUnsupportedMediaType
		if errors.Is(err, errThumbnailAdmission) {
			status = http.StatusServiceUnavailable
		}
		writeResourceThumbnailError(w, record.FileID, "native_raster", status, err)
		return
	}
	if isVideoUpload(record.OriginalName, record.ContentType) {
		if !deps.imageServiceConfigured() {
			writeResourceThumbnailError(w, record.FileID, "video_poster", http.StatusServiceUnavailable, errors.New("video thumbnail service is not configured"))
			return
		}
		deps.proxyBoundedThumbnailPNG(w, r, record.FileID, "video_poster", "/video-poster", url.Values{"path": {path}, "max_size": {"512"}})
		return
	}
	// OME-Zarr thumbnails come from the ngff-service (smallest multiscale level).
	if deps.servesViaNgff(record, path) {
		deps.ngffDeps().proxyBoundedThumbnailPNG(w, r, record.FileID, "ngff", "/thumbnail", url.Values{"path": {path}, "max_size": {"512"}})
		return
	}
	if deps.ngffServiceUnavailable(record, path) {
		writeResourceThumbnailError(w, record.FileID, "ngff", http.StatusServiceUnavailable, errNgffServiceNotConfigured)
		return
	}
	if !isScientificThumbnailCandidate(record) {
		writeResourceThumbnailError(w, record.FileID, "unsupported", http.StatusUnsupportedMediaType, errors.New("resource format does not support thumbnails"))
		return
	}
	if !deps.imageServiceConfigured() {
		writeResourceThumbnailError(w, record.FileID, "derived_scientific", http.StatusServiceUnavailable, errors.New("scientific thumbnail service is not configured"))
		return
	}
	servePath, committed := committedThumbnailDerivative(root, record, path)
	if !committed {
		writeResourceThumbnailError(w, record.FileID, "derived_scientific", http.StatusUnsupportedMediaType, errors.New("scientific thumbnail derivative is not ready"))
		return
	}
	deps.proxyBoundedThumbnailPNG(w, r, record.FileID, "derived_scientific", "/thumbnail", url.Values{"path": {servePath}, "max_size": {"512"}})
}

// goCanDecodeHistogram reports whether the native Go decoder can read this
// resource's pixels for a histogram (standard rasters + TIFF). Microscopy
// containers (CZI/ND2/…) cannot and are routed to libbioimage.
func goCanDecodeHistogram(record resourceRecord) bool {
	if strings.HasPrefix(strings.ToLower(strings.TrimSpace(record.ContentType)), "image/") {
		return true
	}
	return isTIFFUpload(record.OriginalName, record.ContentType)
}

// handleGetUploadHistogramService obtains volume calibration from the original
// source. A display pyramid or Go's first-page decoder cannot be semantic
// authority for C/T/Z selection.
func (deps ServerDeps) handleGetUploadHistogramService(w http.ResponseWriter, r *http.Request) {
	if !deps.imageServiceConfigured() {
		deps.handleGetUploadHistogram(w, r)
		return
	}
	_, record, path, ok := deps.resolveUploadServingRequest(w, r)
	if !ok {
		return
	}
	scope, scopePresent, scopeErr := exactRawQueryValue(
		r.URL.Query(),
		[]string{"scope"},
		"histogram scope",
	)
	if scopeErr != nil {
		writeError(w, http.StatusBadRequest, scopeErr)
		return
	}
	if !scopePresent {
		if isNiftiUpload(record.OriginalName, record.ContentType) || !isTIFFUpload(record.OriginalName, record.ContentType) && goCanDecodeHistogram(record) {
			deps.handleGetUploadHistogram(w, r)
			return
		}
		sourceInfo, timeCount, channelCount, depth, authorityErr :=
			deps.sourceImageServiceViewerInfo(r.Context(), path)
		if authorityErr != nil {
			writeImageSourceAuthorityError(w, authorityErr)
			return
		}
		viewer, _ := jsonObject(sourceInfo["viewer"])
		renderPolicy, _ := viewer["render_policy"].(string)
		if isTIFFUpload(record.OriginalName, record.ContentType) &&
			timeCount == 1 && channelCount <= 4 && depth == 1 && renderPolicy == "display" {
			deps.handleGetUploadHistogram(w, r)
			return
		}
		selectedChannels, timeIndex, selectionErr :=
			parseExactDisplayHistogramSelection(r, channelCount, timeCount)
		if selectionErr != nil {
			writeError(w, http.StatusBadRequest, selectionErr)
			return
		}
		bins := parseUploadHistogramBins(r)
		query := url.Values{
			"path":     {path},
			"bins":     {strconv.Itoa(bins)},
			"scope":    {"display"},
			"channels": {joinIntCSV(selectedChannels)},
			"t":        {strconv.Itoa(timeIndex)},
		}
		core, err := deps.imageServiceGetJSON(r.Context(), "/histogram", query)
		if err != nil {
			writeImageSourceAuthorityError(w, err)
			return
		}
		mapped, err := mapImageServiceDisplayHistogram(
			core,
			record.FileID,
			bins,
			selectedChannels,
			timeIndex,
		)
		if err != nil {
			writeImageSourceAuthorityError(w, err)
			return
		}
		writeJSON(w, http.StatusOK, mapped)
		return
	}
	if scope != "volume" {
		writeError(w, http.StatusBadRequest, errors.New("histogram scope must be volume when supplied"))
		return
	}
	if isNiftiUpload(record.OriginalName, record.ContentType) {
		writeError(w, http.StatusUnprocessableEntity, errors.New("source volume calibration is unsupported for NIfTI"))
		return
	}
	bins := parseUploadHistogramBins(r)
	channelIdx, err := parseExactHistogramChannel(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	timeIdx, err := parseExactScalarIndex(r, []string{"t", "time", "timepoint"})
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	_, timeCount, channelCount, _, authorityErr := deps.sourceImageServiceViewerInfo(
		r.Context(),
		path,
	)
	if authorityErr != nil {
		writeImageSourceAuthorityError(w, authorityErr)
		return
	}
	if channelIdx >= channelCount || timeIdx >= timeCount {
		writeError(w, http.StatusBadRequest, errors.New("histogram channel/time selection is out of range"))
		return
	}
	core, err := deps.imageServiceGetJSON(r.Context(), "/histogram", url.Values{
		"path":    {path},
		"bins":    {strconv.Itoa(bins)},
		"channel": {strconv.Itoa(channelIdx)},
		"t":       {strconv.Itoa(timeIdx)},
		"scope":   {"volume"},
	})
	if err != nil {
		writeImageSourceAuthorityError(w, err)
		return
	}
	mapped, err := mapImageServiceHistogram(
		core,
		record.FileID,
		strings.TrimSpace(record.SHA256),
		channelIdx,
		timeIdx,
		bins,
	)
	if err != nil {
		writeImageSourceAuthorityError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, mapped)
}

func mapImageServiceDisplayHistogram(
	core map[string]any,
	fileID string,
	bins int,
	expectedChannels []int,
	expectedTime int,
) (map[string]any, error) {
	coreBins, binsOK := jsonInt(core["bins"])
	rawChannels, channelsOK := core["channels"].([]any)
	scope, scopeOK := core["scope"].(string)
	if !binsOK || coreBins != bins || !channelsOK || len(rawChannels) == 0 ||
		!scopeOK || scope != "display" {
		return nil, errors.New("image service display histogram response is malformed")
	}
	channelIndices := make([]int, 0, len(rawChannels))
	counts := make([]int, bins)
	sampleCount := 0
	minimum := math.Inf(1)
	maximum := math.Inf(-1)
	var edges []float64
	for _, raw := range rawChannels {
		channel, ok := jsonObject(raw)
		if !ok {
			return nil, errors.New("image service display histogram channel is malformed")
		}
		channelIndex, indexOK := jsonInt(channel["index"])
		channelSampleCount, sampleOK := jsonInt(channel["sample_count"])
		if !sampleOK || channelSampleCount <= 0 {
			channelSampleCount = 0
		}
		channelCounts, channelEdges, channelMin, channelMax, err :=
			validateImageServiceHistogramBins(channel, bins, channelSampleCount)
		if err != nil {
			return nil, err
		}
		if !indexOK || channelIndex < 0 {
			return nil, errors.New("image service display histogram channel index is invalid")
		}
		if edges == nil {
			edges = channelEdges
		} else if !slices.Equal(edges, channelEdges) {
			return nil, errors.New("image service display histogram channels use different edges")
		}
		channelIndices = append(channelIndices, channelIndex)
		for index, value := range channelCounts {
			counts[index] += value
		}
		sampleCount += channelSampleCount
		minimum = math.Min(minimum, channelMin)
		maximum = math.Max(maximum, channelMax)
	}
	dtype, dtypeOK := core["dtype"].(string)
	timeIndex, timeOK := jsonInt(core["t"])
	if !dtypeOK || strings.TrimSpace(dtype) == "" || !timeOK ||
		timeIndex != expectedTime || !slices.Equal(channelIndices, expectedChannels) {
		return nil, errors.New("image service display histogram identity is malformed")
	}
	return map[string]any{
		"file_id":      fileID,
		"bins":         bins,
		"dtype":        dtype,
		"channels":     channelIndices,
		"source":       "image-service-display",
		"sample_count": sampleCount,
		"histogram": map[string]any{
			"bins":            counts,
			"edges":           edges,
			"min":             minimum,
			"max":             maximum,
			"channel_indices": channelIndices,
			"time_index":      timeIndex,
		},
	}, nil
}

func parseExactDisplayHistogramSelection(
	r *http.Request,
	channelCount int,
	timeCount int,
) ([]int, int, error) {
	if r == nil || channelCount <= 0 || timeCount <= 0 {
		return nil, 0, errors.New("display histogram source axes are invalid")
	}
	rawChannels, channelsPresent, err := exactRawQueryValue(
		r.URL.Query(),
		[]string{"channels"},
		"histogram channels selector",
	)
	if err != nil {
		return nil, 0, err
	}
	channels := make([]int, 0, channelCount)
	if !channelsPresent {
		for channel := 0; channel < channelCount; channel++ {
			channels = append(channels, channel)
		}
	} else {
		seen := make(map[int]struct{}, channelCount)
		for _, part := range strings.Split(rawChannels, ",") {
			channel, parseErr := parseExactNonNegativeDecimal(
				strings.TrimSpace(part),
				"histogram channel index",
			)
			if parseErr != nil || channel >= channelCount {
				return nil, 0, errors.New("histogram channel selection is out of range")
			}
			if _, duplicate := seen[channel]; duplicate {
				return nil, 0, errors.New("duplicate histogram channel index")
			}
			seen[channel] = struct{}{}
			channels = append(channels, channel)
		}
		if len(channels) == 0 {
			return nil, 0, errors.New("histogram channel selection is empty")
		}
	}
	timeIndex, err := parseExactScalarIndex(r, []string{"t", "time", "timepoint"})
	if err != nil || timeIndex >= timeCount {
		return nil, 0, errors.New("histogram time selection is out of range")
	}
	return channels, timeIndex, nil
}

func joinIntCSV(values []int) string {
	parts := make([]string, len(values))
	for index, value := range values {
		parts[index] = strconv.Itoa(value)
	}
	return strings.Join(parts, ",")
}

func parseExactHistogramChannel(r *http.Request) (int, error) {
	value, err := parseExactScalarIndex(r, []string{"channel", "c"})
	if err != nil {
		return 0, err
	}
	if r == nil {
		return value, nil
	}
	rawChannels, channelsPresent, channelsErr := exactRawQueryValue(
		r.URL.Query(),
		[]string{"channels"},
		"histogram channels selector",
	)
	if channelsErr != nil {
		return 0, channelsErr
	}
	if !channelsPresent {
		return value, nil
	}
	if _, present, aliasErr := exactRawQueryValue(
		r.URL.Query(),
		[]string{"channel", "c"},
		"histogram channel index",
	); aliasErr != nil || present {
		if aliasErr != nil {
			return 0, aliasErr
		}
		return 0, errors.New("histogram channel must use one unambiguous selector")
	}
	if strings.Contains(rawChannels, ",") {
		return 0, errors.New("volume histogram requires exactly one channel")
	}
	return parseExactNonNegativeDecimal(rawChannels, "histogram channel index")
}

// mapImageServiceHistogram preserves source dtype/bin edges and profiling
// provenance while failing closed if the sidecar did not honor exact C/T.
func mapImageServiceHistogram(
	core map[string]any,
	fileID string,
	sourceSHA string,
	channelIdx, timeIdx, bins int,
) (map[string]any, error) {
	coreChannel, channelOK := jsonInt(core["channel"])
	coreTime, timeOK := jsonInt(core["t"])
	scope, _ := core["scope"].(string)
	if !channelOK || !timeOK || coreChannel != channelIdx || coreTime != timeIdx || scope != "volume" {
		return nil, errors.New("image service histogram identity did not match the requested source C/T")
	}
	channels, _ := core["channels"].([]any)
	var chosen map[string]any
	for _, c := range channels {
		cm, _ := c.(map[string]any)
		if cm == nil {
			continue
		}
		if idx, ok := jsonInt(cm["index"]); ok && idx == channelIdx {
			chosen = cm
			break
		}
	}
	if chosen == nil {
		return nil, errors.New("image service histogram omitted the requested source channel")
	}
	coreBins, binsOK := jsonInt(core["bins"])
	if !binsOK || coreBins != bins {
		return nil, errors.New("image service histogram bin count did not match the request")
	}
	dtype, dtypeOK := core["dtype"].(string)
	if !dtypeOK || strings.TrimSpace(dtype) == "" {
		return nil, errors.New("image service histogram omitted its source dtype")
	}
	sampleCount, sampleOK := jsonInt(core["sample_count"])
	if !sampleOK || sampleCount <= 0 {
		return nil, errors.New("image service histogram sample count is invalid")
	}
	counts, edges, minV, maxV, err := validateImageServiceHistogramBins(
		chosen,
		bins,
		sampleCount,
	)
	if err != nil {
		return nil, err
	}
	sampling, err := validateImageServiceHistogramSampling(core["sampling"], sampleCount)
	if err != nil {
		return nil, err
	}
	threshold, err := validateImageServiceHistogramThreshold(
		core["threshold"],
		channelIdx,
		timeIdx,
		sampleCount,
		sampling,
	)
	if err != nil {
		return nil, err
	}
	threshold["source_sha256"] = sourceSHA
	threshold["bins"] = bins
	threshold["sampling_strategy"] = sampling["strategy"]
	dataSemantics := sanitizeImageServiceDataSemantics(
		core["data_semantics"],
		threshold,
	)
	return map[string]any{
		"file_id":        fileID,
		"bins":           bins,
		"dtype":          dtype,
		"channels":       []int{channelIdx},
		"channel":        channelIdx,
		"t":              timeIdx,
		"source":         "image-service-source",
		"scope":          "volume",
		"sample_count":   sampleCount,
		"sampling":       sampling,
		"threshold":      threshold,
		"data_semantics": dataSemantics,
		"histogram": map[string]any{
			"bins":            counts,
			"edges":           edges,
			"min":             minV,
			"max":             maxV,
			"channel_indices": []int{channelIdx},
			"time_index":      timeIdx,
			"sampling":        sampling,
			"threshold":       threshold,
		},
	}, nil
}

func sanitizeImageServiceDataSemantics(
	value any,
	validatedThreshold map[string]any,
) map[string]any {
	semantics, ok := jsonObject(value)
	if !ok {
		return nil
	}
	kind, kindOK := semantics["kind"].(string)
	strength, strengthOK := semantics["strength"].(string)
	basis, basisOK := semantics["basis"].(string)
	recommended, recommendedOK := semantics["recommended_view"].(string)
	modes, modesOK := jsonStringSlice(semantics["supported_modes"])
	threshold, thresholdOK := jsonObject(semantics["threshold"])
	if !kindOK || (kind != "intensity" && kind != "binary_mask" && kind != "probability_mask") ||
		!strengthOK || (strength != "authoritative" && strength != "exact" &&
		strength != "suggested" && strength != "unknown") ||
		!basisOK || strings.TrimSpace(basis) == "" ||
		!recommendedOK || (recommended != "intensity" && recommended != "mask") ||
		!modesOK || len(modes) == 0 || !thresholdOK {
		return nil
	}
	for _, mode := range modes {
		if mode != "intensity" && mode != "mask" {
			return nil
		}
	}
	for _, key := range []string{
		"method",
		"value",
		"domain",
		"foreground",
		"sample_scope",
		"sample_count",
		"channel",
		"t",
		"sampling_algorithm",
	} {
		if fmt.Sprint(threshold[key]) != fmt.Sprint(validatedThreshold[key]) {
			return nil
		}
	}
	return map[string]any{
		"kind":             kind,
		"basis":            basis,
		"strength":         strength,
		"supported_modes":  modes,
		"recommended_view": recommended,
		"threshold":        validatedThreshold,
	}
}

func validateImageServiceHistogramBins(
	chosen map[string]any,
	bins int,
	sampleCount int,
) ([]int, []float64, float64, float64, error) {
	rawCounts, countsOK := chosen["counts"].([]any)
	rawEdges, edgesOK := chosen["edges"].([]any)
	if !countsOK || len(rawCounts) != bins || !edgesOK || len(rawEdges) != bins+1 {
		return nil, nil, 0, 0, errors.New("image service histogram bins or edges are malformed")
	}
	counts := make([]int, len(rawCounts))
	total := 0
	for index, raw := range rawCounts {
		value, ok := jsonInt(raw)
		if !ok || value < 0 || total > math.MaxInt-value {
			return nil, nil, 0, 0, errors.New("image service histogram counts must be non-negative integers")
		}
		counts[index] = value
		total += value
	}
	if total != sampleCount {
		return nil, nil, 0, 0, errors.New("image service histogram counts do not match sample_count")
	}
	edges := make([]float64, len(rawEdges))
	for index, raw := range rawEdges {
		value, ok := jsonFiniteFloat(raw)
		if !ok || index > 0 && value <= edges[index-1] {
			return nil, nil, 0, 0, errors.New("image service histogram edges must be finite and strictly increasing")
		}
		edges[index] = value
	}
	minimum, minOK := jsonFiniteFloat(chosen["min"])
	maximum, maxOK := jsonFiniteFloat(chosen["max"])
	if !minOK || !maxOK || maximum < minimum {
		return nil, nil, 0, 0, errors.New("image service histogram extrema are invalid")
	}
	return counts, edges, minimum, maximum, nil
}

func validateImageServiceHistogramSampling(
	value any,
	sampleCount int,
) (map[string]any, error) {
	sampling, ok := jsonObject(value)
	if !ok {
		return nil, errors.New("image service histogram sampling provenance is missing")
	}
	algorithm, algorithmOK := sampling["algorithm"].(string)
	scope, scopeOK := sampling["scope"].(string)
	strategy, strategyOK := sampling["strategy"].(string)
	provenanceCount, countOK := jsonInt(sampling["sample_count"])
	zSamples, zSamplesOK := jsonNonNegativeIntSlice(sampling["z_samples"])
	if !algorithmOK || strings.TrimSpace(algorithm) == "" ||
		!scopeOK || scope != "volume" ||
		!strategyOK || (strategy != "exact" && strategy != "stratified-z-spatial") ||
		!countOK || provenanceCount != sampleCount ||
		!zSamplesOK || len(zSamples) == 0 {
		return nil, errors.New("image service histogram sampling provenance is invalid")
	}
	return map[string]any{
		"algorithm":    algorithm,
		"scope":        "volume",
		"strategy":     strategy,
		"sample_count": provenanceCount,
		"z_samples":    zSamples,
	}, nil
}

func validateImageServiceHistogramThreshold(
	value any,
	channelIdx int,
	timeIdx int,
	sampleCount int,
	sampling map[string]any,
) (map[string]any, error) {
	threshold, ok := jsonObject(value)
	if !ok {
		return nil, errors.New("image service histogram threshold provenance is missing")
	}
	method, methodOK := threshold["method"].(string)
	domain, domainOK := threshold["domain"].(string)
	foreground, foregroundOK := threshold["foreground"].(string)
	scope, scopeOK := threshold["sample_scope"].(string)
	thresholdValue, valueOK := jsonFiniteFloat(threshold["value"])
	thresholdCount, countOK := jsonInt(threshold["sample_count"])
	thresholdChannel, channelOK := jsonInt(threshold["channel"])
	thresholdTime, timeOK := jsonInt(threshold["t"])
	algorithm, algorithmOK := threshold["sampling_algorithm"].(string)
	zSamples, zSamplesOK := jsonNonNegativeIntSlice(threshold["z_samples"])
	samplingZSamples, _ := jsonNonNegativeIntSlice(sampling["z_samples"])
	samplingAlgorithm, _ := sampling["algorithm"].(string)
	samplingStrategy, _ := sampling["strategy"].(string)
	expectedScope := "volume"
	if samplingStrategy == "stratified-z-spatial" {
		expectedScope = "stratified_z"
	}
	if !methodOK || method != "otsu-256-v1" ||
		!domainOK || domain != "raw" ||
		!foregroundOK || foreground != "above" ||
		!scopeOK || scope != expectedScope ||
		!valueOK || math.IsNaN(thresholdValue) || math.IsInf(thresholdValue, 0) ||
		!countOK || thresholdCount != sampleCount ||
		!channelOK || thresholdChannel != channelIdx ||
		!timeOK || thresholdTime != timeIdx ||
		!algorithmOK || algorithm == "" || algorithm != samplingAlgorithm ||
		!zSamplesOK || !slices.Equal(zSamples, samplingZSamples) {
		return nil, errors.New("image service histogram threshold provenance is invalid")
	}
	return map[string]any{
		"method":             "otsu-256-v1",
		"value":              thresholdValue,
		"domain":             "raw",
		"foreground":         "above",
		"sample_scope":       scope,
		"sample_count":       thresholdCount,
		"z_samples":          zSamples,
		"channel":            thresholdChannel,
		"t":                  thresholdTime,
		"sampling_algorithm": algorithm,
	}, nil
}

func jsonNonNegativeIntSlice(value any) ([]int, bool) {
	raw, ok := value.([]any)
	if !ok {
		if values, typed := value.([]int); typed {
			raw = make([]any, len(values))
			for index, item := range values {
				raw[index] = item
			}
		} else {
			return nil, false
		}
	}
	out := make([]int, len(raw))
	for index, item := range raw {
		parsed, valid := jsonInt(item)
		if !valid || parsed < 0 || index > 0 && parsed <= out[index-1] {
			return nil, false
		}
		out[index] = parsed
	}
	return out, true
}

// jsonInt/jsonFloat coerce decoded JSON numbers to Go numerics. Integer fields
// reject fractions, infinities, and values outside the platform int range.
func jsonInt(v any) (int, bool) {
	switch n := v.(type) {
	case float64:
		if math.IsNaN(n) || math.IsInf(n, 0) || math.Trunc(n) != n {
			return 0, false
		}
		intLimit := math.Ldexp(1, strconv.IntSize-1)
		if n < -intLimit || n >= intLimit {
			return 0, false
		}
		return int(n), true
	case int:
		return n, true
	case json.Number:
		parsed, err := strconv.ParseInt(string(n), 10, 64)
		if err != nil ||
			strconv.IntSize == 32 && (parsed > math.MaxInt32 || parsed < math.MinInt32) {
			return 0, false
		}
		return int(parsed), true
	}
	return 0, false
}

func jsonFloat(v any) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case int:
		return float64(n)
	}
	return 0
}

// injectControlPlaneViewerFields stamps the control-plane-owned fields the image
// service cannot know (resource identity + the V2 service URLs) onto the
// viewer-info object the sidecar produced.
func injectControlPlaneViewerFields(core map[string]any, record resourceRecord) {
	seg := url.PathEscape(record.FileID)
	serviceURLs := map[string]any{
		"preview":       "/v2/uploads/" + seg + "/preview",
		"display":       "/v2/uploads/" + seg + "/display",
		"slice":         "/v2/uploads/" + seg + "/slice",
		"histogram":     "/v2/uploads/" + seg + "/histogram",
		"tile":          "/v2/uploads/" + seg + "/tiles",
		"atlas":         "/v2/uploads/" + seg + "/atlas",
		"scalar_volume": "/v2/uploads/" + seg + "/scalar-volume",
	}
	core["file_id"] = record.FileID
	core["original_name"] = record.OriginalName
	core["service_urls"] = serviceURLs
	if viewer, ok := core["viewer"].(map[string]any); ok {
		viewer["service_urls"] = serviceURLs
	}
	if meta, ok := core["metadata"].(map[string]any); ok {
		meta["content_type"] = record.ContentType
		meta["size_bytes"] = record.SizeBytes
		meta["sha256"] = record.SHA256
	}
	if phys, ok := core["phys"].(map[string]any); ok {
		phys["name"] = record.OriginalName
	}
	sanitizeScalarMaskCapability(core, record)
	injectViewerCalibrationDefaults(core, record)
}

func sanitizeScalarMaskCapability(core map[string]any, record resourceRecord) {
	raw := core["scalar_mask_capability"]
	delete(core, "scalar_mask_capability")
	sourceSHA := strings.TrimSpace(record.SHA256)
	if sourceSHA == "" || !isTIFFUpload(record.OriginalName, record.ContentType) {
		return
	}
	capability, ok := jsonObject(raw)
	if !ok || !hasOnlyJSONKeys(
		capability,
		"version",
		"source_authority",
		"source_format",
		"dtype",
		"threshold_domain",
		"threshold_foreground",
		"slice_delivery",
		"volume_delivery",
		"volume_sampling",
		"channel_selection",
		"time_selection",
		"surfaces",
	) {
		return
	}
	version, versionOK := jsonInt(capability["version"])
	sourceAuthority, authorityOK := capability["source_authority"].(string)
	sourceFormat, formatOK := capability["source_format"].(string)
	dtype, dtypeOK := capability["dtype"].(string)
	metadata, metadataOK := jsonObject(core["metadata"])
	metadataDtype, metadataDtypeOK := metadata["array_dtype"].(string)
	viewer, viewerOK := jsonObject(core["viewer"])
	renderPolicy, renderPolicyOK := viewer["render_policy"].(string)
	if !versionOK || version != 1 ||
		!authorityOK || sourceAuthority != "original" ||
		!formatOK || (sourceFormat != "tiff" && sourceFormat != "ome-tiff") ||
		!dtypeOK || !isExactScalarMaskDtype(dtype) ||
		!metadataOK || !metadataDtypeOK || metadataDtype != dtype ||
		!viewerOK || !renderPolicyOK || renderPolicy != "scalar" ||
		capability["threshold_domain"] != "raw" ||
		capability["threshold_foreground"] != "above" ||
		capability["slice_delivery"] != "thresholded_png" ||
		capability["volume_delivery"] != "raw_scalar" ||
		capability["volume_sampling"] != "nearest" ||
		capability["channel_selection"] != "single" ||
		capability["time_selection"] != "single" {
		return
	}
	rawSurfaces, surfacesOK := jsonStringSlice(capability["surfaces"])
	rawAvailable, availableOK := jsonStringSlice(viewer["available_surfaces"])
	if !surfacesOK || !availableOK {
		return
	}
	expected := make([]string, 0, len(rawAvailable))
	for _, surface := range rawAvailable {
		if surface == "2d" || surface == "mpr" || surface == "volume" {
			expected = append(expected, surface)
		}
	}
	surfaces := rawSurfaces
	requiredSurfaces := []string{"2d", "mpr", "volume"}
	if !slices.Equal(expected, requiredSurfaces) ||
		!slices.Equal(surfaces, requiredSurfaces) {
		return
	}
	sanitized := map[string]any{
		"version":              1,
		"source_authority":     "original",
		"source_format":        sourceFormat,
		"dtype":                dtype,
		"threshold_domain":     "raw",
		"threshold_foreground": "above",
		"slice_delivery":       "thresholded_png",
		"volume_delivery":      "raw_scalar",
		"volume_sampling":      "nearest",
		"channel_selection":    "single",
		"time_selection":       "single",
		"surfaces":             surfaces,
	}
	sanitized["source_sha256"] = sourceSHA
	core["scalar_mask_capability"] = sanitized
}

func isExactScalarMaskDtype(dtype string) bool {
	switch strings.TrimSpace(strings.ToLower(dtype)) {
	case "uint8", "uint16", "int16":
		return true
	default:
		return false
	}
}

func jsonStringSlice(value any) ([]string, bool) {
	switch values := value.(type) {
	case []string:
		return append([]string(nil), values...), true
	case []any:
		out := make([]string, 0, len(values))
		for _, value := range values {
			text, ok := value.(string)
			if !ok {
				return nil, false
			}
			out = append(out, text)
		}
		return out, true
	default:
		return nil, false
	}
}

func jsonObject(value any) (map[string]any, bool) {
	switch object := value.(type) {
	case map[string]any:
		return object, true
	case domain.JSONMap:
		return map[string]any(object), true
	default:
		return nil, false
	}
}

func hasOnlyJSONKeys(object map[string]any, allowed ...string) bool {
	allow := make(map[string]struct{}, len(allowed))
	for _, key := range allowed {
		allow[key] = struct{}{}
	}
	for key := range object {
		if _, ok := allow[key]; !ok {
			return false
		}
	}
	return true
}

// injectViewerCalibrationDefaults applies only the reserved, versioned,
// source-SHA-bound calibration record. Malformed or stale metadata is ignored;
// no camera, projection, clip, density, or navigation state is accepted here.
func injectViewerCalibrationDefaults(core map[string]any, record resourceRecord) {
	if strings.TrimSpace(record.SHA256) == "" || len(record.Metadata) == 0 {
		return
	}
	if _, capable := jsonObject(core["scalar_mask_capability"]); !capable {
		return
	}
	raw, ok := jsonObject(record.Metadata["ultra_viewer_calibration_v1"])
	if !ok || !hasOnlyJSONKeys(raw, "version", "source_sha256", "selections") {
		return
	}
	version, versionOK := jsonInt(raw["version"])
	sourceSHA, shaOK := raw["source_sha256"].(string)
	if !versionOK || version != 1 || !shaOK || sourceSHA == "" || sourceSHA != record.SHA256 {
		return
	}
	selections, selectionsOK := jsonObject(raw["selections"])
	if !selectionsOK || len(selections) == 0 {
		return
	}
	timeCount, channelCount, _, axesOK := sourceViewerAxes(core)
	if !axesOK {
		return
	}
	sanitizedSelections := make(map[string]any, len(selections))
	for key, value := range selections {
		selection, valid := sanitizeViewerCalibrationSelection(
			key,
			value,
			timeCount,
			channelCount,
			sourceSHA,
			capabilityDtype(core),
			false,
		)
		if !valid {
			continue
		}
		sanitizedSelections[key] = selection
	}
	if len(sanitizedSelections) == 0 {
		return
	}
	sanitized := map[string]any{
		"version":       1,
		"source_sha256": sourceSHA,
		"selections":    sanitizedSelections,
	}
	core["viewer_calibrations"] = sanitized

	// Keep the selected default projection rolling-compatible while the frontend
	// migrates to the complete per-selection map.
	defaults, defaultsOK := jsonObject(core["display_defaults"])
	if !defaultsOK {
		return
	}
	channelIndex, _ := jsonInt(defaults["volume_channel"])
	timeIndex, _ := jsonInt(defaults["time_index"])
	current, currentOK := sanitizedSelections[viewerCalibrationSelectionKey(channelIndex, timeIndex)].(map[string]any)
	if !currentOK {
		return
	}
	defaults["scalar_render_mode"] = current["render_mode"]
	defaults["scalar_threshold_method"] = current["threshold_method"]
	defaults["scalar_threshold_value"] = current["threshold_value"]
	defaults["scalar_threshold_foreground"] = "above"
}

func viewerCalibrationSelectionKey(channel, timeIndex int) string {
	return fmt.Sprintf("c%d:t%d", channel, timeIndex)
}

func sanitizeViewerCalibrationSelection(
	key string,
	value any,
	timeCount int,
	channelCount int,
	sourceSHA string,
	dtype string,
	incoming bool,
) (map[string]any, bool) {
	selection, ok := jsonObject(value)
	if !ok || !hasOnlyJSONKeys(
		selection,
		"channel",
		"t",
		"render_mode",
		"threshold_method",
		"threshold_value",
		"threshold_foreground",
		"threshold_provenance",
		"revision",
		"expected_revision",
	) {
		return nil, false
	}
	channel, channelOK := jsonInt(selection["channel"])
	timeIndex, timeOK := jsonInt(selection["t"])
	renderMode, modeOK := selection["render_mode"].(string)
	method, methodOK := selection["threshold_method"].(string)
	foreground, foregroundOK := selection["threshold_foreground"].(string)
	threshold, thresholdOK := jsonFiniteFloat(selection["threshold_value"])
	revisionField := "revision"
	if incoming {
		revisionField = "expected_revision"
	}
	revision, revisionOK := jsonInt(selection[revisionField])
	if !channelOK || channel < 0 || channel >= channelCount ||
		!timeOK || timeIndex < 0 || timeIndex >= timeCount ||
		key != viewerCalibrationSelectionKey(channel, timeIndex) ||
		!modeOK || (renderMode != "auto" && renderMode != "intensity" && renderMode != "mask") ||
		!methodOK || (method != "otsu-256-v1" && method != "manual") ||
		!foregroundOK || foreground != "above" || !thresholdOK ||
		!revisionOK || incoming && revision < 0 || !incoming && revision <= 0 {
		return nil, false
	}
	threshold, thresholdOK = canonicalIntegerMaskThreshold(threshold, dtype)
	if !thresholdOK {
		return nil, false
	}
	provenance, provenanceOK := jsonObject(selection["threshold_provenance"])
	if !provenanceOK || !hasOnlyJSONKeys(
		provenance,
		"method",
		"value",
		"domain",
		"foreground",
		"channel",
		"t",
		"sample_scope",
		"sample_count",
		"sampling_algorithm",
		"sampling_strategy",
		"z_samples",
		"source_sha256",
		"bins",
	) {
		return nil, false
	}
	provenanceMethod, provenanceMethodOK := provenance["method"].(string)
	provenanceValue, provenanceValueOK := jsonFiniteFloat(provenance["value"])
	domain, domainOK := provenance["domain"].(string)
	provenanceForeground, provenanceForegroundOK := provenance["foreground"].(string)
	provenanceChannel, provenanceChannelOK := jsonInt(provenance["channel"])
	provenanceTime, provenanceTimeOK := jsonInt(provenance["t"])
	sampleScope, scopeOK := provenance["sample_scope"].(string)
	sampleCount, sampleCountOK := jsonInt(provenance["sample_count"])
	samplingAlgorithm, samplingAlgorithmOK := provenance["sampling_algorithm"].(string)
	samplingStrategy, samplingStrategyOK := provenance["sampling_strategy"].(string)
	zSamples, zSamplesOK := jsonNonNegativeIntSlice(provenance["z_samples"])
	provenanceSHA, provenanceSHAOK := provenance["source_sha256"].(string)
	provenanceBins, provenanceBinsOK := jsonInt(provenance["bins"])
	provenanceValue, provenanceValueOK = canonicalIntegerMaskThreshold(
		provenanceValue,
		dtype,
	)
	expectedScope := "volume"
	if samplingStrategy == "stratified-z-spatial" {
		expectedScope = "stratified_z"
	}
	if !provenanceMethodOK || provenanceMethod != "otsu-256-v1" ||
		!provenanceValueOK ||
		!domainOK || domain != "raw" ||
		!provenanceForegroundOK || provenanceForeground != "above" ||
		!provenanceChannelOK || provenanceChannel != channel ||
		!provenanceTimeOK || provenanceTime != timeIndex ||
		!scopeOK || sampleScope != expectedScope ||
		!sampleCountOK || sampleCount <= 0 ||
		!samplingAlgorithmOK || strings.TrimSpace(samplingAlgorithm) == "" ||
		!samplingStrategyOK ||
		(samplingStrategy != "exact" && samplingStrategy != "stratified-z-spatial") ||
		!zSamplesOK || len(zSamples) == 0 ||
		!provenanceSHAOK || strings.TrimSpace(provenanceSHA) != sourceSHA ||
		!provenanceBinsOK || provenanceBins < uploadHistogramMinBins ||
		provenanceBins > uploadHistogramMaxBins ||
		method == "otsu-256-v1" && threshold != provenanceValue {
		return nil, false
	}
	storedRevision := revision
	if incoming {
		storedRevision++
	}
	return map[string]any{
		"channel":              channel,
		"t":                    timeIndex,
		"render_mode":          renderMode,
		"threshold_method":     method,
		"threshold_value":      threshold,
		"threshold_foreground": "above",
		"revision":             storedRevision,
		"threshold_provenance": map[string]any{
			"method":             "otsu-256-v1",
			"value":              provenanceValue,
			"domain":             "raw",
			"foreground":         "above",
			"channel":            channel,
			"t":                  timeIndex,
			"sample_scope":       sampleScope,
			"sample_count":       sampleCount,
			"sampling_algorithm": samplingAlgorithm,
			"sampling_strategy":  samplingStrategy,
			"z_samples":          zSamples,
			"source_sha256":      sourceSHA,
			"bins":               provenanceBins,
		},
	}, true
}

func validateViewerCalibrationMetadata(
	value any,
	sourceSHA string,
	timeCount int,
	channelCount int,
	dtype string,
) (map[string]any, map[string]int, error) {
	raw, ok := jsonObject(value)
	if !ok || !hasOnlyJSONKeys(raw, "version", "source_sha256", "selections") {
		return nil, nil, errors.New("viewer calibration metadata is malformed")
	}
	version, versionOK := jsonInt(raw["version"])
	calibrationSHA, shaOK := raw["source_sha256"].(string)
	selections, selectionsOK := jsonObject(raw["selections"])
	if !versionOK || version != 1 ||
		!shaOK || calibrationSHA == "" || calibrationSHA != sourceSHA ||
		!selectionsOK || len(selections) == 0 {
		return nil, nil, errors.New("viewer calibration metadata is invalid for this source")
	}
	sanitizedSelections := make(map[string]any, len(selections))
	expectedRevisions := make(map[string]int, len(selections))
	for key, value := range selections {
		rawSelection, _ := jsonObject(value)
		expectedRevision, revisionOK := jsonInt(rawSelection["expected_revision"])
		selection, valid := sanitizeViewerCalibrationSelection(
			key,
			value,
			timeCount,
			channelCount,
			sourceSHA,
			dtype,
			true,
		)
		if !valid || !revisionOK {
			return nil, nil, fmt.Errorf("viewer calibration selection %q is invalid", key)
		}
		sanitizedSelections[key] = selection
		expectedRevisions[key] = expectedRevision
	}
	return map[string]any{
		"version":       1,
		"source_sha256": calibrationSHA,
		"selections":    sanitizedSelections,
	}, expectedRevisions, nil
}

func capabilityDtype(core map[string]any) string {
	capability, _ := jsonObject(core["scalar_mask_capability"])
	dtype, _ := capability["dtype"].(string)
	return strings.TrimSpace(strings.ToLower(dtype))
}

func canonicalIntegerMaskThreshold(value float64, dtype string) (float64, bool) {
	if !numberIsFinite(value) {
		return 0, false
	}
	var minimum, maximum float64
	switch strings.TrimSpace(strings.ToLower(dtype)) {
	case "uint8":
		minimum, maximum = -1, math.MaxUint8
	case "uint16":
		minimum, maximum = -1, math.MaxUint16
	case "int16":
		minimum, maximum = math.MinInt16-1, math.MaxInt16
	default:
		return 0, false
	}
	return math.Min(maximum, math.Max(minimum, math.Floor(value))), true
}

func jsonFiniteFloat(value any) (float64, bool) {
	var parsed float64
	switch number := value.(type) {
	case float64:
		parsed = number
	case float32:
		parsed = float64(number)
	case int:
		parsed = float64(number)
	case int64:
		parsed = float64(number)
	case json.Number:
		var err error
		parsed, err = number.Float64()
		if err != nil {
			return 0, false
		}
	default:
		return 0, false
	}
	return parsed, !math.IsNaN(parsed) && !math.IsInf(parsed, 0)
}

// mergePyramidTileScheme folds a derived pyramid's tile_scheme into a source's
// viewer-info, flipping the delivery mode to multiscale tiles.
// viewerIsSliceStackVolume reports whether the sidecar viewer-info describes a
// slice_stack (microscopy z-stack) volume, which serves 3D via the texture atlas
// and 2D via per-plane slices rather than a 2D tile pyramid.
func viewerIsSliceStackVolume(core map[string]any) bool {
	viewer, ok := core["viewer"].(map[string]any)
	if !ok {
		return false
	}
	mode, _ := viewer["volume_mode"].(string)
	return mode == "slice_stack"
}

func mergePyramidTileScheme(core, pyramid map[string]any) {
	ts := pyramid["tile_scheme"]
	if ts == nil {
		return
	}
	core["tile_scheme"] = ts
	core["backend_mode"] = "pyramid"
	if viewer, ok := core["viewer"].(map[string]any); ok {
		viewer["tile_scheme"] = ts
		viewer["backend_mode"] = "pyramid"
		viewer["delivery_mode"] = "deferred_multiscale"
		if prep, ok := viewer["asset_preparation"].(map[string]any); ok {
			prep["tile_pyramid"] = "ready"
		}
	}
}

// --- convert-on-upload: auto-derive a tiled pyramid for new image resources ---

// isPyramidTriggerEvent reports whether a catalog event represents new image
// bytes arriving (upload or import), as opposed to a re-catalog/share/rename.
func isPyramidTriggerEvent(eventType string) bool {
	return eventType == "resource.uploaded" || eventType == "resource.imported"
}

// hasPyramidMicroscopyExtension matches non-TIFF microscopy container formats
// that are N-D and benefit from a pre-built tiled pyramid.
func hasPyramidMicroscopyExtension(name string) bool {
	lower := strings.ToLower(strings.TrimSpace(name))
	for _, ext := range []string{
		".czi", ".nd2", ".lsm", ".lif", ".oib", ".oif", ".vsi",
		".scn", ".svs", ".ndpi", ".sld", ".ims", ".zvi", ".ipl",
		".dv", ".r3d", ".mrc", ".sldy", // DeltaVision/MRC + 3i SlideBook (bioio plugins)
	} {
		// Match a clean extension AND a series/page-suffixed export name (some tools
		// write "scan.lif_15" for series 15) so those still derive a pyramid at upload.
		if strings.HasSuffix(lower, ext) || strings.Contains(lower, ext+"_") {
			return true
		}
	}
	return false
}

// prefersBioioReader reports whether a resource's format should be read via bioio (the
// convert worker transcodes it to a pyramid) in preference to libbioimage's native read:
// Proprietary microscopy readers whose native path is absent or semantically
// unreliable. Mirrors the Python
// PREFER_BIOIO_EXTENSIONS default in imaging/job.py. (.zarr is intentionally excluded:
// OME-Zarr directory bundles are served natively by the ngff-service from the store, so
// they never enter the bioio transcode lane — see shouldDerivePyramid / servesViaNgff.)
func prefersBioioReader(name string) bool {
	lower := strings.ToLower(strings.TrimSpace(name))
	for _, ext := range []string{".czi", ".nd2", ".lif", ".dv", ".r3d"} {
		if strings.HasSuffix(lower, ext) || strings.Contains(lower, ext+"_") {
			return true
		}
	}
	return false
}

// shouldDerivePyramid decides whether a resource warrants a derived pyramid.
// Scientific formats (TIFF/OME-TIFF and microscopy containers) always do — they
// are routinely N-D and 50GB-class. Plain raster images only past a size where
// tiling beats a direct read. NIfTI volumes are served via /scalar-volume.
func shouldDerivePyramid(record resourceRecord) bool {
	// Natively-served special formats (OME-Zarr) need no derived pyramid — the
	// ngff-service serves the store directly.
	if sf := detectSpecialFormatByName(record.OriginalName); sf != nil && sf.Serve == "ngff" {
		return false
	}
	if isNiftiUpload(record.OriginalName, record.ContentType) {
		return false
	}
	if isTIFFUpload(record.OriginalName, record.ContentType) || hasPyramidMicroscopyExtension(record.OriginalName) {
		return true
	}
	if strings.HasPrefix(strings.ToLower(strings.TrimSpace(record.ContentType)), "image/") {
		return record.SizeBytes >= pyramidPlainImageMinBytes
	}
	return false
}

// derivationThrottleWindow bounds how often a derive-pyramid job is (re)enqueued
// for the same resource. It must be long enough that a burst of viewer requests
// (or many tile fetches) for an un-derived image enqueues at most once, but short
// enough that a genuinely lost job is retried promptly on continued access.
const derivationThrottleWindow = 2 * time.Minute

// derivationThrottleMaxEntries caps the throttle map before it prunes stale entries,
// so tracking one timestamp per ever-viewed resource cannot grow without bound.
const derivationThrottleMaxEntries = 8192

// derivationThrottle bounds how often a derive-pyramid job is (re)enqueued per
// resource. The serve-time self-heal calls it on every viewer open for an image
// that should have a pyramid but doesn't; without the throttle a burst of opens
// (or a stretch of failing publishes while NATS reconnects) would flood the queue.
// Per-instance and best-effort: at worst one redundant enqueue per window per
// resource after a restart, which the convert worker's atomic build tolerates.
type derivationThrottle struct {
	mu     sync.Mutex
	window time.Duration
	last   map[string]time.Time
}

func newDerivationThrottle(window time.Duration) *derivationThrottle {
	return &derivationThrottle{window: window, last: make(map[string]time.Time)}
}

// reserve reports whether a derivation may be enqueued for fileID at now, recording
// the attempt time when it returns true. now is a parameter for deterministic tests.
func (t *derivationThrottle) reserve(fileID string, now time.Time) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	if at, ok := t.last[fileID]; ok && now.Sub(at) < t.window {
		return false
	}
	if len(t.last) >= derivationThrottleMaxEntries {
		for key, at := range t.last { // opportunistic prune of expired entries
			if now.Sub(at) >= t.window {
				delete(t.last, key)
			}
		}
	}
	t.last[fileID] = now
	return true
}

var pyramidDerivationThrottle = newDerivationThrottle(derivationThrottleWindow)

// maybeEnqueuePyramidDerivation best-effort enqueues an image.derive_pyramid job
// for a freshly uploaded/imported image (the prewarm path). Gated on the upload
// trigger event; the actual durability comes from ensurePyramidDerivation, which
// also self-heals at view time if this enqueue is lost.
func (deps ServerDeps) maybeEnqueuePyramidDerivation(ctx context.Context, root string, record resourceRecord, path, eventType string) {
	if !isPyramidTriggerEvent(eventType) {
		return
	}
	deps.ensurePyramidDerivation(ctx, root, record, path, "upload")
}

// ensurePyramidDerivation guarantees a pyramid-eligible image gets its derived
// tiled pyramid enqueued at least once, healing a publish that was lost (NATS
// reconnecting, transient error) at upload time. It is safe to call on hot serving
// paths: it no-ops when the queue/image service is absent, the resource is not
// pyramid-eligible, a pyramid already exists, or one was enqueued recently. Unlike
// the old fire-and-forget enqueue, a failed publish is logged (an operator can see
// the drop) and left un-throttled-past-the-window so continued viewing retries it.
func (deps ServerDeps) ensurePyramidDerivation(ctx context.Context, root string, record resourceRecord, path, trigger string) {
	if !shouldDerivePyramid(record) {
		return
	}
	deps.enqueuePyramidDerivation(ctx, root, record, path, trigger)
}

// enqueuePyramidDerivation is ensurePyramidDerivation WITHOUT the format-eligibility
// gate: it still no-ops when the queue/image service is absent, a pyramid already
// exists, a recent derivation permanently failed, or one was enqueued recently. The
// undecodable-source viewer path uses it to request a bioio transcode->pyramid for a
// format libbioimage can't read (which shouldDerivePyramid's extension allowlist may
// not list — e.g. a series-suffixed ".lif_15"), since the engine has already proven it
// recognized-but-couldn't-decode the image.
func (deps ServerDeps) enqueuePyramidDerivation(ctx context.Context, root string, record resourceRecord, path, trigger string) {
	if deps.DataAgentJobs == nil || !deps.imageServiceConfigured() {
		return
	}
	if !lowercaseSHA256Pattern.MatchString(strings.ToLower(strings.TrimSpace(record.SHA256))) || record.SizeBytes < 0 {
		slog.WarnContext(ctx, "pyramid derivation skipped: resource has no immutable source identity",
			"resource_id", record.FileID, "trigger", trigger)
		return
	}
	if recentPyramidFailure(root, record, time.Now()) {
		// A recent derivation PERMANENTLY failed (a source this engine build can't
		// convert). Back off instead of re-minting a doomed convert on every viewer
		// open — that repeated heavy imgcnv work is exactly what starved the engine
		// in the 707k-redelivery incident. The source still serves via direct/slice;
		// an explicit re-derive (handleDeriveUploadPyramid) clears the marker to retry.
		return
	}
	if !pyramidDerivationThrottle.reserve(record.FileID, time.Now()) {
		return // enqueued recently; assume a convert is already in flight
	}
	dst := filepath.Join(root, "derived", derivedPyramidName(record.FileID))
	if err := deps.DataAgentJobs.PublishDataAgentJob(ctx, eventbus.DataAgentJob{
		JobID:         domain.NewID("imgjob"),
		OwnerUserID:   record.Principal.UserID,
		OwnerOrgID:    record.Principal.OrgID,
		JobType:       "image.derive_pyramid",
		ResourceIDs:   []string{record.FileID},
		ResourceCount: 1,
		Metadata: domain.JSONMap{
			"resource_id":       record.FileID,
			"src_path":          path,
			"dst_path":          dst,
			"source_sha256":     strings.ToLower(strings.TrimSpace(record.SHA256)),
			"source_size_bytes": record.SizeBytes,
			"tile_size":         512,
			"compression":       "lzw",
			"layout":            "topdirs", // native level stays tile-addressable (see handleDeriveUploadPyramid)
			"fmt":               "auto",    // z-stack/volume -> OME-BigTIFF (preserves Z); flat 2D -> BigTIFF
			"trigger":           trigger,
		},
	}); err != nil {
		// Visible, not silent: the image stays viewable via the direct/slice fallback
		// meanwhile, and a continued view re-attempts after the throttle window.
		slog.WarnContext(ctx, "pyramid derivation enqueue failed; will retry on next view",
			"resource_id", record.FileID, "trigger", trigger, "error", err)
	}
}
