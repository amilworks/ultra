package httpapi

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
)

// pyramidPlainImageMinBytes is the size past which a plain raster image (JPEG/PNG)
// is worth converting into a tiled pyramid. Below it, the direct path is faster.
const pyramidPlainImageMinBytes = 16 << 20 // 16 MiB

// This file routes the viewer-facing image endpoints (/viewer, /slice,
// /scalar-volume) through the libbioimage image service when it is configured.
// Every handler degrades gracefully: NIfTI keeps its dedicated medical path,
// and any image-service error (or a missing sidecar) falls back to the legacy
// native Go path, so enabling the sidecar never regresses existing behavior.

// derivedPyramidName is the deterministic filename the convert job writes for a
// resource's tiled pyramid (see handleDeriveUploadPyramid). A plain BigTIFF (not
// OME-TIFF) so every level — including the native level 0 — is tile-addressable,
// and so pyramids of 50GB-class sources can exceed the 4GB classic-TIFF limit.
func derivedPyramidName(fileID string) string { return fileID + "__pyramid.tif" }

// derivedPyramidPath returns the on-disk path of a resource's derived pyramid if
// one has been built (non-empty file), else "". Callers prefer it for tiles.
func derivedPyramidPath(root, fileID string) string {
	p := filepath.Join(root, "derived", derivedPyramidName(fileID))
	if fi, err := os.Stat(p); err == nil && !fi.IsDir() && fi.Size() > 0 {
		return p
	}
	return ""
}

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
func recentPyramidFailure(root, fileID string, now time.Time) bool {
	window := pyramidFailureBackoffWindow()
	if window <= 0 {
		return false
	}
	fi, err := os.Stat(derivedPyramidFailedMarkerPath(root, fileID))
	if err != nil || fi.IsDir() {
		return false
	}
	return now.Sub(fi.ModTime()) < window
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
	query := url.Values{"path": {path}}
	cache := deps.imageCache
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
	out, err := deps.imageServiceGetJSON(ctx, "/viewerinfo", query)
	if err != nil {
		return nil, err
	}
	if body, mErr := json.Marshal(out); mErr == nil {
		cache.put(key, &cachedResponse{status: http.StatusOK, contentType: "application/json", body: body}, int64(len(body)))
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
		deps.writeNiftiUploadViewer(w, record, path)
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
	core, err := deps.cachedImageServiceViewerInfo(r.Context(), path)
	if err != nil {
		// The engine recognized the file but cannot decode it (415/422): a permanent,
		// format-level failure (e.g. a Leica .lif — registered but non-functional in
		// this libbioimage build). The legacy native Go viewer supports only a small
		// raster subset, so probe whether it can produce a real plane; if not, surface
		// an explicit "unsupported" descriptor so the viewer shows a clear message +
		// download instead of a broken 1x1 canvas with an endless spinner.
		if imageServiceUndecodable(err) {
			// A bioio transcode->pyramid may already exist for this source (the convert
			// worker reads LIF/etc. via bioio and writes an OME-TIFF pyramid libbioimage
			// CAN serve). The source itself is undecodable, so drive the viewer off the
			// derived pyramid's metadata.
			if dp := derivedPyramidPath(root, record.FileID); dp != "" {
				if pcore, perr := deps.cachedImageServiceViewerInfo(r.Context(), dp); perr == nil {
					injectControlPlaneViewerFields(pcore, record)
					writeJSON(w, http.StatusOK, pcore)
					return
				}
			}
			// No pyramid yet. The legacy native Go viewer reads only a small raster
			// subset; if it can produce a real plane, use it.
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
	// .czi / .zarr are preferentially read by bioio (the convert worker transcodes them to
	// an OME-TIFF pyramid). libbioimage CAN decode a .czi, but renders Zeiss mosaics
	// blocky/unstitched, so when the bioio pyramid exists drive the viewer entirely off it
	// — its geometry/channels then match the pixels /slice and /thumbnail serve from that
	// same pyramid. If it isn't converted yet, kick off the (preferred) derivation.
	if prefersBioioReader(record.OriginalName) {
		if dp := derivedPyramidPath(root, record.FileID); dp != "" {
			if pcore, perr := deps.cachedImageServiceViewerInfo(r.Context(), dp); perr == nil {
				injectControlPlaneViewerFields(pcore, record)
				writeJSON(w, http.StatusOK, pcore)
				return
			}
		} else {
			deps.enqueuePyramidDerivation(r.Context(), root, record, path, "prefer-bioio")
		}
	}
	// A slice_stack volume (microscopy z-stack) derives to an OME-BigTIFF whose
	// embedded -tile reader is broken (the OME wrapper); it serves 3D via /atlas
	// and 2D via /slice, so it must NOT advertise the derived pyramid's tile_scheme
	// (that would route the viewer to the failing deferred-multiscale tile path).
	if !viewerIsSliceStackVolume(core) {
		if dp := derivedPyramidPath(root, record.FileID); dp != "" {
			// The derived pyramid serves the tile PIXELS (resolveUploadTilePathForImageService),
			// so the viewer must use the PYRAMID's tile_scheme — its tile size and level grid —
			// even when the source advertised its own. Otherwise the viewer fetches at the
			// source geometry (e.g. 256-px / 8 levels) while pixels come from the pyramid
			// (512-px / 11 levels): every pyramid tile is decoded 4x and the engine is
			// needlessly overloaded on deep zoom. Overriding here aligns grid with data.
			if pyramid, perr := deps.cachedImageServiceViewerInfo(r.Context(), dp); perr == nil {
				mergePyramidTileScheme(core, pyramid)
			}
		} else if core["tile_scheme"] == nil {
			// No derived pyramid and the source is not directly tile-servable: kick off
			// derivation so a later open gets the bounded DeepZoom path; the direct/slice
			// path still works meanwhile.
			deps.ensurePyramidDerivation(r.Context(), root, record, path, "view")
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
	maskRequest, maskErr := parseMaskSliceRequest(r)
	if maskErr != nil {
		writeError(w, http.StatusBadRequest, maskErr)
		return
	}
	if !deps.imageServiceConfigured() {
		if maskRequest.enabled {
			deps.handleNotConfigured("mask slices require the configured source image service")(w, r)
			return
		}
		deps.handleServeUpload(w, r)
		return
	}
	root, record, path, ok := deps.resolveUploadServingRequest(w, r)
	if !ok {
		return
	}
	if isNiftiUpload(record.OriginalName, record.ContentType) {
		if maskRequest.enabled {
			writeError(w, http.StatusUnprocessableEntity, errors.New("mask slices are unsupported for NIfTI sources"))
			return
		}
		deps.handleServeUpload(w, r) // serveNiftiSliceAsPNG honors slice params
		return
	}
	// OME-Zarr is rendered natively by the ngff-service from the store (bundle dir path).
	if deps.servesViaNgff(record, path) {
		if maskRequest.enabled {
			writeError(w, http.StatusUnprocessableEntity, errors.New("mask slices are unsupported for NGFF sources"))
			return
		}
		q := url.Values{"path": {path}}
		for _, key := range []string{"z", "t", "level", "channels", "full_resolution"} {
			if v := strings.TrimSpace(r.URL.Query().Get(key)); v != "" {
				q.Set(key, v)
			}
		}
		deps.ngffDeps().proxyImageServiceCached(w, r, "/slice", q)
		return
	}
	if deps.ngffServiceUnavailable(record, path) {
		writeError(w, http.StatusServiceUnavailable, errNgffServiceNotConfigured)
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
		if dp := derivedPyramidPath(root, record.FileID); dp != "" {
			servePath = dp
		}
	}
	// channels/colors enable additive multi-channel LUT compositing for fluorescence
	// microscopy (libbioimage fuses the selected channels). full_resolution=false serves
	// a bounded pyramid level for fast scrub frames; true reads the native plane.
	buildSliceQuery := func(p string) url.Values {
		q := url.Values{"path": {p}}
		for _, key := range []string{
			"z",
			"level",
			"channel_colors",
			"full_resolution",
		} {
			if v := strings.TrimSpace(r.URL.Query().Get(key)); v != "" {
				q.Set(key, v)
			}
		}
		if maskRequest.enabled {
			q.Set("channels", strconv.Itoa(maskRequest.channel))
			q.Set("t", strconv.Itoa(maskRequest.time))
			q.Set("scalar_render_mode", "mask")
			q.Set("scalar_threshold_value", maskRequest.thresholdRaw)
			q.Set("scalar_threshold_foreground", "above")
		} else {
			for _, key := range []string{"t", "channels"} {
				if v := strings.TrimSpace(r.URL.Query().Get(key)); v != "" {
					q.Set(key, v)
				}
			}
		}
		return q
	}
	// Robustness: prefer the pyramid, but if it can't be served (a broken/unreadable
	// derived pyramid -> 5xx) retry the SOURCE via the image service — slower but
	// correct (honors z/channels) — then the native Go path. A bad pyramid degrades to
	// a working read instead of "Failed to load image".
	var fallback http.HandlerFunc
	if !maskMode {
		fallback = deps.handleServeUpload
	}
	if servePath != path {
		fallback = func(w http.ResponseWriter, r *http.Request) {
			deps.proxyImageServiceSliceCached(w, r, "/slice", buildSliceQuery(path), deps.handleServeUpload)
		}
	}
	// Route slices through the dedicated slice cache so a z-scrub burst can't evict
	// the DeepZoom viewer's tile/atlas working set from the main image cache.
	deps.proxyImageServiceSliceCached(w, r, "/slice", buildSliceQuery(servePath), fallback)
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
	_, record, path, ok := deps.resolveUploadServingRequest(w, r)
	if !ok {
		return
	}
	if isNiftiUpload(record.OriginalName, record.ContentType) {
		deps.handleGetUploadScalarVolume(w, r)
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
	sampling := strings.TrimSpace(r.URL.Query().Get("sampling"))
	if sampling == "" {
		sampling = "box"
	}
	if sampling != "box" && sampling != "nearest" {
		writeError(w, http.StatusBadRequest, errors.New("scalar volume sampling must be box or nearest"))
		return
	}
	query.Set("sampling", sampling)
	deps.proxyImageService(w, r, "/scalar-volume", query, deps.handleGetUploadScalarVolume)
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

// handleServeResourceThumbnail produces a grid thumbnail for any supported format.
// Common web images and NIfTI keep the fast native path; scientific containers the
// native decoder can't read get a libbioimage thumbnail (preferring the derived
// pyramid for a bounded read). Video has no server-side still — the client renders
// a <video> poster — so we 415 quickly rather than stream the whole file.
func (deps ServerDeps) handleServeResourceThumbnail(w http.ResponseWriter, r *http.Request) {
	if !deps.imageServiceConfigured() {
		deps.handleServeUpload(w, r)
		return
	}
	root, record, path, ok := deps.resolveUploadServingRequest(w, r)
	if !ok {
		return
	}
	if isVideoUpload(record.OriginalName, record.ContentType) {
		// Server-side ffmpeg poster frame (cacheable PNG). If the sidecar can't
		// decode the video it streams a 415, and the client falls back to its own
		// <video> poster on the <img> error.
		deps.proxyImageService(w, r, "/video-poster", url.Values{"path": {path}, "max_size": {"512"}})
		return
	}
	if isNiftiUpload(record.OriginalName, record.ContentType) || goNativeThumbnailable(record) {
		deps.handleServeUpload(w, r)
		return
	}
	// OME-Zarr thumbnails come from the ngff-service (smallest multiscale level).
	if deps.servesViaNgff(record, path) {
		deps.ngffDeps().proxyImageServiceCached(w, r, "/thumbnail", url.Values{"path": {path}, "max_size": {"512"}})
		return
	}
	if deps.ngffServiceUnavailable(record, path) {
		writeError(w, http.StatusServiceUnavailable, errNgffServiceNotConfigured)
		return
	}
	servePath := path
	if dp := derivedPyramidPath(root, record.FileID); dp != "" {
		servePath = dp // bounded read from a low pyramid level
	}
	// Same robustness as /slice, but only when a pyramid is actually in use: if the
	// pyramid read stalls/fails (unreachable or the client-timeout the NFS hang trips),
	// retry the source thumbnail via the image service, then the native Go path. The
	// no-pyramid path keeps its original behavior (a clean upstream error).
	if servePath != path {
		deps.proxyImageServiceCached(w, r, "/thumbnail", url.Values{"path": {servePath}, "max_size": {"512"}}, func(w http.ResponseWriter, r *http.Request) {
			deps.proxyImageServiceCached(w, r, "/thumbnail", url.Values{"path": {path}, "max_size": {"512"}}, deps.handleServeUpload)
		})
		return
	}
	deps.proxyImageServiceCached(w, r, "/thumbnail", url.Values{"path": {servePath}, "max_size": {"512"}})
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
// Zeiss .czi, where libbioimage renders mosaics blocky/unstitched. Mirrors the Python
// PREFER_BIOIO_EXTENSIONS default in imaging/job.py. (.zarr is intentionally excluded:
// OME-Zarr directory bundles are served natively by the ngff-service from the store, so
// they never enter the bioio transcode lane — see shouldDerivePyramid / servesViaNgff.)
func prefersBioioReader(name string) bool {
	lower := strings.ToLower(strings.TrimSpace(name))
	for _, ext := range []string{".czi"} {
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
	if derivedPyramidPath(root, record.FileID) != "" {
		return // already derived
	}
	if recentPyramidFailure(root, record.FileID, time.Now()) {
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
			"resource_id": record.FileID,
			"src_path":    path,
			"dst_path":    dst,
			"tile_size":   512,
			"compression": "lzw",
			"layout":      "topdirs", // native level stays tile-addressable (see handleDeriveUploadPyramid)
			"fmt":         "auto",    // z-stack/volume -> OME-BigTIFF (preserves Z); flat 2D -> BigTIFF
			"trigger":     trigger,
		},
	}); err != nil {
		// Visible, not silent: the image stays viewable via the direct/slice fallback
		// meanwhile, and a continued view re-attempts after the throttle window.
		slog.WarnContext(ctx, "pyramid derivation enqueue failed; will retry on next view",
			"resource_id", record.FileID, "trigger", trigger, "error", err)
	}
}
