package httpapi

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
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
		return nil, fmt.Errorf("image service %s -> %d: %s", endpoint, resp.StatusCode, strings.TrimSpace(string(body)))
	}
	var out map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return out, nil
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
	core, err := deps.imageServiceGetJSON(r.Context(), "/viewerinfo", url.Values{"path": {path}})
	if err != nil {
		// Graceful fallback to the legacy native viewer on any sidecar error.
		deps.handleGetUploadViewer(w, r)
		return
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
			if pyramid, perr := deps.imageServiceGetJSON(r.Context(), "/viewerinfo", url.Values{"path": {dp}}); perr == nil {
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

// handleServeUploadSliceService backs /slice with a real libbioimage plane read
// (honoring z/t/level), which is what makes z-scrub scrub actual planes. NIfTI
// and the unconfigured case keep the legacy behavior.
func (deps ServerDeps) handleServeUploadSliceService(w http.ResponseWriter, r *http.Request) {
	if !deps.imageServiceConfigured() {
		deps.handleServeUpload(w, r)
		return
	}
	root, record, path, ok := deps.resolveUploadServingRequest(w, r)
	if !ok {
		return
	}
	if isNiftiUpload(record.OriginalName, record.ContentType) {
		deps.handleServeUpload(w, r) // serveNiftiSliceAsPNG honors slice params
		return
	}
	// Prefer the derived tiled pyramid when one exists: its native level 0 is
	// pixel-identical to the source but a bounded (tiled) read, so a z-scrub plane
	// decodes ~8x faster than re-decoding a full plane from a non-pyramidal source
	// (1.9s -> 0.23s on the 575MB OME-TIFF). -slice works on the derived OME-BigTIFF
	// even though its -tile reader does not (atlas/thumbnail read it the same way).
	servePath := path
	if dp := derivedPyramidPath(root, record.FileID); dp != "" {
		servePath = dp
	}
	query := url.Values{"path": {servePath}}
	// channels/colors enable additive multi-channel LUT compositing for
	// fluorescence microscopy (libbioimage fuses the selected channels).
	// full_resolution=false lets the engine serve a bounded pyramid level for fast
	// scrub frames; full_resolution=true (settled/measurement view) reads native.
	for _, key := range []string{"z", "t", "level", "channels", "channel_colors", "full_resolution"} {
		if v := strings.TrimSpace(r.URL.Query().Get(key)); v != "" {
			query.Set(key, v)
		}
	}
	deps.proxyImageServiceCached(w, r, "/slice", query)
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
	query := url.Values{
		"path":    {path},
		"channel": {strconv.Itoa(parseUploadScalarChannelIndex(r))},
	}
	if t := strings.TrimSpace(r.URL.Query().Get("t")); t != "" {
		query.Set("t", t)
	}
	deps.proxyImageService(w, r, "/scalar-volume", query)
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
	servePath := path
	if dp := derivedPyramidPath(root, record.FileID); dp != "" {
		servePath = dp // bounded read from a low pyramid level
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

// handleGetUploadHistogramService extends /histogram to microscopy formats via
// libbioimage while keeping standard rasters and NIfTI on the proven native
// path. The image service's per-channel shape is mapped to the frontend's
// {histogram:{bins,min,max}} contract.
func (deps ServerDeps) handleGetUploadHistogramService(w http.ResponseWriter, r *http.Request) {
	if !deps.imageServiceConfigured() {
		deps.handleGetUploadHistogram(w, r)
		return
	}
	root, record, path, ok := deps.resolveUploadServingRequest(w, r)
	if !ok {
		return
	}
	histogramPath := path
	if dp := derivedPyramidPath(root, record.FileID); dp != "" {
		histogramPath = dp
	} else if isNiftiUpload(record.OriginalName, record.ContentType) || goCanDecodeHistogram(record) {
		deps.handleGetUploadHistogram(w, r)
		return
	}
	bins := parseUploadHistogramBins(r)
	channelIdx := parseUploadScalarChannelIndex(r)
	core, err := deps.imageServiceGetJSON(r.Context(), "/histogram", url.Values{
		"path": {histogramPath}, "bins": {strconv.Itoa(bins)},
	})
	if err != nil {
		deps.handleGetUploadHistogram(w, r) // preserve prior (likely 415) behavior
		return
	}
	writeJSON(w, http.StatusOK, mapImageServiceHistogram(core, record.FileID, channelIdx, bins))
}

// mapImageServiceHistogram folds libbioimage's per-channel histogram into the
// single-array shape the upload viewer renders, selecting the requested channel.
func mapImageServiceHistogram(core map[string]any, fileID string, channelIdx, bins int) map[string]any {
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
	if chosen == nil && len(channels) > 0 {
		chosen, _ = channels[0].(map[string]any)
	}
	counts := []int{}
	var minV, maxV float64
	if chosen != nil {
		if raw, ok := chosen["counts"].([]any); ok {
			counts = make([]int, 0, len(raw))
			for _, v := range raw {
				counts = append(counts, int(jsonFloat(v)))
			}
		}
		minV = jsonFloat(chosen["min"])
		maxV = jsonFloat(chosen["max"])
	}
	sampleCount := 0
	for _, n := range counts {
		sampleCount += n
	}
	return map[string]any{
		"file_id":      fileID,
		"bins":         bins,
		"dtype":        "uint16",
		"channels":     []int{channelIdx},
		"source":       "libbioimage",
		"sample_count": sampleCount,
		"histogram": map[string]any{
			"bins":            counts,
			"edges":           []float64{},
			"min":             minV,
			"max":             maxV,
			"channel_indices": []int{channelIdx},
			"time_index":      0,
		},
	}
}

// jsonInt/jsonFloat coerce decoded JSON numbers (always float64) to Go numerics.
func jsonInt(v any) (int, bool) {
	switch n := v.(type) {
	case float64:
		return int(n), true
	case int:
		return n, true
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
		".scn", ".svs", ".ndpi", ".sld", ".ims", ".zvi", ".ipl", ".dv",
	} {
		if strings.HasSuffix(lower, ext) {
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
	if deps.DataAgentJobs == nil || !deps.imageServiceConfigured() {
		return
	}
	if !shouldDerivePyramid(record) {
		return
	}
	if derivedPyramidPath(root, record.FileID) != "" {
		return // already derived
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
