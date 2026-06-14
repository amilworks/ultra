package httpapi

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
)

// writeImageServiceUpstreamError replies to a non-2xx image service response with a
// clean, bounded JSON error instead of forwarding the raw upstream body. That body
// can carry the internal storage path (passed as ?path=) or a sidecar traceback —
// leaking it to the client is both an info leak and an opaque error. The upstream
// status is preserved (the viewer's tile/slice retry + fallback logic keys on it),
// and the upstream detail is logged once for operators.
func writeImageServiceUpstreamError(ctx context.Context, w http.ResponseWriter, endpoint string, status int, detail []byte) {
	slog.WarnContext(ctx, "image service returned a non-success status",
		"endpoint", endpoint, "status", status, "detail", strings.TrimSpace(string(detail)))
	writeError(w, status, fmt.Errorf("image service could not process this request (%d)", status))
}

// imageServiceHTTPClient proxies tile/atlas reads to the libbioimage image service.
// It is tuned for a deep-zoom tile BURST against a possibly-REMOTE node:
//   - a warm keep-alive pool (MaxIdleConnsPerHost) so each viewport's tiles reuse
//     connections instead of paying a TCP/TLS handshake per tile (fast comms);
//   - a hard MaxConnsPerHost ceiling so the control plane can never flood the image
//     service — excess tile requests block on a free connection (backpressure) rather
//     than piling decode work onto the engine. Sized by ULTRA_CONTROL_IMAGE_SERVICE_MAX_CONNS.
var imageServiceHTTPClient = newImageServiceHTTPClient()

func newImageServiceHTTPClient() *http.Client {
	maxConns := 32
	if raw := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_IMAGE_SERVICE_MAX_CONNS")); raw != "" {
		if v, err := strconv.Atoi(raw); err == nil && v > 0 {
			maxConns = v
		}
	}
	return &http.Client{
		Timeout: 60 * time.Second,
		Transport: &http.Transport{
			Proxy:               http.ProxyFromEnvironment,
			DialContext:         (&net.Dialer{Timeout: 5 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
			MaxIdleConns:        maxConns * 2,
			MaxIdleConnsPerHost: maxConns,
			MaxConnsPerHost:     maxConns,
			IdleConnTimeout:     90 * time.Second,
			TLSHandshakeTimeout: 5 * time.Second,
		},
	}
}

// imageServiceConfigured reports whether an image-service base URL is set. When
// it is not, the tile/atlas endpoints report "not configured" exactly as before,
// preserving current behavior until the sidecar is deployed.
func (deps ServerDeps) imageServiceConfigured() bool {
	return strings.TrimSpace(deps.ImageServiceURL) != ""
}

// handleServeUploadTiles proxies a single pyramid tile to the image service.
// Auth and ownership are enforced here (identically to every other upload
// handler) before the internal sidecar is reached.
func (deps ServerDeps) handleServeUploadTiles(w http.ResponseWriter, r *http.Request) {
	if !deps.imageServiceConfigured() {
		deps.handleNotConfigured("upload tile pyramid delivery requires the image service (set ULTRA_CONTROL_IMAGE_SERVICE_URL)")(w, r)
		return
	}
	path, ok := deps.resolveUploadTilePathForImageService(w, r)
	if !ok {
		return
	}
	query := url.Values{}
	query.Set("path", path)
	query.Set("level", chi.URLParam(r, "level"))
	query.Set("col", chi.URLParam(r, "tile_x"))
	query.Set("row", chi.URLParam(r, "tile_y"))
	if axis := chi.URLParam(r, "axis"); axis != "" {
		query.Set("axis", axis)
	}
	if size := strings.TrimSpace(r.URL.Query().Get("size")); size != "" {
		query.Set("size", size)
	}
	// Multi-channel fluorescence compositing: forward selected channels + LUT
	// colors so the image service fuses them additively.
	for _, key := range []string{"channels", "channel_colors"} {
		if v := strings.TrimSpace(r.URL.Query().Get(key)); v != "" {
			query.Set(key, v)
		}
	}
	deps.proxyImageServiceCached(w, r, "/tile", query)
}

// handleServeUploadAtlas proxies a z-stack texture atlas to the image service.
func (deps ServerDeps) handleServeUploadAtlas(w http.ResponseWriter, r *http.Request) {
	if !deps.imageServiceConfigured() {
		deps.handleNotConfigured("upload atlas delivery requires the image service (set ULTRA_CONTROL_IMAGE_SERVICE_URL)")(w, r)
		return
	}
	path, ok := deps.resolveUploadTilePathForImageService(w, r)
	if !ok {
		return
	}
	query := url.Values{}
	query.Set("path", path)
	// channels/channel_colors composite a multi-channel z-stack into an RGB atlas
	// for the 3D fluorescence volume render.
	for _, key := range []string{"level", "grid_rows", "grid_cols", "channels", "channel_colors"} {
		if value := strings.TrimSpace(r.URL.Query().Get(key)); value != "" {
			query.Set(key, value)
		}
	}
	deps.proxyImageServiceCached(w, r, "/atlas", query)
}

// resolveUploadTilePathForImageService resolves the path tiles/atlas should be
// served from: the derived tiled pyramid when one exists (bounded reads from a
// 50GB-class source), else the source itself (natively pyramidal formats already
// tile fine). Auth/ownership is enforced identically to every upload handler.
// resolveUploadServingRequest resolves the upload root, the authorized resource record,
// and its source storage path for a viewer serving handler. On failure it writes the
// appropriate error (500 for an unresolved root, the store error otherwise) and returns
// ok=false. It consolidates the identical preamble shared by every viewer serving handler
// (viewer / slice / scalar-volume / thumbnail / histogram / tile).
func (deps ServerDeps) resolveUploadServingRequest(w http.ResponseWriter, r *http.Request) (root string, record resourceRecord, path string, ok bool) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return "", resourceRecord{}, "", false
	}
	record, path, err = deps.findUploadResourceForRequest(r.Context(), root, deps.principalFromRequest(r, ""), chi.URLParam(r, "file_id"))
	if err != nil {
		writeStoreError(w, err)
		return "", resourceRecord{}, "", false
	}
	return root, record, path, true
}

func (deps ServerDeps) resolveUploadTilePathForImageService(w http.ResponseWriter, r *http.Request) (string, bool) {
	root, record, path, ok := deps.resolveUploadServingRequest(w, r)
	if !ok {
		return "", false
	}
	if dp := derivedPyramidPath(root, record.FileID); dp != "" {
		return dp, true
	}
	return path, true
}

// proxyImageService forwards a GET to the internal image service and streams the
// response back. The control plane has already authorized the request and
// resolved the storage path; the sidecar is never reached directly.
func (deps ServerDeps) proxyImageService(w http.ResponseWriter, r *http.Request, endpoint string, query url.Values) {
	base := strings.TrimRight(strings.TrimSpace(deps.ImageServiceURL), "/")
	target := base + endpoint
	if encoded := query.Encode(); encoded != "" {
		target += "?" + encoded
	}
	req, err := http.NewRequestWithContext(r.Context(), http.MethodGet, target, nil)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	resp, err := imageServiceHTTPClient.Do(req)
	if err != nil {
		writeError(w, http.StatusBadGateway, fmt.Errorf("image service request failed: %w", err))
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		detail, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		writeImageServiceUpstreamError(r.Context(), w, endpoint, resp.StatusCode, detail)
		return
	}
	if contentType := resp.Header.Get("Content-Type"); contentType != "" {
		w.Header().Set("Content-Type", contentType)
	}
	// Forward sidecar metadata headers (e.g. scalar-volume geometry) verbatim so
	// the frontend can size the WebGL volume without a second request.
	for key, values := range resp.Header {
		if lk := strings.ToLower(key); strings.HasPrefix(lk, "x-volume-") || strings.HasPrefix(lk, "x-image-") {
			for _, v := range values {
				w.Header().Add(key, v)
			}
		}
	}
	if resp.StatusCode == http.StatusOK {
		w.Header().Set("Cache-Control", "private, max-age=3600")
	}
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}

// handleDeriveUploadPyramid enqueues an `image.derive_pyramid` batch job: convert
// this upload's source into a tiled pyramid (consumed by the convert worker,
// which runs libbioimage). The job rides the existing Data Agent job queue. Auth
// is enforced via the same upload resolution as the serving handlers.
func (deps ServerDeps) handleDeriveUploadPyramid(w http.ResponseWriter, r *http.Request) {
	if deps.DataAgentJobs == nil {
		deps.handleNotConfigured("image pyramid conversion requires the job queue (set up NATS)")(w, r)
		return
	}
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	principal := deps.principalFromRequest(r, "")
	fileID := chi.URLParam(r, "file_id")
	_, path, err := deps.findUploadResourceForRequest(r.Context(), root, principal, fileID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	// Explicit re-derive: clear any prior permanent-failure marker so this attempt is
	// not suppressed by the backoff (the operator/caller escape hatch to retry).
	clearPyramidFailureMarker(root, fileID)
	jobID := domain.NewID("imgjob")
	dst := filepath.Join(root, "derived", derivedPyramidName(fileID))
	envelope := eventbus.DataAgentJob{
		JobID:         jobID,
		OwnerUserID:   principal.UserID,
		OwnerOrgID:    principal.OrgID,
		JobType:       "image.derive_pyramid",
		ResourceIDs:   []string{fileID},
		ResourceCount: 1,
		Metadata: domain.JSONMap{
			"resource_id": fileID,
			"src_path":    path,
			"dst_path":    dst,
			"tile_size":   512,
			"compression": "lzw",
			// topdirs (each level a full page) keeps the native level tile-addressable.
			// fmt="auto": the worker picks the container from the source's
			// dimensionality — a z-stack/volume derives to OME-BigTIFF (the OME wrapper
			// preserves the Z planes, which a plain BigTIFF would flatten) for /slice +
			// /atlas reads; a flat 2D slide stays BigTIFF, because the OME wrapper breaks
			// this engine's embedded -tile reader and 2D serving is tile-based.
			"layout": "topdirs",
			"fmt":    "auto",
		},
	}
	if err := deps.DataAgentJobs.PublishDataAgentJob(r.Context(), envelope); err != nil {
		writeError(w, http.StatusBadGateway, fmt.Errorf("failed to enqueue pyramid conversion: %w", err))
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]any{
		"job_id":       jobID,
		"job_type":     "image.derive_pyramid",
		"resource_id":  fileID,
		"derived_path": dst,
		"status":       "queued",
	})
}
