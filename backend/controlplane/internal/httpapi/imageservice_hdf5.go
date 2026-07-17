package httpapi

import (
	"net/http"
	"net/url"
	"strings"
)

// HDF5 viewer proxy routes (/v2/uploads/{file_id}/hdf5/*).
//
// The Python image service owns all h5py work (tree walk, dataset summaries,
// slice/atlas/scalar-volume rendering); the control plane
// is a thin authorized proxy, exactly like the sibling upload image endpoints:
// auth + ownership are enforced here (resolveUploadServingRequest -> 404 for
// foreign/unknown files, before the sidecar is ever reached), the resolved
// storage path is forwarded as ?path=, and only an explicit per-endpoint
// allowlist of client query params rides along. HDF5 serves from the SOURCE
// file — no derived-pyramid indirection, no ngff branch — and has no legacy
// native fallback, so an unreachable sidecar degrades to a 502 and upstream
// 4xx statuses (unknown dataset_path, not-an-HDF5-file) pass through with a
// clean bounded JSON error body.

// hdf5QueryKeys lists, per image-service endpoint, the client query params the
// proxy forwards. Everything else is dropped (house convention: never forward
// r.URL.RawQuery wholesale).
var hdf5QueryKeys = map[string][]string{
	"/hdf5/dataset":               {"dataset_path"},
	"/hdf5/preview/slice":         {"dataset_path", "axis", "index", "component"},
	"/hdf5/preview/atlas":         {"dataset_path", "enhancement", "fusion_method", "negative", "channels"},
	"/hdf5/preview/scalar-volume": {"dataset_path", "channel"},
	"/hdf5/preview/histogram":     {"dataset_path", "component", "bins"},
	"/hdf5/preview/table":         {"dataset_path", "offset", "limit"},
}

// proxyUploadHdf5 is the shared skeleton for all six HDF5 routes: not-configured
// guard, auth/ownership resolution, allowlisted query passthrough, then the caller's
// chosen proxy variant (cached vs streaming). No fallback handler is passed — HDF5
// has no legacy native path.
func (deps ServerDeps) proxyUploadHdf5(w http.ResponseWriter, r *http.Request, endpoint string,
	proxy func(http.ResponseWriter, *http.Request, string, url.Values, ...http.HandlerFunc)) {
	if !deps.imageServiceConfigured() {
		deps.handleNotConfigured("HDF5 preview requires the image service (set ULTRA_CONTROL_IMAGE_SERVICE_URL)")(w, r)
		return
	}
	_, record, path, ok := deps.resolveUploadServingRequest(w, r)
	if !ok {
		return
	}
	// file_id is echoed by the sidecar into the JSON payloads (dataset summary,
	// histogram, table). The frontend builds every follow-up preview
	// URL from summary.file_id (Hdf5DatasetPreview), so it must be the real id —
	// always the server-resolved one, never client input.
	query := url.Values{"path": {path}, "file_id": {record.FileID}}
	for _, key := range hdf5QueryKeys[endpoint] {
		if v := strings.TrimSpace(r.URL.Query().Get(key)); v != "" {
			query.Set(key, v)
		}
	}
	proxy(w, r, endpoint, query)
}

// handleGetUploadHdf5Dataset proxies the per-dataset summary (JSON). Small,
// repeatable, keyed by the source file's stat stamp + dataset_path -> main cache.
func (deps ServerDeps) handleGetUploadHdf5Dataset(w http.ResponseWriter, r *http.Request) {
	deps.proxyUploadHdf5(w, r, "/hdf5/dataset", deps.proxyImageServiceCached)
}

// handleServeUploadHdf5Slice proxies a rendered dataset slice (PNG). This is a
// scrub endpoint (the viewer sweeps `index`), so it rides the dedicated slice
// cache partition — the exact analog of /v2/uploads/{id}/slice.
func (deps ServerDeps) handleServeUploadHdf5Slice(w http.ResponseWriter, r *http.Request) {
	deps.proxyUploadHdf5(w, r, "/hdf5/preview/slice", deps.proxyImageServiceSliceCached)
}

// handleServeUploadHdf5Atlas proxies the Z-atlas mosaic (PNG) -> main cache,
// the analog of /v2/uploads/{id}/atlas.
func (deps ServerDeps) handleServeUploadHdf5Atlas(w http.ResponseWriter, r *http.Request) {
	deps.proxyUploadHdf5(w, r, "/hdf5/preview/atlas", deps.proxyImageServiceCached)
}

// handleGetUploadHdf5ScalarVolume proxies the raw voxel buffer (octet-stream).
// Large one-shot payload -> uncached streaming proxy, which already forwards the
// x-volume-* geometry headers (incl. x-volume-channel) verbatim.
func (deps ServerDeps) handleGetUploadHdf5ScalarVolume(w http.ResponseWriter, r *http.Request) {
	deps.proxyUploadHdf5(w, r, "/hdf5/preview/scalar-volume", deps.proxyImageService)
}

// handleGetUploadHdf5Histogram proxies the sampled distribution (JSON). The
// Python side emits the frozen frontend contract directly (no Go-side remap,
// unlike the raster /histogram), so it can ride the main cache.
func (deps ServerDeps) handleGetUploadHdf5Histogram(w http.ResponseWriter, r *http.Request) {
	deps.proxyUploadHdf5(w, r, "/hdf5/preview/histogram", deps.proxyImageServiceCached)
}

// handleGetUploadHdf5Table proxies a bounded row window (JSON). offset/limit are
// part of the cache key via the encoded query -> main cache.
func (deps ServerDeps) handleGetUploadHdf5Table(w http.ResponseWriter, r *http.Request) {
	deps.proxyUploadHdf5(w, r, "/hdf5/preview/table", deps.proxyImageServiceCached)
}
