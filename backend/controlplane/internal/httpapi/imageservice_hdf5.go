package httpapi

import (
	"errors"
	"net/http"
	"net/url"
	"sort"
	"strconv"
	"strings"
)

// HDF5 viewer proxy routes (/v2/uploads/{file_id}/hdf5/*).
//
// The Python image service owns all h5py work (tree walk, dataset summaries,
// slice/atlas/scalar-volume rendering, DREAM3D dashboards); the control plane
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
	"/hdf5/materials/dashboard":   nil,
	"/hdf5/preview/slice":         {"dataset_path", "axis", "index", "component", "feature_ids"},
	"/hdf5/preview/atlas":         {"dataset_path", "component", "enhancement", "fusion_method", "negative", "channels", "feature_ids"},
	"/hdf5/preview/scalar-volume": {"dataset_path", "channel"},
	"/hdf5/preview/histogram":     {"dataset_path", "component", "bins"},
	"/hdf5/preview/table":         {"dataset_path", "offset", "limit"},
}

const (
	hdf5FeatureIDsMaxQueryBytes = 1024
	hdf5FeatureIDsMaxUnique     = 64
)

// canonicalHdf5FeatureIDs validates the public filter grammar and produces one
// stable cache-key representation. It runs only after upload auth/ownership has
// succeeded, so malformed input cannot reveal whether a foreign file exists.
func canonicalHdf5FeatureIDs(values []string) (string, error) {
	if len(values) != 1 {
		return "", errors.New("feature_ids must be supplied exactly once")
	}
	raw := values[0]
	if raw == "" || len(raw) > hdf5FeatureIDsMaxQueryBytes {
		return "", errors.New("feature_ids must be a non-empty comma-separated list up to 1 KiB")
	}
	for _, char := range raw {
		if (char < '0' || char > '9') && char != ',' {
			return "", errors.New("feature_ids must contain only digits and commas")
		}
	}
	unique := make(map[uint32]struct{})
	for _, token := range strings.Split(raw, ",") {
		if token == "" {
			return "", errors.New("feature_ids contains an empty value")
		}
		value, err := strconv.ParseUint(token, 10, 32)
		if err != nil || value == 0 {
			return "", errors.New("feature_ids must contain only positive uint32 values")
		}
		unique[uint32(value)] = struct{}{}
	}
	if len(unique) > hdf5FeatureIDsMaxUnique {
		return "", errors.New("feature_ids supports at most 64 unique values")
	}
	ordered := make([]uint32, 0, len(unique))
	for value := range unique {
		ordered = append(ordered, value)
	}
	sort.Slice(ordered, func(i, j int) bool { return ordered[i] < ordered[j] })
	canonical := make([]string, len(ordered))
	for index, value := range ordered {
		canonical[index] = strconv.FormatUint(uint64(value), 10)
	}
	return strings.Join(canonical, ","), nil
}

// proxyUploadHdf5 is the shared skeleton for all HDF5 routes: not-configured
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
	parsedQuery, err := url.ParseQuery(r.URL.RawQuery)
	if err != nil {
		writeError(w, http.StatusBadRequest, errors.New("invalid HDF5 preview query"))
		return
	}
	// file_id is echoed by the sidecar into the JSON payloads (dataset summary,
	// dashboard, histogram, table). The frontend builds every follow-up preview
	// URL from summary.file_id (Hdf5DatasetPreview), so it must be the real id —
	// always the server-resolved one, never client input.
	query := url.Values{"path": {path}, "file_id": {record.FileID}}
	for _, key := range hdf5QueryKeys[endpoint] {
		values, present := parsedQuery[key]
		if key == "dataset_path" {
			if !present || len(values) != 1 || values[0] == "" {
				writeError(w, http.StatusBadRequest, errors.New("dataset_path must be supplied exactly once and must not be empty"))
				return
			}
		} else if present && len(values) != 1 {
			writeError(w, http.StatusBadRequest, errors.New(key+" must be supplied at most once"))
			return
		}
		if !present {
			continue
		}
		if key == "feature_ids" {
			canonical, err := canonicalHdf5FeatureIDs(values)
			if err != nil {
				writeError(w, http.StatusBadRequest, err)
				return
			}
			query.Set(key, canonical)
			continue
		}
		query.Set(key, values[0])
	}
	proxy(w, r, endpoint, query)
}

// handleGetUploadHdf5Dataset proxies the per-dataset summary (JSON). Small,
// repeatable, keyed by the source file's stat stamp + dataset_path -> main cache.
func (deps ServerDeps) handleGetUploadHdf5Dataset(w http.ResponseWriter, r *http.Request) {
	deps.proxyUploadHdf5(w, r, "/hdf5/dataset", deps.proxyImageServiceCached)
}

// handleGetUploadHdf5MaterialsDashboard proxies the DREAM3D materials dashboard
// (JSON). Expensive to compute (grain stats) and fully deterministic per file ->
// main cache.
func (deps ServerDeps) handleGetUploadHdf5MaterialsDashboard(w http.ResponseWriter, r *http.Request) {
	deps.proxyUploadHdf5(w, r, "/hdf5/materials/dashboard", deps.proxyImageServiceCached)
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
