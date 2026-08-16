package httpapi

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"mime"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-chi/chi/v5"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
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

const (
	defaultScalarVolumeInFlightBytes int64 = 256 << 20
	maxImageTileEdge                       = 1024
	maxCompositeImageChannels              = 8
	maxImageAtlasGridEdge                  = 4096
	maxImageAtlasCells                     = 65536
)

func copyQueryValueIfPresent(dst, src url.Values, key string) {
	values, present := src[key]
	if !present {
		return
	}
	value := ""
	if len(values) > 0 {
		value = values[0]
	}
	dst.Set(key, value)
}

func cloneImageQuery(query url.Values) url.Values {
	cloned := make(url.Values, len(query))
	for key, values := range query {
		cloned[key] = append([]string(nil), values...)
	}
	return cloned
}

func parseImageTileSize(query url.Values) (string, bool, error) {
	raw, present, err := exactRawQueryValue(query, []string{"size"}, "tile size")
	if err != nil || !present {
		return "", present, err
	}
	size, err := strconv.Atoi(raw)
	if err != nil || size <= 0 || size > maxImageTileEdge {
		return "", true, fmt.Errorf("tile size must be an integer between 1 and %d", maxImageTileEdge)
	}
	return strconv.Itoa(size), true, nil
}

type imageAtlasOptions struct {
	level, rows, cols, scale                             string
	levelPresent, rowsPresent, colsPresent, scalePresent bool
}

func parseImageAtlasOptions(query url.Values) (imageAtlasOptions, error) {
	var options imageAtlasOptions
	var err error
	options.level, options.levelPresent, err = exactRawQueryValue(query, []string{"level"}, "atlas level")
	if err != nil {
		return imageAtlasOptions{}, err
	}
	options.rows, options.rowsPresent, err = exactRawQueryValue(query, []string{"grid_rows"}, "atlas grid rows")
	if err != nil {
		return imageAtlasOptions{}, err
	}
	options.cols, options.colsPresent, err = exactRawQueryValue(query, []string{"grid_cols"}, "atlas grid columns")
	if err != nil {
		return imageAtlasOptions{}, err
	}
	options.scale, options.scalePresent, err = exactRawQueryValue(query, []string{"scale"}, "atlas scale")
	if err != nil {
		return imageAtlasOptions{}, err
	}
	if options.levelPresent {
		level, parseErr := parseExactNonNegativeDecimal(options.level, "atlas level")
		if parseErr != nil {
			return imageAtlasOptions{}, parseErr
		}
		options.level = strconv.Itoa(level)
	}
	if options.rowsPresent != options.colsPresent {
		return imageAtlasOptions{}, errors.New("atlas grid rows and columns must be supplied together")
	}
	if options.rowsPresent {
		rows, rowsErr := parseExactNonNegativeDecimal(options.rows, "atlas grid rows")
		cols, colsErr := parseExactNonNegativeDecimal(options.cols, "atlas grid columns")
		if rowsErr != nil {
			return imageAtlasOptions{}, rowsErr
		}
		if colsErr != nil {
			return imageAtlasOptions{}, colsErr
		}
		if rows < 1 || cols < 1 || rows > maxImageAtlasGridEdge || cols > maxImageAtlasGridEdge || rows > maxImageAtlasCells/cols {
			return imageAtlasOptions{}, fmt.Errorf("atlas grid must contain 1 to %d bounded cells", maxImageAtlasCells)
		}
		options.rows, options.cols = strconv.Itoa(rows), strconv.Itoa(cols)
	}
	if options.scalePresent {
		scale, parseErr := strconv.ParseFloat(options.scale, 64)
		if parseErr != nil || math.IsNaN(scale) || math.IsInf(scale, 0) || scale <= 0 || scale > 1 {
			return imageAtlasOptions{}, errors.New("atlas scale must be greater than 0 and at most 1")
		}
		options.scale = strconv.FormatFloat(scale, 'g', -1, 64)
	}
	return options, nil
}

func (options imageAtlasOptions) apply(query url.Values) {
	if options.levelPresent {
		query.Set("level", options.level)
	}
	if options.rowsPresent {
		query.Set("grid_rows", options.rows)
		query.Set("grid_cols", options.cols)
	}
	if options.scalePresent {
		query.Set("scale", options.scale)
	}
}

func parseImageSliceOptions(query url.Values) (url.Values, error) {
	canonical := url.Values{}
	axis, axisPresent, err := exactRawQueryValue(query, []string{"axis"}, "slice axis")
	if err != nil {
		return nil, err
	}
	if axisPresent {
		if axis != "z" {
			return nil, errors.New("only the z slice axis is supported")
		}
		canonical.Set("axis", "z")
	}
	for _, key := range []string{"x", "y"} {
		if _, present, parseErr := exactRawQueryValue(query, []string{key}, "slice "+key+" selector"); parseErr != nil {
			return nil, parseErr
		} else if present {
			return nil, fmt.Errorf("slice %s selectors are unsupported; use z", key)
		}
	}
	level, levelPresent, err := exactRawQueryValue(query, []string{"level"}, "slice level")
	if err != nil {
		return nil, err
	}
	if levelPresent {
		parsed, parseErr := parseExactNonNegativeDecimal(level, "slice level")
		if parseErr != nil {
			return nil, parseErr
		}
		canonical.Set("level", strconv.Itoa(parsed))
	}
	fullResolution, fullResolutionPresent, err := exactRawQueryValue(query, []string{"full_resolution"}, "full resolution selector")
	if err != nil {
		return nil, err
	}
	if fullResolutionPresent {
		if fullResolution != "true" && fullResolution != "false" {
			return nil, errors.New("full resolution selector must be true or false")
		}
		canonical.Set("full_resolution", fullResolution)
	}
	return canonical, nil
}

type niftiMPRSelection struct {
	axis       string
	coordinate int
	time       int
	channel    int
}

// parseNiftiMPRSelection is deliberately separate from the generic scientific
// Z-slice parser. NIfTI is the native MPR format: it accepts x/y/z planes, but
// each plane must name exactly one matching coordinate and at most one canonical
// alias for time/channel. No missing selector is silently centered or clamped.
func parseNiftiMPRSelection(query url.Values) (niftiMPRSelection, error) {
	axis, axisPresent, err := exactRawQueryValue(query, []string{"axis"}, "NIfTI slice axis")
	if err != nil {
		return niftiMPRSelection{}, err
	}
	if !axisPresent || (axis != "x" && axis != "y" && axis != "z") {
		return niftiMPRSelection{}, errors.New("NIfTI slice axis must be exactly one of x, y, or z")
	}
	selection := niftiMPRSelection{axis: axis}
	for _, coordinateAxis := range []string{"x", "y", "z"} {
		raw, present, coordinateErr := exactRawQueryValue(
			query,
			[]string{coordinateAxis},
			"NIfTI "+coordinateAxis+" coordinate",
		)
		if coordinateErr != nil {
			return niftiMPRSelection{}, coordinateErr
		}
		if coordinateAxis != axis && present {
			return niftiMPRSelection{}, errors.New("NIfTI slice must contain only the coordinate matching its axis")
		}
		if coordinateAxis == axis {
			if !present {
				return niftiMPRSelection{}, errors.New("NIfTI slice requires the coordinate matching its axis")
			}
			selection.coordinate, err = parseExactNonNegativeDecimal(raw, "NIfTI "+axis+" coordinate")
			if err != nil {
				return niftiMPRSelection{}, err
			}
		}
	}
	timeRaw, timePresent, err := exactRawQueryValue(query, []string{"t", "time", "timepoint"}, "NIfTI time selector")
	if err != nil {
		return niftiMPRSelection{}, err
	}
	if timePresent {
		selection.time, err = parseExactNonNegativeDecimal(timeRaw, "NIfTI time selector")
		if err != nil {
			return niftiMPRSelection{}, err
		}
	}
	channelRaw, channelPresent, err := exactRawQueryValue(query, []string{"channels", "channel", "c"}, "NIfTI channel selector")
	if err != nil {
		return niftiMPRSelection{}, err
	}
	if channelPresent {
		if strings.Contains(channelRaw, ",") {
			return niftiMPRSelection{}, errors.New("NIfTI slice requires exactly one channel")
		}
		selection.channel, err = parseExactNonNegativeDecimal(channelRaw, "NIfTI channel selector")
		if err != nil {
			return niftiMPRSelection{}, err
		}
	}
	if _, present, colorErr := exactRawQueryValue(query, []string{"channel_colors"}, "NIfTI channel LUT selector"); colorErr != nil {
		return niftiMPRSelection{}, colorErr
	} else if present {
		return niftiMPRSelection{}, errors.New("NIfTI scalar slices do not accept channel LUT selectors")
	}
	return selection, nil
}

func validateNiftiMPRBounds(selection niftiMPRSelection, width, height, depth, timeCount, channelCount int) error {
	axisSize := map[string]int{"x": width, "y": height, "z": depth}[selection.axis]
	if axisSize < 1 || selection.coordinate >= axisSize {
		return fmt.Errorf("NIfTI %s coordinate is out of range", selection.axis)
	}
	if timeCount < 1 || selection.time >= timeCount {
		return errors.New("NIfTI time selector is out of range")
	}
	if channelCount < 1 || selection.channel >= channelCount {
		return errors.New("NIfTI channel selector is out of range")
	}
	return nil
}

func validateCatalogNiftiMPRBounds(record resourceRecord, selection niftiMPRSelection) (bool, error) {
	header, ok := jsonObject(record.Metadata["image_header"])
	if !ok {
		return false, nil
	}
	width, widthOK := jsonInt(header["width"])
	height, heightOK := jsonInt(header["height"])
	depth, depthOK := jsonInt(header["depth"])
	timeCount, timeOK := jsonInt(header["time_count"])
	channelCount, channelOK := jsonInt(header["channel_count"])
	if !widthOK || !heightOK || !depthOK || !timeOK || !channelOK {
		return false, nil
	}
	return true, validateNiftiMPRBounds(selection, width, height, depth, timeCount, channelCount)
}

type scientificImageSelectors struct {
	t               string
	tPresent        bool
	z               string
	zPresent        bool
	channels        string
	channelsPresent bool
	colors          string
	colorsPresent   bool
}

func (selectors scientificImageSelectors) present() bool {
	return selectors.tPresent || selectors.zPresent || selectors.channelsPresent || selectors.colorsPresent
}

func catalogImageSelectorLimits(record resourceRecord) (timeCount, channelCount, depth int, ok bool) {
	header, ok := jsonObject(record.Metadata["image_header"])
	if !ok {
		return 0, 0, 0, false
	}
	// A fallback descriptor carries warnings because the Go header reader could not
	// recover scientific axes. Do not turn its synthetic 1/1/1 values into authority.
	switch warnings := header["warnings"].(type) {
	case []any:
		if len(warnings) > 0 {
			return 0, 0, 0, false
		}
	case []string:
		if len(warnings) > 0 {
			return 0, 0, 0, false
		}
	}
	timeCount, tOK := jsonInt(header["time_count"])
	channelCount, cOK := jsonInt(header["channel_count"])
	depth, zOK := jsonInt(header["depth"])
	if !tOK || !cOK || !zOK || timeCount < 1 || channelCount < 1 || depth < 1 {
		return 0, 0, 0, false
	}
	return timeCount, channelCount, depth, true
}

func parseScientificImageSelectors(query url.Values, record resourceRecord) (scientificImageSelectors, error) {
	var selectors scientificImageSelectors
	var err error
	selectors.t, selectors.tPresent, err = exactRawQueryValue(query, []string{"t"}, "time selector")
	if err != nil {
		return scientificImageSelectors{}, err
	}
	selectors.z, selectors.zPresent, err = exactRawQueryValue(query, []string{"z"}, "z selector")
	if err != nil {
		return scientificImageSelectors{}, err
	}
	selectors.channels, selectors.channelsPresent, err = exactRawQueryValue(
		query, []string{"channels"}, "channel selector",
	)
	if err != nil {
		return scientificImageSelectors{}, err
	}
	singleChannel, singleChannelPresent, err := exactRawQueryValue(
		query, []string{"c", "channel"}, "single channel selector",
	)
	if err != nil {
		return scientificImageSelectors{}, err
	}
	if selectors.channelsPresent && singleChannelPresent {
		return scientificImageSelectors{}, errors.New("channel selector aliases cannot be mixed")
	}
	if singleChannelPresent {
		if _, parseErr := parseExactNonNegativeDecimal(singleChannel, "single channel selector"); parseErr != nil {
			return scientificImageSelectors{}, parseErr
		}
		selectors.channels = singleChannel
		selectors.channelsPresent = true
	}
	selectors.colors, selectors.colorsPresent, err = exactRawQueryValue(
		query, []string{"channel_colors"}, "channel LUT selector",
	)
	if err != nil {
		return scientificImageSelectors{}, err
	}

	timeIndex, zIndex := 0, 0
	if selectors.tPresent {
		timeIndex, err = parseExactNonNegativeDecimal(selectors.t, "time selector")
		if err != nil {
			return scientificImageSelectors{}, err
		}
	}
	if selectors.zPresent {
		zIndex, err = parseExactNonNegativeDecimal(selectors.z, "z selector")
		if err != nil {
			return scientificImageSelectors{}, err
		}
	}

	channels := make([]int, 0, maxCompositeImageChannels)
	if selectors.channelsPresent {
		seen := make(map[int]struct{}, maxCompositeImageChannels)
		for _, raw := range strings.Split(selectors.channels, ",") {
			channel, parseErr := parseExactNonNegativeDecimal(strings.TrimSpace(raw), "channel selector")
			if parseErr != nil {
				return scientificImageSelectors{}, parseErr
			}
			if _, duplicate := seen[channel]; duplicate {
				return scientificImageSelectors{}, errors.New("duplicate channel selector")
			}
			seen[channel] = struct{}{}
			channels = append(channels, channel)
			if len(channels) > maxCompositeImageChannels {
				return scientificImageSelectors{}, fmt.Errorf(
					"channel selector supports at most %d channels", maxCompositeImageChannels,
				)
			}
		}
	}
	if selectors.colorsPresent {
		if !selectors.channelsPresent {
			return scientificImageSelectors{}, errors.New("channel LUT selector requires channels")
		}
		colors := strings.Split(selectors.colors, ",")
		if len(colors) != len(channels) {
			return scientificImageSelectors{}, errors.New("channel LUT count must match channel count")
		}
		for _, color := range colors {
			if len(color) != 7 || color[0] != '#' {
				return scientificImageSelectors{}, errors.New("channel LUT values must be #RRGGBB")
			}
			for _, char := range color[1:] {
				if !((char >= '0' && char <= '9') || (char >= 'a' && char <= 'f') || (char >= 'A' && char <= 'F')) {
					return scientificImageSelectors{}, errors.New("channel LUT values must be #RRGGBB")
				}
			}
		}
	}
	if timeCount, channelCount, depth, authoritative := catalogImageSelectorLimits(record); authoritative {
		if timeIndex >= timeCount {
			return scientificImageSelectors{}, errors.New("time selector is out of range")
		}
		if zIndex >= depth {
			return scientificImageSelectors{}, errors.New("z selector is out of range")
		}
		for _, channel := range channels {
			if channel >= channelCount {
				return scientificImageSelectors{}, errors.New("channel selector is out of range")
			}
		}
	}
	return selectors, nil
}

func (selectors scientificImageSelectors) apply(query url.Values) {
	if selectors.tPresent {
		query.Set("t", selectors.t)
	}
	if selectors.zPresent {
		query.Set("z", selectors.z)
	}
	if selectors.channelsPresent {
		query.Set("channels", selectors.channels)
	}
	if selectors.colorsPresent {
		query.Set("channel_colors", selectors.colors)
	}
}

type byteAdmissionBudget struct {
	mu       sync.Mutex
	maxBytes int64
	used     int64
}

func newByteAdmissionBudget(maxBytes int64) *byteAdmissionBudget {
	if maxBytes < 0 {
		maxBytes = 0
	}
	return &byteAdmissionBudget{maxBytes: maxBytes}
}

func (budget *byteAdmissionBudget) tryAcquire(size int64) bool {
	if budget == nil || size <= 0 {
		return false
	}
	budget.mu.Lock()
	defer budget.mu.Unlock()
	if size > budget.maxBytes-budget.used {
		return false
	}
	budget.used += size
	return true
}

func (budget *byteAdmissionBudget) release(size int64) {
	if budget == nil || size <= 0 {
		return
	}
	budget.mu.Lock()
	defer budget.mu.Unlock()
	budget.used -= size
	if budget.used < 0 {
		budget.used = 0
	}
}

func newScalarVolumeInFlightBudgetFromEnv() *byteAdmissionBudget {
	maxBytes := defaultScalarVolumeInFlightBytes
	if raw := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_SCALAR_VOLUME_INFLIGHT_BYTES")); raw != "" {
		if parsed, err := strconv.ParseInt(raw, 10, 64); err == nil && parsed >= 0 {
			maxBytes = parsed
		}
	}
	return newByteAdmissionBudget(maxBytes)
}

var scalarVolumeInFlightBudget = newScalarVolumeInFlightBudgetFromEnv()

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
	authorization, ok := deps.authorizeUploadServingRequest(w, r)
	if !ok {
		return
	}
	if axis := chi.URLParam(r, "axis"); axis != "z" {
		writeError(w, http.StatusUnprocessableEntity, errors.New("only the z tile axis is supported"))
		return
	}
	record := authorization.record
	selectors, selectorErr := parseScientificImageSelectors(r.URL.Query(), record)
	if selectorErr != nil {
		writeError(w, http.StatusUnprocessableEntity, selectorErr)
		return
	}
	size, sizePresent, sizeErr := parseImageTileSize(r.URL.Query())
	if sizeErr != nil {
		writeError(w, http.StatusUnprocessableEntity, sizeErr)
		return
	}
	path, ok := resolveAuthorizedUploadStorage(w, authorization)
	if !ok {
		return
	}
	root := authorization.root
	// OME-Zarr: a tile is a bounded DeepZoom read of the store at a multiscale level,
	// served natively by the ngff-service (only the tile's covering chunks are decoded —
	// a gigapixel/1 TB level-0 plane is never materialized). level maps 1:1 to the zarr
	// multiscale level advertised in the tile_scheme.
	if deps.servesViaNgff(record, path) {
		q := url.Values{"path": {path}}
		q.Set("level", chi.URLParam(r, "level"))
		q.Set("col", chi.URLParam(r, "tile_x"))
		q.Set("row", chi.URLParam(r, "tile_y"))
		if sizePresent {
			q.Set("size", size)
		}
		selectors.apply(q)
		copyQueryValueIfPresent(q, r.URL.Query(), "cache_key")
		deps.ngffDeps().proxyImageServiceCached(w, r, "/tile", q)
		return
	}
	if !deps.imageServiceConfigured() {
		deps.handleNotConfigured("upload tile pyramid delivery requires the image service (set ULTRA_CONTROL_IMAGE_SERVICE_URL)")(w, r)
		return
	}
	// libbioimage tiles read the derived tiled pyramid (its native level 0 is a bounded read).
	servePath := path
	if dp, _, compatible := deps.compatibleDerivedPyramid(
		r.Context(), root, record, path, nil, derivativeUse{
			capability:      "tile",
			requireT:        selectors.tPresent,
			requireZ:        selectors.zPresent,
			requireChannels: selectors.channelsPresent,
			requireLUT:      selectors.colorsPresent,
		},
	); compatible {
		servePath = dp
	}
	query := url.Values{}
	query.Set("path", servePath)
	query.Set("level", chi.URLParam(r, "level"))
	query.Set("col", chi.URLParam(r, "tile_x"))
	query.Set("row", chi.URLParam(r, "tile_y"))
	if axis := chi.URLParam(r, "axis"); axis != "" {
		query.Set("axis", axis)
	}
	if sizePresent {
		query.Set("size", size)
	}
	// Multi-channel fluorescence compositing: forward selected channels + LUT
	// colors so the image service fuses them additively.
	selectors.apply(query)
	copyQueryValueIfPresent(query, r.URL.Query(), "cache_key")
	if servePath != path {
		deps.proxyImageServiceCached(w, r, "/tile", query, func(w http.ResponseWriter, r *http.Request) {
			sourceQuery := cloneImageQuery(query)
			sourceQuery.Set("path", path)
			deps.proxyImageServiceCached(w, r, "/tile", sourceQuery)
		})
		return
	}
	deps.proxyImageServiceCached(w, r, "/tile", query)
}

// handleServeUploadAtlas proxies a z-stack texture atlas to the image service.
func (deps ServerDeps) handleServeUploadAtlas(w http.ResponseWriter, r *http.Request) {
	authorization, ok := deps.authorizeUploadServingRequest(w, r)
	if !ok {
		return
	}
	record := authorization.record
	selectors, selectorErr := parseScientificImageSelectors(r.URL.Query(), record)
	if selectorErr != nil {
		writeError(w, http.StatusUnprocessableEntity, selectorErr)
		return
	}
	atlasOptions, optionsErr := parseImageAtlasOptions(r.URL.Query())
	if optionsErr != nil {
		writeError(w, http.StatusUnprocessableEntity, optionsErr)
		return
	}
	sourcePath, ok := resolveAuthorizedUploadStorage(w, authorization)
	if !ok {
		return
	}
	root := authorization.root
	if !deps.imageServiceConfigured() {
		deps.handleNotConfigured("upload atlas delivery requires the image service (set ULTRA_CONTROL_IMAGE_SERVICE_URL)")(w, r)
		return
	}
	path := sourcePath
	if dp, _, compatible := deps.compatibleDerivedPyramid(
		r.Context(), root, record, sourcePath, nil, derivativeUse{
			capability:      "atlas",
			requireT:        selectors.tPresent,
			requireChannels: selectors.channelsPresent,
			requireLUT:      selectors.colorsPresent,
		},
	); compatible {
		path = dp
	}
	query := url.Values{}
	query.Set("path", path)
	// channels/channel_colors composite a multi-channel z-stack into an RGB atlas
	// for the 3D fluorescence volume render.
	atlasOptions.apply(query)
	selectors.apply(query)
	if path != sourcePath {
		deps.proxyImageServiceCached(w, r, "/atlas", query, func(w http.ResponseWriter, r *http.Request) {
			sourceQuery := cloneImageQuery(query)
			sourceQuery.Set("path", sourcePath)
			deps.proxyImageServiceCached(w, r, "/atlas", sourceQuery)
		})
		return
	}
	deps.proxyImageServiceCached(w, r, "/atlas", query)
}

// resolveUploadTilePathForImageService resolves the path tiles/atlas should be
// served from: the derived tiled pyramid when one exists (bounded reads from a
// 50GB-class source), else the source itself (natively pyramidal formats already
// tile fine). Auth/ownership is enforced identically to every upload handler.
type uploadServingAuthorization struct {
	root            string
	record          resourceRecord
	catalogResource *domain.ResourceRecord
	legacyPath      string
}

// authorizeUploadServingRequest performs the owner-scoped catalog lookup without
// touching the resource's storage path. This ordering is security-significant:
// an unauthorized malformed request remains a hidden 404, while an owner gets a
// useful 422 for malformed selectors even when the source is temporarily absent.
func (deps ServerDeps) authorizeUploadServingRequest(w http.ResponseWriter, r *http.Request) (uploadServingAuthorization, bool) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return uploadServingAuthorization{}, false
	}
	principal := deps.principalFromRequest(r, "")
	fileID := chi.URLParam(r, "file_id")
	if catalog, hasCatalog := deps.resourceCatalogStore(); hasCatalog {
		if err := deps.ensureUploadCatalogMigrated(r.Context(), root); err != nil {
			writeStoreError(w, err)
			return uploadServingAuthorization{}, false
		}
		resource, err := catalog.GetResourceForUser(r.Context(), fileID, principal.UserID, principal.OrgID)
		if err != nil {
			writeStoreError(w, err)
			return uploadServingAuthorization{}, false
		}
		return uploadServingAuthorization{
			root:            root,
			record:          resourceRecordFromCatalogState(resource, false),
			catalogResource: &resource,
		}, true
	}
	record, path, err := deps.findUploadResourceForRequest(r.Context(), root, principal, fileID)
	if err != nil {
		writeStoreError(w, err)
		return uploadServingAuthorization{}, false
	}
	return uploadServingAuthorization{root: root, record: record, legacyPath: path}, true
}

func (authorization uploadServingAuthorization) resolveStoragePath() (string, error) {
	if authorization.catalogResource == nil {
		return authorization.legacyPath, nil
	}
	path, err := resolveCatalogResourcePath(authorization.root, *authorization.catalogResource)
	if err != nil {
		return "", err
	}
	if _, err := os.Stat(path); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", store.ErrNotFound
		}
		return "", err
	}
	return path, nil
}

func resolveAuthorizedUploadStorage(w http.ResponseWriter, authorization uploadServingAuthorization) (string, bool) {
	path, err := authorization.resolveStoragePath()
	if err != nil {
		writeStoreError(w, err)
		return "", false
	}
	return path, true
}

// resolveUploadServingRequest is retained for viewer routes that do not have
// request selectors. Selector-bearing tile/slice/atlas routes use the explicit
// authorize-then-validate-then-resolve sequence above.
func (deps ServerDeps) resolveUploadServingRequest(w http.ResponseWriter, r *http.Request) (root string, record resourceRecord, path string, ok bool) {
	authorization, ok := deps.authorizeUploadServingRequest(w, r)
	if !ok {
		return "", resourceRecord{}, "", false
	}
	path, ok = resolveAuthorizedUploadStorage(w, authorization)
	if !ok {
		return "", resourceRecord{}, "", false
	}
	return authorization.root, authorization.record, path, true
}

// proxyImageService forwards a GET to the internal image service and streams the
// response back. The control plane has already authorized the request and
// resolved the storage path; the sidecar is never reached directly.
// proxyImageService streams an image-service GET back to the client. When the
// sidecar is unreachable (dial/transport error, before anything is written) and a
// fallback handler is supplied, it degrades to that legacy native handler instead
// of a 502 — so a crashed image service doesn't break serving for formats Go can
// still handle. Variadic so existing callers (no fallback) keep returning 502.
func (deps ServerDeps) proxyImageService(w http.ResponseWriter, r *http.Request, endpoint string, query url.Values, fallback ...http.HandlerFunc) {
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
		if len(fallback) > 0 && fallback[0] != nil {
			fallback[0](w, r) // image service unreachable -> legacy native path
			return
		}
		writeError(w, http.StatusBadGateway, fmt.Errorf("image service request failed: %w", err))
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		detail, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		if len(fallback) > 0 && fallback[0] != nil {
			fallback[0](w, r)
			return
		}
		writeImageServiceUpstreamError(r.Context(), w, endpoint, resp.StatusCode, detail)
		return
	}
	if imageServiceRasterEndpoint(endpoint) {
		contentType, _, parseErr := mime.ParseMediaType(resp.Header.Get("Content-Type"))
		if parseErr != nil || !safeRasterMediaType(contentType) {
			writeError(
				w,
				http.StatusBadGateway,
				errors.New("image service returned an unsafe raster media type"),
			)
			return
		}
	}
	var admittedScalarBytes int64
	if endpoint == "/scalar-volume" {
		admittedScalarBytes = resp.ContentLength
		if admittedScalarBytes <= 0 {
			writeError(
				w,
				http.StatusBadGateway,
				errors.New("image service scalar volume omitted its content length"),
			)
			return
		}
		if !scalarVolumeInFlightBudget.tryAcquire(admittedScalarBytes) {
			w.Header().Set("Retry-After", "1")
			writeError(
				w,
				http.StatusServiceUnavailable,
				errors.New("scalar volume in-flight byte budget is exhausted"),
			)
			return
		}
		defer scalarVolumeInFlightBudget.release(admittedScalarBytes)
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

func imageServiceRasterEndpoint(endpoint string) bool {
	switch endpoint {
	case "/tile", "/atlas", "/slice", "/thumbnail", "/video-poster":
		return true
	default:
		return false
	}
}

func safeRasterMediaType(contentType string) bool {
	switch strings.ToLower(strings.TrimSpace(contentType)) {
	case "image/png", "image/jpeg", "image/gif", "image/webp", "image/bmp":
		return true
	default:
		return false
	}
}

// handleDeriveUploadPyramid enqueues an `image.derive_pyramid` batch job: convert
// this upload's source into a tiled pyramid (consumed by the convert worker,
// which runs libbioimage). The job rides the existing Data Agent job queue. Auth
// is enforced via the same upload resolution as the serving handlers.
func (deps ServerDeps) handleDeriveUploadPyramid(w http.ResponseWriter, r *http.Request) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	principal := deps.principalFromRequest(r, "")
	fileID := chi.URLParam(r, "file_id")
	record, path, err := deps.findUploadResourceForRequest(r.Context(), root, principal, fileID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if deps.DataAgentJobs == nil {
		deps.handleNotConfigured("image pyramid conversion requires the job queue (set up NATS)")(w, r)
		return
	}
	if !lowercaseSHA256Pattern.MatchString(strings.ToLower(strings.TrimSpace(record.SHA256))) || record.SizeBytes < 0 {
		writeError(w, http.StatusUnprocessableEntity, errors.New("resource has no immutable source identity for derivation"))
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
			"resource_id":       fileID,
			"src_path":          path,
			"dst_path":          dst,
			"source_sha256":     strings.ToLower(strings.TrimSpace(record.SHA256)),
			"source_size_bytes": record.SizeBytes,
			"tile_size":         512,
			"compression":       "lzw",
			// topdirs (each level a full page) keeps the native level tile-addressable.
			// fmt="auto": the worker picks the container from the source's
			// dimensionality — a z-stack/volume derives to OME-BigTIFF (the OME wrapper
			// preserves the Z planes, which a plain BigTIFF would flatten) for /slice +
			// /atlas reads; a flat 2D slide stays BigTIFF, because the OME wrapper breaks
			// this engine's embedded -tile reader and 2D serving is tile-based.
			"layout": "topdirs",
			"fmt":    "auto",
			"force":  true,
		},
	}
	if err := deps.DataAgentJobs.PublishDataAgentJob(r.Context(), envelope); err != nil {
		writeError(w, http.StatusBadGateway, fmt.Errorf("failed to enqueue pyramid conversion: %w", err))
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]any{
		"job_id":           jobID,
		"job_type":         "image.derive_pyramid",
		"resource_id":      fileID,
		"destination_hint": derivedPyramidManifestName(fileID),
		"status":           "queued",
	})
}
