package httpapi

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"image/color"
	"image/png"
	"math"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

// uploadNamedFileForProxyTest uploads arbitrary bytes under a chosen filename and
// returns the assigned file_id, so tests can exercise format-specific routing.
func uploadNamedFileForProxyTest(t *testing.T, router http.Handler, filename string, content []byte) string {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("files", filename)
	if err != nil {
		t.Fatalf("CreateFormFile: %v", err)
	}
	if _, err := part.Write(content); err != nil {
		t.Fatalf("write multipart file: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("close multipart writer: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v2/uploads", &body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	setProxyOwnerHeaders(req)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("upload status = %d body=%s", rec.Code, rec.Body.String())
	}
	var resp struct {
		Uploaded []struct {
			FileID string `json:"file_id"`
		} `json:"uploaded"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode upload response: %v", err)
	}
	if len(resp.Uploaded) != 1 || resp.Uploaded[0].FileID == "" {
		t.Fatalf("upload response = %+v, want one uploaded file", resp)
	}
	return resp.Uploaded[0].FileID
}

func TestV2UploadViewerProxiesImageService(t *testing.T) {
	t.Parallel()

	var gotPath string
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/viewerinfo" {
			http.NotFound(w, r)
			return
		}
		gotPath = r.URL.Query().Get("path")
		// Representative libbioimage viewer-info core (no control-plane fields).
		_ = json.NewEncoder(w).Encode(map[string]any{
			"kind":         "image",
			"modality":     "microscopy",
			"backend_mode": "direct",
			"dims_order":   "XYCZT",
			"axis_sizes":   map[string]any{"T": 1, "C": 2, "Z": 20, "Y": 2048, "X": 2048},
			"is_volume":    true,
			"tile_scheme":  nil,
			"viewer":       map[string]any{"volume_mode": "slice_stack", "asset_preparation": map[string]any{"tile_pyramid": "none"}},
			"metadata":     map[string]any{"reader": "libbioimage"},
			"phys":         map[string]any{"channel_names": []string{"DAPI", "FITC"}},
		})
	}))
	defer imageSvc.Close()

	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:           mem,
		UploadRoot:      t.TempDir(),
		ImageServiceURL: imageSvc.URL,
	})

	fileID := uploadNamedFileForProxyTest(t, router, "cells.png", testPNGBytes(t, 4, 4))

	req := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/viewer", nil)
	setProxyOwnerHeaders(req)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("viewer status = %d body=%s", rec.Code, rec.Body.String())
	}
	var vi map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &vi); err != nil {
		t.Fatalf("decode viewer: %v", err)
	}
	if vi["file_id"] != fileID {
		t.Fatalf("viewer file_id = %v, want %s", vi["file_id"], fileID)
	}
	if vi["original_name"] != "cells.png" {
		t.Fatalf("viewer original_name = %v", vi["original_name"])
	}
	if vi["modality"] != "microscopy" {
		t.Fatalf("viewer modality not preserved from image service: %v", vi["modality"])
	}
	urls, ok := vi["service_urls"].(map[string]any)
	if !ok || !strings.Contains(urls["slice"].(string), fileID) {
		t.Fatalf("viewer service_urls not injected: %v", vi["service_urls"])
	}
	if !strings.Contains(gotPath, fileID) {
		t.Fatalf("image service viewerinfo path = %q, want resolved storage path with %q", gotPath, fileID)
	}
}

// TestV2UploadViewerPrefersDerivedPyramidTileScheme locks in the fix for the deep-zoom
// over-fetch: when a derived pyramid exists it serves the tile PIXELS, so the viewer must
// use the PYRAMID's tile_scheme (its tile size + level grid) even when the source
// advertised its own. Otherwise the viewer fetches at the source geometry (e.g. 256-px /
// fewer levels) while pixels come from the 512-tiled pyramid, decoding each tile 4x.
func TestV2UploadViewerPrefersDerivedPyramidTileScheme(t *testing.T) {
	t.Parallel()

	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/viewerinfo" {
			http.NotFound(w, r)
			return
		}
		if strings.Contains(r.URL.Query().Get("path"), "pyramid") {
			// The derived pyramid: 512-px tiles, 3 levels — this is what serves the pixels.
			_ = json.NewEncoder(w).Encode(map[string]any{
				"axis_sizes": map[string]any{"X": 4096, "Y": 4096, "Z": 1, "C": 3, "T": 1},
				"tile_scheme": map[string]any{"tile_size": 512, "format": "png", "levels": []any{
					map[string]any{"level": 0, "width": 4096, "height": 4096, "downsample": 1},
					map[string]any{"level": 1, "width": 2048, "height": 2048, "downsample": 2},
					map[string]any{"level": 2, "width": 1024, "height": 1024, "downsample": 4},
				}},
			})
			return
		}
		// The source advertises its OWN, mismatched 256-px / 2-level scheme.
		_ = json.NewEncoder(w).Encode(map[string]any{
			"axis_sizes": map[string]any{"X": 4096, "Y": 4096, "Z": 1, "C": 3, "T": 1},
			"tile_scheme": map[string]any{"tile_size": 256, "format": "png", "levels": []any{
				map[string]any{"level": 0, "width": 4096, "height": 4096, "downsample": 1},
				map[string]any{"level": 1, "width": 2048, "height": 2048, "downsample": 2},
			}},
		})
	}))
	defer imageSvc.Close()

	mem := store.NewMemoryStore()
	root := t.TempDir()
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:           mem,
		UploadRoot:      root,
		ImageServiceURL: imageSvc.URL,
	})
	fileID := uploadNamedFileForProxyTest(t, router, "ortho.tif", testPNGBytes(t, 4, 4))
	if err := os.MkdirAll(filepath.Join(root, "derived"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "derived", derivedPyramidName(fileID)), []byte("pyramid bytes"), 0o644); err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/viewer", nil)
	setProxyOwnerHeaders(req)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("viewer status = %d body=%s", rec.Code, rec.Body.String())
	}
	var vi map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &vi); err != nil {
		t.Fatal(err)
	}
	ts, _ := vi["tile_scheme"].(map[string]any)
	if ts == nil {
		t.Fatalf("no tile_scheme served: %#v", vi["tile_scheme"])
	}
	if int(ts["tile_size"].(float64)) != 512 {
		t.Fatalf("served tile_size = %v, want 512 (the pyramid's, not the source's 256)", ts["tile_size"])
	}
	if levels, _ := ts["levels"].([]any); len(levels) != 3 {
		t.Fatalf("served %d levels, want 3 (the pyramid's, not the source's 2)", len(levels))
	}
	if vi["backend_mode"] != "pyramid" {
		t.Fatalf("backend_mode = %v, want pyramid", vi["backend_mode"])
	}
}

func TestV2UploadViewerFallsBackWhenImageServiceErrors(t *testing.T) {
	t.Parallel()

	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError) // force fallback
	}))
	defer imageSvc.Close()

	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:           mem,
		UploadRoot:      t.TempDir(),
		ImageServiceURL: imageSvc.URL,
	})
	fileID := uploadNamedFileForProxyTest(t, router, "plain.png", testPNGBytes(t, 8, 8))

	req := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/viewer", nil)
	setProxyOwnerHeaders(req)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	// Legacy native viewer still answers (reader=go-image), so no regression.
	if rec.Code != http.StatusOK {
		t.Fatalf("viewer fallback status = %d body=%s", rec.Code, rec.Body.String())
	}
	var vi map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &vi); err != nil {
		t.Fatalf("decode viewer: %v", err)
	}
	if vi["file_id"] != fileID {
		t.Fatalf("fallback viewer file_id = %v", vi["file_id"])
	}
}

func TestV2UploadSliceProxiesImageServiceWithZ(t *testing.T) {
	t.Parallel()

	wantPNG := []byte("\x89PNG\r\n\x1a\nZPLANE")
	var gotZ, gotPath string
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/slice" {
			http.NotFound(w, r)
			return
		}
		gotZ = r.URL.Query().Get("z")
		gotPath = r.URL.Query().Get("path")
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write(wantPNG)
	}))
	defer imageSvc.Close()

	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:           mem,
		UploadRoot:      t.TempDir(),
		ImageServiceURL: imageSvc.URL,
	})
	fileID := uploadNamedFileForProxyTest(t, router, "stack.png", testPNGBytes(t, 4, 4))

	req := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/slice?axis=z&z=7", nil)
	setProxyOwnerHeaders(req)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("slice status = %d body=%s", rec.Code, rec.Body.String())
	}
	if !bytes.Equal(rec.Body.Bytes(), wantPNG) {
		t.Fatalf("slice body not proxied from image service")
	}
	if gotZ != "7" {
		t.Fatalf("slice z = %q, want 7", gotZ)
	}
	if !strings.Contains(gotPath, fileID) {
		t.Fatalf("slice path = %q, want resolved storage path with %q", gotPath, fileID)
	}
}

func TestV2UploadSlicePrefersDerivedPyramidAndForwardsFullResolution(t *testing.T) {
	t.Parallel()

	// /slice should read the derived tiled pyramid when one exists (its native
	// level is pixel-identical but a far faster bounded read), and forward the
	// full_resolution scrub/settle signal to the image service.
	var gotPath, gotFullRes string
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/slice" {
			http.NotFound(w, r)
			return
		}
		gotPath = r.URL.Query().Get("path")
		gotFullRes = r.URL.Query().Get("full_resolution")
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write([]byte("\x89PNG\r\n\x1a\nX"))
	}))
	defer imageSvc.Close()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:           mem,
		UploadRoot:      uploadRoot,
		ImageServiceURL: imageSvc.URL,
	})
	fileID := uploadNamedFileForProxyTest(t, router, "zstack.ome.tiff", testPNGBytes(t, 4, 4))

	derivedDir := filepath.Join(uploadRoot, "derived")
	if err := os.MkdirAll(derivedDir, 0o755); err != nil {
		t.Fatalf("mkdir derived: %v", err)
	}
	if err := os.WriteFile(filepath.Join(derivedDir, derivedPyramidName(fileID)), []byte("OME-BIGTIFF"), 0o644); err != nil {
		t.Fatalf("write pyramid: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/slice?axis=z&z=5&full_resolution=false", nil)
	setProxyOwnerHeaders(req)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("slice status = %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(gotPath, "__pyramid.tif") {
		t.Fatalf("slice served from %q, want the derived pyramid", gotPath)
	}
	if gotFullRes != "false" {
		t.Fatalf("full_resolution forwarded = %q, want false", gotFullRes)
	}
}

func TestV2UploadSliceFallsBackWhenImageServiceUnreachable(t *testing.T) {
	t.Parallel()

	// A configured-but-unreachable image service (server created, then closed).
	down := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	downURL := down.URL
	down.Close()

	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:           mem,
		UploadRoot:      t.TempDir(),
		ImageServiceURL: downURL,
	})
	fileID := uploadNamedFileForProxyTest(t, router, "plane.png", testPNGBytes(t, 8, 8))

	req := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/slice?axis=z&z=0", nil)
	setProxyOwnerHeaders(req)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	// Degrades to the legacy native path (serves the source image), not a 502.
	if rec.Code != http.StatusOK {
		t.Fatalf("slice fallback status = %d, want 200 (legacy native path)", rec.Code)
	}
}

func TestV2UploadScalarVolumeForwardsHeaders(t *testing.T) {
	t.Parallel()

	raw := bytes.Repeat([]byte{1, 2, 3, 4}, 8)
	var viewerPath, scalarPath, scalarChannel, scalarTime, scalarSampling string
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/viewerinfo":
			viewerPath = r.URL.Query().Get("path")
			writeJSON(w, http.StatusOK, map[string]any{
				"axis_sizes": map[string]any{"T": 2, "C": 3, "Z": 8},
			})
		case "/scalar-volume":
			scalarPath = r.URL.Query().Get("path")
			scalarChannel = r.URL.Query().Get("channel")
			scalarTime = r.URL.Query().Get("t")
			scalarSampling = r.URL.Query().Get("sampling")
			w.Header().Set("Content-Type", "application/octet-stream")
			w.Header().Set("x-volume-width", "2")
			w.Header().Set("x-volume-height", "2")
			w.Header().Set("x-volume-depth", "8")
			w.Header().Set("x-volume-dtype", "float32")
			w.Header().Set("x-volume-bytes-per-voxel", "4")
			_, _ = w.Write(raw)
		default:
			http.NotFound(w, r)
		}
	}))
	defer imageSvc.Close()

	mem := store.NewMemoryStore()
	uploadRoot := t.TempDir()
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:           mem,
		UploadRoot:      uploadRoot,
		ImageServiceURL: imageSvc.URL,
	})
	fileID := uploadNamedFileForProxyTest(t, router, "vol.tif", testPNGBytes(t, 4, 4))
	derivedDir := filepath.Join(uploadRoot, "derived")
	if err := os.MkdirAll(derivedDir, 0o755); err != nil {
		t.Fatalf("create derived dir: %v", err)
	}
	if err := os.WriteFile(
		filepath.Join(derivedDir, derivedPyramidName(fileID)),
		[]byte("display derivative"),
		0o644,
	); err != nil {
		t.Fatalf("create derived pyramid fixture: %v", err)
	}

	req := httptest.NewRequest(
		http.MethodGet,
		"/v2/uploads/"+fileID+"/scalar-volume?channel=2&t=1&sampling=nearest",
		nil,
	)
	setProxyOwnerHeaders(req)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("scalar-volume status = %d body=%s", rec.Code, rec.Body.String())
	}
	if rec.Header().Get("x-volume-dtype") != "float32" || rec.Header().Get("x-volume-depth") != "8" {
		t.Fatalf("x-volume-* headers not forwarded: %v", rec.Header())
	}
	if !bytes.Equal(rec.Body.Bytes(), raw) {
		t.Fatalf("scalar-volume body not proxied")
	}
	if scalarPath == "" || scalarPath != viewerPath {
		t.Fatalf(
			"scalar-volume path = %q, viewer-info source path = %q",
			scalarPath,
			viewerPath,
		)
	}
	if strings.Contains(scalarPath, "__pyramid") {
		t.Fatalf("scalar-volume used display derivative %q instead of source", scalarPath)
	}
	if scalarChannel != "2" || scalarTime != "1" {
		t.Fatalf(
			"scalar-volume selection channel=%q t=%q, want channel=2 t=1",
			scalarChannel,
			scalarTime,
		)
	}
	if scalarSampling != "nearest" {
		t.Fatalf("scalar-volume sampling = %q, want nearest", scalarSampling)
	}
}

func TestSourceViewerAxesRequirePositiveIntegers(t *testing.T) {
	t.Parallel()

	valid := map[string]any{
		"axis_sizes": map[string]any{"T": float64(2), "C": 3, "Z": float64(8)},
	}
	if gotT, gotC, gotZ, ok := sourceViewerAxes(valid); !ok || gotT != 2 || gotC != 3 || gotZ != 8 {
		t.Fatalf("sourceViewerAxes(valid) = (%d, %d, %d, %t)", gotT, gotC, gotZ, ok)
	}

	for name, axes := range map[string]map[string]any{
		"fractional": {"T": 1.5, "C": 3, "Z": 8},
		"zero":       {"T": 1, "C": 0, "Z": 8},
		"missing":    {"T": 1, "C": 3},
		"string":     {"T": 1, "C": 3, "Z": "8"},
	} {
		t.Run(name, func(t *testing.T) {
			if _, _, _, ok := sourceViewerAxes(map[string]any{"axis_sizes": axes}); ok {
				t.Fatal("sourceViewerAxes accepted malformed source axes")
			}
		})
	}
}

func testScalarMaskViewerInfo(timeCount, channelCount, depth int, dtype string) map[string]any {
	return map[string]any{
		"axis_sizes": map[string]any{"T": timeCount, "C": channelCount, "Z": depth},
		"metadata":   map[string]any{"array_dtype": dtype},
		"phys":       map[string]any{},
		"viewer": map[string]any{
			"volume_mode":        "slice_stack",
			"render_policy":      "scalar",
			"available_surfaces": []any{"2d", "metadata", "mpr", "volume"},
		},
		"scalar_mask_capability": map[string]any{
			"version":              1,
			"source_authority":     "original",
			"source_format":        "tiff",
			"dtype":                dtype,
			"threshold_domain":     "raw",
			"threshold_foreground": "above",
			"slice_delivery":       "thresholded_png",
			"volume_delivery":      "raw_scalar",
			"volume_sampling":      "nearest",
			"channel_selection":    "single",
			"time_selection":       "single",
			"surfaces":             []any{"2d", "mpr", "volume"},
		},
	}
}

func TestInjectViewerCalibrationDefaultsRequiresMatchingSourceSHA(t *testing.T) {
	t.Parallel()

	baseCore := func() map[string]any {
		core := testScalarMaskViewerInfo(3, 1, 5, "uint8")
		core["data_semantics"] = map[string]any{
			"supported_modes": []any{"intensity", "mask"},
		}
		core["display_defaults"] = map[string]any{
			"volume_channel":          0,
			"time_index":              0,
			"scalar_render_mode":      "auto",
			"scalar_threshold_method": "otsu-256-v1",
			"scalar_threshold_value":  float64(120),
		}
		return core
	}
	selection := func(timeIndex int, threshold float64) domain.JSONMap {
		return domain.JSONMap{
			"channel":              0,
			"t":                    timeIndex,
			"render_mode":          "mask",
			"threshold_method":     "manual",
			"threshold_value":      threshold,
			"threshold_foreground": "above",
			"revision":             1,
			"threshold_provenance": domain.JSONMap{
				"method":             "otsu-256-v1",
				"value":              threshold - 1,
				"domain":             "raw",
				"foreground":         "above",
				"channel":            0,
				"t":                  timeIndex,
				"sample_scope":       "volume",
				"sample_count":       100,
				"sampling_algorithm": "scalar-profile-otsu-256-v1",
				"sampling_strategy":  "exact",
				"z_samples":          []any{0, 1, 2, 3, 4},
				"source_sha256":      "source-sha",
				"bins":               256,
			},
		}
	}
	record := resourceRecord{
		OriginalName: "source.tif",
		ContentType:  "image/tiff",
		SHA256:       "source-sha",
		Metadata: domain.JSONMap{
			"ultra_viewer_calibration_v1": domain.JSONMap{
				"version":       1,
				"source_sha256": "source-sha",
				"selections": domain.JSONMap{
					"c0:t0": selection(0, 133.5),
					"c0:t2": selection(2, 211.5),
				},
			},
		},
	}
	matching := baseCore()
	injectControlPlaneViewerFields(matching, record)
	defaults := matching["display_defaults"].(map[string]any)
	if defaults["scalar_render_mode"] != "mask" || defaults["scalar_threshold_value"] != float64(133) {
		t.Fatalf("matching calibration defaults = %#v", defaults)
	}
	calibrations, ok := matching["viewer_calibrations"].(map[string]any)
	if !ok {
		t.Fatalf("sanitized per-selection calibration map missing: %#v", matching)
	}
	selections := calibrations["selections"].(map[string]any)
	if len(selections) != 2 {
		t.Fatalf("per-selection calibration map = %#v", selections)
	}

	stale := baseCore()
	staleRecord := record
	staleRecord.SHA256 = "new-source-sha"
	injectControlPlaneViewerFields(stale, staleRecord)
	staleDefaults := stale["display_defaults"].(map[string]any)
	if staleDefaults["scalar_render_mode"] != "auto" || staleDefaults["scalar_threshold_value"] != float64(120) {
		t.Fatalf("stale calibration mutated defaults = %#v", staleDefaults)
	}
	if _, exists := stale["viewer_calibrations"]; exists {
		t.Fatalf("stale calibration map was injected: %#v", stale)
	}

	malformed := baseCore()
	malformedRecord := record
	malformedRecord.Metadata = domain.JSONMap{
		"ultra_viewer_calibration_v1": domain.JSONMap{
			"version":       1,
			"source_sha256": "source-sha",
			"selections": domain.JSONMap{
				"c0:t0": selection(0, 133.5),
				"c0:t2": domain.JSONMap{"threshold_method": "manual"},
			},
		},
	}
	injectControlPlaneViewerFields(malformed, malformedRecord)
	malformedDefaults := malformed["display_defaults"].(map[string]any)
	if malformedDefaults["scalar_render_mode"] != "mask" {
		t.Fatalf("valid calibration selection was not retained = %#v", malformedDefaults)
	}
	malformedCalibrations := malformed["viewer_calibrations"].(map[string]any)
	malformedSelections := malformedCalibrations["selections"].(map[string]any)
	if len(malformedSelections) != 1 {
		t.Fatalf("invalid calibration selection was not skipped individually: %#v", malformedSelections)
	}
}

func TestScalarMaskCapabilityFailsClosedForNonTiffOrMalformedDelivery(t *testing.T) {
	t.Parallel()

	for name, testCase := range map[string]struct {
		record resourceRecord
		mutate func(map[string]any)
	}{
		"non-tiff": {
			record: resourceRecord{OriginalName: "volume.nii", ContentType: "application/x-nifti", SHA256: "sha"},
			mutate: func(map[string]any) {},
		},
		"wrong sampling": {
			record: resourceRecord{OriginalName: "volume.tif", ContentType: "image/tiff", SHA256: "sha"},
			mutate: func(core map[string]any) {
				core["scalar_mask_capability"].(map[string]any)["volume_sampling"] = "box"
			},
		},
		"missing surface": {
			record: resourceRecord{OriginalName: "volume.tif", ContentType: "image/tiff", SHA256: "sha"},
			mutate: func(core map[string]any) {
				core["scalar_mask_capability"].(map[string]any)["surfaces"] = []any{"2d"}
			},
		},
	} {
		t.Run(name, func(t *testing.T) {
			core := testScalarMaskViewerInfo(1, 1, 3, "uint16")
			testCase.mutate(core)
			injectControlPlaneViewerFields(core, testCase.record)
			if _, exists := core["scalar_mask_capability"]; exists {
				t.Fatalf("invalid capability survived sanitization: %#v", core["scalar_mask_capability"])
			}
		})
	}
}

func TestViewerCalibrationPatchValidatesBeforeStoreAndMergesSelections(t *testing.T) {
	t.Parallel()

	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/viewerinfo" {
			http.NotFound(w, r)
			return
		}
		info := testScalarMaskViewerInfo(3, 1, 2, "uint8")
		info["display_defaults"] = map[string]any{
			"volume_channel":          0,
			"time_index":              0,
			"scalar_render_mode":      "auto",
			"scalar_threshold_method": "otsu-256-v1",
			"scalar_threshold_value":  120,
		}
		_ = json.NewEncoder(w).Encode(info)
	}))
	defer imageSvc.Close()

	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:           mem,
		UploadRoot:      t.TempDir(),
		ImageServiceURL: imageSvc.URL,
	})
	fileID := uploadNamedFileForProxyTest(t, router, "calibration.tif", testPNGBytes(t, 4, 4))
	resource, err := mem.GetResourceForUser(
		context.Background(),
		fileID,
		"field-researcher",
		"smithsonian",
	)
	if err != nil {
		t.Fatal(err)
	}

	selection := func(
		timeIndex int,
		method string,
		threshold float64,
		provenanceValue float64,
	) map[string]any {
		return map[string]any{
			"channel":              0,
			"t":                    timeIndex,
			"render_mode":          "mask",
			"threshold_method":     method,
			"threshold_value":      threshold,
			"threshold_foreground": "above",
			"expected_revision":    0,
			"threshold_provenance": map[string]any{
				"method":             "otsu-256-v1",
				"value":              provenanceValue,
				"domain":             "raw",
				"foreground":         "above",
				"channel":            0,
				"t":                  timeIndex,
				"sample_scope":       "volume",
				"sample_count":       8,
				"sampling_algorithm": "scalar-profile-otsu-256-v1",
				"sampling_strategy":  "exact",
				"z_samples":          []any{0, 1},
				"source_sha256":      resource.SHA256,
				"bins":               256,
			},
		}
	}
	patch := func(selectionKey string, selected map[string]any) *httptest.ResponseRecorder {
		body, marshalErr := json.Marshal(map[string]any{
			"metadata": map[string]any{
				"ultra_viewer_calibration_v1": map[string]any{
					"version":       1,
					"source_sha256": resource.SHA256,
					"selections": map[string]any{
						selectionKey: selected,
					},
				},
			},
		})
		if marshalErr != nil {
			t.Fatal(marshalErr)
		}
		req := httptest.NewRequest(
			http.MethodPatch,
			"/v2/resources/"+fileID,
			bytes.NewReader(body),
		)
		req.Header.Set("Content-Type", "application/json")
		setProxyOwnerHeaders(req)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		return rec
	}

	invalid := patch("c0:t0", selection(0, "otsu-256-v1", 121, 120))
	if invalid.Code != http.StatusBadRequest {
		t.Fatalf("invalid Otsu patch status=%d body=%s", invalid.Code, invalid.Body.String())
	}
	unchanged, err := mem.GetResourceForUser(
		context.Background(),
		fileID,
		"field-researcher",
		"smithsonian",
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, exists := unchanged.Metadata["ultra_viewer_calibration_v1"]; exists {
		t.Fatalf("invalid calibration mutated resource metadata: %#v", unchanged.Metadata)
	}

	if rec := patch("c0:t0", selection(0, "otsu-256-v1", 120, 120)); rec.Code != http.StatusOK {
		t.Fatalf("T0 calibration patch status=%d body=%s", rec.Code, rec.Body.String())
	}
	twoSelectionResponse := patch("c0:t2", selection(2, "manual", 230, 220))
	if twoSelectionResponse.Code != http.StatusOK {
		t.Fatalf(
			"T2 calibration patch status=%d body=%s",
			twoSelectionResponse.Code,
			twoSelectionResponse.Body.String(),
		)
	}
	var patched resourceResponse
	if err := json.Unmarshal(twoSelectionResponse.Body.Bytes(), &patched); err != nil {
		t.Fatal(err)
	}
	calibration, ok := jsonObject(patched.Resource.Metadata["ultra_viewer_calibration_v1"])
	if !ok {
		t.Fatalf("PATCH response omitted calibration: %#v", patched.Resource.Metadata)
	}
	selections, ok := jsonObject(calibration["selections"])
	if !ok || len(selections) != 2 {
		t.Fatalf("PATCH response did not preserve T0 while adding T2: %#v", calibration)
	}
	for selectionKey, rawSelection := range selections {
		saved, savedOK := jsonObject(rawSelection)
		revision, revisionOK := jsonInt(saved["revision"])
		provenance, provenanceOK := jsonObject(saved["threshold_provenance"])
		if !savedOK || !revisionOK || revision != 1 || !provenanceOK ||
			provenance["source_sha256"] != resource.SHA256 ||
			provenance["sampling_strategy"] != "exact" {
			t.Fatalf("saved selection %s is not reconstructive: %#v", selectionKey, rawSelection)
		}
		if _, exists := saved["expected_revision"]; exists {
			t.Fatalf("write-only expected revision leaked into selection %s: %#v", selectionKey, saved)
		}
	}

	viewerReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/viewer", nil)
	setProxyOwnerHeaders(viewerReq)
	viewerRec := httptest.NewRecorder()
	router.ServeHTTP(viewerRec, viewerReq)
	if viewerRec.Code != http.StatusOK {
		t.Fatalf("viewer reload status=%d body=%s", viewerRec.Code, viewerRec.Body.String())
	}
	var viewer map[string]any
	if err := json.Unmarshal(viewerRec.Body.Bytes(), &viewer); err != nil {
		t.Fatal(err)
	}
	viewerCalibration, ok := jsonObject(viewer["viewer_calibrations"])
	if !ok {
		t.Fatalf("viewer reload omitted calibration: %#v", viewer)
	}
	viewerSelections, ok := jsonObject(viewerCalibration["selections"])
	if !ok || len(viewerSelections) != 2 {
		t.Fatalf("viewer reload selections=%#v, want T0 and T2", viewerCalibration["selections"])
	}

	outOfBounds := patch("c0:t3", selection(3, "manual", 240, 220))
	if outOfBounds.Code != http.StatusBadRequest {
		t.Fatalf("out-of-bounds patch status=%d body=%s", outOfBounds.Code, outOfBounds.Body.String())
	}
}

func TestV2MaskSliceUsesOriginalSourceAndCanonicalThreshold(t *testing.T) {
	t.Parallel()

	var gotPath, gotMode, gotThreshold, gotChannels, gotTime string
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/viewerinfo":
			_ = json.NewEncoder(w).Encode(testScalarMaskViewerInfo(2, 3, 4, "uint16"))
			return
		case "/slice":
		default:
			http.NotFound(w, r)
			return
		}
		gotPath = r.URL.Query().Get("path")
		gotMode = r.URL.Query().Get("scalar_render_mode")
		gotThreshold = r.URL.Query().Get("scalar_threshold_value")
		gotChannels = r.URL.Query().Get("channels")
		gotTime = r.URL.Query().Get("t")
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write(testPNGBytes(t, 2, 2))
	}))
	defer imageSvc.Close()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:           mem,
		UploadRoot:      uploadRoot,
		ImageServiceURL: imageSvc.URL,
	})
	fileID := uploadNamedFileForProxyTest(t, router, "mask.tif", testPNGBytes(t, 4, 4))
	if err := os.MkdirAll(filepath.Join(uploadRoot, "derived"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(
		filepath.Join(uploadRoot, "derived", derivedPyramidName(fileID)),
		[]byte("display derivative"),
		0o644,
	); err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(
		http.MethodGet,
		"/v2/uploads/"+fileID+"/slice?z=1&channels=2&t=1&scalar_render_mode=mask&scalar_threshold_value=120.9&scalar_threshold_foreground=above",
		nil,
	)
	setProxyOwnerHeaders(req)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("mask slice status = %d body=%s", rec.Code, rec.Body.String())
	}
	if strings.Contains(gotPath, "__pyramid") || gotMode != "mask" || gotThreshold != "120" ||
		gotChannels != "2" || gotTime != "1" {
		t.Fatalf("mask slice path=%q mode=%q threshold=%q channels=%q t=%q", gotPath, gotMode, gotThreshold, gotChannels, gotTime)
	}

	gotMode, gotThreshold = "unreached", "unreached"
	intensityReq := httptest.NewRequest(
		http.MethodGet,
		"/v2/uploads/"+fileID+"/slice?z=1&channels=0&scalar_render_mode=intensity",
		nil,
	)
	setProxyOwnerHeaders(intensityReq)
	intensityRec := httptest.NewRecorder()
	router.ServeHTTP(intensityRec, intensityReq)
	if intensityRec.Code != http.StatusOK || gotMode != "" || gotThreshold != "" {
		t.Fatalf(
			"intensity slice status=%d forwarded_mode=%q forwarded_threshold=%q body=%s",
			intensityRec.Code,
			gotMode,
			gotThreshold,
			intensityRec.Body.String(),
		)
	}
}

func TestParseMaskSliceRequestRejectsNonCanonicalOrAmbiguousSelectors(t *testing.T) {
	t.Parallel()

	valid := httptest.NewRequest(
		http.MethodGet,
		"/slice?scalar_render_mode=mask&scalar_threshold_value=120.5&scalar_threshold_foreground=above",
		nil,
	)
	parsed, err := parseMaskSliceRequest(valid)
	if err != nil || !parsed.enabled || parsed.thresholdRaw != "120.5" {
		t.Fatalf("valid mask selectors parsed as (%+v, %v)", parsed, err)
	}
	intensity := httptest.NewRequest(
		http.MethodGet,
		"/slice?scalar_render_mode=intensity",
		nil,
	)
	if parsed, err := parseMaskSliceRequest(intensity); err != nil || parsed.enabled {
		t.Fatalf("canonical intensity selector parsed as (%+v, %v)", parsed, err)
	}

	for name, rawQuery := range map[string]string{
		"uppercase mode":         "scalar_render_mode=Mask&scalar_threshold_value=1&scalar_threshold_foreground=above",
		"unknown mode":           "scalar_render_mode=auto",
		"intensity with mask":    "scalar_render_mode=intensity&scalar_threshold_value=1",
		"repeated mode":          "scalar_render_mode=mask&scalar_render_mode=mask&scalar_threshold_value=1&scalar_threshold_foreground=above",
		"missing threshold":      "scalar_render_mode=mask&scalar_threshold_foreground=above",
		"missing foreground":     "scalar_render_mode=mask&scalar_threshold_value=1",
		"nonfinite threshold":    "scalar_render_mode=mask&scalar_threshold_value=NaN&scalar_threshold_foreground=above",
		"unknown foreground":     "scalar_render_mode=mask&scalar_threshold_value=1&scalar_threshold_foreground=below",
		"threshold without mode": "scalar_threshold_value=1&scalar_threshold_foreground=above",
		"mixed channel aliases":  "scalar_render_mode=mask&scalar_threshold_value=1&scalar_threshold_foreground=above&channel=0&channels=0",
		"comma channels":         "scalar_render_mode=mask&scalar_threshold_value=1&scalar_threshold_foreground=above&channels=0%2C1",
		"negative channel":       "scalar_render_mode=mask&scalar_threshold_value=1&scalar_threshold_foreground=above&channels=-1",
		"repeated time":          "scalar_render_mode=mask&scalar_threshold_value=1&scalar_threshold_foreground=above&t=0&t=0",
	} {
		t.Run(name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodGet, "/slice?"+rawQuery, nil)
			if _, err := parseMaskSliceRequest(request); err == nil {
				t.Fatalf("accepted query %q", rawQuery)
			}
		})
	}
}

func TestMaskSliceFailsClosedWithoutSidecarOrForNifti(t *testing.T) {
	t.Parallel()

	maskPath := "/v2/uploads/missing/slice?scalar_render_mode=mask&scalar_threshold_value=1&scalar_threshold_foreground=above"
	unconfigured := NewRouter(ServerDeps{})
	req := httptest.NewRequest(http.MethodGet, maskPath, nil)
	rec := httptest.NewRecorder()
	unconfigured.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("unconfigured mask status = %d body=%s", rec.Code, rec.Body.String())
	}

	serviceRequests := 0
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		serviceRequests++
		http.Error(w, "must not proxy NIfTI mask", http.StatusInternalServerError)
	}))
	defer imageSvc.Close()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:           mem,
		UploadRoot:      t.TempDir(),
		ImageServiceURL: imageSvc.URL,
	})
	fileID := uploadNamedFileForProxyTest(t, router, "volume.nii", []byte("nifti"))
	req = httptest.NewRequest(
		http.MethodGet,
		"/v2/uploads/"+fileID+"/slice?scalar_render_mode=mask&scalar_threshold_value=1&scalar_threshold_foreground=above",
		nil,
	)
	setProxyOwnerHeaders(req)
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnprocessableEntity || serviceRequests != 0 {
		t.Fatalf("NIfTI mask status=%d sidecar_requests=%d body=%s", rec.Code, serviceRequests, rec.Body.String())
	}
}

func TestV2UploadScalarVolumeRejectsAmbiguousOrOutOfRangeIndices(t *testing.T) {
	t.Parallel()

	scalarRequests := 0
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/viewerinfo":
			writeJSON(w, http.StatusOK, map[string]any{
				"axis_sizes": map[string]any{"T": 2, "C": 3, "Z": 8},
			})
		case "/scalar-volume":
			scalarRequests++
			http.Error(w, "must not reach scalar decoder", http.StatusInternalServerError)
		default:
			http.NotFound(w, r)
		}
	}))
	defer imageSvc.Close()

	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:           mem,
		UploadRoot:      t.TempDir(),
		ImageServiceURL: imageSvc.URL,
	})
	fileID := uploadNamedFileForProxyTest(t, router, "vol.ome.tiff", testPNGBytes(t, 4, 4))

	for _, rawQuery := range []string{
		"channel=-1",
		"channel=1.5",
		"channel=1&channel=2",
		"channel=1&c=1",
		"channel=3",
		"t=2",
		"time=1&timepoint=1",
		"t=",
	} {
		t.Run(rawQuery, func(t *testing.T) {
			req := httptest.NewRequest(
				http.MethodGet,
				"/v2/uploads/"+fileID+"/scalar-volume?"+rawQuery,
				nil,
			)
			setProxyOwnerHeaders(req)
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf(
					"query %q status = %d body=%s, want 400",
					rawQuery,
					rec.Code,
					rec.Body.String(),
				)
			}
		})
	}
	if scalarRequests != 0 {
		t.Fatalf("invalid selections reached scalar decoder %d times", scalarRequests)
	}
}

func TestConvertOnUploadEnqueuesPyramidForMicroscopy(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, bus),
		Store:           mem,
		UploadRoot:      t.TempDir(),
		DataAgentJobs:   bus,
		ImageServiceURL: "http://image-service.invalid", // only presence matters for the gate
	})

	fileID := uploadNamedFileForProxyTest(t, router, "cells.czi", testPNGBytes(t, 4, 4))

	select {
	case job := <-bus.DataAgentJobs():
		if job.JobType != "image.derive_pyramid" {
			t.Fatalf("auto job type = %q, want image.derive_pyramid", job.JobType)
		}
		if len(job.ResourceIDs) != 1 || job.ResourceIDs[0] != fileID {
			t.Fatalf("auto job resources = %v, want [%s]", job.ResourceIDs, fileID)
		}
		if job.Metadata["trigger"] != "upload" {
			t.Fatalf("auto job trigger = %v, want upload", job.Metadata["trigger"])
		}
	default:
		t.Fatal("convert-on-upload did not enqueue image.derive_pyramid for a .czi upload")
	}
}

func TestConvertOnUploadSkipsSmallPlainImage(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, bus),
		Store:           mem,
		UploadRoot:      t.TempDir(),
		DataAgentJobs:   bus,
		ImageServiceURL: "http://image-service.invalid",
	})

	_ = uploadNamedFileForProxyTest(t, router, "thumb.png", testPNGBytes(t, 4, 4))

	select {
	case job := <-bus.DataAgentJobs():
		t.Fatalf("small plain PNG should not auto-derive a pyramid, but enqueued %q", job.JobType)
	default:
		// expected: no job
	}
}

// TestV2AdminOverviewIncludesImageCacheStats verifies the operator overview surfaces
// the image response cache's hit/miss/saturation so cache effectiveness is observable.
func TestV2AdminOverviewIncludesImageCacheStats(t *testing.T) {
	t.Parallel()

	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/slice" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write([]byte("\x89PNG\r\n\x1a\nSLICE"))
	}))
	defer imageSvc.Close()

	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:           mem,
		UploadRoot:      t.TempDir(),
		ImageServiceURL: imageSvc.URL,
	})
	fileID := uploadNamedFileForProxyTest(t, router, "stack.png", testPNGBytes(t, 4, 4))

	for i := 0; i < 2; i++ { // same slice twice: a miss then a hit
		req := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/slice?axis=z&z=2", nil)
		setProxyOwnerHeaders(req)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("slice %d status = %d body=%s", i, rec.Code, rec.Body.String())
		}
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/admin/overview", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("admin overview status = %d body=%s", rec.Code, rec.Body.String())
	}
	var payload struct {
		ImageCache adminImageCacheStats `json:"image_cache"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode admin overview: %v", err)
	}
	ic := payload.ImageCache
	if !ic.Enabled || ic.MaxBytes <= 0 {
		t.Fatalf("image cache stats = %+v, want enabled with a byte budget", ic)
	}
	if ic.Hits < 1 || ic.Misses < 1 {
		t.Fatalf("image cache stats = %+v, want >=1 hit and >=1 miss after a repeated slice", ic)
	}
	if ic.HitRate <= 0 || ic.HitRate > 1 {
		t.Fatalf("image cache hit_rate = %v, want within (0,1]", ic.HitRate)
	}
}

// TestImageServiceProxyErrorMatrix locks in graceful degradation across the viewer
// serving path: for every upstream failure status, the control plane must preserve the
// status (so the frontend's retry/fallback/placeholder logic keys correctly) but replace
// the body with a clean JSON error — never leaking the internal storage path or a sidecar
// traceback. This is the format-coverage contract for corrupt / unsupported / undecodable
// / out-of-range inputs, which all surface as one of these upstream statuses.
func TestImageServiceProxyErrorMatrix(t *testing.T) {
	t.Parallel()

	const leak = "/var/lib/ultra/uploads/file_secret__scan.ome.tiff: libtiff: bad IFD; traceback ..."
	cases := []struct {
		name     string
		upstream int
	}{
		{"bad_request_out_of_range", http.StatusBadRequest},           // e.g. region/tile past the grid
		{"not_found", http.StatusNotFound},                            // missing plane/level
		{"unsupported_media", http.StatusUnsupportedMediaType},        // format the engine can't render
		{"unprocessable_undecodable", http.StatusUnprocessableEntity}, // the .czi -> 0x0 case
		{"engine_error", http.StatusInternalServerError},              // transient/persistent decode crash
		{"engine_unavailable", http.StatusServiceUnavailable},         // pool saturated / restarting
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/slice" {
					http.NotFound(w, r)
					return
				}
				http.Error(w, leak, tc.upstream)
			}))
			defer imageSvc.Close()

			mem := store.NewMemoryStore()
			router := NewRouter(ServerDeps{
				Version:         "test-version",
				Runs:            runcontrol.NewService(mem, eventbus.NewMemoryBus()),
				Store:           mem,
				UploadRoot:      t.TempDir(),
				ImageServiceURL: imageSvc.URL,
			})
			fileID := uploadNamedFileForProxyTest(t, router, "stack.png", testPNGBytes(t, 4, 4))

			req := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/slice?axis=z&z=3", nil)
			setProxyOwnerHeaders(req)
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)

			if rec.Code != tc.upstream {
				t.Fatalf("status = %d, want upstream %d preserved", rec.Code, tc.upstream)
			}
			body := rec.Body.String()
			for _, secret := range []string{"libtiff", "/var/lib", "file_secret", "traceback"} {
				if strings.Contains(body, secret) {
					t.Fatalf("leaked upstream detail %q in client body: %s", secret, body)
				}
			}
			var e map[string]string
			if err := json.Unmarshal(rec.Body.Bytes(), &e); err != nil {
				t.Fatalf("error body not clean JSON: %q (%v)", body, err)
			}
			if strings.TrimSpace(e["error"]) == "" {
				t.Fatalf("clean error body missing error field: %v", e)
			}
		})
	}
}

// TestThumbnailMapsUnreachableServiceTo503 covers a dependency-down path without
// a native fallback: when the image service can't be reached at all, the viewer gets a
// clean 503 (not a hang, raw connection error, or original file), so the frontend can show a clear
// "preview unavailable" state.
func TestThumbnailMapsUnreachableServiceTo503(t *testing.T) {
	t.Parallel()
	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:           mem,
		UploadRoot:      uploadRoot,
		ImageServiceURL: "http://127.0.0.1:1", // nothing listening
	})
	scientificSource := testOmeTIFFStackBytes(t, 2, 2, 2, 1, []string{"DAPI"}, []uint16{0, 1, 2, 3, 4, 5, 6, 7})
	fileID := uploadNamedFileForProxyTest(t, router, "stack.ome.tiff", scientificSource)
	if err := os.MkdirAll(filepath.Join(uploadRoot, "derived"), 0o755); err != nil {
		t.Fatalf("mkdir derived: %v", err)
	}
	if err := os.WriteFile(filepath.Join(uploadRoot, "derived", derivedPyramidName(fileID)), []byte("ready pyramid"), 0o600); err != nil {
		t.Fatalf("write ready derivative: %v", err)
	}
	req := httptest.NewRequest(http.MethodGet, "/v2/resources/"+fileID+"/thumbnail", nil)
	setProxyOwnerHeaders(req)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d body=%s, want 503 when the image service is unreachable", rec.Code, rec.Body.String())
	}
}

// TestImageServiceProxyHidesUpstreamErrorBody verifies the viewer serving proxy does
// not forward a raw image-service error body to the client (it can carry the internal
// storage path or a sidecar traceback). The upstream status is preserved so the
// frontend's retry/fallback still works, but the body is a clean JSON error.
func TestImageServiceProxyHidesUpstreamErrorBody(t *testing.T) {
	t.Parallel()

	const leak = "/var/lib/ultra/uploads/file_secret__scan.ome.tiff: libtiff error: bad IFD offset"
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/slice" {
			http.NotFound(w, r)
			return
		}
		http.Error(w, leak, http.StatusInternalServerError) // upstream 500 with a leaky body
	}))
	defer imageSvc.Close()

	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:           mem,
		UploadRoot:      t.TempDir(),
		ImageServiceURL: imageSvc.URL,
	})
	fileID := uploadNamedFileForProxyTest(t, router, "stack.png", testPNGBytes(t, 4, 4))

	req := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/slice?axis=z&z=3", nil)
	setProxyOwnerHeaders(req)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d body=%s, want upstream 500 preserved", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	for _, secret := range []string{leak, "libtiff", "/var/lib", "file_secret"} {
		if strings.Contains(body, secret) {
			t.Fatalf("proxy leaked upstream error detail %q in body: %s", secret, body)
		}
	}
	var e map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &e); err != nil {
		t.Fatalf("error body is not clean JSON: %q (%v)", body, err)
	}
	if strings.TrimSpace(e["error"]) == "" {
		t.Fatalf("clean error body missing error field: %v", e)
	}
}

func TestDerivationThrottleReserve(t *testing.T) {
	t.Parallel()

	thr := newDerivationThrottle(2 * time.Minute)
	base := time.Unix(1_700_000_000, 0)
	if !thr.reserve("res-a", base) {
		t.Fatal("first reserve for a resource should be allowed")
	}
	if thr.reserve("res-a", base.Add(30*time.Second)) {
		t.Fatal("a second reserve within the window should be throttled")
	}
	if !thr.reserve("res-b", base.Add(30*time.Second)) {
		t.Fatal("a different resource must not be throttled by another's reservation")
	}
	if !thr.reserve("res-a", base.Add(2*time.Minute)) {
		t.Fatal("a reserve at/after the window should be allowed again")
	}
}

// TestEnsurePyramidDerivationSelfHeals covers the durability guarantee that closes
// the silent-job-loss gap: a pyramid-eligible image with no derived pyramid (its
// upload-time enqueue was lost) must get a derive job re-enqueued at view time, but
// only once per throttle window, and never when a pyramid already exists or the
// resource is not eligible. A failed publish must not panic or wedge serving.
func TestEnsurePyramidDerivationSelfHeals(t *testing.T) {
	t.Parallel()

	newRec := func(name string) resourceRecord {
		return resourceRecord{
			FileID:       domain.NewID("file"), // unique so the package throttle is fresh
			OriginalName: name,
			ContentType:  "image/tiff",
			SizeBytes:    1 << 30,
			Principal:    principalRecord{UserID: "u", OrgID: "o"},
		}
	}

	t.Run("enqueues once then throttles", func(t *testing.T) {
		pub := &recordingDataAgentJobPublisher{}
		deps := ServerDeps{DataAgentJobs: pub, ImageServiceURL: "http://image-service.invalid"}
		rec := newRec("scan.ome.tiff")
		deps.ensurePyramidDerivation(context.Background(), t.TempDir(), rec, "/data/scan.ome.tiff", "view")
		deps.ensurePyramidDerivation(context.Background(), t.TempDir(), rec, "/data/scan.ome.tiff", "view")
		if len(pub.jobs) != 1 {
			t.Fatalf("self-heal enqueued %d jobs, want exactly 1 (throttled)", len(pub.jobs))
		}
		job := pub.jobs[0]
		if job.JobType != "image.derive_pyramid" || job.Metadata["trigger"] != "view" {
			t.Fatalf("job = type %q trigger %v, want image.derive_pyramid/view", job.JobType, job.Metadata["trigger"])
		}
		if len(job.ResourceIDs) != 1 || job.ResourceIDs[0] != rec.FileID {
			t.Fatalf("job resources = %v, want [%s]", job.ResourceIDs, rec.FileID)
		}
	})

	t.Run("skips when a derived pyramid already exists", func(t *testing.T) {
		pub := &recordingDataAgentJobPublisher{}
		deps := ServerDeps{DataAgentJobs: pub, ImageServiceURL: "http://image-service.invalid"}
		rec := newRec("scan.ome.tiff")
		root := t.TempDir()
		if err := os.MkdirAll(filepath.Join(root, "derived"), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, "derived", derivedPyramidName(rec.FileID)), []byte("p"), 0o644); err != nil {
			t.Fatal(err)
		}
		deps.ensurePyramidDerivation(context.Background(), root, rec, "/data/scan.ome.tiff", "view")
		if len(pub.jobs) != 0 {
			t.Fatalf("enqueued %d jobs for an already-derived resource, want 0", len(pub.jobs))
		}
	})

	t.Run("skips a non-eligible resource", func(t *testing.T) {
		pub := &recordingDataAgentJobPublisher{}
		deps := ServerDeps{DataAgentJobs: pub, ImageServiceURL: "http://image-service.invalid"}
		rec := newRec("note.txt")
		rec.ContentType = "text/plain"
		deps.ensurePyramidDerivation(context.Background(), t.TempDir(), rec, "/data/note.txt", "view")
		if len(pub.jobs) != 0 {
			t.Fatalf("enqueued %d jobs for a non-image resource, want 0", len(pub.jobs))
		}
	})

	t.Run("survives a publish failure without panic", func(t *testing.T) {
		pub := &recordingDataAgentJobPublisher{err: context.DeadlineExceeded}
		deps := ServerDeps{DataAgentJobs: pub, ImageServiceURL: "http://image-service.invalid"}
		rec := newRec("scan.ome.tiff")
		deps.ensurePyramidDerivation(context.Background(), t.TempDir(), rec, "/data/scan.ome.tiff", "view")
		// No assertion beyond not panicking: the publish errored, the image stays
		// viewable via the fallback path, and the failure is logged for operators.
	})
}

func TestV2UploadHistogramMicroscopyViaImageService(t *testing.T) {
	t.Parallel()

	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/viewerinfo" {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"axis_sizes": map[string]any{"T": 1, "C": 2, "Z": 3},
			})
			return
		}
		if r.URL.Path != "/histogram" {
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"bins": 8, "channel": 0, "t": 0, "scope": "volume",
			"dtype": "uint8", "sample_count": 10,
			"sampling": map[string]any{
				"algorithm": "scalar-profile-otsu-256-v1", "scope": "volume",
				"strategy": "exact", "sample_count": 10, "z_samples": []any{0, 1, 2},
			},
			"threshold": map[string]any{
				"method": "otsu-256-v1", "value": 120, "domain": "raw",
				"foreground": "above", "sample_scope": "volume", "sample_count": 10,
				"channel": 0, "t": 0, "sampling_algorithm": "scalar-profile-otsu-256-v1",
				"z_samples": []any{0, 1, 2},
			},
			"channels": []any{
				map[string]any{"index": 0, "counts": []any{1, 1, 1, 1, 1, 1, 2, 2}, "edges": []any{0, 32, 64, 96, 128, 160, 192, 224, 256}, "min": 0.0, "max": 255.0},
				map[string]any{"index": 1, "counts": []any{9, 9, 9, 9}, "min": 0.0, "max": 4095.0},
			},
		})
	}))
	defer imageSvc.Close()

	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:           mem,
		UploadRoot:      t.TempDir(),
		ImageServiceURL: imageSvc.URL,
	})
	// .czi is a microscopy container the native Go decoder cannot read.
	fileID := uploadNamedFileForProxyTest(t, router, "cells.czi", testPNGBytes(t, 4, 4))

	req := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/histogram?bins=8&channel=0&scope=volume", nil)
	setProxyOwnerHeaders(req)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("histogram status = %d body=%s", rec.Code, rec.Body.String())
	}
	var resp struct {
		Source      string         `json:"source"`
		Dtype       string         `json:"dtype"`
		Channel     int            `json:"channel"`
		Time        int            `json:"t"`
		Scope       string         `json:"scope"`
		Sampling    map[string]any `json:"sampling"`
		Threshold   map[string]any `json:"threshold"`
		SampleCount int            `json:"sample_count"`
		Histogram   struct {
			Bins  []int     `json:"bins"`
			Edges []float64 `json:"edges"`
			Min   float64   `json:"min"`
			Max   float64   `json:"max"`
		} `json:"histogram"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode histogram: %v", err)
	}
	if resp.Source != "image-service-source" {
		t.Fatalf("histogram source = %q, want image-service-source", resp.Source)
	}
	if len(resp.Histogram.Bins) != 8 || resp.Histogram.Bins[7] != 2 || resp.Histogram.Max != 255.0 {
		t.Fatalf("histogram channel-0 mapping wrong: %+v", resp.Histogram)
	}
	if resp.Dtype != "uint8" || resp.Channel != 0 || resp.Time != 0 || resp.Scope != "volume" ||
		resp.SampleCount != 10 || resp.Sampling["strategy"] != "exact" ||
		resp.Threshold["method"] != "otsu-256-v1" || len(resp.Histogram.Edges) != 9 {
		t.Fatalf("histogram provenance was not preserved: %+v", resp)
	}
}

func TestV2UploadDisplayHistogramPreservesExactScientificSelection(t *testing.T) {
	t.Parallel()

	for _, filename := range []string{"cells.czi", "cells.ome.tiff"} {
		filename := filename
		t.Run(filename, func(t *testing.T) {
			t.Parallel()

			var gotPath, gotScope, gotChannels, gotTime string
			imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/viewerinfo":
					_ = json.NewEncoder(w).Encode(map[string]any{
						"axis_sizes": map[string]any{"T": 2, "C": 3, "Z": 4},
						"viewer":     map[string]any{"render_policy": "scalar"},
					})
				case "/histogram":
					gotPath = r.URL.Query().Get("path")
					gotScope = r.URL.Query().Get("scope")
					gotChannels = r.URL.Query().Get("channels")
					gotTime = r.URL.Query().Get("t")
					_ = json.NewEncoder(w).Encode(map[string]any{
						"bins": 8, "dtype": "uint16", "scope": "display", "t": 1,
						"channels": []any{
							map[string]any{
								"index": 0, "sample_count": 8,
								"counts": []any{1, 1, 1, 1, 1, 1, 1, 1},
								"edges":  []any{0, 1, 2, 3, 4, 5, 6, 7, 8},
								"min":    0, "max": 7,
							},
							map[string]any{
								"index": 2, "sample_count": 8,
								"counts": []any{1, 1, 1, 1, 1, 1, 1, 1},
								"edges":  []any{0, 1, 2, 3, 4, 5, 6, 7, 8},
								"min":    0, "max": 7,
							},
						},
					})
				default:
					http.NotFound(w, r)
				}
			}))
			defer imageSvc.Close()

			mem := store.NewMemoryStore()
			router := NewRouter(ServerDeps{
				Version:         "test-version",
				Runs:            runcontrol.NewService(mem, eventbus.NewMemoryBus()),
				Store:           mem,
				UploadRoot:      t.TempDir(),
				ImageServiceURL: imageSvc.URL,
			})
			fileID := uploadNamedFileForProxyTest(t, router, filename, testPNGBytes(t, 4, 4))
			req := httptest.NewRequest(
				http.MethodGet,
				"/v2/uploads/"+fileID+"/histogram?bins=8&channels=0,2&t=1",
				nil,
			)
			setProxyOwnerHeaders(req)
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)
			if rec.Code != http.StatusOK {
				t.Fatalf("display histogram status=%d body=%s", rec.Code, rec.Body.String())
			}
			if gotPath == "" || gotScope != "display" || gotChannels != "0,2" || gotTime != "1" {
				t.Fatalf(
					"image-service selection path=%q scope=%q channels=%q t=%q",
					gotPath,
					gotScope,
					gotChannels,
					gotTime,
				)
			}
			var response map[string]any
			if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
				t.Fatal(err)
			}
			channels, channelsOK := jsonNonNegativeIntSlice(response["channels"])
			histogram, histogramOK := jsonObject(response["histogram"])
			histogramChannels, histogramChannelsOK := jsonNonNegativeIntSlice(
				histogram["channel_indices"],
			)
			timeIndex, timeOK := jsonInt(histogram["time_index"])
			if !channelsOK || !histogramOK || !histogramChannelsOK ||
				!slices.Equal(channels, []int{0, 2}) ||
				!slices.Equal(histogramChannels, []int{0, 2}) ||
				!timeOK || timeIndex != 1 {
				t.Fatalf("display histogram lost exact identity: %#v", response)
			}
		})
	}
}

func TestMapImageServiceDisplayHistogramRejectsIdentityMismatch(t *testing.T) {
	t.Parallel()

	valid := func() map[string]any {
		return map[string]any{
			"bins": 2, "dtype": "uint16", "scope": "display", "t": 1,
			"channels": []any{
				map[string]any{
					"index": 0, "sample_count": 2,
					"counts": []any{1, 1}, "edges": []any{0, 1, 2},
					"min": 0, "max": 1,
				},
				map[string]any{
					"index": 2, "sample_count": 2,
					"counts": []any{1, 1}, "edges": []any{0, 1, 2},
					"min": 0, "max": 1,
				},
			},
		}
	}
	if _, err := mapImageServiceDisplayHistogram(valid(), "file", 2, []int{0, 2}, 1); err != nil {
		t.Fatalf("valid exact display identity rejected: %v", err)
	}
	for name, mutate := range map[string]func(map[string]any){
		"wrong time": func(core map[string]any) {
			core["t"] = 0
		},
		"reordered channels": func(core map[string]any) {
			channels := core["channels"].([]any)
			core["channels"] = []any{channels[1], channels[0]}
		},
	} {
		t.Run(name, func(t *testing.T) {
			core := valid()
			mutate(core)
			if _, err := mapImageServiceDisplayHistogram(
				core,
				"file",
				2,
				[]int{0, 2},
				1,
			); err == nil {
				t.Fatalf("accepted %s", name)
			}
		})
	}
}

func TestParseExactHistogramChannelRejectsRepeatedLegacySelector(t *testing.T) {
	t.Parallel()

	req := httptest.NewRequest(http.MethodGet, "/histogram?channels=0&channels=1", nil)
	if _, err := parseExactHistogramChannel(req); err == nil {
		t.Fatal("repeated histogram channels selector was accepted")
	}
}

func TestMapImageServiceHistogramRejectsMalformedScientificEvidence(t *testing.T) {
	t.Parallel()

	valid := func() map[string]any {
		return map[string]any{
			"bins": 2, "channel": 0, "t": 2, "scope": "volume",
			"dtype": "uint16", "sample_count": 4,
			"sampling": map[string]any{
				"algorithm": "scalar-profile-otsu-256-v1", "scope": "volume",
				"strategy": "exact", "sample_count": 4, "z_samples": []any{0, 1, 2},
			},
			"threshold": map[string]any{
				"method": "otsu-256-v1", "value": 120, "domain": "raw",
				"foreground": "above", "sample_scope": "volume", "sample_count": 4,
				"channel": 0, "t": 2, "sampling_algorithm": "scalar-profile-otsu-256-v1",
				"z_samples": []any{0, 1, 2},
			},
			"channels": []any{
				map[string]any{
					"index": 0, "counts": []any{1, 3}, "edges": []any{0, 128, 256},
					"min": 0, "max": 255,
				},
			},
		}
	}
	encoded, err := json.Marshal(valid())
	if err != nil {
		t.Fatal(err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if _, err := mapImageServiceHistogram(decoded, "file", "source-sha", 0, 2, 2); err != nil {
		t.Fatalf("valid evidence rejected: %v", err)
	}

	for name, mutate := range map[string]func(map[string]any){
		"fractional count": func(core map[string]any) {
			core["channels"].([]any)[0].(map[string]any)["counts"] = []any{1.5, 2.5}
		},
		"sample mismatch": func(core map[string]any) {
			core["channels"].([]any)[0].(map[string]any)["counts"] = []any{1, 2}
		},
		"nonmonotonic edges": func(core map[string]any) {
			core["channels"].([]any)[0].(map[string]any)["edges"] = []any{0, 128, 127}
		},
		"nonfinite extrema": func(core map[string]any) {
			core["channels"].([]any)[0].(map[string]any)["max"] = math.Inf(1)
		},
		"wrong threshold domain": func(core map[string]any) {
			core["threshold"].(map[string]any)["domain"] = "normalized"
		},
		"wrong threshold selection": func(core map[string]any) {
			core["threshold"].(map[string]any)["t"] = 1
		},
		"missing sampling algorithm": func(core map[string]any) {
			delete(core["sampling"].(map[string]any), "algorithm")
		},
		"stratified sampling with exact scope": func(core map[string]any) {
			core["sampling"].(map[string]any)["strategy"] = "stratified-z-spatial"
		},
	} {
		t.Run(name, func(t *testing.T) {
			core := valid()
			mutate(core)
			if _, err := mapImageServiceHistogram(core, "file", "source-sha", 0, 2, 2); err == nil {
				t.Fatalf("accepted malformed %s evidence", name)
			}
		})
	}

	stratified := valid()
	stratified["sampling"].(map[string]any)["strategy"] = "stratified-z-spatial"
	stratified["threshold"].(map[string]any)["sample_scope"] = "stratified_z"
	if _, err := mapImageServiceHistogram(stratified, "file", "source-sha", 0, 2, 2); err != nil {
		t.Fatalf("matching stratified sampling provenance rejected: %v", err)
	}
}

func TestV2UploadHistogramUsesOriginalSourceImageService(t *testing.T) {
	t.Parallel()

	var gotPath string
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/viewerinfo" {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"axis_sizes": map[string]any{"T": 1, "C": 1, "Z": 2},
			})
			return
		}
		if r.URL.Path != "/histogram" {
			http.NotFound(w, r)
			return
		}
		gotPath = r.URL.Query().Get("path")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"bins": 8, "channel": 0, "t": 0, "scope": "volume",
			"dtype": "uint8", "sample_count": 10,
			"sampling": map[string]any{
				"algorithm": "scalar-profile-otsu-256-v1", "scope": "volume",
				"strategy": "exact", "sample_count": 10, "z_samples": []any{0, 1},
			},
			"threshold": map[string]any{
				"method": "otsu-256-v1", "value": 120, "domain": "raw",
				"foreground": "above", "sample_scope": "volume", "sample_count": 10,
				"channel": 0, "t": 0, "sampling_algorithm": "scalar-profile-otsu-256-v1",
				"z_samples": []any{0, 1},
			},
			"channels": []any{
				map[string]any{"index": 0, "counts": []any{1, 1, 1, 1, 1, 1, 2, 2}, "edges": []any{0, 32, 64, 96, 128, 160, 192, 224, 256}, "min": 0.0, "max": 255.0},
			},
		})
	}))
	defer imageSvc.Close()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:           mem,
		UploadRoot:      uploadRoot,
		ImageServiceURL: imageSvc.URL,
	})
	fileID := uploadNamedFileForProxyTest(t, router, "large.tif", testPNGBytes(t, 4, 4))

	derivedDir := filepath.Join(uploadRoot, "derived")
	if err := os.MkdirAll(derivedDir, 0o755); err != nil {
		t.Fatalf("mkdir derived: %v", err)
	}
	pyramidPath := filepath.Join(derivedDir, derivedPyramidName(fileID))
	if err := os.WriteFile(pyramidPath, []byte("PYRAMID-BYTES"), 0o644); err != nil {
		t.Fatalf("write pyramid: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/histogram?bins=8&scope=volume", nil)
	setProxyOwnerHeaders(req)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("histogram status = %d body=%s", rec.Code, rec.Body.String())
	}
	if strings.Contains(gotPath, "__pyramid.tif") {
		t.Fatalf("histogram served from display derivative %q", gotPath)
	}
}

func TestV2UploadViewerSliceStackDoesNotMergeDerivedTileScheme(t *testing.T) {
	t.Parallel()

	// A z-stack derives to an OME-BigTIFF whose -tile reader is broken; its
	// viewer-info must keep volume_mode=slice_stack (3D via /atlas, 2D via /slice)
	// and NOT adopt the derived pyramid's tile_scheme (which would route the
	// viewer to the failing deferred-multiscale tile path).
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/viewerinfo" {
			http.NotFound(w, r)
			return
		}
		if strings.Contains(r.URL.Query().Get("path"), "__pyramid.tif") {
			// The derived OME-BigTIFF is pyramidal, so it DOES report a tile_scheme.
			_ = json.NewEncoder(w).Encode(map[string]any{
				"viewer":      map[string]any{"tile_scheme": map[string]any{"tile_size": 512, "levels": []any{}}},
				"tile_scheme": map[string]any{"tile_size": 512, "levels": []any{}},
			})
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"kind": "image", "modality": "microscopy", "backend_mode": "direct",
			"axis_sizes": map[string]any{"T": 1, "C": 7, "Z": 80, "Y": 624, "X": 924},
			"is_volume":  true, "tile_scheme": nil,
			"viewer": map[string]any{
				"volume_mode":   "slice_stack",
				"delivery_mode": "direct",
				"atlas_scheme":  map[string]any{"slice_count": 80, "columns": 9, "rows": 9},
			},
		})
	}))
	defer imageSvc.Close()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:           mem,
		UploadRoot:      uploadRoot,
		ImageServiceURL: imageSvc.URL,
	})
	fileID := uploadNamedFileForProxyTest(t, router, "zstack.ome.tiff", testPNGBytes(t, 4, 4))

	derivedDir := filepath.Join(uploadRoot, "derived")
	if err := os.MkdirAll(derivedDir, 0o755); err != nil {
		t.Fatalf("mkdir derived: %v", err)
	}
	if err := os.WriteFile(filepath.Join(derivedDir, derivedPyramidName(fileID)), []byte("OME-BIGTIFF"), 0o644); err != nil {
		t.Fatalf("write pyramid: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/viewer", nil)
	setProxyOwnerHeaders(req)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("viewer status = %d body=%s", rec.Code, rec.Body.String())
	}
	var vi map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &vi); err != nil {
		t.Fatalf("decode viewer: %v", err)
	}
	if vi["tile_scheme"] != nil {
		t.Fatalf("slice_stack viewer adopted a tile_scheme from the derived pyramid: %v", vi["tile_scheme"])
	}
	viewer, _ := vi["viewer"].(map[string]any)
	if viewer["volume_mode"] != "slice_stack" {
		t.Fatalf("volume_mode = %v, want slice_stack (must stay atlas/slice based)", viewer["volume_mode"])
	}
	if viewer["tile_scheme"] != nil {
		t.Fatalf("viewer.tile_scheme leaked from derived pyramid: %v", viewer["tile_scheme"])
	}
	if viewer["atlas_scheme"] == nil {
		t.Fatalf("viewer.atlas_scheme dropped; 3D volume source needs it")
	}
}

func TestV2TilesPreferDerivedPyramid(t *testing.T) {
	t.Parallel()

	var gotPath string
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/tile" {
			gotPath = r.URL.Query().Get("path")
		}
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write([]byte("\x89PNG\r\n\x1a\nX"))
	}))
	defer imageSvc.Close()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:           mem,
		UploadRoot:      uploadRoot,
		ImageServiceURL: imageSvc.URL,
	})
	fileID := uploadNamedFileForProxyTest(t, router, "slide.png", testPNGBytes(t, 4, 4))

	// Simulate a completed pyramid derivation on disk.
	derivedDir := filepath.Join(uploadRoot, "derived")
	if err := os.MkdirAll(derivedDir, 0o755); err != nil {
		t.Fatalf("mkdir derived: %v", err)
	}
	pyramidPath := filepath.Join(derivedDir, derivedPyramidName(fileID))
	if err := os.WriteFile(pyramidPath, []byte("PYRAMID-BYTES"), 0o644); err != nil {
		t.Fatalf("write pyramid: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/tiles/z/0/0/0", nil)
	setProxyOwnerHeaders(req)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("tile status = %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(gotPath, "__pyramid.tif") {
		t.Fatalf("tile served from %q, want the derived pyramid", gotPath)
	}
}

func TestResourceThumbnailRoutesByFormat(t *testing.T) {
	t.Parallel()

	thumbBytes := testPNGBytes(t, 7, 5)
	posterBytes := testPNGBytes(t, 6, 4)
	var gotThumbPath, gotPosterPath string
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/thumbnail":
			gotThumbPath = r.URL.Query().Get("path")
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write(thumbBytes)
		case "/video-poster":
			gotPosterPath = r.URL.Query().Get("path")
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write(posterBytes)
		default:
			http.NotFound(w, r)
		}
	}))
	defer imageSvc.Close()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:           mem,
		UploadRoot:      uploadRoot,
		ImageServiceURL: imageSvc.URL,
	})
	getThumb := func(id string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodGet, "/v2/resources/"+id+"/thumbnail", nil)
		setProxyOwnerHeaders(req)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		return rec
	}

	// Valid multi-plane scientific container -> ready derived pyramid -> libbioimage thumbnail.
	scientificSource := testOmeTIFFStackBytes(t, 2, 2, 2, 1, []string{"DAPI"}, []uint16{0, 1, 2, 3, 4, 5, 6, 7})
	cziID := uploadNamedFileForProxyTest(t, router, "cells.ome.tiff", scientificSource)
	if err := os.MkdirAll(filepath.Join(uploadRoot, "derived"), 0o755); err != nil {
		t.Fatalf("mkdir derived thumbnails: %v", err)
	}
	if err := os.WriteFile(filepath.Join(uploadRoot, "derived", derivedPyramidName(cziID)), []byte("ready pyramid"), 0o600); err != nil {
		t.Fatalf("write derived thumbnail: %v", err)
	}
	cziRec := getThumb(cziID)
	if cziRec.Code != http.StatusOK || !bytes.Equal(cziRec.Body.Bytes(), thumbBytes) {
		t.Fatalf("czi thumbnail not served from image service: code=%d", cziRec.Code)
	}
	if !strings.Contains(gotThumbPath, derivedPyramidName(cziID)) {
		t.Fatalf("image service thumbnail path = %q, want ready derivative %q", gotThumbPath, derivedPyramidName(cziID))
	}

	// Video -> server-side ffmpeg poster frame (not the whole file, not 415).
	vidID := uploadNamedFileForProxyTest(t, router, "clip.mp4", []byte("\x00\x00\x00\x18ftypmp42____moovdata"))
	vidRec := getThumb(vidID)
	if vidRec.Code != http.StatusOK || !bytes.Equal(vidRec.Body.Bytes(), posterBytes) {
		t.Fatalf("video thumbnail not served from /video-poster: code=%d", vidRec.Code)
	}
	if !strings.Contains(gotPosterPath, vidID) {
		t.Fatalf("video poster path = %q, want resolved path with %q", gotPosterPath, vidID)
	}

	// Common web image -> fast native path, always re-encoded as a bounded PNG.
	originalPNG := testPNGBytes(t, 1024, 256)
	pngID := uploadNamedFileForProxyTest(t, router, "plain.png", originalPNG)
	pngRec := getThumb(pngID)
	if pngRec.Code != http.StatusOK || bytes.Equal(pngRec.Body.Bytes(), thumbBytes) || bytes.Equal(pngRec.Body.Bytes(), originalPNG) {
		t.Fatalf("png thumbnail should use the native path, not the image service")
	}
	config, err := png.DecodeConfig(bytes.NewReader(pngRec.Body.Bytes()))
	if err != nil || config.Width != 512 || config.Height != 128 {
		t.Fatalf("bounded PNG thumbnail config = %+v err=%v, want 512x128", config, err)
	}
}

func TestLegacyResourceThumbnailCapabilityMatchesImmediateEndpoint(t *testing.T) {
	t.Parallel()
	thumbnail := testPNGBytes(t, 9, 6)
	var servedPath string
	imageService := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		servedPath = r.URL.Query().Get("path")
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write(thumbnail)
	}))
	defer imageService.Close()

	root := t.TempDir()
	stack := testOmeTIFFStackBytes(t, 2, 2, 2, 1, []string{"DAPI"}, []uint16{0, 1, 2, 3, 4, 5, 6, 7})
	if err := os.WriteFile(filepath.Join(root, "file_ready__stack.ome.tiff"), stack, 0o600); err != nil {
		t.Fatalf("write ready scientific source: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "file_unready__stack.ome.tiff"), stack, 0o600); err != nil {
		t.Fatalf("write unready scientific source: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(root, "derived"), 0o755); err != nil {
		t.Fatalf("mkdir derived: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "derived", derivedPyramidName("file_ready")), []byte("ready pyramid"), 0o600); err != nil {
		t.Fatalf("write ready derivative: %v", err)
	}
	router := NewRouter(ServerDeps{Version: "test-version", UploadRoot: root, ImageServiceURL: imageService.URL})

	listRequest := httptest.NewRequest(http.MethodGet, "/v2/resources", nil)
	listResponse := httptest.NewRecorder()
	router.ServeHTTP(listResponse, listRequest)
	if listResponse.Code != http.StatusOK {
		t.Fatalf("legacy resource list status = %d body=%s", listResponse.Code, listResponse.Body.String())
	}
	var listed resourcesResponse
	if err := json.Unmarshal(listResponse.Body.Bytes(), &listed); err != nil {
		t.Fatalf("decode legacy resources: %v", err)
	}
	byID := make(map[string]resourceRecord, len(listed.Resources))
	for _, resource := range listed.Resources {
		byID[resource.FileID] = resource
	}
	ready := byID["file_ready"]
	if !ready.HasThumbnail || ready.ThumbnailURL == "" || ready.ThumbnailInteraction != "z_scrub" {
		t.Fatalf("ready legacy thumbnail metadata = %+v", ready)
	}
	if unready := byID["file_unready"]; unready.HasThumbnail || unready.ThumbnailURL != "" {
		t.Fatalf("scientific source without derivative was advertised: %+v", unready)
	}
	unreadyRequest := httptest.NewRequest(http.MethodGet, "/v2/resources/file_unready/thumbnail", nil)
	unreadyResponse := httptest.NewRecorder()
	router.ServeHTTP(unreadyResponse, unreadyRequest)
	if unreadyResponse.Code != http.StatusUnsupportedMediaType || strings.Contains(unreadyResponse.Body.String(), root) {
		t.Fatalf("unready scientific thumbnail = %d/%q, want sanitized 415", unreadyResponse.Code, unreadyResponse.Body.String())
	}

	thumbnailRequest := httptest.NewRequest(http.MethodGet, ready.ThumbnailURL, nil)
	thumbnailResponse := httptest.NewRecorder()
	router.ServeHTTP(thumbnailResponse, thumbnailRequest)
	if thumbnailResponse.Code != http.StatusOK || thumbnailResponse.Header().Get("Content-Type") != "image/png" {
		t.Fatalf("advertised thumbnail endpoint = %d/%q body=%s", thumbnailResponse.Code, thumbnailResponse.Header().Get("Content-Type"), thumbnailResponse.Body.String())
	}
	if err := validateThumbnailPNG(thumbnailResponse.Header().Get("Content-Type"), thumbnailResponse.Body.Bytes()); err != nil {
		t.Fatalf("advertised thumbnail is not bounded PNG: %v", err)
	}
	if !strings.Contains(servedPath, derivedPyramidName("file_ready")) {
		t.Fatalf("legacy endpoint served %q, want ready derivative", servedPath)
	}
}

func TestResourceThumbnailLargeRasterFallsThroughOnlyToReadyDerivative(t *testing.T) {
	t.Parallel()
	thumbnail := testPNGBytes(t, 8, 8)
	var servedPath string
	imageService := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		servedPath = r.URL.Query().Get("path")
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write(thumbnail)
	}))
	defer imageService.Close()

	root := t.TempDir()
	sourcePath := filepath.Join(root, "file_large__large.png")
	source, err := os.OpenFile(sourcePath, os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatalf("create large raster: %v", err)
	}
	if _, err := source.Write(testPNGBytes(t, 2, 2)); err != nil {
		_ = source.Close()
		t.Fatalf("write large raster header: %v", err)
	}
	if err := source.Truncate(rasterThumbnailMaxInput + 1); err != nil {
		_ = source.Close()
		t.Fatalf("truncate large raster: %v", err)
	}
	if err := source.Close(); err != nil {
		t.Fatalf("close large raster: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(root, "derived"), 0o755); err != nil {
		t.Fatalf("mkdir derived: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "derived", derivedPyramidName("file_large")), []byte("ready pyramid"), 0o600); err != nil {
		t.Fatalf("write ready derivative: %v", err)
	}
	router := NewRouter(ServerDeps{Version: "test-version", UploadRoot: root, ImageServiceURL: imageService.URL})
	request := httptest.NewRequest(http.MethodGet, "/v2/resources/file_large/thumbnail", nil)
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("large raster thumbnail status = %d body=%s", response.Code, response.Body.String())
	}
	if !strings.Contains(servedPath, derivedPyramidName("file_large")) {
		t.Fatalf("large raster sidecar path = %q, want ready derivative", servedPath)
	}
}

func TestResourceCiftiThumbnailEndpoint(t *testing.T) {
	t.Parallel()
	_, fixture := writeCifti2Fixture(t, "route.dtseries.nii", 220, 300, func(row, col int) float32 {
		return float32(math.Sin(float64(col)/9) + float64(row%11))
	})
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version: "test-version", Runs: runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store: mem, UploadRoot: t.TempDir(),
	})
	fileID := uploadNamedFileForProxyTest(t, router, "route.dtseries.nii", fixture)
	request := func() *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodGet, "/v2/resources/"+fileID+"/thumbnail", nil)
		setProxyOwnerHeaders(req)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		return rec
	}
	first := request()
	second := request()
	if first.Code != http.StatusOK || first.Header().Get("Content-Type") != "image/png" {
		t.Fatalf("CIFTI thumbnail status/type = %d/%q body=%s", first.Code, first.Header().Get("Content-Type"), first.Body.String())
	}
	config, err := png.DecodeConfig(bytes.NewReader(first.Body.Bytes()))
	if err != nil || config.Width != 512 || config.Height != 288 {
		t.Fatalf("CIFTI endpoint PNG config = %+v err=%v", config, err)
	}
	if !bytes.Equal(first.Body.Bytes(), second.Body.Bytes()) {
		t.Fatal("repeated CIFTI endpoint output is not byte-identical")
	}

	var compressed bytes.Buffer
	writer := gzip.NewWriter(&compressed)
	if _, err := writer.Write(fixture); err != nil {
		t.Fatalf("gzip CIFTI fixture: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("close gzip CIFTI fixture: %v", err)
	}
	gzipID := uploadNamedFileForProxyTest(t, router, "route.dtseries.nii.gz", compressed.Bytes())
	gzipReq := httptest.NewRequest(http.MethodGet, "/v2/resources/"+gzipID+"/thumbnail", nil)
	setProxyOwnerHeaders(gzipReq)
	gzipRec := httptest.NewRecorder()
	router.ServeHTTP(gzipRec, gzipReq)
	if gzipRec.Code != http.StatusUnsupportedMediaType {
		t.Fatalf("gzip CIFTI thumbnail status = %d body=%s, want 415", gzipRec.Code, gzipRec.Body.String())
	}
}

func TestRenderNiftiThumbnailBoundsPlaneAndOutput(t *testing.T) {
	t.Parallel()
	width, height := 1024, 256
	values := make([]uint16, width*height)
	for index := range values {
		values[index] = uint16(index % 4096)
	}
	path := filepath.Join(t.TempDir(), "wide-brain.nii")
	if err := os.WriteFile(path, testNifti1Uint16Bytes(t, width, height, 1, values), 0o600); err != nil {
		t.Fatalf("write NIfTI thumbnail fixture: %v", err)
	}
	body, err := renderNiftiThumbnailPNG(path, httptest.NewRequest(http.MethodGet, "/thumbnail", nil), newByteAdmissionBudget(defaultThumbnailInFlightBytes))
	if err != nil {
		t.Fatalf("render NIfTI thumbnail: %v", err)
	}
	if int64(len(body)) > thumbnailMaxEncodedBytes {
		t.Fatalf("NIfTI thumbnail encoded bytes = %d, want <= %d", len(body), thumbnailMaxEncodedBytes)
	}
	config, err := png.DecodeConfig(bytes.NewReader(body))
	if err != nil || config.Width != 512 || config.Height != 128 {
		t.Fatalf("NIfTI thumbnail config = %+v err=%v, want 512x128", config, err)
	}

	header := testNifti1Uint16Bytes(t, 1, 1, 1, []uint16{0})[:niftiHeaderReadSize]
	oversizedAxis := append([]byte(nil), header...)
	binary.LittleEndian.PutUint16(oversizedAxis[42:44], uint16(niftiThumbnailMaxAxis+1))
	axisPath := filepath.Join(t.TempDir(), "oversized-axis.nii")
	if err := os.WriteFile(axisPath, oversizedAxis, 0o600); err != nil {
		t.Fatalf("write oversized-axis NIfTI: %v", err)
	}
	if _, err := niftiThumbnailPreflight(axisPath); err == nil {
		t.Fatal("oversized NIfTI axis passed thumbnail preflight")
	}

	oversizedPlane := append([]byte(nil), header...)
	binary.LittleEndian.PutUint16(oversizedPlane[42:44], 8192)
	binary.LittleEndian.PutUint16(oversizedPlane[44:46], 8192)
	planePath := filepath.Join(t.TempDir(), "oversized-plane.nii")
	if err := os.WriteFile(planePath, oversizedPlane, 0o600); err != nil {
		t.Fatalf("write oversized-plane NIfTI: %v", err)
	}
	if _, err := niftiThumbnailPreflight(planePath); err == nil {
		t.Fatal("oversized NIfTI plane passed thumbnail preflight")
	}
}

func TestRenderNiftiThumbnailSupportsBoundedGzipAndRejectsDeclaredWork(t *testing.T) {
	t.Parallel()
	values := make([]uint16, 64*48*16)
	for index := range values {
		values[index] = uint16(index % 1024)
	}
	var validCompressed bytes.Buffer
	validWriter := gzip.NewWriter(&validCompressed)
	if _, err := validWriter.Write(testNifti1Uint16Bytes(t, 64, 48, 16, values)); err != nil {
		t.Fatalf("gzip valid NIfTI: %v", err)
	}
	if err := validWriter.Close(); err != nil {
		t.Fatalf("close valid NIfTI gzip: %v", err)
	}
	validPath := filepath.Join(t.TempDir(), "brain.nii.gz")
	if err := os.WriteFile(validPath, validCompressed.Bytes(), 0o600); err != nil {
		t.Fatalf("write valid gzip NIfTI: %v", err)
	}
	if _, err := renderNiftiThumbnailPNG(validPath, httptest.NewRequest(http.MethodGet, "/thumbnail", nil), newByteAdmissionBudget(defaultThumbnailInFlightBytes)); err != nil {
		t.Fatalf("bounded gzip NIfTI thumbnail failed: %v", err)
	}

	header := append([]byte(nil), testNifti1Uint16Bytes(t, 1, 1, 1, []uint16{0})[:niftiHeaderReadSize]...)
	binary.LittleEndian.PutUint16(header[42:44], 512)
	binary.LittleEndian.PutUint16(header[44:46], 512)
	binary.LittleEndian.PutUint16(header[46:48], 2048)
	var rejectedCompressed bytes.Buffer
	rejectedWriter := gzip.NewWriter(&rejectedCompressed)
	if _, err := rejectedWriter.Write(header); err != nil {
		t.Fatalf("gzip rejected NIfTI header: %v", err)
	}
	if err := rejectedWriter.Close(); err != nil {
		t.Fatalf("close rejected NIfTI gzip: %v", err)
	}
	rejectedPath := filepath.Join(t.TempDir(), "too-much-work.nii.gz")
	if err := os.WriteFile(rejectedPath, rejectedCompressed.Bytes(), 0o600); err != nil {
		t.Fatalf("write rejected gzip NIfTI: %v", err)
	}
	if _, err := niftiThumbnailPreflight(rejectedPath); err == nil {
		t.Fatal("gzip NIfTI exceeding declared thumbnail work budget passed preflight")
	}
}

func TestNiftiThumbnailIgnoresLegacyDecompressedSidecar(t *testing.T) {
	t.Parallel()
	payload := testNifti1Uint16Bytes(t, 2, 1, 1, []uint16{0, 100})
	var compressed bytes.Buffer
	writer := gzip.NewWriter(&compressed)
	if _, err := writer.Write(payload); err != nil {
		t.Fatalf("gzip NIfTI fixture: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("close NIfTI gzip fixture: %v", err)
	}
	path := filepath.Join(t.TempDir(), "authenticated.nii.gz")
	if err := os.WriteFile(path, compressed.Bytes(), 0o600); err != nil {
		t.Fatalf("write gzip NIfTI fixture: %v", err)
	}
	if err := os.MkdirAll(filepath.Dir(niftiDecompressedSidecarPath(path)), 0o755); err != nil {
		t.Fatalf("mkdir stale sidecar directory: %v", err)
	}
	if err := os.WriteFile(niftiDecompressedSidecarPath(path), []byte("stale untrusted sidecar"), 0o600); err != nil {
		t.Fatalf("write stale sidecar: %v", err)
	}
	if readyDecompressedNiftiSidecar(path) == "" {
		t.Fatal("test setup did not produce a legacy-ready sidecar")
	}
	plan, err := niftiThumbnailPreflight(path)
	if err != nil {
		t.Fatalf("preflight authenticated gzip source: %v", err)
	}
	if plan.path != path || !plan.gzipped {
		t.Fatalf("thumbnail plan selected path=%q gzipped=%v, want authenticated gzip source", plan.path, plan.gzipped)
	}
	if _, err := renderNiftiThumbnailPNG(path, httptest.NewRequest(http.MethodGet, "/thumbnail", nil), newByteAdmissionBudget(defaultThumbnailInFlightBytes)); err != nil {
		t.Fatalf("render authenticated gzip source with stale sidecar present: %v", err)
	}
}

func TestNiftiThumbnailGzipAdmissionIncludesDeepDecompressionWork(t *testing.T) {
	t.Parallel()
	header := append([]byte(nil), testNifti1Uint16Bytes(t, 1, 1, 1, []uint16{0})[:niftiHeaderReadSize]...)
	binary.LittleEndian.PutUint16(header[46:48], 30_000)
	var compressed bytes.Buffer
	writer := gzip.NewWriter(&compressed)
	if _, err := writer.Write(header); err != nil {
		t.Fatalf("gzip deep NIfTI header: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("close deep NIfTI gzip: %v", err)
	}
	path := filepath.Join(t.TempDir(), "deep-tiny-plane.nii.gz")
	if err := os.WriteFile(path, compressed.Bytes(), 0o600); err != nil {
		t.Fatalf("write deep NIfTI gzip: %v", err)
	}
	plan, err := niftiThumbnailPreflight(path)
	if err != nil {
		t.Fatalf("preflight deep NIfTI gzip: %v", err)
	}
	baseReservation := plan.planeBytes + thumbnailDecodeBytesPerPixel
	if plan.decompressionWork <= 0 || plan.reservation <= baseReservation {
		t.Fatalf("gzip admission plan = %+v, want decompression work charged", plan)
	}
	budget := newByteAdmissionBudget(plan.reservation)
	if !budget.tryAcquire(plan.decompressionWork) {
		t.Fatal("occupy gzip decompression portion of thumbnail budget")
	}
	defer budget.release(plan.decompressionWork)
	if _, err := renderNiftiThumbnailPNG(path, httptest.NewRequest(http.MethodGet, "/thumbnail", nil), budget); !errors.Is(err, errThumbnailAdmission) {
		t.Fatalf("deep gzip thumbnail error = %v, want admission rejection before truncated plane work", err)
	}
}

func TestRenderNiftiThumbnailAppliesPhysicalWindow(t *testing.T) {
	t.Parallel()
	payload := testNifti1Int16Bytes(t, 2, 1, 1, []int16{1024, 1104})
	binary.LittleEndian.PutUint32(payload[112:116], math.Float32bits(1))
	binary.LittleEndian.PutUint32(payload[116:120], math.Float32bits(-1024))
	path := filepath.Join(t.TempDir(), "ct.nii")
	if err := os.WriteFile(path, payload, 0o600); err != nil {
		t.Fatalf("write CT NIfTI: %v", err)
	}
	request := httptest.NewRequest(http.MethodGet, "/thumbnail?enhancement=hounsfield:40:80", nil)
	body, err := renderNiftiThumbnailPNG(path, request, newByteAdmissionBudget(defaultThumbnailInFlightBytes))
	if err != nil {
		t.Fatalf("render CT NIfTI thumbnail: %v", err)
	}
	decoded, err := png.Decode(bytes.NewReader(body))
	if err != nil {
		t.Fatalf("decode CT NIfTI thumbnail: %v", err)
	}
	if low, high := color.GrayModel.Convert(decoded.At(0, 0)).(color.Gray).Y, color.GrayModel.Convert(decoded.At(1, 0)).(color.Gray).Y; low != 0 || high != 255 {
		t.Fatalf("physical CT window pixels = %d/%d, want 0/255", low, high)
	}

	bigEndian := make([]byte, niftiHeaderReadSize+4)
	binary.BigEndian.PutUint32(bigEndian[0:4], nifti1HeaderSize)
	binary.BigEndian.PutUint16(bigEndian[40:42], 3)
	binary.BigEndian.PutUint16(bigEndian[42:44], 2)
	binary.BigEndian.PutUint16(bigEndian[44:46], 1)
	binary.BigEndian.PutUint16(bigEndian[46:48], 1)
	binary.BigEndian.PutUint16(bigEndian[70:72], 4)
	binary.BigEndian.PutUint16(bigEndian[72:74], 16)
	binary.BigEndian.PutUint32(bigEndian[108:112], math.Float32bits(float32(niftiHeaderReadSize)))
	copy(bigEndian[344:348], []byte{'n', '+', '1', 0})
	binary.BigEndian.PutUint16(bigEndian[niftiHeaderReadSize:niftiHeaderReadSize+2], 0)
	binary.BigEndian.PutUint16(bigEndian[niftiHeaderReadSize+2:niftiHeaderReadSize+4], 100)
	bigEndianPath := filepath.Join(t.TempDir(), "big-endian.nii")
	if err := os.WriteFile(bigEndianPath, bigEndian, 0o600); err != nil {
		t.Fatalf("write big-endian NIfTI: %v", err)
	}
	bigEndianBody, err := renderNiftiThumbnailPNG(bigEndianPath, httptest.NewRequest(http.MethodGet, "/thumbnail", nil), newByteAdmissionBudget(defaultThumbnailInFlightBytes))
	if err != nil {
		t.Fatalf("render big-endian NIfTI thumbnail: %v", err)
	}
	bigEndianImage, err := png.Decode(bytes.NewReader(bigEndianBody))
	if err != nil {
		t.Fatalf("decode big-endian NIfTI thumbnail: %v", err)
	}
	if low, high := color.GrayModel.Convert(bigEndianImage.At(0, 0)).(color.Gray).Y, color.GrayModel.Convert(bigEndianImage.At(1, 0)).(color.Gray).Y; low != 0 || high != 255 {
		t.Fatalf("big-endian NIfTI pixels = %d/%d, want 0/255", low, high)
	}
}

func TestRenderNiftiThumbnailPreservesNegativeSlopeContrast(t *testing.T) {
	t.Parallel()
	payload := testNifti1Int16Bytes(t, 2, 1, 1, []int16{0, 100})
	binary.LittleEndian.PutUint32(payload[112:116], math.Float32bits(-1))
	binary.LittleEndian.PutUint32(payload[116:120], math.Float32bits(100))
	path := filepath.Join(t.TempDir(), "negative-slope.nii")
	if err := os.WriteFile(path, payload, 0o600); err != nil {
		t.Fatalf("write negative-slope NIfTI: %v", err)
	}
	body, err := renderNiftiThumbnailPNG(path, httptest.NewRequest(http.MethodGet, "/thumbnail", nil), newByteAdmissionBudget(defaultThumbnailInFlightBytes))
	if err != nil {
		t.Fatalf("render negative-slope NIfTI thumbnail: %v", err)
	}
	decoded, err := png.Decode(bytes.NewReader(body))
	if err != nil {
		t.Fatalf("decode negative-slope thumbnail: %v", err)
	}
	first := color.GrayModel.Convert(decoded.At(0, 0)).(color.Gray).Y
	second := color.GrayModel.Convert(decoded.At(1, 0)).(color.Gray).Y
	if first != 255 || second != 0 {
		t.Fatalf("negative-slope pixels = %d/%d, want physical high/low = 255/0", first, second)
	}
}

func TestDerivedPyramidPathRejectsSymlinks(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	derived := filepath.Join(root, "derived")
	if err := os.MkdirAll(derived, 0o755); err != nil {
		t.Fatalf("mkdir derived directory: %v", err)
	}
	target := filepath.Join(root, "outside-ready-pyramid.tif")
	if err := os.WriteFile(target, []byte("target"), 0o600); err != nil {
		t.Fatalf("write symlink target: %v", err)
	}
	link := filepath.Join(derived, derivedPyramidName("file_link"))
	if err := os.Symlink(target, link); err != nil {
		t.Fatalf("create derived pyramid symlink: %v", err)
	}
	if got := derivedPyramidPath(root, "file_link"); got != "" {
		t.Fatalf("derivedPyramidPath accepted symlink %q", got)
	}
	regular := filepath.Join(derived, derivedPyramidName("file_regular"))
	if err := os.WriteFile(regular, []byte("ready"), 0o600); err != nil {
		t.Fatalf("write regular derivative: %v", err)
	}
	resolvedRegular, err := filepath.EvalSymlinks(regular)
	if err != nil {
		t.Fatalf("resolve regular derivative: %v", err)
	}
	if got := derivedPyramidPath(root, "file_regular"); got != resolvedRegular {
		t.Fatalf("derivedPyramidPath regular file = %q, want %q", got, resolvedRegular)
	}
}

func TestThumbnailAdmissionRejectsConcurrentDecodedBytes(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "admission.nii")
	if err := os.WriteFile(path, testNifti1Uint16Bytes(t, 2, 2, 1, []uint16{0, 1, 2, 3}), 0o600); err != nil {
		t.Fatalf("write NIfTI: %v", err)
	}
	plan, err := niftiThumbnailPreflight(path)
	if err != nil {
		t.Fatalf("preflight NIfTI: %v", err)
	}
	budget := newByteAdmissionBudget(plan.reservation)
	if !budget.tryAcquire(plan.reservation) {
		t.Fatal("reserve simulated concurrent thumbnail")
	}
	if _, err := renderNiftiThumbnailPNG(path, httptest.NewRequest(http.MethodGet, "/thumbnail", nil), budget); !errors.Is(err, errThumbnailAdmission) {
		t.Fatalf("concurrent thumbnail error = %v, want admission rejection", err)
	}
	budget.release(plan.reservation)
	if _, err := renderNiftiThumbnailPNG(path, httptest.NewRequest(http.MethodGet, "/thumbnail", nil), budget); err != nil {
		t.Fatalf("thumbnail did not recover after reservation release: %v", err)
	}

	rasterPath := filepath.Join(t.TempDir(), "admission.png")
	if err := os.WriteFile(rasterPath, testPNGBytes(t, 4, 4), 0o600); err != nil {
		t.Fatalf("write raster: %v", err)
	}
	rasterReservation := int64(4*4) * thumbnailDecodeBytesPerPixel
	rasterBudget := newByteAdmissionBudget(rasterReservation)
	if !rasterBudget.tryAcquire(rasterReservation) {
		t.Fatal("reserve simulated concurrent raster thumbnail")
	}
	if _, err := renderBoundedRasterThumbnailPNG(rasterPath, rasterBudget); !errors.Is(err, errThumbnailAdmission) {
		t.Fatalf("concurrent raster thumbnail error = %v, want admission rejection", err)
	}
	rasterBudget.release(rasterReservation)
	if _, err := renderBoundedRasterThumbnailPNG(rasterPath, rasterBudget); err != nil {
		t.Fatalf("raster thumbnail did not recover after reservation release: %v", err)
	}
}

func TestResourceThumbnailAuthorizesBeforeSidecar(t *testing.T) {
	t.Parallel()
	var sidecarCalls atomic.Int64
	sidecarPNG := testPNGBytes(t, 4, 4)
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sidecarCalls.Add(1)
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write(sidecarPNG)
	}))
	defer imageSvc.Close()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version: "test-version", Runs: runcontrol.NewService(mem, eventbus.NewMemoryBus()), Store: mem,
		UploadRoot: t.TempDir(), ImageServiceURL: imageSvc.URL,
	})
	fileID := uploadNamedFileForProxyTest(t, router, "private.czi", []byte("private-scientific-source"))
	req := httptest.NewRequest(http.MethodGet, "/v2/resources/"+fileID+"/thumbnail", nil)
	req.Header.Set("X-Ultra-User-Id", "different-user")
	req.Header.Set("X-Ultra-Org-Id", "smithsonian")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("unauthorized thumbnail status = %d body=%s, want 404", rec.Code, rec.Body.String())
	}
	if calls := sidecarCalls.Load(); calls != 0 {
		t.Fatalf("sidecar invoked %d times before ownership rejection", calls)
	}
}

func TestResourceThumbnailRejectsUnsupportedAndInvalidSidecarResponses(t *testing.T) {
	t.Parallel()
	t.Run("unsupported never returns source", func(t *testing.T) {
		mem := store.NewMemoryStore()
		router := NewRouter(ServerDeps{
			Version: "test-version", Runs: runcontrol.NewService(mem, eventbus.NewMemoryBus()),
			Store: mem, UploadRoot: t.TempDir(),
		})
		source := []byte("%PDF-source-must-not-be-served")
		fileID := uploadNamedFileForProxyTest(t, router, "report.pdf", source)
		req := httptest.NewRequest(http.MethodGet, "/v2/resources/"+fileID+"/thumbnail", nil)
		setProxyOwnerHeaders(req)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != http.StatusUnsupportedMediaType || bytes.Contains(rec.Body.Bytes(), source) {
			t.Fatalf("unsupported thumbnail status/body = %d/%q, want clean 415", rec.Code, rec.Body.Bytes())
		}
	})
	validPNG := testPNGBytes(t, 4, 4)
	for _, tc := range []struct {
		name        string
		contentType string
		body        []byte
	}{
		{name: "wrong type", contentType: "application/octet-stream", body: validPNG},
		{name: "malformed png", contentType: "image/png", body: validPNG[:len(validPNG)-4]},
		{name: "oversized", contentType: "image/png", body: bytes.Repeat([]byte{'x'}, int(thumbnailMaxEncodedBytes)+1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", tc.contentType)
				_, _ = w.Write(tc.body)
			}))
			defer imageSvc.Close()
			uploadRoot := t.TempDir()
			mem := store.NewMemoryStore()
			router := NewRouter(ServerDeps{
				Version: "test-version", Runs: runcontrol.NewService(mem, eventbus.NewMemoryBus()), Store: mem,
				UploadRoot: uploadRoot, ImageServiceURL: imageSvc.URL,
			})
			scientificSource := testOmeTIFFStackBytes(t, 2, 2, 2, 1, []string{"DAPI"}, []uint16{0, 1, 2, 3, 4, 5, 6, 7})
			fileID := uploadNamedFileForProxyTest(t, router, "cells.ome.tiff", scientificSource)
			if err := os.MkdirAll(filepath.Join(uploadRoot, "derived"), 0o755); err != nil {
				t.Fatalf("mkdir derived thumbnails: %v", err)
			}
			if err := os.WriteFile(filepath.Join(uploadRoot, "derived", derivedPyramidName(fileID)), []byte("ready pyramid"), 0o600); err != nil {
				t.Fatalf("write derived thumbnail: %v", err)
			}
			req := httptest.NewRequest(http.MethodGet, "/v2/resources/"+fileID+"/thumbnail", nil)
			setProxyOwnerHeaders(req)
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)
			if rec.Code != http.StatusServiceUnavailable || bytes.Equal(rec.Body.Bytes(), scientificSource) {
				t.Fatalf("invalid sidecar response status/body = %d/%q, want clean 503", rec.Code, rec.Body.Bytes())
			}
		})
	}
}

func TestShouldDerivePyramidMatrix(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		rec  resourceRecord
		want bool
	}{
		{"tiff", resourceRecord{OriginalName: "scan.tif", ContentType: "image/tiff"}, true},
		{"ome-tiff", resourceRecord{OriginalName: "scan.ome.tiff"}, true},
		{"czi", resourceRecord{OriginalName: "cells.czi"}, true},
		{"nd2", resourceRecord{OriginalName: "movie.nd2"}, true},
		{"nifti-skip", resourceRecord{OriginalName: "brain.nii.gz", ContentType: "application/x-nifti"}, false},
		{"small-png", resourceRecord{OriginalName: "i.png", ContentType: "image/png", SizeBytes: 1024}, false},
		{"large-png", resourceRecord{OriginalName: "huge.png", ContentType: "image/png", SizeBytes: 32 << 20}, true},
		{"pdf-skip", resourceRecord{OriginalName: "report.pdf", ContentType: "application/pdf"}, false},
	}
	for _, tc := range cases {
		if got := shouldDerivePyramid(tc.rec); got != tc.want {
			t.Errorf("shouldDerivePyramid(%s) = %v, want %v", tc.name, got, tc.want)
		}
	}
}
