package httpapi

import (
	"bytes"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

func setProxyOwnerHeaders(req *http.Request) {
	req.Header.Set("X-Ultra-User-Id", "field-researcher")
	req.Header.Set("X-Ultra-Org-Id", "smithsonian")
	req.Header.Set("X-Ultra-Role", "admin")
}

func uploadImageForProxyTest(t *testing.T, router http.Handler) string {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("files", "slide.png")
	if err != nil {
		t.Fatalf("CreateFormFile: %v", err)
	}
	if _, err := part.Write(testPNGBytes(t, 4, 4)); err != nil {
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

func TestV2UploadTilesAndAtlasProxyImageService(t *testing.T) {
	t.Parallel()

	wantPNG := []byte("\x89PNG\r\n\x1a\nFAKE")
	var gotTilePath, gotAtlasPath, gotLevel, gotCol, gotRow string
	var gotChannels, gotChannelColors, gotCacheKey, gotTileT, gotTileZ string
	var gotAtlasChannels, gotAtlasColors, gotAtlasT string
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/tile":
			q := r.URL.Query()
			gotTilePath, gotLevel, gotCol, gotRow = q.Get("path"), q.Get("level"), q.Get("col"), q.Get("row")
			gotChannels, gotChannelColors, gotCacheKey = q.Get("channels"), q.Get("channel_colors"), q.Get("cache_key")
			gotTileT, gotTileZ = q.Get("t"), q.Get("z")
		case "/atlas":
			q := r.URL.Query()
			gotAtlasPath = q.Get("path")
			gotAtlasChannels, gotAtlasColors, gotAtlasT = q.Get("channels"), q.Get("channel_colors"), q.Get("t")
		default:
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write(wantPNG)
	}))
	defer imageSvc.Close()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	service := runcontrol.NewService(mem, eventbus.NewMemoryBus())
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            service,
		Store:           mem,
		UploadRoot:      uploadRoot,
		ImageServiceURL: imageSvc.URL,
	})

	fileID := uploadImageForProxyTest(t, router)

	// Authorized tile request is proxied with the resolved path + tile coordinates.
	tileReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/tiles/z/2/3/4?size=256&t=0&z=0&channels=2%2C1%2C0&channel_colors=%230000ff%2C%2300ff00%2C%23ff0000&cache_key=channels-v1", nil)
	setProxyOwnerHeaders(tileReq)
	tileRec := httptest.NewRecorder()
	router.ServeHTTP(tileRec, tileReq)
	if tileRec.Code != http.StatusOK {
		t.Fatalf("tile status = %d body=%s", tileRec.Code, tileRec.Body.String())
	}
	if ct := tileRec.Header().Get("Content-Type"); ct != "image/png" {
		t.Fatalf("tile content-type = %q, want image/png", ct)
	}
	if !bytes.Equal(tileRec.Body.Bytes(), wantPNG) {
		t.Fatalf("tile body was not proxied from the image service")
	}
	if gotLevel != "2" || gotCol != "3" || gotRow != "4" {
		t.Fatalf("tile coords level=%s col=%s row=%s, want 2/3/4", gotLevel, gotCol, gotRow)
	}
	if gotChannels != "2,1,0" || gotChannelColors != "#0000ff,#00ff00,#ff0000" || gotCacheKey != "channels-v1" || gotTileT != "0" || gotTileZ != "0" {
		t.Fatalf("tile identity channels=%q colors=%q cache_key=%q", gotChannels, gotChannelColors, gotCacheKey)
	}
	if !strings.Contains(gotTilePath, fileID) {
		t.Fatalf("tile path = %q, want resolved storage path containing %q", gotTilePath, fileID)
	}

	// Authorized atlas request is proxied.
	atlasReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/atlas?level=2&grid_rows=4&grid_cols=5&t=0&channels=1&channel_colors=%2300ff00", nil)
	setProxyOwnerHeaders(atlasReq)
	atlasRec := httptest.NewRecorder()
	router.ServeHTTP(atlasRec, atlasReq)
	if atlasRec.Code != http.StatusOK {
		t.Fatalf("atlas status = %d body=%s", atlasRec.Code, atlasRec.Body.String())
	}
	if !bytes.Equal(atlasRec.Body.Bytes(), wantPNG) {
		t.Fatalf("atlas body was not proxied from the image service")
	}
	if !strings.Contains(gotAtlasPath, fileID) {
		t.Fatalf("atlas path = %q, want resolved storage path containing %q", gotAtlasPath, fileID)
	}
	if gotAtlasT != "0" || gotAtlasChannels != "1" || gotAtlasColors != "#00ff00" {
		t.Fatalf("atlas identity t=%q channels=%q colors=%q", gotAtlasT, gotAtlasChannels, gotAtlasColors)
	}

	// Ownership is enforced before the proxy: a different user gets 404.
	intruderReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/tiles/z/0/0/0", nil)
	intruderReq.Header.Set("X-Ultra-User-Id", "intruder")
	intruderReq.Header.Set("X-Ultra-Org-Id", "smithsonian")
	intruderRec := httptest.NewRecorder()
	router.ServeHTTP(intruderRec, intruderReq)
	if intruderRec.Code != http.StatusNotFound {
		t.Fatalf("intruder tile status = %d, want 404", intruderRec.Code)
	}
}

func TestV2UploadTilesAndSlicesForwardChannelIdentityToNgffService(t *testing.T) {
	t.Parallel()

	type identity struct {
		channels, colors, cacheKey, time, z, rawQuery string
	}
	got := map[string]identity{}
	ngffSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tile" && r.URL.Path != "/slice" {
			http.NotFound(w, r)
			return
		}
		query := r.URL.Query()
		got[r.URL.Path] = identity{
			channels: query.Get("channels"),
			colors:   query.Get("channel_colors"),
			cacheKey: query.Get("cache_key"),
			time:     query.Get("t"),
			z:        query.Get("z"),
			rawQuery: r.URL.RawQuery,
		}
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write([]byte("\x89PNG\r\n\x1a\nNGFF"))
	}))
	defer ngffSvc.Close()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:        "test-version",
		Runs:           runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:          mem,
		UploadRoot:     uploadRoot,
		NgffServiceURL: ngffSvc.URL,
	})
	fileID := uploadNamedFileForProxyTest(t, router, "scan.ome.zarr", []byte(`{"multiscales":[]}`))

	for _, endpoint := range []string{
		"/tiles/z/0/0/0?t=3&z=7",
		"/slice?axis=z&t=3&z=7",
	} {
		separator := "?"
		if strings.Contains(endpoint, "?") {
			separator = "&"
		}
		req := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+endpoint+separator+"channels=5%2C1%2C259&channel_colors=%230000ff%2C%2300ff00%2C%23ff0000&cache_key=channels-v1", nil)
		setProxyOwnerHeaders(req)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("%s status = %d body=%s", endpoint, rec.Code, rec.Body.String())
		}
	}
	for endpoint, identity := range got {
		if identity.channels != "5,1,259" || identity.colors != "#0000ff,#00ff00,#ff0000" || identity.cacheKey != "channels-v1" || identity.time != "3" || identity.z != "7" {
			t.Fatalf("ngff %s identity channels=%q colors=%q cache_key=%q raw_query=%q", endpoint, identity.channels, identity.colors, identity.cacheKey, identity.rawQuery)
		}
	}
	if len(got) != 2 {
		t.Fatalf("ngff endpoints observed = %v, want tile and slice", got)
	}
}

func TestV2UploadImageProxiesRejectMalformedSelectorsBeforeUpstream(t *testing.T) {
	t.Parallel()

	upstreamCalls := 0
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		upstreamCalls++
		w.WriteHeader(http.StatusOK)
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
		NgffServiceURL:  imageSvc.URL,
	})
	plainID := uploadImageForProxyTest(t, router)
	ngffID := uploadNamedFileForProxyTest(t, router, "scan.ome.zarr", []byte(`{"multiscales":[]}`))

	cases := []struct {
		name        string
		requestPath string
	}{
		{name: "libbio tile empty", requestPath: "/v2/uploads/" + plainID + "/tiles/z/0/0/0?channels="},
		{name: "libbio slice whitespace", requestPath: "/v2/uploads/" + plainID + "/slice?channels=%20%20"},
		{name: "ngff tile duplicate", requestPath: "/v2/uploads/" + ngffID + "/tiles/z/0/0/0?channels=0&channels=1"},
		{name: "ngff slice malformed", requestPath: "/v2/uploads/" + ngffID + "/slice?channels=1.5"},
		{name: "atlas duplicate t", requestPath: "/v2/uploads/" + plainID + "/atlas?t=0&t=1"},
		{name: "tile duplicate z", requestPath: "/v2/uploads/" + plainID + "/tiles/z/0/0/0?z=0&z=1"},
		{name: "libbio tile x axis", requestPath: "/v2/uploads/" + plainID + "/tiles/x/0/0/0"},
		{name: "ngff tile y axis", requestPath: "/v2/uploads/" + ngffID + "/tiles/y/0/0/0"},
		{name: "too many channels", requestPath: "/v2/uploads/" + plainID + "/slice?channels=0,1,2,3,4,5,6,7,8"},
		{name: "lut cardinality mismatch", requestPath: "/v2/uploads/" + plainID + "/atlas?channels=0,1&channel_colors=%23ffffff"},
		{name: "out of range", requestPath: "/v2/uploads/" + plainID + "/slice?channels=999"},
		{name: "mixed channel aliases", requestPath: "/v2/uploads/" + plainID + "/slice?channels=0&c=0"},
		{name: "duplicate channel alias", requestPath: "/v2/uploads/" + plainID + "/slice?channel=0&channel=1"},
		{name: "singular channel alias list", requestPath: "/v2/uploads/" + plainID + "/slice?c=0,1"},
		{name: "slice axis cannot be flattened", requestPath: "/v2/uploads/" + plainID + "/slice?axis=x&x=1"},
		{name: "slice x cannot be ignored", requestPath: "/v2/uploads/" + plainID + "/slice?x=1"},
		{name: "slice y cannot be ignored", requestPath: "/v2/uploads/" + plainID + "/slice?y=1"},
		{name: "atlas duplicate level", requestPath: "/v2/uploads/" + plainID + "/atlas?level=0&level=1"},
		{name: "atlas negative level", requestPath: "/v2/uploads/" + plainID + "/atlas?level=-1"},
		{name: "atlas zero rows", requestPath: "/v2/uploads/" + plainID + "/atlas?grid_rows=0&grid_cols=1"},
		{name: "atlas incomplete grid", requestPath: "/v2/uploads/" + plainID + "/atlas?grid_rows=2"},
		{name: "tile duplicate size", requestPath: "/v2/uploads/" + plainID + "/tiles/z/0/0/0?size=128&size=256"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, tc.requestPath, nil)
			setProxyOwnerHeaders(req)
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)
			if rec.Code != http.StatusUnprocessableEntity {
				t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
			}
		})
	}
	if upstreamCalls != 0 {
		t.Fatalf("malformed selectors reached an image sidecar %d times", upstreamCalls)
	}

	unauthorized := httptest.NewRequest(
		http.MethodGet,
		"/v2/uploads/"+plainID+"/slice?channels=0&channels=1",
		nil,
	)
	unauthorized.Header.Set("X-Ultra-User-Id", "intruder")
	unauthorizedRec := httptest.NewRecorder()
	router.ServeHTTP(unauthorizedRec, unauthorized)
	if unauthorizedRec.Code != http.StatusNotFound {
		t.Fatalf("unauthorized malformed selector status = %d, want hidden 404", unauthorizedRec.Code)
	}
}

func TestV2OwnedMalformedViewerSelectorsWinBeforeUnavailableStorage(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	fileID := uploadImageForProxyTest(t, router)
	matches, err := filepath.Glob(filepath.Join(uploadRoot, fileID+"__*"))
	if err != nil || len(matches) != 1 {
		t.Fatalf("resolve uploaded source: matches=%v err=%v", matches, err)
	}
	if err := os.Remove(matches[0]); err != nil {
		t.Fatalf("make authorized source unavailable: %v", err)
	}

	for _, requestPath := range []string{
		"/v2/uploads/" + fileID + "/tiles/x/0/0/0",
		"/v2/uploads/" + fileID + "/slice?channels=not-an-index",
		"/v2/uploads/" + fileID + "/atlas?grid_rows=0&grid_cols=1",
	} {
		req := httptest.NewRequest(http.MethodGet, requestPath, nil)
		setProxyOwnerHeaders(req)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != http.StatusUnprocessableEntity {
			t.Fatalf("owned malformed request %q status = %d body=%s", requestPath, rec.Code, rec.Body.String())
		}

		unauthorized := httptest.NewRequest(http.MethodGet, requestPath, nil)
		unauthorized.Header.Set("X-Ultra-User-Id", "intruder")
		unauthorizedRec := httptest.NewRecorder()
		router.ServeHTTP(unauthorizedRec, unauthorized)
		if unauthorizedRec.Code != http.StatusNotFound {
			t.Fatalf("unauthorized malformed request %q status = %d, want hidden 404", requestPath, unauthorizedRec.Code)
		}
	}
}

func TestScientificImageSelectorsCanonicalizeAliasesAndEnforceCatalogAxes(t *testing.T) {
	t.Parallel()

	record := resourceRecord{Metadata: domain.JSONMap{
		"image_header": domain.JSONMap{
			"time_count":    2,
			"channel_count": 4,
			"depth":         3,
			"warnings":      []any{},
		},
	}}
	selectors, err := parseScientificImageSelectors(
		map[string][]string{"t": {"1"}, "z": {"2"}, "channel": {"3"}},
		record,
	)
	if err != nil {
		t.Fatalf("parse selectors: %v", err)
	}
	forwarded := make(url.Values)
	selectors.apply(forwarded)
	if got := forwarded.Get("channels"); got != "3" {
		t.Fatalf("canonical channels = %q, want 3", got)
	}
	if _, present := forwarded["channel"]; present {
		t.Fatalf("noncanonical channel alias was forwarded: %v", forwarded)
	}

	for _, query := range []map[string][]string{
		{"t": {"2"}},
		{"z": {"3"}},
		{"c": {"4"}},
		{"c": {"1"}, "channel": {"1"}},
	} {
		if _, err := parseScientificImageSelectors(query, record); err == nil {
			t.Fatalf("selectors %v were admitted", query)
		}
	}
}

func TestV2UploadTilesRejectUnboundedTileSizesBeforeProxy(t *testing.T) {
	t.Parallel()

	proxyCalls := 0
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		proxyCalls++
		w.WriteHeader(http.StatusOK)
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
	fileID := uploadImageForProxyTest(t, router)

	for _, rawSize := range []string{"", "0", "-1", "1025", "bad"} {
		req := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/tiles/z/0/0/0?size="+rawSize, nil)
		setProxyOwnerHeaders(req)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != http.StatusUnprocessableEntity {
			t.Fatalf("size %q status = %d body=%s", rawSize, rec.Code, rec.Body.String())
		}
	}
	if proxyCalls != 0 {
		t.Fatalf("invalid tile sizes reached upstream %d times", proxyCalls)
	}
}

func TestV2UploadTilesNotConfiguredWithoutImageService(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	service := runcontrol.NewService(mem, eventbus.NewMemoryBus())
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       service,
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	fileID := uploadImageForProxyTest(t, router)

	for _, path := range []string{"/v2/uploads/" + fileID + "/tiles/z/0/0/0", "/v2/uploads/" + fileID + "/atlas"} {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		setProxyOwnerHeaders(req)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != http.StatusNotImplemented {
			t.Fatalf("%s status = %d, want 501 not configured", path, rec.Code)
		}
	}
}

func TestV2UploadSliceFailsClosedWithoutScientificService(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	photoID := uploadImageForProxyTest(t, router)
	scientificID := uploadNamedFileForProxyTest(t, router, "scan.ome.tif", []byte("scientific"))

	for _, requestPath := range []string{
		"/v2/uploads/" + photoID + "/slice?t=0",
		"/v2/uploads/" + scientificID + "/slice",
	} {
		req := httptest.NewRequest(http.MethodGet, requestPath, nil)
		setProxyOwnerHeaders(req)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != http.StatusNotImplemented {
			t.Fatalf("%s status = %d body=%s, want 501", requestPath, rec.Code, rec.Body.String())
		}
	}

	request := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+photoID+"/slice", nil)
	setProxyOwnerHeaders(request)
	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK || !bytes.HasPrefix(recorder.Body.Bytes(), []byte("\x89PNG")) {
		t.Fatalf("selector-free photo fallback status=%d body=%q", recorder.Code, recorder.Body.Bytes())
	}
}

func TestV2DeriveUploadPyramidEnqueuesJob(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{
		Version:       "test-version",
		Runs:          service,
		Store:         mem,
		UploadRoot:    uploadRoot,
		DataAgentJobs: bus,
	})
	fileID := uploadImageForProxyTest(t, router)

	req := httptest.NewRequest(http.MethodPost, "/v2/uploads/"+fileID+"/derive-pyramid", nil)
	setProxyOwnerHeaders(req)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusAccepted {
		t.Fatalf("derive-pyramid status = %d body=%s, want 202", rec.Code, rec.Body.String())
	}
	var resp struct {
		JobID   string `json:"job_id"`
		JobType string `json:"job_type"`
		Status  string `json:"status"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode derive-pyramid response: %v", err)
	}
	if resp.JobType != "image.derive_pyramid" || resp.JobID == "" || resp.Status != "queued" {
		t.Fatalf("derive-pyramid response = %+v", resp)
	}

	select {
	case job := <-bus.DataAgentJobs():
		if job.JobType != "image.derive_pyramid" {
			t.Fatalf("published job type = %q, want image.derive_pyramid", job.JobType)
		}
		if len(job.ResourceIDs) != 1 || job.ResourceIDs[0] != fileID {
			t.Fatalf("published resource ids = %v, want [%s]", job.ResourceIDs, fileID)
		}
		if job.Metadata["src_path"] == nil || job.Metadata["dst_path"] == nil {
			t.Fatalf("published job missing src/dst paths: %v", job.Metadata)
		}
		if sha, ok := job.Metadata["source_sha256"].(string); !ok || len(sha) != 64 {
			t.Fatalf("published job missing catalog source sha256: %v", job.Metadata)
		}
		if size, ok := job.Metadata["source_size_bytes"].(int64); !ok || size <= 0 {
			t.Fatalf("published job missing catalog source size: %v", job.Metadata)
		}
		if force, ok := job.Metadata["force"].(bool); !ok || !force {
			t.Fatalf("explicit derive must force regeneration: %v", job.Metadata)
		}
	default:
		t.Fatal("no image.derive_pyramid job was published")
	}
}

func TestV2DeriveUploadPyramidNotConfigured(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	service := runcontrol.NewService(mem, eventbus.NewMemoryBus())
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       service,
		Store:      mem,
		UploadRoot: uploadRoot,
		// DataAgentJobs intentionally nil -> 501 not configured
	})
	fileID := uploadImageForProxyTest(t, router)

	req := httptest.NewRequest(http.MethodPost, "/v2/uploads/"+fileID+"/derive-pyramid", nil)
	setProxyOwnerHeaders(req)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("derive-pyramid (no queue) status = %d, want 501", rec.Code)
	}
}

func TestV2DeriveUploadPyramidAuthorizesBeforeQueueReadiness(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
		// A missing queue must not be observable across the ownership boundary.
	})
	fileID := uploadImageForProxyTest(t, router)
	req := httptest.NewRequest(http.MethodPost, "/v2/uploads/"+fileID+"/derive-pyramid", nil)
	req.Header.Set("X-Ultra-User-Id", "intruder")
	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, req)
	if recorder.Code != http.StatusNotFound {
		t.Fatalf("unauthorized derive status=%d body=%s, want hidden 404", recorder.Code, recorder.Body.String())
	}
}

func TestByteAdmissionBudgetRejectsAggregateInFlightOverflow(t *testing.T) {
	t.Parallel()

	budget := newByteAdmissionBudget(10)
	if !budget.tryAcquire(6) {
		t.Fatal("first in-flight body was rejected")
	}
	if budget.tryAcquire(5) {
		t.Fatal("aggregate overflow was admitted")
	}
	if !budget.tryAcquire(4) {
		t.Fatal("request fitting the remaining aggregate budget was rejected")
	}
	if budget.tryAcquire(1) {
		t.Fatal("request was admitted while the aggregate budget was full")
	}
	budget.release(6)
	if !budget.tryAcquire(5) {
		t.Fatal("released aggregate capacity was not reusable")
	}
	budget.release(4)
	budget.release(5)
	if budget.used != 0 {
		t.Fatalf("budget used bytes = %d, want 0", budget.used)
	}
}
