package httpapi

import (
	"bytes"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

// hdf5ProxyRoutes maps each control-plane HDF5 route suffix to the image-service
// endpoint it must proxy to. Shared by the tests below so route coverage stays in
// one place.
var hdf5ProxyRoutes = map[string]string{
	"/hdf5/dataset":               "/hdf5/dataset",
	"/hdf5/preview/slice":         "/hdf5/preview/slice",
	"/hdf5/preview/atlas":         "/hdf5/preview/atlas",
	"/hdf5/preview/scalar-volume": "/hdf5/preview/scalar-volume",
	"/hdf5/preview/histogram":     "/hdf5/preview/histogram",
	"/hdf5/preview/table":         "/hdf5/preview/table",
}

// uploadHdf5ForProxyTest uploads a small fake .h5 blob and returns its file ID.
// The Go proxy never parses HDF5 content, so arbitrary bytes suffice — only the
// Python image service needs a real file.
func uploadHdf5ForProxyTest(t *testing.T, router http.Handler) string {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("files", "sample.h5")
	if err != nil {
		t.Fatalf("CreateFormFile: %v", err)
	}
	// HDF5 superblock magic + filler so the blob is honest about its format.
	if _, err := part.Write(append([]byte("\x89HDF\r\n\x1a\n"), bytes.Repeat([]byte{0}, 64)...)); err != nil {
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

func newHdf5ProxyRouter(t *testing.T, imageServiceURL string) http.Handler {
	t.Helper()
	mem := store.NewMemoryStore()
	return NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:           mem,
		UploadRoot:      t.TempDir(),
		ImageServiceURL: imageServiceURL,
	})
}

func TestV2UploadHdf5RoutesProxyImageService(t *testing.T) {
	t.Parallel()

	wantPNG := []byte("\x89PNG\r\n\x1a\nFAKE")
	wantJSON := []byte(`{"ok":true}`)
	wantRaw := []byte("\x01\x02\x03\x04")

	gotQueries := make(map[string]url.Values)
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotQueries[r.URL.Path] = r.URL.Query()
		switch r.URL.Path {
		case "/hdf5/dataset", "/hdf5/preview/histogram", "/hdf5/preview/table":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write(wantJSON)
		case "/hdf5/preview/slice", "/hdf5/preview/atlas":
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write(wantPNG)
		case "/hdf5/preview/scalar-volume":
			w.Header().Set("Content-Type", "application/octet-stream")
			w.Header().Set("X-Volume-Width", "4")
			w.Header().Set("X-Volume-Height", "3")
			w.Header().Set("X-Volume-Depth", "2")
			w.Header().Set("X-Volume-Dtype", "float32")
			w.Header().Set("X-Volume-Channel", "0")
			_, _ = w.Write(wantRaw)
		default:
			http.NotFound(w, r)
		}
	}))
	defer imageSvc.Close()

	router := newHdf5ProxyRouter(t, imageSvc.URL)
	fileID := uploadHdf5ForProxyTest(t, router)

	do := func(pathAndQuery string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+pathAndQuery, nil)
		setProxyOwnerHeaders(req)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		return rec
	}

	// JSON endpoints: dataset_path (and per-endpoint params) forwarded, body proxied.
	datasetPath := "/DataContainers/SyntheticVolume/CellData/EulerAngles"
	rec := do("/hdf5/dataset?dataset_path=" + url.QueryEscape(datasetPath))
	if rec.Code != http.StatusOK || !bytes.Equal(rec.Body.Bytes(), wantJSON) {
		t.Fatalf("dataset status=%d body=%s, want 200 + proxied JSON", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Fatalf("dataset content-type = %q, want application/json", ct)
	}
	q := gotQueries["/hdf5/dataset"]
	if q.Get("dataset_path") != datasetPath {
		t.Fatalf("dataset forwarded dataset_path = %q, want %q", q.Get("dataset_path"), datasetPath)
	}
	if !strings.Contains(q.Get("path"), fileID) {
		t.Fatalf("dataset path = %q, want resolved storage path containing %q", q.Get("path"), fileID)
	}
	// The sidecar echoes file_id into the JSON payloads; the frontend builds every
	// follow-up preview URL from summary.file_id, so the real id must be forwarded.
	if q.Get("file_id") != fileID {
		t.Fatalf("dataset forwarded file_id = %q, want %q", q.Get("file_id"), fileID)
	}

	rec = do("/hdf5/preview/histogram?dataset_path=" + url.QueryEscape(datasetPath) + "&bins=24&component=1")
	if rec.Code != http.StatusOK || !bytes.Equal(rec.Body.Bytes(), wantJSON) {
		t.Fatalf("histogram status=%d body=%s, want 200 + proxied JSON", rec.Code, rec.Body.String())
	}
	q = gotQueries["/hdf5/preview/histogram"]
	if q.Get("bins") != "24" || q.Get("component") != "1" || q.Get("dataset_path") != datasetPath {
		t.Fatalf("histogram forwarded query = %v, want bins=24 component=1 dataset_path", q)
	}

	rec = do("/hdf5/preview/table?dataset_path=" + url.QueryEscape(datasetPath) + "&offset=12&limit=12")
	if rec.Code != http.StatusOK || !bytes.Equal(rec.Body.Bytes(), wantJSON) {
		t.Fatalf("table status=%d body=%s, want 200 + proxied JSON", rec.Code, rec.Body.String())
	}
	q = gotQueries["/hdf5/preview/table"]
	if q.Get("offset") != "12" || q.Get("limit") != "12" {
		t.Fatalf("table forwarded query = %v, want offset=12 limit=12", q)
	}

	// PNG endpoints: bytes + content-type pass through untouched.
	rec = do("/hdf5/preview/slice?dataset_path=" + url.QueryEscape(datasetPath) + "&axis=y&index=7&component=2")
	if rec.Code != http.StatusOK || !bytes.Equal(rec.Body.Bytes(), wantPNG) {
		t.Fatalf("slice status=%d, want 200 + proxied PNG", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "image/png" {
		t.Fatalf("slice content-type = %q, want image/png", ct)
	}
	q = gotQueries["/hdf5/preview/slice"]
	if q.Get("axis") != "y" || q.Get("index") != "7" || q.Get("component") != "2" {
		t.Fatalf("slice forwarded query = %v, want axis=y index=7 component=2", q)
	}

	rec = do("/hdf5/preview/atlas?dataset_path=" + url.QueryEscape(datasetPath) + "&channels=0,1&negative=true")
	if rec.Code != http.StatusOK || !bytes.Equal(rec.Body.Bytes(), wantPNG) {
		t.Fatalf("atlas status=%d, want 200 + proxied PNG", rec.Code)
	}
	q = gotQueries["/hdf5/preview/atlas"]
	if q.Get("channels") != "0,1" || q.Get("negative") != "true" {
		t.Fatalf("atlas forwarded query = %v, want channels=0,1 negative=true", q)
	}

	// Scalar volume: raw bytes stream through with the x-volume-* metadata headers.
	rec = do("/hdf5/preview/scalar-volume?dataset_path=" + url.QueryEscape(datasetPath) + "&channel=0")
	if rec.Code != http.StatusOK || !bytes.Equal(rec.Body.Bytes(), wantRaw) {
		t.Fatalf("scalar-volume status=%d, want 200 + proxied raw bytes", rec.Code)
	}
	for header, want := range map[string]string{
		"X-Volume-Width":   "4",
		"X-Volume-Height":  "3",
		"X-Volume-Depth":   "2",
		"X-Volume-Dtype":   "float32",
		"X-Volume-Channel": "0",
	} {
		if got := rec.Header().Get(header); got != want {
			t.Fatalf("scalar-volume header %s = %q, want %q", header, got, want)
		}
	}
	if got := gotQueries["/hdf5/preview/scalar-volume"].Get("channel"); got != "0" {
		t.Fatalf("scalar-volume forwarded channel = %q, want 0", got)
	}
}

func TestV2UploadHdf5QueryPassthroughIsAllowlisted(t *testing.T) {
	t.Parallel()

	var gotQuery url.Values
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotQuery = r.URL.Query()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{}`))
	}))
	defer imageSvc.Close()

	router := newHdf5ProxyRouter(t, imageSvc.URL)
	fileID := uploadHdf5ForProxyTest(t, router)

	req := httptest.NewRequest(http.MethodGet,
		"/v2/uploads/"+fileID+"/hdf5/dataset?dataset_path=/data&evil=../../etc/passwd&path=/etc/shadow&file_id=stolen&bins=9", nil)
	setProxyOwnerHeaders(req)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("dataset status = %d body=%s", rec.Code, rec.Body.String())
	}
	if gotQuery.Get("evil") != "" || gotQuery.Get("bins") != "" {
		t.Fatalf("non-allowlisted params leaked to the sidecar: %v", gotQuery)
	}
	// A client-supplied ?path / ?file_id can never override the resolved values.
	if got := gotQuery.Get("path"); got == "/etc/shadow" || !strings.Contains(got, fileID) {
		t.Fatalf("sidecar path = %q, want the server-resolved storage path", got)
	}
	if got := gotQuery.Get("file_id"); got != fileID {
		t.Fatalf("sidecar file_id = %q, want the server-resolved %q", got, fileID)
	}
}

func TestV2UploadHdf5OwnershipEnforcedBeforeProxy(t *testing.T) {
	t.Parallel()

	sidecarHits := 0
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sidecarHits++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{}`))
	}))
	defer imageSvc.Close()

	router := newHdf5ProxyRouter(t, imageSvc.URL)
	fileID := uploadHdf5ForProxyTest(t, router)

	for suffix := range hdf5ProxyRoutes {
		req := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+suffix+"?dataset_path=/data", nil)
		req.Header.Set("X-Ultra-User-Id", "intruder")
		req.Header.Set("X-Ultra-Org-Id", "smithsonian")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != http.StatusNotFound {
			t.Fatalf("intruder %s status = %d, want 404", suffix, rec.Code)
		}
	}
	if sidecarHits != 0 {
		t.Fatalf("sidecar was reached %d times for unauthorized requests, want 0", sidecarHits)
	}
}

func TestV2UploadHdf5NotConfiguredWithoutImageService(t *testing.T) {
	t.Parallel()

	router := newHdf5ProxyRouter(t, "")
	fileID := uploadHdf5ForProxyTest(t, router)

	for suffix := range hdf5ProxyRoutes {
		req := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+suffix+"?dataset_path=/data", nil)
		setProxyOwnerHeaders(req)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != http.StatusNotImplemented {
			t.Fatalf("%s status = %d, want 501 not configured", suffix, rec.Code)
		}
	}
}

func TestV2UploadHdf5UpstreamErrorsPassThroughCleanly(t *testing.T) {
	t.Parallel()

	upstreamBody := `{"detail":"dataset /nope not found in /internal/uploads/secret.h5"}`
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		status := http.StatusNotFound
		if r.URL.Path == "/hdf5/preview/slice" {
			status = http.StatusUnprocessableEntity
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(upstreamBody))
	}))
	defer imageSvc.Close()

	router := newHdf5ProxyRouter(t, imageSvc.URL)
	fileID := uploadHdf5ForProxyTest(t, router)

	for suffix, wantStatus := range map[string]int{
		"/hdf5/dataset":       http.StatusNotFound,            // unknown dataset_path
		"/hdf5/preview/slice": http.StatusUnprocessableEntity, // not sliceable
	} {
		req := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+suffix+"?dataset_path=/nope", nil)
		setProxyOwnerHeaders(req)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != wantStatus {
			t.Fatalf("%s status = %d, want %d (upstream status preserved)", suffix, rec.Code, wantStatus)
		}
		if strings.Contains(rec.Body.String(), "secret.h5") {
			t.Fatalf("%s leaked the upstream error body: %s", suffix, rec.Body.String())
		}
		var errResp struct {
			Error string `json:"error"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &errResp); err != nil || errResp.Error == "" {
			t.Fatalf("%s error body = %s, want clean JSON {\"error\": ...}", suffix, rec.Body.String())
		}
	}
}
