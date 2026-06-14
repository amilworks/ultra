package httpapi

import (
	"bytes"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

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
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/tile":
			q := r.URL.Query()
			gotTilePath, gotLevel, gotCol, gotRow = q.Get("path"), q.Get("level"), q.Get("col"), q.Get("row")
		case "/atlas":
			gotAtlasPath = r.URL.Query().Get("path")
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
	tileReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/tiles/z/2/3/4?size=256", nil)
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
	if !strings.Contains(gotTilePath, fileID) {
		t.Fatalf("tile path = %q, want resolved storage path containing %q", gotTilePath, fileID)
	}

	// Authorized atlas request is proxied.
	atlasReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/atlas?level=2&grid_rows=4&grid_cols=5", nil)
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
