package httpapi

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

func videoExportRequestBody(mode, profile string) []byte {
	payload := videoExportCreateRequest{
		Mode:             mode,
		Profile:          profile,
		FixedZ:           2,
		FixedT:           1,
		Channels:         []int{0, 2},
		ChannelColors:    []string{"#ff0000", "#00ffff"},
		ScalarRenderMode: "intensity",
	}
	data, _ := json.Marshal(payload)
	return data
}

func TestVideoExportFrameSchedulesAreEndpointPreserving(t *testing.T) {
	t.Parallel()

	preview := uniformlySampledFrameIndices(756, videoExportPreviewFrameLimit)
	if len(preview) != videoExportPreviewFrameLimit || preview[0] != 0 || preview[len(preview)-1] != 755 {
		t.Fatalf("preview schedule = len %d endpoints %d/%d", len(preview), preview[0], preview[len(preview)-1])
	}
	for index := 1; index < len(preview); index++ {
		if preview[index] <= preview[index-1] {
			t.Fatalf("preview schedule is not strictly increasing at %d: %v", index, preview[index-2:index+1])
		}
	}
	complete := completeFrameIndices(405)
	if len(complete) != 405 || complete[0] != 0 || complete[404] != 404 {
		t.Fatalf("complete schedule = %v ... %v", complete[:2], complete[len(complete)-2:])
	}
}

func TestCanonicalVideoRecipeUsesCrossLanguageUTF8Identity(t *testing.T) {
	t.Parallel()

	recipe := videoExportRecipe{Schema: "<scientific>&\u03bc", ResourceID: "file_stack"}
	canonical, err := canonicalVideoExportJSON(recipe)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(canonical, []byte(`\u003c`)) || !bytes.Contains(canonical, []byte("\u03bc")) {
		t.Fatalf("canonical recipe JSON = %s", canonical)
	}
}

func TestCreateUploadVideoExportPublishesSourceBoundRecipe(t *testing.T) {
	t.Parallel()

	imageService := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/viewerinfo" {
			http.NotFound(w, r)
			return
		}
		writeJSON(w, http.StatusOK, derivativeViewerInfoForTest(3, 4, 12, 32, 48, 0))
	}))
	defer imageService.Close()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, bus),
		Store:           mem,
		UploadRoot:      uploadRoot,
		ImageServiceURL: imageService.URL,
		DataAgentJobs:   bus,
	})
	fileID := uploadNamedFileForProxyTest(t, router, "stack.ome.tiff", []byte("source-generation"))
	req := httptest.NewRequest(
		http.MethodPost,
		"/v2/uploads/"+fileID+"/video-exports",
		bytes.NewReader(videoExportRequestBody("z_sweep", "preview")),
	)
	req.Header.Set("Content-Type", "application/json")
	setProxyOwnerHeaders(req)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusAccepted {
		t.Fatalf("create video status=%d body=%s", rec.Code, rec.Body.String())
	}
	var response videoExportResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if !videoExportRenderIDPattern.MatchString(response.RenderID) || response.FramesTotal != 12 || response.SourceFrames != 12 || response.Sampled {
		t.Fatalf("response = %+v", response)
	}
	var videoJob *eventbus.DataAgentJob
	for {
		select {
		case job := <-bus.DataAgentJobs():
			if job.JobType == "image.render_video" {
				videoJob = &job
			}
		default:
			goto jobsDrained
		}
	}

jobsDrained:
	if videoJob == nil {
		t.Fatal("no image.render_video job published")
	}
	job := *videoJob
	if job.JobType != "image.render_video" || job.OwnerUserID == "" || len(job.ResourceIDs) != 1 || job.ResourceIDs[0] != fileID {
		t.Fatalf("job envelope = %+v", job)
	}
	recipe, ok := job.Metadata["recipe"].(videoExportRecipe)
	if !ok {
		t.Fatalf("recipe type = %T", job.Metadata["recipe"])
	}
	if recipe.SourceFrameCount != 12 || recipe.FixedT != 1 || recipe.Channels[1] != 2 || recipe.Source.SHA256 == "" {
		t.Fatalf("recipe = %+v", recipe)
	}
	if filepath.Base(job.Metadata["output_path"].(string)) != videoExportArtifactName(fileID, response.RenderID) {
		t.Fatalf("output path = %v", job.Metadata["output_path"])
	}
}

func TestCreateUploadVideoExportRejectsOversizedCompleteSeries(t *testing.T) {
	t.Parallel()

	record := resourceRecord{FileID: "file_series", OriginalName: "series.ome.tiff", SHA256: string(bytes.Repeat([]byte{'a'}, 64)), SizeBytes: 10}
	request := videoExportCreateRequest{
		Mode: "time_series", Profile: "complete", FixedZ: 0, FixedT: 0,
		Channels: []int{0}, ScalarRenderMode: "intensity",
	}
	_, _, err := canonicalVideoExportRecipe(record, videoExportAxes{T: videoExportCompleteFrameLimit + 1, C: 1, Z: 1}, request)
	if err == nil {
		t.Fatal("oversized complete time series was accepted")
	}
}

func TestWorkerVideoFrameRequiresWorkerAuthenticationAndReusesSlicePath(t *testing.T) {
	t.Parallel()

	var framePNG bytes.Buffer
	imageData := image.NewRGBA(image.Rect(0, 0, 3, 2))
	imageData.Set(1, 1, color.RGBA{R: 255, A: 255})
	if err := png.Encode(&framePNG, imageData); err != nil {
		t.Fatal(err)
	}
	imageService := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/slice" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write(framePNG.Bytes())
	}))
	defer imageService.Close()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:           mem,
		UploadRoot:      uploadRoot,
		ImageServiceURL: imageService.URL,
		WorkerToken:     "worker-secret",
	})
	fileID := uploadNamedFileForProxyTest(t, router, "stack.ome.tiff", []byte("source-generation"))
	path := "/v2/internal/uploads/" + fileID + "/render-frame?axis=z&z=0&t=0&channels=0&full_resolution=false"

	unauthorized := httptest.NewRequest(http.MethodGet, path, nil)
	setProxyOwnerHeaders(unauthorized)
	unauthorizedRec := httptest.NewRecorder()
	router.ServeHTTP(unauthorizedRec, unauthorized)
	if unauthorizedRec.Code != http.StatusUnauthorized {
		t.Fatalf("unauthorized frame status=%d body=%s", unauthorizedRec.Code, unauthorizedRec.Body.String())
	}

	authorized := httptest.NewRequest(http.MethodGet, path, nil)
	setProxyOwnerHeaders(authorized)
	authorized.Header.Set("X-Ultra-Worker-Token", "worker-secret")
	authorizedRec := httptest.NewRecorder()
	router.ServeHTTP(authorizedRec, authorized)
	if authorizedRec.Code != http.StatusOK || !bytes.Equal(authorizedRec.Body.Bytes(), framePNG.Bytes()) {
		t.Fatalf("authorized frame status=%d content-type=%q body=%x", authorizedRec.Code, authorizedRec.Header().Get("Content-Type"), authorizedRec.Body.Bytes())
	}
}

func TestDownloadVideoExportRejectsChangedSourceGeneration(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	sourceBytes := []byte("immutable-source")
	fileID := uploadNamedFileForProxyTest(t, router, "stack.ome.tiff", sourceBytes)
	sourceDigest := sha256.Sum256(sourceBytes)
	record := resourceRecord{
		FileID: fileID, OriginalName: "stack.ome.tiff", SizeBytes: int64(len(sourceBytes)),
		SHA256: fmt.Sprintf("%x", sourceDigest),
	}
	recipe, renderID, err := canonicalVideoExportRecipe(record, videoExportAxes{T: 1, C: 1, Z: 4}, videoExportCreateRequest{
		Mode: "z_sweep", Profile: "complete", Channels: []int{0}, ScalarRenderMode: "intensity",
	})
	if err != nil {
		t.Fatal(err)
	}
	artifactBytes := []byte("strict-mp4-artifact")
	artifactDigest := sha256.Sum256(artifactBytes)
	derived := filepath.Join(uploadRoot, resourceDerivedDir)
	if err := os.MkdirAll(derived, 0o700); err != nil {
		t.Fatal(err)
	}
	artifactName := videoExportArtifactName(fileID, renderID)
	if err := os.WriteFile(filepath.Join(derived, artifactName), artifactBytes, 0o600); err != nil {
		t.Fatal(err)
	}
	manifest := videoExportManifest{
		Schema: videoExportSchema, RenderID: renderID, CreatedAt: time.Now().UTC().Format(time.RFC3339Nano),
		Source: recipe.Source, Recipe: recipe,
		Artifact: videoExportArtifact{
			Basename: artifactName, SHA256: fmt.Sprintf("%x", artifactDigest), SizeBytes: int64(len(artifactBytes)),
			MediaType: "video/mp4", Width: 8, Height: 6, FrameCount: len(recipe.FrameIndices), FPS: videoExportFPS,
		},
	}
	if err := atomicWriteVideoExportJSON(filepath.Join(derived, videoExportManifestName(fileID, renderID)), manifest); err != nil {
		t.Fatal(err)
	}

	downloadPath := "/v2/uploads/" + fileID + "/video-exports/" + renderID + "/download"
	request := httptest.NewRequest(http.MethodGet, downloadPath, nil)
	setProxyOwnerHeaders(request)
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)
	if response.Code != http.StatusOK || !bytes.Equal(response.Body.Bytes(), artifactBytes) {
		t.Fatalf("initial download status=%d body=%q", response.Code, response.Body.Bytes())
	}

	sourcePath := filepath.Join(uploadRoot, fileID+"__stack.ome.tiff")
	if err := os.WriteFile(sourcePath, bytes.Repeat([]byte{'x'}, len(sourceBytes)), 0o600); err != nil {
		t.Fatal(err)
	}
	changedRequest := httptest.NewRequest(http.MethodGet, downloadPath, nil)
	setProxyOwnerHeaders(changedRequest)
	changedResponse := httptest.NewRecorder()
	router.ServeHTTP(changedResponse, changedRequest)
	if changedResponse.Code != http.StatusConflict {
		t.Fatalf("changed-source download status=%d body=%s", changedResponse.Code, changedResponse.Body.String())
	}
}

func TestVideoDerivativeNamesAreOwnedByResource(t *testing.T) {
	t.Parallel()

	digest := string(bytes.Repeat([]byte{'b'}, 64))
	matcher, err := derivativeNameMatcher("file_stack")
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{
		"file_stack__video." + digest + ".mp4",
		"file_stack__video." + digest + ".manifest.json",
		"file_stack__video." + digest + ".progress.json",
		".file_stack__video." + digest + ".tmp-abcdef12.mp4",
		".file_stack__video." + digest + ".queued.json.tmp-abcdef12",
	} {
		if !matcher.MatchString(name) {
			t.Fatalf("video derivative %q is not cleanup-owned", name)
		}
	}
	if matcher.MatchString("file_stack_neighbor__video." + digest + ".mp4") {
		t.Fatal("matcher claimed a neighboring resource video")
	}
}
