package httpapi

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

// TestBatchAnalysisSubmitAndRegisterOutputs walks the full control-plane contract a
// scientist's batch relies on: submit a MegaSeg batch (job + auto results collection),
// then have the worker register a produced mask — verifying it becomes a downloadable
// resource grouped into the results collection with analysis provenance.
func TestBatchAnalysisSubmitAndRegisterOutputs(t *testing.T) {
	t.Parallel()
	mem := store.NewMemoryStore()
	uploadRoot := t.TempDir()
	router := NewRouter(ServerDeps{Version: "test", Store: mem, UploadRoot: uploadRoot})
	const user, org = "u1", "o1"

	if _, err := mem.UpsertResource(context.Background(), domain.UpsertResourceInput{
		ResourceID:   "img-1",
		OriginalName: "cells.ome.tif",
		ContentType:  "image/tiff",
		SizeBytes:    1024,
		SourceType:   "upload",
		ResourceKind: "image",
		OwnerUserID:  user,
		OwnerOrgID:   org,
		Status:       "active",
	}); err != nil {
		t.Fatalf("seed input resource: %v", err)
	}

	submit := analysisAuthedJSON(t, router, http.MethodPost, "/v2/analysis/batch", user, org, map[string]any{
		"model":        "megaseg",
		"resource_ids": []string{"img-1"},
		"params":       map[string]any{"structure_channel": 4},
	})
	if submit.Code != http.StatusAccepted {
		t.Fatalf("submit status = %d, body=%s", submit.Code, submit.Body.String())
	}
	var submitResp batchAnalysisJobResponse
	if err := json.Unmarshal(submit.Body.Bytes(), &submitResp); err != nil {
		t.Fatalf("decode submit: %v", err)
	}
	jobID := submitResp.Job.JobID
	if jobID == "" {
		t.Fatal("submit returned empty job id")
	}
	if submitResp.Job.JobType != "analysis.megaseg" {
		t.Fatalf("job_type = %q, want analysis.megaseg", submitResp.Job.JobType)
	}
	collectionID := submitResp.ResultsCollection.CollectionID
	if collectionID == "" {
		t.Fatal("submit returned empty results collection")
	}

	// The worker writes the output mask into the shared upload root under analysis/<job>/.
	rel := filepath.Join("analysis", jobID, "img-1__megaseg_mask.tif")
	abs := filepath.Join(uploadRoot, rel)
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(abs, []byte("MASKDATA"), 0o644); err != nil {
		t.Fatalf("write output: %v", err)
	}

	reg := analysisAuthedJSON(t, router, http.MethodPost, "/v2/data-agent/jobs/"+jobID+"/outputs", user, org, map[string]any{
		"outputs": []map[string]any{{
			"storage_path":       filepath.ToSlash(rel),
			"original_name":      "cells_megaseg_mask.tif",
			"content_type":       "image/tiff",
			"source_resource_id": "img-1",
			"artifact_kind":      "mask",
			"metadata":           map[string]any{"model_version": "epoch_650"},
		}},
	})
	if reg.Code != http.StatusCreated {
		t.Fatalf("register status = %d, body=%s", reg.Code, reg.Body.String())
	}
	var regResp registerAnalysisOutputsResponse
	if err := json.Unmarshal(reg.Body.Bytes(), &regResp); err != nil {
		t.Fatalf("decode register: %v", err)
	}
	if regResp.Count != 1 || len(regResp.Registered) != 1 {
		t.Fatalf("registered count = %d, registered = %d", regResp.Count, len(regResp.Registered))
	}
	out := regResp.Registered[0]
	if out.FileID == "" {
		t.Fatal("registered output has empty id")
	}
	if out.SourceType != "analysis" {
		t.Fatalf("output source_type = %q, want analysis", out.SourceType)
	}

	stored, err := mem.GetResourceForUser(context.Background(), out.FileID, user, org)
	if err != nil {
		t.Fatalf("get output resource: %v", err)
	}
	if stored.SizeBytes != int64(len("MASKDATA")) {
		t.Fatalf("output size = %d, want %d", stored.SizeBytes, len("MASKDATA"))
	}

	page, err := mem.ListResourcesForCollectionForUser(context.Background(), domain.ResourceCollectionResourceListInput{
		CollectionID: collectionID, UserID: user, OrgID: org, Limit: 50,
	})
	if err != nil {
		t.Fatalf("list collection: %v", err)
	}
	found := false
	for _, r := range page.Resources {
		if r.ResourceID == out.FileID {
			found = true
		}
	}
	if !found {
		t.Fatalf("output %s not grouped into results collection %s (got %d resources)", out.FileID, collectionID, len(page.Resources))
	}

	dl := analysisAuthedGet(t, router, "/v2/resources/"+out.FileID+"/download", user, org)
	if dl.Code != http.StatusOK {
		t.Fatalf("download status = %d, body=%s", dl.Code, dl.Body.String())
	}
	if dl.Body.String() != "MASKDATA" {
		t.Fatalf("download body = %q, want MASKDATA", dl.Body.String())
	}
}

// TestRegisterAnalysisOutputsRejectsUnsafePath asserts the worker-facing endpoint refuses
// to register anything outside the analysis/ output prefix (path traversal / arbitrary
// upload-root files), so a worker token can only catalog files the analysis worker wrote.
func TestRegisterAnalysisOutputsRejectsUnsafePath(t *testing.T) {
	t.Parallel()
	mem := store.NewMemoryStore()
	uploadRoot := t.TempDir()
	router := NewRouter(ServerDeps{Version: "test", Store: mem, UploadRoot: uploadRoot})
	const user, org = "u1", "o1"
	if _, err := mem.UpsertResource(context.Background(), domain.UpsertResourceInput{
		ResourceID: "img-1", OriginalName: "a.tif", SourceType: "upload", ResourceKind: "image",
		OwnerUserID: user, OwnerOrgID: org, Status: "active", SizeBytes: 1,
	}); err != nil {
		t.Fatalf("seed: %v", err)
	}
	submit := analysisAuthedJSON(t, router, http.MethodPost, "/v2/analysis/batch", user, org, map[string]any{
		"model": "rarespot", "resource_ids": []string{"img-1"},
	})
	if submit.Code != http.StatusAccepted {
		t.Fatalf("submit = %d, %s", submit.Code, submit.Body.String())
	}
	var sr batchAnalysisJobResponse
	if err := json.Unmarshal(submit.Body.Bytes(), &sr); err != nil {
		t.Fatalf("decode submit: %v", err)
	}

	for _, bad := range []string{"../escape.tif", "/etc/passwd", "uploads/x.tif", "analysis/../../x.tif", ""} {
		reg := analysisAuthedJSON(t, router, http.MethodPost, "/v2/data-agent/jobs/"+sr.Job.JobID+"/outputs", user, org, map[string]any{
			"outputs": []map[string]any{{"storage_path": bad, "original_name": "x.tif"}},
		})
		if reg.Code != http.StatusBadRequest {
			t.Fatalf("storage_path %q: status = %d, want 400 (body=%s)", bad, reg.Code, reg.Body.String())
		}
	}
}

// TestBatchAnalysisRejectsBadModel guards the model allow-list.
func TestBatchAnalysisRejectsBadModel(t *testing.T) {
	t.Parallel()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{Version: "test", Store: mem, UploadRoot: t.TempDir()})
	resp := analysisAuthedJSON(t, router, http.MethodPost, "/v2/analysis/batch", "u1", "o1", map[string]any{
		"model": "sam3", "resource_ids": []string{"img-1"},
	})
	if resp.Code != http.StatusBadRequest {
		t.Fatalf("bad model status = %d, want 400 (body=%s)", resp.Code, resp.Body.String())
	}
}

// TestDataAgentStatusUpdateDoesNotClobberMetadata locks the fix for the bug where a
// progress status-update carrying an empty metadata map wiped the job's create-time
// metadata (results_collection_id), which silently broke results-collection grouping.
func TestDataAgentStatusUpdateDoesNotClobberMetadata(t *testing.T) {
	t.Parallel()
	mem := store.NewMemoryStore()
	ctx := context.Background()
	job, err := mem.CreateDataAgentJob(ctx, domain.CreateDataAgentJobInput{
		OwnerUserID: "u1", OwnerOrgID: "o1", JobType: "analysis.megaseg", Status: "queued",
		ResourceIDs: nil, ResourceCount: 2,
		Metadata:  domain.JSONMap{"results_collection_id": "col1", "model": "megaseg"},
		CreatedAt: domain.Now(),
	})
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	// The worker's progress update sends an empty metadata map — it must NOT clobber.
	if _, _, err := mem.UpdateDataAgentJob(ctx, domain.UpdateDataAgentJobInput{
		JobID: job.JobID, OwnerUserID: "u1", OwnerOrgID: "o1", Status: "running",
		ProgressCompleted: 1, ProgressTotal: 2,
		Metadata:      domain.JSONMap{},
		OutputSummary: domain.JSONMap{"items": domain.JSONMap{}},
		UpdatedAt:     domain.Now(),
	}); err != nil {
		t.Fatalf("update job: %v", err)
	}
	got, err := mem.GetDataAgentJobForUser(ctx, job.JobID, "u1", "o1")
	if err != nil {
		t.Fatalf("get job: %v", err)
	}
	if got.Metadata["results_collection_id"] != "col1" {
		t.Fatalf("results_collection_id clobbered by status update: metadata=%v", got.Metadata)
	}
}

func analysisAuthedJSON(t *testing.T, router http.Handler, method, path, user, org string, body any) *httptest.ResponseRecorder {
	t.Helper()
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal body: %v", err)
	}
	req := httptest.NewRequest(method, path, bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Ultra-User-Id", user)
	req.Header.Set("X-Ultra-Org-Id", org)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	return rec
}

func analysisAuthedGet(t *testing.T, router http.Handler, path, user, org string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, path, nil)
	req.Header.Set("X-Ultra-User-Id", user)
	req.Header.Set("X-Ultra-Org-Id", org)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	return rec
}
