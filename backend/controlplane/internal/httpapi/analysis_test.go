package httpapi

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

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

	for _, bad := range []string{"../escape.tif", "/etc/passwd", "uploads/x.tif", "analysis/../../x.tif", "analysis/another-job/x.tif", ""} {
		reg := analysisAuthedJSON(t, router, http.MethodPost, "/v2/data-agent/jobs/"+sr.Job.JobID+"/outputs", user, org, map[string]any{
			"outputs": []map[string]any{{"storage_path": bad, "original_name": "x.tif"}},
		})
		if reg.Code != http.StatusBadRequest {
			t.Fatalf("storage_path %q: status = %d, want 400 (body=%s)", bad, reg.Code, reg.Body.String())
		}
	}

	outside := filepath.Join(t.TempDir(), "outside-mask.tif")
	if err := os.WriteFile(outside, []byte("outside"), 0o644); err != nil {
		t.Fatal(err)
	}
	linkedRel := filepath.Join("analysis", sr.Job.JobID, "linked-mask.tif")
	linkedAbs := filepath.Join(uploadRoot, linkedRel)
	if err := os.MkdirAll(filepath.Dir(linkedAbs), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, linkedAbs); err != nil {
		t.Fatal(err)
	}
	reg := analysisAuthedJSON(t, router, http.MethodPost, "/v2/data-agent/jobs/"+sr.Job.JobID+"/outputs", user, org, map[string]any{
		"outputs": []map[string]any{{"storage_path": filepath.ToSlash(linkedRel), "original_name": "linked-mask.tif"}},
	})
	if reg.Code != http.StatusBadRequest {
		t.Fatalf("symlinked output status = %d, want 400 (body=%s)", reg.Code, reg.Body.String())
	}
	if payload, err := os.ReadFile(outside); err != nil || string(payload) != "outside" {
		t.Fatalf("outside analysis target changed: %q err=%v", payload, err)
	}
}

func TestRegisterAnalysisOutputsIgnoresWorkerResourceIDAndPreservesForeignResource(t *testing.T) {
	t.Parallel()
	mem := store.NewMemoryStore()
	uploadRoot := t.TempDir()
	router := NewRouter(ServerDeps{Version: "test", Store: mem, UploadRoot: uploadRoot})
	ctx := context.Background()
	const user, org = "alice", "org-a"
	const foreignID = "file_bob_active"
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID: foreignID, OriginalName: "bob-source.tif", ContentType: "image/tiff",
		StorageURI: "s3://bob/private.tif", SizeBytes: 99, SourceType: "upload", ResourceKind: "image",
		OwnerUserID: "bob", OwnerOrgID: "org-b", Status: domain.ResourceStatusActive,
	}); err != nil {
		t.Fatal(err)
	}
	grant, err := mem.CreateResourceShareGrant(ctx, domain.CreateResourceShareGrantInput{
		ResourceID: foreignID, OwnerUserID: "bob", OwnerOrgID: "org-b",
		GranteeUserID: user, GranteeOrgID: org, Role: "read", Status: "active", CreatedByUserID: "bob",
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID: "img-analysis", OriginalName: "input.tif", SourceType: "upload", ResourceKind: "image",
		OwnerUserID: user, OwnerOrgID: org, Status: domain.ResourceStatusActive,
	}); err != nil {
		t.Fatal(err)
	}
	submit := analysisAuthedJSON(t, router, http.MethodPost, "/v2/analysis/batch", user, org, map[string]any{
		"model": "megaseg", "resource_ids": []string{"img-analysis"},
	})
	if submit.Code != http.StatusAccepted {
		t.Fatalf("submit = %d, %s", submit.Code, submit.Body.String())
	}
	var sr batchAnalysisJobResponse
	if err := json.Unmarshal(submit.Body.Bytes(), &sr); err != nil {
		t.Fatal(err)
	}
	rel := filepath.Join("analysis", sr.Job.JobID, "mask.tif")
	abs := filepath.Join(uploadRoot, rel)
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(abs, []byte("mask"), 0o644); err != nil {
		t.Fatal(err)
	}
	reg := analysisAuthedJSON(t, router, http.MethodPost, "/v2/data-agent/jobs/"+sr.Job.JobID+"/outputs", user, org, map[string]any{
		"outputs": []map[string]any{{
			"resource_id": foreignID, "storage_path": filepath.ToSlash(rel), "original_name": "mask.tif",
		}},
	})
	if reg.Code != http.StatusCreated {
		t.Fatalf("register = %d, %s", reg.Code, reg.Body.String())
	}
	var response registerAnalysisOutputsResponse
	if err := json.Unmarshal(reg.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if len(response.Registered) != 1 || response.Registered[0].FileID == foreignID {
		t.Fatalf("registered resources = %+v, want one server-derived ID", response.Registered)
	}
	foreign, err := mem.GetResourceForUser(ctx, foreignID, "bob", "org-b")
	if err != nil {
		t.Fatal(err)
	}
	if foreign.OriginalName != "bob-source.tif" || foreign.StorageURI != "s3://bob/private.tif" || foreign.SizeBytes != 99 || foreign.OwnerUserID != "bob" {
		t.Fatalf("foreign resource was changed: %+v", foreign)
	}
	grants, err := mem.ListResourceShareGrantsForResource(ctx, domain.ListResourceShareGrantsInput{
		ResourceID: foreignID, OwnerUserID: "bob", OwnerOrgID: "org-b", Status: "active", Limit: 10,
	})
	if err != nil || len(grants) != 1 || grants[0].GrantID != grant.GrantID {
		t.Fatalf("foreign grants = %+v err=%v, want original grant", grants, err)
	}
}

func TestRegisterAnalysisOutputsCleansLateReplayAfterPermanentPurge(t *testing.T) {
	t.Parallel()
	mem := store.NewMemoryStore()
	uploadRoot := t.TempDir()
	router := NewRouter(ServerDeps{Version: "test", Store: mem, UploadRoot: uploadRoot})
	ctx := context.Background()
	const user, org = "analysis-owner", "analysis-org"
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID: "analysis-input", OriginalName: "input.tif", SourceType: "upload", ResourceKind: "image",
		OwnerUserID: user, OwnerOrgID: org, Status: domain.ResourceStatusActive,
	}); err != nil {
		t.Fatal(err)
	}
	submit := analysisAuthedJSON(t, router, http.MethodPost, "/v2/analysis/batch", user, org, map[string]any{
		"model": "megaseg", "resource_ids": []string{"analysis-input"},
	})
	if submit.Code != http.StatusAccepted {
		t.Fatalf("submit = %d, %s", submit.Code, submit.Body.String())
	}
	var submitted batchAnalysisJobResponse
	if err := json.Unmarshal(submit.Body.Bytes(), &submitted); err != nil {
		t.Fatal(err)
	}
	relative := filepath.Join("analysis", submitted.Job.JobID, "late-mask.tif")
	absolute := filepath.Join(uploadRoot, relative)
	if err := os.MkdirAll(filepath.Dir(absolute), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(absolute, []byte("mask-generation"), 0o644); err != nil {
		t.Fatal(err)
	}
	register := func() *httptest.ResponseRecorder {
		return analysisAuthedJSON(t, router, http.MethodPost, "/v2/data-agent/jobs/"+submitted.Job.JobID+"/outputs", user, org, map[string]any{
			"outputs": []map[string]any{{
				"storage_path": filepath.ToSlash(relative), "original_name": "late-mask.tif",
				"content_type": "image/tiff", "artifact_kind": "mask",
			}},
		})
	}
	first := register()
	if first.Code != http.StatusCreated {
		t.Fatalf("initial register = %d, %s", first.Code, first.Body.String())
	}
	resourceID := analysisOutputResourceID(submitted.Job.JobID, relative)
	if _, err := mem.SoftDeleteResourceForUser(ctx, resourceID, user, org, time.Now().Add(-31*24*time.Hour)); err != nil {
		t.Fatal(err)
	}
	if reclaimed, _, err := ReclaimExpiredResources(ctx, mem, uploadRoot, 10); err != nil || reclaimed != 1 {
		t.Fatalf("reclaim analysis output = %d err=%v, want one", reclaimed, err)
	}
	if err := os.WriteFile(absolute, []byte("stale-replay"), 0o644); err != nil {
		t.Fatal(err)
	}
	derived := filepath.Join(uploadRoot, resourceDerivedDir)
	if err := os.MkdirAll(derived, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{derivedPyramidName(resourceID), derivedPyramidManifestName(resourceID)} {
		if err := os.WriteFile(filepath.Join(derived, name), []byte("stale"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	late := register()
	if late.Code != http.StatusConflict {
		t.Fatalf("late register = %d, %s; want lifecycle conflict", late.Code, late.Body.String())
	}
	for _, path := range []string{
		absolute,
		filepath.Join(derived, derivedPyramidName(resourceID)),
		filepath.Join(derived, derivedPyramidManifestName(resourceID)),
	} {
		if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("late analysis generation survived at %q: %v", path, err)
		}
	}
	uploadRootHandle, err := os.OpenRoot(uploadRoot)
	if err != nil {
		t.Fatal(err)
	}
	defer uploadRootHandle.Close()
	if _, err := acquireResourceLifecycleLock(ctx, uploadRootHandle, resourceID, ""); !errors.Is(err, errResourceLifecycleTombstoned) {
		t.Fatalf("publication lock after purge = %v, want filesystem tombstone rejection", err)
	}
}

func TestRegisterAnalysisOutputsPreservesSoftDeletedGenerationOnReplay(t *testing.T) {
	t.Parallel()
	mem := store.NewMemoryStore()
	uploadRoot := t.TempDir()
	router := NewRouter(ServerDeps{Version: "test", Store: mem, UploadRoot: uploadRoot})
	ctx := context.Background()
	const user, org = "analysis-restore-owner", "analysis-restore-org"
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID: "analysis-restore-input", OriginalName: "input.tif", SourceType: "upload", ResourceKind: "image",
		OwnerUserID: user, OwnerOrgID: org, Status: domain.ResourceStatusActive,
	}); err != nil {
		t.Fatal(err)
	}
	submit := analysisAuthedJSON(t, router, http.MethodPost, "/v2/analysis/batch", user, org, map[string]any{
		"model": "megaseg", "resource_ids": []string{"analysis-restore-input"},
	})
	if submit.Code != http.StatusAccepted {
		t.Fatalf("submit = %d, %s", submit.Code, submit.Body.String())
	}
	var submitted batchAnalysisJobResponse
	if err := json.Unmarshal(submit.Body.Bytes(), &submitted); err != nil {
		t.Fatal(err)
	}
	relative := filepath.Join("analysis", submitted.Job.JobID, "restorable-mask.tif")
	absolute := filepath.Join(uploadRoot, relative)
	if err := os.MkdirAll(filepath.Dir(absolute), 0o755); err != nil {
		t.Fatal(err)
	}
	original := []byte("retained-analysis-generation")
	if err := os.WriteFile(absolute, original, 0o644); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(original)
	digestHex := hex.EncodeToString(digest[:])
	register := func() *httptest.ResponseRecorder {
		return analysisAuthedJSON(t, router, http.MethodPost, "/v2/data-agent/jobs/"+submitted.Job.JobID+"/outputs", user, org, map[string]any{
			"outputs": []map[string]any{{
				"storage_path": filepath.ToSlash(relative), "original_name": "restorable-mask.tif",
				"content_type": "image/tiff", "artifact_kind": "mask", "sha256": digestHex,
			}},
		})
	}
	first := register()
	if first.Code != http.StatusCreated {
		t.Fatalf("initial register = %d, %s", first.Code, first.Body.String())
	}
	resourceID := analysisOutputResourceID(submitted.Job.JobID, relative)
	deleted := analysisAuthedJSON(t, router, http.MethodDelete, "/v2/resources/"+resourceID, user, org, nil)
	if deleted.Code != http.StatusOK {
		t.Fatalf("soft delete = %d, %s", deleted.Code, deleted.Body.String())
	}
	replayed := register()
	if replayed.Code != http.StatusConflict {
		t.Fatalf("soft-deleted replay = %d, %s; want lifecycle conflict", replayed.Code, replayed.Body.String())
	}
	retained, err := os.ReadFile(absolute)
	if err != nil {
		t.Fatalf("read retained analysis generation: %v", err)
	}
	if retainedDigest := sha256.Sum256(retained); retainedDigest != digest {
		t.Fatalf("soft-deleted analysis generation changed: got %x want %x", retainedDigest, digest)
	}
	restored := analysisAuthedJSON(t, router, http.MethodPost, "/v2/resources/"+resourceID+"/restore", user, org, nil)
	if restored.Code != http.StatusOK {
		t.Fatalf("restore = %d, %s", restored.Code, restored.Body.String())
	}
	record, err := mem.GetResourceForUser(ctx, resourceID, user, org)
	if err != nil {
		t.Fatal(err)
	}
	if record.Status != domain.ResourceStatusActive || record.SHA256 != digestHex || record.SizeBytes != int64(len(original)) {
		t.Fatalf("restored catalog generation = %+v, want exact original sha/size", record)
	}
	download := analysisAuthedGet(t, router, "/v2/resources/"+resourceID+"/download", user, org)
	if download.Code != http.StatusOK || !bytes.Equal(download.Body.Bytes(), original) {
		t.Fatalf("restored download = %d %q, want exact original bytes", download.Code, download.Body.Bytes())
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
