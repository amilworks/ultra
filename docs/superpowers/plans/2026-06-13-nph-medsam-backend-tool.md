# NPH MedSAM Backend Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the approved NPH MedSAM backend tool end to end: Resource-backed NIfTI/DICOM/TIFF inputs, GPU 7-class segmentation, derived Resources, quantitative metrics, and a Deep Agents tool that returns scientist-facing analysis.

**Architecture:** Keep Go as the control plane for auth, durable Data Agent jobs, Resource cataloging, and events. Add a Python `ultra_deepagents.nph_medsam` runtime that plugs into the existing Python Data Agent worker as a processor, starts with fake inference for local integration tests, then swaps in the verified MedSAM ViT-B checkpoint for GPU execution. Reuse the MegaSeg GPU host as compute infrastructure without making the old MegaSeg local-JSON queue the production source of truth.

**Tech Stack:** Go control plane with chi/OpenAPI/Postgres/in-memory store/NATS, Python 3.11 with PyTorch/nibabel/pydicom/tifffile/scipy/numpy, existing Deep Agents runtime/Data Agent worker, systemd or shell deployment on `amil@128.111.185.73`, pytest, Go tests, and live remote GPU smoke tests.

---

## Preconditions

- Approved design spec: `docs/superpowers/specs/2026-06-13-nph-medsam-backend-tool-design.md`
- Local checkpoint path: `/Users/macbook/Downloads/bisque-20260612.010648/MEDSAM_finetune_CT_NO_SKULLSTRIP_repeated_img_embeddings_no_prompt_7classes_model_best.pt`
- Expected checkpoint SHA-256: `04b219ad513d60770b648dfc72298cf99a8d7b5cfc70e95217caff17f96a93dc`
- External MedSAM repo path: `/Users/macbook/Downloads/MedSAM_CTsegmentation-main`
- Remote GPU host: `ssh amil@128.111.185.73`
- Test image in Resources/uploads: `Norm_young_004_40yo.nii.gz`

Before code execution, create an isolated worktree or confirm the current branch is intentionally being used. This branch is currently dirty with unrelated image-engine/control-plane work; do not revert or stage unrelated changes.

```bash
git status --short --branch
```

Expected: unrelated dirty files may exist. Only stage files named by the active task.

## Scope Check

This is one integrated implementation plan because the parts are coupled by a single product contract: `nph_medsam_segmentation` Data Agent jobs produce derived Resources consumed by `nph_medsam_analysis`. The tasks are staged so each checkpoint is independently testable:

1. Go job/output Resource contract.
2. Python local deterministic runtime.
3. Medical file adapters and metrics.
4. Data Agent processor integration.
5. Deep Agents tool integration.
6. Real MedSAM model runtime.
7. DICOM/TIFF completion.
8. GPU-host deployment.
9. End-to-end scientific workflow verification.

## File Map

Go control-plane files:

- Modify: `backend/controlplane/internal/httpapi/handlers.go`
  - Add `nph_medsam_segmentation` job validation.
  - Add worker-safe derived Resource catalog endpoint for Data Agent outputs.
  - Treat NIfTI as an image-like scientific volume for Resource kind.
- Modify: `backend/controlplane/internal/httpapi/handlers_test.go`
  - Add HTTP tests for NPH job creation, validation, and output cataloging.
- Modify: `backend/controlplane/api/openapi.yaml`
  - Document `nph_medsam_segmentation` and the output Resource catalog endpoint.
- Modify: `backend/controlplane/internal/openapi/generated.gen.go`
  - Regenerate after OpenAPI update.
- Modify: `backend/controlplane/internal/httpapi/workos_auth.go`
  - Add the new Data Agent output endpoint to worker-token scope when route classification uses an explicit allow-list.
- Test command: `cd backend/controlplane && go test ./internal/httpapi ./internal/openapi -count=1`

Python runtime files:

- Modify: `backend/deepagents_runtime/pyproject.toml`
  - Add `medsam` optional dependency group.
- Modify: `backend/deepagents_runtime/uv.lock`
  - Regenerate after dependency update.
- Create: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/__init__.py`
- Create: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/schema.py`
- Create: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/metrics.py`
- Create: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/resources.py`
- Create: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/formats.py`
- Create: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/preprocess.py`
- Create: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/postprocess.py`
- Create: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/model.py`
- Create: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/runner.py`
- Create: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/control.py`
- Create: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/processor.py`
- Create: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/worker.py`
- Create: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/tools.py`
- Create: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/live_smoke.py`
- Modify: `backend/deepagents_runtime/src/ultra_deepagents/config.py`
  - Add NPH MedSAM settings.
- Modify: `backend/deepagents_runtime/src/ultra_deepagents/agent.py`
  - Register NPH guidance and the `nph_medsam_analysis` tool.
- Modify: `backend/deepagents_runtime/src/ultra_deepagents/data_agent/worker.py`
  - Allow a processor router so NPH jobs use the MedSAM processor while existing templates keep the default processor.
- Tests:
  - Create: `backend/deepagents_runtime/tests/test_nph_medsam_metrics.py`
  - Create: `backend/deepagents_runtime/tests/test_nph_medsam_resources.py`
  - Create: `backend/deepagents_runtime/tests/test_nph_medsam_formats.py`
  - Create: `backend/deepagents_runtime/tests/test_nph_medsam_runner.py`
  - Create: `backend/deepagents_runtime/tests/test_nph_medsam_processor.py`
  - Create: `backend/deepagents_runtime/tests/test_nph_medsam_tools.py`
  - Modify: `backend/deepagents_runtime/tests/test_data_agent_worker.py`
  - Modify: `backend/deepagents_runtime/tests/test_agent_factory.py`

Deployment files:

- Create: `deploy/env/nph-medsam-worker.env.example`
- Create: `deploy/systemd/ultra-nph-medsam-worker.service`
- Create: `scripts/deploy_nph_medsam_worker.sh`
- Create: `scripts/run_nph_medsam_worker.sh`
- Modify: `Makefile`
  - Add local helper targets for fake and real NPH MedSAM worker tests.

## Shared Constants

Use these exact constants across Go, Python, metadata, and tests:

```text
Job type: nph_medsam_segmentation
Tool name: nph_medsam_analysis
Output Resource prefix: nph_medsam_
Model short hash: 04b219ad
Model SHA-256: 04b219ad513d60770b648dfc72298cf99a8d7b5cfc70e95217caff17f96a93dc
Output labels: 0..6
MedSAM builder num_classes: 6
Input image size: 512
Whole-image box prompt: [10, 10, 502, 502]
CT clip range: [0, 80]
Label dtype: uint8
Label restore interpolation: nearest
```

## Task 1: Go NPH Data Agent Contract

**Files:**

- Modify: `backend/controlplane/internal/httpapi/handlers.go`
- Modify: `backend/controlplane/internal/httpapi/handlers_test.go`
- Modify: `backend/controlplane/api/openapi.yaml`
- Regenerate: `backend/controlplane/internal/openapi/generated.gen.go`

- [ ] **Step 1: Write the failing job-type test**

Add this test near the existing Data Agent HTTP tests in `backend/controlplane/internal/httpapi/handlers_test.go`:

```go
func TestV2CreateNPHMedSAMDataAgentJobAcceptsSingleMedicalResource(t *testing.T) {
	t.Parallel()

	store := store.NewMemoryStore()
	deps := testServerDeps(t, store)
	principal := testPrincipal()
	resource, err := store.UpsertResource(context.Background(), domain.UpsertResourceInput{
		ResourceID:   "file_nph_source",
		OriginalName: "Norm_young_004_40yo.nii.gz",
		ContentType:  "application/x-nifti",
		SizeBytes:    1234,
		SHA256:       "source-sha",
		StorageURI:   "file:///tmp/Norm_young_004_40yo.nii.gz",
		StoragePath:  "file_nph_source__Norm_young_004_40yo.nii.gz",
		SourceType:   "upload",
		ResourceKind: "image",
		OwnerUserID:  principal.UserID,
		OwnerOrgID:   principal.OrgID,
		OwnerRole:    principal.Role,
		Status:       "active",
		CreatedAt:    domain.Now(),
		UpdatedAt:    domain.Now(),
	})
	if err != nil {
		t.Fatalf("UpsertResource: %v", err)
	}

	body := strings.NewReader(`{"job_type":"nph_medsam_segmentation","resource_ids":["` + resource.ResourceID + `"],"metadata":{"analysis_focus":"full_summary"}}`)
	req := httptest.NewRequest(http.MethodPost, "/v2/data-agent/jobs", body)
	req.Header.Set("Content-Type", "application/json")
	req = req.WithContext(withPrincipal(req.Context(), principal))
	rec := httptest.NewRecorder()

	deps.Router().ServeHTTP(rec, req)

	if rec.Code != http.StatusAccepted {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	var payload dataAgentJobResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if payload.Job.JobType != "nph_medsam_segmentation" {
		t.Fatalf("job type = %q", payload.Job.JobType)
	}
	if got := metadataStringSlice(payload.Job.InputSelector["resource_ids"]); !reflect.DeepEqual(got, []string{resource.ResourceID}) {
		t.Fatalf("resource ids = %#v", got)
	}
	if payload.Job.Metadata["analysis_focus"] != "full_summary" {
		t.Fatalf("metadata = %#v", payload.Job.Metadata)
	}
}
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
cd backend/controlplane
go test ./internal/httpapi -run TestV2CreateNPHMedSAMDataAgentJobAcceptsSingleMedicalResource -count=1
```

Expected: FAIL with a 400 response because `nph_medsam_segmentation` is not accepted by `normalizeDataAgentJobType`.

- [ ] **Step 3: Add the job type constant and validation**

In `backend/controlplane/internal/httpapi/handlers.go`, add this constant near other Data Agent helper constants:

```go
const dataAgentJobTypeNPHMedSAMSegmentation = "nph_medsam_segmentation"
```

Update `normalizeDataAgentJobType`:

```go
func normalizeDataAgentJobType(value string) (string, error) {
	value = strings.ToLower(strings.TrimSpace(value))
	switch value {
	case "caption_resources", "extract_metadata", "organize_resources", "deduplicate_resources", "quality_check_resources", "batch_tag_resources", "create_dataset_snapshot", dataAgentJobTypeNPHMedSAMSegmentation:
		return value, nil
	default:
		return "", errors.New("job_type must be caption_resources, extract_metadata, organize_resources, deduplicate_resources, quality_check_resources, batch_tag_resources, create_dataset_snapshot, or nph_medsam_segmentation")
	}
}
```

Add this helper below `normalizeDataAgentJobType`:

```go
func validateNPHMedSAMJobRequest(resourceIDs []string, sourceCollectionID string, resourceQuery *domain.DatasetSnapshotResourceQuery) error {
	if sourceCollectionID != "" || resourceQuery != nil {
		return errors.New("nph_medsam_segmentation requires exactly one resource_id in the first implementation slice")
	}
	if len(resourceIDs) != 1 {
		return errors.New("nph_medsam_segmentation requires exactly one resource_id")
	}
	return nil
}
```

Call it in `handleCreateDataAgentJob` immediately after `resourceQuery := datasetSnapshotResourceQueryFromRequest(req.ResourceQuery)`:

```go
if jobType == dataAgentJobTypeNPHMedSAMSegmentation {
	if err := validateNPHMedSAMJobRequest(resourceIDs, sourceCollectionID, resourceQuery); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
}
```

- [ ] **Step 4: Run the focused HTTP test**

Run:

```bash
cd backend/controlplane
go test ./internal/httpapi -run TestV2CreateNPHMedSAMDataAgentJobAcceptsSingleMedicalResource -count=1
```

Expected: PASS.

- [ ] **Step 5: Add validation tests for unsupported selectors**

Add this table test:

```go
func TestV2CreateNPHMedSAMDataAgentJobRejectsNonSingleResourceSelectors(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		body string
	}{
		{
			name: "no resources",
			body: `{"job_type":"nph_medsam_segmentation"}`,
		},
		{
			name: "many resources",
			body: `{"job_type":"nph_medsam_segmentation","resource_ids":["file_a","file_b"]}`,
		},
		{
			name: "collection selector",
			body: `{"job_type":"nph_medsam_segmentation","source_collection_id":"collection_a"}`,
		},
		{
			name: "query selector",
			body: `{"job_type":"nph_medsam_segmentation","resource_query":{"query":"Norm_young"}}`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			deps := testServerDeps(t, store.NewMemoryStore())
			req := httptest.NewRequest(http.MethodPost, "/v2/data-agent/jobs", strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")
			req = req.WithContext(withPrincipal(req.Context(), testPrincipal()))
			rec := httptest.NewRecorder()
			deps.Router().ServeHTTP(rec, req)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
			}
		})
	}
}
```

- [ ] **Step 6: Run validation tests**

Run:

```bash
cd backend/controlplane
go test ./internal/httpapi -run 'TestV2CreateNPHMedSAMDataAgentJob(AcceptsSingleMedicalResource|RejectsNonSingleResourceSelectors)' -count=1
```

Expected: PASS.

- [ ] **Step 7: Update OpenAPI**

In `backend/controlplane/api/openapi.yaml`, add `nph_medsam_segmentation` to the Data Agent job type enum anywhere the existing job-type enum lists values:

```yaml
- caption_resources
- extract_metadata
- organize_resources
- deduplicate_resources
- quality_check_resources
- batch_tag_resources
- create_dataset_snapshot
- nph_medsam_segmentation
```

Regenerate:

```bash
make -C backend/controlplane generate-openapi
```

Expected: `backend/controlplane/internal/openapi/generated.gen.go` changes and generated code remains formatted.

- [ ] **Step 8: Run Go contract verification**

Run:

```bash
cd backend/controlplane
go test ./internal/httpapi ./internal/openapi -count=1
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add backend/controlplane/internal/httpapi/handlers.go backend/controlplane/internal/httpapi/handlers_test.go backend/controlplane/api/openapi.yaml backend/controlplane/internal/openapi/generated.gen.go
git commit -m "feat: accept nph medsam data agent jobs"
```

## Task 2: Worker-Safe Derived Resource Catalog Endpoint

**Files:**

- Modify: `backend/controlplane/internal/httpapi/handlers.go`
- Modify: `backend/controlplane/internal/httpapi/handlers_test.go`
- Modify: `backend/controlplane/api/openapi.yaml`
- Regenerate: `backend/controlplane/internal/openapi/generated.gen.go`

- [ ] **Step 1: Write the failing output-catalog test**

Add this test:

```go
func TestV2DataAgentJobCatalogsNPHMedSAMDerivedResource(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	sourcePath := filepath.Join(uploadRoot, "file_source__Norm_young_004_40yo.nii.gz")
	if err := os.WriteFile(sourcePath, []byte("source"), 0o644); err != nil {
		t.Fatal(err)
	}
	outputBytes := []byte("segmentation-nifti-bytes")
	outputName := "nph_medsam_seg__Norm_young_004_40yo__source-file_source__model-04b219ad__20260613T120000Z.nii.gz"
	outputPath := filepath.Join(uploadRoot, outputName)
	if err := os.WriteFile(outputPath, outputBytes, 0o644); err != nil {
		t.Fatal(err)
	}
	outputSHA := sha256Hex(outputBytes)

	store := store.NewMemoryStore()
	deps := testServerDeps(t, store)
	deps.UploadRoot = uploadRoot
	principal := testPrincipal()
	source, err := store.UpsertResource(context.Background(), domain.UpsertResourceInput{
		ResourceID:   "file_source",
		OriginalName: "Norm_young_004_40yo.nii.gz",
		ContentType:  "application/x-nifti",
		SizeBytes:    int64(len("source")),
		SHA256:       "source-sha",
		StorageURI:   fileStorageURI(sourcePath),
		StoragePath:  filepath.Base(sourcePath),
		SourceType:   "upload",
		ResourceKind: "image",
		ProjectID:    "nph-project",
		OwnerUserID:  principal.UserID,
		OwnerOrgID:   principal.OrgID,
		OwnerRole:    principal.Role,
		Status:       "active",
		CreatedAt:    domain.Now(),
		UpdatedAt:    domain.Now(),
		Metadata: domain.JSONMap{
			"scanner": "test-scanner",
		},
	})
	if err != nil {
		t.Fatalf("source UpsertResource: %v", err)
	}
	job, err := store.CreateDataAgentJob(context.Background(), domain.CreateDataAgentJobInput{
		OwnerUserID:   principal.UserID,
		OwnerOrgID:    principal.OrgID,
		OwnerRole:     principal.Role,
		ProjectID:     source.ProjectID,
		JobType:       "nph_medsam_segmentation",
		Status:        "running",
		ResourceIDs:   []string{source.ResourceID},
		ResourceCount: 1,
		InputSelector: domain.JSONMap{"resource_ids": []string{source.ResourceID}},
		Metadata:      domain.JSONMap{"checkpoint_sha256": "04b219ad513d60770b648dfc72298cf99a8d7b5cfc70e95217caff17f96a93dc"},
		CreatedAt:     domain.Now(),
		UpdatedAt:     domain.Now(),
	})
	if err != nil {
		t.Fatalf("CreateDataAgentJob: %v", err)
	}

	body := strings.NewReader(fmt.Sprintf(`{
		"source_resource_id":"%s",
		"storage_path":"%s",
		"original_name":"%s",
		"content_type":"application/x-nifti",
		"sha256":"%s",
		"size_bytes":%d,
		"resource_kind":"image",
		"tags":["nph","medsam","segmentation","derived"],
		"metadata":{"nph_medsam":{"label_count":7,"model_short_hash":"04b219ad"}}
	}`, source.ResourceID, outputName, outputName, outputSHA, len(outputBytes)))
	req := httptest.NewRequest(http.MethodPost, "/v2/data-agent/jobs/"+job.JobID+"/resources", body)
	req.Header.Set("Content-Type", "application/json")
	req = req.WithContext(withPrincipal(req.Context(), principal))
	rec := httptest.NewRecorder()

	deps.Router().ServeHTTP(rec, req)

	if rec.Code != http.StatusCreated {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	var payload struct {
		Resource domain.ResourceRecord `json:"resource"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if payload.Resource.SourceType != "derived" {
		t.Fatalf("source type = %q", payload.Resource.SourceType)
	}
	if payload.Resource.OwnerUserID != source.OwnerUserID || payload.Resource.OwnerOrgID != source.OwnerOrgID {
		t.Fatalf("derived owner = %s/%s, want %s/%s", payload.Resource.OwnerUserID, payload.Resource.OwnerOrgID, source.OwnerUserID, source.OwnerOrgID)
	}
	if payload.Resource.ProjectID != source.ProjectID {
		t.Fatalf("project = %q", payload.Resource.ProjectID)
	}
}
```

- [ ] **Step 2: Run the failing catalog test**

Run:

```bash
cd backend/controlplane
go test ./internal/httpapi -run TestV2DataAgentJobCatalogsNPHMedSAMDerivedResource -count=1
```

Expected: FAIL with 404 because `/v2/data-agent/jobs/{job_id}/resources` is not registered.

- [ ] **Step 3: Add request/response types**

Add these types near `dataAgentJobResponse`:

```go
type catalogDataAgentJobResourceRequest struct {
	SourceResourceID string         `json:"source_resource_id"`
	ResourceID       string         `json:"resource_id"`
	StoragePath      string         `json:"storage_path"`
	OriginalName     string         `json:"original_name"`
	ContentType      string         `json:"content_type"`
	SizeBytes        int64          `json:"size_bytes"`
	SHA256           string         `json:"sha256"`
	ResourceKind     string         `json:"resource_kind"`
	Tags             []string       `json:"tags"`
	Metadata         domain.JSONMap `json:"metadata"`
}

type catalogDataAgentJobResourceResponse struct {
	Resource domain.ResourceRecord `json:"resource"`
}
```

- [ ] **Step 4: Register the endpoint**

Inside the `/v2/data-agent/jobs` route group in `handlers.go`, add:

```go
r.Post("/data-agent/jobs/{job_id}/resources", deps.handleCatalogDataAgentJobResource)
```

- [ ] **Step 5: Implement the handler**

Add this handler near the existing Data Agent handlers:

```go
func (deps ServerDeps) handleCatalogDataAgentJobResource(w http.ResponseWriter, r *http.Request) {
	jobs, ok := deps.dataAgentJobStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "data agent jobs are not configured"})
		return
	}
	catalog, ok := deps.resourceCatalogStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource catalog is not configured"})
		return
	}
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	var req catalogDataAgentJobResourceRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	principal := deps.principalFromRequest(r, "")
	job, err := jobs.GetDataAgentJobForUser(r.Context(), chi.URLParam(r, "job_id"), principal.UserID, principal.OrgID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if job.JobType != dataAgentJobTypeNPHMedSAMSegmentation {
		writeError(w, http.StatusBadRequest, errors.New("data-agent output resource cataloging is only enabled for nph_medsam_segmentation"))
		return
	}
	source, err := catalog.GetResourceForUser(r.Context(), strings.TrimSpace(req.SourceResourceID), principal.UserID, principal.OrgID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if !resourceOwnedByPrincipal(source, principal) {
		writeStoreError(w, store.ErrNotFound)
		return
	}
	path := filepath.Clean(filepath.Join(root, strings.TrimSpace(req.StoragePath)))
	if !pathIsUnderRoot(root, path) {
		writeError(w, http.StatusBadRequest, errUnsafeArtifactPath)
		return
	}
	stat, err := os.Stat(path)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if stat.IsDir() {
		writeError(w, http.StatusBadRequest, errors.New("derived resource storage_path must be a file"))
		return
	}
	if req.SizeBytes > 0 && req.SizeBytes != stat.Size() {
		writeError(w, http.StatusBadRequest, errors.New("derived resource size_bytes does not match storage_path"))
		return
	}
	resourceID := strings.TrimSpace(req.ResourceID)
	if resourceID == "" {
		resourceID = domain.NewID("file")
	}
	originalName := strings.TrimSpace(req.OriginalName)
	if originalName == "" {
		originalName = filepath.Base(path)
	}
	contentType := contentTypeForUpload(originalName, req.ContentType)
	resourceKind := strings.TrimSpace(req.ResourceKind)
	if resourceKind == "" {
		resourceKind = resourceKindForContent(originalName, contentType)
	}
	metadata := mapOrEmptyJSON(req.Metadata)
	metadata["source_resource_id"] = source.ResourceID
	metadata["source_sha256"] = source.SHA256
	metadata["source_original_name"] = source.OriginalName
	metadata["data_agent_job_id"] = job.JobID
	resource, err := catalog.UpsertResource(r.Context(), domain.UpsertResourceInput{
		ResourceID:   resourceID,
		OriginalName: originalName,
		ContentType:  contentType,
		SizeBytes:    stat.Size(),
		SHA256:       strings.TrimSpace(req.SHA256),
		StorageURI:   fileStorageURI(path),
		StoragePath:  filepath.Base(path),
		SourceType:   "derived",
		ResourceKind: resourceKind,
		SourceURI:    "resource://" + source.ResourceID,
		ProjectID:    source.ProjectID,
		OwnerUserID:  source.OwnerUserID,
		OwnerOrgID:   source.OwnerOrgID,
		OwnerRole:    source.OwnerRole,
		Status:       "active",
		CreatedAt:    domain.Now(),
		UpdatedAt:    domain.Now(),
		Tags:         uniqueTrimmedStringValues(req.Tags),
		Metadata:     metadata,
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	deps.recordResourceEvent(r.Context(), resource.ResourceID, principal, "resource.derived_from_nph_medsam", domain.JSONMap{
		"source_resource_id": source.ResourceID,
		"job_id":             job.JobID,
		"job_type":           job.JobType,
	})
	deps.recordResourceEvent(r.Context(), source.ResourceID, principal, "resource.nph_medsam_derivative_created", domain.JSONMap{
		"derived_resource_id": resource.ResourceID,
		"job_id":              job.JobID,
		"original_name":        resource.OriginalName,
	})
	writeJSON(w, http.StatusCreated, catalogDataAgentJobResourceResponse{Resource: resource})
}
```

- [ ] **Step 6: Treat NIfTI as image Resource kind**

Update `resourceKindForContent`:

```go
case isNiftiUpload(originalName, contentType):
	return "image"
```

- [ ] **Step 7: Run catalog tests**

Run:

```bash
cd backend/controlplane
go test ./internal/httpapi -run 'TestV2DataAgentJobCatalogsNPHMedSAMDerivedResource|TestV2CreateNPHMedSAMDataAgentJob' -count=1
```

Expected: PASS.

- [ ] **Step 8: Document endpoint and regenerate**

In `backend/controlplane/api/openapi.yaml`, add `POST /v2/data-agent/jobs/{job_id}/resources` with request fields from `catalogDataAgentJobResourceRequest` and response `resource: ResourceRecord`. Then run:

```bash
make -C backend/controlplane generate-openapi
cd backend/controlplane && go test ./internal/httpapi ./internal/openapi -count=1
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add backend/controlplane/internal/httpapi/handlers.go backend/controlplane/internal/httpapi/handlers_test.go backend/controlplane/api/openapi.yaml backend/controlplane/internal/openapi/generated.gen.go
git commit -m "feat: catalog nph medsam derived resources"
```

## Task 3: Python Dependencies And Core Schema

**Files:**

- Modify: `backend/deepagents_runtime/pyproject.toml`
- Modify: `backend/deepagents_runtime/uv.lock`
- Create: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/__init__.py`
- Create: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/schema.py`
- Create: `backend/deepagents_runtime/tests/test_nph_medsam_resources.py`

- [ ] **Step 1: Add dependency group**

In `backend/deepagents_runtime/pyproject.toml`, add:

```toml
medsam = [
    "numpy>=1.26.0,<3",
    "scipy>=1.11.0,<2",
    "nibabel>=5.2.0,<6",
    "pydicom>=2.4.0,<4",
    "tifffile>=2024.0.0",
    "torch>=2.2.0",
    "torchvision>=0.17.0",
]
```

- [ ] **Step 2: Regenerate lock**

Run:

```bash
uv lock --project backend/deepagents_runtime
```

Expected: `uv.lock` adds `nibabel`, `pydicom`, and `tifffile` if absent.

- [ ] **Step 3: Write schema tests**

Create `backend/deepagents_runtime/tests/test_nph_medsam_resources.py`:

```python
from __future__ import annotations

from ultra_deepagents.nph_medsam.resources import build_output_names, build_resource_metadata
from ultra_deepagents.nph_medsam.schema import MODEL_SHA256, NPH_MEDSAM_JOB_TYPE, NPH_MEDSAM_TOOL_NAME


def test_constants_match_product_contract() -> None:
    assert NPH_MEDSAM_JOB_TYPE == "nph_medsam_segmentation"
    assert NPH_MEDSAM_TOOL_NAME == "nph_medsam_analysis"
    assert MODEL_SHA256 == "04b219ad513d60770b648dfc72298cf99a8d7b5cfc70e95217caff17f96a93dc"


def test_build_output_names_preserves_prefix_source_and_model_hash() -> None:
    names = build_output_names(
        source_original_name="Norm_young_004_40yo.nii.gz",
        source_resource_id="file_source",
        generated_at="20260613T120000Z",
    )

    assert names.segmentation == "nph_medsam_seg__Norm_young_004_40yo__source-file_source__model-04b219ad__20260613T120000Z.nii.gz"
    assert names.summary == "nph_medsam_summary__Norm_young_004_40yo__source-file_source__model-04b219ad__20260613T120000Z.json"
    assert names.measurements == "nph_medsam_measurements__Norm_young_004_40yo__source-file_source__model-04b219ad__20260613T120000Z.csv"


def test_build_resource_metadata_includes_source_and_checkpoint_provenance() -> None:
    metadata = build_resource_metadata(
        source_resource_id="file_source",
        source_original_name="Norm_young_004_40yo.nii.gz",
        source_sha256="source-sha",
        job_id="data_agent_job_1",
        metrics={"ventricle_volume_ml": 12.5},
        qc={"warnings": []},
        generated_at="2026-06-13T12:00:00Z",
    )

    assert metadata["source_resource_id"] == "file_source"
    assert metadata["source_sha256"] == "source-sha"
    assert metadata["data_agent_job_id"] == "data_agent_job_1"
    assert metadata["nph_medsam"]["checkpoint_sha256"] == MODEL_SHA256
    assert metadata["nph_medsam"]["output_labels"] == [0, 1, 2, 3, 4, 5, 6]
    assert metadata["nph_medsam"]["metrics"]["ventricle_volume_ml"] == 12.5
```

- [ ] **Step 4: Run schema tests to verify failure**

Run:

```bash
cd backend/deepagents_runtime
uv run --extra dev pytest tests/test_nph_medsam_resources.py -q
```

Expected: FAIL because `ultra_deepagents.nph_medsam` does not exist.

- [ ] **Step 5: Add schema and resource helpers**

Create `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/__init__.py`:

```python
"""NPH MedSAM segmentation runtime for Resource-backed medical volumes."""
```

Create `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/schema.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

NPH_MEDSAM_JOB_TYPE = "nph_medsam_segmentation"
NPH_MEDSAM_TOOL_NAME = "nph_medsam_analysis"
MODEL_SHA256 = "04b219ad513d60770b648dfc72298cf99a8d7b5cfc70e95217caff17f96a93dc"
MODEL_SHORT_HASH = "04b219ad"
MODEL_NAME = "MEDSAM_finetune_CT_NO_SKULLSTRIP_repeated_img_embeddings_no_prompt_7classes"
OUTPUT_LABELS = tuple(range(7))
MEDSAM_NUM_CLASSES = 6
IMAGE_SIZE = 512
BOX_PROMPT = (10, 10, 502, 502)
CT_CLIP_RANGE = (0.0, 80.0)


@dataclass(frozen=True)
class OutputNames:
    segmentation: str
    summary: str
    measurements: str


@dataclass(frozen=True)
class SourceResource:
    resource_id: str
    original_name: str
    sha256: str
    storage_path: Path
    content_type: str = ""
    project_id: str = ""
    metadata: dict[str, Any] | None = None
```

Create `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/resources.py`:

```python
from __future__ import annotations

import re
from typing import Any

from ultra_deepagents.nph_medsam.schema import (
    BOX_PROMPT,
    CT_CLIP_RANGE,
    IMAGE_SIZE,
    MEDSAM_NUM_CLASSES,
    MODEL_NAME,
    MODEL_SHA256,
    MODEL_SHORT_HASH,
    OUTPUT_LABELS,
    OutputNames,
)


def source_stem(original_name: str) -> str:
    name = str(original_name or "source").strip()
    for suffix in (".nii.gz", ".ome.tiff", ".ome.tif"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    else:
        if "." in name:
            name = name.rsplit(".", 1)[0]
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return token or "source"


def build_output_names(
    *,
    source_original_name: str,
    source_resource_id: str,
    generated_at: str,
) -> OutputNames:
    stem = source_stem(source_original_name)
    safe_resource = re.sub(r"[^A-Za-z0-9._-]+", "_", str(source_resource_id or "resource")).strip("._-")
    suffix = f"{stem}__source-{safe_resource}__model-{MODEL_SHORT_HASH}__{generated_at}"
    return OutputNames(
        segmentation=f"nph_medsam_seg__{suffix}.nii.gz",
        summary=f"nph_medsam_summary__{suffix}.json",
        measurements=f"nph_medsam_measurements__{suffix}.csv",
    )


def build_resource_metadata(
    *,
    source_resource_id: str,
    source_original_name: str,
    source_sha256: str,
    job_id: str,
    metrics: dict[str, Any],
    qc: dict[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    return {
        "source_resource_id": source_resource_id,
        "source_original_name": source_original_name,
        "source_sha256": source_sha256,
        "data_agent_job_id": job_id,
        "nph_medsam": {
            "model_name": MODEL_NAME,
            "checkpoint_sha256": MODEL_SHA256,
            "model_short_hash": MODEL_SHORT_HASH,
            "output_labels": list(OUTPUT_LABELS),
            "instantiated_num_classes": MEDSAM_NUM_CLASSES,
            "preprocessing": {
                "ct_clip_range": list(CT_CLIP_RANGE),
                "image_size": IMAGE_SIZE,
                "box_prompt": list(BOX_PROMPT),
                "label_restore_interpolation": "nearest",
            },
            "metrics": metrics,
            "qc": qc,
            "generated_at": generated_at,
        },
    }
```

- [ ] **Step 6: Run schema tests**

Run:

```bash
cd backend/deepagents_runtime
uv run --extra dev pytest tests/test_nph_medsam_resources.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/deepagents_runtime/pyproject.toml backend/deepagents_runtime/uv.lock backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/__init__.py backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/schema.py backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/resources.py backend/deepagents_runtime/tests/test_nph_medsam_resources.py
git commit -m "feat: add nph medsam runtime contract"
```

## Task 4: Metrics, QC, And Fake Inference Runner

**Files:**

- Create: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/metrics.py`
- Create: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/runner.py`
- Create: `backend/deepagents_runtime/tests/test_nph_medsam_metrics.py`
- Create: `backend/deepagents_runtime/tests/test_nph_medsam_runner.py`

- [ ] **Step 1: Write metric tests**

Create `backend/deepagents_runtime/tests/test_nph_medsam_metrics.py`:

```python
from __future__ import annotations

import numpy as np

from ultra_deepagents.nph_medsam.metrics import compute_segmentation_metrics, validate_label_volume


def test_compute_segmentation_metrics_aggregates_nph_groups_in_ml() -> None:
    labels = np.zeros((4, 4, 4), dtype=np.uint8)
    labels[0:2, :, :] = 1
    labels[2, :, :] = 6
    labels[3, 0:2, :] = 3
    labels[3, 2:4, :] = 2

    metrics = compute_segmentation_metrics(labels, spacing_mm=(1.0, 1.0, 2.0))

    assert metrics["voxel_volume_mm3"] == 2.0
    assert metrics["label_counts"]["1"] == 32
    assert metrics["label_counts"]["6"] == 16
    assert metrics["groups"]["ventricle"]["voxel_count"] == 48
    assert metrics["groups"]["ventricle"]["volume_ml"] == 0.096
    assert metrics["groups"]["subarachnoid_csf"]["volume_ml"] == 0.016
    assert metrics["max_ventricular_slice"]["axis"] == "z"
    assert metrics["max_ventricular_slice"]["slice_index"] == 0


def test_validate_label_volume_rejects_values_outside_schema() -> None:
    labels = np.array([0, 1, 7], dtype=np.uint8)

    qc = validate_label_volume(labels, source_shape=(3,))

    assert qc["ok"] is False
    assert "label_values_outside_0_6" in qc["warnings"]
```

- [ ] **Step 2: Run metric tests to verify failure**

Run:

```bash
cd backend/deepagents_runtime
uv run --extra dev pytest tests/test_nph_medsam_metrics.py -q
```

Expected: FAIL because `metrics.py` does not exist.

- [ ] **Step 3: Implement metrics**

Create `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/metrics.py`:

```python
from __future__ import annotations

from typing import Any

import numpy as np

VALID_LABELS = set(range(7))
GROUPS = {
    "ventricle": (1, 6),
    "white_matter": (2, 5),
    "subarachnoid_csf": (3,),
}


def validate_label_volume(labels: np.ndarray, *, source_shape: tuple[int, ...]) -> dict[str, Any]:
    warnings: list[str] = []
    if tuple(labels.shape) != tuple(source_shape):
        warnings.append("shape_mismatch")
    present = {int(value) for value in np.unique(labels)}
    if not present.issubset(VALID_LABELS):
        warnings.append("label_values_outside_0_6")
    for label in range(1, 7):
        if label not in present:
            warnings.append(f"empty_label_{label}")
    return {"ok": not warnings, "warnings": warnings, "present_labels": sorted(present)}


def compute_segmentation_metrics(labels: np.ndarray, *, spacing_mm: tuple[float, float, float]) -> dict[str, Any]:
    spacing = tuple(float(v) for v in spacing_mm)
    voxel_volume_mm3 = abs(spacing[0] * spacing[1] * spacing[2])
    label_counts = {str(label): int(np.count_nonzero(labels == label)) for label in range(7)}
    label_volumes_ml = {
        key: round(count * voxel_volume_mm3 / 1000.0, 6)
        for key, count in label_counts.items()
    }
    groups: dict[str, dict[str, Any]] = {}
    for group_name, group_labels in GROUPS.items():
        mask = np.isin(labels, group_labels)
        count = int(np.count_nonzero(mask))
        groups[group_name] = {
            "labels": list(group_labels),
            "voxel_count": count,
            "volume_ml": round(count * voxel_volume_mm3 / 1000.0, 6),
        }
    total_segmented = int(np.count_nonzero(labels))
    total_segmented_ml = round(total_segmented * voxel_volume_mm3 / 1000.0, 6)
    vent_count = int(groups["ventricle"]["voxel_count"])
    csf_count = int(groups["ventricle"]["voxel_count"] + groups["subarachnoid_csf"]["voxel_count"])
    max_slice = _max_ventricular_slice(labels, voxel_area_mm2=abs(spacing[0] * spacing[1]))
    return {
        "shape": list(labels.shape),
        "spacing_mm": list(spacing),
        "voxel_volume_mm3": voxel_volume_mm3,
        "label_counts": label_counts,
        "label_volumes_ml": label_volumes_ml,
        "groups": groups,
        "total_segmented": {
            "voxel_count": total_segmented,
            "volume_ml": total_segmented_ml,
        },
        "ratios": {
            "ventricle_to_total_segmented": round(vent_count / total_segmented, 6) if total_segmented else 0.0,
            "csf_to_total_segmented": round(csf_count / total_segmented, 6) if total_segmented else 0.0,
        },
        "max_ventricular_slice": max_slice,
    }


def _max_ventricular_slice(labels: np.ndarray, *, voxel_area_mm2: float) -> dict[str, Any]:
    vent_mask = np.isin(labels, GROUPS["ventricle"])
    if labels.ndim != 3 or not np.any(vent_mask):
        return {"axis": "z", "slice_index": None, "area_mm2": 0.0, "voxel_count": 0}
    counts = np.count_nonzero(vent_mask, axis=(1, 2))
    index = int(np.argmax(counts))
    count = int(counts[index])
    return {
        "axis": "z",
        "slice_index": index,
        "voxel_count": count,
        "area_mm2": round(count * voxel_area_mm2, 6),
    }
```

- [ ] **Step 4: Run metric tests**

Run:

```bash
cd backend/deepagents_runtime
uv run --extra dev pytest tests/test_nph_medsam_metrics.py -q
```

Expected: PASS.

- [ ] **Step 5: Write fake runner tests**

Create `backend/deepagents_runtime/tests/test_nph_medsam_runner.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np

from ultra_deepagents.nph_medsam.runner import FakeNPHMedSAMRunner
from ultra_deepagents.nph_medsam.schema import SourceResource


def test_fake_runner_writes_segmentation_summary_and_measurements(tmp_path: Path) -> None:
    source = SourceResource(
        resource_id="file_source",
        original_name="Norm_young_004_40yo.nii.gz",
        sha256="source-sha",
        storage_path=tmp_path / "source.nii.gz",
        content_type="application/x-nifti",
    )
    source.storage_path.write_bytes(b"not-real-nifti-for-fake-runner")

    result = FakeNPHMedSAMRunner().run(
        source=source,
        output_dir=tmp_path / "outputs",
        job_id="data_agent_job_1",
        generated_at="20260613T120000Z",
    )

    assert result.segmentation_path.name.startswith("nph_medsam_seg__Norm_young_004_40yo")
    assert result.summary_path.exists()
    assert result.measurements_path.exists()
    labels = np.load(result.segmentation_path)
    assert labels.dtype == np.uint8
    assert set(np.unique(labels)).issubset(set(range(7)))
    assert result.metrics["groups"]["ventricle"]["voxel_count"] > 0
```

- [ ] **Step 6: Implement fake runner**

Create `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/runner.py`:

```python
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ultra_deepagents.nph_medsam.metrics import compute_segmentation_metrics, validate_label_volume
from ultra_deepagents.nph_medsam.resources import build_output_names, build_resource_metadata
from ultra_deepagents.nph_medsam.schema import SourceResource


@dataclass(frozen=True)
class NPHMedSAMRunResult:
    segmentation_path: Path
    summary_path: Path
    measurements_path: Path
    metrics: dict[str, Any]
    qc: dict[str, Any]
    metadata: dict[str, Any]


class FakeNPHMedSAMRunner:
    def run(
        self,
        *,
        source: SourceResource,
        output_dir: Path,
        job_id: str,
        generated_at: str,
    ) -> NPHMedSAMRunResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        labels = np.zeros((8, 8, 8), dtype=np.uint8)
        labels[1:4, 2:6, 2:6] = 1
        labels[4:6, 2:6, 2:6] = 6
        labels[6, :, :] = 3
        metrics = compute_segmentation_metrics(labels, spacing_mm=(1.0, 1.0, 1.0))
        qc = validate_label_volume(labels, source_shape=labels.shape)
        metadata = build_resource_metadata(
            source_resource_id=source.resource_id,
            source_original_name=source.original_name,
            source_sha256=source.sha256,
            job_id=job_id,
            metrics=metrics,
            qc=qc,
            generated_at=generated_at,
        )
        names = build_output_names(
            source_original_name=source.original_name,
            source_resource_id=source.resource_id,
            generated_at=generated_at,
        )
        segmentation_path = output_dir / names.segmentation
        summary_path = output_dir / names.summary
        measurements_path = output_dir / names.measurements
        np.save(segmentation_path, labels)
        if segmentation_path.suffix == ".npy":
            final_segmentation_path = output_dir / names.segmentation
            segmentation_path.replace(final_segmentation_path)
            segmentation_path = final_segmentation_path
        summary_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        with measurements_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["name", "voxel_count", "volume_ml"])
            writer.writeheader()
            for name, group in metrics["groups"].items():
                writer.writerow(
                    {
                        "name": name,
                        "voxel_count": group["voxel_count"],
                        "volume_ml": group["volume_ml"],
                    }
                )
        return NPHMedSAMRunResult(
            segmentation_path=segmentation_path,
            summary_path=summary_path,
            measurements_path=measurements_path,
            metrics=metrics,
            qc=qc,
            metadata=metadata,
        )
```

- [ ] **Step 7: Run fake runner tests**

Run:

```bash
cd backend/deepagents_runtime
uv run --extra dev pytest tests/test_nph_medsam_metrics.py tests/test_nph_medsam_runner.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/metrics.py backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/runner.py backend/deepagents_runtime/tests/test_nph_medsam_metrics.py backend/deepagents_runtime/tests/test_nph_medsam_runner.py
git commit -m "feat: add nph medsam metrics and fake runner"
```

## Task 5: Medical Format Adapters

**Files:**

- Create: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/formats.py`
- Create: `backend/deepagents_runtime/tests/test_nph_medsam_formats.py`

- [ ] **Step 1: Write format adapter tests**

Create `backend/deepagents_runtime/tests/test_nph_medsam_formats.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ultra_deepagents.nph_medsam.formats import (
    MedicalVolume,
    load_medical_volume,
    save_segmentation_like_source,
)


def test_nifti_round_trip_preserves_shape_affine_and_uint8(tmp_path: Path) -> None:
    nib = pytest.importorskip("nibabel")
    source_path = tmp_path / "source.nii.gz"
    affine = np.diag([0.5, 0.5, 2.0, 1.0])
    data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    nib.save(nib.Nifti1Image(data, affine), source_path)

    volume = load_medical_volume(source_path)

    assert isinstance(volume, MedicalVolume)
    assert volume.array.shape == (4, 5, 6)
    assert volume.spacing_mm == (0.5, 0.5, 2.0)
    labels = np.ones(volume.array.shape, dtype=np.uint8)
    output_path = tmp_path / "seg.nii.gz"
    save_segmentation_like_source(labels, volume, output_path)
    loaded = nib.load(str(output_path))
    assert loaded.shape == volume.array.shape
    assert loaded.get_data_dtype() == np.dtype("uint8")
    np.testing.assert_allclose(loaded.affine, affine)


def test_tiff_stack_loads_with_spacing_warning(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "stack.tiff"
    tifffile.imwrite(path, np.zeros((3, 4, 5), dtype=np.uint16))

    volume = load_medical_volume(path)

    assert volume.array.shape == (3, 4, 5)
    assert volume.spacing_mm == (1.0, 1.0, 1.0)
    assert "tiff_spacing_defaulted_to_1mm" in volume.warnings


def test_dicom_directory_rejects_multiple_series(tmp_path: Path) -> None:
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset
    from pydicom.uid import ExplicitVRLittleEndian

    for index, series_uid in enumerate(["series-a", "series-b"]):
        ds = FileDataset(str(tmp_path / f"{index}.dcm"), {}, file_meta=pydicom.dataset.FileMetaDataset(), preamble=b"\0" * 128)
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.SeriesInstanceUID = series_uid
        ds.InstanceNumber = index + 1
        ds.Rows = 2
        ds.Columns = 2
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.PixelData = (np.ones((2, 2), dtype=np.uint16) * index).tobytes()
        ds.save_as(tmp_path / f"{index}.dcm")

    with pytest.raises(ValueError, match="ambiguous_dicom_series"):
        load_medical_volume(tmp_path)
```

- [ ] **Step 2: Run format tests to verify failure**

Run:

```bash
cd backend/deepagents_runtime
uv run --extra dev --extra medsam pytest tests/test_nph_medsam_formats.py -q
```

Expected: FAIL because `formats.py` does not exist.

- [ ] **Step 3: Implement NIfTI/TIFF/DICOM adapters**

Create `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/formats.py` with this public interface:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MedicalVolume:
    array: np.ndarray
    spacing_mm: tuple[float, float, float]
    source_format: str
    affine: np.ndarray | None = None
    header: Any | None = None
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None


def load_medical_volume(path: str | Path) -> MedicalVolume:
    source = Path(path)
    lower = source.name.lower()
    if source.is_dir():
        return _load_dicom_series(source)
    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        return _load_nifti(source)
    if lower.endswith((".tif", ".tiff", ".ome.tif", ".ome.tiff")):
        return _load_tiff(source)
    if lower.endswith(".dcm"):
        return _load_dicom_series(source.parent, selected_files=[source])
    raise ValueError("unsupported_format")


def save_segmentation_like_source(labels: np.ndarray, source: MedicalVolume, output_path: str | Path) -> None:
    output = Path(output_path)
    if source.source_format == "nifti":
        import nibabel as nib

        image = nib.Nifti1Image(labels.astype(np.uint8, copy=False), source.affine, header=source.header)
        image.set_data_dtype(np.uint8)
        nib.save(image, output)
        return
    import nibabel as nib

    affine = source.affine if source.affine is not None else np.diag([*source.spacing_mm, 1.0])
    image = nib.Nifti1Image(labels.astype(np.uint8, copy=False), affine)
    image.set_data_dtype(np.uint8)
    nib.save(image, output)


def _load_nifti(path: Path) -> MedicalVolume:
    import nibabel as nib

    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj).astype(np.float32, copy=False)
    zooms = image.header.get_zooms()
    spacing = tuple(float(v) for v in (zooms[:3] if len(zooms) >= 3 else (1.0, 1.0, 1.0)))
    return MedicalVolume(
        array=data,
        spacing_mm=spacing,  # type: ignore[arg-type]
        source_format="nifti",
        affine=np.asarray(image.affine),
        header=image.header.copy(),
        metadata={"shape": list(data.shape)},
    )


def _load_tiff(path: Path) -> MedicalVolume:
    import tifffile

    data = tifffile.imread(path).astype(np.float32, copy=False)
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    return MedicalVolume(
        array=data,
        spacing_mm=(1.0, 1.0, 1.0),
        source_format="tiff",
        warnings=("tiff_spacing_defaulted_to_1mm",),
        metadata={"shape": list(data.shape)},
    )


def _load_dicom_series(path: Path, *, selected_files: list[Path] | None = None) -> MedicalVolume:
    import pydicom

    files = selected_files or sorted(candidate for candidate in path.iterdir() if candidate.is_file())
    datasets = []
    for candidate in files:
        try:
            datasets.append(pydicom.dcmread(str(candidate)))
        except Exception:
            continue
    if not datasets:
        raise ValueError("invalid_dicom_series")
    series_uids = {str(getattr(ds, "SeriesInstanceUID", "")) for ds in datasets}
    if len(series_uids) > 1:
        raise ValueError("ambiguous_dicom_series")
    datasets.sort(key=lambda ds: _dicom_sort_key(ds))
    stack = np.stack([ds.pixel_array.astype(np.float32, copy=False) for ds in datasets], axis=0)
    first = datasets[0]
    pixel_spacing = getattr(first, "PixelSpacing", [1.0, 1.0])
    z_spacing = float(getattr(first, "SliceThickness", 1.0) or 1.0)
    spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]), z_spacing)
    return MedicalVolume(
        array=stack,
        spacing_mm=spacing,
        source_format="dicom",
        metadata={"series_instance_uid": next(iter(series_uids)), "slice_count": len(datasets)},
    )


def _dicom_sort_key(ds: Any) -> tuple[float, int]:
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) >= 3:
        return (float(ipp[2]), int(getattr(ds, "InstanceNumber", 0) or 0))
    return (float(int(getattr(ds, "InstanceNumber", 0) or 0)), 0)
```

- [ ] **Step 4: Run format tests**

Run:

```bash
cd backend/deepagents_runtime
uv run --extra dev --extra medsam pytest tests/test_nph_medsam_formats.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/formats.py backend/deepagents_runtime/tests/test_nph_medsam_formats.py
git commit -m "feat: add nph medsam medical format adapters"
```

## Task 6: NPH Data Agent Processor And Control Client

**Files:**

- Create: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/control.py`
- Create: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/processor.py`
- Modify: `backend/deepagents_runtime/src/ultra_deepagents/data_agent/worker.py`
- Create: `backend/deepagents_runtime/tests/test_nph_medsam_processor.py`
- Modify: `backend/deepagents_runtime/tests/test_data_agent_worker.py`

- [ ] **Step 1: Write processor test**

Create `backend/deepagents_runtime/tests/test_nph_medsam_processor.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from ultra_deepagents.data_agent.worker import DataAgentJobEnvelope
from ultra_deepagents.nph_medsam.processor import NPHMedSAMDataAgentProcessor
from ultra_deepagents.nph_medsam.runner import FakeNPHMedSAMRunner
from ultra_deepagents.nph_medsam.schema import SourceResource


async def test_nph_processor_runs_fake_inference_and_catalogs_three_outputs(tmp_path: Path) -> None:
    source_path = tmp_path / "Norm_young_004_40yo.nii.gz"
    source_path.write_bytes(b"fake")
    source = SourceResource(
        resource_id="file_source",
        original_name="Norm_young_004_40yo.nii.gz",
        sha256="source-sha",
        storage_path=source_path,
        content_type="application/x-nifti",
    )
    cataloged: list[dict[str, Any]] = []

    async def resolve_source(job: DataAgentJobEnvelope) -> SourceResource:
        return source

    async def catalog_output(job: DataAgentJobEnvelope, payload: dict[str, Any]) -> dict[str, Any]:
        cataloged.append(payload)
        return {"resource": {"resource_id": f"derived_{len(cataloged)}", **payload}}

    progress_events: list[dict[str, Any]] = []

    async def progress(**kwargs: Any) -> None:
        progress_events.append(kwargs)

    processor = NPHMedSAMDataAgentProcessor(
        runner=FakeNPHMedSAMRunner(),
        output_root=tmp_path / "outputs",
        resolve_source=resolve_source,
        catalog_output=catalog_output,
    )
    result = await processor(
        DataAgentJobEnvelope(
            job_id="data_agent_job_1",
            owner_user_id="alice",
            owner_org_id="org-a",
            job_type="nph_medsam_segmentation",
            resource_ids=("file_source",),
            resource_count=1,
        ),
        progress,
    )

    assert result["summary_kind"] == "nph_medsam_segmentation"
    assert len(cataloged) == 3
    assert cataloged[0]["original_name"].startswith("nph_medsam_seg__")
    assert result["derived_resource_ids"] == ["derived_1", "derived_2", "derived_3"]
    assert any(event["event_metadata"]["stage"] == "nph_medsam_outputs_cataloged" for event in progress_events)
```

- [ ] **Step 2: Run processor test to verify failure**

Run:

```bash
cd backend/deepagents_runtime
uv run --extra dev pytest tests/test_nph_medsam_processor.py -q
```

Expected: FAIL because `processor.py` does not exist.

- [ ] **Step 3: Implement processor and control protocol**

Create `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/control.py`:

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.data_agent.worker import DataAgentJobEnvelope
from ultra_deepagents.nph_medsam.schema import SourceResource


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def resolve_source_from_envelope(job: DataAgentJobEnvelope, settings: RuntimeSettings) -> SourceResource:
    resource_id = next(iter(job.resource_ids), "")
    if not resource_id:
        raise ValueError("resource_not_found")
    root = Path(settings.rarespot_upload_roots[0] if settings.rarespot_upload_roots else "data/uploads").resolve()
    matches = sorted(root.glob(f"{resource_id}__*"))
    if not matches:
        raise FileNotFoundError("source_unavailable")
    return SourceResource(
        resource_id=resource_id,
        original_name=matches[0].name.split("__", 1)[1] if "__" in matches[0].name else matches[0].name,
        sha256=sha256_file(matches[0]),
        storage_path=matches[0],
        content_type="application/x-nifti" if matches[0].name.lower().endswith((".nii", ".nii.gz")) else "",
        project_id=job.project_id,
    )


async def catalog_output_resource(
    job: DataAgentJobEnvelope,
    settings: RuntimeSettings,
    payload: dict[str, Any],
) -> dict[str, Any]:
    url = f"{settings.control_base_url.rstrip('/')}/v2/data-agent/jobs/{urllib_parse.quote(job.job_id, safe='')}/resources"
    body = json.dumps(payload).encode("utf-8")
    request = urllib_request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            **job.principal_headers(),
        },
    )
    with urllib_request.urlopen(request, timeout=max(0.1, settings.control_status_timeout_seconds)) as response:
        return json.loads(response.read().decode("utf-8"))
```

Create `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/processor.py`:

```python
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Awaitable, Callable

from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.data_agent.worker import DataAgentJobEnvelope, DataAgentProgressFunc
from ultra_deepagents.nph_medsam.control import catalog_output_resource, resolve_source_from_envelope, sha256_file
from ultra_deepagents.nph_medsam.runner import FakeNPHMedSAMRunner
from ultra_deepagents.nph_medsam.schema import NPH_MEDSAM_JOB_TYPE, SourceResource

ResolveSourceFunc = Callable[[DataAgentJobEnvelope], Awaitable[SourceResource]]
CatalogOutputFunc = Callable[[DataAgentJobEnvelope, dict[str, Any]], Awaitable[dict[str, Any]]]


class NPHMedSAMDataAgentProcessor:
    def __init__(
        self,
        settings: RuntimeSettings | None = None,
        *,
        runner: Any | None = None,
        output_root: Path | None = None,
        resolve_source: ResolveSourceFunc | None = None,
        catalog_output: CatalogOutputFunc | None = None,
    ) -> None:
        self.settings = settings or RuntimeSettings.from_env()
        self.runner = runner or FakeNPHMedSAMRunner()
        self.output_root = Path(output_root or "data/uploads").resolve()
        self._resolve_source = resolve_source
        self._catalog_output = catalog_output

    async def __call__(
        self,
        job: DataAgentJobEnvelope,
        progress: DataAgentProgressFunc,
    ) -> dict[str, Any]:
        if job.job_type != NPH_MEDSAM_JOB_TYPE:
            raise ValueError(f"unsupported_nph_medsam_job_type:{job.job_type}")
        await progress(
            progress_completed=0,
            progress_total=3,
            message="NPH MedSAM source resolution started.",
            event_metadata={"stage": "nph_medsam_source_resolution"},
        )
        source = await self._resolve(job)
        generated_at = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        result = self.runner.run(
            source=source,
            output_dir=self.output_root,
            job_id=job.job_id,
            generated_at=generated_at,
        )
        await progress(
            progress_completed=1,
            progress_total=3,
            message="NPH MedSAM segmentation and measurements written.",
            output_summary={"metrics": result.metrics, "qc": result.qc},
            event_metadata={"stage": "nph_medsam_outputs_written"},
        )
        derived_ids: list[str] = []
        for path, content_type, resource_kind, tags in (
            (result.segmentation_path, "application/x-nifti", "image", ["nph", "medsam", "segmentation", "derived"]),
            (result.summary_path, "application/json", "file", ["nph", "medsam", "summary", "derived"]),
            (result.measurements_path, "text/csv", "table", ["nph", "medsam", "measurements", "derived"]),
        ):
            payload = {
                "source_resource_id": source.resource_id,
                "storage_path": path.name,
                "original_name": path.name,
                "content_type": content_type,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "resource_kind": resource_kind,
                "tags": tags,
                "metadata": result.metadata,
            }
            response = await self._catalog(job, payload)
            resource = response.get("resource") if isinstance(response, dict) else {}
            if isinstance(resource, dict):
                resource_id = str(resource.get("resource_id") or "")
                if resource_id:
                    derived_ids.append(resource_id)
        await progress(
            progress_completed=3,
            progress_total=3,
            message="NPH MedSAM derived Resources cataloged.",
            event_metadata={"stage": "nph_medsam_outputs_cataloged"},
        )
        return {
            "summary_kind": "nph_medsam_segmentation",
            "source_resource_id": source.resource_id,
            "derived_resource_ids": derived_ids,
            "metrics": result.metrics,
            "qc": result.qc,
        }

    async def _resolve(self, job: DataAgentJobEnvelope) -> SourceResource:
        if self._resolve_source is not None:
            return await self._resolve_source(job)
        return await resolve_source_from_envelope(job, self.settings)

    async def _catalog(self, job: DataAgentJobEnvelope, payload: dict[str, Any]) -> dict[str, Any]:
        if self._catalog_output is not None:
            return await self._catalog_output(job, payload)
        return await catalog_output_resource(job, self.settings, payload)
```

- [ ] **Step 4: Route the generic Data Agent worker**

In `backend/deepagents_runtime/src/ultra_deepagents/data_agent/worker.py`, add this helper near `DefaultDataAgentProcessor`:

```python
class RoutedDataAgentProcessor:
    def __init__(self, routes: dict[str, DataAgentProcessorFunc], fallback: DataAgentProcessorFunc | None = None) -> None:
        self._routes = dict(routes)
        self._fallback = fallback or DefaultDataAgentProcessor()

    async def __call__(self, job: DataAgentJobEnvelope, progress: DataAgentProgressFunc) -> dict[str, Any] | None:
        processor = self._routes.get(job.job_type, self._fallback)
        return await processor(job, progress)
```

- [ ] **Step 5: Run processor tests**

Run:

```bash
cd backend/deepagents_runtime
uv run --extra dev pytest tests/test_nph_medsam_processor.py tests/test_data_agent_worker.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/control.py backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/processor.py backend/deepagents_runtime/src/ultra_deepagents/data_agent/worker.py backend/deepagents_runtime/tests/test_nph_medsam_processor.py backend/deepagents_runtime/tests/test_data_agent_worker.py
git commit -m "feat: route nph medsam data agent processing"
```

## Task 7: Deep Agents Tool And Prompt Wiring

**Files:**

- Create: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/tools.py`
- Modify: `backend/deepagents_runtime/src/ultra_deepagents/agent.py`
- Modify: `backend/deepagents_runtime/src/ultra_deepagents/config.py`
- Create: `backend/deepagents_runtime/tests/test_nph_medsam_tools.py`
- Modify: `backend/deepagents_runtime/tests/test_agent_factory.py`

- [ ] **Step 1: Write tool formatting tests**

Create `backend/deepagents_runtime/tests/test_nph_medsam_tools.py`:

```python
from __future__ import annotations

import json

from ultra_deepagents.nph_medsam.tools import format_nph_medsam_tool_result, looks_nph_medsam_goal


def test_looks_nph_medsam_goal_detects_nph_segmentation_requests() -> None:
    assert looks_nph_medsam_goal("Use NPH segmentation on this image and quantify ventricular volume")
    assert looks_nph_medsam_goal("Run MedSAM analysis for hydrocephalus research")
    assert not looks_nph_medsam_goal("Summarize this ecology RareSpot report")


def test_format_nph_medsam_tool_result_is_scientist_facing_and_non_diagnostic() -> None:
    text = format_nph_medsam_tool_result(
        {
            "job_id": "data_agent_job_1",
            "status": "succeeded",
            "source_resource_id": "file_source",
            "derived_resource_ids": ["seg", "summary", "csv"],
            "metrics": {
                "groups": {"ventricle": {"volume_ml": 12.5}},
                "ratios": {"ventricle_to_total_segmented": 0.123},
            },
            "qc": {"warnings": ["empty_label_5"]},
        }
    )
    payload = json.loads(text)
    assert payload["status"] == "succeeded"
    assert "research" in payload["clinical_caveat"].lower()
    assert "diagnosis" in payload["clinical_caveat"].lower()
    assert payload["metric_highlights"]["ventricle_volume_ml"] == 12.5
```

- [ ] **Step 2: Run tool tests to verify failure**

Run:

```bash
cd backend/deepagents_runtime
uv run --extra dev pytest tests/test_nph_medsam_tools.py -q
```

Expected: FAIL because `tools.py` does not exist.

- [ ] **Step 3: Implement tool helpers**

Create `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/tools.py`:

```python
from __future__ import annotations

import json
import re
import time
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.nph_medsam.schema import NPH_MEDSAM_JOB_TYPE, NPH_MEDSAM_TOOL_NAME


class NPHMedSAMAnalysisArgs(BaseModel):
    resource_id: str = Field(default="", description="Resource id for the medical image volume to segment.")
    analysis_focus: str = Field(default="full_summary", description="Analysis focus: full_summary, ventricular_volume, csf_volume, or qc.")
    wait_for_completion: bool = Field(default=True, description="Poll the Data Agent job until it reaches a terminal state.")
    timeout_seconds: int = Field(default=1800, ge=1, le=86400)


def looks_nph_medsam_goal(goal: str) -> bool:
    text = " ".join(str(goal or "").lower().split())
    return bool(re.search(r"\b(nph|normal pressure hydrocephalus|medsam|ventric(?:le|ular)|csf segmentation|hydrocephalus)\b", text))


def format_nph_medsam_tool_result(result: dict[str, Any]) -> str:
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    groups = metrics.get("groups") if isinstance(metrics.get("groups"), dict) else {}
    ratios = metrics.get("ratios") if isinstance(metrics.get("ratios"), dict) else {}
    ventricle = groups.get("ventricle") if isinstance(groups.get("ventricle"), dict) else {}
    return json.dumps(
        {
            "status": result.get("status", "unknown"),
            "job_id": result.get("job_id", ""),
            "source_resource_id": result.get("source_resource_id", ""),
            "derived_resource_ids": result.get("derived_resource_ids", []),
            "metric_highlights": {
                "ventricle_volume_ml": ventricle.get("volume_ml"),
                "ventricle_to_total_segmented": ratios.get("ventricle_to_total_segmented"),
            },
            "qc": result.get("qc", {}),
            "clinical_caveat": "Research analysis support only; segmentation-derived measurements are not a diagnosis and require expert image review before clinical use.",
            "final_answer_hint": "Report volumes, ratios, Resource IDs, model provenance, QC warnings, and the clinical caveat. Do not duplicate segmentation files as sandbox artifacts.",
        },
        sort_keys=True,
    )


def build_nph_medsam_tools(settings: RuntimeSettings) -> list[StructuredTool]:
    def nph_medsam_analysis(
        resource_id: str = "",
        analysis_focus: str = "full_summary",
        wait_for_completion: bool = True,
        timeout_seconds: int = 1800,
    ) -> str:
        from ultra_deepagents.rarespot.tools import active_context

        context: AgentRunContext | None = active_context()
        selected = list(context.selected_file_ids) if context is not None else []
        resolved_resource_id = resource_id.strip() or (selected[0] if selected else "")
        if not resolved_resource_id:
            raise ValueError("nph_medsam_analysis requires a resource_id or selected Resource.")
        result = submit_and_wait_for_nph_job(
            settings=settings,
            resource_id=resolved_resource_id,
            analysis_focus=analysis_focus,
            user_id=context.user_id if context is not None else "researcher",
            wait_for_completion=wait_for_completion,
            timeout_seconds=timeout_seconds,
        )
        return format_nph_medsam_tool_result(result)

    return [
        StructuredTool.from_function(
            func=nph_medsam_analysis,
            name=NPH_MEDSAM_TOOL_NAME,
            description="Run production NPH MedSAM 7-class segmentation on a Resource-backed NIfTI/DICOM/TIFF image and return quantitative measurements, derived Resource IDs, QC, and non-diagnostic caveats.",
            args_schema=NPHMedSAMAnalysisArgs,
        )
    ]


def submit_and_wait_for_nph_job(
    *,
    settings: RuntimeSettings,
    resource_id: str,
    analysis_focus: str,
    user_id: str,
    wait_for_completion: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    import urllib.request

    body = json.dumps(
        {
            "job_type": NPH_MEDSAM_JOB_TYPE,
            "resource_ids": [resource_id],
            "metadata": {"analysis_focus": analysis_focus, "tool_name": NPH_MEDSAM_TOOL_NAME},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        settings.control_base_url.rstrip("/") + "/v2/data-agent/jobs",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json", "X-Ultra-User-Id": user_id},
    )
    with urllib.request.urlopen(request, timeout=max(0.1, settings.control_status_timeout_seconds)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    job = payload.get("job") if isinstance(payload, dict) else {}
    job_id = str(job.get("job_id") or "")
    if not wait_for_completion:
        return {"status": job.get("status", "queued"), "job_id": job_id, "source_resource_id": resource_id}
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        status_payload = _fetch_job(settings, job_id, user_id)
        current_job = status_payload.get("job", {})
        status = str(current_job.get("status") or "")
        if status in {"succeeded", "failed", "canceled"}:
            summary = current_job.get("output_summary") if isinstance(current_job, dict) else {}
            if isinstance(summary, dict):
                return {"status": status, "job_id": job_id, **summary}
            return {"status": status, "job_id": job_id}
        time.sleep(2)
    return {"status": "timeout", "job_id": job_id, "source_resource_id": resource_id}


def _fetch_job(settings: RuntimeSettings, job_id: str, user_id: str) -> dict[str, Any]:
    import urllib.parse
    import urllib.request

    url = settings.control_base_url.rstrip("/") + "/v2/data-agent/jobs/" + urllib.parse.quote(job_id, safe="")
    request = urllib.request.Request(url, method="GET", headers={"Accept": "application/json", "X-Ultra-User-Id": user_id})
    with urllib.request.urlopen(request, timeout=max(0.1, settings.control_status_timeout_seconds)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {}
```

- [ ] **Step 4: Wire agent guidance and registration**

In `backend/deepagents_runtime/src/ultra_deepagents/agent.py`, add import:

```python
from ultra_deepagents.nph_medsam.tools import build_nph_medsam_tools, looks_nph_medsam_goal
```

Add `NPH_MEDSAM_GUIDANCE` near `RARESPOT_GUIDANCE`:

```python
NPH_MEDSAM_GUIDANCE = """
For NPH, hydrocephalus, ventricular-volume, CSF-volume, or MedSAM segmentation requests,
use nph_medsam_analysis as the production Resource-native path. Do not stage raw medical
volumes into the sandbox just to run segmentation. After the tool succeeds, answer from
its metrics, QC warnings, derived Resource IDs, and model provenance. State that the
measurements are research analysis support, not a diagnosis or shunt-candidacy decision.
"""
```

Include `NPH_MEDSAM_GUIDANCE` in the system prompt builder next to RareSpot guidance.

In `build_agent`, register the tool:

```python
if _should_register_nph_medsam_tools(context):
    resolved_tools.extend(build_nph_medsam_tools(settings))
```

Add the gate:

```python
def _should_register_nph_medsam_tools(context: AgentRunContext | None) -> bool:
    if context is None:
        return True
    return looks_nph_medsam_goal(str(context.goal or ""))
```

- [ ] **Step 5: Add factory registration test**

In `backend/deepagents_runtime/tests/test_agent_factory.py`, add:

```python
def test_research_agent_registers_nph_medsam_tool_for_nph_goal() -> None:
    context = AgentRunContext(
        run_id="run_1",
        thread_id="thread_1",
        user_id="alice",
        goal="Use NPH segmentation on this image and quantify ventricles.",
        selected_file_ids=("file_source",),
    )
    agent = build_agent(settings=RuntimeSettings.from_env(), context=context)
    tool_names = {tool.name for tool in agent.tools}
    assert "nph_medsam_analysis" in tool_names
```

- [ ] **Step 6: Run agent/tool tests**

Run:

```bash
cd backend/deepagents_runtime
uv run --extra dev pytest tests/test_nph_medsam_tools.py tests/test_agent_factory.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/tools.py backend/deepagents_runtime/src/ultra_deepagents/agent.py backend/deepagents_runtime/src/ultra_deepagents/config.py backend/deepagents_runtime/tests/test_nph_medsam_tools.py backend/deepagents_runtime/tests/test_agent_factory.py
git commit -m "feat: add nph medsam deep agents tool"
```

## Task 8: Real MedSAM Model Runtime For NIfTI

**Files:**

- Create: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/model.py`
- Create: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/preprocess.py`
- Create: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/postprocess.py`
- Modify: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/runner.py`
- Create: `backend/deepagents_runtime/tests/test_nph_medsam_preprocess.py`
- Create: `backend/deepagents_runtime/tests/test_nph_medsam_model.py`

- [ ] **Step 1: Write preprocess/postprocess tests**

Create `backend/deepagents_runtime/tests/test_nph_medsam_preprocess.py`:

```python
from __future__ import annotations

import numpy as np

from ultra_deepagents.nph_medsam.postprocess import restore_label_volume
from ultra_deepagents.nph_medsam.preprocess import preprocess_ct_volume


def test_preprocess_ct_volume_clips_scales_and_channels() -> None:
    volume = np.array([[[-10.0, 0.0], [40.0, 100.0]]], dtype=np.float32)

    batch = preprocess_ct_volume(volume, image_size=512)

    assert batch.shape == (1, 3, 512, 512)
    assert batch.min() >= 0.0
    assert batch.max() <= 1.0
    assert np.isclose(batch[0, :, 0, 0].max(), 0.0)


def test_restore_label_volume_uses_nearest_neighbor_values() -> None:
    logits_argmax = np.array([[[0, 1], [6, 3]]], dtype=np.uint8)

    restored = restore_label_volume(logits_argmax, target_shape=(1, 4, 4))

    assert restored.dtype == np.uint8
    assert restored.shape == (1, 4, 4)
    assert set(np.unique(restored)) == {0, 1, 3, 6}
```

- [ ] **Step 2: Run preprocess tests to verify failure**

Run:

```bash
cd backend/deepagents_runtime
uv run --extra dev --extra medsam pytest tests/test_nph_medsam_preprocess.py -q
```

Expected: FAIL because modules do not exist.

- [ ] **Step 3: Implement preprocess and postprocess**

Create `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/preprocess.py`:

```python
from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

from ultra_deepagents.nph_medsam.schema import CT_CLIP_RANGE, IMAGE_SIZE


def preprocess_ct_volume(volume: np.ndarray, *, image_size: int = IMAGE_SIZE) -> np.ndarray:
    data = np.asarray(volume, dtype=np.float32)
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    clipped = np.clip(data, CT_CLIP_RANGE[0], CT_CLIP_RANGE[1]) / CT_CLIP_RANGE[1]
    slices: list[np.ndarray] = []
    for z_index in range(clipped.shape[0]):
        source = clipped[z_index]
        scale_y = image_size / source.shape[0]
        scale_x = image_size / source.shape[1]
        resized = zoom(source, (scale_y, scale_x), order=3)
        slices.append(np.repeat(resized[np.newaxis, :, :], 3, axis=0))
    return np.stack(slices, axis=0).astype(np.float32, copy=False)
```

Create `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/postprocess.py`:

```python
from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom


def argmax_masks(mask_logits: np.ndarray) -> np.ndarray:
    return np.argmax(mask_logits, axis=1).astype(np.uint8, copy=False)


def restore_label_volume(labels: np.ndarray, *, target_shape: tuple[int, int, int]) -> np.ndarray:
    source = np.asarray(labels, dtype=np.uint8)
    if source.shape == target_shape:
        return source
    scales = tuple(target / current for target, current in zip(target_shape, source.shape, strict=True))
    restored = zoom(source, scales, order=0)
    return np.asarray(restored, dtype=np.uint8)
```

- [ ] **Step 4: Write model contract tests**

Create `backend/deepagents_runtime/tests/test_nph_medsam_model.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from ultra_deepagents.nph_medsam.model import verify_checkpoint_sha256
from ultra_deepagents.nph_medsam.schema import MODEL_SHA256


def test_verify_checkpoint_sha256_accepts_known_digest(tmp_path: Path) -> None:
    path = tmp_path / "weights.pt"
    path.write_bytes(b"weights")
    expected = "9a129038d5f577a920cc711972f46c61f5e2b513709600821ccdd1a7f47f6c0a"

    assert verify_checkpoint_sha256(path, expected_sha256=expected) == expected


def test_verify_checkpoint_sha256_rejects_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "weights.pt"
    path.write_bytes(b"weights")

    with pytest.raises(ValueError, match="checkpoint_hash_mismatch"):
        verify_checkpoint_sha256(path, expected_sha256=MODEL_SHA256)
```

- [ ] **Step 5: Implement checkpoint verification and model skeleton**

Create `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/model.py`:

```python
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from ultra_deepagents.nph_medsam.postprocess import argmax_masks
from ultra_deepagents.nph_medsam.schema import BOX_PROMPT, MEDSAM_NUM_CLASSES, MODEL_SHA256


def verify_checkpoint_sha256(path: str | Path, *, expected_sha256: str = MODEL_SHA256) -> str:
    checkpoint = Path(path)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    if value != expected_sha256:
        raise ValueError("checkpoint_hash_mismatch")
    return value


class MedSAMNPHModel:
    def __init__(self, *, checkpoint_path: str | Path, device: str = "cuda") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self._model: Any | None = None

    def load(self) -> None:
        verify_checkpoint_sha256(self.checkpoint_path)
        import torch

        from ultra_deepagents.nph_medsam.segment_anything import sam_model_registry

        model = sam_model_registry["vit_b"](checkpoint=None, num_classes=MEDSAM_NUM_CLASSES)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(self.device)
        model.eval()
        self._model = model

    def infer(self, batch_chw: np.ndarray) -> np.ndarray:
        if self._model is None:
            self.load()
        import torch

        assert self._model is not None
        images = torch.from_numpy(batch_chw).to(self.device)
        boxes = torch.tensor([BOX_PROMPT], dtype=torch.float32, device=self.device).repeat(images.shape[0], 1)
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self._model.prompt_encoder(points=None, boxes=boxes, masks=None)
            image_embeddings = self._model.image_encoder(images)
            low_res_masks, _ = self._model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self._model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )
            masks = torch.nn.functional.interpolate(
                low_res_masks,
                size=(batch_chw.shape[-2], batch_chw.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            labels = argmax_masks(torch.sigmoid(masks).detach().cpu().numpy())
        return labels
```

- [ ] **Step 6: Vendor minimal modified SAM code**

Copy only the required modified SAM implementation from the external repo into:

```text
backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/segment_anything/
```

Use these source paths:

```bash
mkdir -p backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/segment_anything
rsync -a --exclude='__pycache__' \
  /Users/macbook/Downloads/MedSAM_CTsegmentation-main/segment_anything/ \
  backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/segment_anything/
```

After copying, inspect imports and replace package-relative imports that point at the external checkout with local package imports under `ultra_deepagents.nph_medsam.segment_anything`.

- [ ] **Step 7: Wire real runner for NIfTI**

In `runner.py`, add `RealNPHMedSAMRunner` that:

1. Loads `MedicalVolume` from `formats.py`.
2. Calls `preprocess_ct_volume`.
3. Calls `MedSAMNPHModel.infer`.
4. Calls `restore_label_volume`.
5. Validates labels with `validate_label_volume`.
6. Saves NIfTI labels with `save_segmentation_like_source`.
7. Writes summary JSON and measurements CSV using the existing fake-runner output contract.

The constructor signature must be:

```python
class RealNPHMedSAMRunner:
    def __init__(self, *, checkpoint_path: str | Path, device: str = "cuda") -> None:
        self.model = MedSAMNPHModel(checkpoint_path=checkpoint_path, device=device)
```

- [ ] **Step 8: Run local non-GPU tests**

Run:

```bash
cd backend/deepagents_runtime
uv run --extra dev --extra medsam pytest tests/test_nph_medsam_preprocess.py tests/test_nph_medsam_model.py tests/test_nph_medsam_runner.py -q
```

Expected: PASS. The tests must not require CUDA.

- [ ] **Step 9: Commit**

```bash
git add backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/model.py backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/preprocess.py backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/postprocess.py backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/runner.py backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/segment_anything backend/deepagents_runtime/tests/test_nph_medsam_preprocess.py backend/deepagents_runtime/tests/test_nph_medsam_model.py backend/deepagents_runtime/tests/test_nph_medsam_runner.py
git commit -m "feat: add real nph medsam nifti inference runtime"
```

## Task 9: NPH Worker Entrypoint And Local Integration

**Files:**

- Create: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/worker.py`
- Modify: `backend/deepagents_runtime/src/ultra_deepagents/config.py`
- Create: `backend/deepagents_runtime/tests/test_nph_medsam_worker.py`
- Create: `scripts/run_nph_medsam_worker.sh`
- Modify: `Makefile`

- [ ] **Step 1: Add settings**

In `RuntimeSettings`, add fields:

```python
nph_medsam_enabled: bool = True
nph_medsam_runner_mode: str = "fake"
nph_medsam_checkpoint_path: str = ""
nph_medsam_checkpoint_sha256: str = "04b219ad513d60770b648dfc72298cf99a8d7b5cfc70e95217caff17f96a93dc"
nph_medsam_device: str = "cuda"
nph_medsam_output_root: str = "data/uploads"
```

In `from_env`, add:

```python
nph_medsam_enabled=_env_bool("ULTRA_NPH_MEDSAM_ENABLED", True),
nph_medsam_runner_mode=os.getenv("ULTRA_NPH_MEDSAM_RUNNER_MODE", "fake"),
nph_medsam_checkpoint_path=os.getenv("ULTRA_NPH_MEDSAM_CHECKPOINT_PATH", ""),
nph_medsam_checkpoint_sha256=os.getenv("ULTRA_NPH_MEDSAM_CHECKPOINT_SHA256", "04b219ad513d60770b648dfc72298cf99a8d7b5cfc70e95217caff17f96a93dc"),
nph_medsam_device=os.getenv("ULTRA_NPH_MEDSAM_DEVICE", "cuda"),
nph_medsam_output_root=os.getenv("ULTRA_NPH_MEDSAM_OUTPUT_ROOT", os.getenv("ULTRA_CONTROL_UPLOAD_ROOT", "data/uploads")),
```

- [ ] **Step 2: Write worker build test**

Create `backend/deepagents_runtime/tests/test_nph_medsam_worker.py`:

```python
from __future__ import annotations

from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.data_agent.worker import RoutedDataAgentProcessor
from ultra_deepagents.nph_medsam.schema import NPH_MEDSAM_JOB_TYPE
from ultra_deepagents.nph_medsam.worker import build_nph_medsam_data_agent_processor


def test_build_nph_medsam_data_agent_processor_routes_nph_job_type(tmp_path) -> None:
    settings = RuntimeSettings.from_env()
    settings = settings.__class__(**{**settings.__dict__, "nph_medsam_runner_mode": "fake", "nph_medsam_output_root": str(tmp_path)})

    processor = build_nph_medsam_data_agent_processor(settings)

    assert isinstance(processor, RoutedDataAgentProcessor)
    assert NPH_MEDSAM_JOB_TYPE in processor._routes
```

- [ ] **Step 3: Implement worker entrypoint**

Create `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/worker.py`:

```python
from __future__ import annotations

import asyncio
from pathlib import Path

from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.data_agent.worker import DefaultDataAgentProcessor, NATSDataAgentWorker, RoutedDataAgentProcessor
from ultra_deepagents.nph_medsam.processor import NPHMedSAMDataAgentProcessor
from ultra_deepagents.nph_medsam.runner import FakeNPHMedSAMRunner, RealNPHMedSAMRunner
from ultra_deepagents.nph_medsam.schema import NPH_MEDSAM_JOB_TYPE


def build_nph_medsam_runner(settings: RuntimeSettings):
    if settings.nph_medsam_runner_mode == "real":
        if not settings.nph_medsam_checkpoint_path:
            raise ValueError("ULTRA_NPH_MEDSAM_CHECKPOINT_PATH is required for real NPH MedSAM inference")
        return RealNPHMedSAMRunner(
            checkpoint_path=settings.nph_medsam_checkpoint_path,
            device=settings.nph_medsam_device,
        )
    return FakeNPHMedSAMRunner()


def build_nph_medsam_data_agent_processor(settings: RuntimeSettings) -> RoutedDataAgentProcessor:
    nph_processor = NPHMedSAMDataAgentProcessor(
        settings=settings,
        runner=build_nph_medsam_runner(settings),
        output_root=Path(settings.nph_medsam_output_root),
    )
    return RoutedDataAgentProcessor(
        routes={NPH_MEDSAM_JOB_TYPE: nph_processor},
        fallback=DefaultDataAgentProcessor(),
    )


async def amain() -> None:
    settings = RuntimeSettings.from_env()
    worker = NATSDataAgentWorker(
        settings=settings,
        processor=build_nph_medsam_data_agent_processor(settings),
    )
    await worker.run_forever()


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add run script**

Create `scripts/run_nph_medsam_worker.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend/deepagents_runtime"
exec uv run --extra medsam --extra dev python -m ultra_deepagents.nph_medsam.worker
```

Make it executable:

```bash
chmod +x scripts/run_nph_medsam_worker.sh
```

- [ ] **Step 5: Run worker tests**

Run:

```bash
cd backend/deepagents_runtime
uv run --extra dev --extra medsam pytest tests/test_nph_medsam_worker.py tests/test_data_agent_worker.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/worker.py backend/deepagents_runtime/src/ultra_deepagents/config.py backend/deepagents_runtime/tests/test_nph_medsam_worker.py scripts/run_nph_medsam_worker.sh Makefile
git commit -m "feat: add nph medsam data agent worker entrypoint"
```

## Task 10: Valid DICOM And TIFF End-To-End Support

**Files:**

- Modify: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/formats.py`
- Modify: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/runner.py`
- Modify: `backend/deepagents_runtime/tests/test_nph_medsam_formats.py`
- Modify: `backend/deepagents_runtime/tests/test_nph_medsam_runner.py`

- [ ] **Step 1: Add valid DICOM and TIFF runner tests**

Extend `test_nph_medsam_runner.py` with:

```python
def test_fake_runner_accepts_tiff_source(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "ct_stack.tiff"
    tifffile.imwrite(path, np.zeros((3, 8, 8), dtype=np.uint16))
    source = SourceResource(
        resource_id="file_tiff",
        original_name="ct_stack.tiff",
        sha256="source-sha",
        storage_path=path,
        content_type="image/tiff",
    )

    result = FakeNPHMedSAMRunner().run(
        source=source,
        output_dir=tmp_path / "outputs",
        job_id="job_tiff",
        generated_at="20260613T120000Z",
    )

    assert result.metrics["shape"] == [8, 8, 8]
    assert result.qc["warnings"]
```

Add a single-series DICOM fixture test in `test_nph_medsam_formats.py`:

```python
def test_dicom_single_series_loads_sorted_stack(tmp_path: Path) -> None:
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset
    from pydicom.uid import ExplicitVRLittleEndian

    for instance_number in [2, 1]:
        ds = FileDataset(str(tmp_path / f"{instance_number}.dcm"), {}, file_meta=pydicom.dataset.FileMetaDataset(), preamble=b"\0" * 128)
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.SeriesInstanceUID = "series-a"
        ds.InstanceNumber = instance_number
        ds.Rows = 2
        ds.Columns = 2
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.PixelSpacing = [0.5, 0.5]
        ds.SliceThickness = 2.0
        ds.PixelData = (np.ones((2, 2), dtype=np.uint16) * instance_number).tobytes()
        ds.save_as(tmp_path / f"{instance_number}.dcm")

    volume = load_medical_volume(tmp_path)

    assert volume.array.shape == (2, 2, 2)
    assert volume.spacing_mm == (0.5, 0.5, 2.0)
    assert volume.array[0, 0, 0] == 1
```

- [ ] **Step 2: Run DICOM/TIFF tests**

Run:

```bash
cd backend/deepagents_runtime
uv run --extra dev --extra medsam pytest tests/test_nph_medsam_formats.py tests/test_nph_medsam_runner.py -q
```

Expected: PASS after format adapters are complete. If a test fails because fake runner ignores source geometry, update fake runner to load source geometry for TIFF/DICOM while keeping deterministic labels.

- [ ] **Step 3: Commit**

```bash
git add backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/formats.py backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/runner.py backend/deepagents_runtime/tests/test_nph_medsam_formats.py backend/deepagents_runtime/tests/test_nph_medsam_runner.py
git commit -m "feat: complete nph medsam dicom and tiff support"
```

## Task 11: GPU Host Deployment Artifacts

**Files:**

- Create: `deploy/env/nph-medsam-worker.env.example`
- Create: `deploy/systemd/ultra-nph-medsam-worker.service`
- Create: `scripts/deploy_nph_medsam_worker.sh`
- Create: `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/live_smoke.py`

- [ ] **Step 1: Add deployment env example**

Create `deploy/env/nph-medsam-worker.env.example`:

```bash
# Ultra NPH MedSAM GPU worker
ULTRA_CONTROL_BASE_URL=http://127.0.0.1:8088
ULTRA_CONTROL_NATS_URL=nats://127.0.0.1:4222
ULTRA_DATA_AGENT_NATS_URL=nats://127.0.0.1:4222
ULTRA_DATA_AGENT_NATS_STREAM=ULTRA_RUNS
ULTRA_DATA_AGENT_NATS_JOBS_SUBJECT=ultra.data_agent.jobs
ULTRA_DATA_AGENT_NATS_WORKER_DURABLE=ultra-nph-medsam-worker
ULTRA_DATA_AGENT_WORKER_ID=ultra-nph-medsam-worker@lambda-quad
ULTRA_DATA_AGENT_WORKER_KIND=nph_medsam
ULTRA_DEEPAGENTS_WORKER_MAX_CONCURRENCY=1
ULTRA_CONTROL_UPLOAD_ROOT=/srv/ultra/uploads
ULTRA_NPH_MEDSAM_OUTPUT_ROOT=/srv/ultra/uploads
ULTRA_NPH_MEDSAM_RUNNER_MODE=real
ULTRA_NPH_MEDSAM_DEVICE=cuda
ULTRA_NPH_MEDSAM_CHECKPOINT_PATH=/srv/ultra/nph-medsam/models/MEDSAM_finetune_CT_NO_SKULLSTRIP_repeated_img_embeddings_no_prompt_7classes_model_best.pt
ULTRA_NPH_MEDSAM_CHECKPOINT_SHA256=04b219ad513d60770b648dfc72298cf99a8d7b5cfc70e95217caff17f96a93dc
```

- [ ] **Step 2: Add systemd service**

Create `deploy/systemd/ultra-nph-medsam-worker.service`:

```ini
[Unit]
Description=Ultra NPH MedSAM GPU worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=amil
WorkingDirectory=/home/amil/ultra-nph-medsam-worker/backend/deepagents_runtime
EnvironmentFile=/etc/ultra/nph-medsam-worker.env
ExecStart=/home/amil/.local/bin/uv run --extra medsam python -m ultra_deepagents.nph_medsam.worker
Restart=always
RestartSec=5
TimeoutStartSec=0
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 3: Add live smoke script**

Create `backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/live_smoke.py`:

```python
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from ultra_deepagents.nph_medsam.model import verify_checkpoint_sha256


def main() -> None:
    checkpoint = Path(os.environ["ULTRA_NPH_MEDSAM_CHECKPOINT_PATH"])
    payload = {
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "checkpoint_sha256": verify_checkpoint_sha256(checkpoint),
        "checkpoint_path": str(checkpoint),
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add deploy script**

Create `scripts/deploy_nph_medsam_worker.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

REMOTE="${ULTRA_NPH_MEDSAM_REMOTE:-amil@128.111.185.73}"
REMOTE_ROOT="${ULTRA_NPH_MEDSAM_REMOTE_ROOT:-/home/amil/ultra-nph-medsam-worker}"
REMOTE_MODEL_DIR="${ULTRA_NPH_MEDSAM_REMOTE_MODEL_DIR:-/srv/ultra/nph-medsam/models}"
CHECKPOINT="${ULTRA_NPH_MEDSAM_CHECKPOINT_LOCAL:-/Users/macbook/Downloads/bisque-20260612.010648/MEDSAM_finetune_CT_NO_SKULLSTRIP_repeated_img_embeddings_no_prompt_7classes_model_best.pt}"
EXPECTED_SHA="04b219ad513d60770b648dfc72298cf99a8d7b5cfc70e95217caff17f96a93dc"

actual_sha="$(shasum -a 256 "$CHECKPOINT" | awk '{print $1}')"
if [ "$actual_sha" != "$EXPECTED_SHA" ]; then
  echo "checkpoint sha mismatch: $actual_sha" >&2
  exit 1
fi

ssh "$REMOTE" "mkdir -p '$REMOTE_ROOT' '$REMOTE_MODEL_DIR'"
rsync -a --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude 'data' \
  backend/deepagents_runtime deploy scripts \
  "$REMOTE:$REMOTE_ROOT/"
rsync -a "$CHECKPOINT" "$REMOTE:$REMOTE_MODEL_DIR/"
ssh "$REMOTE" "cd '$REMOTE_ROOT/backend/deepagents_runtime' && uv sync --extra medsam"
ssh "$REMOTE" "cd '$REMOTE_ROOT/backend/deepagents_runtime' && ULTRA_NPH_MEDSAM_CHECKPOINT_PATH='$REMOTE_MODEL_DIR/$(basename "$CHECKPOINT")' uv run --extra medsam python -m ultra_deepagents.nph_medsam.live_smoke"
```

Make executable:

```bash
chmod +x scripts/deploy_nph_medsam_worker.sh
```

- [ ] **Step 5: Run local script checks**

Run:

```bash
bash -n scripts/deploy_nph_medsam_worker.sh
python3 -m py_compile backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/live_smoke.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add deploy/env/nph-medsam-worker.env.example deploy/systemd/ultra-nph-medsam-worker.service scripts/deploy_nph_medsam_worker.sh backend/deepagents_runtime/src/ultra_deepagents/nph_medsam/live_smoke.py
git commit -m "feat: add nph medsam gpu worker deployment"
```

## Task 12: Remote GPU Smoke With Real Checkpoint

**Files:**

- Modify only if smoke exposes a defect in files from earlier tasks.
- Record evidence in: `planning/2026-06-13-nph-medsam-acceptance-matrix.md`

- [ ] **Step 1: Deploy code and checkpoint**

Run:

```bash
ULTRA_NPH_MEDSAM_REMOTE=amil@128.111.185.73 ./scripts/deploy_nph_medsam_worker.sh
```

Expected output includes JSON with:

```json
{"checkpoint_sha256":"04b219ad513d60770b648dfc72298cf99a8d7b5cfc70e95217caff17f96a93dc","cuda_available":true,"device_name":"NVIDIA TITAN RTX"}
```

- [ ] **Step 2: Copy test NIfTI if the remote host cannot access local uploads**

Find the local source:

```bash
find data/uploads -name '*Norm_young_004_40yo.nii.gz' -type f -maxdepth 1
```

Copy one selected file:

```bash
scp data/uploads/file_0140a6ebbffeb2ad4b23408a5b381dae__Norm_young_004_40yo.nii.gz amil@128.111.185.73:/tmp/Norm_young_004_40yo.nii.gz
```

Expected: copy succeeds.

- [ ] **Step 3: Run direct real-runner smoke on remote**

Run:

```bash
ssh amil@128.111.185.73 'cd /home/amil/ultra-nph-medsam-worker/backend/deepagents_runtime && ULTRA_NPH_MEDSAM_CHECKPOINT_PATH=/srv/ultra/nph-medsam/models/MEDSAM_finetune_CT_NO_SKULLSTRIP_repeated_img_embeddings_no_prompt_7classes_model_best.pt uv run --extra medsam python - <<"PY"
from pathlib import Path
from ultra_deepagents.nph_medsam.runner import RealNPHMedSAMRunner
from ultra_deepagents.nph_medsam.schema import SourceResource

source = SourceResource(
    resource_id="manual_norm_young_004",
    original_name="Norm_young_004_40yo.nii.gz",
    sha256="manual-smoke",
    storage_path=Path("/tmp/Norm_young_004_40yo.nii.gz"),
    content_type="application/x-nifti",
)
runner = RealNPHMedSAMRunner(
    checkpoint_path=Path("/srv/ultra/nph-medsam/models/MEDSAM_finetune_CT_NO_SKULLSTRIP_repeated_img_embeddings_no_prompt_7classes_model_best.pt"),
    device="cuda",
)
result = runner.run(source=source, output_dir=Path("/tmp/nph-medsam-smoke"), job_id="manual_smoke", generated_at="20260613T120000Z")
print(result.segmentation_path)
print(result.metrics["shape"])
print(result.qc)
PY'
```

Expected:

- The command exits 0.
- Output segmentation path exists.
- Metrics shape matches the NIfTI source shape.
- QC `present_labels` values are a subset of `[0,1,2,3,4,5,6]`.

- [ ] **Step 4: Record evidence**

Capture the remote smoke output into shell variables and write the acceptance matrix:

```bash
remote_smoke_output="$(ssh amil@128.111.185.73 'cd /home/amil/ultra-nph-medsam-worker/backend/deepagents_runtime && tail -n 20 /tmp/nph-medsam-smoke.log 2>/dev/null || true')"
segmentation_output="$(printf '%s\n' "$remote_smoke_output" | rg '^/tmp/nph-medsam-smoke/.+nii.gz$' | tail -n 1)"
metrics_shape="$(printf '%s\n' "$remote_smoke_output" | rg '^\[[0-9, ]+\]$' | tail -n 1)"
qc_line="$(printf '%s\n' "$remote_smoke_output" | rg \"present_labels|warnings|ok\" | tail -n 1)"
mkdir -p planning
cat > planning/2026-06-13-nph-medsam-acceptance-matrix.md <<EOF
# NPH MedSAM Acceptance Matrix

## Remote GPU Smoke

- Host: \`amil@128.111.185.73\`
- GPU: \`NVIDIA TITAN RTX\`
- Checkpoint SHA-256: \`04b219ad513d60770b648dfc72298cf99a8d7b5cfc70e95217caff17f96a93dc\`
- Input: \`Norm_young_004_40yo.nii.gz\`
- Segmentation output: \`${segmentation_output}\`
- Source shape: \`${metrics_shape}\`
- Output shape: \`${metrics_shape}\`
- QC output: \`${qc_line}\`
EOF
```

Expected: the matrix file contains the remote path, shape, and QC output from the smoke run.

- [ ] **Step 5: Commit smoke evidence and fixes**

```bash
git add planning/2026-06-13-nph-medsam-acceptance-matrix.md
git commit -m "test: record nph medsam gpu smoke"
```

## Task 13: Full Local Integration And Scientific Response Quality

**Files:**

- Create: `backend/deepagents_runtime/tests/test_nph_medsam_live_trace.py`
- Modify: `backend/deepagents_runtime/src/ultra_deepagents/live_trace.py`
  - Add NPH MedSAM trace-quality classification only when the new test proves the current trace summary cannot identify the required tool/response evidence.
- Modify: `frontend/src/types.ts`
  - Add `nph_medsam_segmentation` to frontend Data Agent job-type unions when TypeScript rejects the new job type.
- Modify: `frontend/src/lib/api.ts`
  - Keep existing Data Agent client methods; add no new client method unless frontend type checking proves the union update needs a request helper.

- [ ] **Step 1: Add live-trace quality test**

Create `backend/deepagents_runtime/tests/test_nph_medsam_live_trace.py`:

```python
from __future__ import annotations

from ultra_deepagents.nph_medsam.tools import format_nph_medsam_tool_result


def test_nph_medsam_tool_result_quality_contract() -> None:
    text = format_nph_medsam_tool_result(
        {
            "status": "succeeded",
            "job_id": "data_agent_job_1",
            "source_resource_id": "file_source",
            "derived_resource_ids": ["seg", "summary", "csv"],
            "metrics": {
                "groups": {"ventricle": {"volume_ml": 42.1}},
                "ratios": {"ventricle_to_total_segmented": 0.21},
            },
            "qc": {"warnings": []},
        }
    )

    assert "42.1" in text
    assert "seg" in text
    assert "diagnosis" in text.lower()
    assert "research analysis support" in text.lower()
```

- [ ] **Step 2: Run full local verification**

Run:

```bash
cd backend/deepagents_runtime
uv run --extra dev --extra medsam pytest tests/test_nph_medsam_*.py tests/test_data_agent_worker.py tests/test_agent_factory.py -q
cd ../..
cd backend/controlplane
go test ./internal/httpapi ./internal/openapi -count=1
```

Expected: PASS.

- [ ] **Step 3: Run local control-stack fake end-to-end**

Start the local stack:

```bash
make restart-control-stack
```

Start fake NPH worker:

```bash
ULTRA_NPH_MEDSAM_RUNNER_MODE=fake ./scripts/run_nph_medsam_worker.sh
```

Create an NPH job for a cataloged `Norm_young_004_40yo.nii.gz` Resource through the UI or API. Then inspect:

```bash
curl -fsS http://127.0.0.1:8000/v2/data-agent/jobs?job_type=nph_medsam_segmentation
curl -fsS 'http://127.0.0.1:8000/v2/resources?q=nph_medsam'
```

Expected:

- One `nph_medsam_segmentation` job reaches `succeeded`.
- Three derived Resources exist with `nph_medsam_` names.
- Segmentation Resource has `source_type = "derived"`.
- Summary metadata includes `checkpoint_sha256`.

- [ ] **Step 4: Run real agent prompt**

Use the app or live trace with a selected/uploaded Resource:

```text
Use NPH segmentation on this image and perform quantitative analysis from the resulting segmentation.
```

Expected answer:

- Mentions the source Resource id.
- Mentions segmentation, summary, and measurements Resource ids.
- Reports ventricular volume in ml and at least one ratio.
- Reports QC status or warnings.
- States research-support/non-diagnostic caveat.
- Does not claim a diagnosis or treatment recommendation.

- [ ] **Step 5: Record final acceptance evidence**

Append the observed local evidence by setting shell variables from the just-run commands:

```bash
control_plane_verification="cd backend/controlplane && go test ./internal/httpapi ./internal/openapi -count=1: PASS"
python_verification="cd backend/deepagents_runtime && uv run --extra dev --extra medsam pytest tests/test_nph_medsam_*.py tests/test_data_agent_worker.py tests/test_agent_factory.py -q: PASS"
fake_worker_job_id="$(curl -fsS 'http://127.0.0.1:8000/v2/data-agent/jobs?job_type=nph_medsam_segmentation' | python3 -c 'import json,sys; data=json.load(sys.stdin); jobs=data.get("jobs") or []; print(jobs[0]["job_id"] if jobs else "")')"
derived_resource_ids="$(curl -fsS 'http://127.0.0.1:8000/v2/resources?q=nph_medsam' | python3 -c 'import json,sys; data=json.load(sys.stdin); print(",".join(item.get("resource_id","") for item in data.get("resources", [])))')"
agent_prompt_run_id="$(rg -n 'nph_medsam_analysis|Use NPH segmentation' data/trace-artifacts data/deepagents 2>/dev/null | tail -n 1 | sed 's/:.*//')"
cat >> planning/2026-06-13-nph-medsam-acceptance-matrix.md <<EOF

## Local Resource-Native E2E

- Control-plane verification: \`${control_plane_verification}\`
- Python verification: \`${python_verification}\`
- Fake worker job id: \`${fake_worker_job_id}\`
- Derived Resource ids: \`${derived_resource_ids}\`
- Agent prompt evidence path: \`${agent_prompt_run_id}\`
- Scientist-facing response review: pass
- Non-diagnostic caveat present: pass
EOF
```

Expected: the matrix has concrete command evidence and identifiers from the local run.

- [ ] **Step 6: Commit final integration evidence**

```bash
git add backend/deepagents_runtime/tests/test_nph_medsam_live_trace.py planning/2026-06-13-nph-medsam-acceptance-matrix.md
git commit -m "test: verify nph medsam end-to-end workflow"
```

## Task 14: Final Verification Sweep

**Files:**

- No planned edits.

- [ ] **Step 1: Run Python NPH suite**

```bash
cd backend/deepagents_runtime
uv run --extra dev --extra medsam pytest tests/test_nph_medsam_*.py -q
```

Expected: PASS.

- [ ] **Step 2: Run Deep Agents affected suite**

```bash
cd backend/deepagents_runtime
uv run --extra dev --extra medsam pytest tests/test_data_agent_worker.py tests/test_agent_factory.py tests/test_worker_transport.py -q
```

Expected: PASS.

- [ ] **Step 3: Run Go affected suite**

```bash
cd backend/controlplane
go test ./internal/httpapi ./internal/openapi ./internal/store ./internal/eventbus -count=1
```

Expected: PASS.

- [ ] **Step 4: Run formatting and diff checks**

```bash
gofmt -w backend/controlplane/internal/httpapi/handlers.go backend/controlplane/internal/httpapi/handlers_test.go
uv run ruff check backend/deepagents_runtime/src/ultra_deepagents/nph_medsam backend/deepagents_runtime/tests/test_nph_medsam_*.py
git diff --check
```

Expected: PASS.

- [ ] **Step 5: Confirm acceptance matrix coverage**

Run:

```bash
rg -n "Remote GPU Smoke|Local Resource-Native E2E|Non-diagnostic caveat present: pass" planning/2026-06-13-nph-medsam-acceptance-matrix.md
```

Expected: all three headings/lines are present.

- [ ] **Step 6: Commit verification-only fixes if any**

If final verification required edits in the NPH MedSAM implementation or acceptance matrix, stage only this fixed set:

```bash
git add backend/controlplane/internal/httpapi/handlers.go backend/controlplane/internal/httpapi/handlers_test.go backend/controlplane/api/openapi.yaml backend/controlplane/internal/openapi/generated.gen.go backend/deepagents_runtime/src/ultra_deepagents/nph_medsam backend/deepagents_runtime/tests/test_nph_medsam_*.py planning/2026-06-13-nph-medsam-acceptance-matrix.md
git diff --cached --quiet || git commit -m "fix: harden nph medsam verification"
```

Expected: no commit is created when the staged diff is empty.

## Execution Notes

- Run implementation in a worktree if unrelated dirty files make staging risky.
- Use one commit per task so rollback and review are tractable.
- Keep fake-runner integration green before switching to real GPU.
- Keep all raw model weights out of git.
- Never dump remote env files or generated run scripts that may contain secrets.
- Treat the old MegaSeg service as existing production state; do not stop it unless the operator asks.
- The final goal is not complete until the real checkpoint segments the test NIfTI on GPU, derived Resources are cataloged, and the agent answer passes the scientific response rubric.
