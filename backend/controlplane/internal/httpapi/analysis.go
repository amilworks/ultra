package httpapi

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

// Batch model inference (MegaSeg / RareSpot) rides the Data-Agent job backbone:
// POST /v2/analysis/batch creates an analysis.<model> data-agent job over N images and
// an auto-created results collection; the worker processes each image, writes outputs to
// the shared upload root, and registers them via POST /v2/data-agent/jobs/{id}/outputs so
// they land on the Resources page (grouped + downloadable). Progress/cancel reuse the
// existing data-agent job endpoints (GET .../jobs/{id}, POST .../jobs/{id}/control).

// analysisBatchMaxImages caps a single batch so a runaway job can't tie up the GPU
// indefinitely and progress/ETA stay meaningful. Submissions above this are rejected.
const analysisBatchMaxImages = 1000

// analysisOutputPrefix is the upload-root-relative directory the worker writes batch
// outputs into. The output-registration endpoint refuses to register a file outside it.
const analysisOutputPrefix = "analysis"

// dataAgentJobOutputStore links output resources to a data-agent job. Implemented by the
// Postgres store only; in the in-memory (dev) store the endpoint reports unavailable,
// which is fine because batch analysis runs against NATS + Postgres in production.
type dataAgentJobOutputStore interface {
	LinkDataAgentJobResource(context.Context, domain.LinkDataAgentJobResourceInput) error
}

func (deps ServerDeps) dataAgentJobOutputStore() (dataAgentJobOutputStore, bool) {
	if deps.Store == nil {
		return nil, false
	}
	store, ok := deps.Store.(dataAgentJobOutputStore)
	return store, ok
}

type createBatchAnalysisJobRequest struct {
	Model              string         `json:"model"`
	ResourceIDs        []string       `json:"resource_ids"`
	SourceCollectionID string         `json:"source_collection_id"`
	ProjectID          string         `json:"project_id"`
	Params             domain.JSONMap `json:"params"`
	CollectionName     string         `json:"collection_name"`
}

type batchAnalysisJobResponse struct {
	Job               domain.DataAgentJobRecord        `json:"job"`
	Events            []domain.DataAgentJobEventRecord `json:"events"`
	ResultsCollection domain.ResourceCollectionRecord  `json:"results_collection"`
}

type analysisOutputItem struct {
	ResourceID       string         `json:"resource_id"`
	StoragePath      string         `json:"storage_path"`
	OriginalName     string         `json:"original_name"`
	ContentType      string         `json:"content_type"`
	SizeBytes        int64          `json:"size_bytes"`
	SHA256           string         `json:"sha256"`
	SourceResourceID string         `json:"source_resource_id"`
	ArtifactKind     string         `json:"artifact_kind"`
	Metadata         domain.JSONMap `json:"metadata"`
	AddToCollection  *bool          `json:"add_to_collection"`
}

type registerAnalysisOutputsRequest struct {
	// CollectionID lets the worker name the results collection explicitly (it has it from the
	// job envelope), so registration doesn't depend on the job's stored metadata surviving.
	CollectionID string               `json:"collection_id"`
	Outputs      []analysisOutputItem `json:"outputs"`
}

type registerAnalysisOutputsResponse struct {
	Count      int              `json:"count"`
	Registered []resourceRecord `json:"registered"`
}

func normalizeAnalysisModel(value string) (model string, jobType string, display string, err error) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "megaseg":
		return "megaseg", "analysis.megaseg", "MegaSeg", nil
	case "rarespot":
		return "rarespot", "analysis.rarespot", "RareSpot", nil
	default:
		return "", "", "", errors.New("model must be megaseg or rarespot")
	}
}

func (deps ServerDeps) handleCreateBatchAnalysisJob(w http.ResponseWriter, r *http.Request) {
	jobs, ok := deps.dataAgentJobStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "data agent jobs are not configured"})
		return
	}
	collections, ok := deps.resourceCollectionStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource collections are not configured"})
		return
	}
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if err := deps.ensureUploadCatalogMigrated(r.Context(), root); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	var req createBatchAnalysisJobRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	model, jobType, display, err := normalizeAnalysisModel(req.Model)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	principal := deps.principalFromRequest(r, "")
	sourceCollectionID := strings.TrimSpace(req.SourceCollectionID)
	resourceIDs := uniqueTrimmedStringValues(req.ResourceIDs)
	if len(resourceIDs) == 0 && sourceCollectionID != "" {
		page, err := collections.ListResourcesForCollectionForUser(r.Context(), domain.ResourceCollectionResourceListInput{
			CollectionID: sourceCollectionID,
			UserID:       principal.UserID,
			OrgID:        principal.OrgID,
			Limit:        analysisBatchMaxImages + 1,
		})
		if err != nil {
			writeStoreError(w, err)
			return
		}
		if page.TotalCount > analysisBatchMaxImages {
			writeError(w, http.StatusBadRequest, fmt.Errorf("collection has %d images, above the %d-per-batch limit; split it into smaller batches", page.TotalCount, analysisBatchMaxImages))
			return
		}
		resourceIDs = make([]string, 0, len(page.Resources))
		for _, resource := range page.Resources {
			resourceIDs = append(resourceIDs, resource.ResourceID)
		}
	}
	if len(resourceIDs) == 0 {
		writeError(w, http.StatusBadRequest, errors.New("resource_ids or source_collection_id is required"))
		return
	}
	if len(resourceIDs) > analysisBatchMaxImages {
		writeError(w, http.StatusBadRequest, fmt.Errorf("%d images requested, above the %d-per-batch limit; split it into smaller batches", len(resourceIDs), analysisBatchMaxImages))
		return
	}
	params := mapOrEmptyJSON(req.Params)
	now := domain.Now()

	collectionName := strings.TrimSpace(req.CollectionName)
	if collectionName == "" {
		collectionName = fmt.Sprintf("%s results — %d image%s — %s", display, len(resourceIDs), plural(len(resourceIDs)), now.UTC().Format("2006-01-02 15:04"))
	}
	collection, err := collections.CreateResourceCollection(r.Context(), domain.CreateResourceCollectionInput{
		OwnerUserID:    principal.UserID,
		OwnerOrgID:     principal.OrgID,
		OwnerRole:      principal.Role,
		ProjectID:      strings.TrimSpace(req.ProjectID),
		Name:           collectionName,
		Description:    fmt.Sprintf("%s batch analysis results for %d images.", display, len(resourceIDs)),
		CollectionType: "analysis_results",
		CreatedAt:      now,
		Metadata: domain.JSONMap{
			"analysis": domain.JSONMap{
				"model":         model,
				"model_display": display,
				"input_count":   len(resourceIDs),
				"created_at":    now.UTC().Format(time.RFC3339Nano),
			},
		},
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}

	inputSelector := domain.JSONMap{
		"resource_ids": append([]string(nil), resourceIDs...),
		"model":        model,
		"params":       params,
	}
	if sourceCollectionID != "" {
		inputSelector["source_collection_id"] = sourceCollectionID
	}
	jobMetadata := domain.JSONMap{
		"model":                   model,
		"model_display":           display,
		"params":                  params,
		"results_collection_id":   collection.CollectionID,
		"results_collection_name": collection.Name,
		"analysis_output_prefix":  analysisOutputPrefix,
	}
	job, err := jobs.CreateDataAgentJob(r.Context(), domain.CreateDataAgentJobInput{
		OwnerUserID:     principal.UserID,
		OwnerOrgID:      principal.OrgID,
		OwnerRole:       principal.Role,
		ProjectID:       strings.TrimSpace(req.ProjectID),
		JobType:         jobType,
		Status:          "queued",
		ResourceIDs:     resourceIDs,
		ResourceCount:   len(resourceIDs),
		InputSelector:   inputSelector,
		CreatedByUserID: principal.UserID,
		CreatedAt:       now,
		Metadata:        jobMetadata,
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	for _, resourceID := range resourceIDs {
		deps.recordResourceEvent(r.Context(), resourceID, principal, "resource.analysis_job_queued", domain.JSONMap{
			"job_id":                job.JobID,
			"job_type":              job.JobType,
			"model":                 model,
			"results_collection_id": collection.CollectionID,
		})
	}
	if err := deps.dispatchDataAgentJob(r.Context(), jobs, job, principal); err != nil {
		writeError(w, http.StatusServiceUnavailable, err)
		return
	}
	events, err := jobs.ListDataAgentJobEvents(r.Context(), job.JobID, principal.UserID, principal.OrgID, parseLimit(r, 200))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusAccepted, batchAnalysisJobResponse{Job: job, Events: events, ResultsCollection: collection})
}

// handleRegisterAnalysisOutputs is called by the batch-analysis worker as it produces
// results. Each output file has already been written by the worker into the shared upload
// root under analysis/<job_id>/...; here we register it as a resource (no bytes move),
// link it to the job as io_role='output', and add it to the job's results collection.
func (deps ServerDeps) handleRegisterAnalysisOutputs(w http.ResponseWriter, r *http.Request) {
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
	outputs, ok := deps.dataAgentJobOutputStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "analysis output linkage is not configured"})
		return
	}
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	var req registerAnalysisOutputsRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	if len(req.Outputs) == 0 {
		writeError(w, http.StatusBadRequest, errors.New("outputs is required"))
		return
	}
	principal := deps.principalFromRequest(r, "")
	job, err := jobs.GetDataAgentJobForUser(r.Context(), chi.URLParam(r, "job_id"), principal.UserID, principal.OrgID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	model := jsonMapString(job.Metadata, "model")
	resultsCollectionID := strings.TrimSpace(req.CollectionID)
	if resultsCollectionID == "" {
		resultsCollectionID = jsonMapString(job.Metadata, "results_collection_id")
	}
	collections, hasCollections := deps.resourceCollectionStore()
	now := domain.Now()

	registered := make([]resourceRecord, 0, len(req.Outputs))
	for index, item := range req.Outputs {
		resolved, ok := analysisOutputStoragePath(root, item.StoragePath)
		if !ok {
			writeError(w, http.StatusBadRequest, fmt.Errorf("output %d: storage_path %q must be a relative path under %s/", index, item.StoragePath, analysisOutputPrefix))
			return
		}
		info, statErr := os.Stat(resolved)
		if statErr != nil || info.IsDir() {
			writeError(w, http.StatusBadRequest, fmt.Errorf("output %d: file %q is not readable", index, item.StoragePath))
			return
		}
		size := item.SizeBytes
		if size <= 0 {
			size = info.Size()
		}
		originalName := safeOriginalFilename(item.OriginalName)
		if strings.TrimSpace(item.OriginalName) == "" {
			originalName = safeOriginalFilename(filepath.Base(resolved))
		}
		contentType := strings.TrimSpace(item.ContentType)
		if contentType == "" {
			contentType = contentTypeForUpload(originalName, "")
		}
		resourceID := strings.TrimSpace(item.ResourceID)
		if resourceID == "" {
			resourceID = domain.NewID("file")
		}
		artifactKind := strings.TrimSpace(item.ArtifactKind)
		provenance := domain.JSONMap{
			"source":        "analysis_output",
			"job_id":        job.JobID,
			"model":         model,
			"artifact_kind": artifactKind,
			"produced_at":   now.UTC().Format(time.RFC3339Nano),
		}
		if sourceResourceID := strings.TrimSpace(item.SourceResourceID); sourceResourceID != "" {
			provenance["source_resource_id"] = sourceResourceID
		}
		for key, value := range mapOrEmptyJSON(item.Metadata) {
			provenance[key] = value
		}
		resource, err := catalog.UpsertResource(r.Context(), domain.UpsertResourceInput{
			ResourceID:   resourceID,
			OriginalName: originalName,
			ContentType:  contentType,
			SizeBytes:    size,
			SHA256:       strings.TrimSpace(item.SHA256),
			StorageURI:   fileStorageURI(resolved),
			StoragePath:  filepath.ToSlash(relativeUploadPath(root, resolved)),
			SourceType:   "analysis",
			ResourceKind: resourceKindForContent(originalName, contentType),
			ProjectID:    job.ProjectID,
			OwnerUserID:  principal.UserID,
			OwnerOrgID:   principal.OrgID,
			OwnerRole:    principal.Role,
			Status:       "active",
			CreatedAt:    now,
			UpdatedAt:    now,
			Metadata:     domain.JSONMap{"analysis": provenance},
		})
		if err != nil {
			writeError(w, http.StatusInternalServerError, fmt.Errorf("output %d: register resource: %w", index, err))
			return
		}
		if err := outputs.LinkDataAgentJobResource(r.Context(), domain.LinkDataAgentJobResourceInput{
			JobID:      job.JobID,
			ResourceID: resource.ResourceID,
			IORole:     "output",
			Position:   index,
			Metadata: domain.JSONMap{
				"artifact_kind":      artifactKind,
				"source_resource_id": strings.TrimSpace(item.SourceResourceID),
			},
		}); err != nil {
			writeError(w, http.StatusInternalServerError, fmt.Errorf("output %d: link to job: %w", index, err))
			return
		}
		addToCollection := item.AddToCollection == nil || *item.AddToCollection
		if addToCollection && hasCollections && resultsCollectionID != "" {
			// Grouping is best-effort: a collection hiccup must not drop a produced result
			// (the resource is registered and downloadable regardless).
			_, _ = collections.AddResourcesToCollection(r.Context(), domain.AddResourcesToCollectionInput{
				CollectionID:  resultsCollectionID,
				OwnerUserID:   principal.UserID,
				OwnerOrgID:    principal.OrgID,
				ResourceIDs:   []string{resource.ResourceID},
				AddedByUserID: principal.UserID,
				AddedAt:       now,
			})
		}
		deps.recordResourceEvent(r.Context(), resource.ResourceID, principal, "resource.analysis_output_registered", domain.JSONMap{
			"job_id":                job.JobID,
			"model":                 model,
			"artifact_kind":         artifactKind,
			"source_resource_id":    strings.TrimSpace(item.SourceResourceID),
			"results_collection_id": resultsCollectionID,
		})
		registered = append(registered, deps.resourceRecordFromCatalog(root, resource))
	}
	writeJSON(w, http.StatusCreated, registerAnalysisOutputsResponse{Count: len(registered), Registered: registered})
}

// analysisOutputStoragePath validates a worker-supplied upload-root-relative path: it must
// be relative, stay under the upload root after cleaning, and live under the analysis/
// prefix (so a worker token can only register files the analysis worker wrote there).
func analysisOutputStoragePath(root string, storagePath string) (string, bool) {
	storagePath = strings.TrimSpace(storagePath)
	if storagePath == "" || filepath.IsAbs(storagePath) {
		return "", false
	}
	cleanedRel := filepath.Clean(filepath.FromSlash(storagePath))
	if cleanedRel == "." || strings.HasPrefix(cleanedRel, "..") {
		return "", false
	}
	parts := strings.SplitN(filepath.ToSlash(cleanedRel), "/", 2)
	if len(parts) == 0 || parts[0] != analysisOutputPrefix {
		return "", false
	}
	resolved := filepath.Clean(filepath.Join(root, cleanedRel))
	if !pathIsUnderRoot(root, resolved) {
		return "", false
	}
	return resolved, true
}

func relativeUploadPath(root string, resolved string) string {
	rel, err := filepath.Rel(root, resolved)
	if err != nil {
		return filepath.Base(resolved)
	}
	return rel
}

func jsonMapString(m domain.JSONMap, key string) string {
	if m == nil {
		return ""
	}
	value, ok := m[key]
	if !ok {
		return ""
	}
	text := strings.TrimSpace(fmt.Sprint(value))
	if text == "<nil>" {
		return ""
	}
	return text
}

func plural(n int) string {
	if n == 1 {
		return ""
	}
	return "s"
}
