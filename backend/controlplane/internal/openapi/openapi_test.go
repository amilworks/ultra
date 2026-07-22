package openapi_test

import (
	"os"
	"regexp"
	"sort"
	"strings"
	"testing"
)

func TestOpenAPIIncludesFrontendV2Routes(t *testing.T) {
	t.Parallel()

	data, err := os.ReadFile("../../api/openapi.yaml")
	if err != nil {
		t.Fatalf("read openapi.yaml: %v", err)
	}
	doc := string(data)
	required := []string{
		"/v2/health:",
		"/v2/config/public:",
		"/v2/auth/session:",
		"/v2/auth/request-account:",
		"/v2/auth/guest:",
		"/v2/auth/login:",
		"/v2/auth/logout:",
		"/v2/threads:",
		"/v2/threads/{thread_id}:",
		"/v2/threads/{thread_id}/messages:",
		"/v2/threads/{thread_id}/runs:",
		"/v2/uploads:",
		"/v2/upload-sessions:",
		"/v2/upload-sessions/{session_id}:",
		"/v2/upload-sessions/{session_id}/files/{file_token}/chunks/{chunk_index}:",
		"/v2/upload-sessions/{session_id}/files/{file_token}/complete:",
		"/v2/upload-sessions/{session_id}/cancel:",
		"/v2/upload-sessions/{session_id}/pause:",
		"/v2/upload-sessions/{session_id}/resume:",
		"/v2/uploads/{file_id}/viewer:",
		"/v2/uploads/{file_id}/preview:",
		"/v2/uploads/{file_id}/display:",
		"/v2/uploads/{file_id}/slice:",
		"/v2/uploads/{file_id}/caption:",
		"/v2/uploads/{file_id}/hdf5/dataset:",
		"/v2/uploads/{file_id}/hdf5/preview/slice:",
		"/v2/uploads/{file_id}/hdf5/preview/atlas:",
		"/v2/uploads/{file_id}/hdf5/preview/scalar-volume:",
		"/v2/uploads/{file_id}/hdf5/preview/histogram:",
		"/v2/uploads/{file_id}/hdf5/preview/table:",
		"/v2/uploads/from-bisque:",
		"/v2/bisque/search:",
		"/v2/bisque/download:",
		"/v2/bisque/upload:",
		"/v2/bisque/unlink:",
		"/v2/resources:",
		"/v2/resource-events:",
		"/v2/resources/{file_id}:",
		"/v2/resources/{file_id}/download:",
		"/v2/resources/{file_id}/restore:",
		"/v2/resources/{file_id}/events:",
		"/v2/resources/tags/bulk:",
		"/v2/resources/shares/bulk:",
		"/v2/resources/{file_id}/shares:",
		"/v2/resources/{file_id}/shares/{grant_id}:",
		"/v2/resources/{file_id}/thumbnail:",
		"/v2/resource-collections:",
		"/v2/resource-collections/{collection_id}/resources:",
		"/v2/dataset-snapshots:",
		"/v2/dataset-snapshots/{snapshot_id}:",
		"/v2/dataset-snapshots/{snapshot_id}/shares:",
		"/v2/dataset-snapshots/{snapshot_id}/shares/{grant_id}:",
		"/v2/data-agent/jobs:",
		"/v2/data-agent/jobs/{job_id}:",
		"/v2/data-agent/jobs/{job_id}/events:",
		"/v2/data-agent/jobs/{job_id}/status:",
		"/v2/data-agent/jobs/{job_id}/control:",
		"/v2/data-agent/jobs/{job_id}/lease:",
		"/v2/runs:",
		"/v2/runs/{run_id}:",
		"/v2/runs/{run_id}/cancel:",
		"/v2/runs/{run_id}/events:",
		"/v2/runs/{run_id}/artifacts:",
		"/v2/runs/{run_id}/artifacts/download:",
		"/v2/artifacts/{artifact_id}:",
		"/v2/artifacts/{artifact_id}/download:",
		"/v2/artifacts/{artifact_id}/promote-resource:",
		"/v2/workers/heartbeat:",
		"/v2/admin/overview:",
		"/v2/admin/resources/reconcile:",
		"/v2/admin/orgs:",
		"/v2/admin/users:",
		"/v2/admin/users/{user_id}/status:",
		"/v2/admin/runs:",
		"/v2/admin/issues:",
		"/v2/admin/runs/{run_id}/cancel:",
		"/v2/admin/runs/{run_id}/requeue:",
		"/v2/admin/conversations/{conversation_id}:",
		"/v2/segment/sam3/interactive:",
		"/v2/training/models:",
		"/v2/training/models/{model_key}/status:",
		"/v2/training/models/{model_key}/retrain-requests:",
		"/v2/training/datasets:",
		"/v2/training/datasets/{dataset_id}:",
		"/v2/training/datasets/{dataset_id}/items:",
		"/v2/training/jobs:",
		"/v2/training/jobs/{job_id}:",
		"/v2/training/jobs/{job_id}/control:",
		"/v2/training/preflight:",
		"/v2/training/domains:",
		"/v2/training/domains/{domain_id}/lineages:",
		"/v2/training/lineages/{lineage_id}/fork:",
		"/v2/training/lineages/{lineage_id}/versions:",
		"/v2/training/update-proposals/preview:",
		"/v2/training/update-proposals:",
		"/v2/training/update-proposals/{proposal_id}/approve:",
		"/v2/training/update-proposals/{proposal_id}/reject:",
		"/v2/training/model-versions/{version_id}/promote:",
		"/v2/training/model-versions/{version_id}/rollback:",
		"/v2/training/merge-requests:",
		"/v2/training/merge-requests/{merge_id}/approve:",
		"/v2/training/merge-requests/{merge_id}/reject:",
		"/v2/inference/jobs:",
		"/v2/inference/jobs/{job_id}/result:",
		"/v2/model-health:",
		"/v2/admin/model-health:",
		"V2ThreadRecord:",
		"V2RunRecord:",
		"V2GraphEventRecord:",
		"V2ArtifactRecord:",
		"V2PromoteArtifactResourceRequest:",
		"V2UploadSessionCreateRequest:",
		"V2UploadSessionEventRecord:",
		"V2UploadSessionResponse:",
		"V2UploadChunkResponse:",
		"V2UploadSessionFileCompleteResponse:",
		"V2ResourceBulkTagRequest:",
		"V2ResourceBulkTagResponse:",
		"V2ResourceEventListResponse:",
		"V2ResourceEventsResponse:",
		"V2ResourcePatchRequest:",
		"V2ResourceCollectionRecord:",
		"V2ResourceCollectionCreateRequest:",
		"V2ResourceCollectionShareGrantsCreateResponse:",
		"V2ResourceCollectionAddResourcesResponse:",
		"V2DatasetSnapshotRecord:",
		"V2DatasetSnapshotCreateRequest:",
		"V2DatasetSnapshotResourceQuery:",
		"V2DatasetSnapshotResponse:",
		"V2DatasetSnapshotsResponse:",
		"V2DatasetSnapshotShareGrantRecord:",
		"V2DatasetSnapshotShareGrantCreateRequest:",
		"V2DatasetSnapshotShareGrantResponse:",
		"V2DatasetSnapshotShareGrantsResponse:",
		"V2DataAgentJobRecord:",
		"V2DataAgentJobEventRecord:",
		"V2DataAgentJobCreateRequest:",
		"V2DataAgentJobAppendEventRequest:",
		"V2DataAgentJobStatusUpdateRequest:",
		"V2DataAgentJobControlRequest:",
		"V2DataAgentJobLeaseRecord:",
		"V2DataAgentJobResponse:",
		"V2DataAgentJobsResponse:",
		"V2ResourceReconcileResponse:",
		"V2WorkerHeartbeatRequest:",
		"V2WorkerHeartbeatRecord:",
		"V2AdminWorkerRecord:",
		"V2AdminOverviewResponse:",
		"V2AdminUploadSessionMetrics:",
		"V2TrainingModelsResponse:",
		"V2PrairieStatusResponse:",
		"V2NotConfiguredResponse:",
		"Idempotency-Key",
		"idempotency_key:",
	}
	for _, needle := range required {
		if !strings.Contains(doc, needle) {
			t.Fatalf("openapi.yaml missing %s", needle)
		}
	}
}

func TestHdf5RasterContractsExposeScientificSelectorsAndPngMediaType(t *testing.T) {
	t.Parallel()

	data, err := os.ReadFile("../../api/openapi.yaml")
	if err != nil {
		t.Fatalf("read openapi.yaml: %v", err)
	}
	doc := string(data)
	atlasStart := strings.Index(doc, "  /v2/uploads/{file_id}/hdf5/preview/atlas:")
	atlasEnd := strings.Index(doc, "  /v2/uploads/{file_id}/hdf5/preview/scalar-volume:")
	sliceStart := strings.Index(doc, "  /v2/uploads/{file_id}/hdf5/preview/slice:")
	sliceEnd := atlasStart
	if atlasStart < 0 || atlasEnd <= atlasStart || sliceStart < 0 || sliceEnd <= sliceStart {
		t.Fatal("HDF5 slice/atlas OpenAPI sections are missing or out of order")
	}

	for name, section := range map[string]string{
		"slice": doc[sliceStart:sliceEnd],
		"atlas": doc[atlasStart:atlasEnd],
	} {
		for _, marker := range []string{"- name: feature_ids", "image/png:"} {
			if !strings.Contains(section, marker) {
				t.Errorf("HDF5 %s contract missing %q", name, marker)
			}
		}
	}
	if !strings.Contains(doc[atlasStart:atlasEnd], "- name: component") {
		t.Error("HDF5 atlas contract is missing the vector component selector")
	}
}

func TestScalarVolumeResponseHeadersAreRequired(t *testing.T) {
	t.Parallel()

	data, err := os.ReadFile("../../api/openapi.yaml")
	if err != nil {
		t.Fatalf("read openapi.yaml: %v", err)
	}
	doc := string(data)
	paths := []string{
		"/v2/uploads/{file_id}/scalar-volume:",
		"/v2/uploads/{file_id}/hdf5/preview/scalar-volume:",
	}
	headers := []string{
		"x-volume-width",
		"x-volume-height",
		"x-volume-depth",
		"x-volume-dtype",
		"x-volume-bytes-per-voxel",
		"x-volume-raw-min",
		"x-volume-raw-max",
		"x-volume-scl-slope",
		"x-volume-scl-inter",
		"x-volume-channel",
		"x-volume-time",
		"x-volume-source-width",
		"x-volume-source-height",
		"x-volume-source-depth",
		"x-volume-downsample-x",
		"x-volume-downsample-y",
		"x-volume-downsample-z",
		"x-volume-preview-policy",
	}
	for _, path := range paths {
		start := strings.Index(doc, "  "+path)
		if start < 0 {
			t.Fatalf("missing scalar response path %s", path)
		}
		section := doc[start:]
		if next := strings.Index(section[1:], "\n  /v2/"); next >= 0 {
			section = section[:next+1]
		}
		for _, header := range headers {
			contract := "            " + header + ":\n              required: true"
			if !strings.Contains(section, contract) {
				t.Errorf("%s response header %s is not required", path, header)
			}
		}
	}

	// oapi-codegen honors required response headers as non-pointer fields and
	// writes them unconditionally. Keep this assertion so generator upgrades
	// cannot silently weaken the renderer's boundary contract.
	generated, err := os.ReadFile("generated.gen.go")
	if err != nil {
		t.Fatalf("read generated client: %v", err)
	}
	for _, declaration := range []string{
		"type GetUploadScalarVolume200ResponseHeaders struct {",
		"type GetUploadHdf5ScalarVolume200ResponseHeaders struct {",
		"XVolumeSclSlope      float32",
		"XVolumeSclInter      float32",
	} {
		if !strings.Contains(string(generated), declaration) {
			t.Fatalf("generated response contract changed: missing %q", declaration)
		}
	}
}

func TestFrontendV2RoutesAreDocumented(t *testing.T) {
	t.Parallel()

	openAPIBytes, err := os.ReadFile("../../api/openapi.yaml")
	if err != nil {
		t.Fatalf("read openapi.yaml: %v", err)
	}
	frontendBytes, err := os.ReadFile("../../../../frontend/src/lib/api.ts")
	if err != nil {
		t.Fatalf("read frontend api.ts: %v", err)
	}

	documented := documentedV2Paths(string(openAPIBytes))
	missing := make([]string, 0)
	for _, path := range frontendV2Paths(string(frontendBytes)) {
		if !documented[path] {
			missing = append(missing, path)
		}
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		t.Fatalf("frontend V2 routes missing from openapi.yaml:\n%s", strings.Join(missing, "\n"))
	}
}

func documentedV2Paths(doc string) map[string]bool {
	paths := make(map[string]bool)
	for _, line := range strings.Split(doc, "\n") {
		trimmed := strings.TrimSpace(line)
		if !strings.HasPrefix(trimmed, "/v2/") || !strings.HasSuffix(trimmed, ":") {
			continue
		}
		paths[normalizePathPattern(strings.TrimSuffix(trimmed, ":"))] = true
	}
	return paths
}

func frontendV2Paths(source string) []string {
	source = stripTypeScriptComments(source)
	seen := make(map[string]bool)
	for offset := 0; ; {
		idx := strings.Index(source[offset:], "/v2/")
		if idx < 0 {
			break
		}
		start := offset + idx
		end := len(source)
		for i := start; i < len(source); i++ {
			switch source[i] {
			case '"', '\'', '`':
				end = i
			}
			if end != len(source) {
				break
			}
		}
		path := normalizePathPattern(source[start:end])
		if path != "" {
			seen[path] = true
		}
		offset = start + len("/v2/")
	}
	paths := make([]string, 0, len(seen))
	for path := range seen {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	return paths
}

func stripTypeScriptComments(source string) string {
	var b strings.Builder
	b.Grow(len(source))
	var quote byte
	escaped := false
	inLineComment := false
	inBlockComment := false

	for i := 0; i < len(source); i++ {
		ch := source[i]
		if inLineComment {
			if ch == '\n' {
				inLineComment = false
				b.WriteByte(ch)
			} else {
				b.WriteByte(' ')
			}
			continue
		}
		if inBlockComment {
			if ch == '*' && i+1 < len(source) && source[i+1] == '/' {
				inBlockComment = false
				b.WriteString("  ")
				i++
			} else if ch == '\n' {
				b.WriteByte(ch)
			} else {
				b.WriteByte(' ')
			}
			continue
		}
		if quote != 0 {
			b.WriteByte(ch)
			if escaped {
				escaped = false
				continue
			}
			if ch == '\\' {
				escaped = true
				continue
			}
			if ch == quote {
				quote = 0
			}
			continue
		}
		if ch == '"' || ch == '\'' || ch == '`' {
			quote = ch
			b.WriteByte(ch)
			continue
		}
		if ch == '/' && i+1 < len(source) {
			next := source[i+1]
			if next == '/' {
				inLineComment = true
				b.WriteString("  ")
				i++
				continue
			}
			if next == '*' {
				inBlockComment = true
				b.WriteString("  ")
				i++
				continue
			}
		}
		b.WriteByte(ch)
	}
	return b.String()
}

var (
	templateExprPattern = regexp.MustCompile(`\$\{[^}]+\}`)
	pathParamPattern    = regexp.MustCompile(`\{[^}]+\}`)
)

func normalizePathPattern(path string) string {
	if queryIdx := strings.IndexByte(path, '?'); queryIdx >= 0 {
		path = path[:queryIdx]
	}
	path = templateExprPattern.ReplaceAllString(path, "{param}")
	path = pathParamPattern.ReplaceAllString(path, "{param}")
	path = strings.TrimRight(path, "/")
	return path
}
