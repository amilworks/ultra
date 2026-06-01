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
		"/v2/auth/guest:",
		"/v2/auth/login:",
		"/v2/auth/logout:",
		"/v2/threads:",
		"/v2/threads/{thread_id}:",
		"/v2/threads/{thread_id}/messages:",
		"/v2/threads/{thread_id}/runs:",
		"/v2/uploads:",
		"/v2/uploads/{file_id}/viewer:",
		"/v2/uploads/{file_id}/preview:",
		"/v2/uploads/{file_id}/display:",
		"/v2/uploads/{file_id}/slice:",
		"/v2/uploads/{file_id}/caption:",
		"/v2/uploads/from-bisque:",
		"/v2/resources:",
		"/v2/resources/{file_id}:",
		"/v2/resources/{file_id}/thumbnail:",
		"/v2/runs:",
		"/v2/runs/{run_id}:",
		"/v2/runs/{run_id}/cancel:",
		"/v2/runs/{run_id}/events:",
		"/v2/runs/{run_id}/artifacts:",
		"/v2/runs/{run_id}/artifacts/download:",
		"/v2/artifacts/{artifact_id}:",
		"/v2/artifacts/{artifact_id}/download:",
		"/v2/workers/heartbeat:",
		"/v2/admin/overview:",
		"/v2/admin/orgs:",
		"/v2/admin/users:",
		"/v2/admin/runs:",
		"/v2/admin/issues:",
		"/v2/admin/runs/{run_id}/cancel:",
		"/v2/admin/runs/{run_id}/requeue:",
		"/v2/admin/conversations/{conversation_id}:",
		"/v2/segment/sam3/interactive:",
		"/v2/training/models:",
		"/v2/training/prairie/status:",
		"/v2/training/prairie/retrain-requests:",
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
		"V2WorkerHeartbeatRequest:",
		"V2WorkerHeartbeatRecord:",
		"V2AdminWorkerRecord:",
		"V2AdminOverviewResponse:",
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
