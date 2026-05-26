package openapi_test

import (
	"os"
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
		"/v1/health:",
		"/v1/config/public:",
		"/v1/auth/session:",
		"/v2/threads:",
		"/v2/threads/{thread_id}:",
		"/v2/threads/{thread_id}/messages:",
		"/v2/threads/{thread_id}/runs:",
		"/v2/runs:",
		"/v2/runs/{run_id}:",
		"/v2/runs/{run_id}/cancel:",
		"/v2/runs/{run_id}/events:",
		"/v2/runs/{run_id}/artifacts:",
		"/v2/artifacts/{artifact_id}:",
		"V2ThreadRecord:",
		"V2RunRecord:",
		"V2GraphEventRecord:",
		"V2ArtifactRecord:",
	}
	for _, needle := range required {
		if !strings.Contains(doc, needle) {
			t.Fatalf("openapi.yaml missing %s", needle)
		}
	}
}
