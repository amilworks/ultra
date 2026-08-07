package openapi_test

import (
	"os"
	"strings"
	"testing"

	controlopenapi "github.com/amilworks/bisque-ultra/backend/controlplane/internal/openapi"
)

func TestAdminRetentionBacklogContract(t *testing.T) {
	t.Parallel()

	document, err := os.ReadFile("../../api/openapi.yaml")
	if err != nil {
		t.Fatalf("read openapi.yaml: %v", err)
	}
	section := string(document)
	for _, marker := range []string{
		"V2AdminRetentionBacklog:",
		"retention_backlog:",
		"expired_resources:",
		"reclaimable_bytes:",
		"blocked_resources:",
		"blocked_bytes:",
		"purging_resources:",
		"purging_bytes:",
	} {
		if !strings.Contains(section, marker) {
			t.Fatalf("admin retention backlog contract missing %q", marker)
		}
	}

	// Keep the generated binding in the contract gate: typed clients must be
	// able to represent every operator-visible retention state.
	response := controlopenapi.V2AdminOverviewResponse{
		RetentionBacklog: controlopenapi.V2AdminRetentionBacklog{
			ExpiredResources: 1,
			ReclaimableBytes: 2,
			BlockedResources: 3,
			BlockedBytes:     4,
			PurgingResources: 5,
			PurgingBytes:     6,
		},
	}
	if response.RetentionBacklog.BlockedResources != 3 {
		t.Fatalf("generated retention backlog binding = %+v", response.RetentionBacklog)
	}
}
