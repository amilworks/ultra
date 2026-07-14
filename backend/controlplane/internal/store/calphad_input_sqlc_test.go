package store

import (
	"os"
	"strings"
	"testing"
)

func TestSQLCModelsIncludeExactCalphadInputBlob(t *testing.T) {
	t.Parallel()
	generated, err := os.ReadFile("sqlc/models.go")
	if err != nil {
		t.Fatalf("read generated SQLC models: %v", err)
	}
	for _, required := range []string{
		"type ControlCalphadInputBlob struct",
		"InputSha256",
		"InputSizeBytes",
		"Encoding",
		"Payload",
		"type ControlCalphadRevision struct",
		"AssessmentPressureMinPa",
		"AssessmentPressureMaxPa",
		"type ControlCalphadValidationEvent struct",
		"type ControlCalphadTenantCapacity struct",
		"MaxRetainedBytes",
		"MaxValidationEvents",
		"RetainedInputBytes",
		"RetainedEvidenceBytes",
		"ValidationEvents",
	} {
		if !strings.Contains(string(generated), required) {
			t.Errorf("generated SQLC input-blob contract missing %q", required)
		}
	}
}
