package openapi_test

import (
	"os"
	"strings"
	"testing"
)

func TestOpenAPICalphadLedgerDocumentsBoundedPagingAndExactEvidenceReplay(t *testing.T) {
	t.Parallel()
	docBytes, err := os.ReadFile("../../api/openapi.yaml")
	if err != nil {
		t.Fatalf("read openapi.yaml: %v", err)
	}
	doc := string(docBytes)
	for _, required := range []string{
		"/v2/resources/{file_id}/calphad/validations/{validation_id}/evidence:",
		"operationId: getCalphadValidationEvidence",
		"operationId: getCalphadLedger",
		"maximum: 500",
		"default: 100",
		"name: cursor",
		"latest_validation field always reports",
		"has_more:",
		"next_cursor:",
		"X-Ultra-Calphad-Validation-Id:",
		"X-Ultra-Content-Sha256:",
		`enum: ["private, immutable"]`,
		"format: binary",
		"filesystem paths are never used as a fallback",
	} {
		if !strings.Contains(doc, required) {
			t.Errorf("CALPHAD ledger read contract missing %q", required)
		}
	}
	generated, err := os.ReadFile("generated.gen.go")
	if err != nil {
		t.Fatalf("read generated.gen.go: %v", err)
	}
	for _, required := range []string{
		`json:"has_more"`,
		`json:"next_cursor,omitempty"`,
		"type GetCalphadLedgerParams struct",
		"Limit *int",
		"Cursor *string",
		"type GetCalphadValidationEvidenceRequestObject struct",
		"type GetCalphadValidationEvidence200JSONResponse struct",
	} {
		if !strings.Contains(string(generated), required) {
			t.Errorf("generated CALPHAD ledger read contract missing %q", required)
		}
	}
}
