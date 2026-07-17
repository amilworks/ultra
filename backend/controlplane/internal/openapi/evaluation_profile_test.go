package openapi

import (
	"os"
	"strings"
	"testing"
)

func TestRunCreateRemoteMutationEnumsAreBounded(t *testing.T) {
	t.Parallel()
	if !BisqueUpload.Valid() || !BisqueCreateDataset.Valid() {
		t.Fatal("generated remote mutation intents are not valid")
	}
	if V2RunCreateRequestRemoteMutationIntents("unknown_intent").Valid() {
		t.Fatal("generated remote mutation intent enum accepts an unknown value")
	}

	document, err := os.ReadFile("../../api/openapi.yaml")
	if err != nil {
		t.Fatalf("read OpenAPI document: %v", err)
	}
	for _, required := range []string{
		"remote_mutation_intents:",
		"enum: [bisque.upload, bisque.create_dataset]",
		"Protected evaluation profiles forbid it",
	} {
		if !strings.Contains(string(document), required) {
			t.Errorf("OpenAPI remote mutation contract missing %q", required)
		}
	}
}

// TestRunCreateExposesNoEvaluationProfileWireSurface guards the materials
// removal: the create-run contract must not reintroduce a profile property on
// either side of the generated boundary.
func TestRunCreateExposesNoEvaluationProfileWireSurface(t *testing.T) {
	t.Parallel()
	document, err := os.ReadFile("../../api/openapi.yaml")
	if err != nil {
		t.Fatalf("read OpenAPI document: %v", err)
	}
	if strings.Contains(string(document), "evaluation_profile:") {
		t.Error("OpenAPI document still declares an evaluation_profile property")
	}
	swagger, err := GetSwagger()
	if err != nil {
		t.Fatalf("GetSwagger: %v", err)
	}
	schema, ok := swagger.Components.Schemas["V2RunCreateRequest"]
	if !ok {
		t.Fatal("V2RunCreateRequest schema is missing")
	}
	if _, exists := schema.Value.Properties["evaluation_profile"]; exists {
		t.Error("embedded spec still declares V2RunCreateRequest.evaluation_profile")
	}
}
