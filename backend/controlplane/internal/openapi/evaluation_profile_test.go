package openapi

import (
	"os"
	"strings"
	"testing"
)

func TestRunCreateMaterialsCleanroomAndRemoteMutationEnumsAreBounded(t *testing.T) {
	t.Parallel()
	if !MaterialsCleanroomV1.Valid() {
		t.Fatal("generated materials cleanroom evaluation profile is not valid")
	}
	if V2RunCreateRequestEvaluationProfile("unknown_profile").Valid() {
		t.Fatal("generated evaluation profile enum accepts an unknown value")
	}
	profile := MaterialsCleanroomV1
	request := V2RunCreateRequest{EvaluationProfile: &profile}
	if request.EvaluationProfile == nil || *request.EvaluationProfile != MaterialsCleanroomV1 {
		t.Fatalf("generated create-run profile = %#v", request.EvaluationProfile)
	}

	document, err := os.ReadFile("../../api/openapi.yaml")
	if err != nil {
		t.Fatalf("read OpenAPI document: %v", err)
	}
	for _, required := range []string{
		"evaluation_profile:",
		"enum: [materials_cleanroom_v1]",
		"Only an administrator",
		"server-owned run metadata",
		"remote_mutation_intents:",
		"enum: [bisque.upload, bisque.create_dataset]",
		"Protected evaluation profiles forbid it",
	} {
		if !strings.Contains(string(document), required) {
			t.Errorf("OpenAPI materials capability contract missing %q", required)
		}
	}
}
