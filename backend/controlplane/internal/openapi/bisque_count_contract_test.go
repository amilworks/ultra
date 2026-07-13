package openapi

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func TestBisqueCountOpenAPIPathsMatchImplementedAuthorityContracts(t *testing.T) {
	t.Parallel()
	data, err := os.ReadFile("../../api/openapi.yaml")
	if err != nil {
		t.Fatalf("read OpenAPI document: %v", err)
	}
	document := string(data)
	for _, contract := range []struct {
		path      string
		operation string
		request   string
		response  string
	}{
		{
			path:      "/v2/bisque/dataset-members",
			operation: "getBisqueDatasetMembers",
			request:   "V2BisqueDatasetMembersRequest",
			response:  "V2BisqueDatasetMembersResponse",
		},
		{
			path:      "/v2/bisque/image-annotations",
			operation: "getBisqueImageAnnotations",
			request:   "V2BisqueImageAnnotationsRequest",
			response:  "V2BisqueImageAnnotationsResponse",
		},
		{
			path:      "/v2/bisque/dataset-annotations",
			operation: "getBisqueDatasetAnnotations",
			request:   "V2BisqueDatasetAnnotationsRequest",
			response:  "V2BisqueDatasetAnnotationsResponse",
		},
	} {
		section := openAPIPathSection(t, document, contract.path)
		for _, required := range []string{
			"operationId: " + contract.operation,
			"- UltraWorkOSSession: []",
			"- UltraWorkerToken: []",
			"#/components/parameters/ConditionalUltraWorkerRunID",
			"#/components/parameters/ConditionalUltraWorkerID",
			"#/components/parameters/ConditionalUltraRunLeaseToken",
			"#/components/parameters/OptionalUltraBisqueSessionID",
			"#/components/schemas/" + contract.request,
			"#/components/schemas/" + contract.response,
			"\"200\":",
			"\"400\":",
			"\"401\":",
			"\"403\":",
			"\"413\":",
			"\"500\":",
			"\"501\":",
			"\"502\":",
		} {
			if !strings.Contains(section, required) {
				t.Errorf("%s contract missing %q", contract.path, required)
			}
		}
	}

	for _, required := range []string{
		"UltraWorkOSSession:",
		"name: ultra_workos_session",
		"V2BisqueDatasetMembersRequest:",
		"V2BisqueDatasetMembersResponse:",
		"V2BisqueImageAnnotationsRequest:",
		"V2BisqueImageAnnotationsResponse:",
		"V2BisqueDatasetAnnotationsRequest:",
		"V2BisqueDatasetAnnotationsResponse:",
		"V2BisqueAnnotatedImage:",
		"V2BisqueNotConfiguredResponse:",
		"Number of primitive graphical shape elements",
		"fixed 8,000-image safety bound",
		"concurrency bounded",
	} {
		if !strings.Contains(document, required) {
			t.Errorf("OpenAPI BisQue count contract missing %q", required)
		}
	}
}

func TestGeneratedBisqueCountContractsRoundTrip(t *testing.T) {
	t.Parallel()
	dataset := "00-dataset"
	image := "00-image"
	limit := 25
	offset := 2
	maxImages := 100

	memberRequest := V2BisqueDatasetMembersRequest{
		DatasetUniq: &dataset,
		Limit:       &limit,
		Offset:      &offset,
	}
	memberPayload, err := json.Marshal(memberRequest)
	if err != nil {
		t.Fatalf("marshal generated member request: %v", err)
	}
	var decodedMember V2BisqueDatasetMembersRequest
	if err := json.Unmarshal(memberPayload, &decodedMember); err != nil {
		t.Fatalf("unmarshal generated member request: %v", err)
	}
	if decodedMember.DatasetUniq == nil || *decodedMember.DatasetUniq != dataset ||
		decodedMember.Limit == nil || *decodedMember.Limit != limit ||
		decodedMember.Offset == nil || *decodedMember.Offset != offset {
		t.Fatalf("generated member request round trip = %#v", decodedMember)
	}

	imageRequest := V2BisqueImageAnnotationsRequest{ImageUniq: &image}
	if _, err := json.Marshal(imageRequest); err != nil {
		t.Fatalf("marshal generated image annotation request: %v", err)
	}
	datasetRequest := V2BisqueDatasetAnnotationsRequest{
		DatasetUniq: &dataset,
		MaxImages:   &maxImages,
	}
	if _, err := json.Marshal(datasetRequest); err != nil {
		t.Fatalf("marshal generated dataset annotation request: %v", err)
	}

	labelCounts := map[string]int{"precipitate": 3}
	memberResponse := V2BisqueDatasetMembersResponse{
		DatasetUniq: dataset,
		MemberCount: 1,
		Offset:      0,
		Members: []V2BisqueResource{{
			ResourceUri:  "https://bisque.example/data_service/image/" + image,
			ResourceUniq: &image,
		}},
	}
	imageResponse := V2BisqueImageAnnotationsResponse{
		ImageUniq:       image,
		AnnotationCount: 3,
		LabelCounts:     &labelCounts,
	}
	datasetResponse := V2BisqueDatasetAnnotationsResponse{
		DatasetUniq:           dataset,
		MemberCount:           1,
		ImagesChecked:         1,
		ImagesWithAnnotations: 1,
		TotalAnnotations:      3,
		Inaccessible:          0,
		Truncated:             false,
		AnnotatedImages: []V2BisqueAnnotatedImage{{
			ResourceUniq:    image,
			AnnotationCount: 3,
			LabelCounts:     &labelCounts,
		}},
	}
	for name, response := range map[string]any{
		"members":             memberResponse,
		"image annotations":   imageResponse,
		"dataset annotations": datasetResponse,
	} {
		if _, err := json.Marshal(response); err != nil {
			t.Fatalf("marshal generated %s response: %v", name, err)
		}
	}

	runID := ConditionalUltraWorkerRunID("run-1")
	workerID := ConditionalUltraWorkerID("worker-1")
	leaseToken := ConditionalUltraRunLeaseToken("lease-secret")
	params := GetBisqueDatasetMembersParams{
		XUltraRunId:         &runID,
		XUltraWorkerId:      &workerID,
		XUltraRunLeaseToken: &leaseToken,
	}
	if params.XUltraRunId == nil || params.XUltraWorkerId == nil || params.XUltraRunLeaseToken == nil {
		t.Fatalf("generated worker authority parameters = %#v", params)
	}
}

func openAPIPathSection(t *testing.T, document string, path string) string {
	t.Helper()
	marker := "  " + path + ":\n"
	start := strings.Index(document, marker)
	if start < 0 {
		t.Fatalf("OpenAPI document missing %s", path)
	}
	tail := document[start:]
	if next := strings.Index(tail[len(marker):], "\n  /v"); next >= 0 {
		return tail[:len(marker)+next]
	}
	return tail
}
