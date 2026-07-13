package httpapi

import (
	"reflect"
	"testing"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

func TestCalphadTDBUploadClassification(t *testing.T) {
	if got := contentTypeForUpload("Co-Al-W.TDB", "application/octet-stream"); got != "application/x-thermocalc-tdb" {
		t.Fatalf("content type = %q", got)
	}
	if got := resourceKindForContent("Co-Al-W.tdb", "application/x-thermocalc-tdb"); got != "document" {
		t.Fatalf("resource kind = %q", got)
	}
	if !isTextDocumentUpload("Co-Al-W.tdb", "application/octet-stream") {
		t.Fatal("TDB should be available to the bounded text-document path")
	}
	if isCalphadTDBUpload("notes.txt", "text/plain") {
		t.Fatal("ordinary text must not be classified as a thermodynamic database")
	}
}

func TestCalphadCatalogMetadataStartsFailClosed(t *testing.T) {
	record := resourceRecord{
		OriginalName: "Co-Al-W.tdb",
		ContentType:  "application/x-thermocalc-tdb",
		ResourceKind: "document",
		SHA256:       "aabbcc",
		SizeBytes:    21833,
	}
	metadata := uploadCatalogMetadataForPath("/not-opened/calculation.tdb", record)
	calphad, ok := metadata["calphad"].(domain.JSONMap)
	if !ok {
		t.Fatalf("calphad metadata = %#v", metadata["calphad"])
	}
	if calphad["validation_status"] != "pending_pycalphad_validation" {
		t.Fatalf("validation_status = %#v", calphad["validation_status"])
	}
	if calphad["scientific_status"] != "unverified" {
		t.Fatalf("scientific_status = %#v", calphad["scientific_status"])
	}
	if calphad["content_sha256"] != record.SHA256 {
		t.Fatalf("content_sha256 = %#v", calphad["content_sha256"])
	}
	wantRequired := []string{
		"source",
		"license_id",
		"assessment_scope",
		"reference_state",
		"tdb_temperature_limits_K",
		domain.CalphadAssessmentPressureLimitsMetadataKey,
	}
	if got := calphad["required_provenance_fields"]; !reflect.DeepEqual(got, wantRequired) {
		t.Fatalf("required provenance = %#v", got)
	}
	if _, exists := calphad["verified"]; exists {
		t.Fatal("catalog classification must not mint a scientific verification claim")
	}
}
