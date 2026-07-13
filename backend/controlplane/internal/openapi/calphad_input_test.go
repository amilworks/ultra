package openapi

import (
	"os"
	"strings"
	"testing"
	"time"
)

func TestOpenAPICalphadRevisionInputIsOwnerReplayContract(t *testing.T) {
	t.Parallel()
	document, err := os.ReadFile("../../api/openapi.yaml")
	if err != nil {
		t.Fatalf("read openapi.yaml: %v", err)
	}
	for _, required := range []string{
		"/v2/resources/{file_id}/calphad/revision/input:",
		"operationId: getCalphadRevisionInput",
		"application/octet-stream:",
		"Thermo-Calc TDB or ChemSage DAT",
		"X-Ultra-Calphad-Revision-Id:",
		"X-Ultra-Content-Sha256:",
		"X-Ultra-Calphad-Database-Format:",
		"Content-Disposition:",
		"Content-Length:",
		"ETag:",
		"database_format:",
		"ultra.calphad.owner-declaration.v1",
		"ultra.calphad.retained-evidence.v2",
		"garbage collection",
		`"409":`,
		"assessment_pressure_limits_Pa:",
		"equal fixed bounds are valid",
	} {
		if !strings.Contains(string(document), required) {
			t.Errorf("CALPHAD input replay contract missing %q", required)
		}
	}
	revisionID := "calphad_revision_1"
	sha := strings.Repeat("a", 64)
	databaseFormat := "tdb"
	contentDisposition := `attachment; filename="` + sha + `.tdb"`
	headers := GetCalphadRevisionInput200ResponseHeaders{
		ContentDisposition:          &contentDisposition,
		XUltraCalphadDatabaseFormat: &databaseFormat,
		XUltraCalphadRevisionId:     &revisionID,
		XUltraContentSha256:         &sha,
	}
	if headers.XUltraCalphadRevisionId == nil || headers.XUltraContentSha256 == nil ||
		headers.XUltraCalphadDatabaseFormat == nil || headers.ContentDisposition == nil {
		t.Fatalf("generated CALPHAD replay headers = %#v", headers)
	}
	pressure := []float64{101325, 101325}
	revision := V2CalphadRevisionRecord{
		RevisionId: "revision", ResourceId: "resource", OwnerUserId: "owner",
		Sha256: sha, SizeBytes: 1, DatabaseFormat: V2CalphadRevisionRecordDatabaseFormatTdb,
		AssessmentPressureLimitsPa: pressure,
		CreatedAt:                  time.Now(), Metadata: JsonObject{},
	}
	validation := V2CalphadValidationRecord{
		ValidationId: "validation", RevisionId: revision.RevisionId, ResourceId: revision.ResourceId,
		DatabaseSha256: sha, DatabaseSizeBytes: 1,
		DatabaseFormat:             V2CalphadValidationRecordDatabaseFormatTdb,
		AssessmentPressureLimitsPa: pressure,
		Status:                     "pending", Operation: "registration", EvidenceRetention: "not_applicable",
		Promotable: false, CreatedByAuthority: "control_plane", CreatedAt: time.Now(), Metadata: JsonObject{},
	}
	if len(revision.AssessmentPressureLimitsPa) != 2 || len(validation.AssessmentPressureLimitsPa) != 2 ||
		revision.DatabaseFormat != V2CalphadRevisionRecordDatabaseFormatTdb ||
		validation.DatabaseFormat != V2CalphadValidationRecordDatabaseFormatTdb {
		t.Fatalf("generated CALPHAD pressure contracts revision=%#v validation=%#v", revision, validation)
	}
}
