package integration

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/httpapi"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

// TestCalphadTypedCLIArtifactsPassRealHTTPVerifier is the database-independent
// half of the cross-language contract. The qualification gate still requires
// TestCalphadTypedCLIHTTPPostgresQualification; this focused test exists so real
// Python artifacts can diagnose schema/canonicalization failures even when a
// local disposable PostgreSQL service is unavailable.
func TestCalphadTypedCLIArtifactsPassRealHTTPVerifier(t *testing.T) {
	databaseInputPath := strings.TrimSpace(os.Getenv("ULTRA_CALPHAD_DATABASE_INPUT_ARTIFACT"))
	inspectionPath := strings.TrimSpace(os.Getenv("ULTRA_CALPHAD_INSPECTION_ARTIFACT"))
	equilibriumPath := strings.TrimSpace(os.Getenv("ULTRA_CALPHAD_EQUILIBRIUM_ARTIFACT"))
	expectedRuntimeImage := strings.ToLower(strings.TrimSpace(os.Getenv("ULTRA_CALPHAD_RUNTIME_IMAGE_ID")))
	if databaseInputPath == "" || inspectionPath == "" || equilibriumPath == "" || expectedRuntimeImage == "" {
		t.Skip("real typed CLI artifact paths and runtime image identity are required")
	}
	if !immutableImagePattern.MatchString(expectedRuntimeImage) {
		t.Fatalf("ULTRA_CALPHAD_RUNTIME_IMAGE_ID must be immutable, got %q", expectedRuntimeImage)
	}

	inspection := loadCalphadArtifact(t, inspectionPath, "inspect")
	equilibrium := loadCalphadArtifact(t, equilibriumPath, "equilibrium")
	if !reflect.DeepEqual(inspection.Evidence.DatabaseBinding, equilibrium.Evidence.DatabaseBinding) {
		t.Fatal("inspect and equilibrium artifacts use different database bindings")
	}
	binding := inspection.Evidence.DatabaseBinding
	if len(binding.AssessmentPressureLimitsPa) != 2 ||
		binding.AssessmentPressureLimitsPa[0] != domain.CalphadReferencePressurePa ||
		binding.AssessmentPressureLimitsPa[1] != domain.CalphadReferencePressurePa {
		t.Fatalf("real typed artifacts do not retain the fixed 101325 Pa assessment binding: %+v", binding)
	}
	if inspection.Evidence.Request.RuntimeImageID != expectedRuntimeImage ||
		equilibrium.Evidence.Request.RuntimeImageID != expectedRuntimeImage ||
		equilibrium.Evidence.Request.InspectionArtifactSHA256 != inspection.SHA256 {
		t.Fatal("typed CLI artifacts do not bind the runtime and exact inspection lineage")
	}
	databaseInput := loadCalphadDatabaseInput(
		t, databaseInputPath, binding.SHA256, binding.DatabaseFormat, binding.SizeBytes,
	)

	ctx := context.Background()
	memory := store.NewMemoryStore()
	now := domain.Now()
	owner := "calphad-cross-language-memory-owner"
	ownerOrg := "calphad-cross-language-memory-org"
	workerID := "calphad-cross-language-memory-worker"
	workerToken := "calphad-cross-language-memory-token"
	uploadRoot := t.TempDir()
	storageDirectory := filepath.Join(uploadRoot, "qualification-input")
	if err := os.Mkdir(storageDirectory, 0o700); err != nil {
		t.Fatalf("create qualification input directory: %v", err)
	}
	storageName := binding.SHA256 + "." + binding.DatabaseFormat
	storageRelativePath := filepath.Join("qualification-input", storageName)
	if err := os.WriteFile(filepath.Join(storageDirectory, storageName), databaseInput, 0o600); err != nil {
		t.Fatalf("stage exact CALPHAD database input: %v", err)
	}
	contentType := "application/x-thermocalc-tdb"
	if binding.DatabaseFormat == domain.CalphadDatabaseFormatDAT {
		contentType = "application/octet-stream"
	}
	if _, err := memory.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID: binding.ResourceID, OriginalName: binding.DatabaseID + "." + binding.DatabaseFormat,
		ContentType: contentType, SizeBytes: binding.SizeBytes,
		SHA256: binding.SHA256, SourceType: "upload", ResourceKind: "document",
		StoragePath: filepath.ToSlash(storageRelativePath),
		OwnerUserID: owner, OwnerOrgID: ownerOrg, OwnerRole: "researcher", Status: "active",
		CreatedAt: now, UpdatedAt: now,
		Metadata: domain.JSONMap{"calphad": domain.JSONMap{
			"validation_status": "owner_claimed_validated",
			"database_id":       binding.DatabaseID, "source": binding.Source,
			"license_id": binding.LicenseID, "assessment_scope": binding.AssessmentScope,
			"reference_state":                                 binding.ReferenceState,
			"assessment_temperature_limits_K":                 binding.TemperatureLimitsK,
			domain.CalphadAssessmentPressureLimitsMetadataKey: binding.AssessmentPressureLimitsPa,
		}},
	}); err != nil {
		t.Fatalf("seed CALPHAD memory resource: %v", err)
	}
	thread, err := memory.CreateThread(ctx, domain.CreateThreadInput{
		UserID: owner, Title: "Cross-language CALPHAD HTTP contract",
	})
	if err != nil {
		t.Fatalf("create memory thread: %v", err)
	}
	descriptor := domain.JSONMap{
		"type": "selected_resource", "authority": "control_resource_catalog",
		"binding_schema": "ultra.selected_resource.v1", "resource_id": binding.ResourceID,
		"file_id": binding.ResourceID, "sha256": binding.SHA256, "size_bytes": binding.SizeBytes,
		"original_name":            binding.DatabaseID + "." + binding.DatabaseFormat,
		"database_format":          binding.DatabaseFormat,
		"calphad_governance_scope": "owner_validation",
		"metadata": domain.JSONMap{"calphad": domain.JSONMap{
			"declaration_authority": "resource_owner", "database_id": binding.DatabaseID,
			"source": binding.Source, "license_id": binding.LicenseID,
			"assessment_scope": binding.AssessmentScope, "reference_state": binding.ReferenceState,
			"assessment_temperature_limits_K":                 binding.TemperatureLimitsK,
			domain.CalphadAssessmentPressureLimitsMetadataKey: binding.AssessmentPressureLimitsPa,
		}},
	}
	run, err := memory.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID, UserID: owner, Goal: "verify real typed CALPHAD artifacts",
		Metadata: domain.JSONMap{
			"org_id": ownerOrg, "file_ids": []string{binding.ResourceID},
			"resource_descriptors": []domain.JSONMap{descriptor},
			domain.CalphadRuntimePolicyMetadataKey: domain.JSONMap{
				"schema_version": domain.CalphadRuntimePolicySchema, "authority": "control_plane",
				"runtime_image_id": expectedRuntimeImage, "pycalphad_version": domain.CalphadPycalphadVersion,
				"network": domain.CalphadRuntimeNetwork, "no_new_privileges": true,
				"read_only_root_filesystem": true, "cap_drop_all": true,
				"cpus_at_most":         domain.CalphadRuntimeCPUsAtMost,
				"memory_bytes_at_most": domain.CalphadRuntimeMemoryBytesAtMost,
				"pids_at_most":         domain.CalphadRuntimePIDsAtMost,
			},
			"principal": domain.JSONMap{
				"user_id": owner, "org_id": ownerOrg, "role": "researcher",
			},
		},
	})
	if err != nil {
		t.Fatalf("create memory run: %v", err)
	}
	lease, err := memory.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID: run.RunID, WorkerID: workerID, TTL: time.Hour, Now: now,
	})
	if err != nil {
		t.Fatalf("acquire memory run lease: %v", err)
	}
	router := httpapi.NewRouter(httpapi.ServerDeps{
		Version: "calphad-cross-language-http-contract", Store: memory,
		WorkerToken: workerToken, UploadRoot: uploadRoot,
	})
	ownerHeaders := map[string]string{
		"X-Ultra-User-Id": owner, "X-Ultra-Org-Id": ownerOrg,
	}
	workerHeaders := map[string]string{
		"X-Ultra-Worker-Token": workerToken, "X-Ultra-Run-Id": run.RunID,
		"X-Ultra-Worker-Id": workerID, "X-Ultra-Run-Lease-Token": lease.LeaseToken,
	}
	resourcePath := url.PathEscape(binding.ResourceID)
	created := doRequest(t, router, http.MethodPost,
		"/v2/resources/"+resourcePath+"/calphad/revision", `{}`, ownerHeaders)
	if created.Code != http.StatusCreated {
		t.Fatalf("create memory revision status=%d body=%s", created.Code, created.Body.String())
	}
	callbackPath := "/v2/runs/" + url.PathEscape(run.RunID) + "/resources/" + resourcePath + "/calphad/validations"
	inspectResponse := doRequest(t, router, http.MethodPost, callbackPath,
		callbackBody(t, inspection, "input_validated"), workerHeaders)
	if inspectResponse.Code != http.StatusCreated {
		t.Fatalf("real inspect callback status=%d body=%s", inspectResponse.Code, inspectResponse.Body.String())
	}
	equilibriumResponse := doRequest(t, router, http.MethodPost, callbackPath,
		callbackBody(t, equilibrium, "equilibrium_completed"), workerHeaders)
	if equilibriumResponse.Code != http.StatusCreated {
		t.Fatalf("real equilibrium callback status=%d body=%s", equilibriumResponse.Code, equilibriumResponse.Body.String())
	}
	ledgerResponse := doRequest(t, router, http.MethodGet,
		"/v2/resources/"+resourcePath+"/calphad/ledger", "", ownerHeaders)
	if ledgerResponse.Code != http.StatusOK {
		t.Fatalf("read memory ledger status=%d body=%s", ledgerResponse.Code, ledgerResponse.Body.String())
	}
	var envelope struct {
		Ledger domain.CalphadLedgerRecord `json:"ledger"`
	}
	if err := json.Unmarshal(ledgerResponse.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("decode memory CALPHAD ledger: %v", err)
	}
	latest := envelope.Ledger.LatestValidation
	if latest == nil || latest.Operation != "equilibrium" ||
		latest.InspectionEvidenceSHA256 != inspection.SHA256 ||
		latest.EvidenceRetention != domain.CalphadEvidenceRetentionRetained || !latest.Promotable {
		t.Fatalf("memory CALPHAD ledger did not retain promotable equilibrium lineage: %+v", latest)
	}
}
