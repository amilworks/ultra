package httpapi

import (
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

func calphadPressureLimitsFixture() []float64 {
	return []float64{domain.CalphadReferencePressurePa, domain.CalphadReferencePressurePa}
}

type calphadRevisionInputOverrideStore struct {
	*store.MemoryStore
	input domain.CalphadRevisionInputRecord
	err   error
}

func (s *calphadRevisionInputOverrideStore) GetCalphadRevisionInputForOwner(
	context.Context, string, string, string,
) (domain.CalphadRevisionInputRecord, error) {
	return s.input, s.err
}

func calphadInspectionResult(
	resourceID string,
	databaseSHA string,
	databaseSize int64,
	marker string,
	requestedComponents, requestedPhases []string,
) map[string]any {
	manifest := map[string]any{
		"schema_version":        "1",
		"path":                  "/workspace/.ultra/calphad/staged/" + databaseSHA + ".tdb",
		"name":                  databaseSHA + ".tdb",
		"sha256":                databaseSHA,
		"size_bytes":            databaseSize,
		"format":                "tdb",
		"package_test_database": false,
		"source":                "Owner assessment DOI 10.0000/example",
		"license_id":            "CC-BY-4.0",
		"artifact_id":           resourceID,
		"assessment_scope":      "Assessed binary equilibrium",
		"reference_state":       "SER",
		"requested_components":  requestedComponents,
		"requested_phases":      requestedPhases,
		"components":            []string{"AL", "NI", "VA"},
		"physical_elements":     []string{"AL", "NI"},
		"vacancy_components":    []string{"VA"},
		"pseudo_elements":       []string{},
		"species":               []string{"AL", "NI", "VA"},
		"phases":                []string{"FCC_A1"},
		"available_components":  []string{"AL", "NI", "VA"},
		"available_phases":      []string{"FCC_A1"},
		"phase_models": []any{
			map[string]any{
				"name":                   "FCC_A1",
				"sublattice_site_ratios": []float64{1, 1},
				"sublattices": []any{
					map[string]any{
						"index": 0, "site_ratio": 1,
						"constituents": []string{"AL", "NI", "VA"},
					},
					map[string]any{
						"index": 1, "site_ratio": 1,
						"constituents": []string{"AL", "NI", "VA"},
					},
				},
				"model_hints": map[string]string{},
			},
		},
		"parameter_count": 1,
		"references": map[string]any{
			"count": 0, "included_count": 0, "truncated": false, "entries": []any{},
		},
		"pycalphad_version":                               "0.11.2",
		"registry_manifest":                               nil,
		"assessment_temperature_limits_K":                 []float64{300, 2000},
		domain.CalphadAssessmentPressureLimitsMetadataKey: calphadPressureLimitsFixture(),
		"warnings": []string{"fixture evidence " + marker},
		"limits": map[string]any{
			"max_database_bytes":               64 * 1024 * 1024,
			"max_elements":                     256,
			"max_species":                      4096,
			"max_phases":                       2048,
			"max_parameters":                   1_000_000,
			"database_parse_wall_time_seconds": 15,
		},
	}
	canonical, err := calphadCanonicalJSON(manifest)
	if err != nil {
		panic(err)
	}
	digest := sha256.Sum256(canonical)
	manifest["manifest_sha256"] = fmt.Sprintf("%x", digest[:])
	return manifest
}

func calphadInspectionEvidence(
	resourceID, databaseSHA string,
	databaseSize int64,
	runtimeImageID, marker string,
) map[string]any {
	return map[string]any{
		"schema_version": calphadToolEvidenceSchemaVersion,
		"operation":      "inspect",
		"database_binding": map[string]any{
			"kind": "resource", "database_id": resourceID, "resource_id": resourceID,
			"sha256": databaseSHA, "size_bytes": databaseSize, "database_format": "tdb",
			"source": "Owner assessment DOI 10.0000/example", "license_id": "CC-BY-4.0",
			"assessment_scope": "Assessed binary equilibrium", "reference_state": "SER",
			"temperature_limits_K":                            []float64{300, 2000},
			domain.CalphadAssessmentPressureLimitsMetadataKey: calphadPressureLimitsFixture(),
			"binding_schema":                                  "ultra.selected_resource.v1",
			"binding_authority":                               "control_resource_catalog",
			"declaration_authority":                           "resource_owner",
		},
		"request": map[string]any{
			"operation": "inspect", "runtime_image_id": runtimeImageID,
			"selection": map[string]any{"components": nil, "phases": nil},
		},
		"result": calphadInspectionResult(
			resourceID, databaseSHA, databaseSize, marker,
			[]string{"AL", "NI", "VA"}, []string{"FCC_A1"},
		),
		"execution_contract": map[string]any{
			"interface":            "fixed ultra_deepagents.materials.calphad public surface",
			"caller_code_accepted": false, "caller_models_or_solver_options_accepted": false,
			"network": "none", "no_new_privileges": true, "read_only_root_filesystem": true,
			"cap_drop_all": true, "cpus_at_most": 8.0,
			"memory_bytes_at_most": 32 * 1024 * 1024 * 1024, "pids_at_most": 4096,
			"runtime_image_id": runtimeImageID,
			"max_components":   32, "max_phases": 128, "max_axis_values": 64,
			"max_grid_points": 256, "wall_time_seconds": 30, "max_result_bytes": 16 * 1024 * 1024,
		},
		"validation_persistence": map[string]any{
			"catalog_status": "pending", "catalog_metadata_updated": false,
			"mode": "immutable_per_run_evidence", "note": "server callback pending",
		},
	}
}

func calphadFailureEvidence(
	resourceID, databaseSHA string,
	databaseSize int64,
	runtimeImageID, operation, status, failureDomain, failureStage, failureCode string,
	exitCode any,
	solverStarted bool,
) map[string]any {
	evidence := calphadInspectionEvidence(
		resourceID, databaseSHA, databaseSize, runtimeImageID, "terminal-failure",
	)
	evidence["schema_version"] = domain.CalphadFailureEvidenceSchemaVersion
	evidence["operation"] = operation
	delete(evidence, "result")
	if operation == "equilibrium" {
		evidence["request"] = map[string]any{
			"operation": "equilibrium", "runtime_image_id": runtimeImageID,
			"selection": map[string]any{
				"components": []string{"AL", "NI", "VA"}, "phases": []string{"FCC_A1"},
			},
			"inspection_artifact_sha256": strings.Repeat("5", 64),
			"conditions": map[string]any{
				"temperatures_K": []float64{900}, "pressures_Pa": []float64{101325},
				"independent_compositions": map[string]any{"AL": []float64{0.25}},
			},
		}
	}
	evidence["outcome"] = map[string]any{
		"status": status, "failure_domain": failureDomain, "failure_stage": failureStage,
		"failure_code": failureCode, "exit_code": exitCode, "solver_started": solverStarted,
	}
	return evidence
}

func calphadGzipBytes(t *testing.T, raw []byte) []byte {
	t.Helper()
	var compressed bytes.Buffer
	writer, err := gzip.NewWriterLevel(&compressed, gzip.BestCompression)
	if err != nil {
		t.Fatalf("gzip.NewWriterLevel: %v", err)
	}
	if _, err := writer.Write(raw); err != nil {
		t.Fatalf("gzip write: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("gzip close: %v", err)
	}
	return compressed.Bytes()
}

func calphadValidationBodyForRaw(
	t *testing.T,
	raw []byte,
	operation, status, runtimeImageID, pycalphadVersion string,
	compressedSuffix []byte,
) string {
	t.Helper()
	digest := sha256.Sum256(raw)
	digestText := fmt.Sprintf("%x", digest[:])
	directory := operation
	if operation == "inspect" {
		directory = "inspection"
	}
	compressed := append(calphadGzipBytes(t, raw), compressedSuffix...)
	body, err := json.Marshal(map[string]any{
		"status": status, "operation": operation,
		"evidence_path":   "/outputs/calphad/" + directory + "/" + digestText + ".json",
		"evidence_sha256": digestText, "evidence_size_bytes": len(raw),
		"runtime_image_id": runtimeImageID, "pycalphad_version": pycalphadVersion,
		"evidence_gzip_base64": base64.StdEncoding.EncodeToString(compressed),
	})
	if err != nil {
		t.Fatalf("marshal CALPHAD callback: %v", err)
	}
	return string(body)
}

func calphadFailureValidationBodyForRaw(
	t *testing.T,
	raw []byte,
	operation, status, failureDomain, failureStage, failureCode, runtimeImageID string,
) string {
	t.Helper()
	digest := sha256.Sum256(raw)
	digestText := fmt.Sprintf("%x", digest[:])
	directory := operation
	if operation == "inspect" {
		directory = "inspection"
	}
	body, err := json.Marshal(map[string]any{
		"status": status, "operation": operation,
		"failure_domain": failureDomain, "failure_stage": failureStage,
		"failure_code":    failureCode,
		"evidence_path":   "/outputs/calphad/" + directory + "/" + digestText + ".json",
		"evidence_sha256": digestText, "evidence_size_bytes": len(raw),
		"runtime_image_id": runtimeImageID, "pycalphad_version": "0.11.2",
		"evidence_gzip_base64": base64.StdEncoding.EncodeToString(calphadGzipBytes(t, raw)),
	})
	if err != nil {
		t.Fatalf("marshal CALPHAD failure callback: %v", err)
	}
	return string(body)
}

func calphadInspectionValidationBody(
	t *testing.T,
	resourceID, databaseSHA string,
	databaseSize int64,
	runtimeImageID, marker string,
) string {
	t.Helper()
	evidence := calphadInspectionEvidence(
		resourceID, databaseSHA, databaseSize, runtimeImageID, marker,
	)
	raw, err := json.Marshal(evidence)
	if err != nil {
		t.Fatalf("marshal CALPHAD evidence: %v", err)
	}
	return calphadValidationBodyForRaw(
		t, raw, "inspect", "input_validated", runtimeImageID, "0.11.2", nil,
	)
}

func selectedCalphadDescriptor(resourceID, sha string, size int64) domain.JSONMap {
	return domain.JSONMap{
		"type": "selected_resource", "authority": "control_resource_catalog",
		"binding_schema": "ultra.selected_resource.v1", "resource_id": resourceID,
		"file_id": resourceID, "sha256": sha, "size_bytes": size,
		"original_name": resourceID + ".tdb", "database_format": "tdb",
		"calphad_governance_scope": "owner_validation",
		"metadata": domain.JSONMap{
			"calphad": domain.JSONMap{
				"database_id": resourceID, "source": "Owner assessment DOI 10.0000/example",
				"license_id": "CC-BY-4.0", "assessment_scope": "Assessed binary equilibrium",
				"reference_state": "SER", "tdb_temperature_limits_K": []float64{300, 2000},
				domain.CalphadAssessmentPressureLimitsMetadataKey: calphadPressureLimitsFixture(),
				"declaration_authority":                           "resource_owner",
			},
		},
	}
}

func calphadRunPolicy(runtimeImageID string) domain.JSONMap {
	return domain.JSONMap{
		"schema_version": domain.CalphadRuntimePolicySchema, "authority": "control_plane",
		"runtime_image_id": runtimeImageID, "pycalphad_version": domain.CalphadPycalphadVersion,
		"network": domain.CalphadRuntimeNetwork, "no_new_privileges": true,
		"read_only_root_filesystem": true, "cap_drop_all": true,
		"cpus_at_most":         domain.CalphadRuntimeCPUsAtMost,
		"memory_bytes_at_most": domain.CalphadRuntimeMemoryBytesAtMost,
		"pids_at_most":         domain.CalphadRuntimePIDsAtMost,
	}
}

func TestRunSelectedCalphadBindingPinsOwnerDeclarations(t *testing.T) {
	t.Parallel()
	const resourceID = "calphad-provenance-resource"
	databaseSHA := strings.Repeat("a", 64)
	evidence := verifiedCalphadEvidence{
		ResourceID: resourceID, DatabaseID: resourceID,
		DatabaseSHA256: databaseSHA, DatabaseSizeBytes: 512, DatabaseFormat: "tdb",
		Source: "Owner assessment DOI 10.0000/example", LicenseID: "CC-BY-4.0",
		AssessmentScope: "Assessed binary equilibrium", ReferenceState: "SER",
		TemperatureLimitsK: [2]float64{300, 2000},
		AssessmentPressureLimitsPa: [2]float64{
			domain.CalphadReferencePressurePa, domain.CalphadReferencePressurePa,
		},
	}
	makeRun := func() domain.RunRecord {
		return domain.RunRecord{Metadata: domain.JSONMap{
			"file_ids": []string{resourceID},
			"resource_descriptors": []domain.JSONMap{
				selectedCalphadDescriptor(resourceID, databaseSHA, 512),
			},
		}}
	}
	if !runHasSelectedCalphadBinding(makeRun(), evidence) {
		t.Fatal("exact selected descriptor was not accepted")
	}
	mutations := map[string]func(domain.JSONMap){
		"database_id": func(calphad domain.JSONMap) { calphad["database_id"] = "relabelled" },
		"source":      func(calphad domain.JSONMap) { calphad["source"] = "different source" },
		"license":     func(calphad domain.JSONMap) { calphad["license_id"] = "MIT" },
		"scope":       func(calphad domain.JSONMap) { calphad["assessment_scope"] = "different scope" },
		"reference":   func(calphad domain.JSONMap) { calphad["reference_state"] = "different state" },
		"temperature": func(calphad domain.JSONMap) {
			calphad["tdb_temperature_limits_K"] = []float64{400, 1800}
		},
		"pressure": func(calphad domain.JSONMap) {
			calphad[domain.CalphadAssessmentPressureLimitsMetadataKey] = []float64{100000, 200000}
		},
		"authority": func(calphad domain.JSONMap) { calphad["declaration_authority"] = "worker" },
	}
	for name, mutate := range mutations {
		name, mutate := name, mutate
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			run := makeRun()
			descriptors := run.Metadata["resource_descriptors"].([]domain.JSONMap)
			metadata := descriptors[0]["metadata"].(domain.JSONMap)
			calphad := metadata["calphad"].(domain.JSONMap)
			mutate(calphad)
			if runHasSelectedCalphadBinding(run, evidence) {
				t.Fatal("descriptor with relabelled owner declaration was accepted")
			}
		})
	}
	runWithoutMetadata := makeRun()
	descriptors := runWithoutMetadata.Metadata["resource_descriptors"].([]domain.JSONMap)
	delete(descriptors[0], "metadata")
	if runHasSelectedCalphadBinding(runWithoutMetadata, evidence) {
		t.Fatal("descriptor without owner declarations was accepted")
	}
	for name, mutate := range map[string]func(*domain.RunRecord){
		"missing owner-validation scope": func(run *domain.RunRecord) {
			delete(run.Metadata["resource_descriptors"].([]domain.JSONMap)[0], "calphad_governance_scope")
		},
		"duplicate selected file": func(run *domain.RunRecord) {
			run.Metadata["file_ids"] = []string{resourceID, resourceID}
		},
		"duplicate exact descriptor": func(run *domain.RunRecord) {
			descriptor := selectedCalphadDescriptor(resourceID, databaseSHA, 512)
			run.Metadata["resource_descriptors"] = append(
				run.Metadata["resource_descriptors"].([]domain.JSONMap), descriptor,
			)
		},
		"ambiguous malformed candidate": func(run *domain.RunRecord) {
			run.Metadata["resource_descriptors"] = append(
				run.Metadata["resource_descriptors"].([]domain.JSONMap),
				domain.JSONMap{"resource_id": resourceID, "file_id": "other"},
			)
		},
	} {
		name, mutate := name, mutate
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			run := makeRun()
			mutate(&run)
			if runHasSelectedCalphadBinding(run, evidence) {
				t.Fatal("ambiguous or governance-incomplete descriptor authority was accepted")
			}
		})
	}
}

func TestNormalizedCalphadAssessmentPressureLimitsRejectsMalformedDeclarations(t *testing.T) {
	t.Parallel()
	valid, ok := normalizedCalphadAssessmentPressureLimits(
		[]float64{domain.CalphadReferencePressurePa, domain.CalphadReferencePressurePa},
	)
	if !ok || valid != [2]float64{domain.CalphadReferencePressurePa, domain.CalphadReferencePressurePa} {
		t.Fatalf("fixed pressure declaration = %v ok=%t", valid, ok)
	}
	for name, value := range map[string]any{
		"missing":      nil,
		"one value":    []float64{101325},
		"boolean":      []any{true, 101325.0},
		"string":       []any{"101325", 101325.0},
		"nan":          []float64{math.NaN(), 101325},
		"infinity":     []float64{101325, math.Inf(1)},
		"reversed":     []float64{101326, 101325},
		"below global": []float64{0, 101325},
		"above global": []float64{101325, domain.CalphadMaximumPressurePa + 1},
	} {
		name, value := name, value
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			if limits, accepted := normalizedCalphadAssessmentPressureLimits(value); accepted {
				t.Fatalf("malformed declaration accepted as %v", limits)
			}
		})
	}
}

func TestCalphadChemSageDATReplayUsesFormatNeutralMediaType(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	memory := store.NewMemoryStore()
	payload := []byte("$ ChemSage database\nELEMENT AL FCC_A1 26.9815\n")
	digest := sha256.Sum256(payload)
	sha := fmt.Sprintf("%x", digest[:])
	const (
		resourceID = "calphad-chemsage-dat"
		owner      = "calphad-dat-owner"
		org        = "calphad-dat-org"
	)
	if _, err := memory.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID: resourceID, OriginalName: "assessment.dat", ContentType: "application/octet-stream",
		SizeBytes: int64(len(payload)), SHA256: sha, OwnerUserID: owner, OwnerOrgID: org,
		Status: "active", CreatedAt: domain.Now(), UpdatedAt: domain.Now(),
		Metadata: domain.JSONMap{"calphad": domain.JSONMap{
			"source": "https://example.org/assessment.dat", "license_id": "CC-BY-4.0",
			"assessment_scope": "Assessed binary equilibrium", "reference_state": "SER",
			"tdb_temperature_limits_K":                        []float64{300, 2000},
			domain.CalphadAssessmentPressureLimitsMetadataKey: calphadPressureLimitsFixture(),
		}},
	}); err != nil {
		t.Fatalf("UpsertResource(.dat): %v", err)
	}
	revision, err := memory.CreateCalphadRevision(ctx, domain.CreateCalphadRevisionInput{
		ResourceID: resourceID, OwnerUserID: owner, OwnerOrgID: org,
		AssessmentPressureLimitsPa: [2]float64{
			domain.CalphadReferencePressurePa, domain.CalphadReferencePressurePa,
		},
		InputBytes: payload,
	})
	if err != nil {
		t.Fatalf("CreateCalphadRevision(.dat): %v", err)
	}
	if revision.DatabaseFormat != domain.CalphadDatabaseFormatDAT {
		t.Fatalf("revision database_format=%q, want dat", revision.DatabaseFormat)
	}
	router := NewRouter(ServerDeps{Version: "test", Store: memory})
	ledgerReq := httptest.NewRequest(
		http.MethodGet, "/v2/resources/"+resourceID+"/calphad/ledger", nil,
	)
	ledgerReq.Header.Set("X-Ultra-User-Id", owner)
	ledgerReq.Header.Set("X-Ultra-Org-Id", org)
	ledgerRec := httptest.NewRecorder()
	router.ServeHTTP(ledgerRec, ledgerReq)
	var ledgerBody struct {
		Ledger domain.CalphadLedgerRecord `json:"ledger"`
	}
	if ledgerRec.Code != http.StatusOK {
		t.Fatalf("DAT ledger status=%d body=%s", ledgerRec.Code, ledgerRec.Body.String())
	}
	if err := json.Unmarshal(ledgerRec.Body.Bytes(), &ledgerBody); err != nil {
		t.Fatalf("decode DAT ledger: %v", err)
	}
	if ledgerBody.Ledger.Revision.DatabaseFormat != domain.CalphadDatabaseFormatDAT {
		t.Fatalf("ledger revision database_format=%q, want dat", ledgerBody.Ledger.Revision.DatabaseFormat)
	}
	req := httptest.NewRequest(
		http.MethodGet, "/v2/resources/"+resourceID+"/calphad/revision/input", nil,
	)
	req.Header.Set("X-Ultra-User-Id", owner)
	req.Header.Set("X-Ultra-Org-Id", org)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK || rec.Header().Get("Content-Type") != "application/octet-stream" ||
		rec.Header().Get("Content-Length") != fmt.Sprintf("%d", len(payload)) ||
		rec.Header().Get("ETag") != `"sha256:`+sha+`"` ||
		rec.Header().Get("X-Ultra-Calphad-Revision-Id") != revision.RevisionID ||
		rec.Header().Get("X-Ultra-Content-Sha256") != sha ||
		rec.Header().Get("X-Ultra-Calphad-Database-Format") != domain.CalphadDatabaseFormatDAT ||
		rec.Header().Get("Content-Disposition") != `attachment; filename="`+sha+`.dat"` ||
		!bytes.Equal(rec.Body.Bytes(), payload) {
		t.Fatalf("DAT replay status=%d headers=%v body=%q", rec.Code, rec.Header(), rec.Body.Bytes())
	}
}

func TestCalphadRevisionInputReplayRejectsMissingOrUnsupportedFormat(t *testing.T) {
	t.Parallel()
	for _, databaseFormat := range []string{"", "db"} {
		databaseFormat := databaseFormat
		t.Run(fmt.Sprintf("format_%q", databaseFormat), func(t *testing.T) {
			t.Parallel()
			payload := []byte("retained CALPHAD bytes")
			digest := sha256.Sum256(payload)
			sha := fmt.Sprintf("%x", digest[:])
			ledger := &calphadRevisionInputOverrideStore{
				MemoryStore: store.NewMemoryStore(),
				input: domain.CalphadRevisionInputRecord{
					RevisionID: "revision-with-invalid-format", ResourceID: "resource-with-invalid-format",
					SHA256: sha, SizeBytes: int64(len(payload)), DatabaseFormat: databaseFormat, Bytes: payload,
				},
			}
			router := NewRouter(ServerDeps{Version: "test", Store: ledger})
			req := httptest.NewRequest(
				http.MethodGet, "/v2/resources/resource-with-invalid-format/calphad/revision/input", nil,
			)
			req.Header.Set("X-Ultra-User-Id", "owner")
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)
			if rec.Code != http.StatusConflict {
				t.Fatalf("invalid format replay status=%d body=%s, want 409", rec.Code, rec.Body.String())
			}
			if rec.Header().Get("X-Ultra-Content-Sha256") != "" ||
				rec.Header().Get("X-Ultra-Calphad-Database-Format") != "" ||
				rec.Header().Get("Content-Disposition") != "" {
				t.Fatalf("invalid format replay leaked success headers: %v", rec.Header())
			}
			if bytes.Equal(rec.Body.Bytes(), payload) {
				t.Fatal("invalid format replay returned retained bytes")
			}
		})
	}
}

func TestCalphadGovernanceHTTPIsOwnerReadableAndWorkerWritable(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	memory := store.NewMemoryStore()
	now := time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)
	const (
		resourceID         = "calphad-http-resource"
		unselectedResource = "calphad-http-unselected"
		owner              = "calphad-http-owner"
		ownerOrg           = "calphad-http-org"
		workerKey          = "calphad-worker-secret"
		workerID           = "calphad-worker-1"
	)
	uploadRoot := t.TempDir()
	artifactRoot := t.TempDir()
	resourceBytes := bytes.Repeat([]byte("AL-NI-TDB\n"), 52)[:512]
	resourceDigest := sha256.Sum256(resourceBytes)
	resourceSHA := fmt.Sprintf("%x", resourceDigest[:])
	resourcePath := filepath.Join(uploadRoot, resourceID+"__Al-Ni.tdb")
	if err := os.WriteFile(resourcePath, resourceBytes, 0o600); err != nil {
		t.Fatalf("write retained CALPHAD input: %v", err)
	}
	if err := writeUploadMetadata(uploadRoot, resourceID, requestPrincipal{UserID: owner, OrgID: ownerOrg, Role: "researcher"}); err != nil {
		t.Fatalf("write retained CALPHAD input metadata: %v", err)
	}
	unselectedBytes := bytes.Repeat([]byte("AL-CO-TDB\n"), 26)[:256]
	unselectedDigest := sha256.Sum256(unselectedBytes)
	unselectedSHA := fmt.Sprintf("%x", unselectedDigest[:])
	unselectedSourcePath := filepath.Join(uploadRoot, unselectedResource+"__Al-Co.tdb")
	if err := os.WriteFile(unselectedSourcePath, unselectedBytes, 0o600); err != nil {
		t.Fatalf("write unselected CALPHAD input: %v", err)
	}
	if err := writeUploadMetadata(uploadRoot, unselectedResource, requestPrincipal{UserID: owner, OrgID: ownerOrg, Role: "researcher"}); err != nil {
		t.Fatalf("write unselected CALPHAD input metadata: %v", err)
	}
	runtimeImageID := "sha256:" + strings.Repeat("c", 64)
	if _, err := memory.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID: resourceID, OriginalName: "Al-Ni.tdb", ContentType: "application/x-thermocalc-tdb",
		SizeBytes: 512, SHA256: resourceSHA, SourceType: "upload", ResourceKind: "document",
		StorageURI: fileStorageURI(resourcePath), StoragePath: filepath.Base(resourcePath),
		OwnerUserID: owner, OwnerOrgID: ownerOrg, Status: "active", CreatedAt: now, UpdatedAt: now,
		Metadata: domain.JSONMap{"calphad": domain.JSONMap{
			"validation_status": "owner_claimed_validated",
			"source":            "Owner assessment DOI 10.0000/example", "license_id": "CC-BY-4.0",
			"assessment_scope": "Assessed binary equilibrium", "reference_state": "SER",
			"tdb_temperature_limits_K":                        []float64{300, 2000},
			domain.CalphadAssessmentPressureLimitsMetadataKey: calphadPressureLimitsFixture(),
		}},
	}); err != nil {
		t.Fatalf("UpsertResource: %v", err)
	}
	if _, err := memory.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID: unselectedResource, OriginalName: "Al-Co.tdb", ContentType: "application/x-thermocalc-tdb",
		SizeBytes: 256, SHA256: unselectedSHA, SourceType: "upload", ResourceKind: "document",
		StorageURI: fileStorageURI(unselectedSourcePath), StoragePath: filepath.Base(unselectedSourcePath),
		OwnerUserID: owner, OwnerOrgID: ownerOrg, Status: "active", CreatedAt: now, UpdatedAt: now,
	}); err != nil {
		t.Fatalf("UpsertResource(unselected): %v", err)
	}
	ownerThread, err := memory.CreateThread(ctx, domain.CreateThreadInput{UserID: owner, Title: "CALPHAD"})
	if err != nil {
		t.Fatalf("CreateThread(owner): %v", err)
	}
	ownerRun, err := memory.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: ownerThread.ThreadID, UserID: owner, Goal: "inspect database",
		Metadata: domain.JSONMap{
			"org_id": ownerOrg, "file_ids": []string{resourceID},
			domain.CalphadRuntimePolicyMetadataKey: calphadRunPolicy(runtimeImageID),
			"resource_descriptors": []domain.JSONMap{
				selectedCalphadDescriptor(resourceID, resourceSHA, 512),
			},
			"principal": domain.JSONMap{
				"user_id": owner, "org_id": ownerOrg, "role": "researcher",
			},
		},
	})
	if err != nil {
		t.Fatalf("CreateRun(owner): %v", err)
	}
	bobThread, err := memory.CreateThread(ctx, domain.CreateThreadInput{UserID: "bob", Title: "other"})
	if err != nil {
		t.Fatalf("CreateThread(bob): %v", err)
	}
	foreignRun, err := memory.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: bobThread.ThreadID, UserID: "bob", Goal: "cross tenant",
		Metadata: domain.JSONMap{
			"org_id": "bob-org", "file_ids": []string{resourceID},
			domain.CalphadRuntimePolicyMetadataKey: calphadRunPolicy(runtimeImageID),
			"resource_descriptors": []domain.JSONMap{
				selectedCalphadDescriptor(resourceID, resourceSHA, 512),
			},
			"principal": domain.JSONMap{
				"user_id": "bob", "org_id": "bob-org", "role": "researcher",
			},
		},
	})
	if err != nil {
		t.Fatalf("CreateRun(bob): %v", err)
	}

	ownerLease, err := memory.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID: ownerRun.RunID, WorkerID: workerID, TTL: time.Hour, Now: domain.Now(),
	})
	if err != nil {
		t.Fatalf("AcquireRunLease(owner): %v", err)
	}
	foreignLease, err := memory.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID: foreignRun.RunID, WorkerID: workerID, TTL: time.Hour, Now: domain.Now(),
	})
	if err != nil {
		t.Fatalf("AcquireRunLease(foreign): %v", err)
	}

	router := NewRouter(ServerDeps{
		Version: "test", Store: memory, WorkerToken: workerKey,
		UploadRoot: uploadRoot, ArtifactRoot: artifactRoot,
	})
	do := func(method, path, body string, headers map[string]string) *httptest.ResponseRecorder {
		t.Helper()
		req := httptest.NewRequest(method, path, strings.NewReader(body))
		if body != "" {
			req.Header.Set("Content-Type", "application/json")
		}
		for key, value := range headers {
			if value != "" {
				req.Header.Set(key, value)
			}
		}
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		return rec
	}
	userHeaders := func(userID, orgID string) map[string]string {
		return map[string]string{"X-Ultra-User-Id": userID, "X-Ultra-Org-Id": orgID}
	}
	workerHeaders := func(runID string, lease domain.RunLeaseRecord) map[string]string {
		return map[string]string{
			"X-Ultra-Worker-Token":    workerKey,
			"X-Ultra-Run-Id":          runID,
			"X-Ultra-Worker-Id":       lease.WorkerID,
			"X-Ultra-Run-Lease-Token": lease.LeaseToken,
		}
	}

	create := do(
		http.MethodPost, "/v2/resources/"+resourceID+"/calphad/revision", `{}`,
		userHeaders(owner, ownerOrg),
	)
	if create.Code != http.StatusCreated {
		t.Fatalf("owner create status=%d body=%s", create.Code, create.Body.String())
	}
	get := do(
		http.MethodGet, "/v2/resources/"+resourceID+"/calphad/ledger", "",
		userHeaders(owner, ownerOrg),
	)
	if get.Code != http.StatusOK {
		t.Fatalf("owner get status=%d body=%s", get.Code, get.Body.String())
	}
	var first struct {
		Ledger domain.CalphadLedgerRecord `json:"ledger"`
	}
	if err := json.Unmarshal(get.Body.Bytes(), &first); err != nil {
		t.Fatalf("decode initial ledger: %v", err)
	}
	if first.Ledger.LatestValidation == nil || first.Ledger.LatestValidation.Status != "pending" {
		t.Fatalf("initial latest validation=%+v, want pending", first.Ledger.LatestValidation)
	}
	if first.Ledger.Revision.DatabaseFormat != domain.CalphadDatabaseFormatTDB {
		t.Fatalf("initial revision database_format=%q, want tdb", first.Ledger.Revision.DatabaseFormat)
	}
	if first.Ledger.LatestValidation.EvidenceRetention != domain.CalphadEvidenceRetentionNotApplicable ||
		first.Ledger.LatestValidation.Promotable {
		t.Fatalf("pending validation retention=%+v", first.Ledger.LatestValidation)
	}
	input := do(
		http.MethodGet, "/v2/resources/"+resourceID+"/calphad/revision/input", "",
		userHeaders(owner, ownerOrg),
	)
	if input.Code != http.StatusOK || !bytes.Equal(input.Body.Bytes(), resourceBytes) ||
		input.Header().Get("Content-Type") != "application/octet-stream" ||
		input.Header().Get("Content-Length") != fmt.Sprintf("%d", len(resourceBytes)) ||
		input.Header().Get("ETag") != `"sha256:`+resourceSHA+`"` ||
		input.Header().Get("Cache-Control") != "private, immutable" ||
		input.Header().Get("X-Ultra-Content-Sha256") != resourceSHA ||
		input.Header().Get("X-Ultra-Calphad-Revision-Id") == "" ||
		input.Header().Get("X-Ultra-Calphad-Database-Format") != domain.CalphadDatabaseFormatTDB ||
		input.Header().Get("Content-Disposition") != `attachment; filename="`+resourceSHA+`.tdb"` {
		t.Fatalf("owner retained input status=%d headers=%v body=%q", input.Code, input.Header(), input.Body.Bytes())
	}
	if foreignInput := do(
		http.MethodGet, "/v2/resources/"+resourceID+"/calphad/revision/input", "",
		userHeaders("bob", "bob-org"),
	); foreignInput.Code != http.StatusNotFound {
		t.Fatalf("foreign retained input status=%d body=%s, want 404", foreignInput.Code, foreignInput.Body.String())
	}

	// Generic owner-writable metadata is not a validation authority.
	patch := do(http.MethodPatch, "/v2/resources/"+resourceID,
		`{"metadata":{"calphad":{"validation_status":"equilibrium_completed","verified":true}}}`,
		userHeaders(owner, ownerOrg))
	if patch.Code != http.StatusOK {
		t.Fatalf("owner metadata patch status=%d body=%s", patch.Code, patch.Body.String())
	}
	stillPending := do(
		http.MethodGet, "/v2/resources/"+resourceID+"/calphad/ledger", "",
		userHeaders(owner, ownerOrg),
	)
	var afterPatch struct {
		Ledger domain.CalphadLedgerRecord `json:"ledger"`
	}
	if err := json.Unmarshal(stillPending.Body.Bytes(), &afterPatch); err != nil {
		t.Fatalf("decode ledger after PATCH: %v", err)
	}
	if afterPatch.Ledger.LatestValidation == nil || afterPatch.Ledger.LatestValidation.Status != "pending" {
		t.Fatalf("owner PATCH changed trusted ledger: %+v", afterPatch.Ledger.LatestValidation)
	}

	if foreign := do(
		http.MethodGet, "/v2/resources/"+resourceID+"/calphad/ledger", "",
		userHeaders("bob", "bob-org"),
	); foreign.Code != http.StatusNotFound {
		t.Fatalf("foreign ledger read status=%d body=%s, want 404", foreign.Code, foreign.Body.String())
	}
	validationPath := "/v2/runs/" + ownerRun.RunID + "/resources/" + resourceID + "/calphad/validations"
	validationEvidence := calphadInspectionEvidence(
		resourceID, resourceSHA, 512, runtimeImageID, "first Δ café",
	)
	validationEvidenceBytes, err := json.MarshalIndent(validationEvidence, "", "  ")
	if err != nil {
		t.Fatalf("marshal exact callback evidence: %v", err)
	}
	validationEvidenceBytes = append(validationEvidenceBytes, '\n')
	validationEvidenceDigest := sha256.Sum256(validationEvidenceBytes)
	validationEvidenceSHA := fmt.Sprintf("%x", validationEvidenceDigest[:])
	validationBody := calphadValidationBodyForRaw(
		t, validationEvidenceBytes, "inspect", "input_validated", runtimeImageID, "0.11.2", nil,
	)
	transientArtifactPath := filepath.Join(
		artifactRoot, "outputs", "calphad", "inspection", validationEvidenceSHA+".json",
	)
	if err := os.MkdirAll(filepath.Dir(transientArtifactPath), 0o700); err != nil {
		t.Fatalf("create transient artifact directory: %v", err)
	}
	if err := os.WriteFile(transientArtifactPath, validationEvidenceBytes, 0o600); err != nil {
		t.Fatalf("write transient worker artifact: %v", err)
	}
	if unauth := do(http.MethodPost, validationPath, validationBody, nil); unauth.Code != http.StatusUnauthorized {
		t.Fatalf("unauthenticated validation status=%d body=%s, want 401", unauth.Code, unauth.Body.String())
	}
	missingRunHeader := workerHeaders(ownerRun.RunID, ownerLease)
	delete(missingRunHeader, "X-Ultra-Run-Id")
	if rec := do(http.MethodPost, validationPath, validationBody, missingRunHeader); rec.Code != http.StatusUnauthorized {
		t.Fatalf("missing run identity status=%d body=%s, want 401", rec.Code, rec.Body.String())
	}
	wrongRunHeader := workerHeaders(ownerRun.RunID, ownerLease)
	wrongRunHeader["X-Ultra-Run-Id"] = ownerRun.RunID + "-different"
	if rec := do(http.MethodPost, validationPath, validationBody, wrongRunHeader); rec.Code != http.StatusUnauthorized {
		t.Fatalf("wrong run identity status=%d body=%s, want 401", rec.Code, rec.Body.String())
	}
	wrongWorker := workerHeaders(ownerRun.RunID, ownerLease)
	wrongWorker["X-Ultra-Worker-Id"] = "different-worker"
	if rec := do(http.MethodPost, validationPath, validationBody, wrongWorker); rec.Code != http.StatusUnauthorized {
		t.Fatalf("wrong lease worker status=%d body=%s, want 401", rec.Code, rec.Body.String())
	}
	wrongLease := workerHeaders(ownerRun.RunID, ownerLease)
	wrongLease["X-Ultra-Run-Lease-Token"] = ownerLease.LeaseToken + "-forged"
	if rec := do(http.MethodPost, validationPath, validationBody, wrongLease); rec.Code != http.StatusUnauthorized {
		t.Fatalf("wrong lease token status=%d body=%s, want 401", rec.Code, rec.Body.String())
	}
	policylessRun, err := memory.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: ownerThread.ThreadID, UserID: owner, Goal: "policyless runtime",
		Metadata: domain.JSONMap{
			"org_id": ownerOrg, "file_ids": []string{resourceID},
			"resource_descriptors": []domain.JSONMap{
				selectedCalphadDescriptor(resourceID, resourceSHA, 512),
			},
		},
	})
	if err != nil {
		t.Fatalf("CreateRun(policyless): %v", err)
	}
	policylessLease, err := memory.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID: policylessRun.RunID, WorkerID: workerID, TTL: time.Hour, Now: domain.Now(),
	})
	if err != nil {
		t.Fatalf("AcquireRunLease(policyless): %v", err)
	}
	policylessPath := "/v2/runs/" + policylessRun.RunID + "/resources/" + resourceID + "/calphad/validations"
	if rec := do(
		http.MethodPost, policylessPath, validationBody,
		workerHeaders(policylessRun.RunID, policylessLease),
	); rec.Code != http.StatusConflict {
		t.Fatalf("policyless runtime status=%d body=%s, want 409", rec.Code, rec.Body.String())
	}
	expiredRun, err := memory.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: ownerThread.ThreadID, UserID: owner, Goal: "expired lease",
		Metadata: domain.JSONMap{
			"org_id": ownerOrg, "file_ids": []string{resourceID},
			domain.CalphadRuntimePolicyMetadataKey: calphadRunPolicy(runtimeImageID),
			"resource_descriptors": []domain.JSONMap{
				selectedCalphadDescriptor(resourceID, resourceSHA, 512),
			},
			"principal": domain.JSONMap{
				"user_id": owner, "org_id": ownerOrg, "role": "researcher",
			},
		},
	})
	if err != nil {
		t.Fatalf("CreateRun(expired lease): %v", err)
	}
	expiredLease, err := memory.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID: expiredRun.RunID, WorkerID: workerID, TTL: time.Second,
		Now: domain.Now().Add(-time.Hour),
	})
	if err != nil {
		t.Fatalf("AcquireRunLease(expired): %v", err)
	}
	expiredPath := "/v2/runs/" + expiredRun.RunID + "/resources/" + resourceID + "/calphad/validations"
	if rec := do(
		http.MethodPost, expiredPath, validationBody,
		workerHeaders(expiredRun.RunID, expiredLease),
	); rec.Code != http.StatusUnauthorized {
		t.Fatalf("expired lease status=%d body=%s, want 401", rec.Code, rec.Body.String())
	}
	unselectedPath := "/v2/runs/" + ownerRun.RunID + "/resources/" + unselectedResource + "/calphad/validations"
	unselectedBody := calphadInspectionValidationBody(
		t, unselectedResource, unselectedSHA, 256, runtimeImageID, "unselected",
	)
	if unselected := do(
		http.MethodPost, unselectedPath, unselectedBody,
		workerHeaders(ownerRun.RunID, ownerLease),
	); unselected.Code != http.StatusNotFound {
		t.Fatalf("unselected same-owner validation status=%d body=%s, want 404", unselected.Code, unselected.Body.String())
	}
	// A worker-supplied tenant header cannot override the run's server-stamped tenant.
	validHeaders := workerHeaders(ownerRun.RunID, ownerLease)
	validHeaders["X-Ultra-User-Id"] = "mallory"
	validHeaders["X-Ultra-Org-Id"] = "forged-org"
	validated := do(http.MethodPost, validationPath, validationBody, validHeaders)
	if validated.Code != http.StatusCreated {
		t.Fatalf("worker validation status=%d body=%s", validated.Code, validated.Body.String())
	}
	var firstAppend struct {
		Revision   domain.CalphadRevisionRecord   `json:"revision"`
		Validation domain.CalphadValidationRecord `json:"validation"`
	}
	if err := json.Unmarshal(validated.Body.Bytes(), &firstAppend); err != nil {
		t.Fatalf("decode first validation response: %v", err)
	}
	if firstAppend.Validation.EvidenceSHA256 != validationEvidenceSHA {
		t.Fatalf("retained evidence SHA=%q, want %q", firstAppend.Validation.EvidenceSHA256, validationEvidenceSHA)
	}
	evidencePath := "/v2/resources/" + resourceID + "/calphad/validations/" +
		firstAppend.Validation.ValidationID + "/evidence"
	replayedEvidence := do(http.MethodGet, evidencePath, "", userHeaders(owner, ownerOrg))
	if replayedEvidence.Code != http.StatusOK ||
		!bytes.Equal(replayedEvidence.Body.Bytes(), validationEvidenceBytes) ||
		replayedEvidence.Header().Get("Content-Type") != "application/json" ||
		replayedEvidence.Header().Get("Content-Length") != fmt.Sprintf("%d", len(validationEvidenceBytes)) ||
		replayedEvidence.Header().Get("ETag") != `"sha256:`+validationEvidenceSHA+`"` ||
		replayedEvidence.Header().Get("Cache-Control") != "private, immutable" ||
		replayedEvidence.Header().Get("X-Ultra-Calphad-Validation-Id") != firstAppend.Validation.ValidationID ||
		replayedEvidence.Header().Get("X-Ultra-Content-Sha256") != validationEvidenceSHA {
		t.Fatalf("exact evidence replay status=%d headers=%v body=%q", replayedEvidence.Code, replayedEvidence.Header(), replayedEvidence.Body.Bytes())
	}
	if missingEvidence := do(
		http.MethodGet,
		"/v2/resources/"+resourceID+"/calphad/validations/"+first.Ledger.LatestValidation.ValidationID+"/evidence",
		"", userHeaders(owner, ownerOrg),
	); missingEvidence.Code != http.StatusConflict {
		t.Fatalf("authorized missing evidence status=%d body=%s, want 409", missingEvidence.Code, missingEvidence.Body.String())
	}
	if foreignEvidence := do(http.MethodGet, evidencePath, "", userHeaders("bob", "bob-org")); foreignEvidence.Code != http.StatusNotFound {
		t.Fatalf("foreign evidence status=%d body=%s, want 404", foreignEvidence.Code, foreignEvidence.Body.String())
	}
	if wrongOrgEvidence := do(http.MethodGet, evidencePath, "", userHeaders(owner, "wrong-org")); wrongOrgEvidence.Code != http.StatusNotFound {
		t.Fatalf("cross-org evidence status=%d body=%s, want 404", wrongOrgEvidence.Code, wrongOrgEvidence.Body.String())
	}
	if mismatchedEvidence := do(
		http.MethodGet,
		"/v2/resources/"+unselectedResource+"/calphad/validations/"+firstAppend.Validation.ValidationID+"/evidence",
		"", userHeaders(owner, ownerOrg),
	); mismatchedEvidence.Code != http.StatusNotFound {
		t.Fatalf("cross-resource evidence status=%d body=%s, want 404", mismatchedEvidence.Code, mismatchedEvidence.Body.String())
	}
	if err := os.Remove(transientArtifactPath); err != nil {
		t.Fatalf("remove transient worker artifact: %v", err)
	}
	retry := do(
		http.MethodPost, validationPath, validationBody,
		workerHeaders(ownerRun.RunID, ownerLease),
	)
	if retry.Code != http.StatusCreated {
		t.Fatalf("idempotent retry status=%d body=%s", retry.Code, retry.Body.String())
	}
	var retried struct {
		Revision   domain.CalphadRevisionRecord   `json:"revision"`
		Validation domain.CalphadValidationRecord `json:"validation"`
	}
	if err := json.Unmarshal(retry.Body.Bytes(), &retried); err != nil {
		t.Fatalf("decode retry response: %v", err)
	}
	if retried.Revision.RevisionID != firstAppend.Revision.RevisionID ||
		retried.Validation.ValidationID != firstAppend.Validation.ValidationID {
		t.Fatalf("retry minted new records: first=%+v retry=%+v", firstAppend, retried)
	}
	failureEvidence := calphadFailureEvidence(
		resourceID, resourceSHA, 512, runtimeImageID, "equilibrium", "timeout", "scientific",
		"solver", "calphad_solver_timeout", 124, true,
	)
	failureEvidence["request"].(map[string]any)["inspection_artifact_sha256"] =
		firstAppend.Validation.EvidenceSHA256
	invalidLineageEvidence := failureEvidence
	invalidLineageEvidence["request"].(map[string]any)["inspection_artifact_sha256"] =
		strings.Repeat("9", 64)
	invalidLineageRaw, err := json.Marshal(invalidLineageEvidence)
	if err != nil {
		t.Fatalf("marshal invalid-lineage failure: %v", err)
	}
	failureEvidence["request"].(map[string]any)["inspection_artifact_sha256"] =
		firstAppend.Validation.EvidenceSHA256
	invalidLineageBody := calphadFailureValidationBodyForRaw(
		t, invalidLineageRaw, "equilibrium", "timeout", "scientific", "solver",
		"calphad_solver_timeout", runtimeImageID,
	)
	if invalidLineage := do(
		http.MethodPost, validationPath, invalidLineageBody,
		workerHeaders(ownerRun.RunID, ownerLease),
	); invalidLineage.Code != http.StatusConflict {
		t.Fatalf("invalid inspection lineage status=%d body=%s, want 409", invalidLineage.Code, invalidLineage.Body.String())
	}
	ledgerAfterInvalidLineage, err := memory.GetCalphadLedgerForOwner(ctx, resourceID, owner, ownerOrg)
	if err != nil || len(ledgerAfterInvalidLineage.Validations) != 2 {
		t.Fatalf("invalid inspection lineage appended an event: %+v err=%v", ledgerAfterInvalidLineage, err)
	}
	failureRaw, err := json.Marshal(failureEvidence)
	if err != nil {
		t.Fatalf("marshal retained failure evidence: %v", err)
	}
	failureBody := calphadFailureValidationBodyForRaw(
		t, failureRaw, "equilibrium", "timeout", "scientific", "solver",
		"calphad_solver_timeout", runtimeImageID,
	)
	failureResponse := do(
		http.MethodPost, validationPath, failureBody,
		workerHeaders(ownerRun.RunID, ownerLease),
	)
	if failureResponse.Code != http.StatusCreated {
		t.Fatalf("failure persistence status=%d body=%s", failureResponse.Code, failureResponse.Body.String())
	}
	var failureAppend struct {
		Validation domain.CalphadValidationRecord `json:"validation"`
	}
	if err := json.Unmarshal(failureResponse.Body.Bytes(), &failureAppend); err != nil {
		t.Fatalf("decode failure response: %v", err)
	}
	if failureAppend.Validation.Status != "timeout" ||
		failureAppend.Validation.FailureDomain != domain.CalphadFailureDomainScientific ||
		failureAppend.Validation.FailureStage != domain.CalphadFailureStageSolver ||
		failureAppend.Validation.FailureCode != domain.CalphadFailureCodeSolverTimeout ||
		failureAppend.Validation.InspectionEvidenceSHA256 != firstAppend.Validation.EvidenceSHA256 ||
		failureAppend.Validation.DatabaseInventorySHA256 != firstAppend.Validation.DatabaseInventorySHA256 ||
		failureAppend.Validation.EvidenceRetention != domain.CalphadEvidenceRetentionRetained ||
		failureAppend.Validation.Promotable {
		t.Fatalf("retained terminal failure=%+v", failureAppend.Validation)
	}
	failureRetry := do(
		http.MethodPost, validationPath, failureBody,
		workerHeaders(ownerRun.RunID, ownerLease),
	)
	var retriedFailure struct {
		Validation domain.CalphadValidationRecord `json:"validation"`
	}
	if failureRetry.Code != http.StatusCreated ||
		json.Unmarshal(failureRetry.Body.Bytes(), &retriedFailure) != nil ||
		retriedFailure.Validation.ValidationID != failureAppend.Validation.ValidationID {
		t.Fatalf("failure retry status=%d body=%s", failureRetry.Code, failureRetry.Body.String())
	}
	failureReplay := do(
		http.MethodGet,
		"/v2/resources/"+resourceID+"/calphad/validations/"+
			failureAppend.Validation.ValidationID+"/evidence",
		"", userHeaders(owner, ownerOrg),
	)
	if failureReplay.Code != http.StatusOK || !bytes.Equal(failureReplay.Body.Bytes(), failureRaw) {
		t.Fatalf("failure replay status=%d body=%q", failureReplay.Code, failureReplay.Body.Bytes())
	}
	inconsistentBody := calphadInspectionValidationBody(
		t, resourceID, resourceSHA, 512, runtimeImageID, "inconsistent-replay",
	)
	additional := do(
		http.MethodPost, validationPath, inconsistentBody,
		workerHeaders(ownerRun.RunID, ownerLease),
	)
	if additional.Code != http.StatusCreated {
		t.Fatalf("additional observation status=%d body=%s, want 201", additional.Code, additional.Body.String())
	}
	var secondAppend struct {
		Validation domain.CalphadValidationRecord `json:"validation"`
	}
	if err := json.Unmarshal(additional.Body.Bytes(), &secondAppend); err != nil {
		t.Fatalf("decode additional observation: %v", err)
	}
	if secondAppend.Validation.ValidationID == firstAppend.Validation.ValidationID ||
		secondAppend.Validation.RequestSHA256 != firstAppend.Validation.RequestSHA256 ||
		secondAppend.Validation.EvidenceSHA256 == firstAppend.Validation.EvidenceSHA256 {
		t.Fatalf("same-request observation identity first=%+v second=%+v", firstAppend.Validation, secondAppend.Validation)
	}
	additionalRetry := do(
		http.MethodPost, validationPath, inconsistentBody,
		workerHeaders(ownerRun.RunID, ownerLease),
	)
	var secondRetry struct {
		Validation domain.CalphadValidationRecord `json:"validation"`
	}
	if additionalRetry.Code != http.StatusCreated ||
		json.Unmarshal(additionalRetry.Body.Bytes(), &secondRetry) != nil ||
		secondRetry.Validation.ValidationID != secondAppend.Validation.ValidationID {
		t.Fatalf("additional observation retry status=%d body=%s", additionalRetry.Code, additionalRetry.Body.String())
	}
	pressureDriftPatch := do(
		http.MethodPatch, "/v2/resources/"+resourceID,
		`{"metadata":{"calphad":{"assessment_pressure_limits_Pa":[100000,200000]}}}`,
		userHeaders(owner, ownerOrg),
	)
	if pressureDriftPatch.Code != http.StatusOK {
		t.Fatalf("owner pressure drift patch status=%d body=%s", pressureDriftPatch.Code, pressureDriftPatch.Body.String())
	}
	pressureDriftBody := calphadInspectionValidationBody(
		t, resourceID, resourceSHA, 512, runtimeImageID, "owner-pressure-drift",
	)
	if pressureDrift := do(
		http.MethodPost, validationPath, pressureDriftBody,
		workerHeaders(ownerRun.RunID, ownerLease),
	); pressureDrift.Code != http.StatusConflict {
		t.Fatalf("owner pressure metadata drift status=%d body=%s, want 409", pressureDrift.Code, pressureDrift.Body.String())
	}

	var tamperedPayload map[string]any
	if err := json.Unmarshal([]byte(validationBody), &tamperedPayload); err != nil {
		t.Fatalf("decode callback for tamper test: %v", err)
	}
	tamperedPayload["evidence_sha256"] = strings.Repeat("f", 64)
	tamperedBody, err := json.Marshal(tamperedPayload)
	if err != nil {
		t.Fatalf("marshal tampered callback: %v", err)
	}
	if tampered := do(
		http.MethodPost, validationPath, string(tamperedBody),
		workerHeaders(ownerRun.RunID, ownerLease),
	); tampered.Code != http.StatusBadRequest {
		t.Fatalf("tampered evidence status=%d body=%s, want 400", tampered.Code, tampered.Body.String())
	}

	final := do(
		http.MethodGet, "/v2/resources/"+resourceID+"/calphad/ledger", "",
		userHeaders(owner, ownerOrg),
	)
	var finalLedger struct {
		Ledger domain.CalphadLedgerRecord `json:"ledger"`
	}
	if err := json.Unmarshal(final.Body.Bytes(), &finalLedger); err != nil {
		t.Fatalf("decode final ledger: %v", err)
	}
	if finalLedger.Ledger.LatestValidation == nil ||
		finalLedger.Ledger.LatestValidation.Status != "input_validated" ||
		finalLedger.Ledger.LatestValidation.ValidationID != secondAppend.Validation.ValidationID ||
		finalLedger.Ledger.LatestValidation.RunID != ownerRun.RunID ||
		finalLedger.Ledger.LatestValidation.CreatedByAuthority != "trusted_worker" ||
		finalLedger.Ledger.LatestValidation.EvidenceRetention != domain.CalphadEvidenceRetentionRetained ||
		!finalLedger.Ledger.LatestValidation.Promotable {
		t.Fatalf("final latest validation=%+v", finalLedger.Ledger.LatestValidation)
	}
	if len(finalLedger.Ledger.Validations) != 4 {
		t.Fatalf("artifact-idempotent ledger events=%d, want pending + failure + two observations", len(finalLedger.Ledger.Validations))
	}
	if finalLedger.Ledger.HasMore || finalLedger.Ledger.NextCursor != "" {
		t.Fatalf("default <=100 ledger pagination changed existing records: %+v", finalLedger.Ledger)
	}
	var finalWire struct {
		Ledger map[string]json.RawMessage `json:"ledger"`
	}
	if err := json.Unmarshal(final.Body.Bytes(), &finalWire); err != nil {
		t.Fatalf("decode final wire ledger: %v", err)
	}
	if string(finalWire.Ledger["has_more"]) != "false" {
		t.Fatalf("default bounded ledger has_more=%s, want false", finalWire.Ledger["has_more"])
	}
	if _, present := finalWire.Ledger["next_cursor"]; present {
		t.Fatalf("terminal default ledger exposed next_cursor=%s", finalWire.Ledger["next_cursor"])
	}
	pageOne := do(
		http.MethodGet, "/v2/resources/"+resourceID+"/calphad/ledger?limit=1", "",
		userHeaders(owner, ownerOrg),
	)
	var firstPage struct {
		Ledger domain.CalphadLedgerRecord `json:"ledger"`
	}
	if pageOne.Code != http.StatusOK || json.Unmarshal(pageOne.Body.Bytes(), &firstPage) != nil ||
		len(firstPage.Ledger.Validations) != 1 || !firstPage.Ledger.HasMore ||
		firstPage.Ledger.NextCursor == "" || firstPage.Ledger.LatestValidation == nil ||
		firstPage.Ledger.LatestValidation.ValidationID != secondAppend.Validation.ValidationID {
		t.Fatalf("first cursor page status=%d body=%s", pageOne.Code, pageOne.Body.String())
	}
	pageTwo := do(
		http.MethodGet,
		"/v2/resources/"+resourceID+"/calphad/ledger?limit=1&cursor="+url.QueryEscape(firstPage.Ledger.NextCursor),
		"", userHeaders(owner, ownerOrg),
	)
	var secondPage struct {
		Ledger domain.CalphadLedgerRecord `json:"ledger"`
	}
	if pageTwo.Code != http.StatusOK || json.Unmarshal(pageTwo.Body.Bytes(), &secondPage) != nil ||
		len(secondPage.Ledger.Validations) != 1 || secondPage.Ledger.LatestValidation == nil ||
		secondPage.Ledger.LatestValidation.ValidationID != secondAppend.Validation.ValidationID ||
		secondPage.Ledger.Validations[0].ValidationID == firstPage.Ledger.Validations[0].ValidationID {
		t.Fatalf("second cursor page status=%d body=%s", pageTwo.Code, pageTwo.Body.String())
	}
	for _, test := range []struct {
		name    string
		path    string
		headers map[string]string
		status  int
	}{
		{name: "malformed cursor", path: "/v2/resources/" + resourceID + "/calphad/ledger?cursor=not-base64", headers: userHeaders(owner, ownerOrg), status: http.StatusNotFound},
		{name: "cross resource cursor", path: "/v2/resources/" + unselectedResource + "/calphad/ledger?cursor=" + url.QueryEscape(firstPage.Ledger.NextCursor), headers: userHeaders(owner, ownerOrg), status: http.StatusNotFound},
		{name: "cross owner cursor", path: "/v2/resources/" + resourceID + "/calphad/ledger?cursor=" + url.QueryEscape(firstPage.Ledger.NextCursor), headers: userHeaders("bob", "bob-org"), status: http.StatusNotFound},
		{name: "cross org cursor", path: "/v2/resources/" + resourceID + "/calphad/ledger?cursor=" + url.QueryEscape(firstPage.Ledger.NextCursor), headers: userHeaders(owner, "wrong-org"), status: http.StatusNotFound},
		{name: "zero limit", path: "/v2/resources/" + resourceID + "/calphad/ledger?limit=0", headers: userHeaders(owner, ownerOrg), status: http.StatusBadRequest},
		{name: "oversized limit", path: "/v2/resources/" + resourceID + "/calphad/ledger?limit=501", headers: userHeaders(owner, ownerOrg), status: http.StatusBadRequest},
		{name: "empty limit", path: "/v2/resources/" + resourceID + "/calphad/ledger?limit=", headers: userHeaders(owner, ownerOrg), status: http.StatusBadRequest},
		{name: "duplicated limit", path: "/v2/resources/" + resourceID + "/calphad/ledger?limit=1&limit=2", headers: userHeaders(owner, ownerOrg), status: http.StatusBadRequest},
	} {
		t.Run(test.name, func(t *testing.T) {
			response := do(http.MethodGet, test.path, "", test.headers)
			if response.Code != test.status {
				t.Fatalf("status=%d body=%s, want %d", response.Code, response.Body.String(), test.status)
			}
		})
	}
	if maxPage := do(
		http.MethodGet, "/v2/resources/"+resourceID+"/calphad/ledger?limit=500", "",
		userHeaders(owner, ownerOrg),
	); maxPage.Code != http.StatusOK {
		t.Fatalf("maximum bounded ledger page status=%d body=%s", maxPage.Code, maxPage.Body.String())
	}
	decodedCursor, err := decodeCalphadLedgerCursor(firstPage.Ledger.NextCursor)
	if err != nil {
		t.Fatalf("decode server cursor: %v", err)
	}
	decodedCursor.ValidationID = "calphad-validation-missing-anchor"
	forgedAnchor, err := encodeCalphadLedgerCursor(decodedCursor)
	if err != nil {
		t.Fatalf("encode forged anchor cursor: %v", err)
	}
	if response := do(
		http.MethodGet,
		"/v2/resources/"+resourceID+"/calphad/ledger?cursor="+url.QueryEscape(forgedAnchor),
		"", userHeaders(owner, ownerOrg),
	); response.Code != http.StatusNotFound {
		t.Fatalf("forged missing anchor status=%d body=%s, want 404", response.Code, response.Body.String())
	}

	foreignPath := "/v2/runs/" + foreignRun.RunID + "/resources/" + resourceID + "/calphad/validations"
	if foreignWorker := do(
		http.MethodPost, foreignPath, validationBody,
		workerHeaders(foreignRun.RunID, foreignLease),
	); foreignWorker.Code != http.StatusNotFound {
		t.Fatalf("foreign-run validation status=%d body=%s, want 404", foreignWorker.Code, foreignWorker.Body.String())
	}

	if _, err := memory.CreateResourceShareGrant(ctx, domain.CreateResourceShareGrantInput{
		ResourceID: resourceID, OwnerUserID: owner, OwnerOrgID: ownerOrg,
		GranteeUserID: "bob", GranteeOrgID: "bob-org", Role: "read",
		CreatedByUserID: owner, CreatedAt: domain.Now(),
	}); err != nil {
		t.Fatalf("CreateResourceShareGrant: %v", err)
	}
	sharedRun, err := memory.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: bobThread.ThreadID, UserID: "bob", Goal: "shared CALPHAD",
		Metadata: domain.JSONMap{
			"org_id": "bob-org", "file_ids": []string{resourceID},
			domain.CalphadRuntimePolicyMetadataKey: calphadRunPolicy(runtimeImageID),
			"resource_descriptors": []domain.JSONMap{
				selectedCalphadDescriptor(resourceID, resourceSHA, 512),
			},
			"principal": domain.JSONMap{
				"user_id": "bob", "org_id": "bob-org", "role": "researcher",
			},
		},
	})
	if err != nil {
		t.Fatalf("CreateRun(shared): %v", err)
	}
	sharedLease, err := memory.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID: sharedRun.RunID, WorkerID: workerID, TTL: time.Hour, Now: domain.Now(),
	})
	if err != nil {
		t.Fatalf("AcquireRunLease(shared): %v", err)
	}
	sharedPath := "/v2/runs/" + sharedRun.RunID + "/resources/" + resourceID + "/calphad/validations"
	sharedBody := calphadInspectionValidationBody(
		t, resourceID, resourceSHA, 512, runtimeImageID, "shared-reader",
	)
	shared := do(
		http.MethodPost, sharedPath, sharedBody,
		workerHeaders(sharedRun.RunID, sharedLease),
	)
	if shared.Code != http.StatusNotFound {
		t.Fatalf("shared-reader ledger mutation status=%d body=%s, want 404", shared.Code, shared.Body.String())
	}
	afterShared := do(
		http.MethodGet, "/v2/resources/"+resourceID+"/calphad/ledger", "",
		userHeaders(owner, ownerOrg),
	)
	var retainedOwnerLedger struct {
		Ledger domain.CalphadLedgerRecord `json:"ledger"`
	}
	if err := json.Unmarshal(afterShared.Body.Bytes(), &retainedOwnerLedger); err != nil {
		t.Fatalf("decode ledger after shared mutation attempt: %v", err)
	}
	if len(retainedOwnerLedger.Ledger.Validations) != 4 {
		t.Fatalf("shared grantee mutated owner ledger: %+v", retainedOwnerLedger.Ledger.Validations)
	}
	if bobLedger := do(
		http.MethodGet, "/v2/resources/"+resourceID+"/calphad/ledger", "",
		userHeaders("bob", "bob-org"),
	); bobLedger.Code != http.StatusNotFound {
		t.Fatalf("shared reader read owner-only ledger status=%d body=%s", bobLedger.Code, bobLedger.Body.String())
	}
	if bobEvidence := do(http.MethodGet, evidencePath, "", userHeaders("bob", "bob-org")); bobEvidence.Code != http.StatusNotFound {
		t.Fatalf("shared grantee evidence status=%d body=%s, want 404", bobEvidence.Code, bobEvidence.Body.String())
	}
	if err := os.Remove(resourcePath); err != nil {
		t.Fatalf("remove source TDB before GC replay: %v", err)
	}
	if err := memory.PurgeResource(ctx, resourceID); err != nil {
		t.Fatalf("PurgeResource: %v", err)
	}
	afterGC := do(
		http.MethodGet, "/v2/resources/"+resourceID+"/calphad/revision/input", "",
		userHeaders(owner, ownerOrg),
	)
	if afterGC.Code != http.StatusOK || !bytes.Equal(afterGC.Body.Bytes(), resourceBytes) {
		t.Fatalf("retained input after catalog GC status=%d body=%q", afterGC.Code, afterGC.Body.Bytes())
	}
	ledgerAfterGC := do(
		http.MethodGet, "/v2/resources/"+resourceID+"/calphad/ledger", "",
		userHeaders(owner, ownerOrg),
	)
	if ledgerAfterGC.Code != http.StatusOK {
		t.Fatalf("ledger after catalog GC status=%d body=%s", ledgerAfterGC.Code, ledgerAfterGC.Body.String())
	}
	evidenceAfterGC := do(http.MethodGet, evidencePath, "", userHeaders(owner, ownerOrg))
	if evidenceAfterGC.Code != http.StatusOK || !bytes.Equal(evidenceAfterGC.Body.Bytes(), validationEvidenceBytes) {
		t.Fatalf("evidence after source/artifact/catalog deletion status=%d body=%q", evidenceAfterGC.Code, evidenceAfterGC.Body.Bytes())
	}
}

func TestCalphadWorkerValidationRouteIsExplicitlyAllowlisted(t *testing.T) {
	t.Parallel()
	req := httptest.NewRequest(http.MethodPost, "/v2/runs/run-1/resources/file-1/calphad/validations", nil)
	if !isWorkerScopedEndpoint(req) {
		t.Fatal("CALPHAD validation endpoint must be reachable through WorkOS middleware by an authenticated worker")
	}
	for _, method := range []string{http.MethodGet, http.MethodPatch, http.MethodDelete} {
		req := httptest.NewRequest(method, "/v2/runs/run-1/resources/file-1/calphad/validations", nil)
		if isWorkerScopedEndpoint(req) {
			t.Fatalf("%s unexpectedly worker-allowlisted", method)
		}
	}
}
