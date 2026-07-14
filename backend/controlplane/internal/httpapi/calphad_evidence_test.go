package httpapi

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

func decodeCalphadCallbackForTest(
	t *testing.T,
	body string,
) (appendCalphadValidationRequest, *httptest.ResponseRecorder, bool) {
	t.Helper()
	request := httptest.NewRequest(http.MethodPost, "/calphad", strings.NewReader(body))
	recorder := httptest.NewRecorder()
	var decoded appendCalphadValidationRequest
	ok := decodeCalphadValidationRequest(recorder, request, &decoded)
	return decoded, recorder, ok
}

func TestCalphadCanonicalJSONMatchesPythonUTF8Contract(t *testing.T) {
	t.Parallel()
	canonical, err := calphadCanonicalJSON(map[string]any{
		"a":       "line\u2028paragraph\u2029",
		"literal": `\u2028`,
		"z":       json.Number("1.0"),
	})
	if err != nil {
		t.Fatalf("canonical JSON: %v", err)
	}
	digest := sha256.Sum256(canonical)
	if got, want := fmt.Sprintf("%x", digest[:]), "db5f4aae323b017ec9abcc99d16ffdbe2b4294414acabedbf8743995122c81b9"; got != want {
		t.Fatalf("Python canonical digest=%s want=%s bytes=%x", got, want, canonical)
	}
}

func TestCalphadDatabaseInventoryFingerprintExcludesOnlySelection(t *testing.T) {
	t.Parallel()
	manifest := calphadInspectionResult(
		"calphad-inventory-fingerprint", strings.Repeat("a", 64), 512, "inventory",
		[]string{"AL", "NI", "VA"}, []string{"FCC_A1"},
	)
	original, err := calphadDatabaseInventorySHA256(manifest)
	if err != nil {
		t.Fatalf("original inventory fingerprint: %v", err)
	}
	manifest["requested_components"] = []string{"AL", "NI"}
	manifest["requested_phases"] = []string{"FCC_A1"}
	resealCalphadInspectionManifest(t, manifest)
	selected, err := calphadDatabaseInventorySHA256(manifest)
	if err != nil {
		t.Fatalf("selected inventory fingerprint: %v", err)
	}
	if selected != original {
		t.Fatalf("selection changed inventory fingerprint: original=%s selected=%s", original, selected)
	}
	manifest["source"] = "different owner declaration"
	resealCalphadInspectionManifest(t, manifest)
	relabelled, err := calphadDatabaseInventorySHA256(manifest)
	if err != nil {
		t.Fatalf("relabelled inventory fingerprint: %v", err)
	}
	if relabelled == original {
		t.Fatal("owner provenance relabel did not change inventory fingerprint")
	}
}

func calphadEquilibriumResult(
	resourceID, databaseSHA string,
	databaseSize int64,
	marker string,
) map[string]any {
	manifest := calphadInspectionResult(
		resourceID, databaseSHA, databaseSize, marker,
		[]string{"AL", "NI", "VA"}, []string{"FCC_A1"},
	)
	requestRecord := map[string]any{
		"components": []string{"AL", "NI", "VA"},
		"phases":     []string{"FCC_A1"},
		"conditions": map[string]any{
			"T": map[string]any{"values": []float64{900}, "units": "K"},
			"P": map[string]any{"values": []float64{101325}, "units": "Pa"},
			"N": map[string]any{"values": []float64{1}, "units": "mol"},
			"independent_compositions": map[string]any{
				"AL": map[string]any{"values": []float64{0.25}, "units": "mole_fraction"},
			},
		},
		"dependent_component": "NI",
		"phase_selection": map[string]any{
			"scope": "all_database_phases", "excluded_database_phases": []string{},
			"global_equilibrium_claim_supported": true,
		},
		"composition_closure": map[string]any{
			"grid": []any{map[string]any{"AL": 0.25, "NI": 0.75}},
			"sum":  1.0, "absolute_tolerance": 1e-12, "units": "mole_fraction",
		},
		"grid_points": 1,
		"limits": map[string]any{
			"max_grid_points": 256, "wall_time_seconds": 30,
			"max_result_bytes": 16 * 1024 * 1024,
		},
	}
	resultRecord := map[string]any{
		"point_count":        1,
		"dataset_size_bytes": 128,
		"points": []any{map[string]any{
			"conditions": map[string]any{
				"T_K": 900.0, "P_Pa": 101325.0, "N_mol": 1.0,
				"composition_mole_fraction": map[string]any{"AL": 0.25, "NI": 0.75},
			},
			"stable_phases": []any{
				map[string]any{"name": "FCC_A1", "NP_phase_fraction": 1.0},
			},
			"stable_phase_vertices": []any{map[string]any{
				"vertex_index": 0, "phase": "FCC_A1", "NP_phase_fraction": 1.0,
				"composition_mole_fraction": map[string]any{"AL": 0.25, "NI": 0.75},
				"composition_sum":           1.0,
			}},
			"phase_fraction_sum":                       1.0,
			"reconstructed_composition_mole_fraction":  map[string]any{"AL": 0.25, "NI": 0.75},
			"bulk_composition_residual_by_component":   map[string]any{"AL": 0.0, "NI": 0.0},
			"maximum_bulk_composition_residual":        0.0,
			"GM_J_per_mol":                             -1000.0,
			"chemical_potentials_J_per_mol":            map[string]any{"AL": -1000.0, "NI": -1000.0},
			"gibbs_from_chemical_potentials_J_per_mol": -1000.0,
			"gibbs_euler_residual_J_per_mol":           0.0,
		}},
		"units": map[string]any{
			"T": "K", "P": "Pa", "N": "mol", "X": "mole_fraction",
			"phase_X": "mole_fraction", "bulk_composition_residual": "mole_fraction",
			"NP": "phase_amount_fraction_at_N_equals_1_mol", "GM": "J/mol",
			"MU": "J/mol", "gibbs_euler_residual": "J/mol",
		},
	}
	warnings := []string{
		"A successful numerical solve does not independently validate the database assessment or extrapolation domain.",
		"NP is a phase-amount fraction on the fixed N=1 mol calculation basis.",
		"fixture evidence " + marker,
	}
	canonical, err := calphadCanonicalJSON(map[string]any{
		"schema_version":           calphadEquilibriumSchemaVersion,
		"database_sha256":          databaseSHA,
		"database_manifest_sha256": manifest["manifest_sha256"],
		"request":                  requestRecord,
		"result":                   resultRecord,
		"warnings":                 warnings,
		"pycalphad_version":        "0.11.2",
	})
	if err != nil {
		panic(err)
	}
	digest := sha256.Sum256(canonical)
	return map[string]any{
		"schema_version": calphadEquilibriumSchemaVersion,
		"database":       manifest,
		"request":        requestRecord,
		"result":         resultRecord,
		"warnings":       warnings,
		"evidence": map[string]any{
			"sha256": fmt.Sprintf("%x", digest[:]), "algorithm": "sha256",
			"canonicalization":        "UTF-8 JSON, sorted keys, compact separators, finite numbers",
			"canonical_serialization": true, "solver_replay_determinism_claimed": false,
		},
	}
}

func resealCalphadInspectionManifest(t *testing.T, manifest map[string]any) {
	t.Helper()
	delete(manifest, "manifest_sha256")
	canonical, err := calphadCanonicalJSON(manifest)
	if err != nil {
		t.Fatalf("canonicalize inspection manifest: %v", err)
	}
	digest := sha256.Sum256(canonical)
	manifest["manifest_sha256"] = fmt.Sprintf("%x", digest[:])
}

func setCalphadInspectionEvidenceFormat(
	t *testing.T,
	evidence map[string]any,
	databaseFormat string,
) {
	t.Helper()
	binding := evidence["database_binding"].(map[string]any)
	manifest := evidence["result"].(map[string]any)
	databaseSHA := binding["sha256"].(string)
	binding["database_format"] = databaseFormat
	manifest["format"] = databaseFormat
	manifest["name"] = databaseSHA + "." + databaseFormat
	manifest["path"] = "/workspace/.ultra/calphad/staged/" + databaseSHA + "." + databaseFormat
	resealCalphadInspectionManifest(t, manifest)
}

func resealCalphadEquilibriumResult(t *testing.T, response map[string]any) {
	t.Helper()
	manifest := response["database"].(map[string]any)
	canonical, err := calphadCanonicalJSON(map[string]any{
		"schema_version":           response["schema_version"],
		"database_sha256":          manifest["sha256"],
		"database_manifest_sha256": manifest["manifest_sha256"],
		"request":                  response["request"],
		"result":                   response["result"],
		"warnings":                 response["warnings"],
		"pycalphad_version":        manifest["pycalphad_version"],
	})
	if err != nil {
		t.Fatalf("canonicalize equilibrium result: %v", err)
	}
	digest := sha256.Sum256(canonical)
	response["evidence"].(map[string]any)["sha256"] = fmt.Sprintf("%x", digest[:])
}

func calphadScheilResult(
	t *testing.T,
	resourceID, databaseSHA string,
	databaseSize int64,
	marker string,
) map[string]any {
	t.Helper()
	manifest := calphadInspectionResult(
		resourceID, databaseSHA, databaseSize, marker,
		[]string{"AL", "NI", "VA"}, []string{"FCC_A1", "LIQUID"},
	)
	manifest["phases"] = []string{"FCC_A1", "LIQUID"}
	manifest["available_phases"] = []string{"FCC_A1", "LIQUID"}
	phaseModel := func(name string) map[string]any {
		return map[string]any{
			"name": name, "sublattice_site_ratios": []float64{1, 1},
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
		}
	}
	manifest["phase_models"] = []any{phaseModel("FCC_A1"), phaseModel("LIQUID")}
	resealCalphadInspectionManifest(t, manifest)

	requestRecord := map[string]any{
		"components":                            []string{"AL", "NI", "VA"},
		"phases":                                []string{"FCC_A1", "LIQUID"},
		"independent_composition_mole_fraction": map[string]any{"NI": 0.75},
		"bulk_composition_mole_fraction":        map[string]any{"AL": 0.25, "NI": 0.75},
		"dependent_component":                   "AL",
		"start_temperature_K":                   1500.0,
		"step_temperature_K":                    10.0,
		"pressure_Pa":                           101325.0,
		"total_amount_mol":                      1.0,
		"liquid_phase_name":                     "LIQUID",
		"stop_liquid_fraction":                  0.1,
	}
	resultRecord := map[string]any{
		"point_count":     2,
		"temperatures_K":  []float64{1500, 1400},
		"fraction_solid":  []float64{0, 0.95},
		"fraction_liquid": []float64{1, 0.05},
		"solid_phase_increment_fraction": map[string]any{
			"FCC_A1": []float64{0, 0.95},
		},
		"solid_phase_cumulative_fraction": map[string]any{
			"FCC_A1": []float64{0, 0.95},
		},
		"phase_composition_mole_fraction": map[string]any{
			"FCC_A1": map[string]any{
				"AL": []float64{0.25, 0.25}, "NI": []float64{0.75, 0.75},
			},
			"LIQUID": map[string]any{
				"AL": []float64{0.25, 0.25}, "NI": []float64{0.75, 0.75},
			},
		},
		"elemental_mass_balance": map[string]any{
			"basis": "one_mole_initial_bulk",
			"formula": "bulk_x[c] = fraction_liquid[i] * liquid_x[c,i] + " +
				"sum_phase,sum_step<=i(solid_increment[phase,step] * solid_x[phase,c,step])",
			"absolute_tolerance":               1e-6,
			"maximum_absolute_component_error": 0.0,
			"maximum_absolute_error_by_component": map[string]any{
				"AL": 0.0, "NI": 0.0,
			},
			"final_reconstructed_bulk_composition_mole_fraction": map[string]any{
				"AL": 0.25, "NI": 0.75,
			},
			"all_retained_points_closed": true,
		},
		"converged":                              true,
		"qualified_terminal_point":               "last_residual_liquid_point",
		"discarded_upstream_terminal_fill_point": false,
		"closure_tolerances": map[string]any{
			"phase_fraction_absolute": 1e-6, "composition_absolute": 1e-6,
			"elemental_mass_balance_absolute": 1e-6,
		},
	}
	response := map[string]any{
		"schema_version": calphadScheilSchemaVersion,
		"method":         "Scheil-Gulliver",
		"database":       manifest,
		"request":        requestRecord,
		"result":         resultRecord,
		"assumptions": []string{
			"Perfect mixing (infinite diffusion) in the liquid.",
			"Local equilibrium at the solid/liquid interface.",
			"No diffusion in solid phases after they form.",
			"Constant pressure of 101325 Pa and a one-mole calculation basis.",
		},
		"warnings": []string{
			"A converged numerical path does not validate the thermodynamic assessment or extrapolation domain.",
			"This path is not a back-diffusion, finite-rate diffusion, precipitation, or phase-field calculation.",
			"fixture evidence " + marker,
		},
		"solver": map[string]any{
			"name": "scheil", "version": domain.CalphadScheilVersion,
			"pycalphad_version":              domain.CalphadPycalphadVersion,
			"adaptive_constitution_sampling": true, "replay_determinism_claimed": false,
		},
		"units": map[string]any{
			"temperature": "K", "pressure": "Pa", "amount": "mol",
			"composition": "mole_fraction", "phase_fraction": "fraction_of_one_mole_basis",
		},
		"limits": map[string]any{
			"max_steps": 2048, "wall_time_seconds": 30, "max_result_bytes": 16 * 1024 * 1024,
		},
	}
	resealCalphadScheilResult(t, response)
	return response
}

func resealCalphadScheilResult(t *testing.T, response map[string]any) {
	t.Helper()
	delete(response, "evidence")
	canonical, err := calphadCanonicalJSON(response)
	if err != nil {
		t.Fatalf("canonicalize Scheil result: %v", err)
	}
	digest := sha256.Sum256(canonical)
	response["evidence"] = map[string]any{
		"sha256": fmt.Sprintf("%x", digest[:]), "algorithm": "sha256",
		"canonicalization": "UTF-8 JSON, sorted keys, compact separators, finite numbers",
	}
}

func TestVerifyCalphadEvidenceBindsExactBytesAndRejectsForgery(t *testing.T) {
	t.Parallel()
	resourceID := "calphad-evidence-resource"
	databaseSHA := strings.Repeat("a", 64)
	runtimeImage := "sha256:" + strings.Repeat("b", 64)
	validBody := calphadInspectionValidationBody(
		t, resourceID, databaseSHA, 512, runtimeImage, "valid",
	)
	validRequest, recorder, ok := decodeCalphadCallbackForTest(t, validBody)
	if !ok {
		t.Fatalf("decode valid callback status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	verified, err := verifyCalphadEvidence(validRequest, resourceID)
	if err != nil {
		t.Fatalf("verify valid evidence: %v", err)
	}
	if verified.DatabaseSHA256 != databaseSHA || verified.DatabaseSizeBytes != 512 ||
		int64(len(verified.EvidenceBytes)) != verified.EvidenceSizeBytes {
		t.Fatalf("verified evidence binding=%+v", verified)
	}

	for name, mutate := range map[string]func(map[string]any){
		"database_result_mismatch": func(evidence map[string]any) {
			evidence["database_binding"].(map[string]any)["sha256"] = strings.Repeat("c", 64)
		},
		"runtime_mismatch": func(evidence map[string]any) {
			evidence["request"].(map[string]any)["runtime_image_id"] =
				"sha256:" + strings.Repeat("d", 64)
		},
		"pycalphad_mismatch": func(evidence map[string]any) {
			evidence["result"].(map[string]any)["pycalphad_version"] = "0.10.5"
		},
		"nonselected_binding_schema": func(evidence map[string]any) {
			evidence["database_binding"].(map[string]any)["binding_schema"] =
				"ultra.catalog_resource.v1"
		},
		"weakened_sandbox_contract": func(evidence map[string]any) {
			evidence["execution_contract"].(map[string]any)["no_new_privileges"] = false
		},
	} {
		t.Run(name, func(t *testing.T) {
			evidence := calphadInspectionEvidence(
				resourceID, databaseSHA, 512, runtimeImage, name,
			)
			mutate(evidence)
			raw, marshalErr := json.Marshal(evidence)
			if marshalErr != nil {
				t.Fatalf("marshal evidence: %v", marshalErr)
			}
			body := calphadValidationBodyForRaw(
				t, raw, "inspect", "input_validated", runtimeImage, "0.11.2", nil,
			)
			request, response, decoded := decodeCalphadCallbackForTest(t, body)
			if !decoded {
				t.Fatalf("decode callback status=%d body=%s", response.Code, response.Body.String())
			}
			if _, verifyErr := verifyCalphadEvidence(request, resourceID); verifyErr == nil {
				t.Fatal("forged evidence was accepted")
			}
		})
	}
}

func TestVerifyCalphadFailureEvidenceAcceptsOnlyExactBoundedTerminalTuples(t *testing.T) {
	t.Parallel()
	resourceID := "calphad-failure-resource"
	databaseSHA := strings.Repeat("d", 64)
	runtimeImage := "sha256:" + strings.Repeat("e", 64)
	valid := []struct {
		name          string
		operation     string
		status        string
		failureDomain string
		failureStage  string
		failureCode   string
		exitCode      any
		solverStarted bool
		inspectionSHA string
	}{
		{
			name: "outer sandbox timeout", operation: "inspect", status: "timeout",
			failureDomain: "platform", failureStage: "sandbox_runtime",
			failureCode: "calphad_sandbox_timeout", exitCode: 124, solverStarted: false,
		},
		{
			name: "inner solver timeout", operation: "equilibrium", status: "timeout",
			failureDomain: "scientific", failureStage: "solver",
			failureCode: "calphad_solver_timeout", exitCode: 124, solverStarted: true,
			inspectionSHA: strings.Repeat("5", 64),
		},
		{
			name: "unsupported before solver entry", operation: "equilibrium", status: "unsupported",
			failureDomain: "scientific", failureStage: "solver",
			failureCode: "calphad_solver_unsupported", exitCode: nil, solverStarted: false,
			inspectionSHA: strings.Repeat("5", 64),
		},
	}
	for _, test := range valid {
		t.Run(test.name, func(t *testing.T) {
			evidence := calphadFailureEvidence(
				resourceID, databaseSHA, 512, runtimeImage, test.operation, test.status,
				test.failureDomain, test.failureStage, test.failureCode,
				test.exitCode, test.solverStarted,
			)
			raw, err := json.Marshal(evidence)
			if err != nil {
				t.Fatalf("marshal failure evidence: %v", err)
			}
			body := calphadFailureValidationBodyForRaw(
				t, raw, test.operation, test.status, test.failureDomain, test.failureStage,
				test.failureCode, runtimeImage,
			)
			request, recorder, ok := decodeCalphadCallbackForTest(t, body)
			if !ok {
				t.Fatalf("decode failure callback status=%d body=%s", recorder.Code, recorder.Body.String())
			}
			verified, verifyErr := verifyCalphadEvidence(request, resourceID)
			if verifyErr != nil {
				t.Fatalf("verify exact failure evidence: %v", verifyErr)
			}
			if verified.Status != test.status || string(verified.FailureDomain) != test.failureDomain ||
				string(verified.FailureStage) != test.failureStage ||
				string(verified.FailureCode) != test.failureCode ||
				verified.DatabaseInventorySHA256 != "" ||
				verified.InspectionEvidenceSHA256 != test.inspectionSHA ||
				!calphadEvidenceSHA256Pattern.MatchString(verified.RequestSHA256) {
				t.Fatalf("verified failure tuple=%+v", verified)
			}
		})
	}

	mutations := map[string]func(map[string]any){
		"timeout status with failed code": func(evidence map[string]any) {
			evidence["outcome"].(map[string]any)["status"] = "timeout"
		},
		"platform parse failure": func(evidence map[string]any) {
			evidence["outcome"].(map[string]any)["failure_domain"] = "platform"
		},
		"parse code at solver stage": func(evidence map[string]any) {
			evidence["outcome"].(map[string]any)["failure_stage"] = "solver"
		},
		"inspect claims solver started": func(evidence map[string]any) {
			evidence["outcome"].(map[string]any)["solver_started"] = true
		},
		"exit code is not integer or null": func(evidence map[string]any) {
			evidence["outcome"].(map[string]any)["exit_code"] = "1"
		},
		"raw message field": func(evidence map[string]any) {
			evidence["outcome"].(map[string]any)["message"] = "raw stderr and credential"
		},
		"raw traceback root": func(evidence map[string]any) {
			evidence["traceback"] = "private path"
		},
	}
	for name, mutate := range mutations {
		name, mutate := name, mutate
		t.Run(name, func(t *testing.T) {
			evidence := calphadFailureEvidence(
				resourceID, databaseSHA, 512, runtimeImage, "inspect", "failed", "input",
				"parse", "calphad_parse_failed", 2, false,
			)
			mutate(evidence)
			outcome := evidence["outcome"].(map[string]any)
			outcomeStatus, _ := outcome["status"].(string)
			failureDomain, _ := outcome["failure_domain"].(string)
			failureStage, _ := outcome["failure_stage"].(string)
			failureCode, _ := outcome["failure_code"].(string)
			raw, err := json.Marshal(evidence)
			if err != nil {
				t.Fatalf("marshal adversarial failure evidence: %v", err)
			}
			body := calphadFailureValidationBodyForRaw(
				t, raw, "inspect", outcomeStatus, failureDomain, failureStage, failureCode,
				runtimeImage,
			)
			request, recorder, ok := decodeCalphadCallbackForTest(t, body)
			if !ok {
				t.Fatalf("decode adversarial callback status=%d body=%s", recorder.Code, recorder.Body.String())
			}
			if _, verifyErr := verifyCalphadEvidence(request, resourceID); verifyErr == nil {
				t.Fatal("mismatched or raw-diagnostic failure evidence was accepted")
			}
		})
	}
}

func TestVerifyCalphadEvidenceAcceptsDATAndRejectsFormatDrift(t *testing.T) {
	t.Parallel()
	resourceID := "calphad-format-evidence-resource"
	databaseSHA := strings.Repeat("4", 64)
	runtimeImage := "sha256:" + strings.Repeat("5", 64)

	validDAT := calphadInspectionEvidence(resourceID, databaseSHA, 512, runtimeImage, "valid-dat")
	setCalphadInspectionEvidenceFormat(t, validDAT, domain.CalphadDatabaseFormatDAT)
	rawDAT, err := json.Marshal(validDAT)
	if err != nil {
		t.Fatalf("marshal DAT evidence: %v", err)
	}
	datBody := calphadValidationBodyForRaw(
		t, rawDAT, "inspect", "input_validated", runtimeImage, "0.11.2", nil,
	)
	datRequest, recorder, ok := decodeCalphadCallbackForTest(t, datBody)
	if !ok {
		t.Fatalf("decode DAT callback status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	verified, err := verifyCalphadEvidence(datRequest, resourceID)
	if err != nil {
		t.Fatalf("verify valid DAT evidence: %v", err)
	}
	if verified.DatabaseFormat != domain.CalphadDatabaseFormatDAT {
		t.Fatalf("verified database format=%q, want dat", verified.DatabaseFormat)
	}

	for name, mutate := range map[string]func(map[string]any){
		"missing_binding_format": func(evidence map[string]any) {
			delete(evidence["database_binding"].(map[string]any), "database_format")
		},
		"unsupported_db_even_with_matching_manifest": func(evidence map[string]any) {
			setCalphadInspectionEvidenceFormat(t, evidence, "db")
		},
		"tdb_binding_with_dat_manifest": func(evidence map[string]any) {
			manifest := evidence["result"].(map[string]any)
			manifest["format"] = domain.CalphadDatabaseFormatDAT
			manifest["name"] = databaseSHA + ".dat"
			manifest["path"] = "/workspace/.ultra/calphad/staged/" + databaseSHA + ".dat"
			resealCalphadInspectionManifest(t, manifest)
		},
		"dat_binding_with_tdb_manifest": func(evidence map[string]any) {
			evidence["database_binding"].(map[string]any)["database_format"] = domain.CalphadDatabaseFormatDAT
		},
		"dat_name_with_tdb_suffix": func(evidence map[string]any) {
			setCalphadInspectionEvidenceFormat(t, evidence, domain.CalphadDatabaseFormatDAT)
			manifest := evidence["result"].(map[string]any)
			manifest["name"] = databaseSHA + ".tdb"
			resealCalphadInspectionManifest(t, manifest)
		},
		"dat_path_with_tdb_suffix": func(evidence map[string]any) {
			setCalphadInspectionEvidenceFormat(t, evidence, domain.CalphadDatabaseFormatDAT)
			manifest := evidence["result"].(map[string]any)
			manifest["path"] = "/workspace/.ultra/calphad/staged/" + databaseSHA + ".tdb"
			resealCalphadInspectionManifest(t, manifest)
		},
	} {
		name, mutate := name, mutate
		t.Run(name, func(t *testing.T) {
			evidence := calphadInspectionEvidence(resourceID, databaseSHA, 512, runtimeImage, name)
			mutate(evidence)
			raw, marshalErr := json.Marshal(evidence)
			if marshalErr != nil {
				t.Fatalf("marshal evidence: %v", marshalErr)
			}
			body := calphadValidationBodyForRaw(
				t, raw, "inspect", "input_validated", runtimeImage, "0.11.2", nil,
			)
			request, response, decoded := decodeCalphadCallbackForTest(t, body)
			if !decoded {
				t.Fatalf("decode callback status=%d body=%s", response.Code, response.Body.String())
			}
			if _, verifyErr := verifyCalphadEvidence(request, resourceID); verifyErr == nil {
				t.Fatal("format-drift evidence was accepted")
			}
		})
	}
}

func TestVerifyCalphadEvidenceRejectsInvalidOrDriftingPressureDeclarations(t *testing.T) {
	t.Parallel()
	resourceID := "calphad-pressure-declaration"
	databaseSHA := strings.Repeat("2", 64)
	runtimeImage := "sha256:" + strings.Repeat("3", 64)
	for name, mutate := range map[string]func(map[string]any){
		"missing": func(evidence map[string]any) {
			delete(evidence["database_binding"].(map[string]any), domain.CalphadAssessmentPressureLimitsMetadataKey)
		},
		"boolean": func(evidence map[string]any) {
			evidence["database_binding"].(map[string]any)[domain.CalphadAssessmentPressureLimitsMetadataKey] =
				[]any{true, 101325.0}
		},
		"string": func(evidence map[string]any) {
			evidence["database_binding"].(map[string]any)[domain.CalphadAssessmentPressureLimitsMetadataKey] =
				[]any{"101325", 101325.0}
		},
		"reversed": func(evidence map[string]any) {
			evidence["database_binding"].(map[string]any)[domain.CalphadAssessmentPressureLimitsMetadataKey] =
				[]float64{101326, 101325}
		},
		"outside global": func(evidence map[string]any) {
			evidence["database_binding"].(map[string]any)[domain.CalphadAssessmentPressureLimitsMetadataKey] =
				[]float64{101325, domain.CalphadMaximumPressurePa + 1}
		},
		"binding manifest drift": func(evidence map[string]any) {
			evidence["database_binding"].(map[string]any)[domain.CalphadAssessmentPressureLimitsMetadataKey] =
				[]float64{100000, 200000}
		},
		"manifest binding drift": func(evidence map[string]any) {
			manifest := evidence["result"].(map[string]any)
			manifest[domain.CalphadAssessmentPressureLimitsMetadataKey] = []float64{100000, 200000}
			resealCalphadInspectionManifest(t, manifest)
		},
	} {
		name, mutate := name, mutate
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			evidence := calphadInspectionEvidence(
				resourceID, databaseSHA, 512, runtimeImage, name,
			)
			mutate(evidence)
			raw, err := json.Marshal(evidence)
			if err != nil {
				t.Fatalf("marshal pressure evidence: %v", err)
			}
			body := calphadValidationBodyForRaw(
				t, raw, "inspect", "input_validated", runtimeImage, domain.CalphadPycalphadVersion, nil,
			)
			request, recorder, ok := decodeCalphadCallbackForTest(t, body)
			if !ok {
				t.Fatalf("decode pressure callback status=%d body=%s", recorder.Code, recorder.Body.String())
			}
			if _, verifyErr := verifyCalphadEvidence(request, resourceID); verifyErr == nil {
				t.Fatal("invalid or drifting pressure declaration was accepted")
			}
		})
	}
}

func TestVerifyCalphadInspectionEvidenceRejectsResealedScientificForgeries(t *testing.T) {
	t.Parallel()
	resourceID := "calphad-inspection-scientific-forgery"
	databaseSHA := strings.Repeat("6", 64)
	runtimeImage := "sha256:" + strings.Repeat("7", 64)
	mutations := map[string]func(map[string]any){
		"incomplete_phase_models": func(manifest map[string]any) {
			manifest["phase_models"] = []any{}
		},
		"inconsistent_inventory": func(manifest map[string]any) {
			manifest["available_components"] = []string{"AL", "NI", "VA", "ZN"}
		},
		"forged_embedded_registry": func(manifest map[string]any) {
			manifest["registry_manifest"] = map[string]any{"database_id": "attacker"}
		},
		"wrong_artifact_identity": func(manifest map[string]any) {
			manifest["artifact_id"] = "different-resource"
		},
		"invalid_reference_counts": func(manifest map[string]any) {
			manifest["references"].(map[string]any)["count"] = 1
		},
		"unknown_scientific_claim": func(manifest map[string]any) {
			manifest["scientifically_valid"] = true
		},
	}
	for name, mutate := range mutations {
		name, mutate := name, mutate
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			evidence := calphadInspectionEvidence(
				resourceID, databaseSHA, 2048, runtimeImage, name,
			)
			manifest := evidence["result"].(map[string]any)
			mutate(manifest)
			resealCalphadInspectionManifest(t, manifest)
			raw, err := json.Marshal(evidence)
			if err != nil {
				t.Fatalf("marshal forged inspection evidence: %v", err)
			}
			body := calphadValidationBodyForRaw(
				t, raw, "inspect", "input_validated", runtimeImage, "0.11.2", nil,
			)
			request, recorder, ok := decodeCalphadCallbackForTest(t, body)
			if !ok {
				t.Fatalf("decode forged callback status=%d body=%s", recorder.Code, recorder.Body.String())
			}
			if _, verifyErr := verifyCalphadEvidence(request, resourceID); verifyErr == nil {
				t.Fatal("resealed scientific inspection forgery was accepted")
			}
		})
	}
}

func TestVerifyCalphadEquilibriumEvidenceAcceptsOnlyBoundedTypedRequest(t *testing.T) {
	t.Parallel()
	resourceID := "calphad-equilibrium-resource"
	databaseSHA := strings.Repeat("3", 64)
	runtimeImage := "sha256:" + strings.Repeat("4", 64)
	evidence := calphadInspectionEvidence(
		resourceID, databaseSHA, 1024, runtimeImage, "equilibrium",
	)
	evidence["operation"] = "equilibrium"
	evidence["request"] = map[string]any{
		"operation": "equilibrium", "runtime_image_id": runtimeImage,
		"selection": map[string]any{
			"components": []string{"AL", "NI", "VA"}, "phases": []string{"FCC_A1"},
		},
		"inspection_artifact_sha256": strings.Repeat("5", 64),
		"conditions": map[string]any{
			"temperatures_K": []float64{900}, "pressures_Pa": []float64{101325},
			"independent_compositions": map[string]any{"AL": []float64{0.25}},
		},
	}
	evidence["result"] = calphadEquilibriumResult(resourceID, databaseSHA, 1024, "equilibrium")
	raw, err := json.Marshal(evidence)
	if err != nil {
		t.Fatalf("marshal equilibrium evidence: %v", err)
	}
	body := calphadValidationBodyForRaw(
		t, raw, "equilibrium", "equilibrium_completed", runtimeImage, "0.11.2", nil,
	)
	request, recorder, ok := decodeCalphadCallbackForTest(t, body)
	if !ok {
		t.Fatalf("decode equilibrium callback status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	verified, err := verifyCalphadEvidence(request, resourceID)
	if err != nil {
		t.Fatalf("verify equilibrium evidence: %v", err)
	}
	if verified.InspectionEvidenceSHA256 != strings.Repeat("5", 64) {
		t.Fatalf("inspection lineage=%q", verified.InspectionEvidenceSHA256)
	}

	response := evidence["result"].(map[string]any)
	point := response["result"].(map[string]any)["points"].([]any)[0].(map[string]any)
	point["stable_phases"].([]any)[0].(map[string]any)["NP_phase_fraction"] = 1.0000000000000002
	point["stable_phase_vertices"].([]any)[0].(map[string]any)["NP_phase_fraction"] = 1.0000000000000002
	point["phase_fraction_sum"] = 1.0000000000000002
	resealCalphadEquilibriumResult(t, response)
	roundedRaw, err := json.Marshal(evidence)
	if err != nil {
		t.Fatalf("marshal rounded equilibrium evidence: %v", err)
	}
	roundedBody := calphadValidationBodyForRaw(
		t, roundedRaw, "equilibrium", "equilibrium_completed", runtimeImage, "0.11.2", nil,
	)
	roundedRequest, recorder, ok := decodeCalphadCallbackForTest(t, roundedBody)
	if !ok {
		t.Fatalf("decode rounded callback status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	if _, verifyErr := verifyCalphadEvidence(roundedRequest, resourceID); verifyErr != nil {
		t.Fatalf("floating-point phase fraction within tolerance was rejected: %v", verifyErr)
	}

	resultEvidence := evidence["result"].(map[string]any)["evidence"].(map[string]any)
	originalCanonicalSHA := resultEvidence["sha256"]
	resultEvidence["sha256"] = strings.Repeat("c", 64)
	forgedCanonicalRaw, err := json.Marshal(evidence)
	if err != nil {
		t.Fatalf("marshal forged canonical evidence: %v", err)
	}
	forgedCanonicalBody := calphadValidationBodyForRaw(
		t, forgedCanonicalRaw, "equilibrium", "equilibrium_completed", runtimeImage, "0.11.2", nil,
	)
	forgedCanonicalRequest, recorder, ok := decodeCalphadCallbackForTest(t, forgedCanonicalBody)
	if !ok {
		t.Fatalf("decode forged canonical callback status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	if _, verifyErr := verifyCalphadEvidence(forgedCanonicalRequest, resourceID); verifyErr == nil {
		t.Fatal("forged canonical scientific evidence SHA-256 was accepted")
	}
	resultEvidence["sha256"] = originalCanonicalSHA

	evidence["request"].(map[string]any)["conditions"].(map[string]any)["temperatures_K"] =
		make([]float64, 65)
	oversizedRaw, err := json.Marshal(evidence)
	if err != nil {
		t.Fatalf("marshal oversized equilibrium evidence: %v", err)
	}
	oversizedBody := calphadValidationBodyForRaw(
		t, oversizedRaw, "equilibrium", "equilibrium_completed", runtimeImage, "0.11.2", nil,
	)
	oversizedRequest, recorder, ok := decodeCalphadCallbackForTest(t, oversizedBody)
	if !ok {
		t.Fatalf("decode oversized callback status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	if _, err := verifyCalphadEvidence(oversizedRequest, resourceID); err == nil {
		t.Fatal("oversized typed equilibrium axis was accepted")
	}
}

func TestVerifyCalphadEquilibriumEvidenceRejectsResealedScientificForgeries(t *testing.T) {
	t.Parallel()
	resourceID := "calphad-equilibrium-scientific-forgery"
	databaseSHA := strings.Repeat("8", 64)
	runtimeImage := "sha256:" + strings.Repeat("9", 64)
	mutations := map[string]func(map[string]any){
		"forged_closure_grid": func(response map[string]any) {
			grid := response["request"].(map[string]any)["composition_closure"].(map[string]any)["grid"].([]any)
			grid[0].(map[string]any)["AL"] = 0.3
			grid[0].(map[string]any)["NI"] = 0.7
		},
		"forged_bulk_residual": func(response map[string]any) {
			point := response["result"].(map[string]any)["points"].([]any)[0].(map[string]any)
			point["bulk_composition_residual_by_component"].(map[string]any)["AL"] = 1e-4
			point["maximum_bulk_composition_residual"] = 1e-4
		},
		"forged_gibbs_energy": func(response map[string]any) {
			point := response["result"].(map[string]any)["points"].([]any)[0].(map[string]any)
			point["GM_J_per_mol"] = -900.0
		},
		"phase_fraction_outside_tolerance": func(response map[string]any) {
			point := response["result"].(map[string]any)["points"].([]any)[0].(map[string]any)
			point["stable_phases"].([]any)[0].(map[string]any)["NP_phase_fraction"] = 1.01
			point["stable_phase_vertices"].([]any)[0].(map[string]any)["NP_phase_fraction"] = 1.01
			point["phase_fraction_sum"] = 1.01
		},
		"missing_phase_vertices": func(response map[string]any) {
			point := response["result"].(map[string]any)["points"].([]any)[0].(map[string]any)
			delete(point, "stable_phase_vertices")
		},
		"unknown_scientific_claim": func(response map[string]any) {
			point := response["result"].(map[string]any)["points"].([]any)[0].(map[string]any)
			point["scientifically_correct"] = true
		},
		"missing_required_warning": func(response map[string]any) {
			response["warnings"] = []string{
				"A successful numerical solve does not independently validate the database assessment or extrapolation domain.",
			}
		},
	}
	for name, mutate := range mutations {
		name, mutate := name, mutate
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			evidence := calphadInspectionEvidence(
				resourceID, databaseSHA, 4096, runtimeImage, name,
			)
			evidence["operation"] = "equilibrium"
			evidence["request"] = map[string]any{
				"operation": "equilibrium", "runtime_image_id": runtimeImage,
				"selection": map[string]any{
					"components": []string{"AL", "NI", "VA"}, "phases": []string{"FCC_A1"},
				},
				"inspection_artifact_sha256": strings.Repeat("a", 64),
				"conditions": map[string]any{
					"temperatures_K": []float64{900}, "pressures_Pa": []float64{101325},
					"independent_compositions": map[string]any{"AL": []float64{0.25}},
				},
			}
			response := calphadEquilibriumResult(resourceID, databaseSHA, 4096, name)
			mutate(response)
			resealCalphadEquilibriumResult(t, response)
			evidence["result"] = response
			raw, err := json.Marshal(evidence)
			if err != nil {
				t.Fatalf("marshal forged equilibrium evidence: %v", err)
			}
			body := calphadValidationBodyForRaw(
				t, raw, "equilibrium", "equilibrium_completed", runtimeImage, "0.11.2", nil,
			)
			request, recorder, ok := decodeCalphadCallbackForTest(t, body)
			if !ok {
				t.Fatalf("decode forged callback status=%d body=%s", recorder.Code, recorder.Body.String())
			}
			if _, verifyErr := verifyCalphadEvidence(request, resourceID); verifyErr == nil {
				t.Fatal("resealed scientific equilibrium forgery was accepted")
			}
		})
	}
}

func TestVerifyCalphadScheilEvidenceReconstructsMassAndRejectsResealedForgeries(t *testing.T) {
	t.Parallel()
	resourceID := "calphad-scheil-scientific-evidence"
	databaseSHA := strings.Repeat("6", 64)
	runtimeImage := "sha256:" + strings.Repeat("7", 64)
	inspectionSHA := strings.Repeat("8", 64)

	buildEvidence := func(marker string) map[string]any {
		evidence := calphadInspectionEvidence(
			resourceID, databaseSHA, 4096, runtimeImage, marker,
		)
		evidence["operation"] = "scheil"
		evidence["request"] = map[string]any{
			"operation": "scheil", "runtime_image_id": runtimeImage,
			"selection": map[string]any{
				"components": []string{"AL", "NI", "VA"},
				"phases":     []string{"FCC_A1", "LIQUID"},
			},
			"inspection_artifact_sha256": inspectionSHA,
			"conditions": map[string]any{
				"independent_composition_mole_fraction": map[string]any{"NI": 0.75},
				"start_temperature_K":                   1500.0, "step_temperature_K": 10.0,
				"pressure_Pa": 101325.0, "stop_liquid_fraction": 0.1,
			},
		}
		evidence["result"] = calphadScheilResult(
			t, resourceID, databaseSHA, 4096, marker,
		)
		return evidence
	}

	verify := func(evidence map[string]any) (verifiedCalphadEvidence, error) {
		raw, err := json.Marshal(evidence)
		if err != nil {
			t.Fatalf("marshal Scheil evidence: %v", err)
		}
		body := calphadValidationBodyForRaw(
			t, raw, "scheil", "scheil_completed", runtimeImage,
			domain.CalphadPycalphadVersion, nil,
		)
		request, recorder, ok := decodeCalphadCallbackForTest(t, body)
		if !ok {
			t.Fatalf("decode Scheil callback status=%d body=%s", recorder.Code, recorder.Body.String())
		}
		return verifyCalphadEvidence(request, resourceID)
	}

	verified, err := verify(buildEvidence("valid-scheil"))
	if err != nil {
		t.Fatalf("verify valid Scheil evidence: %v", err)
	}
	if verified.Operation != "scheil" || verified.Status != "scheil_completed" ||
		verified.InspectionEvidenceSHA256 != inspectionSHA {
		t.Fatalf("verified Scheil ledger binding=%+v", verified)
	}

	for name, mutate := range map[string]func(map[string]any){
		"missing_phase_component": func(response map[string]any) {
			compositions := response["result"].(map[string]any)["phase_composition_mole_fraction"].(map[string]any)
			delete(compositions["FCC_A1"].(map[string]any), "NI")
		},
		"forged_mass_closure": func(response map[string]any) {
			massBalance := response["result"].(map[string]any)["elemental_mass_balance"].(map[string]any)
			massBalance["maximum_absolute_component_error"] = 1e-7
			massBalance["maximum_absolute_error_by_component"].(map[string]any)["AL"] = 1e-7
		},
		"forged_liquid_composition": func(response map[string]any) {
			compositions := response["result"].(map[string]any)["phase_composition_mole_fraction"].(map[string]any)
			compositions["LIQUID"].(map[string]any)["AL"] = []float64{0.25, 0.3}
			compositions["LIQUID"].(map[string]any)["NI"] = []float64{0.75, 0.7}
		},
	} {
		name, mutate := name, mutate
		t.Run(name, func(t *testing.T) {
			evidence := buildEvidence(name)
			response := evidence["result"].(map[string]any)
			mutate(response)
			resealCalphadScheilResult(t, response)
			if _, verifyErr := verify(evidence); verifyErr == nil {
				t.Fatal("resealed scientific Scheil forgery was accepted")
			}
		})
	}
}

func TestVerifyCalphadEquilibriumEvidenceRejectsDeclaredTemperatureExtrapolation(t *testing.T) {
	t.Parallel()
	resourceID := "calphad-equilibrium-temperature-extrapolation"
	databaseSHA := strings.Repeat("d", 64)
	runtimeImage := "sha256:" + strings.Repeat("e", 64)
	evidence := calphadInspectionEvidence(
		resourceID, databaseSHA, 4096, runtimeImage, "temperature-extrapolation",
	)
	evidence["operation"] = "equilibrium"
	evidence["request"] = map[string]any{
		"operation": "equilibrium", "runtime_image_id": runtimeImage,
		"selection": map[string]any{
			"components": []string{"AL", "NI", "VA"}, "phases": []string{"FCC_A1"},
		},
		"inspection_artifact_sha256": strings.Repeat("f", 64),
		"conditions": map[string]any{
			"temperatures_K": []float64{2500}, "pressures_Pa": []float64{101325},
			"independent_compositions": map[string]any{"AL": []float64{0.25}},
		},
	}
	response := calphadEquilibriumResult(
		resourceID, databaseSHA, 4096, "temperature-extrapolation",
	)
	response["request"].(map[string]any)["conditions"].(map[string]any)["T"].(map[string]any)["values"] =
		[]float64{2500}
	response["result"].(map[string]any)["points"].([]any)[0].(map[string]any)["conditions"].(map[string]any)["T_K"] =
		2500.0
	resealCalphadEquilibriumResult(t, response)
	evidence["result"] = response
	raw, err := json.Marshal(evidence)
	if err != nil {
		t.Fatalf("marshal extrapolated equilibrium evidence: %v", err)
	}
	body := calphadValidationBodyForRaw(
		t, raw, "equilibrium", "equilibrium_completed", runtimeImage, "0.11.2", nil,
	)
	request, recorder, ok := decodeCalphadCallbackForTest(t, body)
	if !ok {
		t.Fatalf("decode extrapolated callback status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	if _, verifyErr := verifyCalphadEvidence(request, resourceID); verifyErr == nil ||
		!strings.Contains(verifyErr.Error(), "assessment limits") {
		t.Fatalf("declared temperature extrapolation error=%v", verifyErr)
	}
}

func TestVerifyCalphadEquilibriumEvidenceRejectsFixedPressureRangeMismatch(t *testing.T) {
	t.Parallel()
	resourceID := "calphad-equilibrium-fixed-pressure"
	databaseSHA := strings.Repeat("4", 64)
	runtimeImage := "sha256:" + strings.Repeat("5", 64)
	evidence := calphadInspectionEvidence(
		resourceID, databaseSHA, 4096, runtimeImage, "fixed-pressure-mismatch",
	)
	evidence["operation"] = "equilibrium"
	evidence["request"] = map[string]any{
		"operation": "equilibrium", "runtime_image_id": runtimeImage,
		"selection": map[string]any{
			"components": []string{"AL", "NI", "VA"}, "phases": []string{"FCC_A1"},
		},
		"inspection_artifact_sha256": strings.Repeat("6", 64),
		"conditions": map[string]any{
			"temperatures_K": []float64{900}, "pressures_Pa": []float64{101326},
			"independent_compositions": map[string]any{"AL": []float64{0.25}},
		},
	}
	evidence["result"] = calphadEquilibriumResult(
		resourceID, databaseSHA, 4096, "fixed-pressure-mismatch",
	)
	raw, err := json.Marshal(evidence)
	if err != nil {
		t.Fatalf("marshal fixed pressure mismatch evidence: %v", err)
	}
	body := calphadValidationBodyForRaw(
		t, raw, "equilibrium", "equilibrium_completed", runtimeImage,
		domain.CalphadPycalphadVersion, nil,
	)
	request, recorder, ok := decodeCalphadCallbackForTest(t, body)
	if !ok {
		t.Fatalf("decode fixed pressure callback status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	if _, verifyErr := verifyCalphadEvidence(request, resourceID); verifyErr == nil ||
		!strings.Contains(verifyErr.Error(), "owner-declared assessment limits") {
		t.Fatalf("fixed pressure mismatch error=%v", verifyErr)
	}
}

func TestCalphadEvidenceRejectsDuplicateKeysTrailingMembersAndZipBombs(t *testing.T) {
	resourceID := "calphad-evidence-bounds"
	databaseSHA := strings.Repeat("e", 64)
	runtimeImage := "sha256:" + strings.Repeat("f", 64)
	evidence := calphadInspectionEvidence(resourceID, databaseSHA, 64, runtimeImage, "bounds")
	raw, err := json.Marshal(evidence)
	if err != nil {
		t.Fatalf("marshal evidence: %v", err)
	}
	duplicateRaw := append(append([]byte(nil), raw[:len(raw)-1]...), []byte(`,"operation":"inspect"}`)...)
	duplicateBody := calphadValidationBodyForRaw(
		t, duplicateRaw, "inspect", "input_validated", runtimeImage, "0.11.2", nil,
	)
	duplicateRequest, recorder, ok := decodeCalphadCallbackForTest(t, duplicateBody)
	if !ok {
		t.Fatalf("decode duplicate callback status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	if _, err := verifyCalphadEvidence(duplicateRequest, resourceID); err == nil ||
		!strings.Contains(err.Error(), "duplicate key") {
		t.Fatalf("duplicate evidence error=%v", err)
	}

	trailingBody := calphadValidationBodyForRaw(
		t, raw, "inspect", "input_validated", runtimeImage, "0.11.2", []byte("trailing"),
	)
	trailingRequest, recorder, ok := decodeCalphadCallbackForTest(t, trailingBody)
	if !ok {
		t.Fatalf("decode trailing callback status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	if _, err := verifyCalphadEvidence(trailingRequest, resourceID); err == nil ||
		!strings.Contains(err.Error(), "trailing") {
		t.Fatalf("trailing gzip error=%v", err)
	}

	oversizedRaw := bytes.Repeat([]byte("x"), maxCalphadRawEvidenceBytes+1)
	oversizedEncoded := base64.StdEncoding.EncodeToString(calphadGzipBytes(t, oversizedRaw))
	if _, err := decodeBoundedCalphadEvidence(oversizedEncoded); err == nil ||
		!strings.Contains(err.Error(), "fixed bound") {
		t.Fatalf("zip bomb error=%v", err)
	}
}

func TestCalphadValidationEnvelopeRejectsDuplicateAndUnknownFields(t *testing.T) {
	t.Parallel()
	body := calphadInspectionValidationBody(
		t, "calphad-envelope", strings.Repeat("1", 64), 8,
		"sha256:"+strings.Repeat("2", 64), "envelope",
	)
	for name, suffix := range map[string]string{
		"duplicate": `,"status":"input_validated"}`,
		"unknown":   `,"caller_claim":"trusted"}`,
	} {
		t.Run(name, func(t *testing.T) {
			forged := body[:len(body)-1] + suffix
			_, recorder, ok := decodeCalphadCallbackForTest(t, forged)
			if ok || recorder.Code != http.StatusBadRequest {
				t.Fatalf("forged envelope accepted=%t status=%d body=%s", ok, recorder.Code, recorder.Body.String())
			}
		})
	}
}
