package httpapi

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

const (
	calphadCompositionTolerance   = 1e-12
	calphadBulkResidualTolerance  = 1e-8
	calphadPhaseFractionTolerance = 1e-6
	calphadGibbsEulerTolerance    = 1e-3
	calphadScheilSchemaVersion    = "ultra.materials.scheil-gulliver.v1"
	calphadScheilMaximumSteps     = 2048
	calphadScheilClosureTolerance = 1e-6
)

func calphadRequiredKeys(object map[string]any, keys ...string) bool {
	for _, key := range keys {
		if _, ok := object[key]; !ok {
			return false
		}
	}
	return true
}

func calphadStringList(value any, maximum int, allowEmpty bool) ([]string, bool) {
	values, ok := value.([]any)
	if !ok || len(values) > maximum || (!allowEmpty && len(values) == 0) {
		return nil, false
	}
	result := make([]string, 0, len(values))
	previous := ""
	for _, item := range values {
		name, ok := jsonString(item)
		if !ok || !calphadEvidenceNamePattern.MatchString(name) || (previous != "" && name <= previous) {
			return nil, false
		}
		result = append(result, name)
		previous = name
	}
	return result, true
}

func calphadNumberList(value any, maximum int, minimum, upper float64, allowEmpty bool) ([]float64, bool) {
	values, ok := value.([]any)
	if !ok || len(values) > maximum || (!allowEmpty && len(values) == 0) {
		return nil, false
	}
	result := make([]float64, 0, len(values))
	previous := math.Inf(-1)
	for _, item := range values {
		number, ok := jsonFiniteFloat64(item)
		if !ok || number < minimum || number > upper || number <= previous {
			return nil, false
		}
		result = append(result, number)
		previous = number
	}
	return result, true
}

func calphadBoundedNumberList(value any, maximum int, minimum, upper float64, allowEmpty bool) ([]float64, bool) {
	values, ok := value.([]any)
	if !ok || len(values) > maximum || (!allowEmpty && len(values) == 0) {
		return nil, false
	}
	result := make([]float64, 0, len(values))
	for _, item := range values {
		number, ok := jsonFiniteFloat64(item)
		if !ok || number < minimum || number > upper {
			return nil, false
		}
		result = append(result, number)
	}
	return result, true
}

func calphadOptionalStringList(value any, maximum int) ([]string, bool) {
	if value == nil {
		return nil, true
	}
	return calphadStringList(value, maximum, false)
}

func calphadSameStrings(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func calphadClose(left, right, tolerance float64) bool {
	return math.Abs(left-right) <= tolerance
}

func calphadFiniteMap(value any, expected []string, minimum, maximum float64) (map[string]float64, bool) {
	object, ok := jsonObject(value)
	if !ok || len(object) != len(expected) {
		return nil, false
	}
	result := make(map[string]float64, len(object))
	for _, name := range expected {
		number, ok := jsonFiniteFloat64(object[name])
		if !ok || number < minimum || number > maximum {
			return nil, false
		}
		result[name] = number
	}
	return result, true
}

func calphadStringSubset(values, allowed []string) bool {
	set := make(map[string]struct{}, len(allowed))
	for _, value := range allowed {
		set[value] = struct{}{}
	}
	for _, value := range values {
		if _, ok := set[value]; !ok {
			return false
		}
	}
	return true
}

func validateCalphadInspectionManifest(
	result, database map[string]any,
	pycalphadVersion string,
	expectedRequestedComponents, expectedRequestedPhases []string,
) error {
	manifest, ok := exactJSONObject(
		result,
		"schema_version", "path", "name", "sha256", "size_bytes", "format",
		"package_test_database", "source", "license_id", "artifact_id",
		"assessment_scope", "reference_state", "requested_components", "requested_phases",
		"components", "physical_elements", "vacancy_components", "pseudo_elements", "species",
		"phases", "available_components", "available_phases", "phase_models", "parameter_count",
		"references", "pycalphad_version", "registry_manifest",
		"assessment_temperature_limits_K", domain.CalphadAssessmentPressureLimitsMetadataKey,
		"warnings", "limits", "manifest_sha256",
	)
	if !ok {
		return errors.New("CALPHAD inspection result does not match the exact manifest schema")
	}
	if manifest["package_test_database"] != false {
		return errors.New("CALPHAD inspection result is a package test database")
	}
	schema, _ := jsonString(manifest["schema_version"])
	format, _ := jsonString(manifest["format"])
	path, pathOK := jsonString(manifest["path"])
	name, nameOK := jsonString(manifest["name"])
	sha, _ := jsonString(manifest["sha256"])
	size, sizeOK := jsonInt64(manifest["size_bytes"])
	version, _ := jsonString(manifest["pycalphad_version"])
	manifestSHA, manifestSHAOK := jsonString(manifest["manifest_sha256"])
	databaseSHA, _ := jsonString(database["sha256"])
	databaseSize, _ := jsonInt64(database["size_bytes"])
	databaseFormat, _ := jsonString(database["database_format"])
	databaseResourceID, _ := jsonString(database["resource_id"])
	expectedName := databaseSHA + "." + databaseFormat
	validStagedPath := name == expectedName && path == "/workspace/.ultra/calphad/staged/"+expectedName
	if schema != "1" || (format != domain.CalphadDatabaseFormatTDB && format != domain.CalphadDatabaseFormatDAT) ||
		format != databaseFormat || !pathOK || strings.TrimSpace(path) == "" ||
		!nameOK || !validStagedPath || sha != databaseSHA ||
		!sizeOK || size != databaseSize || version != pycalphadVersion || !manifestSHAOK ||
		!calphadEvidenceSHA256Pattern.MatchString(manifestSHA) {
		return errors.New("CALPHAD inspection result database identity is invalid")
	}
	for _, field := range []string{"source", "license_id", "assessment_scope", "reference_state"} {
		observed, observedOK := jsonString(manifest[field])
		expected, expectedOK := jsonString(database[field])
		if !observedOK || !expectedOK || observed != expected || strings.TrimSpace(observed) == "" {
			return fmt.Errorf("CALPHAD inspection result %s differs from the owner declaration", field)
		}
	}
	artifactID, artifactIDOK := jsonString(manifest["artifact_id"])
	if !artifactIDOK || artifactID != databaseResourceID || strings.TrimSpace(artifactID) == "" || len(artifactID) > 512 {
		return errors.New("CALPHAD inspection artifact identity is invalid")
	}
	if manifest["registry_manifest"] != nil {
		return errors.New("resource CALPHAD evidence must not claim embedded-registry provenance")
	}
	components, componentsOK := calphadStringList(manifest["components"], 256, false)
	physical, physicalOK := calphadStringList(manifest["physical_elements"], 256, false)
	vacancies, vacanciesOK := calphadStringList(manifest["vacancy_components"], 8, true)
	pseudo, pseudoOK := calphadStringList(manifest["pseudo_elements"], 8, true)
	species, speciesOK := calphadStringList(manifest["species"], 4096, false)
	phases, phasesOK := calphadStringList(manifest["phases"], 2048, false)
	availableComponents, availableComponentsOK := calphadStringList(manifest["available_components"], 256, false)
	availablePhases, availablePhasesOK := calphadStringList(manifest["available_phases"], 2048, false)
	requestedComponents, requestedComponentsOK := calphadStringList(manifest["requested_components"], 32, true)
	requestedPhases, requestedPhasesOK := calphadStringList(manifest["requested_phases"], 128, true)
	if expectedRequestedComponents == nil {
		expectedRequestedComponents = make([]string, 0, len(components))
		for _, component := range components {
			if component != "/-" {
				expectedRequestedComponents = append(expectedRequestedComponents, component)
			}
		}
	}
	if expectedRequestedPhases == nil {
		expectedRequestedPhases = phases
	}
	if !componentsOK || !physicalOK || !vacanciesOK || !pseudoOK || !speciesOK || !phasesOK ||
		!availableComponentsOK || !availablePhasesOK || !requestedComponentsOK || !requestedPhasesOK ||
		!calphadSameStrings(components, availableComponents) || !calphadSameStrings(phases, availablePhases) ||
		!calphadSameStrings(requestedComponents, expectedRequestedComponents) ||
		!calphadSameStrings(requestedPhases, expectedRequestedPhases) ||
		!calphadStringSubset(physical, components) || !calphadStringSubset(vacancies, components) ||
		!calphadStringSubset(pseudo, components) || !calphadStringSubset(requestedComponents, components) ||
		!calphadStringSubset(requestedPhases, phases) {
		return errors.New("CALPHAD inspection inventory is empty, unsorted, or inconsistent")
	}
	expectedPhysical := make([]string, 0, len(components))
	expectedVacancies := make([]string, 0, 1)
	expectedPseudo := make([]string, 0, 1)
	for _, component := range components {
		switch component {
		case "VA":
			expectedVacancies = append(expectedVacancies, component)
		case "/-":
			expectedPseudo = append(expectedPseudo, component)
		default:
			expectedPhysical = append(expectedPhysical, component)
		}
	}
	if !calphadSameStrings(physical, expectedPhysical) || !calphadSameStrings(vacancies, expectedVacancies) ||
		!calphadSameStrings(pseudo, expectedPseudo) {
		return errors.New("CALPHAD inspection component classification is inconsistent")
	}
	parameterCount, parameterOK := jsonInt64(manifest["parameter_count"])
	if !parameterOK || parameterCount < 0 || parameterCount > 1_000_000 {
		return errors.New("CALPHAD inspection parameter inventory is invalid")
	}
	models, modelsOK := manifest["phase_models"].([]any)
	if !modelsOK || len(models) != len(phases) {
		return errors.New("CALPHAD inspection phase-model inventory is incomplete")
	}
	for index, rawModel := range models {
		model, ok := exactJSONObject(rawModel, "name", "sublattice_site_ratios", "sublattices", "model_hints")
		if !ok {
			return errors.New("CALPHAD inspection phase model has an invalid schema")
		}
		modelName, _ := jsonString(model["name"])
		ratios, ratiosOK := calphadBoundedNumberList(model["sublattice_site_ratios"], 64, math.SmallestNonzeroFloat64, 1e12, false)
		sublattices, sublatticesOK := model["sublattices"].([]any)
		hints, hintsOK := jsonObject(model["model_hints"])
		if modelName != phases[index] || !ratiosOK || !sublatticesOK || len(sublattices) != len(ratios) ||
			!hintsOK || len(hints) > 128 {
			return errors.New("CALPHAD inspection phase model is inconsistent with the phase inventory")
		}
		for hintName, rawHint := range hints {
			if strings.TrimSpace(hintName) == "" {
				return errors.New("CALPHAD inspection model hint is invalid")
			}
			if _, ok := jsonString(rawHint); !ok {
				return errors.New("CALPHAD inspection model hint is not a string")
			}
		}
		for sublatticeIndex, rawSublattice := range sublattices {
			sublattice, ok := exactJSONObject(rawSublattice, "index", "site_ratio", "constituents")
			if !ok {
				return errors.New("CALPHAD inspection sublattice schema is invalid")
			}
			observedIndex, indexOK := jsonInt64(sublattice["index"])
			ratio, ratioOK := jsonFiniteFloat64(sublattice["site_ratio"])
			constituents, constituentsOK := calphadStringList(sublattice["constituents"], 4096, false)
			if !indexOK || observedIndex != int64(sublatticeIndex) || !ratioOK || ratio != ratios[sublatticeIndex] ||
				!constituentsOK || !calphadStringSubset(constituents, species) {
				return errors.New("CALPHAD inspection sublattice inventory is inconsistent")
			}
		}
	}
	references, ok := exactJSONObject(manifest["references"], "count", "included_count", "truncated", "entries")
	if !ok {
		return errors.New("CALPHAD inspection reference inventory is invalid")
	}
	referenceCount, countOK := jsonInt64(references["count"])
	includedCount, includedOK := jsonInt64(references["included_count"])
	referenceEntries, entriesOK := references["entries"].([]any)
	truncated, truncatedOK := references["truncated"].(bool)
	expectedIncludedCount := referenceCount
	if expectedIncludedCount > 512 {
		expectedIncludedCount = 512
	}
	if !countOK || referenceCount < 0 || !includedOK || includedCount != expectedIncludedCount ||
		!entriesOK || int64(len(referenceEntries)) != includedCount ||
		!truncatedOK || truncated != (referenceCount > includedCount) {
		return errors.New("CALPHAD inspection reference inventory is inconsistent")
	}
	for _, rawReference := range referenceEntries {
		reference, ok := exactJSONObject(rawReference, "reference_id", "text")
		referenceID, idOK := jsonString(reference["reference_id"])
		referenceText, textOK := jsonString(reference["text"])
		if !ok || !idOK || !textOK || len(referenceID) > 256 || len(referenceText) > 2048 {
			return errors.New("CALPHAD inspection reference entry is invalid")
		}
	}
	warnings, warningsOK := manifest["warnings"].([]any)
	if !warningsOK || len(warnings) > 1024 {
		return errors.New("CALPHAD inspection warnings are invalid")
	}
	previousWarning := ""
	for _, rawWarning := range warnings {
		warning, ok := jsonString(rawWarning)
		if !ok || strings.TrimSpace(warning) == "" || len(warning) > 4096 ||
			(previousWarning != "" && warning <= previousWarning) {
			return errors.New("CALPHAD inspection warning is invalid")
		}
		previousWarning = warning
	}
	limits, ok := exactJSONObject(
		manifest["limits"], "max_database_bytes", "max_elements", "max_species", "max_phases",
		"max_parameters", "database_parse_wall_time_seconds",
	)
	if !ok {
		return errors.New("CALPHAD inspection limit record is invalid")
	}
	expectedIntegerLimits := map[string]int64{
		"max_database_bytes": 64 * 1024 * 1024,
		"max_elements":       256,
		"max_species":        4096,
		"max_phases":         2048,
		"max_parameters":     1_000_000,
	}
	for field, expected := range expectedIntegerLimits {
		observed, ok := jsonInt64(limits[field])
		if !ok || observed != expected {
			return errors.New("CALPHAD inspection runtime limits differ from the reviewed contract")
		}
	}
	parseWallTime, wallOK := jsonFiniteFloat64(limits["database_parse_wall_time_seconds"])
	if !wallOK || parseWallTime != 15 {
		return errors.New("CALPHAD inspection parse limit differs from the reviewed contract")
	}
	manifestLimits, limitsOK := calphadNumberList(manifest["assessment_temperature_limits_K"], 2, 1, 10_000, false)
	bindingLimits, bindingLimitsOK := calphadNumberList(database["temperature_limits_K"], 2, 1, 10_000, false)
	if !limitsOK || !bindingLimitsOK || len(manifestLimits) != 2 || len(bindingLimits) != 2 ||
		manifestLimits[0] != bindingLimits[0] || manifestLimits[1] != bindingLimits[1] {
		return errors.New("CALPHAD inspection assessment limits differ from the resource binding")
	}
	manifestPressureLimits, manifestPressureOK := calphadBoundedNumberList(
		manifest[domain.CalphadAssessmentPressureLimitsMetadataKey], 2,
		domain.CalphadMinimumPressurePa, domain.CalphadMaximumPressurePa, false,
	)
	bindingPressureLimits, bindingPressureOK := calphadBoundedNumberList(
		database[domain.CalphadAssessmentPressureLimitsMetadataKey], 2,
		domain.CalphadMinimumPressurePa, domain.CalphadMaximumPressurePa, false,
	)
	if !manifestPressureOK || !bindingPressureOK || len(manifestPressureLimits) != 2 ||
		len(bindingPressureLimits) != 2 || manifestPressureLimits[0] > manifestPressureLimits[1] ||
		bindingPressureLimits[0] > bindingPressureLimits[1] ||
		manifestPressureLimits[0] != bindingPressureLimits[0] ||
		manifestPressureLimits[1] != bindingPressureLimits[1] {
		return errors.New("CALPHAD inspection assessment pressure limits differ from the resource binding")
	}
	manifestPayload := make(map[string]any, len(manifest)-1)
	for key, value := range manifest {
		if key != "manifest_sha256" {
			manifestPayload[key] = value
		}
	}
	canonical, err := calphadCanonicalJSON(manifestPayload)
	if err != nil {
		return errors.New("CALPHAD inspection manifest could not be canonicalized")
	}
	digest := sha256.Sum256(canonical)
	if manifestSHA != hex.EncodeToString(digest[:]) {
		return errors.New("CALPHAD inspection manifest SHA-256 is inconsistent")
	}
	return nil
}

func calphadAxisRecord(value any, units string, minimum, maximum float64) ([]float64, bool) {
	record, ok := exactJSONObject(value, "values", "units")
	if !ok || record["units"] != units {
		return nil, false
	}
	return calphadNumberList(record["values"], 64, minimum, maximum, false)
}

func calphadIndex(values []float64, target, tolerance float64) (int, bool) {
	for index, value := range values {
		if calphadClose(value, target, tolerance) {
			return index, true
		}
	}
	return 0, false
}

func calphadCanonicalJSON(value any) ([]byte, error) {
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return nil, err
	}
	canonical := bytes.TrimSuffix(buffer.Bytes(), []byte("\n"))
	// Python's reviewed producer uses json.dumps(..., ensure_ascii=False), while
	// encoding/json always escapes the two Unicode line separators for legacy
	// JSONP safety. They are valid UTF-8 JSON string content, so normalize those
	// two Go-only escapes before reconstructing the producer's canonical hash.
	canonical = calphadNormalizePythonUTF8JSON(canonical)
	return canonical, nil
}

func calphadTypedRequestSHA256(request map[string]any) (string, error) {
	canonical, err := calphadCanonicalJSON(request)
	if err != nil {
		return "", errors.New("CALPHAD typed request could not be canonicalized")
	}
	digest := sha256.Sum256(canonical)
	return hex.EncodeToString(digest[:]), nil
}

// calphadDatabaseInventorySHA256 identifies the immutable database inventory
// independently of one inspect/equilibrium selection. The producer legitimately
// changes requested_components, requested_phases, and therefore manifest_sha256
// for a subset calculation; every other validated manifest fact must remain
// byte-for-byte equivalent to the retained inspection event.
func calphadDatabaseInventorySHA256(manifest map[string]any) (string, error) {
	inventory := make(map[string]any, len(manifest)-3)
	for key, value := range manifest {
		switch key {
		case "requested_components", "requested_phases", "manifest_sha256":
			continue
		default:
			inventory[key] = value
		}
	}
	canonical, err := calphadCanonicalJSON(inventory)
	if err != nil {
		return "", errors.New("CALPHAD database inventory could not be canonicalized")
	}
	digest := sha256.Sum256(canonical)
	return hex.EncodeToString(digest[:]), nil
}

func calphadNormalizePythonUTF8JSON(encoded []byte) []byte {
	normalized := make([]byte, 0, len(encoded))
	for index := 0; index < len(encoded); {
		if index+6 <= len(encoded) && encoded[index] == '\\' && encoded[index+1] == 'u' &&
			encoded[index+2] == '2' && encoded[index+3] == '0' && encoded[index+4] == '2' &&
			(encoded[index+5] == '8' || encoded[index+5] == '9') {
			precedingBackslashes := 0
			for previous := index - 1; previous >= 0 && encoded[previous] == '\\'; previous-- {
				precedingBackslashes++
			}
			if precedingBackslashes%2 == 0 {
				if encoded[index+5] == '8' {
					normalized = append(normalized, []byte("\u2028")...)
				} else {
					normalized = append(normalized, []byte("\u2029")...)
				}
				index += 6
				continue
			}
		}
		normalized = append(normalized, encoded[index])
		index++
	}
	return normalized
}

func validateCalphadEquilibriumResult(result, evidenceRequest, database map[string]any, pycalphadVersion string) error {
	response, ok := exactJSONObject(result, "schema_version", "database", "request", "result", "warnings", "evidence")
	if !ok || response["schema_version"] != calphadEquilibriumSchemaVersion {
		return errors.New("CALPHAD equilibrium result does not match the exact v2 schema")
	}
	selection, _ := jsonObject(evidenceRequest["selection"])
	selectedComponents, componentsOK := calphadStringList(selection["components"], 32, false)
	selectedPhases, phasesOK := calphadStringList(selection["phases"], 128, false)
	if !componentsOK || !phasesOK {
		return errors.New("CALPHAD equilibrium selection is invalid")
	}
	resultComponents := make([]string, 0, len(selectedComponents))
	for _, component := range selectedComponents {
		if component != "VA" && component != "/-" {
			resultComponents = append(resultComponents, component)
		}
	}
	if len(resultComponents) == 0 {
		return errors.New("CALPHAD equilibrium selection has no physical components")
	}
	manifest, ok := jsonObject(response["database"])
	if !ok {
		return errors.New("CALPHAD equilibrium result omits its database manifest")
	}
	if err := validateCalphadInspectionManifest(
		manifest, database, pycalphadVersion, selectedComponents, selectedPhases,
	); err != nil {
		return fmt.Errorf("CALPHAD equilibrium database manifest is invalid: %w", err)
	}
	request, ok := exactJSONObject(
		response["request"], "components", "phases", "conditions", "dependent_component",
		"phase_selection", "composition_closure", "grid_points", "limits",
	)
	if !ok {
		return errors.New("CALPHAD equilibrium result request record is invalid")
	}
	requestComponents, requestComponentsOK := calphadStringList(request["components"], 32, false)
	requestPhases, requestPhasesOK := calphadStringList(request["phases"], 128, false)
	if !requestComponentsOK || !requestPhasesOK || !calphadSameStrings(requestComponents, selectedComponents) ||
		!calphadSameStrings(requestPhases, selectedPhases) {
		return errors.New("CALPHAD equilibrium result selection differs from its typed request")
	}
	conditions, ok := exactJSONObject(request["conditions"], "T", "P", "N", "independent_compositions")
	if !ok {
		return errors.New("CALPHAD equilibrium condition record is invalid")
	}
	temperatures, temperaturesOK := calphadAxisRecord(conditions["T"], "K", 1, 10_000)
	pressures, pressuresOK := calphadAxisRecord(conditions["P"], "Pa", 1e-9, 1e12)
	amounts, amountsOK := calphadAxisRecord(conditions["N"], "mol", 1e-12, 1e12)
	if !temperaturesOK || !pressuresOK || !amountsOK || len(amounts) != 1 || amounts[0] != 1 {
		return errors.New("CALPHAD equilibrium result axes are invalid")
	}
	typedConditions, _ := jsonObject(evidenceRequest["conditions"])
	typedTemperatures, typedTemperaturesOK := calphadNumberList(typedConditions["temperatures_K"], 64, 1, 10_000, false)
	typedPressures, typedPressuresOK := calphadNumberList(typedConditions["pressures_Pa"], 64, 1e-9, 1e12, false)
	if !typedTemperaturesOK || !typedPressuresOK || len(temperatures) != len(typedTemperatures) ||
		len(pressures) != len(typedPressures) {
		return errors.New("CALPHAD equilibrium typed axes are invalid")
	}
	for index := range temperatures {
		if temperatures[index] != typedTemperatures[index] {
			return errors.New("CALPHAD equilibrium temperature axis differs from its typed request")
		}
	}
	for index := range pressures {
		if pressures[index] != typedPressures[index] {
			return errors.New("CALPHAD equilibrium pressure axis differs from its typed request")
		}
	}
	assessmentLimits, assessmentLimitsOK := calphadNumberList(
		manifest["assessment_temperature_limits_K"], 2, 1, 10_000, false,
	)
	if !assessmentLimitsOK || len(assessmentLimits) != 2 ||
		temperatures[0] < assessmentLimits[0] ||
		temperatures[len(temperatures)-1] > assessmentLimits[1] {
		return errors.New("CALPHAD equilibrium temperature axis is outside the declared assessment limits")
	}
	assessmentPressureLimits, assessmentPressureLimitsOK := calphadBoundedNumberList(
		manifest[domain.CalphadAssessmentPressureLimitsMetadataKey], 2,
		domain.CalphadMinimumPressurePa, domain.CalphadMaximumPressurePa, false,
	)
	if !assessmentPressureLimitsOK || len(assessmentPressureLimits) != 2 ||
		assessmentPressureLimits[0] > assessmentPressureLimits[1] ||
		pressures[0] < assessmentPressureLimits[0] ||
		pressures[len(pressures)-1] > assessmentPressureLimits[1] {
		return errors.New("CALPHAD equilibrium pressure axis is outside the declared assessment limits")
	}
	independent, independentOK := jsonObject(conditions["independent_compositions"])
	typedIndependent, typedIndependentOK := jsonObject(typedConditions["independent_compositions"])
	if !independentOK || !typedIndependentOK || len(independent) != len(typedIndependent) ||
		len(independent) != len(resultComponents)-1 {
		return errors.New("CALPHAD equilibrium independent composition record is invalid")
	}
	independentNames := make([]string, 0, len(independent))
	independentAxes := make(map[string][]float64, len(independent))
	for name, rawAxis := range independent {
		if !calphadEvidenceNamePattern.MatchString(name) {
			return errors.New("CALPHAD equilibrium composition component is invalid")
		}
		axis, axisOK := calphadAxisRecord(rawAxis, "mole_fraction", 0, 1)
		typedAxis, typedAxisOK := calphadNumberList(typedIndependent[name], 64, 0, 1, false)
		if !axisOK || !typedAxisOK || len(axis) != len(typedAxis) {
			return errors.New("CALPHAD equilibrium composition axis is invalid")
		}
		for index := range axis {
			if axis[index] != typedAxis[index] {
				return errors.New("CALPHAD equilibrium composition axis differs from its typed request")
			}
		}
		independentNames = append(independentNames, name)
		independentAxes[name] = axis
	}
	sort.Strings(independentNames)
	dependent, dependentOK := jsonString(request["dependent_component"])
	if !dependentOK || !calphadStringSubset(independentNames, resultComponents) {
		return errors.New("CALPHAD equilibrium dependent component record is invalid")
	}
	dependentCount := 0
	for _, component := range resultComponents {
		if component == dependent {
			dependentCount++
		}
	}
	if dependentCount != 1 {
		return errors.New("CALPHAD equilibrium dependent component is not selected")
	}
	for _, name := range independentNames {
		if name == dependent {
			return errors.New("CALPHAD equilibrium dependent component is also independent")
		}
	}
	expectedGridPoints := len(temperatures) * len(pressures)
	for _, name := range independentNames {
		expectedGridPoints *= len(independentAxes[name])
	}
	gridPoints, gridOK := jsonInt64(request["grid_points"])
	if !gridOK || gridPoints != int64(expectedGridPoints) || expectedGridPoints <= 0 || expectedGridPoints > 256 {
		return errors.New("CALPHAD equilibrium grid count is invalid")
	}
	limits, ok := exactJSONObject(request["limits"], "max_grid_points", "wall_time_seconds", "max_result_bytes")
	maxGrid, maxGridOK := jsonInt64(limits["max_grid_points"])
	wallTime, wallTimeOK := jsonFiniteFloat64(limits["wall_time_seconds"])
	maxResult, maxResultOK := jsonInt64(limits["max_result_bytes"])
	if !ok || !maxGridOK || maxGrid != 256 || !wallTimeOK || wallTime != 30 ||
		!maxResultOK || maxResult != 16*1024*1024 {
		return errors.New("CALPHAD equilibrium result limits differ from the reviewed contract")
	}
	phaseSelection, ok := exactJSONObject(request["phase_selection"], "scope", "excluded_database_phases", "global_equilibrium_claim_supported")
	excluded, excludedOK := calphadStringList(phaseSelection["excluded_database_phases"], 2048, true)
	availablePhases, _ := calphadStringList(manifest["available_phases"], 2048, false)
	expectedExcluded := make([]string, 0)
	selectedSet := make(map[string]struct{}, len(selectedPhases))
	for _, phase := range selectedPhases {
		selectedSet[phase] = struct{}{}
	}
	for _, phase := range availablePhases {
		if _, selected := selectedSet[phase]; !selected {
			expectedExcluded = append(expectedExcluded, phase)
		}
	}
	expectedScope := "all_database_phases"
	expectedGlobal := true
	if len(expectedExcluded) > 0 {
		expectedScope = "restricted_database_phase_set"
		expectedGlobal = false
	}
	if !ok || !excludedOK || !calphadSameStrings(excluded, expectedExcluded) ||
		phaseSelection["scope"] != expectedScope || phaseSelection["global_equilibrium_claim_supported"] != expectedGlobal {
		return errors.New("CALPHAD equilibrium phase-selection scope is inconsistent")
	}
	closure, ok := exactJSONObject(request["composition_closure"], "grid", "sum", "absolute_tolerance", "units")
	closureSum, closureSumOK := jsonFiniteFloat64(closure["sum"])
	closureTolerance, closureToleranceOK := jsonFiniteFloat64(closure["absolute_tolerance"])
	closureGrid, closureGridOK := closure["grid"].([]any)
	compositionPointCount := expectedGridPoints / (len(temperatures) * len(pressures))
	if !ok || !closureSumOK || closureSum != 1 || !closureToleranceOK || closureTolerance != calphadCompositionTolerance ||
		closure["units"] != "mole_fraction" || !closureGridOK || len(closureGrid) != compositionPointCount {
		return errors.New("CALPHAD equilibrium composition-closure record is invalid")
	}
	for closureIndex, rawClosurePoint := range closureGrid {
		closurePoint, pointOK := calphadFiniteMap(rawClosurePoint, resultComponents, 0, 1)
		if !pointOK {
			return errors.New("CALPHAD equilibrium composition-closure point is invalid")
		}
		axisStride := compositionPointCount
		independentSum := 0.0
		for _, component := range independentNames {
			axis := independentAxes[component]
			axisStride /= len(axis)
			expectedValue := axis[(closureIndex/axisStride)%len(axis)]
			if closurePoint[component] != expectedValue {
				return errors.New("CALPHAD equilibrium composition-closure grid differs from its axes")
			}
			independentSum += expectedValue
		}
		expectedDependent := 1 - independentSum
		if expectedDependent < -calphadCompositionTolerance || expectedDependent > 1+calphadCompositionTolerance ||
			!calphadClose(closurePoint[dependent], math.Min(1, math.Max(0, expectedDependent)), calphadCompositionTolerance) {
			return errors.New("CALPHAD equilibrium composition-closure grid does not close")
		}
		compositionSum := 0.0
		for _, component := range resultComponents {
			compositionSum += closurePoint[component]
		}
		if !calphadClose(compositionSum, 1, calphadCompositionTolerance) {
			return errors.New("CALPHAD equilibrium composition-closure point does not sum to one")
		}
	}
	resultRecord, ok := exactJSONObject(response["result"], "point_count", "dataset_size_bytes", "points", "units")
	if !ok {
		return errors.New("CALPHAD equilibrium scientific result record is invalid")
	}
	pointCount, pointCountOK := jsonInt64(resultRecord["point_count"])
	datasetSize, datasetSizeOK := jsonInt64(resultRecord["dataset_size_bytes"])
	points, pointsOK := resultRecord["points"].([]any)
	if !pointCountOK || pointCount != int64(expectedGridPoints) || !datasetSizeOK || datasetSize <= 0 ||
		datasetSize > 16*1024*1024 || !pointsOK || len(points) != expectedGridPoints {
		return errors.New("CALPHAD equilibrium point inventory is incomplete or outside bounds")
	}
	expectedUnits := map[string]string{
		"T": "K", "P": "Pa", "N": "mol", "X": "mole_fraction", "phase_X": "mole_fraction",
		"bulk_composition_residual": "mole_fraction", "NP": "phase_amount_fraction_at_N_equals_1_mol",
		"GM": "J/mol", "MU": "J/mol", "gibbs_euler_residual": "J/mol",
	}
	units, unitsOK := jsonObject(resultRecord["units"])
	if !unitsOK || len(units) != len(expectedUnits) {
		return errors.New("CALPHAD equilibrium units are incomplete")
	}
	for name, expected := range expectedUnits {
		if units[name] != expected {
			return errors.New("CALPHAD equilibrium units differ from the v2 contract")
		}
	}
	seenGrid := make(map[string]struct{}, expectedGridPoints)
	for _, rawPoint := range points {
		point, ok := exactJSONObject(
			rawPoint, "conditions", "stable_phases", "stable_phase_vertices", "phase_fraction_sum",
			"reconstructed_composition_mole_fraction", "bulk_composition_residual_by_component",
			"maximum_bulk_composition_residual", "GM_J_per_mol", "chemical_potentials_J_per_mol",
			"gibbs_from_chemical_potentials_J_per_mol", "gibbs_euler_residual_J_per_mol",
		)
		if !ok {
			return errors.New("CALPHAD equilibrium point does not match the exact v2 schema")
		}
		pointConditions, ok := exactJSONObject(point["conditions"], "T_K", "P_Pa", "N_mol", "composition_mole_fraction")
		temperature, temperatureOK := jsonFiniteFloat64(pointConditions["T_K"])
		pressure, pressureOK := jsonFiniteFloat64(pointConditions["P_Pa"])
		amount, amountOK := jsonFiniteFloat64(pointConditions["N_mol"])
		temperatureIndex, temperatureFound := calphadIndex(temperatures, temperature, 0)
		pressureIndex, pressureFound := calphadIndex(pressures, pressure, 0)
		composition, compositionOK := calphadFiniteMap(pointConditions["composition_mole_fraction"], resultComponents, 0, 1)
		if !ok || !temperatureOK || !pressureOK || !amountOK || amount != 1 || !temperatureFound ||
			!pressureFound || !compositionOK {
			return errors.New("CALPHAD equilibrium point conditions are invalid")
		}
		compositionSum := 0.0
		gridKey := strconv.Itoa(temperatureIndex) + ":" + strconv.Itoa(pressureIndex)
		for _, component := range resultComponents {
			compositionSum += composition[component]
		}
		if !calphadClose(compositionSum, 1, calphadCompositionTolerance) {
			return errors.New("CALPHAD equilibrium point composition does not close")
		}
		for _, component := range independentNames {
			axisIndex, found := calphadIndex(independentAxes[component], composition[component], calphadCompositionTolerance)
			if !found {
				return errors.New("CALPHAD equilibrium point is outside the requested composition grid")
			}
			gridKey += ":" + strconv.Itoa(axisIndex)
		}
		if _, duplicate := seenGrid[gridKey]; duplicate {
			return errors.New("CALPHAD equilibrium result duplicates a requested grid point")
		}
		seenGrid[gridKey] = struct{}{}
		stablePhases, phasesOK := point["stable_phases"].([]any)
		vertices, verticesOK := point["stable_phase_vertices"].([]any)
		if !phasesOK || len(stablePhases) == 0 || len(stablePhases) > len(selectedPhases) ||
			!verticesOK || len(vertices) == 0 || len(vertices) > 4096 {
			return errors.New("CALPHAD equilibrium stable phase evidence is empty or unbounded")
		}
		phaseAmounts := map[string]float64{}
		previousPhase := ""
		for _, rawPhase := range stablePhases {
			phase, ok := exactJSONObject(rawPhase, "name", "NP_phase_fraction")
			name, nameOK := jsonString(phase["name"])
			amount, amountOK := jsonFiniteFloat64(phase["NP_phase_fraction"])
			if !ok || !nameOK || !amountOK || amount <= 0 || amount > 1+calphadPhaseFractionTolerance ||
				(previousPhase != "" && name <= previousPhase) || !calphadStringSubset([]string{name}, selectedPhases) {
				return errors.New("CALPHAD equilibrium stable phase amount is invalid")
			}
			phaseAmounts[name] = amount
			previousPhase = name
		}
		vertexAmounts := map[string]float64{}
		reconstructed := make(map[string]float64, len(resultComponents))
		seenVertexIndices := map[int64]struct{}{}
		for _, rawVertex := range vertices {
			vertex, ok := exactJSONObject(rawVertex, "vertex_index", "phase", "NP_phase_fraction", "composition_mole_fraction", "composition_sum")
			index, indexOK := jsonInt64(vertex["vertex_index"])
			phase, phaseOK := jsonString(vertex["phase"])
			amount, amountOK := jsonFiniteFloat64(vertex["NP_phase_fraction"])
			vertexComposition, vertexCompositionOK := calphadFiniteMap(vertex["composition_mole_fraction"], resultComponents, 0, 1)
			compositionSum, compositionSumOK := jsonFiniteFloat64(vertex["composition_sum"])
			computedCompositionSum := 0.0
			for _, component := range resultComponents {
				computedCompositionSum += vertexComposition[component]
			}
			if !ok || !indexOK || index < 0 || !phaseOK || !amountOK || amount <= 0 ||
				amount > 1+calphadPhaseFractionTolerance ||
				!vertexCompositionOK || !compositionSumOK ||
				!calphadClose(compositionSum, computedCompositionSum, calphadCompositionTolerance) ||
				!calphadClose(computedCompositionSum, 1, calphadCompositionTolerance) {
				return errors.New("CALPHAD equilibrium stable vertex is invalid")
			}
			if _, exists := seenVertexIndices[index]; exists {
				return errors.New("CALPHAD equilibrium stable vertex index is duplicated")
			}
			seenVertexIndices[index] = struct{}{}
			if _, selected := phaseAmounts[phase]; !selected {
				return errors.New("CALPHAD equilibrium stable vertex phase is not stable")
			}
			vertexAmounts[phase] += amount
			for _, component := range resultComponents {
				reconstructed[component] += amount * vertexComposition[component]
			}
		}
		phaseFractionSum, phaseSumOK := jsonFiniteFloat64(point["phase_fraction_sum"])
		phaseSum := 0.0
		for phase, amount := range phaseAmounts {
			phaseSum += amount
			if !calphadClose(vertexAmounts[phase], amount, calphadBulkResidualTolerance) {
				return errors.New("CALPHAD equilibrium vertex amounts do not reconstruct phase amounts")
			}
		}
		if !phaseSumOK || !calphadClose(phaseFractionSum, 1, calphadPhaseFractionTolerance) ||
			!calphadClose(phaseSum, phaseFractionSum, calphadBulkResidualTolerance) {
			return errors.New("CALPHAD equilibrium phase amounts do not close")
		}
		reportedReconstruction, reconstructionOK := calphadFiniteMap(point["reconstructed_composition_mole_fraction"], resultComponents, 0, 1)
		residuals, residualsOK := calphadFiniteMap(point["bulk_composition_residual_by_component"], resultComponents, 0, calphadBulkResidualTolerance)
		maximumResidual, maximumResidualOK := jsonFiniteFloat64(point["maximum_bulk_composition_residual"])
		computedMaximum := 0.0
		if !reconstructionOK || !residualsOK || !maximumResidualOK {
			return errors.New("CALPHAD equilibrium composition reconstruction is invalid")
		}
		for _, component := range resultComponents {
			computedResidual := math.Abs(reconstructed[component] - composition[component])
			if computedResidual > computedMaximum {
				computedMaximum = computedResidual
			}
			if !calphadClose(reportedReconstruction[component], reconstructed[component], calphadBulkResidualTolerance) ||
				!calphadClose(residuals[component], computedResidual, calphadBulkResidualTolerance) {
				return errors.New("CALPHAD equilibrium reported composition residual is inconsistent")
			}
		}
		if maximumResidual < 0 || maximumResidual > calphadBulkResidualTolerance ||
			!calphadClose(maximumResidual, computedMaximum, calphadBulkResidualTolerance) {
			return errors.New("CALPHAD equilibrium maximum composition residual is invalid")
		}
		gm, gmOK := jsonFiniteFloat64(point["GM_J_per_mol"])
		chemicalPotentials, chemicalPotentialsOK := calphadFiniteMap(point["chemical_potentials_J_per_mol"], resultComponents, -math.MaxFloat64, math.MaxFloat64)
		gibbsFromMu, gibbsFromMuOK := jsonFiniteFloat64(point["gibbs_from_chemical_potentials_J_per_mol"])
		gibbsResidual, gibbsResidualOK := jsonFiniteFloat64(point["gibbs_euler_residual_J_per_mol"])
		computedGibbs := 0.0
		if !gmOK || !chemicalPotentialsOK || !gibbsFromMuOK || !gibbsResidualOK {
			return errors.New("CALPHAD equilibrium Gibbs/chemical-potential evidence is invalid")
		}
		for _, component := range resultComponents {
			computedGibbs += composition[component] * chemicalPotentials[component]
		}
		computedGibbsResidual := math.Abs(gm - computedGibbs)
		if !calphadClose(gibbsFromMu, computedGibbs, calphadGibbsEulerTolerance) ||
			!calphadClose(gibbsResidual, computedGibbsResidual, calphadGibbsEulerTolerance) ||
			gibbsResidual < 0 || gibbsResidual > calphadGibbsEulerTolerance {
			return errors.New("CALPHAD equilibrium Gibbs-Euler relation is inconsistent")
		}
	}
	if len(seenGrid) != expectedGridPoints {
		return errors.New("CALPHAD equilibrium result does not cover the requested grid")
	}
	warnings, warningsOK := response["warnings"].([]any)
	if !warningsOK || len(warnings) == 0 || len(warnings) > 1024 {
		return errors.New("CALPHAD equilibrium warning record is invalid")
	}
	warningSet := make(map[string]struct{}, len(warnings))
	previousWarning := ""
	for _, rawWarning := range warnings {
		warning, ok := jsonString(rawWarning)
		if !ok || strings.TrimSpace(warning) == "" || len(warning) > 4096 ||
			(previousWarning != "" && warning <= previousWarning) {
			return errors.New("CALPHAD equilibrium warning is invalid")
		}
		warningSet[warning] = struct{}{}
		previousWarning = warning
	}
	requiredWarnings := []string{
		"A successful numerical solve does not independently validate the database assessment or extrapolation domain.",
		"NP is a phase-amount fraction on the fixed N=1 mol calculation basis.",
	}
	if !expectedGlobal {
		requiredWarnings = append(requiredWarnings,
			"The requested phase set excludes database phases; this is a restricted-phase calculation and must not be presented as global equilibrium.",
		)
	}
	if manifestWarnings, ok := manifest["warnings"].([]any); ok {
		for _, rawWarning := range manifestWarnings {
			if warning, ok := jsonString(rawWarning); ok && warning != "" {
				requiredWarnings = append(requiredWarnings, warning)
			}
		}
	}
	for _, requiredWarning := range requiredWarnings {
		if _, found := warningSet[requiredWarning]; !found {
			return errors.New("CALPHAD equilibrium warning record omits a required scope or solver caveat")
		}
	}
	evidence, ok := exactJSONObject(
		response["evidence"], "sha256", "algorithm", "canonicalization",
		"canonical_serialization", "solver_replay_determinism_claimed",
	)
	evidenceSHA, shaOK := jsonString(evidence["sha256"])
	manifestSHA, manifestSHAOK := jsonString(manifest["manifest_sha256"])
	if !ok || !shaOK || !calphadEvidenceSHA256Pattern.MatchString(evidenceSHA) ||
		!manifestSHAOK || !calphadEvidenceSHA256Pattern.MatchString(manifestSHA) ||
		evidence["algorithm"] != "sha256" || evidence["canonical_serialization"] != true ||
		evidence["solver_replay_determinism_claimed"] != false ||
		evidence["canonicalization"] != "UTF-8 JSON, sorted keys, compact separators, finite numbers" {
		return errors.New("CALPHAD equilibrium canonical evidence record is invalid")
	}
	canonicalPayload := map[string]any{
		"schema_version":           calphadEquilibriumSchemaVersion,
		"database_sha256":          manifest["sha256"],
		"database_manifest_sha256": manifest["manifest_sha256"],
		"request":                  request,
		"result":                   resultRecord,
		"warnings":                 warnings,
		"pycalphad_version":        manifest["pycalphad_version"],
	}
	canonical, err := calphadCanonicalJSON(canonicalPayload)
	if err != nil {
		return errors.New("CALPHAD equilibrium canonical evidence could not be reconstructed")
	}
	digest := sha256.Sum256(canonical)
	if evidenceSHA != hex.EncodeToString(digest[:]) {
		return errors.New("CALPHAD equilibrium canonical evidence SHA-256 is inconsistent")
	}
	return nil
}

type calphadNullableSeries struct {
	values  []float64
	present []bool
}

func calphadScheilSeries(value any, count int, allowNull bool) (calphadNullableSeries, bool) {
	rawValues, ok := value.([]any)
	if !ok || len(rawValues) != count {
		return calphadNullableSeries{}, false
	}
	series := calphadNullableSeries{
		values:  make([]float64, count),
		present: make([]bool, count),
	}
	for index, rawValue := range rawValues {
		if rawValue == nil && allowNull {
			continue
		}
		number, numberOK := jsonFiniteFloat64(rawValue)
		if !numberOK {
			return calphadNullableSeries{}, false
		}
		series.values[index] = number
		series.present[index] = true
	}
	return series, true
}

func validateCalphadScheilResult(result, evidenceRequest, database map[string]any, pycalphadVersion string) error {
	response, ok := exactJSONObject(
		result,
		"schema_version", "method", "database", "request", "result", "assumptions",
		"warnings", "solver", "units", "limits", "evidence",
	)
	if !ok || response["schema_version"] != calphadScheilSchemaVersion ||
		response["method"] != "Scheil-Gulliver" {
		return errors.New("CALPHAD Scheil result does not match the exact v1 schema")
	}

	selection, selectionOK := exactJSONObject(evidenceRequest["selection"], "components", "phases")
	selectedComponents, componentsOK := calphadStringList(selection["components"], 32, false)
	selectedPhases, phasesOK := calphadStringList(selection["phases"], 128, false)
	if !selectionOK || !componentsOK || !phasesOK {
		return errors.New("CALPHAD Scheil selection is invalid")
	}
	physicalComponents := make([]string, 0, len(selectedComponents))
	for _, component := range selectedComponents {
		if component != "VA" && component != "/-" {
			physicalComponents = append(physicalComponents, component)
		}
	}
	if len(physicalComponents) == 0 || !calphadStringSubset([]string{"LIQUID"}, selectedPhases) {
		return errors.New("CALPHAD Scheil selection lacks physical components or LIQUID")
	}

	manifest, ok := jsonObject(response["database"])
	if !ok {
		return errors.New("CALPHAD Scheil result omits its database manifest")
	}
	if err := validateCalphadInspectionManifest(
		manifest, database, pycalphadVersion, selectedComponents, selectedPhases,
	); err != nil {
		return fmt.Errorf("CALPHAD Scheil database manifest is invalid: %w", err)
	}

	typedConditions, typedConditionsOK := exactJSONObject(
		evidenceRequest["conditions"],
		"independent_composition_mole_fraction", "start_temperature_K",
		"step_temperature_K", "pressure_Pa", "stop_liquid_fraction",
	)
	requestRecord, requestOK := exactJSONObject(
		response["request"],
		"components", "phases", "independent_composition_mole_fraction",
		"bulk_composition_mole_fraction", "dependent_component", "start_temperature_K",
		"step_temperature_K", "pressure_Pa", "total_amount_mol", "liquid_phase_name",
		"stop_liquid_fraction",
	)
	if !typedConditionsOK || !requestOK {
		return errors.New("CALPHAD Scheil typed or runtime request record is invalid")
	}
	requestComponents, requestComponentsOK := calphadStringList(requestRecord["components"], 32, false)
	requestPhases, requestPhasesOK := calphadStringList(requestRecord["phases"], 128, false)
	if !requestComponentsOK || !requestPhasesOK ||
		!calphadSameStrings(requestComponents, selectedComponents) ||
		!calphadSameStrings(requestPhases, selectedPhases) {
		return errors.New("CALPHAD Scheil runtime selection differs from its typed request")
	}

	typedIndependent, typedIndependentOK := jsonObject(typedConditions["independent_composition_mole_fraction"])
	runtimeIndependent, runtimeIndependentOK := jsonObject(requestRecord["independent_composition_mole_fraction"])
	if !typedIndependentOK || !runtimeIndependentOK ||
		len(typedIndependent) != len(physicalComponents)-1 ||
		len(runtimeIndependent) != len(physicalComponents)-1 {
		return errors.New("CALPHAD Scheil independent composition record is invalid")
	}
	bulkComposition := make(map[string]float64, len(physicalComponents))
	independentSum := 0.0
	for _, component := range physicalComponents[1:] {
		typedValue, typedExists := typedIndependent[component]
		runtimeValue, runtimeExists := runtimeIndependent[component]
		typedFraction, typedOK := jsonFiniteFloat64(typedValue)
		runtimeFraction, runtimeOK := jsonFiniteFloat64(runtimeValue)
		if !typedExists || !runtimeExists || !typedOK || !runtimeOK ||
			typedFraction < 0 || typedFraction > 1 || runtimeFraction != typedFraction {
			return errors.New("CALPHAD Scheil independent composition differs from its typed request")
		}
		independentSum += typedFraction
		bulkComposition[component] = typedFraction
	}
	dependent := physicalComponents[0]
	dependentFraction := 1 - independentSum
	if independentSum > 1+calphadCompositionTolerance ||
		dependentFraction < -calphadCompositionTolerance {
		return errors.New("CALPHAD Scheil independent composition does not close")
	}
	dependentFraction = math.Min(1, math.Max(0, dependentFraction))
	bulkComposition[dependent] = dependentFraction
	runtimeBulk, runtimeBulkOK := calphadFiniteMap(
		requestRecord["bulk_composition_mole_fraction"], physicalComponents, 0, 1,
	)
	if !runtimeBulkOK {
		return errors.New("CALPHAD Scheil bulk-composition record is invalid")
	}
	bulkSum := 0.0
	for _, component := range physicalComponents {
		bulkSum += runtimeBulk[component]
		if !calphadClose(runtimeBulk[component], bulkComposition[component], calphadCompositionTolerance) {
			return errors.New("CALPHAD Scheil bulk composition differs from the canonical closure")
		}
	}
	if !calphadClose(bulkSum, 1, calphadCompositionTolerance) {
		return errors.New("CALPHAD Scheil bulk composition does not sum to one")
	}
	requestDependent, dependentOK := jsonString(requestRecord["dependent_component"])
	typedStart, typedStartOK := jsonFiniteFloat64(typedConditions["start_temperature_K"])
	runtimeStart, runtimeStartOK := jsonFiniteFloat64(requestRecord["start_temperature_K"])
	typedStep, typedStepOK := jsonFiniteFloat64(typedConditions["step_temperature_K"])
	runtimeStep, runtimeStepOK := jsonFiniteFloat64(requestRecord["step_temperature_K"])
	typedPressure, typedPressureOK := jsonFiniteFloat64(typedConditions["pressure_Pa"])
	runtimePressure, runtimePressureOK := jsonFiniteFloat64(requestRecord["pressure_Pa"])
	typedStop, typedStopOK := jsonFiniteFloat64(typedConditions["stop_liquid_fraction"])
	runtimeStop, runtimeStopOK := jsonFiniteFloat64(requestRecord["stop_liquid_fraction"])
	totalAmount, totalAmountOK := jsonFiniteFloat64(requestRecord["total_amount_mol"])
	liquidPhase, liquidPhaseOK := jsonString(requestRecord["liquid_phase_name"])
	if !dependentOK || requestDependent != dependent ||
		!typedStartOK || !runtimeStartOK || typedStart != runtimeStart ||
		!typedStepOK || !runtimeStepOK || typedStep != runtimeStep ||
		!typedPressureOK || !runtimePressureOK || typedPressure != domain.CalphadReferencePressurePa ||
		runtimePressure != typedPressure || !typedStopOK || !runtimeStopOK || typedStop != runtimeStop ||
		!totalAmountOK || totalAmount != 1 || !liquidPhaseOK || liquidPhase != "LIQUID" {
		return errors.New("CALPHAD Scheil scalar request record differs from its typed request")
	}
	assessmentTemperatureLimits, assessmentTemperatureLimitsOK := calphadNumberList(
		manifest["assessment_temperature_limits_K"], 2, 1, 10_000, false,
	)
	assessmentPressureLimits, assessmentPressureLimitsOK := calphadBoundedNumberList(
		manifest[domain.CalphadAssessmentPressureLimitsMetadataKey], 2,
		domain.CalphadMinimumPressurePa, domain.CalphadMaximumPressurePa, false,
	)
	if !assessmentTemperatureLimitsOK || len(assessmentTemperatureLimits) != 2 ||
		!assessmentPressureLimitsOK || len(assessmentPressureLimits) != 2 ||
		typedStart < assessmentTemperatureLimits[0] || typedStart > assessmentTemperatureLimits[1] ||
		typedPressure < assessmentPressureLimits[0] || typedPressure > assessmentPressureLimits[1] {
		return errors.New("CALPHAD Scheil request is outside the declared assessment limits")
	}

	assumptions, assumptionsOK := response["assumptions"].([]any)
	expectedAssumptions := []string{
		"Perfect mixing (infinite diffusion) in the liquid.",
		"Local equilibrium at the solid/liquid interface.",
		"No diffusion in solid phases after they form.",
		"Constant pressure of 101325 Pa and a one-mole calculation basis.",
	}
	if !assumptionsOK || len(assumptions) != len(expectedAssumptions) {
		return errors.New("CALPHAD Scheil assumption record is invalid")
	}
	for index, expected := range expectedAssumptions {
		observed, observedOK := jsonString(assumptions[index])
		if !observedOK || observed != expected {
			return errors.New("CALPHAD Scheil assumption record differs from the reviewed model")
		}
	}
	warnings, warningsOK := response["warnings"].([]any)
	if !warningsOK || len(warnings) == 0 || len(warnings) > 1024 {
		return errors.New("CALPHAD Scheil warning record is invalid")
	}
	requiredWarnings := map[string]bool{
		"This path is not a back-diffusion, finite-rate diffusion, precipitation, or phase-field calculation.": false,
		"A converged numerical path does not validate the thermodynamic assessment or extrapolation domain.":   false,
	}
	previousWarning := ""
	for _, rawWarning := range warnings {
		warning, warningOK := jsonString(rawWarning)
		if !warningOK || strings.TrimSpace(warning) == "" || len(warning) > 4096 ||
			(previousWarning != "" && warning <= previousWarning) {
			return errors.New("CALPHAD Scheil warnings are empty, duplicated, unsorted, or unbounded")
		}
		if _, required := requiredWarnings[warning]; required {
			requiredWarnings[warning] = true
		}
		previousWarning = warning
	}
	for _, found := range requiredWarnings {
		if !found {
			return errors.New("CALPHAD Scheil warnings omit a required model-boundary caveat")
		}
	}

	solver, ok := exactJSONObject(
		response["solver"], "name", "version", "pycalphad_version",
		"adaptive_constitution_sampling", "replay_determinism_claimed",
	)
	if !ok || solver["name"] != "scheil" || solver["version"] != domain.CalphadScheilVersion ||
		solver["pycalphad_version"] != pycalphadVersion ||
		solver["adaptive_constitution_sampling"] != true ||
		solver["replay_determinism_claimed"] != false {
		return errors.New("CALPHAD Scheil solver identity differs from the qualified runtime")
	}
	units, ok := exactJSONObject(
		response["units"], "temperature", "pressure", "amount", "composition", "phase_fraction",
	)
	if !ok || units["temperature"] != "K" || units["pressure"] != "Pa" ||
		units["amount"] != "mol" || units["composition"] != "mole_fraction" ||
		units["phase_fraction"] != "fraction_of_one_mole_basis" {
		return errors.New("CALPHAD Scheil units differ from the reviewed contract")
	}
	limits, ok := exactJSONObject(response["limits"], "max_steps", "wall_time_seconds", "max_result_bytes")
	maxSteps, maxStepsOK := jsonInt64(limits["max_steps"])
	wallTime, wallTimeOK := jsonFiniteFloat64(limits["wall_time_seconds"])
	maxResultBytes, maxResultBytesOK := jsonInt64(limits["max_result_bytes"])
	if !ok || !maxStepsOK || maxSteps != calphadScheilMaximumSteps ||
		!wallTimeOK || wallTime != 30 || !maxResultBytesOK || maxResultBytes != 16*1024*1024 {
		return errors.New("CALPHAD Scheil limits differ from the reviewed contract")
	}

	resultRecord, ok := exactJSONObject(
		response["result"],
		"point_count", "temperatures_K", "fraction_solid", "fraction_liquid",
		"solid_phase_increment_fraction", "solid_phase_cumulative_fraction",
		"phase_composition_mole_fraction", "elemental_mass_balance", "converged",
		"qualified_terminal_point", "discarded_upstream_terminal_fill_point", "closure_tolerances",
	)
	if !ok {
		return errors.New("CALPHAD Scheil scientific result record is invalid")
	}
	pointCount, pointCountOK := jsonInt64(resultRecord["point_count"])
	if !pointCountOK || pointCount < 2 || pointCount > calphadScheilMaximumSteps {
		return errors.New("CALPHAD Scheil point count is outside its fixed bound")
	}
	count := int(pointCount)
	temperatures, temperaturesOK := calphadScheilSeries(resultRecord["temperatures_K"], count, false)
	fractionSolid, fractionSolidOK := calphadScheilSeries(resultRecord["fraction_solid"], count, false)
	fractionLiquid, fractionLiquidOK := calphadScheilSeries(resultRecord["fraction_liquid"], count, false)
	if !temperaturesOK || !fractionSolidOK || !fractionLiquidOK {
		return errors.New("CALPHAD Scheil retained paths are incomplete or non-finite")
	}
	for index := 0; index < count; index++ {
		temperature := temperatures.values[index]
		solid := fractionSolid.values[index]
		liquid := fractionLiquid.values[index]
		if temperature < assessmentTemperatureLimits[0] || temperature > assessmentTemperatureLimits[1] ||
			solid < -calphadBulkResidualTolerance || solid > 1+calphadBulkResidualTolerance ||
			liquid < -calphadBulkResidualTolerance || liquid > 1+calphadBulkResidualTolerance ||
			!calphadClose(solid+liquid, 1, calphadBulkResidualTolerance) {
			return errors.New("CALPHAD Scheil retained path violates assessment or fraction closure")
		}
		if index > 0 && (temperature > temperatures.values[index-1]+1e-10 ||
			solid+calphadBulkResidualTolerance < fractionSolid.values[index-1]) {
			return errors.New("CALPHAD Scheil temperature or solid fraction is not monotonic")
		}
	}
	if !calphadClose(fractionSolid.values[0], 0, calphadBulkResidualTolerance) ||
		!calphadClose(fractionLiquid.values[0], 1, calphadBulkResidualTolerance) ||
		fractionLiquid.values[count-1] >= typedStop+calphadBulkResidualTolerance ||
		resultRecord["converged"] != true ||
		resultRecord["qualified_terminal_point"] != "last_residual_liquid_point" {
		return errors.New("CALPHAD Scheil result does not satisfy its initial or terminal criterion")
	}
	if _, discardedOK := resultRecord["discarded_upstream_terminal_fill_point"].(bool); !discardedOK {
		return errors.New("CALPHAD Scheil terminal-fill disclosure is invalid")
	}
	closureTolerances, ok := exactJSONObject(
		resultRecord["closure_tolerances"],
		"phase_fraction_absolute", "composition_absolute", "elemental_mass_balance_absolute",
	)
	phaseTolerance, phaseToleranceOK := jsonFiniteFloat64(closureTolerances["phase_fraction_absolute"])
	compositionTolerance, compositionToleranceOK := jsonFiniteFloat64(closureTolerances["composition_absolute"])
	massTolerance, massToleranceOK := jsonFiniteFloat64(closureTolerances["elemental_mass_balance_absolute"])
	if !ok || !phaseToleranceOK || phaseTolerance != calphadScheilClosureTolerance ||
		!compositionToleranceOK || compositionTolerance != calphadScheilClosureTolerance ||
		!massToleranceOK || massTolerance != calphadScheilClosureTolerance {
		return errors.New("CALPHAD Scheil closure tolerances differ from the reviewed contract")
	}

	incrementObject, incrementsOK := jsonObject(resultRecord["solid_phase_increment_fraction"])
	cumulativeObject, cumulativeOK := jsonObject(resultRecord["solid_phase_cumulative_fraction"])
	compositionObject, compositionsOK := jsonObject(resultRecord["phase_composition_mole_fraction"])
	if !incrementsOK || len(incrementObject) == 0 || !cumulativeOK ||
		len(cumulativeObject) != len(incrementObject) || !compositionsOK {
		return errors.New("CALPHAD Scheil phase inventory is empty or inconsistent")
	}
	incrementSeries := make(map[string][]float64, len(incrementObject))
	cumulativeSeries := make(map[string][]float64, len(cumulativeObject))
	for phase, rawIncrement := range incrementObject {
		if phase == "LIQUID" || !calphadEvidenceNamePattern.MatchString(phase) ||
			!calphadStringSubset([]string{phase}, selectedPhases) {
			return errors.New("CALPHAD Scheil solid phase is invalid or unrequested")
		}
		rawCumulative, exists := cumulativeObject[phase]
		increments, incrementsOK := calphadScheilSeries(rawIncrement, count, false)
		cumulative, cumulativeOK := calphadScheilSeries(rawCumulative, count, false)
		if !exists || !incrementsOK || !cumulativeOK ||
			!calphadClose(increments.values[0], 0, calphadBulkResidualTolerance) {
			return errors.New("CALPHAD Scheil phase increment/cumulative path is invalid")
		}
		running := 0.0
		for index := 0; index < count; index++ {
			increment := increments.values[index]
			cumulativeValue := cumulative.values[index]
			if increment < -calphadBulkResidualTolerance || cumulativeValue < -calphadBulkResidualTolerance {
				return errors.New("CALPHAD Scheil phase amount is negative")
			}
			running += increment
			if !calphadClose(running, cumulativeValue, calphadScheilClosureTolerance) {
				return errors.New("CALPHAD Scheil cumulative phase amount does not reconstruct from increments")
			}
		}
		incrementSeries[phase] = increments.values
		cumulativeSeries[phase] = cumulative.values
	}
	for phase := range cumulativeObject {
		if _, exists := incrementObject[phase]; !exists {
			return errors.New("CALPHAD Scheil cumulative phase set differs from increments")
		}
	}
	for index, solid := range fractionSolid.values {
		cumulativeSum := 0.0
		for _, values := range cumulativeSeries {
			cumulativeSum += values[index]
		}
		if !calphadClose(cumulativeSum, solid, calphadScheilClosureTolerance) {
			return errors.New("CALPHAD Scheil cumulative solid phases do not close")
		}
	}

	type phaseCompositionRecord map[string]calphadNullableSeries
	phaseCompositions := make(map[string]phaseCompositionRecord, len(compositionObject))
	for phase, rawComponents := range compositionObject {
		if !calphadEvidenceNamePattern.MatchString(phase) ||
			!calphadStringSubset([]string{phase}, selectedPhases) {
			return errors.New("CALPHAD Scheil phase-composition phase is invalid or unrequested")
		}
		componentsObject, componentsObjectOK := jsonObject(rawComponents)
		if !componentsObjectOK || len(componentsObject) != len(physicalComponents) {
			return errors.New("CALPHAD Scheil phase composition lacks a physical component")
		}
		componentSeries := make(phaseCompositionRecord, len(physicalComponents))
		for _, component := range physicalComponents {
			rawSeries, exists := componentsObject[component]
			series, seriesOK := calphadScheilSeries(rawSeries, count, true)
			if !exists || !seriesOK {
				return errors.New("CALPHAD Scheil phase composition lacks a complete retained series")
			}
			componentSeries[component] = series
		}
		for index := 0; index < count; index++ {
			presentCount := 0
			compositionSum := 0.0
			for _, component := range physicalComponents {
				series := componentSeries[component]
				if series.present[index] {
					value := series.values[index]
					if value < -calphadBulkResidualTolerance || value > 1+calphadBulkResidualTolerance {
						return errors.New("CALPHAD Scheil phase composition is outside [0, 1]")
					}
					presentCount++
					compositionSum += value
				}
			}
			if presentCount != 0 && presentCount != len(physicalComponents) {
				return errors.New("CALPHAD Scheil phase composition is partially missing")
			}
			if presentCount > 0 && !calphadClose(compositionSum, 1, calphadScheilClosureTolerance) {
				return errors.New("CALPHAD Scheil phase composition does not close")
			}
		}
		phaseCompositions[phase] = componentSeries
	}
	liquidCompositions, liquidFound := phaseCompositions["LIQUID"]
	if !liquidFound {
		return errors.New("CALPHAD Scheil result lacks LIQUID compositions")
	}
	for phase := range incrementSeries {
		if _, found := phaseCompositions[phase]; !found {
			return errors.New("CALPHAD Scheil solid increment lacks phase-composition evidence")
		}
	}

	runningSolidInventory := make(map[string]float64, len(physicalComponents))
	maximumErrors := make(map[string]float64, len(physicalComponents))
	finalReconstructed := make(map[string]float64, len(physicalComponents))
	for index := 0; index < count; index++ {
		if index > 0 {
			for phase, increments := range incrementSeries {
				increment := increments[index]
				for _, component := range physicalComponents {
					series := phaseCompositions[phase][component]
					if !series.present[index] {
						if increment > calphadBulkResidualTolerance {
							return errors.New("CALPHAD Scheil positive solid increment lacks composition evidence")
						}
						continue
					}
					runningSolidInventory[component] += increment * series.values[index]
				}
			}
		}
		reconstructedSum := 0.0
		for _, component := range physicalComponents {
			reconstructed := bulkComposition[component]
			if index > 0 {
				liquidSeries := liquidCompositions[component]
				if !liquidSeries.present[index] {
					return errors.New("CALPHAD Scheil residual liquid lacks composition evidence")
				}
				reconstructed = fractionLiquid.values[index]*liquidSeries.values[index] +
					runningSolidInventory[component]
			}
			errorValue := math.Abs(reconstructed - bulkComposition[component])
			if errorValue > calphadScheilClosureTolerance {
				return errors.New("CALPHAD Scheil elemental inventory does not close")
			}
			if errorValue > maximumErrors[component] {
				maximumErrors[component] = errorValue
			}
			finalReconstructed[component] = reconstructed
			reconstructedSum += reconstructed
		}
		if !calphadClose(reconstructedSum, 1, calphadScheilClosureTolerance) {
			return errors.New("CALPHAD Scheil reconstructed elemental inventory does not sum to one")
		}
	}

	massBalance, ok := exactJSONObject(
		resultRecord["elemental_mass_balance"],
		"basis", "formula", "absolute_tolerance", "maximum_absolute_component_error",
		"maximum_absolute_error_by_component", "final_reconstructed_bulk_composition_mole_fraction",
		"all_retained_points_closed",
	)
	reportedErrors, reportedErrorsOK := jsonObject(massBalance["maximum_absolute_error_by_component"])
	reportedFinal, reportedFinalOK := jsonObject(massBalance["final_reconstructed_bulk_composition_mole_fraction"])
	reportedMaximum, reportedMaximumOK := jsonFiniteFloat64(massBalance["maximum_absolute_component_error"])
	reportedTolerance, reportedToleranceOK := jsonFiniteFloat64(massBalance["absolute_tolerance"])
	expectedFormula := "bulk_x[c] = fraction_liquid[i] * liquid_x[c,i] + sum_phase,sum_step<=i(solid_increment[phase,step] * solid_x[phase,c,step])"
	computedMaximum := 0.0
	for _, component := range physicalComponents {
		if maximumErrors[component] > computedMaximum {
			computedMaximum = maximumErrors[component]
		}
	}
	if !ok || massBalance["basis"] != "one_mole_initial_bulk" ||
		massBalance["formula"] != expectedFormula || !reportedToleranceOK ||
		reportedTolerance != calphadScheilClosureTolerance ||
		massBalance["all_retained_points_closed"] != true || !reportedMaximumOK ||
		!reportedErrorsOK || len(reportedErrors) != len(physicalComponents) ||
		!reportedFinalOK || len(reportedFinal) != len(physicalComponents) ||
		!calphadClose(reportedMaximum, computedMaximum, 1e-12) {
		return errors.New("CALPHAD Scheil elemental mass-balance summary is invalid")
	}
	for _, component := range physicalComponents {
		reportedError, errorOK := jsonFiniteFloat64(reportedErrors[component])
		reportedComposition, compositionOK := jsonFiniteFloat64(reportedFinal[component])
		if !errorOK || !compositionOK ||
			!calphadClose(reportedError, maximumErrors[component], 1e-12) ||
			!calphadClose(reportedComposition, finalReconstructed[component], 1e-12) {
			return errors.New("CALPHAD Scheil reported elemental mass balance differs from reconstruction")
		}
	}

	evidence, ok := exactJSONObject(response["evidence"], "sha256", "algorithm", "canonicalization")
	evidenceSHA, evidenceSHAOK := jsonString(evidence["sha256"])
	if !ok || !evidenceSHAOK || !calphadEvidenceSHA256Pattern.MatchString(evidenceSHA) ||
		evidence["algorithm"] != "sha256" ||
		evidence["canonicalization"] != "UTF-8 JSON, sorted keys, compact separators, finite numbers" {
		return errors.New("CALPHAD Scheil canonical evidence record is invalid")
	}
	evidencePayload := make(map[string]any, len(response)-1)
	for key, value := range response {
		if key != "evidence" {
			evidencePayload[key] = value
		}
	}
	canonical, err := calphadCanonicalJSON(evidencePayload)
	if err != nil {
		return errors.New("CALPHAD Scheil canonical evidence could not be reconstructed")
	}
	digest := sha256.Sum256(canonical)
	if evidenceSHA != hex.EncodeToString(digest[:]) {
		return errors.New("CALPHAD Scheil canonical evidence SHA-256 is inconsistent")
	}
	return nil
}
