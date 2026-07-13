package httpapi

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"regexp"
	"strings"
	"unicode/utf8"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

var (
	calphadEvidenceSHA256Pattern = regexp.MustCompile(`^[0-9a-f]{64}$`)
	calphadEvidenceImagePattern  = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
	calphadEvidenceNamePattern   = regexp.MustCompile(`^[A-Z0-9_+./:-]{1,128}$`)
)

const (
	calphadToolEvidenceSchemaVersion     = "ultra.calphad.tool-evidence.v3"
	calphadEquilibriumSchemaVersion      = "ultra.calphad.equilibrium.v2"
	maxCalphadRawEvidenceBytes       int = 32 << 20
	maxCalphadGzipEvidenceBytes      int = 11 << 20
	maxCalphadJSONDepth                  = 64
)

type verifiedCalphadEvidence struct {
	ResourceID                 string
	DatabaseID                 string
	DatabaseSHA256             string
	DatabaseSizeBytes          int64
	DatabaseFormat             string
	DatabaseInventorySHA256    string
	RequestSHA256              string
	Source                     string
	LicenseID                  string
	AssessmentScope            string
	ReferenceState             string
	TemperatureLimitsK         [2]float64
	AssessmentPressureLimitsPa [2]float64
	Operation                  string
	Status                     string
	FailureDomain              domain.CalphadFailureDomain
	FailureStage               domain.CalphadFailureStage
	FailureCode                domain.CalphadFailureCode
	EvidencePath               string
	EvidenceSHA256             string
	EvidenceSizeBytes          int64
	RuntimeImageID             string
	PycalphadVersion           string
	InspectionEvidenceSHA256   string
	EvidenceBytes              []byte
}

func decodeCalphadValidationRequest(
	w http.ResponseWriter,
	r *http.Request,
	target *appendCalphadValidationRequest,
) bool {
	defer r.Body.Close()
	payload, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxJSONBodyBytes))
	if err != nil {
		var maxBytesErr *http.MaxBytesError
		if errors.As(err, &maxBytesErr) {
			writeError(w, http.StatusRequestEntityTooLarge, fmt.Errorf("JSON request body exceeds %d bytes", maxJSONBodyBytes))
			return false
		}
		writeError(w, http.StatusBadRequest, err)
		return false
	}
	decoded, err := decodeUniqueJSON(payload)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return false
	}
	object, ok := exactJSONObject(
		decoded, "status", "operation", "evidence_path", "evidence_sha256",
		"evidence_size_bytes", "runtime_image_id", "pycalphad_version", "evidence_gzip_base64",
	)
	if !ok {
		candidate, candidateOK := jsonObject(decoded)
		candidateStatus, statusOK := jsonString(candidate["status"])
		if candidateOK && statusOK && calphadFailureStatus(candidateStatus) {
			object, ok = exactJSONObject(
				decoded, "status", "operation", "failure_domain", "failure_stage", "failure_code",
				"evidence_path", "evidence_sha256", "evidence_size_bytes", "runtime_image_id",
				"pycalphad_version", "evidence_gzip_base64",
			)
		}
	}
	if !ok {
		writeError(w, http.StatusBadRequest, errors.New("CALPHAD validation request schema is invalid"))
		return false
	}
	status, statusOK := jsonString(object["status"])
	operation, operationOK := jsonString(object["operation"])
	evidencePath, pathOK := jsonString(object["evidence_path"])
	evidenceSHA, shaOK := jsonString(object["evidence_sha256"])
	evidenceSize, sizeOK := jsonInt64(object["evidence_size_bytes"])
	runtimeImage, imageOK := jsonString(object["runtime_image_id"])
	pycalphadVersion, versionOK := jsonString(object["pycalphad_version"])
	encodedEvidence, evidenceOK := jsonString(object["evidence_gzip_base64"])
	failureDomain, failureDomainOK := "", true
	failureStage, failureStageOK := "", true
	failureCode, failureCodeOK := "", true
	if calphadFailureStatus(status) {
		failureDomain, failureDomainOK = jsonString(object["failure_domain"])
		failureStage, failureStageOK = jsonString(object["failure_stage"])
		failureCode, failureCodeOK = jsonString(object["failure_code"])
	}
	if !statusOK || !operationOK || !pathOK || !shaOK || !sizeOK || !imageOK ||
		!versionOK || !evidenceOK || !failureDomainOK || !failureStageOK || !failureCodeOK {
		writeError(w, http.StatusBadRequest, errors.New("CALPHAD validation request field types are invalid"))
		return false
	}
	*target = appendCalphadValidationRequest{
		Status: status, Operation: operation, EvidencePath: evidencePath,
		FailureDomain: failureDomain, FailureStage: failureStage, FailureCode: failureCode,
		EvidenceSHA256: evidenceSHA, EvidenceSizeBytes: evidenceSize,
		RuntimeImageID: runtimeImage, PycalphadVersion: pycalphadVersion,
		EvidenceGzipBase64: encodedEvidence,
	}
	return true
}

func calphadFailureStatus(status string) bool {
	return status == "failed" || status == "timeout" || status == "unsupported"
}

func decodeBoundedCalphadEvidence(encoded string) ([]byte, error) {
	if encoded == "" || strings.TrimSpace(encoded) != encoded ||
		strings.ContainsAny(encoded, "\r\n\t ") ||
		len(encoded) > base64.StdEncoding.EncodedLen(maxCalphadGzipEvidenceBytes) {
		return nil, errors.New("CALPHAD evidence encoding is outside its fixed bound")
	}
	compressed, err := base64.StdEncoding.Strict().DecodeString(encoded)
	if err != nil || len(compressed) == 0 || len(compressed) > maxCalphadGzipEvidenceBytes {
		return nil, errors.New("CALPHAD evidence is not bounded canonical base64")
	}
	compressedReader := bytes.NewReader(compressed)
	gzipReader, err := gzip.NewReader(compressedReader)
	if err != nil {
		return nil, errors.New("CALPHAD evidence is not a gzip stream")
	}
	gzipReader.Multistream(false)
	raw, readErr := io.ReadAll(io.LimitReader(gzipReader, int64(maxCalphadRawEvidenceBytes)+1))
	closeErr := gzipReader.Close()
	if readErr != nil || closeErr != nil || len(raw) == 0 || len(raw) > maxCalphadRawEvidenceBytes {
		return nil, errors.New("CALPHAD evidence decompression failed or exceeded its fixed bound")
	}
	if compressedReader.Len() != 0 {
		return nil, errors.New("CALPHAD evidence contains trailing or concatenated gzip data")
	}
	return raw, nil
}

func decodeUniqueJSON(payload []byte) (any, error) {
	if !utf8.Valid(payload) {
		return nil, errors.New("JSON evidence is not valid UTF-8")
	}
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.UseNumber()
	value, err := decodeUniqueJSONValue(decoder, 0)
	if err != nil {
		return nil, err
	}
	if _, err := decoder.Token(); !errors.Is(err, io.EOF) {
		if err == nil {
			return nil, errors.New("JSON evidence contains trailing values")
		}
		return nil, err
	}
	return value, nil
}

func decodeUniqueJSONValue(decoder *json.Decoder, depth int) (any, error) {
	if depth > maxCalphadJSONDepth {
		return nil, errors.New("JSON evidence exceeds its nesting limit")
	}
	token, err := decoder.Token()
	if err != nil {
		return nil, err
	}
	delimiter, isDelimiter := token.(json.Delim)
	if !isDelimiter {
		switch token.(type) {
		case nil, bool, string, json.Number:
			return token, nil
		default:
			return nil, errors.New("JSON evidence contains an unsupported scalar")
		}
	}
	switch delimiter {
	case '{':
		object := map[string]any{}
		for decoder.More() {
			keyToken, keyErr := decoder.Token()
			if keyErr != nil {
				return nil, keyErr
			}
			key, ok := keyToken.(string)
			if !ok {
				return nil, errors.New("JSON object key is not a string")
			}
			if _, duplicate := object[key]; duplicate {
				return nil, fmt.Errorf("JSON evidence contains duplicate key %q", key)
			}
			value, valueErr := decodeUniqueJSONValue(decoder, depth+1)
			if valueErr != nil {
				return nil, valueErr
			}
			object[key] = value
		}
		if end, endErr := decoder.Token(); endErr != nil || end != json.Delim('}') {
			return nil, errors.New("JSON evidence has an unterminated object")
		}
		return object, nil
	case '[':
		array := []any{}
		for decoder.More() {
			value, valueErr := decodeUniqueJSONValue(decoder, depth+1)
			if valueErr != nil {
				return nil, valueErr
			}
			array = append(array, value)
		}
		if end, endErr := decoder.Token(); endErr != nil || end != json.Delim(']') {
			return nil, errors.New("JSON evidence has an unterminated array")
		}
		return array, nil
	default:
		return nil, errors.New("JSON evidence contains an unexpected delimiter")
	}
}

func exactJSONObject(value any, keys ...string) (map[string]any, bool) {
	object, ok := value.(map[string]any)
	if !ok || len(object) != len(keys) {
		return nil, false
	}
	for _, key := range keys {
		if _, exists := object[key]; !exists {
			return nil, false
		}
	}
	return object, true
}

func jsonObject(value any) (map[string]any, bool) {
	object, ok := value.(map[string]any)
	return object, ok
}

func jsonString(value any) (string, bool) {
	text, ok := value.(string)
	return text, ok
}

func jsonInt64(value any) (int64, bool) {
	number, ok := value.(json.Number)
	if !ok {
		return 0, false
	}
	parsed, err := number.Int64()
	return parsed, err == nil
}

func jsonFiniteFloat64(value any) (float64, bool) {
	number, ok := value.(json.Number)
	if !ok {
		return 0, false
	}
	parsed, err := number.Float64()
	return parsed, err == nil && !math.IsNaN(parsed) && !math.IsInf(parsed, 0)
}

func validCalphadEvidenceNames(value any, maximum int, allowNull bool) bool {
	if value == nil {
		return allowNull
	}
	values, ok := value.([]any)
	if !ok || len(values) == 0 || len(values) > maximum {
		return false
	}
	previous := ""
	for _, value := range values {
		name, ok := jsonString(value)
		if !ok || !calphadEvidenceNamePattern.MatchString(name) ||
			(previous != "" && name <= previous) {
			return false
		}
		previous = name
	}
	return true
}

func validCalphadEvidenceAxis(value any, minimum, maximum float64) (int, bool) {
	values, ok := value.([]any)
	if !ok || len(values) == 0 || len(values) > 64 {
		return 0, false
	}
	previous := math.Inf(-1)
	for _, value := range values {
		number, ok := jsonFiniteFloat64(value)
		if !ok || number < minimum || number > maximum || number <= previous {
			return 0, false
		}
		previous = number
	}
	return len(values), true
}

func validCalphadEvidenceTypedRequest(request map[string]any, operation string) bool {
	selection, ok := exactJSONObject(request["selection"], "components", "phases")
	if !ok || !validCalphadEvidenceNames(selection["components"], 32, operation == "inspect") ||
		!validCalphadEvidenceNames(selection["phases"], 128, operation == "inspect") {
		return false
	}
	if operation == "inspect" {
		return true
	}
	inspectionSHA, ok := jsonString(request["inspection_artifact_sha256"])
	if !ok || !calphadEvidenceSHA256Pattern.MatchString(inspectionSHA) {
		return false
	}
	if operation == "scheil" {
		conditions, ok := exactJSONObject(
			request["conditions"],
			"independent_composition_mole_fraction", "start_temperature_K",
			"step_temperature_K", "pressure_Pa", "stop_liquid_fraction",
		)
		if !ok {
			return false
		}
		startTemperature, startOK := jsonFiniteFloat64(conditions["start_temperature_K"])
		stepTemperature, stepOK := jsonFiniteFloat64(conditions["step_temperature_K"])
		pressure, pressureOK := jsonFiniteFloat64(conditions["pressure_Pa"])
		stopFraction, stopOK := jsonFiniteFloat64(conditions["stop_liquid_fraction"])
		components, componentsOK := calphadStringList(selection["components"], 32, false)
		phases, phasesOK := calphadStringList(selection["phases"], 128, false)
		if !startOK || startTemperature < 1 || startTemperature > 10_000 ||
			!stepOK || stepTemperature < 0.01 || stepTemperature > 500 ||
			!pressureOK || pressure != domain.CalphadReferencePressurePa ||
			!stopOK || stopFraction < 1e-8 || stopFraction > 0.1 ||
			!componentsOK || !phasesOK {
			return false
		}
		physicalComponents := make([]string, 0, len(components))
		for _, component := range components {
			if component != "VA" && component != "/-" {
				physicalComponents = append(physicalComponents, component)
			}
		}
		if len(physicalComponents) == 0 || !calphadStringSubset([]string{"LIQUID"}, phases) {
			return false
		}
		compositions, compositionsOK := jsonObject(
			conditions["independent_composition_mole_fraction"],
		)
		if !compositionsOK || len(compositions) != len(physicalComponents)-1 {
			return false
		}
		compositionSum := 0.0
		for _, component := range physicalComponents[1:] {
			value, exists := compositions[component]
			fraction, fractionOK := jsonFiniteFloat64(value)
			if !exists || !fractionOK || fraction < 0 || fraction > 1 {
				return false
			}
			compositionSum += fraction
		}
		return compositionSum <= 1+calphadCompositionTolerance
	}
	if operation != "equilibrium" {
		return false
	}
	conditions, ok := exactJSONObject(
		request["conditions"],
		"temperatures_K", "pressures_Pa", "independent_compositions",
	)
	if !ok {
		return false
	}
	temperatures, temperaturesOK := validCalphadEvidenceAxis(
		conditions["temperatures_K"], 1, 10_000,
	)
	pressures, pressuresOK := validCalphadEvidenceAxis(
		conditions["pressures_Pa"], 1e-9, 1e12,
	)
	compositions, compositionsOK := jsonObject(conditions["independent_compositions"])
	if !temperaturesOK || !pressuresOK || !compositionsOK || len(compositions) > 32 {
		return false
	}
	gridPoints := temperatures * pressures
	for component, axis := range compositions {
		if !calphadEvidenceNamePattern.MatchString(component) {
			return false
		}
		axisValues, axisOK := validCalphadEvidenceAxis(axis, 0, 1)
		if !axisOK || gridPoints > 256/axisValues {
			return false
		}
		gridPoints *= axisValues
	}
	return gridPoints <= 256
}

func validCalphadFailureTuple(
	status string,
	failureDomain domain.CalphadFailureDomain,
	failureStage domain.CalphadFailureStage,
	failureCode domain.CalphadFailureCode,
	solverStarted bool,
) bool {
	if !calphadFailureStatus(status) || !failureDomain.Valid() ||
		!failureStage.Valid() || !failureCode.Valid() {
		return false
	}
	switch failureCode {
	case domain.CalphadFailureCodeParseFailed:
		return status == "failed" && failureStage == domain.CalphadFailureStageParse &&
			(failureDomain == domain.CalphadFailureDomainInput ||
				failureDomain == domain.CalphadFailureDomainScientific) && !solverStarted
	case domain.CalphadFailureCodeParseTimeout:
		return status == "timeout" && failureDomain == domain.CalphadFailureDomainScientific &&
			failureStage == domain.CalphadFailureStageParse && !solverStarted
	case domain.CalphadFailureCodeParseUnsupported:
		return status == "unsupported" && failureDomain == domain.CalphadFailureDomainInput &&
			failureStage == domain.CalphadFailureStageParse && !solverStarted
	case domain.CalphadFailureCodeSolverFailed:
		return status == "failed" && failureStage == domain.CalphadFailureStageSolver &&
			((failureDomain == domain.CalphadFailureDomainInput && !solverStarted) ||
				(failureDomain == domain.CalphadFailureDomainScientific && solverStarted))
	case domain.CalphadFailureCodeSolverTimeout:
		return status == "timeout" && failureDomain == domain.CalphadFailureDomainScientific &&
			failureStage == domain.CalphadFailureStageSolver && solverStarted
	case domain.CalphadFailureCodeSolverUnsupported:
		return status == "unsupported" && failureDomain == domain.CalphadFailureDomainScientific &&
			failureStage == domain.CalphadFailureStageSolver && !solverStarted
	case domain.CalphadFailureCodeResultInvalid:
		return status == "failed" && failureDomain == domain.CalphadFailureDomainScientific &&
			failureStage == domain.CalphadFailureStageResultValidation
	case domain.CalphadFailureCodeRuntimeInternalFailure:
		return status == "failed" && failureDomain == domain.CalphadFailureDomainPlatform &&
			(failureStage == domain.CalphadFailureStageParse ||
				failureStage == domain.CalphadFailureStageSolver) && !solverStarted
	case domain.CalphadFailureCodeSandboxFailed:
		return status == "failed" && failureDomain == domain.CalphadFailureDomainPlatform &&
			failureStage == domain.CalphadFailureStageSandboxRuntime && !solverStarted
	case domain.CalphadFailureCodeSandboxTimeout:
		return status == "timeout" && failureDomain == domain.CalphadFailureDomainPlatform &&
			failureStage == domain.CalphadFailureStageSandboxRuntime && !solverStarted
	default:
		return false
	}
}

func verifyCalphadEvidence(req appendCalphadValidationRequest, resourceID string) (verifiedCalphadEvidence, error) {
	var verified verifiedCalphadEvidence
	raw, err := decodeBoundedCalphadEvidence(req.EvidenceGzipBase64)
	if err != nil {
		return verified, err
	}
	evidenceSize := int64(len(raw))
	evidenceDigest := sha256.Sum256(raw)
	evidenceSHA := hex.EncodeToString(evidenceDigest[:])
	operation := strings.TrimSpace(req.Operation)
	status := strings.TrimSpace(req.Status)
	if req.Operation != operation || req.Status != status ||
		req.FailureDomain != strings.TrimSpace(req.FailureDomain) ||
		req.FailureStage != strings.TrimSpace(req.FailureStage) ||
		req.FailureCode != strings.TrimSpace(req.FailureCode) ||
		req.EvidencePath != strings.TrimSpace(req.EvidencePath) ||
		req.EvidenceSHA256 != strings.ToLower(strings.TrimSpace(req.EvidenceSHA256)) ||
		req.RuntimeImageID != strings.ToLower(strings.TrimSpace(req.RuntimeImageID)) ||
		req.PycalphadVersion != strings.TrimSpace(req.PycalphadVersion) {
		return verified, errors.New("CALPHAD validation request fields are not canonical")
	}
	if req.PycalphadVersion == "" || len(req.PycalphadVersion) > 128 {
		return verified, errors.New("CALPHAD pycalphad version identity is invalid")
	}
	for _, character := range req.PycalphadVersion {
		if character < 32 || character == 127 {
			return verified, errors.New("CALPHAD pycalphad version identity is invalid")
		}
	}
	artifactDirectory := operation
	if operation == "inspect" {
		artifactDirectory = "inspection"
	}
	expectedPath := "/outputs/calphad/" + artifactDirectory + "/" + evidenceSHA + ".json"
	if req.EvidenceSizeBytes != evidenceSize || strings.ToLower(strings.TrimSpace(req.EvidenceSHA256)) != evidenceSHA ||
		strings.TrimSpace(req.EvidencePath) != expectedPath {
		return verified, errors.New("CALPHAD evidence bytes do not match the declared path, SHA-256, and size")
	}
	rootValue, err := decodeUniqueJSON(raw)
	if err != nil {
		return verified, fmt.Errorf("invalid CALPHAD evidence JSON: %w", err)
	}
	root, ok := jsonObject(rootValue)
	if !ok {
		return verified, errors.New("CALPHAD evidence root is not an object")
	}
	schemaVersion, _ := jsonString(root["schema_version"])
	evidenceOperation, _ := jsonString(root["operation"])
	isFailureEvidence := schemaVersion == domain.CalphadFailureEvidenceSchemaVersion
	if isFailureEvidence {
		root, ok = exactJSONObject(
			rootValue, "schema_version", "operation", "database_binding", "request", "outcome",
			"execution_contract", "validation_persistence",
		)
	} else {
		root, ok = exactJSONObject(
			rootValue, "schema_version", "operation", "database_binding", "request", "result",
			"execution_contract", "validation_persistence",
		)
	}
	successStatusMatches := (operation == "inspect" && status == "input_validated") ||
		(operation == "equilibrium" && status == "equilibrium_completed") ||
		(operation == "scheil" && status == "scheil_completed")
	if !ok || evidenceOperation != operation ||
		(operation != "inspect" && operation != "equilibrium" && operation != "scheil") ||
		(isFailureEvidence && !calphadFailureStatus(status)) ||
		(!isFailureEvidence &&
			(schemaVersion != calphadToolEvidenceSchemaVersion || !successStatusMatches)) {
		return verified, errors.New("CALPHAD evidence schema, operation, or status is inconsistent")
	}
	database, ok := exactJSONObject(
		root["database_binding"],
		"kind", "database_id", "resource_id", "sha256", "size_bytes", "database_format", "source",
		"license_id", "assessment_scope", "reference_state", "temperature_limits_K",
		domain.CalphadAssessmentPressureLimitsMetadataKey,
		"binding_schema", "binding_authority", "declaration_authority",
	)
	if !ok {
		return verified, errors.New("CALPHAD evidence has an invalid resource database binding")
	}
	kind, _ := jsonString(database["kind"])
	databaseResourceID, _ := jsonString(database["resource_id"])
	databaseSHA, _ := jsonString(database["sha256"])
	databaseSize, sizeOK := jsonInt64(database["size_bytes"])
	databaseFormat, _ := jsonString(database["database_format"])
	bindingAuthority, _ := jsonString(database["binding_authority"])
	declarationAuthority, _ := jsonString(database["declaration_authority"])
	bindingSchema, _ := jsonString(database["binding_schema"])
	databaseID, _ := jsonString(database["database_id"])
	source, _ := jsonString(database["source"])
	licenseID, _ := jsonString(database["license_id"])
	assessmentScope, _ := jsonString(database["assessment_scope"])
	referenceState, _ := jsonString(database["reference_state"])
	temperatureLimits, limitsOK := database["temperature_limits_K"].([]any)
	if kind != "resource" || databaseResourceID != resourceID || !calphadEvidenceSHA256Pattern.MatchString(databaseSHA) ||
		!sizeOK || databaseSize <= 0 || bindingAuthority != "control_resource_catalog" ||
		(databaseFormat != domain.CalphadDatabaseFormatTDB && databaseFormat != domain.CalphadDatabaseFormatDAT) ||
		declarationAuthority != "resource_owner" ||
		bindingSchema != "ultra.selected_resource.v1" ||
		strings.TrimSpace(databaseID) == "" || strings.TrimSpace(source) == "" ||
		strings.TrimSpace(licenseID) == "" || strings.TrimSpace(assessmentScope) == "" ||
		strings.TrimSpace(referenceState) == "" || !limitsOK || len(temperatureLimits) != 2 {
		return verified, errors.New("CALPHAD evidence database binding is not server-authorized resource evidence")
	}
	minimumTemperature, minimumOK := jsonFiniteFloat64(temperatureLimits[0])
	maximumTemperature, maximumOK := jsonFiniteFloat64(temperatureLimits[1])
	if !minimumOK || !maximumOK || minimumTemperature <= 0 || minimumTemperature >= maximumTemperature ||
		maximumTemperature > 10_000 {
		return verified, errors.New("CALPHAD evidence database temperature limits are invalid")
	}
	pressureLimitValues, pressureLimitsOK := calphadBoundedNumberList(
		database[domain.CalphadAssessmentPressureLimitsMetadataKey], 2,
		domain.CalphadMinimumPressurePa, domain.CalphadMaximumPressurePa, false,
	)
	if !pressureLimitsOK || len(pressureLimitValues) != 2 || pressureLimitValues[0] > pressureLimitValues[1] {
		return verified, errors.New("CALPHAD evidence database assessment pressure limits are invalid")
	}
	pressureLimits := [2]float64{pressureLimitValues[0], pressureLimitValues[1]}
	requestKeys := []string{"operation", "runtime_image_id", "selection"}
	if operation == "equilibrium" || operation == "scheil" {
		requestKeys = append(requestKeys, "inspection_artifact_sha256", "conditions")
	}
	evidenceRequest, ok := exactJSONObject(root["request"], requestKeys...)
	if !ok {
		return verified, errors.New("CALPHAD evidence has an invalid typed request record")
	}
	requestOperation, _ := jsonString(evidenceRequest["operation"])
	requestImage, _ := jsonString(evidenceRequest["runtime_image_id"])
	runtimeImage := strings.ToLower(strings.TrimSpace(req.RuntimeImageID))
	if requestOperation != operation || requestImage != runtimeImage || !calphadEvidenceImagePattern.MatchString(runtimeImage) {
		return verified, errors.New("CALPHAD evidence runtime request identity is inconsistent")
	}
	if !validCalphadEvidenceTypedRequest(evidenceRequest, operation) {
		return verified, errors.New("CALPHAD evidence typed request values are outside the fixed schema")
	}
	if operation == "equilibrium" {
		conditions, _ := jsonObject(evidenceRequest["conditions"])
		pressures, pressuresOK := calphadNumberList(
			conditions["pressures_Pa"], 64,
			domain.CalphadMinimumPressurePa, domain.CalphadMaximumPressurePa, false,
		)
		if !pressuresOK || len(pressures) == 0 || pressures[0] < pressureLimits[0] ||
			pressures[len(pressures)-1] > pressureLimits[1] {
			return verified, errors.New("CALPHAD evidence pressure request is outside the owner-declared assessment limits")
		}
	} else if operation == "scheil" {
		conditions, _ := jsonObject(evidenceRequest["conditions"])
		pressure, pressureOK := jsonFiniteFloat64(conditions["pressure_Pa"])
		if !pressureOK || pressure != domain.CalphadReferencePressurePa ||
			pressure < pressureLimits[0] || pressure > pressureLimits[1] {
			return verified, errors.New("CALPHAD Scheil pressure is outside the owner-declared assessment limits")
		}
	}
	execution, ok := exactJSONObject(
		root["execution_contract"],
		"interface", "caller_code_accepted", "caller_models_or_solver_options_accepted",
		"network", "no_new_privileges", "read_only_root_filesystem", "cap_drop_all",
		"cpus_at_most", "memory_bytes_at_most", "pids_at_most",
		"runtime_image_id", "max_components", "max_phases", "max_axis_values",
		"max_grid_points", "wall_time_seconds", "max_result_bytes",
	)
	if !ok || execution["caller_code_accepted"] != false ||
		execution["caller_models_or_solver_options_accepted"] != false ||
		execution["no_new_privileges"] != true ||
		execution["read_only_root_filesystem"] != true || execution["cap_drop_all"] != true {
		return verified, errors.New("CALPHAD evidence execution contract is invalid")
	}
	executionNetwork, _ := jsonString(execution["network"])
	executionImage, _ := jsonString(execution["runtime_image_id"])
	executionInterface, _ := jsonString(execution["interface"])
	maxComponents, maxComponentsOK := jsonInt64(execution["max_components"])
	maxPhases, maxPhasesOK := jsonInt64(execution["max_phases"])
	maxAxisValues, maxAxisOK := jsonInt64(execution["max_axis_values"])
	maxGridPoints, maxGridOK := jsonInt64(execution["max_grid_points"])
	wallTime, wallTimeOK := jsonFiniteFloat64(execution["wall_time_seconds"])
	maxResultBytes, maxResultOK := jsonInt64(execution["max_result_bytes"])
	maxCPUs, maxCPUsOK := jsonFiniteFloat64(execution["cpus_at_most"])
	maxMemoryBytes, maxMemoryOK := jsonInt64(execution["memory_bytes_at_most"])
	maxPIDs, maxPIDsOK := jsonInt64(execution["pids_at_most"])
	if executionNetwork != "none" || executionImage != runtimeImage ||
		executionInterface != "fixed ultra_deepagents.materials.calphad public surface" ||
		!maxCPUsOK || maxCPUs != 8 || !maxMemoryOK || maxMemoryBytes != 32*1024*1024*1024 ||
		!maxPIDsOK || maxPIDs != 4096 ||
		!maxComponentsOK || maxComponents != 32 || !maxPhasesOK || maxPhases != 128 ||
		!maxAxisOK || maxAxisValues != 64 || !maxGridOK || maxGridPoints != 256 ||
		!wallTimeOK || wallTime != 30 || !maxResultOK || maxResultBytes != 16*1024*1024 {
		return verified, errors.New("CALPHAD evidence execution isolation or image identity is invalid")
	}
	persistence, ok := exactJSONObject(
		root["validation_persistence"],
		"catalog_status", "catalog_metadata_updated", "mode", "note",
	)
	if !ok || persistence["catalog_metadata_updated"] != false {
		return verified, errors.New("CALPHAD evidence persistence contract is invalid")
	}
	catalogStatus, _ := jsonString(persistence["catalog_status"])
	persistenceMode, _ := jsonString(persistence["mode"])
	persistenceNote, noteOK := jsonString(persistence["note"])
	if catalogStatus != "pending" || persistenceMode != "immutable_per_run_evidence" ||
		!noteOK || strings.TrimSpace(persistenceNote) == "" || len(persistenceNote) > 1024 {
		return verified, errors.New("CALPHAD evidence persistence authority is invalid")
	}
	requestSHA, err := calphadTypedRequestSHA256(evidenceRequest)
	if err != nil {
		return verified, err
	}
	pycalphadVersion := strings.TrimSpace(req.PycalphadVersion)
	if isFailureEvidence {
		outcome, outcomeOK := exactJSONObject(
			root["outcome"], "status", "failure_domain", "failure_stage", "failure_code",
			"exit_code", "solver_started",
		)
		if !outcomeOK {
			return verified, errors.New("CALPHAD failure evidence outcome is invalid")
		}
		outcomeStatus, _ := jsonString(outcome["status"])
		failureDomainText, _ := jsonString(outcome["failure_domain"])
		failureStageText, _ := jsonString(outcome["failure_stage"])
		failureCodeText, _ := jsonString(outcome["failure_code"])
		failureDomain := domain.CalphadFailureDomain(failureDomainText)
		failureStage := domain.CalphadFailureStage(failureStageText)
		failureCode := domain.CalphadFailureCode(failureCodeText)
		solverStarted, solverStartedOK := outcome["solver_started"].(bool)
		exitCodeOK := outcome["exit_code"] == nil
		if !exitCodeOK {
			_, exitCodeOK = jsonInt64(outcome["exit_code"])
		}
		operationTupleOK := (operation == "inspect" &&
			failureStage != domain.CalphadFailureStageSolver && !solverStarted) ||
			((operation == "equilibrium" || operation == "scheil") &&
				failureStage != domain.CalphadFailureStageParse)
		if failureCode == domain.CalphadFailureCodeResultInvalid {
			operationTupleOK = solverStarted == (operation != "inspect")
		}
		if outcomeStatus != status || req.FailureDomain != failureDomainText ||
			req.FailureStage != failureStageText || req.FailureCode != failureCodeText ||
			!solverStartedOK || !exitCodeOK || !operationTupleOK ||
			!validCalphadFailureTuple(
				status, failureDomain, failureStage, failureCode, solverStarted,
			) {
			return verified, errors.New("CALPHAD failure evidence tuple is inconsistent")
		}
		inspectionEvidenceSHA := ""
		if operation == "equilibrium" || operation == "scheil" {
			inspectionEvidenceSHA, _ = jsonString(evidenceRequest["inspection_artifact_sha256"])
		}
		return verifiedCalphadEvidence{
			ResourceID: databaseResourceID, DatabaseID: databaseID,
			DatabaseSHA256: databaseSHA, DatabaseSizeBytes: databaseSize,
			DatabaseFormat: databaseFormat, RequestSHA256: requestSHA,
			Source: source, LicenseID: licenseID, AssessmentScope: assessmentScope,
			ReferenceState:             referenceState,
			TemperatureLimitsK:         [2]float64{minimumTemperature, maximumTemperature},
			AssessmentPressureLimitsPa: pressureLimits,
			Operation:                  operation, Status: status,
			FailureDomain: failureDomain, FailureStage: failureStage, FailureCode: failureCode,
			EvidencePath: expectedPath, EvidenceSHA256: evidenceSHA,
			EvidenceSizeBytes: evidenceSize, RuntimeImageID: runtimeImage,
			PycalphadVersion:         pycalphadVersion,
			InspectionEvidenceSHA256: inspectionEvidenceSHA,
			EvidenceBytes:            append([]byte(nil), raw...),
		}, nil
	}
	result, ok := jsonObject(root["result"])
	if !ok {
		return verified, errors.New("CALPHAD evidence result is not an object")
	}
	var resultDatabase map[string]any
	inspectionEvidenceSHA := ""
	selection, selectionOK := exactJSONObject(evidenceRequest["selection"], "components", "phases")
	selectedComponents, selectedComponentsOK := calphadOptionalStringList(selection["components"], 32)
	selectedPhases, selectedPhasesOK := calphadOptionalStringList(selection["phases"], 128)
	if !selectionOK || !selectedComponentsOK || !selectedPhasesOK {
		return verified, errors.New("CALPHAD evidence selection could not be reconstructed")
	}
	if operation == "inspect" {
		resultDatabase = result
		if err := validateCalphadInspectionManifest(
			result, database, pycalphadVersion, selectedComponents, selectedPhases,
		); err != nil {
			return verified, err
		}
	} else if operation == "equilibrium" {
		if err := validateCalphadEquilibriumResult(result, evidenceRequest, database, pycalphadVersion); err != nil {
			return verified, err
		}
		resultDatabase, ok = jsonObject(result["database"])
		if !ok {
			return verified, errors.New("CALPHAD equilibrium evidence omits its database manifest")
		}
		inspectionEvidenceSHA, _ = jsonString(evidenceRequest["inspection_artifact_sha256"])
	} else {
		if err := validateCalphadScheilResult(result, evidenceRequest, database, pycalphadVersion); err != nil {
			return verified, err
		}
		resultDatabase, ok = jsonObject(result["database"])
		if !ok {
			return verified, errors.New("CALPHAD Scheil evidence omits its database manifest")
		}
		inspectionEvidenceSHA, _ = jsonString(evidenceRequest["inspection_artifact_sha256"])
	}
	resultDatabaseSHA, _ := jsonString(resultDatabase["sha256"])
	resultDatabaseSize, resultSizeOK := jsonInt64(resultDatabase["size_bytes"])
	resultPycalphadVersion, _ := jsonString(resultDatabase["pycalphad_version"])
	if resultDatabaseSHA != databaseSHA || !resultSizeOK || resultDatabaseSize != databaseSize ||
		resultPycalphadVersion != pycalphadVersion || pycalphadVersion == "" {
		return verified, errors.New("CALPHAD result database or pycalphad identity is inconsistent")
	}
	databaseInventorySHA, err := calphadDatabaseInventorySHA256(resultDatabase)
	if err != nil {
		return verified, err
	}
	return verifiedCalphadEvidence{
		ResourceID:                 databaseResourceID,
		DatabaseID:                 databaseID,
		DatabaseSHA256:             databaseSHA,
		DatabaseSizeBytes:          databaseSize,
		DatabaseFormat:             databaseFormat,
		DatabaseInventorySHA256:    databaseInventorySHA,
		RequestSHA256:              requestSHA,
		Source:                     source,
		LicenseID:                  licenseID,
		AssessmentScope:            assessmentScope,
		ReferenceState:             referenceState,
		TemperatureLimitsK:         [2]float64{minimumTemperature, maximumTemperature},
		AssessmentPressureLimitsPa: pressureLimits,
		Operation:                  operation,
		Status:                     status,
		EvidencePath:               expectedPath,
		EvidenceSHA256:             evidenceSHA,
		EvidenceSizeBytes:          evidenceSize,
		RuntimeImageID:             runtimeImage,
		PycalphadVersion:           pycalphadVersion,
		InspectionEvidenceSHA256:   inspectionEvidenceSHA,
		EvidenceBytes:              append([]byte(nil), raw...),
	}, nil
}
