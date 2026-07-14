package store

import (
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

func stringFromPointer(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

var (
	calphadSHA256Pattern = regexp.MustCompile(`^[0-9a-f]{64}$`)
	calphadImagePattern  = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
)

const maxCalphadEvidenceBytes int64 = 32 << 20

var ErrCalphadRunLeaseInvalid = errors.New("CALPHAD validation run lease is not active")

var ErrCalphadRuntimePolicyInvalid = fmt.Errorf("%w: CALPHAD validation runtime is not authorized by the run policy", ErrConflict)

var ErrCalphadInspectionRequired = fmt.Errorf("%w: CALPHAD equilibrium requires its exact retained inspection evidence", ErrConflict)

var ErrCalphadInputRetentionRequired = fmt.Errorf("%w: exact CALPHAD input bytes are not retained", ErrConflict)

var ErrCalphadPressureLimitsInvalid = fmt.Errorf("%w: CALPHAD assessment pressure limits are missing, invalid, or inconsistent", ErrConflict)

var ErrCalphadEvidenceRetentionRequired = fmt.Errorf("%w: exact CALPHAD validation evidence bytes are not retained", ErrConflict)

var ErrCalphadDatabaseFormatInvalid = fmt.Errorf("%w: CALPHAD database format is missing, unsupported, or inconsistent", ErrConflict)

var ErrCalphadOwnerDeclarationInvalid = fmt.Errorf("%w: CALPHAD owner declaration is missing, invalid, or inconsistent", ErrConflict)

var ErrCalphadTenantCapacityExceeded = fmt.Errorf("%w: CALPHAD tenant retained-byte or validation-event capacity is exhausted", ErrConflict)

type calphadInputBlob struct {
	SHA256    string
	SizeBytes int64
	Payload   []byte
}

type calphadEvidenceBlob struct {
	SHA256    string
	SizeBytes int64
	Payload   []byte
}

func calphadPolicyInt64(value any) (int64, bool) {
	switch typed := value.(type) {
	case int:
		return int64(typed), true
	case int32:
		return int64(typed), true
	case int64:
		return typed, true
	case float64:
		if math.IsNaN(typed) || math.IsInf(typed, 0) || math.Trunc(typed) != typed ||
			typed < math.MinInt64 || typed > math.MaxInt64 {
			return 0, false
		}
		return int64(typed), true
	case json.Number:
		parsed, err := typed.Int64()
		return parsed, err == nil
	default:
		return 0, false
	}
}

func calphadRuntimePolicy(metadata domain.JSONMap) (string, string, bool) {
	var policy map[string]any
	switch typed := metadata[domain.CalphadRuntimePolicyMetadataKey].(type) {
	case domain.JSONMap:
		policy = map[string]any(typed)
	case map[string]any:
		policy = typed
	default:
		return "", "", false
	}
	if len(policy) != 11 || policy["schema_version"] != domain.CalphadRuntimePolicySchema ||
		policy["authority"] != "control_plane" || policy["pycalphad_version"] != domain.CalphadPycalphadVersion ||
		policy["network"] != domain.CalphadRuntimeNetwork || policy["no_new_privileges"] != true ||
		policy["read_only_root_filesystem"] != true || policy["cap_drop_all"] != true {
		return "", "", false
	}
	cpus, cpusOK := calphadPolicyInt64(policy["cpus_at_most"])
	memoryBytes, memoryOK := calphadPolicyInt64(policy["memory_bytes_at_most"])
	pids, pidsOK := calphadPolicyInt64(policy["pids_at_most"])
	if !cpusOK || cpus != domain.CalphadRuntimeCPUsAtMost ||
		!memoryOK || memoryBytes != domain.CalphadRuntimeMemoryBytesAtMost ||
		!pidsOK || pids != domain.CalphadRuntimePIDsAtMost {
		return "", "", false
	}
	runtimeImage, imageOK := policy["runtime_image_id"].(string)
	pycalphadVersion, versionOK := policy["pycalphad_version"].(string)
	runtimeImage = strings.ToLower(strings.TrimSpace(runtimeImage))
	pycalphadVersion = strings.TrimSpace(pycalphadVersion)
	if !imageOK || !versionOK || !calphadImagePattern.MatchString(runtimeImage) ||
		pycalphadVersion != domain.CalphadPycalphadVersion {
		return "", "", false
	}
	return runtimeImage, pycalphadVersion, true
}

func authorizeCalphadRuntime(run domain.RunRecord, input domain.AppendCalphadValidationInput) error {
	runtimeImage, pycalphadVersion, ok := calphadRuntimePolicy(run.Metadata)
	if !ok || runtimeImage != strings.ToLower(strings.TrimSpace(input.RuntimeImageID)) ||
		pycalphadVersion != strings.TrimSpace(input.PycalphadVersion) {
		return ErrCalphadRuntimePolicyInvalid
	}
	return nil
}

func withCalphadEvidenceRetention(record domain.CalphadValidationRecord, blobRetained bool) domain.CalphadValidationRecord {
	record.Promotable = false
	switch {
	case record.Operation == "registration":
		record.EvidenceRetention = domain.CalphadEvidenceRetentionNotApplicable
	case record.EvidenceContractVersion != domain.CalphadEvidenceContractVersion:
		record.EvidenceRetention = domain.CalphadEvidenceRetentionLegacyUnretained
	case record.EvidenceSHA256 == "":
		record.EvidenceRetention = domain.CalphadEvidenceRetentionUnretained
	case !blobRetained:
		record.EvidenceRetention = domain.CalphadEvidenceRetentionLegacyUnretained
	case !calphadValidationPressureMetadataValid(record):
		record.EvidenceRetention = domain.CalphadEvidenceRetentionUnretained
	default:
		record.EvidenceRetention = domain.CalphadEvidenceRetentionRetained
		record.Promotable = record.Status == "input_validated" ||
			((record.Status == "equilibrium_completed" || record.Status == "scheil_completed") &&
				record.InspectionEvidenceSHA256 != "")
	}
	return record
}

func mapCalphadAppendError(err error) error {
	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) {
		switch {
		case pgErr.Code == "P0002":
			return ErrNotFound
		case strings.Contains(pgErr.Message, "CALPHAD_TENANT_CAPACITY_EXCEEDED"):
			return ErrCalphadTenantCapacityExceeded
		case strings.Contains(pgErr.Message, "CALPHAD_RUNTIME_POLICY_INVALID"):
			return ErrCalphadRuntimePolicyInvalid
		case strings.Contains(pgErr.Message, "CALPHAD_INSPECTION_REQUIRED"):
			return ErrCalphadInspectionRequired
		case strings.Contains(pgErr.Message, "CALPHAD_INPUT_RETENTION_REQUIRED"):
			return ErrCalphadInputRetentionRequired
		case strings.Contains(pgErr.Message, "CALPHAD_EVIDENCE_RETENTION_REQUIRED"):
			return ErrCalphadEvidenceRetentionRequired
		case strings.Contains(pgErr.Message, "CALPHAD_OWNER_DECLARATION_INVALID"):
			return ErrCalphadOwnerDeclarationInvalid
		case strings.Contains(pgErr.Message, "CALPHAD_DATABASE_FORMAT_INVALID"):
			return ErrCalphadDatabaseFormatInvalid
		case strings.Contains(pgErr.Message, "CALPHAD_PRESSURE_BINDING_INVALID"):
			return ErrCalphadPressureLimitsInvalid
		case pgErr.Code == "28000" || strings.Contains(pgErr.Message, "CALPHAD_RUN_LEASE_INVALID"):
			return ErrCalphadRunLeaseInvalid
		case pgErr.Code == "23514":
			return ErrConflict
		}
	}
	return mapPgError(err)
}

func deepCloneCalphadJSONValue(value any) any {
	switch typed := value.(type) {
	case domain.JSONMap:
		return deepCloneCalphadJSONMap(typed)
	case map[string]any:
		return deepCloneCalphadJSONMap(domain.JSONMap(typed))
	case []any:
		cloned := make([]any, len(typed))
		for index, item := range typed {
			cloned[index] = deepCloneCalphadJSONValue(item)
		}
		return cloned
	case []domain.JSONMap:
		cloned := make([]domain.JSONMap, len(typed))
		for index, item := range typed {
			cloned[index] = deepCloneCalphadJSONMap(item)
		}
		return cloned
	case []map[string]any:
		cloned := make([]map[string]any, len(typed))
		for index, item := range typed {
			cloned[index] = map[string]any(deepCloneCalphadJSONMap(domain.JSONMap(item)))
		}
		return cloned
	case []string:
		return append([]string(nil), typed...)
	case []float64:
		return append([]float64(nil), typed...)
	case []int:
		return append([]int(nil), typed...)
	case []bool:
		return append([]bool(nil), typed...)
	default:
		return value
	}
}

func deepCloneCalphadJSONMap(value domain.JSONMap) domain.JSONMap {
	if value == nil {
		return domain.JSONMap{}
	}
	cloned := make(domain.JSONMap, len(value))
	for key, item := range value {
		cloned[key] = deepCloneCalphadJSONValue(item)
	}
	return cloned
}

func calphadPressureNumber(value any) (float64, bool) {
	var number float64
	switch typed := value.(type) {
	case float64:
		number = typed
	case float32:
		number = float64(typed)
	case int:
		number = float64(typed)
	case int8:
		number = float64(typed)
	case int16:
		number = float64(typed)
	case int32:
		number = float64(typed)
	case int64:
		number = float64(typed)
	case uint:
		number = float64(typed)
	case uint8:
		number = float64(typed)
	case uint16:
		number = float64(typed)
	case uint32:
		number = float64(typed)
	case uint64:
		number = float64(typed)
	case json.Number:
		parsed, err := typed.Float64()
		if err != nil {
			return 0, false
		}
		number = parsed
	default:
		return 0, false
	}
	return number, !math.IsNaN(number) && !math.IsInf(number, 0)
}

func validCalphadAssessmentPressureLimits(limits [2]float64) bool {
	return !math.IsNaN(limits[0]) && !math.IsNaN(limits[1]) &&
		!math.IsInf(limits[0], 0) && !math.IsInf(limits[1], 0) &&
		limits[0] >= domain.CalphadMinimumPressurePa &&
		limits[1] <= domain.CalphadMaximumPressurePa && limits[0] <= limits[1]
}

func calphadAssessmentPressureLimitsFromValue(value any) ([2]float64, bool) {
	var values []any
	switch typed := value.(type) {
	case [2]float64:
		return typed, validCalphadAssessmentPressureLimits(typed)
	case []float64:
		if len(typed) != 2 {
			return [2]float64{}, false
		}
		limits := [2]float64{typed[0], typed[1]}
		return limits, validCalphadAssessmentPressureLimits(limits)
	case []any:
		values = typed
	default:
		return [2]float64{}, false
	}
	if len(values) != 2 {
		return [2]float64{}, false
	}
	minimum, minimumOK := calphadPressureNumber(values[0])
	maximum, maximumOK := calphadPressureNumber(values[1])
	limits := [2]float64{minimum, maximum}
	return limits, minimumOK && maximumOK && validCalphadAssessmentPressureLimits(limits)
}

func withCalphadAssessmentPressureMetadata(metadata domain.JSONMap, limits [2]float64) domain.JSONMap {
	cloned := deepCloneCalphadJSONMap(metadata)
	cloned[domain.CalphadAssessmentPressureLimitsMetadataKey] = []float64{limits[0], limits[1]}
	return cloned
}

func validCalphadDatabaseFormat(value string) bool {
	return value == domain.CalphadDatabaseFormatTDB || value == domain.CalphadDatabaseFormatDAT
}

func calphadDatabaseFormatForName(name string) (string, bool) {
	return domain.CalphadDatabaseFormatFromName(name)
}

func calphadFiniteLimitsFromValue(
	value any,
	minimum, maximum float64,
	allowFixed bool,
) ([2]float64, bool) {
	var values []any
	switch typed := value.(type) {
	case [2]float64:
		values = []any{typed[0], typed[1]}
	case []float64:
		if len(typed) != 2 {
			return [2]float64{}, false
		}
		values = []any{typed[0], typed[1]}
	case []any:
		values = typed
	default:
		return [2]float64{}, false
	}
	if len(values) != 2 {
		return [2]float64{}, false
	}
	lower, lowerOK := calphadPressureNumber(values[0])
	upper, upperOK := calphadPressureNumber(values[1])
	limits := [2]float64{lower, upper}
	return limits, lowerOK && upperOK && lower >= minimum && upper <= maximum &&
		(lower < upper || (allowFixed && lower == upper))
}

func calphadOwnerDeclarationJSON(declaration domain.CalphadOwnerDeclaration) domain.JSONMap {
	return domain.JSONMap{
		"schema_version":   declaration.SchemaVersion,
		"authority":        declaration.Authority,
		"database_id":      declaration.DatabaseID,
		"source":           declaration.Source,
		"license_id":       declaration.LicenseID,
		"assessment_scope": declaration.AssessmentScope,
		"reference_state":  declaration.ReferenceState,
		"assessment_temperature_limits_K": []float64{
			declaration.AssessmentTemperatureLimitsK[0],
			declaration.AssessmentTemperatureLimitsK[1],
		},
		domain.CalphadAssessmentPressureLimitsMetadataKey: []float64{
			declaration.AssessmentPressureLimitsPa[0],
			declaration.AssessmentPressureLimitsPa[1],
		},
		"database_format": declaration.DatabaseFormat,
	}
}

func calphadOwnerDeclarationFromValue(value any) (domain.CalphadOwnerDeclaration, bool) {
	declarationMap, ok := resourceMetadataMap(value)
	if !ok || len(declarationMap) != 10 {
		return domain.CalphadOwnerDeclaration{}, false
	}
	requiredKeys := []string{
		"schema_version", "authority", "database_id", "source", "license_id",
		"assessment_scope", "reference_state", "assessment_temperature_limits_K",
		domain.CalphadAssessmentPressureLimitsMetadataKey, "database_format",
	}
	for _, key := range requiredKeys {
		if _, exists := declarationMap[key]; !exists {
			return domain.CalphadOwnerDeclaration{}, false
		}
	}
	text := func(key string, maximum int) (string, bool) {
		value, typeOK := declarationMap[key].(string)
		value = strings.TrimSpace(value)
		return value, typeOK && value == declarationMap[key] && validCalphadLedgerText(value, maximum)
	}
	schema, schemaOK := text("schema_version", 128)
	authority, authorityOK := text("authority", 128)
	databaseID, databaseIDOK := domain.SafeCalphadOwnerDeclaration(declarationMap["database_id"], 512)
	source, sourceOK := domain.SafeCalphadOwnerDeclaration(declarationMap["source"], 1024)
	licenseID, licenseOK := domain.SafeCalphadLicenseIdentifier(declarationMap["license_id"])
	assessmentScope, scopeOK := domain.SafeCalphadOwnerDeclaration(declarationMap["assessment_scope"], 1024)
	referenceState, referenceOK := domain.SafeCalphadOwnerDeclaration(declarationMap["reference_state"], 512)
	databaseFormat, formatOK := text("database_format", 8)
	temperatureLimits, temperatureOK := calphadFiniteLimitsFromValue(
		declarationMap["assessment_temperature_limits_K"], 1, 10_000, false,
	)
	pressureLimits, pressureOK := calphadAssessmentPressureLimitsFromValue(
		declarationMap[domain.CalphadAssessmentPressureLimitsMetadataKey],
	)
	if !schemaOK || schema != domain.CalphadOwnerDeclarationSchema ||
		!authorityOK || authority != "resource_owner" || !databaseIDOK || !sourceOK ||
		!licenseOK || !scopeOK || !referenceOK || !formatOK ||
		!validCalphadDatabaseFormat(databaseFormat) || !temperatureOK || !pressureOK {
		return domain.CalphadOwnerDeclaration{}, false
	}
	return domain.CalphadOwnerDeclaration{
		SchemaVersion: schema, Authority: authority, DatabaseID: databaseID,
		Source: source, LicenseID: licenseID, AssessmentScope: assessmentScope,
		ReferenceState: referenceState, AssessmentTemperatureLimitsK: temperatureLimits,
		AssessmentPressureLimitsPa: pressureLimits, DatabaseFormat: databaseFormat,
	}, true
}

func calphadOwnerDeclarationFromResource(
	resource domain.ResourceRecord,
) (domain.CalphadOwnerDeclaration, error) {
	databaseFormat, formatOK := calphadDatabaseFormatForName(resource.OriginalName)
	if !formatOK {
		return domain.CalphadOwnerDeclaration{}, ErrCalphadDatabaseFormatInvalid
	}
	calphadMetadata, ok := resourceMetadataMap(resource.Metadata["calphad"])
	if !ok {
		return domain.CalphadOwnerDeclaration{}, ErrCalphadOwnerDeclarationInvalid
	}
	databaseID, databaseIDOK := domain.SafeCalphadOwnerDeclaration(calphadMetadata["database_id"], 512)
	if !databaseIDOK {
		databaseID = strings.TrimSpace(resource.ResourceID)
		databaseID, databaseIDOK = domain.SafeCalphadOwnerDeclaration(databaseID, 512)
	}
	source, sourceOK := domain.SafeCalphadOwnerDeclaration(calphadMetadata["source"], 1024)
	licenseID, licenseOK := domain.SafeCalphadLicenseIdentifier(calphadMetadata["license_id"])
	if !licenseOK {
		licenseID, licenseOK = domain.SafeCalphadLicenseIdentifier(calphadMetadata["license_identifier"])
	}
	assessmentScope, scopeOK := domain.SafeCalphadOwnerDeclaration(calphadMetadata["assessment_scope"], 1024)
	referenceState, referenceOK := domain.SafeCalphadOwnerDeclaration(calphadMetadata["reference_state"], 512)
	temperatureValue, hasAssessmentTemperature := calphadMetadata["assessment_temperature_limits_K"]
	legacyTemperatureValue, hasLegacyTemperature := calphadMetadata["tdb_temperature_limits_K"]
	if !hasAssessmentTemperature {
		temperatureValue = legacyTemperatureValue
	}
	temperatureLimits, temperatureOK := calphadFiniteLimitsFromValue(
		temperatureValue, 1, 10_000, false,
	)
	if hasAssessmentTemperature && hasLegacyTemperature {
		legacyLimits, legacyOK := calphadFiniteLimitsFromValue(
			legacyTemperatureValue, 1, 10_000, false,
		)
		temperatureOK = temperatureOK && legacyOK && legacyLimits == temperatureLimits
	}
	pressureLimits, pressureOK := calphadAssessmentPressureLimitsFromValue(
		calphadMetadata[domain.CalphadAssessmentPressureLimitsMetadataKey],
	)
	if !databaseIDOK || !sourceOK || !licenseOK || !scopeOK || !referenceOK ||
		!temperatureOK || !pressureOK {
		return domain.CalphadOwnerDeclaration{}, ErrCalphadOwnerDeclarationInvalid
	}
	return domain.CalphadOwnerDeclaration{
		SchemaVersion: domain.CalphadOwnerDeclarationSchema, Authority: "resource_owner",
		DatabaseID: databaseID, Source: source, LicenseID: licenseID,
		AssessmentScope: assessmentScope, ReferenceState: referenceState,
		AssessmentTemperatureLimitsK: temperatureLimits,
		AssessmentPressureLimitsPa:   pressureLimits, DatabaseFormat: databaseFormat,
	}, nil
}

func CalphadOwnerDeclarationForResource(
	resource domain.ResourceRecord,
) (domain.CalphadOwnerDeclaration, error) {
	return calphadOwnerDeclarationFromResource(resource)
}

func withCalphadRevisionGovernanceMetadata(
	metadata domain.JSONMap,
	declaration domain.CalphadOwnerDeclaration,
) domain.JSONMap {
	cloned := withCalphadAssessmentPressureMetadata(metadata, declaration.AssessmentPressureLimitsPa)
	cloned[domain.CalphadOwnerDeclarationMetadataKey] = calphadOwnerDeclarationJSON(declaration)
	return cloned
}

func calphadRevisionOwnerDeclaration(
	revision domain.CalphadRevisionRecord,
) (domain.CalphadOwnerDeclaration, bool) {
	declaration, ok := calphadOwnerDeclarationFromValue(
		revision.Metadata[domain.CalphadOwnerDeclarationMetadataKey],
	)
	return declaration, ok && declaration.DatabaseFormat == revision.DatabaseFormat &&
		declaration.AssessmentPressureLimitsPa == revision.AssessmentPressureLimitsPa
}

func validateCalphadRevisionGovernanceBinding(revision domain.CalphadRevisionRecord) error {
	if !validCalphadDatabaseFormat(revision.DatabaseFormat) {
		return ErrCalphadDatabaseFormatInvalid
	}
	if !calphadRevisionPressureBindingValid(revision) {
		return ErrCalphadPressureLimitsInvalid
	}
	if _, ok := calphadRevisionOwnerDeclaration(revision); !ok {
		return ErrCalphadOwnerDeclarationInvalid
	}
	return nil
}

func calphadRevisionPressureBindingValid(revision domain.CalphadRevisionRecord) bool {
	metadataLimits, ok := calphadAssessmentPressureLimitsFromValue(
		revision.Metadata[domain.CalphadAssessmentPressureLimitsMetadataKey],
	)
	return ok && validCalphadAssessmentPressureLimits(revision.AssessmentPressureLimitsPa) &&
		metadataLimits == revision.AssessmentPressureLimitsPa
}

func calphadValidationPressureMatchesRevision(
	validation domain.CalphadValidationRecord,
	revision domain.CalphadRevisionRecord,
) bool {
	return validCalphadAssessmentPressureLimits(validation.AssessmentPressureLimitsPa) &&
		calphadRevisionPressureBindingValid(revision) &&
		validation.AssessmentPressureLimitsPa == revision.AssessmentPressureLimitsPa &&
		validCalphadDatabaseFormat(validation.DatabaseFormat) &&
		validation.DatabaseFormat == revision.DatabaseFormat
}

func calphadValidationPressureMetadataValid(validation domain.CalphadValidationRecord) bool {
	metadataLimits, ok := calphadAssessmentPressureLimitsFromValue(
		validation.Metadata[domain.CalphadAssessmentPressureLimitsMetadataKey],
	)
	return ok && validCalphadAssessmentPressureLimits(validation.AssessmentPressureLimitsPa) &&
		metadataLimits == validation.AssessmentPressureLimitsPa
}

func cloneCalphadRevision(record domain.CalphadRevisionRecord) domain.CalphadRevisionRecord {
	record.Metadata = deepCloneCalphadJSONMap(record.Metadata)
	return record
}

func cloneCalphadValidation(record domain.CalphadValidationRecord) domain.CalphadValidationRecord {
	record.Metadata = deepCloneCalphadJSONMap(record.Metadata)
	return record
}

func isCalphadCatalogResource(resource domain.ResourceRecord) bool {
	name := strings.ToLower(strings.TrimSpace(resource.OriginalName))
	contentType := strings.ToLower(strings.TrimSpace(resource.ContentType))
	supportedName := strings.HasSuffix(name, ".tdb") || strings.HasSuffix(name, ".dat")
	supportedContentType := contentType == "" || contentType == "application/octet-stream" ||
		contentType == "text/plain" || contentType == "application/x-thermocalc-tdb"
	return supportedName && supportedContentType
}

func validateCalphadResourceBinding(resource domain.ResourceRecord) error {
	if !isCalphadCatalogResource(resource) || strings.TrimSpace(resource.Status) != "active" {
		return ErrNotFound
	}
	if !calphadSHA256Pattern.MatchString(strings.ToLower(strings.TrimSpace(resource.SHA256))) || resource.SizeBytes <= 0 {
		return fmt.Errorf("CALPHAD resource requires immutable SHA-256 and positive size: %w", ErrConflict)
	}
	return nil
}

func calphadInputBytesMatch(sha string, size int64, payload []byte) bool {
	if size <= 0 || size > domain.CalphadMaxInputBytes || int64(len(payload)) != size ||
		!calphadSHA256Pattern.MatchString(sha) {
		return false
	}
	digest := sha256.Sum256(payload)
	return hex.EncodeToString(digest[:]) == sha
}

func validateCalphadInputBytes(input domain.CreateCalphadRevisionInput, resource domain.ResourceRecord) error {
	sha := strings.ToLower(strings.TrimSpace(resource.SHA256))
	if !calphadInputBytesMatch(sha, resource.SizeBytes, input.InputBytes) {
		return ErrConflict
	}
	return nil
}

func retainedCalphadInputMatches(
	revision domain.CalphadRevisionRecord,
	blob calphadInputBlob,
	found bool,
) bool {
	return found && blob.SHA256 == revision.SHA256 && blob.SizeBytes == revision.SizeBytes &&
		calphadInputBytesMatch(revision.SHA256, revision.SizeBytes, blob.Payload)
}

func persistMemoryCalphadInputBlob(
	blobs map[string]calphadInputBlob,
	revision domain.CalphadRevisionRecord,
	payload []byte,
) error {
	if !calphadInputBytesMatch(revision.SHA256, revision.SizeBytes, payload) {
		return ErrConflict
	}
	if existing, found := blobs[revision.SHA256]; found {
		if !retainedCalphadInputMatches(revision, existing, true) || !bytes.Equal(existing.Payload, payload) {
			return ErrConflict
		}
		return nil
	}
	blobs[revision.SHA256] = calphadInputBlob{
		SHA256: revision.SHA256, SizeBytes: revision.SizeBytes, Payload: append([]byte(nil), payload...),
	}
	return nil
}

func normalizedCalphadRevisionInput(
	input domain.CreateCalphadRevisionInput,
	resource domain.ResourceRecord,
	declaration domain.CalphadOwnerDeclaration,
) domain.CalphadRevisionRecord {
	createdAt := input.CreatedAt
	if createdAt.IsZero() {
		createdAt = domain.Now()
	}
	return domain.CalphadRevisionRecord{
		RevisionID:                 domain.NewID("calphad_revision"),
		ResourceID:                 resource.ResourceID,
		OwnerUserID:                resource.OwnerUserID,
		OwnerOrgID:                 resource.OwnerOrgID,
		SHA256:                     strings.ToLower(strings.TrimSpace(resource.SHA256)),
		SizeBytes:                  resource.SizeBytes,
		DatabaseFormat:             declaration.DatabaseFormat,
		AssessmentPressureLimitsPa: input.AssessmentPressureLimitsPa,
		ParentRevisionID:           strings.TrimSpace(input.ParentRevisionID),
		CreatedByUserID:            strings.TrimSpace(input.CreatedByUserID),
		CreatedAt:                  createdAt.UTC(),
		Metadata: withCalphadRevisionGovernanceMetadata(
			domain.JSONMap{"server_managed": true}, declaration,
		),
	}
}

func pendingCalphadValidation(revision domain.CalphadRevisionRecord) domain.CalphadValidationRecord {
	return withCalphadEvidenceRetention(domain.CalphadValidationRecord{
		ValidationID:               domain.NewID("calphad_validation"),
		RevisionID:                 revision.RevisionID,
		ResourceID:                 revision.ResourceID,
		DatabaseSHA256:             revision.SHA256,
		DatabaseSizeBytes:          revision.SizeBytes,
		DatabaseFormat:             revision.DatabaseFormat,
		AssessmentPressureLimitsPa: revision.AssessmentPressureLimitsPa,
		Status:                     "pending",
		Operation:                  "registration",
		CreatedByAuthority:         "control_plane",
		CreatedAt:                  revision.CreatedAt,
		Metadata: withCalphadAssessmentPressureMetadata(
			domain.JSONMap{"server_managed": true}, revision.AssessmentPressureLimitsPa,
		),
	}, false)
}

func validateCalphadRevisionPressureInput(input domain.CreateCalphadRevisionInput) error {
	if !validCalphadAssessmentPressureLimits(input.AssessmentPressureLimitsPa) {
		return ErrCalphadPressureLimitsInvalid
	}
	return nil
}

func validateExpectedCalphadBinding(input domain.CreateCalphadRevisionInput, resource domain.ResourceRecord) error {
	expectedSHA := strings.ToLower(strings.TrimSpace(input.ExpectedSHA256))
	expectedSize := input.ExpectedSizeBytes
	if expectedSHA == "" && expectedSize == 0 {
		return nil
	}
	if !calphadSHA256Pattern.MatchString(expectedSHA) || expectedSize <= 0 ||
		expectedSHA != strings.ToLower(strings.TrimSpace(resource.SHA256)) ||
		expectedSize != resource.SizeBytes {
		return ErrConflict
	}
	return nil
}

func validateParentRevision(parent domain.CalphadRevisionRecord, resource domain.ResourceRecord) error {
	if parent.OwnerUserID != resource.OwnerUserID || parent.OwnerOrgID != resource.OwnerOrgID || parent.ResourceID == resource.ResourceID {
		return ErrNotFound
	}
	return nil
}

func calphadOwnerVisible(ownerUserID, ownerOrgID, userID, orgID string) bool {
	if strings.TrimSpace(ownerUserID) != strings.TrimSpace(userID) {
		return false
	}
	ownerOrgID = strings.TrimSpace(ownerOrgID)
	return ownerOrgID == "" || ownerOrgID == strings.TrimSpace(orgID)
}

func validCalphadLedgerText(value string, maximum int) bool {
	value = strings.TrimSpace(value)
	if value == "" || len(value) > maximum {
		return false
	}
	for _, character := range value {
		if character < 32 || character == 127 {
			return false
		}
	}
	return true
}

func calphadRunStringValues(value any) ([]string, bool) {
	switch typed := value.(type) {
	case []string:
		return append([]string(nil), typed...), true
	case []any:
		values := make([]string, 0, len(typed))
		for _, item := range typed {
			text, ok := item.(string)
			if !ok {
				return nil, false
			}
			values = append(values, text)
		}
		return values, true
	default:
		return nil, false
	}
}

func calphadRunResourceDescriptors(value any) ([]domain.JSONMap, bool) {
	switch typed := value.(type) {
	case []domain.JSONMap:
		return typed, true
	case []map[string]any:
		descriptors := make([]domain.JSONMap, 0, len(typed))
		for _, descriptor := range typed {
			descriptors = append(descriptors, domain.JSONMap(descriptor))
		}
		return descriptors, true
	case []any:
		descriptors := make([]domain.JSONMap, 0, len(typed))
		for _, item := range typed {
			descriptor, ok := resourceMetadataMap(item)
			if !ok {
				return nil, false
			}
			descriptors = append(descriptors, descriptor)
		}
		return descriptors, true
	default:
		return nil, false
	}
}

// CalphadRunHasSelectedResourceBinding implements the same exact selected-file
// and descriptor authority checked by the production SQL writer. It is shared
// by the HTTP preflight and MemoryStore so tests cannot pass with a weaker
// authorization contract than PostgreSQL.
func CalphadRunHasSelectedResourceBinding(
	run domain.RunRecord,
	resourceID string,
	databaseSHA256 string,
	databaseSizeBytes int64,
	databaseFormat string,
	ownerDeclaration domain.CalphadOwnerDeclaration,
) bool {
	resourceID = strings.TrimSpace(resourceID)
	databaseSHA256 = strings.ToLower(strings.TrimSpace(databaseSHA256))
	if resourceID == "" || !calphadSHA256Pattern.MatchString(databaseSHA256) ||
		databaseSizeBytes <= 0 || !validCalphadDatabaseFormat(databaseFormat) {
		return false
	}
	selectedFiles, ok := calphadRunStringValues(run.Metadata["file_ids"])
	if !ok {
		return false
	}
	selectedCount := 0
	for _, selected := range selectedFiles {
		if selected == resourceID {
			selectedCount++
		}
	}
	if selectedCount != 1 {
		return false
	}
	descriptors, ok := calphadRunResourceDescriptors(run.Metadata["resource_descriptors"])
	if !ok {
		return false
	}
	candidateCount := 0
	exactCount := 0
	for _, descriptor := range descriptors {
		descriptorResourceID, _ := descriptor["resource_id"].(string)
		descriptorFileID, _ := descriptor["file_id"].(string)
		if descriptorResourceID != resourceID && descriptorFileID != resourceID {
			continue
		}
		candidateCount++
		descriptorType, _ := descriptor["type"].(string)
		bindingSchema, _ := descriptor["binding_schema"].(string)
		authority, _ := descriptor["authority"].(string)
		sha, _ := descriptor["sha256"].(string)
		size, sizeOK := calphadPolicyInt64(descriptor["size_bytes"])
		descriptorFormat, formatOK := descriptor["database_format"].(string)
		originalName, nameOK := descriptor["original_name"].(string)
		originalFormat, originalFormatOK := calphadDatabaseFormatForName(originalName)
		governanceScope, _ := descriptor["calphad_governance_scope"].(string)
		metadata, metadataOK := resourceMetadataMap(descriptor["metadata"])
		calphadMetadata, calphadMetadataOK := resourceMetadataMap(metadata["calphad"])
		if descriptorType != "selected_resource" || bindingSchema != "ultra.selected_resource.v1" ||
			authority != "control_resource_catalog" || descriptorResourceID != resourceID ||
			descriptorFileID != resourceID || !sizeOK || size != databaseSizeBytes ||
			strings.ToLower(strings.TrimSpace(sha)) != databaseSHA256 || !formatOK ||
			descriptorFormat != databaseFormat || !nameOK || !originalFormatOK ||
			originalFormat != databaseFormat || governanceScope != "owner_validation" ||
			!metadataOK || !calphadMetadataOK ||
			calphadMetadata["declaration_authority"] != "resource_owner" {
			continue
		}
		declaration, err := calphadOwnerDeclarationFromResource(domain.ResourceRecord{
			ResourceID: resourceID, OriginalName: originalName, Metadata: metadata,
		})
		if err != nil || declaration != ownerDeclaration {
			continue
		}
		exactCount++
	}
	return candidateCount == 1 && exactCount == 1
}

func (s *MemoryStore) CreateCalphadRevision(ctx context.Context, input domain.CreateCalphadRevisionInput) (domain.CalphadRevisionRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	resourceID := strings.TrimSpace(input.ResourceID)
	resource, ok := s.resources[resourceID]
	if !ok || !resourceVisibleToOwner(resource, input.OwnerUserID, input.OwnerOrgID) {
		return domain.CalphadRevisionRecord{}, ErrNotFound
	}
	if err := validateCalphadResourceBinding(resource); err != nil {
		return domain.CalphadRevisionRecord{}, err
	}
	declaration, err := calphadOwnerDeclarationFromResource(resource)
	if err != nil {
		return domain.CalphadRevisionRecord{}, err
	}
	if err := validateCalphadRevisionPressureInput(input); err != nil {
		return domain.CalphadRevisionRecord{}, err
	}
	if input.AssessmentPressureLimitsPa != declaration.AssessmentPressureLimitsPa {
		return domain.CalphadRevisionRecord{}, ErrCalphadPressureLimitsInvalid
	}
	if err := validateExpectedCalphadBinding(input, resource); err != nil {
		return domain.CalphadRevisionRecord{}, err
	}
	if err := validateCalphadInputBytes(input, resource); err != nil {
		return domain.CalphadRevisionRecord{}, err
	}
	if existing, ok := s.calphadRevisions[resourceID]; ok {
		if existing.SHA256 != strings.ToLower(strings.TrimSpace(resource.SHA256)) ||
			existing.SizeBytes != resource.SizeBytes || existing.OwnerUserID != resource.OwnerUserID ||
			existing.OwnerOrgID != resource.OwnerOrgID {
			return domain.CalphadRevisionRecord{}, ErrConflict
		}
		existingDeclaration, declarationOK := calphadRevisionOwnerDeclaration(existing)
		if governanceErr := validateCalphadRevisionGovernanceBinding(existing); governanceErr != nil {
			return domain.CalphadRevisionRecord{}, governanceErr
		}
		if !declarationOK || existingDeclaration != declaration {
			return domain.CalphadRevisionRecord{}, ErrCalphadOwnerDeclarationInvalid
		}
		if existing.AssessmentPressureLimitsPa != input.AssessmentPressureLimitsPa {
			return domain.CalphadRevisionRecord{}, ErrCalphadPressureLimitsInvalid
		}
		requestedParent := strings.TrimSpace(input.ParentRevisionID)
		if requestedParent != strings.TrimSpace(existing.ParentRevisionID) {
			return domain.CalphadRevisionRecord{}, ErrConflict
		}
		if err := persistMemoryCalphadInputBlob(s.calphadInputBlobs, existing, input.InputBytes); err != nil {
			return domain.CalphadRevisionRecord{}, err
		}
		return cloneCalphadRevision(existing), nil
	}
	if parentID := strings.TrimSpace(input.ParentRevisionID); parentID != "" {
		if !validCalphadLedgerText(parentID, 512) {
			return domain.CalphadRevisionRecord{}, ErrConflict
		}
		var parent domain.CalphadRevisionRecord
		found := false
		for _, candidate := range s.calphadRevisions {
			if candidate.RevisionID == parentID {
				parent, found = candidate, true
				break
			}
		}
		if !found || validateParentRevision(parent, resource) != nil {
			return domain.CalphadRevisionRecord{}, ErrNotFound
		}
	}
	revision := normalizedCalphadRevisionInput(input, resource, declaration)
	if err := persistMemoryCalphadInputBlob(s.calphadInputBlobs, revision, input.InputBytes); err != nil {
		return domain.CalphadRevisionRecord{}, err
	}
	s.calphadRevisions[resourceID] = cloneCalphadRevision(revision)
	s.calphadValidations = append(s.calphadValidations, pendingCalphadValidation(revision))
	return cloneCalphadRevision(revision), nil
}

func validateCalphadValidationInput(input domain.AppendCalphadValidationInput) error {
	status := strings.TrimSpace(input.Status)
	operation := strings.TrimSpace(input.Operation)
	if status != "input_validated" && status != "equilibrium_completed" && status != "scheil_completed" &&
		status != "failed" && status != "timeout" && status != "unsupported" {
		return ErrConflict
	}
	if operation != "inspect" && operation != "equilibrium" && operation != "scheil" {
		return ErrConflict
	}
	if !calphadSHA256Pattern.MatchString(strings.ToLower(strings.TrimSpace(input.DatabaseSHA256))) ||
		input.DatabaseSizeBytes <= 0 {
		return ErrConflict
	}
	if !validCalphadAssessmentPressureLimits(input.AssessmentPressureLimitsPa) {
		return ErrCalphadPressureLimitsInvalid
	}
	if !validCalphadDatabaseFormat(input.DatabaseFormat) {
		return ErrCalphadDatabaseFormatInvalid
	}
	declaration, declarationOK := calphadOwnerDeclarationFromValue(
		calphadOwnerDeclarationJSON(input.OwnerDeclaration),
	)
	if !declarationOK || declaration != input.OwnerDeclaration ||
		declaration.DatabaseFormat != input.DatabaseFormat ||
		declaration.AssessmentPressureLimitsPa != input.AssessmentPressureLimitsPa {
		return ErrCalphadOwnerDeclarationInvalid
	}
	inventorySHA := strings.ToLower(strings.TrimSpace(input.DatabaseInventorySHA256))
	requiresInventory := operation == "equilibrium" || operation == "scheil" || status == "input_validated"
	if (requiresInventory && !calphadSHA256Pattern.MatchString(inventorySHA)) ||
		(!requiresInventory && inventorySHA != "" && !calphadSHA256Pattern.MatchString(inventorySHA)) ||
		!calphadSHA256Pattern.MatchString(strings.ToLower(strings.TrimSpace(input.RequestSHA256))) {
		return ErrConflict
	}
	if (status == "input_validated" && operation != "inspect") ||
		(status == "equilibrium_completed" && operation != "equilibrium") ||
		(status == "scheil_completed" && operation != "scheil") {
		return ErrConflict
	}
	if !validCalphadFailureFields(
		status, operation, input.FailureDomain, input.FailureStage, input.FailureCode,
	) {
		return ErrConflict
	}
	if !calphadImagePattern.MatchString(strings.ToLower(strings.TrimSpace(input.RuntimeImageID))) ||
		strings.TrimSpace(input.PycalphadVersion) != domain.CalphadPycalphadVersion ||
		!validCalphadLedgerText(input.RunID, 512) ||
		!validCalphadLedgerText(input.LeaseWorkerID, 512) || !validCalphadLedgerText(input.LeaseToken, 512) {
		return ErrConflict
	}
	if input.LeaseWorkerID != strings.TrimSpace(input.LeaseWorkerID) ||
		input.LeaseToken != strings.TrimSpace(input.LeaseToken) {
		return ErrConflict
	}
	evidenceSHA := strings.ToLower(strings.TrimSpace(input.EvidenceSHA256))
	inspectionEvidenceSHA := strings.ToLower(strings.TrimSpace(input.InspectionEvidenceSHA256))
	if ((operation == "equilibrium" || operation == "scheil") &&
		!calphadSHA256Pattern.MatchString(inspectionEvidenceSHA)) ||
		(operation != "equilibrium" && operation != "scheil" && inspectionEvidenceSHA != "") {
		return ErrConflict
	}
	evidencePath := strings.TrimSpace(input.EvidencePath)
	hasAnyEvidence := evidencePath != "" || evidenceSHA != "" || input.EvidenceSizeBytes != 0 ||
		len(input.EvidenceBytes) != 0
	if !hasAnyEvidence {
		return ErrCalphadEvidenceRetentionRequired
	}
	if hasAnyEvidence {
		artifactDirectory := operation
		if operation == "inspect" {
			artifactDirectory = "inspection"
		}
		expectedPath := "/outputs/calphad/" + artifactDirectory + "/" + evidenceSHA + ".json"
		if !calphadSHA256Pattern.MatchString(evidenceSHA) || evidencePath != expectedPath ||
			input.EvidenceSizeBytes <= 0 || input.EvidenceSizeBytes > maxCalphadEvidenceBytes {
			return ErrConflict
		}
		if int64(len(input.EvidenceBytes)) != input.EvidenceSizeBytes {
			return ErrConflict
		}
		digest := sha256.Sum256(input.EvidenceBytes)
		if hex.EncodeToString(digest[:]) != evidenceSHA {
			return ErrConflict
		}
	}
	if strings.TrimSpace(input.CreatedByAuthority) != "trusted_worker" {
		return ErrConflict
	}
	return nil
}

func validCalphadFailureFields(
	status string,
	operation string,
	failureDomain domain.CalphadFailureDomain,
	failureStage domain.CalphadFailureStage,
	failureCode domain.CalphadFailureCode,
) bool {
	if !calphadFailureStatus(status) {
		return failureDomain == "" && failureStage == "" && failureCode == ""
	}
	if !failureDomain.Valid() || !failureStage.Valid() || !failureCode.Valid() {
		return false
	}
	if (operation == "inspect" && failureStage == domain.CalphadFailureStageSolver) ||
		((operation == "equilibrium" || operation == "scheil") &&
			failureStage == domain.CalphadFailureStageParse) {
		return false
	}
	switch failureCode {
	case domain.CalphadFailureCodeParseFailed:
		return status == "failed" && failureStage == domain.CalphadFailureStageParse &&
			(failureDomain == domain.CalphadFailureDomainInput ||
				failureDomain == domain.CalphadFailureDomainScientific)
	case domain.CalphadFailureCodeParseTimeout:
		return status == "timeout" && failureDomain == domain.CalphadFailureDomainScientific &&
			failureStage == domain.CalphadFailureStageParse
	case domain.CalphadFailureCodeParseUnsupported:
		return status == "unsupported" && failureDomain == domain.CalphadFailureDomainInput &&
			failureStage == domain.CalphadFailureStageParse
	case domain.CalphadFailureCodeSolverFailed:
		return status == "failed" && failureStage == domain.CalphadFailureStageSolver &&
			(failureDomain == domain.CalphadFailureDomainInput ||
				failureDomain == domain.CalphadFailureDomainScientific)
	case domain.CalphadFailureCodeSolverTimeout:
		return status == "timeout" && failureDomain == domain.CalphadFailureDomainScientific &&
			failureStage == domain.CalphadFailureStageSolver
	case domain.CalphadFailureCodeSolverUnsupported:
		return status == "unsupported" && failureDomain == domain.CalphadFailureDomainScientific &&
			failureStage == domain.CalphadFailureStageSolver
	case domain.CalphadFailureCodeResultInvalid:
		return status == "failed" && failureDomain == domain.CalphadFailureDomainScientific &&
			failureStage == domain.CalphadFailureStageResultValidation
	case domain.CalphadFailureCodeRuntimeInternalFailure:
		return status == "failed" && failureDomain == domain.CalphadFailureDomainPlatform &&
			((operation == "inspect" && failureStage == domain.CalphadFailureStageParse) ||
				((operation == "equilibrium" || operation == "scheil") &&
					failureStage == domain.CalphadFailureStageSolver))
	case domain.CalphadFailureCodeSandboxFailed:
		return status == "failed" && failureDomain == domain.CalphadFailureDomainPlatform &&
			failureStage == domain.CalphadFailureStageSandboxRuntime
	case domain.CalphadFailureCodeSandboxTimeout:
		return status == "timeout" && failureDomain == domain.CalphadFailureDomainPlatform &&
			failureStage == domain.CalphadFailureStageSandboxRuntime
	default:
		return false
	}
}

func calphadFailureStatus(status string) bool {
	return status == "failed" || status == "timeout" || status == "unsupported"
}

func normalizedCalphadValidation(input domain.AppendCalphadValidationInput, revision domain.CalphadRevisionRecord) domain.CalphadValidationRecord {
	createdAt := input.CreatedAt
	if createdAt.IsZero() {
		createdAt = domain.Now()
	}
	record := domain.CalphadValidationRecord{
		ValidationID:               domain.NewID("calphad_validation"),
		RevisionID:                 revision.RevisionID,
		ResourceID:                 revision.ResourceID,
		DatabaseSHA256:             strings.ToLower(strings.TrimSpace(input.DatabaseSHA256)),
		DatabaseSizeBytes:          input.DatabaseSizeBytes,
		DatabaseFormat:             input.DatabaseFormat,
		AssessmentPressureLimitsPa: input.AssessmentPressureLimitsPa,
		DatabaseInventorySHA256:    strings.ToLower(strings.TrimSpace(input.DatabaseInventorySHA256)),
		RequestSHA256:              strings.ToLower(strings.TrimSpace(input.RequestSHA256)),
		Status:                     strings.TrimSpace(input.Status),
		Operation:                  strings.TrimSpace(input.Operation),
		FailureDomain:              domain.CalphadFailureDomain(strings.TrimSpace(string(input.FailureDomain))),
		FailureStage:               domain.CalphadFailureStage(strings.TrimSpace(string(input.FailureStage))),
		FailureCode:                domain.CalphadFailureCode(strings.TrimSpace(string(input.FailureCode))),
		EvidencePath:               strings.TrimSpace(input.EvidencePath),
		EvidenceSHA256:             strings.ToLower(strings.TrimSpace(input.EvidenceSHA256)),
		EvidenceSizeBytes:          input.EvidenceSizeBytes,
		RuntimeImageID:             strings.ToLower(strings.TrimSpace(input.RuntimeImageID)),
		PycalphadVersion:           strings.TrimSpace(input.PycalphadVersion),
		RunID:                      strings.TrimSpace(input.RunID),
		InspectionEvidenceSHA256: strings.ToLower(
			strings.TrimSpace(input.InspectionEvidenceSHA256),
		),
		EvidenceContractVersion: domain.CalphadEvidenceContractVersion,
		CreatedByAuthority:      strings.TrimSpace(input.CreatedByAuthority),
		CreatedAt:               createdAt.UTC(),
		Metadata: withCalphadAssessmentPressureMetadata(domain.JSONMap{
			"server_managed": true,
			"revision_id":    revision.RevisionID,
		}, input.AssessmentPressureLimitsPa),
	}
	return withCalphadEvidenceRetention(record, record.EvidenceSHA256 != "")
}

func sameCalphadValidationReplay(existing, candidate domain.CalphadValidationRecord) bool {
	return existing.RevisionID == candidate.RevisionID &&
		existing.ResourceID == candidate.ResourceID &&
		existing.DatabaseSHA256 == candidate.DatabaseSHA256 &&
		existing.DatabaseSizeBytes == candidate.DatabaseSizeBytes &&
		existing.DatabaseFormat == candidate.DatabaseFormat &&
		existing.AssessmentPressureLimitsPa == candidate.AssessmentPressureLimitsPa &&
		existing.DatabaseInventorySHA256 == candidate.DatabaseInventorySHA256 &&
		existing.RequestSHA256 == candidate.RequestSHA256 &&
		existing.Status == candidate.Status && existing.Operation == candidate.Operation &&
		existing.FailureDomain == candidate.FailureDomain &&
		existing.FailureStage == candidate.FailureStage &&
		existing.FailureCode == candidate.FailureCode &&
		existing.EvidencePath == candidate.EvidencePath &&
		existing.EvidenceSHA256 == candidate.EvidenceSHA256 &&
		existing.EvidenceSizeBytes == candidate.EvidenceSizeBytes &&
		existing.RuntimeImageID == candidate.RuntimeImageID &&
		existing.PycalphadVersion == candidate.PycalphadVersion &&
		existing.RunID == candidate.RunID &&
		existing.InspectionEvidenceSHA256 == candidate.InspectionEvidenceSHA256 &&
		existing.EvidenceContractVersion == candidate.EvidenceContractVersion &&
		existing.CreatedByAuthority == candidate.CreatedByAuthority
}

func retainedCalphadBlobMatches(record domain.CalphadValidationRecord, blob calphadEvidenceBlob, found bool) bool {
	if !found || record.EvidenceSHA256 == "" || blob.SHA256 != record.EvidenceSHA256 ||
		blob.SizeBytes != record.EvidenceSizeBytes || int64(len(blob.Payload)) != record.EvidenceSizeBytes {
		return false
	}
	digest := sha256.Sum256(blob.Payload)
	return hex.EncodeToString(digest[:]) == record.EvidenceSHA256
}

func memoryCalphadInspectionRetained(
	records []domain.CalphadValidationRecord,
	blobs map[string]calphadEvidenceBlob,
	candidate domain.CalphadValidationRecord,
) bool {
	if candidate.Operation != "equilibrium" {
		return true
	}
	for _, inspection := range records {
		if inspection.RevisionID != candidate.RevisionID || inspection.RunID != candidate.RunID ||
			inspection.Operation != "inspect" || inspection.Status != "input_validated" ||
			inspection.RuntimeImageID != candidate.RuntimeImageID ||
			inspection.DatabaseInventorySHA256 != candidate.DatabaseInventorySHA256 ||
			inspection.EvidenceContractVersion != domain.CalphadEvidenceContractVersion ||
			inspection.EvidenceSHA256 != candidate.InspectionEvidenceSHA256 {
			continue
		}
		blob, found := blobs[inspection.EvidenceSHA256]
		return retainedCalphadBlobMatches(inspection, blob, found)
	}
	return false
}

func (s *MemoryStore) AppendCalphadValidation(ctx context.Context, input domain.AppendCalphadValidationInput) (domain.CalphadValidationRecord, error) {
	_ = ctx
	if err := validateCalphadValidationInput(input); err != nil {
		return domain.CalphadValidationRecord{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	run, runFound := s.runs[strings.TrimSpace(input.RunID)]
	lease, leaseFound := s.leases[strings.TrimSpace(input.RunID)]
	runOrgID, _ := run.Metadata["org_id"].(string)
	if !runFound || run.Status != domain.RunStatusRunning || !leaseFound ||
		!lease.LeaseExpiresAt.After(domain.Now()) ||
		run.UserID != strings.TrimSpace(input.OwnerUserID) ||
		(strings.TrimSpace(input.OwnerOrgID) != "" && strings.TrimSpace(runOrgID) != strings.TrimSpace(input.OwnerOrgID)) ||
		lease.WorkerID != input.LeaseWorkerID ||
		subtle.ConstantTimeCompare([]byte(lease.LeaseToken), []byte(input.LeaseToken)) != 1 {
		return domain.CalphadValidationRecord{}, ErrCalphadRunLeaseInvalid
	}
	if err := authorizeCalphadRuntime(run, input); err != nil {
		return domain.CalphadValidationRecord{}, err
	}
	resourceID := strings.TrimSpace(input.ResourceID)
	resource, ok := s.resources[resourceID]
	if !ok || !resourceVisibleToOwner(resource, input.OwnerUserID, input.OwnerOrgID) {
		return domain.CalphadValidationRecord{}, ErrNotFound
	}
	if err := validateCalphadResourceBinding(resource); err != nil {
		return domain.CalphadValidationRecord{}, err
	}
	currentDeclaration, err := calphadOwnerDeclarationFromResource(resource)
	if err != nil {
		return domain.CalphadValidationRecord{}, err
	}
	if !CalphadRunHasSelectedResourceBinding(
		run, resourceID, input.DatabaseSHA256, input.DatabaseSizeBytes,
		input.DatabaseFormat, input.OwnerDeclaration,
	) {
		return domain.CalphadValidationRecord{}, ErrConflict
	}
	revision, ok := s.calphadRevisions[resourceID]
	if !ok || revision.SHA256 != strings.ToLower(strings.TrimSpace(resource.SHA256)) ||
		revision.SizeBytes != resource.SizeBytes || revision.OwnerUserID != resource.OwnerUserID ||
		revision.OwnerOrgID != resource.OwnerOrgID {
		return domain.CalphadValidationRecord{}, ErrConflict
	}
	if revision.SHA256 != strings.ToLower(strings.TrimSpace(input.DatabaseSHA256)) ||
		revision.SizeBytes != input.DatabaseSizeBytes {
		return domain.CalphadValidationRecord{}, ErrConflict
	}
	storedDeclaration, declarationOK := calphadRevisionOwnerDeclaration(revision)
	if governanceErr := validateCalphadRevisionGovernanceBinding(revision); governanceErr != nil {
		return domain.CalphadValidationRecord{}, governanceErr
	}
	if !declarationOK || storedDeclaration != currentDeclaration ||
		storedDeclaration != input.OwnerDeclaration || revision.DatabaseFormat != input.DatabaseFormat {
		return domain.CalphadValidationRecord{}, ErrCalphadOwnerDeclarationInvalid
	}
	if revision.AssessmentPressureLimitsPa != input.AssessmentPressureLimitsPa ||
		!calphadRevisionPressureBindingValid(revision) {
		return domain.CalphadValidationRecord{}, ErrCalphadPressureLimitsInvalid
	}
	if blob, found := s.calphadInputBlobs[revision.SHA256]; !retainedCalphadInputMatches(revision, blob, found) {
		return domain.CalphadValidationRecord{}, ErrCalphadInputRetentionRequired
	}
	record := normalizedCalphadValidation(input, revision)
	if !memoryCalphadInspectionRetained(s.calphadValidations, s.calphadEvidenceBlobs, record) {
		return domain.CalphadValidationRecord{}, ErrCalphadInspectionRequired
	}
	for _, existing := range s.calphadValidations {
		if existing.RevisionID != revision.RevisionID || existing.RunID != record.RunID ||
			existing.Operation != record.Operation || record.EvidenceSHA256 == "" ||
			existing.EvidenceSHA256 != record.EvidenceSHA256 {
			continue
		}
		if sameCalphadValidationReplay(existing, record) {
			if record.EvidenceSHA256 != "" {
				blob, found := s.calphadEvidenceBlobs[record.EvidenceSHA256]
				if !found || blob.SizeBytes != record.EvidenceSizeBytes ||
					!bytes.Equal(blob.Payload, input.EvidenceBytes) {
					return domain.CalphadValidationRecord{}, ErrConflict
				}
			}
			return cloneCalphadValidation(existing), nil
		}
		return domain.CalphadValidationRecord{}, ErrConflict
	}
	if record.EvidenceSHA256 != "" {
		if existing, found := s.calphadEvidenceBlobs[record.EvidenceSHA256]; found {
			if existing.SizeBytes != record.EvidenceSizeBytes ||
				!bytes.Equal(existing.Payload, input.EvidenceBytes) {
				return domain.CalphadValidationRecord{}, ErrConflict
			}
		} else {
			s.calphadEvidenceBlobs[record.EvidenceSHA256] = calphadEvidenceBlob{
				SHA256: record.EvidenceSHA256, SizeBytes: record.EvidenceSizeBytes,
				Payload: append([]byte(nil), input.EvidenceBytes...),
			}
		}
	}
	s.calphadValidations = append(s.calphadValidations, cloneCalphadValidation(record))
	return cloneCalphadValidation(record), nil
}

func validationsForRevision(
	records []domain.CalphadValidationRecord,
	blobs map[string]calphadEvidenceBlob,
	revisionID string,
) []domain.CalphadValidationRecord {
	out := make([]domain.CalphadValidationRecord, 0)
	for _, record := range records {
		if record.RevisionID == revisionID {
			blob, found := blobs[record.EvidenceSHA256]
			retained := retainedCalphadBlobMatches(record, blob, found) &&
				memoryCalphadInspectionRetained(records, blobs, record)
			out = append(out, cloneCalphadValidation(withCalphadEvidenceRetention(record, retained)))
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].CreatedAt.Equal(out[j].CreatedAt) {
			return out[i].ValidationID > out[j].ValidationID
		}
		return out[i].CreatedAt.After(out[j].CreatedAt)
	})
	return out
}

func ledgerFromRecords(revision domain.CalphadRevisionRecord, validations []domain.CalphadValidationRecord) domain.CalphadLedgerRecord {
	clonedValidations := make([]domain.CalphadValidationRecord, len(validations))
	for index, validation := range validations {
		clonedValidations[index] = cloneCalphadValidation(validation)
	}
	ledger := domain.CalphadLedgerRecord{
		Revision:    cloneCalphadRevision(revision),
		Validations: clonedValidations,
	}
	if len(clonedValidations) > 0 {
		latest := cloneCalphadValidation(clonedValidations[0])
		ledger.LatestValidation = &latest
	}
	return ledger
}

func validateCalphadLedgerPageInput(input domain.GetCalphadLedgerPageInput) error {
	if input.Limit < 1 || input.Limit > domain.CalphadLedgerMaximumLimit {
		return ErrConflict
	}
	hasTime := !input.BeforeCreatedAt.IsZero()
	hasID := strings.TrimSpace(input.BeforeValidationID) != ""
	if hasTime != hasID || (hasID && !validCalphadLedgerText(input.BeforeValidationID, 512)) {
		return ErrNotFound
	}
	return nil
}

func calphadValidationsMatchRevision(
	revision domain.CalphadRevisionRecord,
	validations []domain.CalphadValidationRecord,
) bool {
	for _, validation := range validations {
		if !calphadValidationPressureMatchesRevision(validation, revision) {
			return false
		}
	}
	return true
}

func (s *MemoryStore) GetCalphadLedgerPageForOwner(
	ctx context.Context,
	input domain.GetCalphadLedgerPageInput,
) (domain.CalphadLedgerRecord, error) {
	_ = ctx
	if err := validateCalphadLedgerPageInput(input); err != nil {
		return domain.CalphadLedgerRecord{}, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	revision, ok := s.calphadRevisions[strings.TrimSpace(input.ResourceID)]
	if !ok || !calphadOwnerVisible(
		revision.OwnerUserID, revision.OwnerOrgID, input.OwnerUserID, input.OwnerOrgID,
	) || (strings.TrimSpace(input.ExpectedRevisionID) != "" &&
		revision.RevisionID != strings.TrimSpace(input.ExpectedRevisionID)) {
		return domain.CalphadLedgerRecord{}, ErrNotFound
	}
	if blob, found := s.calphadInputBlobs[revision.SHA256]; !retainedCalphadInputMatches(revision, blob, found) {
		return domain.CalphadLedgerRecord{}, ErrCalphadInputRetentionRequired
	}
	if err := validateCalphadRevisionGovernanceBinding(revision); err != nil {
		return domain.CalphadLedgerRecord{}, err
	}
	all := validationsForRevision(
		s.calphadValidations, s.calphadEvidenceBlobs, revision.RevisionID,
	)
	start := 0
	if !input.BeforeCreatedAt.IsZero() {
		anchorFound := false
		for index, validation := range all {
			if validation.ValidationID == strings.TrimSpace(input.BeforeValidationID) &&
				validation.CreatedAt.Equal(input.BeforeCreatedAt) {
				start = index + 1
				anchorFound = true
				break
			}
		}
		if !anchorFound {
			return domain.CalphadLedgerRecord{}, ErrNotFound
		}
	}
	end := start + input.Limit + 1
	if end > len(all) {
		end = len(all)
	}
	window := all[start:end]
	hasMore := len(window) > input.Limit
	if hasMore {
		window = window[:input.Limit]
	}
	if !calphadValidationsMatchRevision(revision, window) ||
		(len(all) > 0 && !calphadValidationPressureMatchesRevision(all[0], revision)) {
		return domain.CalphadLedgerRecord{}, ErrCalphadPressureLimitsInvalid
	}
	ledger := ledgerFromRecords(revision, window)
	if len(all) > 0 {
		latest := cloneCalphadValidation(all[0])
		ledger.LatestValidation = &latest
	}
	ledger.HasMore = hasMore
	if hasMore && len(window) > 0 {
		ledger.NextCreatedAt = window[len(window)-1].CreatedAt
		ledger.NextValidationID = window[len(window)-1].ValidationID
	}
	return ledger, nil
}

func (s *MemoryStore) GetCalphadValidationEvidenceForOwner(
	ctx context.Context,
	resourceID, validationID, userID, orgID string,
) (domain.CalphadValidationEvidenceRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	revision, ok := s.calphadRevisions[strings.TrimSpace(resourceID)]
	if !ok || !calphadOwnerVisible(revision.OwnerUserID, revision.OwnerOrgID, userID, orgID) {
		return domain.CalphadValidationEvidenceRecord{}, ErrNotFound
	}
	if err := validateCalphadRevisionGovernanceBinding(revision); err != nil {
		return domain.CalphadValidationEvidenceRecord{}, err
	}
	for _, validation := range s.calphadValidations {
		if validation.ValidationID != strings.TrimSpace(validationID) ||
			validation.RevisionID != revision.RevisionID || validation.ResourceID != revision.ResourceID ||
			validation.DatabaseSHA256 != revision.SHA256 || validation.DatabaseSizeBytes != revision.SizeBytes ||
			validation.DatabaseFormat != revision.DatabaseFormat {
			continue
		}
		blob, found := s.calphadEvidenceBlobs[validation.EvidenceSHA256]
		if validation.EvidenceContractVersion != domain.CalphadEvidenceContractVersion ||
			!retainedCalphadBlobMatches(validation, blob, found) ||
			!memoryCalphadInspectionRetained(s.calphadValidations, s.calphadEvidenceBlobs, validation) {
			return domain.CalphadValidationEvidenceRecord{}, ErrCalphadEvidenceRetentionRequired
		}
		return domain.CalphadValidationEvidenceRecord{
			ValidationID: validation.ValidationID, RevisionID: validation.RevisionID,
			ResourceID: validation.ResourceID, SHA256: validation.EvidenceSHA256,
			SizeBytes: validation.EvidenceSizeBytes, Bytes: append([]byte(nil), blob.Payload...),
		}, nil
	}
	return domain.CalphadValidationEvidenceRecord{}, ErrNotFound
}

func (s *MemoryStore) GetCalphadLedgerForOwner(ctx context.Context, resourceID, userID, orgID string) (domain.CalphadLedgerRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	revision, ok := s.calphadRevisions[strings.TrimSpace(resourceID)]
	if !ok || !calphadOwnerVisible(revision.OwnerUserID, revision.OwnerOrgID, userID, orgID) {
		return domain.CalphadLedgerRecord{}, ErrNotFound
	}
	if blob, found := s.calphadInputBlobs[revision.SHA256]; !retainedCalphadInputMatches(revision, blob, found) {
		return domain.CalphadLedgerRecord{}, ErrCalphadInputRetentionRequired
	}
	if err := validateCalphadRevisionGovernanceBinding(revision); err != nil {
		return domain.CalphadLedgerRecord{}, err
	}
	validations := validationsForRevision(s.calphadValidations, s.calphadEvidenceBlobs, revision.RevisionID)
	for _, validation := range validations {
		if !calphadValidationPressureMatchesRevision(validation, revision) {
			return domain.CalphadLedgerRecord{}, ErrCalphadPressureLimitsInvalid
		}
	}
	return ledgerFromRecords(revision, validations), nil
}

func (s *MemoryStore) GetRetainedCalphadInspectionForOwner(
	ctx context.Context,
	input domain.GetRetainedCalphadInspectionInput,
) (domain.CalphadValidationRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	revision, ok := s.calphadRevisions[strings.TrimSpace(input.ResourceID)]
	if !ok || !calphadOwnerVisible(
		revision.OwnerUserID, revision.OwnerOrgID, input.OwnerUserID, input.OwnerOrgID,
	) || revision.SHA256 != strings.ToLower(strings.TrimSpace(input.DatabaseSHA256)) ||
		revision.SizeBytes != input.DatabaseSizeBytes ||
		revision.DatabaseFormat != strings.TrimSpace(input.DatabaseFormat) {
		return domain.CalphadValidationRecord{}, ErrNotFound
	}
	for _, inspection := range s.calphadValidations {
		if inspection.RevisionID != revision.RevisionID || inspection.ResourceID != revision.ResourceID ||
			inspection.Operation != "inspect" || inspection.Status != "input_validated" ||
			inspection.RunID != strings.TrimSpace(input.RunID) ||
			inspection.RuntimeImageID != strings.ToLower(strings.TrimSpace(input.RuntimeImageID)) ||
			inspection.EvidenceSHA256 != strings.ToLower(strings.TrimSpace(input.EvidenceSHA256)) ||
			inspection.EvidenceContractVersion != domain.CalphadEvidenceContractVersion ||
			!calphadValidationPressureMatchesRevision(inspection, revision) {
			continue
		}
		blob, found := s.calphadEvidenceBlobs[inspection.EvidenceSHA256]
		if !retainedCalphadBlobMatches(inspection, blob, found) {
			return domain.CalphadValidationRecord{}, ErrCalphadEvidenceRetentionRequired
		}
		return cloneCalphadValidation(withCalphadEvidenceRetention(inspection, true)), nil
	}
	return domain.CalphadValidationRecord{}, ErrNotFound
}

func (s *MemoryStore) GetCalphadRevisionInputForOwner(
	ctx context.Context,
	resourceID, userID, orgID string,
) (domain.CalphadRevisionInputRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	revision, ok := s.calphadRevisions[strings.TrimSpace(resourceID)]
	if !ok || !calphadOwnerVisible(revision.OwnerUserID, revision.OwnerOrgID, userID, orgID) {
		return domain.CalphadRevisionInputRecord{}, ErrNotFound
	}
	if err := validateCalphadRevisionGovernanceBinding(revision); err != nil {
		return domain.CalphadRevisionInputRecord{}, err
	}
	blob, found := s.calphadInputBlobs[revision.SHA256]
	if !retainedCalphadInputMatches(revision, blob, found) {
		return domain.CalphadRevisionInputRecord{}, ErrCalphadInputRetentionRequired
	}
	return domain.CalphadRevisionInputRecord{
		RevisionID:     revision.RevisionID,
		ResourceID:     revision.ResourceID,
		SHA256:         revision.SHA256,
		SizeBytes:      revision.SizeBytes,
		DatabaseFormat: revision.DatabaseFormat,
		Bytes:          append([]byte(nil), blob.Payload...),
	}, nil
}

func scanCalphadRevision(row pgx.Row) (domain.CalphadRevisionRecord, error) {
	var record domain.CalphadRevisionRecord
	var ownerOrgID, parentRevisionID, createdByUserID *string
	var pressureMinimum, pressureMaximum *float64
	var metadata []byte
	err := row.Scan(
		&record.RevisionID, &record.ResourceID, &record.OwnerUserID, &ownerOrgID,
		&record.SHA256, &record.SizeBytes, &record.DatabaseFormat,
		&pressureMinimum, &pressureMaximum,
		&parentRevisionID, &createdByUserID, &record.CreatedAt, &metadata,
	)
	if err != nil {
		return record, err
	}
	if pressureMinimum == nil || pressureMaximum == nil {
		return record, ErrCalphadPressureLimitsInvalid
	}
	record.AssessmentPressureLimitsPa = [2]float64{*pressureMinimum, *pressureMaximum}
	record.OwnerOrgID = stringFromPointer(ownerOrgID)
	record.ParentRevisionID = stringFromPointer(parentRevisionID)
	record.CreatedByUserID = stringFromPointer(createdByUserID)
	if err := json.Unmarshal(metadata, &record.Metadata); err != nil {
		return record, err
	}
	record.Metadata = deepCloneCalphadJSONMap(record.Metadata)
	if err := validateCalphadRevisionGovernanceBinding(record); err != nil {
		return domain.CalphadRevisionRecord{}, err
	}
	return record, nil
}

func scanCalphadValidation(row pgx.Row) (domain.CalphadValidationRecord, error) {
	var record domain.CalphadValidationRecord
	var databaseInventorySHA, requestSHA, evidencePath, evidenceSHA, runtimeImage *string
	var pycalphadVersion, runID, inspectionEvidenceSHA, evidenceContractVersion *string
	var failureDomain, failureStage, failureCode *string
	var evidenceSize *int64
	var pressureMinimum, pressureMaximum *float64
	var metadata []byte
	var evidenceBlobRetained bool
	err := row.Scan(
		&record.ValidationID, &record.RevisionID, &record.ResourceID,
		&record.DatabaseSHA256, &record.DatabaseSizeBytes, &record.DatabaseFormat,
		&pressureMinimum, &pressureMaximum,
		&databaseInventorySHA, &requestSHA,
		&record.Status, &record.Operation, &failureDomain, &failureStage, &failureCode,
		&evidencePath, &evidenceSHA, &evidenceSize,
		&runtimeImage, &pycalphadVersion, &runID, &inspectionEvidenceSHA,
		&evidenceContractVersion,
		&record.CreatedByAuthority, &record.CreatedAt, &metadata, &evidenceBlobRetained,
	)
	if err != nil {
		return record, err
	}
	if pressureMinimum == nil || pressureMaximum == nil {
		return record, ErrCalphadPressureLimitsInvalid
	}
	record.AssessmentPressureLimitsPa = [2]float64{*pressureMinimum, *pressureMaximum}
	record.EvidencePath = stringFromPointer(evidencePath)
	record.DatabaseInventorySHA256 = stringFromPointer(databaseInventorySHA)
	record.RequestSHA256 = stringFromPointer(requestSHA)
	record.FailureDomain = domain.CalphadFailureDomain(stringFromPointer(failureDomain))
	record.FailureStage = domain.CalphadFailureStage(stringFromPointer(failureStage))
	record.FailureCode = domain.CalphadFailureCode(stringFromPointer(failureCode))
	record.EvidenceSHA256 = stringFromPointer(evidenceSHA)
	if evidenceSize != nil {
		record.EvidenceSizeBytes = *evidenceSize
	}
	record.RuntimeImageID = stringFromPointer(runtimeImage)
	record.PycalphadVersion = stringFromPointer(pycalphadVersion)
	record.RunID = stringFromPointer(runID)
	record.InspectionEvidenceSHA256 = stringFromPointer(inspectionEvidenceSHA)
	record.EvidenceContractVersion = stringFromPointer(evidenceContractVersion)
	if err := json.Unmarshal(metadata, &record.Metadata); err != nil {
		return record, err
	}
	record.Metadata = deepCloneCalphadJSONMap(record.Metadata)
	if !calphadValidationPressureMetadataValid(record) {
		return domain.CalphadValidationRecord{}, ErrCalphadPressureLimitsInvalid
	}
	if !validCalphadDatabaseFormat(record.DatabaseFormat) {
		return domain.CalphadValidationRecord{}, ErrCalphadDatabaseFormatInvalid
	}
	return withCalphadEvidenceRetention(record, evidenceBlobRetained), nil
}

func persistCalphadEvidenceBlob(
	ctx context.Context,
	tx pgx.Tx,
	record domain.CalphadValidationRecord,
	payload []byte,
) error {
	if record.EvidenceSHA256 == "" {
		return nil
	}
	if _, err := tx.Exec(ctx, `
INSERT INTO control_calphad_evidence_blobs
 (evidence_sha256, evidence_size_bytes, encoding, payload, created_at)
VALUES ($1,$2,'raw',$3,$4)
ON CONFLICT (evidence_sha256) DO NOTHING`,
		record.EvidenceSHA256, record.EvidenceSizeBytes, payload, record.CreatedAt); err != nil {
		return mapPgError(err)
	}
	var storedSize int64
	var storedPayload []byte
	if err := tx.QueryRow(ctx, `
SELECT evidence_size_bytes, payload FROM control_calphad_evidence_blobs
WHERE evidence_sha256=$1`, record.EvidenceSHA256).Scan(&storedSize, &storedPayload); err != nil {
		return mapPgError(err)
	}
	if storedSize != record.EvidenceSizeBytes || !bytes.Equal(storedPayload, payload) {
		return ErrConflict
	}
	return nil
}

func loadCalphadInputBlob(
	ctx context.Context,
	db schemaQuerier,
	revision domain.CalphadRevisionRecord,
) (calphadInputBlob, error) {
	var blob calphadInputBlob
	if err := db.QueryRow(ctx, `
SELECT input_sha256, input_size_bytes, payload
FROM control_calphad_input_blobs
WHERE input_sha256=$1`, revision.SHA256).Scan(&blob.SHA256, &blob.SizeBytes, &blob.Payload); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return calphadInputBlob{}, ErrCalphadInputRetentionRequired
		}
		return calphadInputBlob{}, mapPgError(err)
	}
	if !retainedCalphadInputMatches(revision, blob, true) {
		return calphadInputBlob{}, ErrCalphadInputRetentionRequired
	}
	blob.Payload = append([]byte(nil), blob.Payload...)
	return blob, nil
}

func persistCalphadInputBlob(
	ctx context.Context,
	tx pgx.Tx,
	revision domain.CalphadRevisionRecord,
	payload []byte,
) error {
	if !calphadInputBytesMatch(revision.SHA256, revision.SizeBytes, payload) {
		return ErrConflict
	}
	if _, err := tx.Exec(ctx, `
INSERT INTO control_calphad_input_blobs
 (input_sha256, input_size_bytes, encoding, payload, created_at)
VALUES ($1,$2,'raw',$3,$4)
ON CONFLICT (input_sha256) DO NOTHING`,
		revision.SHA256, revision.SizeBytes, payload, revision.CreatedAt); err != nil {
		return mapPgError(err)
	}
	stored, err := loadCalphadInputBlob(ctx, tx, revision)
	if err != nil {
		return err
	}
	if !bytes.Equal(stored.Payload, payload) {
		return ErrConflict
	}
	return nil
}

func (s *PostgresStore) CreateCalphadRevision(ctx context.Context, input domain.CreateCalphadRevisionInput) (domain.CalphadRevisionRecord, error) {
	resource, err := scanControlResourceRow(s.pool.QueryRow(ctx, `
SELECT resource_id, owner_user_id, owner_org_id, owner_role, original_name, content_type, size_bytes, sha256,
       storage_uri, storage_path, source_type, resource_kind, source_uri, project_id, status, created_at,
       updated_at, deleted_at, retention_expires_at, metadata
FROM control_resources WHERE resource_id=$1 AND owner_user_id=$2
  AND (COALESCE(owner_org_id,'')='' OR owner_org_id=$3)`,
		strings.TrimSpace(input.ResourceID), strings.TrimSpace(input.OwnerUserID),
		strings.TrimSpace(input.OwnerOrgID)))
	if err != nil {
		return domain.CalphadRevisionRecord{}, mapPgError(err)
	}
	resourceRecord := resourceFromRow(resource)
	if err := validateCalphadResourceBinding(resourceRecord); err != nil {
		return domain.CalphadRevisionRecord{}, err
	}
	declaration, err := calphadOwnerDeclarationFromResource(resourceRecord)
	if err != nil {
		return domain.CalphadRevisionRecord{}, err
	}
	if err := validateCalphadRevisionPressureInput(input); err != nil {
		return domain.CalphadRevisionRecord{}, err
	}
	if input.AssessmentPressureLimitsPa != declaration.AssessmentPressureLimitsPa {
		return domain.CalphadRevisionRecord{}, ErrCalphadPressureLimitsInvalid
	}
	if err := validateExpectedCalphadBinding(input, resourceRecord); err != nil {
		return domain.CalphadRevisionRecord{}, err
	}
	if err := validateCalphadInputBytes(input, resourceRecord); err != nil {
		return domain.CalphadRevisionRecord{}, err
	}
	revision := normalizedCalphadRevisionInput(input, resourceRecord, declaration)
	inserted, err := scanCalphadRevision(s.pool.QueryRow(ctx, `
SELECT revision_id, resource_id, owner_user_id, owner_org_id, sha256, size_bytes,
       database_format, assessment_pressure_min_pa, assessment_pressure_max_pa,
       parent_revision_id, created_by_user_id, created_at, metadata
FROM public.ultra_create_calphad_revision_v1(
  $1::text, $2::text, $3::text, $4::text, $5::text, $6::bigint, $7::text,
  $8::double precision, $9::double precision, $10::bytea, $11::jsonb
)`,
		revision.ResourceID, revision.OwnerUserID, revision.OwnerOrgID,
		revision.ParentRevisionID, revision.SHA256, revision.SizeBytes,
		revision.DatabaseFormat, revision.AssessmentPressureLimitsPa[0],
		revision.AssessmentPressureLimitsPa[1], input.InputBytes,
		jsonBytes(revision.Metadata)))
	if err != nil {
		return domain.CalphadRevisionRecord{}, mapCalphadAppendError(err)
	}
	return inserted, nil
}
func (s *PostgresStore) AppendCalphadValidation(ctx context.Context, input domain.AppendCalphadValidationInput) (domain.CalphadValidationRecord, error) {
	if err := validateCalphadValidationInput(input); err != nil {
		return domain.CalphadValidationRecord{}, err
	}
	declaration := calphadOwnerDeclarationJSON(input.OwnerDeclaration)
	inserted, err := scanCalphadValidation(s.pool.QueryRow(ctx, `
SELECT validation_id, revision_id, resource_id, database_sha256, database_size_bytes,
       database_format, assessment_pressure_min_pa, assessment_pressure_max_pa,
       database_inventory_sha256, request_sha256, status, operation,
       failure_domain, failure_stage, failure_code, evidence_path,
       evidence_sha256, evidence_size_bytes, runtime_image_id, pycalphad_version,
       run_id, inspection_evidence_sha256, evidence_contract_version,
       created_by_authority, created_at, metadata, evidence_blob_retained
FROM public.ultra_append_calphad_validation_v1(
  $1::text, $2::text, $3::text, $4::text, $5::bigint, $6::text, $7::jsonb,
  $8::double precision, $9::double precision, $10::text, $11::text,
  $12::text, $13::text, $14::text, $15::text, $16::text, $17::text,
  $18::text, $19::bigint, $20::bytea, $21::text, $22::text, $23::text,
  $24::text, $25::text, $26::text, $27::jsonb
)`,
		strings.TrimSpace(input.ResourceID), strings.TrimSpace(input.OwnerUserID),
		strings.TrimSpace(input.OwnerOrgID), strings.ToLower(strings.TrimSpace(input.DatabaseSHA256)),
		input.DatabaseSizeBytes, strings.TrimSpace(input.DatabaseFormat), jsonBytes(declaration),
		input.AssessmentPressureLimitsPa[0], input.AssessmentPressureLimitsPa[1],
		strings.ToLower(strings.TrimSpace(input.DatabaseInventorySHA256)),
		strings.ToLower(strings.TrimSpace(input.RequestSHA256)), strings.TrimSpace(input.Status),
		strings.TrimSpace(input.Operation), strings.TrimSpace(string(input.FailureDomain)),
		strings.TrimSpace(string(input.FailureStage)), strings.TrimSpace(string(input.FailureCode)),
		strings.TrimSpace(input.EvidencePath), strings.ToLower(strings.TrimSpace(input.EvidenceSHA256)),
		input.EvidenceSizeBytes, input.EvidenceBytes,
		strings.ToLower(strings.TrimSpace(input.RuntimeImageID)),
		strings.TrimSpace(input.PycalphadVersion), strings.TrimSpace(input.RunID),
		strings.ToLower(strings.TrimSpace(input.InspectionEvidenceSHA256)),
		strings.TrimSpace(input.LeaseWorkerID), input.LeaseToken, jsonBytes(input.Metadata)))
	if err != nil {
		return domain.CalphadValidationRecord{}, mapCalphadAppendError(err)
	}
	return inserted, nil
}
func (s *PostgresStore) GetCalphadLedgerForOwner(ctx context.Context, resourceID, userID, orgID string) (domain.CalphadLedgerRecord, error) {
	revision, err := scanCalphadRevision(s.pool.QueryRow(ctx, `
SELECT cr.revision_id, cr.resource_id, cr.owner_user_id, cr.owner_org_id, cr.sha256, cr.size_bytes, cr.database_format,
       cr.assessment_pressure_min_pa, cr.assessment_pressure_max_pa,
       cr.parent_revision_id, cr.created_by_user_id, cr.created_at, cr.metadata
FROM control_calphad_revisions cr
WHERE cr.resource_id=$1 AND cr.owner_user_id=$2
  AND (COALESCE(cr.owner_org_id,'')='' OR cr.owner_org_id=$3)`,
		strings.TrimSpace(resourceID), strings.TrimSpace(userID), strings.TrimSpace(orgID)))
	if err != nil {
		return domain.CalphadLedgerRecord{}, mapPgError(err)
	}
	if _, err := loadCalphadInputBlob(ctx, s.pool, revision); err != nil {
		return domain.CalphadLedgerRecord{}, err
	}
	rows, err := s.pool.Query(ctx, `
SELECT validation.validation_id, validation.revision_id, validation.resource_id,
       validation.database_sha256, validation.database_size_bytes, validation.database_format,
	   validation.assessment_pressure_min_pa, validation.assessment_pressure_max_pa,
	   validation.database_inventory_sha256, validation.request_sha256,
       validation.status, validation.operation, validation.failure_domain,
       validation.failure_stage, validation.failure_code, validation.evidence_path,
       validation.evidence_sha256, validation.evidence_size_bytes, validation.runtime_image_id,
       validation.pycalphad_version, validation.run_id, validation.inspection_evidence_sha256,
	   validation.evidence_contract_version,
       validation.created_by_authority, validation.created_at, validation.metadata,
       EXISTS (SELECT 1 FROM control_calphad_evidence_blobs blob
               WHERE blob.evidence_sha256=validation.evidence_sha256
                 AND blob.evidence_size_bytes=validation.evidence_size_bytes
                 AND octet_length(blob.payload)=validation.evidence_size_bytes
				 AND encode(sha256(blob.payload), 'hex')=validation.evidence_sha256
				 AND validation.evidence_contract_version=$2)
	   AND (validation.operation <> 'equilibrium' OR EXISTS (
		 SELECT 1
		 FROM control_calphad_validation_events inspection
		 JOIN control_calphad_evidence_blobs inspection_blob
		   ON inspection_blob.evidence_sha256=inspection.evidence_sha256
		  AND inspection_blob.evidence_size_bytes=inspection.evidence_size_bytes
		  AND octet_length(inspection_blob.payload)=inspection.evidence_size_bytes
		  AND encode(sha256(inspection_blob.payload), 'hex')=inspection.evidence_sha256
		 WHERE inspection.revision_id=validation.revision_id
		   AND inspection.resource_id=validation.resource_id
		   AND inspection.database_sha256=validation.database_sha256
		   AND inspection.database_size_bytes=validation.database_size_bytes
		   AND inspection.database_format=validation.database_format
		   AND inspection.assessment_pressure_min_pa=validation.assessment_pressure_min_pa
		   AND inspection.assessment_pressure_max_pa=validation.assessment_pressure_max_pa
		   AND inspection.run_id=validation.run_id
		   AND inspection.operation='inspect' AND inspection.status='input_validated'
		   AND inspection.runtime_image_id=validation.runtime_image_id
		   AND inspection.database_inventory_sha256=validation.database_inventory_sha256
		   AND inspection.evidence_contract_version=$2
		   AND inspection.evidence_sha256=validation.inspection_evidence_sha256
	   ))
FROM control_calphad_validation_events validation
WHERE validation.revision_id=$1
ORDER BY validation.created_at DESC, validation.validation_id DESC`, revision.RevisionID,
		domain.CalphadEvidenceContractVersion)
	if err != nil {
		return domain.CalphadLedgerRecord{}, err
	}
	defer rows.Close()
	validations := []domain.CalphadValidationRecord{}
	for rows.Next() {
		record, scanErr := scanCalphadValidation(rows)
		if scanErr != nil {
			return domain.CalphadLedgerRecord{}, scanErr
		}
		if !calphadValidationPressureMatchesRevision(record, revision) {
			return domain.CalphadLedgerRecord{}, ErrCalphadPressureLimitsInvalid
		}
		validations = append(validations, record)
	}
	if err := rows.Err(); err != nil {
		return domain.CalphadLedgerRecord{}, err
	}
	return ledgerFromRecords(revision, validations), nil
}

func (s *PostgresStore) GetCalphadLedgerPageForOwner(
	ctx context.Context,
	input domain.GetCalphadLedgerPageInput,
) (domain.CalphadLedgerRecord, error) {
	if err := validateCalphadLedgerPageInput(input); err != nil {
		return domain.CalphadLedgerRecord{}, err
	}
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{
		IsoLevel: pgx.RepeatableRead, AccessMode: pgx.ReadOnly,
	})
	if err != nil {
		return domain.CalphadLedgerRecord{}, err
	}
	defer tx.Rollback(ctx)

	revision, err := scanCalphadRevision(tx.QueryRow(ctx, `
SELECT cr.revision_id, cr.resource_id, cr.owner_user_id, cr.owner_org_id, cr.sha256, cr.size_bytes, cr.database_format,
       cr.assessment_pressure_min_pa, cr.assessment_pressure_max_pa,
       cr.parent_revision_id, cr.created_by_user_id, cr.created_at, cr.metadata
FROM control_calphad_revisions cr
WHERE cr.resource_id=$1 AND cr.owner_user_id=$2
  AND (COALESCE(cr.owner_org_id,'')='' OR cr.owner_org_id=$3)`,
		strings.TrimSpace(input.ResourceID), strings.TrimSpace(input.OwnerUserID),
		strings.TrimSpace(input.OwnerOrgID)))
	if err != nil {
		return domain.CalphadLedgerRecord{}, mapPgError(err)
	}
	if expected := strings.TrimSpace(input.ExpectedRevisionID); expected != "" && revision.RevisionID != expected {
		return domain.CalphadLedgerRecord{}, ErrNotFound
	}
	if _, err := loadCalphadInputBlob(ctx, tx, revision); err != nil {
		return domain.CalphadLedgerRecord{}, err
	}
	if !input.BeforeCreatedAt.IsZero() {
		var anchorExists bool
		if err := tx.QueryRow(ctx, `
SELECT EXISTS (
  SELECT 1 FROM control_calphad_validation_events
  WHERE revision_id=$1 AND validation_id=$2 AND created_at=$3
)`, revision.RevisionID, strings.TrimSpace(input.BeforeValidationID), input.BeforeCreatedAt.UTC()).Scan(&anchorExists); err != nil {
			return domain.CalphadLedgerRecord{}, mapPgError(err)
		}
		if !anchorExists {
			return domain.CalphadLedgerRecord{}, ErrNotFound
		}
	}

	latest, latestErr := scanCalphadValidation(tx.QueryRow(ctx, `
SELECT validation.validation_id, validation.revision_id, validation.resource_id,
       validation.database_sha256, validation.database_size_bytes, validation.database_format,
	   validation.assessment_pressure_min_pa, validation.assessment_pressure_max_pa,
	   validation.database_inventory_sha256, validation.request_sha256,
       validation.status, validation.operation, validation.failure_domain,
       validation.failure_stage, validation.failure_code, validation.evidence_path,
       validation.evidence_sha256, validation.evidence_size_bytes, validation.runtime_image_id,
       validation.pycalphad_version, validation.run_id, validation.inspection_evidence_sha256,
	   validation.evidence_contract_version,
       validation.created_by_authority, validation.created_at, validation.metadata,
       EXISTS (SELECT 1 FROM control_calphad_evidence_blobs blob
               WHERE blob.evidence_sha256=validation.evidence_sha256
                 AND blob.evidence_size_bytes=validation.evidence_size_bytes
                 AND octet_length(blob.payload)=validation.evidence_size_bytes
				 AND encode(sha256(blob.payload), 'hex')=validation.evidence_sha256
				 AND validation.evidence_contract_version=$2)
	   AND (validation.operation <> 'equilibrium' OR EXISTS (
		 SELECT 1
		 FROM control_calphad_validation_events inspection
		 JOIN control_calphad_evidence_blobs inspection_blob
		   ON inspection_blob.evidence_sha256=inspection.evidence_sha256
		  AND inspection_blob.evidence_size_bytes=inspection.evidence_size_bytes
		  AND octet_length(inspection_blob.payload)=inspection.evidence_size_bytes
		  AND encode(sha256(inspection_blob.payload), 'hex')=inspection.evidence_sha256
		 WHERE inspection.revision_id=validation.revision_id
		   AND inspection.resource_id=validation.resource_id
		   AND inspection.database_sha256=validation.database_sha256
		   AND inspection.database_size_bytes=validation.database_size_bytes
		   AND inspection.database_format=validation.database_format
		   AND inspection.assessment_pressure_min_pa=validation.assessment_pressure_min_pa
		   AND inspection.assessment_pressure_max_pa=validation.assessment_pressure_max_pa
		   AND inspection.run_id=validation.run_id
		   AND inspection.operation='inspect' AND inspection.status='input_validated'
		   AND inspection.runtime_image_id=validation.runtime_image_id
		   AND inspection.database_inventory_sha256=validation.database_inventory_sha256
		   AND inspection.evidence_contract_version=$2
		   AND inspection.evidence_sha256=validation.inspection_evidence_sha256
	   ))
FROM control_calphad_validation_events validation
WHERE validation.revision_id=$1
ORDER BY validation.created_at DESC, validation.validation_id DESC
LIMIT 1`, revision.RevisionID, domain.CalphadEvidenceContractVersion))
	if latestErr != nil && !errors.Is(latestErr, pgx.ErrNoRows) {
		return domain.CalphadLedgerRecord{}, mapPgError(latestErr)
	}
	if latestErr == nil && !calphadValidationPressureMatchesRevision(latest, revision) {
		return domain.CalphadLedgerRecord{}, ErrCalphadPressureLimitsInvalid
	}

	rows, err := tx.Query(ctx, `
SELECT validation.validation_id, validation.revision_id, validation.resource_id,
       validation.database_sha256, validation.database_size_bytes, validation.database_format,
	   validation.assessment_pressure_min_pa, validation.assessment_pressure_max_pa,
	   validation.database_inventory_sha256, validation.request_sha256,
       validation.status, validation.operation, validation.failure_domain,
       validation.failure_stage, validation.failure_code, validation.evidence_path,
       validation.evidence_sha256, validation.evidence_size_bytes, validation.runtime_image_id,
       validation.pycalphad_version, validation.run_id, validation.inspection_evidence_sha256,
	   validation.evidence_contract_version,
       validation.created_by_authority, validation.created_at, validation.metadata,
       EXISTS (SELECT 1 FROM control_calphad_evidence_blobs blob
               WHERE blob.evidence_sha256=validation.evidence_sha256
                 AND blob.evidence_size_bytes=validation.evidence_size_bytes
                 AND octet_length(blob.payload)=validation.evidence_size_bytes
				 AND encode(sha256(blob.payload), 'hex')=validation.evidence_sha256
				 AND validation.evidence_contract_version=$2)
	   AND (validation.operation <> 'equilibrium' OR EXISTS (
		 SELECT 1
		 FROM control_calphad_validation_events inspection
		 JOIN control_calphad_evidence_blobs inspection_blob
		   ON inspection_blob.evidence_sha256=inspection.evidence_sha256
		  AND inspection_blob.evidence_size_bytes=inspection.evidence_size_bytes
		  AND octet_length(inspection_blob.payload)=inspection.evidence_size_bytes
		  AND encode(sha256(inspection_blob.payload), 'hex')=inspection.evidence_sha256
		 WHERE inspection.revision_id=validation.revision_id
		   AND inspection.resource_id=validation.resource_id
		   AND inspection.database_sha256=validation.database_sha256
		   AND inspection.database_size_bytes=validation.database_size_bytes
		   AND inspection.database_format=validation.database_format
		   AND inspection.assessment_pressure_min_pa=validation.assessment_pressure_min_pa
		   AND inspection.assessment_pressure_max_pa=validation.assessment_pressure_max_pa
		   AND inspection.run_id=validation.run_id
		   AND inspection.operation='inspect' AND inspection.status='input_validated'
		   AND inspection.runtime_image_id=validation.runtime_image_id
		   AND inspection.database_inventory_sha256=validation.database_inventory_sha256
		   AND inspection.evidence_contract_version=$2
		   AND inspection.evidence_sha256=validation.inspection_evidence_sha256
	   ))
FROM control_calphad_validation_events validation
WHERE validation.revision_id=$1
  AND ($3::boolean = false OR (validation.created_at, validation.validation_id) < ($4::timestamptz, $5::text))
ORDER BY validation.created_at DESC, validation.validation_id DESC
LIMIT $6`, revision.RevisionID, domain.CalphadEvidenceContractVersion,
		!input.BeforeCreatedAt.IsZero(), input.BeforeCreatedAt.UTC(),
		strings.TrimSpace(input.BeforeValidationID), input.Limit+1)
	if err != nil {
		return domain.CalphadLedgerRecord{}, mapPgError(err)
	}
	validations := make([]domain.CalphadValidationRecord, 0, input.Limit+1)
	for rows.Next() {
		record, scanErr := scanCalphadValidation(rows)
		if scanErr != nil {
			rows.Close()
			return domain.CalphadLedgerRecord{}, scanErr
		}
		if !calphadValidationPressureMatchesRevision(record, revision) {
			rows.Close()
			return domain.CalphadLedgerRecord{}, ErrCalphadPressureLimitsInvalid
		}
		validations = append(validations, record)
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return domain.CalphadLedgerRecord{}, mapPgError(err)
	}
	rows.Close()
	hasMore := len(validations) > input.Limit
	if hasMore {
		validations = validations[:input.Limit]
	}
	ledger := ledgerFromRecords(revision, validations)
	ledger.HasMore = hasMore
	if latestErr == nil {
		latestCopy := cloneCalphadValidation(latest)
		ledger.LatestValidation = &latestCopy
	}
	if hasMore && len(validations) > 0 {
		ledger.NextCreatedAt = validations[len(validations)-1].CreatedAt
		ledger.NextValidationID = validations[len(validations)-1].ValidationID
	}
	if err := tx.Commit(ctx); err != nil {
		return domain.CalphadLedgerRecord{}, mapPgError(err)
	}
	return ledger, nil
}

func (s *PostgresStore) GetRetainedCalphadInspectionForOwner(
	ctx context.Context,
	input domain.GetRetainedCalphadInspectionInput,
) (domain.CalphadValidationRecord, error) {
	record, err := scanCalphadValidation(s.pool.QueryRow(ctx, `
SELECT validation.validation_id, validation.revision_id, validation.resource_id,
       validation.database_sha256, validation.database_size_bytes, validation.database_format,
       validation.assessment_pressure_min_pa, validation.assessment_pressure_max_pa,
       validation.database_inventory_sha256, validation.request_sha256,
       validation.status, validation.operation, validation.failure_domain,
       validation.failure_stage, validation.failure_code, validation.evidence_path,
       validation.evidence_sha256, validation.evidence_size_bytes, validation.runtime_image_id,
       validation.pycalphad_version, validation.run_id, validation.inspection_evidence_sha256,
       validation.evidence_contract_version,
       validation.created_by_authority, validation.created_at, validation.metadata, TRUE
FROM control_calphad_validation_events validation
JOIN control_calphad_revisions revision
  ON revision.revision_id=validation.revision_id
 AND revision.resource_id=validation.resource_id
JOIN control_calphad_evidence_blobs blob
  ON blob.evidence_sha256=validation.evidence_sha256
 AND blob.evidence_size_bytes=validation.evidence_size_bytes
 AND octet_length(blob.payload)=validation.evidence_size_bytes
 AND encode(sha256(blob.payload), 'hex')=validation.evidence_sha256
WHERE revision.resource_id=$1 AND revision.owner_user_id=$2
  AND (COALESCE(revision.owner_org_id,'')='' OR revision.owner_org_id=$3)
  AND validation.operation='inspect' AND validation.status='input_validated'
  AND validation.run_id=$4 AND validation.runtime_image_id=$5
  AND validation.evidence_sha256=$6 AND validation.database_sha256=$7
  AND validation.database_size_bytes=$8 AND validation.database_format=$9
  AND validation.evidence_contract_version=$10
LIMIT 1`, strings.TrimSpace(input.ResourceID), strings.TrimSpace(input.OwnerUserID),
		strings.TrimSpace(input.OwnerOrgID), strings.TrimSpace(input.RunID),
		strings.ToLower(strings.TrimSpace(input.RuntimeImageID)),
		strings.ToLower(strings.TrimSpace(input.EvidenceSHA256)),
		strings.ToLower(strings.TrimSpace(input.DatabaseSHA256)), input.DatabaseSizeBytes,
		strings.TrimSpace(input.DatabaseFormat), domain.CalphadEvidenceContractVersion))
	if err != nil {
		return domain.CalphadValidationRecord{}, mapPgError(err)
	}
	if record.EvidenceRetention != domain.CalphadEvidenceRetentionRetained || !record.Promotable {
		return domain.CalphadValidationRecord{}, ErrCalphadEvidenceRetentionRequired
	}
	return record, nil
}

func (s *PostgresStore) GetCalphadValidationEvidenceForOwner(
	ctx context.Context,
	resourceID, validationID, userID, orgID string,
) (domain.CalphadValidationEvidenceRecord, error) {
	revision, err := scanCalphadRevision(s.pool.QueryRow(ctx, `
SELECT revision_id, resource_id, owner_user_id, owner_org_id, sha256, size_bytes, database_format,
       assessment_pressure_min_pa, assessment_pressure_max_pa,
       parent_revision_id, created_by_user_id, created_at, metadata
FROM control_calphad_revisions
WHERE resource_id=$1 AND owner_user_id=$2
  AND (COALESCE(owner_org_id,'')='' OR owner_org_id=$3)`,
		strings.TrimSpace(resourceID), strings.TrimSpace(userID), strings.TrimSpace(orgID)))
	if err != nil {
		return domain.CalphadValidationEvidenceRecord{}, mapPgError(err)
	}
	var record domain.CalphadValidationEvidenceRecord
	var evidenceContractVersion, databaseFormat, blobSHA string
	var blobSize int64
	var payload []byte
	var inspectionLineageRetained bool
	err = s.pool.QueryRow(ctx, `
SELECT validation.validation_id, validation.revision_id, validation.resource_id,
       validation.database_format,
       COALESCE(validation.evidence_sha256,''), COALESCE(validation.evidence_size_bytes,0),
       COALESCE(validation.evidence_contract_version,''),
       COALESCE(blob.evidence_sha256,''), COALESCE(blob.evidence_size_bytes,0),
       COALESCE(blob.payload,''::bytea),
       validation.operation <> 'equilibrium' OR EXISTS (
         SELECT 1
         FROM control_calphad_validation_events inspection
         JOIN control_calphad_evidence_blobs inspection_blob
           ON inspection_blob.evidence_sha256=inspection.evidence_sha256
          AND inspection_blob.evidence_size_bytes=inspection.evidence_size_bytes
          AND octet_length(inspection_blob.payload)=inspection.evidence_size_bytes
          AND encode(sha256(inspection_blob.payload), 'hex')=inspection.evidence_sha256
         WHERE inspection.revision_id=validation.revision_id
           AND inspection.resource_id=validation.resource_id
           AND inspection.database_sha256=validation.database_sha256
           AND inspection.database_size_bytes=validation.database_size_bytes
           AND inspection.database_format=validation.database_format
           AND inspection.assessment_pressure_min_pa=validation.assessment_pressure_min_pa
           AND inspection.assessment_pressure_max_pa=validation.assessment_pressure_max_pa
           AND inspection.run_id=validation.run_id
           AND inspection.operation='inspect' AND inspection.status='input_validated'
           AND inspection.runtime_image_id=validation.runtime_image_id
           AND inspection.database_inventory_sha256=validation.database_inventory_sha256
           AND inspection.evidence_contract_version=$4
           AND inspection.evidence_sha256=validation.inspection_evidence_sha256
       )
FROM control_calphad_validation_events validation
JOIN control_calphad_revisions revision
  ON revision.revision_id=validation.revision_id
 AND revision.resource_id=validation.resource_id
 AND revision.sha256=validation.database_sha256
 AND revision.size_bytes=validation.database_size_bytes
 AND revision.database_format=validation.database_format
LEFT JOIN control_calphad_evidence_blobs blob
  ON blob.evidence_sha256=validation.evidence_sha256
 AND blob.evidence_size_bytes=validation.evidence_size_bytes
WHERE validation.resource_id=$1 AND validation.validation_id=$2
  AND revision.revision_id=$3 AND revision.resource_id=$1`,
		strings.TrimSpace(resourceID), strings.TrimSpace(validationID), revision.RevisionID,
		domain.CalphadEvidenceContractVersion).Scan(
		&record.ValidationID, &record.RevisionID, &record.ResourceID,
		&databaseFormat, &record.SHA256, &record.SizeBytes, &evidenceContractVersion,
		&blobSHA, &blobSize, &payload, &inspectionLineageRetained,
	)
	if err != nil {
		return domain.CalphadValidationEvidenceRecord{}, mapPgError(err)
	}
	if databaseFormat != revision.DatabaseFormat ||
		evidenceContractVersion != domain.CalphadEvidenceContractVersion ||
		!inspectionLineageRetained {
		return domain.CalphadValidationEvidenceRecord{}, ErrCalphadEvidenceRetentionRequired
	}
	validation := domain.CalphadValidationRecord{
		DatabaseFormat: databaseFormat,
		EvidenceSHA256: record.SHA256, EvidenceSizeBytes: record.SizeBytes,
		EvidenceContractVersion: evidenceContractVersion,
	}
	blob := calphadEvidenceBlob{SHA256: blobSHA, SizeBytes: blobSize, Payload: payload}
	if !retainedCalphadBlobMatches(validation, blob, blobSHA != "") {
		return domain.CalphadValidationEvidenceRecord{}, ErrCalphadEvidenceRetentionRequired
	}
	record.Bytes = append([]byte(nil), payload...)
	return record, nil
}

func (s *PostgresStore) GetCalphadRevisionInputForOwner(
	ctx context.Context,
	resourceID, userID, orgID string,
) (domain.CalphadRevisionInputRecord, error) {
	revision, err := scanCalphadRevision(s.pool.QueryRow(ctx, `
SELECT revision_id, resource_id, owner_user_id, owner_org_id, sha256, size_bytes, database_format,
       assessment_pressure_min_pa, assessment_pressure_max_pa,
       parent_revision_id, created_by_user_id, created_at, metadata
FROM control_calphad_revisions
WHERE resource_id=$1 AND owner_user_id=$2
  AND (COALESCE(owner_org_id,'')='' OR owner_org_id=$3)`,
		strings.TrimSpace(resourceID), strings.TrimSpace(userID), strings.TrimSpace(orgID)))
	if err != nil {
		return domain.CalphadRevisionInputRecord{}, mapPgError(err)
	}
	blob, err := loadCalphadInputBlob(ctx, s.pool, revision)
	if err != nil {
		return domain.CalphadRevisionInputRecord{}, err
	}
	return domain.CalphadRevisionInputRecord{
		RevisionID:     revision.RevisionID,
		ResourceID:     revision.ResourceID,
		SHA256:         revision.SHA256,
		SizeBytes:      revision.SizeBytes,
		DatabaseFormat: revision.DatabaseFormat,
		Bytes:          blob.Payload,
	}, nil
}

// Ensure compile-time interface parity for both stores.
type calphadLedgerStore interface {
	CreateCalphadRevision(context.Context, domain.CreateCalphadRevisionInput) (domain.CalphadRevisionRecord, error)
	AppendCalphadValidation(context.Context, domain.AppendCalphadValidationInput) (domain.CalphadValidationRecord, error)
	GetCalphadLedgerForOwner(context.Context, string, string, string) (domain.CalphadLedgerRecord, error)
	GetCalphadLedgerPageForOwner(context.Context, domain.GetCalphadLedgerPageInput) (domain.CalphadLedgerRecord, error)
	GetRetainedCalphadInspectionForOwner(context.Context, domain.GetRetainedCalphadInspectionInput) (domain.CalphadValidationRecord, error)
	GetCalphadValidationEvidenceForOwner(context.Context, string, string, string, string) (domain.CalphadValidationEvidenceRecord, error)
	GetCalphadRevisionInputForOwner(context.Context, string, string, string) (domain.CalphadRevisionInputRecord, error)
}

var _ calphadLedgerStore = (*MemoryStore)(nil)
var _ calphadLedgerStore = (*PostgresStore)(nil)
