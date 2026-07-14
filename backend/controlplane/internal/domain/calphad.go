package domain

import "time"

const (
	CalphadRuntimePolicyMetadataKey            = "calphad_runtime_policy"
	CalphadRuntimePolicySchema                 = "ultra.calphad.runtime-policy.v2"
	CalphadPycalphadVersion                    = "0.11.2"
	CalphadScheilVersion                       = "0.3.0"
	CalphadEvidenceContractVersion             = "ultra.calphad.retained-evidence.v2"
	CalphadFailureEvidenceSchemaVersion        = "ultra.calphad.failure-evidence.v1"
	CalphadOwnerDeclarationMetadataKey         = "owner_declaration"
	CalphadOwnerDeclarationSchema              = "ultra.calphad.owner-declaration.v1"
	CalphadDatabaseFormatTDB                   = "tdb"
	CalphadDatabaseFormatDAT                   = "dat"
	CalphadRuntimeNetwork                      = "none"
	CalphadRuntimeCPUsAtMost                   = int64(8)
	CalphadRuntimeMemoryBytesAtMost            = int64(34359738368)
	CalphadRuntimePIDsAtMost                   = int64(4096)
	CalphadMaxInputBytes                       = int64(64 << 20)
	CalphadAssessmentPressureLimitsMetadataKey = "assessment_pressure_limits_Pa"
	CalphadMinimumPressurePa                   = 1e-9
	CalphadMaximumPressurePa                   = 1e12
	CalphadReferencePressurePa                 = 101325.0
	CalphadLedgerDefaultLimit                  = 100
	CalphadLedgerMaximumLimit                  = 500

	CalphadEvidenceRetentionNotApplicable    = "not_applicable"
	CalphadEvidenceRetentionRetained         = "retained"
	CalphadEvidenceRetentionUnretained       = "unretained"
	CalphadEvidenceRetentionLegacyUnretained = "legacy_unretained"
)

type CalphadFailureDomain string

const (
	CalphadFailureDomainInput      CalphadFailureDomain = "input"
	CalphadFailureDomainScientific CalphadFailureDomain = "scientific"
	CalphadFailureDomainPlatform   CalphadFailureDomain = "platform"
)

func (value CalphadFailureDomain) Valid() bool {
	return value == CalphadFailureDomainInput ||
		value == CalphadFailureDomainScientific ||
		value == CalphadFailureDomainPlatform
}

type CalphadFailureStage string

const (
	CalphadFailureStageParse            CalphadFailureStage = "parse"
	CalphadFailureStageSolver           CalphadFailureStage = "solver"
	CalphadFailureStageResultValidation CalphadFailureStage = "result_validation"
	CalphadFailureStageSandboxRuntime   CalphadFailureStage = "sandbox_runtime"
)

func (value CalphadFailureStage) Valid() bool {
	return value == CalphadFailureStageParse ||
		value == CalphadFailureStageSolver ||
		value == CalphadFailureStageResultValidation ||
		value == CalphadFailureStageSandboxRuntime
}

type CalphadFailureCode string

const (
	CalphadFailureCodeParseFailed            CalphadFailureCode = "calphad_parse_failed"
	CalphadFailureCodeParseTimeout           CalphadFailureCode = "calphad_parse_timeout"
	CalphadFailureCodeParseUnsupported       CalphadFailureCode = "calphad_parse_unsupported"
	CalphadFailureCodeSolverFailed           CalphadFailureCode = "calphad_solver_failed"
	CalphadFailureCodeSolverTimeout          CalphadFailureCode = "calphad_solver_timeout"
	CalphadFailureCodeSolverUnsupported      CalphadFailureCode = "calphad_solver_unsupported"
	CalphadFailureCodeResultInvalid          CalphadFailureCode = "calphad_result_invalid"
	CalphadFailureCodeRuntimeInternalFailure CalphadFailureCode = "calphad_runtime_internal_failure"
	CalphadFailureCodeSandboxFailed          CalphadFailureCode = "calphad_sandbox_failed"
	CalphadFailureCodeSandboxTimeout         CalphadFailureCode = "calphad_sandbox_timeout"
)

func (value CalphadFailureCode) Valid() bool {
	switch value {
	case CalphadFailureCodeParseFailed,
		CalphadFailureCodeParseTimeout,
		CalphadFailureCodeParseUnsupported,
		CalphadFailureCodeSolverFailed,
		CalphadFailureCodeSolverTimeout,
		CalphadFailureCodeSolverUnsupported,
		CalphadFailureCodeResultInvalid,
		CalphadFailureCodeRuntimeInternalFailure,
		CalphadFailureCodeSandboxFailed,
		CalphadFailureCodeSandboxTimeout:
		return true
	default:
		return false
	}
}

// CalphadRevisionRecord binds one catalog resource to one immutable CALPHAD database revision.
// Thermodynamic functions remain in the TDB or DAT; this record stores governance facts only.
type CalphadRevisionRecord struct {
	RevisionID                 string     `json:"revision_id"`
	ResourceID                 string     `json:"resource_id"`
	OwnerUserID                string     `json:"owner_user_id"`
	OwnerOrgID                 string     `json:"owner_org_id,omitempty"`
	SHA256                     string     `json:"sha256"`
	SizeBytes                  int64      `json:"size_bytes"`
	DatabaseFormat             string     `json:"database_format"`
	AssessmentPressureLimitsPa [2]float64 `json:"assessment_pressure_limits_Pa"`
	ParentRevisionID           string     `json:"parent_revision_id,omitempty"`
	CreatedByUserID            string     `json:"created_by_user_id,omitempty"`
	CreatedAt                  time.Time  `json:"created_at"`
	Metadata                   JSONMap    `json:"metadata"`
}

type CreateCalphadRevisionInput struct {
	ResourceID                 string
	OwnerUserID                string
	OwnerOrgID                 string
	ParentRevisionID           string
	ExpectedSHA256             string
	ExpectedSizeBytes          int64
	AssessmentPressureLimitsPa [2]float64
	InputBytes                 []byte
	CreatedByUserID            string
	CreatedAt                  time.Time
	Metadata                   JSONMap
}

// CalphadRevisionInputRecord is the exact immutable CALPHAD database byte stream retained
// for replay. Bytes are never embedded in JSON ledger responses.
type CalphadRevisionInputRecord struct {
	RevisionID     string `json:"revision_id"`
	ResourceID     string `json:"resource_id"`
	SHA256         string `json:"sha256"`
	SizeBytes      int64  `json:"size_bytes"`
	DatabaseFormat string `json:"database_format"`
	Bytes          []byte `json:"-"`
}

// CalphadOwnerDeclaration is the normalized immutable owner provenance
// snapshot stored under revision.metadata.owner_declaration.
type CalphadOwnerDeclaration struct {
	SchemaVersion                string     `json:"schema_version"`
	Authority                    string     `json:"authority"`
	DatabaseID                   string     `json:"database_id"`
	Source                       string     `json:"source"`
	LicenseID                    string     `json:"license_id"`
	AssessmentScope              string     `json:"assessment_scope"`
	ReferenceState               string     `json:"reference_state"`
	AssessmentTemperatureLimitsK [2]float64 `json:"assessment_temperature_limits_K"`
	AssessmentPressureLimitsPa   [2]float64 `json:"assessment_pressure_limits_Pa"`
	DatabaseFormat               string     `json:"database_format"`
}

// CalphadValidationEvidenceRecord is an exact retained validation artifact.
// Bytes are streamed directly and never embedded in JSON ledger responses.
type CalphadValidationEvidenceRecord struct {
	ValidationID string `json:"validation_id"`
	RevisionID   string `json:"revision_id"`
	ResourceID   string `json:"resource_id"`
	SHA256       string `json:"sha256"`
	SizeBytes    int64  `json:"size_bytes"`
	Bytes        []byte `json:"-"`
}

// CalphadValidationRecord is an append-only server-managed validation event.
// Status describes bounded runtime evidence, not independent assessment correctness.
type CalphadValidationRecord struct {
	ValidationID               string               `json:"validation_id"`
	RevisionID                 string               `json:"revision_id"`
	ResourceID                 string               `json:"resource_id"`
	DatabaseSHA256             string               `json:"database_sha256"`
	DatabaseSizeBytes          int64                `json:"database_size_bytes"`
	DatabaseFormat             string               `json:"database_format"`
	AssessmentPressureLimitsPa [2]float64           `json:"assessment_pressure_limits_Pa"`
	DatabaseInventorySHA256    string               `json:"database_inventory_sha256,omitempty"`
	RequestSHA256              string               `json:"request_sha256,omitempty"`
	Status                     string               `json:"status"`
	Operation                  string               `json:"operation"`
	FailureDomain              CalphadFailureDomain `json:"failure_domain,omitempty"`
	FailureStage               CalphadFailureStage  `json:"failure_stage,omitempty"`
	FailureCode                CalphadFailureCode   `json:"failure_code,omitempty"`
	EvidencePath               string               `json:"evidence_path,omitempty"`
	EvidenceSHA256             string               `json:"evidence_sha256,omitempty"`
	EvidenceSizeBytes          int64                `json:"evidence_size_bytes,omitempty"`
	RuntimeImageID             string               `json:"runtime_image_id,omitempty"`
	PycalphadVersion           string               `json:"pycalphad_version,omitempty"`
	RunID                      string               `json:"run_id,omitempty"`
	// InspectionEvidenceSHA256 binds an equilibrium or Scheil result to the exact retained
	// inspect artifact for this revision, run, and immutable runtime image.
	InspectionEvidenceSHA256 string `json:"inspection_evidence_sha256,omitempty"`
	EvidenceContractVersion  string `json:"evidence_contract_version,omitempty"`
	// EvidenceRetention is derived from the retained blob store at read time.
	// Legacy rows created before blob retention are never represented as retained.
	EvidenceRetention  string    `json:"evidence_retention"`
	Promotable         bool      `json:"promotable"`
	CreatedByAuthority string    `json:"created_by_authority"`
	CreatedAt          time.Time `json:"created_at"`
	Metadata           JSONMap   `json:"metadata"`
}

type AppendCalphadValidationInput struct {
	ResourceID                 string
	OwnerUserID                string
	OwnerOrgID                 string
	DatabaseSHA256             string
	DatabaseSizeBytes          int64
	DatabaseFormat             string
	OwnerDeclaration           CalphadOwnerDeclaration
	AssessmentPressureLimitsPa [2]float64
	DatabaseInventorySHA256    string
	RequestSHA256              string
	Status                     string
	Operation                  string
	FailureDomain              CalphadFailureDomain
	FailureStage               CalphadFailureStage
	FailureCode                CalphadFailureCode
	EvidencePath               string
	EvidenceSHA256             string
	EvidenceSizeBytes          int64
	EvidenceBytes              []byte
	RuntimeImageID             string
	PycalphadVersion           string
	RunID                      string
	InspectionEvidenceSHA256   string
	LeaseWorkerID              string
	LeaseToken                 string
	CreatedByAuthority         string
	CreatedAt                  time.Time
	Metadata                   JSONMap
}

type CalphadLedgerRecord struct {
	Revision         CalphadRevisionRecord     `json:"revision"`
	LatestValidation *CalphadValidationRecord  `json:"latest_validation,omitempty"`
	Validations      []CalphadValidationRecord `json:"validations"`
	HasMore          bool                      `json:"has_more"`
	NextCursor       string                    `json:"next_cursor,omitempty"`
	NextCreatedAt    time.Time                 `json:"-"`
	NextValidationID string                    `json:"-"`
}

type GetCalphadLedgerPageInput struct {
	ResourceID         string
	OwnerUserID        string
	OwnerOrgID         string
	Limit              int
	ExpectedRevisionID string
	BeforeCreatedAt    time.Time
	BeforeValidationID string
}

type GetRetainedCalphadInspectionInput struct {
	ResourceID        string
	OwnerUserID       string
	OwnerOrgID        string
	RunID             string
	RuntimeImageID    string
	EvidenceSHA256    string
	DatabaseSHA256    string
	DatabaseSizeBytes int64
	DatabaseFormat    string
}
