package httpapi

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"errors"
	"io"
	"math"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
	"github.com/go-chi/chi/v5"
)

type calphadLedgerStore interface {
	CreateCalphadRevision(context.Context, domain.CreateCalphadRevisionInput) (domain.CalphadRevisionRecord, error)
	AppendCalphadValidation(context.Context, domain.AppendCalphadValidationInput) (domain.CalphadValidationRecord, error)
	GetCalphadLedgerForOwner(context.Context, string, string, string) (domain.CalphadLedgerRecord, error)
	GetCalphadLedgerPageForOwner(context.Context, domain.GetCalphadLedgerPageInput) (domain.CalphadLedgerRecord, error)
	GetRetainedCalphadInspectionForOwner(context.Context, domain.GetRetainedCalphadInspectionInput) (domain.CalphadValidationRecord, error)
	GetCalphadValidationEvidenceForOwner(context.Context, string, string, string, string) (domain.CalphadValidationEvidenceRecord, error)
	GetCalphadRevisionInputForOwner(context.Context, string, string, string) (domain.CalphadRevisionInputRecord, error)
}

type createCalphadRevisionRequest struct {
	ParentRevisionID string `json:"parent_revision_id,omitempty"`
}

type appendCalphadValidationRequest struct {
	Status             string `json:"status"`
	Operation          string `json:"operation"`
	FailureDomain      string `json:"failure_domain,omitempty"`
	FailureStage       string `json:"failure_stage,omitempty"`
	FailureCode        string `json:"failure_code,omitempty"`
	EvidencePath       string `json:"evidence_path"`
	EvidenceSHA256     string `json:"evidence_sha256"`
	EvidenceSizeBytes  int64  `json:"evidence_size_bytes"`
	RuntimeImageID     string `json:"runtime_image_id"`
	PycalphadVersion   string `json:"pycalphad_version"`
	EvidenceGzipBase64 string `json:"evidence_gzip_base64"`
}

func (deps ServerDeps) calphadLedgerStore() (calphadLedgerStore, bool) {
	ledger, ok := deps.Store.(calphadLedgerStore)
	return ledger, ok
}

func (deps ServerDeps) handleCreateCalphadRevision(w http.ResponseWriter, r *http.Request) {
	ledger, ok := deps.calphadLedgerStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "CALPHAD governance ledger is not configured"})
		return
	}
	var req createCalphadRevisionRequest
	if r.Body != nil && !decodeJSON(w, r, &req) {
		return
	}
	principal := deps.principalFromRequest(r, "")
	file, info, resource, ok := deps.openCatalogResourceForRequest(w, r)
	if !ok {
		return
	}
	defer file.Close()
	if resource.OwnerUserID != principal.UserID ||
		(strings.TrimSpace(resource.OwnerOrgID) != "" && resource.OwnerOrgID != principal.OrgID) {
		writeStoreError(w, store.ErrNotFound)
		return
	}
	ownerDeclaration, err := store.CalphadOwnerDeclarationForResource(resource)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	inputBytes, ok := readVerifiedCalphadInput(w, file, info, resource)
	if !ok {
		return
	}
	revision, err := ledger.CreateCalphadRevision(r.Context(), domain.CreateCalphadRevisionInput{
		ResourceID:                 resource.ResourceID,
		OwnerUserID:                principal.UserID,
		OwnerOrgID:                 principal.OrgID,
		ParentRevisionID:           strings.TrimSpace(req.ParentRevisionID),
		ExpectedSHA256:             strings.ToLower(strings.TrimSpace(resource.SHA256)),
		ExpectedSizeBytes:          resource.SizeBytes,
		AssessmentPressureLimitsPa: ownerDeclaration.AssessmentPressureLimitsPa,
		InputBytes:                 inputBytes,
		CreatedByUserID:            principal.UserID,
		CreatedAt:                  domain.Now(),
		Metadata: domain.JSONMap{
			"lineage_authority": "resource_owner",
			"server_managed":    true,
		},
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, map[string]any{"revision": revision})
}

func (deps ServerDeps) readCalphadInputResource(
	w http.ResponseWriter,
	resource domain.ResourceRecord,
) ([]byte, bool) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return nil, false
	}
	path, err := resolveCatalogResourcePath(root, resource)
	if err != nil {
		writeStoreError(w, err)
		return nil, false
	}
	file, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			writeStoreError(w, store.ErrNotFound)
		} else {
			writeError(w, http.StatusInternalServerError, err)
		}
		return nil, false
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return nil, false
	}
	if info.IsDir() {
		writeStoreError(w, store.ErrNotFound)
		return nil, false
	}
	return readVerifiedCalphadInput(w, file, info, resource)
}

func readVerifiedCalphadInput(
	w http.ResponseWriter,
	file *os.File,
	info os.FileInfo,
	resource domain.ResourceRecord,
) ([]byte, bool) {
	if info.Size() <= 0 || info.Size() > domain.CalphadMaxInputBytes {
		writeError(w, http.StatusRequestEntityTooLarge, errors.New("CALPHAD input exceeds the 64 MiB retention limit"))
		return nil, false
	}
	if info.Size() != resource.SizeBytes {
		writeStoreError(w, store.ErrConflict)
		return nil, false
	}
	payload, err := io.ReadAll(io.LimitReader(file, domain.CalphadMaxInputBytes+1))
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return nil, false
	}
	sha := strings.ToLower(strings.TrimSpace(resource.SHA256))
	digest := sha256.Sum256(payload)
	if int64(len(payload)) != resource.SizeBytes || hex.EncodeToString(digest[:]) != sha {
		writeStoreError(w, store.ErrConflict)
		return nil, false
	}
	return payload, true
}

func (deps ServerDeps) handleGetCalphadLedger(w http.ResponseWriter, r *http.Request) {
	ledger, ok := deps.calphadLedgerStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "CALPHAD governance ledger is not configured"})
		return
	}
	principal := deps.principalFromRequest(r, "")
	resourceID := chi.URLParam(r, "file_id")
	limit := domain.CalphadLedgerDefaultLimit
	if values, present := r.URL.Query()["limit"]; present {
		if len(values) != 1 {
			writeError(w, http.StatusBadRequest, errors.New("CALPHAD ledger limit must be supplied exactly once"))
			return
		}
		parsed, parseErr := strconv.Atoi(values[0])
		if parseErr != nil || parsed < 1 || parsed > domain.CalphadLedgerMaximumLimit {
			writeError(w, http.StatusBadRequest, errors.New("CALPHAD ledger limit must be between 1 and 500"))
			return
		}
		limit = parsed
	}
	input := domain.GetCalphadLedgerPageInput{
		ResourceID: resourceID, OwnerUserID: principal.UserID, OwnerOrgID: principal.OrgID,
		Limit: limit,
	}
	if values, present := r.URL.Query()["cursor"]; present {
		if len(values) != 1 {
			writeStoreError(w, store.ErrNotFound)
			return
		}
		cursor, decodeErr := decodeCalphadLedgerCursor(values[0])
		if decodeErr != nil || cursor.ResourceID != resourceID ||
			cursor.OwnerUserID != principal.UserID || cursor.OwnerOrgID != principal.OrgID {
			writeStoreError(w, store.ErrNotFound)
			return
		}
		input.ExpectedRevisionID = cursor.RevisionID
		input.BeforeCreatedAt = cursor.CreatedAt
		input.BeforeValidationID = cursor.ValidationID
	}
	record, err := ledger.GetCalphadLedgerPageForOwner(r.Context(), input)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if record.HasMore {
		record.NextCursor, err = encodeCalphadLedgerCursor(calphadLedgerCursor{
			ResourceID: resourceID, OwnerUserID: principal.UserID, OwnerOrgID: principal.OrgID,
			RevisionID: record.Revision.RevisionID, CreatedAt: record.NextCreatedAt,
			ValidationID: record.NextValidationID,
		})
		if err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"ledger": record})
}

func (deps ServerDeps) handleGetCalphadValidationEvidence(w http.ResponseWriter, r *http.Request) {
	ledger, ok := deps.calphadLedgerStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "CALPHAD governance ledger is not configured"})
		return
	}
	principal := deps.principalFromRequest(r, "")
	evidence, err := ledger.GetCalphadValidationEvidenceForOwner(
		r.Context(), chi.URLParam(r, "file_id"), chi.URLParam(r, "validation_id"),
		principal.UserID, principal.OrgID,
	)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", strconv.FormatInt(evidence.SizeBytes, 10))
	w.Header().Set("ETag", `"sha256:`+evidence.SHA256+`"`)
	w.Header().Set("Cache-Control", "private, immutable")
	w.Header().Set("X-Ultra-Calphad-Validation-Id", evidence.ValidationID)
	w.Header().Set("X-Ultra-Content-Sha256", evidence.SHA256)
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(evidence.Bytes)
}

func (deps ServerDeps) handleGetCalphadRevisionInput(w http.ResponseWriter, r *http.Request) {
	ledger, ok := deps.calphadLedgerStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "CALPHAD governance ledger is not configured"})
		return
	}
	principal := deps.principalFromRequest(r, "")
	input, err := ledger.GetCalphadRevisionInputForOwner(
		r.Context(), chi.URLParam(r, "file_id"), principal.UserID, principal.OrgID,
	)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if input.DatabaseFormat != domain.CalphadDatabaseFormatTDB &&
		input.DatabaseFormat != domain.CalphadDatabaseFormatDAT {
		writeStoreError(w, store.ErrCalphadDatabaseFormatInvalid)
		return
	}
	// The retained database may be Thermo-Calc TDB or ChemSage DAT. The
	// immutable revision intentionally does not trust a caller-provided media
	// type, so use the format-neutral binary type instead of mislabeling DAT.
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Content-Length", strconv.FormatInt(input.SizeBytes, 10))
	w.Header().Set("ETag", `"sha256:`+input.SHA256+`"`)
	w.Header().Set("Cache-Control", "private, immutable")
	w.Header().Set("X-Ultra-Calphad-Revision-Id", input.RevisionID)
	w.Header().Set("X-Ultra-Content-Sha256", input.SHA256)
	w.Header().Set("X-Ultra-Calphad-Database-Format", input.DatabaseFormat)
	w.Header().Set("Content-Disposition", `attachment; filename="`+input.SHA256+`.`+input.DatabaseFormat+`"`)
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(input.Bytes)
}

func bindCalphadEquilibriumInspectionInventory(
	ctx context.Context,
	ledger calphadLedgerStore,
	verified *verifiedCalphadEvidence,
	runID string,
	ownerUserID string,
	ownerOrgID string,
) error {
	if verified.Operation != "equilibrium" {
		return nil
	}
	inspection, err := ledger.GetRetainedCalphadInspectionForOwner(
		ctx,
		domain.GetRetainedCalphadInspectionInput{
			ResourceID: verified.ResourceID, OwnerUserID: ownerUserID, OwnerOrgID: ownerOrgID,
			RunID: runID, RuntimeImageID: verified.RuntimeImageID,
			EvidenceSHA256: verified.InspectionEvidenceSHA256,
			DatabaseSHA256: verified.DatabaseSHA256, DatabaseSizeBytes: verified.DatabaseSizeBytes,
			DatabaseFormat: verified.DatabaseFormat,
		},
	)
	if err != nil {
		return store.ErrCalphadInspectionRequired
	}
	if !calphadEvidenceSHA256Pattern.MatchString(inspection.DatabaseInventorySHA256) ||
		(verified.DatabaseInventorySHA256 != "" &&
			verified.DatabaseInventorySHA256 != inspection.DatabaseInventorySHA256) {
		return store.ErrCalphadInspectionRequired
	}
	verified.DatabaseInventorySHA256 = inspection.DatabaseInventorySHA256
	return nil
}

func (deps ServerDeps) handleAppendCalphadValidation(w http.ResponseWriter, r *http.Request) {
	if deps.workerRequestAuth(r) != workerAuthValid {
		writeError(w, http.StatusUnauthorized, errors.New("trusted worker authentication required"))
		return
	}
	ledger, ok := deps.calphadLedgerStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "CALPHAD governance ledger is not configured"})
		return
	}
	runID := chi.URLParam(r, "run_id")
	if r.Header.Get("X-Ultra-Run-Id") != runID {
		writeError(w, http.StatusUnauthorized, errors.New("trusted worker run identity mismatch"))
		return
	}
	run, resolved := deps.runForWorkerOrUser(w, r, runID)
	if !resolved {
		return
	}
	if run.Status != domain.RunStatusRunning {
		writeError(w, http.StatusUnauthorized, errors.New("active run lease required"))
		return
	}
	leases, ok := deps.Store.(runLeaseReader)
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "run lease verification is not configured"})
		return
	}
	lease, found, err := leases.GetRunLease(r.Context(), runID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	requestWorkerID := r.Header.Get("X-Ultra-Worker-Id")
	requestLeaseToken := r.Header.Get("X-Ultra-Run-Lease-Token")
	if !found || !lease.LeaseExpiresAt.After(domain.Now()) || requestWorkerID == "" ||
		requestWorkerID != lease.WorkerID || requestLeaseToken == "" ||
		subtle.ConstantTimeCompare([]byte(requestLeaseToken), []byte(lease.LeaseToken)) != 1 {
		writeError(w, http.StatusUnauthorized, errors.New("active run lease required"))
		return
	}
	var req appendCalphadValidationRequest
	if !decodeCalphadValidationRequest(w, r, &req) {
		return
	}
	resourceID := strings.TrimSpace(chi.URLParam(r, "file_id"))
	verifiedEvidence, err := verifyCalphadEvidence(req, resourceID)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	catalog, ok := deps.resourceCatalogStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource catalog is not configured"})
		return
	}
	resource, err := catalog.GetResourceForUser(
		r.Context(), resourceID, run.UserID, trustedRunOrgID(run),
	)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	// Catalog validation is an owner promotion. A read/share grant permits the
	// grantee to use selected bytes but never to mutate the owner's audit ledger.
	if resource.OwnerUserID != run.UserID ||
		(strings.TrimSpace(resource.OwnerOrgID) != "" && resource.OwnerOrgID != trustedRunOrgID(run)) {
		writeStoreError(w, store.ErrNotFound)
		return
	}
	if !runHasSelectedCalphadBinding(run, verifiedEvidence) {
		writeStoreError(w, store.ErrNotFound)
		return
	}
	currentOwnerDeclaration, err := store.CalphadOwnerDeclarationForResource(resource)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	verifiedOwnerDeclaration := domain.CalphadOwnerDeclaration{
		SchemaVersion: domain.CalphadOwnerDeclarationSchema, Authority: "resource_owner",
		DatabaseID: verifiedEvidence.DatabaseID, Source: verifiedEvidence.Source,
		LicenseID: verifiedEvidence.LicenseID, AssessmentScope: verifiedEvidence.AssessmentScope,
		ReferenceState:               verifiedEvidence.ReferenceState,
		AssessmentTemperatureLimitsK: verifiedEvidence.TemperatureLimitsK,
		AssessmentPressureLimitsPa:   verifiedEvidence.AssessmentPressureLimitsPa,
		DatabaseFormat:               verifiedEvidence.DatabaseFormat,
	}
	if currentOwnerDeclaration != verifiedOwnerDeclaration {
		writeStoreError(w, store.ErrCalphadOwnerDeclarationInvalid)
		return
	}
	if strings.ToLower(strings.TrimSpace(resource.SHA256)) != verifiedEvidence.DatabaseSHA256 ||
		resource.SizeBytes != verifiedEvidence.DatabaseSizeBytes {
		writeStoreError(w, store.ErrConflict)
		return
	}
	if err := bindCalphadEquilibriumInspectionInventory(
		r.Context(), ledger, &verifiedEvidence, runID, resource.OwnerUserID, resource.OwnerOrgID,
	); err != nil {
		writeStoreError(w, err)
		return
	}
	inputBytes, inputOK := deps.readCalphadInputResource(w, resource)
	if !inputOK {
		return
	}
	// Evidence decoding and catalog resolution can take time. Re-resolve the run
	// and lease immediately before the atomic ledger mutation so an expired,
	// released, or requeued lease cannot authorize a late callback.
	currentRun, err := deps.Store.GetRun(r.Context(), runID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	currentLease, currentLeaseFound, err := leases.GetRunLease(r.Context(), runID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if currentRun.Status != domain.RunStatusRunning || !currentLeaseFound ||
		!currentLease.LeaseExpiresAt.After(domain.Now()) || currentLease.WorkerID != requestWorkerID ||
		subtle.ConstantTimeCompare([]byte(currentLease.LeaseToken), []byte(requestLeaseToken)) != 1 {
		writeError(w, http.StatusUnauthorized, errors.New("active run lease required"))
		return
	}
	if !runHasSelectedCalphadBinding(currentRun, verifiedEvidence) {
		writeStoreError(w, store.ErrNotFound)
		return
	}
	revision, err := ledger.CreateCalphadRevision(r.Context(), domain.CreateCalphadRevisionInput{
		ResourceID: resourceID, OwnerUserID: resource.OwnerUserID, OwnerOrgID: resource.OwnerOrgID,
		ExpectedSHA256: verifiedEvidence.DatabaseSHA256, ExpectedSizeBytes: verifiedEvidence.DatabaseSizeBytes,
		AssessmentPressureLimitsPa: verifiedEvidence.AssessmentPressureLimitsPa,
		InputBytes:                 inputBytes,
		CreatedByUserID:            run.UserID,
		CreatedAt:                  domain.Now(), Metadata: domain.JSONMap{
			"lineage_authority": "none_declared",
			"server_managed":    true,
			"selected_run_id":   run.RunID,
		},
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	validation, err := ledger.AppendCalphadValidation(r.Context(), domain.AppendCalphadValidationInput{
		ResourceID: resourceID, OwnerUserID: resource.OwnerUserID, OwnerOrgID: resource.OwnerOrgID,
		DatabaseSHA256: verifiedEvidence.DatabaseSHA256, DatabaseSizeBytes: verifiedEvidence.DatabaseSizeBytes,
		DatabaseFormat: verifiedEvidence.DatabaseFormat, OwnerDeclaration: verifiedOwnerDeclaration,
		AssessmentPressureLimitsPa: verifiedEvidence.AssessmentPressureLimitsPa,
		DatabaseInventorySHA256:    verifiedEvidence.DatabaseInventorySHA256,
		RequestSHA256:              verifiedEvidence.RequestSHA256,
		Status:                     verifiedEvidence.Status, Operation: verifiedEvidence.Operation,
		FailureDomain: verifiedEvidence.FailureDomain, FailureStage: verifiedEvidence.FailureStage,
		FailureCode:  verifiedEvidence.FailureCode,
		EvidencePath: verifiedEvidence.EvidencePath, EvidenceSHA256: verifiedEvidence.EvidenceSHA256,
		EvidenceSizeBytes: verifiedEvidence.EvidenceSizeBytes, RuntimeImageID: verifiedEvidence.RuntimeImageID,
		EvidenceBytes:    verifiedEvidence.EvidenceBytes,
		PycalphadVersion: verifiedEvidence.PycalphadVersion, RunID: run.RunID,
		InspectionEvidenceSHA256: verifiedEvidence.InspectionEvidenceSHA256,
		LeaseWorkerID:            requestWorkerID, LeaseToken: requestLeaseToken,
		CreatedByAuthority: "trusted_worker", CreatedAt: domain.Now(),
		Metadata: domain.JSONMap{
			"server_managed": true,
			"revision_id":    revision.RevisionID,
		},
	})
	if err != nil {
		if errors.Is(err, store.ErrCalphadRunLeaseInvalid) {
			writeError(w, http.StatusUnauthorized, errors.New("active run lease required"))
			return
		}
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, map[string]any{
		"revision": revision, "validation": validation,
	})
}

func calphadDescriptorInt64(value any) (int64, bool) {
	switch typed := value.(type) {
	case int:
		return int64(typed), true
	case int64:
		return typed, true
	case float64:
		if !math.IsNaN(typed) && !math.IsInf(typed, 0) && math.Trunc(typed) == typed &&
			typed >= math.MinInt64 && typed <= math.MaxInt64 {
			return int64(typed), true
		}
	}
	return 0, false
}

func calphadDescriptorFloat64(value any) (float64, bool) {
	var converted float64
	switch typed := value.(type) {
	case float64:
		converted = typed
	case float32:
		converted = float64(typed)
	case int:
		converted = float64(typed)
	case int8:
		converted = float64(typed)
	case int16:
		converted = float64(typed)
	case int32:
		converted = float64(typed)
	case int64:
		converted = float64(typed)
	case uint:
		converted = float64(typed)
	case uint8:
		converted = float64(typed)
	case uint16:
		converted = float64(typed)
	case uint32:
		converted = float64(typed)
	case uint64:
		converted = float64(typed)
	default:
		return 0, false
	}
	return converted, !math.IsNaN(converted) && !math.IsInf(converted, 0)
}

func selectedCalphadOwnerDeclaration(
	descriptor domain.JSONMap,
	resourceID string,
) (domain.CalphadOwnerDeclaration, bool) {
	metadata, ok := jsonMapValue(descriptor["metadata"])
	if !ok {
		return domain.CalphadOwnerDeclaration{}, false
	}
	calphad, ok := jsonMapValue(metadata["calphad"])
	if !ok || calphad["declaration_authority"] != "resource_owner" {
		return domain.CalphadOwnerDeclaration{}, false
	}
	databaseID, databaseIDOK := domain.SafeCalphadOwnerDeclaration(calphad["database_id"], 512)
	if !databaseIDOK {
		databaseID, databaseIDOK = domain.SafeCalphadOwnerDeclaration(resourceID, 512)
	}
	licenseID, licenseOK := domain.SafeCalphadLicenseIdentifier(calphad["license_id"])
	if !licenseOK {
		licenseID, licenseOK = domain.SafeCalphadLicenseIdentifier(calphad["license_identifier"])
	}
	source, sourceOK := domain.SafeCalphadOwnerDeclaration(calphad["source"], 1024)
	assessmentScope, scopeOK := domain.SafeCalphadOwnerDeclaration(calphad["assessment_scope"], 1024)
	referenceState, referenceOK := domain.SafeCalphadOwnerDeclaration(calphad["reference_state"], 512)
	rawTemperature := calphad["assessment_temperature_limits_K"]
	if rawTemperature == nil {
		rawTemperature = calphad["tdb_temperature_limits_K"]
	}
	rawLimits, limitsOK := metadataList(rawTemperature)
	if databaseID == "" || source == "" || licenseID == "" || assessmentScope == "" ||
		referenceState == "" || !databaseIDOK || !sourceOK || !licenseOK || !scopeOK ||
		!referenceOK || !limitsOK || len(rawLimits) != 2 {
		return domain.CalphadOwnerDeclaration{}, false
	}
	minimum, minimumOK := calphadDescriptorFloat64(rawLimits[0])
	maximum, maximumOK := calphadDescriptorFloat64(rawLimits[1])
	if !minimumOK || !maximumOK || minimum <= 0 || minimum >= maximum || maximum > 10_000 {
		return domain.CalphadOwnerDeclaration{}, false
	}
	if legacyRaw, legacyPresent := calphad["tdb_temperature_limits_K"]; legacyPresent &&
		calphad["assessment_temperature_limits_K"] != nil {
		legacyLimits, legacyOK := metadataList(legacyRaw)
		if !legacyOK || len(legacyLimits) != 2 {
			return domain.CalphadOwnerDeclaration{}, false
		}
		legacyMinimum, legacyMinimumOK := calphadDescriptorFloat64(legacyLimits[0])
		legacyMaximum, legacyMaximumOK := calphadDescriptorFloat64(legacyLimits[1])
		if !legacyMinimumOK || !legacyMaximumOK || legacyMinimum != minimum || legacyMaximum != maximum {
			return domain.CalphadOwnerDeclaration{}, false
		}
	}
	pressureLimits, pressureLimitsOK := normalizedCalphadAssessmentPressureLimits(
		calphad[domain.CalphadAssessmentPressureLimitsMetadataKey],
	)
	if !pressureLimitsOK {
		return domain.CalphadOwnerDeclaration{}, false
	}
	databaseFormat, formatOK := descriptor["database_format"].(string)
	originalName, originalNameOK := descriptor["original_name"].(string)
	originalFormat, originalFormatOK := domain.CalphadDatabaseFormatFromName(originalName)
	if !formatOK || !originalNameOK || !originalFormatOK || databaseFormat != originalFormat {
		return domain.CalphadOwnerDeclaration{}, false
	}
	declaration := domain.CalphadOwnerDeclaration{
		SchemaVersion: domain.CalphadOwnerDeclarationSchema, Authority: "resource_owner",
		DatabaseID: databaseID, Source: source, LicenseID: licenseID,
		AssessmentScope: assessmentScope, ReferenceState: referenceState,
		AssessmentTemperatureLimitsK: [2]float64{minimum, maximum},
		AssessmentPressureLimitsPa:   pressureLimits, DatabaseFormat: databaseFormat,
	}
	return declaration, true
}

func calphadOwnerAssessmentPressureLimits(metadata domain.JSONMap) ([2]float64, bool) {
	calphad, ok := jsonMapValue(metadata["calphad"])
	if !ok {
		return [2]float64{}, false
	}
	return normalizedCalphadAssessmentPressureLimits(
		calphad[domain.CalphadAssessmentPressureLimitsMetadataKey],
	)
}

func runCalphadResourceDescriptors(run domain.RunRecord) []domain.JSONMap {
	switch typed := run.Metadata["resource_descriptors"].(type) {
	case []domain.JSONMap:
		return typed
	case []map[string]any:
		out := make([]domain.JSONMap, 0, len(typed))
		for _, descriptor := range typed {
			out = append(out, domain.JSONMap(descriptor))
		}
		return out
	case []any:
		out := make([]domain.JSONMap, 0, len(typed))
		for _, item := range typed {
			if descriptor, ok := jsonMapValue(item); ok {
				out = append(out, descriptor)
			}
		}
		return out
	default:
		return nil
	}
}

func runHasSelectedCalphadBinding(run domain.RunRecord, evidence verifiedCalphadEvidence) bool {
	return store.CalphadRunHasSelectedResourceBinding(
		run, evidence.ResourceID, evidence.DatabaseSHA256, evidence.DatabaseSizeBytes,
		evidence.DatabaseFormat, domain.CalphadOwnerDeclaration{
			SchemaVersion: domain.CalphadOwnerDeclarationSchema, Authority: "resource_owner",
			DatabaseID: evidence.DatabaseID, Source: evidence.Source, LicenseID: evidence.LicenseID,
			AssessmentScope: evidence.AssessmentScope, ReferenceState: evidence.ReferenceState,
			AssessmentTemperatureLimitsK: evidence.TemperatureLimitsK,
			AssessmentPressureLimitsPa:   evidence.AssessmentPressureLimitsPa,
			DatabaseFormat:               evidence.DatabaseFormat,
		},
	)
}
