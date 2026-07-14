package integration

import (
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/httpapi"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	crossLanguageEvidenceMarker = "CALPHAD_CROSS_LANGUAGE_EVIDENCE "
	crossLanguageSchemaVersion  = "ultra.calphad.cross-language-qualification.v1"
)

var immutableImagePattern = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)

type typedCalphadEvidence struct {
	SchemaVersion   string `json:"schema_version"`
	Operation       string `json:"operation"`
	DatabaseBinding struct {
		Kind                       string    `json:"kind"`
		DatabaseID                 string    `json:"database_id"`
		ResourceID                 string    `json:"resource_id"`
		SHA256                     string    `json:"sha256"`
		SizeBytes                  int64     `json:"size_bytes"`
		DatabaseFormat             string    `json:"database_format"`
		Source                     string    `json:"source"`
		LicenseID                  string    `json:"license_id"`
		AssessmentScope            string    `json:"assessment_scope"`
		ReferenceState             string    `json:"reference_state"`
		TemperatureLimitsK         []float64 `json:"temperature_limits_K"`
		AssessmentPressureLimitsPa []float64 `json:"assessment_pressure_limits_Pa"`
		BindingSchema              string    `json:"binding_schema"`
		BindingAuthority           string    `json:"binding_authority"`
		DeclarationAuthority       string    `json:"declaration_authority"`
	} `json:"database_binding"`
	Request struct {
		Operation                string `json:"operation"`
		RuntimeImageID           string `json:"runtime_image_id"`
		InspectionArtifactSHA256 string `json:"inspection_artifact_sha256,omitempty"`
	} `json:"request"`
}

type loadedCalphadArtifact struct {
	Raw      []byte
	SHA256   string
	Evidence typedCalphadEvidence
}

type retainedCalphadEvent struct {
	Operation                string
	Status                   string
	EvidenceSHA256           string
	EvidenceSizeBytes        int64
	RuntimeImageID           string
	PycalphadVersion         string
	RunID                    string
	InspectionEvidenceSHA256 string
	DatabaseInventorySHA256  string
	RequestSHA256            string
	DatabaseFormat           string
	AssessmentPressureMinPa  float64
	AssessmentPressureMaxPa  float64
	EvidenceContractVersion  string
	Encoding                 string
	Payload                  []byte
}

func requiredQualificationEnvironment(t *testing.T, name string) string {
	t.Helper()
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		t.Fatalf("%s is required in cross-language qualification mode", name)
	}
	return value
}

func loadCalphadArtifact(t *testing.T, path, operation string) loadedCalphadArtifact {
	t.Helper()
	info, err := os.Lstat(path)
	if err != nil {
		t.Fatalf("stat %s artifact: %v", operation, err)
	}
	if !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 {
		t.Fatalf("%s artifact must be a regular non-symlink file", operation)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s artifact: %v", operation, err)
	}
	if len(raw) == 0 || len(raw) > 32<<20 {
		t.Fatalf("%s artifact has invalid size %d", operation, len(raw))
	}
	digest := sha256.Sum256(raw)
	digestText := hex.EncodeToString(digest[:])
	if info.Size() != int64(len(raw)) || strings.TrimSuffix(info.Name(), ".json") != digestText {
		t.Fatalf("%s artifact is not content-addressed: file=%s sha256=%s", operation, info.Name(), digestText)
	}
	var evidence typedCalphadEvidence
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&evidence); err != nil {
		// The typed projection intentionally omits large result/execution fields, so
		// decode again without DisallowUnknownFields after proving valid JSON below.
		if err := json.Unmarshal(raw, &evidence); err != nil {
			t.Fatalf("decode %s typed evidence: %v", operation, err)
		}
	}
	if evidence.SchemaVersion != "ultra.calphad.tool-evidence.v3" ||
		evidence.Operation != operation || evidence.Request.Operation != operation {
		t.Fatalf("%s artifact operation/schema mismatch: %+v", operation, evidence)
	}
	return loadedCalphadArtifact{Raw: raw, SHA256: digestText, Evidence: evidence}
}

func loadCalphadDatabaseInput(t *testing.T, path, expectedSHA, expectedFormat string, expectedSize int64) []byte {
	t.Helper()
	info, err := os.Lstat(path)
	if err != nil {
		t.Fatalf("stat CALPHAD database input: %v", err)
	}
	if !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 ||
		info.Size() != expectedSize || expectedSize <= 0 || expectedSize > 32<<20 {
		t.Fatalf("CALPHAD database input has invalid file identity: mode=%s size=%d", info.Mode(), info.Size())
	}
	if filepath.Base(path) != expectedSHA+"."+expectedFormat {
		t.Fatalf("CALPHAD database input is not content-addressed: %s", filepath.Base(path))
	}
	file, err := os.Open(path)
	if err != nil {
		t.Fatalf("open CALPHAD database input: %v", err)
	}
	defer file.Close()
	openedInfo, err := file.Stat()
	if err != nil {
		t.Fatalf("fstat CALPHAD database input: %v", err)
	}
	if !openedInfo.Mode().IsRegular() || openedInfo.Size() != expectedSize || !os.SameFile(info, openedInfo) {
		t.Fatal("CALPHAD database input changed between path validation and open")
	}
	raw, err := io.ReadAll(io.LimitReader(file, (32<<20)+1))
	if err != nil {
		t.Fatalf("read CALPHAD database input: %v", err)
	}
	if int64(len(raw)) != expectedSize {
		t.Fatalf("CALPHAD database input size=%d, want %d", len(raw), expectedSize)
	}
	digest := sha256.Sum256(raw)
	if hex.EncodeToString(digest[:]) != expectedSHA {
		t.Fatal("CALPHAD database input digest does not match the typed database binding")
	}
	return raw
}

func gzipBase64(t *testing.T, raw []byte) string {
	t.Helper()
	var compressed bytes.Buffer
	writer, err := gzip.NewWriterLevel(&compressed, gzip.BestCompression)
	if err != nil {
		t.Fatalf("gzip.NewWriterLevel: %v", err)
	}
	if _, err := writer.Write(raw); err != nil {
		t.Fatalf("gzip evidence: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("close gzip evidence: %v", err)
	}
	return base64.StdEncoding.EncodeToString(compressed.Bytes())
}

func callbackBody(t *testing.T, artifact loadedCalphadArtifact, status string) string {
	t.Helper()
	directory := artifact.Evidence.Operation
	if directory == "inspect" {
		directory = "inspection"
	}
	payload, err := json.Marshal(map[string]any{
		"status":               status,
		"operation":            artifact.Evidence.Operation,
		"evidence_path":        "/outputs/calphad/" + directory + "/" + artifact.SHA256 + ".json",
		"evidence_sha256":      artifact.SHA256,
		"evidence_size_bytes":  len(artifact.Raw),
		"runtime_image_id":     artifact.Evidence.Request.RuntimeImageID,
		"pycalphad_version":    domain.CalphadPycalphadVersion,
		"evidence_gzip_base64": gzipBase64(t, artifact.Raw),
	})
	if err != nil {
		t.Fatalf("marshal %s callback: %v", artifact.Evidence.Operation, err)
	}
	return string(payload)
}

func doRequest(
	t *testing.T,
	router http.Handler,
	method, path, body string,
	headers map[string]string,
) *httptest.ResponseRecorder {
	t.Helper()
	server := httptest.NewServer(router)
	defer server.Close()
	req, err := http.NewRequest(method, server.URL+path, strings.NewReader(body))
	if err != nil {
		t.Fatalf("create loopback HTTP request: %v", err)
	}
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	for key, value := range headers {
		req.Header.Set(key, value)
	}
	client := server.Client()
	client.Timeout = 30 * time.Second
	response, err := client.Do(req)
	if err != nil {
		t.Fatalf("execute loopback HTTP request: %v", err)
	}
	defer response.Body.Close()
	payload, err := io.ReadAll(io.LimitReader(response.Body, (32<<20)+1))
	if err != nil {
		t.Fatalf("read loopback HTTP response: %v", err)
	}
	if len(payload) > 32<<20 {
		t.Fatal("loopback HTTP response exceeded 32 MiB")
	}
	recorder := httptest.NewRecorder()
	recorder.Code = response.StatusCode
	for key, values := range response.Header {
		for _, value := range values {
			recorder.Header().Add(key, value)
		}
	}
	_, _ = recorder.Body.Write(payload)
	return recorder
}

func TestCalphadTypedCLIHTTPPostgresQualification(t *testing.T) {
	if os.Getenv("ULTRA_CALPHAD_CROSS_LANGUAGE_QUALIFICATION") != "1" {
		t.Skip("set ULTRA_CALPHAD_CROSS_LANGUAGE_QUALIFICATION=1 only for the dedicated live qualification gate")
	}

	servingDSN := requiredQualificationEnvironment(t, "ULTRA_CONTROL_TEST_DATABASE_URL")
	migrationDSN := requiredQualificationEnvironment(t, "ULTRA_CONTROL_TEST_MIGRATION_DATABASE_URL")
	databaseInputPath := requiredQualificationEnvironment(t, "ULTRA_CALPHAD_DATABASE_INPUT_ARTIFACT")
	inspectionPath := requiredQualificationEnvironment(t, "ULTRA_CALPHAD_INSPECTION_ARTIFACT")
	equilibriumPath := requiredQualificationEnvironment(t, "ULTRA_CALPHAD_EQUILIBRIUM_ARTIFACT")
	expectedRuntimeImage := strings.ToLower(requiredQualificationEnvironment(t, "ULTRA_CALPHAD_RUNTIME_IMAGE_ID"))
	if !immutableImagePattern.MatchString(expectedRuntimeImage) {
		t.Fatalf("ULTRA_CALPHAD_RUNTIME_IMAGE_ID must be immutable, got %q", expectedRuntimeImage)
	}

	inspection := loadCalphadArtifact(t, inspectionPath, "inspect")
	equilibrium := loadCalphadArtifact(t, equilibriumPath, "equilibrium")
	if !reflect.DeepEqual(inspection.Evidence.DatabaseBinding, equilibrium.Evidence.DatabaseBinding) {
		t.Fatal("inspect and equilibrium artifacts use different database bindings")
	}
	binding := inspection.Evidence.DatabaseBinding
	if binding.Kind != "resource" || binding.BindingSchema != "ultra.selected_resource.v1" ||
		binding.BindingAuthority != "control_resource_catalog" ||
		binding.DeclarationAuthority != "resource_owner" || len(binding.TemperatureLimitsK) != 2 ||
		(binding.DatabaseFormat != domain.CalphadDatabaseFormatTDB &&
			binding.DatabaseFormat != domain.CalphadDatabaseFormatDAT) ||
		len(binding.AssessmentPressureLimitsPa) != 2 ||
		binding.AssessmentPressureLimitsPa[0] != domain.CalphadReferencePressurePa ||
		binding.AssessmentPressureLimitsPa[1] != domain.CalphadReferencePressurePa {
		t.Fatalf("artifact database binding is not an owner-declared selected resource: %+v", binding)
	}
	if inspection.Evidence.Request.RuntimeImageID != expectedRuntimeImage ||
		equilibrium.Evidence.Request.RuntimeImageID != expectedRuntimeImage {
		t.Fatal("artifact runtime image does not match qualification runtime image")
	}
	if equilibrium.Evidence.Request.InspectionArtifactSHA256 != inspection.SHA256 {
		t.Fatalf("equilibrium inspection lineage=%q, want %q", equilibrium.Evidence.Request.InspectionArtifactSHA256, inspection.SHA256)
	}
	databaseInput := loadCalphadDatabaseInput(
		t, databaseInputPath, binding.SHA256, binding.DatabaseFormat, binding.SizeBytes,
	)

	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	migrationPool, err := pgxpool.New(ctx, migrationDSN)
	if err != nil {
		t.Fatalf("connect migration PostgreSQL: %v", err)
	}
	defer migrationPool.Close()
	servingPool, err := pgxpool.New(ctx, servingDSN)
	if err != nil {
		t.Fatalf("connect serving PostgreSQL: %v", err)
	}
	defer servingPool.Close()
	if err := migrationPool.Ping(ctx); err != nil {
		t.Fatalf("ping migration PostgreSQL: %v", err)
	}
	if err := servingPool.Ping(ctx); err != nil {
		t.Fatalf("ping serving PostgreSQL: %v", err)
	}
	if err := store.ApplyPostgresSchema(ctx, migrationPool); err != nil {
		t.Fatalf("apply PostgreSQL schema: %v", err)
	}

	connectionTarget, err := url.Parse(servingDSN)
	if err != nil {
		t.Fatalf("parse serving PostgreSQL URL: %v", err)
	}
	connectionTargetPort := 5432
	if portText := connectionTarget.Port(); portText != "" {
		connectionTargetPort, err = strconv.Atoi(portText)
		if err != nil {
			t.Fatalf("parse serving PostgreSQL port: %v", err)
		}
	}
	var servingRole, migrationRole, databaseName, serverAddress, transactionReadOnly string
	var serverPort int
	if err := servingPool.QueryRow(ctx, `
SELECT current_user, current_database(), COALESCE(inet_server_addr()::text, 'local'),
       COALESCE(inet_server_port(), 0), current_setting('transaction_read_only')`).Scan(
		&servingRole, &databaseName, &serverAddress, &serverPort, &transactionReadOnly,
	); err != nil {
		t.Fatalf("load serving PostgreSQL identity: %v", err)
	}
	if transactionReadOnly != "off" {
		t.Fatalf("qualification PostgreSQL is read-only: %q", transactionReadOnly)
	}
	if err := migrationPool.QueryRow(ctx, `SELECT current_user`).Scan(&migrationRole); err != nil {
		t.Fatalf("load migration PostgreSQL identity: %v", err)
	}
	if servingRole == migrationRole {
		t.Fatalf("serving and migration roles are not separated: %q", servingRole)
	}
	if err := store.GrantPostgresServingPrivileges(ctx, migrationPool, servingRole); err != nil {
		t.Fatalf("grant serving privileges: %v", err)
	}
	if err := store.VerifyPostgresSchema(ctx, servingPool); err != nil {
		t.Fatalf("verify PostgreSQL schema: %v", err)
	}
	if err := store.VerifyCalphadServingRole(ctx, servingPool); err != nil {
		t.Fatalf("verify CALPHAD serving role: %v", err)
	}
	roleStatus, err := store.InspectCalphadServingRole(ctx, servingPool)
	if err != nil {
		t.Fatalf("inspect CALPHAD serving role: %v", err)
	}

	postgresStore := store.NewPostgresStore(servingPool)
	now := domain.Now()
	owner := "calphad-cross-language-owner-" + binding.ResourceID
	ownerOrg := "calphad-cross-language-org-" + binding.ResourceID
	workerID := "calphad-cross-language-worker"
	workerToken := "calphad-cross-language-worker-token"
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
	if _, err := postgresStore.UpsertResource(ctx, domain.UpsertResourceInput{
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
		t.Fatalf("seed CALPHAD catalog resource: %v", err)
	}
	thread, err := postgresStore.CreateThread(ctx, domain.CreateThreadInput{
		UserID: owner, Title: "Cross-language CALPHAD qualification",
	})
	if err != nil {
		t.Fatalf("create qualification thread: %v", err)
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
	run, err := postgresStore.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID, UserID: owner, Goal: "inspect and calculate pinned CALPHAD evidence",
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
		t.Fatalf("create qualification run: %v", err)
	}
	lease, err := postgresStore.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID: run.RunID, WorkerID: workerID, TTL: 2 * time.Minute, Now: now,
	})
	if err != nil {
		t.Fatalf("acquire qualification run lease: %v", err)
	}

	router := httpapi.NewRouter(httpapi.ServerDeps{
		Version: "calphad-cross-language-qualification", Store: postgresStore,
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
		t.Fatalf("create revision status=%d body=%s", created.Code, created.Body.String())
	}

	callbackPath := "/v2/runs/" + url.PathEscape(run.RunID) + "/resources/" + resourcePath + "/calphad/validations"
	inspectResponse := doRequest(t, router, http.MethodPost, callbackPath,
		callbackBody(t, inspection, "input_validated"), workerHeaders)
	if inspectResponse.Code != http.StatusCreated {
		t.Fatalf("inspect callback status=%d body=%s", inspectResponse.Code, inspectResponse.Body.String())
	}
	equilibriumResponse := doRequest(t, router, http.MethodPost, callbackPath,
		callbackBody(t, equilibrium, "equilibrium_completed"), workerHeaders)
	if equilibriumResponse.Code != http.StatusCreated {
		t.Fatalf("equilibrium callback status=%d body=%s", equilibriumResponse.Code, equilibriumResponse.Body.String())
	}

	ledgerResponse := doRequest(t, router, http.MethodGet,
		"/v2/resources/"+resourcePath+"/calphad/ledger", "", ownerHeaders)
	if ledgerResponse.Code != http.StatusOK {
		t.Fatalf("read ledger status=%d body=%s", ledgerResponse.Code, ledgerResponse.Body.String())
	}
	var ledgerEnvelope struct {
		Ledger domain.CalphadLedgerRecord `json:"ledger"`
	}
	if err := json.Unmarshal(ledgerResponse.Body.Bytes(), &ledgerEnvelope); err != nil {
		t.Fatalf("decode ledger response: %v", err)
	}
	ledger := ledgerEnvelope.Ledger
	if ledger.Revision.ResourceID != binding.ResourceID || ledger.Revision.SHA256 != binding.SHA256 ||
		ledger.Revision.SizeBytes != binding.SizeBytes || ledger.Revision.DatabaseFormat != binding.DatabaseFormat ||
		ledger.LatestValidation == nil || ledger.LatestValidation.DatabaseFormat != binding.DatabaseFormat ||
		ledger.LatestValidation.Operation != "equilibrium" || !ledger.LatestValidation.Promotable ||
		ledger.LatestValidation.EvidenceRetention != domain.CalphadEvidenceRetentionRetained {
		t.Fatalf("ledger is not retained/promotable equilibrium evidence: %+v", ledger)
	}

	rows, err := servingPool.Query(ctx, `
SELECT event.operation, event.status, event.evidence_sha256, event.evidence_size_bytes,
       event.runtime_image_id, event.pycalphad_version, event.run_id,
       COALESCE(event.inspection_evidence_sha256, ''),
       event.database_inventory_sha256, event.request_sha256,
       event.database_format,
       event.assessment_pressure_min_pa, event.assessment_pressure_max_pa,
       event.evidence_contract_version, blob.encoding, blob.payload
FROM control_calphad_validation_events event
JOIN control_calphad_evidence_blobs blob
  ON blob.evidence_sha256=event.evidence_sha256
 AND blob.evidence_size_bytes=event.evidence_size_bytes
WHERE event.resource_id=$1 AND event.run_id=$2
  AND event.operation IN ('inspect','equilibrium')
ORDER BY CASE event.operation WHEN 'inspect' THEN 1 ELSE 2 END`, binding.ResourceID, run.RunID)
	if err != nil {
		t.Fatalf("query retained CALPHAD events: %v", err)
	}
	defer rows.Close()
	retained := make([]retainedCalphadEvent, 0, 2)
	for rows.Next() {
		var event retainedCalphadEvent
		if err := rows.Scan(
			&event.Operation, &event.Status, &event.EvidenceSHA256, &event.EvidenceSizeBytes,
			&event.RuntimeImageID, &event.PycalphadVersion, &event.RunID,
			&event.InspectionEvidenceSHA256, &event.DatabaseInventorySHA256,
			&event.RequestSHA256, &event.DatabaseFormat,
			&event.AssessmentPressureMinPa, &event.AssessmentPressureMaxPa,
			&event.EvidenceContractVersion, &event.Encoding, &event.Payload,
		); err != nil {
			t.Fatalf("scan retained CALPHAD event: %v", err)
		}
		retained = append(retained, event)
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("iterate retained CALPHAD events: %v", err)
	}
	if len(retained) != 2 {
		t.Fatalf("retained callback events=%d, want exactly 2", len(retained))
	}
	expectedArtifacts := map[string]loadedCalphadArtifact{
		"inspect": inspection, "equilibrium": equilibrium,
	}
	for _, event := range retained {
		artifact, ok := expectedArtifacts[event.Operation]
		if !ok {
			t.Fatalf("unexpected retained operation %q", event.Operation)
		}
		if event.EvidenceSHA256 != artifact.SHA256 || event.EvidenceSizeBytes != int64(len(artifact.Raw)) ||
			!bytes.Equal(event.Payload, artifact.Raw) || event.Encoding != "raw" ||
			event.RuntimeImageID != expectedRuntimeImage || event.PycalphadVersion != domain.CalphadPycalphadVersion ||
			event.RunID != run.RunID || event.EvidenceContractVersion != domain.CalphadEvidenceContractVersion ||
			event.DatabaseFormat != binding.DatabaseFormat ||
			event.AssessmentPressureMinPa != domain.CalphadReferencePressurePa ||
			event.AssessmentPressureMaxPa != domain.CalphadReferencePressurePa ||
			!regexp.MustCompile(`^[0-9a-f]{64}$`).MatchString(event.DatabaseInventorySHA256) ||
			!regexp.MustCompile(`^[0-9a-f]{64}$`).MatchString(event.RequestSHA256) {
			t.Fatalf("retained %s event is not content-bound: %+v", event.Operation, event)
		}
	}
	if retained[0].Operation != "inspect" || retained[0].Status != "input_validated" ||
		retained[0].InspectionEvidenceSHA256 != "" || retained[1].Operation != "equilibrium" ||
		retained[1].Status != "equilibrium_completed" ||
		retained[1].InspectionEvidenceSHA256 != inspection.SHA256 ||
		retained[0].DatabaseInventorySHA256 != retained[1].DatabaseInventorySHA256 ||
		retained[0].RequestSHA256 == retained[1].RequestSHA256 {
		t.Fatalf("retained request/inventory/inspection lineage is invalid: %+v", retained)
	}

	evidence := map[string]any{
		"schema_version":     crossLanguageSchemaVersion,
		"live_http_callback": true, "live_postgres": true,
		"database": map[string]any{
			"name": databaseName, "server_address": serverAddress, "server_port": serverPort,
			"connection_target_host": connectionTarget.Hostname(),
			"connection_target_port": connectionTargetPort,
			"transaction_read_only":  transactionReadOnly,
			"serving_role":           servingRole, "migration_role": migrationRole,
			"serving_role_superuser":                   roleStatus.Superuser,
			"serving_role_create_role":                 roleStatus.CreateRole,
			"serving_role_create_database":             roleStatus.CreateDB,
			"serving_role_replication":                 roleStatus.Replication,
			"serving_role_bypass_rls":                  roleStatus.BypassRLS,
			"serving_role_owned_tables":                roleStatus.OwnedTables,
			"serving_role_owned_functions":             roleStatus.OwnedFunctions,
			"calphad_owner_roles":                      roleStatus.OwnerRoles,
			"calphad_reachable_roles":                  roleStatus.ReachableRoles,
			"calphad_owner_role_reachable":             roleStatus.OwnerRoleReachable,
			"public_schema_owner":                      roleStatus.PublicSchemaOwner,
			"public_owner_role_reachable":              roleStatus.PublicOwnerReachable,
			"can_create_public_schema":                 roleStatus.CanCreatePublicSchema,
			"serving_role_select_all":                  roleStatus.CanSelectAll,
			"serving_role_insert_all":                  roleStatus.CanInsertAll,
			"serving_role_insert_any":                  roleStatus.CanInsertAny,
			"serving_role_mutation_privilege":          roleStatus.CanMutateCalphad,
			"serving_role_execute_create_revision":     roleStatus.CanExecuteCreateRevision,
			"serving_role_execute_append_validation":   roleStatus.CanExecuteAppendValidation,
			"serving_writer_functions_exact":           roleStatus.WriterFunctionsExact,
			"serving_execute_unexpected_writer":        roleStatus.CanExecuteUnexpectedWriter,
			"serving_role_execute_internal":            roleStatus.CanExecuteInternal,
			"serving_role_public_execute":              roleStatus.PublicCanExecute,
			"serving_unexpected_table_acl_grantees":    roleStatus.UnexpectedTableACLGrantees,
			"serving_unexpected_function_acl_grantees": roleStatus.UnexpectedFuncACLGrantees,
		},
		"resource_id": binding.ResourceID, "revision_id": ledger.Revision.RevisionID,
		"run_id": run.RunID, "runtime_image_id": expectedRuntimeImage,
		"pycalphad_version": domain.CalphadPycalphadVersion,
		"database_sha256":   binding.SHA256, "database_size_bytes": binding.SizeBytes,
		"database_format":               binding.DatabaseFormat,
		"assessment_pressure_limits_Pa": binding.AssessmentPressureLimitsPa,
		"database_inventory_sha256":     retained[0].DatabaseInventorySHA256,
		"inspect": map[string]any{
			"evidence_sha256":     retained[0].EvidenceSHA256,
			"evidence_size_bytes": retained[0].EvidenceSizeBytes,
			"request_sha256":      retained[0].RequestSHA256,
			"evidence_retention":  domain.CalphadEvidenceRetentionRetained,
			"promotable":          true, "postgres_bytes_exact": true,
		},
		"equilibrium": map[string]any{
			"evidence_sha256":            retained[1].EvidenceSHA256,
			"evidence_size_bytes":        retained[1].EvidenceSizeBytes,
			"request_sha256":             retained[1].RequestSHA256,
			"inspection_evidence_sha256": retained[1].InspectionEvidenceSHA256,
			"evidence_retention":         domain.CalphadEvidenceRetentionRetained,
			"promotable":                 true, "postgres_bytes_exact": true,
		},
	}
	encoded, err := json.Marshal(evidence)
	if err != nil {
		t.Fatalf("marshal cross-language evidence marker: %v", err)
	}
	t.Logf("%s%s", crossLanguageEvidenceMarker, encoded)
}
