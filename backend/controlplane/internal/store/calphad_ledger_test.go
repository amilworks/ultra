package store

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/url"
	"os"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

func calphadTestEvidence(label string, input domain.AppendCalphadValidationInput) ([]byte, string, string) {
	declaration := input.OwnerDeclaration
	schema := "ultra.calphad.tool-evidence.v3"
	resultKey := "result"
	if input.Status == "failed" || input.Status == "timeout" || input.Status == "unsupported" {
		schema = domain.CalphadFailureEvidenceSchemaVersion
		resultKey = "outcome"
	}
	payload, err := json.Marshal(domain.JSONMap{
		"schema_version": schema, "operation": input.Operation,
		"database_binding": domain.JSONMap{
			"kind": "resource", "database_id": declaration.DatabaseID,
			"resource_id": input.ResourceID, "sha256": input.DatabaseSHA256,
			"size_bytes": input.DatabaseSizeBytes, "database_format": input.DatabaseFormat,
			"source": declaration.Source, "license_id": declaration.LicenseID,
			"assessment_scope": declaration.AssessmentScope,
			"reference_state":  declaration.ReferenceState,
			"temperature_limits_K": []float64{
				declaration.AssessmentTemperatureLimitsK[0], declaration.AssessmentTemperatureLimitsK[1],
			},
			domain.CalphadAssessmentPressureLimitsMetadataKey: []float64{
				declaration.AssessmentPressureLimitsPa[0], declaration.AssessmentPressureLimitsPa[1],
			},
			"binding_schema":        "ultra.selected_resource.v1",
			"binding_authority":     "control_resource_catalog",
			"declaration_authority": "resource_owner",
		},
		"request": domain.JSONMap{"runtime_image_id": input.RuntimeImageID},
		resultKey: domain.JSONMap{"label": label},
		"execution_contract": domain.JSONMap{
			"interface":            "fixed ultra_deepagents.materials.calphad public surface",
			"caller_code_accepted": false, "caller_models_or_solver_options_accepted": false,
			"network": "none", "no_new_privileges": true, "read_only_root_filesystem": true,
			"cap_drop_all": true, "cpus_at_most": 8, "memory_bytes_at_most": int64(34359738368),
			"pids_at_most": 4096, "runtime_image_id": input.RuntimeImageID,
			"max_components": 32, "max_phases": 128, "max_axis_values": 64,
			"max_grid_points": 256, "wall_time_seconds": 30, "max_result_bytes": 16777216,
		},
		"validation_persistence": domain.JSONMap{
			"catalog_status": "pending", "catalog_metadata_updated": false,
			"mode": "immutable_per_run_evidence", "note": "server callback pending",
		},
	})
	if err != nil {
		panic(err)
	}
	return calphadTestEvidenceForBytes(payload, input.Operation)
}

func calphadTestEvidenceForBytes(payload []byte, operation string) ([]byte, string, string) {
	digest := sha256.Sum256(payload)
	sha := hex.EncodeToString(digest[:])
	directory := operation
	if operation == "inspect" {
		directory = "inspection"
	}
	return payload, sha, "/outputs/calphad/" + directory + "/" + sha + ".json"
}

func calphadTestMutateEvidence(
	t *testing.T,
	input domain.AppendCalphadValidationInput,
	mutate func(map[string]any),
) domain.AppendCalphadValidationInput {
	t.Helper()
	var evidence map[string]any
	if err := json.Unmarshal(input.EvidenceBytes, &evidence); err != nil {
		t.Fatalf("decode CALPHAD evidence fixture: %v", err)
	}
	mutate(evidence)
	payload, err := json.Marshal(evidence)
	if err != nil {
		t.Fatalf("encode mutated CALPHAD evidence fixture: %v", err)
	}
	input.EvidenceBytes, input.EvidenceSHA256, input.EvidencePath =
		calphadTestEvidenceForBytes(payload, input.Operation)
	input.EvidenceSizeBytes = int64(len(input.EvidenceBytes))
	return input
}

func calphadTestInput(label string, size int) ([]byte, string) {
	payload := bytes.Repeat([]byte(label), (size/len(label))+1)[:size]
	digest := sha256.Sum256(payload)
	return payload, hex.EncodeToString(digest[:])
}

var calphadTestPressureLimits = [2]float64{
	domain.CalphadReferencePressurePa, domain.CalphadReferencePressurePa,
}

func calphadTestOwnerMetadata(resourceID string) domain.JSONMap {
	return domain.JSONMap{
		"calphad": domain.JSONMap{
			"database_id": resourceID, "source": "https://example.org/assessments/" + resourceID,
			"license_id": "CC-BY-4.0", "assessment_scope": "Assessed binary equilibrium",
			"reference_state": "SER", "tdb_temperature_limits_K": []float64{300, 2000},
			domain.CalphadAssessmentPressureLimitsMetadataKey: []float64{
				calphadTestPressureLimits[0], calphadTestPressureLimits[1],
			},
		},
	}
}

func calphadTestOwnerDeclaration(resourceID, databaseFormat string) domain.CalphadOwnerDeclaration {
	return domain.CalphadOwnerDeclaration{
		SchemaVersion:                domain.CalphadOwnerDeclarationSchema,
		Authority:                    "resource_owner",
		DatabaseID:                   resourceID,
		Source:                       "https://example.org/assessments/" + resourceID,
		LicenseID:                    "CC-BY-4.0",
		AssessmentScope:              "Assessed binary equilibrium",
		ReferenceState:               "SER",
		AssessmentTemperatureLimitsK: [2]float64{300, 2000},
		AssessmentPressureLimitsPa:   calphadTestPressureLimits,
		DatabaseFormat:               databaseFormat,
	}
}

func calphadTestRevisionMetadata(resourceID, databaseFormat string) domain.JSONMap {
	return domain.JSONMap{
		domain.CalphadAssessmentPressureLimitsMetadataKey: []float64{
			calphadTestPressureLimits[0], calphadTestPressureLimits[1],
		},
		domain.CalphadOwnerDeclarationMetadataKey: calphadTestOwnerDeclaration(resourceID, databaseFormat),
	}
}

func calphadTestSelectedDescriptor(resourceID, sha string, size int64) domain.JSONMap {
	declaration := calphadTestOwnerDeclaration(resourceID, domain.CalphadDatabaseFormatTDB)
	return domain.JSONMap{
		"type": "selected_resource", "binding_schema": "ultra.selected_resource.v1",
		"authority": "control_resource_catalog", "resource_id": resourceID,
		"file_id": resourceID, "original_name": resourceID + ".tdb",
		"content_type": "application/x-thermocalc-tdb", "sha256": sha,
		"size_bytes": size, "database_format": domain.CalphadDatabaseFormatTDB,
		"calphad_governance_scope": "owner_validation",
		"metadata": domain.JSONMap{"calphad": domain.JSONMap{
			"declaration_authority": "resource_owner", "database_id": declaration.DatabaseID,
			"source": declaration.Source, "license_id": declaration.LicenseID,
			"assessment_scope": declaration.AssessmentScope,
			"reference_state":  declaration.ReferenceState,
			"assessment_temperature_limits_K": []float64{
				declaration.AssessmentTemperatureLimitsK[0], declaration.AssessmentTemperatureLimitsK[1],
			},
			"tdb_temperature_limits_K": []float64{
				declaration.AssessmentTemperatureLimitsK[0], declaration.AssessmentTemperatureLimitsK[1],
			},
			domain.CalphadAssessmentPressureLimitsMetadataKey: []float64{
				declaration.AssessmentPressureLimitsPa[0], declaration.AssessmentPressureLimitsPa[1],
			},
		}},
	}
}

type calphadLedgerTestStore interface {
	calphadLedgerStore
	CreateThread(context.Context, domain.CreateThreadInput) (domain.ThreadRecord, error)
	CreateRun(context.Context, domain.CreateRunInput) (domain.RunRecord, error)
	AcquireRunLease(context.Context, domain.AcquireRunLeaseInput) (domain.RunLeaseRecord, error)
	UpsertResource(context.Context, domain.UpsertResourceInput) (domain.ResourceRecord, error)
	MergeResourceMetadataForUser(context.Context, domain.MergeResourceMetadataInput) (domain.ResourceRecord, error)
}

func TestCalphadRunSelectedResourceBindingMatchesProductionWriterCardinality(t *testing.T) {
	t.Parallel()
	const resourceID = "calphad-selected-binding"
	databaseSHA := strings.Repeat("a", 64)
	declaration := calphadTestOwnerDeclaration(resourceID, domain.CalphadDatabaseFormatTDB)
	makeRun := func() domain.RunRecord {
		return domain.RunRecord{Metadata: domain.JSONMap{
			"file_ids": []string{resourceID},
			"resource_descriptors": []domain.JSONMap{
				calphadTestSelectedDescriptor(resourceID, databaseSHA, 512),
			},
		}}
	}
	accepts := func(run domain.RunRecord) bool {
		return CalphadRunHasSelectedResourceBinding(
			run, resourceID, databaseSHA, 512, domain.CalphadDatabaseFormatTDB, declaration,
		)
	}
	if !accepts(makeRun()) {
		t.Fatal("exact production selected-resource binding was rejected")
	}
	for name, mutate := range map[string]func(*domain.RunRecord){
		"missing scope": func(run *domain.RunRecord) {
			delete(run.Metadata["resource_descriptors"].([]domain.JSONMap)[0], "calphad_governance_scope")
		},
		"duplicate file": func(run *domain.RunRecord) {
			run.Metadata["file_ids"] = []string{resourceID, resourceID}
		},
		"duplicate descriptor": func(run *domain.RunRecord) {
			run.Metadata["resource_descriptors"] = append(
				run.Metadata["resource_descriptors"].([]domain.JSONMap),
				calphadTestSelectedDescriptor(resourceID, databaseSHA, 512),
			)
		},
		"malformed candidate": func(run *domain.RunRecord) {
			run.Metadata["resource_descriptors"] = append(
				run.Metadata["resource_descriptors"].([]domain.JSONMap),
				domain.JSONMap{"resource_id": resourceID},
			)
		},
	} {
		name, mutate := name, mutate
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			run := makeRun()
			mutate(&run)
			if accepts(run) {
				t.Fatal("ambiguous or governance-incomplete run authority was accepted")
			}
		})
	}
}

func TestCalphadOwnerDeclarationIsServerStampedAndSafe(t *testing.T) {
	t.Parallel()
	resource := domain.ResourceRecord{
		ResourceID: "owner-declaration-fallback-id", OriginalName: "assessment.dat",
		Metadata: domain.JSONMap{"calphad": domain.JSONMap{
			"source":     "https://example.org/assessments/public-dat",
			"license_id": "CC-BY-4.0", "assessment_scope": "Assessed binary equilibrium",
			"reference_state": "SER", "assessment_temperature_limits_K": []float64{300, 2000},
			"tdb_temperature_limits_K":                        []float64{300, 2000},
			domain.CalphadAssessmentPressureLimitsMetadataKey: calphadTestPressureLimits,
			"declaration_authority":                           "worker-supplied-value-is-not-an-authority",
		}},
	}
	declaration, err := CalphadOwnerDeclarationForResource(resource)
	if err != nil {
		t.Fatalf("derive safe owner declaration: %v", err)
	}
	if declaration.DatabaseID != resource.ResourceID || declaration.Authority != "resource_owner" ||
		declaration.DatabaseFormat != domain.CalphadDatabaseFormatDAT ||
		declaration.AssessmentTemperatureLimitsK != [2]float64{300, 2000} ||
		declaration.AssessmentPressureLimitsPa != calphadTestPressureLimits {
		t.Fatalf("server-stamped owner declaration=%+v", declaration)
	}

	for name, mutate := range map[string]func(domain.JSONMap){
		"credentialed source URI": func(metadata domain.JSONMap) {
			metadata["source"] = "https://vendor:credential-secret@example.org/database"
		},
		"query-bearing source URI": func(metadata domain.JSONMap) {
			metadata["source"] = "https://example.org/database?token=opaque"
		},
		"license body": func(metadata domain.JSONMap) {
			metadata["license_id"] = "Confidential proprietary license agreement"
		},
		"temperature alias drift": func(metadata domain.JSONMap) {
			metadata["tdb_temperature_limits_K"] = []float64{400, 2000}
		},
	} {
		name, mutate := name, mutate
		t.Run(name, func(t *testing.T) {
			mutated := cloneJSONMap(resource.Metadata)
			calphad := mutated["calphad"].(domain.JSONMap)
			mutate(calphad)
			candidate := resource
			candidate.Metadata = mutated
			if _, declarationErr := CalphadOwnerDeclarationForResource(candidate); !errors.Is(
				declarationErr, ErrCalphadOwnerDeclarationInvalid,
			) {
				t.Fatalf("unsafe owner declaration err=%v, want ErrCalphadOwnerDeclarationInvalid", declarationErr)
			}
		})
	}
}

func exerciseCalphadLedger(t *testing.T, ledger calphadLedgerTestStore, suffix string) {
	t.Helper()
	ctx := context.Background()
	now := time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)
	owner := "calphad-owner-" + suffix
	org := "calphad-org-" + suffix
	parentID := "calphad-parent-" + suffix
	childID := "calphad-child-" + suffix
	runtimeImageID := "sha256:" + strings.Repeat("d", 64)
	databaseInventorySHA := strings.Repeat("c", 64)
	inspectionRequestSHA := strings.Repeat("1", 64)
	equilibriumRequestSHA := strings.Repeat("2", 64)
	scheilRequestSHA := strings.Repeat("4", 64)
	parentBytes, parentSHA := calphadTestInput("PARENT-TDB", 100)
	childBytes, childSHA := calphadTestInput("CHILD-TDB", 200)
	thread, err := ledger.CreateThread(ctx, domain.CreateThreadInput{UserID: owner, Title: "CALPHAD"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := ledger.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID, UserID: owner, Goal: "CALPHAD validation",
		Metadata: domain.JSONMap{
			"org_id":   org,
			"file_ids": []string{childID},
			"resource_descriptors": []domain.JSONMap{
				calphadTestSelectedDescriptor(childID, childSHA, int64(len(childBytes))),
			},
			domain.CalphadRuntimePolicyMetadataKey: domain.JSONMap{
				"schema_version": domain.CalphadRuntimePolicySchema, "authority": "control_plane",
				"runtime_image_id": runtimeImageID, "pycalphad_version": domain.CalphadPycalphadVersion,
				"network": domain.CalphadRuntimeNetwork, "no_new_privileges": true,
				"read_only_root_filesystem": true, "cap_drop_all": true,
				"cpus_at_most":         domain.CalphadRuntimeCPUsAtMost,
				"memory_bytes_at_most": domain.CalphadRuntimeMemoryBytesAtMost,
				"pids_at_most":         domain.CalphadRuntimePIDsAtMost,
			},
			"principal": domain.JSONMap{
				"user_id": owner, "org_id": org, "role": "researcher",
			},
		},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	lease, err := ledger.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID: run.RunID, WorkerID: "calphad-worker-" + suffix,
		TTL: time.Hour, Now: domain.Now(),
	})
	if err != nil {
		t.Fatalf("AcquireRunLease: %v", err)
	}
	resource := func(resourceID string, payload []byte) {
		digest := sha256.Sum256(payload)
		sha := hex.EncodeToString(digest[:])
		size := int64(len(payload))
		if _, err := ledger.UpsertResource(ctx, domain.UpsertResourceInput{
			ResourceID:   resourceID,
			OriginalName: resourceID + ".tdb",
			ContentType:  "application/x-thermocalc-tdb",
			SizeBytes:    size,
			SHA256:       sha,
			OwnerUserID:  owner,
			OwnerOrgID:   org,
			Status:       "active",
			CreatedAt:    now,
			UpdatedAt:    now,
			Metadata:     calphadTestOwnerMetadata(resourceID),
		}); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resourceID, err)
		}
	}
	resource(parentID, parentBytes)
	resource(childID, childBytes)
	metadataOnlyID := "calphad-metadata-only-" + suffix
	if _, err := ledger.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID: metadataOnlyID, OriginalName: "notes.txt", ContentType: "text/plain",
		SizeBytes: 10, SHA256: strings.Repeat("f", 64), OwnerUserID: owner, OwnerOrgID: org,
		Status: "active", CreatedAt: now, UpdatedAt: now,
		Metadata: domain.JSONMap{"calphad": domain.JSONMap{"validation_status": "owner_claimed"}},
	}); err != nil {
		t.Fatalf("UpsertResource(metadata-only): %v", err)
	}
	if _, err := ledger.CreateCalphadRevision(ctx, domain.CreateCalphadRevisionInput{
		ResourceID: metadataOnlyID, OwnerUserID: owner, OwnerOrgID: org,
	}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("metadata-only CALPHAD classification err=%v, want ErrNotFound", err)
	}
	datID := "calphad-extension-dat-" + suffix
	datBytes, datSHA := calphadTestInput("DAT", 3)
	if _, err := ledger.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID: datID, OriginalName: "thermodynamic-catalog.dat",
		ContentType: "application/octet-stream", SizeBytes: int64(len(datBytes)),
		SHA256: datSHA, OwnerUserID: owner, OwnerOrgID: org,
		Status: "active", CreatedAt: now, UpdatedAt: now,
		Metadata: calphadTestOwnerMetadata(datID),
	}); err != nil {
		t.Fatalf("UpsertResource(.dat): %v", err)
	}
	datRevision, err := ledger.CreateCalphadRevision(ctx, domain.CreateCalphadRevisionInput{
		ResourceID: datID, OwnerUserID: owner, OwnerOrgID: org,
		AssessmentPressureLimitsPa: calphadTestPressureLimits,
		CreatedAt:                  now.Add(-time.Hour), InputBytes: datBytes,
	})
	if err != nil {
		t.Fatalf("CreateCalphadRevision(.dat): %v", err)
	}
	if datRevision.DatabaseFormat != domain.CalphadDatabaseFormatDAT {
		t.Fatalf("CreateCalphadRevision(.dat) format=%q, want dat", datRevision.DatabaseFormat)
	}
	dbID := "calphad-extension-db-" + suffix
	dbBytes, dbSHA := calphadTestInput("DB", 2)
	if _, err := ledger.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID: dbID, OriginalName: "unsupported.db", ContentType: "application/x-thermocalc-tdb",
		SizeBytes: int64(len(dbBytes)), SHA256: dbSHA, OwnerUserID: owner, OwnerOrgID: org,
		Status: "active", CreatedAt: now, UpdatedAt: now,
	}); err != nil {
		t.Fatalf("UpsertResource(.db): %v", err)
	}
	if _, err := ledger.CreateCalphadRevision(ctx, domain.CreateCalphadRevisionInput{
		ResourceID: dbID, OwnerUserID: owner, OwnerOrgID: org, InputBytes: dbBytes,
	}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("unsupported .db revision err=%v, want ErrNotFound", err)
	}

	parent, err := ledger.CreateCalphadRevision(ctx, domain.CreateCalphadRevisionInput{
		ResourceID: parentID, OwnerUserID: owner, OwnerOrgID: org, CreatedByUserID: owner, CreatedAt: now,
		AssessmentPressureLimitsPa: calphadTestPressureLimits, InputBytes: parentBytes,
	})
	if err != nil {
		t.Fatalf("CreateCalphadRevision(parent): %v", err)
	}
	if parent.DatabaseFormat != domain.CalphadDatabaseFormatTDB {
		t.Fatalf("CreateCalphadRevision(.tdb) format=%q, want tdb", parent.DatabaseFormat)
	}
	if _, err := ledger.CreateCalphadRevision(ctx, domain.CreateCalphadRevisionInput{
		ResourceID: childID, OwnerUserID: owner, OwnerOrgID: org,
		ExpectedSHA256: strings.Repeat("a", 64), ExpectedSizeBytes: 199,
		AssessmentPressureLimitsPa: calphadTestPressureLimits, InputBytes: childBytes,
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("stale expected binding err=%v, want ErrConflict", err)
	}
	revisionMetadata := domain.JSONMap{
		"nested": domain.JSONMap{"labels": []any{"original"}},
	}
	child, err := ledger.CreateCalphadRevision(ctx, domain.CreateCalphadRevisionInput{
		ResourceID: childID, OwnerUserID: owner, OwnerOrgID: org, ParentRevisionID: parent.RevisionID,
		ExpectedSHA256: childSHA, ExpectedSizeBytes: 200, InputBytes: childBytes,
		AssessmentPressureLimitsPa: calphadTestPressureLimits,
		CreatedByUserID:            owner, CreatedAt: now.Add(time.Second),
		Metadata: revisionMetadata,
	})
	if err != nil {
		t.Fatalf("CreateCalphadRevision(child): %v", err)
	}
	if child.ParentRevisionID != parent.RevisionID || child.SHA256 != childSHA || child.SizeBytes != 200 ||
		child.DatabaseFormat != domain.CalphadDatabaseFormatTDB {
		t.Fatalf("child revision = %+v", child)
	}
	if child.AssessmentPressureLimitsPa != calphadTestPressureLimits {
		t.Fatalf("child pressure limits = %v, want %v", child.AssessmentPressureLimitsPa, calphadTestPressureLimits)
	}
	childDeclaration, declarationOK := calphadRevisionOwnerDeclaration(child)
	if !declarationOK || childDeclaration.DatabaseFormat != domain.CalphadDatabaseFormatTDB {
		t.Fatalf("child owner declaration=%+v ok=%t", childDeclaration, declarationOK)
	}
	if pressureMetadata, ok := calphadAssessmentPressureLimitsFromValue(
		child.Metadata[domain.CalphadAssessmentPressureLimitsMetadataKey],
	); !ok || pressureMetadata != calphadTestPressureLimits {
		t.Fatalf("child pressure metadata = %v ok=%t", pressureMetadata, ok)
	}
	revisionMetadata["nested"].(domain.JSONMap)["labels"].([]any)[0] = "mutated-input"
	if _, retained := child.Metadata["nested"]; retained {
		t.Fatal("caller-supplied revision metadata escaped the server-owned metadata boundary")
	}
	child.Metadata["server_managed"] = false

	// Parent identity is immutable even on replay; omission cannot weaken it.
	if _, err := ledger.CreateCalphadRevision(ctx, domain.CreateCalphadRevisionInput{
		ResourceID: childID, OwnerUserID: owner, OwnerOrgID: org,
		AssessmentPressureLimitsPa: calphadTestPressureLimits, InputBytes: childBytes,
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("parented revision replay without parent err=%v, want ErrConflict", err)
	}
	// Exact idempotent replay must not mint a second pending event.
	if again, err := ledger.CreateCalphadRevision(ctx, domain.CreateCalphadRevisionInput{
		ResourceID: childID, OwnerUserID: owner, OwnerOrgID: org, ParentRevisionID: parent.RevisionID,
		AssessmentPressureLimitsPa: calphadTestPressureLimits, InputBytes: childBytes,
	}); err != nil || again.RevisionID != child.RevisionID {
		t.Fatalf("idempotent CreateCalphadRevision = %+v err=%v", again, err)
	}
	if _, err := ledger.CreateCalphadRevision(ctx, domain.CreateCalphadRevisionInput{
		ResourceID: childID, OwnerUserID: owner, OwnerOrgID: org,
		AssessmentPressureLimitsPa: [2]float64{100000, 200000}, InputBytes: childBytes,
	}); !errors.Is(err, ErrCalphadPressureLimitsInvalid) && !errors.Is(err, ErrConflict) {
		t.Fatalf("revision pressure drift err=%v, want fail closed conflict", err)
	}
	initial, err := ledger.GetCalphadLedgerForOwner(ctx, childID, owner, org)
	if err != nil {
		t.Fatalf("GetCalphadLedgerForOwner(initial): %v", err)
	}
	if len(initial.Validations) != 1 || initial.LatestValidation == nil ||
		initial.LatestValidation.Status != "pending" ||
		initial.LatestValidation.DatabaseSHA256 != child.SHA256 ||
		initial.LatestValidation.DatabaseSizeBytes != child.SizeBytes ||
		initial.LatestValidation.DatabaseFormat != child.DatabaseFormat {
		t.Fatalf("initial ledger = %+v", initial)
	}
	if managed, ok := initial.Revision.Metadata["server_managed"].(bool); !ok || !managed {
		t.Fatalf("revision server-owned metadata was mutable: %+v", initial.Revision.Metadata)
	}
	if _, retained := initial.Revision.Metadata["nested"]; retained {
		t.Fatal("caller-supplied revision metadata was persisted")
	}
	retainedInput, err := ledger.GetCalphadRevisionInputForOwner(ctx, childID, owner, org)
	if err != nil || retainedInput.RevisionID != child.RevisionID || retainedInput.SHA256 != childSHA ||
		retainedInput.SizeBytes != int64(len(childBytes)) ||
		retainedInput.DatabaseFormat != domain.CalphadDatabaseFormatTDB ||
		!bytes.Equal(retainedInput.Bytes, childBytes) {
		t.Fatalf("retained CALPHAD input = %+v err=%v", retainedInput, err)
	}
	retainedInput.Bytes[0] ^= 0xff
	reloadedInput, err := ledger.GetCalphadRevisionInputForOwner(ctx, childID, owner, org)
	if err != nil || !bytes.Equal(reloadedInput.Bytes, childBytes) {
		t.Fatalf("retained input return aliased storage: %+v err=%v", reloadedInput, err)
	}
	if _, err := ledger.GetCalphadRevisionInputForOwner(ctx, childID, "other-user", org); !errors.Is(err, ErrNotFound) {
		t.Fatalf("cross-owner retained input err=%v, want ErrNotFound", err)
	}

	inspectionInput := domain.AppendCalphadValidationInput{
		ResourceID: childID, OwnerUserID: owner, OwnerOrgID: org,
		DatabaseSHA256: childSHA, DatabaseSizeBytes: 200,
		DatabaseFormat: child.DatabaseFormat, OwnerDeclaration: childDeclaration,
		AssessmentPressureLimitsPa: calphadTestPressureLimits,
		DatabaseInventorySHA256:    databaseInventorySHA, RequestSHA256: inspectionRequestSHA,
		Status: "input_validated", Operation: "inspect",
		RuntimeImageID: runtimeImageID, PycalphadVersion: domain.CalphadPycalphadVersion,
		RunID: run.RunID, LeaseWorkerID: lease.WorkerID, LeaseToken: lease.LeaseToken,
		CreatedByAuthority: "trusted_worker", CreatedAt: now.Add(1500 * time.Millisecond),
		Metadata: domain.JSONMap{"server_managed": true},
	}
	inspectionBytes, inspectionSHA, inspectionPath := calphadTestEvidence("inspection", inspectionInput)
	inspectionInput.EvidenceBytes = inspectionBytes
	inspectionInput.EvidenceSHA256 = inspectionSHA
	inspectionInput.EvidencePath = inspectionPath
	inspectionInput.EvidenceSizeBytes = int64(len(inspectionBytes))
	premature := inspectionInput
	premature.Status = "equilibrium_completed"
	premature.Operation = "equilibrium"
	premature.RequestSHA256 = equilibriumRequestSHA
	premature.InspectionEvidenceSHA256 = inspectionSHA
	prematureBytes, prematureSHA, prematurePath := calphadTestEvidence("premature-equilibrium", premature)
	premature.EvidenceBytes = prematureBytes
	premature.EvidenceSHA256 = prematureSHA
	premature.EvidencePath = prematurePath
	premature.EvidenceSizeBytes = int64(len(prematureBytes))
	if _, err := ledger.AppendCalphadValidation(ctx, premature); !errors.Is(err, ErrCalphadInspectionRequired) {
		t.Fatalf("equilibrium before retained inspection err=%v, want ErrCalphadInspectionRequired", err)
	}
	pressureDrift := inspectionInput
	pressureDrift.AssessmentPressureLimitsPa = [2]float64{100000, 200000}
	if _, err := ledger.AppendCalphadValidation(ctx, pressureDrift); !errors.Is(err, ErrConflict) {
		t.Fatalf("validation pressure drift err=%v, want ErrConflict", err)
	}
	unauthorizedRuntime := inspectionInput
	unauthorizedRuntime.RuntimeImageID = "sha256:" + strings.Repeat("f", 64)
	unauthorizedRuntime.EvidenceBytes, unauthorizedRuntime.EvidenceSHA256, unauthorizedRuntime.EvidencePath =
		calphadTestEvidence("unauthorized-runtime", unauthorizedRuntime)
	unauthorizedRuntime.EvidenceSizeBytes = int64(len(unauthorizedRuntime.EvidenceBytes))
	if _, err := ledger.AppendCalphadValidation(ctx, unauthorizedRuntime); !errors.Is(err, ErrCalphadRuntimePolicyInvalid) {
		t.Fatalf("worker-asserted runtime err=%v, want ErrCalphadRuntimePolicyInvalid", err)
	}
	inspection, err := ledger.AppendCalphadValidation(ctx, inspectionInput)
	if err != nil {
		t.Fatalf("AppendCalphadValidation(inspect): %v", err)
	}
	if inspection.EvidenceRetention != domain.CalphadEvidenceRetentionRetained || !inspection.Promotable {
		t.Fatalf("inspection retention=%q promotable=%t", inspection.EvidenceRetention, inspection.Promotable)
	}

	validationMetadata := domain.JSONMap{"nested": domain.JSONMap{"labels": []any{"evidence"}}}
	validationInput := domain.AppendCalphadValidationInput{
		ResourceID: childID, OwnerUserID: owner, OwnerOrgID: org,
		DatabaseSHA256: childSHA, DatabaseSizeBytes: 200,
		DatabaseFormat: child.DatabaseFormat, OwnerDeclaration: childDeclaration,
		AssessmentPressureLimitsPa: calphadTestPressureLimits,
		DatabaseInventorySHA256:    databaseInventorySHA, RequestSHA256: equilibriumRequestSHA,
		Status: "equilibrium_completed", Operation: "equilibrium",
		RuntimeImageID:   runtimeImageID,
		PycalphadVersion: domain.CalphadPycalphadVersion, RunID: run.RunID,
		InspectionEvidenceSHA256: inspectionSHA,
		LeaseWorkerID:            lease.WorkerID, LeaseToken: lease.LeaseToken,
		CreatedByAuthority: "trusted_worker", CreatedAt: now.Add(2 * time.Second),
		Metadata: validationMetadata,
	}
	evidenceBytes, evidenceSHA, evidencePath := calphadTestEvidence("equilibrium", validationInput)
	validationInput.EvidenceBytes = evidenceBytes
	validationInput.EvidenceSHA256 = evidenceSHA
	validationInput.EvidencePath = evidencePath
	validationInput.EvidenceSizeBytes = int64(len(evidenceBytes))
	wrongInspection := validationInput
	wrongInspection.InspectionEvidenceSHA256 = strings.Repeat("9", 64)
	if _, err := ledger.AppendCalphadValidation(ctx, wrongInspection); !errors.Is(err, ErrCalphadInspectionRequired) {
		t.Fatalf("equilibrium with wrong inspection err=%v, want ErrCalphadInspectionRequired", err)
	}
	wrongInventory := validationInput
	wrongInventory.DatabaseInventorySHA256 = strings.Repeat("9", 64)
	if _, err := ledger.AppendCalphadValidation(ctx, wrongInventory); !errors.Is(err, ErrCalphadInspectionRequired) {
		t.Fatalf("equilibrium with wrong semantic inventory err=%v, want ErrCalphadInspectionRequired", err)
	}
	validation, err := ledger.AppendCalphadValidation(ctx, validationInput)
	if err != nil {
		t.Fatalf("AppendCalphadValidation: %v", err)
	}
	if validation.RevisionID != child.RevisionID {
		t.Fatalf("validation revision = %q, want %q", validation.RevisionID, child.RevisionID)
	}
	if validation.InspectionEvidenceSHA256 != inspectionSHA ||
		validation.DatabaseInventorySHA256 != databaseInventorySHA ||
		validation.RequestSHA256 != equilibriumRequestSHA ||
		validation.EvidenceContractVersion != domain.CalphadEvidenceContractVersion ||
		validation.EvidenceRetention != domain.CalphadEvidenceRetentionRetained || !validation.Promotable {
		t.Fatalf("equilibrium lineage/retention = %+v", validation)
	}
	scheilInput := validationInput
	scheilInput.Status = "scheil_completed"
	scheilInput.Operation = "scheil"
	scheilInput.RequestSHA256 = scheilRequestSHA
	scheilInput.CreatedAt = now.Add(2250 * time.Millisecond)
	scheilInput.Metadata = domain.JSONMap{"method": "Scheil-Gulliver"}
	scheilBytes, scheilSHA, scheilPath := calphadTestEvidence("scheil", scheilInput)
	scheilInput.EvidenceBytes = scheilBytes
	scheilInput.EvidenceSHA256 = scheilSHA
	scheilInput.EvidencePath = scheilPath
	scheilInput.EvidenceSizeBytes = int64(len(scheilBytes))
	scheilValidation, err := ledger.AppendCalphadValidation(ctx, scheilInput)
	if err != nil {
		t.Fatalf("AppendCalphadValidation(scheil): %v", err)
	}
	if scheilValidation.Status != "scheil_completed" || scheilValidation.Operation != "scheil" ||
		scheilValidation.InspectionEvidenceSHA256 != inspectionSHA ||
		scheilValidation.DatabaseInventorySHA256 != databaseInventorySHA ||
		scheilValidation.RequestSHA256 != scheilRequestSHA ||
		scheilValidation.EvidenceRetention != domain.CalphadEvidenceRetentionRetained ||
		!scheilValidation.Promotable {
		t.Fatalf("Scheil lineage/retention = %+v", scheilValidation)
	}
	scheilReplay, err := ledger.AppendCalphadValidation(ctx, scheilInput)
	if err != nil || scheilReplay.ValidationID != scheilValidation.ValidationID {
		t.Fatalf("Scheil idempotent replay=%+v err=%v", scheilReplay, err)
	}
	mismatchedScheilTuple := scheilInput
	mismatchedScheilTuple.Operation = "equilibrium"
	mismatchedScheilTuple.EvidenceBytes, mismatchedScheilTuple.EvidenceSHA256,
		mismatchedScheilTuple.EvidencePath = calphadTestEvidence(
		"mismatched-scheil-tuple", mismatchedScheilTuple,
	)
	mismatchedScheilTuple.EvidenceSizeBytes = int64(len(mismatchedScheilTuple.EvidenceBytes))
	if _, err := ledger.AppendCalphadValidation(ctx, mismatchedScheilTuple); !errors.Is(err, ErrConflict) {
		t.Fatalf("Scheil status/operation mismatch err=%v, want ErrConflict", err)
	}
	wrongLease := validationInput
	wrongLease.LeaseToken += "-forged"
	if _, err := ledger.AppendCalphadValidation(ctx, wrongLease); !errors.Is(err, ErrCalphadRunLeaseInvalid) {
		t.Fatalf("wrong atomic run lease err=%v, want ErrCalphadRunLeaseInvalid", err)
	}
	validationMetadata["nested"].(domain.JSONMap)["labels"].([]any)[0] = "mutated-input"
	if _, retained := validation.Metadata["nested"]; retained {
		t.Fatal("caller-supplied validation metadata escaped the server-owned metadata boundary")
	}
	validation.Metadata["server_managed"] = false
	replayed, err := ledger.AppendCalphadValidation(ctx, validationInput)
	if err != nil || replayed.ValidationID != validation.ValidationID {
		t.Fatalf("idempotent validation replay=%+v err=%v", replayed, err)
	}
	validationInput.EvidenceBytes[0] ^= 0xff
	alternateObservation := validationInput
	otherBytes, otherSHA, otherPath := calphadTestEvidence("inconsistent", alternateObservation)
	alternateObservation.EvidenceBytes = otherBytes
	alternateObservation.EvidenceSHA256 = otherSHA
	alternateObservation.EvidenceSizeBytes = int64(len(otherBytes))
	alternateObservation.EvidencePath = otherPath
	alternate, err := ledger.AppendCalphadValidation(ctx, alternateObservation)
	if err != nil || alternate.ValidationID == validation.ValidationID {
		t.Fatalf("same request with distinct valid evidence=%+v first=%+v err=%v", alternate, validation, err)
	}
	alternateReplay, err := ledger.AppendCalphadValidation(ctx, alternateObservation)
	if err != nil || alternateReplay.ValidationID != alternate.ValidationID {
		t.Fatalf("alternate evidence idempotent replay=%+v err=%v", alternateReplay, err)
	}
	conflictingEvidenceIdentity := alternateObservation
	conflictingEvidenceIdentity.RequestSHA256 = strings.Repeat("8", 64)
	if _, err := ledger.AppendCalphadValidation(ctx, conflictingEvidenceIdentity); !errors.Is(err, ErrConflict) {
		t.Fatalf("same evidence with conflicting request identity err=%v, want ErrConflict", err)
	}

	secondInput := validationInput
	secondInput.RequestSHA256 = strings.Repeat("3", 64)
	secondBytes, secondSHA, secondPath := calphadTestEvidence("equilibrium-2 Δ café", secondInput)
	secondInput.EvidenceBytes = secondBytes
	secondInput.EvidenceSHA256 = secondSHA
	secondInput.EvidenceSizeBytes = int64(len(secondBytes))
	secondInput.EvidencePath = secondPath
	secondInput.CreatedAt = now.Add(2500 * time.Millisecond)
	secondInput.Metadata = domain.JSONMap{"nested": domain.JSONMap{"labels": []any{"evidence-2"}}}
	second, err := ledger.AppendCalphadValidation(ctx, secondInput)
	if err != nil || second.ValidationID == validation.ValidationID {
		t.Fatalf("distinct equilibrium request replay=%+v first=%+v err=%v", second, validation, err)
	}
	secondReplay, err := ledger.AppendCalphadValidation(ctx, secondInput)
	if err != nil || secondReplay.ValidationID != second.ValidationID {
		t.Fatalf("second equilibrium idempotent replay=%+v err=%v", secondReplay, err)
	}

	for _, test := range []struct {
		name       string
		validation domain.CalphadValidationRecord
		want       []byte
	}{
		{name: "inspection", validation: inspection, want: inspectionBytes},
		{name: "Scheil", validation: scheilValidation, want: scheilBytes},
		{name: "unicode whitespace newline", validation: second, want: secondBytes},
	} {
		t.Run("exact evidence replay "+test.name, func(t *testing.T) {
			replayedEvidence, replayErr := ledger.GetCalphadValidationEvidenceForOwner(
				ctx, childID, test.validation.ValidationID, owner, org,
			)
			if replayErr != nil || replayedEvidence.ValidationID != test.validation.ValidationID ||
				replayedEvidence.SHA256 != test.validation.EvidenceSHA256 ||
				replayedEvidence.SizeBytes != int64(len(test.want)) ||
				!bytes.Equal(replayedEvidence.Bytes, test.want) {
				t.Fatalf("exact evidence=%+v bytes=%q err=%v", replayedEvidence, replayedEvidence.Bytes, replayErr)
			}
			replayedEvidence.Bytes[0] ^= 0xff
			againEvidence, replayErr := ledger.GetCalphadValidationEvidenceForOwner(
				ctx, childID, test.validation.ValidationID, owner, org,
			)
			if replayErr != nil || !bytes.Equal(againEvidence.Bytes, test.want) {
				t.Fatalf("evidence return aliased retention bytes=%q err=%v", againEvidence.Bytes, replayErr)
			}
		})
	}
	if _, err := ledger.GetCalphadValidationEvidenceForOwner(
		ctx, childID, second.ValidationID, "other-user", org,
	); !errors.Is(err, ErrNotFound) {
		t.Fatalf("cross-owner evidence replay err=%v, want ErrNotFound", err)
	}
	if _, err := ledger.GetCalphadValidationEvidenceForOwner(
		ctx, parentID, second.ValidationID, owner, org,
	); !errors.Is(err, ErrNotFound) {
		t.Fatalf("cross-resource evidence replay err=%v, want ErrNotFound", err)
	}
	if _, err := ledger.GetCalphadValidationEvidenceForOwner(
		ctx, childID, initial.LatestValidation.ValidationID, owner, org,
	); !errors.Is(err, ErrCalphadEvidenceRetentionRequired) {
		t.Fatalf("authorized missing evidence err=%v, want ErrCalphadEvidenceRetentionRequired", err)
	}
	t.Run("retained_terminal_statuses", func(t *testing.T) {
		for index, terminal := range []struct {
			status string
			domain domain.CalphadFailureDomain
			stage  domain.CalphadFailureStage
			code   domain.CalphadFailureCode
		}{
			{
				status: "failed", domain: domain.CalphadFailureDomainInput,
				stage: domain.CalphadFailureStageParse,
				code:  domain.CalphadFailureCodeParseFailed,
			},
			{
				status: "timeout", domain: domain.CalphadFailureDomainPlatform,
				stage: domain.CalphadFailureStageSandboxRuntime,
				code:  domain.CalphadFailureCodeSandboxTimeout,
			},
			{
				status: "unsupported", domain: domain.CalphadFailureDomainInput,
				stage: domain.CalphadFailureStageParse,
				code:  domain.CalphadFailureCodeParseUnsupported,
			},
		} {
			input := inspectionInput
			input.Status = terminal.status
			input.DatabaseInventorySHA256 = ""
			input.FailureDomain = terminal.domain
			input.FailureStage = terminal.stage
			input.FailureCode = terminal.code
			input.EvidenceBytes, input.EvidenceSHA256, input.EvidencePath = calphadTestEvidence(
				"terminal-"+terminal.status, input,
			)
			input.EvidenceSizeBytes = int64(len(input.EvidenceBytes))
			input.CreatedAt = now.Add(time.Duration(-3+index) * time.Second)
			appended, appendErr := ledger.AppendCalphadValidation(ctx, input)
			if appendErr != nil || appended.Status != terminal.status || appended.Promotable ||
				appended.EvidenceRetention != domain.CalphadEvidenceRetentionRetained {
				t.Fatalf("append terminal %s=%+v err=%v", terminal.status, appended, appendErr)
			}
			retried, retryErr := ledger.AppendCalphadValidation(ctx, input)
			if retryErr != nil || retried.ValidationID != appended.ValidationID {
				t.Fatalf("retry terminal %s=%+v err=%v", terminal.status, retried, retryErr)
			}
			replayed, replayErr := ledger.GetCalphadValidationEvidenceForOwner(
				ctx, childID, appended.ValidationID, owner, org,
			)
			if replayErr != nil || !bytes.Equal(replayed.Bytes, input.EvidenceBytes) {
				t.Fatalf("replay terminal %s=%+v err=%v", terminal.status, replayed, replayErr)
			}
		}
	})

	// More than five full pages share one timestamp. This catches both offset
	// drift and implementations that key only on created_at instead of the
	// deterministic (created_at, validation_id) tuple.
	pageTimestamp := now.Add(-time.Second)
	var firstFailure domain.CalphadValidationRecord
	var firstFailureInput domain.AppendCalphadValidationInput
	for index := 0; index < 501; index++ {
		failed := inspectionInput
		failed.Status = "failed"
		failed.DatabaseInventorySHA256 = ""
		failed.FailureDomain = domain.CalphadFailureDomainInput
		failed.FailureStage = domain.CalphadFailureStageParse
		failed.FailureCode = domain.CalphadFailureCodeParseFailed
		failed.EvidenceBytes, failed.EvidenceSHA256, failed.EvidencePath = calphadTestEvidence(
			fmt.Sprintf("failed-page-%d", index), failed,
		)
		failed.EvidenceSizeBytes = int64(len(failed.EvidenceBytes))
		failed.CreatedAt = pageTimestamp
		failed.Metadata = domain.JSONMap{"adversarial_page_index": index}
		appended, appendErr := ledger.AppendCalphadValidation(ctx, failed)
		if appendErr != nil {
			t.Fatalf("AppendCalphadValidation(adversarial page %d): %v", index, appendErr)
		}
		if appended.Promotable || appended.EvidenceRetention != domain.CalphadEvidenceRetentionRetained {
			t.Fatalf("failure page %d retention/promotability = %+v", index, appended)
		}
		if index == 0 {
			firstFailure = appended
			firstFailureInput = failed
		}
	}
	retriedFailure, err := ledger.AppendCalphadValidation(ctx, firstFailureInput)
	if err != nil || retriedFailure.ValidationID != firstFailure.ValidationID {
		t.Fatalf("idempotent failure retry=%+v err=%v", retriedFailure, err)
	}
	mismatchedTuple := firstFailureInput
	mismatchedTuple.Status = "timeout"
	if _, err := ledger.AppendCalphadValidation(ctx, mismatchedTuple); !errors.Is(err, ErrConflict) {
		t.Fatalf("cross-field mismatched failure tuple err=%v, want ErrConflict", err)
	}
	missingFailureEvidence := firstFailureInput
	missingFailureEvidence.EvidencePath = ""
	missingFailureEvidence.EvidenceSHA256 = ""
	missingFailureEvidence.EvidenceSizeBytes = 0
	missingFailureEvidence.EvidenceBytes = nil
	if _, err := ledger.AppendCalphadValidation(ctx, missingFailureEvidence); !errors.Is(
		err, ErrCalphadEvidenceRetentionRequired,
	) {
		t.Fatalf("failure without retained evidence err=%v, want retention error", err)
	}
	replayedFailure, err := ledger.GetCalphadValidationEvidenceForOwner(
		ctx, childID, firstFailure.ValidationID, owner, org,
	)
	if err != nil || replayedFailure.SHA256 != firstFailure.EvidenceSHA256 {
		t.Fatalf("failure evidence replay=%+v err=%v", replayedFailure, err)
	}
	unbounded, err := ledger.GetCalphadLedgerForOwner(ctx, childID, owner, org)
	if err != nil || len(unbounded.Validations) != 510 || unbounded.LatestValidation == nil {
		t.Fatalf("unbounded ledger validations=%d err=%v, want 510", len(unbounded.Validations), err)
	}
	expectedLatestValidationID := unbounded.LatestValidation.ValidationID
	expectedLatestStatus := unbounded.LatestValidation.Status
	wantOrder := make([]string, len(unbounded.Validations))
	for index := range unbounded.Validations {
		wantOrder[index] = unbounded.Validations[index].ValidationID
	}
	gotOrder := make([]string, 0, len(wantOrder))
	pageInput := domain.GetCalphadLedgerPageInput{
		ResourceID: childID, OwnerUserID: owner, OwnerOrgID: org, Limit: 100,
	}
	for pageNumber := 0; ; pageNumber++ {
		page, pageErr := ledger.GetCalphadLedgerPageForOwner(ctx, pageInput)
		if pageErr != nil {
			t.Fatalf("GetCalphadLedgerPageForOwner(page %d): %v", pageNumber, pageErr)
		}
		if page.LatestValidation == nil || page.LatestValidation.ValidationID != expectedLatestValidationID {
			t.Fatalf("page %d global latest=%+v, want %s", pageNumber, page.LatestValidation, expectedLatestValidationID)
		}
		for _, event := range page.Validations {
			gotOrder = append(gotOrder, event.ValidationID)
		}
		if !page.HasMore {
			if page.NextValidationID != "" || !page.NextCreatedAt.IsZero() {
				t.Fatalf("terminal page exposed continuation anchor: %+v", page)
			}
			break
		}
		if len(page.Validations) != 100 || page.NextValidationID == "" || page.NextCreatedAt.IsZero() {
			t.Fatalf("page %d continuation state invalid: %+v", pageNumber, page)
		}
		pageInput.ExpectedRevisionID = page.Revision.RevisionID
		pageInput.BeforeCreatedAt = page.NextCreatedAt
		pageInput.BeforeValidationID = page.NextValidationID
	}
	if len(gotOrder) != len(wantOrder) {
		t.Fatalf("paged validation count=%d, want %d", len(gotOrder), len(wantOrder))
	}
	for index := range wantOrder {
		if gotOrder[index] != wantOrder[index] {
			t.Fatalf("paged order[%d]=%q, want %q", index, gotOrder[index], wantOrder[index])
		}
	}
	invalidPage := pageInput
	invalidPage.BeforeCreatedAt = pageTimestamp
	invalidPage.BeforeValidationID = "calphad-validation-missing-anchor"
	if _, err := ledger.GetCalphadLedgerPageForOwner(ctx, invalidPage); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing cursor anchor err=%v, want ErrNotFound", err)
	}
	invalidPage = domain.GetCalphadLedgerPageInput{
		ResourceID: childID, OwnerUserID: "other-user", OwnerOrgID: org, Limit: 100,
		ExpectedRevisionID: child.RevisionID, BeforeCreatedAt: pageTimestamp,
		BeforeValidationID: unbounded.Validations[len(unbounded.Validations)-1].ValidationID,
	}
	if _, err := ledger.GetCalphadLedgerPageForOwner(ctx, invalidPage); !errors.Is(err, ErrNotFound) {
		t.Fatalf("cross-owner cursor page err=%v, want ErrNotFound", err)
	}
	invalidPage.ResourceID = parentID
	invalidPage.OwnerUserID = owner
	if _, err := ledger.GetCalphadLedgerPageForOwner(ctx, invalidPage); !errors.Is(err, ErrNotFound) {
		t.Fatalf("cross-resource cursor page err=%v, want ErrNotFound", err)
	}
	if _, err := ledger.GetCalphadLedgerPageForOwner(ctx, domain.GetCalphadLedgerPageInput{
		ResourceID: childID, OwnerUserID: owner, OwnerOrgID: org, Limit: 501,
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("oversized page err=%v, want ErrConflict", err)
	}

	// An owner PATCH can claim any status in generic metadata, but cannot alter the ledger.
	if _, err := ledger.MergeResourceMetadataForUser(ctx, domain.MergeResourceMetadataInput{
		ResourceID: childID, UserID: owner, OrgID: org,
		Patch: domain.JSONMap{"calphad": domain.JSONMap{
			"validation_status": "forged_failed",
			"source":            "https://example.org/assessments/owner-relabelled",
		}},
		UpdatedAt: now.Add(3 * time.Second),
	}); err != nil {
		t.Fatalf("MergeResourceMetadataForUser: %v", err)
	}
	if _, err := ledger.CreateCalphadRevision(ctx, domain.CreateCalphadRevisionInput{
		ResourceID: childID, OwnerUserID: owner, OwnerOrgID: org,
		AssessmentPressureLimitsPa: calphadTestPressureLimits, InputBytes: childBytes,
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("idempotent revision accepted owner-provenance drift err=%v, want conflict", err)
	}
	if _, err := ledger.AppendCalphadValidation(ctx, inspectionInput); !errors.Is(err, ErrConflict) {
		t.Fatalf("validation retry accepted owner-provenance drift err=%v, want conflict", err)
	}
	got, err := ledger.GetCalphadLedgerForOwner(ctx, childID, owner, org)
	if err != nil {
		t.Fatalf("GetCalphadLedgerForOwner(final): %v", err)
	}
	if got.LatestValidation == nil || got.LatestValidation.ValidationID != expectedLatestValidationID ||
		got.LatestValidation.Status != expectedLatestStatus {
		t.Fatalf("latest ledger validation = %+v", got.LatestValidation)
	}
	retainedDeclaration, ok := calphadRevisionOwnerDeclaration(got.Revision)
	if !ok || retainedDeclaration.Source != "https://example.org/assessments/"+childID ||
		retainedDeclaration.Authority != "resource_owner" {
		t.Fatalf("immutable owner declaration drifted with generic catalog metadata: %+v ok=%t", retainedDeclaration, ok)
	}
	if managed, ok := got.LatestValidation.Metadata["server_managed"].(bool); !ok || !managed ||
		got.LatestValidation.Metadata["revision_id"] != child.RevisionID {
		t.Fatalf("validation server-owned metadata drifted: %+v", got.LatestValidation.Metadata)
	}
	if _, retained := got.LatestValidation.Metadata["nested"]; retained {
		t.Fatal("caller-supplied validation metadata was persisted")
	}
	got.LatestValidation.Metadata["server_managed"] = false
	again, err := ledger.GetCalphadLedgerForOwner(ctx, childID, owner, org)
	if err != nil {
		t.Fatalf("GetCalphadLedgerForOwner(after returned mutation): %v", err)
	}
	if managed, ok := again.LatestValidation.Metadata["server_managed"].(bool); !ok || !managed {
		t.Fatalf("ledger metadata return was not cloned: %+v", again.LatestValidation.Metadata)
	}
	if _, err := ledger.GetCalphadLedgerForOwner(ctx, childID, "other-user", org); !errors.Is(err, ErrNotFound) {
		t.Fatalf("cross-tenant ledger read err=%v, want ErrNotFound", err)
	}
	if _, err := ledger.AppendCalphadValidation(ctx, domain.AppendCalphadValidationInput{
		ResourceID: childID, OwnerUserID: owner, OwnerOrgID: org,
		DatabaseSHA256: childSHA, DatabaseSizeBytes: 200,
		AssessmentPressureLimitsPa: calphadTestPressureLimits,
		Status:                     "equilibrium_completed", Operation: "equilibrium",
		EvidencePath: "/outputs/calphad/equilibrium/x.json", EvidenceSHA256: "invalid",
		EvidenceSizeBytes: 1, RuntimeImageID: "mutable:tag", PycalphadVersion: "0.11.2",
		RunID: "run", CreatedByAuthority: "trusted_worker",
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("invalid validation err=%v, want ErrConflict", err)
	}

	// Reusing a resource id for different bytes cannot silently rewrite its revision.
	changedBytes, _ := calphadTestInput("CHANGED-TDB", 201)
	resource(childID, changedBytes)
	if _, err := ledger.CreateCalphadRevision(ctx, domain.CreateCalphadRevisionInput{
		ResourceID: childID, OwnerUserID: owner, OwnerOrgID: org,
		AssessmentPressureLimitsPa: calphadTestPressureLimits, InputBytes: changedBytes,
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("changed resource revision err=%v, want ErrConflict", err)
	}
	if _, err := ledger.AppendCalphadValidation(ctx, domain.AppendCalphadValidationInput{
		ResourceID: childID, OwnerUserID: owner, OwnerOrgID: org,
		DatabaseSHA256: childSHA, DatabaseSizeBytes: 200,
		AssessmentPressureLimitsPa: calphadTestPressureLimits,
		Status:                     "input_validated", Operation: "inspect",
		EvidencePath:   "/outputs/calphad/inspection/" + strings.Repeat("1", 64) + ".json",
		EvidenceSHA256: strings.Repeat("1", 64), EvidenceSizeBytes: 1,
		RuntimeImageID: "sha256:" + strings.Repeat("2", 64), PycalphadVersion: "0.11.2",
		RunID: "run-after-mutation", CreatedByAuthority: "trusted_worker",
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("changed resource validation err=%v, want ErrConflict", err)
	}
	// The immutable audit record remains readable from its tenant snapshot.
	retained, err := ledger.GetCalphadLedgerForOwner(ctx, childID, owner, org)
	if err != nil || retained.Revision.SHA256 != childSHA || retained.Revision.SHA256 == parentSHA {
		t.Fatalf("retained immutable ledger=%+v err=%v", retained, err)
	}
	replayedEvidence, err := ledger.GetCalphadValidationEvidenceForOwner(
		ctx, childID, second.ValidationID, owner, org,
	)
	if err != nil || !bytes.Equal(replayedEvidence.Bytes, secondBytes) {
		t.Fatalf("retained evidence after source mutation=%q err=%v", replayedEvidence.Bytes, err)
	}
}

func TestMemoryStoreCalphadLedgerIsAppendOnlyTenantScopedAndContentBound(t *testing.T) {
	memory := NewMemoryStore()
	exerciseCalphadLedger(t, memory, "memory")
	_, childSHA := calphadTestInput("CHILD-TDB", 200)
	payload, sha, _ := calphadTestEvidence("equilibrium-2 Δ café", domain.AppendCalphadValidationInput{
		ResourceID: "calphad-child-memory", DatabaseSHA256: childSHA, DatabaseSizeBytes: 200,
		DatabaseFormat: domain.CalphadDatabaseFormatTDB,
		OwnerDeclaration: calphadTestOwnerDeclaration(
			"calphad-child-memory", domain.CalphadDatabaseFormatTDB,
		),
		AssessmentPressureLimitsPa: calphadTestPressureLimits,
		Status:                     "equilibrium_completed", Operation: "equilibrium",
		RuntimeImageID: "sha256:" + strings.Repeat("d", 64),
	})
	blob, found := memory.calphadEvidenceBlobs[sha]
	blobDigest := sha256.Sum256(blob.Payload)
	if len(memory.calphadEvidenceBlobs) != 509 || !found || blob.SHA256 != sha ||
		blob.SizeBytes != int64(len(payload)) ||
		hex.EncodeToString(blobDigest[:]) != sha {
		t.Fatalf("retained evidence blob=%+v found=%t", blob, found)
	}
	for index := range memory.calphadValidations {
		if memory.calphadValidations[index].EvidenceSHA256 == sha {
			memory.calphadValidations[index].EvidenceContractVersion = ""
		}
	}
	legacyMarker, err := memory.GetCalphadLedgerForOwner(
		context.Background(), "calphad-child-memory", "calphad-owner-memory", "calphad-org-memory",
	)
	if err != nil || legacyMarker.LatestValidation == nil ||
		legacyMarker.LatestValidation.EvidenceRetention != domain.CalphadEvidenceRetentionLegacyUnretained ||
		legacyMarker.LatestValidation.Promotable {
		t.Fatalf("historical row without evidence-contract marker was promotable: %+v err=%v", legacyMarker.LatestValidation, err)
	}
	for index := range memory.calphadValidations {
		if memory.calphadValidations[index].EvidenceSHA256 == sha {
			memory.calphadValidations[index].EvidenceContractVersion = domain.CalphadEvidenceContractVersion
		}
	}
	inspectionSHA := legacyMarker.LatestValidation.InspectionEvidenceSHA256
	inspectionBlob, inspectionFound := memory.calphadEvidenceBlobs[inspectionSHA]
	if !inspectionFound {
		t.Fatalf("referenced inspection blob %q was not retained", inspectionSHA)
	}
	delete(memory.calphadEvidenceBlobs, inspectionSHA)
	missingLineage, err := memory.GetCalphadLedgerForOwner(
		context.Background(), "calphad-child-memory", "calphad-owner-memory", "calphad-org-memory",
	)
	if err != nil || missingLineage.LatestValidation == nil ||
		missingLineage.LatestValidation.EvidenceRetention != domain.CalphadEvidenceRetentionLegacyUnretained ||
		missingLineage.LatestValidation.Promotable {
		t.Fatalf("equilibrium with missing inspection lineage was promotable: %+v err=%v",
			missingLineage.LatestValidation, err)
	}
	if _, err := memory.GetCalphadValidationEvidenceForOwner(
		context.Background(), "calphad-child-memory", legacyMarker.LatestValidation.ValidationID,
		"calphad-owner-memory", "calphad-org-memory",
	); !errors.Is(err, ErrCalphadEvidenceRetentionRequired) {
		t.Fatalf("equilibrium evidence read with missing inspection lineage err=%v, want retention error", err)
	}
	tamperedInspection := inspectionBlob
	tamperedInspection.Payload = append([]byte(nil), inspectionBlob.Payload...)
	tamperedInspection.Payload[0] ^= 0xff
	memory.calphadEvidenceBlobs[inspectionSHA] = tamperedInspection
	if _, err := memory.GetCalphadValidationEvidenceForOwner(
		context.Background(), "calphad-child-memory", legacyMarker.LatestValidation.ValidationID,
		"calphad-owner-memory", "calphad-org-memory",
	); !errors.Is(err, ErrCalphadEvidenceRetentionRequired) {
		t.Fatalf("equilibrium evidence read with corrupt inspection lineage err=%v, want retention error", err)
	}
	memory.calphadEvidenceBlobs[inspectionSHA] = inspectionBlob
	tamperedBlob := blob
	tamperedBlob.Payload = append([]byte(nil), blob.Payload...)
	tamperedBlob.Payload[0] ^= 0xff
	memory.calphadEvidenceBlobs[sha] = tamperedBlob
	if _, err := memory.GetCalphadValidationEvidenceForOwner(
		context.Background(), "calphad-child-memory", legacyMarker.LatestValidation.ValidationID,
		"calphad-owner-memory", "calphad-org-memory",
	); !errors.Is(err, ErrCalphadEvidenceRetentionRequired) {
		t.Fatalf("corrupt exact evidence replay err=%v, want ErrCalphadEvidenceRetentionRequired", err)
	}
	memory.calphadEvidenceBlobs[sha] = blob
	delete(memory.calphadEvidenceBlobs, sha)
	if _, err := memory.GetCalphadValidationEvidenceForOwner(
		context.Background(), "calphad-child-memory", legacyMarker.LatestValidation.ValidationID,
		"calphad-owner-memory", "calphad-org-memory",
	); !errors.Is(err, ErrCalphadEvidenceRetentionRequired) {
		t.Fatalf("missing exact evidence replay err=%v, want ErrCalphadEvidenceRetentionRequired", err)
	}
	legacy, err := memory.GetCalphadLedgerForOwner(
		context.Background(), "calphad-child-memory", "calphad-owner-memory", "calphad-org-memory",
	)
	if err != nil {
		t.Fatalf("GetCalphadLedgerForOwner(legacy fixture): %v", err)
	}
	if legacy.LatestValidation == nil ||
		legacy.LatestValidation.EvidenceRetention != domain.CalphadEvidenceRetentionLegacyUnretained ||
		legacy.LatestValidation.Promotable {
		t.Fatalf("legacy unretained event was presented as promotable: %+v", legacy.LatestValidation)
	}
}

func TestMemoryStoreCalphadInputRetentionFailsClosedAndSurvivesCatalogGC(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	memory := NewMemoryStore()
	inputBytes, inputSHA := calphadTestInput("EXACT-TDB-INPUT", 4096)
	const resourceID = "calphad-retained-input"
	const owner = "calphad-input-owner"
	const org = "calphad-input-org"
	if _, err := memory.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID: resourceID, OriginalName: "exact.tdb", ContentType: "application/x-thermocalc-tdb",
		SizeBytes: int64(len(inputBytes)), SHA256: inputSHA, OwnerUserID: owner, OwnerOrgID: org,
		Status: "active", CreatedAt: domain.Now(), UpdatedAt: domain.Now(),
		Metadata: calphadTestOwnerMetadata(resourceID),
	}); err != nil {
		t.Fatalf("UpsertResource: %v", err)
	}
	for name, limits := range map[string][2]float64{
		"missing":      {},
		"nan":          {math.NaN(), domain.CalphadReferencePressurePa},
		"reversed":     {101326, 101325},
		"below global": {0, 101325},
		"above global": {101325, domain.CalphadMaximumPressurePa + 1},
	} {
		if _, err := memory.CreateCalphadRevision(ctx, domain.CreateCalphadRevisionInput{
			ResourceID: resourceID, OwnerUserID: owner, OwnerOrgID: org,
			AssessmentPressureLimitsPa: limits, InputBytes: inputBytes,
		}); !errors.Is(err, ErrCalphadPressureLimitsInvalid) {
			t.Fatalf("%s pressure declaration err=%v, want ErrCalphadPressureLimitsInvalid", name, err)
		}
	}
	if _, err := memory.CreateCalphadRevision(ctx, domain.CreateCalphadRevisionInput{
		ResourceID: resourceID, OwnerUserID: owner, OwnerOrgID: org,
		AssessmentPressureLimitsPa: calphadTestPressureLimits,
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("missing exact input bytes err=%v, want ErrConflict", err)
	}
	wrongBytes := append([]byte(nil), inputBytes...)
	wrongBytes[0] ^= 0xff
	if _, err := memory.CreateCalphadRevision(ctx, domain.CreateCalphadRevisionInput{
		ResourceID: resourceID, OwnerUserID: owner, OwnerOrgID: org,
		AssessmentPressureLimitsPa: calphadTestPressureLimits, InputBytes: wrongBytes,
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("input SHA mismatch err=%v, want ErrConflict", err)
	}
	revision, err := memory.CreateCalphadRevision(ctx, domain.CreateCalphadRevisionInput{
		ResourceID: resourceID, OwnerUserID: owner, OwnerOrgID: org,
		AssessmentPressureLimitsPa: calphadTestPressureLimits, InputBytes: inputBytes,
	})
	if err != nil {
		t.Fatalf("CreateCalphadRevision: %v", err)
	}
	if _, err := memory.GetCalphadRevisionInputForOwner(ctx, resourceID, "other-owner", org); !errors.Is(err, ErrNotFound) {
		t.Fatalf("cross-owner input read err=%v, want ErrNotFound", err)
	}

	originalBlob := memory.calphadInputBlobs[inputSHA]
	tamperedBlob := originalBlob
	tamperedBlob.Payload = append([]byte(nil), originalBlob.Payload...)
	tamperedBlob.Payload[0] ^= 0xff
	memory.calphadInputBlobs[inputSHA] = tamperedBlob
	if _, err := memory.GetCalphadRevisionInputForOwner(ctx, resourceID, owner, org); !errors.Is(err, ErrCalphadInputRetentionRequired) {
		t.Fatalf("tampered retained input err=%v, want ErrCalphadInputRetentionRequired", err)
	}
	if _, err := memory.GetCalphadLedgerForOwner(ctx, resourceID, owner, org); !errors.Is(err, ErrCalphadInputRetentionRequired) {
		t.Fatalf("ledger with tampered input err=%v, want fail closed", err)
	}
	memory.calphadInputBlobs[inputSHA] = originalBlob
	originalRevision := memory.calphadRevisions[resourceID]
	tamperedRevision := cloneCalphadRevision(originalRevision)
	tamperedRevision.Metadata[domain.CalphadAssessmentPressureLimitsMetadataKey] = []float64{100000, 200000}
	memory.calphadRevisions[resourceID] = tamperedRevision
	if _, err := memory.GetCalphadLedgerForOwner(ctx, resourceID, owner, org); !errors.Is(err, ErrCalphadPressureLimitsInvalid) {
		t.Fatalf("tampered revision pressure metadata err=%v, want fail closed", err)
	}
	memory.calphadRevisions[resourceID] = originalRevision

	if err := memory.PurgeResource(ctx, resourceID); err != nil {
		t.Fatalf("PurgeResource: %v", err)
	}
	replayed, err := memory.GetCalphadRevisionInputForOwner(ctx, resourceID, owner, org)
	if err != nil || replayed.RevisionID != revision.RevisionID || !bytes.Equal(replayed.Bytes, inputBytes) {
		t.Fatalf("retained input after catalog GC=%+v err=%v", replayed, err)
	}
	if _, err := memory.GetCalphadLedgerForOwner(ctx, resourceID, owner, org); err != nil {
		t.Fatalf("immutable ledger after catalog GC: %v", err)
	}

	delete(memory.calphadInputBlobs, inputSHA)
	if _, err := memory.GetCalphadRevisionInputForOwner(ctx, resourceID, owner, org); !errors.Is(err, ErrCalphadInputRetentionRequired) {
		t.Fatalf("missing retained input err=%v, want ErrCalphadInputRetentionRequired", err)
	}
	if _, err := memory.GetCalphadLedgerForOwner(ctx, resourceID, owner, org); !errors.Is(err, ErrCalphadInputRetentionRequired) {
		t.Fatalf("ledger with missing input err=%v, want fail closed", err)
	}
}

func TestCalphadTriggerRaceErrorsMapToStableAPIClasses(t *testing.T) {
	t.Parallel()
	leaseRace := mapCalphadAppendError(&pgconn.PgError{
		Code: "28000", Message: "CALPHAD_RUN_LEASE_INVALID: lease expired at insert",
	})
	if !errors.Is(leaseRace, ErrCalphadRunLeaseInvalid) {
		t.Fatalf("lease trigger race=%v, want ErrCalphadRunLeaseInvalid", leaseRace)
	}
	runtimePolicy := mapCalphadAppendError(&pgconn.PgError{
		Code: "28000", Message: "CALPHAD_RUNTIME_POLICY_INVALID: mismatch",
	})
	if !errors.Is(runtimePolicy, ErrCalphadRuntimePolicyInvalid) ||
		errors.Is(runtimePolicy, ErrCalphadRunLeaseInvalid) || !errors.Is(runtimePolicy, ErrConflict) {
		t.Fatalf("runtime trigger=%v, want runtime policy conflict", runtimePolicy)
	}
	inspection := mapCalphadAppendError(&pgconn.PgError{
		Code: "23514", Message: "CALPHAD_INSPECTION_REQUIRED: missing",
	})
	if !errors.Is(inspection, ErrCalphadInspectionRequired) || !errors.Is(inspection, ErrConflict) {
		t.Fatalf("inspection trigger=%v, want inspection conflict", inspection)
	}
	inputRetention := mapCalphadAppendError(&pgconn.PgError{
		Code: "23514", Message: "CALPHAD_INPUT_RETENTION_REQUIRED: missing",
	})
	if !errors.Is(inputRetention, ErrCalphadInputRetentionRequired) || !errors.Is(inputRetention, ErrConflict) {
		t.Fatalf("input retention trigger=%v, want fail-closed conflict", inputRetention)
	}
}

func TestCalphadRuntimePolicyRequiresExactServerSandboxV2(t *testing.T) {
	t.Parallel()
	runtimeImage := "sha256:" + strings.Repeat("a", 64)
	validPolicy := domain.JSONMap{
		"schema_version": domain.CalphadRuntimePolicySchema, "authority": "control_plane",
		"runtime_image_id": runtimeImage, "pycalphad_version": domain.CalphadPycalphadVersion,
		"network": domain.CalphadRuntimeNetwork, "no_new_privileges": true,
		"read_only_root_filesystem": true, "cap_drop_all": true,
		"cpus_at_most":         float64(domain.CalphadRuntimeCPUsAtMost),
		"memory_bytes_at_most": float64(domain.CalphadRuntimeMemoryBytesAtMost),
		"pids_at_most":         float64(domain.CalphadRuntimePIDsAtMost),
	}
	if image, version, ok := calphadRuntimePolicy(domain.JSONMap{
		domain.CalphadRuntimePolicyMetadataKey: validPolicy,
	}); !ok || image != runtimeImage || version != domain.CalphadPycalphadVersion {
		t.Fatalf("valid v2 sandbox policy rejected: image=%q version=%q ok=%t", image, version, ok)
	}
	for _, test := range []struct {
		name   string
		mutate func(domain.JSONMap)
	}{
		{name: "missing isolation key", mutate: func(policy domain.JSONMap) { delete(policy, "network") }},
		{name: "network enabled", mutate: func(policy domain.JSONMap) { policy["network"] = "bridge" }},
		{name: "privilege escalation enabled", mutate: func(policy domain.JSONMap) { policy["no_new_privileges"] = false }},
		{name: "writable root", mutate: func(policy domain.JSONMap) { policy["read_only_root_filesystem"] = false }},
		{name: "capabilities retained", mutate: func(policy domain.JSONMap) { policy["cap_drop_all"] = false }},
		{name: "cpu bound widened", mutate: func(policy domain.JSONMap) { policy["cpus_at_most"] = 9 }},
		{name: "memory bound widened", mutate: func(policy domain.JSONMap) { policy["memory_bytes_at_most"] = int64(34359738369) }},
		{name: "pid bound widened", mutate: func(policy domain.JSONMap) { policy["pids_at_most"] = 4097 }},
		{name: "extra caller field", mutate: func(policy domain.JSONMap) { policy["caller_override"] = true }},
	} {
		t.Run(test.name, func(t *testing.T) {
			policy := deepCloneCalphadJSONMap(validPolicy)
			test.mutate(policy)
			if _, _, ok := calphadRuntimePolicy(domain.JSONMap{
				domain.CalphadRuntimePolicyMetadataKey: policy,
			}); ok {
				t.Fatalf("unsafe policy accepted: %#v", policy)
			}
		})
	}
}

func TestPostgresStoreCalphadLedgerIsAppendOnlyTenantScopedAndContentBound(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	migrationDSN := os.Getenv("ULTRA_CONTROL_TEST_MIGRATION_DATABASE_URL")
	if migrationDSN == "" {
		t.Skip("ULTRA_CONTROL_TEST_MIGRATION_DATABASE_URL is required for role-separated qualification")
	}
	ctx := context.Background()
	migrationPool, err := pgxpool.New(ctx, migrationDSN)
	if err != nil {
		t.Fatalf("pgxpool.New(migration): %v", err)
	}
	defer migrationPool.Close()
	if err := ApplyPostgresSchema(ctx, migrationPool); err != nil {
		t.Fatalf("ApplyPostgresSchema: %v", err)
	}
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New(serving): %v", err)
	}
	defer pool.Close()
	var servingRole string
	if err := pool.QueryRow(ctx, `SELECT current_user`).Scan(&servingRole); err != nil {
		t.Fatalf("load serving role: %v", err)
	}
	// Prove privilege normalization removes grants that a table-level UPDATE
	// check alone cannot see.
	roleSQL := pgx.Identifier{servingRole}.Sanitize()
	if _, err := migrationPool.Exec(ctx,
		"GRANT UPDATE (metadata) ON TABLE control_calphad_revisions TO "+roleSQL,
	); err != nil {
		t.Fatalf("seed column-level CALPHAD mutation privilege: %v", err)
	}
	if err := GrantPostgresServingPrivileges(ctx, migrationPool, servingRole); err != nil {
		t.Fatalf("GrantPostgresServingPrivileges: %v", err)
	}
	roleStatus, err := InspectCalphadServingRole(ctx, pool)
	if err != nil {
		t.Fatalf("InspectCalphadServingRole: %v", err)
	}
	if err := VerifyCalphadServingRole(ctx, pool); err != nil {
		t.Fatalf("VerifyCalphadServingRole: %v", err)
	}
	connectionTarget, err := url.Parse(dsn)
	if err != nil {
		t.Fatalf("parse serving qualification DSN: %v", err)
	}
	connectionTargetPort := 5432
	if portText := connectionTarget.Port(); portText != "" {
		connectionTargetPort, err = strconv.Atoi(portText)
		if err != nil {
			t.Fatalf("parse serving qualification port: %v", err)
		}
	}
	var connectedDatabase, serverAddress, databaseRole, transactionReadOnly string
	var serverPort int
	if err := pool.QueryRow(ctx, `
SELECT current_database(), COALESCE(inet_server_addr()::text, 'local'),
       COALESCE(inet_server_port(), 0), current_user,
       current_setting('transaction_read_only')`).Scan(
		&connectedDatabase, &serverAddress, &serverPort, &databaseRole, &transactionReadOnly,
	); err != nil {
		t.Fatalf("load PostgreSQL qualification identity: %v", err)
	}
	identity, err := json.Marshal(map[string]any{
		"database": connectedDatabase, "server_address": serverAddress, "server_port": serverPort,
		"role": databaseRole, "transaction_read_only": transactionReadOnly,
		"role_superuser":                           roleStatus.Superuser,
		"role_create_role":                         roleStatus.CreateRole,
		"role_create_database":                     roleStatus.CreateDB,
		"role_replication":                         roleStatus.Replication,
		"role_bypass_rls":                          roleStatus.BypassRLS,
		"calphad_owned_tables":                     roleStatus.OwnedTables,
		"calphad_owned_functions":                  roleStatus.OwnedFunctions,
		"calphad_owner_roles":                      roleStatus.OwnerRoles,
		"calphad_reachable_roles":                  roleStatus.ReachableRoles,
		"calphad_owner_role_reachable":             roleStatus.OwnerRoleReachable,
		"public_schema_owner":                      roleStatus.PublicSchemaOwner,
		"public_owner_role_reachable":              roleStatus.PublicOwnerReachable,
		"can_create_public_schema":                 roleStatus.CanCreatePublicSchema,
		"calphad_select_all":                       roleStatus.CanSelectAll,
		"calphad_insert_all":                       roleStatus.CanInsertAll,
		"calphad_insert_any":                       roleStatus.CanInsertAny,
		"calphad_mutation_privilege":               roleStatus.CanMutateCalphad,
		"calphad_execute_create_revision":          roleStatus.CanExecuteCreateRevision,
		"calphad_execute_append_validation":        roleStatus.CanExecuteAppendValidation,
		"calphad_writer_functions_exact":           roleStatus.WriterFunctionsExact,
		"calphad_execute_unexpected_writer":        roleStatus.CanExecuteUnexpectedWriter,
		"calphad_execute_internal":                 roleStatus.CanExecuteInternal,
		"calphad_public_execute":                   roleStatus.PublicCanExecute,
		"calphad_unexpected_table_acl_grantees":    roleStatus.UnexpectedTableACLGrantees,
		"calphad_unexpected_function_acl_grantees": roleStatus.UnexpectedFuncACLGrantees,
		"connection_target_host":                   connectionTarget.Hostname(),
		"connection_target_port":                   connectionTargetPort,
	})
	if err != nil {
		t.Fatalf("marshal PostgreSQL qualification identity: %v", err)
	}
	t.Logf("CALPHAD_POSTGRES_IDENTITY %s", identity)
	if transactionReadOnly != "off" {
		t.Fatalf("qualification database is read-only: %q", transactionReadOnly)
	}
	t.Run("serving_role_separated", func(t *testing.T) {
		if roleStatus.Superuser || roleStatus.CreateRole || roleStatus.CreateDB ||
			roleStatus.Replication || roleStatus.BypassRLS || len(roleStatus.ReachableRoles) != 0 ||
			len(roleStatus.OwnedTables) != 0 || len(roleStatus.OwnedFunctions) != 0 ||
			len(roleStatus.OwnerRoles) == 0 || roleStatus.OwnerRoleReachable ||
			roleStatus.PublicSchemaOwner == "" || roleStatus.PublicOwnerReachable ||
			roleStatus.CanCreatePublicSchema || !roleStatus.CanSelectAll ||
			roleStatus.CanInsertAll || roleStatus.CanInsertAny || roleStatus.CanMutateCalphad ||
			!roleStatus.CanExecuteCreateRevision || !roleStatus.CanExecuteAppendValidation ||
			!roleStatus.WriterFunctionsExact || roleStatus.CanExecuteUnexpectedWriter ||
			roleStatus.CanExecuteInternal || roleStatus.PublicCanExecute ||
			len(roleStatus.UnexpectedTableACLGrantees) != 0 ||
			len(roleStatus.UnexpectedFuncACLGrantees) != 0 {
			t.Fatalf("unsafe CALPHAD serving role: %+v", roleStatus)
		}
	})
	t.Run("serving_role_subset_insert_rejected", func(t *testing.T) {
		if _, err := migrationPool.Exec(ctx,
			"GRANT INSERT ON TABLE control_calphad_tenant_capacity TO "+roleSQL,
		); err != nil {
			t.Fatalf("grant subset CALPHAD INSERT: %v", err)
		}
		unsafeStatus, inspectErr := InspectCalphadServingRole(ctx, pool)
		if inspectErr != nil {
			t.Fatalf("inspect subset INSERT role: %v", inspectErr)
		}
		if unsafeStatus.CanInsertAll || !unsafeStatus.CanInsertAny {
			t.Fatalf("subset INSERT audit all=%t any=%t, want false/true",
				unsafeStatus.CanInsertAll, unsafeStatus.CanInsertAny)
		}
		if verifyErr := VerifyCalphadServingRole(ctx, pool); verifyErr == nil {
			t.Fatal("serving-role verifier accepted INSERT on one CALPHAD table")
		}
		if err := GrantPostgresServingPrivileges(ctx, migrationPool, servingRole); err != nil {
			t.Fatalf("restore execute-only privileges after subset INSERT: %v", err)
		}
		if err := VerifyCalphadServingRole(ctx, pool); err != nil {
			t.Fatalf("verify restored serving role: %v", err)
		}
	})
	t.Run("public_and_unexpected_acl_grantees_rejected", func(t *testing.T) {
		if _, err := migrationPool.Exec(ctx,
			"GRANT SELECT ON TABLE control_calphad_revisions TO PUBLIC",
		); err != nil {
			t.Fatalf("grant PUBLIC CALPHAD SELECT: %v", err)
		}
		publicStatus, inspectErr := InspectCalphadServingRole(ctx, pool)
		if inspectErr != nil {
			t.Fatalf("inspect PUBLIC CALPHAD ACL: %v", inspectErr)
		}
		if !slices.Contains(publicStatus.UnexpectedTableACLGrantees, "PUBLIC") {
			t.Fatalf("PUBLIC CALPHAD ACL was not reported: %+v", publicStatus)
		}
		if verifyErr := VerifyCalphadServingRole(ctx, pool); verifyErr == nil {
			t.Fatal("serving-role verifier accepted a PUBLIC CALPHAD table grant")
		}
		if err := GrantPostgresServingPrivileges(ctx, migrationPool, servingRole); err != nil {
			t.Fatalf("restore privileges after PUBLIC CALPHAD grant: %v", err)
		}

		if _, err := migrationPool.Exec(ctx,
			"GRANT SELECT ON TABLE control_calphad_revisions TO pg_monitor",
		); err != nil {
			t.Fatalf("grant unexpected CALPHAD ACL: %v", err)
		}
		unexpectedStatus, inspectErr := InspectCalphadServingRole(ctx, pool)
		if inspectErr != nil {
			t.Fatalf("inspect unexpected ACL grantee: %v", inspectErr)
		}
		if !slices.Contains(unexpectedStatus.UnexpectedTableACLGrantees, "pg_monitor") {
			t.Fatalf("unexpected ACL grantee was not reported: %+v", unexpectedStatus)
		}
		if verifyErr := VerifyCalphadServingRole(ctx, pool); verifyErr == nil {
			t.Fatal("serving-role verifier accepted an unexpected CALPHAD ACL grantee")
		}
		if _, err := migrationPool.Exec(ctx,
			"REVOKE ALL ON TABLE control_calphad_revisions FROM pg_monitor",
		); err != nil {
			t.Fatalf("revoke unexpected CALPHAD ACL: %v", err)
		}
		if err := VerifyCalphadServingRole(ctx, pool); err != nil {
			t.Fatalf("verify restored serving role after ACL probes: %v", err)
		}
	})
	t.Run("unexpected_writer_overload_revoked_and_rejected", func(t *testing.T) {
		const obsoleteSignature = "public.ultra_create_calphad_revision_v1(text)"
		if _, err := migrationPool.Exec(ctx, `
CREATE FUNCTION public.ultra_create_calphad_revision_v1(text)
RETURNS text LANGUAGE sql IMMUTABLE AS $$ SELECT 'obsolete'::text $$`); err != nil {
			t.Fatalf("create obsolete writer overload: %v", err)
		}
		created := true
		defer func() {
			if created {
				_, _ = migrationPool.Exec(ctx, "DROP FUNCTION IF EXISTS "+obsoleteSignature)
				_ = GrantPostgresServingPrivileges(ctx, migrationPool, servingRole)
			}
		}()
		if err := VerifyCalphadServingRole(ctx, pool); err == nil {
			t.Fatal("serving-role verifier accepted an executable unexpected writer overload")
		}
		if err := VerifyPostgresSchema(ctx, migrationPool); err == nil {
			t.Fatal("schema verifier accepted an unexpected writer overload")
		}
		if err := GrantPostgresServingPrivileges(ctx, migrationPool, servingRole); err != nil {
			t.Fatalf("normalize overload privileges: %v", err)
		}
		overloadStatus, inspectErr := InspectCalphadServingRole(ctx, pool)
		if inspectErr != nil {
			t.Fatalf("inspect normalized overload privileges: %v", inspectErr)
		}
		if overloadStatus.WriterFunctionsExact || overloadStatus.CanExecuteUnexpectedWriter ||
			overloadStatus.PublicCanExecute {
			t.Fatalf("unexpected overload was not revoked and rejected: %+v", overloadStatus)
		}
		if _, err := migrationPool.Exec(ctx, "DROP FUNCTION "+obsoleteSignature); err != nil {
			t.Fatalf("drop obsolete writer overload: %v", err)
		}
		created = false
		if err := VerifyPostgresSchema(ctx, migrationPool); err != nil {
			t.Fatalf("verify canonical writer catalog after overload removal: %v", err)
		}
		if err := VerifyCalphadServingRole(ctx, pool); err != nil {
			t.Fatalf("verify canonical writer ACL after overload removal: %v", err)
		}
	})
	t.Run("serving_raw_insert_denied", func(t *testing.T) {
		for _, table := range []string{
			"control_calphad_input_blobs", "control_calphad_revisions",
			"control_calphad_evidence_blobs", "control_calphad_validation_events",
			"control_calphad_tenant_capacity",
		} {
			_, insertErr := pool.Exec(ctx, "INSERT INTO "+table+" DEFAULT VALUES")
			var pgErr *pgconn.PgError
			if !errors.As(insertErr, &pgErr) || pgErr.Code != "42501" {
				t.Fatalf("serving raw INSERT into %s err=%v, want SQLSTATE 42501", table, insertErr)
			}
		}
	})
	suffix := strconv.FormatInt(time.Now().UnixNano(), 36)
	exerciseCalphadLedger(t, NewPostgresStore(pool), suffix)
	t.Run("append_only_revision_update", func(t *testing.T) {
		if _, err := migrationPool.Exec(ctx, `UPDATE control_calphad_revisions SET size_bytes=size_bytes WHERE owner_user_id=$1`, "calphad-owner-"+suffix); err == nil {
			t.Fatal("append-only trigger allowed revision UPDATE")
		}
	})
	var revisionID, resourceID, databaseSHA, databaseFormat, runID string
	var databaseSize int64
	var pressureMinimum, pressureMaximum float64
	if err := pool.QueryRow(ctx, `
SELECT revision_id, resource_id, sha256, size_bytes, database_format,
       assessment_pressure_min_pa, assessment_pressure_max_pa
FROM control_calphad_revisions
WHERE owner_user_id=$1 AND resource_id=$2`, "calphad-owner-"+suffix, "calphad-child-"+suffix).
		Scan(&revisionID, &resourceID, &databaseSHA, &databaseSize, &databaseFormat, &pressureMinimum, &pressureMaximum); err != nil {
		t.Fatalf("load CALPHAD revision for constraints: %v", err)
	}
	if pressureMinimum != domain.CalphadReferencePressurePa || pressureMaximum != domain.CalphadReferencePressurePa {
		t.Fatalf("immutable revision pressure limits=[%v,%v]", pressureMinimum, pressureMaximum)
	}
	if databaseFormat != domain.CalphadDatabaseFormatTDB {
		t.Fatalf("immutable revision database_format=%q, want tdb", databaseFormat)
	}
	t.Run("owner_declaration_required_and_typed", func(t *testing.T) {
		missingOwnerResourceID := "calphad-missing-owner-declaration-" + suffix
		if _, err := migrationPool.Exec(ctx, `
INSERT INTO control_calphad_revisions
 (revision_id, resource_id, owner_user_id, owner_org_id, sha256, size_bytes, database_format,
  assessment_pressure_min_pa, assessment_pressure_max_pa, created_at, metadata)
VALUES ($1,$2,$3,$4,$5,$6,'tdb',101325,101325,now(),
        '{"assessment_pressure_limits_Pa":[101325,101325]}'::jsonb)`,
			"calphad-missing-owner-revision-"+suffix, missingOwnerResourceID,
			"calphad-owner-"+suffix, "calphad-org-"+suffix, databaseSHA, databaseSize,
		); err == nil || !strings.Contains(err.Error(), "control_calphad_revisions_owner_declaration_check") {
			t.Fatalf("missing owner declaration err=%v, want owner-declaration constraint", err)
		}

		wrongTypeResourceID := "calphad-wrong-type-owner-declaration-" + suffix
		wrongTypeDeclaration := calphadOwnerDeclarationJSON(
			calphadTestOwnerDeclaration(wrongTypeResourceID, domain.CalphadDatabaseFormatTDB),
		)
		wrongTypeDeclaration["database_id"] = 42
		wrongTypeMetadata := domain.JSONMap{
			domain.CalphadAssessmentPressureLimitsMetadataKey: []float64{
				calphadTestPressureLimits[0], calphadTestPressureLimits[1],
			},
			domain.CalphadOwnerDeclarationMetadataKey: wrongTypeDeclaration,
		}
		if _, err := migrationPool.Exec(ctx, `
INSERT INTO control_calphad_revisions
 (revision_id, resource_id, owner_user_id, owner_org_id, sha256, size_bytes, database_format,
  assessment_pressure_min_pa, assessment_pressure_max_pa, created_at, metadata)
VALUES ($1,$2,$3,$4,$5,$6,'tdb',101325,101325,now(),$7)`,
			"calphad-wrong-type-owner-revision-"+suffix, wrongTypeResourceID,
			"calphad-owner-"+suffix, "calphad-org-"+suffix, databaseSHA, databaseSize,
			jsonBytes(wrongTypeMetadata),
		); err == nil || !strings.Contains(err.Error(), "control_calphad_revisions_owner_declaration_check") {
			t.Fatalf("non-text owner declaration err=%v, want owner-declaration constraint", err)
		}
	})
	var inputBlobSHA string
	var inputBlobSize int64
	var inputBlobPayload []byte
	if err := pool.QueryRow(ctx, `
SELECT input_sha256, input_size_bytes, payload
FROM control_calphad_input_blobs WHERE input_sha256=$1`, databaseSHA).
		Scan(&inputBlobSHA, &inputBlobSize, &inputBlobPayload); err != nil {
		t.Fatalf("load retained CALPHAD input: %v", err)
	}
	inputDigest := sha256.Sum256(inputBlobPayload)
	if inputBlobSHA != databaseSHA || inputBlobSize != databaseSize ||
		int64(len(inputBlobPayload)) != databaseSize || hex.EncodeToString(inputDigest[:]) != databaseSHA {
		t.Fatalf("retained input binding sha=%q size=%d payload=%d", inputBlobSHA, inputBlobSize, len(inputBlobPayload))
	}
	t.Run("input_blob_hash_and_size_constraints", func(t *testing.T) {
		payload := []byte("constraint-probe")
		digest := sha256.Sum256(payload)
		sha := hex.EncodeToString(digest[:])
		if _, err := migrationPool.Exec(ctx, `
INSERT INTO control_calphad_input_blobs
 (input_sha256,input_size_bytes,encoding,payload,created_at)
VALUES ($1,$2,'raw',$3,now())`, strings.Repeat("9", 64), len(payload), payload); err == nil {
			t.Fatal("database accepted input bytes whose digest disagrees with input_sha256")
		}
		if _, err := migrationPool.Exec(ctx, `
INSERT INTO control_calphad_input_blobs
 (input_sha256,input_size_bytes,encoding,payload,created_at)
VALUES ($1,$2,'raw',$3,now())`, sha, len(payload)+1, payload); err == nil {
			t.Fatal("database accepted input bytes whose size disagrees with input_size_bytes")
		}
	})
	var inspectionValidationID, evidenceSHA, databaseInventorySHA, evidenceContract string
	var evidenceSize int64
	var evidencePayload []byte
	if err := pool.QueryRow(ctx, `
SELECT validation.validation_id, validation.run_id, validation.database_inventory_sha256, validation.evidence_contract_version,
       blob.evidence_sha256, blob.evidence_size_bytes, blob.payload
FROM control_calphad_validation_events validation
JOIN control_calphad_evidence_blobs blob
  ON blob.evidence_sha256=validation.evidence_sha256
 AND blob.evidence_size_bytes=validation.evidence_size_bytes
WHERE validation.revision_id=$1 AND validation.run_id IS NOT NULL
  AND validation.operation='inspect' AND validation.status='input_validated'
LIMIT 1`, revisionID).Scan(&inspectionValidationID, &runID, &databaseInventorySHA, &evidenceContract,
		&evidenceSHA, &evidenceSize, &evidencePayload); err != nil {
		t.Fatalf("load retained CALPHAD evidence: %v", err)
	}
	t.Run("evidence_blob_content_bound", func(t *testing.T) {
		evidenceDigest := sha256.Sum256(evidencePayload)
		if int64(len(evidencePayload)) != evidenceSize || hex.EncodeToString(evidenceDigest[:]) != evidenceSHA {
			t.Fatalf("retained evidence binding sha=%s size=%d payload=%d", evidenceSHA, evidenceSize, len(evidencePayload))
		}
		if !calphadSHA256Pattern.MatchString(databaseInventorySHA) ||
			evidenceContract != domain.CalphadEvidenceContractVersion {
			t.Fatalf("retained evidence identity inventory=%q contract=%q", databaseInventorySHA, evidenceContract)
		}
	})
	inspectionPath := "/outputs/calphad/inspection/" + evidenceSHA + ".json"
	runtimeImage := "sha256:" + strings.Repeat("d", 64)
	t.Run("terminal_failure_constraints_reject_null_and_mismatched_tuples", func(t *testing.T) {
		failurePayload := []byte(`{"schema_version":"failure-sql-probe"}`)
		failureDigest := sha256.Sum256(failurePayload)
		failureSHA := hex.EncodeToString(failureDigest[:])
		failurePath := "/outputs/calphad/inspection/" + failureSHA + ".json"
		if _, err := migrationPool.Exec(ctx, `
INSERT INTO control_calphad_evidence_blobs
 (evidence_sha256,evidence_size_bytes,encoding,payload,created_at)
VALUES ($1,$2,'raw',$3,now()) ON CONFLICT DO NOTHING`,
			failureSHA, len(failurePayload), failurePayload,
		); err != nil {
			t.Fatalf("seed failure constraint evidence: %v", err)
		}
		baseArguments := []any{
			revisionID, resourceID, databaseSHA, databaseSize, strings.Repeat("a", 64),
			failurePath, failureSHA, len(failurePayload), runtimeImage, runID,
			domain.CalphadEvidenceContractVersion,
		}
		_, err := migrationPool.Exec(ctx, `
INSERT INTO control_calphad_validation_events
 (validation_id,revision_id,resource_id,database_sha256,database_size_bytes,database_format,
  assessment_pressure_min_pa,assessment_pressure_max_pa,
  request_sha256,status,operation,evidence_path,evidence_sha256,evidence_size_bytes,
  runtime_image_id,pycalphad_version,run_id,evidence_contract_version,
  created_by_authority,created_at,metadata)
VALUES ('calphad-null-failure-`+suffix+`',$1,$2,$3,$4,'tdb',101325,101325,$5,'failed','inspect',$6,$7,$8,
        $9,'0.11.2',$10,$11,'trusted_worker',now(),
        '{"assessment_pressure_limits_Pa":[101325,101325]}'::jsonb)`, baseArguments...)
		if err == nil || !strings.Contains(err.Error(), "control_calphad_validation_failure_tuple_check") {
			t.Fatalf("NULL failure tuple err=%v, want failure tuple constraint", err)
		}
		_, err = migrationPool.Exec(ctx, `
INSERT INTO control_calphad_validation_events
 (validation_id,revision_id,resource_id,database_sha256,database_size_bytes,database_format,
  assessment_pressure_min_pa,assessment_pressure_max_pa,
  request_sha256,status,operation,failure_domain,failure_stage,failure_code,
  evidence_path,evidence_sha256,evidence_size_bytes,runtime_image_id,pycalphad_version,run_id,
  evidence_contract_version,created_by_authority,created_at,metadata)
VALUES ('calphad-mismatched-failure-`+suffix+`',$1,$2,$3,$4,'tdb',101325,101325,$5,'timeout','inspect',
        'input','parse','calphad_parse_failed',$6,$7,$8,$9,'0.11.2',$10,$11,
        'trusted_worker',now(),'{"assessment_pressure_limits_Pa":[101325,101325]}'::jsonb)`,
			baseArguments...)
		if err == nil || !strings.Contains(err.Error(), "control_calphad_validation_failure_tuple_check") {
			t.Fatalf("mismatched failure tuple err=%v, want failure tuple constraint", err)
		}
		_, err = migrationPool.Exec(ctx, `
INSERT INTO control_calphad_validation_events
 (validation_id,revision_id,resource_id,database_sha256,database_size_bytes,database_format,
  assessment_pressure_min_pa,assessment_pressure_max_pa,
  request_sha256,status,operation,failure_domain,failure_stage,failure_code,
  runtime_image_id,pycalphad_version,run_id,evidence_contract_version,
  created_by_authority,created_at,metadata)
VALUES ('calphad-unretained-failure-`+suffix+`',$1,$2,$3,$4,'tdb',101325,101325,$5,'failed','inspect',
        'input','parse','calphad_parse_failed',$6,'0.11.2',$7,$8,
        'trusted_worker',now(),'{"assessment_pressure_limits_Pa":[101325,101325]}'::jsonb)`,
			revisionID, resourceID, databaseSHA, databaseSize, strings.Repeat("a", 64),
			runtimeImage, runID, domain.CalphadEvidenceContractVersion)
		if err == nil || !strings.Contains(err.Error(), "control_calphad_validation_retained_evidence_check") {
			t.Fatalf("unretained failure err=%v, want retained evidence constraint", err)
		}
	})
	t.Run("pressure_binding_constraints", func(t *testing.T) {
		_, err := migrationPool.Exec(ctx, `
INSERT INTO control_calphad_validation_events
 (validation_id, revision_id, resource_id, database_sha256, database_size_bytes, database_format,
  assessment_pressure_min_pa, assessment_pressure_max_pa,
  database_inventory_sha256, request_sha256, status, operation, evidence_path,
  evidence_sha256, evidence_size_bytes, runtime_image_id, pycalphad_version, run_id,
  evidence_contract_version, created_by_authority, created_at, metadata)
VALUES ($1,$2,$3,$4,$5,'tdb',100000,200000,$6,$7,'input_validated','inspect',$8,$9,$10,$11,
        '0.11.2',$12,$13,'trusted_worker',now(),
        '{"assessment_pressure_limits_Pa":[100000,200000]}'::jsonb)`,
			"calphad-pressure-drift-"+suffix, revisionID, resourceID, databaseSHA, databaseSize,
			databaseInventorySHA, strings.Repeat("7", 64), inspectionPath, evidenceSHA,
			evidenceSize, runtimeImage, runID, domain.CalphadEvidenceContractVersion)
		if err == nil || !strings.Contains(err.Error(), "CALPHAD_PRESSURE_BINDING_INVALID") {
			t.Fatalf("validation pressure drift err=%v, want CALPHAD_PRESSURE_BINDING_INVALID", err)
		}
		_, err = migrationPool.Exec(ctx, `
INSERT INTO control_calphad_validation_events
 (validation_id, revision_id, resource_id, database_sha256, database_size_bytes, database_format,
  assessment_pressure_min_pa, assessment_pressure_max_pa,
  database_inventory_sha256, request_sha256, status, operation, evidence_path,
  evidence_sha256, evidence_size_bytes, runtime_image_id, pycalphad_version, run_id,
  evidence_contract_version, created_by_authority, created_at, metadata)
VALUES ($1,$2,$3,$4,$5,'tdb',101325,101325,$6,$7,'input_validated','inspect',$8,$9,$10,$11,
        '0.11.2',$12,$13,'trusted_worker',now(),
        '{"assessment_pressure_limits_Pa":[100000,200000]}'::jsonb)`,
			"calphad-pressure-metadata-drift-"+suffix, revisionID, resourceID, databaseSHA, databaseSize,
			databaseInventorySHA, strings.Repeat("7", 64), inspectionPath, evidenceSHA,
			evidenceSize, runtimeImage, runID, domain.CalphadEvidenceContractVersion)
		if err == nil || !strings.Contains(err.Error(), "CALPHAD_PRESSURE_BINDING_INVALID") {
			t.Fatalf("validation pressure metadata drift err=%v, want CALPHAD_PRESSURE_BINDING_INVALID", err)
		}
		outOfRangeResourceID := "calphad-pressure-out-of-range-resource-" + suffix
		outOfRangeDeclaration := calphadTestOwnerDeclaration(outOfRangeResourceID, domain.CalphadDatabaseFormatTDB)
		outOfRangeDeclaration.AssessmentPressureLimitsPa = [2]float64{0, domain.CalphadReferencePressurePa}
		outOfRangeMetadata := domain.JSONMap{
			domain.CalphadAssessmentPressureLimitsMetadataKey: []float64{0, domain.CalphadReferencePressurePa},
			domain.CalphadOwnerDeclarationMetadataKey:         outOfRangeDeclaration,
		}
		_, err = migrationPool.Exec(ctx, `
INSERT INTO control_calphad_revisions
 (revision_id, resource_id, owner_user_id, owner_org_id, sha256, size_bytes,
  database_format, assessment_pressure_min_pa, assessment_pressure_max_pa, created_at, metadata)
VALUES ($1,$2,$3,$4,$5,$6,'tdb',0,101325,now(),$7)`,
			"calphad-pressure-out-of-range-"+suffix, outOfRangeResourceID,
			"calphad-owner-"+suffix, "calphad-org-"+suffix, databaseSHA, databaseSize,
			jsonBytes(outOfRangeMetadata))
		if err == nil {
			t.Fatal("database accepted revision pressure outside the global range")
		}
	})
	t.Run("schema_fingerprint_verified", func(t *testing.T) {
		if err := VerifyPostgresSchema(ctx, pool); err != nil {
			t.Fatalf("VerifyPostgresSchema: %v", err)
		}
	})
	t.Run("retry_idempotent", func(t *testing.T) {
		var eventCount, operationCount int
		if err := pool.QueryRow(ctx, `
SELECT COUNT(*), COUNT(DISTINCT operation)
FROM control_calphad_validation_events
WHERE revision_id=$1 AND run_id=$2 AND operation IN ('inspect','equilibrium')
  AND status IN ('input_validated','equilibrium_completed') AND evidence_sha256 IS NOT NULL`,
			revisionID, runID).Scan(&eventCount, &operationCount); err != nil {
			t.Fatalf("query idempotent events: %v", err)
		}
		if eventCount != 4 || operationCount != 2 {
			t.Fatalf("idempotent events=%d operations=%d, want inspect plus three distinct equilibrium artifacts", eventCount, operationCount)
		}
	})
	t.Run("multiple_equilibria_idempotent", func(t *testing.T) {
		var eventCount, distinctRequests, distinctEvidence int
		if err := pool.QueryRow(ctx, `
SELECT COUNT(*), COUNT(DISTINCT request_sha256), COUNT(DISTINCT evidence_sha256)
FROM control_calphad_validation_events
WHERE revision_id=$1 AND run_id=$2 AND operation='equilibrium'`, revisionID, runID).
			Scan(&eventCount, &distinctRequests, &distinctEvidence); err != nil {
			t.Fatalf("query equilibrium request identities: %v", err)
		}
		if eventCount != 3 || distinctRequests != 2 || distinctEvidence != 3 {
			t.Fatalf("equilibrium events=%d requests=%d evidence=%d, want same-request observations plus one distinct request", eventCount, distinctRequests, distinctEvidence)
		}
	})
	t.Run("runtime_policy_authorized", func(t *testing.T) {
		_, err := migrationPool.Exec(ctx, `
INSERT INTO control_calphad_validation_events
 (validation_id, revision_id, resource_id, database_sha256, database_size_bytes, database_format,
  status, operation, evidence_path, evidence_sha256, evidence_size_bytes, runtime_image_id,
  pycalphad_version, run_id, created_by_authority, created_at, metadata)
VALUES ($1,$2,$3,$4,$5,'tdb','input_validated','inspect',$6,$7,$8,$9,'0.11.2',$10,'trusted_worker',now(),'{}')`,
			"calphad-forged-policy-"+suffix, revisionID, resourceID, databaseSHA, databaseSize,
			inspectionPath, evidenceSHA, evidenceSize, "sha256:"+strings.Repeat("4", 64), runID)
		if err == nil || !strings.Contains(err.Error(), "CALPHAD_RUNTIME_POLICY_INVALID") {
			t.Fatalf("runtime policy insert err=%v, want CALPHAD_RUNTIME_POLICY_INVALID", err)
		}
	})
	t.Run("temporary_schema_guarded", func(t *testing.T) {
		pooled, err := pool.Acquire(ctx)
		if err != nil {
			t.Fatalf("acquire temporary-schema test connection: %v", err)
		}
		conn := pooled.Hijack()
		defer conn.Close(ctx)
		fakeRunID := "calphad-temp-shadow-run-" + suffix
		for _, statement := range []string{
			`CREATE TEMP TABLE control_runs (run_id text, status text, user_id text, metadata jsonb)`,
			`CREATE TEMP TABLE control_run_leases (run_id text, lease_expires_at timestamptz)`,
			`CREATE TEMP TABLE control_calphad_revisions (revision_id text, owner_user_id text, owner_org_id text)`,
		} {
			if _, err := conn.Exec(ctx, statement); err != nil {
				var pgErr *pgconn.PgError
				if errors.As(err, &pgErr) && pgErr.Code == "42501" {
					t.Log("serving role cannot create temporary tables; shadow path is unavailable")
					return
				}
				t.Fatalf("create trigger-shadow fixture: %v", err)
			}
		}
		if _, err := conn.Exec(ctx, `
INSERT INTO control_runs (run_id,status,user_id,metadata) VALUES ($1,'running',$2,$3)`,
			fakeRunID, "calphad-owner-"+suffix, jsonBytes(domain.JSONMap{
				"org_id": "calphad-org-" + suffix,
				domain.CalphadRuntimePolicyMetadataKey: domain.JSONMap{
					"schema_version": domain.CalphadRuntimePolicySchema, "authority": "control_plane",
					"runtime_image_id": runtimeImage, "pycalphad_version": domain.CalphadPycalphadVersion,
					"network": domain.CalphadRuntimeNetwork, "no_new_privileges": true,
					"read_only_root_filesystem": true, "cap_drop_all": true,
					"cpus_at_most":         domain.CalphadRuntimeCPUsAtMost,
					"memory_bytes_at_most": domain.CalphadRuntimeMemoryBytesAtMost,
					"pids_at_most":         domain.CalphadRuntimePIDsAtMost,
				},
			})); err != nil {
			t.Fatalf("insert shadow run: %v", err)
		}
		if _, err := conn.Exec(ctx, `
INSERT INTO control_run_leases (run_id,lease_expires_at) VALUES ($1,now()+interval '1 hour')`, fakeRunID); err != nil {
			t.Fatalf("insert shadow lease: %v", err)
		}
		if _, err := conn.Exec(ctx, `
INSERT INTO control_calphad_revisions (revision_id,owner_user_id,owner_org_id) VALUES ($1,$2,$3)`,
			revisionID, "calphad-owner-"+suffix, "calphad-org-"+suffix); err != nil {
			t.Fatalf("insert shadow revision: %v", err)
		}
		_, appendErr := conn.Exec(ctx, `
INSERT INTO public.control_calphad_validation_events
 (validation_id, revision_id, resource_id, database_sha256, database_size_bytes, database_format,
  database_inventory_sha256, request_sha256, status, operation, evidence_path,
  evidence_sha256, evidence_size_bytes, runtime_image_id, pycalphad_version, run_id,
  evidence_contract_version, created_by_authority, created_at, metadata)
VALUES ($1,$2,$3,$4,$5,'tdb',$6,$7,'input_validated','inspect',$8,$9,$10,$11,'0.11.2',$12,
        $13,'trusted_worker',now(),'{}')`,
			"calphad-temp-shadow-event-"+suffix, revisionID, resourceID, databaseSHA, databaseSize,
			databaseInventorySHA, strings.Repeat("6", 64), inspectionPath, evidenceSHA,
			evidenceSize, runtimeImage, fakeRunID, domain.CalphadEvidenceContractVersion)
		var pgErr *pgconn.PgError
		if errors.As(appendErr, &pgErr) && pgErr.Code == "42501" {
			t.Log("execute-only serving role cannot raw-insert the public ledger; temporary shadow is inert")
			return
		}
		if appendErr == nil || !strings.Contains(appendErr.Error(), "CALPHAD_RUN_LEASE_INVALID") {
			t.Fatalf("temporary schema shadow insert err=%v, want public-schema lease guard", appendErr)
		}
	})
	t.Run("inspection_lineage_required", func(t *testing.T) {
		var equilibriumSHA string
		var equilibriumSize int64
		if err := pool.QueryRow(ctx, `
SELECT evidence_sha256, evidence_size_bytes
FROM control_calphad_validation_events
WHERE revision_id=$1 AND run_id=$2 AND operation='equilibrium'`,
			revisionID, runID).Scan(&equilibriumSHA, &equilibriumSize); err != nil {
			t.Fatalf("load equilibrium evidence binding: %v", err)
		}
		_, err := migrationPool.Exec(ctx, `
INSERT INTO control_calphad_validation_events
 (validation_id, revision_id, resource_id, database_sha256, database_size_bytes, database_format,
  status, operation, evidence_path, evidence_sha256, evidence_size_bytes, runtime_image_id,
  pycalphad_version, run_id, inspection_evidence_sha256, created_by_authority, created_at, metadata)
VALUES ($1,$2,$3,$4,$5,'tdb','equilibrium_completed','equilibrium',$6,$7,$8,$9,'0.11.2',$10,$7,'trusted_worker',now(),'{}')`,
			"calphad-forged-lineage-"+suffix, revisionID, resourceID, databaseSHA, databaseSize,
			"/outputs/calphad/equilibrium/"+equilibriumSHA+".json", equilibriumSHA,
			equilibriumSize, runtimeImage, runID)
		if err == nil || !strings.Contains(err.Error(), "CALPHAD_INSPECTION_REQUIRED") {
			t.Fatalf("equilibrium lineage insert err=%v, want CALPHAD_INSPECTION_REQUIRED", err)
		}
	})
	t.Run("inspection_inventory_bound", func(t *testing.T) {
		var equilibriumSHA string
		var equilibriumSize int64
		if err := pool.QueryRow(ctx, `
SELECT evidence_sha256, evidence_size_bytes
FROM control_calphad_validation_events
WHERE revision_id=$1 AND run_id=$2 AND operation='equilibrium'
ORDER BY created_at LIMIT 1`, revisionID, runID).Scan(&equilibriumSHA, &equilibriumSize); err != nil {
			t.Fatalf("load equilibrium evidence binding: %v", err)
		}
		_, err := migrationPool.Exec(ctx, `
INSERT INTO control_calphad_validation_events
 (validation_id, revision_id, resource_id, database_sha256, database_size_bytes, database_format,
  database_inventory_sha256, request_sha256, status, operation, evidence_path,
  evidence_sha256, evidence_size_bytes, runtime_image_id, pycalphad_version, run_id,
  inspection_evidence_sha256, evidence_contract_version, created_by_authority, created_at, metadata)
VALUES ($1,$2,$3,$4,$5,'tdb',$6,$7,'equilibrium_completed','equilibrium',$8,$9,$10,$11,
        '0.11.2',$12,$13,$14,'trusted_worker',now(),'{}')`,
			"calphad-forged-inventory-"+suffix, revisionID, resourceID, databaseSHA, databaseSize,
			strings.Repeat("9", 64), strings.Repeat("5", 64),
			"/outputs/calphad/equilibrium/"+equilibriumSHA+".json", equilibriumSHA,
			equilibriumSize, runtimeImage, runID, evidenceSHA, domain.CalphadEvidenceContractVersion)
		if err == nil || !strings.Contains(err.Error(), "CALPHAD_INSPECTION_REQUIRED") {
			t.Fatalf("equilibrium inventory lineage insert err=%v, want CALPHAD_INSPECTION_REQUIRED", err)
		}
	})
	t.Run("database_revision_binding", func(t *testing.T) {
		if _, err := migrationPool.Exec(ctx, `
INSERT INTO control_calphad_validation_events
 (validation_id, revision_id, resource_id, database_sha256, database_size_bytes, database_format,
  status, operation, evidence_path, evidence_sha256,
  evidence_size_bytes, runtime_image_id, pycalphad_version, run_id, created_by_authority, created_at, metadata)
VALUES ($1,$2,$3,$4,$5,'tdb','input_validated','inspect',$6,$7,$8,$9,'0.11.2',$10,'trusted_worker',now(),'{}')`,
			"calphad-forged-resource-"+suffix, revisionID, "different-resource", databaseSHA, databaseSize,
			inspectionPath, evidenceSHA, evidenceSize, runtimeImage, runID); err == nil {
			t.Fatal("composite revision/resource foreign key allowed mismatched evidence")
		}
		if _, err := migrationPool.Exec(ctx, `
INSERT INTO control_calphad_validation_events
 (validation_id, revision_id, resource_id, database_sha256, database_size_bytes, database_format,
  status, operation, evidence_path, evidence_sha256,
  evidence_size_bytes, runtime_image_id, pycalphad_version, run_id, created_by_authority, created_at, metadata)
VALUES ($1,$2,$3,$4,$5,'dat','input_validated','inspect',$6,$7,$8,$9,'0.11.2',$10,'trusted_worker',now(),'{}')`,
			"calphad-forged-format-"+suffix, revisionID, resourceID, databaseSHA, databaseSize,
			inspectionPath, evidenceSHA, evidenceSize, runtimeImage, runID); err == nil ||
			!strings.Contains(err.Error(), "CALPHAD_INPUT_RETENTION_REQUIRED") {
			t.Fatalf("revision/database format mismatch err=%v, want fail-closed retained-input binding", err)
		}
		missingEvidenceSHA := strings.Repeat("9", 64)
		if _, err := migrationPool.Exec(ctx, `
INSERT INTO control_calphad_validation_events
 (validation_id, revision_id, resource_id, database_sha256, database_size_bytes, database_format,
  status, operation, evidence_path, evidence_sha256, evidence_size_bytes, runtime_image_id,
  pycalphad_version, run_id, created_by_authority, created_at, metadata)
VALUES ($1,$2,$3,$4,$5,'tdb','input_validated','inspect',$6,$7,1,$8,'0.11.2',$9,'trusted_worker',now(),'{}')`,
			"calphad-missing-evidence-"+suffix, revisionID, resourceID, databaseSHA, databaseSize,
			"/outputs/calphad/inspection/"+missingEvidenceSHA+".json", missingEvidenceSHA,
			runtimeImage, runID); err == nil {
			t.Fatal("database allowed a validation event without its exact retained evidence blob")
		}
	})
	t.Run("immutable_runtime_image", func(t *testing.T) {
		if _, err := migrationPool.Exec(ctx, `
INSERT INTO control_calphad_validation_events
 (validation_id, revision_id, resource_id, database_sha256, database_size_bytes, database_format,
  status, operation, evidence_path, evidence_sha256,
  evidence_size_bytes, pycalphad_version, run_id, created_by_authority, created_at, metadata)
VALUES ($1,$2,$3,$4,$5,'tdb','input_validated','inspect',$6,$7,$8,'0.11.2',$9,'trusted_worker',now(),'{}')`,
			"calphad-forged-runtime-"+suffix, revisionID, resourceID, databaseSHA, databaseSize,
			inspectionPath, evidenceSHA, evidenceSize, runID); err == nil {
			t.Fatal("database allowed trusted validation without immutable runtime image identity")
		}
	})
	t.Run("database_digest_binding", func(t *testing.T) {
		if _, err := migrationPool.Exec(ctx, `
INSERT INTO control_calphad_validation_events
 (validation_id, revision_id, resource_id, database_sha256, database_size_bytes, database_format,
  status, operation, evidence_path, evidence_sha256, evidence_size_bytes, runtime_image_id,
  pycalphad_version, run_id, created_by_authority, created_at, metadata)
VALUES ($1,$2,$3,$4,$5,'tdb','input_validated','inspect',$6,$7,$8,$9,'0.11.2',$10,'trusted_worker',now(),'{}')`,
			"calphad-forged-database-"+suffix, revisionID, resourceID, strings.Repeat("6", 64), databaseSize,
			inspectionPath, evidenceSHA, evidenceSize, runtimeImage, runID); err == nil {
			t.Fatal("database allowed validation whose database digest differs from its revision")
		}
	})
	t.Run("durable_run_binding", func(t *testing.T) {
		if _, err := migrationPool.Exec(ctx, `
INSERT INTO control_calphad_validation_events
 (validation_id, revision_id, resource_id, database_sha256, database_size_bytes, database_format,
  status, operation, evidence_path, evidence_sha256, evidence_size_bytes, runtime_image_id,
  pycalphad_version, run_id, created_by_authority, created_at, metadata)
VALUES ($1,$2,$3,$4,$5,'tdb','input_validated','inspect',$6,$7,$8,$9,'0.11.2',$10,'trusted_worker',now(),'{}')`,
			"calphad-forged-run-"+suffix, revisionID, resourceID, databaseSHA, databaseSize,
			inspectionPath, evidenceSHA, evidenceSize, runtimeImage, "missing-run-"+suffix); err == nil {
			t.Fatal("database allowed trusted validation without a durable control run")
		}
	})
	t.Run("parent_same_tenant", func(t *testing.T) {
		forgedResourceID := "calphad-forged-resource-" + suffix
		if _, err := migrationPool.Exec(ctx, `
INSERT INTO control_calphad_revisions
 (revision_id, resource_id, owner_user_id, owner_org_id, sha256, size_bytes,
  database_format, assessment_pressure_min_pa, assessment_pressure_max_pa,
  parent_revision_id, created_at, metadata)
VALUES ($1,$2,'different-owner','different-org',$3,$4,'tdb',101325,101325,$5,now(),$6)`,
			"calphad-forged-parent-"+suffix, forgedResourceID,
			databaseSHA, databaseSize, revisionID,
			jsonBytes(calphadTestRevisionMetadata(forgedResourceID, domain.CalphadDatabaseFormatTDB))); err == nil {
			t.Fatal("database allowed cross-tenant CALPHAD parent lineage")
		}
		rogueRevisionID := "calphad-rogue-tenant-" + suffix
		rogueResourceID := "calphad-rogue-resource-" + suffix
		rogueDatabaseSHA := databaseSHA
		if _, err := migrationPool.Exec(ctx, `
INSERT INTO control_calphad_revisions
 (revision_id, resource_id, owner_user_id, owner_org_id, sha256, size_bytes,
  database_format, assessment_pressure_min_pa, assessment_pressure_max_pa, created_at, metadata)
VALUES ($1,$2,'different-owner','different-org',$3,$4,'tdb',101325,101325,now(),$5)`,
			rogueRevisionID, rogueResourceID, rogueDatabaseSHA, databaseSize,
			jsonBytes(calphadTestRevisionMetadata(rogueResourceID, domain.CalphadDatabaseFormatTDB))); err != nil {
			t.Fatalf("insert isolated tenant test revision: %v", err)
		}
		if _, err := migrationPool.Exec(ctx, `
INSERT INTO control_calphad_validation_events
 (validation_id, revision_id, resource_id, database_sha256, database_size_bytes, database_format,
  status, operation, evidence_path, evidence_sha256, evidence_size_bytes, runtime_image_id,
  pycalphad_version, run_id, created_by_authority, created_at, metadata)
VALUES ($1,$2,$3,$4,$5,'tdb','input_validated','inspect',$6,$7,$8,$9,'0.11.2',$10,'trusted_worker',now(),'{}')`,
			"calphad-forged-run-tenant-"+suffix, rogueRevisionID, rogueResourceID,
			rogueDatabaseSHA, databaseSize, inspectionPath, evidenceSHA, evidenceSize, runtimeImage, runID); err == nil {
			t.Fatal("database allowed an active run to validate another owner's revision")
		}
	})
	t.Run("run_lease_authorized", func(t *testing.T) {
		if _, err := pool.Exec(ctx, `UPDATE control_run_leases SET lease_expires_at=now()-interval '1 minute' WHERE run_id=$1`, runID); err != nil {
			t.Fatalf("expire qualification lease: %v", err)
		}
		_, appendErr := migrationPool.Exec(ctx, `
INSERT INTO control_calphad_validation_events
 (validation_id, revision_id, resource_id, database_sha256, database_size_bytes, database_format,
  status, operation, evidence_path, evidence_sha256, evidence_size_bytes, runtime_image_id,
  pycalphad_version, run_id, created_by_authority, created_at, metadata)
VALUES ($1,$2,$3,$4,$5,'tdb','input_validated','inspect',$6,$7,$8,$9,'0.11.2',$10,'trusted_worker',now(),'{}')`,
			"calphad-expired-lease-"+suffix, revisionID, resourceID, databaseSHA, databaseSize,
			inspectionPath, evidenceSHA, evidenceSize, runtimeImage, runID)
		_, restoreErr := pool.Exec(ctx, `UPDATE control_run_leases SET lease_expires_at=now()+interval '1 hour' WHERE run_id=$1`, runID)
		if restoreErr != nil {
			t.Fatalf("restore qualification lease: %v", restoreErr)
		}
		if appendErr == nil || !strings.Contains(appendErr.Error(), "CALPHAD_RUN_LEASE_INVALID") {
			t.Fatalf("expired lease insert err=%v, want CALPHAD_RUN_LEASE_INVALID", appendErr)
		}
	})
	mutationChecks := map[string]func(*testing.T){
		"append_only_revision_delete": func(t *testing.T) {
			if _, err := migrationPool.Exec(ctx, `DELETE FROM control_calphad_revisions WHERE revision_id=$1`, revisionID); err == nil {
				t.Fatal("append-only trigger allowed revision DELETE")
			}
		},
		"append_only_validation_update": func(t *testing.T) {
			if _, err := migrationPool.Exec(ctx, `UPDATE control_calphad_validation_events SET status=status WHERE revision_id=$1`, revisionID); err == nil {
				t.Fatal("append-only trigger allowed validation UPDATE")
			}
		},
		"append_only_validation_delete": func(t *testing.T) {
			if _, err := migrationPool.Exec(ctx, `DELETE FROM control_calphad_validation_events WHERE revision_id=$1`, revisionID); err == nil {
				t.Fatal("append-only trigger allowed validation DELETE")
			}
		},
		"append_only_evidence_update": func(t *testing.T) {
			if _, err := migrationPool.Exec(ctx, `UPDATE control_calphad_evidence_blobs SET payload=payload WHERE evidence_sha256=$1`, evidenceSHA); err == nil {
				t.Fatal("append-only trigger allowed evidence blob UPDATE")
			}
		},
		"append_only_evidence_delete": func(t *testing.T) {
			if _, err := migrationPool.Exec(ctx, `DELETE FROM control_calphad_evidence_blobs WHERE evidence_sha256=$1`, evidenceSHA); err == nil {
				t.Fatal("append-only trigger allowed evidence blob DELETE")
			}
		},
		"append_only_input_update": func(t *testing.T) {
			if _, err := migrationPool.Exec(ctx, `UPDATE control_calphad_input_blobs SET payload=payload WHERE input_sha256=$1`, databaseSHA); err == nil {
				t.Fatal("append-only trigger allowed input blob UPDATE")
			}
		},
		"append_only_input_delete": func(t *testing.T) {
			if _, err := migrationPool.Exec(ctx, `DELETE FROM control_calphad_input_blobs WHERE input_sha256=$1`, databaseSHA); err == nil {
				t.Fatal("append-only trigger allowed input blob DELETE")
			}
		},
		"append_only_revision_truncate": func(t *testing.T) {
			if _, err := migrationPool.Exec(ctx, `TRUNCATE TABLE control_calphad_revisions`); err == nil {
				t.Fatal("append-only trigger allowed revision TRUNCATE")
			}
		},
		"append_only_validation_truncate": func(t *testing.T) {
			if _, err := migrationPool.Exec(ctx, `TRUNCATE TABLE control_calphad_validation_events`); err == nil {
				t.Fatal("append-only trigger allowed validation TRUNCATE")
			}
		},
		"append_only_evidence_truncate": func(t *testing.T) {
			if _, err := migrationPool.Exec(ctx, `TRUNCATE TABLE control_calphad_evidence_blobs`); err == nil {
				t.Fatal("append-only trigger allowed evidence blob TRUNCATE")
			}
		},
		"append_only_input_truncate": func(t *testing.T) {
			if _, err := migrationPool.Exec(ctx, `TRUNCATE TABLE control_calphad_input_blobs`); err == nil {
				t.Fatal("append-only trigger allowed input blob TRUNCATE")
			}
		},
	}
	for name, check := range mutationChecks {
		t.Run(name, check)
	}
	var retainedBlob, retainedInput, retainedRevision, retainedValidations int
	if err := pool.QueryRow(ctx, `
SELECT COUNT(*) FROM control_calphad_evidence_blobs WHERE evidence_sha256=$1`, evidenceSHA).
		Scan(&retainedBlob); err != nil || retainedBlob != 1 {
		t.Fatalf("evidence blob was not retained after mutation attempts: count=%d err=%v", retainedBlob, err)
	}
	if err := pool.QueryRow(ctx, `SELECT COUNT(*) FROM control_calphad_input_blobs WHERE input_sha256=$1`, databaseSHA).
		Scan(&retainedInput); err != nil || retainedInput != 1 {
		t.Fatalf("input blob was not retained after mutation attempts: count=%d err=%v", retainedInput, err)
	}
	if err := pool.QueryRow(ctx, `SELECT COUNT(*) FROM control_calphad_revisions WHERE revision_id=$1`, revisionID).
		Scan(&retainedRevision); err != nil || retainedRevision != 1 {
		t.Fatalf("revision was not retained after mutation attempts: count=%d err=%v", retainedRevision, err)
	}
	if err := pool.QueryRow(ctx, `SELECT COUNT(*) FROM control_calphad_validation_events WHERE revision_id=$1`, revisionID).
		Scan(&retainedValidations); err != nil || retainedValidations < 2 {
		t.Fatalf("validation events were not retained after mutation attempts: count=%d err=%v", retainedValidations, err)
	}
	postgresStore := NewPostgresStore(pool)
	if err := postgresStore.PurgeResource(ctx, resourceID); err != nil {
		t.Fatalf("PurgeResource before replay: %v", err)
	}
	replayedInput, err := postgresStore.GetCalphadRevisionInputForOwner(
		ctx, resourceID, "calphad-owner-"+suffix, "calphad-org-"+suffix,
	)
	if err != nil || replayedInput.SHA256 != databaseSHA || replayedInput.SizeBytes != databaseSize ||
		!bytes.Equal(replayedInput.Bytes, inputBlobPayload) {
		t.Fatalf("retained PostgreSQL input after catalog GC=%+v err=%v", replayedInput, err)
	}
	if _, err := postgresStore.GetCalphadLedgerForOwner(
		ctx, resourceID, "calphad-owner-"+suffix, "calphad-org-"+suffix,
	); err != nil {
		t.Fatalf("ledger after catalog GC: %v", err)
	}
	replayedEvidence, err := postgresStore.GetCalphadValidationEvidenceForOwner(
		ctx, resourceID, inspectionValidationID, "calphad-owner-"+suffix, "calphad-org-"+suffix,
	)
	if err != nil || replayedEvidence.SHA256 != evidenceSHA ||
		replayedEvidence.SizeBytes != evidenceSize || !bytes.Equal(replayedEvidence.Bytes, evidencePayload) {
		t.Fatalf("retained PostgreSQL evidence after catalog GC=%+v err=%v", replayedEvidence, err)
	}
	t.Run("equilibrium_reads_require_retained_inspection_event", func(t *testing.T) {
		var equilibriumValidationID string
		if err := pool.QueryRow(ctx, `
SELECT validation_id
FROM control_calphad_validation_events
WHERE revision_id=$1 AND operation='equilibrium' AND status='equilibrium_completed'
ORDER BY created_at DESC, validation_id DESC
LIMIT 1`, revisionID).Scan(&equilibriumValidationID); err != nil {
			t.Fatalf("load retained equilibrium validation: %v", err)
		}
		if _, err := postgresStore.GetCalphadValidationEvidenceForOwner(
			ctx, resourceID, equilibriumValidationID,
			"calphad-owner-"+suffix, "calphad-org-"+suffix,
		); err != nil {
			t.Fatalf("read equilibrium before lineage corruption: %v", err)
		}
		if _, err := migrationPool.Exec(ctx, `
ALTER TABLE control_calphad_validation_events
DISABLE TRIGGER control_calphad_validation_append_only`); err != nil {
			t.Fatalf("disable append-only trigger for corruption probe: %v", err)
		}
		triggerDisabled := true
		defer func() {
			if triggerDisabled {
				_, _ = migrationPool.Exec(ctx, `
ALTER TABLE control_calphad_validation_events
ENABLE TRIGGER control_calphad_validation_append_only`)
			}
		}()
		if _, err := migrationPool.Exec(ctx, `
DELETE FROM control_calphad_validation_events WHERE validation_id=$1`, inspectionValidationID); err != nil {
			t.Fatalf("remove inspection event for corruption probe: %v", err)
		}
		if _, err := migrationPool.Exec(ctx, `
ALTER TABLE control_calphad_validation_events
ENABLE TRIGGER control_calphad_validation_append_only`); err != nil {
			t.Fatalf("restore append-only trigger after corruption probe: %v", err)
		}
		triggerDisabled = false

		ledger, err := postgresStore.GetCalphadLedgerForOwner(
			ctx, resourceID, "calphad-owner-"+suffix, "calphad-org-"+suffix,
		)
		if err != nil {
			t.Fatalf("read corrupted-lineage ledger: %v", err)
		}
		var equilibrium *domain.CalphadValidationRecord
		for index := range ledger.Validations {
			if ledger.Validations[index].ValidationID == equilibriumValidationID {
				equilibrium = &ledger.Validations[index]
				break
			}
		}
		if equilibrium == nil || equilibrium.Promotable ||
			equilibrium.EvidenceRetention != domain.CalphadEvidenceRetentionLegacyUnretained {
			t.Fatalf("equilibrium with missing inspection event was promotable: %+v", equilibrium)
		}
		if _, err := postgresStore.GetCalphadValidationEvidenceForOwner(
			ctx, resourceID, equilibriumValidationID,
			"calphad-owner-"+suffix, "calphad-org-"+suffix,
		); !errors.Is(err, ErrCalphadEvidenceRetentionRequired) {
			t.Fatalf("equilibrium evidence read after inspection removal err=%v, want retention error", err)
		}
	})
}

func TestPostgresCalphadTenantCapacityIsTransactionalAndIsolated(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	migrationDSN := os.Getenv("ULTRA_CONTROL_TEST_MIGRATION_DATABASE_URL")
	if migrationDSN == "" {
		t.Skip("ULTRA_CONTROL_TEST_MIGRATION_DATABASE_URL is required for role-separated qualification")
	}
	ctx := context.Background()
	migrationPool, err := pgxpool.New(ctx, migrationDSN)
	if err != nil {
		t.Fatalf("pgxpool.New(migration): %v", err)
	}
	defer migrationPool.Close()
	if err := ApplyPostgresSchema(ctx, migrationPool); err != nil {
		t.Fatalf("ApplyPostgresSchema: %v", err)
	}
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New(serving): %v", err)
	}
	defer pool.Close()
	var servingRole string
	if err := pool.QueryRow(ctx, `SELECT current_user`).Scan(&servingRole); err != nil {
		t.Fatalf("load serving role: %v", err)
	}
	if err := GrantPostgresServingPrivileges(ctx, migrationPool, servingRole); err != nil {
		t.Fatalf("GrantPostgresServingPrivileges: %v", err)
	}
	store := NewPostgresStore(pool)
	suffix := strconv.FormatInt(time.Now().UnixNano(), 36)
	runtimeImageID := "sha256:" + strings.Repeat("d", 64)

	type quotaFixture struct {
		owner      string
		org        string
		resourceID string
		revision   domain.CalphadRevisionRecord
		run        domain.RunRecord
		lease      domain.RunLeaseRecord
	}
	testDigest := func(value string) string {
		digest := sha256.Sum256([]byte(value))
		return hex.EncodeToString(digest[:])
	}
	provision := func(t *testing.T, label string, inputSize int) quotaFixture {
		t.Helper()
		owner := "calphad-capacity-owner-" + label + "-" + suffix
		org := "calphad-capacity-org-" + label + "-" + suffix
		resourceID := "calphad-capacity-resource-" + label + "-" + suffix
		inputBytes, inputSHA := calphadTestInput("CAPACITY-"+label, inputSize)
		thread, createErr := store.CreateThread(ctx, domain.CreateThreadInput{
			UserID: owner, Title: "CALPHAD capacity " + label,
		})
		if createErr != nil {
			t.Fatalf("CreateThread(%s): %v", label, createErr)
		}
		if _, upsertErr := store.UpsertResource(ctx, domain.UpsertResourceInput{
			ResourceID: resourceID, OriginalName: resourceID + ".tdb",
			ContentType: "application/x-thermocalc-tdb", SizeBytes: int64(len(inputBytes)),
			SHA256: inputSHA, OwnerUserID: owner, OwnerOrgID: org, Status: "active",
			CreatedAt: domain.Now(), UpdatedAt: domain.Now(),
			Metadata: calphadTestOwnerMetadata(resourceID),
		}); upsertErr != nil {
			t.Fatalf("UpsertResource(%s): %v", label, upsertErr)
		}
		run, createErr := store.CreateRun(ctx, domain.CreateRunInput{
			ThreadID: thread.ThreadID, UserID: owner, Goal: "CALPHAD capacity qualification",
			Metadata: domain.JSONMap{
				"org_id": org, "file_ids": []string{resourceID},
				"resource_descriptors": []domain.JSONMap{
					calphadTestSelectedDescriptor(resourceID, inputSHA, int64(len(inputBytes))),
				},
				domain.CalphadRuntimePolicyMetadataKey: domain.JSONMap{
					"schema_version": domain.CalphadRuntimePolicySchema, "authority": "control_plane",
					"runtime_image_id": runtimeImageID, "pycalphad_version": domain.CalphadPycalphadVersion,
					"network": domain.CalphadRuntimeNetwork, "no_new_privileges": true,
					"read_only_root_filesystem": true, "cap_drop_all": true,
					"cpus_at_most":         domain.CalphadRuntimeCPUsAtMost,
					"memory_bytes_at_most": domain.CalphadRuntimeMemoryBytesAtMost,
					"pids_at_most":         domain.CalphadRuntimePIDsAtMost,
				},
				"principal": domain.JSONMap{"user_id": owner, "org_id": org, "role": "researcher"},
			},
		})
		if createErr != nil {
			t.Fatalf("CreateRun(%s): %v", label, createErr)
		}
		lease, leaseErr := store.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
			RunID: run.RunID, WorkerID: "calphad-capacity-worker-" + label + "-" + suffix,
			TTL: time.Hour, Now: domain.Now(),
		})
		if leaseErr != nil {
			t.Fatalf("AcquireRunLease(%s): %v", label, leaseErr)
		}
		revision, revisionErr := store.CreateCalphadRevision(ctx, domain.CreateCalphadRevisionInput{
			ResourceID: resourceID, OwnerUserID: "  " + owner + "  ", OwnerOrgID: "\t" + org + "\n",
			ExpectedSHA256: inputSHA, ExpectedSizeBytes: int64(len(inputBytes)), InputBytes: inputBytes,
			AssessmentPressureLimitsPa: calphadTestPressureLimits,
			Metadata:                   domain.JSONMap{"caller_controlled": "must not persist"},
		})
		if revisionErr != nil {
			t.Fatalf("CreateCalphadRevision(%s): %v", label, revisionErr)
		}
		return quotaFixture{
			owner: owner, org: org, resourceID: resourceID,
			revision: revision, run: run, lease: lease,
		}
	}
	validation := func(fixture quotaFixture, label string) domain.AppendCalphadValidationInput {
		input := domain.AppendCalphadValidationInput{
			ResourceID: fixture.resourceID, OwnerUserID: " " + fixture.owner + " ",
			OwnerOrgID: "\t" + fixture.org + " ", DatabaseSHA256: fixture.revision.SHA256,
			DatabaseSizeBytes: fixture.revision.SizeBytes, DatabaseFormat: fixture.revision.DatabaseFormat,
			OwnerDeclaration:           calphadTestOwnerDeclaration(fixture.resourceID, fixture.revision.DatabaseFormat),
			AssessmentPressureLimitsPa: calphadTestPressureLimits,
			DatabaseInventorySHA256:    testDigest("inventory:" + fixture.resourceID),
			RequestSHA256:              testDigest("request:" + label),
			Status:                     "input_validated", Operation: "inspect", RuntimeImageID: runtimeImageID,
			PycalphadVersion: domain.CalphadPycalphadVersion, RunID: fixture.run.RunID,
			LeaseWorkerID: fixture.lease.WorkerID, LeaseToken: fixture.lease.LeaseToken,
			CreatedByAuthority: "trusted_worker",
			Metadata:           domain.JSONMap{"caller_controlled": label},
		}
		input.EvidenceBytes, input.EvidenceSHA256, input.EvidencePath = calphadTestEvidence(label, input)
		input.EvidenceSizeBytes = int64(len(input.EvidenceBytes))
		return input
	}

	type capacitySnapshot struct {
		maximumBytes   int64
		maximumEvents  int64
		inputBytes     int64
		evidenceBytes  int64
		validationRows int64
	}
	loadCapacity := func(t *testing.T, fixture quotaFixture) capacitySnapshot {
		t.Helper()
		var snapshot capacitySnapshot
		if queryErr := pool.QueryRow(ctx, `
SELECT max_retained_bytes, max_validation_events, retained_input_bytes,
       retained_evidence_bytes, validation_events
FROM control_calphad_tenant_capacity
WHERE owner_user_id=$1 AND owner_org_id=$2`, fixture.owner, fixture.org).Scan(
			&snapshot.maximumBytes, &snapshot.maximumEvents, &snapshot.inputBytes,
			&snapshot.evidenceBytes, &snapshot.validationRows,
		); queryErr != nil {
			t.Fatalf("load capacity for %s: %v", fixture.resourceID, queryErr)
		}
		return snapshot
	}
	setCapacityToCurrent := func(t *testing.T, fixture quotaFixture) {
		t.Helper()
		if _, updateErr := migrationPool.Exec(ctx, `
UPDATE control_calphad_tenant_capacity
SET max_retained_bytes=retained_input_bytes+retained_evidence_bytes,
    max_validation_events=validation_events, updated_at=clock_timestamp()
WHERE owner_user_id=$1 AND owner_org_id=$2`, fixture.owner, fixture.org); updateErr != nil {
			t.Fatalf("exhaust tenant capacity for %s: %v", fixture.resourceID, updateErr)
		}
	}

	tenantA := provision(t, "a", 257)
	baselineInput := validation(tenantA, "baseline-a")
	for _, test := range []struct {
		name   string
		mutate func(map[string]any)
	}{
		{
			name: "top-level key substitution",
			mutate: func(evidence map[string]any) {
				delete(evidence, "execution_contract")
				evidence["untrusted_replacement"] = map[string]any{}
			},
		},
		{
			name: "non-object request",
			mutate: func(evidence map[string]any) {
				evidence["request"] = "not-an-object"
			},
		},
		{
			name: "weakened execution contract",
			mutate: func(evidence map[string]any) {
				evidence["execution_contract"].(map[string]any)["caller_code_accepted"] = true
			},
		},
		{
			name: "forged persistence authority",
			mutate: func(evidence map[string]any) {
				evidence["validation_persistence"].(map[string]any)["catalog_metadata_updated"] = true
			},
		},
	} {
		t.Run("writer_rejects_"+test.name, func(t *testing.T) {
			invalid := calphadTestMutateEvidence(t, baselineInput, test.mutate)
			if _, appendErr := store.AppendCalphadValidation(ctx, invalid); !errors.Is(appendErr, ErrConflict) {
				t.Fatalf("invalid evidence append err=%v, want ErrConflict", appendErr)
			}
			var residue int64
			if queryErr := pool.QueryRow(ctx, `
SELECT count(*) FROM control_calphad_evidence_blobs WHERE evidence_sha256=$1`,
				invalid.EvidenceSHA256,
			).Scan(&residue); queryErr != nil || residue != 0 {
				t.Fatalf("invalid evidence residue=%d err=%v", residue, queryErr)
			}
		})
	}
	baseline, err := store.AppendCalphadValidation(ctx, baselineInput)
	if err != nil {
		t.Fatalf("append tenant A baseline: %v", err)
	}
	setCapacityToCurrent(t, tenantA)
	beforeReplay := loadCapacity(t, tenantA)
	replayed, err := store.AppendCalphadValidation(ctx, baselineInput)
	if err != nil || replayed.ValidationID != baseline.ValidationID {
		t.Fatalf("exact replay at capacity=%+v err=%v", replayed, err)
	}
	if afterReplay := loadCapacity(t, tenantA); afterReplay != beforeReplay {
		t.Fatalf("exact replay changed capacity: before=%+v after=%+v", beforeReplay, afterReplay)
	}

	rejectedInput := validation(tenantA, "rejected-a")
	if _, err := store.AppendCalphadValidation(ctx, rejectedInput); !errors.Is(
		err, ErrCalphadTenantCapacityExceeded,
	) {
		t.Fatalf("over-capacity append err=%v, want ErrCalphadTenantCapacityExceeded", err)
	}
	if afterRejection := loadCapacity(t, tenantA); afterRejection != beforeReplay {
		t.Fatalf("quota rejection changed capacity: before=%+v after=%+v", beforeReplay, afterRejection)
	}
	var rejectedBlobs, rejectedEvents int64
	if err := pool.QueryRow(ctx, `
SELECT (SELECT count(*) FROM control_calphad_evidence_blobs WHERE evidence_sha256=$1),
       (SELECT count(*) FROM control_calphad_validation_events
        WHERE revision_id=$2 AND evidence_sha256=$1)`,
		rejectedInput.EvidenceSHA256, tenantA.revision.RevisionID,
	).Scan(&rejectedBlobs, &rejectedEvents); err != nil {
		t.Fatalf("query rejected append residue: %v", err)
	}
	if rejectedBlobs != 0 || rejectedEvents != 0 {
		t.Fatalf("quota rejection left blob/event residue: blobs=%d events=%d", rejectedBlobs, rejectedEvents)
	}

	tenantB := provision(t, "b", 263)
	tenantBBefore := loadCapacity(t, tenantB)
	tenantBInput := validation(tenantB, "accepted-b")
	if _, err := store.AppendCalphadValidation(ctx, tenantBInput); err != nil {
		t.Fatalf("independent normalized tenant append: %v", err)
	}
	tenantBAfter := loadCapacity(t, tenantB)
	if tenantBAfter.evidenceBytes != tenantBBefore.evidenceBytes+tenantBInput.EvidenceSizeBytes ||
		tenantBAfter.validationRows != tenantBBefore.validationRows+1 {
		t.Fatalf("tenant B capacity did not advance independently: before=%+v after=%+v", tenantBBefore, tenantBAfter)
	}
	if tenantAAfterB := loadCapacity(t, tenantA); tenantAAfterB != beforeReplay {
		t.Fatalf("tenant B append changed exhausted tenant A: before=%+v after=%+v", beforeReplay, tenantAAfterB)
	}
	var normalizedTenantRows int64
	if err := pool.QueryRow(ctx, `
SELECT count(*) FROM control_calphad_tenant_capacity
WHERE ((owner_user_id=$1 AND owner_org_id=$2) OR (owner_user_id=$3 AND owner_org_id=$4))
  AND owner_user_id=btrim(owner_user_id) AND owner_org_id=btrim(owner_org_id)`,
		tenantA.owner, tenantA.org, tenantB.owner, tenantB.org,
	).Scan(&normalizedTenantRows); err != nil {
		t.Fatalf("query normalized tenant capacity keys: %v", err)
	}
	if normalizedTenantRows != 2 {
		t.Fatalf("normalized tenant capacity rows=%d, want 2 independent rows", normalizedTenantRows)
	}

	tenantC := provision(t, "c", 269)
	racerA := validation(tenantC, "racer-a")
	racerB := validation(tenantC, "racer-b")
	if racerA.EvidenceSizeBytes != racerB.EvidenceSizeBytes {
		t.Fatalf("concurrency fixtures differ in size: a=%d b=%d", racerA.EvidenceSizeBytes, racerB.EvidenceSizeBytes)
	}
	beforeRace := loadCapacity(t, tenantC)
	if _, err := migrationPool.Exec(ctx, `
UPDATE control_calphad_tenant_capacity
SET max_retained_bytes=retained_input_bytes+retained_evidence_bytes+$3,
    max_validation_events=validation_events+1, updated_at=clock_timestamp()
WHERE owner_user_id=$1 AND owner_org_id=$2`,
		tenantC.owner, tenantC.org, racerA.EvidenceSizeBytes,
	); err != nil {
		t.Fatalf("set one-event race headroom: %v", err)
	}
	start := make(chan struct{})
	results := make(chan error, 2)
	for _, input := range []domain.AppendCalphadValidationInput{racerA, racerB} {
		input := input
		go func() {
			<-start
			_, appendErr := store.AppendCalphadValidation(ctx, input)
			results <- appendErr
		}()
	}
	close(start)
	successes, capacityFailures := 0, 0
	for range 2 {
		appendErr := <-results
		switch {
		case appendErr == nil:
			successes++
		case errors.Is(appendErr, ErrCalphadTenantCapacityExceeded):
			capacityFailures++
		default:
			t.Fatalf("concurrent capacity append returned unexpected error: %v", appendErr)
		}
	}
	if successes != 1 || capacityFailures != 1 {
		t.Fatalf("concurrent near-limit results successes=%d capacity_failures=%d", successes, capacityFailures)
	}
	afterRace := loadCapacity(t, tenantC)
	if afterRace.evidenceBytes != beforeRace.evidenceBytes+racerA.EvidenceSizeBytes ||
		afterRace.validationRows != beforeRace.validationRows+1 {
		t.Fatalf("concurrent admission counters drifted: before=%+v after=%+v", beforeRace, afterRace)
	}
	var admittedBlobs, admittedEvents int64
	if err := pool.QueryRow(ctx, `
SELECT (SELECT count(*) FROM control_calphad_evidence_blobs
        WHERE evidence_sha256 IN ($1,$2)),
       (SELECT count(*) FROM control_calphad_validation_events
        WHERE revision_id=$3 AND evidence_sha256 IN ($1,$2))`,
		racerA.EvidenceSHA256, racerB.EvidenceSHA256, tenantC.revision.RevisionID,
	).Scan(&admittedBlobs, &admittedEvents); err != nil {
		t.Fatalf("query concurrent admission residue: %v", err)
	}
	if admittedBlobs != 1 || admittedEvents != 1 {
		t.Fatalf("concurrent near-limit residue blobs=%d events=%d, want one committed pair", admittedBlobs, admittedEvents)
	}
	if err := VerifyPostgresSchema(ctx, migrationPool); err != nil {
		t.Fatalf("VerifyPostgresSchema after capacity qualification: %v", err)
	}
}

func TestCalphadLedgerSchemaEncodesGovernanceWithoutRelationalizingGibbsModels(t *testing.T) {
	t.Parallel()
	paths := []string{
		"schema.sql",
		"../../migrations/000008_calphad_revision_ledger.up.sql",
	}
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("ReadFile(%s): %v", path, err)
		}
		doc := strings.ToLower(string(data))
		for _, required := range []string{
			"control_calphad_input_blobs",
			"control_calphad_revisions",
			"control_calphad_validation_events",
			"control_calphad_evidence_blobs",
			"unique (revision_id, resource_id, sha256, size_bytes, database_format)",
			"unique (revision_id, assessment_pressure_min_pa, assessment_pressure_max_pa)",
			"foreign key (revision_id, assessment_pressure_min_pa, assessment_pressure_max_pa)",
			"assessment_pressure_limits_pa",
			"assessment_pressure_min_pa >= 1e-9",
			"assessment_pressure_max_pa <= 1e12",
			"assessment_pressure_min_pa <= assessment_pressure_max_pa",
			"foreign key (sha256, size_bytes)",
			"references control_calphad_input_blobs(input_sha256, input_size_bytes)",
			"foreign key (revision_id, resource_id, database_sha256, database_size_bytes, database_format)",
			"database_format in ('tdb', 'dat')",
			"owner_declaration",
			"metadata ? 'owner_declaration'",
			"ultra.calphad.owner-declaration.v1",
			"assessment_temperature_limits_k",
			"foreign key (evidence_sha256, evidence_size_bytes)",
			"foreign key (run_id) references control_runs(run_id)",
			"octet_length(payload) = evidence_size_bytes",
			"encode(sha256(payload), 'hex') = evidence_sha256",
			"encode(sha256(payload), 'hex') = input_sha256",
			"octet_length(payload) = input_size_bytes",
			"between 1 and 67108864",
			"control_calphad_evidence_blob_payload_sha_check",
			"sha256:[0-9a-f]{64}",
			"between 1 and 33554432",
			"/outputs/calphad/",
			"created_by_authority = 'control_plane'",
			"before update or delete",
			"before truncate",
			"ultra_validate_calphad_revision_parent",
			"ultra_validate_calphad_pressure_binding",
			"ultra_validate_calphad_input_retention",
			"ultra_validate_calphad_validation_run_authority",
			"ultra_validate_calphad_equilibrium_inspection_lineage",
			"set search_path = pg_catalog",
			"public.control_runs",
			"public.control_run_leases",
			"public.control_calphad_revisions",
			"public.control_calphad_input_blobs",
			"public.control_calphad_validation_events",
			"public.control_calphad_evidence_blobs",
			"inspection_evidence_sha256",
			"database_inventory_sha256",
			"request_sha256",
			"evidence_contract_version",
			"ultra.calphad.retained-evidence.v2",
			"control_calphad_validation_evidence_uidx",
			"control_calphad_validation_request_idx",
			"calphad_runtime_policy",
			"ultra.calphad.runtime-policy.v2",
			"'network', 'none'",
			"'no_new_privileges', true",
			"'read_only_root_filesystem', true",
			"'cap_drop_all', true",
			"'cpus_at_most', 8",
			"'memory_bytes_at_most', 34359738368",
			"'pids_at_most', 4096",
			"pycalphad_version = '0.11.2'",
			"calphad_run_lease_invalid",
			"calphad_input_retention_required",
			"calphad_pressure_binding_invalid",
			"errcode = '28000'",
			"run_record.user_id = revision.owner_user_id",
			"add column if not exists database_sha256",
			"add column if not exists database_format",
			"add column if not exists assessment_pressure_min_pa",
			"set database_sha256 = revision.sha256",
			"validate constraint control_calphad_evidence_blob_payload_sha_check",
		} {
			if !strings.Contains(doc, required) {
				t.Errorf("%s missing governance invariant %q", path, required)
			}
		}
		for _, forbidden := range []string{
			"gibbs_model",
			"thermodynamic_parameter",
			"parameter_value",
		} {
			if strings.Contains(doc, forbidden) {
				t.Errorf("%s unexpectedly relationalizes or lifecycle-couples CALPHAD bytes via %q", path, forbidden)
			}
		}
		calphadStart := strings.Index(doc, "create table if not exists control_calphad_revisions")
		if calphadStart < 0 {
			continue
		}
		calphadSection := doc[calphadStart:]
		if end := strings.Index(calphadSection, "create table if not exists control_resource_share_grants"); end >= 0 {
			calphadSection = calphadSection[:end]
		}
		if strings.Contains(calphadSection, "references control_resources") {
			t.Errorf("%s lifecycle-couples append-only audit rows to mutable resource garbage collection", path)
		}
	}
	down, err := os.ReadFile("../../migrations/000008_calphad_revision_ledger.down.sql")
	if err != nil {
		t.Fatalf("ReadFile(down migration): %v", err)
	}
	downSQL := strings.ToLower(string(down))
	if !strings.Contains(downSQL, "irreversible") || strings.Contains(downSQL, "drop table") {
		t.Fatalf("CALPHAD down migration must fail closed without dropping audit data: %s", downSQL)
	}
}

func TestCalphadFailureTupleCompatibilityConstraintIsStructurallyBalanced(t *testing.T) {
	t.Parallel()
	pattern := regexp.MustCompile(
		`(?s)\('control_calphad_validation_failure_tuple_check'\s*,\s*\$check\$(CHECK .*?)\$check\$\)`,
	)
	for _, path := range []string{
		"schema.sql",
		"../../migrations/000008_calphad_revision_ledger.up.sql",
	} {
		payload, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("ReadFile(%s): %v", path, err)
		}
		match := pattern.FindSubmatch(payload)
		if len(match) != 2 {
			t.Fatalf("%s missing dynamic failure-tuple constraint definition", path)
		}
		expression := match[1]
		depth := 0
		quoted := false
		for index := 0; index < len(expression); index++ {
			switch expression[index] {
			case '\'':
				if quoted && index+1 < len(expression) && expression[index+1] == '\'' {
					index++
					continue
				}
				quoted = !quoted
			case '(':
				if !quoted {
					depth++
				}
			case ')':
				if !quoted {
					depth--
					if depth < 0 {
						t.Fatalf("%s dynamic failure-tuple constraint has an extra closing parenthesis", path)
					}
				}
			}
		}
		if quoted || depth != 0 {
			t.Fatalf(
				"%s dynamic failure-tuple constraint is structurally invalid: quoted=%t depth=%d",
				path, quoted, depth,
			)
		}
	}
}

func TestPostgresCalphadAppendUsesExactExecuteOnlyWriterSignature(t *testing.T) {
	t.Parallel()
	payload, err := os.ReadFile("calphad_ledger.go")
	if err != nil {
		t.Fatalf("ReadFile(calphad_ledger.go): %v", err)
	}
	normalized := strings.Join(strings.Fields(string(payload)), " ")
	for _, required := range []string{
		"FROM public.ultra_create_calphad_revision_v1(",
		"$8::double precision, $9::double precision, $10::bytea, $11::jsonb",
		"FROM public.ultra_append_calphad_validation_v1(",
		"$24::text, $25::text, $26::text, $27::jsonb",
	} {
		if !strings.Contains(normalized, required) {
			t.Fatalf("PostgreSQL CALPHAD write must use exact execute-only function shape %q", required)
		}
	}
}

func TestCalphadLedgerFunctionsAndTriggersMatchSchemaAndMigration(t *testing.T) {
	t.Parallel()
	schemaBytes, err := os.ReadFile("schema.sql")
	if err != nil {
		t.Fatalf("ReadFile(schema.sql): %v", err)
	}
	migrationBytes, err := os.ReadFile("../../migrations/000008_calphad_revision_ledger.up.sql")
	if err != nil {
		t.Fatalf("ReadFile(CALPHAD migration): %v", err)
	}
	writerMigrationBytes, err := os.ReadFile("../../migrations/000009_calphad_execute_only_writers.up.sql")
	if err != nil {
		t.Fatalf("ReadFile(CALPHAD writer migration): %v", err)
	}
	normalize := func(value string) string {
		return strings.Join(strings.Fields(strings.ToLower(value)), " ")
	}
	extract := func(document []byte, pattern, label string) string {
		t.Helper()
		match := regexp.MustCompile(pattern).Find(document)
		if len(match) == 0 {
			t.Fatalf("missing %s", label)
		}
		return normalize(string(match))
	}
	for _, name := range sortedMapKeys(requiredCalphadFunctionFingerprints) {
		pattern := `(?is)create\s+or\s+replace\s+function\s+public\.` +
			regexp.QuoteMeta(name) + `\s*\(.*?\).*?\$\$\s*;`
		schemaDefinition := extract(schemaBytes, pattern, "schema function "+name)
		functionMigration := migrationBytes
		if strings.HasSuffix(name, "_v1") {
			functionMigration = writerMigrationBytes
		}
		migrationDefinition := extract(functionMigration, pattern, "migration function "+name)
		if schemaDefinition != migrationDefinition {
			t.Errorf("CALPHAD function %s differs between embedded schema and migration", name)
		}
	}
	for _, name := range sortedMapKeys(requiredCalphadTriggerFingerprints) {
		pattern := `(?is)create\s+trigger\s+` + regexp.QuoteMeta(name) + `\s+.*?;`
		schemaDefinition := extract(schemaBytes, pattern, "schema trigger "+name)
		migrationDefinition := extract(migrationBytes, pattern, "migration trigger "+name)
		if schemaDefinition != migrationDefinition {
			t.Errorf("CALPHAD trigger %s differs between embedded schema and migration", name)
		}
	}
	for _, table := range []string{
		"control_calphad_input_blobs", "control_calphad_revisions",
		"control_calphad_evidence_blobs", "control_calphad_validation_events",
	} {
		pattern := `(?is)create\s+table\s+if\s+not\s+exists\s+` + regexp.QuoteMeta(table) + `\s*\(.*?\n\);`
		schemaDefinition := extract(schemaBytes, pattern, "schema table "+table)
		migrationDefinition := extract(migrationBytes, pattern, "migration table "+table)
		if schemaDefinition != migrationDefinition {
			t.Errorf("CALPHAD table %s differs between embedded schema and migration", table)
		}
	}
	capacityPattern := `(?is)create\s+table\s+if\s+not\s+exists\s+control_calphad_tenant_capacity\s*\(.*?\n\);`
	if extract(schemaBytes, capacityPattern, "schema capacity table") !=
		extract(writerMigrationBytes, capacityPattern, "migration capacity table") {
		t.Error("CALPHAD tenant capacity table differs between embedded schema and migration")
	}
	for label, document := range map[string]string{
		"embedded schema": string(schemaBytes), "writer migration": string(writerMigrationBytes),
	} {
		for _, suffix := range []string{"tdb", "dat"} {
			validPattern := `~ '\.` + suffix + `$'`
			invalidPattern := `~ '\\.` + suffix + `$'`
			if count := strings.Count(document, validPattern); count != 3 {
				t.Errorf("%s standard-conforming .%s suffix regex count=%d, want 3", label, suffix, count)
			}
			if strings.Contains(document, invalidPattern) {
				t.Errorf("%s contains double-escaped PostgreSQL .%s suffix regex", label, suffix)
			}
		}
	}
}
