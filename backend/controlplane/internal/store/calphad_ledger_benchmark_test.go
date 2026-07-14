package store

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

const calphadMemoryBenchmarkValidationCount = 501

type calphadMemoryBenchmarkFixture struct {
	store        *MemoryStore
	ownerUserID  string
	ownerOrgID   string
	resourceID   string
	revision     domain.CalphadRevisionRecord
	baseInput    domain.AppendCalphadValidationInput
	latestInput  domain.AppendCalphadValidationInput
	latestRecord domain.CalphadValidationRecord
}

func newCalphadMemoryBenchmarkFixture(
	tb testing.TB,
	validationCount int,
) calphadMemoryBenchmarkFixture {
	tb.Helper()
	ctx := context.Background()
	mem := NewMemoryStore()
	ownerUserID := "calphad-benchmark-owner"
	ownerOrgID := "calphad-benchmark-org"
	resourceID := "calphad-benchmark-resource"
	workerID := "calphad-benchmark-worker"
	runtimeImageID := "sha256:" + strings.Repeat("d", 64)
	inputBytes, inputSHA := calphadTestInput("CALPHAD-BENCHMARK-TDB", 4096)
	now := time.Date(2026, 7, 12, 12, 0, 0, 0, time.UTC)

	thread, err := mem.CreateThread(ctx, domain.CreateThreadInput{
		UserID: ownerUserID,
		Title:  "CALPHAD benchmark",
	})
	if err != nil {
		tb.Fatalf("CreateThread: %v", err)
	}
	run, err := mem.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   ownerUserID,
		Goal:     "Benchmark bounded CALPHAD ledger operations",
		Metadata: domain.JSONMap{
			"org_id":   ownerOrgID,
			"file_ids": []string{resourceID},
			"resource_descriptors": []domain.JSONMap{
				calphadTestSelectedDescriptor(resourceID, inputSHA, int64(len(inputBytes))),
			},
			domain.CalphadRuntimePolicyMetadataKey: domain.JSONMap{
				"schema_version":            domain.CalphadRuntimePolicySchema,
				"authority":                 "control_plane",
				"runtime_image_id":          runtimeImageID,
				"pycalphad_version":         domain.CalphadPycalphadVersion,
				"network":                   domain.CalphadRuntimeNetwork,
				"no_new_privileges":         true,
				"read_only_root_filesystem": true,
				"cap_drop_all":              true,
				"cpus_at_most":              domain.CalphadRuntimeCPUsAtMost,
				"memory_bytes_at_most":      domain.CalphadRuntimeMemoryBytesAtMost,
				"pids_at_most":              domain.CalphadRuntimePIDsAtMost,
			},
		},
	})
	if err != nil {
		tb.Fatalf("CreateRun: %v", err)
	}
	lease, err := mem.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID: run.RunID, WorkerID: workerID, TTL: 24 * time.Hour, Now: domain.Now(),
	})
	if err != nil {
		tb.Fatalf("AcquireRunLease: %v", err)
	}
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID: resourceID, OriginalName: resourceID + ".tdb",
		ContentType: "application/x-thermocalc-tdb", SizeBytes: int64(len(inputBytes)),
		SHA256: inputSHA, OwnerUserID: ownerUserID, OwnerOrgID: ownerOrgID,
		Status: "active", CreatedAt: now, UpdatedAt: now,
		Metadata: calphadTestOwnerMetadata(resourceID),
	}); err != nil {
		tb.Fatalf("UpsertResource: %v", err)
	}
	revision, err := mem.CreateCalphadRevision(ctx, domain.CreateCalphadRevisionInput{
		ResourceID: resourceID, OwnerUserID: ownerUserID, OwnerOrgID: ownerOrgID,
		ExpectedSHA256: inputSHA, ExpectedSizeBytes: int64(len(inputBytes)),
		AssessmentPressureLimitsPa: calphadTestPressureLimits,
		InputBytes:                 inputBytes, CreatedByUserID: ownerUserID, CreatedAt: now,
	})
	if err != nil {
		tb.Fatalf("CreateCalphadRevision: %v", err)
	}

	baseInput := domain.AppendCalphadValidationInput{
		ResourceID: resourceID, OwnerUserID: ownerUserID, OwnerOrgID: ownerOrgID,
		DatabaseSHA256: inputSHA, DatabaseSizeBytes: int64(len(inputBytes)),
		DatabaseFormat:             domain.CalphadDatabaseFormatTDB,
		OwnerDeclaration:           calphadTestOwnerDeclaration(resourceID, domain.CalphadDatabaseFormatTDB),
		AssessmentPressureLimitsPa: calphadTestPressureLimits,
		DatabaseInventorySHA256:    strings.Repeat("c", 64),
		RequestSHA256:              strings.Repeat("1", 64),
		Status:                     "input_validated", Operation: "inspect",
		RuntimeImageID: runtimeImageID, PycalphadVersion: domain.CalphadPycalphadVersion,
		RunID: run.RunID, LeaseWorkerID: workerID, LeaseToken: lease.LeaseToken,
		CreatedByAuthority: "trusted_worker",
	}
	fixture := calphadMemoryBenchmarkFixture{
		store: mem, ownerUserID: ownerUserID, ownerOrgID: ownerOrgID,
		resourceID: resourceID, revision: revision, baseInput: baseInput,
	}
	for index := 0; index < validationCount; index++ {
		input := fixture.inputWithEvidence(fmt.Sprintf("seed-%03d", index))
		input.CreatedAt = now.Add(time.Duration(index+1) * time.Microsecond)
		record, appendErr := mem.AppendCalphadValidation(ctx, input)
		if appendErr != nil {
			tb.Fatalf("AppendCalphadValidation(seed %d): %v", index, appendErr)
		}
		fixture.latestInput = input
		fixture.latestRecord = record
	}
	return fixture
}

func (fixture calphadMemoryBenchmarkFixture) inputWithEvidence(
	label string,
) domain.AppendCalphadValidationInput {
	input := fixture.baseInput
	input.EvidenceBytes, input.EvidenceSHA256, input.EvidencePath = calphadTestEvidence(label, input)
	input.EvidenceSizeBytes = int64(len(input.EvidenceBytes))
	return input
}

func (fixture calphadMemoryBenchmarkFixture) removeAppendedValidation(
	tb testing.TB,
	record domain.CalphadValidationRecord,
) {
	tb.Helper()
	fixture.store.mu.Lock()
	defer fixture.store.mu.Unlock()
	last := len(fixture.store.calphadValidations) - 1
	if last < 0 || fixture.store.calphadValidations[last].ValidationID != record.ValidationID {
		tb.Fatalf("appended validation %q is not the ledger tail", record.ValidationID)
	}
	fixture.store.calphadValidations = fixture.store.calphadValidations[:last]
	delete(fixture.store.calphadEvidenceBlobs, record.EvidenceSHA256)
}

func BenchmarkMemoryCalphadLedger(b *testing.B) {
	b.Run("append-after-501-validations", func(b *testing.B) {
		fixture := newCalphadMemoryBenchmarkFixture(b, calphadMemoryBenchmarkValidationCount)
		input := fixture.inputWithEvidence("measured-append")
		input.CreatedAt = fixture.latestRecord.CreatedAt.Add(time.Microsecond)
		ctx := context.Background()
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			record, err := fixture.store.AppendCalphadValidation(ctx, input)
			if err != nil {
				b.Fatalf("AppendCalphadValidation: %v", err)
			}
			b.StopTimer()
			fixture.removeAppendedValidation(b, record)
			b.StartTimer()
		}
	})

	b.Run("idempotent-retry-after-501-validations", func(b *testing.B) {
		fixture := newCalphadMemoryBenchmarkFixture(b, calphadMemoryBenchmarkValidationCount)
		ctx := context.Background()
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			record, err := fixture.store.AppendCalphadValidation(ctx, fixture.latestInput)
			if err != nil {
				b.Fatalf("AppendCalphadValidation(retry): %v", err)
			}
			if record.ValidationID != fixture.latestRecord.ValidationID {
				b.Fatalf("retry validation=%q, want %q", record.ValidationID, fixture.latestRecord.ValidationID)
			}
		}
	})

	b.Run("keyset-page-500-after-anchor", func(b *testing.B) {
		fixture := newCalphadMemoryBenchmarkFixture(b, calphadMemoryBenchmarkValidationCount)
		input := domain.GetCalphadLedgerPageInput{
			ResourceID: fixture.resourceID, OwnerUserID: fixture.ownerUserID,
			OwnerOrgID: fixture.ownerOrgID, Limit: 500,
			ExpectedRevisionID: fixture.revision.RevisionID,
			BeforeCreatedAt:    fixture.latestRecord.CreatedAt,
			BeforeValidationID: fixture.latestRecord.ValidationID,
		}
		ctx := context.Background()
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			ledger, err := fixture.store.GetCalphadLedgerPageForOwner(ctx, input)
			if err != nil {
				b.Fatalf("GetCalphadLedgerPageForOwner: %v", err)
			}
			if len(ledger.Validations) != 500 || !ledger.HasMore {
				b.Fatalf("ledger page length=%d has_more=%t, want 500/true", len(ledger.Validations), ledger.HasMore)
			}
		}
	})

	b.Run("retained-evidence-replay-after-501-validations", func(b *testing.B) {
		fixture := newCalphadMemoryBenchmarkFixture(b, calphadMemoryBenchmarkValidationCount)
		ctx := context.Background()
		b.SetBytes(fixture.latestRecord.EvidenceSizeBytes)
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			evidence, err := fixture.store.GetCalphadValidationEvidenceForOwner(
				ctx, fixture.resourceID, fixture.latestRecord.ValidationID,
				fixture.ownerUserID, fixture.ownerOrgID,
			)
			if err != nil {
				b.Fatalf("GetCalphadValidationEvidenceForOwner: %v", err)
			}
			if evidence.SHA256 != fixture.latestRecord.EvidenceSHA256 ||
				evidence.SizeBytes != fixture.latestRecord.EvidenceSizeBytes {
				b.Fatalf("retained evidence binding changed: %+v", evidence)
			}
		}
	})
}
