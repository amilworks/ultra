package store

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

// GoldGate training STORE layer (M1): jobs/leases/events (data-agent clone),
// gold-set freeze lifecycle, versions + audit, retrain requests, benchmark
// runs, status upsert, and the seeded gate-policy/guardrail-clause reads.

func TestTrainingJobCreateGetList(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	memory := NewMemoryStore()

	for _, invalid := range []struct {
		name  string
		input domain.CreateTrainingJobInput
	}{
		{name: "missing model key", input: domain.CreateTrainingJobInput{JobType: "training.sync"}},
		{name: "missing job type", input: domain.CreateTrainingJobInput{ModelKey: TrainingSeedModelKey}},
	} {
		if _, err := memory.CreateTrainingJob(ctx, invalid.input); err == nil {
			t.Fatalf("%s: expected validation error", invalid.name)
		}
	}
	if _, err := memory.CreateTrainingJob(ctx, domain.CreateTrainingJobInput{
		ModelKey: "no-such-model",
		JobType:  "training.sync",
	}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("unknown model: err = %v, want ErrNotFound", err)
	}

	job, err := memory.CreateTrainingJob(ctx, domain.CreateTrainingJobInput{
		JobID:       "job-sync-1",
		ModelKey:    TrainingSeedModelKey,
		JobType:     "training.sync",
		GPUPool:     "titan",
		Params:      domain.JSONMap{"dataset": "Prairie_Dog_Active_Learning"},
		OwnerUserID: "user-1",
		Metadata:    domain.JSONMap{"trigger": "manual"},
	})
	if err != nil {
		t.Fatalf("CreateTrainingJob: %v", err)
	}
	if job.JobID != "job-sync-1" || job.Status != "queued" || job.ModelKey != TrainingSeedModelKey {
		t.Fatalf("unexpected job: %+v", job)
	}
	if job.CreatedAt.IsZero() || job.UpdatedAt.IsZero() || !job.StartedAt.IsZero() || !job.CompletedAt.IsZero() {
		t.Fatalf("unexpected job timestamps: %+v", job)
	}
	if job.Params["dataset"] != "Prairie_Dog_Active_Learning" || job.CreatedByUserID != "user-1" {
		t.Fatalf("unexpected job fields: %+v", job)
	}
	if _, err := memory.CreateTrainingJob(ctx, domain.CreateTrainingJobInput{
		JobID:    "job-sync-1",
		ModelKey: TrainingSeedModelKey,
		JobType:  "training.sync",
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("duplicate job id: err = %v, want ErrConflict", err)
	}

	got, err := memory.GetTrainingJob(ctx, "job-sync-1")
	if err != nil || got.JobID != job.JobID || got.JobType != "training.sync" {
		t.Fatalf("GetTrainingJob = %+v (err %v)", got, err)
	}
	if _, err := memory.GetTrainingJob(ctx, "missing"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetTrainingJob missing: err = %v, want ErrNotFound", err)
	}

	events, err := memory.ListTrainingJobEvents(ctx, job.JobID, 0)
	if err != nil || len(events) != 1 {
		t.Fatalf("expected the created event, got %d (err %v)", len(events), err)
	}
	if events[0].Sequence != 1 || events[0].EventType != "training.job.created" {
		t.Fatalf("unexpected created event: %+v", events[0])
	}

	if _, err := memory.CreateTrainingJob(ctx, domain.CreateTrainingJobInput{
		JobID:    "job-finetune-1",
		ModelKey: TrainingSeedModelKey,
		JobType:  "training.finetune",
	}); err != nil {
		t.Fatalf("create second job: %v", err)
	}
	jobs, err := memory.ListTrainingJobs(ctx, TrainingSeedModelKey, "", 0)
	if err != nil || len(jobs) != 2 {
		t.Fatalf("ListTrainingJobs = %d jobs (err %v), want 2", len(jobs), err)
	}
	if jobs[0].CreatedAt.Before(jobs[1].CreatedAt) {
		t.Fatalf("jobs must list newest first: %v then %v", jobs[0].CreatedAt, jobs[1].CreatedAt)
	}
	queued, err := memory.ListTrainingJobs(ctx, TrainingSeedModelKey, "queued", 0)
	if err != nil || len(queued) != 2 {
		t.Fatalf("status filter queued = %d (err %v), want 2", len(queued), err)
	}
	if runs, err := memory.ListTrainingJobs(ctx, TrainingSeedModelKey, "running", 0); err != nil || len(runs) != 0 {
		t.Fatalf("status filter running = %d (err %v), want 0", len(runs), err)
	}
	if others, err := memory.ListTrainingJobs(ctx, "other-model", "", 0); err != nil || len(others) != 0 {
		t.Fatalf("model filter = %d (err %v), want 0", len(others), err)
	}
	if limited, err := memory.ListTrainingJobs(ctx, TrainingSeedModelKey, "", 1); err != nil || len(limited) != 1 {
		t.Fatalf("limit 1 = %d (err %v), want 1", len(limited), err)
	}
}

func TestTrainingJobStatusTransitionsStampTimestamps(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	memory := NewMemoryStore()
	job, err := memory.CreateTrainingJob(ctx, domain.CreateTrainingJobInput{
		ModelKey: TrainingSeedModelKey,
		JobType:  "training.benchmark",
	})
	if err != nil {
		t.Fatalf("CreateTrainingJob: %v", err)
	}

	if _, err := memory.UpdateTrainingJobStatus(ctx, domain.UpdateTrainingJobStatusInput{
		JobID:  job.JobID,
		Status: "sideways",
	}); err == nil {
		t.Fatal("invalid status must be rejected")
	}
	if _, err := memory.UpdateTrainingJobStatus(ctx, domain.UpdateTrainingJobStatusInput{
		JobID:  "missing",
		Status: "running",
	}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing job: err = %v, want ErrNotFound", err)
	}

	running, err := memory.UpdateTrainingJobStatus(ctx, domain.UpdateTrainingJobStatusInput{
		JobID:             job.JobID,
		Status:            "running",
		ProgressCompleted: 5,
		ProgressTotal:     3,
	})
	if err != nil {
		t.Fatalf("update to running: %v", err)
	}
	if running.Status != "running" || running.StartedAt.IsZero() || !running.CompletedAt.IsZero() {
		t.Fatalf("running must stamp started_at only: %+v", running)
	}
	if running.ProgressCompleted != 3 || running.ProgressTotal != 3 {
		t.Fatalf("progress must clamp completed to total: %+v", running)
	}
	startedAt := running.StartedAt

	for _, terminal := range []string{"succeeded", "failed", "canceled"} {
		result, err := memory.UpdateTrainingJobStatus(ctx, domain.UpdateTrainingJobStatusInput{
			JobID:         job.JobID,
			Status:        terminal,
			Error:         "boom",
			OutputSummary: domain.JSONMap{"exit": terminal},
		})
		if err != nil {
			t.Fatalf("update to %s: %v", terminal, err)
		}
		if result.CompletedAt.IsZero() || !result.StartedAt.Equal(startedAt) {
			t.Fatalf("%s must stamp completed_at and keep started_at: %+v", terminal, result)
		}
		if result.Error != "boom" || result.OutputSummary["exit"] != terminal {
			t.Fatalf("%s must persist error/output summary: %+v", terminal, result)
		}
		// Requeue clears completed_at (non-terminal) but keeps started_at.
		requeued, err := memory.UpdateTrainingJobStatus(ctx, domain.UpdateTrainingJobStatusInput{
			JobID:  job.JobID,
			Status: "queued",
		})
		if err != nil {
			t.Fatalf("requeue after %s: %v", terminal, err)
		}
		if !requeued.CompletedAt.IsZero() {
			t.Fatalf("requeue must clear completed_at: %+v", requeued)
		}
		if requeued.OutputSummary["exit"] != terminal {
			t.Fatalf("nil output summary must keep the existing one: %+v", requeued)
		}
	}
}

func TestTrainingJobEventAutoSequencing(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	memory := NewMemoryStore()
	job, err := memory.CreateTrainingJob(ctx, domain.CreateTrainingJobInput{
		ModelKey: TrainingSeedModelKey,
		JobType:  "training.assemble",
	})
	if err != nil {
		t.Fatalf("CreateTrainingJob: %v", err)
	}

	if _, err := memory.AppendTrainingJobEvent(ctx, domain.AppendTrainingJobEventInput{
		JobID:     "missing",
		EventType: "training.job.progressed",
	}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("append to missing job: err = %v, want ErrNotFound", err)
	}

	second, err := memory.AppendTrainingJobEvent(ctx, domain.AppendTrainingJobEventInput{
		JobID:     job.JobID,
		EventType: "training.job.progressed",
		Message:   "tiles staged",
		Metadata:  domain.JSONMap{"staged": float64(128)},
	})
	if err != nil || second.Sequence != 2 {
		t.Fatalf("auto sequence after create = %d (err %v), want 2", second.Sequence, err)
	}
	third, err := memory.AppendTrainingJobEvent(ctx, domain.AppendTrainingJobEventInput{
		JobID:     job.JobID,
		EventType: "training.job.progressed",
	})
	if err != nil || third.Sequence != 3 {
		t.Fatalf("auto sequence = %d (err %v), want 3", third.Sequence, err)
	}
	explicit, err := memory.AppendTrainingJobEvent(ctx, domain.AppendTrainingJobEventInput{
		JobID:     job.JobID,
		Sequence:  10,
		EventType: "training.job.progressed",
	})
	if err != nil || explicit.Sequence != 10 {
		t.Fatalf("explicit sequence = %d (err %v), want 10", explicit.Sequence, err)
	}
	if _, err := memory.AppendTrainingJobEvent(ctx, domain.AppendTrainingJobEventInput{
		EventID:   second.EventID,
		JobID:     job.JobID,
		EventType: "training.job.progressed",
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("duplicate event id: err = %v, want ErrConflict", err)
	}

	events, err := memory.ListTrainingJobEvents(ctx, job.JobID, 0)
	if err != nil || len(events) != 4 {
		t.Fatalf("ListTrainingJobEvents = %d (err %v), want 4", len(events), err)
	}
	for i := 1; i < len(events); i++ {
		if events[i-1].Sequence >= events[i].Sequence {
			t.Fatalf("events must be sequence ASC: %d then %d", events[i-1].Sequence, events[i].Sequence)
		}
	}
	if limited, err := memory.ListTrainingJobEvents(ctx, job.JobID, 2); err != nil || len(limited) != 2 || limited[0].Sequence != 1 {
		t.Fatalf("limit 2 = %+v (err %v), want first two sequences", limited, err)
	}
}

func TestTrainingJobLeaseLifecycle(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	memory := NewMemoryStore()
	job, err := memory.CreateTrainingJob(ctx, domain.CreateTrainingJobInput{
		ModelKey: TrainingSeedModelKey,
		JobType:  "training.finetune",
	})
	if err != nil {
		t.Fatalf("CreateTrainingJob: %v", err)
	}
	t0 := domain.Now()

	if _, _, err := memory.AcquireTrainingJobLease(ctx, domain.AcquireTrainingJobLeaseInput{
		JobID: "missing", WorkerID: "trainer-1", TTL: time.Minute, Now: t0,
	}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("acquire missing job: err = %v, want ErrNotFound", err)
	}

	lease, leased, err := memory.AcquireTrainingJobLease(ctx, domain.AcquireTrainingJobLeaseInput{
		JobID: job.JobID, WorkerID: "trainer-1", TTL: time.Minute, Now: t0,
	})
	if err != nil {
		t.Fatalf("AcquireTrainingJobLease: %v", err)
	}
	if lease.LeaseToken == "" || lease.WorkerID != "trainer-1" || !lease.LeaseExpiresAt.Equal(t0.Add(time.Minute)) {
		t.Fatalf("unexpected lease: %+v", lease)
	}
	if leased.Status != "running" || leased.StartedAt.IsZero() {
		t.Fatalf("acquire must flip the job to running and stamp started_at: %+v", leased)
	}
	events, err := memory.ListTrainingJobEvents(ctx, job.JobID, 0)
	if err != nil || len(events) != 2 || events[1].EventType != "training.job.leased" {
		t.Fatalf("expected the leased event, got %+v (err %v)", events, err)
	}

	// Unexpired lease blocks a second acquirer.
	if _, _, err := memory.AcquireTrainingJobLease(ctx, domain.AcquireTrainingJobLeaseInput{
		JobID: job.JobID, WorkerID: "trainer-2", TTL: time.Minute, Now: t0.Add(30 * time.Second),
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("second acquire on live lease: err = %v, want ErrConflict", err)
	}

	// Wrong token never renews.
	if _, err := memory.RenewTrainingJobLease(ctx, domain.RenewTrainingJobLeaseInput{
		JobID: job.JobID, LeaseToken: "bogus", TTL: time.Minute, Now: t0,
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("renew with wrong token: err = %v, want ErrConflict", err)
	}

	// Token match alone authorizes renewal — even of an EXPIRED lease (the
	// deliberate revival rule for expensive GPU-hours computations).
	revived, err := memory.RenewTrainingJobLease(ctx, domain.RenewTrainingJobLeaseInput{
		JobID: job.JobID, LeaseToken: lease.LeaseToken, TTL: time.Minute, Now: t0.Add(5 * time.Minute),
	})
	if err != nil {
		t.Fatalf("renew of expired lease with matching token must revive: %v", err)
	}
	if !revived.LeaseExpiresAt.Equal(t0.Add(6 * time.Minute)) {
		t.Fatalf("revived lease expiry = %v, want %v", revived.LeaseExpiresAt, t0.Add(6*time.Minute))
	}

	// Once the lease is expired, another worker can re-acquire; the old token
	// is dead afterwards.
	lease2, _, err := memory.AcquireTrainingJobLease(ctx, domain.AcquireTrainingJobLeaseInput{
		JobID: job.JobID, WorkerID: "trainer-2", TTL: time.Minute, Now: t0.Add(10 * time.Minute),
	})
	if err != nil {
		t.Fatalf("re-acquire after expiry: %v", err)
	}
	if lease2.LeaseToken == lease.LeaseToken || lease2.WorkerID != "trainer-2" {
		t.Fatalf("re-acquire must mint a fresh lease: %+v", lease2)
	}
	if _, err := memory.RenewTrainingJobLease(ctx, domain.RenewTrainingJobLeaseInput{
		JobID: job.JobID, LeaseToken: lease.LeaseToken, TTL: time.Minute, Now: t0.Add(10 * time.Minute),
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("stale token after re-acquire: err = %v, want ErrConflict", err)
	}

	// Release: wrong token conflicts, right token releases, repeat is a no-op,
	// and renewing a released lease is the authoritative not-found.
	if err := memory.ReleaseTrainingJobLease(ctx, domain.ReleaseTrainingJobLeaseInput{
		JobID: job.JobID, LeaseToken: lease.LeaseToken,
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("release with wrong token: err = %v, want ErrConflict", err)
	}
	if err := memory.ReleaseTrainingJobLease(ctx, domain.ReleaseTrainingJobLeaseInput{
		JobID: job.JobID, LeaseToken: lease2.LeaseToken,
	}); err != nil {
		t.Fatalf("release: %v", err)
	}
	if err := memory.ReleaseTrainingJobLease(ctx, domain.ReleaseTrainingJobLeaseInput{
		JobID: job.JobID, LeaseToken: lease2.LeaseToken,
	}); err != nil {
		t.Fatalf("repeat release must be a no-op: %v", err)
	}
	if _, err := memory.RenewTrainingJobLease(ctx, domain.RenewTrainingJobLeaseInput{
		JobID: job.JobID, LeaseToken: lease2.LeaseToken, TTL: time.Minute,
	}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("renew with no lease: err = %v, want ErrNotFound", err)
	}

	// Terminal jobs never lease.
	if _, err := memory.UpdateTrainingJobStatus(ctx, domain.UpdateTrainingJobStatusInput{
		JobID: job.JobID, Status: "succeeded",
	}); err != nil {
		t.Fatalf("finish job: %v", err)
	}
	if _, _, err := memory.AcquireTrainingJobLease(ctx, domain.AcquireTrainingJobLeaseInput{
		JobID: job.JobID, WorkerID: "trainer-3", TTL: time.Minute,
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("acquire on terminal job: err = %v, want ErrConflict", err)
	}
}

func TestTrainingGoldSetFreezeLifecycle(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	memory := NewMemoryStore()

	if _, err := memory.GetCurrentTrainingGoldSet(ctx, TrainingSeedModelKey); !errors.Is(err, ErrNotFound) {
		t.Fatalf("no frozen gold set yet: err = %v, want ErrNotFound", err)
	}
	if _, err := memory.CreateTrainingGoldSetDraft(ctx, domain.CreateTrainingGoldSetInput{
		ModelKey: "no-such-model",
	}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("draft for unknown model: err = %v, want ErrNotFound", err)
	}

	draft, err := memory.CreateTrainingGoldSetDraft(ctx, domain.CreateTrainingGoldSetInput{
		ModelKey:        TrainingSeedModelKey,
		CreatedByUserID: "curator-1",
	})
	if err != nil {
		t.Fatalf("CreateTrainingGoldSetDraft: %v", err)
	}
	if draft.Version != 1 || draft.Status != "draft" || draft.CreatedByUserID != "curator-1" {
		t.Fatalf("unexpected draft: %+v", draft)
	}
	if _, err := memory.CreateTrainingGoldSetDraft(ctx, domain.CreateTrainingGoldSetInput{
		ModelKey: TrainingSeedModelKey,
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("second open draft: err = %v, want ErrConflict", err)
	}

	phash := "deadbeef"
	width := int64(512)
	items := []domain.TrainingGoldItemInput{
		{
			ItemID:        "tile-002",
			SourceRef:     domain.JSONMap{"resource_id": "res-2"},
			Slice:         "held_out_test",
			LabelKind:     "boxes",
			ContentSHA256: "sha-b",
			GTLabelSHA256: "gt-b",
			GTLabelURI:    "store://gold/tile-002.txt",
		},
		{
			ItemID:        "tile-001",
			SourceRef:     domain.JSONMap{"resource_id": "res-1"},
			Slice:         "prior_train",
			LabelKind:     "boxes",
			ContentSHA256: "sha-a",
			Phash:         &phash,
			GTLabelSHA256: "gt-a",
			GTLabelURI:    "store://gold/tile-001.txt",
			Width:         &width,
			Height:        &width,
			StrataTags:    domain.JSONMap{"site": "colony-7"},
		},
	}
	count, err := memory.InsertTrainingGoldItems(ctx, draft.GoldSetID, items)
	if err != nil || count != 2 {
		t.Fatalf("InsertTrainingGoldItems = %d (err %v), want 2", count, err)
	}
	if _, err := memory.InsertTrainingGoldItems(ctx, draft.GoldSetID, []domain.TrainingGoldItemInput{{ItemID: "tile-001"}}); !errors.Is(err, ErrConflict) {
		t.Fatalf("duplicate item id: err = %v, want ErrConflict", err)
	}
	if _, err := memory.InsertTrainingGoldItems(ctx, draft.GoldSetID, []domain.TrainingGoldItemInput{{ItemID: "  "}}); err == nil {
		t.Fatal("blank item id must be rejected")
	}
	if _, err := memory.InsertTrainingGoldItems(ctx, "missing", items); !errors.Is(err, ErrNotFound) {
		t.Fatalf("items into missing gold set: err = %v, want ErrNotFound", err)
	}

	// draft can only move to freezing.
	for _, illegal := range []string{"frozen", "retired", "failed"} {
		if _, err := memory.UpdateTrainingGoldSetState(ctx, domain.UpdateTrainingGoldSetStateInput{
			GoldSetID: draft.GoldSetID,
			Status:    illegal,
		}); !errors.Is(err, ErrConflict) {
			t.Fatalf("draft->%s: err = %v, want ErrConflict", illegal, err)
		}
	}
	freezing, err := memory.UpdateTrainingGoldSetState(ctx, domain.UpdateTrainingGoldSetStateInput{
		GoldSetID: draft.GoldSetID,
		Status:    "freezing",
	})
	if err != nil || freezing.Status != "freezing" {
		t.Fatalf("draft->freezing = %+v (err %v)", freezing, err)
	}
	// freezing still accepts items (the freeze worker streams them in), and
	// still blocks a second open gold set.
	if _, err := memory.InsertTrainingGoldItems(ctx, draft.GoldSetID, []domain.TrainingGoldItemInput{{
		ItemID: "tile-003", Slice: "prior_train", LabelKind: "boxes",
		ContentSHA256: "sha-c", GTLabelSHA256: "gt-c", GTLabelURI: "store://gold/tile-003.txt",
	}}); err != nil {
		t.Fatalf("insert while freezing: %v", err)
	}
	if _, err := memory.CreateTrainingGoldSetDraft(ctx, domain.CreateTrainingGoldSetInput{
		ModelKey: TrainingSeedModelKey,
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("draft while freezing: err = %v, want ErrConflict", err)
	}

	frozen, err := memory.UpdateTrainingGoldSetState(ctx, domain.UpdateTrainingGoldSetStateInput{
		GoldSetID:        draft.GoldSetID,
		Status:           "frozen",
		ContentHash:      "hash-v1",
		ItemCount:        3,
		LabelStats:       domain.JSONMap{"prairie_dog": float64(2)},
		SplitManifestURI: "store://gold/v1/manifest.json",
	})
	if err != nil {
		t.Fatalf("freezing->frozen: %v", err)
	}
	if frozen.Status != "frozen" || frozen.ContentHash != "hash-v1" || frozen.ItemCount != 3 || frozen.FrozenAt == nil {
		t.Fatalf("frozen gold set must carry hash/count/frozen_at: %+v", frozen)
	}

	// Frozen is immutable: no items, no transitions except ->retired.
	if _, err := memory.InsertTrainingGoldItems(ctx, draft.GoldSetID, []domain.TrainingGoldItemInput{{ItemID: "tile-999"}}); !errors.Is(err, ErrConflict) {
		t.Fatalf("items into frozen gold set: err = %v, want ErrConflict", err)
	}
	for _, illegal := range []string{"freezing", "failed"} {
		if _, err := memory.UpdateTrainingGoldSetState(ctx, domain.UpdateTrainingGoldSetStateInput{
			GoldSetID: draft.GoldSetID,
			Status:    illegal,
		}); !errors.Is(err, ErrConflict) {
			t.Fatalf("frozen->%s: err = %v, want ErrConflict", illegal, err)
		}
	}

	current, err := memory.GetCurrentTrainingGoldSet(ctx, TrainingSeedModelKey)
	if err != nil || current.GoldSetID != draft.GoldSetID {
		t.Fatalf("GetCurrentTrainingGoldSet = %+v (err %v)", current, err)
	}

	// A second draft is legal once nothing is open; failed drafts retry back
	// into freezing.
	second, err := memory.CreateTrainingGoldSetDraft(ctx, domain.CreateTrainingGoldSetInput{
		ModelKey: TrainingSeedModelKey,
	})
	if err != nil || second.Version != 2 {
		t.Fatalf("second draft = %+v (err %v), want version 2", second, err)
	}
	if _, err := memory.UpdateTrainingGoldSetState(ctx, domain.UpdateTrainingGoldSetStateInput{
		GoldSetID: second.GoldSetID, Status: "freezing",
	}); err != nil {
		t.Fatalf("v2 draft->freezing: %v", err)
	}
	if _, err := memory.UpdateTrainingGoldSetState(ctx, domain.UpdateTrainingGoldSetStateInput{
		GoldSetID: second.GoldSetID, Status: "failed",
	}); err != nil {
		t.Fatalf("v2 freezing->failed: %v", err)
	}
	if _, err := memory.UpdateTrainingGoldSetState(ctx, domain.UpdateTrainingGoldSetStateInput{
		GoldSetID: second.GoldSetID, Status: "freezing",
	}); err != nil {
		t.Fatalf("v2 failed->freezing retry: %v", err)
	}
	if _, err := memory.UpdateTrainingGoldSetState(ctx, domain.UpdateTrainingGoldSetStateInput{
		GoldSetID: second.GoldSetID, Status: "frozen", ContentHash: "hash-v1",
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("content hash reuse: err = %v, want ErrConflict", err)
	}
	if _, err := memory.UpdateTrainingGoldSetState(ctx, domain.UpdateTrainingGoldSetStateInput{
		GoldSetID: second.GoldSetID, Status: "frozen", ContentHash: "hash-v2", ItemCount: 1,
	}); err != nil {
		t.Fatalf("v2 freezing->frozen: %v", err)
	}
	current, err = memory.GetCurrentTrainingGoldSet(ctx, TrainingSeedModelKey)
	if err != nil || current.GoldSetID != second.GoldSetID {
		t.Fatalf("current must be the newest frozen version: %+v (err %v)", current, err)
	}
	if _, err := memory.UpdateTrainingGoldSetState(ctx, domain.UpdateTrainingGoldSetStateInput{
		GoldSetID: draft.GoldSetID, Status: "retired",
	}); err != nil {
		t.Fatalf("frozen->retired: %v", err)
	}

	sets, err := memory.ListTrainingGoldSets(ctx, TrainingSeedModelKey, 0)
	if err != nil || len(sets) != 2 || sets[0].Version != 2 || sets[1].Version != 1 {
		t.Fatalf("ListTrainingGoldSets must be version DESC: %+v (err %v)", sets, err)
	}

	listed, err := memory.ListTrainingGoldItems(ctx, draft.GoldSetID, 0, 0)
	if err != nil || len(listed) != 3 {
		t.Fatalf("ListTrainingGoldItems = %d (err %v), want 3", len(listed), err)
	}
	if listed[0].ItemID != "tile-001" || listed[0].Phash == nil || *listed[0].Phash != "deadbeef" {
		t.Fatalf("items must be item_id ASC with pointers intact: %+v", listed[0])
	}
	if paged, err := memory.ListTrainingGoldItems(ctx, draft.GoldSetID, 1, 1); err != nil || len(paged) != 1 || paged[0].ItemID != "tile-002" {
		t.Fatalf("paged items = %+v (err %v), want tile-002", paged, err)
	}
}

func TestTrainingModelVersionLifecycleAndAudit(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	memory := NewMemoryStore()

	if _, err := memory.CreateTrainingModelVersion(ctx, domain.CreateTrainingModelVersionInput{
		LineageID: "no-such-lineage",
		ModelKey:  TrainingSeedModelKey,
	}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("unknown lineage: err = %v, want ErrNotFound", err)
	}
	if _, err := memory.CreateTrainingModelVersion(ctx, domain.CreateTrainingModelVersionInput{
		LineageID: TrainingSeedLineageID,
		ModelKey:  TrainingSeedModelKey,
		Status:    "shipped",
	}); err == nil {
		t.Fatal("invalid version status must be rejected")
	}

	version, err := memory.CreateTrainingModelVersion(ctx, domain.CreateTrainingModelVersionInput{
		VersionID:   "yolov5_rarespot-v1",
		LineageID:   TrainingSeedLineageID,
		ModelKey:    TrainingSeedModelKey,
		WeightsURI:  "store://training/yolov5_rarespot/v1/best.pt",
		SourceJobID: "job-finetune-1",
		Metadata:    domain.JSONMap{"provenance": "finetune run 1"},
	})
	if err != nil {
		t.Fatalf("CreateTrainingModelVersion: %v", err)
	}
	if version.Status != "candidate" || version.IsFrozen || version.ActivatedAt != nil {
		t.Fatalf("new version must default to an unactivated candidate: %+v", version)
	}
	if _, err := memory.CreateTrainingModelVersion(ctx, domain.CreateTrainingModelVersionInput{
		VersionID: "yolov5_rarespot-v1",
		LineageID: TrainingSeedLineageID,
		ModelKey:  TrainingSeedModelKey,
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("duplicate version id: err = %v, want ErrConflict", err)
	}

	activatedAt := domain.Now()
	updated, err := memory.UpdateTrainingModelVersion(ctx, domain.UpdateTrainingModelVersionInput{
		VersionID:   version.VersionID,
		Status:      "active",
		Metrics:     domain.JSONMap{"map50": 0.61},
		ActivatedAt: &activatedAt,
	})
	if err != nil {
		t.Fatalf("UpdateTrainingModelVersion: %v", err)
	}
	if updated.Status != "active" || updated.Metrics["map50"] != 0.61 || updated.ActivatedAt == nil {
		t.Fatalf("update must replace status/metrics and set activated_at: %+v", updated)
	}
	if updated.Metadata["provenance"] != "finetune run 1" {
		t.Fatalf("nil metadata must keep the existing metadata: %+v", updated)
	}
	if updated.UpdatedAt.Before(version.UpdatedAt) {
		t.Fatalf("updated_at must be stamped: %v then %v", version.UpdatedAt, updated.UpdatedAt)
	}
	if _, err := memory.UpdateTrainingModelVersion(ctx, domain.UpdateTrainingModelVersionInput{
		VersionID: "missing", Status: "retired",
	}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("update missing version: err = %v, want ErrNotFound", err)
	}

	first, err := memory.AppendTrainingModelVersionEvent(ctx, domain.AppendTrainingModelVersionEventInput{
		VersionID:   version.VersionID,
		EventType:   "benchmarked",
		ActorUserID: "system",
		FromStatus:  "candidate",
		ToStatus:    "candidate",
	})
	if err != nil || first.EventID == "" || first.CreatedAt.IsZero() {
		t.Fatalf("AppendTrainingModelVersionEvent = %+v (err %v)", first, err)
	}
	second, err := memory.AppendTrainingModelVersionEvent(ctx, domain.AppendTrainingModelVersionEventInput{
		VersionID:          version.VersionID,
		EventType:          "promoted",
		ActorUserID:        "amil",
		FromStatus:         "candidate",
		ToStatus:           "active",
		BenchmarkRunID:     "benchmark-1",
		GoldSetContentHash: "hash-v1",
		Reason:             "beat active on the frozen gold set",
	})
	if err != nil {
		t.Fatalf("append second version event: %v", err)
	}
	events, err := memory.ListTrainingModelVersionEvents(ctx, version.VersionID, 0)
	if err != nil || len(events) != 2 {
		t.Fatalf("ListTrainingModelVersionEvents = %d (err %v), want 2", len(events), err)
	}
	if events[0].EventID != second.EventID {
		t.Fatalf("version events must list newest first: got %+v", events)
	}
	if events[0].BenchmarkRunID != "benchmark-1" || events[0].GoldSetContentHash != "hash-v1" {
		t.Fatalf("audit fields must round-trip: %+v", events[0])
	}
}

func TestTrainingRetrainRequestLifecycle(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	memory := NewMemoryStore()

	if _, err := memory.CreateTrainingRetrainRequest(ctx, domain.CreateTrainingRetrainRequestInput{
		ModelKey: "no-such-model",
	}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("retrain for unknown model: err = %v, want ErrNotFound", err)
	}
	request, err := memory.CreateTrainingRetrainRequest(ctx, domain.CreateTrainingRetrainRequestInput{
		ModelKey:          TrainingSeedModelKey,
		Note:              "gate opened",
		RequestedByUserID: "amil",
		GatingSummary:     domain.JSONMap{"reviewed": float64(64)},
	})
	if err != nil {
		t.Fatalf("CreateTrainingRetrainRequest: %v", err)
	}
	if request.RequestID == "" || request.Status != "queued" || request.GatingSummary["reviewed"] != float64(64) {
		t.Fatalf("unexpected retrain request: %+v", request)
	}

	startedAt := domain.Now()
	running, err := memory.UpdateTrainingRetrainRequest(ctx, domain.UpdateTrainingRetrainRequestInput{
		RequestID:     request.RequestID,
		Status:        "running",
		TrainingJobID: "job-finetune-9",
		StartedAt:     &startedAt,
	})
	if err != nil {
		t.Fatalf("update retrain to running: %v", err)
	}
	if running.Status != "running" || running.TrainingJobID != "job-finetune-9" || running.StartedAt == nil {
		t.Fatalf("unexpected running retrain request: %+v", running)
	}
	finishedAt := domain.Now()
	done, err := memory.UpdateTrainingRetrainRequest(ctx, domain.UpdateTrainingRetrainRequestInput{
		RequestID:    request.RequestID,
		Status:       "succeeded",
		ModelVersion: "yolov5_rarespot-v1",
		FinishedAt:   &finishedAt,
	})
	if err != nil {
		t.Fatalf("update retrain to succeeded: %v", err)
	}
	if done.Status != "succeeded" || done.ModelVersion != "yolov5_rarespot-v1" || done.FinishedAt == nil || done.StartedAt == nil {
		t.Fatalf("unexpected finished retrain request: %+v", done)
	}
	if _, err := memory.UpdateTrainingRetrainRequest(ctx, domain.UpdateTrainingRetrainRequestInput{
		RequestID: "missing", Status: "failed",
	}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("update missing retrain request: err = %v, want ErrNotFound", err)
	}

	listed, err := memory.ListTrainingRetrainRequests(ctx, TrainingSeedModelKey, 0)
	if err != nil || len(listed) != 1 || listed[0].RequestID != request.RequestID {
		t.Fatalf("ListTrainingRetrainRequests = %+v (err %v)", listed, err)
	}
}

func TestTrainingBenchmarkRunsAndLatestByHash(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	memory := NewMemoryStore()

	if _, err := memory.InsertTrainingBenchmarkRun(ctx, domain.InsertTrainingBenchmarkRunInput{
		GoldSetID: "gold-1",
	}); err == nil {
		t.Fatal("missing model version id must be rejected")
	}
	if _, err := memory.InsertTrainingBenchmarkRun(ctx, domain.InsertTrainingBenchmarkRunInput{
		ModelVersionID: "yolov5_rarespot-v1",
	}); err == nil {
		t.Fatal("missing gold set id must be rejected")
	}

	first, err := memory.InsertTrainingBenchmarkRun(ctx, domain.InsertTrainingBenchmarkRunInput{
		RunID:              "benchmark-1",
		ModelVersionID:     "yolov5_rarespot-v1",
		GoldSetID:          "gold-1",
		GoldSetContentHash: "hash-a",
		MetricSchema:       "detection.v1",
		KernelVersion:      "bench-kernel-1",
		Metrics:            domain.JSONMap{"map50": 0.58},
		GuardrailsPassed:   false,
		GuardrailsReasons:  []string{"agg_map50: dropped 0.02 vs active"},
		ReportURI:          "store://reports/benchmark-1.json",
	})
	if err != nil {
		t.Fatalf("InsertTrainingBenchmarkRun: %v", err)
	}
	if first.CreatedAt.IsZero() || first.GuardrailsPassed || len(first.GuardrailsReasons) != 1 {
		t.Fatalf("unexpected benchmark run: %+v", first)
	}
	if _, err := memory.InsertTrainingBenchmarkRun(ctx, domain.InsertTrainingBenchmarkRunInput{
		RunID:          "benchmark-1",
		ModelVersionID: "yolov5_rarespot-v1",
		GoldSetID:      "gold-1",
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("duplicate run id: err = %v, want ErrConflict", err)
	}
	second, err := memory.InsertTrainingBenchmarkRun(ctx, domain.InsertTrainingBenchmarkRunInput{
		RunID:              "benchmark-2",
		ModelVersionID:     "yolov5_rarespot-v1",
		GoldSetID:          "gold-2",
		GoldSetContentHash: "hash-b",
		MetricSchema:       "detection.v1",
		KernelVersion:      "bench-kernel-1",
		Metrics:            domain.JSONMap{"map50": 0.63},
		GuardrailsPassed:   true,
		ReportURI:          "store://reports/benchmark-2.json",
	})
	if err != nil {
		t.Fatalf("insert second benchmark run: %v", err)
	}

	byHash, err := memory.GetLatestTrainingBenchmarkRun(ctx, "yolov5_rarespot-v1", "hash-a")
	if err != nil || byHash.RunID != first.RunID {
		t.Fatalf("latest by hash = %+v (err %v), want benchmark-1", byHash, err)
	}
	latest, err := memory.GetLatestTrainingBenchmarkRun(ctx, "yolov5_rarespot-v1", "")
	if err != nil || latest.RunID != second.RunID {
		t.Fatalf("latest any hash = %+v (err %v), want benchmark-2", latest, err)
	}
	if _, err := memory.GetLatestTrainingBenchmarkRun(ctx, "yolov5_rarespot-v1", "hash-z"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("unknown hash: err = %v, want ErrNotFound", err)
	}
	if _, err := memory.GetLatestTrainingBenchmarkRun(ctx, "missing-version", ""); !errors.Is(err, ErrNotFound) {
		t.Fatalf("unknown version: err = %v, want ErrNotFound", err)
	}
}

func TestTrainingModelStatusUpsert(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	memory := NewMemoryStore()

	if _, err := memory.UpsertTrainingModelStatus(ctx, domain.TrainingModelStatusRecord{
		ModelKey: "no-such-model",
	}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("status for unknown model: err = %v, want ErrNotFound", err)
	}

	syncedAt := domain.Now()
	updated, err := memory.UpsertTrainingModelStatus(ctx, domain.TrainingModelStatusRecord{
		ModelKey:               TrainingSeedModelKey,
		DatasetName:            "Prairie_Dog_Active_Learning",
		ModelHealth:            "healthy",
		ReviewedImages:         64,
		UnreviewedImages:       12,
		ClassCounts:            domain.JSONMap{"prairie_dog": float64(40), "burrow": float64(24)},
		PerClassNewObjects:     domain.JSONMap{"prairie_dog": float64(22)},
		UnsupportedClassCounts: domain.JSONMap{},
		LastSyncAt:             &syncedAt,
		ActiveModelVersion:     TrainingSeedVersionID,
		RetrainGate:            true,
		RetrainGateReasons:     []string{},
		RetrainGateCounts:      domain.JSONMap{"reviewed": float64(64)},
		RetrainGateThresholds:  domain.JSONMap{"min_reviewed": float64(50)},
	})
	if err != nil {
		t.Fatalf("UpsertTrainingModelStatus: %v", err)
	}
	if updated.ReviewedImages != 64 || !updated.RetrainGate || updated.LastSyncAt == nil {
		t.Fatalf("unexpected upserted status: %+v", updated)
	}
	got, err := memory.GetTrainingModelStatus(ctx, TrainingSeedModelKey)
	if err != nil {
		t.Fatalf("GetTrainingModelStatus after upsert: %v", err)
	}
	if got.ModelHealth != "healthy" || got.ReviewedImages != 64 || got.ClassCounts["burrow"] != float64(24) {
		t.Fatalf("upsert must replace the full row: %+v", got)
	}
	if len(got.RetrainGateReasons) != 0 {
		t.Fatalf("gate reasons must be replaced, got %+v", got.RetrainGateReasons)
	}

	// Second upsert overwrites again (the sync worker is the single writer).
	if _, err := memory.UpsertTrainingModelStatus(ctx, domain.TrainingModelStatusRecord{
		ModelKey:           TrainingSeedModelKey,
		ModelHealth:        "watch",
		RetrainGate:        false,
		RetrainGateReasons: []string{"below min_reviewed"},
	}); err != nil {
		t.Fatalf("second upsert: %v", err)
	}
	got, err = memory.GetTrainingModelStatus(ctx, TrainingSeedModelKey)
	if err != nil || got.ModelHealth != "watch" || got.RetrainGate || len(got.RetrainGateReasons) != 1 {
		t.Fatalf("second upsert must win: %+v (err %v)", got, err)
	}
}

func TestTrainingGuardrailClausesAndGatePolicySeeded(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	memory := NewMemoryStore()
	fixture := loadTrainingContractFixture(t)

	clauses, err := memory.ListTrainingGuardrailClauses(ctx, TrainingSeedModelKey)
	if err != nil {
		t.Fatalf("ListTrainingGuardrailClauses: %v", err)
	}
	if len(clauses) != len(fixture.Seed.GuardrailClauseKeys) {
		t.Fatalf("seeded clauses = %d, fixture pins %d", len(clauses), len(fixture.Seed.GuardrailClauseKeys))
	}
	byKey := map[string]domain.TrainingGuardrailClauseRecord{}
	for _, clause := range clauses {
		if !clause.Enabled || !clause.Required {
			t.Fatalf("seed clauses are all enabled+required: %+v", clause)
		}
		byKey[clause.ClauseKey] = clause
	}
	for _, key := range fixture.Seed.GuardrailClauseKeys {
		if _, ok := byKey[key]; !ok {
			t.Fatalf("seeded clauses are missing fixture key %q", key)
		}
	}
	held := byKey["slice_held_map50"]
	if held.MetricPath != "per_slice.held_out_test.map50" || held.Comparator != "max_drop_vs_active" ||
		held.Value != 0.005 || held.Slice != "held_out_test" || held.Params["min_label_count"] != float64(10) {
		t.Fatalf("slice_held_map50 clause drifted from the schema seed: %+v", held)
	}
	if strict := byKey["class_ap50_abs"]; strict.Comparator != "abs_floor" || strict.Value != 0.10 || strict.Params["strict"] != true {
		t.Fatalf("class_ap50_abs clause drifted from the schema seed: %+v", strict)
	}
	// Clone-on-return: caller mutations must never write through to the seed.
	held.Params["min_label_count"] = float64(999)
	reread, err := memory.ListTrainingGuardrailClauses(ctx, TrainingSeedModelKey)
	if err != nil {
		t.Fatalf("re-list clauses: %v", err)
	}
	for _, clause := range reread {
		if clause.ClauseKey == "slice_held_map50" && clause.Params["min_label_count"] != float64(10) {
			t.Fatalf("caller mutation leaked into the seed: %+v", clause)
		}
	}
	if none, err := memory.ListTrainingGuardrailClauses(ctx, "no-such-model"); err != nil || len(none) != 0 {
		t.Fatalf("clauses for unknown model = %+v (err %v), want empty", none, err)
	}

	policy, err := memory.GetTrainingGatePolicy(ctx, TrainingSeedModelKey)
	if err != nil {
		t.Fatalf("GetTrainingGatePolicy: %v", err)
	}
	if int(policy.MinReviewed) != fixture.Seed.GatePolicy.MinReviewed ||
		int(policy.MinNewObjects) != fixture.Seed.GatePolicy.MinNewObjects ||
		int(policy.MinDays) != fixture.Seed.GatePolicy.MinDays {
		t.Fatalf("gate policy %+v does not match fixture %+v", policy, fixture.Seed.GatePolicy)
	}
	if len(policy.MinPerClassObjects) != len(fixture.Seed.GatePolicy.MinPerClassObjects) {
		t.Fatalf("min_per_class_objects key count: seed %d != fixture %d", len(policy.MinPerClassObjects), len(fixture.Seed.GatePolicy.MinPerClassObjects))
	}
	for name, want := range fixture.Seed.GatePolicy.MinPerClassObjects {
		if got, _ := policy.MinPerClassObjects[name].(float64); int(got) != want {
			t.Fatalf("min_per_class_objects[%s]: seed %v != fixture %d", name, policy.MinPerClassObjects[name], want)
		}
	}
	if _, err := memory.GetTrainingGatePolicy(ctx, "no-such-model"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("policy for unknown model: err = %v, want ErrNotFound", err)
	}
}
