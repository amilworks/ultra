package store

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresStoreThreadRunEventArtifactFlow(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	store := NewPostgresStore(pool)
	thread, err := store.CreateThread(ctx, domain.CreateThreadInput{
		UserID: "pg-user",
		Title:  "Postgres flow",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   "pg-user",
		Goal:     "persist run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "persist run"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	event, err := store.AppendRunEvent(ctx, domain.AppendRunEventInput{
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.accepted",
		Message:   "accepted",
		Payload:   domain.JSONMap{"ok": true},
	})
	if err != nil {
		t.Fatalf("AppendRunEvent: %v", err)
	}
	if event.Sequence < 1 {
		t.Fatalf("event sequence = %d, want >= 1", event.Sequence)
	}
}

func TestPostgresStoreRejectsDuplicateRunIdempotencyKey(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	store := NewPostgresStore(pool)
	userID := "pg-idempotency-" + domain.NewID("test")
	thread, err := store.CreateThread(ctx, domain.CreateThreadInput{
		UserID: userID,
		Title:  "Postgres idempotency",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	metadata := domain.JSONMap{"idempotency_key": "same-submit-key"}
	_, err = store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   userID,
		Goal:     "first run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "first run"}},
		Metadata: metadata,
	})
	if err != nil {
		t.Fatalf("CreateRun first: %v", err)
	}
	_, err = store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   userID,
		Goal:     "duplicate run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "duplicate run"}},
		Metadata: metadata,
	})
	if !errors.Is(err, ErrConflict) {
		t.Fatalf("CreateRun duplicate err = %v, want ErrConflict", err)
	}
}

func TestPostgresStoreCompleteRunRepairsSucceededRunMissingResponseText(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	store := NewPostgresStore(pool)
	userID := "pg-terminal-repair-" + domain.NewID("test")
	thread, err := store.CreateThread(ctx, domain.CreateThreadInput{
		UserID: userID,
		Title:  "Postgres terminal repair",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   userID,
		Goal:     "repair terminal response",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "repair terminal response"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	if _, err := store.UpdateRunStatus(ctx, run.RunID, domain.RunStatusSucceeded, "", ""); err != nil {
		t.Fatalf("UpdateRunStatus succeeded with empty response: %v", err)
	}

	repaired, err := store.CompleteRun(ctx, domain.CompleteRunInput{
		RunID:        run.RunID,
		ResponseText: "Recovered Postgres final answer.",
	})
	if err != nil {
		t.Fatalf("CompleteRun repair: %v", err)
	}
	if repaired.Status != domain.RunStatusSucceeded || repaired.ResponseText != "Recovered Postgres final answer." {
		t.Fatalf("repaired run = %+v, want succeeded with recovered response text", repaired)
	}
	messages, err := store.ListThreadMessages(ctx, thread.ThreadID)
	if err != nil {
		t.Fatalf("ListThreadMessages: %v", err)
	}
	if got, want := len(messages), 2; got != want {
		t.Fatalf("messages = %d, want %d user+assistant messages: %+v", got, want, messages)
	}
	if messages[1].Role != "assistant" || messages[1].Content != "Recovered Postgres final answer." || messages[1].RunID != run.RunID {
		t.Fatalf("assistant message = %+v, want recovered response owned by run", messages[1])
	}
}

func TestPostgresStoreClearRunLeaseEvictsAnyActiveToken(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	store := NewPostgresStore(pool)
	userID := "pg-clear-lease-" + domain.NewID("test")
	thread, err := store.CreateThread(ctx, domain.CreateThreadInput{
		UserID: userID,
		Title:  "Postgres clear lease",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   userID,
		Goal:     "recover leased run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "recover leased run"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	now := time.Date(2026, 6, 1, 12, 0, 0, 0, time.UTC)
	lease, err := store.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    run.RunID,
		WorkerID: "worker-a",
		TTL:      time.Hour,
		Now:      now,
	})
	if err != nil {
		t.Fatalf("AcquireRunLease: %v", err)
	}

	cleared, ok, err := store.ClearRunLease(ctx, run.RunID)
	if err != nil {
		t.Fatalf("ClearRunLease: %v", err)
	}
	if !ok || cleared.LeaseToken != lease.LeaseToken || cleared.WorkerID != "worker-a" {
		t.Fatalf("cleared lease = %+v ok=%v, want worker-a lease", cleared, ok)
	}
	if _, err := store.RenewRunLease(ctx, domain.RenewRunLeaseInput{
		RunID:      run.RunID,
		LeaseToken: lease.LeaseToken,
		TTL:        time.Hour,
		Now:        now.Add(time.Minute),
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("RenewRunLease after clear err = %v, want ErrConflict", err)
	}
	if _, err := store.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    run.RunID,
		WorkerID: "worker-b",
		TTL:      time.Hour,
		Now:      now.Add(time.Minute),
	}); err != nil {
		t.Fatalf("AcquireRunLease replacement: %v", err)
	}
}

func TestPostgresStoreCreateAndListOrganizations(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	store := NewPostgresStore(pool)
	orgID := "pg-org-" + domain.NewID("test")
	org, err := store.CreateOrganization(ctx, domain.CreateOrganizationInput{
		OrgID:    orgID,
		Name:     "Postgres Organization",
		Status:   "active",
		Metadata: domain.JSONMap{"source": "postgres_test"},
	})
	if err != nil {
		t.Fatalf("CreateOrganization: %v", err)
	}
	if org.OrgID != orgID || org.Name != "Postgres Organization" {
		t.Fatalf("organization = %+v, want persisted org", org)
	}

	orgs, err := store.ListOrganizations(ctx, 10, orgID)
	if err != nil {
		t.Fatalf("ListOrganizations: %v", err)
	}
	if len(orgs) != 1 || orgs[0].OrgID != orgID {
		t.Fatalf("organizations = %+v, want created org", orgs)
	}
}

func TestPostgresStoreUpsertsAndListsWorkerHeartbeats(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	store := NewPostgresStore(pool)
	workerID := "pg-worker-" + domain.NewID("test")
	started := time.Date(2026, 6, 1, 12, 0, 0, 0, time.UTC)
	firstBeat := started.Add(10 * time.Second)
	secondBeat := started.Add(70 * time.Second)
	if _, err := store.UpsertWorkerHeartbeat(ctx, domain.UpsertWorkerHeartbeatInput{
		WorkerID:        workerID,
		WorkerKind:      "deepagents",
		Status:          "idle",
		Hostname:        "pg-host",
		Version:         "v1",
		StartedAt:       started,
		LastHeartbeatAt: firstBeat,
		Metadata:        domain.JSONMap{"durable": "ultra-deepagents-worker"},
	}); err != nil {
		t.Fatalf("UpsertWorkerHeartbeat first: %v", err)
	}
	updated, err := store.UpsertWorkerHeartbeat(ctx, domain.UpsertWorkerHeartbeatInput{
		WorkerID:        workerID,
		WorkerKind:      "deepagents",
		Status:          "busy",
		CurrentRunID:    "run_123",
		Hostname:        "pg-host",
		Version:         "v2",
		LastHeartbeatAt: secondBeat,
		Metadata:        domain.JSONMap{"active_tasks": float64(1)},
	})
	if err != nil {
		t.Fatalf("UpsertWorkerHeartbeat second: %v", err)
	}
	if updated.Status != "busy" || updated.CurrentRunID != "run_123" || updated.Version != "v2" {
		t.Fatalf("updated worker = %+v, want busy worker", updated)
	}
	if !updated.StartedAt.Equal(started) {
		t.Fatalf("started_at = %s, want original %s", updated.StartedAt, started)
	}
	workers, err := store.ListWorkerHeartbeats(ctx, 100)
	if err != nil {
		t.Fatalf("ListWorkerHeartbeats: %v", err)
	}
	found := false
	for _, worker := range workers {
		if worker.WorkerID != workerID {
			continue
		}
		found = true
		if worker.Status != "busy" || worker.CurrentRunID != "run_123" || worker.Metadata["active_tasks"] != float64(1) {
			t.Fatalf("listed worker = %+v, want updated worker", worker)
		}
	}
	if !found {
		t.Fatalf("workers = %+v, want %s", workers, workerID)
	}
}
