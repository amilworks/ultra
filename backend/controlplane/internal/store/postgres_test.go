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

func TestPostgresStoreTenantScopedQueriesFilterByUser(t *testing.T) {
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
	suffix := domain.NewID("tenant")
	aliceID := "alice-" + suffix
	bobID := "bob-" + suffix
	aliceThread, err := store.CreateThread(ctx, domain.CreateThreadInput{
		UserID:          aliceID,
		Title:           "Alice Postgres thread",
		InitialMessages: []domain.ThreadMessage{{Role: "user", Content: "alice message"}},
	})
	if err != nil {
		t.Fatalf("CreateThread alice: %v", err)
	}
	aliceRun, err := store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: aliceThread.ThreadID,
		UserID:   aliceID,
		Goal:     "alice postgres run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "alice postgres run"}},
	})
	if err != nil {
		t.Fatalf("CreateRun alice: %v", err)
	}
	if _, err := store.AppendRunEvent(ctx, domain.AppendRunEventInput{
		RunID:     aliceRun.RunID,
		ThreadID:  aliceThread.ThreadID,
		EventKind: "message.delta",
		Message:   "alice trace",
	}); err != nil {
		t.Fatalf("AppendRunEvent alice: %v", err)
	}
	aliceArtifact, err := store.CreateArtifact(ctx, domain.CreateArtifactInput{
		RunID:    aliceRun.RunID,
		ThreadID: aliceThread.ThreadID,
		Kind:     "report",
		Path:     "alice.md",
	})
	if err != nil {
		t.Fatalf("CreateArtifact alice: %v", err)
	}
	bobThread, err := store.CreateThread(ctx, domain.CreateThreadInput{UserID: bobID, Title: "Bob Postgres thread"})
	if err != nil {
		t.Fatalf("CreateThread bob: %v", err)
	}
	if _, err := store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: bobThread.ThreadID,
		UserID:   bobID,
		Goal:     "bob postgres run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "bob postgres run"}},
	}); err != nil {
		t.Fatalf("CreateRun bob: %v", err)
	}

	alicePage, err := store.ListThreadsForUser(ctx, aliceID, 10, 0, "")
	if err != nil {
		t.Fatalf("ListThreadsForUser alice: %v", err)
	}
	if alicePage.TotalCount != 1 || len(alicePage.Threads) != 1 || alicePage.Threads[0].ThreadID != aliceThread.ThreadID {
		t.Fatalf("alice threads = %+v, want only alice thread", alicePage)
	}
	if _, err := store.GetThreadForUser(ctx, aliceThread.ThreadID, bobID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetThreadForUser bob err = %v, want ErrNotFound", err)
	}
	if _, err := store.ListThreadMessagesForUser(ctx, aliceThread.ThreadID, bobID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("ListThreadMessagesForUser bob err = %v, want ErrNotFound", err)
	}
	if _, err := store.GetRunForUser(ctx, aliceRun.RunID, bobID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetRunForUser bob err = %v, want ErrNotFound", err)
	}
	bobRuns, err := store.ListRunsForUser(ctx, bobID, "", "", 10, 0)
	if err != nil {
		t.Fatalf("ListRunsForUser bob: %v", err)
	}
	if len(bobRuns) != 1 || bobRuns[0].UserID != bobID {
		t.Fatalf("bob runs = %+v, want only bob run", bobRuns)
	}
	if _, err := store.ListRunEventsForUser(ctx, aliceRun.RunID, bobID, 10); !errors.Is(err, ErrNotFound) {
		t.Fatalf("ListRunEventsForUser bob err = %v, want ErrNotFound", err)
	}
	if _, err := store.ListRunArtifactsForUser(ctx, aliceRun.RunID, bobID, 10); !errors.Is(err, ErrNotFound) {
		t.Fatalf("ListRunArtifactsForUser bob err = %v, want ErrNotFound", err)
	}
	if _, err := store.GetArtifactForUser(ctx, aliceArtifact.ArtifactID, bobID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetArtifactForUser bob err = %v, want ErrNotFound", err)
	}
}

func TestPostgresStoreResourceCatalogFiltersSoftDeletesAndRestores(t *testing.T) {
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
	suffix := domain.NewID("resource")
	resourceID := "file_alice_" + suffix
	bobResourceID := "file_bob_" + suffix
	if _, err := store.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   resourceID,
		OwnerUserID:  "alice-" + suffix,
		OwnerOrgID:   "org-a",
		OwnerRole:    "researcher",
		OriginalName: "cells.ome.tiff",
		ContentType:  "image/tiff",
		SizeBytes:    128,
		SHA256:       "abc123",
		StorageURI:   "file:///srv/ultra/shared/uploads/" + resourceID + "__cells.ome.tiff",
		StoragePath:  resourceID + "__cells.ome.tiff",
		SourceType:   "upload",
		ResourceKind: "image",
		ProjectID:    "project-ct",
		Status:       "active",
	}); err != nil {
		t.Fatalf("UpsertResource alice: %v", err)
	}
	if _, err := store.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   bobResourceID,
		OwnerUserID:  "bob-" + suffix,
		OwnerOrgID:   "org-a",
		OriginalName: "other.csv",
		SizeBytes:    64,
		SourceType:   "upload",
		ResourceKind: "table",
		Status:       "active",
	}); err != nil {
		t.Fatalf("UpsertResource bob: %v", err)
	}

	page, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID:    "alice-" + suffix,
		OrgID:     "org-a",
		Query:     "ome",
		Kind:      "image",
		Source:    "upload",
		ProjectID: "project-ct",
		Limit:     20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser alice: %v", err)
	}
	if page.TotalCount != 1 || len(page.Resources) != 1 || page.Resources[0].ResourceID != resourceID {
		t.Fatalf("alice resources = %+v, want only alice image", page)
	}
	if page.Resources[0].ProjectID != "project-ct" {
		t.Fatalf("alice project_id = %q, want project-ct", page.Resources[0].ProjectID)
	}
	wrongProject, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: "alice-" + suffix, OrgID: "org-a", ProjectID: "project-other", Limit: 20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser wrong project: %v", err)
	}
	if wrongProject.TotalCount != 0 || len(wrongProject.Resources) != 0 {
		t.Fatalf("wrong project resources = %+v, want none", wrongProject)
	}
	if _, err := store.GetResourceForUser(ctx, resourceID, "bob-"+suffix, "org-a"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetResourceForUser bob err = %v, want ErrNotFound", err)
	}
	deleted, err := store.SoftDeleteResourceForUser(ctx, resourceID, "alice-"+suffix, "org-a", time.Now())
	if err != nil {
		t.Fatalf("SoftDeleteResourceForUser: %v", err)
	}
	if deleted.Status != "deleted" || deleted.DeletedAt.IsZero() || deleted.RetentionExpiresAt.IsZero() {
		t.Fatalf("deleted = %+v, want deleted status with retention expiry", deleted)
	}
	deletedPage, err := store.ListResourcesForUser(ctx, domain.ResourceListInput{UserID: "alice-" + suffix, OrgID: "org-a", Limit: 20})
	if err != nil {
		t.Fatalf("ListResourcesForUser after delete: %v", err)
	}
	if deletedPage.TotalCount != 0 || len(deletedPage.Resources) != 0 {
		t.Fatalf("deleted resources = %+v, want no active rows", deletedPage)
	}
	restored, err := store.RestoreResourceForUser(ctx, resourceID, "alice-"+suffix, "org-a", time.Now())
	if err != nil {
		t.Fatalf("RestoreResourceForUser: %v", err)
	}
	if restored.Status != "active" || !restored.DeletedAt.IsZero() || !restored.RetentionExpiresAt.IsZero() {
		t.Fatalf("restored = %+v, want active with empty retention fields", restored)
	}
	if _, err := store.CreateResourceEvent(ctx, domain.AppendResourceEventInput{
		ResourceID:  resourceID,
		ActorUserID: "alice-" + suffix,
		ActorOrgID:  "org-a",
		EventType:   "resource.restored",
	}); err != nil {
		t.Fatalf("CreateResourceEvent: %v", err)
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
