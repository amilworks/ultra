package app

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/config"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
)

func TestNewAppServesHealth(t *testing.T) {
	t.Parallel()
	application, err := New(config.Config{AppVersion: "test-version"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	req := httptest.NewRequest(http.MethodGet, "/v1/health", nil)
	rec := httptest.NewRecorder()
	application.Handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
}

func TestNewRejectsProductionWithoutDurableBackends(t *testing.T) {
	t.Parallel()
	_, err := New(config.Config{Environment: "production", AppVersion: "test-version"})
	if err == nil {
		t.Fatalf("New() error = nil, want production durability validation error")
	}
}

func TestNewRejectsUnreachablePostgresBeforeServing(t *testing.T) {
	t.Parallel()

	_, err := New(config.Config{
		AppVersion:  "test-version",
		DatabaseURL: "postgresql://postgres:postgres@127.0.0.1:1/ultra?connect_timeout=1",
	})

	if err == nil {
		t.Fatalf("New() error = nil, want unreachable Postgres to fail before serving")
	}
	if !strings.Contains(strings.ToLower(err.Error()), "postgres") {
		t.Fatalf("New() error = %q, want Postgres context", err)
	}
}

func TestAppStartRecoversExpiredRunLeases(t *testing.T) {
	t.Parallel()
	application, err := New(config.Config{
		AppVersion:                "test-version",
		RunRecoveryEnabled:        true,
		RunRecoveryInterval:       10 * time.Millisecond,
		RunRecoveryBatchLimit:     10,
		NATSJobsSubject:           "ultra.runs.jobs",
		NATSRareSpotJobsSubject:   "ultra.runs.rarespot.jobs",
		NATSEventsSubject:         "ultra.runs.events",
		NATSCancelSubject:         "ultra.runs.cancel",
		NATSEventConsumer:         "ultra-control-event-ingest",
		NATSWorkerDurable:         "ultra-deepagents-worker",
		NATSRareSpotWorkerDurable: "rarespot-ecology-worker",
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if application.Start == nil {
		t.Fatalf("Start hook should be configured when run recovery is enabled")
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	thread, err := application.Runs.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "user-1",
		Title:  "auto recovery",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := application.Runs.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "recover me",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "recover me"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	select {
	case <-application.JobSource:
	case <-time.After(time.Second):
		t.Fatalf("expected initial job")
	}
	now := time.Date(2026, 6, 1, 12, 0, 0, 0, time.UTC)
	if _, err := application.Store.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    run.RunID,
		WorkerID: "worker-expired",
		TTL:      time.Minute,
		Now:      now.Add(-2 * time.Minute),
	}); err != nil {
		t.Fatalf("AcquireRunLease: %v", err)
	}
	if err := application.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	select {
	case job := <-application.JobSource:
		if job.RunID != run.RunID || job.DispatchID == "" {
			t.Fatalf("recovery job = %+v, want fresh dispatch for %s", job, run.RunID)
		}
	case <-time.After(time.Second):
		t.Fatalf("expected recovery job")
	}
}

func TestMigratePostgresRequiresDatabaseURL(t *testing.T) {
	t.Parallel()

	err := MigratePostgres(context.Background(), config.Config{})

	if err == nil {
		t.Fatalf("MigratePostgres() error = nil, want database URL validation error")
	}
	if !strings.Contains(err.Error(), "ULTRA_CONTROL_DATABASE_URL") {
		t.Fatalf("MigratePostgres() error = %q, want database URL guidance", err)
	}
}

func TestMigratePostgresRejectsUnreachablePostgres(t *testing.T) {
	t.Parallel()

	err := MigratePostgres(context.Background(), config.Config{
		DatabaseURL: "postgresql://postgres:postgres@127.0.0.1:1/ultra?connect_timeout=1",
	})

	if err == nil {
		t.Fatalf("MigratePostgres() error = nil, want unreachable Postgres to fail")
	}
	if !strings.Contains(strings.ToLower(err.Error()), "postgres") {
		t.Fatalf("MigratePostgres() error = %q, want Postgres context", err)
	}
}
