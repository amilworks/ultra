package app

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/config"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
	"github.com/jackc/pgx/v5/pgxpool"
)

type flakyDataAgentJobPublisher struct {
	failures  int
	published []eventbus.DataAgentJob
}

func (p *flakyDataAgentJobPublisher) PublishDataAgentJob(_ context.Context, job eventbus.DataAgentJob) error {
	if p.failures > 0 {
		p.failures--
		return errors.New("nats publish unavailable")
	}
	p.published = append(p.published, job)
	return nil
}

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

func TestPostgresPoolConfigAppliesConnectionLimits(t *testing.T) {
	t.Parallel()

	poolConfig, err := postgresPoolConfig(config.Config{
		DatabaseURL:      "postgresql://postgres:postgres@127.0.0.1:55432/ultra",
		DatabaseMaxConns: 7,
		DatabaseMinConns: 2,
	})
	if err != nil {
		t.Fatalf("postgresPoolConfig: %v", err)
	}
	if poolConfig.MaxConns != 7 {
		t.Fatalf("MaxConns = %d, want 7", poolConfig.MaxConns)
	}
	if poolConfig.MinConns != 2 {
		t.Fatalf("MinConns = %d, want 2", poolConfig.MinConns)
	}
}

func TestPostgresPoolConfigRejectsMinAboveMax(t *testing.T) {
	t.Parallel()

	_, err := postgresPoolConfig(config.Config{
		DatabaseURL:      "postgresql://postgres:postgres@127.0.0.1:55432/ultra",
		DatabaseMaxConns: 4,
		DatabaseMinConns: 5,
	})
	if err == nil {
		t.Fatalf("postgresPoolConfig error = nil, want min/max validation error")
	}
}

func TestApplyStatementTimeout(t *testing.T) {
	t.Parallel()

	newPoolConfig := func(t *testing.T) *pgxpool.Config {
		t.Helper()
		pc, err := postgresPoolConfig(config.Config{
			DatabaseURL: "postgresql://postgres:postgres@127.0.0.1:55432/ultra",
		})
		if err != nil {
			t.Fatalf("postgresPoolConfig: %v", err)
		}
		return pc
	}

	// A positive timeout becomes the Postgres statement_timeout runtime param, in ms.
	pc := newPoolConfig(t)
	applyStatementTimeout(pc, 30*time.Second)
	if got := pc.ConnConfig.RuntimeParams["statement_timeout"]; got != "30000" {
		t.Fatalf("statement_timeout = %q, want %q", got, "30000")
	}

	// Disabled (0) and negative are no-ops — the param is never set, so behavior is
	// unchanged by default and migrations (which skip this helper) are never bounded.
	for _, d := range []time.Duration{0, -1 * time.Second} {
		pc := newPoolConfig(t)
		applyStatementTimeout(pc, d)
		if _, ok := pc.ConnConfig.RuntimeParams["statement_timeout"]; ok {
			t.Fatalf("statement_timeout set for timeout=%v, want unset", d)
		}
	}
}

func TestNATSBusConfigClassifiesPredecessorPending(t *testing.T) {
	t.Parallel()

	cfg := natsBusConfig(config.Config{
		NATSURL:                    "nats://127.0.0.1:4222",
		NATSStream:                 "ULTRA_TEST",
		NATSJobsSubject:            "ultra.test.jobs",
		NATSDataAgentJobsSubject:   "ultra.test.data_agent.jobs",
		NATSEventsSubject:          "ultra.test.events",
		NATSCancelSubject:          "ultra.test.cancel",
		NATSEventConsumer:          "ultra-test-event-ingest",
		NATSWorkerDurable:          "ultra-test-worker",
		NATSDataAgentWorkerDurable: "ultra-test-data-agent-worker",
	})

	if cfg.IngestErrorClassifier == nil {
		t.Fatal("IngestErrorClassifier is nil; predecessor-pending gaps will be retried as unknown errors")
	}
	if got := cfg.IngestErrorClassifier(runcontrol.ErrRunEventPredecessorPending); got != eventbus.IngestErrorPredecessorPending {
		t.Fatalf("predecessor-pending class = %v, want %v", got, eventbus.IngestErrorPredecessorPending)
	}
}

func TestRetentionGCUploadRootResolvesRelativePath(t *testing.T) {
	tmp := t.TempDir()
	t.Chdir(tmp)

	got, err := retentionGCUploadRoot("data/uploads/../uploads")
	if err != nil {
		t.Fatalf("retentionGCUploadRoot: %v", err)
	}
	want := filepath.Join(tmp, "data", "uploads")
	if got != want {
		t.Fatalf("root = %q, want %q", got, want)
	}

	got, err = retentionGCUploadRoot(" ")
	if err != nil {
		t.Fatalf("retentionGCUploadRoot default: %v", err)
	}
	if got != want {
		t.Fatalf("default root = %q, want %q", got, want)
	}
}

func TestAppStartBackfillsLegacyDeletedResourceFence(t *testing.T) {
	t.Parallel()
	uploadRoot := t.TempDir()
	application, err := New(config.Config{AppVersion: "test-version", UploadRoot: uploadRoot})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	mem, ok := application.Store.(*store.MemoryStore)
	if !ok {
		t.Fatalf("store = %T, want memory store", application.Store)
	}
	const resourceID = "file_preupgrade_deleted"
	if _, err := mem.UpsertResource(context.Background(), domain.UpsertResourceInput{
		ResourceID: resourceID, OwnerUserID: "u", OwnerOrgID: "o", Status: domain.ResourceStatusDeleted,
	}); err != nil {
		t.Fatalf("seed legacy deleted resource: %v", err)
	}
	if application.Start == nil {
		t.Fatal("Start hook is nil; lifecycle backfill would not run before serving")
	}
	if err := application.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	marker := filepath.Join(uploadRoot, ".tombstones", "deleted", resourceID)
	if info, err := os.Stat(marker); err != nil || !info.Mode().IsRegular() {
		t.Fatalf("legacy deleted marker = %v err=%v, want regular reversible fence", info, err)
	}
}

func TestAppStartRecoversExpiredRunLeases(t *testing.T) {
	t.Parallel()
	application, err := New(config.Config{
		AppVersion:                "test-version",
		RunRecoveryEnabled:        true,
		RunRecoveryInterval:       10 * time.Millisecond,
		RunRecoveryFirstPassDelay: time.Millisecond,
		RunRecoveryBatchLimit:     10,
		NATSJobsSubject:           "ultra.runs.jobs",
		NATSEventsSubject:         "ultra.runs.events",
		NATSCancelSubject:         "ultra.runs.cancel",
		NATSEventConsumer:         "ultra-control-event-ingest",
		NATSWorkerDurable:         "ultra-deepagents-worker",
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

func TestAppStartRecoversExpiredDataAgentJobLeases(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	application, err := New(config.Config{
		AppVersion:                 "test-version",
		RunRecoveryEnabled:         true,
		RunRecoveryInterval:        10 * time.Millisecond,
		RunRecoveryBatchLimit:      10,
		NATSJobsSubject:            "ultra.runs.jobs",
		NATSDataAgentJobsSubject:   "ultra.data_agent.jobs",
		NATSEventsSubject:          "ultra.runs.events",
		NATSCancelSubject:          "ultra.runs.cancel",
		NATSEventConsumer:          "ultra-control-event-ingest",
		NATSWorkerDurable:          "ultra-deepagents-worker",
		NATSDataAgentWorkerDurable: "ultra-data-agent-worker",
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if application.Start == nil {
		t.Fatalf("Start hook should be configured when run recovery is enabled")
	}
	mem, ok := application.Store.(*store.MemoryStore)
	if !ok {
		t.Fatalf("store = %T, want memory store", application.Store)
	}
	now := time.Now().UTC()
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   "file_app_data_agent_recover",
		OriginalName: "recover.nii.gz",
		ContentType:  "application/x-nifti",
		SizeBytes:    512,
		SHA256:       "sha-app-data-agent-recover",
		ResourceKind: "file",
		SourceType:   "upload",
		ProjectID:    "nph-study",
		OwnerUserID:  "alice",
		OwnerOrgID:   "org-a",
		Status:       "active",
		CreatedAt:    now,
		UpdatedAt:    now,
	}); err != nil {
		t.Fatalf("UpsertResource: %v", err)
	}
	job, err := mem.CreateDataAgentJob(ctx, domain.CreateDataAgentJobInput{
		JobID:           "data_agent_job_app_recover",
		OwnerUserID:     "alice",
		OwnerOrgID:      "org-a",
		ProjectID:       "nph-study",
		JobType:         "extract_metadata",
		ResourceIDs:     []string{"file_app_data_agent_recover"},
		InputSelector:   domain.JSONMap{"resource_ids": []any{"file_app_data_agent_recover"}},
		CreatedByUserID: "alice",
		CreatedAt:       now.Add(time.Second),
	})
	if err != nil {
		t.Fatalf("CreateDataAgentJob: %v", err)
	}
	if _, _, _, err := mem.AcquireDataAgentJobLease(ctx, domain.AcquireDataAgentJobLeaseInput{
		JobID:       job.JobID,
		OwnerUserID: "alice",
		OwnerOrgID:  "org-a",
		WorkerID:    "data-agent-worker-expired",
		TTL:         time.Minute,
		Now:         now.Add(-5 * time.Minute),
	}); err != nil {
		t.Fatalf("AcquireDataAgentJobLease: %v", err)
	}
	if err := application.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	select {
	case recovered := <-application.DataAgentJobSource:
		if recovered.JobID != job.JobID || recovered.DispatchID == "" || recovered.OwnerUserID != "alice" || recovered.OwnerOrgID != "org-a" {
			t.Fatalf("recovered data-agent job = %+v, want fresh dispatch for %s", recovered, job.JobID)
		}
		if len(recovered.ResourceIDs) != 1 || recovered.ResourceIDs[0] != "file_app_data_agent_recover" {
			t.Fatalf("recovered resource ids = %+v, want original selector resource", recovered.ResourceIDs)
		}
	case <-time.After(time.Second):
		t.Fatalf("expected recovered data-agent job dispatch")
	}
	loaded, err := mem.GetDataAgentJobForUser(ctx, job.JobID, "alice", "org-a")
	if err != nil {
		t.Fatalf("GetDataAgentJobForUser: %v", err)
	}
	if loaded.Status != "queued" || loaded.ProgressCompleted != 0 {
		t.Fatalf("loaded job after recovery = %+v, want queued for worker retry", loaded)
	}
}

func TestRecoverExpiredDataAgentJobLeasesRetriesDispatchAfterPublishFailure(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	now := time.Now().UTC()
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   "file_app_data_agent_retry",
		OriginalName: "retry.nii.gz",
		ContentType:  "application/x-nifti",
		SizeBytes:    512,
		SHA256:       "sha-app-data-agent-retry",
		ResourceKind: "file",
		SourceType:   "upload",
		ProjectID:    "nph-study",
		OwnerUserID:  "alice",
		OwnerOrgID:   "org-a",
		Status:       "active",
		CreatedAt:    now,
		UpdatedAt:    now,
	}); err != nil {
		t.Fatalf("UpsertResource: %v", err)
	}
	job, err := mem.CreateDataAgentJob(ctx, domain.CreateDataAgentJobInput{
		JobID:           "data_agent_job_app_recover_retry",
		OwnerUserID:     "alice",
		OwnerOrgID:      "org-a",
		ProjectID:       "nph-study",
		JobType:         "extract_metadata",
		ResourceIDs:     []string{"file_app_data_agent_retry"},
		InputSelector:   domain.JSONMap{"resource_ids": []any{"file_app_data_agent_retry"}},
		CreatedByUserID: "alice",
		CreatedAt:       now.Add(time.Second),
	})
	if err != nil {
		t.Fatalf("CreateDataAgentJob: %v", err)
	}
	if _, _, _, err := mem.AcquireDataAgentJobLease(ctx, domain.AcquireDataAgentJobLeaseInput{
		JobID:       job.JobID,
		OwnerUserID: "alice",
		OwnerOrgID:  "org-a",
		WorkerID:    "data-agent-worker-expired",
		TTL:         time.Minute,
		Now:         now.Add(-5 * time.Minute),
	}); err != nil {
		t.Fatalf("AcquireDataAgentJobLease: %v", err)
	}
	publisher := &flakyDataAgentJobPublisher{failures: 1}

	recoverExpiredDataAgentJobLeases(ctx, mem, publisher, 10)
	if len(publisher.published) != 0 {
		t.Fatalf("published after failing publish = %+v, want none", publisher.published)
	}
	events, err := mem.ListDataAgentJobEvents(ctx, job.JobID, "alice", "org-a", 20)
	if err != nil {
		t.Fatalf("ListDataAgentJobEvents after failure: %v", err)
	}
	if got := events[len(events)-1].EventType; got != "data_agent.job.dispatch_failed" {
		t.Fatalf("last event after failed publish = %s, want data_agent.job.dispatch_failed", got)
	}

	recoverExpiredDataAgentJobLeases(ctx, mem, publisher, 10)
	if len(publisher.published) != 1 {
		t.Fatalf("published jobs after retry = %+v, want one recovered dispatch", publisher.published)
	}
	recovered := publisher.published[0]
	if recovered.JobID != job.JobID || recovered.DispatchID == "" || recovered.OwnerUserID != "alice" || recovered.OwnerOrgID != "org-a" {
		t.Fatalf("recovered data-agent job = %+v, want fresh dispatch for %s", recovered, job.JobID)
	}
	events, err = mem.ListDataAgentJobEvents(ctx, job.JobID, "alice", "org-a", 20)
	if err != nil {
		t.Fatalf("ListDataAgentJobEvents after retry: %v", err)
	}
	if got := events[len(events)-1].EventType; got != "data_agent.job.dispatched" {
		t.Fatalf("last event after retry = %s, want data_agent.job.dispatched", got)
	}
}

func TestNewAppWiresLocalDataAgentWorker(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	application, err := New(config.Config{AppVersion: "test-version"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if application.DataAgentJobSource == nil {
		t.Fatalf("DataAgentJobSource = nil, want local memory data-agent job source")
	}
	if application.DataAgentWorker == nil {
		t.Fatalf("DataAgentWorker = nil, want local data-agent worker")
	}
	mem, ok := application.Store.(*store.MemoryStore)
	if !ok {
		t.Fatalf("store = %T, want memory store", application.Store)
	}
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   "file_app_data_agent",
		OriginalName: "app-nph.nii.gz",
		ContentType:  "application/x-nifti",
		SizeBytes:    256,
		SHA256:       "sha-app-data-agent",
		ResourceKind: "file",
		SourceType:   "upload",
		OwnerUserID:  "alice",
		OwnerOrgID:   "org-a",
		Status:       "active",
		CreatedAt:    time.Now().UTC(),
		UpdatedAt:    time.Now().UTC(),
	}); err != nil {
		t.Fatalf("UpsertResource: %v", err)
	}
	job, err := mem.CreateDataAgentJob(ctx, domain.CreateDataAgentJobInput{
		JobID:           "data_agent_job_app_worker",
		OwnerUserID:     "alice",
		OwnerOrgID:      "org-a",
		JobType:         "extract_metadata",
		ResourceIDs:     []string{"file_app_data_agent"},
		InputSelector:   domain.JSONMap{"resource_ids": []any{"file_app_data_agent"}},
		CreatedByUserID: "alice",
		CreatedAt:       time.Now().UTC(),
	})
	if err != nil {
		t.Fatalf("CreateDataAgentJob: %v", err)
	}
	publisher, ok := application.Bus.(eventbus.DataAgentJobPublisher)
	if !ok {
		t.Fatalf("bus = %T, want data-agent publisher", application.Bus)
	}
	if err := publisher.PublishDataAgentJob(ctx, eventbus.DataAgentJob{
		JobID:         job.JobID,
		OwnerUserID:   "alice",
		OwnerOrgID:    "org-a",
		JobType:       "extract_metadata",
		ResourceIDs:   []string{"file_app_data_agent"},
		ResourceCount: 1,
	}); err != nil {
		t.Fatalf("PublishDataAgentJob: %v", err)
	}
	var queued eventbus.DataAgentJob
	select {
	case queued = <-application.DataAgentJobSource:
	case <-time.After(time.Second):
		t.Fatalf("expected local data-agent job")
	}
	if err := application.DataAgentWorker.RunJob(ctx, queued); err != nil {
		t.Fatalf("DataAgentWorker.RunJob: %v", err)
	}
	loaded, err := mem.GetDataAgentJobForUser(ctx, job.JobID, "alice", "org-a")
	if err != nil {
		t.Fatalf("GetDataAgentJobForUser: %v", err)
	}
	if loaded.Status != "succeeded" || loaded.OutputSummary["summary"] == "" {
		t.Fatalf("loaded data-agent job = %+v, want succeeded metadata summary", loaded)
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

// Production migrations run with the single serving role and must not impose
// any additional role/URL topology requirement.
func TestMigratePostgresUsesSingleServingRole(t *testing.T) {
	t.Parallel()
	err := MigratePostgres(context.Background(), config.Config{
		Environment: "production",
		DatabaseURL: "postgresql://single:pw@127.0.0.1:1/ultra",
	})
	// It must not reject on any role split; it proceeds to connect (and fails on
	// the unreachable :1 host, which is the expected path).
	if err == nil {
		t.Fatalf("MigratePostgres() error = nil, want an unreachable-Postgres error")
	}
	for _, forbidden := range []string{"distinct non-empty roles", "MIGRATION_DATABASE_URL is required", "same database"} {
		if strings.Contains(err.Error(), forbidden) {
			t.Fatalf("MigratePostgres() error=%q must not enforce a two-role split", err)
		}
	}
}
