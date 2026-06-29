package app

import (
	"context"
	"errors"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/config"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/nats-io/nats.go"
)

func TestMigratePostgresBackfillsResourceSearchFacts(t *testing.T) {
	databaseURL := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL"))
	if databaseURL == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is required")
	}

	ctx := context.Background()
	pool, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()
	if err := store.ApplyPostgresSchema(ctx, pool); err != nil {
		t.Fatalf("ApplyPostgresSchema: %v", err)
	}

	suffix := strings.ToLower(strings.ReplaceAll(t.Name(), "/", "_")) + "_" + time.Now().UTC().Format("20060102150405")
	userID := "app-migrate-search-" + suffix
	orgID := "org-app-migrate-search-" + suffix
	resourceID := "legacy-migrate-old-" + suffix
	createdAt := time.Date(2026, 6, 27, 12, 0, 0, 0, time.UTC)
	if _, err := pool.Exec(ctx, `
INSERT INTO control_resources (
  resource_id, owner_user_id, owner_org_id, original_name, content_type, size_bytes, source_type,
  resource_kind, status, created_at, updated_at, metadata
)
VALUES ($1, $2, $3, 'NPH_shunt_016_79yo.nii.gz', 'application/gzip', 128, 'upload', 'file', 'active', $4, $4, '{"label":"NPH"}'::jsonb)`,
		resourceID,
		userID,
		orgID,
		createdAt,
	); err != nil {
		t.Fatalf("insert legacy resource: %v", err)
	}

	if err := MigratePostgres(ctx, config.Config{DatabaseURL: databaseURL}); err != nil {
		t.Fatalf("MigratePostgres: %v", err)
	}

	pgStore := store.NewPostgresStore(pool)
	page, err := pgStore.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID: userID,
		OrgID:  orgID,
		Query:  "NPH age > 60",
		Limit:  20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser: %v", err)
	}
	if len(page.Resources) != 1 || page.Resources[0].ResourceID != resourceID {
		t.Fatalf("resources = %+v, want migrated legacy resource", page.Resources)
	}
}

func TestAppPostgresAndNATSFlowPersistsWorkerEventsAcrossRestart(t *testing.T) {
	databaseURL := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL"))
	natsURL := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_TEST_NATS_URL"))
	if databaseURL == "" || natsURL == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL and ULTRA_CONTROL_TEST_NATS_URL are required")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	suffix := strings.ToLower(strings.ReplaceAll(t.Name(), "/", "_")) + "_" + time.Now().UTC().Format("20060102150405")
	stream := "ULTRA_TEST_" + suffix
	jobsSubject := "ultra.test." + suffix + ".jobs"
	eventsSubject := "ultra.test." + suffix + ".events"
	cancelSubject := "ultra.test." + suffix + ".cancel"
	consumer := "ultra-test-control-" + suffix
	cleanupNATSStream(t, natsURL, stream)

	cfg := config.Config{
		AppName:                  "test",
		AppVersion:               "test",
		Environment:              "development",
		HTTPAddr:                 "127.0.0.1:0",
		ReadTimeout:              time.Second,
		WriteTimeout:             0,
		IdleTimeout:              time.Second,
		DatabaseURL:              databaseURL,
		NATSURL:                  natsURL,
		NATSStream:               stream,
		NATSJobsSubject:          jobsSubject,
		NATSDataAgentJobsSubject: "ultra.test." + suffix + ".data_agent.jobs",
		NATSEventsSubject:        eventsSubject,
		NATSCancelSubject:        cancelSubject,
		NATSEventConsumer:        consumer,
		NATSWorkerDurable:        "ultra-test-worker-" + suffix,
		ArtifactRoot:             t.TempDir(),
		UploadRoot:               t.TempDir(),
		DevAdminEnabled:          true,
	}
	if err := MigratePostgres(ctx, cfg); err != nil {
		t.Fatalf("MigratePostgres: %v", err)
	}

	application, err := New(cfg)
	if err != nil {
		t.Fatalf("New app: %v", err)
	}
	if application.Start == nil {
		t.Fatalf("expected NATS app start hook")
	}
	if err := application.Start(ctx); err != nil {
		t.Fatalf("Start app: %v", err)
	}
	defer application.Close()

	thread, err := application.Runs.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "user-prod-gate",
		Title:  "Production gate",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := application.Runs.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID:       thread.ThreadID,
		UserID:         "user-prod-gate",
		Goal:           "production durability gate",
		Messages:       []domain.ThreadMessage{{Role: "user", Content: "production durability gate"}},
		IdempotencyKey: "production-gate-" + suffix,
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	workerBus, err := eventbus.NewNATSBus(ctx, eventbus.NATSConfig{
		URL:           natsURL,
		Stream:        stream,
		JobsSubject:   jobsSubject,
		EventsSubject: eventsSubject,
		CancelSubject: cancelSubject,
	})
	if err != nil {
		t.Fatalf("New worker NATS bus: %v", err)
	}
	defer workerBus.Close()

	for _, input := range []domain.RunEventRecord{
		{EventID: "evt_" + run.RunID + "_started", RunID: run.RunID, ThreadID: thread.ThreadID, EventKind: "run.started", Message: "started"},
		{EventID: "evt_" + run.RunID + "_delta_001", RunID: run.RunID, ThreadID: thread.ThreadID, EventKind: "message.delta", Message: "chunk", Payload: domain.JSONMap{"text": "durable "}},
		{EventID: "evt_" + run.RunID + "_delta_002", RunID: run.RunID, ThreadID: thread.ThreadID, EventKind: "message.delta", Message: "chunk", Payload: domain.JSONMap{"text": "reply"}},
		{EventID: "evt_" + run.RunID + "_completed", RunID: run.RunID, ThreadID: thread.ThreadID, EventKind: "run.completed", Message: "completed", Payload: domain.JSONMap{"response_text": "durable reply"}},
	} {
		if err := workerBus.PublishRunEvent(ctx, input); err != nil {
			t.Fatalf("PublishRunEvent %s: %v", input.EventID, err)
		}
	}

	waitForRunStatus(t, ctx, application.Store, run.RunID, domain.RunStatusSucceeded)
	events, err := application.Store.ListRunEventsAfter(ctx, run.RunID, 0, 100)
	if err != nil {
		t.Fatalf("ListRunEventsAfter before restart: %v", err)
	}
	if got, want := len(events), 5; got != want {
		t.Fatalf("events before restart = %d, want %d: %+v", got, want, events)
	}
	if events[0].EventKind != "run.accepted" || events[len(events)-1].EventKind != "run.completed" {
		t.Fatalf("event kinds before restart = first %s last %s, want accepted/completed", events[0].EventKind, events[len(events)-1].EventKind)
	}
	application.Close()

	restarted, err := New(cfg)
	if err != nil {
		t.Fatalf("New restarted app: %v", err)
	}
	defer restarted.Close()
	replayed, err := restarted.Store.ListRunEventsAfter(ctx, run.RunID, 0, 100)
	if err != nil {
		t.Fatalf("ListRunEventsAfter after restart: %v", err)
	}
	if got, want := len(replayed), len(events); got != want {
		t.Fatalf("replayed events after restart = %d, want %d", got, want)
	}
	for index, event := range replayed {
		if event.Sequence != int64(index+1) {
			t.Fatalf("replayed event %d sequence = %d, want %d", index, event.Sequence, index+1)
		}
	}
	restartedRun, err := restarted.Store.GetRun(ctx, run.RunID)
	if err != nil {
		t.Fatalf("GetRun after restart: %v", err)
	}
	if restartedRun.Status != domain.RunStatusSucceeded || restartedRun.ResponseText != "durable reply" {
		t.Fatalf("restarted run status/response = %s/%q, want succeeded/durable reply", restartedRun.Status, restartedRun.ResponseText)
	}
}

func TestAppPostgresAndNATSFlowPersistsThousandEventAutonomousRun(t *testing.T) {
	databaseURL := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL"))
	natsURL := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_TEST_NATS_URL"))
	if databaseURL == "" || natsURL == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL and ULTRA_CONTROL_TEST_NATS_URL are required")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	suffix := strings.ToLower(strings.ReplaceAll(t.Name(), "/", "_")) + "_" + time.Now().UTC().Format("20060102150405")
	stream := "ULTRA_TEST_" + suffix
	jobsSubject := "ultra.test." + suffix + ".jobs"
	eventsSubject := "ultra.test." + suffix + ".events"
	cancelSubject := "ultra.test." + suffix + ".cancel"
	cfg := integrationAppConfig(t, databaseURL, natsURL, stream, jobsSubject, eventsSubject, cancelSubject, "ultra-test-control-"+suffix)
	if err := MigratePostgres(ctx, cfg); err != nil {
		t.Fatalf("MigratePostgres: %v", err)
	}
	cleanupNATSStream(t, natsURL, stream)

	application, err := New(cfg)
	if err != nil {
		t.Fatalf("New app: %v", err)
	}
	if err := application.Start(ctx); err != nil {
		t.Fatalf("Start app: %v", err)
	}
	defer application.Close()

	thread, err := application.Runs.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "user-volume-gate",
		Title:  "Thousand event gate",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := application.Runs.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID:       thread.ThreadID,
		UserID:         "user-volume-gate",
		Goal:           "persist a thousand-event autonomous run",
		Messages:       []domain.ThreadMessage{{Role: "user", Content: "persist a thousand-event autonomous run"}},
		IdempotencyKey: "volume-gate-" + suffix,
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	workerBus, err := eventbus.NewNATSBus(ctx, eventbus.NATSConfig{
		URL:           natsURL,
		Stream:        stream,
		JobsSubject:   jobsSubject,
		EventsSubject: eventsSubject,
		CancelSubject: cancelSubject,
	})
	if err != nil {
		t.Fatalf("New worker NATS bus: %v", err)
	}
	defer workerBus.Close()

	const deltaCount = 1200
	if err := workerBus.PublishRunEvent(ctx, domain.RunEventRecord{
		EventID:   "evt_" + run.RunID + "_started",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.started",
		Message:   "started",
	}); err != nil {
		t.Fatalf("PublishRunEvent started: %v", err)
	}
	for index := 0; index < deltaCount; index++ {
		if err := workerBus.PublishRunEvent(ctx, domain.RunEventRecord{
			EventID:   "evt_" + run.RunID + "_delta_" + fourDigit(index),
			RunID:     run.RunID,
			ThreadID:  thread.ThreadID,
			EventKind: "message.delta",
			Message:   "chunk",
			Payload:   domain.JSONMap{"text": "x"},
		}); err != nil {
			t.Fatalf("PublishRunEvent delta %d: %v", index, err)
		}
	}
	if err := workerBus.PublishRunEvent(ctx, domain.RunEventRecord{
		EventID:   "evt_" + run.RunID + "_completed",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.completed",
		Message:   "completed",
		Payload:   domain.JSONMap{"response_text": "volume complete"},
	}); err != nil {
		t.Fatalf("PublishRunEvent completed: %v", err)
	}

	waitForRunStatus(t, ctx, application.Store, run.RunID, domain.RunStatusSucceeded)
	application.Close()

	restarted, err := New(cfg)
	if err != nil {
		t.Fatalf("New restarted app: %v", err)
	}
	defer restarted.Close()
	replayed := listAllRunEventsAfter(t, ctx, restarted.Store, run.RunID, 0, 250)
	if got, want := len(replayed), deltaCount+3; got != want {
		t.Fatalf("replayed events = %d, want %d", got, want)
	}
	if replayed[0].Sequence != 1 || replayed[len(replayed)-1].Sequence != int64(deltaCount+3) {
		t.Fatalf("replayed sequence range = %d..%d, want 1..%d", replayed[0].Sequence, replayed[len(replayed)-1].Sequence, deltaCount+3)
	}
	if replayed[len(replayed)-1].EventKind != "run.completed" {
		t.Fatalf("last replayed event = %s, want run.completed", replayed[len(replayed)-1].EventKind)
	}
}

func TestAppPostgresAndNATSFlowPersistsMixedToolEventAutonomousRun(t *testing.T) {
	databaseURL := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL"))
	natsURL := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_TEST_NATS_URL"))
	if databaseURL == "" || natsURL == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL and ULTRA_CONTROL_TEST_NATS_URL are required")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
	defer cancel()

	suffix := strings.ToLower(strings.ReplaceAll(t.Name(), "/", "_")) + "_" + time.Now().UTC().Format("20060102150405")
	stream := "ULTRA_TEST_" + suffix
	jobsSubject := "ultra.test." + suffix + ".jobs"
	eventsSubject := "ultra.test." + suffix + ".events"
	cancelSubject := "ultra.test." + suffix + ".cancel"
	cfg := integrationAppConfig(t, databaseURL, natsURL, stream, jobsSubject, eventsSubject, cancelSubject, "ultra-test-control-"+suffix)
	if err := MigratePostgres(ctx, cfg); err != nil {
		t.Fatalf("MigratePostgres: %v", err)
	}
	cleanupNATSStream(t, natsURL, stream)

	application, err := New(cfg)
	if err != nil {
		t.Fatalf("New app: %v", err)
	}
	if err := application.Start(ctx); err != nil {
		t.Fatalf("Start app: %v", err)
	}
	defer application.Close()

	thread, err := application.Runs.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "user-mixed-event-gate",
		Title:  "Mixed event production gate",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := application.Runs.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID:       thread.ThreadID,
		UserID:         "user-mixed-event-gate",
		Goal:           "persist a mixed tool-heavy autonomous run",
		Messages:       []domain.ThreadMessage{{Role: "user", Content: "persist a mixed tool-heavy autonomous run"}},
		IdempotencyKey: "mixed-event-gate-" + suffix,
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	workerBus, err := eventbus.NewNATSBus(ctx, eventbus.NATSConfig{
		URL:           natsURL,
		Stream:        stream,
		JobsSubject:   jobsSubject,
		EventsSubject: eventsSubject,
		CancelSubject: cancelSubject,
	})
	if err != nil {
		t.Fatalf("New worker NATS bus: %v", err)
	}
	defer workerBus.Close()

	const toolIterations = 500
	const artifactEvery = 10
	const heartbeatEvery = 25
	const deltaEvery = 20
	if err := workerBus.PublishRunEvent(ctx, domain.RunEventRecord{
		EventID:   "evt_" + run.RunID + "_started",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.started",
		Message:   "started",
	}); err != nil {
		t.Fatalf("PublishRunEvent started: %v", err)
	}
	for index := 0; index < toolIterations; index++ {
		suffix := fourDigit(index)
		if err := workerBus.PublishRunEvent(ctx, domain.RunEventRecord{
			EventID:   "evt_" + run.RunID + "_tool_started_" + suffix,
			RunID:     run.RunID,
			ThreadID:  thread.ThreadID,
			EventKind: "tool_call.started",
			EventType: "tool_call",
			NodeName:  "execute",
			TaskID:    "tool-" + suffix,
			Message:   "tool started " + suffix,
			Payload:   domain.JSONMap{"tool_name": "execute", "index": index},
		}); err != nil {
			t.Fatalf("PublishRunEvent tool started %d: %v", index, err)
		}
		if err := workerBus.PublishRunEvent(ctx, domain.RunEventRecord{
			EventID:   "evt_" + run.RunID + "_tool_completed_" + suffix,
			RunID:     run.RunID,
			ThreadID:  thread.ThreadID,
			EventKind: "tool_call.completed",
			EventType: "tool_call",
			NodeName:  "execute",
			TaskID:    "tool-" + suffix,
			Message:   "tool completed " + suffix,
			Payload:   domain.JSONMap{"tool_name": "execute", "index": index, "duration_ms": 9},
		}); err != nil {
			t.Fatalf("PublishRunEvent tool completed %d: %v", index, err)
		}
		if index%artifactEvery == 0 {
			if err := workerBus.PublishRunEvent(ctx, domain.RunEventRecord{
				EventID:   "evt_" + run.RunID + "_artifact_" + suffix,
				RunID:     run.RunID,
				ThreadID:  thread.ThreadID,
				EventKind: "artifact.created",
				EventType: "artifact",
				Message:   "artifact " + suffix,
				Payload: domain.JSONMap{
					"artifact_id": "artifact_" + run.RunID + "_" + suffix,
					"kind":        "figure",
					"title":       "Diagnostic figure " + suffix,
					"path":        "figures/diagnostic_" + suffix + ".png",
					"mime_type":   "image/png",
					"tool_name":   "save_figure_output",
					"sha256":      "sha256-" + suffix,
				},
			}); err != nil {
				t.Fatalf("PublishRunEvent artifact %d: %v", index, err)
			}
		}
		if index%heartbeatEvery == 0 {
			if err := workerBus.PublishRunEvent(ctx, domain.RunEventRecord{
				EventID:   "evt_" + run.RunID + "_heartbeat_" + suffix,
				RunID:     run.RunID,
				ThreadID:  thread.ThreadID,
				EventKind: "run.heartbeat",
				EventType: "run",
				Message:   "heartbeat " + suffix,
				Payload:   domain.JSONMap{"tool_iterations_completed": index + 1},
			}); err != nil {
				t.Fatalf("PublishRunEvent heartbeat %d: %v", index, err)
			}
		}
		if index%deltaEvery == 0 {
			if err := workerBus.PublishRunEvent(ctx, domain.RunEventRecord{
				EventID:   "evt_" + run.RunID + "_delta_" + suffix,
				RunID:     run.RunID,
				ThreadID:  thread.ThreadID,
				EventKind: "message.delta",
				EventType: "message",
				Message:   "coordinator delta " + suffix,
				Payload:   domain.JSONMap{"text": "coordinator delta " + suffix},
			}); err != nil {
				t.Fatalf("PublishRunEvent delta %d: %v", index, err)
			}
		}
	}
	if err := workerBus.PublishRunEvent(ctx, domain.RunEventRecord{
		EventID:   "evt_" + run.RunID + "_completed",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.completed",
		EventType: "run",
		Message:   "completed",
		Payload:   domain.JSONMap{"response_text": "mixed tool run complete"},
	}); err != nil {
		t.Fatalf("PublishRunEvent completed: %v", err)
	}

	waitForRunStatus(t, ctx, application.Store, run.RunID, domain.RunStatusSucceeded)
	application.Close()

	restarted, err := New(cfg)
	if err != nil {
		t.Fatalf("New restarted app: %v", err)
	}
	defer restarted.Close()

	expectedArtifacts := toolIterations / artifactEvery
	expectedHeartbeats := toolIterations / heartbeatEvery
	expectedDeltas := toolIterations / deltaEvery
	expectedEvents := 1 + 1 + (toolIterations * 2) + expectedArtifacts + expectedHeartbeats + expectedDeltas + 1
	replayed := listAllRunEventsAfter(t, ctx, restarted.Store, run.RunID, 0, 173)
	if got := len(replayed); got != expectedEvents {
		t.Fatalf("replayed mixed events = %d, want %d", got, expectedEvents)
	}
	for index, event := range replayed {
		if event.Sequence != int64(index+1) {
			t.Fatalf("replayed event %d sequence = %d, want %d", index, event.Sequence, index+1)
		}
	}
	if replayed[0].EventKind != "run.accepted" || replayed[len(replayed)-1].EventKind != "run.completed" {
		t.Fatalf("mixed replay first/last = %s/%s, want run.accepted/run.completed", replayed[0].EventKind, replayed[len(replayed)-1].EventKind)
	}
	artifacts, err := restarted.Store.ListRunArtifacts(ctx, run.RunID, expectedArtifacts+1)
	if err != nil {
		t.Fatalf("ListRunArtifacts after restart: %v", err)
	}
	if got := len(artifacts); got != expectedArtifacts {
		t.Fatalf("artifacts after restart = %d, want %d", got, expectedArtifacts)
	}
	restartedRun, err := restarted.Store.GetRun(ctx, run.RunID)
	if err != nil {
		t.Fatalf("GetRun after restart: %v", err)
	}
	if restartedRun.Status != domain.RunStatusSucceeded || restartedRun.ResponseText != "mixed tool run complete" {
		t.Fatalf("restarted run status/response = %s/%q, want succeeded/mixed tool run complete", restartedRun.Status, restartedRun.ResponseText)
	}
	messages, err := restarted.Store.ListThreadMessages(ctx, thread.ThreadID)
	if err != nil {
		t.Fatalf("ListThreadMessages after restart: %v", err)
	}
	if got, want := len(messages), 2; got != want {
		t.Fatalf("messages after restart = %d, want user+assistant: %+v", got, messages)
	}
	if messages[1].Role != "assistant" || messages[1].RunID != run.RunID || messages[1].Content != "mixed tool run complete" {
		t.Fatalf("assistant message = %+v, want terminal response owned by run %s", messages[1], run.RunID)
	}
}

func TestAppPostgresAndNATSIngestsWorkerEventsPublishedWhileControlPlaneIsDown(t *testing.T) {
	databaseURL := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL"))
	natsURL := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_TEST_NATS_URL"))
	if databaseURL == "" || natsURL == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL and ULTRA_CONTROL_TEST_NATS_URL are required")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	suffix := strings.ToLower(strings.ReplaceAll(t.Name(), "/", "_")) + "_" + time.Now().UTC().Format("20060102150405")
	stream := "ULTRA_TEST_" + suffix
	jobsSubject := "ultra.test." + suffix + ".jobs"
	eventsSubject := "ultra.test." + suffix + ".events"
	cancelSubject := "ultra.test." + suffix + ".cancel"
	consumer := "ultra-test-control-" + suffix
	cfg := integrationAppConfig(t, databaseURL, natsURL, stream, jobsSubject, eventsSubject, cancelSubject, consumer)
	if err := MigratePostgres(ctx, cfg); err != nil {
		t.Fatalf("MigratePostgres: %v", err)
	}
	cleanupNATSStream(t, natsURL, stream)

	application, err := New(cfg)
	if err != nil {
		t.Fatalf("New app: %v", err)
	}
	thread, err := application.Runs.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "user-outage-gate",
		Title:  "Control-plane outage gate",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := application.Runs.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID:       thread.ThreadID,
		UserID:         "user-outage-gate",
		Goal:           "survive control-plane event ingest outage",
		Messages:       []domain.ThreadMessage{{Role: "user", Content: "survive control-plane event ingest outage"}},
		IdempotencyKey: "outage-gate-" + suffix,
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	application.Close()

	workerBus, err := eventbus.NewNATSBus(ctx, eventbus.NATSConfig{
		URL:           natsURL,
		Stream:        stream,
		JobsSubject:   jobsSubject,
		EventsSubject: eventsSubject,
		CancelSubject: cancelSubject,
	})
	if err != nil {
		t.Fatalf("New worker NATS bus: %v", err)
	}
	defer workerBus.Close()
	for _, input := range []domain.RunEventRecord{
		{EventID: "evt_" + run.RunID + "_started", RunID: run.RunID, ThreadID: thread.ThreadID, EventKind: "run.started", Message: "started during outage"},
		{EventID: "evt_" + run.RunID + "_delta_001", RunID: run.RunID, ThreadID: thread.ThreadID, EventKind: "message.delta", Message: "chunk", Payload: domain.JSONMap{"text": "outage "}},
		{EventID: "evt_" + run.RunID + "_completed", RunID: run.RunID, ThreadID: thread.ThreadID, EventKind: "run.completed", Message: "completed during outage", Payload: domain.JSONMap{"response_text": "outage recovered"}},
	} {
		if err := workerBus.PublishRunEvent(ctx, input); err != nil {
			t.Fatalf("PublishRunEvent during outage %s: %v", input.EventID, err)
		}
	}

	restarted, err := New(cfg)
	if err != nil {
		t.Fatalf("New restarted app: %v", err)
	}
	defer restarted.Close()
	if err := restarted.Start(ctx); err != nil {
		t.Fatalf("Start restarted app: %v", err)
	}

	waitForRunStatus(t, ctx, restarted.Store, run.RunID, domain.RunStatusSucceeded)
	replayed := listAllRunEventsAfter(t, ctx, restarted.Store, run.RunID, 0, 100)
	if got, want := len(replayed), 4; got != want {
		t.Fatalf("replayed outage events = %d, want %d: %+v", got, want, replayed)
	}
	if replayed[0].EventKind != "run.accepted" || replayed[len(replayed)-1].EventKind != "run.completed" {
		t.Fatalf("outage replay first/last = %s/%s, want run.accepted/run.completed", replayed[0].EventKind, replayed[len(replayed)-1].EventKind)
	}
	updated, err := restarted.Store.GetRun(ctx, run.RunID)
	if err != nil {
		t.Fatalf("GetRun after outage recovery: %v", err)
	}
	if updated.ResponseText != "outage recovered" {
		t.Fatalf("response text = %q, want outage recovered", updated.ResponseText)
	}
}

func TestAppPostgresAndNATSIdempotentRetriesPublishOneDurableJob(t *testing.T) {
	databaseURL := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL"))
	natsURL := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_TEST_NATS_URL"))
	if databaseURL == "" || natsURL == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL and ULTRA_CONTROL_TEST_NATS_URL are required")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	suffix := strings.ToLower(strings.ReplaceAll(t.Name(), "/", "_")) + "_" + time.Now().UTC().Format("20060102150405")
	stream := "ULTRA_TEST_" + suffix
	jobsSubject := "ultra.test." + suffix + ".jobs"
	eventsSubject := "ultra.test." + suffix + ".events"
	cancelSubject := "ultra.test." + suffix + ".cancel"
	cfg := integrationAppConfig(t, databaseURL, natsURL, stream, jobsSubject, eventsSubject, cancelSubject, "ultra-test-control-"+suffix)
	if err := MigratePostgres(ctx, cfg); err != nil {
		t.Fatalf("MigratePostgres: %v", err)
	}
	cleanupNATSStream(t, natsURL, stream)

	application, err := New(cfg)
	if err != nil {
		t.Fatalf("New app: %v", err)
	}
	defer application.Close()
	thread, err := application.Runs.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "user-retry-gate",
		Title:  "Idempotent retry gate",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}

	const submitCount = 12
	start := make(chan struct{})
	type runResult struct {
		run domain.RunRecord
		err error
	}
	results := make(chan runResult, submitCount)
	var wg sync.WaitGroup
	for index := 0; index < submitCount; index++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			run, err := application.Runs.CreateRun(ctx, runcontrol.CreateRunRequest{
				ThreadID:       thread.ThreadID,
				UserID:         "user-retry-gate",
				Goal:           "dedupe this long autonomous run",
				Messages:       []domain.ThreadMessage{{Role: "user", Content: "dedupe this long autonomous run"}},
				IdempotencyKey: "retry-gate-" + suffix,
			})
			results <- runResult{run: run, err: err}
		}()
	}
	close(start)
	wg.Wait()
	close(results)

	runIDs := map[string]bool{}
	var runID string
	for result := range results {
		if result.err != nil {
			t.Fatalf("CreateRun retry: %v", result.err)
		}
		runIDs[result.run.RunID] = true
		runID = result.run.RunID
	}
	if len(runIDs) != 1 {
		t.Fatalf("retry submissions produced run ids = %+v, want one run", runIDs)
	}
	events, err := application.Store.ListRunEventsAfter(ctx, runID, 0, 100)
	if err != nil {
		t.Fatalf("ListRunEventsAfter: %v", err)
	}
	if len(events) != 1 || events[0].EventKind != "run.accepted" {
		t.Fatalf("events = %+v, want one run.accepted event", events)
	}
	run, err := application.Store.GetRun(ctx, runID)
	if err != nil {
		t.Fatalf("GetRun: %v", err)
	}
	if strings.TrimSpace(stringFromJSON(run.Metadata["job_dispatched_at"])) == "" {
		t.Fatalf("run metadata = %+v, want durable job_dispatched_at marker", run.Metadata)
	}
	if got := natsStreamMessages(t, ctx, natsURL, stream); got != 1 {
		t.Fatalf("NATS stream messages = %d, want one durable worker job for all retries", got)
	}
}

func TestAppPostgresAndNATSHorizontalIdempotentRetriesPublishOneDurableJob(t *testing.T) {
	databaseURL := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL"))
	natsURL := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_TEST_NATS_URL"))
	if databaseURL == "" || natsURL == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL and ULTRA_CONTROL_TEST_NATS_URL are required")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	suffix := strings.ToLower(strings.ReplaceAll(t.Name(), "/", "_")) + "_" + time.Now().UTC().Format("20060102150405")
	stream := "ULTRA_TEST_" + suffix
	jobsSubject := "ultra.test." + suffix + ".jobs"
	eventsSubject := "ultra.test." + suffix + ".events"
	cancelSubject := "ultra.test." + suffix + ".cancel"
	cfgA := integrationAppConfig(t, databaseURL, natsURL, stream, jobsSubject, eventsSubject, cancelSubject, "ultra-test-control-a-"+suffix)
	cfgB := integrationAppConfig(t, databaseURL, natsURL, stream, jobsSubject, eventsSubject, cancelSubject, "ultra-test-control-b-"+suffix)
	if err := MigratePostgres(ctx, cfgA); err != nil {
		t.Fatalf("MigratePostgres: %v", err)
	}
	cleanupNATSStream(t, natsURL, stream)

	appA, err := New(cfgA)
	if err != nil {
		t.Fatalf("New app A: %v", err)
	}
	defer appA.Close()
	appB, err := New(cfgB)
	if err != nil {
		t.Fatalf("New app B: %v", err)
	}
	defer appB.Close()

	thread, err := appA.Runs.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "user-horizontal-retry",
		Title:  "Horizontal idempotent retry gate",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}

	services := []*runcontrol.Service{appA.Runs, appB.Runs}
	const submitCount = 16
	start := make(chan struct{})
	type runResult struct {
		run domain.RunRecord
		err error
	}
	results := make(chan runResult, submitCount)
	var wg sync.WaitGroup
	for index := 0; index < submitCount; index++ {
		wg.Add(1)
		service := services[index%len(services)]
		go func() {
			defer wg.Done()
			<-start
			run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
				ThreadID:       thread.ThreadID,
				UserID:         "user-horizontal-retry",
				Goal:           "dedupe this run across control-plane instances",
				Messages:       []domain.ThreadMessage{{Role: "user", Content: "dedupe this run across control-plane instances"}},
				IdempotencyKey: "horizontal-retry-gate-" + suffix,
			})
			results <- runResult{run: run, err: err}
		}()
	}
	close(start)
	wg.Wait()
	close(results)

	runIDs := map[string]bool{}
	var runID string
	for result := range results {
		if result.err != nil {
			t.Fatalf("CreateRun horizontal retry: %v", result.err)
		}
		runIDs[result.run.RunID] = true
		runID = result.run.RunID
	}
	if len(runIDs) != 1 {
		t.Fatalf("horizontal retry submissions produced run ids = %+v, want one run", runIDs)
	}
	events, err := appA.Store.ListRunEventsAfter(ctx, runID, 0, 100)
	if err != nil {
		t.Fatalf("ListRunEventsAfter: %v", err)
	}
	if len(events) != 1 || events[0].EventID != "evt_"+runID+"_accepted" || events[0].EventKind != "run.accepted" {
		t.Fatalf("events = %+v, want one stable run.accepted event", events)
	}
	run, err := appB.Store.GetRun(ctx, runID)
	if err != nil {
		t.Fatalf("GetRun from app B store: %v", err)
	}
	if strings.TrimSpace(stringFromJSON(run.Metadata["job_dispatched_at"])) == "" {
		t.Fatalf("run metadata = %+v, want durable job_dispatched_at marker", run.Metadata)
	}
	if got := natsStreamMessages(t, ctx, natsURL, stream); got != 1 {
		t.Fatalf("NATS stream messages = %d, want one durable worker job across Go instances", got)
	}
}

func TestAppPostgresAndNATSCompletionPersistsAssistantTranscriptAcrossRestart(t *testing.T) {
	databaseURL := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL"))
	natsURL := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_TEST_NATS_URL"))
	if databaseURL == "" || natsURL == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL and ULTRA_CONTROL_TEST_NATS_URL are required")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	suffix := strings.ToLower(strings.ReplaceAll(t.Name(), "/", "_")) + "_" + time.Now().UTC().Format("20060102150405")
	stream := "ULTRA_TEST_" + suffix
	jobsSubject := "ultra.test." + suffix + ".jobs"
	eventsSubject := "ultra.test." + suffix + ".events"
	cancelSubject := "ultra.test." + suffix + ".cancel"
	cfg := integrationAppConfig(t, databaseURL, natsURL, stream, jobsSubject, eventsSubject, cancelSubject, "ultra-test-control-"+suffix)
	if err := MigratePostgres(ctx, cfg); err != nil {
		t.Fatalf("MigratePostgres: %v", err)
	}
	cleanupNATSStream(t, natsURL, stream)

	application, err := New(cfg)
	if err != nil {
		t.Fatalf("New app: %v", err)
	}
	if err := application.Start(ctx); err != nil {
		t.Fatalf("Start app: %v", err)
	}
	defer application.Close()

	thread, err := application.Runs.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "user-transcript-gate",
		Title:  "Transcript persistence gate",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := application.Runs.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID:       thread.ThreadID,
		UserID:         "user-transcript-gate",
		Goal:           "persist completed assistant transcript",
		Messages:       []domain.ThreadMessage{{Role: "user", Content: "persist completed assistant transcript"}},
		IdempotencyKey: "transcript-gate-" + suffix,
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	workerBus, err := eventbus.NewNATSBus(ctx, eventbus.NATSConfig{
		URL:           natsURL,
		Stream:        stream,
		JobsSubject:   jobsSubject,
		EventsSubject: eventsSubject,
		CancelSubject: cancelSubject,
	})
	if err != nil {
		t.Fatalf("New worker NATS bus: %v", err)
	}
	defer workerBus.Close()
	if err := workerBus.PublishRunEvent(ctx, domain.RunEventRecord{
		EventID:   "evt_" + run.RunID + "_completed",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.completed",
		Message:   "completed",
		Payload:   domain.JSONMap{"response_text": "This answer must hydrate after refresh."},
	}); err != nil {
		t.Fatalf("PublishRunEvent completed: %v", err)
	}

	waitForRunStatus(t, ctx, application.Store, run.RunID, domain.RunStatusSucceeded)
	application.Close()

	restarted, err := New(cfg)
	if err != nil {
		t.Fatalf("New restarted app: %v", err)
	}
	defer restarted.Close()
	messages, err := restarted.Store.ListThreadMessages(ctx, thread.ThreadID)
	if err != nil {
		t.Fatalf("ListThreadMessages after restart: %v", err)
	}
	if len(messages) != 2 {
		t.Fatalf("messages = %+v, want user+assistant transcript", messages)
	}
	if messages[0].Role != "user" || messages[0].RunID != run.RunID {
		t.Fatalf("user message = %+v, want current run-owned user message", messages[0])
	}
	if messages[1].Role != "assistant" || messages[1].Content != "This answer must hydrate after refresh." || messages[1].RunID != run.RunID {
		t.Fatalf("assistant message = %+v, want completed response owned by run %s", messages[1], run.RunID)
	}
}

func TestAppPostgresRunLeaseSurvivesRestartAndAllowsRecoveryAfterExpiry(t *testing.T) {
	databaseURL := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL"))
	natsURL := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_TEST_NATS_URL"))
	if databaseURL == "" || natsURL == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL and ULTRA_CONTROL_TEST_NATS_URL are required")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	suffix := strings.ToLower(strings.ReplaceAll(t.Name(), "/", "_")) + "_" + time.Now().UTC().Format("20060102150405")
	stream := "ULTRA_TEST_" + suffix
	jobsSubject := "ultra.test." + suffix + ".jobs"
	eventsSubject := "ultra.test." + suffix + ".events"
	cancelSubject := "ultra.test." + suffix + ".cancel"
	cfg := integrationAppConfig(t, databaseURL, natsURL, stream, jobsSubject, eventsSubject, cancelSubject, "ultra-test-control-"+suffix)
	if err := MigratePostgres(ctx, cfg); err != nil {
		t.Fatalf("MigratePostgres: %v", err)
	}
	cleanupNATSStream(t, natsURL, stream)

	application, err := New(cfg)
	if err != nil {
		t.Fatalf("New app: %v", err)
	}
	if err := application.Start(ctx); err != nil {
		t.Fatalf("Start app: %v", err)
	}
	defer application.Close()

	thread, err := application.Runs.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "user-lease-failover-gate",
		Title:  "Lease failover gate",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := application.Runs.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID:       thread.ThreadID,
		UserID:         "user-lease-failover-gate",
		Goal:           "long autonomous leased run",
		Messages:       []domain.ThreadMessage{{Role: "user", Content: "long autonomous leased run"}},
		IdempotencyKey: "lease-failover-gate-" + suffix,
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	lease, err := application.Runs.AcquireRunLease(ctx, runcontrol.AcquireRunLeaseRequest{
		RunID:    run.RunID,
		WorkerID: "worker-a",
		TTL:      30 * time.Second,
	})
	if err != nil {
		t.Fatalf("AcquireRunLease worker-a: %v", err)
	}
	application.Close()

	restarted, err := New(cfg)
	if err != nil {
		t.Fatalf("New restarted app: %v", err)
	}
	if err := restarted.Start(ctx); err != nil {
		t.Fatalf("Start restarted app: %v", err)
	}
	defer restarted.Close()

	if _, err := restarted.Runs.AcquireRunLease(ctx, runcontrol.AcquireRunLeaseRequest{
		RunID:    run.RunID,
		WorkerID: "worker-b",
		TTL:      30 * time.Second,
	}); !errors.Is(err, store.ErrConflict) {
		t.Fatalf("AcquireRunLease worker-b while active err = %v, want ErrConflict", err)
	}

	recovered, err := restarted.Store.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    run.RunID,
		WorkerID: "worker-b",
		TTL:      30 * time.Second,
		Now:      lease.LeaseExpiresAt.Add(time.Millisecond),
	})
	if err != nil {
		t.Fatalf("AcquireRunLease worker-b after simulated expiry: %v", err)
	}
	if recovered.WorkerID != "worker-b" || recovered.LeaseToken == "" || recovered.LeaseToken == lease.LeaseToken {
		t.Fatalf("recovered lease = %+v, want fresh worker-b lease replacing worker-a", recovered)
	}
}

func integrationAppConfig(t *testing.T, databaseURL string, natsURL string, stream string, jobsSubject string, eventsSubject string, cancelSubject string, consumer string) config.Config {
	t.Helper()
	return config.Config{
		AppName:                  "test",
		AppVersion:               "test",
		Environment:              "development",
		HTTPAddr:                 "127.0.0.1:0",
		ReadTimeout:              time.Second,
		WriteTimeout:             0,
		IdleTimeout:              time.Second,
		DatabaseURL:              databaseURL,
		NATSURL:                  natsURL,
		NATSStream:               stream,
		NATSJobsSubject:          jobsSubject,
		NATSDataAgentJobsSubject: strings.TrimSuffix(jobsSubject, ".jobs") + ".data_agent.jobs",
		NATSEventsSubject:        eventsSubject,
		NATSCancelSubject:        cancelSubject,
		NATSEventConsumer:        consumer,
		NATSWorkerDurable:        strings.TrimSuffix(consumer, "control") + "worker",
		ArtifactRoot:             t.TempDir(),
		UploadRoot:               t.TempDir(),
		DevAdminEnabled:          true,
	}
}

func cleanupNATSStream(t *testing.T, url string, stream string) {
	t.Helper()
	conn, err := nats.Connect(url)
	if err != nil {
		t.Fatalf("connect NATS cleanup: %v", err)
	}
	t.Cleanup(func() {
		conn.Close()
	})
	js, err := conn.JetStream()
	if err != nil {
		t.Fatalf("NATS JetStream cleanup: %v", err)
	}
	t.Cleanup(func() {
		_ = js.DeleteStream(stream)
	})
}

func natsStreamMessages(t *testing.T, ctx context.Context, url string, stream string) uint64 {
	t.Helper()
	conn, err := nats.Connect(url)
	if err != nil {
		t.Fatalf("connect NATS: %v", err)
	}
	defer conn.Close()
	js, err := conn.JetStream()
	if err != nil {
		t.Fatalf("NATS JetStream: %v", err)
	}
	info, err := js.StreamInfo(stream, nats.Context(ctx))
	if err != nil {
		t.Fatalf("StreamInfo: %v", err)
	}
	return info.State.Msgs
}

func stringFromJSON(value any) string {
	switch typed := value.(type) {
	case string:
		return typed
	default:
		return ""
	}
}

func listAllRunEventsAfter(t *testing.T, ctx context.Context, store runcontrol.Store, runID string, after int64, pageSize int) []domain.RunEventRecord {
	t.Helper()
	var all []domain.RunEventRecord
	cursor := after
	for {
		page, err := store.ListRunEventsAfter(ctx, runID, cursor, pageSize)
		if err != nil {
			t.Fatalf("ListRunEventsAfter after %d: %v", cursor, err)
		}
		if len(page) == 0 {
			return all
		}
		all = append(all, page...)
		cursor = page[len(page)-1].Sequence
	}
}

func fourDigit(value int) string {
	return string([]byte{
		byte('0' + (value/1000)%10),
		byte('0' + (value/100)%10),
		byte('0' + (value/10)%10),
		byte('0' + value%10),
	})
}

func waitForRunStatus(t *testing.T, ctx context.Context, store runcontrol.Store, runID string, want domain.RunStatus) {
	t.Helper()
	ticker := time.NewTicker(25 * time.Millisecond)
	defer ticker.Stop()
	for {
		run, err := store.GetRun(ctx, runID)
		if err == nil && run.Status == want {
			return
		}
		select {
		case <-ctx.Done():
			t.Fatalf("run %s did not reach %s before timeout", runID, want)
		case <-ticker.C:
		}
	}
}
