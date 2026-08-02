package runcontrol

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestServiceCreateRunEmitsAcceptedAndDispatches(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Test thread",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}

	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Run deterministic worker.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Run deterministic worker."}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	if run.Status != domain.RunStatusQueued {
		t.Fatalf("run status = %s, want queued", run.Status)
	}

	events, err := mem.ListRunEvents(ctx, run.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	if len(events) != 1 || events[0].EventKind != "run.accepted" {
		t.Fatalf("events = %+v, want run.accepted", events)
	}
	if got, want := events[0].EventID, "evt_"+run.RunID+"_accepted"; got != want {
		t.Fatalf("accepted event id = %q, want %q", got, want)
	}

	select {
	case job := <-bus.Jobs():
		if job.RunID != run.RunID || job.ThreadID != thread.ThreadID {
			t.Fatalf("job = %+v, want run/thread ids", job)
		}
	case <-time.After(time.Second):
		t.Fatalf("expected dispatched job")
	}
}

func TestServiceCreateRunStampsRuntimeFacts(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	now := time.Date(2026, time.June, 25, 0, 42, 5, 123456789, time.UTC)
	service := NewServiceWithOptions(mem, bus, ServiceOptions{
		Now: func() time.Time { return now },
		RuntimeFacts: RuntimeFactsConfig{
			ProductName:         "Ultra",
			AppName:             "BisQue Ultra Control Plane",
			AppVersion:          "2026.6",
			Environment:         "production",
			PublicURL:           "https://ultra.example.edu",
			DefaultUserTimezone: "UTC",
		},
	})

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Runtime facts",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "What is today's date?",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "What is today's date?"}},
		Metadata: domain.JSONMap{"user_timezone": "America/Los_Angeles"},
		JobMetadata: domain.JSONMap{
			"runtime_facts": domain.JSONMap{"current_datetime_utc": "not-trusted"},
		},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	runtimeFacts, ok := run.Metadata["runtime_facts"].(domain.JSONMap)
	if !ok {
		t.Fatalf("runtime_facts = %#v, want metadata map", run.Metadata["runtime_facts"])
	}
	assertRuntimeFacts(t, runtimeFacts)

	select {
	case job := <-bus.Jobs():
		jobFacts, ok := job.Metadata["runtime_facts"].(domain.JSONMap)
		if !ok {
			t.Fatalf("job runtime_facts = %#v, want metadata map", job.Metadata["runtime_facts"])
		}
		assertRuntimeFacts(t, jobFacts)
	case <-time.After(time.Second):
		t.Fatalf("expected dispatched job")
	}
}

func TestServiceCreateRunWithRetiredRareSpotToolUsesDeepAgentsPathAndPreservesMetadata(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "RareSpot thread",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}

	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID:          thread.ThreadID,
		UserID:            "user-1",
		Goal:              "Run RareSpot ecology inference.",
		Messages:          []domain.ThreadMessage{{Role: "user", Content: "Run RareSpot."}},
		FileIDs:           []string{"file-1"},
		ResourceURIs:      []string{"bisque://resource/abc"},
		DatasetURIs:       []string{"bisque://dataset/def"},
		SelectedToolNames: []string{"rarespot_ecology_inference"},
		WorkflowHint:      domain.JSONMap{"id": "rarespot_ecology"},
		KnowledgeContext:  domain.JSONMap{"active_paper": "arxiv:2509.26626"},
		SelectionContext:  domain.JSONMap{"source": "sidebar"},
		ReasoningMode:     "deep",
		Budgets:           domain.JSONMap{"max_runtime_seconds": 1800},
		Benchmark:         domain.JSONMap{"suite": "rarespot-smoke"},
		Metadata:          domain.JSONMap{"existing": "kept"},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	// RareSpot dispatch is retired: a run that still requests the old tool/hint now
	// takes the normal deep_agents path (the prairie-dog-detection Skill runs in the
	// sandbox), instead of routing to a rarespot queue with no consumer.
	if run.WorkflowKind != "deep_agents" {
		t.Fatalf("workflow kind = %q, want deep_agents", run.WorkflowKind)
	}
	if run.Metadata["existing"] != "kept" {
		t.Fatalf("metadata existing = %v, want kept", run.Metadata["existing"])
	}
	if got := run.Metadata["file_ids"]; !sameStringSlice(got, []string{"file-1"}) {
		t.Fatalf("metadata file_ids = %#v, want file-1", got)
	}
	if got := run.Metadata["resource_uris"]; !sameStringSlice(got, []string{"bisque://resource/abc"}) {
		t.Fatalf("metadata resource_uris = %#v, want resource URI", got)
	}
	if got := run.Metadata["dataset_uris"]; !sameStringSlice(got, []string{"bisque://dataset/def"}) {
		t.Fatalf("metadata dataset_uris = %#v, want dataset URI", got)
	}
	if got := run.Metadata["selected_tool_names"]; !sameStringSlice(got, []string{"rarespot_ecology_inference"}) {
		t.Fatalf("metadata selected_tool_names = %#v, want RareSpot tool", got)
	}
	if workflow, ok := run.Metadata["workflow_hint"].(domain.JSONMap); !ok || workflow["id"] != "rarespot_ecology" {
		t.Fatalf("metadata workflow_hint = %#v, want rarespot_ecology", run.Metadata["workflow_hint"])
	}
	if knowledge, ok := run.Metadata["knowledge_context"].(domain.JSONMap); !ok || knowledge["active_paper"] != "arxiv:2509.26626" {
		t.Fatalf("metadata knowledge_context = %#v, want active paper", run.Metadata["knowledge_context"])
	}
	if run.Metadata["reasoning_mode"] != "deep" {
		t.Fatalf("metadata reasoning_mode = %#v, want deep", run.Metadata["reasoning_mode"])
	}
	if benchmark, ok := run.Metadata["benchmark"].(domain.JSONMap); !ok || benchmark["suite"] != "rarespot-smoke" {
		t.Fatalf("metadata benchmark = %#v, want rarespot smoke", run.Metadata["benchmark"])
	}

	select {
	case job := <-bus.Jobs():
		if job.WorkflowKind != "deep_agents" {
			t.Fatalf("job workflow kind = %q, want deep_agents", job.WorkflowKind)
		}
		if len(job.Messages) != 1 || job.Messages[0].Content != "Run RareSpot." {
			t.Fatalf("job messages = %+v, want full prompt context", job.Messages)
		}
		if got := job.FileIDs; !sameStringSlice(got, []string{"file-1"}) {
			t.Fatalf("job file ids = %#v, want file-1", got)
		}
		if got := job.ResourceURIs; !sameStringSlice(got, []string{"bisque://resource/abc"}) {
			t.Fatalf("job resource uris = %#v, want resource URI", got)
		}
		if got := job.DatasetURIs; !sameStringSlice(got, []string{"bisque://dataset/def"}) {
			t.Fatalf("job dataset uris = %#v, want dataset URI", got)
		}
		if got := job.SelectedToolNames; !sameStringSlice(got, []string{"rarespot_ecology_inference"}) {
			t.Fatalf("job selected tools = %#v, want RareSpot tool", got)
		}
		if job.SelectionContext["source"] != "sidebar" {
			t.Fatalf("job selection context = %#v, want sidebar", job.SelectionContext)
		}
		if job.KnowledgeContext["active_paper"] != "arxiv:2509.26626" {
			t.Fatalf("job knowledge context = %#v, want active paper", job.KnowledgeContext)
		}
		if job.ReasoningMode != "deep" {
			t.Fatalf("job reasoning mode = %q, want deep", job.ReasoningMode)
		}
		if job.Benchmark["suite"] != "rarespot-smoke" {
			t.Fatalf("job benchmark = %#v, want rarespot smoke", job.Benchmark)
		}
		if job.Budgets["max_runtime_seconds"] != 1800 {
			t.Fatalf("job budgets = %#v, want runtime budget", job.Budgets)
		}
		if got := job.Metadata["selected_tool_names"]; !sameStringSlice(got, []string{"rarespot_ecology_inference"}) {
			t.Fatalf("job metadata selected_tool_names = %#v, want RareSpot tool", got)
		}
	case <-time.After(time.Second):
		t.Fatalf("expected dispatched RareSpot job")
	}
}

func assertRuntimeFacts(t *testing.T, facts domain.JSONMap) {
	t.Helper()
	wantInstant := "2026-06-25T00:42:05.123456789Z"
	if facts["run_started_at"] != wantInstant {
		t.Fatalf("run_started_at = %#v, want %s", facts["run_started_at"], wantInstant)
	}
	if facts["current_datetime_utc"] != wantInstant {
		t.Fatalf("current_datetime_utc = %#v, want %s", facts["current_datetime_utc"], wantInstant)
	}
	if facts["current_date_utc"] != "Thursday, June 25, 2026" {
		t.Fatalf("current_date_utc = %#v", facts["current_date_utc"])
	}
	if facts["user_timezone"] != "America/Los_Angeles" {
		t.Fatalf("user_timezone = %#v", facts["user_timezone"])
	}
	if facts["local_datetime"] != "Wednesday, June 24, 2026 17:42:05 PDT" {
		t.Fatalf("local_datetime = %#v", facts["local_datetime"])
	}
	if facts["product_name"] != "Ultra" {
		t.Fatalf("product_name = %#v", facts["product_name"])
	}
	if facts["app_name"] != "BisQue Ultra Control Plane" {
		t.Fatalf("app_name = %#v", facts["app_name"])
	}
	if facts["app_version"] != "2026.6" {
		t.Fatalf("app_version = %#v", facts["app_version"])
	}
	if facts["deployment_environment"] != "production" {
		t.Fatalf("deployment_environment = %#v", facts["deployment_environment"])
	}
	if facts["public_url"] != "https://ultra.example.edu" {
		t.Fatalf("public_url = %#v", facts["public_url"])
	}
}

func TestServiceRequeueRunPublishesExistingRunWithFreshDispatchID(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Recover run",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID:          thread.ThreadID,
		UserID:            "user-1",
		Goal:              "Continue a long analysis.",
		Messages:          []domain.ThreadMessage{{Role: "user", Content: "Continue a long analysis."}},
		FileIDs:           []string{"file-1"},
		ResourceURIs:      []string{"resource://file-1"},
		SelectedToolNames: []string{"python"},
		KnowledgeContext:  domain.JSONMap{"paper_id": "arxiv:2509.26626"},
		IdempotencyKey:    "recover-run-key",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	drainJobs(bus)
	drainRunEvents(bus)
	if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt_" + run.RunID + "_started",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.started",
		Message:   "started",
	}); err != nil {
		t.Fatalf("IngestRunEvent started: %v", err)
	}
	drainRunEvents(bus)
	messagesBefore, err := mem.ListThreadMessages(ctx, thread.ThreadID)
	if err != nil {
		t.Fatalf("ListThreadMessages before: %v", err)
	}

	requeued, err := service.RequeueRun(ctx, RequeueRunRequest{
		RunID:  run.RunID,
		Reason: "lease expired",
	})
	if err != nil {
		t.Fatalf("RequeueRun: %v", err)
	}
	if requeued.RunID != run.RunID || requeued.Status != domain.RunStatusRunning {
		t.Fatalf("requeued run = %+v, want same running run", requeued)
	}

	select {
	case job := <-bus.Jobs():
		if job.RunID != run.RunID || job.ThreadID != thread.ThreadID || job.UserID != "user-1" {
			t.Fatalf("job identity = %+v, want original run/thread/user", job)
		}
		if job.DispatchID == "" {
			t.Fatalf("job dispatch id is empty; explicit requeue must bypass JetStream job:<run_id> dedupe")
		}
		if job.Goal != "Continue a long analysis." || len(job.Messages) != 1 || job.Messages[0].Content != "Continue a long analysis." {
			t.Fatalf("job context = %+v, want original prompt context", job)
		}
		if got := job.Metadata["requeue_reason"]; got != "lease expired" {
			t.Fatalf("job metadata requeue_reason = %#v, want lease expired", got)
		}
		if got := job.Metadata["idempotency_key"]; got != "recover-run-key" {
			t.Fatalf("job metadata idempotency_key = %#v, want original key", got)
		}
	case <-time.After(time.Second):
		t.Fatalf("expected requeued job")
	}
	select {
	case event := <-bus.Events():
		if event.EventKind != "run.requeued" || event.RunID != run.RunID {
			t.Fatalf("event = %+v, want run.requeued for run", event)
		}
	case <-time.After(time.Second):
		t.Fatalf("expected requeued event fanout")
	}
	events, err := mem.ListRunEvents(ctx, run.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	if events[len(events)-1].EventKind != "run.requeued" {
		t.Fatalf("last event = %+v, want run.requeued", events[len(events)-1])
	}
	messagesAfter, err := mem.ListThreadMessages(ctx, thread.ThreadID)
	if err != nil {
		t.Fatalf("ListThreadMessages after: %v", err)
	}
	if len(messagesAfter) != len(messagesBefore) {
		t.Fatalf("thread messages grew from %d to %d; requeue must not duplicate prompts", len(messagesBefore), len(messagesAfter))
	}
}

func TestServiceRequeueRunEvictsActiveLeaseForImmediateRecovery(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Recover leased run",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Continue a leased long analysis.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Continue a leased long analysis."}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	drainJobs(bus)
	drainRunEvents(bus)
	lease, err := service.AcquireRunLease(ctx, AcquireRunLeaseRequest{
		RunID:    run.RunID,
		WorkerID: "worker-a",
		TTL:      time.Hour,
	})
	if err != nil {
		t.Fatalf("AcquireRunLease: %v", err)
	}

	if _, err := service.RequeueRun(ctx, RequeueRunRequest{
		RunID:  run.RunID,
		Reason: "stale worker heartbeat",
	}); err != nil {
		t.Fatalf("RequeueRun: %v", err)
	}

	if _, err := service.RenewRunLease(ctx, RenewRunLeaseRequest{
		RunID:      run.RunID,
		LeaseToken: lease.LeaseToken,
		TTL:        time.Hour,
	}); !errors.Is(err, store.ErrConflict) {
		t.Fatalf("RenewRunLease with evicted token err = %v, want ErrConflict", err)
	}
	replacement, err := service.AcquireRunLease(ctx, AcquireRunLeaseRequest{
		RunID:    run.RunID,
		WorkerID: "worker-b",
		TTL:      time.Hour,
	})
	if err != nil {
		t.Fatalf("AcquireRunLease replacement: %v", err)
	}
	if replacement.WorkerID != "worker-b" {
		t.Fatalf("replacement lease = %+v, want worker-b", replacement)
	}
	drainJobs(bus)
	event := <-bus.Events()
	if event.EventKind != "run.requeued" {
		t.Fatalf("event kind = %s, want run.requeued", event.EventKind)
	}
	if event.Payload["evicted_lease_worker_id"] != "worker-a" {
		t.Fatalf("event payload = %+v, want evicted lease worker", event.Payload)
	}
}

func TestServiceRequeueRunRejectsTerminalRuns(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{UserID: "user-1", Title: "terminal"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Finish.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Finish."}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt_" + run.RunID + "_completed",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.completed",
		Payload:   domain.JSONMap{"response_text": "done"},
	}); err != nil {
		t.Fatalf("IngestRunEvent completed: %v", err)
	}
	drainJobs(bus)
	drainRunEvents(bus)

	_, err = service.RequeueRun(ctx, RequeueRunRequest{RunID: run.RunID, Reason: "operator retry"})
	if !errors.Is(err, store.ErrConflict) {
		t.Fatalf("RequeueRun terminal err = %v, want ErrConflict", err)
	}
	select {
	case job := <-bus.Jobs():
		t.Fatalf("unexpected job for terminal requeue: %+v", job)
	default:
	}
}

func TestServiceIngestRunHeartbeatMarksRunRunning(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Heartbeat thread",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "long silent compute",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "long silent compute"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	drainRunEvents(bus)

	event, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt-heartbeat-1",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.heartbeat",
		EventType: "run",
		NodeName:  "worker",
		Level:     "info",
		Message:   "Worker heartbeat.",
		Payload:   domain.JSONMap{"status": "alive"},
	})
	if err != nil {
		t.Fatalf("IngestRunEvent heartbeat: %v", err)
	}
	if event.EventKind != "run.heartbeat" {
		t.Fatalf("event kind = %s, want run.heartbeat", event.EventKind)
	}
	updated, err := mem.GetRun(ctx, run.RunID)
	if err != nil {
		t.Fatalf("GetRun: %v", err)
	}
	if updated.Status != domain.RunStatusRunning {
		t.Fatalf("status = %s, want running", updated.Status)
	}
	if updated.StartedAt == nil {
		t.Fatalf("heartbeat should initialize StartedAt")
	}
}

func TestServiceRecoverExpiredRunLeasesRequeuesOnlyExpiredLeases(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	now := time.Date(2026, 6, 1, 12, 0, 0, 0, time.UTC)
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Recover expired leases",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	expiredRun, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "expired long analysis",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "expired long analysis"}},
	})
	if err != nil {
		t.Fatalf("CreateRun expired: %v", err)
	}
	activeRun, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "active long analysis",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "active long analysis"}},
	})
	if err != nil {
		t.Fatalf("CreateRun active: %v", err)
	}
	drainJobs(bus)
	drainRunEvents(bus)
	expiredLease, err := mem.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    expiredRun.RunID,
		WorkerID: "worker-expired",
		TTL:      time.Minute,
		Now:      now.Add(-2 * time.Minute),
	})
	if err != nil {
		t.Fatalf("AcquireRunLease expired: %v", err)
	}
	activeLease, err := mem.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    activeRun.RunID,
		WorkerID: "worker-active",
		TTL:      time.Hour,
		Now:      now,
	})
	if err != nil {
		t.Fatalf("AcquireRunLease active: %v", err)
	}

	result, err := service.RecoverExpiredRunLeases(ctx, RecoverExpiredRunLeasesRequest{
		Now:    now,
		Reason: "automatic expired lease recovery",
		Limit:  100,
	})
	if err != nil {
		t.Fatalf("RecoverExpiredRunLeases: %v", err)
	}
	if result.Checked != 2 {
		t.Fatalf("checked = %d, want 2", result.Checked)
	}
	if len(result.RequeuedRuns) != 1 || result.RequeuedRuns[0].RunID != expiredRun.RunID {
		t.Fatalf("requeued runs = %+v, want only expired run", result.RequeuedRuns)
	}
	if _, err := service.RenewRunLease(ctx, RenewRunLeaseRequest{
		RunID:      expiredRun.RunID,
		LeaseToken: expiredLease.LeaseToken,
		TTL:        time.Hour,
	}); !errors.Is(err, store.ErrConflict) {
		t.Fatalf("RenewRunLease expired old token err = %v, want ErrConflict", err)
	}
	if _, err := mem.RenewRunLease(ctx, domain.RenewRunLeaseInput{
		RunID:      activeRun.RunID,
		LeaseToken: activeLease.LeaseToken,
		TTL:        time.Hour,
		Now:        now.Add(time.Minute),
	}); err != nil {
		t.Fatalf("RenewRunLease active token: %v", err)
	}
	requeuedJobs := 0
	for {
		select {
		case job := <-bus.Jobs():
			if job.RunID == expiredRun.RunID && job.DispatchID != "" {
				requeuedJobs++
			}
			if job.RunID == activeRun.RunID {
				t.Fatalf("active run was requeued: %+v", job)
			}
		default:
			if requeuedJobs != 1 {
				t.Fatalf("requeued jobs = %d, want 1", requeuedJobs)
			}
			return
		}
	}
}

func TestServiceRecoverExpiredRunLeasesRequeuesStaleLeaseOwnerHeartbeat(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	now := time.Date(2026, 6, 11, 9, 30, 0, 0, time.UTC)
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Recover stale lease owner",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	staleRun, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "long stale worker analysis",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "long stale worker analysis"}},
	})
	if err != nil {
		t.Fatalf("CreateRun stale: %v", err)
	}
	activeRun, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "long active worker analysis",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "long active worker analysis"}},
	})
	if err != nil {
		t.Fatalf("CreateRun active: %v", err)
	}
	drainJobs(bus)
	drainRunEvents(bus)

	staleLease, err := mem.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    staleRun.RunID,
		WorkerID: "worker-stale",
		TTL:      20 * time.Minute,
		Now:      now.Add(-5 * time.Minute),
	})
	if err != nil {
		t.Fatalf("AcquireRunLease stale: %v", err)
	}
	activeLease, err := mem.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    activeRun.RunID,
		WorkerID: "worker-active",
		TTL:      20 * time.Minute,
		Now:      now.Add(-30 * time.Second),
	})
	if err != nil {
		t.Fatalf("AcquireRunLease active: %v", err)
	}
	if _, err := mem.UpsertWorkerHeartbeat(ctx, domain.UpsertWorkerHeartbeatInput{
		WorkerID:        "worker-stale",
		WorkerKind:      "deepagents",
		Status:          "busy",
		CurrentRunID:    staleRun.RunID,
		LastHeartbeatAt: now.Add(-5 * time.Minute),
	}); err != nil {
		t.Fatalf("UpsertWorkerHeartbeat stale: %v", err)
	}
	if _, err := mem.UpsertWorkerHeartbeat(ctx, domain.UpsertWorkerHeartbeatInput{
		WorkerID:        "worker-active",
		WorkerKind:      "deepagents",
		Status:          "busy",
		CurrentRunID:    activeRun.RunID,
		LastHeartbeatAt: now.Add(-30 * time.Second),
	}); err != nil {
		t.Fatalf("UpsertWorkerHeartbeat active: %v", err)
	}

	result, err := service.RecoverExpiredRunLeases(ctx, RecoverExpiredRunLeasesRequest{
		Now:                       now,
		Reason:                    "automatic expired run lease recovery",
		Limit:                     100,
		WorkerHeartbeatStaleAfter: 2 * time.Minute,
	})
	if err != nil {
		t.Fatalf("RecoverExpiredRunLeases: %v", err)
	}
	if len(result.RequeuedRuns) != 1 || result.RequeuedRuns[0].RunID != staleRun.RunID {
		t.Fatalf("requeued runs = %+v, want only stale heartbeat run", result.RequeuedRuns)
	}
	if _, err := service.RenewRunLease(ctx, RenewRunLeaseRequest{
		RunID:      staleRun.RunID,
		LeaseToken: staleLease.LeaseToken,
		TTL:        time.Hour,
	}); !errors.Is(err, store.ErrConflict) {
		t.Fatalf("RenewRunLease stale old token err = %v, want ErrConflict", err)
	}
	if _, err := mem.RenewRunLease(ctx, domain.RenewRunLeaseInput{
		RunID:      activeRun.RunID,
		LeaseToken: activeLease.LeaseToken,
		TTL:        time.Hour,
		Now:        now,
	}); err != nil {
		t.Fatalf("RenewRunLease active token: %v", err)
	}
	events, err := mem.ListRunEvents(ctx, staleRun.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	last := events[len(events)-1]
	if last.EventKind != "run.requeued" || last.Payload["recovery"] != "stale_run_lease_worker_heartbeat" {
		t.Fatalf("last event = %+v, want stale heartbeat requeue", last)
	}
	if last.Payload["lease_worker_id"] != "worker-stale" {
		t.Fatalf("event payload = %+v, want stale lease worker id", last.Payload)
	}
}

func TestServiceRecoverExpiredRunLeasesDoesNotRequeueStaleHeartbeatWhenRunEventsAreFresh(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	now := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Recover active worker stream",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "long active worker analysis",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "long active worker analysis"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	drainJobs(bus)
	drainRunEvents(bus)

	lease, err := mem.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    run.RunID,
		WorkerID: "worker-active-stream",
		TTL:      20 * time.Minute,
		Now:      now.Add(-5 * time.Minute),
	})
	if err != nil {
		t.Fatalf("AcquireRunLease: %v", err)
	}
	if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt-active-stream-started",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.started",
		EventType: "run",
		Level:     "info",
		TS:        now.Add(-30 * time.Second),
		Message:   "Run started.",
		Payload:   domain.JSONMap{"worker_id": "worker-active-stream"},
	}); err != nil {
		t.Fatalf("IngestRunEvent started: %v", err)
	}
	drainRunEvents(bus)
	if _, err := mem.UpsertWorkerHeartbeat(ctx, domain.UpsertWorkerHeartbeatInput{
		WorkerID:        "worker-active-stream",
		WorkerKind:      "deepagents",
		Status:          "busy",
		CurrentRunID:    run.RunID,
		LastHeartbeatAt: now.Add(-5 * time.Minute),
	}); err != nil {
		t.Fatalf("UpsertWorkerHeartbeat: %v", err)
	}

	result, err := service.RecoverExpiredRunLeases(ctx, RecoverExpiredRunLeasesRequest{
		Now:                       now,
		Reason:                    "automatic expired run lease recovery",
		Limit:                     100,
		WorkerHeartbeatStaleAfter: 2 * time.Minute,
	})
	if err != nil {
		t.Fatalf("RecoverExpiredRunLeases: %v", err)
	}
	if len(result.RequeuedRuns) != 0 {
		t.Fatalf("requeued runs = %+v, want none while run events are fresh", result.RequeuedRuns)
	}
	if _, err := service.RenewRunLease(ctx, RenewRunLeaseRequest{
		RunID:      run.RunID,
		LeaseToken: lease.LeaseToken,
		TTL:        time.Hour,
		Now:        now,
	}); err != nil {
		t.Fatalf("RenewRunLease active token: %v", err)
	}
	select {
	case job := <-bus.Jobs():
		t.Fatalf("unexpected replacement job for active stream: %+v", job)
	default:
	}
}

func TestServiceRecoverExpiredRunLeasesRequeuesMissingLeaseOwnerHeartbeat(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	now := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Recover missing heartbeat owner",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "worker crashes before first heartbeat",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "worker crashes before first heartbeat"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	drainJobs(bus)
	drainRunEvents(bus)

	lease, err := mem.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    run.RunID,
		WorkerID: "worker-missing-heartbeat",
		TTL:      20 * time.Minute,
		Now:      now.Add(-5 * time.Minute),
	})
	if err != nil {
		t.Fatalf("AcquireRunLease: %v", err)
	}

	result, err := service.RecoverExpiredRunLeases(ctx, RecoverExpiredRunLeasesRequest{
		Now:                       now,
		Reason:                    "automatic expired run lease recovery",
		Limit:                     100,
		WorkerHeartbeatStaleAfter: 2 * time.Minute,
	})
	if err != nil {
		t.Fatalf("RecoverExpiredRunLeases: %v", err)
	}
	if len(result.RequeuedRuns) != 1 || result.RequeuedRuns[0].RunID != run.RunID {
		t.Fatalf("requeued runs = %+v, want missing-heartbeat run", result.RequeuedRuns)
	}
	if _, err := service.RenewRunLease(ctx, RenewRunLeaseRequest{
		RunID:      run.RunID,
		LeaseToken: lease.LeaseToken,
		TTL:        time.Hour,
	}); !errors.Is(err, store.ErrConflict) {
		t.Fatalf("RenewRunLease old token err = %v, want ErrConflict", err)
	}
	events, err := mem.ListRunEvents(ctx, run.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	last := events[len(events)-1]
	if last.EventKind != "run.requeued" || last.Payload["recovery"] != "stale_run_lease_worker_heartbeat" {
		t.Fatalf("last event = %+v, want stale heartbeat requeue", last)
	}
	if last.Payload["worker_heartbeat_missing"] != true {
		t.Fatalf("event payload = %+v, want worker_heartbeat_missing marker", last.Payload)
	}
	if _, ok := last.Payload["worker_last_heartbeat_at"]; ok {
		t.Fatalf("event payload = %+v, want no zero worker_last_heartbeat_at for missing heartbeat", last.Payload)
	}
	select {
	case job := <-bus.Jobs():
		if job.RunID != run.RunID || job.DispatchID == "" {
			t.Fatalf("replacement job = %+v, want redispatch for missing-heartbeat run", job)
		}
	default:
		t.Fatalf("expected replacement job for missing-heartbeat run")
	}
}

func TestServiceRecoverExpiredRunLeasesRedispatchesStaleQueuedRunWithoutLease(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Recover stale queued run",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	staleRun, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "long analysis whose dispatched job was lost",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "long analysis"}},
	})
	if err != nil {
		t.Fatalf("CreateRun stale: %v", err)
	}
	drainJobs(bus)
	drainRunEvents(bus)

	// The dispatched job was consumed and dropped: the run is queued with no
	// lease and no worker will ever claim it again without recovery.
	fresh, err := service.RecoverExpiredRunLeases(ctx, RecoverExpiredRunLeasesRequest{
		Now:    domain.Now().Add(30 * time.Second),
		Reason: "automatic expired run lease recovery",
		Limit:  100,
	})
	if err != nil {
		t.Fatalf("RecoverExpiredRunLeases fresh: %v", err)
	}
	if len(fresh.RequeuedRuns) != 0 {
		t.Fatalf("fresh queued run was requeued early: %+v", fresh.RequeuedRuns)
	}

	result, err := service.RecoverExpiredRunLeases(ctx, RecoverExpiredRunLeasesRequest{
		Now:    domain.Now().Add(10 * time.Minute),
		Reason: "automatic expired run lease recovery",
		Limit:  100,
	})
	if err != nil {
		t.Fatalf("RecoverExpiredRunLeases stale: %v", err)
	}
	if len(result.RequeuedRuns) != 1 || result.RequeuedRuns[0].RunID != staleRun.RunID {
		t.Fatalf("requeued runs = %+v, want stale queued run", result.RequeuedRuns)
	}

	requeuedJobs := 0
	for {
		select {
		case job := <-bus.Jobs():
			if job.RunID == staleRun.RunID && job.DispatchID != "" {
				requeuedJobs++
			}
		default:
			if requeuedJobs != 1 {
				t.Fatalf("requeued jobs = %d, want 1", requeuedJobs)
			}
			recovered, err := mem.GetRun(ctx, staleRun.RunID)
			if err != nil {
				t.Fatalf("GetRun recovered: %v", err)
			}
			if recovered.Status != domain.RunStatusQueued {
				t.Fatalf("recovered run status = %s, want queued", recovered.Status)
			}
			return
		}
	}
}

func TestServiceRecoverExpiredRunLeasesScansRecoverableStatusesBeyondTerminalHistory(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "recovery-user",
		Title:  "Recovery scan",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	staleRun, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "recovery-user",
		Goal:     "stale queued run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "stale queued run"}},
	})
	if err != nil {
		t.Fatalf("CreateRun stale: %v", err)
	}
	drainJobs(bus)
	drainRunEvents(bus)
	now := time.Date(2026, 6, 13, 12, 0, 0, 0, time.UTC)
	if _, err := mem.MarkRunDispatched(ctx, staleRun.RunID, now.Add(-30*time.Minute)); err != nil {
		t.Fatalf("MarkRunDispatched stale: %v", err)
	}

	for i := 0; i < 8; i++ {
		run, err := service.CreateRun(ctx, CreateRunRequest{
			ThreadID: thread.ThreadID,
			UserID:   "recovery-user",
			Goal:     fmt.Sprintf("terminal run %d", i),
			Messages: []domain.ThreadMessage{{Role: "user", Content: "terminal run"}},
		})
		if err != nil {
			t.Fatalf("CreateRun terminal %d: %v", i, err)
		}
		drainJobs(bus)
		if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
			EventID:   fmt.Sprintf("evt-terminal-%d", i),
			RunID:     run.RunID,
			ThreadID:  thread.ThreadID,
			EventKind: "run.completed",
			Payload:   domain.JSONMap{"response_text": "done"},
		}); err != nil {
			t.Fatalf("IngestRunEvent terminal %d: %v", i, err)
		}
	}
	drainRunEvents(bus)

	result, err := service.RecoverExpiredRunLeases(ctx, RecoverExpiredRunLeasesRequest{
		Now:                   now,
		Reason:                "automatic expired run lease recovery",
		Limit:                 5,
		RedispatchQueuedAfter: time.Minute,
	})
	if err != nil {
		t.Fatalf("RecoverExpiredRunLeases: %v", err)
	}
	for _, run := range result.RequeuedRuns {
		if run.RunID == staleRun.RunID {
			return
		}
	}
	t.Fatalf("requeued runs = %+v, want stale queued run %s despite newer terminal history", result.RequeuedRuns, staleRun.RunID)
}

func TestServiceRecoverExpiredRunLeasesDoesNotStormRedispatchedRun(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Recovery storm guard",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "stale run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "stale run"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	drainJobs(bus)
	drainRunEvents(bus)

	first, err := service.RecoverExpiredRunLeases(ctx, RecoverExpiredRunLeasesRequest{
		Now:   domain.Now().Add(10 * time.Minute),
		Limit: 100,
	})
	if err != nil {
		t.Fatalf("RecoverExpiredRunLeases first: %v", err)
	}
	if len(first.RequeuedRuns) != 1 {
		t.Fatalf("first pass requeued = %+v, want one run", first.RequeuedRuns)
	}

	// A second recovery pass shortly after the redispatch (e.g. the other
	// replica's loop ticking) must not redispatch the same run again: the
	// requeue refreshed job_dispatched_at.
	second, err := service.RecoverExpiredRunLeases(ctx, RecoverExpiredRunLeasesRequest{
		Now:   domain.Now().Add(30 * time.Second),
		Limit: 100,
	})
	if err != nil {
		t.Fatalf("RecoverExpiredRunLeases second: %v", err)
	}
	if len(second.RequeuedRuns) != 0 {
		t.Fatalf("second pass requeued = %+v, want none", second.RequeuedRuns)
	}
	_ = run
}

func TestServiceRecoverExpiredRunLeasesLeavesRunningLeaselessRunAlone(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Running leaseless run",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "legacy leaseless run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "legacy"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	if _, err := mem.UpdateRunStatus(ctx, run.RunID, domain.RunStatusRunning, "", ""); err != nil {
		t.Fatalf("UpdateRunStatus: %v", err)
	}
	drainJobs(bus)
	drainRunEvents(bus)

	result, err := service.RecoverExpiredRunLeases(ctx, RecoverExpiredRunLeasesRequest{
		Now:    domain.Now().Add(10 * time.Minute),
		Reason: "automatic expired run lease recovery",
		Limit:  100,
	})
	if err != nil {
		t.Fatalf("RecoverExpiredRunLeases: %v", err)
	}
	if len(result.RequeuedRuns) != 0 {
		t.Fatalf("running leaseless run was requeued: %+v", result.RequeuedRuns)
	}
}

func TestServiceIngestRunEventUpdatesLifecycleAndArtifacts(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{UserID: "user-1", Title: "ingest"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "ingest",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "ingest"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	drainRunEvents(bus)

	if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt-start",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.started",
		Payload:   domain.JSONMap{"status": "running"},
	}); err != nil {
		t.Fatalf("IngestRunEvent started: %v", err)
	}
	if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt-artifact",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "artifact.created",
		Message:   "Report",
		Payload: domain.JSONMap{
			"artifact_id": "artifact-python",
			"kind":        "report",
			"path":        "outputs/report.md",
			"title":       "Report",
			"mime_type":   "text/markdown",
			"size_bytes":  42,
			"sha256":      "abc123",
			"tool_name":   "save_report_output",
			"output_id":   "out-report",
		},
	}); err != nil {
		t.Fatalf("IngestRunEvent artifact: %v", err)
	}
	if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt-complete",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.completed",
		Message:   "done",
		Payload:   domain.JSONMap{"response_text": "done"},
	}); err != nil {
		t.Fatalf("IngestRunEvent completed: %v", err)
	}

	updated, err := mem.GetRun(ctx, run.RunID)
	if err != nil {
		t.Fatalf("GetRun: %v", err)
	}
	if updated.Status != domain.RunStatusSucceeded || updated.ResponseText != "done" {
		t.Fatalf("updated run = %+v, want succeeded with response text", updated)
	}
	artifacts, err := mem.ListRunArtifacts(ctx, run.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunArtifacts: %v", err)
	}
	if len(artifacts) != 1 || artifacts[0].Path != "outputs/report.md" {
		t.Fatalf("artifacts = %+v, want ingested artifact", artifacts)
	}
	if artifacts[0].ArtifactID != "artifact-python" {
		t.Fatalf("artifact id = %q, want Python artifact id", artifacts[0].ArtifactID)
	}
	if artifacts[0].Metadata["output_id"] != "out-report" {
		t.Fatalf("artifact metadata = %+v, want output id preserved", artifacts[0].Metadata)
	}
}

func TestServiceIngestRunEventWaitsForSourceSequencePredecessor(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{UserID: "user-1", Title: "ordering"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "preserve source order",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "go"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	drainRunEvents(bus)

	completed := domain.AppendRunEventInput{
		EventID:        "evt-ordering-000004",
		RunID:          run.RunID,
		ThreadID:       thread.ThreadID,
		EventKind:      "run.completed",
		SourceSequence: 4,
		Payload:        domain.JSONMap{"response_text": "done"},
	}
	if _, err := service.IngestRunEvent(ctx, completed); !errors.Is(err, ErrRunEventPredecessorPending) {
		t.Fatalf("IngestRunEvent completed before predecessors err = %v, want ErrRunEventPredecessorPending", err)
	}
	current, err := mem.GetRun(ctx, run.RunID)
	if err != nil {
		t.Fatalf("GetRun: %v", err)
	}
	if current.Status != domain.RunStatusQueued {
		t.Fatalf("run status = %s, want queued until source predecessors arrive", current.Status)
	}

	if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID:        "evt-ordering-000002",
		RunID:          run.RunID,
		ThreadID:       thread.ThreadID,
		EventKind:      "run.started",
		SourceSequence: 2,
	}); err != nil {
		t.Fatalf("IngestRunEvent source sequence 2: %v", err)
	}
	if _, err := service.IngestRunEvent(ctx, completed); !errors.Is(err, ErrRunEventPredecessorPending) {
		t.Fatalf("IngestRunEvent completed before sequence 3 err = %v, want ErrRunEventPredecessorPending", err)
	}
	if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID:        "evt-ordering-000003",
		RunID:          run.RunID,
		ThreadID:       thread.ThreadID,
		EventKind:      "message.delta",
		SourceSequence: 3,
		Message:        "almost done",
	}); err != nil {
		t.Fatalf("IngestRunEvent source sequence 3: %v", err)
	}
	if _, err := service.IngestRunEvent(ctx, completed); err != nil {
		t.Fatalf("IngestRunEvent completed after predecessors: %v", err)
	}

	current, err = mem.GetRun(ctx, run.RunID)
	if err != nil {
		t.Fatalf("GetRun after completed: %v", err)
	}
	if current.Status != domain.RunStatusSucceeded {
		t.Fatalf("run status = %s, want succeeded after ordered terminal event", current.Status)
	}
	events, err := mem.ListRunEventsAfter(ctx, run.RunID, 0, 10)
	if err != nil {
		t.Fatalf("ListRunEventsAfter: %v", err)
	}
	if len(events) != 4 {
		t.Fatalf("events = %d, want accepted plus 3 worker events", len(events))
	}
	for index, event := range events {
		wantSourceSequence := int64(index + 1)
		if event.SourceSequence != wantSourceSequence {
			t.Fatalf("event %d source sequence = %d, want %d (events=%+v)", index, event.SourceSequence, wantSourceSequence, events)
		}
	}
}

func TestServiceIngestRunCompletedPersistsAssistantMessageOnce(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{UserID: "user-1", Title: "completed transcript"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Create a durable report.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Create a durable report."}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	drainJobs(bus)

	completed := domain.AppendRunEventInput{
		EventID:   "evt-completed-transcript",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.completed",
		Payload:   domain.JSONMap{"response_text": "The durable report is ready."},
	}
	if _, err := service.IngestRunEvent(ctx, completed); err != nil {
		t.Fatalf("IngestRunEvent completed: %v", err)
	}
	if _, err := service.IngestRunEvent(ctx, completed); err != nil {
		t.Fatalf("IngestRunEvent duplicate completed: %v", err)
	}

	messages, err := mem.ListThreadMessages(ctx, thread.ThreadID)
	if err != nil {
		t.Fatalf("ListThreadMessages: %v", err)
	}
	if got, want := len(messages), 2; got != want {
		t.Fatalf("persisted messages = %d, want %d: %+v", got, want, messages)
	}
	if messages[0].Role != "user" || messages[0].RunID != run.RunID {
		t.Fatalf("user message = %+v, want run-owned user message", messages[0])
	}
	if messages[1].Role != "assistant" || messages[1].Content != "The durable report is ready." || messages[1].RunID != run.RunID {
		t.Fatalf("assistant message = %+v, want completed response owned by run %s", messages[1], run.RunID)
	}
}

func TestServiceIngestRunCompletedRecordsTokenUsageExactlyOnce(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{UserID: "user-tok", Title: "token usage"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-tok",
		Goal:     "Analyze tokens.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Analyze tokens."}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	drainJobs(bus)

	completed := domain.AppendRunEventInput{
		EventID:   "evt-completed-usage",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.completed",
		Payload: domain.JSONMap{
			"response_text": "Done.",
			// JSON-decoded payloads carry numbers as float64; mirror that here.
			"usage": domain.JSONMap{
				"input_tokens":  float64(1200),
				"output_tokens": float64(300),
				"total_tokens":  float64(1500),
				"model":         "deepseek_v4",
			},
		},
	}
	if _, err := service.IngestRunEvent(ctx, completed); err != nil {
		t.Fatalf("IngestRunEvent completed: %v", err)
	}
	// Redeliver the same terminal event; the increment must not double count.
	if _, err := service.IngestRunEvent(ctx, completed); err != nil {
		t.Fatalf("IngestRunEvent duplicate completed: %v", err)
	}

	stats, err := mem.GetUserTokenUsageStats(ctx, "user-tok")
	if err != nil {
		t.Fatalf("GetUserTokenUsageStats: %v", err)
	}
	if stats.InputTokens != 1200 || stats.OutputTokens != 300 || stats.TotalTokens != 1500 {
		t.Fatalf("lifetime usage = %+v, want single-count 1200/300/1500", stats)
	}
	if stats.PeakDailyTotal != 1500 {
		t.Fatalf("peak daily total = %d, want 1500", stats.PeakDailyTotal)
	}

	daily, err := mem.ListUserTokenUsageDaily(ctx, "user-tok", time.Time{})
	if err != nil {
		t.Fatalf("ListUserTokenUsageDaily: %v", err)
	}
	if len(daily) != 1 {
		t.Fatalf("daily rows = %d, want 1: %+v", len(daily), daily)
	}
	if daily[0].TotalTokens != 1500 || daily[0].RunCount != 1 {
		t.Fatalf("daily[0] = %+v, want total 1500 run_count 1", daily[0])
	}
}

func TestServiceIngestRunTokenUsageEventsAreDurableAndFinalizedOnce(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{UserID: "user-ledger", Title: "token ledger"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-ledger",
		Goal:     "Analyze tokens.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Analyze tokens."}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	drainJobs(bus)

	firstUsage := domain.AppendRunEventInput{
		EventID:   "evt-usage-call-1",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.token_usage",
		EventType: "run",
		Payload: domain.JSONMap{
			"usage_event_id": "model-call-1",
			"input_tokens":   float64(100),
			"output_tokens":  float64(20),
			"total_tokens":   float64(120),
			"model":          "deepseek_v4",
		},
	}
	if _, err := service.IngestRunEvent(ctx, firstUsage); err != nil {
		t.Fatalf("IngestRunEvent first usage: %v", err)
	}
	duplicateUsage := firstUsage
	duplicateUsage.EventID = "evt-usage-call-1-redelivered"
	if _, err := service.IngestRunEvent(ctx, duplicateUsage); err != nil {
		t.Fatalf("IngestRunEvent duplicate usage: %v", err)
	}
	if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt-usage-call-2",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.token_usage",
		EventType: "run",
		Payload: domain.JSONMap{
			"usage_event_id": "model-call-2",
			"input_tokens":   float64(50),
			"output_tokens":  float64(10),
			"total_tokens":   float64(60),
			"model":          "deepseek_v4",
		},
	}); err != nil {
		t.Fatalf("IngestRunEvent second usage: %v", err)
	}
	completed := domain.AppendRunEventInput{
		EventID:   "evt-ledger-completed",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.completed",
		Payload: domain.JSONMap{
			"response_text": "Done.",
		},
	}
	if _, err := service.IngestRunEvent(ctx, completed); err != nil {
		t.Fatalf("IngestRunEvent completed: %v", err)
	}
	if _, err := service.IngestRunEvent(ctx, completed); err != nil {
		t.Fatalf("IngestRunEvent duplicate completed: %v", err)
	}

	stats, err := mem.GetUserTokenUsageStats(ctx, "user-ledger")
	if err != nil {
		t.Fatalf("GetUserTokenUsageStats: %v", err)
	}
	if stats.InputTokens != 150 || stats.OutputTokens != 30 || stats.TotalTokens != 180 {
		t.Fatalf("lifetime usage = %+v, want deduped ledger total 150/30/180", stats)
	}
	daily, err := mem.ListUserTokenUsageDaily(ctx, "user-ledger", time.Time{})
	if err != nil {
		t.Fatalf("ListUserTokenUsageDaily: %v", err)
	}
	if len(daily) != 1 {
		t.Fatalf("daily rows = %d, want 1: %+v", len(daily), daily)
	}
	if daily[0].TotalTokens != 180 || daily[0].RunCount != 1 {
		t.Fatalf("daily[0] = %+v, want total 180 run_count 1", daily[0])
	}
}

func TestServiceIngestRunCompletedRecordsTokenUsageInPostgres(t *testing.T) {
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
	if err := store.ApplyPostgresSchema(ctx, pool); err != nil {
		t.Fatalf("ApplyPostgresSchema: %v", err)
	}
	pg := store.NewPostgresStore(pool)
	service := NewService(pg, eventbus.NewMemoryBus())

	userID := fmt.Sprintf("rc-tok-%d", time.Now().UnixNano())
	thread, err := service.CreateThread(ctx, CreateThreadRequest{UserID: userID, Title: "pg token usage"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   userID,
		Goal:     "Analyze tokens.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Analyze tokens."}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	completed := domain.AppendRunEventInput{
		EventID:   fmt.Sprintf("evt-%s-completed", run.RunID),
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.completed",
		Payload: domain.JSONMap{
			"response_text": "Done.",
			"usage": domain.JSONMap{
				"input_tokens":  float64(1200),
				"output_tokens": float64(300),
				"total_tokens":  float64(1500),
				"model":         "deepseek_v4",
			},
		},
	}
	if _, err := service.IngestRunEvent(ctx, completed); err != nil {
		t.Fatalf("IngestRunEvent completed: %v", err)
	}
	// Redeliver: the increment must remain single-counted through live Postgres.
	if _, err := service.IngestRunEvent(ctx, completed); err != nil {
		t.Fatalf("IngestRunEvent duplicate: %v", err)
	}

	stats, err := pg.GetUserTokenUsageStats(ctx, userID)
	if err != nil {
		t.Fatalf("GetUserTokenUsageStats: %v", err)
	}
	if stats.InputTokens != 1200 || stats.OutputTokens != 300 || stats.TotalTokens != 1500 {
		t.Fatalf("postgres lifetime usage = %+v, want single-count 1200/300/1500", stats)
	}
	if stats.PeakDailyTotal != 1500 {
		t.Fatalf("postgres peak daily = %d, want 1500", stats.PeakDailyTotal)
	}
}

func TestServiceIngestRunCompletedAppliesFirstGeneratedThreadTitle(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Run RareSpot on this prairie dog image and summarize burrow detections.",
		Metadata: domain.JSONMap{
			"frontend_bridge": "v2-chat",
			"title_state": domain.JSONMap{
				"source":   "auto",
				"strategy": "initial_request",
			},
		},
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Run RareSpot on this prairie dog image and summarize burrow detections.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Run RareSpot on this prairie dog image and summarize burrow detections."}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	drainJobs(bus)

	if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt-title-completed",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.completed",
		Payload: domain.JSONMap{
			"response_text":      "RareSpot completed.",
			"conversation_title": "RareSpot Prairie Dog Analysis",
			"title_generation":   domain.JSONMap{"strategy": "llm", "model": "deepseek_v4"},
		},
	}); err != nil {
		t.Fatalf("IngestRunEvent completed: %v", err)
	}

	updated, err := mem.GetThread(ctx, thread.ThreadID)
	if err != nil {
		t.Fatalf("GetThread: %v", err)
	}
	if updated.Title != "RareSpot Prairie Dog Analysis" {
		t.Fatalf("thread title = %q, want generated title", updated.Title)
	}
	titleState, ok := updated.Metadata["title_state"].(domain.JSONMap)
	if !ok {
		t.Fatalf("title_state = %#v, want metadata map", updated.Metadata["title_state"])
	}
	if titleState["source"] != "generated" || titleState["run_id"] != run.RunID || titleState["strategy"] != "llm" {
		t.Fatalf("title_state = %+v, want generated metadata for run", titleState)
	}

	secondRun, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Follow up.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Follow up."}},
	})
	if err != nil {
		t.Fatalf("CreateRun second: %v", err)
	}
	drainJobs(bus)
	if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt-title-second-completed",
		RunID:     secondRun.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.completed",
		Payload: domain.JSONMap{
			"response_text":      "Second run done.",
			"conversation_title": "Generic Follow Up",
			"title_generation":   domain.JSONMap{"strategy": "llm", "model": "deepseek_v4"},
		},
	}); err != nil {
		t.Fatalf("IngestRunEvent second completed: %v", err)
	}
	sticky, err := mem.GetThread(ctx, thread.ThreadID)
	if err != nil {
		t.Fatalf("GetThread sticky: %v", err)
	}
	if sticky.Title != "RareSpot Prairie Dog Analysis" {
		t.Fatalf("thread title after second completion = %q, want first generated title to stay sticky", sticky.Title)
	}
}

func TestServiceIngestRunEventIsIdempotentForDuplicateWorkerEvents(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{UserID: "user-1", Title: "duplicate events"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "duplicate events",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	drainRunEvents(bus)

	input := domain.AppendRunEventInput{
		EventID:   "evt-artifact",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "artifact.created",
		Message:   "Plot",
		Payload: domain.JSONMap{
			"artifact_id": "artifact-python",
			"kind":        "figure",
			"path":        "outputs/plot.png",
			"title":       "Plot",
			"mime_type":   "image/png",
			"size_bytes":  123,
			"sha256":      "abc123",
		},
	}
	if _, err := service.IngestRunEvent(ctx, input); err != nil {
		t.Fatalf("IngestRunEvent first: %v", err)
	}
	drainRunEvents(bus)
	if _, err := service.IngestRunEvent(ctx, input); err != nil {
		t.Fatalf("IngestRunEvent duplicate: %v", err)
	}

	events, err := mem.ListRunEvents(ctx, run.RunID, 20)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	var artifactEvents int
	for _, event := range events {
		if event.EventID == "evt-artifact" {
			artifactEvents++
		}
	}
	if artifactEvents != 1 {
		t.Fatalf("artifact event count = %d, want 1; events=%+v", artifactEvents, events)
	}
	artifacts, err := mem.ListRunArtifacts(ctx, run.RunID, 20)
	if err != nil {
		t.Fatalf("ListRunArtifacts: %v", err)
	}
	if len(artifacts) != 1 || artifacts[0].ArtifactID != "artifact-python" {
		t.Fatalf("artifacts = %+v, want one Python artifact", artifacts)
	}
	select {
	case event := <-bus.Events():
		if event.EventID != "evt-artifact" || event.EventKind != "artifact.created" {
			t.Fatalf("duplicate fanout event = %+v, want original artifact event", event)
		}
	case <-time.After(time.Second):
		t.Fatalf("expected duplicate event to retry fanout with same event id")
	}
}

func TestServiceIngestDuplicateArtifactEventWithoutArtifactIDCreatesOneArtifact(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{UserID: "user-1", Title: "artifact replay"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "artifact replay",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	drainRunEvents(bus)

	input := domain.AppendRunEventInput{
		EventID:   "evt-artifact-no-id",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "artifact.created",
		Message:   "Plot without artifact id",
		Payload: domain.JSONMap{
			"kind":       "figure",
			"path":       "outputs/plot.png",
			"title":      "Plot",
			"mime_type":  "image/png",
			"size_bytes": 123,
			"sha256":     "abc123",
		},
	}
	if _, err := service.IngestRunEvent(ctx, input); err != nil {
		t.Fatalf("IngestRunEvent first: %v", err)
	}
	if _, err := service.IngestRunEvent(ctx, input); err != nil {
		t.Fatalf("IngestRunEvent duplicate: %v", err)
	}

	artifacts, err := mem.ListRunArtifacts(ctx, run.RunID, 20)
	if err != nil {
		t.Fatalf("ListRunArtifacts: %v", err)
	}
	if len(artifacts) != 1 {
		t.Fatalf("artifacts = %+v, want one artifact after duplicate event replay", artifacts)
	}
	if artifacts[0].ArtifactID != "artifact_evt-artifact-no-id" {
		t.Fatalf("artifact id = %q, want deterministic event-derived id", artifacts[0].ArtifactID)
	}
}

func TestServiceIngestRunEventSerializesConcurrentDuplicateEventIDs(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	base := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	serviceStore := &duplicateEventRaceStore{
		MemoryStore: base,
		targetID:    "evt-concurrent-duplicate",
		ready:       make(chan struct{}),
		release:     make(chan struct{}),
	}
	service := NewService(serviceStore, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{UserID: "user-1", Title: "duplicate race"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "duplicate race",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	input := domain.AppendRunEventInput{
		EventID:   serviceStore.targetID,
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "message.delta",
		Message:   "duplicate",
	}
	var wg sync.WaitGroup
	errs := make(chan error, 2)
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := service.IngestRunEvent(ctx, input)
			errs <- err
		}()
	}
	select {
	case <-serviceStore.ready:
	case <-time.After(time.Second):
		t.Fatalf("duplicate ingest calls did not reach the race barrier")
	}
	close(serviceStore.release)
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("IngestRunEvent concurrent duplicate: %v", err)
		}
	}

	events, err := base.ListRunEvents(ctx, run.RunID, 20)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	duplicates := 0
	for _, event := range events {
		if event.EventID == serviceStore.targetID {
			duplicates++
		}
	}
	if duplicates != 1 {
		t.Fatalf("stored duplicate event count = %d, want 1; events=%+v", duplicates, events)
	}
}

func TestServiceIngestDuplicateHeartbeatDoesNotRegressTerminalRun(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{UserID: "user-1", Title: "heartbeat replay"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "heartbeat replay",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	drainRunEvents(bus)

	heartbeat := domain.AppendRunEventInput{
		EventID:   "evt-heartbeat-before-terminal",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.heartbeat",
		Message:   "still working",
	}
	if _, err := service.IngestRunEvent(ctx, heartbeat); err != nil {
		t.Fatalf("IngestRunEvent heartbeat: %v", err)
	}
	if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt-terminal-after-heartbeat",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.completed",
		Payload:   domain.JSONMap{"response_text": "done"},
	}); err != nil {
		t.Fatalf("IngestRunEvent completed: %v", err)
	}
	drainRunEvents(bus)

	if _, err := service.IngestRunEvent(ctx, heartbeat); err != nil {
		t.Fatalf("IngestRunEvent duplicate heartbeat: %v", err)
	}

	updated, err := mem.GetRun(ctx, run.RunID)
	if err != nil {
		t.Fatalf("GetRun: %v", err)
	}
	if updated.Status != domain.RunStatusSucceeded {
		t.Fatalf("run status = %s, want succeeded after duplicate pre-terminal heartbeat", updated.Status)
	}
	if updated.ResponseText != "done" {
		t.Fatalf("response text = %q, want terminal response text", updated.ResponseText)
	}
}

func TestServiceIngestDuplicateEventUsesStoredEventForSideEffects(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{UserID: "user-1", Title: "duplicate collision"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "duplicate collision",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	drainRunEvents(bus)

	storedInput := domain.AppendRunEventInput{
		EventID:   "evt-terminal",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.completed",
		Payload:   domain.JSONMap{"response_text": "verified answer"},
	}
	if _, err := service.IngestRunEvent(ctx, storedInput); err != nil {
		t.Fatalf("IngestRunEvent stored: %v", err)
	}
	drainRunEvents(bus)

	mutatedReplay := domain.AppendRunEventInput{
		EventID:   "evt-terminal",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.failed",
		Message:   "mutated replay should not win",
		Payload:   domain.JSONMap{"error": "mutated replay should not corrupt status"},
	}
	if _, err := service.IngestRunEvent(ctx, mutatedReplay); err != nil {
		t.Fatalf("IngestRunEvent duplicate replay: %v", err)
	}

	updated, err := mem.GetRun(ctx, run.RunID)
	if err != nil {
		t.Fatalf("GetRun: %v", err)
	}
	if updated.Status != domain.RunStatusSucceeded {
		t.Fatalf("run status = %s, want succeeded from stored event", updated.Status)
	}
	if updated.ResponseText != "verified answer" {
		t.Fatalf("response text = %q, want stored completed response", updated.ResponseText)
	}
	if updated.Error != "" {
		t.Fatalf("run error = %q, want no error from mutated duplicate payload", updated.Error)
	}
	select {
	case event := <-bus.Events():
		if event.EventID != "evt-terminal" || event.EventKind != "run.completed" {
			t.Fatalf("duplicate fanout event = %+v, want stored completed event", event)
		}
	case <-time.After(time.Second):
		t.Fatalf("expected duplicate replay to fan out stored event")
	}
}

func TestServiceIngestDuplicateArtifactEventDoesNotTrustMutatedReplay(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{UserID: "user-1", Title: "artifact replay"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "artifact replay",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	drainRunEvents(bus)

	storedInput := domain.AppendRunEventInput{
		EventID:   "evt-artifact",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "artifact.created",
		Message:   "Good plot",
		Payload: domain.JSONMap{
			"artifact_id": "artifact-good",
			"kind":        "figure",
			"path":        "outputs/good.png",
			"title":       "Good plot",
			"mime_type":   "image/png",
		},
	}
	if _, err := service.IngestRunEvent(ctx, storedInput); err != nil {
		t.Fatalf("IngestRunEvent stored artifact: %v", err)
	}
	drainRunEvents(bus)

	mutatedReplay := domain.AppendRunEventInput{
		EventID:   "evt-artifact",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "artifact.created",
		Message:   "Bad replay plot",
		Payload: domain.JSONMap{
			"artifact_id": "artifact-bad",
			"kind":        "figure",
			"path":        "outputs/bad.png",
			"title":       "Bad replay plot",
			"mime_type":   "image/png",
		},
	}
	if _, err := service.IngestRunEvent(ctx, mutatedReplay); err != nil {
		t.Fatalf("IngestRunEvent duplicate artifact replay: %v", err)
	}

	artifacts, err := mem.ListRunArtifacts(ctx, run.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunArtifacts: %v", err)
	}
	if len(artifacts) != 1 {
		t.Fatalf("artifact count = %d, want only stored artifact; artifacts=%+v", len(artifacts), artifacts)
	}
	if artifacts[0].ArtifactID != "artifact-good" || artifacts[0].Path != "outputs/good.png" {
		t.Fatalf("artifact = %+v, want stored artifact metadata", artifacts[0])
	}
	select {
	case event := <-bus.Events():
		if event.EventID != "evt-artifact" || event.Payload["artifact_id"] != "artifact-good" {
			t.Fatalf("duplicate fanout event = %+v, want stored artifact event", event)
		}
	case <-time.After(time.Second):
		t.Fatalf("expected duplicate replay to fan out stored artifact event")
	}
}

func TestServiceIngestDuplicateTerminalEventRetriesLifecycleSideEffects(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := &failOnceStatusStore{
		MemoryStore: store.NewMemoryStore(),
		failStatus:  domain.RunStatusSucceeded,
		err:         errors.New("temporary status write failure"),
	}
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{UserID: "user-1", Title: "retry terminal"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "retry terminal",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	drainRunEvents(bus)

	input := domain.AppendRunEventInput{
		EventID:   "evt-complete",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.completed",
		Payload:   domain.JSONMap{"response_text": "final answer"},
	}
	if _, err := service.IngestRunEvent(ctx, input); err == nil {
		t.Fatalf("first IngestRunEvent error = nil, want transient status write failure")
	}
	afterFailure, err := mem.GetRun(ctx, run.RunID)
	if err != nil {
		t.Fatalf("GetRun after failure: %v", err)
	}
	if afterFailure.Status == domain.RunStatusSucceeded {
		t.Fatalf("run unexpectedly succeeded after injected status failure: %+v", afterFailure)
	}

	if _, err := service.IngestRunEvent(ctx, input); err != nil {
		t.Fatalf("second IngestRunEvent duplicate should finish lifecycle: %v", err)
	}
	updated, err := mem.GetRun(ctx, run.RunID)
	if err != nil {
		t.Fatalf("GetRun: %v", err)
	}
	if updated.Status != domain.RunStatusSucceeded || updated.ResponseText != "final answer" {
		t.Fatalf("updated run = %+v, want succeeded with response text after duplicate retry", updated)
	}
	events, err := mem.ListRunEvents(ctx, run.RunID, 20)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	var completedEvents int
	for _, event := range events {
		if event.EventID == "evt-complete" {
			completedEvents++
		}
	}
	if completedEvents != 1 {
		t.Fatalf("completed event count = %d, want 1; events=%+v", completedEvents, events)
	}
}

func TestServiceCompletedRunDoesNotPersistAssistantWhenTerminalStatusFails(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := &failOnceStatusStore{
		MemoryStore: store.NewMemoryStore(),
		failStatus:  domain.RunStatusSucceeded,
		err:         errors.New("temporary terminal write failure"),
	}
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{UserID: "user-1", Title: "atomic terminal"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "finish atomically",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "finish atomically"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	drainRunEvents(bus)

	input := domain.AppendRunEventInput{
		EventID:   "evt-atomic-complete",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.completed",
		Payload:   domain.JSONMap{"response_text": "durable answer"},
	}
	if _, err := service.IngestRunEvent(ctx, input); err == nil {
		t.Fatalf("first IngestRunEvent error = nil, want transient terminal write failure")
	}
	messagesAfterFailure, err := mem.ListThreadMessages(ctx, thread.ThreadID)
	if err != nil {
		t.Fatalf("ListThreadMessages after failed terminal write: %v", err)
	}
	if got, want := len(messagesAfterFailure), 1; got != want {
		t.Fatalf("messages after failed terminal write = %d, want %d user-only messages: %+v", got, want, messagesAfterFailure)
	}

	if _, err := service.IngestRunEvent(ctx, input); err != nil {
		t.Fatalf("duplicate completed event should finish atomic terminal write: %v", err)
	}
	messagesAfterRetry, err := mem.ListThreadMessages(ctx, thread.ThreadID)
	if err != nil {
		t.Fatalf("ListThreadMessages after retry: %v", err)
	}
	if got, want := len(messagesAfterRetry), 2; got != want {
		t.Fatalf("messages after retry = %d, want %d user+assistant messages: %+v", got, want, messagesAfterRetry)
	}
	if messagesAfterRetry[1].Role != "assistant" || messagesAfterRetry[1].Content != "durable answer" || messagesAfterRetry[1].RunID != run.RunID {
		t.Fatalf("assistant message after retry = %+v, want durable answer owned by run", messagesAfterRetry[1])
	}
	updated, err := mem.GetRun(ctx, run.RunID)
	if err != nil {
		t.Fatalf("GetRun after retry: %v", err)
	}
	if updated.Status != domain.RunStatusSucceeded || updated.ResponseText != "durable answer" {
		t.Fatalf("updated run after retry = %+v, want succeeded with response text", updated)
	}
}

func TestServiceIngestDuplicateTerminalEventRetriesFanoutAfterPublishFailure(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := &failOnceRunEventBus{
		MemoryBus: eventbus.NewMemoryBus(),
		matchKind: "run.completed",
		err:       errors.New("temporary fanout failure"),
	}
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{UserID: "user-1", Title: "retry fanout"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "retry fanout",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	drainRunEvents(bus.MemoryBus)

	input := domain.AppendRunEventInput{
		EventID:   "evt-complete",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.completed",
		Payload:   domain.JSONMap{"response_text": "final answer"},
	}
	if _, err := service.IngestRunEvent(ctx, input); err == nil {
		t.Fatalf("first IngestRunEvent error = nil, want transient fanout failure")
	}
	afterFailure, err := mem.GetRun(ctx, run.RunID)
	if err != nil {
		t.Fatalf("GetRun after failure: %v", err)
	}
	if afterFailure.Status != domain.RunStatusSucceeded {
		t.Fatalf("run status = %s, want succeeded even though fanout failed", afterFailure.Status)
	}
	if _, err := service.IngestRunEvent(ctx, input); err != nil {
		t.Fatalf("duplicate terminal IngestRunEvent should retry fanout: %v", err)
	}

	select {
	case event := <-bus.Events():
		if event.EventID != "evt-complete" || event.EventKind != "run.completed" {
			t.Fatalf("fanned event = %+v, want original completed event", event)
		}
	case <-time.After(time.Second):
		t.Fatalf("expected duplicate terminal event to be fanned out after retry")
	}
	events, err := mem.ListRunEvents(ctx, run.RunID, 20)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	var completedEvents int
	for _, event := range events {
		if event.EventID == "evt-complete" {
			completedEvents++
		}
	}
	if completedEvents != 1 {
		t.Fatalf("completed event count = %d, want 1; events=%+v", completedEvents, events)
	}
}

func TestServiceIngestRunEventDropsLateEventsAfterTerminalRun(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{UserID: "user-1", Title: "late events"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "late events",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	drainRunEvents(bus)

	if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt-complete",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.completed",
		Payload:   domain.JSONMap{"response_text": "final answer"},
	}); err != nil {
		t.Fatalf("IngestRunEvent completed: %v", err)
	}
	drainRunEvents(bus)

	if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt-late",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "message.delta",
		Payload:   domain.JSONMap{"text": "late text"},
	}); err != nil {
		t.Fatalf("IngestRunEvent late: %v", err)
	}

	updated, err := mem.GetRun(ctx, run.RunID)
	if err != nil {
		t.Fatalf("GetRun: %v", err)
	}
	if updated.Status != domain.RunStatusSucceeded || updated.ResponseText != "final answer" {
		t.Fatalf("updated run = %+v, want terminal state preserved", updated)
	}
	events, err := mem.ListRunEvents(ctx, run.RunID, 20)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	for _, event := range events {
		if event.EventID == "evt-late" {
			t.Fatalf("late event should not be persisted after terminal run: %+v", events)
		}
	}
	select {
	case event := <-bus.Events():
		t.Fatalf("late event should not be fanned out: %+v", event)
	default:
	}
}

func TestServiceIngestRunEventDropsUnknownRunEvent(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	event, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt-missing-run-completed",
		RunID:     "run-missing",
		ThreadID:  "thread-missing",
		EventKind: "run.completed",
		Payload:   domain.JSONMap{"response_text": "stale event"},
	})
	if err != nil {
		t.Fatalf("IngestRunEvent missing run error = %v, want stale event dropped", err)
	}
	if event.EventID != "evt-missing-run-completed" || event.RunID != "run-missing" {
		t.Fatalf("dropped event = %+v, want metadata preserved for ack/drop path", event)
	}
}

func TestServiceCreateRunReusesExistingRunForIdempotencyKey(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Idempotent thread",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}

	req := CreateRunRequest{
		ThreadID:         thread.ThreadID,
		UserID:           "user-1",
		Goal:             "Run a long autonomous analysis.",
		Messages:         []domain.ThreadMessage{{Role: "user", Content: "Run a long autonomous analysis."}},
		IdempotencyKey:   "prompt-key-1",
		SelectionContext: domain.JSONMap{"source": "chat"},
	}
	first, err := service.CreateRun(ctx, req)
	if err != nil {
		t.Fatalf("CreateRun first: %v", err)
	}
	second, err := service.CreateRun(ctx, req)
	if err != nil {
		t.Fatalf("CreateRun second: %v", err)
	}

	if second.RunID != first.RunID {
		t.Fatalf("second run id = %q, want original %q", second.RunID, first.RunID)
	}
	events, err := mem.ListRunEvents(ctx, first.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	if len(events) != 1 || events[0].EventKind != "run.accepted" {
		t.Fatalf("events = %+v, want single run.accepted", events)
	}
	select {
	case <-bus.Jobs():
	case <-time.After(time.Second):
		t.Fatalf("expected first job")
	}
	select {
	case job := <-bus.Jobs():
		t.Fatalf("unexpected duplicate job: %+v", job)
	default:
	}
	messages, err := mem.ListThreadMessages(ctx, thread.ThreadID)
	if err != nil {
		t.Fatalf("ListThreadMessages: %v", err)
	}
	if len(messages) != 1 {
		t.Fatalf("thread messages = %d, want exactly one user message", len(messages))
	}
}

func TestServiceCreateRunConcurrentRetriesReuseOneRun(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := newRacingIdempotencyStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Concurrent idempotency",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}

	req := CreateRunRequest{
		ThreadID:       thread.ThreadID,
		UserID:         "user-1",
		Goal:           "Run one expensive job.",
		Messages:       []domain.ThreadMessage{{Role: "user", Content: "Run one expensive job."}},
		IdempotencyKey: "prompt-key-concurrent",
	}
	start := make(chan struct{})
	var wg sync.WaitGroup
	results := make([]domain.RunRecord, 2)
	errs := make([]error, 2)
	for index := range results {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			<-start
			results[index], errs[index] = service.CreateRun(ctx, req)
		}(index)
	}
	close(start)
	wg.Wait()

	for index, err := range errs {
		if err != nil {
			t.Fatalf("CreateRun %d: %v", index, err)
		}
	}
	if results[0].RunID != results[1].RunID {
		t.Fatalf("concurrent run ids = %q and %q, want same run", results[0].RunID, results[1].RunID)
	}
	events, err := mem.ListRunEvents(ctx, results[0].RunID, 10)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	if len(events) != 1 || events[0].EventKind != "run.accepted" {
		t.Fatalf("events = %+v, want one accepted event", events)
	}
	select {
	case <-bus.Jobs():
	case <-time.After(time.Second):
		t.Fatalf("expected one dispatched job")
	}
	select {
	case job := <-bus.Jobs():
		t.Fatalf("unexpected duplicate job: %+v", job)
	default:
	}
}

func TestServiceCreateRunRecoversExistingRunAfterStoreIdempotencyConflict(t *testing.T) {
	ctx := context.Background()
	base := store.NewMemoryStore()
	thread, err := base.CreateThread(ctx, domain.CreateThreadInput{
		UserID: "user-1",
		Title:  "Cross instance idempotency",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	existing, err := base.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Only one long run.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Only one long run."}},
		Metadata: domain.JSONMap{"idempotency_key": "same-browser-submit"},
	})
	if err != nil {
		t.Fatalf("seed CreateRun: %v", err)
	}
	mem := &conflictingIdempotencyStore{MemoryStore: base}
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	recovered, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID:       thread.ThreadID,
		UserID:         "user-1",
		Goal:           "Only one long run.",
		Messages:       []domain.ThreadMessage{{Role: "user", Content: "Only one long run."}},
		IdempotencyKey: "same-browser-submit",
	})
	if err != nil {
		t.Fatalf("CreateRun should recover existing idempotent run after store conflict: %v", err)
	}
	if recovered.RunID != existing.RunID {
		t.Fatalf("recovered run id = %s, want existing %s", recovered.RunID, existing.RunID)
	}
	events, err := base.ListRunEvents(ctx, existing.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	if len(events) != 1 || events[0].EventID != "evt_"+existing.RunID+"_accepted" {
		t.Fatalf("events = %+v, want one stable accepted event", events)
	}
	select {
	case job := <-bus.Jobs():
		if job.RunID != existing.RunID {
			t.Fatalf("job run id = %s, want %s", job.RunID, existing.RunID)
		}
	case <-time.After(time.Second):
		t.Fatalf("expected recovery dispatch job")
	}
}

func TestServiceCompletedRunPersistsAssistantBeforeTerminalStatusVisible(t *testing.T) {
	ctx := context.Background()
	mem := &blockingAppendThreadMessageStore{
		MemoryStore: store.NewMemoryStore(),
		started:     make(chan struct{}),
		release:     make(chan struct{}),
	}
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)
	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Terminal transcript ordering",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Complete with transcript.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Complete with transcript."}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	drainJobs(bus)
	if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt_" + run.RunID + "_started",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.started",
	}); err != nil {
		t.Fatalf("IngestRunEvent started: %v", err)
	}

	result := make(chan error, 1)
	go func() {
		_, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
			EventID:   "evt_" + run.RunID + "_completed",
			RunID:     run.RunID,
			ThreadID:  thread.ThreadID,
			EventKind: "run.completed",
			Payload:   domain.JSONMap{"response_text": "terminal answer"},
		})
		result <- err
	}()

	select {
	case <-mem.started:
	case <-time.After(time.Second):
		t.Fatalf("assistant append was not attempted")
	}
	visible, err := mem.GetRun(ctx, run.RunID)
	if err != nil {
		t.Fatalf("GetRun while assistant append is blocked: %v", err)
	}
	if visible.Status == domain.RunStatusSucceeded {
		t.Fatalf("run status became succeeded before assistant message append completed")
	}
	close(mem.release)
	select {
	case err := <-result:
		if err != nil {
			t.Fatalf("IngestRunEvent completed: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatalf("completed event did not finish after releasing assistant append")
	}
	terminal, err := mem.GetRun(ctx, run.RunID)
	if err != nil {
		t.Fatalf("GetRun terminal: %v", err)
	}
	if terminal.Status != domain.RunStatusSucceeded {
		t.Fatalf("terminal status = %s, want succeeded", terminal.Status)
	}
	messages, err := mem.ListThreadMessages(ctx, thread.ThreadID)
	if err != nil {
		t.Fatalf("ListThreadMessages: %v", err)
	}
	if len(messages) != 2 || messages[1].Role != "assistant" || messages[1].Content != "terminal answer" {
		t.Fatalf("messages = %+v, want user plus assistant terminal answer", messages)
	}
}

func TestServiceControlPlaneSoakConcurrentIdempotentRunsAndWorkerEvents(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Control-plane soak",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}

	const uniqueRuns = 40
	const duplicateSubmitsPerRun = 4
	const deltasPerRun = 32

	start := make(chan struct{})
	var wg sync.WaitGroup
	type submitResult struct {
		key string
		run domain.RunRecord
		err error
	}
	results := make(chan submitResult, uniqueRuns*duplicateSubmitsPerRun)
	for runIndex := 0; runIndex < uniqueRuns; runIndex++ {
		key := "soak-key-" + twoDigit(runIndex)
		for submitIndex := 0; submitIndex < duplicateSubmitsPerRun; submitIndex++ {
			wg.Add(1)
			go func(runIndex int, key string) {
				defer wg.Done()
				<-start
				goal := "Run autonomous soak task " + twoDigit(runIndex)
				run, err := service.CreateRun(ctx, CreateRunRequest{
					ThreadID:       thread.ThreadID,
					UserID:         "user-1",
					Goal:           goal,
					Messages:       []domain.ThreadMessage{{Role: "user", Content: goal}},
					IdempotencyKey: key,
				})
				results <- submitResult{key: key, run: run, err: err}
			}(runIndex, key)
		}
	}
	close(start)
	wg.Wait()
	close(results)

	runsByKey := map[string]string{}
	for result := range results {
		if result.err != nil {
			t.Fatalf("CreateRun for %s: %v", result.key, result.err)
		}
		if existing := runsByKey[result.key]; existing != "" && existing != result.run.RunID {
			t.Fatalf("idempotency key %s produced runs %s and %s", result.key, existing, result.run.RunID)
		}
		runsByKey[result.key] = result.run.RunID
	}
	if len(runsByKey) != uniqueRuns {
		t.Fatalf("unique run count = %d, want %d: %+v", len(runsByKey), uniqueRuns, runsByKey)
	}

	jobsByRunID := map[string]int{}
	for i := 0; i < uniqueRuns; i++ {
		select {
		case job := <-bus.Jobs():
			jobsByRunID[job.RunID]++
		case <-time.After(time.Second):
			t.Fatalf("received %d/%d jobs before timeout", i, uniqueRuns)
		}
	}
	select {
	case job := <-bus.Jobs():
		t.Fatalf("unexpected duplicate job after soak submissions: %+v", job)
	default:
	}
	for key, runID := range runsByKey {
		if jobsByRunID[runID] != 1 {
			t.Fatalf("run %s for key %s job count = %d, want 1; jobs=%+v", runID, key, jobsByRunID[runID], jobsByRunID)
		}
	}

	var eventWG sync.WaitGroup
	for key, runID := range runsByKey {
		eventWG.Add(1)
		go func(key string, runID string) {
			defer eventWG.Done()
			_, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
				EventID:   "evt_" + runID + "_started",
				RunID:     runID,
				ThreadID:  thread.ThreadID,
				EventKind: "run.started",
				EventType: "run",
				Message:   "Worker started.",
				Payload:   domain.JSONMap{"key": key},
			})
			if err != nil {
				t.Errorf("IngestRunEvent started for %s: %v", runID, err)
				return
			}
			for index := 0; index < deltasPerRun; index++ {
				_, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
					EventID:   "evt_" + runID + "_delta_" + twoDigit(index),
					RunID:     runID,
					ThreadID:  thread.ThreadID,
					EventKind: "message.delta",
					EventType: "message",
					Message:   "delta",
					Payload:   domain.JSONMap{"text": "chunk " + twoDigit(index)},
				})
				if err != nil {
					t.Errorf("IngestRunEvent delta %d for %s: %v", index, runID, err)
					return
				}
			}
			_, err = service.IngestRunEvent(ctx, domain.AppendRunEventInput{
				EventID:   "evt_" + runID + "_completed",
				RunID:     runID,
				ThreadID:  thread.ThreadID,
				EventKind: "run.completed",
				EventType: "run",
				Message:   "Run completed.",
				Payload:   domain.JSONMap{"response_text": "completed " + key},
			})
			if err != nil {
				t.Errorf("IngestRunEvent completed for %s: %v", runID, err)
			}
		}(key, runID)
	}
	eventWG.Wait()

	for key, runID := range runsByKey {
		run, err := mem.GetRun(ctx, runID)
		if err != nil {
			t.Fatalf("GetRun %s: %v", runID, err)
		}
		if run.Status != domain.RunStatusSucceeded {
			t.Fatalf("run %s status = %s, want succeeded", runID, run.Status)
		}
		if run.ResponseText != "completed "+key {
			t.Fatalf("run %s response = %q, want completed %s", runID, run.ResponseText, key)
		}
		events, err := mem.ListRunEventsAfter(ctx, runID, 0, 1000)
		if err != nil {
			t.Fatalf("ListRunEventsAfter %s: %v", runID, err)
		}
		if got, want := len(events), deltasPerRun+3; got != want {
			t.Fatalf("run %s events = %d, want %d", runID, got, want)
		}
		if events[0].Sequence != 1 || events[len(events)-1].Sequence != int64(deltasPerRun+3) {
			t.Fatalf("run %s event sequence range = %d..%d, want 1..%d", runID, events[0].Sequence, events[len(events)-1].Sequence, deltasPerRun+3)
		}
		if events[len(events)-1].EventKind != "run.completed" {
			t.Fatalf("run %s last event = %s, want run.completed", runID, events[len(events)-1].EventKind)
		}
	}
	messages, err := mem.ListThreadMessages(ctx, thread.ThreadID)
	if err != nil {
		t.Fatalf("ListThreadMessages: %v", err)
	}
	if got, want := len(messages), uniqueRuns*2; got != want {
		t.Fatalf("thread messages = %d, want %d user+assistant messages", got, want)
	}
}

func TestServiceControlPlaneSoakHighVolumeToolEventsRemainReplayable(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "High-volume tool event soak",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID:       thread.ThreadID,
		UserID:         "user-1",
		Goal:           "Run a long autonomous tool-heavy analysis.",
		Messages:       []domain.ThreadMessage{{Role: "user", Content: "Run a long autonomous tool-heavy analysis."}},
		IdempotencyKey: "tool-heavy-soak",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	const toolIterations = 1000
	const artifactEvery = 10
	const heartbeatEvery = 25
	const deltaEvery = 20
	const replayPageSize = 137

	if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt_" + run.RunID + "_started",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.started",
		EventType: "run",
		Message:   "Worker started tool-heavy run.",
		Payload:   domain.JSONMap{"phase": "started"},
	}); err != nil {
		t.Fatalf("IngestRunEvent started: %v", err)
	}

	for index := 0; index < toolIterations; index++ {
		suffix := fmt.Sprintf("%04d", index)
		if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
			EventID:   "evt_" + run.RunID + "_tool_started_" + suffix,
			RunID:     run.RunID,
			ThreadID:  thread.ThreadID,
			EventKind: "tool_call.started",
			EventType: "tool_call",
			NodeName:  "execute",
			TaskID:    "tool-" + suffix,
			Message:   "execute started " + suffix,
			Payload: domain.JSONMap{
				"tool_name": "execute",
				"index":     index,
			},
		}); err != nil {
			t.Fatalf("IngestRunEvent tool started %d: %v", index, err)
		}
		if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
			EventID:   "evt_" + run.RunID + "_tool_completed_" + suffix,
			RunID:     run.RunID,
			ThreadID:  thread.ThreadID,
			EventKind: "tool_call.completed",
			EventType: "tool_call",
			NodeName:  "execute",
			TaskID:    "tool-" + suffix,
			Message:   "execute completed " + suffix,
			Payload: domain.JSONMap{
				"tool_name":   "execute",
				"index":       index,
				"duration_ms": 7,
			},
		}); err != nil {
			t.Fatalf("IngestRunEvent tool completed %d: %v", index, err)
		}
		if index%artifactEvery == 0 {
			if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
				EventID:   "evt_" + run.RunID + "_artifact_" + suffix,
				RunID:     run.RunID,
				ThreadID:  thread.ThreadID,
				EventKind: "artifact.created",
				EventType: "artifact",
				Message:   "Artifact " + suffix,
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
				t.Fatalf("IngestRunEvent artifact %d: %v", index, err)
			}
		}
		if index%heartbeatEvery == 0 {
			if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
				EventID:   "evt_" + run.RunID + "_heartbeat_" + suffix,
				RunID:     run.RunID,
				ThreadID:  thread.ThreadID,
				EventKind: "run.heartbeat",
				EventType: "run",
				Message:   "heartbeat " + suffix,
				Payload:   domain.JSONMap{"tool_iterations_completed": index + 1},
			}); err != nil {
				t.Fatalf("IngestRunEvent heartbeat %d: %v", index, err)
			}
		}
		if index%deltaEvery == 0 {
			if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
				EventID:   "evt_" + run.RunID + "_delta_" + suffix,
				RunID:     run.RunID,
				ThreadID:  thread.ThreadID,
				EventKind: "message.delta",
				EventType: "message",
				Message:   "coordinator delta " + suffix,
				Payload:   domain.JSONMap{"text": "coordinator delta " + suffix},
			}); err != nil {
				t.Fatalf("IngestRunEvent delta %d: %v", index, err)
			}
		}
	}
	if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt_" + run.RunID + "_completed",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.completed",
		EventType: "run",
		Message:   "Run completed.",
		Payload:   domain.JSONMap{"response_text": "Final coordinator answer from the tool-heavy run."},
	}); err != nil {
		t.Fatalf("IngestRunEvent completed: %v", err)
	}

	expectedArtifacts := toolIterations / artifactEvery
	expectedHeartbeats := toolIterations / heartbeatEvery
	expectedDeltas := toolIterations / deltaEvery
	expectedEvents := 1 + 1 + (toolIterations * 2) + expectedArtifacts + expectedHeartbeats + expectedDeltas + 1

	var replayed []domain.RunEventRecord
	var after int64
	for {
		page, err := mem.ListRunEventsAfter(ctx, run.RunID, after, replayPageSize)
		if err != nil {
			t.Fatalf("ListRunEventsAfter after=%d: %v", after, err)
		}
		if len(page) == 0 {
			break
		}
		for _, event := range page {
			if event.Sequence != after+1 {
				t.Fatalf("event sequence after %d = %d, want %d", after, event.Sequence, after+1)
			}
			after = event.Sequence
			replayed = append(replayed, event)
		}
		if len(page) > replayPageSize {
			t.Fatalf("page length = %d, want <= %d", len(page), replayPageSize)
		}
	}
	if got := len(replayed); got != expectedEvents {
		t.Fatalf("replayed events = %d, want %d", got, expectedEvents)
	}
	if replayed[0].EventKind != "run.accepted" {
		t.Fatalf("first event kind = %s, want run.accepted", replayed[0].EventKind)
	}
	if replayed[len(replayed)-1].EventKind != "run.completed" {
		t.Fatalf("last event kind = %s, want run.completed", replayed[len(replayed)-1].EventKind)
	}

	storedRun, err := mem.GetRun(ctx, run.RunID)
	if err != nil {
		t.Fatalf("GetRun: %v", err)
	}
	if storedRun.Status != domain.RunStatusSucceeded {
		t.Fatalf("run status = %s, want succeeded", storedRun.Status)
	}
	if storedRun.ResponseText != "Final coordinator answer from the tool-heavy run." {
		t.Fatalf("response text = %q, want final coordinator answer only", storedRun.ResponseText)
	}
	artifacts, err := mem.ListRunArtifacts(ctx, run.RunID, expectedArtifacts+1)
	if err != nil {
		t.Fatalf("ListRunArtifacts: %v", err)
	}
	if got := len(artifacts); got != expectedArtifacts {
		t.Fatalf("artifacts = %d, want %d", got, expectedArtifacts)
	}
	messages, err := mem.ListThreadMessages(ctx, thread.ThreadID)
	if err != nil {
		t.Fatalf("ListThreadMessages: %v", err)
	}
	if got, want := len(messages), 2; got != want {
		t.Fatalf("thread messages = %d, want user+assistant", got)
	}
	if messages[1].Role != "assistant" || messages[1].RunID != run.RunID {
		t.Fatalf("assistant message = %+v, want persisted terminal assistant answer for run", messages[1])
	}
}

func TestServiceCreateRunMarksRunFailedWhenJobDispatchFails(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := &failingJobBus{jobErr: errors.New("nats unavailable")}
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Dispatch failure",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}

	created, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID:       thread.ThreadID,
		UserID:         "user-1",
		Goal:           "Run a long autonomous analysis.",
		Messages:       []domain.ThreadMessage{{Role: "user", Content: "Run a long autonomous analysis."}},
		IdempotencyKey: "dispatch-key-1",
	})
	if err == nil {
		t.Fatalf("CreateRun error = nil, want job dispatch error")
	}
	run, found, err := mem.FindRunByIdempotencyKey(ctx, thread.ThreadID, "user-1", "dispatch-key-1")
	if err != nil {
		t.Fatalf("FindRunByIdempotencyKey: %v", err)
	}
	if !found {
		t.Fatalf("created run with idempotency key was not persisted; returned run=%+v", created)
	}
	if run.Status != domain.RunStatusFailed {
		t.Fatalf("run status = %s, want failed so it cannot remain queued forever", run.Status)
	}
	if run.Error == "" {
		t.Fatalf("run error is empty, want enqueue failure detail")
	}
	events, err := mem.ListRunEvents(ctx, run.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	if got, want := events[len(events)-1].EventKind, "run.failed"; got != want {
		t.Fatalf("last event kind = %q, want %q; events=%+v", got, want, events)
	}
	if events[len(events)-1].Payload["stage"] != "job_enqueue" {
		t.Fatalf("failure payload = %+v, want job_enqueue stage", events[len(events)-1].Payload)
	}
	if len(bus.jobs) != 1 {
		t.Fatalf("job publish attempts = %d, want exactly one", len(bus.jobs))
	}
}

func TestServiceCreateRunDoesNotMarkFailedWhenDispatchFailureEventAppendFails(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := &failingAppendEventStore{
		MemoryStore: store.NewMemoryStore(),
		matchKind:   "run.failed",
		err:         errors.New("event store unavailable"),
	}
	bus := &failingJobBus{jobErr: errors.New("nats unavailable")}
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Dispatch event append failure",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}

	created, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID:       thread.ThreadID,
		UserID:         "user-1",
		Goal:           "Run a long autonomous analysis.",
		Messages:       []domain.ThreadMessage{{Role: "user", Content: "Run a long autonomous analysis."}},
		IdempotencyKey: "dispatch-key-event-append-fails",
	})
	if err == nil {
		t.Fatalf("CreateRun error = nil, want event append failure")
	}
	run, found, err := mem.FindRunByIdempotencyKey(ctx, thread.ThreadID, "user-1", "dispatch-key-event-append-fails")
	if err != nil {
		t.Fatalf("FindRunByIdempotencyKey: %v", err)
	}
	if !found {
		t.Fatalf("created run with idempotency key was not persisted; returned run=%+v", created)
	}
	if run.Status == domain.RunStatusFailed {
		t.Fatalf("run status = %s, want non-terminal because run.failed was not durable", run.Status)
	}
	events, err := mem.ListRunEvents(ctx, run.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	for _, event := range events {
		if event.EventKind == "run.failed" {
			t.Fatalf("run.failed event should not be persisted when append fails: %+v", events)
		}
	}
}

func TestServiceCreateRunIdempotentRetryReconcilesStoredDispatchFailureEvent(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := &failOnceStatusStore{
		MemoryStore: store.NewMemoryStore(),
		failStatus:  domain.RunStatusFailed,
		err:         errors.New("temporary status write failure"),
	}
	bus := &failingJobBus{jobErr: errors.New("nats unavailable")}
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Dispatch status retry",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	req := CreateRunRequest{
		ThreadID:       thread.ThreadID,
		UserID:         "user-1",
		Goal:           "Run a long autonomous analysis.",
		Messages:       []domain.ThreadMessage{{Role: "user", Content: "Run a long autonomous analysis."}},
		IdempotencyKey: "dispatch-key-status-retry",
	}

	if _, err := service.CreateRun(ctx, req); err == nil {
		t.Fatalf("first CreateRun error = nil, want status write failure after dispatch failure event append")
	}
	run, found, err := mem.FindRunByIdempotencyKey(ctx, thread.ThreadID, "user-1", req.IdempotencyKey)
	if err != nil {
		t.Fatalf("FindRunByIdempotencyKey: %v", err)
	}
	if !found {
		t.Fatalf("created run with idempotency key was not persisted")
	}
	if run.Status == domain.RunStatusFailed {
		t.Fatalf("run unexpectedly failed after injected status failure: %+v", run)
	}
	events, err := mem.ListRunEvents(ctx, run.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	if got, want := events[len(events)-1].EventKind, "run.failed"; got != want {
		t.Fatalf("last stored event = %q, want durable %q; events=%+v", got, want, events)
	}

	reconciled, err := service.CreateRun(ctx, req)
	if err != nil {
		t.Fatalf("second CreateRun should reconcile stored failure event: %v", err)
	}
	if reconciled.Status != domain.RunStatusFailed {
		t.Fatalf("reconciled run status = %s, want failed from stored dispatch failure event", reconciled.Status)
	}
	if reconciled.Error == "" {
		t.Fatalf("reconciled run error is empty, want dispatch failure text")
	}
	events, err = mem.ListRunEvents(ctx, run.RunID, 20)
	if err != nil {
		t.Fatalf("ListRunEvents after retry: %v", err)
	}
	var failedEvents int
	for _, event := range events {
		if event.EventID == "evt_"+run.RunID+"_dispatch_failed" {
			failedEvents++
		}
	}
	if failedEvents != 1 {
		t.Fatalf("dispatch failure event count = %d, want 1; events=%+v", failedEvents, events)
	}
	if len(bus.jobs) != 1 {
		t.Fatalf("job publish attempts = %d, want original attempt only", len(bus.jobs))
	}
}

func TestServiceCreateRunStillDispatchesWhenAcceptedFanoutFails(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := &failingRunEventBus{eventErr: errors.New("event fanout unavailable")}
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Accepted fanout failure",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}

	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Run despite fanout failure.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Run despite fanout failure."}},
	})
	if err != nil {
		t.Fatalf("CreateRun should not fail when accepted-event fanout fails after durable append: %v", err)
	}
	updated, err := mem.GetRun(ctx, run.RunID)
	if err != nil {
		t.Fatalf("GetRun: %v", err)
	}
	if updated.Status != domain.RunStatusQueued {
		t.Fatalf("run status = %s, want queued for dispatched worker job", updated.Status)
	}
	if len(bus.jobs) != 1 || bus.jobs[0].RunID != run.RunID {
		t.Fatalf("jobs = %+v, want dispatched run job despite accepted fanout failure", bus.jobs)
	}
	events, err := mem.ListRunEvents(ctx, run.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	if len(events) != 1 || events[0].EventKind != "run.accepted" {
		t.Fatalf("stored events = %+v, want durable run.accepted", events)
	}
	if len(bus.events) != 1 || bus.events[0].EventKind != "run.accepted" {
		t.Fatalf("event fanout attempts = %+v, want one accepted fanout attempt", bus.events)
	}
}

func TestServiceCreateRunIdempotentRetryDispatchesQueuedRunAfterAcceptedAppendFailure(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := &failOnceAppendEventStore{
		MemoryStore: store.NewMemoryStore(),
		matchKind:   "run.accepted",
		err:         errors.New("temporary accepted event write failure"),
	}
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Accepted event retry",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	req := CreateRunRequest{
		ThreadID:       thread.ThreadID,
		UserID:         "user-1",
		Goal:           "Run a long autonomous analysis.",
		Messages:       []domain.ThreadMessage{{Role: "user", Content: "Run a long autonomous analysis."}},
		IdempotencyKey: "accepted-append-retry",
	}

	if _, err := service.CreateRun(ctx, req); err == nil {
		t.Fatalf("first CreateRun error = nil, want accepted event append failure")
	}
	run, found, err := mem.FindRunByIdempotencyKey(ctx, thread.ThreadID, "user-1", req.IdempotencyKey)
	if err != nil {
		t.Fatalf("FindRunByIdempotencyKey: %v", err)
	}
	if !found {
		t.Fatalf("run should be persisted even though accepted event append failed")
	}
	if run.Status != domain.RunStatusQueued {
		t.Fatalf("run status = %s, want queued before retry", run.Status)
	}
	select {
	case job := <-bus.Jobs():
		t.Fatalf("unexpected job after failed accepted event append: %+v", job)
	default:
	}

	retried, err := service.CreateRun(ctx, req)
	if err != nil {
		t.Fatalf("second CreateRun should recover and dispatch existing queued run: %v", err)
	}
	if retried.RunID != run.RunID {
		t.Fatalf("retried run id = %s, want existing run %s", retried.RunID, run.RunID)
	}
	select {
	case job := <-bus.Jobs():
		if job.RunID != run.RunID {
			t.Fatalf("job run id = %s, want %s", job.RunID, run.RunID)
		}
	case <-time.After(time.Second):
		t.Fatalf("expected retry to dispatch queued run")
	}
	events, err := mem.ListRunEvents(ctx, run.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	if len(events) != 1 || events[0].EventKind != "run.accepted" {
		t.Fatalf("events = %+v, want recovered run.accepted event", events)
	}
}

func TestServiceCreateRunIdempotentRetryDispatchesQueuedRunAfterAcceptedPersistedBeforePublish(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Accepted-before-publish retry",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	req := CreateRunRequest{
		ThreadID:       thread.ThreadID,
		UserID:         "user-1",
		Goal:           "Run a long autonomous analysis.",
		Messages:       []domain.ThreadMessage{{Role: "user", Content: "Run a long autonomous analysis."}},
		IdempotencyKey: "accepted-before-publish-retry",
	}
	stored, err := mem.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: req.ThreadID,
		UserID:   req.UserID,
		Goal:     req.Goal,
		Messages: req.Messages,
		Metadata: domain.JSONMap{"idempotency_key": req.IdempotencyKey},
	})
	if err != nil {
		t.Fatalf("CreateRun seed: %v", err)
	}
	if _, err := mem.AppendRunEvent(ctx, domain.AppendRunEventInput{
		RunID:     stored.RunID,
		ThreadID:  stored.ThreadID,
		EventKind: "run.accepted",
		Message:   "Run accepted.",
		Payload:   domain.JSONMap{"status": string(stored.Status), "workflow_kind": stored.WorkflowKind},
	}); err != nil {
		t.Fatalf("AppendRunEvent seed accepted: %v", err)
	}

	retried, err := service.CreateRun(ctx, req)
	if err != nil {
		t.Fatalf("CreateRun retry should redispatch accepted queued run: %v", err)
	}
	if retried.RunID != stored.RunID {
		t.Fatalf("retried run id = %s, want existing run %s", retried.RunID, stored.RunID)
	}
	select {
	case job := <-bus.Jobs():
		if job.RunID != stored.RunID {
			t.Fatalf("redispatched job run id = %s, want %s", job.RunID, stored.RunID)
		}
	case <-time.After(time.Second):
		t.Fatalf("expected retry to redispatch queued run after accepted-before-publish crash window")
	}
	events, err := mem.ListRunEvents(ctx, stored.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	if len(events) != 1 || events[0].EventKind != "run.accepted" {
		t.Fatalf("events = %+v, want existing single run.accepted only", events)
	}
}

func TestServiceCreateRunIdempotentQueuedRetryDispatchesOnlyStoredInputContract(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{UserID: "user-1", Title: "Stored retry contract"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	const idempotencyKey = "stored-input-contract-retry"
	storedDescriptor := domain.JSONMap{
		"type":           "selected_resource",
		"binding_schema": "ultra.selected_resource.v1",
		"authority":      "control_resource_catalog",
		"resource_id":    "file-original",
		"file_id":        "file-original",
		"sha256":         strings.Repeat("a", 64),
		"size_bytes":     int64(1234),
	}
	stored, err := mem.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Analyze the originally authorized TDB.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Analyze original.csv"}},
		Metadata: domain.JSONMap{
			"idempotency_key":      idempotencyKey,
			"file_ids":             []string{"file-original"},
			"resource_uris":        []string{"catalog://file-original"},
			"knowledge_context":    domain.JSONMap{"source": "stored"},
			"resource_descriptors": []domain.JSONMap{storedDescriptor},
		},
	})
	if err != nil {
		t.Fatalf("CreateRun seed: %v", err)
	}
	if _, err := mem.AppendRunEvent(ctx, domain.AppendRunEventInput{
		RunID: stored.RunID, ThreadID: stored.ThreadID, EventKind: "run.accepted",
		Message: "Run accepted.", Payload: domain.JSONMap{"status": string(stored.Status)},
	}); err != nil {
		t.Fatalf("AppendRunEvent: %v", err)
	}

	retried, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID:         thread.ThreadID,
		UserID:           "user-1",
		Goal:             "Try to replace the stored contract.",
		Messages:         []domain.ThreadMessage{{Role: "user", Content: "Analyze foreign.csv"}},
		FileIDs:          []string{"file-foreign"},
		ResourceURIs:     []string{"catalog://file-foreign"},
		KnowledgeContext: domain.JSONMap{"source": "retry"},
		ResourceDescriptors: []domain.JSONMap{{
			"type": "selected_resource", "resource_id": "file-foreign", "file_id": "file-foreign",
			"sha256": strings.Repeat("f", 64), "size_bytes": int64(1),
		}},
		IdempotencyKey: idempotencyKey,
	})
	if err != nil {
		t.Fatalf("CreateRun retry: %v", err)
	}
	if retried.RunID != stored.RunID {
		t.Fatalf("retried run id = %s, want stored %s", retried.RunID, stored.RunID)
	}
	select {
	case job := <-bus.Jobs():
		if got := job.FileIDs; len(got) != 1 || got[0] != "file-original" {
			t.Fatalf("retry job file_ids = %#v, want stored original", got)
		}
		if got := job.ResourceURIs; len(got) != 1 || got[0] != "catalog://file-original" {
			t.Fatalf("retry job resource_uris = %#v, want stored original", got)
		}
		if len(job.ResourceDescriptors) != 1 || job.ResourceDescriptors[0]["resource_id"] != "file-original" || job.ResourceDescriptors[0]["sha256"] != strings.Repeat("a", 64) {
			t.Fatalf("retry job descriptors = %#v, want stored binding", job.ResourceDescriptors)
		}
		if len(job.Messages) != 1 || job.Messages[0].Content != "Analyze original.csv" {
			t.Fatalf("retry job messages = %#v, want stored transcript", job.Messages)
		}
		if job.KnowledgeContext["source"] != "stored" || strings.Contains(fmt.Sprint(job), "file-foreign") {
			t.Fatalf("retry request contaminated stored job: %+v", job)
		}
	case <-time.After(time.Second):
		t.Fatal("expected queued retry dispatch")
	}
}

func TestServiceCancelRunPersistsDurableCancellationWhenCancelSignalFails(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := &failingCancelBus{cancelErr: errors.New("nats unavailable")}
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Cancel transport failure",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Run until explicitly stopped.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Run until explicitly stopped."}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	bus.events = nil

	canceled, err := service.CancelRun(ctx, CancelRunRequest{RunID: run.RunID, Reason: "user stop"})
	if err != nil {
		t.Fatalf("CancelRun should persist durable cancellation despite NATS signal failure: %v", err)
	}
	if canceled.Status != domain.RunStatusCanceled {
		t.Fatalf("run status = %s, want canceled", canceled.Status)
	}
	updated, err := mem.GetRun(ctx, run.RunID)
	if err != nil {
		t.Fatalf("GetRun: %v", err)
	}
	if updated.Status != domain.RunStatusCanceled {
		t.Fatalf("stored run status = %s, want canceled", updated.Status)
	}
	events, err := mem.ListRunEvents(ctx, run.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	foundCanceled := false
	for _, event := range events {
		if event.EventKind == "run.canceled" {
			foundCanceled = true
		}
	}
	if !foundCanceled {
		t.Fatalf("events = %+v, want durable run.canceled event despite cancel signal failure", events)
	}
	if len(bus.cancellations) != 1 {
		t.Fatalf("cancel attempts = %d, want exactly one attempted signal", len(bus.cancellations))
	}
	if len(bus.events) != 1 || bus.events[0].EventKind != "run.canceled" {
		t.Fatalf("fanout events = %+v, want run.canceled fanout before best-effort cancel signal", bus.events)
	}
}

func TestServiceCancelRunPersistsDurableCancellationWhenRunEventFanoutFails(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := &failingRunEventBus{eventErr: errors.New("event fanout unavailable")}
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Cancel fanout failure",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Run until explicitly stopped.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Run until explicitly stopped."}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	bus.events = nil

	canceled, err := service.CancelRun(ctx, CancelRunRequest{RunID: run.RunID, Reason: "user stop"})
	if err != nil {
		t.Fatalf("CancelRun should not fail when run-event fanout fails after durable cancel: %v", err)
	}
	if canceled.Status != domain.RunStatusCanceled {
		t.Fatalf("run status = %s, want canceled", canceled.Status)
	}
	if len(bus.events) != 1 || bus.events[0].EventKind != "run.canceled" {
		t.Fatalf("fanout attempts = %+v, want one run.canceled fanout attempt", bus.events)
	}
	if len(bus.cancellations) != 1 || bus.cancellations[0].RunID != run.RunID {
		t.Fatalf("cancel attempts = %+v, want cancel interrupt attempted after durable event", bus.cancellations)
	}
}

func TestServiceCancelRunDoesNotMarkCanceledWhenCanceledEventAppendFails(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := &failingAppendEventStore{
		MemoryStore: store.NewMemoryStore(),
		matchKind:   "run.canceled",
		err:         errors.New("event store unavailable"),
	}
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Cancel event append failure",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Run until explicitly stopped.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Run until explicitly stopped."}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	if _, err := service.CancelRun(ctx, CancelRunRequest{RunID: run.RunID, Reason: "user stop"}); err == nil {
		t.Fatalf("CancelRun error = nil, want event append error")
	}
	updated, err := mem.GetRun(ctx, run.RunID)
	if err != nil {
		t.Fatalf("GetRun: %v", err)
	}
	if updated.Status == domain.RunStatusCanceled {
		t.Fatalf("run status = %s, want non-terminal because run.canceled was not durable", updated.Status)
	}
	events, err := mem.ListRunEvents(ctx, run.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	for _, event := range events {
		if event.EventKind == "run.canceled" {
			t.Fatalf("run.canceled event should not be persisted when append fails: %+v", events)
		}
	}
	select {
	case cancel := <-bus.Cancellations():
		t.Fatalf("cancel signal = %+v, want none because run.canceled was not durable", cancel)
	default:
	}
}

func TestServiceCancelRunUsesWorkerCompatibleCanceledEventID(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Cancel idempotency",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Run until canceled.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Run until canceled."}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	drainRunEvents(bus)

	canceled, err := service.CancelRun(ctx, CancelRunRequest{RunID: run.RunID, Reason: "user stop"})
	if err != nil {
		t.Fatalf("CancelRun: %v", err)
	}
	if canceled.Status != domain.RunStatusCanceled {
		t.Fatalf("run status = %s, want canceled", canceled.Status)
	}
	select {
	case event := <-bus.Events():
		if event.EventID != "evt_"+run.RunID+"_canceled" || event.EventKind != "run.canceled" {
			t.Fatalf("cancel fanout event = %+v, want deterministic run.canceled", event)
		}
	case <-time.After(time.Second):
		t.Fatalf("expected cancel fanout event")
	}

	if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt_" + run.RunID + "_canceled",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.canceled",
		Message:   "worker replay should not create a new terminal event",
		Payload:   domain.JSONMap{"reason": "worker observed cancel"},
	}); err != nil {
		t.Fatalf("IngestRunEvent duplicate canceled: %v", err)
	}
	events, err := mem.ListRunEvents(ctx, run.RunID, 20)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	var canceledEvents int
	for _, event := range events {
		if event.EventID == "evt_"+run.RunID+"_canceled" {
			canceledEvents++
		}
	}
	if canceledEvents != 1 {
		t.Fatalf("canceled event count = %d, want 1; events=%+v", canceledEvents, events)
	}
	select {
	case event := <-bus.Events():
		if event.EventID != "evt_"+run.RunID+"_canceled" || event.Payload["reason"] != "user stop" {
			t.Fatalf("duplicate cancel fanout = %+v, want stored control-plane event", event)
		}
	case <-time.After(time.Second):
		t.Fatalf("expected duplicate worker cancel to fan out stored event")
	}
}

func TestServiceCreateRunPersistsOnlyNewTranscriptSuffixForFollowups(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Followup thread",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}

	firstRun, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Make the first plot.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Make the first plot."}},
	})
	if err != nil {
		t.Fatalf("CreateRun first: %v", err)
	}
	drainJobs(bus)

	fullFollowupTranscript := []domain.ThreadMessage{
		{Role: "user", Content: "Make the first plot."},
		{Role: "assistant", Content: "The first plot is saved."},
		{Role: "user", Content: "Change the plot color to viridis."},
	}
	followupRun, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Change the plot color to viridis.",
		Messages: fullFollowupTranscript,
	})
	if err != nil {
		t.Fatalf("CreateRun followup: %v", err)
	}

	messages, err := mem.ListThreadMessages(ctx, thread.ThreadID)
	if err != nil {
		t.Fatalf("ListThreadMessages: %v", err)
	}
	if got, want := len(messages), 3; got != want {
		t.Fatalf("persisted messages = %d, want %d: %+v", got, want, messages)
	}
	if messages[0].Content != "Make the first plot." || messages[1].Content != "The first plot is saved." || messages[2].Content != "Change the plot color to viridis." {
		t.Fatalf("persisted messages = %+v, want non-duplicated transcript order", messages)
	}
	if messages[0].RunID != firstRun.RunID {
		t.Fatalf("first user message run id = %q, want %q", messages[0].RunID, firstRun.RunID)
	}
	if messages[1].RunID != firstRun.RunID {
		t.Fatalf("previous assistant message run id = %q, want previous run id %q", messages[1].RunID, firstRun.RunID)
	}
	if messages[1].RunID == followupRun.RunID {
		t.Fatalf("previous assistant message was mislabeled with followup run id %q", followupRun.RunID)
	}
	if messages[2].RunID != followupRun.RunID {
		t.Fatalf("followup user message run id = %q, want %q", messages[2].RunID, followupRun.RunID)
	}
	select {
	case job := <-bus.Jobs():
		if len(job.Messages) != len(fullFollowupTranscript) {
			t.Fatalf("job messages = %d, want full transcript %d", len(job.Messages), len(fullFollowupTranscript))
		}
	case <-time.After(time.Second):
		t.Fatalf("expected followup job")
	}
}

func TestServiceCreateRunDoesNotDuplicatePriorAssistantWhenDisplayTextDrifts(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Followup display drift",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}

	firstRun, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Create a plot.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Create a plot."}},
	})
	if err != nil {
		t.Fatalf("CreateRun first: %v", err)
	}
	drainJobs(bus)
	if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt-" + firstRun.RunID + "-completed",
		RunID:     firstRun.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.completed",
		Payload:   domain.JSONMap{"response_text": "Stored answer with /outputs/plot.png."},
	}); err != nil {
		t.Fatalf("IngestRunEvent completed: %v", err)
	}

	fullFollowupTranscript := []domain.ThreadMessage{
		{Role: "user", Content: "Create a plot."},
		{Role: "assistant", Content: "Stored answer with [plot](/v2/artifacts/artifact_1/download)."},
		{Role: "user", Content: "Now explain the axes."},
	}
	followupRun, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Now explain the axes.",
		Messages: fullFollowupTranscript,
	})
	if err != nil {
		t.Fatalf("CreateRun followup: %v", err)
	}

	messages, err := mem.ListThreadMessages(ctx, thread.ThreadID)
	if err != nil {
		t.Fatalf("ListThreadMessages: %v", err)
	}
	if got, want := len(messages), 3; got != want {
		t.Fatalf("persisted messages = %d, want existing user+assistant plus followup user only: %+v", got, messages)
	}
	if messages[1].Content != "Stored answer with /outputs/plot.png." || messages[1].RunID != firstRun.RunID {
		t.Fatalf("previous assistant message = %+v, want original durable assistant owned by first run", messages[1])
	}
	if messages[2].Role != "user" || messages[2].Content != "Now explain the axes." || messages[2].RunID != followupRun.RunID {
		t.Fatalf("followup user message = %+v, want new user turn owned by followup run %s", messages[2], followupRun.RunID)
	}
	select {
	case job := <-bus.Jobs():
		if len(job.Messages) != len(fullFollowupTranscript) {
			t.Fatalf("job messages = %d, want full incoming transcript %d", len(job.Messages), len(fullFollowupTranscript))
		}
	case <-time.After(time.Second):
		t.Fatalf("expected followup job")
	}
}

func TestServiceCreateRunKeepsInternalToolRunsOutOfVisibleThreadState(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "RareSpot followup thread",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}

	firstRun, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Run prairie dog detection on this image.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Run prairie dog detection on this image."}},
	})
	if err != nil {
		t.Fatalf("CreateRun first: %v", err)
	}
	drainJobs(bus)

	internalRun, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID:          thread.ThreadID,
		UserID:            "user-1",
		Goal:              "Run RareSpot ecology inference.",
		Messages:          []domain.ThreadMessage{{Role: "tool", Content: "RareSpot ecology inference requested."}},
		SelectedToolNames: []string{"rarespot_ecology_inference"},
		WorkflowHint:      domain.JSONMap{"id": "rarespot_ecology"},
		Metadata:          domain.JSONMap{"parent_run_id": firstRun.RunID, "tool_name": "rarespot_ecology_inference"},
	})
	if err != nil {
		t.Fatalf("CreateRun internal tool run: %v", err)
	}
	if internalRun.RunID == firstRun.RunID {
		t.Fatalf("internal run reused first run id %q", firstRun.RunID)
	}
	drainJobs(bus)

	if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt-internal-rarespot-completed",
		RunID:     internalRun.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.completed",
		Payload:   domain.JSONMap{"response_text": "Internal RareSpot run completed."},
	}); err != nil {
		t.Fatalf("IngestRunEvent internal completed: %v", err)
	}

	afterInternal, err := mem.GetThread(ctx, thread.ThreadID)
	if err != nil {
		t.Fatalf("GetThread after internal run: %v", err)
	}
	if afterInternal.LatestRunID != firstRun.RunID {
		t.Fatalf("thread latest run id = %q, want user-facing run %q", afterInternal.LatestRunID, firstRun.RunID)
	}
	messagesAfterInternal, err := mem.ListThreadMessages(ctx, thread.ThreadID)
	if err != nil {
		t.Fatalf("ListThreadMessages after internal run: %v", err)
	}
	if got, want := len(messagesAfterInternal), 1; got != want {
		t.Fatalf("messages after internal run = %d, want %d: %+v", got, want, messagesAfterInternal)
	}
	if messagesAfterInternal[0].Role != "user" {
		t.Fatalf("first visible message role = %q, want user", messagesAfterInternal[0].Role)
	}

	followupTranscript := []domain.ThreadMessage{
		{Role: "user", Content: "Run prairie dog detection on this image."},
		{Role: "assistant", Content: "RareSpot detection completed with downloadable outputs."},
		{Role: "user", Content: "Now summarize the GPS metadata and compare detections."},
	}
	followupRun, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Now summarize the GPS metadata and compare detections.",
		Messages: followupTranscript,
	})
	if err != nil {
		t.Fatalf("CreateRun followup: %v", err)
	}

	messages, err := mem.ListThreadMessages(ctx, thread.ThreadID)
	if err != nil {
		t.Fatalf("ListThreadMessages followup: %v", err)
	}
	if got, want := len(messages), 3; got != want {
		t.Fatalf("persisted messages = %d, want %d: %+v", got, want, messages)
	}
	if messages[1].RunID != firstRun.RunID {
		t.Fatalf("previous assistant run id = %q, want first user-facing run %q", messages[1].RunID, firstRun.RunID)
	}
	if messages[2].RunID != followupRun.RunID {
		t.Fatalf("followup user run id = %q, want followup run %q", messages[2].RunID, followupRun.RunID)
	}
}

func TestServiceCreateRunIncludesPriorArtifactsInFollowupJob(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Artifact followup",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}

	firstRun, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Create the first plot.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Create the first plot."}},
	})
	if err != nil {
		t.Fatalf("CreateRun first: %v", err)
	}
	drainJobs(bus)

	_, err = service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt-artifact-1",
		RunID:     firstRun.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "artifact.created",
		EventType: "artifact",
		Message:   "Created plot.",
		Payload: domain.JSONMap{
			"artifact_id": "artifact-plot",
			"output_id":   "output-plot",
			"kind":        "figure",
			"title":       "Squared Plot",
			"path":        "outputs/plot_squared.png",
			"source_path": "/tmp/artifacts/" + firstRun.RunID + "/outputs/plot_squared.png",
			"mime_type":   "image/png",
			"size_bytes":  123,
			"sha256":      "abc123",
			"tool_name":   "outputs_collector",
		},
	})
	if err != nil {
		t.Fatalf("IngestRunEvent artifact: %v", err)
	}

	foreignThread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-2",
		Title:  "Foreign artifact source",
	})
	if err != nil {
		t.Fatalf("CreateThread foreign: %v", err)
	}
	foreignRun, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: foreignThread.ThreadID,
		UserID:   "user-2",
		Goal:     "Create a private foreign table.",
	})
	if err != nil {
		t.Fatalf("CreateRun foreign: %v", err)
	}
	drainJobs(bus)
	if _, err := mem.CreateArtifact(ctx, domain.CreateArtifactInput{
		ArtifactID: "artifact-foreign",
		RunID:      foreignRun.RunID,
		ThreadID:   foreignThread.ThreadID,
		Kind:       "table",
		Path:       "outputs/private.csv",
		SourcePath: "/tmp/artifacts/" + foreignRun.RunID + "/outputs/private.csv",
	}); err != nil {
		t.Fatalf("CreateArtifact foreign: %v", err)
	}
	// A caller-controlled transcript reference must not turn another owner's run
	// into a prior-artifact capability for this thread.
	if _, err := mem.AppendThreadMessage(ctx, domain.ThreadMessage{
		ThreadID: thread.ThreadID,
		Role:     "assistant",
		Content:  "Guessed a foreign run id.",
		RunID:    foreignRun.RunID,
	}); err != nil {
		t.Fatalf("AppendThreadMessage foreign run reference: %v", err)
	}

	_, err = service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Add a reference line.",
		Messages: []domain.ThreadMessage{
			{Role: "user", Content: "Create the first plot."},
			{Role: "assistant", Content: "Created the first plot."},
			{Role: "user", Content: "Add a reference line."},
		},
	})
	if err != nil {
		t.Fatalf("CreateRun followup: %v", err)
	}

	select {
	case job := <-bus.Jobs():
		if got, want := len(job.ResourceDescriptors), 1; got != want {
			t.Fatalf("job resource descriptors = %d, want %d: %+v", got, want, job.ResourceDescriptors)
		}
		descriptor := job.ResourceDescriptors[0]
		if descriptor["artifact_id"] != "artifact-plot" {
			t.Fatalf("descriptor artifact id = %#v, want artifact-plot", descriptor["artifact_id"])
		}
		if descriptor["run_id"] != firstRun.RunID {
			t.Fatalf("descriptor run id = %#v, want %s", descriptor["run_id"], firstRun.RunID)
		}
		if descriptor["path"] != "outputs/plot_squared.png" {
			t.Fatalf("descriptor path = %#v, want outputs/plot_squared.png", descriptor["path"])
		}
	case <-time.After(time.Second):
		t.Fatalf("expected followup job")
	}
}

func TestServiceCreateRunDropsCallerArtifactsButKeepsServerSelectedResources(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{
		UserID: "user-1",
		Title:  "Descriptor trust boundary",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	selected := domain.JSONMap{
		"type":           "selected_resource",
		"binding_schema": "ultra.selected_resource.v1",
		"authority":      "control_resource_catalog",
		"resource_id":    "file-owned",
		"file_id":        "file-owned",
		"sha256":         strings.Repeat("a", 64),
		"size_bytes":     int64(123),
		"metadata": domain.JSONMap{
			"caption": "owned-db",
		},
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Analyze the selected database.",
		ResourceDescriptors: []domain.JSONMap{
			selected,
			{
				"type": "artifact", "artifact_id": "artifact-foreign",
				"run_id": "run-foreign", "path": "outputs/private.csv",
			},
			{
				"artifact_id": "artifact-foreign-untyped",
				"run_id":      "run-foreign",
				"path":        "outputs/private.csv",
			},
		},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	select {
	case job := <-bus.Jobs():
		if len(job.ResourceDescriptors) != 1 {
			t.Fatalf("job descriptors = %#v, want only selected resource", job.ResourceDescriptors)
		}
		if job.ResourceDescriptors[0]["resource_id"] != "file-owned" ||
			job.ResourceDescriptors[0]["authority"] != "control_resource_catalog" {
			t.Fatalf("selected resource binding changed: %#v", job.ResourceDescriptors[0])
		}
		metadata, ok := job.ResourceDescriptors[0]["metadata"].(domain.JSONMap)
		if !ok {
			t.Fatalf("selected resource metadata type = %T", job.ResourceDescriptors[0]["metadata"])
		}
		if metadata["caption"] != "owned-db" {
			t.Fatalf("selected resource metadata changed: %#v", metadata)
		}
	default:
		t.Fatal("CreateRun did not dispatch a job")
	}
	stored, err := mem.GetRun(ctx, run.RunID)
	if err != nil {
		t.Fatalf("GetRun: %v", err)
	}
	descriptors, ok := stored.Metadata["resource_descriptors"].([]domain.JSONMap)
	if !ok || len(descriptors) != 1 || descriptors[0]["resource_id"] != "file-owned" {
		t.Fatalf("stored descriptors = %T %#v, want only selected resource", stored.Metadata["resource_descriptors"], stored.Metadata["resource_descriptors"])
	}
	if strings.Contains(fmt.Sprint(stored.Metadata), "artifact-foreign") ||
		strings.Contains(fmt.Sprint(stored.Metadata), "run-foreign") {
		t.Fatalf("caller artifact capability persisted: %#v", stored.Metadata)
	}
}

type racingIdempotencyStore struct {
	*store.MemoryStore
	mu      sync.Mutex
	calls   int
	release chan struct{}
	once    sync.Once
}

func newRacingIdempotencyStore() *racingIdempotencyStore {
	return &racingIdempotencyStore{
		MemoryStore: store.NewMemoryStore(),
		release:     make(chan struct{}),
	}
}

func (s *racingIdempotencyStore) FindRunByIdempotencyKey(ctx context.Context, threadID string, userID string, idempotencyKey string) (domain.RunRecord, bool, error) {
	run, found, err := s.MemoryStore.FindRunByIdempotencyKey(ctx, threadID, userID, idempotencyKey)
	s.mu.Lock()
	s.calls++
	if s.calls == 2 {
		s.once.Do(func() { close(s.release) })
	}
	s.mu.Unlock()
	select {
	case <-s.release:
	case <-time.After(50 * time.Millisecond):
		s.once.Do(func() { close(s.release) })
	}
	return run, found, err
}

type conflictingIdempotencyStore struct {
	*store.MemoryStore
	mu        sync.Mutex
	findCalls int
}

func (s *conflictingIdempotencyStore) FindRunByIdempotencyKey(ctx context.Context, threadID string, userID string, idempotencyKey string) (domain.RunRecord, bool, error) {
	s.mu.Lock()
	s.findCalls++
	call := s.findCalls
	s.mu.Unlock()
	if call == 1 {
		return domain.RunRecord{}, false, nil
	}
	return s.MemoryStore.FindRunByIdempotencyKey(ctx, threadID, userID, idempotencyKey)
}

func (s *conflictingIdempotencyStore) CreateRun(ctx context.Context, input domain.CreateRunInput) (domain.RunRecord, error) {
	_ = ctx
	_ = input
	return domain.RunRecord{}, store.ErrConflict
}

type blockingAppendThreadMessageStore struct {
	*store.MemoryStore
	once    sync.Once
	started chan struct{}
	release chan struct{}
}

func (s *blockingAppendThreadMessageStore) AppendThreadMessage(ctx context.Context, message domain.ThreadMessage) (domain.ThreadMessage, error) {
	if strings.EqualFold(strings.TrimSpace(message.Role), "assistant") {
		s.once.Do(func() { close(s.started) })
		select {
		case <-s.release:
		case <-ctx.Done():
			return domain.ThreadMessage{}, ctx.Err()
		}
	}
	return s.MemoryStore.AppendThreadMessage(ctx, message)
}

func (s *blockingAppendThreadMessageStore) CompleteRun(ctx context.Context, input domain.CompleteRunInput) (domain.RunRecord, error) {
	if strings.TrimSpace(input.ResponseText) != "" {
		s.once.Do(func() { close(s.started) })
		select {
		case <-s.release:
		case <-ctx.Done():
			return domain.RunRecord{}, ctx.Err()
		}
	}
	return s.MemoryStore.CompleteRun(ctx, input)
}

type failOnceStatusStore struct {
	*store.MemoryStore
	mu         sync.Mutex
	failStatus domain.RunStatus
	err        error
	failed     bool
}

func (s *failOnceStatusStore) UpdateRunStatus(ctx context.Context, runID string, status domain.RunStatus, responseText string, errorText string) (domain.RunRecord, error) {
	s.mu.Lock()
	shouldFail := !s.failed && status == s.failStatus
	if shouldFail {
		s.failed = true
	}
	s.mu.Unlock()
	if shouldFail {
		return domain.RunRecord{}, s.err
	}
	return s.MemoryStore.UpdateRunStatus(ctx, runID, status, responseText, errorText)
}

func (s *failOnceStatusStore) CompleteRun(ctx context.Context, input domain.CompleteRunInput) (domain.RunRecord, error) {
	s.mu.Lock()
	shouldFail := !s.failed && s.failStatus == domain.RunStatusSucceeded
	if shouldFail {
		s.failed = true
	}
	s.mu.Unlock()
	if shouldFail {
		return domain.RunRecord{}, s.err
	}
	return s.MemoryStore.CompleteRun(ctx, input)
}

type failingAppendEventStore struct {
	*store.MemoryStore
	matchKind string
	err       error
}

func (s *failingAppendEventStore) AppendRunEvent(ctx context.Context, input domain.AppendRunEventInput) (domain.RunEventRecord, error) {
	if input.EventKind == s.matchKind {
		return domain.RunEventRecord{}, s.err
	}
	return s.MemoryStore.AppendRunEvent(ctx, input)
}

type failOnceAppendEventStore struct {
	*store.MemoryStore
	mu        sync.Mutex
	matchKind string
	err       error
	failed    bool
}

func (s *failOnceAppendEventStore) AppendRunEvent(ctx context.Context, input domain.AppendRunEventInput) (domain.RunEventRecord, error) {
	s.mu.Lock()
	shouldFail := !s.failed && input.EventKind == s.matchKind
	if shouldFail {
		s.failed = true
	}
	s.mu.Unlock()
	if shouldFail {
		return domain.RunEventRecord{}, s.err
	}
	return s.MemoryStore.AppendRunEvent(ctx, input)
}

type duplicateEventRaceStore struct {
	*store.MemoryStore
	targetID string
	ready    chan struct{}
	release  chan struct{}
	once     sync.Once
	calls    int
	mu       sync.Mutex
}

// AppendRunEventIfRunActive shadows the promoted fast-path method with an
// incompatible signature so this fixture does not satisfy
// activeRunEventAppender: the test exercises the legacy read-then-append
// path, whose check-then-act race is what the event-ID locks guard against.
func (s *duplicateEventRaceStore) AppendRunEventIfRunActive() {}

func (s *duplicateEventRaceStore) GetRunEvent(ctx context.Context, eventID string) (domain.RunEventRecord, bool, error) {
	if eventID != s.targetID {
		return s.MemoryStore.GetRunEvent(ctx, eventID)
	}
	s.mu.Lock()
	s.calls++
	call := s.calls
	if call == 2 {
		s.once.Do(func() { close(s.ready) })
	}
	s.mu.Unlock()
	if call <= 2 {
		<-s.release
		return domain.RunEventRecord{}, false, nil
	}
	return s.MemoryStore.GetRunEvent(ctx, eventID)
}

type failOnceRunEventBus struct {
	*eventbus.MemoryBus
	mu        sync.Mutex
	matchKind string
	err       error
	failed    bool
}

func (b *failOnceRunEventBus) PublishRunEvent(ctx context.Context, event domain.RunEventRecord) error {
	b.mu.Lock()
	shouldFail := !b.failed && event.EventKind == b.matchKind
	if shouldFail {
		b.failed = true
	}
	b.mu.Unlock()
	if shouldFail {
		return b.err
	}
	return b.MemoryBus.PublishRunEvent(ctx, event)
}

func sameStringSlice(value any, want []string) bool {
	switch typed := value.(type) {
	case []string:
		if len(typed) != len(want) {
			return false
		}
		for index := range typed {
			if typed[index] != want[index] {
				return false
			}
		}
		return true
	case []any:
		if len(typed) != len(want) {
			return false
		}
		for index := range typed {
			if typed[index] != want[index] {
				return false
			}
		}
		return true
	default:
		return false
	}
}

func drainRunEvents(bus *eventbus.MemoryBus) {
	for {
		select {
		case <-bus.Events():
		default:
			return
		}
	}
}

func drainJobs(bus *eventbus.MemoryBus) {
	for {
		select {
		case <-bus.Jobs():
		default:
			return
		}
	}
}

func twoDigit(value int) string {
	if value < 10 {
		return "0" + string(rune('0'+value))
	}
	return string(rune('0'+(value/10)%10)) + string(rune('0'+value%10))
}

type failingCancelBus struct {
	cancelErr     error
	jobs          []eventbus.Job
	cancellations []eventbus.CancelSignal
	events        []domain.RunEventRecord
}

func (b *failingCancelBus) PublishJob(ctx context.Context, job eventbus.Job) error {
	_ = ctx
	b.jobs = append(b.jobs, job)
	return nil
}

func (b *failingCancelBus) PublishCancel(ctx context.Context, cancel eventbus.CancelSignal) error {
	_ = ctx
	b.cancellations = append(b.cancellations, cancel)
	return b.cancelErr
}

func (b *failingCancelBus) PublishRunEvent(ctx context.Context, event domain.RunEventRecord) error {
	_ = ctx
	b.events = append(b.events, event)
	return nil
}

type failingJobBus struct {
	jobErr        error
	jobs          []eventbus.Job
	cancellations []eventbus.CancelSignal
	events        []domain.RunEventRecord
}

func (b *failingJobBus) PublishJob(ctx context.Context, job eventbus.Job) error {
	_ = ctx
	b.jobs = append(b.jobs, job)
	return b.jobErr
}

func (b *failingJobBus) PublishCancel(ctx context.Context, cancel eventbus.CancelSignal) error {
	_ = ctx
	b.cancellations = append(b.cancellations, cancel)
	return nil
}

func (b *failingJobBus) PublishRunEvent(ctx context.Context, event domain.RunEventRecord) error {
	_ = ctx
	b.events = append(b.events, event)
	return nil
}

type failingRunEventBus struct {
	eventErr      error
	jobs          []eventbus.Job
	cancellations []eventbus.CancelSignal
	events        []domain.RunEventRecord
}

func (b *failingRunEventBus) PublishJob(ctx context.Context, job eventbus.Job) error {
	_ = ctx
	b.jobs = append(b.jobs, job)
	return nil
}

func (b *failingRunEventBus) PublishCancel(ctx context.Context, cancel eventbus.CancelSignal) error {
	_ = ctx
	b.cancellations = append(b.cancellations, cancel)
	return nil
}

func (b *failingRunEventBus) PublishRunEvent(ctx context.Context, event domain.RunEventRecord) error {
	_ = ctx
	b.events = append(b.events, event)
	return b.eventErr
}

func TestServiceIngestRunEventFastPathDeduplicatesConcurrentDuplicates(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{UserID: "user-1", Title: "fast path duplicates"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "fast path duplicates",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	input := domain.AppendRunEventInput{
		EventID:   "evt-fast-path-duplicate",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "message.delta",
		Message:   "duplicate",
	}
	const workers = 8
	var wg sync.WaitGroup
	results := make(chan domain.RunEventRecord, workers)
	errs := make(chan error, workers)
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			event, err := service.IngestRunEvent(ctx, input)
			results <- event
			errs <- err
		}()
	}
	wg.Wait()
	close(results)
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("IngestRunEvent concurrent duplicate: %v", err)
		}
	}
	var sequence int64
	for event := range results {
		if event.EventID != input.EventID {
			t.Fatalf("event id = %q, want %q", event.EventID, input.EventID)
		}
		if sequence == 0 {
			sequence = event.Sequence
		} else if event.Sequence != sequence {
			t.Fatalf("event sequence = %d, want every caller to observe %d", event.Sequence, sequence)
		}
	}
	events, err := mem.ListRunEvents(ctx, run.RunID, 50)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	stored := 0
	for _, event := range events {
		if event.EventID == input.EventID {
			stored++
		}
	}
	if stored != 1 {
		t.Fatalf("stored duplicate event count = %d, want exactly 1", stored)
	}
}

type statusWriteCountingStore struct {
	*store.MemoryStore
	mu                sync.Mutex
	statusWriteCounts map[string]int
}

func (s *statusWriteCountingStore) UpdateRunStatus(ctx context.Context, runID string, status domain.RunStatus, node string, errorText string) (domain.RunRecord, error) {
	s.mu.Lock()
	if s.statusWriteCounts == nil {
		s.statusWriteCounts = map[string]int{}
	}
	s.statusWriteCounts[runID]++
	s.mu.Unlock()
	return s.MemoryStore.UpdateRunStatus(ctx, runID, status, node, errorText)
}

func TestServiceIngestHeartbeatsThrottleStatusWrites(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	counting := &statusWriteCountingStore{MemoryStore: store.NewMemoryStore()}
	bus := eventbus.NewMemoryBus()
	service := NewService(counting, bus)

	thread, err := service.CreateThread(ctx, CreateThreadRequest{UserID: "user-1", Title: "heartbeat throttle"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "heartbeat throttle",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	const heartbeats = 5
	for i := 0; i < heartbeats; i++ {
		if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
			EventID:   fmt.Sprintf("evt-heartbeat-throttle-%d", i),
			RunID:     run.RunID,
			ThreadID:  thread.ThreadID,
			EventKind: "run.heartbeat",
			Message:   "heartbeat",
		}); err != nil {
			t.Fatalf("IngestRunEvent heartbeat %d: %v", i, err)
		}
	}

	// The first heartbeat must still transition the queued run to running.
	updated, err := counting.GetRun(ctx, run.RunID)
	if err != nil {
		t.Fatalf("GetRun: %v", err)
	}
	if updated.Status != domain.RunStatusRunning {
		t.Fatalf("run status = %s, want running after first heartbeat", updated.Status)
	}
	counting.mu.Lock()
	writes := counting.statusWriteCounts[run.RunID]
	counting.mu.Unlock()
	if writes != 1 {
		t.Fatalf("status writes = %d, want 1 of %d heartbeats inside the throttle window", writes, heartbeats)
	}

	events, err := counting.ListRunEvents(ctx, run.RunID, 50)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	stored := 0
	for _, event := range events {
		if event.EventKind == "run.heartbeat" {
			stored++
		}
	}
	if stored != heartbeats {
		t.Fatalf("stored heartbeats = %d, want all %d events persisted despite throttled status writes", stored, heartbeats)
	}
}

// Fix #3: a successor event whose predecessor is missing must NOT stall
// (ErrRunEventPredecessorPending) when the run is already terminal — the gate
// short-circuits so the normal drop path acks the late event.
func TestServiceIngestTerminalRunMissingPredecessorDropsInsteadOfPending(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	service := NewService(mem, eventbus.NewMemoryBus())
	thread, err := service.CreateThread(ctx, CreateThreadRequest{UserID: "u", Title: "t"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{ThreadID: thread.ThreadID, UserID: "u", Goal: "g"})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	if _, err := mem.UpdateRunStatus(ctx, run.RunID, domain.RunStatusCanceled, "", "done"); err != nil {
		t.Fatalf("UpdateRunStatus: %v", err)
	}
	// SourceSequence=5, predecessor (4) never stored, run terminal.
	_, err = service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID:        "evt-late-5",
		SourceSequence: 5,
		RunID:          run.RunID,
		ThreadID:       thread.ThreadID,
		EventKind:      "message.delta",
		Message:        "late",
	})
	if err != nil {
		t.Fatalf("IngestRunEvent returned error (should drop cleanly): %v", err)
	}
	if errors.Is(err, ErrRunEventPredecessorPending) {
		t.Fatal("terminal run successor must not be predecessor-pending")
	}
}

// Fix #3 (negative): an ACTIVE run with a genuinely missing predecessor still
// pends, so legitimate reordering is retried rather than dropped.
func TestServiceIngestActiveRunMissingPredecessorStillPends(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	service := NewService(mem, eventbus.NewMemoryBus())
	thread, _ := service.CreateThread(ctx, CreateThreadRequest{UserID: "u", Title: "t"})
	run, _ := service.CreateRun(ctx, CreateRunRequest{ThreadID: thread.ThreadID, UserID: "u", Goal: "g"})
	_, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID:        "evt-5",
		SourceSequence: 5,
		RunID:          run.RunID,
		ThreadID:       thread.ThreadID,
		EventKind:      "message.delta",
	})
	if !errors.Is(err, ErrRunEventPredecessorPending) {
		t.Fatalf("active run missing predecessor: err=%v, want ErrRunEventPredecessorPending", err)
	}
}

// Fix #2a: a source_sequence collision (two event_ids, same run_id+source_sequence)
// must resolve to a clean drop (ack), never a hard error that InProgress-loops.
// Uses live Postgres, which enforces the UNIQUE(run_id, source_sequence) index.
func TestServiceIngestSourceSequenceCollisionDropsNotStalls(t *testing.T) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pool: %v", err)
	}
	defer pool.Close()
	if err := store.ApplyPostgresSchema(ctx, pool); err != nil {
		t.Fatalf("schema: %v", err)
	}
	pg := store.NewPostgresStore(pool)
	service := NewService(pg, eventbus.NewMemoryBus())
	uid := fmt.Sprintf("collide-%d", time.Now().UnixNano())
	thread, err := pg.CreateThread(ctx, domain.CreateThreadInput{UserID: uid, Title: "t"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := pg.CreateRun(ctx, domain.CreateRunInput{ThreadID: thread.ThreadID, UserID: uid, Goal: "g", Messages: []domain.ThreadMessage{{Role: "user", Content: "x"}}})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	evtA := "evt-A-" + uid
	evtB := "evt-B-" + uid

	// First event claims (run, source_sequence=1).
	if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID: evtA, SourceSequence: 1, RunID: run.RunID, ThreadID: thread.ThreadID, EventKind: "message.delta", Message: "A",
	}); err != nil {
		t.Fatalf("ingest A: %v", err)
	}
	// Colliding event: different event_id, SAME source_sequence.
	if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID: evtB, SourceSequence: 1, RunID: run.RunID, ThreadID: thread.ThreadID, EventKind: "message.delta", Message: "B",
	}); err != nil {
		t.Fatalf("collision ingest must drop cleanly, got error: %v", err)
	}
	// Only event A is stored at source_sequence 1; B was absorbed.
	events, err := pg.ListRunEvents(ctx, run.RunID, 50)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	count := 0
	for _, e := range events {
		if e.SourceSequence == 1 {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("events at source_sequence 1 = %d, want 1 (collision absorbed)", count)
	}
}

// A4(2): an expired lease alone must NOT requeue a run whose worker is still
// emitting progress events (e.g. the control plane was briefly unreachable
// so renewals failed, but computation continued).
func TestServiceRecoverExpiredLeaseVetoedByFreshWorkerProgress(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)
	base := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)

	thread, _ := service.CreateThread(ctx, CreateThreadRequest{UserID: "u", Title: "t"})
	run, err := service.CreateRun(ctx, CreateRunRequest{ThreadID: thread.ThreadID, UserID: "u", Goal: "g"})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	drainJobs(bus)
	drainRunEvents(bus)
	if _, err := mem.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID: run.RunID, WorkerID: "w1", TTL: time.Minute, Now: base,
	}); err != nil {
		t.Fatalf("AcquireRunLease: %v", err)
	}
	now := base.Add(5 * time.Minute) // lease long expired
	// Fresh worker progress 30s ago.
	if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID: "evt-progress-1", RunID: run.RunID, ThreadID: thread.ThreadID,
		EventKind: "message.delta", TS: now.Add(-30 * time.Second), Message: "delta",
	}); err != nil {
		t.Fatalf("IngestRunEvent: %v", err)
	}
	result, err := service.RecoverExpiredRunLeases(ctx, RecoverExpiredRunLeasesRequest{
		Now: now, Limit: 100, WorkerHeartbeatStaleAfter: 2 * time.Minute,
	})
	if err != nil {
		t.Fatalf("RecoverExpiredRunLeases: %v", err)
	}
	if len(result.RequeuedRuns) != 0 {
		t.Fatalf("requeued = %+v, want none while progress is fresh", result.RequeuedRuns)
	}
	// Later pass: progress is now stale -> the expired lease is reclaimed.
	later := now.Add(10 * time.Minute)
	result, err = service.RecoverExpiredRunLeases(ctx, RecoverExpiredRunLeasesRequest{
		Now: later, Limit: 100, WorkerHeartbeatStaleAfter: 2 * time.Minute,
	})
	if err != nil {
		t.Fatalf("RecoverExpiredRunLeases(later): %v", err)
	}
	if len(result.RequeuedRuns) != 1 {
		t.Fatalf("requeued = %+v, want the dead run reclaimed once progress went stale", result.RequeuedRuns)
	}
}

// A4(3): a worker that outlived a control-plane outage revives its EXPIRED
// lease by token; once recovery cleared the lease, renewal conflicts.
func TestServiceRenewRunLeaseRevivesExpiredLeaseByToken(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	service := NewService(mem, eventbus.NewMemoryBus())
	base := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)

	thread, _ := service.CreateThread(ctx, CreateThreadRequest{UserID: "u", Title: "t"})
	run, _ := service.CreateRun(ctx, CreateRunRequest{ThreadID: thread.ThreadID, UserID: "u", Goal: "g"})
	lease, err := mem.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID: run.RunID, WorkerID: "w1", TTL: time.Minute, Now: base,
	})
	if err != nil {
		t.Fatalf("AcquireRunLease: %v", err)
	}
	expiredNow := base.Add(10 * time.Minute)
	renewed, err := service.RenewRunLease(ctx, RenewRunLeaseRequest{
		RunID: run.RunID, LeaseToken: lease.LeaseToken, TTL: 10 * time.Minute, Now: expiredNow,
	})
	if err != nil {
		t.Fatalf("RenewRunLease of expired lease with matching token must succeed: %v", err)
	}
	if !renewed.LeaseExpiresAt.After(expiredNow) {
		t.Fatalf("revived lease expires %v, want after %v", renewed.LeaseExpiresAt, expiredNow)
	}
	// Cleared lease (requeue path) -> renewal must conflict.
	if _, _, err := mem.ClearRunLease(ctx, run.RunID); err != nil {
		t.Fatalf("ClearRunLease: %v", err)
	}
	_, err = service.RenewRunLease(ctx, RenewRunLeaseRequest{
		RunID: run.RunID, LeaseToken: lease.LeaseToken, TTL: time.Minute, Now: expiredNow,
	})
	if !errors.Is(err, store.ErrConflict) {
		t.Fatalf("renewal after ClearRunLease = %v, want ErrConflict (authoritative stop)", err)
	}
}

// A6: a running run with NO lease, stale dispatch, and no worker progress is
// a zombie and must be reclaimed; fresh progress vetoes.
func TestServiceRecoverReclaimsLeaselessZombieRun(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)

	thread, _ := service.CreateThread(ctx, CreateThreadRequest{UserID: "u", Title: "t"})
	run, err := service.CreateRun(ctx, CreateRunRequest{ThreadID: thread.ThreadID, UserID: "u", Goal: "g"})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	drainJobs(bus)
	drainRunEvents(bus)
	if _, err := mem.UpdateRunStatus(ctx, run.RunID, domain.RunStatusRunning, "", ""); err != nil {
		t.Fatalf("UpdateRunStatus: %v", err)
	}
	// No lease exists. Dispatch happened "now" (real clock at CreateRun).
	// Within the grace window: not reclaimed.
	soon := domain.Now().Add(5 * time.Minute)
	result, err := service.RecoverExpiredRunLeases(ctx, RecoverExpiredRunLeasesRequest{
		Now: soon, Limit: 100, WorkerHeartbeatStaleAfter: 2 * time.Minute,
	})
	if err != nil {
		t.Fatalf("RecoverExpiredRunLeases: %v", err)
	}
	if len(result.RequeuedRuns) != 0 {
		t.Fatalf("requeued = %+v, want none inside the zombie grace window", result.RequeuedRuns)
	}
	// Past the grace with no progress: reclaimed.
	later := domain.Now().Add(20 * time.Minute)
	result, err = service.RecoverExpiredRunLeases(ctx, RecoverExpiredRunLeasesRequest{
		Now: later, Limit: 100, WorkerHeartbeatStaleAfter: 2 * time.Minute,
	})
	if err != nil {
		t.Fatalf("RecoverExpiredRunLeases(later): %v", err)
	}
	if len(result.RequeuedRuns) != 1 {
		t.Fatalf("requeued = %+v, want the zombie reclaimed after grace", result.RequeuedRuns)
	}
}

// Cancel and requeue append control-plane events to a LIVE run whose worker is
// already past sequencer seeding. They must never claim a worker
// source_sequence slot: a defaulted claim either collides with an existing
// worker stamp (the append itself 409s, so the cancel/requeue FAILS) or steals
// the worker's next stamp (ingest silently drops that worker event).
func TestCancelAndRequeueEventsClaimNoWorkerSourceSequence(t *testing.T) {
	t.Parallel()
	ctx, mem, service, run := newSteeringFixture(t)

	if _, err := service.RequeueRun(ctx, RequeueRunRequest{RunID: run.RunID, Reason: "expired lease"}); err != nil {
		t.Fatalf("RequeueRun: %v", err)
	}
	if _, err := service.CancelRun(ctx, CancelRunRequest{RunID: run.RunID, Reason: "user stop"}); err != nil {
		t.Fatalf("CancelRun: %v", err)
	}

	events, err := mem.ListRunEvents(ctx, run.RunID, 100)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	sawCanceled, sawRequeued := false, false
	for _, event := range events {
		switch event.EventKind {
		case "run.canceled":
			sawCanceled = true
			if event.SourceSequence != 0 {
				t.Fatalf("run.canceled claimed source_sequence %d — worker slot theft", event.SourceSequence)
			}
		case "run.requeued":
			sawRequeued = true
			if event.SourceSequence != 0 {
				t.Fatalf("run.requeued claimed source_sequence %d — worker slot theft", event.SourceSequence)
			}
		}
	}
	if !sawCanceled || !sawRequeued {
		t.Fatalf("events missing run.canceled (%v) or run.requeued (%v)", sawCanceled, sawRequeued)
	}
}
