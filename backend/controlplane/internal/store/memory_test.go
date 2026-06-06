package store

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

func TestMemoryStoreThreadRunEventArtifactFlow(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	thread, err := store.CreateThread(ctx, domain.CreateThreadInput{
		UserID: "user-1",
		Title:  "Microscopy analysis",
		InitialMessages: []domain.ThreadMessage{{
			Role:    "user",
			Content: "Segment these images.",
		}},
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	if thread.ThreadID == "" || thread.Status != domain.ThreadStatusActive {
		t.Fatalf("unexpected thread: %+v", thread)
	}

	run, err := store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Segment these images.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Segment these images."}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	if run.Status != domain.RunStatusQueued {
		t.Fatalf("run status = %s, want queued", run.Status)
	}

	event, err := store.AppendRunEvent(ctx, domain.AppendRunEventInput{
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.started",
		Message:   "Run started.",
		Payload:   domain.JSONMap{"phase": "planning"},
	})
	if err != nil {
		t.Fatalf("AppendRunEvent: %v", err)
	}
	if event.EventID == "" {
		t.Fatalf("event id must be set")
	}

	artifact, err := store.CreateArtifact(ctx, domain.CreateArtifactInput{
		RunID:      run.RunID,
		ThreadID:   thread.ThreadID,
		Kind:       "report",
		Path:       "outputs/report.md",
		Title:      "Report",
		MimeType:   "text/markdown",
		SizeBytes:  42,
		SHA256:     "abc123",
		StorageURI: "file://outputs/report.md",
		Metadata:   domain.JSONMap{"source": "stub"},
	})
	if err != nil {
		t.Fatalf("CreateArtifact: %v", err)
	}
	if artifact.ArtifactID == "" {
		t.Fatalf("artifact id must be set")
	}

	events, err := store.ListRunEvents(ctx, run.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	if len(events) != 1 || events[0].EventKind != "run.started" {
		t.Fatalf("events = %+v, want one run.started", events)
	}

	artifacts, err := store.ListRunArtifacts(ctx, run.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunArtifacts: %v", err)
	}
	if len(artifacts) != 1 || artifacts[0].Path != "outputs/report.md" {
		t.Fatalf("artifacts = %+v, want report", artifacts)
	}
}

func TestMemoryStoreListThreadsPaginatesWithTotalCount(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	for _, title := range []string{"first", "second", "third", "fourth"} {
		if _, err := store.CreateThread(ctx, domain.CreateThreadInput{
			UserID: "user-1",
			Title:  title,
		}); err != nil {
			t.Fatalf("CreateThread %q: %v", title, err)
		}
		time.Sleep(time.Millisecond)
	}

	page, err := store.ListThreads(ctx, 2, 1, "")
	if err != nil {
		t.Fatalf("ListThreads: %v", err)
	}
	if page.TotalCount != 4 {
		t.Fatalf("total count = %d, want 4", page.TotalCount)
	}
	if page.Limit != 2 || page.Offset != 1 {
		t.Fatalf("page = limit %d offset %d, want limit 2 offset 1", page.Limit, page.Offset)
	}
	if len(page.Threads) != 2 {
		t.Fatalf("threads = %d, want 2", len(page.Threads))
	}
	if page.Threads[0].Title != "third" || page.Threads[1].Title != "second" {
		t.Fatalf("paged titles = %q, %q; want third, second", page.Threads[0].Title, page.Threads[1].Title)
	}
}

func TestMemoryStoreTenantScopedQueriesFilterByUser(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	aliceThread, err := store.CreateThread(ctx, domain.CreateThreadInput{
		UserID:          "alice",
		Title:           "Alice thread",
		InitialMessages: []domain.ThreadMessage{{Role: "user", Content: "alice message"}},
	})
	if err != nil {
		t.Fatalf("CreateThread alice: %v", err)
	}
	aliceRun, err := store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: aliceThread.ThreadID,
		UserID:   "alice",
		Goal:     "alice run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "alice run"}},
	})
	if err != nil {
		t.Fatalf("CreateRun alice: %v", err)
	}
	aliceEvent, err := store.AppendRunEvent(ctx, domain.AppendRunEventInput{
		RunID:     aliceRun.RunID,
		ThreadID:  aliceThread.ThreadID,
		EventKind: "message.delta",
		Message:   "alice trace",
	})
	if err != nil {
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

	bobThread, err := store.CreateThread(ctx, domain.CreateThreadInput{UserID: "bob", Title: "Bob thread"})
	if err != nil {
		t.Fatalf("CreateThread bob: %v", err)
	}
	if _, err := store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: bobThread.ThreadID,
		UserID:   "bob",
		Goal:     "bob run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "bob run"}},
	}); err != nil {
		t.Fatalf("CreateRun bob: %v", err)
	}

	alicePage, err := store.ListThreadsForUser(ctx, "alice", 10, 0, "")
	if err != nil {
		t.Fatalf("ListThreadsForUser alice: %v", err)
	}
	if alicePage.TotalCount != 1 || len(alicePage.Threads) != 1 || alicePage.Threads[0].ThreadID != aliceThread.ThreadID {
		t.Fatalf("alice threads = %+v, want only alice thread", alicePage)
	}
	if _, err := store.GetThreadForUser(ctx, aliceThread.ThreadID, "bob"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetThreadForUser bob err = %v, want ErrNotFound", err)
	}
	if _, err := store.ListThreadMessagesForUser(ctx, aliceThread.ThreadID, "bob"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("ListThreadMessagesForUser bob err = %v, want ErrNotFound", err)
	}
	if _, err := store.GetRunForUser(ctx, aliceRun.RunID, "bob"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetRunForUser bob err = %v, want ErrNotFound", err)
	}
	bobRuns, err := store.ListRunsForUser(ctx, "bob", "", "", 10, 0)
	if err != nil {
		t.Fatalf("ListRunsForUser bob: %v", err)
	}
	if len(bobRuns) != 1 || bobRuns[0].UserID != "bob" {
		t.Fatalf("bob runs = %+v, want only bob run", bobRuns)
	}
	if _, err := store.ListRunEventsForUser(ctx, aliceRun.RunID, "bob", 10); !errors.Is(err, ErrNotFound) {
		t.Fatalf("ListRunEventsForUser bob err = %v, want ErrNotFound", err)
	}
	if _, err := store.ListRunEventsAfterForUser(ctx, aliceRun.RunID, "bob", aliceEvent.Sequence-1, 10); !errors.Is(err, ErrNotFound) {
		t.Fatalf("ListRunEventsAfterForUser bob err = %v, want ErrNotFound", err)
	}
	if _, err := store.ListRunArtifactsForUser(ctx, aliceRun.RunID, "bob", 10); !errors.Is(err, ErrNotFound) {
		t.Fatalf("ListRunArtifactsForUser bob err = %v, want ErrNotFound", err)
	}
	if _, err := store.GetArtifactForUser(ctx, aliceArtifact.ArtifactID, "bob"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetArtifactForUser bob err = %v, want ErrNotFound", err)
	}
}

func TestMemoryStoreCreateAndListUserAccounts(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	user, err := store.CreateUser(ctx, domain.CreateUserInput{
		Email:       "ada@example.org",
		DisplayName: "Ada Lovelace",
		Role:        "admin",
		OrgID:       "local-org",
		Metadata:    domain.JSONMap{"source": "admin_console"},
	})
	if err != nil {
		t.Fatalf("CreateUser: %v", err)
	}
	if user.UserID == "" {
		t.Fatalf("created user must have user_id")
	}
	if user.Status != "active" {
		t.Fatalf("status = %q, want active", user.Status)
	}

	users, err := store.ListUsers(ctx, 10, "")
	if err != nil {
		t.Fatalf("ListUsers: %v", err)
	}
	if len(users) != 1 {
		t.Fatalf("users = %d, want 1", len(users))
	}
	got := users[0]
	if got.Email != "ada@example.org" || got.DisplayName != "Ada Lovelace" || got.Role != "admin" || got.OrgID != "local-org" {
		t.Fatalf("unexpected user: %+v", got)
	}
	if got.Metadata["source"] != "admin_console" {
		t.Fatalf("metadata = %#v, want source", got.Metadata)
	}

	filtered, err := store.ListUsers(ctx, 10, "lovelace")
	if err != nil {
		t.Fatalf("ListUsers filtered: %v", err)
	}
	if len(filtered) != 1 || filtered[0].UserID != user.UserID {
		t.Fatalf("filtered users = %+v, want created user", filtered)
	}
}

func TestMemoryStoreDeactivateUserAccount(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	user, err := store.CreateUser(ctx, domain.CreateUserInput{
		Email:       "delete-me@example.org",
		DisplayName: "Delete Me",
		Status:      "active",
	})
	if err != nil {
		t.Fatalf("CreateUser: %v", err)
	}

	deactivated, err := store.UpdateUserStatus(ctx, user.UserID, "disabled")
	if err != nil {
		t.Fatalf("UpdateUserStatus: %v", err)
	}
	if deactivated.Status != "disabled" {
		t.Fatalf("status = %q, want disabled", deactivated.Status)
	}
	if !deactivated.UpdatedAt.After(user.UpdatedAt) {
		t.Fatalf("updated_at = %s, want after original %s", deactivated.UpdatedAt, user.UpdatedAt)
	}

	users, err := store.ListUsers(ctx, 10, "delete-me")
	if err != nil {
		t.Fatalf("ListUsers: %v", err)
	}
	if len(users) != 1 || users[0].Status != "disabled" {
		t.Fatalf("users = %+v, want disabled user still visible for audit", users)
	}
}

func TestMemoryStoreCreateAndListOrganizations(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	org, err := store.CreateOrganization(ctx, domain.CreateOrganizationInput{
		OrgID:    "allen-institute",
		Name:     "Allen Institute",
		Status:   "active",
		Metadata: domain.JSONMap{"source": "admin_console"},
	})
	if err != nil {
		t.Fatalf("CreateOrganization: %v", err)
	}
	if org.OrgID != "allen-institute" || org.Name != "Allen Institute" || org.Status != "active" {
		t.Fatalf("organization = %+v, want created organization fields", org)
	}

	orgs, err := store.ListOrganizations(ctx, 10, "allen")
	if err != nil {
		t.Fatalf("ListOrganizations: %v", err)
	}
	if len(orgs) != 1 || orgs[0].OrgID != org.OrgID {
		t.Fatalf("organizations = %+v, want created organization", orgs)
	}
	if orgs[0].Metadata["source"] != "admin_console" {
		t.Fatalf("metadata = %#v, want source", orgs[0].Metadata)
	}
}

func TestMemoryStoreUpsertsAndListsWorkerHeartbeats(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()
	started := time.Date(2026, 6, 1, 12, 0, 0, 0, time.UTC)
	firstBeat := started.Add(10 * time.Second)
	secondBeat := started.Add(70 * time.Second)

	first, err := store.UpsertWorkerHeartbeat(ctx, domain.UpsertWorkerHeartbeatInput{
		WorkerID:        "deepagents-worker-a",
		WorkerKind:      "deepagents",
		Status:          "idle",
		Hostname:        "host-a",
		Version:         "test-version",
		StartedAt:       started,
		LastHeartbeatAt: firstBeat,
		Metadata:        domain.JSONMap{"durable": "ultra-deepagents-worker"},
	})
	if err != nil {
		t.Fatalf("UpsertWorkerHeartbeat first: %v", err)
	}
	if first.WorkerID != "deepagents-worker-a" || first.WorkerKind != "deepagents" || first.Status != "idle" {
		t.Fatalf("first worker = %+v, want deepagents idle record", first)
	}
	if !first.StartedAt.Equal(started) || !first.LastHeartbeatAt.Equal(firstBeat) {
		t.Fatalf("first heartbeat timestamps = %+v", first)
	}

	second, err := store.UpsertWorkerHeartbeat(ctx, domain.UpsertWorkerHeartbeatInput{
		WorkerID:        "deepagents-worker-a",
		WorkerKind:      "deepagents",
		Status:          "busy",
		CurrentRunID:    "run_123",
		Hostname:        "host-a",
		Version:         "test-version-2",
		LastHeartbeatAt: secondBeat,
		Metadata:        domain.JSONMap{"active_tasks": 1},
	})
	if err != nil {
		t.Fatalf("UpsertWorkerHeartbeat second: %v", err)
	}
	if second.Status != "busy" || second.CurrentRunID != "run_123" || second.Version != "test-version-2" {
		t.Fatalf("second worker = %+v, want updated busy record", second)
	}
	if !second.StartedAt.Equal(started) {
		t.Fatalf("second started_at = %s, want original %s", second.StartedAt, started)
	}
	if !second.LastHeartbeatAt.Equal(secondBeat) {
		t.Fatalf("second last heartbeat = %s, want %s", second.LastHeartbeatAt, secondBeat)
	}

	workers, err := store.ListWorkerHeartbeats(ctx, 10)
	if err != nil {
		t.Fatalf("ListWorkerHeartbeats: %v", err)
	}
	if len(workers) != 1 || workers[0].Status != "busy" || workers[0].Metadata["active_tasks"] != 1 {
		t.Fatalf("workers = %+v, want one updated worker heartbeat", workers)
	}
}

func TestMemoryStoreRejectsDuplicateOrganizationID(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	if _, err := store.CreateOrganization(ctx, domain.CreateOrganizationInput{OrgID: "smithsonian", Name: "Smithsonian"}); err != nil {
		t.Fatalf("CreateOrganization first: %v", err)
	}
	if _, err := store.CreateOrganization(ctx, domain.CreateOrganizationInput{OrgID: "smithsonian", Name: "Smithsonian duplicate"}); !errors.Is(err, ErrConflict) {
		t.Fatalf("CreateOrganization duplicate err = %v, want ErrConflict", err)
	}
}

func TestMemoryStoreRejectsDuplicateUserEmailCaseInsensitive(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	if _, err := store.CreateUser(ctx, domain.CreateUserInput{Email: "Ada@example.org"}); err != nil {
		t.Fatalf("CreateUser first: %v", err)
	}
	if _, err := store.CreateUser(ctx, domain.CreateUserInput{Email: "ada@example.org"}); !errors.Is(err, ErrConflict) {
		t.Fatalf("CreateUser duplicate err = %v, want ErrConflict", err)
	}
}

func TestMemoryStoreListRunEventsAfterSequenceReturnsAscendingPage(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	thread, err := store.CreateThread(ctx, domain.CreateThreadInput{
		UserID: "user-1",
		Title:  "Long trace",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Run long autonomous work.",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	for idx := 0; idx < 5; idx++ {
		if _, err := store.AppendRunEvent(ctx, domain.AppendRunEventInput{
			RunID:     run.RunID,
			ThreadID:  thread.ThreadID,
			EventKind: "message.delta",
			Payload:   domain.JSONMap{"idx": idx},
		}); err != nil {
			t.Fatalf("AppendRunEvent %d: %v", idx, err)
		}
	}

	events, err := store.ListRunEventsAfter(ctx, run.RunID, 2, 2)
	if err != nil {
		t.Fatalf("ListRunEventsAfter: %v", err)
	}
	if len(events) != 2 {
		t.Fatalf("events = %d, want 2", len(events))
	}
	got := []int64{events[0].Sequence, events[1].Sequence}
	want := []int64{3, 4}
	if got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("sequences = %v, want %v", got, want)
	}
}

func TestMemoryStoreUpdateRunStatusKeepsTerminalRunImmutable(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	thread, err := store.CreateThread(ctx, domain.CreateThreadInput{
		UserID: "user-1",
		Title:  "Terminal run",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Run a long analysis.",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	completed, err := store.UpdateRunStatus(ctx, run.RunID, domain.RunStatusSucceeded, "final answer", "")
	if err != nil {
		t.Fatalf("UpdateRunStatus succeeded: %v", err)
	}
	if completed.CompletedAt == nil {
		t.Fatalf("completed run must have completed_at set")
	}

	reopened, err := store.UpdateRunStatus(ctx, run.RunID, domain.RunStatusRunning, "", "")
	if err != nil {
		t.Fatalf("UpdateRunStatus stale running: %v", err)
	}
	if reopened.Status != domain.RunStatusSucceeded {
		t.Fatalf("status = %s, want terminal succeeded to be preserved", reopened.Status)
	}
	if reopened.ResponseText != "final answer" {
		t.Fatalf("response text = %q, want first terminal response preserved", reopened.ResponseText)
	}
	if reopened.CompletedAt == nil || !reopened.CompletedAt.Equal(*completed.CompletedAt) {
		t.Fatalf("completed_at changed after stale update: before=%v after=%v", completed.CompletedAt, reopened.CompletedAt)
	}

	failed, err := store.UpdateRunStatus(ctx, run.RunID, domain.RunStatusFailed, "", "late failure")
	if err != nil {
		t.Fatalf("UpdateRunStatus stale failure: %v", err)
	}
	if failed.Status != domain.RunStatusSucceeded || failed.Error != "" {
		t.Fatalf("stale failure mutated terminal run: %+v", failed)
	}
}

func TestMemoryStoreCompleteRunRepairsSucceededRunMissingResponseText(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	thread, err := store.CreateThread(ctx, domain.CreateThreadInput{UserID: "user-1", Title: "Terminal repair"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Repair missing terminal response.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Repair missing terminal response."}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	if _, err := store.UpdateRunStatus(ctx, run.RunID, domain.RunStatusSucceeded, "", ""); err != nil {
		t.Fatalf("UpdateRunStatus succeeded with empty response: %v", err)
	}

	repaired, err := store.CompleteRun(ctx, domain.CompleteRunInput{
		RunID:        run.RunID,
		ResponseText: "Recovered final answer.",
	})
	if err != nil {
		t.Fatalf("CompleteRun repair: %v", err)
	}
	if repaired.Status != domain.RunStatusSucceeded || repaired.ResponseText != "Recovered final answer." {
		t.Fatalf("repaired run = %+v, want succeeded with recovered response text", repaired)
	}
	messages, err := store.ListThreadMessages(ctx, thread.ThreadID)
	if err != nil {
		t.Fatalf("ListThreadMessages: %v", err)
	}
	if got, want := len(messages), 2; got != want {
		t.Fatalf("messages = %d, want %d user+assistant messages: %+v", got, want, messages)
	}
	if messages[1].Role != "assistant" || messages[1].Content != "Recovered final answer." || messages[1].RunID != run.RunID {
		t.Fatalf("assistant message = %+v, want recovered response owned by run", messages[1])
	}
}

func TestMemoryStoreRunLeasePreventsConcurrentWorkersAndCanExpire(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	thread, err := store.CreateThread(ctx, domain.CreateThreadInput{
		UserID: "user-1",
		Title:  "Lease thread",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "long autonomous run",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	now := time.Date(2026, 6, 1, 12, 0, 0, 0, time.UTC)
	first, err := store.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    run.RunID,
		WorkerID: "worker-a",
		TTL:      time.Minute,
		Now:      now,
	})
	if err != nil {
		t.Fatalf("AcquireRunLease first: %v", err)
	}
	if first.LeaseToken == "" || first.WorkerID != "worker-a" || !first.LeaseExpiresAt.Equal(now.Add(time.Minute)) {
		t.Fatalf("first lease = %+v, want worker-a token expiring after ttl", first)
	}
	updatedRun, err := store.GetRun(ctx, run.RunID)
	if err != nil {
		t.Fatalf("GetRun: %v", err)
	}
	if updatedRun.Status != domain.RunStatusRunning || updatedRun.StartedAt == nil {
		t.Fatalf("claimed run = %+v, want running with started_at", updatedRun)
	}

	if _, err := store.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    run.RunID,
		WorkerID: "worker-b",
		TTL:      time.Minute,
		Now:      now.Add(30 * time.Second),
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("AcquireRunLease competing err = %v, want ErrConflict", err)
	}

	second, err := store.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    run.RunID,
		WorkerID: "worker-b",
		TTL:      2 * time.Minute,
		Now:      now.Add(2 * time.Minute),
	})
	if err != nil {
		t.Fatalf("AcquireRunLease after expiry: %v", err)
	}
	if second.WorkerID != "worker-b" || second.LeaseToken == first.LeaseToken || !second.LeaseExpiresAt.Equal(now.Add(4*time.Minute)) {
		t.Fatalf("second lease = %+v, want replacement worker-b lease", second)
	}
}

func TestMemoryStoreRunLeaseRenewAndReleaseRequireToken(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	thread, err := store.CreateThread(ctx, domain.CreateThreadInput{UserID: "user-1", Title: "Lease token"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := store.CreateRun(ctx, domain.CreateRunInput{ThreadID: thread.ThreadID, UserID: "user-1", Goal: "work"})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	now := time.Date(2026, 6, 1, 12, 0, 0, 0, time.UTC)
	lease, err := store.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    run.RunID,
		WorkerID: "worker-a",
		TTL:      time.Minute,
		Now:      now,
	})
	if err != nil {
		t.Fatalf("AcquireRunLease: %v", err)
	}

	if _, err := store.RenewRunLease(ctx, domain.RenewRunLeaseInput{
		RunID:      run.RunID,
		LeaseToken: "wrong-token",
		TTL:        time.Minute,
		Now:        now.Add(30 * time.Second),
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("RenewRunLease wrong token err = %v, want ErrConflict", err)
	}
	renewed, err := store.RenewRunLease(ctx, domain.RenewRunLeaseInput{
		RunID:      run.RunID,
		LeaseToken: lease.LeaseToken,
		TTL:        5 * time.Minute,
		Now:        now.Add(30 * time.Second),
	})
	if err != nil {
		t.Fatalf("RenewRunLease: %v", err)
	}
	if !renewed.LeaseExpiresAt.Equal(now.Add(30*time.Second + 5*time.Minute)) {
		t.Fatalf("renewed lease expiry = %s, want now+ttl", renewed.LeaseExpiresAt)
	}

	if err := store.ReleaseRunLease(ctx, domain.ReleaseRunLeaseInput{RunID: run.RunID, LeaseToken: "wrong-token"}); !errors.Is(err, ErrConflict) {
		t.Fatalf("ReleaseRunLease wrong token err = %v, want ErrConflict", err)
	}
	if err := store.ReleaseRunLease(ctx, domain.ReleaseRunLeaseInput{RunID: run.RunID, LeaseToken: lease.LeaseToken}); err != nil {
		t.Fatalf("ReleaseRunLease: %v", err)
	}
	if _, err := store.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    run.RunID,
		WorkerID: "worker-b",
		TTL:      time.Minute,
		Now:      now.Add(time.Minute),
	}); err != nil {
		t.Fatalf("AcquireRunLease after release: %v", err)
	}
}

func TestMemoryStoreClearRunLeaseEvictsAnyActiveToken(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()

	thread, err := store.CreateThread(ctx, domain.CreateThreadInput{UserID: "user-1", Title: "Clear lease"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := store.CreateRun(ctx, domain.CreateRunInput{ThreadID: thread.ThreadID, UserID: "user-1", Goal: "recover"})
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
