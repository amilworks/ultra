package store

import (
	"context"
	"testing"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

// Conversation delete used to be concealment sold as erasure: the handler ran
// `UPDATE control_threads SET status='deleted'`, so no row was removed, the
// schema's ON DELETE CASCADE chains never fired, and the whole transcript stayed
// readable in metadata.frontend_state while the dialog told the user it had been
// removed from storage.
//
// These tests pin the new contract. The Postgres implementation gets most of its
// sweep from declared cascades; the in-memory one has to clear every map by
// hand, so this is where a missed map shows up.

func seedDeletableThread(t *testing.T, ctx context.Context, store *MemoryStore, userID string) (domain.ThreadRecord, domain.RunRecord) {
	t.Helper()

	thread, err := store.CreateThread(ctx, domain.CreateThreadInput{
		UserID: userID,
		Title:  "Sensitive analysis",
		InitialMessages: []domain.ThreadMessage{{
			Role:    "user",
			Content: "Something the user later wants gone.",
		}},
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}

	run, err := store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   userID,
		Goal:     "Something the user later wants gone.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Something the user later wants gone."}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	if _, err := store.AppendRunEvent(ctx, domain.AppendRunEventInput{
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.started",
		Message:   "Run started.",
	}); err != nil {
		t.Fatalf("AppendRunEvent: %v", err)
	}

	if _, err := store.CreateArtifact(ctx, domain.CreateArtifactInput{
		RunID:      run.RunID,
		ThreadID:   thread.ThreadID,
		Kind:       "figure",
		Path:       "outputs/figure.png",
		Title:      "Figure",
		MimeType:   "image/png",
		StorageURI: "file:///artifacts/" + run.RunID + "/figure.png",
	}); err != nil {
		t.Fatalf("CreateArtifact: %v", err)
	}

	return thread, run
}

func TestHardDeleteRemovesEveryTraceOfTheConversation(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()
	thread, run := seedDeletableThread(t, ctx, store, "user-1")

	storageURIs, err := store.HardDeleteThreadForUser(ctx, thread.ThreadID, "user-1")
	if err != nil {
		t.Fatalf("HardDeleteThreadForUser: %v", err)
	}

	// The blobs are outside Postgres, so the caller has to be handed the URIs or
	// the bytes survive a "permanent" delete.
	if len(storageURIs) != 1 {
		t.Fatalf("storage URIs = %v, want the one artifact's URI", storageURIs)
	}

	// The thread itself.
	if _, err := store.GetThreadForUser(ctx, thread.ThreadID, "user-1"); err == nil {
		t.Fatal("thread still readable after hard delete")
	}

	// Messages: this is the transcript the old path left behind.
	messages, err := store.ListThreadMessages(ctx, thread.ThreadID)
	if err == nil && len(messages) != 0 {
		t.Fatalf("thread messages survived hard delete: %+v", messages)
	}

	// Runs carry goal and response_text — the question and the answer.
	if _, err := store.GetRun(ctx, run.RunID); err == nil {
		t.Fatal("run still readable after hard delete")
	}

	events, err := store.ListRunEvents(ctx, run.RunID, 10)
	if err == nil && len(events) != 0 {
		t.Fatalf("run events survived hard delete: %+v", events)
	}

	artifacts, err := store.ListRunArtifacts(ctx, run.RunID, 10)
	if err == nil && len(artifacts) != 0 {
		t.Fatalf("artifacts survived hard delete: %+v", artifacts)
	}
}

func TestHardDeleteClearsTokenUsageWhichHasNoForeignKey(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()
	thread, run := seedDeletableThread(t, ctx, store, "user-1")

	// control_run_token_usage and control_run_token_usage_finalized key on run_id
	// with NO foreign key, so in Postgres the cascade cannot reach them. They are
	// the rows most likely to be forgotten and left as orphans.
	if _, _, err := store.RecordRunTokenUsage(ctx, domain.RecordRunTokenUsageInput{
		RunID:        run.RunID,
		UsageEventID: "usage-1",
		UserID:       "user-1",
		Model:        "deepseek_v4",
		InputTokens:  100,
		OutputTokens: 20,
		TotalTokens:  120,
	}); err != nil {
		t.Fatalf("RecordRunTokenUsage: %v", err)
	}
	store.mu.Lock()
	store.runTokenUsageFinalized[run.RunID] = true
	store.mu.Unlock()

	if _, err := store.HardDeleteThreadForUser(ctx, thread.ThreadID, "user-1"); err != nil {
		t.Fatalf("HardDeleteThreadForUser: %v", err)
	}

	if _, ok := store.runTokenUsage[run.RunID]; ok {
		t.Fatal("token usage row orphaned by hard delete")
	}
	if _, ok := store.runTokenUsageFinalized[run.RunID]; ok {
		t.Fatal("finalized token usage row orphaned by hard delete")
	}
}

func TestHardDeleteDoesNotTouchUploadedFiles(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()
	thread, _ := seedDeletableThread(t, ctx, store, "user-1")

	// Uploaded files are independent of conversations by design — control_resources
	// has no thread FK and its own hard-delete path (PurgeResource). Deleting a
	// conversation must never delete the user's data.
	//
	// Seeded directly rather than through the resource API: this asserts the
	// sweep's blast radius, and going through the front door would only add ways
	// for the test to fail for reasons unrelated to deletion.
	store.mu.Lock()
	store.resources["resource-1"] = domain.ResourceRecord{
		ResourceID:   "resource-1",
		OriginalName: "microscopy.tif",
	}
	store.mu.Unlock()

	if _, err := store.HardDeleteThreadForUser(ctx, thread.ThreadID, "user-1"); err != nil {
		t.Fatalf("HardDeleteThreadForUser: %v", err)
	}

	store.mu.RLock()
	_, stillThere := store.resources["resource-1"]
	store.mu.RUnlock()
	if !stillThere {
		t.Fatal("a conversation delete destroyed the user's uploaded file")
	}
}

func TestHardDeleteRefusesSomeoneElsesConversation(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()
	thread, _ := seedDeletableThread(t, ctx, store, "user-1")

	if _, err := store.HardDeleteThreadForUser(ctx, thread.ThreadID, "user-2"); err == nil {
		t.Fatal("a different user was allowed to hard delete this conversation")
	}

	// And the refusal must be total, not partial — ownership is checked before
	// anything is removed.
	if _, err := store.GetThreadForUser(ctx, thread.ThreadID, "user-1"); err != nil {
		t.Fatalf("owner lost their conversation to a rejected delete: %v", err)
	}
}

func TestHardDeleteIsNotFoundForAnAlreadyDeletedConversation(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := NewMemoryStore()
	thread, _ := seedDeletableThread(t, ctx, store, "user-1")

	if _, err := store.HardDeleteThreadForUser(ctx, thread.ThreadID, "user-1"); err != nil {
		t.Fatalf("first delete: %v", err)
	}
	if _, err := store.HardDeleteThreadForUser(ctx, thread.ThreadID, "user-1"); err == nil {
		t.Fatal("second delete should report not found, not succeed silently")
	}
}
