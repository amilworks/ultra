package store

import (
	"context"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

func TestPruneRunEventDeltasTerminalOnlyAndPreservesStructural(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	st := NewMemoryStore()
	thread, err := st.CreateThread(ctx, domain.CreateThreadInput{UserID: "u", Title: "t"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}

	seed := func(kinds ...string) string {
		run, err := st.CreateRun(ctx, domain.CreateRunInput{ThreadID: thread.ThreadID, UserID: "u", Goal: "g"})
		if err != nil {
			t.Fatalf("CreateRun: %v", err)
		}
		for _, k := range kinds {
			if _, err := st.AppendRunEvent(ctx, domain.AppendRunEventInput{RunID: run.RunID, EventKind: k}); err != nil {
				t.Fatalf("AppendRunEvent %q: %v", k, err)
			}
		}
		return run.RunID
	}

	// Terminal run: prunable deltas interleaved with structural/terminal/usage events.
	terminalRun := seed(
		"message.delta", "tool_call.started", "subagent.message.delta",
		"run.token_usage", "message.delta", "run.completed",
	)
	if _, err := st.UpdateRunStatus(ctx, terminalRun, domain.RunStatusSucceeded, "done", ""); err != nil {
		t.Fatalf("UpdateRunStatus: %v", err)
	}

	// Active run: deltas that must NEVER be pruned — a reconnecting client needs the full prefix.
	activeRun := seed("message.delta", "message.delta", "subagent.message.delta")

	kinds := []string{"message.delta", "subagent.message.delta"}

	// A cutoff in the PAST: the just-completed run is still inside the grace window -> nothing pruned.
	if n, err := st.PruneRunEventDeltas(ctx, time.Now().Add(-time.Hour), kinds, 1000); err != nil || n != 0 {
		t.Fatalf("recent terminal run must be kept: n=%d err=%v", n, err)
	}

	// A cutoff in the FUTURE: the terminal run is past the TTL -> its 3 deltas prune; structural stays.
	n, err := st.PruneRunEventDeltas(ctx, time.Now().Add(time.Hour), kinds, 1000)
	if err != nil {
		t.Fatalf("prune: %v", err)
	}
	if n != 3 {
		t.Fatalf("pruned %d delta events, want 3", n)
	}

	termEvents, _ := st.ListRunEvents(ctx, terminalRun, 1000)
	kept := map[string]bool{}
	for _, e := range termEvents {
		if e.EventKind == "message.delta" || e.EventKind == "subagent.message.delta" {
			t.Fatalf("a delta survived the prune: %s", e.EventKind)
		}
		kept[e.EventKind] = true
	}
	for _, want := range []string{"tool_call.started", "run.token_usage", "run.completed"} {
		if !kept[want] {
			t.Fatalf("structural event %q was wrongly pruned", want)
		}
	}

	// The active run is untouched even though the cutoff was in the future.
	activeEvents, _ := st.ListRunEvents(ctx, activeRun, 1000)
	if len(activeEvents) != 3 {
		t.Fatalf("active run must be untouched: got %d events, want 3", len(activeEvents))
	}
}

func TestPruneRunEventDeltasRespectsBatchLimit(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	st := NewMemoryStore()
	thread, _ := st.CreateThread(ctx, domain.CreateThreadInput{UserID: "u", Title: "t"})
	run, _ := st.CreateRun(ctx, domain.CreateRunInput{ThreadID: thread.ThreadID, UserID: "u", Goal: "g"})
	for i := 0; i < 10; i++ {
		if _, err := st.AppendRunEvent(ctx, domain.AppendRunEventInput{RunID: run.RunID, EventKind: "message.delta"}); err != nil {
			t.Fatalf("append: %v", err)
		}
	}
	if _, err := st.UpdateRunStatus(ctx, run.RunID, domain.RunStatusFailed, "", "boom"); err != nil {
		t.Fatalf("UpdateRunStatus: %v", err)
	}

	kinds := []string{"message.delta"}
	cutoff := time.Now().Add(time.Hour)
	// Batch of 4 removes only 4, leaving 6.
	if n, _ := st.PruneRunEventDeltas(ctx, cutoff, kinds, 4); n != 4 {
		t.Fatalf("first batch pruned %d, want 4", n)
	}
	remaining, _ := st.ListRunEvents(ctx, run.RunID, 1000)
	if len(remaining) != 6 {
		t.Fatalf("after batch 1: %d events remain, want 6", len(remaining))
	}
}
