package store

import (
	"context"
	"fmt"
	"os"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/jackc/pgx/v5/pgxpool"
)

func appendEventTestStore(t *testing.T) *PostgresStore {
	t.Helper()
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	t.Cleanup(pool.Close)
	if err := ApplyPostgresSchema(ctx, pool); err != nil {
		t.Fatalf("ApplyPostgresSchema: %v", err)
	}
	return NewPostgresStore(pool)
}

func appendEventTestRun(t *testing.T, s *PostgresStore) domain.RunRecord {
	t.Helper()
	ctx := context.Background()
	userID := fmt.Sprintf("append-evt-user-%d", time.Now().UnixNano())
	thread, err := s.CreateThread(ctx, domain.CreateThreadInput{UserID: userID, Title: "append event"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := s.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   userID,
		Goal:     "append event",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "append"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	return run
}

// The single-statement allocator must hand out gapless, duplicate-free
// sequences under heavy same-run concurrency with zero unique-violation
// errors surfacing to callers.
func TestPostgresAppendRunEventConcurrentSameRunSequences(t *testing.T) {
	s := appendEventTestStore(t)
	run := appendEventTestRun(t, s)
	ctx := context.Background()

	const workers = 16
	const eventsPerWorker = 25
	var wg sync.WaitGroup
	sequences := make(chan int64, workers*eventsPerWorker)
	errs := make(chan error, workers*eventsPerWorker)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < eventsPerWorker; i++ {
				event, err := s.AppendRunEvent(ctx, domain.AppendRunEventInput{
					RunID:     run.RunID,
					ThreadID:  run.ThreadID,
					EventKind: "log.delta",
					Message:   "concurrent append",
				})
				if err != nil {
					errs <- err
					return
				}
				sequences <- event.Sequence
			}
		}()
	}
	wg.Wait()
	close(sequences)
	close(errs)
	for err := range errs {
		t.Fatalf("AppendRunEvent: %v", err)
	}
	collected := make([]int64, 0, workers*eventsPerWorker)
	for sequence := range sequences {
		collected = append(collected, sequence)
	}
	if len(collected) != workers*eventsPerWorker {
		t.Fatalf("appended %d events, want %d", len(collected), workers*eventsPerWorker)
	}
	sort.Slice(collected, func(i, j int) bool { return collected[i] < collected[j] })
	for index, sequence := range collected {
		if sequence != int64(index+1) {
			t.Fatalf("sequence[%d] = %d, want gapless 1..%d (got %v...)", index, sequence, len(collected), collected[:index+1])
		}
	}
}

// The allocator must self-heal when its counter row is missing or stale
// relative to the events table (deleted backfill, mixed-version writers).
func TestPostgresAppendRunEventSelfHealsStaleSequenceCounter(t *testing.T) {
	s := appendEventTestStore(t)
	run := appendEventTestRun(t, s)
	ctx := context.Background()

	first, err := s.AppendRunEvent(ctx, domain.AppendRunEventInput{
		RunID: run.RunID, EventKind: "log.delta", Message: "one",
	})
	if err != nil {
		t.Fatalf("AppendRunEvent: %v", err)
	}
	// Sabotage the counter both ways: delete it, then recreate it stale.
	if _, err := s.pool.Exec(ctx, `DELETE FROM control_run_event_sequences WHERE run_id = $1`, run.RunID); err != nil {
		t.Fatalf("delete counter: %v", err)
	}
	second, err := s.AppendRunEvent(ctx, domain.AppendRunEventInput{
		RunID: run.RunID, EventKind: "log.delta", Message: "two",
	})
	if err != nil {
		t.Fatalf("AppendRunEvent after deleted counter: %v", err)
	}
	if second.Sequence <= first.Sequence {
		t.Fatalf("sequence after deleted counter = %d, want > %d", second.Sequence, first.Sequence)
	}
	if _, err := s.pool.Exec(ctx, `UPDATE control_run_event_sequences SET last_sequence = 0 WHERE run_id = $1`, run.RunID); err != nil {
		t.Fatalf("stale counter: %v", err)
	}
	third, err := s.AppendRunEvent(ctx, domain.AppendRunEventInput{
		RunID: run.RunID, EventKind: "log.delta", Message: "three",
	})
	if err != nil {
		t.Fatalf("AppendRunEvent after stale counter: %v", err)
	}
	if third.Sequence <= second.Sequence {
		t.Fatalf("sequence after stale counter = %d, want > %d", third.Sequence, second.Sequence)
	}
}

func TestPostgresAppendRunEventIfRunActiveOutcomes(t *testing.T) {
	s := appendEventTestStore(t)
	run := appendEventTestRun(t, s)
	ctx := context.Background()

	appendedEvent, outcome, err := s.AppendRunEventIfRunActive(ctx, domain.AppendRunEventInput{
		EventID:   "evt-" + run.RunID + "-active",
		RunID:     run.RunID,
		ThreadID:  run.ThreadID,
		EventKind: "message.delta",
		Message:   "original message",
	})
	if err != nil || outcome != RunEventAppendOutcomeAppended {
		t.Fatalf("active append outcome = %v err = %v, want appended", outcome, err)
	}
	if appendedEvent.Sequence < 1 {
		t.Fatalf("appended sequence = %d, want >= 1", appendedEvent.Sequence)
	}

	duplicate, outcome, err := s.AppendRunEventIfRunActive(ctx, domain.AppendRunEventInput{
		EventID:   "evt-" + run.RunID + "-active",
		RunID:     run.RunID,
		ThreadID:  run.ThreadID,
		EventKind: "message.delta",
		Message:   "mutated replay must not win",
	})
	if err != nil || outcome != RunEventAppendOutcomeDuplicate {
		t.Fatalf("duplicate outcome = %v err = %v, want duplicate", outcome, err)
	}
	if duplicate.Message != "original message" {
		t.Fatalf("duplicate returned %q, want stored original", duplicate.Message)
	}

	if _, err := s.UpdateRunStatus(ctx, run.RunID, domain.RunStatusCanceled, "", "done"); err != nil {
		t.Fatalf("UpdateRunStatus: %v", err)
	}
	_, outcome, err = s.AppendRunEventIfRunActive(ctx, domain.AppendRunEventInput{
		EventID:   "evt-" + run.RunID + "-late",
		RunID:     run.RunID,
		EventKind: "message.delta",
		Message:   "late event",
	})
	if err != nil || outcome != RunEventAppendOutcomeDropped {
		t.Fatalf("terminal append outcome = %v err = %v, want dropped", outcome, err)
	}
	// Duplicates must still be reported after the run went terminal so their
	// side effects and fanout can be replayed.
	replayed, outcome, err := s.AppendRunEventIfRunActive(ctx, domain.AppendRunEventInput{
		EventID:   "evt-" + run.RunID + "-active",
		RunID:     run.RunID,
		EventKind: "message.delta",
		Message:   "replay after terminal",
	})
	if err != nil || outcome != RunEventAppendOutcomeDuplicate {
		t.Fatalf("terminal duplicate outcome = %v err = %v, want duplicate", outcome, err)
	}
	if replayed.Message != "original message" {
		t.Fatalf("terminal duplicate returned %q, want stored original", replayed.Message)
	}

	_, outcome, err = s.AppendRunEventIfRunActive(ctx, domain.AppendRunEventInput{
		EventID:   "evt-missing-run",
		RunID:     "run-does-not-exist",
		EventKind: "message.delta",
	})
	if err != nil || outcome != RunEventAppendOutcomeDropped {
		t.Fatalf("missing run outcome = %v err = %v, want dropped", outcome, err)
	}
}

func TestPostgresAppendRunEventPersistsSourceSequence(t *testing.T) {
	s := appendEventTestStore(t)
	run := appendEventTestRun(t, s)
	ctx := context.Background()

	controlEvent, err := s.AppendRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt-" + run.RunID + "-control",
		RunID:     run.RunID,
		ThreadID:  run.ThreadID,
		EventKind: "run.accepted",
	})
	if err != nil {
		t.Fatalf("AppendRunEvent control event: %v", err)
	}
	if controlEvent.SourceSequence != controlEvent.Sequence {
		t.Fatalf("control source sequence = %d, want allocated sequence %d", controlEvent.SourceSequence, controlEvent.Sequence)
	}

	workerEvent, outcome, err := s.AppendRunEventIfRunActive(ctx, domain.AppendRunEventInput{
		EventID:        "evt-" + run.RunID + "-worker-000010",
		RunID:          run.RunID,
		ThreadID:       run.ThreadID,
		EventKind:      "message.delta",
		SourceSequence: 10,
		Message:        "worker source sequence",
	})
	if err != nil || outcome != RunEventAppendOutcomeAppended {
		t.Fatalf("worker append outcome = %v err = %v, want appended", outcome, err)
	}
	if workerEvent.SourceSequence != 10 {
		t.Fatalf("worker source sequence = %d, want 10", workerEvent.SourceSequence)
	}
	bySource, found, err := s.GetRunEventBySourceSequence(ctx, run.RunID, 10)
	if err != nil {
		t.Fatalf("GetRunEventBySourceSequence: %v", err)
	}
	if !found || bySource.EventID != workerEvent.EventID {
		t.Fatalf("source sequence lookup found=%v event=%+v, want %s", found, bySource, workerEvent.EventID)
	}
}
