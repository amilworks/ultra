package runcontrol

import (
	"context"
	"errors"
	"testing"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

func newSteeringFixture(t *testing.T) (context.Context, *store.MemoryStore, *Service, domain.RunRecord) {
	t.Helper()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := NewService(mem, bus)
	thread, err := service.CreateThread(ctx, CreateThreadRequest{UserID: "user-1", Title: "Steering"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "Long analysis.",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "Long analysis."}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	return ctx, mem, service, run
}

func steerEventKinds(t *testing.T, mem *store.MemoryStore, runID string) []string {
	t.Helper()
	events, err := mem.ListRunEvents(context.Background(), runID, 100)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	var kinds []string
	for _, event := range events {
		if len(event.EventKind) > 5 && event.EventKind[:6] == "steer." {
			kinds = append(kinds, event.EventKind)
		}
	}
	return kinds
}

func TestSteerRunAcceptsAndWritesTranscript(t *testing.T) {
	t.Parallel()
	ctx, mem, service, run := newSteeringFixture(t)

	record, err := service.SteerRun(ctx, SteerRunRequest{
		RunID: run.RunID, UserID: "user-1", SteerID: "steer_a", Text: "Also compare against the baseline.",
	})
	if err != nil {
		t.Fatalf("SteerRun: %v", err)
	}
	if record.Status != domain.RunSteerStatusPending {
		t.Fatalf("status = %q, want pending", record.Status)
	}
	if record.MessageID == "" {
		t.Fatal("steer has no message id — the id-dedup invariant depends on it")
	}

	// The transcript row commits with the steer: requeue reseeds jobs from
	// the transcript, so the two must never be observable apart.
	messages, err := mem.ListThreadMessages(ctx, run.ThreadID)
	if err != nil {
		t.Fatalf("ListThreadMessages: %v", err)
	}
	last := messages[len(messages)-1]
	if last.MessageID != record.MessageID || last.Role != "user" {
		t.Fatalf("transcript tail = %+v, want steering user message %s", last, record.MessageID)
	}
	if kind, _ := last.Metadata["kind"].(string); kind != "steering" {
		t.Fatalf("transcript metadata kind = %q, want steering", kind)
	}

	if kinds := steerEventKinds(t, mem, run.RunID); len(kinds) != 1 || kinds[0] != "steer.received" {
		t.Fatalf("steer events = %v, want [steer.received]", kinds)
	}
}

func TestSteerRunRetryIsIdempotent(t *testing.T) {
	t.Parallel()
	ctx, mem, service, run := newSteeringFixture(t)

	first, err := service.SteerRun(ctx, SteerRunRequest{RunID: run.RunID, UserID: "user-1", SteerID: "steer_a", Text: "Add error bars."})
	if err != nil {
		t.Fatalf("SteerRun: %v", err)
	}
	retry, err := service.SteerRun(ctx, SteerRunRequest{RunID: run.RunID, UserID: "user-1", SteerID: "steer_a", Text: "Add error bars."})
	if err != nil {
		t.Fatalf("SteerRun retry: %v", err)
	}
	if retry.MessageID != first.MessageID {
		t.Fatalf("retry minted a new message id %q != %q — duplicate transcript rows", retry.MessageID, first.MessageID)
	}
	messages, _ := mem.ListThreadMessages(ctx, run.ThreadID)
	if len(messages) != 2 {
		t.Fatalf("transcript has %d messages, want 2 (original + one steer)", len(messages))
	}
	if kinds := steerEventKinds(t, mem, run.RunID); len(kinds) != 1 {
		t.Fatalf("steer events = %v, want exactly one steer.received", kinds)
	}
}

func TestSteerRunRejectsEmptyAndOversized(t *testing.T) {
	t.Parallel()
	ctx, _, service, run := newSteeringFixture(t)

	if _, err := service.SteerRun(ctx, SteerRunRequest{RunID: run.RunID, UserID: "user-1", Text: "   "}); !errors.Is(err, ErrInvalidSteer) {
		t.Fatalf("empty text error = %v, want ErrInvalidSteer", err)
	}
	huge := make([]byte, maxSteerTextLength+1)
	for i := range huge {
		huge[i] = 'a'
	}
	if _, err := service.SteerRun(ctx, SteerRunRequest{RunID: run.RunID, UserID: "user-1", Text: string(huge)}); !errors.Is(err, ErrInvalidSteer) {
		t.Fatalf("oversized text error = %v, want ErrInvalidSteer", err)
	}
}

func TestAckFlipsOnceAndEmitsOneEvent(t *testing.T) {
	t.Parallel()
	ctx, mem, service, run := newSteeringFixture(t)

	record, err := service.SteerRun(ctx, SteerRunRequest{RunID: run.RunID, UserID: "user-1", SteerID: "steer_a", Text: "Label the axes."})
	if err != nil {
		t.Fatalf("SteerRun: %v", err)
	}
	applied, err := service.AckRunSteerMessage(ctx, AckRunSteerRequest{RunID: run.RunID, SteerID: record.SteerID, WorkerID: "worker-1"})
	if err != nil {
		t.Fatalf("Ack: %v", err)
	}
	if applied.Status != domain.RunSteerStatusApplied || applied.AppliedAt == nil {
		t.Fatalf("applied record = %+v", applied)
	}
	// Lost-HTTP-response retry: the worker acks again; nothing changes and
	// no second event appends.
	again, err := service.AckRunSteerMessage(ctx, AckRunSteerRequest{RunID: run.RunID, SteerID: record.SteerID, WorkerID: "worker-1"})
	if err != nil {
		t.Fatalf("Ack retry: %v", err)
	}
	if !again.AppliedAt.Equal(*applied.AppliedAt) {
		t.Fatalf("retry moved applied_at %v -> %v", applied.AppliedAt, again.AppliedAt)
	}
	kinds := steerEventKinds(t, mem, run.RunID)
	if len(kinds) != 2 || kinds[1] != "steer.applied" {
		t.Fatalf("steer events = %v, want [steer.received steer.applied]", kinds)
	}
}

func TestBarrierClosesAcceptanceAndReturnsPending(t *testing.T) {
	t.Parallel()
	ctx, _, service, run := newSteeringFixture(t)

	record, err := service.SteerRun(ctx, SteerRunRequest{RunID: run.RunID, UserID: "user-1", SteerID: "steer_a", Text: "One more thing."})
	if err != nil {
		t.Fatalf("SteerRun: %v", err)
	}
	pending, err := service.CloseRunSteerBarrier(ctx, run.RunID)
	if err != nil {
		t.Fatalf("CloseRunSteerBarrier: %v", err)
	}
	if len(pending) != 1 || pending[0].SteerID != record.SteerID {
		t.Fatalf("pending = %+v, want the accepted steer", pending)
	}
	// Post-barrier steers are rejected — the client falls back to Phase 0.
	if _, err := service.SteerRun(ctx, SteerRunRequest{RunID: run.RunID, UserID: "user-1", Text: "Too late."}); !errors.Is(err, store.ErrSteeringClosed) {
		t.Fatalf("post-barrier steer error = %v, want ErrSteeringClosed", err)
	}
	// Idempotent: a second barrier call returns the same pending set.
	pendingAgain, err := service.CloseRunSteerBarrier(ctx, run.RunID)
	if err != nil {
		t.Fatalf("CloseRunSteerBarrier again: %v", err)
	}
	if len(pendingAgain) != 1 {
		t.Fatalf("second barrier pending = %+v", pendingAgain)
	}
}

func TestSteerRejectedOnTerminalRun(t *testing.T) {
	t.Parallel()
	ctx, _, service, run := newSteeringFixture(t)

	if _, err := service.CancelRun(ctx, CancelRunRequest{RunID: run.RunID, Reason: "user stop"}); err != nil {
		t.Fatalf("CancelRun: %v", err)
	}
	if _, err := service.SteerRun(ctx, SteerRunRequest{RunID: run.RunID, UserID: "user-1", Text: "Still there?"}); !errors.Is(err, store.ErrSteeringClosed) {
		t.Fatalf("terminal steer error = %v, want ErrSteeringClosed", err)
	}
}

func TestTerminalSweepMarksPendingSteersMissed(t *testing.T) {
	t.Parallel()
	ctx, mem, service, run := newSteeringFixture(t)

	record, err := service.SteerRun(ctx, SteerRunRequest{RunID: run.RunID, UserID: "user-1", SteerID: "steer_a", Text: "Never seen."})
	if err != nil {
		t.Fatalf("SteerRun: %v", err)
	}
	// The run is canceled with the steer still pending — the sweep must mark
	// it missed and say so on the event stream, never silently drop it.
	if _, err := service.CancelRun(ctx, CancelRunRequest{RunID: run.RunID, Reason: "user stop"}); err != nil {
		t.Fatalf("CancelRun: %v", err)
	}
	records, err := service.ListRunSteerMessages(ctx, run.RunID)
	if err != nil {
		t.Fatalf("ListRunSteerMessages: %v", err)
	}
	if len(records) != 1 || records[0].Status != domain.RunSteerStatusMissed {
		t.Fatalf("records = %+v, want [missed]", records)
	}
	kinds := steerEventKinds(t, mem, run.RunID)
	if len(kinds) != 2 || kinds[1] != "steer.missed" {
		t.Fatalf("steer events = %v, want [steer.received steer.missed]", kinds)
	}
	_ = record
}

func TestRequeueReopensSteerBarrier(t *testing.T) {
	t.Parallel()
	ctx, _, service, run := newSteeringFixture(t)

	if _, err := service.CloseRunSteerBarrier(ctx, run.RunID); err != nil {
		t.Fatalf("CloseRunSteerBarrier: %v", err)
	}
	if _, err := service.SteerRun(ctx, SteerRunRequest{RunID: run.RunID, UserID: "user-1", Text: "Blocked."}); !errors.Is(err, store.ErrSteeringClosed) {
		t.Fatalf("expected barrier to reject, got %v", err)
	}
	// A crashed finalization gets requeued: the fresh attempt must accept
	// steers again.
	if _, err := service.RequeueRun(ctx, RequeueRunRequest{RunID: run.RunID, Reason: "test requeue"}); err != nil {
		t.Fatalf("RequeueRun: %v", err)
	}
	if _, err := service.SteerRun(ctx, SteerRunRequest{RunID: run.RunID, UserID: "user-1", SteerID: "steer_b", Text: "Back open."}); err != nil {
		t.Fatalf("post-requeue steer: %v", err)
	}
}

func TestSteerEventsDoNotCountAsWorkerProgress(t *testing.T) {
	t.Parallel()
	// A user steering a dead worker's run must not veto recovery requeue;
	// a genuine worker ack may.
	if isWorkerProgressRunEvent("steer.received") {
		t.Fatal("steer.received counts as worker progress — a user could keep a dead run alive by steering it")
	}
	if isWorkerProgressRunEvent("steer.missed") {
		t.Fatal("steer.missed counts as worker progress")
	}
	if !isWorkerProgressRunEvent("steer.applied") {
		t.Fatal("steer.applied is worker-originated and should count as progress")
	}
}
