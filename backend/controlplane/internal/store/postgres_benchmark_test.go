package store

import (
	"context"
	"fmt"
	"os"
	"sync/atomic"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/jackc/pgx/v5/pgxpool"
)

var benchIDCounter atomic.Int64

func benchID(prefix string) string {
	return fmt.Sprintf("%s-bench-%d-%d", prefix, time.Now().UnixNano(), benchIDCounter.Add(1))
}

func benchPostgresStore(b *testing.B) *PostgresStore {
	b.Helper()
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		b.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	pool, err := pgxpool.New(context.Background(), dsn)
	if err != nil {
		b.Fatalf("pgxpool.New: %v", err)
	}
	b.Cleanup(pool.Close)
	if err := ApplyPostgresSchema(context.Background(), pool); err != nil {
		b.Fatalf("ApplyPostgresSchema: %v", err)
	}
	return NewPostgresStore(pool)
}

func benchCreateRun(b *testing.B, s *PostgresStore, userID string) domain.RunRecord {
	b.Helper()
	ctx := context.Background()
	thread, err := s.CreateThread(ctx, domain.CreateThreadInput{UserID: userID, Title: "bench thread"})
	if err != nil {
		b.Fatalf("CreateThread: %v", err)
	}
	run, err := s.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   userID,
		Goal:     "bench run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "bench"}},
	})
	if err != nil {
		b.Fatalf("CreateRun: %v", err)
	}
	return run
}

var benchEventPayload = domain.JSONMap{
	"delta":      "some streamed token text that resembles a real model delta payload",
	"node":       "agent",
	"tokens":     float64(17),
	"checkpoint": "ckpt_0123456789",
}

// Serial appends to a single run: mirrors the production NATS event-ingest
// consumer, which processes events one at a time per control-plane replica.
func BenchmarkPostgresAppendRunEventSerial(b *testing.B) {
	s := benchPostgresStore(b)
	run := benchCreateRun(b, s, benchID("user"))
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := s.AppendRunEvent(ctx, domain.AppendRunEventInput{
			RunID:     run.RunID,
			ThreadID:  run.ThreadID,
			EventKind: "log.delta",
			Message:   "bench event",
			Payload:   benchEventPayload,
		})
		if err != nil {
			b.Fatalf("AppendRunEvent: %v", err)
		}
	}
}

// Parallel appends across distinct runs: the per-run advisory lock should not
// serialize unrelated runs, so this shows the cross-run scaling headroom.
func BenchmarkPostgresAppendRunEventParallelRuns(b *testing.B) {
	s := benchPostgresStore(b)
	ctx := context.Background()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		b.StopTimer()
		run := benchCreateRun(b, s, benchID("user"))
		b.StartTimer()
		for pb.Next() {
			_, err := s.AppendRunEvent(ctx, domain.AppendRunEventInput{
				RunID:     run.RunID,
				ThreadID:  run.ThreadID,
				EventKind: "log.delta",
				Message:   "bench event",
				Payload:   benchEventPayload,
			})
			if err != nil {
				b.Errorf("AppendRunEvent: %v", err)
				return
			}
		}
	})
}

// The SSE catch-up query as issued once per second per connected client (and
// once per received bus event) when the stream is idle: returns zero rows.
func BenchmarkPostgresListRunEventsAfterIdleTail(b *testing.B) {
	s := benchPostgresStore(b)
	run := benchCreateRun(b, s, benchID("user"))
	ctx := context.Background()
	var lastSeq int64
	for i := 0; i < 200; i++ {
		event, err := s.AppendRunEvent(ctx, domain.AppendRunEventInput{
			RunID:     run.RunID,
			ThreadID:  run.ThreadID,
			EventKind: "log.delta",
			Message:   "seed event",
			Payload:   benchEventPayload,
		})
		if err != nil {
			b.Fatalf("AppendRunEvent: %v", err)
		}
		lastSeq = event.Sequence
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		events, err := s.ListRunEventsAfter(ctx, run.RunID, lastSeq, 500)
		if err != nil {
			b.Fatalf("ListRunEventsAfter: %v", err)
		}
		if len(events) != 0 {
			b.Fatalf("expected idle tail, got %d events", len(events))
		}
	}
}

// Full 500-event replay page, as served on SSE connect and REST event lists.
func BenchmarkPostgresListRunEventsReplayPage(b *testing.B) {
	s := benchPostgresStore(b)
	run := benchCreateRun(b, s, benchID("user"))
	ctx := context.Background()
	for i := 0; i < 500; i++ {
		if _, err := s.AppendRunEvent(ctx, domain.AppendRunEventInput{
			RunID:     run.RunID,
			ThreadID:  run.ThreadID,
			EventKind: "log.delta",
			Message:   "seed event",
			Payload:   benchEventPayload,
		}); err != nil {
			b.Fatalf("AppendRunEvent: %v", err)
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		events, err := s.ListRunEventsAfter(ctx, run.RunID, 0, 500)
		if err != nil {
			b.Fatalf("ListRunEventsAfter: %v", err)
		}
		if len(events) != 500 {
			b.Fatalf("expected 500 events, got %d", len(events))
		}
	}
}

// ListThreadMessages backs every thread view and every CreateRun call.
// control_thread_messages has no index on thread_id, so this degrades as the
// table grows; the seeded background rows make the planner's choice visible.
func BenchmarkPostgresListThreadMessages(b *testing.B) {
	s := benchPostgresStore(b)
	ctx := context.Background()
	userID := benchID("user")

	// Background noise: messages spread across other threads.
	for t := 0; t < 50; t++ {
		thread, err := s.CreateThread(ctx, domain.CreateThreadInput{UserID: userID, Title: "noise"})
		if err != nil {
			b.Fatalf("CreateThread: %v", err)
		}
		for m := 0; m < 40; m++ {
			if _, err := s.AppendThreadMessage(ctx, domain.ThreadMessage{
				ThreadID: thread.ThreadID,
				Role:     "user",
				Content:  "noise message payload text",
			}); err != nil {
				b.Fatalf("AppendThreadMessage: %v", err)
			}
		}
	}
	target, err := s.CreateThread(ctx, domain.CreateThreadInput{UserID: userID, Title: "target"})
	if err != nil {
		b.Fatalf("CreateThread: %v", err)
	}
	for m := 0; m < 30; m++ {
		if _, err := s.AppendThreadMessage(ctx, domain.ThreadMessage{
			ThreadID: target.ThreadID,
			Role:     "user",
			Content:  "target message payload text",
		}); err != nil {
			b.Fatalf("AppendThreadMessage: %v", err)
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		messages, err := s.ListThreadMessages(ctx, target.ThreadID)
		if err != nil {
			b.Fatalf("ListThreadMessages: %v", err)
		}
		if len(messages) < 30 {
			b.Fatalf("expected >= 30 messages, got %d", len(messages))
		}
	}
}
