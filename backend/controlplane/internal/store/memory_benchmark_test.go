package store

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

func BenchmarkMemoryRunEventReplay(b *testing.B) {
	for _, count := range []int{1000, 100000} {
		count := count
		b.Run(fmt.Sprintf("latest-page/%d-events-limit-500", count), func(b *testing.B) {
			mem, runID, userID := benchmarkMemoryRunWithEvents(b, count)
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				events, err := mem.ListRunEventsForUser(context.Background(), runID, userID, 500)
				if err != nil {
					b.Fatal(err)
				}
				if len(events) != 500 {
					b.Fatalf("events = %d, want 500", len(events))
				}
			}
		})

		b.Run(fmt.Sprintf("after-start/%d-events-limit-500", count), func(b *testing.B) {
			mem, runID, userID := benchmarkMemoryRunWithEvents(b, count)
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				events, err := mem.ListRunEventsAfterForUser(context.Background(), runID, userID, 0, 500)
				if err != nil {
					b.Fatal(err)
				}
				if len(events) != 500 || events[0].Sequence != 1 {
					b.Fatalf("events len=%d first=%d, want len=500 first=1", len(events), events[0].Sequence)
				}
			}
		})

		b.Run(fmt.Sprintf("after-tail/%d-events-limit-500", count), func(b *testing.B) {
			mem, runID, userID := benchmarkMemoryRunWithEvents(b, count)
			after := int64(count - 500)
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				events, err := mem.ListRunEventsAfterForUser(context.Background(), runID, userID, after, 500)
				if err != nil {
					b.Fatal(err)
				}
				if len(events) != 500 || events[0].Sequence != after+1 {
					b.Fatalf("events len=%d first=%d, want len=500 first=%d", len(events), events[0].Sequence, after+1)
				}
			}
		})
	}
}

func benchmarkMemoryRunWithEvents(b *testing.B, count int) (*MemoryStore, string, string) {
	b.Helper()
	mem := NewMemoryStore()
	ctx := context.Background()
	userID := "bench-user"
	thread, err := mem.CreateThread(ctx, domain.CreateThreadInput{
		UserID: userID,
		Title:  "Benchmark thread",
	})
	if err != nil {
		b.Fatalf("create benchmark thread: %v", err)
	}
	run, err := mem.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   userID,
		Goal:     "Benchmark event replay",
	})
	if err != nil {
		b.Fatalf("create benchmark run: %v", err)
	}
	baseTime := time.Unix(1_700_000_000, 0).UTC()
	events := make([]domain.RunEventRecord, count)
	for index := range events {
		sequence := int64(index + 1)
		events[index] = domain.RunEventRecord{
			EventID:   fmt.Sprintf("evt_%06d", index),
			Sequence:  sequence,
			RunID:     run.RunID,
			ThreadID:  thread.ThreadID,
			EventKind: "message.delta",
			EventType: "message.delta",
			NodeName:  "coordinator",
			TS:        baseTime.Add(time.Duration(index) * time.Millisecond),
			Message:   "benchmark event payload",
			Payload: domain.JSONMap{
				"sequence": sequence,
				"phase":    "benchmark",
			},
		}
	}
	mem.mu.Lock()
	mem.events[run.RunID] = events
	mem.mu.Unlock()
	return mem, run.RunID, userID
}
