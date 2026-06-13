package runcontrol

import (
	"context"
	"fmt"
	"os"
	"sync/atomic"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
	"github.com/jackc/pgx/v5/pgxpool"
)

var ingestBenchCounter atomic.Int64

func ingestBenchID(prefix string) string {
	return fmt.Sprintf("%s-ingestbench-%d-%d", prefix, time.Now().UnixNano(), ingestBenchCounter.Add(1))
}

// End-to-end cost of one event through Service.IngestRunEvent against real
// Postgres, exactly as the NATS consumer drives it in production. The consumer
// is a single serial callback per replica, so events/sec ~= 1e9 / (ns/op).
func BenchmarkIngestRunEventPostgres(b *testing.B) {
	dsn := os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL")
	if dsn == "" {
		b.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		b.Fatalf("pgxpool.New: %v", err)
	}
	b.Cleanup(pool.Close)
	if err := store.ApplyPostgresSchema(ctx, pool); err != nil {
		b.Fatalf("ApplyPostgresSchema: %v", err)
	}
	pgStore := store.NewPostgresStore(pool)
	service := NewService(pgStore, eventbus.NewMemoryBus())

	userID := ingestBenchID("user")
	thread, err := pgStore.CreateThread(ctx, domain.CreateThreadInput{UserID: userID, Title: "ingest bench"})
	if err != nil {
		b.Fatalf("CreateThread: %v", err)
	}
	run, err := pgStore.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   userID,
		Goal:     "ingest bench",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "bench"}},
	})
	if err != nil {
		b.Fatalf("CreateRun: %v", err)
	}

	payload := domain.JSONMap{
		"delta": "some streamed token text that resembles a real model delta payload",
		"node":  "agent",
	}

	b.Run("log-delta-with-event-id", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
				EventID:   ingestBenchID("evt"),
				RunID:     run.RunID,
				ThreadID:  run.ThreadID,
				EventKind: "log.delta",
				Message:   "bench delta",
				Payload:   payload,
			})
			if err != nil {
				b.Fatalf("IngestRunEvent: %v", err)
			}
		}
	})

	b.Run("heartbeat-with-event-id", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
				EventID:   ingestBenchID("evt"),
				RunID:     run.RunID,
				ThreadID:  run.ThreadID,
				EventKind: "run.heartbeat",
				Message:   "bench heartbeat",
				Payload:   payload,
			})
			if err != nil {
				b.Fatalf("IngestRunEvent: %v", err)
			}
		}
	})
}
