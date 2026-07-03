package runcontrol

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Measures the per-event cost of IngestRunEvent WITH the source_sequence
// predecessor gate active (SourceSequence set, monotonic) versus WITHOUT it,
// against live Postgres. This isolates the cost the gate adds on top of the
// single-statement append.
func BenchmarkIngestSourceSequenceGate(b *testing.B) {
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
	pg := store.NewPostgresStore(pool)

	newRun := func() domain.RunRecord {
		uid := fmt.Sprintf("ss-bench-%d", time.Now().UnixNano())
		th, err := pg.CreateThread(ctx, domain.CreateThreadInput{UserID: uid, Title: "ss"})
		if err != nil {
			b.Fatalf("CreateThread: %v", err)
		}
		run, err := pg.CreateRun(ctx, domain.CreateRunInput{ThreadID: th.ThreadID, UserID: uid, Goal: "ss", Messages: []domain.ThreadMessage{{Role: "user", Content: "x"}}})
		if err != nil {
			b.Fatalf("CreateRun: %v", err)
		}
		return run
	}
	payload := domain.JSONMap{"delta": "streamed token text resembling a real model delta payload"}

	b.Run("with-source-sequence-gate", func(b *testing.B) {
		svc := NewService(pg, eventbus.NewMemoryBus())
		run := newRun()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			seq := int64(i + 1)
			_, err := svc.IngestRunEvent(ctx, domain.AppendRunEventInput{
				EventID:        fmt.Sprintf("evt-ss-%s-%d", run.RunID, seq),
				SourceSequence: seq,
				RunID:          run.RunID,
				ThreadID:       run.ThreadID,
				EventKind:      "message.delta",
				Message:        "delta",
				Payload:        payload,
			})
			if err != nil {
				b.Fatalf("IngestRunEvent(seq=%d): %v", seq, err)
			}
		}
	})

	b.Run("without-source-sequence", func(b *testing.B) {
		svc := NewService(pg, eventbus.NewMemoryBus())
		run := newRun()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := svc.IngestRunEvent(ctx, domain.AppendRunEventInput{
				EventID:   fmt.Sprintf("evt-noss-%s-%d", run.RunID, i+1),
				RunID:     run.RunID,
				ThreadID:  run.ThreadID,
				EventKind: "message.delta",
				Message:   "delta",
				Payload:   payload,
			})
			if err != nil {
				b.Fatalf("IngestRunEvent: %v", err)
			}
		}
	})
}
