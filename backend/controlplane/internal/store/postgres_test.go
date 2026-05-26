package store

import (
	"context"
	"os"
	"testing"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresStoreThreadRunEventArtifactFlow(t *testing.T) {
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

	store := NewPostgresStore(pool)
	thread, err := store.CreateThread(ctx, domain.CreateThreadInput{
		UserID: "pg-user",
		Title:  "Postgres flow",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   "pg-user",
		Goal:     "persist run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "persist run"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	event, err := store.AppendRunEvent(ctx, domain.AppendRunEventInput{
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.accepted",
		Message:   "accepted",
		Payload:   domain.JSONMap{"ok": true},
	})
	if err != nil {
		t.Fatalf("AppendRunEvent: %v", err)
	}
	if event.Sequence < 1 {
		t.Fatalf("event sequence = %d, want >= 1", event.Sequence)
	}
}
