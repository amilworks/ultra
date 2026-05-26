package eventbus

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

func TestNATSBusPublishesJobAndRunEvent(t *testing.T) {
	url := os.Getenv("ULTRA_CONTROL_TEST_NATS_URL")
	if url == "" {
		t.Skip("ULTRA_CONTROL_TEST_NATS_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	bus, err := NewNATSBus(ctx, NATSConfig{
		URL:           url,
		Stream:        "ULTRA_TEST",
		JobsSubject:   "ultra.test.jobs",
		EventsSubject: "ultra.test.events",
	})
	if err != nil {
		t.Fatalf("NewNATSBus: %v", err)
	}
	defer bus.Close()

	if err := bus.PublishJob(ctx, Job{RunID: "run-1", ThreadID: "thread-1", UserID: "user-1", Goal: "test"}); err != nil {
		t.Fatalf("PublishJob: %v", err)
	}
	if err := bus.PublishRunEvent(ctx, domain.RunEventRecord{RunID: "run-1", EventKind: "run.accepted", Payload: domain.JSONMap{"ok": true}}); err != nil {
		t.Fatalf("PublishRunEvent: %v", err)
	}
}
