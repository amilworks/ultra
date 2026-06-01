package eventbus

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/nats-io/nats.go"
)

func TestJobJSONMatchesPythonWorkerEnvelope(t *testing.T) {
	t.Parallel()
	payload, err := json.Marshal(Job{
		RunID:            "run-1",
		ThreadID:         "thread-1",
		UserID:           "user-1",
		Goal:             "Analyze the paper.",
		KnowledgeContext: domain.JSONMap{"paper_id": "arxiv:2509.26626"},
		SelectionContext: domain.JSONMap{"source": "chat"},
		ReasoningMode:    "deep",
		Benchmark:        domain.JSONMap{"suite": "paper-review"},
	})
	if err != nil {
		t.Fatalf("Marshal Job: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("Unmarshal Job: %v", err)
	}
	for _, key := range []string{
		"run_id",
		"thread_id",
		"user_id",
		"goal",
		"knowledge_context",
		"selection_context",
		"reasoning_mode",
		"benchmark",
	} {
		if _, ok := decoded[key]; !ok {
			t.Fatalf("job JSON missing %q in %s", key, payload)
		}
	}
}

func TestRunEventMessageDispositionAcksMalformedPayload(t *testing.T) {
	t.Parallel()
	calls := 0

	disposition := runEventMessageDisposition(context.Background(), []byte("not-json"), func(context.Context, domain.AppendRunEventInput) error {
		calls++
		return nil
	})

	if disposition != runEventMessageAck {
		t.Fatalf("disposition = %v, want ack/drop for malformed payload", disposition)
	}
	if calls != 0 {
		t.Fatalf("handler calls = %d, want 0 for malformed payload", calls)
	}
}

func TestRunEventMessageDispositionNaksHandlerError(t *testing.T) {
	t.Parallel()

	payload, err := json.Marshal(domain.AppendRunEventInput{
		EventID:   "evt-run-1-1",
		RunID:     "run-1",
		ThreadID:  "thread-1",
		EventKind: "run.completed",
	})
	if err != nil {
		t.Fatalf("Marshal payload: %v", err)
	}
	disposition := runEventMessageDisposition(context.Background(), payload, func(context.Context, domain.AppendRunEventInput) error {
		return context.DeadlineExceeded
	})

	if disposition != runEventMessageNak {
		t.Fatalf("disposition = %v, want nak for handler error", disposition)
	}
}

func TestRunEventMessageDispositionAcksHandledPayload(t *testing.T) {
	t.Parallel()

	payload, err := json.Marshal(domain.AppendRunEventInput{
		EventID:   "evt-run-1-1",
		RunID:     "run-1",
		ThreadID:  "thread-1",
		EventKind: "run.completed",
	})
	if err != nil {
		t.Fatalf("Marshal payload: %v", err)
	}
	disposition := runEventMessageDisposition(context.Background(), payload, func(context.Context, domain.AppendRunEventInput) error {
		return nil
	})

	if disposition != runEventMessageAck {
		t.Fatalf("disposition = %v, want ack for handled payload", disposition)
	}
}

func TestRunEventSubscribeSubjectUsesPushConsumerDeliverSubject(t *testing.T) {
	t.Parallel()

	cfg := NATSConfig{EventsSubject: "ultra.test.events"}
	got := runEventSubscribeSubject(cfg, "ultra-control-event-ingest")
	want := "ultra.test.events.deliver.ultra-control-event-ingest"

	if got != want {
		t.Fatalf("runEventSubscribeSubject() = %q, want %q", got, want)
	}
}

func TestRunEventConsumerReplaysExistingEventsWhenCreatedFresh(t *testing.T) {
	t.Parallel()

	cfg := NATSConfig{EventsSubject: "ultra.test.events"}
	got := runEventConsumerConfig(cfg, "ultra-control-event-ingest")

	if got.DeliverPolicy != nats.DeliverAllPolicy {
		t.Fatalf("DeliverPolicy = %v, want DeliverAllPolicy so a fresh control-plane event consumer recovers events published while Go was offline", got.DeliverPolicy)
	}
}

func TestNATSMessageIDForJobUsesRunID(t *testing.T) {
	t.Parallel()

	got := natsMessageIDForJob(Job{RunID: "run-abc", WorkflowKind: "deepagents"})

	if got != "job:run-abc" {
		t.Fatalf("natsMessageIDForJob() = %q, want job:run-abc", got)
	}
}

func TestNATSMessageIDForExplicitRequeueUsesDispatchID(t *testing.T) {
	t.Parallel()

	got := natsMessageIDForJob(Job{RunID: "run-abc", DispatchID: "dispatch-2"})

	if got != "job:run-abc:dispatch-2" {
		t.Fatalf("natsMessageIDForJob() = %q, want job:run-abc:dispatch-2", got)
	}
}

func TestNATSMessageIDForRunEventUsesEventID(t *testing.T) {
	t.Parallel()

	got := natsMessageIDForRunEvent(domain.RunEventRecord{EventID: "evt-run-abc-1", RunID: "run-abc"})

	if got != "event:evt-run-abc-1" {
		t.Fatalf("natsMessageIDForRunEvent() = %q, want event:evt-run-abc-1", got)
	}
}

func TestNATSMessageIDForCancelUsesRunIDAndReason(t *testing.T) {
	t.Parallel()

	got := natsMessageIDForCancel(CancelSignal{RunID: "run-abc", Reason: "user"})

	if got != "cancel:run-abc:user" {
		t.Fatalf("natsMessageIDForCancel() = %q, want cancel:run-abc:user", got)
	}
}

func TestNATSStreamConfigUsesLongDuplicateWindow(t *testing.T) {
	t.Parallel()

	stream := natsStreamConfig("ULTRA_TEST", []string{"ultra.test.jobs", "ultra.test.events"})

	if stream.Duplicates < 24*time.Hour {
		t.Fatalf("duplicate window = %s, want at least 24h for long-run publish retries", stream.Duplicates)
	}
}

func TestNATSBusPublishesJobAndRunEvent(t *testing.T) {
	url := os.Getenv("ULTRA_CONTROL_TEST_NATS_URL")
	if url == "" {
		t.Skip("ULTRA_CONTROL_TEST_NATS_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	suffix := fmt.Sprintf("%d", time.Now().UnixNano())
	stream := "ULTRA_TEST_PUBLISH_" + suffix
	jobsSubject := "ultra.test." + suffix + ".jobs"
	eventsSubject := "ultra.test." + suffix + ".events"
	cancelSubject := "ultra.test." + suffix + ".cancel"
	bus, err := NewNATSBus(ctx, NATSConfig{
		URL:           url,
		Stream:        stream,
		JobsSubject:   jobsSubject,
		EventsSubject: eventsSubject,
		CancelSubject: cancelSubject,
	})
	if err != nil {
		t.Fatalf("NewNATSBus: %v", err)
	}
	defer bus.Close()
	defer func() {
		_ = bus.js.DeleteStream(stream)
	}()

	if err := bus.PublishJob(ctx, Job{RunID: "run-1", ThreadID: "thread-1", UserID: "user-1", Goal: "test"}); err != nil {
		t.Fatalf("PublishJob: %v", err)
	}
	if err := bus.PublishRunEvent(ctx, domain.RunEventRecord{RunID: "run-1", EventKind: "run.accepted", Payload: domain.JSONMap{"ok": true}}); err != nil {
		t.Fatalf("PublishRunEvent: %v", err)
	}
}

func TestNATSBusDeduplicatesPublishRetriesByDeterministicMessageID(t *testing.T) {
	url := os.Getenv("ULTRA_CONTROL_TEST_NATS_URL")
	if url == "" {
		t.Skip("ULTRA_CONTROL_TEST_NATS_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	suffix := fmt.Sprintf("%d", time.Now().UnixNano())
	stream := "ULTRA_TEST_DEDUP_" + suffix
	jobsSubject := "ultra.test." + suffix + ".jobs"
	eventsSubject := "ultra.test." + suffix + ".events"
	cancelSubject := "ultra.test." + suffix + ".cancel"
	bus, err := NewNATSBus(ctx, NATSConfig{
		URL:           url,
		Stream:        stream,
		JobsSubject:   jobsSubject,
		EventsSubject: eventsSubject,
		CancelSubject: cancelSubject,
	})
	if err != nil {
		t.Fatalf("NewNATSBus: %v", err)
	}
	defer bus.Close()
	defer func() {
		_ = bus.js.DeleteStream(stream)
	}()

	job := Job{RunID: "run-dedup", ThreadID: "thread-dedup", UserID: "user-1", Goal: "dedupe job"}
	if err := bus.PublishJob(ctx, job); err != nil {
		t.Fatalf("PublishJob first: %v", err)
	}
	if err := bus.PublishJob(ctx, job); err != nil {
		t.Fatalf("PublishJob retry: %v", err)
	}
	assertNATSStreamMessages(t, ctx, bus, 1)

	event := domain.RunEventRecord{EventID: "evt-run-dedup-1", RunID: "run-dedup", EventKind: "run.started"}
	if err := bus.PublishRunEvent(ctx, event); err != nil {
		t.Fatalf("PublishRunEvent first: %v", err)
	}
	if err := bus.PublishRunEvent(ctx, event); err != nil {
		t.Fatalf("PublishRunEvent retry: %v", err)
	}
	assertNATSStreamMessages(t, ctx, bus, 2)

	cancelSignal := CancelSignal{RunID: "run-dedup", Reason: "user"}
	if err := bus.PublishCancel(ctx, cancelSignal); err != nil {
		t.Fatalf("PublishCancel first: %v", err)
	}
	if err := bus.PublishCancel(ctx, cancelSignal); err != nil {
		t.Fatalf("PublishCancel retry: %v", err)
	}
	assertNATSStreamMessages(t, ctx, bus, 3)
}

func TestNATSRunEventConsumerSurvivesSubscriberShutdown(t *testing.T) {
	url := os.Getenv("ULTRA_CONTROL_TEST_NATS_URL")
	if url == "" {
		t.Skip("ULTRA_CONTROL_TEST_NATS_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	suffix := fmt.Sprintf("%d", time.Now().UnixNano())
	stream := "ULTRA_TEST_EVENT_CONSUMER_" + suffix
	jobsSubject := "ultra.test." + suffix + ".jobs"
	eventsSubject := "ultra.test." + suffix + ".events"
	cancelSubject := "ultra.test." + suffix + ".cancel"
	consumer := "ultra-test-event-consumer-" + suffix
	bus, err := NewNATSBus(ctx, NATSConfig{
		URL:           url,
		Stream:        stream,
		JobsSubject:   jobsSubject,
		EventsSubject: eventsSubject,
		CancelSubject: cancelSubject,
		EventConsumer: consumer,
	})
	if err != nil {
		t.Fatalf("NewNATSBus: %v", err)
	}
	defer bus.Close()
	defer func() {
		_ = bus.js.DeleteStream(stream)
	}()

	subCtx, stopSub := context.WithCancel(ctx)
	handled := make(chan struct{}, 1)
	if err := bus.SubscribeAllRunEvents(subCtx, func(context.Context, domain.AppendRunEventInput) error {
		handled <- struct{}{}
		return nil
	}); err != nil {
		t.Fatalf("SubscribeAllRunEvents: %v", err)
	}
	if err := bus.PublishRunEvent(ctx, domain.RunEventRecord{
		EventID:   "evt-run-1-started",
		RunID:     "run-1",
		ThreadID:  "thread-1",
		EventKind: "run.started",
	}); err != nil {
		t.Fatalf("PublishRunEvent: %v", err)
	}
	select {
	case <-handled:
	case <-ctx.Done():
		t.Fatalf("handler was not called before timeout")
	}

	stopSub()
	time.Sleep(150 * time.Millisecond)
	if _, err := bus.js.ConsumerInfo(stream, consumer, nats.Context(ctx)); err != nil {
		t.Fatalf("event ingest consumer was removed after subscriber shutdown: %v", err)
	}
}

func TestNATSRunEventConsumerReconcilesExistingUnsafeDefaults(t *testing.T) {
	url := os.Getenv("ULTRA_CONTROL_TEST_NATS_URL")
	if url == "" {
		t.Skip("ULTRA_CONTROL_TEST_NATS_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	suffix := fmt.Sprintf("%d", time.Now().UnixNano())
	stream := "ULTRA_TEST_EVENT_RECONCILE_" + suffix
	jobsSubject := "ultra.test." + suffix + ".jobs"
	eventsSubject := "ultra.test." + suffix + ".events"
	cancelSubject := "ultra.test." + suffix + ".cancel"
	consumer := "ultra-test-event-reconcile-" + suffix
	bus, err := NewNATSBus(ctx, NATSConfig{
		URL:           url,
		Stream:        stream,
		JobsSubject:   jobsSubject,
		EventsSubject: eventsSubject,
		CancelSubject: cancelSubject,
		EventConsumer: consumer,
	})
	if err != nil {
		t.Fatalf("NewNATSBus: %v", err)
	}
	defer bus.Close()
	defer func() {
		_ = bus.js.DeleteStream(stream)
	}()

	_, err = bus.js.AddConsumer(stream, &nats.ConsumerConfig{
		Durable:        consumer,
		DeliverSubject: eventsSubject + ".old-deliver",
		DeliverPolicy:  nats.DeliverAllPolicy,
		AckPolicy:      nats.AckExplicitPolicy,
		AckWait:        time.Second,
		MaxDeliver:     1,
		FilterSubject:  eventsSubject,
		ReplayPolicy:   nats.ReplayInstantPolicy,
		MaxAckPending:  1,
	}, nats.Context(ctx))
	if err != nil {
		t.Fatalf("AddConsumer old config: %v", err)
	}

	subCtx, stopSub := context.WithCancel(ctx)
	defer stopSub()
	if err := bus.SubscribeAllRunEvents(subCtx, func(context.Context, domain.AppendRunEventInput) error {
		return nil
	}); err != nil {
		t.Fatalf("SubscribeAllRunEvents: %v", err)
	}
	info, err := bus.js.ConsumerInfo(stream, consumer, nats.Context(ctx))
	if err != nil {
		t.Fatalf("ConsumerInfo: %v", err)
	}
	if !runEventConsumerConfigMatches(info.Config, runEventConsumerConfig(bus.cfg, consumer)) {
		t.Fatalf("consumer config was not reconciled: %+v", info.Config)
	}
}

func TestNATSRunEventConsumerSupportsMultipleControlPlaneSubscribers(t *testing.T) {
	url := os.Getenv("ULTRA_CONTROL_TEST_NATS_URL")
	if url == "" {
		t.Skip("ULTRA_CONTROL_TEST_NATS_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	suffix := fmt.Sprintf("%d", time.Now().UnixNano())
	stream := "ULTRA_TEST_EVENT_MULTI_" + suffix
	jobsSubject := "ultra.test." + suffix + ".jobs"
	eventsSubject := "ultra.test." + suffix + ".events"
	cancelSubject := "ultra.test." + suffix + ".cancel"
	consumer := "ultra-test-event-multi-" + suffix
	cfg := NATSConfig{
		URL:           url,
		Stream:        stream,
		JobsSubject:   jobsSubject,
		EventsSubject: eventsSubject,
		CancelSubject: cancelSubject,
		EventConsumer: consumer,
	}
	busA, err := NewNATSBus(ctx, cfg)
	if err != nil {
		t.Fatalf("NewNATSBus A: %v", err)
	}
	defer busA.Close()
	defer func() {
		_ = busA.js.DeleteStream(stream)
	}()
	busB, err := NewNATSBus(ctx, cfg)
	if err != nil {
		t.Fatalf("NewNATSBus B: %v", err)
	}
	defer busB.Close()

	subCtx, stopSub := context.WithCancel(ctx)
	defer stopSub()
	handled := make(chan string, 2)
	if err := busA.SubscribeAllRunEvents(subCtx, func(context.Context, domain.AppendRunEventInput) error {
		handled <- "a"
		return nil
	}); err != nil {
		t.Fatalf("SubscribeAllRunEvents A: %v", err)
	}
	if err := busB.SubscribeAllRunEvents(subCtx, func(context.Context, domain.AppendRunEventInput) error {
		handled <- "b"
		return nil
	}); err != nil {
		t.Fatalf("SubscribeAllRunEvents B: %v", err)
	}
	if err := busA.PublishRunEvent(ctx, domain.RunEventRecord{
		EventID:   "evt-run-horizontal-started",
		RunID:     "run-horizontal",
		ThreadID:  "thread-horizontal",
		EventKind: "run.started",
	}); err != nil {
		t.Fatalf("PublishRunEvent: %v", err)
	}

	select {
	case <-handled:
	case <-ctx.Done():
		t.Fatalf("neither control-plane subscriber handled the event before timeout")
	}
	select {
	case second := <-handled:
		t.Fatalf("event was delivered to more than one control-plane subscriber; second handler = %s", second)
	case <-time.After(150 * time.Millisecond):
	}
}

func assertNATSStreamMessages(t *testing.T, ctx context.Context, bus *NATSBus, want uint64) {
	t.Helper()
	info, err := bus.js.StreamInfo(bus.cfg.Stream, nats.Context(ctx))
	if err != nil {
		t.Fatalf("StreamInfo: %v", err)
	}
	if info.State.Msgs != want {
		t.Fatalf("stream messages = %d, want %d", info.State.Msgs, want)
	}
}

func TestMemoryBusSubscribersReceiveIndependentRunEventStreams(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	bus := NewMemoryBus()
	subA, unsubscribeA := bus.SubscribeRunEvents(ctx, "run-1")
	defer unsubscribeA()
	subB, unsubscribeB := bus.SubscribeRunEvents(ctx, "run-1")
	defer unsubscribeB()

	event := domain.RunEventRecord{RunID: "run-1", EventKind: "run.progress"}
	if err := bus.PublishRunEvent(ctx, event); err != nil {
		t.Fatalf("PublishRunEvent: %v", err)
	}
	if err := bus.PublishRunEvent(ctx, domain.RunEventRecord{RunID: "run-other", EventKind: "ignored"}); err != nil {
		t.Fatalf("PublishRunEvent other: %v", err)
	}

	for name, ch := range map[string]<-chan domain.RunEventRecord{"subscriber A": subA, "subscriber B": subB} {
		select {
		case got := <-ch:
			if got.RunID != "run-1" || got.EventKind != "run.progress" {
				t.Fatalf("%s got event = %+v, want run-1 progress", name, got)
			}
		case <-time.After(time.Second):
			t.Fatalf("%s did not receive fanout event", name)
		}
		select {
		case got := <-ch:
			t.Fatalf("%s received unrelated event: %+v", name, got)
		case <-time.After(20 * time.Millisecond):
		}
	}
}

func TestMemoryBusPublishRunEventDoesNotBlockWhenEventsChannelIsUndrained(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	bus := NewMemoryBus()
	for i := 0; i < 1100; i++ {
		if err := bus.PublishRunEvent(ctx, domain.RunEventRecord{
			RunID:     "run-long",
			EventKind: "message.delta",
			Sequence:  int64(i + 1),
		}); err != nil {
			t.Fatalf("PublishRunEvent %d returned %v; long runs must not block when Events() is undrained", i+1, err)
		}
	}
}

func TestMemoryBusPublishJobBuffersThousandJobBurstWhenWorkerIsStarting(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	bus := NewMemoryBus()
	for i := 0; i < 1000; i++ {
		if err := bus.PublishJob(ctx, Job{
			RunID:    "run-burst",
			ThreadID: "thread-burst",
			UserID:   "user-1",
			Goal:     "burst job",
		}); err != nil {
			t.Fatalf("PublishJob %d returned %v; burst submits should survive worker startup lag", i+1, err)
		}
	}
}

func TestMemoryBusPublishCancelBuffersThousandCancelBurstWhenWorkerIsStarting(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	bus := NewMemoryBus()
	for i := 0; i < 1000; i++ {
		if err := bus.PublishCancel(ctx, CancelSignal{
			RunID:  "run-burst",
			UserID: "user-1",
			Reason: "operator soak",
		}); err != nil {
			t.Fatalf("PublishCancel %d returned %v; burst cancels should survive worker startup lag", i+1, err)
		}
	}
}

func TestMemoryBusSubscriberBuffersThousandEventBurst(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	bus := NewMemoryBus()
	events, unsubscribe := bus.SubscribeRunEvents(ctx, "run-long")
	defer unsubscribe()

	for i := 0; i < 1000; i++ {
		if err := bus.PublishRunEvent(ctx, domain.RunEventRecord{
			RunID:     "run-long",
			EventKind: "message.delta",
			Sequence:  int64(i + 1),
		}); err != nil {
			t.Fatalf("PublishRunEvent %d: %v", i+1, err)
		}
	}

	for i := 0; i < 1000; i++ {
		select {
		case event := <-events:
			if event.Sequence != int64(i+1) {
				t.Fatalf("event %d sequence = %d, want %d", i+1, event.Sequence, i+1)
			}
		case <-ctx.Done():
			t.Fatalf("received %d/1000 burst events before timeout", i)
		}
	}
}
