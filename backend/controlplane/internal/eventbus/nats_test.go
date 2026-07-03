package eventbus

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/nats-io/nats.go"
)

type fakeNATSStreamManager struct {
	addErr      error
	infoErr     error
	updateErr   error
	info        *nats.StreamInfo
	updated     *nats.StreamConfig
	infoCalls   int
	updateCalls int
}

func (m *fakeNATSStreamManager) AddStream(*nats.StreamConfig, ...nats.JSOpt) (*nats.StreamInfo, error) {
	return nil, m.addErr
}

func (m *fakeNATSStreamManager) StreamInfo(stream string, _ ...nats.JSOpt) (*nats.StreamInfo, error) {
	m.infoCalls++
	if m.info != nil {
		return m.info, m.infoErr
	}
	return &nats.StreamInfo{Config: nats.StreamConfig{Name: stream}}, m.infoErr
}

func (m *fakeNATSStreamManager) UpdateStream(config *nats.StreamConfig, _ ...nats.JSOpt) (*nats.StreamInfo, error) {
	m.updateCalls++
	copied := *config
	copied.Subjects = append([]string(nil), config.Subjects...)
	m.updated = &copied
	return nil, m.updateErr
}

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

func TestDataAgentJobJSONMatchesQueueEnvelope(t *testing.T) {
	t.Parallel()

	payload, err := json.Marshal(DataAgentJob{
		JobID:         "data_agent_job_1",
		DispatchID:    "dispatch-1",
		OwnerUserID:   "user-1",
		OwnerOrgID:    "org-1",
		ProjectID:     "nph-study",
		JobType:       "caption_resources",
		ResourceIDs:   []string{"file-a", "file-b"},
		ResourceCount: 2,
		InputSelector: domain.JSONMap{"label": "NPH"},
		Metadata:      domain.JSONMap{"requested_from": "resources_page"},
	})
	if err != nil {
		t.Fatalf("Marshal DataAgentJob: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("Unmarshal DataAgentJob: %v", err)
	}
	for _, key := range []string{
		"job_id",
		"dispatch_id",
		"owner_user_id",
		"owner_org_id",
		"project_id",
		"job_type",
		"resource_ids",
		"resource_count",
		"input_selector",
		"metadata",
	} {
		if _, ok := decoded[key]; !ok {
			t.Fatalf("data-agent job JSON missing %q in %s", key, payload)
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

func TestNATSMessageIDForDataAgentJobUsesJobID(t *testing.T) {
	t.Parallel()

	got := natsMessageIDForDataAgentJob(DataAgentJob{JobID: "data_agent_job_abc", JobType: "caption_resources"})

	if got != "data-agent-job:data_agent_job_abc" {
		t.Fatalf("natsMessageIDForDataAgentJob() = %q, want data-agent-job:data_agent_job_abc", got)
	}
}

func TestNATSMessageIDForRetriedDataAgentJobUsesDispatchID(t *testing.T) {
	t.Parallel()

	got := natsMessageIDForDataAgentJob(DataAgentJob{JobID: "data_agent_job_abc", DispatchID: "dispatch-2"})

	if got != "data-agent-job:data_agent_job_abc:dispatch-2" {
		t.Fatalf("natsMessageIDForDataAgentJob() = %q, want data-agent-job:data_agent_job_abc:dispatch-2", got)
	}
}

func TestNATSBusPublishDataAgentJobRequiresConfiguredSubject(t *testing.T) {
	t.Parallel()

	bus := &NATSBus{}

	err := bus.PublishDataAgentJob(context.Background(), DataAgentJob{JobID: "data_agent_job_abc"})
	if err == nil {
		t.Fatal("PublishDataAgentJob error = nil, want configuration error")
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

	stream := natsStreamConfig("ULTRA_TEST", []string{"ultra.test.jobs", "ultra.test.events"}, 0, 0)

	if stream.Duplicates < 24*time.Hour {
		t.Fatalf("duplicate window = %s, want at least 24h for long-run publish retries", stream.Duplicates)
	}
}

func TestNATSStreamSubjectsDoNotInventDataAgentSubject(t *testing.T) {
	t.Parallel()

	subjects := natsStreamSubjects(NATSConfig{
		JobsSubject:   "ultra.test.jobs",
		EventsSubject: "ultra.test.events",
		CancelSubject: "ultra.test.cancel",
	})

	for _, subject := range subjects {
		if subject == "ultra.data_agent.jobs" {
			t.Fatalf("subjects = %v, want no default data-agent subject for a bus that did not configure one", subjects)
		}
	}
}

func TestQueueConsumerDiagnosticsMarksIdlePullConsumerMissing(t *testing.T) {
	t.Parallel()

	diagnostics := queueConsumerDiagnosticsFromInfo(
		QueueConsumerTarget{Name: "ultra-deepagents-worker", Role: "deepagents", Subject: "ultra.runs.jobs"},
		&nats.ConsumerInfo{
			Name: "ultra-deepagents-worker",
			Config: nats.ConsumerConfig{
				Durable:        "ultra-deepagents-worker",
				FilterSubject:  "ultra.runs.jobs",
				AckPolicy:      nats.AckExplicitPolicy,
				AckWait:        10 * time.Minute,
				MaxDeliver:     5,
				MaxAckPending:  4,
				DeliverSubject: "",
			},
			NumPending:     7,
			NumWaiting:     0,
			NumAckPending:  0,
			NumRedelivered: 2,
		},
	)

	if diagnostics.Active {
		t.Fatalf("active = true for idle pull consumer with no waiting pulls or in-flight messages: %+v", diagnostics)
	}
	if diagnostics.PendingMessages != 7 || diagnostics.RedeliveredMessages != 2 {
		t.Fatalf("diagnostics = %+v, want pending/redelivery counters preserved", diagnostics)
	}
}

func TestQueueConsumerDiagnosticsMarksPullConsumerActiveWhenPollingOrInFlight(t *testing.T) {
	t.Parallel()

	for name, info := range map[string]*nats.ConsumerInfo{
		"waiting pull request": {
			Name:       "ultra-deepagents-worker",
			Config:     nats.ConsumerConfig{Durable: "ultra-deepagents-worker", FilterSubject: "ultra.runs.jobs"},
			NumWaiting: 1,
		},
		"in-flight work": {
			Name:          "ultra-deepagents-worker",
			Config:        nats.ConsumerConfig{Durable: "ultra-deepagents-worker", FilterSubject: "ultra.runs.jobs"},
			NumAckPending: 2,
		},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			diagnostics := queueConsumerDiagnosticsFromInfo(
				QueueConsumerTarget{Name: "ultra-deepagents-worker", Role: "deepagents", Subject: "ultra.runs.jobs"},
				info,
			)
			if !diagnostics.Active {
				t.Fatalf("active = false for pull consumer with %s: %+v", name, diagnostics)
			}
		})
	}
}

func TestQueueConsumerDiagnosticsUsesPushBoundForPushConsumers(t *testing.T) {
	t.Parallel()

	baseInfo := nats.ConsumerInfo{
		Name: "ultra-control-event-ingest",
		Config: nats.ConsumerConfig{
			Durable:        "ultra-control-event-ingest",
			DeliverSubject: "ultra.runs.events.deliver.ultra-control-event-ingest",
			DeliverGroup:   "ultra-control-event-ingest",
			FilterSubject:  "ultra.runs.events",
			AckPolicy:      nats.AckExplicitPolicy,
		},
	}

	missing := baseInfo
	missing.PushBound = false
	missingDiagnostics := queueConsumerDiagnosticsFromInfo(
		QueueConsumerTarget{Name: "ultra-control-event-ingest", Role: "event_ingest", Subject: "ultra.runs.events"},
		&missing,
	)
	if missingDiagnostics.Active {
		t.Fatalf("active = true for unbound push consumer: %+v", missingDiagnostics)
	}

	bound := baseInfo
	bound.PushBound = true
	boundDiagnostics := queueConsumerDiagnosticsFromInfo(
		QueueConsumerTarget{Name: "ultra-control-event-ingest", Role: "event_ingest", Subject: "ultra.runs.events"},
		&bound,
	)
	if !boundDiagnostics.Active {
		t.Fatalf("active = false for push-bound consumer: %+v", boundDiagnostics)
	}
}

func TestNATSBusQueueConsumerTargetsIncludesRunEventPartitions(t *testing.T) {
	t.Parallel()

	bus := &NATSBus{cfg: NATSConfig{
		Stream:          "ULTRA_TEST",
		EventsSubject:   "ultra.test.events",
		EventConsumer:   "ultra-test-event-ingest",
		EventPartitions: 3,
		ConsumerTargets: []QueueConsumerTarget{
			{Name: "ultra-test-worker", Role: "deepagents", Subject: "ultra.test.jobs"},
			{Name: "ultra-test-event-ingest", Role: "event_ingest", Subject: "ultra.test.events"},
		},
	}}

	targets := bus.queueConsumerTargets()

	wantNames := []string{
		"ultra-test-worker",
		"ultra-test-event-ingest",
		"ultra-test-event-ingest-p-0",
		"ultra-test-event-ingest-p-1",
		"ultra-test-event-ingest-p-2",
	}
	if len(targets) != len(wantNames) {
		t.Fatalf("targets = %+v, want %d targets", targets, len(wantNames))
	}
	for index, want := range wantNames {
		if targets[index].Name != want {
			t.Fatalf("target %d name = %q, want %q (targets=%+v)", index, targets[index].Name, want, targets)
		}
	}
	if targets[2].Role != "event_ingest_partition" || targets[2].Subject != "ultra.test.events.p.0" {
		t.Fatalf("partition target = %+v, want partition role and subject", targets[2])
	}
}

func TestEnsureNATSStreamReturnsExistingStreamUpdateFailure(t *testing.T) {
	t.Parallel()

	updateErr := fmt.Errorf("update stream failed")
	manager := fakeNATSStreamManager{
		addErr:    nats.ErrStreamNameAlreadyInUse,
		updateErr: updateErr,
	}

	err := ensureNATSStream(context.Background(), &manager, natsStreamConfig("ULTRA_TEST", []string{"ultra.test.jobs"}, 0, 0))
	if !errors.Is(err, updateErr) {
		t.Fatalf("ensureNATSStream error = %v, want update error %v", err, updateErr)
	}
	if manager.updateCalls != 1 {
		t.Fatalf("update calls = %d, want 1", manager.updateCalls)
	}
}

func TestEnsureNATSStreamUpdatesSameStreamOnSubjectOverlap(t *testing.T) {
	t.Parallel()

	manager := fakeNATSStreamManager{
		addErr: fmt.Errorf("nats: subjects overlap with an existing stream"),
	}

	if err := ensureNATSStream(context.Background(), &manager, natsStreamConfig("ULTRA_TEST", []string{"ultra.test.jobs"}, 0, 0)); err != nil {
		t.Fatalf("ensureNATSStream: %v", err)
	}
	if manager.infoCalls != 1 {
		t.Fatalf("stream info calls = %d, want 1", manager.infoCalls)
	}
	if manager.updateCalls != 1 {
		t.Fatalf("update calls = %d, want 1", manager.updateCalls)
	}
}

func TestEnsureNATSStreamPreservesExistingSubjectsWhenUpdatingStream(t *testing.T) {
	t.Parallel()

	manager := fakeNATSStreamManager{
		addErr: nats.ErrStreamNameAlreadyInUse,
		info: &nats.StreamInfo{Config: nats.StreamConfig{
			Name:        "ULTRA_TEST",
			Description: "operator-managed stream",
			Subjects:    []string{"ultra.test.jobs", "ultra.test.events", "ultra.test.cancel", "ultra.test.data_agent.jobs"},
			MaxAge:      2 * time.Hour,
			MaxMsgs:     10_000,
			Storage:     nats.MemoryStorage,
			Duplicates:  2 * time.Minute,
		}},
	}

	if err := ensureNATSStream(context.Background(), &manager, natsStreamConfig("ULTRA_TEST", []string{
		"ultra.test.jobs",
		"ultra.test.events",
		"ultra.test.cancel",
	}, 0, 0)); err != nil {
		t.Fatalf("ensureNATSStream: %v", err)
	}
	if manager.updated == nil {
		t.Fatalf("updated stream config = nil, want merged subjects")
	}
	want := []string{"ultra.test.jobs", "ultra.test.events", "ultra.test.cancel", "ultra.test.data_agent.jobs"}
	if got := manager.updated.Subjects; !stringSlicesEqual(got, want) {
		t.Fatalf("updated subjects = %v, want %v", got, want)
	}
	if manager.updated.Description != "operator-managed stream" {
		t.Fatalf("updated description = %q, want existing stream description preserved", manager.updated.Description)
	}
	if manager.updated.MaxAge != 2*time.Hour || manager.updated.MaxMsgs != 10_000 {
		t.Fatalf("updated limits MaxAge=%s MaxMsgs=%d, want existing retention limits preserved", manager.updated.MaxAge, manager.updated.MaxMsgs)
	}
	if manager.updated.Storage != nats.MemoryStorage {
		t.Fatalf("updated storage = %s, want existing storage preserved", manager.updated.Storage)
	}
	if manager.updated.Duplicates < natsDuplicateWindow {
		t.Fatalf("updated duplicate window = %s, want at least %s", manager.updated.Duplicates, natsDuplicateWindow)
	}
}

func TestEnsureNATSStreamDoesNotHideForeignSubjectOverlap(t *testing.T) {
	t.Parallel()

	addErr := fmt.Errorf("nats: subjects overlap with an existing stream")
	manager := fakeNATSStreamManager{
		addErr:  addErr,
		infoErr: nats.ErrStreamNotFound,
	}

	err := ensureNATSStream(context.Background(), &manager, natsStreamConfig("ULTRA_TEST", []string{"ultra.test.jobs"}, 0, 0))
	if !errors.Is(err, addErr) {
		t.Fatalf("ensureNATSStream error = %v, want add error %v", err, addErr)
	}
	if manager.infoCalls != 1 {
		t.Fatalf("stream info calls = %d, want 1", manager.infoCalls)
	}
	if manager.updateCalls != 0 {
		t.Fatalf("update calls = %d, want 0", manager.updateCalls)
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
	dataAgentJobsSubject := "ultra.test." + suffix + ".data_agent.jobs"
	bus, err := NewNATSBus(ctx, NATSConfig{
		URL:                  url,
		Stream:               stream,
		JobsSubject:          jobsSubject,
		EventsSubject:        eventsSubject,
		CancelSubject:        cancelSubject,
		DataAgentJobsSubject: dataAgentJobsSubject,
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

func TestNATSBusPreservesExistingStreamSubjectsWhenSecondaryProducerOmitsDataAgentSubject(t *testing.T) {
	url := os.Getenv("ULTRA_CONTROL_TEST_NATS_URL")
	if url == "" {
		t.Skip("ULTRA_CONTROL_TEST_NATS_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	suffix := fmt.Sprintf("%d", time.Now().UnixNano())
	stream := "ULTRA_TEST_SUBJECT_UNION_" + suffix
	jobsSubject := "ultra.test." + suffix + ".jobs"
	eventsSubject := "ultra.test." + suffix + ".events"
	cancelSubject := "ultra.test." + suffix + ".cancel"
	dataAgentJobsSubject := "ultra.test." + suffix + ".data_agent.jobs"
	fullBus, err := NewNATSBus(ctx, NATSConfig{
		URL:                  url,
		Stream:               stream,
		JobsSubject:          jobsSubject,
		EventsSubject:        eventsSubject,
		CancelSubject:        cancelSubject,
		DataAgentJobsSubject: dataAgentJobsSubject,
	})
	if err != nil {
		t.Fatalf("NewNATSBus full: %v", err)
	}
	defer fullBus.Close()
	defer func() {
		_ = fullBus.js.DeleteStream(stream)
	}()

	secondaryBus, err := NewNATSBus(ctx, NATSConfig{
		URL:           url,
		Stream:        stream,
		JobsSubject:   jobsSubject,
		EventsSubject: eventsSubject,
		CancelSubject: cancelSubject,
	})
	if err != nil {
		t.Fatalf("NewNATSBus secondary: %v", err)
	}
	defer secondaryBus.Close()

	info, err := fullBus.js.StreamInfo(stream, nats.Context(ctx))
	if err != nil {
		t.Fatalf("StreamInfo: %v", err)
	}
	if !containsString(info.Config.Subjects, dataAgentJobsSubject) {
		t.Fatalf("stream subjects = %v, want preserved data-agent subject %q", info.Config.Subjects, dataAgentJobsSubject)
	}
	if err := fullBus.PublishDataAgentJob(ctx, DataAgentJob{JobID: "data-agent-job-preserved"}); err != nil {
		t.Fatalf("PublishDataAgentJob after secondary bus startup: %v", err)
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
	dataAgentJobsSubject := "ultra.test." + suffix + ".data_agent.jobs"
	bus, err := NewNATSBus(ctx, NATSConfig{
		URL:                  url,
		Stream:               stream,
		JobsSubject:          jobsSubject,
		EventsSubject:        eventsSubject,
		CancelSubject:        cancelSubject,
		DataAgentJobsSubject: dataAgentJobsSubject,
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
	dataAgentJobsSubject := "ultra.test." + suffix + ".data_agent.jobs"
	consumer := "ultra-test-event-consumer-" + suffix
	bus, err := NewNATSBus(ctx, NATSConfig{
		URL:                  url,
		Stream:               stream,
		JobsSubject:          jobsSubject,
		EventsSubject:        eventsSubject,
		CancelSubject:        cancelSubject,
		DataAgentJobsSubject: dataAgentJobsSubject,
		EventConsumer:        consumer,
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

// TestNATSSubscribeRunEventsCloseRaceDoesNotPanic hammers the subscribe -> publish ->
// disconnect cycle so a message dispatched on the NATS callback goroutine collides with
// unsubscribe()'s close(ch). Before the mu/closed guard this raced to "send on closed
// channel", panicking the whole process (an SSE viewer disconnecting mid-run would crash
// the control plane for every other user). The guard must keep it crash-free.
func TestNATSSubscribeRunEventsCloseRaceDoesNotPanic(t *testing.T) {
	url := os.Getenv("ULTRA_CONTROL_TEST_NATS_URL")
	if url == "" {
		t.Skip("ULTRA_CONTROL_TEST_NATS_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	suffix := fmt.Sprintf("%d", time.Now().UnixNano())
	stream := "ULTRA_TEST_EVENT_CLOSE_RACE_" + suffix
	eventsSubject := "ultra.test." + suffix + ".events"
	bus, err := NewNATSBus(ctx, NATSConfig{
		URL:                  url,
		Stream:               stream,
		JobsSubject:          "ultra.test." + suffix + ".jobs",
		EventsSubject:        eventsSubject,
		CancelSubject:        "ultra.test." + suffix + ".cancel",
		DataAgentJobsSubject: "ultra.test." + suffix + ".data_agent.jobs",
		EventConsumer:        "ultra-test-event-close-race-" + suffix,
	})
	if err != nil {
		t.Fatalf("NewNATSBus: %v", err)
	}
	defer bus.Close()
	defer func() { _ = bus.js.DeleteStream(stream) }()

	const rounds = 200
	for i := 0; i < rounds; i++ {
		runID := fmt.Sprintf("race-run-%d", i)
		subCtx, stopSub := context.WithCancel(ctx)
		events, unsubscribe := bus.SubscribeRunEvents(subCtx, runID)
		// Drain so the buffered channel can't fill (irrelevant to the race, but realistic).
		go func() {
			for range events { //nolint:revive // intentional drain
			}
		}()
		// Publish a burst while we tear the subscription down, maximizing the odds that a
		// callback is mid-dispatch when close(ch) runs.
		go func() {
			for j := 0; j < 8; j++ {
				_ = bus.PublishRunEvent(ctx, domain.RunEventRecord{
					EventID:   fmt.Sprintf("%s-evt-%d", runID, j),
					RunID:     runID,
					ThreadID:  "thread-close-race",
					EventKind: "run.delta",
				})
			}
		}()
		stopSub()     // ctx-cancel path -> unsubscribe -> close(ch)
		unsubscribe() // explicit path too (idempotent); both must be panic-safe
	}
	// Reaching here without a panic is the assertion.
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
	dataAgentJobsSubject := "ultra.test." + suffix + ".data_agent.jobs"
	consumer := "ultra-test-event-reconcile-" + suffix
	bus, err := NewNATSBus(ctx, NATSConfig{
		URL:                  url,
		Stream:               stream,
		JobsSubject:          jobsSubject,
		EventsSubject:        eventsSubject,
		CancelSubject:        cancelSubject,
		DataAgentJobsSubject: dataAgentJobsSubject,
		EventConsumer:        consumer,
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
	dataAgentJobsSubject := "ultra.test." + suffix + ".data_agent.jobs"
	consumer := "ultra-test-event-multi-" + suffix
	cfg := NATSConfig{
		URL:                  url,
		Stream:               stream,
		JobsSubject:          jobsSubject,
		EventsSubject:        eventsSubject,
		CancelSubject:        cancelSubject,
		DataAgentJobsSubject: dataAgentJobsSubject,
		EventConsumer:        consumer,
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

func TestNATSRunEventConsumerPreservesSameRunOrderAcrossControlPlaneSubscribers(t *testing.T) {
	url := os.Getenv("ULTRA_CONTROL_TEST_NATS_URL")
	if url == "" {
		t.Skip("ULTRA_CONTROL_TEST_NATS_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	suffix := fmt.Sprintf("%d", time.Now().UnixNano())
	stream := "ULTRA_TEST_EVENT_CROSS_REPLICA_ORDER_" + suffix
	eventsSubject := "ultra.test." + suffix + ".events"
	consumer := "ultra-test-event-cross-replica-order-" + suffix
	cfg := NATSConfig{
		URL:                  url,
		Stream:               stream,
		JobsSubject:          "ultra.test." + suffix + ".jobs",
		EventsSubject:        eventsSubject,
		CancelSubject:        "ultra.test." + suffix + ".cancel",
		DataAgentJobsSubject: "ultra.test." + suffix + ".data_agent.jobs",
		EventConsumer:        consumer,
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
	runID := "run-cross-replica-order-" + suffix
	firstStarted := make(chan struct{})
	releaseFirst := make(chan struct{})
	handledLater := make(chan int, 1)
	var firstOnce sync.Once
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(releaseFirst) }) }
	if err := busA.SubscribeAllRunEvents(subCtx, func(handlerCtx context.Context, input domain.AppendRunEventInput) error {
		if input.RunID != runID {
			return nil
		}
		if input.SourceSequence == 1 {
			firstOnce.Do(func() { close(firstStarted) })
			select {
			case <-releaseFirst:
			case <-handlerCtx.Done():
			}
			return nil
		}
		select {
		case handledLater <- int(input.SourceSequence):
		default:
		}
		return nil
	}); err != nil {
		t.Fatalf("SubscribeAllRunEvents A: %v", err)
	}
	defer release()

	if err := busA.PublishRunEvent(ctx, domain.RunEventRecord{
		EventID:   "evt-" + suffix + "-1",
		Sequence:  1,
		RunID:     runID,
		ThreadID:  "thread-" + suffix,
		EventKind: "message.delta",
		Message:   "first",
	}); err != nil {
		t.Fatalf("PublishRunEvent first: %v", err)
	}
	select {
	case <-firstStarted:
	case <-ctx.Done():
		t.Fatal("first run event was not being handled before timeout")
	}

	if err := busB.SubscribeAllRunEvents(subCtx, func(_ context.Context, input domain.AppendRunEventInput) error {
		if input.RunID != runID || input.SourceSequence <= 1 {
			return nil
		}
		select {
		case handledLater <- int(input.SourceSequence):
		default:
		}
		return nil
	}); err != nil {
		t.Fatalf("SubscribeAllRunEvents B: %v", err)
	}
	for sequence := 2; sequence <= 129; sequence++ {
		if err := busA.PublishRunEvent(ctx, domain.RunEventRecord{
			EventID:   fmt.Sprintf("evt-%s-%d", suffix, sequence),
			Sequence:  int64(sequence),
			RunID:     runID,
			ThreadID:  "thread-" + suffix,
			EventKind: "message.delta",
			Message:   fmt.Sprintf("event-%d", sequence),
		}); err != nil {
			t.Fatalf("PublishRunEvent %d: %v", sequence, err)
		}
	}

	select {
	case sequence := <-handledLater:
		t.Fatalf("same-run source sequence %d was handled before source sequence 1 completed", sequence)
	case <-time.After(500 * time.Millisecond):
	}
	release()
	select {
	case sequence := <-handledLater:
		if sequence <= 1 {
			t.Fatalf("handled source sequence %d after release, want a later event", sequence)
		}
	case <-ctx.Done():
		t.Fatal("no later same-run event was handled after source sequence 1 completed")
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

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func stringSlicesEqual(left []string, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
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

func TestMemoryBusPublishesDataAgentJobs(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	bus := NewMemoryBus()
	job := DataAgentJob{
		JobID:         "data_agent_job_memory",
		OwnerUserID:   "user-1",
		OwnerOrgID:    "org-1",
		JobType:       "extract_metadata",
		ResourceIDs:   []string{"file-a"},
		ResourceCount: 1,
	}

	if err := bus.PublishDataAgentJob(ctx, job); err != nil {
		t.Fatalf("PublishDataAgentJob: %v", err)
	}

	select {
	case got := <-bus.DataAgentJobs():
		if got.JobID != job.JobID || got.JobType != job.JobType || got.OwnerUserID != job.OwnerUserID {
			t.Fatalf("data-agent job = %+v, want %+v", got, job)
		}
	case <-time.After(time.Second):
		t.Fatalf("expected published data-agent job")
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

func TestNATSBusConnectionSurvivesOutagesAndDrainsOnClose(t *testing.T) {
	url := os.Getenv("ULTRA_CONTROL_TEST_NATS_URL")
	if url == "" {
		t.Skip("ULTRA_CONTROL_TEST_NATS_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	suffix := fmt.Sprintf("%d", time.Now().UnixNano())
	stream := "ULTRA_TEST_RESILIENCE_" + suffix
	bus, err := NewNATSBus(ctx, NATSConfig{
		URL:                  url,
		Stream:               stream,
		JobsSubject:          "ultra.test." + suffix + ".jobs",
		EventsSubject:        "ultra.test." + suffix + ".events",
		CancelSubject:        "ultra.test." + suffix + ".cancel",
		DataAgentJobsSubject: "ultra.test." + suffix + ".data_agent.jobs",
	})
	if err != nil {
		t.Fatalf("NewNATSBus: %v", err)
	}
	defer func() {
		_ = bus.js.DeleteStream(stream)
	}()

	// The reconnect policy is the behavioral contract that prevents the
	// connection from dying permanently after a NATS outage longer than the
	// default 60 attempts x 2s window.
	opts := bus.conn.Opts
	if opts.MaxReconnect != -1 {
		t.Fatalf("MaxReconnect = %d, want -1 (retry forever)", opts.MaxReconnect)
	}
	if opts.ReconnectWait != natsReconnectWait {
		t.Fatalf("ReconnectWait = %s, want %s", opts.ReconnectWait, natsReconnectWait)
	}
	if opts.ClosedCB == nil || opts.DisconnectedErrCB == nil || opts.ReconnectedCB == nil || opts.AsyncErrorCB == nil {
		t.Fatal("connection state handlers must be installed for observability")
	}

	done := make(chan struct{})
	go func() {
		bus.Close()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(natsDrainTimeout + 3*time.Second):
		t.Fatal("Close did not complete within the drain budget")
	}
	if !bus.conn.IsClosed() {
		t.Fatal("connection is not closed after Close")
	}
	// Close must be safe to call again after the connection is gone.
	bus.Close()
}

func TestNATSBusPartitionedIngestPreservesPerRunOrder(t *testing.T) {
	url := os.Getenv("ULTRA_CONTROL_TEST_NATS_URL")
	if url == "" {
		t.Skip("ULTRA_CONTROL_TEST_NATS_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	suffix := fmt.Sprintf("%d", time.Now().UnixNano())
	stream := "ULTRA_TEST_PARTITION_" + suffix
	bus, err := NewNATSBus(ctx, NATSConfig{
		URL:                    url,
		Stream:                 stream,
		JobsSubject:            "ultra.test." + suffix + ".jobs",
		EventsSubject:          "ultra.test." + suffix + ".events",
		CancelSubject:          "ultra.test." + suffix + ".cancel",
		DataAgentJobsSubject:   "ultra.test." + suffix + ".data_agent.jobs",
		EventConsumer:          "ingest-" + suffix,
		EventIngestConcurrency: 4,
	})
	if err != nil {
		t.Fatalf("NewNATSBus: %v", err)
	}
	defer bus.Close()
	defer func() {
		_ = bus.js.DeleteStream(stream)
	}()

	const runs = 5
	const eventsPerRun = 40
	var mu sync.Mutex
	received := map[string][]int{}
	done := make(chan struct{})
	total := 0
	err = bus.SubscribeAllRunEvents(ctx, func(_ context.Context, input domain.AppendRunEventInput) error {
		index := 0
		if _, scanErr := fmt.Sscan(input.Message, &index); scanErr != nil {
			return scanErr
		}
		mu.Lock()
		received[input.RunID] = append(received[input.RunID], index)
		total++
		if total == runs*eventsPerRun {
			close(done)
		}
		mu.Unlock()
		return nil
	})
	if err != nil {
		t.Fatalf("SubscribeAllRunEvents: %v", err)
	}

	for index := 0; index < eventsPerRun; index++ {
		for run := 0; run < runs; run++ {
			runID := fmt.Sprintf("run-%s-%d", suffix, run)
			event := domain.RunEventRecord{
				EventID:   fmt.Sprintf("evt-%s-%d-%d", suffix, run, index),
				RunID:     runID,
				EventKind: "message.delta",
				Message:   fmt.Sprintf("%d", index),
			}
			if err := bus.PublishRunEvent(ctx, event); err != nil {
				t.Fatalf("PublishRunEvent: %v", err)
			}
		}
	}

	select {
	case <-done:
	case <-ctx.Done():
		mu.Lock()
		t.Fatalf("received %d/%d events before timeout", total, runs*eventsPerRun)
	}
	mu.Lock()
	defer mu.Unlock()
	for runID, indexes := range received {
		if len(indexes) != eventsPerRun {
			t.Fatalf("run %s received %d events, want %d", runID, len(indexes), eventsPerRun)
		}
		for position, index := range indexes {
			if index != position {
				t.Fatalf("run %s event %d arrived at position %d; per-run order must be preserved (got %v)", runID, index, position, indexes)
			}
		}
	}
}

func TestNATSBusRunEventIngestPreservesProducerSequence(t *testing.T) {
	url := os.Getenv("ULTRA_CONTROL_TEST_NATS_URL")
	if url == "" {
		t.Skip("ULTRA_CONTROL_TEST_NATS_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	suffix := fmt.Sprintf("%d", time.Now().UnixNano())
	stream := "ULTRA_TEST_EVENT_SOURCE_SEQUENCE_" + suffix
	consumer := "ultra-test-event-source-sequence-" + suffix
	bus, err := NewNATSBus(ctx, NATSConfig{
		URL:                  url,
		Stream:               stream,
		JobsSubject:          "ultra.test." + suffix + ".jobs",
		EventsSubject:        "ultra.test." + suffix + ".events",
		CancelSubject:        "ultra.test." + suffix + ".cancel",
		DataAgentJobsSubject: "ultra.test." + suffix + ".data_agent.jobs",
		EventConsumer:        consumer,
	})
	if err != nil {
		t.Fatalf("NewNATSBus: %v", err)
	}
	defer bus.Close()
	defer func() {
		_ = bus.js.DeleteStream(stream)
	}()

	handled := make(chan domain.AppendRunEventInput, 1)
	if err := bus.SubscribeAllRunEvents(ctx, func(_ context.Context, input domain.AppendRunEventInput) error {
		handled <- input
		return nil
	}); err != nil {
		t.Fatalf("SubscribeAllRunEvents: %v", err)
	}
	if err := bus.PublishRunEvent(ctx, domain.RunEventRecord{
		EventID:   "evt-source-sequence-7",
		Sequence:  7,
		RunID:     "run-source-sequence-" + suffix,
		ThreadID:  "thread-source-sequence",
		EventKind: "message.delta",
	}); err != nil {
		t.Fatalf("PublishRunEvent: %v", err)
	}

	select {
	case input := <-handled:
		if input.SourceSequence != 7 {
			t.Fatalf("source sequence = %d, want producer sequence 7", input.SourceSequence)
		}
	case <-ctx.Done():
		t.Fatal("handler was not called before timeout")
	}
}

func TestNATSRunEventConsumerDoesNotLetFullHotPartitionBlockColdRun(t *testing.T) {
	url := os.Getenv("ULTRA_CONTROL_TEST_NATS_URL")
	if url == "" {
		t.Skip("ULTRA_CONTROL_TEST_NATS_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	suffix := fmt.Sprintf("%d", time.Now().UnixNano())
	stream := "ULTRA_TEST_EVENT_FAIRNESS_" + suffix
	consumer := "ultra-test-event-fairness-" + suffix
	bus, err := NewNATSBus(ctx, NATSConfig{
		URL:                  url,
		Stream:               stream,
		JobsSubject:          "ultra.test." + suffix + ".jobs",
		EventsSubject:        "ultra.test." + suffix + ".events",
		CancelSubject:        "ultra.test." + suffix + ".cancel",
		DataAgentJobsSubject: "ultra.test." + suffix + ".data_agent.jobs",
		EventConsumer:        consumer,
		EventPartitions:      2,
	})
	if err != nil {
		t.Fatalf("NewNATSBus: %v", err)
	}
	defer bus.Close()
	defer func() {
		_ = bus.js.DeleteStream(stream)
	}()

	hotRunID := "run-hot-" + suffix
	hotPartition := runEventIngestPartition(hotRunID, 2)
	coldRunID := ""
	for index := 0; index < 128; index++ {
		candidate := fmt.Sprintf("run-cold-%s-%d", suffix, index)
		if runEventIngestPartition(candidate, 2) != hotPartition {
			coldRunID = candidate
			break
		}
	}
	if coldRunID == "" {
		t.Fatal("could not find a cold run ID on a different ingest partition")
	}

	hotStarted := make(chan struct{})
	releaseHot := make(chan struct{})
	coldHandled := make(chan struct{})
	hotLaterHandled := make(chan struct{}, 1)
	var hotOnce sync.Once
	var coldOnce sync.Once
	if err := bus.SubscribeAllRunEvents(ctx, func(handlerCtx context.Context, input domain.AppendRunEventInput) error {
		switch input.RunID {
		case hotRunID:
			if input.SourceSequence > 1 {
				select {
				case hotLaterHandled <- struct{}{}:
				default:
				}
				return nil
			}
			hotOnce.Do(func() { close(hotStarted) })
			select {
			case <-releaseHot:
			case <-handlerCtx.Done():
			}
		case coldRunID:
			coldOnce.Do(func() { close(coldHandled) })
		}
		return nil
	}); err != nil {
		t.Fatalf("SubscribeAllRunEvents: %v", err)
	}
	defer close(releaseHot)

	publish := func(runID string, index int) {
		t.Helper()
		if err := bus.PublishRunEvent(ctx, domain.RunEventRecord{
			EventID:   fmt.Sprintf("evt-%s-%s-%d", suffix, runID, index),
			Sequence:  int64(index + 1),
			RunID:     runID,
			ThreadID:  "thread-" + suffix,
			EventKind: "message.delta",
			Message:   fmt.Sprintf("%d", index),
		}); err != nil {
			t.Fatalf("PublishRunEvent(%s, %d): %v", runID, index, err)
		}
	}

	publish(hotRunID, 0)
	select {
	case <-hotStarted:
	case <-ctx.Done():
		t.Fatal("hot run handler did not start before timeout")
	}

	publish(hotRunID, 1)
	select {
	case <-hotLaterHandled:
		t.Fatal("later hot-run event was handled while the same broker partition was blocked")
	case <-time.After(150 * time.Millisecond):
	}

	start := time.Now()
	publish(coldRunID, 0)
	select {
	case <-coldHandled:
		t.Logf("cold run on partition %d handled while hot run on partition %d was saturated after %s", runEventIngestPartition(coldRunID, 2), hotPartition, time.Since(start))
	case <-time.After(1500 * time.Millisecond):
		t.Fatalf("cold run event on partition %d was blocked behind saturated hot partition %d", runEventIngestPartition(coldRunID, 2), hotPartition)
	}
}

func BenchmarkNATSPartitionedRunEventIngestColdRunsWhileHotPartitionBlocked(b *testing.B) {
	url := os.Getenv("ULTRA_CONTROL_TEST_NATS_URL")
	if url == "" {
		b.Skip("ULTRA_CONTROL_TEST_NATS_URL is not set")
	}
	for iteration := 0; iteration < b.N; iteration++ {
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		suffix := fmt.Sprintf("%d-%d", time.Now().UnixNano(), iteration)
		stream := "ULTRA_BENCH_EVENT_PARTITION_" + suffix
		consumer := "ultra-bench-event-partition-" + suffix
		const partitions = 16
		cfg := NATSConfig{
			URL:                  url,
			Stream:               stream,
			JobsSubject:          "ultra.bench." + suffix + ".jobs",
			EventsSubject:        "ultra.bench." + suffix + ".events",
			CancelSubject:        "ultra.bench." + suffix + ".cancel",
			DataAgentJobsSubject: "ultra.bench." + suffix + ".data_agent.jobs",
			EventConsumer:        consumer,
			EventPartitions:      partitions,
		}
		busA, err := NewNATSBus(ctx, cfg)
		if err != nil {
			cancel()
			b.Fatalf("NewNATSBus A: %v", err)
		}
		busB, err := NewNATSBus(ctx, cfg)
		if err != nil {
			busA.Close()
			cancel()
			b.Fatalf("NewNATSBus B: %v", err)
		}

		hotRunID := "run-hot-" + suffix
		hotPartition := runEventIngestPartition(hotRunID, partitions)
		coldRuns := make([]string, 0, 32)
		for candidate := 0; len(coldRuns) < cap(coldRuns) && candidate < 4096; candidate++ {
			runID := fmt.Sprintf("run-cold-%s-%d", suffix, candidate)
			if runEventIngestPartition(runID, partitions) != hotPartition {
				coldRuns = append(coldRuns, runID)
			}
		}
		if len(coldRuns) != cap(coldRuns) {
			busB.Close()
			busA.Close()
			cancel()
			b.Fatalf("found %d cold runs, want %d", len(coldRuns), cap(coldRuns))
		}

		var mu sync.Mutex
		publishedAt := map[string]time.Time{}
		latencies := make([]time.Duration, 0, len(coldRuns)*32)
		receivedCold := 0
		coldDone := make(chan struct{})
		hotStarted := make(chan struct{})
		releaseHot := make(chan struct{})
		var hotOnce sync.Once
		handler := func(handlerCtx context.Context, input domain.AppendRunEventInput) error {
			if input.RunID == hotRunID {
				hotOnce.Do(func() { close(hotStarted) })
				select {
				case <-releaseHot:
				case <-handlerCtx.Done():
				}
				return nil
			}
			mu.Lock()
			if sentAt, ok := publishedAt[input.EventID]; ok {
				latencies = append(latencies, time.Since(sentAt))
			}
			receivedCold++
			if receivedCold == cap(coldRuns)*32 {
				close(coldDone)
			}
			mu.Unlock()
			return nil
		}
		if err := busA.SubscribeAllRunEvents(ctx, handler); err != nil {
			busB.Close()
			busA.Close()
			cancel()
			b.Fatalf("SubscribeAllRunEvents A: %v", err)
		}
		if err := busB.SubscribeAllRunEvents(ctx, handler); err != nil {
			busB.Close()
			busA.Close()
			cancel()
			b.Fatalf("SubscribeAllRunEvents B: %v", err)
		}

		if err := busA.PublishRunEvent(ctx, domain.RunEventRecord{
			EventID:   "evt-" + suffix + "-hot-1",
			Sequence:  1,
			RunID:     hotRunID,
			ThreadID:  "thread-" + suffix,
			EventKind: "message.delta",
		}); err != nil {
			busB.Close()
			busA.Close()
			cancel()
			b.Fatalf("PublishRunEvent hot: %v", err)
		}
		select {
		case <-hotStarted:
		case <-ctx.Done():
			busB.Close()
			busA.Close()
			cancel()
			b.Fatal("hot run did not start before timeout")
		}

		start := time.Now()
		for _, runID := range coldRuns {
			for sequence := 1; sequence <= 32; sequence++ {
				eventID := fmt.Sprintf("evt-%s-%s-%d", suffix, runID, sequence)
				mu.Lock()
				publishedAt[eventID] = time.Now()
				mu.Unlock()
				if err := busA.PublishRunEvent(ctx, domain.RunEventRecord{
					EventID:   eventID,
					Sequence:  int64(sequence),
					RunID:     runID,
					ThreadID:  "thread-" + suffix,
					EventKind: "message.delta",
				}); err != nil {
					busB.Close()
					busA.Close()
					cancel()
					b.Fatalf("PublishRunEvent cold: %v", err)
				}
			}
		}
		select {
		case <-coldDone:
		case <-ctx.Done():
			busB.Close()
			busA.Close()
			cancel()
			b.Fatalf("received %d/%d cold events before timeout", receivedCold, cap(coldRuns)*32)
		}
		drain := time.Since(start)
		close(releaseHot)
		info, err := busA.js.ConsumerInfo(stream, runEventPartitionConsumerName(consumer, hotPartition), nats.Context(ctx))
		if err != nil {
			busB.Close()
			busA.Close()
			cancel()
			b.Fatalf("ConsumerInfo hot partition: %v", err)
		}
		if info.NumRedelivered != 0 {
			busB.Close()
			busA.Close()
			cancel()
			b.Fatalf("hot partition redelivered %d messages during ordered benchmark", info.NumRedelivered)
		}
		mu.Lock()
		latencySnapshot := append([]time.Duration(nil), latencies...)
		mu.Unlock()
		sort.Slice(latencySnapshot, func(i, j int) bool { return latencySnapshot[i] < latencySnapshot[j] })
		if len(latencySnapshot) > 0 {
			b.ReportMetric(float64(percentileDuration(latencySnapshot, 0.95).Microseconds())/1000, "p95_ms")
			b.ReportMetric(float64(percentileDuration(latencySnapshot, 0.99).Microseconds())/1000, "p99_ms")
		}
		b.ReportMetric(float64(drain.Milliseconds()), "cold_drain_ms")
		b.ReportMetric(float64(cap(coldRuns)*32)/drain.Seconds(), "cold_events_per_sec")
		if drain > 5*time.Second {
			busB.Close()
			busA.Close()
			cancel()
			b.Fatalf("cold events drained in %s, want <= 5s while hot partition is blocked", drain)
		}
		busB.Close()
		busA.Close()
		_ = busA.js.DeleteStream(stream)
		cancel()
	}
}

func percentileDuration(values []time.Duration, percentile float64) time.Duration {
	if len(values) == 0 {
		return 0
	}
	index := int(float64(len(values)-1) * percentile)
	if index < 0 {
		index = 0
	}
	if index >= len(values) {
		index = len(values) - 1
	}
	return values[index]
}

// Fix #1: a persistently-failing ("poison") event must not wedge its partition
// forever. After the bounded InProgress retries, the worker Term()s it, freeing
// the MaxAckPending=1 slot so the next same-partition event is still ingested.
func TestNATSBusPartitionResumesAfterPoisonEvent(t *testing.T) {
	url := os.Getenv("ULTRA_CONTROL_TEST_NATS_URL")
	if url == "" {
		t.Skip("ULTRA_CONTROL_TEST_NATS_URL is not set")
	}
	// Shrink the retry budget so the test finishes quickly.
	origDelay, origBounded, origUnknown := runEventIngestNakDelay, runEventIngestBoundedRetries, runEventIngestUnknownRetries
	runEventIngestNakDelay = 10 * time.Millisecond
	runEventIngestBoundedRetries = 3
	runEventIngestUnknownRetries = 3
	defer func() {
		runEventIngestNakDelay = origDelay
		runEventIngestBoundedRetries = origBounded
		runEventIngestUnknownRetries = origUnknown
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	suffix := fmt.Sprintf("%d", time.Now().UnixNano())
	stream := "ULTRA_TEST_POISON_" + suffix
	bus, err := NewNATSBus(ctx, NATSConfig{
		URL:                  url,
		Stream:               stream,
		JobsSubject:          "ultra.test." + suffix + ".jobs",
		EventsSubject:        "ultra.test." + suffix + ".events",
		CancelSubject:        "ultra.test." + suffix + ".cancel",
		DataAgentJobsSubject: "ultra.test." + suffix + ".data_agent.jobs",
		EventConsumer:        "ingest-" + suffix,
		EventPartitions:      4,
	})
	if err != nil {
		t.Fatalf("NewNATSBus: %v", err)
	}
	defer bus.Close()
	defer func() { _ = bus.js.DeleteStream(stream) }()

	goodReceived := make(chan struct{}, 1)
	var poisonAttempts int32
	err = bus.SubscribeAllRunEvents(ctx, func(_ context.Context, input domain.AppendRunEventInput) error {
		switch input.EventID {
		case "poison":
			atomic.AddInt32(&poisonAttempts, 1)
			return errors.New("permanent ingest failure")
		case "good":
			select {
			case goodReceived <- struct{}{}:
			default:
			}
			return nil
		}
		return nil
	})
	if err != nil {
		t.Fatalf("SubscribeAllRunEvents: %v", err)
	}

	// Same run_id => same partition: the good event is delivered only after the
	// poison event's slot is freed (MaxAckPending=1).
	runID := "run-poison-" + suffix
	if err := bus.PublishRunEvent(ctx, domain.RunEventRecord{EventID: "poison", RunID: runID, EventKind: "message.delta"}); err != nil {
		t.Fatalf("publish poison: %v", err)
	}
	if err := bus.PublishRunEvent(ctx, domain.RunEventRecord{EventID: "good", RunID: runID, EventKind: "message.delta"}); err != nil {
		t.Fatalf("publish good: %v", err)
	}

	select {
	case <-goodReceived:
		// Partition resumed after the poison event was terminated.
	case <-ctx.Done():
		t.Fatalf("good event never ingested; partition wedged by poison (poison attempts=%d)", atomic.LoadInt32(&poisonAttempts))
	}
	if got := atomic.LoadInt32(&poisonAttempts); got < 3 {
		t.Fatalf("poison attempts=%d, want >=3 (bounded retries before Term)", got)
	}
}

// A3: retention limits must propagate to EXISTING streams (the merge starts
// from the server's config, so without explicit handling MaxAge/MaxBytes
// added to natsStreamConfig would be silently dropped), and must be
// tighten-only (never loosen an operator-tuned limit).
func TestEnsureNATSStreamAppliesRetentionToExistingUnlimitedStream(t *testing.T) {
	t.Parallel()
	manager := fakeNATSStreamManager{
		addErr: nats.ErrStreamNameAlreadyInUse,
		info: &nats.StreamInfo{Config: nats.StreamConfig{
			Name:     "ULTRA_TEST",
			Subjects: []string{"ultra.test.jobs"},
			// Existing stream: unlimited (the live-production shape).
			MaxAge:   0,
			MaxBytes: 0,
		}},
	}
	desired := natsStreamConfig("ULTRA_TEST", []string{"ultra.test.jobs"}, 72*time.Hour, 8<<30)
	if err := ensureNATSStream(context.Background(), &manager, desired); err != nil {
		t.Fatalf("ensureNATSStream: %v", err)
	}
	if manager.updated == nil {
		t.Fatal("expected UpdateStream to be called")
	}
	if manager.updated.MaxAge != 72*time.Hour {
		t.Fatalf("merged MaxAge = %v, want 72h applied to unlimited stream", manager.updated.MaxAge)
	}
	if manager.updated.MaxBytes != 8<<30 {
		t.Fatalf("merged MaxBytes = %d, want 8GiB applied to unlimited stream", manager.updated.MaxBytes)
	}
}

func TestEnsureNATSStreamNeverLoosensOperatorTightenedRetention(t *testing.T) {
	t.Parallel()
	manager := fakeNATSStreamManager{
		addErr: nats.ErrStreamNameAlreadyInUse,
		info: &nats.StreamInfo{Config: nats.StreamConfig{
			Name:     "ULTRA_TEST",
			Subjects: []string{"ultra.test.jobs"},
			// Operator tuned tighter than our defaults.
			MaxAge:   36 * time.Hour,
			MaxBytes: 1 << 30,
		}},
	}
	desired := natsStreamConfig("ULTRA_TEST", []string{"ultra.test.jobs"}, 72*time.Hour, 8<<30)
	if err := ensureNATSStream(context.Background(), &manager, desired); err != nil {
		t.Fatalf("ensureNATSStream: %v", err)
	}
	if manager.updated.MaxAge != 36*time.Hour {
		t.Fatalf("merged MaxAge = %v, want operator's tighter 36h preserved", manager.updated.MaxAge)
	}
	if manager.updated.MaxBytes != 1<<30 {
		t.Fatalf("merged MaxBytes = %d, want operator's tighter 1GiB preserved", manager.updated.MaxBytes)
	}
}

// MaxAge below the 24h duplicate-tracking window would be rejected by the
// server and brick startup; the constructor must clamp it.
func TestNATSStreamConfigClampsMaxAgeToDuplicateWindow(t *testing.T) {
	t.Parallel()
	stream := natsStreamConfig("ULTRA_TEST", []string{"ultra.test.jobs"}, 2*time.Hour, 0)
	if stream.MaxAge != natsDuplicateWindow {
		t.Fatalf("MaxAge = %v, want clamped to duplicate window %v", stream.MaxAge, natsDuplicateWindow)
	}
	if stream.MaxBytes != defaultStreamMaxBytes {
		t.Fatalf("MaxBytes = %d, want default %d", stream.MaxBytes, defaultStreamMaxBytes)
	}
}

// A1: transient (store-outage-class) errors must retry indefinitely and never
// Term — a Postgres restart must not destroy real events. The handler fails
// well past every bounded budget, then heals; the event must still ingest.
func TestNATSBusTransientIngestErrorNeverTerminates(t *testing.T) {
	url := os.Getenv("ULTRA_CONTROL_TEST_NATS_URL")
	if url == "" {
		t.Skip("ULTRA_CONTROL_TEST_NATS_URL is not set")
	}
	origDelay, origBounded, origUnknown := runEventIngestNakDelay, runEventIngestBoundedRetries, runEventIngestUnknownRetries
	runEventIngestNakDelay = 5 * time.Millisecond
	runEventIngestBoundedRetries = 3
	runEventIngestUnknownRetries = 6
	defer func() {
		runEventIngestNakDelay = origDelay
		runEventIngestBoundedRetries = origBounded
		runEventIngestUnknownRetries = origUnknown
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	suffix := fmt.Sprintf("%d", time.Now().UnixNano())
	stream := "ULTRA_TEST_TRANSIENT_" + suffix
	bus, err := NewNATSBus(ctx, NATSConfig{
		URL:                  url,
		Stream:               stream,
		JobsSubject:          "ultra.test." + suffix + ".jobs",
		EventsSubject:        "ultra.test." + suffix + ".events",
		CancelSubject:        "ultra.test." + suffix + ".cancel",
		DataAgentJobsSubject: "ultra.test." + suffix + ".data_agent.jobs",
		EventConsumer:        "ingest-" + suffix,
		EventPartitions:      2,
		IngestErrorClassifier: func(error) IngestErrorClass {
			return IngestErrorTransient
		},
	})
	if err != nil {
		t.Fatalf("NewNATSBus: %v", err)
	}
	defer bus.Close()
	defer func() { _ = bus.js.DeleteStream(stream) }()

	var attempts int32
	ingested := make(chan struct{}, 1)
	// Fail for 20 attempts (>> both budgets of 3 and 6), then heal.
	err = bus.SubscribeAllRunEvents(ctx, func(_ context.Context, input domain.AppendRunEventInput) error {
		if atomic.AddInt32(&attempts, 1) <= 20 {
			return errors.New("store is down")
		}
		select {
		case ingested <- struct{}{}:
		default:
		}
		return nil
	})
	if err != nil {
		t.Fatalf("SubscribeAllRunEvents: %v", err)
	}
	if err := bus.PublishRunEvent(ctx, domain.RunEventRecord{EventID: "evt-transient", RunID: "run-transient-" + suffix, EventKind: "message.delta"}); err != nil {
		t.Fatalf("publish: %v", err)
	}
	select {
	case <-ingested:
		// Survived the outage without Term.
	case <-ctx.Done():
		t.Fatalf("event was never ingested after heal (attempts=%d) — transient error must not Term", atomic.LoadInt32(&attempts))
	}
	if got := atomic.LoadInt32(&attempts); got < 21 {
		t.Fatalf("attempts = %d, want > 20 (retried past every bounded budget)", got)
	}
}

// A1: after the predecessor-pending budget is exhausted, the worker must
// BYPASS the ordering gate (handler sees the bypass ctx flag) and the event
// must be ingested — never terminated.
func TestNATSBusPredecessorPendingBypassesGateAfterBudget(t *testing.T) {
	url := os.Getenv("ULTRA_CONTROL_TEST_NATS_URL")
	if url == "" {
		t.Skip("ULTRA_CONTROL_TEST_NATS_URL is not set")
	}
	origDelay, origBounded := runEventIngestNakDelay, runEventIngestBoundedRetries
	runEventIngestNakDelay = 5 * time.Millisecond
	runEventIngestBoundedRetries = 3
	defer func() {
		runEventIngestNakDelay = origDelay
		runEventIngestBoundedRetries = origBounded
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	suffix := fmt.Sprintf("%d", time.Now().UnixNano())
	stream := "ULTRA_TEST_BYPASS_" + suffix
	pendingErr := errors.New("predecessor pending")
	bus, err := NewNATSBus(ctx, NATSConfig{
		URL:                  url,
		Stream:               stream,
		JobsSubject:          "ultra.test." + suffix + ".jobs",
		EventsSubject:        "ultra.test." + suffix + ".events",
		CancelSubject:        "ultra.test." + suffix + ".cancel",
		DataAgentJobsSubject: "ultra.test." + suffix + ".data_agent.jobs",
		EventConsumer:        "ingest-" + suffix,
		EventPartitions:      2,
		IngestErrorClassifier: func(err error) IngestErrorClass {
			if errors.Is(err, pendingErr) {
				return IngestErrorPredecessorPending
			}
			return IngestErrorUnknown
		},
	})
	if err != nil {
		t.Fatalf("NewNATSBus: %v", err)
	}
	defer bus.Close()
	defer func() { _ = bus.js.DeleteStream(stream) }()

	var gateChecks int32
	bypassed := make(chan struct{}, 1)
	err = bus.SubscribeAllRunEvents(ctx, func(handlerCtx context.Context, input domain.AppendRunEventInput) error {
		if domain.RunEventGateBypassed(handlerCtx) {
			select {
			case bypassed <- struct{}{}:
			default:
			}
			return nil
		}
		atomic.AddInt32(&gateChecks, 1)
		return pendingErr
	})
	if err != nil {
		t.Fatalf("SubscribeAllRunEvents: %v", err)
	}
	if err := bus.PublishRunEvent(ctx, domain.RunEventRecord{EventID: "evt-gap-successor", RunID: "run-bypass-" + suffix, EventKind: "message.delta"}); err != nil {
		t.Fatalf("publish: %v", err)
	}
	select {
	case <-bypassed:
		// Gate bypassed and event ingested instead of terminated.
	case <-ctx.Done():
		t.Fatalf("bypass never happened (gate checks=%d)", atomic.LoadInt32(&gateChecks))
	}
	if got := atomic.LoadInt32(&gateChecks); got < 3 {
		t.Fatalf("gate checks = %d, want >= 3 bounded waits before bypass", got)
	}
}

func TestRunEventPredecessorPendingWaitElapsedUsesEventAge(t *testing.T) {
	origDelay, origBounded := runEventIngestNakDelay, runEventIngestBoundedRetries
	runEventIngestNakDelay = 5 * time.Second
	runEventIngestBoundedRetries = 24
	defer func() {
		runEventIngestNakDelay = origDelay
		runEventIngestBoundedRetries = origBounded
	}()

	now := time.Date(2026, 7, 2, 1, 55, 0, 0, time.UTC)
	aged := queuedRunEventMessage{
		attempts: 1,
		input: domain.AppendRunEventInput{
			EventID:        "evt-aged-gap-successor",
			RunID:          "run-aged-gap",
			SourceSequence: 42,
			TS:             now.Add(-3 * time.Minute),
		},
	}
	if !runEventPredecessorPendingWaitElapsed(aged, now) {
		t.Fatal("aged predecessor-pending event should bypass without spending the full retry budget again")
	}

	fresh := aged
	fresh.input.TS = now.Add(-30 * time.Second)
	if runEventPredecessorPendingWaitElapsed(fresh, now) {
		t.Fatal("fresh predecessor-pending event should wait for the bounded retry budget")
	}

	noTimestamp := aged
	noTimestamp.input.TS = time.Time{}
	noTimestamp.attempts = runEventIngestBoundedRetries
	if !runEventPredecessorPendingWaitElapsed(noTimestamp, now) {
		t.Fatal("events without timestamps should still bypass after the bounded retry budget")
	}
}
