package eventbus

import (
	"context"
	"encoding/json"
	"errors"
	"hash/fnv"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/nats-io/nats.go"
)

type NATSConfig struct {
	URL                  string
	Stream               string
	JobsSubject          string
	RareSpotJobsSubject  string
	DataAgentJobsSubject string
	EventsSubject        string
	CancelSubject        string
	EventConsumer        string
	ConsumerTargets      []QueueConsumerTarget
	// EventIngestConcurrency is how many run-event ingest workers process
	// events in parallel. Events are partitioned by run ID, so events of the
	// same run are always processed serially and in delivery order; the
	// parallelism applies across runs. Zero applies the default.
	EventIngestConcurrency int
}

type NATSBus struct {
	conn   *nats.Conn
	js     nats.JetStreamContext
	cfg    NATSConfig
	closed chan struct{}
}

const natsDuplicateWindow = 24 * time.Hour
const natsReconnectWait = 2 * time.Second
const natsDrainTimeout = 5 * time.Second
const runEventConsumerAckWait = 60 * time.Second
const runEventConsumerMaxDeliver = 20
const runEventConsumerMaxAckPending = 4096

type runEventMessageAction int

const (
	runEventMessageAck runEventMessageAction = iota
	runEventMessageNak
)

func NewNATSBus(ctx context.Context, cfg NATSConfig) (*NATSBus, error) {
	_ = ctx
	if cfg.CancelSubject == "" {
		cfg.CancelSubject = "ultra.runs.cancel"
	}
	// The connection must outlive any NATS outage: the default MaxReconnects
	// (60 attempts ~2 minutes) permanently closes the connection afterwards,
	// which would silently stop job dispatch and event ingest until restart.
	// Initial connect still fails fast so a misconfigured deployment surfaces
	// at boot instead of limping along.
	closed := make(chan struct{})
	var closeOnce sync.Once
	conn, err := nats.Connect(cfg.URL,
		nats.MaxReconnects(-1),
		nats.ReconnectWait(natsReconnectWait),
		nats.DrainTimeout(natsDrainTimeout),
		nats.DisconnectErrHandler(func(_ *nats.Conn, err error) {
			slog.Warn("nats disconnected; reconnecting", "error", err)
		}),
		nats.ReconnectHandler(func(conn *nats.Conn) {
			slog.Info("nats reconnected", "url", conn.ConnectedUrl())
		}),
		nats.ErrorHandler(func(_ *nats.Conn, sub *nats.Subscription, err error) {
			subject := ""
			if sub != nil {
				subject = sub.Subject
			}
			slog.Error("nats async error", "subject", subject, "error", err)
		}),
		nats.ClosedHandler(func(conn *nats.Conn) {
			if err := conn.LastError(); err != nil {
				slog.Error("nats connection closed", "error", err)
			}
			closeOnce.Do(func() { close(closed) })
		}),
	)
	if err != nil {
		return nil, err
	}
	js, err := conn.JetStream()
	if err != nil {
		conn.Close()
		return nil, err
	}
	streamConfig := natsStreamConfig(cfg.Stream, natsStreamSubjects(cfg))
	if err := ensureNATSStream(ctx, js, streamConfig); err != nil {
		conn.Close()
		return nil, err
	}
	return &NATSBus{conn: conn, js: js, cfg: cfg, closed: closed}, nil
}

func natsStreamConfig(name string, subjects []string) nats.StreamConfig {
	return nats.StreamConfig{
		Name:       name,
		Subjects:   subjects,
		Storage:    nats.FileStorage,
		Duplicates: natsDuplicateWindow,
	}
}

func natsStreamSubjects(cfg NATSConfig) []string {
	subjects := make([]string, 0, 5)
	for _, subject := range []string{
		cfg.JobsSubject,
		cfg.EventsSubject,
		cfg.CancelSubject,
		cfg.RareSpotJobsSubject,
		cfg.DataAgentJobsSubject,
	} {
		if strings.TrimSpace(subject) != "" {
			subjects = append(subjects, subject)
		}
	}
	return subjects
}

type natsStreamManager interface {
	AddStream(*nats.StreamConfig, ...nats.JSOpt) (*nats.StreamInfo, error)
	StreamInfo(string, ...nats.JSOpt) (*nats.StreamInfo, error)
	UpdateStream(*nats.StreamConfig, ...nats.JSOpt) (*nats.StreamInfo, error)
}

func ensureNATSStream(ctx context.Context, manager natsStreamManager, stream nats.StreamConfig) error {
	if _, err := manager.AddStream(&stream, nats.Context(ctx)); err != nil {
		switch {
		case errors.Is(err, nats.ErrStreamNameAlreadyInUse):
		case natsStreamSubjectOverlapError(err):
			if _, infoErr := manager.StreamInfo(stream.Name, nats.Context(ctx)); infoErr != nil {
				return err
			}
		default:
			return err
		}
		if _, updateErr := manager.UpdateStream(&stream, nats.Context(ctx)); updateErr != nil {
			return updateErr
		}
	}
	return nil
}

func natsStreamSubjectOverlapError(err error) bool {
	return err != nil && strings.Contains(strings.ToLower(err.Error()), "subjects overlap with an existing stream")
}

func (b *NATSBus) PublishJob(ctx context.Context, job Job) error {
	subject := b.cfg.JobsSubject
	if job.WorkflowKind == "rarespot_ecology" && b.cfg.RareSpotJobsSubject != "" {
		subject = b.cfg.RareSpotJobsSubject
	}
	return b.publish(ctx, subject, job, natsMessageIDForJob(job))
}

func (b *NATSBus) PublishDataAgentJob(ctx context.Context, job DataAgentJob) error {
	subject := strings.TrimSpace(b.cfg.DataAgentJobsSubject)
	if subject == "" {
		return errors.New("nats data-agent jobs subject is not configured")
	}
	return b.publish(ctx, subject, job, natsMessageIDForDataAgentJob(job))
}

func (b *NATSBus) QueueDiagnostics(ctx context.Context) (QueueDiagnostics, error) {
	diagnostics := QueueDiagnostics{
		Available: false,
		Mode:      "nats_jetstream",
		Stream:    b.cfg.Stream,
	}
	info, err := b.js.StreamInfo(b.cfg.Stream, nats.Context(ctx))
	if err != nil {
		diagnostics.Error = err.Error()
		return diagnostics, err
	}
	diagnostics.Available = true
	diagnostics.StreamSubjects = append([]string(nil), info.Config.Subjects...)
	diagnostics.StreamMessages = info.State.Msgs
	diagnostics.StreamBytes = info.State.Bytes
	diagnostics.FirstSequence = info.State.FirstSeq
	diagnostics.LastSequence = info.State.LastSeq
	diagnostics.ConsumerCount = info.State.Consumers

	for _, target := range b.queueConsumerTargets() {
		diagnostics.Consumers = append(diagnostics.Consumers, b.queueConsumerDiagnostics(ctx, target))
	}
	return diagnostics, nil
}

func (b *NATSBus) queueConsumerTargets() []QueueConsumerTarget {
	if len(b.cfg.ConsumerTargets) > 0 {
		return b.cfg.ConsumerTargets
	}
	return []QueueConsumerTarget{
		{Name: "ultra-deepagents-worker", Role: "deepagents", Subject: b.cfg.JobsSubject},
		{Name: "rarespot-ecology-worker", Role: "rarespot", Subject: b.cfg.RareSpotJobsSubject},
		{Name: "ultra-data-agent-worker", Role: "data_agent", Subject: b.cfg.DataAgentJobsSubject},
		{Name: firstNonEmptyString(b.cfg.EventConsumer, "ultra-control-event-ingest"), Role: "event_ingest", Subject: b.cfg.EventsSubject},
	}
}

func (b *NATSBus) queueConsumerDiagnostics(ctx context.Context, target QueueConsumerTarget) QueueConsumerDiagnostics {
	diagnostics := QueueConsumerDiagnostics{
		Name:    target.Name,
		Role:    target.Role,
		Subject: target.Subject,
	}
	if target.Name == "" {
		diagnostics.Error = "consumer name is empty"
		return diagnostics
	}
	info, err := b.js.ConsumerInfo(b.cfg.Stream, target.Name, nats.Context(ctx))
	if err != nil {
		diagnostics.Error = err.Error()
		return diagnostics
	}
	return queueConsumerDiagnosticsFromInfo(target, info)
}

func queueConsumerDiagnosticsFromInfo(target QueueConsumerTarget, info *nats.ConsumerInfo) QueueConsumerDiagnostics {
	if info == nil {
		return QueueConsumerDiagnostics{
			Name:    target.Name,
			Role:    target.Role,
			Subject: target.Subject,
			Error:   "consumer info is unavailable",
		}
	}
	subject := target.Subject
	if subject == "" {
		subject = info.Config.FilterSubject
	}
	if subject == "" && len(info.Config.FilterSubjects) > 0 {
		subject = strings.Join(info.Config.FilterSubjects, ",")
	}
	name := target.Name
	if name == "" {
		name = info.Name
	}
	return QueueConsumerDiagnostics{
		Name:                    name,
		Role:                    target.Role,
		Subject:                 subject,
		Active:                  true,
		AckWaitSeconds:          info.Config.AckWait.Seconds(),
		MaxDeliver:              info.Config.MaxDeliver,
		PendingMessages:         info.NumPending,
		InFlightMessages:        info.NumAckPending,
		RedeliveredMessages:     info.NumRedelivered,
		WaitingPullRequests:     info.NumWaiting,
		DeliveredStreamSequence: info.Delivered.Stream,
		AckFloorStreamSequence:  info.AckFloor.Stream,
	}
}

func firstNonEmptyString(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func (b *NATSBus) PublishCancel(ctx context.Context, cancel CancelSignal) error {
	subject := b.cfg.CancelSubject
	if subject == "" {
		subject = "ultra.runs.cancel"
	}
	return b.publish(ctx, subject, cancel, natsMessageIDForCancel(cancel))
}

func (b *NATSBus) PublishRunEvent(ctx context.Context, event domain.RunEventRecord) error {
	return b.publish(ctx, b.cfg.EventsSubject, event, natsMessageIDForRunEvent(event))
}

const defaultRunEventIngestConcurrency = 4
const maxRunEventIngestConcurrency = 64

// runEventIngestNakDelay spaces out redeliveries of events whose ingest
// failed (e.g. the store is briefly down). A bare Nak would redeliver at
// full speed and burn through the consumer's MaxDeliver budget in seconds.
const runEventIngestNakDelay = 5 * time.Second

// runEventIngestWorkerQueueDepth bounds how many decoded events may sit in
// front of one worker. With AckWait at 60s a queued message must not wait
// longer than that, so the queue is kept shallow; a full queue blocks the
// subscription dispatcher, which is the desired backpressure.
const runEventIngestWorkerQueueDepth = 64

type queuedRunEventMessage struct {
	msg   *nats.Msg
	input domain.AppendRunEventInput
}

// SubscribeAllRunEvents consumes the durable run-event stream and hands each
// event to handler. Events are decoded once, partitioned by run ID onto a
// fixed pool of workers (per-run order preserved, cross-run parallelism),
// and acked only after the handler finishes so redeliveries cover crashes.
func (b *NATSBus) SubscribeAllRunEvents(ctx context.Context, handler func(context.Context, domain.AppendRunEventInput) error) error {
	consumer := b.cfg.EventConsumer
	if consumer == "" {
		consumer = "ultra-control-event-ingest"
	}
	if err := b.reconcileRunEventConsumer(ctx, consumer); err != nil {
		return err
	}
	concurrency := b.cfg.EventIngestConcurrency
	if concurrency <= 0 {
		concurrency = defaultRunEventIngestConcurrency
	}
	if concurrency > maxRunEventIngestConcurrency {
		concurrency = maxRunEventIngestConcurrency
	}
	queues := make([]chan queuedRunEventMessage, concurrency)
	for index := range queues {
		queues[index] = make(chan queuedRunEventMessage, runEventIngestWorkerQueueDepth)
		go runEventIngestWorker(ctx, queues[index], handler)
	}
	sub, err := b.conn.QueueSubscribe(runEventSubscribeSubject(b.cfg, consumer), consumer, func(msg *nats.Msg) {
		var input domain.AppendRunEventInput
		if err := json.Unmarshal(msg.Data, &input); err != nil {
			// Poison message: acking drops it instead of redelivering forever.
			_ = msg.Ack()
			return
		}
		queue := queues[runEventIngestPartition(input.RunID, len(queues))]
		select {
		case queue <- queuedRunEventMessage{msg: msg, input: input}:
		case <-ctx.Done():
			_ = msg.NakWithDelay(runEventIngestNakDelay)
		}
	})
	if err != nil {
		return err
	}
	go func() {
		<-ctx.Done()
		_ = sub.Unsubscribe()
	}()
	return nil
}

func runEventIngestWorker(ctx context.Context, queue <-chan queuedRunEventMessage, handler func(context.Context, domain.AppendRunEventInput) error) {
	for {
		select {
		case <-ctx.Done():
			// Queued messages stay unacked and redeliver after AckWait;
			// ingest deduplicates them by event ID.
			return
		case item := <-queue:
			switch runEventInputDisposition(ctx, item.input, handler) {
			case runEventMessageNak:
				_ = item.msg.NakWithDelay(runEventIngestNakDelay)
			default:
				_ = item.msg.Ack()
			}
		}
	}
}

func runEventIngestPartition(runID string, partitions int) int {
	if partitions <= 1 {
		return 0
	}
	hash := fnv.New32a()
	_, _ = hash.Write([]byte(runID))
	return int(hash.Sum32() % uint32(partitions))
}

func runEventSubscribeSubject(cfg NATSConfig, consumer string) string {
	return runEventDeliverSubject(cfg.EventsSubject, consumer)
}

func (b *NATSBus) reconcileRunEventConsumer(ctx context.Context, consumer string) error {
	desired := runEventConsumerConfig(b.cfg, consumer)
	info, err := b.js.ConsumerInfo(b.cfg.Stream, consumer, nats.Context(ctx))
	if errors.Is(err, nats.ErrConsumerNotFound) {
		_, err = b.js.AddConsumer(b.cfg.Stream, &desired, nats.Context(ctx))
		return err
	}
	if err != nil {
		return err
	}
	if runEventConsumerConfigMatches(info.Config, desired) {
		return nil
	}
	_, err = b.js.UpdateConsumer(b.cfg.Stream, &desired, nats.Context(ctx))
	if err == nil {
		return nil
	}
	if deleteErr := b.js.DeleteConsumer(b.cfg.Stream, consumer, nats.Context(ctx)); deleteErr != nil {
		return err
	}
	_, err = b.js.AddConsumer(b.cfg.Stream, &desired, nats.Context(ctx))
	return err
}

func runEventConsumerConfig(cfg NATSConfig, consumer string) nats.ConsumerConfig {
	return nats.ConsumerConfig{
		Durable:        consumer,
		DeliverSubject: runEventDeliverSubject(cfg.EventsSubject, consumer),
		DeliverGroup:   consumer,
		DeliverPolicy:  nats.DeliverAllPolicy,
		AckPolicy:      nats.AckExplicitPolicy,
		AckWait:        runEventConsumerAckWait,
		MaxDeliver:     runEventConsumerMaxDeliver,
		FilterSubject:  cfg.EventsSubject,
		ReplayPolicy:   nats.ReplayInstantPolicy,
		MaxAckPending:  runEventConsumerMaxAckPending,
	}
}

func runEventDeliverSubject(eventsSubject string, consumer string) string {
	subject := strings.TrimSpace(eventsSubject)
	if subject == "" {
		subject = "ultra.runs.events"
	}
	cleanConsumer := strings.NewReplacer(".", "-", "*", "-", ">", "-", " ", "-").Replace(strings.TrimSpace(consumer))
	if cleanConsumer == "" {
		cleanConsumer = "ultra-control-event-ingest"
	}
	return subject + ".deliver." + cleanConsumer
}

func runEventConsumerConfigMatches(existing nats.ConsumerConfig, desired nats.ConsumerConfig) bool {
	return existing.Durable == desired.Durable &&
		existing.DeliverSubject == desired.DeliverSubject &&
		existing.DeliverGroup == desired.DeliverGroup &&
		existing.DeliverPolicy == desired.DeliverPolicy &&
		existing.AckPolicy == desired.AckPolicy &&
		existing.AckWait == desired.AckWait &&
		existing.MaxDeliver == desired.MaxDeliver &&
		existing.FilterSubject == desired.FilterSubject &&
		existing.ReplayPolicy == desired.ReplayPolicy &&
		existing.MaxAckPending == desired.MaxAckPending
}

func runEventMessageDisposition(ctx context.Context, data []byte, handler func(context.Context, domain.AppendRunEventInput) error) runEventMessageAction {
	var input domain.AppendRunEventInput
	if err := json.Unmarshal(data, &input); err != nil {
		return runEventMessageAck
	}
	return runEventInputDisposition(ctx, input, handler)
}

func runEventInputDisposition(ctx context.Context, input domain.AppendRunEventInput, handler func(context.Context, domain.AppendRunEventInput) error) runEventMessageAction {
	if err := handler(ctx, input); err != nil {
		return runEventMessageNak
	}
	return runEventMessageAck
}

func (b *NATSBus) publish(ctx context.Context, subject string, value any, messageID string) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	opts := []nats.PubOpt{nats.Context(ctx)}
	if strings.TrimSpace(messageID) != "" {
		opts = append(opts, nats.MsgId(messageID))
	}
	_, err = b.js.Publish(subject, data, opts...)
	return err
}

func natsMessageIDForJob(job Job) string {
	runID := strings.TrimSpace(job.RunID)
	if runID == "" {
		return ""
	}
	dispatchID := strings.TrimSpace(job.DispatchID)
	if dispatchID != "" {
		return "job:" + runID + ":" + dispatchID
	}
	return "job:" + runID
}

func natsMessageIDForDataAgentJob(job DataAgentJob) string {
	jobID := strings.TrimSpace(job.JobID)
	if jobID == "" {
		return ""
	}
	dispatchID := strings.TrimSpace(job.DispatchID)
	if dispatchID != "" {
		return "data-agent-job:" + jobID + ":" + dispatchID
	}
	return "data-agent-job:" + jobID
}

func natsMessageIDForRunEvent(event domain.RunEventRecord) string {
	eventID := strings.TrimSpace(event.EventID)
	if eventID == "" {
		return ""
	}
	return "event:" + eventID
}

func natsMessageIDForCancel(cancel CancelSignal) string {
	runID := strings.TrimSpace(cancel.RunID)
	if runID == "" {
		return ""
	}
	reason := strings.TrimSpace(cancel.Reason)
	if reason == "" {
		return "cancel:" + runID
	}
	return "cancel:" + runID + ":" + reason
}

// Close drains the connection (letting in-flight subscription callbacks and
// pending messages finish) and waits for the close to complete. The previous
// Drain-then-immediate-Close aborted the drain before it did anything.
func (b *NATSBus) Close() {
	if err := b.conn.Drain(); err != nil {
		b.conn.Close()
	}
	select {
	case <-b.closed:
	case <-time.After(natsDrainTimeout + time.Second):
		b.conn.Close()
		select {
		case <-b.closed:
		case <-time.After(time.Second):
			slog.Warn("nats connection did not confirm close before shutdown")
		}
	}
}

func (b *NATSBus) SubscribeRunEvents(ctx context.Context, runID string) (<-chan domain.RunEventRecord, func()) {
	ch := make(chan domain.RunEventRecord, 128)
	// The NATS message callback runs on the client's dispatcher goroutine,
	// independently of unsubscribe(). Without synchronization, a message dispatched
	// just as a client disconnects (ctx-cancel -> unsubscribe -> close(ch)) would race
	// the callback's send and panic with "send on closed channel" — crashing the whole
	// process, not just one stream. Guard the send and the close with a mutex + flag so
	// the callback never sends after the channel is closed. (sub.Unsubscribe() does not
	// wait for an in-flight callback, so the flag, not Unsubscribe alone, is what makes
	// this safe.)
	var (
		mu     sync.Mutex
		closed bool
	)
	sub, err := b.conn.Subscribe(b.cfg.EventsSubject, func(msg *nats.Msg) {
		var event domain.RunEventRecord
		if err := json.Unmarshal(msg.Data, &event); err != nil {
			return
		}
		if event.RunID != runID {
			return
		}
		mu.Lock()
		defer mu.Unlock()
		if closed {
			return
		}
		select {
		case ch <- event:
		default:
		}
	})
	if err != nil {
		close(ch)
		return ch, func() {}
	}

	var once sync.Once
	unsubscribe := func() {
		once.Do(func() {
			_ = sub.Unsubscribe()
			mu.Lock()
			closed = true
			close(ch)
			mu.Unlock()
		})
	}
	go func() {
		<-ctx.Done()
		unsubscribe()
	}()
	return ch, unsubscribe
}
