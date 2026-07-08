package eventbus

import (
	"context"
	"encoding/json"
	"errors"
	"hash/fnv"
	"log/slog"
	"strconv"
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
	DataAgentJobsSubject string
	TrainingJobsSubject  string
	EventsSubject        string
	CancelSubject        string
	EventConsumer        string
	// StreamMaxAge / StreamMaxBytes bound the JetStream stream (0 applies the
	// defaults). MaxAge is clamped to at least the duplicate window.
	StreamMaxAge   time.Duration
	StreamMaxBytes int64
	// EventPartitions is the number of broker-visible run-event partitions.
	// Each partition gets its own durable push consumer with one in-flight
	// message, preserving same-partition order across control-plane replicas.
	// Zero applies the default.
	EventPartitions int
	ConsumerTargets []QueueConsumerTarget
	// EventIngestConcurrency is how many run-event ingest workers process
	// events in parallel. Events are partitioned by run ID, so events of the
	// same run are always processed serially and in delivery order; the
	// parallelism applies across runs. Zero applies the default.
	EventIngestConcurrency int
	// IngestErrorClassifier maps a handler error to a retry class so the
	// ingest worker can pick the right budget (transient store outages must
	// never terminate real events; deterministic poison must not wedge the
	// partition). Nil treats every error as IngestErrorUnknown.
	IngestErrorClassifier IngestErrorClassifier
}

// IngestErrorClass selects the retry budget for a failed run-event ingest.
type IngestErrorClass int

const (
	// IngestErrorUnknown: unclassified failures get a generous ceiling
	// (better to stall a partition for an hour than silently lose data),
	// then terminate.
	IngestErrorUnknown IngestErrorClass = iota
	// IngestErrorTransient: infrastructure will heal (store restart,
	// failover). Retry indefinitely; never terminate.
	IngestErrorTransient
	// IngestErrorDeterministic: the event itself is unprocessable; short
	// bounded retry then terminate to keep the partition flowing.
	IngestErrorDeterministic
	// IngestErrorPredecessorPending: the ordering gate is waiting for an
	// earlier source_sequence. Bounded wait, then BYPASS the gate and append
	// with a gap instead of terminating (readers order by the store-assigned
	// sequence_number, so a gap is harmless; a terminated event is lost).
	IngestErrorPredecessorPending
)

type IngestErrorClassifier func(error) IngestErrorClass

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
const runEventConsumerMaxAckPending = 1
const defaultRunEventPartitions = 64
const maxRunEventPartitions = 256

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
	streamConfig := natsStreamConfig(cfg.Stream, natsStreamSubjects(cfg), cfg.StreamMaxAge, cfg.StreamMaxBytes)
	if err := ensureNATSStream(ctx, js, streamConfig); err != nil {
		conn.Close()
		return nil, err
	}
	return &NATSBus{conn: conn, js: js, cfg: cfg, closed: closed}, nil
}

// defaultStreamMaxAge / defaultStreamMaxBytes bound the run-event stream.
// Events are durably archived in Postgres at ingest; the stream is transport,
// not the archive, so retention only needs to exceed the worst redelivery
// horizon (AckWait 60s, retry backstops, plus generous control-plane downtime
// tolerance). Without limits the file-backed stream grows forever and disk
// exhaustion halts all of JetStream.
const defaultStreamMaxAge = 72 * time.Hour
const defaultStreamMaxBytes = int64(8) << 30 // 8 GiB

func natsStreamConfig(name string, subjects []string, maxAge time.Duration, maxBytes int64) nats.StreamConfig {
	if maxAge <= 0 {
		maxAge = defaultStreamMaxAge
	}
	// A MaxAge below the duplicate-tracking window is rejected by the server
	// ("duplicates window can not be larger than max age") and would brick
	// startup on both the create and update paths; clamp defensively.
	if maxAge < natsDuplicateWindow {
		maxAge = natsDuplicateWindow
	}
	if maxBytes <= 0 {
		maxBytes = defaultStreamMaxBytes
	}
	return nats.StreamConfig{
		Name:       name,
		Subjects:   subjects,
		Storage:    nats.FileStorage,
		Duplicates: natsDuplicateWindow,
		MaxAge:     maxAge,
		MaxBytes:   maxBytes,
	}
}

func natsStreamSubjects(cfg NATSConfig) []string {
	subjects := make([]string, 0, 6)
	for _, subject := range []string{
		cfg.JobsSubject,
		cfg.EventsSubject,
		runEventPartitionWildcard(cfg.EventsSubject),
		cfg.CancelSubject,
		cfg.DataAgentJobsSubject,
		cfg.TrainingJobsSubject,
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
		var info *nats.StreamInfo
		switch {
		case errors.Is(err, nats.ErrStreamNameAlreadyInUse):
		case natsStreamSubjectOverlapError(err):
			var infoErr error
			info, infoErr = manager.StreamInfo(stream.Name, nats.Context(ctx))
			if infoErr != nil {
				return err
			}
		default:
			return err
		}
		if info == nil {
			var infoErr error
			info, infoErr = manager.StreamInfo(stream.Name, nats.Context(ctx))
			if infoErr != nil {
				return infoErr
			}
		}
		if info != nil {
			updated := info.Config
			updated.Subjects = mergeNATSStreamSubjects(info.Config.Subjects, stream.Subjects)
			if stream.Duplicates > updated.Duplicates {
				updated.Duplicates = stream.Duplicates
			}
			// Tighten-only retention merge: apply our MaxAge/MaxBytes when the
			// existing stream is unlimited or looser, but never LOOSEN a limit
			// an operator tuned tighter on the server. Without this, limits
			// added to natsStreamConfig are silently dropped for existing
			// streams (updated starts from info.Config, and only Subjects and
			// Duplicates were overridden).
			if stream.MaxAge > 0 && (updated.MaxAge <= 0 || stream.MaxAge < updated.MaxAge) {
				updated.MaxAge = stream.MaxAge
			}
			if stream.MaxBytes > 0 && (updated.MaxBytes <= 0 || stream.MaxBytes < updated.MaxBytes) {
				updated.MaxBytes = stream.MaxBytes
			}
			stream = updated
		}
		if _, updateErr := manager.UpdateStream(&stream, nats.Context(ctx)); updateErr != nil {
			return updateErr
		}
	}
	return nil
}

func mergeNATSStreamSubjects(existing []string, requested []string) []string {
	subjects := make([]string, 0, len(existing)+len(requested))
	seen := make(map[string]struct{}, len(existing)+len(requested))
	for _, subject := range append(existing, requested...) {
		subject = strings.TrimSpace(subject)
		if subject == "" {
			continue
		}
		if _, ok := seen[subject]; ok {
			continue
		}
		seen[subject] = struct{}{}
		subjects = append(subjects, subject)
	}
	return subjects
}

func natsStreamSubjectOverlapError(err error) bool {
	return err != nil && strings.Contains(strings.ToLower(err.Error()), "subjects overlap with an existing stream")
}

func (b *NATSBus) PublishJob(ctx context.Context, job Job) error {
	return b.publish(ctx, b.cfg.JobsSubject, job, natsMessageIDForJob(job))
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
		return b.expandRunEventQueueConsumerTargets(b.cfg.ConsumerTargets)
	}
	return b.expandRunEventQueueConsumerTargets([]QueueConsumerTarget{
		{Name: "ultra-deepagents-worker", Role: "deepagents", Subject: b.cfg.JobsSubject},
		{Name: "ultra-data-agent-worker", Role: "data_agent", Subject: b.cfg.DataAgentJobsSubject},
		{Name: firstNonEmptyString(b.cfg.EventConsumer, "ultra-control-event-ingest"), Role: "event_ingest", Subject: b.cfg.EventsSubject},
	})
}

func (b *NATSBus) expandRunEventQueueConsumerTargets(targets []QueueConsumerTarget) []QueueConsumerTarget {
	eventConsumer := firstNonEmptyString(b.cfg.EventConsumer, "ultra-control-event-ingest")
	eventSubject := runEventBaseSubject(b.cfg.EventsSubject)
	expanded := make([]QueueConsumerTarget, 0, len(targets)+runEventPartitionCount(b.cfg))
	for _, target := range targets {
		expanded = append(expanded, target)
		if target.Name != eventConsumer && target.Role != "event_ingest" {
			continue
		}
		for partition := 0; partition < runEventPartitionCount(b.cfg); partition++ {
			expanded = append(expanded, QueueConsumerTarget{
				Name:    runEventPartitionConsumerName(eventConsumer, partition),
				Role:    "event_ingest_partition",
				Subject: runEventPartitionSubjectForIndex(eventSubject, partition),
			})
		}
	}
	return expanded
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
		Active:                  queueConsumerActiveFromInfo(info),
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

func queueConsumerActiveFromInfo(info *nats.ConsumerInfo) bool {
	if info == nil {
		return false
	}
	if strings.TrimSpace(info.Config.DeliverSubject) != "" {
		return info.PushBound
	}
	return info.NumWaiting > 0 || info.NumAckPending > 0
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
	subject := runEventPartitionSubject(b.cfg.EventsSubject, event.RunID, runEventPartitionCount(b.cfg))
	return b.publish(ctx, subject, event, natsMessageIDForRunEvent(event))
}

const defaultRunEventIngestConcurrency = 4
const maxRunEventIngestConcurrency = 64

// runEventIngestNakDelay spaces out in-place retries of events whose ingest
// failed (e.g. the store is briefly down). Strict partition consumers keep the
// message in progress (msg.InProgress) instead of NAKing, so later
// same-partition messages are not delivered before the earlier event succeeds.
// It is a var only so tests can shrink it; production keeps 5s.
var runEventIngestNakDelay = 5 * time.Second

// Retry budgets by error class. msg.InProgress() sends +WPI, which resets
// AckWait without acknowledging and never counts toward MaxDeliver, so these
// loops are the ONLY bound on a failing event. Class semantics:
//   - deterministic / predecessor-pending: ~2 min (24 x 5s), then Term or
//     gate-bypass respectively;
//   - unknown: ~1 h ceiling, then Term (generous because terminating loses
//     the event permanently — Postgres is the archive);
//   - transient: unbounded — the store will heal, and with MaxAckPending=1
//     nothing else progresses during the outage anyway, so terminating real
//     events would trade durability for nothing.
//
// Vars only so tests can shrink them.
var runEventIngestBoundedRetries = 24
var runEventIngestUnknownRetries = 720

// runEventIngestWorkerQueueInitialCapacity seeds each partition FIFO. The FIFO
// can grow while one run is slow because blocking the NATS callback on a single
// partition would starve unrelated runs that could otherwise ingest safely.
// JetStream MaxAckPending still bounds how many unacked messages NATS delivers.
const runEventIngestWorkerQueueInitialCapacity = 64

type queuedRunEventMessage struct {
	msg      *nats.Msg
	input    domain.AppendRunEventInput
	attempts int
	// bypassGate is set after the predecessor-pending budget is exhausted:
	// the next handler invocation skips the source_sequence ordering gate and
	// appends with a gap instead of terminating the event.
	bypassGate bool
}

type runEventIngestQueue struct {
	mu     sync.Mutex
	notify chan struct{}
	items  []queuedRunEventMessage
}

func newRunEventIngestQueue() *runEventIngestQueue {
	return &runEventIngestQueue{
		notify: make(chan struct{}, 1),
		items:  make([]queuedRunEventMessage, 0, runEventIngestWorkerQueueInitialCapacity),
	}
}

func (q *runEventIngestQueue) push(item queuedRunEventMessage) {
	q.mu.Lock()
	q.items = append(q.items, item)
	q.mu.Unlock()
	select {
	case q.notify <- struct{}{}:
	default:
	}
}

func (q *runEventIngestQueue) pop(ctx context.Context) (queuedRunEventMessage, bool) {
	for {
		q.mu.Lock()
		if len(q.items) > 0 {
			item := q.items[0]
			var zero queuedRunEventMessage
			q.items[0] = zero
			q.items = q.items[1:]
			q.mu.Unlock()
			return item, true
		}
		q.mu.Unlock()
		select {
		case <-ctx.Done():
			return queuedRunEventMessage{}, false
		case <-q.notify:
		}
	}
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
	subs := make([]*nats.Subscription, 0, runEventPartitionCount(b.cfg)+1)
	subscribe := func(consumerName string, filterSubject string, queue *runEventIngestQueue) error {
		if err := b.reconcileRunEventConsumer(ctx, consumerName, filterSubject); err != nil {
			return err
		}
		sub, err := b.conn.QueueSubscribe(runEventSubscribeSubject(b.cfg, consumerName), consumerName, func(msg *nats.Msg) {
			var input domain.AppendRunEventInput
			if err := json.Unmarshal(msg.Data, &input); err != nil {
				// Poison message: acking drops it instead of redelivering forever.
				_ = msg.Ack()
				return
			}
			if ctx.Err() != nil {
				_ = msg.InProgress()
				return
			}
			queue.push(queuedRunEventMessage{msg: msg, input: input})
		})
		if err != nil {
			return err
		}
		subs = append(subs, sub)
		return nil
	}

	legacyQueue := newRunEventIngestQueue()
	go runEventIngestWorker(ctx, legacyQueue, handler, b.cfg.IngestErrorClassifier)
	if err := subscribe(consumer, runEventBaseSubject(b.cfg.EventsSubject), legacyQueue); err != nil {
		return err
	}

	partitions := runEventPartitionCount(b.cfg)
	for partition := 0; partition < partitions; partition++ {
		queue := newRunEventIngestQueue()
		go runEventIngestWorker(ctx, queue, handler, b.cfg.IngestErrorClassifier)
		partitionConsumer := runEventPartitionConsumerName(consumer, partition)
		partitionSubject := runEventPartitionSubjectForIndex(b.cfg.EventsSubject, partition)
		if err := subscribe(partitionConsumer, partitionSubject, queue); err != nil {
			for _, sub := range subs {
				_ = sub.Unsubscribe()
			}
			return err
		}
	}
	go func() {
		<-ctx.Done()
		for _, sub := range subs {
			_ = sub.Unsubscribe()
		}
	}()
	return nil
}

func runEventIngestWorker(ctx context.Context, queue *runEventIngestQueue, handler func(context.Context, domain.AppendRunEventInput) error, classify IngestErrorClassifier) {
	for {
		item, ok := queue.pop(ctx)
		if !ok {
			// Queued messages stay unacked and redeliver after AckWait; ingest
			// deduplicates them by event ID.
			return
		}
		handlerCtx := ctx
		if item.bypassGate {
			handlerCtx = domain.WithRunEventGateBypass(ctx)
		}
		action, handlerErr := runEventInputDisposition(handlerCtx, item.input, handler)
		switch action {
		case runEventMessageNak:
			// Shutdown guard: an error during teardown (cancelled contexts,
			// closing pools) must never consume retry budget or terminate the
			// message. Leaving it unacked hands it to AckWait redelivery,
			// where event-id dedup makes the retry safe.
			if ctx.Err() != nil {
				return
			}
			item.attempts++
			class := IngestErrorUnknown
			if classify != nil {
				class = classify(handlerErr)
			}
			logRunEventIngestRetry(item, class, handlerErr)
			switch class {
			case IngestErrorTransient:
				// Unbounded: infrastructure heals; terminating loses data.
			case IngestErrorPredecessorPending:
				now := time.Now()
				if runEventPredecessorPendingWaitElapsed(item, now) {
					// The predecessor never arrived: a permanent producer-side
					// gap. Bypass the ordering gate and store the event with a
					// gap rather than losing it; reset the budget so the
					// bypassed append gets a full retry window of its own.
					slog.Warn("run event predecessor never arrived; bypassing ordering gate",
						"run_id", item.input.RunID,
						"event_id", item.input.EventID,
						"source_sequence", item.input.SourceSequence,
						"waited", runEventPredecessorPendingWait(item, now))
					item.bypassGate = true
					item.attempts = 0
					_ = item.msg.InProgress()
					queue.push(item)
					continue
				}
			default: // IngestErrorDeterministic, IngestErrorUnknown
				budget := runEventIngestBoundedRetries
				if class == IngestErrorUnknown {
					budget = runEventIngestUnknownRetries
				}
				if item.attempts >= budget {
					// Backstop: Term() acknowledges (+TERM) and stops
					// redelivery, freeing the partition's MaxAckPending=1
					// slot so one poison message cannot starve every run
					// behind it.
					_ = item.msg.Term()
					slog.Error("run event ingest gave up; terminated message to unblock partition",
						"run_id", item.input.RunID,
						"event_id", item.input.EventID,
						"event_kind", item.input.EventKind,
						"source_sequence", item.input.SourceSequence,
						"attempts", item.attempts,
						"error", handlerErr)
					continue
				}
			}
			_ = item.msg.InProgress()
			select {
			case <-ctx.Done():
				return
			case <-time.After(runEventIngestNakDelay):
				queue.push(item)
			}
		default:
			_ = item.msg.Ack()
		}
	}
}

func runEventPredecessorPendingWaitElapsed(item queuedRunEventMessage, now time.Time) bool {
	if item.attempts >= runEventIngestBoundedRetries {
		return true
	}
	if item.input.TS.IsZero() {
		return false
	}
	requiredWait := time.Duration(runEventIngestBoundedRetries) * runEventIngestNakDelay
	if requiredWait <= 0 {
		return true
	}
	if item.input.TS.After(now) {
		return false
	}
	return now.Sub(item.input.TS) >= requiredWait
}

func runEventPredecessorPendingWait(item queuedRunEventMessage, now time.Time) time.Duration {
	attemptWait := time.Duration(item.attempts) * runEventIngestNakDelay
	if item.input.TS.IsZero() || item.input.TS.After(now) {
		return attemptWait
	}
	eventAge := now.Sub(item.input.TS)
	if eventAge > attemptWait {
		return eventAge
	}
	return attemptWait
}

// logRunEventIngestRetry logs the first failure immediately, then once per
// ~minute, so a store outage is visible without flooding.
func logRunEventIngestRetry(item queuedRunEventMessage, class IngestErrorClass, err error) {
	if item.attempts != 1 && item.attempts%12 != 0 {
		return
	}
	slog.Warn("run event ingest retrying",
		"run_id", item.input.RunID,
		"event_id", item.input.EventID,
		"event_kind", item.input.EventKind,
		"source_sequence", item.input.SourceSequence,
		"attempts", item.attempts,
		"class", class,
		"bypass_gate", item.bypassGate,
		"error", err)
}

func runEventIngestPartition(runID string, partitions int) int {
	if partitions <= 1 {
		return 0
	}
	hash := fnv.New32a()
	_, _ = hash.Write([]byte(runID))
	return int(hash.Sum32() % uint32(partitions))
}

func runEventPartitionCount(cfg NATSConfig) int {
	partitions := cfg.EventPartitions
	if partitions <= 0 {
		partitions = defaultRunEventPartitions
	}
	if partitions > maxRunEventPartitions {
		return maxRunEventPartitions
	}
	return partitions
}

func runEventBaseSubject(eventsSubject string) string {
	subject := strings.TrimSpace(eventsSubject)
	if subject == "" {
		return "ultra.runs.events"
	}
	return subject
}

func runEventPartitionWildcard(eventsSubject string) string {
	return runEventBaseSubject(eventsSubject) + ".p.*"
}

func runEventPartitionSubject(eventsSubject string, runID string, partitions int) string {
	return runEventPartitionSubjectForIndex(eventsSubject, runEventIngestPartition(runID, partitions))
}

func runEventPartitionSubjectForIndex(eventsSubject string, partition int) string {
	return runEventBaseSubject(eventsSubject) + ".p." + strconv.Itoa(partition)
}

func runEventPartitionConsumerName(consumer string, partition int) string {
	base := strings.TrimSpace(consumer)
	if base == "" {
		base = "ultra-control-event-ingest"
	}
	return base + "-p-" + strconv.Itoa(partition)
}

func runEventSubscribeSubject(cfg NATSConfig, consumer string) string {
	return runEventDeliverSubject(cfg.EventsSubject, consumer)
}

func (b *NATSBus) reconcileRunEventConsumer(ctx context.Context, consumer string, filterSubject string) error {
	desired := runEventConsumerConfigForSubject(b.cfg, consumer, filterSubject)
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
	return runEventConsumerConfigForSubject(cfg, consumer, runEventBaseSubject(cfg.EventsSubject))
}

func runEventConsumerConfigForSubject(cfg NATSConfig, consumer string, filterSubject string) nats.ConsumerConfig {
	return nats.ConsumerConfig{
		Durable:        consumer,
		DeliverSubject: runEventDeliverSubject(cfg.EventsSubject, consumer),
		DeliverGroup:   consumer,
		DeliverPolicy:  nats.DeliverAllPolicy,
		AckPolicy:      nats.AckExplicitPolicy,
		AckWait:        runEventConsumerAckWait,
		MaxDeliver:     runEventConsumerMaxDeliver,
		FilterSubject:  runEventBaseSubject(filterSubject),
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
	action, _ := runEventInputDisposition(ctx, input, handler)
	return action
}

func runEventInputDisposition(ctx context.Context, input domain.AppendRunEventInput, handler func(context.Context, domain.AppendRunEventInput) error) (runEventMessageAction, error) {
	if err := handler(ctx, input); err != nil {
		return runEventMessageNak, err
	}
	return runEventMessageAck, nil
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
	callback := func(msg *nats.Msg) {
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
	}
	exactSub, exactErr := b.conn.Subscribe(runEventBaseSubject(b.cfg.EventsSubject), callback)
	partitionSub, partitionErr := b.conn.Subscribe(runEventPartitionWildcard(b.cfg.EventsSubject), callback)
	if exactErr != nil && partitionErr != nil {
		close(ch)
		return ch, func() {}
	}

	var once sync.Once
	unsubscribe := func() {
		once.Do(func() {
			if exactSub != nil {
				_ = exactSub.Unsubscribe()
			}
			if partitionSub != nil {
				_ = partitionSub.Unsubscribe()
			}
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
