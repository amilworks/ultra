package eventbus

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/nats-io/nats.go"
)

type NATSConfig struct {
	URL                 string
	Stream              string
	JobsSubject         string
	RareSpotJobsSubject string
	EventsSubject       string
	CancelSubject       string
	EventConsumer       string
	ConsumerTargets     []QueueConsumerTarget
}

type NATSBus struct {
	conn *nats.Conn
	js   nats.JetStreamContext
	cfg  NATSConfig
}

const natsDuplicateWindow = 24 * time.Hour
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
	conn, err := nats.Connect(cfg.URL)
	if err != nil {
		return nil, err
	}
	js, err := conn.JetStream()
	if err != nil {
		conn.Close()
		return nil, err
	}
	subjects := []string{cfg.JobsSubject, cfg.EventsSubject}
	if cfg.CancelSubject != "" {
		subjects = append(subjects, cfg.CancelSubject)
	}
	if cfg.RareSpotJobsSubject != "" {
		subjects = append(subjects, cfg.RareSpotJobsSubject)
	}
	streamConfig := natsStreamConfig(cfg.Stream, subjects)
	_, err = js.AddStream(&streamConfig)
	if err != nil && !errors.Is(err, nats.ErrStreamNameAlreadyInUse) {
		conn.Close()
		return nil, err
	}
	if errors.Is(err, nats.ErrStreamNameAlreadyInUse) {
		_, _ = js.UpdateStream(&streamConfig)
	}
	return &NATSBus{conn: conn, js: js, cfg: cfg}, nil
}

func natsStreamConfig(name string, subjects []string) nats.StreamConfig {
	return nats.StreamConfig{
		Name:       name,
		Subjects:   subjects,
		Storage:    nats.FileStorage,
		Duplicates: natsDuplicateWindow,
	}
}

func (b *NATSBus) PublishJob(ctx context.Context, job Job) error {
	subject := b.cfg.JobsSubject
	if job.WorkflowKind == "rarespot_ecology" && b.cfg.RareSpotJobsSubject != "" {
		subject = b.cfg.RareSpotJobsSubject
	}
	return b.publish(ctx, subject, job, natsMessageIDForJob(job))
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

func (b *NATSBus) SubscribeAllRunEvents(ctx context.Context, handler func(context.Context, domain.AppendRunEventInput) error) error {
	consumer := b.cfg.EventConsumer
	if consumer == "" {
		consumer = "ultra-control-event-ingest"
	}
	if err := b.reconcileRunEventConsumer(ctx, consumer); err != nil {
		return err
	}
	sub, err := b.conn.QueueSubscribe(runEventSubscribeSubject(b.cfg, consumer), consumer, func(msg *nats.Msg) {
		switch runEventMessageDisposition(ctx, msg.Data, handler) {
		case runEventMessageNak:
			_ = msg.Nak()
		default:
			_ = msg.Ack()
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

func (b *NATSBus) Close() {
	b.conn.Drain()
	b.conn.Close()
}

func (b *NATSBus) SubscribeRunEvents(ctx context.Context, runID string) (<-chan domain.RunEventRecord, func()) {
	ch := make(chan domain.RunEventRecord, 128)
	sub, err := b.conn.Subscribe(b.cfg.EventsSubject, func(msg *nats.Msg) {
		var event domain.RunEventRecord
		if err := json.Unmarshal(msg.Data, &event); err != nil {
			return
		}
		if event.RunID != runID {
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
			close(ch)
		})
	}
	go func() {
		<-ctx.Done()
		unsubscribe()
	}()
	return ch, unsubscribe
}
