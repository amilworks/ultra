package eventbus

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/nats-io/nats.go"
)

type NATSConfig struct {
	URL           string
	Stream        string
	JobsSubject   string
	EventsSubject string
}

type NATSBus struct {
	conn *nats.Conn
	js   nats.JetStreamContext
	cfg  NATSConfig
}

func NewNATSBus(ctx context.Context, cfg NATSConfig) (*NATSBus, error) {
	_ = ctx
	conn, err := nats.Connect(cfg.URL)
	if err != nil {
		return nil, err
	}
	js, err := conn.JetStream()
	if err != nil {
		conn.Close()
		return nil, err
	}
	_, err = js.AddStream(&nats.StreamConfig{
		Name:     cfg.Stream,
		Subjects: []string{cfg.JobsSubject, cfg.EventsSubject},
		Storage:  nats.FileStorage,
	})
	if err != nil && !errors.Is(err, nats.ErrStreamNameAlreadyInUse) {
		conn.Close()
		return nil, err
	}
	return &NATSBus{conn: conn, js: js, cfg: cfg}, nil
}

func (b *NATSBus) PublishJob(ctx context.Context, job Job) error {
	return b.publish(ctx, b.cfg.JobsSubject, job)
}

func (b *NATSBus) PublishRunEvent(ctx context.Context, event domain.RunEventRecord) error {
	return b.publish(ctx, b.cfg.EventsSubject, event)
}

func (b *NATSBus) publish(ctx context.Context, subject string, value any) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	_, err = b.js.Publish(subject, data, nats.Context(ctx))
	return err
}

func (b *NATSBus) Close() {
	b.conn.Drain()
	b.conn.Close()
}
