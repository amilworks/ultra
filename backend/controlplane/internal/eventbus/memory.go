package eventbus

import (
	"context"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

type MemoryBus struct {
	jobs   chan Job
	events chan domain.RunEventRecord
}

func NewMemoryBus() *MemoryBus {
	return &MemoryBus{
		jobs:   make(chan Job, 64),
		events: make(chan domain.RunEventRecord, 1024),
	}
}

func (b *MemoryBus) PublishJob(ctx context.Context, job Job) error {
	select {
	case b.jobs <- job:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (b *MemoryBus) PublishRunEvent(ctx context.Context, event domain.RunEventRecord) error {
	select {
	case b.events <- event:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (b *MemoryBus) Jobs() <-chan Job {
	return b.jobs
}

func (b *MemoryBus) Events() <-chan domain.RunEventRecord {
	return b.events
}
