package eventbus

import (
	"context"
	"errors"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

type SplitBus struct {
	jobs   Bus
	events Bus
}

func NewSplitBus(jobs Bus, events Bus) *SplitBus {
	return &SplitBus{jobs: jobs, events: events}
}

func (b *SplitBus) PublishJob(ctx context.Context, job Job) error {
	return b.jobs.PublishJob(ctx, job)
}

func (b *SplitBus) PublishDataAgentJob(ctx context.Context, job DataAgentJob) error {
	publisher, ok := b.jobs.(DataAgentJobPublisher)
	if !ok {
		return errors.New("data agent job publisher is not configured")
	}
	return publisher.PublishDataAgentJob(ctx, job)
}

func (b *SplitBus) PublishCancel(ctx context.Context, cancel CancelSignal) error {
	return b.jobs.PublishCancel(ctx, cancel)
}

func (b *SplitBus) PublishRunEvent(ctx context.Context, event domain.RunEventRecord) error {
	return b.events.PublishRunEvent(ctx, event)
}
