package eventbus

import (
	"context"
	"sync"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

type MemoryBus struct {
	jobs          chan Job
	dataAgentJobs chan DataAgentJob
	cancellations chan CancelSignal
	events        chan domain.RunEventRecord
	mu            sync.RWMutex
	nextSubID     int64
	subscribers   map[int64]memoryRunEventSubscriber
}

type memoryRunEventSubscriber struct {
	runID string
	ch    chan domain.RunEventRecord
}

const (
	runEventSubscriberBuffer = 4096
	jobBufferSize            = 4096
	dataAgentJobBufferSize   = 4096
	cancelBufferSize         = 4096
	legacyEventBufferSize    = 1024
)

func NewMemoryBus() *MemoryBus {
	return &MemoryBus{
		jobs:          make(chan Job, jobBufferSize),
		dataAgentJobs: make(chan DataAgentJob, dataAgentJobBufferSize),
		cancellations: make(chan CancelSignal, cancelBufferSize),
		events:        make(chan domain.RunEventRecord, legacyEventBufferSize),
		subscribers:   map[int64]memoryRunEventSubscriber{},
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

func (b *MemoryBus) PublishDataAgentJob(ctx context.Context, job DataAgentJob) error {
	select {
	case b.dataAgentJobs <- job:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (b *MemoryBus) PublishCancel(ctx context.Context, cancel CancelSignal) error {
	select {
	case b.cancellations <- cancel:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (b *MemoryBus) PublishRunEvent(ctx context.Context, event domain.RunEventRecord) error {
	select {
	case b.events <- event:
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	for _, subscriber := range b.subscribers {
		if subscriber.runID != event.RunID {
			continue
		}
		select {
		case subscriber.ch <- event:
		default:
		}
	}
	return nil
}

func (b *MemoryBus) Jobs() <-chan Job {
	return b.jobs
}

func (b *MemoryBus) DataAgentJobs() <-chan DataAgentJob {
	return b.dataAgentJobs
}

func (b *MemoryBus) Cancellations() <-chan CancelSignal {
	return b.cancellations
}

func (b *MemoryBus) Events() <-chan domain.RunEventRecord {
	return b.events
}

func (b *MemoryBus) SubscribeRunEvents(ctx context.Context, runID string) (<-chan domain.RunEventRecord, func()) {
	ch := make(chan domain.RunEventRecord, runEventSubscriberBuffer)
	b.mu.Lock()
	b.nextSubID++
	id := b.nextSubID
	b.subscribers[id] = memoryRunEventSubscriber{runID: runID, ch: ch}
	b.mu.Unlock()

	var once sync.Once
	unsubscribe := func() {
		once.Do(func() {
			b.mu.Lock()
			delete(b.subscribers, id)
			b.mu.Unlock()
			close(ch)
		})
	}
	go func() {
		<-ctx.Done()
		unsubscribe()
	}()
	return ch, unsubscribe
}
