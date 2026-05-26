package store

import (
	"context"
	"errors"
	"sort"
	"sync"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

var ErrNotFound = errors.New("not found")

type MemoryStore struct {
	mu        sync.RWMutex
	threads   map[string]domain.ThreadRecord
	messages  map[string][]domain.ThreadMessage
	runs      map[string]domain.RunRecord
	events    map[string][]domain.RunEventRecord
	artifacts map[string]domain.ArtifactRecord
}

func NewMemoryStore() *MemoryStore {
	return &MemoryStore{
		threads:   map[string]domain.ThreadRecord{},
		messages:  map[string][]domain.ThreadMessage{},
		runs:      map[string]domain.RunRecord{},
		events:    map[string][]domain.RunEventRecord{},
		artifacts: map[string]domain.ArtifactRecord{},
	}
}

func (s *MemoryStore) CreateThread(ctx context.Context, input domain.CreateThreadInput) (domain.ThreadRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()

	now := domain.Now()
	thread := domain.ThreadRecord{
		ThreadID:  domain.NewID("thread"),
		UserID:    input.UserID,
		Title:     input.Title,
		Status:    domain.ThreadStatusActive,
		CreatedAt: now,
		UpdatedAt: now,
		Metadata:  mapOrEmpty(input.Metadata),
	}
	s.threads[thread.ThreadID] = thread
	for _, msg := range input.InitialMessages {
		msg.MessageID = domain.NewID("msg")
		msg.ThreadID = thread.ThreadID
		msg.CreatedAt = now
		msg.Metadata = mapOrEmpty(msg.Metadata)
		s.messages[thread.ThreadID] = append(s.messages[thread.ThreadID], msg)
	}
	return thread, nil
}

func (s *MemoryStore) GetThread(ctx context.Context, threadID string) (domain.ThreadRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	thread, ok := s.threads[threadID]
	if !ok {
		return domain.ThreadRecord{}, ErrNotFound
	}
	return thread, nil
}

func (s *MemoryStore) ListThreads(ctx context.Context, limit int) ([]domain.ThreadRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	threads := make([]domain.ThreadRecord, 0, len(s.threads))
	for _, thread := range s.threads {
		threads = append(threads, thread)
	}
	sort.Slice(threads, func(i, j int) bool {
		return threads[i].UpdatedAt.After(threads[j].UpdatedAt)
	})
	return take(threads, limit), nil
}

func (s *MemoryStore) ListThreadMessages(ctx context.Context, threadID string) ([]domain.ThreadMessage, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	messages := append([]domain.ThreadMessage(nil), s.messages[threadID]...)
	return messages, nil
}

func (s *MemoryStore) CreateRun(ctx context.Context, input domain.CreateRunInput) (domain.RunRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.threads[input.ThreadID]; !ok {
		return domain.RunRecord{}, ErrNotFound
	}
	now := domain.Now()
	run := domain.RunRecord{
		RunID:        domain.NewID("run"),
		ThreadID:     input.ThreadID,
		UserID:       input.UserID,
		Goal:         input.Goal,
		Status:       domain.RunStatusQueued,
		WorkflowKind: "deep_agents",
		Mode:         "durable",
		CreatedAt:    now,
		UpdatedAt:    now,
		Metadata:     mapOrEmpty(input.Metadata),
	}
	s.runs[run.RunID] = run
	thread := s.threads[input.ThreadID]
	thread.LatestRunID = run.RunID
	thread.UpdatedAt = now
	s.threads[input.ThreadID] = thread
	for _, msg := range input.Messages {
		msg.MessageID = domain.NewID("msg")
		msg.ThreadID = input.ThreadID
		msg.RunID = run.RunID
		msg.CreatedAt = now
		msg.Metadata = mapOrEmpty(msg.Metadata)
		s.messages[input.ThreadID] = append(s.messages[input.ThreadID], msg)
	}
	return run, nil
}

func (s *MemoryStore) GetRun(ctx context.Context, runID string) (domain.RunRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	run, ok := s.runs[runID]
	if !ok {
		return domain.RunRecord{}, ErrNotFound
	}
	return run, nil
}

func (s *MemoryStore) UpdateRunStatus(ctx context.Context, runID string, status domain.RunStatus, responseText string, errorText string) (domain.RunRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	run, ok := s.runs[runID]
	if !ok {
		return domain.RunRecord{}, ErrNotFound
	}
	now := domain.Now()
	run.Status = status
	run.ResponseText = responseText
	run.Error = errorText
	run.UpdatedAt = now
	if status == domain.RunStatusRunning && run.StartedAt == nil {
		run.StartedAt = &now
	}
	if status == domain.RunStatusSucceeded || status == domain.RunStatusFailed || status == domain.RunStatusCanceled {
		run.CompletedAt = &now
	}
	s.runs[runID] = run
	return run, nil
}

func (s *MemoryStore) AppendRunEvent(ctx context.Context, input domain.AppendRunEventInput) (domain.RunEventRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.runs[input.RunID]; !ok {
		return domain.RunEventRecord{}, ErrNotFound
	}
	seq := int64(len(s.events[input.RunID]) + 1)
	event := domain.RunEventRecord{
		EventID:   domain.NewID("event"),
		Sequence:  seq,
		RunID:     input.RunID,
		ThreadID:  input.ThreadID,
		EventKind: input.EventKind,
		TS:        domain.Now(),
		Message:   input.Message,
		Payload:   mapOrEmpty(input.Payload),
	}
	s.events[input.RunID] = append(s.events[input.RunID], event)
	return event, nil
}

func (s *MemoryStore) ListRunEvents(ctx context.Context, runID string, limit int) ([]domain.RunEventRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	events := append([]domain.RunEventRecord(nil), s.events[runID]...)
	if limit > 0 && len(events) > limit {
		events = events[len(events)-limit:]
	}
	return events, nil
}

func (s *MemoryStore) CreateArtifact(ctx context.Context, input domain.CreateArtifactInput) (domain.ArtifactRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.runs[input.RunID]; !ok {
		return domain.ArtifactRecord{}, ErrNotFound
	}
	now := domain.Now()
	artifact := domain.ArtifactRecord{
		ArtifactID: domain.NewID("artifact"),
		RunID:      input.RunID,
		ThreadID:   input.ThreadID,
		Kind:       input.Kind,
		Path:       input.Path,
		Title:      input.Title,
		MimeType:   input.MimeType,
		SizeBytes:  input.SizeBytes,
		SHA256:     input.SHA256,
		StorageURI: input.StorageURI,
		CreatedAt:  now,
		UpdatedAt:  now,
		Metadata:   mapOrEmpty(input.Metadata),
	}
	s.artifacts[artifact.ArtifactID] = artifact
	return artifact, nil
}

func (s *MemoryStore) ListRunArtifacts(ctx context.Context, runID string, limit int) ([]domain.ArtifactRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	artifacts := []domain.ArtifactRecord{}
	for _, artifact := range s.artifacts {
		if artifact.RunID == runID {
			artifacts = append(artifacts, artifact)
		}
	}
	sort.Slice(artifacts, func(i, j int) bool {
		return artifacts[i].CreatedAt.After(artifacts[j].CreatedAt)
	})
	return take(artifacts, limit), nil
}

func (s *MemoryStore) GetArtifact(ctx context.Context, artifactID string) (domain.ArtifactRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	artifact, ok := s.artifacts[artifactID]
	if !ok {
		return domain.ArtifactRecord{}, ErrNotFound
	}
	return artifact, nil
}

func mapOrEmpty(value domain.JSONMap) domain.JSONMap {
	if value == nil {
		return domain.JSONMap{}
	}
	return value
}

func take[T any](values []T, limit int) []T {
	if limit <= 0 || len(values) <= limit {
		return values
	}
	return values[:limit]
}
