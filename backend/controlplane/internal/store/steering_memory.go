package store

import (
	"context"
	"errors"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

// ErrSteeringClosed reports that a run no longer accepts steering messages:
// it is terminal, the worker has closed the finalization barrier, or the
// per-run cap is reached. Clients fall back to Phase 0 queueing on this error.
var ErrSteeringClosed = errors.New("run no longer accepts steering messages")

// maxSteerMessagesPerRun bounds context/token amplification: every accepted
// steer is injected into the coordinator's model context. Past the cap the
// client's Phase 0 queue is the right home for further thoughts.
const maxSteerMessagesPerRun = 32

// CreateRunSteerMessage atomically records an accepted steer: the steer row
// AND its thread-transcript message commit together, so a recovery requeue
// (which reseeds jobs from the transcript) can never observe one without the
// other. Retrying an existing steer_id returns the stored row unchanged.
func (s *MemoryStore) CreateRunSteerMessage(
	ctx context.Context,
	input domain.CreateRunSteerMessageInput,
) (domain.RunSteerMessageRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	run, ok := s.runs[input.RunID]
	if !ok {
		return domain.RunSteerMessageRecord{}, ErrNotFound
	}
	for _, existing := range s.steerMessages[input.RunID] {
		if existing.SteerID == input.SteerID {
			return existing, nil
		}
	}
	// steer_id is a global key (client-supplied): a collision with another
	// run's steer is a conflict, never a disclosure of that run's record.
	for runID, rows := range s.steerMessages {
		if runID == input.RunID {
			continue
		}
		for _, existing := range rows {
			if existing.SteerID == input.SteerID {
				return domain.RunSteerMessageRecord{}, ErrConflict
			}
		}
	}
	if isTerminalRunStatus(run.Status) {
		return domain.RunSteerMessageRecord{}, ErrSteeringClosed
	}
	if _, closed := s.steerBarriers[input.RunID]; closed {
		return domain.RunSteerMessageRecord{}, ErrSteeringClosed
	}
	if len(s.steerMessages[input.RunID]) >= maxSteerMessagesPerRun {
		return domain.RunSteerMessageRecord{}, ErrSteeringClosed
	}
	now := input.CreatedAt
	if now.IsZero() {
		now = domain.Now()
	}
	record := domain.RunSteerMessageRecord{
		SteerID:   input.SteerID,
		RunID:     input.RunID,
		ThreadID:  input.ThreadID,
		UserID:    input.UserID,
		MessageID: input.MessageID,
		Content:   input.Content,
		Status:    domain.RunSteerStatusPending,
		CreatedAt: now,
		UpdatedAt: now,
	}
	s.steerMessages[input.RunID] = append(s.steerMessages[input.RunID], record)
	s.messages[input.ThreadID] = append(s.messages[input.ThreadID], domain.ThreadMessage{
		MessageID: input.MessageID,
		ThreadID:  input.ThreadID,
		Role:      "user",
		Content:   input.Content,
		CreatedAt: now,
		Metadata: domain.JSONMap{
			"kind":     "steering",
			"steer_id": input.SteerID,
		},
		RunID: input.RunID,
	})
	return record, nil
}

func (s *MemoryStore) ListRunSteerMessages(
	ctx context.Context,
	runID string,
) ([]domain.RunSteerMessageRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	if _, ok := s.runs[runID]; !ok {
		return nil, ErrNotFound
	}
	return append([]domain.RunSteerMessageRecord(nil), s.steerMessages[runID]...), nil
}

// MarkRunSteerMessageApplied flips pending → applied. The bool reports
// whether this call performed the flip — false means it already happened
// (idempotent worker ack retries) or the steer was marked missed.
func (s *MemoryStore) MarkRunSteerMessageApplied(
	ctx context.Context,
	runID string,
	steerID string,
	appliedAt time.Time,
) (domain.RunSteerMessageRecord, bool, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	rows := s.steerMessages[runID]
	for index, row := range rows {
		if row.SteerID != steerID {
			continue
		}
		if row.Status != domain.RunSteerStatusPending {
			return row, false, nil
		}
		at := appliedAt
		if at.IsZero() {
			at = domain.Now()
		}
		row.Status = domain.RunSteerStatusApplied
		row.AppliedAt = &at
		row.UpdatedAt = at
		rows[index] = row
		return row, true, nil
	}
	return domain.RunSteerMessageRecord{}, false, ErrNotFound
}

// CloseRunSteerBarrier atomically stops steer acceptance for the run and
// returns any still-pending steers, which the worker applies in one final
// continuation before finishing. Serialized against CreateRunSteerMessage,
// so an accepted steer is always either injected mid-run or in this list.
func (s *MemoryStore) CloseRunSteerBarrier(
	ctx context.Context,
	runID string,
	closedAt time.Time,
) ([]domain.RunSteerMessageRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.runs[runID]; !ok {
		return nil, ErrNotFound
	}
	if closedAt.IsZero() {
		closedAt = domain.Now()
	}
	if _, closed := s.steerBarriers[runID]; !closed {
		s.steerBarriers[runID] = closedAt
	}
	var pending []domain.RunSteerMessageRecord
	for _, row := range s.steerMessages[runID] {
		if row.Status == domain.RunSteerStatusPending {
			pending = append(pending, row)
		}
	}
	return pending, nil
}

// OpenRunSteerBarrier re-arms steer acceptance; called when recovery requeues
// the run for a fresh attempt.
func (s *MemoryStore) OpenRunSteerBarrier(ctx context.Context, runID string) error {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.steerBarriers, runID)
	return nil
}

// MarkPendingRunSteerMessagesMissed sweeps leftovers at run-terminal time.
// With the barrier protocol this should return nothing for succeeded runs;
// failed/canceled runs legitimately strand pending steers here.
func (s *MemoryStore) MarkPendingRunSteerMessagesMissed(
	ctx context.Context,
	runID string,
	at time.Time,
) ([]domain.RunSteerMessageRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	if at.IsZero() {
		at = domain.Now()
	}
	rows := s.steerMessages[runID]
	var changed []domain.RunSteerMessageRecord
	for index, row := range rows {
		if row.Status != domain.RunSteerStatusPending {
			continue
		}
		row.Status = domain.RunSteerStatusMissed
		row.UpdatedAt = at
		rows[index] = row
		changed = append(changed, row)
	}
	return changed, nil
}
