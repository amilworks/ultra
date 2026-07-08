package eventbus

import (
	"context"
	"errors"
	"strings"
)

// TrainingJob is the wire envelope for the five generic GoldGate job phases
// (training.{sync,assemble,finetune,benchmark,gold_freeze}). The subject is
// model-neutral; model_key travels as a validated attribute and the worker
// resolves the adapter from it. NATS stays dumb delivery: params reference
// rows/barrel paths, never payloads.
type TrainingJob struct {
	JobID       string         `json:"job_id"`
	DispatchID  string         `json:"dispatch_id,omitempty"`
	ModelKey    string         `json:"model_key"`
	JobType     string         `json:"job_type"`
	GPUPool     string         `json:"gpu_pool,omitempty"`
	OwnerUserID string         `json:"owner_user_id"`
	OwnerOrgID  string         `json:"owner_org_id,omitempty"`
	Params      map[string]any `json:"params,omitempty"`
	Metadata    map[string]any `json:"metadata,omitempty"`
}

type TrainingJobPublisher interface {
	PublishTrainingJob(ctx context.Context, job TrainingJob) error
}

func (b *NATSBus) PublishTrainingJob(ctx context.Context, job TrainingJob) error {
	subject := strings.TrimSpace(b.cfg.TrainingJobsSubject)
	if subject == "" {
		return errors.New("nats training jobs subject is not configured")
	}
	return b.publish(ctx, subject, job, natsMessageIDForTrainingJob(job))
}

func natsMessageIDForTrainingJob(job TrainingJob) string {
	jobID := strings.TrimSpace(job.JobID)
	if jobID == "" {
		return ""
	}
	dispatchID := strings.TrimSpace(job.DispatchID)
	if dispatchID != "" {
		return "training-job:" + jobID + ":" + dispatchID
	}
	return "training-job:" + jobID
}

func (b *MemoryBus) PublishTrainingJob(ctx context.Context, job TrainingJob) error {
	_ = ctx
	b.mu.Lock()
	defer b.mu.Unlock()
	b.trainingJobs = append(b.trainingJobs, job)
	return nil
}

// TrainingJobs returns the jobs published to the in-memory bus (tests/dev).
func (b *MemoryBus) TrainingJobs() []TrainingJob {
	b.mu.Lock()
	defer b.mu.Unlock()
	jobs := make([]TrainingJob, len(b.trainingJobs))
	copy(jobs, b.trainingJobs)
	return jobs
}

func (b *SplitBus) PublishTrainingJob(ctx context.Context, job TrainingJob) error {
	publisher, ok := b.jobs.(TrainingJobPublisher)
	if !ok {
		return errors.New("training job publisher is not configured")
	}
	return publisher.PublishTrainingJob(ctx, job)
}
