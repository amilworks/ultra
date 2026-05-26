package eventbus

import (
	"context"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

type Job struct {
	RunID    string
	ThreadID string
	UserID   string
	Goal     string
}

type Bus interface {
	PublishJob(ctx context.Context, job Job) error
	PublishRunEvent(ctx context.Context, event domain.RunEventRecord) error
}
