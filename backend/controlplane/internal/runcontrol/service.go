package runcontrol

import (
	"context"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
)

type Store interface {
	CreateThread(context.Context, domain.CreateThreadInput) (domain.ThreadRecord, error)
	GetThread(context.Context, string) (domain.ThreadRecord, error)
	ListThreads(context.Context, int) ([]domain.ThreadRecord, error)
	ListThreadMessages(context.Context, string) ([]domain.ThreadMessage, error)
	CreateRun(context.Context, domain.CreateRunInput) (domain.RunRecord, error)
	GetRun(context.Context, string) (domain.RunRecord, error)
	UpdateRunStatus(context.Context, string, domain.RunStatus, string, string) (domain.RunRecord, error)
	AppendRunEvent(context.Context, domain.AppendRunEventInput) (domain.RunEventRecord, error)
	ListRunEvents(context.Context, string, int) ([]domain.RunEventRecord, error)
	CreateArtifact(context.Context, domain.CreateArtifactInput) (domain.ArtifactRecord, error)
	ListRunArtifacts(context.Context, string, int) ([]domain.ArtifactRecord, error)
	GetArtifact(context.Context, string) (domain.ArtifactRecord, error)
}

type Service struct {
	store Store
	bus   eventbus.Bus
}

type CreateThreadRequest struct {
	UserID          string
	Title           string
	Metadata        domain.JSONMap
	InitialMessages []domain.ThreadMessage
}

type CreateRunRequest struct {
	ThreadID string
	UserID   string
	Goal     string
	Messages []domain.ThreadMessage
	Metadata domain.JSONMap
}

func NewService(store Store, bus eventbus.Bus) *Service {
	return &Service{store: store, bus: bus}
}

func (s *Service) CreateThread(ctx context.Context, req CreateThreadRequest) (domain.ThreadRecord, error) {
	return s.store.CreateThread(ctx, domain.CreateThreadInput{
		UserID:          req.UserID,
		Title:           req.Title,
		Metadata:        req.Metadata,
		InitialMessages: req.InitialMessages,
	})
}

func (s *Service) CreateRun(ctx context.Context, req CreateRunRequest) (domain.RunRecord, error) {
	run, err := s.store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: req.ThreadID,
		UserID:   req.UserID,
		Goal:     req.Goal,
		Messages: req.Messages,
		Metadata: req.Metadata,
	})
	if err != nil {
		return domain.RunRecord{}, err
	}
	event, err := s.store.AppendRunEvent(ctx, domain.AppendRunEventInput{
		RunID:     run.RunID,
		ThreadID:  run.ThreadID,
		EventKind: "run.accepted",
		Message:   "Run accepted.",
		Payload:   domain.JSONMap{"status": string(run.Status)},
	})
	if err != nil {
		return domain.RunRecord{}, err
	}
	if err := s.bus.PublishRunEvent(ctx, event); err != nil {
		return domain.RunRecord{}, err
	}
	if err := s.bus.PublishJob(ctx, eventbus.Job{
		RunID:    run.RunID,
		ThreadID: run.ThreadID,
		UserID:   run.UserID,
		Goal:     run.Goal,
	}); err != nil {
		return domain.RunRecord{}, err
	}
	return run, nil
}
