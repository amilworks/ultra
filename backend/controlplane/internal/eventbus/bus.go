package eventbus

import (
	"context"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

type Job struct {
	RunID                 string                        `json:"run_id"`
	DispatchID            string                        `json:"dispatch_id,omitempty"`
	ThreadID              string                        `json:"thread_id"`
	UserID                string                        `json:"user_id"`
	Goal                  string                        `json:"goal"`
	WorkflowKind          string                        `json:"workflow_kind,omitempty"`
	EvaluationProfile     domain.EvaluationProfile      `json:"evaluation_profile,omitempty"`
	RemoteMutationIntents []domain.RemoteMutationIntent `json:"remote_mutation_intents,omitempty"`
	Messages              []domain.ThreadMessage        `json:"messages,omitempty"`
	FileIDs               []string                      `json:"file_ids,omitempty"`
	ResourceURIs          []string                      `json:"resource_uris,omitempty"`
	DatasetURIs           []string                      `json:"dataset_uris,omitempty"`
	SelectedToolNames     []string                      `json:"selected_tool_names,omitempty"`
	KnowledgeContext      domain.JSONMap                `json:"knowledge_context,omitempty"`
	WorkflowHint          domain.JSONMap                `json:"workflow_hint,omitempty"`
	SelectionContext      domain.JSONMap                `json:"selection_context,omitempty"`
	ReasoningMode         string                        `json:"reasoning_mode,omitempty"`
	Budgets               domain.JSONMap                `json:"budgets,omitempty"`
	Benchmark             domain.JSONMap                `json:"benchmark,omitempty"`
	ResourceDescriptors   []domain.JSONMap              `json:"resource_descriptors,omitempty"`
	Metadata              domain.JSONMap                `json:"metadata,omitempty"`
}

type DataAgentJob struct {
	JobID         string         `json:"job_id"`
	DispatchID    string         `json:"dispatch_id,omitempty"`
	OwnerUserID   string         `json:"owner_user_id"`
	OwnerOrgID    string         `json:"owner_org_id,omitempty"`
	ProjectID     string         `json:"project_id,omitempty"`
	JobType       string         `json:"job_type"`
	ResourceIDs   []string       `json:"resource_ids,omitempty"`
	ResourceCount int            `json:"resource_count"`
	InputSelector domain.JSONMap `json:"input_selector,omitempty"`
	Metadata      domain.JSONMap `json:"metadata,omitempty"`
}

type CancelSignal struct {
	RunID    string         `json:"run_id"`
	ThreadID string         `json:"thread_id,omitempty"`
	UserID   string         `json:"user_id,omitempty"`
	Reason   string         `json:"reason,omitempty"`
	Metadata domain.JSONMap `json:"metadata,omitempty"`
}

type QueueConsumerTarget struct {
	Name    string
	Role    string
	Subject string
}

type QueueDiagnostics struct {
	Available      bool                       `json:"available"`
	Mode           string                     `json:"mode"`
	Stream         string                     `json:"stream,omitempty"`
	StreamSubjects []string                   `json:"stream_subjects,omitempty"`
	StreamMessages uint64                     `json:"stream_messages"`
	StreamBytes    uint64                     `json:"stream_bytes"`
	FirstSequence  uint64                     `json:"first_sequence"`
	LastSequence   uint64                     `json:"last_sequence"`
	ConsumerCount  int                        `json:"consumer_count"`
	Consumers      []QueueConsumerDiagnostics `json:"consumers"`
	Error          string                     `json:"error,omitempty"`
}

type QueueConsumerDiagnostics struct {
	Name                    string  `json:"name"`
	Role                    string  `json:"role,omitempty"`
	Subject                 string  `json:"subject,omitempty"`
	Active                  bool    `json:"active"`
	AckWaitSeconds          float64 `json:"ack_wait_seconds,omitempty"`
	MaxDeliver              int     `json:"max_deliver,omitempty"`
	PendingMessages         uint64  `json:"pending_messages"`
	InFlightMessages        int     `json:"in_flight_messages"`
	RedeliveredMessages     int     `json:"redelivered_messages"`
	WaitingPullRequests     int     `json:"waiting_pull_requests"`
	DeliveredStreamSequence uint64  `json:"delivered_stream_sequence,omitempty"`
	AckFloorStreamSequence  uint64  `json:"ack_floor_stream_sequence,omitempty"`
	Error                   string  `json:"error,omitempty"`
}

type QueueDiagnosticsProvider interface {
	QueueDiagnostics(ctx context.Context) (QueueDiagnostics, error)
}

type DataAgentJobPublisher interface {
	PublishDataAgentJob(ctx context.Context, job DataAgentJob) error
}

type Bus interface {
	PublishJob(ctx context.Context, job Job) error
	PublishCancel(ctx context.Context, cancel CancelSignal) error
	PublishRunEvent(ctx context.Context, event domain.RunEventRecord) error
}

type RunEventSource interface {
	SubscribeRunEvents(ctx context.Context, runID string) (<-chan domain.RunEventRecord, func())
}
