package domain

import (
	"crypto/rand"
	"encoding/hex"
	"strings"
	"time"
)

type JSONMap map[string]any

type ThreadStatus string
type RunStatus string

const (
	ThreadStatusActive   ThreadStatus = "active"
	ThreadStatusArchived ThreadStatus = "archived"
	ThreadStatusDeleted  ThreadStatus = "deleted"

	RunStatusQueued          RunStatus = "queued"
	RunStatusRunning         RunStatus = "running"
	RunStatusWaitingForInput RunStatus = "waiting_for_input"
	RunStatusWaitingForTask  RunStatus = "waiting_for_task"
	RunStatusSucceeded       RunStatus = "succeeded"
	RunStatusFailed          RunStatus = "failed"
	RunStatusCanceled        RunStatus = "canceled"
)

type ThreadMessage struct {
	MessageID string    `json:"message_id,omitempty"`
	ThreadID  string    `json:"thread_id,omitempty"`
	Role      string    `json:"role"`
	Content   string    `json:"content"`
	CreatedAt time.Time `json:"created_at,omitempty"`
	Metadata  JSONMap   `json:"metadata,omitempty"`
	RunID     string    `json:"run_id,omitempty"`
}

type ThreadRecord struct {
	ThreadID     string       `json:"thread_id"`
	UserID       string       `json:"user_id,omitempty"`
	Title        string       `json:"title,omitempty"`
	Status       ThreadStatus `json:"status"`
	CreatedAt    time.Time    `json:"created_at"`
	UpdatedAt    time.Time    `json:"updated_at"`
	LatestRunID  string       `json:"latest_run_id,omitempty"`
	CheckpointID string       `json:"checkpoint_id,omitempty"`
	Summary      string       `json:"summary,omitempty"`
	Metadata     JSONMap      `json:"metadata"`
}

type RunRecord struct {
	RunID           string     `json:"run_id"`
	ThreadID        string     `json:"thread_id,omitempty"`
	UserID          string     `json:"user_id,omitempty"`
	Goal            string     `json:"goal"`
	Status          RunStatus  `json:"status"`
	WorkflowKind    string     `json:"workflow_kind"`
	Mode            string     `json:"mode,omitempty"`
	CurrentNode     string     `json:"current_node,omitempty"`
	ParentRunID     string     `json:"parent_run_id,omitempty"`
	PlannerVersion  string     `json:"planner_version,omitempty"`
	AgentRole       string     `json:"agent_role,omitempty"`
	TraceGroupID    string     `json:"trace_group_id,omitempty"`
	CheckpointID    string     `json:"checkpoint_id,omitempty"`
	CheckpointState JSONMap    `json:"checkpoint_state,omitempty"`
	BudgetState     JSONMap    `json:"budget_state,omitempty"`
	ResponseText    string     `json:"response_text,omitempty"`
	Error           string     `json:"error,omitempty"`
	CreatedAt       time.Time  `json:"created_at"`
	UpdatedAt       time.Time  `json:"updated_at"`
	StartedAt       *time.Time `json:"started_at,omitempty"`
	CompletedAt     *time.Time `json:"completed_at,omitempty"`
	Metadata        JSONMap    `json:"metadata"`
}

type RunEventRecord struct {
	EventID      string    `json:"event_id,omitempty"`
	Sequence     int64     `json:"sequence,omitempty"`
	RunID        string    `json:"run_id"`
	ThreadID     string    `json:"thread_id,omitempty"`
	EventKind    string    `json:"event_kind"`
	EventType    string    `json:"event_type,omitempty"`
	NodeName     string    `json:"node_name,omitempty"`
	TaskID       string    `json:"task_id,omitempty"`
	CheckpointID string    `json:"checkpoint_id,omitempty"`
	ScopeID      string    `json:"scope_id,omitempty"`
	AgentRole    string    `json:"agent_role,omitempty"`
	Level        string    `json:"level,omitempty"`
	TS           time.Time `json:"ts,omitempty"`
	Message      string    `json:"message,omitempty"`
	Payload      JSONMap   `json:"payload"`
}

type ArtifactRecord struct {
	ArtifactID    string    `json:"artifact_id"`
	RunID         string    `json:"run_id"`
	ThreadID      string    `json:"thread_id,omitempty"`
	Kind          string    `json:"kind"`
	Path          string    `json:"path,omitempty"`
	SourcePath    string    `json:"source_path,omitempty"`
	PreviewPath   string    `json:"preview_path,omitempty"`
	Title         string    `json:"title,omitempty"`
	ResultGroupID string    `json:"result_group_id,omitempty"`
	MimeType      string    `json:"mime_type,omitempty"`
	SizeBytes     int64     `json:"size_bytes,omitempty"`
	SHA256        string    `json:"sha256,omitempty"`
	StorageURI    string    `json:"storage_uri,omitempty"`
	ToolName      string    `json:"tool_name,omitempty"`
	Category      string    `json:"category,omitempty"`
	CreatedAt     time.Time `json:"created_at"`
	UpdatedAt     time.Time `json:"updated_at,omitempty"`
	Metadata      JSONMap   `json:"metadata"`
}

type UserAccount struct {
	UserID      string    `json:"user_id"`
	Email       string    `json:"email,omitempty"`
	DisplayName string    `json:"display_name,omitempty"`
	Role        string    `json:"role,omitempty"`
	Status      string    `json:"status,omitempty"`
	OrgID       string    `json:"org_id,omitempty"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
	Metadata    JSONMap   `json:"metadata"`
}

type Organization struct {
	OrgID     string    `json:"org_id"`
	Name      string    `json:"name"`
	Status    string    `json:"status,omitempty"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
	Metadata  JSONMap   `json:"metadata"`
}

type RunLeaseRecord struct {
	RunID          string    `json:"run_id"`
	WorkerID       string    `json:"worker_id"`
	LeaseToken     string    `json:"lease_token"`
	LeaseExpiresAt time.Time `json:"lease_expires_at"`
	CreatedAt      time.Time `json:"created_at"`
	UpdatedAt      time.Time `json:"updated_at"`
}

type WorkerHeartbeatRecord struct {
	WorkerID        string    `json:"worker_id"`
	WorkerKind      string    `json:"worker_kind"`
	Status          string    `json:"status"`
	CurrentRunID    string    `json:"current_run_id,omitempty"`
	Hostname        string    `json:"hostname,omitempty"`
	Version         string    `json:"version,omitempty"`
	StartedAt       time.Time `json:"started_at"`
	LastHeartbeatAt time.Time `json:"last_heartbeat_at"`
	UpdatedAt       time.Time `json:"updated_at"`
	Metadata        JSONMap   `json:"metadata"`
}

type CreateUserInput struct {
	UserID      string
	Email       string
	DisplayName string
	Role        string
	Status      string
	OrgID       string
	Metadata    JSONMap
}

type UpdateUserStatusInput struct {
	UserID string
	Status string
}

type CreateOrganizationInput struct {
	OrgID    string
	Name     string
	Status   string
	Metadata JSONMap
}

type AcquireRunLeaseInput struct {
	RunID    string
	WorkerID string
	TTL      time.Duration
	Now      time.Time
}

type RenewRunLeaseInput struct {
	RunID      string
	LeaseToken string
	TTL        time.Duration
	Now        time.Time
}

type ReleaseRunLeaseInput struct {
	RunID      string
	LeaseToken string
}

type UpsertWorkerHeartbeatInput struct {
	WorkerID        string
	WorkerKind      string
	Status          string
	CurrentRunID    string
	Hostname        string
	Version         string
	StartedAt       time.Time
	LastHeartbeatAt time.Time
	Metadata        JSONMap
}

type CreateThreadInput struct {
	UserID          string
	Title           string
	Metadata        JSONMap
	InitialMessages []ThreadMessage
}

type CreateRunInput struct {
	ThreadID     string
	UserID       string
	Goal         string
	WorkflowKind string
	Mode         string
	Messages     []ThreadMessage
	Metadata     JSONMap
	Internal     bool
}

type CompleteRunInput struct {
	RunID        string
	ResponseText string
}

type AppendRunEventInput struct {
	EventID      string    `json:"event_id,omitempty"`
	RunID        string    `json:"run_id"`
	ThreadID     string    `json:"thread_id,omitempty"`
	EventKind    string    `json:"event_kind"`
	EventType    string    `json:"event_type,omitempty"`
	NodeName     string    `json:"node_name,omitempty"`
	TaskID       string    `json:"task_id,omitempty"`
	CheckpointID string    `json:"checkpoint_id,omitempty"`
	ScopeID      string    `json:"scope_id,omitempty"`
	AgentRole    string    `json:"agent_role,omitempty"`
	Level        string    `json:"level,omitempty"`
	TS           time.Time `json:"ts,omitempty"`
	Message      string    `json:"message,omitempty"`
	Payload      JSONMap   `json:"payload"`
}

type CreateArtifactInput struct {
	ArtifactID    string
	RunID         string
	ThreadID      string
	Kind          string
	Path          string
	SourcePath    string
	PreviewPath   string
	Title         string
	ResultGroupID string
	MimeType      string
	SizeBytes     int64
	SHA256        string
	StorageURI    string
	ToolName      string
	Category      string
	Metadata      JSONMap
}

func NewID(prefix string) string {
	var bytes [16]byte
	if _, err := rand.Read(bytes[:]); err != nil {
		return strings.TrimSuffix(prefix, "_") + "_" + time.Now().UTC().Format("20060102150405.000000000")
	}
	return strings.TrimSuffix(prefix, "_") + "_" + hex.EncodeToString(bytes[:])
}

func Now() time.Time {
	return time.Now().UTC()
}
