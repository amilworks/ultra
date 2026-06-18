package domain

import (
	"crypto/rand"
	"encoding/hex"
	"strings"
	"time"
)

type JSONMap map[string]any

const (
	DataAgentQueryResourceHardLimit = 100000
	PublicResourceGranteeUserID     = "__public__"
)

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

type ThreadListPage struct {
	Threads    []ThreadRecord
	TotalCount int
	Limit      int
	Offset     int
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

// RunHistorySearchOptions parameterizes an episodic-memory search over a single
// user's past runs. Query is matched (case-insensitively) against run goal,
// final response, and thread title; Since bounds recency; ExcludeRunID drops the
// current run so the agent never "recalls" the conversation it is in.
type RunHistorySearchOptions struct {
	Query        string
	Since        *time.Time
	Limit        int
	ExcludeRunID string
}

// RunHistoryHit is one past run surfaced by episodic search.
type RunHistoryHit struct {
	RunID           string     `json:"run_id"`
	ThreadID        string     `json:"thread_id,omitempty"`
	Title           string     `json:"title,omitempty"`
	Goal            string     `json:"goal"`
	ResponseSnippet string     `json:"response_snippet,omitempty"`
	CompletedAt     *time.Time `json:"completed_at,omitempty"`
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

type ResourceRecord struct {
	ResourceID         string               `json:"resource_id"`
	OriginalName       string               `json:"original_name"`
	ContentType        string               `json:"content_type,omitempty"`
	SizeBytes          int64                `json:"size_bytes"`
	SHA256             string               `json:"sha256,omitempty"`
	StorageURI         string               `json:"storage_uri,omitempty"`
	StoragePath        string               `json:"storage_path,omitempty"`
	SourceType         string               `json:"source_type"`
	ResourceKind       string               `json:"resource_kind"`
	SourceURI          string               `json:"source_uri,omitempty"`
	ProjectID          string               `json:"project_id,omitempty"`
	OwnerUserID        string               `json:"owner_user_id"`
	OwnerOrgID         string               `json:"owner_org_id,omitempty"`
	OwnerRole          string               `json:"owner_role,omitempty"`
	Status             string               `json:"status"`
	CreatedAt          time.Time            `json:"created_at"`
	UpdatedAt          time.Time            `json:"updated_at"`
	DeletedAt          time.Time            `json:"deleted_at,omitempty"`
	RetentionExpiresAt time.Time            `json:"retention_expires_at,omitempty"`
	Tags               []string             `json:"tags,omitempty"`
	Metadata           JSONMap              `json:"metadata"`
	ShareSummary       ResourceShareSummary `json:"share_summary,omitempty"`
}

type ResourceShareSummary struct {
	ShareStatus      string `json:"share_status"`
	ActiveGrantCount int    `json:"active_grant_count"`
	SharedByMe       bool   `json:"shared_by_me,omitempty"`
	SharedWithMe     bool   `json:"shared_with_me,omitempty"`
	Public           bool   `json:"public,omitempty"`
}

type ResourceListInput struct {
	UserID           string
	OrgID            string
	Query            string
	Kind             string
	Source           string
	ProjectID        string
	Sharing          string
	Tags             []string
	Descriptors      []string
	MetadataFilters  []ResourceMetadataFilter
	CreatedAfter     time.Time
	CreatedBefore    time.Time
	ProcessingStatus string
	Status           string
	Limit            int
	Offset           int
}

type ResourceListPage struct {
	Resources  []ResourceRecord
	TotalCount int
	Limit      int
	Offset     int
}

type ResourceMetadataFilter struct {
	Path     string
	Operator string
	Value    string
}

type UpsertResourceInput struct {
	ResourceID         string
	OriginalName       string
	ContentType        string
	SizeBytes          int64
	SHA256             string
	StorageURI         string
	StoragePath        string
	SourceType         string
	ResourceKind       string
	SourceURI          string
	ProjectID          string
	OwnerUserID        string
	OwnerOrgID         string
	OwnerRole          string
	Status             string
	CreatedAt          time.Time
	UpdatedAt          time.Time
	DeletedAt          time.Time
	RetentionExpiresAt time.Time
	Tags               []string
	Metadata           JSONMap
}

type BulkTagResourcesInput struct {
	OwnerUserID string
	OwnerOrgID  string
	ActorUserID string
	ActorOrgID  string
	ResourceIDs []string
	Tags        []string
	TS          time.Time
	Metadata    JSONMap
}

type BulkTagResourcesResult struct {
	UpdatedCount int
	Resources    []ResourceRecord
	Events       []ResourceEventRecord
}

type MergeResourceMetadataInput struct {
	ResourceID string
	UserID     string
	OrgID      string
	Patch      JSONMap
	UpdatedAt  time.Time
}

type RenameResourceInput struct {
	ResourceID   string
	UserID       string
	OrgID        string
	OriginalName string
	UpdatedAt    time.Time
}

type ResourceEventRecord struct {
	EventID     string    `json:"event_id"`
	ResourceID  string    `json:"resource_id"`
	ActorUserID string    `json:"actor_user_id,omitempty"`
	ActorOrgID  string    `json:"actor_org_id,omitempty"`
	EventType   string    `json:"event_type"`
	TS          time.Time `json:"ts"`
	Metadata    JSONMap   `json:"metadata"`
}

type AppendResourceEventInput struct {
	EventID     string
	ResourceID  string
	ActorUserID string
	ActorOrgID  string
	EventType   string
	TS          time.Time
	Metadata    JSONMap
}

type ResourceEventListInput struct {
	UserID      string
	OrgID       string
	ResourceID  string
	EventType   string
	ActorUserID string
	Limit       int
	Offset      int
}

type ResourceEventListPage struct {
	Events     []ResourceEventRecord
	TotalCount int
	Limit      int
	Offset     int
}

type ResourceShareGrantRecord struct {
	GrantID         string    `json:"grant_id"`
	ResourceID      string    `json:"resource_id"`
	OwnerUserID     string    `json:"owner_user_id"`
	OwnerOrgID      string    `json:"owner_org_id,omitempty"`
	OwnerRole       string    `json:"owner_role,omitempty"`
	GranteeUserID   string    `json:"grantee_user_id,omitempty"`
	GranteeOrgID    string    `json:"grantee_org_id,omitempty"`
	Role            string    `json:"role"`
	Status          string    `json:"status"`
	CreatedByUserID string    `json:"created_by_user_id,omitempty"`
	CreatedAt       time.Time `json:"created_at"`
	UpdatedAt       time.Time `json:"updated_at"`
	RevokedAt       time.Time `json:"revoked_at,omitempty"`
	Metadata        JSONMap   `json:"metadata"`
}

type CreateResourceShareGrantInput struct {
	GrantID         string
	ResourceID      string
	OwnerUserID     string
	OwnerOrgID      string
	OwnerRole       string
	GranteeUserID   string
	GranteeOrgID    string
	Public          bool
	Role            string
	Status          string
	CreatedByUserID string
	CreatedAt       time.Time
	UpdatedAt       time.Time
	Metadata        JSONMap
}

type ListResourceShareGrantsInput struct {
	ResourceID  string
	OwnerUserID string
	OwnerOrgID  string
	Status      string
	Limit       int
}

type RevokeResourceShareGrantInput struct {
	GrantID     string
	ResourceID  string
	OwnerUserID string
	OwnerOrgID  string
	RevokedAt   time.Time
}

type ResourceCollectionShareGrantRecord struct {
	GrantID         string    `json:"grant_id"`
	CollectionID    string    `json:"collection_id"`
	OwnerUserID     string    `json:"owner_user_id"`
	OwnerOrgID      string    `json:"owner_org_id,omitempty"`
	OwnerRole       string    `json:"owner_role,omitempty"`
	GranteeUserID   string    `json:"grantee_user_id,omitempty"`
	GranteeOrgID    string    `json:"grantee_org_id,omitempty"`
	Role            string    `json:"role"`
	Status          string    `json:"status"`
	CreatedByUserID string    `json:"created_by_user_id,omitempty"`
	CreatedAt       time.Time `json:"created_at"`
	UpdatedAt       time.Time `json:"updated_at"`
	RevokedAt       time.Time `json:"revoked_at,omitempty"`
	Metadata        JSONMap   `json:"metadata"`
}

type CreateResourceCollectionShareGrantInput struct {
	GrantID         string
	CollectionID    string
	OwnerUserID     string
	OwnerOrgID      string
	OwnerRole       string
	GranteeUserID   string
	GranteeOrgID    string
	Public          bool
	Role            string
	Status          string
	CreatedByUserID string
	CreatedAt       time.Time
	UpdatedAt       time.Time
	Metadata        JSONMap
}

type CreateResourceCollectionShareGrantResult struct {
	Grant          ResourceCollectionShareGrantRecord
	ResourceGrants []ResourceShareGrantRecord
}

type ResourceStorageStats struct {
	TotalResources int
	TotalBytes     int64
}

// ResourceRetentionBacklog summarizes soft-deleted resources whose undelete window has
// elapsed (retention_expires_at < now) — the storage a retention GC can reclaim.
type ResourceRetentionBacklog struct {
	Count int64
	Bytes int64
}

type ResourceCollectionRecord struct {
	CollectionID       string    `json:"collection_id"`
	OwnerUserID        string    `json:"owner_user_id"`
	OwnerOrgID         string    `json:"owner_org_id,omitempty"`
	OwnerRole          string    `json:"owner_role,omitempty"`
	ProjectID          string    `json:"project_id,omitempty"`
	ParentCollectionID string    `json:"parent_collection_id,omitempty"`
	Name               string    `json:"name"`
	Description        string    `json:"description,omitempty"`
	CollectionType     string    `json:"collection_type"`
	Status             string    `json:"status"`
	ResourceCount      int       `json:"resource_count"`
	CreatedAt          time.Time `json:"created_at"`
	UpdatedAt          time.Time `json:"updated_at"`
	Metadata           JSONMap   `json:"metadata"`
}

type CreateResourceCollectionInput struct {
	CollectionID       string
	OwnerUserID        string
	OwnerOrgID         string
	OwnerRole          string
	ProjectID          string
	ParentCollectionID string
	Name               string
	Description        string
	CollectionType     string
	Status             string
	CreatedAt          time.Time
	UpdatedAt          time.Time
	Metadata           JSONMap
}

type RenameResourceCollectionInput struct {
	CollectionID string
	UserID       string
	OrgID        string
	Name         string
	UpdatedAt    time.Time
}

type ResourceCollectionListInput struct {
	UserID    string
	OrgID     string
	Query     string
	Type      string
	ProjectID string
	Status    string
	Limit     int
	Offset    int
}

type ResourceCollectionListPage struct {
	Collections []ResourceCollectionRecord
	TotalCount  int
	Limit       int
	Offset      int
}

type ResourceCollectionMembershipRecord struct {
	CollectionID  string    `json:"collection_id"`
	ResourceID    string    `json:"resource_id"`
	Position      int64     `json:"position"`
	AddedByUserID string    `json:"added_by_user_id,omitempty"`
	AddedAt       time.Time `json:"added_at"`
	Metadata      JSONMap   `json:"metadata"`
}

type AddResourcesToCollectionInput struct {
	CollectionID  string
	OwnerUserID   string
	OwnerOrgID    string
	ResourceIDs   []string
	AddedByUserID string
	AddedAt       time.Time
	Metadata      JSONMap
}

type AddResourcesToCollectionResult struct {
	Collection           ResourceCollectionRecord
	AddedCount           int
	Memberships          []ResourceCollectionMembershipRecord
	InheritedShareGrants []ResourceShareGrantRecord
}

type RemoveResourcesFromCollectionInput struct {
	CollectionID    string
	OwnerUserID     string
	OwnerOrgID      string
	ResourceIDs     []string
	RemovedByUserID string
	RemovedAt       time.Time
}

type RemoveResourcesFromCollectionResult struct {
	Collection                  ResourceCollectionRecord
	RemovedCount                int
	Memberships                 []ResourceCollectionMembershipRecord
	RevokedInheritedShareGrants []ResourceShareGrantRecord
}

type ResourceCollectionResourceListInput struct {
	CollectionID     string
	UserID           string
	OrgID            string
	Query            string
	Kind             string
	Source           string
	ProjectID        string
	Sharing          string
	Tags             []string
	Descriptors      []string
	MetadataFilters  []ResourceMetadataFilter
	CreatedAfter     time.Time
	CreatedBefore    time.Time
	ProcessingStatus string
	Limit            int
	Offset           int
}

type DatasetSnapshotRecord struct {
	SnapshotID         string    `json:"snapshot_id"`
	OwnerUserID        string    `json:"owner_user_id"`
	OwnerOrgID         string    `json:"owner_org_id,omitempty"`
	OwnerRole          string    `json:"owner_role,omitempty"`
	ProjectID          string    `json:"project_id,omitempty"`
	SourceCollectionID string    `json:"source_collection_id,omitempty"`
	Name               string    `json:"name"`
	Description        string    `json:"description,omitempty"`
	Status             string    `json:"status"`
	ResourceCount      int       `json:"resource_count"`
	TotalBytes         int64     `json:"total_bytes"`
	CreatedByUserID    string    `json:"created_by_user_id,omitempty"`
	CreatedAt          time.Time `json:"created_at"`
	Metadata           JSONMap   `json:"metadata"`
}

type DatasetSnapshotResourceRecord struct {
	SnapshotID        string    `json:"snapshot_id"`
	ResourceID        string    `json:"resource_id"`
	Position          int64     `json:"position"`
	OriginalName      string    `json:"original_name"`
	ContentType       string    `json:"content_type,omitempty"`
	SizeBytes         int64     `json:"size_bytes"`
	SHA256            string    `json:"sha256,omitempty"`
	SourceType        string    `json:"source_type"`
	ResourceKind      string    `json:"resource_kind"`
	StorageURI        string    `json:"storage_uri,omitempty"`
	SourceURI         string    `json:"source_uri,omitempty"`
	ProjectID         string    `json:"project_id,omitempty"`
	ResourceCreatedAt time.Time `json:"resource_created_at,omitempty"`
	Metadata          JSONMap   `json:"metadata"`
}

type CreateDatasetSnapshotInput struct {
	SnapshotID         string
	OwnerUserID        string
	OwnerOrgID         string
	OwnerRole          string
	ProjectID          string
	SourceCollectionID string
	Name               string
	Description        string
	ResourceIDs        []string
	ResourceQuery      *DatasetSnapshotResourceQuery
	CreatedByUserID    string
	CreatedAt          time.Time
	Metadata           JSONMap
}

type DatasetSnapshotResourceQuery struct {
	Query            string
	Kind             string
	Source           string
	ProjectID        string
	Sharing          string
	Tags             []string
	Descriptors      []string
	MetadataFilters  []ResourceMetadataFilter
	CreatedAfter     time.Time
	CreatedBefore    time.Time
	ProcessingStatus string
}

type DatasetSnapshotListInput struct {
	UserID             string
	OrgID              string
	Query              string
	ProjectID          string
	SourceCollectionID string
	Status             string
	Limit              int
	Offset             int
}

type DatasetSnapshotListPage struct {
	Snapshots  []DatasetSnapshotRecord
	TotalCount int
	Limit      int
	Offset     int
}

type DatasetSnapshotEventRecord struct {
	EventID     string    `json:"event_id"`
	SnapshotID  string    `json:"snapshot_id"`
	ActorUserID string    `json:"actor_user_id,omitempty"`
	ActorOrgID  string    `json:"actor_org_id,omitempty"`
	EventType   string    `json:"event_type"`
	TS          time.Time `json:"ts"`
	Metadata    JSONMap   `json:"metadata"`
}

type DatasetSnapshotEventListInput struct {
	SnapshotID  string
	UserID      string
	OrgID       string
	EventType   string
	ActorUserID string
	Limit       int
	Offset      int
}

type DatasetSnapshotEventListPage struct {
	Events     []DatasetSnapshotEventRecord
	TotalCount int
	Limit      int
	Offset     int
}

type DatasetSnapshotShareGrantRecord struct {
	GrantID         string    `json:"grant_id"`
	SnapshotID      string    `json:"snapshot_id"`
	OwnerUserID     string    `json:"owner_user_id"`
	OwnerOrgID      string    `json:"owner_org_id,omitempty"`
	OwnerRole       string    `json:"owner_role,omitempty"`
	GranteeUserID   string    `json:"grantee_user_id,omitempty"`
	GranteeOrgID    string    `json:"grantee_org_id,omitempty"`
	Role            string    `json:"role"`
	Status          string    `json:"status"`
	CreatedByUserID string    `json:"created_by_user_id,omitempty"`
	CreatedAt       time.Time `json:"created_at"`
	UpdatedAt       time.Time `json:"updated_at"`
	RevokedAt       time.Time `json:"revoked_at,omitempty"`
	Metadata        JSONMap   `json:"metadata"`
}

type CreateDatasetSnapshotShareGrantInput struct {
	GrantID         string
	SnapshotID      string
	OwnerUserID     string
	OwnerOrgID      string
	OwnerRole       string
	GranteeUserID   string
	GranteeOrgID    string
	Role            string
	Status          string
	CreatedByUserID string
	CreatedAt       time.Time
	UpdatedAt       time.Time
	Metadata        JSONMap
}

type ListDatasetSnapshotShareGrantsInput struct {
	SnapshotID  string
	OwnerUserID string
	OwnerOrgID  string
	Status      string
	Limit       int
}

type RevokeDatasetSnapshotShareGrantInput struct {
	GrantID          string
	SnapshotID       string
	OwnerUserID      string
	OwnerOrgID       string
	RevokedByUserID  string
	RevokedAt        time.Time
	RevocationReason string
}

type DataAgentJobRecord struct {
	JobID             string    `json:"job_id"`
	OwnerUserID       string    `json:"owner_user_id"`
	OwnerOrgID        string    `json:"owner_org_id,omitempty"`
	OwnerRole         string    `json:"owner_role,omitempty"`
	ProjectID         string    `json:"project_id,omitempty"`
	JobType           string    `json:"job_type"`
	Status            string    `json:"status"`
	ResourceCount     int       `json:"resource_count"`
	ProgressCompleted int       `json:"progress_completed"`
	ProgressTotal     int       `json:"progress_total"`
	Error             string    `json:"error,omitempty"`
	CreatedByUserID   string    `json:"created_by_user_id,omitempty"`
	CreatedAt         time.Time `json:"created_at"`
	UpdatedAt         time.Time `json:"updated_at"`
	StartedAt         time.Time `json:"started_at,omitempty"`
	CompletedAt       time.Time `json:"completed_at,omitempty"`
	InputSelector     JSONMap   `json:"input_selector"`
	OutputSummary     JSONMap   `json:"output_summary"`
	Metadata          JSONMap   `json:"metadata"`
}

type CreateDataAgentJobInput struct {
	JobID             string
	OwnerUserID       string
	OwnerOrgID        string
	OwnerRole         string
	ProjectID         string
	JobType           string
	Status            string
	ResourceIDs       []string
	ResourceCount     int
	ProgressCompleted int
	ProgressTotal     int
	Error             string
	CreatedByUserID   string
	CreatedAt         time.Time
	UpdatedAt         time.Time
	StartedAt         time.Time
	CompletedAt       time.Time
	InputSelector     JSONMap
	OutputSummary     JSONMap
	Metadata          JSONMap
}

type UpdateDataAgentJobInput struct {
	JobID             string
	OwnerUserID       string
	OwnerOrgID        string
	Status            string
	ProgressCompleted int
	ProgressTotal     int
	Error             string
	ActorUserID       string
	ActorOrgID        string
	Message           string
	UpdatedAt         time.Time
	StartedAt         time.Time
	CompletedAt       time.Time
	OutputSummary     JSONMap
	Metadata          JSONMap
	EventMetadata     JSONMap
}

type ControlDataAgentJobInput struct {
	JobID       string
	OwnerUserID string
	OwnerOrgID  string
	Action      string
	Reason      string
	ActorUserID string
	ActorOrgID  string
	TS          time.Time
	Metadata    JSONMap
}

// LinkDataAgentJobResourceInput records a resource against a data-agent job in the
// control_data_agent_job_resources junction. IORole is "input" for the images a job
// consumes (recorded at creation) or "output" for results a job produces (masks,
// bbox CSV, report) — registered by the worker as each item completes.
type LinkDataAgentJobResourceInput struct {
	JobID      string
	ResourceID string
	IORole     string
	Position   int
	Metadata   JSONMap
}

type DataAgentJobListInput struct {
	UserID    string
	OrgID     string
	JobType   string
	Status    string
	ProjectID string
	Limit     int
	Offset    int
}

type DataAgentJobListPage struct {
	Jobs       []DataAgentJobRecord
	TotalCount int
	Limit      int
	Offset     int
}

type DataAgentJobEventRecord struct {
	EventID     string    `json:"event_id"`
	JobID       string    `json:"job_id"`
	Sequence    int64     `json:"sequence"`
	EventType   string    `json:"event_type"`
	ActorUserID string    `json:"actor_user_id,omitempty"`
	ActorOrgID  string    `json:"actor_org_id,omitempty"`
	TS          time.Time `json:"ts"`
	Message     string    `json:"message,omitempty"`
	Metadata    JSONMap   `json:"metadata"`
}

type AppendDataAgentJobEventInput struct {
	EventID     string
	JobID       string
	Sequence    int64
	EventType   string
	ActorUserID string
	ActorOrgID  string
	TS          time.Time
	Message     string
	Metadata    JSONMap
}

type DataAgentJobLeaseRecord struct {
	JobID          string    `json:"job_id"`
	WorkerID       string    `json:"worker_id"`
	LeaseToken     string    `json:"lease_token"`
	LeaseExpiresAt time.Time `json:"lease_expires_at"`
	CreatedAt      time.Time `json:"created_at"`
	UpdatedAt      time.Time `json:"updated_at"`
}

type UploadSessionRecord struct {
	SessionID          string    `json:"session_id"`
	OwnerUserID        string    `json:"owner_user_id"`
	OwnerOrgID         string    `json:"owner_org_id,omitempty"`
	OwnerRole          string    `json:"owner_role,omitempty"`
	ProjectID          string    `json:"project_id,omitempty"`
	SourceType         string    `json:"source_type"`
	Status             string    `json:"status"`
	TotalBytes         int64     `json:"total_bytes"`
	BytesReceived      int64     `json:"bytes_received"`
	BytesVerified      int64     `json:"bytes_verified"`
	BytesCommitted     int64     `json:"bytes_committed"`
	IdempotencyKey     string    `json:"idempotency_key,omitempty"`
	BrowserFingerprint string    `json:"browser_fingerprint,omitempty"`
	Error              string    `json:"error,omitempty"`
	CreatedAt          time.Time `json:"created_at"`
	UpdatedAt          time.Time `json:"updated_at"`
	CompletedAt        time.Time `json:"completed_at,omitempty"`
	Metadata           JSONMap   `json:"metadata"`
}

type CreateUploadSessionInput struct {
	SessionID          string
	OwnerUserID        string
	OwnerOrgID         string
	OwnerRole          string
	ProjectID          string
	SourceType         string
	Status             string
	TotalBytes         int64
	BytesReceived      int64
	BytesVerified      int64
	BytesCommitted     int64
	IdempotencyKey     string
	BrowserFingerprint string
	Error              string
	CreatedAt          time.Time
	UpdatedAt          time.Time
	CompletedAt        time.Time
	Metadata           JSONMap
}

type UpdateUploadSessionInput struct {
	SessionID      string
	OwnerUserID    string
	OwnerOrgID     string
	Status         string
	BytesReceived  int64
	BytesVerified  int64
	BytesCommitted int64
	Error          string
	UpdatedAt      time.Time
	CompletedAt    time.Time
	Metadata       JSONMap
}

type UploadSessionTotals struct {
	BytesReceived  int64
	BytesVerified  int64
	BytesCommitted int64
	AllComplete    bool
}

// AdminWindowCounts carries one metric counted over the admin dashboard's
// standard activity windows.
type AdminWindowCounts struct {
	Total   int
	Last24h int
	Last7d  int
	Last30d int
}

// AdminUserMessageStats is one user's thread-message activity, aggregated in
// the store instead of by fetching every thread's messages.
type AdminUserMessageStats struct {
	UserID            string
	Messages          AdminWindowCounts
	UserMessages      AdminWindowCounts
	AssistantMessages AdminWindowCounts
}

// AdminUserEventStats is one user's tool-call/artifact event activity,
// aggregated in the store instead of by fetching every run's events.
type AdminUserEventStats struct {
	UserID    string
	ToolCalls AdminWindowCounts
	Artifacts AdminWindowCounts
}

// AdminResourceOwnerStats is one owner's (user/org/project) resource usage.
type AdminResourceOwnerStats struct {
	Owner        string
	Uploads      int
	StorageBytes int64
}

// AdminResourceStats summarizes the resource catalog for the admin
// dashboard without listing every resource row.
type AdminResourceStats struct {
	ActiveResources      int
	SoftDeletedResources int
	ActiveBytes          int64
	Users                []AdminResourceOwnerStats
	Orgs                 []AdminResourceOwnerStats
	Projects             []AdminResourceOwnerStats
}

type UploadSessionOperationalMetrics struct {
	Total          int   `json:"total"`
	Active         int   `json:"active"`
	Paused         int   `json:"paused"`
	Completed      int   `json:"completed"`
	Failed         int   `json:"failed"`
	Canceled       int   `json:"canceled"`
	Other          int   `json:"other"`
	BytesTotal     int64 `json:"bytes_total"`
	BytesReceived  int64 `json:"bytes_received"`
	BytesVerified  int64 `json:"bytes_verified"`
	BytesCommitted int64 `json:"bytes_committed"`
}

func (metrics *UploadSessionOperationalMetrics) AddSession(session UploadSessionRecord) {
	metrics.Total++
	switch strings.ToLower(strings.TrimSpace(session.Status)) {
	case "", "active":
		metrics.Active++
	case "paused":
		metrics.Paused++
	case "completed":
		metrics.Completed++
	case "failed":
		metrics.Failed++
	case "canceled", "cancelled":
		metrics.Canceled++
	default:
		metrics.Other++
	}
	metrics.BytesTotal += session.TotalBytes
	metrics.BytesReceived += session.BytesReceived
	metrics.BytesVerified += session.BytesVerified
	metrics.BytesCommitted += session.BytesCommitted
}

type UploadSessionEventRecord struct {
	EventID     string    `json:"event_id"`
	SessionID   string    `json:"session_id"`
	ActorUserID string    `json:"actor_user_id,omitempty"`
	ActorOrgID  string    `json:"actor_org_id,omitempty"`
	EventType   string    `json:"event_type"`
	TS          time.Time `json:"ts"`
	Metadata    JSONMap   `json:"metadata"`
}

type AppendUploadSessionEventInput struct {
	EventID     string
	SessionID   string
	ActorUserID string
	ActorOrgID  string
	EventType   string
	TS          time.Time
	Metadata    JSONMap
}

type UploadSessionFileRecord struct {
	SessionID      string    `json:"session_id"`
	FileToken      string    `json:"file_token"`
	ResourceID     string    `json:"resource_id,omitempty"`
	OriginalName   string    `json:"original_name"`
	RelativePath   string    `json:"relative_path,omitempty"`
	ContentType    string    `json:"content_type,omitempty"`
	SizeBytes      int64     `json:"size_bytes"`
	DeclaredSHA256 string    `json:"declared_sha256,omitempty"`
	ComputedSHA256 string    `json:"computed_sha256,omitempty"`
	Status         string    `json:"status"`
	Error          string    `json:"error,omitempty"`
	CreatedAt      time.Time `json:"created_at"`
	UpdatedAt      time.Time `json:"updated_at"`
	CompletedAt    time.Time `json:"completed_at,omitempty"`
	Metadata       JSONMap   `json:"metadata"`
}

type UpsertUploadSessionFileInput struct {
	SessionID      string
	FileToken      string
	ResourceID     string
	OriginalName   string
	RelativePath   string
	ContentType    string
	SizeBytes      int64
	DeclaredSHA256 string
	ComputedSHA256 string
	Status         string
	Error          string
	CreatedAt      time.Time
	UpdatedAt      time.Time
	CompletedAt    time.Time
	Metadata       JSONMap
}

type UploadChunkRecord struct {
	SessionID  string    `json:"session_id"`
	FileToken  string    `json:"file_token"`
	ChunkIndex int       `json:"chunk_index"`
	Offset     int64     `json:"offset"`
	SizeBytes  int64     `json:"size_bytes"`
	SHA256     string    `json:"sha256"`
	Status     string    `json:"status"`
	StorageURI string    `json:"storage_uri,omitempty"`
	ReceivedAt time.Time `json:"received_at,omitempty"`
	VerifiedAt time.Time `json:"verified_at,omitempty"`
	Error      string    `json:"error,omitempty"`
	Metadata   JSONMap   `json:"metadata"`
}

type UpsertUploadChunkInput struct {
	SessionID  string
	FileToken  string
	ChunkIndex int
	Offset     int64
	SizeBytes  int64
	SHA256     string
	Status     string
	StorageURI string
	ReceivedAt time.Time
	VerifiedAt time.Time
	Error      string
	Metadata   JSONMap
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

// RecordUserTokenUsageInput folds one completed run's token spend into the
// per-user daily and lifetime aggregates.
type RecordUserTokenUsageInput struct {
	UserID       string
	Day          time.Time
	InputTokens  int64
	OutputTokens int64
	TotalTokens  int64
	OccurredAt   time.Time
}

// RecordRunTokenUsageInput records one observed model-call usage delta. The
// (run_id, usage_event_id) pair is the idempotency key for redeliveries.
type RecordRunTokenUsageInput struct {
	RunID        string
	UsageEventID string
	UserID       string
	Model        string
	Day          time.Time
	InputTokens  int64
	OutputTokens int64
	TotalTokens  int64
	OccurredAt   time.Time
}

// RunTokenUsageRecord is one durable model-call usage ledger entry.
type RunTokenUsageRecord struct {
	RunID        string    `json:"run_id"`
	UsageEventID string    `json:"usage_event_id"`
	UserID       string    `json:"user_id"`
	Model        string    `json:"model,omitempty"`
	Day          time.Time `json:"day"`
	InputTokens  int64     `json:"input_tokens"`
	OutputTokens int64     `json:"output_tokens"`
	TotalTokens  int64     `json:"total_tokens"`
	OccurredAt   time.Time `json:"occurred_at"`
	CreatedAt    time.Time `json:"created_at"`
}

// RunTokenUsageSummary is the aggregate token spend for a run.
type RunTokenUsageSummary struct {
	RunID        string    `json:"run_id"`
	UserID       string    `json:"user_id"`
	Model        string    `json:"model,omitempty"`
	Day          time.Time `json:"day"`
	InputTokens  int64     `json:"input_tokens"`
	OutputTokens int64     `json:"output_tokens"`
	TotalTokens  int64     `json:"total_tokens"`
	Finalized    bool      `json:"finalized"`
}

type FinalizeRunTokenUsageInput struct {
	RunID       string
	CompletedAt time.Time
}

// UserTokenUsageStats is the lifetime aggregate row for a user.
type UserTokenUsageStats struct {
	UserID         string     `json:"user_id"`
	InputTokens    int64      `json:"input_tokens"`
	OutputTokens   int64      `json:"output_tokens"`
	TotalTokens    int64      `json:"total_tokens"`
	PeakDailyTotal int64      `json:"peak_daily_total"`
	LastActiveDay  *time.Time `json:"last_active_day,omitempty"`
	UpdatedAt      time.Time  `json:"updated_at"`
}

// UserTokenUsageDaily is one day's token aggregate for a user.
type UserTokenUsageDaily struct {
	Day          time.Time `json:"day"`
	InputTokens  int64     `json:"input_tokens"`
	OutputTokens int64     `json:"output_tokens"`
	TotalTokens  int64     `json:"total_tokens"`
	RunCount     int64     `json:"run_count"`
}

// UserProfile is the self-described, user-editable profile that personalizes a
// researcher's agent runs. It is persisted under UserAccount.metadata.profile.
type UserProfile struct {
	DisplayName       string `json:"display_name,omitempty"`
	Title             string `json:"title,omitempty"`
	Institution       string `json:"institution,omitempty"`
	ResearchInterests string `json:"research_interests,omitempty"`
	Bio               string `json:"bio,omitempty"`
}

// UpdateUserProfileInput carries a full replacement of the editable profile.
type UpdateUserProfileInput struct {
	UserID  string
	Profile UserProfile
}

type BisqueCredentialRecord struct {
	SessionID          string    `json:"session_id"`
	UserID             string    `json:"user_id"`
	OrgID              string    `json:"org_id,omitempty"`
	RootURL            string    `json:"root_url"`
	Username           string    `json:"username"`
	PasswordCiphertext string    `json:"password_ciphertext"`
	PasswordNonce      string    `json:"password_nonce"`
	PasswordKeyID      string    `json:"password_key_id"`
	PasswordAlgorithm  string    `json:"password_algorithm"`
	Status             string    `json:"status"`
	LastVerifiedAt     time.Time `json:"last_verified_at,omitempty"`
	CreatedAt          time.Time `json:"created_at"`
	UpdatedAt          time.Time `json:"updated_at"`
	Metadata           JSONMap   `json:"metadata"`
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

type UpsertBisqueCredentialInput struct {
	SessionID          string
	UserID             string
	OrgID              string
	RootURL            string
	Username           string
	PasswordCiphertext string
	PasswordNonce      string
	PasswordKeyID      string
	PasswordAlgorithm  string
	Status             string
	LastVerifiedAt     time.Time
	Metadata           JSONMap
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

type AcquireDataAgentJobLeaseInput struct {
	JobID       string
	OwnerUserID string
	OwnerOrgID  string
	WorkerID    string
	TTL         time.Duration
	Now         time.Time
}

type RenewDataAgentJobLeaseInput struct {
	JobID       string
	OwnerUserID string
	OwnerOrgID  string
	LeaseToken  string
	TTL         time.Duration
	Now         time.Time
}

type ReleaseDataAgentJobLeaseInput struct {
	JobID       string
	OwnerUserID string
	OwnerOrgID  string
	LeaseToken  string
}

type RecoverExpiredDataAgentJobLeasesInput struct {
	Now    time.Time
	Reason string
	Limit  int
}

type RecoverExpiredDataAgentJobLeasesResult struct {
	Checked      int
	RequeuedJobs []DataAgentJobRecord
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

type UpdateThreadInput struct {
	ThreadID string
	UserID   string
	Title    string
	Metadata JSONMap
}

type ApplyGeneratedThreadTitleInput struct {
	ThreadID   string
	RunID      string
	Title      string
	Generation JSONMap
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
