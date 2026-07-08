package domain

import "time"

// GoldGate training records (M0). The registry row is the model-agnostic seam:
// a new set of weights onboards as a registry row + seed data, never as new
// columns, routes, or enums. Design:
// planning/2026-07-07-goldgate-continual-finetuning-plan.md

// TrainingJobPhases is the complete, frozen set of training job types — the
// "one enum PR": these five literals exist once per language (Python mirror:
// ultra_deepagents.training.manifests.TRAINING_JOB_PHASES), pinned to the
// shared fixture by tests on both sides. M3 and M5 add ZERO literals; a model
// selects from these via its manifest capabilities, never by adding phases.
var TrainingJobPhases = []string{
	"training.sync",
	"training.assemble",
	"training.finetune",
	"training.benchmark",
	"training.gold_freeze",
}

func IsTrainingJobPhase(jobType string) bool {
	for _, phase := range TrainingJobPhases {
		if jobType == phase {
			return true
		}
	}
	return false
}

type TrainingModelRecord struct {
	ModelKey             string    `json:"model_key"`
	TaskType             string    `json:"task_type"`
	DisplayName          string    `json:"display_name"`
	DatasetFormat        string    `json:"dataset_format"`
	MetricSchema         string    `json:"metric_schema"`
	RequiresPhash        bool      `json:"requires_phash"`
	Capabilities         []string  `json:"capabilities"`
	Executor             JSONMap   `json:"executor"`
	Classes              JSONMap   `json:"classes,omitempty"`
	LeakageDefensesExtra []string  `json:"leakage_defenses_extra"`
	Metadata             JSONMap   `json:"metadata"`
	CreatedAt            time.Time `json:"created_at"`
}

type TrainingDomainRecord struct {
	DomainID    string    `json:"domain_id"`
	Name        string    `json:"name"`
	Description string    `json:"description,omitempty"`
	Metadata    JSONMap   `json:"metadata"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

type TrainingLineageRecord struct {
	LineageID       string    `json:"lineage_id"`
	DomainID        string    `json:"domain_id"`
	ModelKey        string    `json:"model_key"`
	Scope           string    `json:"scope"`
	OwnerUserID     string    `json:"owner_user_id,omitempty"`
	ParentLineageID string    `json:"parent_lineage_id,omitempty"`
	ActiveVersionID string    `json:"active_version_id,omitempty"`
	Metadata        JSONMap   `json:"metadata"`
	CreatedAt       time.Time `json:"created_at"`
	UpdatedAt       time.Time `json:"updated_at"`
}

type TrainingModelVersionRecord struct {
	VersionID     string     `json:"version_id"`
	LineageID     string     `json:"lineage_id"`
	ModelKey      string     `json:"model_key"`
	Status        string     `json:"status"`
	IsFrozen      bool       `json:"is_frozen"`
	WeightsURI    string     `json:"weights_uri,omitempty"`
	SourceJobID   string     `json:"source_job_id,omitempty"`
	ArtifactRunID string     `json:"artifact_run_id,omitempty"`
	Metrics       JSONMap    `json:"metrics"`
	Metadata      JSONMap    `json:"metadata"`
	ActivatedAt   *time.Time `json:"activated_at,omitempty"`
	CreatedAt     time.Time  `json:"created_at"`
	UpdatedAt     time.Time  `json:"updated_at"`
}

type TrainingModelStatusRecord struct {
	ModelKey               string     `json:"model_key"`
	DatasetName            string     `json:"dataset_name"`
	DatasetID              string     `json:"dataset_id,omitempty"`
	ModelHealth            string     `json:"model_health"`
	ReviewedImages         int64      `json:"reviewed_images"`
	UnreviewedImages       int64      `json:"unreviewed_images"`
	ClassCounts            JSONMap    `json:"class_counts"`
	PerClassNewObjects     JSONMap    `json:"per_class_new_objects"`
	UnsupportedClassCounts JSONMap    `json:"unsupported_class_counts"`
	LastSyncAt             *time.Time `json:"last_sync_at,omitempty"`
	LastRetrainAt          *time.Time `json:"last_retrain_at,omitempty"`
	ActiveModelVersion     string     `json:"active_model_version,omitempty"`
	RetrainGate            bool       `json:"retrain_gate"`
	RetrainGateReasons     []string   `json:"retrain_gate_reasons"`
	RetrainGateCounts      JSONMap    `json:"retrain_gate_counts"`
	RetrainGateThresholds  JSONMap    `json:"retrain_gate_thresholds"`
}

type TrainingRetrainRequestRecord struct {
	RequestID                 string     `json:"request_id"`
	ModelKey                  string     `json:"model_key"`
	TrainingJobID             string     `json:"training_job_id,omitempty"`
	Status                    string     `json:"status"`
	Note                      string     `json:"note,omitempty"`
	Error                     string     `json:"error,omitempty"`
	ModelVersion              string     `json:"model_version,omitempty"`
	GatingSummary             JSONMap    `json:"gating_summary"`
	BenchmarkReportArtifactID string     `json:"benchmark_report_artifact_id,omitempty"`
	RequestedByUserID         string     `json:"requested_by_user_id,omitempty"`
	CreatedAt                 time.Time  `json:"created_at"`
	StartedAt                 *time.Time `json:"started_at,omitempty"`
	FinishedAt                *time.Time `json:"finished_at,omitempty"`
}

// ---------------------------------------------------------------------------
// GoldGate training writes (M1): jobs/leases/events clone the data-agent job
// family; gold sets carry the freeze-input lifecycle; versions + benchmark
// runs + guardrail clauses feed the M2 gate engine.
// ---------------------------------------------------------------------------

type TrainingJobRecord struct {
	JobID             string    `json:"job_id"`
	ModelKey          string    `json:"model_key"`
	JobType           string    `json:"job_type"`
	Status            string    `json:"status"`
	GPUPool           string    `json:"gpu_pool,omitempty"`
	Params            JSONMap   `json:"params"`
	ProgressCompleted int64     `json:"progress_completed"`
	ProgressTotal     int64     `json:"progress_total"`
	Error             string    `json:"error,omitempty"`
	OwnerUserID       string    `json:"owner_user_id"`
	OwnerOrgID        string    `json:"owner_org_id,omitempty"`
	CreatedByUserID   string    `json:"created_by_user_id,omitempty"`
	CreatedAt         time.Time `json:"created_at"`
	UpdatedAt         time.Time `json:"updated_at"`
	StartedAt         time.Time `json:"started_at,omitempty"`
	CompletedAt       time.Time `json:"completed_at,omitempty"`
	OutputSummary     JSONMap   `json:"output_summary"`
	Metadata          JSONMap   `json:"metadata"`
}

type CreateTrainingJobInput struct {
	JobID           string
	ModelKey        string
	JobType         string
	GPUPool         string
	Params          JSONMap
	OwnerUserID     string
	OwnerOrgID      string
	CreatedByUserID string
	Metadata        JSONMap
}

type UpdateTrainingJobStatusInput struct {
	JobID             string
	Status            string
	Error             string
	OutputSummary     JSONMap
	ProgressCompleted int64
	ProgressTotal     int64
}

type TrainingJobEventRecord struct {
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

type AppendTrainingJobEventInput struct {
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

type TrainingJobLeaseRecord struct {
	JobID          string    `json:"job_id"`
	WorkerID       string    `json:"worker_id"`
	LeaseToken     string    `json:"lease_token"`
	LeaseExpiresAt time.Time `json:"lease_expires_at"`
	CreatedAt      time.Time `json:"created_at"`
	UpdatedAt      time.Time `json:"updated_at"`
}

type AcquireTrainingJobLeaseInput struct {
	JobID    string
	WorkerID string
	TTL      time.Duration
	Now      time.Time
}

type RenewTrainingJobLeaseInput struct {
	JobID      string
	LeaseToken string
	TTL        time.Duration
	Now        time.Time
}

type ReleaseTrainingJobLeaseInput struct {
	JobID      string
	LeaseToken string
}

type TrainingGoldSetRecord struct {
	GoldSetID        string     `json:"gold_set_id"`
	ModelKey         string     `json:"model_key"`
	Version          int64      `json:"version"`
	ContentHash      string     `json:"content_hash,omitempty"`
	ItemCount        int64      `json:"item_count"`
	LabelStats       JSONMap    `json:"label_stats"`
	StrataSummary    JSONMap    `json:"strata_summary"`
	SplitManifestURI string     `json:"split_manifest_uri,omitempty"`
	Provenance       JSONMap    `json:"provenance"`
	Status           string     `json:"status"`
	CreatedAt        time.Time  `json:"created_at"`
	CreatedByUserID  string     `json:"created_by_user_id"`
	FrozenAt         *time.Time `json:"frozen_at,omitempty"`
}

type CreateTrainingGoldSetInput struct {
	GoldSetID       string
	ModelKey        string
	CreatedByUserID string
}

type UpdateTrainingGoldSetStateInput struct {
	GoldSetID        string
	Status           string
	ContentHash      string
	ItemCount        int64
	LabelStats       JSONMap
	StrataSummary    JSONMap
	Provenance       JSONMap
	SplitManifestURI string
	FrozenAt         *time.Time
}

type TrainingGoldItemRecord struct {
	GoldSetID     string  `json:"gold_set_id"`
	ItemID        string  `json:"item_id"`
	SourceRef     JSONMap `json:"source_ref"`
	Slice         string  `json:"slice"`
	LabelKind     string  `json:"label_kind"`
	ContentSHA256 string  `json:"content_sha256"`
	Phash         *string `json:"phash,omitempty"`
	GTLabelSHA256 string  `json:"gt_label_sha256"`
	GTLabelURI    string  `json:"gt_label_uri"`
	Width         *int64  `json:"width,omitempty"`
	Height        *int64  `json:"height,omitempty"`
	Metadata      JSONMap `json:"metadata"`
	FootprintGeom JSONMap `json:"footprint_geom,omitempty"`
	StrataTags    JSONMap `json:"strata_tags"`
}

// TrainingGoldItemInput crosses the HTTP boundary (the gold_freeze worker
// posts items in its gold-result), so it carries wire tags.
type TrainingGoldItemInput struct {
	ItemID        string  `json:"item_id"`
	SourceRef     JSONMap `json:"source_ref"`
	Slice         string  `json:"slice"`
	LabelKind     string  `json:"label_kind"`
	ContentSHA256 string  `json:"content_sha256"`
	Phash         *string `json:"phash"`
	GTLabelSHA256 string  `json:"gt_label_sha256"`
	GTLabelURI    string  `json:"gt_label_uri"`
	Width         *int64  `json:"width"`
	Height        *int64  `json:"height"`
	Metadata      JSONMap `json:"metadata"`
	FootprintGeom JSONMap `json:"footprint_geom"`
	StrataTags    JSONMap `json:"strata_tags"`
}

type CreateTrainingModelVersionInput struct {
	VersionID   string
	LineageID   string
	ModelKey    string
	Status      string
	WeightsURI  string
	SourceJobID string
	Metrics     JSONMap
	Metadata    JSONMap
}

type UpdateTrainingModelVersionInput struct {
	VersionID   string
	Status      string
	Metrics     JSONMap
	Metadata    JSONMap
	ActivatedAt *time.Time
}

type TrainingModelVersionEventRecord struct {
	EventID            string    `json:"event_id"`
	VersionID          string    `json:"version_id"`
	EventType          string    `json:"event_type"`
	ActorUserID        string    `json:"actor_user_id"`
	FromStatus         string    `json:"from_status,omitempty"`
	ToStatus           string    `json:"to_status,omitempty"`
	BenchmarkRunID     string    `json:"benchmark_run_id,omitempty"`
	GoldSetContentHash string    `json:"gold_set_content_hash,omitempty"`
	Reason             string    `json:"reason,omitempty"`
	CreatedAt          time.Time `json:"created_at"`
}

type AppendTrainingModelVersionEventInput struct {
	EventID            string
	VersionID          string
	EventType          string
	ActorUserID        string
	FromStatus         string
	ToStatus           string
	BenchmarkRunID     string
	GoldSetContentHash string
	Reason             string
}

type CreateTrainingRetrainRequestInput struct {
	RequestID         string
	ModelKey          string
	TrainingJobID     string
	Note              string
	RequestedByUserID string
	GatingSummary     JSONMap
}

type UpdateTrainingRetrainRequestInput struct {
	RequestID     string
	Status        string
	Error         string
	ModelVersion  string
	TrainingJobID string
	StartedAt     *time.Time
	FinishedAt    *time.Time
}

type TrainingBenchmarkRunRecord struct {
	RunID              string    `json:"run_id"`
	ModelVersionID     string    `json:"model_version_id"`
	GoldSetID          string    `json:"gold_set_id"`
	GoldSetContentHash string    `json:"gold_set_content_hash"`
	MetricSchema       string    `json:"metric_schema"`
	KernelVersion      string    `json:"kernel_version"`
	Metrics            JSONMap   `json:"metrics"`
	GuardrailsPassed   bool      `json:"guardrails_passed"`
	GuardrailsReasons  []string  `json:"guardrails_reasons"`
	ReportURI          string    `json:"report_uri"`
	CreatedAt          time.Time `json:"created_at"`
}

type InsertTrainingBenchmarkRunInput struct {
	RunID              string
	ModelVersionID     string
	GoldSetID          string
	GoldSetContentHash string
	MetricSchema       string
	KernelVersion      string
	Metrics            JSONMap
	GuardrailsPassed   bool
	GuardrailsReasons  []string
	ReportURI          string
}

type TrainingGuardrailClauseRecord struct {
	ModelKey   string  `json:"model_key"`
	ClauseKey  string  `json:"clause_key"`
	MetricPath string  `json:"metric_path"`
	Comparator string  `json:"comparator"`
	Value      float64 `json:"value"`
	Slice      string  `json:"slice,omitempty"`
	Params     JSONMap `json:"params"`
	Enabled    bool    `json:"enabled"`
	Required   bool    `json:"required"`
}

type TrainingGatePolicyRecord struct {
	ModelKey           string  `json:"model_key"`
	MinReviewed        int64   `json:"min_reviewed"`
	MinNewObjects      int64   `json:"min_new_objects"`
	MinPerClassObjects JSONMap `json:"min_per_class_objects"`
	MinDays            int64   `json:"min_days"`
}
