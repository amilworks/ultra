package domain

import "time"

// GoldGate training records (M0). The registry row is the model-agnostic seam:
// a new set of weights onboards as a registry row + seed data, never as new
// columns, routes, or enums. Design:
// planning/2026-07-07-goldgate-continual-finetuning-plan.md

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
