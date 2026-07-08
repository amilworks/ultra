package store

import (
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

// Canonical GoldGate M0 seed, mirrored three ways: the schema.sql INSERTs
// (Postgres), the MemoryStore constructor (dev/tests), and the shared
// cross-language fixture at backend/contracts/training/yolov5_rarespot.manifest.json
// (pinned by Go and Python parity tests). 'yolov5_rarespot' is the ONE canonical
// model-key spelling across FE, Go, DB, NATS subjects, and storage dir segments.

const (
	TrainingSeedModelKey  = "yolov5_rarespot"
	TrainingSeedVersionID = "yolov5_rarespot-v0"
	TrainingSeedLineageID = "yolov5_rarespot-shared"
	TrainingSeedDomainID  = "ecology"
)

func trainingSeedModel(now time.Time) domain.TrainingModelRecord {
	return domain.TrainingModelRecord{
		ModelKey:      TrainingSeedModelKey,
		TaskType:      "detection",
		DisplayName:   "RareSpot prairie-dog/burrow",
		DatasetFormat: "yolo_txt_tiles_512",
		MetricSchema:  "detection.v1",
		RequiresPhash: true,
		Capabilities:  []string{"SYNC", "ASSEMBLE", "FINETUNE", "BENCHMARK"},
		Executor: domain.JSONMap{
			"gpu_pool":            "titan",
			"min_vram_gb":         float64(8),
			"shm_gb":              float64(8),
			"wall_clock_budget_s": float64(21600),
			"cpu_only":            false,
		},
		Classes:              domain.JSONMap{"0": "prairie_dog", "1": "burrow"},
		LeakageDefensesExtra: []string{"aerial_geospatial_overlap"},
		Metadata: domain.JSONMap{
			"framework":         "PyTorch/YOLOv5",
			"description":       "Prairie dog and burrow detection on aerial survey imagery (512px tiles). Gold-gated continual finetuning.",
			"dimensions":        []any{"2d"},
			"workflow":          "rarespot_ecology",
			"gt_layer_priority": []any{"gt2", "New Ground Truth"},
		},
		CreatedAt: now,
	}
}

func trainingSeedDomain(now time.Time) domain.TrainingDomainRecord {
	return domain.TrainingDomainRecord{
		DomainID:    TrainingSeedDomainID,
		Name:        "Ecology",
		Description: "Field-survey detection and segmentation models.",
		Metadata:    domain.JSONMap{},
		CreatedAt:   now,
		UpdatedAt:   now,
	}
}

func trainingSeedLineage(now time.Time) domain.TrainingLineageRecord {
	return domain.TrainingLineageRecord{
		LineageID:       TrainingSeedLineageID,
		DomainID:        TrainingSeedDomainID,
		ModelKey:        TrainingSeedModelKey,
		Scope:           "shared",
		ActiveVersionID: TrainingSeedVersionID,
		Metadata:        domain.JSONMap{},
		CreatedAt:       now,
		UpdatedAt:       now,
	}
}

func trainingSeedVersion(now time.Time) domain.TrainingModelVersionRecord {
	activated := now
	return domain.TrainingModelVersionRecord{
		VersionID:  TrainingSeedVersionID,
		LineageID:  TrainingSeedLineageID,
		ModelKey:   TrainingSeedModelKey,
		Status:     "active",
		IsFrozen:   true,
		WeightsURI: "data/models/yolo/RareSpotWeights.pt",
		Metrics:    domain.JSONMap{},
		Metadata: domain.JSONMap{
			"is_baked":   true,
			"provenance": "pre-GoldGate checkpoint; trained --noval on all_overfit.yaml (lineage focuswin_generalize4 -> allfullneg_calibrate2); no held-out validation existed at training time",
		},
		ActivatedAt: &activated,
		CreatedAt:   now,
		UpdatedAt:   now,
	}
}

// trainingSeedGatePolicy mirrors the control_training_gate_policies INSERT in
// schema.sql (the fixture's canonical gate_policy; a value tune must change
// both together or the seed-parity tests fail).
func trainingSeedGatePolicy() domain.TrainingGatePolicyRecord {
	return domain.TrainingGatePolicyRecord{
		ModelKey:      TrainingSeedModelKey,
		MinReviewed:   50,
		MinNewObjects: 200,
		MinPerClassObjects: domain.JSONMap{
			"prairie_dog": float64(20),
			"burrow":      float64(20),
		},
		MinDays: 3,
	}
}

// trainingSeedGuardrailClauses mirrors the control_training_guardrail_clauses
// INSERT in schema.sql row-for-row (10 clauses; the fixture pins the keys).
func trainingSeedGuardrailClauses() []domain.TrainingGuardrailClauseRecord {
	clause := func(key string, metricPath string, comparator string, value float64, slice string, params domain.JSONMap) domain.TrainingGuardrailClauseRecord {
		return domain.TrainingGuardrailClauseRecord{
			ModelKey:   TrainingSeedModelKey,
			ClauseKey:  key,
			MetricPath: metricPath,
			Comparator: comparator,
			Value:      value,
			Slice:      slice,
			Params:     params,
			Enabled:    true,
			Required:   true,
		}
	}
	return []domain.TrainingGuardrailClauseRecord{
		clause("agg_map50", "aggregate.map50", "max_drop_vs_active", 0.005, "", domain.JSONMap{}),
		clause("agg_map50_95", "aggregate.map50_95", "max_drop_vs_active", 0.005, "", domain.JSONMap{}),
		clause("class_recall_delta", "per_class.*.recall_at_op", "max_drop_vs_active", 0.02, "", domain.JSONMap{}),
		clause("class_recall_abs", "per_class.*.recall_at_op", "abs_floor", 0.50, "", domain.JSONMap{}),
		clause("slice_prior_map50", "per_slice.prior_train.map50", "max_drop_vs_active", 0.02, "prior_train", domain.JSONMap{"min_label_count": float64(10)}),
		clause("slice_held_map50", "per_slice.held_out_test.map50", "max_drop_vs_active", 0.005, "held_out_test", domain.JSONMap{"min_label_count": float64(10)}),
		clause("class_ap50_collapse", "per_class.*.ap50", "max_drop_vs_active", 0.05, "", domain.JSONMap{}),
		clause("class_ap50_abs", "per_class.*.ap50", "abs_floor", 0.10, "", domain.JSONMap{"strict": true}),
		clause("fp_empty_ceiling", "aggregate.fp_per_empty_frame", "max_rise_vs_active", 0.10, "", domain.JSONMap{}),
		clause("precision_delta", "aggregate.precision_at_op", "max_drop_vs_active", 0.03, "", domain.JSONMap{}),
	}
}

func trainingSeedStatus() domain.TrainingModelStatusRecord {
	return domain.TrainingModelStatusRecord{
		ModelKey:               TrainingSeedModelKey,
		DatasetName:            "Prairie_Dog_Active_Learning",
		ModelHealth:            "watch",
		ClassCounts:            domain.JSONMap{},
		PerClassNewObjects:     domain.JSONMap{},
		UnsupportedClassCounts: domain.JSONMap{},
		ActiveModelVersion:     TrainingSeedVersionID,
		RetrainGate:            false,
		RetrainGateReasons: []string{
			"No reviewed training data has been synced yet - the sync path ships with M1.",
			"Cannot check the gold-set precondition - no gold set has been frozen yet.",
		},
		RetrainGateCounts: domain.JSONMap{},
		RetrainGateThresholds: domain.JSONMap{
			"min_reviewed":    float64(50),
			"min_new_objects": float64(200),
			"min_per_class_objects": map[string]any{
				"prairie_dog": float64(20),
				"burrow":      float64(20),
			},
			"min_days": float64(3),
		},
	}
}

// SeedTrainingModel is the in-memory analog of the documented seed migration
// (the ONLY direct-DB writes the M5 acceptance test permits): registry row +
// gate policy + guardrail clause rows + domain/lineage/version-0. Postgres
// gets the same rows via a SQL seed migration; MemoryStore gets them here so
// the acceptance walk is testable hermetically.
func (s *MemoryStore) SeedTrainingModel(
	model domain.TrainingModelRecord,
	trainingDomain domain.TrainingDomainRecord,
	lineage domain.TrainingLineageRecord,
	versionZero domain.TrainingModelVersionRecord,
	status domain.TrainingModelStatusRecord,
	policy domain.TrainingGatePolicyRecord,
	clauses []domain.TrainingGuardrailClauseRecord,
) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.training.models[model.ModelKey] = model
	if trainingDomain.DomainID != "" {
		s.training.domains[trainingDomain.DomainID] = trainingDomain
	}
	if lineage.LineageID != "" {
		s.training.lineages[lineage.LineageID] = lineage
	}
	if versionZero.VersionID != "" {
		s.training.versions[versionZero.VersionID] = versionZero
	}
	if status.ModelKey != "" {
		s.training.statuses[status.ModelKey] = status
	}
	if policy.ModelKey != "" {
		s.training.gatePolicies[policy.ModelKey] = policy
	}
	if len(clauses) > 0 {
		s.training.guardrailClauses[model.ModelKey] = clauses
	}
}
