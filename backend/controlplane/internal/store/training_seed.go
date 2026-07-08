package store

import (
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

// Canonical GoldGate M0 seed, mirrored three ways: the schema.sql INSERTs
// (Postgres), the MemoryStore constructor (dev/tests), and the shared
// cross-language fixture at backend/contracts/training/yolov5_rarespot.manifest.json
// (pinned by Go and Python parity tests). 'yolov5_rarespot' is the ONE canonical
// model-key spelling across FE, Go, DB, NATS subjects, and barrel dir segments.

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
