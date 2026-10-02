import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { ApiClient } from "@/lib/api";
import { TrainingDashboard } from "./TrainingDashboard";

const loadTrainingSnapshot = vi.hoisted(() => vi.fn());

vi.mock("@/features/training/client", () => ({
  loadTrainingSnapshot,
}));

const NOW = "2026-08-22T16:00:00Z";

const failedCandidateSnapshot = {
  models: [
    {
      key: "yolov5_rarespot",
      name: "Prairie YOLO",
      framework: "yolov5",
      task_type: "object_detection",
      description: "Prairie detector",
      supports_training: true,
      supports_finetune: true,
      supports_inference: true,
      dimensions: ["image"],
      default_config: {
        training_state: "configured",
        capabilities: ["BENCHMARK"],
      },
    },
  ],
  model: {
    key: "yolov5_rarespot",
    name: "Prairie YOLO",
    framework: "yolov5",
    task_type: "object_detection",
    description: "Prairie detector",
    supports_training: true,
    supports_finetune: true,
    supports_inference: true,
    dimensions: ["image"],
    default_config: {
      training_state: "configured",
      capabilities: ["BENCHMARK"],
    },
  },
  status: {
    model_key: "yolov5_rarespot",
    dataset_name: "Prairie_Dog_Active_Learning",
    model_health: "watch",
    reviewed_images: 1_240,
    unreviewed_images: 370,
    class_counts: {},
    unsupported_class_counts: {},
    detection_counts: {},
    latest_metrics: {},
    benchmark_baseline: {},
    benchmark_latest_candidate: {},
    benchmark_ready: true,
    canonical_benchmark_ready: true,
    promotion_benchmark_ready: false,
    retrain_gate: false,
    retrain_gate_reasons: [],
    retrain_gate_counts: {},
    gold: {
      gold_set_id: "gold-1",
      gold_set_version: 1,
      content_hash: "gold-hash",
      freeze_state: "frozen",
      held_out_state: "pending_new_survey",
    },
  },
  lineage: null,
  versions: [
    {
      version_id: "version_candidate",
      lineage_id: "lineage_prairie",
      status: "candidate",
      metrics: {},
      metadata: {
        guardrails: {
          passed: false,
          reasons: ["Prairie-dog recall is below the floor."],
          benchmarked_at: NOW,
          gold_set_content_hash: "gold-hash",
          clauses: [
            {
              clause_key: "class_recall_abs",
              metric_path: "per_class.prairie_dog.recall_at_op",
              comparator: "abs_floor",
              value: 0.5,
              candidate_value: 0.41,
              baseline_value: 0.52,
              outcome: "failed",
              reason: "Prairie-dog recall is below the floor.",
            },
          ],
        },
      },
      created_at: NOW,
      updated_at: NOW,
    },
    {
      version_id: "version_active",
      lineage_id: "lineage_prairie",
      status: "active",
      metrics: {},
      metadata: {},
      created_at: NOW,
      updated_at: NOW,
    },
  ],
  retrainRequests: [],
};

describe("TrainingDashboard", () => {
  beforeEach(() => {
    window.localStorage.clear();
    loadTrainingSnapshot.mockReset();
    loadTrainingSnapshot.mockResolvedValue(failedCandidateSnapshot);
  });

  it("announces a failed benchmark action and preserves the initiating focus", async () => {
    const runTrainingBenchmark = vi
      .fn()
      .mockRejectedValue(new Error("Benchmark service unavailable."));
    const apiClient = { runTrainingBenchmark } as unknown as ApiClient;

    render(
      <TrainingDashboard
        apiClient={apiClient}
        resources={[]}
        resourcesLoading={false}
        resourcesError={null}
        isAdmin={false}
      />
    );

    const rerun = await screen.findByRole("button", { name: "Re-run benchmark" });
    rerun.focus();
    fireEvent.click(rerun);

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Benchmark service unavailable.");
    expect(rerun).toHaveFocus();
    expect(runTrainingBenchmark).toHaveBeenCalledWith("yolov5_rarespot", {
      mode: "promotion_packet",
      version_id: "version_candidate",
    });
  });
});
