// The section-14.5 fixture suite: every phase, variant, overlay, priority
// rule, and degradation of the answer page's state seam. This suite is the
// spec of the page - a rendering change that needs a new state must land here
// first.

import { describe, expect, it } from "vitest";

import type {
  GateGuardrailsWire,
  TrainingModelRecord,
  TrainingModelStatus,
  TrainingModelVersionRecord,
} from "../../types";
import {
  comparatorPhrase,
  deriveBindingConstraint,
  deriveDigest,
  deriveOverlay,
  deriveTrainingPhase,
  groupVerdict,
  normalizeGateClauses,
  verdictSentence,
  type TrainingSnapshot,
} from "./state";

const NOW = new Date("2026-07-07T12:00:00Z");
const GOLD_HASH = "a3f2c9deadbeef";

const model = (overrides?: Partial<TrainingModelRecord>): TrainingModelRecord => ({
  key: "yolov5_rarespot",
  name: "RareSpot prairie-dog/burrow",
  framework: "PyTorch/YOLOv5",
  task_type: "detection",
  description: "",
  supports_training: true,
  supports_finetune: true,
  supports_inference: true,
  dimensions: ["2d"],
  default_config: {
    training_state: "configured",
    capabilities: ["SYNC", "ASSEMBLE", "FINETUNE", "BENCHMARK"],
  },
  ...overrides,
});

const benchmarkOnlyModel = (): TrainingModelRecord =>
  model({
    key: "megaseg",
    default_config: { training_state: "configured", capabilities: ["BENCHMARK"] },
  });

const version = (
  id: string,
  status: TrainingModelVersionRecord["status"] | string,
  metadata: Record<string, unknown> = {}
): TrainingModelVersionRecord => ({
  version_id: id,
  lineage_id: "yolov5_rarespot-shared",
  status: status as TrainingModelVersionRecord["status"],
  metrics: {},
  metadata,
  created_at: "2026-07-01T00:00:00Z",
  updated_at: "2026-07-01T00:00:00Z",
});

const passingGuardrails = (overrides?: Partial<GateGuardrailsWire>): GateGuardrailsWire => ({
  passed: true,
  reasons: [],
  gold_set_content_hash: GOLD_HASH,
  benchmarked_at: "2026-07-07T09:14:00Z",
  clauses: [
    {
      clause_key: "agg_map50",
      metric_path: "aggregate.map50",
      comparator: "max_drop_vs_active",
      value: 0.005,
      candidate_value: 0.842,
      baseline_value: 0.831,
      outcome: "passed",
    },
    {
      clause_key: "slice_held_map50",
      metric_path: "per_slice.held_out_test.map50",
      slice: "held_out_test",
      comparator: "max_drop_vs_active",
      value: 0.005,
      candidate_value: null,
      baseline_value: null,
      outcome: "excluded",
      reason: "excluded - held-out slice pending new survey data",
    },
  ],
  ...overrides,
});

const status = (overrides?: Partial<TrainingModelStatus>): TrainingModelStatus => ({
  dataset_name: "Prairie_Dog_Active_Learning",
  model_health: "watch",
  reviewed_images: 0,
  unreviewed_images: 0,
  class_counts: {},
  unsupported_class_counts: {},
  detection_counts: {},
  latest_metrics: {},
  benchmark_baseline: {},
  benchmark_latest_candidate: {},
  benchmark_ready: false,
  canonical_benchmark_ready: false,
  promotion_benchmark_ready: false,
  retrain_gate: false,
  retrain_gate_reasons: [],
  retrain_gate_counts: {},
  ...overrides,
});

const snapshot = (overrides?: Partial<TrainingSnapshot>): TrainingSnapshot => ({
  model: model(),
  status: status(),
  lineage: null,
  versions: [version("yolov5_rarespot-v0", "active")],
  retrainRequests: [],
  ...overrides,
});

const goldFrozen = (extra?: Record<string, unknown>) => ({
  gold: { gold_set_id: "gold-1", content_hash: GOLD_HASH, freeze_state: "frozen" as const, held_out_state: "pending_new_survey" as const, ...extra },
});

describe("deriveTrainingPhase - the ten phases", () => {
  it("phase 1: gold-blocked", () => {
    const state = deriveTrainingPhase(
      snapshot({ status: status({ gold: { freeze_state: "blocked", qualifying_prior_frames: 62, required_prior_frames: 100 } }) }),
      NOW
    );
    expect(state.phase).toBe("gold-blocked");
    expect(state.phaseClass).toBe("rest");
    expect(state.primary).toBeNull();
  });

  it("phase 2: gold-ready-to-freeze with the freeze primary", () => {
    const state = deriveTrainingPhase(
      snapshot({ status: status({ gold: { freeze_state: "ready", qualifying_prior_frames: 118 } }) }),
      NOW
    );
    expect(state.phase).toBe("gold-ready-to-freeze");
    expect(state.primary).toBe("freeze-gold");
    expect(state.freezeFailed).toBe(false);
  });

  it("phase 2 failed variant: no primary, freezeFailed set", () => {
    const state = deriveTrainingPhase(
      snapshot({ status: status({ gold: { freeze_state: "failed", freeze_failure_reasons: ["geo overlap"] } }) }),
      NOW
    );
    expect(state.phase).toBe("gold-ready-to-freeze");
    expect(state.primary).toBeNull();
    expect(state.freezeFailed).toBe(true);
  });

  it("phase 3: gold-freezing is in-flight with no primary", () => {
    const state = deriveTrainingPhase(
      snapshot({ status: status({ gold: { freeze_state: "freezing" } }) }),
      NOW
    );
    expect(state.phase).toBe("gold-freezing");
    expect(state.phaseClass).toBe("in-flight");
    expect(state.primary).toBeNull();
  });

  it("phase 4: gold-frozen-no-baseline offers run-baseline", () => {
    const state = deriveTrainingPhase(snapshot({ status: status(goldFrozen()) }), NOW);
    expect(state.phase).toBe("gold-frozen-no-baseline");
    expect(state.primary).toBe("run-baseline");
  });

  it("phase 5 rest: idle with no filled button anywhere", () => {
    const state = deriveTrainingPhase(
      snapshot({
        status: status({
          ...goldFrozen(),
          canonical_benchmark_ready: true,
        }),
        versions: [version("yolov5_rarespot-v0", "active", { guardrails: passingGuardrails() })],
      }),
      NOW
    );
    expect(state.phase).toBe("idle");
    expect(state.phaseClass).toBe("rest");
    expect(state.primary).toBeNull();
  });

  it("phase 5 attention: Gate A ready offers request-retrain", () => {
    const state = deriveTrainingPhase(
      snapshot({
        status: status({ ...goldFrozen(), retrain_gate: true, canonical_benchmark_ready: true }),
        versions: [version("yolov5_rarespot-v0", "active", { guardrails: passingGuardrails() })],
      }),
      NOW
    );
    expect(state.phase).toBe("idle");
    expect(state.phaseClass).toBe("attention");
    expect(state.gateReady).toBe(true);
    expect(state.primary).toBe("request-retrain");
  });

  it("phase 6: retrain-running", () => {
    const state = deriveTrainingPhase(
      snapshot({
        retrainRequests: [
          {
            request_id: "r1",
            training_job_id: "j1",
            status: "running",
            created_at: "2026-07-07T10:00:00Z",
            gating_summary: {},
          },
        ],
      }),
      NOW
    );
    expect(state.phase).toBe("retrain-running");
    expect(state.phaseClass).toBe("in-flight");
  });

  it("phase 7: candidate-unbenchmarked offers run-benchmark", () => {
    const state = deriveTrainingPhase(
      snapshot({
        status: status({ ...goldFrozen(), canonical_benchmark_ready: true }),
        versions: [
          version("yolov5_rarespot-v1", "candidate"),
          version("yolov5_rarespot-v0", "active", { guardrails: passingGuardrails() }),
        ],
      }),
      NOW
    );
    expect(state.phase).toBe("candidate-unbenchmarked");
    expect(state.primary).toBe("run-benchmark");
  });

  it("phase 8: benchmark-running from the echo", () => {
    const state = deriveTrainingPhase(
      snapshot({
        status: status({ ...goldFrozen(), running_benchmark: { version_id: "baseline", started_at: "2026-07-07T11:00:00Z" } }),
      }),
      NOW
    );
    expect(state.phase).toBe("benchmark-running");
    expect(state.phaseClass).toBe("in-flight");
  });

  it("phase 9a: candidate-passed offers promote-canary", () => {
    const state = deriveTrainingPhase(
      snapshot({
        status: status({ ...goldFrozen(), canonical_benchmark_ready: true }),
        versions: [
          version("yolov5_rarespot-v1", "candidate", { guardrails: passingGuardrails() }),
          version("yolov5_rarespot-v0", "active", { guardrails: passingGuardrails() }),
        ],
      }),
      NOW
    );
    expect(state.phase).toBe("candidate-passed");
    expect(state.primary).toBe("promote-canary");
  });

  it("phase 9b: candidate-failed has no primary", () => {
    const state = deriveTrainingPhase(
      snapshot({
        status: status({ ...goldFrozen(), canonical_benchmark_ready: true }),
        versions: [
          version("yolov5_rarespot-v1", "candidate", { guardrails: passingGuardrails({ passed: false, reasons: ["prairie_dog recall 0.41 - below the absolute floor of 0.50"] }) }),
          version("yolov5_rarespot-v0", "active", { guardrails: passingGuardrails() }),
        ],
      }),
      NOW
    );
    expect(state.phase).toBe("candidate-failed");
    expect(state.primary).toBeNull();
  });

  it("phase 10: canary-soaking without soak met has no primary", () => {
    const state = deriveTrainingPhase(
      snapshot({
        status: status({
          ...goldFrozen(),
          canonical_benchmark_ready: true,
          canary: {
            canary_version_id: "yolov5_rarespot-v1",
            soak_started_at: "2026-07-07T06:00:00Z", // 6h elapsed < 24h
            runs_observed: 14,
            min_soak_runs: 20,
            min_soak_hours: 24,
            traffic_fraction: 0.1,
          },
        }),
        versions: [
          version("yolov5_rarespot-v1", "canary", { guardrails: passingGuardrails() }),
          version("yolov5_rarespot-v0", "active", { guardrails: passingGuardrails() }),
        ],
      }),
      NOW
    );
    expect(state.phase).toBe("canary-soaking");
    expect(state.primary).toBeNull();
    expect(state.soakMet).toBe(false);
  });
});

describe("priority rule - most-decision-urgent first", () => {
  const soakedCanary = {
    canary_version_id: "yolov5_rarespot-v1",
    soak_started_at: "2026-07-05T00:00:00Z", // 60h elapsed
    runs_observed: 25,
    min_soak_runs: 20,
    min_soak_hours: 24,
    traffic_fraction: 0.1,
  };

  it("in-flight beats a promote-ready canary", () => {
    const state = deriveTrainingPhase(
      snapshot({
        status: status({
          gold: { content_hash: GOLD_HASH, freeze_state: "freezing" },
          canary: soakedCanary,
        }),
        versions: [version("yolov5_rarespot-v1", "canary", { guardrails: passingGuardrails() })],
      }),
      NOW
    );
    expect(state.phase).toBe("gold-freezing");
  });

  it("canary-promote-ready beats a fresh candidate verdict, demoting it to the digest", () => {
    const state = deriveTrainingPhase(
      snapshot({
        status: status({ ...goldFrozen({ held_out_state: "populated" }), canary: soakedCanary }),
        versions: [
          version("yolov5_rarespot-v2", "candidate", { guardrails: passingGuardrails() }),
          version("yolov5_rarespot-v1", "canary", { guardrails: passingGuardrails() }),
          version("yolov5_rarespot-v0", "active", { guardrails: passingGuardrails() }),
        ],
      }),
      NOW
    );
    expect(state.phase).toBe("canary-soaking");
    expect(state.primary).toBe("promote-active");
    expect(state.demoted.some((note) => note.state === "candidate-verdict")).toBe(true);
  });

  it("held-out pending replaces promote-active with the audited override", () => {
    const state = deriveTrainingPhase(
      snapshot({
        status: status({ ...goldFrozen(), canary: soakedCanary }),
        versions: [
          version("yolov5_rarespot-v1", "canary", { guardrails: passingGuardrails() }),
          version("yolov5_rarespot-v0", "active", { guardrails: passingGuardrails() }),
        ],
      }),
      NOW
    );
    expect(state.phase).toBe("canary-soaking");
    expect(state.primary).toBeNull();
    expect(state.requiresOverride).toBe(true);
  });

  it("one-canary rule: a passed verdict shows but its promote primary is suppressed while a canary soaks", () => {
    const state = deriveTrainingPhase(
      snapshot({
        status: status({
          ...goldFrozen(),
          canary: { ...soakedCanary, runs_observed: 5 }, // not promote-ready
        }),
        versions: [
          version("yolov5_rarespot-v2", "candidate", { guardrails: passingGuardrails() }),
          version("yolov5_rarespot-v1", "canary", { guardrails: passingGuardrails() }),
          version("yolov5_rarespot-v0", "active", { guardrails: passingGuardrails() }),
        ],
      }),
      NOW
    );
    expect(state.phase).toBe("candidate-passed");
    expect(state.primary).toBeNull();
    expect(state.canarySuppressesPromote).toBe(true);
    expect(state.demoted.some((note) => note.state === "canary-soaking")).toBe(true);
  });

  it("gate re-arming while a canary soaks is demoted, not hidden", () => {
    const state = deriveTrainingPhase(
      snapshot({
        status: status({ ...goldFrozen(), retrain_gate: true, canary: { ...soakedCanary, runs_observed: 5 } }),
        versions: [
          version("yolov5_rarespot-v1", "canary", { guardrails: passingGuardrails() }),
          version("yolov5_rarespot-v0", "active", { guardrails: passingGuardrails() }),
        ],
      }),
      NOW
    );
    expect(state.phase).toBe("canary-soaking");
    expect(state.demoted.some((note) => note.state === "gate-ready")).toBe(true);
  });
});

describe("degradations and variants", () => {
  it("no gold echo at all: generic no-gold rest line, no primary even when Gate A is ready", () => {
    const state = deriveTrainingPhase(snapshot({ status: status({ retrain_gate: true }) }), NOW);
    expect(state.phase).toBe("idle");
    expect(state.phaseClass).toBe("rest");
    expect(state.goldUnknown).toBe(true);
    expect(state.primary).toBeNull();
  });

  it("a stale verdict (different gold hash) does not derive phase 9", () => {
    const state = deriveTrainingPhase(
      snapshot({
        status: status({ ...goldFrozen(), canonical_benchmark_ready: true }),
        versions: [
          version("yolov5_rarespot-v1", "candidate", {
            guardrails: passingGuardrails({ gold_set_content_hash: "old-hash" }),
          }),
          version("yolov5_rarespot-v0", "active", { guardrails: passingGuardrails() }),
        ],
      }),
      NOW
    );
    expect(state.phase).toBe("candidate-unbenchmarked");
  });

  it("dismissal exits phase 9: a rejected candidate no longer occupies the focal card", () => {
    const state = deriveTrainingPhase(
      snapshot({
        status: status({ ...goldFrozen(), retrain_gate: true, canonical_benchmark_ready: true }),
        versions: [
          version("yolov5_rarespot-v1", "rejected", { guardrails: passingGuardrails({ passed: false }) }),
          version("yolov5_rarespot-v0", "active", { guardrails: passingGuardrails() }),
        ],
      }),
      NOW
    );
    expect(state.phase).toBe("idle");
    expect(state.primary).toBe("request-retrain");
  });

  it("benchmark-only variant: gold lifecycle derives, retrain never does", () => {
    const readyState = deriveTrainingPhase(
      snapshot({
        model: benchmarkOnlyModel(),
        status: status({ gold: { freeze_state: "ready" } }),
        versions: [version("megaseg-v0", "active")],
      }),
      NOW
    );
    expect(readyState.phase).toBe("gold-ready-to-freeze");
    expect(readyState.primary).toBe("freeze-gold");
    expect(readyState.benchmarkOnly).toBe(true);

    const gateState = deriveTrainingPhase(
      snapshot({
        model: benchmarkOnlyModel(),
        status: status({ ...goldFrozen(), retrain_gate: true, canonical_benchmark_ready: true }),
        versions: [version("megaseg-v0", "active", { guardrails: passingGuardrails() })],
      }),
      NOW
    );
    expect(gateState.phase).toBe("idle");
    expect(gateState.primary).toBeNull();
    expect(gateState.gateReady).toBe(false);
  });

  it("not-configured overlay derives from the registry record", () => {
    expect(deriveOverlay(model({ default_config: { training_state: "not_configured" } }))).toBe("not-configured");
    expect(deriveOverlay(model())).toBeNull();
  });
});

describe("normalizeGateClauses + verdict grouping + arithmetic", () => {
  it("pinned clauses render with self-explanatory tolerances", () => {
    const gate = normalizeGateClauses(passingGuardrails());
    expect(gate.degraded).toBe(false);
    expect(gate.passed).toBe(1);
    expect(gate.excluded).toBe(1);
    expect(gate.evaluated).toBe(1);
    expect(gate.clauses[0].tolerance).toBe("≤ 0.005 drop vs active");
    expect(gate.clauses[0].label).toBe("mAP50 (all)");
  });

  it("per-class wildcard expansions surface the class name", () => {
    const gate = normalizeGateClauses({
      passed: false,
      clauses: [
        {
          clause_key: "class_recall_abs",
          metric_path: "per_class.prairie_dog.recall_at_op",
          comparator: "abs_floor",
          value: 0.5,
          candidate_value: 0.41,
          baseline_value: 0.52,
          outcome: "failed",
          reason: "prairie_dog recall 0.41 - below the absolute floor of 0.50",
        },
      ],
    });
    expect(gate.clauses[0].label).toBe("prairie_dog recall floor");
    expect(gate.clauses[0].tolerance).toBe("≥ 0.5");
  });

  it("degrades to reasons-only rows when clauses[] is absent", () => {
    const gate = normalizeGateClauses({ passed: false, reasons: ["x regressed", "y excluded - under 10 gold boxes"] });
    expect(gate.degraded).toBe(true);
    expect(gate.failed).toBe(1);
    expect(gate.excluded).toBe(1);
  });

  it("groups clauses into plain language and never greens an excluded group", () => {
    const groups = groupVerdict(normalizeGateClauses(passingGuardrails()));
    const heldOut = groups.find((row) => row.label === "Performs on unseen geography");
    expect(heldOut?.outcome).toBe("excluded");
    const aggregate = groups.find((row) => row.label === "Overall accuracy has not regressed");
    expect(aggregate?.outcome).toBe("passed");
  });

  it("ungrouped clause keys (another model) fall back to one row per clause", () => {
    const groups = groupVerdict(
      normalizeGateClauses({
        passed: true,
        clauses: [
          {
            clause_key: "agg_miou",
            metric_path: "aggregate.miou",
            comparator: "max_drop_vs_active",
            value: 0.01,
            candidate_value: 0.81,
            baseline_value: 0.8,
            outcome: "passed",
          },
        ],
      })
    );
    expect(groups).toHaveLength(1);
    expect(groups[0].label).toBe("agg_miou");
  });

  it("verdict arithmetic is exclusion-honest", () => {
    const withExclusion = normalizeGateClauses(passingGuardrails());
    expect(verdictSentence("v1", withExclusion)).toBe(
      "Candidate v1 passed every check that can run — 1 of 1. One check is waiting on new survey data."
    );
    // Forbidden forms: never "all N" with exclusions, never "M of N" that reads as failure.
    expect(verdictSentence("v1", withExclusion)).not.toContain("passed all");

    const clean = normalizeGateClauses({
      passed: true,
      clauses: [
        {
          clause_key: "agg_map50",
          metric_path: "aggregate.map50",
          comparator: "max_drop_vs_active",
          value: 0.005,
          candidate_value: 0.84,
          baseline_value: 0.83,
          outcome: "passed",
        },
      ],
    });
    expect(verdictSentence("v1", clean, "gold-v1")).toBe("Candidate v1 passed all 1 checks against gold-v1.");

    const failed = normalizeGateClauses(passingGuardrails({ clauses: undefined, passed: false, reasons: ["r1", "r2"] }));
    expect(verdictSentence("v1", failed)).toContain("failed 2 of 2 evaluated checks — it will not be promoted.");
  });
});

describe("binding constraint + digest", () => {
  it("picks the largest-deficit unmet threshold", () => {
    const constraint = deriveBindingConstraint(
      status({
        retrain_gate_counts: { reviewed_images: 62, total_objects: 40, per_class: { prairie_dog: 7, burrow: 25 } } as never,
        retrain_gate_thresholds: {
          min_reviewed: 50,
          min_new_objects: 200,
          min_per_class_objects: { prairie_dog: 20, burrow: 20 },
          min_days: 3,
        },
      })
    );
    expect(constraint).toEqual({ label: "new labeled objects", have: 40, needed: 200 });
  });

  it("returns null when thresholds are absent (the FE never hardcodes them)", () => {
    expect(deriveBindingConstraint(status())).toBeNull();
  });

  it("digest filters by last visit and caps at three lines", () => {
    const events = [1, 2, 3, 4, 5].map((day) => ({
      ts: `2026-07-0${day}T00:00:00Z`,
      kind: "sync",
      summary: `event ${day}`,
    }));
    const since = deriveDigest(events, "2026-07-01T12:00:00Z", NOW);
    expect(since.map((event) => event.summary)).toEqual(["event 2", "event 3", "event 4"]);
    expect(deriveDigest([], null, NOW)).toEqual([]);
  });
});

describe("comparatorPhrase map is total", () => {
  it("covers all four comparators", () => {
    expect(comparatorPhrase("max_drop_vs_active", 0.02)).toBe("≤ 0.02 drop vs active");
    expect(comparatorPhrase("max_rise_vs_active", 0.1)).toBe("≤ 0.1 rise vs active");
    expect(comparatorPhrase("abs_floor", 0.5)).toBe("≥ 0.5");
    expect(comparatorPhrase("abs_ceiling", 3)).toBe("≤ 3");
  });
});
