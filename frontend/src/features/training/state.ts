// The state seam of the GoldGate answer page (plan section 14.5): one pure
// function derives exactly one of ten phases from the snapshot, a deterministic
// most-decision-urgent priority rule resolves concurrent truths (losers are
// DEMOTED to digest lines, never hidden), and every missing echo degrades per
// the section-14.5 table. This module IS the spec of the page - the fixture
// suite in state.test.ts covers every phase, variant, overlay, and degradation.

import type {
  GateClauseWire,
  GateGuardrailsWire,
  PrairieRetrainRecord,
  TrainingLineageRecord,
  TrainingModelRecord,
  TrainingModelStatus,
  TrainingModelVersionRecord,
  TrainingRecentEvent,
} from "../../types";

export type TrainingPhase =
  | "gold-blocked"
  | "gold-ready-to-freeze"
  | "gold-freezing"
  | "gold-frozen-no-baseline"
  | "idle"
  | "retrain-running"
  | "candidate-unbenchmarked"
  | "benchmark-running"
  | "candidate-passed"
  | "candidate-failed"
  | "canary-soaking";

export type TrainingPhaseClass = "rest" | "attention" | "in-flight";

export type TrainingPrimaryAction =
  | "freeze-gold"
  | "run-baseline"
  | "request-retrain"
  | "run-benchmark"
  | "promote-canary"
  | "promote-active";

export type TrainingDemotedNote = {
  state: string;
  summary: string;
};

export type TrainingSnapshot = {
  model: TrainingModelRecord | null;
  status: TrainingModelStatus;
  lineage: TrainingLineageRecord | null;
  versions: TrainingModelVersionRecord[];
  retrainRequests: PrairieRetrainRecord[];
};

export type TrainingPhaseState = {
  phase: TrainingPhase;
  phaseClass: TrainingPhaseClass;
  // The page's ONLY filled primary; null = no filled button anywhere (rule 7).
  primary: TrainingPrimaryAction | null;
  // Phase-5 variant: FINETUNE absent from the manifest capabilities.
  benchmarkOnly: boolean;
  // Phase-2 variant: freeze_state === 'failed' (suppresses the freeze primary).
  freezeFailed: boolean;
  // Phase-5 split: Gate A satisfied (attention) vs unmet (rest).
  gateReady: boolean;
  // Degradation: no gold echo at all - the focal card renders the generic
  // "No gold set yet" rest line and nothing about the gate can be green.
  goldUnknown: boolean;
  // One-canary rule: a verdict phase whose promote-canary primary is
  // suppressed because a canary is already soaking.
  canarySuppressesPromote: boolean;
  // Held-out pending: promote-to-active requires the audited override
  // (the component renders the quiet override secondary instead of a primary).
  requiresOverride: boolean;
  // Soak progress against policy (null until the canary echo exists).
  soakMet: boolean | null;
  candidate: TrainingModelVersionRecord | null;
  canary: TrainingModelVersionRecord | null;
  active: TrainingModelVersionRecord | null;
  demoted: TrainingDemotedNote[];
};

export type TrainingOverlay = "not-configured" | null;

const capabilityList = (model: TrainingModelRecord | null): string[] => {
  const raw = model?.default_config?.capabilities;
  if (!Array.isArray(raw)) {
    return [];
  }
  return raw.filter((item): item is string => typeof item === "string");
};

export const deriveOverlay = (model: TrainingModelRecord | null): TrainingOverlay => {
  const state = String(model?.default_config?.training_state ?? "");
  return state === "not_configured" ? "not-configured" : null;
};

export const hasCapability = (model: TrainingModelRecord | null, capability: string): boolean =>
  capabilityList(model).includes(capability);

const guardrailsOf = (version: TrainingModelVersionRecord | null): GateGuardrailsWire | null => {
  const raw = version?.metadata?.guardrails;
  if (!raw || typeof raw !== "object") {
    return null;
  }
  return raw as GateGuardrailsWire;
};

// A verdict counts only when it was produced against the CURRENT gold hash;
// without a gold echo (older backend) we degrade to accepting any verdict.
const verdictOnCurrentGold = (
  version: TrainingModelVersionRecord | null,
  status: TrainingModelStatus
): GateGuardrailsWire | null => {
  const guardrails = guardrailsOf(version);
  if (!guardrails || guardrails.passed === undefined) {
    return null;
  }
  const currentHash = status.gold?.content_hash;
  if (currentHash && guardrails.gold_set_content_hash && guardrails.gold_set_content_hash !== currentHash) {
    return null;
  }
  return guardrails;
};

const soakHoursElapsed = (soakStartedAt: string, now: Date): number => {
  const started = new Date(soakStartedAt).getTime();
  if (!Number.isFinite(started)) {
    return 0;
  }
  return Math.max(0, (now.getTime() - started) / 3_600_000);
};

export const deriveTrainingPhase = (snapshot: TrainingSnapshot, now: Date = new Date()): TrainingPhaseState => {
  const { model, status, versions, retrainRequests } = snapshot;
  const finetuneCapable = hasCapability(model, "FINETUNE");
  const benchmarkOnly = model !== null && !finetuneCapable;

  const active = versions.find((row) => row.status === "active") ?? null;
  const canary = versions.find((row) => row.status === "canary") ?? null;
  const candidate =
    versions.find((row) => row.status === "candidate") ?? null; // versions arrive newest-first

  const gold = status.gold ?? null;
  const goldUnknown = gold == null || gold.freeze_state === undefined;
  const freezeState = gold?.freeze_state;

  const activeRetrain = retrainRequests.find((row) =>
    ["queued", "running", "paused"].includes(row.status)
  );

  const candidateVerdict = verdictOnCurrentGold(candidate, status);
  const canaryVerdict = verdictOnCurrentGold(canary, status);
  const heldOutPending = gold?.held_out_state === "pending_new_survey";

  const canaryEcho = status.canary ?? null;
  const soakMet =
    canaryEcho == null
      ? null
      : canaryEcho.runs_observed >= canaryEcho.min_soak_runs &&
        soakHoursElapsed(canaryEcho.soak_started_at, now) >= canaryEcho.min_soak_hours;

  // Baseline existence: the active version's benchmark provenance, on the
  // current hash when both sides carry one (degrades to the readiness flag).
  const activeGuardrails = verdictOnCurrentGold(active, status);
  const baselineExists = activeGuardrails != null || status.canonical_benchmark_ready === true;

  const demoted: TrainingDemotedNote[] = [];
  const gateReady = status.retrain_gate === true && finetuneCapable;

  const build = (
    phase: TrainingPhase,
    phaseClass: TrainingPhaseClass,
    primary: TrainingPrimaryAction | null,
    extras?: Partial<TrainingPhaseState>
  ): TrainingPhaseState => ({
    phase,
    phaseClass,
    primary,
    benchmarkOnly,
    freezeFailed: freezeState === "failed",
    gateReady,
    goldUnknown,
    canarySuppressesPromote: false,
    requiresOverride: false,
    soakMet,
    candidate,
    canary,
    active,
    demoted,
    ...extras,
  });

  const noteDemotion = (state: string, summary: string) => {
    demoted.push({ state, summary });
  };

  // --- Priority tier 1: in-flight (phases 3, 6, 8) --------------------------
  if (freezeState === "freezing") {
    return build("gold-freezing", "in-flight", null);
  }
  if (activeRetrain) {
    return build("retrain-running", "in-flight", null);
  }
  if (status.running_benchmark) {
    return build("benchmark-running", "in-flight", null);
  }

  // --- Priority tier 2: canary promote-ready > verdict > canary-soaking -----
  const canaryQualifies = canary != null && canaryVerdict?.passed === true;
  const canaryPromoteReady = canaryQualifies && soakMet === true;
  const verdictPhase: TrainingPhase | null =
    candidateVerdict == null ? null : candidateVerdict.passed ? "candidate-passed" : "candidate-failed";

  if (canaryPromoteReady) {
    if (verdictPhase) {
      noteDemotion("candidate-verdict", "A new candidate has a benchmark verdict - shown after the canary resolves.");
    }
    if (gateReady) {
      noteDemotion("gate-ready", "Enough new data has accrued for another retrain - available after the canary resolves.");
    }
    // Pending held-out: the hard gate at canary -> active (plan section 8.2).
    if (heldOutPending) {
      return build("canary-soaking", "attention", null, { requiresOverride: true });
    }
    return build("canary-soaking", "attention", "promote-active");
  }

  if (verdictPhase) {
    if (canary) {
      noteDemotion("canary-soaking", "A canary is already soaking - resolve it before promoting another candidate.");
      // One-canary rule: 9a's filled primary is suppressed.
      return build(verdictPhase, "attention", null, { canarySuppressesPromote: verdictPhase === "candidate-passed" });
    }
    if (verdictPhase === "candidate-passed") {
      const promotionReady = candidateVerdict?.passed === true;
      return build(verdictPhase, "attention", promotionReady ? "promote-canary" : null);
    }
    return build(verdictPhase, "attention", null);
  }

  if (canary) {
    if (gateReady) {
      noteDemotion("gate-ready", "Enough new data has accrued for another retrain - available after the canary resolves.");
    }
    return build("canary-soaking", "attention", null);
  }

  // --- Priority tier 3: attention (2 > 4 > 7 > 5-ready) ---------------------
  if (freezeState === "ready" || freezeState === "failed") {
    const canFreeze = hasCapability(model, "BENCHMARK") && freezeState === "ready";
    return build("gold-ready-to-freeze", "attention", canFreeze ? "freeze-gold" : null);
  }
  if (freezeState === "frozen" && !baselineExists) {
    return build(
      "gold-frozen-no-baseline",
      "attention",
      hasCapability(model, "BENCHMARK") ? "run-baseline" : null
    );
  }
  if (candidate && !candidateVerdict) {
    return build(
      "candidate-unbenchmarked",
      "attention",
      hasCapability(model, "BENCHMARK") ? "run-benchmark" : null
    );
  }
  if (gateReady && !goldUnknown && freezeState === "frozen") {
    return build("idle", "attention", "request-retrain");
  }
  if (gateReady) {
    // Gate A says ready but the gold lifecycle isn't at frozen (or is unknown):
    // the fixed active-passes-gold precondition cannot hold, so no primary.
    return build("idle", "rest", null);
  }

  // --- Priority tier 4: rest (1, 5-rest) -------------------------------------
  if (freezeState === "blocked") {
    return build("gold-blocked", "rest", null);
  }
  return build("idle", "rest", null);
};

// --- Gate B clause normalization (plan section 14.4) -------------------------

export type GateClause = {
  key: string;
  metricPath: string;
  slice: string | null;
  label: string;
  candidate: number | null;
  active: number | null;
  tolerance: string;
  comparator: GateClauseWire["comparator"];
  outcome: "passed" | "failed" | "excluded";
  reason: string | null;
};

export type NormalizedGate = {
  clauses: GateClause[];
  // True when clauses[] was absent and rows were derived from reasons alone -
  // the See-the-numbers table and report link are omitted (section 14.4).
  degraded: boolean;
  passed: number;
  failed: number;
  excluded: number;
  evaluated: number;
};

// The comparator -> phrase map is normative: the Tolerance column must be
// self-explanatory without knowing the engine (section 14.4).
export const comparatorPhrase = (comparator: GateClauseWire["comparator"], value: number): string => {
  switch (comparator) {
    case "max_drop_vs_active":
      return `≤ ${value} drop vs active`;
    case "max_rise_vs_active":
      return `≤ ${value} rise vs active`;
    case "abs_floor":
      return `≥ ${value}`;
    case "abs_ceiling":
      return `≤ ${value}`;
  }
};

// Per-model, presentation-only display labels; fallback is the clause_key.
const RARESPOT_CLAUSE_LABELS: Record<string, string> = {
  agg_map50: "mAP50 (all)",
  agg_map50_95: "mAP50-95 (all)",
  class_recall_delta: "per-class recall vs active",
  class_recall_abs: "per-class recall floor",
  slice_prior_map50: "prior-data mAP50",
  slice_held_map50: "held-out mAP50",
  class_ap50_collapse: "per-class AP50 vs active",
  class_ap50_abs: "per-class AP50 floor",
  fp_empty_ceiling: "FP per empty frame",
  precision_delta: "precision vs active",
};

const clauseLabel = (wire: GateClauseWire): string => {
  const base = RARESPOT_CLAUSE_LABELS[wire.clause_key] ?? wire.clause_key;
  // Per-class wildcard expansions carry the class name in the metric path
  // (per_class.<name>.<metric>) - surface it so rows are distinguishable.
  const match = /^per_class\.([^.*]+)\./.exec(wire.metric_path);
  return match ? `${match[1]} ${base.replace(/^per-class /, "")}` : base;
};

export const normalizeGateClauses = (guardrails: GateGuardrailsWire | null): NormalizedGate => {
  if (guardrails?.clauses && guardrails.clauses.length > 0) {
    const clauses = guardrails.clauses.map((wire): GateClause => ({
      key: wire.clause_key,
      metricPath: wire.metric_path,
      slice: wire.slice ?? null,
      label: clauseLabel(wire),
      candidate: wire.candidate_value ?? null,
      active: wire.baseline_value ?? null,
      tolerance: comparatorPhrase(wire.comparator, wire.value),
      comparator: wire.comparator,
      outcome: wire.outcome,
      reason: wire.reason ?? null,
    }));
    return {
      clauses,
      degraded: false,
      passed: clauses.filter((row) => row.outcome === "passed").length,
      failed: clauses.filter((row) => row.outcome === "failed").length,
      excluded: clauses.filter((row) => row.outcome === "excluded").length,
      evaluated: clauses.filter((row) => row.outcome !== "excluded").length,
    };
  }
  const reasons = guardrails?.reasons ?? [];
  const clauses = reasons.map((reason, index): GateClause => {
    const excluded = /excluded/i.test(reason);
    return {
      key: `reason-${index}`,
      metricPath: "",
      slice: null,
      label: reason,
      candidate: null,
      active: null,
      tolerance: "",
      comparator: "abs_floor",
      outcome: excluded ? "excluded" : "failed",
      reason,
    };
  });
  return {
    clauses,
    degraded: true,
    passed: 0,
    failed: clauses.filter((row) => row.outcome === "failed").length,
    excluded: clauses.filter((row) => row.outcome === "excluded").length,
    evaluated: clauses.filter((row) => row.outcome !== "excluded").length,
  };
};

// --- Plain-language verdict grouping (section 14.4, layer 2) ----------------

export type VerdictGroup = {
  label: string;
  outcome: "passed" | "failed" | "excluded";
  reasons: string[];
};

const RARESPOT_VERDICT_GROUPS: Array<{ label: string; keys: string[] }> = [
  { label: "Finds prairie dogs at least as well as the current model", keys: ["class_recall_delta", "class_recall_abs"] },
  { label: "No extra false alarms on empty frames", keys: ["fp_empty_ceiling", "precision_delta"] },
  { label: "Has not forgotten older survey data", keys: ["slice_prior_map50"] },
  { label: "Performs on unseen geography", keys: ["slice_held_map50"] },
  { label: "Overall accuracy has not regressed", keys: ["agg_map50", "agg_map50_95", "class_ap50_collapse", "class_ap50_abs"] },
];

export const groupVerdict = (gate: NormalizedGate): VerdictGroup[] => {
  if (gate.degraded) {
    return gate.clauses.map((row) => ({
      label: row.label,
      outcome: row.outcome,
      reasons: row.reason ? [row.reason] : [],
    }));
  }
  const groups: VerdictGroup[] = [];
  const grouped = new Set<string>();
  for (const group of RARESPOT_VERDICT_GROUPS) {
    const members = gate.clauses.filter((row) => group.keys.includes(row.key));
    if (members.length === 0) {
      continue;
    }
    members.forEach((row) => grouped.add(row.key));
    const outcome = members.some((row) => row.outcome === "failed")
      ? "failed"
      : members.some((row) => row.outcome === "passed")
        ? "passed"
        : "excluded";
    groups.push({
      label: group.label,
      outcome,
      reasons: members.filter((row) => row.reason).map((row) => String(row.reason)),
    });
  }
  // Ungrouped clauses (a different model's set) render one row per clause -
  // onboarding a model must never require editing this map.
  for (const row of gate.clauses) {
    if (!grouped.has(row.key)) {
      groups.push({ label: row.label, outcome: row.outcome, reasons: row.reason ? [row.reason] : [] });
    }
  }
  return groups;
};

// --- The exclusion-honest verdict arithmetic (section 14.6-V, normative) ----

export const verdictSentence = (
  versionId: string,
  gate: NormalizedGate,
  goldLabel: string = "the gold set"
): string => {
  if (gate.failed > 0) {
    return `Candidate ${versionId} failed ${gate.failed} of ${gate.evaluated} evaluated checks — it will not be promoted.`;
  }
  if (gate.excluded > 0) {
    const waiting = gate.excluded === 1 ? "One check is" : `${gate.excluded} checks are`;
    return `Candidate ${versionId} passed every check that can run — ${gate.passed} of ${gate.evaluated}. ${waiting} waiting on new survey data.`;
  }
  return `Candidate ${versionId} passed all ${gate.evaluated} checks against ${goldLabel}.`;
};

// --- The one binding number for the rest line (section 14.4-F) --------------

export type BindingConstraint = {
  label: string;
  have: number;
  needed: number;
};

const THRESHOLD_LABELS: Record<string, string> = {
  min_reviewed: "reviewed images",
  min_new_objects: "new labeled objects",
};

export const deriveBindingConstraint = (status: TrainingModelStatus): BindingConstraint | null => {
  const counts = status.retrain_gate_counts ?? {};
  const thresholds = (status.retrain_gate_thresholds ?? {}) as Record<string, unknown>;
  const candidates: BindingConstraint[] = [];

  const minReviewed = Number(thresholds.min_reviewed);
  if (Number.isFinite(minReviewed)) {
    const have = Number(counts.reviewed_images ?? status.reviewed_images ?? 0);
    if (have < minReviewed) {
      candidates.push({ label: THRESHOLD_LABELS.min_reviewed, have, needed: minReviewed });
    }
  }
  const minObjects = Number(thresholds.min_new_objects);
  if (Number.isFinite(minObjects)) {
    const have = Number(counts.total_objects ?? 0);
    if (have < minObjects) {
      candidates.push({ label: THRESHOLD_LABELS.min_new_objects, have, needed: minObjects });
    }
  }
  const perClass = thresholds.min_per_class_objects;
  if (perClass && typeof perClass === "object") {
    const perClassCounts = (counts as Record<string, unknown>).per_class;
    for (const [name, rawNeeded] of Object.entries(perClass as Record<string, unknown>)) {
      const needed = Number(rawNeeded);
      if (!Number.isFinite(needed)) {
        continue;
      }
      const have = Number(
        perClassCounts && typeof perClassCounts === "object"
          ? ((perClassCounts as Record<string, unknown>)[name] ?? 0)
          : 0
      );
      if (have < needed) {
        candidates.push({ label: `new ${name.replace(/_/g, " ")} labels`, have, needed });
      }
    }
  }
  if (candidates.length === 0) {
    return null;
  }
  candidates.sort((a, b) => (b.needed - b.have) - (a.needed - a.have));
  return candidates[0];
};

// --- Since-your-last-visit digest (section 14.4-G) ---------------------------

export const LAST_VISIT_STORAGE_PREFIX = "training.lastVisit.";
const DEFAULT_DIGEST_WINDOW_DAYS = 30;

export const deriveDigest = (
  events: TrainingRecentEvent[] | undefined,
  lastVisitIso: string | null,
  now: Date = new Date()
): TrainingRecentEvent[] => {
  if (!events || events.length === 0) {
    return [];
  }
  const cutoff = lastVisitIso
    ? new Date(lastVisitIso).getTime()
    : now.getTime() - DEFAULT_DIGEST_WINDOW_DAYS * 24 * 3_600_000;
  if (!Number.isFinite(cutoff)) {
    return [];
  }
  return events
    .filter((event) => {
      const ts = new Date(event.ts).getTime();
      return Number.isFinite(ts) && ts > cutoff;
    })
    .slice(0, 3);
};
