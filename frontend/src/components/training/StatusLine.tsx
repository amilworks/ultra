// The status region (§14.4-S): the page's ONLY live region. Dot + one
// three-clause sentence + the always-visible rollback ghost. The phase clause
// derives from the SAME phase state as the focal card - the two can never
// disagree. The rollback absence exception is keyed to LINEAGE EVIDENCE (no
// retired version exists), never to a phase.

import type { TrainingModelStatus, TrainingModelVersionRecord } from "../../types";
import type { TrainingPhaseState } from "../../features/training/state";
import { formatDay, timeAgo } from "./format";

type StatusTone = "active" | "muted" | "danger" | "running";

const phaseClause = (state: TrainingPhaseState, status: TrainingModelStatus): { text: string; tone: StatusTone } => {
  const gold = status.gold;
  switch (state.phase) {
    case "gold-blocked":
      return {
        text: `gold set blocked on coverage — ${gold?.qualifying_prior_frames ?? 0} of ${gold?.required_prior_frames ?? 100} frames`,
        tone: "muted",
      };
    case "gold-ready-to-freeze":
      return state.freezeFailed
        ? { text: "gold freeze failed — details below", tone: "danger" }
        : { text: "gold set ready to freeze", tone: "muted" };
    case "gold-freezing":
      return { text: "freezing the gold set — leakage checks running", tone: "running" };
    case "gold-frozen-no-baseline":
      return { text: "gold set frozen — baseline not measured yet", tone: "muted" };
    case "retrain-running":
      return { text: "retraining in progress", tone: "running" };
    case "candidate-unbenchmarked":
      return { text: `candidate ${state.candidate?.version_id ?? ""} awaiting benchmark`, tone: "active" };
    case "benchmark-running":
      return { text: "benchmark running", tone: "running" };
    case "candidate-passed":
      return { text: `candidate ${state.candidate?.version_id ?? ""} passed — awaiting promotion below`, tone: "active" };
    case "candidate-failed":
      return { text: `candidate ${state.candidate?.version_id ?? ""} failed the gate — details below`, tone: "danger" };
    case "canary-soaking":
      return { text: `${state.canary?.version_id ?? "a canary"} in canary`, tone: "active" };
    case "idle":
      if (state.goldUnknown) {
        return { text: "no gold set yet", tone: "muted" };
      }
      if (state.benchmarkOnly) {
        return { text: "benchmark-only model", tone: "muted" };
      }
      return state.gateReady
        ? { text: "ready to retrain", tone: "active" }
        : { text: status.canonical_benchmark_ready ? "passing gold" : "gold baseline pending", tone: "active" };
  }
};

export function StatusLine({
  state,
  status,
  versions,
  onRollback,
}: {
  state: TrainingPhaseState;
  status: TrainingModelStatus;
  versions: TrainingModelVersionRecord[];
  onRollback: () => void;
}) {
  const active = state.active;
  const activeClause = active
    ? (active.metadata as Record<string, unknown> | undefined)?.is_baked
      ? "Serving the baked weights (v0) — no trained version yet"
      : `${active.version_id} active since ${formatDay(status.active_version_activated_at ?? active.activated_at ?? active.created_at)}`
    : "No active version";
  const clause = phaseClause(state, status);
  const syncClause = status.last_sync_at ? `last synced ${timeAgo(status.last_sync_at)}` : "no sync yet";
  const previousExists = versions.some((row) => row.status === "retired");
  const dotTone = clause.tone === "danger" ? "danger" : clause.tone === "active" ? undefined : "muted";

  return (
    <div className="training-status-line" role="status">
      <p className="training-status-sentence">
        <span className="training-status-dot" data-tone={dotTone} aria-hidden="true" />
        <span data-clause="active">
          <strong>{activeClause}</strong>
        </span>
        <span className="training-status-sep" aria-hidden="true">
          ·
        </span>
        <span data-clause="phase" className={clause.tone === "danger" ? "training-clause-fail" : undefined}>
          {clause.text}
        </span>
        <span className="training-status-sep" aria-hidden="true">
          ·
        </span>
        <span data-clause="sync">{syncClause}</span>
      </p>
      {previousExists ? (
        <button type="button" className="training-rollback-ghost" onClick={onRollback}>
          Roll back to the previous version
        </button>
      ) : (
        <span className="training-rollback-absent">No previous version to roll back to.</span>
      )}
    </div>
  );
}
