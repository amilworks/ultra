// The morphing focal area (§14.4-F): one region, no border and no background
// of its own, hosting the page's ONLY filled primary. Rest phases collapse to
// a single disclosure line carrying the one binding number; attention and
// in-flight phases render open. Microcopy is the §14.6 catalog verbatim.

import { Button } from "@/components/ui/button";
import type { GateGuardrailsWire, TrainingModelStatus } from "../../types";
import {
  deriveBindingConstraint,
  type TrainingPhaseState,
} from "../../features/training/state";
import { formatCount, formatDay, formatTimestamp, shortHash } from "./format";
import { GateBVerdict } from "./GateBVerdict";

const goldVersionLabel = (status: TrainingModelStatus): string => {
  const version = status.gold?.gold_set_version;
  return version ? `gold-v${version}` : "the gold set";
};

function GateAChecklist({ status, goldOk }: { status: TrainingModelStatus; goldOk: boolean | null }) {
  const counts = (status.retrain_gate_counts ?? {}) as Record<string, unknown>;
  const thresholds = (status.retrain_gate_thresholds ?? {}) as Record<string, unknown>;
  const rows: Array<{ label: string; text: string; met: boolean }> = [];

  const push = (label: string, have: number, needed: number | null) => {
    if (needed == null) {
      rows.push({ label, text: `${label} — ${formatCount(have)}`, met: true });
      return;
    }
    const met = have >= needed;
    rows.push({
      label,
      text: `${label} — ${formatCount(have)} of ${formatCount(needed)} needed`,
      met,
    });
  };

  const minReviewed = Number(thresholds.min_reviewed);
  push(
    "Reviewed images",
    Number(counts.reviewed_images ?? status.reviewed_images ?? 0),
    Number.isFinite(minReviewed) ? minReviewed : null
  );
  const minObjects = Number(thresholds.min_new_objects);
  push("New labeled objects", Number(counts.total_objects ?? 0), Number.isFinite(minObjects) ? minObjects : null);
  const perClassNeeded = thresholds.min_per_class_objects;
  if (perClassNeeded && typeof perClassNeeded === "object") {
    const perClassCounts = (counts.per_class ?? {}) as Record<string, unknown>;
    for (const [name, rawNeeded] of Object.entries(perClassNeeded as Record<string, unknown>)) {
      const needed = Number(rawNeeded);
      if (Number.isFinite(needed)) {
        push(`New ${name.replace(/_/g, " ")} labels`, Number(perClassCounts[name] ?? 0), needed);
      }
    }
  }

  return (
    <div className="training-checklist">
      {rows.map((row) => (
        <span key={row.label} data-state={row.met ? "met" : "unmet"}>
          {row.text} {row.met ? <span aria-hidden="true">✓</span> : <span aria-hidden="true">—</span>}
          <span className="sr-only">{row.met ? "satisfied" : "unmet"}</span>
        </span>
      ))}
      <span data-state={goldOk === true ? "met" : goldOk === false ? "unmet" : "excluded"}>
        {goldOk === true
          ? "Active model passes gold ✓"
          : goldOk === false
            ? "Active model must pass gold first —"
            : "Cannot check the gold precondition — no gold set frozen yet —"}
      </span>
    </div>
  );
}

export function FocalCard({
  state,
  status,
  onFreezeGold,
  onRunBaseline,
  onRequestRetrain,
  onRunBenchmark,
  onPromoteCanary,
  onPromoteActive,
  onOverridePromote,
  onDismissCandidate,
  onRetryFreeze,
}: {
  state: TrainingPhaseState;
  status: TrainingModelStatus;
  onFreezeGold: () => void;
  onRunBaseline: () => void;
  onRequestRetrain: () => void;
  onRunBenchmark: () => void;
  onPromoteCanary: () => void;
  onPromoteActive: () => void;
  onOverridePromote: () => void;
  onDismissCandidate: () => void;
  onRetryFreeze: () => void;
}) {
  const gold = status.gold;
  const heldOutPending = gold?.held_out_state === "pending_new_survey";
  const goldLabel = goldVersionLabel(status);

  // --- Rest phases: a single collapsed line with the one binding number -----
  if (state.phaseClass === "rest") {
    let line: string;
    if (state.phase === "gold-blocked") {
      line = `Gold set blocked on annotation coverage — ${formatCount(gold?.qualifying_prior_frames ?? 0)} of ${formatCount(gold?.required_prior_frames ?? 100)} required reviewed prior frames. It unblocks as review continues.`;
    } else if (state.goldUnknown) {
      line = "No gold set yet — benchmarks are not meaningful until one is frozen. The gold set is the fixed exam every future model must pass.";
    } else if (state.benchmarkOnly) {
      line = `${status.model_key ?? "This model"} is benchmark-only — no retraining. Data and benchmarks stay live.`;
    } else {
      const constraint = deriveBindingConstraint(status);
      line = constraint
        ? `Nothing needs your attention — ${formatCount(constraint.have)} of ${formatCount(constraint.needed)} ${constraint.label} before the next retrain. Data syncs from BisQue automatically.`
        : "Nothing needs your attention. Data syncs from BisQue automatically.";
    }
    return (
      <details className="training-disclosure training-focal">
        <summary>
          <span className="training-disclosure-chevron" aria-hidden="true">
            ›
          </span>
          {line}
        </summary>
        <div className="training-disclosure-body">
          <GateAChecklist status={status} goldOk={state.goldUnknown ? null : status.canonical_benchmark_ready} />
        </div>
      </details>
    );
  }

  // --- Attention / in-flight phases: an open region -------------------------
  return (
    <div className="training-focal" tabIndex={-1} data-phase={state.phase}>
      {state.phase === "gold-freezing" ? (
        <p className="training-focal-lead">
          <span className="training-status-dot" data-tone="muted" aria-hidden="true" />
          Freezing {goldLabel} — running leakage checks…
        </p>
      ) : null}

      {state.phase === "retrain-running" ? (
        <p className="training-focal-lead">
          <span className="training-status-dot" data-tone="muted" aria-hidden="true" />
          Retraining in progress. A candidate will appear here with its benchmark verdict.
        </p>
      ) : null}

      {state.phase === "benchmark-running" ? (
        <p className="training-focal-lead">
          <span className="training-status-dot" data-tone="muted" aria-hidden="true" />
          Benchmark running against {goldLabel}
          {status.running_benchmark?.version_id === "baseline"
            ? " — baseline (active model)…"
            : ` — candidate ${status.running_benchmark?.version_id ?? ""}…`}
        </p>
      ) : null}

      {state.phase === "gold-ready-to-freeze" ? (
        state.freezeFailed ? (
          <>
            <p className="training-focal-lead">
              <span className="training-status-dot" data-tone="danger" aria-hidden="true" />
              Gold freeze failed — the leakage checks found violations. The draft is kept.
            </p>
            {(gold?.freeze_failure_reasons ?? []).map((reason) => (
              <p key={reason} className="training-gloss">
                {reason}
              </p>
            ))}
            <div className="training-actions">
              <button type="button" className="training-ghost-action" onClick={onRetryFreeze}>
                Retry freeze
              </button>
            </div>
          </>
        ) : (
          <>
            <p className="training-focal-lead">
              <span className="training-status-dot" aria-hidden="true" />
              The gold set is ready to freeze — {formatCount(gold?.qualifying_prior_frames ?? 0)} qualifying reviewed
              frames.
            </p>
            <p className="training-gloss">
              The gold set is a frozen exam of reviewed field images every new version must pass before it can replace
              the current model.
            </p>
            {state.primary === "freeze-gold" ? (
              <div className="training-actions">
                <Button className="training-primary-action" onClick={onFreezeGold}>
                  Review & freeze gold set
                </Button>
              </div>
            ) : null}
          </>
        )
      ) : null}

      {state.phase === "gold-frozen-no-baseline" ? (
        <>
          <p className="training-focal-lead">
            <span className="training-status-dot" aria-hidden="true" />
            {goldLabel} is frozen, but the active model has no score on it yet. Run the baseline benchmark so
            regressions can be measured.
          </p>
          <p className="training-gloss">
            The baseline is the current model's own score on the gold set — the bar a candidate has to clear.
          </p>
          {state.primary === "run-baseline" ? (
            <div className="training-actions">
              <Button className="training-primary-action" onClick={onRunBaseline}>
                Run baseline benchmark
              </Button>
            </div>
          ) : null}
        </>
      ) : null}

      {state.phase === "idle" && state.gateReady ? (
        <>
          <p className="training-focal-lead">
            <span className="training-status-dot" aria-hidden="true" />
            Ready to retrain.
          </p>
          <GateAChecklist status={status} goldOk={status.canonical_benchmark_ready} />
          {state.primary === "request-retrain" ? (
            <div className="training-actions">
              <Button className="training-primary-action" onClick={onRequestRetrain}>
                Request retraining
              </Button>
            </div>
          ) : null}
        </>
      ) : null}

      {state.phase === "candidate-unbenchmarked" ? (
        <>
          <p className="training-focal-lead">
            <span className="training-status-dot" aria-hidden="true" />
            Candidate {state.candidate?.version_id} is registered but has not been benchmarked against {goldLabel}.
          </p>
          <div className="training-actions">
            <span>
              {state.primary === "run-benchmark" ? (
                <Button className="training-primary-action" onClick={onRunBenchmark}>
                  Run benchmark
                </Button>
              ) : null}
            </span>
            <button type="button" className="training-ghost-action" onClick={onDismissCandidate}>
              Dismiss candidate
            </button>
          </div>
        </>
      ) : null}

      {(state.phase === "candidate-passed" || state.phase === "candidate-failed") && state.candidate ? (
        <GateBVerdict
          state={state}
          guardrails={(state.candidate.metadata?.guardrails ?? {}) as GateGuardrailsWire}
          goldVersionLabel={goldLabel}
          heldOutPending={heldOutPending}
          onPromote={onPromoteCanary}
          onDismiss={onDismissCandidate}
          onRerunBenchmark={onRunBenchmark}
        />
      ) : null}

      {state.phase === "canary-soaking" ? (
        <>
          <p className="training-focal-lead">
            <span className="training-status-dot" aria-hidden="true" />
            {state.canary?.version_id} is serving as canary
            {status.canary
              ? ` — ${Math.round((status.canary.traffic_fraction ?? 0.1) * 100)}% of runs. ${formatCount(status.canary.runs_observed)} of ${formatCount(status.canary.min_soak_runs)} runs soaked.`
              : ". Soak progress unavailable."}
          </p>
          <p className="training-gloss">
            A canary handles 1 in 10 real runs while the current model handles the rest — a low-risk live trial.
          </p>
          {(() => {
            const guardrails = (state.canary?.metadata?.guardrails ?? null) as GateGuardrailsWire | null;
            if (!guardrails?.benchmarked_at) {
              return null;
            }
            return (
              <p className="training-provenance">
                Passed its gate · benchmarked {formatDay(guardrails.benchmarked_at)} against {goldLabel}
                {guardrails.gold_set_content_hash ? ` (${shortHash(guardrails.gold_set_content_hash)})` : ""}
                {guardrails.report_uri ? (
                  <>
                    {" · "}
                    <a href={guardrails.report_uri} target="_blank" rel="noreferrer">
                      View full report
                    </a>
                  </>
                ) : null}
              </p>
            );
          })()}
          {status.canary?.drift_note ? <p className="training-gloss">{status.canary.drift_note}</p> : null}
          {state.requiresOverride ? (
            <>
              <div className="training-caveat-row">
                <span className="training-held-out-flag">Generalization unchecked</span>
                <p>No held-out survey data yet — promoting to active requires an audited override.</p>
              </div>
              <div className="training-actions">
                <button type="button" className="training-ghost-action" onClick={onOverridePromote}>
                  Promote anyway (override)
                </button>
              </div>
            </>
          ) : state.primary === "promote-active" ? (
            <div className="training-actions">
              <Button className="training-primary-action" onClick={onPromoteActive}>
                Promote to active
              </Button>
            </div>
          ) : null}
        </>
      ) : null}

      {status.last_benchmark_at && state.phase === "idle" ? (
        <p className="training-provenance">Last benchmark {formatTimestamp(status.last_benchmark_at)}.</p>
      ) : null}
    </div>
  );
}
