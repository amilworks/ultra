// The verdict experience (§14.4-F): three layers - exclusion-honest sentence,
// plain-language grouped checked-list, and the See-the-numbers clause table
// (defaults OPEN on fail, with the mandatory trust-restoring sentence). The
// table renders the pinned guardrails.clauses[] shape data-driven; when the
// shape is absent the page degrades to reasons-only and omits the table.

import { Fragment } from "react";

import { Button } from "@/components/ui/button";
import type { GateGuardrailsWire } from "../../types";
import {
  groupVerdict,
  normalizeGateClauses,
  verdictSentence,
  type TrainingPhaseState,
} from "../../features/training/state";
import { formatMetric, formatTimestamp, shortHash } from "./format";

export function GateBVerdict({
  state,
  guardrails,
  goldVersionLabel,
  heldOutPending,
  onPromote,
  onDismiss,
  onRerunBenchmark,
}: {
  state: TrainingPhaseState;
  guardrails: GateGuardrailsWire;
  goldVersionLabel: string;
  heldOutPending: boolean;
  onPromote: () => void;
  onDismiss: () => void;
  onRerunBenchmark: () => void;
}) {
  const gate = normalizeGateClauses(guardrails);
  const groups = groupVerdict(gate);
  const failed = state.phase === "candidate-failed";
  const versionId = state.candidate?.version_id ?? "";

  return (
    <div>
      <p className="training-focal-lead">
        <span className="training-status-dot" data-tone={failed ? "danger" : undefined} aria-hidden="true" />
        <span>{verdictSentence(versionId, gate, goldVersionLabel)}</span>
      </p>
      {failed ? <p className="training-gloss">The candidate was not promoted — your current model is unchanged.</p> : null}
      <p className="training-gloss">A candidate is a newly trained version that has not yet earned its way into production.</p>

      <div className="training-verdict-groups">
        {groups.map((group) => (
          <div key={group.label} data-outcome={group.outcome}>
            <span className={group.outcome === "excluded" ? "training-stat-tile-label" : undefined}>
              {group.label}
              {group.outcome === "excluded" ? " — excluded, waiting on new survey data" : ""}
            </span>
            <span aria-hidden="true" className={group.outcome === "failed" ? "training-clause-fail" : undefined}>
              {group.outcome === "passed" ? "✓" : group.outcome === "failed" ? "×" : "—"}
            </span>
            <span className="sr-only">{group.outcome}</span>
          </div>
        ))}
      </div>
      {groups
        .filter((group) => group.outcome === "failed")
        .flatMap((group) => group.reasons)
        .map((reason) => (
          <p key={reason} className="training-verdict-reason training-gloss">
            {reason}
          </p>
        ))}

      {gate.degraded ? (
        <p className="training-provenance">
          Full clause detail unavailable — benchmark on the current backend to see the numbers.
        </p>
      ) : (
        <details className="training-disclosure" open={failed}>
          <summary>
            <span className="training-disclosure-chevron" aria-hidden="true">
              ›
            </span>
            See the numbers
          </summary>
          <div className="training-disclosure-body">
            <table className="training-clause-table">
              <thead>
                <tr>
                  <th scope="col">Check</th>
                  <th scope="col" data-numeric="true">
                    Candidate
                  </th>
                  <th scope="col" data-numeric="true">
                    Active
                  </th>
                  <th scope="col" data-numeric="true">
                    Tolerance
                  </th>
                  <th scope="col" data-numeric="true">
                    Result
                  </th>
                </tr>
              </thead>
              <tbody>
                {gate.clauses.map((clause) => (
                  <Fragment key={clause.key + clause.metricPath}>
                    <tr>
                      <td>{clause.label}</td>
                      <td data-numeric="true">{formatMetric(clause.candidate)}</td>
                      <td data-numeric="true">{formatMetric(clause.active)}</td>
                      <td data-numeric="true">{clause.tolerance}</td>
                      <td data-numeric="true">
                        {clause.outcome === "passed" ? (
                          <span aria-hidden="true">✓</span>
                        ) : clause.outcome === "failed" ? (
                          <span aria-hidden="true" className="training-clause-fail">
                            ×
                          </span>
                        ) : (
                          <span>— excluded</span>
                        )}
                        <span className="sr-only">{clause.outcome}</span>
                      </td>
                    </tr>
                    {clause.reason ? (
                      <tr className="training-clause-note">
                        <td colSpan={5}>{clause.reason}</td>
                      </tr>
                    ) : null}
                  </Fragment>
                ))}
              </tbody>
            </table>
            <p className="training-provenance">
              Benchmarked {formatTimestamp(guardrails.benchmarked_at)} against {goldVersionLabel}
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
          </div>
        </details>
      )}

      {!failed && heldOutPending ? (
        <div className="training-caveat-row">
          <span className="training-held-out-flag">Generalization unchecked</span>
          <p>
            No held-out survey data yet — the gate currently measures forgetting and the production operating point
            only. It joins automatically when a post-checkpoint survey is added.
          </p>
        </div>
      ) : null}

      {state.canarySuppressesPromote ? (
        <p className="training-gloss">A canary is already soaking — resolve it before promoting another candidate.</p>
      ) : null}

      <div className="training-actions">
        <span>
          {state.primary === "promote-canary" ? (
            <Button className="training-primary-action" onClick={onPromote}>
              Promote to canary
            </Button>
          ) : failed ? (
            <button type="button" className="training-ghost-action" onClick={onRerunBenchmark}>
              Re-run benchmark
            </button>
          ) : null}
        </span>
        <button type="button" className="training-ghost-action" onClick={onDismiss}>
          Dismiss candidate
        </button>
      </div>
    </div>
  );
}
