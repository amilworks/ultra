import type { AssistantContract, ProgressEvent } from "../types";
import { Reasoning, ReasoningContent, ReasoningTrigger, Tool, type ToolPart } from "./prompt-kit";

type DiagnosticsPanelProps = {
  contract?: AssistantContract | null;
  progressEvents?: ProgressEvent[];
  durationSeconds?: number | null;
};

export function DiagnosticsPanel({
  contract,
  progressEvents,
  durationSeconds,
}: DiagnosticsPanelProps) {
  if (!contract && (!progressEvents || progressEvents.length === 0)) {
    return null;
  }

  return (
    <section className="panel">
      <div className="panel-header">
        <h3>Diagnostics</h3>
        {typeof durationSeconds === "number" ? (
          <span className="pill">{durationSeconds.toFixed(2)}s</span>
        ) : null}
      </div>
      <Reasoning className="pk-diagnostics-reasoning">
        <ReasoningTrigger>How this answer was produced</ReasoningTrigger>
        <ReasoningContent>
          Tool traces, measurements, confidence, and warning blocks are collected
          from the assistant contract and run events.
        </ReasoningContent>
      </Reasoning>
      {contract ? (
        <div className="stack-sm">
          {contract.measurements?.length ? (
            <details open>
              <summary>Measurements ({contract.measurements.length})</summary>
              <div className="pk-table-wrap">
                <table className="pk-table">
                  <thead className="pk-table-head">
                    <tr className="pk-table-row">
                      <th className="pk-table-head-cell text-left">Name</th>
                      <th className="pk-table-head-cell text-left">Value</th>
                      <th className="pk-table-head-cell text-left">Unit</th>
                    </tr>
                  </thead>
                  <tbody className="pk-table-body">
                    {contract.measurements.map((measurement, index) => (
                      <tr className="pk-table-row" key={`${measurement.name}-${index}`}>
                        <td className="pk-table-cell">{measurement.name}</td>
                        <td className="pk-table-cell">{String(measurement.value)}</td>
                        <td className="pk-table-cell">{measurement.unit ?? ""}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          ) : null}
          {contract.statistical_analysis?.length ? (
            <details>
              <summary>
                Statistical analysis ({contract.statistical_analysis.length})
              </summary>
              <pre className="json-block">
                {JSON.stringify(contract.statistical_analysis, null, 2)}
              </pre>
            </details>
          ) : null}
          {contract.confidence ? (
            <details>
              <summary>Confidence ({contract.confidence.level})</summary>
              <pre className="json-block">
                {JSON.stringify(contract.confidence, null, 2)}
              </pre>
            </details>
          ) : null}
          {contract.qc_warnings?.length ? (
            <details>
              <summary>QC warnings ({contract.qc_warnings.length})</summary>
              <ul className="list">
                {contract.qc_warnings.map((warning, index) => (
                  <li key={`warning-${index}`}>{warning}</li>
                ))}
              </ul>
            </details>
          ) : null}
          {contract.limitations?.length ? (
            <details>
              <summary>Limitations ({contract.limitations.length})</summary>
              <ul className="list">
                {contract.limitations.map((limitation, index) => (
                  <li key={`limitation-${index}`}>{limitation}</li>
                ))}
              </ul>
            </details>
          ) : null}
          {contract.next_steps?.length ? (
            <details open>
              <summary>Next steps ({contract.next_steps.length})</summary>
              <pre className="json-block">
                {JSON.stringify(contract.next_steps, null, 2)}
              </pre>
            </details>
          ) : null}
        </div>
      ) : null}
      {progressEvents?.length ? (
        <details>
          <summary>Tool progress ({progressEvents.length})</summary>
          <div className="stack-sm">
            {progressEvents.map((event, index) => {
              const toolPart: ToolPart = {
                type: String(event.tool ?? event.event ?? `event-${index}`),
                state: String(event.level ?? "info"),
                input: {
                  event: event.event,
                  ts: event.ts ?? null,
                },
                output: event as Record<string, unknown>,
              };
              return (
                <Tool
                  key={`${toolPart.type}-${event.ts ?? index}`}
                  toolPart={toolPart}
                  defaultOpen={false}
                />
              );
            })}
          </div>
        </details>
      ) : null}
    </section>
  );
}
