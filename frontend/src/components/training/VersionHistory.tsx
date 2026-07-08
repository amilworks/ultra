// The History disclosure (§14.4-D2): one merged chronological timeline from
// the recent-events echo + retrain requests + version transitions. Plain muted
// text; failures get a destructive-text word, never a pill; benchmark entries
// carry the short gold hash and a report link.

import type {
  PrairieRetrainRecord,
  TrainingModelStatus,
  TrainingModelVersionRecord,
} from "../../types";
import { formatTimestamp } from "./format";

type HistoryRow = {
  ts: string;
  text: string;
  failed: boolean;
  reportUri?: string;
};

export function VersionHistory({
  status,
  versions,
  retrainRequests,
}: {
  status: TrainingModelStatus;
  versions: TrainingModelVersionRecord[];
  retrainRequests: PrairieRetrainRecord[];
}) {
  const rows: HistoryRow[] = [];

  for (const event of status.recent_events ?? []) {
    rows.push({
      ts: event.ts,
      text: `${event.summary}${event.gold_hash_short ? ` (${event.gold_hash_short})` : ""}`,
      failed: /fail/i.test(event.kind) || /fail/i.test(event.summary),
      reportUri: event.report_uri,
    });
  }
  for (const request of retrainRequests) {
    rows.push({
      ts: request.finished_at ?? request.created_at,
      text: `Retrain ${request.status}${request.note ? ` — ${request.note}` : ""}${request.error ? ` — ${request.error}` : ""}`,
      failed: request.status === "failed",
    });
  }
  for (const version of versions) {
    rows.push({
      ts: version.created_at,
      text: `${version.version_id} registered (${version.status})`,
      failed: version.status === "rejected",
    });
  }

  rows.sort((a, b) => new Date(b.ts).getTime() - new Date(a.ts).getTime());

  if (rows.length === 0) {
    return <p className="training-gloss">No training activity yet.</p>;
  }

  return (
    <div className="training-history-list">
      {rows.slice(0, 20).map((row, index) => (
        <span key={`${row.ts}-${index}`} className={row.failed ? "training-history-failed" : undefined}>
          {formatTimestamp(row.ts)} — {row.text}
          {row.reportUri ? (
            <>
              {" · "}
              <a href={row.reportUri} target="_blank" rel="noreferrer">
                View report
              </a>
            </>
          ) : null}
        </span>
      ))}
    </div>
  );
}
