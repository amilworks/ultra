// The Data disclosure body (§14.4-D1): soft stat tiles (the token-usage tile
// recipe, auto-fit grid), text-first coverage over the neutral-ink bar, sync
// details, and the disposition-aware taxonomy note as a plain muted sentence.

import type { TrainingModelStatus } from "../../types";
import { formatCount, formatTimestamp } from "./format";

const classLabel = (name: string): string => `${name.replace(/_/g, " ")} labels`;

export function TrainingStatTiles({ status }: { status: TrainingModelStatus }) {
  const reviewed = Number(status.reviewed_images ?? 0);
  const unreviewed = Number(status.unreviewed_images ?? 0);
  const total = reviewed + unreviewed;
  const coverage = total > 0 ? Math.round((reviewed / total) * 100) : 0;
  const classCounts = status.class_counts ?? {};
  const newSinceTrain = Object.values(status.per_class_new_objects ?? {}).reduce(
    (sum, value) => sum + Number(value ?? 0),
    0
  );

  return (
    <div>
      <div className="training-stat-tiles">
        <div className="training-stat-tile">
          <span className="training-stat-tile-value">{formatCount(reviewed)}</span>
          <span className="training-stat-tile-label">reviewed images</span>
        </div>
        <div className="training-stat-tile">
          <span className="training-stat-tile-value">{formatCount(unreviewed)}</span>
          <span className="training-stat-tile-label">unreviewed</span>
        </div>
        {Object.entries(classCounts).map(([name, value]) => (
          <div key={name} className="training-stat-tile">
            <span className="training-stat-tile-value">{formatCount(Number(value ?? 0))}</span>
            <span className="training-stat-tile-label">{classLabel(name)}</span>
          </div>
        ))}
        <div className="training-stat-tile">
          <span className="training-stat-tile-value">{formatCount(newSinceTrain)}</span>
          <span className="training-stat-tile-label">new since last train</span>
        </div>
      </div>

      {total > 0 ? (
        <>
          <p className="training-gloss" style={{ marginTop: "0.9rem" }}>
            {formatCount(reviewed)} of {formatCount(total)} images reviewed · {coverage}%
          </p>
          <div className="training-coverage-bar" role="presentation">
            <div style={{ width: `${Math.max(0, Math.min(100, coverage))}%` }} />
          </div>
        </>
      ) : null}

      <p className="training-gloss" style={{ marginTop: "0.9rem" }}>
        Dataset {status.dataset_name || "—"}
        {status.dataset_id ? ` (${status.dataset_id})` : ""} · last sync {formatTimestamp(status.last_sync_at)} · next
        sync —
      </p>

      {Object.entries(status.unsupported_class_counts ?? {}).map(([name, raw]) => {
        const record = raw as { count?: number; disposition?: string } | number;
        const count = typeof record === "number" ? record : Number(record?.count ?? 0);
        const disposition = typeof record === "object" ? record?.disposition : undefined;
        if (!count) {
          return null;
        }
        return (
          <p key={name} className="training-gloss">
            {formatCount(count)} {name.replace(/_/g, " ")} observations recorded —{" "}
            {disposition === "ignore"
              ? "mapped to ignore regions: masked out of training and excluded from false-positive counting."
              : "this class isn't in the model's taxonomy and won't be trained on."}
          </p>
        );
      })}
    </div>
  );
}
