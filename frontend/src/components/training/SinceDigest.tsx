// The since-your-last-visit digest (§14.4-G): 1-3 muted delta lines from the
// recent-events echo, plus one line per priority-rule demotion (nothing is
// hidden, only demoted). An empty first visit gets neutral orientation copy;
// later zero-event visits render nothing, never an all-time activity claim.

import type { TrainingRecentEvent } from "../../types";
import type { TrainingDemotedNote } from "../../features/training/state";
import { formatDay } from "./format";

export function SinceDigest({
  events,
  demoted,
  firstVisit,
  onOpenHistory,
}: {
  events: TrainingRecentEvent[];
  demoted: TrainingDemotedNote[];
  firstVisit: boolean;
  onOpenHistory: () => void;
}) {
  if (firstVisit && events.length === 0 && demoted.length === 0) {
    return (
      <div className="training-digest">
        <span>This page tracks your detector's retraining.</span>
      </div>
    );
  }
  if (events.length === 0 && demoted.length === 0) {
    return null;
  }
  return (
    <div className="training-digest">
      {events.map((event) => (
        <button
          key={`${event.ts}-${event.summary}`}
          type="button"
          className="training-ghost-action"
          style={{ textAlign: "left", padding: 0 }}
          onClick={onOpenHistory}
        >
          {formatDay(event.ts)}: {event.summary}
        </button>
      ))}
      {demoted.map((note) => (
        <span key={note.state}>{note.summary}</span>
      ))}
    </div>
  );
}
