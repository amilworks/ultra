import { toRecord } from "@/lib/coerce";

// The delta-coalescing invariant (shared by every run-event consumer path: the live SSE reducer
// and the fallback poll). Ephemeral per-token delta events stream the live text but carry no
// durable trace meaning, so they must NEVER accumulate into a message's `runEvents` array — doing
// so rebuilt the whole transcript and ran an O(n) dedup scan PER TOKEN (~44k/turn → tab lockup).
//
// - `message.delta` is the coordinator's answer text: it drives the rAF-batched token stream
//   (onToken) and is handled before this set is consulted.
// - `subagent.message.delta` / `trace.message.delta` are not rendered token-by-token; they are
//   dropped from front-end state entirely (still persisted server-side for the trace).
// - `trace.reasoning.delta` is deliberately NOT ephemeral here — ChatRunSteps renders it, so it
//   must continue to flow into runEvents.
const EPHEMERAL_DELTA_EVENT_KINDS = new Set<string>([
  "message.delta",
  "subagent.message.delta",
  "trace.message.delta",
]);

const eventKindOf = (event: {
  event_kind?: unknown;
  event_type?: unknown;
}): string => {
  const kind = typeof event.event_kind === "string" ? event.event_kind.trim() : "";
  if (kind) {
    return kind;
  }
  return typeof event.event_type === "string" ? event.event_type.trim() : "";
};

export const isEphemeralDeltaEventKind = (kind: string | undefined | null): boolean =>
  typeof kind === "string" && EPHEMERAL_DELTA_EVENT_KINDS.has(kind);

// Convenience for filtering accumulated run-event arrays (the poll path) — keeps the durable
// structural trace, drops the per-token delta bloat.
export const isEphemeralDeltaEvent = (event: {
  event_kind?: unknown;
  event_type?: unknown;
}): boolean => EPHEMERAL_DELTA_EVENT_KINDS.has(eventKindOf(event));

type RunEventLike = {
  event_type?: unknown;
  event_kind?: unknown;
  level?: unknown;
  sequence?: unknown;
  payload?: unknown;
};

const finiteSequence = (value: unknown): number => {
  const sequence = Math.floor(Number(value));
  return Number.isFinite(sequence) && sequence > 0 ? sequence : 0;
};

export const runEventIdentity = (event: RunEventLike): string => {
  const payload = toRecord(event.payload);
  const eventID = String(payload?.event_id ?? "").trim();
  if (eventID) {
    return `event_id:${eventID}`;
  }
  const sequence = finiteSequence(payload?.sequence ?? event.sequence);
  if (sequence > 0) {
    const runID = String(payload?.run_id ?? "").trim();
    return `sequence:${runID}:${sequence}`;
  }
  return JSON.stringify({
    event_type: String(event.event_type || "").trim().toLowerCase(),
    level: String(event.level || "").trim().toLowerCase(),
    payload: payload ?? event.payload ?? null,
  });
};

export const appendUniqueRunEvent = <T extends RunEventLike>(
  events: T[],
  nextEvent: T
): T[] => {
  const nextIdentity = runEventIdentity(nextEvent);
  const nextSequence = finiteSequence(toRecord(nextEvent.payload)?.sequence ?? nextEvent.sequence);
  if (nextSequence > 0) {
    // Bounded backward scan. The array is sequence-sorted: SSE delivery is strictly increasing per
    // run, the fallback poll replaces with a server-ordered snapshot, and resume starts after
    // latestRunEventSequence. So once we pass an event with a strictly SMALLER finite sequence, no
    // earlier event can share nextEvent's identity — a duplicate would carry the same sequence.
    for (let i = events.length - 1; i >= 0; i--) {
      const event = events[i];
      if (runEventIdentity(event) === nextIdentity) {
        return events;
      }
      const sequence = finiteSequence(toRecord(event.payload)?.sequence ?? event.sequence);
      if (sequence > 0 && sequence < nextSequence) {
        break;
      }
    }
    return [...events, nextEvent];
  }
  if (events.some((event) => runEventIdentity(event) === nextIdentity)) {
    return events;
  }
  return [...events, nextEvent];
};

// Reasoning deltas are cumulative snapshots of the live thinking text: every consumer renders only
// the LATEST one (ChatRunSteps upserts a single 'reasoning' step), so keeping the whole history in
// message.runEvents grows the array unbounded (~2.5 events/s while thinking) and makes each append
// quadratic. Coalesce: replace the previous reasoning delta IN PLACE (positional replace preserves
// step order in buildStepItems); with coalescing at most one such event exists, so the backward
// scan is O(1) amortized. Everything else keeps the dedup-append semantics.
export const appendRunEventCoalescing = <T extends RunEventLike>(
  events: T[],
  nextEvent: T
): T[] => {
  if (eventKindOf(nextEvent) === "trace.reasoning.delta") {
    for (let i = events.length - 1; i >= 0; i--) {
      if (eventKindOf(events[i]) === "trace.reasoning.delta") {
        const next = events.slice();
        next[i] = nextEvent;
        return next;
      }
    }
    return [...events, nextEvent];
  }
  return appendUniqueRunEvent(events, nextEvent);
};

// Whether a run is a multi-step agentic run (it ran a tool / executed code) vs. a plain text reply.
// Tool-call events only accumulate, so this is MONOTONIC — once true it stays true, with no flicker
// back. Used to decide the calm surface: an agentic run shows the step timeline as the primary
// surface with the live reasoning folded into an opt-in disclosure; a short reply streams inline.
export const runHasToolActivity = (
  events: Array<{ event_type?: unknown; event_kind?: unknown }>
): boolean =>
  events.some((event) => {
    const type = typeof event.event_type === "string" ? event.event_type : "";
    const kind = typeof event.event_kind === "string" ? event.event_kind : "";
    return (
      type.startsWith("tool_call") ||
      kind.startsWith("tool_call") ||
      type.startsWith("subagent.")
    );
  });
