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
export const EPHEMERAL_DELTA_EVENT_KINDS = new Set<string>([
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
      type === "tool_event" ||
      type.startsWith("subagent.")
    );
  });
