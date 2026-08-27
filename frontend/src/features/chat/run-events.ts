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
  ts?: unknown;
  payload?: unknown;
  redacted?: unknown;
};

const isRedactedFlag = (value: unknown): boolean =>
  value === true || (typeof value === "string" && value.trim().toLowerCase() === "true");

/**
 * A runtime redaction marker is a privacy boundary for the whole assistant turn.
 * Treat a top-level flag defensively too: older event envelopes did not always
 * normalize metadata into `payload` before reaching the client reducer.
 */
export const isRedactedRunEvent = (event: RunEventLike): boolean =>
  isRedactedFlag(toRecord(event.payload)?.redacted) || isRedactedFlag(event.redacted);

// Redacted tool events still need a very small amount of structural metadata to
// render calm activity labels and reconstruct the browser-authorized Notes
// receipt UI. Everything capable of carrying user/model prose is intentionally
// absent (message, text, input, output, previews, command, query, title, body…).
const REDACTED_SAFE_PAYLOAD_KEYS = new Set([
  "redacted",
  "ok",
  "error",
  "status",
  "proposal_status",
  "sequence",
  "event_kind",
  "event_type",
  "event_id",
  "run_id",
  "thread_id",
  "node_name",
  "task_id",
  "checkpoint_id",
  "scope_id",
  "agent_role",
  "tool_name",
  "tool",
  "tool_call_id",
  "call_id",
  // Content-free Notes provenance locked by the browser/runtime contract.
  "note_id",
  "revision",
  "returned_bytes",
  "has_more",
  "result_count",
  "proposal_id",
  "expected_revision",
  "expires_at",
]);

const SAFE_CODE_PAYLOAD_KEYS = new Set([
  "error",
  "status",
  "proposal_status",
  "event_kind",
  "event_type",
  "node_name",
  "agent_role",
  "tool_name",
  "tool",
]);
const SAFE_IDENTIFIER_PAYLOAD_KEYS = new Set([
  "event_id",
  "run_id",
  "thread_id",
  "task_id",
  "checkpoint_id",
  "scope_id",
  "tool_call_id",
  "call_id",
  "note_id",
  "proposal_id",
]);
const SAFE_BOOLEAN_PAYLOAD_KEYS = new Set(["redacted", "ok", "has_more"]);
const SAFE_INTEGER_PAYLOAD_KEYS = new Set([
  "sequence",
  "revision",
  "returned_bytes",
  "result_count",
  "expected_revision",
]);

const safeRedactedPayloadValue = (key: string, value: unknown): unknown => {
  if (SAFE_BOOLEAN_PAYLOAD_KEYS.has(key)) {
    return typeof value === "boolean" ? value : undefined;
  }
  if (SAFE_INTEGER_PAYLOAD_KEYS.has(key)) {
    return Number.isSafeInteger(value) && Number(value) >= 0 ? value : undefined;
  }
  if (SAFE_CODE_PAYLOAD_KEYS.has(key)) {
    const normalized = typeof value === "string" ? value.trim().toLowerCase() : "";
    return /^[a-z][a-z0-9_.-]{0,127}$/.test(normalized) ? normalized : undefined;
  }
  if (SAFE_IDENTIFIER_PAYLOAD_KEYS.has(key)) {
    const normalized = typeof value === "string" ? value.trim() : "";
    return /^[A-Za-z0-9][A-Za-z0-9._:-]{0,511}$/.test(normalized)
      ? normalized
      : undefined;
  }
  if (key === "expires_at") {
    const normalized = typeof value === "string" ? value.trim() : "";
    return normalized.length <= 80 && /^[0-9T:.+Z-]+$/.test(normalized)
      ? normalized
      : undefined;
  }
  return undefined;
};

const safeRedactedEnvelopeCode = (value: unknown): string | undefined => {
  const normalized = typeof value === "string" ? value.trim().toLowerCase() : "";
  return /^[a-z][a-z0-9_.-]{0,127}$/.test(normalized) ? normalized : undefined;
};

const safeRedactedTimestamp = (value: unknown): string | undefined => {
  const normalized = typeof value === "string" ? value.trim() : "";
  return normalized.length <= 80 && /^[0-9T:.+Z-]+$/.test(normalized)
    ? normalized
    : undefined;
};

const sanitizeRedactedRunEvent = <T extends RunEventLike>(
  event: T,
  forceRedacted = false
): T => {
  if (!forceRedacted && !isRedactedRunEvent(event)) {
    return event;
  }
  const payload = toRecord(event.payload) ?? {};
  const safePayload: Record<string, unknown> = { redacted: true };
  for (const [key, value] of Object.entries(payload)) {
    if (REDACTED_SAFE_PAYLOAD_KEYS.has(key)) {
      const safeValue = safeRedactedPayloadValue(key, value);
      if (safeValue !== undefined) {
        safePayload[key] = safeValue;
      }
    }
  }
  // The privacy decision is authoritative even if a malformed event tried to
  // overwrite it with `redacted: false` while carrying a top-level marker.
  safePayload.redacted = true;

  // Rebuild the envelope instead of spreading `event`: legacy/hydrated event
  // records sometimes carried prose-bearing fields such as `message`, `text`,
  // or `query` at the top level rather than inside payload. A redaction marker
  // is authoritative across both locations, so only the structural fields the
  // timeline/deduper need may survive.
  const safeEvent: RunEventLike = { payload: safePayload };
  const eventType = safeRedactedEnvelopeCode(event.event_type);
  const eventKind = safeRedactedEnvelopeCode(event.event_kind);
  const level = safeRedactedEnvelopeCode(event.level);
  const sequence = safeRedactedPayloadValue("sequence", event.sequence);
  const ts = safeRedactedTimestamp(event.ts);
  if (eventType !== undefined) {
    safeEvent.event_type = eventType;
  }
  if (eventKind !== undefined) {
    safeEvent.event_kind = eventKind;
  }
  if (level !== undefined) {
    safeEvent.level = level;
  }
  if (sequence !== undefined) {
    safeEvent.sequence = sequence;
  }
  if (ts !== undefined) {
    safeEvent.ts = ts;
  }
  if (isRedactedFlag(event.redacted)) {
    safeEvent.redacted = true;
  }
  return safeEvent as T;
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

const REASONING_DELTA_KIND = "trace.reasoning.delta";

export const isReasoningRunEvent = (event: RunEventLike): boolean =>
  eventKindOf(event) === REASONING_DELTA_KIND;

/** True once the runtime has declared any trace payload in this turn private. */
export const hasRedactedRunEvent = (events: readonly RunEventLike[]): boolean =>
  events.some(isRedactedRunEvent);

// Once a turn crosses the private-content boundary, only answer/usage and
// user-authored steering lifecycle events remain verbatim. Every internal
// trace/tool/subagent/phase event is force-scrubbed even if a malformed or
// out-of-order legacy producer forgot its own marker.
const isTurnPrivateInternalEvent = (event: RunEventLike): boolean => {
  const kind = eventKindOf(event);
  return !(
    kind === "run.completed" ||
    kind === "run.token_usage" ||
    kind === "message.delta" ||
    kind.startsWith("steer.")
  );
};

/**
 * Make a run-event collection safe for browser state.
 *
 * A marker can arrive after ordinary reasoning fragments already streamed. At
 * that moment those earlier fragments must disappear too, rather than remain in
 * `message.runEvents` until the next reload. We retain one content-free
 * reasoning marker (at the original step position) so the UI can still say
 * “Thinking”, and scrub every individually-redacted tool/phase event through a
 * strict metadata allowlist.
 */
export const sanitizeRunEventsForClient = <T extends RunEventLike>(events: T[]): T[] => {
  if (!hasRedactedRunEvent(events)) {
    return events;
  }

  const firstReasoningIndex = events.findIndex(isReasoningRunEvent);
  let newestReasoning: T | null = null;
  let newestReasoningSequence = -1;
  for (const event of events) {
    if (!isReasoningRunEvent(event)) {
      continue;
    }
    const sequence = finiteSequence(toRecord(event.payload)?.sequence ?? event.sequence);
    if (newestReasoning === null || sequence === 0 || sequence >= newestReasoningSequence) {
      newestReasoning = event;
      newestReasoningSequence = sequence;
    }
  }

  const safeReasoningMarker = newestReasoning
    ? sanitizeRedactedRunEvent(newestReasoning, true)
    : null;
  const safe: T[] = [];
  events.forEach((event, index) => {
    if (isReasoningRunEvent(event)) {
      if (index === firstReasoningIndex && safeReasoningMarker) {
        safe.push(safeReasoningMarker);
      }
      return;
    }
    safe.push(sanitizeRedactedRunEvent(event, isTurnPrivateInternalEvent(event)));
  });
  return safe;
};

// The reasoning text carried by a `trace.reasoning.delta` payload.
const reasoningDeltaText = (event: RunEventLike): string => {
  const text = toRecord(event.payload)?.text;
  return typeof text === "string" ? text : "";
};

const reasoningRunID = (event: RunEventLike): string =>
  String(toRecord(event.payload)?.run_id ?? "").trim();

const reasoningStatus = (event: RunEventLike): string =>
  String(toRecord(event.payload)?.status ?? "").trim().toLowerCase();

// Merge a new reasoning delta into the coalesced one by ACCUMULATING its text. The worker flushes
// reasoning as INCREMENTAL fragments (reasoning_stream.py joins the buffered parts then clears them),
// so each delta carries only the text since the last flush — NOT a cumulative snapshot. We therefore
// concatenate rather than replace, taking the newer event's status/sequence (the closing
// status="completed" flush usually carries no text, so this preserves the full chain-of-thought
// while flipping the step to done).
const mergeReasoningDelta = <T extends RunEventLike>(prev: T, next: T): T =>
  (() => {
    const previousText = reasoningDeltaText(prev);
    const nextText = reasoningDeltaText(next);
    // A completed flush is the boundary between model calls. Without a
    // separator, the last sentence of one tool/reasoning round is glued to the
    // first sentence of the next and the trace reads as scrambled prose.
    const roundSeparator =
      previousText && nextText && reasoningStatus(prev) === "completed" ? "\n\n" : "";
    return {
      ...next,
      payload: {
        ...(toRecord(prev.payload) ?? {}),
        ...(toRecord(next.payload) ?? {}),
        text: previousText + roundSeparator + nextText,
      },
    } as T;
  })();

// Coalesce reasoning deltas into a SINGLE run event (bounded array — a long think is ~2.5 events/s),
// accumulating their text IN PLACE so the full reasoning trace is preserved for the "Thinking"
// expansion instead of only the latest ~160-char fragment. Positional replace keeps step order in
// buildStepItems; with coalescing at most one such event exists, so the backward scan is O(1)
// amortized. Everything else keeps the dedup-append semantics.
export const appendRunEventCoalescing = <T extends RunEventLike>(
  events: T[],
  nextEvent: T
): T[] => {
  const safeEvents = sanitizeRunEventsForClient(events);
  const safeNextEvent = sanitizeRedactedRunEvent(nextEvent);
  const privacyActive = hasRedactedRunEvent(safeEvents) || isRedactedRunEvent(safeNextEvent);

  // Once privacy is active, no later/replayed fragment may re-introduce text.
  // Redacted events are normally unique by id/sequence; replacing an identity
  // defensively handles a replay whose privacy marker was added by the server.
  if (privacyActive) {
    const nextIdentity = runEventIdentity(safeNextEvent);
    const duplicateIndex = safeEvents.findIndex(
      (event) => runEventIdentity(event) === nextIdentity
    );
    const combined =
      duplicateIndex >= 0 && isRedactedRunEvent(safeNextEvent)
        ? safeEvents.map((event, index) => (index === duplicateIndex ? safeNextEvent : event))
        : appendUniqueRunEvent(safeEvents, safeNextEvent);
    return sanitizeRunEventsForClient(combined);
  }

  if (eventKindOf(safeNextEvent) === REASONING_DELTA_KIND) {
    const nextRunID = reasoningRunID(safeNextEvent);
    const nextSequence = finiteSequence(
      toRecord(safeNextEvent.payload)?.sequence ?? safeNextEvent.sequence
    );
    const nextIdentity = runEventIdentity(safeNextEvent);
    for (let i = safeEvents.length - 1; i >= 0; i--) {
      const previous = safeEvents[i];
      if (eventKindOf(previous) === REASONING_DELTA_KIND) {
        const previousRunID = reasoningRunID(previous);
        if (previousRunID && nextRunID && previousRunID !== nextRunID) {
          continue;
        }
        const previousSequence = finiteSequence(
          toRecord(previous.payload)?.sequence ?? previous.sequence
        );
        // Coalescing replaces the stored identity/sequence with the newest
        // fragment. SSE reconnect or poll overlap can replay any older
        // fragment; strict per-run ordering means a <= sequence here is a
        // replay, never a new delta. Do this check BEFORE concatenation.
        if (
          (nextSequence > 0 && previousSequence >= nextSequence) ||
          runEventIdentity(previous) === nextIdentity ||
          (nextSequence === 0 &&
            previousSequence === 0 &&
            reasoningDeltaText(safeNextEvent).length > 0 &&
            reasoningDeltaText(previous).endsWith(reasoningDeltaText(safeNextEvent)))
        ) {
          return safeEvents;
        }
        const next = safeEvents.slice();
        next[i] = mergeReasoningDelta(previous, safeNextEvent);
        return next;
      }
    }
    return appendUniqueRunEvent(safeEvents, safeNextEvent);
  }
  return appendUniqueRunEvent(safeEvents, safeNextEvent);
};

// The full accumulated reasoning text across a message's run events — the durable trace of the
// coordinator's thinking, surfaced under the "Thinking" expansion. After coalescing there is at most
// one reasoning delta, but this sums defensively so the poll path / any un-coalesced history still
// reconstruct the whole trace.
export const reasoningTextFromRunEvents = (
  events: Array<{ event_type?: unknown; event_kind?: unknown; payload?: unknown }>
): string => {
  if (hasRedactedRunEvent(events)) {
    return "";
  }
  let text = "";
  let previousStatus = "";
  for (const event of events) {
    if (eventKindOf(event) === REASONING_DELTA_KIND) {
      const fragment = reasoningDeltaText(event);
      if (text && fragment && previousStatus === "completed") {
        text += "\n\n";
      }
      text += fragment;
      previousStatus = reasoningStatus(event);
    }
  }
  return text.trim();
};

/**
 * Sticky live reasoning unless a privacy marker has appeared. The explicit
 * `undefined` on redaction is what clears an already-populated
 * `message.reasoning` field in React state.
 */
export const reasoningTextAfterRunEvents = (
  events: Array<{ event_type?: unknown; event_kind?: unknown; payload?: unknown }>,
  previous?: string
): string | undefined => {
  if (hasRedactedRunEvent(events)) {
    return undefined;
  }
  return reasoningTextFromRunEvents(events) || previous || undefined;
};

/**
 * The exact reasoning fields written into a conversation snapshot. Normal
 * deltas remain omitted (their accumulated text is stored once); a safe generic
 * redaction marker is retained so hydration cannot accidentally revive a stale
 * reasoning string.
 */
export const reasoningFieldsForPersistence = <T extends RunEventLike>(
  events: T[],
  accumulatedReasoning?: string
): { runEvents: T[]; reasoning?: string } => {
  const safeEvents = sanitizeRunEventsForClient(events);
  const redacted = hasRedactedRunEvent(safeEvents);
  return {
    runEvents: safeEvents.filter(
      (event) => !isReasoningRunEvent(event) || isRedactedRunEvent(event)
    ),
    reasoning: redacted
      ? undefined
      : (accumulatedReasoning ?? reasoningTextFromRunEvents(safeEvents)) || undefined,
  };
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
