import { describe, expect, it } from "vitest";

import {
  appendRunEventCoalescing,
  appendUniqueRunEvent,
  isEphemeralDeltaEvent,
  isEphemeralDeltaEventKind,
  reasoningTextFromRunEvents,
  runEventIdentity,
  runHasToolActivity,
} from "./run-events";

describe("ephemeral delta event invariant", () => {
  it("classifies per-token text deltas as ephemeral", () => {
    expect(isEphemeralDeltaEventKind("message.delta")).toBe(true);
    expect(isEphemeralDeltaEventKind("subagent.message.delta")).toBe(true);
    expect(isEphemeralDeltaEventKind("trace.message.delta")).toBe(true);
  });

  it("keeps structural and rendered events (incl. reasoning) durable", () => {
    // trace.reasoning.delta is rendered by ChatRunSteps, so it must NOT be treated as ephemeral.
    expect(isEphemeralDeltaEventKind("trace.reasoning.delta")).toBe(false);
    expect(isEphemeralDeltaEventKind("tool_call.started")).toBe(false);
    expect(isEphemeralDeltaEventKind("run.token_usage")).toBe(false);
    expect(isEphemeralDeltaEventKind("run.completed")).toBe(false);
    expect(isEphemeralDeltaEventKind("artifact.created")).toBe(false);
    expect(isEphemeralDeltaEventKind(undefined)).toBe(false);
  });

  it("filters accumulated event arrays by kind or type", () => {
    const events = [
      { event_kind: "message.delta" },
      { event_type: "subagent.message.delta" },
      { event_kind: "tool_call.started" },
      { event_kind: "trace.reasoning.delta" },
      { event_kind: "run.completed" },
    ];
    const durable = events.filter((event) => !isEphemeralDeltaEvent(event));
    expect(durable.map((event) => event.event_kind ?? event.event_type)).toEqual([
      "tool_call.started",
      "trace.reasoning.delta",
      "run.completed",
    ]);
  });
});

describe("runHasToolActivity (agentic-run detection)", () => {
  it("is false for a plain text reply (reasoning only)", () => {
    expect(
      runHasToolActivity([
        { event_type: "run.started" },
        { event_type: "trace.reasoning.delta" },
        { event_type: "run.token_usage" },
      ])
    ).toBe(false);
  });

  it("is true once a tool / code execution or subagent appears", () => {
    expect(runHasToolActivity([{ event_type: "trace.reasoning.delta" }, { event_type: "tool_call.started" }])).toBe(true);
    expect(runHasToolActivity([{ event_kind: "tool_call.completed" }])).toBe(true);
    expect(runHasToolActivity([{ event_type: "subagent.message.delta" }])).toBe(true);
  });

  it("is false for an empty event stream", () => {
    expect(runHasToolActivity([])).toBe(false);
  });
});

describe("appendUniqueRunEvent", () => {
  it("dedupes replayed trace events by event id before they can double live reasoning text", () => {
    const first = {
      event_type: "trace.reasoning.delta",
      payload: {
        event_id: "evt-trace-1",
        sequence: 10,
        run_id: "run_1",
        text: "Now I see the issue.",
      },
    };
    const replay = {
      event_type: "trace.reasoning.delta",
      payload: {
        event_id: "evt-trace-1",
        sequence: 10,
        run_id: "run_1",
        text: "Now I see the issue.",
        replayed: true,
      },
    };

    const existing = [first];
    const events = appendUniqueRunEvent(existing, replay);

    expect(events).toBe(existing);
    expect(events).toHaveLength(1);
    expect(runEventIdentity(first)).toBe("event_id:evt-trace-1");
  });

  it("dedupes replayed structural events by run sequence when event id is absent", () => {
    const first: { event_type: string; payload: Record<string, unknown> } = {
      event_type: "trace.reasoning.delta",
      payload: { sequence: 12, run_id: "run_1", text: "Checking the dataframe." },
    };
    const replay: { event_type: string; payload: Record<string, unknown> } = {
      event_type: "trace.reasoning.delta",
      payload: { sequence: "12", run_id: "run_1", text: "Checking the dataframe." },
    };

    expect(appendUniqueRunEvent([first], replay)).toHaveLength(1);
    expect(runEventIdentity(first)).toBe("sequence:run_1:12");
  });

  it("still dedupes a duplicate-sequence replay via the bounded backward scan", () => {
    // Sequence-sorted array (the streaming invariant): the scan starts at the tail, so the
    // duplicate at the tail is found before the break condition can trigger.
    const events = [
      { event_type: "run.started", payload: { sequence: 1, run_id: "run_1" } },
      { event_type: "tool_call.started", payload: { sequence: 2, run_id: "run_1" } },
      { event_type: "tool_call.completed", payload: { sequence: 3, run_id: "run_1" } },
    ];
    const replay = { event_type: "tool_call.completed", payload: { sequence: 3, run_id: "run_1" } };
    expect(appendUniqueRunEvent(events, replay)).toBe(events);
  });

  it("appends a genuinely new higher-sequence event without scanning past smaller sequences", () => {
    const events = [
      { event_type: "run.started", payload: { sequence: 1, run_id: "run_1" } },
      { event_type: "tool_call.started", payload: { sequence: 2, run_id: "run_1" } },
    ];
    const next = { event_type: "tool_call.completed", payload: { sequence: 3, run_id: "run_1" } };
    const appended = appendUniqueRunEvent(events, next);
    expect(appended).toHaveLength(3);
    expect(appended[2]).toBe(next);
  });

  it("dedupes sequence-less events via the full-scan fallback", () => {
    const first = { event_type: "run.log", level: "info", payload: { text: "hello" } };
    const replay = { event_type: "run.log", level: "info", payload: { text: "hello" } };
    const events = [first];
    expect(appendUniqueRunEvent(events, replay)).toBe(events);

    const other = { event_type: "run.log", level: "info", payload: { text: "different" } };
    expect(appendUniqueRunEvent(events, other)).toHaveLength(2);
  });
});

describe("appendRunEventCoalescing", () => {
  it("accumulates reasoning delta text in place (incremental fragments, not snapshots)", () => {
    const started = { event_kind: "tool_call.started", payload: { sequence: 1, run_id: "run_1" } };
    const reasoning1 = {
      event_kind: "trace.reasoning.delta",
      payload: { sequence: 2, run_id: "run_1", text: "Thinking… ", status: "running" },
    };
    const completed = { event_kind: "tool_call.completed", payload: { sequence: 3, run_id: "run_1" } };
    const reasoning2 = {
      event_kind: "trace.reasoning.delta",
      payload: { sequence: 4, run_id: "run_1", text: "more.", status: "completed" },
    };

    const events = [started, reasoning1, completed];
    const coalesced = appendRunEventCoalescing(events, reasoning2);

    expect(coalesced).toHaveLength(3);
    // Same slot (step order preserved), but the reasoning event ACCUMULATES its text and takes the
    // newer event's status/sequence.
    expect(coalesced[0]).toBe(started);
    expect(coalesced[2]).toBe(completed);
    const merged = coalesced[1] as { payload: Record<string, unknown> };
    expect(merged.payload.text).toBe("Thinking… more.");
    expect(merged.payload.status).toBe("completed");
    expect(merged.payload.sequence).toBe(4);
    // Input array + its events are not mutated.
    expect(events[1]).toBe(reasoning1);
    expect((reasoning1 as { payload: Record<string, unknown> }).payload.text).toBe("Thinking… ");
  });

  it("appends the first reasoning delta when none exists yet", () => {
    const started = { event_kind: "tool_call.started", payload: { sequence: 1, run_id: "run_1" } };
    const reasoning = {
      event_kind: "trace.reasoning.delta",
      payload: { sequence: 2, run_id: "run_1", text: "Thinking…" },
    };
    const appended = appendRunEventCoalescing([started], reasoning);
    expect(appended).toHaveLength(2);
    expect(appended[1]).toBe(reasoning);
  });

  it("accumulates reasoning deltas matched by event_type when event_kind is absent", () => {
    const reasoning1 = {
      event_type: "trace.reasoning.delta",
      payload: { sequence: 1, run_id: "run_1", text: "a" },
    };
    const reasoning2 = {
      event_type: "trace.reasoning.delta",
      payload: { sequence: 2, run_id: "run_1", text: "b" },
    };
    const coalesced = appendRunEventCoalescing([reasoning1], reasoning2);
    expect(coalesced).toHaveLength(1);
    expect((coalesced[0] as { payload: Record<string, unknown> }).payload.text).toBe("ab");
  });

  it("delegates non-reasoning events to appendUniqueRunEvent (dedup preserved)", () => {
    const first = { event_kind: "tool_call.started", payload: { sequence: 5, run_id: "run_1" } };
    const events = [first];
    const replay = { event_kind: "tool_call.started", payload: { sequence: 5, run_id: "run_1" } };
    expect(appendRunEventCoalescing(events, replay)).toBe(events);

    const next = { event_kind: "tool_call.completed", payload: { sequence: 6, run_id: "run_1" } };
    expect(appendRunEventCoalescing(events, next)).toHaveLength(2);
  });
});

describe("reasoningTextFromRunEvents", () => {
  it("returns the accumulated reasoning text (trimmed), ignoring non-reasoning events", () => {
    const events = [
      { event_kind: "tool_call.started", payload: { sequence: 1 } },
      {
        event_kind: "trace.reasoning.delta",
        payload: { sequence: 2, text: "  Step one. Step two.  " },
      },
    ];
    expect(reasoningTextFromRunEvents(events)).toBe("Step one. Step two.");
  });

  it("sums any un-coalesced reasoning fragments in order", () => {
    const events = [
      { event_type: "trace.reasoning.delta", payload: { text: "part one " } },
      { event_type: "trace.reasoning.delta", payload: { text: "part two" } },
    ];
    expect(reasoningTextFromRunEvents(events)).toBe("part one part two");
  });

  it("is empty when there is no reasoning", () => {
    expect(
      reasoningTextFromRunEvents([{ event_type: "message.delta", payload: { text: "hi" } }])
    ).toBe("");
  });
});
