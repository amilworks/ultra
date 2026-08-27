import { describe, expect, it } from "vitest";

import {
  appendRunEventCoalescing,
  appendUniqueRunEvent,
  hasRedactedRunEvent,
  isEphemeralDeltaEvent,
  isEphemeralDeltaEventKind,
  reasoningFieldsForPersistence,
  reasoningTextAfterRunEvents,
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

  it("dedupes an older reasoning fragment replayed after later fragments were coalesced", () => {
    const first = {
      event_kind: "trace.reasoning.delta",
      payload: { event_id: "evt-r1", sequence: 2, run_id: "run_1", text: "first " },
    };
    const second = {
      event_kind: "trace.reasoning.delta",
      payload: { event_id: "evt-r2", sequence: 4, run_id: "run_1", text: "second" },
    };
    const coalesced = appendRunEventCoalescing(
      appendRunEventCoalescing([], first),
      second
    );
    const replay = { ...first, payload: { ...first.payload, replayed: true } };

    const afterReplay = appendRunEventCoalescing(coalesced, replay);

    expect(afterReplay).toBe(coalesced);
    expect((afterReplay[0] as { payload: Record<string, unknown> }).payload.text).toBe(
      "first second"
    );
  });

  it("separates completed model reasoning rounds instead of gluing their prose", () => {
    const firstRound = {
      event_kind: "trace.reasoning.delta",
      payload: { sequence: 2, run_id: "run_1", text: "Check the inputs.", status: "completed" },
    };
    const secondRound = {
      event_kind: "trace.reasoning.delta",
      payload: { sequence: 5, run_id: "run_1", text: "Now use the result.", status: "running" },
    };

    const coalesced = appendRunEventCoalescing([firstRound], secondRound);

    expect((coalesced[0] as { payload: Record<string, unknown> }).payload.text).toBe(
      "Check the inputs.\n\nNow use the result."
    );
  });

  it("does not merge reasoning events from different runs", () => {
    const first = {
      event_kind: "trace.reasoning.delta",
      payload: { sequence: 2, run_id: "run_1", text: "first" },
    };
    const otherRun = {
      event_kind: "trace.reasoning.delta",
      payload: { sequence: 2, run_id: "run_2", text: "second" },
    };

    expect(appendRunEventCoalescing([first], otherRun)).toHaveLength(2);
  });

  it("delegates non-reasoning events to appendUniqueRunEvent (dedup preserved)", () => {
    const first = { event_kind: "tool_call.started", payload: { sequence: 5, run_id: "run_1" } };
    const events = [first];
    const replay = { event_kind: "tool_call.started", payload: { sequence: 5, run_id: "run_1" } };
    expect(appendRunEventCoalescing(events, replay)).toBe(events);

    const next = { event_kind: "tool_call.completed", payload: { sequence: 6, run_id: "run_1" } };
    expect(appendRunEventCoalescing(events, next)).toHaveLength(2);
  });

  it("treats a redaction marker as a turn-wide privacy boundary", () => {
    const legacyUnmarkedTool = {
      event_type: "tool_call.started",
      payload: {
        sequence: 0,
        run_id: "run_private",
        tool_name: "execute",
        tool_call_id: "call_private",
        status: "running",
        message: "legacy private tool detail",
      },
    };
    const prior = {
      event_type: "trace.reasoning.delta",
      payload: { sequence: 1, run_id: "run_private", text: "already streamed private text" },
    };
    const marker = {
      event_type: "trace.reasoning.delta",
      payload: {
        sequence: 2,
        run_id: "run_private",
        status: "running",
        redacted: true,
        text: "RAW_REDACTED_SENTINEL",
        message: "RAW_REDACTED_SENTINEL",
      },
    };

    const safe = appendRunEventCoalescing([legacyUnmarkedTool, prior], marker);

    expect(hasRedactedRunEvent(safe)).toBe(true);
    expect(reasoningTextAfterRunEvents(safe, "already captured private text")).toBeUndefined();
    expect(JSON.stringify(safe)).not.toContain("already streamed private text");
    expect(JSON.stringify(safe)).not.toContain("legacy private tool detail");
    expect(JSON.stringify(safe)).not.toContain("RAW_REDACTED_SENTINEL");
    expect(safe).toEqual([
      {
        event_type: "tool_call.started",
        payload: {
          redacted: true,
          sequence: 0,
          run_id: "run_private",
          tool_name: "execute",
          tool_call_id: "call_private",
          status: "running",
        },
      },
      {
        event_type: "trace.reasoning.delta",
        payload: {
          redacted: true,
          sequence: 2,
          run_id: "run_private",
          status: "running",
        },
      },
    ]);
  });

  it("never writes redacted reasoning content into a conversation snapshot", () => {
    const sentinel = "RAW_REASONING_MUST_NOT_REACH_THE_RECORD";
    const rawEvents = [
      {
        event_type: "trace.reasoning.delta",
        level: "debug",
        ts: "2026-08-25T12:34:56Z",
        message: sentinel,
        text: sentinel,
        query: sentinel,
        payload: {
          redacted: true,
          text: sentinel,
          sequence: 7,
          run_id: "run_notes",
          status: "completed",
        },
      },
      {
        event_type: "tool_call.completed",
        message: sentinel,
        output_preview: sentinel,
        payload: {
          redacted: true,
          tool_name: "read_note",
          tool_call_id: "call_read_note",
          note_id: "note_1",
          revision: 4,
          ok: false,
          error: "notes_tool_failed",
          proposal_status: "pending",
          output_preview: sentinel,
          sequence: 8,
          run_id: "run_notes",
          status: "completed",
        },
      },
    ];

    // This mirrors the reasoning fields embedded by conversationToRecord.
    const snapshotRecord = {
      state: {
        messages: [
          reasoningFieldsForPersistence(rawEvents, "private text captured before marker"),
        ],
      },
    };
    const serialized = JSON.stringify(snapshotRecord);

    expect(serialized).not.toContain(sentinel);
    expect(serialized).not.toContain("private text captured before marker");
    expect(snapshotRecord.state.messages[0].reasoning).toBeUndefined();
    expect(snapshotRecord.state.messages[0].runEvents).toEqual([
      {
        event_type: "trace.reasoning.delta",
        level: "debug",
        ts: "2026-08-25T12:34:56Z",
        payload: {
          redacted: true,
          sequence: 7,
          run_id: "run_notes",
          status: "completed",
        },
      },
      {
        event_type: "tool_call.completed",
        payload: {
          redacted: true,
          tool_name: "read_note",
          tool_call_id: "call_read_note",
          note_id: "note_1",
          revision: 4,
          ok: false,
          error: "notes_tool_failed",
          proposal_status: "pending",
          sequence: 8,
          run_id: "run_notes",
          status: "completed",
        },
      },
    ]);
  });

  it("leaves ordinary unredacted event envelopes unchanged", () => {
    const ordinary = {
      event_type: "tool_call.completed",
      message: "ordinary visible detail",
      query: "ordinary query",
      payload: { status: "completed", output_preview: "ordinary result" },
    };

    const events = appendRunEventCoalescing([], ordinary);

    expect(events[0]).toBe(ordinary);
    expect(events[0]).toEqual(ordinary);
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

  it("separates un-coalesced model rounds at completed boundaries", () => {
    const events = [
      {
        event_type: "trace.reasoning.delta",
        payload: { text: "round one", status: "running" },
      },
      { event_type: "trace.reasoning.delta", payload: { text: "", status: "completed" } },
      {
        event_type: "trace.reasoning.delta",
        payload: { text: "round two", status: "completed" },
      },
    ];

    expect(reasoningTextFromRunEvents(events)).toBe("round one\n\nround two");
  });

  it("is empty when there is no reasoning", () => {
    expect(
      reasoningTextFromRunEvents([{ event_type: "message.delta", payload: { text: "hi" } }])
    ).toBe("");
  });
});
