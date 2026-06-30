import { describe, expect, it } from "vitest";

import { isEphemeralDeltaEvent, isEphemeralDeltaEventKind, runHasToolActivity } from "./run-events";

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
    expect(runHasToolActivity([{ event_type: "tool_event" }])).toBe(true);
    expect(runHasToolActivity([{ event_type: "subagent.message.delta" }])).toBe(true);
  });

  it("is false for an empty event stream", () => {
    expect(runHasToolActivity([])).toBe(false);
  });
});
