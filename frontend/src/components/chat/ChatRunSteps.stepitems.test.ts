import { describe, expect, it } from "vitest";

import type { RunEvent } from "@/types";

import { buildStepItems } from "./ChatRunSteps";

const reasoningDelta = (i: number): RunEvent =>
  ({ event_type: "trace.reasoning.delta", payload: { text: `thought ${i}`, status: "running" } }) as RunEvent;

const toolEvent = (type: string, id: string, name: string, status?: string): RunEvent =>
  ({ event_type: type, payload: { tool_name: name, tool_call_id: id, status } }) as RunEvent;

describe("buildStepItems (canonical RunStep grouping)", () => {
  it("coalesces many reasoning deltas into a single Thinking step (O(steps), not O(events))", () => {
    const events = Array.from({ length: 50 }, (_, i) => reasoningDelta(i));
    const steps = buildStepItems(events, []);
    const reasoning = steps.filter((step) => step.id === "reasoning");
    expect(reasoning).toHaveLength(1);
    expect(reasoning[0].label).toBe("Thinking");
  });

  it("coalesces a tool's started+completed into one step and is fully deterministic", () => {
    const events = [
      reasoningDelta(0),
      toolEvent("tool_call.started", "call-1", "code_runner", "running"),
      reasoningDelta(1),
      toolEvent("tool_call.completed", "call-1", "code_runner", "succeeded"),
    ];

    const a = buildStepItems(events, []);
    const b = buildStepItems(events, []);
    // Deterministic: identical input always yields identical steps (required for reconnect/replica
    // consistency of the trace).
    expect(a).toEqual(b);

    // Bounded: 4 events collapse to 2 steps (one reasoning + one tool), not 4.
    expect(a.filter((step) => step.kind === "tool")).toHaveLength(1);
    expect(a.filter((step) => step.id === "reasoning")).toHaveLength(1);
    expect(a).toHaveLength(2);
  });

  it("returns no steps for an empty event stream", () => {
    expect(buildStepItems([], [])).toEqual([]);
  });
});
