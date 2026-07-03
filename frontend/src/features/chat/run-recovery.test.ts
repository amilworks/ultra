import { describe, expect, it } from "vitest";

import {
  isLegacyRunLookup404Content,
  latestRunEventSequence,
  shouldDropHydratedLegacy404Message,
  shouldRecoverRunResultMessage,
} from "./run-recovery";

describe("shouldRecoverRunResultMessage", () => {
  it("recovers stale legacy result-route 404 bubbles when a run id is present", () => {
    expect(
      shouldRecoverRunResultMessage({
        role: "assistant",
        runId: "run_123",
        content: "Error: Request failed with status 404: 404 page not found",
      })
    ).toBe(true);
  });

  it("does not recover normal completed assistant text", () => {
    expect(
      shouldRecoverRunResultMessage({
        role: "assistant",
        runId: "run_123",
        content: "The plot demonstrates quadratic growth.",
      })
    ).toBe(false);
  });

  it("does not open a recovery stream for a live local stream", () => {
    expect(
      shouldRecoverRunResultMessage(
        {
          role: "assistant",
          runId: "run_123",
          content: "",
          liveStream: {},
        },
        { isStreamingMessage: true }
      )
    ).toBe(false);
  });

  it("does not open recovery while the same run is already streaming locally", () => {
    expect(
      shouldRecoverRunResultMessage(
        {
          role: "assistant",
          runId: "run_123",
          content: "",
        },
        { isLocalRunActive: true }
      )
    ).toBe(false);
  });

  it("does not treat terminal run failures as stale lookup errors", () => {
    expect(
      shouldRecoverRunResultMessage({
        role: "assistant",
        runId: "run_123",
        content: "Error: Run failed",
      })
    ).toBe(false);
  });

  it("never re-recovers a turn the user stopped or that we marked failed", () => {
    // These carry the calm Retry/Edit affordance; auto-recovery would clobber it.
    expect(
      shouldRecoverRunResultMessage({
        role: "assistant",
        runId: "run_123",
        content: "",
        status: "stopped",
      })
    ).toBe(false);
    expect(
      shouldRecoverRunResultMessage({
        role: "assistant",
        runId: "run_123",
        content: "",
        status: "failed",
      })
    ).toBe(false);
  });

  it("identifies old legacy 404 content independent of the Error prefix", () => {
    expect(
      isLegacyRunLookup404Content("Error: Request failed with status 404: 404 page not found")
    ).toBe(true);
    expect(isLegacyRunLookup404Content("Request failed with status 404")).toBe(true);
    expect(isLegacyRunLookup404Content("Error: Run failed")).toBe(false);
  });

  it("drops hydrated legacy 404 bubbles only when there is no run to recover", () => {
    expect(
      shouldDropHydratedLegacy404Message({
        role: "assistant",
        content: "Error: Request failed with status 404: 404 page not found",
      })
    ).toBe(true);
    expect(
      shouldDropHydratedLegacy404Message({
        role: "assistant",
        runId: "run_123",
        content: "Error: Request failed with status 404: 404 page not found",
      })
    ).toBe(false);
    expect(
      shouldDropHydratedLegacy404Message({
        role: "user",
        content: "Request failed with status 404: 404 page not found",
      })
    ).toBe(false);
  });

  it("computes the highest hydrated event sequence for stream resume cursors", () => {
    expect(
      latestRunEventSequence([
        { payload: { sequence: 2 } },
        { payload: { sequence: 5 } },
        { payload: { sequence: "4" } },
        { payload: { sequence: 0 } },
      ])
    ).toBe(5);
  });
});
