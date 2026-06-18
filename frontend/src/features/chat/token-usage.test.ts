import { describe, expect, it } from "vitest";

import { extractRunTokenUsage } from "./token-usage";

describe("extractRunTokenUsage", () => {
  it("sums live run.token_usage events and dedupes repeated usage event ids", () => {
    expect(
      extractRunTokenUsage({
        responseMetadata: {
          usage: {
            input_tokens: 999,
            output_tokens: 1,
            total_tokens: 1000,
            model: "fallback-model",
          },
        },
        runEvents: [
          {
            event_type: "run.token_usage",
            payload: {
              usage_event_id: "usage-1",
              input_tokens: 100,
              output_tokens: 20,
              total_tokens: 120,
              model: "deepseek_v4",
              sequence: 4,
            },
          },
          {
            event_type: "run.token_usage",
            payload: {
              usage_event_id: "usage-1",
              input_tokens: 100,
              output_tokens: 20,
              total_tokens: 120,
              model: "deepseek_v4",
              sequence: 4,
            },
          },
          {
            event_type: "run.token_usage",
            payload: {
              usage_event_id: "usage-2",
              input_tokens: 40,
              output_tokens: 8,
              total_tokens: 48,
              model: "deepseek_v4",
              sequence: 7,
            },
          },
        ],
      })
    ).toEqual({
      input_tokens: 140,
      output_tokens: 28,
      total_tokens: 168,
      model: "deepseek_v4",
    });
  });

  it("falls back to final metadata when no live usage events are available", () => {
    expect(
      extractRunTokenUsage({
        responseMetadata: {
          usage: {
            input_tokens: 10,
            output_tokens: 2,
            total_tokens: 12,
            model: "deepseek_v4",
          },
        },
        runEvents: [],
      })
    ).toEqual({
      input_tokens: 10,
      output_tokens: 2,
      total_tokens: 12,
      model: "deepseek_v4",
    });
  });

  it("dedupes live usage events by normalized stream sequence when usage_event_id is absent", () => {
    expect(
      extractRunTokenUsage({
        runEvents: [
          {
            event_type: "run.token_usage",
            payload: {
              sequence: 12,
              run_id: "run-1",
              input_tokens: 15,
              output_tokens: 5,
              total_tokens: 20,
            },
          },
          {
            event_type: "run.token_usage",
            payload: {
              sequence: 12,
              run_id: "run-1",
              input_tokens: 15,
              output_tokens: 5,
              total_tokens: 20,
            },
          },
        ],
      })
    ).toEqual({
      input_tokens: 15,
      output_tokens: 5,
      total_tokens: 20,
    });
  });

  it("falls back to run.completed payload usage when response metadata is absent", () => {
    expect(
      extractRunTokenUsage({
        runEvents: [
          {
            event_type: "run.completed",
            payload: {
              usage: {
                input_tokens: 10,
                output_tokens: 2,
                total_tokens: 12,
              },
            },
          },
        ],
      })
    ).toEqual({
      input_tokens: 10,
      output_tokens: 2,
      total_tokens: 12,
    });
  });
});
