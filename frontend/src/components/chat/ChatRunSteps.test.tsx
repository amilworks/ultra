import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ChatRunSteps } from "./ChatRunSteps";
import type { RunEvent } from "@/types";

describe("ChatRunSteps", () => {
  it("renders live V2 tool-call events as visible autonomous progress", () => {
    const runEvents: RunEvent[] = [
      {
        event_type: "run.started",
        payload: {
          event_kind: "run.started",
          message: "Deep Agents run started.",
          sequence: 1,
        },
      },
      {
        event_type: "tool_call.started",
        payload: {
          event_kind: "tool_call.started",
          message: "execute started",
          status: "started",
          tool_name: "execute",
          tool_call_id: "call-execute-1",
          command: "python train.py",
          sequence: 2,
        },
      },
    ];

    render(
      <ChatRunSteps
        runEvents={runEvents}
        progressEvents={[]}
        isStreaming
        fallbackLabel="BisQue Ultra is processing"
      />
    );

    expect(screen.getByTestId("chat-run-steps")).toBeInTheDocument();
    expect(screen.getByText("Steps: Execute")).toBeInTheDocument();
    expect(screen.getByText("Execute")).toBeInTheDocument();
    expect(screen.getByText("execute started")).toBeInTheDocument();
  });

  it("shows a live thinking step for streamed reasoning deltas", () => {
    const runEvents: RunEvent[] = [
      {
        event_type: "trace.reasoning.delta",
        payload: {
          event_kind: "trace.reasoning.delta",
          text: "Weighing the estimator variance against bias.",
          status: "running",
          sequence: 2,
        },
      },
    ];

    render(
      <ChatRunSteps
        runEvents={runEvents}
        progressEvents={[]}
        isStreaming
        fallbackLabel={null}
      />
    );

    expect(screen.getByText("Steps: Thinking")).toBeInTheDocument();
    expect(
      screen.getByText("Weighing the estimator variance against bias.")
    ).toBeInTheDocument();
  });

  it("collapses sequential reasoning deltas into one completed step", () => {
    const runEvents: RunEvent[] = [
      {
        event_type: "trace.reasoning.delta",
        payload: {
          event_kind: "trace.reasoning.delta",
          text: "First pass over the request.",
          status: "running",
          sequence: 2,
        },
      },
      {
        event_type: "trace.reasoning.delta",
        payload: {
          event_kind: "trace.reasoning.delta",
          text: "Settled on an approach.",
          status: "completed",
          sequence: 3,
        },
      },
      {
        event_type: "tool_call.started",
        payload: {
          event_kind: "tool_call.started",
          message: "execute started",
          status: "started",
          tool_name: "execute",
          tool_call_id: "call-execute-1",
          sequence: 4,
        },
      },
    ];

    render(
      <ChatRunSteps
        runEvents={runEvents}
        progressEvents={[]}
        isStreaming
        fallbackLabel={null}
      />
    );

    expect(screen.getByText("Thinking")).toBeInTheDocument();
    expect(screen.getByText("Settled on an approach.")).toBeInTheDocument();
    expect(
      screen.queryByText("First pass over the request.")
    ).not.toBeInTheDocument();
    expect(screen.getByText("Steps: Execute")).toBeInTheDocument();
  });
});
