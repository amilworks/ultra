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
});
