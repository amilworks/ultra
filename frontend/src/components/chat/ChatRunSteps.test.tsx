import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ChatRunSteps } from "./ChatRunSteps";
import type { RunEvent } from "@/types";

const reasoning = (text: string, status: string, sequence: number): RunEvent => ({
  event_type: "trace.reasoning.delta",
  payload: { event_kind: "trace.reasoning.delta", text, status, sequence },
});

const toolCall = (tool: string, sequence: number): RunEvent => ({
  event_type: "tool_call.started",
  payload: {
    event_kind: "tool_call.started",
    message: `${tool} started`,
    status: "started",
    tool_name: tool,
    tool_call_id: `call-${tool}-${sequence}`,
    sequence,
  },
});

describe("ChatRunSteps", () => {
  it("shows a calm humanized status line, with no 'Steps:' prefix or raw tool ids", () => {
    render(
      <ChatRunSteps
        runEvents={[reasoning("done", "completed", 1), toolCall("execute", 2)]}
        progressEvents={[]}
        isStreaming
        statusText="Working through your request"
      />
    );

    expect(screen.getByTestId("chat-run-steps")).toBeInTheDocument();
    // Humanized active label drives the calm line.
    expect(screen.getByText("Running code")).toBeInTheDocument();
    // No developer-y chrome.
    expect(screen.queryByText(/Steps:/)).not.toBeInTheDocument();
    expect(screen.queryByText("execute started")).not.toBeInTheDocument();
    expect(screen.queryByText("Execute")).not.toBeInTheDocument();
  });

  it("keeps steps collapsed until opened, then reveals humanized steps without raw detail", () => {
    const runEvents = [
      reasoning("Weighing the variance.", "running", 1),
      reasoning("Settled on an approach.", "completed", 2),
      toolCall("execute", 3),
    ];
    render(
      <ChatRunSteps runEvents={runEvents} progressEvents={[]} isStreaming statusText={null} />
    );

    // Collapsed: neither the list nor the raw reasoning detail is shown.
    expect(screen.queryByText("Thinking")).not.toBeInTheDocument();
    expect(screen.queryByText("Weighing the variance.")).not.toBeInTheDocument();

    fireEvent.click(screen.getByText("Running code"));

    // Expanded: humanized steps appear; reasoning detail text never renders.
    expect(screen.getByText("Thinking")).toBeInTheDocument();
    expect(screen.getAllByText("Running code").length).toBeGreaterThanOrEqual(1);
    expect(screen.queryByText("Weighing the variance.")).not.toBeInTheDocument();
    expect(screen.queryByText("Settled on an approach.")).not.toBeInTheDocument();
  });

  it("collapses repeated identical tool steps into one calm line", () => {
    const runEvents = [toolCall("ls", 1), toolCall("ls", 2), reasoning("t", "completed", 3)];
    render(
      <ChatRunSteps runEvents={runEvents} progressEvents={[]} isStreaming statusText={null} />
    );

    fireEvent.click(screen.getByText("Looking through files"));

    // One in the trigger + exactly one in the de-duplicated list = 2 (not 3).
    expect(screen.getAllByText("Looking through files").length).toBe(2);
  });
});
