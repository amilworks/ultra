import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ProModeDevTrace } from "./ProModeDevTrace";

const conversation = {
  rounds: [
    {
      round_index: 1,
      messages: [{ sender_role: "Critic", content: "objection body text" }],
    },
    {
      round_index: 2,
      messages: [{ sender_role: "Synthesizer", content: "second round body" }],
    },
  ],
};

describe("ProModeDevTrace", () => {
  it("shows only the summary counts when collapsed and mounts the round tree once expanded", () => {
    render(
      <ProModeDevTrace
        messageId="m1"
        conversation={conversation}
        isCopied={false}
        onCopy={() => {}}
      />
    );

    // Collapsed (default): cheap summary counts render, but the heavy nested round/message tree
    // is NOT mounted — so a large trace costs nothing on first paint.
    expect(screen.getByText(/2 rounds/)).toBeInTheDocument();
    expect(screen.queryByText("objection body text")).not.toBeInTheDocument();
    expect(screen.queryByText("second round body")).not.toBeInTheDocument();

    // Expanding mounts the full tree on demand.
    const details = screen.getByTestId("pro-mode-dev-trace") as HTMLDetailsElement;
    details.open = true;
    fireEvent(details, new Event("toggle"));

    expect(screen.getByText("objection body text")).toBeInTheDocument();
    expect(screen.getByText("second round body")).toBeInTheDocument();
  });

  it("renders nothing when there is no trace content", () => {
    const { container } = render(
      <ProModeDevTrace messageId="m2" conversation={{}} isCopied={false} onCopy={() => {}} />
    );
    expect(container).toBeEmptyDOMElement();
  });
});
