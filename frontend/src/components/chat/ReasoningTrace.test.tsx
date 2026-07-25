import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ReasoningTrace } from "./ReasoningTrace";

describe("ReasoningTrace", () => {
  it("renders nothing when there is no reasoning text", () => {
    const { container } = render(<ReasoningTrace text="   " />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders a collapsed-by-default disclosure containing the reasoning text", () => {
    render(<ReasoningTrace text="First I considered A, then B." />);

    // The affordance is present and labeled.
    expect(screen.getByText("Thought process")).toBeInTheDocument();

    // It is a native <details>, collapsed by default (no open attribute), so a reader must opt in.
    const details = screen.getByText("Thought process").closest("details");
    expect(details).not.toBeNull();
    expect(details).not.toHaveAttribute("open");

    // The full reasoning text is mounted (details keeps its body in the DOM) and revealed on open.
    expect(screen.getByText("First I considered A, then B.")).toBeInTheDocument();
  });
});
