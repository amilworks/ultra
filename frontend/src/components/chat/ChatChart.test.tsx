import { render, screen } from "@testing-library/react";
import { beforeAll, describe, expect, it, vi } from "vitest";

import ChatChart from "./ChatChart";

// The code-block fallback (shiki theme detection) reads matchMedia, absent in jsdom.
beforeAll(() => {
  if (!window.matchMedia) {
    vi.stubGlobal(
      "matchMedia",
      (query: string) => ({
        matches: false,
        media: query,
        onchange: null,
        addListener: () => {},
        removeListener: () => {},
        addEventListener: () => {},
        removeEventListener: () => {},
        dispatchEvent: () => false,
      }),
    );
  }
});

const validSpec = JSON.stringify({
  type: "line",
  title: "Weekly runs",
  x: "week",
  series: [{ key: "runs", label: "Runs" }],
  data: [
    { week: "W1", runs: 10 },
    { week: "W2", runs: 14 },
  ],
});

describe("ChatChart", () => {
  it("renders a chart figure for a valid spec", () => {
    render(<ChatChart source={validSpec} />);
    // title + accessible group render outside recharts' sizing container.
    expect(screen.getByText("Weekly runs")).toBeInTheDocument();
    expect(screen.getByRole("group", { name: "Weekly runs" })).toBeInTheDocument();
  });

  it("falls back to a code block for an incomplete/streaming spec", () => {
    const partial = '{"type":"line","x":"week","series":[';
    render(<ChatChart source={partial} />);
    expect(screen.queryByRole("group")).not.toBeInTheDocument();
    // the raw source is shown (as a code block), not an error
    expect(screen.getByText(/"type":"line"/)).toBeInTheDocument();
  });

  it("falls back to a code block for an invalid (unknown type) spec", () => {
    const bad = JSON.stringify({ type: "sankey", x: "a", series: [{ key: "b" }], data: [{ a: 1, b: 2 }] });
    render(<ChatChart source={bad} />);
    expect(screen.queryByRole("group")).not.toBeInTheDocument();
  });
});
