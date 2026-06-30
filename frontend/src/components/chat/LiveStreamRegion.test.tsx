import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { LiveStreamRegion } from "./LiveStreamRegion";

beforeEach(() => {
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  });
});

// eslint-disable-next-line require-yield
async function* emptyStream(): AsyncIterable<string> {
  return;
}

describe("LiveStreamRegion", () => {
  it("streams inline for a plain reply — no disclosure", () => {
    render(
      <LiveStreamRegion
        messageId="m1"
        liveStream={emptyStream()}
        foldIntoReasoning={false}
        onComplete={() => {}}
      />
    );
    expect(screen.queryByText("Show live reasoning")).not.toBeInTheDocument();
    // The stream body renders inline.
    expect(document.getElementById("m1")).toBeInTheDocument();
  });

  it("folds the narration into an opt-in disclosure for an agentic run", () => {
    const { container } = render(
      <LiveStreamRegion
        messageId="m2"
        liveStream={emptyStream()}
        foldIntoReasoning={true}
        onComplete={() => {}}
      />
    );
    expect(screen.getByText("Show live reasoning")).toBeInTheDocument();
    const details = container.querySelector("details.live-reasoning");
    expect(details).toBeInTheDocument();
    // Calm default: collapsed (no `open` attribute).
    expect(details?.hasAttribute("open")).toBe(false);
    // Load-bearing: the stream body is still MOUNTED in the DOM while collapsed, so the single-
    // consumer liveStream keeps being consumed and accumulating — expanding loses no tokens.
    expect(document.getElementById("m2")).toBeInTheDocument();
  });
});
