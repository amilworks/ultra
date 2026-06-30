import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AnimatedTokenCount } from "./AnimatedTokenCount";

const installMatchMedia = (matches: boolean) => {
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: query === "(prefers-reduced-motion: reduce)" ? matches : false,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  });
};

describe("AnimatedTokenCount", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    installMatchMedia(false);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("counts upward between increasing token totals", () => {
    const { rerender } = render(<AnimatedTokenCount value={1000} durationMs={600} />);
    expect(screen.getByText("1.0K tokens")).toBeInTheDocument();

    rerender(<AnimatedTokenCount value={4000} durationMs={600} />);

    act(() => {
      vi.advanceTimersByTime(300);
    });

    const midway = screen.getByText(/K tokens$/).textContent ?? "";
    expect(midway).not.toBe("1.0K tokens");
    expect(midway).not.toBe("4.0K tokens");

    act(() => {
      vi.advanceTimersByTime(600);
    });

    expect(screen.getByText("4.0K tokens")).toBeInTheDocument();
  });

  it("jumps to the next value for reduced-motion users", () => {
    installMatchMedia(true);
    const { rerender } = render(<AnimatedTokenCount value={1000} durationMs={600} />);

    rerender(<AnimatedTokenCount value={4000} durationMs={600} />);

    expect(screen.getByText("4.0K tokens")).toBeInTheDocument();
  });

  it("snaps small increments instantly instead of animating (no frame churn)", () => {
    const { rerender } = render(<AnimatedTokenCount value={1000} durationMs={600} />);
    expect(screen.getByText("1.0K tokens")).toBeInTheDocument();

    // A sub-threshold (+20) increment shows the exact new total with no intermediate animation.
    rerender(<AnimatedTokenCount value={1020} durationMs={600} />);
    expect(screen.getByText("1.0K tokens")).toBeInTheDocument(); // 1020 still formats to 1.0K
    // No pending rAF work to advance — it already settled.
    act(() => {
      vi.advanceTimersByTime(600);
    });
    expect(screen.getByLabelText("1,020 tokens")).toBeInTheDocument();
  });

  it("jumps to the exact total when the tab is hidden (rAF is paused in background tabs)", () => {
    const hiddenSpy = vi.spyOn(document, "hidden", "get").mockReturnValue(true);
    const { rerender } = render(<AnimatedTokenCount value={1000} durationMs={600} />);

    rerender(<AnimatedTokenCount value={9000} durationMs={600} />);
    // Even a large increment snaps while hidden, rather than starting an animation that would
    // stall (and never complete) until the tab is foregrounded again.
    expect(screen.getByText("9.0K tokens")).toBeInTheDocument();

    hiddenSpy.mockRestore();
  });
});
