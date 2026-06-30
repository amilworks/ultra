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
});
