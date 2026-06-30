import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

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

// Avoid the async Shiki dynamic import in jsdom; the clamp affordance is driven by the synchronous
// line count, not the highlighter.
vi.mock("@/lib/shiki", () => ({
  codeToHtml: vi.fn(async ({ code }: { code: string }) => `<pre><code>${code}</code></pre>`),
}));

import { CodeBlockCode } from "./code-block";

describe("CodeBlockCode clamping", () => {
  it("clamps a giant code block behind an expand toggle and back", () => {
    const code = Array.from({ length: 500 }, (_, i) => `line ${i}`).join("\n");
    render(<CodeBlockCode code={code} language="text" showCopyButton={false} showLanguage={false} />);

    const expand = screen.getByRole("button", { name: /show all 500 lines/i });
    expect(expand).toHaveAttribute("aria-expanded", "false");

    fireEvent.click(expand);
    expect(screen.getByRole("button", { name: /collapse/i })).toHaveAttribute(
      "aria-expanded",
      "true"
    );
  });

  it("does not clamp a small code block", () => {
    render(<CodeBlockCode code={"a\nb\nc"} language="text" showCopyButton={false} showLanguage={false} />);
    expect(screen.queryByRole("button", { name: /show all/i })).not.toBeInTheDocument();
  });
});
