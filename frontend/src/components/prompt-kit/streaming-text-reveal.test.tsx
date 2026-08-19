/// <reference types="node" />

import { readFileSync } from "node:fs";
import path from "node:path";
import { render, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { Markdown } from "./markdown";
import { MarkdownResponseStream } from "./markdown-response-stream";

const stylesSource = readFileSync(
  path.join(process.cwd(), "src/styles.css"),
  "utf8"
);

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

describe("streaming prose reveal", () => {
  it("wraps only the newest prose word without duplicating the response", () => {
    const { container } = render(
      <Markdown streamingReveal>Measurements become precise</Markdown>
    );

    const tails = container.querySelectorAll(".pk-stream-tail");
    expect(tails).toHaveLength(1);
    expect(tails[0]).toHaveTextContent("precise");
    expect(container).toHaveTextContent("Measurements become precise");
  });

  it("holds one phase while a word grows, then restarts at the next word", () => {
    const { container, rerender } = render(
      <Markdown streamingReveal>Measure</Markdown>
    );
    const firstPhase = container.querySelector(".pk-stream-tail")?.className;

    rerender(<Markdown streamingReveal>Measurement</Markdown>);
    expect(container.querySelector(".pk-stream-tail")?.className).toBe(firstPhase);

    rerender(<Markdown streamingReveal>Measurement settles</Markdown>);
    const nextPhase = container.querySelector(".pk-stream-tail")?.className;
    expect(nextPhase).not.toBe(firstPhase);

    rerender(<Markdown streamingReveal>Measurement settles.</Markdown>);
    expect(container.querySelector(".pk-stream-tail")?.className).toBe(nextPhase);
  });

  it("does not alter settled Markdown", () => {
    const { container } = render(<Markdown>Already resolved</Markdown>);

    expect(container.querySelector(".pk-stream-tail")).not.toBeInTheDocument();
  });

  it("is enabled by the real streaming Markdown path", async () => {
    const { container } = render(
      <MarkdownResponseStream
        textStream="A measured field resolves"
        speed={100}
        characterChunkSize={100}
      />
    );

    await waitFor(() => {
      expect(container.querySelector(".pk-stream-tail")).toHaveTextContent(
        "resolves"
      );
    });
  });

  it.each([
    ["heading", "## Stable heading"],
    ["code", "```ts\nconst value = 3;\n```"],
    ["table", "| metric | value |\n| --- | ---: |\n| signal | 3 |"],
    ["math", "The estimate is $x^2$."],
  ])("keeps %s content optically stable", (_surface, markdown) => {
    const { container } = render(
      <Markdown streamingReveal>{markdown}</Markdown>
    );

    expect(container.querySelector(".pk-stream-tail")).not.toBeInTheDocument();
  });

  it("drops the decorative traversal for a pathological single block", () => {
    const { container } = render(
      <Markdown streamingReveal>{"a".repeat(12_001)}</Markdown>
    );

    expect(container.querySelector(".pk-stream-tail")).not.toBeInTheDocument();
  });

  it("uses a short focus transition and honors reduced motion", () => {
    expect(stylesSource).toMatch(
      /\.pk-stream-tail\s*\{[^}]*animation-duration:\s*180ms;[^}]*will-change:\s*filter, opacity;/s
    );
    expect(stylesSource).toMatch(/@keyframes pk-stream-focus-0/);
    expect(stylesSource).toMatch(/@keyframes pk-stream-focus-1/);
    expect(stylesSource).toMatch(
      /@media \(prefers-reduced-motion: reduce\)\s*\{\s*\.pk-stream-tail\s*\{[^}]*animation:\s*none;[^}]*filter:\s*none;[^}]*opacity:\s*1;/s
    );
  });
});
