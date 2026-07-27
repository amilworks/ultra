/// <reference types="node" />

import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const stylesSource = readFileSync(path.join(process.cwd(), "src/styles.css"), "utf8");

/** The light `:root` block, up to the first closing brace at column 0. */
const lightRoot = stylesSource.slice(
  stylesSource.indexOf(":root {"),
  stylesSource.indexOf("\n}", stylesSource.indexOf(":root {"))
);
/**
 * The `.dark` TOKEN block. There are two `.dark {` rules in the file — an early
 * one carrying component overrides and a later one carrying the palette — so
 * this anchors on the last, not the first.
 */
const darkBlock = (() => {
  const start = stylesSource.lastIndexOf(".dark {");
  return start === -1 ? "" : stylesSource.slice(start, stylesSource.indexOf("\n}", start));
})();

describe("light theme ink", () => {
  it("moves body ink off black and secondary text to the sage tone", () => {
    // 10.59:1 on --bg-page, down from #171717's 16.4:1 — still far past AA at
    // every size, and the whole theme reads airier for it.
    expect(lightRoot).toMatch(/--text-main:\s*#3b383c;/);
    // 3.77:1 — clears AA for large text, deliberately just under it for small.
    expect(lightRoot).toMatch(/--text-muted:\s*#787f78;/);
  });

  it("points the shadcn foreground tokens at the house ink tokens", () => {
    // Two palettes for one job is how a theme drifts: the app draws most text
    // through --text-main/--text-muted while Tailwind and shadcn primitives draw
    // through these, so an ink change touching one set left the other behind.
    for (const token of [
      "foreground",
      "card-foreground",
      "popover-foreground",
      "secondary-foreground",
      "accent-foreground",
    ]) {
      expect(lightRoot).toMatch(new RegExp(`--${token}:\\s*var\\(--text-main\\);`));
    }
    expect(lightRoot).toMatch(/--muted-foreground:\s*var\(--text-muted\);/);
  });

  it("defines --sidebar-ink-hover in BOTH themes", () => {
    // The sidebar rules reference this var in both themes. An undefined var()
    // makes `color` invalid at computed-value time and the row silently
    // inherits — so a light-only definition breaks dark without erroring.
    expect(lightRoot).toMatch(/--sidebar-ink-hover:\s*#0a0a0a;/);
    expect(darkBlock).toMatch(/--sidebar-ink-hover:\s*#ffffff;/);
    expect(lightRoot).toMatch(/--sidebar-accent-foreground:\s*var\(--sidebar-ink-hover\);/);
    expect(darkBlock).toMatch(/--sidebar-accent-foreground:\s*var\(--sidebar-ink-hover\);/);
  });

  it("darkens sidebar rows on hover with real declarations, not just a token", () => {
    // The rows pin `color: var(--sidebar-foreground)` in unlayered CSS, which
    // beats Tailwind v4's LAYERED hover:text-sidebar-accent-foreground utility
    // no matter its specificity — so the token alone never fires.
    expect(stylesSource).toMatch(
      /\.app-new-chat-button:hover,\s*\.app-resource-browser-button:hover,\s*\.app-bisque-browser-button:hover\s*\{[^}]*color:\s*var\(--sidebar-ink-hover\);/s
    );
    expect(stylesSource).toMatch(
      /\.app-history-button:hover\s*\{[^}]*color:\s*var\(--sidebar-ink-hover\);/s
    );
    expect(stylesSource).toMatch(
      /\.app-history-button\[data-active="true"\]\s*\{[^}]*color:\s*var\(--sidebar-ink-hover\);/s
    );
    // The sidebar rows must be ink at REST, or hover has nowhere to go.
    expect(lightRoot).toMatch(/--sidebar-foreground:\s*var\(--text-main\);/);
  });

  it("drops the ss02 stylistic set for stock Inter", () => {
    expect(stylesSource).toMatch(/font-feature-settings:\s*"liga" 1, "calt" 1;/);
    // Assert on the DECLARATION, not the file: the comment above it names ss02
    // to explain why it was dropped, and should not fail its own guard.
    expect(stylesSource).not.toMatch(/font-feature-settings:[^;]*ss02/);
  });
});

describe("response reading typography", () => {
  it("gives prose a measure without narrowing what needs the width", () => {
    // 49rem of column ran prose to a measured mean of 86.5 characters a line
    // (8 real answer paragraphs, range 77–106) against the 45–75 the eye tracks.
    // 37.5rem lands ~70.6.
    expect(lightRoot).toMatch(/--reading-measure:\s*37\.5rem;/);
    // In rem, NOT ch: `ch` resolves per ELEMENT font-size, so one token handed
    // the h2 an 867px measure and the h3 763px while prose got 654px — three
    // right edges instead of one column.
    expect(lightRoot).not.toMatch(/--reading-measure:\s*[\d.]+ch;/);
    const measureRule = stylesSource.match(
      /\n\.pk-markdown > p,[\s\S]*?\{[^}]*max-width:\s*var\(--reading-measure\);[^}]*\}/
    )?.[0];
    expect(measureRule).toBeTruthy();
    // Narrowing a table to a prose measure re-wraps cells that were readable at
    // full width — trading one legibility problem for a worse one.
    expect(measureRule).not.toMatch(/\btable\b|\bpre\b|codeblock/);
  });

  it("keeps reading size and leading in the comfortable band", () => {
    expect(lightRoot).toMatch(/--font-size-reading:\s*1rem;/);
    expect(lightRoot).toMatch(/--line-height-reading:\s*1\.62;/);
    expect(lightRoot).toMatch(/--font-weight-reading-body:\s*400;/);
  });

  it("never lets inline emphasis outweigh a heading", () => {
    // h2/h3/h4 are all --font-weight-reading-heading. At 700, inline **emphasis**
    // was heavier than every heading above it, and at h4's 16px it beat the
    // heading at identical size.
    const heading = lightRoot.match(/--font-weight-reading-heading:\s*(\d+);/)?.[1];
    const strong = lightRoot.match(/--font-weight-reading-strong:\s*(\d+);/)?.[1];
    expect(heading).toBe("600");
    expect(strong).toBe("600");
    expect(Number(strong)).toBeLessThanOrEqual(Number(heading));
    // UI chrome keeps 700 — different job, sparse use.
    expect(lightRoot).toMatch(/--font-weight-strong:\s*700;/);
  });

  it("stops auto-hyphenating prose once it has a measure", () => {
    // At ~86 characters `hyphens: auto` absorbed the ragged gaps; at ~70 it fired
    // constantly ("bur-rows", "enhance-ment"), which reads as a rendering defect.
    expect(stylesSource).toMatch(
      /\.pk-markdown :where\(p, li, blockquote\)\s*\{[^}]*hyphens:\s*manual;/s
    );
    // overflow-wrap still catches tokens too long to fit, so nothing overflows.
    expect(stylesSource).toMatch(
      /\.pk-markdown :where\(p, li, blockquote, td, th, a, strong, em\)\s*\{[^}]*overflow-wrap:\s*break-word;/s
    );
  });
});
