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
  it("keeps body ink near-black and secondary text on the sage tone", () => {
    // 16.12:1 on --bg-page. Deliberately near-black: this is the reading surface
    // for scientific answers, and the earlier #3b383c (10.59:1) gave up more
    // legibility than the airiness was worth. Airiness is bought elsewhere — a
    // quieter sidebar and a real reading measure — not out of the text itself.
    expect(lightRoot).toMatch(/--text-main:\s*#191919;/);
    // 3.77:1 — clears AA for large text, deliberately just under it for small.
    expect(lightRoot).toMatch(/--text-muted:\s*#787f78;/);
  });

  it("defines the strong hairline in BOTH theme blocks", () => {
    // --line-strong was consumed by three hover rules for months while never
    // being defined: border-color is non-inherited, so the invalid var()
    // resolved to currentcolor and those hovers silently painted full text
    // ink. The undefined-var() failure mode is invisible — no error, no
    // console warning — so the definitions are pinned here in both blocks
    // (any token consumed by theme-agnostic rules must exist in every theme).
    expect(lightRoot).toMatch(/--line-strong:\s*rgba\(23, 23, 23, 0\.16\);/);
    expect(darkBlock).toMatch(/--line-strong:\s*rgba\(255, 255, 255, 0\.24\);/);
    // And it must stay consumed — a defined-but-dead token is the opposite drift.
    expect(stylesSource).toMatch(/border-color:\s*var\(--line-strong\)/);
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

  it("splits the sidebar: structural nav at full ink, Recents quiet", () => {
    // Only the Recents list is quieted. It is long, repetitive and the actual
    // source of sidebar noise; the handful of fixed nav rows are the app's spine
    // and reading them as noise made the whole sidebar mushy.
    expect(lightRoot).toMatch(/--sidebar-nav-foreground:\s*var\(--text-main\);/);
    // 6.41:1 vs body ink's 16.12:1, and past AA's 4.5:1 for the 14px/500 labels.
    expect(lightRoot).toMatch(/--sidebar-foreground:\s*#575a57;/);
    expect(lightRoot).not.toMatch(/--sidebar-foreground:\s*var\(--text-main\);/);
    // Structural nav draws the nav token, not the quiet one.
    expect(stylesSource).toMatch(
      /\.app-new-chat-button,\s*\.app-resource-browser-button,\s*\.app-bisque-browser-button\s*\{[^}]*color:\s*var\(--sidebar-nav-foreground\);/s
    );
    expect(stylesSource).toMatch(
      /\.app-bisque-link-button\s*\{[^}]*color:\s*var\(--sidebar-nav-foreground\);/s
    );
    expect(stylesSource).toMatch(
      /\.app-sidebar-brand-button\[data-slot="button"\]\s*\{[^}]*color:\s*var\(--sidebar-nav-foreground\);/s
    );
    // Recents keeps the quiet token.
    expect(stylesSource).toMatch(
      /\n\.app-history-button\s*\{[^}]*color:\s*var\(--sidebar-foreground\);/s
    );
  });

  it("puts the hover affordance where it can actually be seen", () => {
    // Recents rows darken: from 6.41:1 to --sidebar-ink-hover's 18.15:1 is a
    // legible 2.83:1 step. This must be a real declaration, not just the
    // --sidebar-accent-foreground token: the rows pin `color:
    // var(--sidebar-foreground)` in unlayered CSS, which beats Tailwind v4's
    // LAYERED hover:text-sidebar-accent-foreground utility at any specificity.
    expect(stylesSource).toMatch(
      /\.app-history-button:hover\s*\{[^}]*color:\s*var\(--sidebar-ink-hover\);/s
    );
    expect(stylesSource).toMatch(
      /\.app-history-button\[data-active="true"\]\s*\{[^}]*color:\s*var\(--sidebar-ink-hover\);/s
    );
    // Structural nav must NOT darken on hover: already at full ink, so the step
    // to #0a0a0a is 1.13:1 — invisible, and misleading to keep as if it worked.
    // Its background shift carries the affordance instead.
    const navHover = stylesSource.match(
      /\.app-new-chat-button:hover,\s*\.app-resource-browser-button:hover,\s*\.app-bisque-browser-button:hover\s*\{[^}]*\}/s
    )?.[0];
    // Lookbehind so `border-color: transparent` does not count as a text colour.
    const TEXT_COLOR = /(?<![-\w])color\s*:/;
    expect(navHover).toBeTruthy();
    expect(navHover).toMatch(/background:/);
    expect(navHover).not.toMatch(TEXT_COLOR);
    const linkHover = stylesSource.match(/\.app-bisque-link-button:hover\s*\{[^}]*\}/s)?.[0];
    expect(linkHover).toMatch(/background:/);
    expect(linkHover).not.toMatch(TEXT_COLOR);
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
    // Measured on ONE fixed set of 5 answer paragraphs so the values compare:
    // 37.5rem = 72.3 cpl, 40rem = 78.8, 42.5rem = 83.9, 44rem = 83.9,
    // unconstrained = 92.9. 44rem sits KNOWINGLY above the 45–75 guideline at
    // ~84 — the column is a modest share of a wide screen and reads cramped when
    // held to the middle of the band. The cap still does real work: the measure
    // that made readers re-scan was the unconstrained 92.9, not 75.
    expect(lightRoot).toMatch(/--reading-measure:\s*44rem;/);
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

  it("gives wide equations their own scrollport instead of amputating them", () => {
    // KaTeX sets `white-space: nowrap` on display math, and html/body are both
    // `overflow-x: clip` so the page never scrolls sideways. `clip` creates NO
    // scrollport, so the two together silently truncate a wide formula: measured
    // on a 390px phone, a radar-equation display ran to 728px with 338px
    // unreachable — no scrollbar, no gesture that could reveal it.
    const rule = stylesSource.match(
      /\.pk-markdown \.katex-display,[\s\S]*?\{[^}]*\}/
    )?.[0];
    expect(rule).toBeTruthy();
    expect(rule).toMatch(/overflow-x:\s*auto;/);
    // Must NOT be `hidden`/`clip` — those re-create the amputation.
    expect(rule).not.toMatch(/overflow-x:\s*(hidden|clip|visible);/);
    // A horizontal scrollport reports fractional vertical overflow from KaTeX's
    // negative struts, which would show a useless vertical scrollbar on tall math.
    expect(rule).toMatch(/overflow-y:\s*hidden;/);
    // The page itself must still never scroll sideways.
    expect(stylesSource).toMatch(/overflow-x:\s*clip;/);
    // Equations are NOT in the reading-measure list — a formula needs the full
    // column, and narrowing it only makes the scroll longer.
    const measureRule = stylesSource.match(
      /\n\.pk-markdown > p,[\s\S]*?\{[^}]*max-width:\s*var\(--reading-measure\);[^}]*\}/
    )?.[0];
    expect(measureRule).not.toMatch(/katex/);
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
