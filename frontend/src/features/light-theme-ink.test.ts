/// <reference types="node" />

import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const stylesSource = readFileSync(path.join(process.cwd(), "src/styles.css"), "utf8");
const typographySource = readFileSync(path.join(process.cwd(), "src/typography.css"), "utf8");

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
  it("uses Ultra Sans as the product face with Inter as its coverage fallback", () => {
    const productStack =
      '"BisQue Ultra Sans", "BisQue Inter Coverage", system-ui, "Segoe UI", sans-serif';
    expect(typographySource).toContain(`--font-sans: ${productStack};`);
    expect(typographySource).toContain(`--font-reading: ${productStack};`);
  });

  it("puts body and secondary text on the Meridian magnitude ladder", () => {
    // Meridian Drift · Day. Hierarchy is a geometric series in CONTRAST RATIO
    // at 1.80x per step, solved numerically against the ground the text sits
    // on — not a set of greys anyone picked.
    //   m0 #171b1d 16.73:1 body · m1 #424547 9.32:1 · m2 #696b6d 5.16:1
    // m0 is not #000000 for the same reason Night's ink is not #ffffff: a
    // clipped top end reads as a display rather than as a material.
    expect(lightRoot).toMatch(/--text-main:\s*#171b1d;/);
    // m2, the last rung that can legally hold text. 5.16:1 on the ground — the
    // grey it replaced sat under the AA line.
    expect(lightRoot).toMatch(/--text-muted:\s*#696b6d;/);
  });

  it("re-solves the ladder for Night rather than inverting it", () => {
    // The ladder was first derived in LUMINANCE space, which works on a dark
    // ground only because contrast is (L_hi + .05)/(L_lo + .05) and the .05
    // floor dominates when L_lo is near zero. Mirrored onto a light ground it
    // collapses — m1 lands at #acadae, 2.02:1. Contrast-space is
    // ground-independent, so ONE step governs both blocks and each solves its
    // own values. Same step, different hexes: that is the invariant.
    expect(darkBlock).toMatch(/--text-main:\s*#dce3ea;/);
    expect(darkBlock).toMatch(/--text-muted:\s*#777c82;/);
    // Neither ground clips to pure black or pure white.
    expect(lightRoot).not.toMatch(/--text-main:\s*#000000;/);
    expect(darkBlock).not.toMatch(/--text-main:\s*#(fff|ffffff);/);
  });

  it("measures the ladder — a 1.80x contrast step, computed, on both grounds", () => {
    // The design's claim is that hierarchy is a DERIVATION, not a set of greys
    // somebody picked. So derive it: every adjacent pair of rungs must sit one
    // 1.80x contrast step apart on its own ground, and m2 — the last rung
    // allowed to carry text — must hold AA. Pinned hexes alone can't catch a
    // future edit that swaps one grey for a plausible-looking neighbour.
    const luminance = (hex: string): number => {
      const channel = (c: number): number => {
        const s = c / 255;
        return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
      };
      const n = parseInt(hex.slice(1), 16);
      return (
        0.2126 * channel((n >> 16) & 255) +
        0.7152 * channel((n >> 8) & 255) +
        0.0722 * channel(n & 255)
      );
    };
    const contrast = (a: string, b: string): number => {
      const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
      return (hi + 0.05) / (lo + 0.05);
    };
    const ladders = [
      { ground: "#fafbfb", rungs: ["#171b1d", "#424547", "#696b6d", "#939697", "#c8c9ca"] },
      { ground: "#0b0e11", rungs: ["#dce3ea", "#a5abb0", "#777c82", "#505559", "#2b2e32"] },
    ];
    for (const { ground, rungs } of ladders) {
      const ratios = rungs.map((rung) => contrast(rung, ground));
      expect(ratios[0]).toBeGreaterThan(14); // m0 body ink
      expect(ratios[2]).toBeGreaterThanOrEqual(4.5); // m2 must hold AA
      for (let i = 1; i < ratios.length; i++) {
        const step = ratios[i - 1] / ratios[i];
        expect(step, `${ground} m${i - 1}->m${i}`).toBeGreaterThan(1.68);
        expect(step, `${ground} m${i - 1}->m${i}`).toBeLessThan(1.95);
      }
    }
    // The shipped tokens must BE rungs of these ladders, so the hex pins above
    // and this derivation cannot drift apart silently.
    expect(lightRoot).toMatch(/--bg-main:\s*#fafbfb;/);
    expect(darkBlock).toMatch(/--bg-main:\s*#0b0e11;/);
  });

  it("keeps the Day chrome achromatic — crisp, not tinted", () => {
    // The plate. Day surfaces run a channel spread of 1–2; past 3 they start
    // reading as a tinted panel, which is the failure Night hit at 12–16.
    const spread = (hex: string): number => {
      const c = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
      return Math.max(...c) - Math.min(...c);
    };
    for (const token of ["bg-main", "bg-panel-strong", "sidebar"]) {
      const hex = lightRoot.match(new RegExp(`--${token}:\\s*(#[0-9a-f]{6});`))?.[1];
      expect(hex, `${token} must be a hex literal`).toBeTruthy();
      expect(spread(hex as string), `${token} (${hex}) is tinted`).toBeLessThanOrEqual(3);
    }
  });

  it("gives Night its own danger instead of inheriting paper's", () => {
    // --danger was light-only for the theme's whole life: dark inherited
    // #c62828, which measures 3.44:1 on the Night ground — sub-AA error text.
    // #d45e5e is the first stop along #c62828 -> white clearing AA on both
    // Night surfaces it sits on (5.13:1 ground, 4.76:1 panel).
    expect(lightRoot).toMatch(/--danger:\s*#c62828;/);
    expect(darkBlock).toMatch(/--danger:\s*#d45e5e;/);
  });

  it("keeps the Night chrome from reading as a blue panel", () => {
    // An earlier cut ran the dark surfaces at a channel spread of 12–16 and the
    // sidebar read as blue rather than as black. Baffle black keeps a spread of
    // 6 because optical flock genuinely is cool; everything above it is held
    // tighter. Guard the surfaces that carry large areas.
    const spread = (hex: string) => {
      const c = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
      return Math.max(...c) - Math.min(...c);
    };
    for (const token of ["bg-main", "bg-panel-strong", "sidebar"]) {
      const hex = darkBlock.match(new RegExp(`--${token}:\\s*(#[0-9a-f]{6});`))?.[1];
      expect(hex, `${token} must be a hex literal`).toBeTruthy();
      expect(spread(hex as string), `${token} (${hex}) is too chromatic`).toBeLessThanOrEqual(8);
    }
  });

  it("defines the strong hairline in BOTH theme blocks", () => {
    // --line-strong was consumed by three hover rules for months while never
    // being defined: border-color is non-inherited, so the invalid var()
    // resolved to currentcolor and those hovers silently painted full text
    // ink. The undefined-var() failure mode is invisible — no error, no
    // console warning — so the definitions are pinned here in both blocks
    // (any token consumed by theme-agnostic rules must exist in every theme).
    // Both grounds now carry the SAME alpha pair (.12 rest / .22 hover), which
    // they could not before: each resolves onto its own m4 rung, so the
    // perceived step matches without the hand-tuned per-theme offsets this
    // token used to need.
    expect(lightRoot).toMatch(/--line-strong:\s*rgba\(23, 27, 29, 0\.22\);/);
    expect(darkBlock).toMatch(/--line-strong:\s*rgba\(220, 227, 234, 0\.22\);/);
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
    expect(lightRoot).toMatch(/--sidebar-ink-hover:\s*#0b0e0f;/);
    expect(darkBlock).toMatch(/--sidebar-ink-hover:\s*#eef3f8;/);
    expect(lightRoot).toMatch(/--sidebar-accent-foreground:\s*var\(--sidebar-ink-hover\);/);
    expect(darkBlock).toMatch(/--sidebar-accent-foreground:\s*var\(--sidebar-ink-hover\);/);
  });

  it("keeps the sidebar on the ladder: rows at full ink, supporting copy at m1", () => {
    // Upstream moved Recents rows to full ink with pill-carried hover (the
    // macOS-sidebar model), superseding the earlier quiet-row split. Meridian
    // keeps that model and puts the QUIET tier — account copy and labels that
    // draw --sidebar-foreground — on m1, solved against the sidebar's own
    // ground (8.07:1 light / 8.04:1 dark).
    expect(lightRoot).toMatch(/--sidebar-nav-foreground:\s*var\(--text-main\);/);
    // m1. After the one-step-lighter shift the sidebar sits on the original
    // stage rung (m2 measures a legal 4.81:1 there), but the rows keep m1 —
    // the quiet tier is a hierarchy decision, one clean magnitude below the
    // nav ink above it, with margin instead of shipping at the AA line.
    expect(lightRoot).toMatch(/--sidebar-foreground:\s*#424547;/);
    expect(lightRoot).not.toMatch(/--sidebar-foreground:\s*var\(--text-main\);/);
    // Dark shipped its Recents at full white — it never got the quiet-sidebar
    // split light had. Both grounds now carry it.
    expect(darkBlock).toMatch(/--sidebar-foreground:\s*#a5abb0;/);
    expect(darkBlock).not.toMatch(/--sidebar-foreground:\s*var\(--text-main\);/);
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
      /\n\.app-history-button\s*\{[^}]*color:\s*var\(--sidebar-nav-foreground\);/s
    );
    expect(stylesSource).toMatch(/color:\s*var\(--sidebar-foreground\);/);
  });

  it("puts the hover affordance where it can actually be seen", () => {
    // Recents rows darken: rest m1 to --sidebar-ink-hover is a legible ~2x
    // contrast step. This must be a real declaration, not just the
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

  it("keeps Ultra Sans on its authored default glyph system", () => {
    expect(stylesSource).toMatch(/font-feature-settings:\s*"liga" 1, "calt" 1;/);
    // The custom C/O/G/s are defaults. The slashed zero remains dormant until a
    // data surface opts in, so no global stylistic set belongs here.
    expect(stylesSource).not.toMatch(/font-feature-settings:[^;]*ss02/);
    expect(stylesSource).not.toMatch(/font-feature-settings:[^;]*zero/);
  });
});

describe("response reading typography", () => {
  it("gives prose a measure without narrowing what needs the width", () => {
    // A fixed answer sample at 16px put 44rem at 83.9 characters per line and
    // the unconstrained 49rem column at 92.9. A 45rem cap keeps the intentionally
    // relaxed conversational measure near that calibrated wrap point while
    // preserving the wider column for code, tables and figures.
    expect(lightRoot).toMatch(/--reading-measure:\s*45rem;/);
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
    // Ultra Sans Regular at 16px keeps long-form answers lighter than the
    // custom family's compact UI design point at 430 / opsz 15.
    expect(lightRoot).toMatch(/--font-weight-reading-body:\s*400;/);
    expect(stylesSource).toMatch(
      /\.pk-message-content-plain\s*\{[^}]*font-weight:\s*var\(--font-weight-reading-body\);/s
    );
  });

  it("keeps the New Chat invitation lighter than the reading voice", () => {
    const desktopInvitation = lightRoot.match(
      /--font-weight-desktop-invitation:\s*(\d+);/
    )?.[1];
    const invitation = lightRoot.match(/--font-weight-invitation:\s*(\d+);/)?.[1];
    const reading = lightRoot.match(/--font-weight-reading-body:\s*(\d+);/)?.[1];
    expect(desktopInvitation).toBe("300");
    expect(invitation).toBe("350");
    expect(reading).toBe("400");
    expect(Number(desktopInvitation)).toBeLessThan(Number(invitation));
    expect(Number(invitation)).toBeLessThan(Number(reading));
    expect(stylesSource).toMatch(
      /\.blank-chat-welcome-hero\s*\{[^}]*font-size:\s*1\.625rem;[^}]*font-weight:\s*var\(--font-weight-desktop-invitation\);/s
    );
    expect(stylesSource).toMatch(
      /\.mobile-chat-hero-title\s*\{[^}]*font-weight:\s*var\(--font-weight-invitation\);/s
    );
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
    // UI chrome is 690 — a separate, sparse display job. Retired Inter at
    // opsz24/w670 measured lowercase stem/x-height 0.262550. Ultra Sans pinned
    // at opsz13 solves to 0.26254 at w689.18, rounded to the nearest ten.
    expect(lightRoot).toMatch(/--font-weight-strong:\s*690;/);
    // Whatever the chrome weight is, it must still outrank a reading heading.
    const chromeStrong = lightRoot.match(/--font-weight-strong:\s*(\d+);/)?.[1];
    expect(Number(chromeStrong)).toBeGreaterThan(Number(heading));
  });

  it("pins mono surfaces so they never inherit the proportional reading grade", () => {
    // Ultra Mono has its own variable weight axis. Code and data stay on the
    // nominal Regular master instead of inheriting the surrounding sans role.
    expect(lightRoot).toMatch(/--font-weight-mono:\s*400;/);
    const mustPin = [
      /\.pk-inline-code\s*\{[^}]*font-weight:\s*var\(--font-weight-mono\);/s,
      /\.pk-code-render :where\(pre, code\)\s*\{[^}]*font-weight:\s*var\(--font-weight-mono\);/s,
    ];
    for (const pattern of mustPin) expect(stylesSource).toMatch(pattern);
  });

  it("uses Ultra Sans spacing derived by size and role", () => {
    for (const [token, value] of [
      ["display-regular", "-0.01em"],
      ["display-strong", "-0.018em"],
      ["reading-h1", "-0.016em"],
      ["reading-h2", "-0.011em"],
      ["reading-h3", "-0.007em"],
      ["reading-small", "-0.003em"],
    ]) {
      expect(lightRoot).toMatch(
        new RegExp(`--tracking-${token}:\\s*${value.replace(".", "\\.")};`)
      );
    }
    expect(stylesSource).toMatch(
      /\.hero-title\s*\{[^}]*letter-spacing:\s*var\(--tracking-display-strong\);/s
    );
    expect(stylesSource).toMatch(
      /\.pk-markdown > :where\(h1\)\s*\{[^}]*letter-spacing:\s*var\(--tracking-reading-h1\);/s
    );
    expect(stylesSource).toMatch(
      /\.pk-markdown > :where\(h2\)\s*\{[^}]*letter-spacing:\s*var\(--tracking-reading-h2\);/s
    );
    expect(stylesSource).toMatch(
      /\.pk-markdown > :where\(h3\)\s*\{[^}]*letter-spacing:\s*var\(--tracking-reading-h3\);/s
    );
    expect(stylesSource).toMatch(
      /\.app-composer-textarea\s*\{[^}]*letter-spacing:\s*0;/s
    );
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
