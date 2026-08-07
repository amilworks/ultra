/// <reference types="node" />

import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

/**
 * The Meridian identity contract — the parts of the language that are not
 * colour tokens: the reserved brass accent, the symbol set, the de-boxed
 * chrome, and the welcome-stage field. The magnitude ladder itself is pinned
 * and DERIVED in light-theme-ink.test.ts; this file guards everything built
 * on top of it.
 */

const read = (p: string): string => readFileSync(path.join(process.cwd(), p), "utf8");

const stylesSource = read("src/styles.css");
const appSource = read("src/App.tsx");
const faviconSource = read("public/favicon.svg");

const lightRoot = stylesSource.slice(
  stylesSource.indexOf(":root {"),
  stylesSource.indexOf("\n}", stylesSource.indexOf(":root {"))
);
const darkBlock = (() => {
  const start = stylesSource.lastIndexOf(".dark {");
  return start === -1 ? "" : stylesSource.slice(start, stylesSource.indexOf("\n}", start));
})();

describe("the point of light — brass is reserved", () => {
  it("defines the live accent in both themes, halo dark-only", () => {
    expect(lightRoot).toMatch(/--accent-live:\s*#8a6a12;/);
    expect(darkBlock).toMatch(/--accent-live:\s*#e8b84b;/);
    // Halation is a dark-ground phenomenon: Day's halo is transparent so the
    // shared glow declarations become no-ops on paper.
    expect(lightRoot).toMatch(/--accent-live-halo:\s*transparent;/);
    expect(darkBlock).toMatch(
      /--accent-live-halo:\s*color-mix\(in srgb, var\(--accent-live\) 45%, transparent\);/
    );
  });

  it("keeps brass unborrowable — each pigment literal appears exactly once", () => {
    // The accent stays findable only while nothing else is allowed to use it.
    // Every consumer must go through var(--accent-live); the raw pigment
    // exists only at its definition.
    expect(stylesSource.match(/#8a6a12/gi)).toHaveLength(1);
    expect(stylesSource.match(/#e8b84b/gi)).toHaveLength(1);
  });

  it("gives brass its three wired meanings: running, thinking, the field", () => {
    // Running conversations — the point of light in the sidebar.
    expect(stylesSource).toMatch(
      /\.running-status-pill\s*\{[^}]*--running-status-ink:\s*var\(--accent-live\);/s
    );
    // Live thinking — the constellation and the trace.
    expect(stylesSource).toMatch(
      /\.thinking-constellation\s*\{[^}]*color:\s*var\(--accent-live\);/s
    );
    expect(stylesSource).toMatch(
      /\.thinking-bar-trace\s*\{[^}]*color:\s*var\(--accent-live\);/s
    );
  });

  it("breathes instead of spinning, and holds still under reduced motion", () => {
    // Light, not motion: the point dims and returns, never disappears, never
    // rotates. The trace accumulates and holds — a record, not a loop.
    expect(stylesSource).toMatch(/@keyframes running-point-breathe/);
    expect(stylesSource).not.toMatch(/running-point[^}]*rotate/s);
    expect(stylesSource).toMatch(/@keyframes trace-write/);
    expect(stylesSource).toMatch(
      /@media \(prefers-reduced-motion: reduce\)\s*\{\s*\.running-status-point\s*\{\s*animation:\s*none;/s
    );
    expect(stylesSource).toMatch(
      /@media \(prefers-reduced-motion: reduce\)\s*\{\s*\.thinking-bar-trace path\s*\{\s*animation:\s*none;\s*stroke-dashoffset:\s*0;/s
    );
  });
});

describe("the palette holds — no off-ladder colour returns", () => {
  it("keeps every retired literal retired", () => {
    // The consistency pass replaced these with ladder rungs, tokens, or
    // token-mixes. Each one reappearing means someone minted a grey or an
    // accent outside the system — the exact drift this file exists to stop.
    // (Viewer glass, the figure lightbox, and axis/slice colours are exempt
    // ON-IMAGE families and do not appear here.)
    const retired = [
      // pre-Meridian theme tokens
      "#f8f8f7", "#f5f5f4", "#fafafa", "#0f0f10", "#111113", "#1b1b1d",
      "#2a2a2f", "#575a57", "#787f78", "#c3c7c3",
      // stray chrome killed in the consistency pass
      "#1d2f57", "#f8fafc", "#1f2937", "#334155", "#0f172a", "#6aa9ff",
      "#050505", "#f8faf9",
      // the earth-tone figure tints and PDF sage
      "#9b8867", "#a78d63", "#efe2c0", "#f6efe0", "#f1e4c8", "#efe4c9",
      "#d7c7a0", "#8b9a6d", "#9eb08a", "#715638", "#4d3a2a",
      "#4f8073", "#28594e", "#536b8f", "#6f8f85",
      // ad-hoc semantics, now tokens
      "#b45309", "#a16207", "#d4503b",
    ];
    // --status-warn's own definition is the one permitted #b45309.
    const withoutWarnDefinition = stylesSource.replace("--status-warn: #b45309;", "");
    for (const hex of retired) {
      expect(withoutWarnDefinition.includes(hex), `${hex} has returned`).toBe(false);
    }
  });

  it("defines the warning semantic on both grounds, orange never gold", () => {
    // Brass is reserved for "running"; a yellow warning would counterfeit it.
    // Night's warn is measurably darker than brass (lum 0.30 vs 0.52).
    expect(lightRoot).toMatch(/--status-warn:\s*#b45309;/);
    expect(darkBlock).toMatch(/--status-warn:\s*#d97f3e;/);
    // Consumers draw the token, and the dead `--warning` fallback is gone.
    expect(stylesSource).not.toMatch(/var\(--warning[,)]/);
    expect(stylesSource).toMatch(/\.viewer-caption-badge\s*\{[^}]*var\(--status-warn\)/s);
    expect(stylesSource).toMatch(/\.training-held-out-flag\s*\{[^}]*var\(--status-warn\)/s);
  });

  it("sends the primary composer action through the ladder, not clipped black", () => {
    expect(stylesSource).toMatch(
      /\.app-composer-submit-button\s*\{[^}]*background:\s*var\(--primary\);[^}]*color:\s*var\(--primary-foreground\);/s
    );
    expect(stylesSource).toMatch(
      /\.app-composer-submit-button:not\(:disabled\):hover,[^{]*\{[^}]*background:\s*var\(--brand-strong\);/s
    );
  });

  it("marks viewer progress with the live accent — loading IS the instrument running", () => {
    expect(stylesSource).toMatch(/\.viewer-progress-bar\s*\{[^}]*background:\s*var\(--accent-live\);/s);
    // The dead --accent-strong fallback went with it.
    expect(stylesSource).not.toMatch(/var\(--accent-strong[,)]/);
  });
});

describe("the symbol set", () => {
  it("keeps the brand on the BisQue mark — the reticle is reserved, not the logo", () => {
    // DECISION, reversed once on purpose: the reticle briefly served as the
    // app mark and was pulled back — the product's lineage outranks the
    // language's own iconography. Brand surfaces (sidebar, auth, favicon)
    // carry the BisQue glyph; the reticle stays defined in the set for
    // alignment/calibration surfaces that may want it later.
    expect(appSource).toMatch(/app-sidebar-brand-mark[\s\S]{0,400}<BisqueMarkIcon/);
    expect(appSource).not.toContain("<ReticleIcon");
    expect(read("src/components/auth/AuthScreen.tsx")).toContain("<BisqueMarkIcon");
    expect(read("src/components/auth/AuthShellScreens.tsx")).toContain("<BisqueMarkIcon");
    // The favicon is the BisQue U-and-wave, not the fiducial.
    expect(faviconSource).toContain("M32 32V72A32 32 0 0 0 96 72V32");
    expect(faviconSource).not.toContain('r="38"');
    // Reserved means still defined: the set keeps the reticle for later.
    expect(read("src/components/icons/MeridianIcons.tsx")).toContain("export function ReticleIcon");
  });

  it("normalises the trace's dash math with pathLength", () => {
    // pathLength=1 lets the CSS write-on animation use dasharray 1 with no
    // geometry-dependent magic numbers.
    expect(read("src/components/icons/MeridianIcons.tsx")).toMatch(/pathLength=\{1\}/);
    expect(stylesSource).toMatch(/\.thinking-bar-trace path\s*\{[^}]*stroke-dasharray:\s*1;/s);
  });
});

describe("de-boxed chrome — depth by value, edges only where real", () => {
  it("separates resource cards by value step, not enclosure", () => {
    expect(stylesSource).toMatch(
      /\.resource-browser-card\s*\{[^}]*border:\s*1px solid transparent;[^}]*background:\s*var\(--bg-panel-strong\);/s
    );
    // Hover is lift and shadow — a border appearing on hover would
    // reintroduce the box the rule deletes.
    const hover = stylesSource.match(
      /\.resource-browser-card:hover,\s*\.resource-browser-card:focus-visible\s*\{[^}]*\}/s
    )?.[0];
    expect(hover).toBeTruthy();
    expect(hover).not.toMatch(/border-color/);
    // Selection keeps its edge on purpose: a selection boundary is a real
    // edge, like an ink contour on a silhouette.
    expect(stylesSource).toMatch(
      /\.resource-browser-card\[data-selected="true"\]\s*\{[^}]*border-color:/s
    );
  });

  it("sinks the search well instead of bordering it", () => {
    expect(lightRoot).toMatch(/--bg-sunk:\s*#e4e6e6;/);
    expect(darkBlock).toMatch(/--bg-sunk:\s*#0d1012;/);
    expect(stylesSource).toMatch(
      /\.resource-browser-search-field\s*\{[^}]*border:\s*1px solid transparent;[^}]*background:\s*var\(--bg-sunk\);/s
    );
  });

  it("keeps pointer focus calm and reserves the ring for the keyboard", () => {
    // Clicking a text input is self-announcing (the caret); the well must not
    // light up, rise, or grow a ring. The type steps one magnitude brighter
    // instead — m2 -> m1 via the named --text-secondary rung.
    expect(lightRoot).toMatch(/--text-secondary:\s*#424547;/);
    expect(darkBlock).toMatch(/--text-secondary:\s*#a5abb0;/);
    const pointerFocus = stylesSource.match(
      /\.resource-browser-search-field:focus-within\s*\{[^}]*\}/s
    )?.[0];
    expect(pointerFocus).toBeTruthy();
    expect(pointerFocus).toContain("background: var(--bg-sunk)");
    expect(pointerFocus).not.toMatch(/box-shadow|border-color/);
    expect(stylesSource).toMatch(
      /\.resource-browser-search-field:focus-within svg\s*\{[^}]*color:\s*var\(--text-secondary\);/s
    );
    // WCAG 2.4.11 is about operability without a pointer: keyboard focus
    // keeps the full ring, scoped by :has(:focus-visible) to exactly the
    // users who need it.
    expect(stylesSource).toMatch(
      /\.resource-browser-search-field:has\(\.resource-browser-search:focus-visible\)\s*\{[^}]*box-shadow:/s
    );
  });

  it("removes the back card — the browser sits on the canvas", () => {
    // A browser inside a panel inside a page is the box-in-box-in-box the
    // language deletes. The shell keeps layout only.
    const shell = stylesSource.match(/\.resource-browser-shell\s*\{[^}]*\}/s)?.[0];
    expect(shell).toBeTruthy();
    expect(shell).toContain("background: transparent");
    expect(shell).toContain("border: 0");
    expect(shell).toContain("box-shadow: none");
  });

  it("de-boxes the secondary toolbar control into the same sunk family", () => {
    expect(stylesSource).toMatch(
      /\.resource-browser-filter-trigger\s*\{[^}]*border-color:\s*transparent;[^}]*background:\s*var\(--bg-sunk\);/s
    );
  });
});

describe("the composer — calm focus, and a baseline that records", () => {
  it("reserves the 2px focus line for the keyboard", () => {
    // Clicking the product's most-used text control is self-announcing (the
    // caret); the old :focus-within rule fired a 2px ink line across it on
    // every click. Pointer focus now lights nothing; :has(:focus-visible)
    // scopes the WCAG 2.4.11 indicator to keyboard users.
    expect(stylesSource).not.toMatch(
      /\.app-composer-card:focus-within\s*\{[^}]*box-shadow:\s*inset/s
    );
    expect(stylesSource).toMatch(
      /\.app-composer-card:has\(\.app-composer-textarea:focus-visible\)\s*\{[^}]*box-shadow:\s*inset 0 1px 0 var\(--text-muted\);/s
    );
  });

  it("writes the recorder trace on the baseline only while running", () => {
    // A line earns its place by recording something: while a run is live the
    // brass trace lies ON the composer's top hairline. Brass touches this
    // control at no other time.
    expect(stylesSource).toMatch(
      /\.app-composer-recorder\s*\{[^}]*color:\s*var\(--accent-live\);/s
    );
    expect(stylesSource).toMatch(
      /@media \(prefers-reduced-motion: reduce\)\s*\{\s*\.app-composer-recorder path\s*\{\s*animation:\s*none;\s*stroke-dashoffset:\s*0;/s
    );
    expect(appSource).toMatch(
      /\{activeSending \? \(\s*<RecorderTraceIcon className="app-composer-recorder" \/>\s*\) : null\}\s*<PromptInput/
    );
    // ...and OUTSIDE the card: the card clips (overflow: hidden), which once
    // beheaded the wiggle into a flat line with one dip. The wrapper anchors.
    expect(stylesSource).toMatch(
      /\.app-composer-shell \.pk-file-upload\s*\{\s*position:\s*relative;/s
    );
    // The recorder is its OWN geometry, not the compact thinking-bar glyph:
    // 96x10 viewBox rendered in a 6rem x 10px box (1:1 units, crisp stroke),
    // flat lead-in and long tail lying on the baseline, wiggle centred at
    // y=5. The compact icon letterboxed into a floating squiggle once.
    const icons = read("src/components/icons/MeridianIcons.tsx");
    expect(icons).toMatch(/RecorderTraceIcon[\s\S]{0,600}viewBox="0 0 96 10"/);
    expect(icons).toMatch(/M1 5h18l3-3\.4 3 6 3-7 3 5 3-2\.6 3 1\.8h56/);
    expect(stylesSource).toMatch(
      /\.app-composer-recorder\s*\{[^}]*width:\s*6rem;[^}]*height:\s*10px;/s
    );
  });
});

describe("the field — one impossibility, welcome stage only", () => {
  it("mounts MeridianField on the blank welcome and nowhere else", () => {
    expect(appSource).toMatch(/blank-chat-welcome">\s*<MeridianField \/>/);
    expect(appSource.match(/<MeridianField/g)).toHaveLength(1);
  });

  it("stays decoration: hidden from the tree, inert to the pointer", () => {
    expect(read("src/components/chat/MeridianField.tsx")).toContain('aria-hidden="true"');
    expect(stylesSource).toMatch(/\.meridian-field\s*\{[^}]*pointer-events:\s*none;/s);
  });

  it("is fluid, density-scaled, and cannot outgrow its container", () => {
    // Width is a share of the column, height leans on the viewport, and the
    // drawing scales star count by AREA so wide never reads sparse.
    expect(stylesSource).toMatch(/\.meridian-field\s*\{[^}]*width:\s*min\(46rem, 100%\);/s);
    expect(stylesSource).toMatch(/\.meridian-field\s*\{[^}]*height:\s*clamp\(/s);
    const fieldSource = read("src/components/chat/MeridianField.tsx");
    expect(fieldSource).toMatch(/\(width \* height\) \/ \d+/);
    // The runaway guard: a dev-server CSS hiccup once let the attribute-sized
    // canvas feedback-loop to full bleed. Layout-change gating plus an inline
    // max-width cap must both stay.
    expect(fieldSource).toContain("canvas.clientWidth === drawnWidth");
    expect(fieldSource).toContain('maxWidth: "100%"');
  });
});
