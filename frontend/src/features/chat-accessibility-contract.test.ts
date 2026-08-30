import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const stylesSource = readFileSync(path.join(process.cwd(), "src/styles.css"), "utf8");
const appSource = readFileSync(path.join(process.cwd(), "src/App.tsx"), "utf8");

describe("chat accessibility + reading-flow contract", () => {
  it("keeps exactly one <main> landmark in the app shell", () => {
    // SidebarInset (components/ui/sidebar.tsx) already renders <main>. The app
    // shell nested a second one inside it, so assistive tech announced two main
    // regions and "skip to main content" was ambiguous.
    expect(appSource).toMatch(
      /<SidebarInset>[\s\S]{0,600}?<div\s+ref=\{setMainShellElement\}/
    );
    expect(appSource).not.toMatch(
      /<SidebarInset>[\s\S]{0,600}?<main\s+ref=\{setMainShellElement\}/
    );
  });

  it("gives every view a single screen-reader document heading", () => {
    // Without this the outline started at an H2 inside model prose, so a
    // heading rotor opened onto whatever the model happened to write.
    expect(appSource).toMatch(/const documentHeading =/);
    expect(appSource).toMatch(/<h1 className="sr-only">\{documentHeading\}<\/h1>/);
  });

  it("labels each transcript turn with its speaker", () => {
    // The transcript is one role="log" region; unlabeled turn divs read as
    // continuous prose with no way to tell who was speaking.
    expect(appSource).toMatch(/role="article"\s+aria-label="You said"/);
    expect(appSource).toMatch(/role="article"\s+aria-label="Ultra said"/);
  });

  it("contains scroll inside reasoning traces so gestures do not chain", () => {
    // Measured 19,029px of content in a 352px window sitting in the reading
    // path: an un-contained trace swallows the gesture, then hands the
    // momentum back to the transcript mid-scroll.
    expect(stylesSource).toMatch(
      /\.reasoning-trace-body\s*\{[^}]*overscroll-behavior:\s*contain;/s
    );
  });
});

describe("sidebar keyboard focus", () => {
  it("restores a visible focus ring on the flat sidebar controls", () => {
    // These controls ship Tailwind's `outline-hidden` + `focus-visible:ring-2`,
    // and Tailwind paints that ring through box-shadow — which every flat
    // sidebar control then erases with a resting `box-shadow: none`. Measured
    // under real keyboard focus: :focus-visible matched and --tw-ring-shadow
    // resolved to a real 2px ring while the composed box-shadow stayed none,
    // leaving the conversation list with no visible focus at all (WCAG 2.4.7).
    // Stated as an outline so a resting box-shadow can never erase it again.
    expect(stylesSource).toMatch(
      /\.app-history-button:focus-visible,[\s\S]{0,400}?\{[^}]*outline:\s*2px solid var\(--sidebar-ring\);[^}]*outline-offset:\s*-2px;/s
    );
  });

  it("keeps the resting sidebar flat", () => {
    // The fix must be focus-visible-only: a pointer click never triggers it.
    expect(stylesSource).toMatch(/\.app-history-button\s*\{[^}]*box-shadow:\s*none;/s);
  });
});

describe("display-math copy affordance", () => {
  it("reserves a gutter so the button never sits on the equation", () => {
    // Measured 27x13px of overlap on every display block at both widths —
    // on touch the button is permanently visible, so it covered glyphs
    // permanently. Symmetric padding keeps display math optically centered.
    expect(stylesSource).toMatch(
      /\.pk-math-display-shell \.katex-display,[\s\S]{0,120}?\.katex-display\s*\{[^}]*padding-inline:\s*2\.1rem;/s
    );
  });

  it("grows the hit area to the house touch minimum without growing the ink", () => {
    expect(stylesSource).toMatch(
      /\.pk-math-copy::after\s*\{[^}]*position:\s*absolute;[^}]*inset:\s*-0\.53rem;/s
    );
    // The visual box stays a ghost-sized 1.7rem.
    expect(stylesSource).toMatch(/\.pk-math-copy\s*\{[^}]*width:\s*1\.7rem;/s);
  });
});
