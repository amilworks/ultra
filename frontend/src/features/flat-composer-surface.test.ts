/// <reference types="node" />

import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const stylesSource = readFileSync(path.join(process.cwd(), "src/styles.css"), "utf8");

describe("flat composer surface", () => {
  it("draws the composer as one hairline instead of a floating pill", () => {
    expect(stylesSource).toMatch(
      /\.app-composer-shell \.app-composer-card\s*\{[^}]*border:\s*0;[^}]*border-top:\s*1px solid color-mix\(in oklab, var\(--line\) 88%, transparent\);[^}]*border-radius:\s*0;[^}]*background:\s*transparent;/s
    );
    // No visible boundary, so the whole region has to read as the field.
    expect(stylesSource).toMatch(
      /\.app-composer-shell \.app-composer-card\s*\{[^}]*cursor:\s*text;/s
    );
  });

  it("scopes the flat surface through the shell so .pk-prompt-input cannot win", () => {
    // .pk-prompt-input sets border + a 22.4px radius + its own panel background
    // ~4300 lines later at equal specificity. A bare `.app-composer-card {` rule
    // loses to it, and the background is the dangerous one — it paints a visible
    // panel block in dark theme instead of reading as the page edge.
    expect(stylesSource).toMatch(/^\.pk-prompt-input\s*\{[^}]*border-radius:/ms);
    expect(stylesSource).not.toMatch(/^\.app-composer-card\s*\{/m);
    expect(stylesSource).not.toMatch(/^\.app-composer-card:focus-within\s*\{/m);
  });

  it("keeps the 2px focus state for the keyboard, and calm for the pointer", () => {
    // SUPERSESSION, on purpose: this test previously pinned the 2px inset on
    // :focus-within — every click on the product's most-used control fired
    // the loudest focus indicator in the app. Meridian's calm-focus rule
    // (established on the resource search well) scopes the indicator with
    // :has(:focus-visible): keyboard users keep the full WCAG 2.4.11
    // treatment, pointer users get the caret, which already announces the
    // activation. The indicator itself is unchanged — the inset doubles the
    // hairline to 2px WITHOUT widening the border, which would shove the
    // whole row down 1px.
    expect(stylesSource).toMatch(
      /\.app-composer-shell \.app-composer-card:has\(\.app-composer-textarea:focus-visible\)\s*\{[^}]*border-top-color:\s*var\(--text-muted\);[^}]*box-shadow:\s*inset 0 1px 0 var\(--text-muted\);/s
    );
    expect(stylesSource).not.toMatch(
      /\.app-composer-shell \.app-composer-card:focus-within\s*\{/s
    );
  });

  it("keeps the drop affordance inside the rule rather than ringing the dock", () => {
    // The old outer `0 0 0 3px` ring traced the pill; on a flat full-width
    // surface it drew a rectangle around the whole dock.
    expect(stylesSource).toMatch(
      /\.pk-file-upload-drag \.app-composer-card\s*\{[^}]*border-top-color:[^}]*box-shadow:\s*inset 0 2px 0/s
    );
    expect(stylesSource).not.toMatch(
      /\.pk-file-upload-drag \.app-composer-card\s*\{[^}]*box-shadow:\s*0 0 0 3px/s
    );
  });

  it("centres the resting row on one optical line, desktop only", () => {
    // The one-line resting row only exists at >=641px (data-composer-slim is
    // itself a min-width: 641px feature). Forcing a centred single-line row on
    // the phone clipped the placeholder and scattered the controls.
    // Pick the min-width:641px block that actually carries the resting-row
    // rules, not simply the first one in the file — there is now more than one
    // (the read-mode collapse added its own), and matching the first made this
    // assert against whichever block happened to come earlier.
    const desktopBlock = (stylesSource.match(/@media \(min-width:\s*641px\)\s*\{[\s\S]*?\n\}\n/g) ?? [])
      .find((block) => block.includes(".app-composer-shell .app-composer-card-body"));
    expect(desktopBlock).toBeTruthy();
    expect(desktopBlock).toMatch(
      /\.app-composer-shell \.app-composer-card-body\s*\{[^}]*justify-content:\s*center;[^}]*min-height:\s*4\.75rem;/s
    );
    // COLUMN flex: justify-content is the vertical axis here. align-items would
    // only centre it horizontally, and would shrink it off full width.
    expect(stylesSource).toMatch(
      /\.app-composer-card-body\s*\{[^}]*flex-direction:\s*column;/s
    );
    expect(desktopBlock).not.toMatch(
      /\.app-composer-shell \.app-composer-card-body\s*\{[^}]*align-items:\s*center;/s
    );
  });

  it("keeps the slim send anchor on an animatable offset", () => {
    // `top: 50%; bottom: auto` centres it too, but `auto` is not animatable, so
    // the button would jump instead of gliding between the two anchors.
    expect(stylesSource).toMatch(
      /\.app-composer-shell\[data-composer-slim="true"\] \.app-composer-actions-end\s*\{[^}]*right:\s*1rem;[^}]*bottom:\s*1\.125rem;/s
    );
    expect(stylesSource).not.toMatch(
      /\.app-composer-shell\[data-composer-slim="true"\] \.app-composer-actions-end\s*\{[^}]*bottom:\s*auto;/s
    );
  });

  it("leaves no pill geometry or dead shadow tokens behind", () => {
    expect(stylesSource).not.toContain("--composer-shell-shadow");
    expect(stylesSource).not.toMatch(/\.app-composer-card\s*\{[^}]*border-radius:\s*999px;/s);
    expect(stylesSource).not.toMatch(/\.app-composer-card\s*\{[^}]*border-radius:\s*1\.55rem;/s);
    expect(stylesSource).not.toMatch(
      /\.app-composer-shell \.app-composer-card\s*\{[^}]*border-radius:\s*calc\(var\(--radius\)/s
    );
  });
});
