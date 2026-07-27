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

  it("keeps a visible focus state now that the border and shadow are gone", () => {
    // The textarea's own outline is none, so the rule carries focus. The inset
    // doubles the hairline to 2px WITHOUT widening the border, which would shove
    // the whole row down 1px on every focus.
    expect(stylesSource).toMatch(
      /\.app-composer-shell \.app-composer-card:focus-within\s*\{[^}]*border-top-color:\s*var\(--text-muted\);[^}]*box-shadow:\s*inset 0 1px 0 var\(--text-muted\);/s
    );
    expect(stylesSource).not.toMatch(
      /\.app-composer-shell \.app-composer-card:focus-within\s*\{[^}]*border-top-width:/s
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
    const desktopBlock = stylesSource.match(
      /@media \(min-width:\s*641px\)\s*\{[\s\S]*?\n\}\n/
    )?.[0];
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
