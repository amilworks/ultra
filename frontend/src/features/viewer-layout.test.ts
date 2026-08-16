import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const stylesSource = readFileSync(path.join(process.cwd(), "src/styles.css"), "utf8");

describe("scientific viewer responsive layout", () => {
  it("keeps the direct-image WebGL canvas out of the mobile height calculation", () => {
    // Three.js updates the backing canvas dimensions after every ResizeObserver
    // measurement. The canvas must not be a normal-flow child, or its intrinsic
    // backing-store height can feed back into the 2D shell and grow the Lens viewer.
    expect(stylesSource).toMatch(
      /\.viewer-direct-image-canvas\s*\{[^}]*position:\s*absolute;[^}]*inset:\s*0;/s
    );
  });

  it("fits the 2D canvas to the workspace instead of covering following controls", () => {
    expect(stylesSource).toMatch(
      /\.viewer-workspace-surface-2d\s+\.viewer-canvas-shell-2d\s*\{[^}]*height:\s*100%;[^}]*min-height:\s*0;/s,
    );
  });

  it("keeps high-dimensional channel catalogs in a bounded virtual viewport", () => {
    expect(stylesSource).toMatch(
      /\.viewer-channel-browser-dialog\s*\{[^}]*max-height:\s*calc\(100dvh - 2rem\);[^}]*background:\s*var\(--background\);/s,
    );
    expect(stylesSource).toMatch(
      /\.viewer-channel-browser-viewport\s*\{[^}]*overflow-y:\s*auto;[^}]*scrollbar-width:\s*none;/s,
    );
    expect(stylesSource).toMatch(
      /\.viewer-channel-browser-row\s*\{[^}]*position:\s*absolute;[^}]*height:\s*var\(--viewer-channel-row-height\);/s,
    );
  });
});
