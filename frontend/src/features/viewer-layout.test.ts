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
});
