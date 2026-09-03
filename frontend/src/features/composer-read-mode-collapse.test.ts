import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

/* Read mode: scrolled away from the end of a long answer, the composer closes
   to the bare bar and an instruction — at every width — and comes back on
   focus, at the bottom, or when a run starts. The bar never changes height. */
const read = (relativePath: string): string =>
  readFileSync(path.join(process.cwd(), relativePath), "utf8");

const appSource = read("src/App.tsx");
const composerSource = read("src/components/composer/Composer.tsx");
const modelSource = read("src/components/composer/composerModel.ts");
const stylesSource = read("src/styles.css");

const rule = (selector: string): string => {
  const start = stylesSource.indexOf(`${selector} {`);
  expect(start, `missing rule ${selector}`).toBeGreaterThan(-1);
  return stylesSource.slice(start, stylesSource.indexOf("}", start));
};

describe("composer read-mode collapse", () => {
  it("collapses at every width, not just on phones", () => {
    const start = stylesSource.indexOf('.composer[data-read-mode="true"] .composer-attach,');
    expect(start).toBeGreaterThan(-1);
    const before = stylesSource.slice(0, start);
    const lastMedia = before.lastIndexOf("@media");
    const lastClose = before.lastIndexOf("\n}\n");
    expect(lastMedia === -1 || lastClose > lastMedia).toBe(true);
  });

  it("drives off scrolled-AWAY, not actively-scrolling, through one explicit prop", () => {
    expect(appSource).toMatch(/onScrolledAwayChange=\{setComposerScrolledAway\}/);
    expect(appSource).toMatch(/readMode=\{composerScrolledAway\}/);
    expect(composerSource).toMatch(/const collapsed = readMode && !running && !focused;/);
    expect(composerSource).toMatch(/data-read-mode=\{collapsed \? "true" : undefined\}/);
  });

  it("hides the controls, and hides them properly", () => {
    const start = stylesSource.indexOf('.composer[data-read-mode="true"] .composer-attach,');
    const block = stylesSource.slice(start, stylesSource.indexOf("}", start));
    for (const control of [".composer-attach", ".composer-tag", ".composer-end"]) {
      expect(block).toContain(control);
    }
    expect(block).toMatch(/visibility:\s*hidden;/);
    expect(block).toMatch(/pointer-events:\s*none;/);
  });

  it("closes the text block and keeps the bar at its one height", () => {
    const closed = rule('.composer[data-stage="rest"] .composer-text,\n.composer[data-read-mode="true"] .composer-text');
    expect(closed).toMatch(/max-height:\s*0;/);
    // No strip: nothing in the composer's own section restates the bar's height.
    const section = stylesSource.slice(
      stylesSource.indexOf("The composer: one bar."),
      stylesSource.indexOf(".welcome-starters {")
    );
    expect(section).not.toMatch(/min-height:\s*(1\.8|2\.55)rem/);
    // The closed text block sets max-height; no read-mode rule sets a height.
    expect(section).not.toMatch(/data-read-mode="true"\][^{]*\{[^}]*\n\s*height:/);
  });

  it("restores the full composer on focus and during a run", () => {
    expect(composerSource).toMatch(/const collapsed = readMode && !running && !focused;/);
    expect(modelSource).toMatch(/if \(inputs\.readMode && !inputs\.running\) \{\s*return "Just start typing";/);
  });

  it("swaps the status to the instruction and keeps it legible", () => {
    expect(composerSource).toMatch(/\} else if \(collapsed\) \{\s*status = placeholder;/);
    expect(rule('.composer[data-read-mode="true"] .composer-status')).toMatch(/color:\s*var\(--text-muted\);/);
  });
});
