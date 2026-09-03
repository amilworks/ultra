import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

/* Read mode: scrolled away from the end of a long answer, the composer drops to
   a strip and a hint — at every width — and comes back on focus, at the bottom,
   or when a run starts. The composer decides (one explicit state), the sheet
   draws it. */
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
    // The strip rule is top-level: no media query gates it.
    const start = stylesSource.indexOf('.composer[data-read-mode="true"] .composer-line {');
    expect(start).toBeGreaterThan(-1);
    const before = stylesSource.slice(0, start);
    const lastMedia = before.lastIndexOf("@media");
    const lastClose = before.lastIndexOf("\n}\n");
    expect(lastMedia === -1 || lastClose > lastMedia).toBe(true);
    expect(rule('.composer[data-read-mode="true"] .composer-line')).toMatch(/min-height:\s*1\.8rem;/);
  });

  it("drives off scrolled-AWAY, not actively-scrolling, through one explicit prop", () => {
    expect(appSource).toMatch(/onScrolledAwayChange=\{setComposerScrolledAway\}/);
    expect(appSource).toMatch(/readMode=\{composerScrolledAway\}/);
    expect(composerSource).toMatch(/const collapsed = readMode && !running && !focused;/);
    expect(composerSource).toMatch(/data-read-mode=\{collapsed \? "true" : undefined\}/);
  });

  it("hides the controls, and hides them properly", () => {
    // visibility, not just opacity: a 0-opacity button is still tabbable and
    // still announced. Exactly one set of composer controls is ever in the tree.
    const start = stylesSource.indexOf('.composer[data-read-mode="true"] .composer-attach,');
    expect(start).toBeGreaterThan(-1);
    const block = stylesSource.slice(start, stylesSource.indexOf("}", start));
    for (const control of [".composer-attach", ".composer-end", ".composer-prefix", ".composer-whisper"]) {
      expect(block).toContain(control);
    }
    expect(block).toMatch(/visibility:\s*hidden;/);
    expect(block).toMatch(/pointer-events:\s*none;/);
  });

  it("halves the strip and takes the run-meta line's type", () => {
    const field = rule('.composer[data-read-mode="true"] .composer-field');
    expect(field).toMatch(/max-height:\s*1\.8rem;/);
    expect(field).toMatch(/font-size:\s*0\.75rem;/);
    expect(field).toMatch(/line-height:\s*20px;/);
    expect(field).toMatch(/padding-top:\s*calc\(\(1\.8rem - 20px\) \/ 2\);/);
  });

  it("keeps phones on a tappable strip rather than the 30px desktop one", () => {
    const phone = stylesSource.slice(
      stylesSource.indexOf("/* Phones keep a tappable strip"),
      stylesSource.indexOf("/* The @ picker.")
    );
    expect(phone).toMatch(/@media \(max-width:\s*640px\)/);
    expect(phone).toMatch(/min-height:\s*2\.55rem;/);
    expect(phone).not.toMatch(/min-height:\s*1\.8rem;/);
  });

  it("restores the full composer on focus and during a run", () => {
    expect(composerSource).toMatch(/const collapsed = readMode && !running && !focused;/);
    expect(modelSource).toMatch(/if \(inputs\.readMode && !inputs\.running\) \{\s*return "Just start typing";/);
  });

  it("keeps the hint legible while collapsed", () => {
    expect(stylesSource).toMatch(
      /\.composer\[data-read-mode="true"\] \.composer-placeholder,\s*\.composer\[data-read-mode="true"\] textarea\.composer-editor::placeholder\s*\{[^}]*color:\s*var\(--text-muted\);/s
    );
  });
});
