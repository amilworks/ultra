import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const stylesSource = readFileSync(path.join(process.cwd(), "src/styles.css"), "utf8");
const composerSource = readFileSync(
  path.join(process.cwd(), "src/components/composer/Composer.tsx"),
  "utf8"
);

const rule = (selector: string): string => {
  const start = stylesSource.indexOf(`${selector} {`);
  expect(start, `missing rule ${selector}`).toBeGreaterThan(-1);
  return stylesSource.slice(start, stylesSource.indexOf("}", start));
};

describe("flat composer surface", () => {
  it("draws the composer as one hairline instead of a floating pill", () => {
    const surface = rule(".composer-surface");
    expect(surface).toMatch(/border-top:\s*1px solid color-mix\(in oklab, var\(--line\) 88%, transparent\);/);
    expect(surface).toMatch(/background:\s*transparent;/);
    expect(surface).not.toMatch(/border-radius/);
    // No resting shadow — box-shadow appears only as a transition property.
    expect(surface).not.toMatch(/\n\s*box-shadow:/);
    expect(stylesSource).not.toContain("--composer-shell-shadow");
  });

  it("keeps the 2px focus state for the keyboard, and calm for the pointer", () => {
    expect(stylesSource).not.toMatch(/\.composer-surface:focus-within\s*\{/);
    const focus = rule(".composer-surface:has(.composer-editor:focus-visible)");
    expect(focus).toMatch(/border-top-color:\s*var\(--text-muted\);/);
    expect(focus).toMatch(/box-shadow:\s*inset 0 1px 0 var\(--text-muted\);/);
  });

  it("keeps the drop affordance inside the rule rather than ringing the dock", () => {
    const drag = rule(".pk-file-upload-drag .composer-surface");
    expect(drag).toMatch(/border-top-color:\s*color-mix\(in oklab, var\(--text-main\) 40%, transparent\);/);
    expect(drag).toMatch(/box-shadow:\s*inset 0 2px 0/);
    expect(drag).not.toMatch(/outline|border-radius/);
  });

  it("centres a one-line brief on the controls' axis, and grows it upward", () => {
    // Controls sit on the LAST text line: a one-line brief centres everything
    // on one optical line; a long one grows above the attach and send buttons.
    const line = rule(".composer-line");
    expect(line).toMatch(/align-items:\s*flex-end;/);
    expect(line).toMatch(/min-height:\s*var\(--composer-control\);/);
    const field = rule(".composer-field");
    expect(field).toMatch(
      /padding:\s*calc\(\(var\(--composer-control\) - 1em \* var\(--composer-line-height\)\) \/ 2\)/
    );
    expect(field).toMatch(/max-height:\s*240px;/);
    expect(field).toMatch(/overflow-y:\s*auto;/);
  });

  it("lets the chips lead only the first line", () => {
    // The prefix is a float, so the brief's first line starts after the chips
    // and every later line runs the full width beneath them.
    const prefix = rule(".composer-prefix");
    expect(prefix).toMatch(/float:\s*left;/);
    // The field scrolls, never the editor: an overflow on the editor would
    // start a new formatting context and stop the float from indenting it.
    expect(rule(".composer-editor")).not.toMatch(/overflow(-x|-y)?:/);
    expect(composerSource).toMatch(/<span ref=\{prefixRef\} className="composer-prefix">/);
  });

  it("keeps the send button on the ladder and the whole region as the field", () => {
    expect(rule(".composer-send")).toMatch(/background:\s*var\(--primary\);/);
    expect(rule(".composer-surface")).toMatch(/cursor:\s*text;/);
    expect(composerSource).toMatch(/onMouseDown=\{handleSurfaceMouseDown\}/);
  });
});
