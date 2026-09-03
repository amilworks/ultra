import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

/* The composer is one bar: a sunk well at rest, a raised card while
   composing, and the same 3rem bar under both. Every control centres on the
   bar's axis; the text block above it never shares a row with a control. */
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

describe("the composer bar", () => {
  it("rests as a sunk well and rises into a raised card — never a bordered box at rest", () => {
    const card = rule(".composer-card");
    expect(card).toMatch(/border:\s*1px solid transparent;/);
    expect(card).toMatch(/border-radius:\s*var\(--radius\);/);
    expect(card).toMatch(/background:\s*var\(--bg-sunk\);/);
    expect(card).not.toMatch(/\n\s*box-shadow:/);
    const raised = rule('.composer[data-stage="composing"] .composer-card,\n.composer[data-stage="running"] .composer-card');
    expect(raised).toMatch(/background:\s*var\(--bg-raised\);/);
    expect(raised).toMatch(/border-color:\s*color-mix\(in oklab, var\(--line\) 88%, transparent\);/);
    expect(raised).toMatch(/box-shadow:\s*var\(--shadow-raised\);/);
    expect(stylesSource).not.toContain("--composer-shell-shadow");
  });

  it("keeps one bar as the only vertical unit, identical in every state", () => {
    expect(rule(".composer")).toMatch(/--composer-bar:\s*3rem;/);
    const bar = rule(".composer-bar");
    expect(bar).toMatch(/align-items:\s*center;/);
    expect(bar).toMatch(/height:\s*var\(--composer-bar\);/);
    // Controls inset symmetrically: the same distance from the edge as the text.
    expect(bar).toMatch(/padding:\s*0 calc\(\(var\(--composer-bar\) - var\(--composer-control\)\) \/ 2\);/);
    expect(rule(".composer-control")).toMatch(/height:\s*var\(--composer-control\);/);
    // No control ever bottom-anchors to a growing text block.
    expect(stylesSource).not.toMatch(/\.composer-bar\s*\{[^}]*align-items:\s*flex-end/s);
  });

  it("puts the text above the bar with its own padding, and closes it at rest", () => {
    const text = rule(".composer-text");
    expect(text).toMatch(/max-height:\s*240px;/);
    expect(text).toMatch(/overflow-y:\s*auto;/);
    expect(text).toMatch(/padding:\s*0\.7rem 1rem 0\.35rem;/);
    expect(text).toMatch(/font-size:\s*var\(--font-size-body\);/);
    const closed = rule('.composer[data-stage="rest"] .composer-text,\n.composer[data-read-mode="true"] .composer-text');
    expect(closed).toMatch(/max-height:\s*0;/);
    expect(closed).toMatch(/opacity:\s*0;/);
    expect(composerSource).toMatch(/<div className="composer-text">/);
    expect(composerSource).toMatch(/<div className="composer-bar">/);
    // The bar renders after the text block: text above, controls beneath.
    expect(composerSource.indexOf('<div className="composer-text">')).toBeLessThan(
      composerSource.indexOf('<div className="composer-bar">')
    );
  });

  it("keeps the 2px focus line for the keyboard, and calm for the pointer", () => {
    expect(stylesSource).not.toMatch(/\.composer-card:focus-within\s*\{/);
    const focus = rule(".composer-card:has(.composer-editor:focus-visible)");
    expect(focus).toMatch(/border-color:\s*var\(--text-muted\);/);
    expect(focus).toMatch(/inset 0 0 0 1px var\(--text-muted\)/);
  });

  it("keeps the drop affordance inside the card rather than ringing the dock", () => {
    const drag = rule(".pk-file-upload-drag .composer-card");
    expect(drag).toMatch(/border-color:\s*color-mix\(in oklab, var\(--text-main\) 40%, transparent\);/);
    expect(drag).toMatch(/box-shadow:\s*inset 0 0 0 2px/);
    expect(drag).not.toMatch(/outline/);
  });

  it("carries the workflow and the mode as mono tags in the bar, and the send on the ladder", () => {
    const tag = rule(".composer-tag");
    expect(tag).toMatch(/font-family:\s*var\(--font-mono\);/);
    expect(tag).toMatch(/text-transform:\s*uppercase;/);
    expect(tag).toMatch(/height:\s*var\(--composer-tag\);/);
    expect(composerSource).toMatch(/data-testid="composer-mode-tag"/);
    expect(composerSource).toMatch(/data-testid="composer-workflow-tag"/);
    expect(rule(".composer-send")).toMatch(/background:\s*var\(--primary\);/);
    expect(rule(".composer-card")).toMatch(/cursor:\s*text;/);
    expect(composerSource).toMatch(/onMouseDown=\{handleCardMouseDown\}/);
  });
});
