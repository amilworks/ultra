import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const readSource = (relativePath: string): string =>
  readFileSync(path.join(process.cwd(), relativePath), "utf8");

const appSource = readSource("src/App.tsx");
const composerSource = readSource("src/components/composer/Composer.tsx");
const tooltipSource = readSource("src/components/composer/ComposerTooltip.tsx");
const tooltipPrimitiveSource = readSource("src/components/ui/tooltip.tsx");
const stylesSource = readSource("src/styles.css");

const cssBlock = (selector: string): string => {
  const match = new RegExp(`\\${selector}\\s*\\{([^}]*)\\}`, "s").exec(stylesSource);
  if (!match) {
    throw new Error(`Missing CSS selector ${selector}`);
  }
  return match[1];
};

describe("composer send tooltip contract", () => {
  it("shares the exact disabled predicate across one real send button and its tooltip", () => {
    expect(appSource).toMatch(/submitDisabled=\{composerSubmitDisabled\}/);
    const sendStart = composerSource.indexOf('aria-label="Send prompt"');
    expect(sendStart).toBeGreaterThan(-1);
    const sendButton = composerSource.slice(
      composerSource.lastIndexOf("<Button", sendStart),
      composerSource.indexOf("</Button>", sendStart)
    );
    expect(sendButton).toContain('type="submit"');
    expect(sendButton).toContain("disabled={submitDisabled}");
    expect(sendButton).not.toContain("title=");
    expect(composerSource).toMatch(/<ComposerTooltip\s+disabled=\{submitDisabled\}\s+className="app-composer-submit-tooltip"/);
  });

  it("wraps only an enabled send with the asChild tooltip primitive", () => {
    expect(tooltipSource).toMatch(/if \(disabled\) \{\s*return <>\{children\}<\/>;/);
    expect(tooltipSource).toMatch(/<TooltipTrigger\s+asChild/);
    expect(tooltipSource).toMatch(/delayDuration=\{350\}/);
    expect(tooltipPrimitiveSource).toContain("TooltipTrigger");
  });

  it("pins visible copy, hidden keyboard guidance, and accessible button naming", () => {
    expect(composerSource).toContain("<span>Send prompt</span>");
    expect(composerSource).toMatch(
      /<span className="app-composer-submit-tooltip-key" aria-hidden="true">\s*↵\s*<\/span>/
    );
    expect(composerSource).toContain("Press Enter to send. Shift+Enter starts a new line.");
    expect(composerSource).toContain('aria-label="Send prompt"');
    expect(composerSource).not.toContain('title="Send message"');
  });

  it("preserves form-submit and plain Enter send paths", () => {
    expect(composerSource).toMatch(/<form\s+className="composer"/);
    expect(composerSource).toMatch(/onSubmit=\{\(event\) => \{\s*event\.preventDefault\(\);\s*if \(!hydrated \|\| running \|\| submitDisabled\) \{\s*return;\s*\}\s*props\.onSubmit\(\);/);
    expect(composerSource).toMatch(/if \(!state\.submitDisabled\) \{\s*state\.onSubmit\(\);\s*\}\s*return true;\s*\}, \[\]\);/);
  });

  it("uses a compact neutral bubble without device-hiding the tooltip subtree", () => {
    const block = cssBlock(".app-composer-submit-tooltip");
    expect(block).not.toMatch(/display:\s*none/);
    expect(cssBlock(".app-composer-submit-tooltip-row")).toMatch(/display:\s*(inline-)?flex/);
  });
});
