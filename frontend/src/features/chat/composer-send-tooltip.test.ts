import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const readSource = (relativePath: string): string =>
  readFileSync(path.join(process.cwd(), relativePath), "utf8");

const appSource = readSource("src/App.tsx");
const promptInputSource = readSource("src/components/prompt-kit/prompt-input.tsx");
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
  it("shares the exact existing disabled predicate across one real send button and tooltip", () => {
    expect(appSource).toMatch(
      /const composerSubmitDisabled =\s*!activeConversationHydrated \|\|\s*!activePrompt\.trim\(\) \|\|\s*slashMenuOpen;/
    );
    expect(appSource.match(/disabled=\{composerSubmitDisabled\}/g)).toHaveLength(2);
    expect(appSource.match(/className="app-composer-submit-button/g)).toHaveLength(1);

    const sendButton = appSource.match(
      /<Button\s+size="icon"\s+type="submit"[\s\S]*?className="app-composer-submit-button size-11 rounded-full sm:size-10"[\s\S]*?<ArrowUp size=\{18\} \/>\s*<\/Button>/
    )?.[0];
    expect(sendButton).toBeDefined();
    expect(sendButton).toContain('aria-label="Send prompt"');
    expect(sendButton).toContain("disabled={composerSubmitDisabled}");
    expect(sendButton).not.toContain("title=");
  });

  it("wraps only enabled send with the existing asChild tooltip primitive", () => {
    expect(appSource).toMatch(
      /\{activeSending \? \(\s*<Button[\s\S]*?aria-label="Stop response"[\s\S]*?title="Stop response"[\s\S]*?\) : \(\s*<PromptInputAction/
    );
    expect(appSource).toMatch(
      /<PromptInputAction[\s\S]*?disabled=\{composerSubmitDisabled\}[\s\S]*?side="top"[\s\S]*?sideOffset=\{8\}[\s\S]*?delayDuration=\{350\}[\s\S]*?className="app-composer-submit-tooltip"/
    );
    expect(appSource).not.toContain("disableHoverableContent");
    expect(promptInputSource).toMatch(/<TooltipTrigger\s+asChild/);
    expect(promptInputSource).toContain("if (disabled)");
    expect(promptInputSource).toMatch(/return <>\{children\}<\/>;/);
    expect(promptInputSource).toContain("disabled={contextDisabled}");
    expect(promptInputSource).toContain(
      "<TooltipContent side={side} sideOffset={sideOffset} className={className}>"
    );
  });

  it("pins visible copy, hidden keyboard guidance, and accessible button naming", () => {
    expect(appSource).toContain(">Send prompt</span>");
    expect(appSource).toMatch(
      /className="app-composer-submit-tooltip-key"[\s\S]*?aria-hidden="true"\s*>\s*↵\s*<\/span>/
    );
    expect(appSource).toMatch(
      /className="sr-only">\s*Press Enter to send\. Shift\+Enter starts a new line\.\s*<\/span>/
    );
    expect(appSource).toContain('aria-label="Send prompt"');
    expect(appSource).not.toContain('title="Send message"');
  });

  it("preserves form-submit and plain Enter send paths", () => {
    expect(appSource).toMatch(
      /<PromptInput[\s\S]*?onSubmit=\{\(\) => \{\s*void handleSubmit\(\);\s*\}\}/
    );
    expect(appSource).toMatch(
      /event\.key === "Enter" &&\s*!event\.shiftKey &&\s*!event\.metaKey &&\s*!event\.ctrlKey &&\s*!event\.altKey &&\s*!event\.nativeEvent\.isComposing[\s\S]*?event\.preventDefault\(\);\s*void handleSubmit\(\);/
    );
  });

  it("uses a compact neutral bubble without device-hiding the tooltip subtree", () => {
    const tooltipRules = cssBlock(".app-composer-submit-tooltip");
    expect(tooltipRules).toMatch(/border-radius:\s*0\.625rem;/);
    expect(tooltipRules).toMatch(/padding:\s*0\.5rem 0\.625rem;/);
    expect(tooltipRules).toMatch(/font-size:\s*0\.75rem;/);
    expect(tooltipRules).toMatch(/line-height:\s*1rem;/);
    expect(tooltipRules).toMatch(/font-weight:\s*500;/);
    expect(tooltipRules).toMatch(/white-space:\s*nowrap;/);
    expect(tooltipRules).toMatch(/box-shadow:/);
    expect(tooltipRules).not.toMatch(/(?:background|color):/);
    expect(tooltipPrimitiveSource).toContain("bg-foreground text-background");

    const rowRules = cssBlock(".app-composer-submit-tooltip-row");
    expect(rowRules).toMatch(/gap:\s*0\.625rem;/);
    const keyRules = cssBlock(".app-composer-submit-tooltip-key");
    expect(keyRules).toMatch(/color:\s*currentColor;/);
    expect(keyRules).toMatch(/opacity:\s*0\.6;/);
    expect(keyRules).not.toMatch(/border:/);

    expect(stylesSource).not.toMatch(
      /@media \(hover: none\), \(pointer: coarse\)\s*\{\s*\.app-composer-submit-tooltip\s*\{/
    );
    expect(stylesSource).not.toMatch(
      /\.app-composer-submit-tooltip\s*\{[^}]*display:\s*none;/s
    );
    expect(tooltipRules).not.toMatch(/blue|--ultra-launch-signal|--chart-/);
  });
});
