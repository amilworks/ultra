import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const appSource = readFileSync(path.join(process.cwd(), "src/App.tsx"), "utf8");
const stylesSource = readFileSync(path.join(process.cwd(), "src/styles.css"), "utf8");

describe("mobile composer layout", () => {
  it("wires scroll-away state to a phone-only compact composer", () => {
    expect(appSource).toMatch(/onScrolledAwayChange\?: \(away: boolean\) => void;/);
    expect(appSource).toMatch(/const \[composerScrolledAway,\s*setComposerScrolledAway\]/);
    expect(appSource).toMatch(/onScrolledAwayChange=\{setComposerScrolledAway\}/);
    expect(appSource).toMatch(
      /data-composer-compact=\{\s*isPhoneView && composerScrolledAway && !activeSending \? "true" : undefined\s*\}/s
    );
  });

  it("collapses composer actions only on phones and expands on focus", () => {
    expect(stylesSource).toMatch(
      /@media \(max-width:\s*640px\)\s*\{[\s\S]*\.app-composer-shell\[data-composer-compact="true"\]:not\(:focus-within\) \.app-composer-textarea\s*\{[^}]*min-height:\s*2\.55rem;[\s\S]*\.app-composer-shell\[data-composer-compact="true"\]:not\(:focus-within\) \.app-composer-actions\s*\{[^}]*max-height:\s*0;[^}]*pointer-events:\s*none;/s
    );
    expect(stylesSource).toMatch(
      /@media \(max-width:\s*640px\) and \(prefers-reduced-motion:\s*reduce\)\s*\{[\s\S]*\.app-composer-shell \.app-composer-textarea,[\s\S]*\.app-composer-shell \.app-composer-actions\s*\{[^}]*transition:\s*none;/s
    );
  });
});
