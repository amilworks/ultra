/// <reference types="node" />

import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const stylesSource = readFileSync(path.join(process.cwd(), "src/styles.css"), "utf8");
const appSource = readFileSync(path.join(process.cwd(), "src/App.tsx"), "utf8");

const desktopCompactBlock = stylesSource.slice(
  stylesSource.indexOf("Desktop half of the read-mode collapse"),
  stylesSource.indexOf("Phone-only: collapse the composer to a slim line")
);
// Bounded to the media block itself. Slicing to end-of-file swept in unrelated
// later rules and made the "phones keep their own sizing" guard pass for the
// wrong reason.
const phoneCompactBlock = (() => {
  const start = stylesSource.indexOf("Phone-only: collapse the composer to a slim line");
  const end = stylesSource.indexOf("\n}\n", stylesSource.indexOf("@media (max-width: 640px)", start));
  return stylesSource.slice(start, end);
})();

describe("composer read-mode collapse", () => {
  it("collapses at every width, not just on phones", () => {
    expect(appSource).toMatch(
      /data-composer-compact=\{\s*composerScrolledAway && !activeSending \? "true" : undefined\s*\}/s
    );
    expect(appSource).not.toMatch(
      /data-composer-compact=\{\s*isPhoneView && composerScrolledAway/s
    );
  });

  it("drives off scrolled-AWAY, not actively-scrolling", () => {
    // An is-scrolling trigger re-expands the composer the instant you stop to
    // read a paragraph — exactly when you want it gone. The scrolled-away signal
    // already carries hysteresis (160px out, 48px back) so it will not flutter.
    expect(appSource).toMatch(/onScrolledAwayChange=\{setComposerScrolledAway\}/);
    expect(appSource).not.toMatch(/onScrollActivityChange/);
  });

  it("hides the three desktop idle controls, and hides them properly", () => {
    // Desktop cannot reuse the phone approach: in the slim state
    // .app-composer-actions is ALREADY max-height 0, and the visible controls are
    // the idle attach, the idle model echo and the absolute send cluster.
    for (const control of [
      "app-composer-idle-attach",
      "app-composer-idle-mode",
      "app-composer-actions-end",
    ]) {
      expect(desktopCompactBlock).toContain(control);
    }
    // visibility, not opacity alone: a 0-opacity button is still tabbable and
    // still announced, so both control sets would sit in the a11y tree at once.
    expect(desktopCompactBlock).toMatch(/visibility:\s*hidden;/);
    expect(desktopCompactBlock).toMatch(/pointer-events:\s*none;/);
  });

  it("halves the strip and takes the run-meta line's type", () => {
    // 77px -> ~30px. Zeroing the body's block padding is what actually gets it
    // there: the expanded row's 0.6rem was 19.2px of the 49px it still measured.
    expect(desktopCompactBlock).toMatch(
      /\.app-composer-card-body\s*\{[^}]*min-height:\s*1\.8rem;[^}]*padding-block:\s*0;/s
    );
    // 12px / 400 / 20px — the "487K tokens · 1m 6s" footer.
    expect(desktopCompactBlock).toMatch(
      /\.app-composer-textarea\s*\{[^}]*font-size:\s*0\.75rem;[^}]*line-height:\s*20px;/s
    );
    expect(desktopCompactBlock).toMatch(/padding-top:\s*calc\(\(1\.8rem - 20px\) \/ 2\);/);
  });

  it("keeps phones on a tappable strip rather than the 30px desktop one", () => {
    // 30px is below the 44px touch-target floor, and while collapsed the field is
    // the ONLY route back to typing — so phones keep their own larger sizing.
    expect(phoneCompactBlock).toMatch(/min-height:\s*2\.55rem;/);
    expect(phoneCompactBlock).not.toMatch(/min-height:\s*1\.8rem;/);
    expect(desktopCompactBlock).toMatch(/@media \(min-width:\s*641px\)/);
  });

  it("restores the full composer on focus", () => {
    // Without :not(:focus-within) the user would be typing into a 30px strip with
    // no send button.
    const compactSelectors =
      desktopCompactBlock.match(/\[data-composer-compact="true"\][^{]*\{/g) ?? [];
    expect(compactSelectors.length).toBeGreaterThan(0);
    for (const selector of compactSelectors) {
      expect(selector).toContain(":not(:focus-within)");
    }
  });

  it("swaps the hint to say what still works", () => {
    // With the attach, model and send controls gone, an instruction is more use
    // than the product's name.
    expect(appSource).toMatch(/\? "Just start typing"/);
    expect(appSource).toMatch(/: "Ask Ultra"/);
    // The accessible name must NOT follow the placeholder — it stays stable.
    expect(appSource).toMatch(/aria-label="Ask Ultra"/);
  });
});
