import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const read = (relativePath: string): string =>
  readFileSync(path.join(process.cwd(), relativePath), "utf8");

const appSource = read("src/App.tsx");
const composerSource = read("src/components/composer/Composer.tsx");
const stylesSource = read("src/styles.css");

describe("mobile composer layout", () => {
  it("wires scroll-away state to the composer's read mode", () => {
    expect(appSource).toMatch(/const \[composerScrolledAway,\s*setComposerScrolledAway\]/);
    expect(appSource).toMatch(/onScrolledAwayChange=\{setComposerScrolledAway\}/);
    expect(appSource).toMatch(/readMode=\{composerScrolledAway\}/);
    expect(appSource).toMatch(/phone=\{isPhoneView\}/);
  });

  it("gives phones a taller bar with 44px controls", () => {
    const phone = stylesSource.slice(stylesSource.indexOf("/* Phones: a taller bar"));
    expect(phone).toMatch(/@media \(max-width:\s*640px\)\s*\{[\s\S]*--composer-bar:\s*3\.25rem;/);
    expect(phone).toMatch(/--composer-control:\s*2\.75rem;/);
    expect(composerSource).toMatch(/data-layout=\{phone \? "phone" : "desktop"\}/);
  });

  it("raises the composer above transcript controls while a menu or the picker is open", () => {
    expect(composerSource).toMatch(/data-menu-open=\{menuOpen \|\| mentionOpen \? "true" : undefined\}/);
    expect(appSource).toMatch(/menuOpen=\{slashMenuOpen \|\| composerResourcePickerOpen\}/);
    expect(stylesSource).toMatch(/\.composer\[data-menu-open="true"\]\s*\{\s*z-index:\s*40;/);
  });

  it("lays the @ picker inside the card as a sheet on phones", () => {
    expect(composerSource).toMatch(/variant=\{phone \? "sheet" : "popover"\}/);
    expect(composerSource).toMatch(/\{phone \? picker : null\}\s*<div className="composer-bar">/);
    expect(stylesSource).toMatch(/\.composer-mention-picker-sheet\s*\{\s*position:\s*relative;/);
  });

  it("keeps the in-card menu slot in normal flow — one rule, no zero height", () => {
    // A stale `height: 0` on the slot once let a 376px menu overflow the card
    // and run off the bottom of the viewport in a docked conversation.
    const section = stylesSource.slice(
      stylesSource.indexOf("The composer: one bar."),
      stylesSource.indexOf(".welcome-starters {")
    );
    const slotRules = section.match(/\n\.composer-menus\s*\{[^}]*\}/g) ?? [];
    expect(slotRules).toHaveLength(1);
    expect(slotRules[0]).not.toMatch(/height:\s*0|position:\s*absolute/);
    expect(section).toMatch(/\.composer-menu-list\s*\{[^}]*max-height:\s*min\(328px, 40dvh\);/s);
  });
});
