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

  it("gives phones 44px controls and a tappable collapsed strip", () => {
    const phone = stylesSource.slice(stylesSource.indexOf("/* Phones: 44px targets"));
    expect(phone).toMatch(/@media \(max-width:\s*640px\)\s*\{[\s\S]*--composer-control:\s*2\.75rem;/);
    expect(stylesSource).toMatch(
      /@media \(max-width:\s*640px\)\s*\{\s*\.composer\[data-read-mode="true"\] \.composer-line\s*\{\s*min-height:\s*2\.55rem;/
    );
    expect(composerSource).toMatch(/data-layout=\{phone \? "phone" : "desktop"\}/);
  });

  it("raises the composer above transcript controls while a menu or the picker is open", () => {
    expect(composerSource).toMatch(/data-menu-open=\{menuOpen \|\| mentionOpen \? "true" : undefined\}/);
    expect(appSource).toMatch(/menuOpen=\{slashMenuOpen \|\| composerResourcePickerOpen\}/);
    expect(stylesSource).toMatch(/\.composer\[data-menu-open="true"\]\s*\{\s*z-index:\s*40;/);
  });

  it("lays the @ picker under the line as a sheet on phones", () => {
    expect(composerSource).toMatch(/variant=\{phone \? "sheet" : "popover"\}/);
    expect(stylesSource).toMatch(/\.composer-mention-picker-sheet\s*\{\s*position:\s*relative;/);
  });
});
