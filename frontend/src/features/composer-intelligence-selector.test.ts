import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const read = (relativePath: string): string =>
  readFileSync(path.join(process.cwd(), relativePath), "utf8");

const app = read("src/App.tsx");
const composer = read("src/components/composer/Composer.tsx");
const styles = read("src/styles.css");

describe("composer intelligence selector", () => {
  it("keeps one mode control: the mono tag in the bar", () => {
    expect(app).toMatch(/type ComposerIntelligenceMode = "high" \| "pro";/);
    expect(app).toMatch(/const activeComposerIntelligenceMode: ComposerIntelligenceMode =/);
    expect(app).toMatch(/const handleSelectComposerIntelligenceMode = useCallback/);
    expect(app).toMatch(/mode=\{activeComposerIntelligenceMode\}/);
    expect(app).toMatch(/onChangeMode=\{handleSelectComposerIntelligenceMode\}/);

    expect(composer).toMatch(/data-testid="composer-mode-tag"/);
    expect(composer).toMatch(/aria-label=\{`Intelligence: \$\{modeLabel\}`\}/);
    expect(composer).toMatch(/<DropdownMenuLabel>Intelligence<\/DropdownMenuLabel>/);
    expect(composer).toMatch(/data-intelligence-mode="pro"/);
    expect(composer).toMatch(/className="app-composer-intelligence-menu"/);
    // The old toolbar selector and slim echo are gone: exactly one control.
    expect(app).not.toMatch(/composer-intelligence-selector|composer-slim-intelligence-trigger/);
    expect(composer).not.toMatch(/composer-intelligence-selector|composer-slim-intelligence-trigger/);

    expect(styles).toMatch(/\.composer-tag-mode|\.composer-tag\b/);
    expect(styles).toMatch(/\.app-composer-intelligence-menu\[data-slot="dropdown-menu-content"\]/);
    expect(styles).toMatch(/@keyframes app-composer-pro-sheen/);
    expect(styles).not.toMatch(/\.composer-pro-button/);
  });

  it("introduces Pro with the reusable one-time release notice, anchored to the tag", () => {
    expect(composer).toContain('import { OneTimeNotice } from "@/components/ui/one-time-notice";');
    expect(composer).toMatch(/<OneTimeNotice[\s\S]{0,700}\{modeTag\}[\s\S]{0,40}<\/OneTimeNotice>/);
    expect(app).toMatch(/noticeId: "pro-mode-launch-2026-08-20"/);
    expect(app).toMatch(/enabled: proModeIntroEligible && newChatLandingActive/);
    expect(app).not.toContain('className="composer-pro-intro"');
    expect(styles).not.toContain(".composer-pro-intro");
  });
});
