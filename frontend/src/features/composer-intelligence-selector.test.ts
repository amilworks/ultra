import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const readSource = (relativePath: string): string =>
  readFileSync(path.join(process.cwd(), relativePath), "utf8");

describe("composer intelligence selector", () => {
  it("replaces the Pro pill with a quiet shadcn intelligence dropdown", () => {
    const app = readSource("src/App.tsx");
    const styles = readSource("src/styles.css");

    expect(app).toMatch(/type ComposerIntelligenceMode = "high" \| "pro";/);
    expect(app).toMatch(/const activeComposerIntelligenceMode: ComposerIntelligenceMode =/);
    expect(app).toMatch(/const handleSelectComposerIntelligenceMode = useCallback/);
    expect(app).toMatch(/<DropdownMenuLabel>Intelligence<\/DropdownMenuLabel>/);
    expect(app).toMatch(/<DropdownMenuGroup>/);
    expect(app).toMatch(/data-testid="composer-intelligence-selector"/);
    expect(app).toMatch(/className="app-composer-intelligence-trigger"/);
    expect(app).toMatch(/className="app-composer-intelligence-menu"/);
    expect(app).toMatch(/<span>High<\/span>[\s\S]*<span>Pro<\/span>/);
    expect(app).toMatch(/<ChevronDown data-icon="inline-end" aria-hidden="true" \/>/);
    expect(app).not.toMatch(/composer-pro-button/);
    expect(app).not.toMatch(/composer-pro-mode-toggle/);

    expect(styles).toMatch(/\.app-composer-intelligence-trigger/);
    expect(styles).toMatch(
      /\.app-composer-intelligence-trigger\s*\{[^}]*background:\s*transparent;[^}]*border:\s*0;/s
    );
    expect(styles).toMatch(
      /\.app-composer-intelligence-trigger\s*\{[^}]*font-weight:\s*400;/s
    );
    expect(styles).toMatch(/\.app-composer-intelligence-menu\[data-slot="dropdown-menu-content"\]/);
    expect(styles).toMatch(
      /\.app-composer-intelligence-menu \[data-slot="dropdown-menu-label"\]\s*\{[^}]*font-weight:\s*400;/s
    );
    expect(styles).toMatch(
      /\.app-composer-intelligence-menu \[data-slot="dropdown-menu-item"\]\s*\{[^}]*font-size:\s*0\.9375rem;/s
    );
    expect(styles).not.toMatch(/\.composer-pro-button/);
  });
});
