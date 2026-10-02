import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const stylesSource = readFileSync(path.join(process.cwd(), "src/styles.css"), "utf8");

describe("Settings section keyboard focus", () => {
  it("keeps a visible focus outline inside the scrollable tab strip", () => {
    const focusRules = [
      ...stylesSource.matchAll(
        /\.app-settings-nav-item\[data-slot="tabs-trigger"\]:focus-visible\s*\{([^}]*)\}/gs
      ),
    ];
    const focusRule = focusRules.find((match) => match[1].includes("outline:"));

    expect(focusRule, "Settings tab focus outline rule is missing").toBeDefined();
    expect(focusRule?.[1]).toMatch(/outline:\s*2px solid var\(--ring\);/);
    expect(focusRule?.[1]).toMatch(/outline-offset:\s*-2px;/);
  });
});
