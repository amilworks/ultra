import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const stylesSource = readFileSync(path.join(process.cwd(), "src/styles.css"), "utf8");

describe("resource browser responsive layout", () => {
  it("keeps the mobile filter sheet inside the viewport with internal scrolling", () => {
    expect(stylesSource).toMatch(
      /@media \(max-width:\s*720px\)\s*\{[\s\S]*\.resource-browser-filter-sheet\s*\{[^}]*max-height:\s*100svh;/s
    );
  });

  it("wraps selected-resource actions on mobile so destructive actions stay visible", () => {
    // The bulk bar wraps rather than scrolling sideways (a hidden horizontal
    // scroller buried Trash/Move off the right edge), and the summary takes its
    // own row above the icon buttons.
    expect(stylesSource).toMatch(
      /@media \(max-width:\s*720px\)\s*\{[\s\S]*\.resource-browser-bulk-toolbar\s*\{[^}]*flex-wrap:\s*wrap;[^}]*overflow-x:\s*visible;[\s\S]*\.resource-browser-bulk-summary\s*\{[^}]*flex:\s*1 1 100%;[\s\S]*\.resource-browser-bulk-action-label\s*\{[^}]*display:\s*none;/s
    );
  });

  it("prioritizes filenames in the dense resource table for constrained panes", () => {
    expect(stylesSource).toMatch(
      /@media \(max-width:\s*960px\)\s*\{[\s\S]*\.resource-browser-table\s*\{[^}]*min-width:\s*0;[\s\S]*\.resource-browser-table th:nth-child\(2\),[\s\S]*\.resource-browser-table td:nth-child\(2\),[\s\S]*\.resource-browser-table th:nth-child\(3\),[\s\S]*\.resource-browser-table td:nth-child\(3\),[\s\S]*\.resource-browser-table th:nth-child\(4\),[\s\S]*\.resource-browser-table td:nth-child\(4\),[\s\S]*\.resource-browser-table th:nth-child\(5\),[\s\S]*\.resource-browser-table td:nth-child\(5\),[\s\S]*\.resource-browser-table th:nth-child\(6\),[\s\S]*\.resource-browser-table td:nth-child\(6\)\s*\{[^}]*display:\s*none;/s
    );
  });

  it("keeps active-folder context compact on mobile with a 44px clear target", () => {
    // Icon-only clear (label hidden) but a touch-sized 2.75rem hit area.
    expect(stylesSource).toMatch(
      /@media \(max-width:\s*720px\)\s*\{[\s\S]*\.resource-browser-active-collection\s*\{[^}]*grid-template-columns:\s*minmax\(0,\s*1fr\)\s*auto;[\s\S]*\.resource-browser-active-collection-actions\s*\{[^}]*flex-direction:\s*row;[\s\S]*\.resource-browser-active-collection-clear\s*\{[^}]*width:\s*2\.75rem;[\s\S]*\.resource-browser-active-collection-clear-label\s*\{[^}]*display:\s*none;/s
    );
  });

  it("keeps active-folder context as a calm breadcrumb", () => {
    expect(stylesSource).not.toMatch(
      /\.resource-browser-active-collection\s*\{[^}]*border-left:/s
    );
    expect(stylesSource).toMatch(
      /\.resource-browser-active-collection-current\s*\{[^}]*background:\s*transparent;/s
    );
  });

  it("keeps the resource share sheet visually opaque", () => {
    expect(stylesSource).toMatch(
      /\.resource-browser-share-sheet\s*\{[^}]*background:\s*var\(--bg-panel-strong\);/s
    );
  });
});
