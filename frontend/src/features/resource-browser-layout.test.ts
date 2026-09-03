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

  it("keeps the mobile resources toolbar in one calm search row", () => {
    expect(stylesSource).toMatch(
      /@media \(max-width:\s*720px\)\s*\{[\s\S]*\.resource-browser-toolbar\s*\{[^}]*grid-template-columns:\s*minmax\(0,\s*1fr\)\s*2\.75rem\s*2\.75rem;[\s\S]*\.resource-browser-search-field\s*\{[^}]*min-width:\s*0;[\s\S]*\.resource-browser-filter-label\s*\{[^}]*display:\s*none;/s
    );
  });

  it("does not pin the resources header over the mobile list", () => {
    // Search lives in the app nav bar on mobile, so this row scrolls away instead
    // of permanently occupying ~63px of the viewport.
    expect(stylesSource).toMatch(
      /@media \(max-width:\s*720px\)\s*\{[\s\S]*\.resource-browser-header\s*\{[^}]*position:\s*static;/s
    );
  });

  it("sizes mobile resource rows to their content", () => {
    // `grid-auto-rows: 1fr` resolves as minmax(auto, 1fr), so in one column every
    // card inherited the tallest card's height — a compact card that wants ~92px
    // was stretched past 300px of empty panel.
    expect(stylesSource).toMatch(
      /@media \(max-width:\s*720px\)\s*\{[\s\S]*\.resource-browser-grid\s*\{[^}]*grid-auto-rows:\s*min-content;/s
    );
  });

  it("keeps tag pills reachable on mobile tiles", () => {
    // Hiding tags is only justified where the tile height is fixed. Below 721px the
    // card is height:auto AND the table is downgraded to cards, so an unscoped rule
    // would make tags unreachable rather than tidy.
    expect(stylesSource).toMatch(
      /@media \(min-width:\s*721px\)\s*\{\s*\.resource-browser-card\[data-preview="true"\]\s*\.resource-browser-resource-tags\s*\{[^}]*display:\s*none;/s
    );
  });

  it("keeps the tile status overlay legible over any thumbnail", () => {
    // Overlaid on unpredictable image content, so it cannot be translucent: the
    // 92% wash measured 3.96:1 over a dark thumbnail in light theme.
    expect(stylesSource).toMatch(
      /\.resource-browser-status-overlay > div > span\s*\{[^}]*background:\s*var\(--bg-panel-strong\);[^}]*color:\s*var\(--text-main\);/s
    );
  });

  it("keeps the desktop tile geometry out of the mobile block", () => {
    const mobileBlockStart = stylesSource.indexOf("@media (max-width: 720px)");
    const cardRule = stylesSource.indexOf(".resource-browser-card {");
    // The fixed desktop card + its meta row floor live outside any media query.
    expect(cardRule).toBeGreaterThan(-1);
    expect(cardRule).toBeLessThan(mobileBlockStart);
    expect(stylesSource).toMatch(
      /\.resource-browser-card\s*\{[^}]*height:\s*17\.75rem;[^}]*grid-template-rows:\s*10\.75rem minmax\(0, 1fr\);/s
    );
    // The implicit column must be pinned to the card width: a `white-space: pre`
    // preview snippet's longest line otherwise inflates the inner track (1877px
    // measured in a 365px card), and filenames hard-clip on one giant line
    // instead of clamping to two. Never let this column go implicit again.
    expect(stylesSource).toMatch(
      /\.resource-browser-card\s*\{[^}]*grid-template-columns:\s*minmax\(0, 1fr\);/s
    );
    // Content-sized meta rows are what stop a chip from crushing the type/size and
    // date lines. Never silently revert this.
    expect(stylesSource).toMatch(
      /\.resource-browser-meta\s*\{[^}]*grid-auto-rows:\s*min-content;/s
    );
    // The desktop grid's implicit rows must be content-sized too: `1fr` sizes
    // every row to the tallest row in the grid, which stretched the calendar
    // section eyebrows to full card height (284px for a 29px label). Cards
    // carry a fixed height, so `auto` keeps card rows identical anyway.
    expect(stylesSource).toMatch(
      /\.resource-browser-grid\s*\{[^}]*grid-auto-rows:\s*auto;/s
    );
    expect(stylesSource).toMatch(
      /\.resource-browser-section-label\s*\{[^}]*grid-column:\s*1 \/ -1;/s
    );
  });

  it("treats filenames as scan text instead of dense mini-headings", () => {
    // The label rung (nav token, regime-stable — body inflates on mobile) with
    // tabular figures: machine-stamped siblings differ only in a digit run, and
    // equal-width digits let adjacent cards diff at a glance while scrolling.
    expect(stylesSource).toMatch(
      /\.resource-browser-name\s*\{[^}]*-webkit-line-clamp:\s*2;[^}]*font-size:\s*var\(--font-size-nav\);[^}]*font-weight:\s*var\(--font-weight-action\);[^}]*font-variant-numeric:\s*tabular-nums;[^}]*line-height:\s*1\.32;/s
    );
    expect(stylesSource).toMatch(
      /\.resource-browser-details,\s*\.resource-browser-date\s*\{[^}]*font-variant-numeric:\s*tabular-nums;/s
    );
    expect(stylesSource).toMatch(
      /\.resource-browser-card\[data-preview="false"\]\s*\.resource-browser-meta\s*\{[^}]*justify-content:\s*stretch;/s
    );
    expect(stylesSource).not.toMatch(
      /\.resource-browser-card\[data-preview="false"\]\s*\.resource-browser-name\s*\{[^}]*-webkit-line-clamp:\s*1;/s
    );
  });

  it("keeps one source of truth for the mobile resources toolbar", () => {
    // A later `max-width: 640px` copy of these rules used to win on cascade order,
    // so edits to the 720px block silently no-op'd on small phones.
    const narrowBlocks = stylesSource.split(/@media \(max-width:\s*640px\)/).slice(1);
    for (const block of narrowBlocks) {
      const body = block.slice(0, block.indexOf("\n}\n"));
      expect(body).not.toMatch(/\.resource-browser-toolbar\s*\{[^}]*grid-template-columns:/s);
    }
  });

  it("keeps the mobile resources header compact under the shell nav", () => {
    expect(stylesSource).toMatch(
      /@media \(max-width:\s*720px\)\s*\{[\s\S]*\.resource-browser-header\s*\{[^}]*gap:\s*0\.55rem;[^}]*padding:\s*0\.65rem 0\.9rem 0\.55rem;[\s\S]*\.resource-browser-title\s*\{[^}]*display:\s*none;[\s\S]*\.resource-browser-result-summary\s*\{[^}]*font-size:\s*0\.8rem;/s
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
