/// <reference types="node" />

import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const read = (name: string) =>
  readFileSync(path.join(process.cwd(), "src/components/viewer/hdf5", name), "utf8");

const css = read("hdf5-viewer.css");
const navigatorSource = read("Hdf5Navigator.tsx");

/** A top-level @media block, from its marker to the next top-level @media. */
const mediaBlock = (marker: string, useLast = false) => {
  const start = useLast ? css.lastIndexOf(marker) : css.indexOf(marker);
  expect(start, `expected to find ${marker}`).toBeGreaterThan(-1);
  const next = css.indexOf("\n@media", start + marker.length);
  return css.slice(start, next === -1 ? css.length : next);
};

const stackedBlock = mediaBlock("@media (max-width: 1120px)");
const phoneBlock = mediaBlock("@media (max-width: 620px)");
const coarseBlock = mediaBlock("@media (pointer: coarse)", true);

describe("hdf5 viewer mobile layout", () => {
  it("keeps every fix inside a mobile/touch regime — desktop rules stay stock", () => {
    /* Owner constraint on this change: fix the phone, do not touch the main
       (desktop) viewer. The base rules for the tab lists must therefore keep
       the stock fixed-height box; only the regime blocks may grow it. */
    const baseListRule = css.slice(
      css.indexOf("\n.viewer-hdf-detail-tabs-list,"),
      css.indexOf("}", css.indexOf("\n.viewer-hdf-detail-tabs-list,"))
    );
    expect(baseListRule).toBeTruthy();
    expect(baseListRule).not.toMatch(/height:/);
  });

  it("lets wrap-capable tab lists grow in both touch regimes", () => {
    /* The primitive's TabsList is a fixed h-10 box while these lists are
       allowed to wrap AND coarse-pointer triggers are 44px tall — wrapped
       rows painted straight over the content below (the Metadata/File row
       sat on top of the preview copy on phones). The box must grow wherever
       either condition can hold: stacked widths and touch pointers. */
    for (const block of [stackedBlock, coarseBlock]) {
      const listRule = block.slice(block.indexOf(".viewer-hdf-workspace-tabs-list,"));
      expect(listRule).toMatch(
        /\.viewer-hdf-workspace-tabs-list,\s*\.viewer-hdf-detail-tabs-list,\s*\.viewer-hdf-preview-tabs-list\s*\{[^}]*height:\s*auto;[^}]*min-height:\s*2\.5rem;/
      );
    }
  });

  it("content-sizes the stacked dashboard rows so the scrollport can scroll", () => {
    /* The shell keeps the dashboard at height:100% as the ONE scrollport in
       the stacked regime — but AUTO rows stop growing once a definite
       container is full, so the stacked cards were fit-distributed to the
       viewport (the inspector got 248px for ~500px of content and its
       overflow:hidden ate the tabs). max-content rows overflow into the
       scrollport instead, which is the entire point of having one. */
    expect(stackedBlock).toMatch(
      /\.viewer-hdf-dashboard\s*\{[^}]*grid-auto-rows:\s*max-content;/
    );
  });

  it("drops the desktop scroll floors inside the capped navigator card", () => {
    /* Desktop floors these scrollports at 260px and the command primitive
       caps its list at 320px; inside the min(420px, 46dvh) navigator card
       the floor + toolbar + header exceed the cap on phones, which clipped
       the search input in half. The bounded track governs in this regime. */
    expect(stackedBlock).toMatch(
      /\.viewer-hdf-tree-scroll,\s*\.viewer-hdf-detail-scroll,\s*\.viewer-hdf-search-results\s*\{[^}]*min-height:\s*120px;/
    );
    expect(stackedBlock).toMatch(/\.viewer-hdf-search-results\s*\{[^}]*max-height:\s*none;/);
    expect(stackedBlock).toMatch(
      /\.viewer-hdf-navigator-content\s*\{[^}]*grid-template-rows:\s*minmax\(0,\s*1fr\);/
    );
    /* Desktop keeps its floors. */
    expect(css).toMatch(
      /\n\.viewer-hdf-tree-scroll,\s*\.viewer-hdf-detail-scroll\s*\{[^}]*min-height:\s*260px;/
    );
  });

  it("stacks the split header at phone width", () => {
    expect(phoneBlock).toMatch(
      /\.viewer-hdf-dashboard-header-split\s*\{[^}]*flex-direction:\s*column;/
    );
  });

  it("never lets navigator auto-scroll escape the dataset list", () => {
    /* scrollIntoView walks EVERY scrollable ancestor. In the stacked layout
       that includes the dashboard scrollport, so both this component's
       active-row follower and cmdk's own selection scroller yanked the page
       down to the dataset list (on open and on every filter keystroke).
       The follower must do bounded scrollTop math on the list element, and
       cmdk must be starved of a selection in the stacked regime (its
       scroller targets [aria-selected="true"]). */
    expect(navigatorSource).not.toMatch(/\.scrollIntoView\(/);
    expect(navigatorSource).toMatch(/closest\('\[data-hdf5-dataset-list="true"\]'\)/);
    expect(navigatorSource).toMatch(/useBreakpoint\(1121\)/);
    expect(navigatorSource).toMatch(/stackedLayout \? \{ value: "", defaultValue: "" \} : \{\}/);
  });
});
