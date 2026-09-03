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

/** A top-level @container block, from its marker to the next regime block. */
const containerBlock = (marker: string) => {
  const start = css.indexOf(marker);
  expect(start, `expected to find ${marker}`).toBeGreaterThan(-1);
  const next = css.slice(start + marker.length).search(/\n@(?:container|media)/);
  return next === -1 ? css.slice(start) : css.slice(start, start + marker.length + next);
};

const stackedBlock = containerBlock("@container hdf5shell (max-width: 719px)");
const phoneBlock = containerBlock("@container hdf5shell (max-width: 620px)");
const coarseBlock = mediaBlock("@media (pointer: coarse)", true);

describe("hdf5 viewer layout regimes", () => {
  it("decides the regime by the shell container, never the viewport", () => {
    /* An expanded app sidebar eats ~240px, so a "desktop" window can still
       host a narrow viewer. The stacked/two-pane decision must follow the
       shell's own width; viewport media queries reintroduce the crushed
       two-pane the owner hit in an ~880px window. The remaining @media are
       device traits (pointer) and the chat-embedded height, which sit
       outside the shell. */
    expect(css).toMatch(/\.viewer-hdf-shell\s*\{[^}]*container:\s*hdf5shell \/ size;/);
    expect(css).not.toMatch(/@media \(max-width: (?:839|1120)px\)/);
    expect(css).not.toMatch(/@media \(min-width: (?:840|1121)px\)/);
    /* Column widths inside the container are container-relative units. */
    expect(css).toMatch(/grid-template-columns: clamp\(280px, 25cqw, 312px\)/);
    expect(css).not.toMatch(/\.viewer-hdf-dashboard\s*\{[^}]*vw/);
  });

  it("lets wrap-capable tab lists grow everywhere", () => {
    /* The primitive's TabsList is a fixed h-10 box while these lists are
       allowed to wrap in any narrow container, and coarse-pointer triggers
       are 44px tall — wrapped or tall rows painted straight over the
       content below (the Metadata/File row sat on top of the preview copy).
       The box grows in the BASE rules: a single stock-height row still
       renders at exactly 40px, so the wide two-pane look is unchanged. */
    /* Match at column 0 — the top-level (unscoped) rule, not a regime copy. */
    expect(css).toMatch(
      /\n\.viewer-hdf-detail-tabs-list,\n\.viewer-hdf-preview-tabs-list\s*\{[^}]*height:\s*auto;[^}]*min-height:\s*2\.5rem;/
    );
    expect(css).toMatch(
      /\n\.viewer-hdf-workspace-tabs-list\s*\{[^}]*height:\s*auto;[^}]*min-height:\s*2\.5rem;/
    );
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

  it("lets the dataset list fill its rail and drops desktop scroll floors inside the capped card", () => {
    /* CardContent renders one Command child. The single bounded base track and
       the local max-height reset let that command/list consume the desktop
       rail instead of leaving an empty 1fr track below a 320px primitive. */
    expect(css).toMatch(
      /\n\.viewer-hdf-navigator-content\s*\{[^}]*grid-template-rows:\s*minmax\(0,\s*1fr\);/
    );
    expect(css).toMatch(
      /\n\.viewer-hdf-search-results\s*\{[^}]*max-height:\s*none;/
    );
    /* Inside the min(420px, 46dvh) navigator card, the desktop 260px floor
       plus the toolbar and header exceed the cap on phones and can clip the
       search input. The bounded track governs in this regime. */
    expect(stackedBlock).toMatch(
      /\.viewer-hdf-tree-scroll,\s*\.viewer-hdf-detail-scroll,\s*\.viewer-hdf-search-results\s*\{[^}]*min-height:\s*120px;/
    );
    /* Two-pane keeps its floors. */
    expect(css).toMatch(
      /\n\.viewer-hdf-tree-scroll,\s*\.viewer-hdf-detail-scroll\s*\{[^}]*min-height:\s*260px;/
    );
  });

  it("stacks the split header at phone width", () => {
    expect(phoneBlock).toMatch(
      /\.viewer-hdf-dashboard-header-split\s*\{[^}]*flex-direction:\s*column;/
    );
    expect(coarseBlock).toMatch(/min-height:\s*44px/);
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
    /* The JS twin observes the SAME container the CSS queries — the shell —
       at the same 720px threshold. A viewport hook here silently diverges
       from the container regime when the app sidebar is expanded. */
    expect(navigatorSource).toMatch(/closest\(".viewer-hdf-shell"\)/);
    expect(navigatorSource).toMatch(/clientWidth < 720/);
    expect(navigatorSource).not.toMatch(/useBreakpoint\(/);
    expect(navigatorSource).toMatch(/stackedLayout \? \{ value: "", defaultValue: "" \} : \{\}/);
  });

  it("keeps the explainer copy out of the viewer chrome", () => {
    /* Owner direction: the surface is for visualizing data, not narrating
       the UI. These strings were removed on purpose — the tab names and
       controls are the documentation. Reintroducing marketing-voice helper
       paragraphs should fail here, not in review. */
    const inspectorSource = read("Hdf5Inspector.tsx");
    const previewSource = read("Hdf5DatasetPreview.tsx");
    expect(inspectorSource).not.toMatch(/Keep the selected dataset context/);
    expect(inspectorSource).not.toMatch(/Dataset details/);
    expect(previewSource).not.toMatch(/Use Volume for 3D preview inspection/);
    expect(previewSource).not.toMatch(/uses bounded atlas data/);
    expect(previewSource).not.toMatch(/Use charts for a quick read/);
    expect(previewSource).not.toMatch(/<strong>Table preview<\/strong>/);
    expect(previewSource).not.toMatch(/viewer-hdf-preview-toolbar-copy/);
    /* The one explanatory survivor is the quiet caption voice. */
    expect(previewSource).toMatch(/viewer-hdf-detail-caption/);
    expect(inspectorSource).toMatch(/viewer-hdf-detail-caption/);
  });

  it("uses the neutral Meridian ladder for dataset selection", () => {
    const selectedStart = css.indexOf(".viewer-hdf-command-item.is-selected {");
    expect(selectedStart).toBeGreaterThan(-1);
    const selectedRule = css.slice(selectedStart, css.indexOf("}", selectedStart) + 1);

    expect(selectedRule).toMatch(/background-color:\s*var\(--bg-sunk\);/);
    expect(selectedRule).toMatch(/border-color:\s*transparent;/);
    expect(selectedRule).toMatch(/box-shadow:\s*none;/);
    expect(selectedRule).not.toMatch(/--accent/);
    expect(css).toMatch(
      /\.viewer-hdf-command-item:focus-visible\s*\{[^}]*outline:\s*2px solid var\(--ring\);/
    );
  });

  it("gives the hero filename the house title voice", () => {
    /* Tailwind's preflight sets heading weight to `inherit`, so this h3 was
       rendering at body 400. */
    expect(css).toMatch(
      /\.viewer-hdf-hero-heading h3\s*\{[^}]*font-weight:\s*600;/
    );
  });

  it("fills the two-pane preview column instead of clamping the canvas", () => {
    const twoPane = containerBlock("@container hdf5shell (min-width: 720px)");
    expect(twoPane).toMatch(
      /\.viewer-hdf-slice-canvas,\s*\[data-hdf5-slice-preview="true"\] \.viewer-hdf-slice-canvas\s*\{[^}]*height:\s*100%;[^}]*max-height:\s*none;/
    );
    /* Gated on data-state so the display override cannot defeat [hidden]
       on inactive Radix tab panes. */
    expect(twoPane).toMatch(
      /\.viewer-hdf-preview-tab\[data-state="active"\]\s*\{[^}]*display:\s*flex;/
    );
    expect(twoPane).not.toMatch(/\n\s*\.viewer-hdf-preview-tab\s*\{[^}]*display:/);
    /* Side-by-side slice columns need more room than the two-pane floor. */
    expect(css).toMatch(/@container hdf5shell \(min-width: 920px\)/);
  });
});
