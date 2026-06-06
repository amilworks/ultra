/// <reference types="node" />

import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const appSource = readFileSync(path.join(process.cwd(), "src/App.tsx"), "utf8");
const stylesSource = readFileSync(path.join(process.cwd(), "src/styles.css"), "utf8");

describe("sidebar collapse layout", () => {
  it("wires the desktop sidebar to the collapsed icon rail", () => {
    expect(appSource.includes("function CollapsedSidebarRail")).toBe(true);
    expect(appSource.includes('aria-label="Collapsed navigation"')).toBe(true);
    expect(appSource.includes('collapsible="icon"')).toBe(true);
    expect(appSource.includes("<CollapsedSidebarRail")).toBe(true);
    expect(appSource.includes("onOpenRecent={openHistoryItem}")).toBe(true);
    expect(appSource.includes("const collapsedRecentItems")).toBe(true);
  });

  it("keeps the reference rail motion and welcome title weight", () => {
    expect(stylesSource.includes("--sidebar-motion-duration: 280ms;")).toBe(true);
    expect(stylesSource.includes(".app-collapsed-sidebar-rail")).toBe(true);
    expect(
      stylesSource.includes(
        '[data-slot="sidebar"][data-collapsible="icon"] .app-collapsed-sidebar-rail'
      )
    ).toBe(true);
    expect(
      stylesSource.includes(
        '[data-slot="sidebar"][data-collapsible="icon"] + [data-slot="sidebar-inset"] .app-sidebar-trigger'
      )
    ).toBe(true);
    expect(stylesSource).toMatch(/\.hero-title-welcome\s*\{[^}]*font-weight:\s*300;/s);
  });

  it("keeps a mobile-visible trigger for the sidebar sheet", () => {
    expect(appSource).toMatch(/className="app-mobile-shell-bar md:hidden"/);
    expect(appSource).toMatch(/className="app-mobile-sidebar-trigger"/);
    expect(appSource).toMatch(/aria-label="Open navigation"/);
    expect(stylesSource).toMatch(/\.app-mobile-shell-bar\s*\{[^}]*min-height:/s);
    expect(stylesSource).toMatch(
      /\.app-mobile-sidebar-trigger\[data-slot="sidebar-trigger"\]\s*\{[^}]*width:\s*2\.35rem;/s
    );
    expect(stylesSource).toMatch(
      /\.app-sidebar\[data-mobile="true"\] \.app-sidebar-content\s*\{[^}]*overflow-y:\s*auto;/s
    );
    expect(stylesSource).toMatch(
      /\.app-sidebar\[data-mobile="true"\] \.app-sidebar-history-scroll\s*\{[^}]*overflow:\s*visible;/s
    );
    expect(stylesSource).toMatch(
      /@media \(min-width:\s*768px\)\s*\{[^}]*\.app-mobile-shell-bar\s*\{[^}]*display:\s*none;/s
    );
  });
});
