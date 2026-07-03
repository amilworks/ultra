/// <reference types="node" />

import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const appSource = readFileSync(path.join(process.cwd(), "src/App.tsx"), "utf8");
const collapsedRailSource = readFileSync(
  path.join(process.cwd(), "src/components/chat/CollapsedSidebarRail.tsx"),
  "utf8"
);
const runningStatusSource = readFileSync(
  path.join(process.cwd(), "src/components/chat/RunningStatusPill.tsx"),
  "utf8"
);
const stylesSource = readFileSync(path.join(process.cwd(), "src/styles.css"), "utf8");

describe("sidebar collapse layout", () => {
  it("wires the desktop sidebar to the collapsed icon rail", () => {
    expect(collapsedRailSource.includes("function CollapsedSidebarRail")).toBe(true);
    expect(collapsedRailSource.includes('aria-label="Collapsed navigation"')).toBe(true);
    expect(appSource.includes('collapsible="icon"')).toBe(true);
    expect(appSource.includes("<CollapsedSidebarRail")).toBe(true);
    expect(appSource.includes("onOpenRecent={openHistoryItem}")).toBe(true);
    expect(appSource.includes("const collapsedRecentItems")).toBe(true);
  });

  it("keeps the reference rail motion and usage-first blank chat", () => {
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
    expect(appSource.includes("<UserTokenUsagePanel")).toBe(true);
    expect(appSource.includes('const welcomeHeadline = "What are you working on?";')).toBe(false);
    expect(appSource.includes("Santa Barbara weather")).toBe(false);
    expect(stylesSource).toMatch(/\.blank-chat-usage-state\s*\{/);
    expect(stylesSource).toMatch(/\.user-token-usage-stats\s*\{[^}]*grid-template-columns:\s*repeat\(5,\s*minmax\(0,\s*1fr\)\);/s);
    expect(stylesSource).toMatch(/\.app-shell-brand\s*\{[^}]*font-weight:\s*400;/s);
  });

  it("makes the sidebar brand a ghost new-chat button", () => {
    expect(appSource).toMatch(
      /<Button\s+type="button"\s+variant="ghost"\s+className="app-sidebar-brand-button[^"]*"\s+onClick=\{createNewConversation\}/s
    );
    expect(appSource).toMatch(/aria-label="Start a new chat"/);
    expect(stylesSource).toMatch(/\.app-sidebar-brand-button\[data-slot="button"\]/);
    expect(stylesSource).toMatch(/\.app-sidebar-brand-button\[data-slot="button"\]:hover/);
  });

  it("uses the prompt-kit circular loader for running sidebar chats", () => {
    expect(runningStatusSource.includes("CircularLoader")).toBe(true);
    expect(runningStatusSource.includes("running-status-pill-beacon")).toBe(false);
    expect(stylesSource.includes(".pk-circular-loader")).toBe(true);
    expect(stylesSource.includes(".running-status-pill-loader")).toBe(true);
  });

  it("keeps Lens on the original sidebar icon with active-only top-layer fill", () => {
    expect(appSource.includes("LensSidebarIcon")).toBe(true);
    expect(stylesSource).toMatch(/\.app-lens-sidebar-icon\[data-lens-icon="active"\]/);
    expect(stylesSource).toMatch(
      /\.app-resource-browser-button\[data-variant="secondary"\]:hover\s+\.app-lens-sidebar-icon\[data-lens-icon="active"\]\s*>\s*path:first-of-type\s*\{[^}]*fill:\s*currentColor;/s
    );
    expect(stylesSource).not.toMatch(/@keyframes app-lens-active-hover/);
    expect(stylesSource).not.toMatch(/app-lens-active-hover/);
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
