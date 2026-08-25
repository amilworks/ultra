import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const readSource = (relativePath: string): string =>
  readFileSync(path.join(process.cwd(), relativePath), "utf8");

describe("sidebar settings menu", () => {
  it("composes a shadcn settings entry below chat history with a relevant settings dialog", () => {
    const app = readSource("src/App.tsx");
    const accountButton = readSource("src/components/chat/SidebarAccountSettingsButton.tsx");
    const deferredToaster = readSource("src/components/DeferredToaster.tsx");
    const toastLib = readSource("src/lib/toast.ts");
    const settingsDialog = readSource("src/components/AppSettingsDialog.tsx");
    const styles = readSource("src/styles.css");

    expect(accountButton).toMatch(/function SidebarAccountSettingsButton/);
    expect(app).toMatch(/const LazyAppSettingsDialog = lazyNamed/);
    expect(settingsDialog).toMatch(/export function AppSettingsDialog/);
    expect(settingsDialog).toMatch(/<Avatar/);
    expect(settingsDialog).toMatch(/<AvatarFallback>/);
    expect(settingsDialog).toMatch(/<Dialog open={open} onOpenChange={onOpenChange}>/);
    expect(settingsDialog).toMatch(/<DialogTitle>Settings<\/DialogTitle>/);
    expect(settingsDialog).toMatch(/<TabsList/);
    expect(settingsDialog).toMatch(/<SelectGroup>/);
    expect(settingsDialog).toMatch(/<Separator/);
    expect(deferredToaster).toMatch(/const LazyToaster = lazy/);
    expect(toastLib).toMatch(/import type \* as Sonner from "sonner";/);
    expect(app).not.toMatch(
      /import\s+\{[^}]*\b(?:Toaster|toast)\b[^}]*\}\s+from "sonner";/
    );
    expect(deferredToaster).not.toMatch(
      /import\s+\{[^}]*\b(?:Toaster|toast)\b[^}]*\}\s+from "sonner";/
    );
    expect(toastLib).not.toMatch(
      /import\s+\{[^}]*\b(?:Toaster|toast)\b[^}]*\}\s+from "sonner";/
    );
    expect(settingsDialog).toMatch(/import \{ toast \} from "sonner";/);
    expect(settingsDialog).toMatch(/Alert,\s*AlertAction,\s*AlertDescription,\s*AlertTitle/);
    expect(settingsDialog).toMatch(/onLinkBisqueAccount/);
    expect(settingsDialog).toMatch(/onUnlinkBisqueAccount/);
    expect(settingsDialog).toMatch(/settings-bisque-username/);
    expect(settingsDialog).toMatch(/settings-bisque-password/);
    expect(app).toMatch(/const \[bisqueCredentialsLinked,\s*setBisqueCredentialsLinked\]/);
    expect(app).toMatch(/setBisqueCredentialsLinked\(Boolean\(session\.bisque_linked\)\)/);
    expect(app).toMatch(
      /authStatus !== "authenticated" \|\|\s*\(authMode !== "bisque" && authMode !== "workos"\) \|\|\s*!bisqueCredentialsLinked \|\|\s*!bisqueNavLinks/
    );
    expect(settingsDialog).toMatch(/const bisqueLinked = Boolean\(bisqueCredentialsLinked && authUser\)/);
    expect(app).toMatch(
      /apiClient\.searchBisqueResources\(\{\s*resourceType: "image",\s*scope: "owner",\s*limit: 1,\s*countAll: true,/s
    );
    expect(app).toMatch(/type BisqueResourceCounts/);
    expect(app).toMatch(/formatBisqueShortcutLabel/);
    expect(app).toMatch(/resourceType: "image",\s*scope: "owner",\s*limit: 1,\s*countAll: true/s);
    expect(app).toMatch(/resourceType: "dataset",\s*scope: "owner",\s*limit: 1,\s*countAll: true/s);
    expect(app).toMatch(/resourceType: "table",\s*scope: "owner",\s*limit: 1,\s*countAll: true/s);
    expect(app).toMatch(
      /formatBisqueShortcutLabel\(\s*bisqueResourceCounts\?\.image,\s*"Image",\s*"Images"\s*\)/s
    );
    expect(app).toMatch(
      /formatBisqueShortcutLabel\(\s*bisqueResourceCounts\?\.dataset,\s*"Dataset",\s*"Datasets"\s*\)/s
    );
    expect(app).toMatch(
      /formatBisqueShortcutLabel\(\s*bisqueResourceCounts\?\.table,\s*"Table",\s*"Tables"\s*\)/s
    );
    expect(app).not.toMatch(/<span>View Images<\/span>/);
    expect(app).not.toMatch(/<span>View Datasets<\/span>/);
    expect(app).not.toMatch(/<span>View Tables<\/span>/);
    expect(settingsDialog).toMatch(/BisQue account linked/);
    expect(settingsDialog).toMatch(/Open BisQue/);
    expect(settingsDialog).toMatch(/Unlink account/);
    expect(app).toMatch(/showSuccessToast\("Successfully linked BisQue account"/);
    expect(deferredToaster).toMatch(/<LazyToaster/);
    expect(accountButton).toMatch(/app-sidebar-user-menu/);
    expect(accountButton).toMatch(/<Avatar className="app-sidebar-account-avatar">/);
    expect(accountButton).toMatch(/className="app-sidebar-account-menu"/);
    expect(accountButton).toMatch(/<span>Dark<\/span>[\s\S]*<span>Settings<\/span>/);
    expect(app).not.toMatch(/app-theme-menu-button/);
    expect(app).toMatch(/<span>New chat<\/span>/);
    // The mobile sidebar chat-search was removed by request; guard against reintroduction.
    expect(app).not.toMatch(/Search chats/);
    expect(app).not.toMatch(/mobileConversationQuery/);
    expect(accountButton).toMatch(/<Settings data-icon="inline-start" aria-hidden="true" \/>/);
    expect(settingsDialog).toMatch(/GitBranch/);
    expect(settingsDialog).toMatch(
      /<GitBranch data-icon="inline-start" aria-hidden="true" \/>\s*GitHub/
    );
    expect(app).not.toMatch(/app-settings-nav-icon/);

    expect(styles).toMatch(/\.app-sidebar-user-menu/);
    expect(styles).toMatch(/\.app-sidebar-user-menu\s*\{[^}]*border-top:\s*0;/s);
    expect(styles).toMatch(
      /--sidebar-radius-action:\s*calc\(var\(--radius\) \+ 0\.05rem\);/
    );
    expect(styles).toMatch(
      /--sidebar-radius-row:\s*calc\(var\(--radius\) - 0\.125rem\);/
    );
    expect(styles).toMatch(
      /--sidebar-radius-menu:\s*calc\(var\(--radius\) \+ 0\.2rem\);/
    );
    expect(styles).toMatch(
      /--sidebar-radius-menu-item:\s*calc\(var\(--radius\) - 0\.15rem\);/
    );
    expect(styles).toMatch(/\.app-sidebar-account-button/);
    expect(styles).toMatch(
      /\.app-sidebar-account-button\s*\{[^}]*min-height:\s*2\.55rem;/s
    );
    expect(styles).toMatch(
      /\.app-sidebar-account-button\s*\{[^}]*border-radius:\s*var\(--sidebar-radius-row\);/s
    );
    expect(styles).toMatch(/\.app-history-group \[data-slot="sidebar-group-label"\]/);
    // One voice for every sidebar section label: type comes from the shared eyebrow
    // tokens on the BASE group-label rule. This block previously asserted the
    // opposite — font-size: 0.875rem / font-weight: 600 on the history label — which
    // was pinning the defect that made "Recents" outrank the rows it labels.
    expect(styles).toMatch(/--sidebar-eyebrow-size:\s*0\.68rem;/);
    expect(styles).toMatch(
      /\.app-sidebar-content \[data-slot="sidebar-group-label"\]\s*\{[^}]*font-size:\s*var\(--sidebar-eyebrow-size\);[^}]*font-weight:\s*var\(--sidebar-eyebrow-weight\);[^}]*letter-spacing:\s*var\(--sidebar-eyebrow-tracking\);/s
    );

    // ...and the history-group override stays LAYOUT-ONLY. Matched explicitly rather
    // than with `?? ""` so that deleting the rule fails here instead of silently
    // satisfying the negative assertion below.
    const historyLabelMatch = styles.match(
      /\.app-sidebar-content \.app-history-group \[data-slot="sidebar-group-label"\]\s*\{([^}]*)\}/s
    );
    expect(historyLabelMatch, "history group-label rule is missing").not.toBeNull();
    const historyLabelRule = historyLabelMatch?.[1] ?? "";
    expect(historyLabelRule).toMatch(/line-height:\s*1\.25;/);
    // The `[^-]` guard keeps this from false-matching `border-color:`.
    expect(historyLabelRule).not.toMatch(/font-size|font-weight|letter-spacing|(^|[^-])\bcolor:/);
    expect(styles).toMatch(
      /\.app-new-chat-button\s*\{[^}]*background:\s*color-mix\(in oklab, var\(--sidebar-accent\) 74%, transparent\);/s
    );
    expect(styles).toMatch(
      /\.app-new-chat-button\s*\{[^}]*border-radius:\s*var\(--sidebar-radius-action\);/s
    );
    expect(styles).toMatch(
      /\.app-new-chat-button,\s*\.app-resource-browser-button,\s*\.app-bisque-browser-button\s*\{[^}]*border-radius:\s*var\(--sidebar-radius-row\);/s
    );
    expect(styles).toMatch(
      /\.app-bisque-link-button\s*\{[^}]*border-radius:\s*var\(--sidebar-radius-row\);/s
    );
    expect(styles).toMatch(/\.app-history-button\s*\{[^}]*font-size:\s*0\.875rem;/s);
    expect(styles).toMatch(
      /\.app-history-button\s*\{[^}]*border-radius:\s*var\(--sidebar-radius-row\);/s
    );
    expect(styles).toMatch(
      /\.app-sidebar-account-name\s*\{[^}]*font-size:\s*0\.82rem;[^}]*font-weight:\s*500;/s
    );
    expect(styles).toMatch(/\.app-sidebar-account-button:hover/);
    expect(styles).toMatch(/\.app-sidebar-account-menu\[data-slot="dropdown-menu-content"\]/);
    expect(styles).toMatch(
      /\.app-sidebar-account-menu\[data-slot="dropdown-menu-content"\]\s*\{[^}]*border-radius:\s*var\(--sidebar-radius-menu\);/s
    );
    expect(styles).toMatch(
      /\.app-sidebar-account-menu \[data-slot="dropdown-menu-item"\]\s*\{[^}]*border-radius:\s*var\(--sidebar-radius-menu-item\);/s
    );
    expect(styles).toMatch(/\.app-settings-dialog/);
    expect(styles).toMatch(
      /\.app-settings-dialog\[data-slot="dialog-content"\]\s*\{[^}]*background:\s*var\(--popover\);[^}]*color:\s*var\(--popover-foreground\);/s
    );
    expect(styles).toMatch(
      /body:has\(\[data-slot="sidebar"\]\[data-state="collapsed"\]\) \.app-settings-dialog\[data-slot="dialog-content"\]\s*\{[^}]*--settings-dialog-main-offset:\s*calc\(var\(--sidebar-width-icon, 4rem\) \/ 2\);/s
    );
    expect(styles).toMatch(
      /\.app-settings-sidebar-pane\s*\{[^}]*background:\s*color-mix\(in oklab, var\(--popover\) 98%, var\(--muted\) 2%\);/s
    );
    expect(styles).toMatch(
      /\.app-settings-shell\s*\{[^}]*grid-template-columns:\s*minmax\(11\.25rem,\s*13rem\) minmax\(0,\s*1fr\);/s
    );
    expect(styles).toMatch(
      /\.app-settings-nav-item\[data-slot="tabs-trigger"\]\s*\{[^}]*min-height:\s*2\.08rem;[^}]*gap:\s*var\(--sidebar-item-gap\);[^}]*font-size:\s*0\.875rem;[^}]*font-weight:\s*500;/s
    );
    expect(styles).toMatch(
      /\.app-settings-nav-item\[data-slot="tabs-trigger"\] svg\[data-icon\]\s*\{[^}]*width:\s*1rem;[^}]*height:\s*1rem;[^}]*flex:\s*0 0 1rem;/s
    );
    expect(styles).toMatch(
      /\.app-settings-nav-item\[data-state="active"\]\s*\{[^}]*background:\s*color-mix\(in oklab, var\(--muted\) 68%, transparent\);[^}]*font-weight:\s*500;/s
    );
    expect(styles).toMatch(
      /\.app-settings-row,\s*\.app-settings-account-summary\s*\{[^}]*min-height:\s*3\.25rem;/s
    );
    expect(styles).toMatch(
      /\.app-settings-header \[data-slot="dialog-title"\]\s*\{[^}]*font-size:\s*0\.95rem;[^}]*font-weight:\s*600;/s
    );
    expect(styles).toMatch(
      /\.app-settings-panel-heading h2\s*\{[^}]*font-size:\s*0\.98rem;[^}]*font-weight:\s*var\(--font-weight-panel-heading\);/s
    );
    expect(styles).toMatch(
      /\.app-settings-row-title,\s*\.app-settings-account-name\s*\{[^}]*font-size:\s*0\.86rem;[^}]*font-weight:\s*500;/s
    );
    expect(styles).toMatch(
      /\.app-settings-inline-actions\s*\{[^}]*flex-wrap:\s*nowrap;/s
    );
    expect(styles).toMatch(/\.app-settings-bisque-link-form/);
    expect(styles).toMatch(
      /\.app-settings-bisque-credential-grid\s*\{[^}]*grid-template-columns:\s*repeat\(2,\s*minmax\(0,\s*1fr\)\);/s
    );
    expect(styles).toMatch(/\.app-settings-bisque-link-status/);
    expect(styles).not.toMatch(/var\(--fg\)/);
    expect(styles).toMatch(/@media \(max-width: 760px\)/);
  });

  it("keeps the Settings dialog inside the viewport while centering on the main pane", () => {
    const styles = readSource("src/styles.css");
    const ruleMatch = styles.match(
      /\.app-settings-dialog\[data-slot="dialog-content"\]\s*\{([^}]*)\}/s
    );

    expect(ruleMatch, "settings dialog rule is missing").not.toBeNull();
    const rule = ruleMatch?.[1] ?? "";
    expect(rule).toMatch(
      /--settings-dialog-main-offset:\s*calc\(var\(--sidebar-width\) \/ 2\);/
    );
    expect(rule).toMatch(
      /left:\s*min\(\s*calc\(50% \+ var\(--settings-dialog-main-offset\)\),\s*calc\(100vw - \(var\(--settings-dialog-width\) \/ 2\) - 1rem\)\s*\);/s
    );
    expect(rule).toMatch(
      /--settings-dialog-width:\s*min\(calc\(var\(--user-chat-width\) \+ 8rem\), calc\(100vw - 2rem\)\);/
    );
    expect(rule).toMatch(/width:\s*var\(--settings-dialog-width\);/);
    expect(rule).toMatch(/max-width:\s*var\(--settings-dialog-width\);/);
  });

  it("offers a sidebar CTA to link BisQue and opens settings on the BisQue tab", () => {
    const app = readSource("src/App.tsx");
    const settingsDialog = readSource("src/components/AppSettingsDialog.tsx");

    // The settings dialog can be opened directly to a chosen tab.
    expect(settingsDialog).toMatch(/initialTab\?: SettingsTab;/);
    expect(settingsDialog).toMatch(/initialTab = "general"/);
    expect(settingsDialog).toMatch(/<Tabs defaultValue=\{initialTab\}/);
    expect(app).toMatch(/const openSettings = useCallback\(/);
    expect(app).toMatch(/initialTab=\{settingsInitialTab\}/);
    expect(app).toMatch(/onOpenSettings=\{\(\) => openSettings\("general"\)\}/);

    // Unlinked users get a Link-BisQue CTA that jumps to the BisQue settings tab;
    // the resource counts only render once linked.
    expect(app).toMatch(/app-bisque-link-cta/);
    expect(app).toMatch(/onClick=\{\(\) => openSettings\("bisque"\)\}/);
    expect(app).toMatch(/<span>Link BisQue account<\/span>/);
    expect(app).toMatch(/\{bisqueCredentialsLinked \? \(/);

    // The BisQue settings tab is decluttered: keep the why/login, drop the
    // redundant production/shortcut/Ultra-panel sections.
    expect(settingsDialog).toMatch(/Why link BisQue\?/);
    expect(settingsDialog).not.toMatch(/BisQue production/);
    expect(settingsDialog).not.toMatch(/Ultra panels/);
    expect(settingsDialog).not.toMatch(/app-settings-link-grid/);
  });
});
