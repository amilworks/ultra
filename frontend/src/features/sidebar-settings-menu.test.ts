import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const readSource = (relativePath: string): string =>
  readFileSync(path.join(process.cwd(), relativePath), "utf8");

describe("sidebar settings menu", () => {
  it("composes a shadcn settings entry below chat history with a relevant settings dialog", () => {
    const app = readSource("src/App.tsx");
    const styles = readSource("src/styles.css");

    expect(app).toMatch(/function SidebarAccountSettingsButton/);
    expect(app).toMatch(/function AppSettingsDialog/);
    expect(app).toMatch(/<Avatar/);
    expect(app).toMatch(/<AvatarFallback>/);
    expect(app).toMatch(/<Dialog open={open} onOpenChange={onOpenChange}>/);
    expect(app).toMatch(/<DialogTitle>Settings<\/DialogTitle>/);
    expect(app).toMatch(/<TabsList/);
    expect(app).toMatch(/<SelectGroup>/);
    expect(app).toMatch(/<Separator/);
    expect(app).toMatch(/app-sidebar-user-menu/);
    expect(app).toMatch(/<Avatar className="app-sidebar-account-avatar">/);
    expect(app).toMatch(/className="app-sidebar-account-menu"/);
    expect(app).toMatch(/<span>Dark<\/span>[\s\S]*<span>Settings<\/span>/);
    expect(app).not.toMatch(/app-theme-menu-button/);
    expect(app).toMatch(/<span>New chat<\/span>/);
    expect(app).toMatch(/isMobileConversationSearchActive \? "Search results" : "Recents"/);
    expect(app).toMatch(/<Settings data-icon="inline-start" aria-hidden="true" \/>/);
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
    expect(styles).toMatch(
      /\.app-sidebar-content \.app-history-group \[data-slot="sidebar-group-label"\]\s*\{[^}]*font-size:\s*0\.875rem;[^}]*font-weight:\s*560;/s
    );
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
      /\.app-sidebar-account-name\s*\{[^}]*font-size:\s*0\.82rem;[^}]*font-weight:\s*300;/s
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
      /\.app-settings-sidebar-pane\s*\{[^}]*background:\s*color-mix\(in oklab, var\(--popover\) 96%, var\(--muted\) 4%\);/s
    );
    expect(styles).toMatch(
      /\.app-settings-nav-item\[data-slot="tabs-trigger"\]\s*\{[^}]*min-height:\s*2\.125rem;[^}]*font-size:\s*0\.8125rem;[^}]*font-weight:\s*400;/s
    );
    expect(styles).toMatch(
      /\.app-settings-row,\s*\.app-settings-account-summary\s*\{[^}]*min-height:\s*3\.25rem;/s
    );
    expect(styles).toMatch(
      /\.app-settings-header \[data-slot="dialog-title"\]\s*\{[^}]*font-size:\s*0\.98rem;[^}]*font-weight:\s*500;/s
    );
    expect(styles).toMatch(
      /\.app-settings-panel-heading h2\s*\{[^}]*font-size:\s*0\.98rem;[^}]*font-weight:\s*500;/s
    );
    expect(styles).toMatch(
      /\.app-settings-row-title,\s*\.app-settings-account-name\s*\{[^}]*font-size:\s*0\.86rem;[^}]*font-weight:\s*470;/s
    );
    expect(styles).not.toMatch(/var\(--fg\)/);
    expect(styles).toMatch(/@media \(max-width: 760px\)/);
  });
});
