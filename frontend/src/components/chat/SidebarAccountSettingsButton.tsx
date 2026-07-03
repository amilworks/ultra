import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  mobileSidebarCloseProps,
  mobileSidebarKeepOpenProps,
} from "@/components/ui/sidebar";
import { Laptop, LogOut, Moon, Settings, Sun } from "lucide-react";

type ThemePreference = "system" | "light" | "dark";
type AuthMode = "bisque" | "guest" | "workos";

const getAccountDisplayName = (
  authUser: string | null,
  authMode: AuthMode | null
): string => {
  const trimmedUser = String(authUser ?? "").trim();
  if (trimmedUser) {
    return trimmedUser;
  }
  if (authMode === "guest") {
    return "Guest";
  }
  if (authMode === "workos") {
    return "Researcher";
  }
  return "BisQue user";
};

const getAccountSubtitle = (
  authMode: AuthMode | null,
  authIsAdmin: boolean
): string => {
  if (authMode === "guest") {
    return "Guest access";
  }
  if (authMode === "workos") {
    return authIsAdmin ? "Administrator" : "Researcher";
  }
  if (authIsAdmin) {
    return "Admin account";
  }
  return "BisQue account";
};

const getAccountInitials = (displayName: string): string => {
  const normalized = displayName.replace(/@.*$/, "").trim();
  const parts = normalized.split(/[\s._-]+/).filter(Boolean);
  const source =
    parts.length >= 2
      ? `${parts[0]?.[0] ?? ""}${parts[1]?.[0] ?? ""}`
      : normalized.slice(0, 2);
  return source.trim().toUpperCase() || "BU";
};

type SidebarAccountSettingsButtonProps = {
  authUser: string | null;
  authMode: AuthMode | null;
  authIsAdmin: boolean;
  themePreference: ThemePreference;
  onThemePreferenceChange: (value: ThemePreference) => void;
  onOpenSettings: () => void;
  onLogout: () => Promise<void>;
};

export function SidebarAccountSettingsButton({
  authUser,
  authMode,
  authIsAdmin,
  themePreference,
  onThemePreferenceChange,
  onOpenSettings,
  onLogout,
}: SidebarAccountSettingsButtonProps) {
  const accountName = getAccountDisplayName(authUser, authMode);
  const accountSubtitle = getAccountSubtitle(authMode, authIsAdmin);
  const initials = getAccountInitials(accountName);

  return (
    <div className="app-sidebar-user-menu">
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            type="button"
            variant="ghost"
            className="app-sidebar-account-button"
            aria-label="Open account menu"
            {...mobileSidebarKeepOpenProps}
          >
            <Avatar className="app-sidebar-account-avatar">
              <AvatarFallback>{initials}</AvatarFallback>
            </Avatar>
            <span className="app-sidebar-account-copy">
              <span className="app-sidebar-account-name">{accountName}</span>
              <span className="app-sidebar-account-subtitle">{accountSubtitle}</span>
            </span>
            <Settings data-icon="inline-end" aria-hidden="true" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          side="top"
          align="start"
          sideOffset={8}
          className="app-sidebar-account-menu"
        >
          <DropdownMenuItem onClick={() => onThemePreferenceChange("system")}>
            <Laptop data-icon="inline-start" aria-hidden="true" />
            <span>System</span>
            {themePreference === "system" ? (
              <span className="app-sidebar-account-menu-current" aria-hidden="true">
                •
              </span>
            ) : null}
          </DropdownMenuItem>
          <DropdownMenuItem onClick={() => onThemePreferenceChange("light")}>
            <Sun data-icon="inline-start" aria-hidden="true" />
            <span>Light</span>
            {themePreference === "light" ? (
              <span className="app-sidebar-account-menu-current" aria-hidden="true">
                •
              </span>
            ) : null}
          </DropdownMenuItem>
          <DropdownMenuItem onClick={() => onThemePreferenceChange("dark")}>
            <Moon data-icon="inline-start" aria-hidden="true" />
            <span>Dark</span>
            {themePreference === "dark" ? (
              <span className="app-sidebar-account-menu-current" aria-hidden="true">
                •
              </span>
            ) : null}
          </DropdownMenuItem>
          <DropdownMenuItem onClick={onOpenSettings} {...mobileSidebarCloseProps}>
            <Settings data-icon="inline-start" aria-hidden="true" />
            <span>Settings</span>
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem onClick={() => void onLogout()}>
            <LogOut data-icon="inline-start" aria-hidden="true" />
            <span>
              {authUser
                ? authMode === "guest"
                  ? `Sign out (${authUser}, guest)`
                  : `Sign out (${authUser})`
                : "Sign out"}
            </span>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}
