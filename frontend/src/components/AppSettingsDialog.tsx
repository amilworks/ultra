import { useState, type FormEvent } from "react";
import {
  Check,
  Database,
  ExternalLink,
  FolderOpen,
  Images,
  Info,
  Link2,
  LogOut,
  Settings,
  Shield,
  Table2,
  Unlink,
  UserRound,
  X,
} from "lucide-react";
import { toast } from "sonner";

import {
  Alert,
  AlertAction,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { BisqueNavLinks } from "@/features/auth/bisqueNavigation";
import { DEFAULT_BISQUE_BROWSER_URL } from "@/lib/config";
import { SystemMessage } from "./prompt-kit/system-message";

type ThemePreference = "system" | "light" | "dark";
type AuthMode = "bisque" | "guest" | "workos";

type ThemeOption = {
  value: ThemePreference;
  label: string;
};

type LinkBisqueAccountPayload = {
  username: string;
  password: string;
};

export type AppSettingsDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  authUser: string | null;
  authMode: AuthMode | null;
  authIsAdmin: boolean;
  bisqueCredentialsLinked: boolean;
  themePreference: ThemePreference;
  resolvedTheme: "light" | "dark";
  bisqueNavLinks: BisqueNavLinks | null;
  onThemePreferenceChange: (value: ThemePreference) => void;
  onOpenResources: () => void;
  onOpenTraining: () => void;
  onOpenAdmin: () => void;
  onLogout: () => Promise<void>;
  onUnlinkBisqueAccount: () => Promise<void>;
  onLinkBisqueAccount: (payload: LinkBisqueAccountPayload) => Promise<{
    imageCount: number;
  }>;
  formatError: (error: unknown) => string;
};

const THEME_OPTIONS: ThemeOption[] = [
  {
    value: "system",
    label: "System",
  },
  {
    value: "light",
    label: "Light",
  },
  {
    value: "dark",
    label: "Dark",
  },
];

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
    return "WorkOS user";
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
    return authIsAdmin ? "WorkOS admin" : "WorkOS account";
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

export function AppSettingsDialog({
  open,
  onOpenChange,
  authUser,
  authMode,
  authIsAdmin,
  bisqueCredentialsLinked,
  themePreference,
  resolvedTheme,
  bisqueNavLinks,
  onThemePreferenceChange,
  onOpenResources,
  onOpenTraining,
  onOpenAdmin,
  onLogout,
  onUnlinkBisqueAccount,
  onLinkBisqueAccount,
  formatError,
}: AppSettingsDialogProps) {
  const accountName = getAccountDisplayName(authUser, authMode);
  const accountSubtitle = getAccountSubtitle(authMode, authIsAdmin);
  const initials = getAccountInitials(accountName);
  const resolvedThemeLabel = resolvedTheme === "dark" ? "Dark" : "Light";
  const defaultBisqueUsername = authMode === "bisque" ? authUser ?? "" : "";
  const [bisqueUsernameState, setBisqueUsernameState] = useState(() => ({
    key: defaultBisqueUsername,
    value: defaultBisqueUsername,
  }));
  const bisqueUsername =
    bisqueUsernameState.key === defaultBisqueUsername
      ? bisqueUsernameState.value
      : defaultBisqueUsername;
  const setBisqueUsername = (value: string): void => {
    setBisqueUsernameState({ key: defaultBisqueUsername, value });
  };
  const [bisquePassword, setBisquePassword] = useState("");
  const [bisqueLinking, setBisqueLinking] = useState(false);
  const [bisqueUnlinking, setBisqueUnlinking] = useState(false);
  const [bisqueLinkError, setBisqueLinkError] = useState<string | null>(null);
  const [bisqueImageCount, setBisqueImageCount] = useState<number | null>(null);
  const bisqueLinked = Boolean(bisqueCredentialsLinked && authUser);
  const bisqueRootHref = bisqueNavLinks?.home ?? DEFAULT_BISQUE_BROWSER_URL;

  const runAndClose = (action: () => void): void => {
    onOpenChange(false);
    action();
  };

  const logoutAndClose = (): void => {
    onOpenChange(false);
    void onLogout();
  };

  const submitBisqueLink = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    const username = bisqueUsername.trim();
    if (!username || !bisquePassword) {
      setBisqueLinkError("Enter your BisQue username and password.");
      return;
    }
    setBisqueLinking(true);
    setBisqueLinkError(null);
    try {
      const result = await onLinkBisqueAccount({
        username,
        password: bisquePassword,
      });
      setBisquePassword("");
      setBisqueImageCount(result.imageCount);
    } catch (error) {
      const message = formatError(error);
      setBisqueLinkError(message);
      toast.error("Could not link BisQue account", {
        description: message,
      });
      setBisquePassword("");
    } finally {
      setBisqueLinking(false);
    }
  };

  const unlinkBisqueAccount = async (): Promise<void> => {
    setBisqueUnlinking(true);
    setBisqueLinkError(null);
    setBisquePassword("");
    setBisqueImageCount(null);
    try {
      await onUnlinkBisqueAccount();
    } catch (error) {
      const message = formatError(error);
      setBisqueLinkError(message);
      toast.error("Could not unlink BisQue account", {
        description: message,
      });
    } finally {
      setBisqueUnlinking(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="app-settings-dialog" showCloseButton={false}>
        <Tabs defaultValue="general" className="app-settings-shell">
          <aside className="app-settings-sidebar-pane">
            <DialogClose asChild>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="app-settings-close-button"
                aria-label="Close settings"
              >
                <X data-icon="inline-start" aria-hidden="true" />
              </Button>
            </DialogClose>
            <DialogHeader className="app-settings-header">
              <DialogTitle>Settings</DialogTitle>
              <DialogDescription>
                Tune the interface and account shortcuts for this workspace.
              </DialogDescription>
            </DialogHeader>
            <TabsList className="app-settings-nav-list" aria-label="Settings sections">
              <TabsTrigger value="general" className="app-settings-nav-item">
                <Settings data-icon="inline-start" aria-hidden="true" />
                General
              </TabsTrigger>
              <TabsTrigger value="account" className="app-settings-nav-item">
                <UserRound data-icon="inline-start" aria-hidden="true" />
                Account
              </TabsTrigger>
              <TabsTrigger value="bisque" className="app-settings-nav-item">
                <Database data-icon="inline-start" aria-hidden="true" />
                BisQue
              </TabsTrigger>
              <TabsTrigger value="about" className="app-settings-nav-item">
                <Info data-icon="inline-start" aria-hidden="true" />
                About
              </TabsTrigger>
            </TabsList>
          </aside>

          <section className="app-settings-content-pane">
            <TabsContent value="general" className="app-settings-tab-content">
              <div className="app-settings-panel-heading">
                <h2>General</h2>
                <p>Keep the day-to-day interface fast, readable, and predictable.</p>
              </div>
              <Separator />
              <div className="app-settings-row">
                <div className="app-settings-row-copy">
                  <div className="app-settings-row-title">Appearance</div>
                  <p>Match your system preference or pin BisQue Ultra to one mode.</p>
                </div>
                <Select
                  value={themePreference}
                  onValueChange={(value) =>
                    onThemePreferenceChange(value as ThemePreference)
                  }
                >
                  <SelectTrigger
                    size="sm"
                    className="app-settings-select-trigger"
                    aria-label="Appearance"
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectGroup>
                      {THEME_OPTIONS.map((option) => (
                        <SelectItem key={option.value} value={option.value}>
                          {option.label}
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  </SelectContent>
                </Select>
              </div>
              <Separator />
              <div className="app-settings-row">
                <div className="app-settings-row-copy">
                  <div className="app-settings-row-title">Resolved theme</div>
                  <p>
                    System mode currently resolves to the active browser color scheme.
                  </p>
                </div>
                <Badge variant="secondary">{resolvedThemeLabel}</Badge>
              </div>
            </TabsContent>

            <TabsContent value="account" className="app-settings-tab-content">
              <div className="app-settings-panel-heading">
                <h2>Account</h2>
                <p>Review the active session and sign out when you are finished.</p>
              </div>
              <Separator />
              <div className="app-settings-account-summary">
                <Avatar size="lg" className="app-settings-account-avatar">
                  <AvatarFallback>{initials}</AvatarFallback>
                </Avatar>
                <div className="app-settings-account-copy">
                  <div className="app-settings-account-name">{accountName}</div>
                  <div className="app-settings-account-subtitle">{accountSubtitle}</div>
                </div>
                <Badge variant={authMode === "guest" ? "secondary" : "default"}>
                  {authMode === "guest" ? "Guest" : "Signed in"}
                </Badge>
              </div>
              <Separator />
              <div className="app-settings-row">
                <div className="app-settings-row-copy">
                  <div className="app-settings-row-title">Admin console</div>
                  <p>
                    {authIsAdmin
                      ? "Open the operational view for users, runs, and issues."
                      : "This account does not have admin console access."}
                  </p>
                </div>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  disabled={!authIsAdmin}
                  onClick={() => runAndClose(onOpenAdmin)}
                >
                  <Shield data-icon="inline-start" aria-hidden="true" />
                  Open
                </Button>
              </div>
              <Separator />
              <div className="app-settings-action-row">
                <Button type="button" variant="outline" onClick={logoutAndClose}>
                  <LogOut data-icon="inline-start" aria-hidden="true" />
                  Sign out
                </Button>
              </div>
            </TabsContent>

            <TabsContent value="bisque" className="app-settings-tab-content">
              <div className="app-settings-panel-heading">
                <h2>BisQue</h2>
                <p>Link your BisQue account or jump into production resources.</p>
              </div>
              <Separator />
              <form className="app-settings-bisque-link-form" onSubmit={submitBisqueLink}>
                <div className="app-settings-row">
                  <div className="app-settings-row-copy">
                    <div className="app-settings-row-title">Linked account</div>
                    <p>
                      Store a local session so Ultra can query, download, and upload
                      BisQue data during autonomous runs.
                    </p>
                  </div>
                </div>
                {bisqueLinked && authUser ? (
                  <Alert className="app-settings-bisque-linked-alert">
                    <Check data-icon="inline-start" aria-hidden="true" />
                    <AlertTitle>BisQue account linked</AlertTitle>
                    <AlertDescription>
                      <span>
                        Signed in as <strong>{authUser}</strong>
                        {bisqueImageCount != null
                          ? `. Found ${bisqueImageCount.toLocaleString()} image${
                              bisqueImageCount === 1 ? "" : "s"
                            }.`
                          : "."}
                      </span>
                      <AlertAction>
                        <Button asChild variant="outline" size="sm">
                          <a href={bisqueRootHref} target="_blank" rel="noreferrer">
                            Open BisQue
                            <ExternalLink data-icon="inline-end" aria-hidden="true" />
                          </a>
                        </Button>
                      </AlertAction>
                    </AlertDescription>
                  </Alert>
                ) : (
                  <div className="app-settings-bisque-credential-grid">
                    <div className="app-settings-field">
                      <Label htmlFor="settings-bisque-username">Username</Label>
                      <Input
                        id="settings-bisque-username"
                        value={bisqueUsername}
                        onChange={(event) => setBisqueUsername(event.target.value)}
                        autoComplete="username"
                        disabled={bisqueLinking}
                      />
                    </div>
                    <div className="app-settings-field">
                      <Label htmlFor="settings-bisque-password">Password</Label>
                      <Input
                        id="settings-bisque-password"
                        type="password"
                        value={bisquePassword}
                        onChange={(event) => setBisquePassword(event.target.value)}
                        autoComplete="current-password"
                        disabled={bisqueLinking}
                      />
                    </div>
                  </div>
                )}
                {bisqueLinkError ? (
                  <SystemMessage variant="error" fill>
                    {bisqueLinkError}
                  </SystemMessage>
                ) : null}
                {bisqueImageCount != null && !bisqueLinkError && !bisqueLinked ? (
                  <div className="app-settings-bisque-link-status">
                    <Check data-icon="inline-start" aria-hidden="true" />
                    BisQue account linked. Found {bisqueImageCount.toLocaleString()} image
                    {bisqueImageCount === 1 ? "" : "s"}.
                  </div>
                ) : null}
                <div className="app-settings-action-row">
                  {bisqueLinked ? (
                    <Button
                      type="button"
                      variant="outline"
                      disabled={bisqueUnlinking}
                      onClick={() => {
                        void unlinkBisqueAccount();
                      }}
                    >
                      <Unlink data-icon="inline-start" aria-hidden="true" />
                      {bisqueUnlinking ? "Unlinking..." : "Unlink account"}
                    </Button>
                  ) : (
                    <Button
                      type="submit"
                      variant="outline"
                      disabled={bisqueLinking || !bisqueUsername.trim() || !bisquePassword}
                    >
                      <Link2 data-icon="inline-start" aria-hidden="true" />
                      {bisqueLinking ? "Testing account..." : "Link account"}
                    </Button>
                  )}
                </div>
              </form>
              <Separator />
              <div className="app-settings-row">
                <div className="app-settings-row-copy">
                  <div className="app-settings-row-title">BisQue production</div>
                  <p>Open the configured BisQue instance in a new browser tab.</p>
                </div>
                {bisqueNavLinks ? (
                  <Button asChild variant="outline" size="sm">
                    <a href={bisqueNavLinks.home} target="_blank" rel="noreferrer">
                      <ExternalLink data-icon="inline-start" aria-hidden="true" />
                      Open
                    </a>
                  </Button>
                ) : (
                  <Button type="button" variant="outline" size="sm" disabled>
                    Open
                  </Button>
                )}
              </div>
              <Separator />
              <div className="app-settings-link-grid">
                {bisqueNavLinks ? (
                  <>
                    <Button asChild variant="ghost" className="app-settings-link-button">
                      <a href={bisqueNavLinks.images} target="_blank" rel="noreferrer">
                        <Images data-icon="inline-start" aria-hidden="true" />
                        Images
                        <ExternalLink data-icon="inline-end" aria-hidden="true" />
                      </a>
                    </Button>
                    <Button asChild variant="ghost" className="app-settings-link-button">
                      <a href={bisqueNavLinks.datasets} target="_blank" rel="noreferrer">
                        <Database data-icon="inline-start" aria-hidden="true" />
                        Datasets
                        <ExternalLink data-icon="inline-end" aria-hidden="true" />
                      </a>
                    </Button>
                    <Button asChild variant="ghost" className="app-settings-link-button">
                      <a href={bisqueNavLinks.tables} target="_blank" rel="noreferrer">
                        <Table2 data-icon="inline-start" aria-hidden="true" />
                        Tables
                        <ExternalLink data-icon="inline-end" aria-hidden="true" />
                      </a>
                    </Button>
                  </>
                ) : (
                  <div className="app-settings-unavailable">
                    BisQue shortcuts are unavailable until the production root is configured.
                  </div>
                )}
              </div>
              <Separator />
              <div className="app-settings-row">
                <div className="app-settings-row-copy">
                  <div className="app-settings-row-title">Ultra panels</div>
                  <p>Open resources or model training inside this app.</p>
                </div>
                <div className="app-settings-inline-actions">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => runAndClose(onOpenResources)}
                  >
                    <FolderOpen data-icon="inline-start" aria-hidden="true" />
                    Resources
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => runAndClose(onOpenTraining)}
                  >
                    <Database data-icon="inline-start" aria-hidden="true" />
                    Training
                  </Button>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="about" className="app-settings-tab-content">
              <div className="app-settings-panel-heading">
                <h2>About</h2>
                <p>Project background, links, and workspace context.</p>
              </div>
              <Alert className="app-settings-about-alert">
                <Info data-icon="inline-start" aria-hidden="true" />
                <AlertTitle>Created within the UCSB Vision Research Lab</AlertTitle>
                <AlertDescription>
                  BisQue Ultra was created by Amil Khan, a PhD student in the
                  UCSB Vision Research Lab. The lab is led by Professor B.S.
                  Manjunath in the Department of Electrical and Computer
                  Engineering, with research spanning computer vision, image
                  processing, and machine learning applications.
                </AlertDescription>
              </Alert>
              <Separator />
              <div className="app-settings-row">
                <div className="app-settings-row-copy">
                  <div className="app-settings-row-title">Project links</div>
                  <p>Open the source repository or public project website.</p>
                </div>
                <div className="app-settings-inline-actions">
                  <Button asChild variant="outline" size="sm">
                    <a
                      href="https://github.com/amilworks/ultra"
                      target="_blank"
                      rel="noreferrer"
                    >
                      <ExternalLink data-icon="inline-start" aria-hidden="true" />
                      GitHub
                    </a>
                  </Button>
                  <Button asChild variant="outline" size="sm">
                    <a
                      href="https://amilworks.github.io/ultra_website/"
                      target="_blank"
                      rel="noreferrer"
                    >
                      <ExternalLink data-icon="inline-start" aria-hidden="true" />
                      Website
                    </a>
                  </Button>
                </div>
              </div>
              <Separator />
              <div className="app-settings-row">
                <div className="app-settings-row-copy">
                  <div className="app-settings-row-title">Platform</div>
                  <p>
                    A scientific imaging workbench for reproducible, tool-guided
                    analysis with BisQue-backed data management, a Go control plane,
                    Deep Agents execution, and long-running workflow visibility.
                  </p>
                </div>
              </div>
              <Separator />
              <div className="app-settings-row">
                <div className="app-settings-row-copy">
                  <div className="app-settings-row-title">Contact us</div>
                  <p>Reach out to the creator through public project channels.</p>
                </div>
                <div className="app-settings-inline-actions">
                  <Button asChild variant="outline" size="sm">
                    <a href="https://github.com/amilworks" target="_blank" rel="noreferrer">
                      <ExternalLink data-icon="inline-start" aria-hidden="true" />
                      Creator
                    </a>
                  </Button>
                  <Button asChild variant="outline" size="sm">
                    <a
                      href="https://github.com/amilworks/ultra/issues"
                      target="_blank"
                      rel="noreferrer"
                    >
                      <ExternalLink data-icon="inline-start" aria-hidden="true" />
                      Issues
                    </a>
                  </Button>
                </div>
              </div>
            </TabsContent>
          </section>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}
