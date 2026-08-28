import { useEffect, useState, type FormEvent } from "react";
import {
  Activity,
  Check,
  Database,
  ExternalLink,
  GitBranch,
  IdCard,
  Info,
  Link2,
  Loader2,
  LogOut,
  Settings,
  Shield,
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
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import type { BisqueNavLinks } from "@/features/auth/bisqueNavigation";
import { DEFAULT_BISQUE_BROWSER_URL } from "@/lib/config";
import type {
  CurrentUserProfile,
  CurrentUserResponse,
  TokenUsageResponse,
} from "@/types";
import { UserTokenUsagePanel } from "./UserTokenUsagePanel";
import { SystemMessage } from "./prompt-kit/system-message";

type ThemePreference = "system" | "light" | "dark";
type AuthMode = "bisque" | "guest" | "workos";
export type SettingsTab =
  | "general"
  | "profile"
  | "usage"
  | "account"
  | "bisque"
  | "about";

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
  initialTab?: SettingsTab;
  authUser: string | null;
  authMode: AuthMode | null;
  authIsAdmin: boolean;
  bisqueCredentialsLinked: boolean;
  themePreference: ThemePreference;
  resolvedTheme: "light" | "dark";
  bisqueNavLinks: BisqueNavLinks | null;
  onThemePreferenceChange: (value: ThemePreference) => void;
  onOpenAdmin: () => void;
  onLogout: () => Promise<void>;
  onUnlinkBisqueAccount: () => Promise<void>;
  onLinkBisqueAccount: (payload: LinkBisqueAccountPayload) => Promise<{
    imageCount: number;
  }>;
  loadProfile?: () => Promise<CurrentUserResponse>;
  saveProfile?: (profile: CurrentUserProfile) => Promise<CurrentUserResponse>;
  loadTokenUsage?: (days: number) => Promise<TokenUsageResponse>;
  formatError: (error: unknown) => string;
};

type ProfileFormState = {
  displayName: string;
  title: string;
  institution: string;
  researchInterests: string;
  bio: string;
};

const EMPTY_PROFILE_FORM: ProfileFormState = {
  displayName: "",
  title: "",
  institution: "",
  researchInterests: "",
  bio: "",
};

const profileFormFromResponse = (response: CurrentUserResponse): ProfileFormState => ({
  displayName: response.profile.display_name ?? response.user.display_name ?? "",
  title: response.profile.title ?? "",
  institution: response.profile.institution ?? "",
  researchInterests: response.profile.research_interests ?? "",
  bio: response.profile.bio ?? "",
});

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

export function AppSettingsDialog({
  open,
  onOpenChange,
  initialTab = "general",
  authUser,
  authMode,
  authIsAdmin,
  bisqueCredentialsLinked,
  themePreference,
  resolvedTheme,
  bisqueNavLinks,
  onThemePreferenceChange,
  onOpenAdmin,
  onLogout,
  onUnlinkBisqueAccount,
  onLinkBisqueAccount,
  loadProfile,
  saveProfile,
  loadTokenUsage,
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

  const profileEnabled = Boolean(loadProfile && saveProfile);
  const usageEnabled = Boolean(loadTokenUsage);

  const [profileForm, setProfileForm] = useState<ProfileFormState>(EMPTY_PROFILE_FORM);
  const [profileLoading, setProfileLoading] = useState(false);
  const [profileSaving, setProfileSaving] = useState(false);
  const [profileError, setProfileError] = useState<string | null>(null);
  const [profileSaved, setProfileSaved] = useState(false);

  const [tokenUsage, setTokenUsage] = useState<TokenUsageResponse | null>(null);
  const [usageLoading, setUsageLoading] = useState(false);
  const [usageError, setUsageError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) {
      return;
    }
    let cancelled = false;
    if (loadProfile) {
      void (async () => {
        setProfileLoading(true);
        setProfileError(null);
        try {
          const response = await loadProfile();
          if (!cancelled) {
            setProfileForm(profileFormFromResponse(response));
          }
        } catch (error) {
          if (!cancelled) {
            setProfileError(formatError(error));
          }
        } finally {
          if (!cancelled) {
            setProfileLoading(false);
          }
        }
      })();
    }
    if (loadTokenUsage) {
      void (async () => {
        setUsageLoading(true);
        setUsageError(null);
        try {
          const response = await loadTokenUsage(365);
          if (!cancelled) {
            setTokenUsage(response);
          }
        } catch (error) {
          if (!cancelled) {
            setUsageError(formatError(error));
          }
        } finally {
          if (!cancelled) {
            setUsageLoading(false);
          }
        }
      })();
    }
    return () => {
      cancelled = true;
    };
  }, [open, loadProfile, loadTokenUsage, formatError]);

  const updateProfileField = (field: keyof ProfileFormState, value: string): void => {
    setProfileSaved(false);
    setProfileForm((current) => ({ ...current, [field]: value }));
  };

  const submitProfile = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    if (!saveProfile || profileSaving) {
      return;
    }
    setProfileSaving(true);
    setProfileError(null);
    try {
      const response = await saveProfile({
        display_name: profileForm.displayName.trim(),
        title: profileForm.title.trim(),
        institution: profileForm.institution.trim(),
        research_interests: profileForm.researchInterests.trim(),
        bio: profileForm.bio.trim(),
      });
      setProfileForm(profileFormFromResponse(response));
      setProfileSaved(true);
      toast.success("Profile saved", {
        description: "Your research agent will use this to personalize runs.",
      });
    } catch (error) {
      const message = formatError(error);
      setProfileError(message);
      toast.error("Could not save profile", { description: message });
    } finally {
      setProfileSaving(false);
    }
  };

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
    if (bisqueUnlinking) {
      return;
    }
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
        <Tabs defaultValue={initialTab} className="app-settings-shell">
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
              {profileEnabled ? (
                <TabsTrigger value="profile" className="app-settings-nav-item">
                  <IdCard data-icon="inline-start" aria-hidden="true" />
                  Profile
                </TabsTrigger>
              ) : null}
              {usageEnabled ? (
                <TabsTrigger value="usage" className="app-settings-nav-item">
                  <Activity data-icon="inline-start" aria-hidden="true" />
                  Usage
                </TabsTrigger>
              ) : null}
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

            {profileEnabled ? (
              <TabsContent value="profile" className="app-settings-tab-content">
                <div className="app-settings-panel-heading">
                  <h2>Profile</h2>
                  <p>
                    Tell Ultra about your work. Your research agent reads this to tailor
                    depth, terminology, and examples to you.
                  </p>
                </div>
                <Separator />
                {profileLoading ? (
                  <div className="app-settings-profile-loading">
                    <Skeleton className="h-9 w-full" />
                    <Skeleton className="h-9 w-full" />
                    <Skeleton className="h-24 w-full" />
                  </div>
                ) : (
                  <form className="app-settings-profile-form" onSubmit={submitProfile}>
                    <div className="app-settings-field-grid">
                      <div className="app-settings-field">
                        <Label htmlFor="settings-profile-name">Display name</Label>
                        <Input
                          id="settings-profile-name"
                          value={profileForm.displayName}
                          onChange={(event) =>
                            updateProfileField("displayName", event.target.value)
                          }
                          placeholder="How Ultra should address you"
                          disabled={profileSaving}
                        />
                      </div>
                      <div className="app-settings-field">
                        <Label htmlFor="settings-profile-title">Title or role</Label>
                        <Input
                          id="settings-profile-title"
                          value={profileForm.title}
                          onChange={(event) =>
                            updateProfileField("title", event.target.value)
                          }
                          placeholder="e.g. PhD candidate, PI"
                          disabled={profileSaving}
                        />
                      </div>
                    </div>
                    <div className="app-settings-field">
                      <Label htmlFor="settings-profile-institution">Institution</Label>
                      <Input
                        id="settings-profile-institution"
                        value={profileForm.institution}
                        onChange={(event) =>
                          updateProfileField("institution", event.target.value)
                        }
                        placeholder="e.g. UCSB Vision Research Lab"
                        disabled={profileSaving}
                      />
                    </div>
                    <div className="app-settings-field">
                      <Label htmlFor="settings-profile-interests">
                        Research interests
                      </Label>
                      <Textarea
                        id="settings-profile-interests"
                        value={profileForm.researchInterests}
                        onChange={(event) =>
                          updateProfileField("researchInterests", event.target.value)
                        }
                        placeholder="Areas, methods, organisms, or imaging modalities you focus on"
                        rows={2}
                        disabled={profileSaving}
                      />
                    </div>
                    <div className="app-settings-field">
                      <Label htmlFor="settings-profile-bio">About you</Label>
                      <Textarea
                        id="settings-profile-bio"
                        value={profileForm.bio}
                        onChange={(event) =>
                          updateProfileField("bio", event.target.value)
                        }
                        placeholder="A few sentences about your background and what you are working toward."
                        rows={4}
                        disabled={profileSaving}
                      />
                      <p className="app-settings-field-hint">
                        Saved to your private agent memory and used to personalize runs.
                        Never shared with other users.
                      </p>
                    </div>
                    {profileError ? (
                      <SystemMessage variant="error" fill>
                        {profileError}
                      </SystemMessage>
                    ) : null}
                    <div className="app-settings-action-row">
                      <Button
                        type="submit"
                        aria-disabled={profileSaving || undefined}
                        className="aria-disabled:pointer-events-none aria-disabled:opacity-50"
                      >
                        {profileSaving ? (
                          <Loader2
                            data-icon="inline-start"
                            className="animate-spin"
                            aria-hidden="true"
                          />
                        ) : (
                          <Check data-icon="inline-start" aria-hidden="true" />
                        )}
                        {profileSaving
                          ? "Saving..."
                          : profileSaved
                            ? "Saved"
                            : "Save profile"}
                      </Button>
                    </div>
                  </form>
                )}
              </TabsContent>
            ) : null}

            {usageEnabled ? (
              <TabsContent value="usage" className="app-settings-tab-content">
                <UserTokenUsagePanel
                  tokenUsage={tokenUsage}
                  loading={usageLoading}
                  error={usageError}
                  density="compact"
                />
              </TabsContent>
            ) : null}

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
                <p>Connect your BisQue account so Ultra can work directly with your data.</p>
              </div>
              <Separator />
              <div className="app-settings-row">
                <div className="app-settings-row-copy">
                  <div className="app-settings-row-title">Why link BisQue?</div>
                  <p>
                    BisQue is the UCSB image database where your microscopy images,
                    datasets, and tables live. Link your account once and Ultra can
                    search, open, and analyze your BisQue data — and save results
                    back — during a run, with no manual uploads.
                  </p>
                </div>
              </div>
              <form className="app-settings-bisque-link-form" onSubmit={submitBisqueLink}>
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
                      aria-disabled={bisqueUnlinking || undefined}
                      className="aria-disabled:pointer-events-none aria-disabled:opacity-50"
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
                      <GitBranch data-icon="inline-start" aria-hidden="true" />
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
                      <GitBranch data-icon="inline-start" aria-hidden="true" />
                      Creator
                    </a>
                  </Button>
                  <Button asChild variant="outline" size="sm">
                    <a
                      href="https://github.com/amilworks/ultra/issues"
                      target="_blank"
                      rel="noreferrer"
                    >
                      <GitBranch data-icon="inline-start" aria-hidden="true" />
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
